#include "ChromaspaceViewerCameraInteraction.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace {

using namespace ChromaspaceViewer;

ViewerWorkspaceState makeWorkspace() {
  ViewerWorkspaceState workspace{};
  workspace.focusedWindowId = 42;
  workspace.nextWindowId = 100;
  workspace.layoutPresetSelection = "Camera Test";
  PlotWindowDomainState cube{};
  cube.windowId = 42;
  cube.rect = {0.05f, 0.05f, 0.42f, 0.80f};
  cube.selected = true;
  cube.viewState.plotModel = kPlotModelCube;
  PlotWindowDomainState scope{};
  scope.windowId = 7;
  scope.rect = {0.52f, 0.05f, 0.42f, 0.80f};
  scope.viewState.plotModel = kPlotModelWaveform;
  scope.camera.orthographic = true;
  scope.camera.distance = 6.35f;
  workspace.windows = {cube, scope};
  assert(validateViewerWorkspaceState(workspace));
  return workspace;
}

ViewerUiScene makeScene(const ViewerWorkspaceState& workspace) {
  ViewerFramePlanRequest planRequest{};
  planRequest.windowWidth = 1000;
  planRequest.windowHeight = 600;
  planRequest.framebufferWidth = 2000;
  planRequest.framebufferHeight = 1200;
  // Deliberately reverse workspace order to prove the join is by ID.
  planRequest.windows = {
      {7, workspace.windows[1].rect, kPlotModelWaveform, 1u, true},
      {42, workspace.windows[0].rect, kPlotModelCube, 1u, true},
  };
  const ViewerFramePlan plan = buildViewerFramePlan(planRequest);
  ViewerUiSceneInput input{};
  input.focusedWindowId = 42;
  for (const auto& planned : plan.windows) {
    ViewerUiPlotWindowInput window{};
    window.windowId = planned.windowId;
    window.title = "Plot";
    window.titleMetrics.fontAvailable = false;
    input.windows.push_back(window);
  }
  const ViewerUiScene scene = buildViewerUiScene(plan, input);
  assert(scene.ready());
  return scene;
}

ScreenRect content(const ViewerUiScene& scene, int id) {
  for (const auto& window : scene.windows) {
    if (window.windowId == id) return window.contentRect;
  }
  assert(false);
  return {};
}

ViewerSessionReduceResult accepted(uint64_t sequence,
                                   ViewerSessionInputKind kind) {
  ViewerSessionReduceResult result{};
  result.status = ViewerSessionReduceStatus::Applied;
  result.inputAccepted = true;
  result.acceptedInput.sequence = sequence;
  result.acceptedInput.kind = kind;
  return result;
}

ViewerCameraInteractionResult reduce(
    ViewerCameraInteractionState* state,
    ViewerSessionReduceResult* input,
    ViewerSessionState* session,
    const ViewerUiScene& scene,
    const ViewerWorkspaceState& workspace,
    bool authorize = false) {
  return reduceViewerCameraInteraction(
      state, {input, session, &scene, &workspace, authorize});
}

void commit(ViewerWorkspaceState* workspace,
            const ViewerCameraInteractionResult& result) {
  if (!result.cameraChanged) return;
  const auto committed = updateViewerWorkspaceCamera(
      workspace, result.windowId, result.camera);
  assert(committed.accepted() && committed.changed);
}

void testOrbitCaptureOutsideAndRelease() {
  ViewerWorkspaceState workspace = makeWorkspace();
  const ViewerUiScene scene = makeScene(workspace);
  const ScreenRect rect = content(scene, 42);
  ViewerSessionState session{};
  ViewerCameraInteractionState state{};
  auto press = accepted(1u, ViewerSessionInputKind::PointerButton);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  press.acceptedInput.logicalX = (rect.x0 + rect.x1) * 0.5f;
  press.acceptedInput.logicalY = (rect.y0 + rect.y1) * 0.5f;
  auto began = reduce(&state, &press, &session, scene, workspace, true);
  assert(began.accepted() && began.stateChanged && !began.cameraChanged);
  assert(state.pointerCaptureActive && state.pointerWindowId == 42 &&
         state.pointerMode == ViewerCameraInteractionMode::Orbit);

  auto move = accepted(2u, ViewerSessionInputKind::PointerMoved);
  move.acceptedInput.logicalX = rect.x1 + 150.0;
  move.acceptedInput.logicalY = rect.y0 - 60.0;
  auto rotated = reduce(&state, &move, &session, scene, workspace);
  assert(rotated.accepted() && rotated.cameraChanged &&
         rotated.windowId == 42);
  const double norm = rotated.camera.qx * rotated.camera.qx +
                      rotated.camera.qy * rotated.camera.qy +
                      rotated.camera.qz * rotated.camera.qz +
                      rotated.camera.qw * rotated.camera.qw;
  assert(std::abs(norm - 1.0) < 1.0e-5);
  commit(&workspace, rotated);

  auto release = accepted(3u, ViewerSessionInputKind::PointerButton);
  release.acceptedInput.button = ViewerSessionPointerButton::Primary;
  release.acceptedInput.pressed = false;
  auto ended = reduce(&state, &release, &session, scene, workspace);
  assert(ended.accepted() && ended.stateChanged &&
         !state.pointerCaptureActive);
}

void testPanZoomAndAnalyticalPrimary() {
  ViewerWorkspaceState workspace = makeWorkspace();
  const ViewerUiScene scene = makeScene(workspace);
  ViewerSessionState session{};
  ViewerCameraInteractionState state{};
  const ScreenRect scope = content(scene, 7);
  auto press = accepted(1u, ViewerSessionInputKind::PointerButton);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  press.acceptedInput.logicalX = scope.x0 + 80.0;
  press.acceptedInput.logicalY = scope.y0 + 80.0;
  assert(reduce(&state, &press, &session, scene, workspace, true).accepted());
  assert(state.pointerMode == ViewerCameraInteractionMode::Pan);
  auto move = accepted(2u, ViewerSessionInputKind::PointerMoved);
  move.acceptedInput.logicalX = press.acceptedInput.logicalX + 30.0;
  move.acceptedInput.logicalY = press.acceptedInput.logicalY + 15.0;
  const auto panned = reduce(&state, &move, &session, scene, workspace);
  assert(panned.cameraChanged && panned.windowId == 7 &&
         panned.camera.panX != workspace.windows[1].camera.panX &&
         panned.camera.panY != workspace.windows[1].camera.panY);

  state = {};
  const ScreenRect cube = content(scene, 42);
  press = accepted(3u, ViewerSessionInputKind::PointerButton);
  press.acceptedInput.button = ViewerSessionPointerButton::Secondary;
  press.acceptedInput.pressed = true;
  press.acceptedInput.logicalX = cube.x0 + 100.0;
  press.acceptedInput.logicalY = cube.y0 + 100.0;
  assert(reduce(&state, &press, &session, scene, workspace, true).accepted());
  assert(state.pointerMode == ViewerCameraInteractionMode::Zoom);
  move = accepted(4u, ViewerSessionInputKind::PointerMoved);
  move.acceptedInput.logicalX = press.acceptedInput.logicalX;
  move.acceptedInput.logicalY = press.acceptedInput.logicalY + 40.0;
  const auto zoomed = reduce(&state, &move, &session, scene, workspace);
  assert(zoomed.cameraChanged &&
         zoomed.camera.distance > workspace.windows[0].camera.distance);
}

void testScrollModifiersAndGestures() {
  const ViewerWorkspaceState workspace = makeWorkspace();
  const ViewerUiScene scene = makeScene(workspace);
  const ScreenRect cube = content(scene, 42);
  ViewerSessionState session{};
  session.pointerX = cube.x0 + 100.0;
  session.pointerY = cube.y0 + 100.0;

  ViewerCameraInteractionState normalState{};
  auto normal = accepted(1u, ViewerSessionInputKind::Scroll);
  normal.acceptedInput.deltaY = 1.0;
  const auto normalZoom = reduce(
      &normalState, &normal, &session, scene, workspace);
  assert(normalZoom.cameraChanged);

  ViewerCameraInteractionState preciseState{};
  auto precise = accepted(1u, ViewerSessionInputKind::Scroll);
  precise.acceptedInput.deltaY = 1.0;
  precise.acceptedInput.modifiers = kViewerSessionModifierShift;
  const auto preciseZoom = reduce(
      &preciseState, &precise, &session, scene, workspace);
  assert(preciseZoom.cameraChanged);
  assert(std::abs(preciseZoom.camera.distance - workspace.windows[0].camera.distance) <
         std::abs(normalZoom.camera.distance - workspace.windows[0].camera.distance));

  ViewerCameraInteractionState gestureState{};
  auto magnify = accepted(1u, ViewerSessionInputKind::Gesture);
  magnify.acceptedInput.gesture = ViewerSessionGestureKind::Magnify;
  magnify.acceptedInput.gesturePhase = ViewerSessionGesturePhase::Begin;
  magnify.acceptedInput.gestureDelta = 0.2;
  auto magnified = reduce(
      &gestureState, &magnify, &session, scene, workspace);
  assert(magnified.cameraChanged && gestureState.gestureCaptureActive);

  ViewerWorkspaceState magnifiedWorkspace = workspace;
  commit(&magnifiedWorkspace, magnified);
  auto end = accepted(2u, ViewerSessionInputKind::Gesture);
  end.acceptedInput.gesture = ViewerSessionGestureKind::Magnify;
  end.acceptedInput.gesturePhase = ViewerSessionGesturePhase::End;
  const auto magnifyEnd = reduce(
      &gestureState, &end, &session, scene, magnifiedWorkspace);
  assert(magnifyEnd.accepted() && !gestureState.gestureCaptureActive);

  ViewerCameraInteractionState rotateState{};
  auto rotate = accepted(1u, ViewerSessionInputKind::Gesture);
  rotate.acceptedInput.gesture = ViewerSessionGestureKind::Rotate;
  rotate.acceptedInput.gesturePhase = ViewerSessionGesturePhase::Begin;
  rotate.acceptedInput.gestureDelta = 0.3;
  const auto rolled = reduce(
      &rotateState, &rotate, &session, scene, workspace);
  assert(rolled.cameraChanged && std::abs(rolled.camera.qz) > 0.01f);
}

void testCancellationReplayInvalidAndClamp() {
  ViewerWorkspaceState workspace = makeWorkspace();
  const ViewerUiScene scene = makeScene(workspace);
  const ScreenRect cube = content(scene, 42);
  ViewerSessionState session{};
  ViewerCameraInteractionState state{};
  auto press = accepted(5u, ViewerSessionInputKind::PointerButton);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  press.acceptedInput.logicalX = cube.x0 + 50.0;
  press.acceptedInput.logicalY = cube.y0 + 50.0;
  assert(reduce(&state, &press, &session, scene, workspace, true).accepted());
  const ViewerCameraInteractionState afterPress = state;
  assert(reduce(&state, &press, &session, scene, workspace, true).status ==
         ViewerCameraInteractionStatus::ReplayedInput);
  assert(state.pointerCaptureActive == afterPress.pointerCaptureActive &&
         state.interactionRevision == afterPress.interactionRevision);

  auto cancelled = accepted(6u, ViewerSessionInputKind::Cancelled);
  cancelled.cancelInteractions = true;
  const auto cancel = reduce(&state, &cancelled, &session, scene, workspace);
  assert(cancel.accepted() && !state.pointerCaptureActive);

  ViewerUiScene invalidScene = scene;
  invalidScene.status = ViewerUiSceneStatus::InvalidViewport;
  auto scroll = accepted(7u, ViewerSessionInputKind::Scroll);
  scroll.acceptedInput.deltaY = 1.0;
  const ViewerCameraInteractionState beforeInvalid = state;
  assert(reduceViewerCameraInteraction(
             &state, {&scroll, &session, &invalidScene, &workspace, false})
             .status == ViewerCameraInteractionStatus::InvalidScene);
  assert(state.lastInputSequence == beforeInvalid.lastInputSequence);

  workspace.windows[0].camera.orthographic = true;
  workspace.windows[0].camera.distance = 1.1f;
  session.pointerX = cube.x0 + 50.0;
  session.pointerY = cube.y0 + 50.0;
  ViewerCameraInteractionState clampState{};
  auto huge = accepted(1u, ViewerSessionInputKind::Scroll);
  huge.acceptedInput.deltaY = 1.0e8;
  const auto clamped = reduce(
      &clampState, &huge, &session, scene, workspace);
  assert(clamped.cameraChanged && std::isfinite(clamped.camera.distance));
  assert(clamped.camera.distance >= 1.0f && clamped.camera.distance <= 1000.0f);
}

void testSourceSignalDoesNotCaptureCamera() {
  ViewerWorkspaceState workspace = makeWorkspace();
  workspace.windows[0].viewState.plotModel = kPlotModelSourceSignal;
  const ViewerUiScene scene = makeScene(workspace);
  const ScreenRect rect = content(scene, 42);
  ViewerSessionState session{};
  session.pointerX = (rect.x0 + rect.x1) * 0.5;
  session.pointerY = (rect.y0 + rect.y1) * 0.5;
  ViewerCameraInteractionState state{};
  auto press = accepted(1u, ViewerSessionInputKind::PointerButton);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  press.acceptedInput.logicalX = session.pointerX;
  press.acceptedInput.logicalY = session.pointerY;
  const auto began = reduce(&state, &press, &session, scene, workspace, true);
  assert(began.accepted() && !state.pointerCaptureActive &&
         !began.cameraChanged);

  auto scroll = accepted(2u, ViewerSessionInputKind::Scroll);
  scroll.acceptedInput.deltaY = 3.0;
  const auto zoom = reduce(&state, &scroll, &session, scene, workspace);
  assert(zoom.accepted() && !zoom.cameraChanged);
}

}  // namespace

int main() {
  testOrbitCaptureOutsideAndRelease();
  testPanZoomAndAnalyticalPrimary();
  testScrollModifiersAndGestures();
  testCancellationReplayInvalidAndClamp();
  testSourceSignalDoesNotCaptureCamera();
  return 0;
}
