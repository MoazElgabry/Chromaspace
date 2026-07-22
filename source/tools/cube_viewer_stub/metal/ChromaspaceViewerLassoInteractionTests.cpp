#include "ChromaspaceViewerLassoInteraction.h"

#include <cassert>
#include <cmath>

namespace {

using namespace ChromaspaceViewer;

ViewerWorkspaceState workspace() {
  ViewerWorkspaceState state{};
  state.focusedWindowId = 91;
  state.nextWindowId = 100;
  state.sourceLassoSessionActive = true;
  state.sourceLassoTargetWindowId = 91;
  PlotWindowDomainState source{};
  source.windowId = 17;
  source.rect = {0.48f, 0.08f, 0.46f, 0.82f};
  source.viewState.plotModel = kPlotModelSourceSignal;
  PlotWindowDomainState owner{};
  owner.windowId = 91;
  owner.rect = {0.05f, 0.08f, 0.38f, 0.82f};
  owner.viewState.plotModel = kPlotModelCube;
  owner.viewState.volumeSliceLassoRegion = true;
  owner.selected = true;
  // Deliberately unrelated order: every reducer join must be by ID.
  state.windows = {source, owner};
  assert(validateViewerWorkspaceState(state));
  return state;
}

ViewerUiScene scene(const ViewerWorkspaceState& state) {
  ViewerFramePlanRequest request{};
  request.windowWidth = 1000;
  request.windowHeight = 700;
  request.framebufferWidth = 2000;
  request.framebufferHeight = 1400;
  request.windows = {
      {91, state.windows[1].rect, kPlotModelCube, 1u, true},
      {17, state.windows[0].rect, kPlotModelSourceSignal, 1u, true},
  };
  const auto plan = buildViewerFramePlan(request);
  ViewerUiSceneInput input{};
  input.focusedWindowId = 91;
  for (const auto& planned : plan.windows) {
    ViewerUiPlotWindowInput window{};
    window.windowId = planned.windowId;
    window.title = "Plot";
    window.sourceLasso.visible = planned.windowId == 17;
    input.windows.push_back(window);
  }
  const auto result = buildViewerUiScene(plan, input);
  assert(result.ready());
  return result;
}

const ViewerUiWindowScene& uiWindow(const ViewerUiScene& value, int id) {
  for (const auto& window : value.windows) {
    if (window.windowId == id) return window;
  }
  assert(false);
  return value.windows.front();
}

ViewerSessionReduceResult input(uint64_t sequence,
                                ViewerSessionInputKind kind,
                                double x,
                                double y) {
  ViewerSessionReduceResult result{};
  result.status = ViewerSessionReduceStatus::Applied;
  result.inputAccepted = true;
  result.acceptedInput.sequence = sequence;
  result.acceptedInput.kind = kind;
  result.acceptedInput.logicalX = x;
  result.acceptedInput.logicalY = y;
  return result;
}

ViewerLassoInteractionResult reduce(
    ViewerLassoInteractionState* state,
    ViewerSessionReduceResult* event,
    const ViewerSessionState& session,
    const ViewerUiScene& ui,
    const ViewerWorkspaceState& workspace,
    bool authorize = false,
    double width = 1920.0,
    double height = 1080.0) {
  return reduceViewerLassoInteraction(
      state, {event, &session, &ui, &workspace, authorize, false,
              width, height});
}

void testAspectFitAndMapping() {
  ViewerLassoSourceImageRect fitted{};
  assert(computeViewerLassoSourceImageRect({0.0f, 0.0f, 100.0f, 100.0f},
                                           200.0, 100.0, &fitted));
  assert(std::abs(fitted.rect.y0 - 25.0f) < 1.0e-5f &&
         std::abs(fitted.rect.y1 - 75.0f) < 1.0e-5f);
  LassoPointNorm point{};
  assert(!mapViewerLassoPointerToSource(fitted, 50.0, 10.0, false, &point));
  assert(mapViewerLassoPointerToSource(fitted, 50.0, 25.0, false, &point));
  assert(std::abs(point.xNorm - 0.5f) < 1.0e-5f && point.yNorm == 1.0f);
  assert(mapViewerLassoPointerToSource(fitted, -50.0, 200.0, true, &point));
  assert(point.xNorm == 0.0f && point.yNorm == 0.0f);
}

void testCaptureClampSubtractAndCompletion() {
  ViewerWorkspaceState state = workspace();
  state.sourceLassoSubtractMode = true;
  const ViewerUiScene ui = scene(state);
  const auto& source = uiWindow(ui, 17);
  ViewerLassoSourceImageRect image{};
  assert(computeViewerLassoSourceImageRect(source.contentRect, 1920.0,
                                           1080.0, &image));
  ViewerSessionState session{};
  ViewerLassoInteractionState interaction{};
  auto press = input(1u, ViewerSessionInputKind::PointerButton,
                     (image.rect.x0 + image.rect.x1) * 0.5,
                     (image.rect.y0 + image.rect.y1) * 0.5);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  const auto began = reduce(&interaction, &press, session, ui, state, true);
  assert(began.accepted() && interaction.pointerCaptureActive &&
         interaction.pointerWindowId == 17 && interaction.strokeSubtract &&
         interaction.pointCount == 1u);

  state.sourceLassoSubtractMode = false;  // begin snapshot remains authoritative
  auto duplicate = input(2u, ViewerSessionInputKind::PointerMoved,
                         press.acceptedInput.logicalX,
                         press.acceptedInput.logicalY);
  const auto ignored = reduce(&interaction, &duplicate, session, ui, state);
  assert(ignored.accepted() && !ignored.pointAppended &&
         interaction.pointCount == 1u);

  auto upperLeft = input(3u, ViewerSessionInputKind::PointerMoved,
                         image.rect.x0 - 100.0, image.rect.y0 - 100.0);
  assert(reduce(&interaction, &upperLeft, session, ui, state).pointAppended);
  auto lowerRight = input(4u, ViewerSessionInputKind::PointerMoved,
                          image.rect.x1 + 100.0, image.rect.y1 + 100.0);
  assert(reduce(&interaction, &lowerRight, session, ui, state).pointAppended);
  auto release = input(5u, ViewerSessionInputKind::PointerButton,
                       image.rect.x1 + 100.0, image.rect.y1 + 100.0);
  release.acceptedInput.button = ViewerSessionPointerButton::Primary;
  release.acceptedInput.pressed = false;
  const auto completed = reduce(&interaction, &release, session, ui, state);
  assert(completed.accepted() && completed.strokeCompleted &&
         !interaction.pointerCaptureActive && completed.stroke.subtract &&
         completed.stroke.points.size() == 3u &&
         completed.stroke.boundsValid);
  assert(completed.stroke.minXNorm >= 0.0f &&
         completed.stroke.maxXNorm <= 1.0f &&
         completed.stroke.minYNorm >= 0.0f &&
         completed.stroke.maxYNorm <= 1.0f);
}

void testLetterboxDiscardReplayCancelAndCapacity() {
  const ViewerWorkspaceState state = workspace();
  const ViewerUiScene ui = scene(state);
  const auto& source = uiWindow(ui, 17);
  ViewerLassoSourceImageRect image{};
  assert(computeViewerLassoSourceImageRect(source.contentRect, 1920.0,
                                           1080.0, &image));
  ViewerSessionState session{};
  ViewerLassoInteractionState interaction{};
  auto bar = input(1u, ViewerSessionInputKind::PointerButton,
                   (source.contentRect.x0 + source.contentRect.x1) * 0.5,
                   source.contentRect.y0 + 1.0);
  bar.acceptedInput.button = ViewerSessionPointerButton::Primary;
  bar.acceptedInput.pressed = true;
  const auto rejectedBar = reduce(&interaction, &bar, session, ui, state, true);
  assert(rejectedBar.status == ViewerLassoInteractionStatus::LetterboxRejected &&
         !interaction.pointerCaptureActive);
  assert(reduce(&interaction, &bar, session, ui, state, true).status ==
         ViewerLassoInteractionStatus::ReplayedInput);

  auto press = input(2u, ViewerSessionInputKind::PointerButton,
                     (image.rect.x0 + image.rect.x1) * 0.5,
                     (image.rect.y0 + image.rect.y1) * 0.5);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  assert(reduce(&interaction, &press, session, ui, state, true).accepted());
  auto release = input(3u, ViewerSessionInputKind::PointerButton,
                       press.acceptedInput.logicalX,
                       press.acceptedInput.logicalY);
  release.acceptedInput.button = ViewerSessionPointerButton::Primary;
  release.acceptedInput.pressed = false;
  const auto discarded = reduce(&interaction, &release, session, ui, state);
  assert(discarded.status == ViewerLassoInteractionStatus::TooFewPoints &&
         discarded.strokeDiscarded && !discarded.strokeCompleted);

  press.acceptedInput.sequence = 4u;
  assert(reduce(&interaction, &press, session, ui, state, true).accepted());
  auto cancel = input(5u, ViewerSessionInputKind::Cancelled, 0.0, 0.0);
  cancel.cancelInteractions = true;
  assert(reduce(&interaction, &cancel, session, ui, state).accepted() &&
         !interaction.pointerCaptureActive);

  ViewerLassoInteractionState full{};
  full.pointerCaptureActive = true;
  full.pointerWindowId = 17;
  full.captureSourceWidth = 1920.0;
  full.captureSourceHeight = 1080.0;
  full.pointCount = full.points.size();
  full.boundsValid = true;
  full.minXNorm = full.maxXNorm = 0.0f;
  full.minYNorm = full.maxYNorm = 0.0f;
  auto move = input(1u, ViewerSessionInputKind::PointerMoved,
                    image.rect.x1, image.rect.y0);
  const auto capacity = reduce(&full, &move, session, ui, state);
  assert(capacity.status == ViewerLassoInteractionStatus::CapacityExceeded &&
         full.pointCount == kViewerWorkspaceMaxLassoPointsPerStroke);
}

void testInvalidSnapshotsAndVanishedWindow() {
  ViewerWorkspaceState state = workspace();
  ViewerUiScene ui = scene(state);
  const auto& source = uiWindow(ui, 17);
  ViewerSessionState session{};
  ViewerLassoInteractionState interaction{};
  auto press = input(1u, ViewerSessionInputKind::PointerButton,
                     (source.contentRect.x0 + source.contentRect.x1) * 0.5,
                     (source.contentRect.y0 + source.contentRect.y1) * 0.5);
  press.acceptedInput.button = ViewerSessionPointerButton::Primary;
  press.acceptedInput.pressed = true;
  const auto before = interaction;
  assert(reduce(&interaction, &press, session, ui, state, true, 0.0, 1080.0)
             .status == ViewerLassoInteractionStatus::InvalidSourceDimensions);
  assert(interaction.interactionRevision == before.interactionRevision);

  ViewerUiScene invalidUi = ui;
  invalidUi.status = ViewerUiSceneStatus::InvalidViewport;
  assert(reduce(&interaction, &press, session, invalidUi, state, true).status ==
         ViewerLassoInteractionStatus::InvalidScene);
  ViewerWorkspaceState invalidWorkspace = state;
  invalidWorkspace.focusedWindowId = 404;
  assert(reduce(&interaction, &press, session, ui, invalidWorkspace, true)
             .status == ViewerLassoInteractionStatus::InvalidWorkspace);

  assert(reduce(&interaction, &press, session, ui, state, true).accepted());
  ViewerWorkspaceState vanished = state;
  vanished.windows.erase(vanished.windows.begin());
  auto move = input(2u, ViewerSessionInputKind::PointerMoved,
                    press.acceptedInput.logicalX + 5.0,
                    press.acceptedInput.logicalY + 5.0);
  const auto captured = interaction;
  assert(reduce(&interaction, &move, session, ui, vanished).status ==
         ViewerLassoInteractionStatus::MissingWindow);
  assert(interaction.pointerCaptureActive == captured.pointerCaptureActive &&
         interaction.pointCount == captured.pointCount &&
         interaction.interactionRevision == captured.interactionRevision);
}

void testRetainedAndActiveOverlayProjection() {
  ViewerWorkspaceState state = workspace();
  LassoStroke retained{};
  retained.points = {{0.1f, 0.1f}, {0.8f, 0.2f}, {0.5f, 0.9f}};
  retained.boundsValid = true;
  retained.minXNorm = 0.1f;
  retained.maxXNorm = 0.8f;
  retained.minYNorm = 0.1f;
  retained.maxYNorm = 0.9f;
  assert(appendViewerWorkspaceLassoStroke(&state, retained).accepted());

  ViewerLassoInteractionState interaction{};
  interaction.pointerCaptureActive = true;
  interaction.pointerWindowId = 17;
  interaction.captureSourceWidth = 1920.0;
  interaction.captureSourceHeight = 1080.0;
  interaction.pointCount = 2u;
  interaction.points[0] = {0.2f, 0.3f};
  interaction.points[1] = {0.7f, 0.6f};
  interaction.boundsValid = true;
  interaction.minXNorm = 0.2f;
  interaction.maxXNorm = 0.7f;
  interaction.minYNorm = 0.3f;
  interaction.maxYNorm = 0.6f;
  ViewerUiScene ui = scene(state);
  const std::size_t before = ui.vectors.size();
  const auto overlay = appendViewerLassoOverlay(
      interaction, state, 1920.0, 1080.0, &ui);
  assert(overlay.ready() && overlay.retainedSegments == 3u &&
         overlay.activeSegments == 1u && overlay.appendedVertices == 24u &&
         ui.vectors.size() == before + 24u);
  for (std::size_t i = before; i < ui.vectors.size(); ++i) {
    assert(std::isfinite(ui.vectors[i].x) && std::isfinite(ui.vectors[i].y) &&
           ui.vectors[i].windowId == 17);
  }
}

void testDockedSourceHidesOverlay() {
  ViewerWorkspaceState state = workspace();
  state.windows[0].sourceSignalDocked = true;
  state.windows[0].sourceSignalDockOwnerWindowId = 91;
  state.windows[0].sourceSignalRestoreRect = state.windows[0].rect;
  assert(validateViewerWorkspaceState(state));
  ViewerUiScene ui = scene(state);
  ui.windows.erase(ui.windows.begin() + 1);
  ViewerLassoInteractionState interaction{};
  const std::size_t before = ui.vectors.size();
  const auto overlay = appendViewerLassoOverlay(
      interaction, state, 1920.0, 1080.0, &ui);
  assert(overlay.ready() && overlay.appendedVertices == 0u &&
         overlay.retainedSegments == 0u && overlay.activeSegments == 0u &&
         ui.vectors.size() == before);
}

}  // namespace

int main() {
  testAspectFitAndMapping();
  testCaptureClampSubtractAndCompletion();
  testLetterboxDiscardReplayCancelAndCapacity();
  testInvalidSnapshotsAndVanishedWindow();
  testRetainedAndActiveOverlayProjection();
  testDockedSourceHidesOverlay();
  return 0;
}
