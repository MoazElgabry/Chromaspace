#include "ChromaspaceViewerController.h"
#include "ChromaspaceViewerState.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using namespace ChromaspaceViewer;

ViewerSessionViewport viewport(bool retina = true) {
  return {640, 360, retina ? 1280 : 640, retina ? 720 : 360,
          retina ? 2.0f : 1.0f, retina ? 2.0f : 1.0f};
}

ViewerFramePlan makePlan(bool retina = true) {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 360;
  request.framebufferWidth = retina ? 1280 : 640;
  request.framebufferHeight = retina ? 720 : 360;
  request.windows = {
      {1, {0.05f, 0.12f, 0.40f, 0.72f}, kPlotModelCube, 10, true},
      {2, {0.55f, 0.12f, 0.40f, 0.72f}, kPlotModelSourceSignal, 20, true},
  };
  return buildViewerFramePlan(request);
}

ViewerFramePlan makeSinglePlan(int windowId, bool retina = true) {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 360;
  request.framebufferWidth = retina ? 1280 : 640;
  request.framebufferHeight = retina ? 720 : 360;
  request.windows = {{windowId,
                      windowId == 1 ? PlotWindowRectNorm{0.05f, 0.12f,
                                                           0.40f, 0.72f}
                                    : PlotWindowRectNorm{0.55f, 0.12f,
                                                         0.40f, 0.72f},
                      windowId == 1 ? kPlotModelCube
                                    : kPlotModelSourceSignal,
                      30,
                      true}};
  return buildViewerFramePlan(request);
}

ViewerUiScene makeScene(const ViewerFramePlan& plan,
                        int focusedWindowId = 1,
                        bool controls = true,
                        bool sourceSelection = true,
                        bool sourceRestore = false) {
  ViewerUiSceneInput input{};
  input.toolbar.visible = controls;
  input.toolbar.layoutLabel = "Split 2";
  input.toolbar.layoutIndex = 1;
  input.focusedWindowId = focusedWindowId;
  input.windows.reserve(plan.windows.size());
  for (const auto& window : plan.windows) {
    ViewerUiPlotWindowInput requested{};
    requested.windowId = window.windowId;
    requested.title = window.windowId == 1 ? "Cube" : "Source";
    requested.metadata = "Signal";
    requested.titleMetrics.titleExtraHeight = 2.0f;
    requested.titleMetrics.fontAvailable = false;
    requested.closable = true;
    if (controls && window.windowId == 1) {
      requested.slicing.visible = true;
      requested.slicing.drawerOpen = true;
      requested.slicing.active = true;
      requested.slicing.vectors = {true, false, true, false, true, false};
      requested.slicing.lassoActive = true;
      requested.slicing.animationProgress = 1.0f;
      if (sourceRestore) requested.sourceSignalRestoreWindowIds = {2};
    }
    if (controls && window.windowId == 2) {
      requested.sourceLasso.visible = true;
      requested.sourceLasso.hasSelection = sourceSelection;
    }
    input.windows.push_back(requested);
  }
  return buildViewerUiScene(plan, input);
}

ViewerSessionState initializedSession(bool retina = true) {
  ViewerSessionState state{};
  const auto result = reduceViewerSession(
      &state, {1, ViewerSessionInitialize{viewport(retina), true, true, false}});
  assert(result.accepted());
  return state;
}

template <typename Payload>
ViewerSessionReduceResult input(ViewerSessionState* state, Payload payload) {
  ViewerSessionEvent event{};
  event.sequence = viewerSessionNextSequence(*state);
  event.payload = payload;
  return reduceViewerSession(state, event);
}

std::pair<float, float> center(const ViewerUiHitRegion& hit) {
  return {(hit.rect.x0 + hit.rect.x1) * 0.5f,
          (hit.rect.y0 + hit.rect.y1) * 0.5f};
}

const ViewerUiHitRegion* findHit(const ViewerUiScene& scene,
                                 int windowId,
                                 ViewerUiControlKind control,
                                 int controlIndex = -1) {
  for (const auto& hit : scene.hits) {
    if (hit.windowId == windowId && hit.control == control &&
        hit.controlIndex == controlIndex && hit.actionable) {
      return &hit;
    }
  }
  return nullptr;
}

bool sameSnapshot(const ViewerControllerStateSnapshot& a,
                  const ViewerControllerStateSnapshot& b) {
  return a.focusedWindowId == b.focusedWindowId &&
         a.hoveredWindowId == b.hoveredWindowId &&
         a.hoveredDragMode == b.hoveredDragMode &&
         a.windowDragActive == b.windowDragActive &&
         a.windowDragWindowId == b.windowDragWindowId &&
         a.windowDragMode == b.windowDragMode &&
         a.windowDragStartX == b.windowDragStartX &&
         a.windowDragStartY == b.windowDragStartY &&
         a.windowDragStartRect.x == b.windowDragStartRect.x &&
         a.windowDragStartRect.y == b.windowDragStartRect.y &&
         a.windowDragStartRect.w == b.windowDragStartRect.w &&
         a.windowDragStartRect.h == b.windowDragStartRect.h &&
         a.slicingPaintActive == b.slicingPaintActive &&
         a.slicingPaintWindowId == b.slicingPaintWindowId &&
         a.slicingPaintDesired == b.slicingPaintDesired &&
         a.slicingPaintLastIndex == b.slicingPaintLastIndex &&
         a.lastConsumedInputSequence == b.lastConsumedInputSequence;
}

ViewerControllerCommandBatch press(ViewerController* controller,
                                   ViewerSessionState* session,
                                   float x,
                                   float y,
                                   ViewerSessionPointerButton button =
                                       ViewerSessionPointerButton::Primary,
                                   ViewerSessionModifierMask modifiers = 0,
                                   uint8_t clickCount = 1) {
  const auto result = input(
      session, ViewerSessionPointerButtonChanged{button, true, x, y,
                                                 modifiers, clickCount});
  assert(result.accepted() && result.inputAccepted);
  return controller->consume(result, *session);
}

ViewerControllerCommandBatch move(ViewerController* controller,
                                  ViewerSessionState* session,
                                  float x,
                                  float y) {
  const auto result = input(session, ViewerSessionPointerMoved{x, y});
  assert(result.accepted() && result.inputAccepted);
  return controller->consume(result, *session);
}

ViewerControllerCommandBatch release(ViewerController* controller,
                                     ViewerSessionState* session,
                                     float x,
                                     float y,
                                     ViewerSessionPointerButton button =
                                         ViewerSessionPointerButton::Primary) {
  const auto result = input(session,
                            ViewerSessionPointerButtonChanged{
                                button, false, x, y, 0, 1});
  assert(result.accepted() && result.inputAccepted);
  return controller->consume(result, *session);
}

ViewerControllerCommandBatch click(ViewerController* controller,
                                   ViewerSessionState* session,
                                   float x,
                                   float y,
                                   ViewerSessionPointerButton button =
                                       ViewerSessionPointerButton::Primary) {
  const auto result = press(controller, session, x, y, button);
  (void)release(controller, session, x, y, button);
  return result;
}

void testAtomicPublishAndFocusProjection() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene = makeScene(plan, 1, false);
  ViewerController invalidSeed;
  assert(!invalidSeed.publishScene(scene, 99));
  assert(!invalidSeed.hasScene());
  ViewerController controller;
  assert(controller.publishScene(scene, 2));
  assert(controller.state().focusedWindowId == 2);

  ViewerUiScene nonfinite = scene;
  nonfinite.geometry.scaleX = std::numeric_limits<float>::quiet_NaN();
  assert(!controller.publishScene(nonfinite));
  assert(controller.state().focusedWindowId == 2);

  ViewerUiScene invalid = scene;
  invalid.windows.front().primitiveBegin = invalid.primitives.size() + 1u;
  assert(!controller.publishScene(invalid));
  assert(controller.state().focusedWindowId == 2);
  assert(controller.scene().windows.size() == 2u);

  ViewerUiScene orphan = scene;
  orphan.primitives[orphan.windows.front().primitiveBegin].windowId = 99;
  assert(!controller.publishScene(orphan));
  assert(controller.state().focusedWindowId == 2);

  const ViewerUiScene oneWindow = makeScene(makeSinglePlan(1), 1, false);
  assert(controller.publishScene(oneWindow, 1));
  assert(controller.state().focusedWindowId == 1);
}

void testBackgroundAndToolbarSemantics() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene = makeScene(plan, 1, true);
  ViewerController controller;
  assert(controller.publishScene(scene, 1));
  ViewerSessionState session = initializedSession();

  const auto background = press(&controller, &session, 630.0f, 350.0f);
  assert(!background.consumed && background.empty());
  (void)release(&controller, &session, 630.0f, 350.0f);

  const auto* menu = findHit(scene, -1, ViewerUiControlKind::ToolbarMenu);
  assert(menu != nullptr);
  const auto point = center(*menu);
  const auto toolbar = press(&controller, &session, point.first, point.second);
  assert(toolbar.consumed && toolbar.count == 1u);
  assert(toolbar[0].kind == ViewerControllerCommandKind::ToolbarMenu);
  (void)release(&controller, &session, point.first, point.second);

  const ViewerControllerCommandKind toolbarKinds[] = {
      ViewerControllerCommandKind::ToolbarAddPlot,
      ViewerControllerCommandKind::ToolbarLayoutPreset,
  };
  const ViewerUiControlKind toolbarControls[] = {
      ViewerUiControlKind::ToolbarAddPlot,
      ViewerUiControlKind::ToolbarLayoutPreset,
  };
  for (std::size_t i = 0; i < 2u; ++i) {
    const auto* hit = findHit(scene, -1, toolbarControls[i]);
    assert(hit != nullptr);
    const auto p = center(*hit);
    const auto command = click(&controller, &session, p.first, p.second);
    assert(command.consumed && command.count == 1u);
    assert(command[0].kind == toolbarKinds[i]);
  }

  const auto hoverResult = input(&session, ViewerSessionPointerMoved{60, 120});
  assert(hoverResult.accepted() && hoverResult.inputAccepted);
  (void)controller.consume(hoverResult, session);
  assert(controller.state().hoveredWindowId == 1);
  const auto leaveResult = input(&session, ViewerSessionPointerLeft{});
  assert(leaveResult.accepted() && leaveResult.inputAccepted);
  (void)controller.consume(leaveResult, session);
  assert(controller.state().hoveredWindowId == -1);

  const auto replay = controller.consume(
      {ViewerSessionReduceStatus::Applied,
       session.lastInput,
       true,
       true,
       false,
       false,
       false,
       false},
      session);
  assert(replay.empty() && !replay.consumed);

  const ViewerControllerStateSnapshot beforeRejected = controller.state();
  const auto staleResult = reduceViewerSession(
      &session, {session.lastAcceptedSequence,
                 ViewerSessionVisibilityChanged{true}});
  assert(!staleResult.accepted());
  const auto staleBatch = controller.consume(staleResult, session);
  assert(staleBatch.empty());
  assert(sameSnapshot(beforeRejected, controller.state()));
}

void testPlotFocusAndDrag() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene = makeScene(plan, 1, false);
  ViewerController controller;
  assert(controller.publishScene(scene, 1));
  ViewerSessionState session = initializedSession();

  const auto& window = scene.windows.front();
  const float centerX = (window.contentRect.x0 + window.contentRect.x1) * 0.5f;
  const float centerY = (window.contentRect.y0 + window.contentRect.y1) * 0.5f;
  const auto middle = click(&controller, &session, centerX, centerY,
                            ViewerSessionPointerButton::Middle);
  assert(!middle.consumed && middle.count == 1u);
  assert(middle[0].kind == ViewerControllerCommandKind::FocusWindow);
  const auto camera = press(&controller, &session, centerX, centerY);
  assert(!camera.consumed && camera.continueCamera &&
         camera.continueSourceLasso);
  assert(camera.count == 1u &&
         camera[0].kind == ViewerControllerCommandKind::FocusWindow);
  (void)release(&controller, &session, centerX, centerY);

  const float edgeX = window.rect.x0 + 1.0f;
  const float edgeY = (window.rect.y0 + window.rect.y1) * 0.5f;
  const auto hit = viewerUiHitTest(scene, edgeX, edgeY);
  assert(hit.control == ViewerUiControlKind::PlotBody);
  assert(hit.dragMode != PlotWindowDragMode::None);
  const auto begin = press(&controller, &session, edgeX, edgeY);
  assert(begin.consumed && begin.count == 2u);
  assert(begin[1].kind == ViewerControllerCommandKind::BeginWindowDrag);
  const auto update = move(&controller, &session, edgeX - 30.0f, edgeY);
  assert(update.consumed && update.count == 1u);
  assert(update[0].kind == ViewerControllerCommandKind::UpdateWindowDrag);
  assert(update[0].rect.w > 0.0f && update[0].rect.h > 0.0f);
  const auto end = release(&controller, &session, edgeX - 30.0f, edgeY);
  assert(end.consumed && end.count == 1u);
  assert(end[0].kind == ViewerControllerCommandKind::EndWindowDrag);
  assert(!controller.state().windowDragActive);

  const auto beginAgain = press(&controller, &session, edgeX, edgeY);
  assert(beginAgain.consumed && controller.state().windowDragActive);
  const auto focusLost = input(&session, ViewerSessionFocusChanged{false});
  assert(focusLost.accepted() && focusLost.cancelInteractions);
  const auto cancelled = controller.consume(focusLost, session);
  assert(cancelled.empty() && !controller.state().windowDragActive);
}

void testDragClampAndRetinaInvariant() {
  ViewerController logicalController;
  ViewerController retinaController;
  const ViewerUiScene logicalScene = makeScene(makePlan(false), 1, false);
  const ViewerUiScene retinaScene = makeScene(makePlan(true), 1, false);
  assert(logicalController.publishScene(logicalScene, 1));
  assert(retinaController.publishScene(retinaScene, 1));
  ViewerSessionState logicalSession = initializedSession(false);
  ViewerSessionState retinaSession = initializedSession(true);

  const auto& logicalWindow = logicalScene.windows.front();
  const float edgeX = logicalWindow.rect.x0 + 1.0f;
  const float edgeY = (logicalWindow.rect.y0 + logicalWindow.rect.y1) * 0.5f;
  const auto logicalBegin = press(&logicalController, &logicalSession, edgeX,
                                  edgeY);
  const auto retinaBegin = press(&retinaController, &retinaSession, edgeX,
                                 edgeY);
  assert(logicalBegin.count == 2u && retinaBegin.count == 2u);
  const auto logicalUpdate = move(&logicalController, &logicalSession,
                                  -10000.0f, 10000.0f);
  const auto retinaUpdate = move(&retinaController, &retinaSession,
                                 -10000.0f, 10000.0f);
  assert(logicalUpdate.consumed && retinaUpdate.consumed);
  assert(logicalUpdate.count == 1u && retinaUpdate.count == 1u);
  const PlotWindowRectNorm a = logicalUpdate[0].rect;
  const PlotWindowRectNorm b = retinaUpdate[0].rect;
  assert(std::fabs(a.x - b.x) < 1e-5f && std::fabs(a.y - b.y) < 1e-5f &&
         std::fabs(a.w - b.w) < 1e-5f && std::fabs(a.h - b.h) < 1e-5f);
  assert(a.x >= 0.0f && a.y >= 0.0f && a.x + a.w <= 1.0f + 1e-5f &&
         a.y + a.h <= 1.0f + 1e-5f);
  assert(a.w >= plotWindowMinNormWidth(640) - 1e-5f);
  assert(a.h >= plotWindowMinNormHeight(360) - 1e-5f);
}

void testControlsPaintAndCancellation() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene = makeScene(plan, 1, true);
  ViewerController controller;
  assert(controller.publishScene(scene, 1));
  ViewerSessionState session = initializedSession();
  const auto* vector = findHit(scene, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(vector != nullptr);
  const auto vectorPoint = center(*vector);
  const auto first = press(&controller, &session, vectorPoint.first,
                           vectorPoint.second);
  assert(first.consumed && first.count == 2u);
  assert(first[0].kind == ViewerControllerCommandKind::FocusWindow);
  assert(first[1].kind == ViewerControllerCommandKind::SetSlicingVector);
  assert(first[1].enabled == !vector->selected);
  const auto duplicate = move(&controller, &session, vectorPoint.first,
                              vectorPoint.second);
  assert(duplicate.consumed && duplicate.empty());

  const auto* nextVector =
      findHit(scene, 1, ViewerUiControlKind::SlicingVector, 1);
  assert(nextVector != nullptr);
  const auto nextVectorPoint = center(*nextVector);
  const auto paint = move(&controller, &session, nextVectorPoint.first,
                          nextVectorPoint.second);
  assert(paint.consumed && paint.count == 1u);
  assert(paint[0].kind == ViewerControllerCommandKind::SetSlicingVector);
  (void)release(&controller, &session, nextVectorPoint.first,
                nextVectorPoint.second);

  const auto* quick =
      findHit(scene, 1, ViewerUiControlKind::SlicingQuickToggle);
  const auto* lasso = findHit(scene, 1, ViewerUiControlKind::SlicingLasso);
  assert(quick != nullptr && lasso != nullptr);
  const auto quickPoint = center(*quick);
  const auto quickCommand = click(&controller, &session, quickPoint.first,
                                  quickPoint.second);
  assert(quickCommand.consumed && quickCommand.count == 2u);
  assert(quickCommand[1].kind ==
         ViewerControllerCommandKind::ToggleSlicingDrawer);
  const auto lassoPoint = center(*lasso);
  const auto lassoCommand = click(&controller, &session, lassoPoint.first,
                                  lassoPoint.second);
  assert(lassoCommand.consumed && lassoCommand.count == 2u);
  assert(lassoCommand[1].kind ==
         ViewerControllerCommandKind::ToggleSlicingLasso);

  const ViewerUiControlKind sourceControls[] = {
      ViewerUiControlKind::SourceLassoAdd,
      ViewerUiControlKind::SourceLassoSubtract,
      ViewerUiControlKind::SourceLassoClear,
  };
  const ViewerControllerCommandKind sourceCommands[] = {
      ViewerControllerCommandKind::SourceLassoAdd,
      ViewerControllerCommandKind::SourceLassoSubtract,
      ViewerControllerCommandKind::SourceLassoClear,
  };
  for (std::size_t i = 0; i < 3u; ++i) {
    const auto* source = findHit(scene, 2, sourceControls[i]);
    assert(source != nullptr);
    const auto sourcePoint = center(*source);
    const auto sourceCommand = click(&controller, &session,
                                     sourcePoint.first, sourcePoint.second);
    assert(sourceCommand.consumed && sourceCommand.count == 1u);
    assert(sourceCommand[0].kind == sourceCommands[i]);
    assert(controller.state().focusedWindowId == 1);
  }

  ViewerController soloController;
  assert(soloController.publishScene(scene, 1));
  ViewerSessionState soloSession = initializedSession();
  const auto soloPress = press(
      &soloController, &soloSession, vectorPoint.first, vectorPoint.second,
      ViewerSessionPointerButton::Primary, kViewerSessionModifierAlt, 1);
  assert(soloPress.consumed && soloPress.count == 2u);
  assert(soloPress[1].kind == ViewerControllerCommandKind::SoloSlicingVector);
  assert(!soloController.state().slicingPaintActive);

  ViewerController doubleController;
  assert(doubleController.publishScene(scene, 1));
  ViewerSessionState doubleSession = initializedSession();
  const auto doublePress = press(
      &doubleController, &doubleSession, vectorPoint.first,
      vectorPoint.second, ViewerSessionPointerButton::Primary, 0, 2);
  assert(doublePress.consumed && doublePress.count == 2u);
  assert(doublePress[1].kind ==
         ViewerControllerCommandKind::ToggleAllSlicingVectors);
  assert(!doubleController.state().slicingPaintActive);

  const ViewerUiScene disabledScene = makeScene(plan, 1, true, false);
  assert(controller.publishScene(disabledScene));
  const auto* disabledClear =
      findHit(disabledScene, 2, ViewerUiControlKind::SourceLassoClear);
  assert(disabledClear == nullptr);
  const auto disabledPrimitive = [&]() {
    for (const auto& primitive : disabledScene.primitives) {
      if (primitive.windowId == 2 &&
          primitive.control == ViewerUiControlKind::SourceLassoClear) {
        return &primitive;
      }
    }
    return static_cast<const ViewerUiSolidRect*>(nullptr);
  }();
  assert(disabledPrimitive != nullptr);
  const auto disabledPoint = center(ViewerUiHitRegion{
      ViewerUiControlKind::SourceLassoClear, 2, disabledPrimitive->rect, {},
      false, -1, false, false, false});
  const auto disabledCommand = click(&controller, &session,
                                     disabledPoint.first, disabledPoint.second);
  assert(disabledCommand.count == 0u ||
         disabledCommand[0].kind != ViewerControllerCommandKind::SourceLassoClear);

  assert(controller.publishScene(scene, 1));
  const auto* vectorAgain =
      findHit(scene, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(vectorAgain != nullptr);
  const auto vectorAgainPoint = center(*vectorAgain);
  const auto activeAgain = press(&controller, &session, vectorAgainPoint.first,
                                 vectorAgainPoint.second);
  assert(activeAgain.consumed && controller.state().slicingPaintActive);
  const auto cancelResult = input(&session, ViewerSessionFocusChanged{false});
  assert(cancelResult.accepted() && cancelResult.cancelInteractions);
  const auto cancelled = controller.consume(cancelResult, session);
  assert(cancelled.empty() && !controller.state().slicingPaintActive);
}

void testCloseAndDisappearReconcilesCapture() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene = makeScene(plan, 2, false);
  ViewerController controller;
  assert(controller.publishScene(scene, 2));
  ViewerSessionState session = initializedSession();
  const auto* close = findHit(scene, 2, ViewerUiControlKind::PlotClose);
  assert(close != nullptr);
  const auto closePoint = center(*close);
  const auto command = press(&controller, &session, closePoint.first,
                             closePoint.second);
  assert(command.consumed && command.count == 1u);
  assert(command[0].kind == ViewerControllerCommandKind::RequestCloseWindow);

  const ViewerUiScene remaining = makeScene(makeSinglePlan(1), 1, false);
  assert(controller.publishScene(remaining, 1));
  assert(controller.state().focusedWindowId == 1);
  assert(controller.state().hoveredWindowId == -1);

  ViewerController dragController;
  const ViewerUiScene dragScene = makeScene(makePlan(), 1, false);
  assert(dragController.publishScene(dragScene, 1));
  ViewerSessionState dragSession = initializedSession();
  const auto& dragWindow = dragScene.windows.front();
  const float dragX = dragWindow.rect.x0 + 1.0f;
  const float dragY = (dragWindow.rect.y0 + dragWindow.rect.y1) * 0.5f;
  const auto dragBegin = press(&dragController, &dragSession, dragX, dragY);
  assert(dragBegin.consumed && dragController.state().windowDragActive);
  const ViewerUiScene onlySecond = makeScene(makeSinglePlan(2), 2, false);
  assert(dragController.publishScene(onlySecond, 2));
  assert(!dragController.state().windowDragActive);
  assert(dragController.state().focusedWindowId == 2);

  ViewerController slicingController;
  const ViewerUiScene slicingScene = makeScene(makePlan(), 1, true);
  assert(slicingController.publishScene(slicingScene, 1));
  ViewerSessionState slicingSession = initializedSession();
  const auto* vector =
      findHit(slicingScene, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(vector != nullptr);
  const auto vectorPoint = center(*vector);
  const auto vectorPress = press(&slicingController, &slicingSession,
                                 vectorPoint.first, vectorPoint.second);
  assert(vectorPress.consumed && slicingController.state().slicingPaintActive);
  assert(slicingController.publishScene(onlySecond, 2));
  assert(!slicingController.state().slicingPaintActive);
}

void testSourceLassoUndoShortcut() {
  const ViewerUiScene scene = makeScene(makePlan(), 1, true);
  ViewerController controller;
  assert(controller.publishScene(scene, 1));
  ViewerSessionState session = initializedSession();

  const auto control = input(
      &session, ViewerSessionKeyChanged{ViewerSessionKey::Z, true, false,
                                        kViewerSessionModifierControl});
  const auto controlBatch = controller.consume(control, session);
  assert(controlBatch.consumed && controlBatch.count == 1u &&
         controlBatch[0].kind == ViewerControllerCommandKind::SourceLassoUndo &&
         controlBatch[0].windowId == -1);

  const auto controlRepeat = input(
      &session, ViewerSessionKeyChanged{ViewerSessionKey::Z, true, true,
                                        kViewerSessionModifierControl});
  assert(controller.consume(controlRepeat, session).empty());
  const auto controlRelease = input(
      &session, ViewerSessionKeyChanged{ViewerSessionKey::Z, false, false,
                                        kViewerSessionModifierControl});
  assert(controller.consume(controlRelease, session).empty());

  const auto super = input(
      &session, ViewerSessionKeyChanged{ViewerSessionKey::Z, true, false,
                                        kViewerSessionModifierSuper});
  const auto superBatch = controller.consume(super, session);
  assert(superBatch.consumed && superBatch.count == 1u &&
         superBatch[0].kind == ViewerControllerCommandKind::SourceLassoUndo);
  const auto superRelease = input(
      &session, ViewerSessionKeyChanged{ViewerSessionKey::Z, false, false,
                                        kViewerSessionModifierSuper});
  assert(controller.consume(superRelease, session).empty());

  const ViewerSessionKeyChanged rejectedPresses[] = {
      {ViewerSessionKey::A, true, false, kViewerSessionModifierControl},
      {ViewerSessionKey::Z, true, false, 0u},
      {ViewerSessionKey::Z, true, false,
       static_cast<ViewerSessionModifierMask>(
           kViewerSessionModifierControl | kViewerSessionModifierShift)},
      {ViewerSessionKey::Z, true, false,
       static_cast<ViewerSessionModifierMask>(
           kViewerSessionModifierSuper | kViewerSessionModifierAlt)},
  };
  for (const auto& key : rejectedPresses) {
    const auto pressed = input(&session, key);
    const auto batch = controller.consume(pressed, session);
    assert(batch.empty() && !batch.consumed);
    const auto released = input(
        &session, ViewerSessionKeyChanged{key.key, false, false,
                                          key.modifiers});
    assert(controller.consume(released, session).empty());
  }
}

void testSourceSignalRestoreIntent() {
  const ViewerUiScene scene = makeScene(makePlan(), 1, true, true, true);
  ViewerController controller;
  assert(controller.publishScene(scene, 1));
  ViewerSessionState session = initializedSession();
  const auto* restore = findHit(
      scene, 1, ViewerUiControlKind::SourceSignalRestore, 2);
  assert(restore != nullptr);
  const auto point = center(*restore);
  const auto batch = press(&controller, &session, point.first, point.second);
  assert(batch.consumed);
  assert(batch.count == 1u);
  assert(batch[0].kind == ViewerControllerCommandKind::SourceSignalRestore);
  assert(batch[0].windowId == 2);
  assert(batch[0].controlIndex == 2);
  assert(controller.state().focusedWindowId == 1);
}

}  // namespace

int main() {
  testAtomicPublishAndFocusProjection();
  testBackgroundAndToolbarSemantics();
  testPlotFocusAndDrag();
  testDragClampAndRetinaInvariant();
  testControlsPaintAndCancellation();
  testCloseAndDisappearReconcilesCapture();
  testSourceLassoUndoShortcut();
  testSourceSignalRestoreIntent();
  return 0;
}
