#include "ChromaspaceViewerWorkspace.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using namespace ChromaspaceViewer;

bool sameStroke(const LassoStroke& a, const LassoStroke& b) {
  if (a.subtract != b.subtract || a.boundsValid != b.boundsValid ||
      a.minXNorm != b.minXNorm || a.maxXNorm != b.maxXNorm ||
      a.minYNorm != b.minYNorm || a.maxYNorm != b.maxYNorm ||
      a.points.size() != b.points.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.points.size(); ++i) {
    if (a.points[i].xNorm != b.points[i].xNorm ||
        a.points[i].yNorm != b.points[i].yNorm) {
      return false;
    }
  }
  return true;
}

bool sameWindow(const PlotWindowDomainState& a, const PlotWindowDomainState& b) {
  if (a.windowId != b.windowId || a.rect.x != b.rect.x || a.rect.y != b.rect.y ||
      a.rect.w != b.rect.w || a.rect.h != b.rect.h ||
      a.viewState.stateRevision != b.viewState.stateRevision ||
      a.viewState.plotModel != b.viewState.plotModel ||
      a.viewState.volumeSliceRed != b.viewState.volumeSliceRed ||
      a.viewState.volumeSliceYellow != b.viewState.volumeSliceYellow ||
      a.viewState.volumeSliceGreen != b.viewState.volumeSliceGreen ||
      a.viewState.volumeSliceCyan != b.viewState.volumeSliceCyan ||
      a.viewState.volumeSliceBlue != b.viewState.volumeSliceBlue ||
      a.viewState.volumeSliceMagenta != b.viewState.volumeSliceMagenta ||
      a.viewState.volumeSliceLassoRegion != b.viewState.volumeSliceLassoRegion ||
      a.camera.qx != b.camera.qx || a.camera.qy != b.camera.qy ||
      a.camera.qz != b.camera.qz || a.camera.qw != b.camera.qw ||
      a.camera.distance != b.camera.distance ||
      a.camera.panX != b.camera.panX || a.camera.panY != b.camera.panY ||
      a.camera.orthographic != b.camera.orthographic ||
      a.camera.orthographicView != b.camera.orthographicView ||
      a.viewerLassoRevision != b.viewerLassoRevision ||
      a.viewerLassoData != b.viewerLassoData || a.syncLabel != b.syncLabel ||
      a.stableSyncLabel != b.stableSyncLabel || a.selected != b.selected ||
      a.sourceSignalDocked != b.sourceSignalDocked ||
      a.sourceSignalTemporaryLassoSurface !=
          b.sourceSignalTemporaryLassoSurface ||
      a.sourceSignalDockOwnerWindowId !=
          b.sourceSignalDockOwnerWindowId ||
      a.sourceSignalRestoreRect.x != b.sourceSignalRestoreRect.x ||
      a.sourceSignalRestoreRect.y != b.sourceSignalRestoreRect.y ||
      a.sourceSignalRestoreRect.w != b.sourceSignalRestoreRect.w ||
      a.sourceSignalRestoreRect.h != b.sourceSignalRestoreRect.h ||
      a.slicingDrawerOpen != b.slicingDrawerOpen ||
      a.viewerLassoStrokes.size() != b.viewerLassoStrokes.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.viewerLassoStrokes.size(); ++i) {
    if (!sameStroke(a.viewerLassoStrokes[i], b.viewerLassoStrokes[i])) return false;
  }
  return true;
}

bool sameWorkspace(const ViewerWorkspaceState& a, const ViewerWorkspaceState& b) {
  if (a.focusedWindowId != b.focusedWindowId || a.nextWindowId != b.nextWindowId ||
      a.layoutPresetSelection != b.layoutPresetSelection ||
      a.layoutPresetBeforeSolo != b.layoutPresetBeforeSolo ||
      a.layoutPresetNameInput != b.layoutPresetNameInput ||
      a.activeToolbarPanel != b.activeToolbarPanel ||
      a.toolbarPanelAnchorX != b.toolbarPanelAnchorX ||
      a.toolbarPanelAnchorY != b.toolbarPanelAnchorY ||
      a.windowDragActive != b.windowDragActive ||
      a.windowDragWindowId != b.windowDragWindowId ||
      a.windowDragMode != b.windowDragMode ||
      a.sourceLassoSubtractMode != b.sourceLassoSubtractMode ||
      a.sourceLassoHasSelection != b.sourceLassoHasSelection ||
      a.sourceLassoGlobalHasSelection != b.sourceLassoGlobalHasSelection ||
      a.sourceLassoSelectionsSynced != b.sourceLassoSelectionsSynced ||
      a.sourceLassoTargetWindowId != b.sourceLassoTargetWindowId ||
      a.sourceLassoSessionActive != b.sourceLassoSessionActive ||
      a.sourceLassoRevision != b.sourceLassoRevision || a.revision != b.revision ||
      a.sourceLassoStrokes.size() != b.sourceLassoStrokes.size() ||
      a.windows.size() != b.windows.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.sourceLassoStrokes.size(); ++i) {
    if (!sameStroke(a.sourceLassoStrokes[i], b.sourceLassoStrokes[i])) return false;
  }
  for (std::size_t i = 0; i < a.windows.size(); ++i) {
    if (!sameWindow(a.windows[i], b.windows[i])) return false;
  }
  return true;
}

ViewerWorkspaceState makeWorkspace(std::size_t count = 2u) {
  ViewerWorkspaceState state{};
  state.windows.reserve(count);
  state.focusedWindowId = 1;
  state.nextWindowId = static_cast<int>(count + 1u);
  state.layoutPresetSelection = "Split 2";
  state.revision = 1u;
  for (std::size_t i = 0; i < count; ++i) {
    PlotWindowDomainState window{};
    window.windowId = static_cast<int>(i + 1u);
    window.rect = i == 0u ? PlotWindowRectNorm{0.0f, 0.0f, 0.6f, 0.8f}
                          : PlotWindowRectNorm{0.4f, 0.2f, 0.5f, 0.6f};
    window.viewState.stateRevision = 1u;
    window.viewState.plotModel = i == 1u ? kPlotModelSourceSignal : kPlotModelCube;
    window.selected = window.windowId == state.focusedWindowId;
    state.windows.push_back(std::move(window));
  }
  assert(validateViewerWorkspaceState(state));
  return state;
}

const PlotWindowDomainState& windowById(const ViewerWorkspaceState& state, int id) {
  for (const auto& window : state.windows) {
    if (window.windowId == id) return window;
  }
  assert(false);
  return state.windows.front();
}

ViewerControllerCommandBatch one(ViewerControllerCommandKind kind,
                                 int windowId = -1) {
  ViewerControllerCommandBatch batch{};
  batch.count = 1u;
  batch.commands[0].kind = kind;
  batch.commands[0].windowId = windowId;
  return batch;
}

void testValidationAndAtomicity() {
  ViewerWorkspaceState state = makeWorkspace();
  const ViewerWorkspaceState before = state;
  ViewerControllerCommandBatch invalid = one(ViewerControllerCommandKind::FocusWindow, 1);
  invalid.commands[0].kind = static_cast<ViewerControllerCommandKind>(255u);
  const auto rejected = reduceViewerWorkspace(&state, invalid);
  assert(!rejected.accepted());
  assert(rejected.status == ViewerWorkspaceReduceStatus::InvalidCommand);
  assert(sameWorkspace(state, before));

  ViewerControllerCommandBatch stale{};
  stale.count = 2u;
  stale.commands[0].kind = ViewerControllerCommandKind::RequestCloseWindow;
  stale.commands[0].windowId = 1;
  stale.commands[1].kind = ViewerControllerCommandKind::FocusWindow;
  stale.commands[1].windowId = 1;
  const auto staleResult = reduceViewerWorkspace(&state, stale);
  assert(!staleResult.accepted());
  assert(state.windows.size() == 2u && state.focusedWindowId == 1);

  state.revision = std::numeric_limits<uint64_t>::max();
  const auto overflow = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::ToolbarMenu));
  assert(!overflow.accepted());
  assert(overflow.status == ViewerWorkspaceReduceStatus::RevisionOverflow);
}

void testFocusCloseAndSourcePolicy() {
  ViewerWorkspaceState state = makeWorkspace();
  auto focus = one(ViewerControllerCommandKind::FocusWindow, 2);
  assert(reduceViewerWorkspace(&state, focus).accepted());
  assert(state.focusedWindowId == 2 && state.windows.back().windowId == 2);
  auto close = one(ViewerControllerCommandKind::RequestCloseWindow, 2);
  const auto closed = reduceViewerWorkspace(&state, close);
  assert(closed.accepted() && state.windows.size() == 1u);
  assert(closed.effects.contains(ViewerWorkspaceEffectKind::ReleaseWindowResources));
  assert(closed.effects.contains(ViewerWorkspaceEffectKind::RefreshResample));
  assert(closed.effects.contains(ViewerWorkspaceEffectKind::PersistSuggested));
  assert(state.focusedWindowId == 1 && state.layoutPresetSelection == "Custom");

  state = makeWorkspace();
  state.sourceLassoSessionActive = true;
  state.sourceLassoTargetWindowId = 1;
  const PlotWindowRectNorm sourceRect = windowById(state, 2).rect;
  const auto sourceClose = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::RequestCloseWindow, 2));
  assert(sourceClose.accepted() && state.windows.size() == 2u);
  assert(sourceClose.effects.contains(ViewerWorkspaceEffectKind::DockSourceSignalForLasso));
  assert(sourceClose.effects.contains(ViewerWorkspaceEffectKind::RefreshReinterpret));
  assert(sourceClose.effects.contains(ViewerWorkspaceEffectKind::ReleaseWindowResources));
  const auto& docked = windowById(state, 2);
  assert(docked.sourceSignalDocked &&
         docked.sourceSignalDockOwnerWindowId == 1 &&
         docked.sourceSignalRestoreRect.x == sourceRect.x &&
         docked.sourceSignalRestoreRect.y == sourceRect.y &&
         docked.sourceSignalRestoreRect.w == sourceRect.w &&
         docked.sourceSignalRestoreRect.h == sourceRect.h &&
         state.focusedWindowId == 1);

  const auto restored = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::SourceSignalRestore, 2));
  assert(restored.accepted() && restored.changed &&
         restored.effects.contains(
             ViewerWorkspaceEffectKind::RefreshReinterpret));
  const auto& visibleSource = windowById(state, 2);
  assert(!visibleSource.sourceSignalDocked &&
         visibleSource.sourceSignalDockOwnerWindowId == -1 &&
         visibleSource.rect.x == sourceRect.x &&
         visibleSource.rect.y == sourceRect.y &&
         visibleSource.rect.w == sourceRect.w &&
         visibleSource.rect.h == sourceRect.h &&
         state.focusedWindowId == 1 &&
         state.sourceLassoTargetWindowId == 1);

  const ViewerWorkspaceState beforeDuplicateRestore = state;
  const auto duplicateRestore = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::SourceSignalRestore, 2));
  assert(duplicateRestore.status == ViewerWorkspaceReduceStatus::InvalidCommand &&
         sameWorkspace(state, beforeDuplicateRestore));

  assert(reduceViewerWorkspace(
             &state,
             one(ViewerControllerCommandKind::RequestCloseWindow, 2))
             .accepted());
  state.revision = std::numeric_limits<uint64_t>::max();
  const ViewerWorkspaceState beforeRestoreOverflow = state;
  const auto restoreOverflow = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::SourceSignalRestore, 2));
  assert(restoreOverflow.status ==
             ViewerWorkspaceReduceStatus::RevisionOverflow &&
         sameWorkspace(state, beforeRestoreOverflow));
}

void testDragPanelsAndSlicing() {
  ViewerWorkspaceState state = makeWorkspace();
  ViewerControllerCommandBatch begin = one(ViewerControllerCommandKind::BeginWindowDrag, 1);
  begin.commands[0].dragMode = PlotWindowDragMode::Move;
  begin.commands[0].rect = state.windows.front().rect;
  begin.commands[0].pointerX = 10.0f;
  begin.commands[0].pointerY = 20.0f;
  assert(reduceViewerWorkspace(&state, begin).accepted());
  ViewerControllerCommandBatch update = one(ViewerControllerCommandKind::UpdateWindowDrag, 1);
  update.commands[0].dragMode = PlotWindowDragMode::Move;
  update.commands[0].rect = {0.1f, 0.1f, 0.5f, 0.6f};
  update.commands[0].pointerX = 30.0f;
  update.commands[0].pointerY = 40.0f;
  const auto updateResult = reduceViewerWorkspace(&state, update);
  assert(updateResult.accepted() && updateResult.effects.contains(
      ViewerWorkspaceEffectKind::SnapPreviewUpdated));
  assert(updateResult.effects.count == 1u);
  assert(updateResult.effects[0].kind == ViewerWorkspaceEffectKind::SnapPreviewUpdated);
  ViewerControllerCommandBatch end = one(ViewerControllerCommandKind::EndWindowDrag, 1);
  end.commands[0].dragMode = PlotWindowDragMode::Move;
  end.commands[0].rect = update.commands[0].rect;
  end.commands[0].pointerX = 50.0f;
  end.commands[0].pointerY = 60.0f;
  const auto endResult = reduceViewerWorkspace(&state, end);
  assert(endResult.accepted() && !state.windowDragActive &&
         endResult.effects.contains(ViewerWorkspaceEffectKind::SnapCommitted));
  const auto& dragged = windowById(state, 1);
  assert(dragged.rect.x == update.commands[0].rect.x &&
         dragged.rect.y == update.commands[0].rect.y &&
         dragged.rect.w == update.commands[0].rect.w &&
         dragged.rect.h == update.commands[0].rect.h);

  auto menu = one(ViewerControllerCommandKind::ToolbarMenu);
  menu.commands[0].pointerX = 4.0f;
  menu.commands[0].pointerY = 5.0f;
  assert(reduceViewerWorkspace(&state, menu).accepted());
  assert(state.activeToolbarPanel == ViewerWorkspaceToolbarPanel::MainMenu);
  assert(state.toolbarPanelAnchorX == 4.0f && state.toolbarPanelAnchorY == 5.0f);
  assert(reduceViewerWorkspace(&state, menu).accepted());
  assert(state.activeToolbarPanel == ViewerWorkspaceToolbarPanel::None);
  auto add = one(ViewerControllerCommandKind::ToolbarAddPlot);
  add.commands[0].pointerX = 100.0f;
  add.commands[0].pointerY = 110.0f;
  assert(reduceViewerWorkspace(&state, add).accepted());
  assert(state.activeToolbarPanel == ViewerWorkspaceToolbarPanel::AddPlot);

  auto set = one(ViewerControllerCommandKind::SetSlicingVector, 1);
  set.commands[0].controlIndex = 2;
  set.commands[0].enabled = true;
  const uint64_t revision = windowById(state, 1).viewState.stateRevision;
  const auto setResult = reduceViewerWorkspace(&state, set);
  assert(setResult.accepted() &&
         windowById(state, 1).viewState.stateRevision == revision + 1u);
  assert(setResult.effects.contains(ViewerWorkspaceEffectKind::RefreshReinterpret));
  assert(slicingVectorEnabled(windowById(state, 1).viewState, 2));
  const auto idempotent = reduceViewerWorkspace(&state, set);
  assert(idempotent.accepted() && !idempotent.changed);
}

void testDrawerAndVectorExclusivity() {
  ViewerWorkspaceState state = makeWorkspace();
  auto drawer = one(ViewerControllerCommandKind::ToggleSlicingDrawer, 1);
  auto opened = reduceViewerWorkspace(&state, drawer);
  assert(opened.accepted() && state.windows[0].slicingDrawerOpen &&
         !state.windows[1].slicingDrawerOpen &&
         opened.effects.contains(ViewerWorkspaceEffectKind::SlicingDrawerChanged));
  assert(opened.effects[0].enabled);
  auto closed = reduceViewerWorkspace(&state, drawer);
  assert(closed.accepted() && !state.windows[0].slicingDrawerOpen &&
         !state.windows[1].slicingDrawerOpen);
  assert(closed.effects.contains(ViewerWorkspaceEffectKind::SlicingDrawerChanged));
  assert(!closed.effects[0].enabled);
  auto other = one(ViewerControllerCommandKind::ToggleSlicingDrawer, 2);
  assert(reduceViewerWorkspace(&state, other).accepted());
  assert(!state.windows[0].slicingDrawerOpen && state.windows[1].slicingDrawerOpen);

  auto solo = one(ViewerControllerCommandKind::SoloSlicingVector, 1);
  solo.commands[0].controlIndex = 2;
  assert(reduceViewerWorkspace(&state, solo).accepted());
  assert(!state.windows[0].viewState.volumeSliceRed &&
         state.windows[0].viewState.volumeSliceGreen &&
         !state.windows[0].viewState.volumeSliceBlue);
  auto all = one(ViewerControllerCommandKind::ToggleAllSlicingVectors, 1);
  assert(reduceViewerWorkspace(&state, all).accepted());
  assert(allSlicingVectorsEnabled(state.windows[0].viewState));
  assert(reduceViewerWorkspace(&state, all).accepted());
  assert(!allSlicingVectorsEnabled(state.windows[0].viewState));

  ViewerControllerCommandBatch mixed{};
  mixed.count = 2u;
  mixed.commands[0] = one(ViewerControllerCommandKind::SetSlicingVector, 1).commands[0];
  mixed.commands[0].controlIndex = 0;
  mixed.commands[0].enabled = true;
  mixed.commands[1] = one(ViewerControllerCommandKind::ToggleSlicingLasso, 1).commands[0];
  const auto mixedResult = reduceViewerWorkspace(&state, mixed);
  assert(mixedResult.accepted());
  std::size_t refreshCount = 0u;
  for (std::size_t i = 0; i < mixedResult.effects.count; ++i) {
    refreshCount += mixedResult.effects[i].kind == ViewerWorkspaceEffectKind::RefreshReinterpret ||
                    mixedResult.effects[i].kind == ViewerWorkspaceEffectKind::RefreshResample;
  }
  assert(refreshCount == 1u &&
         mixedResult.effects.contains(ViewerWorkspaceEffectKind::RefreshResample));
}

void testCapacityAndRevisionBounds() {
  ViewerWorkspaceState maxWindows = makeWorkspace(kViewerWorkspaceMaxWindows);
  assert(validateViewerWorkspaceState(maxWindows));
  PlotWindowDomainState extra = maxWindows.windows.back();
  extra.windowId = static_cast<int>(kViewerWorkspaceMaxWindows + 1u);
  maxWindows.windows.push_back(std::move(extra));
  assert(!validateViewerWorkspaceState(maxWindows));

  ViewerWorkspaceState lassoOverflow = makeWorkspace();
  for (int i = 0; i < 4; ++i) {
    LassoStroke many{};
    many.points.resize(kViewerWorkspaceMaxLassoPointsPerStroke);
    lassoOverflow.windows.front().viewerLassoStrokes.push_back(std::move(many));
  }
  LassoStroke oneMore{};
  oneMore.points.resize(1u);
  lassoOverflow.windows[1].viewerLassoStrokes.push_back(std::move(oneMore));
  assert(!validateViewerWorkspaceState(lassoOverflow));

  ViewerWorkspaceState largeSerialized = makeWorkspace();
  largeSerialized.windows.front().viewerLassoData.assign(1024u, 'x');
  assert(validateViewerWorkspaceState(largeSerialized));
  largeSerialized.windows[1].viewerLassoData.assign(
      kViewerWorkspaceMaxSerializedLassoBytes, 'y');
  assert(!validateViewerWorkspaceState(largeSerialized));

  ViewerWorkspaceState overflow = makeWorkspace();
  overflow.windows.front().viewState.stateRevision =
      std::numeric_limits<uint64_t>::max();
  auto set = one(ViewerControllerCommandKind::SetSlicingVector, 1);
  set.commands[0].controlIndex = 0;
  set.commands[0].enabled = !slicingVectorEnabled(overflow.windows.front().viewState, 0);
  const ViewerWorkspaceState before = overflow;
  const auto result = reduceViewerWorkspace(&overflow, set);
  assert(!result.accepted() &&
         result.status == ViewerWorkspaceReduceStatus::RevisionOverflow &&
         sameWorkspace(overflow, before));
}

void testCloseTargetFallback() {
  ViewerWorkspaceState state = makeWorkspace();
  state.sourceLassoSelectionsSynced = false;
  state.sourceLassoTargetWindowId = 1;
  state.sourceLassoHasSelection = true;
  state.sourceLassoGlobalHasSelection = true;
  const auto result = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::RequestCloseWindow, 1));
  assert(result.accepted() && state.sourceLassoTargetWindowId == -1 &&
         state.sourceLassoHasSelection && validateViewerWorkspaceState(state));
}

void testLassoScopesAndBounds() {
  ViewerWorkspaceState state = makeWorkspace();
  state.sourceLassoSelectionsSynced = true;
  state.sourceLassoHasSelection = true;
  state.sourceLassoGlobalHasSelection = true;
  state.sourceLassoStrokes.push_back({false, {{0.1f, 0.1f}, {0.2f, 0.2f}}, true,
                                      0.1f, 0.2f, 0.1f, 0.2f});
  const uint64_t globalRevision = state.sourceLassoRevision;
  const auto clear = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::SourceLassoClear));
  assert(clear.accepted() && !state.sourceLassoHasSelection &&
         state.sourceLassoStrokes.empty() &&
         state.sourceLassoRevision == globalRevision + 1u &&
         clear.effects[0].windowId == -1);

  state = makeWorkspace();
  state.sourceLassoSelectionsSynced = false;
  state.sourceLassoTargetWindowId = 1;
  state.sourceLassoGlobalHasSelection = true;
  state.sourceLassoStrokes.push_back({false, {{0.7f, 0.1f}, {0.8f, 0.2f}}, true,
                                      0.7f, 0.8f, 0.1f, 0.2f});
  const auto globalStrokesBefore = state.sourceLassoStrokes;
  const uint64_t globalRevisionBefore = state.sourceLassoRevision;
  state.windows.front().viewerLassoRevision = 7u;
  state.windows.front().viewerLassoStrokes.push_back({false, {{0.1f, 0.1f}}, false,
                                                       0.0f, 0.0f, 0.0f, 0.0f});
  state.sourceLassoHasSelection = true;
  const auto unsynced = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::SourceLassoClear));
  assert(unsynced.accepted() && !state.sourceLassoHasSelection &&
         state.sourceLassoStrokes.size() == globalStrokesBefore.size() &&
         sameStroke(state.sourceLassoStrokes.front(), globalStrokesBefore.front()) &&
         state.sourceLassoRevision == globalRevisionBefore &&
         state.windows.front().viewerLassoStrokes.empty() &&
         state.windows.front().viewerLassoRevision == 8u &&
         unsynced.effects[0].windowId == 1);

  auto toggle = one(ViewerControllerCommandKind::ToggleSlicingLasso, 1);
  const auto lasso = reduceViewerWorkspace(&state, toggle);
  assert(lasso.accepted() && lasso.effects.contains(
      ViewerWorkspaceEffectKind::SlicingLassoChanged));
  assert(lasso.effects.contains(ViewerWorkspaceEffectKind::RefreshResample));
  const auto disable = reduceViewerWorkspace(&state, toggle);
  assert(disable.accepted() &&
         disable.effects.contains(ViewerWorkspaceEffectKind::SlicingLassoChanged) &&
         disable.effects.contains(ViewerWorkspaceEffectKind::RefreshReinterpret) &&
         !disable.effects.contains(ViewerWorkspaceEffectKind::RefreshResample));

  ViewerWorkspaceState invalid = makeWorkspace();
  invalid.windows.front().rect.x = std::numeric_limits<float>::quiet_NaN();
  assert(!validateViewerWorkspaceState(invalid));
}

void testCameraUpdateTransaction() {
  ViewerWorkspaceState state = makeWorkspace();
  const uint64_t workspaceRevision = state.revision;
  const uint64_t windowRevision = state.windows.front().viewState.stateRevision;
  CameraState camera = state.windows.front().camera;
  camera.distance = 3.25f;
  camera.panX = 0.2f;
  const auto updated = updateViewerWorkspaceCamera(&state, 1, camera);
  assert(updated.accepted() && updated.changed);
  assert(state.windows.front().camera.distance == 3.25f);
  assert(state.windows.front().camera.panX == 0.2f);
  assert(state.revision == workspaceRevision + 1u);
  assert(state.windows.front().viewState.stateRevision == windowRevision + 1u);
  assert(updated.workspaceRevision == state.revision);
  assert(updated.windowRevision == state.windows.front().viewState.stateRevision);

  const auto unchanged = updateViewerWorkspaceCamera(&state, 1, camera);
  assert(unchanged.accepted() && !unchanged.changed);
  assert(state.revision == workspaceRevision + 1u);

  const ViewerWorkspaceState beforeInvalid = state;
  CameraState invalid = camera;
  invalid.qw = 0.0f;
  assert(updateViewerWorkspaceCamera(&state, 1, invalid).status ==
         ViewerWorkspaceCameraUpdateStatus::InvalidCamera);
  assert(sameWorkspace(state, beforeInvalid));
  assert(updateViewerWorkspaceCamera(&state, 404, camera).status ==
         ViewerWorkspaceCameraUpdateStatus::MissingWindow);
  assert(sameWorkspace(state, beforeInvalid));

  state.revision = std::numeric_limits<uint64_t>::max();
  CameraState overflowCamera = camera;
  overflowCamera.panY += 0.1f;
  const ViewerWorkspaceState beforeOverflow = state;
  assert(updateViewerWorkspaceCamera(&state, 1, overflowCamera).status ==
         ViewerWorkspaceCameraUpdateStatus::RevisionOverflow);
  assert(sameWorkspace(state, beforeOverflow));
}

LassoStroke triangle(bool subtract = false) {
  LassoStroke stroke{};
  stroke.subtract = subtract;
  stroke.points = {{0.1f, 0.2f}, {0.8f, 0.25f}, {0.4f, 0.9f}};
  stroke.boundsValid = true;
  stroke.minXNorm = 0.1f;
  stroke.maxXNorm = 0.8f;
  stroke.minYNorm = 0.2f;
  stroke.maxYNorm = 0.9f;
  return stroke;
}

void testLassoAppendTransactionAndCanonicalData() {
  std::string encoded = "unchanged";
  const LassoStroke additive = triangle();
  assert(encodeCanonicalViewerLassoData(4u, {additive}, &encoded));
  assert(encoded ==
         "v1|4|a,3,0.100000,0.200000,0.800000,0.250000,0.400000,0.900000");

  ViewerWorkspaceState global = makeWorkspace();
  global.sourceLassoSelectionsSynced = true;
  const uint64_t globalWorkspaceRevision = global.revision;
  const auto globalResult =
      appendViewerWorkspaceLassoStroke(&global, additive);
  assert(globalResult.accepted() && globalResult.changed &&
         globalResult.globalSelection && globalResult.targetWindowId == -1);
  assert(global.sourceLassoStrokes.size() == 1u &&
         global.sourceLassoHasSelection &&
         global.sourceLassoGlobalHasSelection &&
         global.sourceLassoRevision == 1u &&
         global.revision == globalWorkspaceRevision + 1u);

  ViewerWorkspaceState targeted = makeWorkspace();
  targeted.sourceLassoSelectionsSynced = false;
  targeted.sourceLassoTargetWindowId = 1;
  const uint64_t targetWindowRevision =
      windowById(targeted, 1).viewState.stateRevision;
  const auto targetResult =
      appendViewerWorkspaceLassoStroke(&targeted, triangle(true));
  const auto& target = windowById(targeted, 1);
  assert(targetResult.accepted() && !targetResult.globalSelection &&
         targetResult.targetWindowId == 1 && targeted.sourceLassoHasSelection);
  assert(target.viewerLassoStrokes.size() == 1u &&
         target.viewerLassoStrokes.front().subtract &&
         target.viewerLassoRevision == 1u &&
         target.viewState.stateRevision == targetWindowRevision + 1u &&
         target.viewerLassoData.rfind("v1|1|s,3,", 0u) == 0u);

  LassoStroke invalid = additive;
  invalid.points.resize(2u);
  const ViewerWorkspaceState beforeInvalid = targeted;
  assert(appendViewerWorkspaceLassoStroke(&targeted, invalid).status ==
         ViewerWorkspaceLassoAppendStatus::InvalidStroke);
  assert(sameWorkspace(targeted, beforeInvalid));

  ViewerWorkspaceState full = makeWorkspace();
  full.sourceLassoSelectionsSynced = true;
  full.sourceLassoHasSelection = true;
  full.sourceLassoGlobalHasSelection = true;
  full.sourceLassoRevision = 1u;
  full.sourceLassoStrokes.assign(kViewerWorkspaceMaxLassoStrokes, additive);
  const ViewerWorkspaceState beforeFull = full;
  assert(appendViewerWorkspaceLassoStroke(&full, additive).status ==
         ViewerWorkspaceLassoAppendStatus::CapacityExceeded);
  assert(sameWorkspace(full, beforeFull));

  ViewerWorkspaceState overflow = makeWorkspace();
  overflow.sourceLassoSelectionsSynced = true;
  overflow.revision = std::numeric_limits<uint64_t>::max();
  const ViewerWorkspaceState beforeOverflow = overflow;
  assert(appendViewerWorkspaceLassoStroke(&overflow, additive).status ==
         ViewerWorkspaceLassoAppendStatus::RevisionOverflow);
  assert(sameWorkspace(overflow, beforeOverflow));
}

void testSourceLassoSessionTransaction() {
  ViewerWorkspaceState state = makeWorkspace(1u);
  state.windows.front().viewState.volumeSliceLassoRegion = true;
  const int focused = state.focusedWindowId;
  const auto enabled =
      updateViewerWorkspaceSourceLassoSession(&state, focused, true);
  assert(enabled.accepted() && enabled.changed &&
         enabled.sourceSurfaceCreated && enabled.sourceSurfaceWindowId == 2);
  assert(state.sourceLassoSessionActive &&
         state.sourceLassoTargetWindowId == focused &&
         state.focusedWindowId == focused && state.windows.size() == 2u);
  const auto& source = windowById(state, enabled.sourceSurfaceWindowId);
  assert(source.viewState.plotModel == kPlotModelSourceSignal &&
         source.sourceSignalTemporaryLassoSurface &&
         source.sourceSignalDockOwnerWindowId == focused &&
         !source.selected && validateViewerWorkspaceState(state));

  const uint64_t stableRevision = state.revision;
  const auto unchanged =
      updateViewerWorkspaceSourceLassoSession(&state, focused, true);
  assert(unchanged.accepted() && !unchanged.changed &&
         state.revision == stableRevision && state.windows.size() == 2u);

  const auto docked = reduceViewerWorkspace(
      &state, one(ViewerControllerCommandKind::RequestCloseWindow,
                  enabled.sourceSurfaceWindowId));
  assert(docked.accepted() &&
         windowById(state, enabled.sourceSurfaceWindowId).sourceSignalDocked);
  state.windows.front().viewState.volumeSliceLassoRegion = false;
  const auto disabled =
      updateViewerWorkspaceSourceLassoSession(&state, focused, false);
  assert(disabled.accepted() && disabled.changed &&
         !state.sourceLassoSessionActive && state.windows.size() == 2u &&
         !windowById(state, enabled.sourceSurfaceWindowId).sourceSignalDocked);

  const ViewerWorkspaceState beforeUnsupported = state;
  assert(updateViewerWorkspaceSourceLassoSession(
             &state, enabled.sourceSurfaceWindowId, true)
             .status == ViewerWorkspaceLassoSessionStatus::UnsupportedOwner);
  assert(sameWorkspace(state, beforeUnsupported));

  ViewerWorkspaceState overflow = makeWorkspace(1u);
  overflow.windows.front().viewState.volumeSliceLassoRegion = true;
  overflow.revision = std::numeric_limits<uint64_t>::max();
  const ViewerWorkspaceState beforeOverflow = overflow;
  assert(updateViewerWorkspaceSourceLassoSession(&overflow, 1, true).status ==
         ViewerWorkspaceLassoSessionStatus::RevisionOverflow);
  assert(sameWorkspace(overflow, beforeOverflow));
}

void testLassoUndoTransaction() {
  ViewerWorkspaceState global = makeWorkspace();
  global.sourceLassoSelectionsSynced = true;
  assert(appendViewerWorkspaceLassoStroke(&global, triangle()).accepted());
  assert(appendViewerWorkspaceLassoStroke(&global, triangle(true)).accepted());
  const uint64_t globalRevision = global.sourceLassoRevision;
  const uint64_t workspaceRevision = global.revision;
  const auto globalUndo = reduceViewerWorkspace(
      &global, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(globalUndo.accepted() && globalUndo.changed &&
         globalUndo.effects.contains(
             ViewerWorkspaceEffectKind::RefreshReinterpret));
  assert(global.sourceLassoStrokes.size() == 1u &&
         !global.sourceLassoStrokes.front().subtract &&
         global.sourceLassoHasSelection &&
         global.sourceLassoGlobalHasSelection &&
         global.sourceLassoRevision == globalRevision + 1u &&
         global.revision == workspaceRevision + 1u);

  const auto lastGlobal = reduceViewerWorkspace(
      &global, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(lastGlobal.accepted() && lastGlobal.changed &&
         global.sourceLassoStrokes.empty() &&
         !global.sourceLassoHasSelection &&
         !global.sourceLassoGlobalHasSelection);
  const uint64_t emptyRevision = global.revision;
  const auto emptyGlobal = reduceViewerWorkspace(
      &global, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(emptyGlobal.accepted() && !emptyGlobal.changed &&
         global.revision == emptyRevision && emptyGlobal.effects.count == 0u);

  ViewerWorkspaceState targeted = makeWorkspace();
  targeted.sourceLassoSelectionsSynced = false;
  targeted.sourceLassoTargetWindowId = 1;
  assert(appendViewerWorkspaceLassoStroke(&targeted, triangle()).accepted());
  assert(appendViewerWorkspaceLassoStroke(&targeted, triangle(true)).accepted());
  const auto targetBefore = windowById(targeted, 1);
  const uint64_t globalLassoRevision = targeted.sourceLassoRevision;
  const auto targetUndo = reduceViewerWorkspace(
      &targeted, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(targetUndo.accepted() && targetUndo.changed);
  const auto& target = windowById(targeted, 1);
  assert(target.viewerLassoStrokes.size() == 1u &&
         !target.viewerLassoStrokes.front().subtract &&
         target.viewerLassoRevision == targetBefore.viewerLassoRevision + 1u &&
         target.viewState.stateRevision ==
             targetBefore.viewState.stateRevision + 1u &&
         target.viewerLassoData.rfind("v1|3|a,3,", 0u) == 0u &&
         targeted.sourceLassoHasSelection &&
         targeted.sourceLassoRevision == globalLassoRevision);

  ViewerWorkspaceState overflow = makeWorkspace();
  overflow.sourceLassoSelectionsSynced = true;
  assert(appendViewerWorkspaceLassoStroke(&overflow, triangle()).accepted());
  overflow.sourceLassoRevision = std::numeric_limits<uint64_t>::max();
  const ViewerWorkspaceState beforeOverflow = overflow;
  const auto overflowResult = reduceViewerWorkspace(
      &overflow, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(overflowResult.status ==
             ViewerWorkspaceReduceStatus::RevisionOverflow &&
         sameWorkspace(overflow, beforeOverflow));

  ViewerWorkspaceState missing = makeWorkspace();
  missing.sourceLassoSelectionsSynced = false;
  missing.sourceLassoTargetWindowId = 99;
  const ViewerWorkspaceState beforeMissing = missing;
  const auto missingResult = reduceViewerWorkspace(
      &missing, one(ViewerControllerCommandKind::SourceLassoUndo));
  assert(missingResult.status == ViewerWorkspaceReduceStatus::InvalidState &&
         sameWorkspace(missing, beforeMissing));
}

}  // namespace

int main() {
  testValidationAndAtomicity();
  testFocusCloseAndSourcePolicy();
  testDragPanelsAndSlicing();
  testDrawerAndVectorExclusivity();
  testCapacityAndRevisionBounds();
  testCloseTargetFallback();
  testLassoScopesAndBounds();
  testCameraUpdateTransaction();
  testLassoAppendTransactionAndCanonicalData();
  testSourceLassoSessionTransaction();
  testLassoUndoTransaction();
  return 0;
}
