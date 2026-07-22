#include "ChromaspaceViewerUiProjection.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ChromaspaceViewer;

ViewerWorkspaceState makeWorkspace(const std::vector<int>& ids,
                                   const std::vector<int>& models) {
  assert(!ids.empty() && ids.size() == models.size());
  ViewerWorkspaceState workspace{};
  workspace.focusedWindowId = ids.front();
  workspace.layoutPresetSelection = "Data-driven";
  workspace.nextWindowId = *std::max_element(ids.begin(), ids.end()) + 1;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    PlotWindowDomainState window{};
    window.windowId = ids[index];
    window.rect = {0.02f * static_cast<float>(index),
                   0.01f * static_cast<float>(index), 0.42f, 0.48f};
    window.viewState.plotModel = models[index];
    window.viewState.stateRevision = static_cast<uint64_t>(index + 1u);
    window.syncLabel = "Live";
    window.stableSyncLabel = "Proxy";
    window.selected = index == 0u;
    workspace.windows.push_back(std::move(window));
  }
  assert(validateViewerWorkspaceState(workspace));
  return workspace;
}

ViewerFramePlan makePlan(const ViewerWorkspaceState& workspace,
                         const std::vector<int>& order) {
  ViewerFramePlanRequest request{};
  request.windowWidth = 900;
  request.windowHeight = 600;
  request.framebufferWidth = 1800;
  request.framebufferHeight = 1200;
  request.reservedLeftPixels = 14.0f;
  for (int id : order) {
    const auto found = std::find_if(
        workspace.windows.begin(), workspace.windows.end(),
        [id](const PlotWindowDomainState& window) { return window.windowId == id; });
    assert(found != workspace.windows.end());
    request.windows.push_back({found->windowId, found->rect,
                               found->viewState.plotModel,
                               found->viewState.stateRevision, true});
  }
  const ViewerFramePlan plan = buildViewerFramePlan(request);
  assert(plan.ready());
  return plan;
}

std::vector<ViewerUiProjectionWindowFacts> makeFacts(
    const std::vector<int>& ids) {
  std::vector<ViewerUiProjectionWindowFacts> facts;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    ViewerUiProjectionWindowFacts item{};
    item.windowId = ids[index];
    item.titleMetrics.titleExtraHeight = 2.0f;
    item.titleMetrics.textScale = 0.92f;
    item.titleMetrics.fontAvailable = false;
    item.slicingAnimationProgress =
        static_cast<float>(index) / static_cast<float>(ids.size());
    facts.push_back(item);
  }
  return facts;
}

ViewerUiProjectionRequest makeRequest(const ViewerFramePlan& plan,
                                      const ViewerWorkspaceState& workspace) {
  ViewerUiProjectionRequest request{};
  request.framePlan = &plan;
  request.workspace = &workspace;
  request.windows = makeFacts(
      [&]() {
        std::vector<int> ids;
        for (auto it = workspace.windows.rbegin();
             it != workspace.windows.rend(); ++it) {
          ids.push_back(it->windowId);
        }
        return ids;
      }());
  return request;
}

const ViewerUiPlotWindowInput& projectedWindow(
    const ViewerUiProjectionResult& result,
    int id) {
  const auto found = std::find_if(
      result.input.windows.begin(), result.input.windows.end(),
      [id](const ViewerUiPlotWindowInput& window) {
        return window.windowId == id;
      });
  assert(found != result.input.windows.end());
  return *found;
}

void testIdJoinOrderAndAllTitles() {
  std::vector<int> ids;
  std::vector<int> models;
  for (int model = 0; model < kPlotModelCount; ++model) {
    ids.push_back(101 + model * 7);
    models.push_back(model);
  }
  ViewerWorkspaceState workspace = makeWorkspace(ids, models);
  workspace.windows[3].syncLabel.clear();
  const std::vector<int> order(ids.rbegin(), ids.rend());
  const ViewerFramePlan plan = makePlan(workspace, order);
  ViewerUiProjectionRequest request = makeRequest(plan, workspace);
  const ViewerUiProjectionResult result = projectViewerUi(request);
  assert(result.ready());
  assert(result.input.windows.size() == order.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    assert(result.input.windows[index].windowId == order[index]);
    const auto workspaceWindow = std::find_if(
        workspace.windows.begin(), workspace.windows.end(),
        [&](const PlotWindowDomainState& window) {
          return window.windowId == order[index];
        });
    const std::string sync = workspaceWindow->syncLabel.empty()
                                 ? "Proxy"
                                 : "Live";
    assert(result.input.windows[index].title ==
           std::string(plotModelLabel(workspaceWindow->viewState.plotModel)) +
               " | " + sync);
  }
}

void testMetadataCapabilitiesAndCloseness() {
  const std::vector<int> ids = {31, 7, 88, 52, 19};
  const std::vector<int> models = {kPlotModelSourceSignal, kPlotModelWaveform,
                                   kPlotModelHistogram, kPlotModelGlossView,
                                   kPlotModelCube};
  ViewerWorkspaceState workspace = makeWorkspace(ids, models);
  workspace.windows[0].viewState.sourceDetailMode = 3;
  workspace.windows[0].viewState.sourceSyncSelections = true;
  workspace.windows[1].viewState.waveformMode = 1;
  workspace.windows[2].viewState.histogramMode = 1;
  workspace.windows[4].viewState.volumeSliceRed = true;
  workspace.windows[4].viewState.volumeSliceLassoRegion = true;
  workspace.windows[4].slicingDrawerOpen = true;
  workspace.sourceLassoSessionActive = true;
  workspace.sourceLassoSubtractMode = true;
  workspace.sourceLassoGlobalHasSelection = true;
  const ViewerFramePlan plan = makePlan(workspace, {19, 31, 52, 7, 88});
  ViewerUiProjectionRequest request = makeRequest(plan, workspace);
  request.source = {true, 3840, 2160, 1920, 1080};
  request.showSliceButtonInPlotWindows = true;
  const ViewerUiProjectionResult result = projectViewerUi(request);
  assert(result.ready());
  assert(projectedWindow(result, 31).metadata ==
         "Source Quality 3840x2160 -> 1920x1080 | Synced lasso");
  assert(projectedWindow(result, 7).metadata == "RGB Parade");
  assert(projectedWindow(result, 88).metadata == "Luma");
  assert(projectedWindow(result, 52).metadata == "Gloss");
  assert(projectedWindow(result, 19).metadata.empty());
  assert(projectedWindow(result, 19).slicing.visible);
  assert(projectedWindow(result, 19).slicing.drawerOpen);
  assert(projectedWindow(result, 19).slicing.active);
  assert(projectedWindow(result, 19).slicing.vectors[0]);
  assert(projectedWindow(result, 19).slicing.lassoActive);
  assert(!projectedWindow(result, 31).slicing.visible);
  assert(!projectedWindow(result, 52).slicing.visible);
  assert(projectedWindow(result, 31).sourceLasso.visible);
  assert(projectedWindow(result, 31).sourceLasso.subtract);
  assert(projectedWindow(result, 31).sourceLasso.hasSelection);
  assert(!projectedWindow(result, 19).sourceLasso.visible);
  for (const auto& window : result.input.windows) assert(window.closable);

  request.source = {};
  const auto unavailable = projectViewerUi(request);
  assert(unavailable.ready());
  assert(projectedWindow(unavailable, 31).metadata ==
         "No source packet | Synced lasso");

  ViewerWorkspaceState single = makeWorkspace({44}, {kPlotModelCube});
  const ViewerFramePlan singlePlan = makePlan(single, {44});
  ViewerUiProjectionRequest singleRequest = makeRequest(singlePlan, single);
  const auto singleResult = projectViewerUi(singleRequest);
  assert(singleResult.ready());
  assert(!singleResult.input.windows.front().closable);
}

void testToolbarControllerAndPointerPassThrough() {
  ViewerWorkspaceState workspace = makeWorkspace({12, 91},
      {kPlotModelCube, kPlotModelSourceSignal});
  workspace.activeToolbarPanel = ViewerWorkspaceToolbarPanel::LayoutPreset;
  const ViewerFramePlan plan = makePlan(workspace, {91, 12});
  ViewerUiProjectionRequest request = makeRequest(plan, workspace);
  request.showWorkspaceButtons = false;
  request.textScale = 1.25f;
  request.layoutIndex = 6;
  request.hasPointer = true;
  request.pointerX = 123.0f;
  request.pointerY = 234.0f;
  request.controller.focusedWindowId = 91;
  request.controller.hoveredWindowId = 12;
  request.controller.hoveredDragMode = PlotWindowDragMode::ResizeBottomRight;
  request.controller.windowDragActive = true;
  request.controller.windowDragWindowId = 91;
  request.controller.windowDragMode = PlotWindowDragMode::Move;
  const auto result = projectViewerUi(request);
  assert(result.ready());
  assert(!result.input.toolbar.visible);
  assert(result.input.toolbar.logicalWidth == 900);
  assert(result.input.toolbar.logicalHeight == 600);
  assert(result.input.toolbar.reservedLeftPixels == 14.0f);
  assert(result.input.toolbar.layoutActive);
  assert(result.input.toolbar.layoutIndex == 6);
  assert(result.input.toolbar.layoutLabel == "Data-driven");
  assert(result.input.toolbar.textScale == 1.25f);
  assert(result.input.hasPointer && result.input.pointerX == 123.0f &&
         result.input.pointerY == 234.0f);
  assert(result.input.focusedWindowId == 91);
  assert(result.input.hoveredWindowId == 12);
  assert(result.input.hoveredDragMode ==
         PlotWindowDragMode::ResizeBottomRight);
  assert(result.input.activeDragWindowId == 91);
  assert(result.input.activeDragMode == PlotWindowDragMode::Move);
}

void testDockedSourceProjection() {
  ViewerWorkspaceState workspace = makeWorkspace(
      {12, 91}, {kPlotModelCube, kPlotModelSourceSignal});
  workspace.sourceLassoSessionActive = true;
  workspace.sourceLassoTargetWindowId = 12;
  workspace.windows[1].sourceSignalDocked = true;
  workspace.windows[1].sourceSignalDockOwnerWindowId = 12;
  workspace.windows[1].sourceSignalRestoreRect = workspace.windows[1].rect;
  assert(validateViewerWorkspaceState(workspace));

  const ViewerFramePlan plan = makePlan(workspace, {12});
  ViewerUiProjectionRequest request{};
  request.framePlan = &plan;
  request.workspace = &workspace;
  request.windows = makeFacts({12});
  const auto result = projectViewerUi(request);
  assert(result.ready() && result.input.windows.size() == 1u);
  const auto& owner = projectedWindow(result, 12);
  assert(owner.sourceSignalRestoreWindowIds.size() == 1u &&
         owner.sourceSignalRestoreWindowIds.front() == 91);

  const ViewerFramePlan invalidPlan = makePlan(workspace, {12, 91});
  request.framePlan = &invalidPlan;
  request.windows = makeFacts({12, 91});
  const auto invalid = projectViewerUi(request);
  assert(invalid.status == ViewerUiProjectionStatus::WindowCountMismatch &&
         invalid.input.windows.empty());
}

void assertAtomicFailure(const ViewerUiProjectionResult& result,
                         ViewerUiProjectionStatus status) {
  assert(result.status == status);
  assert(result.input.windows.empty());
  assert(result.input.toolbar.layoutLabel.empty());
}

void testAtomicRejectionsAndBounds() {
  ViewerWorkspaceState workspace = makeWorkspace({2, 9},
      {kPlotModelCube, kPlotModelSourceSignal});
  ViewerFramePlan plan = makePlan(workspace, {9, 2});
  ViewerUiProjectionRequest request = makeRequest(plan, workspace);

  ViewerFramePlan duplicatePlan = plan;
  duplicatePlan.windows[1].windowId = duplicatePlan.windows[0].windowId;
  request.framePlan = &duplicatePlan;
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::DuplicatePlanWindowId);

  request.framePlan = &plan;
  ViewerWorkspaceState duplicateWorkspace = workspace;
  duplicateWorkspace.windows[1].windowId = duplicateWorkspace.windows[0].windowId;
  request.workspace = &duplicateWorkspace;
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::DuplicateWorkspaceWindowId);

  request.workspace = &workspace;
  auto duplicateFacts = request.windows;
  duplicateFacts[1].windowId = duplicateFacts[0].windowId;
  request.windows = duplicateFacts;
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::DuplicateWindowFactsId);

  request = makeRequest(plan, workspace);
  request.windows.pop_back();
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::WindowCountMismatch);

  request = makeRequest(plan, workspace);
  request.windows[0].windowId = 77;
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::MissingWindowFacts);

  ViewerWorkspaceState missingWorkspace = workspace;
  missingWorkspace.windows[1].windowId = 10;
  missingWorkspace.nextWindowId = 11;
  request = makeRequest(plan, missingWorkspace);
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::MissingWorkspaceWindow);

  ViewerWorkspaceState mismatchWorkspace = workspace;
  mismatchWorkspace.windows[0].viewState.plotModel = kPlotModelHsl;
  request = makeRequest(plan, mismatchWorkspace);
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::PlotModelMismatch);

  request = makeRequest(plan, workspace);
  request.windows[0].slicingAnimationProgress =
      std::numeric_limits<float>::quiet_NaN();
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::InvalidFacts);

  ViewerFramePlan oversized{};
  oversized.windows.resize(kViewerWorkspaceMaxWindows + 1u);
  request.framePlan = &oversized;
  assertAtomicFailure(projectViewerUi(request),
                      ViewerUiProjectionStatus::CapacityExceeded);
}

}  // namespace

int main() {
  testIdJoinOrderAndAllTitles();
  testMetadataCapabilitiesAndCloseness();
  testToolbarControllerAndPointerPassThrough();
  testDockedSourceProjection();
  testAtomicRejectionsAndBounds();
  return 0;
}
