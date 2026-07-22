#include "ChromaspaceViewerUiProjection.h"

#include "ChromaspaceViewerState.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace ChromaspaceViewer {
namespace {

template <typename Collection, typename Id>
bool containsDuplicateId(const Collection& values, Id id) noexcept {
  std::size_t count = 0u;
  for (const auto& value : values) {
    if (value.windowId == id && ++count > 1u) return true;
  }
  return false;
}

const PlotWindowDomainState* findWorkspaceWindow(
    const ViewerWorkspaceState& workspace,
    int windowId) noexcept {
  const auto found = std::find_if(
      workspace.windows.begin(), workspace.windows.end(),
      [windowId](const PlotWindowDomainState& window) {
        return window.windowId == windowId;
      });
  return found == workspace.windows.end() ? nullptr : &*found;
}

const ViewerUiProjectionWindowFacts* findWindowFacts(
    const ViewerUiProjectionRequest& request,
    int windowId) noexcept {
  const auto found = std::find_if(
      request.windows.begin(), request.windows.end(),
      [windowId](const ViewerUiProjectionWindowFacts& facts) {
        return facts.windowId == windowId;
      });
  return found == request.windows.end() ? nullptr : &*found;
}

bool framePlanContainsWindow(const ViewerFramePlan& plan,
                             int windowId) noexcept {
  return std::any_of(
      plan.windows.begin(), plan.windows.end(),
      [windowId](const ViewerFramePlanWindow& window) {
        return window.windowId == windowId;
      });
}

bool finiteTitleMetrics(const ViewerUiTitleMetrics& metrics) noexcept {
  return std::isfinite(metrics.titleExtraHeight) &&
         std::isfinite(metrics.fontAscent) &&
         std::isfinite(metrics.fontDescent) &&
         std::isfinite(metrics.textScale) &&
         std::isfinite(metrics.measuredMetadataWidth) &&
         metrics.titleExtraHeight >= 0.0f && metrics.fontAscent >= 0.0f &&
         metrics.fontDescent >= 0.0f && metrics.textScale > 0.0f &&
         metrics.measuredMetadataWidth >= 0.0f;
}

bool validSourceFacts(const ViewerUiProjectionSourceFacts& source) noexcept {
  if (!source.available) return true;
  return source.sourceWidth > 0 && source.sourceHeight > 0 &&
         source.displayWidth > 0 && source.displayHeight > 0;
}

std::string windowTitle(const PlotWindowDomainState& window) {
  std::string title(plotModelLabel(window.viewState.plotModel));
  const std::string& sync =
      !window.syncLabel.empty() ? window.syncLabel : window.stableSyncLabel;
  if (!sync.empty()) title += " | " + sync;
  return title;
}

std::string windowMetadata(
    const PlotWindowDomainState& window,
    const ViewerUiProjectionSourceFacts& source) {
  if (isSourceSignalPlotWindow(window)) {
    std::ostringstream text;
    if (source.available) {
      text << "Source " << sourceDetailLabel(window.viewState.sourceDetailMode)
           << ' ' << source.sourceWidth << 'x' << source.sourceHeight;
      if (source.displayWidth != source.sourceWidth ||
          source.displayHeight != source.sourceHeight) {
        text << " -> " << source.displayWidth << 'x' << source.displayHeight;
      }
    } else {
      text << "No source packet";
    }
    if (window.viewState.sourceSyncSelections) text << " | Synced lasso";
    return text.str();
  }
  if (window.viewState.plotModel == kPlotModelWaveform) {
    return waveformModeLabel(window.viewState.waveformMode);
  }
  if (window.viewState.plotModel == kPlotModelHistogram) {
    return histogramModeLabel(window.viewState.histogramMode);
  }
  if (window.viewState.plotModel == kPlotModelGlossView) return "Gloss";
  return {};
}

ViewerUiProjectionResult failed(ViewerUiProjectionStatus status) noexcept {
  ViewerUiProjectionResult result{};
  result.status = status;
  return result;
}

}  // namespace

ViewerUiProjectionResult projectViewerUi(
    const ViewerUiProjectionRequest& request) noexcept {
  if (!request.framePlan || !request.workspace) {
    return failed(ViewerUiProjectionStatus::InvalidArgument);
  }
  const ViewerFramePlan& plan = *request.framePlan;
  const ViewerWorkspaceState& workspace = *request.workspace;
  if (!plan.ready() || plan.geometry.windowWidth <= 0 ||
      plan.geometry.windowHeight <= 0) {
    return failed(ViewerUiProjectionStatus::InvalidFramePlan);
  }
  if (plan.windows.size() > kViewerWorkspaceMaxWindows ||
      workspace.windows.size() > kViewerWorkspaceMaxWindows ||
      request.windows.size() > kViewerWorkspaceMaxWindows) {
    return failed(ViewerUiProjectionStatus::CapacityExceeded);
  }
  for (const auto& window : plan.windows) {
    if (containsDuplicateId(plan.windows, window.windowId)) {
      return failed(ViewerUiProjectionStatus::DuplicatePlanWindowId);
    }
  }
  for (const auto& window : workspace.windows) {
    if (containsDuplicateId(workspace.windows, window.windowId)) {
      return failed(ViewerUiProjectionStatus::DuplicateWorkspaceWindowId);
    }
  }
  for (const auto& facts : request.windows) {
    if (containsDuplicateId(request.windows, facts.windowId)) {
      return failed(ViewerUiProjectionStatus::DuplicateWindowFactsId);
    }
  }
  if (!validateViewerWorkspaceState(workspace)) {
    return failed(ViewerUiProjectionStatus::InvalidWorkspace);
  }
  if (plan.windows.size() != request.windows.size()) {
    return failed(ViewerUiProjectionStatus::WindowCountMismatch);
  }
  const std::size_t visibleWorkspaceWindowCount =
      static_cast<std::size_t>(std::count_if(
          workspace.windows.begin(), workspace.windows.end(),
          [](const PlotWindowDomainState& window) {
            return !window.sourceSignalDocked;
          }));
  if (plan.windows.size() != visibleWorkspaceWindowCount) {
    return failed(ViewerUiProjectionStatus::WindowCountMismatch);
  }
  for (const auto& window : workspace.windows) {
    if (window.sourceSignalDocked &&
        framePlanContainsWindow(plan, window.windowId)) {
      return failed(ViewerUiProjectionStatus::WindowCountMismatch);
    }
  }
  if (!std::isfinite(request.textScale) || request.textScale <= 0.0f ||
      (request.hasPointer &&
       (!std::isfinite(request.pointerX) ||
        !std::isfinite(request.pointerY))) ||
      !validSourceFacts(request.source)) {
    return failed(ViewerUiProjectionStatus::InvalidFacts);
  }

  try {
    ViewerUiSceneInput projected{};
    projected.toolbar.logicalWidth = plan.geometry.windowWidth;
    projected.toolbar.logicalHeight = plan.geometry.windowHeight;
    projected.toolbar.reservedLeftPixels = plan.geometry.reservedLeftPixels;
    projected.toolbar.visible = request.showWorkspaceButtons;
    projected.toolbar.textScale = request.textScale;
    projected.toolbar.menuActive =
        workspace.activeToolbarPanel == ViewerWorkspaceToolbarPanel::MainMenu;
    projected.toolbar.addPlotActive =
        workspace.activeToolbarPanel == ViewerWorkspaceToolbarPanel::AddPlot;
    projected.toolbar.layoutActive =
        workspace.activeToolbarPanel == ViewerWorkspaceToolbarPanel::LayoutPreset;
    projected.toolbar.layoutIndex = request.layoutIndex;
    projected.toolbar.layoutLabel = workspace.layoutPresetSelection;

    projected.hasPointer = request.hasPointer;
    projected.pointerX = request.pointerX;
    projected.pointerY = request.pointerY;
    projected.focusedWindowId = request.controller.focusedWindowId >= 0
                                    ? request.controller.focusedWindowId
                                    : workspace.focusedWindowId;
    projected.hoveredWindowId = request.controller.hoveredWindowId;
    projected.hoveredDragMode = request.controller.hoveredDragMode;
    projected.activeDragWindowId = request.controller.windowDragActive
                                       ? request.controller.windowDragWindowId
                                       : -1;
    projected.activeDragMode = request.controller.windowDragActive
                                   ? request.controller.windowDragMode
                                   : PlotWindowDragMode::None;
    projected.windows.reserve(plan.windows.size());

    for (const ViewerFramePlanWindow& planned : plan.windows) {
      const PlotWindowDomainState* window =
          findWorkspaceWindow(workspace, planned.windowId);
      if (!window) {
        return failed(ViewerUiProjectionStatus::MissingWorkspaceWindow);
      }
      if (window->viewState.plotModel != planned.plotModel) {
        return failed(ViewerUiProjectionStatus::PlotModelMismatch);
      }
      const ViewerUiProjectionWindowFacts* facts =
          findWindowFacts(request, planned.windowId);
      if (!facts) {
        return failed(ViewerUiProjectionStatus::MissingWindowFacts);
      }
      if (!finiteTitleMetrics(facts->titleMetrics) ||
          !std::isfinite(facts->slicingAnimationProgress) ||
          facts->slicingAnimationProgress < 0.0f ||
          facts->slicingAnimationProgress > 1.0f) {
        return failed(ViewerUiProjectionStatus::InvalidFacts);
      }

      ViewerUiPlotWindowInput input{};
      input.windowId = planned.windowId;
      input.title = windowTitle(*window);
      input.metadata = windowMetadata(*window, request.source);
      input.titleMetrics = facts->titleMetrics;
      input.closable = plan.windows.size() > 1u;

      input.slicing.visible = request.showSliceButtonInPlotWindows &&
                              supportsSlicingQuickButton(*window);
      input.slicing.drawerOpen =
          input.slicing.visible && window->slicingDrawerOpen;
      for (int index = 0; index < kViewerWorkspaceSlicingVectorCount; ++index) {
        input.slicing.vectors[static_cast<std::size_t>(index)] =
            slicingVectorEnabled(window->viewState, index);
      }
      input.slicing.lassoActive =
          window->viewState.volumeSliceLassoRegion;
      input.slicing.active = input.slicing.lassoActive ||
          std::any_of(input.slicing.vectors.begin(), input.slicing.vectors.end(),
                      [](bool enabled) { return enabled; });
      input.slicing.animationProgress = facts->slicingAnimationProgress;

      input.sourceLasso.visible = workspace.sourceLassoSessionActive &&
                                  isSourceSignalPlotWindow(*window);
      input.sourceLasso.subtract = workspace.sourceLassoSubtractMode;
      input.sourceLasso.hasSelection =
          input.sourceLasso.visible &&
          (workspace.sourceLassoHasSelection ||
           workspace.sourceLassoGlobalHasSelection);
      for (const auto& candidate : workspace.windows) {
        if (candidate.sourceSignalDocked &&
            candidate.sourceSignalDockOwnerWindowId == window->windowId) {
          input.sourceSignalRestoreWindowIds.push_back(candidate.windowId);
        }
      }
      projected.windows.push_back(std::move(input));
    }

    ViewerUiProjectionResult result{};
    result.status = ViewerUiProjectionStatus::Ready;
    result.input = std::move(projected);
    return result;
  } catch (...) {
    return failed(ViewerUiProjectionStatus::AllocationFailure);
  }
}

}  // namespace ChromaspaceViewer
