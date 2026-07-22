#include "ChromaspaceViewerDomain.h"

namespace ChromaspaceViewer {

bool isSourceSignalPlotWindow(const PlotWindowDomainState& window) noexcept {
  return window.viewState.plotModel == kPlotModelSourceSignal;
}

bool isTemporarySourceSignalLassoSurface(
    const PlotWindowDomainState& window) noexcept {
  return isSourceSignalPlotWindow(window) &&
         window.sourceSignalTemporaryLassoSurface;
}

bool isDockedSourceSignalPlotWindow(
    const PlotWindowDomainState& window) noexcept {
  return isSourceSignalPlotWindow(window) && window.sourceSignalDocked;
}

bool supportsSlicingQuickButton(
    const PlotWindowDomainState& window) noexcept {
  return !isSourceSignalPlotWindow(window) &&
         window.viewState.plotModel != kPlotModelGlossView;
}

bool participatesInLayoutSlots(
    const PlotWindowDomainState& window) noexcept {
  return !isTemporarySourceSignalLassoSurface(window);
}

void updatePlotWindowSyncLabel(PlotWindowDomainState* window,
                               const std::string& nextLabel,
                               double nowSeconds) {
  if (!window) return;
  constexpr double kSyncDowngradeGraceSeconds = 1.15;
  window->syncLabel = nextLabel;
  const bool healthy = nextLabel == "Live" || nextLabel == "Proxy" ||
                       nextLabel == "Refining";
  if (healthy) {
    window->stableSyncLabel = nextLabel;
    window->lastHealthySyncLabelTime = nowSeconds;
    return;
  }
  const bool resolveWait = nextLabel == "Waiting for Resolve" ||
                           nextLabel == "Waiting for Resolve render";
  const bool stableHealthy = window->stableSyncLabel == "Live" ||
                             window->stableSyncLabel == "Proxy" ||
                             window->stableSyncLabel == "Refining";
  if (resolveWait && window->lastHealthySyncLabelTime >= 0.0 &&
      nowSeconds - window->lastHealthySyncLabelTime <=
          kSyncDowngradeGraceSeconds + 1e-9 &&
      stableHealthy) {
    window->syncLabel = window->stableSyncLabel;
    return;
  }
  window->stableSyncLabel = nextLabel;
}

}  // namespace ChromaspaceViewer
