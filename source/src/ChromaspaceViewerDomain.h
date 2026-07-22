#pragma once

#include "ChromaspaceViewerLayout.h"
#include "ChromaspaceViewerState.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceViewer {

struct CameraState {
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  float qw = 1.0f;
  float distance = 6.0f;
  float panX = 0.0f;
  float panY = 0.03f;
  bool orthographic = false;
  int orthographicView = -1;
};

struct LassoPointNorm {
  float xNorm = 0.0f;
  float yNorm = 0.0f;
};

struct LassoStroke {
  bool subtract = false;
  std::vector<LassoPointNorm> points;
  bool boundsValid = false;
  float minXNorm = 0.0f;
  float maxXNorm = 0.0f;
  float minYNorm = 0.0f;
  float maxYNorm = 0.0f;
};

struct LassoRegionState {
  uint64_t revision = 0;
  std::vector<LassoStroke> strokes;

  bool empty() const { return strokes.empty(); }
};

struct PlotWindowDomainState {
  int windowId = 1;
  PlotWindowRectNorm rect{};
  ViewerRuntimeState viewState{};
  CameraState camera{};
  std::vector<LassoStroke> viewerLassoStrokes;
  uint64_t viewerLassoRevision = 0;
  std::string viewerLassoData;
  std::string syncLabel = "Waiting for Resolve";
  std::string stableSyncLabel = "Waiting for Resolve";
  double lastHealthySyncLabelTime = -10.0;
  bool fitRequested = false;
  bool selected = false;
  bool sourceSignalDocked = false;
  bool sourceSignalTemporaryLassoSurface = false;
  int sourceSignalDockOwnerWindowId = -1;
  PlotWindowRectNorm sourceSignalRestoreRect{};
  double sourceSignalDockAnimStart = -10.0;
  bool sourceSignalDockAnimatingToDock = false;
  bool slicingDrawerOpen = false;
  double slicingDrawerAnimStart = -10.0;
};

bool isSourceSignalPlotWindow(const PlotWindowDomainState& window) noexcept;
bool isTemporarySourceSignalLassoSurface(
    const PlotWindowDomainState& window) noexcept;
bool isDockedSourceSignalPlotWindow(
    const PlotWindowDomainState& window) noexcept;
bool supportsSlicingQuickButton(
    const PlotWindowDomainState& window) noexcept;
bool participatesInLayoutSlots(
    const PlotWindowDomainState& window) noexcept;

void updatePlotWindowSyncLabel(PlotWindowDomainState* window,
                               const std::string& nextLabel,
                               double nowSeconds);

}  // namespace ChromaspaceViewer
