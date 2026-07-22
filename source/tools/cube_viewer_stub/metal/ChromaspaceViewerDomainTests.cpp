#include "../../../src/ChromaspaceViewerDomain.h"

#include <cassert>
#include <iostream>

namespace {

using ChromaspaceViewer::LassoPointNorm;
using ChromaspaceViewer::LassoRegionState;
using ChromaspaceViewer::LassoStroke;
using ChromaspaceViewer::PlotWindowDomainState;

void defaults() {
  const PlotWindowDomainState state{};
  assert(state.windowId == 1);
  assert(state.rect.x == 0.0f && state.rect.y == 0.0f);
  assert(state.rect.w == 1.0f && state.rect.h == 1.0f);
  assert(state.camera.qx == 0.0f && state.camera.qy == 0.0f);
  assert(state.camera.qz == 0.0f && state.camera.qw == 1.0f);
  assert(state.camera.distance == 6.0f);
  assert(state.camera.panX == 0.0f && state.camera.panY == 0.03f);
  assert(!state.camera.orthographic);
  assert(state.camera.orthographicView == -1);
  assert(state.viewerLassoStrokes.empty());
  assert(state.viewerLassoRevision == 0);
  assert(state.viewerLassoData.empty());
  assert(state.syncLabel == "Waiting for Resolve");
  assert(state.stableSyncLabel == "Waiting for Resolve");
  assert(state.lastHealthySyncLabelTime == -10.0);
  assert(!state.fitRequested && !state.selected);
  assert(!state.sourceSignalDocked);
  assert(!state.sourceSignalTemporaryLassoSurface);
  assert(state.sourceSignalDockOwnerWindowId == -1);
  assert(state.sourceSignalRestoreRect.w == 1.0f &&
         state.sourceSignalRestoreRect.h == 1.0f);
  assert(state.sourceSignalDockAnimStart == -10.0);
  assert(!state.sourceSignalDockAnimatingToDock);
  assert(!state.slicingDrawerOpen);
  assert(state.slicingDrawerAnimStart == -10.0);

  const LassoRegionState region{};
  assert(region.revision == 0);
  assert(region.empty());
}

void lassoCopiesAreIndependent() {
  PlotWindowDomainState original{};
  original.viewerLassoRevision = 12;
  original.viewerLassoData = "serialized-lasso";
  LassoStroke stroke{};
  stroke.subtract = true;
  stroke.boundsValid = true;
  stroke.minXNorm = 0.1f;
  stroke.maxXNorm = 0.9f;
  stroke.points.push_back({0.2f, 0.3f});
  original.viewerLassoStrokes.push_back(stroke);

  PlotWindowDomainState copy = original;
  original.viewerLassoRevision = 99;
  original.viewerLassoData = "changed";
  original.viewerLassoStrokes[0].points[0] = LassoPointNorm{0.8f, 0.7f};
  original.viewerLassoStrokes.push_back(LassoStroke{});

  assert(copy.viewerLassoRevision == 12);
  assert(copy.viewerLassoData == "serialized-lasso");
  assert(copy.viewerLassoStrokes.size() == 1u);
  assert(copy.viewerLassoStrokes[0].subtract);
  assert(copy.viewerLassoStrokes[0].points.size() == 1u);
  assert(copy.viewerLassoStrokes[0].points[0].xNorm == 0.2f);
  assert(copy.viewerLassoStrokes[0].points[0].yNorm == 0.3f);
}

void classifiers() {
  PlotWindowDomainState cube{};
  cube.viewState.plotModel = ChromaspaceViewer::kPlotModelCube;
  assert(!ChromaspaceViewer::isSourceSignalPlotWindow(cube));
  assert(!ChromaspaceViewer::isTemporarySourceSignalLassoSurface(cube));
  assert(!ChromaspaceViewer::isDockedSourceSignalPlotWindow(cube));
  assert(ChromaspaceViewer::supportsSlicingQuickButton(cube));
  assert(ChromaspaceViewer::participatesInLayoutSlots(cube));

  PlotWindowDomainState gloss = cube;
  gloss.viewState.plotModel = ChromaspaceViewer::kPlotModelGlossView;
  assert(!ChromaspaceViewer::isSourceSignalPlotWindow(gloss));
  assert(!ChromaspaceViewer::supportsSlicingQuickButton(gloss));
  assert(ChromaspaceViewer::participatesInLayoutSlots(gloss));

  PlotWindowDomainState source = cube;
  source.viewState.plotModel = ChromaspaceViewer::kPlotModelSourceSignal;
  assert(ChromaspaceViewer::isSourceSignalPlotWindow(source));
  assert(!ChromaspaceViewer::supportsSlicingQuickButton(source));
  assert(ChromaspaceViewer::participatesInLayoutSlots(source));

  source.sourceSignalTemporaryLassoSurface = true;
  assert(ChromaspaceViewer::isTemporarySourceSignalLassoSurface(source));
  assert(!ChromaspaceViewer::participatesInLayoutSlots(source));
  source.sourceSignalDocked = true;
  assert(ChromaspaceViewer::isDockedSourceSignalPlotWindow(source));
  assert(ChromaspaceViewer::isTemporarySourceSignalLassoSurface(source));

  PlotWindowDomainState dockedOnly = source;
  dockedOnly.sourceSignalTemporaryLassoSurface = false;
  assert(ChromaspaceViewer::isDockedSourceSignalPlotWindow(dockedOnly));
  assert(ChromaspaceViewer::participatesInLayoutSlots(dockedOnly));
}

void syncLabelGrace() {
  const char* healthyLabels[] = {"Live", "Proxy", "Refining"};
  for (const char* healthy : healthyLabels) {
    PlotWindowDomainState state{};
    ChromaspaceViewer::updatePlotWindowSyncLabel(&state, healthy, 10.0);
    assert(state.syncLabel == healthy);
    assert(state.stableSyncLabel == healthy);
    assert(state.lastHealthySyncLabelTime == 10.0);
    ChromaspaceViewer::updatePlotWindowSyncLabel(
        &state, "Waiting for Resolve", 11.15);
    assert(state.syncLabel == healthy);
    assert(state.stableSyncLabel == healthy);
    ChromaspaceViewer::updatePlotWindowSyncLabel(
        &state, "Waiting for Resolve render", 11.150001);
    assert(state.syncLabel == "Waiting for Resolve render");
    assert(state.stableSyncLabel == "Waiting for Resolve render");
  }

  PlotWindowDomainState state{};
  ChromaspaceViewer::updatePlotWindowSyncLabel(&state, "Proxy", 20.0);
  ChromaspaceViewer::updatePlotWindowSyncLabel(&state,
                                               "Waiting for raster",
                                               20.01);
  assert(state.syncLabel == "Waiting for raster");
  assert(state.stableSyncLabel == "Waiting for raster");
}

}  // namespace

int main() {
  defaults();
  lassoCopiesAreIndependent();
  classifiers();
  syncLabelGrace();
  std::cout << "Chromaspace viewer domain tests passed\n";
  return 0;
}

