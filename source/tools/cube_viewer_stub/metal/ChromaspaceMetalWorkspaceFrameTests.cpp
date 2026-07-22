#include "ChromaspaceMetalWorkspaceFrame.h"

#include <cstdlib>
#include <iostream>

namespace {

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::abort();
  }
}

ChromaspaceViewer::ViewerWorkspaceState workspace() {
  ChromaspaceViewer::ViewerWorkspaceState value{};
  value.focusedWindowId = 1;
  value.nextWindowId = 3;
  ChromaspaceViewer::PlotWindowDomainState cube{};
  cube.windowId = 1;
  cube.rect = {0.05f, 0.10f, 0.40f, 0.60f};
  cube.viewState.plotModel = ChromaspaceViewer::kPlotModelCube;
  cube.viewState.stateRevision = 3u;
  cube.selected = true;
  ChromaspaceViewer::PlotWindowDomainState source{};
  source.windowId = 2;
  source.rect = {0.55f, 0.10f, 0.40f, 0.60f};
  source.viewState.plotModel = ChromaspaceViewer::kPlotModelSourceSignal;
  source.viewState.stateRevision = 4u;
  value.windows.push_back(cube);
  value.windows.push_back(source);
  return value;
}

ChromaspaceViewer::ViewerFramePlan plan(
    const ChromaspaceViewer::ViewerWorkspaceState& value) {
  ChromaspaceViewer::ViewerFramePlanRequest request{};
  request.windowWidth = 1000;
  request.windowHeight = 600;
  request.framebufferWidth = 2000;
  request.framebufferHeight = 1200;
  for (const auto& window : value.windows) {
    request.windows.push_back({window.windowId, window.rect,
                               window.viewState.plotModel,
                               window.viewState.stateRevision, true});
  }
  return ChromaspaceViewer::buildViewerFramePlan(request);
}

ChromaspaceMetal::ImportedSourceTexture source() {
  ChromaspaceMetal::ImportedSourceTexture value{};
  value.sourceId = 91u;
  value.senderId = "sender-a";
  value.deviceRegistryId = 8u;
  value.senderGeneration = 2u;
  value.sequence = 7u;
  value.slotIndex = 1u;
  value.slotGeneration = 5u;
  value.readyValue = 6u;
  value.contentHash = 123u;
  value.width = 640;
  value.height = 360;
  value.pixelFormat = 0;
  value.bytesPerRow = 640u * 8u;
  value.byteSize = value.bytesPerRow * 360u;
  value.semantics.sourceWidth = 640u;
  value.semantics.sourceHeight = 360u;
  value.semantics.sampledWidth = 640u;
  value.semantics.sampledHeight = 360u;
  value.semantics.authoritative = true;
  value.semantics.colorPrimaries = "rec709";
  value.semantics.transferFunction = "gamma24";
  return value;
}

void compilesEveryWindowFromOneResidentSource() {
  const auto state = workspace();
  const auto framePlan = plan(state);
  const auto resident = source();
  ChromaspaceMetalWorkspaceFrame::CompileRequest request{};
  request.workspace = &state;
  request.framePlan = &framePlan;
  request.residentSource = &resident;
  request.frameRevision = 9u;
  const auto result =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(result.ready(), "resident workspace frame compiles");
  expect(result.frame.commandCount == 2u, "all planned windows compile");
  expect(result.frame.hasResidentSource &&
             result.frame.residentSource.sourceId == resident.sourceId,
         "one authoritative source is attached to frame");
  expect(result.frame.commands[0].kind ==
             ChromaspaceMetalPlotRenderer::PlotKind::ResidentRaster,
         "cube uses resident raster");
  expect(result.frame.commands[1].kind ==
             ChromaspaceMetalPlotRenderer::PlotKind::SourceSignal,
         "source signal uses source surface");
  expect(result.frame.commands[0].raster.pointCount > 0 &&
             result.frame.commands[0].raster.pointCount <=
                 ChromaspaceMetalPlotRenderer::kMaximumResidentRasterPoints,
         "resident sampling is bounded");
}

void absentSourceScaffoldsEveryWindow() {
  const auto state = workspace();
  const auto framePlan = plan(state);
  ChromaspaceMetalWorkspaceFrame::CompileRequest request{};
  request.workspace = &state;
  request.framePlan = &framePlan;
  request.sourceDiagnostic = "broker-reconnecting";
  const auto result =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(result.ready() && result.frame.commandCount == 2u,
         "unavailable workspace still compiles completely");
  expect(!result.frame.hasResidentSource,
         "unavailable frame carries no source");
  for (std::size_t index = 0; index < result.frame.commandCount; ++index) {
    expect(result.frame.commands[index].kind ==
               ChromaspaceMetalPlotRenderer::PlotKind::Scaffold &&
               result.frame.commands[index].unavailableReason ==
                   "broker-reconnecting",
           "each unavailable window has explicit diagnostic");
  }
}

void liveStateUpdatesOnlyFocusedWindow() {
  auto state = workspace();
  const auto secondBefore = state.windows[1].viewState;
  ChromaspaceViewer::ViewerLiveCommandParams params{};
  params.stateRevision = 22u;
  params.viewerState.plotModel = ChromaspaceViewer::kPlotModelWaveform;
  params.viewerState.stateRevision = 21u;
  params.viewerState.waveformMode = 2;
  int updated = -1;
  std::string error;
  expect(ChromaspaceMetalWorkspaceFrame::applyLiveParamsToFocusedWindow(
             params, &state, &updated, &error),
         "live state applies transactionally");
  expect(updated == 1 &&
             state.windows[0].viewState.plotModel ==
                 ChromaspaceViewer::kPlotModelWaveform &&
             state.windows[0].viewState.stateRevision == 22u,
         "focused window receives complete normalized state");
  expect(state.windows[1].viewState.plotModel == secondBefore.plotModel &&
             state.windows[1].viewState.stateRevision ==
                 secondBefore.stateRevision,
         "unfocused window is preserved");
}

void joinFailureIsAtomic() {
  const auto state = workspace();
  auto framePlan = plan(state);
  framePlan.windows[1].windowId = 77;
  ChromaspaceMetalWorkspaceFrame::CompileRequest request{};
  request.workspace = &state;
  request.framePlan = &framePlan;
  const auto result =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(result.status ==
             ChromaspaceMetalWorkspaceFrame::CompileStatus::MissingWorkspaceWindow &&
             result.rejectedWindowId == 77 && result.frame.commandCount == 0u,
         "join failure publishes no partial frame");
}

ChromaspaceViewer::LassoStroke lassoStroke(bool subtract) {
  ChromaspaceViewer::LassoStroke stroke{};
  stroke.subtract = subtract;
  stroke.points = {{0.1f, 0.2f}, {0.8f, 0.2f}, {0.5f, 0.9f}};
  stroke.boundsValid = true;
  stroke.minXNorm = 0.1f;
  stroke.maxXNorm = 0.8f;
  stroke.minYNorm = 0.2f;
  stroke.maxYNorm = 0.9f;
  return stroke;
}

void resolvesWorkspaceOwnedLassoScope() {
  auto state = workspace();
  state.windows[0].viewState.volumeSliceLassoRegion = true;
  state.sourceLassoSelectionsSynced = true;
  state.sourceLassoHasSelection = true;
  state.sourceLassoGlobalHasSelection = true;
  state.sourceLassoRevision = 1u;
  state.sourceLassoStrokes.push_back(lassoStroke(false));
  auto framePlan = plan(state);
  const auto resident = source();
  ChromaspaceMetalWorkspaceFrame::CompileRequest request{};
  request.workspace = &state;
  request.framePlan = &framePlan;
  request.residentSource = &resident;
  const auto global =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(global.ready() && global.frame.commands[0].raster.lassoEnabled == 1 &&
             global.frame.commands[0].raster.lassoStrokeCount == 1 &&
             global.frame.commands[0].raster.lassoStrokeSubtract[0] == 0,
         "synced selection resolves from global workspace strokes");
  const uint64_t globalContentRevision =
      global.frame.commands[0].contentRevision;
  ++state.sourceLassoRevision;
  const auto revisedGlobal =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(revisedGlobal.ready() &&
             revisedGlobal.frame.commands[0].contentRevision !=
                 globalContentRevision,
         "global lasso revision invalidates resident plot content");

  state.sourceLassoSelectionsSynced = false;
  state.sourceLassoTargetWindowId = 1;
  state.sourceLassoHasSelection = true;
  state.windows[0].viewerLassoRevision = 1u;
  state.windows[0].viewerLassoStrokes = {lassoStroke(true)};
  std::string encoded;
  expect(ChromaspaceViewer::encodeCanonicalViewerLassoData(
             1u, state.windows[0].viewerLassoStrokes, &encoded),
         "target lasso serializes");
  state.windows[0].viewerLassoData = encoded;
  framePlan = plan(state);
  const auto targeted =
      ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(request);
  expect(targeted.ready() &&
             targeted.frame.commands[0].raster.lassoEnabled == 1 &&
             targeted.frame.commands[0].raster.lassoStrokeSubtract[0] == 1 &&
             targeted.frame.commands[0].contentRevision !=
                 revisedGlobal.frame.commands[0].contentRevision,
         "unsynced selection resolves from target window strokes");
}

}  // namespace

int main() {
  compilesEveryWindowFromOneResidentSource();
  absentSourceScaffoldsEveryWindow();
  liveStateUpdatesOnlyFocusedWindow();
  joinFailureIsAtomic();
  resolvesWorkspaceOwnedLassoScope();
  std::cout << "Chromaspace Metal workspace frame tests passed\n";
  return 0;
}
