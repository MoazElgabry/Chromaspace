#include "ChromaspaceMetalPlotCompiler.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

using ChromaspaceMetalPlotCompiler::GlossPresentation;
using ChromaspaceMetalPlotCompiler::Input;
using ChromaspaceMetalPlotRenderer::FrameRequest;
using ChromaspaceMetalPlotRenderer::PlotKind;

ChromaspaceMetal::RasterSourceRequest normalizedRaster() {
  ChromaspaceMetal::RasterSourceRequest raster{};
  raster.pointCount = 64;
  raster.basePointCount = 64;
  raster.sourceWidth = 8;
  raster.sourceHeight = 8;
  raster.sampleStride = 1;
  raster.sampleCountX = 8;
  raster.sourceAspect = 1.0f;
  return raster;
}

Input baseInput(int plotModel) {
  Input input{};
  input.windowId = 11;
  input.plotModel = plotModel;
  input.destination = {3.0f, 5.0f, 320.0f, 240.0f};
  input.targetWidth = 320;
  input.targetHeight = 240;
  input.viewRevision = 7u;
  input.hasRasterRequest = plotModel != ChromaspaceViewer::kPlotModelSourceSignal;
  input.raster = normalizedRaster();
  input.hasCameraMatrices = true;
  input.modelView[0] = input.modelView[5] = input.modelView[10] =
      input.modelView[15] = 1.0f;
  input.projection[0] = input.projection[5] = input.projection[10] =
      input.projection[15] = 1.0f;
  return input;
}

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::abort();
  }
}

void sourceUnavailableEmitsScaffold() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelCube);
  input.sourceAvailable = false;
  input.hasRasterRequest = false;
  input.scope.waveformMode = 99;
  input.hasCameraMatrices = false;
  input.unavailableReason = "broker-source-not-ready";
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "unavailable source compiles scaffold");
  const auto& command = frame.commands[0];
  expect(command.plotModel == ChromaspaceViewer::kPlotModelCube &&
             command.kind == PlotKind::Scaffold &&
             command.unavailableReason == input.unavailableReason &&
             command.vectorVertexCount > 0u,
         "scaffold reason and geometry");
}

void unrelatedStateDoesNotBlockSourceSignal() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelSourceSignal);
  input.hasRasterRequest = false;
  input.scope.waveformMode = 99;
  input.scope.histogramMode = 99;
  input.glossDerivationHash = 0u;
  input.glossGridWidth = -1;
  input.hasCameraMatrices = false;
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "unrelated state does not block source signal");
  expect(frame.commands[0].kind == PlotKind::SourceSignal,
         "source signal remains selected");
}

void sourceSignalDoesNotRequireRaster() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelSourceSignal);
  input.hasRasterRequest = false;
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "source signal compiles without raster");
  expect(frame.commands[0].kind == PlotKind::SourceSignal &&
             frame.vectorVertexArena.empty(),
         "source signal has no raster or guide vectors");
}

void histogramSemanticsAreNormalized() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelHistogram);
  input.scope.histogramMode = 1;
  input.scope.scopeRangeMode = 1;
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "manual histogram compiles");
  const auto& manual = frame.commands[0].histogram;
  expect(manual.scopeMode == 1 && manual.lumaMethod == 0 &&
             manual.rangeMin == 0.0f && manual.invRange == 0.25f &&
             manual.useGpuAutoRange == 0 && manual.autoRange.waveform == 0 &&
             manual.autoRange.includeRed == 0 &&
             manual.autoRange.includeGreen == 0 &&
             manual.autoRange.includeBlue == 0 &&
             manual.autoRange.includeLuma == 1,
         "histogram luma/manual range invariant");

  FrameRequest autoFrame;
  input.windowId = 12;
  input.scope.histogramMode = 0;
  input.scope.scopeRangeMode = 2;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &autoFrame, &error),
         "auto histogram compiles");
  const auto& automatic = autoFrame.commands[0].histogram;
  expect(automatic.useGpuAutoRange == 1 && automatic.autoRange.waveform == 0 &&
             automatic.autoRange.includeRed == 1 &&
             automatic.autoRange.includeGreen == 1 &&
             automatic.autoRange.includeBlue == 1 &&
             automatic.autoRange.includeLuma == 0,
         "histogram RGB auto range invariant");
}

void waveformSemanticsAreNormalized() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelWaveform);
  input.scope.waveformMode = 2;
  input.scope.waveformLumaMethod = 3;
  input.scope.waveformChannelRed = 1;
  input.scope.waveformChannelGreen = 0;
  input.scope.waveformChannelBlue = 1;
  input.scope.scopeRangeMode = 2;
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "waveform compiles");
  const auto& waveform = frame.commands[0].waveform;
  expect(waveform.scopeMode == 2 && waveform.lumaMethod == 3 &&
             waveform.includeRed == 0 && waveform.includeGreen == 0 &&
             waveform.includeBlue == 0 && waveform.includeLuma == 1 &&
             waveform.useGpuAutoRange == 1 &&
             waveform.autoRange.waveform == 1 &&
             waveform.autoRange.includeLuma == 1,
         "waveform luma/auto range invariant");
}

void residentFieldsAndOrientationAreStable() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelChromaticity);
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "resident chromaticity compiles");
  const auto& command = frame.commands[0];
  expect(command.kind == PlotKind::ResidentRaster &&
             command.point.pointCount == input.raster.pointCount &&
             command.point.modelView[0] == 1.0f && command.vectorVertexCount > 6u,
         "resident request/matrices/guides");
  const auto& vertices = frame.vectorVertexArena;
  expect(vertices[0].x == 0.0f && vertices[0].y == 0.0f &&
             vertices[1].x == static_cast<float>(input.targetWidth) &&
             vertices[2].y == 2.0f,
         "primer ordering");

  // The first asymmetric locus segment must begin at (.12w,.18h) and end at
  // (.22w,.36h), proving the positive chart-space Y orientation and ordering.
  const float x0 = 0.12f * input.targetWidth;
  const float y0 = 0.18f * input.targetHeight;
  const float x1 = 0.22f * input.targetWidth;
  const float y1 = 0.36f * input.targetHeight;
  bool foundStart = false;
  bool foundEndAfterStart = false;
  for (const auto& vertex : vertices) {
    const bool start = std::fabs(vertex.x - x0) < 2.0f &&
                       std::fabs(vertex.y - y0) < 2.0f;
    const bool end = std::fabs(vertex.x - x1) < 2.0f &&
                     std::fabs(vertex.y - y1) < 2.0f;
    if (start) foundStart = true;
    if (foundStart && end) {
      foundEndAfterStart = true;
      break;
    }
  }
  expect(foundStart && foundEndAfterStart, "ordered asymmetric locus orientation");
}

void glossKeepsHashAndViewRevisionSeparate() {
  FrameRequest fieldFrame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelGlossView);
  input.glossGridWidth = 73;
  input.glossGridHeight = 41;
  input.glossDerivationHash = 0x1234u;
  input.viewRevision = 23u;
  input.hasCameraMatrices = false;
  std::string error;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &fieldFrame, &error),
         "Gloss 2D compiles");
  const auto& field = fieldFrame.commands[0];
  expect(field.kind == PlotKind::GlossField2D && field.glossField.gridWidth == 73 &&
             field.glossField.gridHeight == 41 &&
             field.glossDerivationHash == 0x1234u && field.viewRevision == 23u,
         "Gloss 2D dimensions/hash/revision");

  FrameRequest projectionFrame;
  input.windowId = 12;
  input.viewRevision = 24u;
  input.glossPresentation = GlossPresentation::Projection3D;
  input.hasCameraMatrices = true;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &projectionFrame, &error),
         "Gloss 3D compiles");
  const auto& projection = projectionFrame.commands[0];
  expect(projection.kind == PlotKind::GlossProjection3D &&
             projection.glossDerivationHash == field.glossDerivationHash &&
             projection.viewRevision != field.viewRevision &&
             projection.glossProjectionSurface.sourceAspect == input.raster.sourceAspect,
         "Gloss 2D/3D hash and view separation");
}

void glossPresentationIsStrictlyValidated() {
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelGlossView);
  input.glossPresentation = static_cast<GlossPresentation>(99);
  std::string error;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "invalid Gloss presentation rejected");
}

void missingCameraUsesPerWindowScaffold() {
  std::string error;
  FrameRequest residentFrame;
  Input resident = baseInput(ChromaspaceViewer::kPlotModelCube);
  resident.hasCameraMatrices = false;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(
             resident, &residentFrame, &error),
         "resident missing camera scaffolds");
  expect(residentFrame.commands[0].kind == PlotKind::Scaffold &&
             residentFrame.commands[0].unavailableReason ==
                 "camera-matrices-unavailable" &&
             residentFrame.commands[0].vectorVertexCount > 0u,
         "resident camera scaffold has reason and geometry");

  FrameRequest glossFrame;
  Input gloss = baseInput(ChromaspaceViewer::kPlotModelGlossView);
  gloss.glossPresentation = GlossPresentation::Projection3D;
  gloss.hasCameraMatrices = false;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(
             gloss, &glossFrame, &error),
         "Gloss 3D missing camera scaffolds");
  expect(glossFrame.commands[0].kind == PlotKind::Scaffold &&
             glossFrame.commands[0].unavailableReason ==
                 "camera-matrices-unavailable" &&
             glossFrame.commands[0].vectorVertexCount > 0u,
         "Gloss camera scaffold has reason and geometry");

  Input malformed = baseInput(ChromaspaceViewer::kPlotModelCube);
  malformed.hasCameraMatrices = true;
  malformed.modelView[0] = std::numeric_limits<float>::quiet_NaN();
  FrameRequest malformedFrame;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(
             malformed, &malformedFrame, &error),
         "non-finite supplied camera rejected");
}

void invalidAndCapacityInputsRollBack() {
  std::string error;
  FrameRequest frame;
  Input input = baseInput(ChromaspaceViewer::kPlotModelWaveform);
  input.scope.waveformMode = 9;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error) &&
             frame.commandCount == 0u && frame.vectorVertexArena.empty(),
         "invalid scope leaves frame unchanged");

  input = baseInput(ChromaspaceViewer::kPlotModelGlossView);
  input.glossDerivationHash = 0u;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error),
         "invalid Gloss hash rejected");
  input.glossDerivationHash = 1u;
  input.glossPresentation = GlossPresentation::Projection3D;
  input.hasCameraMatrices = false;
  expect(ChromaspaceMetalPlotCompiler::compileAndAppend(input, &frame, &error) &&
             frame.commands[frame.commandCount - 1u].kind == PlotKind::Scaffold,
         "missing Gloss camera scaffolds");

  FrameRequest vectorFull;
  vectorFull.vectorVertexArena.resize(
      ChromaspaceMetalPlotRenderer::kMaximumFrameVectorVertices);
  const auto vectorCount = vectorFull.vectorVertexArena.size();
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(
             baseInput(ChromaspaceViewer::kPlotModelCube), &vectorFull, &error) &&
             vectorFull.commandCount == 0u &&
             vectorFull.vectorVertexArena.size() == vectorCount,
         "vector capacity rollback");

  FrameRequest commandFull;
  commandFull.commandCount = ChromaspaceMetalPlotRenderer::kMaximumPlotWindows;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(
             baseInput(ChromaspaceViewer::kPlotModelSourceSignal), &commandFull,
             &error) &&
             commandFull.commandCount == ChromaspaceMetalPlotRenderer::kMaximumPlotWindows,
         "command capacity rollback");

  FrameRequest invalidBaseline;
  invalidBaseline.commandCount = ChromaspaceMetalPlotRenderer::kMaximumPlotWindows + 1u;
  expect(!ChromaspaceMetalPlotCompiler::compileAndAppend(
             baseInput(ChromaspaceViewer::kPlotModelSourceSignal), &invalidBaseline,
             &error),
         "invalid baseline rejected");
}

}  // namespace

int main() {
  sourceUnavailableEmitsScaffold();
  unrelatedStateDoesNotBlockSourceSignal();
  sourceSignalDoesNotRequireRaster();
  histogramSemanticsAreNormalized();
  waveformSemanticsAreNormalized();
  residentFieldsAndOrientationAreStable();
  glossKeepsHashAndViewRevisionSeparate();
  glossPresentationIsStrictlyValidated();
  missingCameraUsesPerWindowScaffold();
  invalidAndCapacityInputsRollBack();
  std::cout << "ChromaspaceMetalPlotCompilerTests: PASS\n";
  return 0;
}
