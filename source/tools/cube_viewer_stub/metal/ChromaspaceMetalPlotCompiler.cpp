#include "ChromaspaceMetalPlotCompiler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace ChromaspaceMetalPlotCompiler {
namespace {

using ChromaspaceMetal::FrameVectorVertex;
using ChromaspaceMetalPlotRenderer::FrameRequest;
using ChromaspaceMetalPlotRenderer::PlotCommand;
using ChromaspaceMetalPlotRenderer::PlotKind;

constexpr float kMaximumCoordinate =
    ChromaspaceMetalPlotRenderer::kMaximumPlotCoordinate;

void setError(std::string* error, const char* value) noexcept {
  if (!error) return;
  try {
    *error = value != nullptr ? value : "plot-compiler-error";
  } catch (...) {
  }
}

bool finite(float value) noexcept { return std::isfinite(value) != 0; }

bool flag01(int value) noexcept { return value == 0 || value == 1; }

bool finiteMatrix(const std::array<float, 16>& matrix) noexcept {
  for (float value : matrix) {
    if (!finite(value) || std::fabs(value) > kMaximumCoordinate) return false;
  }
  return true;
}

bool finiteRect(const ChromaspaceMetalPlotRenderer::PlotRect& rect) noexcept {
  return finite(rect.x) && finite(rect.y) && finite(rect.width) &&
         finite(rect.height) && rect.width > 0.0f && rect.height > 0.0f &&
         rect.x >= -kMaximumCoordinate && rect.x <= kMaximumCoordinate &&
         rect.y >= -kMaximumCoordinate && rect.y <= kMaximumCoordinate &&
         rect.width <= kMaximumCoordinate && rect.height <= kMaximumCoordinate;
}

bool validPlotModel(int plotModel) noexcept {
  return plotModel >= ChromaspaceViewer::kPlotModelCube &&
         plotModel <= ChromaspaceViewer::kPlotModelSourceSignal;
}

bool isSourceSignal(int plotModel) noexcept {
  return plotModel == ChromaspaceViewer::kPlotModelSourceSignal;
}

bool isGloss(int plotModel) noexcept {
  return plotModel == ChromaspaceViewer::kPlotModelGlossView;
}

bool requiresCamera(int plotModel,
                    GlossPresentation presentation) noexcept {
  const bool residentRaster =
      plotModel >= ChromaspaceViewer::kPlotModelCube &&
      plotModel <= ChromaspaceViewer::kPlotModelChromaticity;
  return residentRaster ||
         (plotModel == ChromaspaceViewer::kPlotModelGlossView &&
          presentation == GlossPresentation::Projection3D);
}

PlotKind kindFor(int plotModel,
                 GlossPresentation presentation,
                 bool sourceAvailable) noexcept {
  if (!sourceAvailable) return PlotKind::Scaffold;
  switch (plotModel) {
    case ChromaspaceViewer::kPlotModelSourceSignal:
      return PlotKind::SourceSignal;
    case ChromaspaceViewer::kPlotModelWaveform:
      return PlotKind::Waveform;
    case ChromaspaceViewer::kPlotModelHistogram:
      return PlotKind::Histogram;
    case ChromaspaceViewer::kPlotModelGlossView:
      return presentation == GlossPresentation::Projection3D
                 ? PlotKind::GlossProjection3D
                 : PlotKind::GlossField2D;
    default:
      return PlotKind::ResidentRaster;
  }
}

// This is intentionally only the compiler's direct safety contract. The
// renderer's exhaustive RasterSourceRequest validator remains authoritative.
bool validRasterDirect(const ChromaspaceMetal::RasterSourceRequest& raster) noexcept {
  if (raster.pointCount <= 0 ||
      raster.pointCount > ChromaspaceMetalPlotRenderer::kMaximumResidentRasterPoints ||
      raster.sourceWidth <= 0 || raster.sourceHeight <= 0 ||
      raster.sourceWidth > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      raster.sourceHeight > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      raster.sampleStride <= 0 || raster.sampleCountX <= 0 ||
      raster.sampleCountX > 1 + (raster.sourceWidth - 1) / raster.sampleStride ||
      raster.pixelFormat < 0 || raster.pixelFormat > 1 ||
      !finite(raster.sourceAspect) || raster.sourceAspect <= 0.0f ||
      raster.sourceAspect > kMaximumCoordinate || !finite(raster.glossLiftScale) ||
      !finite(raster.pointAlphaScale) || !finite(raster.denseAlphaBias) ||
      !finite(raster.colorSaturation)) {
    return false;
  }
  return true;
}

bool validScope(const NormalizedScopeState& scope) noexcept {
  return scope.waveformMode >= 0 && scope.waveformMode <= 2 &&
         scope.histogramMode >= 0 && scope.histogramMode <= 1 &&
         flag01(scope.waveformChannelRed) &&
         flag01(scope.waveformChannelGreen) &&
         flag01(scope.waveformChannelBlue) &&
         flag01(scope.waveformChannelLuma) &&
         flag01(scope.waveformShowOverflow) &&
         flag01(scope.waveformHighlightOverflow) &&
         scope.waveformLumaMethod >= 0 && scope.waveformLumaMethod <= 3 &&
         flag01(scope.histogramShowOverflow) &&
         flag01(scope.histogramHighlightOverflow) &&
         scope.scopeRangeMode >= 0 && scope.scopeRangeMode <= 2 &&
         finite(scope.waveformPointBrightness) &&
         scope.waveformPointBrightness >= 0.1f &&
         scope.waveformPointBrightness <= 2.0f &&
         finite(scope.waveformSaturation) && scope.waveformSaturation >= 0.0f &&
         scope.waveformSaturation <= 1.0f;
}

bool validGlossControls(const Input& input) noexcept {
  const auto& controls = input.gloss;
  return input.glossGridWidth > 0 && input.glossGridHeight > 0 &&
         input.glossGridWidth <= ChromaspaceMetalPlotRenderer::kMaximumPlotDimension &&
         input.glossGridHeight <= ChromaspaceMetalPlotRenderer::kMaximumPlotDimension &&
         flag01(input.glossShowOverflow) &&
         input.glossNeighborhoodChoice >= 0 &&
         input.glossNeighborhoodChoice <= 2 && input.glossDerivationHash != 0u &&
         controls.algorithm >= 0 && controls.algorithm <= 1 &&
         controls.colorMode >= 0 && controls.colorMode <= 1 &&
         controls.debugMode >= 0 && controls.debugMode <= 4 &&
         controls.diagnosticMode >= 0 && controls.diagnosticMode <= 2 &&
         finite(controls.colorSaturation) &&
         finite(controls.glossBodyOpacity) &&
         finite(controls.glossHighlightOpacity) &&
         finite(controls.glossLiftScale);
}

bool validPresentation(GlossPresentation presentation) noexcept {
  return presentation == GlossPresentation::Field2D ||
         presentation == GlossPresentation::Projection3D;
}

bool validInput(const Input& input, std::string* error) noexcept {
  if (!validPlotModel(input.plotModel)) {
    setError(error, "plot-compiler-model-invalid");
    return false;
  }
  if (input.windowId <= 0 || !finiteRect(input.destination) ||
      input.targetWidth <= 0 || input.targetHeight <= 0 ||
      input.targetWidth > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      input.targetHeight > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      (input.targetPixelFormat != 0 && input.targetPixelFormat != 1) ||
      input.viewRevision == 0u || input.contentRevision == 0u) {
    setError(error, "plot-compiler-input-invalid");
    return false;
  }
  if (!input.sourceAvailable) {
    if (input.unavailableReason.empty() || input.unavailableReason.size() > 512u) {
      setError(error, "plot-compiler-source-reason-invalid");
      return false;
    }
    return true;
  }
  const bool scopeModel =
      input.plotModel == ChromaspaceViewer::kPlotModelHistogram ||
      input.plotModel == ChromaspaceViewer::kPlotModelWaveform;
  if (scopeModel && !validScope(input.scope)) {
    setError(error, "plot-compiler-scope-invalid");
    return false;
  }
  if (!isSourceSignal(input.plotModel) &&
      (!input.hasRasterRequest || !validRasterDirect(input.raster))) {
    setError(error, "plot-compiler-raster-direct-invalid");
    return false;
  }
  if (input.hasCameraMatrices &&
      (!finiteMatrix(input.modelView) || !finiteMatrix(input.projection))) {
    setError(error, "plot-compiler-camera-matrices-malformed");
    return false;
  }
  if (isGloss(input.plotModel)) {
    if (!validPresentation(input.glossPresentation) ||
        !validGlossControls(input)) {
      setError(error, "plot-compiler-gloss-invalid");
      return false;
    }
    if (input.glossPresentation == GlossPresentation::Projection3D &&
        (!finite(input.pointRadiusPixels) || input.pointRadiusPixels < 0.0f ||
         input.pointRadiusPixels > kMaximumCoordinate)) {
      setError(error, "plot-compiler-point-controls-invalid");
      return false;
    }
  }
  if (requiresCamera(input.plotModel, input.glossPresentation)) {
    if (!finite(input.pointRadiusPixels) || input.pointRadiusPixels < 0.0f ||
        input.pointRadiusPixels > kMaximumCoordinate) {
      setError(error, "plot-compiler-point-controls-invalid");
      return false;
    }
  }
  const bool residentRaster =
      input.plotModel >= ChromaspaceViewer::kPlotModelCube &&
      input.plotModel <= ChromaspaceViewer::kPlotModelChromaticity;
  if (residentRaster) {
    if (!finite(input.backgroundR) || !finite(input.backgroundG) ||
        !finite(input.backgroundB) || input.backgroundR < 0.0f ||
        input.backgroundR > 1.0f || input.backgroundG < 0.0f ||
        input.backgroundG > 1.0f || input.backgroundB < 0.0f ||
        input.backgroundB > 1.0f) {
      setError(error, "plot-compiler-background-invalid");
      return false;
    }
  }
  return true;
}

void appendVertex(std::vector<FrameVectorVertex>* vertices,
                  float x,
                  float y,
                  const std::array<float, 4>& color) {
  vertices->push_back({x, y, color[0], color[1], color[2], color[3]});
}

void appendTriangle(std::vector<FrameVectorVertex>* vertices,
                    float x0,
                    float y0,
                    float x1,
                    float y1,
                    float x2,
                    float y2,
                    const std::array<float, 4>& color) {
  appendVertex(vertices, x0, y0, color);
  appendVertex(vertices, x1, y1, color);
  appendVertex(vertices, x2, y2, color);
}

void appendLine(std::vector<FrameVectorVertex>* vertices,
                float x0,
                float y0,
                float x1,
                float y1,
                const std::array<float, 4>& color) {
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float length = std::sqrt(dx * dx + dy * dy);
  if (!finite(length) || length <= 1.0e-4f) return;
  const float nx = -dy / length * 0.75f;
  const float ny = dx / length * 0.75f;
  appendTriangle(vertices, x0 - nx, y0 - ny, x1 - nx, y1 - ny,
                 x1 + nx, y1 + ny, color);
  appendTriangle(vertices, x0 - nx, y0 - ny, x1 + nx, y1 + ny,
                 x0 + nx, y0 + ny, color);
}

void appendPlotGuides(std::vector<FrameVectorVertex>* vertices,
                      float width,
                      float height,
                      int plotModel) {
  const std::array<float, 4> grid{{0.18f, 0.24f, 0.32f, 0.60f}};
  const std::array<float, 4> axis{{0.55f, 0.64f, 0.76f, 0.85f}};
  for (int step = 1; step < 10; ++step) {
    const float x = width * static_cast<float>(step) / 10.0f;
    const float y = height * static_cast<float>(step) / 10.0f;
    appendLine(vertices, x, 0.0f, x, height, grid);
    appendLine(vertices, 0.0f, y, width, y, grid);
  }
  appendLine(vertices, 0.0f, 0.0f, width, 0.0f, axis);
  appendLine(vertices, width, 0.0f, width, height, axis);
  appendLine(vertices, width, height, 0.0f, height, axis);
  appendLine(vertices, 0.0f, height, 0.0f, 0.0f, axis);
  if (plotModel == ChromaspaceViewer::kPlotModelChromaticity) {
    const std::array<float, 4> locus{{0.70f, 0.85f, 0.35f, 0.85f}};
    // Deliberately asymmetric; +Y is down in chart space.
    constexpr float points[][2] = {
        {0.12f, 0.18f}, {0.22f, 0.36f}, {0.38f, 0.62f},
        {0.58f, 0.78f}, {0.78f, 0.70f}, {0.88f, 0.44f},
        {0.72f, 0.22f}, {0.42f, 0.12f},
    };
    for (std::size_t index = 1; index < std::size(points); ++index) {
      appendLine(vertices, points[index - 1][0] * width,
                 points[index - 1][1] * height, points[index][0] * width,
                 points[index][1] * height, locus);
    }
  }
}

void appendPrimer(std::vector<FrameVectorVertex>* vertices,
                  float width,
                  const std::array<float, 4>& color) {
  appendTriangle(vertices, 0.0f, 0.0f, width, 0.0f, width, 2.0f, color);
  appendTriangle(vertices, 0.0f, 0.0f, width, 2.0f, 0.0f, 2.0f, color);
}

void setScopeRange(int mode, float* rangeMin, float* invRange,
                   int* useGpuAutoRange) noexcept {
  *rangeMin = 0.0f;
  *invRange = mode == 1 ? 0.25f : 1.0f;
  *useGpuAutoRange = mode == 2 ? 1 : 0;
}

void fillHistogram(const Input& input, PlotCommand* command) noexcept {
  auto& request = command->histogram;
  request.pointCount = command->raster.pointCount;
  request.width = command->targetWidth;
  request.height = command->targetHeight;
  request.scopeMode = input.scope.histogramMode;
  request.rangeMin = 0.0f;
  request.invRange = input.scope.scopeRangeMode == 1 ? 0.25f : 1.0f;
  request.showOverflow = input.scope.histogramShowOverflow;
  request.highlightOverflow = input.scope.histogramHighlightOverflow;
  request.lumaMethod = 0;
  setScopeRange(input.scope.scopeRangeMode, &request.rangeMin,
                &request.invRange, &request.useGpuAutoRange);
  request.autoRange.pointCount = request.pointCount;
  request.autoRange.waveform = 0;
  request.autoRange.scopeMode = request.scopeMode;
  request.autoRange.includeRed = request.scopeMode == 1 ? 0 : 1;
  request.autoRange.includeGreen = request.scopeMode == 1 ? 0 : 1;
  request.autoRange.includeBlue = request.scopeMode == 1 ? 0 : 1;
  request.autoRange.includeLuma = request.scopeMode == 1 ? 1 : 0;
  request.autoRange.includeOverflow = request.showOverflow;
  request.autoRange.lumaMethod = 0;
}

void fillWaveform(const Input& input, PlotCommand* command) noexcept {
  auto& request = command->waveform;
  request.pointCount = command->raster.pointCount;
  request.width = command->targetWidth;
  request.height = command->targetHeight;
  request.scopeMode = input.scope.waveformMode;
  request.lumaMethod = input.scope.waveformLumaMethod;
  request.showOverflow = input.scope.waveformShowOverflow;
  request.highlightOverflow = input.scope.waveformHighlightOverflow;
  request.includeRed = input.scope.waveformChannelRed;
  request.includeGreen = input.scope.waveformChannelGreen;
  request.includeBlue = input.scope.waveformChannelBlue;
  request.includeLuma = input.scope.waveformChannelLuma;
  if (request.scopeMode == 2) {
    request.includeRed = request.includeGreen = request.includeBlue = 0;
    request.includeLuma = 1;
  }
  request.pointBrightness = input.scope.waveformPointBrightness;
  request.colorSaturation = input.scope.waveformSaturation;
  request.coverageAlpha = 1.0f;
  setScopeRange(input.scope.scopeRangeMode, &request.rangeMin,
                &request.invRange, &request.useGpuAutoRange);
  request.autoRange.pointCount = request.pointCount;
  request.autoRange.waveform = 1;
  request.autoRange.scopeMode = request.scopeMode;
  request.autoRange.includeRed = request.includeRed;
  request.autoRange.includeGreen = request.includeGreen;
  request.autoRange.includeBlue = request.includeBlue;
  request.autoRange.includeLuma = request.includeLuma;
  request.autoRange.includeOverflow = request.showOverflow;
  request.autoRange.lumaMethod = request.lumaMethod;
}

void fillGloss(const Input& input, PlotCommand* command) noexcept {
  command->glossField.gridWidth = input.glossGridWidth;
  command->glossField.gridHeight = input.glossGridHeight;
  command->glossField.showOverflow = input.glossShowOverflow;
  command->glossField.neighborhoodChoice = input.glossNeighborhoodChoice;
  command->glossDerivationHash = input.glossDerivationHash;
  if (command->kind == PlotKind::GlossField2D) {
    auto& request = command->glossFieldSurface;
    request.width = command->targetWidth;
    request.height = command->targetHeight;
    request.algorithm = input.gloss.algorithm;
    request.colorMode = input.gloss.colorMode;
    request.debugMode = input.gloss.debugMode;
    request.diagnosticMode = input.gloss.diagnosticMode;
    request.colorSaturation = input.gloss.colorSaturation;
    request.glossBodyOpacity = input.gloss.glossBodyOpacity;
    request.glossHighlightOpacity = input.gloss.glossHighlightOpacity;
    request.glossLiftScale = input.gloss.glossLiftScale;
    return;
  }
  auto& request = command->glossProjectionSurface;
  request.width = command->targetWidth;
  request.height = command->targetHeight;
  request.algorithm = input.gloss.algorithm;
  request.colorMode = input.gloss.colorMode;
  request.debugMode = input.gloss.debugMode;
  request.diagnosticMode = input.gloss.diagnosticMode;
  request.sourceAspect = command->raster.sourceAspect;
  request.colorSaturation = input.gloss.colorSaturation;
  request.glossBodyOpacity = input.gloss.glossBodyOpacity;
  request.glossHighlightOpacity = input.gloss.glossHighlightOpacity;
  request.glossLiftScale = input.gloss.glossLiftScale;
  request.pointRadiusPixels = input.pointRadiusPixels;
  std::copy(input.modelView.begin(), input.modelView.end(), request.modelView);
  std::copy(input.projection.begin(), input.projection.end(), request.projection);
}

}  // namespace

const char* plotModelLabel(int plotModel) noexcept {
  switch (plotModel) {
    case ChromaspaceViewer::kPlotModelCube: return "cube";
    case ChromaspaceViewer::kPlotModelHsl: return "hsl";
    case ChromaspaceViewer::kPlotModelHsv: return "hsv";
    case ChromaspaceViewer::kPlotModelChen: return "chen";
    case ChromaspaceViewer::kPlotModelNormCone: return "norm-cone";
    case ChromaspaceViewer::kPlotModelJpConical: return "jp-conical";
    case ChromaspaceViewer::kPlotModelReuleaux: return "reuleaux";
    case ChromaspaceViewer::kPlotModelChromaticity: return "chromaticity";
    case ChromaspaceViewer::kPlotModelGlossView: return "gloss-view";
    case ChromaspaceViewer::kPlotModelWaveform: return "waveform";
    case ChromaspaceViewer::kPlotModelHistogram: return "histogram";
    case ChromaspaceViewer::kPlotModelSourceSignal: return "source-signal";
    default: return "unknown";
  }
}

bool compileAndAppend(const Input& input, FrameRequest* frame,
                      std::string* error) noexcept {
  if (error) {
    try {
      error->clear();
    } catch (...) {
    }
  }
  if (!frame) {
    setError(error, "plot-compiler-frame-missing");
    return false;
  }
  const std::size_t originalCommands = frame->commandCount;
  const std::size_t originalVertices = frame->vectorVertexArena.size();
  try {
    if (originalCommands > ChromaspaceMetalPlotRenderer::kMaximumPlotWindows ||
        originalVertices > ChromaspaceMetalPlotRenderer::kMaximumFrameVectorVertices) {
      setError(error, "plot-compiler-baseline-frame-invalid");
      return false;
    }
    if (originalCommands >= ChromaspaceMetalPlotRenderer::kMaximumPlotWindows) {
      setError(error, "plot-compiler-command-limit");
      return false;
    }
    if (!validInput(input, error)) return false;

    PlotCommand command{};
    command.windowId = input.windowId;
    command.plotModel = input.plotModel;
    command.destination = input.destination;
    command.targetWidth = input.targetWidth;
    command.targetHeight = input.targetHeight;
    command.targetPixelFormat = input.targetPixelFormat;
    command.viewRevision = input.viewRevision;
    command.contentRevision = input.contentRevision;
    command.kind = kindFor(input.plotModel, input.glossPresentation,
                           input.sourceAvailable);
    if (!input.sourceAvailable) command.unavailableReason = input.unavailableReason;
    if (input.sourceAvailable &&
        requiresCamera(input.plotModel, input.glossPresentation) &&
        !input.hasCameraMatrices) {
      command.kind = PlotKind::Scaffold;
      command.unavailableReason = "camera-matrices-unavailable";
    }
    if (input.sourceAvailable && !isSourceSignal(input.plotModel)) {
      command.raster = input.raster;
    }
    switch (command.kind) {
      case PlotKind::Histogram: fillHistogram(input, &command); break;
      case PlotKind::Waveform: fillWaveform(input, &command); break;
      case PlotKind::ResidentRaster:
        command.point.pointCount = command.raster.pointCount;
        command.point.width = command.targetWidth;
        command.point.height = command.targetHeight;
        command.point.pointRadiusPixels = input.pointRadiusPixels;
        command.point.backgroundR = input.backgroundR;
        command.point.backgroundG = input.backgroundG;
        command.point.backgroundB = input.backgroundB;
        command.point.backgroundA = 1.0f;
        std::copy(input.modelView.begin(), input.modelView.end(), command.point.modelView);
        std::copy(input.projection.begin(), input.projection.end(), command.point.projection);
        break;
      case PlotKind::GlossField2D:
      case PlotKind::GlossProjection3D: fillGloss(input, &command); break;
      default: break;
    }

    // Reserve once before generating guides; no external frame state is
    // touched until the complete vector batch is available.
    std::vector<FrameVectorVertex> staged;
    staged.reserve(command.kind == PlotKind::ResidentRaster ? 192u
                   : command.kind == PlotKind::Scaffold ? 6u : 0u);
    if (command.kind == PlotKind::Scaffold) {
      command.vectorClearBeforeDraw = true;
      command.vectorClearColor = {0.04f, 0.05f, 0.07f, 1.0f};
      appendPrimer(&staged, static_cast<float>(command.targetWidth),
                   command.vectorClearColor);
    } else if (command.kind == PlotKind::ResidentRaster) {
      command.vectorClearBeforeDraw = false;
      command.vectorClearColor = {0.22f, 0.34f, 0.48f, 0.75f};
      appendPrimer(&staged, static_cast<float>(command.targetWidth),
                   command.vectorClearColor);
      appendPlotGuides(&staged, static_cast<float>(command.targetWidth),
                       static_cast<float>(command.targetHeight), input.plotModel);
    }
    if (staged.size() > ChromaspaceMetalPlotRenderer::kMaximumCommandVectorVertices ||
        (staged.size() % 3u) != 0u ||
        staged.size() > ChromaspaceMetalPlotRenderer::kMaximumFrameVectorVertices -
                           originalVertices) {
      setError(error, "plot-compiler-vector-limit");
      return false;
    }
    if (!staged.empty() &&
        !frame->appendVectorVertices(staged.data(), staged.size(), &command)) {
      frame->vectorVertexArena.resize(originalVertices);
      setError(error, "plot-compiler-vector-append-failed");
      return false;
    }
    if (!frame->append(command)) {
      frame->vectorVertexArena.resize(originalVertices);
      frame->commandCount = originalCommands;
      setError(error, "plot-compiler-command-append-failed");
      return false;
    }
    return true;
  } catch (...) {
    try {
      frame->vectorVertexArena.resize(originalVertices);
      frame->commandCount = originalCommands;
    } catch (...) {
    }
    setError(error, "plot-compiler-exception");
    return false;
  }
}

}  // namespace ChromaspaceMetalPlotCompiler
