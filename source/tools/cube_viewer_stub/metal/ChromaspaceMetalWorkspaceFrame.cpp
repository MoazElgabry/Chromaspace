#include "ChromaspaceMetalWorkspaceFrame.h"

#include "ChromaspaceViewerCamera.h"
#include "color/ColorManagement.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ChromaspaceMetalWorkspaceFrame {
namespace {

using ChromaspaceViewer::PlotWindowDomainState;
using ChromaspaceViewer::LassoStroke;
using ChromaspaceViewer::ViewerRuntimeState;

void setError(std::string* error, const char* value) noexcept {
  if (!error) return;
  try {
    *error = value ? value : "workspace-frame-error";
  } catch (...) {
  }
}

const PlotWindowDomainState* findWindow(
    const ChromaspaceViewer::ViewerWorkspaceState& workspace,
    int windowId) noexcept {
  for (const auto& window : workspace.windows) {
    if (window.windowId == windowId) return &window;
  }
  return nullptr;
}

int remapPlotMode(int plotModel) noexcept {
  switch (plotModel) {
    case ChromaspaceViewer::kPlotModelHsl: return 1;
    case ChromaspaceViewer::kPlotModelHsv: return 2;
    case ChromaspaceViewer::kPlotModelChen: return 3;
    case ChromaspaceViewer::kPlotModelJpConical: return 5;
    case ChromaspaceViewer::kPlotModelNormCone: return 6;
    case ChromaspaceViewer::kPlotModelReuleaux: return 7;
    case ChromaspaceViewer::kPlotModelChromaticity: return 8;
    case ChromaspaceViewer::kPlotModelGlossView: return 9;
    default: return 0;
  }
}

uint64_t hashAppend(uint64_t hash, uint64_t value) noexcept {
  for (int shift = 0; shift < 64; shift += 8) {
    hash ^= (value >> shift) & 0xffu;
    hash *= 1099511628211ull;
  }
  return hash;
}

int samplingStride(int width, int height, const ViewerRuntimeState& state) noexcept {
  constexpr uint64_t kQualityBudgets[] = {131072u, 524288u, 1048576u};
  constexpr float kScaleFactors[] = {0.25f, 0.5f, 0.75f, 1.0f};
  const int quality = std::clamp(state.quality, 0, 2);
  const int scale = std::clamp(state.scale, 0, 3);
  const uint64_t budget = std::max<uint64_t>(
      1u, static_cast<uint64_t>(kQualityBudgets[quality] * kScaleFactors[scale]));
  int stride = 1;
  for (;;) {
    const uint64_t countX =
        (static_cast<uint64_t>(width) + stride - 1u) / stride;
    const uint64_t countY =
        (static_cast<uint64_t>(height) + stride - 1u) / stride;
    if (countX * countY <= budget &&
        countX * countY <=
            ChromaspaceMetalPlotRenderer::kMaximumResidentRasterPoints) {
      return stride;
    }
    if (stride >= std::max(width, height)) return std::max(width, height);
    ++stride;
  }
}

void fillColorRemap(const ViewerRuntimeState& state,
                    ChromaspaceMetal::RemapUniforms* remap) noexcept {
  remap->plotMode = remapPlotMode(state.plotModel);
  remap->circularHsl = state.circularHsl ? 1 : 0;
  remap->circularHsv = state.circularHsv ? 1 : 0;
  remap->normConeNormalized = state.normConeNormalized ? 1 : 0;
  remap->showOverflow = state.showOverflow ? 1 : 0;
  remap->highlightOverflow = state.highlightOverflow ? 1 : 0;
  const auto primaries = WorkshopColor::primariesIdFromChoiceIndex(
      state.chromaticityInputPrimaries);
  const auto transfer = WorkshopColor::transferFunctionIdFromChoiceIndex(
      state.chromaticityInputTransfer);
  const auto white = WorkshopColor::whitePoint(primaries);
  const auto rgbToXyz = WorkshopColor::rgbToXyzMatrix(primaries);
  const auto xyzToRgb = WorkshopColor::xyzToRgbMatrix(primaries);
  remap->chromaticityInputTransfer = static_cast<int>(transfer);
  remap->chromaticityReferenceBasis =
      state.chromaticityReferenceBasis == 1 ? 1 : 0;
  remap->chromaticityWhiteX = white.x;
  remap->chromaticityWhiteY = white.y;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const int index = row * 3 + column;
      remap->chromaticityRgbToXyz[index] = rgbToXyz.m[row][column];
      remap->chromaticityXyzToRgb[index] = xyzToRgb.m[row][column];
    }
  }
}

void fillLasso(const std::vector<LassoStroke>& strokes,
               ChromaspaceMetal::RasterSourceRequest* raster) noexcept {
  int pointOffset = 0;
  int strokeCount = 0;
  for (const auto& stroke : strokes) {
    if (strokeCount >= 16 || pointOffset >= 256) break;
    const int available = 256 - pointOffset;
    const int pointCount = std::min<int>(
        available, static_cast<int>(stroke.points.size()));
    if (pointCount < 3) continue;
    raster->lassoStrokeFirst[strokeCount] = pointOffset;
    raster->lassoStrokeCountPerStroke[strokeCount] = pointCount;
    raster->lassoStrokeSubtract[strokeCount] = stroke.subtract ? 1 : 0;
    for (int index = 0; index < pointCount; ++index) {
      raster->lassoX[pointOffset] = stroke.points[index].xNorm;
      raster->lassoY[pointOffset] = stroke.points[index].yNorm;
      ++pointOffset;
    }
    ++strokeCount;
  }
  raster->lassoStrokeCount = strokeCount;
  raster->lassoPointCount = pointOffset;
  raster->lassoEnabled = strokeCount > 0 ? 1 : 0;
}

bool fillRaster(const PlotWindowDomainState& window,
                const std::vector<LassoStroke>& lassoStrokes,
                const ChromaspaceMetal::ImportedSourceTexture& source,
                ChromaspaceMetal::RasterSourceRequest* raster) noexcept {
  if (!raster || source.width <= 0 || source.height <= 0 ||
      source.width > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      source.height > ChromaspaceMetalPlotRenderer::kMaximumPlotDimension ||
      (source.pixelFormat != 0 && source.pixelFormat != 1)) {
    return false;
  }
  const ViewerRuntimeState state =
      ChromaspaceViewer::clampedViewerRuntimeState(window.viewState);
  const int stride = samplingStride(source.width, source.height, state);
  const int countX = 1 + (source.width - 1) / stride;
  const int countY = 1 + (source.height - 1) / stride;
  const int pointCount = countX * countY;
  ChromaspaceMetal::RasterSourceRequest result{};
  result.pointCount = pointCount;
  result.basePointCount = pointCount;
  result.sourceWidth = source.width;
  result.sourceHeight = source.height;
  result.sampleStride = stride;
  result.sampleCountX = countX;
  result.pixelFormat = source.pixelFormat;
  const uint32_t semanticWidth = source.semantics.sourceWidth;
  const uint32_t semanticHeight = source.semantics.sourceHeight;
  result.sourceAspect = semanticWidth > 0u && semanticHeight > 0u
                            ? static_cast<float>(semanticWidth) /
                                  static_cast<float>(semanticHeight)
                            : static_cast<float>(source.width) /
                                  static_cast<float>(source.height);
  result.glossLiftScale = static_cast<float>(state.glossLiftScale);
  result.colorSaturation = static_cast<float>(state.colorSaturation);
  result.plotLinear = state.plotDisplayLinear ? 1 : 0;
  result.plotLinearTransfer = std::clamp(state.plotDisplayLinearTransfer, 0, 17);
  result.excludeIdentityData = state.excludeIdentityData ? 1 : 0;
  result.isolateIdentityData = state.isolateIdentityData ? 1 : 0;
  result.readIdentityPlot = state.readIdentityPlot ? 1 : 0;
  result.readGrayRamp = state.readGrayRamp ? 1 : 0;
  if (source.semantics.identityStripPresent) {
    result.identityCubeY1 = source.semantics.identityCubeY1;
    result.identityCubeY2 = source.semantics.identityCubeY2;
    result.identityRampY1 = source.semantics.identityRampY1;
    result.identityRampY2 = source.semantics.identityRampY2;
  }
  result.occupancyFill = state.occupancyGuidedFill ? 1 : 0;
  result.cubeSlicingEnabled =
      (state.volumeSliceLassoRegion || state.volumeSliceRed ||
       state.volumeSliceYellow || state.volumeSliceGreen ||
       state.volumeSliceCyan || state.volumeSliceBlue ||
       state.volumeSliceMagenta)
          ? 1
          : 0;
  result.neutralRadiusEnabled = state.neutralRadius < 0.999999 ? 1 : 0;
  result.neutralRadius = static_cast<float>(state.neutralRadius);
  result.cubeSliceRed = state.volumeSliceRed ? 1 : 0;
  result.cubeSliceYellow = state.volumeSliceYellow ? 1 : 0;
  result.cubeSliceGreen = state.volumeSliceGreen ? 1 : 0;
  result.cubeSliceCyan = state.volumeSliceCyan ? 1 : 0;
  result.cubeSliceBlue = state.volumeSliceBlue ? 1 : 0;
  result.cubeSliceMagenta = state.volumeSliceMagenta ? 1 : 0;
  fillLasso(lassoStrokes, &result);
  fillColorRemap(state, &result.remap);
  *raster = result;
  return true;
}

void fillScope(const ViewerRuntimeState& state,
               ChromaspaceMetalPlotCompiler::NormalizedScopeState* scope) noexcept {
  scope->waveformMode = state.waveformMode;
  scope->histogramMode = state.histogramMode;
  scope->waveformChannelRed = state.waveformChannelRed ? 1 : 0;
  scope->waveformChannelGreen = state.waveformChannelGreen ? 1 : 0;
  scope->waveformChannelBlue = state.waveformChannelBlue ? 1 : 0;
  scope->waveformChannelLuma = state.waveformChannelLuma ? 1 : 0;
  scope->waveformShowOverflow = state.waveformShowOverflow ? 1 : 0;
  scope->waveformHighlightOverflow = state.waveformHighlightOverflow ? 1 : 0;
  scope->waveformLumaMethod = state.waveformLumaMethod;
  scope->histogramShowOverflow = state.histogramShowOverflow ? 1 : 0;
  scope->histogramHighlightOverflow = state.histogramHighlightOverflow ? 1 : 0;
  scope->scopeRangeMode = state.scopeRangeMode;
  scope->waveformPointBrightness =
      static_cast<float>(state.waveformPointBrightness);
  scope->waveformSaturation = static_cast<float>(state.waveformSaturation);
}

bool fillCamera(const PlotWindowDomainState& window,
                int width,
                int height,
                ChromaspaceMetalPlotCompiler::Input* input) noexcept {
  ChromaspaceViewer::ViewerCameraMatricesRequest cameraRequest{};
  cameraRequest.camera = window.camera;
  cameraRequest.viewportWidth = width;
  cameraRequest.viewportHeight = height;
  ChromaspaceViewer::ViewerCameraMatrices matrices{};
  if (!ChromaspaceViewer::buildViewerCameraMatrices(cameraRequest, &matrices)) {
    return false;
  }
  input->modelView = matrices.modelView;
  input->projection = matrices.projection;
  input->hasCameraMatrices = true;
  return true;
}

}  // namespace

bool applyLiveParamsToFocusedWindow(
    const ChromaspaceViewer::ViewerLiveCommandParams& params,
    ChromaspaceViewer::ViewerWorkspaceState* workspace,
    int* updatedWindowId,
    std::string* error) noexcept {
  if (updatedWindowId) *updatedWindowId = -1;
  if (error) {
    try {
      error->clear();
    } catch (...) {
    }
  }
  if (!workspace ||
      !ChromaspaceViewer::validateViewerWorkspaceState(*workspace)) {
    setError(error, "live-workspace-invalid");
    return false;
  }
  if (workspace->revision == std::numeric_limits<uint64_t>::max()) {
    setError(error, "live-workspace-revision-overflow");
    return false;
  }
  PlotWindowDomainState* target = nullptr;
  for (auto& window : workspace->windows) {
    if (window.windowId == workspace->focusedWindowId) {
      target = &window;
      break;
    }
  }
  if (!target) {
    setError(error, "live-workspace-focus-missing");
    return false;
  }
  try {
    ViewerRuntimeState candidate =
        ChromaspaceViewer::clampedViewerRuntimeState(params.viewerState);
    candidate.stateRevision = std::max<uint64_t>(
        1u, std::max(candidate.stateRevision, params.stateRevision));
    std::swap(target->viewState, candidate);
    const uint64_t oldRevision = workspace->revision;
    ++workspace->revision;
    if (!ChromaspaceViewer::validateViewerWorkspaceState(*workspace)) {
      std::swap(target->viewState, candidate);
      workspace->revision = oldRevision;
      setError(error, "live-workspace-result-invalid");
      return false;
    }
    if (updatedWindowId) *updatedWindowId = target->windowId;
    return true;
  } catch (...) {
    setError(error, "live-workspace-allocation-failure");
    return false;
  }
}

CompileResult compileWorkspaceFrame(const CompileRequest& request) noexcept {
  CompileResult result{};
  try {
    if (!request.framePlan || !request.workspace || request.frameRevision == 0u) {
      result.status = CompileStatus::InvalidRequest;
      result.diagnostic = "workspace-frame-request-invalid";
      return result;
    }
    if (!ChromaspaceViewer::validateViewerWorkspaceState(*request.workspace)) {
      result.status = CompileStatus::InvalidWorkspace;
      result.diagnostic = "workspace-frame-workspace-invalid";
      return result;
    }
    if (!request.framePlan->ready()) {
      result.status = CompileStatus::InvalidFramePlan;
      result.rejectedWindowId = request.framePlan->rejectedWindowId;
      result.diagnostic = "workspace-frame-plan-invalid";
      return result;
    }

    std::string sourceError;
    const bool sourceAvailable =
        request.residentSource != nullptr &&
        ChromaspaceMetalPlotRenderer::validateResidentSource(
            *request.residentSource, &sourceError);
    result.frame.frameRevision = request.frameRevision;
    result.frame.hasResidentSource = sourceAvailable;
    if (sourceAvailable) result.frame.residentSource = *request.residentSource;
    const std::vector<LassoStroke> emptyLasso;

    for (const auto& planned : request.framePlan->windows) {
      const PlotWindowDomainState* window =
          findWindow(*request.workspace, planned.windowId);
      if (!window) {
        result.frame.clear();
        result.status = CompileStatus::MissingWorkspaceWindow;
        result.rejectedWindowId = planned.windowId;
        result.diagnostic = "workspace-frame-window-missing";
        return result;
      }
      const ViewerRuntimeState state =
          ChromaspaceViewer::clampedViewerRuntimeState(window->viewState);
      ChromaspaceMetalPlotCompiler::Input input{};
      input.windowId = planned.windowId;
      input.plotModel = planned.plotModel;
      input.glossPresentation = request.glossPresentation;
      input.destination = {
          planned.framebufferRect.x0,
          planned.framebufferRect.y0,
          std::max(1.0f, planned.framebufferRect.x1 - planned.framebufferRect.x0),
          std::max(1.0f, planned.framebufferRect.y1 - planned.framebufferRect.y0)};
      input.targetWidth = std::max(1, planned.renderTargetWidth);
      input.targetHeight = std::max(1, planned.renderTargetHeight);
      input.viewRevision = std::max<uint64_t>(1u, planned.viewRevision);
      input.contentRevision = input.viewRevision;
      input.sourceAvailable = sourceAvailable;
      input.unavailableReason = !request.sourceDiagnostic.empty()
                                    ? request.sourceDiagnostic
                                    : (!sourceError.empty()
                                           ? "resident-source-invalid:" + sourceError
                                           : "resident-source-unavailable");
      fillScope(state, &input.scope);
      input.pointRadiusPixels =
          std::clamp(static_cast<float>(state.pointSize), 0.75f, 5.0f);
      input.backgroundR = static_cast<float>(state.backgroundR);
      input.backgroundG = static_cast<float>(state.backgroundG);
      input.backgroundB = static_cast<float>(state.backgroundB);
      input.glossShowOverflow = state.showOverflow ? 1 : 0;
      input.glossNeighborhoodChoice = state.glossNeighborhood;
      input.gloss.algorithm = 0;
      input.gloss.colorMode = 0;
      input.gloss.debugMode = 0;
      input.gloss.diagnosticMode = 0;
      input.gloss.colorSaturation = static_cast<float>(state.colorSaturation);
      input.gloss.glossBodyOpacity = static_cast<float>(state.glossBodyOpacity);
      input.gloss.glossHighlightOpacity =
          static_cast<float>(state.glossHighlightOpacity);
      input.gloss.glossLiftScale = static_cast<float>(state.glossLiftScale);
      if (sourceAvailable &&
          planned.plotModel != ChromaspaceViewer::kPlotModelSourceSignal) {
        const std::vector<LassoStroke>* activeLasso = &emptyLasso;
        if (state.volumeSliceLassoRegion) {
          uint64_t lassoRevision = 0u;
          const bool globalLasso =
              request.workspace->sourceLassoSelectionsSynced ||
              request.workspace->sourceLassoTargetWindowId <= 0;
          if (globalLasso) {
            activeLasso = &request.workspace->sourceLassoStrokes;
            lassoRevision = request.workspace->sourceLassoRevision;
          } else if (request.workspace->sourceLassoTargetWindowId ==
                     window->windowId) {
            activeLasso = &window->viewerLassoStrokes;
            lassoRevision = window->viewerLassoRevision;
          }
          uint64_t contentHash = 1469598103934665603ull;
          contentHash = hashAppend(contentHash, input.viewRevision);
          contentHash = hashAppend(contentHash, globalLasso ? 1u : 2u);
          contentHash = hashAppend(
              contentHash,
              static_cast<uint64_t>(
                  std::max(0, request.workspace->sourceLassoTargetWindowId)));
          contentHash = hashAppend(contentHash, lassoRevision);
          input.contentRevision = contentHash == 0u ? 1u : contentHash;
        }
        input.hasRasterRequest =
            fillRaster(*window, *activeLasso, *request.residentSource,
                       &input.raster);
      }
      if (sourceAvailable) {
        (void)fillCamera(*window, input.targetWidth, input.targetHeight, &input);
        uint64_t glossHash = 1469598103934665603ull;
        glossHash = hashAppend(glossHash, request.residentSource->sourceId);
        glossHash = hashAppend(glossHash, request.residentSource->sequence);
        glossHash = hashAppend(glossHash, request.residentSource->slotGeneration);
        glossHash = hashAppend(glossHash, state.stateRevision);
        glossHash = hashAppend(glossHash, static_cast<uint64_t>(state.glossNeighborhood));
        input.glossDerivationHash = glossHash == 0u ? 1u : glossHash;
      }
      std::string compileError;
      if (!ChromaspaceMetalPlotCompiler::compileAndAppend(
              input, &result.frame, &compileError)) {
        result.frame.clear();
        result.status = CompileStatus::PlotCompileFailed;
        result.rejectedWindowId = planned.windowId;
        result.diagnostic = compileError.empty()
                                ? "workspace-frame-plot-compile-failed"
                                : compileError;
        return result;
      }
    }
    result.status = CompileStatus::Ready;
    result.diagnostic.clear();
    return result;
  } catch (...) {
    result.frame.clear();
    result.status = CompileStatus::AllocationFailure;
    result.diagnostic = "workspace-frame-allocation-failure";
    return result;
  }
}

const char* compileStatusLabel(CompileStatus status) noexcept {
  switch (status) {
    case CompileStatus::Ready: return "ready";
    case CompileStatus::InvalidRequest: return "invalid-request";
    case CompileStatus::InvalidWorkspace: return "invalid-workspace";
    case CompileStatus::InvalidFramePlan: return "invalid-frame-plan";
    case CompileStatus::MissingWorkspaceWindow: return "missing-workspace-window";
    case CompileStatus::PlotCompileFailed: return "plot-compile-failed";
    case CompileStatus::AllocationFailure: return "allocation-failure";
    default: return "unknown";
  }
}

}  // namespace ChromaspaceMetalWorkspaceFrame
