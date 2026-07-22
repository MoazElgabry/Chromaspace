#pragma once

#include "ChromaspaceMetalPlotRenderer.h"
#include "ChromaspaceViewerState.h"

#include <array>
#include <cstdint>
#include <string>

namespace ChromaspaceMetalPlotCompiler {

enum class GlossPresentation : uint8_t { Field2D = 0, Projection3D };

// Values are canonical ChromaspaceViewer::kPlotModel* constants.  The
// compiler deliberately does not duplicate the viewer model enum/table.
struct NormalizedScopeState {
  int waveformMode = 0;              // 0 overlay, 1 parade, 2 luma.
  int histogramMode = 0;             // 0 RGB, 1 luma.
  int waveformChannelRed = 1;
  int waveformChannelGreen = 1;
  int waveformChannelBlue = 1;
  int waveformChannelLuma = 0;
  int waveformShowOverflow = 1;
  int waveformHighlightOverflow = 1;
  int waveformLumaMethod = 0;        // 0..3.
  int histogramShowOverflow = 1;
  int histogramHighlightOverflow = 1;
  int scopeRangeMode = 0;            // 0 [0,1], 1 [0,4], 2 GPU auto.
  float waveformPointBrightness = 0.4f;
  float waveformSaturation = 0.75f;
};

struct GlossControls {
  int algorithm = 0;
  int colorMode = 0;
  int debugMode = 0;
  int diagnosticMode = 0;
  float colorSaturation = 2.0f;
  float glossBodyOpacity = 0.10f;
  float glossHighlightOpacity = 0.42f;
  float glossLiftScale = 1.0f;
};

struct Input {
  int windowId = 0;
  int plotModel = ChromaspaceViewer::kPlotModelCube;
  GlossPresentation glossPresentation = GlossPresentation::Field2D;
  ChromaspaceMetalPlotRenderer::PlotRect destination{};
  int targetWidth = 1;
  int targetHeight = 1;
  int targetPixelFormat = 0;
  uint64_t viewRevision = 1u;
  uint64_t contentRevision = 1u;

  // A missing source emits a visible scaffold.  No CPU pixels or source
  // transport objects are accepted by this module.
  bool sourceAvailable = true;
  std::string unavailableReason;
  bool hasRasterRequest = false;
  ChromaspaceMetal::RasterSourceRequest raster{};

  NormalizedScopeState scope{};
  float pointRadiusPixels = 2.0f;
  float backgroundR = 0.035f;
  float backgroundG = 0.040f;
  float backgroundB = 0.052f;

  int glossGridWidth = 96;
  int glossGridHeight = 96;
  int glossShowOverflow = 0;
  int glossNeighborhoodChoice = 1;
  uint64_t glossDerivationHash = 1u;
  GlossControls gloss{};

  bool hasCameraMatrices = false;
  std::array<float, 16> modelView{};
  std::array<float, 16> projection{};
};

// Compiles and appends exactly one normalized command. Vectors are staged in a
// local batch then appended with the command. On false, commandCount and
// vectorVertexArena.size() remain unchanged. PlotRenderer::prepare remains the
// exhaustive final command validator and owns GPU/source resource policy.
// Guide coordinates use drawable/chart space (+X right, +Y down).
bool compileAndAppend(const Input& input,
                      ChromaspaceMetalPlotRenderer::FrameRequest* frame,
                      std::string* error = nullptr) noexcept;

const char* plotModelLabel(int plotModel) noexcept;

}  // namespace ChromaspaceMetalPlotCompiler
