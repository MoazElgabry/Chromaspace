#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

namespace ChromaspaceViewer {

constexpr int kDefaultRec709PrimariesChoice = 9;
constexpr int kDefaultGamma24TransferChoice = 9;

enum PlotModelId {
  kPlotModelCube = 0,
  kPlotModelHsl = 1,
  kPlotModelHsv = 2,
  kPlotModelChen = 3,
  kPlotModelNormCone = 4,
  kPlotModelJpConical = 5,
  kPlotModelReuleaux = 6,
  kPlotModelChromaticity = 7,
  kPlotModelGlossView = 8,
  kPlotModelWaveform = 9,
  kPlotModelHistogram = 10,
  kPlotModelSourceSignal = 11,
  kPlotModelCount = 12,
};

struct ViewerRuntimeState {
  uint64_t stateRevision = 1;
  std::string sampleSettingsKey;
  std::string refreshPolicy = "none";
  bool requiresHostSamples = false;
  std::string sourceSessionId;
  uint64_t hostRefreshRequestedRevision = 0;

  int plotModel = 0;
  bool circularHsl = false;
  bool circularHsv = false;
  bool normConeNormalized = true;
  bool plotDisplayLinear = false;
  int plotDisplayLinearTransfer = kDefaultGamma24TransferChoice;

  bool liveUpdate = true;
  int updateMode = 0;
  int quality = 0;
  int scale = 3;
  int sampling = 0;
  bool occupancyGuidedFill = true;

  int plotStyle = 1;
  int pointShape = 0;
  double pointSize = 1.1;
  double colorSaturation = 2.0;
  bool keepOnTop = true;
  bool resetViewOnPlotSwitch = true;

  bool showOverflow = false;
  bool highlightOverflow = true;
  double overflowHighlightR = 1.0;
  double overflowHighlightG = 0.0;
  double overflowHighlightB = 0.0;
  double backgroundR = 0.08;
  double backgroundG = 0.08;
  double backgroundB = 0.09;

  bool fillVolume = false;
  bool fillGrayRamp = false;
  int fillResolution = 29;
  bool readGrayRamp = false;
  bool readIdentityPlot = false;
  bool isolateIdentityData = false;
  bool excludeIdentityData = false;
  int identityReadResolution = 29;

  bool volumeSliceLassoRegion = false;
  bool volumeSliceRed = false;
  bool volumeSliceYellow = false;
  bool volumeSliceGreen = false;
  bool volumeSliceCyan = false;
  bool volumeSliceBlue = false;
  bool volumeSliceMagenta = false;
  double neutralRadius = 1.0;

  int chromaticityInputPrimaries = kDefaultRec709PrimariesChoice;
  int chromaticityInputTransfer = kDefaultGamma24TransferChoice;
  int chromaticityReferenceBasis = 0;
  int chromaticityOverlayPrimaries = 0;
  bool chromaticityPlanckianLocus = true;
  bool chromaticitySpectralLocus3D = true;

  int glossNeighborhood = 1;
  double glossLiftScale = 1.0;
  bool glossSpatialInset = false;
  double glossBodyOpacity = 1.0;
  double glossHighlightOpacity = 1.0;
  double glossPointCrispness = 0.0;
  bool glossHideText = false;

  int waveformMode = 0;
  bool waveformHighDetail = true;
  bool waveformContinuousHighDetail = true;
  bool waveformHighDetailRequested = false;
  int waveformSampleColumns = 768;
  int waveformSamplesPerColumn = 96;
  double waveformPointBrightness = 1.5;
  double waveformGridBrightness = 1.0;
  double waveformSaturation = 0.75;
  double waveformDotSize = 0.25;
  bool waveformChannelRed = true;
  bool waveformChannelGreen = true;
  bool waveformChannelBlue = true;
  bool waveformShowOverflow = true;
  bool waveformHighlightOverflow = true;
  int waveformLumaMethod = 0;
  int histogramMode = 0;
  bool histogramShowOverflow = true;
  bool histogramHighlightOverflow = true;
  int scopeRangeMode = 0;

  int sourceDetailMode = 0;
  int sourceMaxProxyLongEdge = 2048;
  bool sourceUseNativeWhenAvailable = true;
  bool sourceSyncSelections = false;
};

inline int clampChoice(int value, int hi, int fallback = 0) {
  return value < 0 || value > hi ? fallback : value;
}

inline int clampOverlaySize(int size) {
  return std::max(4, std::min(65, size));
}

inline double clampDouble(double value, double lo, double hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}

inline const char* normalizedRefreshPolicy(const std::string& policy) {
  if (policy == "reinterpret") return "reinterpret";
  if (policy == "resample") return "resample";
  return "none";
}

inline ViewerRuntimeState clampedViewerRuntimeState(ViewerRuntimeState s) {
  s.refreshPolicy = normalizedRefreshPolicy(s.refreshPolicy);
  if (s.refreshPolicy == "resample") s.requiresHostSamples = true;
  s.plotModel = clampChoice(s.plotModel, kPlotModelCount - 1, kPlotModelCube);
  s.plotDisplayLinearTransfer = std::max(0, s.plotDisplayLinearTransfer);
  s.updateMode = clampChoice(s.updateMode, 2, 0);
  s.quality = clampChoice(s.quality, 2, 0);
  s.scale = clampChoice(s.scale, 3, 3);
  s.sampling = clampChoice(s.sampling, 2, 0);
  s.plotStyle = clampChoice(s.plotStyle, 1, 1);
  s.pointShape = clampChoice(s.pointShape, 1, 0);
  s.pointSize = clampDouble(s.pointSize, 0.35, 3.0);
  s.colorSaturation = clampDouble(s.colorSaturation, 0.8, 6.0);
  s.overflowHighlightR = clampDouble(s.overflowHighlightR, 0.0, 1.0);
  s.overflowHighlightG = clampDouble(s.overflowHighlightG, 0.0, 1.0);
  s.overflowHighlightB = clampDouble(s.overflowHighlightB, 0.0, 1.0);
  s.backgroundR = clampDouble(s.backgroundR, 0.0, 1.0);
  s.backgroundG = clampDouble(s.backgroundG, 0.0, 1.0);
  s.backgroundB = clampDouble(s.backgroundB, 0.0, 1.0);
  s.fillResolution = clampOverlaySize(s.fillResolution);
  s.identityReadResolution = clampOverlaySize(s.identityReadResolution);
  s.neutralRadius = clampDouble(s.neutralRadius, 0.0, 1.0);
  s.chromaticityReferenceBasis = clampChoice(s.chromaticityReferenceBasis, 1, 0);
  s.glossNeighborhood = clampChoice(s.glossNeighborhood, 2, 1);
  s.glossLiftScale = clampDouble(s.glossLiftScale, 0.25, 3.0);
  s.glossBodyOpacity = clampDouble(s.glossBodyOpacity, 0.0, 1.0);
  s.glossHighlightOpacity = clampDouble(s.glossHighlightOpacity, 0.0, 1.0);
  s.glossPointCrispness = clampDouble(s.glossPointCrispness, 0.0, 1.0);
  s.waveformMode = clampChoice(s.waveformMode, 2, 0);
  s.waveformSampleColumns = std::max(0, std::min(1536, s.waveformSampleColumns));
  s.waveformSamplesPerColumn = std::max(0, std::min(192, s.waveformSamplesPerColumn));
  s.waveformPointBrightness = clampDouble(s.waveformPointBrightness, 0.1, 2.0);
  s.waveformGridBrightness = clampDouble(s.waveformGridBrightness, 0.0, 2.0);
  s.waveformSaturation = clampDouble(s.waveformSaturation, 0.0, 1.5);
  s.waveformDotSize = clampDouble(s.waveformDotSize, 0.05, 1.5);
  s.waveformLumaMethod = clampChoice(s.waveformLumaMethod, 3, 0);
  s.histogramMode = clampChoice(s.histogramMode, 1, 0);
  s.scopeRangeMode = clampChoice(s.scopeRangeMode, 2, 0);
  s.sourceDetailMode = clampChoice(s.sourceDetailMode, 4, 0);
  s.sourceMaxProxyLongEdge = std::max(768, std::min(4096, s.sourceMaxProxyLongEdge));
  if (s.plotModel == kPlotModelGlossView) {
    s.fillVolume = false;
    s.fillGrayRamp = false;
    s.readGrayRamp = false;
    s.readIdentityPlot = false;
    s.isolateIdentityData = false;
    s.excludeIdentityData = false;
    s.volumeSliceLassoRegion = false;
  }
  if (s.plotModel == kPlotModelWaveform || s.plotModel == kPlotModelHistogram) {
    s.fillVolume = false;
    s.fillGrayRamp = false;
    s.readGrayRamp = false;
    s.readIdentityPlot = false;
    s.isolateIdentityData = false;
    s.excludeIdentityData = false;
    s.occupancyGuidedFill = false;
    s.showOverflow = false;
    s.highlightOverflow = false;
  }
  if (s.plotModel == kPlotModelSourceSignal) {
    s.fillVolume = false;
    s.fillGrayRamp = false;
    s.readGrayRamp = false;
    s.readIdentityPlot = false;
    s.isolateIdentityData = false;
    s.excludeIdentityData = false;
    s.occupancyGuidedFill = false;
    s.showOverflow = false;
    s.highlightOverflow = false;
    s.volumeSliceRed = false;
    s.volumeSliceYellow = false;
    s.volumeSliceGreen = false;
    s.volumeSliceCyan = false;
    s.volumeSliceBlue = false;
    s.volumeSliceMagenta = false;
    s.neutralRadius = 1.0;
  }
  if (!s.readGrayRamp && !s.readIdentityPlot) s.isolateIdentityData = false;
  return s;
}

inline const char* plotModeForModel(int plotModel) {
  switch (clampChoice(plotModel, kPlotModelCount - 1, kPlotModelCube)) {
    case 1: return "hsl";
    case 2: return "hsv";
    case 3: return "chen";
    case 4: return "norm_cone";
    case 5: return "jp_conical";
    case 6: return "reuleaux";
    case 7: return "chromaticity";
    case 8: return "gloss_view";
    case 9: return "waveform";
    case 10: return "histogram";
    case 11: return "source_signal";
    default: return "rgb";
  }
}

inline int plotModelForMode(const std::string& plotMode) {
  if (plotMode == "hsl") return 1;
  if (plotMode == "hsv") return 2;
  if (plotMode == "chen") return 3;
  if (plotMode == "norm_cone") return 4;
  if (plotMode == "jp_conical") return 5;
  if (plotMode == "reuleaux") return 6;
  if (plotMode == "chromaticity") return 7;
  if (plotMode == "gloss_view" || plotMode == "gloss_lift") return 8;
  if (plotMode == "waveform") return 9;
  if (plotMode == "histogram") return 10;
  if (plotMode == "source_signal") return 11;
  return 0;
}

inline const char* plotModelLabel(int plotModel) {
  switch (clampChoice(plotModel, kPlotModelCount - 1, kPlotModelCube)) {
    case 1: return "HSL";
    case 2: return "HSV";
    case 3: return "Chen";
    case 4: return "Norm-Cone";
    case 5: return "JP-Conical";
    case 6: return "Reuleaux";
    case 7: return "Chromaticity";
    case 8: return "Gloss View";
    case 9: return "Waveform";
    case 10: return "Histogram";
    case 11: return "Source Signal";
    default: return "Cube";
  }
}

inline const char* qualityLabel(int quality) {
  switch (clampChoice(quality, 2, 0)) {
    case 1: return "Medium";
    case 2: return "High";
    default: return "Low";
  }
}

inline int qualityResolution(int quality) {
  switch (clampChoice(quality, 2, 0)) {
    case 1: return 41;
    case 2: return 57;
    default: return 25;
  }
}

inline const char* scaleLabel(int scale) {
  switch (clampChoice(scale, 3, 3)) {
    case 0: return "25%";
    case 1: return "50%";
    case 2: return "75%";
    default: return "100%";
  }
}

inline const char* sourceDetailLabel(int mode) {
  switch (clampChoice(mode, 4, 0)) {
    case 1: return "Performance";
    case 2: return "Balanced";
    case 3: return "Quality";
    case 4: return "Native";
    default: return "Auto";
  }
}

inline double scaleFactor(int scale) {
  switch (clampChoice(scale, 3, 3)) {
    case 0: return 0.25;
    case 1: return 0.50;
    case 2: return 0.75;
    default: return 1.00;
  }
}

inline const char* samplingLabel(int sampling) {
  switch (clampChoice(sampling, 2, 0)) {
    case 1: return "Stratified";
    case 2: return "Random";
    default: return "Balanced";
  }
}

inline const char* plotStyleLabel(int plotStyle) {
  return clampChoice(plotStyle, 1, 1) == 1 ? "Space" : "Plain Scope";
}

inline const char* pointShapeLabel(int pointShape) {
  return clampChoice(pointShape, 1, 0) == 1 ? "Square" : "Circle";
}

inline const char* updateModeLabel(int updateMode) {
  switch (clampChoice(updateMode, 2, 0)) {
    case 1: return "Fluid";
    case 2: return "Scheduled";
    default: return "Auto";
  }
}

inline bool isGlossView(const ViewerRuntimeState& state) {
  return clampChoice(state.plotModel, kPlotModelCount - 1, kPlotModelCube) == kPlotModelGlossView;
}

inline const char* waveformModeLabel(int mode) {
  switch (clampChoice(mode, 2, 0)) {
    case 1: return "RGB Parade";
    case 2: return "Luma";
    default: return "RGB Overlay";
  }
}

inline const char* waveformLumaMethodLabel(int method) {
  switch (clampChoice(method, 3, 0)) {
    case 1: return "Rec.2020";
    case 2: return "Rec.601";
    case 3: return "Average";
    default: return "Rec.709";
  }
}

inline const char* histogramModeLabel(int mode) {
  return clampChoice(mode, 1, 0) == 1 ? "Luma" : "RGB Overlay";
}

inline const char* scopeRangeModeLabel(int mode) {
  switch (clampChoice(mode, 2, 0)) {
    case 1: return "0 - 4";
    case 2: return "Auto";
    default: return "0 - 1";
  }
}

inline bool isChromaticity(const ViewerRuntimeState& state) {
  return clampChoice(state.plotModel, kPlotModelCount - 1, kPlotModelCube) == kPlotModelChromaticity;
}

inline bool isAnalyticalScope(const ViewerRuntimeState& state) {
  const int model = clampChoice(state.plotModel, kPlotModelCount - 1, kPlotModelCube);
  return model == kPlotModelWaveform || model == kPlotModelHistogram;
}

inline bool isSourceSignal(const ViewerRuntimeState& state) {
  return clampChoice(state.plotModel, kPlotModelCount - 1, kPlotModelCube) == kPlotModelSourceSignal;
}

inline bool showOverflowSupported(const ViewerRuntimeState& state, bool drawOnImage) {
  if (drawOnImage) return false;
  const int model = clampChoice(state.plotModel, kPlotModelCount - 1, kPlotModelCube);
  return model == 0 || model == 1 || model == 2 || model == 3 ||
         model == 5 || model == 6 || model == 7 || model == 8;
}

inline bool readStripData(const ViewerRuntimeState& state, bool drawOnImage) {
  return !drawOnImage && !isGlossView(state) && !state.excludeIdentityData &&
         (state.readIdentityPlot || state.readGrayRamp);
}

inline bool volumeSlicingSupported(const ViewerRuntimeState& state, bool drawOnImage) {
  return !drawOnImage && !isGlossView(state) && !isSourceSignal(state);
}

inline bool hueSlicingAllowed(const ViewerRuntimeState& state, bool drawOnImage) {
  return volumeSlicingSupported(state, drawOnImage) && !isSourceSignal(state);
}

inline bool neutralRadiusAllowed(const ViewerRuntimeState& state, bool drawOnImage) {
  return volumeSlicingSupported(state, drawOnImage) && !isChromaticity(state) && !isSourceSignal(state);
}

inline bool anyHueSliceSelected(const ViewerRuntimeState& state) {
  return state.volumeSliceRed || state.volumeSliceYellow || state.volumeSliceGreen ||
         state.volumeSliceCyan || state.volumeSliceBlue || state.volumeSliceMagenta;
}

inline bool neutralRadiusEnabled(const ViewerRuntimeState& state, bool drawOnImage) {
  return neutralRadiusAllowed(state, drawOnImage) &&
         state.neutralRadius < 0.999999;
}

inline bool hueSectorSlicingEnabled(const ViewerRuntimeState& state, bool drawOnImage) {
  return hueSlicingAllowed(state, drawOnImage) &&
         anyHueSliceSelected(state);
}

inline bool volumeSlicingEnabled(const ViewerRuntimeState& state, bool drawOnImage, bool ofxLassoHasData) {
  (void)ofxLassoHasData;
  if (!volumeSlicingSupported(state, drawOnImage)) return false;
  return state.volumeSliceLassoRegion ||
         hueSectorSlicingEnabled(state, drawOnImage) ||
         neutralRadiusEnabled(state, drawOnImage);
}

inline std::string sampleSettingsKey(const ViewerRuntimeState& state, bool drawOnImage) {
  const ViewerRuntimeState s = clampedViewerRuntimeState(state);
  const bool imageLassoActive = !drawOnImage && s.volumeSliceLassoRegion;
  const bool readStrip = readStripData(s, drawOnImage) && !imageLassoActive;
  const bool effectiveOccupancyFill = s.occupancyGuidedFill && !readStrip;
  std::ostringstream oss;
  oss << "quality=" << qualityLabel(s.quality)
      << "|resolution=" << qualityResolution(s.quality)
      << "|sampling=" << samplingLabel(s.sampling)
      << "|occupancyFill=" << (effectiveOccupancyFill ? 1 : 0)
      << "|scale=" << scaleLabel(s.scale)
      << "|glossView=" << ((!drawOnImage && isGlossView(s)) ? 1 : 0)
      << "|drawMode=" << (drawOnImage ? 1 : 0)
      << "|useInstance1=" << (readStrip ? 1 : 0)
      << "|showIdentityOnly=" << ((readStrip && s.isolateIdentityData) ? 1 : 0)
      << "|readIdentityPlot=" << ((readStrip && s.readIdentityPlot) ? 1 : 0)
      << "|readGrayRamp=" << ((readStrip && s.readGrayRamp) ? 1 : 0)
      << "|sampleDrawnCubeSize=" << clampOverlaySize(s.identityReadResolution)
      << "|waveformDetail=" << ((!drawOnImage && s.waveformHighDetailRequested) ? 1 : 0)
      << "|waveformColumns=" << ((!drawOnImage && s.waveformHighDetailRequested) ? s.waveformSampleColumns : 0)
      << "|waveformRows=" << ((!drawOnImage && s.waveformHighDetailRequested) ? s.waveformSamplesPerColumn : 0);
  return oss.str();
}

}  // namespace ChromaspaceViewer
