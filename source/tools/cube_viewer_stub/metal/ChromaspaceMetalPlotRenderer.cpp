#include "ChromaspaceMetalPlotRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace ChromaspaceMetalPlotRenderer {
namespace {

void setError(std::string* error, const char* message) {
  if (error) *error = message != nullptr ? message : "plot-renderer-error";
}

void setError(std::string* error, const std::string& message) {
  if (error) *error = message;
}

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "plot-renderer-error";
  } catch (...) {
  }
}

bool finite(float value) noexcept { return std::isfinite(value); }

void hashWord(uint64_t value, uint64_t* hash) noexcept {
  if (!hash) return;
  for (unsigned byte = 0u; byte < 8u; ++byte) {
    *hash ^= (value >> (byte * 8u)) & 0xffu;
    *hash *= 1099511628211ull;
  }
}

void hashInt(int value, uint64_t* hash) noexcept {
  hashWord(static_cast<uint64_t>(static_cast<uint32_t>(value)), hash);
}

void hashFloat(float value, uint64_t* hash) noexcept {
  uint32_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value), "float hash width");
  std::memcpy(&bits, &value, sizeof(bits));
  hashWord(static_cast<uint64_t>(bits), hash);
}

void hashScopeRasterDerivation(
    const ChromaspaceMetal::RasterSourceRequest& raster,
    uint64_t* hash) noexcept {
  hashInt(raster.pointCount, hash);
  hashInt(raster.basePointCount, hash);
  hashInt(raster.sourceWidth, hash);
  hashInt(raster.sourceHeight, hash);
  hashInt(raster.sampleStride, hash);
  hashInt(raster.sampleCountX, hash);
  hashInt(raster.pixelFormat, hash);
  hashInt(raster.plotLinear, hash);
  hashInt(raster.plotLinearTransfer, hash);
  hashInt(raster.excludeIdentityData, hash);
  hashInt(raster.isolateIdentityData, hash);
  hashInt(raster.readIdentityPlot, hash);
  hashInt(raster.readGrayRamp, hash);
  hashInt(raster.identityCubeY1, hash);
  hashInt(raster.identityCubeY2, hash);
  hashInt(raster.identityRampY1, hash);
  hashInt(raster.identityRampY2, hash);
  hashInt(raster.identityCubeAppendOffset, hash);
  hashInt(raster.identityCubeAppendCount, hash);
  hashInt(raster.identityCubeAppendY1, hash);
  hashInt(raster.identityCubeAppendY2, hash);
  hashInt(raster.identityCubeAppendRowStep, hash);
  hashInt(raster.identityCubeAppendXStep, hash);
  hashInt(raster.identityRampAppendOffset, hash);
  hashInt(raster.identityRampAppendCount, hash);
  hashInt(raster.identityRampAppendY1, hash);
  hashInt(raster.identityRampAppendY2, hash);
  hashInt(raster.identityRampAppendRowStep, hash);
  hashInt(raster.identityRampAppendXStep, hash);
  hashInt(raster.lassoEnabled, hash);
  hashInt(raster.lassoStrokeCount, hash);
  hashInt(raster.lassoPointCount, hash);
  for (int index = 0; index < 16; ++index) {
    hashInt(raster.lassoStrokeFirst[index], hash);
    hashInt(raster.lassoStrokeCountPerStroke[index], hash);
    hashInt(raster.lassoStrokeSubtract[index], hash);
  }
  for (int index = 0; index < 256; ++index) {
    hashFloat(raster.lassoX[index], hash);
    hashFloat(raster.lassoY[index], hash);
  }
  hashInt(raster.cubeSlicingEnabled, hash);
  hashInt(raster.neutralRadiusEnabled, hash);
  hashFloat(raster.neutralRadius, hash);
  hashInt(raster.cubeSliceRed, hash);
  hashInt(raster.cubeSliceYellow, hash);
  hashInt(raster.cubeSliceGreen, hash);
  hashInt(raster.cubeSliceCyan, hash);
  hashInt(raster.cubeSliceBlue, hash);
  hashInt(raster.cubeSliceMagenta, hash);
  // Scope sampling uses plotMode only to decide whether transfer decoding is
  // meaningful. Presentation-only remap/color fields do not invalidate the
  // resident analytical density.
  hashInt(raster.remap.plotMode, hash);
}

void hashPointRasterDerivation(
    const ChromaspaceMetal::RasterSourceRequest& raster,
    uint64_t* hash) noexcept {
  // Begin with the complete sampling/selection topology, then include every
  // remap/color input consumed while generating resident point positions and
  // colors. Camera, target dimensions, background, radius, and vector guides
  // are presentation-only and deliberately absent.
  hashScopeRasterDerivation(raster, hash);
  hashFloat(raster.sourceAspect, hash);
  hashFloat(raster.glossLiftScale, hash);
  hashFloat(raster.pointAlphaScale, hash);
  hashFloat(raster.denseAlphaBias, hash);
  hashFloat(raster.colorSaturation, hash);
  hashInt(raster.remap.plotMode, hash);
  hashInt(raster.remap.circularHsl, hash);
  hashInt(raster.remap.circularHsv, hash);
  hashInt(raster.remap.normConeNormalized, hash);
  hashInt(raster.remap.showOverflow, hash);
  hashInt(raster.remap.highlightOverflow, hash);
  hashInt(raster.remap.chromaticityInputTransfer, hash);
  hashInt(raster.remap.chromaticityReferenceBasis, hash);
  hashFloat(raster.remap.chromaticityWhiteX, hash);
  hashFloat(raster.remap.chromaticityWhiteY, hash);
  for (int index = 0; index < 9; ++index) {
    hashFloat(raster.remap.chromaticityRgbToXyz[index], hash);
    hashFloat(raster.remap.chromaticityXyzToRgb[index], hash);
  }
}

void hashAutoRangeDerivation(const ChromaspaceMetal::ScopeRangeRequest& range,
                             uint64_t* hash) noexcept {
  hashInt(range.includeRed, hash);
  hashInt(range.includeGreen, hash);
  hashInt(range.includeBlue, hash);
  hashInt(range.includeLuma, hash);
  hashInt(range.previousRangeValid, hash);
  hashFloat(range.previousRangeMin, hash);
  hashFloat(range.previousRangeMax, hash);
}

bool scopeDerivationHash(const PlotCommand& command,
                         uint64_t* outHash) noexcept {
  if (!outHash || (command.kind != PlotKind::Histogram &&
                   command.kind != PlotKind::Waveform)) {
    return false;
  }
  uint64_t hash = 1469598103934665603ull;
  // Version the structural key so future shader semantics can invalidate all
  // old derivations without changing cache ownership.
  hashWord(1u, &hash);
  hashScopeRasterDerivation(command.raster, &hash);
  if (command.kind == PlotKind::Histogram) {
    const auto& scope = command.histogram;
    hashWord(0x484953544f475241ull, &hash);  // "HISTOGRA"
    hashInt(scope.scopeMode, &hash);
    hashInt(scope.width, &hash);
    hashFloat(scope.rangeMin, &hash);
    hashFloat(scope.invRange, &hash);
    hashInt(scope.showOverflow, &hash);
    hashInt(scope.lumaMethod, &hash);
    hashInt(scope.useGpuAutoRange, &hash);
    if (scope.useGpuAutoRange != 0) {
      hashAutoRangeDerivation(scope.autoRange, &hash);
    }
  } else {
    const auto& scope = command.waveform;
    hashWord(0x57415645464f524dull, &hash);  // "WAVEFORM"
    hashInt(scope.scopeMode, &hash);
    hashInt(scope.width, &hash);
    hashInt(scope.height, &hash);
    hashFloat(scope.rangeMin, &hash);
    hashFloat(scope.invRange, &hash);
    hashInt(scope.showOverflow, &hash);
    hashInt(scope.lumaMethod, &hash);
    hashInt(scope.includeLuma, &hash);
    hashInt(scope.useGpuAutoRange, &hash);
    if (scope.useGpuAutoRange != 0) {
      hashInt(scope.includeRed, &hash);
      hashInt(scope.includeGreen, &hash);
      hashInt(scope.includeBlue, &hash);
      hashAutoRangeDerivation(scope.autoRange, &hash);
    }
  }
  *outHash = hash != 0u ? hash : 1u;
  return true;
}

bool multiplyExact(uint64_t left, uint64_t right, uint64_t* result) noexcept {
  if (!result || (left != 0u && right >
                                   std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *result = left * right;
  return true;
}

bool pointDerivationHash(const PlotCommand& command,
                         uint64_t* outHash) noexcept {
  if (!outHash || command.kind != PlotKind::ResidentRaster) return false;
  uint64_t hash = 1469598103934665603ull;
  hashWord(1u, &hash);
  hashWord(0x504f494e54434c44ull, &hash);  // "POINTCLD"
  hashPointRasterDerivation(command.raster, &hash);
  *outHash = hash != 0u ? hash : 1u;
  return true;
}

bool pointByteEstimate(const PlotCommand& command,
                       uint64_t* outBytes) noexcept {
  if (!outBytes || command.kind != PlotKind::ResidentRaster ||
      command.raster.pointCount <= 0) {
    return false;
  }
  constexpr uint64_t kResidentBytesPerPoint = 12u + 16u;
  constexpr uint64_t kIndirectArgumentBytes = 4u * sizeof(uint32_t);
  uint64_t pointBytes = 0u;
  if (!multiplyExact(static_cast<uint64_t>(command.raster.pointCount),
                     kResidentBytesPerPoint, &pointBytes) ||
      pointBytes >
          std::numeric_limits<uint64_t>::max() - kIndirectArgumentBytes) {
    return false;
  }
  *outBytes = pointBytes + kIndirectArgumentBytes;
  return true;
}

bool glossByteEstimate(const PlotCommand& command,
                       uint64_t* outBytes) noexcept {
  if (!outBytes || (command.kind != PlotKind::GlossField2D &&
                    command.kind != PlotKind::GlossProjection3D) ||
      command.glossField.gridWidth <= 0 || command.glossField.gridHeight <= 0) {
    return false;
  }
  uint64_t cellCount = 0u;
  if (!multiplyExact(static_cast<uint64_t>(command.glossField.gridWidth),
                     static_cast<uint64_t>(command.glossField.gridHeight),
                     &cellCount)) {
    return false;
  }
  // The retained native Gloss record contains 21 float buffers. Scratch
  // buffers and the build-only reduction are not resident.
  constexpr uint64_t kResidentGlossBuffers = 21u;
  constexpr uint64_t kGlossBytesPerCell =
      kResidentGlossBuffers * sizeof(float);
  if (!multiplyExact(cellCount, kGlossBytesPerCell, outBytes)) return false;
  return *outBytes != 0u;
}

void hashGlossSourceFingerprint(
    const ChromaspaceMetal::ImportedSourceTexture& source,
    uint64_t* hash) noexcept {
  // The structural key carries this tuple too, but hashing the complete
  // authoritative publication fingerprint prevents a coarse caller hash from
  // becoming the sole identity input.
  hashWord(source.sourceId, hash);
  hashWord(source.deviceRegistryId, hash);
  hashWord(source.senderGeneration, hash);
  hashWord(source.sequence, hash);
  hashWord(static_cast<uint64_t>(source.slotIndex), hash);
  hashWord(source.slotGeneration, hash);
  hashWord(source.contentHash, hash);
  hashInt(source.width, hash);
  hashInt(source.height, hash);
  hashInt(source.pixelFormat, hash);
  hashWord(static_cast<uint64_t>(source.bytesPerRow), hash);
  hashWord(static_cast<uint64_t>(source.byteSize), hash);
}

bool glossDerivationHash(
    const ChromaspaceMetal::ImportedSourceTexture& source,
    const PlotCommand& command,
    uint64_t* outHash) noexcept {
  if (!outHash || (command.kind != PlotKind::GlossField2D &&
                   command.kind != PlotKind::GlossProjection3D)) {
    return false;
  }
  uint64_t hash = 1469598103934665603ull;
  // Version the Gloss field shader/record contract independently of the
  // externally supplied coarse glossDerivationHash.
  hashWord(2u, &hash);
  hashWord(0x474c4f5353464945ull, &hash);  // "GLOSSFIE"
  hashGlossSourceFingerprint(source, &hash);
  hashScopeRasterDerivation(command.raster, &hash);
  hashInt(command.glossField.gridWidth, &hash);
  hashInt(command.glossField.gridHeight, &hash);
  hashInt(command.glossField.showOverflow, &hash);
  hashInt(command.glossField.neighborhoodChoice, &hash);
  *outHash = hash != 0u ? hash : 1u;
  return true;
}

bool scopeByteEstimate(const PlotCommand& command,
                       uint64_t* outBytes) noexcept {
  if (!outBytes) return false;
  uint64_t count = 0u;
  bool overflow = false;
  bool autoRange = false;
  if (command.kind == PlotKind::Histogram) {
    const uint64_t channels = command.histogram.scopeMode == 1 ? 1u : 3u;
    if (command.histogram.width <= 0 ||
        !multiplyExact(static_cast<uint64_t>(command.histogram.width),
                       channels, &count)) {
      return false;
    }
    overflow = command.histogram.showOverflow != 0;
    autoRange = command.histogram.useGpuAutoRange != 0;
  } else if (command.kind == PlotKind::Waveform) {
    if (command.waveform.width <= 0 || command.waveform.height <= 0) return false;
    const bool lumaOnly = command.waveform.scopeMode == 2;
    const bool paradeLuma = command.waveform.scopeMode == 1 &&
                            command.waveform.includeLuma != 0;
    const uint64_t channels = lumaOnly ? 1u : (paradeLuma ? 4u : 3u);
    uint64_t pixels = 0u;
    if (!multiplyExact(static_cast<uint64_t>(command.waveform.width),
                       static_cast<uint64_t>(command.waveform.height),
                       &pixels) ||
        !multiplyExact(pixels, channels, &count)) {
      return false;
    }
    overflow = command.waveform.showOverflow != 0;
    autoRange = command.waveform.useGpuAutoRange != 0;
  } else {
    return false;
  }
  uint64_t bytes = 0u;
  if (!multiplyExact(count, sizeof(uint32_t), &bytes)) return false;
  if (overflow) {
    if (bytes > (std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)) / 2u) {
      return false;
    }
    bytes *= 2u;
  }
  const uint64_t fixedBytes = sizeof(uint32_t) +
                              (autoRange ? 3u * sizeof(uint32_t) : 0u);
  if (bytes > std::numeric_limits<uint64_t>::max() - fixedBytes) {
    return false;
  }
  *outBytes = bytes + fixedBytes;
  return *outBytes != 0u;
}

ChromaspaceMetalDerivedCache::DerivedKey makeDerivedKey(
    const FrameRequest& request,
    const PlotCommand& command) noexcept {
  ChromaspaceMetalDerivedCache::DerivedKey key{};
  const auto& source = request.residentSource;
  key.sourceId = source.sourceId;
  key.deviceRegistryId = source.deviceRegistryId;
  key.senderGeneration = source.senderGeneration;
  key.sequence = source.sequence;
  key.slotIndex = source.slotIndex;
  key.slotGeneration = source.slotGeneration;
  key.contentHash = source.contentHash;
  if (command.kind == PlotKind::Histogram) {
    key.family = ChromaspaceMetalDerivedCache::Family::Histogram;
    scopeDerivationHash(command, &key.derivationHash);
  } else if (command.kind == PlotKind::Waveform) {
    key.family = ChromaspaceMetalDerivedCache::Family::Waveform;
    scopeDerivationHash(command, &key.derivationHash);
  } else if (command.kind == PlotKind::ResidentRaster) {
    key.family = ChromaspaceMetalDerivedCache::Family::RasterPointCloud;
    pointDerivationHash(command, &key.derivationHash);
  } else {
    key.family = ChromaspaceMetalDerivedCache::Family::GlossField;
    glossDerivationHash(source, command, &key.derivationHash);
  }
  return key;
}

bool finiteColor(const std::array<float, 4>& color) noexcept {
  return finite(color[0]) && finite(color[1]) && finite(color[2]) &&
         finite(color[3]) && color[0] >= 0.0f && color[0] <= 1.0f &&
         color[1] >= 0.0f && color[1] <= 1.0f && color[2] >= 0.0f &&
         color[2] <= 1.0f && color[3] >= 0.0f && color[3] <= 1.0f;
}

bool finiteRect(const PlotRect& rect) noexcept {
  return finite(rect.x) && finite(rect.y) && finite(rect.width) &&
         finite(rect.height) && rect.width > 0.0f && rect.height > 0.0f &&
         rect.x >= -kMaximumPlotCoordinate &&
         rect.x <= kMaximumPlotCoordinate &&
         rect.y >= -kMaximumPlotCoordinate &&
         rect.y <= kMaximumPlotCoordinate &&
         rect.width <= kMaximumPlotCoordinate &&
         rect.height <= kMaximumPlotCoordinate;
}

bool finiteMatrix(const float* matrix, std::size_t count) noexcept {
  if (!matrix) return false;
  for (std::size_t index = 0; index < count; ++index) {
    if (!finite(matrix[index]) ||
        std::fabs(matrix[index]) > kMaximumPlotCoordinate) {
      return false;
    }
  }
  return true;
}

bool validTarget(const PlotCommand& command) noexcept {
  return command.targetWidth > 0 && command.targetHeight > 0 &&
         command.targetWidth <= kMaximumPlotDimension &&
         command.targetHeight <= kMaximumPlotDimension &&
         (command.targetPixelFormat == 0 || command.targetPixelFormat == 1);
}

bool flag01(int value) noexcept { return value == 0 || value == 1; }

bool validRemap(const ChromaspaceMetal::RemapUniforms& remap) noexcept {
  if (!flag01(remap.circularHsl) || !flag01(remap.circularHsv) ||
      !flag01(remap.normConeNormalized) || !flag01(remap.showOverflow) ||
      !flag01(remap.highlightOverflow) ||
      remap.plotMode < 0 || remap.plotMode > 9 ||
      remap.chromaticityInputTransfer < 0 ||
      remap.chromaticityInputTransfer > 17 ||
      remap.chromaticityReferenceBasis < 0 ||
      remap.chromaticityReferenceBasis > 1 ||
      !finite(remap.chromaticityWhiteX) ||
      !finite(remap.chromaticityWhiteY) || remap.chromaticityWhiteX < 0.0f ||
      remap.chromaticityWhiteX > 1.0f || remap.chromaticityWhiteY < 0.0f ||
      remap.chromaticityWhiteY > 1.0f ||
      !finiteMatrix(remap.chromaticityRgbToXyz, 9u) ||
      !finiteMatrix(remap.chromaticityXyzToRgb, 9u)) {
    return false;
  }
  return true;
}

bool validPlotKindValue(PlotKind kind) noexcept {
  const int value = static_cast<int>(kind);
  return value >= static_cast<int>(PlotKind::SourceSignal) &&
         value <= static_cast<int>(PlotKind::Scaffold);
}

bool plotKindNeedsSource(PlotKind kind) noexcept {
  return kind != PlotKind::Scaffold;
}

bool validRasterRequest(const ChromaspaceMetal::RasterSourceRequest& request) noexcept {
  if (request.pointCount < 0 || request.basePointCount < 0 ||
      request.sourceWidth < 0 || request.sourceHeight < 0 ||
      request.sampleStride <= 0 || request.sampleCountX < 0 ||
      (request.pixelFormat != 0 && request.pixelFormat != 1) ||
      !finite(request.sourceAspect) || request.sourceAspect <= 0.0f ||
      request.sourceAspect > kMaximumPlotCoordinate ||
      !finite(request.glossLiftScale) || !finite(request.pointAlphaScale) ||
      !finite(request.denseAlphaBias) || !finite(request.colorSaturation)) {
    return false;
  }
  if (request.pointCount > kMaximumResidentRasterPoints ||
      request.basePointCount > kMaximumResidentRasterPoints ||
      request.sourceWidth > kMaximumPlotDimension ||
      request.sourceHeight > kMaximumPlotDimension ||
      request.lassoStrokeCount < 0 || request.lassoStrokeCount > 16 ||
      request.lassoPointCount < 0 || request.lassoPointCount > 256 ||
      !validRemap(request.remap)) {
    return false;
  }
  if (!flag01(request.plotLinear) || request.plotLinearTransfer < 0 ||
      request.plotLinearTransfer > 17 ||
      !flag01(request.excludeIdentityData) ||
      !flag01(request.isolateIdentityData) || !flag01(request.readIdentityPlot) ||
      !flag01(request.readGrayRamp) || !flag01(request.occupancyFill) ||
      !flag01(request.cubeSlicingEnabled) ||
      !flag01(request.neutralRadiusEnabled) || !flag01(request.cubeSliceRed) ||
      !flag01(request.cubeSliceYellow) || !flag01(request.cubeSliceGreen) ||
      !flag01(request.cubeSliceCyan) || !flag01(request.cubeSliceBlue) ||
      !flag01(request.cubeSliceMagenta) || request.identityCubeAppendOffset < 0 ||
      request.identityCubeAppendCount < 0 ||
      request.identityCubeAppendCount > kMaximumResidentRasterPoints ||
      request.identityRampAppendOffset < 0 || request.identityRampAppendCount < 0 ||
      request.identityRampAppendCount > kMaximumResidentRasterPoints ||
      request.occupancyAppendOffset < 0 || request.occupancyAppendCount < 0 ||
      request.occupancyCandidateCount < 0 ||
      request.occupancyAppendCount > kMaximumResidentRasterPoints ||
      request.occupancyCandidateCount > kMaximumResidentRasterPoints ||
      request.lassoEnabled < 0 || request.lassoEnabled > 1 ||
      request.neutralRadius < 0.0f || !finite(request.neutralRadius)) {
    return false;
  }
  if (request.basePointCount > request.pointCount || request.pointCount == 0 ||
      request.sampleCountX <= 0 || request.sourceWidth <= 0 ||
      request.sourceHeight <= 0) {
    return false;
  }
  const int maximumSampleCountX =
      1 + (request.sourceWidth - 1) / request.sampleStride;
  if (request.sampleCountX > maximumSampleCountX) return false;
  const auto validRange = [pointCount = request.pointCount](int offset,
                                                              int count) noexcept {
    return offset >= 0 && count >= 0 && offset <= pointCount &&
           count <= pointCount - offset;
  };
  if (!validRange(request.identityCubeAppendOffset,
                  request.identityCubeAppendCount) ||
      !validRange(request.identityRampAppendOffset,
                  request.identityRampAppendCount) ||
      !validRange(request.occupancyAppendOffset,
                  request.occupancyAppendCount)) {
    return false;
  }
  const auto validSentinelRange = [sourceHeight = request.sourceHeight](int y1,
                                                                         int y2) noexcept {
    if (y1 == -1 || y2 == -1) return y1 == -1 && y2 == -1;
    return y1 >= 0 && y2 > y1 && y2 <= sourceHeight;
  };
  if (!validSentinelRange(request.identityCubeY1, request.identityCubeY2) ||
      !validSentinelRange(request.identityRampY1, request.identityRampY2) ||
      !validSentinelRange(request.identityCubeAppendY1,
                          request.identityCubeAppendY2) ||
      !validSentinelRange(request.identityRampAppendY1,
                          request.identityRampAppendY2)) {
    return false;
  }
  for (int index = 0; index < 16; ++index) {
    if (!flag01(request.lassoStrokeSubtract[index]) ||
        request.lassoStrokeFirst[index] < 0 ||
        request.lassoStrokeCountPerStroke[index] < 0 ||
        request.lassoStrokeFirst[index] > request.lassoPointCount ||
        request.lassoStrokeCountPerStroke[index] >
            request.lassoPointCount - request.lassoStrokeFirst[index]) {
      return false;
    }
  }
  for (int index = 0; index < 256; ++index) {
    if (!finite(request.lassoX[index]) || !finite(request.lassoY[index]) ||
        std::fabs(request.lassoX[index]) > kMaximumPlotCoordinate ||
        std::fabs(request.lassoY[index]) > kMaximumPlotCoordinate) {
      return false;
    }
  }
  if (request.identityCubeAppendRowStep <= 0 ||
      request.identityCubeAppendXStep <= 0 ||
      request.identityRampAppendRowStep <= 0 ||
      request.identityRampAppendXStep <= 0) {
    return false;
  }
  return true;
}

bool validHistogramRequest(
    const ChromaspaceMetal::HistogramSurfaceRequest& request) noexcept {
  const auto& range = request.autoRange;
  return request.pointCount >= 0 && request.pointCount <=
             kMaximumResidentRasterPoints &&
         request.width > 0 && request.width <= kMaximumPlotDimension &&
         request.height > 0 && request.height <= kMaximumPlotDimension &&
         finite(request.rangeMin) && finite(request.invRange) &&
         request.invRange >= 0.0f && flag01(request.showOverflow) &&
         flag01(request.highlightOverflow) && request.scopeMode >= 0 &&
         request.scopeMode <= 1 && request.lumaMethod >= 0 &&
         request.lumaMethod <= 3 && finite(request.autoRange.previousRangeMin) &&
         finite(request.autoRange.previousRangeMax) &&
         flag01(request.useGpuAutoRange) && range.pointCount >= 0 &&
         range.pointCount <= kMaximumResidentRasterPoints && flag01(range.waveform) &&
         (!request.useGpuAutoRange ||
          (range.waveform == 0 && range.pointCount == request.pointCount)) &&
         range.scopeMode >= 0 && range.scopeMode <= 1 &&
         flag01(range.includeRed) && flag01(range.includeGreen) &&
         flag01(range.includeBlue) && flag01(range.includeLuma) &&
         flag01(range.includeOverflow) && range.lumaMethod >= 0 &&
         range.lumaMethod <= 3 && flag01(range.previousRangeValid) &&
         range.previousRangeMin <= range.previousRangeMax;
}

bool validWaveformRequest(
    const ChromaspaceMetal::WaveformSurfaceRequest& request) noexcept {
  const auto& range = request.autoRange;
  return request.pointCount >= 0 && request.pointCount <=
             kMaximumResidentRasterPoints &&
         request.width > 0 && request.width <= kMaximumPlotDimension &&
         request.height > 0 && request.height <= kMaximumPlotDimension &&
         finite(request.rangeMin) && finite(request.invRange) &&
         request.invRange >= 0.0f && flag01(request.showOverflow) &&
         flag01(request.highlightOverflow) && request.scopeMode >= 0 &&
         request.scopeMode <= 2 && request.lumaMethod >= 0 &&
         request.lumaMethod <= 3 && flag01(request.includeRed) &&
         flag01(request.includeGreen) && flag01(request.includeBlue) &&
         flag01(request.includeLuma) && finite(request.pointBrightness) &&
         finite(request.colorSaturation) && finite(request.coverageAlpha) &&
         request.coverageAlpha >= 0.0f && flag01(request.useGpuAutoRange) &&
         range.pointCount >= 0 && range.pointCount <= kMaximumResidentRasterPoints &&
         flag01(range.waveform) && range.scopeMode >= 0 && range.scopeMode <= 2 &&
         (!request.useGpuAutoRange ||
          (range.waveform == 1 && range.pointCount == request.pointCount)) &&
         flag01(range.includeRed) && flag01(range.includeGreen) &&
         flag01(range.includeBlue) && flag01(range.includeLuma) &&
         flag01(range.includeOverflow) && range.lumaMethod >= 0 &&
         range.lumaMethod <= 3 && flag01(range.previousRangeValid) &&
         finite(range.previousRangeMin) && finite(range.previousRangeMax) &&
         range.previousRangeMin <= range.previousRangeMax;
}

bool validPointRequest(
    const ChromaspaceMetal::RasterPointSurfaceRequest& request) noexcept {
  if (request.pointCount < 0 || request.pointCount > kMaximumResidentRasterPoints ||
      request.width <= 0 || request.width > kMaximumPlotDimension ||
      request.height <= 0 || request.height > kMaximumPlotDimension ||
      !finite(request.pointRadiusPixels) || request.pointRadiusPixels < 0.0f ||
      !finite(request.backgroundR) || !finite(request.backgroundG) ||
      !finite(request.backgroundB) || !finite(request.backgroundA) ||
      request.backgroundR < 0.0f || request.backgroundR > 1.0f ||
      request.backgroundG < 0.0f || request.backgroundG > 1.0f ||
      request.backgroundB < 0.0f || request.backgroundB > 1.0f ||
      request.backgroundA < 0.0f || request.backgroundA > 1.0f) {
    return false;
  }
  return finiteMatrix(request.modelView, 16u) &&
         finiteMatrix(request.projection, 16u);
}

bool validGlossFieldRequest(
    const ChromaspaceMetal::GlossFieldRequest& request) noexcept {
  return request.gridWidth > 0 && request.gridHeight > 0 &&
         request.gridWidth <= kMaximumPlotDimension &&
         request.gridHeight <= kMaximumPlotDimension &&
         request.showOverflow >= 0 && request.showOverflow <= 1 &&
         request.neighborhoodChoice >= 0 && request.neighborhoodChoice <= 2;
}

bool validGlossFieldSurfaceRequest(
    const ChromaspaceMetal::GlossFieldSurfaceRequest& request) noexcept {
  return request.width > 0 && request.width <= kMaximumPlotDimension &&
         request.height > 0 && request.height <= kMaximumPlotDimension &&
         request.algorithm >= 0 && request.algorithm <= 1 &&
         request.colorMode >= 0 && request.colorMode <= 1 &&
         request.debugMode >= 0 && request.debugMode <= 4 &&
         request.diagnosticMode >= 0 && request.diagnosticMode <= 2 && finite(request.colorSaturation) &&
         finite(request.glossBodyOpacity) &&
         finite(request.glossHighlightOpacity) && finite(request.glossLiftScale);
}

bool validGlossProjectionRequest(
    const ChromaspaceMetal::GlossProjectionSurfaceRequest& request) noexcept {
  return request.width > 0 && request.width <= kMaximumPlotDimension &&
         request.height > 0 && request.height <= kMaximumPlotDimension &&
         request.algorithm >= 0 && request.algorithm <= 1 &&
         request.colorMode >= 0 && request.colorMode <= 1 &&
         request.debugMode >= 0 && request.debugMode <= 4 &&
         request.diagnosticMode >= 0 && request.diagnosticMode <= 2 && finite(request.sourceAspect) &&
         request.sourceAspect > 0.0f && finite(request.colorSaturation) &&
         finite(request.glossBodyOpacity) &&
         finite(request.glossHighlightOpacity) && finite(request.glossLiftScale) &&
         finite(request.pointRadiusPixels) && request.pointRadiusPixels >= 0.0f &&
         finiteMatrix(request.modelView, 16u) &&
         finiteMatrix(request.projection, 16u);
}

bool validCommand(const PlotCommand& command,
                  const FrameRequest& frame,
                  std::string* error) {
  if (command.windowId <= 0 || !validPlotKindValue(command.kind) ||
      !finiteRect(command.destination) || !validTarget(command) ||
      command.viewRevision == 0u || command.contentRevision == 0u ||
      command.vectorVertexCount > kMaximumCommandVectorVertices ||
      command.vectorVertexCount % 3u != 0u ||
      command.vectorVertexOffset > frame.vectorVertexArena.size() ||
      command.vectorVertexCount >
          frame.vectorVertexArena.size() - command.vectorVertexOffset ||
      !finiteColor(command.vectorClearColor) ||
      command.unavailableReason.size() > 512u) {
    setError(error, "plot-command-invalid");
    return false;
  }
  if (command.kind == PlotKind::Scaffold) {
    if (command.vectorVertexCount == 0u || command.unavailableReason.empty()) {
      setError(error, "plot-scaffold-reason-or-geometry-missing");
      return false;
    }
  } else if (!command.unavailableReason.empty()) {
    setError(error, "plot-unavailable-reason-on-resident-command");
    return false;
  }
  if (command.kind == PlotKind::GlossField2D ||
      command.kind == PlotKind::GlossProjection3D) {
    if (!validGlossFieldRequest(command.glossField) ||
        !validRasterRequest(command.raster)) {
      setError(error, "plot-gloss-request-invalid");
      return false;
    }
    if (command.kind == PlotKind::GlossField2D &&
        !validGlossFieldSurfaceRequest(command.glossFieldSurface)) {
      setError(error, "plot-gloss-field-surface-invalid");
      return false;
    }
    if (command.kind == PlotKind::GlossProjection3D &&
        !validGlossProjectionRequest(command.glossProjectionSurface)) {
      setError(error, "plot-gloss-projection-surface-invalid");
      return false;
    }
  } else if (command.kind == PlotKind::SourceSignal) {
    // Source-signal encoding consumes only the imported source handle and
    // target surface; no analytical point descriptor is required.
  } else if (command.kind == PlotKind::Histogram) {
    if (!validRasterRequest(command.raster) ||
        !validHistogramRequest(command.histogram)) {
      setError(error, "plot-histogram-request-invalid");
      return false;
    }
  } else if (command.kind == PlotKind::Waveform) {
    if (!validRasterRequest(command.raster) ||
        !validWaveformRequest(command.waveform)) {
      setError(error, "plot-waveform-request-invalid");
      return false;
    }
  } else if (command.kind == PlotKind::ResidentRaster) {
    if (!validRasterRequest(command.raster) ||
        !validPointRequest(command.point) ||
        command.point.pointCount != command.raster.pointCount ||
        command.point.width != command.targetWidth ||
        command.point.height != command.targetHeight) {
      setError(error, "plot-raster-request-invalid");
      return false;
    }
  }
  for (std::size_t index = command.vectorVertexOffset;
       index < command.vectorVertexOffset + command.vectorVertexCount; ++index) {
    const ChromaspaceMetal::FrameVectorVertex& vertex =
        frame.vectorVertexArena[index];
    if (!finite(vertex.x) || !finite(vertex.y) || !finite(vertex.r) ||
        !finite(vertex.g) || !finite(vertex.b) || !finite(vertex.a) ||
        vertex.r < 0.0f || vertex.r > 1.0f || vertex.g < 0.0f ||
        vertex.g > 1.0f || vertex.b < 0.0f || vertex.b > 1.0f ||
        vertex.a < 0.0f || vertex.a > 1.0f) {
      setError(error, "plot-vector-vertex-invalid");
      return false;
    }
  }
  if (plotKindNeedsSource(command.kind) && !frame.hasResidentSource) {
    setError(error, "plot-resident-source-required");
    return false;
  }
  if (plotKindNeedsSource(command.kind) && command.kind != PlotKind::SourceSignal &&
      (command.raster.sourceWidth <= 0 || command.raster.sourceHeight <= 0)) {
    setError(error, "plot-source-dimensions-missing");
    return false;
  }
  if (plotKindNeedsSource(command.kind) && command.kind != PlotKind::SourceSignal &&
      (command.raster.sourceWidth != frame.residentSource.width ||
       command.raster.sourceHeight != frame.residentSource.height)) {
    setError(error, "plot-source-dimensions-mismatch");
    return false;
  }
  return true;
}

#if !defined(__APPLE__)
bool unavailableCreate(void*, uint64_t, int, int, int,
                       ChromaspaceMetal::PlotSurface*, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
void unavailableRelease(void*, uint64_t, uint32_t) noexcept {}
bool unavailableSource(void*, const ChromaspaceMetal::FrameSubmission&, uint64_t,
                       uint32_t, int, int, int, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableHistogram(void*, const ChromaspaceMetal::FrameSubmission&,
                          const ChromaspaceMetal::RasterSourceRequest&,
                          const ChromaspaceMetal::HistogramSurfaceRequest&,
                          uint64_t, uint32_t, int, int, int,
                          std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableWaveform(void*, const ChromaspaceMetal::FrameSubmission&,
                         const ChromaspaceMetal::RasterSourceRequest&,
                         const ChromaspaceMetal::WaveformSurfaceRequest&,
                         uint64_t, uint32_t, int, int, int,
                         std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableRaster(void*, const ChromaspaceMetal::FrameSubmission&,
                       ChromaspaceMetal::ResidentDerivedCache*,
                       const ChromaspaceMetal::RasterSourceRequest&,
                       const ChromaspaceMetal::RasterPointSurfaceRequest&,
                       uint64_t, uint64_t, uint32_t, int, int, int,
                       std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableGlossField(void*, const ChromaspaceMetal::FrameSubmission&,
                           ChromaspaceMetal::GlossFieldCache*,
                           const ChromaspaceMetal::RasterSourceRequest&,
                           const ChromaspaceMetal::GlossFieldRequest&, uint64_t,
                           uint64_t, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableGlossSurface(void*, const ChromaspaceMetal::FrameSubmission&,
                             const ChromaspaceMetal::GlossFieldCache&,
                             const ChromaspaceMetal::GlossFieldSurfaceRequest&,
                             uint32_t, int, int, int,
                             std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableGlossProjection(void*, const ChromaspaceMetal::FrameSubmission&,
                                const ChromaspaceMetal::GlossFieldCache&,
                                const ChromaspaceMetal::GlossProjectionSurfaceRequest&,
                                uint32_t, int, int, int,
                                std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableVectors(void*, const ChromaspaceMetal::FrameSubmission&, uint32_t,
                        int, int, int,
                        const ChromaspaceMetal::FrameVectorVertex*, std::size_t,
                        bool, const std::array<float, 4>&,
                        std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
ChromaspaceMetal::GlossFieldCacheState unavailableGlossState(
    void*, const ChromaspaceMetal::GlossFieldCache&) noexcept {
  return ChromaspaceMetal::GlossFieldCacheState::Missing;
}
void unavailableGlossRelease(void*, ChromaspaceMetal::GlossFieldCache*) noexcept {}
bool unavailableHistogramCached(
    void*, const ChromaspaceMetal::FrameSubmission&,
    ChromaspaceMetal::ResidentDerivedCache*,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::HistogramSurfaceRequest&, uint64_t, uint64_t,
    uint32_t, int, int, int, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
bool unavailableWaveformCached(
    void*, const ChromaspaceMetal::FrameSubmission&,
    ChromaspaceMetal::ResidentDerivedCache*,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::WaveformSurfaceRequest&, uint64_t, uint64_t,
    uint32_t, int, int, int, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-plot-renderer-backend-unavailable");
  return false;
}
ChromaspaceMetal::ResidentDerivedCacheState unavailableDerivedState(
    void*, const ChromaspaceMetal::ResidentDerivedCache&) noexcept {
  return ChromaspaceMetal::ResidentDerivedCacheState::Missing;
}
void unavailableDerivedRelease(
    void*, ChromaspaceMetal::ResidentDerivedCache*) noexcept {}

const RendererBackend kUnavailableBackend{
    nullptr,
    unavailableCreate,
    unavailableRelease,
    unavailableSource,
    unavailableHistogram,
    unavailableWaveform,
    unavailableRaster,
    unavailableGlossField,
    unavailableGlossSurface,
    unavailableGlossProjection,
    unavailableVectors,
    unavailableGlossState,
    unavailableGlossRelease,
    unavailableHistogramCached,
    unavailableWaveformCached,
    unavailableDerivedState,
    unavailableDerivedRelease};
#endif

bool sameSurface(const ChromaspaceMetal::PlotSurface& surface,
                 const PlotCommand& command) noexcept {
  return surface.surfaceId != 0u && surface.width == command.targetWidth &&
         surface.height == command.targetHeight &&
         surface.pixelFormat == command.targetPixelFormat;
}

bool surfaceByteEstimate(const PlotCommand& command,
                         std::size_t* bytes) noexcept {
  if (!bytes || command.targetWidth <= 0 || command.targetHeight <= 0 ||
      (command.targetPixelFormat != 0 && command.targetPixelFormat != 1)) {
    return false;
  }
  const std::size_t width = static_cast<std::size_t>(command.targetWidth);
  const std::size_t height = static_cast<std::size_t>(command.targetHeight);
  const std::size_t bytesPerPixel = command.targetPixelFormat == 0 ? 8u : 16u;
  if (width > std::numeric_limits<std::size_t>::max() / height) return false;
  const std::size_t pixels = width * height;
  if (pixels > std::numeric_limits<std::size_t>::max() / bytesPerPixel) {
    return false;
  }
  *bytes = pixels * bytesPerPixel;
  return *bytes != 0u;
}

bool cacheablePlotContent(PlotKind) noexcept {
  // Gloss final pixels are transactional too; contentRevision and the target
  // descriptor carry presentation invalidation independently of field reuse.
  return true;
}

PlotContentKey makePlotContentKey(
    const FrameRequest& frame,
    const PlotCommand& command) noexcept {
  PlotContentKey key{};
  if (plotKindNeedsSource(command.kind) && frame.hasResidentSource) {
    const auto& source = frame.residentSource;
    key.sourceId = source.sourceId;
    key.deviceRegistryId = source.deviceRegistryId;
    key.senderGeneration = source.senderGeneration;
    key.sequence = source.sequence;
    key.slotIndex = source.slotIndex;
    key.slotGeneration = source.slotGeneration;
    key.contentHash = source.contentHash;
  }
  key.contentRevision = command.contentRevision;
  if (command.kind == PlotKind::Histogram ||
      command.kind == PlotKind::Waveform) {
    scopeDerivationHash(command, &key.derivationHash);
  } else if (command.kind == PlotKind::ResidentRaster) {
    pointDerivationHash(command, &key.derivationHash);
  } else if (command.kind == PlotKind::GlossField2D ||
             command.kind == PlotKind::GlossProjection3D) {
    glossDerivationHash(frame.residentSource, command, &key.derivationHash);
  }
  key.kind = command.kind;
  key.width = command.targetWidth;
  key.height = command.targetHeight;
  key.pixelFormat = command.targetPixelFormat;
  return key;
}

}  // namespace

const char* plotKindLabel(PlotKind kind) noexcept {
  switch (kind) {
    case PlotKind::SourceSignal: return "source-signal";
    case PlotKind::Histogram: return "histogram";
    case PlotKind::Waveform: return "waveform";
    case PlotKind::GlossField2D: return "gloss-field-2d";
    case PlotKind::GlossProjection3D: return "gloss-projection-3d";
    case PlotKind::ResidentRaster: return "resident-raster";
    case PlotKind::Scaffold: return "scaffold";
  }
  return "unknown";
}

const char* statusLabel(WindowStatus status) noexcept {
  switch (status) {
    case WindowStatus::Created: return "created";
    case WindowStatus::Reused: return "reused";
    case WindowStatus::Resized: return "resized";
    case WindowStatus::Replaced: return "replaced";
    case WindowStatus::Encoded: return "encoded";
    case WindowStatus::Scaffolded: return "scaffolded";
    case WindowStatus::Failed: return "failed";
    case WindowStatus::Unavailable: return "unavailable";
  }
  return "unknown";
}

const char* trimStatusLabel(TrimStatus status) noexcept {
  switch (status) {
    case TrimStatus::Accepted: return "accepted";
    case TrimStatus::InvalidLevel: return "invalid-level";
    case TrimStatus::TransactionActive: return "transaction-active";
    case TrimStatus::RendererUnavailable: return "renderer-unavailable";
    case TrimStatus::DerivedCacheResetFailed:
      return "derived-cache-reset-failed";
  }
  return "unknown";
}

FrameRequest::FrameRequest() {
  // Keep transient requests small; the persistent renderer request retains
  // any larger capacity after a genuinely large frame.
  vectorVertexArena.reserve(kMaximumCommandVectorVertices);
}

bool FrameRequest::append(const PlotCommand& command) noexcept {
  if (commandCount >= kMaximumPlotWindows) return false;
  try {
    commands[commandCount] = command;
    ++commandCount;
    return true;
  } catch (...) {
    return false;
  }
}

bool FrameRequest::appendVectorVertices(
    const ChromaspaceMetal::FrameVectorVertex* vertices,
    std::size_t vertexCount,
    PlotCommand* command) noexcept {
  if (!command || vertexCount > kMaximumCommandVectorVertices ||
      (vertexCount % 3u) != 0u ||
      vectorVertexArena.size() > kMaximumFrameVectorVertices ||
      vertexCount > kMaximumFrameVectorVertices - vectorVertexArena.size() ||
      (vertexCount != 0u && vertices == nullptr)) {
    return false;
  }
  try {
    const std::size_t offset = vectorVertexArena.size();
    vectorVertexArena.insert(vectorVertexArena.end(), vertices,
                             vertices + vertexCount);
    command->vectorVertexOffset = offset;
    command->vectorVertexCount = vertexCount;
    return true;
  } catch (...) {
    return false;
  }
}

bool FrameRequest::appendScaffoldVertices(
    const ChromaspaceMetal::FrameVectorVertex* vertices,
    std::size_t vertexCount,
    PlotCommand* command) noexcept {
  return appendVectorVertices(vertices, vertexCount, command);
}

 #if !defined(__APPLE__)
const RendererBackend* defaultRendererBackend() noexcept {
  return &kUnavailableBackend;
}
#endif

bool validateResidentSource(const ChromaspaceMetal::ImportedSourceTexture& source,
                            std::string* error) {
  if (error) error->clear();
  if (source.sourceId == 0u || source.senderId.empty() ||
      source.senderId.size() > ChromaspaceSourceExchange::kMaximumSemanticIdentifierBytes ||
      source.deviceRegistryId == 0u || source.senderGeneration == 0u ||
      source.sequence == 0u || source.slotIndex >= ChromaspaceSourceExchange::kMaximumSlots ||
      source.slotGeneration == 0u || source.readyValue == 0u || source.contentHash == 0u ||
      source.width <= 0 || source.height <= 0 ||
      source.width > kMaximumPlotDimension || source.height > kMaximumPlotDimension ||
      (source.pixelFormat != 0 && source.pixelFormat != 1) ||
      source.bytesPerRow == 0u || source.byteSize == 0u ||
      source.byteSize > ChromaspaceSourceExchange::kMaximumSurfaceBytes ||
      !ChromaspaceSourceExchange::validSourceSemanticMetadata(source.semantics) ||
      !source.semantics.authoritative ||
      source.semantics.coverage != ChromaspaceSourceExchange::SourceCoverage::FullSource ||
      source.semantics.sourceWidth == 0u || source.semantics.sourceHeight == 0u ||
      source.semantics.sourceWidth > kMaximumPlotDimension ||
      source.semantics.sourceHeight > kMaximumPlotDimension) {
    setError(error, "resident-source-invalid");
    return false;
  }
  const uint64_t bytesPerPixel = source.pixelFormat == 0 ? 8u : 16u;
  const uint64_t minimumRowBytes =
      static_cast<uint64_t>(source.width) * bytesPerPixel;
  if (source.bytesPerRow >
      std::numeric_limits<uint64_t>::max() /
          static_cast<uint64_t>(source.height)) {
    setError(error, "resident-source-layout-overflow");
    return false;
  }
  const uint64_t minimumBytes =
      static_cast<uint64_t>(source.height) * source.bytesPerRow;
  if (source.bytesPerRow < minimumRowBytes || minimumBytes > source.byteSize) {
    setError(error, "resident-source-layout-invalid");
    return false;
  }
  return true;
}

bool validateFrameRequest(const FrameRequest& request, std::string* error) {
  if (error) error->clear();
  if (request.commandCount > kMaximumPlotWindows ||
      request.vectorVertexArena.size() > kMaximumFrameVectorVertices) {
    setError(error, "plot-frame-count-limit");
    return false;
  }
  if (request.hasResidentSource && !validateResidentSource(request.residentSource, error)) {
    return false;
  }
  for (std::size_t index = 0; index < request.commandCount; ++index) {
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (request.commands[prior].windowId == request.commands[index].windowId) {
        setError(error, "plot-window-id-duplicate");
        return false;
      }
    }
    if (!validCommand(request.commands[index], request, error)) return false;
  }
  return true;
}

PlotRenderer::PlotRenderer(const RendererBackend* backend,
                           const ResidencyConfig& residencyConfig) noexcept
    : residencyConfig_(residencyConfig),
      derivedCache_(new (std::nothrow)
                      ChromaspaceMetalDerivedCache::DerivedCache(
                          ChromaspaceMetalDerivedCache::Config{
                              static_cast<uint64_t>(
                                  residencyConfig.maxResidentDerivedBytes),
                              static_cast<uint64_t>(
                                  residencyConfig.maxTransientDerivedBytes)})),
      residencyConfigValid_(
          residencyConfig.maxResidentSurfaceBytes != 0u &&
          residencyConfig.maxTransientSurfaceBytes >=
              residencyConfig.maxResidentSurfaceBytes &&
          residencyConfig.maxResidentDerivedBytes != 0u &&
          residencyConfig.maxTransientDerivedBytes >=
              residencyConfig.maxResidentDerivedBytes &&
          derivedCache_ != nullptr && derivedCache_->configValid()) {
  if (backend) {
    backend_ = *backend;
  } else if (const RendererBackend* fallback = defaultRendererBackend()) {
    backend_ = *fallback;
  }
}

PlotRenderer::~PlotRenderer() { shutdown(); }

bool PlotRenderer::validBackend(const RendererBackend& backend) noexcept {
  return backend.createSurface != nullptr && backend.releaseSurface != nullptr &&
         backend.encodeSourceSignal != nullptr && backend.encodeHistogram != nullptr &&
         backend.encodeWaveform != nullptr &&
         backend.encodeResidentRasterCached != nullptr &&
         backend.encodeGlossField != nullptr && backend.encodeGlossFieldSurface != nullptr &&
         backend.encodeGlossProjectionSurface != nullptr && backend.encodeVectors != nullptr &&
         backend.glossCacheState != nullptr && backend.releaseGlossCache != nullptr &&
         backend.encodeHistogramCached != nullptr &&
         backend.encodeWaveformCached != nullptr &&
         backend.derivedCacheState != nullptr &&
         backend.releaseDerivedCache != nullptr;
}

bool PlotRenderer::sourceRequired(PlotKind kind) noexcept {
  return kind != PlotKind::Scaffold;
}

bool PlotRenderer::validPlotKind(PlotKind kind) noexcept {
  const int value = static_cast<int>(kind);
  return value >= static_cast<int>(PlotKind::SourceSignal) &&
         value <= static_cast<int>(PlotKind::Scaffold);
}

void PlotRenderer::addEvent(RenderResult* result,
                            int windowId,
                            PlotKind kind,
                            WindowStatus status,
                            uint32_t surfaceId,
                            const std::string& reason) noexcept {
  if (!result) return;
  try {
    for (std::size_t index = 0; index < result->eventCount; ++index) {
      if (result->events[index].windowId == windowId) {
        if (result->events[index].status == WindowStatus::Reused &&
            result->events[index].reason == "gloss-cache-pending-reused-surface" &&
            status == WindowStatus::Encoded) {
          return;
        }
        result->events[index].kind = kind;
        result->events[index].status = status;
        result->events[index].surfaceId = surfaceId;
        result->events[index].reason = reason;
        return;
      }
    }
    if (result->eventCount >= kMaximumRendererEvents) return;
    WindowEvent& event = result->events[result->eventCount++];
    event.windowId = windowId;
    event.kind = kind;
    event.status = status;
    event.surfaceId = surfaceId;
    event.reason = reason;
  } catch (...) {
  }
}

void PlotRenderer::releaseResource(WindowResource* resource) noexcept {
  if (!resource) return;
  resource->contentKey = PlotContentKey{};
  resource->hasContentKey = false;
  if (resource->surface.surfaceId != 0u) {
    backend_.releaseSurface(backend_.context, compositorId_, resource->surface.surfaceId);
  }
  resource->surface = ChromaspaceMetal::PlotSurface{};
  resource->windowId = 0;
  resource->kind = PlotKind::Scaffold;
}

PlotRenderer::PendingResource* PlotRenderer::findPendingResource(
    int windowId) noexcept {
  for (std::size_t index = 0; index < pendingResourceCount_; ++index) {
    if (pendingResources_[index].windowId == windowId) return &pendingResources_[index];
  }
  return nullptr;
}

const PlotRenderer::PendingResource* PlotRenderer::findPendingResource(
    int windowId) const noexcept {
  for (std::size_t index = 0; index < pendingResourceCount_; ++index) {
    if (pendingResources_[index].windowId == windowId) return &pendingResources_[index];
  }
  return nullptr;
}

PlotRenderer::WindowResource* PlotRenderer::findResource(int windowId) noexcept {
  for (std::size_t index = 0; index < resourceCount_; ++index) {
    if (resources_[index].windowId == windowId) return &resources_[index];
  }
  return nullptr;
}

const PlotRenderer::WindowResource* PlotRenderer::findResource(
    int windowId) const noexcept {
  for (std::size_t index = 0; index < resourceCount_; ++index) {
    if (resources_[index].windowId == windowId) return &resources_[index];
  }
  return nullptr;
}

bool PlotRenderer::stageSurface(uint64_t compositorId,
                                const PlotCommand& command,
                                PendingResource* pending,
                                WindowStatus* outStatus,
                                RenderResult* result,
                                std::string* error) {
  if (!pending || !outStatus) {
    setError(error, "plot-stage-arguments-invalid");
    return false;
  }
  pending->candidate.kind = command.kind;
  pending->candidate.windowId = command.windowId;
  const bool sameDescriptor = sameSurface(pending->candidate.surface, command);
  const bool committedContentMatches =
      pending->hadCommitted && cacheablePlotContent(command.kind) &&
      pending->previous.hasContentKey &&
      pending->previous.contentKey ==
          makePlotContentKey(pendingRequest_, command);
  if (sameDescriptor &&
      (!pending->hadCommitted || !cacheablePlotContent(command.kind) ||
       committedContentMatches)) {
    *outStatus = pending->hadCommitted ? WindowStatus::Reused : WindowStatus::Created;
    if (result) {
      if (*outStatus == WindowStatus::Reused) ++result->reusedSurfaceCount;
      else ++result->createdSurfaceCount;
    }
    addEvent(result, command.windowId, command.kind, *outStatus,
             pending->candidate.surface.surfaceId, "");
    return true;
  }
  ChromaspaceMetal::PlotSurface surface{};
  std::size_t estimatedBytes = 0u;
  if (!surfaceByteEstimate(command, &estimatedBytes)) {
    setError(error, "plot-surface-byte-estimate-invalid");
    return false;
  }
  if (residentSurfaceBytes_ >
          residencyConfig_.maxTransientSurfaceBytes ||
      pendingOwnedSurfaceBytes_ >
          residencyConfig_.maxTransientSurfaceBytes - residentSurfaceBytes_ ||
      estimatedBytes > residencyConfig_.maxTransientSurfaceBytes -
                           residentSurfaceBytes_ -
                           pendingOwnedSurfaceBytes_) {
    setError(error, "plot-transient-residency-budget-exceeded");
    return false;
  }
  if (!backend_.createSurface(backend_.context, compositorId, command.targetWidth,
                              command.targetHeight, command.targetPixelFormat,
                              &surface, error) || surface.surfaceId == 0u) {
    setError(error, error && !error->empty() ? *error : "plot-surface-create-failed");
    addEvent(result, command.windowId, command.kind, WindowStatus::Failed, 0u,
             error ? *error : "plot-surface-create-failed");
    return false;
  }
  if (surface.width != command.targetWidth || surface.height != command.targetHeight ||
      surface.pixelFormat != command.targetPixelFormat ||
      surface.byteSize < estimatedBytes) {
    backend_.releaseSurface(backend_.context, compositorId, surface.surfaceId);
    setError(error, "plot-surface-descriptor-mismatch");
    addEvent(result, command.windowId, command.kind, WindowStatus::Failed, 0u,
             "plot-surface-descriptor-mismatch");
    return false;
  }
  if (surface.byteSize > residencyConfig_.maxTransientSurfaceBytes -
                             residentSurfaceBytes_ -
                             pendingOwnedSurfaceBytes_) {
    backend_.releaseSurface(backend_.context, compositorId, surface.surfaceId);
    setError(error, "plot-transient-residency-budget-exceeded");
    return false;
  }
  pending->candidate.surface = surface;
  pending->ownsSurface = true;
  pendingOwnedSurfaceBytes_ += surface.byteSize;
  if (result) {
    result->transientSurfaceBytes =
        residentSurfaceBytes_ + pendingOwnedSurfaceBytes_;
  }
  *outStatus = pending->hadCommitted
                   ? (sameDescriptor ? WindowStatus::Replaced
                                     : WindowStatus::Resized)
                   : WindowStatus::Created;
  if (result) {
    if (*outStatus == WindowStatus::Resized) {
      ++result->resizedSurfaceCount;
    } else if (*outStatus == WindowStatus::Replaced) {
      ++result->replacedSurfaceCount;
    } else {
      ++result->createdSurfaceCount;
    }
  }
  addEvent(result, command.windowId, command.kind, *outStatus, surface.surfaceId, "");
  return true;
}

bool PlotRenderer::prepare(const FrameRequest& request,
                           uint64_t compositorId,
                           RenderResult* result,
                           std::string* error) {
  if (error) error->clear();
  if (result) result->clear();
  if (transactionActive_) {
    setError(error, "plot-transaction-already-active");
    return false;
  }
  if (!derivedCache_) {
    setError(error, "plot-derived-cache-allocation-failed");
    return false;
  }
  if (!residencyConfigValid_) {
    setError(error, "plot-residency-config-invalid");
    return false;
  }
  if (compositorId == 0u || !validBackend(backend_)) {
    setError(error, "plot-backend-unavailable");
    return false;
  }
  if (!validateFrameRequest(request, error)) return false;
  try {
    pendingRequest_.clear();
    for (std::size_t index = 0; index < request.commandCount; ++index) {
      pendingRequest_.commands[index] = request.commands[index];
    }
    pendingRequest_.vectorVertexArena.clear();
    if (pendingRequest_.vectorVertexArena.capacity() <
        request.vectorVertexArena.size()) {
      pendingRequest_.vectorVertexArena.reserve(request.vectorVertexArena.size());
    }
    pendingRequest_.vectorVertexArena.insert(
        pendingRequest_.vectorVertexArena.end(), request.vectorVertexArena.begin(),
        request.vectorVertexArena.end());
    pendingRequest_.commandCount = request.commandCount;
    pendingRequest_.frameRevision = request.frameRevision;
    pendingRequest_.hasResidentSource = request.hasResidentSource;
    pendingRequest_.residentSource = request.residentSource;
  } catch (...) {
    setError(error, "plot-request-copy-failed");
    return false;
  }
  compositorId_ = compositorId;
  pendingResourceCount_ = 0u;
  pendingOwnedSurfaceBytes_ = 0u;
  if (derivedUseEpoch_ == std::numeric_limits<uint64_t>::max()) {
    setError(error, "plot-derived-cache-epoch-exhausted");
    return false;
  }
  ++derivedUseEpoch_;
  const auto derivedBegin = derivedCache_->begin(derivedUseEpoch_);
  if (derivedBegin != ChromaspaceMetalDerivedCache::Status::Ok) {
    setError(error, ChromaspaceMetalDerivedCache::statusLabel(derivedBegin));
    return false;
  }
  transactionActive_ = true;
  for (std::size_t index = 0; index < pendingRequest_.commandCount; ++index) {
    PendingResource& pending = pendingResources_[pendingResourceCount_++];
    pending = PendingResource{};
    const PlotCommand& command = pendingRequest_.commands[index];
    pending.windowId = command.windowId;
    const WindowResource* committed = findResource(command.windowId);
    pending.hadCommitted = committed != nullptr;
    if (committed) {
      pending.previous = *committed;
      pending.candidate = *committed;
    }
    WindowStatus status = WindowStatus::Created;
    if (!stageSurface(compositorId_, command, &pending, &status, result, error)) {
      finish(false, result);
      return false;
    }
    const PlotContentKey contentKey =
        makePlotContentKey(pendingRequest_, command);
    const bool contentReused =
        cacheablePlotContent(command.kind) && pending.hadCommitted &&
        !pending.ownsSurface && pending.previous.hasContentKey &&
        pending.previous.contentKey == contentKey &&
        pending.candidate.surface.surfaceId == pending.previous.surface.surfaceId;
    if ((command.kind == PlotKind::Histogram ||
         command.kind == PlotKind::Waveform ||
         command.kind == PlotKind::ResidentRaster ||
         command.kind == PlotKind::GlossField2D ||
         command.kind == PlotKind::GlossProjection3D) &&
        !contentReused) {
      uint64_t estimatedBytes = 0u;
      bool estimateValid = false;
      if (command.kind == PlotKind::ResidentRaster) {
        estimateValid = pointByteEstimate(command, &estimatedBytes);
      } else if (command.kind == PlotKind::Histogram ||
                 command.kind == PlotKind::Waveform) {
        estimateValid = scopeByteEstimate(command, &estimatedBytes);
      } else {
        estimateValid = glossByteEstimate(command, &estimatedBytes);
      }
      if (!estimateValid) {
        setError(error, "plot-derived-cache-byte-estimate-invalid");
        finish(false, result);
        return false;
      }
      const auto acquisition =
          derivedCache_->acquire(makeDerivedKey(pendingRequest_, command),
                                 estimatedBytes);
      if (!acquisition) {
        setError(error,
                 ChromaspaceMetalDerivedCache::statusLabel(acquisition.status));
        finish(false, result);
        return false;
      }
      pending.derivedAcquireKind = acquisition.kind;
      pending.derivedCacheIndex =
          acquisition.kind == ChromaspaceMetalDerivedCache::AcquireKind::Hit
              ? acquisition.committedIndex
              : acquisition.stagedIndex;
      if (result) {
        if (acquisition.kind ==
            ChromaspaceMetalDerivedCache::AcquireKind::Hit) {
          bool firstUseOfCommittedDerivation = true;
          for (std::size_t earlierIndex = 0u;
               earlierIndex + 1u < pendingResourceCount_; ++earlierIndex) {
            const PendingResource& earlier = pendingResources_[earlierIndex];
            if (earlier.derivedAcquireKind ==
                    ChromaspaceMetalDerivedCache::AcquireKind::Hit &&
                earlier.derivedCacheIndex == pending.derivedCacheIndex) {
              firstUseOfCommittedDerivation = false;
              break;
            }
          }
          if (firstUseOfCommittedDerivation) {
            ++result->residentDerivedHitCount;
          }
        } else if (!acquisition.reused) {
          ++result->residentDerivedCandidateCount;
        }
      }
    }
  }
  std::size_t projectedResidentBytes = 0u;
  for (std::size_t index = 0u; index < pendingResourceCount_; ++index) {
    const std::size_t bytes = pendingResources_[index].candidate.surface.byteSize;
    if (bytes > residencyConfig_.maxResidentSurfaceBytes -
                    projectedResidentBytes) {
      setError(error, "plot-resident-surface-budget-exceeded");
      finish(false, result);
      return false;
    }
    projectedResidentBytes += bytes;
  }
  if (result) {
    result->residentSurfaceBytes = residentSurfaceBytes_;
    result->transientSurfaceBytes =
        residentSurfaceBytes_ + pendingOwnedSurfaceBytes_;
    result->residentDerivedBytes =
        static_cast<std::size_t>(derivedCache_->residentByteSize());
    result->transientDerivedBytes =
        static_cast<std::size_t>(derivedCache_->transientByteSize());
  }
  if (result) result->commandCount = pendingRequest_.commandCount;
  return true;
}

bool PlotRenderer::encodeCommand(
    const PlotCommand& command,
    const ChromaspaceMetalFrameExecutor::FrameExecutionContext& context,
    WindowResource* resource,
    RenderResult* result,
    std::string* error) {
  if (!resource || !context.submission || !backend_.encodeVectors) {
    setError(error, "plot-encode-arguments-invalid");
    return false;
  }
  const uint32_t surfaceId = resource->surface.surfaceId;
  const int width = resource->surface.width;
  const int height = resource->surface.height;
  const int format = resource->surface.pixelFormat;
  const uint64_t sourceId = pendingRequest_.residentSource.sourceId;
  PendingResource* pending = findPendingResource(command.windowId);
  const auto prepareDerivedCache =
      [&](ChromaspaceMetal::ResidentDerivedCache* cache,
          bool* requiresMaterialization) -> bool {
    if (!cache || !requiresMaterialization || !pending ||
        pending->derivedAcquireKind ==
            ChromaspaceMetalDerivedCache::AcquireKind::Failure) {
      setError(error, "plot-derived-cache-acquisition-missing");
      return false;
    }
    *cache = ChromaspaceMetal::ResidentDerivedCache{};
    if (command.kind == PlotKind::Histogram) {
      cache->family = ChromaspaceMetal::ResidentDerivedFamily::Histogram;
    } else if (command.kind == PlotKind::Waveform) {
      cache->family = ChromaspaceMetal::ResidentDerivedFamily::Waveform;
    } else {
      cache->family = ChromaspaceMetal::ResidentDerivedFamily::RasterPointCloud;
    }
    if (pending->derivedAcquireKind ==
        ChromaspaceMetalDerivedCache::AcquireKind::Hit) {
      const auto* metadata =
          derivedCache_->committedEntry(pending->derivedCacheIndex);
      if (!metadata || !metadata->occupied || metadata->cacheId == 0u ||
          metadata->byteSize == 0u) {
        setError(error, "plot-derived-cache-hit-metadata-invalid");
        return false;
      }
      cache->cacheId = metadata->cacheId;
      cache->ownerCompositorId = compositorId_;
      cache->builtSerial = metadata->cacheId;
      cache->byteSize = static_cast<std::size_t>(metadata->byteSize);
      cache->available = true;
      *requiresMaterialization = false;
      return true;
    }
    const auto* metadata =
        derivedCache_->stagedAcquisition(pending->derivedCacheIndex);
    if (!metadata) {
      setError(error, "plot-derived-cache-candidate-metadata-invalid");
      return false;
    }
    if (metadata->materialized) {
      if (metadata->cacheId == 0u || metadata->byteSize == 0u) {
        setError(error, "plot-derived-cache-candidate-handle-invalid");
        return false;
      }
      cache->cacheId = metadata->cacheId;
      cache->ownerCompositorId = compositorId_;
      cache->builtSerial = metadata->cacheId;
      cache->byteSize = static_cast<std::size_t>(metadata->byteSize);
      cache->available = true;
      *requiresMaterialization = false;
    } else {
      *requiresMaterialization = true;
    }
    return true;
  };
  const auto finishDerivedEncode =
      [&](ChromaspaceMetal::ResidentDerivedCache* cache,
          bool requiresMaterialization) -> bool {
    if (!cache || cache->cacheId == 0u || cache->byteSize == 0u ||
        !cache->available ||
        backend_.derivedCacheState(backend_.context, *cache) ==
            ChromaspaceMetal::ResidentDerivedCacheState::Missing) {
      setError(error, "plot-derived-cache-missing-after-encode");
      return false;
    }
    if (!requiresMaterialization) return true;
    const auto materialized = derivedCache_->materializeCandidate(
        pending->derivedCacheIndex, cache->cacheId,
        static_cast<uint64_t>(cache->byteSize));
    if (materialized != ChromaspaceMetalDerivedCache::Status::Ok) {
      backend_.releaseDerivedCache(backend_.context, cache);
      setError(error,
               ChromaspaceMetalDerivedCache::statusLabel(materialized));
      return false;
    }
    if (result) {
      result->residentDerivedBytes =
          static_cast<std::size_t>(derivedCache_->residentByteSize());
      result->transientDerivedBytes =
          static_cast<std::size_t>(derivedCache_->transientByteSize());
    }
    return true;
  };
  bool primaryOk = true;
  if (command.kind == PlotKind::Scaffold) {
    primaryOk = backend_.encodeVectors(
        backend_.context, *context.submission, surfaceId, width, height, format,
        pendingRequest_.vectorVertexArena.data() + command.vectorVertexOffset,
        command.vectorVertexCount, true, command.vectorClearColor, error);
  } else if (command.kind == PlotKind::SourceSignal) {
    primaryOk = backend_.encodeSourceSignal(
        backend_.context, *context.submission, sourceId, surfaceId, width, height,
        format, error);
  } else if (command.kind == PlotKind::Histogram) {
    ChromaspaceMetal::ResidentDerivedCache cache{};
    bool requiresMaterialization = false;
    if (!prepareDerivedCache(&cache, &requiresMaterialization)) return false;
    if (requiresMaterialization &&
        derivedBuildSerial_ == std::numeric_limits<uint64_t>::max()) {
      setError(error, "plot-derived-cache-build-serial-exhausted");
      return false;
    }
    const uint64_t buildSerial = requiresMaterialization
                                     ? ++derivedBuildSerial_
                                     : cache.builtSerial;
    primaryOk = backend_.encodeHistogramCached(
        backend_.context, *context.submission, &cache, command.raster,
        command.histogram, sourceId, buildSerial, surfaceId, width, height,
        format, error);
    if (primaryOk && !finishDerivedEncode(&cache, requiresMaterialization)) {
      return false;
    }
  } else if (command.kind == PlotKind::Waveform) {
    ChromaspaceMetal::ResidentDerivedCache cache{};
    bool requiresMaterialization = false;
    if (!prepareDerivedCache(&cache, &requiresMaterialization)) return false;
    if (requiresMaterialization &&
        derivedBuildSerial_ == std::numeric_limits<uint64_t>::max()) {
      setError(error, "plot-derived-cache-build-serial-exhausted");
      return false;
    }
    const uint64_t buildSerial = requiresMaterialization
                                     ? ++derivedBuildSerial_
                                     : cache.builtSerial;
    primaryOk = backend_.encodeWaveformCached(
        backend_.context, *context.submission, &cache, command.raster,
        command.waveform, sourceId, buildSerial, surfaceId, width, height,
        format, error);
    if (primaryOk && !finishDerivedEncode(&cache, requiresMaterialization)) {
      return false;
    }
  } else if (command.kind == PlotKind::ResidentRaster) {
    ChromaspaceMetal::ResidentDerivedCache cache{};
    bool requiresMaterialization = false;
    if (!prepareDerivedCache(&cache, &requiresMaterialization)) return false;
    if (requiresMaterialization &&
        derivedBuildSerial_ == std::numeric_limits<uint64_t>::max()) {
      setError(error, "plot-derived-cache-build-serial-exhausted");
      return false;
    }
    const uint64_t buildSerial = requiresMaterialization
                                     ? ++derivedBuildSerial_
                                     : cache.builtSerial;
    primaryOk = backend_.encodeResidentRasterCached(
        backend_.context, *context.submission, &cache, command.raster,
        command.point, sourceId, buildSerial, surfaceId, width, height, format,
        error);
    if (primaryOk &&
        !finishDerivedEncode(&cache, requiresMaterialization)) {
      return false;
    }
  } else if (command.kind == PlotKind::GlossField2D ||
             command.kind == PlotKind::GlossProjection3D) {
    if (!pending ||
        pending->derivedAcquireKind ==
            ChromaspaceMetalDerivedCache::AcquireKind::Failure) {
      setError(error, "plot-gloss-cache-acquisition-missing");
      return false;
    }
    ChromaspaceMetal::GlossFieldCache cache{};
    bool requiresMaterialization = false;
    const auto hydrate = [&](const ChromaspaceMetalDerivedCache::CandidateMetadata& metadata) {
      if (metadata.cacheId == 0u || metadata.byteSize == 0u) return false;
      cache.cacheId = metadata.cacheId;
      cache.ownerCompositorId = compositorId_;
      cache.gridWidth = command.glossField.gridWidth;
      cache.gridHeight = command.glossField.gridHeight;
      // The generalized cache owns lifecycle identity.  A non-zero stable
      // token keeps the compatibility backend seam valid on cache hits.
      cache.builtSerial = metadata.cacheId;
      cache.byteSize = static_cast<std::size_t>(metadata.byteSize);
      cache.available = true;
      return true;
    };
    if (pending->derivedAcquireKind ==
        ChromaspaceMetalDerivedCache::AcquireKind::Hit) {
      const auto* metadata =
          derivedCache_->committedEntry(pending->derivedCacheIndex);
      if (!metadata || metadata->key.family !=
                           ChromaspaceMetalDerivedCache::Family::GlossField ||
          !hydrate(ChromaspaceMetalDerivedCache::CandidateMetadata{
              metadata->key, metadata->cacheId, metadata->byteSize,
              metadata->lastUseEpoch, metadata->byteSize, true})) {
        setError(error, "plot-gloss-cache-hit-metadata-invalid");
        return false;
      }
    } else {
      const auto* metadata =
          derivedCache_->stagedAcquisition(pending->derivedCacheIndex);
      if (!metadata || metadata->key.family !=
                           ChromaspaceMetalDerivedCache::Family::GlossField) {
        setError(error, "plot-gloss-cache-candidate-metadata-invalid");
        return false;
      }
      if (metadata->materialized) {
        if (!hydrate(*metadata)) {
          setError(error, "plot-gloss-cache-candidate-handle-invalid");
          return false;
        }
      } else {
        requiresMaterialization = true;
      }
    }
    if (requiresMaterialization) {
      if (derivedBuildSerial_ == std::numeric_limits<uint64_t>::max()) {
        setError(error, "plot-derived-cache-build-serial-exhausted");
        return false;
      }
      cache.gridWidth = command.glossField.gridWidth;
      cache.gridHeight = command.glossField.gridHeight;
      cache.builtSerial = ++derivedBuildSerial_;
      if (!backend_.encodeGlossField(
              backend_.context, *context.submission, &cache, command.raster,
              command.glossField, sourceId, cache.builtSerial, error)) {
        return false;
      }
      const auto state = backend_.glossCacheState(backend_.context, cache);
      if (state == ChromaspaceMetal::GlossFieldCacheState::Missing ||
          cache.cacheId == 0u || cache.byteSize == 0u || !cache.available) {
        backend_.releaseGlossCache(backend_.context, &cache);
        setError(error, "plot-gloss-cache-missing-after-encode");
        return false;
      }
      const auto materialized = derivedCache_->materializeCandidate(
          pending->derivedCacheIndex, cache.cacheId,
          static_cast<uint64_t>(cache.byteSize));
      if (materialized != ChromaspaceMetalDerivedCache::Status::Ok) {
        backend_.releaseGlossCache(backend_.context, &cache);
        setError(error,
                 ChromaspaceMetalDerivedCache::statusLabel(materialized));
        return false;
      }
      if (result) {
        result->residentDerivedBytes =
            static_cast<std::size_t>(derivedCache_->residentByteSize());
        result->transientDerivedBytes =
            static_cast<std::size_t>(derivedCache_->transientByteSize());
      }
    }
    if (command.kind == PlotKind::GlossField2D) {
      primaryOk = backend_.encodeGlossFieldSurface(
          backend_.context, *context.submission, cache,
          command.glossFieldSurface, surfaceId, width, height, format, error);
    } else {
      primaryOk = backend_.encodeGlossProjectionSurface(
          backend_.context, *context.submission, cache,
          command.glossProjectionSurface, surfaceId, width, height, format, error);
    }
  }
  if (!primaryOk) return false;
  if (command.kind != PlotKind::Scaffold && command.vectorVertexCount != 0u) {
    if (!backend_.encodeVectors(
            backend_.context, *context.submission, surfaceId, width, height, format,
            pendingRequest_.vectorVertexArena.data() + command.vectorVertexOffset,
            command.vectorVertexCount, command.vectorClearBeforeDraw,
            command.vectorClearColor, error)) {
      return false;
    }
  }
  return true;
}

bool PlotRenderer::encodePrepared(
    const ChromaspaceMetalFrameExecutor::FrameExecutionContext& context,
    ChromaspaceMetalFrameExecutor::FrameBatch* batch,
    RenderResult* result,
    std::string* error) {
  if (error) error->clear();
  if (result) result->frameSucceeded = false;
  if (!transactionActive_ || !batch || !context.submission ||
      context.compositorId != compositorId_) {
    setError(error, "plot-encode-without-transaction");
    return false;
  }
  if (batch->compositeItems.size() > batch->compositeItems.max_size() ||
      batch->compositeItems.size() + pendingRequest_.commandCount >
          batch->compositeItems.capacity() ||
      batch->compositeItems.size() + pendingRequest_.commandCount >
          ChromaspaceMetalFrameExecutor::kMaxSurfaceItems) {
    setError(error, "plot-composite-capacity-insufficient");
    return false;
  }
  std::array<ChromaspaceMetal::SurfaceCompositeItem, kMaximumPlotWindows> staged{};
  for (std::size_t index = 0; index < pendingRequest_.commandCount; ++index) {
    const PlotCommand& command = pendingRequest_.commands[index];
    PendingResource* pending = findPendingResource(command.windowId);
    if (!pending) {
      setError(error, "plot-pending-resource-missing");
      return false;
    }
    const PlotContentKey contentKey =
        makePlotContentKey(pendingRequest_, command);
    const bool contentReused =
        cacheablePlotContent(command.kind) && pending->hadCommitted &&
        !pending->ownsSurface && pending->previous.hasContentKey &&
        pending->previous.contentKey == contentKey &&
        pending->candidate.surface.surfaceId ==
            pending->previous.surface.surfaceId;
    if (contentReused) {
      if (result) ++result->residentContentHitCount;
      addEvent(result, command.windowId, command.kind, WindowStatus::Reused,
               pending->candidate.surface.surfaceId,
               "resident-plot-content-cache-hit");
    } else {
      if (!encodeCommand(command, context, &pending->candidate, result, error)) {
        addEvent(result, command.windowId, command.kind, WindowStatus::Failed,
                 pending->candidate.surface.surfaceId,
                 error ? *error : "plot-encode-failed");
        return false;
      }
      if (cacheablePlotContent(command.kind)) {
        pending->candidate.contentKey = contentKey;
        pending->candidate.hasContentKey = true;
      } else {
        pending->candidate.contentKey = PlotContentKey{};
        pending->candidate.hasContentKey = false;
      }
    }
    staged[index].surfaceId = pending->candidate.surface.surfaceId;
    staged[index].surfaceWidth = pending->candidate.surface.width;
    staged[index].surfaceHeight = pending->candidate.surface.height;
    staged[index].surfacePixelFormat = pending->candidate.surface.pixelFormat;
    staged[index].dstX = command.destination.x;
    staged[index].dstY = command.destination.y;
    staged[index].dstW = command.destination.width;
    staged[index].dstH = command.destination.height;
    staged[index].opacity = 1.0f;
    if (!contentReused) {
      addEvent(result, command.windowId, command.kind,
               command.kind == PlotKind::Scaffold ? WindowStatus::Scaffolded
                                                   : WindowStatus::Encoded,
               staged[index].surfaceId,
               command.kind == PlotKind::Scaffold ? command.unavailableReason : "");
    }
  }
  try {
    batch->compositeItems.insert(batch->compositeItems.end(), staged.begin(),
                                 staged.begin() + pendingRequest_.commandCount);
  } catch (...) {
    setError(error, "plot-composite-append-failed");
    return false;
  }
  if (result) {
    result->compositeItemCount = pendingRequest_.commandCount;
  }
  return true;
}

bool PlotRenderer::finish(bool submitted, RenderResult* result) noexcept {
  if (!transactionActive_) return !submitted;
  if (!submitted) {
    for (std::size_t index = 0; index < pendingResourceCount_; ++index) {
      PendingResource& pending = pendingResources_[index];
      if (pending.ownsSurface && pending.candidate.surface.surfaceId != 0u) {
        backend_.releaseSurface(backend_.context, compositorId_,
                                pending.candidate.surface.surfaceId);
      }
    }
    if (derivedCache_ && derivedCache_->transactionActive()) {
      const auto aborted = derivedCache_->abort();
      if (aborted.succeeded()) releaseDerivedCaches(aborted.releases);
    }
    if (result) result->frameSucceeded = false;
    transactionActive_ = false;
    pendingResourceCount_ = 0u;
    pendingOwnedSurfaceBytes_ = 0u;
    if (result) {
      result->residentSurfaceBytes = residentSurfaceBytes_;
      result->transientSurfaceBytes = residentSurfaceBytes_;
      result->residentDerivedBytes =
          static_cast<std::size_t>(derivedCache_->residentByteSize());
      result->transientDerivedBytes =
          static_cast<std::size_t>(derivedCache_->transientByteSize());
    }
    pendingRequest_.clear();
    return true;
  }
  // Commit only after the executor reports a successful final submit.
  std::size_t retainedCount = 0u;
  for (std::size_t resourceIndex = 0; resourceIndex < resourceCount_; ++resourceIndex) {
    for (std::size_t pendingIndex = 0; pendingIndex < pendingResourceCount_;
         ++pendingIndex) {
      if (resources_[resourceIndex].windowId == pendingResources_[pendingIndex].windowId) {
        ++retainedCount;
        break;
      }
    }
  }
  if (retainedCount + (pendingResourceCount_ - retainedCount) >
      kMaximumPlotWindows) {
    finish(false, result);
    return false;
  }
  const auto derivedCommitted = derivedCache_->commit();
  if (!derivedCommitted.succeeded()) {
    finish(false, result);
    return false;
  }
  releaseDerivedCaches(derivedCommitted.releases);
  if (result) {
    result->evictedDerivedCacheCount += derivedCommitted.releases.count;
  }
  // Drop committed windows absent from this frame first so the fixed table
  // always has room for a wholly new set of windows.
  for (std::size_t index = 0; index < resourceCount_;) {
    bool present = false;
    for (std::size_t pendingIndex = 0; pendingIndex < pendingResourceCount_;
         ++pendingIndex) {
      if (resources_[index].windowId == pendingResources_[pendingIndex].windowId) {
        present = true;
        break;
      }
    }
    if (present) {
      ++index;
      continue;
    }
    releaseResource(&resources_[index]);
    resources_[index] = resources_[resourceCount_ - 1u];
    --resourceCount_;
    if (result) ++result->prunedSurfaceCount;
  }
  for (std::size_t index = 0; index < pendingResourceCount_; ++index) {
    PendingResource& pending = pendingResources_[index];
    std::size_t slot = resourceCount_;
    for (std::size_t resourceIndex = 0; resourceIndex < resourceCount_; ++resourceIndex) {
      if (resources_[resourceIndex].windowId == pending.windowId) {
        slot = resourceIndex;
        break;
      }
    }
    if (slot == resourceCount_) {
      resources_[resourceCount_++] = pending.candidate;
    } else {
      if (pending.hadCommitted &&
          pending.previous.surface.surfaceId != pending.candidate.surface.surfaceId &&
          pending.previous.surface.surfaceId != 0u) {
        backend_.releaseSurface(backend_.context, compositorId_,
                                pending.previous.surface.surfaceId);
      }
      resources_[slot] = pending.candidate;
    }
    pending.ownsSurface = false;
  }
  if (result) {
    result->frameSucceeded = true;
    result->commandCount = pendingRequest_.commandCount;
  }
  residentSurfaceBytes_ = 0u;
  for (std::size_t index = 0u; index < resourceCount_; ++index) {
    const std::size_t bytes = resources_[index].surface.byteSize;
    if (bytes > std::numeric_limits<std::size_t>::max() -
                    residentSurfaceBytes_) {
      residentSurfaceBytes_ = std::numeric_limits<std::size_t>::max();
      break;
    }
    residentSurfaceBytes_ += bytes;
  }
  transactionActive_ = false;
  pendingResourceCount_ = 0u;
  pendingOwnedSurfaceBytes_ = 0u;
  if (result) {
    result->residentSurfaceBytes = residentSurfaceBytes_;
    result->transientSurfaceBytes = residentSurfaceBytes_;
    result->residentDerivedBytes =
        static_cast<std::size_t>(derivedCache_->residentByteSize());
    result->transientDerivedBytes =
        static_cast<std::size_t>(derivedCache_->transientByteSize());
  }
  pendingRequest_.clear();
  return true;
}

void PlotRenderer::shutdown() noexcept {
  if (transactionActive_) finish(false, nullptr);
  for (std::size_t index = 0; index < resourceCount_; ++index) {
    releaseResource(&resources_[index]);
  }
  resourceCount_ = 0u;
  residentSurfaceBytes_ = 0u;
  pendingOwnedSurfaceBytes_ = 0u;
  pendingResourceCount_ = 0u;
  if (derivedCache_) {
    const auto resetDerived = derivedCache_->reset();
    if (resetDerived.succeeded()) releaseDerivedCaches(resetDerived.releases);
  }
  compositorId_ = 0u;
  pendingRequest_.clear();
}

ResidencySnapshot PlotRenderer::residencySnapshot() const noexcept {
  ResidencySnapshot snapshot{};
  snapshot.surfaceCount = resourceCount_;
  snapshot.surfaceBytes = residentSurfaceBytes_;
  if (derivedCache_) {
    snapshot.derivedCacheCount = derivedCache_->committedCount();
    const uint64_t bytes = derivedCache_->residentByteSize();
    snapshot.derivedCacheBytes =
        bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(bytes);
  }
  return snapshot;
}

TrimResult PlotRenderer::trim(TrimLevel level) noexcept {
  TrimResult result{};
  result.level = level;

  // Validate the enum representation before querying any state.  This keeps
  // malformed external memory-pressure messages completely side-effect free.
  const uint8_t levelValue = static_cast<uint8_t>(level);
  if (levelValue > static_cast<uint8_t>(TrimLevel::AllPlotResources)) {
    result.status = TrimStatus::InvalidLevel;
    return result;
  }
  if (transactionActive_) {
    result.status = TrimStatus::TransactionActive;
    result.before = residencySnapshot();
    result.after = result.before;
    return result;
  }
  if (!derivedCache_ || !residencyConfigValid_ ||
      !validBackend(backend_)) {
    result.status = TrimStatus::RendererUnavailable;
    result.before = residencySnapshot();
    result.after = result.before;
    return result;
  }

  result.before = residencySnapshot();

  // DerivedCache::reset() is the reusable non-terminal reset path.  It
  // atomically prepares the complete release list before mutating cache
  // accounting, so a non-success status leaves all committed state intact.
  ChromaspaceMetalDerivedCache::ReleaseList releases{};
  const auto resetStatus = derivedCache_->reset(&releases);
  if (resetStatus != ChromaspaceMetalDerivedCache::Status::Ok) {
    result.status = TrimStatus::DerivedCacheResetFailed;
    result.after = result.before;
    return result;
  }
  releaseDerivedCaches(releases);
  result.releasedDerivedCacheCount = releases.count;
  for (std::size_t index = 0u; index < releases.count; ++index) {
    const uint64_t bytes = releases.records[index].byteSize;
    const std::size_t value =
        bytes > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(bytes);
    if (value > std::numeric_limits<std::size_t>::max() -
                    result.releasedDerivedCacheBytes) {
      result.releasedDerivedCacheBytes =
          std::numeric_limits<std::size_t>::max();
    } else {
      result.releasedDerivedCacheBytes += value;
    }
  }

  if (level == TrimLevel::AllPlotResources) {
    // Surface release is intentionally limited to committed resources.  A
    // prepared/staged transaction was rejected above and can therefore never
    // be accidentally reclaimed by this event-path operation.
    result.releasedSurfaceCount = result.before.surfaceCount;
    result.releasedSurfaceBytes = result.before.surfaceBytes;
    for (std::size_t index = 0u; index < resourceCount_; ++index) {
      releaseResource(&resources_[index]);
    }
    resources_.fill(WindowResource{});
    resourceCount_ = 0u;
    residentSurfaceBytes_ = 0u;
    pendingOwnedSurfaceBytes_ = 0u;
  }

  result.status = TrimStatus::Accepted;
  result.after = residencySnapshot();
  return result;
}

void PlotRenderer::releaseDerivedCaches(
    const ChromaspaceMetalDerivedCache::ReleaseList& releases) noexcept {
  for (std::size_t index = 0u; index < releases.count; ++index) {
    const auto& record = releases.records[index];
    if (record.cacheId == 0u) continue;
    if (record.family == ChromaspaceMetalDerivedCache::Family::GlossField) {
      if (!backend_.releaseGlossCache) continue;
      ChromaspaceMetal::GlossFieldCache cache{};
      cache.cacheId = record.cacheId;
      cache.ownerCompositorId = compositorId_;
      cache.byteSize = static_cast<std::size_t>(record.byteSize);
      cache.builtSerial = record.cacheId;
      cache.available = true;
      backend_.releaseGlossCache(backend_.context, &cache);
      continue;
    }
    if (!backend_.releaseDerivedCache) continue;
    ChromaspaceMetal::ResidentDerivedCache cache{};
    cache.cacheId = record.cacheId;
    cache.ownerCompositorId = compositorId_;
    cache.byteSize = static_cast<std::size_t>(record.byteSize);
    cache.available = true;
    switch (record.family) {
      case ChromaspaceMetalDerivedCache::Family::Histogram:
        cache.family = ChromaspaceMetal::ResidentDerivedFamily::Histogram;
        break;
      case ChromaspaceMetalDerivedCache::Family::Waveform:
        cache.family = ChromaspaceMetal::ResidentDerivedFamily::Waveform;
        break;
      case ChromaspaceMetalDerivedCache::Family::RasterPointCloud:
        cache.family = ChromaspaceMetal::ResidentDerivedFamily::RasterPointCloud;
        break;
      case ChromaspaceMetalDerivedCache::Family::GlossField:
        continue;
    }
    backend_.releaseDerivedCache(backend_.context, &cache);
  }
}

std::size_t PlotRenderer::glossCacheCount() const noexcept {
  return derivedCache_
             ? derivedCache_->committedCount(
                   ChromaspaceMetalDerivedCache::Family::GlossField)
             : 0u;
}

bool PlotRenderer::hasResource(int windowId) const noexcept {
  for (std::size_t index = 0; index < resourceCount_; ++index) {
    if (resources_[index].windowId == windowId) {
      return true;
    }
  }
  return false;
}

}  // namespace ChromaspaceMetalPlotRenderer
