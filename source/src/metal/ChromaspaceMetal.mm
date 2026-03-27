#include "ChromaspaceMetal.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

namespace ChromaspaceMetal {

namespace {

inline size_t packedRowBytesForWidth(int width) {
  return static_cast<size_t>(width) * 4u * sizeof(float);
}

inline bool validateFloatRowBytes(size_t rowBytes) {
  return rowBytes >= 4u * sizeof(float) && (rowBytes % sizeof(float)) == 0u;
}

inline size_t offsetForOrigin(size_t rowBytes, int originX, int originY) {
  return static_cast<size_t>(originY) * rowBytes + static_cast<size_t>(originX) * 4u * sizeof(float);
}

bool checkedProductToInt(int a, int b, int* out) {
  if (!out || a < 0 || b < 0) return false;
  const long long product = static_cast<long long>(a) * static_cast<long long>(b);
  if (product > static_cast<long long>(std::numeric_limits<int>::max())) return false;
  *out = static_cast<int>(product);
  return true;
}

bool checkedCubeToInt(int value, int* out) {
  if (!out || value < 0) return false;
  const long long v = static_cast<long long>(value);
  const long long product = v * v * v;
  if (product > static_cast<long long>(std::numeric_limits<int>::max())) return false;
  *out = static_cast<int>(product);
  return true;
}

bool encodeCopyRows(
    id<MTLBlitCommandEncoder> blit,
    id<MTLBuffer> src,
    id<MTLBuffer> dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int width,
    int height) {
  const size_t packedRowBytes = packedRowBytesForWidth(width);
  if (packedRowBytes == 0 || height <= 0) return false;
  for (int y = 0; y < height; ++y) {
    const size_t srcRowOffset = srcOffset + static_cast<size_t>(y) * srcRowBytes;
    const size_t dstRowOffset = dstOffset + static_cast<size_t>(y) * dstRowBytes;
    [blit copyFromBuffer:src sourceOffset:srcRowOffset toBuffer:dst destinationOffset:dstRowOffset size:packedRowBytes];
  }
  return true;
}

bool validateRows(int width, int height, size_t srcRowBytes, size_t dstRowBytes) {
  if (width <= 0 || height <= 0) return false;
  const size_t packedRowBytes = packedRowBytesForWidth(width);
  return srcRowBytes >= packedRowBytes && dstRowBytes >= packedRowBytes;
}

std::string nsErrorString(NSError* error) {
  if (!error) return {};
  NSString* description = error.localizedDescription ?: @"unknown metal error";
  return std::string(description.UTF8String ? description.UTF8String : "unknown metal error");
}

struct WholeImageRequestGpu {
  int width = 0;
  int height = 0;
  int originX = 0;
  int originY = 0;
  uint32_t srcRowFloats = 0;
  int scaledWidth = 0;
  int scaledHeight = 0;
  int pointCount = 0;
  int candidateTarget = 0;
  int maxPrimaryAttempts = 0;
  int maxCandidateAttempts = 0;
  int samplingMode = 0;
  int preserveOverflow = 0;
  int plotMode = 0;
  int circularHsl = 0;
  int circularHsv = 0;
  int normConeNormalized = 1;
  int showOverflow = 0;
  int plotDisplayLinearEnabled = 0;
  int plotDisplayLinearTransfer = 0;
  int neutralRadiusEnabled = 0;
  float neutralRadius = 1.0f;
};

struct StripRequestGpu {
  int width = 0;
  int height = 0;
  int originX = 0;
  int originY = 0;
  uint32_t srcRowFloats = 0;
  int resolution = 0;
  int preserveOverflow = 0;
  int plotDisplayLinearEnabled = 0;
  int plotDisplayLinearTransfer = 0;
  int cubeY1 = 0;
  int stripHeight = 0;
  int rampY1 = 0;
  int rampHeight = 0;
  int rampSampleRows = 0;
  float cellWidth = 1.0f;
};

struct CombinedPackRequestGpu {
  int stripCount = 0;
  int maxPrimaryCount = 0;
};

struct AppendSelectRequestGpu {
  int candidateCount = 0;
  int extraPointCount = 0;
  int occupancyThreshold = 0;
  float radiusMin = 0.0f;
  float radiusMax = 1.0f;
};

struct PipelineBundle {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> primary = nil;
  id<MTLComputePipelineState> occupancy = nil;
  id<MTLComputePipelineState> stripCube = nil;
  id<MTLComputePipelineState> stripRamp = nil;
  id<MTLComputePipelineState> packCombined = nil;
  id<MTLComputePipelineState> appendSelect = nil;
};

std::mutex gPipelineMutex;
PipelineBundle gPipelines;

const char* kCloudMetalSourcePart1 = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Sample {
  float xNorm;
  float yNorm;
  float zReserved;
  float r;
  float g;
  float b;
};

struct OccupancyCandidate {
  Sample sample;
  float normalizedNeutralRadius;
  int bin;
  uint tie;
};

struct WholeImageRequestGpu {
  int width;
  int height;
  int originX;
  int originY;
  uint srcRowFloats;
  int scaledWidth;
  int scaledHeight;
  int pointCount;
  int candidateTarget;
  int maxPrimaryAttempts;
  int maxCandidateAttempts;
  int samplingMode;
  int preserveOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int showOverflow;
  int plotDisplayLinearEnabled;
  int plotDisplayLinearTransfer;
  int neutralRadiusEnabled;
  float neutralRadius;
};

struct StripRequestGpu {
  int width;
  int height;
  int originX;
  int originY;
  uint srcRowFloats;
  int resolution;
  int preserveOverflow;
  int plotDisplayLinearEnabled;
  int plotDisplayLinearTransfer;
  int cubeY1;
  int stripHeight;
  int rampY1;
  int rampHeight;
  int rampSampleRows;
  float cellWidth;
};

constant float kRgbAxisMaxRadius = 0.8164965809277260f;
constant float kPolarMax = 0.9553166181245093f;
constant float kChenPolarScale = 1.0467733744265997f;

inline float clamp01Safe(float value) { return clamp(value, 0.0f, 1.0f); }
inline float safeDiv(float num, float den, float fallback, float eps) { return fabs(den) < eps ? fallback : num / den; }
inline float safeDiv(float num, float den) { return safeDiv(num, den, 0.0f, 1e-6f); }
inline float safeSqrt(float value) { return sqrt(fmax(value, 0.0f)); }
inline float safePowPos(float value, float exponent, float fallback) { return value <= 0.0f ? fallback : pow(value, exponent); }
inline float safePowPos(float value, float exponent) { return safePowPos(value, exponent, 0.0f); }
inline float safeExp2Clamped(float value) { return exp2(clamp(value, -126.0f, 126.0f)); }
inline float safeHypot2(float x, float y) { return safeSqrt(x * x + y * y); }
inline float safeAtan2(float y, float x, float fallback) { return (fabs(x) < 1e-8f && fabs(y) < 1e-8f) ? fallback : atan2(y, x); }
inline float safeAtan2(float y, float x) { return safeAtan2(y, x, 0.0f); }
inline float safeAcosUnit(float value) { return acos(clamp(value, -1.0f, 1.0f)); }
inline float safeAsinUnit(float value) { return asin(clamp(value, -1.0f, 1.0f)); }
inline float wrapUnitSafe(float value) { float wrapped = value - floor(value); if (wrapped < 0.0f) wrapped += 1.0f; return wrapped; }
inline float wrapPeriodSafe(float value, float period) { if (period <= 0.0f) return 0.0f; float wrapped = value - period * floor(value / period); if (wrapped < 0.0f) wrapped += period; return wrapped; }
inline float signPreservingPowSafe(float value, float exponent) { return value == 0.0f ? 0.0f : copysign(safePowPos(fabs(value), exponent, 0.0f), value); }
inline float exp10CompatSafe(float x) { return safeExp2Clamped(x * 3.3219280948873626f); }
inline float effectiveNeutralRadiusThresholdSafe(float sliderValue) { return clamp(safePowPos(clamp(sliderValue, 0.0f, 1.0f), 2.0f, 0.0f), 0.0f, 1.0f); }
inline uint hash32Fast(uint value) { value ^= value >> 16; value *= 0x7feb352dU; value ^= value >> 15; value *= 0x846ca68bU; value ^= value >> 16; return value; }
inline float unitHash01Fast(uint value) { return (float)hash32Fast(value) / 4294967295.0f; }
inline float haltonFast(uint index, uint base) { float factor = 1.0f; float result = 0.0f; while (index > 0u) { factor /= (float)base; result += factor * (float)(index % base); index /= base; } return result; }

inline float decodeTransferChannelFast(float x, int tf) {
  switch (tf) {
    case 0: return x;
    case 1: { const float a = fabs(x); const float decoded = (a <= 0.04045f) ? safeDiv(a, 12.92f) : safePowPos(safeDiv(a + 0.055f, 1.055f), 2.4f, 0.0f); return copysign(decoded, x); }
    case 2: return signPreservingPowSafe(x, 2.4f);
    case 3: return x <= 0.02740668f ? safeDiv(x, 10.44426855f) : safeExp2Clamped(safeDiv(x, 0.07329248f) - 7.0f) - 0.0075f;
    case 4: return x <= 0.155251141552511f ? safeDiv(x - 0.0729055341958355f, 10.5402377416545f) : safeExp2Clamped(x * 17.52f - 9.72f);
    case 5: return x < 5.367655f * 0.010591f + 0.092809f ? safeDiv(x - 0.092809f, 5.367655f) : safeDiv(exp10CompatSafe(safeDiv(x - 0.385537f, 0.247190f)) - 0.052272f, 5.555556f);
    case 6: return x < -0.7774983977293537f ? x * 0.3033266726886969f - 0.7774983977293537f : safeDiv(safeExp2Clamped(14.0f * safeDiv(x - 0.09286412512218964f, 0.9071358748778103f) + 6.0f) - 64.0f, 2231.8263090676883f);
    case 7: { constexpr float kCut = 0.092864125f; constexpr float kScale = 0.24136077f; constexpr float kGain = 87.099375f; const float decoded = x < kCut ? -safeDiv(exp10CompatSafe(safeDiv(kCut - x, kScale)) - 1.0f, kGain) : safeDiv(exp10CompatSafe(safeDiv(x - kCut, kScale)) - 1.0f, kGain); return decoded * 0.9f; }
    case 8: return x < 171.2102946929f / 1023.0f ? safeDiv((x * 1023.0f - 95.0f) * 0.01125f, 171.2102946929f - 95.0f) : (exp10CompatSafe(safeDiv(x * 1023.0f - 420.0f, 261.5f)) * 0.19f - 0.01f);
    case 9: if (x < 0.04076162f) return -safeDiv(exp10CompatSafe(safeDiv(0.069886632f - x, 0.42889912f)) - 1.0f, 14.98325f); if (x <= 0.105357102f) return safeDiv(x - 0.073059361f, 2.3069815f); return safeDiv(exp10CompatSafe(safeDiv(x - 0.073059361f, 0.36726845f)) - 1.0f, 14.98325f);
    case 10: return x < 0.0f ? safeDiv(x, 15.1927f) - 0.01f : safeDiv(exp10CompatSafe(safeDiv(x, 0.224282f)) - 1.0f, 155.975327f) - 0.01f;
    default: return x;
  }
}

inline float rawRgbHue01Fast(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6f) return 0.0f;
  float h = 0.0f;
  if (cMax == r) h = wrapPeriodSafe(safeDiv(g - b, delta), 6.0f);
  else if (cMax == g) h = safeDiv(b - r, delta) + 2.0f;
  else h = safeDiv(r - g, delta) + 4.0f;
  return wrapUnitSafe(safeDiv(h, 6.0f));
}

inline void rgbToHsvHexconePlaneFast(float r, float g, float b, thread float& outX, thread float& outZ) { outX = r - 0.5f * g - 0.5f * b; outZ = 0.8660254037844386f * (g - b); }
inline void rgbToPlotCircularHsvFast(float r, float g, float b, thread float& outH, thread float& outRadius, thread float& outV) { const float cMax = max(r, max(g, b)); const float cMin = min(r, min(g, b)); const float delta = cMax - cMin; outH = rawRgbHue01Fast(r, g, b, cMax, delta); outRadius = (delta > 1e-6f && cMax > 1e-6f) ? safeDiv(delta, cMax) : 0.0f; outV = cMax; }
inline void rgbToPlotCircularHslFast(float r, float g, float b, thread float& outH, thread float& outRadius, thread float& outL) { const float cMax = max(r, max(g, b)); const float cMin = min(r, min(g, b)); const float delta = cMax - cMin; const float light = 0.5f * (cMax + cMin); float hue = rawRgbHue01Fast(r, g, b, cMax, delta); float satDenom = 1.0f - fabs(2.0f * light - 1.0f); if (delta > 1e-6f && satDenom < 0.0f) hue = wrapUnitSafe(hue + 0.5f); if (fabs(satDenom) <= 1e-6f) satDenom = satDenom < 0.0f ? -1e-6f : 1e-6f; outH = hue; outRadius = fabs(safeDiv(delta, satDenom)); outL = light; }
)METAL";

const char* kCloudMetalSourcePart2 = R"METAL(
inline void rgbToChenFast(float r, float g, float b, bool allowOverflow, thread float& outHue, thread float& outChroma, thread float& outLight) { constexpr float kTau = 6.28318530717958647692f; if (!allowOverflow) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); } const float rotX = r * 0.81649658f + g * -0.40824829f + b * -0.40824829f; const float rotY = g * 0.70710678f + b * -0.70710678f; const float rotZ = (r + g + b) * 0.57735027f; const float azimuth = safeAtan2(rotY, rotX); const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ); const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ); outHue = azimuth < 0.0f ? safeDiv(azimuth + kTau, kTau) : safeDiv(azimuth, kTau); outChroma = polar * kChenPolarScale; outLight = radius * 0.5773502691896258f; }
inline void rgbToNormConeCoordsFast(float r, float g, float b, bool normalized, bool allowOverflow, thread float& outHue, thread float& outChroma, thread float& outValue) { constexpr float kTau = 6.28318530717958647692f; const float maxRgb = max(r, max(g, b)); const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b; const float rotY = 0.70710678118f * g - 0.70710678118f * b; const float rotZ = 0.57735026919f * (r + g + b); float hue = safeDiv(safeAtan2(rotY, rotX), kTau); if (hue < 0.0f) hue += 1.0f; const float chromaRadius = safeHypot2(rotX, rotY); const float polar = safeAtan2(chromaRadius, rotZ); float chroma = safeDiv(polar, kPolarMax); if (normalized) { const float angle = hue * kTau - 0.52359877559829887308f; const float cosPolar = cos(polar); const float safeCos = fabs(cosPolar) > 1e-6f ? cosPolar : (cosPolar < 0.0f ? -1e-6f : 1e-6f); const float cone = safeDiv(sin(polar), safeCos) / safeSqrt(2.0f); const float sinTerm = clamp(sin(3.0f * angle), -1.0f, 1.0f); const float chromaGain = safeDiv(1.0f, 2.0f * cos(safeAcosUnit(sinTerm) / 3.0f), 0.0f, 1e-6f); chroma = chromaGain > 1e-6f ? safeDiv(cone, chromaGain) : 0.0f; if (allowOverflow && chroma < 0.0f) { chroma = -chroma; hue += 0.5f; if (hue >= 1.0f) hue -= 1.0f; } } outHue = hue; outChroma = allowOverflow ? max(chroma, 0.0f) : clamp01Safe(chroma); outValue = allowOverflow ? maxRgb : clamp01Safe(maxRgb); }
inline void rgbToRgbConeFast(float r, float g, float b, thread float& outMagnitude, thread float& outHue, thread float& outPolar) { constexpr float kTau = 6.28318530717958647692f; r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b; const float rotY = 0.70710678118f * g - 0.70710678118f * b; const float rotZ = 0.57735026919f * (r + g + b); const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ); float hue = safeAtan2(rotY, rotX); if (hue < 0.0f) hue += kTau; const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ); outMagnitude = clamp01Safe(radius * 0.576f); outHue = safeDiv(hue, kTau); outPolar = clamp01Safe(safeDiv(polar, kPolarMax)); }
inline void rgbToJpConicalFast(float r, float g, float b, bool allowOverflow, thread float& outMagnitude, thread float& outHue, thread float& outPolar) { constexpr float kPi = 3.14159265358979323846f; constexpr float kTau = 6.28318530717958647692f; const float kAsinInvSqrt2 = safeAsinUnit(safeDiv(1.0f, safeSqrt(2.0f))); const float kAsinInvSqrt3 = safeAsinUnit(safeDiv(1.0f, safeSqrt(3.0f))); const float kHueCoef1 = safeDiv(1.0f, 2.0f - safeDiv(kAsinInvSqrt2, kAsinInvSqrt3), 0.0f, 1e-6f); if (!allowOverflow) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); } const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b; const float rotY = 0.70710678118f * g - 0.70710678118f * b; const float rotZ = 0.57735026919f * (r + g + b); const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ); float hue = safeAtan2(rotY, rotX); if (hue < 0.0f) hue += kTau; const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ); const float huecoef2 = 2.0f * polar * sin((2.0f * kPi / 3.0f) - wrapPeriodSafe(hue, kPi / 3.0f)) / safeSqrt(3.0f); const float huemag = (safeDiv(safeAcosUnit(cos(3.0f * hue + kPi)), kPi * kHueCoef1, 0.0f, 1e-6f) + (safeDiv(kAsinInvSqrt2, kAsinInvSqrt3, 0.0f, 1e-6f) - 1.0f)) * huecoef2; const float satmag = sin(huemag + kAsinInvSqrt3); float magnitude = radius * satmag; if (allowOverflow && magnitude < 0.0f) { magnitude = -magnitude; hue += kPi; if (hue >= kTau) hue -= kTau; } outMagnitude = allowOverflow ? magnitude : clamp01Safe(magnitude); outHue = safeDiv(hue, kTau); outPolar = allowOverflow ? max(safeDiv(polar, kPolarMax), 0.0f) : clamp01Safe(safeDiv(polar, kPolarMax)); }
inline void rgbToReuleauxFast(float r, float g, float b, bool allowOverflow, thread float& outHue, thread float& outSat, thread float& outValue) { constexpr float kPi = 3.14159265358979323846f; constexpr float kTau = 6.28318530717958647692f; constexpr float kMaxSat = 1.41421356237f; if (!allowOverflow) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); } const float rotX = 0.33333333333f * (2.0f * r - g - b) * 0.70710678118f; const float rotY = (g - b) * 0.40824829046f; const float rotZ = (r + g + b) / 3.0f; float hue = kPi - safeAtan2(rotY, -rotX); if (hue < 0.0f) hue += kTau; if (hue >= kTau) hue = wrapPeriodSafe(hue, kTau); float sat = fabs(rotZ) <= 1e-6f ? 0.0f : safeDiv(safeHypot2(rotX, rotY), rotZ); if (allowOverflow && sat < 0.0f) { sat = -sat; hue += kPi; if (hue >= kTau) hue -= kTau; } outHue = safeDiv(hue, kTau); outSat = allowOverflow ? safeDiv(sat, kMaxSat) : clamp(safeDiv(sat, kMaxSat), 0.0f, 1.0f); outValue = allowOverflow ? max(r, max(g, b)) : clamp01Safe(max(r, max(g, b))); }

inline float normalizedNeutralRadiusForPoint(constant WholeImageRequestGpu& request, float r, float g, float b) {
  switch (request.plotMode) {
    case 0: { const float rr = clamp01Safe(r); const float gg = clamp01Safe(g); const float bb = clamp01Safe(b); const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb; const float rotY = 0.70710678118f * gg - 0.70710678118f * bb; return clamp(safeDiv(safeHypot2(rotX, rotY), kRgbAxisMaxRadius), 0.0f, 1.0f); }
    case 1: { if (request.circularHsl != 0) { float h = 0.0f; float radius = 0.0f; float l = 0.0f; rgbToPlotCircularHslFast(r, g, b, h, radius, l); return clamp(radius, 0.0f, 1.0f); } const float cMax = max(r, max(g, b)); const float cMin = min(r, min(g, b)); return clamp(cMax - cMin, 0.0f, 1.0f); }
    case 2: { if (request.circularHsv != 0) { float h = 0.0f; float radius = 0.0f; float v = 0.0f; rgbToPlotCircularHsvFast(r, g, b, h, radius, v); return clamp(radius, 0.0f, 1.0f); } float x = 0.0f; float z = 0.0f; rgbToHsvHexconePlaneFast(r, g, b, x, z); return clamp(safeHypot2(x, z), 0.0f, 1.0f); }
    case 3: { float h = 0.0f; float chroma = 0.0f; float light = 0.0f; rgbToChenFast(r, g, b, request.showOverflow != 0, h, chroma, light); const float polar = chroma / kChenPolarScale; const float radius = light * sin(polar) / kRgbAxisMaxRadius; return clamp(radius, 0.0f, 1.0f); }
    case 4: { float magnitude = 0.0f; float hue = 0.0f; float polar = 0.0f; rgbToRgbConeFast(r, g, b, magnitude, hue, polar); const float radial = magnitude * sin(polar * kPolarMax); return clamp(radial / sin(kPolarMax), 0.0f, 1.0f); }
    case 5: { float magnitude = 0.0f; float hue = 0.0f; float polar = 0.0f; rgbToJpConicalFast(r, g, b, request.showOverflow != 0, magnitude, hue, polar); const float radial = magnitude * sin(polar * kPolarMax); return clamp(radial / sin(kPolarMax), 0.0f, 1.0f); }
    case 6: { float hue = 0.0f; float chroma = 0.0f; float value = 0.0f; rgbToNormConeCoordsFast(r, g, b, request.normConeNormalized != 0, request.showOverflow != 0, hue, chroma, value); return clamp(chroma, 0.0f, 1.0f); }
    case 7: { float hue = 0.0f; float sat = 0.0f; float value = 0.0f; rgbToReuleauxFast(r, g, b, request.showOverflow != 0, hue, sat, value); return clamp(sat, 0.0f, 1.0f); }
    default: return 0.0f;
  }
}

inline float neutralRadiusAcceptanceProbability(constant WholeImageRequestGpu& request, float normalizedRadius) { if (request.neutralRadiusEnabled == 0) return 1.0f; const float threshold = effectiveNeutralRadiusThresholdSafe(request.neutralRadius); if (threshold <= 1e-6f) return normalizedRadius <= threshold + 1e-6f ? 1.0f : 0.0f; if (normalizedRadius > threshold + 1e-6f) return 0.0f; const float clippedFraction = clamp(1.0f - threshold, 0.0f, 1.0f); const float normalizedInside = clamp(safeDiv(normalizedRadius, threshold), 0.0f, 1.0f); const float edgePenalty = 0.78f * clippedFraction; return clamp(1.0f - edgePenalty * safePowPos(normalizedInside, 1.35f, 0.0f), 0.0f, 1.0f); }
inline void sampleUvForAttempt(constant WholeImageRequestGpu& request, int attemptIndex, thread float& outU, thread float& outV) { float u = 0.0f; float v = 0.0f; switch (request.samplingMode) { case 1: { const int grid = max(1, (int)ceil(safeSqrt((float)request.pointCount))); const int gx = attemptIndex % grid; const int gy = attemptIndex / grid; u = safeDiv((float)gx + unitHash01Fast((uint)(attemptIndex * 2 + 1)), (float)grid); v = safeDiv((float)gy + unitHash01Fast((uint)(attemptIndex * 2 + 2)), (float)grid); break; } case 2: u = unitHash01Fast((uint)(attemptIndex * 2 + 11)); v = unitHash01Fast((uint)(attemptIndex * 2 + 37)); break; default: { const int grid = max(1, (int)ceil(safeSqrt((float)request.pointCount))); const int gx = attemptIndex % grid; const int gy = attemptIndex / grid; u = safeDiv((float)gx + 0.5f, (float)grid); v = safeDiv((float)gy + 0.5f, (float)grid); break; } } outU = clamp(u, 0.0f, 1.0f); outV = clamp(v, 0.0f, 1.0f); }
inline int occupancyBinComponent(constant WholeImageRequestGpu& request, float value) { if (request.preserveOverflow == 0) return clamp((int)floor(clamp01Safe(value) * 16.0f), 0, 15); if (value < 0.0f) return 0; if (value > 1.0f) return 17; return 1 + clamp((int)floor(value * 16.0f), 0, 15); }
inline int occupancyBinIndex(constant WholeImageRequestGpu& request, float r, float g, float b) { const int binsPerAxis = request.preserveOverflow != 0 ? 18 : 16; return (occupancyBinComponent(request, r) * binsPerAxis + occupancyBinComponent(request, g)) * binsPerAxis + occupancyBinComponent(request, b); }
inline bool loadPixel(constant WholeImageRequestGpu& request, device const float* srcFloats, int x, int y, thread float& r, thread float& g, thread float& b) { const int px = request.originX + x; const int py = request.originY + y; if (px < 0 || py < 0) return false; const uint base = (uint)py * request.srcRowFloats + (uint)px * 4u; r = srcFloats[base + 0u]; g = srcFloats[base + 1u]; b = srcFloats[base + 2u]; return true; }
)METAL";

const char* kCloudMetalSourcePart3 = R"METAL(
kernel void primaryPassKernel(device const float* srcFloats [[buffer(0)]], device Sample* primaryOut [[buffer(1)]], device atomic_uint* primaryCount [[buffer(2)]], device atomic_uint* occupancyBins [[buffer(3)]], constant WholeImageRequestGpu& request [[buffer(4)]], uint tid [[thread_position_in_grid]]) {
  const int attemptIndex = (int)tid; if (attemptIndex >= request.maxPrimaryAttempts) return;
  float u = 0.0f; float v = 0.0f; sampleUvForAttempt(request, attemptIndex, u, v);
  const int sx = clamp((int)(u * (float)(request.scaledWidth - 1)), 0, request.scaledWidth - 1);
  const int sy = clamp((int)(v * (float)(request.scaledHeight - 1)), 0, request.scaledHeight - 1);
  const int x = clamp((int)(((float(sx) + 0.5f) / float(request.scaledWidth)) * float(request.width)), 0, request.width - 1);
  const int y = clamp((int)(((float(sy) + 0.5f) / float(request.scaledHeight)) * float(request.height)), 0, request.height - 1);
  float r = 0.0f; float g = 0.0f; float b = 0.0f; if (!loadPixel(request, srcFloats, x, y, r, g, b)) return;
  if (request.preserveOverflow == 0) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); }
  if (request.plotDisplayLinearEnabled != 0) { r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer); g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer); b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer); }
  if (request.neutralRadiusEnabled != 0) { const float normalizedRadius = normalizedNeutralRadiusForPoint(request, r, g, b); const float acceptProbability = neutralRadiusAcceptanceProbability(request, normalizedRadius); if (acceptProbability <= 0.0f) return; if (acceptProbability < 0.999999f) { const uint samplingSeed = (uint)(attemptIndex * 2654435761u) ^ (uint)(x * 911u + y * 3571u); if (unitHash01Fast(samplingSeed) > acceptProbability) return; } }
  const uint outIndex = atomic_fetch_add_explicit(primaryCount, 1u, memory_order_relaxed); if ((int)outIndex >= request.pointCount) return;
  Sample sample; sample.xNorm = (float(x) + 0.5f) / float(request.width); sample.yNorm = (float(y) + 0.5f) / float(request.height); sample.zReserved = 0.0f; sample.r = r; sample.g = g; sample.b = b; primaryOut[outIndex] = sample;
  atomic_fetch_add_explicit(&occupancyBins[occupancyBinIndex(request, r, g, b)], 1u, memory_order_relaxed);
}

kernel void occupancyPassKernel(device const float* srcFloats [[buffer(0)]], device OccupancyCandidate* candidateOut [[buffer(1)]], device atomic_uint* candidateCount [[buffer(2)]], constant WholeImageRequestGpu& request [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
  const int attemptIndex = (int)tid; if (attemptIndex >= request.maxCandidateAttempts) return;
  const float u = haltonFast((uint)(attemptIndex + 1), 2u); const float v = haltonFast((uint)(attemptIndex + 1), 3u);
  const int sx = clamp((int)(u * (float)(request.scaledWidth - 1)), 0, request.scaledWidth - 1);
  const int sy = clamp((int)(v * (float)(request.scaledHeight - 1)), 0, request.scaledHeight - 1);
  const int x = clamp((int)(((float(sx) + 0.5f) / float(request.scaledWidth)) * float(request.width)), 0, request.width - 1);
  const int y = clamp((int)(((float(sy) + 0.5f) / float(request.scaledHeight)) * float(request.height)), 0, request.height - 1);
  float r = 0.0f; float g = 0.0f; float b = 0.0f; if (!loadPixel(request, srcFloats, x, y, r, g, b)) return;
  if (request.preserveOverflow == 0) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); }
  if (request.plotDisplayLinearEnabled != 0) { r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer); g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer); b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer); }
  float normalizedRadius = 0.0f;
  if (request.neutralRadiusEnabled != 0) { normalizedRadius = normalizedNeutralRadiusForPoint(request, r, g, b); const float acceptProbability = neutralRadiusAcceptanceProbability(request, normalizedRadius); if (acceptProbability <= 0.0f) return; if (acceptProbability < 0.999999f) { const uint samplingSeed = (uint)((attemptIndex + 1) * 2246822519u) ^ (uint)(x * 977u + y * 4051u); if (unitHash01Fast(samplingSeed) > acceptProbability) return; } }
  const uint outIndex = atomic_fetch_add_explicit(candidateCount, 1u, memory_order_relaxed); if ((int)outIndex >= request.candidateTarget) return;
  OccupancyCandidate candidate; candidate.sample.xNorm = (float(x) + 0.5f) / float(request.width); candidate.sample.yNorm = (float(y) + 0.5f) / float(request.height); candidate.sample.zReserved = 0.0f; candidate.sample.r = r; candidate.sample.g = g; candidate.sample.b = b; candidate.normalizedNeutralRadius = normalizedRadius; candidate.bin = occupancyBinIndex(request, r, g, b); candidate.tie = (uint)attemptIndex; candidateOut[outIndex] = candidate;
}

kernel void stripCubeKernel(device const float* srcFloats [[buffer(0)]], device Sample* outSamples [[buffer(1)]], constant StripRequestGpu& request [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  const int index = (int)tid; const int resolution = request.resolution; const int total = resolution * resolution * resolution; if (index >= total) return;
  const int denom = max(1, resolution - 1); const int bz = index / (resolution * resolution); const int rem = index % (resolution * resolution); const int gy = rem / resolution; const int rx = rem % resolution;
  const float layerStart = float(bz) * request.cellWidth; const float localX = layerStart + (request.cellWidth <= 1.0f ? 0.0f : float(rx) / float(denom) * max(0.0f, request.cellWidth - 1.0f));
  const int x = clamp((int)rint(localX), 0, request.width - 1); const int y = request.cubeY1 + clamp((int)rint((float(gy) / float(denom)) * float(max(0, request.stripHeight - 1))), 0, max(0, request.stripHeight - 1));
  const int px = request.originX + x; const int py = request.originY + y; if (px < 0 || py < 0) return; const uint base = (uint)py * request.srcRowFloats + (uint)px * 4u;
  float r = srcFloats[base + 0u]; float g = srcFloats[base + 1u]; float b = srcFloats[base + 2u];
  if (request.preserveOverflow == 0) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); }
  if (request.plotDisplayLinearEnabled != 0) { r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer); g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer); b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer); }
  Sample sample; sample.xNorm = r; sample.yNorm = g; sample.zReserved = b; sample.r = r; sample.g = g; sample.b = b; outSamples[index] = sample;
}

kernel void stripRampKernel(device const float* srcFloats [[buffer(0)]], device Sample* outSamples [[buffer(1)]], constant StripRequestGpu& request [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  const int index = (int)tid; const int sampleCols = request.width; const int sampleRows = request.rampSampleRows; const int total = sampleCols * sampleRows; if (index >= total) return;
  const int rowIndex = index / sampleCols; const int colIndex = index % sampleCols; const int rowDenom = max(1, sampleRows - 1);
  const int y = request.rampY1 + clamp((int)rint((float(rowIndex) / float(rowDenom)) * float(max(0, request.rampHeight - 1))), 0, max(0, request.rampHeight - 1)); const int x = clamp(colIndex, 0, request.width - 1);
  const int px = request.originX + x; const int py = request.originY + y; if (px < 0 || py < 0) return; const uint base = (uint)py * request.srcRowFloats + (uint)px * 4u;
  float r = srcFloats[base + 0u]; float g = srcFloats[base + 1u]; float b = srcFloats[base + 2u];
  if (request.preserveOverflow == 0) { r = clamp01Safe(r); g = clamp01Safe(g); b = clamp01Safe(b); }
  if (request.plotDisplayLinearEnabled != 0) { r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer); g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer); b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer); }
  Sample sample; sample.xNorm = r; sample.yNorm = g; sample.zReserved = b; sample.r = r; sample.g = g; sample.b = b; outSamples[index] = sample;
}

kernel void packCombinedKernel(device const Sample* stripSamples [[buffer(0)]],
                               device const Sample* primarySamples [[buffer(1)]],
                               device const atomic_uint* primaryCount [[buffer(2)]],
                               device Sample* combinedOut [[buffer(3)]],
                               constant CombinedPackRequestGpu& request [[buffer(4)]],
                               uint tid [[thread_position_in_grid]]) {
  const int index = (int)tid;
  const int primaryAccepted = (int)atomic_load_explicit(primaryCount, memory_order_relaxed);
  const int total = request.stripCount + min(primaryAccepted, request.maxPrimaryCount);
  if (index >= total) return;
  if (index < request.stripCount) {
    combinedOut[index] = stripSamples[index];
  } else {
    const int primaryIndex = index - request.stripCount;
    combinedOut[index] = primarySamples[primaryIndex];
  }
}

kernel void appendSelectKernel(device const OccupancyCandidate* candidateIn [[buffer(0)]],
                               device const atomic_uint* occupancyBins [[buffer(1)]],
                               device atomic_uint* appendCount [[buffer(2)]],
                               device Sample* appendOut [[buffer(3)]],
                               constant AppendSelectRequestGpu& request [[buffer(4)]],
                               uint tid [[thread_position_in_grid]]) {
  const int index = (int)tid;
  if (index >= request.candidateCount) return;
  const OccupancyCandidate candidate = candidateIn[index];
  if (candidate.bin < 0) return;
  const uint occ = atomic_load_explicit(&occupancyBins[candidate.bin], memory_order_relaxed);
  if ((int)occ != request.occupancyThreshold) return;
  if (candidate.normalizedNeutralRadius < request.radiusMin) return;
  const bool lastBucket = request.radiusMax >= 0.9999f;
  if (lastBucket) {
    if (candidate.normalizedNeutralRadius > request.radiusMax) return;
  } else {
    if (candidate.normalizedNeutralRadius >= request.radiusMax) return;
  }
  const uint outIndex = atomic_fetch_add_explicit(appendCount, 1u, memory_order_relaxed);
  if ((int)outIndex >= request.extraPointCount) return;
  appendOut[outIndex] = candidate.sample;
}
)METAL";

bool ensurePipelines(id<MTLDevice> device, PipelineBundle* out, std::string* error) {
  if (!device || !out) return false;
  std::lock_guard<std::mutex> lock(gPipelineMutex);
  if (gPipelines.device == device && gPipelines.primary != nil && gPipelines.occupancy != nil &&
      gPipelines.stripCube != nil && gPipelines.stripRamp != nil && gPipelines.packCombined != nil &&
      gPipelines.appendSelect != nil) { *out = gPipelines; return true; }
  NSError* compileError = nil;
  const std::string sourceString = std::string(kCloudMetalSourcePart1) + kCloudMetalSourcePart2 + kCloudMetalSourcePart3;
  NSString* source = [NSString stringWithUTF8String:sourceString.c_str()];
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = NO;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&compileError];
  if (library == nil) { if (error) *error = nsErrorString(compileError); return false; }
  auto makePipeline = [&](NSString* name, id<MTLComputePipelineState>* outState) -> bool { NSError* localError = nil; id<MTLFunction> function = [library newFunctionWithName:name]; if (function == nil) { if (error) *error = std::string("missing Metal function: ") + (name.UTF8String ? name.UTF8String : "unknown"); return false; } id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&localError]; if (pipeline == nil) { if (error) *error = nsErrorString(localError); return false; } *outState = pipeline; return true; };
  PipelineBundle built{}; built.device = device;
  if (!makePipeline(@"primaryPassKernel", &built.primary) ||
      !makePipeline(@"occupancyPassKernel", &built.occupancy) ||
      !makePipeline(@"stripCubeKernel", &built.stripCube) ||
      !makePipeline(@"stripRampKernel", &built.stripRamp) ||
      !makePipeline(@"packCombinedKernel", &built.packCombined) ||
      !makePipeline(@"appendSelectKernel", &built.appendSelect)) return false;
  gPipelines = built; *out = gPipelines; return true;
}

void dispatch1D(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline, NSUInteger count) {
  if (!encoder || !pipeline || count == 0) return;
  const NSUInteger width = std::min<NSUInteger>(256, std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup));
  [encoder setComputePipelineState:pipeline];
  [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

template <typename T>
std::vector<T> readSharedBuffer(id<MTLBuffer> buffer, std::size_t count) {
  std::vector<T> out(count);
  if (count == 0 || buffer == nil) return out;
  std::memcpy(out.data(), buffer.contents, count * sizeof(T));
  return out;
}

std::vector<int> sortedUniqueOccupancyValuesAscending(const std::vector<int>& occupancy) {
  std::vector<int> values;
  values.reserve(occupancy.size());
  for (int value : occupancy) {
    if (value >= 0) values.push_back(value);
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

std::vector<Sample> selectOccupancySamplesGpu(
    id<MTLCommandQueue> queue,
    const PipelineBundle& pipelines,
    id<MTLBuffer> candidateBuffer,
    uint32_t candidateAccepted,
    id<MTLBuffer> occupancyBuffer,
    const std::vector<int>& occupancy,
    int extraPointCount,
    bool neutralRadiusEnabled,
    std::string* error) {
  if (queue == nil || candidateBuffer == nil || occupancyBuffer == nil || candidateAccepted == 0 || extraPointCount <= 0) {
    return {};
  }
  id<MTLBuffer> appendBuffer =
      [queue.device newBufferWithLength:static_cast<std::size_t>(extraPointCount) * sizeof(Sample) options:MTLResourceStorageModeShared];
  id<MTLBuffer> appendCountBuffer = [queue.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  if (appendBuffer == nil || appendCountBuffer == nil) {
    if (error) *error = "metal-append-buffer-allocation-failed";
    return {};
  }
  std::memset(appendCountBuffer.contents, 0, sizeof(uint32_t));

  const std::vector<int> thresholds = sortedUniqueOccupancyValuesAscending(occupancy);
  const int radiusBuckets = neutralRadiusEnabled ? 6 : 1;
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  if (cmd == nil) {
    if (error) *error = "metal-append-command-buffer-failed";
    return {};
  }

  for (int threshold : thresholds) {
    for (int bucket = 0; bucket < radiusBuckets; ++bucket) {
      AppendSelectRequestGpu request{};
      request.candidateCount = static_cast<int>(candidateAccepted);
      request.extraPointCount = extraPointCount;
      request.occupancyThreshold = threshold;
      request.radiusMin = neutralRadiusEnabled ? (static_cast<float>(bucket) / static_cast<float>(radiusBuckets)) : 0.0f;
      request.radiusMax = neutralRadiusEnabled ? (static_cast<float>(bucket + 1) / static_cast<float>(radiusBuckets)) : 1.0f;

      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        if (error) *error = "metal-append-encoder-failed";
        return {};
      }
      [encoder setBuffer:candidateBuffer offset:0 atIndex:0];
      [encoder setBuffer:occupancyBuffer offset:0 atIndex:1];
      [encoder setBuffer:appendCountBuffer offset:0 atIndex:2];
      [encoder setBuffer:appendBuffer offset:0 atIndex:3];
      [encoder setBytes:&request length:sizeof(request) atIndex:4];
      dispatch1D(encoder, pipelines.appendSelect, static_cast<NSUInteger>(candidateAccepted));
      [encoder endEncoding];
    }
  }

  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) {
    if (error) *error = "metal-append-command-failed";
    return {};
  }

  uint32_t accepted = *reinterpret_cast<uint32_t*>(appendCountBuffer.contents);
  accepted = std::min<uint32_t>(accepted, static_cast<uint32_t>(extraPointCount));
  return readSharedBuffer<Sample>(appendBuffer, static_cast<std::size_t>(accepted));
}

}  // namespace

bool buildWholeImageCloud(const Request& request, Result* out) {
  if (!out) return false;
  *out = Result{};
  @autoreleasepool {
    if (!request.srcMetalBuffer || !request.metalCommandQueue || request.srcRowBytes == 0 ||
        request.width <= 0 || request.height <= 0 || request.pointCount <= 0) {
      out->error = "invalid-request";
      return false;
    }
    if (!validateFloatRowBytes(request.srcRowBytes)) {
      out->error = "invalid-row-bytes";
      return false;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)request.metalCommandQueue;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)request.srcMetalBuffer;
    if (queue == nil || src == nil || queue.device == nil) { out->error = "metal-unavailable"; return false; }
    PipelineBundle pipelines{};
    if (!ensurePipelines(queue.device, &pipelines, &out->error)) return false;

    const int binsPerAxis = request.preserveOverflow != 0 ? 18 : 16;
    const std::size_t occupancyCount = static_cast<std::size_t>(binsPerAxis * binsPerAxis * binsPerAxis);
    id<MTLBuffer> primaryBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(request.pointCount) * sizeof(Sample) options:MTLResourceStorageModeShared];
    id<MTLBuffer> primaryCountBuffer = [queue.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> occupancyBuffer = [queue.device newBufferWithLength:occupancyCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (primaryBuffer == nil || primaryCountBuffer == nil || occupancyBuffer == nil) { out->error = "metal-buffer-allocation-failed"; return false; }
    std::memset(primaryCountBuffer.contents, 0, sizeof(uint32_t));
    std::memset(occupancyBuffer.contents, 0, occupancyCount * sizeof(uint32_t));

    id<MTLBuffer> candidateBuffer = nil;
    id<MTLBuffer> candidateCountBuffer = nil;
    if (request.occupancyFill != 0 && request.candidateTarget > 0 && request.maxCandidateAttempts > 0) {
      candidateBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(request.candidateTarget) * sizeof(OccupancyCandidate) options:MTLResourceStorageModeShared];
      candidateCountBuffer = [queue.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
      if (candidateBuffer == nil || candidateCountBuffer == nil) { out->error = "metal-candidate-buffer-allocation-failed"; return false; }
      std::memset(candidateCountBuffer.contents, 0, sizeof(uint32_t));
    }

    WholeImageRequestGpu gpuRequest{};
    gpuRequest.width = request.width; gpuRequest.height = request.height;
    gpuRequest.originX = request.originX; gpuRequest.originY = request.originY;
    gpuRequest.srcRowFloats = static_cast<uint32_t>(request.srcRowBytes / sizeof(float));
    gpuRequest.scaledWidth = request.scaledWidth; gpuRequest.scaledHeight = request.scaledHeight;
    gpuRequest.pointCount = request.pointCount; gpuRequest.candidateTarget = request.candidateTarget;
    gpuRequest.maxPrimaryAttempts = request.maxPrimaryAttempts; gpuRequest.maxCandidateAttempts = request.maxCandidateAttempts;
    gpuRequest.samplingMode = request.samplingMode; gpuRequest.preserveOverflow = request.preserveOverflow;
    gpuRequest.plotMode = request.plotMode; gpuRequest.circularHsl = request.circularHsl; gpuRequest.circularHsv = request.circularHsv;
    gpuRequest.normConeNormalized = request.normConeNormalized; gpuRequest.showOverflow = request.showOverflow;
    gpuRequest.plotDisplayLinearEnabled = request.plotDisplayLinearEnabled; gpuRequest.plotDisplayLinearTransfer = request.plotDisplayLinearTransfer;
    gpuRequest.neutralRadiusEnabled = request.neutralRadiusEnabled; gpuRequest.neutralRadius = request.neutralRadius;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { out->error = "metal-command-buffer-failed"; return false; }
    if (request.maxPrimaryAttempts > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) { out->error = "metal-primary-encoder-failed"; return false; }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:primaryBuffer offset:0 atIndex:1];
      [encoder setBuffer:primaryCountBuffer offset:0 atIndex:2];
      [encoder setBuffer:occupancyBuffer offset:0 atIndex:3];
      [encoder setBytes:&gpuRequest length:sizeof(gpuRequest) atIndex:4];
      dispatch1D(encoder, pipelines.primary, static_cast<NSUInteger>(request.maxPrimaryAttempts));
      [encoder endEncoding];
    }
    if (candidateBuffer != nil && candidateCountBuffer != nil) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) { out->error = "metal-occupancy-encoder-failed"; return false; }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:candidateBuffer offset:0 atIndex:1];
      [encoder setBuffer:candidateCountBuffer offset:0 atIndex:2];
      [encoder setBytes:&gpuRequest length:sizeof(gpuRequest) atIndex:3];
      dispatch1D(encoder, pipelines.occupancy, static_cast<NSUInteger>(request.maxCandidateAttempts));
      [encoder endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) { out->error = "metal-cloud-command-failed"; return false; }

    uint32_t primaryAccepted = *reinterpret_cast<uint32_t*>(primaryCountBuffer.contents);
    primaryAccepted = std::min<uint32_t>(primaryAccepted, static_cast<uint32_t>(request.pointCount));
    out->primaryAttempts = request.maxPrimaryAttempts;
    out->primaryAccepted = static_cast<int>(primaryAccepted);
    out->extraPointCount = request.extraPointCount;
    const auto occupancyRaw = readSharedBuffer<uint32_t>(occupancyBuffer, occupancyCount);
    out->occupancy.assign(occupancyRaw.begin(), occupancyRaw.end());
    out->primarySamples = readSharedBuffer<Sample>(primaryBuffer, static_cast<std::size_t>(primaryAccepted));
    if (candidateBuffer != nil && candidateCountBuffer != nil) {
      uint32_t candidateAccepted = *reinterpret_cast<uint32_t*>(candidateCountBuffer.contents);
      candidateAccepted = std::min<uint32_t>(candidateAccepted, static_cast<uint32_t>(request.candidateTarget));
      if (candidateAccepted > 0 && request.extraPointCount > 0) {
        out->appendedSamples = selectOccupancySamplesGpu(queue,
                                                         pipelines,
                                                         candidateBuffer,
                                                         candidateAccepted,
                                                         occupancyBuffer,
                                                         out->occupancy,
                                                         request.extraPointCount,
                                                         request.neutralRadiusEnabled != 0,
                                                         &out->error);
        if (out->appendedSamples.empty() && !out->error.empty()) return false;
      }
    }
    out->success = true;
    return true;
  }
}

bool buildIdentityStripCloud(const StripRequest& request, StripResult* out) {
  if (!out) return false;
  *out = StripResult{};
  @autoreleasepool {
    if (!request.srcMetalBuffer || !request.metalCommandQueue || request.srcRowBytes == 0 ||
        request.width <= 0 || request.height <= 0 || request.resolution <= 0) {
      out->error = "invalid-strip-request";
      return false;
    }
    if (!validateFloatRowBytes(request.srcRowBytes)) {
      out->error = "invalid-strip-row-bytes";
      return false;
    }
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)request.metalCommandQueue;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)request.srcMetalBuffer;
    if (queue == nil || src == nil || queue.device == nil) { out->error = "metal-unavailable"; return false; }
    PipelineBundle pipelines{};
    if (!ensurePipelines(queue.device, &pipelines, &out->error)) return false;

    int cubeCount = 0;
    if (request.readCube != 0 && !checkedCubeToInt(request.resolution, &cubeCount)) {
      out->error = "strip-cube-count-overflow";
      return false;
    }
    int rampCount = 0;
    if (request.readRamp != 0 &&
        !checkedProductToInt(request.width, std::max(0, request.rampSampleRows), &rampCount)) {
      out->error = "strip-ramp-count-overflow";
      return false;
    }
    const int totalCount = cubeCount + rampCount;
    if (totalCount <= 0) { out->error = "empty-strip-request"; return false; }

    id<MTLBuffer> samplesBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(totalCount) * sizeof(Sample) options:MTLResourceStorageModeShared];
    if (samplesBuffer == nil) { out->error = "metal-strip-buffer-allocation-failed"; return false; }

    StripRequestGpu gpuRequest{};
    gpuRequest.width = request.width; gpuRequest.height = request.height;
    gpuRequest.originX = request.originX; gpuRequest.originY = request.originY;
    gpuRequest.srcRowFloats = static_cast<uint32_t>(request.srcRowBytes / sizeof(float));
    gpuRequest.resolution = request.resolution; gpuRequest.preserveOverflow = request.preserveOverflow;
    gpuRequest.plotDisplayLinearEnabled = request.plotDisplayLinearEnabled; gpuRequest.plotDisplayLinearTransfer = request.plotDisplayLinearTransfer;
    gpuRequest.cubeY1 = request.cubeY1; gpuRequest.stripHeight = request.stripHeight;
    gpuRequest.rampY1 = request.rampY1; gpuRequest.rampHeight = request.rampHeight;
    gpuRequest.rampSampleRows = request.rampSampleRows; gpuRequest.cellWidth = request.cellWidth;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) { out->error = "metal-strip-command-buffer-failed"; return false; }
    int offset = 0;
    if (cubeCount > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) { out->error = "metal-strip-cube-encoder-failed"; return false; }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:samplesBuffer offset:static_cast<NSUInteger>(offset) * sizeof(Sample) atIndex:1];
      [encoder setBytes:&gpuRequest length:sizeof(gpuRequest) atIndex:2];
      dispatch1D(encoder, pipelines.stripCube, static_cast<NSUInteger>(cubeCount));
      [encoder endEncoding];
      offset += cubeCount;
    }
    if (rampCount > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) { out->error = "metal-strip-ramp-encoder-failed"; return false; }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:samplesBuffer offset:static_cast<NSUInteger>(offset) * sizeof(Sample) atIndex:1];
      [encoder setBytes:&gpuRequest length:sizeof(gpuRequest) atIndex:2];
      dispatch1D(encoder, pipelines.stripRamp, static_cast<NSUInteger>(rampCount));
      [encoder endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) { out->error = "metal-strip-command-failed"; return false; }
    out->samples = readSharedBuffer<Sample>(samplesBuffer, static_cast<std::size_t>(totalCount));
    out->success = true;
    return true;
  }
}

bool buildWholeImageAndIdentityStripCloud(const Request& wholeImageRequest, const StripRequest& stripRequest, CombinedResult* out) {
  if (!out) return false;
  *out = CombinedResult{};
  @autoreleasepool {
    if (!wholeImageRequest.srcMetalBuffer || !wholeImageRequest.metalCommandQueue || wholeImageRequest.srcRowBytes == 0 ||
        wholeImageRequest.width <= 0 || wholeImageRequest.height <= 0 || wholeImageRequest.pointCount <= 0) {
      out->error = "invalid-whole-image-request";
      return false;
    }
    if (!validateFloatRowBytes(wholeImageRequest.srcRowBytes)) {
      out->error = "invalid-whole-image-row-bytes";
      return false;
    }
    if (!stripRequest.srcMetalBuffer || !stripRequest.metalCommandQueue || stripRequest.srcRowBytes == 0 ||
        stripRequest.width <= 0 || stripRequest.height <= 0 || stripRequest.resolution <= 0) {
      out->error = "invalid-strip-request";
      return false;
    }
    if (!validateFloatRowBytes(stripRequest.srcRowBytes)) {
      out->error = "invalid-strip-row-bytes";
      return false;
    }
    if (wholeImageRequest.srcMetalBuffer != stripRequest.srcMetalBuffer ||
        wholeImageRequest.metalCommandQueue != stripRequest.metalCommandQueue ||
        wholeImageRequest.width != stripRequest.width ||
        wholeImageRequest.height != stripRequest.height) {
      out->error = "mismatched-combined-request";
      return false;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)wholeImageRequest.metalCommandQueue;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)wholeImageRequest.srcMetalBuffer;
    if (queue == nil || src == nil || queue.device == nil) {
      out->error = "metal-unavailable";
      return false;
    }

    PipelineBundle pipelines{};
    if (!ensurePipelines(queue.device, &pipelines, &out->error)) return false;

    const int binsPerAxis = wholeImageRequest.preserveOverflow != 0 ? 18 : 16;
    const std::size_t occupancyCount = static_cast<std::size_t>(binsPerAxis * binsPerAxis * binsPerAxis);
    id<MTLBuffer> primaryBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(wholeImageRequest.pointCount) * sizeof(Sample) options:MTLResourceStorageModeShared];
    id<MTLBuffer> primaryCountBuffer = [queue.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> occupancyBuffer = [queue.device newBufferWithLength:occupancyCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (primaryBuffer == nil || primaryCountBuffer == nil || occupancyBuffer == nil) {
      out->error = "metal-buffer-allocation-failed";
      return false;
    }
    std::memset(primaryCountBuffer.contents, 0, sizeof(uint32_t));
    std::memset(occupancyBuffer.contents, 0, occupancyCount * sizeof(uint32_t));

    id<MTLBuffer> candidateBuffer = nil;
    id<MTLBuffer> candidateCountBuffer = nil;
    if (wholeImageRequest.occupancyFill != 0 &&
        wholeImageRequest.candidateTarget > 0 &&
        wholeImageRequest.maxCandidateAttempts > 0) {
      candidateBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(wholeImageRequest.candidateTarget) * sizeof(OccupancyCandidate) options:MTLResourceStorageModeShared];
      candidateCountBuffer = [queue.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
      if (candidateBuffer == nil || candidateCountBuffer == nil) {
        out->error = "metal-candidate-buffer-allocation-failed";
        return false;
      }
      std::memset(candidateCountBuffer.contents, 0, sizeof(uint32_t));
    }

    int cubeCount = 0;
    if (stripRequest.readCube != 0 && !checkedCubeToInt(stripRequest.resolution, &cubeCount)) {
      out->error = "strip-cube-count-overflow";
      return false;
    }
    int rampCount = 0;
    if (stripRequest.readRamp != 0 &&
        !checkedProductToInt(stripRequest.width, std::max(0, stripRequest.rampSampleRows), &rampCount)) {
      out->error = "strip-ramp-count-overflow";
      return false;
    }
    const int totalStripCount = cubeCount + rampCount;
    if (totalStripCount <= 0) {
      out->error = "empty-strip-request";
      return false;
    }
    id<MTLBuffer> stripSamplesBuffer = [queue.device newBufferWithLength:static_cast<std::size_t>(totalStripCount) * sizeof(Sample) options:MTLResourceStorageModeShared];
    id<MTLBuffer> combinedSamplesBuffer =
        [queue.device newBufferWithLength:static_cast<std::size_t>(totalStripCount + wholeImageRequest.pointCount) * sizeof(Sample)
                                  options:MTLResourceStorageModeShared];
    if (stripSamplesBuffer == nil || combinedSamplesBuffer == nil) {
      out->error = "metal-strip-buffer-allocation-failed";
      return false;
    }

    WholeImageRequestGpu wholeGpuRequest{};
    wholeGpuRequest.width = wholeImageRequest.width;
    wholeGpuRequest.height = wholeImageRequest.height;
    wholeGpuRequest.originX = wholeImageRequest.originX;
    wholeGpuRequest.originY = wholeImageRequest.originY;
    wholeGpuRequest.srcRowFloats = static_cast<uint32_t>(wholeImageRequest.srcRowBytes / sizeof(float));
    wholeGpuRequest.scaledWidth = wholeImageRequest.scaledWidth;
    wholeGpuRequest.scaledHeight = wholeImageRequest.scaledHeight;
    wholeGpuRequest.pointCount = wholeImageRequest.pointCount;
    wholeGpuRequest.candidateTarget = wholeImageRequest.candidateTarget;
    wholeGpuRequest.maxPrimaryAttempts = wholeImageRequest.maxPrimaryAttempts;
    wholeGpuRequest.maxCandidateAttempts = wholeImageRequest.maxCandidateAttempts;
    wholeGpuRequest.samplingMode = wholeImageRequest.samplingMode;
    wholeGpuRequest.preserveOverflow = wholeImageRequest.preserveOverflow;
    wholeGpuRequest.plotMode = wholeImageRequest.plotMode;
    wholeGpuRequest.circularHsl = wholeImageRequest.circularHsl;
    wholeGpuRequest.circularHsv = wholeImageRequest.circularHsv;
    wholeGpuRequest.normConeNormalized = wholeImageRequest.normConeNormalized;
    wholeGpuRequest.showOverflow = wholeImageRequest.showOverflow;
    wholeGpuRequest.plotDisplayLinearEnabled = wholeImageRequest.plotDisplayLinearEnabled;
    wholeGpuRequest.plotDisplayLinearTransfer = wholeImageRequest.plotDisplayLinearTransfer;
    wholeGpuRequest.neutralRadiusEnabled = wholeImageRequest.neutralRadiusEnabled;
    wholeGpuRequest.neutralRadius = wholeImageRequest.neutralRadius;

    StripRequestGpu stripGpuRequest{};
    stripGpuRequest.width = stripRequest.width;
    stripGpuRequest.height = stripRequest.height;
    stripGpuRequest.originX = stripRequest.originX;
    stripGpuRequest.originY = stripRequest.originY;
    stripGpuRequest.srcRowFloats = static_cast<uint32_t>(stripRequest.srcRowBytes / sizeof(float));
    stripGpuRequest.resolution = stripRequest.resolution;
    stripGpuRequest.preserveOverflow = stripRequest.preserveOverflow;
    stripGpuRequest.plotDisplayLinearEnabled = stripRequest.plotDisplayLinearEnabled;
    stripGpuRequest.plotDisplayLinearTransfer = stripRequest.plotDisplayLinearTransfer;
    stripGpuRequest.cubeY1 = stripRequest.cubeY1;
    stripGpuRequest.stripHeight = stripRequest.stripHeight;
    stripGpuRequest.rampY1 = stripRequest.rampY1;
    stripGpuRequest.rampHeight = stripRequest.rampHeight;
    stripGpuRequest.rampSampleRows = stripRequest.rampSampleRows;
    stripGpuRequest.cellWidth = stripRequest.cellWidth;

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (cmd == nil) {
      out->error = "metal-command-buffer-failed";
      return false;
    }

    if (wholeImageRequest.maxPrimaryAttempts > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        out->error = "metal-primary-encoder-failed";
        return false;
      }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:primaryBuffer offset:0 atIndex:1];
      [encoder setBuffer:primaryCountBuffer offset:0 atIndex:2];
      [encoder setBuffer:occupancyBuffer offset:0 atIndex:3];
      [encoder setBytes:&wholeGpuRequest length:sizeof(wholeGpuRequest) atIndex:4];
      dispatch1D(encoder, pipelines.primary, static_cast<NSUInteger>(wholeImageRequest.maxPrimaryAttempts));
      [encoder endEncoding];
    }

    if (candidateBuffer != nil && candidateCountBuffer != nil) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        out->error = "metal-occupancy-encoder-failed";
        return false;
      }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:candidateBuffer offset:0 atIndex:1];
      [encoder setBuffer:candidateCountBuffer offset:0 atIndex:2];
      [encoder setBytes:&wholeGpuRequest length:sizeof(wholeGpuRequest) atIndex:3];
      dispatch1D(encoder, pipelines.occupancy, static_cast<NSUInteger>(wholeImageRequest.maxCandidateAttempts));
      [encoder endEncoding];
    }

    int stripOffset = 0;
    if (cubeCount > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        out->error = "metal-strip-cube-encoder-failed";
        return false;
      }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:stripSamplesBuffer offset:static_cast<NSUInteger>(stripOffset) * sizeof(Sample) atIndex:1];
      [encoder setBytes:&stripGpuRequest length:sizeof(stripGpuRequest) atIndex:2];
      dispatch1D(encoder, pipelines.stripCube, static_cast<NSUInteger>(cubeCount));
      [encoder endEncoding];
      stripOffset += cubeCount;
    }

    if (rampCount > 0) {
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        out->error = "metal-strip-ramp-encoder-failed";
        return false;
      }
      [encoder setBuffer:src offset:0 atIndex:0];
      [encoder setBuffer:stripSamplesBuffer offset:static_cast<NSUInteger>(stripOffset) * sizeof(Sample) atIndex:1];
      [encoder setBytes:&stripGpuRequest length:sizeof(stripGpuRequest) atIndex:2];
      dispatch1D(encoder, pipelines.stripRamp, static_cast<NSUInteger>(rampCount));
      [encoder endEncoding];
    }

    if (wholeImageRequest.pointCount > 0 || totalStripCount > 0) {
      CombinedPackRequestGpu packRequest{};
      packRequest.stripCount = totalStripCount;
      packRequest.maxPrimaryCount = wholeImageRequest.pointCount;
      id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
      if (encoder == nil) {
        out->error = "metal-pack-encoder-failed";
        return false;
      }
      [encoder setBuffer:stripSamplesBuffer offset:0 atIndex:0];
      [encoder setBuffer:primaryBuffer offset:0 atIndex:1];
      [encoder setBuffer:primaryCountBuffer offset:0 atIndex:2];
      [encoder setBuffer:combinedSamplesBuffer offset:0 atIndex:3];
      [encoder setBytes:&packRequest length:sizeof(packRequest) atIndex:4];
      dispatch1D(encoder, pipelines.packCombined, static_cast<NSUInteger>(totalStripCount + wholeImageRequest.pointCount));
      [encoder endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      out->error = "metal-combined-command-failed";
      return false;
    }

    uint32_t primaryAccepted = *reinterpret_cast<uint32_t*>(primaryCountBuffer.contents);
    primaryAccepted = std::min<uint32_t>(primaryAccepted, static_cast<uint32_t>(wholeImageRequest.pointCount));
    out->primaryAttempts = wholeImageRequest.maxPrimaryAttempts;
    out->primaryAccepted = static_cast<int>(primaryAccepted);
    out->extraPointCount = wholeImageRequest.extraPointCount;

    const auto occupancyRaw = readSharedBuffer<uint32_t>(occupancyBuffer, occupancyCount);
    out->occupancy.assign(occupancyRaw.begin(), occupancyRaw.end());
    out->stripSamples = readSharedBuffer<Sample>(stripSamplesBuffer, static_cast<std::size_t>(totalStripCount));
    out->combinedSamples = readSharedBuffer<Sample>(combinedSamplesBuffer, static_cast<std::size_t>(totalStripCount + primaryAccepted));

    if (candidateBuffer != nil && candidateCountBuffer != nil) {
      uint32_t candidateAccepted = *reinterpret_cast<uint32_t*>(candidateCountBuffer.contents);
      candidateAccepted = std::min<uint32_t>(candidateAccepted, static_cast<uint32_t>(wholeImageRequest.candidateTarget));
      if (candidateAccepted > 0 && wholeImageRequest.extraPointCount > 0) {
        out->appendedSamples = selectOccupancySamplesGpu(queue,
                                                         pipelines,
                                                         candidateBuffer,
                                                         candidateAccepted,
                                                         occupancyBuffer,
                                                         out->occupancy,
                                                         wholeImageRequest.extraPointCount,
                                                         wholeImageRequest.neutralRadiusEnabled != 0,
                                                         &out->error);
        if (out->appendedSamples.empty() && !out->error.empty()) return false;
      }
    }

    out->success = true;
    return true;
  }
}

bool copyHostBuffers(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    const float* overlayPixels,
    int overlayX,
    int overlayY,
    int overlayWidth,
    int overlayHeight) {
  if (!srcMetalBuffer || !dstMetalBuffer || !metalCommandQueue) return false;
  if (!validateRows(width, height, srcRowBytes, dstRowBytes)) return false;

  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCommandQueue;
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcMetalBuffer;
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstMetalBuffer;
  if (queue == nil || src == nil || dst == nil) return false;
  id<MTLDevice> device = queue.device;
  if (device == nil) return false;

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  if (cmd == nil) return false;
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  if (blit == nil) return false;

  const size_t srcOffset = offsetForOrigin(srcRowBytes, originX, originY);
  const size_t dstOffset = offsetForOrigin(dstRowBytes, originX, originY);
  if (!encodeCopyRows(blit, src, dst, srcOffset, dstOffset, srcRowBytes, dstRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  if (overlayPixels != nullptr && overlayWidth > 0 && overlayHeight > 0) {
    const size_t overlayPackedRowBytes = packedRowBytesForWidth(overlayWidth);
    const size_t overlayBytes = overlayPackedRowBytes * static_cast<size_t>(overlayHeight);
    id<MTLBuffer> overlayBuffer =
        [device newBufferWithBytes:overlayPixels length:overlayBytes options:MTLResourceStorageModeShared];
    if (overlayBuffer == nil) {
      [blit endEncoding];
      return false;
    }
    const size_t overlayDstOffset = offsetForOrigin(dstRowBytes, overlayX, overlayY);
    if (!encodeCopyRows(blit, overlayBuffer, dst, 0, overlayDstOffset, overlayPackedRowBytes, dstRowBytes, overlayWidth, overlayHeight)) {
      [blit endEncoding];
      return false;
    }
  }
  [blit endEncoding];
  [cmd commit];
  return true;
}

bool copyHostBuffersReadback(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    float* readbackSrc,
    size_t readbackSrcRowBytes) {
  if (!srcMetalBuffer || !dstMetalBuffer || !metalCommandQueue || !readbackSrc) return false;
  if (!validateRows(width, height, srcRowBytes, dstRowBytes)) return false;
  if (readbackSrcRowBytes < packedRowBytesForWidth(width)) return false;

  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCommandQueue;
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcMetalBuffer;
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstMetalBuffer;
  if (queue == nil || src == nil || dst == nil) return false;

  id<MTLDevice> device = queue.device;
  if (device == nil) return false;

  const size_t readbackBytes = readbackSrcRowBytes * static_cast<size_t>(height);
  id<MTLBuffer> readbackBuffer =
      [device newBufferWithLength:readbackBytes options:MTLResourceStorageModeShared];
  if (readbackBuffer == nil) return false;

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  if (cmd == nil) return false;
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  if (blit == nil) return false;

  const size_t srcOffset = offsetForOrigin(srcRowBytes, originX, originY);
  const size_t dstOffset = offsetForOrigin(dstRowBytes, originX, originY);
  if (!encodeCopyRows(blit, src, dst, srcOffset, dstOffset, srcRowBytes, dstRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  if (!encodeCopyRows(blit, src, readbackBuffer, srcOffset, 0, srcRowBytes, readbackSrcRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) return false;

  std::memcpy(readbackSrc, readbackBuffer.contents, readbackBytes);
  return true;
}

}  // namespace ChromaspaceMetal

#endif
