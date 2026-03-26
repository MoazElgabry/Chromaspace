#include "cuda/ChromaspaceCloudCuda.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace ChromaspaceCloudCuda {
namespace {

constexpr float kRgbAxisMaxRadius = 0.8164965809277260f;
constexpr float kPolarMax = 0.9553166181245093f;
constexpr float kChenPolarScale = 1.0467733744265997f;

template <typename T>
__host__ __device__ inline T clampValue(T value, T minValue, T maxValue) {
  return value < minValue ? minValue : (value > maxValue ? maxValue : value);
}

__host__ __device__ inline float clamp01Safe(float value) {
  return clampValue(value, 0.0f, 1.0f);
}

__host__ __device__ inline float safeDiv(float num, float den, float fallback, float eps) {
  return fabsf(den) < eps ? fallback : num / den;
}

__host__ __device__ inline float safeDiv(float num, float den) {
  return safeDiv(num, den, 0.0f, 1e-6f);
}

__host__ __device__ inline float safeSqrt(float value) {
  return sqrtf(fmaxf(value, 0.0f));
}

__host__ __device__ inline float safePowPos(float value, float exponent, float fallback) {
  return value <= 0.0f ? fallback : powf(value, exponent);
}

__host__ __device__ inline float safePowPos(float value, float exponent) {
  return safePowPos(value, exponent, 0.0f);
}

__host__ __device__ inline float safeExp2Clamped(float value) {
  return exp2f(clampValue(value, -126.0f, 126.0f));
}

__host__ __device__ inline float safeHypot2(float x, float y) {
  return safeSqrt(x * x + y * y);
}

__host__ __device__ inline float safeAtan2(float y, float x, float fallback) {
  return (fabsf(x) < 1e-8f && fabsf(y) < 1e-8f) ? fallback : atan2f(y, x);
}

__host__ __device__ inline float safeAtan2(float y, float x) {
  return safeAtan2(y, x, 0.0f);
}

__host__ __device__ inline float safeAcosUnit(float value) {
  return acosf(clampValue(value, -1.0f, 1.0f));
}

__host__ __device__ inline float safeAsinUnit(float value) {
  return asinf(clampValue(value, -1.0f, 1.0f));
}

__host__ __device__ inline float wrapUnitSafe(float value) {
  float wrapped = value - floorf(value);
  if (wrapped < 0.0f) wrapped += 1.0f;
  return wrapped;
}

__host__ __device__ inline float wrapPeriodSafe(float value, float period) {
  if (period <= 0.0f) return 0.0f;
  float wrapped = value - period * floorf(value / period);
  if (wrapped < 0.0f) wrapped += period;
  return wrapped;
}

__host__ __device__ inline float signPreservingPowSafe(float value, float exponent) {
  if (value == 0.0f) return 0.0f;
  return copysignf(safePowPos(fabsf(value), exponent, 0.0f), value);
}

__host__ __device__ inline float exp10CompatSafe(float x) {
  return safeExp2Clamped(x * 3.3219280948873626f);
}

__host__ __device__ inline float effectiveNeutralRadiusThresholdSafe(float sliderValue) {
  return clampValue(safePowPos(clampValue(sliderValue, 0.0f, 1.0f), 2.0f, 0.0f), 0.0f, 1.0f);
}

__host__ __device__ inline float decodeTransferChannelFast(float x, int tf) {
  switch (tf) {
    case 0:
      return x;
    case 1: {
      const float a = fabsf(x);
      const float decoded = (a <= 0.04045f) ? safeDiv(a, 12.92f)
                                            : safePowPos(safeDiv(a + 0.055f, 1.055f), 2.4f, 0.0f);
      return copysignf(decoded, x);
    }
    case 2:
      return signPreservingPowSafe(x, 2.4f);
    case 3:
      return x <= 0.02740668f ? safeDiv(x, 10.44426855f) : safeExp2Clamped(safeDiv(x, 0.07329248f) - 7.0f) - 0.0075f;
    case 4:
      return x <= 0.155251141552511f ? safeDiv(x - 0.0729055341958355f, 10.5402377416545f)
                                     : safeExp2Clamped(x * 17.52f - 9.72f);
    case 5:
      return x < 5.367655f * 0.010591f + 0.092809f
                 ? safeDiv(x - 0.092809f, 5.367655f)
                 : safeDiv(exp10CompatSafe(safeDiv(x - 0.385537f, 0.247190f)) - 0.052272f, 5.555556f);
    case 6:
      return x < -0.7774983977293537f
                 ? x * 0.3033266726886969f - 0.7774983977293537f
                 : safeDiv(safeExp2Clamped(14.0f * safeDiv(x - 0.09286412512218964f, 0.9071358748778103f) + 6.0f) - 64.0f, 2231.8263090676883f);
    case 7: {
      constexpr float kCut = 0.092864125f;
      constexpr float kScale = 0.24136077f;
      constexpr float kGain = 87.099375f;
      const float decoded = x < kCut ? -safeDiv(exp10CompatSafe(safeDiv(kCut - x, kScale)) - 1.0f, kGain)
                                     : safeDiv(exp10CompatSafe(safeDiv(x - kCut, kScale)) - 1.0f, kGain);
      return decoded * 0.9f;
    }
    case 8:
      return x < 171.2102946929f / 1023.0f
                 ? safeDiv((x * 1023.0f - 95.0f) * 0.01125f, 171.2102946929f - 95.0f)
                 : (exp10CompatSafe(safeDiv(x * 1023.0f - 420.0f, 261.5f)) * 0.19f - 0.01f);
    case 9:
      if (x < 0.04076162f) {
        return -safeDiv(exp10CompatSafe(safeDiv(0.069886632f - x, 0.42889912f)) - 1.0f, 14.98325f);
      }
      if (x <= 0.105357102f) {
        return safeDiv(x - 0.073059361f, 2.3069815f);
      }
      return safeDiv(exp10CompatSafe(safeDiv(x - 0.073059361f, 0.36726845f)) - 1.0f, 14.98325f);
    case 10:
      return x < 0.0f ? safeDiv(x, 15.1927f) - 0.01f
                      : safeDiv(exp10CompatSafe(safeDiv(x, 0.224282f)) - 1.0f, 155.975327f) - 0.01f;
    default:
      return x;
  }
}

__host__ __device__ inline uint32_t hash32Fast(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__host__ __device__ inline float unitHash01Fast(uint32_t value) {
  return static_cast<float>(hash32Fast(value)) / static_cast<float>(0xffffffffU);
}

__host__ __device__ inline float haltonFast(uint32_t index, uint32_t base) {
  float factor = 1.0f;
  float result = 0.0f;
  while (index > 0u) {
    factor /= static_cast<float>(base);
    result += factor * static_cast<float>(index % base);
    index /= base;
  }
  return result;
}

__host__ __device__ inline float rawRgbHue01Fast(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6f) return 0.0f;
  float h = 0.0f;
  if (cMax == r) {
    h = wrapPeriodSafe(safeDiv(g - b, delta), 6.0f);
  } else if (cMax == g) {
    h = safeDiv(b - r, delta) + 2.0f;
  } else {
    h = safeDiv(r - g, delta) + 4.0f;
  }
  return wrapUnitSafe(safeDiv(h, 6.0f));
}

__host__ __device__ inline void rgbToHsvHexconePlaneFast(float r, float g, float b, float* outX, float* outZ) {
  *outX = r - 0.5f * g - 0.5f * b;
  *outZ = 0.8660254037844386f * (g - b);
}

__host__ __device__ inline void rgbToPlotCircularHslFast(float r, float g, float b, float* outH, float* outRadius, float* outL) {
  const float cMax = fmaxf(r, fmaxf(g, b));
  const float cMin = fminf(r, fminf(g, b));
  const float delta = cMax - cMin;
  const float light = 0.5f * (cMax + cMin);
  float hue = rawRgbHue01Fast(r, g, b, cMax, delta);
  float satDenom = 1.0f - fabsf(2.0f * light - 1.0f);
  if (delta > 1e-6f && satDenom < 0.0f) {
    hue = wrapUnitSafe(hue + 0.5f);
  }
  if (fabsf(satDenom) <= 1e-6f) {
    satDenom = satDenom < 0.0f ? -1e-6f : 1e-6f;
  }
  *outH = hue;
  *outRadius = fabsf(safeDiv(delta, satDenom));
  *outL = light;
}

__host__ __device__ inline void rgbToPlotCircularHsvFast(float r, float g, float b, float* outH, float* outRadius, float* outV) {
  const float cMax = fmaxf(r, fmaxf(g, b));
  const float cMin = fminf(r, fminf(g, b));
  const float delta = cMax - cMin;
  *outH = rawRgbHue01Fast(r, g, b, cMax, delta);
  *outRadius = (delta > 1e-6f && cMax > 1e-6f) ? safeDiv(delta, cMax) : 0.0f;
  *outV = cMax;
}

__host__ __device__ inline void rgbToChenFast(float r, float g, float b, bool allowOverflow, float* outHue, float* outChroma, float* outLight) {
  constexpr float kTau = 6.28318530717958647692f;
  if (!allowOverflow) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  const float rotX = r * 0.81649658f + g * -0.40824829f + b * -0.40824829f;
  const float rotY = g * 0.70710678f + b * -0.70710678f;
  const float rotZ = (r + g + b) * 0.57735027f;
  const float azimuth = safeAtan2(rotY, rotX);
  const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ);
  *outHue = azimuth < 0.0f ? safeDiv(azimuth + kTau, kTau) : safeDiv(azimuth, kTau);
  *outChroma = polar * kChenPolarScale;
  *outLight = radius * 0.5773502691896258f;
}

__host__ __device__ inline void rgbToNormConeCoordsFast(float r, float g, float b, bool normalized, bool allowOverflow,
                                                        float* outHue, float* outChroma, float* outValue) {
  constexpr float kTau = 6.28318530717958647692f;
  const float maxRgb = fmaxf(r, fmaxf(g, b));
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  float hue = safeDiv(safeAtan2(rotY, rotX), kTau);
  if (hue < 0.0f) hue += 1.0f;
  const float chromaRadius = safeHypot2(rotX, rotY);
  const float polar = safeAtan2(chromaRadius, rotZ);
  float chroma = safeDiv(polar, kPolarMax);
  if (normalized) {
    const float angle = hue * kTau - 0.52359877559829887308f;
    const float cosPolar = cosf(polar);
    const float safeCos = fabsf(cosPolar) > 1e-6f ? cosPolar : (cosPolar < 0.0f ? -1e-6f : 1e-6f);
    const float cone = safeDiv(sinf(polar), safeCos) / safeSqrt(2.0f);
    const float sinTerm = clampValue(sinf(3.0f * angle), -1.0f, 1.0f);
    const float chromaGain = safeDiv(1.0f, 2.0f * cosf(safeAcosUnit(sinTerm) / 3.0f), 0.0f, 1e-6f);
    chroma = chromaGain > 1e-6f ? safeDiv(cone, chromaGain) : 0.0f;
    if (allowOverflow && chroma < 0.0f) {
      chroma = -chroma;
      hue += 0.5f;
      if (hue >= 1.0f) hue -= 1.0f;
    }
  }
  *outHue = hue;
  *outChroma = allowOverflow ? fmaxf(chroma, 0.0f) : clamp01Safe(chroma);
  *outValue = allowOverflow ? maxRgb : clamp01Safe(maxRgb);
}

__host__ __device__ inline void rgbToRgbConeFast(float r, float g, float b, float* outMagnitude, float* outHue, float* outPolar) {
  constexpr float kTau = 6.28318530717958647692f;
  r = clamp01Safe(r);
  g = clamp01Safe(g);
  b = clamp01Safe(b);
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  float hue = safeAtan2(rotY, rotX);
  if (hue < 0.0f) hue += kTau;
  const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ);
  *outMagnitude = clamp01Safe(radius * 0.576f);
  *outHue = safeDiv(hue, kTau);
  *outPolar = clamp01Safe(safeDiv(polar, kPolarMax));
}

__host__ __device__ inline void rgbToJpConicalFast(float r, float g, float b, bool allowOverflow,
                                                   float* outMagnitude, float* outHue, float* outPolar) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTau = 6.28318530717958647692f;
  const float kAsinInvSqrt2 = safeAsinUnit(safeDiv(1.0f, safeSqrt(2.0f)));
  const float kAsinInvSqrt3 = safeAsinUnit(safeDiv(1.0f, safeSqrt(3.0f)));
  const float kHueCoef1 = safeDiv(1.0f, 2.0f - safeDiv(kAsinInvSqrt2, kAsinInvSqrt3), 0.0f, 1e-6f);
  if (!allowOverflow) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  const float radius = safeSqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  float hue = safeAtan2(rotY, rotX);
  if (hue < 0.0f) hue += kTau;
  const float polar = safeAtan2(safeHypot2(rotX, rotY), rotZ);
  const float huecoef2 = 2.0f * polar * sinf((2.0f * kPi / 3.0f) - wrapPeriodSafe(hue, kPi / 3.0f)) / safeSqrt(3.0f);
  const float huemag =
      (safeDiv(safeAcosUnit(cosf(3.0f * hue + kPi)), kPi * kHueCoef1, 0.0f, 1e-6f) + (safeDiv(kAsinInvSqrt2, kAsinInvSqrt3, 0.0f, 1e-6f) - 1.0f)) * huecoef2;
  const float satmag = sinf(huemag + kAsinInvSqrt3);
  float magnitude = radius * satmag;
  if (allowOverflow && magnitude < 0.0f) {
    magnitude = -magnitude;
    hue += kPi;
    if (hue >= kTau) hue -= kTau;
  }
  *outMagnitude = allowOverflow ? magnitude : clamp01Safe(magnitude);
  *outHue = safeDiv(hue, kTau);
  *outPolar = allowOverflow ? fmaxf(safeDiv(polar, kPolarMax), 0.0f) : clamp01Safe(safeDiv(polar, kPolarMax));
}

__host__ __device__ inline void rgbToReuleauxFast(float r, float g, float b, bool allowOverflow, float* outHue, float* outSat, float* outValue) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kMaxSat = 1.41421356237f;
  if (!allowOverflow) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  const float rotX = 0.33333333333f * (2.0f * r - g - b) * 0.70710678118f;
  const float rotY = (g - b) * 0.40824829046f;
  const float rotZ = (r + g + b) / 3.0f;
  float hue = kPi - safeAtan2(rotY, -rotX);
  if (hue < 0.0f) hue += kTau;
  if (hue >= kTau) hue = wrapPeriodSafe(hue, kTau);
  float sat = fabsf(rotZ) <= 1e-6f ? 0.0f : safeDiv(safeHypot2(rotX, rotY), rotZ);
  if (allowOverflow && sat < 0.0f) {
    sat = -sat;
    hue += kPi;
    if (hue >= kTau) hue -= kTau;
  }
  *outHue = safeDiv(hue, kTau);
  *outSat = allowOverflow ? safeDiv(sat, kMaxSat) : clampValue(safeDiv(sat, kMaxSat), 0.0f, 1.0f);
  *outValue = allowOverflow ? fmaxf(r, fmaxf(g, b)) : clamp01Safe(fmaxf(r, fmaxf(g, b)));
}

__host__ __device__ inline float normalizedNeutralRadiusForPoint(const Request& request, float r, float g, float b) {
  switch (request.plotMode) {
    case 0: {
      const float rr = clamp01Safe(r);
      const float gg = clamp01Safe(g);
      const float bb = clamp01Safe(b);
      const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
      const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
      return clampValue(safeDiv(safeHypot2(rotX, rotY), kRgbAxisMaxRadius), 0.0f, 1.0f);
    }
    case 1: {
      if (request.circularHsl != 0) {
        float h = 0.0f;
        float radius = 0.0f;
        float l = 0.0f;
        rgbToPlotCircularHslFast(r, g, b, &h, &radius, &l);
        return clampValue(radius, 0.0f, 1.0f);
      }
      const float cMax = fmaxf(r, fmaxf(g, b));
      const float cMin = fminf(r, fminf(g, b));
      return clampValue(cMax - cMin, 0.0f, 1.0f);
    }
    case 2: {
      if (request.circularHsv != 0) {
        float h = 0.0f;
        float radius = 0.0f;
        float v = 0.0f;
        rgbToPlotCircularHsvFast(r, g, b, &h, &radius, &v);
        return clampValue(radius, 0.0f, 1.0f);
      }
      float x = 0.0f;
      float z = 0.0f;
      rgbToHsvHexconePlaneFast(r, g, b, &x, &z);
      return clampValue(safeHypot2(x, z), 0.0f, 1.0f);
    }
    case 3: {
      float h = 0.0f;
      float chroma = 0.0f;
      float light = 0.0f;
      rgbToChenFast(r, g, b, request.showOverflow != 0, &h, &chroma, &light);
      const float polar = chroma / kChenPolarScale;
      const float radius = light * sinf(polar) / kRgbAxisMaxRadius;
      return clampValue(radius, 0.0f, 1.0f);
    }
    case 4: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToRgbConeFast(r, g, b, &magnitude, &hue, &polar);
      const float radial = magnitude * sinf(polar * kPolarMax);
      return clampValue(radial / sinf(kPolarMax), 0.0f, 1.0f);
    }
    case 5: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToJpConicalFast(r, g, b, request.showOverflow != 0, &magnitude, &hue, &polar);
      const float radial = magnitude * sinf(polar * kPolarMax);
      return clampValue(radial / sinf(kPolarMax), 0.0f, 1.0f);
    }
    case 6: {
      float hue = 0.0f;
      float chroma = 0.0f;
      float value = 0.0f;
      rgbToNormConeCoordsFast(r, g, b, request.normConeNormalized != 0, request.showOverflow != 0, &hue, &chroma, &value);
      return clampValue(chroma, 0.0f, 1.0f);
    }
    case 7: {
      float hue = 0.0f;
      float sat = 0.0f;
      float value = 0.0f;
      rgbToReuleauxFast(r, g, b, request.showOverflow != 0, &hue, &sat, &value);
      return clampValue(sat, 0.0f, 1.0f);
    }
    default:
      return 0.0f;
  }
}

__host__ __device__ inline float neutralRadiusAcceptanceProbability(const Request& request, float normalizedRadius) {
  if (request.neutralRadiusEnabled == 0) return 1.0f;
  const float threshold = effectiveNeutralRadiusThresholdSafe(request.neutralRadius);
  if (threshold <= 1e-6f) return normalizedRadius <= threshold + 1e-6f ? 1.0f : 0.0f;
  if (normalizedRadius > threshold + 1e-6f) return 0.0f;
  const float clippedFraction = clampValue(1.0f - threshold, 0.0f, 1.0f);
  const float normalizedInside = clampValue(safeDiv(normalizedRadius, threshold), 0.0f, 1.0f);
  const float edgePenalty = 0.78f * clippedFraction;
  return clampValue(1.0f - edgePenalty * safePowPos(normalizedInside, 1.35f, 0.0f), 0.0f, 1.0f);
}

__device__ inline void sampleUvForAttempt(const Request& request, int attemptIndex, float* outU, float* outV) {
  float u = 0.0f;
  float v = 0.0f;
  switch (request.samplingMode) {
    case 1: {
      const int grid = max(1, static_cast<int>(ceilf(safeSqrt(static_cast<float>(request.pointCount)))));
      const int gx = attemptIndex % grid;
      const int gy = attemptIndex / grid;
      u = safeDiv(static_cast<float>(gx) + unitHash01Fast(static_cast<uint32_t>(attemptIndex * 2 + 1)), static_cast<float>(grid));
      v = safeDiv(static_cast<float>(gy) + unitHash01Fast(static_cast<uint32_t>(attemptIndex * 2 + 2)), static_cast<float>(grid));
      break;
    }
    case 2:
      u = unitHash01Fast(static_cast<uint32_t>(attemptIndex * 2 + 11));
      v = unitHash01Fast(static_cast<uint32_t>(attemptIndex * 2 + 37));
      break;
    default: {
      const int grid = max(1, static_cast<int>(ceilf(safeSqrt(static_cast<float>(request.pointCount)))));
      const int gx = attemptIndex % grid;
      const int gy = attemptIndex / grid;
      u = safeDiv(static_cast<float>(gx) + 0.5f, static_cast<float>(grid));
      v = safeDiv(static_cast<float>(gy) + 0.5f, static_cast<float>(grid));
      break;
    }
  }
  *outU = clampValue(u, 0.0f, 1.0f);
  *outV = clampValue(v, 0.0f, 1.0f);
}

__device__ inline int occupancyBinIndex(const Request& request, float r, float g, float b) {
  const int binsPerAxis = request.preserveOverflow != 0 ? 18 : 16;
  auto toBin = [&](float value) -> int {
    if (request.preserveOverflow == 0) {
        return clampValue(static_cast<int>(floorf(clamp01Safe(value) * 16.0f)), 0, 15);
    }
    if (value < 0.0f) return 0;
    if (value > 1.0f) return 17;
    return 1 + clampValue(static_cast<int>(floorf(value * 16.0f)), 0, 15);
  };
  return (toBin(r) * binsPerAxis + toBin(g)) * binsPerAxis + toBin(b);
}

__global__ void primaryPassKernel(Request request, Sample* primaryOut, int* primaryCount, int* occupancyBins) {
  const int attemptIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (attemptIndex >= request.maxPrimaryAttempts) return;
  float u = 0.0f;
  float v = 0.0f;
  sampleUvForAttempt(request, attemptIndex, &u, &v);
  const int sx = clampValue(static_cast<int>(u * static_cast<float>(request.scaledWidth - 1)), 0, request.scaledWidth - 1);
  const int sy = clampValue(static_cast<int>(v * static_cast<float>(request.scaledHeight - 1)), 0, request.scaledHeight - 1);
  const int x = clampValue(static_cast<int>(((static_cast<float>(sx) + 0.5f) / static_cast<float>(request.scaledWidth)) * static_cast<float>(request.width)),
                           0, request.width - 1);
  const int y = clampValue(static_cast<int>(((static_cast<float>(sy) + 0.5f) / static_cast<float>(request.scaledHeight)) * static_cast<float>(request.height)),
                           0, request.height - 1);

  const char* rowBase = reinterpret_cast<const char*>(request.srcBase) + static_cast<std::size_t>(y) * request.srcRowBytes;
  const float* pix = reinterpret_cast<const float*>(rowBase) + static_cast<std::size_t>(x) * 4u;
  float r = pix[0];
  float g = pix[1];
  float b = pix[2];
  if (request.preserveOverflow == 0) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  if (request.plotDisplayLinearEnabled != 0) {
    r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer);
    g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer);
    b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer);
  }

  if (request.neutralRadiusEnabled != 0) {
    const float normalizedRadius = normalizedNeutralRadiusForPoint(request, r, g, b);
    const float acceptProbability = neutralRadiusAcceptanceProbability(request, normalizedRadius);
    if (acceptProbability <= 0.0f) return;
    if (acceptProbability < 0.999999f) {
      const uint32_t samplingSeed = static_cast<uint32_t>(attemptIndex * 2654435761u) ^
                                    static_cast<uint32_t>(x * 911u + y * 3571u);
      if (unitHash01Fast(samplingSeed) > acceptProbability) return;
    }
  }

  const int outIndex = atomicAdd(primaryCount, 1);
  if (outIndex >= request.pointCount) return;
  Sample sample{};
  sample.xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(request.width);
  sample.yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(request.height);
  sample.r = r;
  sample.g = g;
  sample.b = b;
  primaryOut[outIndex] = sample;
  atomicAdd(&occupancyBins[occupancyBinIndex(request, r, g, b)], 1);
}

__global__ void occupancyPassKernel(Request request, OccupancyCandidate* candidateOut, int* candidateCount) {
  const int attemptIndex = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (attemptIndex >= request.maxCandidateAttempts) return;
  const float u = haltonFast(static_cast<uint32_t>(attemptIndex + 1), 2u);
  const float v = haltonFast(static_cast<uint32_t>(attemptIndex + 1), 3u);
  const int sx = clampValue(static_cast<int>(u * static_cast<float>(request.scaledWidth - 1)), 0, request.scaledWidth - 1);
  const int sy = clampValue(static_cast<int>(v * static_cast<float>(request.scaledHeight - 1)), 0, request.scaledHeight - 1);
  const int x = clampValue(static_cast<int>(((static_cast<float>(sx) + 0.5f) / static_cast<float>(request.scaledWidth)) * static_cast<float>(request.width)),
                           0, request.width - 1);
  const int y = clampValue(static_cast<int>(((static_cast<float>(sy) + 0.5f) / static_cast<float>(request.scaledHeight)) * static_cast<float>(request.height)),
                           0, request.height - 1);

  const char* rowBase = reinterpret_cast<const char*>(request.srcBase) + static_cast<std::size_t>(y) * request.srcRowBytes;
  const float* pix = reinterpret_cast<const float*>(rowBase) + static_cast<std::size_t>(x) * 4u;
  float r = pix[0];
  float g = pix[1];
  float b = pix[2];
  if (request.preserveOverflow == 0) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  if (request.plotDisplayLinearEnabled != 0) {
    r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer);
    g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer);
    b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer);
  }

  float normalizedRadius = 0.0f;
  if (request.neutralRadiusEnabled != 0) {
    normalizedRadius = normalizedNeutralRadiusForPoint(request, r, g, b);
    const float acceptProbability = neutralRadiusAcceptanceProbability(request, normalizedRadius);
    if (acceptProbability <= 0.0f) return;
    if (acceptProbability < 0.999999f) {
      const uint32_t samplingSeed = static_cast<uint32_t>((attemptIndex + 1) * 2246822519u) ^
                                    static_cast<uint32_t>(x * 977u + y * 4051u);
      if (unitHash01Fast(samplingSeed) > acceptProbability) return;
    }
  }

  const int outIndex = atomicAdd(candidateCount, 1);
  if (outIndex >= request.candidateTarget) return;
  OccupancyCandidate candidate{};
  candidate.sample.xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(request.width);
  candidate.sample.yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(request.height);
  candidate.sample.r = r;
  candidate.sample.g = g;
  candidate.sample.b = b;
  candidate.normalizedNeutralRadius = normalizedRadius;
  candidate.bin = occupancyBinIndex(request, r, g, b);
  candidate.tie = static_cast<uint32_t>(attemptIndex);
  candidateOut[outIndex] = candidate;
}

__global__ void stripCubeKernel(StripRequest request, Sample* outSamples) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int resolution = request.resolution;
  const int total = resolution * resolution * resolution;
  if (index >= total) return;
  const int denom = max(1, resolution - 1);
  const int bz = index / (resolution * resolution);
  const int rem = index % (resolution * resolution);
  const int gy = rem / resolution;
  const int rx = rem % resolution;
  const float layerStart = static_cast<float>(bz) * request.cellWidth;
  const float localX = layerStart +
                       (request.cellWidth <= 1.0f
                            ? 0.0f
                            : static_cast<float>(rx) / static_cast<float>(denom) * max(0.0f, request.cellWidth - 1.0f));
  const int x = clampValue(static_cast<int>(lrintf(localX)), 0, request.width - 1);
  const int y = request.cubeY1 + clampValue(static_cast<int>(lrintf(
                   (static_cast<float>(gy) / static_cast<float>(denom)) * static_cast<float>(max(0, request.stripHeight - 1)))),
                   0, max(0, request.stripHeight - 1));
  const char* rowBase = reinterpret_cast<const char*>(request.srcBase) + static_cast<std::size_t>(y) * request.srcRowBytes;
  const float* pix = reinterpret_cast<const float*>(rowBase) + static_cast<std::size_t>(x) * 4u;
  float r = pix[0];
  float g = pix[1];
  float b = pix[2];
  if (request.preserveOverflow == 0) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  if (request.plotDisplayLinearEnabled != 0) {
    r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer);
    g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer);
    b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer);
  }
  Sample sample{};
  sample.xNorm = r;
  sample.yNorm = g;
  sample.zReserved = b;
  sample.r = r;
  sample.g = g;
  sample.b = b;
  outSamples[index] = sample;
}

__global__ void stripRampKernel(StripRequest request, Sample* outSamples) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int sampleCols = request.width;
  const int sampleRows = request.rampSampleRows;
  const int total = sampleCols * sampleRows;
  if (index >= total) return;
  const int rowIndex = index / sampleCols;
  const int colIndex = index % sampleCols;
  const int rowDenom = max(1, sampleRows - 1);
  const int y = request.rampY1 + clampValue(static_cast<int>(lrintf(
                   (static_cast<float>(rowIndex) / static_cast<float>(rowDenom)) * static_cast<float>(max(0, request.rampHeight - 1)))),
                   0, max(0, request.rampHeight - 1));
  const int x = clampValue(colIndex, 0, request.width - 1);
  const char* rowBase = reinterpret_cast<const char*>(request.srcBase) + static_cast<std::size_t>(y) * request.srcRowBytes;
  const float* pix = reinterpret_cast<const float*>(rowBase) + static_cast<std::size_t>(x) * 4u;
  float r = pix[0];
  float g = pix[1];
  float b = pix[2];
  if (request.preserveOverflow == 0) {
    r = clamp01Safe(r);
    g = clamp01Safe(g);
    b = clamp01Safe(b);
  }
  if (request.plotDisplayLinearEnabled != 0) {
    r = decodeTransferChannelFast(r, request.plotDisplayLinearTransfer);
    g = decodeTransferChannelFast(g, request.plotDisplayLinearTransfer);
    b = decodeTransferChannelFast(b, request.plotDisplayLinearTransfer);
  }
  Sample sample{};
  sample.xNorm = r;
  sample.yNorm = g;
  sample.zReserved = b;
  sample.r = r;
  sample.g = g;
  sample.b = b;
  outSamples[index] = sample;
}

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr) cudaFree(ptr);
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
bool allocBuffer(DeviceBuffer<T>* buffer, std::size_t count, std::string* error) {
  if (!buffer) return false;
  if (count == 0) {
    buffer->ptr = nullptr;
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&buffer->ptr), count * sizeof(T)) != cudaSuccess) {
    if (error) *error = "cudaMalloc failed";
    buffer->ptr = nullptr;
    return false;
  }
  return true;
}

}  // namespace

bool buildWholeImageCloud(const Request& request, Result* out) {
  if (!out) return false;
  *out = Result{};
  if (!request.srcBase || request.width <= 0 || request.height <= 0 || request.pointCount <= 0) {
    out->error = "invalid-request";
    return false;
  }

  const int binsPerAxis = request.preserveOverflow != 0 ? 18 : 16;
  const std::size_t occupancyCount = static_cast<std::size_t>(binsPerAxis * binsPerAxis * binsPerAxis);

  DeviceBuffer<Sample> primary;
  DeviceBuffer<int> primaryCount;
  DeviceBuffer<int> occupancy;
  if (!allocBuffer(&primary, static_cast<std::size_t>(request.pointCount), &out->error) ||
      !allocBuffer(&primaryCount, 1u, &out->error) ||
      !allocBuffer(&occupancy, occupancyCount, &out->error)) {
    return false;
  }

  if (cudaMemsetAsync(primaryCount.ptr, 0, sizeof(int), request.stream) != cudaSuccess ||
      cudaMemsetAsync(occupancy.ptr, 0, occupancyCount * sizeof(int), request.stream) != cudaSuccess) {
    out->error = "cudaMemsetAsync failed";
    return false;
  }

  const int primaryBlocks = std::max(1, (request.maxPrimaryAttempts + 255) / 256);
  primaryPassKernel<<<primaryBlocks, 256, 0, request.stream>>>(request, primary.ptr, primaryCount.ptr, occupancy.ptr);
  if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(request.stream) != cudaSuccess) {
    out->error = "primary-pass failed";
    return false;
  }

  int primaryAccepted = 0;
  if (cudaMemcpy(&primaryAccepted, primaryCount.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    out->error = "primary-count readback failed";
    return false;
  }
  primaryAccepted = std::min(primaryAccepted, request.pointCount);
  out->primaryAttempts = request.maxPrimaryAttempts;
  out->primaryAccepted = primaryAccepted;
  out->extraPointCount = request.extraPointCount;
  out->occupancy.assign(occupancyCount, 0);
  if (!out->occupancy.empty() &&
      cudaMemcpy(out->occupancy.data(), occupancy.ptr, occupancyCount * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
    out->error = "occupancy readback failed";
    return false;
  }
  out->primarySamples.assign(static_cast<std::size_t>(primaryAccepted), Sample{});
  if (primaryAccepted > 0 &&
      cudaMemcpy(out->primarySamples.data(), primary.ptr, static_cast<std::size_t>(primaryAccepted) * sizeof(Sample), cudaMemcpyDeviceToHost) != cudaSuccess) {
    out->error = "primary-sample readback failed";
    return false;
  }

  if (request.occupancyFill != 0 && request.candidateTarget > 0 && request.maxCandidateAttempts > 0) {
    DeviceBuffer<OccupancyCandidate> candidates;
    DeviceBuffer<int> candidateCount;
    if (!allocBuffer(&candidates, static_cast<std::size_t>(request.candidateTarget), &out->error) ||
        !allocBuffer(&candidateCount, 1u, &out->error)) {
      return false;
    }
    if (cudaMemsetAsync(candidateCount.ptr, 0, sizeof(int), request.stream) != cudaSuccess) {
      out->error = "candidate-reset failed";
      return false;
    }
    const int candidateBlocks = std::max(1, (request.maxCandidateAttempts + 255) / 256);
    occupancyPassKernel<<<candidateBlocks, 256, 0, request.stream>>>(request, candidates.ptr, candidateCount.ptr);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(request.stream) != cudaSuccess) {
      out->error = "occupancy-pass failed";
      return false;
    }
    int candidateAccepted = 0;
    if (cudaMemcpy(&candidateAccepted, candidateCount.ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
      out->error = "candidate-count readback failed";
      return false;
    }
    candidateAccepted = std::min(candidateAccepted, request.candidateTarget);
    out->occupancyCandidates.assign(static_cast<std::size_t>(candidateAccepted), OccupancyCandidate{});
    if (candidateAccepted > 0 &&
        cudaMemcpy(out->occupancyCandidates.data(), candidates.ptr,
                   static_cast<std::size_t>(candidateAccepted) * sizeof(OccupancyCandidate),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      out->error = "candidate readback failed";
      return false;
    }
  }

  out->success = true;
  return true;
}

bool buildIdentityStripCloud(const StripRequest& request, StripResult* out) {
  if (!out) return false;
  *out = StripResult{};
  if (!request.srcBase || request.srcRowBytes == 0 || request.width <= 0 || request.height <= 0 || request.resolution <= 0) {
    out->error = "invalid-strip-request";
    return false;
  }

  const int cubeCount = request.readCube != 0 ? request.resolution * request.resolution * request.resolution : 0;
  const int rampCount = request.readRamp != 0 ? request.width * max(0, request.rampSampleRows) : 0;
  const int totalCount = cubeCount + rampCount;
  if (totalCount <= 0) {
    out->error = "empty-strip-request";
    return false;
  }

  DeviceBuffer<Sample> samples;
  if (!allocBuffer(&samples, static_cast<std::size_t>(totalCount), &out->error)) {
    return false;
  }

  int offset = 0;
  if (cubeCount > 0) {
    const int blocks = max(1, (cubeCount + 255) / 256);
    stripCubeKernel<<<blocks, 256, 0, request.stream>>>(request, samples.ptr + offset);
    if (cudaGetLastError() != cudaSuccess) {
      out->error = "strip-cube-kernel failed";
      return false;
    }
    offset += cubeCount;
  }
  if (rampCount > 0) {
    const int blocks = max(1, (rampCount + 255) / 256);
    stripRampKernel<<<blocks, 256, 0, request.stream>>>(request, samples.ptr + offset);
    if (cudaGetLastError() != cudaSuccess) {
      out->error = "strip-ramp-kernel failed";
      return false;
    }
  }
  if (cudaStreamSynchronize(request.stream) != cudaSuccess) {
    out->error = "strip-kernel sync failed";
    return false;
  }

  out->samples.assign(static_cast<std::size_t>(totalCount), Sample{});
  if (cudaMemcpy(out->samples.data(),
                 samples.ptr,
                 static_cast<std::size_t>(totalCount) * sizeof(Sample),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    out->error = "strip readback failed";
    return false;
  }
  out->success = true;
  return true;
}

}  // namespace ChromaspaceCloudCuda
