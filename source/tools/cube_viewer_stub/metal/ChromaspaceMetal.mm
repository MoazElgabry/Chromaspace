#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <sstream>

#include "ChromaspaceMetal.h"

namespace ChromaspaceMetal {
namespace {

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> overlayPipeline = nil;
  id<MTLComputePipelineState> inputPipeline = nil;
  id<MTLComputePipelineState> rasterSourcePipeline = nil;
  id<MTLComputePipelineState> rasterOccupancyCountPipeline = nil;
  id<MTLComputePipelineState> inputSamplePipeline = nil;
  id<MTLComputePipelineState> scopeDensityPipeline = nil;
  id<MTLComputePipelineState> glossFieldAccumulatePipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldMaxPipeline = nil;
  id<MTLComputePipelineState> glossFieldNormalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldBlurPipeline = nil;
  id<MTLComputePipelineState> glossFieldBodyPipeline = nil;
  id<MTLComputePipelineState> glossFieldRawSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldWeightedSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalNormalizePipeline = nil;
  std::string deviceName;
  std::string initError;
  bool initAttempted = false;
  bool ready = false;
};

struct OverlayUniforms {
  int cubeSize;
  int ramp;
  int useInputPoints;
  int pointCount;
  float colorSaturation;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
};

struct InputUniforms {
  int pointCount;
  int inputStride;
  int glossView;
  float sourceAspect;
  float glossLiftScale;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct RasterSourceUniforms {
  InputUniforms input;
  int basePointCount;
  int sourceWidth;
  int sourceHeight;
  int sampleStride;
  int sampleCountX;
  int pixelFormat;
  int plotLinear;
  int plotLinearTransfer;
  int excludeIdentityData;
  int isolateIdentityData;
  int readIdentityPlot;
  int readGrayRamp;
  int identityCubeY1;
  int identityCubeY2;
  int identityRampY1;
  int identityRampY2;
  int identityCubeAppendOffset;
  int identityCubeAppendCount;
  int identityCubeAppendY1;
  int identityCubeAppendY2;
  int identityCubeAppendRowStep;
  int identityCubeAppendXStep;
  int identityRampAppendOffset;
  int identityRampAppendCount;
  int identityRampAppendY1;
  int identityRampAppendY2;
  int identityRampAppendRowStep;
  int identityRampAppendXStep;
  int occupancyFill;
  int occupancyAppendOffset;
  int occupancyAppendCount;
  int occupancyCandidateCount;
  int occupancyTargetThreshold;
  int lassoEnabled;
  int lassoStrokeCount;
  int lassoPointCount;
  int lassoStrokeFirst[16];
  int lassoStrokeCountPerStroke[16];
  int lassoStrokeSubtract[16];
  float lassoX[256];
  float lassoY[256];
  int cubeSlicingEnabled;
  int neutralRadiusEnabled;
  float neutralRadius;
  int cubeSliceRed;
  int cubeSliceYellow;
  int cubeSliceGreen;
  int cubeSliceCyan;
  int cubeSliceBlue;
  int cubeSliceMagenta;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct ScopeDensityUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int excludeOverflow;
};

struct GlossFieldAccumulateUniforms {
  int pointCount;
  int gridWidth;
  int gridHeight;
  int showOverflow;
};

struct GlossFieldCellUniforms {
  int cellCount;
  int gridWidth;
  int gridHeight;
  int neighborhoodChoice;
};

struct PackedFloat3 {
  float x;
  float y;
  float z;
};

MetalContext& context() {
  static MetalContext ctx;
  return ctx;
}

const char* kMetalSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float kTau = 6.28318530717958647692;
constant float kPi = 3.14159265358979323846;

struct OverlayUniforms {
  int cubeSize;
  int ramp;
  int useInputPoints;
  int pointCount;
  float colorSaturation;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
};

struct InputUniforms {
  int pointCount;
  int inputStride;
  int glossView;
  float sourceAspect;
  float glossLiftScale;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct RasterSourceUniforms {
  InputUniforms input;
  int basePointCount;
  int sourceWidth;
  int sourceHeight;
  int sampleStride;
  int sampleCountX;
  int pixelFormat;
  int plotLinear;
  int plotLinearTransfer;
  int excludeIdentityData;
  int isolateIdentityData;
  int readIdentityPlot;
  int readGrayRamp;
  int identityCubeY1;
  int identityCubeY2;
  int identityRampY1;
  int identityRampY2;
  int identityCubeAppendOffset;
  int identityCubeAppendCount;
  int identityCubeAppendY1;
  int identityCubeAppendY2;
  int identityCubeAppendRowStep;
  int identityCubeAppendXStep;
  int identityRampAppendOffset;
  int identityRampAppendCount;
  int identityRampAppendY1;
  int identityRampAppendY2;
  int identityRampAppendRowStep;
  int identityRampAppendXStep;
  int occupancyFill;
  int occupancyAppendOffset;
  int occupancyAppendCount;
  int occupancyCandidateCount;
  int occupancyTargetThreshold;
  int lassoEnabled;
  int lassoStrokeCount;
  int lassoPointCount;
  int lassoStrokeFirst[16];
  int lassoStrokeCountPerStroke[16];
  int lassoStrokeSubtract[16];
  float lassoX[256];
  float lassoY[256];
  int cubeSlicingEnabled;
  int neutralRadiusEnabled;
  float neutralRadius;
  int cubeSliceRed;
  int cubeSliceYellow;
  int cubeSliceGreen;
  int cubeSliceCyan;
  int cubeSliceBlue;
  int cubeSliceMagenta;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct ScopeDensityUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int excludeOverflow;
};

struct GlossFieldAccumulateUniforms {
  int pointCount;
  int gridWidth;
  int gridHeight;
  int showOverflow;
};

struct GlossFieldCellUniforms {
  int cellCount;
  int gridWidth;
  int gridHeight;
  int neighborhoodChoice;
};

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

constant float kGlossFieldAccumScale = 1024.0;
constant float kGlossFieldAccumInvScale = 1.0 / 1024.0;

uint glossEncodeAccum(float v) {
  return uint(clamp(v, 0.0, 2.0) * kGlossFieldAccumScale + 0.5);
}

float glossDecodeAccum(uint v) {
  return float(v) * kGlossFieldAccumInvScale;
}

float glossCommonComponent(float r, float g, float b) {
  return max(0.0, min(r, min(g, b)));
}

float glossNeutrality(float r, float g, float b) {
  float common = glossCommonComponent(r, g, b);
  float maxRgb = max(r, max(g, b));
  return maxRgb > 1e-6 ? clamp(common / maxRgb, 0.0, 1.0) : 0.0;
}

float glossStrengthCue(float r, float g, float b) {
  float common = glossCommonComponent(r, g, b);
  float neutrality = glossNeutrality(r, g, b);
  return clamp(common * (0.75 + 0.85 * neutrality), 0.0, 1.0);
}

float glossPresenceWeight(float glossCue) {
  float t = clamp((glossCue - 0.06) / 0.22, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

float wrapHue01(float h) {
  h = fmod(h, 1.0);
  if (h < 0.0) h += 1.0;
  return h;
}

float luminanceAwareAlpha(float baseAlpha, float cr, float cg, float cb, float denseAlphaBias, bool overflowPoint,
                          float pointAlphaScale) {
  float alpha = baseAlpha * pointAlphaScale;
  if (overflowPoint || denseAlphaBias <= 0.0) {
    return clamp(alpha, 0.0, 1.0);
  }
  float luma = clamp(cr * 0.2126 + cg * 0.7152 + cb * 0.0722, 0.0, 1.0);
  float maxRgb = clamp(max(cr, max(cg, cb)), 0.0, 1.0);
  float value = mix(maxRgb, luma, 0.28);
  float highlightKnee = clamp((value - 0.70) / 0.24, 0.0, 1.0);
  float shadowMidProtect = 1.0 - clamp((value - 0.58) / 0.30, 0.0, 1.0);
  float multiplier = clamp(1.0 + 0.22 * denseAlphaBias * shadowMidProtect
                               - 0.12 * denseAlphaBias * highlightKnee,
                           0.94, 1.18);
  return clamp(alpha * multiplier, 0.0, 1.0);
}

float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6) return 0.0;
  float h = 0.0;
  if (cMax == r) {
    h = fmod((g - b) / delta, 6.0);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0;
  } else {
    h = ((r - g) / delta) + 4.0;
  }
  return wrapHue01(h / 6.0);
}

float safeDiv(float num, float den) {
  return fabs(den) < 1e-6 ? 0.0 : num / den;
}

float safeExp2Clamped(float value) {
  return exp2(clamp(value, -126.0, 126.0));
}

float safePowPos(float value, float exponent) {
  return value <= 0.0 ? 0.0 : pow(value, exponent);
}

float signPreservingPow(float value, float exponent) {
  return value == 0.0 ? 0.0 : copysign(safePowPos(fabs(value), exponent), value);
}

float exp10Compat(float value) {
  return safeExp2Clamped(value * 3.3219280948873626);
}

float decodeTransferChannel(float x, int tf) {
  switch (tf) {
    case 0: return x;
    case 1: {
      float a = fabs(x);
      float decoded = (a <= 0.04045) ? safeDiv(a, 12.92) : safePowPos(safeDiv(a + 0.055, 1.055), 2.4);
      return copysign(decoded, x);
    }
    case 2: return signPreservingPow(x, 2.4);
    case 3: return x <= 0.02740668 ? safeDiv(x, 10.44426855) : safeExp2Clamped(safeDiv(x, 0.07329248) - 7.0) - 0.0075;
    case 4: return x <= 0.155251141552511 ? safeDiv(x - 0.0729055341958355, 10.5402377416545) : safeExp2Clamped(x * 17.52 - 9.72);
    case 5: return x < 5.367655 * 0.010591 + 0.092809 ? safeDiv(x - 0.092809, 5.367655) : safeDiv(exp10Compat(safeDiv(x - 0.385537, 0.247190)) - 0.052272, 5.555556);
    case 6: return x < -0.7774983977293537 ? x * 0.3033266726886969 - 0.7774983977293537 : safeDiv(safeExp2Clamped(14.0 * safeDiv(x - 0.09286412512218964, 0.9071358748778103) + 6.0) - 64.0, 2231.8263090676883);
    case 7: {
      constexpr float kCut = 0.092864125;
      constexpr float kScale = 0.24136077;
      constexpr float kGain = 87.099375;
      float decoded = x < kCut ? -safeDiv(exp10Compat(safeDiv(kCut - x, kScale)) - 1.0, kGain) : safeDiv(exp10Compat(safeDiv(x - kCut, kScale)) - 1.0, kGain);
      return decoded * 0.9;
    }
    case 8: return x < 171.2102946929 / 1023.0 ? safeDiv((x * 1023.0 - 95.0) * 0.01125, 171.2102946929 - 95.0) : (exp10Compat(safeDiv(x * 1023.0 - 420.0, 261.5)) * 0.19 - 0.01);
    case 9:
      if (x < 0.04076162) return -safeDiv(exp10Compat(safeDiv(0.069886632 - x, 0.42889912)) - 1.0, 14.98325);
      if (x <= 0.105357102) return safeDiv(x - 0.073059361, 2.3069815);
      return safeDiv(exp10Compat(safeDiv(x - 0.073059361, 0.36726845)) - 1.0, 14.98325);
    case 10: return x < 0.0 ? safeDiv(x, 15.1927) - 0.01 : safeDiv(exp10Compat(safeDiv(x, 0.224282)) - 1.0, 155.975327) - 0.01;
    case 11: {
      constexpr float kA = 8.283605932402494;
      constexpr float kB = 0.09246575342465753;
      constexpr float kC = 0.5300133392291939;
      constexpr float kD = 0.08692876065491224;
      constexpr float kE = 0.005494072432257808;
      constexpr float kCut = kA * 0.005 + kB;
      return x < kCut ? safeDiv(x - kB, kA) : exp(safeDiv(x - kC, kD)) - kE;
    }
    case 12: return x <= 0.14 ? safeDiv(x - 0.0929, 6.025) : safeDiv(exp10Compat(3.89616 * x - 2.27752) - 0.0108, 0.9892);
    case 13: {
      constexpr float kA = 0.555556;
      constexpr float kB = 0.009468;
      constexpr float kC = 0.344676;
      constexpr float kD = 0.790453;
      constexpr float kE = 8.735631;
      constexpr float kF = 0.092864;
      constexpr float kCut = 0.100537775223865;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 14: {
      constexpr float kA = 5.555556;
      constexpr float kB = 0.064829;
      constexpr float kC = 0.245281;
      constexpr float kD = 0.384316;
      constexpr float kE = 8.799461;
      constexpr float kF = 0.092864;
      constexpr float kCut = 0.100686685370811;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 15: return x < 0.181 ? safeDiv(x - 0.125, 5.6) : exp10Compat(safeDiv(x - 0.598206, 0.241514)) - 0.00873;
    case 16: return signPreservingPow(x, 2.2);
    case 17: return signPreservingPow(x, 2.6);
    default: return x;
  }
}

float3 mulRows(constant float* m, float3 v) {
  return float3(dot(float3(m[0], m[1], m[2]), v),
                dot(float3(m[3], m[4], m[5]), v),
                dot(float3(m[6], m[7], m[8]), v));
}

float3 xyToXyz(float2 xy, float Y) {
  if (fabs(xy.y) <= 1e-8) return float3(xy.x, Y, 1.0 - xy.x);
  return float3(xy.x * Y / xy.y, Y, (1.0 - xy.x - xy.y) * Y / xy.y);
}

float3 xyzToXyY(float3 xyz, float2 fallbackWhite) {
  if (fabs(xyz.y) <= 1e-8) return float3(fallbackWhite.x, fallbackWhite.y, 0.0);
  float sum = xyz.x + xyz.y + xyz.z;
  if (fabs(sum) <= 1e-8) return float3(fallbackWhite.x, fallbackWhite.y, xyz.y);
  return float3(xyz.x / sum, xyz.y / sum, xyz.y);
}

float3 mapChromaticityPosition(float r, float g, float b, constant InputUniforms& u) {
  float3 linear = float3(decodeTransferChannel(r, u.chromaticityInputTransfer),
                         decodeTransferChannel(g, u.chromaticityInputTransfer),
                         decodeTransferChannel(b, u.chromaticityInputTransfer));
  if (u.showOverflow == 0) linear = clamp(linear, 0.0, 1.0);
  float3 xyz = mulRows(u.chromaticityRgbToXyz, linear);
  float2 white = float2(u.chromaticityWhiteX, u.chromaticityWhiteY);
  float3 xyY = xyzToXyY(xyz, white);
  float2 xy = xyY.xy;
  if (u.chromaticityReferenceBasis != 0) {
    float3 basisXyz = xyToXyz(xy, 1.0);
    float3 rgb = mulRows(u.chromaticityXyzToRgb, basisXyz);
    xy = xyzToXyY(rgb, float2(1.0 / 3.0, 1.0 / 3.0)).xy;
  }
  float viewerHeight = ((u.showOverflow != 0) ? xyY.z : clamp(xyY.z, 0.0, 1.0)) * 2.0 - 1.0;
  return float3((xy.x - (1.0 / 3.0)) * 3.0,
                (xy.y - (1.0 / 3.0)) * 3.0,
                viewerHeight);
}

float2 rgbToHsvHexconePlane(float r, float g, float b) {
  return float2(r - 0.5 * g - 0.5 * b, 0.8660254037844386 * (g - b));
}

float3 mapPlotPosition(float r, float g, float b, int plotMode, int circularHsl, int circularHsv, int normConeNormalized, int showOverflow) {
  if (plotMode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    float delta = cMax - cMin;
    float l = 0.5 * (cMax + cMin);
    float h = rawRgbHue01(r, g, b, cMax, delta);
    float satDenom = 1.0 - abs(2.0 * l - 1.0);
    if (delta > 1e-6 && satDenom < 0.0) {
      h = wrapHue01(h + 0.5);
    }
    float angle = h * kTau;
    float radius = delta;
    if (circularHsl != 0) {
      float denom = satDenom;
      if (abs(denom) <= 1e-6) {
        denom = (denom < 0.0) ? -1e-6 : 1e-6;
      }
      radius = abs(delta / denom);
    }
    return float3(cos(angle) * radius, l * 2.0 - 1.0, sin(angle) * radius);
  }
  if (plotMode == 2) {
    float cMax = max(r, max(g, b));
    if (circularHsv != 0) {
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      float h = rawRgbHue01(r, g, b, cMax, delta);
      float sat = (delta > 1e-6 && cMax > 1e-6) ? (delta / cMax) : 0.0;
      float angle = h * kTau;
      return float3(cos(angle) * sat, cMax * 2.0 - 1.0, sin(angle) * sat);
    }
    float2 plane = rgbToHsvHexconePlane(r, g, b);
    return float3(plane.x, cMax * 2.0 - 1.0, plane.y);
  }
  if (plotMode == 3) {
    float rotX = r * 0.81649658 + g * -0.40824829 + b * -0.40824829;
    float rotY = g * 0.70710678 + b * -0.70710678;
    float rotZ = r * 0.57735027 + g * 0.57735027 + b * 0.57735027;
    float azimuth = atan2(rotY, rotX);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float wrappedHue = azimuth < 0.0 ? azimuth + kTau : azimuth;
    float polar = atan2(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float c = polar * 1.0467733744265997;
    float l = radius3 * 0.5773502691896258;
    float polarScaled = c * 0.9553166181245093;
    float radial = l * sin(polarScaled) / 0.816496580927726;
    float angle = wrappedHue;
    return float3(cos(angle) * radial, l * 2.0 - 1.0, sin(angle) * radial);
  }
  if (plotMode == 4 || plotMode == 5) {
    bool jpOverflow = (showOverflow != 0 && plotMode == 5);
    float rr = jpOverflow ? r : clamp01(r);
    float gg = jpOverflow ? g : clamp01(g);
    float bb = jpOverflow ? b : clamp01(b);
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan2(rotY, rotX);
    if (hue < 0.0) hue += kTau;
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float magnitude = 0.0;
    if (plotMode == 4) {
      magnitude = clamp(radius3 * 0.576, 0.0, 1.0);
    } else {
      float kAsinInvSqrt2 = asin(1.0 / sqrt(2.0));
      float kAsinInvSqrt3 = asin(1.0 / sqrt(3.0));
      float kHueCoef1 = 1.0 / (2.0 - (kAsinInvSqrt2 / kAsinInvSqrt3));
      float huecoef2 = 2.0 * polar * sin((2.0 * kPi / 3.0) - fmod(hue, kPi / 3.0)) / sqrt(3.0);
      float huemag = ((acos(cos(3.0 * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0)) * huecoef2;
      float satmag = sin(huemag + kAsinInvSqrt3);
      magnitude = radius3 * satmag;
      if (jpOverflow && magnitude < 0.0) {
        magnitude = -magnitude;
        hue += kPi;
        if (hue >= kTau) hue -= kTau;
      }
      magnitude = jpOverflow ? magnitude : clamp(magnitude, 0.0, 1.0);
    }
    float phiNorm = jpOverflow ? max(polar / 0.9553166181245093, 0.0) : clamp(polar / 0.9553166181245093, 0.0, 1.0);
    float phi = phiNorm * 0.9553166181245093;
    float radial = magnitude * sin(phi);
    return float3(cos(hue) * radial, magnitude * cos(phi) * 2.0 - 1.0, sin(hue) * radial);
  }
  if (plotMode == 6) {
    bool normConeOverflow = (showOverflow != 0 && plotMode == 6);
    float rr = normConeOverflow ? r : clamp(r, 0.0, 1.0);
    float gg = normConeOverflow ? g : clamp(g, 0.0, 1.0);
    float bb = normConeOverflow ? b : clamp(b, 0.0, 1.0);
    float maxRgb = max(rr, max(gg, bb));
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan2(rotY, rotX) / kTau;
    if (hue < 0.0) hue += 1.0;
    float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
    float polar = atan2(chromaRadius, rotZ);
    float chroma = polar / 0.9553166181245093;
    if (normConeNormalized != 0) {
      float angle = hue * kTau - kPi / 6.0;
      float cosPolar = cos(polar);
      float safeCos = abs(cosPolar) > 1e-6 ? cosPolar : (cosPolar < 0.0 ? -1e-6 : 1e-6);
      float cone = (sin(polar) / safeCos) / sqrt(2.0);
      float sinTerm = clamp(sin(3.0 * angle), -1.0, 1.0);
      float chromaGain = 1.0 / (2.0 * cos(acos(sinTerm) / 3.0));
      chroma = chromaGain > 1e-6 ? cone / chromaGain : 0.0;
      if (normConeOverflow && chroma < 0.0) {
        chroma = -chroma;
        hue += 0.5;
        if (hue >= 1.0) hue -= 1.0;
      }
    }
    chroma = normConeOverflow ? max(chroma, 0.0) : clamp(chroma, 0.0, 1.0);
    float value = normConeOverflow ? maxRgb : clamp(maxRgb, 0.0, 1.0);
    float angle = hue * kTau;
    return float3(cos(angle) * chroma, value * 2.0 - 1.0, sin(angle) * chroma);
  }
  if (plotMode == 7) {
    bool reuleauxOverflow = (showOverflow != 0 && plotMode == 7);
    float rr = reuleauxOverflow ? r : clamp01(r);
    float gg = reuleauxOverflow ? g : clamp01(g);
    float bb = reuleauxOverflow ? b : clamp01(b);
    float rotX = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float rotY = (gg - bb) * 0.40824829046;
    float rotZ = (rr + gg + bb) / 3.0;
    float hue = kPi - atan2(rotY, -rotX);
    if (hue < 0.0) hue += kTau;
    if (hue >= kTau) hue = fmod(hue, kTau);
    float sat = abs(rotZ) <= 1e-6 ? 0.0 : length(float2(rotX, rotY)) / rotZ;
    if (reuleauxOverflow && sat < 0.0) {
      sat = -sat;
      hue += kPi;
      if (hue >= kTau) hue -= kTau;
    }
    sat = reuleauxOverflow ? sat / 1.41421356237 : clamp(sat / 1.41421356237, 0.0, 1.0);
    float value = reuleauxOverflow ? max(rr, max(gg, bb)) : clamp(max(rr, max(gg, bb)), 0.0, 1.0);
    return float3(cos(hue) * sat, value * 2.0 - 1.0, sin(hue) * sat);
  }
  return float3(r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0);
}

void mapDisplayColor(float inR, float inG, float inB, thread float& outR, thread float& outG, thread float& outB) {
  outR = pow(clamp01(inR), 1.0 / 2.2);
  outG = pow(clamp01(inG), 1.0 / 2.2);
  outB = pow(clamp01(inB), 1.0 / 2.2);
}

void rgbToHsl(float r, float g, float b, thread float& outH, thread float& outS, thread float& outL) {
  float cMax = max(r, max(g, b));
  float cMin = min(r, min(g, b));
  float delta = cMax - cMin;
  outL = 0.5 * (cMax + cMin);
  outH = 0.0;
  outS = 0.0;
  if (delta > 1e-6) {
    float denom = max(1e-6, 1.0 - abs(2.0 * outL - 1.0));
    outS = delta / denom;
    outH = rawRgbHue01(r, g, b, cMax, delta);
  }
}

float hueToRgbChannel(float p, float q, float t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
  if (t < 1.0 / 2.0) return q;
  if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  return p;
}

void hslToRgb(float h, float s, float l, thread float& outR, thread float& outG, thread float& outB) {
  h = wrapHue01(h);
  s = clamp01(s);
  l = clamp01(l);
  if (s <= 1e-6) {
    outR = l;
    outG = l;
    outB = l;
    return;
  }
  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;
  outR = clamp01(hueToRgbChannel(p, q, h + 1.0 / 3.0));
  outG = clamp01(hueToRgbChannel(p, q, h));
  outB = clamp01(hueToRgbChannel(p, q, h - 1.0 / 3.0));
}

void applyDisplaySaturation(float saturation, thread float& r, thread float& g, thread float& b) {
  float sat = clamp(saturation, 1.0, 6.0);
  float baseR = clamp01(r);
  float baseG = clamp01(g);
  float baseB = clamp01(b);
  float luma = clamp(baseR * 0.2126 + baseG * 0.7152 + baseB * 0.0722, 0.0, 1.0);
  if (sat <= 1.0) {
    r = max(0.0, luma + (baseR - luma) * sat);
    g = max(0.0, luma + (baseG - luma) * sat);
    b = max(0.0, luma + (baseB - luma) * sat);
  } else {
    float h = 0.0;
    float s = 0.0;
    float l = 0.0;
    rgbToHsl(baseR, baseG, baseB, h, s, l);
    if (s <= 1e-5) {
      r = baseR;
      g = baseG;
      b = baseB;
    } else {
      float t = clamp((sat - 1.0) / 5.0, 0.0, 1.0);
      float shaped = pow(t, 0.55);
      float targetS = clamp(s + (1.0 - s) * (0.32 + 0.68 * shaped), 0.0, 1.0);
      float highlight = clamp((l - 0.58) / 0.34, 0.0, 1.0);
      float targetL = clamp(l - highlight * (0.08 + 0.10 * shaped), 0.0, 1.0);
      float boostedR = baseR;
      float boostedG = baseG;
      float boostedB = baseB;
      hslToRgb(h, targetS, targetL, boostedR, boostedG, boostedB);
      float mixAmount = clamp(0.24 + 0.76 * shaped, 0.0, 1.0);
      r = max(0.0, baseR * (1.0 - mixAmount) + boostedR * mixAmount);
      g = max(0.0, baseG * (1.0 - mixAmount) + boostedG * mixAmount);
      b = max(0.0, baseB * (1.0 - mixAmount) + boostedB * mixAmount);
    }
  }
  float peak = max(r, max(g, b));
  if (peak > 1.0) {
    r /= peak;
    g /= peak;
    b /= peak;
  }
  r = clamp(r, 0.0, 1.0);
  g = clamp(g, 0.0, 1.0);
  b = clamp(b, 0.0, 1.0);
}

bool pointOverflowsCube(float r, float g, float b) {
  return r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0;
}

void writeHiddenInputPoint(device packed_float3* vertVals, device float4* colorVals, uint index) {
  vertVals[index] = packed_float3(0.0, 0.0, 0.0);
  colorVals[index] = float4(0.0, 0.0, 0.0, 0.0);
}

void writeMappedInputPoint(device packed_float3* vertVals,
                           device float4* colorVals,
                           uint index,
                           float xNorm,
                           float yNorm,
                           float r,
                           float g,
                           float b,
                           constant InputUniforms& u) {
  bool overflowPoint = pointOverflowsCube(r, g, b);
  float plotR = (u.showOverflow != 0) ? r : clamp01(r);
  float plotG = (u.showOverflow != 0) ? g : clamp01(g);
  float plotB = (u.showOverflow != 0) ? b : clamp01(b);
  float3 pos = mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow);
  if (u.plotMode == 8) {
    pos = mapChromaticityPosition(r, g, b, u);
  }
  if (u.glossView != 0) {
    float aspect = clamp(u.sourceAspect, 0.25, 4.0);
    float halfWidth = aspect >= 1.0 ? 1.22 : (1.22 * aspect);
    float halfDepth = aspect >= 1.0 ? (1.22 / aspect) : 1.22;
    float common = glossCommonComponent(plotR, plotG, plotB);
    float bodyR = max(plotR - common, 0.0);
    float bodyG = max(plotG - common, 0.0);
    float bodyB = max(plotB - common, 0.0);
    float bodyLuma = clamp(bodyR * 0.2126 + bodyG * 0.7152 + bodyB * 0.0722, 0.0, 1.0);
    float glossCue = glossStrengthCue(plotR, plotG, plotB);
    float glossPresence = glossPresenceWeight(glossCue);
    float xPos = -halfWidth + (2.0 * halfWidth * xNorm);
    float zPos = halfDepth - (2.0 * halfDepth * yNorm);
    float yPos = -0.92 + bodyLuma * 0.92 + glossCue * glossPresence * u.glossLiftScale * 1.34;
    pos = float3(xPos, yPos, zPos);
  }
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(r, g, b, cr, cg, cb);
    applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
    if (u.glossView != 0) {
      float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
      float neutralBlend = clamp(0.08 + 0.52 * glossPresence, 0.0, 0.62);
      float brightnessGain = 1.18 + 1.20 * glossPresence;
      cr = clamp((cr * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cg = clamp((cg * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cb = clamp((cb * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
    }
  }
  bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint);
  float baseAlpha = overflowHighlighted ? 0.95 : 0.72;
  if (u.glossView != 0 && !overflowHighlighted) {
    float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
    baseAlpha = 0.01 + 0.97 * glossPresence;
  }
  colorVals[index] = float4(cr, cg, cb,
                            luminanceAwareAlpha(baseAlpha,
                                                cr,
                                                cg,
                                                cb,
                                                u.denseAlphaBias,
                                                overflowHighlighted,
                                                u.pointAlphaScale));
}

bool rasterSourceRowInRange(int y, int y1, int y2) {
  return y1 >= 0 && y2 > y1 && y >= y1 && y < y2;
}

bool rasterSourceRowInCube(constant RasterSourceUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityCubeY1, u.identityCubeY2);
}

bool rasterSourceRowInRamp(constant RasterSourceUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityRampY1, u.identityRampY2);
}

bool rasterAppendSampleCoords(uint index,
                              int offset,
                              int count,
                              int y1,
                              int y2,
                              int rowStep,
                              int xStep,
                              int sourceWidth,
                              thread int* outX,
                              thread int* outY) {
  if (count <= 0 || index < uint(max(offset, 0)) || index >= uint(max(offset + count, offset))) {
    return false;
  }
  int local = int(index) - offset;
  int safeXStep = max(xStep, 1);
  int safeRowStep = max(rowStep, 1);
  int samplesPerRow = max(1, (max(sourceWidth, 0) + safeXStep - 1) / safeXStep);
  int rowIndex = local / samplesPerRow;
  int xIndex = local - rowIndex * samplesPerRow;
  *outX = min(max(xIndex * safeXStep, 0), max(sourceWidth - 1, 0));
  *outY = min(max(y1 + rowIndex * safeRowStep, y1), max(y2 - 1, y1));
  return true;
}

float rasterHalton(uint index, uint base) {
  float f = 1.0;
  float r = 0.0;
  while (index > 0u) {
    f /= float(base);
    r += f * float(index % base);
    index /= base;
  }
  return r;
}

bool rasterOccupancySampleCoords(uint index, constant RasterSourceUniforms& u, thread int* outX, thread int* outY) {
  if (u.occupancyAppendCount <= 0 || index < uint(max(u.occupancyAppendOffset, 0)) ||
      index >= uint(max(u.occupancyAppendOffset + u.occupancyAppendCount, u.occupancyAppendOffset))) {
    return false;
  }
  uint local = index - uint(max(u.occupancyAppendOffset, 0));
  uint attemptCount = uint(max(u.occupancyCandidateCount, u.occupancyAppendCount));
  uint attempt = attemptCount > 0u ? (local * max(attemptCount, 1u)) / uint(max(u.occupancyAppendCount, 1)) : local;
  float xNorm = rasterHalton(attempt + 1u, 2u);
  float yNorm = rasterHalton(attempt + 1u, 3u);
  *outX = min(max(int(xNorm * float(max(u.sourceWidth, 1))), 0), max(u.sourceWidth - 1, 0));
  *outY = min(max(int(yNorm * float(max(u.sourceHeight, 1))), 0), max(u.sourceHeight - 1, 0));
  return true;
}

bool rasterLassoPointInStroke(constant RasterSourceUniforms& u, int strokeIndex, float xNorm, float yNorm) {
  if (strokeIndex < 0 || strokeIndex >= min(max(u.lassoStrokeCount, 0), 16)) return false;
  int first = u.lassoStrokeFirst[strokeIndex];
  int count = u.lassoStrokeCountPerStroke[strokeIndex];
  if (count < 3 || first < 0 || first + count > min(max(u.lassoPointCount, 0), 256)) return false;
  bool inside = false;
  for (int i = 0, j = count - 1; i < count; j = i++) {
    float xi = u.lassoX[first + i];
    float yi = u.lassoY[first + i];
    float xj = u.lassoX[first + j];
    float yj = u.lassoY[first + j];
    bool intersects = ((yi > yNorm) != (yj > yNorm)) &&
                      (xNorm < (xj - xi) * (yNorm - yi) / ((yj - yi) + 1.0e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

bool rasterLassoContainsPoint(constant RasterSourceUniforms& u, float xNorm, float yNorm) {
  bool inside = false;
  int strokeCount = min(max(u.lassoStrokeCount, 0), 16);
  for (int stroke = 0; stroke < strokeCount; ++stroke) {
    if (!rasterLassoPointInStroke(u, stroke, xNorm, yNorm)) continue;
    inside = u.lassoStrokeSubtract[stroke] == 0;
  }
  return inside;
}

float rasterNeutralRadius(float r, float g, float b, constant RasterSourceUniforms& u) {
  const float kRgbAxisMaxRadius = 0.8164965809277260;
  const float kPolarMax = 0.9553166181245093;
  const float kChenPolarScale = 1.0467733744265997;
  int mode = u.input.plotMode;
  if (mode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    if (u.input.circularHsl != 0) {
      float l = 0.5 * (cMax + cMin);
      float denom = 1.0 - abs(2.0 * l - 1.0);
      if (abs(denom) <= 1e-6) denom = denom < 0.0 ? -1e-6 : 1e-6;
      return clamp01(abs((cMax - cMin) / denom));
    }
    return clamp01(cMax - cMin);
  }
  if (mode == 2) {
    if (u.input.circularHsv != 0) {
      float cMax = max(r, max(g, b));
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      return (delta > 1e-6 && cMax > 1e-6) ? clamp01(delta / cMax) : 0.0;
    }
    float x = r - 0.5 * g - 0.5 * b;
    float z = 0.8660254037844386 * (g - b);
    return clamp01(sqrt(x * x + z * z));
  }
  bool overflowMode = u.input.showOverflow != 0 && (mode == 5 || mode == 6 || mode == 7);
  float rr = overflowMode ? r : clamp01(r);
  float gg = overflowMode ? g : clamp01(g);
  float bb = overflowMode ? b : clamp01(b);
  float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
  float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
  float rotZ = 0.57735026919 * (rr + gg + bb);
  float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
  if (mode == 3) {
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(chromaRadius, max(rotZ, 1e-8));
    float light = radius3 * 0.5773502691896258;
    float radial = light * sin(polar * kChenPolarScale) / kRgbAxisMaxRadius;
    return clamp01(radial);
  }
  if (mode == 4 || mode == 5) {
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float polar = atan2(chromaRadius, rotZ);
    float radial = radius3 * sin((polar / kPolarMax) * kPolarMax);
    return clamp01(radial / sin(kPolarMax));
  }
  if (mode == 6) {
    float polar = atan2(chromaRadius, rotZ);
    return clamp01(polar / kPolarMax);
  }
  if (mode == 7) {
    float rotZAvg = (rr + gg + bb) / 3.0;
    float rx = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float ry = (gg - bb) * 0.40824829046;
    float sat = abs(rotZAvg) <= 1e-6 ? 0.0 : sqrt(rx * rx + ry * ry) / rotZAvg;
    return clamp01(abs(sat) / 1.41421356237);
  }
  return clamp01(sqrt(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius);
}

bool rasterCubeSliceContains(float r, float g, float b, constant RasterSourceUniforms& u) {
  if (u.neutralRadiusEnabled != 0 && u.input.plotMode != 8 && u.input.showOverflow == 0) {
    float threshold = clamp01(u.neutralRadius) * clamp01(u.neutralRadius);
    if (rasterNeutralRadius(r, g, b, u) > threshold + 1.0e-6) return false;
  }
  if (u.cubeSlicingEnabled == 0) return true;
  bool anySelected = u.cubeSliceRed || u.cubeSliceYellow || u.cubeSliceGreen ||
                     u.cubeSliceCyan || u.cubeSliceBlue || u.cubeSliceMagenta;
  if (!anySelected) return false;
  if (u.input.plotMode == 0 || u.input.glossView != 0) {
    const float kEps = 1.0e-6;
    bool geRG = r + kEps >= g;
    bool geGB = g + kEps >= b;
    bool geGR = g + kEps >= r;
    bool geRB = r + kEps >= b;
    bool geBG = b + kEps >= g;
    bool geBR = b + kEps >= r;
    if (u.cubeSliceRed && geRG && geGB) return true;
    if (u.cubeSliceYellow && geGR && geRB) return true;
    if (u.cubeSliceGreen && geGB && geBR) return true;
    if (u.cubeSliceCyan && geBG && geGR) return true;
    if (u.cubeSliceBlue && geBR && geRG) return true;
    if (u.cubeSliceMagenta && geRB && geBG) return true;
    return false;
  }
  float cMax = max(r, max(g, b));
  float cMin = min(r, min(g, b));
  float delta = cMax - cMin;
  if (delta <= 1.0e-6) return false;
  float hue = wrapHue01(rawRgbHue01(r, g, b, cMax, delta));
  int sector = int(floor((hue + (1.0 / 12.0)) * 6.0)) % 6;
  if (sector == 0) return u.cubeSliceRed != 0;
  if (sector == 1) return u.cubeSliceYellow != 0;
  if (sector == 2) return u.cubeSliceGreen != 0;
  if (sector == 3) return u.cubeSliceCyan != 0;
  if (sector == 4) return u.cubeSliceBlue != 0;
  return u.cubeSliceMagenta != 0;
}

void readRasterSourceRgb(const device half4* source16,
                         const device float4* source32,
                         constant RasterSourceUniforms& u,
                         int x,
                         int y,
                         thread float& r,
                         thread float& g,
                         thread float& b) {
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  uint pixel = uint(y * u.sourceWidth + x);
  if (u.pixelFormat == 1) {
    float4 v = source32[pixel];
    r = v.x;
    g = v.y;
    b = v.z;
  } else {
    half4 v = source16[pixel];
    r = float(v.x);
    g = float(v.y);
    b = float(v.z);
  }
}

int rasterOccupancyComponentBin(float value) {
  if (value < 0.0) return 0;
  if (value > 1.0) return 17;
  return 1 + min(max(int(floor(value * 16.0)), 0), 15);
}

int rasterOccupancyBinIndex(float r, float g, float b) {
  return (rasterOccupancyComponentBin(r) * 18 + rasterOccupancyComponentBin(g)) * 18 +
         rasterOccupancyComponentBin(b);
}

bool rasterSampleVisible(constant RasterSourceUniforms& u,
                         int x,
                         int y,
                         float xNorm,
                         float yNorm,
                         float r,
                         float g,
                         float b) {
  bool inCubeStrip = rasterSourceRowInCube(u, y);
  bool inRampStrip = rasterSourceRowInRamp(u, y);
  bool inAnyIdentityStrip = inCubeStrip || inRampStrip;
  bool visible = true;
  if (u.excludeIdentityData != 0 && inAnyIdentityStrip) {
    visible = false;
  } else if (u.isolateIdentityData != 0) {
    visible = (u.readIdentityPlot != 0 && inCubeStrip) || (u.readGrayRamp != 0 && inRampStrip);
  }
  if (visible && u.lassoEnabled != 0 && !rasterLassoContainsPoint(u, xNorm, yNorm)) {
    visible = false;
  }
  if (visible && !rasterCubeSliceContains(r, g, b, u)) {
    visible = false;
  }
  (void)x;
  return visible;
}

void rasterReadTransformedSample(const device half4* source16,
                                 const device float4* source32,
                                 constant RasterSourceUniforms& u,
                                 int x,
                                 int y,
                                 thread float& r,
                                 thread float& g,
                                 thread float& b) {
  readRasterSourceRgb(source16, source32, u, x, y, r, g, b);
  if (u.plotLinear != 0 && u.input.plotMode != 8) {
    r = decodeTransferChannel(r, u.plotLinearTransfer);
    g = decodeTransferChannel(g, u.plotLinearTransfer);
    b = decodeTransferChannel(b, u.plotLinearTransfer);
  }
}

kernel void rasterOccupancyCountKernel(const device half4* source16 [[buffer(0)]],
                                       const device float4* source32 [[buffer(1)]],
                                       device atomic_uint* occupancyBins [[buffer(2)]],
                                       device atomic_uint* visibleCount [[buffer(3)]],
                                       constant RasterSourceUniforms& u [[buffer(4)]],
                                       uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.basePointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSample(source16, source32, u, x, y, r, g, b);
  if (!rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b)) return;
  atomic_fetch_add_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&visibleCount[0], 1u, memory_order_relaxed);
}

kernel void rasterSourceKernel(const device half4* source16 [[buffer(0)]],
                               const device float4* source32 [[buffer(1)]],
                               device packed_float3* vertVals [[buffer(2)]],
                               device float4* colorVals [[buffer(3)]],
                               device atomic_uint* occupancyBins [[buffer(4)]],
                               constant RasterSourceUniforms& u [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.input.pointCount, 0));
  if (index >= total) return;
  int sampleCountX = max(u.sampleCountX, 1);
  int stride = max(u.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  bool occupancyCandidate = false;
  if (!rasterOccupancySampleCoords(index, u, &x, &y)) {
    if (!rasterAppendSampleCoords(index,
                                  u.identityCubeAppendOffset,
                                  u.identityCubeAppendCount,
                                  u.identityCubeAppendY1,
                                  u.identityCubeAppendY2,
                                  u.identityCubeAppendRowStep,
                                  u.identityCubeAppendXStep,
                                  u.sourceWidth,
                                  &x,
                                  &y)) {
      rasterAppendSampleCoords(index,
                               u.identityRampAppendOffset,
                               u.identityRampAppendCount,
                               u.identityRampAppendY1,
                               u.identityRampAppendY2,
                               u.identityRampAppendRowStep,
                               u.identityRampAppendXStep,
                               u.sourceWidth,
                               &x,
                               &y);
    }
  } else {
    occupancyCandidate = true;
  }
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(u.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(u.sourceHeight, 1));
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  rasterReadTransformedSample(source16, source32, u, x, y, r, g, b);
  bool visible = rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b);
  if (visible && occupancyCandidate) {
    uint binCount = atomic_load_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], memory_order_relaxed);
    visible = int(binCount) <= max(u.occupancyTargetThreshold, 0);
  }
  if (!visible) {
    writeHiddenInputPoint(vertVals, colorVals, index);
    return;
  }
  writeMappedInputPoint(vertVals, colorVals, index, xNorm, yNorm, r, g, b, u.input);
}

kernel void overlayKernel(const device float4* inputVals [[buffer(0)]],
                          device packed_float3* vertVals [[buffer(1)]],
                          device float4* colorVals [[buffer(2)]],
                          constant OverlayUniforms& u [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  uint cubeSize = uint(max(u.cubeSize, 1));
  uint cubePoints = cubeSize * cubeSize * cubeSize;
  uint rampPoints = (u.ramp != 0) ? (cubeSize * cubeSize) : 0u;
  uint uploadedPoints = uint(max(u.pointCount, 0));
  uint total = (u.useInputPoints != 0) ? uploadedPoints : (cubePoints + rampPoints);
  if (index >= total) return;

  float r;
  float g;
  float b;
  float alpha;
  if (u.useInputPoints != 0) {
    float4 p = inputVals[index];
    r = p.x;
    g = p.y;
    b = p.z;
    alpha = p.w;
  } else if (index < cubePoints) {
    uint denom = max(cubeSize - 1u, 1u);
    uint rx = index % cubeSize;
    uint gy = (index / cubeSize) % cubeSize;
    uint bz = index / (cubeSize * cubeSize);
    r = float(rx) / float(denom);
    g = float(gy) / float(denom);
    b = float(bz) / float(denom);
    alpha = 0.24;
  } else {
    uint rampIndex = index - cubePoints;
    uint rampCount = max(rampPoints, 1u);
    float t = float(rampIndex) / float(max(rampCount - 1u, 1u));
    r = t;
    g = t;
    b = t;
    alpha = 0.92;
  }

  float3 pos = mapPlotPosition(r, g, b, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, 0);
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  mapDisplayColor(r, g, b, cr, cg, cb);
  applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
  colorVals[index] = float4(cr, cg, cb, alpha);
}

kernel void inputKernel(const device float* inputVals [[buffer(0)]],
                        device packed_float3* vertVals [[buffer(1)]],
                        device float4* colorVals [[buffer(2)]],
                        constant InputUniforms& u [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total) return;
  uint stride = uint(max(u.inputStride, 3));
  uint base = index * stride;
  float xNorm = 0.5;
  float yNorm = 0.5;
  float r = inputVals[base + 0];
  float g = inputVals[base + 1];
  float b = inputVals[base + 2];
  if (u.glossView != 0 && stride >= 6) {
    xNorm = clamp(inputVals[base + 0], 0.0, 1.0);
    yNorm = clamp(inputVals[base + 1], 0.0, 1.0);
    r = inputVals[base + 3];
    g = inputVals[base + 4];
    b = inputVals[base + 5];
  }
  bool overflowPoint = pointOverflowsCube(r, g, b);
  float plotR = (u.showOverflow != 0) ? r : clamp01(r);
  float plotG = (u.showOverflow != 0) ? g : clamp01(g);
  float plotB = (u.showOverflow != 0) ? b : clamp01(b);
  float3 pos = mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow);
  if (u.plotMode == 8) {
    pos = mapChromaticityPosition(r, g, b, u);
  }
  if (u.glossView != 0) {
    float aspect = clamp(u.sourceAspect, 0.25, 4.0);
    float halfWidth = aspect >= 1.0 ? 1.22 : (1.22 * aspect);
    float halfDepth = aspect >= 1.0 ? (1.22 / aspect) : 1.22;
    float common = glossCommonComponent(plotR, plotG, plotB);
    float bodyR = max(plotR - common, 0.0);
    float bodyG = max(plotG - common, 0.0);
    float bodyB = max(plotB - common, 0.0);
    float bodyLuma = clamp(bodyR * 0.2126 + bodyG * 0.7152 + bodyB * 0.0722, 0.0, 1.0);
    float glossCue = glossStrengthCue(plotR, plotG, plotB);
    float glossPresence = glossPresenceWeight(glossCue);
    float xPos = -halfWidth + (2.0 * halfWidth * xNorm);
    float zPos = halfDepth - (2.0 * halfDepth * yNorm);
    float yPos = -0.92 + bodyLuma * 0.92 + glossCue * glossPresence * u.glossLiftScale * 1.34;
    pos = float3(xPos, yPos, zPos);
  }
  vertVals[index] = packed_float3(pos.x, pos.y, pos.z);
  float cr;
  float cg;
  float cb;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(r, g, b, cr, cg, cb);
    applyDisplaySaturation(u.colorSaturation, cr, cg, cb);
    if (u.glossView != 0) {
      float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
      float neutralBlend = clamp(0.08 + 0.52 * glossPresence, 0.0, 0.62);
      float brightnessGain = 1.18 + 1.20 * glossPresence;
      cr = clamp((cr * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cg = clamp((cg * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
      cb = clamp((cb * (1.0 - neutralBlend) + neutralBlend) * brightnessGain, 0.0, 1.0);
    }
  }
  bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint);
  float baseAlpha = overflowHighlighted ? 0.95 : 0.72;
  if (u.glossView != 0 && !overflowHighlighted) {
    float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
    baseAlpha = 0.01 + 0.97 * glossPresence;
  }
  colorVals[index] = float4(cr, cg, cb,
                            luminanceAwareAlpha(baseAlpha,
                                                cr,
                                                cg,
                                                cb,
                                                u.denseAlphaBias,
                                                overflowHighlighted,
                                                u.pointAlphaScale));
}

kernel void inputSampleKernel(const device packed_float3* srcVerts [[buffer(0)]],
                              const device float4* srcColors [[buffer(1)]],
                              device packed_float3* dstVerts [[buffer(2)]],
                              device float4* dstColors [[buffer(3)]],
                              constant InputSampleUniforms& u [[buffer(4)]],
                              uint index [[thread_position_in_grid]]) {
  uint visible = uint(max(u.visiblePointCount, 0));
  uint full = uint(max(u.fullPointCount, 0));
  if (index >= visible) return;
  uint srcIndex = 0u;
  if (visible > 1u && full > 1u) {
    float t = float(index) / float(visible - 1u);
    srcIndex = min(uint(floor(t * float(full - 1u) + 0.5)), full - 1u);
  }
  packed_float3 src = srcVerts[srcIndex];
  dstVerts[index] = packed_float3(src.x, src.y, src.z);
  dstColors[index] = srcColors[srcIndex];
}

void accumulateScopeDensity(device atomic_uint* density,
                            constant ScopeDensityUniforms& u,
                            int channel,
                            float xNorm,
                            float value) {
  if (u.excludeOverflow != 0 && (value < 0.0 || value > 1.0)) return;
  int x = clamp(int(xNorm * float(u.width)), 0, max(u.width - 1, 0));
  int signalBins = u.waveform != 0 ? u.height : u.width;
  int y = clamp(int((value - u.rangeMin) * u.invRange * float(signalBins)),
                0,
                max(signalBins - 1, 0));
  int binIndex = u.waveform != 0
      ? (channel * u.width + x) * u.height + y
      : channel * u.width + y;
  atomic_fetch_add_explicit(&density[binIndex], 1u, memory_order_relaxed);
}

kernel void scopeDensityKernel(const device float* samples [[buffer(0)]],
                               device atomic_uint* density [[buffer(1)]],
                               constant ScopeDensityUniforms& u [[buffer(2)]],
                               uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total || u.width <= 0 || u.height <= 0) return;
  uint base = index * 5u;
  float xNorm = clamp(samples[base + 0u], 0.0, 1.0);
  float r = samples[base + 2u];
  float g = samples[base + 3u];
  float b = samples[base + 4u];
  bool lumaOnly = (u.waveform != 0 && u.scopeMode == 2) ||
                  (u.waveform == 0 && u.scopeMode == 1);
  if (lumaOnly) {
    accumulateScopeDensity(density, u, 0, xNorm, 0.2126 * r + 0.7152 * g + 0.0722 * b);
  } else {
    accumulateScopeDensity(density, u, 0, xNorm, r);
    accumulateScopeDensity(density, u, 1, xNorm, g);
    accumulateScopeDensity(density, u, 2, xNorm, b);
  }
}

int glossNeighborhoodRadiusCells(int neighborhoodChoice) {
  switch (clamp(neighborhoodChoice, 0, 2)) {
    case 0: return 1;
    case 2: return 3;
    case 1:
    default: return 2;
  }
}

float sampleGridClamped(const device float* values, int width, int height, int x, int y) {
  if (values == nullptr || width <= 0 || height <= 0) return 0.0;
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  return values[uint(y * width + x)];
}

kernel void glossFieldAccumulateKernel(const device float* packedPoints [[buffer(0)]],
                                       device atomic_uint* occupancyCounts [[buffer(1)]],
                                       device atomic_uint* sumR [[buffer(2)]],
                                       device atomic_uint* sumG [[buffer(3)]],
                                       device atomic_uint* sumB [[buffer(4)]],
                                       device atomic_uint* sumY [[buffer(5)]],
                                       device atomic_uint* sumMax [[buffer(6)]],
                                       device atomic_uint* sumMin [[buffer(7)]],
                                       device atomic_uint* sumNeutrality [[buffer(8)]],
                                       constant GlossFieldAccumulateUniforms& u [[buffer(9)]],
                                       uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total) return;
  uint base = index * 6u;
  float xNorm = clamp(packedPoints[base + 0u], 0.0, 1.0);
  float yNorm = clamp(packedPoints[base + 1u], 0.0, 1.0);
  float r = packedPoints[base + 3u];
  float g = packedPoints[base + 4u];
  float b = packedPoints[base + 5u];
  if (u.showOverflow == 0) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  float maxRgb = clamp(max(r, max(g, b)), 0.0, 1.0);
  float minRgb = clamp(max(0.0, min(r, min(g, b))), 0.0, 1.0);
  float neutralityValue = maxRgb > 1e-6 ? clamp(minRgb / maxRgb, 0.0, 1.0) : 0.0;
  float luma = clamp(r * 0.2126 + g * 0.7152 + b * 0.0722, 0.0, 1.0);
  int x = clamp(int(xNorm * float(u.gridWidth)), 0, max(u.gridWidth - 1, 0));
  int y = clamp(int((1.0 - yNorm) * float(u.gridHeight)), 0, max(u.gridHeight - 1, 0));
  uint cellIndex = uint(y * u.gridWidth + x);
  atomic_fetch_add_explicit(&occupancyCounts[cellIndex], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&sumR[cellIndex], glossEncodeAccum(r), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumG[cellIndex], glossEncodeAccum(g), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumB[cellIndex], glossEncodeAccum(b), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumY[cellIndex], glossEncodeAccum(luma), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMax[cellIndex], glossEncodeAccum(maxRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumMin[cellIndex], glossEncodeAccum(minRgb), memory_order_relaxed);
  atomic_fetch_add_explicit(&sumNeutrality[cellIndex], glossEncodeAccum(neutralityValue), memory_order_relaxed);
}

kernel void glossFieldFinalizeKernel(const device atomic_uint* occupancyCounts [[buffer(0)]],
                                     const device atomic_uint* sumR [[buffer(1)]],
                                     const device atomic_uint* sumG [[buffer(2)]],
                                     const device atomic_uint* sumB [[buffer(3)]],
                                     const device atomic_uint* sumY [[buffer(4)]],
                                     const device atomic_uint* sumMax [[buffer(5)]],
                                     const device atomic_uint* sumMin [[buffer(6)]],
                                     const device atomic_uint* sumNeutrality [[buffer(7)]],
                                     device float* occupancy [[buffer(8)]],
                                     device float* meanR [[buffer(9)]],
                                     device float* meanG [[buffer(10)]],
                                     device float* meanB [[buffer(11)]],
                                     device float* carrierY [[buffer(12)]],
                                     device float* carrierMax [[buffer(13)]],
                                     device float* carrierMin [[buffer(14)]],
                                     device float* neutrality [[buffer(15)]],
                                     constant GlossFieldCellUniforms& u [[buffer(16)]],
                                     uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  uint count = atomic_load_explicit(&occupancyCounts[index], memory_order_relaxed);
  occupancy[index] = float(count);
  if (count == 0u) {
    meanR[index] = 0.0;
    meanG[index] = 0.0;
    meanB[index] = 0.0;
    carrierY[index] = 0.0;
    carrierMax[index] = 0.0;
    carrierMin[index] = 0.0;
    neutrality[index] = 0.0;
    return;
  }
  float invCount = 1.0 / float(count);
  meanR[index] = glossDecodeAccum(atomic_load_explicit(&sumR[index], memory_order_relaxed)) * invCount;
  meanG[index] = glossDecodeAccum(atomic_load_explicit(&sumG[index], memory_order_relaxed)) * invCount;
  meanB[index] = glossDecodeAccum(atomic_load_explicit(&sumB[index], memory_order_relaxed)) * invCount;
  carrierY[index] = glossDecodeAccum(atomic_load_explicit(&sumY[index], memory_order_relaxed)) * invCount;
  carrierMax[index] = glossDecodeAccum(atomic_load_explicit(&sumMax[index], memory_order_relaxed)) * invCount;
  carrierMin[index] = glossDecodeAccum(atomic_load_explicit(&sumMin[index], memory_order_relaxed)) * invCount;
  neutrality[index] = glossDecodeAccum(atomic_load_explicit(&sumNeutrality[index], memory_order_relaxed)) * invCount;
}

kernel void glossFieldMaxKernel(const device float* values [[buffer(0)]],
                                device atomic_uint* outBits [[buffer(1)]],
                                constant GlossFieldCellUniforms& u [[buffer(2)]],
                                uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  atomic_fetch_max_explicit(&outBits[0], as_type<uint>(max(values[index], 0.0)), memory_order_relaxed);
}

kernel void glossFieldNormalizeKernel(const device float* src [[buffer(0)]],
                                      device float* dst [[buffer(1)]],
                                      const device atomic_uint* maxBits [[buffer(2)]],
                                      constant GlossFieldCellUniforms& u [[buffer(3)]],
                                      uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  float denom = max(as_type<float>(atomic_load_explicit(&maxBits[0], memory_order_relaxed)), 1e-5);
  dst[index] = clamp(src[index] / denom, 0.0, 1.0);
}

kernel void glossFieldBlurKernel(const device float* src [[buffer(0)]],
                                 device float* dst [[buffer(1)]],
                                 constant GlossFieldCellUniforms& u [[buffer(2)]],
                                 uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float accum = 0.0;
  float weight = 0.0;
  for (int oy = -1; oy <= 1; ++oy) {
    int yy = y + oy;
    if (yy < 0 || yy >= u.gridHeight) continue;
    for (int ox = -1; ox <= 1; ++ox) {
      int xx = x + ox;
      if (xx < 0 || xx >= u.gridWidth) continue;
      float kernel = (ox == 0 && oy == 0) ? 0.30 : ((ox == 0 || oy == 0) ? 0.13 : 0.08);
      accum += src[uint(yy * u.gridWidth + xx)] * kernel;
      weight += kernel;
    }
  }
  dst[index] = weight > 1e-6 ? (accum / weight) : 0.0;
}

kernel void glossFieldBodyKernel(const device float* occupancy [[buffer(0)]],
                                 const device float* meanR [[buffer(1)]],
                                 const device float* meanG [[buffer(2)]],
                                 const device float* meanB [[buffer(3)]],
                                 const device float* carrierMax [[buffer(4)]],
                                 device float* body [[buffer(5)]],
                                 constant GlossFieldCellUniforms& u [[buffer(6)]],
                                 uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    body[index] = 0.0;
    return;
  }
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  int radiusCells = glossNeighborhoodRadiusCells(u.neighborhoodChoice);
  const int kMaxNeighborhood = 49;
  float carriers[kMaxNeighborhood];
  int neighborIndices[kMaxNeighborhood];
  int count = 0;
  float centerCarrier = carrierMax[index];
  float centerR = meanR[index];
  float centerG = meanG[index];
  float centerB = meanB[index];
  for (int oy = -radiusCells; oy <= radiusCells; ++oy) {
    int yy = y + oy;
    if (yy < 0 || yy >= u.gridHeight) continue;
    for (int ox = -radiusCells; ox <= radiusCells; ++ox) {
      int xx = x + ox;
      if (xx < 0 || xx >= u.gridWidth) continue;
      uint neighborIndex = uint(yy * u.gridWidth + xx);
      if (occupancy[neighborIndex] <= 0.5) continue;
      float carrier = carrierMax[neighborIndex];
      float dr = meanR[neighborIndex] - centerR;
      float dg = meanG[neighborIndex] - centerG;
      float db = meanB[neighborIndex] - centerB;
      float colorDistance = sqrt(dr * dr + dg * dg + db * db);
      if (abs(carrier - centerCarrier) > 0.26 && colorDistance > 0.20) continue;
      if (count < kMaxNeighborhood) {
        carriers[count] = carrier;
        neighborIndices[count] = int(neighborIndex);
        ++count;
      }
    }
  }
  if (count <= 0) {
    body[index] = centerCarrier;
    return;
  }
  for (int i = 1; i < count; ++i) {
    float keyCarrier = carriers[i];
    int keyIndex = neighborIndices[i];
    int j = i - 1;
    while (j >= 0 && (carriers[j] > keyCarrier || (carriers[j] == keyCarrier && neighborIndices[j] > keyIndex))) {
      carriers[j + 1] = carriers[j];
      neighborIndices[j + 1] = neighborIndices[j];
      --j;
    }
    carriers[j + 1] = keyCarrier;
    neighborIndices[j + 1] = keyIndex;
  }
  int trim = count >= 6 ? max(1, count / 6) : 0;
  int begin = min(trim, count);
  int end = max(begin + 1, count - trim);
  float bodySum = 0.0;
  float bodyWeight = 0.0;
  for (int i = begin; i < end; ++i) {
    int neighborIndex = neighborIndices[i];
    int neighborX = neighborIndex % u.gridWidth;
    int neighborY = neighborIndex / u.gridWidth;
    float dx = float(neighborX - x);
    float dy = float(neighborY - y);
    float spatialWeight = 1.0 / (1.0 + dx * dx + dy * dy);
    bodySum += carriers[i] * spatialWeight;
    bodyWeight += spatialWeight;
  }
  body[index] = bodyWeight > 1e-6 ? (bodySum / bodyWeight) : centerCarrier;
}

kernel void glossFieldRawSignalKernel(const device float* occupancy [[buffer(0)]],
                                      const device float* carrierMax [[buffer(1)]],
                                      const device float* body [[buffer(2)]],
                                      device float* rawSignal [[buffer(3)]],
                                      device atomic_uint* maxBits [[buffer(4)]],
                                      constant GlossFieldCellUniforms& u [[buffer(5)]],
                                      uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    rawSignal[index] = 0.0;
    return;
  }
  float bodyValue = max(body[index], 0.0);
  float rawPositive = max(0.0, carrierMax[index] - bodyValue);
  float rawNegative = max(0.0, bodyValue - carrierMax[index]);
  rawSignal[index] = rawPositive - rawNegative;
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(bodyValue), memory_order_relaxed);
}

kernel void glossFieldWeightedSignalKernel(const device float* occupancyNorm [[buffer(0)]],
                                           const device float* body [[buffer(1)]],
                                           const device float* rawSignal [[buffer(2)]],
                                           device float* positive [[buffer(3)]],
                                           device float* negative [[buffer(4)]],
                                           device float* boundary [[buffer(5)]],
                                           device float* congruence [[buffer(6)]],
                                           device float* confidence [[buffer(7)]],
                                           device float* signal [[buffer(8)]],
                                           device atomic_uint* maxBits [[buffer(9)]],
                                           constant GlossFieldCellUniforms& u [[buffer(10)]],
                                           uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float occCenter = sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y);
  if (occCenter <= 0.0) {
    positive[index] = 0.0;
    negative[index] = 0.0;
    boundary[index] = 0.0;
    congruence[index] = 0.0;
    confidence[index] = 0.0;
    signal[index] = 0.0;
    return;
  }
  float gxCarrier = sampleGridClamped(body, u.gridWidth, u.gridHeight, x + 1, y) -
                    sampleGridClamped(body, u.gridWidth, u.gridHeight, x - 1, y);
  float gyCarrier = sampleGridClamped(body, u.gridWidth, u.gridHeight, x, y + 1) -
                    sampleGridClamped(body, u.gridWidth, u.gridHeight, x, y - 1);
  float gxSignal = sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x + 1, y) -
                   sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x - 1, y);
  float gySignal = sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x, y + 1) -
                   sampleGridClamped(rawSignal, u.gridWidth, u.gridHeight, x, y - 1);
  float magCarrier = sqrt(gxCarrier * gxCarrier + gyCarrier * gyCarrier);
  float magSignal = sqrt(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0;
  if (magCarrier > 1e-6 && magSignal > 1e-6) {
    localCongruence = abs((gxCarrier * gxSignal + gyCarrier * gySignal) / (magCarrier * magSignal));
  } else if (magSignal > 1e-6) {
    localCongruence = 0.35;
  }
  float occNeighborhood =
      (occCenter +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x + 1, y) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x - 1, y) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y + 1) +
       sampleGridClamped(occupancyNorm, u.gridWidth, u.gridHeight, x, y - 1)) / 5.0;
  float localConfidence = clamp(sqrt(occCenter) * clamp(0.28 + 0.72 * occNeighborhood, 0.0, 1.0), 0.0, 1.0);
  float posWeighted = max(0.0, rawSignal[index]) * (0.30 + 0.70 * localCongruence) * localConfidence;
  float negWeighted = max(0.0, -rawSignal[index]) * (0.30 + 0.70 * localCongruence) * localConfidence;
  float boundaryValue = clamp(magSignal * 4.0, 0.0, 1.0) * localConfidence;
  positive[index] = posWeighted;
  negative[index] = negWeighted;
  boundary[index] = boundaryValue;
  congruence[index] = localCongruence;
  confidence[index] = localConfidence;
  signal[index] = posWeighted - negWeighted;
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(max(posWeighted, 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[1], as_type<uint>(max(negWeighted, 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[2], as_type<uint>(max(boundaryValue, 0.0)), memory_order_relaxed);
}

kernel void glossFieldFinalNormalizeKernel(device float* body [[buffer(0)]],
                                           device float* signal [[buffer(1)]],
                                           device float* positive [[buffer(2)]],
                                           device float* negative [[buffer(3)]],
                                           device float* boundary [[buffer(4)]],
                                           const device atomic_uint* maxBits [[buffer(5)]],
                                           constant GlossFieldCellUniforms& u [[buffer(6)]],
                                           uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  float maxBody = max(as_type<float>(atomic_load_explicit(&maxBits[0], memory_order_relaxed)), 1e-5);
  float maxPositive = max(as_type<float>(atomic_load_explicit(&maxBits[1], memory_order_relaxed)), 1e-5);
  float maxNegative = max(as_type<float>(atomic_load_explicit(&maxBits[2], memory_order_relaxed)), 1e-5);
  float maxBoundary = max(as_type<float>(atomic_load_explicit(&maxBits[3], memory_order_relaxed)), 1e-5);
  float maxAbsSignal = max(max(maxPositive, maxNegative), 1e-5);
  body[index] = clamp(body[index] / maxBody, 0.0, 1.0);
  positive[index] = clamp(positive[index] / maxPositive, 0.0, 1.0);
  negative[index] = clamp(negative[index] / maxNegative, 0.0, 1.0);
  signal[index] = clamp(signal[index] / maxAbsSignal, -1.0, 1.0);
  boundary[index] = clamp(boundary[index] / maxBoundary, 0.0, 1.0);
}
)MSL";

bool ensureContext(std::string* error) {
  static std::once_flag once;
  MetalContext& ctx = context();
  std::call_once(once, []() {
    @autoreleasepool {
      MetalContext& c = context();
      c.initAttempted = true;
      c.device = MTLCreateSystemDefaultDevice();
      if (c.device == nil) {
        c.initError = "No Metal device available.";
        return;
      }
      c.deviceName = [[c.device name] UTF8String] ?: "";
      c.queue = [c.device newCommandQueue];
      if (c.queue == nil) {
        c.initError = "Failed to create Metal command queue.";
        return;
      }
      NSError* libraryError = nil;
      NSString* source = [NSString stringWithUTF8String:kMetalSource];
      c.library = [c.device newLibraryWithSource:source options:nil error:&libraryError];
      if (c.library == nil) {
        c.initError = libraryError != nil ? [[libraryError localizedDescription] UTF8String] : "Failed to compile Metal library.";
        return;
      }
      NSError* pipelineError = nil;
      id<MTLFunction> overlayFn = [c.library newFunctionWithName:@"overlayKernel"];
      if (overlayFn == nil) {
        c.initError = "Missing overlay Metal kernel.";
        return;
      }
      c.overlayPipeline = [c.device newComputePipelineStateWithFunction:overlayFn error:&pipelineError];
      if (c.overlayPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create overlay Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> inputFn = [c.library newFunctionWithName:@"inputKernel"];
      if (inputFn == nil) {
        c.initError = "Missing input Metal kernel.";
        return;
      }
      c.inputPipeline = [c.device newComputePipelineStateWithFunction:inputFn error:&pipelineError];
      if (c.inputPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create input Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterSourceFn = [c.library newFunctionWithName:@"rasterSourceKernel"];
      if (rasterSourceFn == nil) {
        c.initError = "Missing raster source Metal kernel.";
        return;
      }
      c.rasterSourcePipeline = [c.device newComputePipelineStateWithFunction:rasterSourceFn error:&pipelineError];
      if (c.rasterSourcePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster source Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterOccupancyFn = [c.library newFunctionWithName:@"rasterOccupancyCountKernel"];
      if (rasterOccupancyFn == nil) {
        c.initError = "Missing raster occupancy Metal kernel.";
        return;
      }
      c.rasterOccupancyCountPipeline = [c.device newComputePipelineStateWithFunction:rasterOccupancyFn error:&pipelineError];
      if (c.rasterOccupancyCountPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster occupancy Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> inputSampleFn = [c.library newFunctionWithName:@"inputSampleKernel"];
      if (inputSampleFn == nil) {
        c.initError = "Missing input sample Metal kernel.";
        return;
      }
      c.inputSamplePipeline = [c.device newComputePipelineStateWithFunction:inputSampleFn error:&pipelineError];
      if (c.inputSamplePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create input sample Metal pipeline.";
        return;
      }
      auto buildPipeline = [&](NSString* name, const char* missingMsg, const char* failMsg) -> id<MTLComputePipelineState> {
        pipelineError = nil;
        id<MTLFunction> fn = [c.library newFunctionWithName:name];
        if (fn == nil) {
          c.initError = missingMsg;
          return nil;
        }
        id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithFunction:fn error:&pipelineError];
        if (pipeline == nil) {
          c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : failMsg;
          return nil;
        }
        return pipeline;
      };
      c.scopeDensityPipeline = buildPipeline(@"scopeDensityKernel",
                                             "Missing scope density Metal kernel.",
                                             "Failed to create scope density Metal pipeline.");
      if (c.scopeDensityPipeline == nil) {
        return;
      }
      c.glossFieldAccumulatePipeline = buildPipeline(@"glossFieldAccumulateKernel",
                                                     "Missing gloss field accumulate Metal kernel.",
                                                     "Failed to create gloss field accumulate Metal pipeline.");
      if (c.glossFieldAccumulatePipeline == nil) {
        return;
      }
      c.glossFieldFinalizePipeline = buildPipeline(@"glossFieldFinalizeKernel",
                                                   "Missing gloss field finalize Metal kernel.",
                                                   "Failed to create gloss field finalize Metal pipeline.");
      if (c.glossFieldFinalizePipeline == nil) {
        return;
      }
      c.glossFieldMaxPipeline = buildPipeline(@"glossFieldMaxKernel",
                                              "Missing gloss field max Metal kernel.",
                                              "Failed to create gloss field max Metal pipeline.");
      if (c.glossFieldMaxPipeline == nil) {
        return;
      }
      c.glossFieldNormalizePipeline = buildPipeline(@"glossFieldNormalizeKernel",
                                                    "Missing gloss field normalize Metal kernel.",
                                                    "Failed to create gloss field normalize Metal pipeline.");
      if (c.glossFieldNormalizePipeline == nil) {
        return;
      }
      c.glossFieldBlurPipeline = buildPipeline(@"glossFieldBlurKernel",
                                               "Missing gloss field blur Metal kernel.",
                                               "Failed to create gloss field blur Metal pipeline.");
      if (c.glossFieldBlurPipeline == nil) {
        return;
      }
      c.glossFieldBodyPipeline = buildPipeline(@"glossFieldBodyKernel",
                                               "Missing gloss field body Metal kernel.",
                                               "Failed to create gloss field body Metal pipeline.");
      if (c.glossFieldBodyPipeline == nil) {
        return;
      }
      c.glossFieldRawSignalPipeline = buildPipeline(@"glossFieldRawSignalKernel",
                                                    "Missing gloss field raw signal Metal kernel.",
                                                    "Failed to create gloss field raw signal Metal pipeline.");
      if (c.glossFieldRawSignalPipeline == nil) {
        return;
      }
      c.glossFieldWeightedSignalPipeline = buildPipeline(@"glossFieldWeightedSignalKernel",
                                                         "Missing gloss field weighted signal Metal kernel.",
                                                         "Failed to create gloss field weighted signal Metal pipeline.");
      if (c.glossFieldWeightedSignalPipeline == nil) {
        return;
      }
      c.glossFieldFinalNormalizePipeline = buildPipeline(@"glossFieldFinalNormalizeKernel",
                                                         "Missing gloss field final normalize Metal kernel.",
                                                         "Failed to create gloss field final normalize Metal pipeline.");
      if (c.glossFieldFinalNormalizePipeline == nil) {
        return;
      }
      c.ready = true;
    }
  });
  if (!ctx.ready && error) *error = ctx.initError;
  return ctx.ready;
}

bool runCompute(id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> inputBuffer,
                id<MTLBuffer> vertBuffer,
                id<MTLBuffer> colorBuffer,
                id<MTLBuffer> uniformBuffer,
                NSUInteger pointCount,
                std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || pipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal context unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal compute encoder.";
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    if (inputBuffer != nil) [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:vertBuffer offset:0 atIndex:1];
    [encoder setBuffer:colorBuffer offset:0 atIndex:2];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:3];
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(pointCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

bool runInputSampleCompute(id<MTLBuffer> srcVertBuffer,
                           id<MTLBuffer> srcColorBuffer,
                           id<MTLBuffer> dstVertBuffer,
                           id<MTLBuffer> dstColorBuffer,
                           id<MTLBuffer> uniformBuffer,
                           NSUInteger pointCount,
                           std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || ctx.inputSamplePipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal input sample pipeline unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal sample command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal sample encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.inputSamplePipeline];
    [encoder setBuffer:srcVertBuffer offset:0 atIndex:0];
    [encoder setBuffer:srcColorBuffer offset:0 atIndex:1];
    [encoder setBuffer:dstVertBuffer offset:0 atIndex:2];
    [encoder setBuffer:dstColorBuffer offset:0 atIndex:3];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:4];
    NSUInteger width = ctx.inputSamplePipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(pointCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

template <size_t N>
bool runComputeBuffers(id<MTLComputePipelineState> pipeline,
                       const std::array<id<MTLBuffer>, N>& buffers,
                       NSUInteger threadCount,
                       std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || pipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal compute pipeline unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal compute encoder.";
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < N; ++i) {
      if (buffers[i] != nil) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(threadCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

template <typename T>
id<MTLBuffer> makeSharedBuffer(const T* values, size_t count) {
  MetalContext& ctx = context();
  if (!ctx.ready) return nil;
  const NSUInteger bytes = static_cast<NSUInteger>(count * sizeof(T));
  return [ctx.device newBufferWithBytes:(values != nullptr ? values : nullptr)
                                 length:bytes
                                options:MTLResourceStorageModeShared];
}

id<MTLBuffer> makeEmptySharedBuffer(NSUInteger bytes) {
  MetalContext& ctx = context();
  if (!ctx.ready) return nil;
  return [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

void clearSharedBuffer(id<MTLBuffer> buffer) {
  if (buffer == nil) return;
  std::memset([buffer contents], 0, static_cast<size_t>([buffer length]));
}

template <typename T>
void copySharedBuffer(id<MTLBuffer> buffer, size_t count, std::vector<float>* out) {
  if (!out) return;
  out->clear();
  if (buffer == nil || count == 0) return;
  out->resize(count * sizeof(T) / sizeof(float));
  std::memcpy(out->data(), [buffer contents], count * sizeof(T));
}

}  // namespace

bool activateWindow(void* nativeWindow) {
  if (nativeWindow == nullptr) return false;
  @autoreleasepool {
    NSWindow* window = (__bridge NSWindow*)nativeWindow;
    if (window == nil) return false;
    [NSApp activateIgnoringOtherApps:YES];
    [window orderFrontRegardless];
    [window makeKeyAndOrderFront:nil];
    return [NSApp isActive] && [window isKeyWindow];
  }
}

uint32_t currentModifierFlags() {
  @autoreleasepool {
    const NSEventModifierFlags flags = [NSEvent modifierFlags];
    uint32_t out = 0;
    if ((flags & NSEventModifierFlagShift) != 0) out |= ModifierFlagShift;
    if ((flags & NSEventModifierFlagControl) != 0) out |= ModifierFlagControl;
    if ((flags & NSEventModifierFlagOption) != 0) out |= ModifierFlagAlt;
    if ((flags & NSEventModifierFlagCommand) != 0) out |= ModifierFlagSuper;
    return out;
  }
}

ProbeResult probe() {
  ProbeResult result{};
  std::string error;
  result.available = ensureContext(&error);
  MetalContext& ctx = context();
  result.queueReady = (ctx.queue != nil);
  result.deviceName = ctx.deviceName.c_str();
  return result;
}

bool buildOverlayMesh(const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      std::vector<float>* outVerts,
                      std::vector<float>* outColors,
                      std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int cubeSize = request.cubeSize > 0 ? request.cubeSize : 1;
  const NSUInteger cubePoints = static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize);
  const NSUInteger rampPoints = request.ramp != 0 ? static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize) : 0u;
  const NSUInteger uploadedPoints = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  const NSUInteger totalPoints = request.useInputPoints != 0 ? uploadedPoints : (cubePoints + rampPoints);
  if (totalPoints == 0) {
    if (outVerts) outVerts->clear();
    if (outColors) outColors->clear();
    return true;
  }

  MetalContext& ctx = context();
  OverlayUniforms uniforms{};
  uniforms.cubeSize = cubeSize;
  uniforms.ramp = request.ramp;
  uniforms.useInputPoints = request.useInputPoints;
  uniforms.pointCount = request.pointCount;
  uniforms.colorSaturation = request.colorSaturation;
  uniforms.plotMode = request.remap.plotMode;
  uniforms.circularHsl = request.remap.circularHsl;
  uniforms.circularHsv = request.remap.circularHsv;
  uniforms.normConeNormalized = request.remap.normConeNormalized;

  id<MTLBuffer> inputBuffer = nil;
  if (request.useInputPoints != 0) {
    if (inputPoints.size() < uploadedPoints * 4u) {
      if (error) *error = "Overlay Metal input point buffer is undersized.";
      return false;
    }
    inputBuffer = makeSharedBuffer(reinterpret_cast<const simd_float4*>(inputPoints.data()), uploadedPoints);
  } else {
    simd_float4 dummy = {0.0f, 0.0f, 0.0f, 0.0f};
    inputBuffer = makeSharedBuffer(&dummy, 1u);
  }
  id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(totalPoints * sizeof(PackedFloat3));
  id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(totalPoints * sizeof(simd_float4));
  id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
  if (inputBuffer == nil || vertBuffer == nil || colorBuffer == nil || uniformBuffer == nil) {
    if (error) *error = "Failed to allocate Metal overlay buffers.";
    return false;
  }

  if (!runCompute(ctx.overlayPipeline, inputBuffer, vertBuffer, colorBuffer, uniformBuffer, totalPoints, &localError)) {
    if (error) *error = localError;
    return false;
  }

  copySharedBuffer<PackedFloat3>(vertBuffer, totalPoints, outVerts);
  copySharedBuffer<simd_float4>(colorBuffer, totalPoints, outColors);
  return true;
}

void fillInputUniforms(const InputRequest& request, InputUniforms* uniforms) {
  if (!uniforms) return;
  uniforms->pointCount = request.pointCount;
  uniforms->inputStride = request.inputStride;
  uniforms->glossView = request.glossView;
  uniforms->sourceAspect = request.sourceAspect;
  uniforms->glossLiftScale = request.glossLiftScale;
  uniforms->showOverflow = request.remap.showOverflow;
  uniforms->highlightOverflow = request.remap.highlightOverflow;
  uniforms->plotMode = request.remap.plotMode;
  uniforms->circularHsl = request.remap.circularHsl;
  uniforms->circularHsv = request.remap.circularHsv;
  uniforms->normConeNormalized = request.remap.normConeNormalized;
  uniforms->chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms->chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms->chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms->chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms->chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms->chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }
  uniforms->pointAlphaScale = request.pointAlphaScale;
  uniforms->denseAlphaBias = request.denseAlphaBias;
  uniforms->colorSaturation = request.colorSaturation;
}

void fillRasterSourceUniforms(const RasterSourceRequest& request, RasterSourceUniforms* uniforms) {
  if (!uniforms) return;
  InputRequest inputRequest{};
  inputRequest.pointCount = request.pointCount;
  inputRequest.inputStride = 3;
  inputRequest.glossView = request.remap.plotMode == 9 ? 1 : 0;
  inputRequest.sourceAspect = request.sourceAspect;
  inputRequest.glossLiftScale = request.glossLiftScale;
  inputRequest.pointAlphaScale = request.pointAlphaScale;
  inputRequest.denseAlphaBias = request.denseAlphaBias;
  inputRequest.colorSaturation = request.colorSaturation;
  inputRequest.remap = request.remap;
  fillInputUniforms(inputRequest, &uniforms->input);
  uniforms->basePointCount = request.basePointCount > 0 ? request.basePointCount : request.pointCount;
  uniforms->sourceWidth = request.sourceWidth;
  uniforms->sourceHeight = request.sourceHeight;
  uniforms->sampleStride = request.sampleStride;
  uniforms->sampleCountX = request.sampleCountX;
  uniforms->pixelFormat = request.pixelFormat;
  uniforms->plotLinear = request.plotLinear;
  uniforms->plotLinearTransfer = request.plotLinearTransfer;
  uniforms->excludeIdentityData = request.excludeIdentityData;
  uniforms->isolateIdentityData = request.isolateIdentityData;
  uniforms->readIdentityPlot = request.readIdentityPlot;
  uniforms->readGrayRamp = request.readGrayRamp;
  uniforms->identityCubeY1 = request.identityCubeY1;
  uniforms->identityCubeY2 = request.identityCubeY2;
  uniforms->identityRampY1 = request.identityRampY1;
  uniforms->identityRampY2 = request.identityRampY2;
  uniforms->identityCubeAppendOffset = request.identityCubeAppendOffset;
  uniforms->identityCubeAppendCount = request.identityCubeAppendCount;
  uniforms->identityCubeAppendY1 = request.identityCubeAppendY1;
  uniforms->identityCubeAppendY2 = request.identityCubeAppendY2;
  uniforms->identityCubeAppendRowStep = request.identityCubeAppendRowStep;
  uniforms->identityCubeAppendXStep = request.identityCubeAppendXStep;
  uniforms->identityRampAppendOffset = request.identityRampAppendOffset;
  uniforms->identityRampAppendCount = request.identityRampAppendCount;
  uniforms->identityRampAppendY1 = request.identityRampAppendY1;
  uniforms->identityRampAppendY2 = request.identityRampAppendY2;
  uniforms->identityRampAppendRowStep = request.identityRampAppendRowStep;
  uniforms->identityRampAppendXStep = request.identityRampAppendXStep;
  uniforms->occupancyFill = request.occupancyFill;
  uniforms->occupancyAppendOffset = request.occupancyAppendOffset;
  uniforms->occupancyAppendCount = request.occupancyAppendCount;
  uniforms->occupancyCandidateCount = request.occupancyCandidateCount;
  uniforms->occupancyTargetThreshold = 0;
  uniforms->lassoEnabled = request.lassoEnabled;
  uniforms->lassoStrokeCount = request.lassoStrokeCount;
  uniforms->lassoPointCount = request.lassoPointCount;
  for (int i = 0; i < 16; ++i) {
    uniforms->lassoStrokeFirst[i] = request.lassoStrokeFirst[i];
    uniforms->lassoStrokeCountPerStroke[i] = request.lassoStrokeCountPerStroke[i];
    uniforms->lassoStrokeSubtract[i] = request.lassoStrokeSubtract[i];
  }
  for (int i = 0; i < 256; ++i) {
    uniforms->lassoX[i] = request.lassoX[i];
    uniforms->lassoY[i] = request.lassoY[i];
  }
  uniforms->cubeSlicingEnabled = request.cubeSlicingEnabled;
  uniforms->neutralRadiusEnabled = request.neutralRadiusEnabled;
  uniforms->neutralRadius = request.neutralRadius;
  uniforms->cubeSliceRed = request.cubeSliceRed;
  uniforms->cubeSliceYellow = request.cubeSliceYellow;
  uniforms->cubeSliceGreen = request.cubeSliceGreen;
  uniforms->cubeSliceCyan = request.cubeSliceCyan;
  uniforms->cubeSliceBlue = request.cubeSliceBlue;
  uniforms->cubeSliceMagenta = request.cubeSliceMagenta;
}

bool buildInputMesh(const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    std::vector<float>* outVerts,
                    std::vector<float>* outColors,
                    std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0) {
    if (outVerts) outVerts->clear();
    if (outColors) outColors->clear();
    return true;
  }
  const size_t inputStride = static_cast<size_t>(std::max(request.inputStride, 3));
  if (rawPoints.size() < pointCount * inputStride) {
    if (error) *error = "Input Metal raw point buffer is undersized.";
    return false;
  }

  MetalContext& ctx = context();
  InputUniforms uniforms{};
  fillInputUniforms(request, &uniforms);

  id<MTLBuffer> inputBuffer = makeSharedBuffer(rawPoints.data(), rawPoints.size());
  id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
  id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
  id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
  if (inputBuffer == nil || vertBuffer == nil || colorBuffer == nil || uniformBuffer == nil) {
    if (error) *error = "Failed to allocate Metal input-cloud buffers.";
    return false;
  }

  if (!runCompute(ctx.inputPipeline, inputBuffer, vertBuffer, colorBuffer, uniformBuffer, pointCount, &localError)) {
    if (error) *error = localError;
    return false;
  }

  copySharedBuffer<PackedFloat3>(vertBuffer, pointCount, outVerts);
  copySharedBuffer<simd_float4>(colorBuffer, pointCount, outColors);
  return true;
}

bool buildRasterSourceMesh(const RasterSourceRequest& request,
                           const void* sourceBytes,
                           size_t sourceByteCount,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0 || sourceBytes == nullptr || sourceByteCount == 0 ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 || request.sampleCountX <= 0) {
    if (error) *error = "Invalid Metal raster source request.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterSourcePipeline == nil || ctx.rasterOccupancyCountPipeline == nil) {
    if (error) *error = "Metal raster source pipeline unavailable.";
    return false;
  }

  RasterSourceUniforms uniforms{};
  fillRasterSourceUniforms(request, &uniforms);

  @autoreleasepool {
    id<MTLBuffer> sourceBuffer = [ctx.device newBufferWithBytes:sourceBytes
                                                         length:static_cast<NSUInteger>(sourceByteCount)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
    id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    constexpr NSUInteger kRasterOccupancyBinCount = 18u * 18u * 18u;
    id<MTLBuffer> occupancyBuffer =
        makeEmptySharedBuffer(kRasterOccupancyBinCount * sizeof(uint32_t));
    if (sourceBuffer == nil || vertBuffer == nil || colorBuffer == nil ||
        uniformBuffer == nil || occupancyBuffer == nil) {
      if (error) *error = "Failed to allocate Metal raster source buffers.";
      return false;
    }
    clearSharedBuffer(occupancyBuffer);

    if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
      id<MTLBuffer> visibleCountBuffer = makeEmptySharedBuffer(sizeof(uint32_t));
      if (visibleCountBuffer == nil) {
        if (error) *error = "Failed to allocate Metal raster occupancy counter.";
        return false;
      }
      clearSharedBuffer(visibleCountBuffer);

      id<MTLCommandBuffer> countCommand = [ctx.queue commandBuffer];
      if (countCommand == nil) {
        if (error) *error = "Failed to create Metal raster occupancy command buffer.";
        return false;
      }
      id<MTLComputeCommandEncoder> countEncoder = [countCommand computeCommandEncoder];
      if (countEncoder == nil) {
        if (error) *error = "Failed to create Metal raster occupancy encoder.";
        return false;
      }
      [countEncoder setComputePipelineState:ctx.rasterOccupancyCountPipeline];
      [countEncoder setBuffer:sourceBuffer offset:0 atIndex:0];
      [countEncoder setBuffer:sourceBuffer offset:0 atIndex:1];
      [countEncoder setBuffer:occupancyBuffer offset:0 atIndex:2];
      [countEncoder setBuffer:visibleCountBuffer offset:0 atIndex:3];
      [countEncoder setBuffer:uniformBuffer offset:0 atIndex:4];
      NSUInteger width = ctx.rasterOccupancyCountPipeline.maxTotalThreadsPerThreadgroup;
      if (width == 0) width = 64;
      width = std::min<NSUInteger>(width, 256);
      [countEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(std::max(uniforms.basePointCount, 0)), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [countEncoder endEncoding];
      [countCommand commit];
      [countCommand waitUntilCompleted];
      NSError* countError = countCommand.error;
      if (countError != nil) {
        if (error) *error = [[countError localizedDescription] UTF8String];
        return false;
      }

      uint32_t visibleCount = 0;
      std::memcpy(&visibleCount, [visibleCountBuffer contents], sizeof(visibleCount));
      const float meanOccupancy =
          static_cast<float>(visibleCount) / static_cast<float>(kRasterOccupancyBinCount);
      uniforms.occupancyTargetThreshold =
          std::max(0, static_cast<int>(std::ceil(meanOccupancy * 0.72f)));
      std::memcpy([uniformBuffer contents], &uniforms, sizeof(uniforms));
    }

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal raster source command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal raster source encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterSourcePipeline];
    [encoder setBuffer:sourceBuffer offset:0 atIndex:0];
    [encoder setBuffer:sourceBuffer offset:0 atIndex:1];
    [encoder setBuffer:vertBuffer offset:0 atIndex:2];
    [encoder setBuffer:colorBuffer offset:0 atIndex:3];
    [encoder setBuffer:occupancyBuffer offset:0 atIndex:4];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:5];
    NSUInteger width = ctx.rasterSourcePipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 256);
    [encoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    copySharedBuffer<PackedFloat3>(vertBuffer, pointCount, outVerts);
    copySharedBuffer<simd_float4>(colorBuffer, pointCount, outColors);
  }
  return true;
}

bool buildInputSampledMesh(const InputSampleRequest& request,
                           const std::vector<float>& fullVerts,
                           const std::vector<float>& fullColors,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger fullPointCount = static_cast<NSUInteger>(std::max(request.fullPointCount, 0));
  const NSUInteger visiblePointCount = static_cast<NSUInteger>(std::max(request.visiblePointCount, 0));
  if (fullPointCount == 0 || visiblePointCount == 0 ||
      fullVerts.size() < fullPointCount * 3u || fullColors.size() < fullPointCount * 4u) {
    if (error) *error = "Invalid Metal thinning source arrays.";
    return false;
  }
  @autoreleasepool {
    id<MTLBuffer> srcVertBuffer = makeSharedBuffer(reinterpret_cast<const PackedFloat3*>(fullVerts.data()), fullPointCount);
    id<MTLBuffer> srcColorBuffer = makeSharedBuffer(reinterpret_cast<const simd_float4*>(fullColors.data()), fullPointCount);
    id<MTLBuffer> dstVertBuffer = makeEmptySharedBuffer(visiblePointCount * sizeof(PackedFloat3));
    id<MTLBuffer> dstColorBuffer = makeEmptySharedBuffer(visiblePointCount * sizeof(simd_float4));
    InputSampleUniforms uniforms{};
    uniforms.fullPointCount = request.fullPointCount;
    uniforms.visiblePointCount = request.visiblePointCount;
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1);
    if (srcVertBuffer == nil || srcColorBuffer == nil || dstVertBuffer == nil || dstColorBuffer == nil || uniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal thinning buffers.";
      return false;
    }
    if (!runInputSampleCompute(srcVertBuffer, srcColorBuffer, dstVertBuffer, dstColorBuffer, uniformBuffer, visiblePointCount, &localError)) {
      if (error) *error = localError;
      return false;
    }
    copySharedBuffer<PackedFloat3>(dstVertBuffer, visiblePointCount, outVerts);
    copySharedBuffer<simd_float4>(dstColorBuffer, visiblePointCount, outColors);
  }
  return true;
}

bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       std::vector<float>* outDensity,
                       std::string* error) {
  std::string localError;
  if (!outDensity) {
    if (error) *error = "Missing Metal scope density output.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(request.pointCount, 0);
  const int width = std::max(request.width, 1);
  const int height = std::max(request.height, 1);
  const size_t expectedSampleFloats = static_cast<size_t>(pointCount) * 5u;
  if (pointCount == 0 || packedSamples.size() < expectedSampleFloats) {
    if (error) *error = "Invalid Metal scope density sample buffer.";
    return false;
  }
  const size_t binCount = static_cast<size_t>(width) *
                          static_cast<size_t>(height) * 3u;
  if (binCount == 0) {
    if (error) *error = "Invalid Metal scope density dimensions.";
    return false;
  }

  MetalContext& ctx = context();
  @autoreleasepool {
    ScopeDensityUniforms uniforms{};
    uniforms.pointCount = pointCount;
    uniforms.waveform = request.waveform;
    uniforms.scopeMode = request.scopeMode;
    uniforms.width = width;
    uniforms.height = height;
    uniforms.rangeMin = request.rangeMin;
    uniforms.invRange = request.invRange;
    uniforms.excludeOverflow = request.excludeOverflow;

    id<MTLBuffer> sampleBuffer = makeSharedBuffer(packedSamples.data(), expectedSampleFloats);
    id<MTLBuffer> densityBuffer = makeEmptySharedBuffer(static_cast<NSUInteger>(binCount * sizeof(uint32_t)));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    if (sampleBuffer == nil || densityBuffer == nil || uniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal scope density buffers.";
      return false;
    }
    clearSharedBuffer(densityBuffer);
    if (!runComputeBuffers(ctx.scopeDensityPipeline,
                           std::array<id<MTLBuffer>, 3>{sampleBuffer, densityBuffer, uniformBuffer},
                           static_cast<NSUInteger>(pointCount),
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    outDensity->assign(binCount, 0.0f);
    const uint32_t* counts = reinterpret_cast<const uint32_t*>([densityBuffer contents]);
    for (size_t i = 0; i < binCount; ++i) {
      (*outDensity)[i] = static_cast<float>(counts[i]);
    }
  }
  return true;
}

bool buildGlossField(const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     GlossFieldResult* out,
                     std::string* error) {
  std::string localError;
  if (!out) {
    if (error) *error = "Missing Metal gloss-field output.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int gridWidth = std::max(request.gridWidth, 1);
  const int gridHeight = std::max(request.gridHeight, 1);
  const NSUInteger pointCount = static_cast<NSUInteger>(packedPoints.size() / 6u);
  const NSUInteger cellCount = static_cast<NSUInteger>(gridWidth) * static_cast<NSUInteger>(gridHeight);
  if (pointCount == 0u || cellCount == 0u) {
    if (error) *error = "Invalid Metal gloss-field request.";
    return false;
  }

  MetalContext& ctx = context();
  @autoreleasepool {
    GlossFieldAccumulateUniforms accumulateUniforms{};
    accumulateUniforms.pointCount = static_cast<int>(pointCount);
    accumulateUniforms.gridWidth = gridWidth;
    accumulateUniforms.gridHeight = gridHeight;
    accumulateUniforms.showOverflow = request.showOverflow;

    GlossFieldCellUniforms cellUniforms{};
    cellUniforms.cellCount = static_cast<int>(cellCount);
    cellUniforms.gridWidth = gridWidth;
    cellUniforms.gridHeight = gridHeight;
    cellUniforms.neighborhoodChoice = request.neighborhoodChoice;

    id<MTLBuffer> inputBuffer = makeSharedBuffer(packedPoints.data(), packedPoints.size());
    id<MTLBuffer> occupancyCountsBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumRBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumGBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumBBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumYBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMaxBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMinBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumNeutralityBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> occupancyBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanRBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanGBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanBBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierYBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMaxBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMinBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> neutralityBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> occupancyNormBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> tempBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> rawSignalBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> positiveBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> negativeBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> boundaryBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> congruenceBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> confidenceBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> signalBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> reductionBuffer = makeEmptySharedBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> accumulateUniformBuffer = makeSharedBuffer(&accumulateUniforms, 1u);
    id<MTLBuffer> cellUniformBuffer = makeSharedBuffer(&cellUniforms, 1u);
    if (inputBuffer == nil || occupancyCountsBuffer == nil || sumRBuffer == nil || sumGBuffer == nil ||
        sumBBuffer == nil || sumYBuffer == nil || sumMaxBuffer == nil || sumMinBuffer == nil ||
        sumNeutralityBuffer == nil || occupancyBuffer == nil || meanRBuffer == nil || meanGBuffer == nil ||
        meanBBuffer == nil || carrierYBuffer == nil || carrierMaxBuffer == nil || carrierMinBuffer == nil ||
        neutralityBuffer == nil || occupancyNormBuffer == nil || tempBuffer == nil || bodyBuffer == nil ||
        rawSignalBuffer == nil || positiveBuffer == nil || negativeBuffer == nil || boundaryBuffer == nil ||
        congruenceBuffer == nil || confidenceBuffer == nil || signalBuffer == nil || reductionBuffer == nil ||
        accumulateUniformBuffer == nil || cellUniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal gloss-field buffers.";
      return false;
    }

    const auto clearReduction = [&]() {
      clearSharedBuffer(reductionBuffer);
    };
    clearSharedBuffer(occupancyCountsBuffer);
    clearSharedBuffer(sumRBuffer);
    clearSharedBuffer(sumGBuffer);
    clearSharedBuffer(sumBBuffer);
    clearSharedBuffer(sumYBuffer);
    clearSharedBuffer(sumMaxBuffer);
    clearSharedBuffer(sumMinBuffer);
    clearSharedBuffer(sumNeutralityBuffer);
    clearSharedBuffer(occupancyBuffer);
    clearSharedBuffer(meanRBuffer);
    clearSharedBuffer(meanGBuffer);
    clearSharedBuffer(meanBBuffer);
    clearSharedBuffer(carrierYBuffer);
    clearSharedBuffer(carrierMaxBuffer);
    clearSharedBuffer(carrierMinBuffer);
    clearSharedBuffer(neutralityBuffer);
    clearSharedBuffer(occupancyNormBuffer);
    clearSharedBuffer(tempBuffer);
    clearSharedBuffer(bodyBuffer);
    clearSharedBuffer(rawSignalBuffer);
    clearSharedBuffer(positiveBuffer);
    clearSharedBuffer(negativeBuffer);
    clearSharedBuffer(boundaryBuffer);
    clearSharedBuffer(congruenceBuffer);
    clearSharedBuffer(confidenceBuffer);
    clearSharedBuffer(signalBuffer);
    clearReduction();

    if (!runComputeBuffers(ctx.glossFieldAccumulatePipeline,
                           std::array<id<MTLBuffer>, 10>{inputBuffer,
                                                         occupancyCountsBuffer,
                                                         sumRBuffer,
                                                         sumGBuffer,
                                                         sumBBuffer,
                                                         sumYBuffer,
                                                         sumMaxBuffer,
                                                         sumMinBuffer,
                                                         sumNeutralityBuffer,
                                                         accumulateUniformBuffer},
                           pointCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!runComputeBuffers(ctx.glossFieldFinalizePipeline,
                           std::array<id<MTLBuffer>, 17>{occupancyCountsBuffer,
                                                         sumRBuffer,
                                                         sumGBuffer,
                                                         sumBBuffer,
                                                         sumYBuffer,
                                                         sumMaxBuffer,
                                                         sumMinBuffer,
                                                         sumNeutralityBuffer,
                                                         occupancyBuffer,
                                                         meanRBuffer,
                                                         meanGBuffer,
                                                         meanBBuffer,
                                                         carrierYBuffer,
                                                         carrierMaxBuffer,
                                                         carrierMinBuffer,
                                                         neutralityBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyBuffer, occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldBlurPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, tempBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::memcpy([occupancyNormBuffer contents], [tempBuffer contents], static_cast<size_t>(cellCount) * sizeof(float));
    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyNormBuffer, occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    const auto blurInPlace = [&](id<MTLBuffer> buffer) -> bool {
      if (!runComputeBuffers(ctx.glossFieldBlurPipeline,
                             std::array<id<MTLBuffer>, 3>{buffer, tempBuffer, cellUniformBuffer},
                             cellCount,
                             &localError)) {
        return false;
      }
      std::memcpy([buffer contents], [tempBuffer contents], static_cast<size_t>(cellCount) * sizeof(float));
      return true;
    };
    if (!blurInPlace(carrierYBuffer) || !blurInPlace(carrierMaxBuffer) ||
        !blurInPlace(carrierMinBuffer) || !blurInPlace(neutralityBuffer)) {
      if (error) *error = localError;
      return false;
    }

    if (!runComputeBuffers(ctx.glossFieldBodyPipeline,
                           std::array<id<MTLBuffer>, 7>{occupancyBuffer,
                                                        meanRBuffer,
                                                        meanGBuffer,
                                                        meanBBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldRawSignalPipeline,
                           std::array<id<MTLBuffer>, 6>{occupancyBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        rawSignalBuffer,
                                                        reductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::array<uint32_t, 4> reductionValues = {0u, 0u, 0u, 0u};
    std::memcpy(reductionValues.data(), [reductionBuffer contents], sizeof(uint32_t));

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldWeightedSignalPipeline,
                           std::array<id<MTLBuffer>, 11>{occupancyNormBuffer,
                                                         bodyBuffer,
                                                         rawSignalBuffer,
                                                         positiveBuffer,
                                                         negativeBuffer,
                                                         boundaryBuffer,
                                                         congruenceBuffer,
                                                         confidenceBuffer,
                                                         signalBuffer,
                                                         reductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::array<uint32_t, 4> weightedValues = {reductionValues[0], 0u, 0u, 0u};
    std::memcpy(weightedValues.data() + 1u, [reductionBuffer contents], 3u * sizeof(uint32_t));
    std::memcpy([reductionBuffer contents], weightedValues.data(), 4u * sizeof(uint32_t));

    if (!runComputeBuffers(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{bodyBuffer,
                                                        signalBuffer,
                                                        positiveBuffer,
                                                        negativeBuffer,
                                                        boundaryBuffer,
                                                        reductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    out->gridWidth = gridWidth;
    out->gridHeight = gridHeight;
    out->occupancy.assign(cellCount, 0.0f);
    out->meanRgb.assign(cellCount * 3u, 0.0f);
    out->carrierY.assign(cellCount, 0.0f);
    out->carrierMax.assign(cellCount, 0.0f);
    out->carrierMin.assign(cellCount, 0.0f);
    out->neutrality.assign(cellCount, 0.0f);
    out->body.assign(cellCount, 0.0f);
    out->signal.assign(cellCount, 0.0f);
    out->positive.assign(cellCount, 0.0f);
    out->negative.assign(cellCount, 0.0f);
    out->boundary.assign(cellCount, 0.0f);
    out->congruence.assign(cellCount, 0.0f);
    out->confidence.assign(cellCount, 0.0f);
    std::vector<float> meanRHost;
    std::vector<float> meanGHost;
    std::vector<float> meanBHost;
    copySharedBuffer<float>(occupancyBuffer, cellCount, &out->occupancy);
    copySharedBuffer<float>(meanRBuffer, cellCount, &meanRHost);
    copySharedBuffer<float>(meanGBuffer, cellCount, &meanGHost);
    copySharedBuffer<float>(meanBBuffer, cellCount, &meanBHost);
    copySharedBuffer<float>(carrierYBuffer, cellCount, &out->carrierY);
    copySharedBuffer<float>(carrierMaxBuffer, cellCount, &out->carrierMax);
    copySharedBuffer<float>(carrierMinBuffer, cellCount, &out->carrierMin);
    copySharedBuffer<float>(neutralityBuffer, cellCount, &out->neutrality);
    copySharedBuffer<float>(bodyBuffer, cellCount, &out->body);
    copySharedBuffer<float>(signalBuffer, cellCount, &out->signal);
    copySharedBuffer<float>(positiveBuffer, cellCount, &out->positive);
    copySharedBuffer<float>(negativeBuffer, cellCount, &out->negative);
    copySharedBuffer<float>(boundaryBuffer, cellCount, &out->boundary);
    copySharedBuffer<float>(congruenceBuffer, cellCount, &out->congruence);
    copySharedBuffer<float>(confidenceBuffer, cellCount, &out->confidence);
    for (NSUInteger idx = 0; idx < cellCount; ++idx) {
      out->meanRgb[idx * 3u + 0u] = idx < meanRHost.size() ? meanRHost[idx] : 0.0f;
      out->meanRgb[idx * 3u + 1u] = idx < meanGHost.size() ? meanGHost[idx] : 0.0f;
      out->meanRgb[idx * 3u + 2u] = idx < meanBHost.size() ? meanBHost[idx] : 0.0f;
    }
  }
  return true;
}

}  // namespace ChromaspaceMetal
