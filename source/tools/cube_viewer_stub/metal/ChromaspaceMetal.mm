#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#import <OpenGL/gl.h>
#import <OpenGL/OpenGL.h>
#import <simd/simd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "ChromaspaceMetal.h"

#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif

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
  id<MTLComputePipelineState> rasterSourceTexturePipeline = nil;
  id<MTLComputePipelineState> rasterOccupancyTextureCountPipeline = nil;
  id<MTLComputePipelineState> inputSamplePipeline = nil;
  id<MTLComputePipelineState> scopeDensityPipeline = nil;
  id<MTLComputePipelineState> rasterScopeDensityTexturePipeline = nil;
  id<MTLComputePipelineState> rasterScopeRangeTexturePipeline = nil;
  id<MTLComputePipelineState> rasterScopeRangeHistogramTexturePipeline = nil;
  id<MTLComputePipelineState> scopeRangeHistogramPercentilePipeline = nil;
  id<MTLComputePipelineState> scopeRangeFinalizePipeline = nil;
  id<MTLComputePipelineState> histogramMaxPipeline = nil;
  id<MTLComputePipelineState> histogramSurfaceRenderPipeline = nil;
  id<MTLComputePipelineState> glossFieldAccumulatePipeline = nil;
  id<MTLComputePipelineState> rasterGlossFieldAccumulateTexturePipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldMaxPipeline = nil;
  id<MTLComputePipelineState> glossFieldNormalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldBlurPipeline = nil;
  id<MTLComputePipelineState> glossFieldBodyPipeline = nil;
  id<MTLComputePipelineState> glossFieldRawSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldWeightedSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldMergeMaxBitsPipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalNormalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldLocalPercentilePipeline = nil;
  id<MTLComputePipelineState> glossFieldCandidate2RawPipeline = nil;
  id<MTLComputePipelineState> glossFieldAssembleUnifiedPipeline = nil;
  id<MTLComputePipelineState> glossFieldSurfaceRenderPipeline = nil;
  id<MTLComputePipelineState> glossProjectionSurfaceSelectPipeline = nil;
  id<MTLComputePipelineState> glossProjectionSurfaceShadePipeline = nil;
  id<MTLComputePipelineState> plotSurfaceClearPipeline = nil;
  std::string deviceName;
  std::string initError;
  bool initAttempted = false;
  bool ready = false;
};

struct PlotSurfaceRecord {
  IOSurfaceRef surface = nullptr;
  id<MTLTexture> texture = nil;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;
  size_t byteSize = 0;
};

struct GlossFieldResidentRecord {
  int gridWidth = 0;
  int gridHeight = 0;
  uint64_t builtSerial = 0;
  id<MTLBuffer> occupancy = nil;
  id<MTLBuffer> meanR = nil;
  id<MTLBuffer> meanG = nil;
  id<MTLBuffer> meanB = nil;
  id<MTLBuffer> carrierY = nil;
  id<MTLBuffer> carrierMax = nil;
  id<MTLBuffer> carrierMin = nil;
  id<MTLBuffer> neutrality = nil;
  id<MTLBuffer> occupancyNorm = nil;
  id<MTLBuffer> body = nil;
  id<MTLBuffer> rawSignal = nil;
  id<MTLBuffer> positive = nil;
  id<MTLBuffer> negative = nil;
  id<MTLBuffer> boundary = nil;
  id<MTLBuffer> congruence = nil;
  id<MTLBuffer> confidence = nil;
  id<MTLBuffer> signal = nil;
  id<MTLBuffer> body2 = nil;
  id<MTLBuffer> positive2 = nil;
  id<MTLBuffer> negative2 = nil;
  id<MTLBuffer> boundary2 = nil;
  id<MTLBuffer> congruence2 = nil;
  id<MTLBuffer> confidence2 = nil;
  id<MTLBuffer> signal2 = nil;
  id<MTLBuffer> temp = nil;
  id<MTLBuffer> reduction = nil;
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
  int onlyOverflow;
  int channelCount;
  int lumaMethod;
};

struct ScopeRangeUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  int includeOverflow;
  int lumaMethod;
  int previousRangeValid;
  float previousRangeMin;
  float previousRangeMax;
  int histogramBinCount;
};

struct HistogramSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
};

struct PlotSurfaceClearUniforms {
  float r;
  float g;
  float b;
  float a;
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

struct GlossFieldSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
};

struct GlossProjectionSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float sourceAspect;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
  float pointRadiusPixels;
  float modelView[16];
  float projection[16];
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

MTLPixelFormat sourceSignalMetalPixelFormat(int pixelFormat) {
  return pixelFormat == 1 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float;
}

OSType sourceSignalIOSurfacePixelFormat(int pixelFormat) {
  return pixelFormat == 1 ? static_cast<OSType>('RGBA') : static_cast<OSType>('RGhA');
}

size_t sourceSignalBytesPerElement(int pixelFormat) {
  return pixelFormat == 1 ? 16u : 8u;
}

std::mutex& plotSurfaceMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint32_t, PlotSurfaceRecord>& plotSurfaceRegistry() {
  static std::unordered_map<uint32_t, PlotSurfaceRecord> registry;
  return registry;
}

std::mutex& glossFieldRegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, GlossFieldResidentRecord>& glossFieldRegistry() {
  static std::unordered_map<uint64_t, GlossFieldResidentRecord> registry;
  return registry;
}

uint64_t nextGlossFieldCacheId() {
  static uint64_t nextId = 1;
  return nextId++;
}

id<MTLTexture> makeTextureFromIOSurface(MetalContext& ctx,
                                        uint32_t surfaceId,
                                        int width,
                                        int height,
                                        int pixelFormat,
                                        std::string* error) {
  if (surfaceId == 0 || width <= 0 || height <= 0) {
    if (error) *error = "invalid-iosurface-texture-request";
    return nil;
  }
  IOSurfaceRef surface = IOSurfaceLookup(static_cast<IOSurfaceID>(surfaceId));
  if (surface == nullptr) {
    if (error) *error = "iosurface-lookup-failed";
    return nil;
  }
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sourceSignalMetalPixelFormat(pixelFormat)
                                                         width:static_cast<NSUInteger>(width)
                                                        height:static_cast<NSUInteger>(height)
                                                     mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [ctx.device newTextureWithDescriptor:desc iosurface:surface plane:0];
  CFRelease(surface);
  if (texture == nil && error) *error = "iosurface-metal-texture-failed";
  return texture;
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
  int onlyOverflow;
  int channelCount;
  int lumaMethod;
};

struct ScopeRangeUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  int includeOverflow;
  int lumaMethod;
  int previousRangeValid;
  float previousRangeMin;
  float previousRangeMax;
  int histogramBinCount;
};

struct HistogramSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
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

float3 mapChromaticityPosition(float r, float g, float b, constant OverlayUniforms& u) {
  float3 linear = float3(decodeTransferChannel(r, u.chromaticityInputTransfer),
                         decodeTransferChannel(g, u.chromaticityInputTransfer),
                         decodeTransferChannel(b, u.chromaticityInputTransfer));
  linear = clamp(linear, 0.0, 1.0);
  float3 xyz = mulRows(u.chromaticityRgbToXyz, linear);
  float2 white = float2(u.chromaticityWhiteX, u.chromaticityWhiteY);
  float3 xyY = xyzToXyY(xyz, white);
  float2 xy = xyY.xy;
  if (u.chromaticityReferenceBasis != 0) {
    float3 basisXyz = xyToXyz(xy, 1.0);
    float3 rgb = mulRows(u.chromaticityXyzToRgb, basisXyz);
    xy = xyzToXyY(rgb, float2(1.0 / 3.0, 1.0 / 3.0)).xy;
  }
  float viewerHeight = clamp(xyY.z, 0.0, 1.0) * 2.0 - 1.0;
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

void rasterReadTransformedSampleTexture(texture2d<float, access::read> sourceTexture,
                                        constant RasterSourceUniforms& u,
                                        int x,
                                        int y,
                                        thread float& r,
                                        thread float& g,
                                        thread float& b) {
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  float4 v = sourceTexture.read(uint2(uint(x), uint(y)));
  r = v.x;
  g = v.y;
  b = v.z;
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

kernel void rasterOccupancyTextureCountKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                              device atomic_uint* occupancyBins [[buffer(0)]],
                                              device atomic_uint* visibleCount [[buffer(1)]],
                                              constant RasterSourceUniforms& u [[buffer(2)]],
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
  rasterReadTransformedSampleTexture(sourceTexture, u, x, y, r, g, b);
  if (!rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b)) return;
  atomic_fetch_add_explicit(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], 1u, memory_order_relaxed);
  atomic_fetch_add_explicit(&visibleCount[0], 1u, memory_order_relaxed);
}

kernel void rasterSourceTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                      device packed_float3* vertVals [[buffer(0)]],
                                      device float4* colorVals [[buffer(1)]],
                                      device atomic_uint* occupancyBins [[buffer(2)]],
                                      constant RasterSourceUniforms& u [[buffer(3)]],
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
  rasterReadTransformedSampleTexture(sourceTexture, u, x, y, r, g, b);
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

  float3 pos = u.plotMode == 8
                   ? mapChromaticityPosition(r, g, b, u)
                   : mapPlotPosition(r, g, b, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, 0);
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
  int channelCount = max(u.channelCount, 1);
  if (channel < 0 || channel >= channelCount) return;
  bool overflow = (value < 0.0 || value > 1.0);
  if (u.onlyOverflow != 0 && !overflow) return;
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

float scopeLuma(float r, float g, float b, int method) {
  switch (method) {
    case 1:
      return 0.2627 * r + 0.6780 * g + 0.0593 * b;
    case 2:
      return 0.2990 * r + 0.5870 * g + 0.1140 * b;
    case 3:
      return (r + g + b) / 3.0;
    default:
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
}

uint orderedUintFromFloat(float value) {
  uint bits = as_type<uint>(value);
  return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
}

float floatFromOrderedUint(uint value) {
  uint bits = (value & 0x80000000u) != 0u ? (value ^ 0x80000000u) : ~value;
  return as_type<float>(bits);
}

void includeScopeRangeValue(float value,
                            constant ScopeRangeUniforms& range,
                            device atomic_uint* rangeBits) {
  if (rangeBits == nullptr) return;
  if (range.includeOverflow == 0 && (value < 0.0 || value > 1.0)) return;
  uint ordered = orderedUintFromFloat(value);
  atomic_fetch_min_explicit(&rangeBits[0], ordered, memory_order_relaxed);
  atomic_fetch_max_explicit(&rangeBits[1], ordered, memory_order_relaxed);
  atomic_fetch_add_explicit(&rangeBits[2], 1u, memory_order_relaxed);
}

void includeScopeRangeValues(float r,
                             float g,
                             float b,
                             constant ScopeRangeUniforms& range,
                             device atomic_uint* rangeBits) {
  bool lumaOnly = (range.waveform != 0 && range.scopeMode == 2) ||
                  (range.waveform == 0 && range.scopeMode == 1);
  if (lumaOnly) {
    includeScopeRangeValue(scopeLuma(r, g, b, range.lumaMethod), range, rangeBits);
    return;
  }
  if (range.includeRed != 0) includeScopeRangeValue(r, range, rangeBits);
  if (range.includeGreen != 0) includeScopeRangeValue(g, range, rangeBits);
  if (range.includeBlue != 0) includeScopeRangeValue(b, range, rangeBits);
  if (range.waveform != 0 && range.scopeMode == 1 && range.includeLuma != 0) {
    includeScopeRangeValue(scopeLuma(r, g, b, range.lumaMethod), range, rangeBits);
  }
}

void includeScopeRangeHistogramValue(float value,
                                     constant ScopeRangeUniforms& range,
                                     float minValue,
                                     float invRange,
                                     device atomic_uint* histogram) {
  if (histogram == nullptr || range.histogramBinCount <= 0) return;
  if (range.includeOverflow == 0 && (value < 0.0 || value > 1.0)) return;
  int bin = clamp(int(floor((value - minValue) * invRange * float(range.histogramBinCount))),
                  0,
                  max(range.histogramBinCount - 1, 0));
  atomic_fetch_add_explicit(&histogram[uint(bin)], 1u, memory_order_relaxed);
}

void includeScopeRangeHistogramValues(float r,
                                      float g,
                                      float b,
                                      constant ScopeRangeUniforms& range,
                                      float minValue,
                                      float invRange,
                                      device atomic_uint* histogram) {
  bool lumaOnly = (range.waveform != 0 && range.scopeMode == 2) ||
                  (range.waveform == 0 && range.scopeMode == 1);
  if (lumaOnly) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogram);
    return;
  }
  if (range.includeRed != 0) {
    includeScopeRangeHistogramValue(r, range, minValue, invRange, histogram);
  }
  if (range.includeGreen != 0) {
    includeScopeRangeHistogramValue(g, range, minValue, invRange, histogram);
  }
  if (range.includeBlue != 0) {
    includeScopeRangeHistogramValue(b, range, minValue, invRange, histogram);
  }
  if (range.waveform != 0 && range.scopeMode == 1 && range.includeLuma != 0) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogram);
  }
}

bool rasterScopeSampleFromTexture(texture2d<float, access::read> sourceTexture,
                                  constant RasterSourceUniforms& raster,
                                  uint index,
                                  thread float* outXNorm,
                                  thread float* outYNorm,
                                  thread float* outR,
                                  thread float* outG,
                                  thread float* outB) {
  int sampleCountX = max(raster.sampleCountX, 1);
  int stride = max(raster.sampleStride, 1);
  int x = int(index % uint(sampleCountX)) * stride;
  int y = int(index / uint(sampleCountX)) * stride;
  bool haveCoords = index < uint(max(raster.basePointCount, 0));
  if (!haveCoords) {
    haveCoords = rasterAppendSampleCoords(index,
                                          raster.identityCubeAppendOffset,
                                          raster.identityCubeAppendCount,
                                          raster.identityCubeAppendY1,
                                          raster.identityCubeAppendY2,
                                          raster.identityCubeAppendRowStep,
                                          raster.identityCubeAppendXStep,
                                          raster.sourceWidth,
                                          &x,
                                          &y);
    if (!haveCoords) {
      haveCoords = rasterAppendSampleCoords(index,
                                            raster.identityRampAppendOffset,
                                            raster.identityRampAppendCount,
                                            raster.identityRampAppendY1,
                                            raster.identityRampAppendY2,
                                            raster.identityRampAppendRowStep,
                                            raster.identityRampAppendXStep,
                                            raster.sourceWidth,
                                            &x,
                                            &y);
    }
  }
  if (!haveCoords) return false;
  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  float xNorm = (float(x) + 0.5) / float(max(raster.sourceWidth, 1));
  float yNorm = (float(y) + 0.5) / float(max(raster.sourceHeight, 1));
  if (outXNorm != nullptr) *outXNorm = xNorm;
  if (outYNorm != nullptr) *outYNorm = yNorm;
  rasterReadTransformedSampleTexture(sourceTexture, raster, x, y, *outR, *outG, *outB);
  return rasterSampleVisible(raster, x, y, xNorm, yNorm, *outR, *outG, *outB);
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
    accumulateScopeDensity(density, u, 0, xNorm, scopeLuma(r, g, b, u.lumaMethod));
  } else {
    accumulateScopeDensity(density, u, 0, xNorm, r);
    accumulateScopeDensity(density, u, 1, xNorm, g);
    accumulateScopeDensity(density, u, 2, xNorm, b);
    if (u.waveform != 0 && u.scopeMode == 1 && u.channelCount >= 4) {
      accumulateScopeDensity(density, u, 3, xNorm, scopeLuma(r, g, b, u.lumaMethod));
    }
  }
}

kernel void rasterScopeDensityTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                            device atomic_uint* density [[buffer(0)]],
                                            constant RasterSourceUniforms& raster [[buffer(1)]],
                                            constant ScopeDensityUniforms& scope [[buffer(2)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || scope.width <= 0 || scope.height <= 0 || density == nullptr) return;
  float xNorm = 0.5;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, &xNorm, nullptr, &r, &g, &b)) return;

  bool lumaOnly = (scope.waveform != 0 && scope.scopeMode == 2) ||
                  (scope.waveform == 0 && scope.scopeMode == 1);
  if (lumaOnly) {
    accumulateScopeDensity(density, scope, 0, xNorm, scopeLuma(r, g, b, scope.lumaMethod));
  } else {
    accumulateScopeDensity(density, scope, 0, xNorm, r);
    accumulateScopeDensity(density, scope, 1, xNorm, g);
    accumulateScopeDensity(density, scope, 2, xNorm, b);
    if (scope.waveform != 0 && scope.scopeMode == 1 && scope.channelCount >= 4) {
      accumulateScopeDensity(density, scope, 3, xNorm, scopeLuma(r, g, b, scope.lumaMethod));
    }
  }
}

kernel void rasterScopeRangeTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                          device atomic_uint* rangeBits [[buffer(0)]],
                                          constant RasterSourceUniforms& raster [[buffer(1)]],
                                          constant ScopeRangeUniforms& range [[buffer(2)]],
                                          uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || rangeBits == nullptr) return;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, nullptr, nullptr, &r, &g, &b)) return;
  includeScopeRangeValues(r, g, b, range, rangeBits);
}

kernel void rasterScopeRangeHistogramTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                                   device atomic_uint* histogram [[buffer(0)]],
                                                   const device atomic_uint* rangeBits [[buffer(1)]],
                                                   constant RasterSourceUniforms& raster [[buffer(2)]],
                                                   constant ScopeRangeUniforms& range [[buffer(3)]],
                                                   uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total || histogram == nullptr || rangeBits == nullptr) return;
  uint validCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (validCount == 0u) return;
  float minValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[0], memory_order_relaxed));
  float maxValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[1], memory_order_relaxed));
  float invRange = 1.0 / max(1.0e-7, maxValue - minValue);
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, nullptr, nullptr, &r, &g, &b)) return;
  includeScopeRangeHistogramValues(r, g, b, range, minValue, invRange, histogram);
}

kernel void rasterGlossFieldAccumulateTextureKernel(texture2d<float, access::read> sourceTexture [[texture(0)]],
                                                    device atomic_uint* occupancyCounts [[buffer(0)]],
                                                    device atomic_uint* sumR [[buffer(1)]],
                                                    device atomic_uint* sumG [[buffer(2)]],
                                                    device atomic_uint* sumB [[buffer(3)]],
                                                    device atomic_uint* sumY [[buffer(4)]],
                                                    device atomic_uint* sumMax [[buffer(5)]],
                                                    device atomic_uint* sumMin [[buffer(6)]],
                                                    device atomic_uint* sumNeutrality [[buffer(7)]],
                                                    constant RasterSourceUniforms& raster [[buffer(8)]],
                                                    constant GlossFieldAccumulateUniforms& u [[buffer(9)]],
                                                    uint index [[thread_position_in_grid]]) {
  uint total = uint(max(raster.input.pointCount, 0));
  if (index >= total) return;
  float xNorm = 0.5;
  float yNorm = 0.5;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;
  if (!rasterScopeSampleFromTexture(sourceTexture, raster, index, &xNorm, &yNorm, &r, &g, &b)) return;
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

kernel void scopeRangeHistogramPercentileKernel(const device atomic_uint* histogram [[buffer(0)]],
                                                const device atomic_uint* rangeBits [[buffer(1)]],
                                                device atomic_uint* percentileBits [[buffer(2)]],
                                                constant ScopeRangeUniforms& range [[buffer(3)]],
                                                uint index [[thread_position_in_grid]]) {
  if (index != 0u || histogram == nullptr || rangeBits == nullptr ||
      percentileBits == nullptr || range.histogramBinCount <= 0) return;
  uint totalCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (totalCount == 0u) {
    atomic_store_explicit(&percentileBits[0], orderedUintFromFloat(0.0), memory_order_relaxed);
    atomic_store_explicit(&percentileBits[1], orderedUintFromFloat(1.0), memory_order_relaxed);
    return;
  }
  float minValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[0], memory_order_relaxed));
  float maxValue = floatFromOrderedUint(atomic_load_explicit(&rangeBits[1], memory_order_relaxed));
  ulong lowTarget = ulong(totalCount) / 1000ul;
  ulong highTarget = (ulong(totalCount) * 999ul) / 1000ul;
  ulong accumulated = 0ul;
  float lowValue = minValue;
  float highValue = maxValue;
  bool foundLow = false;
  bool foundHigh = false;
  float span = maxValue - minValue;
  for (int bin = 0; bin < range.histogramBinCount; ++bin) {
    accumulated += ulong(atomic_load_explicit(&histogram[uint(bin)], memory_order_relaxed));
    if (!foundLow && accumulated > lowTarget) {
      float t = (float(bin) + 0.5) / float(range.histogramBinCount);
      lowValue = minValue + t * span;
      foundLow = true;
    }
    if (!foundHigh && accumulated > highTarget) {
      float t = (float(bin) + 0.5) / float(range.histogramBinCount);
      highValue = minValue + t * span;
      foundHigh = true;
      break;
    }
  }
  atomic_store_explicit(&percentileBits[0], orderedUintFromFloat(lowValue), memory_order_relaxed);
  atomic_store_explicit(&percentileBits[1], orderedUintFromFloat(highValue), memory_order_relaxed);
}

kernel void scopeRangeFinalizeKernel(const device atomic_uint* percentileBits [[buffer(0)]],
                                     const device atomic_uint* rangeBits [[buffer(1)]],
                                     device atomic_uint* finalRangeBits [[buffer(2)]],
                                     constant ScopeRangeUniforms& range [[buffer(3)]],
                                     uint index [[thread_position_in_grid]]) {
  if (index != 0u || percentileBits == nullptr || rangeBits == nullptr || finalRangeBits == nullptr) return;
  uint validCount = atomic_load_explicit(&rangeBits[2], memory_order_relaxed);
  if (validCount == 0u) {
    atomic_store_explicit(&finalRangeBits[0], orderedUintFromFloat(0.0), memory_order_relaxed);
    atomic_store_explicit(&finalRangeBits[1], orderedUintFromFloat(1.0), memory_order_relaxed);
    atomic_store_explicit(&finalRangeBits[2], 0u, memory_order_relaxed);
    return;
  }
  float rangeMin = min(0.0, floatFromOrderedUint(atomic_load_explicit(&percentileBits[0], memory_order_relaxed)));
  float rangeMax = max(1.0, floatFromOrderedUint(atomic_load_explicit(&percentileBits[1], memory_order_relaxed)));
  float pad = max(0.02, (rangeMax - rangeMin) * 0.04);
  rangeMin -= pad;
  rangeMax += pad;
  if (!(rangeMax > rangeMin + 1.0e-5)) {
    rangeMin = 0.0;
    rangeMax = 1.0;
  }
  if (range.previousRangeValid != 0 && range.previousRangeMax > range.previousRangeMin + 1.0e-5) {
    rangeMin = rangeMin < range.previousRangeMin
                   ? rangeMin
                   : range.previousRangeMin + (rangeMin - range.previousRangeMin) * 0.16;
    rangeMax = rangeMax > range.previousRangeMax
                   ? rangeMax
                   : range.previousRangeMax + (rangeMax - range.previousRangeMax) * 0.16;
  }
  atomic_store_explicit(&finalRangeBits[0], orderedUintFromFloat(rangeMin), memory_order_relaxed);
  atomic_store_explicit(&finalRangeBits[1], orderedUintFromFloat(rangeMax), memory_order_relaxed);
  atomic_store_explicit(&finalRangeBits[2], validCount, memory_order_relaxed);
}

uint histogramDensityAt(const device atomic_uint* density, int channel, int bin, int width) {
  if (density == nullptr || channel < 0 || bin < 0 || width <= 0) return 0u;
  return atomic_load_explicit(&density[uint(channel * width + bin)], memory_order_relaxed);
}

float smoothedHistogramDensity(const device atomic_uint* density, int channel, int bin, int width) {
  constexpr float weights[5] = {1.0, 4.0, 6.0, 4.0, 1.0};
  float sum = 0.0;
  float weightSum = 0.0;
  for (int tap = -2; tap <= 2; ++tap) {
    int sourceBin = clamp(bin + tap, 0, max(width - 1, 0));
    float weight = weights[tap + 2];
    sum += float(histogramDensityAt(density, channel, sourceBin, width)) * weight;
    weightSum += weight;
  }
  return weightSum > 0.0 ? sum / weightSum : 0.0;
}

kernel void histogramSurfaceMaxKernel(const device atomic_uint* density [[buffer(0)]],
                                      device atomic_uint* maxDensity [[buffer(1)]],
                                      constant HistogramSurfaceUniforms& u [[buffer(2)]],
                                      uint index [[thread_position_in_grid]]) {
  int width = max(u.width, 1);
  int channelCount = max(u.channelCount, 1);
  uint total = uint(width * channelCount);
  if (index >= total || density == nullptr || maxDensity == nullptr) return;
  int channel = int(index / uint(width));
  int bin = int(index % uint(width));
  uint smoothed = uint(round(smoothedHistogramDensity(density, channel, bin, width)));
  atomic_fetch_max_explicit(&maxDensity[0], smoothed, memory_order_relaxed);
}

float4 overColor(float4 dst, float4 src) {
  float invA = 1.0 - clamp(src.a, 0.0, 1.0);
  return float4(src.rgb * src.a + dst.rgb * invA, src.a + dst.a * invA);
}

float4 histogramChannelColor(int channel, bool lumaOnly, bool overflow, bool highlightOverflow) {
  if (overflow) {
    if (highlightOverflow) return float4(0.78, 0.30, 1.0, 0.94);
    if (lumaOnly) return float4(0.74, 0.82, 0.90, 0.72);
    if (channel == 0) return float4(1.0, 0.24, 0.18, 0.76);
    if (channel == 1) return float4(0.26, 1.0, 0.36, 0.76);
    return float4(0.32, 0.58, 1.0, 0.76);
  }
  if (lumaOnly) return float4(0.88, 0.92, 0.96, 0.88);
  if (channel == 0) return float4(1.0, 0.16, 0.12, 0.76);
  if (channel == 1) return float4(0.20, 1.0, 0.28, 0.76);
  return float4(0.24, 0.52, 1.0, 0.76);
}

kernel void histogramSurfaceRenderKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                         const device atomic_uint* density [[buffer(0)]],
                                         const device atomic_uint* overflowDensity [[buffer(1)]],
                                         const device atomic_uint* maxDensity [[buffer(2)]],
                                         constant HistogramSurfaceUniforms& u [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]]) {
  uint outWidth = outTexture.get_width();
  uint outHeight = outTexture.get_height();
  if (gid.x >= outWidth || gid.y >= outHeight) return;
  float2 uv = float2((float(gid.x) + 0.5) / float(max(outWidth, 1u)),
                     (float(gid.y) + 0.5) / float(max(outHeight, 1u)));
  int width = max(u.width, 1);
  int bin = clamp(int(floor(uv.x * float(width))), 0, width - 1);
  int channelCount = max(u.channelCount, 1);
  bool lumaOnly = u.scopeMode == 1 || channelCount == 1;
  uint maxValue = maxDensity == nullptr ? 1u : atomic_load_explicit(&maxDensity[0], memory_order_relaxed);
  float invMax = 1.0 / max(1.0, float(maxValue));
  float4 color = float4(0.0);
  float lineThickness = max(1.5 / float(max(outHeight, 1u)), 0.003);
  for (int channel = 0; channel < channelCount; ++channel) {
    float densityValue = smoothedHistogramDensity(density, channel, bin, width);
    if (densityValue > 0.0) {
      float curve = sqrt(clamp(densityValue * invMax, 0.0, 1.0)) * 0.965;
      float fillAlpha = uv.y <= curve ? (lumaOnly ? 0.14 : 0.10) : 0.0;
      float lineAlpha = 1.0 - smoothstep(lineThickness, lineThickness * 2.5, abs(uv.y - curve));
      float4 base = histogramChannelColor(channel, lumaOnly, false, false);
      if (fillAlpha > 0.0) color = overColor(color, float4(base.rgb, fillAlpha));
      if (lineAlpha > 0.0) color = overColor(color, float4(base.rgb, base.a * lineAlpha));
    }
    if (u.showOverflow != 0 && overflowDensity != nullptr) {
      float overflowValue = smoothedHistogramDensity(overflowDensity, channel, bin, width);
      if (overflowValue > 0.0) {
        float curve = sqrt(clamp(overflowValue * invMax, 0.0, 1.0)) * 0.965;
        float fillAlpha = uv.y <= curve ? 0.16 : 0.0;
        float lineAlpha = 1.0 - smoothstep(lineThickness, lineThickness * 2.5, abs(uv.y - curve));
        float4 base = histogramChannelColor(channel, lumaOnly, true, u.highlightOverflow != 0);
        if (fillAlpha > 0.0) color = overColor(color, float4(base.rgb, fillAlpha));
        if (lineAlpha > 0.0) color = overColor(color, float4(base.rgb, base.a * lineAlpha));
      }
    }
  }
  outTexture.write(color, gid);
}

int glossNeighborhoodRadiusCells(int neighborhoodChoice) {
  switch (clamp(neighborhoodChoice, 0, 2)) {
    case 0: return 1;
    case 2: return 3;
    case 1:
    default: return 2;
  }
}

int glossAnalysisRadiusCells(int neighborhoodChoice) {
  return max(2, glossNeighborhoodRadiusCells(neighborhoodChoice) * 2);
}

float sampleGridClamped(const device float* values, int width, int height, int x, int y) {
  if (values == nullptr || width <= 0 || height <= 0) return 0.0;
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  return values[uint(y * width + x)];
}

float glossGradientMagnitude(const device float* values, int width, int height, int x, int y) {
  float gx = 0.5 * (sampleGridClamped(values, width, height, x + 1, y) -
                    sampleGridClamped(values, width, height, x - 1, y));
  float gy = 0.5 * (sampleGridClamped(values, width, height, x, y + 1) -
                    sampleGridClamped(values, width, height, x, y - 1));
  return sqrt(gx * gx + gy * gy);
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
      float tapWeight = (ox == 0 && oy == 0) ? 0.30 : ((ox == 0 || oy == 0) ? 0.13 : 0.08);
      accum += src[uint(yy * u.gridWidth + xx)] * tapWeight;
      weight += tapWeight;
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

kernel void glossFieldMergeMaxBitsKernel(const device atomic_uint* bodyMaxBits [[buffer(0)]],
                                         const device atomic_uint* weightedMaxBits [[buffer(1)]],
                                         device atomic_uint* finalMaxBits [[buffer(2)]],
                                         uint index [[thread_position_in_grid]]) {
  if (index != 0u) return;
  atomic_store_explicit(&finalMaxBits[0],
                        atomic_load_explicit(&bodyMaxBits[0], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[1],
                        atomic_load_explicit(&weightedMaxBits[0], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[2],
                        atomic_load_explicit(&weightedMaxBits[1], memory_order_relaxed),
                        memory_order_relaxed);
  atomic_store_explicit(&finalMaxBits[3],
                        atomic_load_explicit(&weightedMaxBits[2], memory_order_relaxed),
                        memory_order_relaxed);
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

kernel void glossFieldLocalPercentileKernel(const device float* values [[buffer(0)]],
                                            const device float* occupancy [[buffer(1)]],
                                            device float* outValues [[buffer(2)]],
                                            constant GlossFieldCellUniforms& u [[buffer(3)]],
                                            constant float& percentile [[buffer(4)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  if (occupancy != nullptr && occupancy[index] <= 0.5) {
    outValues[index] = 0.0;
    return;
  }
  int radius = max(1, glossAnalysisRadiusCells(u.neighborhoodChoice));
  int bins[96];
  for (int i = 0; i < 96; ++i) bins[i] = 0;
  int count = 0;
  float minValue = 1.0e20;
  float maxValue = -1.0e20;
  for (int yy = max(0, y - radius); yy <= min(u.gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(u.gridWidth - 1, x + radius); ++xx) {
      uint nidx = uint(yy * u.gridWidth + xx);
      if (occupancy != nullptr && occupancy[nidx] <= 0.5) continue;
      float v = values[nidx];
      minValue = min(minValue, v);
      maxValue = max(maxValue, v);
      ++count;
    }
  }
  if (count <= 0 || maxValue - minValue <= 1e-7) {
    outValues[index] = values[index];
    return;
  }
  float invRange = 1.0 / (maxValue - minValue);
  for (int yy = max(0, y - radius); yy <= min(u.gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(u.gridWidth - 1, x + radius); ++xx) {
      uint nidx = uint(yy * u.gridWidth + xx);
      if (occupancy != nullptr && occupancy[nidx] <= 0.5) continue;
      int bin = min(95,
                    max(0, int((values[nidx] - minValue) * invRange *
                               95.0 + 0.5)));
      bins[bin] += 1;
    }
  }
  int target = min(count - 1,
                   max(0, int(floor(clamp(percentile / 100.0, 0.0, 1.0) *
                                    float(count - 1) + 0.5))));
  int accumulated = 0;
  int chosen = 0;
  for (int i = 0; i < 96; ++i) {
    accumulated += bins[i];
    if (accumulated > target) {
      chosen = i;
      break;
    }
  }
  outValues[index] = minValue + (float(chosen) / 95.0) * (maxValue - minValue);
}

kernel void glossFieldCandidate2RawKernel(const device float* occupancy [[buffer(0)]],
                                          const device float* occupancySupport [[buffer(1)]],
                                          const device float* meanR [[buffer(2)]],
                                          const device float* meanG [[buffer(3)]],
                                          const device float* meanB [[buffer(4)]],
                                          const device float* carrier [[buffer(5)]],
                                          const device float* viewerBody [[buffer(6)]],
                                          const device float* bodyCore [[buffer(7)]],
                                          const device float* bodyContext [[buffer(8)]],
                                          const device float* retinexBody [[buffer(9)]],
                                          const device float* dogLow [[buffer(10)]],
                                          const device float* dogHigh [[buffer(11)]],
                                          device float* adaptiveBody [[buffer(12)]],
                                          device float* positiveRaw [[buffer(13)]],
                                          device float* negativeRaw [[buffer(14)]],
                                          device float* confidence [[buffer(15)]],
                                          device float* agreement [[buffer(16)]],
                                          constant GlossFieldCellUniforms& u [[buffer(17)]],
                                          uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  if (occupancy[index] <= 0.5) {
    adaptiveBody[index] = 0.0;
    positiveRaw[index] = 0.0;
    negativeRaw[index] = 0.0;
    confidence[index] = 0.0;
    agreement[index] = 0.0;
    return;
  }
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  int analysisRadius = glossAnalysisRadiusCells(u.neighborhoodChoice);
  float hybridBody = 0.65 * bodyCore[index] + 0.35 * bodyContext[index];
  float viewerPositive = max(0.0, carrier[index] - viewerBody[index]);
  float viewerNegative = max(0.0, viewerBody[index] - carrier[index]);
  float hybridPositive = max(0.0, carrier[index] - bodyCore[index]);
  float hybridNegative = max(0.0, bodyContext[index] - carrier[index]);
  float chromaSpread = max(meanR[index], max(meanG[index], meanB[index])) -
                       min(meanR[index], min(meanG[index], meanB[index]));
  float shapeSupport = clamp(glossGradientMagnitude(bodyContext, u.gridWidth, u.gridHeight, x, y) * 8.0, 0.0, 1.0);
  float chromaSupport = clamp(chromaSpread * 2.5, 0.0, 1.0);
  float ambiguity = clamp(1.0 - (0.72 * shapeSupport + 0.28 * chromaSupport), 0.0, 1.0);
  float body = ambiguity * viewerBody[index] + (1.0 - ambiguity) * hybridBody;
  float bodyAgreement = clamp(1.0 - abs(viewerBody[index] - hybridBody), 0.0, 1.0);
  float positiveAgreement = clamp(1.0 - abs(viewerPositive - hybridPositive) * 4.0, 0.0, 1.0);
  float localPositiveSupport = 0.0;
  float supportWeight = 0.0;
  for (int oy = -analysisRadius; oy <= analysisRadius; ++oy) {
    for (int ox = -analysisRadius; ox <= analysisRadius; ++ox) {
      int xx = clamp(x + ox, 0, u.gridWidth - 1);
      int yy = clamp(y + oy, 0, u.gridHeight - 1);
      uint nidx = uint(yy * u.gridWidth + xx);
      float distSq = float(ox * ox + oy * oy);
      float w = 1.0 / (1.0 + distSq);
      localPositiveSupport += max(0.0, carrier[nidx] - bodyCore[nidx]) * w;
      supportWeight += w;
    }
  }
  localPositiveSupport = clamp((supportWeight > 1e-6 ? localPositiveSupport / supportWeight : 0.0) * 4.0, 0.0, 1.0);
  float permission = clamp(0.32 * positiveAgreement +
                           0.24 * bodyAgreement +
                           0.24 * shapeSupport +
                           0.20 * localPositiveSupport,
                           0.0,
                           1.0);
  float consensusPositive = viewerPositive * (0.25 + 0.75 * clamp(hybridPositive * 4.0, 0.0, 1.0));
  float panelMix = clamp(sqrt(max(0.0, ambiguity)) * (0.55 + 0.45 * positiveAgreement), 0.0, 1.0);
  float retinexResidual = carrier[index] - retinexBody[index];
  float dogResidual = dogLow[index] - dogHigh[index];
  float dogPositive = max(0.0, dogResidual);
  float dogNegative = max(0.0, -dogResidual);
  float dogPositiveAgreement = clamp(1.0 - abs(dogPositive - hybridPositive) * 4.0, 0.0, 1.0);
  float dogNegativeAgreement = clamp(1.0 - abs(dogNegative - hybridNegative) * 4.0, 0.0, 1.0);
  float positiveRetinexGate = clamp(0.18 + 0.34 * permission + 0.18 * positiveAgreement +
                                    0.16 * shapeSupport + 0.14 * localPositiveSupport,
                                    0.0,
                                    1.0);
  float negativeRetinexGate = clamp(0.30 + 0.40 * (1.0 - ambiguity) +
                                    0.18 * bodyAgreement + 0.12 * permission,
                                    0.0,
                                    1.0);
  float dogPositiveGate = clamp(0.16 + 0.30 * permission + 0.18 * positiveAgreement +
                                0.16 * shapeSupport + 0.12 * dogPositiveAgreement +
                                0.08 * localPositiveSupport,
                                0.0,
                                1.0);
  float dogNegativeGate = clamp(0.30 + 0.36 * (1.0 - ambiguity) +
                                0.20 * bodyAgreement + 0.14 * dogNegativeAgreement,
                                0.0,
                                1.0);
  float pos = (1.0 - panelMix) * hybridPositive +
              panelMix * consensusPositive +
              0.20 * positiveRetinexGate * max(0.0, retinexResidual) +
              0.18 * dogPositiveGate * dogPositive;
  float neg = (1.0 - ambiguity) * hybridNegative +
              ambiguity * (0.55 * hybridNegative + 0.45 * viewerNegative) +
              0.16 * negativeRetinexGate * max(0.0, -retinexResidual) +
              0.12 * dogNegativeGate * dogNegative;
  adaptiveBody[index] = body;
  positiveRaw[index] = pos;
  negativeRaw[index] = neg;
  float attachment = clamp(0.31 * shapeSupport +
                           0.21 * localPositiveSupport +
                           0.20 * permission +
                           0.20 * positiveAgreement +
                           0.08 * bodyAgreement,
                           0.0,
                           1.0);
  float support = sqrt(clamp(occupancySupport[index], 0.0, 1.0));
  confidence[index] = clamp((0.10 + 0.90 * (0.28 * bodyAgreement +
                                            0.22 * positiveAgreement +
                                            0.20 * permission +
                                            0.15 * (1.0 - ambiguity) +
                                            0.15 * attachment)) *
                            (0.30 + 0.70 * support),
                            0.0,
                            1.0);
  agreement[index] = clamp(0.40 * bodyAgreement + 0.35 * positiveAgreement + 0.25 * permission, 0.0, 1.0);
}

kernel void glossFieldAssembleUnifiedKernel(const device float* bodyRaw [[buffer(0)]],
                                            const device float* positiveRaw [[buffer(1)]],
                                            const device float* negativeRaw [[buffer(2)]],
                                            const device float* confidenceRaw [[buffer(3)]],
                                            const device float* agreementRaw [[buffer(4)]],
                                            device float* body [[buffer(5)]],
                                            device float* signal [[buffer(6)]],
                                            device float* positive [[buffer(7)]],
                                            device float* negative [[buffer(8)]],
                                            device float* boundary [[buffer(9)]],
                                            device float* congruence [[buffer(10)]],
                                            device float* confidence [[buffer(11)]],
                                            device atomic_uint* maxBits [[buffer(12)]],
                                            constant GlossFieldCellUniforms& u [[buffer(13)]],
                                            uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.cellCount, 0));
  if (index >= total) return;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float gxBody = 0.5 * (sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x + 1, y) -
                        sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x - 1, y));
  float gyBody = 0.5 * (sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x, y + 1) -
                        sampleGridClamped(bodyRaw, u.gridWidth, u.gridHeight, x, y - 1));
  float signalXp = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x + 1, y) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x + 1, y);
  float signalXm = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x - 1, y) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x - 1, y);
  float signalYp = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x, y + 1) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x, y + 1);
  float signalYm = sampleGridClamped(positiveRaw, u.gridWidth, u.gridHeight, x, y - 1) -
                   sampleGridClamped(negativeRaw, u.gridWidth, u.gridHeight, x, y - 1);
  float gxSignal = 0.5 * (signalXp - signalXm);
  float gySignal = 0.5 * (signalYp - signalYm);
  float magBody = sqrt(gxBody * gxBody + gyBody * gyBody);
  float magSignal = sqrt(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0;
  if (magBody > 1e-6 && magSignal > 1e-6) {
    localCongruence = clamp(abs((gxBody * gxSignal + gyBody * gySignal) / (magBody * magSignal)), 0.0, 1.0);
  } else if (magSignal > 1e-6) {
    localCongruence = 0.35;
  }
  float localBoundary = magSignal * 4.0;
  float weight = (0.35 + 0.65 * clamp(localCongruence, 0.0, 1.0)) *
                 (0.45 + 0.55 * clamp(confidenceRaw[index], 0.0, 1.0));
  body[index] = max(0.0, bodyRaw[index]);
  positive[index] = max(0.0, positiveRaw[index]) * weight;
  negative[index] = max(0.0, negativeRaw[index]) * weight;
  signal[index] = positive[index] - negative[index];
  boundary[index] = max(0.0, localBoundary);
  congruence[index] = localCongruence;
  confidence[index] = confidenceRaw[index];
  atomic_fetch_max_explicit(&maxBits[0], as_type<uint>(max(body[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[1], as_type<uint>(max(positive[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[2], as_type<uint>(max(negative[index], 0.0)), memory_order_relaxed);
  atomic_fetch_max_explicit(&maxBits[3], as_type<uint>(max(boundary[index], 0.0)), memory_order_relaxed);
}

struct PlotSurfaceClearUniforms {
  float r;
  float g;
  float b;
  float a;
};

struct GlossFieldSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
};

struct GlossProjectionSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float sourceAspect;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
  float pointRadiusPixels;
  float modelView[16];
  float projection[16];
};

float3 mixGloss3(float3 a, float3 b, float t) {
  float k = clamp(t, 0.0, 1.0);
  return a + (b - a) * k;
}

float glossSourcePresence(const device float* meanR,
                          const device float* meanG,
                          const device float* meanB,
                          uint idx) {
  return clamp(max(meanR[idx], max(meanG[idx], meanB[idx])), 0.0, 1.0);
}

float3 glossSurfaceSourceHueColor(const device float* meanR,
                                  const device float* meanG,
                                  const device float* meanB,
                                  constant GlossFieldSurfaceUniforms& u,
                                  uint idx) {
  float sr = 0.0;
  float sg = 0.0;
  float sb = 0.0;
  mapDisplayColor(meanR[idx], meanG[idx], meanB[idx], sr, sg, sb);
  applyDisplaySaturation(min(3.0, u.colorSaturation), sr, sg, sb);
  return float3(sr, sg, sb);
}

float4 glossSurfaceUnderlayColor(const device float* meanR,
                                 const device float* meanG,
                                 const device float* meanB,
                                 const device float* body,
                                 const device float* confidence,
                                 constant GlossFieldSurfaceUniforms& u,
                                 uint idx) {
  float sr = 0.0;
  float sg = 0.0;
  float sb = 0.0;
  mapDisplayColor(meanR[idx], meanG[idx], meanB[idx], sr, sg, sb);
  float sourceLuma = clamp(0.2126 * sr + 0.7152 * sg + 0.0722 * sb, 0.0, 1.0);
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, idx);
  float confidenceValue = clamp(confidence[idx], 0.0, 1.0);
  float bodyValue = clamp(body[idx], 0.0, 1.0);
  float structure = max(sqrt(confidenceValue), sqrt(sourcePresence));
  float bodyGain = 0.34 + 0.66 * clamp(u.glossBodyOpacity, 0.0, 1.0);
  float3 color;
  if (u.colorMode == 1) {
    float3 sourceHue = glossSurfaceSourceHueColor(meanR, meanG, meanB, u, idx);
    float shaped = pow(max(sourceLuma, 0.0), 0.85);
    float3 neutralBase = float3(0.10 + 0.52 * shaped,
                                0.10 + 0.50 * shaped,
                                0.11 + 0.46 * shaped);
    color = mixGloss3(neutralBase, sourceHue, 0.42);
  } else {
    float gray = 0.11 + 0.62 * pow(max(sourceLuma, 0.35 * bodyValue), 0.84);
    color = float3(gray * 0.98, gray * 0.985, gray);
  }
  float alpha = clamp((0.10 + 0.48 * structure) * bodyGain, 0.0, 0.68);
  return float4(clamp(color, 0.0, 1.0), alpha);
}

float4 glossSurfaceDisplayColor(const device float* meanR,
                                const device float* meanG,
                                const device float* meanB,
                                const device float* carrierY,
                                const device float* carrierMax,
                                const device float* carrierMin,
                                const device float* neutrality,
                                const device float* body,
                                const device float* positive,
                                const device float* negative,
                                const device float* boundary,
                                const device float* congruence,
                                const device float* confidence,
                                const device float* signal,
                                constant GlossFieldSurfaceUniforms& u,
                                uint idx) {
  float base = clamp(body[idx], 0.0, 1.0);
  float pos = clamp(positive[idx], 0.0, 1.0);
  float neg = clamp(negative[idx], 0.0, 1.0);
  if (u.debugMode != 0) {
    float scalar = 0.0;
    if (u.debugMode == 1) scalar = carrierMax[idx];
    else if (u.debugMode == 2) scalar = carrierY[idx];
    else if (u.debugMode == 3) scalar = carrierMin[idx];
    else if (u.debugMode == 4) scalar = neutrality[idx];
    pos = clamp(scalar, 0.0, 1.0);
    neg = 0.0;
    base = clamp(scalar, 0.0, 1.0);
  }
  float confidenceValue = clamp(confidence[idx], 0.0, 1.0);
  float congruenceValue = clamp(congruence[idx], 0.0, 1.0);
  float boundaryValue = clamp(boundary[idx], 0.0, 1.0);
  float ambiguity = clamp(1.0 - confidenceValue, 0.0, 1.0);
  float signalScale = max(1.0, u.glossLiftScale);
  pos = clamp(pos * signalScale, 0.0, 1.0);
  neg = clamp(neg * signalScale, 0.0, 1.0);
  float positiveDisplay = smoothstep(0.035, 1.0, pos);
  float negativeDisplay = smoothstep(0.035, 1.0, neg);
  float signalPresence = max(positiveDisplay, negativeDisplay);
  float structureStrength = max(congruenceValue, boundaryValue);
  float3 color;
  if (u.colorMode == 1) {
    float3 sourceHue = glossSurfaceSourceHueColor(meanR, meanG, meanB, u, idx);
    float baseMix = clamp(u.glossBodyOpacity * (0.22 + 0.78 * confidenceValue) *
                              (0.86 - 0.22 * signalPresence),
                          0.0,
                          1.0);
    float shaped = pow(max(base, 0.0), 0.78);
    float3 neutralBase = float3(0.16 + 0.60 * shaped,
                                0.16 + 0.58 * shaped,
                                0.17 + 0.54 * shaped);
    color = mixGloss3(float3(0.03, 0.03, 0.04), mixGloss3(neutralBase, sourceHue, 0.68), baseMix);
    if (positiveDisplay > 0.0) {
      float3 warm = mixGloss3(sourceHue, float3(1.0, 0.95, 0.86), 0.54);
      color = mixGloss3(color,
                        warm,
                        clamp(u.glossHighlightOpacity * positiveDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
    if (negativeDisplay > 0.0) {
      float3 cool = mixGloss3(sourceHue, float3(0.08, 0.14, 0.24), 0.74);
      color = mixGloss3(color,
                        cool,
                        clamp(u.glossHighlightOpacity * negativeDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
  } else {
    float shaped = pow(max(base, 0.0), 0.78);
    float3 neutralBase = float3(0.16 + 0.64 * shaped,
                                0.16 + 0.64 * shaped,
                                0.17 + 0.60 * shaped);
    color = mixGloss3(float3(0.03, 0.03, 0.04),
                      neutralBase,
                      clamp(u.glossBodyOpacity * (0.22 + 0.78 * confidenceValue) *
                                (0.86 - 0.22 * signalPresence),
                            0.0,
                            1.0));
    if (positiveDisplay > 0.0) {
      color = mixGloss3(color,
                        float3(1.0, 0.89, 0.36),
                        clamp(u.glossHighlightOpacity * positiveDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
    if (negativeDisplay > 0.0) {
      color = mixGloss3(color,
                        float3(0.22, 0.76, 1.0),
                        clamp(u.glossHighlightOpacity * negativeDisplay * (0.22 + 0.78 * structureStrength),
                              0.0,
                              1.0));
    }
  }
  if (boundaryValue > 0.0) {
    color = mixGloss3(color, float3(0.98, 0.98, 0.94), clamp(0.10 + 0.26 * boundaryValue, 0.0, 0.34));
  }
  float alpha = clamp(u.glossBodyOpacity * (0.12 + 0.62 * confidenceValue) *
                          (0.82 - 0.18 * signalPresence) +
                          u.glossHighlightOpacity * signalPresence * (0.16 + 0.84 * structureStrength),
                      0.018,
                      1.0);
  if (u.diagnosticMode == 1) {
    float gray = 0.16 + 0.78 * confidenceValue;
    color = mixGloss3(color, float3(gray, gray, gray), 0.36);
    color = mixGloss3(color, float3(1.0, 1.0, 0.96), 0.10 * boundaryValue);
    alpha = clamp(alpha * (0.55 + 0.45 * confidenceValue) + 0.10 * confidenceValue, 0.018, 1.0);
  } else if (u.diagnosticMode == 2) {
    float gray = 0.12 + 0.74 * ambiguity;
    color = mixGloss3(color, float3(gray * 0.94, gray * 0.97, gray), 0.34);
    color = mixGloss3(color, float3(0.80, 0.90, 1.0), 0.10 * boundaryValue * ambiguity);
    alpha = clamp(alpha * (0.48 + 0.52 * ambiguity) + 0.08 * ambiguity, 0.018, 1.0);
  }
  (void)signal;
  return float4(clamp(color, 0.0, 1.0), alpha);
}

kernel void glossFieldSurfaceRenderKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                          const device float* meanR [[buffer(0)]],
                                          const device float* meanG [[buffer(1)]],
                                          const device float* meanB [[buffer(2)]],
                                          const device float* carrierY [[buffer(3)]],
                                          const device float* carrierMax [[buffer(4)]],
                                          const device float* carrierMin [[buffer(5)]],
                                          const device float* neutrality [[buffer(6)]],
                                          const device float* body [[buffer(7)]],
                                          const device float* positive [[buffer(8)]],
                                          const device float* negative [[buffer(9)]],
                                          const device float* boundary [[buffer(10)]],
                                          const device float* congruence [[buffer(11)]],
                                          const device float* confidence [[buffer(12)]],
                                          const device float* signal [[buffer(13)]],
                                          constant GlossFieldSurfaceUniforms& u [[buffer(14)]],
                                          uint2 gid [[thread_position_in_grid]]) {
  uint outWidth = outTexture.get_width();
  uint outHeight = outTexture.get_height();
  if (gid.x >= outWidth || gid.y >= outHeight ||
      u.gridWidth <= 0 || u.gridHeight <= 0 ||
      u.surfaceWidth <= 0 || u.surfaceHeight <= 0) return;
  float2 uv = float2((float(gid.x) + 0.5) / float(max(outWidth, 1u)),
                     (float(gid.y) + 0.5) / float(max(outHeight, 1u)));
  int x = clamp(int(floor(uv.x * float(u.gridWidth))), 0, u.gridWidth - 1);
  int yFromBottom = clamp(int(floor(uv.y * float(u.gridHeight))), 0, u.gridHeight - 1);
  int y = u.gridHeight - 1 - yFromBottom;
  uint idx = uint(y * u.gridWidth + x);
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, idx);
  float confidenceValue = clamp(confidence[idx], 0.0, 1.0);
  if (sourcePresence <= 0.01 && confidenceValue <= 0.01) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  float4 underlay = glossSurfaceUnderlayColor(meanR, meanG, meanB, body, confidence, u, idx);
  float4 display = glossSurfaceDisplayColor(meanR, meanG, meanB,
                                            carrierY, carrierMax, carrierMin, neutrality,
                                            body, positive, negative, boundary, congruence,
                                            confidence, signal, u, idx);
  float4 color = overColor(underlay, display);
  outTexture.write(color, gid);
}

float4 multiplyProjectionMatrix(constant float* m, float4 v) {
  return float4(m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
                m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
                m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
                m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w);
}

bool glossProjectionCellScreenPosition(const device float* carrierY,
                                       const device float* carrierMax,
                                       const device float* carrierMin,
                                       const device float* neutrality,
                                       const device float* signal,
                                       constant GlossProjectionSurfaceUniforms& u,
                                       uint index,
                                       thread int* outPx,
                                       thread int* outPy,
                                       thread float* outNdcZ) {
  if (outPx == nullptr || outPy == nullptr || outNdcZ == nullptr) return false;
  int x = int(index % uint(u.gridWidth));
  int y = int(index / uint(u.gridWidth));
  float aspect = clamp(u.sourceAspect, 0.25, 4.0);
  const float kMajorHalf = 1.22;
  float halfWidth = aspect >= 1.0 ? kMajorHalf : kMajorHalf * aspect;
  float halfDepth = aspect >= 1.0 ? kMajorHalf / aspect : kMajorHalf;
  float xNorm = (float(x) + 0.5) / float(u.gridWidth);
  float yNormInv = (float(y) + 0.5) / float(u.gridHeight);
  float xPos = -halfWidth + 2.0 * halfWidth * xNorm;
  float imageY = halfDepth - 2.0 * halfDepth * yNormInv;

  float signedValue = signal[index];
  if (u.debugMode == 1) {
    signedValue = clamp(carrierMax[index], 0.0, 1.0);
  } else if (u.debugMode == 2) {
    signedValue = clamp(carrierY[index], 0.0, 1.0);
  } else if (u.debugMode == 3) {
    signedValue = clamp(carrierMin[index], 0.0, 1.0);
  } else if (u.debugMode == 4) {
    signedValue = clamp(neutrality[index], 0.0, 1.0);
  }
  float zPos = u.debugMode == 0 ? signedValue : max(0.0, signedValue);

  float4 world = float4(xPos, imageY, zPos, 1.0);
  float4 eye = multiplyProjectionMatrix(u.modelView, world);
  float4 clip = multiplyProjectionMatrix(u.projection, eye);
  if (abs(clip.w) <= 1e-6) return false;
  float invW = 1.0 / clip.w;
  float ndcX = clip.x * invW;
  float ndcY = clip.y * invW;
  float ndcZ = clip.z * invW;
  if (ndcX < -1.12 || ndcX > 1.12 || ndcY < -1.12 || ndcY > 1.12 || ndcZ < -1.25 || ndcZ > 1.25) {
    return false;
  }
  *outPx = int(round((ndcX * 0.5 + 0.5) * float(u.surfaceWidth - 1)));
  int py = int(round((ndcY * 0.5 + 0.5) * float(u.surfaceHeight - 1)));
  *outPy = u.surfaceHeight - 1 - py;
  *outNdcZ = ndcZ;
  return true;
}

kernel void glossProjectionSurfaceSelectKernel(device atomic_uint* selection [[buffer(0)]],
                                               const device float* meanR [[buffer(1)]],
                                               const device float* meanG [[buffer(2)]],
                                               const device float* meanB [[buffer(3)]],
                                               const device float* carrierY [[buffer(4)]],
                                               const device float* carrierMax [[buffer(5)]],
                                               const device float* carrierMin [[buffer(6)]],
                                               const device float* neutrality [[buffer(7)]],
                                               const device float* confidence [[buffer(8)]],
                                               const device float* signal [[buffer(9)]],
                                               constant GlossProjectionSurfaceUniforms& u [[buffer(10)]],
                                               uint index [[thread_position_in_grid]]) {
  uint cellCount = uint(max(u.gridWidth * u.gridHeight, 0));
  if (index >= cellCount || u.surfaceWidth <= 0 || u.surfaceHeight <= 0) return;
  float sourcePresence = glossSourcePresence(meanR, meanG, meanB, index);
  float confidenceValue = clamp(confidence[index], 0.0, 1.0);
  if (sourcePresence <= 0.01 && confidenceValue <= 0.01) return;
  int px = 0;
  int py = 0;
  float ndcZ = 0.0;
  if (!glossProjectionCellScreenPosition(carrierY, carrierMax, carrierMin, neutrality, signal,
                                         u, index, &px, &py, &ndcZ)) {
    return;
  }
  float radius = clamp(u.pointRadiusPixels * (0.75 + 0.45 * confidenceValue), 1.0, 5.0);
  int r = int(ceil(radius));
  uint depthPriority = uint(clamp((1.25 - ndcZ) / 2.5, 0.0, 1.0) * 4095.0) + 1u;
  uint cellBits = min(index + 1u, 0x000FFFFFu);
  uint packed = (depthPriority << 20) | cellBits;
  for (int oy = -r; oy <= r; ++oy) {
    for (int ox = -r; ox <= r; ++ox) {
      float dist = sqrt(float(ox * ox + oy * oy));
      if (dist > radius) continue;
      int sx = px + ox;
      int sy = py + oy;
      if (sx < 0 || sx >= u.surfaceWidth || sy < 0 || sy >= u.surfaceHeight) continue;
      uint pixelIndex = uint(sy * u.surfaceWidth + sx);
      atomic_fetch_max_explicit(&selection[pixelIndex], packed, memory_order_relaxed);
    }
  }
}

kernel void glossProjectionSurfaceShadeKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                              device atomic_uint* selection [[buffer(0)]],
                                              const device float* meanR [[buffer(1)]],
                                              const device float* meanG [[buffer(2)]],
                                              const device float* meanB [[buffer(3)]],
                                              const device float* carrierY [[buffer(4)]],
                                              const device float* carrierMax [[buffer(5)]],
                                              const device float* carrierMin [[buffer(6)]],
                                              const device float* neutrality [[buffer(7)]],
                                              const device float* body [[buffer(8)]],
                                              const device float* positive [[buffer(9)]],
                                              const device float* negative [[buffer(10)]],
                                              const device float* boundary [[buffer(11)]],
                                              const device float* congruence [[buffer(12)]],
                                              const device float* confidence [[buffer(13)]],
                                              const device float* signal [[buffer(14)]],
                                              constant GlossProjectionSurfaceUniforms& u [[buffer(15)]],
                                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() ||
      u.surfaceWidth <= 0 || u.surfaceHeight <= 0) {
    return;
  }
  uint pixelIndex = gid.y * uint(u.surfaceWidth) + gid.x;
  uint packed = atomic_load_explicit(&selection[pixelIndex], memory_order_relaxed);
  uint cellIndex = packed & 0x000FFFFFu;
  if (cellIndex == 0u) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  uint index = cellIndex - 1u;
  uint cellCount = uint(max(u.gridWidth * u.gridHeight, 0));
  if (index >= cellCount) {
    outTexture.write(float4(0.0), gid);
    return;
  }
  GlossFieldSurfaceUniforms fieldU;
  fieldU.gridWidth = u.gridWidth;
  fieldU.gridHeight = u.gridHeight;
  fieldU.surfaceWidth = u.surfaceWidth;
  fieldU.surfaceHeight = u.surfaceHeight;
  fieldU.algorithm = u.algorithm;
  fieldU.colorMode = u.colorMode;
  fieldU.debugMode = u.debugMode;
  fieldU.diagnosticMode = u.diagnosticMode;
  fieldU.colorSaturation = u.colorSaturation;
  fieldU.glossBodyOpacity = u.glossBodyOpacity;
  fieldU.glossHighlightOpacity = u.glossHighlightOpacity;
  fieldU.glossLiftScale = u.glossLiftScale;
  float4 display = glossSurfaceDisplayColor(meanR, meanG, meanB,
                                            carrierY, carrierMax, carrierMin, neutrality,
                                            body, positive, negative, boundary, congruence,
                                            confidence, signal, fieldU, index);
  outTexture.write(display, gid);
}

kernel void plotSurfaceClearKernel(texture2d<float, access::write> outTexture [[texture(0)]],
                                   constant PlotSurfaceClearUniforms& u [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height()) return;
  outTexture.write(float4(u.r, u.g, u.b, u.a), gid);
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
      id<MTLFunction> rasterSourceTextureFn = [c.library newFunctionWithName:@"rasterSourceTextureKernel"];
      if (rasterSourceTextureFn == nil) {
        c.initError = "Missing raster source texture Metal kernel.";
        return;
      }
      c.rasterSourceTexturePipeline = [c.device newComputePipelineStateWithFunction:rasterSourceTextureFn error:&pipelineError];
      if (c.rasterSourceTexturePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster source texture Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterOccupancyTextureFn = [c.library newFunctionWithName:@"rasterOccupancyTextureCountKernel"];
      if (rasterOccupancyTextureFn == nil) {
        c.initError = "Missing raster occupancy texture Metal kernel.";
        return;
      }
      c.rasterOccupancyTextureCountPipeline = [c.device newComputePipelineStateWithFunction:rasterOccupancyTextureFn error:&pipelineError];
      if (c.rasterOccupancyTextureCountPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster occupancy texture Metal pipeline.";
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
      c.rasterScopeDensityTexturePipeline = buildPipeline(@"rasterScopeDensityTextureKernel",
                                                          "Missing raster scope density texture Metal kernel.",
                                                          "Failed to create raster scope density texture Metal pipeline.");
      if (c.rasterScopeDensityTexturePipeline == nil) {
        return;
      }
      c.rasterScopeRangeTexturePipeline = buildPipeline(@"rasterScopeRangeTextureKernel",
                                                        "Missing raster scope range texture Metal kernel.",
                                                        "Failed to create raster scope range texture Metal pipeline.");
      if (c.rasterScopeRangeTexturePipeline == nil) {
        return;
      }
      c.rasterScopeRangeHistogramTexturePipeline =
          buildPipeline(@"rasterScopeRangeHistogramTextureKernel",
                        "Missing raster scope range histogram texture Metal kernel.",
                        "Failed to create raster scope range histogram texture Metal pipeline.");
      if (c.rasterScopeRangeHistogramTexturePipeline == nil) {
        return;
      }
      c.scopeRangeHistogramPercentilePipeline =
          buildPipeline(@"scopeRangeHistogramPercentileKernel",
                        "Missing scope range histogram percentile Metal kernel.",
                        "Failed to create scope range histogram percentile Metal pipeline.");
      if (c.scopeRangeHistogramPercentilePipeline == nil) {
        return;
      }
      c.scopeRangeFinalizePipeline = buildPipeline(@"scopeRangeFinalizeKernel",
                                                   "Missing scope range finalize Metal kernel.",
                                                   "Failed to create scope range finalize Metal pipeline.");
      if (c.scopeRangeFinalizePipeline == nil) {
        return;
      }
      c.histogramMaxPipeline = buildPipeline(@"histogramSurfaceMaxKernel",
                                             "Missing histogram surface max Metal kernel.",
                                             "Failed to create histogram surface max Metal pipeline.");
      if (c.histogramMaxPipeline == nil) {
        return;
      }
      c.histogramSurfaceRenderPipeline = buildPipeline(@"histogramSurfaceRenderKernel",
                                                       "Missing histogram surface render Metal kernel.",
                                                       "Failed to create histogram surface render Metal pipeline.");
      if (c.histogramSurfaceRenderPipeline == nil) {
        return;
      }
      c.glossFieldAccumulatePipeline = buildPipeline(@"glossFieldAccumulateKernel",
                                                     "Missing gloss field accumulate Metal kernel.",
                                                     "Failed to create gloss field accumulate Metal pipeline.");
      if (c.glossFieldAccumulatePipeline == nil) {
        return;
      }
      c.rasterGlossFieldAccumulateTexturePipeline =
          buildPipeline(@"rasterGlossFieldAccumulateTextureKernel",
                        "Missing raster gloss field accumulate texture Metal kernel.",
                        "Failed to create raster gloss field accumulate texture Metal pipeline.");
      if (c.rasterGlossFieldAccumulateTexturePipeline == nil) {
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
      c.glossFieldMergeMaxBitsPipeline = buildPipeline(@"glossFieldMergeMaxBitsKernel",
                                                       "Missing gloss field merge max-bits Metal kernel.",
                                                       "Failed to create gloss field merge max-bits Metal pipeline.");
      if (c.glossFieldMergeMaxBitsPipeline == nil) {
        return;
      }
      c.glossFieldFinalNormalizePipeline = buildPipeline(@"glossFieldFinalNormalizeKernel",
                                                         "Missing gloss field final normalize Metal kernel.",
                                                         "Failed to create gloss field final normalize Metal pipeline.");
      if (c.glossFieldFinalNormalizePipeline == nil) {
        return;
      }
      c.glossFieldLocalPercentilePipeline = buildPipeline(@"glossFieldLocalPercentileKernel",
                                                          "Missing gloss field local percentile Metal kernel.",
                                                          "Failed to create gloss field local percentile Metal pipeline.");
      if (c.glossFieldLocalPercentilePipeline == nil) {
        return;
      }
      c.glossFieldCandidate2RawPipeline = buildPipeline(@"glossFieldCandidate2RawKernel",
                                                        "Missing gloss field candidate 2 raw Metal kernel.",
                                                        "Failed to create gloss field candidate 2 raw Metal pipeline.");
      if (c.glossFieldCandidate2RawPipeline == nil) {
        return;
      }
      c.glossFieldAssembleUnifiedPipeline = buildPipeline(@"glossFieldAssembleUnifiedKernel",
                                                          "Missing gloss field assemble unified Metal kernel.",
                                                          "Failed to create gloss field assemble unified Metal pipeline.");
      if (c.glossFieldAssembleUnifiedPipeline == nil) {
        return;
      }
      c.glossFieldSurfaceRenderPipeline = buildPipeline(@"glossFieldSurfaceRenderKernel",
                                                        "Missing gloss field surface render Metal kernel.",
                                                        "Failed to create gloss field surface render Metal pipeline.");
      if (c.glossFieldSurfaceRenderPipeline == nil) {
        return;
      }
      c.glossProjectionSurfaceSelectPipeline = buildPipeline(@"glossProjectionSurfaceSelectKernel",
                                                             "Missing gloss projection surface select Metal kernel.",
                                                             "Failed to create gloss projection surface select Metal pipeline.");
      if (c.glossProjectionSurfaceSelectPipeline == nil) {
        return;
      }
      c.glossProjectionSurfaceShadePipeline = buildPipeline(@"glossProjectionSurfaceShadeKernel",
                                                            "Missing gloss projection surface shade Metal kernel.",
                                                            "Failed to create gloss projection surface shade Metal pipeline.");
      if (c.glossProjectionSurfaceShadePipeline == nil) {
        return;
      }
      c.plotSurfaceClearPipeline = buildPipeline(@"plotSurfaceClearKernel",
                                                 "Missing plot surface clear Metal kernel.",
                                                 "Failed to create plot surface clear Metal pipeline.");
      if (c.plotSurfaceClearPipeline == nil) {
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

bool copyBufferOnDevice(id<MTLBuffer> src, id<MTLBuffer> dst, NSUInteger bytes, std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || src == nil || dst == nil || bytes == 0u ||
      [src length] < bytes || [dst length] < bytes) {
    if (error) *error = "metal-buffer-copy-unavailable";
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-buffer-copy-command-buffer-failed";
      return false;
    }
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-buffer-copy-encoder-failed";
      return false;
    }
    [encoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
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

id<MTLBuffer> makeEmptyPrivateBuffer(NSUInteger bytes) {
  MetalContext& ctx = context();
  if (!ctx.ready) return nil;
  return [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

void clearSharedBuffer(id<MTLBuffer> buffer) {
  if (buffer == nil) return;
  std::memset([buffer contents], 0, static_cast<size_t>([buffer length]));
}

bool clearBufferOnDevice(id<MTLBuffer> buffer, std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || buffer == nil) {
    if (error) *error = "metal-buffer-clear-unavailable";
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-buffer-clear-command-buffer-failed";
      return false;
    }
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-buffer-clear-encoder-failed";
      return false;
    }
    [encoder fillBuffer:buffer range:NSMakeRange(0, [buffer length]) value:0];
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

ResidentReadiness residentReadiness() {
  ResidentReadiness result{};
  std::string error;
  result.contextReady = ensureContext(&error);
  MetalContext& ctx = context();
  auto markMissing = [&](const char* name, bool ready) {
    if (ready) return;
    if (!result.missing.empty()) result.missing += ",";
    result.missing += name ? name : "unknown";
  };
  if (!result.contextReady) {
    result.missing = error.empty() ? (ctx.initError.empty() ? "metal-context" : ctx.initError)
                                   : error;
    return result;
  }
  result.rasterSourceTextureReady =
      ctx.rasterSourceTexturePipeline != nil &&
      ctx.rasterOccupancyTextureCountPipeline != nil;
  result.analyticalScopeReady =
      ctx.rasterScopeDensityTexturePipeline != nil &&
      ctx.rasterScopeRangeTexturePipeline != nil &&
      ctx.rasterScopeRangeHistogramTexturePipeline != nil &&
      ctx.scopeRangeHistogramPercentilePipeline != nil &&
      ctx.scopeRangeFinalizePipeline != nil;
  result.histogramSurfaceReady =
      result.analyticalScopeReady &&
      ctx.histogramMaxPipeline != nil &&
      ctx.histogramSurfaceRenderPipeline != nil;
  result.glossFieldCacheReady =
      ctx.rasterGlossFieldAccumulateTexturePipeline != nil &&
      ctx.glossFieldFinalizePipeline != nil &&
      ctx.glossFieldMaxPipeline != nil &&
      ctx.glossFieldNormalizePipeline != nil &&
      ctx.glossFieldBlurPipeline != nil &&
      ctx.glossFieldBodyPipeline != nil &&
      ctx.glossFieldRawSignalPipeline != nil &&
      ctx.glossFieldWeightedSignalPipeline != nil &&
      ctx.glossFieldMergeMaxBitsPipeline != nil &&
      ctx.glossFieldFinalNormalizePipeline != nil &&
      ctx.glossFieldLocalPercentilePipeline != nil &&
      ctx.glossFieldCandidate2RawPipeline != nil &&
      ctx.glossFieldAssembleUnifiedPipeline != nil;
  result.glossFieldSurfaceReady = result.glossFieldCacheReady &&
                                  ctx.glossFieldSurfaceRenderPipeline != nil &&
                                  ctx.plotSurfaceClearPipeline != nil;
  result.glossProjectionSurfaceReady = result.glossFieldCacheReady &&
                                       ctx.glossProjectionSurfaceSelectPipeline != nil &&
                                       ctx.glossProjectionSurfaceShadePipeline != nil &&
                                       ctx.plotSurfaceClearPipeline != nil;
  result.plotSurfaceReady = ctx.plotSurfaceClearPipeline != nil;
  markMissing("raster-source-texture", result.rasterSourceTextureReady);
  markMissing("analytical-scope", result.analyticalScopeReady);
  markMissing("histogram-surface", result.histogramSurfaceReady);
  markMissing("gloss-field-cache", result.glossFieldCacheReady);
  markMissing("gloss-field-surface", result.glossFieldSurfaceReady);
  markMissing("gloss-projection-surface", result.glossProjectionSurfaceReady);
  markMissing("plot-surface", result.plotSurfaceReady);
  return result;
}

std::string residentPipelineUnavailableReason(const char* stage) {
  ResidentReadiness readiness = residentReadiness();
  std::string reason = stage && stage[0] != '\0' ? stage : "metal-resident-pipeline";
  reason += "-pipeline-unavailable";
  if (!readiness.contextReady) {
    reason += ":context";
  } else if (!readiness.missing.empty()) {
    reason += ":missing=" + readiness.missing;
  }
  return reason;
}

bool createPlotSurface(int width,
                       int height,
                       int pixelFormat,
                       PlotSurface* outSurface,
                       std::string* error) {
  if (outSurface) *outSurface = PlotSurface{};
  if (error) error->clear();
  if (!outSurface) {
    if (error) *error = "missing-plot-surface-output";
    return false;
  }
  if (width <= 0 || height <= 0) {
    if (error) *error = "invalid-plot-surface-size";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-format";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  MetalContext& ctx = context();
  @autoreleasepool {
    const size_t bytesPerElement = sourceSignalBytesPerElement(pixelFormat);
    const size_t bytesPerRow = static_cast<size_t>(width) * bytesPerElement;
    const size_t byteSize = bytesPerRow * static_cast<size_t>(height);
    NSDictionary* surfaceProperties = @{
      (__bridge NSString*)kIOSurfaceWidth: @(width),
      (__bridge NSString*)kIOSurfaceHeight: @(height),
      (__bridge NSString*)kIOSurfaceBytesPerElement: @(bytesPerElement),
      (__bridge NSString*)kIOSurfaceBytesPerRow: @(bytesPerRow),
      (__bridge NSString*)kIOSurfaceAllocSize: @(byteSize),
      (__bridge NSString*)kIOSurfacePixelFormat: @(sourceSignalIOSurfacePixelFormat(pixelFormat)),
    };
    IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)surfaceProperties);
    if (surface == nullptr) {
      if (error) *error = "plot-surface-iosurface-allocation-failed";
      return false;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sourceSignalMetalPixelFormat(pixelFormat)
                                                           width:static_cast<NSUInteger>(width)
                                                          height:static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [ctx.device newTextureWithDescriptor:desc iosurface:surface plane:0];
    if (texture == nil) {
      CFRelease(surface);
      if (error) *error = "plot-surface-metal-texture-failed";
      return false;
    }
    const uint32_t surfaceId = static_cast<uint32_t>(IOSurfaceGetID(surface));
    {
      std::lock_guard<std::mutex> lock(plotSurfaceMutex());
      auto& registry = plotSurfaceRegistry();
      auto existing = registry.find(surfaceId);
      if (existing != registry.end()) {
        if (existing->second.surface) CFRelease(existing->second.surface);
        registry.erase(existing);
      }
      PlotSurfaceRecord record{};
      record.surface = surface;
      record.texture = texture;
      record.width = width;
      record.height = height;
      record.pixelFormat = pixelFormat;
      record.byteSize = byteSize;
      registry.emplace(surfaceId, record);
    }
    if (outSurface) {
      outSurface->surfaceId = surfaceId;
      outSurface->width = width;
      outSurface->height = height;
      outSurface->pixelFormat = pixelFormat;
      outSurface->byteSize = byteSize;
    }
    return true;
  }
}

bool clearPlotSurface(uint32_t surfaceId,
                      int width,
                      int height,
                      int pixelFormat,
                      float r,
                      float g,
                      float b,
                      float a,
                      std::string* error) {
  if (error) error->clear();
  if (surfaceId == 0 || width <= 0 || height <= 0) {
    if (error) *error = "invalid-plot-surface-clear-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-format";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  MetalContext& ctx = context();
  if (ctx.plotSurfaceClearPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("plot-surface-clear");
    return false;
  }
  id<MTLTexture> texture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(surfaceId);
    if (it != registry.end() &&
        it->second.width == width &&
        it->second.height == height &&
        it->second.pixelFormat == pixelFormat) {
      texture = it->second.texture;
    }
  }
  if (texture == nil) {
    texture = makeTextureFromIOSurface(ctx, surfaceId, width, height, pixelFormat, error);
    if (texture == nil) return false;
  }
  @autoreleasepool {
    PlotSurfaceClearUniforms uniforms{r, g, b, a};
    id<MTLBuffer> uniformBuffer =
        [ctx.device newBufferWithBytes:&uniforms
                                length:sizeof(uniforms)
                               options:MTLResourceStorageModeShared];
    if (uniformBuffer == nil) {
      if (error) *error = "plot-surface-clear-uniform-allocation-failed";
      return false;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "plot-surface-clear-command-buffer-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "plot-surface-clear-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.plotSurfaceClearPipeline];
    [encoder setTexture:texture atIndex:0];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
    NSUInteger groupWidth = ctx.plotSurfaceClearPipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.plotSurfaceClearPipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    MTLSize threadsPerGroup = MTLSizeMake(groupWidth, groupHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(width),
                                         static_cast<NSUInteger>(height),
                                         1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
    return true;
  }
}

void releasePlotSurface(uint32_t surfaceId) {
  if (surfaceId == 0) return;
  std::lock_guard<std::mutex> lock(plotSurfaceMutex());
  auto& registry = plotSurfaceRegistry();
  auto it = registry.find(surfaceId);
  if (it == registry.end()) return;
  if (it->second.surface) CFRelease(it->second.surface);
  registry.erase(it);
}

void releaseGlossFieldCache(GlossFieldCache* cache) {
  if (!cache) return;
  if (cache->cacheId != 0) {
    std::lock_guard<std::mutex> lock(glossFieldRegistryMutex());
    glossFieldRegistry().erase(cache->cacheId);
  }
  *cache = GlossFieldCache{};
}

bool bindIOSurfaceToOpenGLTexture(uint32_t surfaceId,
                                  int width,
                                  int height,
                                  int pixelFormat,
                                  uint32_t glTexture,
                                  std::string* error) {
  if (error) error->clear();
  if (surfaceId == 0 || width <= 0 || height <= 0 || glTexture == 0) {
    if (error) *error = "invalid-iosurface-gl-texture-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-iosurface-pixel-format";
    return false;
  }
  @autoreleasepool {
    IOSurfaceRef surface = IOSurfaceLookup(static_cast<IOSurfaceID>(surfaceId));
    if (surface == nullptr) {
      if (error) *error = "iosurface-lookup-failed";
      return false;
    }
    CGLContextObj cgl = CGLGetCurrentContext();
    if (cgl == nullptr) {
      CFRelease(surface);
      if (error) *error = "no-current-cgl-context";
      return false;
    }
    glBindTexture(GL_TEXTURE_RECTANGLE, static_cast<GLuint>(glTexture));
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    const GLenum internalFormat = pixelFormat == 1 ? GL_RGBA32F : GL_RGBA16F;
    const GLenum type = pixelFormat == 1 ? GL_FLOAT : GL_HALF_FLOAT;
    const CGLError cgError = CGLTexImageIOSurface2D(cgl,
                                                   GL_TEXTURE_RECTANGLE,
                                                   internalFormat,
                                                   static_cast<GLsizei>(width),
                                                   static_cast<GLsizei>(height),
                                                   GL_RGBA,
                                                   type,
                                                   surface,
                                                   0);
    glBindTexture(GL_TEXTURE_RECTANGLE, 0);
    CFRelease(surface);
    if (cgError != kCGLNoError) {
      if (error) *error = std::string("CGLTexImageIOSurface2D failed error=") + std::to_string(cgError);
      return false;
    }
    return true;
  }
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
  uniforms.chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms.chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms.chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms.chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms.chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms.chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }

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

bool buildRasterSourceMeshFromIOSurface(const RasterSourceRequest& request,
                                        uint32_t surfaceId,
                                        int surfaceWidth,
                                        int surfaceHeight,
                                        int surfacePixelFormat,
                                        std::vector<float>* outVerts,
                                        std::vector<float>* outColors,
                                        std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0 || surfaceId == 0 || surfaceWidth <= 0 || surfaceHeight <= 0 ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 || request.sampleCountX <= 0 ||
      surfaceWidth < request.sourceWidth || surfaceHeight < request.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface raster source request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface raster source format.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterSourceTexturePipeline == nil || ctx.rasterOccupancyTextureCountPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-raster-source");
    return false;
  }

  RasterSourceUniforms uniforms{};
  fillRasterSourceUniforms(request, &uniforms);
  uniforms.pixelFormat = surfacePixelFormat;

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
    id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    constexpr NSUInteger kRasterOccupancyBinCount = 18u * 18u * 18u;
    id<MTLBuffer> occupancyBuffer =
        makeEmptySharedBuffer(kRasterOccupancyBinCount * sizeof(uint32_t));
    if (sourceTexture == nil || vertBuffer == nil || colorBuffer == nil ||
        uniformBuffer == nil || occupancyBuffer == nil) {
      if (error) *error = localError.empty() ? "Failed to allocate Metal IOSurface raster buffers." : localError;
      return false;
    }
    clearSharedBuffer(occupancyBuffer);

    if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
      id<MTLBuffer> visibleCountBuffer = makeEmptySharedBuffer(sizeof(uint32_t));
      if (visibleCountBuffer == nil) {
        if (error) *error = "Failed to allocate Metal IOSurface raster occupancy counter.";
        return false;
      }
      clearSharedBuffer(visibleCountBuffer);

      id<MTLCommandBuffer> countCommand = [ctx.queue commandBuffer];
      if (countCommand == nil) {
        if (error) *error = "Failed to create Metal IOSurface raster occupancy command buffer.";
        return false;
      }
      id<MTLComputeCommandEncoder> countEncoder = [countCommand computeCommandEncoder];
      if (countEncoder == nil) {
        if (error) *error = "Failed to create Metal IOSurface raster occupancy encoder.";
        return false;
      }
      [countEncoder setComputePipelineState:ctx.rasterOccupancyTextureCountPipeline];
      [countEncoder setTexture:sourceTexture atIndex:0];
      [countEncoder setBuffer:occupancyBuffer offset:0 atIndex:0];
      [countEncoder setBuffer:visibleCountBuffer offset:0 atIndex:1];
      [countEncoder setBuffer:uniformBuffer offset:0 atIndex:2];
      NSUInteger width = ctx.rasterOccupancyTextureCountPipeline.maxTotalThreadsPerThreadgroup;
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
      if (error) *error = "Failed to create Metal IOSurface raster command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface raster encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterSourceTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:vertBuffer offset:0 atIndex:0];
    [encoder setBuffer:colorBuffer offset:0 atIndex:1];
    [encoder setBuffer:occupancyBuffer offset:0 atIndex:2];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:3];
    NSUInteger width = ctx.rasterSourceTexturePipeline.maxTotalThreadsPerThreadgroup;
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
                       bool allowReadback,
                       std::vector<float>* outDensity,
                       std::string* error) {
  std::string localError;
  if (!outDensity) {
    if (error) *error = "Missing Metal scope density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "Metal compact scope density readback disabled.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(request.pointCount, 0);
  const int width = std::max(request.width, 1);
  const int height = std::max(request.height, 1);
  const int channelCount = std::max(request.channelCount, 1);
  const size_t expectedSampleFloats = static_cast<size_t>(pointCount) * 5u;
  if (pointCount == 0 || packedSamples.size() < expectedSampleFloats) {
    if (error) *error = "Invalid Metal scope density sample buffer.";
    return false;
  }
  const size_t binCount = static_cast<size_t>(width) *
                          static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
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
    uniforms.onlyOverflow = request.onlyOverflow;
    uniforms.channelCount = channelCount;
    uniforms.lumaMethod = std::clamp(request.lumaMethod, 0, 3);

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

bool buildScopeDensityFromIOSurface(const RasterSourceRequest& rasterRequest,
                                    const ScopeDensityRequest& scopeRequest,
                                    uint32_t surfaceId,
                                    int surfaceWidth,
                                    int surfaceHeight,
                                    int surfacePixelFormat,
                                    bool allowReadback,
                                    std::vector<float>* outDensity,
                                    std::string* error) {
  std::string localError;
  if (!outDensity) {
    if (error) *error = "Missing Metal IOSurface scope density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "Metal IOSurface compact scope density readback disabled.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(scopeRequest.width, 1);
  const int height = std::max(scopeRequest.height, 1);
  const int channelCount = std::max(scopeRequest.channelCount, 1);
  const size_t binCount = static_cast<size_t>(width) *
                          static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  if (pointCount <= 0 || binCount == 0u || surfaceId == 0 || surfaceWidth <= 0 ||
      surfaceHeight <= 0 || rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface scope density request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface scope density format.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterScopeDensityTexturePipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-scope-density");
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeDensityUniforms scopeUniforms{};
  scopeUniforms.pointCount = pointCount;
  scopeUniforms.waveform = scopeRequest.waveform;
  scopeUniforms.scopeMode = scopeRequest.scopeMode;
  scopeUniforms.width = width;
  scopeUniforms.height = height;
  scopeUniforms.rangeMin = scopeRequest.rangeMin;
  scopeUniforms.invRange = scopeRequest.invRange;
  scopeUniforms.excludeOverflow = scopeRequest.excludeOverflow != 0 ? 1 : 0;
  scopeUniforms.onlyOverflow = scopeRequest.onlyOverflow != 0 ? 1 : 0;
  scopeUniforms.channelCount = channelCount;
  scopeUniforms.lumaMethod = std::clamp(scopeRequest.lumaMethod, 0, 3);

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> densityBuffer =
        makeEmptySharedBuffer(static_cast<NSUInteger>(binCount * sizeof(uint32_t)));
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    id<MTLBuffer> scopeUniformBuffer = makeSharedBuffer(&scopeUniforms, 1u);
    if (sourceTexture == nil || densityBuffer == nil || rasterUniformBuffer == nil ||
        scopeUniformBuffer == nil) {
      if (error) {
        *error = localError.empty() ? "Failed to allocate Metal IOSurface scope density resources."
                                    : localError;
      }
      return false;
    }
    clearSharedBuffer(densityBuffer);

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope density command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope density encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterScopeDensityTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:densityBuffer offset:0 atIndex:0];
    [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
    [encoder setBuffer:scopeUniformBuffer offset:0 atIndex:2];
    NSUInteger threads = ctx.rasterScopeDensityTexturePipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0) threads = 64;
    threads = std::min<NSUInteger>(threads, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
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

bool buildScopeRangeFromIOSurface(const RasterSourceRequest& rasterRequest,
                                  const ScopeRangeRequest& rangeRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  ScopeRangeResult* outRange,
                                  std::string* error) {
  std::string localError;
  if (!outRange) {
    if (error) *error = "Missing Metal IOSurface scope range output.";
    return false;
  }
  *outRange = ScopeRangeResult{};
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  constexpr uint32_t kRangeHistogramBins = 2048u;
  if (pointCount <= 0 || surfaceId == 0 || surfaceWidth <= 0 || surfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface scope range request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface scope range format.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterScopeRangeTexturePipeline == nil ||
      ctx.rasterScopeRangeHistogramTexturePipeline == nil ||
      ctx.scopeRangeHistogramPercentilePipeline == nil ||
      ctx.scopeRangeFinalizePipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-scope-range");
    return false;
  }

  const auto orderedUintFromFloatHost = [](float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
  };
  const auto floatFromOrderedUintHost = [](uint32_t value) {
    uint32_t bits = (value & 0x80000000u) != 0u ? (value ^ 0x80000000u) : ~value;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  };

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeRangeUniforms rangeUniforms{};
  rangeUniforms.pointCount = pointCount;
  rangeUniforms.waveform = rangeRequest.waveform;
  rangeUniforms.scopeMode = rangeRequest.scopeMode;
  rangeUniforms.includeRed = rangeRequest.includeRed != 0 ? 1 : 0;
  rangeUniforms.includeGreen = rangeRequest.includeGreen != 0 ? 1 : 0;
  rangeUniforms.includeBlue = rangeRequest.includeBlue != 0 ? 1 : 0;
  rangeUniforms.includeLuma = rangeRequest.includeLuma != 0 ? 1 : 0;
  rangeUniforms.includeOverflow = rangeRequest.includeOverflow != 0 ? 1 : 0;
  rangeUniforms.lumaMethod = std::clamp(rangeRequest.lumaMethod, 0, 3);
  rangeUniforms.previousRangeValid = rangeRequest.previousRangeValid != 0 ? 1 : 0;
  rangeUniforms.previousRangeMin = rangeRequest.previousRangeMin;
  rangeUniforms.previousRangeMax = rangeRequest.previousRangeMax;
  rangeUniforms.histogramBinCount = static_cast<int>(kRangeHistogramBins);

  const uint32_t initRangeBits[3] = {
      orderedUintFromFloatHost(std::numeric_limits<float>::infinity()),
      orderedUintFromFloatHost(-std::numeric_limits<float>::infinity()),
      0u,
  };
  const uint32_t initPercentiles[2] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
  };
  const uint32_t initFinalRange[3] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
      0u,
  };

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> rangeBitsBuffer = makeSharedBuffer(initRangeBits, 3u);
    id<MTLBuffer> histogramBuffer =
        makeEmptySharedBuffer(static_cast<NSUInteger>(kRangeHistogramBins * sizeof(uint32_t)));
    id<MTLBuffer> percentileBuffer = makeSharedBuffer(initPercentiles, 2u);
    id<MTLBuffer> finalRangeBuffer = makeSharedBuffer(initFinalRange, 3u);
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    id<MTLBuffer> rangeUniformBuffer = makeSharedBuffer(&rangeUniforms, 1u);
    if (sourceTexture == nil || rangeBitsBuffer == nil || histogramBuffer == nil ||
        percentileBuffer == nil || finalRangeBuffer == nil || rasterUniformBuffer == nil ||
        rangeUniformBuffer == nil) {
      if (error) {
        *error = localError.empty() ? "Failed to allocate Metal IOSurface scope range resources."
                                    : localError;
      }
      return false;
    }
    clearSharedBuffer(histogramBuffer);

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope range command buffer.";
      return false;
    }
    const auto dispatchPointKernel = [&](id<MTLComputePipelineState> pipeline,
                                         void (^bind)(id<MTLComputeCommandEncoder>)) -> bool {
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) return false;
      [encoder setComputePipelineState:pipeline];
      bind(encoder);
      NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
      if (threads == 0) threads = 64;
      threads = std::min<NSUInteger>(threads, 256);
      [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
      return true;
    };
    if (!dispatchPointKernel(ctx.rasterScopeRangeTexturePipeline,
                             ^(id<MTLComputeCommandEncoder> encoder) {
                               [encoder setTexture:sourceTexture atIndex:0];
                               [encoder setBuffer:rangeBitsBuffer offset:0 atIndex:0];
                               [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
                               [encoder setBuffer:rangeUniformBuffer offset:0 atIndex:2];
                             })) {
      if (error) *error = "Failed to create Metal IOSurface scope range encoder.";
      return false;
    }
    if (!dispatchPointKernel(ctx.rasterScopeRangeHistogramTexturePipeline,
                             ^(id<MTLComputeCommandEncoder> encoder) {
                               [encoder setTexture:sourceTexture atIndex:0];
                               [encoder setBuffer:histogramBuffer offset:0 atIndex:0];
                               [encoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
                               [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:2];
                               [encoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
                             })) {
      if (error) *error = "Failed to create Metal IOSurface scope range histogram encoder.";
      return false;
    }
    id<MTLComputeCommandEncoder> percentileEncoder = [commandBuffer computeCommandEncoder];
    if (percentileEncoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope range percentile encoder.";
      return false;
    }
    [percentileEncoder setComputePipelineState:ctx.scopeRangeHistogramPercentilePipeline];
    [percentileEncoder setBuffer:histogramBuffer offset:0 atIndex:0];
    [percentileEncoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
    [percentileEncoder setBuffer:percentileBuffer offset:0 atIndex:2];
    [percentileEncoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
    [percentileEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [percentileEncoder endEncoding];

    id<MTLComputeCommandEncoder> finalizeEncoder = [commandBuffer computeCommandEncoder];
    if (finalizeEncoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope range finalize encoder.";
      return false;
    }
    [finalizeEncoder setComputePipelineState:ctx.scopeRangeFinalizePipeline];
    [finalizeEncoder setBuffer:percentileBuffer offset:0 atIndex:0];
    [finalizeEncoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
    [finalizeEncoder setBuffer:finalRangeBuffer offset:0 atIndex:2];
    [finalizeEncoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
    [finalizeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [finalizeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    uint32_t packed[3] = {initFinalRange[0], initFinalRange[1], initFinalRange[2]};
    std::memcpy(packed, [finalRangeBuffer contents], sizeof(packed));
    outRange->minValue = floatFromOrderedUintHost(packed[0]);
    outRange->maxValue = floatFromOrderedUintHost(packed[1]);
    outRange->validCount = packed[2];
  }
  return true;
}

bool renderHistogramSurfaceFromIOSurface(const RasterSourceRequest& rasterRequest,
                                         const HistogramSurfaceRequest& histogramRequest,
                                         uint32_t sourceSurfaceId,
                                         int sourceSurfaceWidth,
                                         int sourceSurfaceHeight,
                                         int sourceSurfacePixelFormat,
                                         uint32_t outputSurfaceId,
                                         int outputSurfaceWidth,
                                         int outputSurfaceHeight,
                                         int outputSurfacePixelFormat,
                                         std::string* error) {
  if (error) error->clear();
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int binCount = std::max(histogramRequest.width, 1);
  const int channelCount = histogramRequest.scopeMode == 1 ? 1 : 3;
  const size_t densityCount = static_cast<size_t>(binCount) * static_cast<size_t>(channelCount);
  if (pointCount <= 0 || sourceSurfaceId == 0 || outputSurfaceId == 0 ||
      sourceSurfaceWidth <= 0 || sourceSurfaceHeight <= 0 ||
      outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || sourceSurfaceWidth < rasterRequest.sourceWidth ||
      sourceSurfaceHeight < rasterRequest.sourceHeight || densityCount == 0u) {
    if (error) *error = "invalid-metal-histogram-surface-request";
    return false;
  }
  if ((sourceSurfacePixelFormat != 0 && sourceSurfacePixelFormat != 1) ||
      (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1)) {
    if (error) *error = "unsupported-metal-histogram-surface-format";
    return false;
  }
  MetalContext& ctx = context();
  if (ctx.rasterScopeDensityTexturePipeline == nil ||
      ctx.histogramMaxPipeline == nil ||
      ctx.histogramSurfaceRenderPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-histogram-surface");
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = sourceSurfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeDensityUniforms densityUniforms{};
  densityUniforms.pointCount = pointCount;
  densityUniforms.waveform = 0;
  densityUniforms.scopeMode = histogramRequest.scopeMode;
  densityUniforms.width = binCount;
  densityUniforms.height = 1;
  densityUniforms.rangeMin = histogramRequest.rangeMin;
  densityUniforms.invRange = histogramRequest.invRange;
  densityUniforms.excludeOverflow = 1;
  densityUniforms.onlyOverflow = 0;
  densityUniforms.channelCount = channelCount;
  densityUniforms.lumaMethod = std::clamp(histogramRequest.lumaMethod, 0, 3);

  HistogramSurfaceUniforms surfaceUniforms{};
  surfaceUniforms.pointCount = pointCount;
  surfaceUniforms.scopeMode = histogramRequest.scopeMode;
  surfaceUniforms.width = binCount;
  surfaceUniforms.height = std::max(histogramRequest.height, 1);
  surfaceUniforms.rangeMin = histogramRequest.rangeMin;
  surfaceUniforms.invRange = histogramRequest.invRange;
  surfaceUniforms.showOverflow = histogramRequest.showOverflow != 0 ? 1 : 0;
  surfaceUniforms.highlightOverflow = histogramRequest.highlightOverflow != 0 ? 1 : 0;
  surfaceUniforms.lumaMethod = std::clamp(histogramRequest.lumaMethod, 0, 3);
  surfaceUniforms.channelCount = channelCount;

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() &&
        it->second.width == outputSurfaceWidth &&
        it->second.height == outputSurfaceHeight &&
        it->second.pixelFormat == outputSurfacePixelFormat) {
      outputTexture = it->second.texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-histogram-output-surface-missing";
    return false;
  }

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx,
                                 sourceSurfaceId,
                                 sourceSurfaceWidth,
                                 sourceSurfaceHeight,
                                 sourceSurfacePixelFormat,
                                 &localError);
    id<MTLBuffer> densityBuffer =
        makeEmptySharedBuffer(static_cast<NSUInteger>(densityCount * sizeof(uint32_t)));
    id<MTLBuffer> overflowDensityBuffer =
        surfaceUniforms.showOverflow != 0
            ? makeEmptySharedBuffer(static_cast<NSUInteger>(densityCount * sizeof(uint32_t)))
            : nil;
    id<MTLBuffer> maxDensityBuffer = makeEmptySharedBuffer(sizeof(uint32_t));
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    id<MTLBuffer> densityUniformBuffer = makeSharedBuffer(&densityUniforms, 1u);
    id<MTLBuffer> surfaceUniformBuffer = makeSharedBuffer(&surfaceUniforms, 1u);
    if (sourceTexture == nil || densityBuffer == nil || maxDensityBuffer == nil ||
        rasterUniformBuffer == nil || densityUniformBuffer == nil ||
        surfaceUniformBuffer == nil ||
        (surfaceUniforms.showOverflow != 0 && overflowDensityBuffer == nil)) {
      if (error) {
        *error = localError.empty() ? "metal-histogram-surface-allocation-failed" : localError;
      }
      return false;
    }
    clearSharedBuffer(densityBuffer);
    clearSharedBuffer(maxDensityBuffer);
    if (overflowDensityBuffer != nil) clearSharedBuffer(overflowDensityBuffer);

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-histogram-surface-command-buffer-failed";
      return false;
    }
    auto dispatchPointKernel = [&](id<MTLComputePipelineState> pipeline,
                                   id<MTLBuffer> targetDensity,
                                   const ScopeDensityUniforms& uniforms) -> bool {
      id<MTLBuffer> uniformsBuffer = makeSharedBuffer(&uniforms, 1u);
      if (uniformsBuffer == nil) return false;
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) return false;
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:sourceTexture atIndex:0];
      [encoder setBuffer:targetDensity offset:0 atIndex:0];
      [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
      [encoder setBuffer:uniformsBuffer offset:0 atIndex:2];
      NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
      if (threads == 0) threads = 64;
      threads = std::min<NSUInteger>(threads, 256);
      [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
      return true;
    };

    if (!dispatchPointKernel(ctx.rasterScopeDensityTexturePipeline,
                             densityBuffer,
                             densityUniforms)) {
      if (error) *error = "metal-histogram-density-encoder-failed";
      return false;
    }
    if (overflowDensityBuffer != nil) {
      ScopeDensityUniforms overflowUniforms = densityUniforms;
      overflowUniforms.excludeOverflow = 0;
      overflowUniforms.onlyOverflow = 1;
      if (!dispatchPointKernel(ctx.rasterScopeDensityTexturePipeline,
                               overflowDensityBuffer,
                               overflowUniforms)) {
        if (error) *error = "metal-histogram-overflow-density-encoder-failed";
        return false;
      }
    }

    id<MTLComputeCommandEncoder> maxEncoder = [commandBuffer computeCommandEncoder];
    if (maxEncoder == nil) {
      if (error) *error = "metal-histogram-max-encoder-failed";
      return false;
    }
    [maxEncoder setComputePipelineState:ctx.histogramMaxPipeline];
    [maxEncoder setBuffer:densityBuffer offset:0 atIndex:0];
    [maxEncoder setBuffer:maxDensityBuffer offset:0 atIndex:1];
    [maxEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:2];
    NSUInteger maxThreads = ctx.histogramMaxPipeline.maxTotalThreadsPerThreadgroup;
    if (maxThreads == 0) maxThreads = 64;
    maxThreads = std::min<NSUInteger>(maxThreads, 256);
    [maxEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(densityCount), 1, 1)
           threadsPerThreadgroup:MTLSizeMake(maxThreads, 1, 1)];
    [maxEncoder endEncoding];

    id<MTLComputeCommandEncoder> renderEncoder = [commandBuffer computeCommandEncoder];
    if (renderEncoder == nil) {
      if (error) *error = "metal-histogram-render-encoder-failed";
      return false;
    }
    [renderEncoder setComputePipelineState:ctx.histogramSurfaceRenderPipeline];
    [renderEncoder setTexture:outputTexture atIndex:0];
    [renderEncoder setBuffer:densityBuffer offset:0 atIndex:0];
    [renderEncoder setBuffer:overflowDensityBuffer offset:0 atIndex:1];
    [renderEncoder setBuffer:maxDensityBuffer offset:0 atIndex:2];
    [renderEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:3];
    NSUInteger groupWidth = ctx.histogramSurfaceRenderPipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.histogramSurfaceRenderPipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    [renderEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                               static_cast<NSUInteger>(outputSurfaceHeight),
                                               1)
              threadsPerThreadgroup:MTLSizeMake(groupWidth, groupHeight, 1)];
    [renderEncoder endEncoding];

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

bool buildGlossField(const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     bool allowReadback,
                     GlossFieldResult* out,
                     std::string* error) {
  std::string localError;
  if (!out) {
    if (error) *error = "Missing Metal gloss-field output.";
    return false;
  }
  *out = GlossFieldResult{};
  if (!allowReadback) {
    if (error) *error = "Metal packed gloss-field readback disabled.";
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

bool buildGlossFieldFromIOSurface(GlossFieldCache* cache,
                                  const RasterSourceRequest& rasterRequest,
                                  const GlossFieldRequest& fieldRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  uint64_t buildSerial,
                                  std::string* error) {
  if (error) error->clear();
  if (!cache) {
    if (error) *error = "missing-metal-gloss-field-cache";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int gridWidth = std::max(fieldRequest.gridWidth, 1);
  const int gridHeight = std::max(fieldRequest.gridHeight, 1);
  const NSUInteger cellCount =
      static_cast<NSUInteger>(gridWidth) * static_cast<NSUInteger>(gridHeight);
  if (pointCount <= 0 || cellCount == 0u || surfaceId == 0 ||
      surfaceWidth <= 0 || surfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight ||
      (surfacePixelFormat != 0 && surfacePixelFormat != 1)) {
    if (error) *error = "invalid-metal-iosurface-gloss-field-request";
    return false;
  }
  MetalContext& ctx = context();
  if (ctx.rasterGlossFieldAccumulateTexturePipeline == nil ||
      ctx.glossFieldFinalizePipeline == nil ||
      ctx.glossFieldMaxPipeline == nil ||
      ctx.glossFieldNormalizePipeline == nil ||
      ctx.glossFieldBlurPipeline == nil ||
      ctx.glossFieldBodyPipeline == nil ||
      ctx.glossFieldRawSignalPipeline == nil ||
      ctx.glossFieldWeightedSignalPipeline == nil ||
      ctx.glossFieldMergeMaxBitsPipeline == nil ||
      ctx.glossFieldFinalNormalizePipeline == nil ||
      ctx.glossFieldLocalPercentilePipeline == nil ||
      ctx.glossFieldCandidate2RawPipeline == nil ||
      ctx.glossFieldAssembleUnifiedPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-gloss-field");
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  GlossFieldAccumulateUniforms accumulateUniforms{};
  accumulateUniforms.pointCount = pointCount;
  accumulateUniforms.gridWidth = gridWidth;
  accumulateUniforms.gridHeight = gridHeight;
  accumulateUniforms.showOverflow = fieldRequest.showOverflow;

  GlossFieldCellUniforms cellUniforms{};
  cellUniforms.cellCount = static_cast<int>(cellCount);
  cellUniforms.gridWidth = gridWidth;
  cellUniforms.gridHeight = gridHeight;
  cellUniforms.neighborhoodChoice = fieldRequest.neighborhoodChoice;
  const int neighborhoodChoice = std::clamp(fieldRequest.neighborhoodChoice, 0, 2);
  const int neighborhoodRadius = neighborhoodChoice == 0 ? 1 : (neighborhoodChoice == 2 ? 3 : 2);
  const int analysisRadius = std::max(2, neighborhoodRadius * 2);
  const float percentile50 = 50.0f;
  const float percentile35 = 35.0f;

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> occupancyCountsBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumRBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumGBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumBBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumYBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMaxBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMinBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumNeutralityBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> occupancyBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanRBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanGBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanBBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierYBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMaxBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMinBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> neutralityBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> occupancyNormBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> tempBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> viewerBodyRawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> rawSignalBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> positiveBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> negativeBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> boundaryBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> congruenceBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> confidenceBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> signalBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyCoreBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyContextBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> retinexBodyBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> dogLowBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> body2RawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> positive2RawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> negative2RawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> confidence2RawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> agreement2RawBuffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> body2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> positive2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> negative2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> boundary2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> congruence2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> confidence2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> signal2Buffer = makeEmptyPrivateBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyReductionBuffer = makeEmptyPrivateBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> weightedReductionBuffer = makeEmptyPrivateBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> finalReductionBuffer = makeEmptyPrivateBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> candidate2ReductionBuffer = makeEmptyPrivateBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    id<MTLBuffer> accumulateUniformBuffer = makeSharedBuffer(&accumulateUniforms, 1u);
    id<MTLBuffer> cellUniformBuffer = makeSharedBuffer(&cellUniforms, 1u);
    id<MTLBuffer> percentile50Buffer = makeSharedBuffer(&percentile50, 1u);
    id<MTLBuffer> percentile35Buffer = makeSharedBuffer(&percentile35, 1u);
    if (sourceTexture == nil || occupancyCountsBuffer == nil || sumRBuffer == nil ||
        sumGBuffer == nil || sumBBuffer == nil || sumYBuffer == nil || sumMaxBuffer == nil ||
        sumMinBuffer == nil || sumNeutralityBuffer == nil || occupancyBuffer == nil ||
        meanRBuffer == nil || meanGBuffer == nil || meanBBuffer == nil ||
        carrierYBuffer == nil || carrierMaxBuffer == nil || carrierMinBuffer == nil ||
        neutralityBuffer == nil || occupancyNormBuffer == nil || tempBuffer == nil ||
        bodyBuffer == nil || viewerBodyRawBuffer == nil || rawSignalBuffer == nil || positiveBuffer == nil ||
        negativeBuffer == nil || boundaryBuffer == nil || congruenceBuffer == nil ||
        confidenceBuffer == nil || signalBuffer == nil || bodyCoreBuffer == nil ||
        bodyContextBuffer == nil || retinexBodyBuffer == nil || dogLowBuffer == nil ||
        body2RawBuffer == nil || positive2RawBuffer == nil || negative2RawBuffer == nil ||
        confidence2RawBuffer == nil || agreement2RawBuffer == nil || body2Buffer == nil ||
        positive2Buffer == nil || negative2Buffer == nil || boundary2Buffer == nil ||
        congruence2Buffer == nil || confidence2Buffer == nil || signal2Buffer == nil ||
        bodyReductionBuffer == nil || weightedReductionBuffer == nil || finalReductionBuffer == nil ||
        candidate2ReductionBuffer == nil || rasterUniformBuffer == nil ||
        accumulateUniformBuffer == nil || cellUniformBuffer == nil ||
        percentile50Buffer == nil || percentile35Buffer == nil) {
      if (error) *error = localError.empty() ? "metal-iosurface-gloss-field-allocation-failed" : localError;
      return false;
    }
    const std::array<id<MTLBuffer>, 47> buffersToClear = {
        occupancyCountsBuffer, sumRBuffer, sumGBuffer, sumBBuffer, sumYBuffer,
        sumMaxBuffer, sumMinBuffer, sumNeutralityBuffer, occupancyBuffer, meanRBuffer,
        meanGBuffer, meanBBuffer, carrierYBuffer, carrierMaxBuffer, carrierMinBuffer,
        neutralityBuffer, occupancyNormBuffer, tempBuffer, bodyBuffer, viewerBodyRawBuffer, rawSignalBuffer,
        positiveBuffer, negativeBuffer, boundaryBuffer, congruenceBuffer, confidenceBuffer,
        signalBuffer, bodyCoreBuffer, bodyContextBuffer, retinexBodyBuffer, dogLowBuffer,
        body2RawBuffer, positive2RawBuffer, negative2RawBuffer, confidence2RawBuffer,
        agreement2RawBuffer, body2Buffer, positive2Buffer, negative2Buffer, boundary2Buffer,
        congruence2Buffer, confidence2Buffer, signal2Buffer, bodyReductionBuffer,
        weightedReductionBuffer, finalReductionBuffer, candidate2ReductionBuffer};
    for (id<MTLBuffer> buffer : buffersToClear) {
      if (buffer != nil && !clearBufferOnDevice(buffer, &localError)) {
        if (error) *error = localError;
        return false;
      }
    }

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-iosurface-gloss-field-command-buffer-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-iosurface-gloss-field-accumulate-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterGlossFieldAccumulateTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:occupancyCountsBuffer offset:0 atIndex:0];
    [encoder setBuffer:sumRBuffer offset:0 atIndex:1];
    [encoder setBuffer:sumGBuffer offset:0 atIndex:2];
    [encoder setBuffer:sumBBuffer offset:0 atIndex:3];
    [encoder setBuffer:sumYBuffer offset:0 atIndex:4];
    [encoder setBuffer:sumMaxBuffer offset:0 atIndex:5];
    [encoder setBuffer:sumMinBuffer offset:0 atIndex:6];
    [encoder setBuffer:sumNeutralityBuffer offset:0 atIndex:7];
    [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:8];
    [encoder setBuffer:accumulateUniformBuffer offset:0 atIndex:9];
    NSUInteger threads = ctx.rasterGlossFieldAccumulateTexturePipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0) threads = 64;
    threads = std::min<NSUInteger>(threads, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    const auto clearBodyReduction = [&]() -> bool {
      return clearBufferOnDevice(bodyReductionBuffer, &localError);
    };
    const auto clearWeightedReduction = [&]() -> bool {
      return clearBufferOnDevice(weightedReductionBuffer, &localError);
    };
    const auto clearFinalReduction = [&]() -> bool {
      return clearBufferOnDevice(finalReductionBuffer, &localError);
    };
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
    if (!clearBodyReduction() ||
        !runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyBuffer, occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldBlurPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, tempBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!copyBufferOnDevice(tempBuffer, occupancyNormBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBodyReduction() ||
        !runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyNormBuffer, occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
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
      return copyBufferOnDevice(tempBuffer, buffer, cellCount * sizeof(float), &localError);
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
    if (!copyBufferOnDevice(bodyBuffer, viewerBodyRawBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBodyReduction() ||
        !runComputeBuffers(ctx.glossFieldRawSignalPipeline,
                           std::array<id<MTLBuffer>, 6>{occupancyBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        rawSignalBuffer,
                                                        bodyReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearWeightedReduction() ||
        !runComputeBuffers(ctx.glossFieldWeightedSignalPipeline,
                           std::array<id<MTLBuffer>, 11>{occupancyNormBuffer,
                                                         bodyBuffer,
                                                         rawSignalBuffer,
                                                         positiveBuffer,
                                                         negativeBuffer,
                                                         boundaryBuffer,
                                                         congruenceBuffer,
                                                         confidenceBuffer,
                                                         signalBuffer,
                                                         weightedReductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearFinalReduction() ||
        !runComputeBuffers(ctx.glossFieldMergeMaxBitsPipeline,
                           std::array<id<MTLBuffer>, 3>{bodyReductionBuffer,
                                                        weightedReductionBuffer,
                                                        finalReductionBuffer},
                           1u,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!runComputeBuffers(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{bodyBuffer,
                                                        signalBuffer,
                                                        positiveBuffer,
                                                        negativeBuffer,
                                                        boundaryBuffer,
                                                        finalReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    if (!runComputeBuffers(ctx.glossFieldLocalPercentilePipeline,
                           std::array<id<MTLBuffer>, 5>{carrierYBuffer,
                                                        occupancyBuffer,
                                                        bodyCoreBuffer,
                                                        cellUniformBuffer,
                                                        percentile50Buffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!copyBufferOnDevice(carrierYBuffer, bodyContextBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    for (int i = 0; i < std::max(2, analysisRadius * 2); ++i) {
      if (!blurInPlace(bodyContextBuffer)) {
        if (error) *error = localError;
        return false;
      }
    }
    if (!runComputeBuffers(ctx.glossFieldLocalPercentilePipeline,
                           std::array<id<MTLBuffer>, 5>{carrierYBuffer,
                                                        occupancyBuffer,
                                                        retinexBodyBuffer,
                                                        cellUniformBuffer,
                                                        percentile35Buffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!copyBufferOnDevice(carrierYBuffer, dogLowBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    for (int i = 0; i < std::max(1, analysisRadius / 2); ++i) {
      if (!blurInPlace(dogLowBuffer)) {
        if (error) *error = localError;
        return false;
      }
    }
    if (!runComputeBuffers(ctx.glossFieldCandidate2RawPipeline,
                           std::array<id<MTLBuffer>, 18>{occupancyBuffer,
                                                         occupancyNormBuffer,
                                                         meanRBuffer,
                                                         meanGBuffer,
                                                         meanBBuffer,
                                                         carrierYBuffer,
                                                         viewerBodyRawBuffer,
                                                         bodyCoreBuffer,
                                                         bodyContextBuffer,
                                                         retinexBodyBuffer,
                                                         dogLowBuffer,
                                                         bodyContextBuffer,
                                                         body2RawBuffer,
                                                         positive2RawBuffer,
                                                         negative2RawBuffer,
                                                         confidence2RawBuffer,
                                                         agreement2RawBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBufferOnDevice(candidate2ReductionBuffer, &localError) ||
        !runComputeBuffers(ctx.glossFieldAssembleUnifiedPipeline,
                           std::array<id<MTLBuffer>, 14>{body2RawBuffer,
                                                         positive2RawBuffer,
                                                         negative2RawBuffer,
                                                         confidence2RawBuffer,
                                                         agreement2RawBuffer,
                                                         body2Buffer,
                                                         signal2Buffer,
                                                         positive2Buffer,
                                                         negative2Buffer,
                                                         boundary2Buffer,
                                                         congruence2Buffer,
                                                         confidence2Buffer,
                                                         candidate2ReductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!runComputeBuffers(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{body2Buffer,
                                                        signal2Buffer,
                                                        positive2Buffer,
                                                        negative2Buffer,
                                                        boundary2Buffer,
                                                        candidate2ReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    GlossFieldResidentRecord record{};
    record.gridWidth = gridWidth;
    record.gridHeight = gridHeight;
    record.builtSerial = buildSerial;
    record.occupancy = occupancyBuffer;
    record.meanR = meanRBuffer;
    record.meanG = meanGBuffer;
    record.meanB = meanBBuffer;
    record.carrierY = carrierYBuffer;
    record.carrierMax = carrierMaxBuffer;
    record.carrierMin = carrierMinBuffer;
    record.neutrality = neutralityBuffer;
    record.occupancyNorm = occupancyNormBuffer;
    record.body = bodyBuffer;
    record.rawSignal = rawSignalBuffer;
    record.positive = positiveBuffer;
    record.negative = negativeBuffer;
    record.boundary = boundaryBuffer;
    record.congruence = congruenceBuffer;
    record.confidence = confidenceBuffer;
    record.signal = signalBuffer;
    record.body2 = body2Buffer;
    record.positive2 = positive2Buffer;
    record.negative2 = negative2Buffer;
    record.boundary2 = boundary2Buffer;
    record.congruence2 = congruence2Buffer;
    record.confidence2 = confidence2Buffer;
    record.signal2 = signal2Buffer;
    record.temp = tempBuffer;
    record.reduction = finalReductionBuffer;
    {
      std::lock_guard<std::mutex> lock(glossFieldRegistryMutex());
      if (cache->cacheId == 0) cache->cacheId = nextGlossFieldCacheId();
      glossFieldRegistry()[cache->cacheId] = record;
    }
    cache->gridWidth = gridWidth;
    cache->gridHeight = gridHeight;
    cache->builtSerial = buildSerial;
    cache->available = true;
  }
  return true;
}

bool renderGlossFieldSurfaceFromCache(const GlossFieldCache& cache,
                                      const GlossFieldSurfaceRequest& surfaceRequest,
                                      uint32_t outputSurfaceId,
                                      int outputSurfaceWidth,
                                      int outputSurfaceHeight,
                                      int outputSurfacePixelFormat,
                                      std::string* error) {
  if (error) error->clear();
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  if (!cache.available || cache.cacheId == 0 || cache.gridWidth <= 0 || cache.gridHeight <= 0 ||
      outputSurfaceId == 0 || outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0) {
    if (error) *error = "invalid-metal-gloss-field-surface-request";
    return false;
  }
  const bool useCandidate2 = surfaceRequest.algorithm != 0;
  if (surfaceRequest.algorithm < 0 || surfaceRequest.algorithm > 1) {
    if (error) *error = "unsupported-metal-gloss-field-surface-algorithm";
    return false;
  }
  if (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1) {
    if (error) *error = "unsupported-metal-gloss-field-surface-format";
    return false;
  }
  MetalContext& ctx = context();
  if (ctx.glossFieldSurfaceRenderPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-gloss-field-surface");
    return false;
  }

  GlossFieldResidentRecord record{};
  {
    std::lock_guard<std::mutex> lock(glossFieldRegistryMutex());
    auto it = glossFieldRegistry().find(cache.cacheId);
    if (it == glossFieldRegistry().end()) {
      if (error) *error = "metal-gloss-field-cache-missing";
      return false;
    }
    record = it->second;
  }
  if (record.gridWidth != cache.gridWidth || record.gridHeight != cache.gridHeight ||
      record.meanR == nil || record.meanG == nil || record.meanB == nil ||
      record.carrierY == nil || record.carrierMax == nil || record.carrierMin == nil ||
      record.neutrality == nil || record.body == nil || record.positive == nil ||
      record.negative == nil || record.boundary == nil || record.congruence == nil ||
      record.confidence == nil || record.signal == nil ||
      (useCandidate2 && (record.body2 == nil || record.positive2 == nil ||
                         record.negative2 == nil || record.boundary2 == nil ||
                         record.congruence2 == nil || record.confidence2 == nil ||
                         record.signal2 == nil))) {
    if (error) *error = "metal-gloss-field-cache-incomplete";
    return false;
  }

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() &&
        it->second.width == outputSurfaceWidth &&
        it->second.height == outputSurfaceHeight &&
        it->second.pixelFormat == outputSurfacePixelFormat) {
      outputTexture = it->second.texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-gloss-field-output-surface-missing";
    return false;
  }

  GlossFieldSurfaceUniforms uniforms{};
  uniforms.gridWidth = cache.gridWidth;
  uniforms.gridHeight = cache.gridHeight;
  uniforms.surfaceWidth = outputSurfaceWidth;
  uniforms.surfaceHeight = outputSurfaceHeight;
  uniforms.algorithm = surfaceRequest.algorithm;
  uniforms.colorMode = std::clamp(surfaceRequest.colorMode, 0, 1);
  uniforms.debugMode = std::clamp(surfaceRequest.debugMode, 0, 4);
  uniforms.diagnosticMode = std::clamp(surfaceRequest.diagnosticMode, 0, 2);
  uniforms.colorSaturation = std::clamp(surfaceRequest.colorSaturation, 0.8f, 6.0f);
  uniforms.glossBodyOpacity = std::clamp(surfaceRequest.glossBodyOpacity, 0.0f, 1.0f);
  uniforms.glossHighlightOpacity = std::clamp(surfaceRequest.glossHighlightOpacity, 0.0f, 1.0f);
  uniforms.glossLiftScale = std::max(0.01f, surfaceRequest.glossLiftScale);

  @autoreleasepool {
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    if (uniformBuffer == nil) {
      if (error) *error = "metal-gloss-field-surface-uniform-allocation-failed";
      return false;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-gloss-field-surface-command-buffer-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-field-surface-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossFieldSurfaceRenderPipeline];
    [encoder setTexture:outputTexture atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:0];
    [encoder setBuffer:record.meanG offset:0 atIndex:1];
    [encoder setBuffer:record.meanB offset:0 atIndex:2];
    [encoder setBuffer:record.carrierY offset:0 atIndex:3];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:5];
    [encoder setBuffer:record.neutrality offset:0 atIndex:6];
    [encoder setBuffer:(useCandidate2 ? record.body2 : record.body) offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.positive2 : record.positive) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.negative2 : record.negative) offset:0 atIndex:9];
    [encoder setBuffer:(useCandidate2 ? record.boundary2 : record.boundary) offset:0 atIndex:10];
    [encoder setBuffer:(useCandidate2 ? record.congruence2 : record.congruence) offset:0 atIndex:11];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:12];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:13];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:14];
    NSUInteger width = ctx.glossFieldSurfaceRenderPipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::max<NSUInteger>(
        1,
        std::min<NSUInteger>(16, static_cast<NSUInteger>(std::sqrt(static_cast<double>(width)))));
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                         static_cast<NSUInteger>(outputSurfaceHeight),
                                         1)
       threadsPerThreadgroup:MTLSizeMake(width, width, 1)];
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

bool renderGlossProjectionSurfaceFromCache(const GlossFieldCache& cache,
                                           const GlossProjectionSurfaceRequest& projectionRequest,
                                           uint32_t outputSurfaceId,
                                           int outputSurfaceWidth,
                                           int outputSurfaceHeight,
                                           int outputSurfacePixelFormat,
                                           std::string* error) {
  if (error) error->clear();
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  if (!cache.available || cache.cacheId == 0 || cache.gridWidth <= 0 || cache.gridHeight <= 0 ||
      outputSurfaceId == 0 || outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0) {
    if (error) *error = "invalid-metal-gloss-projection-surface-request";
    return false;
  }
  const bool useCandidate2 = projectionRequest.algorithm != 0;
  if (projectionRequest.algorithm < 0 || projectionRequest.algorithm > 1) {
    if (error) *error = "unsupported-metal-gloss-projection-surface-algorithm";
    return false;
  }
  if (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1) {
    if (error) *error = "unsupported-metal-gloss-projection-surface-format";
    return false;
  }
  MetalContext& ctx = context();
  if (ctx.glossProjectionSurfaceSelectPipeline == nil ||
      ctx.glossProjectionSurfaceShadePipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-gloss-projection-surface");
    return false;
  }

  GlossFieldResidentRecord record{};
  {
    std::lock_guard<std::mutex> lock(glossFieldRegistryMutex());
    auto it = glossFieldRegistry().find(cache.cacheId);
    if (it == glossFieldRegistry().end()) {
      if (error) *error = "metal-gloss-field-cache-missing";
      return false;
    }
    record = it->second;
  }
  if (record.gridWidth != cache.gridWidth || record.gridHeight != cache.gridHeight ||
      record.meanR == nil || record.meanG == nil || record.meanB == nil ||
      record.carrierY == nil || record.carrierMax == nil || record.carrierMin == nil ||
      record.neutrality == nil || record.body == nil || record.positive == nil ||
      record.negative == nil || record.boundary == nil || record.congruence == nil ||
      record.confidence == nil || record.signal == nil ||
      (useCandidate2 && (record.body2 == nil || record.positive2 == nil ||
                         record.negative2 == nil || record.boundary2 == nil ||
                         record.congruence2 == nil || record.confidence2 == nil ||
                         record.signal2 == nil))) {
    if (error) *error = "metal-gloss-field-cache-incomplete";
    return false;
  }

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() &&
        it->second.width == outputSurfaceWidth &&
        it->second.height == outputSurfaceHeight &&
        it->second.pixelFormat == outputSurfacePixelFormat) {
      outputTexture = it->second.texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-gloss-projection-output-surface-missing";
    return false;
  }
  if (!clearPlotSurface(outputSurfaceId,
                        outputSurfaceWidth,
                        outputSurfaceHeight,
                        outputSurfacePixelFormat,
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f,
                        &localError)) {
    if (error) *error = localError.empty() ? "metal-gloss-projection-clear-failed" : localError;
    return false;
  }

  GlossProjectionSurfaceUniforms uniforms{};
  uniforms.gridWidth = cache.gridWidth;
  uniforms.gridHeight = cache.gridHeight;
  uniforms.surfaceWidth = outputSurfaceWidth;
  uniforms.surfaceHeight = outputSurfaceHeight;
  uniforms.algorithm = projectionRequest.algorithm;
  uniforms.colorMode = std::clamp(projectionRequest.colorMode, 0, 1);
  uniforms.debugMode = std::clamp(projectionRequest.debugMode, 0, 4);
  uniforms.diagnosticMode = std::clamp(projectionRequest.diagnosticMode, 0, 2);
  uniforms.sourceAspect = std::clamp(projectionRequest.sourceAspect, 0.25f, 4.0f);
  uniforms.colorSaturation = std::clamp(projectionRequest.colorSaturation, 0.8f, 6.0f);
  uniforms.glossBodyOpacity = std::clamp(projectionRequest.glossBodyOpacity, 0.0f, 1.0f);
  uniforms.glossHighlightOpacity = std::clamp(projectionRequest.glossHighlightOpacity, 0.0f, 1.0f);
  uniforms.glossLiftScale = std::max(0.01f, projectionRequest.glossLiftScale);
  uniforms.pointRadiusPixels = std::clamp(projectionRequest.pointRadiusPixels, 1.0f, 6.0f);
  std::copy(projectionRequest.modelView, projectionRequest.modelView + 16, uniforms.modelView);
  std::copy(projectionRequest.projection, projectionRequest.projection + 16, uniforms.projection);

  @autoreleasepool {
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    const size_t selectionBytes =
        static_cast<size_t>(std::max(outputSurfaceWidth, 0)) *
        static_cast<size_t>(std::max(outputSurfaceHeight, 0)) *
        sizeof(uint32_t);
    id<MTLBuffer> selectionBuffer = makeEmptyPrivateBuffer(selectionBytes);
    if (uniformBuffer == nil || selectionBuffer == nil) {
      if (error) *error = "metal-gloss-projection-surface-allocation-failed";
      return false;
    }
    if (!clearBufferOnDevice(selectionBuffer, &localError)) {
      if (error) *error = localError.empty() ? "metal-gloss-projection-selection-clear-failed" : localError;
      return false;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-gloss-projection-surface-command-buffer-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-projection-surface-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossProjectionSurfaceSelectPipeline];
    [encoder setBuffer:selectionBuffer offset:0 atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:1];
    [encoder setBuffer:record.meanG offset:0 atIndex:2];
    [encoder setBuffer:record.meanB offset:0 atIndex:3];
    [encoder setBuffer:record.carrierY offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:5];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:6];
    [encoder setBuffer:record.neutrality offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:9];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:10];
    NSUInteger width = ctx.glossProjectionSurfaceSelectPipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    const NSUInteger cellCount =
        static_cast<NSUInteger>(std::max(cache.gridWidth, 0)) *
        static_cast<NSUInteger>(std::max(cache.gridHeight, 0));
    [encoder dispatchThreads:MTLSizeMake(cellCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];

    encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-projection-surface-shade-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossProjectionSurfaceShadePipeline];
    [encoder setTexture:outputTexture atIndex:0];
    [encoder setBuffer:selectionBuffer offset:0 atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:1];
    [encoder setBuffer:record.meanG offset:0 atIndex:2];
    [encoder setBuffer:record.meanB offset:0 atIndex:3];
    [encoder setBuffer:record.carrierY offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:5];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:6];
    [encoder setBuffer:record.neutrality offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.body2 : record.body) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.positive2 : record.positive) offset:0 atIndex:9];
    [encoder setBuffer:(useCandidate2 ? record.negative2 : record.negative) offset:0 atIndex:10];
    [encoder setBuffer:(useCandidate2 ? record.boundary2 : record.boundary) offset:0 atIndex:11];
    [encoder setBuffer:(useCandidate2 ? record.congruence2 : record.congruence) offset:0 atIndex:12];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:13];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:14];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:15];
    NSUInteger shadeWidth = ctx.glossProjectionSurfaceShadePipeline.maxTotalThreadsPerThreadgroup;
    if (shadeWidth == 0) shadeWidth = 64;
    shadeWidth = std::max<NSUInteger>(
        1,
        std::min<NSUInteger>(16, static_cast<NSUInteger>(std::sqrt(static_cast<double>(shadeWidth)))));
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                         static_cast<NSUInteger>(outputSurfaceHeight),
                                         1)
       threadsPerThreadgroup:MTLSizeMake(shadeWidth, shadeWidth, 1)];
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

}  // namespace ChromaspaceMetal
