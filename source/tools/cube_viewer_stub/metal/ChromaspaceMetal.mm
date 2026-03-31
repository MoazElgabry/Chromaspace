#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <cstring>
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
  id<MTLComputePipelineState> inputSamplePipeline = nil;
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
};

struct InputUniforms {
  int pointCount;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
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
};

struct InputUniforms {
  int pointCount;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
};

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
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

kernel void inputKernel(const device packed_float3* inputVals [[buffer(0)]],
                        device packed_float3* vertVals [[buffer(1)]],
                        device float4* colorVals [[buffer(2)]],
                        constant InputUniforms& u [[buffer(3)]],
                        uint index [[thread_position_in_grid]]) {
  uint total = uint(max(u.pointCount, 0));
  if (index >= total) return;
  packed_float3 p = inputVals[index];
  float r = p.x;
  float g = p.y;
  float b = p.z;
  bool overflowPoint = pointOverflowsCube(r, g, b);
  float plotR = (u.showOverflow != 0) ? r : clamp01(r);
  float plotG = (u.showOverflow != 0) ? g : clamp01(g);
  float plotB = (u.showOverflow != 0) ? b : clamp01(b);
  float3 pos = mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow);
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
  }
  bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint);
  colorVals[index] = float4(cr, cg, cb,
                            luminanceAwareAlpha(overflowHighlighted ? 0.95 : 0.72,
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
  if (rawPoints.size() < pointCount * 3u) {
    if (error) *error = "Input Metal raw point buffer is undersized.";
    return false;
  }

  MetalContext& ctx = context();
  InputUniforms uniforms{};
  uniforms.pointCount = request.pointCount;
  uniforms.showOverflow = request.remap.showOverflow;
  uniforms.highlightOverflow = request.remap.highlightOverflow;
  uniforms.plotMode = request.remap.plotMode;
  uniforms.circularHsl = request.remap.circularHsl;
  uniforms.circularHsv = request.remap.circularHsv;
  uniforms.normConeNormalized = request.remap.normConeNormalized;
  uniforms.pointAlphaScale = request.pointAlphaScale;
  uniforms.denseAlphaBias = request.denseAlphaBias;
  uniforms.colorSaturation = request.colorSaturation;

  id<MTLBuffer> inputBuffer = makeSharedBuffer(reinterpret_cast<const PackedFloat3*>(rawPoints.data()), pointCount);
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

}  // namespace ChromaspaceMetal
