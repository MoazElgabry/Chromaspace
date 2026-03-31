#include "ChromaspaceCuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#if defined(_WIN32)
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

namespace ChromaspaceCuda {
namespace {

struct CudaContext {
  bool initAttempted = false;
  bool ready = false;
  bool interopReady = false;
  int device = -1;
  std::string deviceName;
  std::string reason;
};

struct OverlayKernelUniforms {
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

struct InputKernelUniforms {
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

struct InputSampleKernelUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct CacheImpl {
  cudaGraphicsResource* vertsResource = nullptr;
  cudaGraphicsResource* colorsResource = nullptr;
  GLuint registeredVerts = 0;
  GLuint registeredColors = 0;
  size_t pointCapacity = 0;
  float* deviceInput = nullptr;
  size_t inputCapacityFloats = 0;
  unsigned int* deviceBounds = nullptr;
};

struct SampleCacheImpl {
  cudaGraphicsResource* vertsResource = nullptr;
  cudaGraphicsResource* colorsResource = nullptr;
  GLuint registeredVerts = 0;
  GLuint registeredColors = 0;
  size_t pointCapacity = 0;
};

CudaContext& context() {
  static CudaContext ctx;
  return ctx;
}

const char* errorString(cudaError_t err) {
  return cudaGetErrorString(err);
}

float floatFromOrderedUint(unsigned int ordered) {
  const unsigned int bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool ensureContext(std::string* error) {
  static std::once_flag once;
  CudaContext& ctx = context();
  std::call_once(once, []() {
    CudaContext& c = context();
    c.initAttempted = true;
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaGetDeviceCount failed: ") + errorString(err);
      return;
    }
    if (deviceCount <= 0) {
      c.reason = "No CUDA devices found.";
      return;
    }

    unsigned int glCount = 0;
    int glDevices[8] = {};
    err = cudaGLGetDevices(&glCount, glDevices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess || glCount == 0) {
      c.reason = std::string("CUDA-GL interop probe failed: ") + errorString(err == cudaSuccess ? cudaErrorUnknown : err);
      return;
    }

    c.device = glDevices[0];
    err = cudaSetDevice(c.device);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaSetDevice failed: ") + errorString(err);
      return;
    }
    err = cudaFree(0);
    if (err != cudaSuccess) {
      c.reason = std::string("CUDA warm-up failed: ") + errorString(err);
      return;
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, c.device);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaGetDeviceProperties failed: ") + errorString(err);
      return;
    }
    c.deviceName = prop.name;
    c.interopReady = true;
    c.ready = true;
  });

  if (!ctx.ready && error) *error = ctx.reason;
  return ctx.ready;
}

template <typename CacheT>
CacheImpl* ensureImpl(CacheT* cache) {
  if (!cache) return nullptr;
  if (!cache->internal) cache->internal = new CacheImpl();
  return reinterpret_cast<CacheImpl*>(cache->internal);
}

void releaseImpl(CacheImpl* impl) {
  if (!impl) return;
  if (impl->vertsResource) cudaGraphicsUnregisterResource(impl->vertsResource);
  if (impl->colorsResource) cudaGraphicsUnregisterResource(impl->colorsResource);
  if (impl->deviceInput) cudaFree(impl->deviceInput);
  if (impl->deviceBounds) cudaFree(impl->deviceBounds);
  delete impl;
}

void releaseSampleImpl(SampleCacheImpl* impl) {
  if (!impl) return;
  if (impl->vertsResource) cudaGraphicsUnregisterResource(impl->vertsResource);
  if (impl->colorsResource) cudaGraphicsUnregisterResource(impl->colorsResource);
  delete impl;
}

template <typename CacheT>
void releaseCache(CacheT* cache) {
  if (!cache) return;
  releaseImpl(reinterpret_cast<CacheImpl*>(cache->internal));
  cache->internal = nullptr;
  cache->builtSerial = 0;
  cache->pointCount = 0;
  cache->available = false;
}

template <typename CacheT>
SampleCacheImpl* ensureSampleImpl(CacheT* cache) {
  if (!cache) return nullptr;
  if (!cache->internal) cache->internal = new SampleCacheImpl();
  return reinterpret_cast<SampleCacheImpl*>(cache->internal);
}

template <typename CacheT>
void releaseSampleCache(CacheT* cache) {
  if (!cache) return;
  releaseSampleImpl(reinterpret_cast<SampleCacheImpl*>(cache->internal));
  cache->internal = nullptr;
  cache->builtSerial = 0;
  cache->pointCount = 0;
  cache->available = false;
}

bool ensureRegistered(GLuint verts, GLuint colors, size_t pointCapacity, CacheImpl* impl, std::string* error) {
  if (!impl || verts == 0 || colors == 0) {
    if (error) *error = "Missing GL buffers for CUDA interop.";
    return false;
  }
  if (impl->registeredVerts == verts && impl->registeredColors == colors &&
      impl->pointCapacity == pointCapacity && impl->vertsResource && impl->colorsResource) {
    return true;
  }
  if (impl->vertsResource) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
  }
  if (impl->colorsResource) {
    cudaGraphicsUnregisterResource(impl->colorsResource);
    impl->colorsResource = nullptr;
  }

  cudaError_t err = cudaGraphicsGLRegisterBuffer(&impl->vertsResource, verts, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to register CUDA verts buffer: ") + errorString(err);
    return false;
  }
  err = cudaGraphicsGLRegisterBuffer(&impl->colorsResource, colors, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
    if (error) *error = std::string("Failed to register CUDA colors buffer: ") + errorString(err);
    return false;
  }
  impl->registeredVerts = verts;
  impl->registeredColors = colors;
  impl->pointCapacity = pointCapacity;
  return true;
}

bool ensureSampleRegistered(GLuint verts, GLuint colors, size_t pointCapacity, SampleCacheImpl* impl, std::string* error) {
  if (!impl || verts == 0 || colors == 0) {
    if (error) *error = "Missing GL sample buffers for CUDA interop.";
    return false;
  }
  if (impl->registeredVerts == verts && impl->registeredColors == colors &&
      impl->pointCapacity == pointCapacity && impl->vertsResource && impl->colorsResource) {
    return true;
  }
  if (impl->vertsResource) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
  }
  if (impl->colorsResource) {
    cudaGraphicsUnregisterResource(impl->colorsResource);
    impl->colorsResource = nullptr;
  }

  cudaError_t err = cudaGraphicsGLRegisterBuffer(&impl->vertsResource, verts, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to register CUDA sampled verts buffer: ") + errorString(err);
    return false;
  }
  err = cudaGraphicsGLRegisterBuffer(&impl->colorsResource, colors, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
    if (error) *error = std::string("Failed to register CUDA sampled colors buffer: ") + errorString(err);
    return false;
  }
  impl->registeredVerts = verts;
  impl->registeredColors = colors;
  impl->pointCapacity = pointCapacity;
  return true;
}

bool ensureInputCapacity(CacheImpl* impl, size_t floatCount, std::string* error) {
  if (!impl) return false;
  if (floatCount <= impl->inputCapacityFloats && impl->deviceInput != nullptr) return true;
  if (impl->deviceInput) {
    cudaFree(impl->deviceInput);
    impl->deviceInput = nullptr;
    impl->inputCapacityFloats = 0;
  }
  cudaError_t err = cudaMalloc(&impl->deviceInput, floatCount * sizeof(float));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA input buffer: ") + errorString(err);
    return false;
  }
  impl->inputCapacityFloats = floatCount;
  return true;
}

bool ensureBoundsCapacity(CacheImpl* impl, std::string* error) {
  if (!impl) return false;
  if (impl->deviceBounds != nullptr) return true;
  cudaError_t err = cudaMalloc(&impl->deviceBounds, 6u * sizeof(unsigned int));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA bounds buffer: ") + errorString(err);
    return false;
  }
  return true;
}

inline __device__ unsigned int orderedUintFromFloat(float value) {
  unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

inline __device__ float clamp01(float v) {
  return fminf(fmaxf(v, 0.0f), 1.0f);
}

inline __device__ float wrapHue01(float h) {
  h = fmodf(h, 1.0f);
  if (h < 0.0f) h += 1.0f;
  return h;
}

inline __device__ float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6f) return 0.0f;
  float h = 0.0f;
  if (cMax == r) {
    h = fmodf((g - b) / delta, 6.0f);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0f;
  } else {
    h = ((r - g) / delta) + 4.0f;
  }
  return wrapHue01(h / 6.0f);
}

inline __device__ void mapPlotPosition(float r, float g, float b, int plotMode, int circularHsl, int circularHsv, int normConeNormalized, int showOverflow,
                                       float* outX, float* outY, float* outZ) {
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kPi = 3.14159265358979323846f;
  if (plotMode == 1) {
    const float cMax = fmaxf(r, fmaxf(g, b));
    const float cMin = fminf(r, fminf(g, b));
    const float delta = cMax - cMin;
    const float l = 0.5f * (cMax + cMin);
    float h = rawRgbHue01(r, g, b, cMax, delta);
    const float satDenom = 1.0f - fabsf(2.0f * l - 1.0f);
    if (delta > 1e-6f && satDenom < 0.0f) h = wrapHue01(h + 0.5f);
    const float angle = h * kTau;
    float radius = delta;
    if (circularHsl != 0) {
      float denom = satDenom;
      if (fabsf(denom) <= 1e-6f) denom = (denom < 0.0f) ? -1e-6f : 1e-6f;
      radius = fabsf(delta / denom);
    }
    *outX = cosf(angle) * radius;
    *outY = l * 2.0f - 1.0f;
    *outZ = sinf(angle) * radius;
    return;
  }
  if (plotMode == 2) {
    const float cMax = fmaxf(r, fmaxf(g, b));
    if (circularHsv != 0) {
      const float cMin = fminf(r, fminf(g, b));
      const float delta = cMax - cMin;
      const float h = rawRgbHue01(r, g, b, cMax, delta);
      const float sat = (delta > 1e-6f && cMax > 1e-6f) ? (delta / cMax) : 0.0f;
      const float angle = h * kTau;
      *outX = cosf(angle) * sat;
      *outY = cMax * 2.0f - 1.0f;
      *outZ = sinf(angle) * sat;
      return;
    }
    *outX = r - 0.5f * g - 0.5f * b;
    *outY = cMax * 2.0f - 1.0f;
    *outZ = 0.8660254037844386f * (g - b);
    return;
  }
  if (plotMode == 3) {
    const float rotX = r * 0.81649658f + g * -0.40824829f + b * -0.40824829f;
    const float rotY = g * 0.70710678f + b * -0.70710678f;
    const float rotZ = r * 0.57735027f + g * 0.57735027f + b * 0.57735027f;
    const float azimuth = atan2f(rotY, rotX);
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    const float wrappedHue = azimuth < 0.0f ? azimuth + kTau : azimuth;
    const float polar = atanf(sqrtf(rotX * rotX + rotY * rotY) / fmaxf(rotZ, 1e-8f));
    const float c = polar * 1.0467733744265997f;
    const float l = radius3 * 0.5773502691896258f;
    const float polarScaled = c * 0.9553166181245093f;
    const float radial = l * sinf(polarScaled) / 0.816496580927726f;
    *outX = cosf(wrappedHue) * radial;
    *outY = l * 2.0f - 1.0f;
    *outZ = sinf(wrappedHue) * radial;
    return;
  }
  if (plotMode == 4 || plotMode == 5) {
    const bool jpOverflow = (showOverflow != 0 && plotMode == 5);
    const float rr = jpOverflow ? r : clamp01(r);
    const float gg = jpOverflow ? g : clamp01(g);
    const float bb = jpOverflow ? b : clamp01(b);
    const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
    const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
    const float rotZ = 0.57735026919f * (rr + gg + bb);
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float hue = atan2f(rotY, rotX);
    if (hue < 0.0f) hue += kTau;
    const float polar = atan2f(sqrtf(rotX * rotX + rotY * rotY), rotZ);
    float magnitude = 0.0f;
    if (plotMode == 4) {
      magnitude = fminf(fmaxf(radius3 * 0.576f, 0.0f), 1.0f);
    } else {
      const float kAsinInvSqrt2 = asinf(1.0f / sqrtf(2.0f));
      const float kAsinInvSqrt3 = asinf(1.0f / sqrtf(3.0f));
      const float kHueCoef1 = 1.0f / (2.0f - (kAsinInvSqrt2 / kAsinInvSqrt3));
      const float huecoef2 = 2.0f * polar * sinf((2.0f * kPi / 3.0f) - fmodf(hue, kPi / 3.0f)) / sqrtf(3.0f);
      const float huemag = ((acosf(cosf(3.0f * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0f)) * huecoef2;
      const float satmag = sinf(huemag + kAsinInvSqrt3);
      magnitude = radius3 * satmag;
      if (jpOverflow && magnitude < 0.0f) {
        magnitude = -magnitude;
        hue += kPi;
        if (hue >= kTau) hue -= kTau;
      }
      magnitude = jpOverflow ? magnitude
                             : fminf(fmaxf(magnitude, 0.0f), 1.0f);
    }
    const float phiNorm = jpOverflow ? fmaxf(polar / 0.9553166181245093f, 0.0f)
                                     : fminf(fmaxf(polar / 0.9553166181245093f, 0.0f), 1.0f);
    const float phi = phiNorm * 0.9553166181245093f;
    const float radial = magnitude * sinf(phi);
    *outX = cosf(hue) * radial;
    *outY = magnitude * cosf(phi) * 2.0f - 1.0f;
    *outZ = sinf(hue) * radial;
    return;
  }
  if (plotMode == 6) {
    const bool normConeOverflow = (showOverflow != 0 && plotMode == 6);
    const float rr = normConeOverflow ? r : fminf(fmaxf(r, 0.0f), 1.0f);
    const float gg = normConeOverflow ? g : fminf(fmaxf(g, 0.0f), 1.0f);
    const float bb = normConeOverflow ? b : fminf(fmaxf(b, 0.0f), 1.0f);
    const float maxRgb = fmaxf(rr, fmaxf(gg, bb));
    const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
    const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
    const float rotZ = 0.57735026919f * (rr + gg + bb);
    float hue = atan2f(rotY, rotX) / kTau;
    if (hue < 0.0f) hue += 1.0f;
    const float chromaRadius = sqrtf(rotX * rotX + rotY * rotY);
    const float polar = atan2f(chromaRadius, rotZ);
    float chroma = polar / 0.9553166181245093f;
    if (normConeNormalized != 0) {
      const float angle = hue * kTau - kPi / 6.0f;
      const float cosPolar = cosf(polar);
      const float safeCos = fabsf(cosPolar) > 1e-6f ? cosPolar : (cosPolar < 0.0f ? -1e-6f : 1e-6f);
      const float cone = (sinf(polar) / safeCos) / sqrtf(2.0f);
      const float sinTerm = fminf(fmaxf(sinf(3.0f * angle), -1.0f), 1.0f);
      const float chromaGain = 1.0f / (2.0f * cosf(acosf(sinTerm) / 3.0f));
      chroma = chromaGain > 1e-6f ? cone / chromaGain : 0.0f;
      if (normConeOverflow && chroma < 0.0f) {
        chroma = -chroma;
        hue += 0.5f;
        if (hue >= 1.0f) hue -= 1.0f;
      }
    }
    chroma = normConeOverflow ? fmaxf(chroma, 0.0f) : fminf(fmaxf(chroma, 0.0f), 1.0f);
    const float value = normConeOverflow ? maxRgb : fminf(fmaxf(maxRgb, 0.0f), 1.0f);
    const float angle = hue * kTau;
    *outX = cosf(angle) * chroma;
    *outY = value * 2.0f - 1.0f;
    *outZ = sinf(angle) * chroma;
    return;
  }
  if (plotMode == 7) {
    const bool reuleauxOverflow = (showOverflow != 0 && plotMode == 7);
    const float rr = reuleauxOverflow ? r : clamp01(r);
    const float gg = reuleauxOverflow ? g : clamp01(g);
    const float bb = reuleauxOverflow ? b : clamp01(b);
    const float rotX = 0.33333333333f * (2.0f * rr - gg - bb) * 0.70710678118f;
    const float rotY = (gg - bb) * 0.40824829046f;
    const float rotZ = (rr + gg + bb) / 3.0f;
    float hue = kPi - atan2f(rotY, -rotX);
    if (hue < 0.0f) hue += kTau;
    if (hue >= kTau) hue = fmodf(hue, kTau);
    float sat = fabsf(rotZ) <= 1e-6f ? 0.0f : sqrtf(rotX * rotX + rotY * rotY) / rotZ;
    if (reuleauxOverflow && sat < 0.0f) {
      sat = -sat;
      hue += kPi;
      if (hue >= kTau) hue -= kTau;
    }
    sat = reuleauxOverflow ? sat / 1.41421356237f
                           : fminf(fmaxf(sat / 1.41421356237f, 0.0f), 1.0f);
    const float value = reuleauxOverflow ? fmaxf(rr, fmaxf(gg, bb))
                                         : fminf(fmaxf(fmaxf(rr, fmaxf(gg, bb)), 0.0f), 1.0f);
    *outX = cosf(hue) * sat;
    *outY = value * 2.0f - 1.0f;
    *outZ = sinf(hue) * sat;
    return;
  }
  *outX = r * 2.0f - 1.0f;
  *outY = g * 2.0f - 1.0f;
  *outZ = b * 2.0f - 1.0f;
}

inline __device__ bool outOfBounds(float r, float g, float b) {
  return r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f;
}

inline __device__ void mapDisplayColor(float inR, float inG, float inB, float* outR, float* outG, float* outB) {
  *outR = powf(clamp01(inR), 1.0f / 2.2f);
  *outG = powf(clamp01(inG), 1.0f / 2.2f);
  *outB = powf(clamp01(inB), 1.0f / 2.2f);
}

inline __device__ void rgbToHsl(float r, float g, float b, float* outH, float* outS, float* outL) {
  const float cMax = fmaxf(r, fmaxf(g, b));
  const float cMin = fminf(r, fminf(g, b));
  const float delta = cMax - cMin;
  const float l = 0.5f * (cMax + cMin);
  float h = 0.0f;
  float s = 0.0f;
  if (delta > 1e-6f) {
    const float denom = fmaxf(1e-6f, 1.0f - fabsf(2.0f * l - 1.0f));
    s = delta / denom;
    h = rawRgbHue01(r, g, b, cMax, delta);
  }
  if (outH) *outH = h;
  if (outS) *outS = s;
  if (outL) *outL = l;
}

inline __device__ float hueToRgbChannel(float p, float q, float t) {
  if (t < 0.0f) t += 1.0f;
  if (t > 1.0f) t -= 1.0f;
  if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
  if (t < 1.0f / 2.0f) return q;
  if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
  return p;
}

inline __device__ void hslToRgb(float h, float s, float l, float* outR, float* outG, float* outB) {
  h = wrapHue01(h);
  s = clamp01(s);
  l = clamp01(l);
  if (s <= 1e-6f) {
    *outR = l;
    *outG = l;
    *outB = l;
    return;
  }
  const float q = l < 0.5f ? l * (1.0f + s) : l + s - l * s;
  const float p = 2.0f * l - q;
  *outR = clamp01(hueToRgbChannel(p, q, h + 1.0f / 3.0f));
  *outG = clamp01(hueToRgbChannel(p, q, h));
  *outB = clamp01(hueToRgbChannel(p, q, h - 1.0f / 3.0f));
}

inline __device__ void applyDisplaySaturation(float saturation, float* r, float* g, float* b) {
  const float sat = fminf(6.0f, fmaxf(1.0f, saturation));
  const float baseR = clamp01(*r);
  const float baseG = clamp01(*g);
  const float baseB = clamp01(*b);
  const float luma = clamp01(baseR * 0.2126f + baseG * 0.7152f + baseB * 0.0722f);
  if (sat <= 1.0f) {
    *r = fmaxf(0.0f, luma + (baseR - luma) * sat);
    *g = fmaxf(0.0f, luma + (baseG - luma) * sat);
    *b = fmaxf(0.0f, luma + (baseB - luma) * sat);
  } else {
    float h = 0.0f;
    float s = 0.0f;
    float l = 0.0f;
    rgbToHsl(baseR, baseG, baseB, &h, &s, &l);
    if (s <= 1e-5f) {
      *r = baseR;
      *g = baseG;
      *b = baseB;
    } else {
      const float t = fminf(1.0f, fmaxf(0.0f, (sat - 1.0f) / 5.0f));
      const float shaped = powf(t, 0.55f);
      const float targetS = fminf(1.0f, fmaxf(0.0f, s + (1.0f - s) * (0.32f + 0.68f * shaped)));
      const float highlight = fminf(1.0f, fmaxf(0.0f, (l - 0.58f) / 0.34f));
      const float targetL = fminf(1.0f, fmaxf(0.0f, l - highlight * (0.08f + 0.10f * shaped)));
      float boostedR = baseR;
      float boostedG = baseG;
      float boostedB = baseB;
      hslToRgb(h, targetS, targetL, &boostedR, &boostedG, &boostedB);
      const float mixAmount = fminf(1.0f, fmaxf(0.0f, 0.24f + 0.76f * shaped));
      *r = fmaxf(0.0f, baseR * (1.0f - mixAmount) + boostedR * mixAmount);
      *g = fmaxf(0.0f, baseG * (1.0f - mixAmount) + boostedG * mixAmount);
      *b = fmaxf(0.0f, baseB * (1.0f - mixAmount) + boostedB * mixAmount);
    }
  }
  const float peak = fmaxf(*r, fmaxf(*g, *b));
  if (peak > 1.0f) {
    *r /= peak;
    *g /= peak;
    *b /= peak;
  }
  *r = clamp01(*r);
  *g = clamp01(*g);
  *b = clamp01(*b);
}

__global__ void overlayKernel(float* verts, float* colors, const float* input, OverlayKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cubeSize = static_cast<unsigned int>(max(u.cubeSize, 1));
  const unsigned int cubePoints = cubeSize * cubeSize * cubeSize;
  const unsigned int rampPoints = u.ramp != 0 ? (cubeSize * cubeSize) : 0u;
  const unsigned int total = u.useInputPoints != 0 ? static_cast<unsigned int>(max(u.pointCount, 0)) : (cubePoints + rampPoints);
  if (index >= total) return;
  float r, g, b, alpha;
  if (u.useInputPoints != 0) {
    const unsigned int base = index * 4u;
    r = input[base + 0u];
    g = input[base + 1u];
    b = input[base + 2u];
    alpha = input[base + 3u];
  } else if (index < cubePoints) {
    const unsigned int denom = max(cubeSize - 1u, 1u);
    const unsigned int rx = index % cubeSize;
    const unsigned int gy = (index / cubeSize) % cubeSize;
    const unsigned int bz = index / (cubeSize * cubeSize);
    r = static_cast<float>(rx) / static_cast<float>(denom);
    g = static_cast<float>(gy) / static_cast<float>(denom);
    b = static_cast<float>(bz) / static_cast<float>(denom);
    alpha = 0.24f;
  } else {
    const unsigned int rampIndex = index - cubePoints;
    const unsigned int rampCount = max(rampPoints, 1u);
    const float t = static_cast<float>(rampIndex) / static_cast<float>(max(rampCount - 1u, 1u));
    r = g = b = t;
    alpha = 0.92f;
  }
  float x, y, z;
  mapPlotPosition(r, g, b, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, 0, &x, &y, &z);
  const unsigned int vbase = index * 3u;
  verts[vbase + 0u] = x;
  verts[vbase + 1u] = y;
  verts[vbase + 2u] = z;
  float cr, cg, cb;
  mapDisplayColor(r, g, b, &cr, &cg, &cb);
  applyDisplaySaturation(u.colorSaturation, &cr, &cg, &cb);
  const unsigned int cbase = index * 4u;
  colors[cbase + 0u] = cr;
  colors[cbase + 1u] = cg;
  colors[cbase + 2u] = cb;
  colors[cbase + 3u] = alpha;
}

__global__ void inputKernel(float* verts, float* colors, const float* input, InputKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(u.pointCount, 0));
  if (index >= total) return;
  const unsigned int ibase = index * 3u;
  const float r = input[ibase + 0u];
  const float g = input[ibase + 1u];
  const float b = input[ibase + 2u];
  const bool overflow = outOfBounds(r, g, b);
  const float plotR = u.showOverflow != 0 ? r : clamp01(r);
  const float plotG = u.showOverflow != 0 ? g : clamp01(g);
  const float plotB = u.showOverflow != 0 ? b : clamp01(b);
  float x, y, z;
  mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow, &x, &y, &z);
  const unsigned int vbase = index * 3u;
  verts[vbase + 0u] = x;
  verts[vbase + 1u] = y;
  verts[vbase + 2u] = z;
  float cr, cg, cb;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflow) {
    cr = 1.0f;
    cg = 0.0f;
    cb = 0.0f;
  } else {
    mapDisplayColor(r, g, b, &cr, &cg, &cb);
    applyDisplaySaturation(u.colorSaturation, &cr, &cg, &cb);
  }
  const unsigned int cbase = index * 4u;
  colors[cbase + 0u] = cr;
  colors[cbase + 1u] = cg;
  colors[cbase + 2u] = cb;
  const bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflow);
  float alpha = (overflowHighlighted ? 0.95f : 0.72f) * u.pointAlphaScale;
  if (!overflowHighlighted && u.denseAlphaBias > 0.0f) {
    const float luma = clamp01(cr * 0.2126f + cg * 0.7152f + cb * 0.0722f);
    const float maxRgb = clamp01(fmaxf(cr, fmaxf(cg, cb)));
    const float value = (1.0f - 0.28f) * maxRgb + 0.28f * luma;
    const float highlightKnee = clamp01((value - 0.70f) / 0.24f);
    const float shadowMidProtect = 1.0f - clamp01((value - 0.58f) / 0.30f);
    const float multiplier =
        fminf(1.18f,
              fmaxf(0.94f,
                    1.0f + 0.22f * u.denseAlphaBias * shadowMidProtect - 0.12f * u.denseAlphaBias * highlightKnee));
    alpha = clamp01(alpha * multiplier);
  } else {
    alpha = clamp01(alpha);
  }
  colors[cbase + 3u] = alpha;
}

__global__ void boundsKernel(const float* verts, unsigned int* boundsVals, int pointCount) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned int>(max(pointCount, 0))) return;
  const unsigned int base = index * 3u;
  const unsigned int ox = orderedUintFromFloat(verts[base + 0u]);
  const unsigned int oy = orderedUintFromFloat(verts[base + 1u]);
  const unsigned int oz = orderedUintFromFloat(verts[base + 2u]);
  atomicMin(&boundsVals[0], ox);
  atomicMin(&boundsVals[1], oy);
  atomicMin(&boundsVals[2], oz);
  atomicMax(&boundsVals[3], ox);
  atomicMax(&boundsVals[4], oy);
  atomicMax(&boundsVals[5], oz);
}

__global__ void inputSampleKernel(float* dstVerts,
                                  float* dstColors,
                                  const float* srcVerts,
                                  const float* srcColors,
                                  InputSampleKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int visible = static_cast<unsigned int>(max(u.visiblePointCount, 0));
  const unsigned int full = static_cast<unsigned int>(max(u.fullPointCount, 0));
  if (index >= visible) return;
  unsigned int srcIndex = 0u;
  if (visible > 1u && full > 1u) {
    const float t = static_cast<float>(index) / static_cast<float>(visible - 1u);
    srcIndex = static_cast<unsigned int>(floorf(t * static_cast<float>(full - 1u) + 0.5f));
    srcIndex = min(srcIndex, full - 1u);
  }
  const unsigned int srcVertBase = srcIndex * 3u;
  const unsigned int srcColorBase = srcIndex * 4u;
  const unsigned int dstVertBase = index * 3u;
  const unsigned int dstColorBase = index * 4u;
  dstVerts[dstVertBase + 0u] = srcVerts[srcVertBase + 0u];
  dstVerts[dstVertBase + 1u] = srcVerts[srcVertBase + 1u];
  dstVerts[dstVertBase + 2u] = srcVerts[srcVertBase + 2u];
  dstColors[dstColorBase + 0u] = srcColors[srcColorBase + 0u];
  dstColors[dstColorBase + 1u] = srcColors[srcColorBase + 1u];
  dstColors[dstColorBase + 2u] = srcColors[srcColorBase + 2u];
  dstColors[dstColorBase + 3u] = srcColors[srcColorBase + 3u];
}

template <typename CacheT, typename Uniforms, typename Launcher>
bool buildMesh(CacheT* cache,
               size_t pointCount,
               const float* hostInput,
               size_t inputFloatCount,
               const Uniforms& kernelUniforms,
               Launcher launchKernel,
               unsigned long long serial,
               std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA cache.";
    return false;
  }
  if (!ensureRegistered(cache->verts, cache->colors, pointCount, impl, &localError)) {
    if (error) *error = localError;
    return false;
  }
  if (inputFloatCount > 0) {
    if (!ensureInputCapacity(impl, inputFloatCount, &localError)) {
      if (error) *error = localError;
      return false;
    }
    cudaError_t err = cudaMemcpy(impl->deviceInput, hostInput, inputFloatCount * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      if (error) *error = std::string("Failed to upload CUDA input data: ") + errorString(err);
      return false;
    }
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  cudaError_t err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA-GL resources: ") + errorString(err);
    return false;
  }

  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts), &vertsBytes, impl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors), &colorsBytes, impl->colorsResource);
  }
  if (err != cudaSuccess) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) *error = std::string("Failed to access mapped CUDA-GL buffers: ") + errorString(err);
    return false;
  }

  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  launchKernel(devVerts, devColors, impl->deviceInput, kernelUniforms, blocks);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA kernel execution failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(pointCount);
  cache->available = true;
  return true;
}

bool computeInputBounds(InputCache* cache, std::string* error) {
  if (!cache || cache->verts == 0 || cache->pointCount <= 0) {
    if (error) *error = "CUDA input cache has no point buffer for bounds.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl || !impl->vertsResource) {
    if (error) *error = "CUDA input cache is not registered for bounds.";
    return false;
  }
  if (!ensureBoundsCapacity(impl, &localError)) {
    if (error) *error = localError;
    return false;
  }

  const unsigned int initVals[6] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0u, 0u, 0u};
  cudaError_t err = cudaMemcpy(impl->deviceBounds, initVals, sizeof(initVals), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to initialize CUDA bounds buffer: ") + errorString(err);
    return false;
  }

  std::array<cudaGraphicsResource*, 1> resources = {impl->vertsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA bounds resource: ") + errorString(err);
    return false;
  }

  float* devVerts = nullptr;
  size_t vertsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts), &vertsBytes, impl->vertsResource);
  if (err == cudaSuccess) {
    const unsigned int threads = 256u;
    const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(cache->pointCount) + threads - 1u) / threads);
    boundsKernel<<<blocks, threads>>>(devVerts, impl->deviceBounds, cache->pointCount);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
  }
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA bounds kernel failed: ") + errorString(err);
    return false;
  }

  unsigned int packed[6] = {};
  err = cudaMemcpy(packed, impl->deviceBounds, sizeof(packed), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA bounds buffer: ") + errorString(err);
    return false;
  }
  cache->fitMin[0] = floatFromOrderedUint(packed[0]);
  cache->fitMin[1] = floatFromOrderedUint(packed[1]);
  cache->fitMin[2] = floatFromOrderedUint(packed[2]);
  cache->fitMax[0] = floatFromOrderedUint(packed[3]);
  cache->fitMax[1] = floatFromOrderedUint(packed[4]);
  cache->fitMax[2] = floatFromOrderedUint(packed[5]);
  cache->hasFitBounds = std::isfinite(cache->fitMin[0]) && std::isfinite(cache->fitMin[1]) && std::isfinite(cache->fitMin[2]) &&
                        std::isfinite(cache->fitMax[0]) && std::isfinite(cache->fitMax[1]) && std::isfinite(cache->fitMax[2]);
  return cache->hasFitBounds;
}

void launchOverlay(float* verts, float* colors, const float* input, OverlayKernelUniforms uniforms, unsigned int blocks) {
  overlayKernel<<<blocks, 256u>>>(verts, colors, input, uniforms);
}

void launchInput(float* verts, float* colors, const float* input, InputKernelUniforms uniforms, unsigned int blocks) {
  inputKernel<<<blocks, 256u>>>(verts, colors, input, uniforms);
}

void launchInputSample(float* dstVerts,
                       float* dstColors,
                       const float* srcVerts,
                       const float* srcColors,
                       InputSampleKernelUniforms uniforms,
                       unsigned int blocks) {
  inputSampleKernel<<<blocks, 256u>>>(dstVerts, dstColors, srcVerts, srcColors, uniforms);
}

}  // namespace

ProbeResult probe() {
  ProbeResult result{};
  std::string error;
  result.available = ensureContext(&error);
  CudaContext& ctx = context();
  result.interopReady = ctx.interopReady;
  result.deviceName = ctx.deviceName.c_str();
  result.reason = ctx.reason.c_str();
  return result;
}

StartupValidationResult validateStartup() {
  StartupValidationResult result{};
  std::string error;
  if (!ensureContext(&error)) {
    result.reason = error.empty() ? std::string("CUDA context unavailable.") : error;
    return result;
  }
  result.ready = true;
  return result;
}

void releaseOverlayCache(OverlayCache* cache) {
  releaseCache(cache);
}

void releaseInputCache(InputCache* cache) {
  releaseCache(cache);
  if (cache) cache->hasFitBounds = false;
}

void releaseInputSampleCache(InputSampleCache* cache) {
  releaseSampleCache(cache);
}

bool buildOverlayMesh(OverlayCache* cache,
                      const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      unsigned long long serial,
                      std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA overlay cache has no GL buffers.";
    return false;
  }
  const int cubeSize = std::max(request.cubeSize, 1);
  const size_t cubePointCount = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  const size_t rampPointCount = request.ramp != 0
                                  ? std::max<size_t>(static_cast<size_t>(cubeSize), static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize))
                                  : 0u;
  const size_t pointCount = request.useInputPoints != 0 ? static_cast<size_t>(std::max(request.pointCount, 0)) : (cubePointCount + rampPointCount);
  OverlayKernelUniforms uniforms{};
  uniforms.cubeSize = cubeSize;
  uniforms.ramp = request.ramp;
  uniforms.useInputPoints = request.useInputPoints;
  uniforms.pointCount = request.pointCount;
  uniforms.colorSaturation = request.colorSaturation;
  uniforms.plotMode = request.remap.plotMode;
  uniforms.circularHsl = request.remap.circularHsl;
  uniforms.circularHsv = request.remap.circularHsv;
  uniforms.normConeNormalized = request.remap.normConeNormalized;
  const float* inputPtr = request.useInputPoints != 0 ? inputPoints.data() : nullptr;
  const size_t inputFloatCount = request.useInputPoints != 0 ? inputPoints.size() : 0u;
  return buildMesh(cache, pointCount, inputPtr, inputFloatCount, uniforms, launchOverlay, serial, error);
}

bool buildInputMesh(InputCache* cache,
                    const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    unsigned long long serial,
                    std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA input cache has no GL buffers.";
    return false;
  }
  const size_t pointCount = rawPoints.size() / 3u;
  InputKernelUniforms uniforms{};
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
  cache->hasFitBounds = false;
  if (!buildMesh(cache, pointCount, rawPoints.data(), rawPoints.size(), uniforms, launchInput, serial, error)) {
    return false;
  }
  std::string localError;
  if (!computeInputBounds(cache, &localError) && error && error->empty()) {
    *error = localError;
  }
  return true;
}

bool buildInputSampledMesh(InputCache* sourceCache,
                           InputSampleCache* sampleCache,
                           const InputSampleRequest& request,
                           unsigned long long serial,
                           std::string* error) {
  if (!sourceCache || !sampleCache || sourceCache->verts == 0 || sourceCache->colors == 0 ||
      sampleCache->verts == 0 || sampleCache->colors == 0) {
    if (error) *error = "CUDA sampled input cache is missing GL buffers.";
    return false;
  }
  if (request.visiblePointCount <= 0 || request.fullPointCount <= 0) {
    if (error) *error = "Invalid CUDA input thinning request.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* sourceImpl = reinterpret_cast<CacheImpl*>(sourceCache->internal);
  if (!sourceImpl || !sourceImpl->vertsResource || !sourceImpl->colorsResource) {
    if (error) *error = "CUDA source mesh is not registered for thinning.";
    return false;
  }
  SampleCacheImpl* sampleImpl = ensureSampleImpl(sampleCache);
  if (!sampleImpl) {
    if (error) *error = "Failed to allocate CUDA sampled cache.";
    return false;
  }
  const size_t visiblePointCount = static_cast<size_t>(request.visiblePointCount);
  if (!ensureSampleRegistered(sampleCache->verts, sampleCache->colors, visiblePointCount, sampleImpl, &localError)) {
    if (error) *error = localError;
    return false;
  }

  std::array<cudaGraphicsResource*, 4> resources = {
      sourceImpl->vertsResource,
      sourceImpl->colorsResource,
      sampleImpl->vertsResource,
      sampleImpl->colorsResource
  };
  cudaError_t err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA thinning resources: ") + errorString(err);
    return false;
  }

  float* srcVerts = nullptr;
  float* srcColors = nullptr;
  float* dstVerts = nullptr;
  float* dstColors = nullptr;
  size_t srcVertsBytes = 0;
  size_t srcColorsBytes = 0;
  size_t dstVertsBytes = 0;
  size_t dstColorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&srcVerts), &srcVertsBytes, sourceImpl->vertsResource);
  if (err == cudaSuccess) err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&srcColors), &srcColorsBytes, sourceImpl->colorsResource);
  if (err == cudaSuccess) err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dstVerts), &dstVertsBytes, sampleImpl->vertsResource);
  if (err == cudaSuccess) err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dstColors), &dstColorsBytes, sampleImpl->colorsResource);
  if (err != cudaSuccess) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) *error = std::string("Failed to access CUDA thinning buffers: ") + errorString(err);
    return false;
  }

  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((visiblePointCount + threads - 1u) / threads);
  InputSampleKernelUniforms uniforms{};
  uniforms.fullPointCount = request.fullPointCount;
  uniforms.visiblePointCount = request.visiblePointCount;
  launchInputSample(dstVerts, dstColors, srcVerts, srcColors, uniforms, blocks);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA thinning kernel failed: ") + errorString(err);
    return false;
  }

  sampleCache->builtSerial = serial;
  sampleCache->pointCount = request.visiblePointCount;
  sampleCache->available = true;
  return true;
}

}  // namespace ChromaspaceCuda
