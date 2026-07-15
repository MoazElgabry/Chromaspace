#include "ChromaspaceCuda.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
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

constexpr size_t kSharedReductionUintCount = 16u;

// Gloss field resident workspace layout:
//   0..16  shared occupancy/color/carrier intermediates
//   17..24 Candidate 1 solution family
//   25..32 Candidate 2 solution family
//   33..45 transient solver grids for resident field solver variants
// Projection and 2D field geometry requests select the active candidate without
// readback or CPU-side reformation, so the Gloss A toggle changes real GPU data.
constexpr size_t kGlossFieldWorkspaceArrayCount = 46u;
constexpr size_t kGlossFieldCandidate1Base = 17u;
constexpr size_t kGlossFieldCandidate2Base = 25u;
constexpr size_t kGlossFieldScratchBase = 33u;

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
  int cubeSlicingEnabled;
  int neutralRadiusEnabled;
  float neutralRadius;
  int cubeSliceRed;
  int cubeSliceYellow;
  int cubeSliceGreen;
  int cubeSliceCyan;
  int cubeSliceBlue;
  int cubeSliceMagenta;
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

struct InputKernelUniforms {
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

struct InputSampleKernelUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct RasterSourceKernelUniforms {
  InputKernelUniforms input;
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

struct CacheImpl {
  cudaGraphicsResource* vertsResource = nullptr;
  cudaGraphicsResource* colorsResource = nullptr;
  GLuint registeredVerts = 0;
  GLuint registeredColors = 0;
  size_t pointCapacity = 0;
  float* deviceInput = nullptr;
  size_t inputCapacityFloats = 0;
  unsigned char* deviceSource = nullptr;
  size_t sourceCapacityBytes = 0;
  unsigned int* deviceBounds = nullptr;
  float* deviceFieldWorkspace = nullptr;
  size_t fieldWorkspaceFloats = 0;
};

struct SampleCacheImpl {
  cudaGraphicsResource* vertsResource = nullptr;
  cudaGraphicsResource* colorsResource = nullptr;
  GLuint registeredVerts = 0;
  GLuint registeredColors = 0;
  size_t pointCapacity = 0;
};

struct ScopeGeometryCacheImpl {
  cudaGraphicsResource* lineVertsResource = nullptr;
  cudaGraphicsResource* lineColorsResource = nullptr;
  cudaGraphicsResource* fillVertsResource = nullptr;
  cudaGraphicsResource* fillColorsResource = nullptr;
  GLuint registeredLineVerts = 0;
  GLuint registeredLineColors = 0;
  GLuint registeredFillVerts = 0;
  GLuint registeredFillColors = 0;
  size_t lineVertexCapacity = 0;
  size_t fillVertexCapacity = 0;
};

struct SourceTextureCacheImpl {
  cudaGraphicsResource* textureResource = nullptr;
  GLuint registeredTexture = 0;
  int width = 0;
  int height = 0;
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

unsigned int orderedUintFromFloatHost(float value) {
  unsigned int bits = 0u;
  std::memcpy(&bits, &value, sizeof(value));
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool decodeHexBytes(const std::string& hex, void* outData, size_t byteCount) {
  if (!outData || hex.size() != byteCount * 2u) return false;
  auto* out = reinterpret_cast<unsigned char*>(outData);
  for (size_t i = 0; i < byteCount; ++i) {
    const int hi = hexValue(hex[i * 2u]);
    const int lo = hexValue(hex[i * 2u + 1u]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<unsigned char>((hi << 4) | lo);
  }
  return true;
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
  if (impl->deviceSource) cudaFree(impl->deviceSource);
  if (impl->deviceBounds) cudaFree(impl->deviceBounds);
  if (impl->deviceFieldWorkspace) cudaFree(impl->deviceFieldWorkspace);
  delete impl;
}

void releaseSampleImpl(SampleCacheImpl* impl) {
  if (!impl) return;
  if (impl->vertsResource) cudaGraphicsUnregisterResource(impl->vertsResource);
  if (impl->colorsResource) cudaGraphicsUnregisterResource(impl->colorsResource);
  delete impl;
}

void releaseScopeGeometryImpl(ScopeGeometryCacheImpl* impl) {
  if (!impl) return;
  if (impl->lineVertsResource) cudaGraphicsUnregisterResource(impl->lineVertsResource);
  if (impl->lineColorsResource) cudaGraphicsUnregisterResource(impl->lineColorsResource);
  if (impl->fillVertsResource) cudaGraphicsUnregisterResource(impl->fillVertsResource);
  if (impl->fillColorsResource) cudaGraphicsUnregisterResource(impl->fillColorsResource);
  delete impl;
}

void releaseSourceTextureImpl(SourceTextureCacheImpl* impl) {
  if (!impl) return;
  if (impl->textureResource) cudaGraphicsUnregisterResource(impl->textureResource);
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

ScopeGeometryCacheImpl* ensureScopeGeometryImpl(ScopeGeometryCache* cache) {
  if (!cache) return nullptr;
  if (!cache->internal) cache->internal = new ScopeGeometryCacheImpl();
  return reinterpret_cast<ScopeGeometryCacheImpl*>(cache->internal);
}

void releaseScopeGeometryCacheInternal(ScopeGeometryCache* cache) {
  if (!cache) return;
  releaseScopeGeometryImpl(reinterpret_cast<ScopeGeometryCacheImpl*>(cache->internal));
  cache->internal = nullptr;
  cache->builtSerial = 0;
  cache->lineVertexCount = 0;
  cache->fillVertexCount = 0;
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

bool registerWriteDiscard(GLuint buffer,
                          cudaGraphicsResource** resource,
                          const char* label,
                          std::string* error) {
  cudaError_t err = cudaGraphicsGLRegisterBuffer(resource, buffer, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    if (error) {
      *error = std::string("Failed to register CUDA ") + label + " buffer: " + errorString(err);
    }
    *resource = nullptr;
    return false;
  }
  return true;
}

bool ensureScopeGeometryRegistered(GLuint lineVerts,
                                   GLuint lineColors,
                                   GLuint fillVerts,
                                   GLuint fillColors,
                                   size_t lineVertexCapacity,
                                   size_t fillVertexCapacity,
                                   ScopeGeometryCacheImpl* impl,
                                   std::string* error) {
  if (!impl || lineVerts == 0 || lineColors == 0 || fillVerts == 0 || fillColors == 0) {
    if (error) *error = "Missing GL buffers for CUDA scope geometry interop.";
    return false;
  }
  if (impl->registeredLineVerts == lineVerts &&
      impl->registeredLineColors == lineColors &&
      impl->registeredFillVerts == fillVerts &&
      impl->registeredFillColors == fillColors &&
      impl->lineVertexCapacity == lineVertexCapacity &&
      impl->fillVertexCapacity == fillVertexCapacity &&
      impl->lineVertsResource &&
      impl->lineColorsResource &&
      impl->fillVertsResource &&
      impl->fillColorsResource) {
    return true;
  }
  if (impl->lineVertsResource) {
    cudaGraphicsUnregisterResource(impl->lineVertsResource);
    impl->lineVertsResource = nullptr;
  }
  if (impl->lineColorsResource) {
    cudaGraphicsUnregisterResource(impl->lineColorsResource);
    impl->lineColorsResource = nullptr;
  }
  if (impl->fillVertsResource) {
    cudaGraphicsUnregisterResource(impl->fillVertsResource);
    impl->fillVertsResource = nullptr;
  }
  if (impl->fillColorsResource) {
    cudaGraphicsUnregisterResource(impl->fillColorsResource);
    impl->fillColorsResource = nullptr;
  }

  if (!registerWriteDiscard(lineVerts, &impl->lineVertsResource, "histogram line verts", error) ||
      !registerWriteDiscard(lineColors, &impl->lineColorsResource, "histogram line colors", error) ||
      !registerWriteDiscard(fillVerts, &impl->fillVertsResource, "histogram fill verts", error) ||
      !registerWriteDiscard(fillColors, &impl->fillColorsResource, "histogram fill colors", error)) {
    if (impl->lineVertsResource) cudaGraphicsUnregisterResource(impl->lineVertsResource);
    if (impl->lineColorsResource) cudaGraphicsUnregisterResource(impl->lineColorsResource);
    if (impl->fillVertsResource) cudaGraphicsUnregisterResource(impl->fillVertsResource);
    if (impl->fillColorsResource) cudaGraphicsUnregisterResource(impl->fillColorsResource);
    impl->lineVertsResource = nullptr;
    impl->lineColorsResource = nullptr;
    impl->fillVertsResource = nullptr;
    impl->fillColorsResource = nullptr;
    return false;
  }

  impl->registeredLineVerts = lineVerts;
  impl->registeredLineColors = lineColors;
  impl->registeredFillVerts = fillVerts;
  impl->registeredFillColors = fillColors;
  impl->lineVertexCapacity = lineVertexCapacity;
  impl->fillVertexCapacity = fillVertexCapacity;
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

bool ensureSourceCapacity(CacheImpl* impl, size_t byteCount, std::string* error) {
  if (!impl) return false;
  if (byteCount <= impl->sourceCapacityBytes && impl->deviceSource != nullptr) return true;
  if (impl->deviceSource) {
    cudaFree(impl->deviceSource);
    impl->deviceSource = nullptr;
    impl->sourceCapacityBytes = 0;
  }
  cudaError_t err = cudaMalloc(&impl->deviceSource, byteCount);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA source buffer: ") + errorString(err);
    return false;
  }
  impl->sourceCapacityBytes = byteCount;
  return true;
}

bool ensureBoundsCapacity(CacheImpl* impl, std::string* error) {
  if (!impl) return false;
  if (impl->deviceBounds != nullptr) return true;
  cudaError_t err = cudaMalloc(&impl->deviceBounds, kSharedReductionUintCount * sizeof(unsigned int));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA bounds buffer: ") + errorString(err);
    return false;
  }
  return true;
}

bool ensureFieldWorkspace(CacheImpl* impl, size_t floatCount, std::string* error) {
  if (!impl) return false;
  if (floatCount <= impl->fieldWorkspaceFloats && impl->deviceFieldWorkspace != nullptr) return true;
  if (impl->deviceFieldWorkspace) {
    cudaFree(impl->deviceFieldWorkspace);
    impl->deviceFieldWorkspace = nullptr;
    impl->fieldWorkspaceFloats = 0;
  }
  cudaError_t err = cudaMalloc(&impl->deviceFieldWorkspace, floatCount * sizeof(float));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA gloss-field workspace: ") + errorString(err);
    return false;
  }
  impl->fieldWorkspaceFloats = floatCount;
  return true;
}

inline __device__ unsigned int orderedUintFromFloat(float value) {
  unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

inline __device__ float floatFromOrderedUintDevice(unsigned int ordered) {
  const unsigned int bits = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
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

inline __device__ float safeDiv(float num, float den) {
  return fabsf(den) < 1e-6f ? 0.0f : num / den;
}

inline __device__ float safeExp2Clamped(float value) {
  return exp2f(fminf(fmaxf(value, -126.0f), 126.0f));
}

inline __device__ float safePowPos(float value, float exponent) {
  return value <= 0.0f ? 0.0f : powf(value, exponent);
}

inline __device__ float signPreservingPow(float value, float exponent) {
  return value == 0.0f ? 0.0f : copysignf(safePowPos(fabsf(value), exponent), value);
}

inline __device__ float exp10Compat(float value) {
  return safeExp2Clamped(value * 3.3219280948873626f);
}

inline __device__ float decodeTransferChannel(float x, int tf) {
  switch (tf) {
    case 0: return x;
    case 1: {
      const float a = fabsf(x);
      const float decoded = (a <= 0.04045f) ? safeDiv(a, 12.92f) : safePowPos(safeDiv(a + 0.055f, 1.055f), 2.4f);
      return copysignf(decoded, x);
    }
    case 2: return signPreservingPow(x, 2.4f);
    case 3: return x <= 0.02740668f ? safeDiv(x, 10.44426855f) : safeExp2Clamped(safeDiv(x, 0.07329248f) - 7.0f) - 0.0075f;
    case 4: return x <= 0.155251141552511f ? safeDiv(x - 0.0729055341958355f, 10.5402377416545f) : safeExp2Clamped(x * 17.52f - 9.72f);
    case 5: return x < 5.367655f * 0.010591f + 0.092809f ? safeDiv(x - 0.092809f, 5.367655f) : safeDiv(exp10Compat(safeDiv(x - 0.385537f, 0.247190f)) - 0.052272f, 5.555556f);
    case 6: return x < -0.7774983977293537f ? x * 0.3033266726886969f - 0.7774983977293537f : safeDiv(safeExp2Clamped(14.0f * safeDiv(x - 0.09286412512218964f, 0.9071358748778103f) + 6.0f) - 64.0f, 2231.8263090676883f);
    case 7: {
      constexpr float kCut = 0.092864125f;
      constexpr float kScale = 0.24136077f;
      constexpr float kGain = 87.099375f;
      const float decoded = x < kCut ? -safeDiv(exp10Compat(safeDiv(kCut - x, kScale)) - 1.0f, kGain)
                                     : safeDiv(exp10Compat(safeDiv(x - kCut, kScale)) - 1.0f, kGain);
      return decoded * 0.9f;
    }
    case 8: return x < 171.2102946929f / 1023.0f ? safeDiv((x * 1023.0f - 95.0f) * 0.01125f, 171.2102946929f - 95.0f) : (exp10Compat(safeDiv(x * 1023.0f - 420.0f, 261.5f)) * 0.19f - 0.01f);
    case 9:
      if (x < 0.04076162f) return -safeDiv(exp10Compat(safeDiv(0.069886632f - x, 0.42889912f)) - 1.0f, 14.98325f);
      if (x <= 0.105357102f) return safeDiv(x - 0.073059361f, 2.3069815f);
      return safeDiv(exp10Compat(safeDiv(x - 0.073059361f, 0.36726845f)) - 1.0f, 14.98325f);
    case 10: return x < 0.0f ? safeDiv(x, 15.1927f) - 0.01f : safeDiv(exp10Compat(safeDiv(x, 0.224282f)) - 1.0f, 155.975327f) - 0.01f;
    case 11: {
      constexpr float kA = 8.283605932402494f;
      constexpr float kB = 0.09246575342465753f;
      constexpr float kC = 0.5300133392291939f;
      constexpr float kD = 0.08692876065491224f;
      constexpr float kE = 0.005494072432257808f;
      constexpr float kCut = kA * 0.005f + kB;
      return x < kCut ? safeDiv(x - kB, kA) : expf(safeDiv(x - kC, kD)) - kE;
    }
    case 12: return x <= 0.14f ? safeDiv(x - 0.0929f, 6.025f) : safeDiv(exp10Compat(3.89616f * x - 2.27752f) - 0.0108f, 0.9892f);
    case 13: {
      constexpr float kA = 0.555556f;
      constexpr float kB = 0.009468f;
      constexpr float kC = 0.344676f;
      constexpr float kD = 0.790453f;
      constexpr float kE = 8.735631f;
      constexpr float kF = 0.092864f;
      constexpr float kCut = 0.100537775223865f;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 14: {
      constexpr float kA = 5.555556f;
      constexpr float kB = 0.064829f;
      constexpr float kC = 0.245281f;
      constexpr float kD = 0.384316f;
      constexpr float kE = 8.799461f;
      constexpr float kF = 0.092864f;
      constexpr float kCut = 0.100686685370811f;
      return x >= kCut ? safeDiv(exp10Compat(safeDiv(x - kD, kC)), kA) - safeDiv(kB, kA) : safeDiv(x - kF, kE);
    }
    case 15: return x < 0.181f ? safeDiv(x - 0.125f, 5.6f) : exp10Compat(safeDiv(x - 0.598206f, 0.241514f)) - 0.00873f;
    case 16: return signPreservingPow(x, 2.2f);
    case 17: return signPreservingPow(x, 2.6f);
    default: return x;
  }
}

inline __device__ void mulRows(const float* m, float x, float y, float z, float* outX, float* outY, float* outZ) {
  *outX = m[0] * x + m[1] * y + m[2] * z;
  *outY = m[3] * x + m[4] * y + m[5] * z;
  *outZ = m[6] * x + m[7] * y + m[8] * z;
}

inline __device__ void xyToXyz(float x, float y, float Y, float* outX, float* outY, float* outZ) {
  if (fabsf(y) <= 1e-8f) {
    *outX = x;
    *outY = Y;
    *outZ = 1.0f - x;
    return;
  }
  *outX = x * Y / y;
  *outY = Y;
  *outZ = (1.0f - x - y) * Y / y;
}

inline __device__ void xyzToXyY(float x, float y, float z, float fallbackX, float fallbackY, float* outX, float* outY, float* outYValue) {
  if (fabsf(y) <= 1e-8f) {
    *outX = fallbackX;
    *outY = fallbackY;
    *outYValue = 0.0f;
    return;
  }
  const float sum = x + y + z;
  if (fabsf(sum) <= 1e-8f) {
    *outX = fallbackX;
    *outY = fallbackY;
    *outYValue = y;
    return;
  }
  *outX = x / sum;
  *outY = y / sum;
  *outYValue = y;
}

inline __device__ void mapChromaticityPosition(float r, float g, float b, const InputKernelUniforms& u,
                                               float* outX, float* outY, float* outZ) {
  float linearR = decodeTransferChannel(r, u.chromaticityInputTransfer);
  float linearG = decodeTransferChannel(g, u.chromaticityInputTransfer);
  float linearB = decodeTransferChannel(b, u.chromaticityInputTransfer);
  if (u.showOverflow == 0) {
    linearR = clamp01(linearR);
    linearG = clamp01(linearG);
    linearB = clamp01(linearB);
  }
  float xyzX = 0.0f;
  float xyzY = 0.0f;
  float xyzZ = 0.0f;
  mulRows(u.chromaticityRgbToXyz, linearR, linearG, linearB, &xyzX, &xyzY, &xyzZ);
  float xyX = u.chromaticityWhiteX;
  float xyY = u.chromaticityWhiteY;
  float Y = 0.0f;
  xyzToXyY(xyzX, xyzY, xyzZ, u.chromaticityWhiteX, u.chromaticityWhiteY, &xyX, &xyY, &Y);
  if (u.chromaticityReferenceBasis != 0) {
    float basisXyzX = 0.0f;
    float basisXyzY = 0.0f;
    float basisXyzZ = 0.0f;
    xyToXyz(xyX, xyY, 1.0f, &basisXyzX, &basisXyzY, &basisXyzZ);
    float rgbX = 0.0f;
    float rgbY = 0.0f;
    float rgbZ = 0.0f;
    mulRows(u.chromaticityXyzToRgb, basisXyzX, basisXyzY, basisXyzZ, &rgbX, &rgbY, &rgbZ);
    xyzToXyY(rgbX, rgbY, rgbZ, 1.0f / 3.0f, 1.0f / 3.0f, &xyX, &xyY, &basisXyzY);
  }
  const float viewerHeight = (u.showOverflow != 0 ? Y : clamp01(Y)) * 2.0f - 1.0f;
  *outX = (xyX - (1.0f / 3.0f)) * 3.0f;
  *outY = (xyY - (1.0f / 3.0f)) * 3.0f;
  *outZ = viewerHeight;
}

inline __device__ void mapChromaticityPosition(float r, float g, float b, const OverlayKernelUniforms& u,
                                               float* outX, float* outY, float* outZ) {
  float linearR = clamp01(decodeTransferChannel(r, u.chromaticityInputTransfer));
  float linearG = clamp01(decodeTransferChannel(g, u.chromaticityInputTransfer));
  float linearB = clamp01(decodeTransferChannel(b, u.chromaticityInputTransfer));
  float xyzX = 0.0f;
  float xyzY = 0.0f;
  float xyzZ = 0.0f;
  mulRows(u.chromaticityRgbToXyz, linearR, linearG, linearB, &xyzX, &xyzY, &xyzZ);
  float xyX = u.chromaticityWhiteX;
  float xyY = u.chromaticityWhiteY;
  float Y = 0.0f;
  xyzToXyY(xyzX, xyzY, xyzZ, u.chromaticityWhiteX, u.chromaticityWhiteY, &xyX, &xyY, &Y);
  if (u.chromaticityReferenceBasis != 0) {
    float basisXyzX = 0.0f;
    float basisXyzY = 0.0f;
    float basisXyzZ = 0.0f;
    xyToXyz(xyX, xyY, 1.0f, &basisXyzX, &basisXyzY, &basisXyzZ);
    float rgbX = 0.0f;
    float rgbY = 0.0f;
    float rgbZ = 0.0f;
    mulRows(u.chromaticityXyzToRgb, basisXyzX, basisXyzY, basisXyzZ, &rgbX, &rgbY, &rgbZ);
    xyzToXyY(rgbX, rgbY, rgbZ, 1.0f / 3.0f, 1.0f / 3.0f, &xyX, &xyY, &basisXyzY);
  }
  const float viewerHeight = clamp01(Y) * 2.0f - 1.0f;
  *outX = (xyX - (1.0f / 3.0f)) * 3.0f;
  *outY = (xyY - (1.0f / 3.0f)) * 3.0f;
  *outZ = viewerHeight;
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

inline __device__ float glossCommonComponent(float r, float g, float b) {
  return fmaxf(0.0f, fminf(r, fminf(g, b)));
}

inline __device__ float glossNeutrality(float r, float g, float b) {
  const float common = glossCommonComponent(r, g, b);
  const float maxRgb = fmaxf(r, fmaxf(g, b));
  return maxRgb > 1e-6f ? fminf(fmaxf(common / maxRgb, 0.0f), 1.0f) : 0.0f;
}

inline __device__ float glossStrengthCue(float r, float g, float b) {
  const float common = glossCommonComponent(r, g, b);
  const float neutrality = glossNeutrality(r, g, b);
  return fminf(fmaxf(common * (0.75f + 0.85f * neutrality), 0.0f), 1.0f);
}

inline __device__ float glossPresenceWeight(float glossCue) {
  const float t = fminf(fmaxf((glossCue - 0.06f) / 0.22f, 0.0f), 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline __device__ float glossLuma(float r, float g, float b) {
  return r * 0.2126f + g * 0.7152f + b * 0.0722f;
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

__device__ float overlayNeutralRadius(float r, float g, float b, const OverlayKernelUniforms& u) {
  constexpr float kRgbAxisMaxRadius = 0.8164965809277260f;
  constexpr float kPolarMax = 0.9553166181245093f;
  constexpr float kChenPolarScale = 1.0467733744265997f;
  const int mode = u.plotMode;
  if (mode == 1) {
    const float cMax = fmaxf(r, fmaxf(g, b));
    const float cMin = fminf(r, fminf(g, b));
    if (u.circularHsl != 0) {
      const float l = 0.5f * (cMax + cMin);
      float denom = 1.0f - fabsf(2.0f * l - 1.0f);
      if (fabsf(denom) <= 1e-6f) denom = denom < 0.0f ? -1e-6f : 1e-6f;
      return clamp01(fabsf((cMax - cMin) / denom));
    }
    return clamp01(cMax - cMin);
  }
  if (mode == 2) {
    if (u.circularHsv != 0) {
      const float cMax = fmaxf(r, fmaxf(g, b));
      const float cMin = fminf(r, fminf(g, b));
      const float delta = cMax - cMin;
      return (delta > 1e-6f && cMax > 1e-6f) ? clamp01(delta / cMax) : 0.0f;
    }
    const float x = r - 0.5f * g - 0.5f * b;
    const float z = 0.8660254037844386f * (g - b);
    return clamp01(sqrtf(x * x + z * z));
  }
  const bool overflowMode = false;
  const float rr = overflowMode ? r : clamp01(r);
  const float gg = overflowMode ? g : clamp01(g);
  const float bb = overflowMode ? b : clamp01(b);
  const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
  const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
  const float rotZ = 0.57735026919f * (rr + gg + bb);
  const float chromaRadius = sqrtf(rotX * rotX + rotY * rotY);
  if (mode == 3) {
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    const float polar = atanf(chromaRadius / fmaxf(rotZ, 1e-8f));
    const float light = radius3 * 0.5773502691896258f;
    const float radial = light * sinf(polar * kChenPolarScale) / kRgbAxisMaxRadius;
    return clamp01(radial);
  }
  if (mode == 4 || mode == 5) {
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    const float polar = atan2f(chromaRadius, rotZ);
    const float radial = radius3 * sinf((polar / kPolarMax) * kPolarMax);
    return clamp01(radial / sinf(kPolarMax));
  }
  if (mode == 6) {
    const float polar = atan2f(chromaRadius, rotZ);
    return clamp01(polar / kPolarMax);
  }
  if (mode == 7) {
    const float rotZAvg = (rr + gg + bb) / 3.0f;
    const float rx = 0.33333333333f * (2.0f * rr - gg - bb) * 0.70710678118f;
    const float ry = (gg - bb) * 0.40824829046f;
    const float sat = fabsf(rotZAvg) <= 1e-6f ? 0.0f : sqrtf(rx * rx + ry * ry) / rotZAvg;
    return clamp01(fabsf(sat) / 1.41421356237f);
  }
  return clamp01(sqrtf(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius);
}

__device__ bool overlayCubeSliceContains(float r, float g, float b, const OverlayKernelUniforms& u) {
  if (u.neutralRadiusEnabled != 0 && u.plotMode != 8) {
    const float threshold = clamp01(u.neutralRadius) * clamp01(u.neutralRadius);
    if (overlayNeutralRadius(r, g, b, u) > threshold + 1.0e-6f) return false;
  }
  if (u.cubeSlicingEnabled == 0) return true;
  const bool anySelected = u.cubeSliceRed || u.cubeSliceYellow || u.cubeSliceGreen ||
                           u.cubeSliceCyan || u.cubeSliceBlue || u.cubeSliceMagenta;
  if (!anySelected) return false;
  if (u.plotMode == 0) {
    constexpr float kEps = 1.0e-6f;
    const bool geRG = r + kEps >= g;
    const bool geGB = g + kEps >= b;
    const bool geGR = g + kEps >= r;
    const bool geRB = r + kEps >= b;
    const bool geBG = b + kEps >= g;
    const bool geBR = b + kEps >= r;
    if (u.cubeSliceRed && geRG && geGB) return true;
    if (u.cubeSliceYellow && geGR && geRB) return true;
    if (u.cubeSliceGreen && geGB && geBR) return true;
    if (u.cubeSliceCyan && geBG && geGR) return true;
    if (u.cubeSliceBlue && geBR && geRG) return true;
    if (u.cubeSliceMagenta && geRB && geBG) return true;
    return false;
  }
  const float cMax = fmaxf(r, fmaxf(g, b));
  const float cMin = fminf(r, fminf(g, b));
  const float delta = cMax - cMin;
  if (delta <= 1.0e-6f) return false;
  const float hue = wrapHue01(rawRgbHue01(r, g, b, cMax, delta));
  const int sector = static_cast<int>(floorf((hue + (1.0f / 12.0f)) * 6.0f)) % 6;
  if (sector == 0) return u.cubeSliceRed != 0;
  if (sector == 1) return u.cubeSliceYellow != 0;
  if (sector == 2) return u.cubeSliceGreen != 0;
  if (sector == 3) return u.cubeSliceCyan != 0;
  if (sector == 4) return u.cubeSliceBlue != 0;
  return u.cubeSliceMagenta != 0;
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
  if (u.useInputPoints == 0 && !overlayCubeSliceContains(r, g, b, u)) {
    alpha = 0.0f;
  }
  float x, y, z;
  if (u.plotMode == 8) {
    mapChromaticityPosition(r, g, b, u, &x, &y, &z);
  } else {
    mapPlotPosition(r, g, b, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, 0, &x, &y, &z);
  }
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

__device__ void writeMappedInputPoint(float* verts,
                                      float* colors,
                                      unsigned int index,
                                      float xNorm,
                                      float yNorm,
                                      float r,
                                      float g,
                                      float b,
                                      InputKernelUniforms u) {
  const bool overflow = outOfBounds(r, g, b);
  const float plotR = u.showOverflow != 0 ? r : clamp01(r);
  const float plotG = u.showOverflow != 0 ? g : clamp01(g);
  const float plotB = u.showOverflow != 0 ? b : clamp01(b);
  float x, y, z;
  if (u.plotMode == 8) {
    mapChromaticityPosition(r, g, b, u, &x, &y, &z);
  } else {
    mapPlotPosition(plotR, plotG, plotB, u.plotMode, u.circularHsl, u.circularHsv, u.normConeNormalized, u.showOverflow, &x, &y, &z);
  }
  if (u.glossView != 0) {
    const float aspect = fminf(fmaxf(u.sourceAspect, 0.25f), 4.0f);
    const float halfWidth = aspect >= 1.0f ? 1.22f : (1.22f * aspect);
    const float halfDepth = aspect >= 1.0f ? (1.22f / aspect) : 1.22f;
    const float common = glossCommonComponent(plotR, plotG, plotB);
    const float bodyR = fmaxf(plotR - common, 0.0f);
    const float bodyG = fmaxf(plotG - common, 0.0f);
    const float bodyB = fmaxf(plotB - common, 0.0f);
    const float bodyLuma = fminf(fmaxf(bodyR * 0.2126f + bodyG * 0.7152f + bodyB * 0.0722f, 0.0f), 1.0f);
    const float glossCue = glossStrengthCue(plotR, plotG, plotB);
    const float glossPresence = glossPresenceWeight(glossCue);
    x = -halfWidth + (2.0f * halfWidth * xNorm);
    z = halfDepth - (2.0f * halfDepth * yNorm);
    y = -0.92f + bodyLuma * 0.92f + glossCue * glossPresence * u.glossLiftScale * 1.34f;
  }
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
    if (u.glossView != 0) {
      const float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
      const float neutralBlend = fminf(fmaxf(0.08f + 0.52f * glossPresence, 0.0f), 0.62f);
      const float brightnessGain = 1.18f + 1.20f * glossPresence;
      cr = fminf(fmaxf((cr * (1.0f - neutralBlend) + neutralBlend) * brightnessGain, 0.0f), 1.0f);
      cg = fminf(fmaxf((cg * (1.0f - neutralBlend) + neutralBlend) * brightnessGain, 0.0f), 1.0f);
      cb = fminf(fmaxf((cb * (1.0f - neutralBlend) + neutralBlend) * brightnessGain, 0.0f), 1.0f);
    }
  }
  const unsigned int cbase = index * 4u;
  colors[cbase + 0u] = cr;
  colors[cbase + 1u] = cg;
  colors[cbase + 2u] = cb;
  const bool overflowHighlighted = (u.showOverflow != 0 && u.highlightOverflow != 0 && overflow);
  float alpha = ((overflowHighlighted ? 0.95f : 0.72f)) * u.pointAlphaScale;
  if (u.glossView != 0 && !overflowHighlighted) {
    const float glossPresence = glossPresenceWeight(glossStrengthCue(plotR, plotG, plotB));
    alpha = (0.01f + 0.97f * glossPresence) * u.pointAlphaScale;
  }
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

__device__ void writeHiddenInputPoint(float* verts, float* colors, unsigned int index) {
  const unsigned int vbase = index * 3u;
  verts[vbase + 0u] = 0.0f;
  verts[vbase + 1u] = 0.0f;
  verts[vbase + 2u] = 0.0f;
  const unsigned int cbase = index * 4u;
  colors[cbase + 0u] = 0.0f;
  colors[cbase + 1u] = 0.0f;
  colors[cbase + 2u] = 0.0f;
  colors[cbase + 3u] = 0.0f;
}

__device__ bool rasterSourceRowInRange(int y, int y1, int y2) {
  return y1 >= 0 && y2 > y1 && y >= y1 && y < y2;
}

__device__ bool rasterSourceRowInCube(const RasterSourceKernelUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityCubeY1, u.identityCubeY2);
}

__device__ bool rasterSourceRowInRamp(const RasterSourceKernelUniforms& u, int y) {
  return rasterSourceRowInRange(y, u.identityRampY1, u.identityRampY2);
}

__device__ bool rasterAppendSampleCoords(unsigned int index,
                                         int offset,
                                         int count,
                                         int y1,
                                         int y2,
                                         int rowStep,
                                         int xStep,
                                         int sourceWidth,
                                         int* outX,
                                         int* outY) {
  if (!outX || !outY || count <= 0 || index < static_cast<unsigned int>(max(offset, 0)) ||
      index >= static_cast<unsigned int>(max(offset + count, offset))) {
    return false;
  }
  const int local = static_cast<int>(index) - offset;
  const int safeXStep = max(xStep, 1);
  const int safeRowStep = max(rowStep, 1);
  const int samplesPerRow = max(1, (max(sourceWidth, 0) + safeXStep - 1) / safeXStep);
  const int rowIndex = local / samplesPerRow;
  const int xIndex = local - rowIndex * samplesPerRow;
  *outX = min(max(xIndex * safeXStep, 0), max(sourceWidth - 1, 0));
  *outY = min(max(y1 + rowIndex * safeRowStep, y1), max(y2 - 1, y1));
  return true;
}

__device__ float rasterHalton(unsigned int index, unsigned int base) {
  float f = 1.0f;
  float r = 0.0f;
  while (index > 0u) {
    f /= static_cast<float>(base);
    r += f * static_cast<float>(index % base);
    index /= base;
  }
  return r;
}

__device__ bool rasterOccupancySampleCoords(unsigned int index,
                                            const RasterSourceKernelUniforms& u,
                                            int* outX,
                                            int* outY) {
  if (!outX || !outY || u.occupancyAppendCount <= 0 ||
      index < static_cast<unsigned int>(max(u.occupancyAppendOffset, 0)) ||
      index >= static_cast<unsigned int>(max(u.occupancyAppendOffset + u.occupancyAppendCount,
                                             u.occupancyAppendOffset))) {
    return false;
  }
  const unsigned int local = index - static_cast<unsigned int>(max(u.occupancyAppendOffset, 0));
  const unsigned int attemptCount = static_cast<unsigned int>(max(u.occupancyCandidateCount, u.occupancyAppendCount));
  const unsigned int attempt = attemptCount > 0u
                                   ? (local * max(attemptCount, 1u)) /
                                         static_cast<unsigned int>(max(u.occupancyAppendCount, 1))
                                   : local;
  const float xNorm = rasterHalton(attempt + 1u, 2u);
  const float yNorm = rasterHalton(attempt + 1u, 3u);
  *outX = min(max(static_cast<int>(xNorm * static_cast<float>(max(u.sourceWidth, 1))), 0),
              max(u.sourceWidth - 1, 0));
  *outY = min(max(static_cast<int>(yNorm * static_cast<float>(max(u.sourceHeight, 1))), 0),
              max(u.sourceHeight - 1, 0));
  return true;
}

__device__ bool rasterLassoPointInStroke(const RasterSourceKernelUniforms& u,
                                         int strokeIndex,
                                         float xNorm,
                                         float yNorm) {
  if (strokeIndex < 0 || strokeIndex >= min(max(u.lassoStrokeCount, 0), 16)) return false;
  const int first = u.lassoStrokeFirst[strokeIndex];
  const int count = u.lassoStrokeCountPerStroke[strokeIndex];
  if (count < 3 || first < 0 || first + count > min(max(u.lassoPointCount, 0), 256)) return false;
  bool inside = false;
  for (int i = 0, j = count - 1; i < count; j = i++) {
    const float xi = u.lassoX[first + i];
    const float yi = u.lassoY[first + i];
    const float xj = u.lassoX[first + j];
    const float yj = u.lassoY[first + j];
    const bool intersects = ((yi > yNorm) != (yj > yNorm)) &&
                            (xNorm < (xj - xi) * (yNorm - yi) / ((yj - yi) + 1.0e-12f) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

__device__ bool rasterLassoContainsPoint(const RasterSourceKernelUniforms& u,
                                         float xNorm,
                                         float yNorm) {
  bool inside = false;
  const int strokeCount = min(max(u.lassoStrokeCount, 0), 16);
  for (int stroke = 0; stroke < strokeCount; ++stroke) {
    if (!rasterLassoPointInStroke(u, stroke, xNorm, yNorm)) continue;
    inside = u.lassoStrokeSubtract[stroke] == 0;
  }
  return inside;
}

__device__ float rasterNeutralRadius(float r, float g, float b, const RasterSourceKernelUniforms& u) {
  constexpr float kRgbAxisMaxRadius = 0.8164965809277260f;
  constexpr float kPolarMax = 0.9553166181245093f;
  constexpr float kChenPolarScale = 1.0467733744265997f;
  const int mode = u.input.plotMode;
  if (mode == 1) {
    const float cMax = fmaxf(r, fmaxf(g, b));
    const float cMin = fminf(r, fminf(g, b));
    if (u.input.circularHsl != 0) {
      const float l = 0.5f * (cMax + cMin);
      float denom = 1.0f - fabsf(2.0f * l - 1.0f);
      if (fabsf(denom) <= 1e-6f) denom = denom < 0.0f ? -1e-6f : 1e-6f;
      return clamp01(fabsf((cMax - cMin) / denom));
    }
    return clamp01(cMax - cMin);
  }
  if (mode == 2) {
    if (u.input.circularHsv != 0) {
      const float cMax = fmaxf(r, fmaxf(g, b));
      const float cMin = fminf(r, fminf(g, b));
      const float delta = cMax - cMin;
      return (delta > 1e-6f && cMax > 1e-6f) ? clamp01(delta / cMax) : 0.0f;
    }
    const float x = r - 0.5f * g - 0.5f * b;
    const float z = 0.8660254037844386f * (g - b);
    return clamp01(sqrtf(x * x + z * z));
  }
  const bool overflowMode = u.input.showOverflow != 0 && (mode == 5 || mode == 6 || mode == 7);
  const float rr = overflowMode ? r : clamp01(r);
  const float gg = overflowMode ? g : clamp01(g);
  const float bb = overflowMode ? b : clamp01(b);
  const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
  const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
  const float rotZ = 0.57735026919f * (rr + gg + bb);
  const float chromaRadius = sqrtf(rotX * rotX + rotY * rotY);
  if (mode == 3) {
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    const float polar = atanf(chromaRadius / fmaxf(rotZ, 1e-8f));
    const float light = radius3 * 0.5773502691896258f;
    const float radial = light * sinf(polar * kChenPolarScale) / kRgbAxisMaxRadius;
    return clamp01(radial);
  }
  if (mode == 4 || mode == 5) {
    const float radius3 = sqrtf(rotX * rotX + rotY * rotY + rotZ * rotZ);
    const float polar = atan2f(chromaRadius, rotZ);
    const float radial = radius3 * sinf((polar / kPolarMax) * kPolarMax);
    return clamp01(radial / sinf(kPolarMax));
  }
  if (mode == 6) {
    const float polar = atan2f(chromaRadius, rotZ);
    return clamp01(polar / kPolarMax);
  }
  if (mode == 7) {
    const float rotZAvg = (rr + gg + bb) / 3.0f;
    const float rx = 0.33333333333f * (2.0f * rr - gg - bb) * 0.70710678118f;
    const float ry = (gg - bb) * 0.40824829046f;
    const float sat = fabsf(rotZAvg) <= 1e-6f ? 0.0f : sqrtf(rx * rx + ry * ry) / rotZAvg;
    return clamp01(fabsf(sat) / 1.41421356237f);
  }
  return clamp01(sqrtf(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius);
}

__device__ bool rasterCubeSliceContains(float r, float g, float b, const RasterSourceKernelUniforms& u) {
  if (u.neutralRadiusEnabled != 0 && u.input.plotMode != 8 && u.input.showOverflow == 0) {
    const float threshold = clamp01(u.neutralRadius) * clamp01(u.neutralRadius);
    if (rasterNeutralRadius(r, g, b, u) > threshold + 1.0e-6f) return false;
  }
  if (u.cubeSlicingEnabled == 0) return true;
  const bool anySelected = u.cubeSliceRed || u.cubeSliceYellow || u.cubeSliceGreen ||
                           u.cubeSliceCyan || u.cubeSliceBlue || u.cubeSliceMagenta;
  if (!anySelected) return false;
  if (u.input.plotMode == 0 || u.input.glossView != 0) {
    constexpr float kEps = 1.0e-6f;
    const bool geRG = r + kEps >= g;
    const bool geGB = g + kEps >= b;
    const bool geGR = g + kEps >= r;
    const bool geRB = r + kEps >= b;
    const bool geBG = b + kEps >= g;
    const bool geBR = b + kEps >= r;
    if (u.cubeSliceRed && geRG && geGB) return true;
    if (u.cubeSliceYellow && geGR && geRB) return true;
    if (u.cubeSliceGreen && geGB && geBR) return true;
    if (u.cubeSliceCyan && geBG && geGR) return true;
    if (u.cubeSliceBlue && geBR && geRG) return true;
    if (u.cubeSliceMagenta && geRB && geBG) return true;
    return false;
  }
  const float cMax = fmaxf(r, fmaxf(g, b));
  const float cMin = fminf(r, fminf(g, b));
  const float delta = cMax - cMin;
  if (delta <= 1.0e-6f) return false;
  const float hue = wrapHue01(rawRgbHue01(r, g, b, cMax, delta));
  const int sector = static_cast<int>(floorf((hue + (1.0f / 12.0f)) * 6.0f)) % 6;
  if (sector == 0) return u.cubeSliceRed != 0;
  if (sector == 1) return u.cubeSliceYellow != 0;
  if (sector == 2) return u.cubeSliceGreen != 0;
  if (sector == 3) return u.cubeSliceCyan != 0;
  if (sector == 4) return u.cubeSliceBlue != 0;
  return u.cubeSliceMagenta != 0;
}

__device__ float halfBitsToFloatDevice(unsigned short h) {
  const float signScale = (h & 0x8000u) != 0u ? -1.0f : 1.0f;
  unsigned int exp = static_cast<unsigned int>(h & 0x7C00u) >> 10u;
  unsigned int mant = static_cast<unsigned int>(h & 0x03FFu);
  if (exp == 0u) {
    return signScale * ldexpf(static_cast<float>(mant), -24);
  }
  const unsigned int sign = (static_cast<unsigned int>(h & 0x8000u)) << 16u;
  unsigned int bits = 0u;
  if (exp == 31u) {
    bits = sign | 0x7F800000u | (mant << 13u);
  } else {
    bits = sign | ((exp + 112u) << 23u) | (mant << 13u);
  }
  return __uint_as_float(bits);
}

__device__ void readRasterSourceRgb(const unsigned char* source,
                                    RasterSourceKernelUniforms u,
                                    int x,
                                    int y,
                                    float* r,
                                    float* g,
                                    float* b) {
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  const unsigned int pixel = static_cast<unsigned int>(y * u.sourceWidth + x);
  if (u.pixelFormat == 1) {
    const float* values = reinterpret_cast<const float*>(source);
    const unsigned int base = pixel * 4u;
    *r = values[base + 0u];
    *g = values[base + 1u];
    *b = values[base + 2u];
    return;
  }
  const unsigned short* values = reinterpret_cast<const unsigned short*>(source);
  const unsigned int base = pixel * 4u;
  *r = halfBitsToFloatDevice(values[base + 0u]);
  *g = halfBitsToFloatDevice(values[base + 1u]);
  *b = halfBitsToFloatDevice(values[base + 2u]);
}

__device__ int rasterOccupancyBinIndex(float r, float g, float b) {
  constexpr int binsPerAxis = 18;
  auto toBin = [](float value) {
    if (value < 0.0f) return 0;
    if (value > 1.0f) return 17;
    return 1 + min(max(static_cast<int>(floorf(value * 16.0f)), 0), 15);
  };
  return (toBin(r) * binsPerAxis + toBin(g)) * binsPerAxis + toBin(b);
}

__device__ bool rasterSampleVisible(const RasterSourceKernelUniforms& u,
                                    int x,
                                    int y,
                                    float xNorm,
                                    float yNorm,
                                    float r,
                                    float g,
                                    float b) {
  const bool inCubeStrip = rasterSourceRowInCube(u, y);
  const bool inRampStrip = rasterSourceRowInRamp(u, y);
  const bool inAnyIdentityStrip = inCubeStrip || inRampStrip;
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

__device__ void rasterReadTransformedSample(const unsigned char* source,
                                            const RasterSourceKernelUniforms& u,
                                            int x,
                                            int y,
                                            float* r,
                                            float* g,
                                            float* b) {
  readRasterSourceRgb(source, u, x, y, r, g, b);
  if (u.plotLinear != 0 && u.input.plotMode != 8) {
    *r = decodeTransferChannel(*r, u.plotLinearTransfer);
    *g = decodeTransferChannel(*g, u.plotLinearTransfer);
    *b = decodeTransferChannel(*b, u.plotLinearTransfer);
  }
}

__global__ void rasterOccupancyCountKernel(const unsigned char* source,
                                           RasterSourceKernelUniforms u,
                                           int* occupancyBins,
                                           int* visibleCount) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(u.basePointCount, 0));
  if (index >= total || !source || !occupancyBins || !visibleCount) return;
  const int sampleCountX = max(u.sampleCountX, 1);
  const int stride = max(u.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  x = min(max(x, 0), max(u.sourceWidth - 1, 0));
  y = min(max(y, 0), max(u.sourceHeight - 1, 0));
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(max(u.sourceWidth, 1));
  const float yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(max(u.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, u, x, y, &r, &g, &b);
  if (!rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b)) return;
  atomicAdd(&occupancyBins[rasterOccupancyBinIndex(r, g, b)], 1);
  atomicAdd(visibleCount, 1);
}

__global__ void rasterOccupancyThresholdKernel(const int* visibleCount,
                                               int* occupancyTargetThreshold) {
  if (!visibleCount || !occupancyTargetThreshold || blockIdx.x != 0 || threadIdx.x != 0) return;
  constexpr int kRasterOccupancyBinCount = 18 * 18 * 18;
  const float meanOccupancy =
      static_cast<float>(max(*visibleCount, 0)) / static_cast<float>(kRasterOccupancyBinCount);
  *occupancyTargetThreshold = max(0, static_cast<int>(ceilf(meanOccupancy * 0.72f)));
}

__global__ void inputKernel(float* verts, float* colors, const float* input, InputKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(u.pointCount, 0));
  if (index >= total) return;
  const unsigned int stride = static_cast<unsigned int>(max(u.inputStride, 3));
  const unsigned int ibase = index * stride;
  float xNorm = 0.5f;
  float yNorm = 0.5f;
  float r = input[ibase + 0u];
  float g = input[ibase + 1u];
  float b = input[ibase + 2u];
  if (u.glossView != 0 && stride >= 6u) {
    xNorm = fminf(fmaxf(input[ibase + 0u], 0.0f), 1.0f);
    yNorm = fminf(fmaxf(input[ibase + 1u], 0.0f), 1.0f);
    r = input[ibase + 3u];
    g = input[ibase + 4u];
    b = input[ibase + 5u];
  }
  writeMappedInputPoint(verts, colors, index, xNorm, yNorm, r, g, b, u);
}

__global__ void rasterSourceKernel(float* verts,
                                   float* colors,
                                   const unsigned char* source,
                                   RasterSourceKernelUniforms u,
                                   const int* occupancyBins,
                                   const int* occupancyTargetThreshold) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(u.input.pointCount, 0));
  if (index >= total || !source) return;
  const int sampleCountX = max(u.sampleCountX, 1);
  const int stride = max(u.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
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
  const float xNorm = (static_cast<float>(min(max(x, 0), max(u.sourceWidth - 1, 0))) + 0.5f) /
                      static_cast<float>(max(u.sourceWidth, 1));
  const float yNorm = (static_cast<float>(min(max(y, 0), max(u.sourceHeight - 1, 0))) + 0.5f) /
                      static_cast<float>(max(u.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, u, x, y, &r, &g, &b);
  bool visible = rasterSampleVisible(u, x, y, xNorm, yNorm, r, g, b);
  if (visible && occupancyCandidate && occupancyBins) {
    const int bin = rasterOccupancyBinIndex(r, g, b);
    const int threshold =
        occupancyTargetThreshold ? max(*occupancyTargetThreshold, 0) : max(u.occupancyTargetThreshold, 0);
    visible = occupancyBins[bin] <= threshold;
  }
  if (!visible) {
    writeHiddenInputPoint(verts, colors, index);
    return;
  }
  writeMappedInputPoint(verts, colors, index, xNorm, yNorm, r, g, b, u.input);
}

__global__ void boundsKernel(const float* verts, const float* colors, unsigned int* boundsVals, int pointCount) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<unsigned int>(max(pointCount, 0))) return;
  if (colors && colors[index * 4u + 3u] <= 1.0e-6f) return;
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

inline __device__ void accumulateScopeDensity(unsigned int* density,
                                              const ScopeDensityRequest& request,
                                              int channel,
                                              float xNorm,
                                              float value) {
  const int channelCount = max(request.channelCount, 1);
  if (channel < 0 || channel >= channelCount) return;
  const bool overflowValue = value < 0.0f || value > 1.0f;
  if (request.onlyOverflow != 0 && !overflowValue) return;
  if (request.onlyOverflow == 0 && request.excludeOverflow != 0 && overflowValue) return;
  const int width = max(request.width, 1);
  const int height = max(request.height, 1);
  const int x = min(max(static_cast<int>(xNorm * static_cast<float>(width)), 0), width - 1);
  const int signalBins = request.waveform != 0 ? height : width;
  const int y = min(max(static_cast<int>((value - request.rangeMin) * request.invRange *
                                         static_cast<float>(signalBins)),
                        0),
                    signalBins - 1);
  const unsigned int binIndex =
      request.waveform != 0
          ? static_cast<unsigned int>((channel * width + x) * height + y)
          : static_cast<unsigned int>(channel * width + y);
  atomicAdd(&density[binIndex], 1u);
}

inline __device__ float scopeLuma(float r, float g, float b, int method) {
  switch (method) {
    case 1:
      return 0.2627f * r + 0.6780f * g + 0.0593f * b;
    case 2:
      return 0.2990f * r + 0.5870f * g + 0.1140f * b;
    case 3:
      return (r + g + b) / 3.0f;
    default:
      return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }
}

__global__ void scopeDensityKernel(const float* samples, unsigned int* density, ScopeDensityRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(request.pointCount, 0));
  if (index >= total || !samples || !density) return;
  const unsigned int base = index * 5u;
  const float xNorm = samples[base + 0u];
  const float r = samples[base + 2u];
  const float g = samples[base + 3u];
  const float b = samples[base + 4u];
  const bool lumaOnly =
      (request.waveform != 0 && request.scopeMode == 2) ||
      (request.waveform == 0 && request.scopeMode == 1);

  if (lumaOnly) {
    accumulateScopeDensity(density, request, 0, xNorm, scopeLuma(r, g, b, request.lumaMethod));
  } else {
    accumulateScopeDensity(density, request, 0, xNorm, r);
    accumulateScopeDensity(density, request, 1, xNorm, g);
    accumulateScopeDensity(density, request, 2, xNorm, b);
    if (request.waveform != 0 && request.scopeMode == 1 && request.channelCount >= 4) {
      accumulateScopeDensity(density, request, 3, xNorm, scopeLuma(r, g, b, request.lumaMethod));
    }
  }
}

__global__ void rasterScopeDensityKernel(const unsigned char* source,
                                         RasterSourceKernelUniforms raster,
                                         ScopeDensityRequest scope,
                                         const unsigned int* rangeBits,
                                         unsigned int* density) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(raster.input.pointCount, 0));
  if (index >= total || !source || !density) return;
  if (rangeBits) {
    const float rangeMin = floatFromOrderedUintDevice(rangeBits[0]);
    const float rangeMax = floatFromOrderedUintDevice(rangeBits[1]);
    scope.rangeMin = rangeMin;
    scope.invRange = 1.0f / fmaxf(1.0e-6f, rangeMax - rangeMin);
  }

  const int sampleCountX = max(raster.sampleCountX, 1);
  const int stride = max(raster.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  bool haveCoords = index < static_cast<unsigned int>(max(raster.basePointCount, 0));
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
  if (!haveCoords) return;

  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(max(raster.sourceWidth, 1));
  const float yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(max(raster.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, raster, x, y, &r, &g, &b);
  if (!rasterSampleVisible(raster, x, y, xNorm, yNorm, r, g, b)) return;

  const bool lumaOnly =
      (scope.waveform != 0 && scope.scopeMode == 2) ||
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

inline __device__ void includeScopeRangeValue(float value,
                                              const ScopeRangeRequest& range,
                                              unsigned int* rangeBits) {
  if (!rangeBits) return;
  if (range.includeOverflow == 0 && (value < 0.0f || value > 1.0f)) return;
  const unsigned int ordered = orderedUintFromFloat(value);
  atomicMin(&rangeBits[0], ordered);
  atomicMax(&rangeBits[1], ordered);
  atomicAdd(&rangeBits[2], 1u);
}

__global__ void rasterScopeRangeKernel(const unsigned char* source,
                                       RasterSourceKernelUniforms raster,
                                       ScopeRangeRequest range,
                                       unsigned int* rangeBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(raster.input.pointCount, 0));
  if (index >= total || !source || !rangeBits) return;

  const int sampleCountX = max(raster.sampleCountX, 1);
  const int stride = max(raster.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  bool haveCoords = index < static_cast<unsigned int>(max(raster.basePointCount, 0));
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
  if (!haveCoords) return;

  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(max(raster.sourceWidth, 1));
  const float yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(max(raster.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, raster, x, y, &r, &g, &b);
  if (!rasterSampleVisible(raster, x, y, xNorm, yNorm, r, g, b)) return;

  const bool lumaOnly =
      (range.waveform != 0 && range.scopeMode == 2) ||
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

inline __device__ void includeScopeRangeHistogramValue(float value,
                                                       const ScopeRangeRequest& range,
                                                       float minValue,
                                                       float invRange,
                                                       unsigned int histogramBinCount,
                                                       unsigned int* histogram) {
  if (!histogram || histogramBinCount == 0u) return;
  if (range.includeOverflow == 0 && (value < 0.0f || value > 1.0f)) return;
  const float scaled = (value - minValue) * invRange;
  const unsigned int bin =
      min(histogramBinCount - 1u,
          static_cast<unsigned int>(max(0, static_cast<int>(floorf(scaled * static_cast<float>(histogramBinCount))))));
  atomicAdd(&histogram[bin], 1u);
}

__global__ void rasterScopeRangeHistogramKernel(const unsigned char* source,
                                                RasterSourceKernelUniforms raster,
                                                ScopeRangeRequest range,
                                                const unsigned int* rangeBits,
                                                unsigned int histogramBinCount,
                                                unsigned int* histogram) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(raster.input.pointCount, 0));
  if (index >= total || !source || !histogram || !rangeBits) return;
  const float minValue = floatFromOrderedUintDevice(rangeBits[0]);
  const float maxValue = floatFromOrderedUintDevice(rangeBits[1]);
  const float invRange = 1.0f / fmaxf(1.0e-7f, maxValue - minValue);

  const int sampleCountX = max(raster.sampleCountX, 1);
  const int stride = max(raster.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  bool haveCoords = index < static_cast<unsigned int>(max(raster.basePointCount, 0));
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
  if (!haveCoords) return;

  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(max(raster.sourceWidth, 1));
  const float yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(max(raster.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, raster, x, y, &r, &g, &b);
  if (!rasterSampleVisible(raster, x, y, xNorm, yNorm, r, g, b)) return;

  const bool lumaOnly =
      (range.waveform != 0 && range.scopeMode == 2) ||
      (range.waveform == 0 && range.scopeMode == 1);
  if (lumaOnly) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogramBinCount,
                                    histogram);
    return;
  }
  if (range.includeRed != 0) {
    includeScopeRangeHistogramValue(r, range, minValue, invRange, histogramBinCount, histogram);
  }
  if (range.includeGreen != 0) {
    includeScopeRangeHistogramValue(g, range, minValue, invRange, histogramBinCount, histogram);
  }
  if (range.includeBlue != 0) {
    includeScopeRangeHistogramValue(b, range, minValue, invRange, histogramBinCount, histogram);
  }
  if (range.waveform != 0 && range.scopeMode == 1 && range.includeLuma != 0) {
    includeScopeRangeHistogramValue(scopeLuma(r, g, b, range.lumaMethod),
                                    range,
                                    minValue,
                                    invRange,
                                    histogramBinCount,
                                    histogram);
  }
}

__global__ void scopeRangeHistogramPercentileKernel(const unsigned int* histogram,
                                                    unsigned int histogramBinCount,
                                                    const unsigned int* rangeBits,
                                                    unsigned int* percentileBits) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || !histogram || !percentileBits ||
      !rangeBits || histogramBinCount == 0u || rangeBits[2] == 0u) {
    return;
  }
  const float minValue = floatFromOrderedUintDevice(rangeBits[0]);
  const float maxValue = floatFromOrderedUintDevice(rangeBits[1]);
  const unsigned int totalCount = rangeBits[2];
  const unsigned long long lowTarget = static_cast<unsigned long long>(totalCount) / 1000ull;
  const unsigned long long highTarget =
      (static_cast<unsigned long long>(totalCount) * 999ull) / 1000ull;
  unsigned long long accumulated = 0ull;
  float lowValue = minValue;
  float highValue = maxValue;
  bool foundLow = false;
  bool foundHigh = false;
  const float range = maxValue - minValue;
  for (unsigned int bin = 0u; bin < histogramBinCount; ++bin) {
    accumulated += static_cast<unsigned long long>(histogram[bin]);
    if (!foundLow && accumulated > lowTarget) {
      const float t = (static_cast<float>(bin) + 0.5f) / static_cast<float>(histogramBinCount);
      lowValue = minValue + t * range;
      foundLow = true;
    }
    if (!foundHigh && accumulated > highTarget) {
      const float t = (static_cast<float>(bin) + 0.5f) / static_cast<float>(histogramBinCount);
      highValue = minValue + t * range;
      foundHigh = true;
      break;
    }
  }
  percentileBits[0] = orderedUintFromFloat(lowValue);
  percentileBits[1] = orderedUintFromFloat(highValue);
}

__global__ void scopeRangeFinalizeKernel(const unsigned int* percentileBits,
                                         const unsigned int* rangeBits,
                                         int previousRangeValid,
                                         float previousRangeMin,
                                         float previousRangeMax,
                                         unsigned int* finalRangeBits) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || !percentileBits || !rangeBits || !finalRangeBits) return;
  if (rangeBits[2] == 0u) {
    finalRangeBits[0] = orderedUintFromFloat(0.0f);
    finalRangeBits[1] = orderedUintFromFloat(1.0f);
    finalRangeBits[2] = 0u;
    return;
  }
  float rangeMin = fminf(0.0f, floatFromOrderedUintDevice(percentileBits[0]));
  float rangeMax = fmaxf(1.0f, floatFromOrderedUintDevice(percentileBits[1]));
  const float pad = fmaxf(0.02f, (rangeMax - rangeMin) * 0.04f);
  rangeMin -= pad;
  rangeMax += pad;
  if (!(rangeMax > rangeMin + 1.0e-5f)) {
    rangeMin = 0.0f;
    rangeMax = 1.0f;
  }
  if (previousRangeValid != 0 && previousRangeMax > previousRangeMin + 1.0e-5f) {
    rangeMin = rangeMin < previousRangeMin
                   ? rangeMin
                   : previousRangeMin + (rangeMin - previousRangeMin) * 0.16f;
    rangeMax = rangeMax > previousRangeMax
                   ? rangeMax
                   : previousRangeMax + (rangeMax - previousRangeMax) * 0.16f;
  }
  finalRangeBits[0] = orderedUintFromFloat(rangeMin);
  finalRangeBits[1] = orderedUintFromFloat(rangeMax);
  finalRangeBits[2] = rangeBits[2];
}

__global__ void scopeDensityMaxUintKernel(const unsigned int* density,
                                          unsigned int binCount,
                                          unsigned int* maxValue) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= binCount || !density || !maxValue) return;
  atomicMax(maxValue, density[index]);
}

inline __device__ void waveformChannelColorDevice(int channel,
                                                  int lumaOnly,
                                                  float* r,
                                                  float* g,
                                                  float* b) {
  if (lumaOnly != 0 || channel == 3) {
    *r = 0.88f;
    *g = 0.92f;
    *b = 0.96f;
    return;
  }
  if (channel == 0) {
    *r = 1.00f;
    *g = 0.12f;
    *b = 0.04f;
  } else if (channel == 1) {
    *r = 0.12f;
    *g = 1.00f;
    *b = 0.24f;
  } else {
    *r = 0.20f;
    *g = 0.46f;
    *b = 1.00f;
  }
}

inline __device__ void waveformApplySaturationDevice(float saturation,
                                                     float* r,
                                                     float* g,
                                                     float* b) {
  const float effective = 0.34f + (1.0f - 0.34f) * clamp01(saturation);
  const float luma = 0.2126f * *r + 0.7152f * *g + 0.0722f * *b;
  *r = clamp01(luma + (*r - luma) * effective);
  *g = clamp01(luma + (*g - luma) * effective);
  *b = clamp01(luma + (*b - luma) * effective);
}

__global__ void waveformScopeDensityToPointsKernel(float* verts,
                                                   float* colors,
                                                   const unsigned int* density,
                                                   const unsigned int* overflowDensity,
                                                   unsigned int binCount,
                                                   const unsigned int* maxDensity,
                                                   WaveformScopePointRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int layerCount = request.showOverflow != 0 && overflowDensity ? 2u : 1u;
  const unsigned int total = binCount * layerCount;
  if (index >= total || !verts || !colors || !density || !maxDensity) return;

  const bool overflowLayer = index >= binCount;
  const unsigned int local = overflowLayer ? index - binCount : index;
  const unsigned int rawDensity = overflowLayer ? overflowDensity[local] : density[local];
  const int width = max(request.density.width, 1);
  const int height = max(request.density.height, 1);
  const int channelCount = max(request.density.channelCount, 1);
  const int y = static_cast<int>(local % static_cast<unsigned int>(height));
  const unsigned int xAndChannel = local / static_cast<unsigned int>(height);
  const int x = static_cast<int>(xAndChannel % static_cast<unsigned int>(width));
  const int channel = static_cast<int>(xAndChannel / static_cast<unsigned int>(width));
  const bool enabled = channel >= 0 && channel < channelCount &&
                       channel < 4 && request.channelEnabled[channel] != 0;
  const unsigned int vertBase = index * 3u;
  const unsigned int colorBase = index * 4u;
  const unsigned int safeMaxDensity = max(maxDensity[0], 1u);
  if (!enabled || rawDensity == 0u) {
    verts[vertBase + 0u] = 0.0f;
    verts[vertBase + 1u] = 0.0f;
    verts[vertBase + 2u] = 0.0f;
    colors[colorBase + 0u] = 0.0f;
    colors[colorBase + 1u] = 0.0f;
    colors[colorBase + 2u] = 0.0f;
    colors[colorBase + 3u] = 0.0f;
    return;
  }

  constexpr float kLeft = -0.82f;
  constexpr float kRight = 0.96f;
  constexpr float kBottom = -0.88f;
  constexpr float kTop = 0.88f;
  constexpr float kWidth = kRight - kLeft;
  constexpr float kHeight = kTop - kBottom;
  const int parade = request.density.scopeMode == 1 ? 1 : 0;
  const int lumaOnly = request.density.scopeMode == 2 ? 1 : 0;
  const float plotX =
      parade != 0 && lumaOnly == 0
          ? kLeft + (static_cast<float>(channel) +
                     (static_cast<float>(x) + 0.5f) / static_cast<float>(width)) *
                        (kWidth / static_cast<float>(channelCount))
          : kLeft + kWidth * (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
  const float plotY = kBottom + kHeight * (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
  const float reference = log1pf(static_cast<float>(safeMaxDensity));
  const float normalized = reference > 1.0e-6f ? clamp01(log1pf(static_cast<float>(rawDensity)) / reference) : 1.0f;
  const float intensity =
      clamp01((0.22f + 0.78f * powf(normalized, 0.62f)) * fmaxf(request.pointBrightness, 0.0f));
  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  waveformChannelColorDevice(channel, lumaOnly, &cr, &cg, &cb);
  if (overflowLayer && request.highlightOverflow != 0) {
    constexpr float overlay = 0.68f;
    cr = cr * (1.0f - overlay) + 0.82f * overlay;
    cg = cg * (1.0f - overlay) + 0.24f * overlay;
    cb = cb * (1.0f - overlay) + 1.00f * overlay;
  }
  waveformApplySaturationDevice(request.colorSaturation, &cr, &cg, &cb);
  const float layerGain = overflowLayer ? 0.82f : 1.0f;
  verts[vertBase + 0u] = plotX;
  verts[vertBase + 1u] = plotY;
  verts[vertBase + 2u] = 0.0f;
  colors[colorBase + 0u] = cr * intensity * layerGain;
  colors[colorBase + 1u] = cg * intensity * layerGain;
  colors[colorBase + 2u] = cb * intensity * layerGain;
  colors[colorBase + 3u] = clamp01(request.coverageAlpha);
}

__global__ void histogramSmoothDensityKernel(const unsigned int* density,
                                             float* smoothed,
                                             int binCount,
                                             int channelCount) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(binCount, 0) * max(channelCount, 0));
  if (index >= total || !density || !smoothed) return;
  const int bins = max(binCount, 1);
  const int channel = static_cast<int>(index / static_cast<unsigned int>(bins));
  const int bin = static_cast<int>(index % static_cast<unsigned int>(bins));
  constexpr float weights[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
  float sum = 0.0f;
  float weightSum = 0.0f;
  for (int tap = -2; tap <= 2; ++tap) {
    const int sourceBin = min(max(bin + tap, 0), bins - 1);
    const float weight = weights[tap + 2];
    sum += static_cast<float>(density[channel * bins + sourceBin]) * weight;
    weightSum += weight;
  }
  smoothed[index] = weightSum > 0.0f ? sum / weightSum : 0.0f;
}

__global__ void scopeDensityMaxFloatKernel(const float* density,
                                           unsigned int binCount,
                                           unsigned int* maxValue) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= binCount || !density || !maxValue) return;
  const float value = fmaxf(density[index], 0.0f);
  atomicMax(maxValue, orderedUintFromFloat(value));
}

inline __device__ void histogramBaseColorDevice(int channel,
                                                int lumaOnly,
                                                float* r,
                                                float* g,
                                                float* b) {
  if (lumaOnly != 0) {
    *r = 0.88f;
    *g = 0.92f;
    *b = 0.96f;
    return;
  }
  if (channel == 0) {
    *r = 1.00f;
    *g = 0.16f;
    *b = 0.12f;
  } else if (channel == 1) {
    *r = 0.20f;
    *g = 1.00f;
    *b = 0.28f;
  } else {
    *r = 0.24f;
    *g = 0.52f;
    *b = 1.00f;
  }
}

inline __device__ void histogramOverflowColorDevice(int channel,
                                                    int lumaOnly,
                                                    int highlighted,
                                                    float* r,
                                                    float* g,
                                                    float* b) {
  if (lumaOnly != 0) {
    *r = 0.88f;
    *g = 0.92f;
    *b = 0.96f;
  } else if (channel == 0) {
    *r = 1.00f;
    *g = 0.12f;
    *b = 0.04f;
  } else if (channel == 1) {
    *r = 0.12f;
    *g = 1.00f;
    *b = 0.24f;
  } else {
    *r = 0.20f;
    *g = 0.46f;
    *b = 1.00f;
  }
  if (highlighted == 0) return;
  constexpr float overlay = 0.68f;
  *r = *r * (1.0f - overlay) + 0.82f * overlay;
  *g = *g * (1.0f - overlay) + 0.24f * overlay;
  *b = *b * (1.0f - overlay) + 1.00f * overlay;
}

__global__ void histogramScopeDensityToGeometryKernel(float* lineVerts,
                                                      float* lineColors,
                                                      float* fillVerts,
                                                      float* fillColors,
                                                      const float* density,
                                                      const float* overflowDensity,
                                                      unsigned int segmentCount,
                                                      const unsigned int* maxDensityBits,
                                                      HistogramScopeGeometryRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int layerCount = request.showOverflow != 0 && overflowDensity ? 2u : 1u;
  const unsigned int total = segmentCount * layerCount;
  if (index >= total || !lineVerts || !lineColors || !fillVerts || !fillColors ||
      !density || !maxDensityBits) return;

  constexpr int kSignalBins = 512;
  constexpr float kLeft = -0.82f;
  constexpr float kRight = 0.96f;
  constexpr float kBottom = -0.88f;
  constexpr float kTop = 0.88f;
  constexpr float kWidth = kRight - kLeft;
  constexpr float kHeight = kTop - kBottom;
  const int channelCount = max(request.density.channelCount, 1);
  const bool overflowLayer = index >= segmentCount;
  const unsigned int local = overflowLayer ? index - segmentCount : index;
  const int bin = static_cast<int>(local % static_cast<unsigned int>(kSignalBins - 1)) + 1;
  const int channel = static_cast<int>(local / static_cast<unsigned int>(kSignalBins - 1));
  const bool lumaOnly = request.density.scopeMode == 1;
  const size_t d0 = static_cast<size_t>(channel * kSignalBins + bin - 1);
  const size_t d1 = static_cast<size_t>(channel * kSignalBins + bin);
  const float previousDensity = overflowLayer ? (overflowDensity ? overflowDensity[d0] : 0.0f) : density[d0];
  const float currentDensity = overflowLayer ? (overflowDensity ? overflowDensity[d1] : 0.0f) : density[d1];
  const bool visible = channel >= 0 && channel < channelCount &&
                       (!overflowLayer || previousDensity > 0.0f || currentDensity > 0.0f);
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float lineAlpha = 0.0f;
  float fillAlpha = 0.0f;
  if (visible) {
    if (overflowLayer) {
      histogramOverflowColorDevice(channel, lumaOnly ? 1 : 0, request.highlightOverflow, &r, &g, &b);
      lineAlpha = 0.94f;
      fillAlpha = 0.16f;
    } else {
      histogramBaseColorDevice(channel, lumaOnly ? 1 : 0, &r, &g, &b);
      lineAlpha = lumaOnly ? 0.88f : 0.76f;
      fillAlpha = lumaOnly ? 0.14f : 0.10f;
    }
  }
  const float safeMax = fmaxf(floatFromOrderedUintDevice(maxDensityBits[0]), 1.0f);
  const float x0 = kLeft + kWidth * static_cast<float>(bin - 1) / static_cast<float>(kSignalBins - 1);
  const float x1 = kLeft + kWidth * static_cast<float>(bin) / static_cast<float>(kSignalBins - 1);
  const float y0 = kBottom + (kHeight - 0.06f) * sqrtf(clamp01(previousDensity / safeMax));
  const float y1 = kBottom + (kHeight - 0.06f) * sqrtf(clamp01(currentDensity / safeMax));

  const unsigned int lineBase = index * 2u;
  const unsigned int lineVertBase = lineBase * 3u;
  const unsigned int lineColorBase = lineBase * 4u;
  lineVerts[lineVertBase + 0u] = visible ? x0 : 0.0f;
  lineVerts[lineVertBase + 1u] = visible ? y0 : 0.0f;
  lineVerts[lineVertBase + 2u] = 0.0f;
  lineVerts[lineVertBase + 3u] = visible ? x1 : 0.0f;
  lineVerts[lineVertBase + 4u] = visible ? y1 : 0.0f;
  lineVerts[lineVertBase + 5u] = 0.0f;
  for (int vertex = 0; vertex < 2; ++vertex) {
    const unsigned int base = lineColorBase + static_cast<unsigned int>(vertex) * 4u;
    lineColors[base + 0u] = r;
    lineColors[base + 1u] = g;
    lineColors[base + 2u] = b;
    lineColors[base + 3u] = visible ? lineAlpha : 0.0f;
  }

  const unsigned int fillBase = index * 6u;
  const unsigned int fillVertBase = fillBase * 3u;
  const float vx[6] = {x0, x0, x1, x0, x1, x1};
  const float vy[6] = {kBottom, y0, y1, kBottom, y1, kBottom};
  for (int vertex = 0; vertex < 6; ++vertex) {
    const unsigned int vbase = fillVertBase + static_cast<unsigned int>(vertex) * 3u;
    fillVerts[vbase + 0u] = visible ? vx[vertex] : 0.0f;
    fillVerts[vbase + 1u] = visible ? vy[vertex] : 0.0f;
    fillVerts[vbase + 2u] = 0.0f;
    const unsigned int cbase = fillBase * 4u + static_cast<unsigned int>(vertex) * 4u;
    const bool upperVertex = vertex == 1 || vertex == 2 || vertex == 4;
    fillColors[cbase + 0u] = r;
    fillColors[cbase + 1u] = g;
    fillColors[cbase + 2u] = b;
    fillColors[cbase + 3u] = visible ? (upperVertex ? fillAlpha : fillAlpha * 0.12f) : 0.0f;
  }
}

__global__ void glossFieldAccumulateKernel(const float* packedPoints,
                                           int pointCount,
                                           int gridWidth,
                                           int gridHeight,
                                           int showOverflow,
                                           float* occupancy,
                                           float* sumR,
                                           float* sumG,
                                           float* sumB,
                                           float* sumY,
                                           float* sumMax,
                                           float* sumMin,
                                           float* sumNeutrality) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(pointCount, 0));
  if (index >= total) return;
  const unsigned int base = index * 6u;
  const float xNorm = fminf(fmaxf(packedPoints[base + 0u], 0.0f), 1.0f);
  const float yNorm = fminf(fmaxf(packedPoints[base + 1u], 0.0f), 1.0f);
  float r = packedPoints[base + 3u];
  float g = packedPoints[base + 4u];
  float b = packedPoints[base + 5u];
  if (showOverflow == 0) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  const float maxRgb = clamp01(fmaxf(r, fmaxf(g, b)));
  const float minRgb = clamp01(fmaxf(0.0f, fminf(r, fminf(g, b))));
  const float neutrality = maxRgb > 1e-6f ? fminf(fmaxf(minRgb / maxRgb, 0.0f), 1.0f) : 0.0f;
  const float luma = clamp01(glossLuma(r, g, b));
  const int x = min(max(static_cast<int>(xNorm * static_cast<float>(gridWidth)), 0), max(gridWidth - 1, 0));
  const int y =
      min(max(static_cast<int>((1.0f - yNorm) * static_cast<float>(gridHeight)), 0), max(gridHeight - 1, 0));
  const unsigned int cellIndex = static_cast<unsigned int>(y * gridWidth + x);
  atomicAdd(&occupancy[cellIndex], 1.0f);
  atomicAdd(&sumR[cellIndex], r);
  atomicAdd(&sumG[cellIndex], g);
  atomicAdd(&sumB[cellIndex], b);
  atomicAdd(&sumY[cellIndex], luma);
  atomicAdd(&sumMax[cellIndex], maxRgb);
  atomicAdd(&sumMin[cellIndex], minRgb);
  atomicAdd(&sumNeutrality[cellIndex], neutrality);
}

inline __device__ float rasterGlossVisualTopNorm(float sourceBottomNorm) {
  // Source Signal rows are sampled in OFX/OpenGL bottom-origin order. Gloss field
  // cells and image-plane projection use visual top-origin rows, so keep that
  // conversion shared between field accumulation and resident source projection.
  return 1.0f - fminf(fmaxf(sourceBottomNorm, 0.0f), 1.0f);
}

__global__ void rasterGlossFieldAccumulateKernel(const unsigned char* source,
                                                 RasterSourceKernelUniforms raster,
                                                 int gridWidth,
                                                 int gridHeight,
                                                 int showOverflow,
                                                 float* occupancy,
                                                 float* sumR,
                                                 float* sumG,
                                                 float* sumB,
                                                 float* sumY,
                                                 float* sumMax,
                                                 float* sumMin,
                                                 float* sumNeutrality) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(raster.input.pointCount, 0));
  if (index >= total || !source) return;

  const int sampleCountX = max(raster.sampleCountX, 1);
  const int stride = max(raster.sampleStride, 1);
  int x = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int y = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  bool haveCoords = index < static_cast<unsigned int>(max(raster.basePointCount, 0));
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
  if (!haveCoords) return;

  x = min(max(x, 0), max(raster.sourceWidth - 1, 0));
  y = min(max(y, 0), max(raster.sourceHeight - 1, 0));
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(max(raster.sourceWidth, 1));
  const float yNorm = (static_cast<float>(y) + 0.5f) / static_cast<float>(max(raster.sourceHeight, 1));
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  rasterReadTransformedSample(source, raster, x, y, &r, &g, &b);
  if (!rasterSampleVisible(raster, x, y, xNorm, yNorm, r, g, b)) return;
  if (showOverflow == 0) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  const float maxRgb = clamp01(fmaxf(r, fmaxf(g, b)));
  const float minRgb = clamp01(fmaxf(0.0f, fminf(r, fminf(g, b))));
  const float neutrality = maxRgb > 1e-6f ? fminf(fmaxf(minRgb / maxRgb, 0.0f), 1.0f) : 0.0f;
  const float luma = clamp01(glossLuma(r, g, b));
  const int cellX = min(max(static_cast<int>(xNorm * static_cast<float>(gridWidth)), 0), max(gridWidth - 1, 0));
  const float visualTopNorm = rasterGlossVisualTopNorm(yNorm);
  const int cellY =
      min(max(static_cast<int>(visualTopNorm * static_cast<float>(gridHeight)), 0), max(gridHeight - 1, 0));
  const unsigned int cellIndex = static_cast<unsigned int>(cellY * gridWidth + cellX);
  atomicAdd(&occupancy[cellIndex], 1.0f);
  atomicAdd(&sumR[cellIndex], r);
  atomicAdd(&sumG[cellIndex], g);
  atomicAdd(&sumB[cellIndex], b);
  atomicAdd(&sumY[cellIndex], luma);
  atomicAdd(&sumMax[cellIndex], maxRgb);
  atomicAdd(&sumMin[cellIndex], minRgb);
  atomicAdd(&sumNeutrality[cellIndex], neutrality);
}

__global__ void glossFieldFinalizeKernel(int cellCount,
                                         const float* occupancy,
                                         const float* sumR,
                                         const float* sumG,
                                         const float* sumB,
                                         const float* sumY,
                                         const float* sumMax,
                                         const float* sumMin,
                                         const float* sumNeutrality,
                                         float* meanR,
                                         float* meanG,
                                         float* meanB,
                                         float* carrierY,
                                         float* carrierMax,
                                         float* carrierMin,
                                         float* neutrality) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total) return;
  const float count = occupancy[index];
  if (count <= 1e-6f) {
    meanR[index] = 0.0f;
    meanG[index] = 0.0f;
    meanB[index] = 0.0f;
    carrierY[index] = 0.0f;
    carrierMax[index] = 0.0f;
    carrierMin[index] = 0.0f;
    neutrality[index] = 0.0f;
    return;
  }
  const float invCount = 1.0f / count;
  meanR[index] = sumR[index] * invCount;
  meanG[index] = sumG[index] * invCount;
  meanB[index] = sumB[index] * invCount;
  carrierY[index] = sumY[index] * invCount;
  carrierMax[index] = sumMax[index] * invCount;
  carrierMin[index] = sumMin[index] * invCount;
  neutrality[index] = sumNeutrality[index] * invCount;
}

inline __device__ float sampleGridClampedDevice(const float* values, int width, int height, int x, int y) {
  if (!values || width <= 0 || height <= 0) return 0.0f;
  x = min(max(x, 0), width - 1);
  y = min(max(y, 0), height - 1);
  return values[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)];
}

__global__ void glossFieldMaxKernel(int cellCount, const float* values, unsigned int* outBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !outBits) return;
  const float value = fmaxf(values[index], 0.0f);
  atomicMax(&outBits[0], __float_as_uint(value));
}

__global__ void glossFieldNormalizeKernel(int cellCount,
                                          const float* src,
                                          float* dst,
                                          const unsigned int* maxBits,
                                          int signedValues) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !src || !dst || !maxBits) return;
  const float denom = fmaxf(__uint_as_float(maxBits[0]), 1e-5f);
  const float scaled = src[index] / denom;
  dst[index] = signedValues != 0 ? fminf(fmaxf(scaled, -1.0f), 1.0f) : clamp01(scaled);
}

__global__ void glossFieldBlurKernel(int gridWidth,
                                     int gridHeight,
                                     const float* src,
                                     float* dst) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !src || !dst) return;
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  float accum = 0.0f;
  float weight = 0.0f;
  for (int oy = -1; oy <= 1; ++oy) {
    const int yy = y + oy;
    if (yy < 0 || yy >= gridHeight) continue;
    for (int ox = -1; ox <= 1; ++ox) {
      const int xx = x + ox;
      if (xx < 0 || xx >= gridWidth) continue;
      const float kernel = (ox == 0 && oy == 0) ? 0.30f : ((ox == 0 || oy == 0) ? 0.13f : 0.08f);
      accum += src[static_cast<size_t>(yy) * static_cast<size_t>(gridWidth) + static_cast<size_t>(xx)] * kernel;
      weight += kernel;
    }
  }
  dst[index] = weight > 1e-6f ? (accum / weight) : 0.0f;
}

__global__ void glossFieldFinalNormalizeKernel(int cellCount,
                                               float* body,
                                               float* signal,
                                               float* positive,
                                               float* negative,
                                               float* boundary,
                                               const unsigned int* maxBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !body || !signal || !positive || !negative || !boundary || !maxBits) return;
  const float maxBody = fmaxf(__uint_as_float(maxBits[0]), 1e-5f);
  const float maxPositive = fmaxf(__uint_as_float(maxBits[1]), 1e-5f);
  const float maxNegative = fmaxf(__uint_as_float(maxBits[2]), 1e-5f);
  const float maxBoundary = fmaxf(__uint_as_float(maxBits[3]), 1e-5f);
  const float maxAbsSignal = fmaxf(fmaxf(maxPositive, maxNegative), 1e-5f);
  body[index] = clamp01(body[index] / maxBody);
  positive[index] = clamp01(positive[index] / maxPositive);
  negative[index] = clamp01(negative[index] / maxNegative);
  signal[index] = fminf(fmaxf(signal[index] / maxAbsSignal, -1.0f), 1.0f);
  boundary[index] = clamp01(boundary[index] / maxBoundary);
}

inline __device__ int glossTrimmedRadiusCellsDevice(int neighborhoodChoice) {
  if (neighborhoodChoice <= 0) return 1;
  if (neighborhoodChoice >= 2) return 3;
  return 2;
}

inline __device__ int glossAnalysisRadiusCellsDevice(int neighborhoodChoice) {
  if (neighborhoodChoice <= 0) return 3;
  if (neighborhoodChoice >= 2) return 10;
  return 6;
}

inline __device__ float glossHybridCarrierDevice(const float* carrierMax,
                                                 const float* carrierY,
                                                 const float* carrierMin,
                                                 size_t idx) {
  return 0.70f * carrierMax[idx] + 0.20f * carrierY[idx] + 0.10f * carrierMin[idx];
}

inline __device__ float glossGradientMagnitudeDevice(const float* values, int width, int height, int x, int y) {
  const float gx = 0.5f * (sampleGridClampedDevice(values, width, height, x + 1, y) -
                           sampleGridClampedDevice(values, width, height, x - 1, y));
  const float gy = 0.5f * (sampleGridClampedDevice(values, width, height, x, y + 1) -
                           sampleGridClampedDevice(values, width, height, x, y - 1));
  return sqrtf(gx * gx + gy * gy);
}

inline __device__ float glossGradientCongruenceDevice(const float* body,
                                                      const float* signal,
                                                      int width,
                                                      int height,
                                                      int x,
                                                      int y) {
  const float gxBody = 0.5f * (sampleGridClampedDevice(body, width, height, x + 1, y) -
                               sampleGridClampedDevice(body, width, height, x - 1, y));
  const float gyBody = 0.5f * (sampleGridClampedDevice(body, width, height, x, y + 1) -
                               sampleGridClampedDevice(body, width, height, x, y - 1));
  const float gxSignal = 0.5f * (sampleGridClampedDevice(signal, width, height, x + 1, y) -
                                 sampleGridClampedDevice(signal, width, height, x - 1, y));
  const float gySignal = 0.5f * (sampleGridClampedDevice(signal, width, height, x, y + 1) -
                                 sampleGridClampedDevice(signal, width, height, x, y - 1));
  const float magBody = sqrtf(gxBody * gxBody + gyBody * gyBody);
  const float magSignal = sqrtf(gxSignal * gxSignal + gySignal * gySignal);
  if (magBody > 1.0e-6f && magSignal > 1.0e-6f) {
    return clamp01(fabsf((gxBody * gxSignal + gyBody * gySignal) / (magBody * magSignal)));
  }
  return magSignal > 1.0e-6f ? 0.35f : 0.0f;
}

__global__ void glossFieldHybridCarrierKernel(int cellCount,
                                              const float* carrierMax,
                                              const float* carrierY,
                                              const float* carrierMin,
                                              float* outCarrier) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !carrierMax || !carrierY || !carrierMin || !outCarrier) return;
  outCarrier[index] = glossHybridCarrierDevice(carrierMax, carrierY, carrierMin, index);
}

__global__ void glossFieldTrimmedBodyKernel(int gridWidth,
                                            int gridHeight,
                                            int neighborhoodChoice,
                                            const float* occupancy,
                                            const float* meanR,
                                            const float* meanG,
                                            const float* meanB,
                                            const float* carrier,
                                            float* body) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !occupancy || !meanR || !meanG || !meanB || !carrier || !body) return;
  if (occupancy[index] <= 0.5f) {
    body[index] = 0.0f;
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const int radius = glossTrimmedRadiusCellsDevice(neighborhoodChoice);
  const float centerCarrier = carrier[index];
  const float centerR = meanR[index];
  const float centerG = meanG[index];
  const float centerB = meanB[index];

  constexpr int kMaxEntries = 49;
  float values[kMaxEntries];
  int xs[kMaxEntries];
  int ys[kMaxEntries];
  int count = 0;
  for (int oy = -radius; oy <= radius; ++oy) {
    const int yy = y + oy;
    if (yy < 0 || yy >= gridHeight) continue;
    for (int ox = -radius; ox <= radius; ++ox) {
      const int xx = x + ox;
      if (xx < 0 || xx >= gridWidth || count >= kMaxEntries) continue;
      const unsigned int nidx = static_cast<unsigned int>(yy * gridWidth + xx);
      if (occupancy[nidx] <= 0.5f) continue;
      const float neighborCarrier = carrier[nidx];
      const float dr = meanR[nidx] - centerR;
      const float dg = meanG[nidx] - centerG;
      const float db = meanB[nidx] - centerB;
      const float colorDistance = sqrtf(dr * dr + dg * dg + db * db);
      if (fabsf(neighborCarrier - centerCarrier) > 0.26f && colorDistance > 0.20f) continue;
      int insert = count++;
      while (insert > 0 && values[insert - 1] > neighborCarrier) {
        values[insert] = values[insert - 1];
        xs[insert] = xs[insert - 1];
        ys[insert] = ys[insert - 1];
        --insert;
      }
      values[insert] = neighborCarrier;
      xs[insert] = xx;
      ys[insert] = yy;
    }
  }
  if (count <= 0) {
    body[index] = centerCarrier;
    return;
  }
  const int trim = count >= 6 ? max(1, count / 6) : 0;
  const int begin = min(trim, count - 1);
  const int end = max(begin + 1, count - trim);
  float bodySum = 0.0f;
  float bodyWeight = 0.0f;
  for (int i = begin; i < end; ++i) {
    const float dx = static_cast<float>(xs[i] - x);
    const float dy = static_cast<float>(ys[i] - y);
    const float spatialWeight = 1.0f / (1.0f + dx * dx + dy * dy);
    bodySum += values[i] * spatialWeight;
    bodyWeight += spatialWeight;
  }
  body[index] = bodyWeight > 1e-6f ? bodySum / bodyWeight : centerCarrier;
}

__global__ void glossFieldLocalPercentileKernel(int gridWidth,
                                                int gridHeight,
                                                int radiusCells,
                                                float percentile,
                                                const float* values,
                                                const float* occupancy,
                                                float* outValues) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !values || !outValues) return;
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  if (occupancy && occupancy[index] <= 0.5f) {
    outValues[index] = 0.0f;
    return;
  }
  const int radius = max(1, radiusCells);
  constexpr int kBinCount = 96;
  int bins[kBinCount];
  for (int i = 0; i < kBinCount; ++i) bins[i] = 0;
  int count = 0;
  float minValue = 1.0e20f;
  float maxValue = -1.0e20f;
  for (int yy = max(0, y - radius); yy <= min(gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(gridWidth - 1, x + radius); ++xx) {
      const unsigned int nidx = static_cast<unsigned int>(yy * gridWidth + xx);
      if (occupancy && occupancy[nidx] <= 0.5f) continue;
      const float v = values[nidx];
      minValue = fminf(minValue, v);
      maxValue = fmaxf(maxValue, v);
      ++count;
    }
  }
  if (count <= 0 || maxValue - minValue <= 1e-7f) {
    outValues[index] = values[index];
    return;
  }
  const float invRange = 1.0f / (maxValue - minValue);
  for (int yy = max(0, y - radius); yy <= min(gridHeight - 1, y + radius); ++yy) {
    for (int xx = max(0, x - radius); xx <= min(gridWidth - 1, x + radius); ++xx) {
      const unsigned int nidx = static_cast<unsigned int>(yy * gridWidth + xx);
      if (occupancy && occupancy[nidx] <= 0.5f) continue;
      const int bin = min(kBinCount - 1, max(0, static_cast<int>((values[nidx] - minValue) * invRange *
                                                                 static_cast<float>(kBinCount - 1) + 0.5f)));
      bins[bin] += 1;
    }
  }
  const int target = min(count - 1, max(0, static_cast<int>(floorf(clamp01(percentile / 100.0f) *
                                                                  static_cast<float>(count - 1) + 0.5f))));
  int accumulated = 0;
  int chosen = 0;
  for (int i = 0; i < kBinCount; ++i) {
    accumulated += bins[i];
    if (accumulated > target) {
      chosen = i;
      break;
    }
  }
  outValues[index] = minValue + (static_cast<float>(chosen) / static_cast<float>(kBinCount - 1)) *
                                    (maxValue - minValue);
}

__global__ void glossFieldCandidate1PrepareExactKernel(int cellCount,
                                                       const float* occupancy,
                                                       const float* carrier,
                                                       const float* viewerBody,
                                                       const float* bodyCore,
                                                       const float* bodyContext,
                                                       float* adaptiveBody,
                                                       float* basePositive,
                                                       float* baseNegative,
                                                       float* consensusPositive,
                                                       unsigned int* maxBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !occupancy || !carrier || !viewerBody || !bodyCore || !bodyContext ||
      !adaptiveBody || !basePositive || !baseNegative || !consensusPositive || !maxBits) {
    return;
  }
  if (occupancy[index] <= 0.5f) {
    adaptiveBody[index] = 0.0f;
    basePositive[index] = 0.0f;
    baseNegative[index] = 0.0f;
    consensusPositive[index] = 0.0f;
    return;
  }
  const float body = 0.72f * viewerBody[index] + 0.20f * bodyCore[index] + 0.08f * bodyContext[index];
  const float positive = fmaxf(0.0f, carrier[index] - body);
  const float negative = fmaxf(0.0f, body - carrier[index]);
  const float compactBody = fminf(viewerBody[index], bodyCore[index]);
  const float consensus = fmaxf(0.0f, carrier[index] - compactBody);
  adaptiveBody[index] = body;
  basePositive[index] = positive;
  baseNegative[index] = negative;
  consensusPositive[index] = consensus;
  atomicMax(&maxBits[0], __float_as_uint(fmaxf(viewerBody[index], 0.0f)));
  atomicMax(&maxBits[1], __float_as_uint(fmaxf(bodyCore[index], 0.0f)));
  atomicMax(&maxBits[2], __float_as_uint(positive));
  atomicMax(&maxBits[3], __float_as_uint(consensus));
}

__global__ void glossFieldCandidate1FinalizeExactKernel(int gridWidth,
                                                        int gridHeight,
                                                        const float* occupancy,
                                                        const float* occupancySupport,
                                                        const float* adaptiveBody,
                                                        const float* viewerBody,
                                                        const float* bodyCore,
                                                        const float* basePositive,
                                                        const float* baseNegative,
                                                        const float* consensusPositive,
                                                        const float* positiveSupport,
                                                        const float* consensusSupport,
                                                        const unsigned int* prepareMaxBits,
                                                        float* positiveRaw,
                                                        float* negativeRaw,
                                                        float* confidence,
                                                        float* agreement) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !occupancy || !occupancySupport || !adaptiveBody || !viewerBody || !bodyCore || !basePositive ||
      !baseNegative || !consensusPositive || !positiveSupport || !consensusSupport || !prepareMaxBits ||
      !positiveRaw || !negativeRaw || !confidence || !agreement) {
    return;
  }
  if (occupancy[index] <= 0.5f) {
    positiveRaw[index] = 0.0f;
    negativeRaw[index] = 0.0f;
    confidence[index] = 0.0f;
    agreement[index] = 0.0f;
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const float viewerMax = fmaxf(__uint_as_float(prepareMaxBits[0]), 1.0e-5f);
  const float coreMax = fmaxf(__uint_as_float(prepareMaxBits[1]), 1.0e-5f);
  const float basePositiveMax = fmaxf(__uint_as_float(prepareMaxBits[2]), 1.0e-5f);
  const float consensusPositiveMax = fmaxf(__uint_as_float(prepareMaxBits[3]), 1.0e-5f);
  const float bodyAgreement = clamp01(1.0f - fabsf(clamp01(viewerBody[index] / viewerMax) -
                                                   clamp01(bodyCore[index] / coreMax)));
  const float positiveAgreement =
      clamp01(1.0f - fabsf(clamp01(basePositive[index] / basePositiveMax) -
                           clamp01(consensusPositive[index] / consensusPositiveMax)));
  const float positiveBodyCongruence =
      glossGradientCongruenceDevice(adaptiveBody, basePositive, gridWidth, gridHeight, x, y);
  const float fillGuard = clamp01(0.30f * positiveSupport[index] +
                                  0.20f * consensusSupport[index] +
                                  0.18f * positiveAgreement +
                                  0.16f * bodyAgreement +
                                  0.16f * positiveBodyCongruence);
  const float posRaw = (0.60f * consensusPositive[index] + 0.40f * basePositive[index]) *
                       (0.14f + 0.86f * fillGuard);
  const float negRaw = baseNegative[index] * (0.38f + 0.62f * bodyAgreement);
  const float attachment = clamp01(0.42f * positiveBodyCongruence +
                                   0.26f * positiveSupport[index] +
                                   0.20f * positiveAgreement +
                                   0.12f * bodyAgreement);
  const float support = sqrtf(clamp01(occupancySupport[index]));
  positiveRaw[index] = posRaw;
  negativeRaw[index] = negRaw;
  confidence[index] = clamp01((0.18f + 0.82f * (0.38f * fillGuard +
                                                0.22f * bodyAgreement +
                                                0.20f * positiveAgreement +
                                                0.20f * attachment)) *
                              (0.28f + 0.72f * support));
  agreement[index] = clamp01(0.50f * bodyAgreement + 0.50f * positiveAgreement);
}

__global__ void glossFieldCandidate2RawKernel(int gridWidth,
                                              int gridHeight,
                                              int neighborhoodChoice,
                                              const float* occupancy,
                                              const float* occupancySupport,
                                              const float* meanR,
                                              const float* meanG,
                                              const float* meanB,
                                              const float* carrier,
                                              const float* viewerBody,
                                              const float* bodyCore,
                                              const float* bodyContext,
                                              const float* retinexBody,
                                              const float* dogLow,
                                              const float* dogHigh,
                                              float* adaptiveBody,
                                              float* positiveRaw,
                                              float* negativeRaw,
                                              float* confidence,
                                              float* agreement) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !occupancy || !occupancySupport || !meanR || !meanG || !meanB || !carrier ||
      !viewerBody || !bodyCore || !bodyContext || !retinexBody || !dogLow || !dogHigh ||
      !adaptiveBody || !positiveRaw || !negativeRaw || !confidence || !agreement) {
    return;
  }
  if (occupancy[index] <= 0.5f) {
    adaptiveBody[index] = 0.0f;
    positiveRaw[index] = 0.0f;
    negativeRaw[index] = 0.0f;
    confidence[index] = 0.0f;
    agreement[index] = 0.0f;
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const int analysisRadius = glossAnalysisRadiusCellsDevice(neighborhoodChoice);
  const float hybridBody = 0.65f * bodyCore[index] + 0.35f * bodyContext[index];
  const float viewerPositive = fmaxf(0.0f, carrier[index] - viewerBody[index]);
  const float viewerNegative = fmaxf(0.0f, viewerBody[index] - carrier[index]);
  const float hybridPositive = fmaxf(0.0f, carrier[index] - bodyCore[index]);
  const float hybridNegative = fmaxf(0.0f, bodyContext[index] - carrier[index]);
  const float chromaSpread = fmaxf(meanR[index], fmaxf(meanG[index], meanB[index])) -
                             fminf(meanR[index], fminf(meanG[index], meanB[index]));
  const float shapeSupport = clamp01(glossGradientMagnitudeDevice(bodyContext, gridWidth, gridHeight, x, y) * 8.0f);
  const float chromaSupport = clamp01(chromaSpread * 2.5f);
  const float ambiguity = clamp01(1.0f - (0.72f * shapeSupport + 0.28f * chromaSupport));
  const float body = ambiguity * viewerBody[index] + (1.0f - ambiguity) * hybridBody;
  const float bodyAgreement = clamp01(1.0f - fabsf(viewerBody[index] - hybridBody));
  const float positiveAgreement = clamp01(1.0f - fabsf(viewerPositive - hybridPositive) * 4.0f);
  float localPositiveSupport = 0.0f;
  float supportWeight = 0.0f;
  for (int oy = -analysisRadius; oy <= analysisRadius; ++oy) {
    for (int ox = -analysisRadius; ox <= analysisRadius; ++ox) {
      const int xx = min(max(x + ox, 0), gridWidth - 1);
      const int yy = min(max(y + oy, 0), gridHeight - 1);
      const unsigned int nidx = static_cast<unsigned int>(yy * gridWidth + xx);
      const float distSq = static_cast<float>(ox * ox + oy * oy);
      const float w = 1.0f / (1.0f + distSq);
      localPositiveSupport += fmaxf(0.0f, carrier[nidx] - bodyCore[nidx]) * w;
      supportWeight += w;
    }
  }
  localPositiveSupport = clamp01((supportWeight > 1e-6f ? localPositiveSupport / supportWeight : 0.0f) * 4.0f);
  const float permission = clamp01(0.32f * positiveAgreement +
                                   0.24f * bodyAgreement +
                                   0.24f * shapeSupport +
                                   0.20f * localPositiveSupport);
  const float consensusPositive = viewerPositive * (0.25f + 0.75f * clamp01(hybridPositive * 4.0f));
  const float panelMix = clamp01(sqrtf(fmaxf(0.0f, ambiguity)) * (0.55f + 0.45f * positiveAgreement));
  const float retinexResidual = carrier[index] - retinexBody[index];
  const float dogResidual = dogLow[index] - dogHigh[index];
  const float dogPositive = fmaxf(0.0f, dogResidual);
  const float dogNegative = fmaxf(0.0f, -dogResidual);
  const float dogPositiveAgreement = clamp01(1.0f - fabsf(dogPositive - hybridPositive) * 4.0f);
  const float dogNegativeAgreement = clamp01(1.0f - fabsf(dogNegative - hybridNegative) * 4.0f);
  const float positiveRetinexGate = clamp01(0.18f +
                                            0.34f * permission +
                                            0.18f * positiveAgreement +
                                            0.16f * shapeSupport +
                                            0.14f * localPositiveSupport);
  const float negativeRetinexGate = clamp01(0.30f +
                                            0.40f * (1.0f - ambiguity) +
                                            0.18f * bodyAgreement +
                                            0.12f * permission);
  const float dogPositiveGate = clamp01(0.16f +
                                        0.30f * permission +
                                        0.18f * positiveAgreement +
                                        0.16f * shapeSupport +
                                        0.12f * dogPositiveAgreement +
                                        0.08f * localPositiveSupport);
  const float dogNegativeGate = clamp01(0.30f +
                                        0.36f * (1.0f - ambiguity) +
                                        0.20f * bodyAgreement +
                                        0.14f * dogNegativeAgreement);
  const float pos = (1.0f - panelMix) * hybridPositive +
                    panelMix * consensusPositive +
                    0.20f * positiveRetinexGate * fmaxf(0.0f, retinexResidual) +
                    0.18f * dogPositiveGate * dogPositive;
  const float neg = (1.0f - ambiguity) * hybridNegative +
                    ambiguity * (0.55f * hybridNegative + 0.45f * viewerNegative) +
                    0.16f * negativeRetinexGate * fmaxf(0.0f, -retinexResidual) +
                    0.12f * dogNegativeGate * dogNegative;
  adaptiveBody[index] = body;
  positiveRaw[index] = pos;
  negativeRaw[index] = neg;
  const float gradientAttachment = shapeSupport;
  const float attachment = clamp01(0.31f * gradientAttachment +
                                   0.21f * localPositiveSupport +
                                   0.20f * permission +
                                   0.20f * positiveAgreement +
                                   0.08f * bodyAgreement);
  const float support = sqrtf(clamp01(occupancySupport[index]));
  confidence[index] = clamp01((0.10f + 0.90f * (0.28f * bodyAgreement +
                                                0.22f * positiveAgreement +
                                                0.20f * permission +
                                                0.15f * (1.0f - ambiguity) +
                                                0.15f * attachment)) *
                              (0.30f + 0.70f * support));
  agreement[index] = clamp01(0.40f * bodyAgreement + 0.35f * positiveAgreement + 0.25f * permission);
}

__global__ void glossFieldAssembleUnifiedKernel(int gridWidth,
                                                int gridHeight,
                                                const float* bodyRaw,
                                                const float* positiveRaw,
                                                const float* negativeRaw,
                                                const float* confidenceRaw,
                                                const float* agreementRaw,
                                                float* body,
                                                float* signal,
                                                float* positive,
                                                float* negative,
                                                float* boundary,
                                                float* congruence,
                                                float* confidence,
                                                unsigned int* maxBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !bodyRaw || !positiveRaw || !negativeRaw || !confidenceRaw || !agreementRaw ||
      !body || !signal || !positive || !negative || !boundary || !congruence || !confidence || !maxBits) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const float gxBody = 0.5f * (sampleGridClampedDevice(bodyRaw, gridWidth, gridHeight, x + 1, y) -
                               sampleGridClampedDevice(bodyRaw, gridWidth, gridHeight, x - 1, y));
  const float gyBody = 0.5f * (sampleGridClampedDevice(bodyRaw, gridWidth, gridHeight, x, y + 1) -
                               sampleGridClampedDevice(bodyRaw, gridWidth, gridHeight, x, y - 1));
  const float signalXp = sampleGridClampedDevice(positiveRaw, gridWidth, gridHeight, x + 1, y) -
                         sampleGridClampedDevice(negativeRaw, gridWidth, gridHeight, x + 1, y);
  const float signalXm = sampleGridClampedDevice(positiveRaw, gridWidth, gridHeight, x - 1, y) -
                         sampleGridClampedDevice(negativeRaw, gridWidth, gridHeight, x - 1, y);
  const float signalYp = sampleGridClampedDevice(positiveRaw, gridWidth, gridHeight, x, y + 1) -
                         sampleGridClampedDevice(negativeRaw, gridWidth, gridHeight, x, y + 1);
  const float signalYm = sampleGridClampedDevice(positiveRaw, gridWidth, gridHeight, x, y - 1) -
                         sampleGridClampedDevice(negativeRaw, gridWidth, gridHeight, x, y - 1);
  const float gxSignal = 0.5f * (signalXp - signalXm);
  const float gySignal = 0.5f * (signalYp - signalYm);
  const float magBody = sqrtf(gxBody * gxBody + gyBody * gyBody);
  const float magSignal = sqrtf(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0f;
  if (magBody > 1e-6f && magSignal > 1e-6f) {
    localCongruence = clamp01(fabsf((gxBody * gxSignal + gyBody * gySignal) / (magBody * magSignal)));
  } else if (magSignal > 1e-6f) {
    localCongruence = 0.35f;
  }
  const float localBoundary = magSignal * 4.0f;
  const float weight = (0.35f + 0.65f * clamp01(localCongruence)) *
                       (0.45f + 0.55f * clamp01(confidenceRaw[index]));
  body[index] = fmaxf(0.0f, bodyRaw[index]);
  positive[index] = fmaxf(0.0f, positiveRaw[index]) * weight;
  negative[index] = fmaxf(0.0f, negativeRaw[index]) * weight;
  signal[index] = positive[index] - negative[index];
  boundary[index] = fmaxf(0.0f, localBoundary);
  congruence[index] = localCongruence;
  confidence[index] = confidenceRaw[index];
  atomicMax(&maxBits[0], __float_as_uint(fmaxf(body[index], 0.0f)));
  atomicMax(&maxBits[1], __float_as_uint(fmaxf(positive[index], 0.0f)));
  atomicMax(&maxBits[2], __float_as_uint(fmaxf(negative[index], 0.0f)));
  atomicMax(&maxBits[3], __float_as_uint(fmaxf(boundary[index], 0.0f)));
}

inline __device__ float glossProjectionGamma(float value) {
  return powf(clamp01(value), 1.0f / 2.2f);
}

inline __device__ void glossProjectionMix(float ax,
                                          float ay,
                                          float az,
                                          float bx,
                                          float by,
                                          float bz,
                                          float t,
                                          float* ox,
                                          float* oy,
                                          float* oz) {
  const float u = clamp01(t);
  *ox = ax * (1.0f - u) + bx * u;
  *oy = ay * (1.0f - u) + by * u;
  *oz = az * (1.0f - u) + bz * u;
}

inline __device__ void glossProjectionApplySaturation(float saturation,
                                                      float* r,
                                                      float* g,
                                                      float* b) {
  const float luma = 0.2126f * *r + 0.7152f * *g + 0.0722f * *b;
  const float s = fminf(fmaxf(saturation, 0.0f), 3.0f);
  *r = clamp01(luma + (*r - luma) * s);
  *g = clamp01(luma + (*g - luma) * s);
  *b = clamp01(luma + (*b - luma) * s);
}

__global__ void glossProjectionFromFieldKernel(float* verts,
                                               float* colors,
                                               const float* fieldWorkspace,
                                               GlossProjectionRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int gridWidth = max(request.gridWidth, 1);
  const int gridHeight = max(request.gridHeight, 1);
  const unsigned int cellCount = static_cast<unsigned int>(gridWidth * gridHeight);
  if (index >= cellCount || !verts || !colors || !fieldWorkspace) return;

  const size_t cell = static_cast<size_t>(index);
  const size_t count = static_cast<size_t>(cellCount);
  const float* occupancy = fieldWorkspace + count * 0u;
  const float* meanR = fieldWorkspace + count * 8u;
  const float* meanG = fieldWorkspace + count * 9u;
  const float* meanB = fieldWorkspace + count * 10u;
  const float* carrierY = fieldWorkspace + count * 11u;
  const float* carrierMax = fieldWorkspace + count * 12u;
  const float* carrierMin = fieldWorkspace + count * 13u;
  const float* neutrality = fieldWorkspace + count * 14u;
  const size_t solutionBase =
      request.algorithm != 0 ? kGlossFieldCandidate2Base : kGlossFieldCandidate1Base;
  const float* body = fieldWorkspace + count * (solutionBase + 0u);
  const float* positive = fieldWorkspace + count * (solutionBase + 2u);
  const float* negative = fieldWorkspace + count * (solutionBase + 3u);
  const float* boundaryValues = fieldWorkspace + count * (solutionBase + 4u);
  const float* congruence = fieldWorkspace + count * (solutionBase + 5u);
  const float* confidenceValues = fieldWorkspace + count * (solutionBase + 6u);
  const float* signal = fieldWorkspace + count * (solutionBase + 7u);

  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const float aspect = fminf(fmaxf(request.sourceAspect, 0.25f), 4.0f);
  constexpr float kMajorHalf = 1.22f;
  const float halfWidth = aspect >= 1.0f ? kMajorHalf : kMajorHalf * aspect;
  const float halfDepth = aspect >= 1.0f ? kMajorHalf / aspect : kMajorHalf;
  const float xNorm = (static_cast<float>(x) + 0.5f) / static_cast<float>(gridWidth);
  const float yNormInv = (static_cast<float>(y) + 0.5f) / static_cast<float>(gridHeight);
  const float xPos = -halfWidth + 2.0f * halfWidth * xNorm;
  const float imageY = halfDepth - 2.0f * halfDepth * yNormInv;

  float signedValue = signal[cell];
  float positiveValue = positive[cell];
  float negativeValue = negative[cell];
  if (request.debugMode == 1) {
    signedValue = positiveValue = clamp01(carrierMax[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 2) {
    signedValue = positiveValue = clamp01(carrierY[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 3) {
    signedValue = positiveValue = clamp01(carrierMin[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 4) {
    signedValue = positiveValue = clamp01(neutrality[cell]);
    negativeValue = 0.0f;
  }
  const float zPos = request.debugMode == 0 ? signedValue : fmaxf(0.0f, signedValue);

  const float mr = meanR[cell];
  const float mg = meanG[cell];
  const float mb = meanB[cell];
  const float sourcePresence = clamp01(fmaxf(mr, fmaxf(mg, mb)));
  const float confidence = clamp01(confidenceValues[cell]);
  const bool active = occupancy[cell] > 0.5f && (confidence > 0.01f || sourcePresence > 0.01f);
  const float base = clamp01(body[cell]);
  const float structureStrength = fmaxf(clamp01(congruence[cell]), clamp01(boundaryValues[cell]));
  const float signalScale = fmaxf(request.glossLiftScale, 1.0f);
  positiveValue = clamp01(positiveValue * signalScale);
  negativeValue = clamp01(negativeValue * signalScale);
  const float positiveT = clamp01((positiveValue - 0.035f) / (1.0f - 0.035f));
  const float negativeT = clamp01((negativeValue - 0.035f) / (1.0f - 0.035f));
  const float positiveDisplay = positiveT * positiveT * (3.0f - 2.0f * positiveT);
  const float negativeDisplay = negativeT * negativeT * (3.0f - 2.0f * negativeT);
  const float signalPresence = fmaxf(positiveDisplay, negativeDisplay);

  float sr = glossProjectionGamma(mr);
  float sg = glossProjectionGamma(mg);
  float sb = glossProjectionGamma(mb);
  glossProjectionApplySaturation(request.colorSaturation, &sr, &sg, &sb);

  float cr = 0.03f;
  float cg = 0.03f;
  float cb = 0.04f;
  if (request.colorMode == 0) {
    const float baseMix = clamp01(request.glossBodyOpacity *
                                  (0.22f + 0.78f * confidence) *
                                  (0.86f - 0.22f * signalPresence));
    const float basePow = powf(base, 0.78f);
    float nr = 0.16f + 0.60f * basePow;
    float ng = 0.16f + 0.58f * basePow;
    float nb = 0.17f + 0.54f * basePow;
    float mixR = 0.0f;
    float mixG = 0.0f;
    float mixB = 0.0f;
    glossProjectionMix(nr, ng, nb, sr, sg, sb, 0.68f, &mixR, &mixG, &mixB);
    glossProjectionMix(0.03f, 0.03f, 0.04f, mixR, mixG, mixB, baseMix, &cr, &cg, &cb);
    if (positiveDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 1.0f, 0.95f, 0.86f, 0.54f, &wr, &wg, &wb);
      glossProjectionMix(cr,
                         cg,
                         cb,
                         wr,
                         wg,
                         wb,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr,
                         &cg,
                         &cb);
    }
    if (negativeDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 0.08f, 0.14f, 0.24f, 0.74f, &wr, &wg, &wb);
      glossProjectionMix(cr,
                         cg,
                         cb,
                         wr,
                         wg,
                         wb,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr,
                         &cg,
                         &cb);
    }
  } else {
    const float basePow = powf(base, 0.78f);
    const float nr = 0.16f + 0.64f * basePow;
    const float ng = 0.16f + 0.64f * basePow;
    const float nb = 0.17f + 0.60f * basePow;
    glossProjectionMix(0.03f,
                       0.03f,
                       0.04f,
                       nr,
                       ng,
                       nb,
                       clamp01(request.glossBodyOpacity *
                               (0.22f + 0.78f * confidence) *
                               (0.86f - 0.22f * signalPresence)),
                       &cr,
                       &cg,
                       &cb);
    if (positiveDisplay > 0.0f) {
      glossProjectionMix(cr,
                         cg,
                         cb,
                         1.0f,
                         0.89f,
                         0.36f,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr,
                         &cg,
                         &cb);
    }
    if (negativeDisplay > 0.0f) {
      glossProjectionMix(cr,
                         cg,
                         cb,
                         0.22f,
                         0.76f,
                         1.0f,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr,
                         &cg,
                         &cb);
    }
  }
  const float boundary = clamp01(boundaryValues[cell]);
  if (boundary > 0.0f) {
    glossProjectionMix(cr,
                       cg,
                       cb,
                       0.98f,
                       0.98f,
                       0.94f,
                       fminf(0.10f + 0.26f * boundary, 0.34f),
                       &cr,
                       &cg,
                       &cb);
  }
  float alpha = clamp01(request.glossBodyOpacity * (0.12f + 0.62f * confidence) *
                            (0.82f - 0.18f * signalPresence) +
                        request.glossHighlightOpacity * signalPresence *
                            (0.16f + 0.84f * structureStrength));
  alpha = fminf(fmaxf(alpha, 0.018f), 1.0f);
  if (request.diagnosticMode == 1) {
    const float gray = 0.16f + 0.78f * confidence;
    glossProjectionMix(cr, cg, cb, gray, gray, gray, 0.36f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 1.0f, 1.0f, 0.96f, 0.10f * boundary, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.55f + 0.45f * confidence) + 0.10f * confidence);
  } else if (request.diagnosticMode == 2) {
    const float ambiguity = clamp01(1.0f - confidence);
    const float gray = 0.12f + 0.74f * ambiguity;
    glossProjectionMix(cr, cg, cb, gray * 0.94f, gray * 0.97f, gray, 0.34f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 0.80f, 0.90f, 1.0f, 0.10f * boundary * ambiguity, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.48f + 0.52f * ambiguity) + 0.08f * ambiguity);
  }

  const unsigned int vbase = index * 3u;
  const unsigned int cbase = index * 4u;
  verts[vbase + 0u] = active ? xPos : 0.0f;
  verts[vbase + 1u] = active ? imageY : 0.0f;
  verts[vbase + 2u] = active ? zPos : 0.0f;
  colors[cbase + 0u] = clamp01(cr);
  colors[cbase + 1u] = clamp01(cg);
  colors[cbase + 2u] = clamp01(cb);
  colors[cbase + 3u] = active ? alpha : 0.0f;
}

__global__ void glossProjectionFromRasterSourceKernel(float* verts,
                                                      float* colors,
                                                      const unsigned char* source,
                                                      RasterSourceKernelUniforms raster,
                                                      const float* fieldWorkspace,
                                                      GlossProjectionRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int pointCount = static_cast<unsigned int>(max(raster.basePointCount, 0));
  if (index >= pointCount || !verts || !colors || !source || !fieldWorkspace) return;

  const int gridWidth = max(request.gridWidth, 1);
  const int gridHeight = max(request.gridHeight, 1);
  const unsigned int cellCount = static_cast<unsigned int>(gridWidth * gridHeight);
  const int sampleCountX = max(raster.sampleCountX, 1);
  const int stride = max(raster.sampleStride, 1);
  int sourceX = static_cast<int>(index % static_cast<unsigned int>(sampleCountX)) * stride;
  int sourceY = static_cast<int>(index / static_cast<unsigned int>(sampleCountX)) * stride;
  sourceX = min(max(sourceX, 0), max(raster.sourceWidth - 1, 0));
  sourceY = min(max(sourceY, 0), max(raster.sourceHeight - 1, 0));
  const float xNorm =
      (static_cast<float>(sourceX) + 0.5f) / static_cast<float>(max(raster.sourceWidth, 1));
  const float yNorm =
      (static_cast<float>(sourceY) + 0.5f) / static_cast<float>(max(raster.sourceHeight, 1));

  float sampleR = 0.0f;
  float sampleG = 0.0f;
  float sampleB = 0.0f;
  rasterReadTransformedSample(source, raster, sourceX, sourceY, &sampleR, &sampleG, &sampleB);
  const bool visible = rasterSampleVisible(raster, sourceX, sourceY, xNorm, yNorm, sampleR, sampleG, sampleB);
  const int cellX = min(max(static_cast<int>(xNorm * static_cast<float>(gridWidth)), 0), gridWidth - 1);
  const float visualTopNorm = rasterGlossVisualTopNorm(yNorm);
  const int cellY =
      min(max(static_cast<int>(visualTopNorm * static_cast<float>(gridHeight)), 0), gridHeight - 1);
  const unsigned int cell = static_cast<unsigned int>(cellY * gridWidth + cellX);
  if (cell >= cellCount) return;

  const size_t count = static_cast<size_t>(cellCount);
  const float* occupancy = fieldWorkspace + count * 0u;
  const float* meanR = fieldWorkspace + count * 8u;
  const float* meanG = fieldWorkspace + count * 9u;
  const float* meanB = fieldWorkspace + count * 10u;
  const float* carrierY = fieldWorkspace + count * 11u;
  const float* carrierMax = fieldWorkspace + count * 12u;
  const float* carrierMin = fieldWorkspace + count * 13u;
  const float* neutrality = fieldWorkspace + count * 14u;
  const size_t solutionBase =
      request.algorithm != 0 ? kGlossFieldCandidate2Base : kGlossFieldCandidate1Base;
  const float* body = fieldWorkspace + count * (solutionBase + 0u);
  const float* positive = fieldWorkspace + count * (solutionBase + 2u);
  const float* negative = fieldWorkspace + count * (solutionBase + 3u);
  const float* boundaryValues = fieldWorkspace + count * (solutionBase + 4u);
  const float* congruence = fieldWorkspace + count * (solutionBase + 5u);
  const float* confidenceValues = fieldWorkspace + count * (solutionBase + 6u);
  const float* signal = fieldWorkspace + count * (solutionBase + 7u);

  const float aspect = fminf(fmaxf(request.sourceAspect, 0.25f), 4.0f);
  constexpr float kMajorHalf = 1.22f;
  const float halfWidth = aspect >= 1.0f ? kMajorHalf : kMajorHalf * aspect;
  const float halfDepth = aspect >= 1.0f ? kMajorHalf / aspect : kMajorHalf;
  const float xPos = -halfWidth + 2.0f * halfWidth * xNorm;
  const float imageY = halfDepth - 2.0f * halfDepth * visualTopNorm;

  float signedValue = signal[cell];
  float positiveValue = positive[cell];
  float negativeValue = negative[cell];
  if (request.debugMode == 1) {
    signedValue = positiveValue = clamp01(carrierMax[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 2) {
    signedValue = positiveValue = clamp01(carrierY[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 3) {
    signedValue = positiveValue = clamp01(carrierMin[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 4) {
    signedValue = positiveValue = clamp01(neutrality[cell]);
    negativeValue = 0.0f;
  }
  const float zPos = request.debugMode == 0 ? signedValue : fmaxf(0.0f, signedValue);

  const float mr = visible ? sampleR : meanR[cell];
  const float mg = visible ? sampleG : meanG[cell];
  const float mb = visible ? sampleB : meanB[cell];
  const float sourcePresence = clamp01(fmaxf(mr, fmaxf(mg, mb)));
  const float confidence = clamp01(confidenceValues[cell]);
  const bool active = visible && occupancy[cell] > 0.5f && (confidence > 0.01f || sourcePresence > 0.01f);
  const float base = clamp01(body[cell]);
  const float structureStrength = fmaxf(clamp01(congruence[cell]), clamp01(boundaryValues[cell]));
  const float signalScale = fmaxf(request.glossLiftScale, 1.0f);
  positiveValue = clamp01(positiveValue * signalScale);
  negativeValue = clamp01(negativeValue * signalScale);
  const float positiveT = clamp01((positiveValue - 0.035f) / (1.0f - 0.035f));
  const float negativeT = clamp01((negativeValue - 0.035f) / (1.0f - 0.035f));
  const float positiveDisplay = positiveT * positiveT * (3.0f - 2.0f * positiveT);
  const float negativeDisplay = negativeT * negativeT * (3.0f - 2.0f * negativeT);
  const float signalPresence = fmaxf(positiveDisplay, negativeDisplay);

  float sr = glossProjectionGamma(mr);
  float sg = glossProjectionGamma(mg);
  float sb = glossProjectionGamma(mb);
  glossProjectionApplySaturation(request.colorSaturation, &sr, &sg, &sb);

  float cr = 0.03f;
  float cg = 0.03f;
  float cb = 0.04f;
  if (request.colorMode == 0) {
    const float baseMix = clamp01(request.glossBodyOpacity *
                                  (0.22f + 0.78f * confidence) *
                                  (0.86f - 0.22f * signalPresence));
    const float basePow = powf(base, 0.78f);
    float nr = 0.16f + 0.60f * basePow;
    float ng = 0.16f + 0.58f * basePow;
    float nb = 0.17f + 0.54f * basePow;
    float mixR = 0.0f;
    float mixG = 0.0f;
    float mixB = 0.0f;
    glossProjectionMix(nr, ng, nb, sr, sg, sb, 0.68f, &mixR, &mixG, &mixB);
    glossProjectionMix(0.03f, 0.03f, 0.04f, mixR, mixG, mixB, baseMix, &cr, &cg, &cb);
    if (positiveDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 1.0f, 0.95f, 0.86f, 0.54f, &wr, &wg, &wb);
      glossProjectionMix(cr, cg, cb, wr, wg, wb,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
    if (negativeDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 0.08f, 0.14f, 0.24f, 0.74f, &wr, &wg, &wb);
      glossProjectionMix(cr, cg, cb, wr, wg, wb,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
  } else {
    const float basePow = powf(base, 0.78f);
    const float nr = 0.16f + 0.64f * basePow;
    const float ng = 0.16f + 0.64f * basePow;
    const float nb = 0.17f + 0.60f * basePow;
    glossProjectionMix(0.03f, 0.03f, 0.04f, nr, ng, nb,
                       clamp01(request.glossBodyOpacity *
                               (0.22f + 0.78f * confidence) *
                               (0.86f - 0.22f * signalPresence)),
                       &cr, &cg, &cb);
    if (positiveDisplay > 0.0f) {
      glossProjectionMix(cr, cg, cb, 1.0f, 0.89f, 0.36f,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
    if (negativeDisplay > 0.0f) {
      glossProjectionMix(cr, cg, cb, 0.22f, 0.76f, 1.0f,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
  }
  const float boundary = clamp01(boundaryValues[cell]);
  if (boundary > 0.0f) {
    glossProjectionMix(cr, cg, cb, 0.98f, 0.98f, 0.94f,
                       fminf(0.10f + 0.26f * boundary, 0.34f),
                       &cr, &cg, &cb);
  }
  float alpha = clamp01(request.glossBodyOpacity * (0.12f + 0.62f * confidence) *
                            (0.82f - 0.18f * signalPresence) +
                        request.glossHighlightOpacity * signalPresence *
                            (0.16f + 0.84f * structureStrength));
  alpha = fminf(fmaxf(alpha, 0.018f), 1.0f);
  if (request.diagnosticMode == 1) {
    const float gray = 0.16f + 0.78f * confidence;
    glossProjectionMix(cr, cg, cb, gray, gray, gray, 0.36f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 1.0f, 1.0f, 0.96f, 0.10f * boundary, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.55f + 0.45f * confidence) + 0.10f * confidence);
  } else if (request.diagnosticMode == 2) {
    const float ambiguity = clamp01(1.0f - confidence);
    const float gray = 0.12f + 0.74f * ambiguity;
    glossProjectionMix(cr, cg, cb, gray * 0.94f, gray * 0.97f, gray, 0.34f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 0.80f, 0.90f, 1.0f, 0.10f * boundary * ambiguity, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.48f + 0.52f * ambiguity) + 0.08f * ambiguity);
  }

  const unsigned int vbase = index * 3u;
  const unsigned int cbase = index * 4u;
  verts[vbase + 0u] = active ? xPos : 0.0f;
  verts[vbase + 1u] = active ? imageY : 0.0f;
  verts[vbase + 2u] = active ? zPos : 0.0f;
  colors[cbase + 0u] = clamp01(cr);
  colors[cbase + 1u] = clamp01(cg);
  colors[cbase + 2u] = clamp01(cb);
  colors[cbase + 3u] = active ? alpha : 0.0f;
}

inline __device__ void glossField2DEmitQuad(float* verts,
                                            float* colors,
                                            unsigned int vertexBase,
                                            float x0,
                                            float y0,
                                            float x1,
                                            float y1,
                                            float r,
                                            float g,
                                            float b,
                                            float a) {
  const float vx[6] = {x0, x1, x1, x0, x1, x0};
  const float vy[6] = {y0, y0, y1, y0, y1, y1};
  for (int vertex = 0; vertex < 6; ++vertex) {
    const unsigned int vbase = (vertexBase + static_cast<unsigned int>(vertex)) * 3u;
    const unsigned int cbase = (vertexBase + static_cast<unsigned int>(vertex)) * 4u;
    verts[vbase + 0u] = a > 0.0f ? vx[vertex] : 0.0f;
    verts[vbase + 1u] = a > 0.0f ? vy[vertex] : 0.0f;
    verts[vbase + 2u] = 0.0f;
    colors[cbase + 0u] = clamp01(r);
    colors[cbase + 1u] = clamp01(g);
    colors[cbase + 2u] = clamp01(b);
    colors[cbase + 3u] = clamp01(a);
  }
}

__global__ void glossField2DGeometryFromFieldKernel(float* lineVerts,
                                                    float* lineColors,
                                                    float* fillVerts,
                                                    float* fillColors,
                                                    const float* fieldWorkspace,
                                                    GlossField2DGeometryRequest request) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int gridWidth = max(request.gridWidth, 1);
  const int gridHeight = max(request.gridHeight, 1);
  const unsigned int cellCount = static_cast<unsigned int>(gridWidth * gridHeight);
  if (!lineVerts || !lineColors || !fillVerts || !fillColors || !fieldWorkspace) return;

  if (index == 0u) {
    for (int vertex = 0; vertex < 2; ++vertex) {
      const unsigned int vbase = static_cast<unsigned int>(vertex) * 3u;
      const unsigned int cbase = static_cast<unsigned int>(vertex) * 4u;
      lineVerts[vbase + 0u] = 0.0f;
      lineVerts[vbase + 1u] = 0.0f;
      lineVerts[vbase + 2u] = 0.0f;
      lineColors[cbase + 0u] = 0.0f;
      lineColors[cbase + 1u] = 0.0f;
      lineColors[cbase + 2u] = 0.0f;
      lineColors[cbase + 3u] = 0.0f;
    }
  }
  if (index >= cellCount) return;

  const size_t cell = static_cast<size_t>(index);
  const size_t count = static_cast<size_t>(cellCount);
  const float* occupancy = fieldWorkspace + count * 0u;
  const float* meanR = fieldWorkspace + count * 8u;
  const float* meanG = fieldWorkspace + count * 9u;
  const float* meanB = fieldWorkspace + count * 10u;
  const float* carrierY = fieldWorkspace + count * 11u;
  const float* carrierMax = fieldWorkspace + count * 12u;
  const float* carrierMin = fieldWorkspace + count * 13u;
  const float* neutrality = fieldWorkspace + count * 14u;
  const size_t solutionBase =
      request.algorithm != 0 ? kGlossFieldCandidate2Base : kGlossFieldCandidate1Base;
  const float* body = fieldWorkspace + count * (solutionBase + 0u);
  const float* positive = fieldWorkspace + count * (solutionBase + 2u);
  const float* negative = fieldWorkspace + count * (solutionBase + 3u);
  const float* boundaryValues = fieldWorkspace + count * (solutionBase + 4u);
  const float* congruence = fieldWorkspace + count * (solutionBase + 5u);
  const float* confidenceValues = fieldWorkspace + count * (solutionBase + 6u);

  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const float cellW = (request.right - request.left) / static_cast<float>(gridWidth);
  const float cellH = (request.top - request.bottom) / static_cast<float>(gridHeight);
  const float x0 = request.left + static_cast<float>(x) * cellW;
  const float x1 = x0 + cellW + 0.4f;
  const float y1 = request.top - static_cast<float>(y) * cellH;
  const float y0 = y1 - cellH - 0.4f;

  const float mr = meanR[cell];
  const float mg = meanG[cell];
  const float mb = meanB[cell];
  const float sr0 = glossProjectionGamma(mr);
  const float sg0 = glossProjectionGamma(mg);
  const float sb0 = glossProjectionGamma(mb);
  float sr = sr0;
  float sg = sg0;
  float sb = sb0;
  glossProjectionApplySaturation(request.colorSaturation, &sr, &sg, &sb);
  const float sourceLuma = clamp01(0.2126f * sr + 0.7152f * sg + 0.0722f * sb);
  const float sourcePresence = clamp01(fmaxf(mr, fmaxf(mg, mb)));
  const float confidence = clamp01(confidenceValues[cell]);
  const float base = clamp01(body[cell]);
  const float structure = fmaxf(sqrtf(confidence), sqrtf(sourcePresence));
  const bool hasSource = occupancy[cell] > 0.5f || sourcePresence > 0.01f;

  float underR = 0.0f;
  float underG = 0.0f;
  float underB = 0.0f;
  float underA = 0.0f;
  if (hasSource) {
    if (request.colorMode == 0) {
      const float neutralPow = powf(sourceLuma, 0.85f);
      const float nr = 0.10f + 0.52f * neutralPow;
      const float ng = 0.10f + 0.50f * neutralPow;
      const float nb = 0.11f + 0.46f * neutralPow;
      glossProjectionMix(nr, ng, nb, sr, sg, sb, 0.42f, &underR, &underG, &underB);
    } else {
      const float gray = 0.11f + 0.62f * powf(fmaxf(sourceLuma, 0.35f * base), 0.84f);
      underR = gray * 0.98f;
      underG = gray * 0.985f;
      underB = gray;
    }
    underA = fminf(fmaxf((0.10f + 0.48f * structure) *
                         (0.34f + 0.66f * request.glossBodyOpacity),
                         0.0f),
                   0.68f);
  }

  float positiveValue = positive[cell];
  float negativeValue = negative[cell];
  if (request.debugMode == 1) {
    positiveValue = clamp01(carrierMax[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 2) {
    positiveValue = clamp01(carrierY[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 3) {
    positiveValue = clamp01(carrierMin[cell]);
    negativeValue = 0.0f;
  } else if (request.debugMode == 4) {
    positiveValue = clamp01(neutrality[cell]);
    negativeValue = 0.0f;
  }

  const float structureStrength = fmaxf(clamp01(congruence[cell]), clamp01(boundaryValues[cell]));
  const float signalScale = fmaxf(request.glossLiftScale, 1.0f);
  positiveValue = clamp01(positiveValue * signalScale);
  negativeValue = clamp01(negativeValue * signalScale);
  const float positiveT = clamp01((positiveValue - 0.035f) / (1.0f - 0.035f));
  const float negativeT = clamp01((negativeValue - 0.035f) / (1.0f - 0.035f));
  const float positiveDisplay = positiveT * positiveT * (3.0f - 2.0f * positiveT);
  const float negativeDisplay = negativeT * negativeT * (3.0f - 2.0f * negativeT);
  const float signalPresence = fmaxf(positiveDisplay, negativeDisplay);

  float cr = 0.03f;
  float cg = 0.03f;
  float cb = 0.04f;
  if (request.colorMode == 0) {
    const float baseMix = clamp01(request.glossBodyOpacity *
                                  (0.22f + 0.78f * confidence) *
                                  (0.86f - 0.22f * signalPresence));
    const float basePow = powf(base, 0.78f);
    float nr = 0.16f + 0.60f * basePow;
    float ng = 0.16f + 0.58f * basePow;
    float nb = 0.17f + 0.54f * basePow;
    float mixR = 0.0f;
    float mixG = 0.0f;
    float mixB = 0.0f;
    glossProjectionMix(nr, ng, nb, sr, sg, sb, 0.68f, &mixR, &mixG, &mixB);
    glossProjectionMix(0.03f, 0.03f, 0.04f, mixR, mixG, mixB, baseMix, &cr, &cg, &cb);
    if (positiveDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 1.0f, 0.95f, 0.86f, 0.54f, &wr, &wg, &wb);
      glossProjectionMix(cr, cg, cb, wr, wg, wb,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
    if (negativeDisplay > 0.0f) {
      float wr = 0.0f;
      float wg = 0.0f;
      float wb = 0.0f;
      glossProjectionMix(sr, sg, sb, 0.08f, 0.14f, 0.24f, 0.74f, &wr, &wg, &wb);
      glossProjectionMix(cr, cg, cb, wr, wg, wb,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
  } else {
    const float basePow = powf(base, 0.78f);
    glossProjectionMix(0.03f,
                       0.03f,
                       0.04f,
                       0.16f + 0.64f * basePow,
                       0.16f + 0.64f * basePow,
                       0.17f + 0.60f * basePow,
                       clamp01(request.glossBodyOpacity *
                               (0.22f + 0.78f * confidence) *
                               (0.86f - 0.22f * signalPresence)),
                       &cr,
                       &cg,
                       &cb);
    if (positiveDisplay > 0.0f) {
      glossProjectionMix(cr, cg, cb, 1.0f, 0.89f, 0.36f,
                         clamp01(request.glossHighlightOpacity * positiveDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
    if (negativeDisplay > 0.0f) {
      glossProjectionMix(cr, cg, cb, 0.22f, 0.76f, 1.0f,
                         clamp01(request.glossHighlightOpacity * negativeDisplay *
                                  (0.22f + 0.78f * structureStrength)),
                         &cr, &cg, &cb);
    }
  }

  const float boundary = clamp01(boundaryValues[cell]);
  if (boundary > 0.0f) {
    glossProjectionMix(cr, cg, cb, 0.98f, 0.98f, 0.94f,
                       fminf(0.10f + 0.26f * boundary, 0.34f),
                       &cr, &cg, &cb);
  }
  float alpha = fminf(fmaxf(request.glossBodyOpacity * (0.12f + 0.62f * confidence) *
                               (0.82f - 0.18f * signalPresence) +
                           request.glossHighlightOpacity * signalPresence *
                               (0.16f + 0.84f * structureStrength),
                           0.018f),
                      1.0f);
  if (request.diagnosticMode == 1) {
    const float gray = 0.16f + 0.78f * confidence;
    glossProjectionMix(cr, cg, cb, gray, gray, gray, 0.36f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 1.0f, 1.0f, 0.96f, 0.10f * boundary, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.55f + 0.45f * confidence) + 0.10f * confidence);
  } else if (request.diagnosticMode == 2) {
    const float ambiguity = clamp01(1.0f - confidence);
    const float gray = 0.12f + 0.74f * ambiguity;
    glossProjectionMix(cr, cg, cb, gray * 0.94f, gray * 0.97f, gray, 0.34f, &cr, &cg, &cb);
    glossProjectionMix(cr, cg, cb, 0.80f, 0.90f, 1.0f, 0.10f * boundary * ambiguity, &cr, &cg, &cb);
    alpha = clamp01(alpha * (0.48f + 0.52f * ambiguity) + 0.08f * ambiguity);
  }
  if (!hasSource || (confidence <= 0.01f && sourcePresence <= 0.01f)) {
    alpha = 0.0f;
  }

  const unsigned int fillBase = index * 12u;
  glossField2DEmitQuad(fillVerts, fillColors, fillBase, x0, y0, x1, y1,
                       underR, underG, underB, underA > 0.01f ? underA : 0.0f);
  glossField2DEmitQuad(fillVerts, fillColors, fillBase + 6u, x0, y0, x1, y1,
                       cr, cg, cb, alpha > 0.01f ? alpha : 0.0f);
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

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA bounds resource: ") + errorString(err);
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
  if (err == cudaSuccess) {
    const unsigned int threads = 256u;
    const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(cache->pointCount) + threads - 1u) / threads);
    boundsKernel<<<blocks, threads>>>(devVerts, devColors, impl->deviceBounds, cache->pointCount);
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

StartupValidationResult warmupRuntime() {
  StartupValidationResult result{};
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount <= 0) {
    result.reason = err != cudaSuccess
                        ? std::string("CUDA runtime warm-up failed: ") + errorString(err)
                        : std::string("No CUDA devices found.");
    return result;
  }
  err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    result.reason = std::string("CUDA runtime device selection failed: ") + errorString(err);
    return result;
  }
  err = cudaFree(0);
  if (err != cudaSuccess) {
    result.reason = std::string("CUDA runtime warm-up failed: ") + errorString(err);
    return result;
  }
  result.ready = true;
  result.reason = "runtime-warmed";
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

void releaseScopeGeometryCache(ScopeGeometryCache* cache) {
  releaseScopeGeometryCacheInternal(cache);
}

void releaseImportedSource(ImportedSource* source) {
  if (!source) return;
  if (source->devicePtr) {
    cudaIpcCloseMemHandle(source->devicePtr);
  }
  *source = ImportedSource{};
}

void releaseSourceTextureCache(SourceTextureCache* cache) {
  if (!cache) return;
  releaseSourceTextureImpl(reinterpret_cast<SourceTextureCacheImpl*>(cache->internal));
  *cache = SourceTextureCache{};
}

bool importSourceIpc(ImportedSource* source,
                     const std::string& handleHex,
                     size_t byteSize,
                     std::string* error) {
  if (!source || handleHex.empty() || byteSize == 0) {
    if (error) *error = "Invalid CUDA IPC source metadata.";
    return false;
  }
  if (source->available && source->devicePtr && source->handleHex == handleHex &&
      source->byteSize == byteSize) {
    return true;
  }
  releaseImportedSource(source);

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  cudaIpcMemHandle_t handle{};
  if (!decodeHexBytes(handleHex, &handle, sizeof(handle))) {
    if (error) *error = "Invalid CUDA IPC handle encoding.";
    return false;
  }
  void* devicePtr = nullptr;
  const cudaError_t err = cudaIpcOpenMemHandle(&devicePtr, handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess || !devicePtr) {
    if (error) *error = std::string("cudaIpcOpenMemHandle failed: ") + errorString(err);
    return false;
  }
  source->devicePtr = devicePtr;
  source->byteSize = byteSize;
  source->handleHex = handleHex;
  source->available = true;
  return true;
}

bool copyDeviceRgba32fToTexture(SourceTextureCache* cache,
                                const void* devicePtr,
                                size_t byteSize,
                                size_t sourceRowBytes,
                                int width,
                                int height,
                                unsigned int glTexture,
                                std::string* error) {
  if (!cache || !devicePtr || byteSize == 0 || sourceRowBytes == 0 ||
      width <= 0 || height <= 0 || glTexture == 0) {
    if (error) *error = "Invalid CUDA texture copy request.";
    return false;
  }
  const size_t rowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
  if (sourceRowBytes < rowBytes || byteSize < sourceRowBytes * static_cast<size_t>(height)) {
    if (error) *error = "CUDA texture copy source is too small.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  SourceTextureCacheImpl* impl = reinterpret_cast<SourceTextureCacheImpl*>(cache->internal);
  if (!impl) {
    impl = new SourceTextureCacheImpl();
    cache->internal = impl;
  }
  if (impl->textureResource &&
      (impl->registeredTexture != static_cast<GLuint>(glTexture) ||
       impl->width != width ||
       impl->height != height)) {
    releaseSourceTextureImpl(impl);
    impl = new SourceTextureCacheImpl();
    cache->internal = impl;
    cache->available = false;
  }
  if (!impl->textureResource) {
    cudaError_t registerErr = cudaGraphicsGLRegisterImage(&impl->textureResource,
                                                          static_cast<GLuint>(glTexture),
                                                          GL_TEXTURE_2D,
                                                          cudaGraphicsRegisterFlagsWriteDiscard);
    if (registerErr != cudaSuccess) {
      releaseSourceTextureCache(cache);
      if (error) *error = std::string("Failed to register CUDA source texture: ") + errorString(registerErr);
      return false;
    }
    impl->registeredTexture = static_cast<GLuint>(glTexture);
    impl->width = width;
    impl->height = height;
  }
  cudaGraphicsResource* resource = impl->textureResource;
  cudaError_t err = cudaGraphicsMapResources(1, &resource, 0);
  const bool mapped = err == cudaSuccess;
  cudaArray_t array = nullptr;
  if (err == cudaSuccess) {
    err = cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy2DToArray(array,
                              0,
                              0,
                              devicePtr,
                              sourceRowBytes,
                              rowBytes,
                              static_cast<size_t>(height),
                              cudaMemcpyDeviceToDevice);
  }
  if (mapped) {
    cudaError_t unmapErr = cudaGraphicsUnmapResources(1, &resource, 0);
    if (err == cudaSuccess) err = unmapErr;
  }
  if (err != cudaSuccess) {
    releaseSourceTextureCache(cache);
    if (error) *error = std::string("CUDA source texture copy failed: ") + errorString(err);
    return false;
  }
  cache->glTexture = static_cast<unsigned int>(glTexture);
  cache->width = width;
  cache->height = height;
  cache->available = true;
  return true;
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
  uniforms.cubeSlicingEnabled = request.cubeSlicingEnabled;
  uniforms.neutralRadiusEnabled = request.neutralRadiusEnabled;
  uniforms.neutralRadius = request.neutralRadius;
  uniforms.cubeSliceRed = request.cubeSliceRed;
  uniforms.cubeSliceYellow = request.cubeSliceYellow;
  uniforms.cubeSliceGreen = request.cubeSliceGreen;
  uniforms.cubeSliceCyan = request.cubeSliceCyan;
  uniforms.cubeSliceBlue = request.cubeSliceBlue;
  uniforms.cubeSliceMagenta = request.cubeSliceMagenta;
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
  const float* inputPtr = request.useInputPoints != 0 ? inputPoints.data() : nullptr;
  const size_t inputFloatCount = request.useInputPoints != 0 ? inputPoints.size() : 0u;
  return buildMesh(cache, pointCount, inputPtr, inputFloatCount, uniforms, launchOverlay, serial, error);
}

bool buildInputMesh(InputCache* cache,
                    const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    bool allowHostUpload,
                    unsigned long long serial,
                    std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA input cache has no GL buffers.";
    return false;
  }
  if (!allowHostUpload) {
    if (error) *error = "CUDA input host upload disabled; resident source path required.";
    return false;
  }
  const size_t pointCount = request.inputStride > 0 ? (rawPoints.size() / static_cast<size_t>(request.inputStride)) : 0u;
  InputKernelUniforms uniforms{};
  uniforms.pointCount = request.pointCount;
  uniforms.inputStride = request.inputStride;
  uniforms.glossView = request.glossView;
  uniforms.sourceAspect = request.sourceAspect;
  uniforms.glossLiftScale = request.glossLiftScale;
  uniforms.showOverflow = request.remap.showOverflow;
  uniforms.highlightOverflow = request.remap.highlightOverflow;
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

bool buildRasterSourceMesh(InputCache* cache,
                           const RasterSourceRequest& request,
                           const void* sourceBytes,
                           size_t sourceByteCount,
                           bool allowHostUpload,
                           unsigned long long serial,
                           std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA raster source cache has no GL buffers.";
    return false;
  }
  if (request.sourceBytesAreDevice == 0 && !allowHostUpload) {
    if (error) *error = "CUDA raster source host upload disabled; CUDA IPC/device source required.";
    return false;
  }
  const size_t pointCount = static_cast<size_t>(std::max(request.pointCount, 0));
  if (pointCount == 0 || !sourceBytes || sourceByteCount == 0 ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 || request.sampleCountX <= 0) {
    if (error) *error = "Invalid CUDA raster source request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA raster source cache.";
    return false;
  }
  if (!ensureRegistered(cache->verts, cache->colors, pointCount, impl, &localError)) {
    if (error) *error = localError;
    return false;
  }
  const unsigned char* kernelSource = nullptr;
  cudaError_t err = cudaSuccess;
  if (request.sourceBytesAreDevice != 0) {
    kernelSource = static_cast<const unsigned char*>(sourceBytes);
  } else {
    // Host-byte upload is intentionally isolated here. Full CUDA residency
    // requires the caller to use buildRasterSourceMeshFromDevice with a
    // CUDA-visible Source Signal resource instead of shared-memory bytes.
    if (!ensureSourceCapacity(impl, sourceByteCount, &localError)) {
      if (error) *error = localError;
      return false;
    }
    err = cudaMemcpy(impl->deviceSource, sourceBytes, sourceByteCount, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      if (error) *error = std::string("Failed to upload CUDA raster source: ") + errorString(err);
      return false;
    }
    kernelSource = impl->deviceSource;
  }

  RasterSourceKernelUniforms uniforms{};
  uniforms.input.pointCount = request.pointCount;
  uniforms.input.inputStride = 3;
  uniforms.input.glossView = request.remap.plotMode == 9 ? 1 : 0;
  uniforms.input.sourceAspect = request.sourceAspect;
  uniforms.input.glossLiftScale = request.glossLiftScale;
  uniforms.input.showOverflow = request.remap.showOverflow;
  uniforms.input.highlightOverflow = request.remap.highlightOverflow;
  uniforms.input.plotMode = request.remap.plotMode;
  uniforms.input.circularHsl = request.remap.circularHsl;
  uniforms.input.circularHsv = request.remap.circularHsv;
  uniforms.input.normConeNormalized = request.remap.normConeNormalized;
  uniforms.input.chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms.input.chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms.input.chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms.input.chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms.input.chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms.input.chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }
  uniforms.input.pointAlphaScale = request.pointAlphaScale;
  uniforms.input.denseAlphaBias = request.denseAlphaBias;
  uniforms.input.colorSaturation = request.colorSaturation;
  uniforms.basePointCount = request.basePointCount > 0 ? request.basePointCount : request.pointCount;
  uniforms.sourceWidth = request.sourceWidth;
  uniforms.sourceHeight = request.sourceHeight;
  uniforms.sampleStride = request.sampleStride;
  uniforms.sampleCountX = request.sampleCountX;
  uniforms.pixelFormat = request.pixelFormat;
  uniforms.plotLinear = request.plotLinear;
  uniforms.plotLinearTransfer = request.plotLinearTransfer;
  uniforms.excludeIdentityData = request.excludeIdentityData;
  uniforms.isolateIdentityData = request.isolateIdentityData;
  uniforms.readIdentityPlot = request.readIdentityPlot;
  uniforms.readGrayRamp = request.readGrayRamp;
  uniforms.identityCubeY1 = request.identityCubeY1;
  uniforms.identityCubeY2 = request.identityCubeY2;
  uniforms.identityRampY1 = request.identityRampY1;
  uniforms.identityRampY2 = request.identityRampY2;
  uniforms.identityCubeAppendOffset = request.identityCubeAppendOffset;
  uniforms.identityCubeAppendCount = request.identityCubeAppendCount;
  uniforms.identityCubeAppendY1 = request.identityCubeAppendY1;
  uniforms.identityCubeAppendY2 = request.identityCubeAppendY2;
  uniforms.identityCubeAppendRowStep = request.identityCubeAppendRowStep;
  uniforms.identityCubeAppendXStep = request.identityCubeAppendXStep;
  uniforms.identityRampAppendOffset = request.identityRampAppendOffset;
  uniforms.identityRampAppendCount = request.identityRampAppendCount;
  uniforms.identityRampAppendY1 = request.identityRampAppendY1;
  uniforms.identityRampAppendY2 = request.identityRampAppendY2;
  uniforms.identityRampAppendRowStep = request.identityRampAppendRowStep;
  uniforms.identityRampAppendXStep = request.identityRampAppendXStep;
  uniforms.occupancyFill = request.occupancyFill;
  uniforms.occupancyAppendOffset = request.occupancyAppendOffset;
  uniforms.occupancyAppendCount = request.occupancyAppendCount;
  uniforms.occupancyCandidateCount = request.occupancyCandidateCount;
  uniforms.occupancyTargetThreshold = 0;
  uniforms.lassoEnabled = request.lassoEnabled;
  uniforms.lassoStrokeCount = request.lassoStrokeCount;
  uniforms.lassoPointCount = request.lassoPointCount;
  for (int i = 0; i < 16; ++i) {
    uniforms.lassoStrokeFirst[i] = request.lassoStrokeFirst[i];
    uniforms.lassoStrokeCountPerStroke[i] = request.lassoStrokeCountPerStroke[i];
    uniforms.lassoStrokeSubtract[i] = request.lassoStrokeSubtract[i];
  }
  for (int i = 0; i < 256; ++i) {
    uniforms.lassoX[i] = request.lassoX[i];
    uniforms.lassoY[i] = request.lassoY[i];
  }
  uniforms.cubeSlicingEnabled = request.cubeSlicingEnabled;
  uniforms.neutralRadiusEnabled = request.neutralRadiusEnabled;
  uniforms.neutralRadius = request.neutralRadius;
  uniforms.cubeSliceRed = request.cubeSliceRed;
  uniforms.cubeSliceYellow = request.cubeSliceYellow;
  uniforms.cubeSliceGreen = request.cubeSliceGreen;
  uniforms.cubeSliceCyan = request.cubeSliceCyan;
  uniforms.cubeSliceBlue = request.cubeSliceBlue;
  uniforms.cubeSliceMagenta = request.cubeSliceMagenta;

  int* occupancyBins = nullptr;
  int* visibleCount = nullptr;
  int* occupancyTargetThreshold = nullptr;
  constexpr int kRasterOccupancyBinCount = 18 * 18 * 18;
  if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
    err = cudaMalloc(reinterpret_cast<void**>(&occupancyBins),
                     static_cast<size_t>(kRasterOccupancyBinCount) * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&visibleCount), sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&occupancyTargetThreshold), sizeof(int));
    if (err != cudaSuccess || !occupancyBins || !visibleCount || !occupancyTargetThreshold) {
      if (occupancyBins) cudaFree(occupancyBins);
      if (visibleCount) cudaFree(visibleCount);
      if (occupancyTargetThreshold) cudaFree(occupancyTargetThreshold);
      if (error) *error = std::string("Failed to allocate CUDA raster occupancy buffers: ") + errorString(err);
      return false;
    }
    err = cudaMemset(occupancyBins, 0, static_cast<size_t>(kRasterOccupancyBinCount) * sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(visibleCount, 0, sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(occupancyTargetThreshold, 0, sizeof(int));
    if (err != cudaSuccess) {
      cudaFree(occupancyBins);
      cudaFree(visibleCount);
      cudaFree(occupancyTargetThreshold);
      if (error) *error = std::string("Failed to clear CUDA raster occupancy buffers: ") + errorString(err);
      return false;
    }
    const unsigned int threads = 256u;
    const unsigned int countBlocks =
        static_cast<unsigned int>((static_cast<size_t>(std::max(uniforms.basePointCount, 0)) + threads - 1u) / threads);
    rasterOccupancyCountKernel<<<std::max(1u, countBlocks), threads>>>(
        kernelSource, uniforms, occupancyBins, visibleCount);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
      rasterOccupancyThresholdKernel<<<1u, 1u>>>(visibleCount, occupancyTargetThreshold);
      err = cudaGetLastError();
    }
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      cudaFree(occupancyBins);
      cudaFree(visibleCount);
      cudaFree(occupancyTargetThreshold);
      if (error) *error = std::string("CUDA raster occupancy count failed: ") + errorString(err);
      return false;
    }
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (occupancyBins) cudaFree(occupancyBins);
    if (visibleCount) cudaFree(visibleCount);
    if (occupancyTargetThreshold) cudaFree(occupancyTargetThreshold);
    if (error) *error = std::string("Failed to map CUDA raster GL resources: ") + errorString(err);
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
    if (occupancyBins) cudaFree(occupancyBins);
    if (visibleCount) cudaFree(visibleCount);
    if (occupancyTargetThreshold) cudaFree(occupancyTargetThreshold);
    if (error) *error = std::string("Failed to access CUDA raster mapped buffers: ") + errorString(err);
    return false;
  }

  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  rasterSourceKernel<<<blocks, threads>>>(
      devVerts, devColors, kernelSource, uniforms, occupancyBins, occupancyTargetThreshold);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (occupancyBins) cudaFree(occupancyBins);
  if (visibleCount) cudaFree(visibleCount);
  if (occupancyTargetThreshold) cudaFree(occupancyTargetThreshold);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster source kernel failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(pointCount);
  cache->available = true;
  cache->hasFitBounds = false;
  // Keep the source pixels and drawable geometry resident while crossing only
  // compact fit metadata back to the CPU for viewer camera fitting.
  std::string boundsError;
  if (!computeInputBounds(cache, &boundsError) && error && error->empty()) {
    *error = boundsError;
  }
  return true;
}

bool buildRasterSourceMeshFromDevice(InputCache* cache,
                                     const RasterSourceRequest& request,
                                     const void* sourceDeviceBytes,
                                     size_t sourceByteCount,
                                     unsigned long long serial,
                                     std::string* error) {
  RasterSourceRequest deviceRequest = request;
  deviceRequest.sourceBytesAreDevice = 1;
  return buildRasterSourceMesh(cache, deviceRequest, sourceDeviceBytes, sourceByteCount, false, serial, error);
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

bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       bool allowReadback,
                       std::vector<float>* outDensity,
                       std::string* error) {
  if (!outDensity) {
    if (error) *error = "Missing CUDA scope-density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "CUDA compact scope-density readback disabled; resident source-to-draw path required.";
    return false;
  }
  const int pointCount = request.pointCount > 0
                             ? request.pointCount
                             : static_cast<int>(packedSamples.size() / 5u);
  const int width = std::max(request.width, 1);
  const int height = std::max(request.height, 1);
  const int channelCount = std::max(1, request.channelCount);
  const size_t binCount = static_cast<size_t>(width) * static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  if (pointCount <= 0 || packedSamples.size() < static_cast<size_t>(pointCount) * 5u || binCount == 0u) {
    if (error) *error = "Invalid CUDA scope-density request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  float* deviceSamples = nullptr;
  unsigned int* deviceDensity = nullptr;
  cudaError_t err = cudaMalloc(&deviceSamples, static_cast<size_t>(pointCount) * 5u * sizeof(float));
  if (err == cudaSuccess) err = cudaMalloc(&deviceDensity, binCount * sizeof(unsigned int));
  if (err != cudaSuccess) {
    if (deviceSamples) cudaFree(deviceSamples);
    if (deviceDensity) cudaFree(deviceDensity);
    if (error) *error = std::string("Failed to allocate CUDA scope-density buffers: ") + errorString(err);
    return false;
  }

  err = cudaMemcpy(deviceSamples,
                   packedSamples.data(),
                   static_cast<size_t>(pointCount) * 5u * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemset(deviceDensity, 0, binCount * sizeof(unsigned int));
  if (err != cudaSuccess) {
    cudaFree(deviceSamples);
    cudaFree(deviceDensity);
    if (error) *error = std::string("Failed to upload CUDA scope-density input: ") + errorString(err);
    return false;
  }

  ScopeDensityRequest kernelRequest = request;
  kernelRequest.pointCount = pointCount;
  kernelRequest.width = width;
  kernelRequest.height = height;
  kernelRequest.channelCount = channelCount;
  kernelRequest.lumaMethod = std::clamp(kernelRequest.lumaMethod, 0, 3);
  kernelRequest.onlyOverflow = kernelRequest.onlyOverflow != 0 ? 1 : 0;
  kernelRequest.excludeOverflow = kernelRequest.excludeOverflow != 0 ? 1 : 0;
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  scopeDensityKernel<<<blocks, threads>>>(deviceSamples, deviceDensity, kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(deviceSamples);
    cudaFree(deviceDensity);
    if (error) *error = std::string("CUDA scope-density kernel failed: ") + errorString(err);
    return false;
  }

  std::vector<unsigned int> bins(binCount, 0u);
  err = cudaMemcpy(bins.data(), deviceDensity, binCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(deviceSamples);
  cudaFree(deviceDensity);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA scope-density bins: ") + errorString(err);
    return false;
  }
  outDensity->resize(binCount);
  std::transform(bins.begin(), bins.end(), outDensity->begin(),
                 [](unsigned int value) { return static_cast<float>(value); });
  return true;
}

RasterSourceKernelUniforms makeRasterSourceUniforms(const RasterSourceRequest& request) {
  RasterSourceKernelUniforms uniforms{};
  uniforms.input.pointCount = request.pointCount;
  uniforms.input.inputStride = 3;
  uniforms.input.glossView = request.remap.plotMode == 9 ? 1 : 0;
  uniforms.input.sourceAspect = request.sourceAspect;
  uniforms.input.glossLiftScale = request.glossLiftScale;
  uniforms.input.showOverflow = request.remap.showOverflow;
  uniforms.input.highlightOverflow = request.remap.highlightOverflow;
  uniforms.input.plotMode = request.remap.plotMode;
  uniforms.input.circularHsl = request.remap.circularHsl;
  uniforms.input.circularHsv = request.remap.circularHsv;
  uniforms.input.normConeNormalized = request.remap.normConeNormalized;
  uniforms.input.chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms.input.chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms.input.chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms.input.chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms.input.chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms.input.chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }
  uniforms.input.pointAlphaScale = request.pointAlphaScale;
  uniforms.input.denseAlphaBias = request.denseAlphaBias;
  uniforms.input.colorSaturation = request.colorSaturation;
  uniforms.basePointCount = request.basePointCount > 0 ? request.basePointCount : request.pointCount;
  uniforms.sourceWidth = request.sourceWidth;
  uniforms.sourceHeight = request.sourceHeight;
  uniforms.sampleStride = request.sampleStride;
  uniforms.sampleCountX = request.sampleCountX;
  uniforms.pixelFormat = request.pixelFormat;
  uniforms.plotLinear = request.plotLinear;
  uniforms.plotLinearTransfer = request.plotLinearTransfer;
  uniforms.excludeIdentityData = request.excludeIdentityData;
  uniforms.isolateIdentityData = request.isolateIdentityData;
  uniforms.readIdentityPlot = request.readIdentityPlot;
  uniforms.readGrayRamp = request.readGrayRamp;
  uniforms.identityCubeY1 = request.identityCubeY1;
  uniforms.identityCubeY2 = request.identityCubeY2;
  uniforms.identityRampY1 = request.identityRampY1;
  uniforms.identityRampY2 = request.identityRampY2;
  uniforms.identityCubeAppendOffset = request.identityCubeAppendOffset;
  uniforms.identityCubeAppendCount = request.identityCubeAppendCount;
  uniforms.identityCubeAppendY1 = request.identityCubeAppendY1;
  uniforms.identityCubeAppendY2 = request.identityCubeAppendY2;
  uniforms.identityCubeAppendRowStep = request.identityCubeAppendRowStep;
  uniforms.identityCubeAppendXStep = request.identityCubeAppendXStep;
  uniforms.identityRampAppendOffset = request.identityRampAppendOffset;
  uniforms.identityRampAppendCount = request.identityRampAppendCount;
  uniforms.identityRampAppendY1 = request.identityRampAppendY1;
  uniforms.identityRampAppendY2 = request.identityRampAppendY2;
  uniforms.identityRampAppendRowStep = request.identityRampAppendRowStep;
  uniforms.identityRampAppendXStep = request.identityRampAppendXStep;
  uniforms.occupancyFill = request.occupancyFill;
  uniforms.occupancyAppendOffset = request.occupancyAppendOffset;
  uniforms.occupancyAppendCount = request.occupancyAppendCount;
  uniforms.occupancyCandidateCount = request.occupancyCandidateCount;
  uniforms.occupancyTargetThreshold = 0;
  uniforms.lassoEnabled = request.lassoEnabled;
  uniforms.lassoStrokeCount = request.lassoStrokeCount;
  uniforms.lassoPointCount = request.lassoPointCount;
  for (int i = 0; i < 16; ++i) {
    uniforms.lassoStrokeFirst[i] = request.lassoStrokeFirst[i];
    uniforms.lassoStrokeCountPerStroke[i] = request.lassoStrokeCountPerStroke[i];
    uniforms.lassoStrokeSubtract[i] = request.lassoStrokeSubtract[i];
  }
  for (int i = 0; i < 256; ++i) {
    uniforms.lassoX[i] = request.lassoX[i];
    uniforms.lassoY[i] = request.lassoY[i];
  }
  uniforms.cubeSlicingEnabled = request.cubeSlicingEnabled;
  uniforms.neutralRadiusEnabled = request.neutralRadiusEnabled;
  uniforms.neutralRadius = request.neutralRadius;
  uniforms.cubeSliceRed = request.cubeSliceRed;
  uniforms.cubeSliceYellow = request.cubeSliceYellow;
  uniforms.cubeSliceGreen = request.cubeSliceGreen;
  uniforms.cubeSliceCyan = request.cubeSliceCyan;
  uniforms.cubeSliceBlue = request.cubeSliceBlue;
  uniforms.cubeSliceMagenta = request.cubeSliceMagenta;
  return uniforms;
}

bool buildScopeDensityFromRasterSourceDevice(const RasterSourceRequest& rasterRequest,
                                             const ScopeDensityRequest& scopeRequest,
                                             const void* sourceDeviceBytes,
                                             size_t sourceByteCount,
                                             bool allowReadback,
                                             std::vector<float>* outDensity,
                                             std::string* error) {
  if (!outDensity) {
    if (error) *error = "Missing CUDA raster scope-density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "CUDA raster scope-density readback disabled; resident draw path required.";
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(scopeRequest.width, 1);
  const int height = std::max(scopeRequest.height, 1);
  const int channelCount = std::max(1, scopeRequest.channelCount);
  const size_t binCount = static_cast<size_t>(width) * static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  const size_t pixelCount = static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) *
                            static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  const size_t expectedBytes =
      pixelCount * 4u * (rasterRequest.pixelFormat == 1 ? sizeof(float) : sizeof(unsigned short));
  if (pointCount <= 0 || !sourceDeviceBytes || sourceByteCount < expectedBytes ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || binCount == 0u) {
    if (error) *error = "Invalid CUDA raster source scope-density request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  unsigned int* deviceDensity = nullptr;
  cudaError_t err = cudaMalloc(&deviceDensity, binCount * sizeof(unsigned int));
  if (err == cudaSuccess) err = cudaMemset(deviceDensity, 0, binCount * sizeof(unsigned int));
  if (err != cudaSuccess || !deviceDensity) {
    if (deviceDensity) cudaFree(deviceDensity);
    if (error) *error = std::string("Failed to allocate CUDA raster scope-density bins: ") + errorString(err);
    return false;
  }

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = pointCount;
  ScopeDensityRequest kernelScope = scopeRequest;
  kernelScope.pointCount = pointCount;
  kernelScope.width = width;
  kernelScope.height = height;
  kernelScope.channelCount = channelCount;
  kernelScope.lumaMethod = std::clamp(kernelScope.lumaMethod, 0, 3);
  kernelScope.onlyOverflow = kernelScope.onlyOverflow != 0 ? 1 : 0;
  kernelScope.excludeOverflow = kernelScope.excludeOverflow != 0 ? 1 : 0;

  const unsigned int threads = 256u;
  const unsigned int blocks =
      static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  rasterScopeDensityKernel<<<blocks, threads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, kernelScope, nullptr, deviceDensity);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(deviceDensity);
    if (error) *error = std::string("CUDA raster source scope-density kernel failed: ") + errorString(err);
    return false;
  }

  std::vector<unsigned int> bins(binCount, 0u);
  err = cudaMemcpy(bins.data(), deviceDensity, binCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaFree(deviceDensity);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA raster source scope-density bins: ") + errorString(err);
    return false;
  }
  outDensity->resize(binCount);
  std::transform(bins.begin(), bins.end(), outDensity->begin(),
                 [](unsigned int value) { return static_cast<float>(value); });
  return true;
}

bool buildScopeAutoRangeDevice(const RasterSourceKernelUniforms& rasterUniforms,
                               const ScopeRangeRequest& rangeRequest,
                               const void* sourceDeviceBytes,
                               unsigned int pointCount,
                               int previousRangeValid,
                               float previousRangeMin,
                               float previousRangeMax,
                               unsigned int** outFinalRangeBits,
                               std::string* error) {
  if (outFinalRangeBits) *outFinalRangeBits = nullptr;
  if (!outFinalRangeBits || !sourceDeviceBytes || pointCount == 0u) {
    if (error) *error = "Invalid CUDA resident scope auto-range request.";
    return false;
  }
  constexpr unsigned int kRangeHistogramBins = 2048u;
  const unsigned int initRangeBits[3] = {
      orderedUintFromFloatHost(std::numeric_limits<float>::infinity()),
      orderedUintFromFloatHost(-std::numeric_limits<float>::infinity()),
      0u,
  };
  unsigned int* deviceRangeBits = nullptr;
  unsigned int* deviceHistogram = nullptr;
  unsigned int* devicePercentileBits = nullptr;
  unsigned int* deviceFinalRangeBits = nullptr;
  cudaError_t err = cudaMalloc(&deviceRangeBits, sizeof(initRangeBits));
  if (err == cudaSuccess) err = cudaMalloc(&deviceHistogram, static_cast<size_t>(kRangeHistogramBins) * sizeof(unsigned int));
  if (err == cudaSuccess) err = cudaMalloc(&devicePercentileBits, 2u * sizeof(unsigned int));
  if (err == cudaSuccess) err = cudaMalloc(&deviceFinalRangeBits, 3u * sizeof(unsigned int));
  if (err == cudaSuccess) err = cudaMemcpy(deviceRangeBits, initRangeBits, sizeof(initRangeBits), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) err = cudaMemset(deviceHistogram, 0, static_cast<size_t>(kRangeHistogramBins) * sizeof(unsigned int));
  const unsigned int initPercentiles[2] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
  };
  if (err == cudaSuccess) {
    err = cudaMemcpy(devicePercentileBits, initPercentiles, sizeof(initPercentiles), cudaMemcpyHostToDevice);
  }
  auto freeTemps = [&]() {
    if (deviceRangeBits) cudaFree(deviceRangeBits);
    if (deviceHistogram) cudaFree(deviceHistogram);
    if (devicePercentileBits) cudaFree(devicePercentileBits);
    if (deviceFinalRangeBits) cudaFree(deviceFinalRangeBits);
  };
  if (err != cudaSuccess || !deviceRangeBits || !deviceHistogram ||
      !devicePercentileBits || !deviceFinalRangeBits) {
    freeTemps();
    if (error) *error = std::string("Failed to allocate CUDA resident scope auto-range buffers: ") + errorString(err);
    return false;
  }

  ScopeRangeRequest kernelRange = rangeRequest;
  kernelRange.pointCount = static_cast<int>(pointCount);
  kernelRange.lumaMethod = std::clamp(kernelRange.lumaMethod, 0, 3);
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  rasterScopeRangeKernel<<<blocks, threads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, kernelRange, deviceRangeBits);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    rasterScopeRangeHistogramKernel<<<blocks, threads>>>(static_cast<const unsigned char*>(sourceDeviceBytes),
                                                         rasterUniforms,
                                                         kernelRange,
                                                         deviceRangeBits,
                                                         kRangeHistogramBins,
                                                         deviceHistogram);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    scopeRangeHistogramPercentileKernel<<<1, 1>>>(deviceHistogram,
                                                  kRangeHistogramBins,
                                                  deviceRangeBits,
                                                  devicePercentileBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    scopeRangeFinalizeKernel<<<1, 1>>>(devicePercentileBits,
                                       deviceRangeBits,
                                       previousRangeValid,
                                       previousRangeMin,
                                       previousRangeMax,
                                       deviceFinalRangeBits);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    freeTemps();
    if (error) *error = std::string("CUDA resident scope auto-range failed: ") + errorString(err);
    return false;
  }
  cudaFree(deviceRangeBits);
  cudaFree(deviceHistogram);
  cudaFree(devicePercentileBits);
  *outFinalRangeBits = deviceFinalRangeBits;
  return true;
}

bool readScopeFinalRangeDevice(const unsigned int* deviceFinalRangeBits,
                               ScopeRangeResult* outRange,
                               std::string* error) {
  if (!outRange || !deviceFinalRangeBits) return true;
  unsigned int packed[3] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
      0u,
  };
  cudaError_t err = cudaMemcpy(packed, deviceFinalRangeBits, sizeof(packed), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA resident scope auto-range metadata: ") + errorString(err);
    return false;
  }
  outRange->minValue = floatFromOrderedUint(packed[0]);
  outRange->maxValue = floatFromOrderedUint(packed[1]);
  outRange->validCount = packed[2];
  return true;
}

bool buildWaveformScopePointsFromRasterSourceDevice(InputCache* cache,
                                                    const RasterSourceRequest& rasterRequest,
                                                    const WaveformScopePointRequest& pointRequest,
                                                    const void* sourceDeviceBytes,
                                                    size_t sourceByteCount,
                                                    ScopeRangeResult* outRange,
                                                    unsigned long long serial,
                                                    std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA waveform scope cache has no GL buffers.";
    return false;
  }
  ScopeDensityRequest normalScope = pointRequest.density;
  normalScope.waveform = 1;
  normalScope.excludeOverflow = 1;
  normalScope.onlyOverflow = 0;
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(normalScope.width, 1);
  const int height = std::max(normalScope.height, 1);
  const int channelCount = std::max(1, normalScope.channelCount);
  const size_t binCount = static_cast<size_t>(width) * static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  const size_t layerCount = pointRequest.showOverflow != 0 ? 2u : 1u;
  const size_t outputPointCount = binCount * layerCount;
  const size_t pixelCount = static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) *
                            static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  const size_t expectedBytes =
      pixelCount * 4u * (rasterRequest.pixelFormat == 1 ? sizeof(float) : sizeof(unsigned short));
  if (pointCount <= 0 || outputPointCount == 0u ||
      outputPointCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !sourceDeviceBytes || sourceByteCount < expectedBytes ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0) {
    if (error) *error = "Invalid CUDA resident waveform scope request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA waveform scope cache.";
    return false;
  }
  if (!ensureRegistered(cache->verts, cache->colors, outputPointCount, impl, &localError)) {
    if (error) *error = localError;
    return false;
  }
  if (!ensureBoundsCapacity(impl, &localError)) {
    if (error) *error = localError;
    return false;
  }

  unsigned int* normalDensity = nullptr;
  unsigned int* overflowDensity = nullptr;
  cudaError_t err = cudaMalloc(&normalDensity, binCount * sizeof(unsigned int));
  if (err == cudaSuccess && pointRequest.showOverflow != 0) {
    err = cudaMalloc(&overflowDensity, binCount * sizeof(unsigned int));
  }
  if (err == cudaSuccess) err = cudaMemset(normalDensity, 0, binCount * sizeof(unsigned int));
  if (err == cudaSuccess && overflowDensity) {
    err = cudaMemset(overflowDensity, 0, binCount * sizeof(unsigned int));
  }
  if (err != cudaSuccess || !normalDensity || (pointRequest.showOverflow != 0 && !overflowDensity)) {
    if (normalDensity) cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (error) *error = std::string("Failed to allocate CUDA resident waveform density: ") + errorString(err);
    return false;
  }

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = pointCount;
  normalScope.pointCount = pointCount;
  normalScope.width = width;
  normalScope.height = height;
  normalScope.channelCount = channelCount;
  normalScope.lumaMethod = std::clamp(normalScope.lumaMethod, 0, 3);
  unsigned int* deviceAutoRangeBits = nullptr;
  if (pointRequest.useAutoRange != 0) {
    if (!buildScopeAutoRangeDevice(rasterUniforms,
                                   pointRequest.autoRange,
                                   sourceDeviceBytes,
                                   static_cast<unsigned int>(pointCount),
                                   pointRequest.previousRangeValid,
                                   pointRequest.previousRangeMin,
                                   pointRequest.previousRangeMax,
                                   &deviceAutoRangeBits,
                                   error)) {
      cudaFree(normalDensity);
      if (overflowDensity) cudaFree(overflowDensity);
      return false;
    }
  }
  const unsigned int threads = 256u;
  const unsigned int pointBlocks =
      static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  rasterScopeDensityKernel<<<pointBlocks, threads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, normalScope, deviceAutoRangeBits, normalDensity);
  err = cudaGetLastError();
  if (err == cudaSuccess && overflowDensity) {
    ScopeDensityRequest overflowScope = normalScope;
    overflowScope.excludeOverflow = 0;
    overflowScope.onlyOverflow = 1;
    rasterScopeDensityKernel<<<pointBlocks, threads>>>(
        static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, overflowScope, deviceAutoRangeBits, overflowDensity);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
    cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (error) *error = std::string("CUDA resident waveform density failed: ") + errorString(err);
    return false;
  }

  err = cudaMemset(impl->deviceBounds, 0, sizeof(unsigned int));
  const unsigned int binBlocks =
      static_cast<unsigned int>((binCount + threads - 1u) / threads);
  if (err == cudaSuccess) {
    scopeDensityMaxUintKernel<<<binBlocks, threads>>>(normalDensity,
                                                      static_cast<unsigned int>(binCount),
                                                      impl->deviceBounds);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && overflowDensity) {
    scopeDensityMaxUintKernel<<<binBlocks, threads>>>(overflowDensity,
                                                      static_cast<unsigned int>(binCount),
                                                      impl->deviceBounds);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
    cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (error) *error = std::string("CUDA resident waveform max-density failed: ") + errorString(err);
    return false;
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
    cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (error) *error = std::string("Failed to map CUDA resident waveform buffers: ") + errorString(err);
    return false;
  }
  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts), &vertsBytes, impl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors),
                                               &colorsBytes,
                                               impl->colorsResource);
  }
  if (err != cudaSuccess ||
      vertsBytes < outputPointCount * 3u * sizeof(float) ||
      colorsBytes < outputPointCount * 4u * sizeof(float)) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
    cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (error) *error = err != cudaSuccess
                            ? std::string("Failed to access CUDA resident waveform buffers: ") + errorString(err)
                            : std::string("CUDA resident waveform GL buffers are undersized.");
    return false;
  }

  WaveformScopePointRequest kernelRequest = pointRequest;
  kernelRequest.density = normalScope;
  kernelRequest.showOverflow = pointRequest.showOverflow != 0 ? 1 : 0;
  kernelRequest.highlightOverflow = pointRequest.highlightOverflow != 0 ? 1 : 0;
  const unsigned int outputBlocks =
      static_cast<unsigned int>((outputPointCount + threads - 1u) / threads);
  waveformScopeDensityToPointsKernel<<<outputBlocks, threads>>>(devVerts,
                                                                devColors,
                                                                normalDensity,
                                                                overflowDensity,
                                                                static_cast<unsigned int>(binCount),
                                                                impl->deviceBounds,
                                                                kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err == cudaSuccess && deviceAutoRangeBits) {
    err = readScopeFinalRangeDevice(deviceAutoRangeBits, outRange, error) ? cudaSuccess : cudaErrorUnknown;
  }
  if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
  cudaFree(normalDensity);
  if (overflowDensity) cudaFree(overflowDensity);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA resident waveform point build failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(outputPointCount);
  cache->available = true;
  cache->hasFitBounds = true;
  cache->fitMin[0] = -0.82f;
  cache->fitMin[1] = -0.88f;
  cache->fitMin[2] = 0.0f;
  cache->fitMax[0] = 0.96f;
  cache->fitMax[1] = 0.88f;
  cache->fitMax[2] = 0.0f;
  return true;
}

bool buildHistogramScopeGeometryFromRasterSourceDevice(ScopeGeometryCache* cache,
                                                       const RasterSourceRequest& rasterRequest,
                                                       const HistogramScopeGeometryRequest& geometryRequest,
                                                       const void* sourceDeviceBytes,
                                                       size_t sourceByteCount,
                                                       ScopeRangeResult* outRange,
                                                       unsigned long long serial,
                                                       std::string* error) {
  if (!cache || cache->lineVerts == 0 || cache->lineColors == 0 ||
      cache->fillVerts == 0 || cache->fillColors == 0) {
    if (error) *error = "CUDA histogram scope cache has no GL buffers.";
    return false;
  }
  ScopeDensityRequest normalScope = geometryRequest.density;
  normalScope.waveform = 0;
  normalScope.excludeOverflow = 1;
  normalScope.onlyOverflow = 0;
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(normalScope.width, 1);
  const int height = std::max(normalScope.height, 1);
  const int channelCount = std::max(1, normalScope.channelCount);
  const size_t binCount = static_cast<size_t>(width) * static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  const size_t segmentCount = static_cast<size_t>(std::max(width - 1, 0)) *
                              static_cast<size_t>(channelCount);
  const size_t layerCount = geometryRequest.showOverflow != 0 ? 2u : 1u;
  const size_t lineVertexCount = segmentCount * layerCount * 2u;
  const size_t fillVertexCount = segmentCount * layerCount * 6u;
  const size_t pixelCount = static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) *
                            static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  const size_t expectedBytes =
      pixelCount * 4u * (rasterRequest.pixelFormat == 1 ? sizeof(float) : sizeof(unsigned short));
  if (pointCount <= 0 || width != 512 || height != 1 || channelCount > 3 ||
      binCount == 0u || segmentCount == 0u ||
      lineVertexCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      fillVertexCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !sourceDeviceBytes || sourceByteCount < expectedBytes ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0) {
    if (error) *error = "Invalid CUDA resident histogram scope request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  ScopeGeometryCacheImpl* impl = ensureScopeGeometryImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA histogram scope cache.";
    return false;
  }
  if (!ensureScopeGeometryRegistered(cache->lineVerts,
                                     cache->lineColors,
                                     cache->fillVerts,
                                     cache->fillColors,
                                     lineVertexCount,
                                     fillVertexCount,
                                     impl,
                                     &localError)) {
    if (error) *error = localError;
    return false;
  }

  unsigned int* normalDensity = nullptr;
  unsigned int* overflowDensity = nullptr;
  float* smoothedNormal = nullptr;
  float* smoothedOverflow = nullptr;
  unsigned int* maxDensityBits = nullptr;
  unsigned int* deviceAutoRangeBits = nullptr;
  cudaError_t err = cudaMalloc(&normalDensity, binCount * sizeof(unsigned int));
  if (err == cudaSuccess && geometryRequest.showOverflow != 0) {
    err = cudaMalloc(&overflowDensity, binCount * sizeof(unsigned int));
  }
  if (err == cudaSuccess) err = cudaMalloc(&smoothedNormal, binCount * sizeof(float));
  if (err == cudaSuccess && geometryRequest.showOverflow != 0) {
    err = cudaMalloc(&smoothedOverflow, binCount * sizeof(float));
  }
  if (err == cudaSuccess) err = cudaMalloc(&maxDensityBits, sizeof(unsigned int));
  if (err == cudaSuccess) err = cudaMemset(normalDensity, 0, binCount * sizeof(unsigned int));
  if (err == cudaSuccess && overflowDensity) {
    err = cudaMemset(overflowDensity, 0, binCount * sizeof(unsigned int));
  }
  const unsigned int hostMaxInit = orderedUintFromFloatHost(1.0f);
  if (err == cudaSuccess) {
    err = cudaMemcpy(maxDensityBits, &hostMaxInit, sizeof(hostMaxInit), cudaMemcpyHostToDevice);
  }
  auto freeTemps = [&]() {
    if (normalDensity) cudaFree(normalDensity);
    if (overflowDensity) cudaFree(overflowDensity);
    if (smoothedNormal) cudaFree(smoothedNormal);
    if (smoothedOverflow) cudaFree(smoothedOverflow);
    if (maxDensityBits) cudaFree(maxDensityBits);
    if (deviceAutoRangeBits) cudaFree(deviceAutoRangeBits);
  };
  if (err != cudaSuccess || !normalDensity || !smoothedNormal || !maxDensityBits ||
      (geometryRequest.showOverflow != 0 && (!overflowDensity || !smoothedOverflow))) {
    freeTemps();
    if (error) *error = std::string("Failed to allocate CUDA resident histogram buffers: ") + errorString(err);
    return false;
  }

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = pointCount;
  normalScope.pointCount = pointCount;
  normalScope.width = width;
  normalScope.height = height;
  normalScope.channelCount = channelCount;
  normalScope.lumaMethod = std::clamp(normalScope.lumaMethod, 0, 3);
  if (geometryRequest.useAutoRange != 0) {
    if (!buildScopeAutoRangeDevice(rasterUniforms,
                                   geometryRequest.autoRange,
                                   sourceDeviceBytes,
                                   static_cast<unsigned int>(pointCount),
                                   geometryRequest.previousRangeValid,
                                   geometryRequest.previousRangeMin,
                                   geometryRequest.previousRangeMax,
                                   &deviceAutoRangeBits,
                                   error)) {
      freeTemps();
      return false;
    }
  }

  const unsigned int threads = 256u;
  const unsigned int pointBlocks =
      static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  rasterScopeDensityKernel<<<pointBlocks, threads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, normalScope, deviceAutoRangeBits, normalDensity);
  err = cudaGetLastError();
  if (err == cudaSuccess && overflowDensity) {
    ScopeDensityRequest overflowScope = normalScope;
    overflowScope.excludeOverflow = 0;
    overflowScope.onlyOverflow = 1;
    rasterScopeDensityKernel<<<pointBlocks, threads>>>(
        static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, overflowScope, deviceAutoRangeBits, overflowDensity);
    err = cudaGetLastError();
  }
  const unsigned int binBlocks = static_cast<unsigned int>((binCount + threads - 1u) / threads);
  if (err == cudaSuccess) {
    histogramSmoothDensityKernel<<<binBlocks, threads>>>(normalDensity,
                                                         smoothedNormal,
                                                         width,
                                                         channelCount);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess && overflowDensity && smoothedOverflow) {
    histogramSmoothDensityKernel<<<binBlocks, threads>>>(overflowDensity,
                                                         smoothedOverflow,
                                                         width,
                                                         channelCount);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    scopeDensityMaxFloatKernel<<<binBlocks, threads>>>(smoothedNormal,
                                                       static_cast<unsigned int>(binCount),
                                                       maxDensityBits);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    freeTemps();
    if (error) *error = std::string("CUDA resident histogram density build failed: ") + errorString(err);
    return false;
  }

  std::array<cudaGraphicsResource*, 4> resources = {
      impl->lineVertsResource,
      impl->lineColorsResource,
      impl->fillVertsResource,
      impl->fillColorsResource
  };
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    freeTemps();
    if (error) *error = std::string("Failed to map CUDA resident histogram buffers: ") + errorString(err);
    return false;
  }
  float* devLineVerts = nullptr;
  float* devLineColors = nullptr;
  float* devFillVerts = nullptr;
  float* devFillColors = nullptr;
  size_t lineVertsBytes = 0;
  size_t lineColorsBytes = 0;
  size_t fillVertsBytes = 0;
  size_t fillColorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devLineVerts),
                                             &lineVertsBytes,
                                             impl->lineVertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devLineColors),
                                               &lineColorsBytes,
                                               impl->lineColorsResource);
  }
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devFillVerts),
                                               &fillVertsBytes,
                                               impl->fillVertsResource);
  }
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devFillColors),
                                               &fillColorsBytes,
                                               impl->fillColorsResource);
  }
  if (err != cudaSuccess ||
      lineVertsBytes < lineVertexCount * 3u * sizeof(float) ||
      lineColorsBytes < lineVertexCount * 4u * sizeof(float) ||
      fillVertsBytes < fillVertexCount * 3u * sizeof(float) ||
      fillColorsBytes < fillVertexCount * 4u * sizeof(float)) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    freeTemps();
    if (error) {
      *error = err != cudaSuccess
                   ? std::string("Failed to access CUDA resident histogram buffers: ") + errorString(err)
                   : std::string("CUDA resident histogram GL buffers are undersized.");
    }
    return false;
  }

  HistogramScopeGeometryRequest kernelRequest = geometryRequest;
  kernelRequest.density = normalScope;
  kernelRequest.showOverflow = geometryRequest.showOverflow != 0 ? 1 : 0;
  kernelRequest.highlightOverflow = geometryRequest.highlightOverflow != 0 ? 1 : 0;
  const unsigned int geometryItems = static_cast<unsigned int>(segmentCount * layerCount);
  const unsigned int geometryBlocks =
      static_cast<unsigned int>((static_cast<size_t>(geometryItems) + threads - 1u) / threads);
  histogramScopeDensityToGeometryKernel<<<geometryBlocks, threads>>>(devLineVerts,
                                                                     devLineColors,
                                                                     devFillVerts,
                                                                     devFillColors,
                                                                     smoothedNormal,
                                                                     smoothedOverflow,
                                                                     static_cast<unsigned int>(segmentCount),
                                                                     maxDensityBits,
                                                                     kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err == cudaSuccess && deviceAutoRangeBits) {
    err = readScopeFinalRangeDevice(deviceAutoRangeBits, outRange, error) ? cudaSuccess : cudaErrorUnknown;
  }
  freeTemps();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA resident histogram geometry build failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->lineVertexCount = static_cast<int>(lineVertexCount);
  cache->fillVertexCount = static_cast<int>(fillVertexCount);
  cache->available = true;
  return true;
}

bool buildGlossProjectionFromResidentField(InputCache* fieldCache,
                                           InputSampleCache* projectionCache,
                                           const GlossProjectionRequest& request,
                                           unsigned long long serial,
                                           std::string* error) {
  if (!fieldCache || !projectionCache || projectionCache->verts == 0 || projectionCache->colors == 0) {
    if (error) *error = "CUDA gloss projection cache has no GL buffers.";
    return false;
  }
  const int gridWidth = std::max(request.gridWidth, 1);
  const int gridHeight = std::max(request.gridHeight, 1);
  const size_t pointCount = static_cast<size_t>(gridWidth) * static_cast<size_t>(gridHeight);
  if (pointCount == 0u || pointCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
    if (error) *error = "Invalid CUDA gloss projection request.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* fieldImpl = reinterpret_cast<CacheImpl*>(fieldCache->internal);
  if (!fieldImpl || !fieldImpl->deviceFieldWorkspace ||
      fieldImpl->fieldWorkspaceFloats < pointCount * kGlossFieldWorkspaceArrayCount) {
    if (error) *error = "CUDA gloss field workspace is not resident.";
    return false;
  }
  SampleCacheImpl* projectionImpl = ensureSampleImpl(projectionCache);
  if (!projectionImpl) {
    if (error) *error = "Failed to allocate CUDA gloss projection cache.";
    return false;
  }
  if (!ensureSampleRegistered(projectionCache->verts,
                              projectionCache->colors,
                              pointCount,
                              projectionImpl,
                              &localError)) {
    if (error) *error = localError;
    return false;
  }

  std::array<cudaGraphicsResource*, 2> resources = {
      projectionImpl->vertsResource,
      projectionImpl->colorsResource
  };
  cudaError_t err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA gloss projection buffers: ") + errorString(err);
    return false;
  }
  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts),
                                             &vertsBytes,
                                             projectionImpl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors),
                                               &colorsBytes,
                                               projectionImpl->colorsResource);
  }
  if (err != cudaSuccess ||
      vertsBytes < pointCount * 3u * sizeof(float) ||
      colorsBytes < pointCount * 4u * sizeof(float)) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) {
      *error = err != cudaSuccess
                   ? std::string("Failed to access CUDA gloss projection buffers: ") + errorString(err)
                   : std::string("CUDA gloss projection GL buffers are undersized.");
    }
    return false;
  }

  GlossProjectionRequest kernelRequest = request;
  kernelRequest.gridWidth = gridWidth;
  kernelRequest.gridHeight = gridHeight;
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  glossProjectionFromFieldKernel<<<blocks, threads>>>(devVerts,
                                                      devColors,
                                                      fieldImpl->deviceFieldWorkspace,
                                                      kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss projection build failed: ") + errorString(err);
    return false;
  }

  projectionCache->builtSerial = serial;
  projectionCache->pointCount = static_cast<int>(pointCount);
  projectionCache->available = true;
  return true;
}

bool buildGlossProjectionFromRasterSourceDevice(InputCache* fieldCache,
                                                InputSampleCache* projectionCache,
                                                const RasterSourceRequest& rasterRequest,
                                                const GlossProjectionRequest& projectionRequest,
                                                const void* sourceDeviceBytes,
                                                size_t sourceByteCount,
                                                unsigned long long serial,
                                                std::string* error) {
  if (!fieldCache || !projectionCache || projectionCache->verts == 0 || projectionCache->colors == 0) {
    if (error) *error = "CUDA gloss raster projection cache has no GL buffers.";
    return false;
  }
  const int gridWidth = std::max(projectionRequest.gridWidth, 1);
  const int gridHeight = std::max(projectionRequest.gridHeight, 1);
  const size_t cellCount = static_cast<size_t>(gridWidth) * static_cast<size_t>(gridHeight);
  const size_t pointCount = static_cast<size_t>(std::max(rasterRequest.basePointCount, 0));
  if (cellCount == 0u || pointCount == 0u ||
      pointCount > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      !sourceDeviceBytes || sourceByteCount == 0) {
    if (error) *error = "Invalid CUDA gloss raster projection request.";
    return false;
  }
  if (rasterRequest.sourceBytesAreDevice == 0) {
    if (error) *error = "CUDA gloss raster projection requires resident device source.";
    return false;
  }
  const size_t expectedRowBytes =
      static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) * 4u * sizeof(float);
  const size_t expectedBytes = expectedRowBytes * static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  if (rasterRequest.pixelFormat != 1 || sourceByteCount < expectedBytes) {
    if (error) *error = "CUDA gloss raster projection source must be resident rgba32f.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* fieldImpl = reinterpret_cast<CacheImpl*>(fieldCache->internal);
  if (!fieldImpl || !fieldImpl->deviceFieldWorkspace ||
      fieldImpl->fieldWorkspaceFloats < cellCount * kGlossFieldWorkspaceArrayCount) {
    if (error) *error = "CUDA gloss field workspace is not resident.";
    return false;
  }
  SampleCacheImpl* projectionImpl = ensureSampleImpl(projectionCache);
  if (!projectionImpl) {
    if (error) *error = "Failed to allocate CUDA gloss raster projection cache.";
    return false;
  }
  if (!ensureSampleRegistered(projectionCache->verts,
                              projectionCache->colors,
                              pointCount,
                              projectionImpl,
                              &localError)) {
    if (error) *error = localError;
    return false;
  }

  std::array<cudaGraphicsResource*, 2> resources = {
      projectionImpl->vertsResource,
      projectionImpl->colorsResource
  };
  cudaError_t err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA gloss raster projection buffers: ") + errorString(err);
    return false;
  }
  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts),
                                             &vertsBytes,
                                             projectionImpl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors),
                                               &colorsBytes,
                                               projectionImpl->colorsResource);
  }
  if (err != cudaSuccess ||
      vertsBytes < pointCount * 3u * sizeof(float) ||
      colorsBytes < pointCount * 4u * sizeof(float)) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) {
      *error = err != cudaSuccess
                   ? std::string("Failed to access CUDA gloss raster projection buffers: ") + errorString(err)
                   : std::string("CUDA gloss raster projection GL buffers are undersized.");
    }
    return false;
  }

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = static_cast<int>(pointCount);
  rasterUniforms.basePointCount = static_cast<int>(pointCount);
  GlossProjectionRequest kernelRequest = projectionRequest;
  kernelRequest.gridWidth = gridWidth;
  kernelRequest.gridHeight = gridHeight;
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  glossProjectionFromRasterSourceKernel<<<blocks, threads>>>(devVerts,
                                                             devColors,
                                                             static_cast<const unsigned char*>(sourceDeviceBytes),
                                                             rasterUniforms,
                                                             fieldImpl->deviceFieldWorkspace,
                                                             kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss raster projection build failed: ") + errorString(err);
    return false;
  }

  projectionCache->builtSerial = serial;
  projectionCache->pointCount = static_cast<int>(pointCount);
  projectionCache->available = true;
  return true;
}

bool buildGlossField2DGeometryFromResidentField(InputCache* fieldCache,
                                                ScopeGeometryCache* geometryCache,
                                                const GlossField2DGeometryRequest& request,
                                                unsigned long long serial,
                                                std::string* error) {
  if (!fieldCache || !geometryCache || geometryCache->fillVerts == 0 || geometryCache->fillColors == 0 ||
      geometryCache->lineVerts == 0 || geometryCache->lineColors == 0) {
    if (error) *error = "CUDA gloss field 2D cache has no GL buffers.";
    return false;
  }
  const int gridWidth = std::max(request.gridWidth, 1);
  const int gridHeight = std::max(request.gridHeight, 1);
  const size_t cellCount = static_cast<size_t>(gridWidth) * static_cast<size_t>(gridHeight);
  const size_t lineVertexCount = 2u;
  const size_t fillVertexCount = cellCount * 12u;
  if (cellCount == 0u ||
      fillVertexCount == 0u ||
      fillVertexCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
    if (error) *error = "Invalid CUDA gloss field 2D request.";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* fieldImpl = reinterpret_cast<CacheImpl*>(fieldCache->internal);
  if (!fieldImpl || !fieldImpl->deviceFieldWorkspace ||
      fieldImpl->fieldWorkspaceFloats < cellCount * kGlossFieldWorkspaceArrayCount) {
    if (error) *error = "CUDA gloss field workspace is not resident.";
    return false;
  }
  ScopeGeometryCacheImpl* geometryImpl = ensureScopeGeometryImpl(geometryCache);
  if (!geometryImpl) {
    if (error) *error = "Failed to allocate CUDA gloss field 2D geometry cache.";
    return false;
  }
  if (!ensureScopeGeometryRegistered(geometryCache->lineVerts,
                                     geometryCache->lineColors,
                                     geometryCache->fillVerts,
                                     geometryCache->fillColors,
                                     lineVertexCount,
                                     fillVertexCount,
                                     geometryImpl,
                                     &localError)) {
    if (error) *error = localError;
    return false;
  }

  std::array<cudaGraphicsResource*, 4> resources = {
      geometryImpl->lineVertsResource,
      geometryImpl->lineColorsResource,
      geometryImpl->fillVertsResource,
      geometryImpl->fillColorsResource
  };
  cudaError_t err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA gloss field 2D buffers: ") + errorString(err);
    return false;
  }
  float* devLineVerts = nullptr;
  float* devLineColors = nullptr;
  float* devFillVerts = nullptr;
  float* devFillColors = nullptr;
  size_t lineVertsBytes = 0;
  size_t lineColorsBytes = 0;
  size_t fillVertsBytes = 0;
  size_t fillColorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devLineVerts),
                                             &lineVertsBytes,
                                             geometryImpl->lineVertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devLineColors),
                                               &lineColorsBytes,
                                               geometryImpl->lineColorsResource);
  }
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devFillVerts),
                                               &fillVertsBytes,
                                               geometryImpl->fillVertsResource);
  }
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devFillColors),
                                               &fillColorsBytes,
                                               geometryImpl->fillColorsResource);
  }
  if (err != cudaSuccess ||
      lineVertsBytes < lineVertexCount * 3u * sizeof(float) ||
      lineColorsBytes < lineVertexCount * 4u * sizeof(float) ||
      fillVertsBytes < fillVertexCount * 3u * sizeof(float) ||
      fillColorsBytes < fillVertexCount * 4u * sizeof(float)) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) {
      *error = err != cudaSuccess
                   ? std::string("Failed to access CUDA gloss field 2D buffers: ") + errorString(err)
                   : std::string("CUDA gloss field 2D GL buffers are undersized.");
    }
    return false;
  }

  GlossField2DGeometryRequest kernelRequest = request;
  kernelRequest.gridWidth = gridWidth;
  kernelRequest.gridHeight = gridHeight;
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((cellCount + threads - 1u) / threads);
  glossField2DGeometryFromFieldKernel<<<blocks, threads>>>(devLineVerts,
                                                           devLineColors,
                                                           devFillVerts,
                                                           devFillColors,
                                                           fieldImpl->deviceFieldWorkspace,
                                                           kernelRequest);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss field 2D geometry build failed: ") + errorString(err);
    return false;
  }

  geometryCache->builtSerial = serial;
  geometryCache->lineVertexCount = static_cast<int>(lineVertexCount);
  geometryCache->fillVertexCount = static_cast<int>(fillVertexCount);
  geometryCache->available = true;
  return true;
}

bool buildScopeRangeFromRasterSourceDevice(const RasterSourceRequest& rasterRequest,
                                           const ScopeRangeRequest& rangeRequest,
                                           const void* sourceDeviceBytes,
                                           size_t sourceByteCount,
                                           ScopeRangeResult* outRange,
                                           std::string* error) {
  if (!outRange) {
    if (error) *error = "Missing CUDA raster scope-range output.";
    return false;
  }
  *outRange = ScopeRangeResult{};
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const size_t pixelCount = static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) *
                            static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  const size_t expectedBytes =
      pixelCount * 4u * (rasterRequest.pixelFormat == 1 ? sizeof(float) : sizeof(unsigned short));
  if (pointCount <= 0 || !sourceDeviceBytes || sourceByteCount < expectedBytes ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0) {
    if (error) *error = "Invalid CUDA raster source scope-range request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  unsigned int* deviceRangeBits = nullptr;
  const unsigned int initRangeBits[3] = {
      orderedUintFromFloatHost(std::numeric_limits<float>::infinity()),
      orderedUintFromFloatHost(-std::numeric_limits<float>::infinity()),
      0u,
  };
  cudaError_t err = cudaMalloc(&deviceRangeBits, sizeof(initRangeBits));
  if (err == cudaSuccess) {
    err = cudaMemcpy(deviceRangeBits, initRangeBits, sizeof(initRangeBits), cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess || !deviceRangeBits) {
    if (deviceRangeBits) cudaFree(deviceRangeBits);
    if (error) *error = std::string("Failed to allocate CUDA raster scope-range reduction: ") + errorString(err);
    return false;
  }

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = pointCount;
  ScopeRangeRequest kernelRange = rangeRequest;
  kernelRange.pointCount = pointCount;
  kernelRange.lumaMethod = std::clamp(kernelRange.lumaMethod, 0, 3);

  const unsigned int threads = 256u;
  const unsigned int blocks =
      static_cast<unsigned int>((static_cast<size_t>(pointCount) + threads - 1u) / threads);
  rasterScopeRangeKernel<<<blocks, threads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes), rasterUniforms, kernelRange, deviceRangeBits);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(deviceRangeBits);
    if (error) *error = std::string("CUDA raster source scope-range kernel failed: ") + errorString(err);
    return false;
  }

  unsigned int packed[3] = {};
  err = cudaMemcpy(packed, deviceRangeBits, sizeof(packed), cudaMemcpyDeviceToHost);
  cudaFree(deviceRangeBits);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA raster source scope range: ") + errorString(err);
    return false;
  }
  if (packed[2] == 0u) {
    if (error) *error = "CUDA raster source scope range found no visible values.";
    return false;
  }
  const float minValue = floatFromOrderedUint(packed[0]);
  const float maxValue = floatFromOrderedUint(packed[1]);
  if (!(maxValue > minValue + 1.0e-7f)) {
    outRange->minValue = minValue;
    outRange->maxValue = maxValue;
    outRange->validCount = packed[2];
    return true;
  }

  constexpr unsigned int kRangeHistogramBins = 2048u;
  unsigned int* deviceHistogram = nullptr;
  unsigned int* devicePercentileBits = nullptr;
  err = cudaMalloc(&deviceHistogram, static_cast<size_t>(kRangeHistogramBins) * sizeof(unsigned int));
  if (err == cudaSuccess) {
    err = cudaMemset(deviceHistogram, 0, static_cast<size_t>(kRangeHistogramBins) * sizeof(unsigned int));
  }
  if (err == cudaSuccess) {
    err = cudaMalloc(&devicePercentileBits, 2u * sizeof(unsigned int));
  }
  if (err != cudaSuccess || !deviceHistogram || !devicePercentileBits) {
    if (deviceHistogram) cudaFree(deviceHistogram);
    if (devicePercentileBits) cudaFree(devicePercentileBits);
    if (error) *error = std::string("Failed to allocate CUDA raster scope-range histogram: ") + errorString(err);
    return false;
  }

  rasterScopeRangeHistogramKernel<<<blocks, threads>>>(static_cast<const unsigned char*>(sourceDeviceBytes),
                                                       rasterUniforms,
                                                       kernelRange,
                                                       deviceRangeBits,
                                                       kRangeHistogramBins,
                                                       deviceHistogram);
  err = cudaGetLastError();
  if (err == cudaSuccess) {
    scopeRangeHistogramPercentileKernel<<<1, 1>>>(deviceHistogram,
                                                  kRangeHistogramBins,
                                                  deviceRangeBits,
                                                  devicePercentileBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  unsigned int percentileBits[2] = {
      orderedUintFromFloatHost(minValue),
      orderedUintFromFloatHost(maxValue),
  };
  if (err == cudaSuccess) {
    err = cudaMemcpy(percentileBits, devicePercentileBits, sizeof(percentileBits), cudaMemcpyDeviceToHost);
  }
  cudaFree(deviceHistogram);
  cudaFree(devicePercentileBits);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster source scope-range histogram failed: ") + errorString(err);
    return false;
  }

  outRange->minValue = floatFromOrderedUint(percentileBits[0]);
  outRange->maxValue = floatFromOrderedUint(percentileBits[1]);
  outRange->validCount = packed[2];
  return true;
}

bool runGlossFieldResearchCandidateSolvers(int gridWidth,
                                           int gridHeight,
                                           int neighborhoodChoice,
                                           float* workspace,
                                           float* occupancy,
                                           float* meanR,
                                           float* meanG,
                                           float* meanB,
                                           float* carrierY,
                                           float* carrierMax,
                                           float* carrierMin,
                                           float* occupancySupport,
                                           float* body,
                                           float* positive,
                                           float* negative,
                                           float* boundary,
                                           float* congruence,
                                           float* confidence,
                                           float* signal,
                                           float* body2,
                                           float* positive2,
                                           float* negative2,
                                           float* boundary2,
                                           float* congruence2,
                                           float* confidence2,
                                           float* signal2,
                                           unsigned int* reductionBits,
                                           unsigned int cellBlocks,
                                           unsigned int cellThreads,
                                           const char* label,
                                           std::string* error) {
  if (!workspace || !occupancy || !meanR || !meanG || !meanB || !carrierY || !carrierMax || !carrierMin ||
      !occupancySupport || !body || !positive || !negative || !boundary || !congruence || !confidence ||
      !signal || !body2 || !positive2 || !negative2 || !boundary2 || !congruence2 || !confidence2 ||
      !signal2 || !reductionBits) {
    if (error) *error = "Missing CUDA gloss-field research solver buffers.";
    return false;
  }
  const int cellCount = std::max(gridWidth * gridHeight, 0);
  if (cellCount <= 0) {
    if (error) *error = "Invalid CUDA gloss-field research solver dimensions.";
    return false;
  }
  const int analysisRadius = neighborhoodChoice <= 0 ? 3 : (neighborhoodChoice >= 2 ? 10 : 6);
  float* carrierHybrid = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 0u);
  float* viewerBody = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 1u);
  float* bodyCore = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 2u);
  float* bodyContext = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 3u);
  float* positiveRaw = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 4u);
  float* negativeRaw = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 5u);
  float* confidenceRaw = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 6u);
  float* agreementRaw = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 7u);
  float* retinexBody = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 8u);
  float* dogLow = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 9u);
  float* tempA = workspace + static_cast<size_t>(cellCount) * (kGlossFieldScratchBase + 10u);

  auto fail = [&](const std::string& stage, cudaError_t err) -> bool {
    if (error) {
      *error = std::string("CUDA ") + (label ? label : "gloss-field") + " " + stage + " failed: " + errorString(err);
    }
    return false;
  };
  auto check = [&](const std::string& stage) -> bool {
    cudaError_t err = cudaGetLastError();
    return err == cudaSuccess ? true : fail(stage, err);
  };
  auto blurRepeated = [&](float* src, float* dst, int passes, const char* stage) -> bool {
    cudaError_t err = cudaMemcpy(dst, src, static_cast<size_t>(cellCount) * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) return fail(std::string(stage) + " seed copy", err);
    const int clampedPasses = std::clamp(passes, 1, 24);
    for (int pass = 0; pass < clampedPasses; ++pass) {
      glossFieldBlurKernel<<<cellBlocks, cellThreads>>>(gridWidth, gridHeight, dst, tempA);
      if (!check(std::string(stage) + " blur")) return false;
      err = cudaMemcpy(dst, tempA, static_cast<size_t>(cellCount) * sizeof(float), cudaMemcpyDeviceToDevice);
      if (err != cudaSuccess) return fail(std::string(stage) + " blur copy", err);
    }
    return true;
  };
  auto normalizePositiveInPlace = [&](float* values, const char* stage) -> bool {
    cudaError_t err = cudaMemset(reductionBits + 8u, 0, sizeof(unsigned int));
    if (err != cudaSuccess) return fail(std::string(stage) + " max clear", err);
    glossFieldMaxKernel<<<cellBlocks, cellThreads>>>(cellCount, values, reductionBits + 8u);
    if (!check(std::string(stage) + " max")) return false;
    glossFieldNormalizeKernel<<<cellBlocks, cellThreads>>>(cellCount, values, values, reductionBits + 8u, 0);
    return check(std::string(stage) + " normalize");
  };
  auto assembleAndNormalize = [&](float* rawBody,
                                  float* rawPositive,
                                  float* rawNegative,
                                  float* rawConfidence,
                                  float* rawAgreement,
                                  float* outBody,
                                  float* outPositive,
                                  float* outNegative,
                                  float* outBoundary,
                                  float* outCongruence,
                                  float* outConfidence,
                                  float* outSignal,
                                  const char* stage) -> bool {
    cudaError_t err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
    if (err != cudaSuccess) return fail(std::string(stage) + " reduction clear", err);
    glossFieldAssembleUnifiedKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                                 gridHeight,
                                                                 rawBody,
                                                                 rawPositive,
                                                                 rawNegative,
                                                                 rawConfidence,
                                                                 rawAgreement,
                                                                 outBody,
                                                                 outSignal,
                                                                 outPositive,
                                                                 outNegative,
                                                                 outBoundary,
                                                                 outCongruence,
                                                                 outConfidence,
                                                                 reductionBits);
    if (!check(std::string(stage) + " assemble")) return false;
    glossFieldFinalNormalizeKernel<<<cellBlocks, cellThreads>>>(cellCount,
                                                                outBody,
                                                                outSignal,
                                                                outPositive,
                                                                outNegative,
                                                                outBoundary,
                                                                reductionBits);
    return check(std::string(stage) + " normalize");
  };

  glossFieldHybridCarrierKernel<<<cellBlocks, cellThreads>>>(cellCount, carrierMax, carrierY, carrierMin, carrierHybrid);
  if (!check("candidate 1 hybrid carrier")) return false;
  glossFieldTrimmedBodyKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                          gridHeight,
                                                          neighborhoodChoice,
                                                          occupancy,
                                                          meanR,
                                                          meanG,
                                                          meanB,
                                                          carrierHybrid,
                                                          viewerBody);
  if (!check("candidate 1 trimmed body")) return false;
  glossFieldLocalPercentileKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                              gridHeight,
                                                              analysisRadius,
                                                              45.0f,
                                                              carrierHybrid,
                                                              occupancy,
                                                              bodyCore);
  if (!check("candidate 1 body percentile")) return false;
  if (!blurRepeated(carrierHybrid, bodyContext, std::max(1, analysisRadius), "candidate 1 body context")) return false;
  {
    cudaError_t err = cudaMemset(reductionBits, 0, kSharedReductionUintCount * sizeof(unsigned int));
    if (err != cudaSuccess) return fail("candidate 1 prepare reduction clear", err);
  }
  glossFieldCandidate1PrepareExactKernel<<<cellBlocks, cellThreads>>>(cellCount,
                                                                      occupancy,
                                                                      carrierHybrid,
                                                                      viewerBody,
                                                                      bodyCore,
                                                                      bodyContext,
                                                                      body,
                                                                      positiveRaw,
                                                                      negativeRaw,
                                                                      confidenceRaw,
                                                                      reductionBits);
  if (!check("candidate 1 exact prepare")) return false;
  if (!blurRepeated(positiveRaw, agreementRaw, std::max(1, analysisRadius), "candidate 1 positive support")) {
    return false;
  }
  if (!normalizePositiveInPlace(agreementRaw, "candidate 1 positive support")) return false;
  if (!blurRepeated(confidenceRaw, bodyContext, std::max(1, analysisRadius), "candidate 1 consensus support")) {
    return false;
  }
  if (!normalizePositiveInPlace(bodyContext, "candidate 1 consensus support")) return false;
  glossFieldCandidate1FinalizeExactKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                                       gridHeight,
                                                                       occupancy,
                                                                       occupancySupport,
                                                                       body,
                                                                       viewerBody,
                                                                       bodyCore,
                                                                       positiveRaw,
                                                                       negativeRaw,
                                                                       confidenceRaw,
                                                                       agreementRaw,
                                                                       bodyContext,
                                                                       reductionBits,
                                                                       carrierHybrid,
                                                                       tempA,
                                                                       confidenceRaw,
                                                                       agreementRaw);
  if (!check("candidate 1 exact finalize")) return false;
  if (!assembleAndNormalize(body,
                            carrierHybrid,
                            tempA,
                            confidenceRaw,
                            agreementRaw,
                            body,
                            positive,
                            negative,
                            boundary,
                            congruence,
                            confidence,
                            signal,
                            "candidate 1")) {
    return false;
  }

  glossFieldTrimmedBodyKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                          gridHeight,
                                                          neighborhoodChoice,
                                                          occupancy,
                                                          meanR,
                                                          meanG,
                                                          meanB,
                                                          carrierY,
                                                          viewerBody);
  if (!check("candidate 2 trimmed body")) return false;
  glossFieldLocalPercentileKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                              gridHeight,
                                                              analysisRadius,
                                                              50.0f,
                                                              carrierY,
                                                              occupancy,
                                                              bodyCore);
  if (!check("candidate 2 body percentile")) return false;
  if (!blurRepeated(carrierY, bodyContext, std::max(2, analysisRadius * 2), "candidate 2 body context")) return false;
  glossFieldLocalPercentileKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                              gridHeight,
                                                              analysisRadius,
                                                              35.0f,
                                                              carrierY,
                                                              occupancy,
                                                              retinexBody);
  if (!check("candidate 2 retinex percentile")) return false;
  if (!blurRepeated(carrierY, dogLow, std::max(1, analysisRadius / 2), "candidate 2 DoG low")) return false;
  glossFieldCandidate2RawKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                            gridHeight,
                                                            neighborhoodChoice,
                                                            occupancy,
                                                            occupancySupport,
                                                            meanR,
                                                            meanG,
                                                            meanB,
                                                            carrierY,
                                                            viewerBody,
                                                            bodyCore,
                                                            bodyContext,
                                                            retinexBody,
                                                            dogLow,
                                                            bodyContext,
                                                            body2,
                                                            positiveRaw,
                                                            negativeRaw,
                                                            confidenceRaw,
                                                            agreementRaw);
  if (!check("candidate 2 raw solve")) return false;
  return assembleAndNormalize(body2,
                              positiveRaw,
                              negativeRaw,
                              confidenceRaw,
                              agreementRaw,
                              body2,
                              positive2,
                              negative2,
                              boundary2,
                              congruence2,
                              confidence2,
                              signal2,
                              "candidate 2");
}

bool buildGlossFieldFromRasterSourceDevice(InputCache* cache,
                                           const RasterSourceRequest& rasterRequest,
                                           const GlossFieldRequest& fieldRequest,
                                           const void* sourceDeviceBytes,
                                           size_t sourceByteCount,
                                           GlossFieldResult* out,
                                           std::string* error) {
  if (!cache || !out) {
    if (error) *error = "Missing CUDA raster gloss-field output.";
    return false;
  }
  *out = GlossFieldResult{};
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int gridWidth = std::max(fieldRequest.gridWidth, 1);
  const int gridHeight = std::max(fieldRequest.gridHeight, 1);
  const size_t cellCount = static_cast<size_t>(gridWidth) * static_cast<size_t>(gridHeight);
  const size_t pixelCount = static_cast<size_t>(std::max(rasterRequest.sourceWidth, 0)) *
                            static_cast<size_t>(std::max(rasterRequest.sourceHeight, 0));
  const size_t expectedBytes =
      pixelCount * 4u * (rasterRequest.pixelFormat == 1 ? sizeof(float) : sizeof(unsigned short));
  if (pointCount <= 0 || !sourceDeviceBytes || sourceByteCount < expectedBytes ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || cellCount == 0u) {
    if (error) *error = "Invalid CUDA raster gloss-field request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA raster gloss-field cache.";
    return false;
  }
  if (!ensureBoundsCapacity(impl, &localError)) {
    if (error) *error = localError;
    return false;
  }

  if (!ensureFieldWorkspace(impl, cellCount * kGlossFieldWorkspaceArrayCount, &localError)) {
    if (error) *error = localError;
    return false;
  }

  cudaError_t err = cudaMemset(
      impl->deviceFieldWorkspace, 0, cellCount * kGlossFieldWorkspaceArrayCount * sizeof(float));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to clear CUDA raster gloss-field workspace: ") + errorString(err);
    return false;
  }

  float* workspace = impl->deviceFieldWorkspace;
  float* occupancy = workspace + cellCount * 0u;
  float* sumR = workspace + cellCount * 1u;
  float* sumG = workspace + cellCount * 2u;
  float* sumB = workspace + cellCount * 3u;
  float* sumY = workspace + cellCount * 4u;
  float* sumMax = workspace + cellCount * 5u;
  float* sumMin = workspace + cellCount * 6u;
  float* sumNeutrality = workspace + cellCount * 7u;
  float* meanR = workspace + cellCount * 8u;
  float* meanG = workspace + cellCount * 9u;
  float* meanB = workspace + cellCount * 10u;
  float* carrierY = workspace + cellCount * 11u;
  float* carrierMax = workspace + cellCount * 12u;
  float* carrierMin = workspace + cellCount * 13u;
  float* neutrality = workspace + cellCount * 14u;
  float* occupancyNorm = workspace + cellCount * 15u;
  float* temp = workspace + cellCount * 16u;
  float* body = workspace + cellCount * (kGlossFieldCandidate1Base + 0u);
  float* positive = workspace + cellCount * (kGlossFieldCandidate1Base + 2u);
  float* negative = workspace + cellCount * (kGlossFieldCandidate1Base + 3u);
  float* boundary = workspace + cellCount * (kGlossFieldCandidate1Base + 4u);
  float* congruence = workspace + cellCount * (kGlossFieldCandidate1Base + 5u);
  float* confidence = workspace + cellCount * (kGlossFieldCandidate1Base + 6u);
  float* signal = workspace + cellCount * (kGlossFieldCandidate1Base + 7u);
  float* body2 = workspace + cellCount * (kGlossFieldCandidate2Base + 0u);
  float* positive2 = workspace + cellCount * (kGlossFieldCandidate2Base + 2u);
  float* negative2 = workspace + cellCount * (kGlossFieldCandidate2Base + 3u);
  float* boundary2 = workspace + cellCount * (kGlossFieldCandidate2Base + 4u);
  float* congruence2 = workspace + cellCount * (kGlossFieldCandidate2Base + 5u);
  float* confidence2 = workspace + cellCount * (kGlossFieldCandidate2Base + 6u);
  float* signal2 = workspace + cellCount * (kGlossFieldCandidate2Base + 7u);
  unsigned int* reductionBits = impl->deviceBounds;

  RasterSourceKernelUniforms rasterUniforms = makeRasterSourceUniforms(rasterRequest);
  rasterUniforms.input.pointCount = pointCount;
  const unsigned int pointThreads = 256u;
  const unsigned int pointBlocks =
      static_cast<unsigned int>((static_cast<size_t>(pointCount) + pointThreads - 1u) / pointThreads);
  rasterGlossFieldAccumulateKernel<<<pointBlocks, pointThreads>>>(
      static_cast<const unsigned char*>(sourceDeviceBytes),
      rasterUniforms,
      gridWidth,
      gridHeight,
      fieldRequest.showOverflow,
      occupancy,
      sumR,
      sumG,
      sumB,
      sumY,
      sumMax,
      sumMin,
      sumNeutrality);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster gloss-field accumulation failed: ") + errorString(err);
    return false;
  }

  const unsigned int cellThreads = 256u;
  const unsigned int cellBlocks = static_cast<unsigned int>((cellCount + cellThreads - 1u) / cellThreads);
  glossFieldFinalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                        occupancy,
                                                        sumR,
                                                        sumG,
                                                        sumB,
                                                        sumY,
                                                        sumMax,
                                                        sumMin,
                                                        sumNeutrality,
                                                        meanR,
                                                        meanG,
                                                        meanB,
                                                        carrierY,
                                                        carrierMax,
                                                        carrierMin,
                                                        neutrality);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster gloss-field finalize failed: ") + errorString(err);
    return false;
  }

  err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldMaxKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount), occupancy, reductionBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldNormalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                           occupancy,
                                                           occupancyNorm,
                                                           reductionBits,
                                                           0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldBlurKernel<<<cellBlocks, cellThreads>>>(gridWidth, gridHeight, occupancyNorm, temp);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(occupancyNorm, temp, cellCount * sizeof(float), cudaMemcpyDeviceToDevice);
  }
  if (err == cudaSuccess) err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldMaxKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount), occupancyNorm, reductionBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldNormalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                           occupancyNorm,
                                                           occupancyNorm,
                                                           reductionBits,
                                                           0);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster gloss-field occupancy normalization failed: ") + errorString(err);
    return false;
  }

  auto blurInPlace = [&](float* values, const char* label) -> bool {
    glossFieldBlurKernel<<<cellBlocks, cellThreads>>>(gridWidth, gridHeight, values, temp);
    cudaError_t blurErr = cudaGetLastError();
    if (blurErr != cudaSuccess) {
      if (error) *error = std::string("CUDA raster gloss-field blur failed for ") + label + ": " + errorString(blurErr);
      return false;
    }
    blurErr = cudaMemcpy(values, temp, cellCount * sizeof(float), cudaMemcpyDeviceToDevice);
    if (blurErr != cudaSuccess) {
      if (error) {
        *error = std::string("CUDA raster gloss-field blur copy failed for ") + label + ": " + errorString(blurErr);
      }
      return false;
    }
    return true;
  };
  if (!blurInPlace(carrierY, "carrierY")) return false;
  if (!blurInPlace(carrierMax, "carrierMax")) return false;
  if (!blurInPlace(carrierMin, "carrierMin")) return false;
  if (!blurInPlace(neutrality, "neutrality")) return false;

  if (!runGlossFieldResearchCandidateSolvers(gridWidth,
                                             gridHeight,
                                             fieldRequest.neighborhoodChoice,
                                             workspace,
                                             occupancy,
                                             meanR,
                                             meanG,
                                             meanB,
                                             carrierY,
                                             carrierMax,
                                             carrierMin,
                                             occupancyNorm,
                                             body,
                                             positive,
                                             negative,
                                             boundary,
                                             congruence,
                                             confidence,
                                             signal,
                                             body2,
                                             positive2,
                                             negative2,
                                             boundary2,
                                             congruence2,
                                             confidence2,
                                             signal2,
                                             reductionBits,
                                             cellBlocks,
                                             cellThreads,
                                             "raster gloss-field",
                                             error)) {
    return false;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster gloss-field synchronization failed: ") + errorString(err);
    return false;
  }

  out->gridWidth = gridWidth;
  out->gridHeight = gridHeight;
  if (fieldRequest.readbackResult == 0) {
    return true;
  }
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
  out->body2.assign(cellCount, 0.0f);
  out->signal2.assign(cellCount, 0.0f);
  out->positive2.assign(cellCount, 0.0f);
  out->negative2.assign(cellCount, 0.0f);
  out->boundary2.assign(cellCount, 0.0f);
  out->congruence2.assign(cellCount, 0.0f);
  out->confidence2.assign(cellCount, 0.0f);

  auto copyBack = [&](std::vector<float>* dst, const float* src, size_t floatCount, const char* label) -> bool {
    if (!dst || dst->size() != floatCount) return false;
    cudaError_t copyErr = cudaMemcpy(dst->data(), src, floatCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
      if (error) *error = std::string("Failed to read CUDA raster gloss-field ") + label + ": " + errorString(copyErr);
      return false;
    }
    return true;
  };

  if (!copyBack(&out->occupancy, occupancy, cellCount, "occupancy")) return false;
  std::vector<float> meanRHost(cellCount, 0.0f);
  std::vector<float> meanGHost(cellCount, 0.0f);
  std::vector<float> meanBHost(cellCount, 0.0f);
  if (!copyBack(&meanRHost, meanR, cellCount, "meanR")) return false;
  if (!copyBack(&meanGHost, meanG, cellCount, "meanG")) return false;
  if (!copyBack(&meanBHost, meanB, cellCount, "meanB")) return false;
  if (!copyBack(&out->carrierY, carrierY, cellCount, "carrierY")) return false;
  if (!copyBack(&out->carrierMax, carrierMax, cellCount, "carrierMax")) return false;
  if (!copyBack(&out->carrierMin, carrierMin, cellCount, "carrierMin")) return false;
  if (!copyBack(&out->neutrality, neutrality, cellCount, "neutrality")) return false;
  if (!copyBack(&out->body, body, cellCount, "body")) return false;
  if (!copyBack(&out->signal, signal, cellCount, "signal")) return false;
  if (!copyBack(&out->positive, positive, cellCount, "positive")) return false;
  if (!copyBack(&out->negative, negative, cellCount, "negative")) return false;
  if (!copyBack(&out->boundary, boundary, cellCount, "boundary")) return false;
  if (!copyBack(&out->congruence, congruence, cellCount, "congruence")) return false;
  if (!copyBack(&out->confidence, confidence, cellCount, "confidence")) return false;
  if (!copyBack(&out->body2, body2, cellCount, "body2")) return false;
  if (!copyBack(&out->signal2, signal2, cellCount, "signal2")) return false;
  if (!copyBack(&out->positive2, positive2, cellCount, "positive2")) return false;
  if (!copyBack(&out->negative2, negative2, cellCount, "negative2")) return false;
  if (!copyBack(&out->boundary2, boundary2, cellCount, "boundary2")) return false;
  if (!copyBack(&out->congruence2, congruence2, cellCount, "congruence2")) return false;
  if (!copyBack(&out->confidence2, confidence2, cellCount, "confidence2")) return false;
  for (size_t idx = 0; idx < cellCount; ++idx) {
    out->meanRgb[idx * 3u + 0u] = meanRHost[idx];
    out->meanRgb[idx * 3u + 1u] = meanGHost[idx];
    out->meanRgb[idx * 3u + 2u] = meanBHost[idx];
  }
  return true;
}

bool buildGlossField(InputCache* cache,
                     const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     bool allowHostUpload,
                     bool allowReadback,
                     GlossFieldResult* out,
                     std::string* error) {
  if (!cache || !out) {
    if (error) *error = "Missing CUDA gloss-field output.";
    return false;
  }
  if (!allowHostUpload) {
    if (error) *error = "CUDA packed gloss-field host upload disabled; resident raster Gloss path required.";
    return false;
  }
  if (!allowReadback) {
    if (error) *error = "CUDA packed gloss-field readback disabled; resident raster Gloss path required.";
    return false;
  }
  const int gridWidth = max(request.gridWidth, 1);
  const int gridHeight = max(request.gridHeight, 1);
  const size_t pointCount = packedPoints.size() / 6u;
  const size_t cellCount = static_cast<size_t>(gridWidth) * static_cast<size_t>(gridHeight);
  if (pointCount == 0u || cellCount == 0u) {
    if (error) *error = "Invalid CUDA gloss-field request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA gloss-field cache.";
    return false;
  }
  if (!ensureInputCapacity(impl, packedPoints.size(), &localError)) {
    if (error) *error = localError;
    return false;
  }
  if (!ensureBoundsCapacity(impl, &localError)) {
    if (error) *error = localError;
    return false;
  }

  if (!ensureFieldWorkspace(impl, cellCount * kGlossFieldWorkspaceArrayCount, &localError)) {
    if (error) *error = localError;
    return false;
  }

  cudaError_t err = cudaMemcpy(impl->deviceInput,
                               packedPoints.data(),
                               packedPoints.size() * sizeof(float),
                               cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to upload CUDA gloss-field input: ") + errorString(err);
    return false;
  }

  err = cudaMemset(
      impl->deviceFieldWorkspace, 0, cellCount * kGlossFieldWorkspaceArrayCount * sizeof(float));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to clear CUDA gloss-field workspace: ") + errorString(err);
    return false;
  }

  float* workspace = impl->deviceFieldWorkspace;
  float* occupancy = workspace + cellCount * 0u;
  float* sumR = workspace + cellCount * 1u;
  float* sumG = workspace + cellCount * 2u;
  float* sumB = workspace + cellCount * 3u;
  float* sumY = workspace + cellCount * 4u;
  float* sumMax = workspace + cellCount * 5u;
  float* sumMin = workspace + cellCount * 6u;
  float* sumNeutrality = workspace + cellCount * 7u;
  float* meanR = workspace + cellCount * 8u;
  float* meanG = workspace + cellCount * 9u;
  float* meanB = workspace + cellCount * 10u;
  float* carrierY = workspace + cellCount * 11u;
  float* carrierMax = workspace + cellCount * 12u;
  float* carrierMin = workspace + cellCount * 13u;
  float* neutrality = workspace + cellCount * 14u;
  float* occupancyNorm = workspace + cellCount * 15u;
  float* temp = workspace + cellCount * 16u;
  float* body = workspace + cellCount * (kGlossFieldCandidate1Base + 0u);
  float* positive = workspace + cellCount * (kGlossFieldCandidate1Base + 2u);
  float* negative = workspace + cellCount * (kGlossFieldCandidate1Base + 3u);
  float* boundary = workspace + cellCount * (kGlossFieldCandidate1Base + 4u);
  float* congruence = workspace + cellCount * (kGlossFieldCandidate1Base + 5u);
  float* confidence = workspace + cellCount * (kGlossFieldCandidate1Base + 6u);
  float* signal = workspace + cellCount * (kGlossFieldCandidate1Base + 7u);
  float* body2 = workspace + cellCount * (kGlossFieldCandidate2Base + 0u);
  float* positive2 = workspace + cellCount * (kGlossFieldCandidate2Base + 2u);
  float* negative2 = workspace + cellCount * (kGlossFieldCandidate2Base + 3u);
  float* boundary2 = workspace + cellCount * (kGlossFieldCandidate2Base + 4u);
  float* congruence2 = workspace + cellCount * (kGlossFieldCandidate2Base + 5u);
  float* confidence2 = workspace + cellCount * (kGlossFieldCandidate2Base + 6u);
  float* signal2 = workspace + cellCount * (kGlossFieldCandidate2Base + 7u);
  unsigned int* reductionBits = impl->deviceBounds;

  const unsigned int pointThreads = 256u;
  constexpr size_t kGlossFieldBatchPointCount = 262144u;
  for (size_t pointOffset = 0u; pointOffset < pointCount; pointOffset += kGlossFieldBatchPointCount) {
    const size_t batchPointCount = std::min(kGlossFieldBatchPointCount, pointCount - pointOffset);
    const unsigned int pointBlocks = static_cast<unsigned int>((batchPointCount + pointThreads - 1u) / pointThreads);
    glossFieldAccumulateKernel<<<pointBlocks, pointThreads>>>(impl->deviceInput + pointOffset * 6u,
                                                              static_cast<int>(batchPointCount),
                                                              gridWidth,
                                                              gridHeight,
                                                              request.showOverflow,
                                                              occupancy,
                                                              sumR,
                                                              sumG,
                                                              sumB,
                                                              sumY,
                                                              sumMax,
                                                              sumMin,
                                                              sumNeutrality);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      if (error) {
        *error = std::string("CUDA gloss-field accumulation failed at batch ") +
                 std::to_string(pointOffset / kGlossFieldBatchPointCount) + ": " + errorString(err);
      }
      return false;
    }
  }

  const unsigned int cellThreads = 256u;
  const unsigned int cellBlocks = static_cast<unsigned int>((cellCount + cellThreads - 1u) / cellThreads);
  glossFieldFinalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                        occupancy,
                                                        sumR,
                                                        sumG,
                                                        sumB,
                                                        sumY,
                                                        sumMax,
                                                        sumMin,
                                                        sumNeutrality,
                                                        meanR,
                                                        meanG,
                                                        meanB,
                                                        carrierY,
                                                        carrierMax,
                                                        carrierMin,
                                                        neutrality);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field finalize failed: ") + errorString(err);
    return false;
  }

  err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldMaxKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount), occupancy, reductionBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldNormalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                           occupancy,
                                                           occupancyNorm,
                                                           reductionBits,
                                                           0);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldBlurKernel<<<cellBlocks, cellThreads>>>(gridWidth, gridHeight, occupancyNorm, temp);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(occupancyNorm, temp, cellCount * sizeof(float), cudaMemcpyDeviceToDevice);
  }
  if (err == cudaSuccess) err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldMaxKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount), occupancyNorm, reductionBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) {
    glossFieldNormalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                           occupancyNorm,
                                                           occupancyNorm,
                                                            reductionBits,
                                                            0);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field occupancy normalization failed: ") + errorString(err);
    return false;
  }

  auto blurInPlace = [&](float* values) -> bool {
    glossFieldBlurKernel<<<cellBlocks, cellThreads>>>(gridWidth, gridHeight, values, temp);
    cudaError_t blurErr = cudaGetLastError();
    if (blurErr != cudaSuccess) {
      if (error) *error = std::string("CUDA gloss-field blur failed: ") + errorString(blurErr);
      return false;
    }
    blurErr = cudaMemcpy(values, temp, cellCount * sizeof(float), cudaMemcpyDeviceToDevice);
    if (blurErr != cudaSuccess) {
      if (error) *error = std::string("CUDA gloss-field blur copy failed: ") + errorString(blurErr);
      return false;
    }
    return true;
  };
  if (err == cudaSuccess && !blurInPlace(carrierY)) return false;
  if (err == cudaSuccess && !blurInPlace(carrierMax)) return false;
  if (err == cudaSuccess && !blurInPlace(carrierMin)) return false;
  if (err == cudaSuccess && !blurInPlace(neutrality)) return false;

  if (!runGlossFieldResearchCandidateSolvers(gridWidth,
                                             gridHeight,
                                             request.neighborhoodChoice,
                                             workspace,
                                             occupancy,
                                             meanR,
                                             meanG,
                                             meanB,
                                             carrierY,
                                             carrierMax,
                                             carrierMin,
                                             occupancyNorm,
                                             body,
                                             positive,
                                             negative,
                                             boundary,
                                             congruence,
                                             confidence,
                                             signal,
                                             body2,
                                             positive2,
                                             negative2,
                                             boundary2,
                                             congruence2,
                                             confidence2,
                                             signal2,
                                             reductionBits,
                                             cellBlocks,
                                             cellThreads,
                                             "packed gloss-field",
                                             error)) {
    return false;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field synchronization failed: ") + errorString(err);
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
  out->body2.assign(cellCount, 0.0f);
  out->signal2.assign(cellCount, 0.0f);
  out->positive2.assign(cellCount, 0.0f);
  out->negative2.assign(cellCount, 0.0f);
  out->boundary2.assign(cellCount, 0.0f);
  out->congruence2.assign(cellCount, 0.0f);
  out->confidence2.assign(cellCount, 0.0f);

  auto copyBack = [&](std::vector<float>* dst, const float* src, size_t floatCount, const char* label) -> bool {
    if (!dst || dst->size() != floatCount) return false;
    cudaError_t copyErr = cudaMemcpy(dst->data(), src, floatCount * sizeof(float), cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
      if (error) *error = std::string("Failed to read CUDA gloss-field ") + label + ": " + errorString(copyErr);
      return false;
    }
    return true;
  };

  if (!copyBack(&out->occupancy, occupancy, cellCount, "occupancy")) return false;
  std::vector<float> meanRHost(cellCount, 0.0f);
  std::vector<float> meanGHost(cellCount, 0.0f);
  std::vector<float> meanBHost(cellCount, 0.0f);
  if (!copyBack(&meanRHost, meanR, cellCount, "meanR")) return false;
  if (!copyBack(&meanGHost, meanG, cellCount, "meanG")) return false;
  if (!copyBack(&meanBHost, meanB, cellCount, "meanB")) return false;
  if (!copyBack(&out->carrierY, carrierY, cellCount, "carrierY")) return false;
  if (!copyBack(&out->carrierMax, carrierMax, cellCount, "carrierMax")) return false;
  if (!copyBack(&out->carrierMin, carrierMin, cellCount, "carrierMin")) return false;
  if (!copyBack(&out->neutrality, neutrality, cellCount, "neutrality")) return false;
  if (!copyBack(&out->body, body, cellCount, "body")) return false;
  if (!copyBack(&out->signal, signal, cellCount, "signal")) return false;
  if (!copyBack(&out->positive, positive, cellCount, "positive")) return false;
  if (!copyBack(&out->negative, negative, cellCount, "negative")) return false;
  if (!copyBack(&out->boundary, boundary, cellCount, "boundary")) return false;
  if (!copyBack(&out->congruence, congruence, cellCount, "congruence")) return false;
  if (!copyBack(&out->confidence, confidence, cellCount, "confidence")) return false;
  if (!copyBack(&out->body2, body2, cellCount, "body2")) return false;
  if (!copyBack(&out->signal2, signal2, cellCount, "signal2")) return false;
  if (!copyBack(&out->positive2, positive2, cellCount, "positive2")) return false;
  if (!copyBack(&out->negative2, negative2, cellCount, "negative2")) return false;
  if (!copyBack(&out->boundary2, boundary2, cellCount, "boundary2")) return false;
  if (!copyBack(&out->congruence2, congruence2, cellCount, "congruence2")) return false;
  if (!copyBack(&out->confidence2, confidence2, cellCount, "confidence2")) return false;
  for (size_t idx = 0; idx < cellCount; ++idx) {
    out->meanRgb[idx * 3u + 0u] = meanRHost[idx];
    out->meanRgb[idx * 3u + 1u] = meanGHost[idx];
    out->meanRgb[idx * 3u + 2u] = meanBHost[idx];
  }
  return true;
}

}  // namespace ChromaspaceCuda
