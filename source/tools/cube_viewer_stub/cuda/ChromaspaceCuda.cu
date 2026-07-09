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
  cudaError_t err = cudaMalloc(&impl->deviceBounds, 6u * sizeof(unsigned int));
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
                                   const int* occupancyBins) {
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
    visible = occupancyBins[bin] <= max(u.occupancyTargetThreshold, 0);
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
  if (request.excludeOverflow != 0 && (value < 0.0f || value > 1.0f)) return;
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

inline __device__ int glossNeighborhoodRadiusCells(int neighborhoodChoice) {
  switch (max(0, min(neighborhoodChoice, 2))) {
    case 0: return 1;
    case 2: return 3;
    case 1:
    default: return 2;
  }
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

__global__ void glossFieldBodyKernel(int gridWidth,
                                     int gridHeight,
                                     int neighborhoodChoice,
                                     const float* occupancy,
                                     const float* meanR,
                                     const float* meanG,
                                     const float* meanB,
                                     const float* carrierMax,
                                     float* body) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !occupancy || !meanR || !meanG || !meanB || !carrierMax || !body) return;
  if (occupancy[index] <= 0.5f) {
    body[index] = 0.0f;
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const int radiusCells = glossNeighborhoodRadiusCells(neighborhoodChoice);
  constexpr int kMaxNeighborhood = 49;
  float carriers[kMaxNeighborhood];
  int neighborIndices[kMaxNeighborhood];
  int count = 0;
  const float centerCarrier = carrierMax[index];
  const float centerR = meanR[index];
  const float centerG = meanG[index];
  const float centerB = meanB[index];
  for (int oy = -radiusCells; oy <= radiusCells; ++oy) {
    const int yy = y + oy;
    if (yy < 0 || yy >= gridHeight) continue;
    for (int ox = -radiusCells; ox <= radiusCells; ++ox) {
      const int xx = x + ox;
      if (xx < 0 || xx >= gridWidth) continue;
      const unsigned int neighborIndex = static_cast<unsigned int>(yy * gridWidth + xx);
      if (occupancy[neighborIndex] <= 0.5f) continue;
      const float carrier = carrierMax[neighborIndex];
      const float dr = meanR[neighborIndex] - centerR;
      const float dg = meanG[neighborIndex] - centerG;
      const float db = meanB[neighborIndex] - centerB;
      const float colorDistance = sqrtf(dr * dr + dg * dg + db * db);
      if (fabsf(carrier - centerCarrier) > 0.26f && colorDistance > 0.20f) continue;
      if (count < kMaxNeighborhood) {
        carriers[count] = carrier;
        neighborIndices[count] = static_cast<int>(neighborIndex);
        ++count;
      }
    }
  }
  if (count <= 0) {
    body[index] = centerCarrier;
    return;
  }
  for (int i = 1; i < count; ++i) {
    const float keyCarrier = carriers[i];
    const int keyIndex = neighborIndices[i];
    int j = i - 1;
    while (j >= 0 && (carriers[j] > keyCarrier || (carriers[j] == keyCarrier && neighborIndices[j] > keyIndex))) {
      carriers[j + 1] = carriers[j];
      neighborIndices[j + 1] = neighborIndices[j];
      --j;
    }
    carriers[j + 1] = keyCarrier;
    neighborIndices[j + 1] = keyIndex;
  }
  const int trim = count >= 6 ? max(1, count / 6) : 0;
  const int begin = min(trim, count);
  const int end = max(begin + 1, count - trim);
  float bodySum = 0.0f;
  float bodyWeight = 0.0f;
  for (int i = begin; i < end; ++i) {
    const int neighborIndex = neighborIndices[i];
    const int neighborX = neighborIndex % gridWidth;
    const int neighborY = neighborIndex / gridWidth;
    const float dx = static_cast<float>(neighborX - x);
    const float dy = static_cast<float>(neighborY - y);
    const float spatialWeight = 1.0f / (1.0f + dx * dx + dy * dy);
    bodySum += carriers[i] * spatialWeight;
    bodyWeight += spatialWeight;
  }
  body[index] = bodyWeight > 1e-6f ? (bodySum / bodyWeight) : centerCarrier;
}

__global__ void glossFieldRawSignalKernel(int cellCount,
                                          const float* occupancy,
                                          const float* carrierMax,
                                          const float* body,
                                          float* rawSignal,
                                          unsigned int* maxBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(cellCount, 0));
  if (index >= total || !occupancy || !carrierMax || !body || !rawSignal || !maxBits) return;
  if (occupancy[index] <= 0.5f) {
    rawSignal[index] = 0.0f;
    return;
  }
  const float bodyValue = fmaxf(body[index], 0.0f);
  const float rawPositive = fmaxf(0.0f, carrierMax[index] - bodyValue);
  const float rawNegative = fmaxf(0.0f, bodyValue - carrierMax[index]);
  rawSignal[index] = rawPositive - rawNegative;
  atomicMax(&maxBits[0], __float_as_uint(bodyValue));
}

__global__ void glossFieldWeightedSignalKernel(int gridWidth,
                                               int gridHeight,
                                               const float* occupancyNorm,
                                               const float* body,
                                               const float* rawSignal,
                                               float* positive,
                                               float* negative,
                                               float* boundary,
                                               float* congruence,
                                               float* confidence,
                                               float* signal,
                                               unsigned int* maxBits) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = static_cast<unsigned int>(max(gridWidth * gridHeight, 0));
  if (index >= total || !occupancyNorm || !body || !rawSignal || !positive || !negative || !boundary ||
      !congruence || !confidence || !signal || !maxBits) {
    return;
  }
  const int x = static_cast<int>(index % static_cast<unsigned int>(gridWidth));
  const int y = static_cast<int>(index / static_cast<unsigned int>(gridWidth));
  const float occCenter = sampleGridClampedDevice(occupancyNorm, gridWidth, gridHeight, x, y);
  if (occCenter <= 0.0f) {
    positive[index] = 0.0f;
    negative[index] = 0.0f;
    boundary[index] = 0.0f;
    congruence[index] = 0.0f;
    confidence[index] = 0.0f;
    signal[index] = 0.0f;
    return;
  }
  const float gxCarrier = sampleGridClampedDevice(body, gridWidth, gridHeight, x + 1, y) -
                          sampleGridClampedDevice(body, gridWidth, gridHeight, x - 1, y);
  const float gyCarrier = sampleGridClampedDevice(body, gridWidth, gridHeight, x, y + 1) -
                          sampleGridClampedDevice(body, gridWidth, gridHeight, x, y - 1);
  const float gxSignal = sampleGridClampedDevice(rawSignal, gridWidth, gridHeight, x + 1, y) -
                         sampleGridClampedDevice(rawSignal, gridWidth, gridHeight, x - 1, y);
  const float gySignal = sampleGridClampedDevice(rawSignal, gridWidth, gridHeight, x, y + 1) -
                         sampleGridClampedDevice(rawSignal, gridWidth, gridHeight, x, y - 1);
  const float magCarrier = sqrtf(gxCarrier * gxCarrier + gyCarrier * gyCarrier);
  const float magSignal = sqrtf(gxSignal * gxSignal + gySignal * gySignal);
  float localCongruence = 0.0f;
  if (magCarrier > 1e-6f && magSignal > 1e-6f) {
    localCongruence = fabsf((gxCarrier * gxSignal + gyCarrier * gySignal) / (magCarrier * magSignal));
  } else if (magSignal > 1e-6f) {
    localCongruence = 0.35f;
  }
  const float occNeighborhood =
      (occCenter +
       sampleGridClampedDevice(occupancyNorm, gridWidth, gridHeight, x + 1, y) +
       sampleGridClampedDevice(occupancyNorm, gridWidth, gridHeight, x - 1, y) +
       sampleGridClampedDevice(occupancyNorm, gridWidth, gridHeight, x, y + 1) +
       sampleGridClampedDevice(occupancyNorm, gridWidth, gridHeight, x, y - 1)) /
      5.0f;
  const float localConfidence =
      clamp01(sqrtf(occCenter) * clamp01(0.28f + 0.72f * occNeighborhood));
  const float posWeighted = fmaxf(0.0f, rawSignal[index]) * (0.30f + 0.70f * localCongruence) * localConfidence;
  const float negWeighted = fmaxf(0.0f, -rawSignal[index]) * (0.30f + 0.70f * localCongruence) * localConfidence;
  const float boundaryValue = clamp01(magSignal * 4.0f) * localConfidence;
  positive[index] = posWeighted;
  negative[index] = negWeighted;
  boundary[index] = boundaryValue;
  congruence[index] = localCongruence;
  confidence[index] = localConfidence;
  signal[index] = posWeighted - negWeighted;
  atomicMax(&maxBits[0], __float_as_uint(fmaxf(posWeighted, 0.0f)));
  atomicMax(&maxBits[1], __float_as_uint(fmaxf(negWeighted, 0.0f)));
  atomicMax(&maxBits[2], __float_as_uint(fmaxf(boundaryValue, 0.0f)));
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
                           unsigned long long serial,
                           std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA raster source cache has no GL buffers.";
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
  if (!ensureSourceCapacity(impl, sourceByteCount, &localError)) {
    if (error) *error = localError;
    return false;
  }
  cudaError_t err = cudaMemcpy(impl->deviceSource, sourceBytes, sourceByteCount, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to upload CUDA raster source: ") + errorString(err);
    return false;
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
  constexpr int kRasterOccupancyBinCount = 18 * 18 * 18;
  if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
    err = cudaMalloc(reinterpret_cast<void**>(&occupancyBins),
                     static_cast<size_t>(kRasterOccupancyBinCount) * sizeof(int));
    if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&visibleCount), sizeof(int));
    if (err != cudaSuccess || !occupancyBins || !visibleCount) {
      if (occupancyBins) cudaFree(occupancyBins);
      if (visibleCount) cudaFree(visibleCount);
      if (error) *error = std::string("Failed to allocate CUDA raster occupancy buffers: ") + errorString(err);
      return false;
    }
    err = cudaMemset(occupancyBins, 0, static_cast<size_t>(kRasterOccupancyBinCount) * sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(visibleCount, 0, sizeof(int));
    if (err != cudaSuccess) {
      cudaFree(occupancyBins);
      cudaFree(visibleCount);
      if (error) *error = std::string("Failed to clear CUDA raster occupancy buffers: ") + errorString(err);
      return false;
    }
    const unsigned int threads = 256u;
    const unsigned int countBlocks =
        static_cast<unsigned int>((static_cast<size_t>(std::max(uniforms.basePointCount, 0)) + threads - 1u) / threads);
    rasterOccupancyCountKernel<<<std::max(1u, countBlocks), threads>>>(
        impl->deviceSource, uniforms, occupancyBins, visibleCount);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    int hostVisibleCount = 0;
    if (err == cudaSuccess) err = cudaMemcpy(&hostVisibleCount, visibleCount, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      cudaFree(occupancyBins);
      cudaFree(visibleCount);
      if (error) *error = std::string("CUDA raster occupancy count failed: ") + errorString(err);
      return false;
    }
    const float meanOccupancy =
        static_cast<float>(std::max(hostVisibleCount, 0)) / static_cast<float>(kRasterOccupancyBinCount);
    uniforms.occupancyTargetThreshold = std::max(0, static_cast<int>(std::ceil(meanOccupancy * 0.72f)));
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (occupancyBins) cudaFree(occupancyBins);
    if (visibleCount) cudaFree(visibleCount);
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
    if (error) *error = std::string("Failed to access CUDA raster mapped buffers: ") + errorString(err);
    return false;
  }

  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  rasterSourceKernel<<<blocks, threads>>>(devVerts, devColors, impl->deviceSource, uniforms, occupancyBins);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (occupancyBins) cudaFree(occupancyBins);
  if (visibleCount) cudaFree(visibleCount);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA raster source kernel failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(pointCount);
  cache->available = true;
  cache->hasFitBounds = false;
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

bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       std::vector<float>* outDensity,
                       std::string* error) {
  if (!outDensity) {
    if (error) *error = "Missing CUDA scope-density output.";
    return false;
  }
  outDensity->clear();
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

bool buildGlossField(InputCache* cache,
                     const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     GlossFieldResult* out,
                     std::string* error) {
  if (!cache || !out) {
    if (error) *error = "Missing CUDA gloss-field output.";
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

  constexpr size_t kWorkspaceArrayCount = 25u;
  if (!ensureFieldWorkspace(impl, cellCount * kWorkspaceArrayCount, &localError)) {
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

  err = cudaMemset(impl->deviceFieldWorkspace, 0, cellCount * kWorkspaceArrayCount * sizeof(float));
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
  float* body = workspace + cellCount * 17u;
  float* rawSignal = workspace + cellCount * 18u;
  float* positive = workspace + cellCount * 19u;
  float* negative = workspace + cellCount * 20u;
  float* boundary = workspace + cellCount * 21u;
  float* congruence = workspace + cellCount * 22u;
  float* confidence = workspace + cellCount * 23u;
  float* signal = workspace + cellCount * 24u;
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

  glossFieldBodyKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                    gridHeight,
                                                    request.neighborhoodChoice,
                                                    occupancy,
                                                    meanR,
                                                    meanG,
                                                    meanB,
                                                    carrierMax,
                                                    body);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field body fit failed: ") + errorString(err);
    return false;
  }

  err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldRawSignalKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                           occupancy,
                                                           carrierMax,
                                                           body,
                                                           rawSignal,
                                                           reductionBits);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field raw signal failed: ") + errorString(err);
    return false;
  }

  unsigned int bodyMaxBitsHost = 0u;
  err = cudaMemcpy(&bodyMaxBitsHost, reductionBits, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA gloss-field body max: ") + errorString(err);
    return false;
  }

  err = cudaMemset(reductionBits, 0, 6u * sizeof(unsigned int));
  if (err == cudaSuccess) {
    glossFieldWeightedSignalKernel<<<cellBlocks, cellThreads>>>(gridWidth,
                                                                gridHeight,
                                                                occupancyNorm,
                                                                body,
                                                                rawSignal,
                                                                positive,
                                                                negative,
                                                                boundary,
                                                                congruence,
                                                                confidence,
                                                                signal,
                                                                reductionBits);
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field weighted signal failed: ") + errorString(err);
    return false;
  }

  std::array<unsigned int, 4> weightedMaxBitsHost = {bodyMaxBitsHost, 0u, 0u, 0u};
  err = cudaMemcpy(weightedMaxBitsHost.data() + 1u, reductionBits, 3u * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA gloss-field maxima: ") + errorString(err);
    return false;
  }
  err = cudaMemcpy(reductionBits, weightedMaxBitsHost.data(), 4u * sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    glossFieldFinalNormalizeKernel<<<cellBlocks, cellThreads>>>(static_cast<int>(cellCount),
                                                                body,
                                                                signal,
                                                                positive,
                                                                negative,
                                                                boundary,
                                                                reductionBits);
    err = cudaGetLastError();
  }
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA gloss-field normalization failed: ") + errorString(err);
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
  for (size_t idx = 0; idx < cellCount; ++idx) {
    out->meanRgb[idx * 3u + 0u] = meanRHost[idx];
    out->meanRgb[idx * 3u + 1u] = meanGHost[idx];
    out->meanRgb[idx * 3u + 2u] = meanBHost[idx];
  }
  return true;
}

}  // namespace ChromaspaceCuda
