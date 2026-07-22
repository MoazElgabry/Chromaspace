#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceMetal {

constexpr int kMaxLassoStrokes = 64;
constexpr int kMaxLassoPoints = 1024;

struct Sample {
  float xNorm = 0.0f;
  float yNorm = 0.0f;
  float zReserved = 0.0f;
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
};

#if defined(CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT)
struct SharedSourceSignalSurface {
  std::uint32_t surfaceId = 0;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  std::size_t byteSize = 0;
  void* retainedSurface = nullptr;
  bool global = false;
  bool selfLookupOk = false;
};
#endif

struct OccupancyCandidate {
  Sample sample{};
  float normalizedNeutralRadius = 0.0f;
  int bin = 0;
  std::uint32_t tie = 0;
};

struct Request {
  const void* srcMetalBuffer = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
  int originX = 0;
  int originY = 0;
  int scaledWidth = 0;
  int scaledHeight = 0;
  int pointCount = 0;
  int extraPointCount = 0;
  int candidateTarget = 0;
  int maxPrimaryAttempts = 0;
  int maxCandidateAttempts = 0;
  int samplingMode = 0;
  int samplingGridWidth = 0;
  int samplingGridHeight = 0;
  int preserveOverflow = 0;
  int occupancyFill = 0;
  int plotMode = 0;
  int circularHsl = 0;
  int circularHsv = 0;
  int normConeNormalized = 1;
  int showOverflow = 0;
  int plotDisplayLinearEnabled = 0;
  int plotDisplayLinearTransfer = 0;
  int neutralRadiusEnabled = 0;
  float neutralRadius = 1.0f;
  int imageLassoEnabled = 0;
  int lassoBoundsValid = 0;
  float lassoMinX = 0.0f;
  float lassoMinY = 0.0f;
  float lassoMaxX = 1.0f;
  float lassoMaxY = 1.0f;
  int lassoStrokeCount = 0;
  int lassoPointCount = 0;
  int lassoStrokeStart[kMaxLassoStrokes] = {};
  int lassoStrokePointCount[kMaxLassoStrokes] = {};
  int lassoStrokeSubtract[kMaxLassoStrokes] = {};
  float lassoPointX[kMaxLassoPoints] = {};
  float lassoPointY[kMaxLassoPoints] = {};
  void* metalCommandQueue = nullptr;
};

struct Result {
  std::vector<Sample> primarySamples;
  std::vector<Sample> appendedSamples;
  std::vector<OccupancyCandidate> occupancyCandidates;
  std::vector<int> occupancy;
  int primaryAttempts = 0;
  int primaryAccepted = 0;
  int extraPointCount = 0;
  std::string error;
  bool success = false;
};

bool buildWholeImageCloud(const Request& request, Result* out);

struct RampLayoutRequest {
  const void* srcMetalBuffer = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
  int originX = 0;
  int originY = 0;
  int candidateY1[2] = {};
  int candidateHeight[2] = {};
  void* metalCommandQueue = nullptr;
};

struct RampLayoutResult {
  float scores[2] = {};
  int selectedCandidate = 0;
  std::string error;
  bool success = false;
};

bool detectGrayRampLayout(const RampLayoutRequest& request, RampLayoutResult* out);

struct StripRequest {
  const void* srcMetalBuffer = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
  int originX = 0;
  int originY = 0;
  int resolution = 0;
  int preserveOverflow = 0;
  int readCube = 0;
  int readRamp = 0;
  int plotDisplayLinearEnabled = 0;
  int plotDisplayLinearTransfer = 0;
  int cubeY1 = 0;
  int stripHeight = 0;
  int rampY1 = 0;
  int rampHeight = 0;
  int rampSampleRows = 0;
  float cellWidth = 1.0f;
  void* metalCommandQueue = nullptr;
};

struct StripResult {
  std::vector<Sample> samples;
  std::string error;
  bool success = false;
};

bool buildIdentityStripCloud(const StripRequest& request, StripResult* out);

struct CombinedResult {
  std::vector<Sample> primarySamples;
  std::vector<Sample> combinedSamples;
  std::vector<Sample> appendedSamples;
  std::vector<OccupancyCandidate> occupancyCandidates;
  std::vector<int> occupancy;
  std::vector<Sample> stripSamples;
  int primaryAttempts = 0;
  int primaryAccepted = 0;
  int extraPointCount = 0;
  std::string error;
  bool success = false;
};

bool buildWholeImageAndIdentityStripCloud(const Request& wholeImageRequest, const StripRequest& stripRequest, CombinedResult* out);

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
    int overlayHeight);

bool copySourceToHost(
    const void* srcMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    float* readbackSrc,
    size_t readbackSrcRowBytes);

bool copySourceProxyToHost(
    const void* srcMetalBuffer,
    int sourceWidth,
    int sourceHeight,
    size_t srcRowBytes,
    int originX,
    int originY,
    int proxyWidth,
    int proxyHeight,
    void* metalCommandQueue,
    float* readbackProxy,
    size_t readbackProxyRowBytes,
    std::string* error = nullptr);

using SourceProxyCompletionCallback =
    void (*)(void* context, bool completedSuccessfully);

// Warm the Source Signal proxy pipeline outside the OFX render callback.
bool prepareSourceProxyPipeline(
    void* metalCommandQueue,
    std::string* error = nullptr);

// Encodes a source proxy into an already allocated shareable texture, signals
// the supplied shared event in the same command buffer, and commits without a
// completion wait. The callback, when non-null, runs on Metal's completion
// executor and must not call OFX host interfaces.
bool enqueueSourceProxyToSharedTexture(
    const void* srcMetalBuffer,
    int sourceWidth,
    int sourceHeight,
    size_t srcRowBytes,
    int originX,
    int originY,
    int proxyWidth,
    int proxyHeight,
    int pixelFormat,
    void* destinationTexture,
    void* sharedEvent,
    std::uint64_t readyValue,
    void* metalCommandQueue,
    SourceProxyCompletionCallback completion,
    void* completionContext,
    std::string* error = nullptr);

#if defined(CHROMASPACE_MACOS_LEGACY_IOSURFACE_COMPAT)
bool copySourceProxyToIOSurface(
    const void* srcMetalBuffer,
    int sourceWidth,
    int sourceHeight,
    size_t srcRowBytes,
    int originX,
    int originY,
    int proxyWidth,
    int proxyHeight,
    int pixelFormat,
    void* metalCommandQueue,
    SharedSourceSignalSurface* out,
    std::string* error = nullptr);
#endif

bool copySourceRowsToHost(
    const void* srcMetalBuffer,
    int width,
    size_t srcRowBytes,
    const int* rows,
    int rowCount,
    int originX,
    int originY,
    void* metalCommandQueue,
    float* readbackSrc,
    size_t readbackSrcRowBytes);

}  // namespace ChromaspaceMetal
