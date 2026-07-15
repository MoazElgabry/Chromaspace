#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace ChromaspaceCloudCuda {

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

struct OccupancyCandidate {
  Sample sample{};
  float normalizedNeutralRadius = 0.0f;
  int bin = 0;
  uint32_t tie = 0;
};

struct Request {
  const float* srcBase = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
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
  cudaStream_t stream = nullptr;
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
  const float* srcBase = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
  int candidateY1[2] = {};
  int candidateHeight[2] = {};
  cudaStream_t stream = nullptr;
};

struct RampLayoutResult {
  float scores[2] = {};
  int selectedCandidate = 0;
  std::string error;
  bool success = false;
};

bool detectGrayRampLayout(const RampLayoutRequest& request, RampLayoutResult* out);

struct StripRequest {
  const float* srcBase = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
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
  cudaStream_t stream = nullptr;
};

struct StripResult {
  std::vector<Sample> samples;
  std::string error;
  bool success = false;
};

bool buildIdentityStripCloud(const StripRequest& request, StripResult* out);

struct SourceProxyRequest {
  const float* srcBase = nullptr;
  std::size_t srcRowBytes = 0;
  void* outputDevicePtr = nullptr;
  std::size_t outputByteSize = 0;
  int sampledWidth = 0;
  int sampledHeight = 0;
  int proxyWidth = 0;
  int proxyHeight = 0;
  int exportIpc = 0;
  int readback = 1;
  cudaStream_t stream = nullptr;
};

struct SourceProxyResult {
  std::vector<float> rgba32f;
  std::size_t rowBytes = 0;
  void* devicePtr = nullptr;
  std::size_t byteSize = 0;
  cudaIpcMemHandle_t ipcHandle{};
  int hasIpcHandle = 0;
  std::string error;
  bool success = false;
};

bool buildSourceSignalProxy(const SourceProxyRequest& request, SourceProxyResult* out);

}  // namespace ChromaspaceCloudCuda
