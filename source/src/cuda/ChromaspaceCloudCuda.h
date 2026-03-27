#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace ChromaspaceCloudCuda {

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

}  // namespace ChromaspaceCloudCuda
