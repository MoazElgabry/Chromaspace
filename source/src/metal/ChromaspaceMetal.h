#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceMetal {

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

bool copyHostBuffersReadback(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    float* readbackSrc,
    size_t readbackSrcRowBytes);

}  // namespace ChromaspaceMetal
