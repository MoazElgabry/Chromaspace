#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ChromaspaceCuda {

struct ProbeResult {
  bool available = false;
  bool interopReady = false;
  const char* deviceName = "";
  const char* reason = "";
};

struct StartupValidationResult {
  bool ready = false;
  std::string reason;
};

struct RemapUniforms {
  int plotMode = 0;
  int circularHsl = 0;
  int circularHsv = 0;
  int normConeNormalized = 1;
  int showOverflow = 0;
  int highlightOverflow = 1;
  int chromaticityInputTransfer = 0;
  int chromaticityReferenceBasis = 0;
  float chromaticityWhiteX = 1.0f / 3.0f;
  float chromaticityWhiteY = 1.0f / 3.0f;
  float chromaticityRgbToXyz[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float chromaticityXyzToRgb[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
};

struct OverlayRequest {
  int cubeSize = 25;
  int ramp = 0;
  int useInputPoints = 0;
  int pointCount = 0;
  float colorSaturation = 1.18f;
  RemapUniforms remap;
};

struct InputRequest {
  int pointCount = 0;
  int inputStride = 3;
  int glossView = 0;
  float sourceAspect = 16.0f / 9.0f;
  float glossLiftScale = 1.0f;
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  float colorSaturation = 1.18f;
  RemapUniforms remap;
};

struct RasterSourceRequest {
  int pointCount = 0;
  int basePointCount = 0;
  int sourceWidth = 0;
  int sourceHeight = 0;
  int sampleStride = 1;
  int sampleCountX = 0;
  int pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  float sourceAspect = 16.0f / 9.0f;
  float glossLiftScale = 1.0f;
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  float colorSaturation = 1.18f;
  int plotLinear = 0;
  int plotLinearTransfer = 0;
  int excludeIdentityData = 0;
  int isolateIdentityData = 0;
  int readIdentityPlot = 0;
  int readGrayRamp = 0;
  int identityCubeY1 = -1;
  int identityCubeY2 = -1;
  int identityRampY1 = -1;
  int identityRampY2 = -1;
  int identityCubeAppendOffset = 0;
  int identityCubeAppendCount = 0;
  int identityCubeAppendY1 = -1;
  int identityCubeAppendY2 = -1;
  int identityCubeAppendRowStep = 1;
  int identityCubeAppendXStep = 1;
  int identityRampAppendOffset = 0;
  int identityRampAppendCount = 0;
  int identityRampAppendY1 = -1;
  int identityRampAppendY2 = -1;
  int identityRampAppendRowStep = 1;
  int identityRampAppendXStep = 1;
  int occupancyFill = 0;
  int occupancyAppendOffset = 0;
  int occupancyAppendCount = 0;
  int occupancyCandidateCount = 0;
  int lassoEnabled = 0;
  int lassoStrokeCount = 0;
  int lassoPointCount = 0;
  int lassoStrokeFirst[16] = {};
  int lassoStrokeCountPerStroke[16] = {};
  int lassoStrokeSubtract[16] = {};
  float lassoX[256] = {};
  float lassoY[256] = {};
  int cubeSlicingEnabled = 0;
  int neutralRadiusEnabled = 0;
  float neutralRadius = 1.0f;
  int cubeSliceRed = 0;
  int cubeSliceYellow = 0;
  int cubeSliceGreen = 0;
  int cubeSliceCyan = 0;
  int cubeSliceBlue = 0;
  int cubeSliceMagenta = 0;
  RemapUniforms remap;
};

struct InputSampleRequest {
  int fullPointCount = 0;
  int visiblePointCount = 0;
};

struct GlossFieldRequest {
  int gridWidth = 96;
  int gridHeight = 96;
  int showOverflow = 0;
  int neighborhoodChoice = 1;
};

struct GlossFieldResult {
  int gridWidth = 0;
  int gridHeight = 0;
  std::vector<float> occupancy;
  std::vector<float> meanRgb;
  std::vector<float> carrierY;
  std::vector<float> carrierMax;
  std::vector<float> carrierMin;
  std::vector<float> neutrality;
  std::vector<float> body;
  std::vector<float> signal;
  std::vector<float> positive;
  std::vector<float> negative;
  std::vector<float> boundary;
  std::vector<float> congruence;
  std::vector<float> confidence;
};

struct ScopeDensityRequest {
  int pointCount = 0;
  int waveform = 1;
  int scopeMode = 0;
  int width = 768;
  int height = 512;
  float rangeMin = 0.0f;
  float invRange = 1.0f;
  int excludeOverflow = 1;
};

struct OverlayCache {
  unsigned int verts = 0;
  unsigned int colors = 0;
  unsigned long long builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

struct InputCache {
  unsigned int verts = 0;
  unsigned int colors = 0;
  unsigned long long builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  bool hasFitBounds = false;
  float fitMin[3] = {0.0f, 0.0f, 0.0f};
  float fitMax[3] = {0.0f, 0.0f, 0.0f};
  void* internal = nullptr;
};

struct InputSampleCache {
  unsigned int verts = 0;
  unsigned int colors = 0;
  unsigned long long builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

ProbeResult probe();
StartupValidationResult warmupRuntime();
StartupValidationResult validateStartup();
void releaseOverlayCache(OverlayCache* cache);
void releaseInputCache(InputCache* cache);
void releaseInputSampleCache(InputSampleCache* cache);
bool buildOverlayMesh(OverlayCache* cache,
                      const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      unsigned long long serial,
                      std::string* error);
bool buildInputMesh(InputCache* cache,
                    const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    unsigned long long serial,
                    std::string* error);
bool buildRasterSourceMesh(InputCache* cache,
                           const RasterSourceRequest& request,
                           const void* sourceBytes,
                           size_t sourceByteCount,
                           unsigned long long serial,
                           std::string* error);
bool buildInputSampledMesh(InputCache* sourceCache,
                           InputSampleCache* sampleCache,
                           const InputSampleRequest& request,
                           unsigned long long serial,
                           std::string* error);
bool buildGlossField(InputCache* cache,
                     const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     GlossFieldResult* out,
                     std::string* error);
bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       std::vector<float>* outDensity,
                       std::string* error);

}  // namespace ChromaspaceCuda
