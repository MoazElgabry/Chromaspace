#pragma once

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

}  // namespace ChromaspaceCuda
