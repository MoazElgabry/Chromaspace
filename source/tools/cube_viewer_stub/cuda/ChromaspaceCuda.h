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
  RemapUniforms remap;
};

struct InputRequest {
  int pointCount = 0;
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  RemapUniforms remap;
};

struct InputSampleRequest {
  int fullPointCount = 0;
  int visiblePointCount = 0;
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

}  // namespace ChromaspaceCuda
