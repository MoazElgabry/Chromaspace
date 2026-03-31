#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceMetal {

struct ProbeResult {
  bool available = false;
  bool queueReady = false;
  const char* deviceName = "";
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
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  float colorSaturation = 1.18f;
  RemapUniforms remap;
};

struct InputSampleRequest {
  int fullPointCount = 0;
  int visiblePointCount = 0;
};

enum ModifierFlags : uint32_t {
  ModifierFlagShift = 1u << 0,
  ModifierFlagControl = 1u << 1,
  ModifierFlagAlt = 1u << 2,
  ModifierFlagSuper = 1u << 3,
};

bool activateWindow(void* nativeWindow);
uint32_t currentModifierFlags();
ProbeResult probe();
bool buildOverlayMesh(const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      std::vector<float>* outVerts,
                      std::vector<float>* outColors,
                      std::string* error);
bool buildInputMesh(const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    std::vector<float>* outVerts,
                    std::vector<float>* outColors,
                    std::string* error);
bool buildInputSampledMesh(const InputSampleRequest& request,
                           const std::vector<float>& fullVerts,
                           const std::vector<float>& fullColors,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error);

}  // namespace ChromaspaceMetal
