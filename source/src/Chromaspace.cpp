#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <optional>
#include <random>
#include <array>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <csignal>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <spawn.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "ofxKeySyms.h"
#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsParam.h"
#include "color/ColorManagement.h"

#if defined(CHROMASPACE_HAS_CUDA)
#include <cuda_runtime_api.h>
#if defined(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
#include "cuda/ChromaspaceCloudCuda.h"
#endif
#endif

#if defined(CHROMASPACE_HAS_OPENCL)
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#if __has_include(<CL/cl.h>)
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif
#endif

#if defined(__APPLE__)
#include "metal/ChromaspaceMetal.h"
#endif

namespace {

using namespace OFX;

constexpr const char* kPluginIdentifier = "com.moazelgabry.chromaspace";
constexpr const char* kPluginGrouping = "Moaz Elgabry";
constexpr int kPluginVersionMajor = 1;
constexpr int kPluginVersionMinor = 5;
constexpr const char* kPluginVersionLabel = "v1.0.6Beta";
constexpr const char* kPluginName = "Chromaspace";
constexpr const char* kWebsiteUrl = "https://moazelgabry.com";
constexpr const char* kReleasesUrl = "https://github.com/MoazElgabry/Chromaspace/releases/latest";
constexpr const char* kIssueUrl = "https://github.com/MoazElgabry/Chromaspace/issues";
constexpr const char* kPluginManagerProductName = "Moaz Elgabry Plugins";
constexpr const char* kPluginManagerBundleId = "com.moazelgabry.pluginmanager";

std::string cubeViewerLogPath() {
#if defined(_WIN32)
  const char* localAppData = std::getenv("LOCALAPPDATA");
  if (localAppData && localAppData[0] != '\0') {
    return (std::filesystem::path(localAppData) / "Chromaspace.log").string();
  }
  return "Chromaspace.log";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/Chromaspace.log";
  return (std::filesystem::path(home) / "Library" / "Logs" / "Chromaspace.log").string();
#else
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/Chromaspace.log";
  return (std::filesystem::path(home) / ".cache" / "Chromaspace.log").string();
#endif
}

std::string parentDir(const std::string& path) {
  const size_t p = path.find_last_of("/\\");
  if (p == std::string::npos) return std::string();
  return path.substr(0, p);
}

std::string filenameOnly(const std::string& path) {
  const size_t p = path.find_last_of("/\\");
  if (p == std::string::npos) return path;
  return path.substr(p + 1);
}

std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  const char last = a.back();
  if (last == '/' || last == '\\') return a + b;
#if defined(_WIN32)
  return a + "\\" + b;
#else
  return a + "/" + b;
#endif
}

bool isAbsolutePath(const std::string& p) {
  if (p.empty()) return false;
#if defined(_WIN32)
  if (p.size() >= 2 && std::isalpha(static_cast<unsigned char>(p[0])) && p[1] == ':') return true;
  if (p.size() >= 2 && p[0] == '\\' && p[1] == '\\') return true;
  return false;
#else
  return p[0] == '/';
#endif
}

bool fileExistsForLaunch(const std::string& p) {
  if (p.empty()) return false;
#if defined(_WIN32)
  const DWORD attrs = GetFileAttributesA(p.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) return false;
  return (attrs & FILE_ATTRIBUTE_DIRECTORY) == 0;
#else
  return ::access(p.c_str(), X_OK) == 0;
#endif
}

std::string findBundleRootFromModule(const std::string& modulePath) {
  std::string current = parentDir(modulePath);
  while (!current.empty()) {
    const std::string name = filenameOnly(current);
    if (name.size() >= 11 && name.find(".ofx.bundle") != std::string::npos) {
      return current;
    }
    const std::string next = parentDir(current);
    if (next == current) break;
    current = next;
  }
  return std::string();
}

std::string pluginModulePath() {
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&pluginModulePath),
          &module)) {
    return std::string();
  }
  char buf[MAX_PATH] = {0};
  const DWORD n = GetModuleFileNameA(module, buf, static_cast<DWORD>(sizeof(buf)));
  if (n == 0 || n >= sizeof(buf)) return std::string();
  return std::string(buf, n);
#else
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&pluginModulePath), &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  return std::string(info.dli_fname);
#endif
}

bool cubeViewerDebugEnabled() {
  const char* direct = std::getenv("CHROMASPACE_DEBUG_LOG");
  if (direct && direct[0] != '\0' && std::strcmp(direct, "0") != 0) return true;
  const char* multi = std::getenv("CHROMASPACE_MULTI_INSTANCE_DEBUG");
  if (multi && multi[0] != '\0' && std::strcmp(multi, "0") != 0) return true;
  return false;
}

bool cubeViewerMultiInstanceDebugEnabled() {
  const char* direct = std::getenv("CHROMASPACE_MULTI_INSTANCE_DEBUG");
  if (direct && direct[0] != '\0' && std::strcmp(direct, "0") != 0) return true;
  return false;
}

void cubeViewerDebugLog(const std::string& msg) {
  if (!cubeViewerDebugEnabled()) return;
  const std::string path = cubeViewerLogPath();
  std::error_code ec;
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
  }
  FILE* f = std::fopen(path.c_str(), "a");
  if (!f) return;
  std::fprintf(f, "[Chromaspace] %s\n", msg.c_str());
  std::fclose(f);
}

void cubeViewerMultiInstanceDebugLog(const std::string& msg) {
  if (!cubeViewerMultiInstanceDebugEnabled()) return;
  cubeViewerDebugLog(std::string("[multi] ") + msg);
}

struct ViewerCloudTransportBlob;

struct PendingMessage {
  std::string reason;
  std::string payload;
  std::shared_ptr<ViewerCloudTransportBlob> keepAliveBlob;
  bool valid = false;
};

struct ViewerCloudSample {
  float xNorm = 0.0f;
  float yNorm = 0.0f;
  float zReserved = 0.0f;
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
};

struct ViewerCloudCandidate {
  ViewerCloudSample sample{};
  float normalizedNeutralRadius = 0.0f;
  int bin = 0;
  uint32_t tie = 0;
};

struct ViewerCloudTransportBlob {
  std::string name;
  std::size_t byteSize = 0;
#if defined(_WIN32)
  HANDLE mappingHandle = nullptr;
#else
  int fd = -1;
#endif

  ~ViewerCloudTransportBlob() {
#if defined(_WIN32)
    if (mappingHandle != nullptr) {
      CloseHandle(mappingHandle);
      mappingHandle = nullptr;
    }
#else
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
    if (!name.empty()) {
      shm_unlink(name.c_str());
    }
#endif
  }
};

struct CachedCloud {
  std::string payload;
  std::string pointsPayload;
  std::string paramHash;
  std::string quality;
  std::string sourceId;
  std::string settingsKey;
  std::shared_ptr<ViewerCloudTransportBlob> fastBlob;
  std::vector<ViewerCloudSample> samples;
  uint64_t contentHash = 0;
  std::size_t sampleCount = 0;
  int resolution = 25;
  int sourceWidth = 0;
  int sourceHeight = 0;
  bool valid = false;
};

struct CachedIdentityStripCloud {
  std::vector<ViewerCloudSample> samples;
  std::string cacheKey;
  std::string paramHash;
  int resolution = 25;
  bool valid = false;
};

struct CloudBuildResult {
  std::string payload;
  std::string pointsPayload;
  std::string paramHash;
  std::string quality;
  std::shared_ptr<ViewerCloudTransportBlob> fastBlob;
  std::vector<ViewerCloudSample> samples;
  std::string backendName = "CPU";
  uint64_t contentHash = 0;
  std::size_t sampleCount = 0;
  int resolution = 25;
  int sourceWidth = 0;
  int sourceHeight = 0;
  bool success = false;
};

struct ViewerCloudBuildRequest {
  const float* srcBase = nullptr;
  std::size_t srcRowBytes = 0;
  int width = 0;
  int height = 0;
  double time = 0.0;
  bool previewMode = false;
  int qualityIndex = 0;
  int samplingMode = 0;
  int scaleIndex = 3;
  double scaleFactor = 1.0;
  int resolution = 25;
  bool preserveOverflow = false;
  bool occupancyFill = false;
  bool plotDisplayLinearEnabled = false;
  int plotDisplayLinearTransferId = 0;
  std::string plotMode;
  std::string settingsKey;
};

// Identity-plot recognition was previously implemented here as a heuristic gate for
// "Use Identity Plot". It has been intentionally removed from processing so that
// Chromaspace always treats the toggle as an explicit user decision.

struct OverlayStripData {
  int x1 = 0;
  int y1 = 0;
  int width = 0;
  int height = 0;
  std::vector<float> pixels;
};

struct ViewerProbeResult {
  bool ok = false;
  bool visible = true;
  bool iconified = false;
  bool focused = true;
};

bool sameViewerProbeState(const ViewerProbeResult& a, const ViewerProbeResult& b) {
  return a.ok == b.ok && a.visible == b.visible && a.iconified == b.iconified && a.focused == b.focused;
}

enum class VolumeSlicingMode {
  HueSectors = 0,
  LassoRegion = 1,
};

enum class ViewerUpdateMode {
  Auto = 0,
  Fluid = 1,
  Scheduled = 2,
};

struct LassoPointNorm {
  float xNorm = 0.0f;
  float yNorm = 0.0f;
};

struct LassoStroke {
  bool subtract = false;
  std::vector<LassoPointNorm> points;
};

struct LassoRegionState {
  uint64_t revision = 0;
  std::vector<LassoStroke> strokes;

  bool empty() const {
    return strokes.empty();
  }
};

std::vector<std::string> splitString(const std::string& text, char delimiter) {
  std::vector<std::string> parts;
  std::string current;
  for (const char c : text) {
    if (c == delimiter) {
      parts.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  parts.push_back(current);
  return parts;
}

std::string formatSerializedFloat(float value) {
  char buffer[32];
  const int n = std::snprintf(buffer, sizeof(buffer), "%.6f", static_cast<double>(value));
  return n > 0 ? std::string(buffer, static_cast<size_t>(n)) : std::string("0");
}

bool parseUint64(const std::string& text, uint64_t* value) {
  if (!value) return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool parseIntStrict(const std::string& text, int* value) {
  if (!value) return false;
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool parseFloatStrict(const std::string& text, float* value) {
  if (!value) return false;
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (!end || *end != '\0') return false;
  *value = parsed;
  return true;
}

std::string serializeLassoRegionState(const LassoRegionState& state) {
  std::ostringstream oss;
  oss << "v1|" << state.revision;
  for (const auto& stroke : state.strokes) {
    oss << "|" << (stroke.subtract ? 's' : 'a') << "," << stroke.points.size();
    for (const auto& point : stroke.points) {
      oss << "," << formatSerializedFloat(point.xNorm)
          << "," << formatSerializedFloat(point.yNorm);
    }
  }
  return oss.str();
}

LassoRegionState parseLassoRegionState(const std::string& serialized) {
  LassoRegionState state{};
  if (serialized.empty()) return state;
  const auto records = splitString(serialized, '|');
  if (records.size() < 2 || records[0] != "v1") return state;
  if (!parseUint64(records[1], &state.revision)) return LassoRegionState{};
  for (size_t i = 2; i < records.size(); ++i) {
    const auto fields = splitString(records[i], ',');
    if (fields.size() < 2 || fields[0].size() != 1) return LassoRegionState{};
    int pointCount = 0;
    if (!parseIntStrict(fields[1], &pointCount) || pointCount < 3) return LassoRegionState{};
    if (fields.size() != static_cast<size_t>(2 + pointCount * 2)) return LassoRegionState{};
    LassoStroke stroke{};
    stroke.subtract = (fields[0][0] == 's' || fields[0][0] == 'S');
    stroke.points.reserve(static_cast<size_t>(pointCount));
    for (int pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
      float xNorm = 0.0f;
      float yNorm = 0.0f;
      if (!parseFloatStrict(fields[2 + pointIndex * 2], &xNorm) ||
          !parseFloatStrict(fields[3 + pointIndex * 2], &yNorm)) {
        return LassoRegionState{};
      }
      stroke.points.push_back({std::clamp(xNorm, 0.0f, 1.0f), std::clamp(yNorm, 0.0f, 1.0f)});
    }
    state.strokes.push_back(std::move(stroke));
  }
  return state;
}

bool pointInPolygonNormalized(const std::vector<LassoPointNorm>& polygon, double xNorm, double yNorm) {
  if (polygon.size() < 3) return false;
  bool inside = false;
  const size_t count = polygon.size();
  for (size_t i = 0, j = count - 1; i < count; j = i++) {
    const double xi = polygon[i].xNorm;
    const double yi = polygon[i].yNorm;
    const double xj = polygon[j].xNorm;
    const double yj = polygon[j].yNorm;
    const bool intersects = ((yi > yNorm) != (yj > yNorm)) &&
                            (xNorm < (xj - xi) * (yNorm - yi) / ((yj - yi) + 1e-12) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

bool lassoRegionContainsPoint(const LassoRegionState& state, double xNorm, double yNorm) {
  bool inside = false;
  for (const auto& stroke : state.strokes) {
    if (!pointInPolygonNormalized(stroke.points, xNorm, yNorm)) continue;
    inside = !stroke.subtract;
  }
  return inside;
}

#if defined(_WIN32)
HANDLE openViewerPipeHandle(const std::string& pipe) {
  for (int attempt = 0; attempt < 12; ++attempt) {
    HANDLE pipeHandle = CreateFileA(pipe.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (pipeHandle != INVALID_HANDLE_VALUE) {
      DWORD mode = PIPE_READMODE_BYTE;
      SetNamedPipeHandleState(pipeHandle, &mode, nullptr, nullptr);
      return pipeHandle;
    }
    const DWORD err = GetLastError();
    if (err != ERROR_PIPE_BUSY) {
      cubeViewerDebugLog(std::string("CreateFile failed err=") + std::to_string(err));
      return INVALID_HANDLE_VALUE;
    }
    if (!WaitNamedPipeA(pipe.c_str(), 150)) {
      cubeViewerDebugLog(std::string("WaitNamedPipe timeout err=") + std::to_string(GetLastError()));
      break;
    }
  }
  return INVALID_HANDLE_VALUE;
}
#endif

inline float clamp01(float v) {
  return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Sampling presets stay intentionally sparse because the viewer shows a cloud, not a full voxel volume.
// These values are the baseline lattice sizes that the rest of the quality heuristics build from.
int qualityResolutionForIndex(int q) {
  switch (q) {
    case 0: return 25;
    case 1: return 41;
    default: return 57;
  }
}

// Point-count budgets are lower than full cube sizes so interactive updates remain usable at host playback rates.
int qualityPointCountForIndex(int q, bool previewMode) {
  (void)previewMode;
  switch (q) {
    case 0: return 30000;
    case 1: return 60000;
    default: return 110000;
  }
}

int clampOverlayCubeSize(int size) {
  return std::max(4, std::min(65, size));
}

double scaleFactorForIndex(int idx) {
  switch (idx) {
    case 0: return 0.25;
    case 1: return 0.50;
    case 2: return 0.75;
    default: return 1.00;
  }
}

const char* scaleLabelForIndex(int idx) {
  switch (idx) {
    case 0: return "25%";
    case 1: return "50%";
    case 2: return "75%";
    default: return "100%";
  }
}

// Overlay size 25 acts as "Auto": resolve to something that visually fills the solid while respecting
// the current quality/scale point budget instead of blindly forcing a dense identity cube every time.
int resolvedOverlayCubeSize(int requestedSize, int qualityIndex, int scaleIndex) {
  const int clampedRequested = clampOverlayCubeSize(requestedSize);
  if (clampedRequested != 25) return clampedRequested;
  const double scaleFactor = scaleFactorForIndex(scaleIndex);
  const int pointBudget = std::max(512, static_cast<int>(std::lround(
      static_cast<double>(qualityPointCountForIndex(qualityIndex, false)) * scaleFactor * scaleFactor)));
  const int budgetLimited = std::max(4, static_cast<int>(std::floor(std::cbrt(static_cast<double>(pointBudget)))));
  return clampOverlayCubeSize(std::min(qualityResolutionForIndex(qualityIndex), budgetLimited));
}

float identityStripCellWidth(int imageWidth, int cubeSize) {
  return cubeSize <= 0 ? 1.0f : std::max(1.0f, static_cast<float>(imageWidth) / static_cast<float>(cubeSize));
}

// Stage: map the synthetic draw-on-image strip into image-space bands.
// The cube always occupies the bottom band; the optional ramp sits directly above it.
bool computeIdentityStripLayout(const OfxRectI& bounds,
                                int cubeSize,
                                bool overlayRamp,
                                int* stripHeight,
                                int* cubeY1,
                                int* cubeY2,
                                int* rampY1,
                                int* rampY2) {
  const int imageWidth = bounds.x2 - bounds.x1;
  const int imageHeight = bounds.y2 - bounds.y1;
  if (imageWidth <= 0 || imageHeight <= 0 || cubeSize <= 0) return false;
  const float cellWidth = identityStripCellWidth(imageWidth, cubeSize);
  const int strip = std::max(1, static_cast<int>(std::lround(cellWidth)));
  const int totalStripHeight = std::min(imageHeight, strip * (overlayRamp ? 2 : 1));
  const int baseY1 = bounds.y1;
  const int baseY2 = std::min(bounds.y2, baseY1 + totalStripHeight);
  if (overlayRamp) {
    if (cubeY1) *cubeY1 = baseY1;
    if (cubeY2) *cubeY2 = std::min(baseY2, baseY1 + strip);
    if (rampY1) *rampY1 = std::min(baseY2, baseY1 + strip);
    if (rampY2) *rampY2 = std::min(baseY2, baseY1 + strip * 2);
  } else {
    if (rampY1) *rampY1 = baseY1;
    if (rampY2) *rampY2 = baseY1;
    if (cubeY1) *cubeY1 = baseY1;
    if (cubeY2) *cubeY2 = baseY2;
  }
  if (stripHeight) *stripHeight = strip;
  return (cubeY1 == nullptr || cubeY2 == nullptr || *cubeY2 > *cubeY1);
}

const char* qualityLabelForIndex(int q) {
  switch (q) {
    case 0: return "Low";
    case 1: return "Medium";
    default: return "High";
  }
}

float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

int getSamplingModeValue(int index, int fallback = 0) {
  return index < 0 || index > 2 ? fallback : index;
}

int getPointShapeValue(int index, int fallback = 0) {
  return index < 0 || index > 1 ? fallback : index;
}

int getPlotStyleValue(int index, int fallback = 0) {
  return index < 0 || index > 1 ? fallback : index;
}

const char* plotStyleLabelForIndex(int index) {
  switch (getPlotStyleValue(index)) {
    case 1: return "Space";
    default: return "Plain Scope";
  }
}

const char* pointShapeLabelForIndex(int index) {
  switch (getPointShapeValue(index)) {
    case 1: return "Square";
    default: return "Circle";
  }
}

const char* samplingModeLabelForIndex(int index) {
  switch (getSamplingModeValue(index)) {
    case 1: return "Stratified";
    case 2: return "Random";
    default: return "Balanced";
  }
}

// Larger splats need fewer points to stay readable, while smaller splats need a denser cloud to avoid thinning out.
// The exponent keeps the compensation gentle rather than making point size feel like a hidden quality control.
float derivedDensityScaleForPointSize(double pointSize) {
  const float size = clampf(static_cast<float>(pointSize), 0.35f, 3.0f);
  if (size <= 1.0f) {
    return clampf(std::pow(0.6f / size, 0.6f), 0.85f, 1.3f);
  }
  const float t = clampf((size - 1.0f) / 2.0f, 0.0f, 1.0f);
  return clampf(0.85f + 0.20f * std::pow(t, 0.75f), 0.85f, 1.05f);
}

const char* viewerUpdateModeLabelForIndex(int mode) {
  switch (mode) {
    case 1: return "Fluid";
    case 2: return "Scheduled";
    default: return "Auto";
  }
}

double halton(uint32_t index, uint32_t base) {
  double f = 1.0;
  double r = 0.0;
  while (index > 0) {
    f /= static_cast<double>(base);
    r += f * static_cast<double>(index % base);
    index /= base;
  }
  return r;
}

uint32_t hash32(uint32_t v) {
  v ^= v >> 16;
  v *= 0x7feb352dU;
  v ^= v >> 15;
  v *= 0x846ca68bU;
  v ^= v >> 16;
  return v;
}

double unitHash01(uint32_t v) {
  return static_cast<double>(hash32(v)) / static_cast<double>(0xffffffffU);
}

enum class SlicePlotModeKind {
  Rgb = 0,
  Hsl = 1,
  Hsv = 2,
  Chen = 3,
  RgbToCone = 4,
  JpConical = 5,
  NormCone = 6,
  Reuleaux = 7,
};

struct SliceSelectionSpec {
  SlicePlotModeKind plotMode = SlicePlotModeKind::Rgb;
  bool showOverflow = false;
  bool circularHsl = false;
  bool circularHsv = false;
  bool normConeNormalized = true;
  bool enabled = false;
  bool neutralRadiusEnabled = false;
  float neutralRadius = 1.0f;
  bool cubeSliceRed = true;
  bool cubeSliceGreen = false;
  bool cubeSliceBlue = false;
  bool cubeSliceCyan = false;
  bool cubeSliceYellow = false;
  bool cubeSliceMagenta = false;
};

struct NormalizedBounds {
  double x1 = 0.0;
  double y1 = 0.0;
  double x2 = 1.0;
  double y2 = 1.0;
  bool valid = false;
};

float wrapHue01(float h) {
  h = std::fmod(h, 1.0f);
  if (h < 0.0f) h += 1.0f;
  return h;
}

float effectiveNeutralRadiusThreshold(float sliderValue) {
  constexpr float kNeutralRadiusResponsePower = 2.0f;
  return clampf(std::pow(clampf(sliderValue, 0.0f, 1.0f), kNeutralRadiusResponsePower), 0.0f, 1.0f);
}

float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6f) return 0.0f;
  float h = 0.0f;
  if (cMax == r) {
    h = std::fmod((g - b) / delta, 6.0f);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0f;
  } else {
    h = ((r - g) / delta) + 4.0f;
  }
  return wrapHue01(h / 6.0f);
}

void rgbToHsvHexconePlaneSlice(float r, float g, float b, float* outX, float* outZ) {
  *outX = r - 0.5f * g - 0.5f * b;
  *outZ = 0.8660254037844386f * (g - b);
}

void rgbToPlotCircularHslSlice(float r, float g, float b, float* outH, float* outRadius, float* outL) {
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  const float l = 0.5f * (cMax + cMin);
  float h = rawRgbHue01(r, g, b, cMax, delta);
  float satDenom = 1.0f - std::fabs(2.0f * l - 1.0f);
  if (delta > 1e-6f && satDenom < 0.0f) {
    h = wrapHue01(h + 0.5f);
  }
  if (std::fabs(satDenom) <= 1e-6f) {
    satDenom = satDenom < 0.0f ? -1e-6f : 1e-6f;
  }
  *outH = h;
  *outRadius = std::fabs(delta / satDenom);
  *outL = l;
}

void rgbToPlotCircularHsvSlice(float r, float g, float b, float* outH, float* outRadius, float* outV) {
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  *outH = rawRgbHue01(r, g, b, cMax, delta);
  *outRadius = (delta > 1e-6f && cMax > 1e-6f) ? (delta / cMax) : 0.0f;
  *outV = cMax;
}

void rgbToChenSlice(float r, float g, float b, bool allowOverflow, float* outHue, float* outChroma, float* outLight) {
  constexpr float kTau = 6.28318530717958647692f;
  if (!allowOverflow) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  const float rotX = r * 0.81649658f + g * -0.40824829f + b * -0.40824829f;
  const float rotY = r * 0.0f + g * 0.70710678f + b * -0.70710678f;
  const float rotZ = r * 0.57735027f + g * 0.57735027f + b * 0.57735027f;
  const float azimuth = std::atan2(rotY, rotX);
  const float radius = std::sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  const float wrappedHue = azimuth < 0.0f ? azimuth + kTau : azimuth;
  const float polar = std::atan2(std::sqrt(rotX * rotX + rotY * rotY), rotZ);
  *outHue = wrappedHue / kTau;
  *outChroma = polar * 1.0467733744265997f;
  *outLight = radius * 0.5773502691896258f;
}

void rgbToNormConeCoordsSlice(float r, float g, float b, bool normalized, bool allowOverflow, float* outHue, float* outChroma, float* outValue) {
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kSphericalMax = 0.9553166181245093f;
  const float maxRgb = std::max(r, std::max(g, b));
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  float hue = std::atan2(rotY, rotX) / kTau;
  if (hue < 0.0f) hue += 1.0f;
  const float chromaRadius = std::sqrt(rotX * rotX + rotY * rotY);
  const float polar = std::atan2(chromaRadius, rotZ);
  float chroma = polar / kSphericalMax;
  if (normalized) {
    const float angle = hue * kTau - 0.52359877559829887308f;
    const float cosPolar = std::cos(polar);
    const float safeCos = std::abs(cosPolar) > 1e-6f ? cosPolar : (cosPolar < 0.0f ? -1e-6f : 1e-6f);
    const float cone = (std::sin(polar) / safeCos) / std::sqrt(2.0f);
    const float sinTerm = clampf(std::sin(3.0f * angle), -1.0f, 1.0f);
    const float chromaGain = 1.0f / (2.0f * std::cos(std::acos(sinTerm) / 3.0f));
    chroma = chromaGain > 1e-6f ? cone / chromaGain : 0.0f;
    if (allowOverflow && chroma < 0.0f) {
      chroma = -chroma;
      hue += 0.5f;
      if (hue >= 1.0f) hue -= 1.0f;
    }
  }
  *outHue = hue;
  *outChroma = allowOverflow ? std::max(chroma, 0.0f) : clamp01(chroma);
  *outValue = allowOverflow ? maxRgb : clamp01(maxRgb);
}

void rgbToRgbConeSlice(float r, float g, float b, float* outMagnitude, float* outHue, float* outPolar) {
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kPolarMax = 0.9553166181245093f;
  r = clamp01(r);
  g = clamp01(g);
  b = clamp01(b);
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  const float radius = std::sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  float hue = std::atan2(rotY, rotX);
  if (hue < 0.0f) hue += kTau;
  const float polar = std::atan2(std::sqrt(rotX * rotX + rotY * rotY), rotZ);
  *outMagnitude = clamp01(radius * 0.576f);
  *outHue = hue / kTau;
  *outPolar = clamp01(polar / kPolarMax);
}

void rgbToJpConicalSlice(float r, float g, float b, bool allowOverflow, float* outMagnitude, float* outHue, float* outPolar) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kPolarMax = 0.9553166181245093f;
  const float kAsinInvSqrt2 = std::asin(1.0f / std::sqrt(2.0f));
  const float kAsinInvSqrt3 = std::asin(1.0f / std::sqrt(3.0f));
  const float kHueCoef1 = 1.0f / (2.0f - (kAsinInvSqrt2 / kAsinInvSqrt3));
  if (!allowOverflow) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  const float rotX = 0.81649658093f * r - 0.40824829046f * g - 0.40824829046f * b;
  const float rotY = 0.70710678118f * g - 0.70710678118f * b;
  const float rotZ = 0.57735026919f * (r + g + b);
  const float radius = std::sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
  float hue = std::atan2(rotY, rotX);
  if (hue < 0.0f) hue += kTau;
  const float polar = std::atan2(std::sqrt(rotX * rotX + rotY * rotY), rotZ);
  const float huecoef2 = 2.0f * polar * std::sin((2.0f * kPi / 3.0f) - std::fmod(hue, kPi / 3.0f)) / std::sqrt(3.0f);
  const float huemag = ((std::acos(std::cos(3.0f * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0f)) * huecoef2;
  const float satmag = std::sin(huemag + kAsinInvSqrt3);
  float magnitude = radius * satmag;
  if (allowOverflow && magnitude < 0.0f) {
    magnitude = -magnitude;
    hue += kPi;
    if (hue >= kTau) hue -= kTau;
  }
  *outMagnitude = allowOverflow ? magnitude : clamp01(magnitude);
  *outHue = hue / kTau;
  *outPolar = allowOverflow ? std::max(polar / kPolarMax, 0.0f) : clamp01(polar / kPolarMax);
}

void rgbToReuleauxSlice(float r, float g, float b, bool allowOverflow, float* outHue, float* outSat, float* outValue) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kMaxSat = 1.41421356237f;
  if (!allowOverflow) {
    r = clamp01(r);
    g = clamp01(g);
    b = clamp01(b);
  }
  const float rotX = 0.33333333333f * (2.0f * r - g - b) * 0.70710678118f;
  const float rotY = (g - b) * 0.40824829046f;
  const float rotZ = (r + g + b) / 3.0f;
  float hue = kPi - std::atan2(rotY, -rotX);
  if (hue < 0.0f) hue += kTau;
  if (hue >= kTau) hue = std::fmod(hue, kTau);
  float sat = std::fabs(rotZ) <= 1e-6f ? 0.0f : std::hypot(rotX, rotY) / rotZ;
  if (allowOverflow && sat < 0.0f) {
    sat = -sat;
    hue += kPi;
    if (hue >= kTau) hue -= kTau;
  }
  *outHue = hue / kTau;
  *outSat = allowOverflow ? sat / kMaxSat : clampf(sat / kMaxSat, 0.0f, 1.0f);
  *outValue = allowOverflow ? std::max(r, std::max(g, b)) : clamp01(std::max(r, std::max(g, b)));
}

float normalizedNeutralRadiusForSlice(const SliceSelectionSpec& spec, float r, float g, float b) {
  constexpr float kRgbAxisMaxRadius = 0.8164965809277260f;
  constexpr float kPolarMax = 0.9553166181245093f;
  constexpr float kChenPolarScale = 1.0467733744265997f;

  switch (spec.plotMode) {
    case SlicePlotModeKind::Rgb: {
      const float rr = clamp01(r);
      const float gg = clamp01(g);
      const float bb = clamp01(b);
      const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
      const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
      return clampf(std::sqrt(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius, 0.0f, 1.0f);
    }
    case SlicePlotModeKind::Hsl: {
      if (spec.circularHsl) {
        float h = 0.0f;
        float radius = 0.0f;
        float l = 0.0f;
        rgbToPlotCircularHslSlice(r, g, b, &h, &radius, &l);
        return clampf(radius, 0.0f, 1.0f);
      }
      const float cMax = std::max(r, std::max(g, b));
      const float cMin = std::min(r, std::min(g, b));
      return clampf(cMax - cMin, 0.0f, 1.0f);
    }
    case SlicePlotModeKind::Hsv: {
      if (spec.circularHsv) {
        float h = 0.0f;
        float radius = 0.0f;
        float v = 0.0f;
        rgbToPlotCircularHsvSlice(r, g, b, &h, &radius, &v);
        return clampf(radius, 0.0f, 1.0f);
      }
      float x = 0.0f;
      float z = 0.0f;
      rgbToHsvHexconePlaneSlice(r, g, b, &x, &z);
      return clampf(std::hypot(x, z), 0.0f, 1.0f);
    }
    case SlicePlotModeKind::Chen: {
      float h = 0.0f;
      float chroma = 0.0f;
      float light = 0.0f;
      rgbToChenSlice(r, g, b, spec.showOverflow, &h, &chroma, &light);
      const float polar = chroma / kChenPolarScale;
      const float radius = light * std::sin(polar) / kRgbAxisMaxRadius;
      return clampf(radius, 0.0f, 1.0f);
    }
    case SlicePlotModeKind::RgbToCone: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToRgbConeSlice(r, g, b, &magnitude, &hue, &polar);
      const float radial = magnitude * std::sin(polar * kPolarMax);
      return clampf(radial / std::sin(kPolarMax), 0.0f, 1.0f);
    }
    case SlicePlotModeKind::JpConical: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToJpConicalSlice(r, g, b, spec.showOverflow, &magnitude, &hue, &polar);
      const float radial = magnitude * std::sin(polar * kPolarMax);
      return clampf(radial / std::sin(kPolarMax), 0.0f, 1.0f);
    }
    case SlicePlotModeKind::NormCone: {
      float hue = 0.0f;
      float chroma = 0.0f;
      float value = 0.0f;
      rgbToNormConeCoordsSlice(r, g, b, spec.normConeNormalized, spec.showOverflow, &hue, &chroma, &value);
      return clampf(chroma, 0.0f, 1.0f);
    }
    case SlicePlotModeKind::Reuleaux: {
      float hue = 0.0f;
      float sat = 0.0f;
      float value = 0.0f;
      rgbToReuleauxSlice(r, g, b, spec.showOverflow, &hue, &sat, &value);
      return clampf(sat, 0.0f, 1.0f);
    }
    default:
      return 0.0f;
  }
}

bool neutralRadiusContainsPoint(const SliceSelectionSpec& spec, float r, float g, float b) {
  if (!spec.neutralRadiusEnabled) return true;
  return normalizedNeutralRadiusForSlice(spec, r, g, b) <= effectiveNeutralRadiusThreshold(spec.neutralRadius) + 1e-6f;
}

float neutralRadiusSamplingAcceptanceProbability(const SliceSelectionSpec& spec, float normalizedRadius) {
  if (!spec.neutralRadiusEnabled) return 1.0f;
  const float threshold = effectiveNeutralRadiusThreshold(spec.neutralRadius);
  if (threshold <= 1e-6f) return normalizedRadius <= threshold + 1e-6f ? 1.0f : 0.0f;
  if (normalizedRadius > threshold + 1e-6f) return 0.0f;
  const float clippedFraction = clampf(1.0f - threshold, 0.0f, 1.0f);
  const float normalizedInside = clampf(normalizedRadius / threshold, 0.0f, 1.0f);
  const float edgePenalty = 0.78f * clippedFraction;
  return clampf(1.0f - edgePenalty * std::pow(normalizedInside, 1.35f), 0.0f, 1.0f);
}

bool neutralRadiusSamplingAcceptsPoint(const SliceSelectionSpec& spec,
                                       float r,
                                       float g,
                                       float b,
                                       uint32_t samplingSeed) {
  if (!spec.neutralRadiusEnabled) return true;
  const float normalizedRadius = normalizedNeutralRadiusForSlice(spec, r, g, b);
  const float acceptProbability = neutralRadiusSamplingAcceptanceProbability(spec, normalizedRadius);
  if (acceptProbability <= 0.0f) return false;
  if (acceptProbability >= 0.999999f) return true;
  return unitHash01(samplingSeed) <= static_cast<double>(acceptProbability);
}

bool volumeSliceContainsPoint(const SliceSelectionSpec& spec, float r, float g, float b) {
  if (!neutralRadiusContainsPoint(spec, r, g, b)) return false;
  const bool anyRegionSelected = spec.cubeSliceRed || spec.cubeSliceGreen || spec.cubeSliceBlue ||
                                 spec.cubeSliceCyan || spec.cubeSliceYellow || spec.cubeSliceMagenta;
  if (!spec.enabled) return true;
  if (!anyRegionSelected) return false;
  if (spec.plotMode == SlicePlotModeKind::Rgb) {
    constexpr float kEps = 1e-6f;
    const auto ge = [&](float a, float c) { return a + kEps >= c; };
    if (spec.cubeSliceRed && ge(r, g) && ge(g, b)) return true;
    if (spec.cubeSliceYellow && ge(g, r) && ge(r, b)) return true;
    if (spec.cubeSliceGreen && ge(g, b) && ge(b, r)) return true;
    if (spec.cubeSliceCyan && ge(b, g) && ge(g, r)) return true;
    if (spec.cubeSliceBlue && ge(b, r) && ge(r, g)) return true;
    if (spec.cubeSliceMagenta && ge(r, b) && ge(b, g)) return true;
    return false;
  }

  float hue = 0.0f;
  bool hueDefined = false;
  switch (spec.plotMode) {
    case SlicePlotModeKind::Hsl:
    case SlicePlotModeKind::Hsv: {
      const float cMax = std::max(r, std::max(g, b));
      const float cMin = std::min(r, std::min(g, b));
      const float delta = cMax - cMin;
      if (delta > 1e-6f) {
        hue = rawRgbHue01(r, g, b, cMax, delta);
        hueDefined = true;
      }
      break;
    }
    case SlicePlotModeKind::Chen: {
      float chroma = 0.0f;
      float light = 0.0f;
      rgbToChenSlice(r, g, b, spec.showOverflow, &hue, &chroma, &light);
      hueDefined = chroma > 1e-6f;
      break;
    }
    case SlicePlotModeKind::RgbToCone: {
      float magnitude = 0.0f;
      float polar = 0.0f;
      rgbToRgbConeSlice(r, g, b, &magnitude, &hue, &polar);
      hueDefined = magnitude > 1e-6f && polar > 1e-6f;
      break;
    }
    case SlicePlotModeKind::JpConical: {
      float magnitude = 0.0f;
      float polar = 0.0f;
      rgbToJpConicalSlice(r, g, b, spec.showOverflow, &magnitude, &hue, &polar);
      hueDefined = magnitude > 1e-6f && polar > 1e-6f;
      break;
    }
    case SlicePlotModeKind::NormCone: {
      float chroma = 0.0f;
      float value = 0.0f;
      rgbToNormConeCoordsSlice(r, g, b, spec.normConeNormalized, spec.showOverflow, &hue, &chroma, &value);
      hueDefined = chroma > 1e-6f;
      break;
    }
    case SlicePlotModeKind::Reuleaux: {
      float sat = 0.0f;
      float value = 0.0f;
      rgbToReuleauxSlice(r, g, b, spec.showOverflow, &hue, &sat, &value);
      hueDefined = sat > 1e-6f;
      break;
    }
    case SlicePlotModeKind::Rgb:
    default:
      break;
  }
  if (!hueDefined) return false;
  const float wrappedHue = wrapHue01(hue);
  const int sector = static_cast<int>(std::floor((wrappedHue + (1.0f / 12.0f)) * 6.0f)) % 6;
  switch (sector) {
    case 0: return spec.cubeSliceRed;
    case 1: return spec.cubeSliceYellow;
    case 2: return spec.cubeSliceGreen;
    case 3: return spec.cubeSliceCyan;
    case 4: return spec.cubeSliceBlue;
    case 5: return spec.cubeSliceMagenta;
    default: return false;
  }
}

bool computeLassoSamplingBounds(const LassoRegionState& state, NormalizedBounds* out) {
  if (!out || state.strokes.empty()) return false;
  double minX = 1.0;
  double minY = 1.0;
  double maxX = 0.0;
  double maxY = 0.0;
  bool sawPoint = false;
  for (const auto& stroke : state.strokes) {
    for (const auto& point : stroke.points) {
      minX = std::min(minX, static_cast<double>(point.xNorm));
      minY = std::min(minY, static_cast<double>(point.yNorm));
      maxX = std::max(maxX, static_cast<double>(point.xNorm));
      maxY = std::max(maxY, static_cast<double>(point.yNorm));
      sawPoint = true;
    }
  }
  if (!sawPoint) return false;
  out->x1 = std::clamp(minX, 0.0, 1.0);
  out->y1 = std::clamp(minY, 0.0, 1.0);
  out->x2 = std::clamp(std::max(maxX, out->x1 + 1e-6), 0.0, 1.0);
  out->y2 = std::clamp(std::max(maxY, out->y1 + 1e-6), 0.0, 1.0);
  out->valid = out->x2 > out->x1 && out->y2 > out->y1;
  return out->valid;
}

// Identity-plot recognition math used to live here. It is intentionally disabled so
// the downstream instance no longer tries to infer whether the strip is "valid enough".

uint64_t fnv1a64(const std::string& s) {
  uint64_t hash = 1469598103934665603ull;
  for (unsigned char c : s) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t fnv1a64Bytes(const void* data, std::size_t byteCount) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (std::size_t i = 0; i < byteCount; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string jsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"': out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

bool extractJsonStringField(const std::string& json, const std::string& key, std::string* out) {
  if (!out) return false;
  const std::string token = "\"" + key + "\":\"";
  const size_t start = json.find(token);
  if (start == std::string::npos) return false;
  std::string value;
  bool escaped = false;
  for (size_t i = start + token.size(); i < json.size(); ++i) {
    const char c = json[i];
    if (escaped) {
      value.push_back(c);
      escaped = false;
      continue;
    }
    if (c == '\\') {
      escaped = true;
      continue;
    }
    if (c == '"') {
      *out = value;
      return true;
    }
    value.push_back(c);
  }
  return false;
}

std::string cubeViewerPipeName() {
  const char* env = std::getenv("CHROMASPACE_PIPE");
  if (env && env[0] != '\0') return std::string(env);
#if defined(_WIN32)
  return "\\\\.\\pipe\\Chromaspace";
#else
  return "/tmp/chromaspace.sock";
#endif
}

std::string viewerExecutableName() {
#if defined(_WIN32)
  return "Chromaspace_CubeViewer.exe";
#else
  return "Chromaspace_CubeViewer";
#endif
}

std::string supportString(const std::string& label, const std::string& value) {
  return label + ": " + value;
}

void openUrl(const std::string& url) {
#if defined(_WIN32)
  ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#elif defined(__APPLE__)
  pid_t pid = 0;
  const char* argv[] = {"open", url.c_str(), nullptr};
  posix_spawnp(&pid, "open", nullptr, nullptr, const_cast<char* const*>(argv), environ);
#else
  pid_t pid = 0;
  const char* argv[] = {"xdg-open", url.c_str(), nullptr};
  posix_spawnp(&pid, "xdg-open", nullptr, nullptr, const_cast<char* const*>(argv), environ);
#endif
}

#if defined(_WIN32)
bool launchPathWithShell(const std::string& path) {
  const INT_PTR result = reinterpret_cast<INT_PTR>(
      ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
  return result > 32;
}

std::string envValueOrEmpty(const char* name) {
  const char* value = std::getenv(name);
  return (value && value[0] != '\0') ? std::string(value) : std::string();
}
#else
bool spawnDetached(const std::vector<std::string>& args) {
  if (args.empty()) return false;
  std::vector<char*> argv;
  argv.reserve(args.size() + 1u);
  for (const auto& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);
  pid_t pid = 0;
  return posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ) == 0;
}

bool pathExists(const std::string& path) {
  if (path.empty()) return false;
  std::error_code ec;
  return std::filesystem::exists(std::filesystem::path(path), ec);
}
#endif

bool openPluginManager() {
#if defined(_WIN32)
  const std::string exeName = std::string(kPluginManagerProductName) + ".exe";
  std::vector<std::string> candidates;
  const std::string localAppData = envValueOrEmpty("LOCALAPPDATA");
  if (!localAppData.empty()) {
    candidates.push_back(joinPath(joinPath(joinPath(localAppData, "Programs"), kPluginManagerProductName), exeName));
    candidates.push_back(joinPath(joinPath(joinPath(localAppData, "Programs"), "MoazElgabryPluginManager"), exeName));
  }
  const std::string programFiles = envValueOrEmpty("ProgramFiles");
  if (!programFiles.empty()) {
    candidates.push_back(joinPath(joinPath(programFiles, kPluginManagerProductName), exeName));
  }
  const std::string programFilesX86 = envValueOrEmpty("ProgramFiles(x86)");
  if (!programFilesX86.empty()) {
    candidates.push_back(joinPath(joinPath(programFilesX86, kPluginManagerProductName), exeName));
  }
  for (const auto& candidate : candidates) {
    if (fileExistsForLaunch(candidate) && launchPathWithShell(candidate)) return true;
  }
  return false;
#elif defined(__APPLE__)
  std::vector<std::string> appPaths = {
      std::string("/Applications/") + kPluginManagerProductName + ".app"
  };
  const char* home = std::getenv("HOME");
  if (home && home[0] != '\0') {
    appPaths.push_back(joinPath(joinPath(home, "Applications"), std::string(kPluginManagerProductName) + ".app"));
  }
  for (const auto& appPath : appPaths) {
    if (pathExists(appPath) && spawnDetached({"open", "-a", appPath})) return true;
  }
  return false;
#else
  const char* home = std::getenv("HOME");
  std::vector<std::pair<std::string, std::string>> desktopCandidates;
  if (home && home[0] != '\0') {
    desktopCandidates.push_back({joinPath(joinPath(joinPath(home, ".local"), "share"), "applications"), kPluginManagerBundleId});
    desktopCandidates.push_back({joinPath(joinPath(joinPath(home, ".local"), "share"), "applications"), "moaz-elgabry-plugins"});
  }
  desktopCandidates.push_back({"/usr/share/applications", kPluginManagerBundleId});
  desktopCandidates.push_back({"/usr/share/applications", "moaz-elgabry-plugins"});
  for (const auto& candidate : desktopCandidates) {
    const std::string desktopPath = joinPath(candidate.first, candidate.second + ".desktop");
    if (pathExists(desktopPath) && spawnDetached({"gtk-launch", candidate.second})) return true;
  }
  return spawnDetached({"moaz-elgabry-plugins"});
#endif
}

const std::vector<WorkshopColor::TransferFunctionId>& plotLinearTransferChoices() {
  static const std::vector<WorkshopColor::TransferFunctionId> kChoices = []() {
    std::vector<WorkshopColor::TransferFunctionId> ids;
    ids.reserve(WorkshopColor::transferFunctionCount());
    for (std::size_t i = 0; i < WorkshopColor::transferFunctionCount(); ++i) {
      const auto id = WorkshopColor::transferFunctionDefinition(i).id;
      if (id == WorkshopColor::TransferFunctionId::Linear) continue;
      ids.push_back(id);
    }
    return ids;
  }();
  return kChoices;
}

int plotLinearTransferChoiceIndex(WorkshopColor::TransferFunctionId id) {
  const auto& choices = plotLinearTransferChoices();
  const auto it = std::find(choices.begin(), choices.end(), id);
  if (it == choices.end()) return 0;
  return static_cast<int>(std::distance(choices.begin(), it));
}

WorkshopColor::TransferFunctionId plotLinearTransferIdFromChoice(int index) {
  const auto& choices = plotLinearTransferChoices();
  if (choices.empty()) return WorkshopColor::TransferFunctionId::Gamma24;
  const int clamped = std::clamp(index, 0, static_cast<int>(choices.size()) - 1);
  return choices[static_cast<std::size_t>(clamped)];
}

struct BoolScope {
  bool& ref;
  const bool previous;
  explicit BoolScope(bool& target, bool next = true) : ref(target), previous(target) {
    ref = next;
  }
  ~BoolScope() {
    ref = previous;
  }
};

struct ChromaspacePresetValues {
  int plotModel = 0;
  bool plotInLinear = false;
  int inputTransferFunction = static_cast<int>(WorkshopColor::TransferFunctionId::Gamma24);
  bool showOverflow = false;
  bool highlightOverflow = true;
  bool fillVolume = false;
  int fillResolution = 29;
  int identityReadResolution = 29;
  bool volumeSliceLassoRegion = false;
  bool volumeSliceRed = false;
  bool volumeSliceYellow = false;
  bool volumeSliceGreen = false;
  bool volumeSliceCyan = false;
  bool volumeSliceBlue = false;
  bool volumeSliceMagenta = false;
  double neutralRadius = 1.0;
  bool readGrayRamp = false;
  bool readIdentityPlot = false;
  bool isolateIdentityData = false;
  bool liveUpdate = true;
  bool keepOnTop = true;
  int updateMode = 0;
  int quality = 0;
  int scale = 3;
  int plotStyle = 1;
  double pointSize = 1.1;
  double colorSaturation = 2.0;
  int pointShape = 0;
  int sampling = 0;
  bool occupancyGuidedFill = true;
};

struct ChromaspaceUserPreset {
  std::string id;
  std::string name;
  std::string createdAtUtc;
  std::string updatedAtUtc;
  ChromaspacePresetValues values{};
};

struct ChromaspacePresetStore {
  bool loaded = false;
  ChromaspaceUserPreset defaultPreset{};
  std::vector<ChromaspaceUserPreset> userPresets;
};

constexpr const char* kChromaspacePresetDefaultName = "Default";
constexpr const char* kChromaspacePresetCustomLabel = "(Custom)";

ChromaspacePresetStore& chromaspacePresetStore() {
  static ChromaspacePresetStore store;
  return store;
}

std::mutex& chromaspacePresetMutex() {
  static std::mutex mutex;
  return mutex;
}

ChromaspacePresetValues chromaspaceFactoryPresetValues() {
  ChromaspacePresetValues values{};
  values.plotModel = 0;
  values.plotInLinear = false;
  values.inputTransferFunction = static_cast<int>(WorkshopColor::TransferFunctionId::Gamma24);
  values.showOverflow = false;
  values.highlightOverflow = true;
  values.fillVolume = false;
  values.fillResolution = 29;
  values.identityReadResolution = 29;
  values.volumeSliceLassoRegion = false;
  values.volumeSliceRed = false;
  values.volumeSliceYellow = false;
  values.volumeSliceGreen = false;
  values.volumeSliceCyan = false;
  values.volumeSliceBlue = false;
  values.volumeSliceMagenta = false;
  values.neutralRadius = 1.0;
  values.readGrayRamp = false;
  values.readIdentityPlot = false;
  values.isolateIdentityData = false;
  values.liveUpdate = true;
  values.keepOnTop = true;
  values.updateMode = 0;
  values.quality = 0;
  values.scale = 3;
  values.plotStyle = 1;
  values.pointSize = 1.1;
  values.colorSaturation = 2.0;
  values.pointShape = 0;
  values.sampling = 0;
  values.occupancyGuidedFill = true;
  return values;
}

std::string nowUtcIso8601() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  char buffer[32] = {0};
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return std::string(buffer);
}

std::string normalizePresetNameKey(const std::string& name) {
  std::string out;
  out.reserve(name.size());
  bool inWhitespace = false;
  for (char c : name) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (std::isspace(uc)) {
      inWhitespace = true;
      continue;
    }
    if (inWhitespace && !out.empty()) out.push_back(' ');
    inWhitespace = false;
    out.push_back(static_cast<char>(std::tolower(uc)));
  }
  while (!out.empty() && out.front() == ' ') out.erase(out.begin());
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::string sanitizePresetName(const std::string& in, const char* fallback) {
  std::string out;
  out.reserve(in.size());
  for (char c : in) {
    if (c == '\n' || c == '\r' || c == '\t') continue;
    out.push_back(c);
  }
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
  while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
  if (out.empty()) out = fallback ? std::string(fallback) : std::string("Preset");
  if (out.size() > 96) out.resize(96);
  return out;
}

std::string makePresetId(const std::string& prefix) {
  static std::atomic<unsigned long> counter{1};
  std::ostringstream os;
  os << prefix << '_' << std::time(nullptr) << '_' << counter.fetch_add(1, std::memory_order_relaxed);
  return os.str();
}

std::filesystem::path chromaspacePresetDirPath() {
#ifdef _WIN32
  const char* base = std::getenv("APPDATA");
  if (!base || !*base) base = std::getenv("LOCALAPPDATA");
  if (base && *base) return std::filesystem::path(base) / "Chromaspace";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (home && *home) return std::filesystem::path(home) / "Library" / "Application Support" / "Chromaspace";
#else
  const char* home = std::getenv("HOME");
  if (home && *home) return std::filesystem::path(home) / ".config" / "Chromaspace";
#endif
  return std::filesystem::path(".");
}

std::filesystem::path chromaspacePresetFilePath() {
  return chromaspacePresetDirPath() / "presets_v1.json";
}

bool chromaspacePresetValuesEqual(const ChromaspacePresetValues& a, const ChromaspacePresetValues& b) {
  return a.plotModel == b.plotModel &&
         a.plotInLinear == b.plotInLinear &&
         a.inputTransferFunction == b.inputTransferFunction &&
         a.showOverflow == b.showOverflow &&
         a.highlightOverflow == b.highlightOverflow &&
         a.fillVolume == b.fillVolume &&
         a.fillResolution == b.fillResolution &&
         a.identityReadResolution == b.identityReadResolution &&
         a.volumeSliceLassoRegion == b.volumeSliceLassoRegion &&
         a.volumeSliceRed == b.volumeSliceRed &&
         a.volumeSliceYellow == b.volumeSliceYellow &&
         a.volumeSliceGreen == b.volumeSliceGreen &&
         a.volumeSliceCyan == b.volumeSliceCyan &&
         a.volumeSliceBlue == b.volumeSliceBlue &&
         a.volumeSliceMagenta == b.volumeSliceMagenta &&
         std::abs(a.neutralRadius - b.neutralRadius) <= 1e-6 &&
         a.readGrayRamp == b.readGrayRamp &&
         a.readIdentityPlot == b.readIdentityPlot &&
         a.isolateIdentityData == b.isolateIdentityData &&
         a.liveUpdate == b.liveUpdate &&
         a.keepOnTop == b.keepOnTop &&
         a.updateMode == b.updateMode &&
         a.quality == b.quality &&
         a.scale == b.scale &&
         a.plotStyle == b.plotStyle &&
         std::abs(a.pointSize - b.pointSize) <= 1e-6 &&
         std::abs(a.colorSaturation - b.colorSaturation) <= 1e-6 &&
         a.pointShape == b.pointShape &&
         a.sampling == b.sampling &&
         a.occupancyGuidedFill == b.occupancyGuidedFill;
}

bool chromaspacePresetNameReserved(const std::string& name) {
  return normalizePresetNameKey(name) == "default";
}

bool chromaspacePresetUserNameExistsLocked(const std::string& name,
                                           const std::string* ignoreId = nullptr) {
  const std::string key = normalizePresetNameKey(name);
  if (key.empty() || key == "default") return key == "default";
  for (const auto& preset : chromaspacePresetStore().userPresets) {
    if (ignoreId && !ignoreId->empty() && preset.id == *ignoreId) continue;
    if (normalizePresetNameKey(preset.name) == key) return true;
  }
  return false;
}

int chromaspaceUserPresetIndexByNameLocked(const std::string& name) {
  const std::string key = normalizePresetNameKey(name);
  const auto& presets = chromaspacePresetStore().userPresets;
  for (int i = 0; i < static_cast<int>(presets.size()); ++i) {
    if (normalizePresetNameKey(presets[static_cast<std::size_t>(i)].name) == key) return i;
  }
  return -1;
}

bool extractJsonBoolField(const std::string& json, const std::string& key, bool* out) {
  if (!out) return false;
  const std::string token = "\"" + key + "\":";
  const size_t start = json.find(token);
  if (start == std::string::npos) return false;
  size_t pos = start + token.size();
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (json.compare(pos, 4, "true") == 0) {
    *out = true;
    return true;
  }
  if (json.compare(pos, 5, "false") == 0) {
    *out = false;
    return true;
  }
  return false;
}

bool extractJsonNumberFieldText(const std::string& json, const std::string& key, std::string* out) {
  if (!out) return false;
  const std::string token = "\"" + key + "\":";
  const size_t start = json.find(token);
  if (start == std::string::npos) return false;
  size_t pos = start + token.size();
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  size_t end = pos;
  while (end < json.size()) {
    const char c = json[end];
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
      break;
    }
    ++end;
  }
  if (end <= pos) return false;
  *out = json.substr(pos, end - pos);
  return true;
}

bool extractJsonIntField(const std::string& json, const std::string& key, int* out) {
  if (!out) return false;
  std::string text;
  if (!extractJsonNumberFieldText(json, key, &text)) return false;
  try {
    *out = std::stoi(text);
    return true;
  } catch (...) {
    return false;
  }
}

bool extractJsonDoubleField(const std::string& json, const std::string& key, double* out) {
  if (!out) return false;
  std::string text;
  if (!extractJsonNumberFieldText(json, key, &text)) return false;
  try {
    *out = std::stod(text);
    return true;
  } catch (...) {
    return false;
  }
}

bool extractJsonStructuredField(const std::string& json,
                                const std::string& key,
                                char openChar,
                                char closeChar,
                                std::string* out) {
  if (!out) return false;
  const std::string token = "\"" + key + "\":";
  const size_t start = json.find(token);
  if (start == std::string::npos) return false;
  size_t pos = start + token.size();
  while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
  if (pos >= json.size() || json[pos] != openChar) return false;
  int depth = 0;
  bool inString = false;
  bool escaped = false;
  size_t end = pos;
  for (; end < json.size(); ++end) {
    const char c = json[end];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
      continue;
    }
    if (c == openChar) {
      ++depth;
    } else if (c == closeChar) {
      --depth;
      if (depth == 0) {
        *out = json.substr(pos, end - pos + 1);
        return true;
      }
    }
  }
  return false;
}

bool extractJsonObjectField(const std::string& json, const std::string& key, std::string* out) {
  return extractJsonStructuredField(json, key, '{', '}', out);
}

bool extractJsonArrayField(const std::string& json, const std::string& key, std::string* out) {
  return extractJsonStructuredField(json, key, '[', ']', out);
}

std::vector<std::string> extractJsonObjectsFromArray(const std::string& arrayJson) {
  std::vector<std::string> out;
  bool inString = false;
  bool escaped = false;
  int depth = 0;
  size_t objectStart = std::string::npos;
  for (size_t i = 0; i < arrayJson.size(); ++i) {
    const char c = arrayJson[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      inString = true;
      continue;
    }
    if (c == '{') {
      if (depth == 0) objectStart = i;
      ++depth;
      continue;
    }
    if (c == '}') {
      --depth;
      if (depth == 0 && objectStart != std::string::npos) {
        out.push_back(arrayJson.substr(objectStart, i - objectStart + 1));
        objectStart = std::string::npos;
      }
    }
  }
  return out;
}

std::string chromaspacePresetValuesAsJson(const ChromaspacePresetValues& values) {
  std::ostringstream os;
  os << "{";
  os << "\"plotModel\":" << values.plotModel << ",";
  os << "\"plotInLinear\":" << (values.plotInLinear ? "true" : "false") << ",";
  os << "\"inputTransferFunction\":" << values.inputTransferFunction << ",";
  os << "\"showOverflow\":" << (values.showOverflow ? "true" : "false") << ",";
  os << "\"highlightOverflow\":" << (values.highlightOverflow ? "true" : "false") << ",";
  os << "\"fillVolume\":" << (values.fillVolume ? "true" : "false") << ",";
  os << "\"fillResolution\":" << values.fillResolution << ",";
  os << "\"identityReadResolution\":" << values.identityReadResolution << ",";
  os << "\"volumeSliceLassoRegion\":" << (values.volumeSliceLassoRegion ? "true" : "false") << ",";
  os << "\"volumeSliceRed\":" << (values.volumeSliceRed ? "true" : "false") << ",";
  os << "\"volumeSliceYellow\":" << (values.volumeSliceYellow ? "true" : "false") << ",";
  os << "\"volumeSliceGreen\":" << (values.volumeSliceGreen ? "true" : "false") << ",";
  os << "\"volumeSliceCyan\":" << (values.volumeSliceCyan ? "true" : "false") << ",";
  os << "\"volumeSliceBlue\":" << (values.volumeSliceBlue ? "true" : "false") << ",";
  os << "\"volumeSliceMagenta\":" << (values.volumeSliceMagenta ? "true" : "false") << ",";
  os << "\"neutralRadius\":" << std::setprecision(15) << values.neutralRadius << ",";
  os << "\"readGrayRamp\":" << (values.readGrayRamp ? "true" : "false") << ",";
  os << "\"readIdentityPlot\":" << (values.readIdentityPlot ? "true" : "false") << ",";
  os << "\"isolateIdentityData\":" << (values.isolateIdentityData ? "true" : "false") << ",";
  os << "\"liveUpdate\":" << (values.liveUpdate ? "true" : "false") << ",";
  os << "\"keepOnTop\":" << (values.keepOnTop ? "true" : "false") << ",";
  os << "\"updateMode\":" << values.updateMode << ",";
  os << "\"quality\":" << values.quality << ",";
  os << "\"scale\":" << values.scale << ",";
  os << "\"plotStyle\":" << values.plotStyle << ",";
  os << "\"pointSize\":" << std::setprecision(15) << values.pointSize << ",";
  os << "\"colorSaturation\":" << std::setprecision(15) << values.colorSaturation << ",";
  os << "\"pointShape\":" << values.pointShape << ",";
  os << "\"sampling\":" << values.sampling << ",";
  os << "\"occupancyGuidedFill\":" << (values.occupancyGuidedFill ? "true" : "false");
  os << "}";
  return os.str();
}

bool parseChromaspacePresetValuesFromJson(const std::string& json, ChromaspacePresetValues* out) {
  if (!out) return false;
  ChromaspacePresetValues values = chromaspaceFactoryPresetValues();
  (void)extractJsonIntField(json, "plotModel", &values.plotModel);
  (void)extractJsonBoolField(json, "plotInLinear", &values.plotInLinear);
  (void)extractJsonIntField(json, "inputTransferFunction", &values.inputTransferFunction);
  (void)extractJsonBoolField(json, "showOverflow", &values.showOverflow);
  (void)extractJsonBoolField(json, "highlightOverflow", &values.highlightOverflow);
  (void)extractJsonBoolField(json, "fillVolume", &values.fillVolume);
  (void)extractJsonIntField(json, "fillResolution", &values.fillResolution);
  (void)extractJsonIntField(json, "identityReadResolution", &values.identityReadResolution);
  (void)extractJsonBoolField(json, "volumeSliceLassoRegion", &values.volumeSliceLassoRegion);
  (void)extractJsonBoolField(json, "volumeSliceRed", &values.volumeSliceRed);
  (void)extractJsonBoolField(json, "volumeSliceYellow", &values.volumeSliceYellow);
  (void)extractJsonBoolField(json, "volumeSliceGreen", &values.volumeSliceGreen);
  (void)extractJsonBoolField(json, "volumeSliceCyan", &values.volumeSliceCyan);
  (void)extractJsonBoolField(json, "volumeSliceBlue", &values.volumeSliceBlue);
  (void)extractJsonBoolField(json, "volumeSliceMagenta", &values.volumeSliceMagenta);
  (void)extractJsonDoubleField(json, "neutralRadius", &values.neutralRadius);
  (void)extractJsonBoolField(json, "readGrayRamp", &values.readGrayRamp);
  (void)extractJsonBoolField(json, "readIdentityPlot", &values.readIdentityPlot);
  (void)extractJsonBoolField(json, "isolateIdentityData", &values.isolateIdentityData);
  (void)extractJsonBoolField(json, "liveUpdate", &values.liveUpdate);
  (void)extractJsonBoolField(json, "keepOnTop", &values.keepOnTop);
  (void)extractJsonIntField(json, "updateMode", &values.updateMode);
  (void)extractJsonIntField(json, "quality", &values.quality);
  (void)extractJsonIntField(json, "scale", &values.scale);
  (void)extractJsonIntField(json, "plotStyle", &values.plotStyle);
  (void)extractJsonDoubleField(json, "pointSize", &values.pointSize);
  (void)extractJsonDoubleField(json, "colorSaturation", &values.colorSaturation);
  (void)extractJsonIntField(json, "pointShape", &values.pointShape);
  (void)extractJsonIntField(json, "sampling", &values.sampling);
  (void)extractJsonBoolField(json, "occupancyGuidedFill", &values.occupancyGuidedFill);
  *out = values;
  return true;
}

void saveChromaspacePresetStoreLocked() {
  const auto path = chromaspacePresetFilePath();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::ofstream os(path, std::ios::binary | std::ios::trunc);
  if (!os.is_open()) return;

  ChromaspacePresetStore& store = chromaspacePresetStore();
  os << "{\n";
  os << "  \"schemaVersion\":1,\n";
  os << "  \"updatedAtUtc\":\"" << jsonEscape(nowUtcIso8601()) << "\",\n";
  os << "  \"defaultPreset\":{\n";
  os << "    \"id\":\"" << jsonEscape(store.defaultPreset.id) << "\",\n";
  os << "    \"name\":\"" << jsonEscape(store.defaultPreset.name) << "\",\n";
  os << "    \"updatedAtUtc\":\"" << jsonEscape(store.defaultPreset.updatedAtUtc) << "\",\n";
  os << "    \"values\":" << chromaspacePresetValuesAsJson(store.defaultPreset.values) << "\n";
  os << "  },\n";
  os << "  \"userPresets\":[\n";
  for (size_t i = 0; i < store.userPresets.size(); ++i) {
    const auto& preset = store.userPresets[i];
    os << "    {\n";
    os << "      \"id\":\"" << jsonEscape(preset.id) << "\",\n";
    os << "      \"name\":\"" << jsonEscape(preset.name) << "\",\n";
    os << "      \"createdAtUtc\":\"" << jsonEscape(preset.createdAtUtc) << "\",\n";
    os << "      \"updatedAtUtc\":\"" << jsonEscape(preset.updatedAtUtc) << "\",\n";
    os << "      \"values\":" << chromaspacePresetValuesAsJson(preset.values) << "\n";
    os << "    }" << (i + 1 < store.userPresets.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

void ensureChromaspacePresetStoreLoadedLocked() {
  ChromaspacePresetStore& store = chromaspacePresetStore();
  if (store.loaded) return;
  store = ChromaspacePresetStore{};
  store.loaded = true;

  const ChromaspacePresetValues factory = chromaspaceFactoryPresetValues();
  store.defaultPreset.id = "default";
  store.defaultPreset.name = kChromaspacePresetDefaultName;
  store.defaultPreset.createdAtUtc = nowUtcIso8601();
  store.defaultPreset.updatedAtUtc = store.defaultPreset.createdAtUtc;
  store.defaultPreset.values = factory;

  bool needsSave = false;
  std::ifstream is(chromaspacePresetFilePath(), std::ios::binary);
  if (!is.is_open()) {
    saveChromaspacePresetStoreLocked();
    return;
  }

  std::string json((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  if (json.empty()) {
    saveChromaspacePresetStoreLocked();
    return;
  }

  std::string defaultObj;
  if (extractJsonObjectField(json, "defaultPreset", &defaultObj)) {
    std::string defaultValuesJson;
    if (extractJsonObjectField(defaultObj, "values", &defaultValuesJson)) {
      parseChromaspacePresetValuesFromJson(defaultValuesJson, &store.defaultPreset.values);
    } else {
      needsSave = true;
    }
    std::string updatedAtUtc;
    if (extractJsonStringField(defaultObj, "updatedAtUtc", &updatedAtUtc)) {
      store.defaultPreset.updatedAtUtc = updatedAtUtc;
    }
  } else {
    needsSave = true;
  }

  std::string arrayJson;
  if (extractJsonArrayField(json, "userPresets", &arrayJson)) {
    for (const auto& objectJson : extractJsonObjectsFromArray(arrayJson)) {
      ChromaspaceUserPreset preset{};
      std::string name;
      if (!extractJsonStringField(objectJson, "name", &name)) continue;
      name = sanitizePresetName(name, "Preset");
      if (chromaspacePresetNameReserved(name)) continue;
      if (chromaspacePresetUserNameExistsLocked(name)) continue;
      std::string valuesJson;
      if (!extractJsonObjectField(objectJson, "values", &valuesJson)) continue;
      parseChromaspacePresetValuesFromJson(valuesJson, &preset.values);
      if (!extractJsonStringField(objectJson, "id", &preset.id) || preset.id.empty()) {
        preset.id = makePresetId("chromaspace");
        needsSave = true;
      }
      preset.name = name;
      if (!extractJsonStringField(objectJson, "createdAtUtc", &preset.createdAtUtc) || preset.createdAtUtc.empty()) {
        preset.createdAtUtc = nowUtcIso8601();
        needsSave = true;
      }
      if (!extractJsonStringField(objectJson, "updatedAtUtc", &preset.updatedAtUtc) || preset.updatedAtUtc.empty()) {
        preset.updatedAtUtc = preset.createdAtUtc;
        needsSave = true;
      }
      store.userPresets.push_back(preset);
    }
  }

  if (needsSave) saveChromaspacePresetStoreLocked();
}

void reloadChromaspacePresetStoreFromDiskLocked() {
  chromaspacePresetStore() = ChromaspacePresetStore{};
  ensureChromaspacePresetStoreLoadedLocked();
}

ChromaspacePresetValues describeChromaspaceDefaultValues() {
  std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
  ensureChromaspacePresetStoreLoadedLocked();
  return chromaspacePresetStore().defaultPreset.values;
}

std::vector<std::string> visibleChromaspaceUserPresetNames() {
  std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
  ensureChromaspacePresetStoreLoadedLocked();
  std::vector<std::string> out;
  out.reserve(chromaspacePresetStore().userPresets.size());
  for (const auto& preset : chromaspacePresetStore().userPresets) out.push_back(preset.name);
  return out;
}

#ifdef _WIN32
bool confirmChromaspacePresetOverwriteDialog(const std::string& presetName) {
  const std::string message = "Preset '" + presetName + "' already exists. Overwrite?";
  return MessageBoxA(nullptr, message.c_str(), "Chromaspace", MB_ICONQUESTION | MB_YESNO) == IDYES;
}

void showChromaspacePresetInfoDialog(const std::string& text) {
  MessageBoxA(nullptr, text.c_str(), "Chromaspace", MB_ICONINFORMATION | MB_OK);
}

bool confirmChromaspacePresetDeleteDialog(const std::string& presetName) {
  const std::string message = "Delete preset '" + presetName + "'? This cannot be undone.";
  return MessageBoxA(nullptr, message.c_str(), "Chromaspace", MB_ICONWARNING | MB_YESNO) == IDYES;
}
#elif defined(__APPLE__)
std::string execAndReadChromaspace(const std::string& cmd) {
  std::string out;
  FILE* f = popen(cmd.c_str(), "r");
  if (!f) return out;
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), f)) out += buffer;
  pclose(f);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

bool confirmChromaspacePresetOverwriteDialog(const std::string& presetName) {
  std::string safe = presetName;
  for (char& c : safe) if (c == '"') c = '\'';
  const std::string cmd =
      "osascript -e 'button returned of (display dialog \"Preset \\\"" + safe +
      "\\\" already exists. Overwrite?\" buttons {\"Cancel\",\"Overwrite\"} default button \"Overwrite\")' 2>/dev/null";
  return execAndReadChromaspace(cmd) == "Overwrite";
}

void showChromaspacePresetInfoDialog(const std::string& text) {
  std::string safe = text;
  for (char& c : safe) if (c == '"') c = '\'';
  const std::string cmd =
      "osascript -e 'display dialog \"" + safe + "\" buttons {\"OK\"} default button \"OK\"' 2>/dev/null";
  (void)execAndReadChromaspace(cmd);
}

bool confirmChromaspacePresetDeleteDialog(const std::string& presetName) {
  std::string safe = presetName;
  for (char& c : safe) if (c == '"') c = '\'';
  const std::string cmd =
      "osascript -e 'button returned of (display dialog \"Delete preset \\\"" + safe +
      "\\\"? This cannot be undone.\" buttons {\"Cancel\",\"Delete\"} default button \"Delete\")' 2>/dev/null";
  return execAndReadChromaspace(cmd) == "Delete";
}
#else
bool linuxChromaspaceCommandExists(const char* cmd) {
  if (!cmd || !*cmd) return false;
  std::string probe = "command -v ";
  probe += cmd;
  probe += " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

bool confirmChromaspacePresetOverwriteDialog(const std::string& presetName) {
  if (linuxChromaspaceCommandExists("zenity")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "zenity --question --title=\"Chromaspace\" --text=\"Preset '" + safe + "' already exists. Overwrite?\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  if (linuxChromaspaceCommandExists("kdialog")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "kdialog --warningyesno \"Preset '" + safe + "' already exists. Overwrite?\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  std::fprintf(stderr, "[Chromaspace] overwrite confirmation unavailable for preset '%s'.\n", presetName.c_str());
  return false;
}

void showChromaspacePresetInfoDialog(const std::string& text) {
  if (linuxChromaspaceCommandExists("zenity")) {
    std::string safe = text;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "zenity --info --title=\"Chromaspace\" --text=\"" + safe + "\" 2>/dev/null";
    (void)std::system(cmd.c_str());
    return;
  }
  if (linuxChromaspaceCommandExists("kdialog")) {
    std::string safe = text;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd = "kdialog --msgbox \"" + safe + "\" 2>/dev/null";
    (void)std::system(cmd.c_str());
    return;
  }
  std::fprintf(stderr, "[Chromaspace] %s\n", text.c_str());
}

bool confirmChromaspacePresetDeleteDialog(const std::string& presetName) {
  if (linuxChromaspaceCommandExists("zenity")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "zenity --question --title=\"Chromaspace\" --text=\"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  if (linuxChromaspaceCommandExists("kdialog")) {
    std::string safe = presetName;
    for (char& c : safe) if (c == '"') c = '\'';
    const std::string cmd =
        "kdialog --warningyesno \"Delete preset '" + safe + "'? This cannot be undone.\" 2>/dev/null";
    return std::system(cmd.c_str()) == 0;
  }
  std::fprintf(stderr, "[Chromaspace] delete confirmation unavailable for preset '%s'.\n", presetName.c_str());
  return false;
}
#endif

std::atomic<int> gSharedCubeViewerRequestCount{0};
std::mutex gSharedCubeViewerSenderMutex;
std::string gSharedCubeViewerActiveSenderId;
std::string gSharedCubeViewerActiveSourceId;
std::atomic<int64_t> gSharedCubeViewerActiveRenderMs{0};
std::mutex gSharedCubeViewerTransportMutex;
std::atomic<uint64_t> gSharedCubeViewerSeqCounter{1};

class ChromaspaceEffect : public ImageEffect {
 public:
  friend class ChromaspaceOverlayInteract;

  explicit ChromaspaceEffect(OfxImageEffectHandle handle)
      : ImageEffect(handle) {
    dstClip_ = fetchClip(kOfxImageEffectOutputClipName);
    srcClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);

    cubeViewerLive_ = getBoolValue("cubeViewerLive", 0.0, true);
    cubeViewerUpdateMode_ = getChoiceValue("cubeViewerUpdateMode", 0.0, 0);
    cubeViewerQuality_ = getChoiceValue("cubeViewerQuality", 0.0, 0);
    cubeViewerRequested_ = false;
    cubeViewerConnected_ = false;
    cubeViewerWindowUsable_ = false;
    senderId_ = buildSenderId();
    setStatusLabel("Disconnected");
    flushStatusLabelToHost();
    updateDrawOnImageModeUi(0.0);
    syncIdentityOverlayGroupOpenState(0.0);
    updateCircularHslToggleVisibility(0.0);
    updateCircularHsvToggleVisibility(0.0);
    updateNormConeToggleVisibility(0.0);
    syncChromaspacePresetMenuFromDisk(0.0);
  }

  ~ChromaspaceEffect() override {
    stopStatusThread();
    stopIoWorker();
    releaseSharedViewerSession();
    cubeViewerRequested_ = false;
  }

  void getClipPreferences(ClipPreferencesSetter& clipPreferences) override {
    clipPreferences.setOutputFrameVarying(true);
  }

  void syncPrivateData(void) override {
    flushStatusLabelToHost();
    updateDrawOnImageModeUi(0.0);
    syncIdentityOverlayGroupOpenState(0.0);
    updateCircularHslToggleVisibility(0.0);
    updateCircularHsvToggleVisibility(0.0);
    updateNormConeToggleVisibility(0.0);
    syncChromaspacePresetMenuState(0.0);
    logSharedViewerEvent("syncPrivateData");
    if (cubeViewerRequested_ && viewerSessionRequested() && !sharedViewerActiveForThisSender()) {
      markSharedViewerActiveSender();
      cubeViewerInputCloudRefreshPending_ = true;
      deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
      logSharedViewerEvent("syncPrivateData/activate");
      pushParamsUpdate(0.0, "syncPrivateData/activate");
      (void)trySendCachedCloud(0.0, "syncPrivateData/activate");
    }
    ImageEffect::syncPrivateData();
  }

  bool viewerSessionRequested() const {
    return cubeViewerRequested_ || gSharedCubeViewerRequestCount.load(std::memory_order_relaxed) > 0;
  }

  std::string sharedViewerDebugStateLocked() const {
    std::ostringstream os;
    os << "inst=" << static_cast<const void*>(this)
       << " sender=" << senderId_
       << " requested=" << (cubeViewerRequested_ ? 1 : 0)
       << " sharedReq=" << gSharedCubeViewerRequestCount.load(std::memory_order_relaxed)
       << " activeSender=" << gSharedCubeViewerActiveSenderId
       << " activeSource=" << gSharedCubeViewerActiveSourceId
       << " activeRenderMs=" << gSharedCubeViewerActiveRenderMs.load(std::memory_order_relaxed);
    return os.str();
  }

  void logSharedViewerEvent(const std::string& event,
                            const std::string& sourceId = std::string(),
                            const std::string& extra = std::string()) const {
    if (!cubeViewerMultiInstanceDebugEnabled()) return;
    std::ostringstream os;
    os << event;
    if (!sourceId.empty()) os << " source=" << sourceId;
    if (!extra.empty()) os << " " << extra;
    {
      std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
      os << " " << sharedViewerDebugStateLocked();
    }
    cubeViewerMultiInstanceDebugLog(os.str());
  }

  void ensureViewerSessionTransportReady() {
    if (!viewerSessionRequested()) return;
    startIoWorker();
  }

  bool drivesSharedViewer() const {
    return viewerSessionRequested() && sharedViewerActiveForThisSender();
  }

  bool sharedViewerActiveForThisSender() const {
    std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
    return !gSharedCubeViewerActiveSenderId.empty() && gSharedCubeViewerActiveSenderId == senderId_;
  }

  void markSharedViewerActiveSender(const std::string& sourceId = std::string()) {
    std::ostringstream os;
    {
      std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
      gSharedCubeViewerActiveSenderId = senderId_;
      if (!sourceId.empty()) gSharedCubeViewerActiveSourceId = sourceId;
      gSharedCubeViewerActiveRenderMs.store(monotonicNowMs(), std::memory_order_relaxed);
      if (cubeViewerMultiInstanceDebugEnabled()) {
        os << "markActive";
        if (!sourceId.empty()) os << " source=" << sourceId;
        os << " " << sharedViewerDebugStateLocked();
      }
    }
    if (!os.str().empty()) cubeViewerMultiInstanceDebugLog(os.str());
  }

  bool shouldClaimSharedViewerForSource(const std::string& sourceId) const {
    if (!viewerSessionRequested() || sourceId.empty()) {
      logSharedViewerEvent("claimCheck", sourceId,
                           std::string("result=0 reason=") +
                               (!viewerSessionRequested() ? "no-session" : "empty-source"));
      return false;
    }
    const int64_t activeRenderMs = gSharedCubeViewerActiveRenderMs.load(std::memory_order_relaxed);
    const int64_t nowMs = monotonicNowMs();
    constexpr int64_t kSharedViewerSourceStaleMs = 240;
    bool result = false;
    std::string reason;
    std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
    if (gSharedCubeViewerActiveSenderId.empty()) {
      result = true;
      reason = "no-active-sender";
    } else if (gSharedCubeViewerActiveSenderId == senderId_) {
      result = true;
      reason = "already-active-sender";
    } else if (gSharedCubeViewerActiveSourceId == sourceId) {
      result = false;
      reason = "same-source-active";
    } else {
      result = activeRenderMs <= 0 || (nowMs - activeRenderMs) >= kSharedViewerSourceStaleMs;
      reason = result ? "previous-source-stale" : "previous-source-still-active";
    }
    if (cubeViewerMultiInstanceDebugEnabled()) {
      std::ostringstream os;
      os << "claimCheck"
         << " source=" << sourceId
         << " result=" << (result ? 1 : 0)
         << " reason=" << reason
         << " nowMs=" << nowMs
         << " activeAgeMs=" << (activeRenderMs > 0 ? (nowMs - activeRenderMs) : -1)
         << " " << sharedViewerDebugStateLocked();
      cubeViewerMultiInstanceDebugLog(os.str());
    }
    return result;
  }

  void noteSharedViewerRenderActivity(const std::string& sourceId) {
    if (!sharedViewerActiveForThisSender()) return;
    if (!sourceId.empty()) {
      std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
      gSharedCubeViewerActiveSourceId = sourceId;
    }
    gSharedCubeViewerActiveRenderMs.store(monotonicNowMs(), std::memory_order_relaxed);
    logSharedViewerEvent("renderActivity", sourceId);
  }

  void retainSharedViewerSession() {
    if (cubeViewerRequested_) return;
    gSharedCubeViewerRequestCount.fetch_add(1, std::memory_order_relaxed);
    logSharedViewerEvent("retainSession");
  }

  void releaseSharedViewerSession() {
    if (!cubeViewerRequested_) return;
    const int previous = gSharedCubeViewerRequestCount.fetch_sub(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
      if (gSharedCubeViewerActiveSenderId == senderId_ || previous <= 1) {
        gSharedCubeViewerActiveSenderId.clear();
        gSharedCubeViewerActiveSourceId.clear();
      }
    }
    if (previous <= 1) gSharedCubeViewerActiveRenderMs.store(0, std::memory_order_relaxed);
    if (previous <= 1) {
      gSharedCubeViewerRequestCount.store(0, std::memory_order_relaxed);
    }
    logSharedViewerEvent("releaseSession", std::string(), std::string("previous=") + std::to_string(previous));
  }

  void render(const RenderArguments& args) override {
    flushStatusLabelToHost();
    syncIdentityOverlayGroupOpenState(args.time);
    std::unique_ptr<Image> dst(dstClip_->fetchImage(args.time));
    if (!dst) return;
    std::unique_ptr<Image> src(srcClip_->fetchImage(args.time));
    const bool drawOnImageMode = currentDrawOnImageMode(args.time);
    const bool sessionRequested = viewerSessionRequested();
    const std::string sourceId = currentSourceIdentifier(src.get());
    if (cubeViewerMultiInstanceDebugEnabled()) {
      std::ostringstream os;
      os << "time=" << args.time
         << " drawOnImage=" << (drawOnImageMode ? 1 : 0)
         << " sessionRequested=" << (sessionRequested ? 1 : 0)
         << " source=" << sourceId;
      logSharedViewerEvent("render", sourceId, os.str());
    }
    const bool sourceHandoffNeeded =
        !drawOnImageMode && sessionRequested && !sharedViewerActiveForThisSender() &&
        shouldClaimSharedViewerForSource(sourceId);
    if (sourceHandoffNeeded) {
      markSharedViewerActiveSender(sourceId);
      cubeViewerInputCloudRefreshPending_ = true;
      deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
      logSharedViewerEvent("render/sourceHandoff", sourceId);
      pushParamsUpdate(args.time, "render/source-handoff");
      (void)trySendCachedCloud(args.time, "render/source-handoff");
    }
    const bool drivesViewer = drivesSharedViewer();
    logSharedViewerEvent("render/drivesViewer", sourceId,
                         std::string("drives=") + (drivesViewer ? "1" : "0"));
    if (drivesViewer) noteSharedViewerRenderActivity(sourceId);
    const auto renderNow = std::chrono::steady_clock::now();
    const std::string settingsKey = currentCloudSettingsKey(args.time);
    const bool needCloudWork = !drawOnImageMode && drivesViewer && cubeViewerLive_;
    const bool previewMode = shouldUseInteractivePreview(renderNow);
    const bool firstHandoff = cubeViewerInputCloudRefreshPending_;
    const bool steadyState = shouldEmitSteadyStateCloud(args.time, sourceId, settingsKey, previewMode);
    const bool needCloud = needCloudWork && (firstHandoff || steadyState);
    lastRenderSeenAt_ = renderNow;
    if (needCloudWork && !firstHandoff && !steadyState && cloudQueuedOrInFlight_.load(std::memory_order_relaxed)) {
      deferredLatestCloudRefresh_.store(true, std::memory_order_relaxed);
    }
    if (!drawOnImageMode && drivesViewer && !cubeViewerConnected_) {
      pushParamsUpdate(args.time, "render/connect");
    }
    std::unique_ptr<Image> cloudSrc;
    if (needCloud && srcClip_ != nullptr) {
      const OfxRectD rod = srcClip_->getRegionOfDefinition(args.time);
      cloudSrc.reset(srcClip_->fetchImage(args.time, rod));
      if (!cloudSrc) {
        cubeViewerDebugLog("Failed to fetch full-source image for cloud build; falling back to render-window source.");
      }
    }
    Image* cloudImage = cloudSrc ? cloudSrc.get() : src.get();
    const std::string cloudSourceId = currentSourceIdentifier(cloudImage);
    CloudBuildResult built{};
    OverlayStripData overlay{};
    const bool haveOverlay = drawOnImageMode && buildIdentityOverlayStripData(dst->getBounds(), args.renderWindow, args.time, &overlay);

    // Stage 1: draw-on-image mode is a passthrough render with an optional synthetic strip burn-in.
    if (drawOnImageMode) {
      if (tryRenderGpuBackends(src.get(), dst.get(), args, false, false, nullptr, haveOverlay ? &overlay : nullptr)) {
        return;
      }
      copySourceToDestination(src.get(), dst.get(), args.renderWindow);
      if (haveOverlay) {
        applyIdentityOverlayToDestination(dst.get(), args.renderWindow, args.time);
      }
      return;
    }

    // Stage 2: plot mode keeps the host image copy on the original render-window source.
    // Any full-source fetch is only for cloud extraction; feeding that larger image into the
    // GPU passthrough path can mismatch the destination bounds on hosts like Resolve.
    const bool copiedViaGpu = tryRenderGpuBackends(src.get(), dst.get(), args, false, false, nullptr, nullptr);
    if (!copiedViaGpu) {
      copySourceToDestination(src.get(), dst.get(), args.renderWindow);
    }
    if (!needCloud || !cloudImage) return;

    built = buildViewerCloudPayload(cloudImage, dst.get(), args, previewMode);
    if (!built.success) return;
    bool cloudChanged = false;
    const std::string effectiveSettingsKey = currentCloudSettingsKey(args.time);
    if (!promoteBuiltCloud(built, cloudSourceId, effectiveSettingsKey, &cloudChanged)) {
      if (firstHandoff) {
        (void)trySendCachedCloud(args.time, "first-handoff/rejected-smaller-source");
      }
      return;
    }

    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    if (cloudChanged) {
      enqueueCloudMessage(built.payload, firstHandoff ? "first-handoff/render" : "steady-state/render");
    }
    lastCloudTime_ = args.time;
    lastCloudBuiltAt_ = std::chrono::steady_clock::now();
    lastCloudSourceId_ = cloudSourceId;
    lastCloudSettingsKey_ = effectiveSettingsKey;
    cubeViewerInputCloudRefreshPending_ = false;
    if (cloudChanged) {
      setStatusLabel("Updating");
    }
  }

  void changedParam(const InstanceChangedArgs& args, const std::string& paramName) override {
    flushStatusLabelToHost();
    if (suppressChromaspacePresetChangedHandling_) return;
    if (paramName == "chromaspacePresetMenu") {
      applySelectedChromaspacePreset(args.time);
      return;
    }
    if (paramName == "chromaspacePresetSave") {
      const std::string name = sanitizePresetName(getStringValue("chromaspacePresetName", args.time, "Preset"), "Preset");
      if (chromaspacePresetNameReserved(name)) {
        showChromaspacePresetInfoDialog("The preset name 'Default' is reserved. Use 'Save Defaults' to overwrite the protected Default preset.");
        return;
      }

      ChromaspacePresetSelection preferred{};
      preferred.kind = ChromaspacePresetSelection::Kind::User;
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        const ChromaspacePresetValues values = captureCurrentChromaspacePresetValues(args.time);
        const std::string now = nowUtcIso8601();
        int existingIndex = chromaspaceUserPresetIndexByNameLocked(name);
        if (existingIndex >= 0) {
          if (!confirmChromaspacePresetOverwriteDialog(name)) return;
          auto& preset = chromaspacePresetStore().userPresets[static_cast<std::size_t>(existingIndex)];
          preset.values = values;
          preset.updatedAtUtc = now;
          preferred.userIndex = existingIndex;
        } else {
          ChromaspaceUserPreset preset{};
          preset.id = makePresetId("chromaspace");
          preset.name = name;
          preset.createdAtUtc = now;
          preset.updatedAtUtc = now;
          preset.values = values;
          chromaspacePresetStore().userPresets.push_back(preset);
          preferred.userIndex = static_cast<int>(chromaspacePresetStore().userPresets.size()) - 1;
        }
        saveChromaspacePresetStoreLocked();
      }
      if (auto* p = fetchStringParam("chromaspacePresetName")) p->setValue(name);
      syncChromaspacePresetMenuFromDisk(args.time, preferred);
      return;
    }
    if (paramName == "chromaspacePresetSaveDefaults") {
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        auto& preset = chromaspacePresetStore().defaultPreset;
        preset.id = "default";
        preset.name = kChromaspacePresetDefaultName;
        if (preset.createdAtUtc.empty()) preset.createdAtUtc = nowUtcIso8601();
        preset.updatedAtUtc = nowUtcIso8601();
        preset.values = captureCurrentChromaspacePresetValues(args.time);
        saveChromaspacePresetStoreLocked();
      }
      ChromaspacePresetSelection preferred{};
      preferred.kind = ChromaspacePresetSelection::Kind::Default;
      syncChromaspacePresetMenuFromDisk(args.time, preferred);
      showChromaspacePresetInfoDialog("Chromaspace defaults saved.\n\nThese new defaults will be used from the next plugin or host restart.");
      return;
    }
    if (paramName == "chromaspacePresetRestoreDefaults") {
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        auto& preset = chromaspacePresetStore().defaultPreset;
        preset.id = "default";
        preset.name = kChromaspacePresetDefaultName;
        if (preset.createdAtUtc.empty()) preset.createdAtUtc = nowUtcIso8601();
        preset.updatedAtUtc = nowUtcIso8601();
        preset.values = chromaspaceFactoryPresetValues();
        saveChromaspacePresetStoreLocked();
      }
      ChromaspacePresetSelection selection{};
      selection.kind = ChromaspacePresetSelection::Kind::Default;
      applyChromaspacePresetSelection(args.time, selection, "chromaspacePresetRestoreDefaults");
      showChromaspacePresetInfoDialog("Chromaspace defaults restored.\n\nThe protected Default preset has been reset to the factory developer defaults, and those defaults will be used for new instances after the next plugin or host restart.");
      return;
    }
    if (paramName == "chromaspacePresetUpdate") {
      const ChromaspacePresetSelection selection = selectedChromaspacePresetFromMenu(args.time);
      if (selection.kind != ChromaspacePresetSelection::Kind::User || selection.userIndex < 0) return;
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        if (selection.userIndex >= static_cast<int>(chromaspacePresetStore().userPresets.size())) return;
        auto& preset = chromaspacePresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)];
        preset.values = captureCurrentChromaspacePresetValues(args.time);
        preset.updatedAtUtc = nowUtcIso8601();
        saveChromaspacePresetStoreLocked();
      }
      syncChromaspacePresetMenuFromDisk(args.time, selection);
      return;
    }
    if (paramName == "chromaspacePresetRename") {
      const ChromaspacePresetSelection selection = selectedChromaspacePresetFromMenu(args.time);
      if (selection.kind != ChromaspacePresetSelection::Kind::User || selection.userIndex < 0) return;
      const std::string newName = sanitizePresetName(getStringValue("chromaspacePresetName", args.time, "Preset"), "Preset");
      if (chromaspacePresetNameReserved(newName)) {
        showChromaspacePresetInfoDialog("The preset name 'Default' is reserved. Use 'Save Defaults' to overwrite the protected Default preset.");
        return;
      }
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        if (selection.userIndex >= static_cast<int>(chromaspacePresetStore().userPresets.size())) return;
        auto& preset = chromaspacePresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)];
        if (chromaspacePresetUserNameExistsLocked(newName, &preset.id)) {
          showChromaspacePresetInfoDialog("A Chromaspace preset with that name already exists.");
          return;
        }
        preset.name = newName;
        preset.updatedAtUtc = nowUtcIso8601();
        saveChromaspacePresetStoreLocked();
      }
      if (auto* p = fetchStringParam("chromaspacePresetName")) p->setValue(newName);
      syncChromaspacePresetMenuFromDisk(args.time, selection);
      return;
    }
    if (paramName == "chromaspacePresetDelete") {
      const ChromaspacePresetSelection selection = selectedChromaspacePresetFromMenu(args.time);
      if (selection.kind != ChromaspacePresetSelection::Kind::User || selection.userIndex < 0) return;
      std::string presetName;
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        if (selection.userIndex >= static_cast<int>(chromaspacePresetStore().userPresets.size())) return;
        presetName = chromaspacePresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)].name;
      }
      if (!confirmChromaspacePresetDeleteDialog(presetName)) return;
      {
        std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
        ensureChromaspacePresetStoreLoadedLocked();
        if (selection.userIndex < 0 || selection.userIndex >= static_cast<int>(chromaspacePresetStore().userPresets.size())) return;
        chromaspacePresetStore().userPresets.erase(chromaspacePresetStore().userPresets.begin() + selection.userIndex);
        saveChromaspacePresetStoreLocked();
      }
      syncChromaspacePresetMenuFromDisk(args.time);
      return;
    }
    if (paramName == "openCubeViewer") {
      cubeViewerDebugLog("changedParam(openCubeViewer)");
      openCubeViewerSession(args.time);
      return;
    }
    if (paramName == "closeCubeViewer") {
      cubeViewerDebugLog("changedParam(closeCubeViewer)");
      closeCubeViewerSession();
      return;
    }
    if (paramName == "cubeViewerModeToggle") {
      const bool nextDrawOnImage = !currentDrawOnImageMode(args.time);
      if (auto* p = fetchBooleanParam("cubeViewerDrawOnImageEnabled")) {
        p->setValue(nextDrawOnImage);
      }
      if (nextDrawOnImage) {
        if (auto* p = fetchBooleanParam("cubeViewerIdentityOverlayEnabledDraw")) {
          p->setValue(true);
        }
      }
      cubeViewerDebugLog(std::string("changedParam(cubeViewerModeToggle) -> ") + (nextDrawOnImage ? "draw-on-image" : "plot"));
      updateDrawOnImageModeUi(args.time);
      updateCircularHslToggleVisibility(args.time);
      updateCircularHsvToggleVisibility(args.time);
      setGroupOpenState("grp_cube_viewer_identity_overlay", nextDrawOnImage);
      updateNormConeToggleVisibility(args.time);
      invalidateCubeViewerCloudState();
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerModeToggle");
      }
      return;
    }
    if (paramName == "cubeViewerLive") {
      cubeViewerLive_ = getBoolValue("cubeViewerLive", args.time, true);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerLive) -> ") + (cubeViewerLive_ ? "1" : "0"));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerLive");
      }
      return;
    }
    if (paramName == "cubeViewerUpdateMode") {
      cubeViewerUpdateMode_ = getChoiceValue("cubeViewerUpdateMode", args.time, 0);
      deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerUpdateMode) -> ") +
                         viewerUpdateModeLabelForIndex(cubeViewerUpdateMode_));
      syncChromaspacePresetMenuState(args.time);
      return;
    }
    if (paramName == "cubeViewerQuality") {
      cubeViewerQuality_ = getChoiceValue("cubeViewerQuality", args.time, 0);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerQuality) -> ") + qualityLabelForIndex(cubeViewerQuality_));
      resolveOverlaySizeParamIfAuto(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerQuality");
      }
      return;
    }
    if (paramName == "cubeViewerScale") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerScale) -> ") +
                         scaleLabelForIndex(getChoiceValue("cubeViewerScale", args.time, 3)));
      resolveOverlaySizeParamIfAuto(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerScale");
      }
      return;
    }
    if (paramName == "cubeViewerPointSize") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerPointSize) -> ") +
                         std::to_string(getDoubleValue("cubeViewerPointSize", args.time, 1.4)));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerPointSize");
      }
      return;
    }
    if (paramName == "cubeViewerPlotStyle") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerPlotStyle) -> ") +
                         plotStyleLabelForIndex(getChoiceValue("cubeViewerPlotStyle", args.time, 0)));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerPlotStyle");
      }
      return;
    }
    if (paramName == "cubeViewerColorSaturation") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerColorSaturation) -> ") +
                         std::to_string(getDoubleValue("cubeViewerColorSaturation", args.time, 2.0)));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerColorSaturation");
      }
      return;
    }
    if (paramName == "cubeViewerSamplingMode") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerSamplingMode) -> ") +
                         samplingModeLabelForIndex(getChoiceValue("cubeViewerSamplingMode", args.time, 0)));
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerSamplingMode");
      }
      return;
    }
    if (paramName == "cubeViewerOccupancyGuidedFill") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerOccupancyGuidedFill) -> ") +
                         (getBoolValue("cubeViewerOccupancyGuidedFill", args.time, false) ? "1" : "0"));
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerOccupancyGuidedFill");
      }
      return;
    }
    if (paramName == "cubeViewerPointShape") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerPointShape) -> ") +
                         pointShapeLabelForIndex(getChoiceValue("cubeViewerPointShape", args.time, 0)));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerPointShape");
      }
      return;
    }
    if (paramName == "cubeViewerShowOverflow") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerShowOverflow) -> ") +
                         (getBoolValue("cubeViewerShowOverflow", args.time, false) ? "1" : "0"));
      updateDrawOnImageModeUi(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerShowOverflow");
        (void)trySendCachedCloud(args.time, "cubeViewerShowOverflow");
      }
      return;
    }
    if (paramName == "cubeViewerPlotDisplayLinear" ||
        paramName == "cubeViewerPlotDisplayLinearTransfer") {
      cubeViewerDebugLog(std::string("changedParam(") + paramName + ")");
      updateDrawOnImageModeUi(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, paramName);
        (void)trySendCachedCloud(args.time, paramName);
      }
      return;
    }
    if (paramName == "cubeViewerNeutralRadius") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerNeutralRadius) -> ") +
                         std::to_string(currentNeutralRadiusValue(args.time)));
      requestCubeViewerCloudResample();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerNeutralRadius");
      }
      return;
    }
    if (paramName == "cubeViewerHighlightOverflow") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerHighlightOverflow) -> ") +
                         (getBoolValue("cubeViewerHighlightOverflow", args.time, true) ? "1" : "0"));
      syncShowOverflowSupport(args.time);
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerHighlightOverflow");
        (void)trySendCachedCloud(args.time, "cubeViewerHighlightOverflow");
      }
      return;
    }
    if (paramName == "cubeViewerLassoRegionMode") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerLassoRegionMode) -> ") +
                         (currentLassoRegionSlicingEnabled(args.time) ? "1" : "0"));
      syncCubeSlicingUi(args.time);
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerLassoRegionMode");
      }
      redrawOverlays();
      return;
    }
    if (paramName == "cubeViewerSliceRed" || paramName == "cubeViewerSliceGreen" ||
        paramName == "cubeViewerSliceBlue" || paramName == "cubeViewerSliceCyan" ||
        paramName == "cubeViewerSliceYellow" || paramName == "cubeViewerSliceMagenta") {
      cubeViewerDebugLog(std::string("changedParam(") + paramName + ") -> " +
                         (getBoolValue(paramName, args.time, true) ? "1" : "0"));
      syncCubeSlicingUi(args.time);
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, paramName);
      }
      return;
    }
    if (paramName == "cubeViewerLassoOperation") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerLassoOperation) -> ") +
                         (currentLassoSubtractOperation(args.time) ? "subtract" : "add"));
      redrawOverlays();
      return;
    }
    if (paramName == "cubeViewerLassoUndo") {
      undoLassoRegionStroke(args.time, "cubeViewerLassoUndo");
      return;
    }
    if (paramName == "cubeViewerLassoReset") {
      resetLassoRegion(args.time, "cubeViewerLassoReset");
      return;
    }
    if (paramName == "cubeViewerLassoData") {
      if (!suppressLassoDataChangedHandling_) {
        syncLassoDataChange(args.time, "cubeViewerLassoData");
      }
      return;
    }
    if (paramName == "cubeViewerOverflowHighlightColor") {
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerOverflowHighlightColor");
        (void)trySendCachedCloud(args.time, "cubeViewerOverflowHighlightColor");
      }
      return;
    }
    if (paramName == "cubeViewerBackgroundColor") {
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerBackgroundColor");
      }
      return;
    }
    if (paramName == "cubeViewerOnTop") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerOnTop) -> ") +
                         (getBoolValue("cubeViewerOnTop", args.time, true) ? "1" : "0"));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerOnTop");
      }
      return;
    }
    if (paramName == "cubeViewerIdentityOverlayEnabled") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerIdentityOverlayEnabled) -> ") +
                         (getBoolValue("cubeViewerIdentityOverlayEnabled", args.time, false) ? "1" : "0"));
      updateDrawOnImageModeUi(args.time);
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerIdentityOverlayEnabled");
      }
      return;
    }
    if (paramName == "cubeViewerIdentityOverlaySize") {
      const int requested = getIntValue("cubeViewerIdentityOverlaySize", args.time, 29);
      const int qualityIndex = getChoiceValue("cubeViewerQuality", args.time, cubeViewerQuality_);
      const int scaleIndex = getChoiceValue("cubeViewerScale", args.time, 3);
      const int resolved = currentDrawOnImageMode(args.time)
                               ? clampOverlayCubeSize(requested)
                               : resolvedOverlayCubeSize(requested, qualityIndex, scaleIndex);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerIdentityOverlaySize) -> requested=") +
                         std::to_string(requested) + " resolved=" + std::to_string(resolved));
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerIdentityOverlaySize");
      }
      return;
    }
    if (paramName == "cubeViewerIdentityOverlayRamp") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerIdentityOverlayRamp) -> ") +
                         (getBoolValue("cubeViewerIdentityOverlayRamp", args.time, false) ? "1" : "0"));
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerIdentityOverlayRamp");
      }
      return;
    }
    if (paramName == "cubeViewerIdentityOverlayEnabledDraw") {
      const bool enabled = getBoolValue("cubeViewerIdentityOverlayEnabledDraw", args.time, false);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerIdentityOverlayEnabledDraw) -> ") +
                         (enabled ? "1" : "0"));
      updateDrawOnImageModeUi(args.time);
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerIdentityOverlayEnabledDraw");
      }
      return;
    }
    if (paramName == "cubeViewerIdentityOverlayRampDraw") {
      const bool enabled = getBoolValue("cubeViewerIdentityOverlayRampDraw", args.time, false);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerIdentityOverlayRampDraw) -> ") +
                         (enabled ? "1" : "0"));
      updateDrawOnImageModeUi(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerIdentityOverlayRampDraw");
      }
      return;
    }
    if (paramName == "cubeViewerSampleDrawnCubeOnly") {
      const bool useInstance1 = getBoolValue("cubeViewerSampleDrawnCubeOnly", args.time, false);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerSampleDrawnCubeOnly) -> ") +
                         (useInstance1 ? "1" : "0"));
      if (!useInstance1 && !getBoolValue("cubeViewerReadGrayRamp", args.time, false)) {
        if (auto* p = fetchBooleanParam("cubeViewerShowIdentityOnly")) p->setValue(false);
      }
      syncIdentityReadbackUi(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerSampleDrawnCubeOnly");
      }
      return;
    }
    if (paramName == "cubeViewerReadGrayRamp") {
      const bool readGrayRamp = getBoolValue("cubeViewerReadGrayRamp", args.time, false);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerReadGrayRamp) -> ") +
                         (readGrayRamp ? "1" : "0"));
      if (!readGrayRamp && !getBoolValue("cubeViewerSampleDrawnCubeOnly", args.time, false)) {
        if (auto* p = fetchBooleanParam("cubeViewerShowIdentityOnly")) p->setValue(false);
      }
      syncIdentityReadbackUi(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerReadGrayRamp");
      }
      return;
    }
    if (paramName == "cubeViewerShowIdentityOnly") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerShowIdentityOnly) -> ") +
                         (getBoolValue("cubeViewerShowIdentityOnly", args.time, false) ? "1" : "0"));
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerShowIdentityOnly");
      }
      return;
    }
    if (paramName == "cubeViewerSampleDrawnCubeSize") {
      const int requested = getIntValue("cubeViewerSampleDrawnCubeSize", args.time, 29);
      const int qualityIndex = getChoiceValue("cubeViewerQuality", args.time, cubeViewerQuality_);
      const int scaleIndex = getChoiceValue("cubeViewerScale", args.time, 3);
      const int resolved = clampOverlayCubeSize(requested);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerSampleDrawnCubeSize) -> requested=") +
                         std::to_string(requested) + " resolved=" + std::to_string(resolved));
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerSampleDrawnCubeSize");
      }
      return;
    }
    if (paramName == "cubeViewerPlotModel") {
      const std::string plotMode = currentPlotMode(args.time);
      cubeViewerDebugLog(std::string("changedParam(cubeViewerPlotModel) -> ") + plotMode);
      updateDrawOnImageModeUi(args.time);
      syncShowOverflowSupport(args.time);
      syncCubeSlicingUi(args.time);
      updateCircularHslToggleVisibility(args.time);
      updateCircularHsvToggleVisibility(args.time);
      updateNormConeToggleVisibility(args.time);
      invalidateCubeViewerCloudState();
      syncChromaspacePresetMenuState(args.time);
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerPlotModel");
        (void)trySendCachedCloud(args.time, "cubeViewerPlotModel");
      }
      return;
    }
    if (paramName == "cubeViewerCircularHsl") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerCircularHsl) -> ") +
                         (getBoolValue("cubeViewerCircularHsl", args.time, false) ? "1" : "0"));
      invalidateCubeViewerCloudState();
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerCircularHsl");
        (void)trySendCachedCloud(args.time, "cubeViewerCircularHsl");
      }
      return;
    }
    if (paramName == "cubeViewerCircularHsv") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerCircularHsv) -> ") +
                         (getBoolValue("cubeViewerCircularHsv", args.time, false) ? "1" : "0"));
      invalidateCubeViewerCloudState();
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerCircularHsv");
        (void)trySendCachedCloud(args.time, "cubeViewerCircularHsv");
      }
      return;
    }
    if (paramName == "cubeViewerNormConeNormalized") {
      cubeViewerDebugLog(std::string("changedParam(cubeViewerNormConeNormalized) -> ") +
                         (getBoolValue("cubeViewerNormConeNormalized", args.time, true) ? "1" : "0"));
      invalidateCubeViewerCloudState();
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, "cubeViewerNormConeNormalized");
        (void)trySendCachedCloud(args.time, "cubeViewerNormConeNormalized");
      }
      return;
    }
    if (paramName == "cubeViewerChromaticityInputPrimaries" ||
        paramName == "cubeViewerChromaticityInputTransfer" ||
        paramName == "cubeViewerChromaticityReferenceBasis" ||
        paramName == "cubeViewerChromaticityOverlayPrimaries") {
      cubeViewerDebugLog(std::string("changedParam(") + paramName + ")");
      invalidateCubeViewerCloudState();
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, paramName);
        (void)trySendCachedCloud(args.time, paramName);
      }
      return;
    }
    if (paramName == "cubeViewerChromaticityPlanckianLocus") {
      cubeViewerDebugLog(std::string("changedParam(") + paramName + ")");
      if (viewerSessionRequested()) {
        pushParamsUpdate(args.time, paramName);
      }
      return;
    }
    if (paramName == "supportWebsite") {
      openUrl(kWebsiteUrl);
      return;
    }
    if (paramName == "supportLatestReleases") {
      if (!openPluginManager()) openUrl(kReleasesUrl);
      return;
    }
    if (paramName == "supportReportIssue") {
      openUrl(kIssueUrl);
      return;
    }
  }

 private:
  Clip* dstClip_ = nullptr;
  Clip* srcClip_ = nullptr;

  std::mutex stateMutex_;
  std::mutex ioMutex_;
  std::condition_variable ioCv_;
  std::thread ioThread_;
  std::thread statusThread_;
  bool ioStop_ = false;
  bool statusStop_ = false;
  PendingMessage pendingParams_;
  PendingMessage pendingCloud_;
  std::atomic<bool> cloudQueuedOrInFlight_{false};
  std::atomic<bool> deferredLatestCloudRefresh_{false};
  std::atomic<int64_t> lastViewerTransportActivityMs_{0};
  std::mutex statusMutex_;
  std::string pendingStatusText_;
  bool statusDirty_ = false;

  std::string senderId_;
  bool cubeViewerRequested_ = false;
  bool cubeViewerConnected_ = false;
  bool cubeViewerWindowUsable_ = false;
  bool cubeViewerLive_ = true;
  int cubeViewerUpdateMode_ = 0;
  int cubeViewerQuality_ = 0;
  bool cubeViewerInputCloudRefreshPending_ = false;
  int playbackRenderBurstCount_ = 0;
  bool suppressLassoDataChangedHandling_ = false;
  bool suppressChromaspacePresetChangedHandling_ = false;
  bool chromaspacePresetMenuHasCustom_ = false;
  int chromaspacePresetCustomIndex_ = -1;
  int chromaspacePresetMenuUserCount_ = 0;
  CachedCloud cachedCloud_;
  CachedIdentityStripCloud cachedIdentityStripCloud_;
  double lastCloudTime_ = std::numeric_limits<double>::quiet_NaN();
  std::chrono::steady_clock::time_point lastCloudBuiltAt_{};
  std::chrono::steady_clock::time_point lastRenderSeenAt_{};
  std::chrono::steady_clock::time_point previewModeUntil_{};
  std::chrono::steady_clock::time_point lastHeartbeatAt_{};
  ViewerProbeResult lastLoggedHeartbeatProbe_{};
  bool hasLoggedHeartbeatProbe_ = false;
  std::string lastCloudSourceId_;
  std::string lastCloudSettingsKey_;
  std::string statusCache_;
  std::vector<float> stageSrc_;
  std::mutex stageMutex_;
  std::string buildSenderId() const {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << "cube-" << now << "-" << std::this_thread::get_id();
    return oss.str();
  }

  // Cloud caching prefers newer or equivalent builds, but rejects regressions where a smaller source image
  // would overwrite a more complete cloud for the same settings.
  bool shouldAcceptBuiltCloudLocked(
      const CloudBuildResult& built,
      const std::string& sourceId,
      const std::string& settingsKey,
      std::string* reason) const {
    if (!built.success || built.sourceWidth <= 0 || built.sourceHeight <= 0) {
      if (reason) *reason = "invalid-build";
      return false;
    }
    if (!cachedCloud_.valid) {
      if (reason) *reason = "cache-empty";
      return true;
    }
    if (cachedCloud_.quality != built.quality || cachedCloud_.resolution != built.resolution) {
      if (reason) *reason = "quality-changed";
      return true;
    }
    if (cachedCloud_.sourceId != sourceId || cachedCloud_.settingsKey != settingsKey) {
      if (reason) *reason = "source-changed";
      return true;
    }
    if (cachedCloud_.contentHash != 0 && built.contentHash != 0 && cachedCloud_.contentHash == built.contentHash) {
      if (reason) *reason = "unchanged-content";
      return false;
    }
    const int cachedPixels = cachedCloud_.sourceWidth * cachedCloud_.sourceHeight;
    const int builtPixels = built.sourceWidth * built.sourceHeight;
    if (cachedPixels > 0 && builtPixels < cachedPixels) {
      if (reason) *reason = "smaller-source";
      return false;
    }
    if (reason) *reason = "same-or-larger-source";
    return true;
  }

  // Promote a freshly built cloud into the reusable cache only after the settings/source checks above agree
  // that it is the best representation of the current state.
  bool promoteBuiltCloud(
      const CloudBuildResult& built,
      const std::string& sourceId,
      const std::string& settingsKey,
      bool* changed = nullptr) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    std::string decisionReason;
    if (!shouldAcceptBuiltCloudLocked(built, sourceId, settingsKey, &decisionReason)) {
      if (changed) *changed = false;
      cubeViewerDebugLog(std::string("Rejected cloud payload: reason=") + decisionReason +
                         " built=" + std::to_string(built.sourceWidth) + "x" + std::to_string(built.sourceHeight) +
                         " cached=" + std::to_string(cachedCloud_.sourceWidth) + "x" + std::to_string(cachedCloud_.sourceHeight) +
                         " quality=" + built.quality +
                         " res=" + std::to_string(built.resolution));
      return decisionReason == "unchanged-content";
    }
    cachedCloud_.payload = built.payload;
    cachedCloud_.pointsPayload = built.pointsPayload;
    cachedCloud_.paramHash = built.paramHash;
    cachedCloud_.quality = built.quality;
    cachedCloud_.sourceId = sourceId;
    cachedCloud_.settingsKey = settingsKey;
    cachedCloud_.fastBlob = built.fastBlob;
    cachedCloud_.samples = built.samples;
    cachedCloud_.contentHash = built.contentHash;
    cachedCloud_.sampleCount = built.sampleCount;
    cachedCloud_.resolution = built.resolution;
    cachedCloud_.sourceWidth = built.sourceWidth;
    cachedCloud_.sourceHeight = built.sourceHeight;
    cachedCloud_.valid = true;
    if (changed) *changed = true;
    cubeViewerDebugLog(std::string("Accepted cloud payload: reason=") + decisionReason +
                       " source=" + std::to_string(built.sourceWidth) + "x" + std::to_string(built.sourceHeight) +
                       " quality=" + built.quality +
                       " res=" + std::to_string(built.resolution));
    return true;
  }

  void copySourceToDestination(Image* src, Image* dst, const OfxRectI& renderWindow) {
    const int width = renderWindow.x2 - renderWindow.x1;
    if (!src) {
      for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
        float* dstPix = reinterpret_cast<float*>(dst->getPixelAddress(renderWindow.x1, y));
        if (!dstPix) continue;
        std::fill(dstPix, dstPix + width * 4, 0.0f);
      }
      return;
    }

    for (int y = renderWindow.y1; y < renderWindow.y2; ++y) {
      float* dstPix = reinterpret_cast<float*>(dst->getPixelAddress(renderWindow.x1, y));
      const float* srcPix = reinterpret_cast<const float*>(src->getPixelAddress(renderWindow.x1, y));
      if (!dstPix || !srcPix) continue;
      std::memcpy(dstPix, srcPix, static_cast<size_t>(width) * 4u * sizeof(float));
    }
  }

  // Stage: synthesize the image-space strip used by draw-on-image mode.
  // This is the shared contract that downstream "sample drawn cube only" reads back.
  bool buildIdentityOverlayStripData(const OfxRectI& bounds, const OfxRectI& renderWindow, double time, OverlayStripData* out) {
    if (!out || !getBoolValue("cubeViewerDrawOnImageEnabled", time, false)) return false;
    if (!currentIdentityOverlayEnabled(time)) return false;
    const int imageWidth = bounds.x2 - bounds.x1;
    const int imageHeight = bounds.y2 - bounds.y1;
    if (imageWidth <= 0 || imageHeight <= 0) return false;
    const bool overlayRamp = currentIdentityOverlayRamp(time);
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", time, 29);
    const int cubeSize = clampOverlayCubeSize(requestedSize);
    const int denom = std::max(1, cubeSize - 1);
    const float cellWidth = identityStripCellWidth(imageWidth, cubeSize);
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    if (!computeIdentityStripLayout(bounds, cubeSize, overlayRamp, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) return false;
    const int drawY1 = std::max(renderWindow.y1, bounds.y1);
    const int drawY2 = std::min(renderWindow.y2, overlayRamp ? rampY2 : cubeY2);
    if (drawY1 >= drawY2) return false;
    OverlayStripData overlay{};
    overlay.x1 = renderWindow.x1;
    overlay.y1 = drawY1;
    overlay.width = renderWindow.x2 - renderWindow.x1;
    overlay.height = drawY2 - drawY1;
    if (overlay.width <= 0 || overlay.height <= 0) return false;
    overlay.pixels.assign(static_cast<size_t>(overlay.width) * static_cast<size_t>(overlay.height) * 4u, 0.0f);
    for (int y = drawY1; y < drawY2; ++y) {
      float* row = overlay.pixels.data() + static_cast<size_t>(y - drawY1) * static_cast<size_t>(overlay.width) * 4u;
      const bool inRampBand = overlayRamp && y >= rampY1 && y < rampY2;
      const bool inCubeBand = y >= cubeY1 && y < cubeY2;
      if (!inRampBand && !inCubeBand) continue;
      for (int x = renderWindow.x1; x < renderWindow.x2; ++x) {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
        if (inRampBand) {
          const float t = imageWidth <= 1 ? 0.0f
                                          : static_cast<float>(x - bounds.x1) / static_cast<float>(std::max(1, imageWidth - 1));
          r = g = b = clamp01(t);
        } else {
          const float localX = static_cast<float>(x - bounds.x1);
          const int blueIndex = std::clamp(static_cast<int>(std::floor(localX / cellWidth)), 0, denom);
          const float layerStart = static_cast<float>(blueIndex) * cellWidth;
          const float layerOffset = localX - layerStart;
          r = cellWidth <= 1.0f ? 0.0f : clamp01(layerOffset / std::max(1.0f, cellWidth - 1.0f));
          const int cubeBandY = y - cubeY1;
          g = stripHeight <= 1 ? 1.0f
                               : clamp01(static_cast<float>(cubeBandY) / static_cast<float>(std::max(1, stripHeight - 1)));
          b = static_cast<float>(blueIndex) / static_cast<float>(denom);
        }
        float* px = row + static_cast<size_t>(x - renderWindow.x1) * 4u;
        px[0] = r;
        px[1] = g;
        px[2] = b;
        px[3] = 1.0f;
      }
    }
    *out = std::move(overlay);
    return true;
  }

  void applyIdentityOverlayToDestination(Image* dst, const OfxRectI& renderWindow, double time) {
    if (!dst) return;
    OverlayStripData overlay{};
    if (!buildIdentityOverlayStripData(dst->getBounds(), renderWindow, time, &overlay)) return;
    const size_t packedRowBytes = static_cast<size_t>(overlay.width) * 4u * sizeof(float);
    for (int rowIndex = 0; rowIndex < overlay.height; ++rowIndex) {
      float* row = reinterpret_cast<float*>(dst->getPixelAddress(overlay.x1, overlay.y1 + rowIndex));
      if (!row) continue;
      const float* src = overlay.pixels.data() + static_cast<size_t>(rowIndex) * static_cast<size_t>(overlay.width) * 4u;
      std::memcpy(row, src, packedRowBytes);
    }
  }

  bool ensureStageBuffer(size_t pixelCount) {
    std::lock_guard<std::mutex> lock(stageMutex_);
    const size_t needed = pixelCount * 4u;
    if (stageSrc_.size() < needed) {
      stageSrc_.assign(needed, 0.0f);
    }
    return true;
  }

  float* stageSrcPtr() {
    return stageSrc_.empty() ? nullptr : stageSrc_.data();
  }

  bool getBoolValue(const std::string& name, double time, bool fallback) const {
    if (auto* p = fetchBooleanParam(name)) {
      bool value = fallback;
      p->getValueAtTime(time, value);
      return value;
    }
    return fallback;
  }

  int getChoiceValue(const std::string& name, double time, int fallback) const {
    if (auto* p = fetchChoiceParam(name)) {
      int value = fallback;
      p->getValueAtTime(time, value);
      return value;
    }
    return fallback;
  }

  double getDoubleValue(const std::string& name, double time, double fallback) const {
    if (auto* p = fetchDoubleParam(name)) {
      double value = fallback;
      p->getValueAtTime(time, value);
      return value;
    }
    return fallback;
  }

  std::array<double, 3> getRGBValue(const std::string& name, double time, const std::array<double, 3>& fallback) {
    if (auto* p = fetchRGBParam(name)) {
      double r = fallback[0];
      double g = fallback[1];
      double b = fallback[2];
      p->getValueAtTime(time, r, g, b);
      return {r, g, b};
    }
    return fallback;
  }

  int getIntValue(const std::string& name, double time, int fallback) const {
    if (auto* p = fetchIntParam(name)) {
      int value = fallback;
      p->getValueAtTime(time, value);
      return value;
    }
    return fallback;
  }

  void setStatusLabel(const std::string& text) {
    bool changed = false;
    {
      std::lock_guard<std::mutex> lock(statusMutex_);
      if (pendingStatusText_ != text) {
        pendingStatusText_ = text;
        statusDirty_ = true;
        changed = true;
      }
    }
    if (changed) {
      cubeViewerDebugLog(std::string("Viewer status -> ") + text);
    }
  }

  void flushStatusLabelToHost() {
    std::string text;
    {
      std::lock_guard<std::mutex> lock(statusMutex_);
      if (!statusDirty_) return;
      text = pendingStatusText_;
      statusDirty_ = false;
    }
    if (statusCache_ == text) return;
    statusCache_ = text;
    if (auto* p = fetchStringParam("cubeViewerStatus")) p->setValue(text);
  }

  void resolveOverlaySizeParamIfAuto(double time) {
    if (currentDrawOnImageMode(time)) return;
    auto* p = fetchIntParam("cubeViewerIdentityOverlaySize");
    if (!p) return;
    int requested = 25;
    p->getValueAtTime(time, requested);
    if (clampOverlayCubeSize(requested) != 25) return;
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int scaleIndex = getChoiceValue("cubeViewerScale", time, 3);
    const int resolved = resolvedOverlayCubeSize(requested, qualityIndex, scaleIndex);
    if (resolved != requested) {
      p->setValue(resolved);
    }
  }

  std::string currentSourceMode(double time) {
    (void)time;
    return "input";
  }

  bool currentIdentityOverlayEnabled(double time) {
    return currentDrawOnImageMode(time)
               ? getBoolValue("cubeViewerIdentityOverlayEnabledDraw", time, false)
               : getBoolValue("cubeViewerIdentityOverlayEnabled", time, false);
  }

  bool currentIdentityOverlayRamp(double time) {
    return currentDrawOnImageMode(time)
               ? getBoolValue("cubeViewerIdentityOverlayRampDraw", time, false)
               : getBoolValue("cubeViewerIdentityOverlayRamp", time, false);
  }

  std::string currentPlotMode(double time) {
    switch (getChoiceValue("cubeViewerPlotModel", time, 0)) {
      case 1: return "hsl";
      case 2: return "hsv";
      case 3: return "chen";
      case 4: return "norm_cone";
      case 5: return "jp_conical";
      case 6: return "reuleaux";
      case 7: return "chromaticity";
      default: return "rgb";
    }
  }

  bool currentChromaticityPlotMode(double time) {
    return !currentDrawOnImageMode(time) && currentPlotMode(time) == "chromaticity";
  }

  bool currentPlotDisplayLinearAllowed(double time) {
    return !currentDrawOnImageMode(time) && !currentChromaticityPlotMode(time);
  }

  bool currentPlotDisplayLinearEnabled(double time) {
    return currentPlotDisplayLinearAllowed(time) &&
           getBoolValue("cubeViewerPlotDisplayLinear", time, false);
  }

  int currentPlotDisplayLinearTransferChoice(double time) const {
    return std::clamp(getChoiceValue("cubeViewerPlotDisplayLinearTransfer", time,
                                     plotLinearTransferChoiceIndex(
                                         WorkshopColor::TransferFunctionId::Gamma24)),
                      0, static_cast<int>(plotLinearTransferChoices().size()) - 1);
  }

  WorkshopColor::TransferFunctionId currentPlotDisplayLinearTransferId(double time) const {
    return plotLinearTransferIdFromChoice(currentPlotDisplayLinearTransferChoice(time));
  }

  bool currentDrawOnImageMode(double time) {
    return getBoolValue("cubeViewerDrawOnImageEnabled", time, false);
  }

  bool currentUseInstance1Requested(double time) {
    return !currentDrawOnImageMode(time) &&
           (getBoolValue("cubeViewerSampleDrawnCubeOnly", time, false) ||
            getBoolValue("cubeViewerReadGrayRamp", time, false));
  }

  bool currentUseInstance1(double time) {
    return currentUseInstance1Requested(time);
  }

  bool currentReadIdentityPlot(double time) {
    return !currentDrawOnImageMode(time) && getBoolValue("cubeViewerSampleDrawnCubeOnly", time, false);
  }

  bool currentReadGrayRamp(double time) {
    return !currentDrawOnImageMode(time) && getBoolValue("cubeViewerReadGrayRamp", time, false);
  }

  bool currentShowIdentityOnly(double time) {
    return currentUseInstance1(time) && getBoolValue("cubeViewerShowIdentityOnly", time, false);
  }

  bool currentOccupancyGuidedFill(double time) {
    return !currentDrawOnImageMode(time) && getBoolValue("cubeViewerOccupancyGuidedFill", time, false);
  }

  bool currentShowOverflow(double time) {
    return !currentDrawOnImageMode(time) &&
            (currentPlotMode(time) == "rgb" || currentPlotMode(time) == "hsl" ||
             currentPlotMode(time) == "hsv" || currentPlotMode(time) == "chen" ||
             currentPlotMode(time) == "jp_conical" || currentPlotMode(time) == "reuleaux" ||
             currentPlotMode(time) == "chromaticity") &&
           getBoolValue("cubeViewerShowOverflow", time, false);
  }

  bool currentCircularHsl(double time) {
    return !currentDrawOnImageMode(time) &&
           currentPlotMode(time) == "hsl" &&
           getBoolValue("cubeViewerCircularHsl", time, false);
  }

  bool currentCircularHsv(double time) {
    return !currentDrawOnImageMode(time) &&
           currentPlotMode(time) == "hsv" &&
           getBoolValue("cubeViewerCircularHsv", time, false);
  }

  VolumeSlicingMode currentVolumeSlicingMode(double time) {
    return getBoolValue("cubeViewerLassoRegionMode", time, false)
               ? VolumeSlicingMode::LassoRegion
               : VolumeSlicingMode::HueSectors;
  }

  bool currentCubeSlicingSupported(double time) {
    return !currentDrawOnImageMode(time);
  }

  bool currentHueSectorSlicingAllowed(double time) {
    return currentCubeSlicingSupported(time) && !currentChromaticityPlotMode(time);
  }

  bool currentNeutralRadiusSlicingAllowed(double time) {
    return currentCubeSlicingSupported(time) &&
           !currentChromaticityPlotMode(time);
  }

  double currentNeutralRadiusValue(double time) {
    return std::clamp(getDoubleValue("cubeViewerNeutralRadius", time, 1.0), 0.0, 1.0);
  }

  bool currentNeutralRadiusSlicingEnabled(double time) {
    return currentNeutralRadiusSlicingAllowed(time) &&
           !currentShowOverflow(time) &&
           currentNeutralRadiusValue(time) < 0.999999;
  }

  bool currentVolumeSlicingEnabled(double time) {
    if (!currentCubeSlicingSupported(time)) return false;
    if (getBoolValue("cubeViewerLassoRegionMode", time, false)) return true;
    if (currentHueSectorSlicingAllowed(time) && anyCubeSliceRegionSelected(time)) return true;
    return currentNeutralRadiusSlicingEnabled(time);
  }

  bool currentHueSectorSlicingEnabled(double time) {
    return currentHueSectorSlicingAllowed(time) &&
           currentVolumeSlicingMode(time) == VolumeSlicingMode::HueSectors &&
           anyCubeSliceRegionSelected(time);
  }

  bool currentLassoRegionSlicingEnabled(double time) {
    return currentVolumeSlicingEnabled(time) &&
           currentVolumeSlicingMode(time) == VolumeSlicingMode::LassoRegion;
  }

  bool anyCubeSliceRegionSelected(double time) {
    return getBoolValue("cubeViewerSliceRed", time, false) ||
           getBoolValue("cubeViewerSliceGreen", time, false) ||
           getBoolValue("cubeViewerSliceBlue", time, false) ||
           getBoolValue("cubeViewerSliceCyan", time, false) ||
           getBoolValue("cubeViewerSliceYellow", time, false) ||
           getBoolValue("cubeViewerSliceMagenta", time, false);
  }

  SliceSelectionSpec currentHueSectorSliceSpec(double time) {
    SliceSelectionSpec spec{};
    const std::string plotMode = currentPlotMode(time);
    if (plotMode == "hsl") {
      spec.plotMode = SlicePlotModeKind::Hsl;
    } else if (plotMode == "hsv") {
      spec.plotMode = SlicePlotModeKind::Hsv;
    } else if (plotMode == "chen") {
      spec.plotMode = SlicePlotModeKind::Chen;
    } else if (plotMode == "rgb_to_cone") {
      spec.plotMode = SlicePlotModeKind::RgbToCone;
    } else if (plotMode == "jp_conical") {
      spec.plotMode = SlicePlotModeKind::JpConical;
    } else if (plotMode == "norm_cone") {
      spec.plotMode = SlicePlotModeKind::NormCone;
    } else if (plotMode == "reuleaux") {
      spec.plotMode = SlicePlotModeKind::Reuleaux;
    } else {
      spec.plotMode = SlicePlotModeKind::Rgb;
    }
    spec.showOverflow = currentShowOverflow(time);
    spec.circularHsl = currentCircularHsl(time);
    spec.circularHsv = currentCircularHsv(time);
    spec.normConeNormalized = getBoolValue("cubeViewerNormConeNormalized", time, true);
    spec.enabled = currentHueSectorSlicingEnabled(time);
    spec.neutralRadiusEnabled = currentNeutralRadiusSlicingEnabled(time);
    spec.neutralRadius = static_cast<float>(currentNeutralRadiusValue(time));
    spec.cubeSliceRed = getBoolValue("cubeViewerSliceRed", time, true);
    spec.cubeSliceGreen = getBoolValue("cubeViewerSliceGreen", time, false);
    spec.cubeSliceBlue = getBoolValue("cubeViewerSliceBlue", time, false);
    spec.cubeSliceCyan = getBoolValue("cubeViewerSliceCyan", time, false);
    spec.cubeSliceYellow = getBoolValue("cubeViewerSliceYellow", time, false);
    spec.cubeSliceMagenta = getBoolValue("cubeViewerSliceMagenta", time, false);
    return spec;
  }

  int currentChromaticityInputPrimariesChoice(double time) {
    return std::clamp(getChoiceValue("cubeViewerChromaticityInputPrimaries", time,
                                     WorkshopColor::primariesChoiceIndex(
                                         WorkshopColor::ColorPrimariesId::DavinciWideGamut)),
                      0, static_cast<int>(WorkshopColor::primariesCount()) - 1);
  }

  int currentChromaticityInputTransferChoice(double time) {
    return std::clamp(getChoiceValue("cubeViewerChromaticityInputTransfer", time,
                                     WorkshopColor::transferFunctionChoiceIndex(
                                         WorkshopColor::TransferFunctionId::DavinciIntermediate)),
                      0, static_cast<int>(WorkshopColor::transferFunctionCount()) - 1);
  }

  int currentChromaticityReferenceBasisChoice(double time) {
    return std::clamp(getChoiceValue("cubeViewerChromaticityReferenceBasis", time, 0), 0, 1);
  }

  int currentChromaticityOverlayPrimariesChoice(double time) {
    return std::clamp(getChoiceValue("cubeViewerChromaticityOverlayPrimaries", time,
                                     WorkshopColor::overlayPrimariesChoiceIndex(
                                         true, WorkshopColor::ColorPrimariesId::Rec709)),
                      0, static_cast<int>(WorkshopColor::primariesCount()));
  }

  bool currentChromaticityPlanckianLocusEnabled(double time) {
    return getBoolValue("cubeViewerChromaticityPlanckianLocus", time, true);
  }

  std::string getStringValue(const std::string& name, double time, const std::string& fallback) const {
    if (auto* p = fetchStringParam(name)) {
      std::string value;
      p->getValueAtTime(time, value);
      return value;
    }
    return fallback;
  }

  struct ChromaspacePresetSelection {
    enum class Kind {
      Default,
      User,
      Custom
    };
    Kind kind = Kind::Default;
    int userIndex = -1;
    bool modified = false;
  };

  ChromaspacePresetValues captureCurrentChromaspacePresetValues(double time) const {
    ChromaspacePresetValues values = chromaspaceFactoryPresetValues();
    const bool drawOnImage = getBoolValue("cubeViewerDrawOnImageEnabled", time, false);
    values.plotModel = getChoiceValue("cubeViewerPlotModel", time, values.plotModel);
    values.plotInLinear = getBoolValue("cubeViewerPlotDisplayLinear", time, values.plotInLinear);
    values.inputTransferFunction = static_cast<int>(currentPlotDisplayLinearTransferId(time));
    values.showOverflow = getBoolValue("cubeViewerShowOverflow", time, values.showOverflow);
    values.highlightOverflow = getBoolValue("cubeViewerHighlightOverflow", time, values.highlightOverflow);
    values.fillVolume = drawOnImage
                            ? getBoolValue("cubeViewerIdentityOverlayEnabledDraw", time, values.fillVolume)
                            : getBoolValue("cubeViewerIdentityOverlayEnabled", time, values.fillVolume);
    values.fillResolution = getIntValue("cubeViewerIdentityOverlaySize", time, values.fillResolution);
    values.identityReadResolution = getIntValue("cubeViewerSampleDrawnCubeSize", time, values.identityReadResolution);
    values.volumeSliceLassoRegion = getBoolValue("cubeViewerLassoRegionMode", time, values.volumeSliceLassoRegion);
    values.volumeSliceRed = getBoolValue("cubeViewerSliceRed", time, values.volumeSliceRed);
    values.volumeSliceYellow = getBoolValue("cubeViewerSliceYellow", time, values.volumeSliceYellow);
    values.volumeSliceGreen = getBoolValue("cubeViewerSliceGreen", time, values.volumeSliceGreen);
    values.volumeSliceCyan = getBoolValue("cubeViewerSliceCyan", time, values.volumeSliceCyan);
    values.volumeSliceBlue = getBoolValue("cubeViewerSliceBlue", time, values.volumeSliceBlue);
    values.volumeSliceMagenta = getBoolValue("cubeViewerSliceMagenta", time, values.volumeSliceMagenta);
    values.neutralRadius = std::clamp(getDoubleValue("cubeViewerNeutralRadius", time, values.neutralRadius), 0.0, 1.0);
    values.readGrayRamp = !drawOnImage && getBoolValue("cubeViewerReadGrayRamp", time, values.readGrayRamp);
    values.readIdentityPlot = !drawOnImage && getBoolValue("cubeViewerSampleDrawnCubeOnly", time, values.readIdentityPlot);
    values.isolateIdentityData = !drawOnImage && getBoolValue("cubeViewerShowIdentityOnly", time, values.isolateIdentityData);
    values.liveUpdate = getBoolValue("cubeViewerLive", time, values.liveUpdate);
    values.keepOnTop = getBoolValue("cubeViewerOnTop", time, values.keepOnTop);
    values.updateMode = getChoiceValue("cubeViewerUpdateMode", time, values.updateMode);
    values.quality = getChoiceValue("cubeViewerQuality", time, values.quality);
    values.scale = getChoiceValue("cubeViewerScale", time, values.scale);
    values.plotStyle = getChoiceValue("cubeViewerPlotStyle", time, values.plotStyle);
    values.pointSize = getDoubleValue("cubeViewerPointSize", time, values.pointSize);
    values.colorSaturation = getDoubleValue("cubeViewerColorSaturation", time, values.colorSaturation);
    values.pointShape = getChoiceValue("cubeViewerPointShape", time, values.pointShape);
    values.sampling = getChoiceValue("cubeViewerSamplingMode", time, values.sampling);
    values.occupancyGuidedFill = getBoolValue("cubeViewerOccupancyGuidedFill", time, values.occupancyGuidedFill);
    return values;
  }

  void writeChromaspacePresetValuesToParams(const ChromaspacePresetValues& values) {
    if (auto* p = fetchChoiceParam("cubeViewerPlotModel")) p->setValue(values.plotModel);
    if (auto* p = fetchBooleanParam("cubeViewerPlotDisplayLinear")) p->setValue(values.plotInLinear);
    if (auto* p = fetchChoiceParam("cubeViewerPlotDisplayLinearTransfer")) {
      const auto transferId = static_cast<WorkshopColor::TransferFunctionId>(values.inputTransferFunction);
      p->setValue(plotLinearTransferChoiceIndex(transferId));
    }
    if (auto* p = fetchBooleanParam("cubeViewerShowOverflow")) p->setValue(values.showOverflow);
    if (auto* p = fetchBooleanParam("cubeViewerHighlightOverflow")) p->setValue(values.highlightOverflow);
    if (auto* p = fetchBooleanParam("cubeViewerIdentityOverlayEnabled")) p->setValue(values.fillVolume);
    if (auto* p = fetchBooleanParam("cubeViewerIdentityOverlayEnabledDraw")) p->setValue(values.fillVolume);
    if (auto* p = fetchIntParam("cubeViewerIdentityOverlaySize")) p->setValue(values.fillResolution);
    if (auto* p = fetchIntParam("cubeViewerSampleDrawnCubeSize")) p->setValue(values.identityReadResolution);
    if (auto* p = fetchBooleanParam("cubeViewerLassoRegionMode")) p->setValue(values.volumeSliceLassoRegion);
    if (auto* p = fetchBooleanParam("cubeViewerSliceRed")) p->setValue(values.volumeSliceRed);
    if (auto* p = fetchBooleanParam("cubeViewerSliceYellow")) p->setValue(values.volumeSliceYellow);
    if (auto* p = fetchBooleanParam("cubeViewerSliceGreen")) p->setValue(values.volumeSliceGreen);
    if (auto* p = fetchBooleanParam("cubeViewerSliceCyan")) p->setValue(values.volumeSliceCyan);
    if (auto* p = fetchBooleanParam("cubeViewerSliceBlue")) p->setValue(values.volumeSliceBlue);
    if (auto* p = fetchBooleanParam("cubeViewerSliceMagenta")) p->setValue(values.volumeSliceMagenta);
    if (auto* p = fetchDoubleParam("cubeViewerNeutralRadius")) p->setValue(values.neutralRadius);
    if (auto* p = fetchBooleanParam("cubeViewerReadGrayRamp")) p->setValue(values.readGrayRamp);
    if (auto* p = fetchBooleanParam("cubeViewerSampleDrawnCubeOnly")) p->setValue(values.readIdentityPlot);
    if (auto* p = fetchBooleanParam("cubeViewerShowIdentityOnly")) p->setValue(values.isolateIdentityData);
    if (auto* p = fetchBooleanParam("cubeViewerLive")) p->setValue(values.liveUpdate);
    if (auto* p = fetchBooleanParam("cubeViewerOnTop")) p->setValue(values.keepOnTop);
    if (auto* p = fetchChoiceParam("cubeViewerUpdateMode")) p->setValue(values.updateMode);
    if (auto* p = fetchChoiceParam("cubeViewerQuality")) p->setValue(values.quality);
    if (auto* p = fetchChoiceParam("cubeViewerScale")) p->setValue(values.scale);
    if (auto* p = fetchChoiceParam("cubeViewerPlotStyle")) p->setValue(values.plotStyle);
    if (auto* p = fetchDoubleParam("cubeViewerPointSize")) p->setValue(values.pointSize);
    if (auto* p = fetchDoubleParam("cubeViewerColorSaturation")) p->setValue(values.colorSaturation);
    if (auto* p = fetchChoiceParam("cubeViewerPointShape")) p->setValue(values.pointShape);
    if (auto* p = fetchChoiceParam("cubeViewerSamplingMode")) p->setValue(values.sampling);
    if (auto* p = fetchBooleanParam("cubeViewerOccupancyGuidedFill")) p->setValue(values.occupancyGuidedFill);
  }

  ChromaspacePresetSelection matchingChromaspacePresetSelection(double time) const {
    const ChromaspacePresetValues current = captureCurrentChromaspacePresetValues(time);
    std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
    ensureChromaspacePresetStoreLoadedLocked();
    if (chromaspacePresetValuesEqual(current, chromaspacePresetStore().defaultPreset.values)) {
      return {};
    }
    for (int i = 0; i < static_cast<int>(chromaspacePresetStore().userPresets.size()); ++i) {
      if (chromaspacePresetValuesEqual(current, chromaspacePresetStore().userPresets[static_cast<std::size_t>(i)].values)) {
        ChromaspacePresetSelection selection{};
        selection.kind = ChromaspacePresetSelection::Kind::User;
        selection.userIndex = i;
        return selection;
      }
    }
    ChromaspacePresetSelection selection{};
    selection.kind = ChromaspacePresetSelection::Kind::Custom;
    return selection;
  }

  ChromaspacePresetSelection resolvedChromaspacePresetSelection(double time) const {
    const ChromaspacePresetSelection exact = matchingChromaspacePresetSelection(time);
    if (exact.kind != ChromaspacePresetSelection::Kind::Custom) return exact;

    const ChromaspacePresetSelection currentMenuSelection = selectedChromaspacePresetFromMenu(time);
    if (currentMenuSelection.kind == ChromaspacePresetSelection::Kind::User) {
      ChromaspacePresetSelection modified = currentMenuSelection;
      modified.modified = true;
      return modified;
    }
    return exact;
  }

  ChromaspacePresetSelection selectedChromaspacePresetFromMenu(double time) const {
    ChromaspacePresetSelection selection{};
    const int selectedIndex = getChoiceValue("chromaspacePresetMenu", time, 0);
    if (selectedIndex == 0) return selection;
    if (chromaspacePresetMenuHasCustom_ && selectedIndex == chromaspacePresetCustomIndex_) {
      selection.kind = ChromaspacePresetSelection::Kind::Custom;
      return selection;
    }
    const int userCount = chromaspacePresetMenuUserCount_;
    if (selectedIndex >= 1 && selectedIndex <= userCount) {
      selection.kind = ChromaspacePresetSelection::Kind::User;
      selection.userIndex = selectedIndex - 1;
      return selection;
    }
    selection.kind = ChromaspacePresetSelection::Kind::Custom;
    return selection;
  }

  void rebuildChromaspacePresetMenu(double time, const ChromaspacePresetSelection& selection) {
    auto* param = fetchChoiceParam("chromaspacePresetMenu");
    if (!param) return;

    param->resetOptions();
    param->appendOption(kChromaspacePresetDefaultName);

    std::vector<std::string> names;
    {
      std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
      ensureChromaspacePresetStoreLoadedLocked();
      names.reserve(chromaspacePresetStore().userPresets.size());
      for (const auto& preset : chromaspacePresetStore().userPresets) names.push_back(preset.name);
    }
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
      if (selection.kind == ChromaspacePresetSelection::Kind::User &&
          selection.modified &&
          selection.userIndex == i) {
        param->appendOption(names[static_cast<std::size_t>(i)] + " (Modified)");
      } else {
        param->appendOption(names[static_cast<std::size_t>(i)]);
      }
    }

    chromaspacePresetMenuUserCount_ = static_cast<int>(names.size());
    chromaspacePresetMenuHasCustom_ = (selection.kind == ChromaspacePresetSelection::Kind::Custom);
    chromaspacePresetCustomIndex_ = -1;
    int selectedIndex = 0;
    if (selection.kind == ChromaspacePresetSelection::Kind::User &&
        selection.userIndex >= 0 &&
        selection.userIndex < chromaspacePresetMenuUserCount_) {
      selectedIndex = selection.userIndex + 1;
    } else if (selection.kind == ChromaspacePresetSelection::Kind::Custom) {
      param->appendOption(kChromaspacePresetCustomLabel);
      chromaspacePresetCustomIndex_ = chromaspacePresetMenuUserCount_ + 1;
      selectedIndex = chromaspacePresetCustomIndex_;
    }
    BoolScope scope(suppressChromaspacePresetChangedHandling_);
    param->setValue(selectedIndex);
  }

  void updateChromaspacePresetActionState(double time) {
    const ChromaspacePresetSelection selection = selectedChromaspacePresetFromMenu(time);
    const bool userSelected = selection.kind == ChromaspacePresetSelection::Kind::User;
    if (auto* p = fetchPushButtonParam("chromaspacePresetUpdate")) p->setEnabled(userSelected);
    if (auto* p = fetchPushButtonParam("chromaspacePresetRename")) p->setEnabled(userSelected);
    if (auto* p = fetchPushButtonParam("chromaspacePresetDelete")) p->setEnabled(userSelected);
  }

  void syncChromaspacePresetMenuState(double time) {
    rebuildChromaspacePresetMenu(time, resolvedChromaspacePresetSelection(time));
    updateChromaspacePresetActionState(time);
  }

  void syncChromaspacePresetMenuFromDisk(double time,
                                         const std::optional<ChromaspacePresetSelection>& preferred = std::nullopt) {
    {
      std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
      reloadChromaspacePresetStoreFromDiskLocked();
    }
    rebuildChromaspacePresetMenu(time, preferred ? *preferred : resolvedChromaspacePresetSelection(time));
    updateChromaspacePresetActionState(time);
  }

  void finalizeChromaspacePresetApplication(double time, const std::string& reason) {
    cubeViewerLive_ = getBoolValue("cubeViewerLive", time, true);
    cubeViewerUpdateMode_ = getChoiceValue("cubeViewerUpdateMode", time, 0);
    cubeViewerQuality_ = getChoiceValue("cubeViewerQuality", time, 0);
    resolveOverlaySizeParamIfAuto(time);
    updateDrawOnImageModeUi(time);
    syncIdentityOverlayGroupOpenState(time);
    updateCircularHslToggleVisibility(time);
    updateCircularHsvToggleVisibility(time);
    updateNormConeToggleVisibility(time);
    syncShowOverflowSupport(time);
    invalidateCubeViewerCloudState();
    syncChromaspacePresetMenuState(time);
    if (viewerSessionRequested()) {
      pushParamsUpdate(time, reason);
      (void)trySendCachedCloud(time, reason);
    }
  }

  void applyChromaspacePresetSelection(double time,
                                       const ChromaspacePresetSelection& selection,
                                       const char* reasonParam) {
    if (selection.kind == ChromaspacePresetSelection::Kind::Custom) return;

    ChromaspacePresetValues values = chromaspaceFactoryPresetValues();
    std::string presetName;
    {
      std::lock_guard<std::mutex> lock(chromaspacePresetMutex());
      ensureChromaspacePresetStoreLoadedLocked();
      if (selection.kind == ChromaspacePresetSelection::Kind::Default) {
        values = chromaspacePresetStore().defaultPreset.values;
        presetName = chromaspacePresetStore().defaultPreset.name;
      } else if (selection.userIndex >= 0 &&
                 selection.userIndex < static_cast<int>(chromaspacePresetStore().userPresets.size())) {
        values = chromaspacePresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)].values;
        presetName = chromaspacePresetStore().userPresets[static_cast<std::size_t>(selection.userIndex)].name;
      } else {
        return;
      }
    }

    BoolScope scope(suppressChromaspacePresetChangedHandling_);
    writeChromaspacePresetValuesToParams(values);
    finalizeChromaspacePresetApplication(time, reasonParam ? reasonParam : "chromaspacePresetMenu");
    syncChromaspacePresetMenuFromDisk(time, selection);
    if (selection.kind == ChromaspacePresetSelection::Kind::User) {
      if (auto* p = fetchStringParam("chromaspacePresetName")) p->setValue(presetName);
    }
  }

  void applySelectedChromaspacePreset(double time) {
    applyChromaspacePresetSelection(time, selectedChromaspacePresetFromMenu(time), "chromaspacePresetMenu");
  }

  bool isChromaspacePresetManagedParam(const std::string& paramName) const {
    return paramName == "cubeViewerPlotModel" ||
           paramName == "cubeViewerPlotDisplayLinear" ||
           paramName == "cubeViewerPlotDisplayLinearTransfer" ||
           paramName == "cubeViewerShowOverflow" ||
           paramName == "cubeViewerHighlightOverflow" ||
           paramName == "cubeViewerIdentityOverlayEnabled" ||
           paramName == "cubeViewerLassoRegionMode" ||
           paramName == "cubeViewerSliceRed" ||
           paramName == "cubeViewerSliceYellow" ||
           paramName == "cubeViewerSliceGreen" ||
           paramName == "cubeViewerSliceCyan" ||
           paramName == "cubeViewerSliceBlue" ||
           paramName == "cubeViewerSliceMagenta" ||
           paramName == "cubeViewerNeutralRadius" ||
           paramName == "cubeViewerIdentityOverlaySize" ||
           paramName == "cubeViewerReadGrayRamp" ||
           paramName == "cubeViewerSampleDrawnCubeOnly" ||
           paramName == "cubeViewerShowIdentityOnly" ||
           paramName == "cubeViewerSampleDrawnCubeSize" ||
           paramName == "cubeViewerLive" ||
           paramName == "cubeViewerOnTop" ||
           paramName == "cubeViewerUpdateMode" ||
           paramName == "cubeViewerQuality" ||
           paramName == "cubeViewerScale" ||
           paramName == "cubeViewerPlotStyle" ||
           paramName == "cubeViewerPointSize" ||
           paramName == "cubeViewerColorSaturation" ||
           paramName == "cubeViewerPointShape" ||
           paramName == "cubeViewerSamplingMode" ||
           paramName == "cubeViewerOccupancyGuidedFill";
  }

  LassoRegionState currentLassoRegionState(double time) {
    return parseLassoRegionState(getStringValue("cubeViewerLassoData", time, ""));
  }

  OfxRectD currentLassoImageRect(double time) {
    if (srcClip_) {
      const OfxRectD rod = srcClip_->getRegionOfDefinition(time);
      if (rod.x2 > rod.x1 && rod.y2 > rod.y1) return rod;
    }
    return OfxRectD{0.0, 0.0, 1.0, 1.0};
  }

  bool currentLassoSubtractOperation(double time) {
    return getChoiceValue("cubeViewerLassoOperation", time, 0) == 1;
  }

  std::string currentVolumeSlicingModeLabel(double time) {
    return currentVolumeSlicingMode(time) == VolumeSlicingMode::LassoRegion ? "lasso" : "hue";
  }

  void applyLassoRegionState(double time, const LassoRegionState& state, const std::string& reason) {
    const std::string serialized = serializeLassoRegionState(state);
    if (getStringValue("cubeViewerLassoData", time, "") == serialized) return;
    if (auto* p = fetchStringParam("cubeViewerLassoData")) {
      suppressLassoDataChangedHandling_ = true;
      p->setValue(serialized);
      suppressLassoDataChangedHandling_ = false;
    }
    cubeViewerDebugLog(std::string("Applied lasso region state: reason=") + reason +
                       " revision=" + std::to_string(state.revision) +
                       " strokes=" + std::to_string(state.strokes.size()));
    syncCubeSlicingUi(time);
    if (viewerSessionRequested()) {
      pushParamsUpdate(time, reason);
    }
    redrawOverlays();
  }

  void undoLassoRegionStroke(double time, const std::string& reason) {
    LassoRegionState state = currentLassoRegionState(time);
    if (state.strokes.empty()) return;
    state.strokes.pop_back();
    ++state.revision;
    applyLassoRegionState(time, state, reason);
  }

  void resetLassoRegion(double time, const std::string& reason) {
    LassoRegionState state{};
    state.revision = currentLassoRegionState(time).revision + 1;
    applyLassoRegionState(time, state, reason);
  }

  bool commitLassoStroke(double time, const std::vector<OfxPointD>& points, bool subtract, const std::string& reason) {
    if (!currentLassoRegionSlicingEnabled(time) || points.size() < 3) return false;
    const OfxRectD rect = currentLassoImageRect(time);
    const double width = std::max(1e-6, rect.x2 - rect.x1);
    const double height = std::max(1e-6, rect.y2 - rect.y1);
    LassoStroke stroke{};
    stroke.subtract = subtract;
    stroke.points.reserve(points.size());
    for (const auto& point : points) {
      const float xNorm = std::clamp(static_cast<float>((point.x - rect.x1) / width), 0.0f, 1.0f);
      const float yNorm = std::clamp(static_cast<float>((point.y - rect.y1) / height), 0.0f, 1.0f);
      if (!stroke.points.empty()) {
        const auto& last = stroke.points.back();
        if (std::fabs(last.xNorm - xNorm) < 1e-5f && std::fabs(last.yNorm - yNorm) < 1e-5f) continue;
      }
      stroke.points.push_back({xNorm, yNorm});
    }
    if (stroke.points.size() < 3) return false;
    LassoRegionState state = currentLassoRegionState(time);
    ++state.revision;
    state.strokes.push_back(std::move(stroke));
    applyLassoRegionState(time, state, reason);
    return true;
  }

  void syncLassoDataChange(double time, const std::string& reason) {
    const LassoRegionState state = currentLassoRegionState(time);
    cubeViewerDebugLog(std::string("Lasso region changed: reason=") + reason +
                       " revision=" + std::to_string(state.revision) +
                       " strokes=" + std::to_string(state.strokes.size()));
    syncCubeSlicingUi(time);
    if (viewerSessionRequested()) {
      pushParamsUpdate(time, reason);
    }
    redrawOverlays();
  }

  void syncCubeSlicingUi(double time) {
    const bool supported = currentCubeSlicingSupported(time);
    const bool hueSectorAllowed = currentHueSectorSlicingAllowed(time);
    const bool neutralRadiusVisible = currentNeutralRadiusSlicingAllowed(time);
    const bool lassoSelected = supported && getBoolValue("cubeViewerLassoRegionMode", time, false);
    const bool lassoMode = supported && currentVolumeSlicingMode(time) == VolumeSlicingMode::LassoRegion;
    const bool neutralRadiusEnabled = neutralRadiusVisible &&
                                      !currentShowOverflow(time);
    const bool hueOptionsVisible = supported && hueSectorAllowed && !lassoSelected;
    setParamVisibility(fetchGroupParam("grp_cube_viewer_slicing"), supported);
    setParamVisibility(fetchBooleanParam("cubeViewerLassoRegionMode"), supported);
    if (auto* p = fetchDoubleParam("cubeViewerNeutralRadius")) {
      p->setIsSecret(!neutralRadiusVisible);
      p->setEnabled(neutralRadiusEnabled);
    }
    setParamVisibility(fetchBooleanParam("cubeViewerSliceRed"), hueOptionsVisible);
    setParamVisibility(fetchBooleanParam("cubeViewerSliceGreen"), hueOptionsVisible);
    setParamVisibility(fetchBooleanParam("cubeViewerSliceBlue"), hueOptionsVisible);
    setParamVisibility(fetchBooleanParam("cubeViewerSliceCyan"), hueOptionsVisible);
    setParamVisibility(fetchBooleanParam("cubeViewerSliceYellow"), hueOptionsVisible);
    setParamVisibility(fetchBooleanParam("cubeViewerSliceMagenta"), hueOptionsVisible);
    setParamVisibility(fetchChoiceParam("cubeViewerLassoOperation"), lassoMode);
    setParamVisibility(fetchPushButtonParam("cubeViewerLassoUndo"), lassoMode);
    setParamVisibility(fetchPushButtonParam("cubeViewerLassoReset"), lassoMode);
  }

  void syncShowOverflowSupport(double time) {
    const bool drawOnImage = currentDrawOnImageMode(time);
    const std::string plotMode = currentPlotMode(time);
    const bool supported = !drawOnImage &&
                            (plotMode == "rgb" || plotMode == "hsl" ||
                             plotMode == "hsv" || plotMode == "chen" ||
                             plotMode == "jp_conical" || plotMode == "reuleaux" ||
                             plotMode == "chromaticity");
    const bool showOverflow = supported && getBoolValue("cubeViewerShowOverflow", time, false);
    if (auto* p = fetchBooleanParam("cubeViewerShowOverflow")) {
      if (!supported) {
        bool current = false;
        p->getValueAtTime(time, current);
        if (current) p->setValue(false);
      }
      p->setIsSecret(drawOnImage);
      p->setEnabled(supported);
    }
    if (auto* p = fetchBooleanParam("cubeViewerHighlightOverflow")) {
      p->setIsSecret(!showOverflow);
      p->setEnabled(showOverflow);
    }
    if (auto* p = fetchRGBParam("cubeViewerOverflowHighlightColor")) {
      p->setIsSecret(true);
      p->setEnabled(false);
    }
    if (auto* p = fetchRGBParam("cubeViewerBackgroundColor")) {
      p->setIsSecret(true);
      p->setEnabled(false);
    }
  }

  void setParamVisibility(Param* param, bool visible) {
    if (!param) return;
    param->setIsSecret(!visible);
    param->setEnabled(visible);
  }

  void setGroupOpenState(const std::string& name, bool open) {
    auto* effectSuite = const_cast<OfxImageEffectSuiteV1*>(
        reinterpret_cast<const OfxImageEffectSuiteV1*>(fetchSuite(kOfxImageEffectSuite, 1, true)));
    auto* paramSuite = const_cast<OfxParameterSuiteV1*>(
        reinterpret_cast<const OfxParameterSuiteV1*>(fetchSuite(kOfxParameterSuite, 1, true)));
    auto* propSuite = const_cast<OfxPropertySuiteV1*>(
        reinterpret_cast<const OfxPropertySuiteV1*>(fetchSuite(kOfxPropertySuite, 1, true)));
    if (!effectSuite || !paramSuite || !propSuite) return;
    OfxParamSetHandle paramSetHandle = nullptr;
    if (effectSuite->getParamSet(getHandle(), &paramSetHandle) != kOfxStatOK || !paramSetHandle) return;
    OfxParamHandle paramHandle = nullptr;
    OfxPropertySetHandle propHandle = nullptr;
    if (paramSuite->paramGetHandle(paramSetHandle, name.c_str(), &paramHandle, &propHandle) != kOfxStatOK || !propHandle) return;
    propSuite->propSetInt(propHandle, kOfxParamPropGroupOpen, 0, open ? 1 : 0);
  }

  void syncIdentityOverlayGroupOpenState(double time) {
    // Resolve can re-render repeatedly while a slider is being dragged.
    // Only force this group open for workflows that must keep it visible,
    // and avoid forcing it closed during normal interactive edits.
    if (currentDrawOnImageMode(time) || currentUseInstance1(time)) {
      setGroupOpenState("grp_cube_viewer_identity_overlay", true);
    }
  }

  void syncIdentityReadbackUi(double time) {
    const bool drawOnImage = currentDrawOnImageMode(time);
    const bool readIdentityPlot = currentReadIdentityPlot(time);
    const bool readGrayRamp = currentReadGrayRamp(time);
    const bool useInstance1 = currentUseInstance1(time);
    setParamVisibility(fetchIntParam("cubeViewerSampleDrawnCubeSize"),
                       drawOnImage ? currentIdentityOverlayEnabled(time) : (readIdentityPlot || readGrayRamp));
    setParamVisibility(fetchBooleanParam("cubeViewerShowIdentityOnly"),
                       !drawOnImage && useInstance1);
    syncIdentityOverlayGroupOpenState(time);
  }

  void updateDrawOnImageModeUi(double time) {
    const bool drawOnImage = currentDrawOnImageMode(time);
    const bool overlayEnabled = currentIdentityOverlayEnabled(time);
    const bool useInstance1Requested = currentUseInstance1Requested(time);
    const bool useInstance1 = currentUseInstance1(time);
    const bool readIdentityPlot = currentReadIdentityPlot(time);
    const bool readGrayRamp = currentReadGrayRamp(time);
    const bool chromaticityMode = currentChromaticityPlotMode(time);
    if (drawOnImage || useInstance1Requested) {
      setGroupOpenState("grp_cube_viewer_identity_overlay", true);
    }
    if (auto* drawCube = fetchBooleanParam("cubeViewerIdentityOverlayEnabled")) {
      drawCube->setLabel("Fill Volume");
    }
    if (auto* drawCubeProxy = fetchBooleanParam("cubeViewerIdentityOverlayEnabledDraw")) {
      drawCubeProxy->setLabel("Generate Identity Plot");
    }
    if (auto* overlaySize = fetchIntParam("cubeViewerIdentityOverlaySize")) {
      overlaySize->setLabel("Fill Resolution");
    }
    if (auto* grayRamp = fetchBooleanParam("cubeViewerIdentityOverlayRamp")) {
      grayRamp->setLabel("Overlay Gray Ramp");
    }
    if (auto* grayRampProxy = fetchBooleanParam("cubeViewerIdentityOverlayRampDraw")) {
      grayRampProxy->setLabel("Draw Gray Ramp");
    }
    if (auto* note = fetchStringParam("cubeViewerIdentityOverlayNote")) {
      note->setIsSecret(drawOnImage);
      note->setEnabled(false);
    }
    if (auto* toggle = fetchPushButtonParam("cubeViewerModeToggle")) {
      toggle->setLabel(drawOnImage ? "Switch to 3D Viewer" : "Switch to Identity Generator");
    }
    setParamVisibility(fetchPushButtonParam("openCubeViewer"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerPlotDisplayLinear"),
                       !drawOnImage && !chromaticityMode);
    setParamVisibility(fetchChoiceParam("cubeViewerPlotDisplayLinearTransfer"),
                       !drawOnImage && !chromaticityMode && currentPlotDisplayLinearEnabled(time));
    setParamVisibility(fetchChoiceParam("cubeViewerPlotModel"), !drawOnImage);
    syncShowOverflowSupport(time);
    syncCubeSlicingUi(time);
    setParamVisibility(fetchPushButtonParam("closeCubeViewer"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerLive"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerOnTop"), !drawOnImage);
    setParamVisibility(fetchChoiceParam("cubeViewerQuality"), !drawOnImage);
    setParamVisibility(fetchChoiceParam("cubeViewerScale"), !drawOnImage);
    setParamVisibility(fetchDoubleParam("cubeViewerPointSize"), !drawOnImage);
    setParamVisibility(fetchDoubleParam("cubeViewerColorSaturation"), !drawOnImage);
    setParamVisibility(fetchChoiceParam("cubeViewerPointShape"), !drawOnImage);
    setParamVisibility(fetchChoiceParam("cubeViewerSamplingMode"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerOccupancyGuidedFill"), !drawOnImage);
    if (auto* status = fetchStringParam("cubeViewerStatus")) {
      status->setIsSecret(drawOnImage);
      status->setEnabled(false);
    }
    setParamVisibility(fetchBooleanParam("cubeViewerCircularHsl"),
                       !drawOnImage && currentPlotMode(time) == "hsl");
    setParamVisibility(fetchBooleanParam("cubeViewerCircularHsv"),
                       !drawOnImage && currentPlotMode(time) == "hsv");
    setParamVisibility(fetchBooleanParam("cubeViewerNormConeNormalized"),
                       !drawOnImage && currentPlotMode(time) == "norm_cone");
    setParamVisibility(fetchGroupParam("grp_cube_viewer_chromaticity_cm"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchBooleanParam("cubeViewerChromaticityPlanckianLocus"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchChoiceParam("cubeViewerChromaticityInputPrimaries"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchChoiceParam("cubeViewerChromaticityInputTransfer"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchChoiceParam("cubeViewerChromaticityReferenceBasis"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchChoiceParam("cubeViewerChromaticityOverlayPrimaries"),
                       !drawOnImage && chromaticityMode);
    setParamVisibility(fetchBooleanParam("cubeViewerIdentityOverlayEnabled"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerIdentityOverlayEnabledDraw"), drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerIdentityOverlayRampDraw"), drawOnImage && overlayEnabled);
    setParamVisibility(fetchIntParam("cubeViewerIdentityOverlaySize"), !drawOnImage && overlayEnabled);
    setParamVisibility(fetchBooleanParam("cubeViewerIdentityOverlayRamp"), !drawOnImage && overlayEnabled);
    setParamVisibility(fetchBooleanParam("cubeViewerSampleDrawnCubeOnly"), !drawOnImage);
    setParamVisibility(fetchBooleanParam("cubeViewerReadGrayRamp"), !drawOnImage);
    setParamVisibility(fetchIntParam("cubeViewerSampleDrawnCubeSize"),
                       drawOnImage ? overlayEnabled : (readIdentityPlot || readGrayRamp));
    setParamVisibility(fetchBooleanParam("cubeViewerShowIdentityOnly"), !drawOnImage && useInstance1);
    setParamVisibility(fetchGroupParam("grp_cube_viewer"), !drawOnImage);
    setParamVisibility(fetchGroupParam("grp_cube_viewer_identity_overlay"), true);
    setParamVisibility(fetchGroupParam("grp_support_root"), true);
    cubeViewerDebugLog(std::string("Draw-on-image UI state -> ") + (drawOnImage ? "draw" : "plot"));
  }

  void updateNormConeToggleVisibility(double time) {
    const bool visible = !currentDrawOnImageMode(time) && currentPlotMode(time) == "norm_cone";
    if (auto* p = fetchBooleanParam("cubeViewerNormConeNormalized")) {
      p->setIsSecret(!visible);
      p->setEnabled(visible);
    }
  }

  void updateCircularHslToggleVisibility(double time) {
    const bool visible = !currentDrawOnImageMode(time) && currentPlotMode(time) == "hsl";
    if (auto* p = fetchBooleanParam("cubeViewerCircularHsl")) {
      p->setIsSecret(!visible);
      p->setEnabled(visible);
    }
  }

  void updateCircularHsvToggleVisibility(double time) {
    const bool visible = !currentDrawOnImageMode(time) && currentPlotMode(time) == "hsv";
    if (auto* p = fetchBooleanParam("cubeViewerCircularHsv")) {
      p->setIsSecret(!visible);
      p->setEnabled(visible);
    }
  }

  void invalidateCubeViewerCloudState() {
    cubeViewerInputCloudRefreshPending_ = true;
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    playbackRenderBurstCount_ = 0;
    lastCloudTime_ = std::numeric_limits<double>::quiet_NaN();
    lastCloudBuiltAt_ = std::chrono::steady_clock::time_point{};
    lastCloudSourceId_.clear();
    lastCloudSettingsKey_.clear();
    {
      std::lock_guard<std::mutex> lock(stateMutex_);
      cachedCloud_ = CachedCloud{};
      cachedIdentityStripCloud_ = CachedIdentityStripCloud{};
    }
  }

  void requestCubeViewerCloudResample() {
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    playbackRenderBurstCount_ = 0;
    lastCloudTime_ = std::numeric_limits<double>::quiet_NaN();
    lastCloudBuiltAt_ = std::chrono::steady_clock::time_point{};
  }

  std::string buildIdentityStripParamHash(int width, int height, int resolution, bool readCube, bool readRamp, double time) {
    std::ostringstream hash;
    hash << width << 'x' << height << ':' << resolution << ':' << qualityLabelForIndex(getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_));
    if (readCube) hash << ":drawn-cube=1";
    if (readRamp) hash << ":drawn-ramp=1";
    return hash.str();
  }

  std::string currentIdentityStripCacheKey(
      double time,
      int width,
      int height,
      int originX,
      int originY,
      int resolution,
      bool readCube,
      bool readRamp,
      const std::string& backendTag) {
    std::ostringstream oss;
    oss << backendTag
        << "|width=" << width
        << "|height=" << height
        << "|origin=" << originX << "," << originY
        << "|resolution=" << resolution
        << "|quality=" << qualityLabelForIndex(getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_))
        << "|readCube=" << (readCube ? 1 : 0)
        << "|readRamp=" << (readRamp ? 1 : 0)
        << "|showOverflow=" << (currentShowOverflow(time) ? 1 : 0)
        << "|plotDisplayLinear=" << (currentPlotDisplayLinearEnabled(time) ? 1 : 0)
        << "|plotDisplayLinearTransfer=" << static_cast<int>(currentPlotDisplayLinearTransferId(time));
    return oss.str();
  }

  bool tryGetCachedIdentityStripCloud(
      const std::string& cacheKey,
      std::vector<ViewerCloudSample>* outSamples,
      std::string* outParamHash,
      int* outResolution) {
    if (!outSamples) return false;
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!cachedIdentityStripCloud_.valid || cachedIdentityStripCloud_.cacheKey != cacheKey) return false;
    *outSamples = cachedIdentityStripCloud_.samples;
    if (outParamHash) *outParamHash = cachedIdentityStripCloud_.paramHash;
    if (outResolution) *outResolution = cachedIdentityStripCloud_.resolution;
    return true;
  }

  void storeCachedIdentityStripCloud(
      const std::string& cacheKey,
      const std::vector<ViewerCloudSample>& samples,
      const std::string& paramHash,
      int resolution) {
    std::lock_guard<std::mutex> lock(stateMutex_);
    cachedIdentityStripCloud_.samples = samples;
    cachedIdentityStripCloud_.cacheKey = cacheKey;
    cachedIdentityStripCloud_.paramHash = paramHash;
    cachedIdentityStripCloud_.resolution = resolution;
    cachedIdentityStripCloud_.valid = true;
  }

  std::string currentQualityLabel(double time) {
    return qualityLabelForIndex(getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_));
  }

  // This key is the contract between the OFX instance and the external viewer. Any setting that changes
  // how the cloud should be interpreted must be represented here so stale clouds can be rejected safely.
  std::string currentCloudSettingsKey(double time) {
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int samplingMode = getSamplingModeValue(getChoiceValue("cubeViewerSamplingMode", time, 0));
    const int scaleIndex = getChoiceValue("cubeViewerScale", time, 3);
    const bool occupancyFill = currentOccupancyGuidedFill(time);
    const bool useInstance1 = currentUseInstance1(time);
    const bool readIdentityPlot = currentReadIdentityPlot(time);
    const bool readGrayRamp = currentReadGrayRamp(time);
    const bool showIdentityOnly = currentShowIdentityOnly(time);
    const int sampleDrawnCubeRequestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", time, 29);
    const int sampleDrawnCubeResolvedSize = clampOverlayCubeSize(sampleDrawnCubeRequestedSize);
    const std::string plotMode = currentPlotMode(time);
    const bool showOverflow = currentShowOverflow(time);
    const bool highlightOverflow = showOverflow && getBoolValue("cubeViewerHighlightOverflow", time, true);
    const bool circularHsl = currentCircularHsl(time);
    const bool circularHsv = currentCircularHsv(time);
    const bool normConeNormalized = getBoolValue("cubeViewerNormConeNormalized", time, true);
    const bool plotDisplayLinear = currentPlotDisplayLinearEnabled(time);
    const int plotDisplayLinearTransfer = currentPlotDisplayLinearTransferChoice(time);
    const int chromaticityInputPrimaries = currentChromaticityInputPrimariesChoice(time);
    const int chromaticityInputTransfer = currentChromaticityInputTransferChoice(time);
    const int chromaticityReferenceBasis = currentChromaticityReferenceBasisChoice(time);
    const int chromaticityOverlayPrimaries = currentChromaticityOverlayPrimariesChoice(time);
    const auto overflowColor = getRGBValue("cubeViewerOverflowHighlightColor", time, {1.0, 0.0, 0.0});
    const auto backgroundColor = getRGBValue("cubeViewerBackgroundColor", time, {0.08, 0.08, 0.09});
    std::ostringstream oss;
    oss << "quality=" << qualityLabelForIndex(qualityIndex)
        << "|resolution=" << qualityResolutionForIndex(qualityIndex)
        << "|sampling=" << samplingModeLabelForIndex(samplingMode)
        << "|occupancyFill=" << (occupancyFill ? 1 : 0)
        << "|scale=" << scaleLabelForIndex(scaleIndex)
        << "|plotMode=" << plotMode
         << "|circularHsl=" << (circularHsl ? 1 : 0)
         << "|circularHsv=" << (circularHsv ? 1 : 0)
         << "|normConeNormalized=" << (normConeNormalized ? 1 : 0)
         << "|plotDisplayLinear=" << (plotDisplayLinear ? 1 : 0)
         << "|plotDisplayLinearTransfer=" << plotDisplayLinearTransfer
         << "|chromaticityInputPrimaries=" << chromaticityInputPrimaries
         << "|chromaticityInputTransfer=" << chromaticityInputTransfer
         << "|chromaticityReferenceBasis=" << chromaticityReferenceBasis
         << "|chromaticityOverlayPrimaries=" << chromaticityOverlayPrimaries
         << "|showOverflow=" << (showOverflow ? 1 : 0)
         << "|highlightOverflow=" << (highlightOverflow ? 1 : 0)
         << "|overflowColor=" << overflowColor[0] << "," << overflowColor[1] << "," << overflowColor[2]
        << "|backgroundColor=" << backgroundColor[0] << "," << backgroundColor[1] << "," << backgroundColor[2]
        << "|drawMode=" << (currentDrawOnImageMode(time) ? 1 : 0)
        << "|useInstance1=" << (useInstance1 ? 1 : 0)
         << "|showIdentityOnly=" << (showIdentityOnly ? 1 : 0)
         << "|readIdentityPlot=" << (readIdentityPlot ? 1 : 0)
         << "|readGrayRamp=" << (readGrayRamp ? 1 : 0)
         << "|sampleDrawnCubeSize=" << sampleDrawnCubeResolvedSize;
    return oss.str();
  }

  std::string currentSourceIdentifier(Image* img) {
    if (img) {
      const std::string& uniqueId = img->getUniqueIdentifier();
      if (!uniqueId.empty()) return uniqueId;
    }
    // Resolve can leave image unique IDs empty in some clip/group/timeline render contexts.
    // Fall back to the OFX instance sender so shared-viewer ownership can still hand off.
    return std::string("sender:") + senderId_;
  }

  // Params payloads describe how the viewer should interpret any subsequent cloud payloads and overlays.
  // The viewer applies params first, then only accepts clouds whose settings key matches this snapshot.
  std::string buildParamsPayload(double time) {
    const uint64_t seq = gSharedCubeViewerSeqCounter.fetch_add(1, std::memory_order_relaxed);
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int samplingMode = getSamplingModeValue(getChoiceValue("cubeViewerSamplingMode", time, 0));
    const int scaleIndex = getChoiceValue("cubeViewerScale", time, 3);
    const bool occupancyFill = currentOccupancyGuidedFill(time);
    const int pointShape = getPointShapeValue(getChoiceValue("cubeViewerPointShape", time, 0));
    const int plotStyle = getPlotStyleValue(getChoiceValue("cubeViewerPlotStyle", time, 0));
    const int resolution = qualityResolutionForIndex(qualityIndex);
    const bool onTop = getBoolValue("cubeViewerOnTop", time, true);
    const double pointSize = getDoubleValue("cubeViewerPointSize", time, 1.4);
    const double colorSaturation = getDoubleValue("cubeViewerColorSaturation", time, 2.0);
    const double pointDensity = derivedDensityScaleForPointSize(pointSize);
    const std::string sourceMode = currentSourceMode(time);
    const std::string plotMode = currentPlotMode(time);
    const bool circularHsl = currentCircularHsl(time);
    const bool circularHsv = currentCircularHsv(time);
    const bool normConeNormalized = getBoolValue("cubeViewerNormConeNormalized", time, true);
    const bool plotDisplayLinear = currentPlotDisplayLinearEnabled(time);
    const int plotDisplayLinearTransfer = currentPlotDisplayLinearTransferChoice(time);
    const int chromaticityInputPrimaries = currentChromaticityInputPrimariesChoice(time);
    const int chromaticityInputTransfer = currentChromaticityInputTransferChoice(time);
    const int chromaticityReferenceBasis = currentChromaticityReferenceBasisChoice(time);
    const int chromaticityOverlayPrimaries = currentChromaticityOverlayPrimariesChoice(time);
    const bool chromaticityPlanckianLocus = currentChromaticityPlanckianLocusEnabled(time);
    const bool overlayEnabled = currentIdentityOverlayEnabled(time);
    const bool overlayRamp = currentIdentityOverlayRamp(time);
    const bool showOverflow = currentShowOverflow(time);
    const bool highlightOverflow = showOverflow && getBoolValue("cubeViewerHighlightOverflow", time, true);
    const bool volumeSlicingEnabled = currentVolumeSlicingEnabled(time);
    const bool cubeSlicingEnabled = currentHueSectorSlicingEnabled(time);
    const std::string slicingMode = currentVolumeSlicingModeLabel(time);
    const bool lassoRegionEmpty = currentLassoRegionState(time).empty();
    const std::string lassoData = getStringValue("cubeViewerLassoData", time, "");
    const auto overflowColor = getRGBValue("cubeViewerOverflowHighlightColor", time, {1.0, 0.0, 0.0});
    const auto backgroundColor = getRGBValue("cubeViewerBackgroundColor", time, {0.08, 0.08, 0.09});
    const bool drawOnImageMode = currentDrawOnImageMode(time);
    const int overlayRequestedSize = drawOnImageMode
                                         ? getIntValue("cubeViewerSampleDrawnCubeSize", time, 29)
                                         : getIntValue("cubeViewerIdentityOverlaySize", time, 29);
    const int overlayResolvedSize = drawOnImageMode
                                        ? clampOverlayCubeSize(overlayRequestedSize)
                                        : resolvedOverlayCubeSize(overlayRequestedSize, qualityIndex, scaleIndex);
    const bool overlayAuto = !drawOnImageMode && clampOverlayCubeSize(overlayRequestedSize) == 25;
    const std::string cloudSettingsKey = currentCloudSettingsKey(time);
    std::ostringstream oss;
    oss << "{\"type\":\"params\",\"seq\":" << seq
        << ",\"senderId\":\"" << jsonEscape(senderId_) << "\""
        << ",\"sourceMode\":\"" << sourceMode << "\""
        << ",\"plotMode\":\"" << plotMode << "\""
        << ",\"circularHsl\":" << (circularHsl ? 1 : 0)
        << ",\"circularHsv\":" << (circularHsv ? 1 : 0)
        << ",\"cloudSettingsKey\":\"" << jsonEscape(cloudSettingsKey) << "\""
        << ",\"normConeNormalized\":" << (normConeNormalized ? 1 : 0)
        << ",\"plotDisplayLinear\":" << (plotDisplayLinear ? 1 : 0)
        << ",\"plotDisplayLinearTransfer\":" << plotDisplayLinearTransfer
        << ",\"chromaticityInputPrimaries\":" << chromaticityInputPrimaries
        << ",\"chromaticityInputTransfer\":" << chromaticityInputTransfer
        << ",\"chromaticityReferenceBasis\":" << chromaticityReferenceBasis
        << ",\"chromaticityOverlayPrimaries\":" << chromaticityOverlayPrimaries
        << ",\"chromaticityPlanckianLocus\":" << (chromaticityPlanckianLocus ? 1 : 0)
        << ",\"alwaysOnTop\":" << (onTop ? 1 : 0)
        << ",\"quality\":\"" << qualityLabelForIndex(qualityIndex) << "\""
        << ",\"sampling\":\"" << samplingModeLabelForIndex(samplingMode) << "\""
        << ",\"occupancyFill\":" << (occupancyFill ? 1 : 0)
        << ",\"scale\":\"" << scaleLabelForIndex(scaleIndex) << "\""
        << ",\"plotStyle\":\"" << plotStyleLabelForIndex(plotStyle) << "\""
        << ",\"pointShape\":\"" << pointShapeLabelForIndex(pointShape) << "\""
        << ",\"resolution\":" << resolution
        << ",\"pointSize\":" << pointSize
        << ",\"colorSaturation\":" << colorSaturation
        << ",\"pointDensity\":" << pointDensity
        << ",\"showOverflow\":" << (showOverflow ? 1 : 0)
        << ",\"highlightOverflow\":" << (highlightOverflow ? 1 : 0)
        << ",\"volumeSlicingEnabled\":" << (volumeSlicingEnabled ? 1 : 0)
        << ",\"volumeSlicingMode\":\"" << slicingMode << "\""
        << ",\"lassoRegionEmpty\":" << (lassoRegionEmpty ? 1 : 0)
        << ",\"lassoData\":\"" << jsonEscape(lassoData) << "\""
        << ",\"cubeSlicingEnabled\":" << (cubeSlicingEnabled ? 1 : 0)
        << ",\"neutralRadiusEnabled\":" << (currentNeutralRadiusSlicingEnabled(time) ? 1 : 0)
        << ",\"neutralRadius\":" << currentNeutralRadiusValue(time)
        << ",\"cubeSliceRed\":" << (getBoolValue("cubeViewerSliceRed", time, true) ? 1 : 0)
        << ",\"cubeSliceGreen\":" << (getBoolValue("cubeViewerSliceGreen", time, false) ? 1 : 0)
        << ",\"cubeSliceBlue\":" << (getBoolValue("cubeViewerSliceBlue", time, false) ? 1 : 0)
        << ",\"cubeSliceCyan\":" << (getBoolValue("cubeViewerSliceCyan", time, false) ? 1 : 0)
        << ",\"cubeSliceYellow\":" << (getBoolValue("cubeViewerSliceYellow", time, false) ? 1 : 0)
        << ",\"cubeSliceMagenta\":" << (getBoolValue("cubeViewerSliceMagenta", time, false) ? 1 : 0)
        << ",\"overflowHighlightColorR\":" << overflowColor[0]
        << ",\"overflowHighlightColorG\":" << overflowColor[1]
        << ",\"overflowHighlightColorB\":" << overflowColor[2]
        << ",\"viewerBackgroundColorR\":" << backgroundColor[0]
        << ",\"viewerBackgroundColorG\":" << backgroundColor[1]
        << ",\"viewerBackgroundColorB\":" << backgroundColor[2]
        << ",\"identityOverlayEnabled\":" << (overlayEnabled ? 1 : 0)
        << ",\"identityOverlayRamp\":" << (overlayRamp ? 1 : 0)
        << ",\"identityOverlayAuto\":" << (overlayAuto ? 1 : 0)
        << ",\"identityOverlayRequestedSize\":" << clampOverlayCubeSize(overlayRequestedSize)
        << ",\"identityOverlaySize\":" << overlayResolvedSize
        << ",\"version\":\"" << kPluginVersionLabel << "\"}\n";
    cubeViewerDebugLog(std::string("Built params payload: sourceMode=") + sourceMode +
                        " plotMode=" + plotMode +
                        " circularHsl=" + (circularHsl ? "1" : "0") +
                        " circularHsv=" + (circularHsv ? "1" : "0") +
                        " normConeNormalized=" + (normConeNormalized ? "1" : "0") +
                         " plotDisplayLinear=" + (plotDisplayLinear ? "1" : "0") +
                         " plotDisplayLinearTransfer=" + std::to_string(plotDisplayLinearTransfer) +
                         " quality=" + qualityLabelForIndex(qualityIndex) +
                         " sampling=" + samplingModeLabelForIndex(samplingMode) +
                         " occupancyFill=" + (occupancyFill ? "1" : "0") +
                         " scale=" + scaleLabelForIndex(scaleIndex) +
                         " pointShape=" + pointShapeLabelForIndex(pointShape) +
                         " res=" + std::to_string(resolution) +
                         " pointSize=" + std::to_string(pointSize) +
                         " colorSaturation=" + std::to_string(colorSaturation) +
                         " pointDensity=" + std::to_string(pointDensity) +
                           " showOverflow=" + (showOverflow ? "1" : "0") +
                           " highlightOverflow=" + (highlightOverflow ? "1" : "0") +
                          " volumeSlicing=" + (volumeSlicingEnabled ? "1" : "0") +
                          " slicingMode=" + slicingMode +
                          " lassoEmpty=" + (lassoRegionEmpty ? "1" : "0") +
                          " cloudKey=" + cloudSettingsKey +
                          " overlay=" + (overlayEnabled ? "1" : "0") +
                         " overlayRamp=" + (overlayRamp ? "1" : "0") +
                        " overlayReq=" + std::to_string(clampOverlayCubeSize(overlayRequestedSize)) +
                        " overlayRes=" + std::to_string(overlayResolvedSize) +
                        " onTop=" + (onTop ? "1" : "0"));
    return oss.str();
  }

  bool shouldUseFastCloudTransport(std::size_t sampleCount) const {
    // Prefer binary/shared-memory handoff whenever there is actual cloud data.
    // JSON remains the fallback if mapping creation fails.
    return sampleCount > 0u;
  }

  std::string buildCloudTransportName(uint64_t seq) const {
    const uint64_t senderHash = fnv1a64(senderId_);
#if defined(_WIN32)
    std::ostringstream oss;
    oss << "ChromaspaceCloud_" << std::hex << std::nouppercase
        << senderHash << "_" << seq << "_" << static_cast<unsigned long>(GetCurrentProcessId());
    return oss.str();
#else
    std::ostringstream oss;
    oss << "/chromaspace_cloud_" << std::hex << std::nouppercase
        << senderHash << "_" << seq << "_" << static_cast<long>(::getpid());
    return oss.str();
#endif
  }

  std::shared_ptr<ViewerCloudTransportBlob> createCloudTransportBlob(
      const std::vector<ViewerCloudSample>& samples,
      uint64_t seq) {
    if (samples.empty()) return {};
    const std::size_t byteSize = samples.size() * sizeof(ViewerCloudSample);
    const std::string baseName = buildCloudTransportName(seq);
    auto blob = std::make_shared<ViewerCloudTransportBlob>();
    blob->byteSize = byteSize;
#if defined(_WIN32)
    const HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE,
                                              nullptr,
                                              PAGE_READWRITE,
                                              static_cast<DWORD>((byteSize >> 32) & 0xffffffffu),
                                              static_cast<DWORD>(byteSize & 0xffffffffu),
                                              baseName.c_str());
    if (mapping == nullptr) {
      return {};
    }
    void* mapped = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, byteSize);
    if (!mapped) {
      CloseHandle(mapping);
      return {};
    }
    std::memcpy(mapped, samples.data(), byteSize);
    UnmapViewOfFile(mapped);
    blob->name = baseName;
    blob->mappingHandle = mapping;
#else
    int fd = -1;
    std::string chosenName;
    for (int attempt = 0; attempt < 8; ++attempt) {
      std::ostringstream nameBuilder;
      nameBuilder << baseName;
      if (attempt > 0) nameBuilder << "_" << attempt;
      chosenName = nameBuilder.str();
      fd = shm_open(chosenName.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
      if (fd >= 0) break;
      if (errno != EEXIST) return {};
    }
    if (fd < 0) return {};
    if (ftruncate(fd, static_cast<off_t>(byteSize)) != 0) {
      ::close(fd);
      shm_unlink(chosenName.c_str());
      return {};
    }
    void* mapped = mmap(nullptr, byteSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
      ::close(fd);
      shm_unlink(chosenName.c_str());
      return {};
    }
    std::memcpy(mapped, samples.data(), byteSize);
    munmap(mapped, byteSize);
    blob->name = chosenName;
    blob->fd = fd;
#endif
    return blob;
  }

  std::string serializeViewerCloudSamples(const std::vector<ViewerCloudSample>& samples) const {
    std::string points;
    points.reserve(samples.size() * 52u);
    bool first = true;
    for (const auto& sample : samples) {
      appendCloudPointSample(&points, &first, sample.xNorm, sample.yNorm, sample.r, sample.g, sample.b);
    }
    return points;
  }

  std::string buildInputCloudJson(
      const std::string& points,
      const std::string& paramHash,
      const std::string& settingsKey,
      int resolution,
      int qualityIndex) {
    const uint64_t seq = gSharedCubeViewerSeqCounter.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream oss;
    oss << "{\"type\":\"input_cloud\",\"seq\":" << seq
        << ",\"senderId\":\"" << jsonEscape(senderId_) << "\""
        << ",\"quality\":\"" << qualityLabelForIndex(qualityIndex) << "\""
        << ",\"resolution\":" << resolution
        << ",\"paramHash\":\"" << jsonEscape(paramHash) << "\""
        << ",\"settingsKey\":\"" << jsonEscape(settingsKey) << "\""
        << ",\"transport\":\"json\""
        << ",\"points\":\"" << jsonEscape(points) << "\"}\n";
    return oss.str();
  }

  std::string buildInputCloudJson(
      const std::vector<ViewerCloudSample>& samples,
      const std::string& paramHash,
      const std::string& settingsKey,
      int resolution,
      int qualityIndex,
      std::shared_ptr<ViewerCloudTransportBlob>* outBlob,
      std::string* outTransportMode) {
    const uint64_t seq = gSharedCubeViewerSeqCounter.fetch_add(1, std::memory_order_relaxed);
    if (outBlob) outBlob->reset();
    if (outTransportMode) *outTransportMode = "json";
    if (shouldUseFastCloudTransport(samples.size())) {
      std::shared_ptr<ViewerCloudTransportBlob> blob = createCloudTransportBlob(samples, seq);
      if (blob) {
        if (outBlob) *outBlob = blob;
        if (outTransportMode) *outTransportMode = "shm";
        std::ostringstream oss;
        oss << "{\"type\":\"input_cloud\",\"seq\":" << seq
            << ",\"senderId\":\"" << jsonEscape(senderId_) << "\""
            << ",\"quality\":\"" << qualityLabelForIndex(qualityIndex) << "\""
            << ",\"resolution\":" << resolution
            << ",\"paramHash\":\"" << jsonEscape(paramHash) << "\""
            << ",\"settingsKey\":\"" << jsonEscape(settingsKey) << "\""
            << ",\"transport\":\"shm\""
            << ",\"pointCount\":" << samples.size()
            << ",\"pointStride\":" << sizeof(ViewerCloudSample)
            << ",\"shmName\":\"" << jsonEscape(blob->name) << "\""
            << ",\"shmSize\":" << blob->byteSize
            << "}\n";
        return oss.str();
      }
    }
    const std::string points = serializeViewerCloudSamples(samples);
    return buildInputCloudJson(points, paramHash, settingsKey, resolution, qualityIndex);
  }

  float sampledChannelValue(float value, bool preserveOverflow) const {
    return preserveOverflow ? value : clamp01(value);
  }

  WorkshopColor::Vec3f transformCloudSampleForPlot(double time, float r, float g, float b) {
    WorkshopColor::Vec3f sample{r, g, b};
    if (!currentPlotDisplayLinearEnabled(time)) return sample;
    return WorkshopColor::decodeToLinear(sample, currentPlotDisplayLinearTransferId(time));
  }

  void appendCloudPointSample(std::string* pts, bool* first, float xNorm, float yNorm, float r, float g, float b) const {
    if (!pts || !first) return;
    if (!*first) pts->push_back(';');
    *first = false;
    char sample[96];
    const int n = std::snprintf(sample, sizeof(sample), "%.6f %.6f %.6f %.6f %.6f %.6f", xNorm, yNorm, 0.0f, r, g, b);
    if (n > 0) pts->append(sample, static_cast<size_t>(n));
  }

  struct ViewerCloudSamplesBuildResult {
    std::vector<ViewerCloudSample> samples;
    std::vector<ViewerCloudSample> identityStripSamples;
    std::string paramHash;
    std::string identityStripParamHash;
    std::string quality;
    std::string backendName = "CPU";
    int resolution = 25;
    int identityStripResolution = 25;
    int sourceWidth = 0;
    int sourceHeight = 0;
    int primaryAttempts = 0;
    int primaryAccepted = 0;
    bool success = false;
  };

  ViewerCloudBuildRequest makeViewerCloudBuildRequest(
      const float* srcBase,
      std::size_t srcRowBytes,
      int width,
      int height,
      double time,
      bool previewMode) {
    ViewerCloudBuildRequest request{};
    request.srcBase = srcBase;
    request.srcRowBytes = srcRowBytes;
    request.width = width;
    request.height = height;
    request.time = time;
    request.previewMode = previewMode;
    request.qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    request.samplingMode = getSamplingModeValue(getChoiceValue("cubeViewerSamplingMode", time, 0));
    request.scaleIndex = getChoiceValue("cubeViewerScale", time, 3);
    request.scaleFactor = scaleFactorForIndex(request.scaleIndex);
    request.resolution = qualityResolutionForIndex(request.qualityIndex);
    request.preserveOverflow = currentShowOverflow(time);
    request.occupancyFill = currentOccupancyGuidedFill(time);
    request.plotDisplayLinearEnabled = currentPlotDisplayLinearEnabled(time);
    request.plotDisplayLinearTransferId = static_cast<int>(currentPlotDisplayLinearTransferId(time));
    request.plotMode = currentPlotMode(time);
    request.settingsKey = currentCloudSettingsKey(time);
    return request;
  }

  static bool stableSampleLess(const ViewerCloudSample& a, const ViewerCloudSample& b) {
    if (a.xNorm != b.xNorm) return a.xNorm < b.xNorm;
    if (a.yNorm != b.yNorm) return a.yNorm < b.yNorm;
    if (a.r != b.r) return a.r < b.r;
    if (a.g != b.g) return a.g < b.g;
    return a.b < b.b;
  }

  template <typename SampleT>
  ViewerCloudSample toViewerCloudSample(const SampleT& sample) {
    ViewerCloudSample out{};
    out.xNorm = sample.xNorm;
    out.yNorm = sample.yNorm;
    out.zReserved = sample.zReserved;
    out.r = sample.r;
    out.g = sample.g;
    out.b = sample.b;
    return out;
  }

  template <typename SampleT>
  std::vector<ViewerCloudSample> toViewerCloudSamples(const std::vector<SampleT>& sourceSamples) {
    std::vector<ViewerCloudSample> out;
    const std::size_t sampleCount = sourceSamples.size();
    out.reserve(sampleCount);
    for (std::size_t i = 0; i < sampleCount; ++i) {
      out.push_back(toViewerCloudSample(sourceSamples[i]));
    }
    return out;
  }

#if defined(__APPLE__)
#endif

  CloudBuildResult finalizeCloudBuildFromSamples(
      const std::vector<ViewerCloudSample>& samples,
      const std::string& paramHash,
      const std::string& settingsKey,
      int resolution,
      int qualityIndex,
      int width,
      int height,
      const std::string& backendName,
      int primaryAttempts,
      int primaryAccepted,
      bool allowFastTransport = true) {
    CloudBuildResult out{};
    std::shared_ptr<ViewerCloudTransportBlob> fastBlob;
    std::string transportMode = "json";
    out.paramHash = paramHash;
    out.samples = samples;
    out.contentHash = fnv1a64Bytes(samples.data(), samples.size() * sizeof(ViewerCloudSample));
    if (allowFastTransport) {
      out.payload = buildInputCloudJson(samples, out.paramHash, settingsKey, resolution, qualityIndex, &fastBlob, &transportMode);
    } else {
      const std::string points = serializeViewerCloudSamples(samples);
      out.payload = buildInputCloudJson(points, out.paramHash, settingsKey, resolution, qualityIndex);
      transportMode = "json";
    }
    out.fastBlob = std::move(fastBlob);
    out.backendName = backendName;
    out.quality = qualityLabelForIndex(qualityIndex);
    out.sampleCount = samples.size();
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    cubeViewerDebugLog(std::string("Built viewer cloud backend=") + backendName +
                       " transport=" + transportMode +
                       " samples=" + std::to_string(samples.size()) +
                       " primaryAttempts=" + std::to_string(primaryAttempts) +
                       " primaryAccepted=" + std::to_string(primaryAccepted) +
                       " res=" + std::to_string(resolution));
    return out;
  }

#if defined(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
  bool buildWholeImageCloudCuda(
      const ViewerCloudBuildRequest& request,
      SliceSelectionSpec sliceSpec,
      cudaStream_t stream,
      ViewerCloudSamplesBuildResult* out,
      std::string* reason) {
    if (!out) return false;
    *out = ViewerCloudSamplesBuildResult{};
    if (!request.srcBase || request.srcRowBytes == 0 || request.width <= 0 || request.height <= 0) {
      if (reason) *reason = "invalid-source";
      return false;
    }

    const int scaledWidth = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.width) * request.scaleFactor)));
    const int scaledHeight = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.height) * request.scaleFactor)));
    const int pointCount = std::max(512, static_cast<int>(std::lround(
        static_cast<double>(qualityPointCountForIndex(request.qualityIndex, request.previewMode)) *
        request.scaleFactor * request.scaleFactor)));
    int selectionRetryMultiplier = sliceSpec.enabled ? 12 : 1;
    const float effectiveNeutralRadius = effectiveNeutralRadiusThreshold(sliceSpec.neutralRadius);
    if (sliceSpec.neutralRadiusEnabled) {
      const int neutralRetryMultiplier =
          8 + static_cast<int>(std::lround(clampf(1.0f - effectiveNeutralRadius, 0.0f, 1.0f) * 18.0f));
      selectionRetryMultiplier = std::max(selectionRetryMultiplier, neutralRetryMultiplier);
    }
    const bool anyHueRegionSelected = sliceSpec.cubeSliceRed || sliceSpec.cubeSliceGreen || sliceSpec.cubeSliceBlue ||
                                      sliceSpec.cubeSliceCyan || sliceSpec.cubeSliceYellow || sliceSpec.cubeSliceMagenta;
    const bool noHueRegionSelected = sliceSpec.enabled && !anyHueRegionSelected;
    const bool selectionImpossible = noHueRegionSelected;
    const int maxPrimaryAttempts = selectionImpossible
                                       ? 0
                                       : (selectionRetryMultiplier <= 1
                                              ? pointCount
                                              : std::min(std::max(pointCount * selectionRetryMultiplier, pointCount + 4096),
                                                         request.previewMode ? 750000 : 3000000));
    const int extraPointCount = std::max(2048, pointCount / (request.previewMode ? 4 : 2));
    const int candidateTarget = std::min(std::max(extraPointCount * 3, 8192), request.previewMode ? 32768 : 131072);
    const int maxCandidateAttempts = !request.occupancyFill || selectionImpossible
                                         ? 0
                                         : (selectionRetryMultiplier <= 1
                                                ? candidateTarget
                                                : std::min(std::max(candidateTarget * selectionRetryMultiplier, candidateTarget + 4096),
                                                           request.previewMode ? 750000 : 3000000));

    int plotMode = 0;
    switch (sliceSpec.plotMode) {
      case SlicePlotModeKind::Hsl: plotMode = 1; break;
      case SlicePlotModeKind::Hsv: plotMode = 2; break;
      case SlicePlotModeKind::Chen: plotMode = 3; break;
      case SlicePlotModeKind::RgbToCone: plotMode = 4; break;
      case SlicePlotModeKind::JpConical: plotMode = 5; break;
      case SlicePlotModeKind::NormCone: plotMode = 6; break;
      case SlicePlotModeKind::Reuleaux: plotMode = 7; break;
      default: plotMode = 0; break;
    }

    ChromaspaceCloudCuda::Request cudaRequest{};
    cudaRequest.srcBase = request.srcBase;
    cudaRequest.srcRowBytes = request.srcRowBytes;
    cudaRequest.width = request.width;
    cudaRequest.height = request.height;
    cudaRequest.scaledWidth = scaledWidth;
    cudaRequest.scaledHeight = scaledHeight;
    cudaRequest.pointCount = pointCount;
    cudaRequest.extraPointCount = request.occupancyFill ? extraPointCount : 0;
    cudaRequest.candidateTarget = request.occupancyFill ? candidateTarget : 0;
    cudaRequest.maxPrimaryAttempts = maxPrimaryAttempts;
    cudaRequest.maxCandidateAttempts = maxCandidateAttempts;
    cudaRequest.samplingMode = request.samplingMode;
    cudaRequest.preserveOverflow = request.preserveOverflow ? 1 : 0;
    cudaRequest.occupancyFill = request.occupancyFill ? 1 : 0;
    cudaRequest.plotMode = plotMode;
    cudaRequest.circularHsl = sliceSpec.circularHsl ? 1 : 0;
    cudaRequest.circularHsv = sliceSpec.circularHsv ? 1 : 0;
    cudaRequest.normConeNormalized = sliceSpec.normConeNormalized ? 1 : 0;
    cudaRequest.showOverflow = sliceSpec.showOverflow ? 1 : 0;
    cudaRequest.plotDisplayLinearEnabled = request.plotDisplayLinearEnabled ? 1 : 0;
    cudaRequest.plotDisplayLinearTransfer = request.plotDisplayLinearTransferId;
    cudaRequest.neutralRadiusEnabled = sliceSpec.neutralRadiusEnabled ? 1 : 0;
    cudaRequest.neutralRadius = sliceSpec.neutralRadius;
    cudaRequest.stream = stream;

    ChromaspaceCloudCuda::Result cudaResult{};
    const auto start = std::chrono::steady_clock::now();
    if (!ChromaspaceCloudCuda::buildWholeImageCloud(cudaRequest, &cudaResult) || !cudaResult.success) {
      if (reason) *reason = cudaResult.error.empty() ? "cuda-cloud-build-failed" : cudaResult.error;
      return false;
    }

    std::vector<ViewerCloudSample> samples;
    samples.reserve(cudaResult.primarySamples.size() + static_cast<std::size_t>(cudaResult.extraPointCount));
    for (const auto& sample : cudaResult.primarySamples) {
      samples.push_back({sample.xNorm, sample.yNorm, sample.zReserved, sample.r, sample.g, sample.b});
    }
    std::sort(samples.begin(), samples.end(), stableSampleLess);

    if (!cudaResult.appendedSamples.empty()) {
      const auto appended = toViewerCloudSamples(cudaResult.appendedSamples);
      samples.insert(samples.end(), appended.begin(), appended.end());
    }

    std::ostringstream hash;
    hash << request.width << 'x' << request.height << ':' << request.resolution << ':'
         << qualityLabelForIndex(request.qualityIndex)
         << ":sampling=" << samplingModeLabelForIndex(request.samplingMode)
         << ":occupancyFill=" << (request.occupancyFill ? 1 : 0)
         << ":scale=" << scaleLabelForIndex(request.scaleIndex);

    out->samples = std::move(samples);
    out->paramHash = hash.str();
    out->quality = qualityLabelForIndex(request.qualityIndex);
    out->backendName = "CUDA";
    out->resolution = request.resolution;
    out->sourceWidth = request.width;
    out->sourceHeight = request.height;
    out->primaryAttempts = cudaResult.primaryAttempts;
    out->primaryAccepted = cudaResult.primaryAccepted;
    out->success = true;
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    cubeViewerDebugLog(std::string("Viewer cloud CUDA kernel path succeeded samples=") + std::to_string(out->samples.size()) +
                       " primaryAccepted=" + std::to_string(out->primaryAccepted) +
                       " elapsedMs=" + std::to_string(elapsedMs));
    return true;
  }

  bool buildInstance1StripCloudCuda(
      Image* src,
      const RenderArguments& args,
      std::vector<ViewerCloudSample>* outSamples,
      std::string* outParamHash,
      int* outResolution,
      std::string* reason) {
    if (outSamples) outSamples->clear();
    if (!src || !outSamples || !outParamHash || !outResolution) {
      if (reason) *reason = "invalid-output";
      return false;
    }
    if (!args.isEnabledCudaRender || args.pCudaStream == nullptr || src->getPixelData() == nullptr) {
      if (reason) *reason = "cuda-unavailable";
      return false;
    }
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    if (width <= 0 || height <= 0) {
      if (reason) *reason = "invalid-bounds";
      return false;
    }
    const bool readCube = currentReadIdentityPlot(args.time);
    const bool readRamp = currentReadGrayRamp(args.time);
    if (!readCube && !readRamp) {
      if (reason) *reason = "instance1-disabled";
      return false;
    }
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", args.time, 29);
    const int resolution = clampOverlayCubeSize(requestedSize);
    const std::string cacheKey =
        currentIdentityStripCacheKey(args.time, width, height, bounds.x1, bounds.y1, resolution, readCube, readRamp, "cuda-strip");
    const std::string paramHash = buildIdentityStripParamHash(width, height, resolution, readCube, readRamp, args.time);
    if (tryGetCachedIdentityStripCloud(cacheKey, outSamples, outParamHash, outResolution)) {
      cubeViewerDebugLog(std::string("Viewer identity strip CUDA cache hit samples=") + std::to_string(outSamples->size()));
      return true;
    }
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    if (!computeIdentityStripLayout(bounds, resolution, readRamp, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) {
      if (reason) *reason = "layout-failed";
      return false;
    }
    ChromaspaceCloudCuda::StripRequest request{};
    request.width = width;
    request.height = height;
    request.resolution = resolution;
    request.preserveOverflow = currentShowOverflow(args.time) ? 1 : 0;
    request.readCube = readCube ? 1 : 0;
    request.readRamp = readRamp ? 1 : 0;
    request.plotDisplayLinearEnabled = currentPlotDisplayLinearEnabled(args.time) ? 1 : 0;
    request.plotDisplayLinearTransfer = static_cast<int>(currentPlotDisplayLinearTransferId(args.time));
    request.cubeY1 = cubeY1 - bounds.y1;
    request.stripHeight = stripHeight;
    request.rampY1 = rampY1 - bounds.y1;
    request.rampHeight = std::max(0, rampY2 - rampY1);
    request.rampSampleRows = std::min(request.rampHeight, std::max(4, std::min(8, resolution / 8 + 2)));
    request.cellWidth = identityStripCellWidth(width, resolution);
    request.stream = reinterpret_cast<cudaStream_t>(args.pCudaStream);
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
    request.srcRowBytes = srcRowBytes;
    request.srcBase = reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(src->getPixelData()) +
        static_cast<size_t>(bounds.y1) * srcRowBytes +
        static_cast<size_t>(bounds.x1) * 4u * sizeof(float));

    ChromaspaceCloudCuda::StripResult stripResult{};
    const auto start = std::chrono::steady_clock::now();
    if (!ChromaspaceCloudCuda::buildIdentityStripCloud(request, &stripResult) || !stripResult.success) {
      if (reason) *reason = stripResult.error.empty() ? "cuda-strip-build-failed" : stripResult.error;
      return false;
    }
    *outSamples = toViewerCloudSamples(stripResult.samples);
    *outParamHash = paramHash;
    *outResolution = resolution;
    storeCachedIdentityStripCloud(cacheKey, *outSamples, *outParamHash, *outResolution);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    cubeViewerDebugLog(std::string("Viewer identity strip CUDA path succeeded samples=") + std::to_string(outSamples->size()) +
                       " elapsedMs=" + std::to_string(elapsedMs));
    return true;
  }
#endif

#if defined(__APPLE__)
  bool buildWholeImageCloudMetal(
      const ViewerCloudBuildRequest& request,
      SliceSelectionSpec sliceSpec,
      Image* src,
      const RenderArguments& args,
      ViewerCloudSamplesBuildResult* out,
      std::string* reason) {
    if (!out) return false;
    *out = ViewerCloudSamplesBuildResult{};
    if (!src || !src->getPixelData() || !args.isEnabledMetalRender || args.pMetalCmdQ == nullptr ||
        request.width <= 0 || request.height <= 0) {
      if (reason) *reason = "metal-unavailable";
      return false;
    }

    const int scaledWidth = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.width) * request.scaleFactor)));
    const int scaledHeight = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.height) * request.scaleFactor)));
    const int pointCount = std::max(512, static_cast<int>(std::lround(
        static_cast<double>(qualityPointCountForIndex(request.qualityIndex, request.previewMode)) *
        request.scaleFactor * request.scaleFactor)));
    int selectionRetryMultiplier = sliceSpec.enabled ? 12 : 1;
    const float effectiveNeutralRadius = effectiveNeutralRadiusThreshold(sliceSpec.neutralRadius);
    if (sliceSpec.neutralRadiusEnabled) {
      const int neutralRetryMultiplier =
          8 + static_cast<int>(std::lround(clampf(1.0f - effectiveNeutralRadius, 0.0f, 1.0f) * 18.0f));
      selectionRetryMultiplier = std::max(selectionRetryMultiplier, neutralRetryMultiplier);
    }
    const bool anyHueRegionSelected = sliceSpec.cubeSliceRed || sliceSpec.cubeSliceGreen || sliceSpec.cubeSliceBlue ||
                                      sliceSpec.cubeSliceCyan || sliceSpec.cubeSliceYellow || sliceSpec.cubeSliceMagenta;
    const bool noHueRegionSelected = sliceSpec.enabled && !anyHueRegionSelected;
    const bool selectionImpossible = noHueRegionSelected;
    const int maxPrimaryAttempts = selectionImpossible
                                       ? 0
                                       : (selectionRetryMultiplier <= 1
                                              ? pointCount
                                              : std::min(std::max(pointCount * selectionRetryMultiplier, pointCount + 4096),
                                                         request.previewMode ? 750000 : 3000000));
    const int extraPointCount = std::max(2048, pointCount / (request.previewMode ? 4 : 2));
    const int candidateTarget = std::min(std::max(extraPointCount * 3, 8192), request.previewMode ? 32768 : 131072);
    const int maxCandidateAttempts = !request.occupancyFill || selectionImpossible
                                         ? 0
                                         : (selectionRetryMultiplier <= 1
                                                ? candidateTarget
                                                : std::min(std::max(candidateTarget * selectionRetryMultiplier, candidateTarget + 4096),
                                                           request.previewMode ? 750000 : 3000000));

    int plotMode = 0;
    switch (sliceSpec.plotMode) {
      case SlicePlotModeKind::Hsl: plotMode = 1; break;
      case SlicePlotModeKind::Hsv: plotMode = 2; break;
      case SlicePlotModeKind::Chen: plotMode = 3; break;
      case SlicePlotModeKind::RgbToCone: plotMode = 4; break;
      case SlicePlotModeKind::JpConical: plotMode = 5; break;
      case SlicePlotModeKind::NormCone: plotMode = 6; break;
      case SlicePlotModeKind::Reuleaux: plotMode = 7; break;
      default: plotMode = 0; break;
    }

    const OfxRectI bounds = src->getBounds();
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t packedRowBytes = static_cast<size_t>(request.width) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = packedRowBytes;

    ChromaspaceMetal::Request metalRequest{};
    metalRequest.srcMetalBuffer = src->getPixelData();
    metalRequest.srcRowBytes = srcRowBytes;
    metalRequest.width = request.width;
    metalRequest.height = request.height;
    metalRequest.originX = bounds.x1;
    metalRequest.originY = bounds.y1;
    metalRequest.scaledWidth = scaledWidth;
    metalRequest.scaledHeight = scaledHeight;
    metalRequest.pointCount = pointCount;
    metalRequest.extraPointCount = request.occupancyFill ? extraPointCount : 0;
    metalRequest.candidateTarget = request.occupancyFill ? candidateTarget : 0;
    metalRequest.maxPrimaryAttempts = maxPrimaryAttempts;
    metalRequest.maxCandidateAttempts = maxCandidateAttempts;
    metalRequest.samplingMode = request.samplingMode;
    metalRequest.preserveOverflow = request.preserveOverflow ? 1 : 0;
    metalRequest.occupancyFill = request.occupancyFill ? 1 : 0;
    metalRequest.plotMode = plotMode;
    metalRequest.circularHsl = sliceSpec.circularHsl ? 1 : 0;
    metalRequest.circularHsv = sliceSpec.circularHsv ? 1 : 0;
    metalRequest.normConeNormalized = sliceSpec.normConeNormalized ? 1 : 0;
    metalRequest.showOverflow = sliceSpec.showOverflow ? 1 : 0;
    metalRequest.plotDisplayLinearEnabled = request.plotDisplayLinearEnabled ? 1 : 0;
    metalRequest.plotDisplayLinearTransfer = request.plotDisplayLinearTransferId;
    metalRequest.neutralRadiusEnabled = sliceSpec.neutralRadiusEnabled ? 1 : 0;
    metalRequest.neutralRadius = sliceSpec.neutralRadius;
    metalRequest.metalCommandQueue = args.pMetalCmdQ;

    ChromaspaceMetal::Result metalResult{};
    const auto start = std::chrono::steady_clock::now();
    if (!ChromaspaceMetal::buildWholeImageCloud(metalRequest, &metalResult) || !metalResult.success) {
      if (reason) *reason = metalResult.error.empty() ? "metal-cloud-build-failed" : metalResult.error;
      return false;
    }

    std::vector<ViewerCloudSample> samples = toViewerCloudSamples(metalResult.primarySamples);
    std::sort(samples.begin(), samples.end(), stableSampleLess);

    if (!metalResult.appendedSamples.empty()) {
      const auto appended = toViewerCloudSamples(metalResult.appendedSamples);
      samples.insert(samples.end(), appended.begin(), appended.end());
    }

    std::ostringstream hash;
    hash << request.width << 'x' << request.height << ':' << request.resolution << ':'
         << qualityLabelForIndex(request.qualityIndex)
         << ":sampling=" << samplingModeLabelForIndex(request.samplingMode)
         << ":occupancyFill=" << (request.occupancyFill ? 1 : 0)
         << ":scale=" << scaleLabelForIndex(request.scaleIndex);

    out->samples = std::move(samples);
    out->paramHash = hash.str();
    out->quality = qualityLabelForIndex(request.qualityIndex);
    out->backendName = "Metal";
    out->resolution = request.resolution;
    out->sourceWidth = request.width;
    out->sourceHeight = request.height;
    out->primaryAttempts = metalResult.primaryAttempts;
    out->primaryAccepted = metalResult.primaryAccepted;
    out->success = true;
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  cubeViewerDebugLog(std::string("Viewer cloud Metal kernel path succeeded samples=") + std::to_string(out->samples.size()) +
                     " primaryAccepted=" + std::to_string(out->primaryAccepted) +
                     " elapsedMs=" + std::to_string(elapsedMs));
  return true;
}

  bool buildWholeImageAndInstance1CloudMetal(
      const ViewerCloudBuildRequest& request,
      SliceSelectionSpec sliceSpec,
      Image* src,
      const RenderArguments& args,
      ViewerCloudSamplesBuildResult* out,
      int* outStripResolution,
      std::string* reason) {
    if (!out) return false;
    *out = ViewerCloudSamplesBuildResult{};
    if (outStripResolution) *outStripResolution = request.resolution;
    if (!src || !src->getPixelData() || !args.isEnabledMetalRender || args.pMetalCmdQ == nullptr ||
        request.width <= 0 || request.height <= 0) {
      if (reason) *reason = "metal-unavailable";
      return false;
    }

    const int scaledWidth = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.width) * request.scaleFactor)));
    const int scaledHeight = std::max(1, static_cast<int>(std::lround(static_cast<double>(request.height) * request.scaleFactor)));
    const int pointCount = std::max(512, static_cast<int>(std::lround(
        static_cast<double>(qualityPointCountForIndex(request.qualityIndex, request.previewMode)) *
        request.scaleFactor * request.scaleFactor)));
    int selectionRetryMultiplier = sliceSpec.enabled ? 12 : 1;
    const float effectiveNeutralRadius = effectiveNeutralRadiusThreshold(sliceSpec.neutralRadius);
    if (sliceSpec.neutralRadiusEnabled) {
      const int neutralRetryMultiplier =
          8 + static_cast<int>(std::lround(clampf(1.0f - effectiveNeutralRadius, 0.0f, 1.0f) * 18.0f));
      selectionRetryMultiplier = std::max(selectionRetryMultiplier, neutralRetryMultiplier);
    }
    const bool anyHueRegionSelected = sliceSpec.cubeSliceRed || sliceSpec.cubeSliceGreen || sliceSpec.cubeSliceBlue ||
                                      sliceSpec.cubeSliceCyan || sliceSpec.cubeSliceYellow || sliceSpec.cubeSliceMagenta;
    const bool noHueRegionSelected = sliceSpec.enabled && !anyHueRegionSelected;
    const bool selectionImpossible = noHueRegionSelected;
    const int maxPrimaryAttempts = selectionImpossible
                                       ? 0
                                       : (selectionRetryMultiplier <= 1
                                              ? pointCount
                                              : std::min(std::max(pointCount * selectionRetryMultiplier, pointCount + 4096),
                                                         request.previewMode ? 750000 : 3000000));
    const int extraPointCount = std::max(2048, pointCount / (request.previewMode ? 4 : 2));
    const int candidateTarget = std::min(std::max(extraPointCount * 3, 8192), request.previewMode ? 32768 : 131072);
    const int maxCandidateAttempts = !request.occupancyFill || selectionImpossible
                                         ? 0
                                         : (selectionRetryMultiplier <= 1
                                                ? candidateTarget
                                                : std::min(std::max(candidateTarget * selectionRetryMultiplier, candidateTarget + 4096),
                                                           request.previewMode ? 750000 : 3000000));

    int plotMode = 0;
    switch (sliceSpec.plotMode) {
      case SlicePlotModeKind::Hsl: plotMode = 1; break;
      case SlicePlotModeKind::Hsv: plotMode = 2; break;
      case SlicePlotModeKind::Chen: plotMode = 3; break;
      case SlicePlotModeKind::RgbToCone: plotMode = 4; break;
      case SlicePlotModeKind::JpConical: plotMode = 5; break;
      case SlicePlotModeKind::NormCone: plotMode = 6; break;
      case SlicePlotModeKind::Reuleaux: plotMode = 7; break;
      default: plotMode = 0; break;
    }

    const OfxRectI bounds = src->getBounds();
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t packedRowBytes = static_cast<size_t>(request.width) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = packedRowBytes;

    const bool readCube = currentReadIdentityPlot(args.time);
    const bool readRamp = currentReadGrayRamp(args.time);
    if (!readCube && !readRamp) {
      if (reason) *reason = "instance1-disabled";
      return false;
    }
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", args.time, 29);
    const int stripResolution = clampOverlayCubeSize(requestedSize);
    if (outStripResolution) *outStripResolution = stripResolution;
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    if (!computeIdentityStripLayout(bounds, stripResolution, readRamp, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) {
      if (reason) *reason = "layout-failed";
      return false;
    }

    ChromaspaceMetal::Request wholeRequest{};
    wholeRequest.srcMetalBuffer = src->getPixelData();
    wholeRequest.srcRowBytes = srcRowBytes;
    wholeRequest.width = request.width;
    wholeRequest.height = request.height;
    wholeRequest.originX = bounds.x1;
    wholeRequest.originY = bounds.y1;
    wholeRequest.scaledWidth = scaledWidth;
    wholeRequest.scaledHeight = scaledHeight;
    wholeRequest.pointCount = pointCount;
    wholeRequest.extraPointCount = request.occupancyFill ? extraPointCount : 0;
    wholeRequest.candidateTarget = request.occupancyFill ? candidateTarget : 0;
    wholeRequest.maxPrimaryAttempts = maxPrimaryAttempts;
    wholeRequest.maxCandidateAttempts = maxCandidateAttempts;
    wholeRequest.samplingMode = request.samplingMode;
    wholeRequest.preserveOverflow = request.preserveOverflow ? 1 : 0;
    wholeRequest.occupancyFill = request.occupancyFill ? 1 : 0;
    wholeRequest.plotMode = plotMode;
    wholeRequest.circularHsl = sliceSpec.circularHsl ? 1 : 0;
    wholeRequest.circularHsv = sliceSpec.circularHsv ? 1 : 0;
    wholeRequest.normConeNormalized = sliceSpec.normConeNormalized ? 1 : 0;
    wholeRequest.showOverflow = sliceSpec.showOverflow ? 1 : 0;
    wholeRequest.plotDisplayLinearEnabled = request.plotDisplayLinearEnabled ? 1 : 0;
    wholeRequest.plotDisplayLinearTransfer = request.plotDisplayLinearTransferId;
    wholeRequest.neutralRadiusEnabled = sliceSpec.neutralRadiusEnabled ? 1 : 0;
    wholeRequest.neutralRadius = sliceSpec.neutralRadius;
    wholeRequest.metalCommandQueue = args.pMetalCmdQ;

    ChromaspaceMetal::StripRequest stripRequest{};
    stripRequest.srcMetalBuffer = src->getPixelData();
    stripRequest.srcRowBytes = srcRowBytes;
    stripRequest.width = request.width;
    stripRequest.height = request.height;
    stripRequest.originX = bounds.x1;
    stripRequest.originY = bounds.y1;
    stripRequest.resolution = stripResolution;
    stripRequest.preserveOverflow = currentShowOverflow(args.time) ? 1 : 0;
    stripRequest.readCube = readCube ? 1 : 0;
    stripRequest.readRamp = readRamp ? 1 : 0;
    stripRequest.plotDisplayLinearEnabled = currentPlotDisplayLinearEnabled(args.time) ? 1 : 0;
    stripRequest.plotDisplayLinearTransfer = static_cast<int>(currentPlotDisplayLinearTransferId(args.time));
    stripRequest.cubeY1 = cubeY1 - bounds.y1;
    stripRequest.stripHeight = stripHeight;
    stripRequest.rampY1 = rampY1 - bounds.y1;
    stripRequest.rampHeight = std::max(0, rampY2 - rampY1);
    stripRequest.rampSampleRows = std::min(stripRequest.rampHeight, std::max(4, std::min(8, stripResolution / 8 + 2)));
    stripRequest.cellWidth = identityStripCellWidth(request.width, stripResolution);
    stripRequest.metalCommandQueue = args.pMetalCmdQ;

    ChromaspaceMetal::CombinedResult combinedResult{};
    const auto start = std::chrono::steady_clock::now();
    if (!ChromaspaceMetal::buildWholeImageAndIdentityStripCloud(wholeRequest, stripRequest, &combinedResult) || !combinedResult.success) {
      if (reason) *reason = combinedResult.error.empty() ? "metal-combined-cloud-build-failed" : combinedResult.error;
      return false;
    }

    std::sort(combinedResult.primarySamples.begin(), combinedResult.primarySamples.end(),
              [](const ChromaspaceMetal::Sample& a, const ChromaspaceMetal::Sample& b) {
                if (a.xNorm != b.xNorm) return a.xNorm < b.xNorm;
                if (a.yNorm != b.yNorm) return a.yNorm < b.yNorm;
                if (a.r != b.r) return a.r < b.r;
                if (a.g != b.g) return a.g < b.g;
                return a.b < b.b;
              });

    std::vector<ViewerCloudSample> samples = toViewerCloudSamples(combinedResult.combinedSamples);
    if (!combinedResult.appendedSamples.empty()) {
      const auto appended = toViewerCloudSamples(combinedResult.appendedSamples);
      samples.insert(samples.end(), appended.begin(), appended.end());
    }

    std::ostringstream hash;
    hash << request.width << 'x' << request.height << ':' << request.resolution << ':'
         << qualityLabelForIndex(request.qualityIndex)
         << ":sampling=" << samplingModeLabelForIndex(request.samplingMode)
         << ":occupancyFill=" << (request.occupancyFill ? 1 : 0)
         << ":scale=" << scaleLabelForIndex(request.scaleIndex);

    out->samples = std::move(samples);
    out->paramHash = hash.str() + "+instance1=" + std::to_string(stripResolution);
    out->identityStripSamples = toViewerCloudSamples(combinedResult.stripSamples);
    out->identityStripParamHash =
        buildIdentityStripParamHash(request.width, request.height, stripResolution, readCube, readRamp, args.time);
    out->quality = qualityLabelForIndex(request.qualityIndex);
    out->backendName = "Metal+Metal-strip";
    out->resolution = request.resolution;
    out->identityStripResolution = stripResolution;
    out->sourceWidth = request.width;
    out->sourceHeight = request.height;
    out->primaryAttempts = combinedResult.primaryAttempts;
    out->primaryAccepted = combinedResult.primaryAccepted + static_cast<int>(combinedResult.stripSamples.size());
    out->success = true;
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    cubeViewerDebugLog(std::string("Viewer combined Metal kernel path succeeded samples=") + std::to_string(out->samples.size()) +
                       " primaryAccepted=" + std::to_string(out->primaryAccepted) +
                       " elapsedMs=" + std::to_string(elapsedMs));
    return true;
  }

  bool buildInstance1StripCloudMetal(
      Image* src,
      const RenderArguments& args,
      std::vector<ViewerCloudSample>* outSamples,
      std::string* outParamHash,
      int* outResolution,
      std::string* reason) {
    if (outSamples) outSamples->clear();
    if (!src || !outSamples || !outParamHash || !outResolution) {
      if (reason) *reason = "invalid-output";
      return false;
    }
    if (!args.isEnabledMetalRender || args.pMetalCmdQ == nullptr || src->getPixelData() == nullptr) {
      if (reason) *reason = "metal-unavailable";
      return false;
    }
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    if (width <= 0 || height <= 0) {
      if (reason) *reason = "invalid-bounds";
      return false;
    }
    const bool readCube = currentReadIdentityPlot(args.time);
    const bool readRamp = currentReadGrayRamp(args.time);
    if (!readCube && !readRamp) {
      if (reason) *reason = "instance1-disabled";
      return false;
    }
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", args.time, 29);
    const int resolution = clampOverlayCubeSize(requestedSize);
    const std::string cacheKey =
        currentIdentityStripCacheKey(args.time, width, height, bounds.x1, bounds.y1, resolution, readCube, readRamp, "metal-strip");
    const std::string paramHash = buildIdentityStripParamHash(width, height, resolution, readCube, readRamp, args.time);
    if (tryGetCachedIdentityStripCloud(cacheKey, outSamples, outParamHash, outResolution)) {
      cubeViewerDebugLog(std::string("Viewer identity strip Metal cache hit samples=") + std::to_string(outSamples->size()));
      return true;
    }
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    if (!computeIdentityStripLayout(bounds, resolution, readRamp, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) {
      if (reason) *reason = "layout-failed";
      return false;
    }

    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = packedRowBytes;

    ChromaspaceMetal::StripRequest request{};
    request.srcMetalBuffer = src->getPixelData();
    request.srcRowBytes = srcRowBytes;
    request.width = width;
    request.height = height;
    request.originX = bounds.x1;
    request.originY = bounds.y1;
    request.resolution = resolution;
    request.preserveOverflow = currentShowOverflow(args.time) ? 1 : 0;
    request.readCube = readCube ? 1 : 0;
    request.readRamp = readRamp ? 1 : 0;
    request.plotDisplayLinearEnabled = currentPlotDisplayLinearEnabled(args.time) ? 1 : 0;
    request.plotDisplayLinearTransfer = static_cast<int>(currentPlotDisplayLinearTransferId(args.time));
    request.cubeY1 = cubeY1 - bounds.y1;
    request.stripHeight = stripHeight;
    request.rampY1 = rampY1 - bounds.y1;
    request.rampHeight = std::max(0, rampY2 - rampY1);
    request.rampSampleRows = std::min(request.rampHeight, std::max(4, std::min(8, resolution / 8 + 2)));
    request.cellWidth = identityStripCellWidth(width, resolution);
    request.metalCommandQueue = args.pMetalCmdQ;

    ChromaspaceMetal::StripResult stripResult{};
    const auto start = std::chrono::steady_clock::now();
    if (!ChromaspaceMetal::buildIdentityStripCloud(request, &stripResult) || !stripResult.success) {
      if (reason) *reason = stripResult.error.empty() ? "metal-strip-build-failed" : stripResult.error;
      return false;
    }
    *outSamples = toViewerCloudSamples(stripResult.samples);
    *outParamHash = paramHash;
    *outResolution = resolution;
    storeCachedIdentityStripCloud(cacheKey, *outSamples, *outParamHash, *outResolution);
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    cubeViewerDebugLog(std::string("Viewer identity strip Metal path succeeded samples=") + std::to_string(outSamples->size()) +
                       " elapsedMs=" + std::to_string(elapsedMs));
    return true;
  }
#endif

  int occupancyBinIndex(float r, float g, float b, bool preserveOverflow) const {
    constexpr int kCoreBins = 16;
    constexpr int kOverflowBinsPerAxis = kCoreBins + 2;
    const auto toBin = [preserveOverflow, kCoreBins, kOverflowBinsPerAxis](float v) -> int {
      if (!preserveOverflow) {
        const float clamped = clamp01(v);
        return std::clamp(static_cast<int>(std::floor(clamped * static_cast<float>(kCoreBins))), 0, kCoreBins - 1);
      }
      if (v < 0.0f) return 0;
      if (v > 1.0f) return kOverflowBinsPerAxis - 1;
      return 1 + std::clamp(static_cast<int>(std::floor(v * static_cast<float>(kCoreBins))), 0, kCoreBins - 1);
    };
    const int binsPerAxis = preserveOverflow ? kOverflowBinsPerAxis : kCoreBins;
    const int ri = toBin(r);
    const int gi = toBin(g);
    const int bi = toBin(b);
    return (ri * binsPerAxis + gi) * binsPerAxis + bi;
  }

  template <typename FetchPixelFn>
  CloudBuildResult buildInputCloudPayloadFromWholeImageSamples(
      int width,
      int height,
      double time,
      bool previewMode,
      FetchPixelFn&& fetchPixel) {
    CloudBuildResult out;
    if (width <= 0 || height <= 0) return out;

    struct OccupancyCandidate {
      float xNorm = 0.0f;
      float yNorm = 0.0f;
      float r = 0.0f;
      float g = 0.0f;
      float b = 0.0f;
      float normalizedNeutralRadius = 0.0f;
      int bin = 0;
      int binRank = 0;
      uint32_t tie = 0;
    };

    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int samplingMode = getSamplingModeValue(getChoiceValue("cubeViewerSamplingMode", time, 0));
    const int scaleIndex = getChoiceValue("cubeViewerScale", time, 3);
    const double scaleFactor = scaleFactorForIndex(scaleIndex);
    const int resolution = qualityResolutionForIndex(qualityIndex);
    const bool preserveOverflow = currentShowOverflow(time);
    const bool occupancyFill = currentOccupancyGuidedFill(time);
    const SliceSelectionSpec hueSliceSpec = currentHueSectorSliceSpec(time);
    const float effectiveNeutralRadius = effectiveNeutralRadiusThreshold(hueSliceSpec.neutralRadius);
    const bool anyHueRegionSelected = hueSliceSpec.cubeSliceRed || hueSliceSpec.cubeSliceGreen ||
                                      hueSliceSpec.cubeSliceBlue || hueSliceSpec.cubeSliceCyan ||
                                      hueSliceSpec.cubeSliceYellow || hueSliceSpec.cubeSliceMagenta;
    const bool noHueRegionSelected = hueSliceSpec.enabled && !anyHueRegionSelected;
    const bool selectionImpossible = noHueRegionSelected;
    const int scaledWidth = std::max(1, static_cast<int>(std::lround(static_cast<double>(width) * scaleFactor)));
    const int scaledHeight = std::max(1, static_cast<int>(std::lround(static_cast<double>(height) * scaleFactor)));
    const int pointCount = std::max(512, static_cast<int>(std::lround(
        static_cast<double>(qualityPointCountForIndex(qualityIndex, previewMode)) * scaleFactor * scaleFactor)));
    int selectionRetryMultiplier = hueSliceSpec.enabled ? 12 : 1;
    if (hueSliceSpec.neutralRadiusEnabled) {
      const int neutralRetryMultiplier =
          8 + static_cast<int>(std::lround(clampf(1.0f - effectiveNeutralRadius, 0.0f, 1.0f) * 18.0f));
      selectionRetryMultiplier = std::max(selectionRetryMultiplier, neutralRetryMultiplier);
    }
    const int maxPrimaryAttempts = selectionImpossible
                                       ? 0
                                       : (selectionRetryMultiplier <= 1
                                              ? pointCount
                                              : std::min(std::max(pointCount * selectionRetryMultiplier, pointCount + 4096),
                                                         previewMode ? 750000 : 3000000));
    std::string pts;
    pts.reserve(static_cast<size_t>(pointCount) * 52u);
    bool first = true;
    const int occupancyBinsPerAxis = preserveOverflow ? 18 : 16;
    std::vector<int> occupancy(static_cast<size_t>(occupancyBinsPerAxis * occupancyBinsPerAxis * occupancyBinsPerAxis), 0);
    int primaryAttempts = 0;
    int primaryAccepted = 0;

    auto sampleUvForAttempt = [&](int attemptIndex, double* outU, double* outV) {
      double u = 0.0;
      double v = 0.0;
      switch (samplingMode) {
        case 1: {
          const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pointCount)))));
          const int gx = attemptIndex % grid;
          const int gy = attemptIndex / grid;
          const double jitterX = unitHash01(static_cast<uint32_t>(attemptIndex * 2 + 1));
          const double jitterY = unitHash01(static_cast<uint32_t>(attemptIndex * 2 + 2));
          u = (static_cast<double>(gx) + jitterX) / static_cast<double>(grid);
          v = (static_cast<double>(gy) + jitterY) / static_cast<double>(grid);
          break;
        }
        case 2:
          u = unitHash01(static_cast<uint32_t>(attemptIndex * 2 + 11));
          v = unitHash01(static_cast<uint32_t>(attemptIndex * 2 + 37));
          break;
        default: {
          const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pointCount)))));
          const int gx = attemptIndex % grid;
          const int gy = attemptIndex / grid;
          u = (static_cast<double>(gx) + 0.5) / static_cast<double>(grid);
          v = (static_cast<double>(gy) + 0.5) / static_cast<double>(grid);
          break;
        }
      }
      *outU = std::clamp(u, 0.0, 1.0);
      *outV = std::clamp(v, 0.0, 1.0);
    };
    auto sampleAcceptedBySelection = [&](int x, int y, float r, float g, float b, uint32_t samplingSeed) {
      (void)x;
      (void)y;
      return neutralRadiusSamplingAcceptsPoint(hueSliceSpec, r, g, b, samplingSeed);
    };

    for (int attemptIndex = 0; attemptIndex < maxPrimaryAttempts && primaryAccepted < pointCount; ++attemptIndex) {
      double u = 0.0;
      double v = 0.0;
      sampleUvForAttempt(attemptIndex, &u, &v);
      ++primaryAttempts;
      const int sx = std::clamp(static_cast<int>(u * static_cast<double>(scaledWidth - 1)), 0, scaledWidth - 1);
      const int sy = std::clamp(static_cast<int>(v * static_cast<double>(scaledHeight - 1)), 0, scaledHeight - 1);
      const int x = std::clamp(static_cast<int>(((static_cast<double>(sx) + 0.5) / static_cast<double>(scaledWidth)) * static_cast<double>(width)), 0, width - 1);
      const int y = std::clamp(static_cast<int>(((static_cast<double>(sy) + 0.5) / static_cast<double>(scaledHeight)) * static_cast<double>(height)), 0, height - 1);
      const float* pix = fetchPixel(x, y);
      if (!pix) continue;
      const WorkshopColor::Vec3f linearSample = transformCloudSampleForPlot(
          time,
          sampledChannelValue(pix[0], preserveOverflow),
          sampledChannelValue(pix[1], preserveOverflow),
          sampledChannelValue(pix[2], preserveOverflow));
      const float r = linearSample.x;
      const float g = linearSample.y;
      const float b = linearSample.z;
      const uint32_t samplingSeed = static_cast<uint32_t>(attemptIndex * 2654435761u) ^
                                    static_cast<uint32_t>(x * 911u + y * 3571u);
      if (!sampleAcceptedBySelection(x, y, r, g, b, samplingSeed)) continue;
      const float xNorm = static_cast<float>((static_cast<double>(x) + 0.5) / static_cast<double>(width));
      const float yNorm = static_cast<float>((static_cast<double>(y) + 0.5) / static_cast<double>(height));
      appendCloudPointSample(&pts, &first, xNorm, yNorm, r, g, b);
      ++occupancy[occupancyBinIndex(r, g, b, preserveOverflow)];
      ++primaryAccepted;
    }

    if (occupancyFill) {
      const int extraPointCount = std::max(2048, pointCount / (previewMode ? 4 : 2));
      const int candidateTarget = std::min(std::max(extraPointCount * 3, 8192), previewMode ? 32768 : 131072);
      const int maxCandidateAttempts = selectionImpossible
                                           ? 0
                                           : (selectionRetryMultiplier <= 1
                                                  ? candidateTarget
                                                  : std::min(std::max(candidateTarget * selectionRetryMultiplier, candidateTarget + 4096),
                                                             previewMode ? 750000 : 3000000));
      std::vector<int> binCandidateRank(occupancy.size(), 0);
      std::vector<OccupancyCandidate> candidates;
      candidates.reserve(static_cast<size_t>(candidateTarget));
      for (int attemptIndex = 0; attemptIndex < maxCandidateAttempts &&
                                static_cast<int>(candidates.size()) < candidateTarget; ++attemptIndex) {
        double u = halton(static_cast<uint32_t>(attemptIndex + 1), 2);
        double v = halton(static_cast<uint32_t>(attemptIndex + 1), 3);
        const int sx = std::clamp(static_cast<int>(u * static_cast<double>(scaledWidth - 1)), 0, scaledWidth - 1);
        const int sy = std::clamp(static_cast<int>(v * static_cast<double>(scaledHeight - 1)), 0, scaledHeight - 1);
        const int x = std::clamp(static_cast<int>(((static_cast<double>(sx) + 0.5) / static_cast<double>(scaledWidth)) * static_cast<double>(width)), 0, width - 1);
        const int y = std::clamp(static_cast<int>(((static_cast<double>(sy) + 0.5) / static_cast<double>(scaledHeight)) * static_cast<double>(height)), 0, height - 1);
        const float* pix = fetchPixel(x, y);
        if (!pix) continue;
        const WorkshopColor::Vec3f linearSample = transformCloudSampleForPlot(
            time,
            sampledChannelValue(pix[0], preserveOverflow),
            sampledChannelValue(pix[1], preserveOverflow),
            sampledChannelValue(pix[2], preserveOverflow));
        const float r = linearSample.x;
        const float g = linearSample.y;
        const float b = linearSample.z;
        const uint32_t samplingSeed = static_cast<uint32_t>((attemptIndex + 1) * 2246822519u) ^
                                      static_cast<uint32_t>(x * 977u + y * 4051u);
        if (!sampleAcceptedBySelection(x, y, r, g, b, samplingSeed)) continue;
        const int bin = occupancyBinIndex(r, g, b, preserveOverflow);
        OccupancyCandidate candidate{};
        candidate.xNorm = static_cast<float>((static_cast<double>(x) + 0.5) / static_cast<double>(width));
        candidate.yNorm = static_cast<float>((static_cast<double>(y) + 0.5) / static_cast<double>(height));
        candidate.r = r;
        candidate.g = g;
        candidate.b = b;
        candidate.normalizedNeutralRadius = normalizedNeutralRadiusForSlice(hueSliceSpec, r, g, b);
        candidate.bin = bin;
        candidate.binRank = binCandidateRank[bin]++;
        candidate.tie = static_cast<uint32_t>(attemptIndex);
        candidates.push_back(candidate);
      }
      std::sort(candidates.begin(), candidates.end(), [&](const OccupancyCandidate& a, const OccupancyCandidate& b) {
        if (occupancy[a.bin] != occupancy[b.bin]) return occupancy[a.bin] < occupancy[b.bin];
        if (hueSliceSpec.neutralRadiusEnabled &&
            std::fabs(a.normalizedNeutralRadius - b.normalizedNeutralRadius) > 1e-6f) {
          return a.normalizedNeutralRadius < b.normalizedNeutralRadius;
        }
        if (a.binRank != b.binRank) return a.binRank < b.binRank;
        return a.tie < b.tie;
      });
      const int appendCount = std::min<int>(extraPointCount, static_cast<int>(candidates.size()));
      pts.reserve(pts.size() + static_cast<size_t>(appendCount) * 52u);
      for (int i = 0; i < appendCount; ++i) {
        const auto& candidate = candidates[static_cast<size_t>(i)];
        appendCloudPointSample(&pts, &first, candidate.xNorm, candidate.yNorm, candidate.r, candidate.g, candidate.b);
      }
    }

    std::ostringstream hash;
    hash << width << 'x' << height << ':' << resolution << ':' << qualityLabelForIndex(qualityIndex)
         << ":sampling=" << samplingModeLabelForIndex(samplingMode)
         << ":occupancyFill=" << (occupancyFill ? 1 : 0)
         << ":scale=" << scaleLabelForIndex(scaleIndex);
    out.paramHash = hash.str();
    out.pointsPayload = pts;
    out.contentHash = fnv1a64(pts);
    out.payload = buildInputCloudJson(pts, out.paramHash, currentCloudSettingsKey(time), resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    cubeViewerDebugLog(std::string("Built whole-image cloud payload: quality=") + out.quality +
                       " sampling=" + samplingModeLabelForIndex(samplingMode) +
                       " occupancyFill=" + (occupancyFill ? "1" : "0") +
                       " hueSlice=" + (hueSliceSpec.enabled ? "1" : "0") +
                       " neutralRadius=" + (hueSliceSpec.neutralRadiusEnabled ? std::to_string(effectiveNeutralRadius) : std::string("off")) +
                       " attempts=" + std::to_string(primaryAttempts) +
                       " accepted=" + std::to_string(primaryAccepted) +
                       " scale=" + scaleLabelForIndex(scaleIndex) +
                       " res=" + std::to_string(out.resolution) +
                       " pointBytes=" + std::to_string(pts.size()) +
                       " hash=" + out.paramHash);
    return out;
  }

  CloudBuildResult buildInputCloudPayloadFromIdentityStrip(Image* src, double time) {
    CloudBuildResult out;
    if (!src) return out;
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    if (width <= 0 || height <= 0) return out;
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const bool preserveOverflow = currentShowOverflow(time);
    const int requestedSize = getIntValue("cubeViewerIdentityOverlaySize", time, 29);
    const int resolution = clampOverlayCubeSize(requestedSize);
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    // Identity-strip sampling should remain limited to the cube band. The ramp has its own
    // dedicated readback path when the user explicitly enables it.
    if (!computeIdentityStripLayout(bounds, resolution, false, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) return out;

    std::string pts;
    pts.reserve(static_cast<size_t>(width) * static_cast<size_t>(std::max(1, cubeY2 - cubeY1)) * 52u);
    bool first = true;
    for (int y = cubeY1; y < cubeY2; ++y) {
      for (int x = bounds.x1; x < bounds.x2; ++x) {
        const float* pix = reinterpret_cast<const float*>(src->getPixelAddress(x, y));
        if (!pix) continue;
        const WorkshopColor::Vec3f linearSample = transformCloudSampleForPlot(
            time,
            sampledChannelValue(pix[0], preserveOverflow),
            sampledChannelValue(pix[1], preserveOverflow),
            sampledChannelValue(pix[2], preserveOverflow));
        const float r = linearSample.x;
        const float g = linearSample.y;
        const float b = linearSample.z;
        if (!first) pts.push_back(';');
        first = false;
        char sample[96];
        const int n = std::snprintf(sample, sizeof(sample), "%.6f %.6f %.6f %.6f %.6f %.6f", r, g, b, r, g, b);
        if (n > 0) pts.append(sample, static_cast<size_t>(n));
      }
    }
    if (pts.empty()) return out;
    std::ostringstream hash;
    hash << width << 'x' << height << ':' << resolution << ':' << qualityLabelForIndex(qualityIndex)
         << ":identity-strip=1";
    out.paramHash = hash.str();
    out.pointsPayload = pts;
    out.contentHash = fnv1a64(pts);
    out.payload = buildInputCloudJson(pts, out.paramHash, currentCloudSettingsKey(time), resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    cubeViewerDebugLog(std::string("Built input cloud from identity strip: res=") + std::to_string(resolution) +
                       " stripHeight=" + std::to_string(stripHeight));
    return out;
  }

  // Stage: convert the drawn strip back into a cube lattice for downstream plotting.
  // We sample the intended cube lattice, not every strip pixel, so the requested size stays meaningful.
  template <typename FetchPixelFn>
  CloudBuildResult buildInputCloudPayloadFromDrawnCubeSamples(
      int width,
      int height,
      double time,
      FetchPixelFn&& fetchPixel) {
    CloudBuildResult out;
    if (width <= 0 || height <= 0) return out;
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const bool preserveOverflow = currentShowOverflow(time);
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", time, 29);
    const int resolution = clampOverlayCubeSize(requestedSize);
    const OfxRectI bounds{0, 0, width, height};
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    // The cube-strip readback is deliberately isolated from the ramp contract.
    // Even if a ramp exists elsewhere in the strip, this concentrated sampler should only
    // reconstruct the cube lattice from the cube band itself.
    if (!computeIdentityStripLayout(bounds, resolution, false, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) return out;
    const int denom = std::max(1, resolution - 1);
    const float cellWidth = identityStripCellWidth(width, resolution);
    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * 3u);
    std::string pts;
    pts.reserve(static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * static_cast<size_t>(resolution) * 52u);
    bool first = true;
    for (int bz = 0; bz < resolution; ++bz) {
      const float layerStart = static_cast<float>(bz) * cellWidth;
      for (int gy = 0; gy < resolution; ++gy) {
        const int y = cubeY1 + std::clamp(static_cast<int>(std::lround(
            (static_cast<double>(gy) / static_cast<double>(denom)) * static_cast<double>(std::max(0, stripHeight - 1)))), 0, std::max(0, stripHeight - 1));
        for (int rx = 0; rx < resolution; ++rx) {
          const float localX = layerStart +
                               (cellWidth <= 1.0f
                                    ? 0.0f
                                    : static_cast<float>(rx) / static_cast<float>(denom) * std::max(0.0f, cellWidth - 1.0f));
          const int x = std::clamp(static_cast<int>(std::lround(localX)), 0, width - 1);
          const float* pix = fetchPixel(x, y);
          if (!pix) continue;
        const WorkshopColor::Vec3f linearSample = transformCloudSampleForPlot(
            time,
            sampledChannelValue(pix[0], preserveOverflow),
            sampledChannelValue(pix[1], preserveOverflow),
            sampledChannelValue(pix[2], preserveOverflow));
        const float r = linearSample.x;
        const float g = linearSample.y;
        const float b = linearSample.z;
          samples.push_back(r);
          samples.push_back(g);
          samples.push_back(b);
          if (!first) pts.push_back(';');
          first = false;
          char sample[96];
          const int n = std::snprintf(sample, sizeof(sample), "%.6f %.6f %.6f %.6f %.6f %.6f", r, g, b, r, g, b);
          if (n > 0) pts.append(sample, static_cast<size_t>(n));
        }
      }
    }
    if (pts.empty()) return out;
    std::ostringstream hash;
    hash << width << 'x' << height << ':' << resolution << ':' << qualityLabelForIndex(qualityIndex)
         << ":drawn-cube=1";
    out.paramHash = hash.str();
    out.pointsPayload = pts;
    out.contentHash = fnv1a64(pts);
    out.payload = buildInputCloudJson(pts, out.paramHash, currentCloudSettingsKey(time), resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    cubeViewerDebugLog(std::string("Built input cloud from drawn cube lattice: res=") + std::to_string(resolution) +
                       " stripHeight=" + std::to_string(stripHeight) +
                       " pointCount=" + std::to_string((pts.empty() ? 0 : (resolution * resolution * resolution))));
    return out;
  }

  template <typename FetchPixelFn>
  CloudBuildResult buildInputCloudPayloadFromDrawnRampSamples(
      int width,
      int height,
      double time,
      FetchPixelFn&& fetchPixel) {
    CloudBuildResult out;
    if (width <= 0 || height <= 0) return out;
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const bool expectRamp = currentReadGrayRamp(time) || currentIdentityOverlayRamp(time);
    if (!expectRamp) return out;
    const bool preserveOverflow = currentShowOverflow(time);
    const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", time, 29);
    const int resolution = clampOverlayCubeSize(requestedSize);
    const OfxRectI bounds{0, 0, width, height};
    int stripHeight = 0;
    int cubeY1 = 0;
    int cubeY2 = 0;
    int rampY1 = 0;
    int rampY2 = 0;
    if (!computeIdentityStripLayout(bounds, resolution, true, &stripHeight, &cubeY1, &cubeY2, &rampY1, &rampY2)) return out;
    if (rampY2 <= rampY1) return out;
    std::string pts;
    const int rampHeight = std::max(0, rampY2 - rampY1);
    // Gray-ramp readback only needs to cover the full horizontal extent and enough vertical repeats
    // to keep the achromatic line visibly thick. Sampling every ramp pixel is unnecessarily heavy
    // because each row encodes the same grayscale gradient.
    // For the ramp, connectedness matters more than aggressive decimation.
    // Keep full horizontal coverage so the achromatic line reads continuous,
    // and control thickness mainly by limiting how many ramp rows we read.
    const int sampleCols = width;
    const int sampleRows = std::min(rampHeight, std::max(4, std::min(8, resolution / 8 + 2)));
    pts.reserve(static_cast<size_t>(sampleCols) * static_cast<size_t>(std::max(1, sampleRows)) * 52u);
    bool first = true;
    int sampleCount = 0;
    const int colDenom = std::max(1, sampleCols - 1);
    const int rowDenom = std::max(1, sampleRows - 1);
    for (int rowIndex = 0; rowIndex < sampleRows; ++rowIndex) {
      const int y = rampY1 + std::clamp(static_cast<int>(std::lround(
          (static_cast<double>(rowIndex) / static_cast<double>(rowDenom)) * static_cast<double>(std::max(0, rampHeight - 1)))),
          0, std::max(0, rampHeight - 1));
      for (int colIndex = 0; colIndex < sampleCols; ++colIndex) {
        const int x = std::clamp(static_cast<int>(std::lround(
            (static_cast<double>(colIndex) / static_cast<double>(colDenom)) * static_cast<double>(std::max(0, width - 1)))),
            0, std::max(0, width - 1));
        const float* pix = fetchPixel(x, y);
        if (!pix) continue;
        const WorkshopColor::Vec3f linearSample = transformCloudSampleForPlot(
            time,
            sampledChannelValue(pix[0], preserveOverflow),
            sampledChannelValue(pix[1], preserveOverflow),
            sampledChannelValue(pix[2], preserveOverflow));
        const float r = linearSample.x;
        const float g = linearSample.y;
        const float b = linearSample.z;
        if (!first) pts.push_back(';');
        first = false;
        char sample[96];
        const int n = std::snprintf(sample, sizeof(sample), "%.6f %.6f %.6f %.6f %.6f %.6f", r, g, b, r, g, b);
        if (n > 0) pts.append(sample, static_cast<size_t>(n));
        ++sampleCount;
      }
    }
    if (pts.empty()) return out;
    std::ostringstream hash;
    hash << width << 'x' << height << ':' << resolution << ':' << qualityLabelForIndex(qualityIndex)
         << ":drawn-ramp=1";
    out.paramHash = hash.str();
    out.pointsPayload = pts;
    out.contentHash = fnv1a64(pts);
    out.payload = buildInputCloudJson(pts, out.paramHash, currentCloudSettingsKey(time), resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    cubeViewerDebugLog(std::string("Built input cloud from drawn gray ramp: samples=") + std::to_string(sampleCount) +
                       " cols=" + std::to_string(sampleCols) +
                       " rows=" + std::to_string(sampleRows) +
                       " rampBand=" + std::to_string(rampY1) + "-" + std::to_string(rampY2));
    return out;
  }

  CloudBuildResult buildInputCloudPayloadFromDrawnCubeImage(Image* src, double time) {
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    return buildInputCloudPayloadFromDrawnCubeSamples(width, height, time, [&](int x, int y) -> const float* {
      return src ? reinterpret_cast<const float*>(src->getPixelAddress(bounds.x1 + x, bounds.y1 + y)) : nullptr;
    });
  }

  CloudBuildResult buildInputCloudPayloadFromDrawnRampImage(Image* src, double time) {
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    return buildInputCloudPayloadFromDrawnRampSamples(width, height, time, [&](int x, int y) -> const float* {
      return src ? reinterpret_cast<const float*>(src->getPixelAddress(bounds.x1 + x, bounds.y1 + y)) : nullptr;
    });
  }

  CloudBuildResult buildInputCloudPayloadFromDrawnCubeBuffer(
      const float* srcBase,
      size_t srcRowBytes,
      int width,
      int height,
      double time) {
    return buildInputCloudPayloadFromDrawnCubeSamples(width, height, time, [&](int x, int y) -> const float* {
      if (!srcBase) return nullptr;
      const char* rowBase = reinterpret_cast<const char*>(srcBase) + static_cast<size_t>(y) * srcRowBytes;
      return reinterpret_cast<const float*>(rowBase) + static_cast<size_t>(x) * 4u;
    });
  }

  CloudBuildResult buildInputCloudPayloadFromDrawnRampBuffer(
      const float* srcBase,
      size_t srcRowBytes,
      int width,
      int height,
      double time) {
    return buildInputCloudPayloadFromDrawnRampSamples(width, height, time, [&](int x, int y) -> const float* {
      if (!srcBase) return nullptr;
      const char* rowBase = reinterpret_cast<const char*>(srcBase) + static_cast<size_t>(y) * srcRowBytes;
      return reinterpret_cast<const float*>(rowBase) + static_cast<size_t>(x) * 4u;
    });
  }

  CloudBuildResult combineInstance1AndImageClouds(
      const CloudBuildResult& identity,
      const CloudBuildResult& image,
      double time) {
    if (!identity.success) return image;
    if (!image.success) return identity;

    std::string identityPoints;
    std::string imagePoints;
    if (!extractJsonStringField(identity.payload, "points", &identityPoints) ||
        !extractJsonStringField(image.payload, "points", &imagePoints)) {
      return image;
    }

    std::string mergedPoints = identityPoints;
    if (!mergedPoints.empty() && !imagePoints.empty()) mergedPoints.push_back(';');
    mergedPoints += imagePoints;

    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    CloudBuildResult out{};
    out.paramHash = image.paramHash + "+instance1=" + std::to_string(identity.resolution);
    out.pointsPayload = mergedPoints;
    out.contentHash = fnv1a64(mergedPoints);
    out.payload = buildInputCloudJson(mergedPoints, out.paramHash, currentCloudSettingsKey(time), image.resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = image.resolution;
    out.sourceWidth = std::max(identity.sourceWidth, image.sourceWidth);
    out.sourceHeight = std::max(identity.sourceHeight, image.sourceHeight);
    out.success = true;
    cubeViewerDebugLog(std::string("Built combined instance-1 cloud: identityRes=") + std::to_string(identity.resolution) +
                       " imageRes=" + std::to_string(image.resolution) +
                       " mergedBytes=" + std::to_string(mergedPoints.size()));
    return out;
  }

  CloudBuildResult buildInstance1StripCloudPayload(Image* src, double time) {
    CloudBuildResult stripCloud{};
    if (!src) return stripCloud;
    if (currentReadIdentityPlot(time)) {
      stripCloud = buildInputCloudPayloadFromDrawnCubeImage(src, time);
    }
    if (currentReadGrayRamp(time)) {
      CloudBuildResult ramp = buildInputCloudPayloadFromDrawnRampImage(src, time);
      stripCloud = stripCloud.success ? combineInstance1AndImageClouds(stripCloud, ramp, time) : ramp;
    }
    if (!stripCloud.success && currentShowIdentityOnly(time)) {
      const OfxRectI bounds = src->getBounds();
      return buildEmptyInstance1Cloud(time, bounds.x2 - bounds.x1, bounds.y2 - bounds.y1, "image");
    }
    return stripCloud;
  }

  CloudBuildResult buildInstance1StripCloudPayloadFromBuffer(
      const float* srcBase,
      size_t srcRowBytes,
      int width,
      int height,
      double time) {
    CloudBuildResult stripCloud{};
    if (currentReadIdentityPlot(time)) {
      stripCloud = buildInputCloudPayloadFromDrawnCubeBuffer(srcBase, srcRowBytes, width, height, time);
    }
    if (currentReadGrayRamp(time)) {
      CloudBuildResult ramp = buildInputCloudPayloadFromDrawnRampBuffer(srcBase, srcRowBytes, width, height, time);
      stripCloud = stripCloud.success ? combineInstance1AndImageClouds(stripCloud, ramp, time) : ramp;
    }
    if (!stripCloud.success && currentShowIdentityOnly(time)) {
      return buildEmptyInstance1Cloud(time, width, height, "buffer");
    }
    return stripCloud;
  }

  CloudBuildResult buildEmptyInstance1Cloud(double time, int width, int height, const std::string& tag) {
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int resolution = qualityResolutionForIndex(qualityIndex);
    CloudBuildResult out{};
    out.paramHash = std::string("instance1-empty:") + tag;
    out.pointsPayload.clear();
    out.contentHash = fnv1a64(std::string{});
    out.payload = buildInputCloudJson("", out.paramHash, currentCloudSettingsKey(time), resolution, qualityIndex);
    out.quality = qualityLabelForIndex(qualityIndex);
    out.resolution = resolution;
    out.sourceWidth = width;
    out.sourceHeight = height;
    out.success = true;
    return out;
  }

  CloudBuildResult buildInputCloudPayloadFromWholeImage(Image* src, double time, bool previewMode) {
    CloudBuildResult out;
    if (!src) return out;
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    return buildInputCloudPayloadFromWholeImageSamples(width, height, time, previewMode, [&](int x, int y) -> const float* {
      return reinterpret_cast<const float*>(src->getPixelAddress(bounds.x1 + x, bounds.y1 + y));
    });
  }

  CloudBuildResult buildInputCloudPayloadFromWholeImageBuffer(
      const float* srcBase,
      size_t srcRowBytes,
      int width,
      int height,
      double time,
      bool previewMode) {
    return buildInputCloudPayloadFromWholeImageSamples(width, height, time, previewMode, [&](int x, int y) -> const float* {
      if (!srcBase) return nullptr;
      const char* rowBase = reinterpret_cast<const char*>(srcBase) + static_cast<size_t>(y) * srcRowBytes;
      return reinterpret_cast<const float*>(rowBase) + static_cast<size_t>(x) * 4u;
    });
  }

  // Stage: sample the current image into the serialized point-cloud transport format.
  // Normal plot mode samples the whole image; drawn-cube mode routes into the strip-specific sampler above.
  CloudBuildResult buildInputCloudPayload(Image* src, double time, bool previewMode) {
    if (currentUseInstance1Requested(time)) {
      CloudBuildResult stripCloud = buildInstance1StripCloudPayload(src, time);
      if (!stripCloud.success) return buildInputCloudPayloadFromWholeImage(src, time, previewMode);
      if (currentShowIdentityOnly(time)) return stripCloud;
      CloudBuildResult image = buildInputCloudPayloadFromWholeImage(src, time, previewMode);
      return combineInstance1AndImageClouds(stripCloud, image, time);
    }
    return buildInputCloudPayloadFromWholeImage(src, time, previewMode);
  }

  // GPU readbacks and CPU images share the same cloud-build rules; this path just swaps pixel access.
  CloudBuildResult buildInputCloudPayloadFromBuffer(
      const float* srcBase,
      size_t srcRowBytes,
      int width,
      int height,
      double time,
      bool previewMode) {
    if (currentUseInstance1Requested(time)) {
      CloudBuildResult stripCloud = buildInstance1StripCloudPayloadFromBuffer(srcBase, srcRowBytes, width, height, time);
      if (!stripCloud.success) return buildInputCloudPayloadFromWholeImageBuffer(srcBase, srcRowBytes, width, height, time, previewMode);
      if (currentShowIdentityOnly(time)) return stripCloud;
      CloudBuildResult image = buildInputCloudPayloadFromWholeImageBuffer(srcBase, srcRowBytes, width, height, time, previewMode);
      return combineInstance1AndImageClouds(stripCloud, image, time);
    }
    return buildInputCloudPayloadFromWholeImageBuffer(srcBase, srcRowBytes, width, height, time, previewMode);
  }

  CloudBuildResult buildInputCloudPayloadFromCudaReadback(
      Image* src,
      const RenderArguments& args,
      bool previewMode) {
#if defined(CHROMASPACE_HAS_CUDA)
    CloudBuildResult out{};
    if (!args.isEnabledCudaRender || args.pCudaStream == nullptr || !src) return out;
    const void* srcRaw = src->getPixelData();
    if (!srcRaw) return out;
    int fullWidth = 0;
    int fullHeight = 0;
    int fullX1 = 0;
    int fullY1 = 0;
    if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return out;
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = fullPackedRowBytes;
    if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return out;
    float* readback = stageSrcPtr();
    if (!readback) return out;
    const size_t fullOffset =
        static_cast<size_t>(fullY1) * srcRowBytes + static_cast<size_t>(fullX1) * 4u * sizeof(float);
    const char* fullSrcBytes = reinterpret_cast<const char*>(srcRaw) + fullOffset;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(args.pCudaStream);
    if (cudaMemcpy2DAsync(readback,
                          fullPackedRowBytes,
                          fullSrcBytes,
                          srcRowBytes,
                          fullPackedRowBytes,
                          static_cast<size_t>(fullHeight),
                          cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess) {
      return out;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return out;
    out = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
    if (out.success) out.backendName = "CPU-readback/CUDA";
    return out;
#else
    (void)src;
    (void)args;
    (void)previewMode;
    return {};
#endif
  }

  CloudBuildResult buildInstance1StripCloudPayloadFromCudaReadback(
      Image* src,
      const RenderArguments& args) {
#if defined(CHROMASPACE_HAS_CUDA)
    CloudBuildResult out{};
    if (!args.isEnabledCudaRender || args.pCudaStream == nullptr || !src) return out;
    const void* srcRaw = src->getPixelData();
    if (!srcRaw) return out;
    int fullWidth = 0;
    int fullHeight = 0;
    int fullX1 = 0;
    int fullY1 = 0;
    if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return out;
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = fullPackedRowBytes;
    if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return out;
    float* readback = stageSrcPtr();
    if (!readback) return out;
    const size_t fullOffset =
        static_cast<size_t>(fullY1) * srcRowBytes + static_cast<size_t>(fullX1) * 4u * sizeof(float);
    const char* fullSrcBytes = reinterpret_cast<const char*>(srcRaw) + fullOffset;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(args.pCudaStream);
    if (cudaMemcpy2DAsync(readback,
                          fullPackedRowBytes,
                          fullSrcBytes,
                          srcRowBytes,
                          fullPackedRowBytes,
                          static_cast<size_t>(fullHeight),
                          cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess) {
      return out;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return out;
    out = buildInstance1StripCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time);
    if (out.success) out.backendName = "CPU-readback/CUDA-strip";
    return out;
#else
    (void)src;
    (void)args;
    return {};
#endif
  }

  CloudBuildResult buildInputCloudPayloadFromOpenCLReadback(
      Image* src,
      const RenderArguments& args,
      bool previewMode) {
#if defined(CHROMASPACE_HAS_OPENCL)
    CloudBuildResult out{};
    if (!args.isEnabledOpenCLRender || args.pOpenCLCmdQ == nullptr || !src) return out;
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(args.pOpenCLCmdQ);
    if (!queue) return out;
    int fullWidth = 0;
    int fullHeight = 0;
    int fullX1 = 0;
    int fullY1 = 0;
    if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return out;
    const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
    if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return out;
    float* readback = stageSrcPtr();
    if (!readback) return out;
    if (src->getPixelData() != nullptr) {
      const size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
      cl_mem srcBuffer = reinterpret_cast<cl_mem>(src->getPixelData());
      const size_t fullBufferOffset =
          static_cast<size_t>(fullY1) * srcRowBytes + static_cast<size_t>(fullX1) * 4u * sizeof(float);
      const size_t fullSrcOrigin[3] = {fullBufferOffset, 0, 0};
      const size_t hostOrigin[3] = {0, 0, 0};
      const size_t fullRegion[3] = {fullPackedRowBytes, static_cast<size_t>(fullHeight), 1};
      if (clEnqueueReadBufferRect(queue, srcBuffer, CL_TRUE, fullSrcOrigin, hostOrigin, fullRegion,
                                  srcRowBytes, 0, fullPackedRowBytes, 0, readback,
                                  0, nullptr, nullptr) != CL_SUCCESS) {
        return out;
      }
    } else if (src->getOpenCLImage() != nullptr) {
      cl_mem srcImage = reinterpret_cast<cl_mem>(src->getOpenCLImage());
      const size_t fullOrigin[3] = {static_cast<size_t>(fullX1), static_cast<size_t>(fullY1), 0};
      const size_t fullRegion[3] = {static_cast<size_t>(fullWidth), static_cast<size_t>(fullHeight), 1};
      if (clEnqueueReadImage(queue, srcImage, CL_TRUE, fullOrigin, fullRegion, fullPackedRowBytes, 0,
                             readback, 0, nullptr, nullptr) != CL_SUCCESS) {
        return out;
      }
    } else {
      return out;
    }
    out = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
    if (out.success) out.backendName = "CPU-readback/OpenCL";
    return out;
#else
    (void)src;
    (void)args;
    (void)previewMode;
    return {};
#endif
  }

  CloudBuildResult buildInputCloudPayloadFromMetalReadback(
      Image* src,
      Image* dst,
      const RenderArguments& args,
      bool previewMode) {
#if defined(__APPLE__)
    CloudBuildResult out{};
    if (!src || !dst || !args.isEnabledMetalRender || args.pMetalCmdQ == nullptr ||
        src->getPixelData() == nullptr || dst->getPixelData() == nullptr) {
      return out;
    }
    int fullWidth = 0;
    int fullHeight = 0;
    int fullX1 = 0;
    int fullY1 = 0;
    if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return out;
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    size_t dstRowBytes = static_cast<size_t>(std::abs(dst->getRowBytes()));
    const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
    if (srcRowBytes == 0) srcRowBytes = fullPackedRowBytes;
    if (dstRowBytes == 0) dstRowBytes = fullPackedRowBytes;
    if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return out;
    float* readback = stageSrcPtr();
    if (!readback) return out;
    if (!ChromaspaceMetal::copyHostBuffersReadback(src->getPixelData(),
                                                   dst->getPixelData(),
                                                   fullWidth,
                                                   fullHeight,
                                                   srcRowBytes,
                                                   dstRowBytes,
                                                   fullX1,
                                                   fullY1,
                                                   args.pMetalCmdQ,
                                                   readback,
                                                   fullPackedRowBytes)) {
      return out;
    }
    out = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
    if (out.success) out.backendName = "CPU-readback/Metal";
    return out;
#else
    (void)src;
    (void)dst;
    (void)args;
    (void)previewMode;
    return {};
#endif
  }

  CloudBuildResult buildViewerCloudPayload(
      Image* src,
      Image* dst,
      const RenderArguments& args,
      bool previewMode) {
    if (!src) return {};
    const OfxRectI bounds = src->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;
    if (width <= 0 || height <= 0) return {};
    const ViewerCloudBuildRequest request = makeViewerCloudBuildRequest(
        src->getPixelData() ? reinterpret_cast<const float*>(src->getPixelData()) : nullptr,
        static_cast<size_t>(std::abs(src->getRowBytes())),
        width,
        height,
        args.time,
        previewMode);
    const bool useInstance1Requested = currentUseInstance1Requested(args.time);
    const bool showIdentityOnly = currentShowIdentityOnly(args.time);
#if defined(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
    if (useInstance1Requested &&
        args.isEnabledCudaRender && args.pCudaStream != nullptr && src->getPixelData() != nullptr) {
      std::vector<ViewerCloudSample> stripSamples;
      std::string stripParamHash;
      int stripResolution = request.resolution;
      std::string stripReason;
      const bool stripBuilt = buildInstance1StripCloudCuda(src, args, &stripSamples, &stripParamHash, &stripResolution, &stripReason);
      if (showIdentityOnly) {
        if (stripBuilt) {
          cubeViewerDebugLog("Viewer cloud backend selected: CUDA-strip");
          return finalizeCloudBuildFromSamples(stripSamples,
                                               stripParamHash,
                                               request.settingsKey,
                                               stripResolution,
                                               request.qualityIndex,
                                               width,
                                               height,
                                               "CUDA-strip",
                                               static_cast<int>(stripSamples.size()),
                                               static_cast<int>(stripSamples.size()));
        }
      } else {
        const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
        size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
        if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
        const float* fullSrcBase = reinterpret_cast<const float*>(
            reinterpret_cast<const char*>(src->getPixelData()) +
            static_cast<size_t>(bounds.y1) * srcRowBytes +
            static_cast<size_t>(bounds.x1) * 4u * sizeof(float));
        ViewerCloudBuildRequest cudaRequest = request;
        cudaRequest.srcBase = fullSrcBase;
        cudaRequest.srcRowBytes = srcRowBytes;
        ViewerCloudSamplesBuildResult cudaBuilt{};
        std::string reason;
        if (buildWholeImageCloudCuda(cudaRequest,
                                     currentHueSectorSliceSpec(args.time),
                                     reinterpret_cast<cudaStream_t>(args.pCudaStream),
                                     &cudaBuilt,
                                     &reason)) {
          if (stripBuilt) {
            std::vector<ViewerCloudSample> mergedSamples;
            mergedSamples.reserve(stripSamples.size() + cudaBuilt.samples.size());
            mergedSamples.insert(mergedSamples.end(), stripSamples.begin(), stripSamples.end());
            mergedSamples.insert(mergedSamples.end(), cudaBuilt.samples.begin(), cudaBuilt.samples.end());
            cubeViewerDebugLog("Viewer cloud backend selected: CUDA-kernel + CUDA-strip");
            return finalizeCloudBuildFromSamples(mergedSamples,
                                                 cudaBuilt.paramHash + "+instance1=" + std::to_string(stripResolution),
                                                 request.settingsKey,
                                                 cudaBuilt.resolution,
                                                 request.qualityIndex,
                                                 cudaBuilt.sourceWidth,
                                                 cudaBuilt.sourceHeight,
                                                 "CUDA+CUDA-strip",
                                                 cudaBuilt.primaryAttempts,
                                                 cudaBuilt.primaryAccepted + static_cast<int>(stripSamples.size()));
          }
          cubeViewerDebugLog(std::string("Viewer identity strip CUDA path fell back: ") + stripReason);
          return finalizeCloudBuildFromSamples(cudaBuilt.samples,
                                               cudaBuilt.paramHash,
                                               request.settingsKey,
                                               cudaBuilt.resolution,
                                               request.qualityIndex,
                                               cudaBuilt.sourceWidth,
                                               cudaBuilt.sourceHeight,
                                               cudaBuilt.backendName,
                                               cudaBuilt.primaryAttempts,
                                               cudaBuilt.primaryAccepted);
        }
        cubeViewerDebugLog(std::string("Viewer cloud CUDA kernel path fell back in instance1 mode: ") + reason);
      }
      if (!stripBuilt && !stripReason.empty()) {
        cubeViewerDebugLog(std::string("Viewer identity strip CUDA path unavailable: ") + stripReason);
      }
    }
#endif
#if defined(CHROMASPACE_PLUGIN_HAS_CUDA_KERNELS)
    if (!useInstance1Requested &&
        args.isEnabledCudaRender && args.pCudaStream != nullptr && src->getPixelData() != nullptr) {
      const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
      size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
      if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
      const float* fullSrcBase = reinterpret_cast<const float*>(
          reinterpret_cast<const char*>(src->getPixelData()) +
          static_cast<size_t>(bounds.y1) * srcRowBytes +
          static_cast<size_t>(bounds.x1) * 4u * sizeof(float));
      ViewerCloudBuildRequest cudaRequest = request;
      cudaRequest.srcBase = fullSrcBase;
      cudaRequest.srcRowBytes = srcRowBytes;
      ViewerCloudSamplesBuildResult cudaBuilt{};
      std::string reason;
      if (buildWholeImageCloudCuda(cudaRequest,
                                   currentHueSectorSliceSpec(args.time),
                                   reinterpret_cast<cudaStream_t>(args.pCudaStream),
                                   &cudaBuilt,
                                   &reason)) {
        return finalizeCloudBuildFromSamples(cudaBuilt.samples,
                                             cudaBuilt.paramHash,
                                             request.settingsKey,
                                             cudaBuilt.resolution,
                                             request.qualityIndex,
                                             cudaBuilt.sourceWidth,
                                             cudaBuilt.sourceHeight,
                                             cudaBuilt.backendName,
                                             cudaBuilt.primaryAttempts,
                                             cudaBuilt.primaryAccepted);
      }
      cubeViewerDebugLog(std::string("Viewer cloud CUDA kernel path fell back: ") + reason);
    }
#endif
#if defined(__APPLE__)
    if (useInstance1Requested &&
        args.isEnabledMetalRender && args.pMetalCmdQ != nullptr && src->getPixelData() != nullptr) {
      const bool readCube = currentReadIdentityPlot(args.time);
      const bool readRamp = currentReadGrayRamp(args.time);
      int stripResolution = request.resolution;
      std::string stripReason;
      if (showIdentityOnly) {
        std::vector<ViewerCloudSample> stripSamples;
        std::string stripParamHash;
        const bool stripBuilt =
            buildInstance1StripCloudMetal(src, args, &stripSamples, &stripParamHash, &stripResolution, &stripReason);
        if (stripBuilt) {
          cubeViewerDebugLog("Viewer cloud backend selected: Metal-strip");
          return finalizeCloudBuildFromSamples(stripSamples,
                                               stripParamHash,
                                               request.settingsKey,
                                               stripResolution,
                                               request.qualityIndex,
                                               width,
                                               height,
                                               "Metal-strip",
                                               static_cast<int>(stripSamples.size()),
                                               static_cast<int>(stripSamples.size()));
        }
      } else {
        ViewerCloudSamplesBuildResult metalBuilt{};
        std::string reason;
        const int requestedSize = getIntValue("cubeViewerSampleDrawnCubeSize", args.time, 29);
        const int cachedStripResolution = clampOverlayCubeSize(requestedSize);
        const std::string stripCacheKey =
            currentIdentityStripCacheKey(args.time, width, height, bounds.x1, bounds.y1, cachedStripResolution, readCube, readRamp, "metal-strip");
        std::vector<ViewerCloudSample> cachedStripSamples;
        int cachedResolution = cachedStripResolution;
        if (tryGetCachedIdentityStripCloud(stripCacheKey, &cachedStripSamples, nullptr, &cachedResolution) &&
            buildWholeImageCloudMetal(request, currentHueSectorSliceSpec(args.time), src, args, &metalBuilt, &reason)) {
          std::vector<ViewerCloudSample> mergedSamples;
          mergedSamples.reserve(cachedStripSamples.size() + metalBuilt.samples.size());
          mergedSamples.insert(mergedSamples.end(), cachedStripSamples.begin(), cachedStripSamples.end());
          mergedSamples.insert(mergedSamples.end(), metalBuilt.samples.begin(), metalBuilt.samples.end());
          cubeViewerDebugLog("Viewer cloud backend selected: Metal-kernel + cached Metal-strip");
          return finalizeCloudBuildFromSamples(mergedSamples,
                                               metalBuilt.paramHash + "+instance1=" + std::to_string(cachedResolution),
                                               request.settingsKey,
                                               metalBuilt.resolution,
                                               request.qualityIndex,
                                               metalBuilt.sourceWidth,
                                               metalBuilt.sourceHeight,
                                               "Metal+Cached-strip",
                                               metalBuilt.primaryAttempts,
                                               metalBuilt.primaryAccepted + static_cast<int>(cachedStripSamples.size()));
        }
        if (buildWholeImageAndInstance1CloudMetal(
                request, currentHueSectorSliceSpec(args.time), src, args, &metalBuilt, &stripResolution, &reason)) {
          if (!metalBuilt.identityStripSamples.empty() && !metalBuilt.identityStripParamHash.empty()) {
            storeCachedIdentityStripCloud(stripCacheKey,
                                          metalBuilt.identityStripSamples,
                                          metalBuilt.identityStripParamHash,
                                          metalBuilt.identityStripResolution);
          }
          cubeViewerDebugLog("Viewer cloud backend selected: Metal-kernel + Metal-strip");
          return finalizeCloudBuildFromSamples(metalBuilt.samples,
                                               metalBuilt.paramHash,
                                               request.settingsKey,
                                               metalBuilt.resolution,
                                               request.qualityIndex,
                                               metalBuilt.sourceWidth,
                                               metalBuilt.sourceHeight,
                                               metalBuilt.backendName,
                                               metalBuilt.primaryAttempts,
                                               metalBuilt.primaryAccepted);
        }
        cubeViewerDebugLog(std::string("Viewer combined Metal kernel path fell back in instance1 mode: ") + reason);
        if (buildWholeImageCloudMetal(request, currentHueSectorSliceSpec(args.time), src, args, &metalBuilt, &reason)) {
          cubeViewerDebugLog("Viewer cloud backend selected: Metal-kernel");
          return finalizeCloudBuildFromSamples(metalBuilt.samples,
                                               metalBuilt.paramHash,
                                               request.settingsKey,
                                               metalBuilt.resolution,
                                               request.qualityIndex,
                                               metalBuilt.sourceWidth,
                                               metalBuilt.sourceHeight,
                                               metalBuilt.backendName,
                                               metalBuilt.primaryAttempts,
                                               metalBuilt.primaryAccepted);
        }
        cubeViewerDebugLog(std::string("Viewer cloud Metal kernel path fell back in instance1 mode: ") + reason);
      }
      if (showIdentityOnly && !stripReason.empty()) {
        cubeViewerDebugLog(std::string("Viewer identity strip Metal path unavailable: ") + stripReason);
      }
    }
    if (!useInstance1Requested &&
        args.isEnabledMetalRender && args.pMetalCmdQ != nullptr && src->getPixelData() != nullptr) {
      ViewerCloudSamplesBuildResult metalBuilt{};
      std::string reason;
      if (buildWholeImageCloudMetal(request, currentHueSectorSliceSpec(args.time), src, args, &metalBuilt, &reason)) {
        cubeViewerDebugLog("Viewer cloud backend selected: Metal-kernel");
        return finalizeCloudBuildFromSamples(metalBuilt.samples,
                                             metalBuilt.paramHash,
                                             request.settingsKey,
                                             metalBuilt.resolution,
                                             request.qualityIndex,
                                             metalBuilt.sourceWidth,
                                             metalBuilt.sourceHeight,
                                             metalBuilt.backendName,
                                             metalBuilt.primaryAttempts,
                                             metalBuilt.primaryAccepted);
      }
      cubeViewerDebugLog(std::string("Viewer cloud Metal kernel path fell back: ") + reason);
    }
#endif
    if (args.isEnabledCudaRender && args.pCudaStream != nullptr) {
      CloudBuildResult cudaReadback = buildInputCloudPayloadFromCudaReadback(src, args, previewMode);
      if (cudaReadback.success) {
        cubeViewerDebugLog("Viewer cloud backend selected: CPU-readback/CUDA");
        return cudaReadback;
      }
    }
    if (args.isEnabledOpenCLRender && args.pOpenCLCmdQ != nullptr) {
      CloudBuildResult openclReadback = buildInputCloudPayloadFromOpenCLReadback(src, args, previewMode);
      if (openclReadback.success) {
        cubeViewerDebugLog("Viewer cloud backend selected: CPU-readback/OpenCL");
        return openclReadback;
      }
    }
    if (args.isEnabledMetalRender && args.pMetalCmdQ != nullptr) {
      CloudBuildResult metalReadback = buildInputCloudPayloadFromMetalReadback(src, dst, args, previewMode);
      if (metalReadback.success) {
        cubeViewerDebugLog("Viewer cloud backend selected: CPU-readback/Metal");
        return metalReadback;
      }
    }
    CloudBuildResult cpuBuilt = buildInputCloudPayload(src, args.time, previewMode);
    if (cpuBuilt.success) {
      cpuBuilt.backendName = "CPU";
      cubeViewerDebugLog("Viewer cloud backend selected: CPU");
    }
    return cpuBuilt;
  }

  bool fullSourceDimensions(Image* src, int* width, int* height, int* x1, int* y1) {
    if (!src || !width || !height || !x1 || !y1) return false;
    const OfxRectI bounds = src->getBounds();
    const int w = bounds.x2 - bounds.x1;
    const int h = bounds.y2 - bounds.y1;
    if (w <= 0 || h <= 0) return false;
    *width = w;
    *height = h;
    *x1 = bounds.x1;
    *y1 = bounds.y1;
    return true;
  }

#if defined(CHROMASPACE_HAS_CUDA)
  bool renderCUDAHostBuffers(
      Image* src,
      Image* dst,
      const RenderArguments& args,
      bool needCloud,
      bool previewMode,
      CloudBuildResult* built,
      const OverlayStripData* overlay) {
    if (!args.isEnabledCudaRender || args.pCudaStream == nullptr || !src || !dst) return false;
    const void* srcRaw = src->getPixelData();
    void* dstRaw = dst->getPixelData();
    if (!srcRaw || !dstRaw) return false;

    const int renderWidth = args.renderWindow.x2 - args.renderWindow.x1;
    const int renderHeight = args.renderWindow.y2 - args.renderWindow.y1;
    const size_t packedRowBytes = static_cast<size_t>(renderWidth) * 4u * sizeof(float);
    size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    size_t dstRowBytes = static_cast<size_t>(std::abs(dst->getRowBytes()));
    if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
    if (dstRowBytes == 0) dstRowBytes = packedRowBytes;
    if (srcRowBytes < packedRowBytes || dstRowBytes < packedRowBytes) return false;

    const size_t offset =
        static_cast<size_t>(args.renderWindow.y1) * srcRowBytes + static_cast<size_t>(args.renderWindow.x1) * 4u * sizeof(float);
    const size_t dstOffset =
        static_cast<size_t>(args.renderWindow.y1) * dstRowBytes + static_cast<size_t>(args.renderWindow.x1) * 4u * sizeof(float);
    const char* srcBytes = reinterpret_cast<const char*>(srcRaw) + offset;
    char* dstBytes = reinterpret_cast<char*>(dstRaw) + dstOffset;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(args.pCudaStream);

    if (cudaMemcpy2DAsync(dstBytes, dstRowBytes, srcBytes, srcRowBytes, packedRowBytes, static_cast<size_t>(renderHeight),
                          cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return false;
    }

    if (overlay != nullptr && overlay->width > 0 && overlay->height > 0 && !overlay->pixels.empty()) {
      const size_t overlayPackedRowBytes = static_cast<size_t>(overlay->width) * 4u * sizeof(float);
      const size_t overlayDstOffset =
          static_cast<size_t>(overlay->y1) * dstRowBytes + static_cast<size_t>(overlay->x1) * 4u * sizeof(float);
      char* overlayDstBytes = reinterpret_cast<char*>(dstRaw) + overlayDstOffset;
      if (cudaMemcpy2DAsync(overlayDstBytes,
                            dstRowBytes,
                            overlay->pixels.data(),
                            overlayPackedRowBytes,
                            overlayPackedRowBytes,
                            static_cast<size_t>(overlay->height),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
        return false;
      }
    }

    if (!needCloud) return true;

    // Stage: GPU render path still stages a full-source readback when the viewer needs a cloud payload.
    int fullWidth = 0;
    int fullHeight = 0;
    int fullX1 = 0;
    int fullY1 = 0;
    if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return false;
    const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
    const size_t fullOffset =
        static_cast<size_t>(fullY1) * srcRowBytes + static_cast<size_t>(fullX1) * 4u * sizeof(float);
    const char* fullSrcBytes = reinterpret_cast<const char*>(srcRaw) + fullOffset;
    if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return false;
    float* readback = stageSrcPtr();
    if (!readback) return false;
    if (cudaMemcpy2DAsync(readback,
                          fullPackedRowBytes,
                          fullSrcBytes,
                          srcRowBytes,
                          fullPackedRowBytes,
                          static_cast<size_t>(fullHeight),
                          cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess) {
      return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
    if (built != nullptr) {
      *built = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
    }
    return true;
  }
#endif

#if defined(CHROMASPACE_HAS_OPENCL)
  bool renderOpenCLHostBuffers(
      Image* src,
      Image* dst,
      const RenderArguments& args,
      bool needCloud,
      bool previewMode,
      CloudBuildResult* built,
      const OverlayStripData* overlay) {
    if (!args.isEnabledOpenCLRender || args.pOpenCLCmdQ == nullptr || !src || !dst) return false;
    const int renderWidth = args.renderWindow.x2 - args.renderWindow.x1;
    const int renderHeight = args.renderWindow.y2 - args.renderWindow.y1;
    const size_t packedRowBytes = static_cast<size_t>(renderWidth) * 4u * sizeof(float);
    const size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
    const size_t dstRowBytes = static_cast<size_t>(std::abs(dst->getRowBytes()));
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(args.pOpenCLCmdQ);
    if (queue == nullptr) return false;

    const size_t bufferOffset =
        static_cast<size_t>(args.renderWindow.y1) * srcRowBytes + static_cast<size_t>(args.renderWindow.x1) * 4u * sizeof(float);
    const size_t dstBufferOffset =
        static_cast<size_t>(args.renderWindow.y1) * dstRowBytes + static_cast<size_t>(args.renderWindow.x1) * 4u * sizeof(float);

    if (src->getPixelData() != nullptr && dst->getPixelData() != nullptr && srcRowBytes >= packedRowBytes && dstRowBytes >= packedRowBytes) {
      cl_mem srcBuffer = reinterpret_cast<cl_mem>(src->getPixelData());
      cl_mem dstBuffer = reinterpret_cast<cl_mem>(dst->getPixelData());
      const size_t srcOrigin[3] = {bufferOffset, 0, 0};
      const size_t dstOrigin[3] = {dstBufferOffset, 0, 0};
      const size_t region[3] = {packedRowBytes, static_cast<size_t>(renderHeight), 1};
      if (clEnqueueCopyBufferRect(queue, srcBuffer, dstBuffer, srcOrigin, dstOrigin, region, srcRowBytes, 0, dstRowBytes, 0, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
      if (overlay != nullptr && overlay->width > 0 && overlay->height > 0 && !overlay->pixels.empty()) {
        const size_t overlayPackedRowBytes = static_cast<size_t>(overlay->width) * 4u * sizeof(float);
        const size_t hostOrigin[3] = {0, 0, 0};
        const size_t overlayDstOrigin[3] = {
            static_cast<size_t>(overlay->y1) * dstRowBytes + static_cast<size_t>(overlay->x1) * 4u * sizeof(float), 0, 0};
        const size_t overlayRegion[3] = {overlayPackedRowBytes, static_cast<size_t>(overlay->height), 1};
        if (clEnqueueWriteBufferRect(queue, dstBuffer, CL_TRUE, overlayDstOrigin, hostOrigin, overlayRegion, dstRowBytes, 0,
                                     overlayPackedRowBytes, 0, overlay->pixels.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
          return false;
        }
      }
      if (!needCloud) return true;
      int fullWidth = 0;
      int fullHeight = 0;
      int fullX1 = 0;
      int fullY1 = 0;
      if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return false;
      const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
      const size_t fullBufferOffset =
          static_cast<size_t>(fullY1) * srcRowBytes + static_cast<size_t>(fullX1) * 4u * sizeof(float);
      const size_t fullSrcOrigin[3] = {fullBufferOffset, 0, 0};
      const size_t fullRegion[3] = {fullPackedRowBytes, static_cast<size_t>(fullHeight), 1};
      if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return false;
      float* readback = stageSrcPtr();
      if (!readback) return false;
      const size_t hostOrigin[3] = {0, 0, 0};
      if (clEnqueueReadBufferRect(queue, srcBuffer, CL_TRUE, fullSrcOrigin, hostOrigin, fullRegion, srcRowBytes, 0, fullPackedRowBytes, 0, readback, 0,
                                  nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
      if (built != nullptr) {
        *built = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
      }
      return true;
    }

    if (src->getOpenCLImage() != nullptr && dst->getOpenCLImage() != nullptr) {
      cl_mem srcImage = reinterpret_cast<cl_mem>(src->getOpenCLImage());
      cl_mem dstImage = reinterpret_cast<cl_mem>(dst->getOpenCLImage());
      const size_t origin[3] = {static_cast<size_t>(args.renderWindow.x1), static_cast<size_t>(args.renderWindow.y1), 0};
      const size_t dstOrigin[3] = {static_cast<size_t>(args.renderWindow.x1), static_cast<size_t>(args.renderWindow.y1), 0};
      const size_t region[3] = {static_cast<size_t>(renderWidth), static_cast<size_t>(renderHeight), 1};
      if (clEnqueueCopyImage(queue, srcImage, dstImage, origin, dstOrigin, region, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
      if (overlay != nullptr && overlay->width > 0 && overlay->height > 0 && !overlay->pixels.empty()) {
        const size_t overlayPackedRowBytes = static_cast<size_t>(overlay->width) * 4u * sizeof(float);
        const size_t overlayOrigin[3] = {static_cast<size_t>(overlay->x1), static_cast<size_t>(overlay->y1), 0};
        const size_t overlayRegion[3] = {static_cast<size_t>(overlay->width), static_cast<size_t>(overlay->height), 1};
        if (clEnqueueWriteImage(queue, dstImage, CL_TRUE, overlayOrigin, overlayRegion, overlayPackedRowBytes, 0,
                                overlay->pixels.data(), 0, nullptr, nullptr) != CL_SUCCESS) {
          return false;
        }
      }
      if (!needCloud) return true;
      int fullWidth = 0;
      int fullHeight = 0;
      int fullX1 = 0;
      int fullY1 = 0;
      if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return false;
      const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
      const size_t fullOrigin[3] = {static_cast<size_t>(fullX1), static_cast<size_t>(fullY1), 0};
      const size_t fullRegion[3] = {static_cast<size_t>(fullWidth), static_cast<size_t>(fullHeight), 1};
      if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return false;
      float* readback = stageSrcPtr();
      if (!readback) return false;
      if (clEnqueueReadImage(queue, srcImage, CL_TRUE, fullOrigin, fullRegion, fullPackedRowBytes, 0, readback, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
      if (built != nullptr) {
        *built = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
      }
      return true;
    }

    return false;
  }
#endif

  bool tryRenderGpuBackends(
      Image* src,
      Image* dst,
      const RenderArguments& args,
      bool needCloud,
      bool previewMode,
      CloudBuildResult* built,
      const OverlayStripData* overlay) {
#if defined(CHROMASPACE_HAS_CUDA)
    if (renderCUDAHostBuffers(src, dst, args, needCloud, previewMode, built, overlay)) return true;
#endif
#if defined(__APPLE__)
    if (src && dst && args.isEnabledMetalRender && args.pMetalCmdQ != nullptr && src->getPixelData() != nullptr &&
        dst->getPixelData() != nullptr) {
      const int renderWidth = args.renderWindow.x2 - args.renderWindow.x1;
      const int renderHeight = args.renderWindow.y2 - args.renderWindow.y1;
      const size_t packedRowBytes = static_cast<size_t>(renderWidth) * 4u * sizeof(float);
      size_t srcRowBytes = static_cast<size_t>(std::abs(src->getRowBytes()));
      size_t dstRowBytes = static_cast<size_t>(std::abs(dst->getRowBytes()));
      if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
      if (dstRowBytes == 0) dstRowBytes = packedRowBytes;
      if (!needCloud) {
        if (ChromaspaceMetal::copyHostBuffers(src->getPixelData(),
                                                 dst->getPixelData(),
                                                 renderWidth,
                                                 renderHeight,
                                                 srcRowBytes,
                                                 dstRowBytes,
                                                 args.renderWindow.x1,
                                                 args.renderWindow.y1,
                                                 args.pMetalCmdQ,
                                                 overlay != nullptr ? overlay->pixels.data() : nullptr,
                                                 overlay != nullptr ? overlay->x1 : 0,
                                                 overlay != nullptr ? overlay->y1 : 0,
                                                 overlay != nullptr ? overlay->width : 0,
                                                 overlay != nullptr ? overlay->height : 0)) {
          return true;
        }
      } else {
        int fullWidth = 0;
        int fullHeight = 0;
        int fullX1 = 0;
        int fullY1 = 0;
        if (!fullSourceDimensions(src, &fullWidth, &fullHeight, &fullX1, &fullY1)) return false;
        const size_t fullPackedRowBytes = static_cast<size_t>(fullWidth) * 4u * sizeof(float);
        if (!ensureStageBuffer(static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight))) return false;
        float* readback = stageSrcPtr();
        if (!readback) return false;
        if (ChromaspaceMetal::copyHostBuffersReadback(src->getPixelData(),
                                                         dst->getPixelData(),
                                                         fullWidth,
                                                         fullHeight,
                                                         srcRowBytes,
                                                         dstRowBytes,
                                                         fullX1,
                                                         fullY1,
                                                         args.pMetalCmdQ,
                                                         readback,
                                                         fullPackedRowBytes)) {
          if (built != nullptr) {
            *built = buildInputCloudPayloadFromBuffer(readback, fullPackedRowBytes, fullWidth, fullHeight, args.time, previewMode);
          }
          return true;
        }
      }
    }
#endif
#if defined(CHROMASPACE_HAS_OPENCL)
    if (renderOpenCLHostBuffers(src, dst, args, needCloud, previewMode, built, overlay)) return true;
#endif
    return false;
  }

  bool shouldUseInteractivePreview(std::chrono::steady_clock::time_point now) {
    const ViewerUpdateMode updateMode = static_cast<ViewerUpdateMode>(std::clamp(cubeViewerUpdateMode_, 0, 2));
    if (updateMode == ViewerUpdateMode::Fluid) {
      previewModeUntil_ = std::chrono::steady_clock::time_point{};
      playbackRenderBurstCount_ = 0;
      return false;
    }
    if (updateMode == ViewerUpdateMode::Scheduled) {
      previewModeUntil_ = now + std::chrono::milliseconds(220);
      playbackRenderBurstCount_ = 0;
      return true;
    }
    constexpr auto kPlaybackRenderGap = std::chrono::milliseconds(60);
    constexpr auto kPlaybackGapReset = std::chrono::milliseconds(125);
    constexpr auto kPreviewHold = std::chrono::milliseconds(95);
    constexpr auto kTransportPressureWindow = std::chrono::milliseconds(180);
    constexpr int kBurstThreshold = 4;
    const int64_t lastTransportActivityMs = lastViewerTransportActivityMs_.load(std::memory_order_relaxed);
    const bool recentTransportActivity =
        lastTransportActivityMs > 0 &&
        (monotonicNowMs() - lastTransportActivityMs) <= kTransportPressureWindow.count();
    const bool cloudBackpressure = deferredLatestCloudRefresh_.load(std::memory_order_relaxed) ||
                                   cloudQueuedOrInFlight_.load(std::memory_order_relaxed) ||
                                   recentTransportActivity;
    if (lastRenderSeenAt_ != std::chrono::steady_clock::time_point{}) {
      const auto renderGap = now - lastRenderSeenAt_;
      if (cloudBackpressure && renderGap <= kPlaybackRenderGap) {
        playbackRenderBurstCount_ = std::min(playbackRenderBurstCount_ + 1, kBurstThreshold);
      } else if (!cloudBackpressure || renderGap >= kPlaybackGapReset) {
        playbackRenderBurstCount_ = 0;
      } else if (playbackRenderBurstCount_ > 0) {
        --playbackRenderBurstCount_;
      }
      if (playbackRenderBurstCount_ >= kBurstThreshold) {
        previewModeUntil_ = now + kPreviewHold;
      }
    }
    if (!cloudBackpressure && playbackRenderBurstCount_ == 0) {
      previewModeUntil_ = std::chrono::steady_clock::time_point{};
      return false;
    }
    return previewModeUntil_ != std::chrono::steady_clock::time_point{} && now < previewModeUntil_;
  }

  int steadyStateCloudIntervalMs(bool previewMode) const {
    if (previewMode) {
      switch (cubeViewerQuality_) {
        case 0: return 80;
        case 1: return 120;
        default: return 170;
      }
    }
    switch (cubeViewerQuality_) {
      case 0: return 20;
      case 1: return 32;
      default: return 48;
    }
  }

  bool shouldEmitSteadyStateCloud(double time, const std::string& sourceId, const std::string& settingsKey, bool previewMode) {
    if (!drivesSharedViewer() || !cubeViewerLive_) return false;
    if (cubeViewerInputCloudRefreshPending_) return false;
    if (cloudQueuedOrInFlight_.load(std::memory_order_relaxed)) return false;
    if (deferredLatestCloudRefresh_.load(std::memory_order_relaxed)) return true;
    if (!settingsKey.empty() && settingsKey != lastCloudSettingsKey_) return true;
    if (!sourceId.empty() && sourceId != lastCloudSourceId_) return true;
    const auto now = std::chrono::steady_clock::now();
    if (lastCloudBuiltAt_ == std::chrono::steady_clock::time_point{}) return true;
    const auto elapsedMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - lastCloudBuiltAt_).count();
    if (currentUseInstance1(time)) {
      // The two-instance workflow is intentionally downstream-facing: graphs such as layer mixers can
      // change the incoming pixels without providing a host image identifier change that we can trust.
      // Keep this mode on a more eager steady-state refresh policy instead of waiting for a weaker sourceId signal.
      const ViewerUpdateMode updateMode = static_cast<ViewerUpdateMode>(std::clamp(cubeViewerUpdateMode_, 0, 2));
      if (updateMode == ViewerUpdateMode::Fluid) return elapsedMs >= 16;
      if (updateMode == ViewerUpdateMode::Scheduled) return elapsedMs >= 170;
      return elapsedMs >= (previewMode ? 120 : 16);
    }
    return elapsedMs >= steadyStateCloudIntervalMs(previewMode);
  }

  int64_t monotonicNowMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void markViewerTransportActivity() {
    lastViewerTransportActivityMs_.store(monotonicNowMs(), std::memory_order_relaxed);
  }

  void startIoWorker() {
    if (ioThread_.joinable()) return;
    ioStop_ = false;
    ioThread_ = std::thread([this]() { ioWorkerLoop(); });
  }

  void stopIoWorker() {
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      ioStop_ = true;
    }
    ioCv_.notify_all();
    if (ioThread_.joinable()) ioThread_.join();
  }

  void startStatusThread() {
    if (statusThread_.joinable()) return;
    statusStop_ = false;
    statusThread_ = std::thread([this]() { statusLoop(); });
  }

  void stopStatusThread() {
    statusStop_ = true;
    if (statusThread_.joinable()) statusThread_.join();
  }

  void releaseViewerRuntimeResources() {
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      pendingParams_ = PendingMessage{};
      pendingCloud_ = PendingMessage{};
      cloudQueuedOrInFlight_.store(false, std::memory_order_relaxed);
    }
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    playbackRenderBurstCount_ = 0;
    lastViewerTransportActivityMs_.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(stageMutex_);
      stageSrc_.clear();
      stageSrc_.shrink_to_fit();
    }
  }

  void markViewerInactive(const std::string& reason) {
    gSharedCubeViewerRequestCount.store(0, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(gSharedCubeViewerSenderMutex);
      gSharedCubeViewerActiveSenderId.clear();
      gSharedCubeViewerActiveSourceId.clear();
    }
    gSharedCubeViewerActiveRenderMs.store(0, std::memory_order_relaxed);
    cubeViewerRequested_ = false;
    cubeViewerConnected_ = false;
    cubeViewerWindowUsable_ = false;
    cubeViewerInputCloudRefreshPending_ = false;
    lastCloudBuiltAt_ = std::chrono::steady_clock::time_point{};
    lastHeartbeatAt_ = std::chrono::steady_clock::time_point{};
    lastCloudSourceId_.clear();
    lastCloudSettingsKey_.clear();
    releaseViewerRuntimeResources();
    setStatusLabel("Disconnected");
    cubeViewerDebugLog(reason);
  }

  void statusLoop() {
    int failedProbeCount = 0;
    while (!statusStop_) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (statusStop_) break;
      if (!cubeViewerRequested_) {
        failedProbeCount = 0;
        continue;
      }

      ViewerProbeResult probe = probeViewer();
      if (!probe.ok) {
        const bool hadViewerSession = cubeViewerConnected_ || cubeViewerWindowUsable_ ||
                                      lastHeartbeatAt_ != std::chrono::steady_clock::time_point{};
        const auto now = std::chrono::steady_clock::now();
        const auto msSinceHeartbeat = lastHeartbeatAt_ == std::chrono::steady_clock::time_point{}
                                          ? std::chrono::milliseconds::max().count()
                                          : std::chrono::duration_cast<std::chrono::milliseconds>(now - lastHeartbeatAt_).count();
        const int64_t lastTransportActivityMs = lastViewerTransportActivityMs_.load(std::memory_order_relaxed);
        const int64_t msSinceTransportActivity =
            lastTransportActivityMs > 0 ? (monotonicNowMs() - lastTransportActivityMs)
                                        : std::numeric_limits<int64_t>::max();
        const bool transportBusy = cloudQueuedOrInFlight_.load(std::memory_order_relaxed) ||
                                   msSinceTransportActivity < 3000;
        if (hadViewerSession && transportBusy) {
          failedProbeCount = 0;
          setStatusLabel(cubeViewerLive_ ? "Updating" : "Connecting...");
          cubeViewerDebugLog("Viewer probe deferred while transport is still active.");
          continue;
        }

        ++failedProbeCount;
        cubeViewerConnected_ = false;
        cubeViewerWindowUsable_ = false;
        if (hadViewerSession && failedProbeCount >= 5 && msSinceHeartbeat >= 1000) {
          markViewerInactive("Viewer closed or unreachable; auto-disconnecting.");
          failedProbeCount = 0;
          continue;
        }
        if (failedProbeCount == 1 || failedProbeCount % 10 == 0) {
          cubeViewerDebugLog(hadViewerSession
                                 ? std::string("Viewer probe failed after connection; retrying. count=") + std::to_string(failedProbeCount)
                                 : "Viewer probe failed while connecting; retrying.");
        }
        setStatusLabel("Connecting...");
        continue;
      }

      failedProbeCount = 0;
      cubeViewerConnected_ = true;
      cubeViewerWindowUsable_ = probe.visible && !probe.iconified;
      lastHeartbeatAt_ = std::chrono::steady_clock::now();
      if (!probe.visible || probe.iconified) {
        markViewerInactive(probe.iconified ? "Viewer minimized; auto-disconnecting."
                                           : "Viewer hidden; auto-disconnecting.");
        continue;
      }

      setStatusLabel(cubeViewerLive_ ? "Updating" : "Connected");
    }
  }

  void enqueueParamsMessage(const std::string& payload, const std::string& reason) {
    markViewerTransportActivity();
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      pendingParams_.payload = payload;
      pendingParams_.reason = reason;
      pendingParams_.valid = true;
    }
    cubeViewerDebugLog(std::string("Queued params payload bytes=") + std::to_string(payload.size()) +
                       " reason=" + reason);
    ioCv_.notify_one();
  }

  void enqueueCloudMessage(const std::string& payload,
                           const std::string& reason,
                           std::shared_ptr<ViewerCloudTransportBlob> keepAliveBlob = {}) {
    markViewerTransportActivity();
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      pendingCloud_.payload = payload;
      pendingCloud_.reason = reason;
      pendingCloud_.keepAliveBlob = std::move(keepAliveBlob);
      pendingCloud_.valid = true;
      cloudQueuedOrInFlight_.store(true, std::memory_order_relaxed);
    }
    cubeViewerDebugLog(std::string("Queued cloud payload bytes=") + std::to_string(payload.size()) +
                       " reason=" + reason);
    ioCv_.notify_one();
  }

  // Cached clouds are only reusable when the viewer-facing interpretation is unchanged.
  // This keeps reconnect/refresh fast without letting old strip/cloud states leak into new modes.
  bool trySendCachedCloud(double time, const std::string& reason) {
    if (!viewerSessionRequested() || !cubeViewerLive_) return false;
    ensureViewerSessionTransportReady();
    std::lock_guard<std::mutex> lock(stateMutex_);
    if (!cachedCloud_.valid) {
      cubeViewerDebugLog(std::string("Cached cloud miss: reason=") + reason);
      return false;
    }
    const int qualityIndex = getChoiceValue("cubeViewerQuality", time, cubeViewerQuality_);
    const int resolution = qualityResolutionForIndex(qualityIndex);
    const std::string settingsKey = currentCloudSettingsKey(time);
    if (cachedCloud_.resolution != resolution ||
        cachedCloud_.quality != qualityLabelForIndex(qualityIndex) ||
        cachedCloud_.settingsKey != settingsKey) {
      cubeViewerDebugLog(std::string("Cached cloud stale for current settings: reason=") + reason);
      return false;
    }
    std::shared_ptr<ViewerCloudTransportBlob> freshBlob;
    std::string freshPayload;
    if (!cachedCloud_.samples.empty()) {
      std::string transportMode;
      freshPayload = buildInputCloudJson(cachedCloud_.samples,
                                         cachedCloud_.paramHash,
                                         cachedCloud_.settingsKey,
                                         cachedCloud_.resolution,
                                         qualityIndex,
                                         &freshBlob,
                                         &transportMode);
    } else {
      freshPayload = buildInputCloudJson(cachedCloud_.pointsPayload,
                                         cachedCloud_.paramHash,
                                         cachedCloud_.settingsKey,
                                         cachedCloud_.resolution,
                                         qualityIndex);
    }
    cachedCloud_.payload = freshPayload;
    cachedCloud_.fastBlob = freshBlob;
    enqueueCloudMessage(freshPayload, reason + "/cached", freshBlob);
    cubeViewerInputCloudRefreshPending_ = false;
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    logSharedViewerEvent("cachedCloudQueued", lastCloudSourceId_,
                         std::string("reason=") + reason +
                             " settings=" + cachedCloud_.settingsKey);
    cubeViewerDebugLog(std::string("Cached cloud queued: reason=") + reason +
                       " quality=" + cachedCloud_.quality +
                       " res=" + std::to_string(cachedCloud_.resolution));
    return true;
  }

  // All viewer IPC runs on a worker thread so host renders never block on named-pipe/socket latency.
  // The worker coalesces the latest params/cloud payloads and retries transient connection failures.
  void ioWorkerLoop() {
    for (;;) {
      PendingMessage params;
      PendingMessage cloud;
      {
        std::unique_lock<std::mutex> lock(ioMutex_);
        ioCv_.wait(lock, [this] { return ioStop_ || pendingParams_.valid || pendingCloud_.valid; });
        if (ioStop_) break;
        if (pendingParams_.valid) {
          params = pendingParams_;
          pendingParams_.valid = false;
        }
        if (pendingCloud_.valid) {
          cloud = pendingCloud_;
          pendingCloud_.valid = false;
        }
      }
      bool paramsSucceeded = !params.valid;
      std::lock_guard<std::mutex> transportLock(gSharedCubeViewerTransportMutex);
      if (params.valid) {
        logSharedViewerEvent("io/sendParams/begin", std::string(),
                             std::string("reason=") + params.reason +
                                 " bytes=" + std::to_string(params.payload.size()));
        cubeViewerDebugLog(std::string("Sending params payload bytes=") + std::to_string(params.payload.size()) +
                           " reason=" + params.reason);
        if (sendViewerMessageWithRetry(params.payload, false)) {
          markViewerTransportActivity();
          cubeViewerConnected_ = true;
          cubeViewerWindowUsable_ = true;
          markSharedViewerActiveSender();
          setStatusLabel(cubeViewerLive_ ? "Updating" : "Connected");
          cubeViewerDebugLog("Params payload send succeeded.");
          logSharedViewerEvent("io/sendParams/success");
          paramsSucceeded = true;
        } else {
          paramsSucceeded = false;
          cubeViewerDebugLog("Params payload send failed.");
          logSharedViewerEvent("io/sendParams/fail");
        }
      }
      if (cloud.valid && paramsSucceeded) {
        logSharedViewerEvent("io/sendCloud/begin", std::string(),
                             std::string("reason=") + cloud.reason +
                                 " bytes=" + std::to_string(cloud.payload.size()));
        cubeViewerDebugLog(std::string("Sending cloud payload bytes=") + std::to_string(cloud.payload.size()) +
                           " reason=" + cloud.reason);
        if (sendViewerMessageWithRetry(cloud.payload, true)) {
          markViewerTransportActivity();
          cubeViewerConnected_ = true;
          cubeViewerWindowUsable_ = true;
          setStatusLabel("Updating");
          cubeViewerDebugLog("Cloud payload send succeeded.");
          logSharedViewerEvent("io/sendCloud/success");
        } else {
          cubeViewerConnected_ = false;
          cubeViewerWindowUsable_ = false;
          cubeViewerDebugLog("Cloud payload send failed.");
          logSharedViewerEvent("io/sendCloud/fail");
        }
        cloudQueuedOrInFlight_.store(false, std::memory_order_relaxed);
      } else if (cloud.valid) {
        cubeViewerDebugLog("Skipped cloud payload because params handoff did not succeed.");
        logSharedViewerEvent("io/sendCloud/skipped");
        cloudQueuedOrInFlight_.store(false, std::memory_order_relaxed);
      }
    }
  }

  bool sendViewerMessageWithRetry(const std::string& payload, bool isCloud) {
    const auto retrySleep = cubeViewerConnected_ ? std::chrono::milliseconds(25)
                                                 : std::chrono::milliseconds(60);
    const auto retryBudget = cubeViewerConnected_ ? std::chrono::milliseconds(180)
                                                  : std::chrono::milliseconds(4500);
    const auto deadline = std::chrono::steady_clock::now() + retryBudget;
    int attempt = 1;
    for (;;) {
      if (sendViewerMessage(payload)) return true;
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) break;
      cubeViewerDebugLog(std::string(isCloud ? "Cloud" : "Params") +
                         " payload send retry " + std::to_string(attempt) +
                         " budgetMs=" + std::to_string(
                             std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count()));
      ++attempt;
      std::this_thread::sleep_for(retrySleep);
    }
    cubeViewerDebugLog(std::string(isCloud ? "Cloud" : "Params") +
                       " payload send exhausted retry budget after attempts=" + std::to_string(attempt));
    return false;
  }

  bool sendViewerMessage(const std::string& payload) {
#if defined(_WIN32)
    const std::string pipe = cubeViewerPipeName();
    HANDLE pipeHandle = openViewerPipeHandle(pipe);
    if (pipeHandle == INVALID_HANDLE_VALUE) return false;
    DWORD totalWritten = 0;
    BOOL ok = TRUE;
    while (totalWritten < payload.size()) {
      DWORD chunkWritten = 0;
      const DWORD remaining = static_cast<DWORD>(std::min<size_t>(payload.size() - totalWritten, 1u << 20));
      ok = WriteFile(pipeHandle, payload.data() + totalWritten, remaining, &chunkWritten, nullptr);
      if (!ok || chunkWritten == 0) break;
      totalWritten += chunkWritten;
    }
    CloseHandle(pipeHandle);
    return ok && totalWritten == payload.size();
#else
    const std::string sock = cubeViewerPipeName();
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      cubeViewerDebugLog(std::string("socket connect failed errno=") + std::to_string(errno));
      ::close(fd);
      return false;
    }
    size_t total = 0;
    while (total < payload.size()) {
      const ssize_t sent = ::send(fd, payload.data() + total, payload.size() - total, MSG_NOSIGNAL);
      if (sent <= 0) {
        cubeViewerDebugLog(std::string("socket send failed after ") + std::to_string(total) + "/" +
                           std::to_string(payload.size()) + " errno=" + std::to_string(errno));
        ::close(fd);
        return false;
      }
      total += static_cast<size_t>(sent);
    }
    ::close(fd);
    return true;
#endif
  }

  // Heartbeats are a lightweight liveness and window-state probe used before launching or reconnecting.
  // They let the plugin reuse the existing singleton viewer instead of spawning duplicates.
  ViewerProbeResult probeViewer() {
    ViewerProbeResult result;
    const uint64_t seq = gSharedCubeViewerSeqCounter.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream payload;
    payload << "{\"type\":\"heartbeat\",\"seq\":" << seq
            << ",\"senderId\":\"" << jsonEscape(senderId_) << "\"}\n";

#if defined(_WIN32)
    const std::string pipe = cubeViewerPipeName();
    HANDLE pipeHandle = openViewerPipeHandle(pipe);
    if (pipeHandle == INVALID_HANDLE_VALUE) return result;
    DWORD written = 0;
    if (!WriteFile(pipeHandle, payload.str().data(), static_cast<DWORD>(payload.str().size()), &written, nullptr)) {
      CloseHandle(pipeHandle);
      return result;
    }
    char buffer[256] = {};
    DWORD read = 0;
    if (!ReadFile(pipeHandle, buffer, sizeof(buffer) - 1, &read, nullptr)) {
      CloseHandle(pipeHandle);
      return result;
    }
    CloseHandle(pipeHandle);
    std::string reply(buffer, buffer + read);
#else
    const std::string sock = cubeViewerPipeName();
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return result;
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 100000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      cubeViewerDebugLog(std::string("heartbeat connect failed errno=") + std::to_string(errno));
      ::close(fd);
      return result;
    }
    const std::string msg = payload.str();
    size_t total = 0;
    while (total < msg.size()) {
      const ssize_t sent = ::send(fd, msg.data() + total, msg.size() - total, MSG_NOSIGNAL);
      if (sent <= 0) {
        cubeViewerDebugLog(std::string("heartbeat send failed after ") + std::to_string(total) + "/" +
                           std::to_string(msg.size()) + " errno=" + std::to_string(errno));
        ::close(fd);
        return result;
      }
      total += static_cast<size_t>(sent);
    }
    char buffer[256] = {};
    const ssize_t r = ::recv(fd, buffer, sizeof(buffer) - 1, 0);
    ::close(fd);
    if (r <= 0) return result;
    std::string reply(buffer, buffer + r);
#endif
    result.ok = reply.find("\"type\":\"heartbeat_ack\"") != std::string::npos;
    result.visible = reply.find("\"visible\":0") == std::string::npos;
    result.iconified = reply.find("\"iconified\":1") != std::string::npos;
    result.focused = reply.find("\"focused\":0") == std::string::npos;
    if (result.ok && (!hasLoggedHeartbeatProbe_ || !sameViewerProbeState(result, lastLoggedHeartbeatProbe_))) {
      cubeViewerDebugLog(std::string("Heartbeat ok visible=") + (result.visible ? "1" : "0") +
                         " iconified=" + (result.iconified ? "1" : "0") +
                         " focused=" + (result.focused ? "1" : "0"));
      lastLoggedHeartbeatProbe_ = result;
      hasLoggedHeartbeatProbe_ = true;
    }
    return result;
  }

  // Opening a session means "ensure a usable viewer exists, then mark this OFX instance as active."
  // Reuse is preferred over relaunch so repeated connect/refresh actions stay within the single-viewer rule.
  void openCubeViewerSession(double time) {
    startIoWorker();
    startStatusThread();
    ViewerProbeResult existing = probeViewer();
    if (existing.ok) {
      cubeViewerDebugLog("Reusing already running viewer instance.");
      const uint64_t seq = gSharedCubeViewerSeqCounter.fetch_add(1, std::memory_order_relaxed);
      std::ostringstream payload;
      payload << "{\"type\":\"bring_to_front\",\"seq\":" << seq
              << ",\"senderId\":\"" << jsonEscape(senderId_) << "\"}\n";
      (void)sendViewerMessage(payload.str());
    } else {
      launchViewerProcess();
    }
    retainSharedViewerSession();
    cubeViewerRequested_ = true;
    markViewerTransportActivity();
    cubeViewerConnected_ = existing.ok;
    cubeViewerWindowUsable_ = existing.ok ? (existing.visible && !existing.iconified) : true;
    cubeViewerInputCloudRefreshPending_ = true;
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    playbackRenderBurstCount_ = 0;
    lastCloudBuiltAt_ = std::chrono::steady_clock::time_point{};
    lastHeartbeatAt_ = existing.ok ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    {
      std::lock_guard<std::mutex> lock(ioMutex_);
      pendingCloud_.valid = false;
      cloudQueuedOrInFlight_.store(false, std::memory_order_relaxed);
    }
    pushParamsUpdate(time, "openCubeViewer");
    (void)trySendCachedCloud(time, "openCubeViewer");
    setStatusLabel(existing.ok ? (cubeViewerLive_ ? "Updating" : "Connected") : "Connecting...");
    cubeViewerDebugLog("Cube viewer session opened.");
  }

  void closeCubeViewerSession() {
    releaseSharedViewerSession();
    cubeViewerRequested_ = false;
    cubeViewerConnected_ = false;
    cubeViewerWindowUsable_ = false;
    cubeViewerInputCloudRefreshPending_ = false;
    deferredLatestCloudRefresh_.store(false, std::memory_order_relaxed);
    playbackRenderBurstCount_ = 0;
    lastCloudBuiltAt_ = std::chrono::steady_clock::time_point{};
    lastHeartbeatAt_ = std::chrono::steady_clock::time_point{};
    lastCloudSourceId_.clear();
    lastCloudSettingsKey_.clear();
    releaseViewerRuntimeResources();
    setStatusLabel("Disconnected");
    cubeViewerDebugLog("Cube viewer session closed.");
  }

  void pushParamsUpdate(double time, const std::string& reason) {
    if (!viewerSessionRequested()) return;
    ensureViewerSessionTransportReady();
    logSharedViewerEvent("queueParams", std::string(),
                         std::string("reason=") + reason +
                             " cloudKey=" + currentCloudSettingsKey(time));
    enqueueParamsMessage(buildParamsPayload(time), reason);
  }

  std::vector<std::string> viewerExecutableCandidates() const {
    std::vector<std::string> out;
    const std::string exeName = viewerExecutableName();
    const std::string modulePath = pluginModulePath();
    const std::string moduleDir = parentDir(modulePath);
    const std::string bundleRoot = findBundleRootFromModule(modulePath);
    const char* envOverride = std::getenv("CHROMASPACE_EXE");
    if (envOverride && envOverride[0] != '\0') {
      const std::string envPath(envOverride);
      out.push_back(envPath);
      if (!isAbsolutePath(envPath) && !moduleDir.empty()) out.push_back(joinPath(moduleDir, envPath));
    }
    if (!bundleRoot.empty()) {
#if defined(_WIN32)
      out.push_back(joinPath(joinPath(joinPath(bundleRoot, "Contents"), "Win64"), exeName));
#elif defined(__APPLE__)
      out.push_back(joinPath(joinPath(joinPath(bundleRoot, "Contents"), "MacOS"), exeName));
#else
      out.push_back(joinPath(joinPath(joinPath(bundleRoot, "Contents"), "Linux-x86-64"), exeName));
#endif
    }
    if (!moduleDir.empty()) {
      out.push_back(joinPath(moduleDir, exeName));
      const std::string contentsDir = parentDir(moduleDir);
      if (!contentsDir.empty()) {
        out.push_back(joinPath(contentsDir, exeName));
        out.push_back(joinPath(joinPath(contentsDir, "Resources"), exeName));
      }
    }
    const std::string cwdCandidate = joinPath(std::filesystem::current_path().string(), exeName);
    out.push_back(cwdCandidate);
    out.push_back(exeName);
    std::vector<std::string> unique;
    for (const auto& candidate : out) {
      if (candidate.empty()) continue;
      if (std::find(unique.begin(), unique.end(), candidate) == unique.end()) {
        unique.push_back(candidate);
      }
    }
    return unique;
  }

  // Launch tries bundle-relative paths first so installed OFX bundles work without manual viewer setup,
  // then falls back to cwd/PATH-style candidates for workshop and local build workflows.
  void launchViewerProcess() {
    const auto candidates = viewerExecutableCandidates();
    std::ostringstream attempted;
#if defined(_WIN32)
    for (const auto& candidate : candidates) {
      const bool literalCandidate = candidate == std::string(viewerExecutableName());
      if (!literalCandidate && !fileExistsForLaunch(candidate)) {
        attempted << (attempted.tellp() > 0 ? "; " : "") << candidate;
        continue;
      }
      cubeViewerDebugLog(std::string("Launching viewer from: ") + candidate);
      STARTUPINFOA si{};
      PROCESS_INFORMATION pi{};
      si.cb = sizeof(si);
      std::string cmdLine = "\"" + candidate + "\"";
      const BOOL ok = CreateProcessA(
          nullptr,
          cmdLine.data(),
          nullptr,
          nullptr,
          FALSE,
          CREATE_NEW_PROCESS_GROUP,
          nullptr,
          literalCandidate ? nullptr : parentDir(candidate).c_str(),
          &si,
          &pi);
      if (ok == TRUE) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        cubeViewerDebugLog(std::string("CreateProcess succeeded: ") + candidate);
        return;
      }
      attempted << (attempted.tellp() > 0 ? "; " : "") << candidate << " (err=" << GetLastError() << ")";
    }
    cubeViewerDebugLog(std::string("Viewer launch failed. attempted: ") + attempted.str());
    setStatusLabel("Viewer launch failed");
#else
    for (const auto& candidate : candidates) {
      const bool literalCandidate = candidate == std::string(viewerExecutableName());
      if (!literalCandidate && !fileExistsForLaunch(candidate)) {
        attempted << (attempted.tellp() > 0 ? "; " : "") << candidate;
        continue;
      }
      cubeViewerDebugLog(std::string("Launching viewer from: ") + candidate);
      pid_t pid = 0;
      const std::string exe = candidate;
      char* const argv[] = {const_cast<char*>(exe.c_str()), nullptr};
      const int spawnErr = posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv, environ);
      if (spawnErr == 0) {
        cubeViewerDebugLog(std::string("posix_spawn succeeded: ") + candidate);
        return;
      }
      attempted << (attempted.tellp() > 0 ? "; " : "") << candidate << " (err=" << spawnErr << ")";
    }
    cubeViewerDebugLog(std::string("Viewer launch failed. attempted: ") + attempted.str());
    setStatusLabel("Viewer launch failed");
#endif
  }
};

class ChromaspaceOverlayInteract : public OFX::OverlayInteract {
 public:
  ChromaspaceOverlayInteract(OfxInteractHandle handle, OFX::ImageEffect* effect)
      : OFX::OverlayInteract(handle)
      , effect_(static_cast<ChromaspaceEffect*>(effect)) {}

  bool draw(const OFX::DrawArgs& args) override {
    if (!effect_ || !effect_->currentLassoRegionSlicingEnabled(args.time)) {
      cancelActiveStroke();
      return false;
    }

    const OfxRectD rect = effect_->currentLassoImageRect(args.time);
    const LassoRegionState state = effect_->currentLassoRegionState(args.time);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(2.0f);

    for (const auto& stroke : state.strokes) {
      drawCommittedStroke(stroke, rect);
    }
    if (!activeStroke_.empty()) {
      drawActiveStroke(rect);
    }
    return !state.strokes.empty() || !activeStroke_.empty();
  }

  bool penDown(const OFX::PenArgs& args) override {
    if (!effect_ || !effect_->currentLassoRegionSlicingEnabled(args.time)) {
      cancelActiveStroke();
      return false;
    }
    const OfxPointD clampedPoint = clampPenToImage(args.penPosition, effect_->currentLassoImageRect(args.time));
    if (shiftHeld_) {
      cancelActiveStroke();
      effect_->resetLassoRegion(args.time, "cubeViewerLassoReset/shift-click");
      effect_->redrawOverlays();
      return true;
    }
    cancelActiveStroke();
    strokeActive_ = true;
    strokeSubtract_ = altHeld_ || effect_->currentLassoSubtractOperation(args.time);
    appendActivePoint(clampedPoint, args, true);
    effect_->redrawOverlays();
    return true;
  }

  bool penMotion(const OFX::PenArgs& args) override {
    if (!strokeActive_ || !effect_ || !effect_->currentLassoRegionSlicingEnabled(args.time)) return false;
    appendActivePoint(clampPenToImage(args.penPosition, effect_->currentLassoImageRect(args.time)), args, false);
    effect_->redrawOverlays();
    return true;
  }

  bool penUp(const OFX::PenArgs& args) override {
    if (!strokeActive_ || !effect_) return false;
    appendActivePoint(clampPenToImage(args.penPosition, effect_->currentLassoImageRect(args.time)), args, true);
    const std::vector<OfxPointD> stroke = activeStroke_;
    const bool subtract = strokeSubtract_;
    cancelActiveStroke();
    effect_->redrawOverlays();
    if (stroke.size() < 3) return true;
    effect_->commitLassoStroke(args.time, stroke, subtract, "cubeViewerLassoData/interact");
    return true;
  }

  bool keyDown(const OFX::KeyArgs& args) override {
    if (!effect_ || !effect_->currentLassoRegionSlicingEnabled(args.time)) return false;
    if (args.keySymbol == kOfxKey_Control_L || args.keySymbol == kOfxKey_Control_R) {
      controlHeld_ = true;
      return false;
    }
    if (args.keySymbol == kOfxKey_Alt_L || args.keySymbol == kOfxKey_Alt_R) {
      altHeld_ = true;
      if (strokeActive_) {
        strokeSubtract_ = true;
        effect_->redrawOverlays();
      }
      return strokeActive_;
    }
    if (args.keySymbol == kOfxKey_Shift_L || args.keySymbol == kOfxKey_Shift_R) {
      shiftHeld_ = true;
      return false;
    }
    if (args.keySymbol == kOfxKey_Escape) {
      if (!strokeActive_) return false;
      cancelActiveStroke();
      effect_->redrawOverlays();
      return true;
    }
    if (args.keySymbol == kOfxKey_BackSpace || args.keySymbol == kOfxKey_Delete) {
      effect_->undoLassoRegionStroke(args.time, "cubeViewerLassoUndo/key");
      return true;
    }
    return false;
  }

  bool keyUp(const OFX::KeyArgs& args) override {
    if (args.keySymbol == kOfxKey_Control_L || args.keySymbol == kOfxKey_Control_R) {
      controlHeld_ = false;
      return false;
    }
    if (args.keySymbol == kOfxKey_Alt_L || args.keySymbol == kOfxKey_Alt_R) {
      altHeld_ = false;
      if (strokeActive_ && effect_) {
        strokeSubtract_ = effect_->currentLassoSubtractOperation(args.time);
        effect_->redrawOverlays();
        return true;
      }
      return false;
    }
    if (args.keySymbol != kOfxKey_Shift_L && args.keySymbol != kOfxKey_Shift_R) return false;
    shiftHeld_ = false;
    if (strokeActive_ && effect_) {
      return false;
    }
    return false;
  }

 private:
  ChromaspaceEffect* effect_ = nullptr;
  bool strokeActive_ = false;
  bool strokeSubtract_ = false;
  bool shiftHeld_ = false;
  bool controlHeld_ = false;
  bool altHeld_ = false;
  std::vector<OfxPointD> activeStroke_;

  void cancelActiveStroke() {
    strokeActive_ = false;
    strokeSubtract_ = false;
    activeStroke_.clear();
  }

  OfxPointD clampPenToImage(const OfxPointD& penPosition, const OfxRectD& rect) const {
    return OfxPointD{
        std::clamp(penPosition.x, rect.x1, rect.x2),
        std::clamp(penPosition.y, rect.y1, rect.y2),
    };
  }

  void appendActivePoint(const OfxPointD& point, const OFX::PenArgs& args, bool force) {
    if (activeStroke_.empty()) {
      activeStroke_.push_back(point);
      return;
    }
    const auto& last = activeStroke_.back();
    const double dx = point.x - last.x;
    const double dy = point.y - last.y;
    const double minDistance = std::max(args.pixelScale.x, args.pixelScale.y) * 2.0;
    if (!force && (dx * dx + dy * dy) < (minDistance * minDistance)) return;
    activeStroke_.push_back(point);
  }

  static OfxPointD denormalizePoint(const LassoPointNorm& point, const OfxRectD& rect) {
    return OfxPointD{
        rect.x1 + static_cast<double>(point.xNorm) * (rect.x2 - rect.x1),
        rect.y1 + static_cast<double>(point.yNorm) * (rect.y2 - rect.y1),
    };
  }

  void drawCommittedStroke(const LassoStroke& stroke, const OfxRectD& rect) const {
    if (stroke.points.size() < 3) return;
    const float fillR = stroke.subtract ? 0.95f : 0.15f;
    const float fillG = stroke.subtract ? 0.20f : 0.85f;
    const float fillB = stroke.subtract ? 0.15f : 0.95f;
    const float lineR = stroke.subtract ? 1.00f : 0.35f;
    const float lineG = stroke.subtract ? 0.50f : 1.00f;
    const float lineB = stroke.subtract ? 0.30f : 1.00f;
    glColor4f(fillR, fillG, fillB, 0.12f);
    glBegin(GL_POLYGON);
    for (const auto& point : stroke.points) {
      const OfxPointD denorm = denormalizePoint(point, rect);
      glVertex2d(denorm.x, denorm.y);
    }
    glEnd();
    glColor4f(lineR, lineG, lineB, 0.95f);
    glBegin(GL_LINE_LOOP);
    for (const auto& point : stroke.points) {
      const OfxPointD denorm = denormalizePoint(point, rect);
      glVertex2d(denorm.x, denorm.y);
    }
    glEnd();
  }

  void drawActiveStroke(const OfxRectD& rect) const {
    if (activeStroke_.empty()) return;
    const float lineR = strokeSubtract_ ? 1.00f : 0.95f;
    const float lineG = strokeSubtract_ ? 0.55f : 0.95f;
    const float lineB = strokeSubtract_ ? 0.30f : 0.20f;
    glColor4f(lineR, lineG, lineB, 1.0f);
    glBegin(activeStroke_.size() >= 3 ? GL_LINE_LOOP : GL_LINE_STRIP);
    for (const auto& point : activeStroke_) {
      const OfxPointD clamped = clampPenToImage(point, rect);
      glVertex2d(clamped.x, clamped.y);
    }
    glEnd();
  }
};

class ChromaspaceOverlayDescriptor
    : public OFX::DefaultEffectOverlayDescriptor<ChromaspaceOverlayDescriptor, ChromaspaceOverlayInteract> {};

class ChromaspaceFactory : public PluginFactoryHelper<ChromaspaceFactory> {
 public:
  ChromaspaceFactory()
      : PluginFactoryHelper<ChromaspaceFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}

  void load() override {}
  void unload() override {}

  void describe(ImageEffectDescriptor& d) override {
    static const std::string nameWithVersion = std::string(kPluginName) + " " + kPluginVersionLabel;
    d.setLabels(nameWithVersion.c_str(), nameWithVersion.c_str(), nameWithVersion.c_str());
    d.setPluginGrouping(kPluginGrouping);
    d.setPluginDescription("Standalone 3D cube viewer for plotting the input image as a point cloud.");
    d.getPropertySet().propSetString(kOfxPropIcon, "", 0, false);
    d.getPropertySet().propSetString(kOfxPropIcon, "com.moazelgabry.chromaspace.png", 1, false);
    d.addSupportedContext(eContextFilter);
    d.addSupportedBitDepth(eBitDepthFloat);
    d.setSingleInstance(false);
    d.setSupportsTiles(false);
    d.setSupportsMultiResolution(false);
    d.setTemporalClipAccess(false);
    d.setOverlayInteractDescriptor(new ChromaspaceOverlayDescriptor);
#if defined(CHROMASPACE_HAS_CUDA)
    d.setSupportsCudaRender(true);
    d.setSupportsCudaStream(true);
#endif
#if defined(__APPLE__)
    d.setSupportsMetalRender(true);
#endif
#if defined(CHROMASPACE_HAS_OPENCL)
    d.setSupportsOpenCLBuffersRender(true);
    d.setSupportsOpenCLImagesRender(true);
#endif
  }

  void describeInContext(ImageEffectDescriptor& d, ContextEnum) override {
    ClipDescriptor* src = d.defineClip(kOfxImageEffectSimpleSourceClipName);
    src->addSupportedComponent(ePixelComponentRGBA);
    src->setTemporalClipAccess(false);
    src->setSupportsTiles(false);

    ClipDescriptor* dst = d.defineClip(kOfxImageEffectOutputClipName);
    dst->addSupportedComponent(ePixelComponentRGBA);
    dst->setSupportsTiles(false);

    auto tooltipFor = [](const std::string& name) -> const char* {
      static const std::map<std::string, const char*> kHints = {
          {"openCubeViewer", "Open the standalone 3D cube viewer."},
          {"closeCubeViewer", "Disconnect this OFX instance from the viewer without closing the external window."},
          {"cubeViewerLive", "When enabled, changes update the viewer continuously."},
          {"cubeViewerUpdateMode", "Choose how live viewer refreshes are scheduled. Auto adapts between fluid and scheduled behavior, Fluid prioritizes the smoothest point-cloud updates, and Scheduled prioritizes steadier host playback when live updates become heavy."},
          {"cubeViewerOnTop", "Keep the external viewer above the host application."},
          {"cubeViewerQuality", "Viewer sampling density for the 3D cube (Low=25^3, about 45k points; Medium=41^3, about 90k points; High=57^3, about 180k points)."},
          {"cubeViewerScale", "Scales the sampled image domain used for cube generation to lighten processing. 100% keeps full size, while lower values reduce cloud-build work."},
          {"cubeViewerPointSize", "Makes points larger or smaller and automatically adjusts point density in the opposite direction to keep the cloud readable. Sizes above 1.0 use a looser density reduction so the cloud can clump more densely instead of opening up too much."},
          {"cubeViewerColorSaturation", "Adjust how vivid the plotted colors appear in the viewer. Higher values reduce the washed-out white look and make it easier to read what hues are being plotted."},
          {"cubeViewerSamplingMode", "Balanced uses a deterministic lattice for stable coverage, Stratified adds jitter for cleaner coverage, and Random gives a noisier organic scatter."},
          {"cubeViewerOccupancyGuidedFill", "Adds a second occupancy-guided pass after the normal image sampler so sparsely occupied RGB regions receive more support and the plot reads denser without switching to the instance-1 workflow."},
          {"cubeViewerPointShape", "Choose whether points are rendered as circular or square splats in the viewer."},
          {"cubeViewerShowOverflow", "Cube, HSL, HSV, Chen, JP-Conical, and Reuleaux plot modes: allow points outside the nominal guide bounds so out-of-range transforms remain visible instead of being clamped back into the model."},
          {"cubeViewerHighlightOverflow", "When enabled, color out-of-bound plot points with a dedicated highlight color."},
          {"cubeViewerCircularHsl", "For HSL only: switch from the default HSL bicone view to a circular cylindrical HSL view with hue as angle, saturation as radius, and lightness as height. When overflow is enabled, out-of-range RGB values are converted with the raw HSL formula so useful out-of-bound cylindrical points remain visible."},
          {"cubeViewerCircularHsv", "For HSV only: switch from the default Smith hexcone view to a circular cylindrical HSV view with hue as angle, saturation as radius, and value as height."},
          {"cubeViewerLassoRegionMode", "When enabled, Volume Slicing uses a drawn lasso region on the image instead of the hue-sector filters."},
          {"cubeViewerNeutralRadius", "Keep only samples within this normalized distance from the achromatic axis so the extreme outer saturation shell is hidden and the more typical image range is easier to inspect."},
          {"cubeViewerSliceRed", "Show only the red sector. In Cube mode this is the red-dominant tetrahedral region; in the other plot models it is the red-centered hue sector."},
          {"cubeViewerSliceGreen", "Show only the green sector. In Cube mode this is the green-dominant tetrahedral region; in the other plot models it is the green-centered hue sector."},
          {"cubeViewerSliceBlue", "Show only the blue sector. In Cube mode this is the blue-dominant tetrahedral region; in the other plot models it is the blue-centered hue sector."},
          {"cubeViewerSliceCyan", "Show only the cyan sector. In Cube mode this is the blue-over-green tetrahedral region; in the other plot models it is the cyan-centered hue sector."},
          {"cubeViewerSliceYellow", "Show only the yellow sector. In Cube mode this is the green-over-red tetrahedral region; in the other plot models it is the yellow-centered hue sector."},
          {"cubeViewerSliceMagenta", "Show only the magenta sector. In Cube mode this is the red-over-blue tetrahedral region; in the other plot models it is the magenta-centered hue sector."},
          {"cubeViewerLassoOperation", "When Volume Slicing is in Lasso Region mode, choose whether the next stroke adds to the selection or subtracts from it. Shortcut: hold Alt while drawing for a temporary subtract stroke."},
          {"cubeViewerLassoUndo", "Remove the most recently committed lasso stroke."},
          {"cubeViewerLassoReset", "Clear the current lasso region so you can draw a fresh selection. Shortcut: Shift+left click on the image."},
          {"cubeViewerOverflowHighlightColor", "Choose the color used to highlight out-of-bound plot points."},
          {"cubeViewerBackgroundColor", "Choose the viewer background color used behind the 3D plot."},
{"cubeViewerIdentityOverlayEnabled", "Enable the synthetic identity plot reference for the current mode so you can compare or pass a known solid through the pipeline."},
{"cubeViewerIdentityOverlaySize", "Controls fill-volume density in the 3D viewer from 4 to 65. Higher values create a denser synthetic volume."},
          {"cubeViewerIdentityOverlayRamp", "Adds a linear 0..1 neutral ramp to the identity overlay so luma appears as a grayscale diagonal through the 3D plot."},
          {"cubeViewerIdentityOverlayNote", "For this to work you need to use another instance in generate mode Upstream"},
          {"cubeViewerSampleDrawnCubeOnly", "Enable this on a downstream instance when an earlier instance is generating the identity plot strip into the image. The downstream plot will read that strip and combine its dense identity-solid sampling with the normal whole-image sampling."},
          {"cubeViewerReadGrayRamp", "Enable this on a downstream instance to add a dedicated concentrated readback of the gray ramp band from the identity plot strip."},
          {"cubeViewerShowIdentityOnly", "Only available when using the identity plot from instance 1. When enabled, the downstream plot reads only the drawn identity strip and skips the normal whole-image cloud, so you see just the transformed identity data."},
{"cubeViewerSampleDrawnCubeSize", "Sets the identity-strip resolution from 4 to 65. In the identity generator this controls the generated strip density, and in a downstream instance it should match instance 1 so the strip can be decoded correctly."},
{"cubeViewerModeToggle", "Switch between the 3D viewer and the identity generator. The identity generator burns the identity strip into the image and hides plot-only controls."},
          {"cubeViewerPlotModel", "Choose which 3D color geometry is used to plot the current signal: RGB cube, HSL bicone or circular HSL, HSV hexcone or circular HSV, Chen, Norm-Cone, JP-Conical, Reuleaux, or Chromaticity xyY."},
          {"cubeViewerPlotDisplayLinear", "Decode sampled input values from the selected input transfer function to linear light before building the 3D plot. Intended for non-chromaticity plot modes."},
          {"cubeViewerPlotDisplayLinearTransfer", "Choose the assumed input transfer function used when Plot in Display Linear is enabled."},
          {"cubeViewerNormConeNormalized", "For Norm-Cone only: when enabled, use the normalized cone chroma from JP's DCTL; when disabled, use the raw spherical chroma variant instead."},
          {"cubeViewerChromaticityInputPrimaries", "Choose the assumed input primaries used to convert incoming RGB into XYZ before plotting chromaticity xyY."},
          {"cubeViewerChromaticityInputTransfer", "Choose the assumed input transfer function used to decode incoming RGB to linear light before chromaticity conversion."},
          {"cubeViewerChromaticityReferenceBasis", "Plot chromaticity coordinates relative to the CIE standard observer basis or the selected input observer basis."},
          {"cubeViewerChromaticityOverlayPrimaries", "Choose which gamut triangle to overlay on the chromaticity plot, or set None to hide the overlay triangle."},
          {"cubeViewerChromaticityPlanckianLocus", "Show the Planckian locus overlay and Kelvin labels in chromaticity mode."},
          {"cubeViewerStatus", "Connection state for the external cube viewer."},
      };
      auto it = kHints.find(name);
      return it == kHints.end() ? nullptr : it->second;
    };
    const ChromaspacePresetValues chromaspaceDefaultValues = describeChromaspaceDefaultValues();

    auto* cubeViewerSource = d.defineChoiceParam("cubeViewerSource");
    cubeViewerSource->appendOption("Input Image");
    cubeViewerSource->setDefault(0);
    cubeViewerSource->setIsSecret(true);

    auto* cubeViewerPlotModel = d.defineChoiceParam("cubeViewerPlotModel");
    cubeViewerPlotModel->setLabel("Plot Model");
    cubeViewerPlotModel->appendOption("Cube");
    cubeViewerPlotModel->appendOption("HSL");
    cubeViewerPlotModel->appendOption("HSV");
    cubeViewerPlotModel->appendOption("Chen");
    cubeViewerPlotModel->appendOption("Norm-Cone");
    cubeViewerPlotModel->appendOption("JP-Conical");
    cubeViewerPlotModel->appendOption("Reuleaux");
    cubeViewerPlotModel->appendOption("Chromaticity");
    cubeViewerPlotModel->setDefault(std::clamp(chromaspaceDefaultValues.plotModel, 0, 7));
    if (const char* hint = tooltipFor("cubeViewerPlotModel")) cubeViewerPlotModel->setHint(hint);

    auto* openCubeViewer = d.definePushButtonParam("openCubeViewer");
    openCubeViewer->setLabel("Open 3D Viewer");
    if (const char* hint = tooltipFor("openCubeViewer")) openCubeViewer->setHint(hint);

    auto* cubeViewerPlotDisplayLinear = d.defineBooleanParam("cubeViewerPlotDisplayLinear");
    cubeViewerPlotDisplayLinear->setLabel("Plot in Linear");
    cubeViewerPlotDisplayLinear->setDefault(chromaspaceDefaultValues.plotInLinear);
    if (const char* hint = tooltipFor("cubeViewerPlotDisplayLinear")) cubeViewerPlotDisplayLinear->setHint(hint);

    auto* cubeViewerPlotDisplayLinearTransfer = d.defineChoiceParam("cubeViewerPlotDisplayLinearTransfer");
    cubeViewerPlotDisplayLinearTransfer->setLabel("Input Transfer Function");
    for (const auto id : plotLinearTransferChoices()) {
      cubeViewerPlotDisplayLinearTransfer->appendOption(WorkshopColor::transferFunctionDefinition(id).label);
    }
    cubeViewerPlotDisplayLinearTransfer->setDefault(
        plotLinearTransferChoiceIndex(static_cast<WorkshopColor::TransferFunctionId>(chromaspaceDefaultValues.inputTransferFunction)));
    cubeViewerPlotDisplayLinearTransfer->setIsSecret(true);
    cubeViewerPlotDisplayLinearTransfer->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerPlotDisplayLinearTransfer")) cubeViewerPlotDisplayLinearTransfer->setHint(hint);

    auto* cubeViewerCircularHsl = d.defineBooleanParam("cubeViewerCircularHsl");
    cubeViewerCircularHsl->setLabel("Circular HSL");
    cubeViewerCircularHsl->setDefault(false);
    cubeViewerCircularHsl->setIsSecret(true);
    cubeViewerCircularHsl->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerCircularHsl")) cubeViewerCircularHsl->setHint(hint);

    auto* cubeViewerCircularHsv = d.defineBooleanParam("cubeViewerCircularHsv");
    cubeViewerCircularHsv->setLabel("Circular HSV");
    cubeViewerCircularHsv->setDefault(false);
    cubeViewerCircularHsv->setIsSecret(true);
    cubeViewerCircularHsv->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerCircularHsv")) cubeViewerCircularHsv->setHint(hint);

    auto* cubeViewerNormConeNormalized = d.defineBooleanParam("cubeViewerNormConeNormalized");
    cubeViewerNormConeNormalized->setLabel("Normalized Chroma");
    cubeViewerNormConeNormalized->setDefault(true);
    cubeViewerNormConeNormalized->setIsSecret(true);
    cubeViewerNormConeNormalized->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerNormConeNormalized")) cubeViewerNormConeNormalized->setHint(hint);

    auto* cubeViewerShowOverflow = d.defineBooleanParam("cubeViewerShowOverflow");
    cubeViewerShowOverflow->setLabel("Show Overflow");
    cubeViewerShowOverflow->setDefault(chromaspaceDefaultValues.showOverflow);
    if (const char* hint = tooltipFor("cubeViewerShowOverflow")) cubeViewerShowOverflow->setHint(hint);

    auto* cubeViewerHighlightOverflow = d.defineBooleanParam("cubeViewerHighlightOverflow");
    cubeViewerHighlightOverflow->setLabel("Highlight Overflow");
    cubeViewerHighlightOverflow->setDefault(chromaspaceDefaultValues.highlightOverflow);
    if (const char* hint = tooltipFor("cubeViewerHighlightOverflow")) cubeViewerHighlightOverflow->setHint(hint);

    auto* cubeViewerIdentityOverlayEnabled = d.defineBooleanParam("cubeViewerIdentityOverlayEnabled");
    cubeViewerIdentityOverlayEnabled->setLabel("Fill Volume");
    cubeViewerIdentityOverlayEnabled->setDefault(false);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlayEnabled")) cubeViewerIdentityOverlayEnabled->setHint(hint);

    auto* cubeViewerIdentityOverlaySize = d.defineIntParam("cubeViewerIdentityOverlaySize");
    cubeViewerIdentityOverlaySize->setLabel("Fill Resolution");
    cubeViewerIdentityOverlaySize->setDefault(std::clamp(chromaspaceDefaultValues.fillResolution, 4, 65));
    cubeViewerIdentityOverlaySize->setRange(4, 65);
    cubeViewerIdentityOverlaySize->setDisplayRange(4, 65);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlaySize")) cubeViewerIdentityOverlaySize->setHint(hint);

    auto* cubeViewerIdentityOverlayRamp = d.defineBooleanParam("cubeViewerIdentityOverlayRamp");
    cubeViewerIdentityOverlayRamp->setLabel("Linear Gray Ramp");
    cubeViewerIdentityOverlayRamp->setDefault(false);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlayRamp")) cubeViewerIdentityOverlayRamp->setHint(hint);

    auto* cubeViewerChromaticityPlanckianLocus = d.defineBooleanParam("cubeViewerChromaticityPlanckianLocus");
    cubeViewerChromaticityPlanckianLocus->setLabel("Planckian Locus");
    cubeViewerChromaticityPlanckianLocus->setDefault(true);
    if (const char* hint = tooltipFor("cubeViewerChromaticityPlanckianLocus")) cubeViewerChromaticityPlanckianLocus->setHint(hint);

    auto* grpCubeViewerChromaticityCm = d.defineGroupParam("grp_cube_viewer_chromaticity_cm");
    grpCubeViewerChromaticityCm->setLabel("Chromaticity Color Management");
    grpCubeViewerChromaticityCm->setOpen(true);
    grpCubeViewerChromaticityCm->setEnabled(false);

    auto* cubeViewerChromaticityInputPrimaries = d.defineChoiceParam("cubeViewerChromaticityInputPrimaries");
    cubeViewerChromaticityInputPrimaries->setLabel("Input Primaries");
    for (std::size_t i = 0; i < WorkshopColor::primariesCount(); ++i) {
      cubeViewerChromaticityInputPrimaries->appendOption(WorkshopColor::primariesDefinition(i).label);
    }
    cubeViewerChromaticityInputPrimaries->setDefault(
        WorkshopColor::primariesChoiceIndex(WorkshopColor::ColorPrimariesId::Rec709));
    cubeViewerChromaticityInputPrimaries->setParent(*grpCubeViewerChromaticityCm);
    if (const char* hint = tooltipFor("cubeViewerChromaticityInputPrimaries")) cubeViewerChromaticityInputPrimaries->setHint(hint);

    auto* cubeViewerChromaticityInputTransfer = d.defineChoiceParam("cubeViewerChromaticityInputTransfer");
    cubeViewerChromaticityInputTransfer->setLabel("Input Transfer");
    for (std::size_t i = 0; i < WorkshopColor::transferFunctionCount(); ++i) {
      cubeViewerChromaticityInputTransfer->appendOption(WorkshopColor::transferFunctionDefinition(i).label);
    }
    cubeViewerChromaticityInputTransfer->setDefault(
        WorkshopColor::transferFunctionChoiceIndex(WorkshopColor::TransferFunctionId::Gamma24));
    cubeViewerChromaticityInputTransfer->setParent(*grpCubeViewerChromaticityCm);
    if (const char* hint = tooltipFor("cubeViewerChromaticityInputTransfer")) cubeViewerChromaticityInputTransfer->setHint(hint);

    auto* cubeViewerChromaticityReferenceBasis = d.defineChoiceParam("cubeViewerChromaticityReferenceBasis");
    cubeViewerChromaticityReferenceBasis->setLabel("Reference Basis");
    cubeViewerChromaticityReferenceBasis->appendOption("CIE Standard Observer");
    cubeViewerChromaticityReferenceBasis->appendOption("Input Observer");
    cubeViewerChromaticityReferenceBasis->setDefault(0);
    cubeViewerChromaticityReferenceBasis->setParent(*grpCubeViewerChromaticityCm);
    if (const char* hint = tooltipFor("cubeViewerChromaticityReferenceBasis")) cubeViewerChromaticityReferenceBasis->setHint(hint);

    auto* cubeViewerChromaticityOverlayPrimaries = d.defineChoiceParam("cubeViewerChromaticityOverlayPrimaries");
    cubeViewerChromaticityOverlayPrimaries->setLabel("Overlay Primaries");
    cubeViewerChromaticityOverlayPrimaries->appendOption("None");
    for (std::size_t i = 0; i < WorkshopColor::primariesCount(); ++i) {
      cubeViewerChromaticityOverlayPrimaries->appendOption(WorkshopColor::primariesDefinition(i).label);
    }
    cubeViewerChromaticityOverlayPrimaries->setDefault(
        WorkshopColor::overlayPrimariesChoiceIndex(true, WorkshopColor::ColorPrimariesId::Rec709));
    cubeViewerChromaticityOverlayPrimaries->setParent(*grpCubeViewerChromaticityCm);
    if (const char* hint = tooltipFor("cubeViewerChromaticityOverlayPrimaries")) cubeViewerChromaticityOverlayPrimaries->setHint(hint);

    auto* grpCubeViewerSlicing = d.defineGroupParam("grp_cube_viewer_slicing");
    grpCubeViewerSlicing->setLabel("Volume Slice");
    grpCubeViewerSlicing->setOpen(false);

    auto* cubeViewerLassoRegionMode = d.defineBooleanParam("cubeViewerLassoRegionMode");
    cubeViewerLassoRegionMode->setLabel("Lasso Region");
    cubeViewerLassoRegionMode->setDefault(false);
    cubeViewerLassoRegionMode->setParent(*grpCubeViewerSlicing);
    cubeViewerLassoRegionMode->setIsSecret(true);
    cubeViewerLassoRegionMode->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerLassoRegionMode")) cubeViewerLassoRegionMode->setHint(hint);

    auto* cubeViewerSliceRed = d.defineBooleanParam("cubeViewerSliceRed");
    cubeViewerSliceRed->setLabel("Red");
    cubeViewerSliceRed->setDefault(false);
    cubeViewerSliceRed->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceRed")) cubeViewerSliceRed->setHint(hint);

    auto* cubeViewerSliceYellow = d.defineBooleanParam("cubeViewerSliceYellow");
    cubeViewerSliceYellow->setLabel("Yellow");
    cubeViewerSliceYellow->setDefault(false);
    cubeViewerSliceYellow->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceYellow")) cubeViewerSliceYellow->setHint(hint);

    auto* cubeViewerSliceGreen = d.defineBooleanParam("cubeViewerSliceGreen");
    cubeViewerSliceGreen->setLabel("Green");
    cubeViewerSliceGreen->setDefault(false);
    cubeViewerSliceGreen->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceGreen")) cubeViewerSliceGreen->setHint(hint);

    auto* cubeViewerSliceCyan = d.defineBooleanParam("cubeViewerSliceCyan");
    cubeViewerSliceCyan->setLabel("Cyan");
    cubeViewerSliceCyan->setDefault(false);
    cubeViewerSliceCyan->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceCyan")) cubeViewerSliceCyan->setHint(hint);

    auto* cubeViewerSliceBlue = d.defineBooleanParam("cubeViewerSliceBlue");
    cubeViewerSliceBlue->setLabel("Blue");
    cubeViewerSliceBlue->setDefault(false);
    cubeViewerSliceBlue->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceBlue")) cubeViewerSliceBlue->setHint(hint);

    auto* cubeViewerSliceMagenta = d.defineBooleanParam("cubeViewerSliceMagenta");
    cubeViewerSliceMagenta->setLabel("Magenta");
    cubeViewerSliceMagenta->setDefault(false);
    cubeViewerSliceMagenta->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerSliceMagenta")) cubeViewerSliceMagenta->setHint(hint);

    auto* cubeViewerNeutralRadius = d.defineDoubleParam("cubeViewerNeutralRadius");
    cubeViewerNeutralRadius->setLabel("Neutral Radius");
    cubeViewerNeutralRadius->setRange(0.0, 1.0);
    cubeViewerNeutralRadius->setDisplayRange(0.0, 1.0);
    cubeViewerNeutralRadius->setDefault(1.0);
    cubeViewerNeutralRadius->setParent(*grpCubeViewerSlicing);
    if (const char* hint = tooltipFor("cubeViewerNeutralRadius")) cubeViewerNeutralRadius->setHint(hint);

    auto* cubeViewerLassoOperation = d.defineChoiceParam("cubeViewerLassoOperation");
    cubeViewerLassoOperation->setLabel("Lasso Operation");
    cubeViewerLassoOperation->appendOption("Add");
    cubeViewerLassoOperation->appendOption("Subtract");
    cubeViewerLassoOperation->setDefault(0);
    cubeViewerLassoOperation->setParent(*grpCubeViewerSlicing);
    cubeViewerLassoOperation->setIsSecret(true);
    cubeViewerLassoOperation->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerLassoOperation")) cubeViewerLassoOperation->setHint(hint);

    auto* cubeViewerLassoUndo = d.definePushButtonParam("cubeViewerLassoUndo");
    cubeViewerLassoUndo->setLabel("Undo Last Stroke");
    cubeViewerLassoUndo->setParent(*grpCubeViewerSlicing);
    cubeViewerLassoUndo->setIsSecret(true);
    cubeViewerLassoUndo->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerLassoUndo")) cubeViewerLassoUndo->setHint(hint);

    auto* cubeViewerLassoReset = d.definePushButtonParam("cubeViewerLassoReset");
    cubeViewerLassoReset->setLabel("Reset Region");
    cubeViewerLassoReset->setParent(*grpCubeViewerSlicing);
    cubeViewerLassoReset->setIsSecret(true);
    cubeViewerLassoReset->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerLassoReset")) cubeViewerLassoReset->setHint(hint);

    auto* cubeViewerLassoData = d.defineStringParam("cubeViewerLassoData");
    cubeViewerLassoData->setDefault("");
    cubeViewerLassoData->setIsSecret(true);
    cubeViewerLassoData->setEnabled(false);
    cubeViewerLassoData->setAnimates(false);

    auto* cubeViewerStatus = d.defineStringParam("cubeViewerStatus");
    cubeViewerStatus->setLabel("Viewer Status");
    cubeViewerStatus->setStringType(eStringTypeLabel);
    cubeViewerStatus->setDefault("Disconnected");
    cubeViewerStatus->setAnimates(false);
    cubeViewerStatus->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerStatus")) cubeViewerStatus->setHint(hint);

    auto* grpCubeViewerIdentityOverlay = d.defineGroupParam("grp_cube_viewer_identity_overlay");
    grpCubeViewerIdentityOverlay->setLabel("Identity Plot");
    grpCubeViewerIdentityOverlay->setOpen(false);

    auto* cubeViewerDrawOnImageEnabled = d.defineBooleanParam("cubeViewerDrawOnImageEnabled");
    cubeViewerDrawOnImageEnabled->setDefault(false);
    cubeViewerDrawOnImageEnabled->setIsSecret(true);

    auto* cubeViewerIdentityOverlayNote = d.defineStringParam("cubeViewerIdentityOverlayNote");
    cubeViewerIdentityOverlayNote->setLabel("Note!");
    cubeViewerIdentityOverlayNote->setStringType(eStringTypeLabel);
    cubeViewerIdentityOverlayNote->setDefault("Requires Generator Upstream");
    cubeViewerIdentityOverlayNote->setAnimates(false);
    cubeViewerIdentityOverlayNote->setEnabled(false);
    cubeViewerIdentityOverlayNote->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlayNote")) cubeViewerIdentityOverlayNote->setHint(hint);

    auto* cubeViewerIdentityOverlayEnabledDraw = d.defineBooleanParam("cubeViewerIdentityOverlayEnabledDraw");
    cubeViewerIdentityOverlayEnabledDraw->setLabel("Generate Identity Plot");
    cubeViewerIdentityOverlayEnabledDraw->setDefault(false);
    cubeViewerIdentityOverlayEnabledDraw->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlayEnabled")) cubeViewerIdentityOverlayEnabledDraw->setHint(hint);

    auto* cubeViewerIdentityOverlayRampDraw = d.defineBooleanParam("cubeViewerIdentityOverlayRampDraw");
    cubeViewerIdentityOverlayRampDraw->setLabel("Draw Gray Ramp");
    cubeViewerIdentityOverlayRampDraw->setDefault(false);
    cubeViewerIdentityOverlayRampDraw->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerIdentityOverlayRamp")) cubeViewerIdentityOverlayRampDraw->setHint(hint);

    auto* cubeViewerReadGrayRamp = d.defineBooleanParam("cubeViewerReadGrayRamp");
    cubeViewerReadGrayRamp->setLabel("Read Gray Ramp");
    cubeViewerReadGrayRamp->setDefault(false);
    cubeViewerReadGrayRamp->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerReadGrayRamp")) cubeViewerReadGrayRamp->setHint(hint);

    auto* cubeViewerSampleDrawnCubeOnly = d.defineBooleanParam("cubeViewerSampleDrawnCubeOnly");
    cubeViewerSampleDrawnCubeOnly->setLabel("Read Identity Plot");
    cubeViewerSampleDrawnCubeOnly->setDefault(false);
    cubeViewerSampleDrawnCubeOnly->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerSampleDrawnCubeOnly")) cubeViewerSampleDrawnCubeOnly->setHint(hint);

    auto* cubeViewerShowIdentityOnly = d.defineBooleanParam("cubeViewerShowIdentityOnly");
    cubeViewerShowIdentityOnly->setLabel("Isolate Identity Data");
    cubeViewerShowIdentityOnly->setDefault(false);
    cubeViewerShowIdentityOnly->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerShowIdentityOnly")) cubeViewerShowIdentityOnly->setHint(hint);

    auto* cubeViewerSampleDrawnCubeSize = d.defineIntParam("cubeViewerSampleDrawnCubeSize");
    cubeViewerSampleDrawnCubeSize->setLabel("Resolution");
    cubeViewerSampleDrawnCubeSize->setDefault(std::clamp(chromaspaceDefaultValues.identityReadResolution, 4, 65));
    cubeViewerSampleDrawnCubeSize->setRange(4, 65);
    cubeViewerSampleDrawnCubeSize->setDisplayRange(4, 65);
    cubeViewerSampleDrawnCubeSize->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerSampleDrawnCubeSize")) cubeViewerSampleDrawnCubeSize->setHint(hint);

    auto* cubeViewerModeToggle = d.definePushButtonParam("cubeViewerModeToggle");
    cubeViewerModeToggle->setLabel("Switch to Identity Generator");
    cubeViewerModeToggle->setParent(*grpCubeViewerIdentityOverlay);
    if (const char* hint = tooltipFor("cubeViewerModeToggle")) cubeViewerModeToggle->setHint(hint);

    auto* grpCubeViewer = d.defineGroupParam("grp_cube_viewer");
    grpCubeViewer->setLabel("Settings");
    grpCubeViewer->setOpen(false);

    cubeViewerStatus->setParent(*grpCubeViewer);

    auto* closeCubeViewer = d.definePushButtonParam("closeCubeViewer");
    closeCubeViewer->setLabel("Disconnect Viewer");
    closeCubeViewer->setParent(*grpCubeViewer);
    if (const char* hint = tooltipFor("closeCubeViewer")) closeCubeViewer->setHint(hint);

    auto* cubeViewerLive = d.defineBooleanParam("cubeViewerLive");
    cubeViewerLive->setLabel("Live Update Viewer");
    cubeViewerLive->setDefault(chromaspaceDefaultValues.liveUpdate);
    cubeViewerLive->setParent(*grpCubeViewer);
    if (const char* hint = tooltipFor("cubeViewerLive")) cubeViewerLive->setHint(hint);

    auto* cubeViewerOnTop = d.defineBooleanParam("cubeViewerOnTop");
    cubeViewerOnTop->setLabel("Keep Viewer On Top");
    cubeViewerOnTop->setDefault(chromaspaceDefaultValues.keepOnTop);
    cubeViewerOnTop->setParent(*grpCubeViewer);
    if (const char* hint = tooltipFor("cubeViewerOnTop")) cubeViewerOnTop->setHint(hint);

    auto* grpCubeViewerPerformance = d.defineGroupParam("grp_cube_viewer_performance");
    grpCubeViewerPerformance->setLabel("Performance");
    grpCubeViewerPerformance->setOpen(true);
    grpCubeViewerPerformance->setParent(*grpCubeViewer);

    auto* cubeViewerUpdateMode = d.defineChoiceParam("cubeViewerUpdateMode");
    cubeViewerUpdateMode->setLabel("Viewer Update Mode");
    cubeViewerUpdateMode->appendOption("Auto");
    cubeViewerUpdateMode->appendOption("Fluid");
    cubeViewerUpdateMode->appendOption("Scheduled");
    cubeViewerUpdateMode->setDefault(std::clamp(chromaspaceDefaultValues.updateMode, 0, 2));
    cubeViewerUpdateMode->setParent(*grpCubeViewerPerformance);
    if (const char* hint = tooltipFor("cubeViewerUpdateMode")) cubeViewerUpdateMode->setHint(hint);

    auto* cubeViewerQuality = d.defineChoiceParam("cubeViewerQuality");
    cubeViewerQuality->setLabel("Viewer Quality");
    cubeViewerQuality->appendOption("Low");
    cubeViewerQuality->appendOption("Medium");
    cubeViewerQuality->appendOption("High");
    cubeViewerQuality->setDefault(std::clamp(chromaspaceDefaultValues.quality, 0, 2));
    cubeViewerQuality->setParent(*grpCubeViewerPerformance);
    if (const char* hint = tooltipFor("cubeViewerQuality")) cubeViewerQuality->setHint(hint);

    auto* cubeViewerScale = d.defineChoiceParam("cubeViewerScale");
    cubeViewerScale->setLabel("Scale");
    cubeViewerScale->appendOption("25%");
    cubeViewerScale->appendOption("50%");
    cubeViewerScale->appendOption("75%");
    cubeViewerScale->appendOption("100%");
    cubeViewerScale->setDefault(std::clamp(chromaspaceDefaultValues.scale, 0, 3));
    cubeViewerScale->setParent(*grpCubeViewerPerformance);
    if (const char* hint = tooltipFor("cubeViewerScale")) cubeViewerScale->setHint(hint);

    auto* grpCubeViewerAppearance = d.defineGroupParam("grp_cube_viewer_appearance");
    grpCubeViewerAppearance->setLabel("Appearance");
    grpCubeViewerAppearance->setOpen(false);
    grpCubeViewerAppearance->setParent(*grpCubeViewer);

    auto* cubeViewerPlotStyle = d.defineChoiceParam("cubeViewerPlotStyle");
    cubeViewerPlotStyle->setLabel("Plot Style");
    cubeViewerPlotStyle->appendOption("Plain Scope");
    cubeViewerPlotStyle->appendOption("Space");
    cubeViewerPlotStyle->setDefault(std::clamp(chromaspaceDefaultValues.plotStyle, 0, 1));
    cubeViewerPlotStyle->setParent(*grpCubeViewerAppearance);

    auto* cubeViewerPointSize = d.defineDoubleParam("cubeViewerPointSize");
    cubeViewerPointSize->setLabel("Point Size");
    cubeViewerPointSize->setDefault(std::clamp(chromaspaceDefaultValues.pointSize, 0.35, 3.0));
    cubeViewerPointSize->setRange(0.35, 3.0);
    cubeViewerPointSize->setDisplayRange(0.35, 3.0);
    cubeViewerPointSize->setIncrement(0.025);
    cubeViewerPointSize->setParent(*grpCubeViewerAppearance);
    if (const char* hint = tooltipFor("cubeViewerPointSize")) cubeViewerPointSize->setHint(hint);

    auto* cubeViewerColorSaturation = d.defineDoubleParam("cubeViewerColorSaturation");
    cubeViewerColorSaturation->setLabel("Color Saturation");
    cubeViewerColorSaturation->setDefault(std::clamp(chromaspaceDefaultValues.colorSaturation, 1.0, 6.0));
    cubeViewerColorSaturation->setRange(1.0, 6.0);
    cubeViewerColorSaturation->setDisplayRange(1.0, 6.0);
    cubeViewerColorSaturation->setIncrement(0.01);
    cubeViewerColorSaturation->setParent(*grpCubeViewerAppearance);
    if (const char* hint = tooltipFor("cubeViewerColorSaturation")) cubeViewerColorSaturation->setHint(hint);

    auto* cubeViewerPointShape = d.defineChoiceParam("cubeViewerPointShape");
    cubeViewerPointShape->setLabel("Point Shape");
    cubeViewerPointShape->appendOption("Circle");
    cubeViewerPointShape->appendOption("Square");
    cubeViewerPointShape->setDefault(std::clamp(chromaspaceDefaultValues.pointShape, 0, 1));
    cubeViewerPointShape->setParent(*grpCubeViewerAppearance);
    if (const char* hint = tooltipFor("cubeViewerPointShape")) cubeViewerPointShape->setHint(hint);

    auto* cubeViewerSamplingMode = d.defineChoiceParam("cubeViewerSamplingMode");
    cubeViewerSamplingMode->setLabel("Sampling");
    cubeViewerSamplingMode->appendOption("Balanced");
    cubeViewerSamplingMode->appendOption("Stratified");
    cubeViewerSamplingMode->appendOption("Random");
    cubeViewerSamplingMode->setDefault(std::clamp(chromaspaceDefaultValues.sampling, 0, 2));
    cubeViewerSamplingMode->setParent(*grpCubeViewerAppearance);
    if (const char* hint = tooltipFor("cubeViewerSamplingMode")) cubeViewerSamplingMode->setHint(hint);

    auto* cubeViewerOccupancyGuidedFill = d.defineBooleanParam("cubeViewerOccupancyGuidedFill");
    cubeViewerOccupancyGuidedFill->setLabel("Occupancy-guided fill");
    cubeViewerOccupancyGuidedFill->setDefault(chromaspaceDefaultValues.occupancyGuidedFill);
    cubeViewerOccupancyGuidedFill->setParent(*grpCubeViewerAppearance);
    if (const char* hint = tooltipFor("cubeViewerOccupancyGuidedFill")) cubeViewerOccupancyGuidedFill->setHint(hint);

    auto* grpChromaspaceDefaultsPresets = d.defineGroupParam("grp_chromaspace_defaults_presets");
    grpChromaspaceDefaultsPresets->setLabel("Defaults & Presets");
    grpChromaspaceDefaultsPresets->setOpen(false);
    grpChromaspaceDefaultsPresets->setParent(*grpCubeViewer);

    auto* chromaspacePresetMenu = d.defineChoiceParam("chromaspacePresetMenu");
    chromaspacePresetMenu->setLabel("Chromaspace Preset");
    chromaspacePresetMenu->appendOption(kChromaspacePresetDefaultName);
    for (const auto& name : visibleChromaspaceUserPresetNames()) chromaspacePresetMenu->appendOption(name);
    chromaspacePresetMenu->setDefault(0);
    chromaspacePresetMenu->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetName = d.defineStringParam("chromaspacePresetName");
    chromaspacePresetName->setLabel("Preset Name");
    chromaspacePresetName->setDefault("");
    chromaspacePresetName->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetSave = d.definePushButtonParam("chromaspacePresetSave");
    chromaspacePresetSave->setLabel("Save Preset");
    chromaspacePresetSave->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetSaveDefaults = d.definePushButtonParam("chromaspacePresetSaveDefaults");
    chromaspacePresetSaveDefaults->setLabel("Save Defaults");
    chromaspacePresetSaveDefaults->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetRestoreDefaults = d.definePushButtonParam("chromaspacePresetRestoreDefaults");
    chromaspacePresetRestoreDefaults->setLabel("Restore Defaults");
    chromaspacePresetRestoreDefaults->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetUpdate = d.definePushButtonParam("chromaspacePresetUpdate");
    chromaspacePresetUpdate->setLabel("Update Preset");
    chromaspacePresetUpdate->setEnabled(false);
    chromaspacePresetUpdate->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetRename = d.definePushButtonParam("chromaspacePresetRename");
    chromaspacePresetRename->setLabel("Rename Preset");
    chromaspacePresetRename->setEnabled(false);
    chromaspacePresetRename->setParent(*grpChromaspaceDefaultsPresets);

    auto* chromaspacePresetDelete = d.definePushButtonParam("chromaspacePresetDelete");
    chromaspacePresetDelete->setLabel("Delete Preset");
    chromaspacePresetDelete->setEnabled(false);
    chromaspacePresetDelete->setParent(*grpChromaspaceDefaultsPresets);

    auto* cubeViewerOverflowHighlightColor = d.defineRGBParam("cubeViewerOverflowHighlightColor");
    cubeViewerOverflowHighlightColor->setLabel("Highlight Color");
    cubeViewerOverflowHighlightColor->setDefault(1.0, 0.0, 0.0);
    cubeViewerOverflowHighlightColor->setRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    cubeViewerOverflowHighlightColor->setDisplayRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    cubeViewerOverflowHighlightColor->setIsSecret(true);
    cubeViewerOverflowHighlightColor->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerOverflowHighlightColor")) cubeViewerOverflowHighlightColor->setHint(hint);

    auto* cubeViewerBackgroundColor = d.defineRGBParam("cubeViewerBackgroundColor");
    cubeViewerBackgroundColor->setLabel("Viewer Background");
    cubeViewerBackgroundColor->setDefault(0.08, 0.08, 0.09);
    cubeViewerBackgroundColor->setRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    cubeViewerBackgroundColor->setDisplayRange(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
    cubeViewerBackgroundColor->setIsSecret(true);
    cubeViewerBackgroundColor->setEnabled(false);
    if (const char* hint = tooltipFor("cubeViewerBackgroundColor")) cubeViewerBackgroundColor->setHint(hint);

    auto* grpSupportRoot = d.defineGroupParam("grp_support_root");
    grpSupportRoot->setLabel("Support");
    grpSupportRoot->setOpen(false);

    auto* supportWebsite = d.definePushButtonParam("supportWebsite");
    supportWebsite->setLabel("Website");
    supportWebsite->setParent(*grpSupportRoot);

    auto* supportLatestReleases = d.definePushButtonParam("supportLatestReleases");
    supportLatestReleases->setLabel("Latest Releases");
    supportLatestReleases->setParent(*grpSupportRoot);

    auto* supportReportIssue = d.definePushButtonParam("supportReportIssue");
    supportReportIssue->setLabel("Submit an Issue");
    supportReportIssue->setParent(*grpSupportRoot);

    auto* supportVersion = d.defineStringParam("supportVersion");
    supportVersion->setLabel("OFX version");
    supportVersion->setDefault(kPluginVersionLabel);
    supportVersion->setEnabled(false);
    supportVersion->setParent(*grpSupportRoot);
  }

  ImageEffect* createInstance(OfxImageEffectHandle h, ContextEnum) override {
    return new ChromaspaceEffect(h);
  }
};

}  // namespace

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& ids) {
  static ChromaspaceFactory p;
  ids.push_back(&p);
}




