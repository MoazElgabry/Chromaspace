#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "color/ColorManagement.h"
#include "text/FontRenderer.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#endif
#include <GLFW/glfw3.h>
#if defined(_WIN32) || defined(__APPLE__)
#include <GLFW/glfw3native.h>
#endif

#if defined(__APPLE__)
#include "metal/ChromaspaceMetal.h"
#elif defined(CHROMASPACE_VIEWER_HAS_CUDA)
#include "cuda/ChromaspaceCuda.h"
#endif

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_LINK_STATUS
#define GL_LINK_STATUS 0x8B82
#endif
#ifndef GL_COMPILE_STATUS
#define GL_COMPILE_STATUS 0x8B81
#endif
#ifndef GL_INFO_LOG_LENGTH
#define GL_INFO_LOG_LENGTH 0x8B84
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#define GL_SHADER_STORAGE_BARRIER_BIT 0x2000
#endif
#ifndef GL_BUFFER_UPDATE_BARRIER_BIT
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x0200
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT 0x00000001
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

namespace {

std::atomic<bool> gRun{true};
std::atomic<bool> gConnected{false};
std::atomic<bool> gBringToFront{false};
std::atomic<int> gWindowVisible{1};
std::atomic<int> gWindowIconified{0};
std::atomic<int> gWindowFocused{1};

void onSignal(int) {
  gRun.store(false);
}

std::string pipeName() {
  const char* env = std::getenv("CHROMASPACE_PIPE");
  if (env && env[0] != '\0') return std::string(env);
#if defined(_WIN32)
  return "\\\\.\\pipe\\Chromaspace";
#else
  return "/tmp/chromaspace.sock";
#endif
}

std::string viewerLogPath() {
#if defined(_WIN32)
  const char* localAppData = std::getenv("LOCALAPPDATA");
  if (localAppData && localAppData[0] != '\0') {
    return (std::filesystem::path(localAppData) / "Chromaspace_CubeViewer.log").string();
  }
  return "Chromaspace_CubeViewer.log";
#elif defined(__APPLE__)
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/Chromaspace.log";
  return std::string(home) + "/Library/Logs/Chromaspace.log";
#else
  const char* home = std::getenv("HOME");
  if (!home || home[0] == '\0') return "/tmp/Chromaspace.log";
  return std::string(home) + "/.cache/Chromaspace.log";
#endif
}

std::string viewerExecutableDir() {
#if defined(_WIN32)
  char buf[MAX_PATH] = {0};
  const DWORD n = GetModuleFileNameA(nullptr, buf, static_cast<DWORD>(sizeof(buf)));
  if (n == 0 || n >= sizeof(buf)) return std::string();
  return std::filesystem::path(std::string(buf, n)).parent_path().string();
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  if (size == 0) return std::filesystem::current_path().string();
  std::vector<char> buf(size + 1u, '\0');
  if (_NSGetExecutablePath(buf.data(), &size) != 0) return std::filesystem::current_path().string();
  std::error_code ec;
  const std::filesystem::path resolved = std::filesystem::weakly_canonical(std::filesystem::path(buf.data()), ec);
  return (ec ? std::filesystem::path(buf.data()) : resolved).parent_path().string();
#else
  std::array<char, 4096> buf{};
  const ssize_t n = ::readlink("/proc/self/exe", buf.data(), buf.size() - 1u);
  if (n <= 0) return std::filesystem::current_path().string();
  buf[static_cast<std::size_t>(n)] = '\0';
  return std::filesystem::path(buf.data()).parent_path().string();
#endif
}

bool nativeShiftModifierPressed();
bool nativeControlModifierPressed();
bool nativeAltModifierPressed();
bool nativeSuperModifierPressed();

bool platformRollModifierPressed(GLFWwindow* window) {
  if (!window) return false;
  if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) return true;
#if defined(__APPLE__)
  return glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
         glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS ||
         nativeSuperModifierPressed();
#else
  return false;
#endif
}

bool nativeShiftModifierPressed() {
#if defined(__APPLE__)
  return (ChromaspaceMetal::currentModifierFlags() & ChromaspaceMetal::ModifierFlagShift) != 0;
#else
  return false;
#endif
}

bool nativeControlModifierPressed() {
#if defined(__APPLE__)
  return (ChromaspaceMetal::currentModifierFlags() & ChromaspaceMetal::ModifierFlagControl) != 0;
#else
  return false;
#endif
}

bool nativeAltModifierPressed() {
#if defined(__APPLE__)
  return (ChromaspaceMetal::currentModifierFlags() & ChromaspaceMetal::ModifierFlagAlt) != 0;
#else
  return false;
#endif
}

bool nativeSuperModifierPressed() {
#if defined(__APPLE__)
  return (ChromaspaceMetal::currentModifierFlags() & ChromaspaceMetal::ModifierFlagSuper) != 0;
#else
  return false;
#endif
}

bool viewerDebugLogEnabled() {
  const char* direct = std::getenv("CHROMASPACE_DEBUG_LOG");
  if (direct && direct[0] != '\0' && std::strcmp(direct, "0") != 0) return true;
  const char* diagnostics = std::getenv("CHROMASPACE_DIAGNOSTICS");
  return diagnostics != nullptr && diagnostics[0] != '\0' && std::strcmp(diagnostics, "0") != 0;
}

void logViewerEvent(const std::string& msg) {
  if (!viewerDebugLogEnabled()) return;
  const std::string path = viewerLogPath();
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

bool viewerDiagnosticsEnabled() {
  const char* env = std::getenv("CHROMASPACE_DIAGNOSTICS");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool viewerEnvFlagEnabled(const char* name, bool defaultValue) {
  const char* env = std::getenv(name);
  if (!env || env[0] == '\0') return defaultValue;
  switch (env[0]) {
    case '0':
    case 'f':
    case 'F':
    case 'n':
    case 'N':
      return false;
    default:
      return true;
  }
}

bool viewerCudaDisabled() {
  return viewerEnvFlagEnabled("CHROMASPACE_DISABLE_CUDA", false);
}

bool viewerParityChecksEnabled() {
  return viewerEnvFlagEnabled("CHROMASPACE_PARITY_CHECK", false);
}

void logViewerDiagnostic(bool enabled, const std::string& msg) {
  if (!enabled) return;
  logViewerEvent(std::string("[diag] ") + msg);
}

#if defined(_WIN32)
void applyWindowsWindowIcon(GLFWwindow* window) {
  if (!window) return;
  HWND hwnd = glfwGetWin32Window(window);
  if (!hwnd) return;
  HMODULE module = GetModuleHandleW(nullptr);
  if (!module) return;
  HICON iconLarge = static_cast<HICON>(LoadImageW(module, L"GLFW_ICON", IMAGE_ICON,
                                                   GetSystemMetrics(SM_CXICON),
                                                   GetSystemMetrics(SM_CYICON), 0));
  HICON iconSmall = static_cast<HICON>(LoadImageW(module, L"GLFW_ICON", IMAGE_ICON,
                                                   GetSystemMetrics(SM_CXSMICON),
                                                   GetSystemMetrics(SM_CYSMICON), 0));
  if (iconLarge) SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(iconLarge));
  if (iconSmall) SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(iconSmall));
}

HANDLE acquireViewerSingletonMutex() {
  HANDLE mutex = CreateMutexW(nullptr, TRUE, L"Local\\Chromaspace_Singleton");
  if (!mutex) return nullptr;
  if (GetLastError() == ERROR_ALREADY_EXISTS) {
    CloseHandle(mutex);
    return nullptr;
  }
  return mutex;
}

void notifyExistingViewerBringToFront() {
  const std::string pipe = pipeName();
  HANDLE pipeHandle = CreateFileA(pipe.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
  if (pipeHandle == INVALID_HANDLE_VALUE) return;
  const char* msg = "{\"type\":\"bring_to_front\",\"senderId\":\"viewer-bootstrap\"}\n";
  DWORD written = 0;
  WriteFile(pipeHandle, msg, static_cast<DWORD>(std::strlen(msg)), &written, nullptr);
  CloseHandle(pipeHandle);
}
#endif

const char* kViewerVersionString = "v1.0.5Beta";

#if !defined(_WIN32)
bool sendAllSocket(int fd, const char* data, size_t size) {
  if (fd < 0 || !data) return false;
  size_t totalSent = 0;
  while (totalSent < size) {
    const ssize_t sent = ::send(fd, data + totalSent, size - totalSent, 0);
    if (sent <= 0) {
      logViewerEvent(std::string("socket send failed after ") + std::to_string(totalSent) + "/" +
                     std::to_string(size) + " bytes: errno=" + std::to_string(errno));
      return false;
    }
    totalSent += static_cast<size_t>(sent);
  }
  return true;
}
#endif

struct CameraState {
  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  float qw = 1.0f;
  float distance = 6.0f;
  float panX = 0.0f;
  float panY = 0.03f;
  bool orthographic = false;
  int orthographicView = -1;
};

struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Quat {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

constexpr float kMinCameraDistance = 0.008f;
constexpr float kMaxCameraDistance = 100.0f;
constexpr float kMinOrthoHalfHeight = 0.25f;
constexpr float kViewerFovYDegrees = 28.0f;

inline float tanHalfFovDegrees(float fovyDegrees) {
  return std::tan(fovyDegrees * 0.5f * 3.14159265358979323846f / 180.0f);
}

inline float minCameraDistanceForView(const CameraState& cam, float fovyDegrees = kViewerFovYDegrees) {
  if (!cam.orthographic) return kMinCameraDistance;
  const float tanHalfFovy = tanHalfFovDegrees(fovyDegrees);
  if (tanHalfFovy <= 1e-6f) return kMinCameraDistance;
  return std::max(kMinCameraDistance, kMinOrthoHalfHeight / tanHalfFovy);
}

inline float computeTightZoomBlend(float distance) {
  // Keep the wide-view feel unchanged, then gradually add close-range assistance
  // once the user pushes into tight inspection distances.
  return clampf((1.75f - distance) / 1.20f, 0.0f, 1.0f);
}

inline float shiftPrecisionFactor() {
  // Keep the precision modifier noticeably slower than the default interaction,
  // especially on macOS where tester feedback showed the previous reduction was too subtle.
  return 0.22f;
}

inline float clamp01(float v) {
  return clampf(v, 0.0f, 1.0f);
}

inline float length3(Vec3 v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline Vec3 normalize3(Vec3 v) {
  const float len = length3(v);
  if (len <= 1e-8f) return Vec3{};
  const float inv = 1.0f / len;
  return Vec3{v.x * inv, v.y * inv, v.z * inv};
}

inline Vec3 cross3(Vec3 a, Vec3 b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float dot3(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Quat normalizeQ(Quat q) {
  const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len <= 1e-8f) return Quat{};
  const float inv = 1.0f / len;
  return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat mulQ(Quat a, Quat b) {
  return Quat{
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat conjugateQ(Quat q) {
  return Quat{-q.x, -q.y, -q.z, q.w};
}

inline Quat axisAngleQ(Vec3 axis, float radians) {
  const Vec3 n = normalize3(axis);
  const float h = radians * 0.5f;
  const float s = std::sin(h);
  return normalizeQ(Quat{n.x * s, n.y * s, n.z * s, std::cos(h)});
}

inline Vec3 rotateVecByQuat(Quat q, Vec3 v) {
  q = normalizeQ(q);
  const Vec3 qv{q.x, q.y, q.z};
  const Vec3 uv = cross3(qv, v);
  const Vec3 uuv = cross3(qv, uv);
  return Vec3{
      v.x + 2.0f * (q.w * uv.x + uuv.x),
      v.y + 2.0f * (q.w * uv.y + uuv.y),
      v.z + 2.0f * (q.w * uv.z + uuv.z)};
}

Vec3 mapArcball(double sx, double sy, int width, int height) {
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const float nx = static_cast<float>((2.0 * sx - static_cast<double>(width)) / static_cast<double>(width));
  const float ny = static_cast<float>((static_cast<double>(height) - 2.0 * sy) / static_cast<double>(height));
  const float d2 = nx * nx + ny * ny;
  // Use a sphere near the center and a hyperbolic sheet outside it so orbit remains responsive
  // even when the cursor travels well beyond the nominal trackball radius.
  if (d2 <= 0.5f) return normalize3(Vec3{nx, ny, std::sqrt(1.0f - d2)});
  const float d = std::sqrt(std::max(d2, 1e-8f));
  return normalize3(Vec3{nx, ny, 0.5f / d});
}

float arcballRadius(double sx, double sy, int width, int height) {
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  const float nx = static_cast<float>((2.0 * sx - static_cast<double>(width)) / static_cast<double>(width));
  const float ny = static_cast<float>((static_cast<double>(height) - 2.0 * sy) / static_cast<double>(height));
  return std::sqrt(nx * nx + ny * ny);
}

void quatToMatrix(Quat q, float out16[16]) {
  q = normalizeQ(q);
  const float xx = q.x * q.x;
  const float yy = q.y * q.y;
  const float zz = q.z * q.z;
  const float xy = q.x * q.y;
  const float xz = q.x * q.z;
  const float yz = q.y * q.z;
  const float wx = q.w * q.x;
  const float wy = q.w * q.y;
  const float wz = q.w * q.z;
  out16[0] = 1.0f - 2.0f * (yy + zz);
  out16[1] = 2.0f * (xy + wz);
  out16[2] = 2.0f * (xz - wy);
  out16[3] = 0.0f;
  out16[4] = 2.0f * (xy - wz);
  out16[5] = 1.0f - 2.0f * (xx + zz);
  out16[6] = 2.0f * (yz + wx);
  out16[7] = 0.0f;
  out16[8] = 2.0f * (xz + wy);
  out16[9] = 2.0f * (yz - wx);
  out16[10] = 1.0f - 2.0f * (xx + yy);
  out16[11] = 0.0f;
  out16[12] = 0.0f;
  out16[13] = 0.0f;
  out16[14] = 0.0f;
  out16[15] = 1.0f;
}

// Default orbit for the cube plot: slightly pitched and yawed so users land in a readable 3D view.
void resetCamera(CameraState* cam) {
  if (!cam) return;
  cam->orthographic = false;
  cam->orthographicView = -1;
  cam->distance = 6.0f;
  cam->panX = 0.0f;
  cam->panY = 0.06f;
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  const Quat qPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, 10.0f * deg2rad);
  const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, -24.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qPitch, qYaw));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

void resetChenCamera(CameraState* cam) {
  resetCamera(cam);
  if (!cam) return;
  cam->distance = 4.8f;
  cam->panY = 0.0f;
}

void resetChromaticityCamera(CameraState* cam) {
  if (!cam) return;
  // Land on the familiar flat chromaticity-chart view by looking straight at the xy plane.
  cam->orthographic = true;
  cam->orthographicView = 0;
  cam->distance = 6.2f;
  cam->panX = 0.0f;
  cam->panY = -0.16f;
  cam->qx = 0.0f;
  cam->qy = 0.0f;
  cam->qz = 0.0f;
  cam->qw = 1.0f;
}

void resetHslCamera(CameraState* cam) {
  resetCamera(cam);
  if (!cam) return;
  cam->distance = 5.2f;
  cam->panY = -0.08f;
}

// Preset camera for cube-like vectorscope reading where the RGB corners feel familiar to grading users.
void resetVectorscopeCamera(CameraState* cam) {
  if (!cam) return;
  cam->orthographic = false;
  cam->orthographicView = -1;
  cam->distance = 6.7f;
  cam->panX = 0.0f;
  cam->panY = 0.0f;
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  // Look roughly down the neutral axis so RGB corners read like a vectorscope.
  const Quat qPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, 35.2643897f * deg2rad);
  const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, -45.0f * deg2rad);
  const Quat qView = normalizeQ(mulQ(qPitch, qYaw));
  // Rotate within the view plane to match the familiar vectorscope orientation:
  // red upper-left, green lower-left, blue right.
  const Quat qRoll = axisAngleQ(Vec3{0.0f, 0.0f, 1.0f}, 135.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qRoll, qView));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

// Polar solids use a different "under the shape" framing so circular hue layouts read more naturally.
void resetPolarVectorscopeCamera(CameraState* cam) {
  if (!cam) return;
  cam->orthographic = false;
  cam->orthographicView = -1;
  cam->distance = 5.7f;
  cam->panX = 0.0f;
  cam->panY = 0.0f;
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  // Look from beneath the polar solid, then roll into a broadcast-style vectorscope layout.
  const Quat qBottom = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, -90.0f * deg2rad);
  const Quat qRoll = axisAngleQ(Vec3{0.0f, 0.0f, 1.0f}, 120.0f * deg2rad);
  const Quat q = normalizeQ(mulQ(qRoll, qBottom));
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

void resetChenVectorscopeCamera(CameraState* cam) {
  resetPolarVectorscopeCamera(cam);
  if (!cam) return;
  cam->distance = 2.35f;
}

void resetChromaticityVectorscopeCamera(CameraState* cam) {
  resetChromaticityCamera(cam);
  if (!cam) return;
  cam->distance = 7.8f;
}

void resetTightPolarVectorscopeCamera(CameraState* cam) {
  resetPolarVectorscopeCamera(cam);
  if (!cam) return;
  cam->distance = 5.0f;
}

void resetHslTopCamera(CameraState* cam) {
  resetPolarVectorscopeCamera(cam);
  if (!cam) return;
  cam->distance = 5.7f;
}

void setCameraQuaternion(CameraState* cam, Quat q) {
  if (!cam) return;
  q = normalizeQ(q);
  cam->qx = q.x;
  cam->qy = q.y;
  cam->qz = q.z;
  cam->qw = q.w;
}

void setOrthographicInspectionCamera(CameraState* cam, int viewIndex) {
  if (!cam) return;
  cam->orthographic = true;
  cam->orthographicView = ((viewIndex % 3) + 3) % 3;
  cam->distance = 6.0f;
  cam->panX = 0.0f;
  cam->panY = 0.0f;
  const float deg2rad = 3.14159265358979323846f / 180.0f;
  switch (cam->orthographicView) {
    case 0:
      setCameraQuaternion(cam, Quat{});
      break;
    case 1:
      setCameraQuaternion(cam, axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, 90.0f * deg2rad));
      break;
    case 2:
    default:
      setCameraQuaternion(cam, axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, -90.0f * deg2rad));
      break;
  }
}

const char* orthographicViewLabel(const CameraState& cam) {
  if (!cam.orthographic) return nullptr;
  switch (cam.orthographicView) {
    case 0: return "Ortho Front";
    case 1: return "Ortho Side";
    case 2: return "Ortho Top";
    default: return "Ortho";
  }
}

using ViewerGLsizeiptr = ptrdiff_t;
using ViewerGLintptr = ptrdiff_t;
using ViewerGLGenBuffersProc = void(APIENTRY *)(GLsizei, GLuint*);
using ViewerGLDeleteBuffersProc = void(APIENTRY *)(GLsizei, const GLuint*);
using ViewerGLBindBufferProc = void(APIENTRY *)(GLenum, GLuint);
using ViewerGLBufferDataProc = void(APIENTRY *)(GLenum, ViewerGLsizeiptr, const void*, GLenum);
using ViewerGLGetBufferSubDataProc = void(APIENTRY *)(GLenum, ViewerGLintptr, ViewerGLsizeiptr, void*);
using ViewerGLCreateShaderProc = GLuint(APIENTRY *)(GLenum);
using ViewerGLShaderSourceProc = void(APIENTRY *)(GLuint, GLsizei, const char* const*, const GLint*);
using ViewerGLCompileShaderProc = void(APIENTRY *)(GLuint);
using ViewerGLGetShaderivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using ViewerGLGetShaderInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using ViewerGLCreateProgramProc = GLuint(APIENTRY *)(void);
using ViewerGLAttachShaderProc = void(APIENTRY *)(GLuint, GLuint);
using ViewerGLLinkProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLGetProgramivProc = void(APIENTRY *)(GLuint, GLenum, GLint*);
using ViewerGLGetProgramInfoLogProc = void(APIENTRY *)(GLuint, GLsizei, GLsizei*, char*);
using ViewerGLDeleteShaderProc = void(APIENTRY *)(GLuint);
using ViewerGLDeleteProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLUseProgramProc = void(APIENTRY *)(GLuint);
using ViewerGLGetUniformLocationProc = GLint(APIENTRY *)(GLuint, const char*);
using ViewerGLUniform1iProc = void(APIENTRY *)(GLint, GLint);
using ViewerGLUniform1fProc = void(APIENTRY *)(GLint, GLfloat);
using ViewerGLBindBufferBaseProc = void(APIENTRY *)(GLenum, GLuint, GLuint);
using ViewerGLDispatchComputeProc = void(APIENTRY *)(GLuint, GLuint, GLuint);
using ViewerGLMemoryBarrierProc = void(APIENTRY *)(GLbitfield);

struct ViewerGlBufferApi {
  bool available = false;
  ViewerGLGenBuffersProc genBuffers = nullptr;
  ViewerGLDeleteBuffersProc deleteBuffers = nullptr;
  ViewerGLBindBufferProc bindBuffer = nullptr;
  ViewerGLBufferDataProc bufferData = nullptr;
  ViewerGLGetBufferSubDataProc getBufferSubData = nullptr;
};

enum class ViewerComputeBackendKind {
  CpuRef,
  GlBuffer,
  GlCompute,
  CudaCompute,
  MetalCompute,
};

constexpr size_t kPlotModeKindCount = 9u;

const char* backendKindLabel(ViewerComputeBackendKind kind) {
  switch (kind) {
    case ViewerComputeBackendKind::CudaCompute: return "cuda-compute-mesh";
    case ViewerComputeBackendKind::GlCompute: return "gl-compute-mesh";
    case ViewerComputeBackendKind::GlBuffer: return "gl-buffer";
    case ViewerComputeBackendKind::MetalCompute: return "metal-compute-mesh";
    case ViewerComputeBackendKind::CpuRef:
    default: return "cpu-ref";
  }
}

struct ViewerGpuCapabilities {
  bool glBufferObjects = false;
  bool glComputeShaders = false;
  bool overlayComputeEnabled = false;
  bool inputComputeEnabled = false;
  bool cudaViewerAvailable = false;
  bool cudaInteropReady = false;
  bool cudaComputeEnabled = false;
  bool cudaStartupValidated = false;
  bool metalViewerAvailable = false;
  bool metalQueueReady = false;
  ViewerComputeBackendKind sessionBackend = ViewerComputeBackendKind::CpuRef;
  std::string activeBackendLabel = "cpu-ref";
  std::string roadmapLabel = "cpu-ref";
  std::string glVersion;
  std::string cudaDeviceName;
  std::string cudaReason;
  std::string cudaStartupReason;
  std::string metalDeviceName;
};

const ViewerGlBufferApi& viewerGlBufferApi() {
  static ViewerGlBufferApi api = []() {
    ViewerGlBufferApi a{};
    a.genBuffers = reinterpret_cast<ViewerGLGenBuffersProc>(glfwGetProcAddress("glGenBuffers"));
    a.deleteBuffers = reinterpret_cast<ViewerGLDeleteBuffersProc>(glfwGetProcAddress("glDeleteBuffers"));
    a.bindBuffer = reinterpret_cast<ViewerGLBindBufferProc>(glfwGetProcAddress("glBindBuffer"));
    a.bufferData = reinterpret_cast<ViewerGLBufferDataProc>(glfwGetProcAddress("glBufferData"));
    a.getBufferSubData = reinterpret_cast<ViewerGLGetBufferSubDataProc>(glfwGetProcAddress("glGetBufferSubData"));
    a.available = a.genBuffers != nullptr && a.deleteBuffers != nullptr && a.bindBuffer != nullptr && a.bufferData != nullptr;
    return a;
  }();
  return api;
}

struct ViewerGlComputeApi {
  bool available = false;
  ViewerGLCreateShaderProc createShader = nullptr;
  ViewerGLShaderSourceProc shaderSource = nullptr;
  ViewerGLCompileShaderProc compileShader = nullptr;
  ViewerGLGetShaderivProc getShaderiv = nullptr;
  ViewerGLGetShaderInfoLogProc getShaderInfoLog = nullptr;
  ViewerGLCreateProgramProc createProgram = nullptr;
  ViewerGLAttachShaderProc attachShader = nullptr;
  ViewerGLLinkProgramProc linkProgram = nullptr;
  ViewerGLGetProgramivProc getProgramiv = nullptr;
  ViewerGLGetProgramInfoLogProc getProgramInfoLog = nullptr;
  ViewerGLDeleteShaderProc deleteShader = nullptr;
  ViewerGLDeleteProgramProc deleteProgram = nullptr;
  ViewerGLUseProgramProc useProgram = nullptr;
  ViewerGLGetUniformLocationProc getUniformLocation = nullptr;
  ViewerGLUniform1iProc uniform1i = nullptr;
  ViewerGLUniform1fProc uniform1f = nullptr;
  ViewerGLBindBufferBaseProc bindBufferBase = nullptr;
  ViewerGLDispatchComputeProc dispatchCompute = nullptr;
  ViewerGLMemoryBarrierProc memoryBarrier = nullptr;
};

const ViewerGlComputeApi& viewerGlComputeApi() {
  static ViewerGlComputeApi api = []() {
    ViewerGlComputeApi a{};
    a.createShader = reinterpret_cast<ViewerGLCreateShaderProc>(glfwGetProcAddress("glCreateShader"));
    a.shaderSource = reinterpret_cast<ViewerGLShaderSourceProc>(glfwGetProcAddress("glShaderSource"));
    a.compileShader = reinterpret_cast<ViewerGLCompileShaderProc>(glfwGetProcAddress("glCompileShader"));
    a.getShaderiv = reinterpret_cast<ViewerGLGetShaderivProc>(glfwGetProcAddress("glGetShaderiv"));
    a.getShaderInfoLog = reinterpret_cast<ViewerGLGetShaderInfoLogProc>(glfwGetProcAddress("glGetShaderInfoLog"));
    a.createProgram = reinterpret_cast<ViewerGLCreateProgramProc>(glfwGetProcAddress("glCreateProgram"));
    a.attachShader = reinterpret_cast<ViewerGLAttachShaderProc>(glfwGetProcAddress("glAttachShader"));
    a.linkProgram = reinterpret_cast<ViewerGLLinkProgramProc>(glfwGetProcAddress("glLinkProgram"));
    a.getProgramiv = reinterpret_cast<ViewerGLGetProgramivProc>(glfwGetProcAddress("glGetProgramiv"));
    a.getProgramInfoLog = reinterpret_cast<ViewerGLGetProgramInfoLogProc>(glfwGetProcAddress("glGetProgramInfoLog"));
    a.deleteShader = reinterpret_cast<ViewerGLDeleteShaderProc>(glfwGetProcAddress("glDeleteShader"));
    a.deleteProgram = reinterpret_cast<ViewerGLDeleteProgramProc>(glfwGetProcAddress("glDeleteProgram"));
    a.useProgram = reinterpret_cast<ViewerGLUseProgramProc>(glfwGetProcAddress("glUseProgram"));
    a.getUniformLocation = reinterpret_cast<ViewerGLGetUniformLocationProc>(glfwGetProcAddress("glGetUniformLocation"));
    a.uniform1i = reinterpret_cast<ViewerGLUniform1iProc>(glfwGetProcAddress("glUniform1i"));
    a.uniform1f = reinterpret_cast<ViewerGLUniform1fProc>(glfwGetProcAddress("glUniform1f"));
    a.bindBufferBase = reinterpret_cast<ViewerGLBindBufferBaseProc>(glfwGetProcAddress("glBindBufferBase"));
    a.dispatchCompute = reinterpret_cast<ViewerGLDispatchComputeProc>(glfwGetProcAddress("glDispatchCompute"));
    a.memoryBarrier = reinterpret_cast<ViewerGLMemoryBarrierProc>(glfwGetProcAddress("glMemoryBarrier"));
    a.available = a.createShader != nullptr && a.shaderSource != nullptr && a.compileShader != nullptr &&
                  a.getShaderiv != nullptr && a.getShaderInfoLog != nullptr && a.createProgram != nullptr &&
                  a.attachShader != nullptr && a.linkProgram != nullptr && a.getProgramiv != nullptr &&
                  a.getProgramInfoLog != nullptr && a.deleteShader != nullptr && a.deleteProgram != nullptr &&
                  a.useProgram != nullptr && a.getUniformLocation != nullptr && a.uniform1i != nullptr &&
                  a.uniform1f != nullptr &&
                  a.bindBufferBase != nullptr && a.dispatchCompute != nullptr && a.memoryBarrier != nullptr;
    return a;
  }();
  return api;
}

std::string currentGlVersionString() {
  const GLubyte* version = glGetString(GL_VERSION);
  return version != nullptr ? reinterpret_cast<const char*>(version) : std::string();
}

std::string readShaderLog(GLuint handle, bool program, const ViewerGlComputeApi& api) {
  GLint logLength = 0;
  if (program) {
    api.getProgramiv(handle, GL_INFO_LOG_LENGTH, &logLength);
  } else {
    api.getShaderiv(handle, GL_INFO_LOG_LENGTH, &logLength);
  }
  if (logLength <= 1) return std::string();
  std::string log(static_cast<size_t>(logLength), '\0');
  GLsizei written = 0;
  if (program) {
    api.getProgramInfoLog(handle, logLength, &written, log.data());
  } else {
    api.getShaderInfoLog(handle, logLength, &written, log.data());
  }
  if (written > 0 && static_cast<size_t>(written) < log.size()) {
    log.resize(static_cast<size_t>(written));
  }
  return log;
}

#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
bool runViewerCudaStartupSelfTest(std::string* reason);
#endif

ViewerGpuCapabilities detectViewerGpuCapabilities() {
  ViewerGpuCapabilities caps{};
  caps.glVersion = currentGlVersionString();
  caps.glBufferObjects = viewerGlBufferApi().available;
  caps.glComputeShaders = viewerGlComputeApi().available;
  const bool overlayRequested = viewerEnvFlagEnabled("CHROMASPACE_OVERLAY_COMPUTE", true);
  const bool inputRequested = viewerEnvFlagEnabled("CHROMASPACE_INPUT_COMPUTE", true);
#if defined(__APPLE__)
  const ChromaspaceMetal::ProbeResult metalProbe = ChromaspaceMetal::probe();
  caps.metalViewerAvailable = metalProbe.available;
  caps.metalQueueReady = metalProbe.queueReady;
  caps.metalDeviceName = metalProbe.deviceName != nullptr ? metalProbe.deviceName : "";
  caps.overlayComputeEnabled = caps.metalQueueReady && overlayRequested;
  caps.inputComputeEnabled = caps.metalQueueReady && inputRequested;
  caps.sessionBackend = caps.metalQueueReady ? ViewerComputeBackendKind::MetalCompute
                                             : (caps.glBufferObjects ? ViewerComputeBackendKind::GlBuffer
                                                                     : ViewerComputeBackendKind::CpuRef);
  caps.activeBackendLabel = (caps.overlayComputeEnabled || caps.inputComputeEnabled)
                                ? "metal-compute-mesh"
                                : (caps.glBufferObjects ? "gl-buffer" : "cpu-ref");
  caps.roadmapLabel = caps.metalQueueReady ? "metal-compute-mesh-phase2" : "metal-phase2";
#else
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (viewerCudaDisabled()) {
    caps.cudaReason = "disabled-by-env";
  } else {
    const ChromaspaceCuda::ProbeResult cudaProbe = ChromaspaceCuda::probe();
    caps.cudaViewerAvailable = cudaProbe.available;
    caps.cudaInteropReady = cudaProbe.interopReady;
    caps.cudaDeviceName = cudaProbe.deviceName != nullptr ? cudaProbe.deviceName : "";
    caps.cudaReason = cudaProbe.reason != nullptr ? cudaProbe.reason : "";
    if (caps.cudaInteropReady && (overlayRequested || inputRequested)) {
      std::string startupReason;
      caps.cudaStartupValidated = runViewerCudaStartupSelfTest(&startupReason);
      caps.cudaStartupReason = startupReason;
      caps.cudaComputeEnabled = caps.cudaStartupValidated;
    }
  }
#endif
  if (caps.cudaComputeEnabled) {
    caps.overlayComputeEnabled = overlayRequested;
    caps.inputComputeEnabled = inputRequested;
    caps.sessionBackend = ViewerComputeBackendKind::CudaCompute;
    caps.activeBackendLabel = backendKindLabel(caps.sessionBackend);
    caps.roadmapLabel = "cuda-compute-mesh-phase3";
  } else {
    caps.overlayComputeEnabled = caps.glComputeShaders && overlayRequested;
    caps.inputComputeEnabled = caps.glComputeShaders && inputRequested;
    caps.sessionBackend = (caps.overlayComputeEnabled || caps.inputComputeEnabled)
                              ? ViewerComputeBackendKind::GlCompute
                              : (caps.glBufferObjects ? ViewerComputeBackendKind::GlBuffer
                                                      : ViewerComputeBackendKind::CpuRef);
    caps.activeBackendLabel = (caps.overlayComputeEnabled || caps.inputComputeEnabled)
                                  ? "gl-compute-mesh"
                                  : (caps.glBufferObjects ? "gl-buffer" : "cpu-ref");
    caps.roadmapLabel = (caps.overlayComputeEnabled || caps.inputComputeEnabled)
                            ? "gl-compute-mesh-phase2"
                            : (caps.glBufferObjects ? "gl-buffer-phase1" : "cpu-ref");
  }
#endif
  return caps;
}

struct MeshData {
  int resolution = 25;
  std::string quality = "Low";
  std::string paramHash;
  bool renderOk = true;
  uint64_t serial = 0;
  std::vector<float> pointVerts;
  std::vector<float> pointColors;
  size_t pointCount = 0;
};

uint64_t nextMeshSerial() {
  static std::atomic<uint64_t> serial{1u};
  return serial.fetch_add(1u, std::memory_order_relaxed);
}

struct PointBufferCache {
  GLuint verts = 0;
  GLuint colors = 0;
  uint64_t uploadedSerial = 0;
  size_t pointCount = 0;
};

struct OverlayComputeCache {
  GLuint input = 0;
  GLuint verts = 0;
  GLuint colors = 0;
  GLuint program = 0;
  GLint cubeSizeLoc = -1;
  GLint rampLoc = -1;
  GLint useInputPointsLoc = -1;
  GLint pointCountLoc = -1;
  GLint plotModeLoc = -1;
  GLint circularHslLoc = -1;
  GLint circularHsvLoc = -1;
  GLint normConeNormalizedLoc = -1;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
};

struct InputCloudComputeCache {
  GLuint input = 0;
  GLuint verts = 0;
  GLuint colors = 0;
  GLuint program = 0;
  GLint pointCountLoc = -1;
  GLint showOverflowLoc = -1;
  GLint highlightOverflowLoc = -1;
  GLint plotModeLoc = -1;
  GLint circularHslLoc = -1;
  GLint circularHsvLoc = -1;
  GLint normConeNormalizedLoc = -1;
  GLint pointAlphaScaleLoc = -1;
  GLint denseAlphaBiasLoc = -1;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
};

#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
struct OverlayCudaCache {
  GLuint verts = 0;
  GLuint colors = 0;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

struct InputCloudCudaCache {
  GLuint verts = 0;
  GLuint colors = 0;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

struct InputCloudSampleCudaCache {
  GLuint verts = 0;
  GLuint colors = 0;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};
#endif

struct PointSelectionSpec {
  uint64_t sourceSerial = 0;
  size_t fullPointCount = 0;
  size_t visiblePointCount = 0;
  bool needsThinning = false;
};

struct PointDrawBuffers {
  GLuint verts = 0;
  GLuint colors = 0;
  size_t pointCount = 0;
  uint64_t sourceSerial = 0;
  size_t visiblePointCount = 0;
  bool available = false;
  std::vector<float> cpuVerts;
  std::vector<float> cpuColors;
};

struct InputCloudSampleComputeCache {
  GLuint verts = 0;
  GLuint colors = 0;
  GLuint program = 0;
  GLint fullPointCountLoc = -1;
  GLint visiblePointCountLoc = -1;
  uint64_t builtSerial = 0;
  GLsizei pointCount = 0;
  size_t visiblePointCount = 0;
  bool available = false;
};

struct InputCloudSampleMetalCache {
  PointDrawBuffers draw;
};

void releasePointBufferCache(PointBufferCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& api = viewerGlBufferApi();
  if (api.available) {
    GLuint buffers[2] = {cache->verts, cache->colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) api.deleteBuffers(count, toDelete);
    api.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  *cache = PointBufferCache{};
}

void releaseOverlayComputeCache(OverlayComputeCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (bufferApi.available) {
    GLuint buffers[3] = {cache->input, cache->verts, cache->colors};
    GLuint toDelete[3] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  if (computeApi.available && cache->program != 0) {
    computeApi.deleteProgram(cache->program);
  }
  *cache = OverlayComputeCache{};
}

void releaseInputCloudComputeCache(InputCloudComputeCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (bufferApi.available) {
    GLuint buffers[3] = {cache->input, cache->verts, cache->colors};
    GLuint toDelete[3] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  if (computeApi.available && cache->program != 0) {
    computeApi.deleteProgram(cache->program);
  }
  *cache = InputCloudComputeCache{};
}

void releaseInputCloudSampleComputeCache(InputCloudSampleComputeCache* cache) {
  if (!cache) return;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (bufferApi.available) {
    GLuint buffers[2] = {cache->verts, cache->colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  if (computeApi.available && cache->program != 0) {
    computeApi.deleteProgram(cache->program);
  }
  *cache = InputCloudSampleComputeCache{};
}

#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
void releaseOverlayCudaCache(OverlayCudaCache* cache) {
  if (!cache) return;
  ChromaspaceCuda::releaseOverlayCache(reinterpret_cast<ChromaspaceCuda::OverlayCache*>(cache));
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (bufferApi.available) {
    GLuint buffers[2] = {cache->verts, cache->colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  *cache = OverlayCudaCache{};
}

void releaseInputCloudCudaCache(InputCloudCudaCache* cache) {
  if (!cache) return;
  ChromaspaceCuda::releaseInputCache(reinterpret_cast<ChromaspaceCuda::InputCache*>(cache));
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (bufferApi.available) {
    GLuint buffers[2] = {cache->verts, cache->colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  *cache = InputCloudCudaCache{};
}

void releaseInputCloudSampleCudaCache(InputCloudSampleCudaCache* cache) {
  if (!cache) return;
  ChromaspaceCuda::releaseInputSampleCache(reinterpret_cast<ChromaspaceCuda::InputSampleCache*>(cache));
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (bufferApi.available) {
    GLuint buffers[2] = {cache->verts, cache->colors};
    GLuint toDelete[2] = {};
    GLsizei count = 0;
    for (GLuint id : buffers) {
      if (id != 0) toDelete[count++] = id;
    }
    if (count > 0) bufferApi.deleteBuffers(count, toDelete);
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  }
  *cache = InputCloudSampleCudaCache{};
}

bool runViewerCudaStartupSelfTest(std::string* reason) {
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  if (!bufferApi.available) {
    if (reason) *reason = "OpenGL buffer API unavailable for CUDA startup validation.";
    return false;
  }
  const ChromaspaceCuda::StartupValidationResult startup = ChromaspaceCuda::validateStartup();
  if (!startup.ready) {
    if (reason) *reason = startup.reason.empty() ? std::string("CUDA startup validation failed.") : startup.reason;
    return false;
  }

  GLuint buffers[4] = {};
  bufferApi.genBuffers(4, buffers);
  for (GLuint id : buffers) {
    if (id == 0) {
      GLuint toDelete[4] = {};
      GLsizei count = 0;
      for (GLuint candidate : buffers) {
        if (candidate != 0) toDelete[count++] = candidate;
      }
      if (count > 0) bufferApi.deleteBuffers(count, toDelete);
      if (reason) *reason = "Failed to allocate startup-validation GL buffers.";
      return false;
    }
  }

  auto allocateBuffer = [&](GLuint id, size_t floatCount) {
    bufferApi.bindBuffer(GL_ARRAY_BUFFER, id);
    bufferApi.bufferData(GL_ARRAY_BUFFER,
                         static_cast<ViewerGLsizeiptr>(floatCount * sizeof(float)),
                         nullptr,
                         GL_DYNAMIC_DRAW);
  };

  ChromaspaceCuda::OverlayCache overlay{};
  overlay.verts = buffers[0];
  overlay.colors = buffers[1];
  ChromaspaceCuda::InputCache input{};
  input.verts = buffers[2];
  input.colors = buffers[3];

  const int cubeSize = 4;
  const size_t overlayPointCount = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) +
                                   static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  allocateBuffer(overlay.verts, overlayPointCount * 3u);
  allocateBuffer(overlay.colors, overlayPointCount * 4u);
  allocateBuffer(input.verts, 4u * 3u);
  allocateBuffer(input.colors, 4u * 4u);
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);

  ChromaspaceCuda::OverlayRequest overlayRequest{};
  overlayRequest.cubeSize = cubeSize;
  overlayRequest.ramp = 1;
  overlayRequest.useInputPoints = 0;
  overlayRequest.pointCount = static_cast<int>(overlayPointCount);
  overlayRequest.remap.plotMode = 0;
  overlayRequest.remap.normConeNormalized = 1;

  std::string localReason;
  if (!ChromaspaceCuda::buildOverlayMesh(&overlay, overlayRequest, {}, 1u, &localReason)) {
    ChromaspaceCuda::releaseOverlayCache(&overlay);
    ChromaspaceCuda::releaseInputCache(&input);
    bufferApi.deleteBuffers(4, buffers);
    if (reason) *reason = localReason.empty() ? std::string("CUDA overlay startup dispatch failed.") : localReason;
    return false;
  }

  const std::vector<float> rawPoints = {
      0.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f
  };
  ChromaspaceCuda::InputRequest inputRequest{};
  inputRequest.pointCount = static_cast<int>(rawPoints.size() / 3u);
  inputRequest.pointAlphaScale = 1.0f;
  inputRequest.remap.plotMode = 0;
  inputRequest.remap.normConeNormalized = 1;

  if (!ChromaspaceCuda::buildInputMesh(&input, inputRequest, rawPoints, 2u, &localReason)) {
    ChromaspaceCuda::releaseOverlayCache(&overlay);
    ChromaspaceCuda::releaseInputCache(&input);
    bufferApi.deleteBuffers(4, buffers);
    if (reason) *reason = localReason.empty() ? std::string("CUDA input startup dispatch failed.") : localReason;
    return false;
  }

  ChromaspaceCuda::releaseOverlayCache(&overlay);
  ChromaspaceCuda::releaseInputCache(&input);
  bufferApi.deleteBuffers(4, buffers);
  return true;
}
#endif

bool ensurePointBufferCacheUploaded(const MeshData& mesh, PointBufferCache* cache) {
  if (!cache) return false;
  const ViewerGlBufferApi& api = viewerGlBufferApi();
  if (!api.available || mesh.pointVerts.empty() || mesh.pointColors.empty()) return false;
  if (cache->uploadedSerial == mesh.serial && cache->verts != 0 && cache->colors != 0) return true;
  if (cache->verts == 0) api.genBuffers(1, &cache->verts);
  if (cache->colors == 0) api.genBuffers(1, &cache->colors);
  if (cache->verts == 0 || cache->colors == 0) return false;
  api.bindBuffer(GL_ARRAY_BUFFER, cache->verts);
  api.bufferData(GL_ARRAY_BUFFER,
                 static_cast<ViewerGLsizeiptr>(mesh.pointVerts.size() * sizeof(float)),
                 mesh.pointVerts.data(),
                 GL_DYNAMIC_DRAW);
  api.bindBuffer(GL_ARRAY_BUFFER, cache->colors);
  api.bufferData(GL_ARRAY_BUFFER,
                 static_cast<ViewerGLsizeiptr>(mesh.pointColors.size() * sizeof(float)),
                 mesh.pointColors.data(),
                 GL_DYNAMIC_DRAW);
  api.bindBuffer(GL_ARRAY_BUFFER, 0);
  cache->uploadedSerial = mesh.serial;
  cache->pointCount = mesh.pointCount;
  return true;
}

uint64_t floatSampleSignature(const float* values, size_t count) {
  if (!values || count == 0) return 0;
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(values[i]), "float bit size mismatch");
    std::memcpy(&bits, &values[i], sizeof(bits));
    hash ^= static_cast<uint64_t>(bits);
    hash *= 1099511628211ull;
  }
  return hash;
}

uint64_t vectorSampleSignature(const std::vector<float>& values, size_t maxCount) {
  return floatSampleSignature(values.data(), std::min(values.size(), maxCount));
}

std::vector<float> gpuBufferSampleValues(GLuint bufferId, size_t floatCount) {
  const ViewerGlBufferApi& api = viewerGlBufferApi();
  if (!api.available || api.getBufferSubData == nullptr || bufferId == 0 || floatCount == 0) return {};
  std::vector<float> sample(floatCount, 0.0f);
  api.bindBuffer(GL_ARRAY_BUFFER, bufferId);
  api.getBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<ViewerGLsizeiptr>(floatCount * sizeof(float)), sample.data());
  api.bindBuffer(GL_ARRAY_BUFFER, 0);
  return sample;
}

bool sampledFloatsNear(const std::vector<float>& cpu, const std::vector<float>& gpu, float epsilon) {
  if (cpu.size() != gpu.size() || cpu.empty()) return false;
  for (size_t i = 0; i < cpu.size(); ++i) {
    if (std::fabs(cpu[i] - gpu[i]) > epsilon) return false;
  }
  return true;
}

void logParityCheckResult(bool enabled, const std::string& label, bool ok, const std::string& detail) {
  if (!enabled) return;
  logViewerDiagnostic(true, std::string("Parity ") + label + ": " + (ok ? "ok" : "mismatch") + " " + detail);
}

std::string pointDrawSourceLabel(bool useInputCudaBuffers,
                                 bool useInputComputeBuffers,
                                 bool usePointBuffers,
                                 bool useSampledCudaBuffers,
                                 bool useSampledComputeBuffers,
                                 bool useSampledCpuArrays,
                                 bool hasCpuArrays) {
  if (useSampledCudaBuffers) return "cuda-sampled";
  if (useSampledComputeBuffers) return "gl-sampled";
  if (useSampledCpuArrays) return "cpu-sampled-array";
  if (useInputCudaBuffers) return "cuda-compute-input";
  if (useInputComputeBuffers) return "gl-compute-input";
  if (usePointBuffers) return "gl-buffer";
  if (hasCpuArrays) return "cpu-array";
  return "none";
}

struct ResolvedPayload;
struct InputCloudPayload;
struct InputCloudSample;
struct PlotRemapSpec;
struct ComputeSessionState;

void drawHslGuide();
void drawCircularHslGuide();
void drawHsvGuide();
void drawCircularHsvGuide();
void drawChenGuide();
void drawMappedBoundaryGuide(const ResolvedPayload& payload);
void drawReuleauxGuide();
void drawRgbGuide(const ResolvedPayload& payload);

bool canUseOverlayComputePath(const ResolvedPayload& payload);
bool ensureOverlayComputeProgram(OverlayComputeCache* cache);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
bool runViewerCudaStartupSelfTest(std::string* reason);
#endif
bool sessionWantsCuda(const ViewerGpuCapabilities& gpuCaps);
bool shouldForceCpuOverlayInCudaSession(const ResolvedPayload& payload,
                                        const ViewerGpuCapabilities& gpuCaps);
bool canUseCudaOverlayPath(const ViewerGpuCapabilities& gpuCaps,
                           const ComputeSessionState* state,
                           const PlotRemapSpec& remap);
bool canUseCudaInputPath(const ViewerGpuCapabilities& gpuCaps,
                         const ComputeSessionState* state,
                         const PlotRemapSpec& remap);
void demoteCudaOverlayPath(const PlotRemapSpec& remap,
                           ComputeSessionState* state,
                           const std::string& reason);
void demoteCudaInputPath(const PlotRemapSpec& remap,
                         ComputeSessionState* state,
                         const std::string& reason);
bool buildIdentityOverlayMeshOnGpu(const ResolvedPayload& payload,
                                   const ViewerGpuCapabilities& gpuCaps,
                                   ComputeSessionState* sessionState,
                                   OverlayComputeCache* cache,
                                   MeshData* out
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                   , OverlayCudaCache* cudaCache
#endif
                                   );
bool canUseInputCloudComputePath(const ResolvedPayload& payload);
bool ensureInputCloudComputeProgram(InputCloudComputeCache* cache);
bool parseInputCloudSamples(const InputCloudPayload& cloud, std::vector<InputCloudSample>* samples);
bool cubeSliceContainsPoint(const PlotRemapSpec& spec, float r, float g, float b);
bool cubeSliceContainsPoint(const ResolvedPayload& payload, float r, float g, float b);
float pointAlphaScaleForPlot(float pointSize, float pointDensity, int resolution);
float denseAlphaBiasForPlot(float pointSize, float pointDensity, int resolution);
float denseLumaProtectedAlpha(float baseAlpha, float pointSize, float pointDensity, int resolution,
                              float cr, float cg, float cb, bool overflowPoint);
float scaledPointAlpha(float baseAlpha, float pointSize, float pointDensity, int resolution);
size_t overlayIdentityPointCap(const ResolvedPayload& payload, int cubeSize);
bool buildInputCloudMeshCpu(const ResolvedPayload& payload,
                            const InputCloudPayload& cloud,
                            const std::vector<float>& rawPoints,
                            MeshData* out);
bool buildInputCloudMeshOnGpu(const ResolvedPayload& payload,
                              const ViewerGpuCapabilities& gpuCaps,
                              ComputeSessionState* sessionState,
                              const InputCloudPayload& cloud,
                              const std::vector<float>& rawPoints,
                              InputCloudComputeCache* cache,
                              MeshData* out
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                              , InputCloudCudaCache* cudaCache
#endif
                              );
float wrapHue01(float h);
float rawRgbHue01(float r, float g, float b, float cMax, float delta);
void rgbToHsvHexconePlane(float r, float g, float b, float* outX, float* outZ);
void rgbToPlotCircularHsl(float r, float g, float b, float* outH, float* outRadius, float* outL);
void rgbToPlotCircularHsv(float r, float g, float b, float* outH, float* outRadius, float* outV);
void rgbToNormConeCoords(float r, float g, float b, bool normalized, bool allowOverflow, float* outHue, float* outChroma, float* outValue);
void rgbToChen(float r, float g, float b, bool allowOverflow, float* outHue, float* outChroma, float* outLight);
void rgbToRgbCone(float r, float g, float b, float* outMagnitude, float* outHue, float* outPolar);
void rgbToJpConical(float r, float g, float b, bool allowOverflow, float* outMagnitude, float* outHue, float* outPolar);
void rgbToReuleaux(float r, float g, float b, bool allowOverflow, float* outHue, float* outSat, float* outValue);
bool runOverlayParityCheck(const ResolvedPayload& payload,
                           const OverlayComputeCache& cache,
                           const MeshData& gpuMesh,
                           bool enabled);
bool runOverlayParityCheckWithBuffers(const ResolvedPayload& payload,
                                      GLuint vertsBuffer,
                                      GLuint colorsBuffer,
                                      const MeshData& gpuMesh,
                                      bool enabled);
bool runInputParityCheck(const ResolvedPayload& payload,
                         const InputCloudPayload& cloud,
                         const std::vector<float>& rawPoints,
                         const InputCloudComputeCache& cache,
                         const MeshData& gpuMesh,
                         bool enabled);
bool runInputParityCheckWithBuffers(const ResolvedPayload& payload,
                                    const InputCloudPayload& cloud,
                                    const std::vector<float>& rawPoints,
                                    GLuint vertsBuffer,
                                    GLuint colorsBuffer,
                                    const MeshData& gpuMesh,
                                    bool enabled);

struct PendingMessage {
  uint64_t seq = 0;
  std::string line;
};

// Plot-model identity is shared across CPU mapping, GPU remap shaders, guide drawing, and future Metal wiring.
// Keeping one canonical classifier here makes backend selection and parity work much easier to reason about.
enum class PlotModeKind {
  Rgb = 0,
  Hsl = 1,
  Hsv = 2,
  Chen = 3,
  RgbToCone = 4,
  JpConical = 5,
  NormCone = 6,
  Reuleaux = 7,
  Chromaticity = 8,
};

struct PlotRemapSpec {
  PlotModeKind plotMode = PlotModeKind::Rgb;
  bool circularHsl = false;
  bool circularHsv = false;
  bool normConeNormalized = true;
  bool showOverflow = false;
  bool highlightOverflow = true;
  bool cubeSlicingEnabled = false;
  bool neutralRadiusEnabled = false;
  float neutralRadius = 1.0f;
  bool cubeSliceRed = true;
  bool cubeSliceGreen = false;
  bool cubeSliceBlue = false;
  bool cubeSliceCyan = false;
  bool cubeSliceYellow = false;
  bool cubeSliceMagenta = false;
  float overflowHighlightR = 1.0f;
  float overflowHighlightG = 0.0f;
  float overflowHighlightB = 0.0f;
  WorkshopColor::ChromaticityColorSpec chromaticity{};
  WorkshopColor::Mat3f chromaticityRgbToXyz{};
  WorkshopColor::Vec2f chromaticityWhite{};
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

struct ComputeSessionState {
  bool overlayCudaFamilyDemoted = false;
  bool inputCudaFamilyDemoted = false;
  bool inputCudaSamplingDemoted = false;
  bool inputGlSamplingDemoted = false;
  bool inputMetalSamplingDemoted = false;
  std::array<bool, kPlotModeKindCount> overlayCudaPlotDemoted{};
  std::array<bool, kPlotModeKindCount> inputCudaPlotDemoted{};
};

struct ComputeRemapUniforms {
  GLint plotMode = 0;
  GLint circularHsl = 0;
  GLint circularHsv = 0;
  GLint normConeNormalized = 1;
  GLint showOverflow = 0;
  GLint highlightOverflow = 1;
};

struct ResolvedPayload {
  uint64_t seq = 0;
  std::string senderId;
  std::string sourceMode = "input";
  std::string plotMode = "rgb";
  std::string cloudSettingsKey;
  bool volumeSlicingEnabled = false;
  std::string volumeSlicingMode = "hue";
  bool lassoRegionEmpty = false;
  std::string lassoData;
  bool circularHsl = false;
  bool circularHsv = false;
  bool normConeNormalized = true;
  bool alwaysOnTop = true;
  std::string quality = "Low";
  int resolution = 25;
  float pointSize = 1.4f;
  float pointDensity = 1.0f;
  std::string pointShape = "Circle";
  bool showOverflow = false;
  bool highlightOverflow = true;
  bool cubeSlicingEnabled = false;
  bool neutralRadiusEnabled = false;
  float neutralRadius = 1.0f;
  bool cubeSliceRed = true;
  bool cubeSliceGreen = false;
  bool cubeSliceBlue = false;
  bool cubeSliceCyan = false;
  bool cubeSliceYellow = false;
  bool cubeSliceMagenta = false;
  float overflowHighlightR = 1.0f;
  float overflowHighlightG = 0.0f;
  float overflowHighlightB = 0.0f;
  float backgroundColorR = 0.08f;
  float backgroundColorG = 0.08f;
  float backgroundColorB = 0.09f;
  bool identityOverlayEnabled = false;
  bool identityOverlayRamp = false;
  bool identityOverlayAuto = true;
  int identityOverlayRequestedSize = 25;
  int identityOverlaySize = 25;
  int chromaticityInputPrimaries = WorkshopColor::primariesChoiceIndex(WorkshopColor::ColorPrimariesId::DavinciWideGamut);
  int chromaticityInputTransfer = WorkshopColor::transferFunctionChoiceIndex(WorkshopColor::TransferFunctionId::DavinciIntermediate);
  int chromaticityReferenceBasis = 0;
  int chromaticityOverlayPrimaries =
      WorkshopColor::overlayPrimariesChoiceIndex(true, WorkshopColor::ColorPrimariesId::Rec709);
  bool chromaticityPlanckianLocus = true;
  std::string version;
};

struct InputCloudPayload {
  uint64_t seq = 0;
  std::string senderId;
  std::string quality = "Low";
  int resolution = 25;
  std::string paramHash;
  std::string settingsKey;
  std::string transport = "json";
  uint64_t pointCount = 0;
  int pointStride = 24;
  std::string shmName;
  uint64_t shmSize = 0;
  std::string points;
  std::vector<float> packedPoints;
};

struct InputCloudSample {
  float xNorm = 0.5f;
  float yNorm = 0.5f;
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
};

bool canUseOverlayComputePath(const ResolvedPayload& payload) {
  return payload.identityOverlayEnabled && payload.plotMode != "chromaticity";
}

PlotModeKind classifyPlotMode(const ResolvedPayload& payload) {
  if (payload.plotMode == "hsl") return PlotModeKind::Hsl;
  if (payload.plotMode == "hsv") return PlotModeKind::Hsv;
  if (payload.plotMode == "chen") return PlotModeKind::Chen;
  if (payload.plotMode == "rgb_to_cone") return PlotModeKind::RgbToCone;
  if (payload.plotMode == "jp_conical") return PlotModeKind::JpConical;
  if (payload.plotMode == "norm_cone") return PlotModeKind::NormCone;
  if (payload.plotMode == "reuleaux") return PlotModeKind::Reuleaux;
  if (payload.plotMode == "chromaticity") return PlotModeKind::Chromaticity;
  return PlotModeKind::Rgb;
}

PlotRemapSpec makePlotRemapSpec(const ResolvedPayload& payload) {
  PlotRemapSpec spec{};
  spec.plotMode = classifyPlotMode(payload);
  spec.circularHsl = payload.circularHsl;
  spec.circularHsv = payload.circularHsv;
  spec.normConeNormalized = payload.normConeNormalized;
  spec.showOverflow = payload.showOverflow;
  spec.highlightOverflow = payload.highlightOverflow;
  spec.cubeSlicingEnabled = payload.cubeSlicingEnabled;
  spec.neutralRadiusEnabled = payload.neutralRadiusEnabled;
  spec.neutralRadius = payload.neutralRadius;
  spec.cubeSliceRed = payload.cubeSliceRed;
  spec.cubeSliceGreen = payload.cubeSliceGreen;
  spec.cubeSliceBlue = payload.cubeSliceBlue;
  spec.cubeSliceCyan = payload.cubeSliceCyan;
  spec.cubeSliceYellow = payload.cubeSliceYellow;
  spec.cubeSliceMagenta = payload.cubeSliceMagenta;
  spec.overflowHighlightR = payload.overflowHighlightR;
  spec.overflowHighlightG = payload.overflowHighlightG;
  spec.overflowHighlightB = payload.overflowHighlightB;
  spec.chromaticity.inputPrimaries =
      WorkshopColor::primariesIdFromChoiceIndex(payload.chromaticityInputPrimaries);
  spec.chromaticity.inputTransfer =
      WorkshopColor::transferFunctionIdFromChoiceIndex(payload.chromaticityInputTransfer);
  spec.chromaticity.referenceBasis =
      payload.chromaticityReferenceBasis == 1
          ? WorkshopColor::ChromaticityReferenceBasis::InputObserver
          : WorkshopColor::ChromaticityReferenceBasis::CieStandardObserver;
  spec.chromaticity.overlayEnabled =
      WorkshopColor::overlayPrimariesChoiceEnabled(payload.chromaticityOverlayPrimaries);
  spec.chromaticity.overlayPrimaries =
      WorkshopColor::overlayPrimariesIdFromChoiceIndex(payload.chromaticityOverlayPrimaries);
  spec.chromaticityRgbToXyz = WorkshopColor::rgbToXyzMatrix(spec.chromaticity.inputPrimaries);
  spec.chromaticityWhite = WorkshopColor::whitePoint(spec.chromaticity.inputPrimaries);
  return spec;
}

PlotRemapSpec makeOverlayPlotRemapSpec(const ResolvedPayload& payload) {
  PlotRemapSpec spec = makePlotRemapSpec(payload);
  if (spec.plotMode == PlotModeKind::Chromaticity) {
    // Synthetic fill/reference lattices should stay in linear light so wide-gamut log encodings do
    // not explode the overlay shape. The real image cloud still uses the selected transfer.
    spec.chromaticity.inputTransfer = WorkshopColor::TransferFunctionId::Linear;
  }
  return spec;
}

int plotModeIndex(const ResolvedPayload& payload) {
  return static_cast<int>(classifyPlotMode(payload));
}

int plotModeIndex(const PlotRemapSpec& spec) {
  return static_cast<int>(spec.plotMode);
}

ComputeRemapUniforms makeComputeRemapUniforms(const PlotRemapSpec& spec) {
  ComputeRemapUniforms uniforms{};
  uniforms.plotMode = static_cast<GLint>(plotModeIndex(spec));
  uniforms.circularHsl = spec.circularHsl ? 1 : 0;
  uniforms.circularHsv = spec.circularHsv ? 1 : 0;
  uniforms.normConeNormalized = spec.normConeNormalized ? 1 : 0;
  uniforms.showOverflow = spec.showOverflow ? 1 : 0;
  uniforms.highlightOverflow = spec.highlightOverflow ? 1 : 0;
  return uniforms;
}

#if defined(__APPLE__)
ChromaspaceMetal::RemapUniforms makeMetalRemapUniforms(const ComputeRemapUniforms& uniforms) {
  ChromaspaceMetal::RemapUniforms metal{};
  metal.plotMode = uniforms.plotMode;
  metal.circularHsl = uniforms.circularHsl;
  metal.circularHsv = uniforms.circularHsv;
  metal.normConeNormalized = uniforms.normConeNormalized;
  metal.showOverflow = uniforms.showOverflow;
  metal.highlightOverflow = uniforms.highlightOverflow;
  return metal;
}
#elif defined(CHROMASPACE_VIEWER_HAS_CUDA)
ChromaspaceCuda::RemapUniforms makeCudaRemapUniforms(const ComputeRemapUniforms& uniforms) {
  ChromaspaceCuda::RemapUniforms cuda{};
  cuda.plotMode = uniforms.plotMode;
  cuda.circularHsl = uniforms.circularHsl;
  cuda.circularHsv = uniforms.circularHsv;
  cuda.normConeNormalized = uniforms.normConeNormalized;
  cuda.showOverflow = uniforms.showOverflow;
  cuda.highlightOverflow = uniforms.highlightOverflow;
  return cuda;
}
#endif

const char* plotModeLabel(const ResolvedPayload& payload) {
  switch (classifyPlotMode(payload)) {
    case PlotModeKind::Hsl: return payload.circularHsl ? "Circular HSL" : "HSL";
    case PlotModeKind::Hsv: return payload.circularHsv ? "Circular HSV" : "HSV";
    case PlotModeKind::Chen: return "Chen";
    case PlotModeKind::RgbToCone: return "RGB to Cone";
    case PlotModeKind::JpConical: return "JP-Conical";
    case PlotModeKind::NormCone: return payload.normConeNormalized ? "Norm-Cone" : "Cone Chroma";
    case PlotModeKind::Reuleaux: return "Reuleaux";
    case PlotModeKind::Chromaticity: return "Chromaticity";
    case PlotModeKind::Rgb:
    default: return "Cube";
  }
}

void drawVolumeSliceHueGuides(const ResolvedPayload& payload);
struct HudTextRenderer;
void drawChromaticityGuide(const ResolvedPayload& payload,
                           const CameraState& cam,
                           int viewportHeight,
                           float fovyDegrees,
                           const HudTextRenderer* hudText = nullptr);

void drawGuideForPlotMode(const ResolvedPayload& payload,
                         const CameraState& cam,
                         int viewportHeight,
                         float fovyDegrees,
                         const HudTextRenderer* hudText = nullptr) {
  switch (classifyPlotMode(payload)) {
    case PlotModeKind::Hsl:
      if (payload.circularHsl) {
        drawCircularHslGuide();
      } else {
        drawHslGuide();
      }
      break;
    case PlotModeKind::Hsv:
      if (payload.circularHsv) {
        drawCircularHsvGuide();
      } else {
        drawHsvGuide();
      }
      break;
    case PlotModeKind::Chen:
      drawChenGuide();
      break;
    case PlotModeKind::NormCone:
    case PlotModeKind::JpConical:
      drawMappedBoundaryGuide(payload);
      break;
    case PlotModeKind::Reuleaux:
      drawMappedBoundaryGuide(payload);
      break;
    case PlotModeKind::RgbToCone:
      drawMappedBoundaryGuide(payload);
      break;
    case PlotModeKind::Chromaticity:
      drawChromaticityGuide(payload, cam, viewportHeight, fovyDegrees, hudText);
      break;
    case PlotModeKind::Rgb:
    default:
      drawRgbGuide(payload);
      break;
  }
  if (payload.cubeSlicingEnabled &&
      classifyPlotMode(payload) != PlotModeKind::Rgb &&
      classifyPlotMode(payload) != PlotModeKind::Chromaticity) {
    drawVolumeSliceHueGuides(payload);
  }
}

bool ensureOverlayComputeProgram(OverlayComputeCache* cache) {
  if (!cache) return false;
  if (cache->program != 0) return true;
  const ViewerGlComputeApi& api = viewerGlComputeApi();
  if (!api.available) return false;
  static const char* kShaderSrc = R"GLSL(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) writeonly buffer VertBuffer { float vertVals[]; };
layout(std430, binding = 1) writeonly buffer ColorBuffer { float colorVals[]; };
layout(std430, binding = 2) readonly buffer InputBuffer { float inputVals[]; };
uniform int uCubeSize;
uniform int uRamp;
uniform int uUseInputPoints;
uniform int uPointCount;
uniform int uPlotMode;
uniform int uCircularHsl;
uniform int uCircularHsv;
uniform int uNormConeNormalized;

const float kTau = 6.28318530717958647692;
const float kPi = 3.14159265358979323846;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

float wrapHue01(float h) {
  h = mod(h, 1.0);
  if (h < 0.0) h += 1.0;
  return h;
}

float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6) return 0.0;
  float h = 0.0;
  if (cMax == r) {
    h = mod((g - b) / delta, 6.0);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0;
  } else {
    h = ((r - g) / delta) + 4.0;
  }
  return wrapHue01(h / 6.0);
}

vec3 mapPlotPosition(float r, float g, float b) {
  if (uPlotMode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    float delta = cMax - cMin;
    float l = 0.5 * (cMax + cMin);
    float h = rawRgbHue01(r, g, b, cMax, delta);
    float satDenom = 1.0 - abs(2.0 * l - 1.0);
    if (delta > 1e-6 && satDenom < 0.0) {
      h = wrapHue01(h + 0.5);
    }
    float angle = h * kTau;
    float radius = delta;
    if (uCircularHsl != 0) {
      float denom = satDenom;
      if (abs(denom) <= 1e-6) {
        denom = (denom < 0.0) ? -1e-6 : 1e-6;
      }
      radius = abs(delta / denom);
    }
    return vec3(cos(angle) * radius, l * 2.0 - 1.0, sin(angle) * radius);
  }
  if (uPlotMode == 2) {
    float cMax = max(r, max(g, b));
    if (uCircularHsv != 0) {
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      float h = rawRgbHue01(r, g, b, cMax, delta);
      float sat = (delta > 1e-6 && cMax > 1e-6) ? (delta / cMax) : 0.0;
      float angle = h * kTau;
      return vec3(cos(angle) * sat, cMax * 2.0 - 1.0, sin(angle) * sat);
    }
    float planeX = r - 0.5 * g - 0.5 * b;
    float planeZ = 0.8660254037844386 * (g - b);
    return vec3(planeX, cMax * 2.0 - 1.0, planeZ);
  }
  if (uPlotMode == 3) {
    float rotX = r * 0.81649658 + g * -0.40824829 + b * -0.40824829;
    float rotY = g * 0.70710678 + b * -0.70710678;
    float rotZ = r * 0.57735027 + g * 0.57735027 + b * 0.57735027;
    float azimuth = atan(rotY, rotX);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float wrappedHue = azimuth < 0.0 ? azimuth + kTau : azimuth;
    float polar = atan(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float c = polar * 1.0467733744265997;
    float l = radius3 * 0.5773502691896258;
    float polarScaled = c * 0.9553166181245093;
    float radial = l * sin(polarScaled) / 0.816496580927726;
    float angle = wrappedHue;
    return vec3(cos(angle) * radial, l * 2.0 - 1.0, sin(angle) * radial);
  }
  if (uPlotMode == 4 || uPlotMode == 5) {
    bool jpOverflow = (uShowOverflow != 0 && uPlotMode == 5);
    float rr = jpOverflow ? r : clamp01(r);
    float gg = jpOverflow ? g : clamp01(g);
    float bb = jpOverflow ? b : clamp01(b);
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float hue = atan(rotY, rotX);
    if (hue < 0.0) hue += kTau;
    float polar = atan(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float magnitude;
    if (uPlotMode == 4) {
      magnitude = clamp(radius3 * 0.576, 0.0, 1.0);
    } else {
      float kAsinInvSqrt2 = asin(1.0 / sqrt(2.0));
      float kAsinInvSqrt3 = asin(1.0 / sqrt(3.0));
      float kHueCoef1 = 1.0 / (2.0 - (kAsinInvSqrt2 / kAsinInvSqrt3));
      float huecoef2 = 2.0 * polar * sin((2.0 * kPi / 3.0) - mod(hue, kPi / 3.0)) / sqrt(3.0);
      float huemag = ((acos(cos(3.0 * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0)) * huecoef2;
      float satmag = sin(huemag + kAsinInvSqrt3);
      magnitude = radius3 * satmag;
      if (jpOverflow && magnitude < 0.0) {
        magnitude = -magnitude;
        hue += kPi;
        if (hue >= kTau) hue -= kTau;
      }
      magnitude = jpOverflow ? magnitude : clamp(magnitude, 0.0, 1.0);
    }
    float phi = (jpOverflow ? max(polar / 0.9553166181245093, 0.0) : clamp(polar / 0.9553166181245093, 0.0, 1.0)) * 0.9553166181245093;
    float radial = magnitude * sin(phi);
    return vec3(cos(hue) * radial, magnitude * cos(phi) * 2.0 - 1.0, sin(hue) * radial);
  }
  if (uPlotMode == 6) {
    bool normConeOverflow = (uShowOverflow != 0 && uPlotMode == 6);
    float rr = normConeOverflow ? r : clamp01(r);
    float gg = normConeOverflow ? g : clamp01(g);
    float bb = normConeOverflow ? b : clamp01(b);
    float maxRgb = max(rr, max(gg, bb));
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan(rotY, rotX) / kTau;
    if (hue < 0.0) hue += 1.0;
    float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
    float polar = atan(chromaRadius, rotZ);
    float chroma = polar / 0.9553166181245093;
    if (uNormConeNormalized != 0) {
      float angle = hue * kTau - kPi / 6.0;
      float cosPolar = cos(polar);
      float safeCos = abs(cosPolar) > 1e-6 ? cosPolar : (cosPolar < 0.0 ? -1e-6 : 1e-6);
      float cone = (sin(polar) / safeCos) / sqrt(2.0);
      float sinTerm = clamp(sin(3.0 * angle), -1.0, 1.0);
      float chromaGain = 1.0 / (2.0 * cos(acos(sinTerm) / 3.0));
      chroma = chromaGain > 1e-6 ? cone / chromaGain : 0.0;
      if (normConeOverflow && chroma < 0.0) {
        chroma = -chroma;
        hue = wrapHue01(hue + 0.5);
      }
    }
    chroma = normConeOverflow ? max(chroma, 0.0) : clamp(chroma, 0.0, 1.0);
    float value = normConeOverflow ? maxRgb : clamp(maxRgb, 0.0, 1.0);
    float angle = hue * kTau;
    return vec3(cos(angle) * chroma, value * 2.0 - 1.0, sin(angle) * chroma);
  }
  if (uPlotMode == 7) {
    bool reuleauxOverflow = (uShowOverflow != 0 && uPlotMode == 7);
    float rr = reuleauxOverflow ? r : clamp01(r);
    float gg = reuleauxOverflow ? g : clamp01(g);
    float bb = reuleauxOverflow ? b : clamp01(b);
    float rotX = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float rotY = (gg - bb) * 0.40824829046;
    float rotZ = (rr + gg + bb) / 3.0;
    float hue = kPi - atan(rotY, -rotX);
    if (hue < 0.0) hue += kTau;
    if (hue >= kTau) hue = mod(hue, kTau);
    float sat = abs(rotZ) <= 1e-6 ? 0.0 : length(vec2(rotX, rotY)) / rotZ;
    if (reuleauxOverflow && sat < 0.0) {
      sat = -sat;
      hue += kPi;
      if (hue >= kTau) hue -= kTau;
    }
    sat = reuleauxOverflow ? sat / 1.41421356237 : clamp(sat / 1.41421356237, 0.0, 1.0);
    float value = reuleauxOverflow ? max(rr, max(gg, bb)) : clamp(max(rr, max(gg, bb)), 0.0, 1.0);
    return vec3(cos(hue) * sat, value * 2.0 - 1.0, sin(hue) * sat);
  }
  return vec3(r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0);
}

void mapDisplayColor(float inR, float inG, float inB, out float outR, out float outG, out float outB) {
  outR = pow(clamp01(inR), 1.0 / 2.2);
  outG = pow(clamp01(inG), 1.0 / 2.2);
  outB = pow(clamp01(inB), 1.0 / 2.2);
}

void main() {
  uint index = gl_GlobalInvocationID.x;
  uint cubeSize = uint(max(uCubeSize, 1));
  uint cubePoints = cubeSize * cubeSize * cubeSize;
  uint rampPoints = (uRamp != 0) ? (cubeSize * cubeSize) : 0u;
  uint uploadedPoints = uint(max(uPointCount, 0));
  uint total = (uUseInputPoints != 0) ? uploadedPoints : (cubePoints + rampPoints);
  if (index >= total) return;
  uint vertBase = index * 3u;
  uint colorBase = index * 4u;
  float r;
  float g;
  float b;
  float alpha;

  if (uUseInputPoints != 0) {
    uint inBase = index * 4u;
    r = inputVals[inBase + 0u];
    g = inputVals[inBase + 1u];
    b = inputVals[inBase + 2u];
    alpha = inputVals[inBase + 3u];
  } else {
    uint denom = max(cubeSize - 1u, 1u);
    if (index < cubePoints) {
      uint plane = cubeSize * cubeSize;
      uint bz = index / plane;
      uint rem = index - bz * plane;
      uint gy = rem / cubeSize;
      uint rx = rem - gy * cubeSize;
      r = float(rx) / float(denom);
      g = float(gy) / float(denom);
      b = float(bz) / float(denom);
      alpha = 0.24;
    } else {
      uint rampIndex = index - cubePoints;
      uint rampDenom = max(rampPoints - 1u, 1u);
      float t = float(rampIndex) / float(rampDenom);
      r = t;
      g = t;
      b = t;
      alpha = 0.92;
    }
  }

  vec3 pos = mapPlotPosition(r, g, b);
  vertVals[vertBase + 0u] = pos.x;
  vertVals[vertBase + 1u] = pos.y;
  vertVals[vertBase + 2u] = pos.z;

  float cr;
  float cg;
  float cb;
  mapDisplayColor(r, g, b, cr, cg, cb);
  colorVals[colorBase + 0u] = cr;
  colorVals[colorBase + 1u] = cg;
  colorVals[colorBase + 2u] = cb;
  colorVals[colorBase + 3u] = alpha;
}
)GLSL";

  const GLuint shader = api.createShader(GL_COMPUTE_SHADER);
  if (shader == 0) return false;
  api.shaderSource(shader, 1, &kShaderSrc, nullptr);
  api.compileShader(shader);
  GLint compiled = 0;
  api.getShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    logViewerEvent(std::string("Overlay compute shader compile failed: ") + readShaderLog(shader, false, api));
    api.deleteShader(shader);
    return false;
  }

  const GLuint program = api.createProgram();
  if (program == 0) {
    api.deleteShader(shader);
    return false;
  }
  api.attachShader(program, shader);
  api.linkProgram(program);
  api.deleteShader(shader);

  GLint linked = 0;
  api.getProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    logViewerEvent(std::string("Overlay compute program link failed: ") + readShaderLog(program, true, api));
    api.deleteProgram(program);
    return false;
  }

  cache->program = program;
  cache->cubeSizeLoc = api.getUniformLocation(program, "uCubeSize");
  cache->rampLoc = api.getUniformLocation(program, "uRamp");
  cache->useInputPointsLoc = api.getUniformLocation(program, "uUseInputPoints");
  cache->pointCountLoc = api.getUniformLocation(program, "uPointCount");
  cache->plotModeLoc = api.getUniformLocation(program, "uPlotMode");
  cache->circularHslLoc = api.getUniformLocation(program, "uCircularHsl");
  cache->circularHsvLoc = api.getUniformLocation(program, "uCircularHsv");
  cache->normConeNormalizedLoc = api.getUniformLocation(program, "uNormConeNormalized");
  cache->available = cache->cubeSizeLoc >= 0 && cache->rampLoc >= 0 &&
                     cache->useInputPointsLoc >= 0 && cache->pointCountLoc >= 0 &&
                     cache->plotModeLoc >= 0 && cache->circularHslLoc >= 0 &&
                     cache->circularHsvLoc >= 0 &&
                     cache->normConeNormalizedLoc >= 0;
  if (!cache->available) {
    logViewerEvent("Overlay compute program missing one or more uniforms; falling back to CPU.");
    releaseOverlayComputeCache(cache);
    return false;
  }
  return true;
}

void appendOverlayRawPoint(const ResolvedPayload& payload,
                           float r, float g, float b, float alpha,
                           std::vector<float>* rawPoints) {
  const PlotRemapSpec remap = makeOverlayPlotRemapSpec(payload);
  if (!rawPoints) return;
  if (!cubeSliceContainsPoint(remap, r, g, b)) return;
  rawPoints->push_back(r);
  rawPoints->push_back(g);
  rawPoints->push_back(b);
  rawPoints->push_back(alpha);
}

void buildIdentityOverlayRawPoints(const ResolvedPayload& payload, std::vector<float>* rawPoints) {
  if (!rawPoints) return;
  rawPoints->clear();
  const int cubeSize = std::clamp(payload.identityOverlaySize, 4, 65);
  const int denom = std::max(1, cubeSize - 1);
  const size_t totalPoints = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  const size_t pointCap = overlayIdentityPointCap(payload, cubeSize);
  const size_t rampSamples = payload.identityOverlayRamp
                                 ? std::min<size_t>(std::max<size_t>(static_cast<size_t>(cubeSize),
                                                                      static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize)),
                                                    65536u)
                                 : 0u;
  rawPoints->reserve((std::min<size_t>(totalPoints, pointCap) + rampSamples) * 4u);
  if (totalPoints <= pointCap) {
    for (int bz = 0; bz < cubeSize; ++bz) {
      for (int gy = 0; gy < cubeSize; ++gy) {
        for (int rx = 0; rx < cubeSize; ++rx) {
          const float r = static_cast<float>(rx) / static_cast<float>(denom);
          const float g = static_cast<float>(gy) / static_cast<float>(denom);
          const float b = static_cast<float>(bz) / static_cast<float>(denom);
          appendOverlayRawPoint(payload, r, g, b, 0.24f, rawPoints);
        }
      }
    }
  } else {
    const size_t width = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
    const size_t height = static_cast<size_t>(cubeSize);
    const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pointCap)))));
    for (size_t i = 0; i < pointCap; ++i) {
      const int gx = static_cast<int>(i % static_cast<size_t>(grid));
      const int gy = static_cast<int>(i / static_cast<size_t>(grid));
      const double u = (static_cast<double>(gx) + 0.5) / static_cast<double>(grid);
      const double v = (static_cast<double>(gy) + 0.5) / static_cast<double>(grid);
      const size_t x = std::min(width - 1, static_cast<size_t>(u * static_cast<double>(width)));
      const size_t y = std::min(height - 1, static_cast<size_t>(v * static_cast<double>(height)));
      const int rx = static_cast<int>(x % static_cast<size_t>(cubeSize));
      const int bz = static_cast<int>(x / static_cast<size_t>(cubeSize));
      const int gyInv = denom - static_cast<int>(y);
      const float r = static_cast<float>(rx) / static_cast<float>(denom);
      const float g = static_cast<float>(std::clamp(gyInv, 0, denom)) / static_cast<float>(denom);
      const float b = static_cast<float>(std::clamp(bz, 0, denom)) / static_cast<float>(denom);
      appendOverlayRawPoint(payload, r, g, b, 0.24f, rawPoints);
    }
  }
  if (payload.identityOverlayRamp) {
    const int rampDenom = std::max(1, static_cast<int>(rampSamples) - 1);
    for (size_t i = 0; i < rampSamples; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(rampDenom);
      appendOverlayRawPoint(payload, t, t, t, 0.92f, rawPoints);
    }
  }
}

bool buildIdentityOverlayMeshOnGpu(const ResolvedPayload& payload,
                                   const ViewerGpuCapabilities& gpuCaps,
                                   ComputeSessionState* sessionState,
                                   OverlayComputeCache* cache,
                                   MeshData* out
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                   , OverlayCudaCache* cudaCache
#endif
                                   ) {
  if (!cache || !out) return false;
  if (!canUseOverlayComputePath(payload)) return false;
  const PlotRemapSpec remap = makePlotRemapSpec(payload);
  const ComputeRemapUniforms uniforms = makeComputeRemapUniforms(remap);
#if defined(__APPLE__)
  std::vector<float> uploadedPoints;
  const bool useUploadedPoints = payload.cubeSlicingEnabled;
  if (useUploadedPoints) {
    buildIdentityOverlayRawPoints(payload, &uploadedPoints);
    if (uploadedPoints.empty()) return false;
  }
  const int cubeSize = std::clamp(payload.identityOverlaySize, 4, 65);
  const size_t pointCount = useUploadedPoints
                              ? (uploadedPoints.size() / 4u)
                              : (static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) +
                                 (payload.identityOverlayRamp ? static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) : 0u));
  ChromaspaceMetal::OverlayRequest request{};
  request.cubeSize = cubeSize;
  request.ramp = payload.identityOverlayRamp ? 1 : 0;
  request.useInputPoints = useUploadedPoints ? 1 : 0;
  request.pointCount = static_cast<int>(pointCount);
  request.remap = makeMetalRemapUniforms(uniforms);
  MeshData mesh{};
  mesh.resolution = payload.resolution;
  mesh.quality = payload.quality;
  mesh.paramHash = payload.cloudSettingsKey.empty() ? payload.version : payload.cloudSettingsKey;
  mesh.serial = nextMeshSerial();
  std::string error;
  if (!ChromaspaceMetal::buildOverlayMesh(request, uploadedPoints, &mesh.pointVerts, &mesh.pointColors, &error)) {
    if (!error.empty()) logViewerEvent(std::string("Metal overlay compute failed: ") + error);
    return false;
  }
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  if (mesh.pointCount == 0 || mesh.pointColors.size() != mesh.pointCount * 4u) return false;
  *out = std::move(mesh);
  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(out->pointCount);
  runOverlayParityCheck(payload, *cache, *out, viewerParityChecksEnabled());
  return true;
#else
  std::vector<float> uploadedPoints;
  const bool useUploadedPoints = payload.cubeSlicingEnabled;
  if (useUploadedPoints) {
    buildIdentityOverlayRawPoints(payload, &uploadedPoints);
    if (uploadedPoints.empty()) return false;
  }
  const int cubeSize = std::clamp(payload.identityOverlaySize, 4, 65);
  const size_t cubePointCount = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  const size_t rampPointCount = payload.identityOverlayRamp
                                    ? std::max<size_t>(static_cast<size_t>(cubeSize),
                                                       static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize))
                                    : 0u;
  const size_t pointCount = useUploadedPoints ? (uploadedPoints.size() / 4u) : (cubePointCount + rampPointCount);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (shouldForceCpuOverlayInCudaSession(payload, gpuCaps)) {
    return false;
  }
  if (canUseCudaOverlayPath(gpuCaps, sessionState, remap) && cudaCache) {
    const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
    auto ensureBuffer = [&](GLuint* id) {
      if (*id == 0) bufferApi.genBuffers(1, id);
      return *id != 0;
    };
    if (!bufferApi.available || !ensureBuffer(&cudaCache->verts) || !ensureBuffer(&cudaCache->colors)) {
      demoteCudaOverlayPath(remap, sessionState, "buffer-allocation");
    } else {
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, cudaCache->verts);
      bufferApi.bufferData(GL_ARRAY_BUFFER,
                           static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                           nullptr,
                           GL_DYNAMIC_DRAW);
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, cudaCache->colors);
      bufferApi.bufferData(GL_ARRAY_BUFFER,
                           static_cast<ViewerGLsizeiptr>(pointCount * 4u * sizeof(float)),
                           nullptr,
                           GL_DYNAMIC_DRAW);
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
      ChromaspaceCuda::OverlayRequest request{};
      request.cubeSize = cubeSize;
      request.ramp = payload.identityOverlayRamp ? 1 : 0;
      request.useInputPoints = useUploadedPoints ? 1 : 0;
      request.pointCount = static_cast<int>(pointCount);
      request.remap = makeCudaRemapUniforms(uniforms);
      std::string error;
      if (ChromaspaceCuda::buildOverlayMesh(reinterpret_cast<ChromaspaceCuda::OverlayCache*>(cudaCache),
                                             request,
                                             uploadedPoints,
                                             nextMeshSerial(),
                                             &error)) {
        MeshData mesh{};
        mesh.resolution = cubeSize;
        mesh.quality = payload.identityOverlayAuto ? "Overlay Auto" : "Overlay";
        mesh.paramHash = "identity_overlay_cuda";
        mesh.serial = cudaCache->builtSerial;
        mesh.pointCount = pointCount;
        *out = std::move(mesh);
        const bool parityOk = runOverlayParityCheckWithBuffers(payload, cudaCache->verts, cudaCache->colors, *out, viewerParityChecksEnabled());
        if (!parityOk) {
          demoteCudaOverlayPath(remap, sessionState, "parity-mismatch");
          return false;
        }
        return true;
      }
      demoteCudaOverlayPath(remap, sessionState, error.empty() ? std::string("runtime-failure") : error);
    }
    return false;
  }
#endif
  if (sessionWantsCuda(gpuCaps)) return false;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (!bufferApi.available || !computeApi.available || !ensureOverlayComputeProgram(cache)) return false;
  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if ((useUploadedPoints && !ensureBuffer(&cache->input)) ||
      !ensureBuffer(&cache->verts) || !ensureBuffer(&cache->colors)) {
    logViewerEvent("Overlay compute buffer allocation failed; falling back to CPU.");
    return false;
  }

  if (useUploadedPoints) {
    bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->input);
    bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                         static_cast<ViewerGLsizeiptr>(uploadedPoints.size() * sizeof(float)),
                         uploadedPoints.data(),
                         GL_DYNAMIC_DRAW);
  }
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->verts);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->colors);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 4u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);

  computeApi.useProgram(cache->program);
  computeApi.uniform1i(cache->cubeSizeLoc, cubeSize);
  computeApi.uniform1i(cache->rampLoc, payload.identityOverlayRamp ? 1 : 0);
  computeApi.uniform1i(cache->useInputPointsLoc, useUploadedPoints ? 1 : 0);
  computeApi.uniform1i(cache->pointCountLoc, static_cast<GLint>(pointCount));
  computeApi.uniform1i(cache->plotModeLoc, uniforms.plotMode);
  computeApi.uniform1i(cache->circularHslLoc, uniforms.circularHsl);
  computeApi.uniform1i(cache->circularHsvLoc, uniforms.circularHsv);
  computeApi.uniform1i(cache->normConeNormalizedLoc, uniforms.normConeNormalized);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cache->verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cache->colors);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, useUploadedPoints ? cache->input : 0);
  const GLuint groups = static_cast<GLuint>((pointCount + 63u) / 64u);
  computeApi.dispatchCompute(groups, 1, 1);
  computeApi.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
  computeApi.useProgram(0);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  MeshData mesh{};
  mesh.resolution = cubeSize;
  mesh.quality = payload.identityOverlayAuto ? "Overlay Auto" : "Overlay";
  mesh.paramHash = "identity_overlay_gpu";
  mesh.serial = nextMeshSerial();
  mesh.pointCount = pointCount;
  *out = std::move(mesh);
  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(pointCount);
  runOverlayParityCheck(payload, *cache, *out, viewerParityChecksEnabled());
  return true;
#endif
}

bool canUseInputCloudComputePath(const ResolvedPayload& payload) {
  return payload.plotMode != "chromaticity";
}

size_t plotModeSlot(const PlotRemapSpec& remap) {
  return std::min(static_cast<size_t>(plotModeIndex(remap)), kPlotModeKindCount - 1u);
}

bool sessionWantsCuda(const ViewerGpuCapabilities& gpuCaps) {
  return gpuCaps.sessionBackend == ViewerComputeBackendKind::CudaCompute;
}

bool shouldForceCpuOverlayInCudaSession(const ResolvedPayload& payload,
                                        const ViewerGpuCapabilities& gpuCaps) {
  return sessionWantsCuda(gpuCaps) && payload.sourceMode == "input";
}

bool canUseCudaOverlayPath(const ViewerGpuCapabilities& gpuCaps,
                           const ComputeSessionState* state,
                           const PlotRemapSpec& remap) {
  if (!sessionWantsCuda(gpuCaps) || !gpuCaps.cudaComputeEnabled || !state) return false;
  const size_t slot = plotModeSlot(remap);
  return !state->overlayCudaFamilyDemoted && !state->overlayCudaPlotDemoted[slot];
}

bool canUseCudaInputPath(const ViewerGpuCapabilities& gpuCaps,
                         const ComputeSessionState* state,
                         const PlotRemapSpec& remap) {
  if (!sessionWantsCuda(gpuCaps) || !gpuCaps.cudaComputeEnabled || !state) return false;
  const size_t slot = plotModeSlot(remap);
  return !state->inputCudaFamilyDemoted && !state->inputCudaPlotDemoted[slot];
}

void demoteCudaOverlayPath(const PlotRemapSpec& remap,
                           ComputeSessionState* state,
                           const std::string& reason) {
  if (!state) return;
  state->overlayCudaFamilyDemoted = true;
  state->overlayCudaPlotDemoted[plotModeSlot(remap)] = true;
  logViewerEvent(std::string("CUDA overlay demoted to CPU: ") + reason);
}

void demoteCudaInputPath(const PlotRemapSpec& remap,
                         ComputeSessionState* state,
                         const std::string& reason) {
  if (!state) return;
  state->inputCudaFamilyDemoted = true;
  state->inputCudaPlotDemoted[plotModeSlot(remap)] = true;
  logViewerEvent(std::string("CUDA input demoted to CPU: ") + reason);
}

std::string overlayComputeReason(const ResolvedPayload& payload,
                                 const ViewerGpuCapabilities& gpuCaps,
                                 const ComputeSessionState* state) {
  if (!payload.identityOverlayEnabled) return "overlay-off";
  if (classifyPlotMode(payload) == PlotModeKind::Chromaticity) return "cpu-chromaticity";
#if defined(__APPLE__)
  if (!gpuCaps.overlayComputeEnabled) return gpuCaps.metalQueueReady ? "overlay-metal-disabled" : "no-metal-compute";
#else
  if (sessionWantsCuda(gpuCaps)) {
    if (!gpuCaps.cudaComputeEnabled) {
      const std::string reason = !gpuCaps.cudaStartupReason.empty() ? gpuCaps.cudaStartupReason : gpuCaps.cudaReason;
      return "cuda-fallback-" + (reason.empty() ? std::string("startup") : reason);
    }
    if (shouldForceCpuOverlayInCudaSession(payload, gpuCaps)) {
      return "cuda-cpu-overlay-fallback-input-mode";
    }
    const PlotRemapSpec remap = makePlotRemapSpec(payload);
    if (state) {
      if (state->overlayCudaFamilyDemoted) return "cuda-demoted-family";
      if (state->overlayCudaPlotDemoted[plotModeSlot(remap)]) return "cuda-demoted-plot";
    }
    return "cuda-eligible";
  }
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (gpuCaps.cudaViewerAvailable && !gpuCaps.cudaInteropReady) {
    return "cuda-fallback-" + (gpuCaps.cudaReason.empty() ? std::string("no-interop") : gpuCaps.cudaReason);
  }
#endif
  if (!gpuCaps.overlayComputeEnabled) return gpuCaps.glComputeShaders ? "overlay-compute-disabled" : "no-gl-compute";
#endif
  return "eligible";
}

std::string inputCloudComputeReason(const ResolvedPayload& payload,
                                    const ViewerGpuCapabilities& gpuCaps,
                                    const ComputeSessionState* state) {
  if (payload.sourceMode != "input") return std::string("source-") + payload.sourceMode;
  if (classifyPlotMode(payload) == PlotModeKind::Chromaticity) return "cpu-chromaticity";
#if defined(__APPLE__)
  if (!gpuCaps.inputComputeEnabled) return gpuCaps.metalQueueReady ? "input-metal-disabled" : "no-metal-compute";
#else
  if (sessionWantsCuda(gpuCaps)) {
    if (!gpuCaps.cudaComputeEnabled) {
      const std::string reason = !gpuCaps.cudaStartupReason.empty() ? gpuCaps.cudaStartupReason : gpuCaps.cudaReason;
      return "cuda-fallback-" + (reason.empty() ? std::string("startup") : reason);
    }
    const PlotRemapSpec remap = makePlotRemapSpec(payload);
    if (state) {
      if (state->inputCudaFamilyDemoted) return "cuda-demoted-family";
      if (state->inputCudaPlotDemoted[plotModeSlot(remap)]) return "cuda-demoted-plot";
    }
    return "cuda-eligible";
  }
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (gpuCaps.cudaViewerAvailable && !gpuCaps.cudaInteropReady) {
    return "cuda-fallback-" + (gpuCaps.cudaReason.empty() ? std::string("no-interop") : gpuCaps.cudaReason);
  }
#endif
  if (!gpuCaps.inputComputeEnabled) return gpuCaps.glComputeShaders ? "input-compute-disabled" : "no-gl-compute";
#endif
  return "eligible";
}

bool ensureInputCloudComputeProgram(InputCloudComputeCache* cache) {
  if (!cache) return false;
  if (cache->program != 0) return true;
  const ViewerGlComputeApi& api = viewerGlComputeApi();
  if (!api.available) return false;
  static const char* kShaderSrc = R"GLSL(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer InputBuffer { float inputVals[]; };
layout(std430, binding = 1) writeonly buffer VertBuffer { float vertVals[]; };
layout(std430, binding = 2) writeonly buffer ColorBuffer { float colorVals[]; };
uniform int uPointCount;
uniform int uShowOverflow;
uniform int uHighlightOverflow;
uniform int uPlotMode;
uniform int uCircularHsl;
uniform int uCircularHsv;
uniform int uNormConeNormalized;
uniform float uPointAlphaScale;
uniform float uDenseAlphaBias;

const float kTau = 6.28318530717958647692;
const float kPi = 3.14159265358979323846;

float clamp01(float v) {
  return clamp(v, 0.0, 1.0);
}

float wrapHue01(float h) {
  h = mod(h, 1.0);
  if (h < 0.0) h += 1.0;
  return h;
}

float rawRgbHue01(float r, float g, float b, float cMax, float delta) {
  if (delta <= 1e-6) return 0.0;
  float h = 0.0;
  if (cMax == r) {
    h = mod((g - b) / delta, 6.0);
  } else if (cMax == g) {
    h = ((b - r) / delta) + 2.0;
  } else {
    h = ((r - g) / delta) + 4.0;
  }
  return wrapHue01(h / 6.0);
}

vec3 mapPlotPosition(float r, float g, float b) {
  if (uPlotMode == 1) {
    float cMax = max(r, max(g, b));
    float cMin = min(r, min(g, b));
    float delta = cMax - cMin;
    float l = 0.5 * (cMax + cMin);
    float h = rawRgbHue01(r, g, b, cMax, delta);
    float satDenom = 1.0 - abs(2.0 * l - 1.0);
    if (delta > 1e-6 && satDenom < 0.0) {
      h = wrapHue01(h + 0.5);
    }
    float angle = h * kTau;
    float radius = delta;
    if (uCircularHsl != 0) {
      float denom = satDenom;
      if (abs(denom) <= 1e-6) {
        denom = (denom < 0.0) ? -1e-6 : 1e-6;
      }
      radius = abs(delta / denom);
    }
    return vec3(cos(angle) * radius, l * 2.0 - 1.0, sin(angle) * radius);
  }
  if (uPlotMode == 2) {
    float cMax = max(r, max(g, b));
    if (uCircularHsv != 0) {
      float cMin = min(r, min(g, b));
      float delta = cMax - cMin;
      float h = rawRgbHue01(r, g, b, cMax, delta);
      float sat = (delta > 1e-6 && cMax > 1e-6) ? (delta / cMax) : 0.0;
      float angle = h * kTau;
      return vec3(cos(angle) * sat, cMax * 2.0 - 1.0, sin(angle) * sat);
    }
    float planeX = r - 0.5 * g - 0.5 * b;
    float planeZ = 0.8660254037844386 * (g - b);
    return vec3(planeX, cMax * 2.0 - 1.0, planeZ);
  }
  if (uPlotMode == 3) {
    float rotX = r * 0.81649658 + g * -0.40824829 + b * -0.40824829;
    float rotY = g * 0.70710678 + b * -0.70710678;
    float rotZ = r * 0.57735027 + g * 0.57735027 + b * 0.57735027;
    float azimuth = atan(rotY, rotX);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float wrappedHue = azimuth < 0.0 ? azimuth + kTau : azimuth;
    float polar = atan(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float c = polar * 1.0467733744265997;
    float l = radius3 * 0.5773502691896258;
    float polarScaled = c * 0.9553166181245093;
    float radial = l * sin(polarScaled) / 0.816496580927726;
    return vec3(cos(wrappedHue) * radial, l * 2.0 - 1.0, sin(wrappedHue) * radial);
  }
  if (uPlotMode == 4 || uPlotMode == 5) {
    float rr = (uShowOverflow != 0 && uPlotMode == 5) ? r : clamp01(r);
    float gg = (uShowOverflow != 0 && uPlotMode == 5) ? g : clamp01(g);
    float bb = (uShowOverflow != 0 && uPlotMode == 5) ? b : clamp01(b);
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float radius3 = sqrt(rotX * rotX + rotY * rotY + rotZ * rotZ);
    float hue = atan(rotY, rotX);
    if (hue < 0.0) hue += kTau;
    float polar = atan(sqrt(rotX * rotX + rotY * rotY), rotZ);
    float magnitude;
    if (uPlotMode == 4) {
      magnitude = clamp(radius3 * 0.576, 0.0, 1.0);
    } else {
      float kAsinInvSqrt2 = asin(1.0 / sqrt(2.0));
      float kAsinInvSqrt3 = asin(1.0 / sqrt(3.0));
      float kHueCoef1 = 1.0 / (2.0 - (kAsinInvSqrt2 / kAsinInvSqrt3));
      float huecoef2 = 2.0 * polar * sin((2.0 * kPi / 3.0) - mod(hue, kPi / 3.0)) / sqrt(3.0);
      float huemag = ((acos(cos(3.0 * hue + kPi))) / (kPi * kHueCoef1) + ((kAsinInvSqrt2 / kAsinInvSqrt3) - 1.0)) * huecoef2;
      float satmag = sin(huemag + kAsinInvSqrt3);
      magnitude = radius3 * satmag;
      if ((uShowOverflow != 0 && uPlotMode == 5) && magnitude < 0.0) {
        magnitude = -magnitude;
        hue += kPi;
        if (hue >= kTau) hue -= kTau;
      }
      magnitude = (uShowOverflow != 0 && uPlotMode == 5) ? magnitude : clamp(magnitude, 0.0, 1.0);
    }
    float phiNorm = (uShowOverflow != 0 && uPlotMode == 5) ? max(polar / 0.9553166181245093, 0.0) : clamp(polar / 0.9553166181245093, 0.0, 1.0);
    float phi = phiNorm * 0.9553166181245093;
    float radial = magnitude * sin(phi);
    return vec3(cos(hue) * radial, magnitude * cos(phi) * 2.0 - 1.0, sin(hue) * radial);
  }
  if (uPlotMode == 6) {
    bool normConeOverflow = (uShowOverflow != 0 && uPlotMode == 6);
    float rr = normConeOverflow ? r : clamp01(r);
    float gg = normConeOverflow ? g : clamp01(g);
    float bb = normConeOverflow ? b : clamp01(b);
    float maxRgb = max(rr, max(gg, bb));
    float rotX = 0.81649658093 * rr - 0.40824829046 * gg - 0.40824829046 * bb;
    float rotY = 0.70710678118 * gg - 0.70710678118 * bb;
    float rotZ = 0.57735026919 * (rr + gg + bb);
    float hue = atan(rotY, rotX) / kTau;
    if (hue < 0.0) hue += 1.0;
    float chromaRadius = sqrt(rotX * rotX + rotY * rotY);
    float polar = atan(chromaRadius, rotZ);
    float chroma = polar / 0.9553166181245093;
    if (uNormConeNormalized != 0) {
      float angle = hue * kTau - kPi / 6.0;
      float cosPolar = cos(polar);
      float safeCos = abs(cosPolar) > 1e-6 ? cosPolar : (cosPolar < 0.0 ? -1e-6 : 1e-6);
      float cone = (sin(polar) / safeCos) / sqrt(2.0);
      float sinTerm = clamp(sin(3.0 * angle), -1.0, 1.0);
      float chromaGain = 1.0 / (2.0 * cos(acos(sinTerm) / 3.0));
      chroma = chromaGain > 1e-6 ? cone / chromaGain : 0.0;
      if (normConeOverflow && chroma < 0.0) {
        chroma = -chroma;
        hue = wrapHue01(hue + 0.5);
      }
    }
    chroma = normConeOverflow ? max(chroma, 0.0) : clamp(chroma, 0.0, 1.0);
    float value = normConeOverflow ? maxRgb : clamp(maxRgb, 0.0, 1.0);
    float angle = hue * kTau;
    return vec3(cos(angle) * chroma, value * 2.0 - 1.0, sin(angle) * chroma);
  }
  if (uPlotMode == 7) {
    bool reuleauxOverflow = (uShowOverflow != 0 && uPlotMode == 7);
    float rr = reuleauxOverflow ? r : clamp01(r);
    float gg = reuleauxOverflow ? g : clamp01(g);
    float bb = reuleauxOverflow ? b : clamp01(b);
    float rotX = 0.33333333333 * (2.0 * rr - gg - bb) * 0.70710678118;
    float rotY = (gg - bb) * 0.40824829046;
    float rotZ = (rr + gg + bb) / 3.0;
    float hue = kPi - atan(rotY, -rotX);
    if (hue < 0.0) hue += kTau;
    if (hue >= kTau) hue = mod(hue, kTau);
    float sat = abs(rotZ) <= 1e-6 ? 0.0 : length(vec2(rotX, rotY)) / rotZ;
    if (reuleauxOverflow && sat < 0.0) {
      sat = -sat;
      hue += kPi;
      if (hue >= kTau) hue -= kTau;
    }
    sat = reuleauxOverflow ? sat / 1.41421356237 : clamp(sat / 1.41421356237, 0.0, 1.0);
    float value = reuleauxOverflow ? max(rr, max(gg, bb)) : clamp(max(rr, max(gg, bb)), 0.0, 1.0);
    return vec3(cos(hue) * sat, value * 2.0 - 1.0, sin(hue) * sat);
  }
  return vec3(r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0);
}

bool outOfBounds(float r, float g, float b) {
  return r < 0.0 || r > 1.0 || g < 0.0 || g > 1.0 || b < 0.0 || b > 1.0;
}

void mapDisplayColor(float inR, float inG, float inB, out float outR, out float outG, out float outB) {
  outR = pow(clamp01(inR), 1.0 / 2.2);
  outG = pow(clamp01(inG), 1.0 / 2.2);
  outB = pow(clamp01(inB), 1.0 / 2.2);
}

float luminanceAwareAlpha(float baseAlpha, float cr, float cg, float cb, bool overflowPoint) {
  float alpha = baseAlpha * uPointAlphaScale;
  if (overflowPoint || uDenseAlphaBias <= 0.0) return clamp(alpha, 0.0, 1.0);
  float luma = clamp(cr * 0.2126 + cg * 0.7152 + cb * 0.0722, 0.0, 1.0);
  float highlightKnee = clamp((luma - 0.70) / 0.24, 0.0, 1.0);
  float shadowMidProtect = 1.0 - clamp((luma - 0.58) / 0.30, 0.0, 1.0);
  float multiplier = clamp(1.0 + 0.16 * uDenseAlphaBias * shadowMidProtect
                               - 0.30 * uDenseAlphaBias * highlightKnee,
                           0.76, 1.12);
  return clamp(alpha * multiplier, 0.0, 1.0);
}

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= uint(max(uPointCount, 0))) return;
  uint inBase = index * 3u;
  float r = inputVals[inBase + 0u];
  float g = inputVals[inBase + 1u];
  float b = inputVals[inBase + 2u];
  bool overflowPoint = outOfBounds(r, g, b);
  float plotR = (uShowOverflow != 0) ? r : clamp01(r);
  float plotG = (uShowOverflow != 0) ? g : clamp01(g);
  float plotB = (uShowOverflow != 0) ? b : clamp01(b);

  uint vertBase = index * 3u;
  uint colorBase = index * 4u;
  vec3 pos = mapPlotPosition(plotR, plotG, plotB);
  vertVals[vertBase + 0u] = pos.x;
  vertVals[vertBase + 1u] = pos.y;
  vertVals[vertBase + 2u] = pos.z;

  float cr;
  float cg;
  float cb;
  if (uShowOverflow != 0 && uHighlightOverflow != 0 && overflowPoint) {
    cr = 1.0;
    cg = 0.0;
    cb = 0.0;
  } else {
    mapDisplayColor(r, g, b, cr, cg, cb);
  }
  colorVals[colorBase + 0u] = cr;
  colorVals[colorBase + 1u] = cg;
  colorVals[colorBase + 2u] = cb;
  colorVals[colorBase + 3u] =
      luminanceAwareAlpha(((uShowOverflow != 0 && uHighlightOverflow != 0 && overflowPoint) ? 0.95 : 0.72),
                          cr, cg, cb, overflowPoint);
}
)GLSL";

  const GLuint shader = api.createShader(GL_COMPUTE_SHADER);
  if (shader == 0) return false;
  api.shaderSource(shader, 1, &kShaderSrc, nullptr);
  api.compileShader(shader);
  GLint compiled = 0;
  api.getShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    logViewerEvent(std::string("Input-cloud compute shader compile failed: ") + readShaderLog(shader, false, api));
    api.deleteShader(shader);
    return false;
  }

  const GLuint program = api.createProgram();
  if (program == 0) {
    api.deleteShader(shader);
    return false;
  }
  api.attachShader(program, shader);
  api.linkProgram(program);
  api.deleteShader(shader);

  GLint linked = 0;
  api.getProgramiv(program, GL_LINK_STATUS, &linked);
  if (!linked) {
    logViewerEvent(std::string("Input-cloud compute program link failed: ") + readShaderLog(program, true, api));
    api.deleteProgram(program);
    return false;
  }

  cache->program = program;
  cache->pointCountLoc = api.getUniformLocation(program, "uPointCount");
  cache->showOverflowLoc = api.getUniformLocation(program, "uShowOverflow");
  cache->highlightOverflowLoc = api.getUniformLocation(program, "uHighlightOverflow");
  cache->plotModeLoc = api.getUniformLocation(program, "uPlotMode");
  cache->circularHslLoc = api.getUniformLocation(program, "uCircularHsl");
  cache->circularHsvLoc = api.getUniformLocation(program, "uCircularHsv");
  cache->normConeNormalizedLoc = api.getUniformLocation(program, "uNormConeNormalized");
  cache->pointAlphaScaleLoc = api.getUniformLocation(program, "uPointAlphaScale");
  cache->denseAlphaBiasLoc = api.getUniformLocation(program, "uDenseAlphaBias");
  cache->available = cache->pointCountLoc >= 0 &&
                     cache->showOverflowLoc >= 0 &&
                     cache->highlightOverflowLoc >= 0 &&
                     cache->plotModeLoc >= 0 &&
                     cache->circularHslLoc >= 0 &&
                     cache->circularHsvLoc >= 0 &&
                     cache->normConeNormalizedLoc >= 0 &&
                     cache->pointAlphaScaleLoc >= 0 &&
                     cache->denseAlphaBiasLoc >= 0;
  if (!cache->available) {
    logViewerEvent("Input-cloud compute program missing one or more uniforms; falling back to CPU.");
    releaseInputCloudComputeCache(cache);
    return false;
  }
  return true;
}

std::vector<std::string> splitViewerString(const std::string& text, char delimiter) {
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

bool parseViewerUint64(const std::string& text, uint64_t* value) {
  if (!value) return false;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

bool parseViewerIntStrict(const std::string& text, int* value) {
  if (!value) return false;
  char* end = nullptr;
  const long parsed = std::strtol(text.c_str(), &end, 10);
  if (!end || *end != '\0') return false;
  *value = static_cast<int>(parsed);
  return true;
}

bool parseViewerFloatStrict(const std::string& text, float* value) {
  if (!value) return false;
  char* end = nullptr;
  const float parsed = std::strtof(text.c_str(), &end);
  if (!end || *end != '\0') return false;
  *value = parsed;
  return true;
}

LassoRegionState parseViewerLassoRegionState(const std::string& serialized) {
  LassoRegionState state{};
  if (serialized.empty()) return state;
  const auto records = splitViewerString(serialized, '|');
  if (records.size() < 2 || records[0] != "v1") return state;
  if (!parseViewerUint64(records[1], &state.revision)) return LassoRegionState{};
  for (size_t i = 2; i < records.size(); ++i) {
    const auto fields = splitViewerString(records[i], ',');
    if (fields.size() < 2 || fields[0].size() != 1) return LassoRegionState{};
    int pointCount = 0;
    if (!parseViewerIntStrict(fields[1], &pointCount) || pointCount < 3) return LassoRegionState{};
    if (fields.size() != static_cast<size_t>(2 + pointCount * 2)) return LassoRegionState{};
    LassoStroke stroke{};
    stroke.subtract = (fields[0][0] == 's' || fields[0][0] == 'S');
    stroke.points.reserve(static_cast<size_t>(pointCount));
    for (int pointIndex = 0; pointIndex < pointCount; ++pointIndex) {
      float xNorm = 0.0f;
      float yNorm = 0.0f;
      if (!parseViewerFloatStrict(fields[2 + pointIndex * 2], &xNorm) ||
          !parseViewerFloatStrict(fields[3 + pointIndex * 2], &yNorm)) {
        return LassoRegionState{};
      }
      stroke.points.push_back({clampf(xNorm, 0.0f, 1.0f), clampf(yNorm, 0.0f, 1.0f)});
    }
    state.strokes.push_back(std::move(stroke));
  }
  return state;
}

bool pointInViewerLassoPolygon(const std::vector<LassoPointNorm>& polygon, double xNorm, double yNorm) {
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

bool viewerLassoContainsPoint(const LassoRegionState& state, double xNorm, double yNorm) {
  bool inside = false;
  for (const auto& stroke : state.strokes) {
    if (!pointInViewerLassoPolygon(stroke.points, xNorm, yNorm)) continue;
    inside = !stroke.subtract;
  }
  return inside;
}

bool loadInputCloudSharedMemory(const InputCloudPayload& cloud, std::vector<float>* outPackedPoints) {
  if (!outPackedPoints) return false;
  outPackedPoints->clear();
  if (cloud.transport != "shm" || cloud.shmName.empty() || cloud.pointCount == 0 || cloud.pointStride <= 0) return false;
  const uint64_t expectedBytes = cloud.pointCount * static_cast<uint64_t>(cloud.pointStride);
  if (expectedBytes == 0 || (cloud.shmSize != 0 && expectedBytes > cloud.shmSize)) return false;
  if (cloud.pointStride != static_cast<int>(6u * sizeof(float))) return false;
  outPackedPoints->assign(static_cast<size_t>(cloud.pointCount) * 6u, 0.0f);
#if defined(_WIN32)
  HANDLE mapping = OpenFileMappingA(FILE_MAP_READ, FALSE, cloud.shmName.c_str());
  if (mapping == nullptr) return false;
  const void* mapped = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, static_cast<SIZE_T>(expectedBytes));
  if (!mapped) {
    CloseHandle(mapping);
    return false;
  }
  std::memcpy(outPackedPoints->data(), mapped, static_cast<size_t>(expectedBytes));
  UnmapViewOfFile(mapped);
  CloseHandle(mapping);
  return true;
#else
  const int fd = shm_open(cloud.shmName.c_str(), O_RDONLY, 0);
  if (fd < 0) return false;
  void* mapped = mmap(nullptr, static_cast<size_t>(expectedBytes), PROT_READ, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    ::close(fd);
    return false;
  }
  std::memcpy(outPackedPoints->data(), mapped, static_cast<size_t>(expectedBytes));
  munmap(mapped, static_cast<size_t>(expectedBytes));
  ::close(fd);
  return true;
#endif
}

bool parseInputCloudSamples(const InputCloudPayload& cloud, std::vector<InputCloudSample>* samples) {
  if (!samples) return false;
  samples->clear();
  if (!cloud.packedPoints.empty()) {
    const size_t pointCount = cloud.packedPoints.size() / 6u;
    samples->reserve(pointCount);
    for (size_t i = 0; i < pointCount; ++i) {
      const size_t base = i * 6u;
      samples->push_back({clampf(cloud.packedPoints[base + 0], 0.0f, 1.0f),
                          clampf(cloud.packedPoints[base + 1], 0.0f, 1.0f),
                          cloud.packedPoints[base + 3],
                          cloud.packedPoints[base + 4],
                          cloud.packedPoints[base + 5]});
    }
    return !samples->empty();
  }
  std::string flattened = cloud.points;
  std::replace(flattened.begin(), flattened.end(), ';', ' ');
  std::istringstream is(flattened);
  float x = 0.0f, y = 0.0f, z = 0.0f, r = 0.0f, g = 0.0f, b = 0.0f;
  while (is >> x >> y >> z >> r >> g >> b) {
    (void)z;
    samples->push_back({clampf(x, 0.0f, 1.0f), clampf(y, 0.0f, 1.0f), r, g, b});
  }
  return !samples->empty();
}

void filterInputCloudSamples(const ResolvedPayload& payload, std::vector<InputCloudSample>* samples) {
  if (!samples || samples->empty()) return;
  const PlotRemapSpec remap = makePlotRemapSpec(payload);
  const bool lassoEnabled = payload.volumeSlicingEnabled && payload.volumeSlicingMode == "lasso";
  const LassoRegionState lassoState = lassoEnabled ? parseViewerLassoRegionState(payload.lassoData) : LassoRegionState{};
  const bool cubeFilterEnabled = remap.cubeSlicingEnabled || remap.neutralRadiusEnabled;
  if (!lassoEnabled && !cubeFilterEnabled) return;
  if (lassoEnabled && (payload.lassoRegionEmpty || lassoState.strokes.empty())) {
    samples->clear();
    return;
  }
  std::vector<InputCloudSample> filtered;
  filtered.reserve(samples->size());
  for (const auto& sample : *samples) {
    if (lassoEnabled && !viewerLassoContainsPoint(lassoState, sample.xNorm, sample.yNorm)) continue;
    const float r = sample.r;
    const float g = sample.g;
    const float b = sample.b;
    if (!cubeSliceContainsPoint(remap, r, g, b)) continue;
    filtered.push_back(sample);
  }
  samples->swap(filtered);
}

bool buildInputCloudMeshOnGpu(const ResolvedPayload& payload,
                              const ViewerGpuCapabilities& gpuCaps,
                              ComputeSessionState* sessionState,
                              const InputCloudPayload& cloud,
                              const std::vector<float>& rawPoints,
                              InputCloudComputeCache* cache,
                              MeshData* out
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                              , InputCloudCudaCache* cudaCache
#endif
                              ) {
  if (!cache || !out) return false;
  if (!canUseInputCloudComputePath(payload)) return false;
  const PlotRemapSpec remap = makePlotRemapSpec(payload);
  const ComputeRemapUniforms uniforms = makeComputeRemapUniforms(remap);
#if defined(__APPLE__)
  const size_t pointCount = rawPoints.size() / 3u;
  if (pointCount == 0) return false;
  ChromaspaceMetal::InputRequest request{};
  request.pointCount = static_cast<int>(pointCount);
  request.pointAlphaScale = pointAlphaScaleForPlot(payload.pointSize, payload.pointDensity, payload.resolution);
  request.denseAlphaBias = denseAlphaBiasForPlot(payload.pointSize, payload.pointDensity, payload.resolution);
  request.remap = makeMetalRemapUniforms(uniforms);
  MeshData mesh{};
  mesh.resolution = cloud.resolution <= 25 ? 25 : (cloud.resolution <= 41 ? 41 : 57);
  mesh.quality = cloud.quality;
  mesh.paramHash = cloud.paramHash;
  mesh.serial = nextMeshSerial();
  std::string error;
  if (!ChromaspaceMetal::buildInputMesh(request, rawPoints, &mesh.pointVerts, &mesh.pointColors, &error)) {
    if (!error.empty()) logViewerEvent(std::string("Metal input compute failed: ") + error);
    return false;
  }
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  if (mesh.pointCount == 0 || mesh.pointColors.size() != mesh.pointCount * 4u) return false;
  *out = std::move(mesh);
  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(out->pointCount);
  runInputParityCheck(payload, cloud, rawPoints, *cache, *out, viewerParityChecksEnabled());
  return true;
#else
  const size_t pointCount = rawPoints.size() / 3u;
  if (pointCount == 0) return false;
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (canUseCudaInputPath(gpuCaps, sessionState, remap) && cudaCache) {
    const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
    auto ensureBuffer = [&](GLuint* id) {
      if (*id == 0) bufferApi.genBuffers(1, id);
      return *id != 0;
    };
    if (!bufferApi.available || !ensureBuffer(&cudaCache->verts) || !ensureBuffer(&cudaCache->colors)) {
      demoteCudaInputPath(remap, sessionState, "buffer-allocation");
    } else {
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, cudaCache->verts);
      bufferApi.bufferData(GL_ARRAY_BUFFER,
                           static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                           nullptr,
                           GL_DYNAMIC_DRAW);
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, cudaCache->colors);
      bufferApi.bufferData(GL_ARRAY_BUFFER,
                           static_cast<ViewerGLsizeiptr>(pointCount * 4u * sizeof(float)),
                           nullptr,
                           GL_DYNAMIC_DRAW);
      bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
      ChromaspaceCuda::InputRequest request{};
      request.pointCount = static_cast<int>(pointCount);
      request.pointAlphaScale = pointAlphaScaleForPlot(payload.pointSize, payload.pointDensity, payload.resolution);
      request.denseAlphaBias = denseAlphaBiasForPlot(payload.pointSize, payload.pointDensity, payload.resolution);
      request.remap = makeCudaRemapUniforms(uniforms);
      std::string error;
      if (ChromaspaceCuda::buildInputMesh(reinterpret_cast<ChromaspaceCuda::InputCache*>(cudaCache),
                                           request,
                                           rawPoints,
                                           nextMeshSerial(),
                                           &error)) {
        MeshData mesh{};
        mesh.resolution = cloud.resolution <= 25 ? 25 : (cloud.resolution <= 41 ? 41 : 57);
        mesh.quality = cloud.quality;
        mesh.paramHash = cloud.paramHash;
        mesh.serial = cudaCache->builtSerial;
        mesh.pointCount = pointCount;
        *out = std::move(mesh);
        const bool parityOk = runInputParityCheckWithBuffers(payload, cloud, rawPoints, cudaCache->verts, cudaCache->colors, *out, viewerParityChecksEnabled());
        if (!parityOk) {
          demoteCudaInputPath(remap, sessionState, "parity-mismatch");
          return false;
        }
        return true;
      }
      demoteCudaInputPath(remap, sessionState, error.empty() ? std::string("runtime-failure") : error);
    }
    return false;
  }
#endif
  if (sessionWantsCuda(gpuCaps)) return false;
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (!bufferApi.available || !computeApi.available || !ensureInputCloudComputeProgram(cache)) return false;

  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if (!ensureBuffer(&cache->input) || !ensureBuffer(&cache->verts) || !ensureBuffer(&cache->colors)) {
    logViewerEvent("Input-cloud compute buffer allocation failed; falling back to CPU.");
    return false;
  }

  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->input);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(rawPoints.size() * sizeof(float)),
                       rawPoints.data(),
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->verts);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->colors);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(pointCount * 4u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);

  computeApi.useProgram(cache->program);
  computeApi.uniform1i(cache->pointCountLoc, static_cast<GLint>(pointCount));
  computeApi.uniform1i(cache->showOverflowLoc, uniforms.showOverflow);
  computeApi.uniform1i(cache->highlightOverflowLoc, uniforms.highlightOverflow);
  computeApi.uniform1i(cache->plotModeLoc, uniforms.plotMode);
  computeApi.uniform1i(cache->circularHslLoc, uniforms.circularHsl);
  computeApi.uniform1i(cache->circularHsvLoc, uniforms.circularHsv);
  computeApi.uniform1i(cache->normConeNormalizedLoc, uniforms.normConeNormalized);
  computeApi.uniform1f(cache->pointAlphaScaleLoc,
                       pointAlphaScaleForPlot(payload.pointSize, payload.pointDensity, payload.resolution));
  computeApi.uniform1f(cache->denseAlphaBiasLoc,
                       denseAlphaBiasForPlot(payload.pointSize, payload.pointDensity, payload.resolution));
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, cache->input);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cache->verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cache->colors);
  const GLuint groups = static_cast<GLuint>((pointCount + 63u) / 64u);
  computeApi.dispatchCompute(groups, 1, 1);
  computeApi.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
  computeApi.useProgram(0);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  MeshData mesh{};
  mesh.resolution = cloud.resolution <= 25 ? 25 : (cloud.resolution <= 41 ? 41 : 57);
  mesh.quality = cloud.quality;
  mesh.paramHash = cloud.paramHash;
  mesh.serial = nextMeshSerial();
  mesh.pointCount = pointCount;
  *out = std::move(mesh);
  cache->builtSerial = out->serial;
  cache->pointCount = static_cast<GLsizei>(pointCount);
  runInputParityCheck(payload, cloud, rawPoints, *cache, *out, viewerParityChecksEnabled());
  return true;
#endif
}

bool senderMatchesCurrent(const std::string& currentSenderId, const std::string& senderId) {
  return currentSenderId.empty() || senderId.empty() || senderId == currentSenderId;
}

// The viewer only trusts clouds from the active sender and for the exact settings snapshot in force.
bool cloudMatchesResolved(const ResolvedPayload& resolved, const InputCloudPayload& cloud) {
  return senderMatchesCurrent(resolved.senderId, cloud.senderId) &&
         (!resolved.cloudSettingsKey.empty() && cloud.settingsKey == resolved.cloudSettingsKey);
}

std::mutex gMsgMutex;
PendingMessage gPendingParamsMsg;
PendingMessage gPendingCloudMsg;
bool gHasPendingParamsMsg = false;
bool gHasPendingCloudMsg = false;

std::string heartbeatAckJson() {
  std::ostringstream os;
  os << "{\"type\":\"heartbeat_ack\",\"visible\":" << gWindowVisible.load()
     << ",\"iconified\":" << gWindowIconified.load()
     << ",\"focused\":" << gWindowFocused.load() << "}";
  return os.str();
}

bool extractQuoted(const std::string& line, const std::string& key, std::string* out) {
  const std::string needle = "\"" + key + "\":\"";
  const size_t pos = line.find(needle);
  if (pos == std::string::npos) return false;
  const size_t start = pos + needle.size();
  const size_t end = line.find('"', start);
  if (end == std::string::npos) return false;
  *out = line.substr(start, end - start);
  return true;
}

bool extractUInt64(const std::string& line, const std::string& key, uint64_t* out) {
  const std::string needle = "\"" + key + "\":";
  const size_t pos = line.find(needle);
  if (pos == std::string::npos) return false;
  size_t start = pos + needle.size();
  size_t end = start;
  while (end < line.size() && std::isdigit(static_cast<unsigned char>(line[end]))) ++end;
  if (end == start) return false;
  *out = static_cast<uint64_t>(std::strtoull(line.substr(start, end - start).c_str(), nullptr, 10));
  return true;
}

bool extractInt(const std::string& line, const std::string& key, int* out) {
  uint64_t v = 0;
  if (!extractUInt64(line, key, &v)) return false;
  *out = static_cast<int>(v);
  return true;
}

bool extractFloat(const std::string& line, const std::string& key, float* out) {
  const std::string needle = "\"" + key + "\":";
  const size_t pos = line.find(needle);
  if (pos == std::string::npos) return false;
  size_t start = pos + needle.size();
  size_t end = start;
  while (end < line.size()) {
    const char c = line[end];
    if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+') {
      ++end;
      continue;
    }
    break;
  }
  if (end == start) return false;
  *out = static_cast<float>(std::strtod(line.substr(start, end - start).c_str(), nullptr));
  return true;
}

// Params are the authoritative description of how to interpret clouds, overlays, and draw state.
// Clamp incoming values here so stale or malformed messages do not destabilize the GL render path.
bool parseParamsMessage(const std::string& line, ResolvedPayload* out) {
  if (line.find("\"type\":\"params\"") == std::string::npos) return false;
  ResolvedPayload p{};
  extractUInt64(line, "seq", &p.seq);
  extractQuoted(line, "senderId", &p.senderId);
  extractQuoted(line, "sourceMode", &p.sourceMode);
  extractQuoted(line, "plotMode", &p.plotMode);
  extractQuoted(line, "cloudSettingsKey", &p.cloudSettingsKey);
  int volumeSlicingEnabled = 0;
  extractInt(line, "volumeSlicingEnabled", &volumeSlicingEnabled);
  p.volumeSlicingEnabled = (volumeSlicingEnabled != 0);
  extractQuoted(line, "volumeSlicingMode", &p.volumeSlicingMode);
  int lassoRegionEmpty = 0;
  extractInt(line, "lassoRegionEmpty", &lassoRegionEmpty);
  p.lassoRegionEmpty = (lassoRegionEmpty != 0);
  extractQuoted(line, "lassoData", &p.lassoData);
  extractQuoted(line, "quality", &p.quality);
  extractQuoted(line, "pointShape", &p.pointShape);
  extractQuoted(line, "version", &p.version);
  int resolution = 25;
  extractInt(line, "resolution", &resolution);
  p.resolution = resolution;
  int alwaysOnTop = 1;
  extractInt(line, "alwaysOnTop", &alwaysOnTop);
  p.alwaysOnTop = (alwaysOnTop != 0);
  int circularHsl = 0;
  extractInt(line, "circularHsl", &circularHsl);
  p.circularHsl = (circularHsl != 0);
  int circularHsv = 0;
  extractInt(line, "circularHsv", &circularHsv);
  p.circularHsv = (circularHsv != 0);
  int normConeNormalized = 1;
  extractInt(line, "normConeNormalized", &normConeNormalized);
  p.normConeNormalized = (normConeNormalized != 0);
  int showOverflow = 0;
  extractInt(line, "showOverflow", &showOverflow);
  p.showOverflow = (showOverflow != 0);
  int highlightOverflow = 1;
  extractInt(line, "highlightOverflow", &highlightOverflow);
  p.highlightOverflow = (highlightOverflow != 0);
  int cubeSlicingEnabled = 0;
  extractInt(line, "cubeSlicingEnabled", &cubeSlicingEnabled);
  p.cubeSlicingEnabled = (cubeSlicingEnabled != 0);
  int neutralRadiusEnabled = 0;
  extractInt(line, "neutralRadiusEnabled", &neutralRadiusEnabled);
  p.neutralRadiusEnabled = (neutralRadiusEnabled != 0);
  extractFloat(line, "neutralRadius", &p.neutralRadius);
  int cubeSliceRed = 1;
  extractInt(line, "cubeSliceRed", &cubeSliceRed);
  p.cubeSliceRed = (cubeSliceRed != 0);
  int cubeSliceGreen = 0;
  extractInt(line, "cubeSliceGreen", &cubeSliceGreen);
  p.cubeSliceGreen = (cubeSliceGreen != 0);
  int cubeSliceBlue = 0;
  extractInt(line, "cubeSliceBlue", &cubeSliceBlue);
  p.cubeSliceBlue = (cubeSliceBlue != 0);
  int cubeSliceCyan = 0;
  extractInt(line, "cubeSliceCyan", &cubeSliceCyan);
  p.cubeSliceCyan = (cubeSliceCyan != 0);
  int cubeSliceYellow = 0;
  extractInt(line, "cubeSliceYellow", &cubeSliceYellow);
  p.cubeSliceYellow = (cubeSliceYellow != 0);
  int cubeSliceMagenta = 0;
  extractInt(line, "cubeSliceMagenta", &cubeSliceMagenta);
  p.cubeSliceMagenta = (cubeSliceMagenta != 0);
  extractFloat(line, "overflowHighlightColorR", &p.overflowHighlightR);
  extractFloat(line, "overflowHighlightColorG", &p.overflowHighlightG);
  extractFloat(line, "overflowHighlightColorB", &p.overflowHighlightB);
  extractFloat(line, "viewerBackgroundColorR", &p.backgroundColorR);
  extractFloat(line, "viewerBackgroundColorG", &p.backgroundColorG);
  extractFloat(line, "viewerBackgroundColorB", &p.backgroundColorB);
  extractFloat(line, "pointSize", &p.pointSize);
  extractFloat(line, "pointDensity", &p.pointDensity);
  int identityOverlayEnabled = 0;
  extractInt(line, "identityOverlayEnabled", &identityOverlayEnabled);
  p.identityOverlayEnabled = (identityOverlayEnabled != 0);
  int identityOverlayRamp = 0;
  extractInt(line, "identityOverlayRamp", &identityOverlayRamp);
  p.identityOverlayRamp = (identityOverlayRamp != 0);
  int identityOverlayAuto = 1;
  extractInt(line, "identityOverlayAuto", &identityOverlayAuto);
  p.identityOverlayAuto = (identityOverlayAuto != 0);
  extractInt(line, "identityOverlayRequestedSize", &p.identityOverlayRequestedSize);
  extractInt(line, "identityOverlaySize", &p.identityOverlaySize);
  extractInt(line, "chromaticityInputPrimaries", &p.chromaticityInputPrimaries);
  extractInt(line, "chromaticityInputTransfer", &p.chromaticityInputTransfer);
  extractInt(line, "chromaticityReferenceBasis", &p.chromaticityReferenceBasis);
  extractInt(line, "chromaticityOverlayPrimaries", &p.chromaticityOverlayPrimaries);
  int chromaticityPlanckianLocus = 1;
  extractInt(line, "chromaticityPlanckianLocus", &chromaticityPlanckianLocus);
  p.chromaticityPlanckianLocus = (chromaticityPlanckianLocus != 0);
  p.pointSize = clampf(p.pointSize, 0.35f, 3.0f);
  p.pointDensity = clampf(p.pointDensity, 0.1f, 4.0f);
  p.neutralRadius = clampf(p.neutralRadius, 0.0f, 1.0f);
  p.overflowHighlightR = clamp01(p.overflowHighlightR);
  p.overflowHighlightG = clamp01(p.overflowHighlightG);
  p.overflowHighlightB = clamp01(p.overflowHighlightB);
  p.backgroundColorR = clamp01(p.backgroundColorR);
  p.backgroundColorG = clamp01(p.backgroundColorG);
  p.backgroundColorB = clamp01(p.backgroundColorB);
  p.identityOverlayRequestedSize = std::clamp(p.identityOverlayRequestedSize, 4, 65);
  p.identityOverlaySize = std::clamp(p.identityOverlaySize, 4, 65);
  p.chromaticityInputPrimaries =
      std::clamp(p.chromaticityInputPrimaries, 0, static_cast<int>(WorkshopColor::primariesCount()) - 1);
  p.chromaticityInputTransfer =
      std::clamp(p.chromaticityInputTransfer, 0, static_cast<int>(WorkshopColor::transferFunctionCount()) - 1);
  p.chromaticityReferenceBasis = std::clamp(p.chromaticityReferenceBasis, 0, 1);
  p.chromaticityOverlayPrimaries =
      std::clamp(p.chromaticityOverlayPrimaries, 0, static_cast<int>(WorkshopColor::primariesCount()));
  *out = std::move(p);
  return true;
}

// Stage: parse the plot cloud payload exactly as shipped by the OFX instance.
bool parseInputCloudMessage(const std::string& line, InputCloudPayload* out) {
  if (line.find("\"type\":\"input_cloud\"") == std::string::npos) return false;
  InputCloudPayload p{};
  extractUInt64(line, "seq", &p.seq);
  extractQuoted(line, "senderId", &p.senderId);
  extractQuoted(line, "quality", &p.quality);
  extractQuoted(line, "paramHash", &p.paramHash);
  extractQuoted(line, "settingsKey", &p.settingsKey);
  extractQuoted(line, "transport", &p.transport);
  extractQuoted(line, "shmName", &p.shmName);
  extractQuoted(line, "points", &p.points);
  int resolution = 25;
  extractInt(line, "resolution", &resolution);
  p.resolution = resolution;
  uint64_t pointCount = 0;
  extractUInt64(line, "pointCount", &pointCount);
  p.pointCount = pointCount;
  int pointStride = 24;
  extractInt(line, "pointStride", &pointStride);
  p.pointStride = pointStride;
  uint64_t shmSize = 0;
  extractUInt64(line, "shmSize", &shmSize);
  p.shmSize = shmSize;
  if (p.transport == "shm" && !loadInputCloudSharedMemory(p, &p.packedPoints)) {
    logViewerEvent(std::string("Failed to load shared-memory cloud: ") + p.shmName);
    return false;
  }
  *out = std::move(p);
  return true;
}

void mapDisplayColor(float r, float g, float b, float* outR, float* outG, float* outB) {
  auto gamma = [](float v) { return std::pow(clamp01(v), 1.0f / 2.2f); };
  *outR = gamma(r);
  *outG = gamma(g);
  *outB = gamma(b);
}

bool pointOverflowsCube(float r, float g, float b) {
  return r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f;
}

float normalizedNeutralRadiusForPoint(const PlotRemapSpec& spec, float r, float g, float b) {
  constexpr float kRgbAxisMaxRadius = 0.8164965809277260f;
  constexpr float kPolarMax = 0.9553166181245093f;
  constexpr float kChenPolarScale = 1.0467733744265997f;

  switch (spec.plotMode) {
    case PlotModeKind::Rgb: {
      const float rr = clamp01(r);
      const float gg = clamp01(g);
      const float bb = clamp01(b);
      const float rotX = 0.81649658093f * rr - 0.40824829046f * gg - 0.40824829046f * bb;
      const float rotY = 0.70710678118f * gg - 0.70710678118f * bb;
      return clampf(std::sqrt(rotX * rotX + rotY * rotY) / kRgbAxisMaxRadius, 0.0f, 1.0f);
    }
    case PlotModeKind::Hsl: {
      if (spec.circularHsl) {
        float h = 0.0f;
        float radius = 0.0f;
        float l = 0.0f;
        rgbToPlotCircularHsl(r, g, b, &h, &radius, &l);
        return clampf(radius, 0.0f, 1.0f);
      }
      const float cMax = std::max(r, std::max(g, b));
      const float cMin = std::min(r, std::min(g, b));
      return clampf(cMax - cMin, 0.0f, 1.0f);
    }
    case PlotModeKind::Hsv: {
      if (spec.circularHsv) {
        float h = 0.0f;
        float radius = 0.0f;
        float v = 0.0f;
        rgbToPlotCircularHsv(r, g, b, &h, &radius, &v);
        return clampf(radius, 0.0f, 1.0f);
      }
      float x = 0.0f;
      float z = 0.0f;
      rgbToHsvHexconePlane(r, g, b, &x, &z);
      return clampf(std::hypot(x, z), 0.0f, 1.0f);
    }
    case PlotModeKind::Chen: {
      float h = 0.0f;
      float chroma = 0.0f;
      float light = 0.0f;
      rgbToChen(r, g, b, spec.showOverflow, &h, &chroma, &light);
      const float polar = chroma / kChenPolarScale;
      const float radius = light * std::sin(polar) / kRgbAxisMaxRadius;
      return clampf(radius, 0.0f, 1.0f);
    }
    case PlotModeKind::RgbToCone: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToRgbCone(r, g, b, &magnitude, &hue, &polar);
      const float radial = magnitude * std::sin(polar * kPolarMax);
      return clampf(radial / std::sin(kPolarMax), 0.0f, 1.0f);
    }
    case PlotModeKind::JpConical: {
      float magnitude = 0.0f;
      float hue = 0.0f;
      float polar = 0.0f;
      rgbToJpConical(r, g, b, spec.showOverflow, &magnitude, &hue, &polar);
      const float radial = magnitude * std::sin(polar * kPolarMax);
      return clampf(radial / std::sin(kPolarMax), 0.0f, 1.0f);
    }
    case PlotModeKind::NormCone: {
      float hue = 0.0f;
      float chroma = 0.0f;
      float value = 0.0f;
      rgbToNormConeCoords(r, g, b, spec.normConeNormalized, spec.showOverflow, &hue, &chroma, &value);
      return clampf(chroma, 0.0f, 1.0f);
    }
    case PlotModeKind::Reuleaux: {
      float hue = 0.0f;
      float sat = 0.0f;
      float value = 0.0f;
      rgbToReuleaux(r, g, b, spec.showOverflow, &hue, &sat, &value);
      return clampf(sat, 0.0f, 1.0f);
    }
    case PlotModeKind::Chromaticity:
    default:
      return 0.0f;
  }
}

float effectiveNeutralRadiusThreshold(float sliderValue) {
  constexpr float kNeutralRadiusResponsePower = 2.0f;
  return clampf(std::pow(clampf(sliderValue, 0.0f, 1.0f), kNeutralRadiusResponsePower), 0.0f, 1.0f);
}

bool neutralRadiusContainsPoint(const PlotRemapSpec& spec, float r, float g, float b) {
  if (!spec.neutralRadiusEnabled || spec.plotMode == PlotModeKind::Chromaticity || spec.showOverflow) return true;
  return normalizedNeutralRadiusForPoint(spec, r, g, b) <= effectiveNeutralRadiusThreshold(spec.neutralRadius) + 1e-6f;
}

bool cubeSliceContainsPoint(const PlotRemapSpec& spec, float r, float g, float b) {
  if (!neutralRadiusContainsPoint(spec, r, g, b)) return false;
  const bool anyRegionSelected = spec.cubeSliceRed || spec.cubeSliceGreen || spec.cubeSliceBlue ||
                                 spec.cubeSliceCyan || spec.cubeSliceYellow || spec.cubeSliceMagenta;
  if (!spec.cubeSlicingEnabled) return true;
  if (!anyRegionSelected) return false;
  if (spec.plotMode == PlotModeKind::Rgb) {
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
    case PlotModeKind::Hsl:
    case PlotModeKind::Hsv: {
      const float cMax = std::max(r, std::max(g, b));
      const float cMin = std::min(r, std::min(g, b));
      const float delta = cMax - cMin;
      if (delta > 1e-6f) {
        hue = rawRgbHue01(r, g, b, cMax, delta);
        hueDefined = true;
      }
      break;
    }
    case PlotModeKind::Chen: {
      float chroma = 0.0f;
      float light = 0.0f;
      rgbToChen(r, g, b, spec.showOverflow, &hue, &chroma, &light);
      hueDefined = chroma > 1e-6f;
      break;
    }
    case PlotModeKind::RgbToCone: {
      float magnitude = 0.0f;
      float polar = 0.0f;
      rgbToRgbCone(r, g, b, &magnitude, &hue, &polar);
      hueDefined = magnitude > 1e-6f && polar > 1e-6f;
      break;
    }
    case PlotModeKind::JpConical: {
      float magnitude = 0.0f;
      float polar = 0.0f;
      rgbToJpConical(r, g, b, spec.showOverflow, &magnitude, &hue, &polar);
      hueDefined = magnitude > 1e-6f && polar > 1e-6f;
      break;
    }
    case PlotModeKind::NormCone: {
      float chroma = 0.0f;
      float value = 0.0f;
      rgbToNormConeCoords(r, g, b, spec.normConeNormalized, spec.showOverflow, &hue, &chroma, &value);
      hueDefined = chroma > 1e-6f;
      break;
    }
    case PlotModeKind::Reuleaux: {
      float sat = 0.0f;
      float value = 0.0f;
      rgbToReuleaux(r, g, b, spec.showOverflow, &hue, &sat, &value);
      hueDefined = sat > 1e-6f;
      break;
    }
    case PlotModeKind::Rgb:
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

bool cubeSliceContainsPoint(const ResolvedPayload& payload, float r, float g, float b) {
  return cubeSliceContainsPoint(makePlotRemapSpec(payload), r, g, b);
}

float wrapHue01(float h) {
  h = std::fmod(h, 1.0f);
  if (h < 0.0f) h += 1.0f;
  return h;
}

void rgbToHsvHexconePlane(float r, float g, float b, float* outX, float* outZ) {
  // Smith's HSV hexcone can be viewed as the RGB cube projected onto the chroma plane.
  // The equal-channel gray component cancels out, leaving a regular hexagon for the pure hues.
  *outX = r - 0.5f * g - 0.5f * b;
  *outZ = 0.8660254037844386f * (g - b);
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

void rgbToHsl(float r, float g, float b, float* outH, float* outS, float* outL) {
  r = clamp01(r);
  g = clamp01(g);
  b = clamp01(b);
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  float h = 0.0f;
  const float l = 0.5f * (cMax + cMin);
  float s = 0.0f;
  if (delta > 1e-6f) {
    s = delta / (1.0f - std::fabs(2.0f * l - 1.0f));
    if (cMax == r) {
      h = std::fmod((g - b) / delta, 6.0f);
    } else if (cMax == g) {
      h = ((b - r) / delta) + 2.0f;
    } else {
      h = ((r - g) / delta) + 4.0f;
    }
    h /= 6.0f;
    if (h < 0.0f) h += 1.0f;
  }
  *outH = h;
  *outS = clamp01(s);
  *outL = clamp01(l);
}

void rgbToPlotHsl(float r, float g, float b, float* outH, float* outRadius, float* outL) {
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  const float l = 0.5f * (cMax + cMin);
  float h = rawRgbHue01(r, g, b, cMax, delta);
  const float satDenom = 1.0f - std::fabs(2.0f * l - 1.0f);
  if (delta > 1e-6f && satDenom < 0.0f) {
    // CSS Color 4 treats negative HSL saturation as a hue rotation by 180 degrees.
    h = wrapHue01(h + 0.5f);
  }
  *outH = h;
  *outRadius = delta;
  *outL = l;
}

void rgbToPlotCircularHsl(float r, float g, float b, float* outH, float* outRadius, float* outL) {
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  const float l = 0.5f * (cMax + cMin);
  float h = rawRgbHue01(r, g, b, cMax, delta);
  float satDenom = 1.0f - std::fabs(2.0f * l - 1.0f);
  if (delta > 1e-6f && satDenom < 0.0f) {
    // CSS Color 4 resolves negative HSL saturation by rotating hue 180 degrees and using |s|.
    h = wrapHue01(h + 0.5f);
  }
  if (std::fabs(satDenom) <= 1e-6f) {
    satDenom = satDenom < 0.0f ? -1e-6f : 1e-6f;
  }
  *outH = h;
  *outRadius = std::fabs(delta / satDenom);
  *outL = l;
}

void rgbToPlotHsvHexcone(float r, float g, float b, float* outX, float* outZ, float* outV) {
  const float cMax = std::max(r, std::max(g, b));
  rgbToHsvHexconePlane(r, g, b, outX, outZ);
  *outV = cMax;
}

void rgbToPlotCircularHsv(float r, float g, float b, float* outH, float* outRadius, float* outV) {
  const float cMax = std::max(r, std::max(g, b));
  const float cMin = std::min(r, std::min(g, b));
  const float delta = cMax - cMin;
  *outH = rawRgbHue01(r, g, b, cMax, delta);
  *outRadius = (delta > 1e-6f && cMax > 1e-6f) ? (delta / cMax) : 0.0f;
  *outV = cMax;
}

bool overflowHighlightApplies(const PlotRemapSpec& spec, float r, float g, float b) {
  if (!spec.showOverflow || !spec.highlightOverflow) return false;
  if (spec.plotMode != PlotModeKind::Rgb &&
      spec.plotMode != PlotModeKind::Hsl &&
      spec.plotMode != PlotModeKind::Hsv &&
      spec.plotMode != PlotModeKind::Chen &&
      spec.plotMode != PlotModeKind::JpConical &&
      spec.plotMode != PlotModeKind::Reuleaux &&
      spec.plotMode != PlotModeKind::Chromaticity) return false;
  return pointOverflowsCube(r, g, b);
}

bool overflowHighlightApplies(const ResolvedPayload& payload, float r, float g, float b) {
  return overflowHighlightApplies(makePlotRemapSpec(payload), r, g, b);
}

void rgbToNormConeCoords(float r, float g, float b, bool normalized, bool allowOverflow, float* outHue, float* outChroma, float* outValue) {
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

void rgbToChen(float r, float g, float b, bool allowOverflow, float* outHue, float* outChroma, float* outLight) {
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

Vec3 mapChenPlot(float r, float g, float b, bool allowOverflow) {
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kChenPolarMax = 0.9553166181245093f;
  constexpr float kChenMaxRadius = 0.81649658092772603273f;  // sin(kChenPolarMax)
  constexpr float kChenVisualScale = 1.21f;
  float h = 0.0f;
  float c = 0.0f;
  float l = 0.0f;
  rgbToChen(r, g, b, allowOverflow, &h, &c, &l);
  const float angle = h * kTau;
  const float polar = c * kChenPolarMax;
  const float radius = l * std::sin(polar) / kChenMaxRadius;
  const Vec3 out{std::cos(angle) * radius * kChenVisualScale,
                 (l * 2.0f - 1.0f) * kChenVisualScale,
                 std::sin(angle) * radius * kChenVisualScale};
  return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
}

Vec3 mapChenPlot(float r, float g, float b) {
  return mapChenPlot(r, g, b, false);
}

Vec3 mapConePolarPlot(float magnitude, float hue, float polarNorm, bool allowOverflow = false) {
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kPolarMax = 0.9553166181245093f;
  const float rho = allowOverflow ? std::max(magnitude, 0.0f) : clamp01(magnitude);
  const float phi = (allowOverflow ? std::max(polarNorm, 0.0f) : clamp01(polarNorm)) * kPolarMax;
  const float angle = hue * kTau;
  const float radial = rho * std::sin(phi);
  const Vec3 out{std::cos(angle) * radial, rho * std::cos(phi) * 2.0f - 1.0f, std::sin(angle) * radial};
  return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
}

float normalizeGainNormCone(float hueNorm, float chromaNorm) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kSphericalMax = 0.9553166181245093f;
  float angle = hueNorm * 6.28318530717958647692f - kPi / 6.0f;
  float chroma = chromaNorm * kSphericalMax;
  chroma = std::tan(chroma) / std::sqrt(2.0f);
  const float sinTerm = clampf(std::sin(3.0f * angle), -1.0f, 1.0f);
  return 1.0f / (2.0f * std::cos(std::acos(sinTerm) / 3.0f));
}

void rgbToRgbCone(float r, float g, float b, float* outMagnitude, float* outHue, float* outPolar) {
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

void rgbToJpConical(float r, float g, float b, bool allowOverflow, float* outMagnitude, float* outHue, float* outPolar) {
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

void rgbToReuleaux(float r, float g, float b, bool allowOverflow, float* outHue, float* outSat, float* outValue) {
  constexpr float kPi = 3.14159265358979323846f;
  constexpr float kTau = 6.28318530717958647692f;
  constexpr float kMaxSat = 1.41421356237f;  // sqrt(2)
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

WorkshopColor::Vec2f mapChromaticityBasis(const PlotRemapSpec& spec, WorkshopColor::Vec2f xy) {
  if (spec.chromaticity.referenceBasis == WorkshopColor::ChromaticityReferenceBasis::InputObserver) {
    xy = WorkshopColor::standardObserverToInputObserver(xy, spec.chromaticity.inputPrimaries);
  }
  return WorkshopColor::isFinite(xy) ? xy : spec.chromaticityWhite;
}

Vec3 mapChromaticityCoordsToViewer(const PlotRemapSpec& spec, WorkshopColor::Vec2f xy, float Y) {
  xy = mapChromaticityBasis(spec, xy);
  const float viewerHeight = (spec.showOverflow ? Y : clampf(Y, 0.0f, 1.0f)) * 2.0f - 1.0f;
  // Keep xy on the viewer's front-facing plane so switching from the cube view lands on a familiar
  // chromaticity-chart orientation without needing a camera reset. Luminance extrudes out of that
  // plane toward the viewer for free 3D inspection.
  const Vec3 out{(xy.x - (1.0f / 3.0f)) * 3.0f,
                 (xy.y - (1.0f / 3.0f)) * 3.0f,
                 viewerHeight};
  return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
}

Vec3 mapChromaticityPlot(const PlotRemapSpec& spec, float r, float g, float b) {
  WorkshopColor::Vec3f linear =
      WorkshopColor::decodeToLinear({r, g, b}, spec.chromaticity.inputTransfer);
  if (!spec.showOverflow) {
    linear = WorkshopColor::clamp(linear, 0.0f, 1.0f);
  }
  const WorkshopColor::Vec3f xyz = WorkshopColor::mul(spec.chromaticityRgbToXyz, linear);
  if (!WorkshopColor::isFinite(xyz)) {
    return mapChromaticityCoordsToViewer(spec, spec.chromaticityWhite, 0.0f);
  }
  const float xyzSum = xyz.x + xyz.y + xyz.z;
  if (std::fabs(xyzSum) <= 1e-8f) {
    return mapChromaticityCoordsToViewer(spec, spec.chromaticityWhite, 0.0f);
  }
  const WorkshopColor::XyY xyY = WorkshopColor::xyzToXyY(xyz, spec.chromaticityWhite);
  return mapChromaticityCoordsToViewer(spec, {xyY.x, xyY.y}, xyY.Y);
}

// Every cloud/overlay sample is remapped through this single switch so each plot model shares
// the same downstream mesh/render code once RGB has been converted into plot-space coordinates.
Vec3 mapPointToPlotMode(const PlotRemapSpec& spec, float r, float g, float b) {
  constexpr float kTau = 6.28318530717958647692f;
  if (spec.plotMode == PlotModeKind::Chromaticity) {
    return mapChromaticityPlot(spec, r, g, b);
  }
  if (spec.plotMode == PlotModeKind::Hsl) {
    float h = 0.0f, radius = 0.0f, l = 0.0f;
    if (spec.circularHsl) {
      rgbToPlotCircularHsl(r, g, b, &h, &radius, &l);
    } else {
      rgbToPlotHsl(r, g, b, &h, &radius, &l);
    }
    const float angle = h * kTau;
    const Vec3 out{std::cos(angle) * radius, l * 2.0f - 1.0f, std::sin(angle) * radius};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
  }
  if (spec.plotMode == PlotModeKind::Hsv) {
    float x = 0.0f;
    float z = 0.0f;
    float v = 0.0f;
    if (spec.circularHsv) {
      float h = 0.0f;
      float radius = 0.0f;
      rgbToPlotCircularHsv(r, g, b, &h, &radius, &v);
      const float angle = h * kTau;
      x = std::cos(angle) * radius;
      z = std::sin(angle) * radius;
    } else {
      rgbToPlotHsvHexcone(r, g, b, &x, &z, &v);
    }
    const Vec3 out{x, v * 2.0f - 1.0f, z};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
  }
  if (spec.plotMode == PlotModeKind::Chen) {
    return mapChenPlot(r, g, b, spec.showOverflow);
  }
  if (spec.plotMode == PlotModeKind::RgbToCone) {
    float magnitude = 0.0f;
    float hue = 0.0f;
    float polar = 0.0f;
    rgbToRgbCone(r, g, b, &magnitude, &hue, &polar);
    return mapConePolarPlot(magnitude, hue, polar);
  }
  if (spec.plotMode == PlotModeKind::JpConical) {
    constexpr float kJpConicalVisualScale = 1.17f;
    float magnitude = 0.0f;
    float hue = 0.0f;
    float polar = 0.0f;
    rgbToJpConical(r, g, b, spec.showOverflow, &magnitude, &hue, &polar);
    const Vec3 base = mapConePolarPlot(magnitude, hue, polar, spec.showOverflow);
    const Vec3 out{base.x * kJpConicalVisualScale,
                   base.y * kJpConicalVisualScale,
                   base.z * kJpConicalVisualScale};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
  }
  if (spec.plotMode == PlotModeKind::NormCone) {
    float hue = 0.0f;
    float chroma = 0.0f;
    float value = 0.0f;
    rgbToNormConeCoords(r, g, b, spec.normConeNormalized, spec.showOverflow, &hue, &chroma, &value);
    const float angle = hue * kTau;
    const Vec3 out{std::cos(angle) * chroma, value * 2.0f - 1.0f, std::sin(angle) * chroma};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
  }
  if (spec.plotMode == PlotModeKind::Reuleaux) {
    float hue = 0.0f;
    float sat = 0.0f;
    float value = 0.0f;
    rgbToReuleaux(r, g, b, spec.showOverflow, &hue, &sat, &value);
    const float angle = hue * kTau;
    const Vec3 out{std::cos(angle) * sat, value * 2.0f - 1.0f, std::sin(angle) * sat};
    return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z) ? out : Vec3{};
  }
  return Vec3{r * 2.0f - 1.0f, g * 2.0f - 1.0f, b * 2.0f - 1.0f};
}

Vec3 mapPointToPlotMode(const ResolvedPayload& payload, float r, float g, float b) {
  return mapPointToPlotMode(makePlotRemapSpec(payload), r, g, b);
}

float pointAlphaScaleForPlot(float pointSize, float pointDensity, int resolution) {
  const float size = clampf(pointSize, 0.35f, 3.0f);
  const float density = clampf(pointDensity, 0.35f, 4.0f);
  const float resolutionScale = clampf(static_cast<float>(resolution) / 41.0f, 0.60f, 1.45f);
  const float sizeNorm = std::max(1.0f, size / 1.15f);
  const float densityNorm = std::max(1.0f, density * resolutionScale);
  const float sizeBlend = clampf((size - 1.10f) / 1.40f, 0.0f, 1.0f);
  const float densityBlend = clampf((densityNorm - 1.0f) / 0.55f, 0.0f, 1.0f);
  const float blend = clampf(std::max(sizeBlend, densityBlend), 0.0f, 1.0f);
  if (blend <= 0.0f) return 1.0f;

  const float eased = blend * blend * (3.0f - 2.0f * blend);
  const float areaComp = 1.0f / std::pow(sizeNorm, 1.55f);
  const float densityComp = 1.0f / std::pow(densityNorm, 0.35f);
  const float targetScale = clampf(areaComp * densityComp, 0.34f, 1.0f);
  return clampf((1.0f - eased) + eased * targetScale, 0.34f, 1.0f);
}

float denseAlphaBiasForPlot(float pointSize, float pointDensity, int resolution) {
  const float size = clampf(pointSize, 0.35f, 3.0f);
  const float density = clampf(pointDensity, 0.35f, 4.0f);
  const float resolutionScale = clampf(static_cast<float>(resolution) / 41.0f, 0.60f, 1.45f);
  const float sizeBlend = clampf((size - 1.10f) / 1.40f, 0.0f, 1.0f);
  const float densityBlend = clampf((density * resolutionScale - 1.0f) / 0.55f, 0.0f, 1.0f);
  const float blend = clampf(std::max(sizeBlend, densityBlend), 0.0f, 1.0f);
  return blend * blend * (3.0f - 2.0f * blend);
}

float denseLumaProtectedAlpha(float baseAlpha,
                              float pointSize,
                              float pointDensity,
                              int resolution,
                              float cr,
                              float cg,
                              float cb,
                              bool overflowPoint) {
  const float scaled = scaledPointAlpha(baseAlpha, pointSize, pointDensity, resolution);
  if (overflowPoint) return scaled;
  const float denseBias = denseAlphaBiasForPlot(pointSize, pointDensity, resolution);
  if (denseBias <= 0.0f) return scaled;
  const float luma = clampf(cr * 0.2126f + cg * 0.7152f + cb * 0.0722f, 0.0f, 1.0f);
  const float highlightKnee = clampf((luma - 0.70f) / 0.24f, 0.0f, 1.0f);
  const float shadowMidProtect = 1.0f - clampf((luma - 0.58f) / 0.30f, 0.0f, 1.0f);
  const float multiplier = clampf(1.0f + 0.16f * denseBias * shadowMidProtect - 0.30f * denseBias * highlightKnee,
                                  0.76f, 1.12f);
  return clampf(scaled * multiplier, 0.0f, 1.0f);
}

float scaledPointAlpha(float baseAlpha, float pointSize, float pointDensity, int resolution) {
  return clampf(baseAlpha * pointAlphaScaleForPlot(pointSize, pointDensity, resolution), 0.0f, 1.0f);
}

float densePlotGlowSuppression(float pointSize, float pointDensity, int resolution) {
  const float size = clampf(pointSize, 0.35f, 3.0f);
  const float density = clampf(pointDensity, 0.35f, 4.0f);
  const float resolutionScale = clampf(static_cast<float>(resolution) / 41.0f, 0.65f, 1.50f);
  const float sizeBlend = clampf((size - 1.20f) / 1.30f, 0.0f, 1.0f);
  const float densityBlend = clampf((density * resolutionScale - 1.0f) / 0.50f, 0.0f, 1.0f);
  const float blend = clampf(sizeBlend * densityBlend, 0.0f, 1.0f);
  return blend * blend * (3.0f - 2.0f * blend);
}

size_t overlayIdentityPointCap(const ResolvedPayload& payload, int cubeSize) {
  const size_t total = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  (void)payload;
  // The overlay size is already clamped to 65^3, so drawing the full lattice stays within the
  // user-facing safety limit and avoids the noticeable density drop that appeared at 64 and 65.
  return total;
}

void appendOverlayPoint(const ResolvedPayload& payload,
                        float r, float g, float b, float alpha,
                        std::vector<float>* verts,
                        std::vector<float>* colors) {
  if (!verts || !colors) return;
  const PlotRemapSpec remap = makeOverlayPlotRemapSpec(payload);
  if (!cubeSliceContainsPoint(remap, r, g, b)) return;
  const Vec3 pos = mapPointToPlotMode(remap, r, g, b);
  verts->push_back(pos.x);
  verts->push_back(pos.y);
  verts->push_back(pos.z);
  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  mapDisplayColor(r, g, b, &cr, &cg, &cb);
  colors->push_back(cr);
  colors->push_back(cg);
  colors->push_back(cb);
  colors->push_back(alpha);
}

void appendOverlayDirectGrid(const ResolvedPayload& payload, int cubeSize, float alpha, MeshData* mesh) {
  const int denom = std::max(1, cubeSize - 1);
  for (int bz = 0; bz < cubeSize; ++bz) {
    for (int gy = 0; gy < cubeSize; ++gy) {
      for (int rx = 0; rx < cubeSize; ++rx) {
        const float r = static_cast<float>(rx) / static_cast<float>(denom);
        const float g = static_cast<float>(gy) / static_cast<float>(denom);
        const float b = static_cast<float>(bz) / static_cast<float>(denom);
        appendOverlayPoint(payload, r, g, b, alpha, &mesh->pointVerts, &mesh->pointColors);
      }
    }
  }
}

void appendOverlayStripSamples(const ResolvedPayload& payload, int cubeSize, size_t sampleCount, float alpha, MeshData* mesh) {
  const int denom = std::max(1, cubeSize - 1);
  const size_t width = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  const size_t height = static_cast<size_t>(cubeSize);
  const int grid = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(sampleCount)))));
  for (size_t i = 0; i < sampleCount; ++i) {
    const int gx = static_cast<int>(i % static_cast<size_t>(grid));
    const int gy = static_cast<int>(i / static_cast<size_t>(grid));
    const double u = (static_cast<double>(gx) + 0.5) / static_cast<double>(grid);
    const double v = (static_cast<double>(gy) + 0.5) / static_cast<double>(grid);
    const size_t x = std::min(width - 1, static_cast<size_t>(u * static_cast<double>(width)));
    const size_t y = std::min(height - 1, static_cast<size_t>(v * static_cast<double>(height)));
    const int rx = static_cast<int>(x % static_cast<size_t>(cubeSize));
    const int bz = static_cast<int>(x / static_cast<size_t>(cubeSize));
    const int gyInv = denom - static_cast<int>(y);
    const float r = static_cast<float>(rx) / static_cast<float>(denom);
    const float g = static_cast<float>(std::clamp(gyInv, 0, denom)) / static_cast<float>(denom);
    const float b = static_cast<float>(std::clamp(bz, 0, denom)) / static_cast<float>(denom);
    appendOverlayPoint(payload, r, g, b, alpha, &mesh->pointVerts, &mesh->pointColors);
  }
}

void appendOverlayRamp(const ResolvedPayload& payload, int cubeSize, MeshData* mesh) {
  const size_t rampSamples = std::max<size_t>(static_cast<size_t>(cubeSize), static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize));
  const size_t cappedRampSamples = std::min<size_t>(rampSamples, 65536u);
  const int denom = std::max(1, static_cast<int>(cappedRampSamples) - 1);
  for (size_t i = 0; i < cappedRampSamples; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(std::max(1, denom));
    appendOverlayPoint(payload, t, t, t, 0.92f, &mesh->pointVerts, &mesh->pointColors);
  }
}

// Identity overlay mesh generation mirrors the plugin-side strip logic, but stays purely viewer-side.
// Auto mode may draw the full lattice, while explicit large sizes fall back to strip-style sampling caps.
bool buildIdentityOverlayMesh(const ResolvedPayload& payload, MeshData* out) {
  if (!out || !payload.identityOverlayEnabled) return false;
  MeshData mesh{};
  const int cubeSize = std::clamp(payload.identityOverlaySize, 4, 65);
  mesh.resolution = cubeSize;
  mesh.quality = payload.identityOverlayAuto ? "Overlay Auto" : "Overlay";
  mesh.paramHash = payload.identityOverlayRamp ? "identity_overlay+ramp" : "identity_overlay";
  mesh.serial = nextMeshSerial();
  const size_t totalPoints = static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize);
  const size_t pointCap = overlayIdentityPointCap(payload, cubeSize);
  const size_t reservePoints = std::min<size_t>(totalPoints + (payload.identityOverlayRamp ? static_cast<size_t>(cubeSize) * static_cast<size_t>(cubeSize) : 0u),
                                                 pointCap + (payload.identityOverlayRamp ? 65536u : 0u));
  mesh.pointVerts.reserve(reservePoints * 3u);
  mesh.pointColors.reserve(reservePoints * 4u);
  if (totalPoints <= pointCap) {
    appendOverlayDirectGrid(payload, cubeSize, 0.24f, &mesh);
  } else {
    appendOverlayStripSamples(payload, cubeSize, pointCap, 0.24f, &mesh);
  }
  if (payload.identityOverlayRamp) {
    appendOverlayRamp(payload, cubeSize, &mesh);
  }
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  if (mesh.pointCount == 0) return false;
  *out = std::move(mesh);
  return true;
}

bool buildIdentityMesh(const ResolvedPayload& payload, MeshData* out) {
  if (!out) return false;
  MeshData mesh{};
  mesh.resolution = payload.resolution <= 25 ? 25 : (payload.resolution <= 41 ? 41 : 57);
  mesh.quality = payload.quality;
  mesh.paramHash = "identity";
  mesh.serial = nextMeshSerial();
  mesh.pointVerts.reserve(static_cast<size_t>(mesh.resolution) * mesh.resolution * mesh.resolution * 3u);
  mesh.pointColors.reserve(static_cast<size_t>(mesh.resolution) * mesh.resolution * mesh.resolution * 4u);
  const int denom = std::max(1, mesh.resolution - 1);
  for (int bz = 0; bz < mesh.resolution; ++bz) {
    for (int gy = 0; gy < mesh.resolution; ++gy) {
      for (int rx = 0; rx < mesh.resolution; ++rx) {
        const float r = static_cast<float>(rx) / static_cast<float>(denom);
        const float g = static_cast<float>(gy) / static_cast<float>(denom);
        const float b = static_cast<float>(bz) / static_cast<float>(denom);
        if (!cubeSliceContainsPoint(payload, r, g, b)) continue;
        const Vec3 pos = mapPointToPlotMode(payload, r, g, b);
        mesh.pointVerts.push_back(pos.x);
        mesh.pointVerts.push_back(pos.y);
        mesh.pointVerts.push_back(pos.z);
        float cr = 0.0f, cg = 0.0f, cb = 0.0f;
        mapDisplayColor(r, g, b, &cr, &cg, &cb);
        mesh.pointColors.push_back(cr);
        mesh.pointColors.push_back(cg);
        mesh.pointColors.push_back(cb);
        mesh.pointColors.push_back(
            denseLumaProtectedAlpha(0.72f, payload.pointSize, payload.pointDensity, payload.resolution, cr, cg, cb, false));
      }
    }
  }
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  *out = std::move(mesh);
  return true;
}

// Serialized clouds carry RGB values as both position and color payloads; the viewer ignores the transmitted
// xyz for now and remaps from RGB locally so plot-model changes can be applied without rebuilding the source cloud.
bool buildInputCloudMeshCpu(const ResolvedPayload& payload,
                            const InputCloudPayload& cloud,
                            const std::vector<float>& rawPoints,
                            MeshData* out) {
  if (!out) return false;
  const PlotRemapSpec remap = makePlotRemapSpec(payload);
  MeshData mesh{};
  mesh.resolution = cloud.resolution <= 25 ? 25 : (cloud.resolution <= 41 ? 41 : 57);
  mesh.quality = cloud.quality;
  mesh.paramHash = cloud.paramHash;
  mesh.serial = nextMeshSerial();
  for (size_t i = 0; i + 2 < rawPoints.size(); i += 3u) {
    const float r = rawPoints[i + 0u];
    const float g = rawPoints[i + 1u];
    const float b = rawPoints[i + 2u];
    if (!cubeSliceContainsPoint(remap, r, g, b)) continue;
    const Vec3 pos = mapPointToPlotMode(remap, r, g, b);
    mesh.pointVerts.push_back(pos.x);
    mesh.pointVerts.push_back(pos.y);
    mesh.pointVerts.push_back(pos.z);
    float cr = 0.0f, cg = 0.0f, cb = 0.0f;
    const bool overflowPoint = overflowHighlightApplies(remap, r, g, b);
    if (overflowPoint) {
      cr = remap.overflowHighlightR;
      cg = remap.overflowHighlightG;
      cb = remap.overflowHighlightB;
    } else {
      mapDisplayColor(r, g, b, &cr, &cg, &cb);
    }
    mesh.pointColors.push_back(cr);
    mesh.pointColors.push_back(cg);
    mesh.pointColors.push_back(cb);
    mesh.pointColors.push_back(
        denseLumaProtectedAlpha(overflowPoint ? 0.95f : 0.72f,
                                payload.pointSize,
                                payload.pointDensity,
                                payload.resolution,
                                cr, cg, cb,
                                overflowPoint));
  }
  if (mesh.pointVerts.empty()) return false;
  mesh.pointCount = mesh.pointVerts.size() / 3u;
  *out = std::move(mesh);
  return true;
}

void buildEmptyInputCloudMesh(const InputCloudPayload& cloud, MeshData* out) {
  if (!out) return;
  MeshData mesh{};
  mesh.resolution = cloud.resolution <= 25 ? 25 : (cloud.resolution <= 41 ? 41 : 57);
  mesh.quality = cloud.quality;
  mesh.paramHash = cloud.paramHash;
  mesh.serial = nextMeshSerial();
  mesh.pointCount = 0;
  *out = std::move(mesh);
}

bool buildInputCloudMesh(const ResolvedPayload& payload,
                         const ViewerGpuCapabilities& gpuCaps,
                         ComputeSessionState* sessionState,
                         InputCloudComputeCache* computeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                         InputCloudCudaCache* cudaCache,
#endif
                         const InputCloudPayload& cloud,
                         MeshData* out) {
  if (!out) return false;
  std::vector<InputCloudSample> samples;
  std::vector<float> rawPoints;
  if (!parseInputCloudSamples(cloud, &samples)) return false;
  filterInputCloudSamples(payload, &samples);
  rawPoints.reserve(samples.size() * 3u);
  for (const auto& sample : samples) {
    rawPoints.push_back(sample.r);
    rawPoints.push_back(sample.g);
    rawPoints.push_back(sample.b);
  }
  if (rawPoints.empty()) {
    buildEmptyInputCloudMesh(cloud, out);
    return true;
  }
  if (gpuCaps.inputComputeEnabled && computeCache && canUseInputCloudComputePath(payload) &&
      buildInputCloudMeshOnGpu(payload, gpuCaps, sessionState, cloud, rawPoints, computeCache, out
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                               , cudaCache
#endif
                               )) {
    return true;
  }
  return buildInputCloudMeshCpu(payload, cloud, rawPoints, out);
}

PointSelectionSpec makePointSelectionSpec(uint64_t sourceSerial,
                                         size_t fullPointCount,
                                         float densityForView) {
  PointSelectionSpec spec{};
  spec.sourceSerial = sourceSerial;
  spec.fullPointCount = fullPointCount;
  spec.visiblePointCount = fullPointCount == 0
                               ? 0
                               : std::min(fullPointCount,
                                          std::max<size_t>(1, static_cast<size_t>(std::lround(
                                              static_cast<double>(fullPointCount) *
                                              clampf(densityForView, 0.1f, 4.0f)))));
  spec.needsThinning = spec.visiblePointCount > 0 && spec.visiblePointCount < fullPointCount;
  return spec;
}

bool buildCpuSampledPointDrawBuffers(const MeshData& mesh,
                                     const PointSelectionSpec& spec,
                                     PointDrawBuffers* out) {
  if (!out) return false;
  out->cpuVerts.clear();
  out->cpuColors.clear();
  out->verts = 0;
  out->colors = 0;
  out->pointCount = 0;
  out->available = false;
  out->sourceSerial = spec.sourceSerial;
  out->visiblePointCount = spec.visiblePointCount;
  if (!spec.needsThinning || mesh.pointCount == 0 || mesh.pointVerts.size() < mesh.pointCount * 3u ||
      mesh.pointColors.size() < mesh.pointCount * 4u) {
    return false;
  }
  out->cpuVerts.reserve(spec.visiblePointCount * 3u);
  out->cpuColors.reserve(spec.visiblePointCount * 4u);
  const double maxIndex = static_cast<double>(mesh.pointCount - 1);
  const double denom = static_cast<double>(std::max<size_t>(1, spec.visiblePointCount - 1));
  for (size_t i = 0; i < spec.visiblePointCount; ++i) {
    const size_t srcIndex = static_cast<size_t>(std::llround((static_cast<double>(i) / denom) * maxIndex));
    const size_t clampedIndex = std::min(srcIndex, mesh.pointCount - 1);
    const size_t vertOffset = clampedIndex * 3u;
    const size_t colorOffset = clampedIndex * 4u;
    out->cpuVerts.insert(out->cpuVerts.end(),
                         mesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(vertOffset),
                         mesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(vertOffset + 3u));
    out->cpuColors.insert(out->cpuColors.end(),
                          mesh.pointColors.begin() + static_cast<std::ptrdiff_t>(colorOffset),
                          mesh.pointColors.begin() + static_cast<std::ptrdiff_t>(colorOffset + 4u));
  }
  out->pointCount = spec.visiblePointCount;
  out->available = !out->cpuVerts.empty() && !out->cpuColors.empty();
  return out->available;
}

bool ensureInputCloudSampleComputeProgram(InputCloudSampleComputeCache* cache) {
  if (!cache) return false;
  if (cache->program != 0) return true;
  const ViewerGlComputeApi& api = viewerGlComputeApi();
  if (!api.available) return false;
  static const char* kShaderSrc = R"GLSL(
#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer SrcVertBuffer { float srcVertVals[]; };
layout(std430, binding = 1) readonly buffer SrcColorBuffer { float srcColorVals[]; };
layout(std430, binding = 2) writeonly buffer DstVertBuffer { float dstVertVals[]; };
layout(std430, binding = 3) writeonly buffer DstColorBuffer { float dstColorVals[]; };
uniform int uFullPointCount;
uniform int uVisiblePointCount;
void main() {
  uint index = gl_GlobalInvocationID.x;
  uint visible = uint(max(uVisiblePointCount, 0));
  uint full = uint(max(uFullPointCount, 0));
  if (index >= visible) return;
  uint srcIndex = 0u;
  if (visible > 1u && full > 1u) {
    float t = float(index) / float(visible - 1u);
    srcIndex = min(uint(floor(t * float(full - 1u) + 0.5)), full - 1u);
  }
  uint srcVertBase = srcIndex * 3u;
  uint srcColorBase = srcIndex * 4u;
  uint dstVertBase = index * 3u;
  uint dstColorBase = index * 4u;
  dstVertVals[dstVertBase + 0u] = srcVertVals[srcVertBase + 0u];
  dstVertVals[dstVertBase + 1u] = srcVertVals[srcVertBase + 1u];
  dstVertVals[dstVertBase + 2u] = srcVertVals[srcVertBase + 2u];
  dstColorVals[dstColorBase + 0u] = srcColorVals[srcColorBase + 0u];
  dstColorVals[dstColorBase + 1u] = srcColorVals[srcColorBase + 1u];
  dstColorVals[dstColorBase + 2u] = srcColorVals[srcColorBase + 2u];
  dstColorVals[dstColorBase + 3u] = srcColorVals[srcColorBase + 3u];
}
)GLSL";
  const GLuint shader = api.createShader(GL_COMPUTE_SHADER);
  if (shader == 0) return false;
  api.shaderSource(shader, 1, &kShaderSrc, nullptr);
  api.compileShader(shader);
  GLint ok = 0;
  api.getShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    logViewerEvent(std::string("Input-cloud thinning shader compile failed: ") + readShaderLog(shader, false, api));
    api.deleteShader(shader);
    return false;
  }
  const GLuint program = api.createProgram();
  if (program == 0) {
    api.deleteShader(shader);
    return false;
  }
  api.attachShader(program, shader);
  api.linkProgram(program);
  api.deleteShader(shader);
  api.getProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    logViewerEvent(std::string("Input-cloud thinning program link failed: ") + readShaderLog(program, true, api));
    api.deleteProgram(program);
    return false;
  }
  cache->program = program;
  cache->fullPointCountLoc = api.getUniformLocation(program, "uFullPointCount");
  cache->visiblePointCountLoc = api.getUniformLocation(program, "uVisiblePointCount");
  cache->available = cache->fullPointCountLoc >= 0 && cache->visiblePointCountLoc >= 0;
  if (!cache->available) {
    logViewerEvent("Input-cloud thinning program missing one or more uniforms; falling back to CPU.");
    releaseInputCloudSampleComputeCache(cache);
  }
  return cache->available;
}

bool buildInputCloudSampledGlBuffers(const InputCloudComputeCache& sourceCache,
                                     InputCloudSampleComputeCache* cache,
                                     const PointSelectionSpec& spec,
                                     std::string* error) {
  if (!cache || !spec.needsThinning || sourceCache.verts == 0 || sourceCache.colors == 0) return false;
  if (cache->available && cache->builtSerial == spec.sourceSerial && cache->visiblePointCount == spec.visiblePointCount &&
      cache->verts != 0 && cache->colors != 0 && cache->pointCount == static_cast<GLsizei>(spec.visiblePointCount)) {
    return true;
  }
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  const ViewerGlComputeApi& computeApi = viewerGlComputeApi();
  if (!bufferApi.available || !computeApi.available || !ensureInputCloudSampleComputeProgram(cache)) return false;
  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if (!ensureBuffer(&cache->verts) || !ensureBuffer(&cache->colors)) {
    if (error) *error = "gl-buffer-allocation";
    return false;
  }
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->verts);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(spec.visiblePointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, cache->colors);
  bufferApi.bufferData(GL_SHADER_STORAGE_BUFFER,
                       static_cast<ViewerGLsizeiptr>(spec.visiblePointCount * 4u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  computeApi.useProgram(cache->program);
  computeApi.uniform1i(cache->fullPointCountLoc, static_cast<GLint>(spec.fullPointCount));
  computeApi.uniform1i(cache->visiblePointCountLoc, static_cast<GLint>(spec.visiblePointCount));
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sourceCache.verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, sourceCache.colors);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cache->verts);
  computeApi.bindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, cache->colors);
  const GLuint groups = static_cast<GLuint>((spec.visiblePointCount + 63u) / 64u);
  computeApi.dispatchCompute(groups, 1, 1);
  computeApi.memoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
  computeApi.useProgram(0);
  bufferApi.bindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  cache->builtSerial = spec.sourceSerial;
  cache->visiblePointCount = spec.visiblePointCount;
  cache->pointCount = static_cast<GLsizei>(spec.visiblePointCount);
  cache->available = true;
  return true;
}

#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
bool buildInputCloudSampledCudaBuffers(InputCloudCudaCache* sourceCache,
                                       InputCloudSampleCudaCache* sampleCache,
                                       const PointSelectionSpec& spec,
                                       std::string* error) {
  if (!sourceCache || !sampleCache || !spec.needsThinning || sourceCache->verts == 0 || sourceCache->colors == 0) return false;
  if (sampleCache->available && sampleCache->builtSerial == spec.sourceSerial &&
      sampleCache->pointCount == static_cast<GLsizei>(spec.visiblePointCount) &&
      sampleCache->verts != 0 && sampleCache->colors != 0) {
    return true;
  }
  const ViewerGlBufferApi& bufferApi = viewerGlBufferApi();
  auto ensureBuffer = [&](GLuint* id) {
    if (*id == 0) bufferApi.genBuffers(1, id);
    return *id != 0;
  };
  if (!bufferApi.available || !ensureBuffer(&sampleCache->verts) || !ensureBuffer(&sampleCache->colors)) {
    if (error) *error = "cuda-sampled-buffer-allocation";
    return false;
  }
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, sampleCache->verts);
  bufferApi.bufferData(GL_ARRAY_BUFFER,
                       static_cast<ViewerGLsizeiptr>(spec.visiblePointCount * 3u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, sampleCache->colors);
  bufferApi.bufferData(GL_ARRAY_BUFFER,
                       static_cast<ViewerGLsizeiptr>(spec.visiblePointCount * 4u * sizeof(float)),
                       nullptr,
                       GL_DYNAMIC_DRAW);
  bufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
  ChromaspaceCuda::InputSampleRequest request{};
  request.fullPointCount = static_cast<int>(spec.fullPointCount);
  request.visiblePointCount = static_cast<int>(spec.visiblePointCount);
  return ChromaspaceCuda::buildInputSampledMesh(reinterpret_cast<ChromaspaceCuda::InputCache*>(sourceCache),
                                                 reinterpret_cast<ChromaspaceCuda::InputSampleCache*>(sampleCache),
                                                 request,
                                                 spec.sourceSerial,
                                                 error);
}
#endif

bool buildInputCloudSampledMetalBuffers(const MeshData& mesh,
                                        InputCloudSampleMetalCache* cache,
                                        const PointSelectionSpec& spec,
                                        std::string* error) {
#if defined(__APPLE__)
  if (!cache || !spec.needsThinning) return false;
  if (cache->draw.available && cache->draw.sourceSerial == spec.sourceSerial &&
      cache->draw.visiblePointCount == spec.visiblePointCount && cache->draw.pointCount == spec.visiblePointCount) {
    return true;
  }
  ChromaspaceMetal::InputSampleRequest request{};
  request.fullPointCount = static_cast<int>(spec.fullPointCount);
  request.visiblePointCount = static_cast<int>(spec.visiblePointCount);
  cache->draw = PointDrawBuffers{};
  cache->draw.sourceSerial = spec.sourceSerial;
  cache->draw.visiblePointCount = spec.visiblePointCount;
  if (!ChromaspaceMetal::buildInputSampledMesh(request,
                                                mesh.pointVerts,
                                                mesh.pointColors,
                                                &cache->draw.cpuVerts,
                                                &cache->draw.cpuColors,
                                                error)) {
    return false;
  }
  cache->draw.pointCount = spec.visiblePointCount;
  cache->draw.available = !cache->draw.cpuVerts.empty() && !cache->draw.cpuColors.empty();
  return cache->draw.available;
#else
  (void)mesh;
  (void)cache;
  (void)spec;
  (void)error;
  return false;
#endif
}

bool buildInputPointDrawBuffers(const MeshData& mesh,
                                const PointSelectionSpec& spec,
                                const ViewerGpuCapabilities& gpuCaps,
                                ComputeSessionState* sessionState,
                                const InputCloudComputeCache& fullComputeCache,
                                InputCloudSampleComputeCache* sampleComputeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                InputCloudCudaCache* fullCudaCache,
                                InputCloudSampleCudaCache* sampleCudaCache,
#endif
                                InputCloudSampleMetalCache* sampleMetalCache,
                                bool diagnosticsEnabled,
                                PointDrawBuffers* out) {
  if (!out) return false;
  *out = PointDrawBuffers{};
  out->sourceSerial = spec.sourceSerial;
  out->visiblePointCount = spec.visiblePointCount;
  if (!spec.needsThinning) return false;
  std::string reason;
#if defined(__APPLE__)
  if (gpuCaps.sessionBackend == ViewerComputeBackendKind::MetalCompute && sessionState && !sessionState->inputMetalSamplingDemoted) {
    if (buildInputCloudSampledMetalBuffers(mesh, sampleMetalCache, spec, &reason)) {
      if (sampleMetalCache) *out = sampleMetalCache->draw;
      if (diagnosticsEnabled) logViewerDiagnostic(true, "Input thinning backend: metal-sampled");
      return out->available;
    }
    if (sessionState) sessionState->inputMetalSamplingDemoted = true;
    if (diagnosticsEnabled) logViewerDiagnostic(true, std::string("Input thinning fallback: metal -> cpu reason=") + (reason.empty() ? "runtime-failure" : reason));
  }
#else
#if defined(CHROMASPACE_VIEWER_HAS_CUDA)
  if (sessionWantsCuda(gpuCaps) && sessionState && !sessionState->inputCudaSamplingDemoted &&
      fullCudaCache && sampleCudaCache && fullCudaCache->available && fullCudaCache->builtSerial == spec.sourceSerial) {
    if (buildInputCloudSampledCudaBuffers(fullCudaCache, sampleCudaCache, spec, &reason)) {
      out->verts = sampleCudaCache->verts;
      out->colors = sampleCudaCache->colors;
      out->pointCount = spec.visiblePointCount;
      out->sourceSerial = spec.sourceSerial;
      out->visiblePointCount = spec.visiblePointCount;
      out->available = true;
      if (diagnosticsEnabled) logViewerDiagnostic(true, "Input thinning backend: cuda-sampled");
      return true;
    }
    sessionState->inputCudaSamplingDemoted = true;
    logViewerEvent(std::string("CUDA input thinning demoted to CPU: ") + (reason.empty() ? "runtime-failure" : reason));
    if (diagnosticsEnabled) logViewerDiagnostic(true, std::string("Input thinning fallback: cuda -> cpu reason=") + (reason.empty() ? "runtime-failure" : reason));
  }
  if (sessionWantsCuda(gpuCaps)) {
    return buildCpuSampledPointDrawBuffers(mesh, spec, out);
  }
#endif
  if (gpuCaps.inputComputeEnabled && sessionState && !sessionState->inputGlSamplingDemoted &&
      fullComputeCache.available && fullComputeCache.builtSerial == spec.sourceSerial &&
      sampleComputeCache) {
    if (buildInputCloudSampledGlBuffers(fullComputeCache, sampleComputeCache, spec, &reason)) {
      out->verts = sampleComputeCache->verts;
      out->colors = sampleComputeCache->colors;
      out->pointCount = spec.visiblePointCount;
      out->sourceSerial = spec.sourceSerial;
      out->visiblePointCount = spec.visiblePointCount;
      out->available = true;
      if (diagnosticsEnabled) logViewerDiagnostic(true, "Input thinning backend: gl-sampled");
      return true;
    }
    sessionState->inputGlSamplingDemoted = true;
    logViewerEvent(std::string("GL input thinning demoted to CPU: ") + (reason.empty() ? "runtime-failure" : reason));
    if (diagnosticsEnabled) logViewerDiagnostic(true, std::string("Input thinning fallback: gl -> cpu reason=") + (reason.empty() ? "runtime-failure" : reason));
  }
#endif
  return buildCpuSampledPointDrawBuffers(mesh, spec, out);
}

bool runOverlayParityCheckWithBuffers(const ResolvedPayload& payload,
                                      GLuint vertsBuffer,
                                      GLuint colorsBuffer,
                                      const MeshData& gpuMesh,
                                      bool enabled) {
  if (!enabled) return true;
  MeshData cpuMesh{};
  if (!buildIdentityOverlayMesh(payload, &cpuMesh)) return true;
  const std::vector<float> cpuVerts(cpuMesh.pointVerts.begin(), cpuMesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(cpuMesh.pointVerts.size(), 24u)));
  const std::vector<float> cpuColors(cpuMesh.pointColors.begin(), cpuMesh.pointColors.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(cpuMesh.pointColors.size(), 32u)));
#if defined(__APPLE__)
  const std::vector<float> gpuVerts(gpuMesh.pointVerts.begin(), gpuMesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(gpuMesh.pointVerts.size(), cpuVerts.size())));
  const std::vector<float> gpuColors(gpuMesh.pointColors.begin(), gpuMesh.pointColors.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(gpuMesh.pointColors.size(), cpuColors.size())));
#else
  if (vertsBuffer == 0 || colorsBuffer == 0) return true;
  const std::vector<float> gpuVerts = gpuBufferSampleValues(vertsBuffer, cpuVerts.size());
  const std::vector<float> gpuColors = gpuBufferSampleValues(colorsBuffer, cpuColors.size());
#endif
  if (gpuVerts.empty() || gpuColors.empty()) {
    logParityCheckResult(enabled, "overlay", true, "gpu-readback-unavailable");
    return true;
  }
  const uint64_t cpuVertSig = floatSampleSignature(cpuVerts.data(), cpuVerts.size());
  const uint64_t cpuColorSig = floatSampleSignature(cpuColors.data(), cpuColors.size());
  const uint64_t gpuVertSig = floatSampleSignature(gpuVerts.data(), gpuVerts.size());
  const uint64_t gpuColorSig = floatSampleSignature(gpuColors.data(), gpuColors.size());
  const bool ok = cpuMesh.pointCount == gpuMesh.pointCount &&
                  sampledFloatsNear(cpuVerts, gpuVerts, 1e-6f) &&
                  sampledFloatsNear(cpuColors, gpuColors, 2e-4f);
  std::ostringstream os;
  os << "points cpu=" << cpuMesh.pointCount
     << " gpu=" << gpuMesh.pointCount
     << " vertSig=" << std::hex << cpuVertSig << "/" << gpuVertSig
     << " colorSig=" << cpuColorSig << "/" << gpuColorSig;
  logParityCheckResult(enabled, "overlay", ok, os.str());
  return ok;
}

bool runOverlayParityCheck(const ResolvedPayload& payload,
                           const OverlayComputeCache& cache,
                           const MeshData& gpuMesh,
                           bool enabled) {
  return runOverlayParityCheckWithBuffers(payload, cache.verts, cache.colors, gpuMesh, enabled);
}

bool runInputParityCheckWithBuffers(const ResolvedPayload& payload,
                                    const InputCloudPayload& cloud,
                                    const std::vector<float>& rawPoints,
                                    GLuint vertsBuffer,
                                    GLuint colorsBuffer,
                                    const MeshData& gpuMesh,
                                    bool enabled) {
  if (!enabled) return true;
  MeshData cpuMesh{};
  if (!buildInputCloudMeshCpu(payload, cloud, rawPoints, &cpuMesh)) return true;
  const std::vector<float> cpuVerts(cpuMesh.pointVerts.begin(), cpuMesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(cpuMesh.pointVerts.size(), 24u)));
  const std::vector<float> cpuColors(cpuMesh.pointColors.begin(), cpuMesh.pointColors.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(cpuMesh.pointColors.size(), 32u)));
#if defined(__APPLE__)
  const std::vector<float> gpuVerts(gpuMesh.pointVerts.begin(), gpuMesh.pointVerts.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(gpuMesh.pointVerts.size(), cpuVerts.size())));
  const std::vector<float> gpuColors(gpuMesh.pointColors.begin(), gpuMesh.pointColors.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(gpuMesh.pointColors.size(), cpuColors.size())));
#else
  if (vertsBuffer == 0 || colorsBuffer == 0) return true;
  const std::vector<float> gpuVerts = gpuBufferSampleValues(vertsBuffer, cpuVerts.size());
  const std::vector<float> gpuColors = gpuBufferSampleValues(colorsBuffer, cpuColors.size());
#endif
  if (gpuVerts.empty() || gpuColors.empty()) {
    logParityCheckResult(enabled, "input", true, "gpu-readback-unavailable");
    return true;
  }
  const uint64_t cpuVertSig = floatSampleSignature(cpuVerts.data(), cpuVerts.size());
  const uint64_t cpuColorSig = floatSampleSignature(cpuColors.data(), cpuColors.size());
  const uint64_t gpuVertSig = floatSampleSignature(gpuVerts.data(), gpuVerts.size());
  const uint64_t gpuColorSig = floatSampleSignature(gpuColors.data(), gpuColors.size());
  const bool ok = cpuMesh.pointCount == gpuMesh.pointCount &&
                  sampledFloatsNear(cpuVerts, gpuVerts, 1e-6f) &&
                  sampledFloatsNear(cpuColors, gpuColors, 2e-4f);
  std::ostringstream os;
  os << "points cpu=" << cpuMesh.pointCount
     << " gpu=" << gpuMesh.pointCount
     << " vertSig=" << std::hex << cpuVertSig << "/" << gpuVertSig
     << " colorSig=" << cpuColorSig << "/" << gpuColorSig;
  logParityCheckResult(enabled, "input", ok, os.str());
  return ok;
}

bool runInputParityCheck(const ResolvedPayload& payload,
                         const InputCloudPayload& cloud,
                         const std::vector<float>& rawPoints,
                         const InputCloudComputeCache& cache,
                         const MeshData& gpuMesh,
                         bool enabled) {
  return runInputParityCheckWithBuffers(payload, cloud, rawPoints, cache.verts, cache.colors, gpuMesh, enabled);
}

struct AppState {
  CameraState cam;
  Quat modelOrientation;
  std::string plotMode = "rgb";
  int orientAxisLock = 0;
  double orientAxisFeedbackUntil = 0.0;
  double slowFeedbackUntil = 0.0;
  double speedFeedbackUntil = 0.0;
  bool leftDown = false;
  bool panMode = false;
  bool shiftPanGesture = false;
  bool rollMode = false;
  bool zoomMode = false;
  float panVelocityX = 0.0f;
  float panVelocityY = 0.0f;
  int rollDirection = 1;
  double rollFeedbackUntil = 0.0;
  double lastX = 0.0;
  double lastY = 0.0;
  double hoverX = 0.0;
  double hoverY = 0.0;
  double orbitVirtualX = 0.0;
  double orbitVirtualY = 0.0;
  double lastClick = -10.0;
  bool keepOnTop = true;
  bool appliedTopmost = false;
  bool diagTransitions = false;
  bool parityChecks = false;
  ViewerGpuCapabilities gpuCaps;
  ComputeSessionState computeSession;
  std::string lastDrawSourceLabel;
  std::string lastOverlayComputeReason;
  std::string lastInputComputeReason;
  bool shiftHeld = false;
  bool ctrlHeld = false;
  bool altHeld = false;
  bool superHeld = false;
  bool rollKeyHeld = false;
  double lastHoverActivationAttempt = -10.0;
};

bool platformRollModifierPressed(const AppState& app) {
  if (app.rollKeyHeld) return true;
#if defined(__APPLE__)
  return app.superHeld;
#else
  return false;
#endif
}

struct HudTextRenderer {
  WorkshopText::FontAtlas atlas;
  GLuint texture = 0;
  bool available = false;
};

std::vector<std::string> defaultHudFontCandidates() {
  std::vector<std::string> paths;
  const std::string exeDir = viewerExecutableDir();
  if (!exeDir.empty()) {
    const std::filesystem::path bundled = std::filesystem::path(exeDir) / "OpenSans-Regular.ttf";
    if (std::filesystem::exists(bundled)) {
      paths.push_back(bundled.string());
    }
  }
  std::error_code ec;
  const std::filesystem::path buildBundled =
      std::filesystem::current_path(ec) / "artifacts" / "viewer" / "OpenSans-Regular.ttf";
  if (!ec && std::filesystem::exists(buildBundled)) {
    paths.push_back(buildBundled.string());
  }
#if defined(_WIN32)
  paths.push_back("C:\\Windows\\Fonts\\segoeui.ttf");
  paths.push_back("C:\\Windows\\Fonts\\arial.ttf");
#elif defined(__APPLE__)
  paths.push_back("/System/Library/Fonts/Supplemental/Helvetica.ttc");
  paths.push_back("/System/Library/Fonts/Supplemental/Arial.ttf");
  paths.push_back("/System/Library/Fonts/Apple Symbols.ttf");
#else
  paths.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
#endif
  return paths;
}

std::string defaultHudSymbolFontPath() {
#if defined(_WIN32)
  return "C:\\Windows\\Fonts\\seguisym.ttf";
#elif defined(__APPLE__)
  return "/System/Library/Fonts/Apple Symbols.ttf";
#else
  return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
#endif
}

bool initializeTextRenderer(HudTextRenderer* renderer,
                            const std::string& fontPath,
                            int pixelSize,
                            const char* label) {
  if (!renderer) return false;
  *renderer = HudTextRenderer{};
  if (fontPath.empty()) {
    logViewerEvent(std::string(label ? label : "Text renderer") + " disabled: font path is empty.");
    return false;
  }
  std::string error;
  // Small HUD text is baked at its actual on-screen size. Downscaling hinted
  // bitmap glyphs makes stem weights and spacing look inconsistent.
  if (!WorkshopText::loadFontAtlas(fontPath, pixelSize, &renderer->atlas, &error)) {
    logViewerEvent(std::string(label ? label : "Text renderer") + " disabled: " +
                   (error.empty() ? "font atlas build failed" : error));
    return false;
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glGenTextures(1, &renderer->texture);
  if (renderer->texture == 0) {
    logViewerEvent(std::string(label ? label : "Text renderer") + " disabled: failed to allocate GL texture.");
    renderer->atlas = WorkshopText::FontAtlas{};
    return false;
  }
  glBindTexture(GL_TEXTURE_2D, renderer->texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_ALPHA,
               renderer->atlas.width,
               renderer->atlas.height,
               0,
               GL_ALPHA,
               GL_UNSIGNED_BYTE,
               renderer->atlas.pixels.data());
  glBindTexture(GL_TEXTURE_2D, 0);
  renderer->available = true;
  logViewerEvent(std::string(label ? label : "Text renderer") + " enabled with font: " + fontPath);
  return true;
}

bool initializeHudTextRenderer(HudTextRenderer* renderer) {
  if (!renderer) return false;
  for (const std::string& fontPath : defaultHudFontCandidates()) {
    if (fontPath.empty()) continue;
    if (initializeTextRenderer(renderer, fontPath, 18, "HUD text")) {
      return true;
    }
  }
  *renderer = HudTextRenderer{};
  logViewerEvent("HUD text disabled: no usable font candidate loaded.");
  return false;
}

bool initializeHudSymbolRenderer(HudTextRenderer* renderer) {
  return initializeTextRenderer(renderer, defaultHudSymbolFontPath(), 22, "HUD symbol text");
}

void releaseHudTextRenderer(HudTextRenderer* renderer) {
  if (!renderer) return;
  if (renderer->texture != 0) {
    glDeleteTextures(1, &renderer->texture);
  }
  *renderer = HudTextRenderer{};
}

const HudTextRenderer* preferredHudRenderer(const HudTextRenderer* primary,
                                            const HudTextRenderer* fallback = nullptr) {
  if (primary && primary->available) return primary;
  if (fallback && fallback->available) return fallback;
  return nullptr;
}

void refreshModifierState(GLFWwindow* window, AppState* app) {
  if (!window || !app) return;
  app->shiftHeld = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS ||
                    nativeShiftModifierPressed());
  app->ctrlHeld = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                   glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                   nativeControlModifierPressed());
  app->altHeld = (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                  glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS ||
                  nativeAltModifierPressed());
  app->superHeld = (glfwGetKey(window, GLFW_KEY_LEFT_SUPER) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS ||
                    nativeSuperModifierPressed());
  app->rollKeyHeld = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
}

void applyMouseModifierBits(AppState* app, int mods) {
  if (!app) return;
  app->shiftHeld = app->shiftHeld || (mods & GLFW_MOD_SHIFT) != 0 || nativeShiftModifierPressed();
  app->ctrlHeld = app->ctrlHeld || (mods & GLFW_MOD_CONTROL) != 0 || nativeControlModifierPressed();
  app->altHeld = app->altHeld || (mods & GLFW_MOD_ALT) != 0 || nativeAltModifierPressed();
  app->superHeld = app->superHeld || (mods & GLFW_MOD_SUPER) != 0 || nativeSuperModifierPressed();
}

void keyCallback(GLFWwindow* window, int, int, int, int) {
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  refreshModifierState(window, app);
}

void tryActivateViewerOnHover(GLFWwindow* window, AppState* app) {
#if defined(__APPLE__)
  if (!window || !app) return;
  if (gWindowFocused.load()) return;
  const double now = glfwGetTime();
  if ((now - app->lastHoverActivationAttempt) < 0.12) return;
  app->lastHoverActivationAttempt = now;
  void* nativeWindow = glfwGetCocoaWindow(window);
  if (nativeWindow != nullptr && ChromaspaceMetal::activateWindow(nativeWindow)) {
    gWindowFocused.store(1);
    refreshModifierState(window, app);
    logViewerEvent("Activated viewer on hover.");
  }
#else
  (void)window;
  (void)app;
#endif
}

void cursorEnterCallback(GLFWwindow* window, int entered) {
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!window || !app || entered != GLFW_TRUE) return;
  tryActivateViewerOnHover(window, app);
}

void logComputeEligibilityTransitions(const ResolvedPayload& resolved, AppState* app) {
  if (!app || !app->diagTransitions) return;
  const std::string overlayReason = overlayComputeReason(resolved, app->gpuCaps, &app->computeSession);
  if (overlayReason != app->lastOverlayComputeReason) {
    logViewerDiagnostic(true, std::string("Overlay compute path: ") + overlayReason);
    app->lastOverlayComputeReason = overlayReason;
  }
  const std::string inputReason = inputCloudComputeReason(resolved, app->gpuCaps, &app->computeSession);
  if (inputReason != app->lastInputComputeReason) {
    logViewerDiagnostic(true, std::string("Input compute path: ") + inputReason);
    app->lastInputComputeReason = inputReason;
  }
}

void drawRgbAxesAndNeutralAxis() {
  static const GLfloat kAxes[] = {
      -1, -1, -1, 1.2f, -1, -1,
      -1, -1, -1, -1, 1.2f, -1,
      -1, -1, -1, -1, -1, 1.2f
  };
  static const GLfloat kNeutralAxis[] = {-1,-1,-1, 1,1,1};

  glEnableClientState(GL_VERTEX_ARRAY);
  glLineWidth(1.5f);
  glVertexPointer(3, GL_FLOAT, 0, kAxes);
  glColor4f(1.0f, 0.32f, 0.32f, 0.9f);
  glDrawArrays(GL_LINES, 0, 2);
  glColor4f(0.35f, 1.0f, 0.35f, 0.9f);
  glDrawArrays(GL_LINES, 2, 2);
  glColor4f(0.35f, 0.60f, 1.0f, 0.9f);
  glDrawArrays(GL_LINES, 4, 2);
  glLineWidth(1.2f);
  glVertexPointer(3, GL_FLOAT, 0, kNeutralAxis);
  glColor4f(1.0f, 1.0f, 1.0f, 0.38f);
  glDrawArrays(GL_LINES, 0, 2);
  glDisableClientState(GL_VERTEX_ARRAY);
}

void drawRgbGuide(const ResolvedPayload& payload) {
  static const GLfloat kCubeEdges[] = {
      -1,-1,-1,  1,-1,-1,   1,-1,-1,  1, 1,-1,   1, 1,-1, -1, 1,-1,  -1, 1,-1, -1,-1,-1,
      -1,-1, 1,  1,-1, 1,   1,-1, 1,  1, 1, 1,   1, 1, 1, -1, 1, 1,  -1, 1, 1, -1,-1, 1,
      -1,-1,-1, -1,-1, 1,   1,-1,-1,  1,-1, 1,   1, 1,-1,  1, 1, 1,  -1, 1,-1, -1, 1, 1
  };
  struct TetraRegion {
    bool enabled;
    int ids[4];
  };
  static const Vec3 kVerts[] = {
      {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f}, {-1.0f, -1.0f,  1.0f},
      { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f}, { 1.0f,  1.0f,  1.0f}
  };
  const bool showFullCube = !payload.cubeSlicingEnabled;

  glEnableClientState(GL_VERTEX_ARRAY);
  glLineWidth(1.15f);
  glColor4f(0.97f, 0.97f, 0.97f, 0.55f);
  if (showFullCube) {
    glVertexPointer(3, GL_FLOAT, 0, kCubeEdges);
    glDrawArrays(GL_LINES, 0, 24);
  } else {
    const TetraRegion regions[] = {
        {payload.cubeSliceRed,     {0, 1, 4, 7}},
        {payload.cubeSliceYellow,  {0, 2, 4, 7}},
        {payload.cubeSliceGreen,   {0, 2, 5, 7}},
        {payload.cubeSliceCyan,    {0, 3, 5, 7}},
        {payload.cubeSliceBlue,    {0, 3, 6, 7}},
        {payload.cubeSliceMagenta, {0, 1, 6, 7}},
    };
    bool drawn[8][8] = {};
    glDisableClientState(GL_VERTEX_ARRAY);
    glBegin(GL_LINES);
    for (const auto& region : regions) {
      if (!region.enabled) continue;
      for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
          const int ia = region.ids[a];
          const int ib = region.ids[b];
          const int lo = std::min(ia, ib);
          const int hi = std::max(ia, ib);
          if (drawn[lo][hi]) continue;
          drawn[lo][hi] = true;
          glVertex3f(kVerts[lo].x, kVerts[lo].y, kVerts[lo].z);
          glVertex3f(kVerts[hi].x, kVerts[hi].y, kVerts[hi].z);
        }
      }
    }
    glEnd();
  }
  glDisableClientState(GL_VERTEX_ARRAY);
  drawRgbAxesAndNeutralAxis();
}

void drawChromaticityTriangle(const PlotRemapSpec& spec,
                              WorkshopColor::ColorPrimariesId primariesId,
                              float r,
                              float g,
                              float b,
                              float alpha) {
  const WorkshopColor::PrimariesDefinition& primaries = WorkshopColor::primariesDefinition(primariesId);
  glColor4f(r, g, b, alpha);
  glBegin(GL_LINE_LOOP);
  const Vec3 red = mapChromaticityCoordsToViewer(spec, primaries.red, 0.0f);
  const Vec3 green = mapChromaticityCoordsToViewer(spec, primaries.green, 0.0f);
  const Vec3 blue = mapChromaticityCoordsToViewer(spec, primaries.blue, 0.0f);
  glVertex3f(red.x, red.y, red.z);
  glVertex3f(green.x, green.y, green.z);
  glVertex3f(blue.x, blue.y, blue.z);
  glEnd();
}

const std::array<uint8_t, 7>* bitmapGlyphRows(char glyph) {
  static const std::array<uint8_t, 7> kSpace = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  static const std::array<uint8_t, 7> kDash = {0x00, 0x00, 0x00, 0x1Fu, 0x00, 0x00, 0x00};
  static const std::array<uint8_t, 7> kDot = {0x00, 0x00, 0x00, 0x00, 0x00, 0x0Cu, 0x0Cu};
  static const std::array<uint8_t, 7> kColon = {0x00, 0x0Cu, 0x0Cu, 0x00, 0x0Cu, 0x0Cu, 0x00};
  static const std::array<uint8_t, 7> kSlash = {0x01u, 0x02u, 0x04u, 0x08u, 0x10u, 0x00u, 0x00u};
  static const std::array<uint8_t, 7> k0 = {0x0Eu, 0x11u, 0x13u, 0x15u, 0x19u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> k1 = {0x04u, 0x0Cu, 0x04u, 0x04u, 0x04u, 0x04u, 0x0Eu};
  static const std::array<uint8_t, 7> k2 = {0x0Eu, 0x11u, 0x01u, 0x02u, 0x04u, 0x08u, 0x1Fu};
  static const std::array<uint8_t, 7> k3 = {0x1Eu, 0x01u, 0x01u, 0x0Eu, 0x01u, 0x01u, 0x1Eu};
  static const std::array<uint8_t, 7> k4 = {0x02u, 0x06u, 0x0Au, 0x12u, 0x1Fu, 0x02u, 0x02u};
  static const std::array<uint8_t, 7> k5 = {0x1Fu, 0x10u, 0x10u, 0x1Eu, 0x01u, 0x01u, 0x1Eu};
  static const std::array<uint8_t, 7> k6 = {0x0Eu, 0x10u, 0x10u, 0x1Eu, 0x11u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> k7 = {0x1Fu, 0x01u, 0x02u, 0x04u, 0x08u, 0x08u, 0x08u};
  static const std::array<uint8_t, 7> k8 = {0x0Eu, 0x11u, 0x11u, 0x0Eu, 0x11u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> k9 = {0x0Eu, 0x11u, 0x11u, 0x0Fu, 0x01u, 0x01u, 0x0Eu};
  static const std::array<uint8_t, 7> kA = {0x0Eu, 0x11u, 0x11u, 0x1Fu, 0x11u, 0x11u, 0x11u};
  static const std::array<uint8_t, 7> kB = {0x1Eu, 0x11u, 0x11u, 0x1Eu, 0x11u, 0x11u, 0x1Eu};
  static const std::array<uint8_t, 7> kC = {0x0Eu, 0x11u, 0x10u, 0x10u, 0x10u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> kD = {0x1Cu, 0x12u, 0x11u, 0x11u, 0x11u, 0x12u, 0x1Cu};
  static const std::array<uint8_t, 7> kE = {0x1Fu, 0x10u, 0x10u, 0x1Eu, 0x10u, 0x10u, 0x1Fu};
  static const std::array<uint8_t, 7> kF = {0x1Fu, 0x10u, 0x10u, 0x1Eu, 0x10u, 0x10u, 0x10u};
  static const std::array<uint8_t, 7> kG = {0x0Eu, 0x11u, 0x10u, 0x17u, 0x11u, 0x11u, 0x0Fu};
  static const std::array<uint8_t, 7> kH = {0x11u, 0x11u, 0x11u, 0x1Fu, 0x11u, 0x11u, 0x11u};
  static const std::array<uint8_t, 7> kI = {0x0Eu, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x0Eu};
  static const std::array<uint8_t, 7> kJ = {0x01u, 0x01u, 0x01u, 0x01u, 0x11u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> kK = {0x11u, 0x12u, 0x14u, 0x18u, 0x14u, 0x12u, 0x11u};
  static const std::array<uint8_t, 7> kL = {0x10u, 0x10u, 0x10u, 0x10u, 0x10u, 0x10u, 0x1Fu};
  static const std::array<uint8_t, 7> kM = {0x11u, 0x1Bu, 0x15u, 0x15u, 0x11u, 0x11u, 0x11u};
  static const std::array<uint8_t, 7> kN = {0x11u, 0x19u, 0x15u, 0x13u, 0x11u, 0x11u, 0x11u};
  static const std::array<uint8_t, 7> kO = {0x0Eu, 0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> kP = {0x1Eu, 0x11u, 0x11u, 0x1Eu, 0x10u, 0x10u, 0x10u};
  static const std::array<uint8_t, 7> kQ = {0x0Eu, 0x11u, 0x11u, 0x11u, 0x15u, 0x12u, 0x0Du};
  static const std::array<uint8_t, 7> kR = {0x1Eu, 0x11u, 0x11u, 0x1Eu, 0x14u, 0x12u, 0x11u};
  static const std::array<uint8_t, 7> kS = {0x0Fu, 0x10u, 0x10u, 0x0Eu, 0x01u, 0x01u, 0x1Eu};
  static const std::array<uint8_t, 7> kT = {0x1Fu, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u, 0x04u};
  static const std::array<uint8_t, 7> kU = {0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0Eu};
  static const std::array<uint8_t, 7> kV = {0x11u, 0x11u, 0x11u, 0x11u, 0x11u, 0x0Au, 0x04u};
  static const std::array<uint8_t, 7> kW = {0x11u, 0x11u, 0x11u, 0x15u, 0x15u, 0x15u, 0x0Au};
  static const std::array<uint8_t, 7> kX = {0x11u, 0x11u, 0x0Au, 0x04u, 0x0Au, 0x11u, 0x11u};
  static const std::array<uint8_t, 7> kY = {0x11u, 0x11u, 0x0Au, 0x04u, 0x04u, 0x04u, 0x04u};
  static const std::array<uint8_t, 7> kZ = {0x1Fu, 0x01u, 0x02u, 0x04u, 0x08u, 0x10u, 0x1Fu};

  switch (glyph) {
    case ' ': return &kSpace;
    case '-': return &kDash;
    case '.': return &kDot;
    case ':': return &kColon;
    case '/': return &kSlash;
    case '0': return &k0;
    case '1': return &k1;
    case '2': return &k2;
    case '3': return &k3;
    case '4': return &k4;
    case '5': return &k5;
    case '6': return &k6;
    case '7': return &k7;
    case '8': return &k8;
    case '9': return &k9;
    case 'A': return &kA;
    case 'B': return &kB;
    case 'C': return &kC;
    case 'D': return &kD;
    case 'E': return &kE;
    case 'F': return &kF;
    case 'G': return &kG;
    case 'H': return &kH;
    case 'I': return &kI;
    case 'J': return &kJ;
    case 'K': return &kK;
    case 'L': return &kL;
    case 'M': return &kM;
    case 'N': return &kN;
    case 'O': return &kO;
    case 'P': return &kP;
    case 'Q': return &kQ;
    case 'R': return &kR;
    case 'S': return &kS;
    case 'T': return &kT;
    case 'U': return &kU;
    case 'V': return &kV;
    case 'W': return &kW;
    case 'X': return &kX;
    case 'Y': return &kY;
    case 'Z': return &kZ;
    default: return &kSpace;
  }
}

void drawWorldBitmapGlyph(char glyph, float x, float y, float z, float scale) {
  const std::array<uint8_t, 7>* rows = bitmapGlyphRows(glyph);
  if (!rows) return;
  const float pixelWidth = scale * 0.18f;
  const float pixelHeight = scale * 0.18f;
  const float cellStepX = scale * 0.20f;
  const float cellStepY = scale * 0.20f;

  glBegin(GL_QUADS);
  for (int row = 0; row < 7; ++row) {
    const uint8_t bits = (*rows)[row];
    for (int col = 0; col < 5; ++col) {
      const uint8_t mask = static_cast<uint8_t>(1u << (4 - col));
      if ((bits & mask) == 0) continue;
      const float px = x + static_cast<float>(col) * cellStepX;
      const float py = y + static_cast<float>(6 - row) * cellStepY;
      glVertex3f(px, py, z);
      glVertex3f(px + pixelWidth, py, z);
      glVertex3f(px + pixelWidth, py + pixelHeight, z);
      glVertex3f(px, py + pixelHeight, z);
    }
  }
  glEnd();
}

void drawWorldText(const std::string& text, float x, float y, float z, float scale) {
  if (text.empty()) return;
  float penX = x;
  for (char glyph : text) {
    char upper = glyph;
    if (upper >= 'a' && upper <= 'z') upper = static_cast<char>(upper - 'a' + 'A');
    drawWorldBitmapGlyph(upper, penX, y, z, scale);
    penX += scale * (upper == ' ' ? 0.56f : 1.18f);
  }
}

float bitmapTextWidth(const std::string& text, float scale) {
  float width = 0.0f;
  for (char glyph : text) {
    const char upper = (glyph >= 'a' && glyph <= 'z') ? static_cast<char>(glyph - 'a' + 'A') : glyph;
    width += scale * (upper == ' ' ? 0.56f : 1.18f);
  }
  return width;
}

void drawScreenText(const std::string& text, float x, float y, float scale) {
  if (text.empty()) return;
  float penX = x;
  for (char glyph : text) {
    char upper = glyph;
    if (upper >= 'a' && upper <= 'z') upper = static_cast<char>(upper - 'a' + 'A');
    drawWorldBitmapGlyph(upper, penX, y, 0.0f, scale);
    penX += scale * (upper == ' ' ? 0.56f : 1.18f);
  }
}

Vec3 wavelengthGuideColor(float wavelengthNm) {
  const float w = clampf(wavelengthNm, 380.0f, 700.0f);
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  if (w < 440.0f) {
    r = -(w - 440.0f) / 60.0f;
    b = 1.0f;
  } else if (w < 490.0f) {
    g = (w - 440.0f) / 50.0f;
    b = 1.0f;
  } else if (w < 510.0f) {
    g = 1.0f;
    b = -(w - 510.0f) / 20.0f;
  } else if (w < 580.0f) {
    r = (w - 510.0f) / 70.0f;
    g = 1.0f;
  } else if (w < 645.0f) {
    r = 1.0f;
    g = -(w - 645.0f) / 65.0f;
  } else {
    r = 1.0f;
  }
  float attenuation = 1.0f;
  if (w < 420.0f) {
    attenuation = 0.35f + 0.65f * (w - 380.0f) / 40.0f;
  } else if (w > 645.0f) {
    attenuation = 0.35f + 0.65f * (700.0f - w) / 55.0f;
  }
  auto adjust = [attenuation](float c) {
    return c <= 0.0f ? 0.0f : std::pow(c * attenuation, 0.80f);
  };
  return {adjust(r), adjust(g), adjust(b)};
}

Vec3 blackBodyGuideColor(float kelvin) {
  const float t = clampf((kelvin - 1800.0f) / (20000.0f - 1800.0f), 0.0f, 1.0f);
  struct Knot {
    float t;
    Vec3 color;
  };
  static const Knot kKnots[] = {
      {0.00f, {1.00f, 0.58f, 0.16f}},
      {0.18f, {1.00f, 0.78f, 0.30f}},
      {0.34f, {1.00f, 0.92f, 0.72f}},
      {0.52f, {0.95f, 0.97f, 1.00f}},
      {0.74f, {0.70f, 0.84f, 1.00f}},
      {1.00f, {0.50f, 0.72f, 1.00f}},
  };
  for (std::size_t i = 1; i < std::size(kKnots); ++i) {
    if (t <= kKnots[i].t) {
      const float span = std::max(kKnots[i].t - kKnots[i - 1].t, 1e-6f);
      const float localT = (t - kKnots[i - 1].t) / span;
      return {
          kKnots[i - 1].color.x + (kKnots[i].color.x - kKnots[i - 1].color.x) * localT,
          kKnots[i - 1].color.y + (kKnots[i].color.y - kKnots[i - 1].color.y) * localT,
          kKnots[i - 1].color.z + (kKnots[i].color.z - kKnots[i - 1].color.z) * localT,
      };
    }
  }
  return kKnots[std::size(kKnots) - 1].color;
}

void drawWorldAtlasText(const HudTextRenderer& renderer,
                        const std::string& text,
                        float baselineX,
                        float baselineY,
                        float z,
                        float scale,
                        float alpha,
                        float r,
                        float g,
                        float b) {
  if (!renderer.available || text.empty()) return;
  std::vector<WorkshopText::TextQuadVertex> quads;
  WorkshopText::appendTextQuads(renderer.atlas, text, baselineX, baselineY, scale, &quads);
  if (quads.empty()) return;
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, renderer.texture);
  glBegin(GL_TRIANGLES);
  glColor4f(r, g, b, alpha);
  for (const auto& v : quads) {
    glTexCoord2f(v.u, v.v);
    glVertex3f(v.x, v.y, z);
  }
  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}

void drawWorldAtlasTextShadowed(const HudTextRenderer& renderer,
                                const std::string& text,
                                float baselineX,
                                float baselineY,
                                float z,
                                float scale,
                                float alpha,
                                float r,
                                float g,
                                float b,
                                float shadowOffset) {
  if (!renderer.available || text.empty()) return;
  const float offset = std::max(0.0008f, shadowOffset);
  drawWorldAtlasText(renderer, text, baselineX + offset, baselineY - offset, z - 0.0002f,
                     scale, alpha * 0.56f, 0.02f, 0.02f, 0.025f);
  drawWorldAtlasText(renderer, text, baselineX - offset * 0.45f, baselineY - offset * 0.45f, z - 0.0001f,
                     scale, alpha * 0.26f, 0.02f, 0.02f, 0.025f);
  drawWorldAtlasText(renderer, text, baselineX, baselineY, z, scale, alpha, r, g, b);
}

float worldAtlasScaleForHeight(const HudTextRenderer& renderer, float targetHeight) {
  if (!renderer.available) return 0.0f;
  const float metricHeight = static_cast<float>(std::max(1, renderer.atlas.ascent + renderer.atlas.descent));
  return targetHeight / metricHeight;
}

float worldUnitsPerScreenPixel(const CameraState& cam, int viewportHeight, float fovyDegrees) {
  if (viewportHeight <= 0) return 0.0f;
  const float tanHalfFovy = tanHalfFovDegrees(fovyDegrees);
  if (cam.orthographic) {
    const float orthoHalfHeight = std::max(kMinOrthoHalfHeight, cam.distance * tanHalfFovy);
    return (orthoHalfHeight * 2.0f) / static_cast<float>(viewportHeight);
  }
  const float viewDepth = std::max(0.05f, cam.distance);
  return (2.0f * viewDepth * tanHalfFovy) / static_cast<float>(viewportHeight);
}

float planckianLabelWorldHeight(const CameraState& cam, int viewportHeight, float fovyDegrees) {
  const float worldPerPixel = worldUnitsPerScreenPixel(cam, viewportHeight, fovyDegrees);
  if (worldPerPixel <= 0.0f) return 0.022f;
  const float closeBoost = computeTightZoomBlend(cam.distance);
  const float wideBoost = clampf((cam.distance - 6.2f) / 8.0f, 0.0f, 1.0f);
  const float farAttenuation = clampf((cam.distance - 12.5f) / 10.0f, 0.0f, 1.0f);
  const float targetPixels =
      clampf(20.2f + closeBoost * 10.0f + wideBoost * 4.8f - farAttenuation * 7.1f, 17.0f, 31.0f);
  return worldPerPixel * targetPixels;
}

void drawChromaticityBlackBodyGuide(const PlotRemapSpec& spec,
                                    const CameraState& cam,
                                    int viewportHeight,
                                    float fovyDegrees,
                                    const HudTextRenderer* hudText) {
  constexpr float kBlackBodyMinKelvin = 1800.0f;
  constexpr float kBlackBodyMaxKelvin = 20000.0f;
  const auto curve = WorkshopColor::blackBodyChromaticityCurve(kBlackBodyMinKelvin, kBlackBodyMaxKelvin, 480);
  if (curve.size() < 2) return;

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glLineWidth(1.5f);
  glBegin(GL_LINE_STRIP);
  for (std::size_t i = 0; i < curve.size(); ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(curve.size() - 1);
    const float kelvin = kBlackBodyMinKelvin + (kBlackBodyMaxKelvin - kBlackBodyMinKelvin) * t;
    const Vec3 color = blackBodyGuideColor(kelvin);
    glColor4f(color.x, color.y, color.z, 0.90f);
    const Vec3 pos = mapChromaticityCoordsToViewer(spec, curve[i], 0.0f);
    glVertex3f(pos.x, pos.y, pos.z + 0.003f);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  struct BlackBodyAnchor {
    float kelvin;
    float dx;
    float dy;
  };
  const BlackBodyAnchor anchors[] = {
      {2000.0f, 0.020f, 0.014f},
      {3200.0f, 0.021f, 0.014f},
      {4000.0f, 0.020f, -0.019f},
      {5000.0f, 0.021f, -0.022f},
      {6500.0f, -0.050f, 0.014f},
      {10000.0f, -0.068f, 0.018f},
  };

  glPointSize(5.2f);
  glBegin(GL_POINTS);
  for (const auto& anchor : anchors) {
    WorkshopColor::Vec2f xy{};
    if (!WorkshopColor::blackBodyChromaticity(anchor.kelvin, &xy)) continue;
    const Vec3 pos = mapChromaticityCoordsToViewer(spec, xy, 0.0f);
    const Vec3 color = blackBodyGuideColor(anchor.kelvin);
    glColor4f(color.x, color.y, color.z, 0.98f);
    glVertex3f(pos.x, pos.y, pos.z + 0.005f);
  }
  glEnd();

  glColor4f(0.98f, 0.94f, 0.84f, 0.90f);
  glLineWidth(1.1f);
  const float labelWorldHeight = planckianLabelWorldHeight(cam, viewportHeight, fovyDegrees);
  const float labelScale = worldAtlasScaleForHeight(hudText ? *hudText : HudTextRenderer{}, labelWorldHeight);
  const float shadowOffset = labelWorldHeight * 0.16f;
  for (const auto& anchor : anchors) {
    WorkshopColor::Vec2f xy{};
    if (!WorkshopColor::blackBodyChromaticity(anchor.kelvin, &xy)) continue;
    const Vec3 pos = mapChromaticityCoordsToViewer(spec, xy, 0.0f);
    const std::string label = std::to_string(static_cast<int>(anchor.kelvin)) + "K";
    if (hudText && hudText->available && labelScale > 0.0f) {
      drawWorldAtlasTextShadowed(*hudText, label, pos.x + anchor.dx, pos.y + anchor.dy, pos.z + 0.006f,
                                 labelScale, 0.94f, 0.98f, 0.94f, 0.84f, shadowOffset);
    } else {
      drawWorldText(label, pos.x + anchor.dx, pos.y + anchor.dy, pos.z + 0.006f, 0.016f);
    }
  }

  const float nearestKelvin =
      WorkshopColor::nearestBlackBodyTemperature(spec.chromaticityWhite, 1800.0f, 20000.0f);
  WorkshopColor::Vec2f nearestXy{};
  if (nearestKelvin > 0.0f && WorkshopColor::blackBodyChromaticity(nearestKelvin, &nearestXy)) {
    const Vec3 white = mapChromaticityCoordsToViewer(spec, spec.chromaticityWhite, 0.0f);
    const Vec3 nearest = mapChromaticityCoordsToViewer(spec, nearestXy, 0.0f);
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor4f(0.96f, 0.92f, 0.70f, 0.44f);
    glVertex3f(white.x, white.y, white.z + 0.004f);
    glVertex3f(nearest.x, nearest.y, nearest.z + 0.004f);
    glEnd();
    glColor4f(0.98f, 0.96f, 0.88f, 0.92f);
    const std::string label = std::to_string(static_cast<int>(nearestKelvin)) + "K";
    if (hudText && hudText->available && labelScale > 0.0f) {
      drawWorldAtlasTextShadowed(*hudText, label, white.x + 0.020f, white.y + 0.014f, white.z + 0.006f,
                                 labelScale, 0.94f, 0.98f, 0.96f, 0.88f, shadowOffset);
    } else {
      drawWorldText(label, white.x + 0.018f, white.y + 0.012f, white.z + 0.006f, 0.016f);
    }
  }
}

void drawHudTextLine(const HudTextRenderer& renderer,
                     const std::string& text,
                     float x,
                     float baselineY,
                     float scale,
                     float alpha,
                     float r,
                     float g,
                     float b) {
  if (!renderer.available || text.empty()) return;
  x = std::round(x);
  baselineY = std::round(baselineY);
  std::vector<WorkshopText::TextQuadVertex> quads;
  WorkshopText::appendTextQuads(renderer.atlas, text, x, baselineY, scale, &quads);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, renderer.texture);
  glBegin(GL_TRIANGLES);
  glColor4f(r, g, b, alpha);
  for (const auto& v : quads) {
    glTexCoord2f(v.u, v.v);
    glVertex2f(v.x, v.y);
  }
  glEnd();
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_2D);
}

void drawHudBackdrop(float x0, float y0, float x1, float y1, float alpha) {
  if (alpha <= 0.0f || x1 <= x0 || y1 <= y0) return;
  glDisable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glColor4f(0.02f, 0.02f, 0.03f, alpha * 0.85f);
  glVertex2f(x0, y0);
  glVertex2f(x1, y0);
  glColor4f(0.02f, 0.02f, 0.03f, alpha * 0.28f);
  glVertex2f(x1, y1);
  glVertex2f(x0, y1);
  glEnd();
}

void drawChromaticityInfoOverlay(const PlotRemapSpec& spec,
                                 int width,
                                 int height,
                                 double hoverX,
                                 double hoverY,
                                 const HudTextRenderer& renderer) {
  if (width <= 0 || height <= 0) return;
  const WorkshopColor::PrimariesDefinition& inputPrimaries =
      WorkshopColor::primariesDefinition(spec.chromaticity.inputPrimaries);
  const WorkshopColor::TransferFunctionDefinition& inputTransfer =
      WorkshopColor::transferFunctionDefinition(spec.chromaticity.inputTransfer);
  const std::string referenceLine =
      std::string("Reference: ") +
      (spec.chromaticity.referenceBasis == WorkshopColor::ChromaticityReferenceBasis::InputObserver
           ? "Input Observer"
           : "CIE Standard Observer");
  const std::string overlayLine =
      std::string("Overlay: ") +
      (spec.chromaticity.overlayEnabled
           ? WorkshopColor::primariesDefinition(spec.chromaticity.overlayPrimaries).label
           : "None");

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  const float scale = renderer.available ? 1.0f : 6.4f;
  const float leftX = 16.0f;
  const float bottomY = 8.0f;
  const float lineAdvance = renderer.available ? static_cast<float>(renderer.atlas.lineHeight) + 1.0f : 11.5f;
  const float line2Y = renderer.available
                           ? bottomY + static_cast<float>(renderer.atlas.descent + 1)
                           : bottomY + 5.0f;
  const float line1Y = line2Y + lineAdvance;
  const float rightWidth = renderer.available
                               ? std::max(WorkshopText::measureTextWidth(renderer.atlas, referenceLine, scale),
                                          WorkshopText::measureTextWidth(renderer.atlas, overlayLine, scale))
                               : std::max(bitmapTextWidth(referenceLine, scale), bitmapTextWidth(overlayLine, scale));
  const float rightX = static_cast<float>(width) - 16.0f - rightWidth;
  const float boxHeight = renderer.available
                              ? static_cast<float>(renderer.atlas.ascent + renderer.atlas.descent) + lineAdvance + 8.0f
                              : 24.0f;
  const float leftWidth = renderer.available
                              ? std::max(WorkshopText::measureTextWidth(renderer.atlas, inputPrimaries.label, scale),
                                         WorkshopText::measureTextWidth(renderer.atlas, inputTransfer.label, scale))
                              : std::max(bitmapTextWidth(inputPrimaries.label, scale), bitmapTextWidth(inputTransfer.label, scale));
  const bool hoverLeft = hoverX >= leftX - 6.0f && hoverX <= leftX + leftWidth + 6.0f &&
                         (static_cast<double>(height) - hoverY) >= line2Y - 4.0f &&
                         (static_cast<double>(height) - hoverY) <= line2Y + boxHeight;
  const bool hoverRight = hoverX >= rightX - 6.0f && hoverX <= rightX + rightWidth + 6.0f &&
                          (static_cast<double>(height) - hoverY) >= line2Y - 4.0f &&
                          (static_cast<double>(height) - hoverY) <= line2Y + boxHeight;
  const float leftAlpha = hoverLeft ? 0.76f : 0.12f;
  const float rightAlpha = hoverRight ? 0.76f : 0.12f;
  const float leftBackdropAlpha = hoverLeft ? 0.17f : 0.08f;
  const float rightBackdropAlpha = hoverRight ? 0.17f : 0.08f;

  drawHudBackdrop(leftX - 8.0f, line2Y - 6.0f, leftX + leftWidth + 10.0f, line1Y + 8.0f, leftBackdropAlpha);
  drawHudBackdrop(rightX - 10.0f, line2Y - 6.0f, rightX + rightWidth + 8.0f, line1Y + 8.0f, rightBackdropAlpha);

  if (renderer.available) {
    drawHudTextLine(renderer, inputPrimaries.label, leftX, line1Y, scale, leftAlpha, 0.92f, 0.94f, 0.97f);
    drawHudTextLine(renderer, inputTransfer.label, leftX, line2Y, scale, leftAlpha, 0.82f, 0.85f, 0.89f);
    drawHudTextLine(renderer, referenceLine, rightX, line1Y, scale, rightAlpha, 0.92f, 0.94f, 0.97f);
    drawHudTextLine(renderer, overlayLine, rightX, line2Y, scale, rightAlpha, 0.82f, 0.85f, 0.89f);
  } else {
    glColor4f(0.0f, 0.0f, 0.0f, leftAlpha * 0.55f);
    drawScreenText(inputPrimaries.label, leftX + 1.5f, line1Y - 1.5f, scale);
    drawScreenText(inputTransfer.label, leftX + 1.5f, line2Y - 1.5f, scale);
    glColor4f(0.94f, 0.96f, 1.0f, leftAlpha);
    drawScreenText(inputPrimaries.label, leftX, line1Y, scale);
    drawScreenText(inputTransfer.label, leftX, line2Y, scale);
    glColor4f(0.0f, 0.0f, 0.0f, rightAlpha * 0.55f);
    drawScreenText(referenceLine, rightX + 1.5f, line1Y - 1.5f, scale);
    drawScreenText(overlayLine, rightX + 1.5f, line2Y - 1.5f, scale);
    glColor4f(0.98f, 0.95f, 0.88f, rightAlpha);
    drawScreenText(referenceLine, rightX, line1Y, scale);
    glColor4f(0.96f, 0.91f, 0.82f, rightAlpha);
    drawScreenText(overlayLine, rightX, line2Y, scale);
  }

  glDisable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawChromaticityWhitePoint(const PlotRemapSpec& spec) {
  const Vec3 white = mapChromaticityCoordsToViewer(spec, spec.chromaticityWhite, 0.0f);
  const float arm = 0.028f;
  glColor4f(1.0f, 1.0f, 1.0f, 0.78f);
  glBegin(GL_LINES);
  glVertex3f(white.x - arm, white.y, white.z);
  glVertex3f(white.x + arm, white.y, white.z);
  glVertex3f(white.x, white.y, white.z - arm);
  glVertex3f(white.x, white.y, white.z + arm);
  glEnd();
}

void drawChromaticityGuide(const ResolvedPayload& payload,
                           const CameraState& cam,
                           int viewportHeight,
                           float fovyDegrees,
                           const HudTextRenderer* hudText) {
  const PlotRemapSpec spec = makePlotRemapSpec(payload);
  const auto& locus = WorkshopColor::cie1931XyzCmfs5nm();
  size_t locusLast = locus.size() - 1;
  if (locusLast > 0) {
    const WorkshopColor::Vec3f& first = locus.front();
    const WorkshopColor::Vec3f& last = locus.back();
    const bool wrapped = std::fabs(first.x - last.x) < 1e-6f &&
                         std::fabs(first.y - last.y) < 1e-6f &&
                         std::fabs(first.z - last.z) < 1e-6f;
    if (wrapped) locusLast -= 1;
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glLineWidth(1.2f);
  glBegin(GL_LINE_STRIP);
  for (size_t i = 0; i <= locusLast; ++i) {
    const WorkshopColor::Vec2f xy =
        WorkshopColor::xyzToXy(locus[i], spec.chromaticityWhite);
    const Vec3 pos = mapChromaticityCoordsToViewer(spec, xy, 0.0f);
    const Vec3 color = wavelengthGuideColor(380.0f + static_cast<float>(i) * 5.0f);
    glColor4f(color.x, color.y, color.z, 0.88f);
    glVertex3f(pos.x, pos.y, pos.z);
  }
  glEnd();
  glDisable(GL_LINE_SMOOTH);

  const Vec3 purpleStart =
      mapChromaticityCoordsToViewer(spec, WorkshopColor::xyzToXy(locus.front(), spec.chromaticityWhite), 0.0f);
  const Vec3 purpleEnd =
      mapChromaticityCoordsToViewer(spec, WorkshopColor::xyzToXy(locus[locusLast], spec.chromaticityWhite), 0.0f);
  glColor4f(0.90f, 0.68f, 0.96f, 0.52f);
  glBegin(GL_LINES);
  glVertex3f(purpleStart.x, purpleStart.y, purpleStart.z);
  glVertex3f(purpleEnd.x, purpleEnd.y, purpleEnd.z);
  glEnd();

  if (payload.chromaticityPlanckianLocus) {
    drawChromaticityBlackBodyGuide(spec, cam, viewportHeight, fovyDegrees, hudText);
  }

  glLineWidth(1.15f);
  drawChromaticityTriangle(spec, spec.chromaticity.inputPrimaries, 0.96f, 0.96f, 0.98f, 0.56f);
  if (spec.chromaticity.overlayEnabled) {
    drawChromaticityTriangle(spec, spec.chromaticity.overlayPrimaries, 1.0f, 0.78f, 0.42f, 0.78f);
  }
  drawChromaticityWhitePoint(spec);
}

void drawCylinderGuide(float ringY, float radius, float alpha) {
  const int segments = 96;
  glColor4f(0.97f, 0.97f, 0.97f, alpha);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < segments; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(segments);
    const float a = t * 6.28318530717958647692f;
    glVertex3f(std::cos(a) * radius, ringY, std::sin(a) * radius);
  }
  glEnd();
}

void drawHslGuide() {
  const float ringAlpha = 0.52f;
  const float spokeAlpha = 0.36f;
  const float axisAlpha = 0.65f;
  const float radius = 1.0f;

  glLineWidth(1.15f);
  drawCylinderGuide(0.0f, radius, ringAlpha);

  glLineWidth(1.1f);
  glColor4f(0.92f, 0.92f, 0.95f, spokeAlpha);
  glBegin(GL_LINES);
  for (int i = 0; i < 6; ++i) {
    const float a = (static_cast<float>(i) / 6.0f) * 6.28318530717958647692f;
    const float x = std::cos(a) * radius;
    const float z = std::sin(a) * radius;
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(x, 0.0f, z);
    glVertex3f(0.0f, -1.0f, 0.0f);
    glVertex3f(x, 0.0f, z);
  }
  glEnd();

  glLineWidth(1.35f);
  glColor4f(1.0f, 1.0f, 1.0f, axisAlpha);
  glBegin(GL_LINES);
  glVertex3f(0.0f, -1.2f, 0.0f);
  glVertex3f(0.0f, 1.2f, 0.0f);
  glEnd();
}

void drawHsvGuide() {
  static const GLfloat kTopHexagon[] = {
      1.0f,  1.0f,  0.0f,
      0.5f,  1.0f,  0.8660254f,
     -0.5f,  1.0f,  0.8660254f,
     -1.0f,  1.0f,  0.0f,
     -0.5f,  1.0f, -0.8660254f,
      0.5f,  1.0f, -0.8660254f
  };
  const GLfloat apex[3] = {0.0f, -1.0f, 0.0f};

  glLineWidth(1.15f);
  glColor4f(0.97f, 0.97f, 0.97f, 0.58f);
  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, kTopHexagon);
  glDrawArrays(GL_LINE_LOOP, 0, 6);

  glColor4f(0.92f, 0.92f, 0.95f, 0.34f);
  glBegin(GL_LINES);
  for (int i = 0; i < 6; ++i) {
    const float* v = &kTopHexagon[i * 3];
    glVertex3fv(apex);
    glVertex3f(v[0], v[1], v[2]);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(v[0], v[1], v[2]);
  }
  glEnd();

  glDisableClientState(GL_VERTEX_ARRAY);
  glColor4f(1.0f, 1.0f, 1.0f, 0.6f);
  glBegin(GL_LINES);
  glVertex3f(0.0f, -1.2f, 0.0f);
  glVertex3f(0.0f, 1.2f, 0.0f);
  glEnd();
}

void drawCircularHsvGuide() {
  const float ringAlpha = 0.52f;
  const float spokeAlpha = 0.34f;
  const float axisAlpha = 0.6f;
  const float radius = 1.0f;
  const float topY = 1.0f;
  const float bottomY = -1.0f;

  glLineWidth(1.15f);
  drawCylinderGuide(topY, radius, ringAlpha);
  drawCylinderGuide(bottomY, radius, 0.26f);

  glLineWidth(1.1f);
  glColor4f(0.92f, 0.92f, 0.95f, spokeAlpha);
  glBegin(GL_LINES);
  for (int i = 0; i < 6; ++i) {
    const float a = (static_cast<float>(i) / 6.0f) * 6.28318530717958647692f;
    const float x = std::cos(a) * radius;
    const float z = std::sin(a) * radius;
    glVertex3f(x, bottomY, z);
    glVertex3f(x, topY, z);
    glVertex3f(0.0f, topY, 0.0f);
    glVertex3f(x, topY, z);
  }
  glEnd();

  glLineWidth(1.35f);
  glColor4f(1.0f, 1.0f, 1.0f, axisAlpha);
  glBegin(GL_LINES);
  glVertex3f(0.0f, -1.2f, 0.0f);
  glVertex3f(0.0f, 1.2f, 0.0f);
  glEnd();
}

void drawCircularHslGuide() {
  const float ringAlpha = 0.52f;
  const float spokeAlpha = 0.34f;
  const float axisAlpha = 0.6f;
  const float radius = 1.0f;
  const float topY = 0.96f;
  const float bottomY = -0.96f;
  const float topPointY = 1.0f;
  const float bottomPointY = -1.0f;

  glLineWidth(1.15f);
  drawCylinderGuide(topY, radius, ringAlpha);
  drawCylinderGuide(bottomY, radius, 0.26f);

  glLineWidth(1.1f);
  glColor4f(0.92f, 0.92f, 0.95f, spokeAlpha);
  glBegin(GL_LINES);
  for (int i = 0; i < 6; ++i) {
    const float a = (static_cast<float>(i) / 6.0f) * 6.28318530717958647692f;
    const float x = std::cos(a) * radius;
    const float z = std::sin(a) * radius;
    glVertex3f(x, bottomY, z);
    glVertex3f(x, topY, z);
    glVertex3f(0.0f, topPointY, 0.0f);
    glVertex3f(x, topY, z);
    glVertex3f(0.0f, bottomPointY, 0.0f);
    glVertex3f(x, bottomY, z);
  }
  glEnd();

  glLineWidth(1.35f);
  glColor4f(1.0f, 1.0f, 1.0f, axisAlpha);
  glBegin(GL_LINES);
  glVertex3f(0.0f, -1.2f, 0.0f);
  glVertex3f(0.0f, 1.2f, 0.0f);
  glEnd();
}

void drawMappedEdgeCurve(const ResolvedPayload& payload,
                         float r0, float g0, float b0,
                         float r1, float g1, float b1,
                         int steps,
                         float alpha) {
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(std::max(1, steps));
    const float r = r0 + (r1 - r0) * t;
    const float g = g0 + (g1 - g0) * t;
    const float b = b0 + (b1 - b0) * t;
    const Vec3 pos = mapPointToPlotMode(payload, r, g, b);
    float cr = 0.0f;
    float cg = 0.0f;
    float cb = 0.0f;
    mapDisplayColor(r, g, b, &cr, &cg, &cb);
    glColor4f(cr, cg, cb, alpha);
    glVertex3f(pos.x, pos.y, pos.z);
  }
  glEnd();
}

// Non-cube plots draw their guide by remapping familiar RGB cube edges into the active plot geometry.
// That keeps the guide semantically tied to RGB corners even when the geometry itself is curved.
void drawMappedBoundaryGuide(const ResolvedPayload& payload) {
  const int steps = 56;
  glLineWidth(1.15f);
  drawMappedEdgeCurve(payload, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, steps, 0.78f);
  drawMappedEdgeCurve(payload, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, steps, 0.78f);
  drawMappedEdgeCurve(payload, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, steps, 0.78f);
  drawMappedEdgeCurve(payload, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, steps, 0.68f);
  drawMappedEdgeCurve(payload, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, steps, 0.68f);
  drawMappedEdgeCurve(payload, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, steps, 0.68f);
  glLineWidth(1.0f);
  drawMappedEdgeCurve(payload, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, steps, 0.48f);
  drawMappedEdgeCurve(payload, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, steps, 0.48f);
  drawMappedEdgeCurve(payload, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, steps, 0.48f);
  drawMappedEdgeCurve(payload, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, steps, 0.48f);
  drawMappedEdgeCurve(payload, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, steps, 0.48f);
  drawMappedEdgeCurve(payload, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, steps, 0.48f);
}

void drawChenEdgeCurve(float r0, float g0, float b0, float r1, float g1, float b1, int steps, float alpha) {
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(std::max(1, steps));
    const float r = r0 + (r1 - r0) * t;
    const float g = g0 + (g1 - g0) * t;
    const float b = b0 + (b1 - b0) * t;
    const Vec3 pos = mapChenPlot(r, g, b);
    float cr = 0.0f;
    float cg = 0.0f;
    float cb = 0.0f;
    mapDisplayColor(r, g, b, &cr, &cg, &cb);
    glColor4f(cr, cg, cb, alpha);
    glVertex3f(pos.x, pos.y, pos.z);
  }
  glEnd();
}

void drawChenGuide() {
  const int steps = 56;
  glLineWidth(1.25f);
  drawChenEdgeCurve(0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, steps, 0.78f);
  drawChenEdgeCurve(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, steps, 0.78f);
  drawChenEdgeCurve(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, steps, 0.78f);
  drawChenEdgeCurve(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, steps, 0.70f);
  drawChenEdgeCurve(1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, steps, 0.70f);
  drawChenEdgeCurve(1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, steps, 0.70f);

  glLineWidth(1.05f);
  drawChenEdgeCurve(1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, steps, 0.54f);
  drawChenEdgeCurve(1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, steps, 0.54f);
  drawChenEdgeCurve(0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, steps, 0.54f);
  drawChenEdgeCurve(0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, steps, 0.54f);
  drawChenEdgeCurve(0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, steps, 0.54f);
  drawChenEdgeCurve(0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, steps, 0.54f);

  const Vec3 black = mapChenPlot(0.0f, 0.0f, 0.0f);
  const Vec3 white = mapChenPlot(1.0f, 1.0f, 1.0f);
  glLineWidth(1.1f);
  glColor4f(1.0f, 1.0f, 1.0f, 0.42f);
  glBegin(GL_LINES);
  glVertex3f(black.x, black.y, black.z);
  glVertex3f(white.x, white.y, white.z);
  glEnd();
}

void drawReuleauxGuide() {
  glLineWidth(1.15f);
  drawCylinderGuide(-1.0f, 1.0f, 0.36f);
  drawCylinderGuide(1.0f, 1.0f, 0.56f);
  glColor4f(0.92f, 0.92f, 0.95f, 0.34f);
  glBegin(GL_LINES);
  for (int i = 0; i < 12; ++i) {
    const float a = (static_cast<float>(i) / 12.0f) * 6.28318530717958647692f;
    const float x = std::cos(a);
    const float z = std::sin(a);
    glVertex3f(x, -1.0f, z);
    glVertex3f(x, 1.0f, z);
  }
  glEnd();
  glColor4f(1.0f, 1.0f, 1.0f, 0.58f);
  glBegin(GL_LINES);
  glVertex3f(0.0f, -1.2f, 0.0f);
  glVertex3f(0.0f, 1.2f, 0.0f);
  glEnd();
}

void hueToRgbUnit(float hue01, float* outR, float* outG, float* outB) {
  if (!outR || !outG || !outB) return;
  const float h = wrapHue01(hue01) * 6.0f;
  const int sector = static_cast<int>(std::floor(h)) % 6;
  const float f = h - std::floor(h);
  const float q = 1.0f - f;
  switch (sector) {
    case 0: *outR = 1.0f; *outG = f;    *outB = 0.0f; break;
    case 1: *outR = q;    *outG = 1.0f; *outB = 0.0f; break;
    case 2: *outR = 0.0f; *outG = 1.0f; *outB = f;    break;
    case 3: *outR = 0.0f; *outG = q;    *outB = 1.0f; break;
    case 4: *outR = f;    *outG = 0.0f; *outB = 1.0f; break;
    case 5:
    default:*outR = 1.0f; *outG = 0.0f; *outB = q;    break;
  }
}

void drawVolumeSliceHueGuides(const ResolvedPayload& payload) {
  const struct SliceGuide {
    bool enabled;
    float hue;
  } guides[] = {
      {payload.cubeSliceRed, 0.0f / 6.0f},
      {payload.cubeSliceYellow, 1.0f / 6.0f},
      {payload.cubeSliceGreen, 2.0f / 6.0f},
      {payload.cubeSliceCyan, 3.0f / 6.0f},
      {payload.cubeSliceBlue, 4.0f / 6.0f},
      {payload.cubeSliceMagenta, 5.0f / 6.0f},
  };
  const int steps = 52;
  glLineWidth(1.1f);
  for (const auto& guide : guides) {
    if (!guide.enabled) continue;
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    hueToRgbUnit(guide.hue, &r, &g, &b);
    drawMappedEdgeCurve(payload, 0.0f, 0.0f, 0.0f, r, g, b, steps, 0.42f);
    drawMappedEdgeCurve(payload, 1.0f, 1.0f, 1.0f, r, g, b, steps, 0.34f);
  }
}

void drawOrientationLockIndicator(int width, int height, int axisLock, float pulse, float yOffset = 34.0f) {
  if (axisLock == 0 || width <= 0 || height <= 0) return;
  const float cx = static_cast<float>(width) - 34.0f;
  const float cy = yOffset;
  const float arm = 12.0f + 1.8f * pulse;
  const float faintAlpha = 0.18f;
  const float activeAlpha = 0.82f + 0.18f * pulse;
  const float activeWidth = 1.75f + 0.95f * pulse;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  glLineWidth(1.0f);
  glBegin(GL_LINES);
  glColor4f(1.0f, 1.0f, 1.0f, axisLock == 1 ? faintAlpha : activeAlpha);
  glVertex2f(cx, cy - arm);
  glVertex2f(cx, cy + arm);
  glColor4f(1.0f, 1.0f, 1.0f, axisLock == 2 ? faintAlpha : activeAlpha);
  glVertex2f(cx - arm, cy);
  glVertex2f(cx + arm, cy);
  glEnd();

  glLineWidth(activeWidth);
  glBegin(GL_LINES);
  glColor4f(0.92f, 0.96f, 1.0f, activeAlpha);
  if (axisLock == 1) {
    glVertex2f(cx - arm, cy);
    glVertex2f(cx + arm, cy);
  } else {
    glVertex2f(cx, cy - arm);
    glVertex2f(cx, cy + arm);
  }
  glEnd();

  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawModifierSymbolIndicator(int width, int height, char symbol, float xOffset, float yOffset, float alpha) {
  if (width <= 0 || height <= 0 || alpha <= 0.0f) return;
  const float cx = static_cast<float>(width) - xOffset;
  const float cy = yOffset;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (symbol == 'P') {
    const int segments = 28;
    const float radius = 7.2f;
    glLineWidth(1.4f);
    glColor4f(0.96f, 0.98f, 1.0f, 0.82f * alpha);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
      const float t = static_cast<float>(i) / static_cast<float>(segments);
      const float a = t * 6.28318530717958647692f;
      glVertex2f(cx + std::cos(a) * radius, cy + std::sin(a) * radius);
    }
    glEnd();
    glPointSize(4.0f);
    glBegin(GL_POINTS);
    glColor4f(0.96f, 0.98f, 1.0f, 0.95f * alpha);
    glVertex2f(cx, cy);
    glEnd();
  } else if (symbol == 'S') {
    const float halfWidth = 8.5f;
    const float amp = 1.7f;
    const int steps = 18;
    glLineWidth(1.6f);
    for (int row = 0; row < 2; ++row) {
      const float baseline = cy + (row == 0 ? 2.6f : -2.6f);
      glBegin(GL_LINE_STRIP);
      glColor4f(0.96f, 0.98f, 1.0f, (row == 0 ? 0.88f : 0.72f) * alpha);
      for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        const float x = -halfWidth + t * (halfWidth * 2.0f);
        const float y = baseline + std::sin(t * 6.28318530717958647692f) * amp;
        glVertex2f(cx + x, y);
      }
      glEnd();
    }
  }

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawSlowModifierIndicator(int width,
                               int height,
                               float xOffset,
                               float yOffset,
                               float alpha,
                               const HudTextRenderer* textRenderer = nullptr) {
  if (width <= 0 || height <= 0 || alpha <= 0.0f) return;
  const float cx = static_cast<float>(width) - xOffset;
  const float cy = yOffset;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (textRenderer && textRenderer->available) {
    const std::string symbol = u8"\u2026";
    const float scale = 0.95f;
    const float textWidth = WorkshopText::measureTextWidth(textRenderer->atlas, symbol, scale);
    if (textWidth > 0.5f) {
      const float ascent = static_cast<float>(std::max(1, textRenderer->atlas.ascent));
      const float descent = static_cast<float>(std::max(0, textRenderer->atlas.descent));
      const float baselineY = std::round(cy - ((ascent - descent) * scale * 0.5f));
      std::vector<WorkshopText::TextQuadVertex> quads;
      WorkshopText::appendTextQuads(textRenderer->atlas,
                                    symbol,
                                    std::round(cx - textWidth * 0.5f),
                                    baselineY,
                                    scale,
                                    &quads);
      if (!quads.empty()) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textRenderer->texture);
        glBegin(GL_TRIANGLES);
        glColor4f(0.96f, 0.98f, 1.0f, alpha);
        for (const auto& v : quads) {
          glTexCoord2f(v.u, v.v);
          glVertex2f(v.x, v.y);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        return;
      }
    }
  }

  glPointSize(3.0f);
  glBegin(GL_POINTS);
  glColor4f(0.96f, 0.98f, 1.0f, 0.92f * alpha);
  glVertex2f(cx - 4.6f, cy);
  glVertex2f(cx, cy);
  glVertex2f(cx + 4.6f, cy);
  glEnd();

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawNeutralRadiusIndicator(int width,
                                int height,
                                float xOffset,
                                float yOffset,
                                float alpha,
                                const HudTextRenderer* textRenderer = nullptr) {
  if (width <= 0 || height <= 0 || alpha <= 0.0f) return;
  const float cx = static_cast<float>(width) - xOffset;
  const float cy = static_cast<float>(height) - yOffset;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (textRenderer && textRenderer->available) {
    const std::string symbol = u8"\u22A2\u25CB\u22A3";
    const float scale = 0.92f;
    const float textWidth = WorkshopText::measureTextWidth(textRenderer->atlas, symbol, scale);
    if (textWidth > 0.5f) {
      const float ascent = static_cast<float>(std::max(1, textRenderer->atlas.ascent));
      const float descent = static_cast<float>(std::max(0, textRenderer->atlas.descent));
      const float baselineY = std::round(cy - ((ascent - descent) * scale * 0.5f));
      std::vector<WorkshopText::TextQuadVertex> quads;
      WorkshopText::appendTextQuads(textRenderer->atlas,
                                    symbol,
                                    std::round(cx - textWidth * 0.5f),
                                    baselineY,
                                    scale,
                                    &quads);
      if (!quads.empty()) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textRenderer->texture);
        glBegin(GL_TRIANGLES);
        glColor4f(0.96f, 0.98f, 1.0f, alpha);
        for (const auto& v : quads) {
          glTexCoord2f(v.u, v.v);
          glVertex2f(v.x, v.y);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        return;
      }
    }
  }

  glLineWidth(1.4f);
  glColor4f(0.96f, 0.98f, 1.0f, 0.90f * alpha);
  glBegin(GL_LINES);
  glVertex2f(cx - 12.0f, cy + 6.0f);
  glVertex2f(cx - 12.0f, cy - 6.0f);
  glVertex2f(cx - 12.0f, cy);
  glVertex2f(cx - 7.2f, cy);
  glVertex2f(cx + 12.0f, cy + 6.0f);
  glVertex2f(cx + 12.0f, cy - 6.0f);
  glVertex2f(cx + 12.0f, cy);
  glVertex2f(cx + 7.2f, cy);
  glEnd();

  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 28; ++i) {
    const float t = (static_cast<float>(i) / 28.0f) * 6.28318530718f;
    glVertex2f(cx + std::cos(t) * 4.8f, cy + std::sin(t) * 4.8f);
  }
  glEnd();

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void drawRollDirectionIndicator(int width,
                                int height,
                                int direction,
                                float pulse,
                                float yOffset = 30.0f,
                                const HudTextRenderer* symbolText = nullptr,
                                const HudTextRenderer* hudText = nullptr) {
  if (width <= 0 || height <= 0 || direction == 0) return;
  const float cx = static_cast<float>(width) - 34.0f;
  const float cy = yOffset;
  const float radius = 7.6f + 1.0f * pulse;
  const float alpha = 0.60f + 0.34f * pulse;
  const int steps = 34;
  // Match the feedback to the intended Unicode symbols:
  // clockwise   ↻ (U+21BB)
  // counterclockwise ↺ (U+21BA)
  const float tailAngle = 1.57079632679f;
  const float startAngle = direction > 0 ? tailAngle + 0.78f : tailAngle - 0.78f;
  const float endAngle = direction > 0 ? tailAngle - 5.05f : tailAngle + 5.05f;
  const float headAngle = endAngle;
  const float headLen = 4.4f + 0.6f * pulse;
  const float tailLen = 3.8f + 0.4f * pulse;

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(0.0, static_cast<double>(width), 0.0, static_cast<double>(height), -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  const HudTextRenderer* textRenderer =
      (symbolText && symbolText->available) ? symbolText : ((hudText && hudText->available) ? hudText : nullptr);
  if (textRenderer && textRenderer->available) {
    const std::string symbol = direction > 0 ? u8"\u21BB" : u8"\u21BA";
    const float scale = 1.0f;
    const float textWidth = WorkshopText::measureTextWidth(textRenderer->atlas, symbol, scale);
    if (textWidth > 0.5f) {
      const float baselineY = std::round(cy - 8.0f);
      std::vector<WorkshopText::TextQuadVertex> quads;
      WorkshopText::appendTextQuads(textRenderer->atlas, symbol, std::round(cx - textWidth * 0.5f), baselineY, scale, &quads);
      if (!quads.empty()) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textRenderer->texture);
        glBegin(GL_TRIANGLES);
        glColor4f(0.96f, 0.98f, 1.0f, alpha);
        for (const auto& v : quads) {
          glTexCoord2f(v.u, v.v);
          glVertex2f(v.x, v.y);
        }
        glEnd();
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
        glEnable(GL_DEPTH_TEST);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        return;
      }
    }
  }

  glLineWidth(1.7f + 0.4f * pulse);
  glColor4f(0.96f, 0.98f, 1.0f, alpha);
  glBegin(GL_LINE_STRIP);
  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(steps);
    const float a = startAngle + (endAngle - startAngle) * t;
    glVertex2f(cx + std::cos(a) * radius, cy + std::sin(a) * radius);
  }
  glEnd();

  // Short vertical tail so the overall silhouette reads more like ↻ / ↺ than a plain circular arrow.
  glBegin(GL_LINES);
  glVertex2f(cx, cy + radius + tailLen);
  glVertex2f(cx, cy + radius - 1.6f);
  glEnd();

  const float hx = cx + std::cos(headAngle) * radius;
  const float hy = cy + std::sin(headAngle) * radius;
  const float tangentX = direction > 0 ? -std::sin(headAngle) : std::sin(headAngle);
  const float tangentY = direction > 0 ?  std::cos(headAngle) : -std::cos(headAngle);
  glBegin(GL_LINES);
  glVertex2f(hx, hy);
  glVertex2f(hx - tangentX * headLen - std::cos(headAngle) * 2.2f,
             hy - tangentY * headLen - std::sin(headAngle) * 2.2f);
  glVertex2f(hx, hy);
  glVertex2f(hx + tangentX * headLen - std::cos(headAngle) * 2.2f,
             hy + tangentY * headLen - std::sin(headAngle) * 2.2f);
  glEnd();

  glEnable(GL_DEPTH_TEST);
  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  refreshModifierState(window, app);
  applyMouseModifierBits(app, mods);
  if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
  const double now = glfwGetTime();
  if ((now - app->lastClick) < 0.3) {
    const bool ctrl = app->ctrlHeld ||
                      glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                      nativeControlModifierPressed();
    const bool shift = app->shiftHeld ||
                       glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS ||
                       nativeShiftModifierPressed();
    if (shift) {
      const int nextOrthoView = app->cam.orthographic ? (app->cam.orthographicView + 1) : 0;
      setOrthographicInspectionCamera(&app->cam, nextOrthoView);
    } else {
      app->modelOrientation = Quat{};
      if (ctrl) {
      if (app->plotMode == "hsl") {
        resetHslTopCamera(&app->cam);
      } else if (app->plotMode == "rgb") {
        resetVectorscopeCamera(&app->cam);
      } else if (app->plotMode == "chromaticity") {
        resetChromaticityVectorscopeCamera(&app->cam);
      } else if (app->plotMode == "chen") {
        resetChenVectorscopeCamera(&app->cam);
      } else if (app->plotMode == "jp_conical" || app->plotMode == "reuleaux") {
        resetTightPolarVectorscopeCamera(&app->cam);
      } else {
        resetPolarVectorscopeCamera(&app->cam);
      }
    } else {
      if (app->plotMode == "hsl") {
        resetHslCamera(&app->cam);
      } else if (app->plotMode == "chromaticity") {
        resetChromaticityCamera(&app->cam);
      } else if (app->plotMode == "chen") {
        resetChenCamera(&app->cam);
      } else {
        resetCamera(&app->cam);
      }
    }
    }
    app->leftDown = false;
    app->panMode = false;
    app->shiftPanGesture = false;
    app->rollMode = false;
    app->zoomMode = false;
    app->panVelocityX = 0.0f;
    app->panVelocityY = 0.0f;
    app->orientAxisLock = 0;
    app->orientAxisFeedbackUntil = 0.0;
    app->rollFeedbackUntil = 0.0;
  }
  app->lastClick = now;
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  tryActivateViewerOnHover(window, app);
  refreshModifierState(window, app);
  app->hoverX = xpos;
  app->hoverY = ypos;

  const int l = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  const int m = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
  const int r = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
  const bool anyDown = (l == GLFW_PRESS || m == GLFW_PRESS || r == GLFW_PRESS);
  const bool shift = app->shiftHeld ||
                     glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS ||
                     nativeShiftModifierPressed();
  const bool alt = app->altHeld ||
                   glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                   glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS ||
                   nativeAltModifierPressed();
  const bool rollModifier = platformRollModifierPressed(*app);

  if (anyDown && !app->leftDown) {
    app->leftDown = true;
    app->zoomMode = (r == GLFW_PRESS);
    const bool shiftPanFromLeft = !app->zoomMode && !rollModifier && (l == GLFW_PRESS) && shift && (m != GLFW_PRESS);
    app->rollMode = !app->zoomMode && !shiftPanFromLeft &&
                    (l == GLFW_PRESS) && (m != GLFW_PRESS) && rollModifier;
    app->panMode = !app->zoomMode && !app->rollMode && ((m == GLFW_PRESS) || shiftPanFromLeft);
    app->shiftPanGesture = app->panMode && shift;
    app->lastX = xpos;
    app->lastY = ypos;
    app->orbitVirtualX = xpos;
    app->orbitVirtualY = ypos;
    app->panVelocityX = 0.0f;
    app->panVelocityY = 0.0f;
    app->orientAxisLock = 0;
    app->orientAxisFeedbackUntil = 0.0;
  } else if (!anyDown && app->leftDown) {
    app->leftDown = false;
    app->panMode = false;
    app->shiftPanGesture = false;
    app->rollMode = false;
    app->zoomMode = false;
    app->panVelocityX = 0.0f;
    app->panVelocityY = 0.0f;
    app->orientAxisLock = 0;
    app->orientAxisFeedbackUntil = 0.0;
    app->rollFeedbackUntil = 0.0;
  }

  if (!app->leftDown) return;

  const float dx = static_cast<float>(xpos - app->lastX);
  const float dy = static_cast<float>(ypos - app->lastY);
  app->lastX = xpos;
  app->lastY = ypos;

  int width = 1, height = 1;
  glfwGetFramebufferSize(window, &width, &height);

  if (app->zoomMode) {
    const bool ctrl = app->ctrlHeld ||
                      glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                      nativeControlModifierPressed();
    const float zoomScale = ctrl ? 0.02f : (shift ? (0.01f * shiftPrecisionFactor()) : 0.01f);
    const float verticalZoomDrag = dy;
    const float horizontalZoomDrag = -dx;
    const float zoomDrag =
        (verticalZoomDrag * horizontalZoomDrag > 0.0f)
            ? (verticalZoomDrag + horizontalZoomDrag)
            : (std::fabs(verticalZoomDrag) >= std::fabs(horizontalZoomDrag)
                   ? verticalZoomDrag
                   : horizontalZoomDrag);
    if (ctrl) app->speedFeedbackUntil = glfwGetTime() + 0.18;
    if (!ctrl && shift) app->slowFeedbackUntil = glfwGetTime() + 0.18;
    app->cam.distance = clampf(app->cam.distance * std::exp(zoomDrag * zoomScale),
                               minCameraDistanceForView(app->cam),
                               kMaxCameraDistance);
    return;
  }
  if (app->rollMode) {
    const bool ctrl = app->ctrlHeld ||
                      glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                      nativeControlModifierPressed();
    const float drag = std::fabs(dx) >= std::fabs(dy) ? dx : -dy;
    if (std::fabs(drag) > 0.05f) {
      app->rollDirection = drag > 0.0f ? 1 : -1;
      app->rollFeedbackUntil = glfwGetTime() + 0.18;
    }
    const float rollScale = ctrl ? 2.0f : (shift ? shiftPrecisionFactor() : 1.0f);
    if (ctrl) app->speedFeedbackUntil = glfwGetTime() + 0.18;
    if (!ctrl && shift) app->slowFeedbackUntil = glfwGetTime() + 0.18;
    Quat next{app->cam.qx, app->cam.qy, app->cam.qz, app->cam.qw};
    // Prepend a roll around the view axis so it behaves like rotating the screen frame itself,
    // not like spinning the object in its own local coordinates.
    const Quat qRoll = axisAngleQ(Vec3{0.0f, 0.0f, 1.0f}, -drag * 0.0065f * rollScale);
    next = normalizeQ(mulQ(qRoll, next));
    app->cam.qx = next.x;
    app->cam.qy = next.y;
    app->cam.qz = next.z;
    app->cam.qw = next.w;
    return;
  }
  if (app->panMode) {
    const bool ctrl = app->ctrlHeld ||
                      glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                      nativeControlModifierPressed();
    const bool precisionPan = shift && !app->shiftPanGesture;
    const float zoomBlend = computeTightZoomBlend(app->cam.distance);
    const float distanceScale = clampf(app->cam.distance / 6.0f, 0.11f, 1.0f);
    const float panBoost = ctrl ? 2.15f : (precisionPan ? shiftPrecisionFactor() : 1.0f);
    const float panScale = (2.0f / static_cast<float>(std::max(1, height))) * distanceScale * panBoost;
    const float smoothing = 0.18f + zoomBlend * 0.62f;
    if (ctrl) app->speedFeedbackUntil = glfwGetTime() + 0.18;
    if (!ctrl && precisionPan) app->slowFeedbackUntil = glfwGetTime() + 0.18;
    app->panVelocityX = app->panVelocityX * smoothing + dx * (1.0f - smoothing);
    app->panVelocityY = app->panVelocityY * smoothing + dy * (1.0f - smoothing);
    app->cam.panX += app->panVelocityX * panScale;
    app->cam.panY -= app->panVelocityY * panScale;
    return;
  }

  const double prevX = app->orbitVirtualX;
  const double prevY = app->orbitVirtualY;
  app->orbitVirtualX += static_cast<double>(dx);
  app->orbitVirtualY += static_cast<double>(dy);
  const double currentX = app->orbitVirtualX;
  const double currentY = app->orbitVirtualY;
  const Vec3 va = mapArcball(prevX, prevY, width, height);
  const Vec3 vb = mapArcball(currentX, currentY, width, height);
  const Vec3 axis = cross3(va, vb);
  const float dot = clampf(dot3(va, vb), -1.0f, 1.0f);
  const bool fastOrbit = (l == GLFW_PRESS) &&
                         (app->ctrlHeld ||
                          glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                          glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                          nativeControlModifierPressed());
  const bool slowOrbit = (l == GLFW_PRESS) && !app->panMode && !fastOrbit && shift;
  if (fastOrbit) app->speedFeedbackUntil = glfwGetTime() + 0.18;
  if (slowOrbit) app->slowFeedbackUntil = glfwGetTime() + 0.18;
  const float angleScale = fastOrbit ? 2.0f : (slowOrbit ? shiftPrecisionFactor() : 1.0f);
  const bool lockView = alt && !app->panMode && !app->zoomMode;
  const Quat cur = Quat{app->cam.qx, app->cam.qy, app->cam.qz, app->cam.qw};
  auto commitOrientation = [&](const Quat& q) {
    app->cam.qx = q.x;
    app->cam.qy = q.y;
    app->cam.qz = q.z;
    app->cam.qw = q.w;
  };
  if (lockView) {
    Quat next = cur;
    float yawAngle = (dx * 3.14159265358979323846f / static_cast<float>(std::max(1, width))) * angleScale;
    float pitchAngle = (dy * 3.14159265358979323846f / static_cast<float>(std::max(1, height))) * angleScale;
    const float absDx = std::fabs(dx);
    const float absDy = std::fabs(dy);
    int nextAxisLock = app->orientAxisLock;
    if (nextAxisLock == 0) {
      nextAxisLock = (absDx >= absDy) ? 1 : 2;
    } else {
      const float switchMinTravel = 2.2f;
      const float switchRatio = 1.65f;
      const float switchSlack = 1.35f;
      if ((absDx + absDy) >= switchMinTravel) {
        if (nextAxisLock == 1) {
          if (absDy > absDx * switchRatio + switchSlack) nextAxisLock = 2;
        } else {
          if (absDx > absDy * switchRatio + switchSlack) nextAxisLock = 1;
        }
      }
    }
    if (app->orientAxisLock != nextAxisLock) {
      app->orientAxisLock = nextAxisLock;
      app->orientAxisFeedbackUntil = glfwGetTime() + 0.18;
    }
    if (nextAxisLock == 1) {
      pitchAngle = 0.0f;
    } else {
      yawAngle = 0.0f;
    }
    if (std::fabs(yawAngle) > 1e-6f) {
      const Quat qViewYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, yawAngle);
      next = normalizeQ(mulQ(qViewYaw, next));
    }
    if (std::fabs(pitchAngle) > 1e-6f) {
      const Quat qViewPitch = axisAngleQ(Vec3{1.0f, 0.0f, 0.0f}, pitchAngle);
      next = normalizeQ(mulQ(qViewPitch, next));
    }
    commitOrientation(next);
    return;
  }
  const float currentRadius = arcballRadius(currentX, currentY, width, height);
  const float prevRadius = arcballRadius(prevX, prevY, width, height);
  const float outsideRadius = std::max(currentRadius, prevRadius);
  const float absDx = std::fabs(dx);
  const float absDy = std::fabs(dy);
  const float dragSum = absDx + absDy;
  // Once the drag moves beyond the main trackball, pure arcball deltas collapse toward zero.
  // Blend in incremental yaw/pitch there so long drags continue to accumulate rotation instead
  // of visually "hitting a wall" while still preserving the original arcball feel near center.
  const float outsideBlend = clampf((outsideRadius - 0.92f) / 0.55f, 0.0f, 1.0f);
  // Hybrid assist: preserve the tactile trackball, but bias straight drags toward cleaner
  // screen-space yaw/pitch so edge grabs don't unexpectedly invert the perceived direction.
  const float straightness = dragSum > 1e-5f ? (std::fabs(absDx - absDy) / dragSum) : 0.0f;
  const float edgeAssist = clampf((outsideRadius - 0.38f) / 0.52f, 0.0f, 1.0f);
  const float hybridAssistBlend = clampf((0.08f + 0.28f * edgeAssist) * straightness, 0.0f, 0.38f);
  const float viewAssistBlend = std::max(outsideBlend, hybridAssistBlend);
  const float arcballWeight = 1.0f - clampf(outsideBlend + hybridAssistBlend * 0.55f, 0.0f, 0.92f);
  const float angle = std::acos(dot) * angleScale * arcballWeight;
  Quat next = cur;
  bool changed = false;
  if (length3(axis) > 1e-6f && std::isfinite(angle) && angle > 1e-6f) {
    const Quat delta = axisAngleQ(axis, angle);
    next = normalizeQ(mulQ(delta, next));
    changed = true;
  }
  if (viewAssistBlend > 0.0f) {
    const float yawAngle =
        (dx * 3.14159265358979323846f / static_cast<float>(std::max(1, width))) * angleScale * viewAssistBlend;
    const float pitchAngle =
        (dy * 3.14159265358979323846f / static_cast<float>(std::max(1, height))) * angleScale * viewAssistBlend;
    if (std::fabs(yawAngle) > 1e-6f) {
      const Quat qYaw = axisAngleQ(Vec3{0.0f, 1.0f, 0.0f}, yawAngle);
      next = normalizeQ(mulQ(qYaw, next));
      changed = true;
    }
    if (std::fabs(pitchAngle) > 1e-6f) {
      const Vec3 localRight = normalize3(rotateVecByQuat(next, Vec3{1.0f, 0.0f, 0.0f}));
      const Quat qPitch = axisAngleQ(localRight, pitchAngle);
      next = normalizeQ(mulQ(qPitch, next));
      changed = true;
    }
  }
  if (changed) {
    commitOrientation(next);
  }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  refreshModifierState(window, app);
  const bool shift = app->shiftHeld ||
                     glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                     glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS ||
                     nativeShiftModifierPressed();
  const bool ctrl = app->ctrlHeld ||
                    glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS ||
                    nativeControlModifierPressed();
  double scrollDelta = yoffset;
#if defined(__APPLE__)
  if (std::fabs(scrollDelta) < 1e-5 && std::fabs(xoffset) > 1e-5) {
    scrollDelta = xoffset;
  }
#endif
  const float scale = ctrl ? 0.20f : (shift ? (0.12f * shiftPrecisionFactor()) : 0.12f);
  if (ctrl) app->speedFeedbackUntil = glfwGetTime() + 0.18;
  if (!ctrl && shift) app->slowFeedbackUntil = glfwGetTime() + 0.18;
  app->cam.distance = clampf(app->cam.distance * std::exp(static_cast<float>(-scrollDelta) * scale),
                             minCameraDistanceForView(app->cam),
                             kMaxCameraDistance);
}

void windowCloseCallback(GLFWwindow*) {
  gRun.store(false);
}

void iconifyCallback(GLFWwindow*, int iconified) {
  gWindowIconified.store(iconified ? 1 : 0);
}

void focusCallback(GLFWwindow* window, int focused) {
  gWindowFocused.store(focused ? 1 : 0);
  AppState* app = reinterpret_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!app) return;
  if (focused) {
    app->lastHoverActivationAttempt = -10.0;
    refreshModifierState(window, app);
    return;
  }
  app->shiftHeld = false;
  app->ctrlHeld = false;
  app->altHeld = false;
  app->superHeld = false;
  app->rollKeyHeld = false;
}

void refreshCallback(GLFWwindow*, int width, int height) {
  gWindowVisible.store((width > 0 && height > 0) ? 1 : 0);
}

#if defined(_WIN32)
DWORD WINAPI ipcThreadMain(LPVOID);
#else
void ipcThreadMain();
#endif

void wakeIpcServer() {
#if defined(_WIN32)
  const std::string pipe = pipeName();
  HANDLE h = CreateFileA(pipe.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    const char* msg = "{\"type\":\"shutdown\"}\n";
    DWORD written = 0;
    WriteFile(h, msg, static_cast<DWORD>(std::strlen(msg)), &written, nullptr);
    CloseHandle(h);
  }
#else
  const std::string sock = pipeName();
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    const char* msg = "{\"type\":\"shutdown\"}\n";
    sendAllSocket(fd, msg, std::strlen(msg));
  }
  ::close(fd);
#endif
}

std::string handleIncomingLine(const std::string& line) {
  if (line.empty()) return std::string();
  if (line.find("\"type\":\"heartbeat\"") != std::string::npos) {
    gConnected.store(true);
    return heartbeatAckJson();
  }
  if (line.find("\"type\":\"bring_to_front\"") != std::string::npos) {
    gBringToFront.store(true);
    gConnected.store(true);
    logViewerEvent("Received bring_to_front.");
    return std::string();
  }
  if (line.find("\"type\":\"disconnect\"") != std::string::npos) {
    logViewerEvent("Received disconnect.");
    gConnected.store(false);
    return std::string();
  }
  if (line.find("\"type\":\"shutdown\"") != std::string::npos) {
    return std::string();
  }
  if (line.find("\"type\":\"params\"") != std::string::npos) {
    ResolvedPayload payload{};
    if (parseParamsMessage(line, &payload)) {
      std::lock_guard<std::mutex> lock(gMsgMutex);
      gPendingParamsMsg.line = line;
      gPendingParamsMsg.seq = payload.seq;
      gHasPendingParamsMsg = true;
    }
    return std::string();
  }
  if (line.find("\"type\":\"input_cloud\"") != std::string::npos) {
    InputCloudPayload payload{};
    if (parseInputCloudMessage(line, &payload)) {
      std::lock_guard<std::mutex> lock(gMsgMutex);
      gPendingCloudMsg.line = line;
      gPendingCloudMsg.seq = payload.seq;
      gHasPendingCloudMsg = true;
    }
  }
  return std::string();
}

#if !defined(_WIN32)
void handleSocketClient(int client) {
  gConnected.store(true);
  std::string buffer;
  char chunk[4096];
  for (;;) {
    const ssize_t read = ::recv(client, chunk, sizeof(chunk), 0);
    if (read <= 0) break;
    buffer.append(chunk, chunk + read);
    size_t newline = std::string::npos;
    while ((newline = buffer.find('\n')) != std::string::npos) {
      std::string line = buffer.substr(0, newline);
      buffer.erase(0, newline + 1);
      const std::string reply = handleIncomingLine(line);
      if (!reply.empty()) {
        sendAllSocket(client, reply.c_str(), reply.size());
      }
    }
  }
  if (!buffer.empty()) {
    logViewerEvent(std::string("Connection closed with unterminated payload bytes=") + std::to_string(buffer.size()));
  }
  ::close(client);
}
#endif

#if defined(_WIN32)
DWORD WINAPI ipcThreadMain(LPVOID) {
  const std::string pipe = pipeName();
  while (gRun.load()) {
    HANDLE pipeHandle = CreateNamedPipeA(pipe.c_str(),
                                         PIPE_ACCESS_DUPLEX,
                                         PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                                         4,
                                         16 << 20,
                                         16 << 20,
                                         0,
                                         nullptr);
    if (pipeHandle == INVALID_HANDLE_VALUE) return 0;
    BOOL connected = ConnectNamedPipe(pipeHandle, nullptr) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);
    if (connected) {
      gConnected.store(true);
      std::string buffer;
      char chunk[4096];
      while (gRun.load()) {
        DWORD read = 0;
        if (!ReadFile(pipeHandle, chunk, sizeof(chunk), &read, nullptr) || read == 0) break;
        buffer.append(chunk, chunk + read);
        size_t newline = std::string::npos;
        while ((newline = buffer.find('\n')) != std::string::npos) {
          std::string line = buffer.substr(0, newline);
          buffer.erase(0, newline + 1);
          const std::string reply = handleIncomingLine(line);
          if (!reply.empty()) {
            DWORD written = 0;
            WriteFile(pipeHandle, reply.c_str(), static_cast<DWORD>(reply.size()), &written, nullptr);
          }
        }
      }
    }
    DisconnectNamedPipe(pipeHandle);
    CloseHandle(pipeHandle);
  }
  return 0;
}
#else
void ipcThreadMain() {
  const std::string sock = pipeName();
  ::unlink(sock.c_str());
  const int server = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (server < 0) return;
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sock.c_str());
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(server);
    return;
  }
  ::listen(server, 4);
  while (gRun.load()) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(server, &rfds);
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 250000;
    const int ready = ::select(server + 1, &rfds, nullptr, nullptr, &tv);
    if (ready <= 0) continue;
    const int client = ::accept(server, nullptr, nullptr);
    if (client < 0) continue;
    std::thread([client]() { handleSocketClient(client); }).detach();
  }
  ::close(server);
  ::unlink(sock.c_str());
}
#endif

}  // namespace

int main() {
  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  logViewerEvent(std::string("Viewer startup ok ") + kViewerVersionString);

#if defined(_WIN32)
  HANDLE singletonMutex = acquireViewerSingletonMutex();
  if (singletonMutex == nullptr) {
    notifyExistingViewerBringToFront();
    return 0;
  }
#endif

  if (!glfwInit()) {
#if defined(_WIN32)
    CloseHandle(singletonMutex);
#endif
    return 1;
  }

#if defined(__APPLE__)
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#endif

  GLFWwindow* window = glfwCreateWindow(720, 600, "Chromaspace", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
#if defined(_WIN32)
    CloseHandle(singletonMutex);
#endif
    return 1;
  }

#if defined(_WIN32)
  applyWindowsWindowIcon(window);
#endif
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  ViewerGpuCapabilities gpuCaps = detectViewerGpuCapabilities();
  {
    const bool parityChecks = viewerParityChecksEnabled();
    std::ostringstream os;
    os << "Viewer GPU capabilities: glVersion=" << gpuCaps.glVersion
       << " buffers=" << (gpuCaps.glBufferObjects ? "1" : "0")
       << " compute=" << (gpuCaps.glComputeShaders ? "1" : "0")
       << " overlayCompute=" << (gpuCaps.overlayComputeEnabled ? "1" : "0")
       << " inputCompute=" << (gpuCaps.inputComputeEnabled ? "1" : "0")
       << " parityChecks=" << (parityChecks ? "1" : "0")
       << " active=" << gpuCaps.activeBackendLabel
       << " roadmap=" << gpuCaps.roadmapLabel;
#if defined(__APPLE__)
    os << " metalViewer=" << (gpuCaps.metalViewerAvailable ? "1" : "0")
       << " metalQueue=" << (gpuCaps.metalQueueReady ? "1" : "0");
    if (!gpuCaps.metalDeviceName.empty()) os << " metalDevice=" << gpuCaps.metalDeviceName;
#elif defined(CHROMASPACE_VIEWER_HAS_CUDA)
     os << " cudaViewer=" << (gpuCaps.cudaViewerAvailable ? "1" : "0")
       << " cudaInterop=" << (gpuCaps.cudaInteropReady ? "1" : "0")
       << " cudaStartupValidated=" << (gpuCaps.cudaStartupValidated ? "1" : "0")
       << " sessionBackend=" << backendKindLabel(gpuCaps.sessionBackend);
     if (!gpuCaps.cudaDeviceName.empty()) os << " cudaDevice=" << gpuCaps.cudaDeviceName;
     if (!gpuCaps.cudaReason.empty()) os << " cudaReason=" << gpuCaps.cudaReason;
     if (!gpuCaps.cudaStartupReason.empty()) os << " cudaStartupReason=" << gpuCaps.cudaStartupReason;
#endif
    logViewerEvent(os.str());
  }

  AppState app{};
  resetCamera(&app.cam);
  app.diagTransitions = viewerDiagnosticsEnabled();
  app.parityChecks = viewerParityChecksEnabled();
  app.gpuCaps = gpuCaps;
  HudTextRenderer hudText{};
  HudTextRenderer hudSymbolText{};
  (void)initializeHudTextRenderer(&hudText);
  (void)initializeHudSymbolRenderer(&hudSymbolText);
  if (app.diagTransitions) {
    logViewerDiagnostic(true, std::string("Transition diagnostics enabled. log=") + viewerLogPath());
    if (app.parityChecks) {
      logViewerDiagnostic(true, "CPU/GPU parity checks enabled.");
    }
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
    std::ostringstream os;
    os << " cudaViewer=" << (gpuCaps.cudaViewerAvailable ? "1" : "0")
       << " cudaInterop=" << (gpuCaps.cudaInteropReady ? "1" : "0");
    if (!gpuCaps.cudaDeviceName.empty()) os << " cudaDevice=" << gpuCaps.cudaDeviceName;
    if (!gpuCaps.cudaReason.empty()) os << " cudaReason=" << gpuCaps.cudaReason;
    logViewerDiagnostic(true, os.str());
#endif
  }
  glfwSetWindowUserPointer(window, &app);
  refreshModifierState(window, &app);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorEnterCallback(window, cursorEnterCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSetWindowCloseCallback(window, windowCloseCallback);
  glfwSetWindowIconifyCallback(window, iconifyCallback);
  glfwSetWindowFocusCallback(window, focusCallback);
  glfwSetFramebufferSizeCallback(window, refreshCallback);

#if defined(_WIN32)
  HANDLE ipcThread = CreateThread(nullptr, 0, ipcThreadMain, nullptr, 0, nullptr);
#else
  std::thread ipcThread(ipcThreadMain);
#endif

  MeshData mesh{};
  MeshData identityMesh{};
  MeshData overlayMesh{};
  PointBufferCache meshPointBufferCache{};
  PointBufferCache overlayPointBufferCache{};
  OverlayComputeCache overlayComputeCache{};
  InputCloudComputeCache inputCloudComputeCache{};
  InputCloudSampleComputeCache inputCloudSampleComputeCache{};
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
  OverlayCudaCache overlayCudaCache{};
  InputCloudCudaCache inputCloudCudaCache{};
  InputCloudSampleCudaCache inputCloudSampleCudaCache{};
#endif
  InputCloudSampleMetalCache inputCloudSampleMetalCache{};
  PointDrawBuffers sampledPointDraw{};
  ResolvedPayload resolved{};
  resolved.sourceMode = "input";
  resolved.plotMode = "rgb";
  resolved.quality = "Low";
  resolved.resolution = 25;
  mesh.quality = "Waiting";
  mesh.resolution = resolved.resolution;
  mesh.paramHash.clear();
  InputCloudPayload deferredCloud{};
  InputCloudPayload currentCloud{};
  bool hasDeferredCloud = false;
  bool hasCurrentCloud = false;
  uint64_t lastCloudSeq = 0;
  uint64_t lastParamsSeq = 0;

  while (gRun.load() && !glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (gBringToFront.exchange(false)) {
      glfwShowWindow(window);
      glfwFocusWindow(window);
    }

    PendingMessage pendingParams;
    PendingMessage pendingCloud;
    bool haveParams = false;
    bool haveCloud = false;
    {
      std::lock_guard<std::mutex> lock(gMsgMutex);
      if (gHasPendingParamsMsg) {
        pendingParams = gPendingParamsMsg;
        gHasPendingParamsMsg = false;
        haveParams = true;
      }
      if (gHasPendingCloudMsg) {
        pendingCloud = gPendingCloudMsg;
        gHasPendingCloudMsg = false;
        haveCloud = true;
      }
    }

    // Stage 1: apply the latest params first so any incoming cloud can be validated against the active mode/settings.
    if (haveParams) {
      ResolvedPayload next{};
      if (parseParamsMessage(pendingParams.line, &next)) {
        if (next.seq <= lastParamsSeq) {
          logViewerEvent("Ignored stale params sequence.");
        } else {
          lastParamsSeq = next.seq;
          const std::string prevSourceMode = resolved.sourceMode;
          const std::string prevPlotMode = resolved.plotMode;
          resolved = next;
          app.plotMode = resolved.plotMode;
          app.keepOnTop = resolved.alwaysOnTop;
          if (prevPlotMode != resolved.plotMode && resolved.plotMode == "chromaticity") {
            setOrthographicInspectionCamera(&app.cam, 0);
            app.cam.distance = 6.2f;
            app.cam.panX = 0.0f;
            app.cam.panY = -0.16f;
            app.modelOrientation = Quat{};
            app.panVelocityX = 0.0f;
            app.panVelocityY = 0.0f;
            app.orientAxisLock = 0;
            app.orientAxisFeedbackUntil = 0.0;
            app.rollMode = false;
            app.zoomMode = false;
            app.panMode = false;
            app.shiftPanGesture = false;
            logViewerEvent("Applied default orthographic chart view for chromaticity mode.");
          }
          if (app.diagTransitions && prevSourceMode != resolved.sourceMode) {
            std::ostringstream os;
            os << "Source mode transition: " << prevSourceMode << " -> " << resolved.sourceMode
               << " paramsSeq=" << resolved.seq
               << " meshSerial=" << mesh.serial
               << " meshPoints=" << mesh.pointCount
               << " deferredCloud=" << (hasDeferredCloud ? "1" : "0");
            logViewerDiagnostic(true, os.str());
          }
          {
            std::ostringstream os;
            os << "Params received: sender=" << resolved.senderId
               << " seq=" << resolved.seq
               << " mode=" << resolved.sourceMode
               << " plotMode=" << resolved.plotMode
               << " quality=" << resolved.quality
               << " res=" << resolved.resolution
               << " pointSize=" << resolved.pointSize
               << " density=" << resolved.pointDensity
               << " pointShape=" << resolved.pointShape
               << " cloudKey=" << resolved.cloudSettingsKey
               << " overlay=" << (resolved.identityOverlayEnabled ? 1 : 0)
               << " overlayReq=" << resolved.identityOverlayRequestedSize
               << " overlayRes=" << resolved.identityOverlaySize
               << " ramp=" << (resolved.identityOverlayRamp ? 1 : 0);
            logViewerEvent(os.str());
          }
          logComputeEligibilityTransitions(resolved, &app);
          if (resolved.identityOverlayEnabled) {
            MeshData nextOverlay{};
            if ((app.gpuCaps.overlayComputeEnabled && buildIdentityOverlayMeshOnGpu(resolved, app.gpuCaps, &app.computeSession, &overlayComputeCache, &nextOverlay
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                                                                     , &overlayCudaCache
#endif
                                                                                     )) ||
                buildIdentityOverlayMesh(resolved, &nextOverlay)) {
              overlayMesh = std::move(nextOverlay);
            }
          } else {
            overlayMesh = MeshData{};
            overlayComputeCache.builtSerial = 0;
            overlayComputeCache.pointCount = 0;
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
            overlayCudaCache.builtSerial = 0;
            overlayCudaCache.pointCount = 0;
#endif
          }
          if (resolved.sourceMode == "identity") {
            if (buildIdentityMesh(resolved, &identityMesh)) {
              mesh = identityMesh;
            }
          } else if (hasDeferredCloud && cloudMatchesResolved(resolved, deferredCloud)) {
            MeshData nextMesh{};
            if (buildInputCloudMesh(resolved, app.gpuCaps, &app.computeSession, &inputCloudComputeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                    &inputCloudCudaCache,
#endif
                                    deferredCloud, &nextMesh)) {
              mesh = std::move(nextMesh);
              currentCloud = deferredCloud;
              hasCurrentCloud = true;
              lastCloudSeq = deferredCloud.seq;
              hasDeferredCloud = false;
              logViewerEvent("Applied deferred input cloud after params switched to matching settings.");
            }
          } else if (hasCurrentCloud && cloudMatchesResolved(resolved, currentCloud)) {
            MeshData nextMesh{};
            if (buildInputCloudMesh(resolved, app.gpuCaps, &app.computeSession, &inputCloudComputeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                    &inputCloudCudaCache,
#endif
                                    currentCloud, &nextMesh)) {
              mesh = std::move(nextMesh);
              logViewerEvent("Rebuilt active input cloud after params update.");
            }
          } else {
            hasCurrentCloud = false;
            mesh = MeshData{};
            mesh.quality = "Waiting";
            mesh.resolution = resolved.resolution;
            mesh.paramHash.clear();
            logViewerEvent("Cleared stale input cloud after params changed settings.");
          }
        }
      }
    }

    // Stage 2: apply cloud payloads only when they match the active sender/settings contract.
    if (haveCloud) {
      InputCloudPayload cp{};
      if (parseInputCloudMessage(pendingCloud.line, &cp)) {
        {
          std::ostringstream os;
          os << "Input cloud received: sender=" << cp.senderId
             << " seq=" << cp.seq
             << " quality=" << cp.quality
             << " res=" << cp.resolution
             << " bytes=" << cp.points.size();
          logViewerEvent(os.str());
        }
        if (!senderMatchesCurrent(resolved.senderId, cp.senderId)) {
          logViewerEvent("Ignored input cloud from non-active sender.");
        } else if (cp.seq <= lastCloudSeq) {
          logViewerEvent("Ignored stale input cloud sequence.");
        } else if (resolved.sourceMode != "input") {
          deferredCloud = cp;
          hasDeferredCloud = true;
          logViewerEvent("Deferred input cloud until params switch viewer to input mode.");
        } else if (!cloudMatchesResolved(resolved, cp)) {
          deferredCloud = cp;
          hasDeferredCloud = true;
          logViewerEvent("Deferred input cloud until params match cloud settings.");
        } else {
          MeshData nextMesh{};
            if (buildInputCloudMesh(resolved, app.gpuCaps, &app.computeSession, &inputCloudComputeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                    &inputCloudCudaCache,
#endif
                                    cp, &nextMesh)) {
              mesh = std::move(nextMesh);
              currentCloud = cp;
            hasCurrentCloud = true;
            lastCloudSeq = cp.seq;
            hasDeferredCloud = false;
            std::ostringstream os;
            os << "Applied input cloud: seq=" << cp.seq
               << " points=" << (mesh.pointCount > 0 ? mesh.pointCount : (mesh.pointVerts.size() / 3u))
               << " quality=" << mesh.quality
               << " hash=" << mesh.paramHash;
            logViewerEvent(os.str());
          } else {
            logViewerEvent("Input cloud payload parsed but mesh build failed.");
          }
        }
      }
    }

    // Stage 3: render the current guide, overlay reference, and active point cloud.
    if (app.keepOnTop != app.appliedTopmost) {
      glfwSetWindowAttrib(window, GLFW_FLOATING, app.keepOnTop ? GLFW_TRUE : GLFW_FALSE);
      app.appliedTopmost = app.keepOnTop;
    }

    std::ostringstream title;
    title << "Chromaspace | " << (gConnected.load() ? "Connected" : "Waiting")
          << " | " << plotModeLabel(resolved)
          << " | " << mesh.quality
          << ' ' << mesh.resolution << "^3";
    title << " | viewer:" << app.gpuCaps.activeBackendLabel;
    if (resolved.identityOverlayEnabled) {
      title << " | Overlay " << resolved.identityOverlaySize << "^3";
      if (resolved.identityOverlayRamp) title << " + Ramp";
    }
    if (const char* orthoLabel = orthographicViewLabel(app.cam)) {
      title << " | " << orthoLabel;
    }
    if (!mesh.paramHash.empty()) title << " | hash " << mesh.paramHash;
    glfwSetWindowTitle(window, title.str().c_str());

    int width = 1, height = 1;
    glfwGetFramebufferSize(window, &width, &height);
    int windowWidth = 1, windowHeight = 1;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    const float aspect = height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    const float fovy = kViewerFovYDegrees;
    // Let the near plane relax as the user pushes into very close inspection
    // so points near the camera do not disappear just because the orbit distance
    // is much tighter than the original default framing.
    const float zNear = clampf(app.cam.distance * 0.025f, 0.0018f, 0.08f);
    const float zFar = 100.0f;
    const float ymax = zNear * tanHalfFovDegrees(fovy);
    const float xmax = ymax * aspect;
    if (app.cam.orthographic) {
      const float orthoHalfHeight = std::max(kMinOrthoHalfHeight, app.cam.distance * tanHalfFovDegrees(fovy));
      const float orthoHalfWidth = orthoHalfHeight * aspect;
      glOrtho(-orthoHalfWidth, orthoHalfWidth, -orthoHalfHeight, orthoHalfHeight, zNear, zFar);
    } else {
      glFrustum(-xmax, xmax, -ymax, ymax, zNear, zFar);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(app.cam.panX, app.cam.panY, -app.cam.distance);
    float rotM[16];
    quatToMatrix(Quat{app.cam.qx, app.cam.qy, app.cam.qz, app.cam.qw}, rotM);
    glMultMatrixf(rotM);
    float modelRotM[16];
    quatToMatrix(app.modelOrientation, modelRotM);
    glMultMatrixf(modelRotM);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    float backgroundR = resolved.backgroundColorR;
    float backgroundG = resolved.backgroundColorG;
    float backgroundB = resolved.backgroundColorB;
    if (resolved.volumeSlicingEnabled &&
        resolved.volumeSlicingMode == "lasso" &&
        resolved.lassoRegionEmpty) {
      backgroundR = clamp01(backgroundR * 0.82f + 0.05f);
      backgroundG = clamp01(backgroundG * 0.86f + 0.09f);
      backgroundB = clamp01(backgroundB * 1.10f + 0.20f);
    }
    if (resolved.neutralRadiusEnabled) {
      backgroundR = 0.118f;
      backgroundG = 0.110f;
      backgroundB = 0.102f;
    }
    glClearColor(backgroundR, backgroundG, backgroundB, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    const HudTextRenderer* overlayTextRenderer = preferredHudRenderer(&hudText, &hudSymbolText);
    drawGuideForPlotMode(resolved, app.cam, height, fovy, overlayTextRenderer);

    // Viewer-side draw tuning intentionally stays separate from OFX sampling quality:
    // density controls how many received points are shown, while point size/halo keep the cloud legible.
    const float densityForView = std::max(0.35f, resolved.pointDensity);
    const PointSelectionSpec pointSelection =
        makePointSelectionSpec(mesh.serial, mesh.pointCount, densityForView);
    const float* activePointVerts = mesh.pointVerts.empty() ? nullptr : mesh.pointVerts.data();
    const float* activePointColors = mesh.pointColors.empty() ? nullptr : mesh.pointColors.data();
    size_t activePointCount = mesh.pointCount;
    const bool sampledDrawReady =
        buildInputPointDrawBuffers(mesh,
                                   pointSelection,
                                   app.gpuCaps,
                                   &app.computeSession,
                                   inputCloudComputeCache,
                                   &inputCloudSampleComputeCache,
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
                                   &inputCloudCudaCache,
                                   &inputCloudSampleCudaCache,
#endif
                                   &inputCloudSampleMetalCache,
                                   app.diagTransitions,
                                   &sampledPointDraw);
    const bool useSampledCpuArrays = sampledDrawReady && sampledPointDraw.available &&
                                     sampledPointDraw.verts == 0 && sampledPointDraw.colors == 0 &&
                                     !sampledPointDraw.cpuVerts.empty() && !sampledPointDraw.cpuColors.empty();
    if (useSampledCpuArrays) {
      activePointVerts = sampledPointDraw.cpuVerts.data();
      activePointColors = sampledPointDraw.cpuColors.data();
      activePointCount = sampledPointDraw.pointCount;
    } else if (pointSelection.needsThinning && sampledDrawReady) {
      activePointCount = sampledPointDraw.pointCount;
    }
    float pointSize = 2.5f;
    if (mesh.resolution <= 25) pointSize = 3.3f;
    else if (mesh.resolution <= 41) pointSize = 2.9f;
    pointSize *= std::max(0.5f, resolved.pointSize);
    pointSize *= std::pow(densityForView, -0.12f);
    // Compensate for perspective zoom without over-growing points at close distances.
    const float zoomComp = clampf(std::pow(6.0f / std::max(0.2f, app.cam.distance), 0.52f), 0.90f, 2.35f);
    pointSize = clampf(pointSize * zoomComp, 1.0f, 24.0f);
    const bool useSquarePoints = resolved.pointShape == "Square";
    const float denseGlowSuppress =
        densePlotGlowSuppression(resolved.pointSize, densityForView, mesh.resolution);
    if (useSquarePoints) {
      glDisable(GL_POINT_SMOOTH);
    } else {
      glEnable(GL_POINT_SMOOTH);
      glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    const ViewerGlBufferApi& glBufferApi = viewerGlBufferApi();
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
    const bool useOverlayCudaBuffers =
        overlayMesh.pointCount > 0 && app.gpuCaps.cudaComputeEnabled &&
        overlayCudaCache.available &&
        overlayCudaCache.builtSerial == overlayMesh.serial &&
        overlayCudaCache.verts != 0 && overlayCudaCache.colors != 0 &&
        overlayCudaCache.pointCount > 0;
#else
    const bool useOverlayCudaBuffers = false;
#endif
    const bool useOverlayComputeBuffers =
        !useOverlayCudaBuffers &&
        overlayMesh.pointCount > 0 && overlayComputeCache.available &&
        overlayComputeCache.builtSerial == overlayMesh.serial &&
        overlayComputeCache.verts != 0 && overlayComputeCache.colors != 0 &&
        overlayComputeCache.pointCount > 0;
    const bool useOverlayPointBuffers =
        !useOverlayCudaBuffers &&
        !useOverlayComputeBuffers &&
        overlayMesh.pointCount > 0 && ensurePointBufferCacheUploaded(overlayMesh, &overlayPointBufferCache);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
    const bool useInputCudaBuffers =
        !pointSelection.needsThinning &&
        activePointCount == mesh.pointCount && app.gpuCaps.cudaComputeEnabled &&
        inputCloudCudaCache.available &&
        inputCloudCudaCache.builtSerial == mesh.serial &&
        inputCloudCudaCache.verts != 0 && inputCloudCudaCache.colors != 0 &&
        inputCloudCudaCache.pointCount > 0;
    const bool useInputSampledCudaBuffers =
        pointSelection.needsThinning && sampledDrawReady && sampledPointDraw.available &&
        app.gpuCaps.cudaComputeEnabled &&
        sampledPointDraw.verts != 0 && sampledPointDraw.colors != 0 &&
        sampledPointDraw.pointCount == activePointCount;
#else
    const bool useInputCudaBuffers = false;
    const bool useInputSampledCudaBuffers = false;
#endif
    const bool useInputComputeBuffers =
        !useInputCudaBuffers &&
        !useInputSampledCudaBuffers &&
        !pointSelection.needsThinning &&
        activePointCount == mesh.pointCount &&
        inputCloudComputeCache.available &&
        inputCloudComputeCache.builtSerial == mesh.serial &&
        inputCloudComputeCache.verts != 0 &&
        inputCloudComputeCache.colors != 0 &&
        inputCloudComputeCache.pointCount > 0;
    const bool useInputSampledComputeBuffers =
        !useInputSampledCudaBuffers &&
        pointSelection.needsThinning && sampledDrawReady && sampledPointDraw.available &&
        sampledPointDraw.verts != 0 && sampledPointDraw.colors != 0 &&
        sampledPointDraw.pointCount == activePointCount;
    const bool usePointBuffers =
        !useInputCudaBuffers &&
        !useInputComputeBuffers &&
        !useInputSampledCudaBuffers &&
        !useInputSampledComputeBuffers &&
        activePointCount == mesh.pointCount && activePointVerts == mesh.pointVerts.data() &&
        activePointColors == mesh.pointColors.data() && ensurePointBufferCacheUploaded(mesh, &meshPointBufferCache);
    const bool haveDrawablePointSource = useInputSampledCudaBuffers || useInputSampledComputeBuffers ||
                                         useInputCudaBuffers || useInputComputeBuffers || usePointBuffers ||
                                         (activePointVerts != nullptr && activePointColors != nullptr);
    const std::string drawSourceLabel = pointDrawSourceLabel(useInputCudaBuffers,
                                                             useInputComputeBuffers,
                                                             usePointBuffers,
                                                             useInputSampledCudaBuffers,
                                                             useInputSampledComputeBuffers,
                                                             useSampledCpuArrays,
                                                             activePointVerts != nullptr && activePointColors != nullptr);
    if (app.diagTransitions && drawSourceLabel != app.lastDrawSourceLabel) {
      std::ostringstream os;
      os << "Draw source changed: mode=" << resolved.sourceMode
         << " meshSerial=" << mesh.serial
         << " meshPoints=" << activePointCount
         << " source=" << drawSourceLabel;
      logViewerDiagnostic(true, os.str());
      app.lastDrawSourceLabel = drawSourceLabel;
    }
    glPointSize(pointSize);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
    if (useInputSampledCudaBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, sampledPointDraw.verts);
      glVertexPointer(3, GL_FLOAT, 0, nullptr);
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, sampledPointDraw.colors);
      glColorPointer(4, GL_FLOAT, 0, nullptr);
    } else if (useInputCudaBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, inputCloudCudaCache.verts);
      glVertexPointer(3, GL_FLOAT, 0, nullptr);
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, inputCloudCudaCache.colors);
      glColorPointer(4, GL_FLOAT, 0, nullptr);
    } else
#endif
    if (useInputSampledComputeBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, sampledPointDraw.verts);
      glVertexPointer(3, GL_FLOAT, 0, nullptr);
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, sampledPointDraw.colors);
      glColorPointer(4, GL_FLOAT, 0, nullptr);
    } else if (useInputComputeBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, inputCloudComputeCache.verts);
      glVertexPointer(3, GL_FLOAT, 0, nullptr);
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, inputCloudComputeCache.colors);
      glColorPointer(4, GL_FLOAT, 0, nullptr);
    } else if (usePointBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, meshPointBufferCache.verts);
      glVertexPointer(3, GL_FLOAT, 0, nullptr);
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, meshPointBufferCache.colors);
      glColorPointer(4, GL_FLOAT, 0, nullptr);
    } else {
      glVertexPointer(3, GL_FLOAT, 0, activePointVerts);
      glColorPointer(4, GL_FLOAT, 0, activePointColors);
    }
    if (activePointCount > 0 && haveDrawablePointSource) {
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(activePointCount));
    }
    if (activePointCount > 0 && haveDrawablePointSource) {
      glDisable(GL_DEPTH_TEST);
      glDisableClientState(GL_COLOR_ARRAY);
      // The white interior thickening/halo passes help sparse plots, but they quickly wash out dense large splats.
      // Fade them down aggressively in that regime so the colored structure stays crisp.
      const float thickeningAlpha =
          clampf((0.05f / std::sqrt(densityForView)) * (1.0f - 0.92f * denseGlowSuppress), 0.0f, 0.05f);
      if (thickeningAlpha > 0.006f) {
        glColor4f(0.95f, 0.96f, 1.0f, thickeningAlpha);
        glPointSize(pointSize * clampf((0.56f / std::sqrt(densityForView)) * (1.0f - 0.22f * denseGlowSuppress),
                                       0.24f, 0.58f));
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(activePointCount));
      }
      const float haloAlpha =
          clampf((0.06f / densityForView) * (1.0f - 0.97f * denseGlowSuppress), 0.0f, 0.05f);
      if (haloAlpha > 0.006f) {
        glColor4f(0.95f, 0.96f, 1.0f, haloAlpha);
        glPointSize(pointSize * clampf((0.52f / std::sqrt(densityForView)) * (1.0f - 0.26f * denseGlowSuppress),
                                       0.22f, 0.55f));
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(activePointCount));
      }
      glEnableClientState(GL_COLOR_ARRAY);
      glEnable(GL_DEPTH_TEST);
    }
    if (resolved.showOverflow && resolved.highlightOverflow &&
        activePointCount > 0 &&
        activePointVerts != nullptr && activePointColors != nullptr) {
      std::vector<float> overflowVerts;
      std::vector<float> overflowColors;
      overflowVerts.reserve(activePointCount * 3u / 16u);
      overflowColors.reserve(activePointCount * 4u / 16u);
      for (size_t i = 0; i < activePointCount; ++i) {
        const size_t colorBase = i * 4u;
        if (activePointColors[colorBase + 3u] < 0.9f) continue;
        const size_t vertBase = i * 3u;
        overflowVerts.push_back(activePointVerts[vertBase + 0u]);
        overflowVerts.push_back(activePointVerts[vertBase + 1u]);
        overflowVerts.push_back(activePointVerts[vertBase + 2u]);
        overflowColors.push_back(activePointColors[colorBase + 0u]);
        overflowColors.push_back(activePointColors[colorBase + 1u]);
        overflowColors.push_back(activePointColors[colorBase + 2u]);
        overflowColors.push_back(1.0f);
      }
      if (!overflowVerts.empty()) {
        if (useOverlayCudaBuffers || useOverlayComputeBuffers || useOverlayPointBuffers ||
            useInputSampledCudaBuffers || useInputSampledComputeBuffers ||
            useInputCudaBuffers || useInputComputeBuffers || usePointBuffers) {
          glBufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
        }
        glDisable(GL_DEPTH_TEST);
        glPointSize(clampf(pointSize * 1.28f, 1.0f, 28.0f));
        glVertexPointer(3, GL_FLOAT, 0, overflowVerts.data());
        glColorPointer(4, GL_FLOAT, 0, overflowColors.data());
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(overflowVerts.size() / 3u));
        glEnable(GL_DEPTH_TEST);
      }
    }
    if (overlayMesh.pointCount > 0) {
      // Draw the identity solid as a true overlay so it stays readable without replacing the image cloud.
      const float overlayPointSize = std::max(1.0f, pointSize * 0.72f);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(GL_FALSE);
      glPointSize(overlayPointSize);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
      if (useOverlayCudaBuffers) {
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayCudaCache.verts);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayCudaCache.colors);
        glColorPointer(4, GL_FLOAT, 0, nullptr);
      } else
#endif
      if (useOverlayComputeBuffers) {
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayComputeCache.verts);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayComputeCache.colors);
        glColorPointer(4, GL_FLOAT, 0, nullptr);
      } else if (useOverlayPointBuffers) {
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayPointBufferCache.verts);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBufferApi.bindBuffer(GL_ARRAY_BUFFER, overlayPointBufferCache.colors);
        glColorPointer(4, GL_FLOAT, 0, nullptr);
      } else {
        glVertexPointer(3, GL_FLOAT, 0, overlayMesh.pointVerts.empty() ? nullptr : overlayMesh.pointVerts.data());
        glColorPointer(4, GL_FLOAT, 0, overlayMesh.pointColors.empty() ? nullptr : overlayMesh.pointColors.data());
      }
      glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(overlayMesh.pointCount));
      glDepthMask(GL_TRUE);
      glEnable(GL_DEPTH_TEST);
    }
    if (useOverlayCudaBuffers || useOverlayComputeBuffers || useOverlayPointBuffers ||
        useInputSampledCudaBuffers || useInputSampledComputeBuffers ||
        useInputCudaBuffers || useInputComputeBuffers || usePointBuffers) {
      glBufferApi.bindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    if (resolved.plotMode == "chromaticity") {
      const double hoverScaleX = static_cast<double>(width) / static_cast<double>(std::max(1, windowWidth));
      const double hoverScaleY = static_cast<double>(height) / static_cast<double>(std::max(1, windowHeight));
      drawChromaticityInfoOverlay(makePlotRemapSpec(resolved),
                                  width,
                                  height,
                                  app.hoverX * hoverScaleX,
                                  app.hoverY * hoverScaleY,
                                  overlayTextRenderer ? *overlayTextRenderer : hudText);
    }

    int indicatorSlot = 0;
    auto indicatorYOffset = [&](int slot) {
      return 30.0f + static_cast<float>(slot) * 26.0f;
    };
    const float slowPulse = clampf(static_cast<float>((app.slowFeedbackUntil - glfwGetTime()) / 0.18), 0.0f, 1.0f);
    const bool combinedSlowModifierHeld =
        app.shiftHeld && !app.ctrlHeld && (app.altHeld || app.superHeld || app.rollKeyHeld);
    const float slowIndicatorStrength =
        std::max(slowPulse, combinedSlowModifierHeld ? 0.55f : 0.0f);
    if (slowIndicatorStrength > 0.0f) {
      drawSlowModifierIndicator(width,
                                height,
                                34.0f,
                                indicatorYOffset(indicatorSlot++),
                                0.50f + 0.50f * slowIndicatorStrength,
                                overlayTextRenderer);
    }
    if (app.panMode) {
      drawModifierSymbolIndicator(width, height, 'P', 34.0f, indicatorYOffset(indicatorSlot++), 1.0f);
    }
    if (app.rollMode) {
      const float feedback = clampf(static_cast<float>((app.rollFeedbackUntil - glfwGetTime()) / 0.18), 0.0f, 1.0f);
      const float breathe = 0.5f + 0.5f * std::sin(static_cast<float>(glfwGetTime()) * 8.0f);
      drawRollDirectionIndicator(width, height, app.rollDirection,
                                 0.30f + 0.70f * std::max(feedback, breathe * 0.6f),
                                 indicatorYOffset(indicatorSlot++),
                                 &hudSymbolText,
                                 &hudText);
    }
    if (app.orientAxisLock != 0) {
      const float pulse = clampf(static_cast<float>((app.orientAxisFeedbackUntil - glfwGetTime()) / 0.18), 0.0f, 1.0f);
      drawOrientationLockIndicator(width, height, app.orientAxisLock, pulse, indicatorYOffset(indicatorSlot++));
    }
    const float speedPulse = clampf(static_cast<float>((app.speedFeedbackUntil - glfwGetTime()) / 0.18), 0.0f, 1.0f);
    if (speedPulse > 0.0f) {
      drawModifierSymbolIndicator(width,
                                  height,
                                  'S',
                                  34.0f,
                                  indicatorYOffset(indicatorSlot++),
                                  0.55f + 0.45f * speedPulse);
    }
    if (resolved.neutralRadiusEnabled) {
      drawNeutralRadiusIndicator(width,
                                 height,
                                 42.0f,
                                 28.0f,
                                 0.82f,
                                 overlayTextRenderer);
    }

    glfwSwapBuffers(window);
  }

  gRun.store(false);
  wakeIpcServer();
#if defined(_WIN32)
  WaitForSingleObject(ipcThread, INFINITE);
  CloseHandle(ipcThread);
  CloseHandle(singletonMutex);
#else
  if (ipcThread.joinable()) ipcThread.join();
#endif
  releasePointBufferCache(&meshPointBufferCache);
  releasePointBufferCache(&overlayPointBufferCache);
  releaseOverlayComputeCache(&overlayComputeCache);
  releaseInputCloudComputeCache(&inputCloudComputeCache);
  releaseInputCloudSampleComputeCache(&inputCloudSampleComputeCache);
  releaseHudTextRenderer(&hudSymbolText);
  releaseHudTextRenderer(&hudText);
#if defined(CHROMASPACE_VIEWER_HAS_CUDA) && !defined(__APPLE__)
  releaseOverlayCudaCache(&overlayCudaCache);
  releaseInputCloudCudaCache(&inputCloudCudaCache);
  releaseInputCloudSampleCudaCache(&inputCloudSampleCudaCache);
#endif
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
