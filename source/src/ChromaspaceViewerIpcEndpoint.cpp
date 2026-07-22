#include "ChromaspaceViewerIpcEndpoint.h"

#include <cstdlib>
#include <new>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace ChromaspaceViewer {
namespace {

ViewerIpcEndpoint invalidEndpoint(const char* diagnostic) noexcept {
  ViewerIpcEndpoint endpoint{};
  try {
    endpoint.diagnostic = diagnostic ? diagnostic : "ipc-endpoint-invalid";
  } catch (...) {
  }
  return endpoint;
}

bool containsControlOrNul(std::string_view value) noexcept {
  for (const unsigned char c : value) {
    if (c == 0u || c < 0x20u || c == 0x7fu) return true;
  }
  return false;
}

}  // namespace

ViewerIpcEndpoint resolveViewerIpcEndpoint(
    std::string_view overrideValue) noexcept {
  try {
    ViewerIpcEndpoint endpoint{};
    if (!overrideValue.empty()) {
      if (overrideValue.size() > kViewerIpcEndpointMaximumBytes) {
        return invalidEndpoint("ipc-override-too-long");
      }
      if (containsControlOrNul(overrideValue)) {
        return invalidEndpoint("ipc-override-control-character");
      }
#if defined(_WIN32)
      if (overrideValue.rfind(R"(\\.\pipe\)", 0u) != 0u) {
        return invalidEndpoint("ipc-override-not-named-pipe");
      }
#else
      if (overrideValue.front() != '/') {
        return invalidEndpoint("ipc-override-not-absolute");
      }
#endif
      endpoint.path.assign(overrideValue.data(), overrideValue.size());
      endpoint.valid = true;
      endpoint.environmentOverride = true;
      return endpoint;
    }

#if defined(_WIN32)
    endpoint.path = R"(\\.\pipe\Chromaspace)";
#else
    endpoint.path = "/tmp/chromaspace-" +
                    std::to_string(static_cast<unsigned long long>(geteuid())) +
                    "/viewer.sock";
#endif
    if (endpoint.path.empty() ||
        endpoint.path.size() > kViewerIpcEndpointMaximumBytes) {
      return invalidEndpoint("ipc-default-invalid");
    }
    endpoint.valid = true;
    return endpoint;
  } catch (const std::bad_alloc&) {
    return invalidEndpoint("ipc-endpoint-allocation-failure");
  } catch (...) {
    return invalidEndpoint("ipc-endpoint-resolution-failure");
  }
}

ViewerIpcEndpoint viewerIpcEndpointFromEnvironment() noexcept {
  const char* value = std::getenv("CHROMASPACE_PIPE");
  return resolveViewerIpcEndpoint(value ? std::string_view(value)
                                        : std::string_view{});
}

}  // namespace ChromaspaceViewer
