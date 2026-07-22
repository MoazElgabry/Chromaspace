#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ChromaspaceViewer {

constexpr std::size_t kViewerIpcEndpointMaximumBytes = 103u;

struct ViewerIpcEndpoint {
  bool valid = false;
  bool environmentOverride = false;
  std::string path;
  std::string diagnostic;
};

// Resolves the platform endpoint without consulting process-global state.
// Tests use this overload to validate overrides deterministically.
ViewerIpcEndpoint resolveViewerIpcEndpoint(
    std::string_view overrideValue) noexcept;

// Production resolver. CHROMASPACE_PIPE remains an explicit developer/test
// override; the normal POSIX endpoint is scoped by effective user ID.
ViewerIpcEndpoint viewerIpcEndpointFromEnvironment() noexcept;

}  // namespace ChromaspaceViewer
