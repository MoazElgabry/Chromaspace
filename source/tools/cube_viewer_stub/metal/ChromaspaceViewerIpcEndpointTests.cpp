#include "ChromaspaceViewerIpcEndpoint.h"

#include <cassert>
#include <string>

int main() {
  using namespace ChromaspaceViewer;

  const ViewerIpcEndpoint normal = resolveViewerIpcEndpoint({});
  assert(normal.valid);
  assert(!normal.environmentOverride);
  assert(!normal.path.empty());
  assert(normal.path.size() <= kViewerIpcEndpointMaximumBytes);
#if defined(_WIN32)
  assert(normal.path == R"(\\.\pipe\Chromaspace)");
  const auto custom = resolveViewerIpcEndpoint(R"(\\.\pipe\ChromaspaceTest)");
#else
  assert(normal.path.rfind("/tmp/chromaspace-", 0u) == 0u);
  assert(normal.path.find("/viewer.sock") != std::string::npos);
  const auto custom = resolveViewerIpcEndpoint("/tmp/chromaspace-test.sock");
#endif
  assert(custom.valid);
  assert(custom.environmentOverride);

#if defined(_WIN32)
  assert(!resolveViewerIpcEndpoint("relative-name").valid);
  assert(!resolveViewerIpcEndpoint(R"(C:\temp\viewer.sock)").valid);
#else
  assert(!resolveViewerIpcEndpoint("relative-name").valid);
#endif
  assert(!resolveViewerIpcEndpoint(std::string(
              kViewerIpcEndpointMaximumBytes + 1u, 'x'))
              .valid);
  assert(!resolveViewerIpcEndpoint(std::string("/tmp/bad\npath")).valid);
  return 0;
}
