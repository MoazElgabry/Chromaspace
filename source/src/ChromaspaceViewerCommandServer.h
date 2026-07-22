#pragma once

#include "ChromaspaceViewerIpcEndpoint.h"
#include "ChromaspaceViewerLiveCommand.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace ChromaspaceViewer {

// The command server owns only the transport boundary.  JSON parsing,
// sequence ordering, sender handoff, and coalescing remain in
// ViewerLiveCommandReducer.
constexpr std::size_t kViewerCommandServerDefaultMaximumConnections = 8u;
constexpr std::size_t kViewerCommandServerMaximumConnections = 64u;
constexpr std::size_t kViewerCommandServerDefaultMaximumPendingOutputBytes =
    64u * 1024u;
constexpr uint32_t kViewerCommandServerDefaultIdleTimeoutMilliseconds =
    5000u;
constexpr uint32_t kViewerCommandServerMaximumIdleTimeoutMilliseconds =
    60000u;

enum class ViewerCommandServerState : uint8_t {
  Stopped = 0,
  Starting,
  Running,
  StopRequested,
  Failed,
};

struct ViewerCommandServerSnapshot {
  ViewerCommandServerState state = ViewerCommandServerState::Stopped;
  bool listenerReady = false;
  std::size_t activeConnections = 0u;
  uint64_t acceptedConnections = 0u;
  uint64_t closedConnections = 0u;
  uint64_t submittedLines = 0u;
  uint64_t rejectedLines = 0u;
  uint64_t oversizedLines = 0u;
  uint64_t incompleteLines = 0u;
  uint64_t heartbeatCount = 0u;
  uint64_t heartbeatAckCount = 0u;
  uint64_t droppedHeartbeatAckCount = 0u;
  std::string endpointPath;
  std::string lastError;
};

struct ViewerCommandServerHeartbeatRequest {
  uint64_t seq = 0u;
  std::string senderId;
};

// The callback runs on the command-server thread after an accepted heartbeat
// has been submitted to the reducer.  It may return one bounded JSON object;
// the server appends the JSONL newline and queues the response on that same
// client connection.  Returning false or an empty response sends nothing.
// The context remains caller-owned and must outlive stopAndJoin().
using ViewerCommandServerHeartbeatAckHook = bool (*)(
    const ViewerCommandServerHeartbeatRequest& request,
    const ViewerCommandServerSnapshot& snapshot,
    std::string* response,
    void* context);

struct ViewerCommandServerOptions {
  // An invalid endpoint requests the production environment resolver during
  // construction.  Tests and direct-Cocoa adapters should pass an explicit
  // resolved endpoint so they do not depend on process-global environment.
  ViewerIpcEndpoint endpoint{};
  std::size_t maximumLineBytes = kViewerLiveCommandMaxLineBytes;
  std::size_t maximumConnections =
      kViewerCommandServerDefaultMaximumConnections;
  std::size_t maximumPendingOutputBytes =
      kViewerCommandServerDefaultMaximumPendingOutputBytes;
  // Idle clients are closed after this receive/send inactivity window.  The
  // bounded value prevents idle connections from exhausting the listener's
  // finite connection slots; zero and values above the capped maximum are
  // rejected by start().
  uint32_t idleTimeoutMilliseconds =
      kViewerCommandServerDefaultIdleTimeoutMilliseconds;
  ViewerCommandServerHeartbeatAckHook heartbeatAck = nullptr;
  void* heartbeatAckContext = nullptr;
};

class ViewerCommandServer final {
 public:
  explicit ViewerCommandServer(
      ViewerLiveCommandReducer* reducer,
      ViewerCommandServerOptions options = ViewerCommandServerOptions{});
  ~ViewerCommandServer() noexcept;

  ViewerCommandServer(const ViewerCommandServer&) = delete;
  ViewerCommandServer& operator=(const ViewerCommandServer&) = delete;
  ViewerCommandServer(ViewerCommandServer&&) = delete;
  ViewerCommandServer& operator=(ViewerCommandServer&&) = delete;

  // start() performs endpoint validation, secure-directory preparation,
  // socket creation, and listener setup synchronously.  A true result means
  // the joinable worker owns a ready listener.
  bool start() noexcept;

  // requestStop() is non-blocking.  It sets the stop flag and wakes poll()
  // through the private socketpair; stopAndJoin() is the deterministic teardown
  // helper for owners.
  void requestStop() noexcept;
  void stop() noexcept { requestStop(); }
  bool join() noexcept;
  bool stopAndJoin() noexcept;

  ViewerCommandServerState state() const noexcept;
  ViewerCommandServerSnapshot snapshot() const noexcept;
  bool running() const noexcept {
    return state() == ViewerCommandServerState::Running;
  }
  bool joinable() const noexcept;
  const ViewerIpcEndpoint& endpoint() const noexcept { return endpoint_; }

 private:
  struct PreparedSockets;

  bool prepareSockets(PreparedSockets* prepared) noexcept;
  void run() noexcept;
  void setError(const char* message) noexcept;
  void setError(const std::string& message) noexcept;
  void closeEndpointIfOwned() noexcept;
  void resetCounters() noexcept;

  ViewerLiveCommandReducer* reducer_ = nullptr;
  ViewerCommandServerOptions options_{};
  ViewerIpcEndpoint endpoint_{};

  mutable std::mutex lifecycleMutex_;
  mutable std::mutex diagnosticMutex_;
  std::thread worker_;
  std::atomic<ViewerCommandServerState> state_{
      ViewerCommandServerState::Stopped};
  std::atomic<bool> stopRequested_{false};
  std::atomic<int> wakeReadFd_{-1};
  std::atomic<int> wakeWriteFd_{-1};
  std::atomic<int> listenerFd_{-1};

  std::atomic<std::size_t> activeConnections_{0u};
  std::atomic<uint64_t> acceptedConnections_{0u};
  std::atomic<uint64_t> closedConnections_{0u};
  std::atomic<uint64_t> submittedLines_{0u};
  std::atomic<uint64_t> rejectedLines_{0u};
  std::atomic<uint64_t> oversizedLines_{0u};
  std::atomic<uint64_t> incompleteLines_{0u};
  std::atomic<uint64_t> heartbeatCount_{0u};
  std::atomic<uint64_t> heartbeatAckCount_{0u};
  std::atomic<uint64_t> droppedHeartbeatAckCount_{0u};

#if !defined(_WIN32)
  bool socketIdentityValid_ = false;
  unsigned long long socketDevice_ = 0u;
  unsigned long long socketInode_ = 0u;
#endif
  std::atomic<bool> listenerReady_{false};
  std::string lastError_;
};

}  // namespace ChromaspaceViewer
