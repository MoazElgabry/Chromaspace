#include "ChromaspaceViewerCommandServer.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace ChromaspaceViewer {
namespace {

void incrementSaturated(std::atomic<uint64_t>* value) noexcept {
  if (!value) return;
  uint64_t current = value->load(std::memory_order_relaxed);
  while (current != std::numeric_limits<uint64_t>::max() &&
         !value->compare_exchange_weak(current, current + 1u,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

#if !defined(_WIN32)

constexpr std::size_t kSocketReadChunkBytes = 16u * 1024u;

struct ClientConnection {
  int fd = -1;
  std::string line;
  std::string pendingOutput;
  bool discardingOversizedLine = false;
  std::chrono::steady_clock::time_point lastActivity{};
};

struct EndpointIdentity {
  bool valid = false;
  unsigned long long device = 0u;
  unsigned long long inode = 0u;
};

enum class ExistingEndpointProbeResult : uint8_t {
  Active = 0,
  Stale,
  Unknown,
};

bool setCloseOnExec(int fd) noexcept {
  if (fd < 0) return false;
  const int flags = ::fcntl(fd, F_GETFD, 0);
  if (flags < 0) return false;
  return ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

bool setNonBlocking(int fd) noexcept {
  if (fd < 0) return false;
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  return ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

bool setNoSigPipe(int fd) noexcept {
#if defined(SO_NOSIGPIPE)
  if (fd < 0) return false;
  int enabled = 1;
  return ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &enabled,
                      sizeof(enabled)) == 0;
#else
  (void)fd;
  return true;
#endif
}

void closeFd(int* fd) noexcept {
  if (!fd || *fd < 0) return;
  const int value = *fd;
  *fd = -1;
  while (::close(value) != 0 && errno == EINTR) {
  }
}

bool pathHasUnsafeComponent(std::string_view path) noexcept {
  std::size_t begin = 0u;
  while (begin < path.size()) {
    while (begin < path.size() && path[begin] == '/') ++begin;
    const std::size_t end = path.find('/', begin);
    const std::size_t componentEnd =
        end == std::string_view::npos ? path.size() : end;
    const std::string_view component = path.substr(begin, componentEnd - begin);
    if (component == "." || component == "..") return true;
    if (end == std::string_view::npos) break;
    begin = end + 1u;
  }
  return false;
}

bool statOwnedDirectory(const std::string& path, uid_t uid,
                        struct stat* output) noexcept {
  if (!output || ::lstat(path.c_str(), output) != 0) return false;
  return S_ISDIR(output->st_mode) && output->st_uid == uid;
}

bool ensureSecureDirectory(const std::string& directory, uid_t uid,
                           std::string* reason) noexcept {
  if (directory.empty() || directory == "/" ||
      pathHasUnsafeComponent(directory)) {
    if (reason) *reason = "ipc-directory-path-invalid";
    return false;
  }

  struct stat directoryStat {};
  if (::lstat(directory.c_str(), &directoryStat) != 0) {
    if (errno != ENOENT || ::mkdir(directory.c_str(), 0700) != 0) {
      if (reason) *reason = "ipc-directory-create-failed";
      return false;
    }
    if (!statOwnedDirectory(directory, uid, &directoryStat)) {
      if (reason) *reason = "ipc-directory-owner-invalid";
      return false;
    }
  } else if (!S_ISDIR(directoryStat.st_mode) ||
             directoryStat.st_uid != uid) {
    if (reason) *reason = "ipc-directory-owner-invalid";
    return false;
  }

  if ((directoryStat.st_mode & 0077) != 0) {
    if (::chmod(directory.c_str(), 0700) != 0 ||
        !statOwnedDirectory(directory, uid, &directoryStat) ||
        (directoryStat.st_mode & 0077) != 0) {
      if (reason) *reason = "ipc-directory-permissions-invalid";
      return false;
    }
  }
  return true;
}

bool splitParent(const std::string& path, std::string* parent) noexcept {
  if (!parent) return false;
  const std::size_t slash = path.rfind('/');
  if (slash == std::string::npos || slash == 0u || slash + 1u >= path.size()) {
    return false;
  }
  parent->assign(path.data(), slash);
  return !parent->empty();
}

bool buildUnixAddress(const std::string& path, sockaddr_un* address) noexcept {
  if (!address || path.empty() || path.size() >= sizeof(address->sun_path)) {
    return false;
  }
  *address = sockaddr_un{};
  address->sun_family = AF_UNIX;
  std::memcpy(address->sun_path, path.data(), path.size());
  address->sun_path[path.size()] = '\0';
  return true;
}

ExistingEndpointProbeResult probeExistingEndpoint(
    const std::string& path) noexcept {
  sockaddr_un address{};
  if (!buildUnixAddress(path, &address)) {
    return ExistingEndpointProbeResult::Unknown;
  }

  int probe = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (probe < 0) return ExistingEndpointProbeResult::Unknown;
  if (!setCloseOnExec(probe) || !setNonBlocking(probe) ||
      !setNoSigPipe(probe)) {
    closeFd(&probe);
    return ExistingEndpointProbeResult::Unknown;
  }

  const int result = ::connect(probe, reinterpret_cast<sockaddr*>(&address),
                               sizeof(address));
  const int connectError = result == 0 ? 0 : errno;
  closeFd(&probe);

  if (result == 0 || connectError == EINPROGRESS ||
      connectError == EALREADY || connectError == EAGAIN ||
      connectError == EWOULDBLOCK) {
    // Nonblocking connect may still be completing.  Treat that state as an
    // active endpoint rather than risking removal of another viewer's inode.
    return ExistingEndpointProbeResult::Active;
  }
  if (connectError == ECONNREFUSED) {
    // A refused AF_UNIX stream connect is the only result considered a stale
    // listener marker.  All other errors are retained conservatively.
    return ExistingEndpointProbeResult::Stale;
  }
  return ExistingEndpointProbeResult::Unknown;
}

bool recordIdentity(const std::string& path, EndpointIdentity* identity) noexcept {
  if (!identity) return false;
  struct stat socketStat {};
  if (::lstat(path.c_str(), &socketStat) != 0 ||
      !S_ISSOCK(socketStat.st_mode) || socketStat.st_uid != ::geteuid()) {
    return false;
  }
  identity->valid = true;
  identity->device = static_cast<unsigned long long>(socketStat.st_dev);
  identity->inode = static_cast<unsigned long long>(socketStat.st_ino);
  return true;
}

bool validateSocketPermissions(const std::string& path,
                               const EndpointIdentity& identity) noexcept {
  struct stat socketStat {};
  if (::lstat(path.c_str(), &socketStat) != 0 ||
      !S_ISSOCK(socketStat.st_mode) || socketStat.st_uid != ::geteuid() ||
      (socketStat.st_mode & 0077) != 0) {
    return false;
  }
  if (!identity.valid) return true;
  return static_cast<unsigned long long>(socketStat.st_dev) == identity.device &&
         static_cast<unsigned long long>(socketStat.st_ino) == identity.inode;
}

bool sendNoSignal(int fd, const char* bytes, std::size_t size,
                  ssize_t* sent) noexcept {
  if (!sent) return false;
  int flags = 0;
#if defined(MSG_NOSIGNAL)
  flags |= MSG_NOSIGNAL;
#endif
  const ssize_t result = ::send(fd, bytes, size, flags);
  if (result < 0) {
    *sent = -1;
    return false;
  }
  *sent = result;
  return true;
}

#endif  // !defined(_WIN32)

}  // namespace

struct ViewerCommandServer::PreparedSockets {
  int listener = -1;
  int wakeRead = -1;
  int wakeWrite = -1;
};

ViewerCommandServer::ViewerCommandServer(
    ViewerLiveCommandReducer* reducer, ViewerCommandServerOptions options)
    : reducer_(reducer), options_(std::move(options)) {
  try {
    endpoint_ = options_.endpoint.valid ? options_.endpoint
                                        : viewerIpcEndpointFromEnvironment();
  } catch (...) {
    endpoint_ = ViewerIpcEndpoint{};
    setError("ipc-endpoint-allocation-failure");
  }
}

ViewerCommandServer::~ViewerCommandServer() noexcept {
  (void)stopAndJoin();
}

void ViewerCommandServer::setError(const char* message) noexcept {
  try {
    std::lock_guard<std::mutex> lock(diagnosticMutex_);
    lastError_ = message ? message : "viewer-command-server-error";
  } catch (...) {
  }
}

void ViewerCommandServer::setError(const std::string& message) noexcept {
  try {
    std::lock_guard<std::mutex> lock(diagnosticMutex_);
    lastError_ = message;
  } catch (...) {
  }
}

void ViewerCommandServer::resetCounters() noexcept {
  activeConnections_.store(0u, std::memory_order_relaxed);
  acceptedConnections_.store(0u, std::memory_order_relaxed);
  closedConnections_.store(0u, std::memory_order_relaxed);
  submittedLines_.store(0u, std::memory_order_relaxed);
  rejectedLines_.store(0u, std::memory_order_relaxed);
  oversizedLines_.store(0u, std::memory_order_relaxed);
  incompleteLines_.store(0u, std::memory_order_relaxed);
  heartbeatCount_.store(0u, std::memory_order_relaxed);
  heartbeatAckCount_.store(0u, std::memory_order_relaxed);
  droppedHeartbeatAckCount_.store(0u, std::memory_order_relaxed);
}

#if !defined(_WIN32)

bool ViewerCommandServer::prepareSockets(PreparedSockets* prepared) noexcept {
  if (!prepared) return false;
  *prepared = PreparedSockets{};

  if (!reducer_) {
    setError("viewer-command-server-reducer-missing");
    return false;
  }
  if (!endpoint_.valid || endpoint_.path.empty()) {
    setError(endpoint_.diagnostic.empty() ? "ipc-endpoint-invalid"
                                           : endpoint_.diagnostic);
    return false;
  }
  if (endpoint_.path.size() >= sizeof(((sockaddr_un*)nullptr)->sun_path) ||
      endpoint_.path.front() != '/' || pathHasUnsafeComponent(endpoint_.path)) {
    setError("ipc-endpoint-path-invalid");
    return false;
  }
  if (options_.maximumLineBytes == 0u ||
      options_.maximumLineBytes > kViewerLiveCommandMaxLineBytes ||
      options_.maximumConnections == 0u ||
      options_.maximumConnections > kViewerCommandServerMaximumConnections ||
      options_.maximumPendingOutputBytes == 0u ||
      options_.idleTimeoutMilliseconds == 0u ||
      options_.idleTimeoutMilliseconds >
          kViewerCommandServerMaximumIdleTimeoutMilliseconds) {
    setError("viewer-command-server-bounds-invalid");
    return false;
  }

  std::string parent;
  if (!splitParent(endpoint_.path, &parent)) {
    setError("ipc-endpoint-parent-invalid");
    return false;
  }
  std::string directoryReason;
  if (!ensureSecureDirectory(parent, ::geteuid(), &directoryReason)) {
    setError(directoryReason.empty() ? "ipc-directory-invalid" : directoryReason);
    return false;
  }

  struct stat existing {};
  if (::lstat(endpoint_.path.c_str(), &existing) == 0) {
    if (!S_ISSOCK(existing.st_mode) || existing.st_uid != ::geteuid()) {
      setError("ipc-endpoint-collision-invalid");
      return false;
    }
    const ExistingEndpointProbeResult probe =
        probeExistingEndpoint(endpoint_.path);
    if (probe == ExistingEndpointProbeResult::Active) {
      setError("ipc-endpoint-already-active");
      return false;
    }
    if (probe == ExistingEndpointProbeResult::Unknown) {
      setError("ipc-endpoint-active-probe-failed");
      return false;
    }
    if (::unlink(endpoint_.path.c_str()) != 0) {
      setError("ipc-stale-endpoint-remove-failed");
      return false;
    }
  } else if (errno != ENOENT) {
    setError("ipc-endpoint-stat-failed");
    return false;
  }

  prepared->listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (prepared->listener < 0 || !setCloseOnExec(prepared->listener) ||
      !setNonBlocking(prepared->listener)) {
    setError("ipc-listener-create-failed");
    closeFd(&prepared->listener);
    return false;
  }
  sockaddr_un address{};
  if (!buildUnixAddress(endpoint_.path, &address) ||
      ::bind(prepared->listener, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
    setError("ipc-listener-bind-failed");
    closeFd(&prepared->listener);
    return false;
  }

  EndpointIdentity identity{};
  if (!recordIdentity(endpoint_.path, &identity) || !identity.valid) {
    setError("ipc-listener-identity-failed");
    closeFd(&prepared->listener);
    return false;
  }
  socketIdentityValid_ = true;
  socketDevice_ = identity.device;
  socketInode_ = identity.inode;
  if (::chmod(endpoint_.path.c_str(), 0600) != 0 ||
      !validateSocketPermissions(endpoint_.path, identity) ||
      ::listen(prepared->listener,
               static_cast<int>(options_.maximumConnections)) != 0) {
    setError("ipc-listener-permissions-or-listen-failed");
    closeFd(&prepared->listener);
    closeEndpointIfOwned();
    return false;
  }

  int wake[2] = {-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM, 0, wake) != 0 ||
      !setCloseOnExec(wake[0]) || !setCloseOnExec(wake[1]) ||
      !setNonBlocking(wake[0]) || !setNonBlocking(wake[1]) ||
      !setNoSigPipe(wake[0]) || !setNoSigPipe(wake[1])) {
    setError("ipc-wake-create-failed");
    closeFd(&wake[0]);
    closeFd(&wake[1]);
    closeFd(&prepared->listener);
    closeEndpointIfOwned();
    return false;
  }
  prepared->wakeRead = wake[0];
  prepared->wakeWrite = wake[1];
  return true;
}

void ViewerCommandServer::closeEndpointIfOwned() noexcept {
  if (!socketIdentityValid_ || endpoint_.path.empty()) return;
  struct stat current {};
  if (::lstat(endpoint_.path.c_str(), &current) == 0 &&
      S_ISSOCK(current.st_mode) && current.st_uid == ::geteuid() &&
      static_cast<unsigned long long>(current.st_dev) == socketDevice_ &&
      static_cast<unsigned long long>(current.st_ino) == socketInode_) {
    (void)::unlink(endpoint_.path.c_str());
  }
  socketIdentityValid_ = false;
  socketDevice_ = 0u;
  socketInode_ = 0u;
}

#else

bool ViewerCommandServer::prepareSockets(PreparedSockets*) noexcept {
  setError("viewer-command-server-unsupported-platform");
  return false;
}

void ViewerCommandServer::closeEndpointIfOwned() noexcept {}

#endif  // !defined(_WIN32)

bool ViewerCommandServer::start() noexcept {
  try {
    std::lock_guard<std::mutex> lock(lifecycleMutex_);
    if (worker_.joinable() ||
        state_.load(std::memory_order_acquire) ==
            ViewerCommandServerState::Running ||
        state_.load(std::memory_order_acquire) ==
            ViewerCommandServerState::Starting ||
        state_.load(std::memory_order_acquire) ==
            ViewerCommandServerState::StopRequested) {
      return false;
    }
    resetCounters();
    {
      std::lock_guard<std::mutex> diagnosticLock(diagnosticMutex_);
      lastError_.clear();
    }
    listenerReady_.store(false, std::memory_order_release);
    stopRequested_.store(false, std::memory_order_release);
    state_.store(ViewerCommandServerState::Starting, std::memory_order_release);

    PreparedSockets prepared{};
    if (!prepareSockets(&prepared)) {
#if !defined(_WIN32)
      closeFd(&prepared.listener);
      closeFd(&prepared.wakeRead);
      closeFd(&prepared.wakeWrite);
#endif
      closeEndpointIfOwned();
      state_.store(ViewerCommandServerState::Failed,
                   std::memory_order_release);
      return false;
    }
    listenerFd_.store(prepared.listener, std::memory_order_release);
    wakeReadFd_.store(prepared.wakeRead, std::memory_order_release);
    wakeWriteFd_.store(prepared.wakeWrite, std::memory_order_release);
    listenerReady_.store(true, std::memory_order_release);
    state_.store(ViewerCommandServerState::Running, std::memory_order_release);

#if !defined(_WIN32)
    // The worker owns the read side and the listener from this point forward.
    worker_ = std::thread([this]() noexcept { run(); });
#else
    (void)prepared;
#endif
    return true;
  } catch (const std::bad_alloc&) {
    setError("viewer-command-server-allocation-failure");
  } catch (...) {
    setError("viewer-command-server-start-failure");
  }

#if !defined(_WIN32)
  {
    int listener = listenerFd_.exchange(-1, std::memory_order_acq_rel);
    int wakeRead = wakeReadFd_.exchange(-1, std::memory_order_acq_rel);
    int wakeWrite = wakeWriteFd_.exchange(-1, std::memory_order_acq_rel);
    closeFd(&listener);
    closeFd(&wakeRead);
    closeFd(&wakeWrite);
    closeEndpointIfOwned();
  }
#endif
  listenerReady_.store(false, std::memory_order_release);
  stopRequested_.store(true, std::memory_order_release);
  state_.store(ViewerCommandServerState::Failed, std::memory_order_release);
  return false;
}

void ViewerCommandServer::requestStop() noexcept {
  stopRequested_.store(true, std::memory_order_release);
  const ViewerCommandServerState current =
      state_.load(std::memory_order_acquire);
  if (current == ViewerCommandServerState::Running ||
      current == ViewerCommandServerState::Starting) {
    state_.store(ViewerCommandServerState::StopRequested,
                 std::memory_order_release);
  }
#if !defined(_WIN32)
  const int wake = wakeWriteFd_.load(std::memory_order_acquire);
  if (wake >= 0) {
    const char byte = 'x';
    ssize_t ignored = 0;
    (void)sendNoSignal(wake, &byte, 1u, &ignored);
  }
#endif
}

bool ViewerCommandServer::join() noexcept {
  try {
    std::lock_guard<std::mutex> lock(lifecycleMutex_);
    if (!worker_.joinable()) return true;
    if (worker_.get_id() == std::this_thread::get_id()) return false;
    worker_.join();
    return true;
  } catch (...) {
    return false;
  }
}

bool ViewerCommandServer::stopAndJoin() noexcept {
  requestStop();
  return join();
}

ViewerCommandServerState ViewerCommandServer::state() const noexcept {
  return state_.load(std::memory_order_acquire);
}

ViewerCommandServerSnapshot ViewerCommandServer::snapshot() const noexcept {
  ViewerCommandServerSnapshot output{};
  try {
    output.state = state();
    output.listenerReady = listenerReady_.load(std::memory_order_acquire);
    output.activeConnections = activeConnections_.load(std::memory_order_relaxed);
    output.acceptedConnections =
        acceptedConnections_.load(std::memory_order_relaxed);
    output.closedConnections = closedConnections_.load(std::memory_order_relaxed);
    output.submittedLines = submittedLines_.load(std::memory_order_relaxed);
    output.rejectedLines = rejectedLines_.load(std::memory_order_relaxed);
    output.oversizedLines = oversizedLines_.load(std::memory_order_relaxed);
    output.incompleteLines = incompleteLines_.load(std::memory_order_relaxed);
    output.heartbeatCount = heartbeatCount_.load(std::memory_order_relaxed);
    output.heartbeatAckCount =
        heartbeatAckCount_.load(std::memory_order_relaxed);
    output.droppedHeartbeatAckCount =
        droppedHeartbeatAckCount_.load(std::memory_order_relaxed);
    output.endpointPath = endpoint_.path;
    {
      std::lock_guard<std::mutex> lock(diagnosticMutex_);
      output.lastError = lastError_;
    }
  } catch (...) {
    output.state = ViewerCommandServerState::Failed;
  }
  return output;
}

bool ViewerCommandServer::joinable() const noexcept {
  try {
    std::lock_guard<std::mutex> lock(lifecycleMutex_);
    return worker_.joinable();
  } catch (...) {
    return false;
  }
}

#if !defined(_WIN32)

void ViewerCommandServer::run() noexcept {
  int listener = listenerFd_.load(std::memory_order_acquire);
  int wakeRead = wakeReadFd_.load(std::memory_order_acquire);
  std::vector<ClientConnection> clients;

  try {
    clients.reserve(options_.maximumConnections);

    auto decrementActive = [this]() noexcept {
      std::size_t current = activeConnections_.load(std::memory_order_relaxed);
      while (current != 0u &&
             !activeConnections_.compare_exchange_weak(
                 current, current - 1u, std::memory_order_relaxed,
                 std::memory_order_relaxed)) {
      }
    };

    auto closeClient = [&](ClientConnection* client) noexcept {
      if (!client) return;
      if (!client->line.empty() || client->discardingOversizedLine) {
        incrementSaturated(&incompleteLines_);
      }
      closeFd(&client->fd);
      decrementActive();
      incrementSaturated(&closedConnections_);
    };

    auto queueHeartbeatAck = [&](ClientConnection* client,
                                 const ViewerLiveCommandSubmitResult& result)
        noexcept {
      if (!client || !result.accepted() ||
          result.kind != ViewerLiveCommandKind::Heartbeat) {
        return;
      }
      incrementSaturated(&heartbeatCount_);
      if (!options_.heartbeatAck) return;
      try {
        ViewerCommandServerHeartbeatRequest request{};
        request.seq = result.seq;
        request.senderId = result.senderId;
        std::string response;
        const ViewerCommandServerSnapshot current = snapshot();
        bool produced = false;
        try {
          produced = options_.heartbeatAck(request, current, &response,
                                            options_.heartbeatAckContext);
        } catch (...) {
          produced = false;
        }
        if (!produced || response.empty()) return;
        if (!response.empty() && response.back() == '\n') response.pop_back();
        if (!response.empty() && response.back() == '\r') response.pop_back();
        if (response.empty() || response.find_first_of("\r\n") !=
                                    std::string::npos ||
            response.size() >= options_.maximumPendingOutputBytes ||
            client->pendingOutput.size() >
                options_.maximumPendingOutputBytes - response.size() - 1u) {
          incrementSaturated(&droppedHeartbeatAckCount_);
          return;
        }
        response.push_back('\n');
        client->pendingOutput.append(response);
        incrementSaturated(&heartbeatAckCount_);
      } catch (...) {
        incrementSaturated(&droppedHeartbeatAckCount_);
      }
    };

    auto submitLine = [&](ClientConnection* client, std::string_view line)
        noexcept {
      if (!reducer_) return;
      ViewerLiveCommandSubmitResult result{};
      try {
        result = reducer_->submitLine(line);
      } catch (...) {
        incrementSaturated(&rejectedLines_);
        return;
      }
      incrementSaturated(&submittedLines_);
      if (!result.accepted()) incrementSaturated(&rejectedLines_);
      queueHeartbeatAck(client, result);
    };

    auto consumeBytes = [&](ClientConnection* client, const char* bytes,
                            std::size_t count) noexcept {
      if (!client || !bytes) return;
      std::size_t offset = 0u;
      while (offset < count) {
        if (client->discardingOversizedLine) {
          const void* found = std::memchr(bytes + offset, '\n', count - offset);
          if (!found) return;
          offset = static_cast<std::size_t>(
                       static_cast<const char*>(found) - bytes) +
                   1u;
          client->discardingOversizedLine = false;
          continue;
        }

        const void* found = std::memchr(bytes + offset, '\n', count - offset);
        const std::size_t segmentEnd =
            found ? static_cast<std::size_t>(
                       static_cast<const char*>(found) - bytes)
                  : count;
        const std::size_t segmentBytes = segmentEnd - offset;
        const std::size_t maximumBufferedBytes =
            options_.maximumLineBytes + 1u;  // optional CR before LF
        if (segmentBytes > maximumBufferedBytes -
                               std::min(maximumBufferedBytes,
                                        client->line.size())) {
          client->line.clear();
          incrementSaturated(&oversizedLines_);
          if (!found) {
            client->discardingOversizedLine = true;
            return;
          }
          offset = segmentEnd + 1u;
          continue;
        }

        try {
          client->line.append(bytes + offset, segmentBytes);
        } catch (...) {
          client->line.clear();
          incrementSaturated(&oversizedLines_);
          client->discardingOversizedLine = !found;
          if (found) offset = segmentEnd + 1u;
          return;
        }
        offset = segmentEnd;
        if (!found) return;

        std::string_view complete(client->line);
        if (!complete.empty() && complete.back() == '\r') {
          complete.remove_suffix(1u);
        }
        if (complete.size() > options_.maximumLineBytes) {
          incrementSaturated(&oversizedLines_);
        } else {
          submitLine(client, complete);
        }
        client->line.clear();
        ++offset;  // consume LF
      }
    };

    const std::chrono::milliseconds idleTimeout(
        options_.idleTimeoutMilliseconds);

    while (!stopRequested_.load(std::memory_order_acquire)) {
      std::vector<pollfd> pollDescriptors;
      pollDescriptors.reserve(2u + clients.size());
      pollDescriptors.push_back(pollfd{wakeRead, POLLIN, 0});
      pollDescriptors.push_back(pollfd{listener, POLLIN, 0});
      for (const ClientConnection& client : clients) {
        short events = POLLIN;
        if (!client.pendingOutput.empty()) events |= POLLOUT;
        pollDescriptors.push_back(pollfd{client.fd, events, 0});
      }

      int pollTimeout = -1;
      if (!clients.empty()) {
        const auto now = std::chrono::steady_clock::now();
        pollTimeout = static_cast<int>(options_.idleTimeoutMilliseconds);
        for (const ClientConnection& client : clients) {
          const auto elapsed = std::chrono::duration_cast<
              std::chrono::milliseconds>(now - client.lastActivity);
          if (elapsed >= idleTimeout) {
            pollTimeout = 0;
            break;
          }
          const auto remaining =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  idleTimeout - elapsed);
          const int remainingMilliseconds =
              static_cast<int>(remaining.count() <= 0 ? 1 : remaining.count());
          pollTimeout = std::min(pollTimeout, remainingMilliseconds);
        }
      }

      const int ready = ::poll(pollDescriptors.data(),
                               static_cast<nfds_t>(pollDescriptors.size()),
                               pollTimeout);
      if (ready < 0) {
        if (errno == EINTR) continue;
        setError("ipc-poll-failed");
        break;
      }
      if ((pollDescriptors[0].revents & POLLIN) != 0) {
        char wakeBytes[64];
        while (::recv(wakeRead, wakeBytes, sizeof(wakeBytes), MSG_DONTWAIT) >
               0) {
        }
      }
      if (stopRequested_.load(std::memory_order_acquire)) break;

      if ((pollDescriptors[1].revents & POLLIN) != 0) {
        for (;;) {
          const int clientFd = ::accept(listener, nullptr, nullptr);
          if (clientFd < 0) {
            if (errno == EINTR) continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
              setError("ipc-accept-failed");
            }
            break;
          }
          if (clients.size() >= options_.maximumConnections ||
              !setCloseOnExec(clientFd) || !setNonBlocking(clientFd) ||
              !setNoSigPipe(clientFd)) {
            int rejected = clientFd;
            closeFd(&rejected);
            continue;
          }
          try {
            ClientConnection clientConnection{};
            clientConnection.fd = clientFd;
            clientConnection.lastActivity =
                std::chrono::steady_clock::now();
            clients.push_back(std::move(clientConnection));
            activeConnections_.fetch_add(1u, std::memory_order_relaxed);
            incrementSaturated(&acceptedConnections_);
          } catch (...) {
            int rejected = clientFd;
            closeFd(&rejected);
          }
        }
      } else if ((pollDescriptors[1].revents & (POLLERR | POLLHUP | POLLNVAL)) !=
                 0) {
        setError("ipc-listener-failed");
        break;
      }

      // Process clients in reverse order so erasing a closed connection does
      // not invalidate the indices of clients that still need service.
      for (std::size_t index = clients.size(); index > 0u; --index) {
        const std::size_t clientIndex = index - 1u;
        if (clientIndex + 2u >= pollDescriptors.size()) continue;
        const short events = pollDescriptors[clientIndex + 2u].revents;
        if (events == 0) continue;
        ClientConnection& client = clients[clientIndex];
        const bool invalidDescriptor = (events & POLLNVAL) != 0;
        const bool terminalSocketEvent = (events & (POLLERR | POLLHUP)) != 0;
        bool closeConnection = invalidDescriptor;

        if (!closeConnection && (events & POLLOUT) != 0 &&
            !client.pendingOutput.empty()) {
          ssize_t sent = 0;
          const bool sentResult = sendNoSignal(
              client.fd, client.pendingOutput.data(), client.pendingOutput.size(),
              &sent);
          if (sentResult && sent > 0) {
            client.pendingOutput.erase(0u, static_cast<std::size_t>(sent));
            client.lastActivity = std::chrono::steady_clock::now();
          } else if (!sentResult && errno != EAGAIN && errno != EWOULDBLOCK &&
                     errno != EINTR) {
            closeConnection = true;
          }
        }

        // A peer that writes and immediately closes may surface POLLIN and
        // POLLHUP together (notably on macOS).  Drain bytes already queued by
        // the kernel before honoring the terminal event, otherwise complete
        // commands and transport diagnostics can be silently lost.  A
        // terminal socket has a finite receive queue, so continue to EOF or
        // EAGAIN; ordinary readable sockets keep the one-chunk fairness bound.
        const bool shouldRead =
            !invalidDescriptor &&
            (events & (POLLIN | POLLERR | POLLHUP)) != 0;
        if (shouldRead) {
          for (;;) {
            char bytes[kSocketReadChunkBytes];
            const ssize_t count =
                ::recv(client.fd, bytes, sizeof(bytes), MSG_DONTWAIT);
            if (count > 0) {
              client.lastActivity = std::chrono::steady_clock::now();
              consumeBytes(&client, bytes, static_cast<std::size_t>(count));
              if (!terminalSocketEvent) break;
              continue;
            }
            if (count == 0) {
              closeConnection = true;
              break;
            }
            if (errno == EINTR) continue;
            if (errno != EAGAIN && errno != EWOULDBLOCK) {
              closeConnection = true;
            }
            break;
          }
        }
        if (terminalSocketEvent) closeConnection = true;

        if (closeConnection) {
          closeClient(&client);
          clients.erase(clients.begin() + static_cast<std::ptrdiff_t>(clientIndex));
        }
      }

      const auto idleNow = std::chrono::steady_clock::now();
      for (std::size_t index = clients.size(); index > 0u; --index) {
        const std::size_t clientIndex = index - 1u;
        const auto inactive = std::chrono::duration_cast<
            std::chrono::milliseconds>(idleNow -
                                       clients[clientIndex].lastActivity);
        if (inactive < idleTimeout) continue;
        ClientConnection& client = clients[clientIndex];
        closeClient(&client);
        clients.erase(clients.begin() +
                      static_cast<std::ptrdiff_t>(clientIndex));
      }
    }

    for (ClientConnection& client : clients) closeClient(&client);
    clients.clear();
  } catch (const std::bad_alloc&) {
    setError("viewer-command-server-allocation-failure");
  } catch (...) {
    setError("viewer-command-server-worker-failure");
  }

  for (ClientConnection& client : clients) {
    closeFd(&client.fd);
  }
  clients.clear();
  activeConnections_.store(0u, std::memory_order_relaxed);
  closeFd(&listener);
  closeFd(&wakeRead);
  listenerFd_.store(-1, std::memory_order_release);
  wakeReadFd_.store(-1, std::memory_order_release);
  const int wakeWrite = wakeWriteFd_.exchange(-1, std::memory_order_acq_rel);
  int writeFd = wakeWrite;
  closeFd(&writeFd);
  listenerReady_.store(false, std::memory_order_release);
  closeEndpointIfOwned();

  const ViewerCommandServerState finalState =
      stopRequested_.load(std::memory_order_acquire)
          ? ViewerCommandServerState::Stopped
          : ViewerCommandServerState::Failed;
  if (finalState == ViewerCommandServerState::Failed) {
    ViewerCommandServerSnapshot current = snapshot();
    if (current.lastError.empty()) setError("ipc-worker-stopped-unexpectedly");
  }
  state_.store(finalState, std::memory_order_release);
}

#else

void ViewerCommandServer::run() noexcept {}

#endif

}  // namespace ChromaspaceViewer
