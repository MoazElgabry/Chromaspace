#include "ChromaspaceViewerCommandServer.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace {

#if !defined(_WIN32)

struct HeartbeatContext {
  std::atomic<uint64_t> calls{0u};
  std::atomic<uint64_t> snapshotConnections{0u};
};

bool heartbeatAck(
    const ChromaspaceViewer::ViewerCommandServerHeartbeatRequest& request,
    const ChromaspaceViewer::ViewerCommandServerSnapshot& snapshot,
    std::string* response, void* context) {
  auto* state = static_cast<HeartbeatContext*>(context);
  if (!state || !response || request.seq == 0u || request.senderId.empty()) {
    return false;
  }
  state->calls.fetch_add(1u, std::memory_order_relaxed);
  state->snapshotConnections.store(snapshot.activeConnections,
                                   std::memory_order_relaxed);
  *response = "{\"type\":\"heartbeat_ack\",\"seq\":" +
              std::to_string(request.seq) + ",\"senderId\":\"" +
              request.senderId + "\"}";
  return true;
}

std::string testSocketPath() {
  return "/tmp/chromaspace-" +
         std::to_string(static_cast<unsigned long long>(::geteuid())) +
         "/command-server-test-" + std::to_string(static_cast<long long>(::getpid())) +
         ".sock";
}

bool connectSocket(const std::string& path, int* output) {
  if (!output) return false;
  *output = -1;
  for (int attempt = 0; attempt != 100; ++attempt) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    if (path.size() >= sizeof(address.sun_path)) {
      ::close(fd);
      return false;
    }
    std::memcpy(address.sun_path, path.data(), path.size());
    address.sun_path[path.size()] = '\0';
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address),
                  sizeof(address)) == 0) {
      *output = fd;
      return true;
    }
    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

bool receiveLine(int fd, std::string* output) {
  if (fd < 0 || !output) return false;
  output->clear();
  char bytes[256];
  for (int attempt = 0; attempt != 100; ++attempt) {
    pollfd descriptor{fd, POLLIN, 0};
    const int ready = ::poll(&descriptor, 1u, 10);
    if (ready < 0 && errno == EINTR) continue;
    if (ready <= 0) continue;
    const ssize_t count = ::recv(fd, bytes, sizeof(bytes), 0);
    if (count <= 0) return false;
    output->append(bytes, static_cast<std::size_t>(count));
    const std::size_t newline = output->find('\n');
    if (newline != std::string::npos) {
      output->resize(newline);
      return true;
    }
  }
  return false;
}

bool waitForParams(ChromaspaceViewer::ViewerLiveCommandReducer* reducer,
                   uint64_t expectedSequence) {
  if (!reducer) return false;
  for (int attempt = 0; attempt != 100; ++attempt) {
    ChromaspaceViewer::ViewerLiveCommandBatch batch{};
    if (reducer->drain(&batch) && batch.hasParams &&
        batch.params.seq == expectedSequence) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return false;
}

template <typename Predicate>
bool waitForServerSnapshot(
    ChromaspaceViewer::ViewerCommandServer* server, Predicate predicate,
    std::chrono::milliseconds timeout = std::chrono::seconds(1)) {
  if (!server) return false;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    if (predicate(server->snapshot())) return true;
    if (std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return predicate(server->snapshot());
}

#endif

}  // namespace

int main() {
#if defined(_WIN32)
  // The bounded server is intentionally a POSIX/macOS transport seam.  Keep
  // the portable target buildable on Windows; named-pipe ownership remains a
  // separate migration decision.
  return 0;
#else
  using namespace ChromaspaceViewer;

  const std::string path = testSocketPath();
  (void)::unlink(path.c_str());
  const ViewerIpcEndpoint endpoint = resolveViewerIpcEndpoint(path);
  assert(endpoint.valid);

  ViewerLiveCommandReducer reducer;
  HeartbeatContext context{};
  ViewerCommandServerOptions options{};
  options.endpoint = endpoint;
  options.maximumLineBytes = 128u;
  options.maximumConnections = 2u;
  options.idleTimeoutMilliseconds = 100u;
  options.heartbeatAck = heartbeatAck;
  options.heartbeatAckContext = &context;
  ViewerCommandServer server(&reducer, options);
  assert(server.state() == ViewerCommandServerState::Stopped);
  assert(server.start());
  struct stat endpointBeforeDuplicate {};
  assert(::lstat(path.c_str(), &endpointBeforeDuplicate) == 0);
  assert(S_ISSOCK(endpointBeforeDuplicate.st_mode));

  // A second launch must preserve the active listener inode and leave the
  // first server usable.  The endpoint probe runs before any stale unlink.
  ViewerCommandServer duplicateServer(&reducer, options);
  assert(!duplicateServer.start());
  assert(duplicateServer.state() == ViewerCommandServerState::Failed);
  assert(duplicateServer.snapshot().lastError ==
         "ipc-endpoint-already-active");
  struct stat activeEndpointStat {};
  assert(::lstat(path.c_str(), &activeEndpointStat) == 0);
  assert(S_ISSOCK(activeEndpointStat.st_mode));
  assert(activeEndpointStat.st_dev == endpointBeforeDuplicate.st_dev);
  assert(activeEndpointStat.st_ino == endpointBeforeDuplicate.st_ino);
  assert(server.snapshot().listenerReady);
  const auto beforeDuplicateSync = server.snapshot();
  int duplicateSyncClient = -1;
  assert(connectSocket(path, &duplicateSyncClient));
  const std::string duplicateSyncParams =
      "{\"type\":\"params\",\"seq\":1,\"senderId\":\"duplicate-sync\"}\n";
  assert(::send(duplicateSyncClient, duplicateSyncParams.data(),
                duplicateSyncParams.size(), 0) ==
         static_cast<ssize_t>(duplicateSyncParams.size()));
  assert(waitForParams(&reducer, 1u));
  ::close(duplicateSyncClient);
  assert(waitForServerSnapshot(
      &server, [beforeDuplicateSync](const ViewerCommandServerSnapshot& value) {
        return value.acceptedConnections >
                   beforeDuplicateSync.acceptedConnections &&
               value.closedConnections > beforeDuplicateSync.closedConnections &&
               value.activeConnections == 0u;
      }));

  struct stat directoryStat {};
  const std::string directory = path.substr(0u, path.rfind('/'));
  assert(::lstat(directory.c_str(), &directoryStat) == 0);
  assert(S_ISDIR(directoryStat.st_mode));
  assert(directoryStat.st_uid == ::geteuid());
  assert((directoryStat.st_mode & 0077) == 0);

  int client = -1;
  assert(connectSocket(path, &client));
  struct stat socketStat {};
  assert(::lstat(path.c_str(), &socketStat) == 0);
  assert(S_ISSOCK(socketStat.st_mode));
  assert(socketStat.st_uid == ::geteuid());
  assert((socketStat.st_mode & 0077) == 0);

  const std::string params =
      "{\"type\":\"params\",\"seq\":7,\"senderId\":\"server-test\"}\n";
  const std::string heartbeat =
      "{\"type\":\"heartbeat\",\"seq\":8,\"senderId\":\"server-test\"}\n";
  assert(::send(client, params.data(), 9u, 0) == 9);
  assert(::send(client, params.data() + 9u, params.size() - 9u, 0) ==
         static_cast<ssize_t>(params.size() - 9u));
  assert(::send(client, heartbeat.data(), heartbeat.size(), 0) ==
         static_cast<ssize_t>(heartbeat.size()));

  std::string ack;
  assert(receiveLine(client, &ack));
  assert(ack.find("\"type\":\"heartbeat_ack\"") != std::string::npos);
  assert(ack.find("\"seq\":8") != std::string::npos);
  assert(waitForParams(&reducer, 7u));
  assert(context.calls.load(std::memory_order_relaxed) == 1u);
  assert(context.snapshotConnections.load(std::memory_order_relaxed) >= 1u);

  const std::string oversized(200u, 'x');
  assert(::send(client, oversized.data(), oversized.size(), 0) ==
         static_cast<ssize_t>(oversized.size()));
  const char newline = '\n';
  assert(::send(client, &newline, 1u, 0) == 1);
  ::close(client);
  assert(waitForServerSnapshot(
      &server, [](const ViewerCommandServerSnapshot& value) {
        return value.oversizedLines >= 1u && value.activeConnections == 0u;
      }));

  const auto beforeIncomplete = server.snapshot();
  int incomplete = -1;
  assert(connectSocket(path, &incomplete));
  const std::string partial =
      "{\"type\":\"heartbeat\",\"seq\":99,\"senderId\":\"partial\"}";
  assert(::send(incomplete, partial.data(), partial.size(), 0) ==
         static_cast<ssize_t>(partial.size()));
  ::close(incomplete);
  assert(waitForServerSnapshot(
      &server,
      [beforeIncomplete](const ViewerCommandServerSnapshot& value) {
        return value.acceptedConnections >
                   beforeIncomplete.acceptedConnections &&
               value.closedConnections > beforeIncomplete.closedConnections &&
               value.incompleteLines > beforeIncomplete.incompleteLines &&
               value.activeConnections == 0u;
      }));
  const auto beforeStop = server.snapshot();
  assert(beforeStop.submittedLines >= 2u);
  assert(beforeStop.oversizedLines >= 1u);
  assert(beforeStop.incompleteLines >= 1u);
  assert(beforeStop.heartbeatCount == 1u);
  assert(beforeStop.heartbeatAckCount == 1u);

  const auto beforeIdle = server.snapshot();
  int idle = -1;
  assert(connectSocket(path, &idle));
  assert(waitForServerSnapshot(
      &server, [beforeIdle](const ViewerCommandServerSnapshot& value) {
        return value.acceptedConnections > beforeIdle.acceptedConnections &&
               value.activeConnections >= 1u;
      }));
  assert(waitForServerSnapshot(
      &server, [beforeIdle](const ViewerCommandServerSnapshot& value) {
        return value.closedConnections > beforeIdle.closedConnections &&
               value.activeConnections == 0u;
      }));
  const auto afterIdle = server.snapshot();
  assert(afterIdle.activeConnections == 0u);
  assert(afterIdle.acceptedConnections > beforeIdle.acceptedConnections);
  assert(afterIdle.closedConnections > beforeIdle.closedConnections);
  if (idle >= 0) ::close(idle);

  const auto stopStart = std::chrono::steady_clock::now();
  assert(server.stopAndJoin());
  const auto stopElapsed = std::chrono::steady_clock::now() - stopStart;
  assert(stopElapsed < std::chrono::seconds(1));
  assert(server.state() == ViewerCommandServerState::Stopped);
  assert(!server.joinable());
  assert(::lstat(path.c_str(), &socketStat) != 0);
  assert(server.join());

  // A non-socket collision is never removed or replaced.
  const std::string collision = path + ".collision";
  {
    FILE* file = std::fopen(collision.c_str(), "wb");
    assert(file != nullptr);
    assert(std::fputs("keep", file) >= 0);
    std::fclose(file);
  }
  ViewerCommandServerOptions collisionOptions{};
  collisionOptions.endpoint = resolveViewerIpcEndpoint(collision);
  ViewerCommandServer collisionServer(&reducer, collisionOptions);
  assert(!collisionServer.start());
  assert(collisionServer.state() == ViewerCommandServerState::Failed);
  assert(::access(collision.c_str(), F_OK) == 0);
  (void)::unlink(collision.c_str());
  return 0;
#endif
}
