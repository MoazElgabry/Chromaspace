#include "ChromaspaceResidentSourceSession.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace ChromaspaceResidentSource {
namespace {

void setError(std::string* error, const char* message) {
  if (error) *error = message != nullptr ? message : "resident-source-error";
}

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "resident-source-error";
  } catch (...) {
  }
}

void copyDiagnosticNoThrow(std::array<char, kMaxDiagnosticBytes>* destination,
                           const std::string& source) noexcept {
  if (destination == nullptr) return;
  destination->fill('\0');
  const std::size_t count =
      std::min(source.size(), destination->size() - 1u);
  if (count != 0u) {
    std::memcpy(destination->data(), source.data(), count);
  }
}

#if !defined(CHROMASPACE_METAL_EXTERNAL_DEFAULT_BACKENDS)
void* unavailableCreate(void*,
                        const std::string&,
                        uint64_t,
                        std::string* error) noexcept {
  setErrorNoThrow(error, "resident-source-client-unavailable");
  return nullptr;
}

bool unavailableStart(void*, void*, std::string* error) noexcept {
  setErrorNoThrow(error, "resident-source-client-unavailable");
  return false;
}

bool unavailableClear(void*, void*, std::string* error) noexcept {
  setErrorNoThrow(error, "resident-source-client-unavailable");
  return false;
}

bool unavailableSnapshot(void*,
                         const void*,
                         ClientSnapshot* snapshot,
                         std::string* error) noexcept {
  if (snapshot != nullptr) {
    snapshot->health = ClientHealth::Stopped;
    snapshot->viewerGeneration = 0u;
    snapshot->lastObservedSequence = 0u;
    snapshot->liveKeyCount = 0u;
    snapshot->hasActiveSource = false;
    try {
      snapshot->activeSource = ChromaspaceMetal::ImportedSourceTexture{};
    } catch (...) {
    }
    try {
      snapshot->diagnostic.clear();
    } catch (...) {
    }
  }
  setErrorNoThrow(error, "resident-source-client-unavailable");
  return false;
}

void unavailableDestroy(void*, void*) noexcept {}

const ClientAdapter kUnavailableClientAdapter{
    nullptr,
    unavailableCreate,
    unavailableStart,
    unavailableClear,
    unavailableSnapshot,
    unavailableDestroy};
#endif

}  // namespace

#if !defined(CHROMASPACE_METAL_EXTERNAL_DEFAULT_BACKENDS)
const ClientAdapter* defaultResidentSourceClientAdapter() noexcept {
  return &kUnavailableClientAdapter;
}
#endif

SessionHealth ResidentSourceSession::sessionHealthForClient(
    ClientHealth health) noexcept {
  switch (health) {
    case ClientHealth::Stopped: return SessionHealth::Starting;
    case ClientHealth::Registering:
    case ClientHealth::BindingRoute: return SessionHealth::Starting;
    case ClientHealth::Ready: return SessionHealth::Ready;
    case ClientHealth::Failed: return SessionHealth::Failed;
    default: return SessionHealth::Failed;
  }
}

uint32_t ResidentSourceSession::reconnectDelayForExponent(
    uint32_t exponent) noexcept {
  const uint32_t clamped =
      std::min(exponent, kReconnectMaximumExponent);
  const uint64_t delay =
      static_cast<uint64_t>(kReconnectInitialDelayMilliseconds) << clamped;
  return static_cast<uint32_t>(std::min<uint64_t>(
      delay, kReconnectMaximumDelayMilliseconds));
}

bool ResidentSourceSession::validClientAdapter(
    const ClientAdapter& adapter) noexcept {
  return adapter.create != nullptr && adapter.start != nullptr &&
         adapter.clear != nullptr && adapter.snapshot != nullptr &&
         adapter.destroy != nullptr;
}

bool ResidentSourceSession::validDrainAdapter(
    const DrainAdapter& adapter) noexcept {
  return adapter.drain != nullptr;
}

ResidentSourceSession::ResidentSourceSession(
    const ClientAdapter* clientAdapter,
    const DrainAdapter* drainAdapter) noexcept {
  const ClientAdapter* selectedClient =
      clientAdapter != nullptr ? clientAdapter
                               : defaultResidentSourceClientAdapter();
  if (selectedClient != nullptr) clientAdapter_ = *selectedClient;
  if (drainAdapter != nullptr) drainAdapter_ = *drainAdapter;
  snapshot_.health = SessionHealth::Stopped;
  lastHealth_ = SessionHealth::Stopped;
}

ResidentSourceSession::~ResidentSourceSession() { (void)shutdown(); }

bool ResidentSourceSession::requestSender(const std::string& senderId,
                                           uint64_t deviceRegistryId,
                                           int64_t nowMilliseconds,
                                           std::string* error) {
  if (error) error->clear();
  if (shutdown_) {
    setError(error, "resident-source-session-shutdown");
    return false;
  }
  if (senderId.empty()) {
    setError(error, "resident-source-sender-empty");
    return false;
  }
  if (!validClientAdapter(clientAdapter_)) {
    setError(error, "resident-source-client-adapter-invalid");
    return false;
  }
  if (desiredSender_ == senderId &&
      desiredDeviceRegistryId_ == deviceRegistryId && !bindPending_) {
    return true;
  }
  if (desiredSender_ != senderId ||
      desiredDeviceRegistryId_ != deviceRegistryId) {
    if (snapshot_.hasActiveSource || activeIdentityValid_) {
      clearVisibleSource("sender-switch");
    }
    desiredSender_ = senderId;
    desiredDeviceRegistryId_ = deviceRegistryId;
    snapshot_.senderId = desiredSender_;
    unavailableReported_ = false;
  }
  bindPending_ = true;
  retryScheduled_ = false;
  nextRetryMilliseconds_ = nowMilliseconds;
  return true;
}

bool ResidentSourceSession::requestClear(const std::string& senderId,
                                          std::string* error) {
  if (error) error->clear();
  if (shutdown_) {
    setError(error, "resident-source-session-shutdown");
    return false;
  }
  if (senderId.empty()) {
    setError(error, "resident-source-sender-empty");
    return false;
  }
  if (senderId != desiredSender_) return true;
  clearVisibleSource("clear-request");
  clearSender_ = senderId;
  clearPending_ = true;
  return true;
}

void ResidentSourceSession::enqueueEvent(SessionEvent event) noexcept {
  if (pendingEventCount_ >= pendingEvents_.size()) return;
  try {
    pendingEvents_[pendingEventCount_] = std::move(event);
    ++pendingEventCount_;
  } catch (...) {
  }
}

void ResidentSourceSession::setHealth(SessionHealth health,
                                      const std::string& diagnostic) {
  snapshot_.health = health;
  snapshot_.diagnostic = diagnostic;
  if (lastHealth_ == health && lastHealthDiagnostic_ == diagnostic) return;
  lastHealth_ = health;
  lastHealthDiagnostic_ = diagnostic;
  SessionEvent event{};
  event.kind = EventKind::HealthChanged;
  event.health = health;
  event.senderId = desiredSender_;
  event.diagnostic = diagnostic;
  event.viewerGeneration = snapshot_.viewerGeneration;
  event.lastObservedSequence = snapshot_.lastObservedSequence;
  event.liveKeyCount = snapshot_.liveKeyCount;
  enqueueEvent(std::move(event));
}

void ResidentSourceSession::clearVisibleSource(
    const std::string& reason) noexcept {
  const bool hadSource = snapshot_.hasActiveSource || activeIdentityValid_;
  snapshot_.hasActiveSource = false;
  activeIdentityValid_ = false;
  activeIdentity_ = SourceIdentity{};
  try {
    snapshot_.activeSource = ChromaspaceMetal::ImportedSourceTexture{};
  } catch (...) {
  }
  if (!hadSource) return;
  try {
    SessionEvent event{};
    event.kind = EventKind::SourceCleared;
    event.senderId = desiredSender_;
    event.diagnostic = reason;
    enqueueEvent(std::move(event));
  } catch (...) {
  }
}

void ResidentSourceSession::scheduleRetry(int64_t nowMilliseconds,
                                           const std::string& diagnostic) {
  if (retryScheduled_) return;
  const uint32_t delay = reconnectDelayForExponent(reconnectExponent_);
  reconnectExponent_ = std::min(
      reconnectExponent_ + 1u, kReconnectMaximumExponent);
  nextRetryMilliseconds_ =
      nowMilliseconds + static_cast<int64_t>(delay);
  retryScheduled_ = true;
  SessionEvent event{};
  event.kind = EventKind::RetryScheduled;
  event.senderId = desiredSender_;
  event.diagnostic = diagnostic;
  event.retryDelayMilliseconds = delay;
  enqueueEvent(std::move(event));
}

void ResidentSourceSession::processClientSnapshot(
    int64_t nowMilliseconds) {
  if (client_ == nullptr) return;
  ClientSnapshot clientSnapshot{};
  std::string snapshotError;
  if (!clientAdapter_.snapshot(clientAdapter_.context, client_,
                              &clientSnapshot, &snapshotError)) {
    clearVisibleSource("snapshot-failed");
    const std::string diagnostic = snapshotError.empty()
                                       ? "resident-source-snapshot-failed"
                                       : snapshotError;
    setHealth(SessionHealth::Failed, diagnostic);
    scheduleRetry(nowMilliseconds, diagnostic);
    return;
  }

  snapshot_.viewerGeneration = clientSnapshot.viewerGeneration;
  snapshot_.lastObservedSequence = clientSnapshot.lastObservedSequence;
  snapshot_.liveKeyCount = clientSnapshot.liveKeyCount;
  const SessionHealth nextHealth =
      sessionHealthForClient(clientSnapshot.health);
  setHealth(nextHealth, clientSnapshot.diagnostic);
  if (nextHealth == SessionHealth::Ready) {
    reconnectExponent_ = 0u;
    retryScheduled_ = false;
    nextRetryMilliseconds_ = 0;
  } else if (nextHealth == SessionHealth::Failed) {
    clearVisibleSource("client-failed");
    scheduleRetry(nowMilliseconds,
                  clientSnapshot.diagnostic.empty()
                      ? "resident-source-client-failed"
                      : clientSnapshot.diagnostic);
    return;
  }

  SourceIdentity nextIdentity{};
  const bool sourceVisible =
      clientSnapshot.hasActiveSource &&
      clientSnapshot.activeSource.senderId == desiredSender_ &&
      clientSnapshot.activeSource.sourceId != 0u &&
      !clientSnapshot.activeSource.senderId.empty();
  if (sourceVisible) {
    nextIdentity.sourceId = clientSnapshot.activeSource.sourceId;
    nextIdentity.senderId = clientSnapshot.activeSource.senderId;
    nextIdentity.sequence = clientSnapshot.activeSource.sequence;
    nextIdentity.slotIndex = clientSnapshot.activeSource.slotIndex;
    nextIdentity.slotGeneration = clientSnapshot.activeSource.slotGeneration;
  }
  if (!sourceVisible) {
    if (snapshot_.hasActiveSource || activeIdentityValid_) {
      clearVisibleSource("source-not-active");
    }
    return;
  }
  snapshot_.hasActiveSource = true;
  snapshot_.activeSource = clientSnapshot.activeSource;
  if (!activeIdentityValid_ || !(activeIdentity_ == nextIdentity)) {
    activeIdentity_ = nextIdentity;
    activeIdentityValid_ = true;
    SessionEvent event{};
    event.kind = EventKind::SourceActivated;
    event.senderId = nextIdentity.senderId;
    event.sourceId = nextIdentity.sourceId;
    event.sequence = nextIdentity.sequence;
    event.slotIndex = nextIdentity.slotIndex;
    event.slotGeneration = nextIdentity.slotGeneration;
    event.sourceWidth = clientSnapshot.activeSource.width;
    event.sourceHeight = clientSnapshot.activeSource.height;
    enqueueEvent(std::move(event));
  }
}

void ResidentSourceSession::tearDownClient(
    const std::string& reason,
    ShutdownResult* shutdownResult) noexcept {
  if (client_ == nullptr) return;
  std::string retiringSender;
  try {
    retiringSender = clientSender_;
  } catch (...) {
  }
  clearVisibleSource(reason);

  std::string clearError;
  const bool cleared =
      clientAdapter_.clear(clientAdapter_.context, client_, &clearError);
  if (!cleared && !clearError.empty()) {
    try {
      SessionEvent clearEvent{};
      clearEvent.kind = EventKind::SourceClearFailed;
      clearEvent.senderId = retiringSender;
      clearEvent.diagnostic = clearError;
      enqueueEvent(std::move(clearEvent));
    } catch (...) {
    }
  }

  try {
    SessionEvent drainStarted{};
    drainStarted.kind = EventKind::DrainStarted;
    drainStarted.senderId = retiringSender;
    drainStarted.diagnostic = reason;
    enqueueEvent(std::move(drainStarted));
  } catch (...) {
  }

  std::string drainError;
  bool drained = false;
  try {
    if (validDrainAdapter(drainAdapter_)) {
      drained = drainAdapter_.drain(drainAdapter_.context,
                                    kFrameDrainTimeoutMilliseconds,
                                    &drainError);
    } else {
      drainError = "resident-source-drain-adapter-invalid";
    }
  } catch (...) {
    drainError = "resident-source-drain-exception";
    drained = false;
  }
  if (shutdownResult != nullptr) {
    shutdownResult->drainAttempted = true;
    shutdownResult->drainSucceeded = drained;
    if (!drained) {
      copyDiagnosticNoThrow(&shutdownResult->drainDiagnostic, drainError);
    }
  }
  if (!drained) {
    try {
      SessionEvent drainFailed{};
      drainFailed.kind = EventKind::DrainFailed;
      drainFailed.senderId = retiringSender;
      drainFailed.diagnostic = drainError.empty()
                                   ? "resident-source-drain-failed"
                                   : drainError;
      enqueueEvent(std::move(drainFailed));
    } catch (...) {
    }
  }

  clientAdapter_.destroy(clientAdapter_.context, client_);
  client_ = nullptr;
  clientSender_.clear();
  if (shutdownResult != nullptr) shutdownResult->clientDestroyed = true;
  try {
    SessionEvent destroyed{};
    destroyed.kind = EventKind::ClientDestroyed;
    destroyed.senderId = retiringSender;
    destroyed.diagnostic = reason;
    enqueueEvent(std::move(destroyed));
  } catch (...) {
  }
}

void ResidentSourceSession::processPendingBind(int64_t nowMilliseconds) {
  if (!bindPending_) return;
  bindPending_ = false;

  if (client_ != nullptr) {
    const char* reason = clientSender_ == desiredSender_ ? "rebind"
                                                          : "sender-switch";
    tearDownClient(reason, nullptr);
  }

  if (desiredDeviceRegistryId_ == 0u) {
    setHealth(SessionHealth::Unavailable,
              "missing-device-registry-id");
    if (!unavailableReported_) {
      unavailableReported_ = true;
      SessionEvent event{};
      event.kind = EventKind::DeviceUnavailable;
      event.senderId = desiredSender_;
      event.diagnostic = "missing-device-registry-id";
      enqueueEvent(std::move(event));
    }
    return;
  }
  unavailableReported_ = false;

  std::string createError;
  client_ = clientAdapter_.create(clientAdapter_.context, desiredSender_,
                                  desiredDeviceRegistryId_, &createError);
  if (client_ == nullptr) {
    const std::string diagnostic = createError.empty()
                                       ? "resident-source-client-create-failed"
                                       : createError;
    setHealth(SessionHealth::Failed, diagnostic);
    scheduleRetry(nowMilliseconds, diagnostic);
    return;
  }
  clientSender_ = desiredSender_;
  SessionEvent created{};
  created.kind = EventKind::ClientCreated;
  created.senderId = clientSender_;
  enqueueEvent(std::move(created));

  std::string startError;
  if (!clientAdapter_.start(clientAdapter_.context, client_, &startError)) {
    const std::string diagnostic = startError.empty()
                                       ? "resident-source-client-start-failed"
                                       : startError;
    tearDownClient("client-start-failed", nullptr);
    setHealth(SessionHealth::Failed, diagnostic);
    scheduleRetry(nowMilliseconds, diagnostic);
    return;
  }
  SessionEvent started{};
  started.kind = EventKind::ClientStarted;
  started.senderId = clientSender_;
  enqueueEvent(std::move(started));
  processClientSnapshot(nowMilliseconds);
}

TickResult ResidentSourceSession::takeTickResult() noexcept {
  TickResult result{};
  try {
    result.snapshot = snapshot_;
    result.eventCount = std::min(pendingEventCount_, result.events.size());
    for (std::size_t index = 0; index < result.eventCount; ++index) {
      result.events[index] = std::move(pendingEvents_[index]);
    }
  } catch (...) {
    result.eventCount = 0u;
    result.snapshot = ResidentSourceSnapshot{};
  }
  pendingEventCount_ = 0u;
  return result;
}

TickResult ResidentSourceSession::tick(int64_t nowMilliseconds) {
  if (!shutdown_) {
    auto applyMatchingClear = [&]() {
      if (!clearPending_ || client_ == nullptr ||
          clientSender_ != clearSender_) {
        return;
      }
      clearPending_ = false;
      std::string clearError;
      if (!clientAdapter_.clear(clientAdapter_.context, client_,
                                &clearError) &&
          !clearError.empty()) {
        SessionEvent event{};
        event.kind = EventKind::SourceClearFailed;
        event.senderId = clearSender_;
        event.diagnostic = clearError;
        enqueueEvent(std::move(event));
      }
      clearSender_.clear();
    };

    // A clear for the currently bound sender is applied before a same-sender
    // retry. A clear for a newly requested sender is deferred until that
    // sender has been created; it must never clear the old client by mistake.
    applyMatchingClear();

    if (bindPending_) {
      processPendingBind(nowMilliseconds);
      applyMatchingClear();
    } else if (client_ != nullptr) {
      if (retryScheduled_ && nowMilliseconds >= nextRetryMilliseconds_) {
        retryScheduled_ = false;
        bindPending_ = true;
        SessionEvent attempt{};
        attempt.kind = EventKind::RetryAttempt;
        attempt.senderId = desiredSender_;
        enqueueEvent(std::move(attempt));
        processPendingBind(nowMilliseconds);
      } else {
        processClientSnapshot(nowMilliseconds);
      }
    } else if (retryScheduled_ && nowMilliseconds >= nextRetryMilliseconds_) {
      retryScheduled_ = false;
      bindPending_ = true;
      SessionEvent attempt{};
      attempt.kind = EventKind::RetryAttempt;
      attempt.senderId = desiredSender_;
      enqueueEvent(std::move(attempt));
      processPendingBind(nowMilliseconds);
      applyMatchingClear();
    }
    if (clearPending_ && client_ == nullptr &&
        clearSender_ != desiredSender_) {
      clearPending_ = false;
      clearSender_.clear();
    }
  }
  return takeTickResult();
}

void ResidentSourceSession::resetAfterShutdown() noexcept {
  client_ = nullptr;
  clientSender_.clear();
  desiredSender_.clear();
  desiredDeviceRegistryId_ = 0u;
  bindPending_ = false;
  clearPending_ = false;
  clearSender_.clear();
  nextRetryMilliseconds_ = 0;
  reconnectExponent_ = 0u;
  retryScheduled_ = false;
  unavailableReported_ = false;
  snapshot_.senderId.clear();
  snapshot_.viewerGeneration = 0u;
  snapshot_.lastObservedSequence = 0u;
  snapshot_.liveKeyCount = 0u;
  snapshot_.diagnostic.clear();
  snapshot_.health = SessionHealth::Stopped;
  lastHealth_ = SessionHealth::Stopped;
  lastHealthDiagnostic_.clear();
  activeIdentityValid_ = false;
  activeIdentity_ = SourceIdentity{};
  try {
    snapshot_.activeSource = ChromaspaceMetal::ImportedSourceTexture{};
  } catch (...) {
  }
}

ShutdownResult ResidentSourceSession::shutdown() noexcept {
  ShutdownResult result{};
  if (shutdown_) {
    result.alreadyShutdown = true;
    return result;
  }
  shutdown_ = true;
  clearVisibleSource("shutdown");
  tearDownClient("shutdown", &result);
  resetAfterShutdown();
  try {
    SessionEvent event{};
    event.kind = EventKind::Shutdown;
    event.diagnostic = result.drainSucceeded
                           ? "resident-source-session-shutdown"
                           : "resident-source-session-drain-failed";
    enqueueEvent(std::move(event));
  } catch (...) {
  }
  return result;
}

}  // namespace ChromaspaceResidentSource
