#pragma once

#include "ChromaspaceMetal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace ChromaspaceResidentSource {

constexpr std::size_t kMaxEventsPerTick = 16u;
constexpr uint32_t kReconnectInitialDelayMilliseconds = 250u;
constexpr uint32_t kReconnectMaximumDelayMilliseconds = 8000u;
constexpr uint32_t kReconnectMaximumExponent = 5u;
constexpr uint32_t kFrameDrainTimeoutMilliseconds = 5000u;
constexpr std::size_t kMaxDiagnosticBytes = 256u;

enum class ClientHealth : uint8_t {
  Stopped,
  Registering,
  BindingRoute,
  Ready,
  Failed,
};

enum class SessionHealth : uint8_t {
  Stopped,
  Unavailable,
  Starting,
  Ready,
  Failed,
};

enum class EventKind : uint8_t {
  SourceCleared,
  SourceClearFailed,
  DrainStarted,
  DrainFailed,
  ClientDestroyed,
  ClientCreated,
  ClientStarted,
  HealthChanged,
  RetryScheduled,
  RetryAttempt,
  SourceActivated,
  DeviceUnavailable,
  Shutdown,
};

struct ClientSnapshot {
  ClientHealth health = ClientHealth::Stopped;
  uint64_t viewerGeneration = 0u;
  uint64_t lastObservedSequence = 0u;
  std::size_t liveKeyCount = 0u;
  bool hasActiveSource = false;
  ChromaspaceMetal::ImportedSourceTexture activeSource{};
  std::string diagnostic;
};

struct ResidentSourceSnapshot {
  SessionHealth health = SessionHealth::Stopped;
  std::string senderId;
  uint64_t viewerGeneration = 0u;
  uint64_t lastObservedSequence = 0u;
  std::size_t liveKeyCount = 0u;
  bool hasActiveSource = false;
  ChromaspaceMetal::ImportedSourceTexture activeSource{};
  std::string diagnostic;
};

struct SessionEvent {
  EventKind kind = EventKind::HealthChanged;
  SessionHealth health = SessionHealth::Stopped;
  std::string senderId;
  std::string diagnostic;
  uint64_t viewerGeneration = 0u;
  uint64_t lastObservedSequence = 0u;
  std::size_t liveKeyCount = 0u;
  uint64_t sourceId = 0u;
  uint64_t sequence = 0u;
  uint32_t slotIndex = 0u;
  uint64_t slotGeneration = 0u;
  int sourceWidth = 0;
  int sourceHeight = 0;
  uint32_t retryDelayMilliseconds = 0u;
};

struct TickResult {
  ResidentSourceSnapshot snapshot{};
  std::array<SessionEvent, kMaxEventsPerTick> events{};
  std::size_t eventCount = 0u;
};

struct ShutdownResult {
  bool alreadyShutdown = false;
  bool drainAttempted = false;
  bool drainSucceeded = true;
  bool clientDestroyed = false;
  std::array<char, kMaxDiagnosticBytes> drainDiagnostic{};
};

// This adapter is the only seam that knows how a platform creates and polls
// the resident source client. The session owns the returned opaque handle and
// never exposes it to the viewer event loop.
struct ClientAdapter {
  void* context = nullptr;
  void* (*create)(void* context,
                  const std::string& senderId,
                  uint64_t deviceRegistryId,
                  std::string* error) noexcept = nullptr;
  bool (*start)(void* context,
                void* client,
                std::string* error) noexcept = nullptr;
  bool (*clear)(void* context,
                void* client,
                std::string* error) noexcept = nullptr;
  bool (*snapshot)(void* context,
                   const void* client,
                   ClientSnapshot* snapshot,
                   std::string* error) noexcept = nullptr;
  void (*destroy)(void* context, void* client) noexcept = nullptr;
};

// The frame executor is injected through this narrow callback so the
// portable session policy can be tested without a Metal object. Production
// supplies a wrapper around FrameExecutor::drain.
struct DrainAdapter {
  void* context = nullptr;
  bool (*drain)(void* context,
                uint32_t timeoutMilliseconds,
                std::string* error) noexcept = nullptr;
};

const ClientAdapter* defaultResidentSourceClientAdapter() noexcept;

class ResidentSourceSession final {
 public:
  explicit ResidentSourceSession(
      const ClientAdapter* clientAdapter = nullptr,
      const DrainAdapter* drainAdapter = nullptr) noexcept;
  ~ResidentSourceSession();

  ResidentSourceSession(const ResidentSourceSession&) = delete;
  ResidentSourceSession& operator=(const ResidentSourceSession&) = delete;

  bool requestSender(const std::string& senderId,
                     uint64_t deviceRegistryId,
                     int64_t nowMilliseconds,
                     std::string* error = nullptr);
  bool requestClear(const std::string& senderId,
                    std::string* error = nullptr);
  TickResult tick(int64_t nowMilliseconds);
  ShutdownResult shutdown() noexcept;

  const ResidentSourceSnapshot& snapshot() const noexcept {
    return snapshot_;
  }
  bool shutdownRequested() const noexcept { return shutdown_; }

 private:
  struct SourceIdentity {
    uint64_t sourceId = 0u;
    std::string senderId;
    uint64_t sequence = 0u;
    uint32_t slotIndex = 0u;
    uint64_t slotGeneration = 0u;

    bool operator==(const SourceIdentity& other) const {
      return sourceId == other.sourceId && senderId == other.senderId &&
             sequence == other.sequence && slotIndex == other.slotIndex &&
             slotGeneration == other.slotGeneration;
    }
  };

  static SessionHealth sessionHealthForClient(ClientHealth health) noexcept;
  static uint32_t reconnectDelayForExponent(uint32_t exponent) noexcept;
  static bool validClientAdapter(const ClientAdapter& adapter) noexcept;
  static bool validDrainAdapter(const DrainAdapter& adapter) noexcept;

  void enqueueEvent(SessionEvent event) noexcept;
  void setHealth(SessionHealth health, const std::string& diagnostic);
  void clearVisibleSource(const std::string& reason) noexcept;
  void scheduleRetry(int64_t nowMilliseconds,
                     const std::string& diagnostic);
  void processClientSnapshot(int64_t nowMilliseconds);
  void processPendingBind(int64_t nowMilliseconds);
  void tearDownClient(const std::string& reason,
                      ShutdownResult* shutdownResult) noexcept;
  void resetAfterShutdown() noexcept;
  TickResult takeTickResult() noexcept;

  ClientAdapter clientAdapter_{};
  DrainAdapter drainAdapter_{};
  void* client_ = nullptr;
  std::string clientSender_;
  std::string desiredSender_;
  uint64_t desiredDeviceRegistryId_ = 0u;
  bool bindPending_ = false;
  bool clearPending_ = false;
  std::string clearSender_;
  int64_t nextRetryMilliseconds_ = 0;
  uint32_t reconnectExponent_ = 0u;
  bool retryScheduled_ = false;
  bool unavailableReported_ = false;
  bool shutdown_ = false;
  SessionHealth lastHealth_ = SessionHealth::Stopped;
  std::string lastHealthDiagnostic_;
  bool activeIdentityValid_ = false;
  SourceIdentity activeIdentity_{};
  ResidentSourceSnapshot snapshot_{};
  std::array<SessionEvent, kMaxEventsPerTick> pendingEvents_{};
  std::size_t pendingEventCount_ = 0u;
};

}  // namespace ChromaspaceResidentSource
