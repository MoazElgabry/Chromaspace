#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "ChromaspaceMetalFrameExecutor.h"
#include "ChromaspaceMetalQualificationCampaign.h"

// Portable, qualification-only fault injection around the frame-executor
// callback table.  This seam deliberately has no platform, process, clock,
// thread, or transport dependencies.  The native qualification adapter owns
// the wrapper for the complete lifetime of the FrameExecutor that consumes
// backend().
namespace ChromaspaceMetalQualification {

enum class RuntimeObservation : uint8_t {
  RetryLater = 0,
  SuspendUntilVisible,
  RuntimeRecreated,
  Presented,
  TerminalFailure,
  Count,
};

enum class RecoveryPhase : uint8_t {
  Idle = 0,
  Armed,
  DrawableRetryPending,
  DrawablePresentedPending,
  PriorRecreationPending,
  PriorPresentedPending,
  Failed,
};

struct CompletionExpectation final {
  uint32_t expectedFaultCount = 2u;
  bool requireBothKinds = true;
};

struct FaultSnapshot final {
  std::array<uint32_t, kActionKindCount> armedByKind{};
  std::array<uint32_t, kActionKindCount> firedByKind{};
  std::array<uint32_t, kActionKindCount> recoveredByKind{};
  bool ready = false;
  bool failed = false;
  bool completed = false;
  uint32_t armedCount = 0u;
  uint32_t firedCount = 0u;
  uint32_t recoveredCount = 0u;
  uint64_t lastAcceptedOrdinal = 0u;
  uint64_t activeOrdinal = 0u;
  ActionKind activeKind = ActionKind::None;
  RecoveryPhase recoveryPhase = RecoveryPhase::Idle;
};

class FaultBackend final {
 public:
  explicit FaultBackend(
      const ChromaspaceMetalFrameExecutor::FrameExecutorBackend* base) noexcept;
  explicit FaultBackend(
      const ChromaspaceMetalFrameExecutor::FrameExecutorBackend& base) noexcept
      : FaultBackend(&base) {}
  ~FaultBackend() = default;

  FaultBackend(const FaultBackend&) = delete;
  FaultBackend& operator=(const FaultBackend&) = delete;
  FaultBackend(FaultBackend&&) = delete;
  FaultBackend& operator=(FaultBackend&&) = delete;

  // The returned table remains valid until this object is destroyed.  The
  // object must outlive every FrameExecutor constructed with this pointer.
  const ChromaspaceMetalFrameExecutor::FrameExecutorBackend* backend()
      const noexcept {
    return &wrappedBackend_;
  }

  bool ready() const noexcept { return ready_; }
  bool failed() const noexcept { return failed_; }
  bool completed() const noexcept { return completed_; }
  const char* diagnostic() const noexcept { return diagnostic_; }

  bool arm(const Action& action, std::string* diagnostic = nullptr) noexcept;

  // Ordinal zero is reserved for ordinary observations with no active fault.
  // While a fault is active, the observation must carry that exact action's
  // ordinal.  A wrong, stale, or terminal observation permanently fails the
  // wrapper.
  bool observe(RuntimeObservation observation,
               uint64_t ordinal,
               std::string* diagnostic = nullptr) noexcept;

  bool finish(const CompletionExpectation& expectation,
              std::string* diagnostic = nullptr) noexcept;
  bool finish(std::string* diagnostic = nullptr) noexcept;

  FaultSnapshot snapshot() const noexcept;

 private:
  static bool createCallback(void* context,
                             void* nativeWindow,
                             int drawableWidth,
                             int drawableHeight,
                             float contentsScale,
                             ChromaspaceMetalFrameExecutor::FrameCompositorState*
                                 outState,
                             std::string* error) noexcept;
  static bool resizeCallback(void* context,
                             uint64_t compositorId,
                             int drawableWidth,
                             int drawableHeight,
                             float contentsScale,
                             std::string* error) noexcept;
  static bool drainCallback(void* context,
                            uint64_t compositorId,
                            uint32_t timeoutMilliseconds,
                            std::string* error) noexcept;
  static void destroyCallback(void* context,
                              uint64_t compositorHandle) noexcept;
  static bool beginCallback(
      void* context,
      uint64_t compositorId,
      ChromaspaceMetal::FrameSubmission* outSubmission,
      std::string* error,
      ChromaspaceMetalFrameExecutor::FrameFailure* failure) noexcept;
  static bool submitCallback(
      void* context,
      ChromaspaceMetal::FrameSubmission* submission,
      const ChromaspaceMetalFrameExecutor::FrameBatch& batch,
      std::string* error,
      ChromaspaceMetalFrameExecutor::FrameFailure* failure) noexcept;
  static void abandonCallback(
      void* context,
      ChromaspaceMetal::FrameSubmission* submission) noexcept;
  static bool transientMemoryStatsCallback(
      void* context,
      uint64_t compositorId,
      ChromaspaceMetal::FrameTransientMemoryStats* outStats) noexcept;
  static bool completionStatsCallback(
      void* context,
      uint64_t compositorId,
      ChromaspaceMetal::FrameCompletionStats* outStats) noexcept;

  bool forwardCreate(void* nativeWindow,
                     int drawableWidth,
                     int drawableHeight,
                     float contentsScale,
                     ChromaspaceMetalFrameExecutor::FrameCompositorState*
                         outState,
                     std::string* error) noexcept;
  bool forwardResize(uint64_t compositorId,
                     int drawableWidth,
                     int drawableHeight,
                     float contentsScale,
                     std::string* error) noexcept;
  bool forwardDrain(uint64_t compositorId,
                    uint32_t timeoutMilliseconds,
                    std::string* error) noexcept;
  void forwardDestroy(uint64_t compositorHandle) noexcept;
  bool forwardBegin(
      uint64_t compositorId,
      ChromaspaceMetal::FrameSubmission* outSubmission,
      std::string* error,
      ChromaspaceMetalFrameExecutor::FrameFailure* failure) noexcept;
  bool forwardSubmit(
      ChromaspaceMetal::FrameSubmission* submission,
      const ChromaspaceMetalFrameExecutor::FrameBatch& batch,
      std::string* error,
      ChromaspaceMetalFrameExecutor::FrameFailure* failure) noexcept;
  void forwardAbandon(ChromaspaceMetal::FrameSubmission* submission) noexcept;
  bool forwardTransientMemoryStats(
      uint64_t compositorId,
      ChromaspaceMetal::FrameTransientMemoryStats* outStats) noexcept;
  bool forwardCompletionStats(
      uint64_t compositorId,
      ChromaspaceMetal::FrameCompletionStats* outStats) noexcept;

  bool validObservation(RuntimeObservation observation) const noexcept;
  bool validFaultKind(ActionKind kind) const noexcept;
  bool failPermanently(const char* diagnostic,
                       std::string* output) noexcept;
  bool finishWithExpectation(const CompletionExpectation& expectation,
                             std::string* diagnostic) noexcept;

  ChromaspaceMetalFrameExecutor::FrameExecutorBackend baseBackend_{};
  ChromaspaceMetalFrameExecutor::FrameExecutorBackend wrappedBackend_{};
  bool ready_ = false;
  bool failed_ = false;
  bool completed_ = false;
  RecoveryPhase recoveryPhase_ = RecoveryPhase::Idle;
  Action activeAction_{};
  uint64_t lastAcceptedOrdinal_ = 0u;
  uint32_t armedCount_ = 0u;
  uint32_t firedCount_ = 0u;
  uint32_t recoveredCount_ = 0u;
  std::array<uint32_t, kActionKindCount> armedByKind_{};
  std::array<uint32_t, kActionKindCount> firedByKind_{};
  std::array<uint32_t, kActionKindCount> recoveredByKind_{};
  const char* diagnostic_ = "qualification-fault-backend-invalid-base";
};

}  // namespace ChromaspaceMetalQualification
