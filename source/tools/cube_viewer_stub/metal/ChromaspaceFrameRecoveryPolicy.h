#pragma once

#include <cstdint>

#include "ChromaspaceMetalFrameFailure.h"

namespace ChromaspaceFrameRecoveryPolicy {

constexpr uint32_t kDefaultTransientRetryLimit = 3u;
constexpr uint32_t kDefaultMaxRecreationAttempts = 3u;
constexpr uint32_t kDefaultBaseBackoffMilliseconds = 8u;
constexpr uint32_t kDefaultMaxBackoffMilliseconds = 250u;

enum class SurfaceVisibility : uint8_t {
  Visible = 0,
  Occluded,
  Unavailable,
};

enum class Action : uint8_t {
  Continue = 0,
  RetryLater,
  SuspendUntilVisible,
  RecreateRuntime,
  Terminate,
};

struct Config {
  // Three visible transient failures are retried.  The fourth asks for a
  // runtime recreation.  These finite limits keep a broken drawable from
  // turning into an unbounded busy loop.
  uint32_t transientRetryLimit = kDefaultTransientRetryLimit;
  uint32_t maxRecreationAttempts = kDefaultMaxRecreationAttempts;
  uint32_t baseBackoffMilliseconds = kDefaultBaseBackoffMilliseconds;
  uint32_t maxBackoffMilliseconds = kDefaultMaxBackoffMilliseconds;
};

struct Decision {
  Action action = Action::Terminate;
  uint32_t backoffMilliseconds = 0u;
  uint32_t transientFailureCount = 0u;
  uint32_t recreationAttempts = 0u;
};

class FrameRecoveryPolicy final {
 public:
  explicit FrameRecoveryPolicy(const Config& config = Config{}) noexcept;

  FrameRecoveryPolicy(const FrameRecoveryPolicy&) = delete;
  FrameRecoveryPolicy& operator=(const FrameRecoveryPolicy&) = delete;

  // A presented frame is the only success that resets all accumulated
  // recovery state, including the bounded recreation budget.
  Decision onPresentedFrame() noexcept;

  // Classifies the typed backend/executor result without touching platform
  // state.  Occluded or unavailable surfaces suspend transient drawable
  // failures without consuming retry or recreation budget.
  Decision onFailure(ChromaspaceMetalFrameFailure::Kind failure,
                     SurfaceVisibility visibility) noexcept;

  // The caller invokes this after an action of RecreateRuntime.  A failed
  // recreation consumes the already-issued attempt and either requests a
  // bounded retry or terminates.  A successful recreation leaves recovery
  // counters intact until a frame is actually presented.
  Decision onRuntimeRecreationResult(bool succeeded) noexcept;

  void reset() noexcept;
  uint32_t transientFailureCount() const noexcept { return transientFailures_; }
  uint32_t recreationAttempts() const noexcept { return recreationAttempts_; }

 private:
  Decision make(Action action, uint32_t backoff = 0u) const noexcept;
  Decision makeRecreationDecision() noexcept;
  uint32_t transientBackoffMilliseconds() const noexcept;
  uint32_t recreationBackoffMilliseconds() const noexcept;

  Config config_{};
  uint32_t transientFailures_ = 0u;
  uint32_t recreationAttempts_ = 0u;
  bool recreationPending_ = false;
};

const char* actionLabel(Action action) noexcept;
const char* visibilityLabel(SurfaceVisibility visibility) noexcept;

}  // namespace ChromaspaceFrameRecoveryPolicy
