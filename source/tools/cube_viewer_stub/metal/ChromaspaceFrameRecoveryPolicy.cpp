#include "ChromaspaceFrameRecoveryPolicy.h"

#include <algorithm>

namespace ChromaspaceFrameRecoveryPolicy {
namespace {

using FailureKind = ChromaspaceMetalFrameFailure::Kind;

bool transient(FailureKind failure) noexcept {
  return failure == FailureKind::DrawableUnavailable ||
         failure == FailureKind::BackpressureTimeout;
}

bool needsRecreation(FailureKind failure) noexcept {
  return failure == FailureKind::PriorGpuSubmissionFailure ||
         failure == FailureKind::CompositorUnavailable ||
         failure == FailureKind::MetalContextUnavailable ||
         failure == FailureKind::CommandBufferUnavailable;
}

uint32_t boundedDouble(uint32_t value, uint32_t maximum) noexcept {
  if (value >= maximum) return maximum;
  if (value > maximum / 2u) return maximum;
  return std::min(maximum, value * 2u);
}

}  // namespace

FrameRecoveryPolicy::FrameRecoveryPolicy(const Config& config) noexcept
    : config_(config) {
  config_.maxBackoffMilliseconds =
      std::max(config_.baseBackoffMilliseconds, config_.maxBackoffMilliseconds);
}

Decision FrameRecoveryPolicy::make(Action action, uint32_t backoff) const noexcept {
  Decision result{};
  result.action = action;
  result.backoffMilliseconds = backoff;
  result.transientFailureCount = transientFailures_;
  result.recreationAttempts = recreationAttempts_;
  return result;
}

uint32_t FrameRecoveryPolicy::transientBackoffMilliseconds() const noexcept {
  if (transientFailures_ == 0u || config_.baseBackoffMilliseconds == 0u) {
    return 0u;
  }
  uint32_t result = config_.baseBackoffMilliseconds;
  for (uint32_t index = 1u; index < transientFailures_; ++index) {
    result = boundedDouble(result, config_.maxBackoffMilliseconds);
  }
  return result;
}

uint32_t FrameRecoveryPolicy::recreationBackoffMilliseconds() const noexcept {
  if (recreationAttempts_ == 0u || config_.baseBackoffMilliseconds == 0u) {
    return 0u;
  }
  uint32_t result = config_.baseBackoffMilliseconds;
  for (uint32_t index = 1u; index < recreationAttempts_; ++index) {
    result = boundedDouble(result, config_.maxBackoffMilliseconds);
  }
  return result;
}

Decision FrameRecoveryPolicy::makeRecreationDecision() noexcept {
  if (config_.maxRecreationAttempts == 0u ||
      recreationAttempts_ >= config_.maxRecreationAttempts) {
    recreationPending_ = false;
    return make(Action::Terminate);
  }
  ++recreationAttempts_;
  recreationPending_ = true;
  transientFailures_ = 0u;
  return make(Action::RecreateRuntime, recreationBackoffMilliseconds());
}

Decision FrameRecoveryPolicy::onPresentedFrame() noexcept {
  reset();
  return make(Action::Continue);
}

Decision FrameRecoveryPolicy::onFailure(FailureKind failure,
                                         SurfaceVisibility visibility) noexcept {
  if (failure == FailureKind::None ||
      static_cast<uint8_t>(failure) >=
          static_cast<uint8_t>(FailureKind::Count)) {
    return make(Action::Terminate);
  }

  if (transient(failure)) {
    if (visibility != SurfaceVisibility::Visible) {
      // A window with no drawable is not a failing runtime.  Do not consume
      // the transient budget while it is occluded or unavailable.
      return make(Action::SuspendUntilVisible);
    }
    if (transientFailures_ == UINT32_MAX) {
      return makeRecreationDecision();
    }
    ++transientFailures_;
    if (transientFailures_ <= config_.transientRetryLimit) {
      return make(Action::RetryLater, transientBackoffMilliseconds());
    }
    return makeRecreationDecision();
  }

  if (needsRecreation(failure)) return makeRecreationDecision();

  // Invalid, invariant, encoding, unknown, and any future unclassified
  // result terminate fail-closed.  There is no safe retry without a typed
  // recovery category.
  return make(Action::Terminate);
}

Decision FrameRecoveryPolicy::onRuntimeRecreationResult(bool succeeded) noexcept {
  if (!recreationPending_) {
    return make(succeeded ? Action::Continue : Action::Terminate);
  }
  recreationPending_ = false;
  if (succeeded) return make(Action::Continue);
  if (recreationAttempts_ >= config_.maxRecreationAttempts) {
    return make(Action::Terminate);
  }
  // Keep the attempt budget consumed and issue the next bounded recreation.
  ++recreationAttempts_;
  recreationPending_ = true;
  return make(Action::RecreateRuntime, recreationBackoffMilliseconds());
}

void FrameRecoveryPolicy::reset() noexcept {
  transientFailures_ = 0u;
  recreationAttempts_ = 0u;
  recreationPending_ = false;
}

const char* actionLabel(Action action) noexcept {
  switch (action) {
    case Action::Continue: return "continue";
    case Action::RetryLater: return "retry-later";
    case Action::SuspendUntilVisible: return "suspend-until-visible";
    case Action::RecreateRuntime: return "recreate-runtime";
    case Action::Terminate: return "terminate";
  }
  return "terminate";
}

const char* visibilityLabel(SurfaceVisibility visibility) noexcept {
  switch (visibility) {
    case SurfaceVisibility::Visible: return "visible";
    case SurfaceVisibility::Occluded: return "occluded";
    case SurfaceVisibility::Unavailable: return "unavailable";
  }
  return "unavailable";
}

}  // namespace ChromaspaceFrameRecoveryPolicy
