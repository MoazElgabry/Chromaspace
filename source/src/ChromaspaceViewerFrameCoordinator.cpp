#include "ChromaspaceViewerFrameCoordinator.h"

#include <algorithm>
#include <limits>

namespace ChromaspaceViewer {
namespace {

constexpr uint32_t kMaximumWaitMilliseconds = 1000u;

std::size_t reasonIndex(uint32_t bit) noexcept {
  std::size_t index = 0u;
  while ((bit & 1u) == 0u && index + 1u < 32u) {
    bit >>= 1u;
    ++index;
  }
  return index;
}

bool revisionChanged(uint64_t before, uint64_t after) noexcept {
  return before != after;
}

}  // namespace

ViewerFrameCoordinator::ViewerFrameCoordinator(
    ViewerFrameCoordinatorConfig config) noexcept
    : config_(config) {
  config_.frameIntervalMilliseconds = std::min(
      config_.frameIntervalMilliseconds, kMaximumWaitMilliseconds);
  config_.hiddenPollMilliseconds = std::clamp(
      config_.hiddenPollMilliseconds, 1u, kMaximumWaitMilliseconds);
  config_.inFlightPollMilliseconds = std::clamp(
      config_.inFlightPollMilliseconds, 1u, kMaximumWaitMilliseconds);
}

void ViewerFrameCoordinator::markDirty(ViewerFrameDirtyReason reason) noexcept {
  const uint32_t bit = viewerFrameDirtyMask(reason);
  if (bit == 0u) return;
  pendingReasons_ |= bit;
  const std::size_t index = reasonIndex(bit);
  if (reasonEpochs_[index] != std::numeric_limits<uint64_t>::max()) {
    ++reasonEpochs_[index];
  }
}

void ViewerFrameCoordinator::markDirtyMask(uint32_t mask) noexcept {
  while (mask != 0u) {
    const uint32_t bit = mask & (~mask + 1u);
    markDirty(static_cast<ViewerFrameDirtyReason>(bit));
    mask &= ~bit;
  }
}

void ViewerFrameCoordinator::markContinuousReasons() noexcept {
  if (observation_.animationContinuous) markDirty(ViewerFrameDirtyReason::Animation);
  if (observation_.qualificationContinuous) {
    markDirty(ViewerFrameDirtyReason::Qualification);
  }
}

uint32_t ViewerFrameCoordinator::clampWait(uint64_t milliseconds) const noexcept {
  return static_cast<uint32_t>(std::min<uint64_t>(
      milliseconds, static_cast<uint64_t>(kMaximumWaitMilliseconds)));
}

void ViewerFrameCoordinator::observe(
    const ViewerFrameCoordinatorObservation& observation) noexcept {
  if (!hasObservation_) {
    observation_ = observation;
    hasObservation_ = true;
    // Keep work requested while hidden so the first visible observation still
    // receives a frame.
    markDirty(ViewerFrameDirtyReason::Initial);
    if (observation_.animationContinuous) {
      markDirty(ViewerFrameDirtyReason::Animation);
    }
    if (observation_.qualificationContinuous) {
      markDirty(ViewerFrameDirtyReason::Qualification);
    }
    return;
  }

  if (revisionChanged(observation_.lifecycleRevision,
                      observation.lifecycleRevision)) {
    markDirty(ViewerFrameDirtyReason::Lifecycle);
  }
  if (revisionChanged(observation_.viewportRevision,
                      observation.viewportRevision)) {
    markDirty(ViewerFrameDirtyReason::Viewport);
  }
  if (revisionChanged(observation_.inputRevision, observation.inputRevision)) {
    markDirty(ViewerFrameDirtyReason::Input);
  }
  if (revisionChanged(observation_.workspaceRevision,
                      observation.workspaceRevision)) {
    markDirty(ViewerFrameDirtyReason::Workspace);
  }
  if (revisionChanged(observation_.sourceRevision,
                      observation.sourceRevision)) {
    markDirty(ViewerFrameDirtyReason::Source);
  }
  if (revisionChanged(observation_.runtimeRevision,
                      observation.runtimeRevision)) {
    markDirty(ViewerFrameDirtyReason::Recovery);
  }
  if (observation.animationContinuous && !observation_.animationContinuous) {
    markDirty(ViewerFrameDirtyReason::Animation);
  }
  if (observation.qualificationContinuous &&
      !observation_.qualificationContinuous) {
    markDirty(ViewerFrameDirtyReason::Qualification);
  }
  observation_ = observation;
}

ViewerFrameCoordinatorDecision ViewerFrameCoordinator::decide(
    uint64_t monotonicTimeMilliseconds) noexcept {
  ViewerFrameCoordinatorDecision decision{};
  if (hasObservation_ && observation_.closeRequested) {
    decision.kind = ViewerFrameDecisionKind::Close;
    decision.reasonMask = pendingReasons_;
    return decision;
  }
  if (!hasObservation_) {
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = config_.hiddenPollMilliseconds;
    decision.waitIndefinitely = true;
    return decision;
  }

  if (inFlight_) {
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = config_.inFlightPollMilliseconds;
    decision.waitIndefinitely = false;
    decision.reasonMask = pendingReasons_;
    return decision;
  }

  markContinuousReasons();
  if (!observation_.renderable) {
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = config_.hiddenPollMilliseconds;
    decision.waitIndefinitely = true;
    decision.reasonMask = pendingReasons_;
    return decision;
  }
  if (pendingReasons_ == 0u) {
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = config_.hiddenPollMilliseconds;
    decision.waitIndefinitely = false;
    return decision;
  }

  if (hasPresentedTime_ &&
      monotonicTimeMilliseconds < lastPresentedTimeMilliseconds_) {
    // A caller supplied clock must be monotonic.  Treat a backwards sample
    // as due now rather than making a frame wait for an unbounded interval.
    monotonicTimeMilliseconds = lastPresentedTimeMilliseconds_;
  }
  if (hasPresentedTime_ &&
      monotonicTimeMilliseconds - lastPresentedTimeMilliseconds_ <
          config_.frameIntervalMilliseconds) {
    const uint64_t elapsed = monotonicTimeMilliseconds -
                             lastPresentedTimeMilliseconds_;
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = clampWait(
        static_cast<uint64_t>(config_.frameIntervalMilliseconds) - elapsed);
    decision.waitIndefinitely = false;
    decision.reasonMask = pendingReasons_;
    return decision;
  }

  decision.kind = ViewerFrameDecisionKind::Render;
  decision.reasonMask = pendingReasons_;
  if (nextTicket_ == 0u) {
    // Ticket zero is reserved as the invalid/unknown value.  Exhaustion is a
    // safe wait; callers can continue pumping without issuing an ambiguous
    // completion.
    decision.kind = ViewerFrameDecisionKind::Wait;
    decision.waitMilliseconds = config_.hiddenPollMilliseconds;
    decision.waitIndefinitely = true;
    return decision;
  }
  decision.renderTicket = nextTicket_;
  lastIssuedTicket_ = nextTicket_;
  if (nextTicket_ != std::numeric_limits<uint64_t>::max()) {
    ++nextTicket_;
  } else {
    nextTicket_ = 0u;
  }
  issuedReasons_ = pendingReasons_;
  issuedTimeMilliseconds_ = monotonicTimeMilliseconds;
  for (uint32_t mask = issuedReasons_; mask != 0u;) {
    const uint32_t bit = mask & (~mask + 1u);
    issuedReasonEpochs_[reasonIndex(bit)] = reasonEpochs_[reasonIndex(bit)];
    mask &= ~bit;
  }
  inFlight_ = true;
  return decision;
}

ViewerFrameCompletionResult ViewerFrameCoordinator::complete(
    uint64_t renderTicket,
    ViewerFrameCompletionKind completion) noexcept {
  ViewerFrameCompletionResult result{};
  if (renderTicket == 0u) {
    result.status = ViewerFrameCompletionStatus::RejectedUnknownTicket;
    result.retainedReasonMask = pendingReasons_;
    return result;
  }
  if (renderTicket != lastIssuedTicket_) {
    result.status = renderTicket < lastIssuedTicket_
                        ? ViewerFrameCompletionStatus::RejectedStaleTicket
                        : ViewerFrameCompletionStatus::RejectedUnknownTicket;
    result.retainedReasonMask = pendingReasons_;
    return result;
  }
  if (!inFlight_) {
    result.status = ViewerFrameCompletionStatus::RejectedStaleTicket;
    result.retainedReasonMask = pendingReasons_;
    return result;
  }

  result.status = ViewerFrameCompletionStatus::Accepted;
  inFlight_ = false;
  if (completion == ViewerFrameCompletionKind::Presented) {
    // Only reasons whose epoch is unchanged since issuance are consumed.  A
    // revision arriving during the render therefore survives this ticket.
    for (uint32_t mask = issuedReasons_; mask != 0u;) {
      const uint32_t bit = mask & (~mask + 1u);
      const std::size_t index = reasonIndex(bit);
      if (reasonEpochs_[index] == issuedReasonEpochs_[index]) {
        pendingReasons_ &= ~bit;
      }
      mask &= ~bit;
    }
    lastPresentedTimeMilliseconds_ = issuedTimeMilliseconds_;
    hasPresentedTime_ = true;
  } else if (completion == ViewerFrameCompletionKind::RuntimeRecreated) {
    markDirty(ViewerFrameDirtyReason::Recovery);
  }
  // Retry/Suspend deliberately retain all pending reasons.  Recovery also
  // retains the issued reasons and adds an explicit rebuild reason.
  result.retainedReasonMask = pendingReasons_;
  issuedReasons_ = 0u;
  return result;
}

void ViewerFrameCoordinator::invalidate(ViewerFrameDirtyReason reason) noexcept {
  markDirty(reason);
}

}  // namespace ChromaspaceViewer
