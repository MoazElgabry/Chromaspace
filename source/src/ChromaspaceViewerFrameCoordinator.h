#pragma once

#include <array>
#include <cstdint>

namespace ChromaspaceViewer {

// Reasons are intentionally a stable, portable vocabulary.  Platform
// adapters report revisions through ViewerFrameCoordinatorObservation and the
// coordinator coalesces all changes until the corresponding frame is
// presented.
enum class ViewerFrameDirtyReason : uint32_t {
  None = 0u,
  Initial = 1u << 0,
  Lifecycle = 1u << 1,
  Viewport = 1u << 2,
  Input = 1u << 3,
  Workspace = 1u << 4,
  Source = 1u << 5,
  Animation = 1u << 6,
  Recovery = 1u << 7,
  Qualification = 1u << 8,
};

constexpr uint32_t viewerFrameDirtyMask(ViewerFrameDirtyReason reason) noexcept {
  return static_cast<uint32_t>(reason);
}

constexpr ViewerFrameDirtyReason operator|(ViewerFrameDirtyReason lhs,
                                            ViewerFrameDirtyReason rhs) noexcept {
  return static_cast<ViewerFrameDirtyReason>(
      viewerFrameDirtyMask(lhs) | viewerFrameDirtyMask(rhs));
}

constexpr ViewerFrameDirtyReason operator&(ViewerFrameDirtyReason lhs,
                                            ViewerFrameDirtyReason rhs) noexcept {
  return static_cast<ViewerFrameDirtyReason>(
      viewerFrameDirtyMask(lhs) & viewerFrameDirtyMask(rhs));
}

constexpr ViewerFrameDirtyReason& operator|=(ViewerFrameDirtyReason& lhs,
                                             ViewerFrameDirtyReason rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

constexpr bool viewerFrameDirtyContains(uint32_t mask,
                                        ViewerFrameDirtyReason reason) noexcept {
  return (mask & viewerFrameDirtyMask(reason)) != 0u;
}

// All time values are on one monotonic clock chosen by the caller.  A zero
// time is valid; it is useful in deterministic tests and does not mean
// "unknown".
struct ViewerFrameCoordinatorObservation {
  uint64_t monotonicTimeMilliseconds = 0u;
  bool renderable = false;
  bool closeRequested = false;

  uint64_t lifecycleRevision = 0u;
  uint64_t viewportRevision = 0u;
  uint64_t inputRevision = 0u;
  uint64_t workspaceRevision = 0u;
  uint64_t sourceRevision = 0u;
  uint64_t runtimeRevision = 0u;

  bool animationContinuous = false;
  bool qualificationContinuous = false;
};

enum class ViewerFrameDecisionKind : uint8_t {
  Wait = 0,
  Render,
  Close,
};

struct ViewerFrameCoordinatorDecision {
  ViewerFrameDecisionKind kind = ViewerFrameDecisionKind::Wait;
  // For Wait this is a bounded event-pump recommendation.  A true
  // waitIndefinitely means that non-renderability is not a terminal state;
  // the recommendation may still be finite so live commands/source changes
  // are observed without a busy loop.  For Render this is always zero.
  uint32_t waitMilliseconds = 0u;
  bool waitIndefinitely = false;
  uint32_t reasonMask = 0u;
  uint64_t renderTicket = 0u;

  bool shouldWait() const noexcept {
    return kind == ViewerFrameDecisionKind::Wait;
  }
  bool shouldRender() const noexcept {
    return kind == ViewerFrameDecisionKind::Render;
  }
  bool shouldClose() const noexcept {
    return kind == ViewerFrameDecisionKind::Close;
  }
};

enum class ViewerFrameCompletionKind : uint8_t {
  Presented = 0,
  Retry,
  Suspend,
  RuntimeRecreated,
};

enum class ViewerFrameCompletionStatus : uint8_t {
  Accepted = 0,
  RejectedUnknownTicket,
  RejectedStaleTicket,
};

struct ViewerFrameCompletionResult {
  ViewerFrameCompletionStatus status =
      ViewerFrameCompletionStatus::RejectedUnknownTicket;
  uint32_t retainedReasonMask = 0u;

  bool accepted() const noexcept {
    return status == ViewerFrameCompletionStatus::Accepted;
  }
};

struct ViewerFrameCoordinatorConfig {
  // The default is deliberately close to a 60 Hz display cadence while the
  // integer unit keeps test clocks deterministic.
  uint32_t frameIntervalMilliseconds = 16u;
  // Finite polling is only a responsiveness aid while hidden/minimized or
  // occluded; it never becomes a timeout or close condition.
  uint32_t hiddenPollMilliseconds = 33u;
  // Used while a ticket is in flight.  It prevents a native event loop from
  // spinning if a backend completion callback is delayed.
  uint32_t inFlightPollMilliseconds = 1u;
};

class ViewerFrameCoordinator final {
 public:
  explicit ViewerFrameCoordinator(
      ViewerFrameCoordinatorConfig config = {}) noexcept;

  ViewerFrameCoordinator(const ViewerFrameCoordinator&) = delete;
  ViewerFrameCoordinator& operator=(const ViewerFrameCoordinator&) = delete;

  // Observe is cheap and idempotent.  Revision deltas become dirty reasons;
  // continuous modes keep their reason active until disabled.
  void observe(const ViewerFrameCoordinatorObservation& observation) noexcept;

  ViewerFrameCoordinatorDecision decide(
      uint64_t monotonicTimeMilliseconds) noexcept;

  ViewerFrameCompletionResult complete(
      uint64_t renderTicket,
      ViewerFrameCompletionKind completion) noexcept;

  // Explicit invalidation is useful for an adapter event that has no
  // revision-bearing domain object yet (for example a runtime rebuild).
  void invalidate(ViewerFrameDirtyReason reason) noexcept;

  uint32_t pendingReasonMask() const noexcept { return pendingReasons_; }
  bool hasInFlightTicket() const noexcept { return inFlight_; }
  uint64_t lastPresentedTimeMilliseconds() const noexcept {
    return lastPresentedTimeMilliseconds_;
  }
  uint64_t lastIssuedTicket() const noexcept { return lastIssuedTicket_; }

 private:
  static constexpr std::size_t kReasonCount = 32u;

  void markDirty(ViewerFrameDirtyReason reason) noexcept;
  void markDirtyMask(uint32_t mask) noexcept;
  void markContinuousReasons() noexcept;
  uint32_t clampWait(uint64_t milliseconds) const noexcept;

  ViewerFrameCoordinatorConfig config_{};
  ViewerFrameCoordinatorObservation observation_{};
  bool hasObservation_ = false;
  bool inFlight_ = false;
  uint32_t pendingReasons_ = 0u;
  uint64_t nextTicket_ = 1u;
  uint64_t lastIssuedTicket_ = 0u;
  uint64_t issuedTimeMilliseconds_ = 0u;
  uint64_t lastPresentedTimeMilliseconds_ = 0u;
  bool hasPresentedTime_ = false;
  uint32_t issuedReasons_ = 0u;
  std::array<uint64_t, kReasonCount> reasonEpochs_{};
  std::array<uint64_t, kReasonCount> issuedReasonEpochs_{};
};

}  // namespace ChromaspaceViewer
