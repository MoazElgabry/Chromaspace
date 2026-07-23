#include "ChromaspaceMetalQualificationCampaign.h"

#include <algorithm>
#include <limits>

namespace ChromaspaceMetalQualification {
namespace {

constexpr uint32_t kMaximumTargetPresentedFrames = 10000u;
constexpr uint64_t kMinimumTickAllowance = 64u;
constexpr uint64_t kTicksPerPresentedFrame = 8u;
constexpr uint64_t kTicksPerActionAllowance = 8u;
constexpr uint64_t kShowEligibilityTicks = 2u;

constexpr const char* kNotConfigured = "qualification-not-configured";
constexpr const char* kReady = "";
constexpr const char* kInvalidScenario = "qualification-invalid-scenario";
constexpr const char* kInvalidTarget = "qualification-invalid-target";
constexpr const char* kNullAction = "qualification-null-action";
constexpr const char* kCampaignFailed = "qualification-campaign-failed";
constexpr const char* kCampaignCompleted = "qualification-already-completed";
constexpr const char* kPresentedOverTarget =
    "qualification-presented-frames-over-target";
constexpr const char* kPresentedRegression =
    "qualification-presented-frames-regressed";
constexpr const char* kTickCeiling = "qualification-tick-ceiling-exceeded";
constexpr const char* kOrdinalOverflow =
    "qualification-action-ordinal-overflow";
constexpr const char* kPendingAction = "qualification-pending-action";
constexpr const char* kNoPendingAction = "qualification-no-pending-action";
constexpr const char* kAckOrdinalMismatch =
    "qualification-acknowledgement-ordinal-mismatch";
constexpr const char* kAckNotApplied =
    "qualification-action-application-failed";
constexpr const char* kSourceGenerationOverflow =
    "qualification-source-generation-overflow";
constexpr const char* kFinishPresentedMismatch =
    "qualification-finish-presented-frame-mismatch";
constexpr const char* kFinishPending = "qualification-finish-pending-action";
constexpr const char* kFinishHidden = "qualification-finish-hidden";
constexpr const char* kFinishActionsIncomplete =
    "qualification-finish-required-actions-incomplete";
constexpr const char* kFinishNotDrained = "qualification-finish-not-drained";
constexpr const char* kFinishCompletionDirty =
    "qualification-finish-completion-dirty";
constexpr const char* kFinishFaultsIncomplete =
    "qualification-finish-fault-observation-incomplete";
constexpr const char* kFinishRecreationMismatch =
    "qualification-finish-runtime-recreation-mismatch";

struct ResizeSpec final {
  uint32_t width;
  uint32_t height;
  float scale;
};

// Includes odd dimensions and both integer and fractional Retina scales.  The
// table is fixed so a replay can compare exact action payloads byte-for-byte.
constexpr ResizeSpec kResizeSpecs[] = {
    {640u, 480u, 1.0f},  {641u, 481u, 1.0f},  {1280u, 720u, 2.0f},
    {1279u, 721u, 1.5f}, {1921u, 1081u, 2.0f}, {801u, 603u, 1.25f},
    {3840u, 2160u, 1.0f}, {321u, 257u, 1.0f},
};

constexpr uint32_t kResizeSpecCount =
    static_cast<uint32_t>(sizeof(kResizeSpecs) / sizeof(kResizeSpecs[0]));

void setDiagnostic(std::string* diagnostic, const char* value) {
  if (diagnostic != nullptr) *diagnostic = value;
}

bool validScenario(Scenario scenario) noexcept {
  return static_cast<uint8_t>(scenario) <
         static_cast<uint8_t>(Scenario::Count);
}

bool validActionKind(ActionKind kind) noexcept {
  return static_cast<uint8_t>(kind) <
         static_cast<uint8_t>(ActionKind::Count);
}

std::size_t kindIndex(ActionKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

std::size_t familyIndex(ActionFamily family) noexcept {
  return static_cast<std::size_t>(family);
}

ActionFamily familyForKind(ActionKind kind) noexcept {
  switch (kind) {
    case ActionKind::Resize:
      return ActionFamily::Resize;
    case ActionKind::Hide:
    case ActionKind::Show:
      return ActionFamily::Drawable;
    case ActionKind::ReplaceSource:
    case ActionKind::ClearSource:
      return ActionFamily::Source;
    case ActionKind::InjectDrawableUnavailable:
    case ActionKind::InjectPriorGpuSubmissionFailure:
      return ActionFamily::RecoveryFault;
    case ActionKind::MemoryPressureWarning:
    case ActionKind::MemoryPressureCritical:
      return ActionFamily::MemoryPressure;
    case ActionKind::None:
    case ActionKind::Count:
      break;
  }
  return ActionFamily::Count;
}

uint32_t distributedDue(uint32_t index,
                        uint32_t count,
                        uint32_t target) noexcept {
  if (count == 0u) return 0u;
  // Keep the final two presentations free of injected work.  The surviving
  // runtime generation must prove both alternating Gloss renderer variants
  // after recovery, and an event must never become due only after the
  // frame-budget loop has already terminated.
  const uint32_t lastActionFrame = target > 2u ? target - 2u : 1u;
  if (count == 1u) return std::max<uint32_t>(1u, lastActionFrame / 2u);
  const uint64_t numerator =
      static_cast<uint64_t>(index) *
      static_cast<uint64_t>(lastActionFrame - 1u);
  return 1u + static_cast<uint32_t>(numerator / (count - 1u));
}

// Places a family inside the target interval rather than making every family
// compete for frame 1 in the soak schedule.  `leadingSlots` is fixed per
// family and never comes from untrusted input.
uint32_t interiorDue(uint32_t index,
                     uint32_t count,
                     uint32_t target,
                     uint32_t leadingSlots) noexcept {
  if (count == 0u) return 0u;
  const uint64_t denominator = static_cast<uint64_t>(count) + leadingSlots;
  const uint64_t numerator =
      static_cast<uint64_t>(index + leadingSlots) *
      static_cast<uint64_t>(target - 1u);
  return 1u + static_cast<uint32_t>(numerator / denominator);
}

uint32_t minimumTarget(Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::Steady:
      return 2u;
    case Scenario::ResizeStorm:
      return 8u;
    case Scenario::DrawableLoss:
      return 16u;
    case Scenario::SourceChurn:
      return 8u;
    case Scenario::RecoveryFault:
      return 8u;
    case Scenario::Soak:
      return 64u;
    case Scenario::MemoryPressure:
      return 8u;
    case Scenario::Count:
      break;
  }
  return std::numeric_limits<uint32_t>::max();
}

}  // namespace

const char* scenarioLabel(Scenario scenario) noexcept {
  switch (scenario) {
    case Scenario::Steady:
      return "steady";
    case Scenario::ResizeStorm:
      return "resize-storm";
    case Scenario::DrawableLoss:
      return "drawable-loss";
    case Scenario::SourceChurn:
      return "source-churn";
    case Scenario::RecoveryFault:
      return "recovery-fault";
    case Scenario::Soak:
      return "soak";
    case Scenario::MemoryPressure:
      return "memory-pressure";
    case Scenario::Count:
      break;
  }
  return "invalid";
}

bool parseScenarioLabel(std::string_view label,
                        Scenario& outputScenario) noexcept {
  Scenario parsed = Scenario::Count;
  if (label == "steady") {
    parsed = Scenario::Steady;
  } else if (label == "resize-storm") {
    parsed = Scenario::ResizeStorm;
  } else if (label == "drawable-loss") {
    parsed = Scenario::DrawableLoss;
  } else if (label == "source-churn") {
    parsed = Scenario::SourceChurn;
  } else if (label == "recovery-fault") {
    parsed = Scenario::RecoveryFault;
  } else if (label == "soak") {
    parsed = Scenario::Soak;
  } else if (label == "memory-pressure") {
    parsed = Scenario::MemoryPressure;
  } else {
    return false;
  }
  outputScenario = parsed;
  return true;
}

const char* actionKindLabel(ActionKind kind) noexcept {
  switch (kind) {
    case ActionKind::None:
      return "none";
    case ActionKind::Resize:
      return "resize";
    case ActionKind::Hide:
      return "hide";
    case ActionKind::Show:
      return "show";
    case ActionKind::ReplaceSource:
      return "replace-source";
    case ActionKind::ClearSource:
      return "clear-source";
    case ActionKind::InjectDrawableUnavailable:
      return "inject-drawable-unavailable";
    case ActionKind::InjectPriorGpuSubmissionFailure:
      return "inject-prior-gpu-submission-failure";
    case ActionKind::MemoryPressureWarning:
      return "memory-pressure-warning";
    case ActionKind::MemoryPressureCritical:
      return "memory-pressure-critical";
    case ActionKind::Count:
      break;
  }
  return "invalid";
}

Campaign::Campaign(const CampaignConfig& config) noexcept {
  // configure() only writes bounded scalar/array state and does not allocate;
  // the diagnostic argument is omitted so this constructor remains noexcept.
  (void)configure(config, nullptr);
}

void Campaign::resetState() noexcept {
  config_ = CampaignConfig{};
  policy_ = Policy{};
  configurationDiagnostic_ = kNotConfigured;
  ready_ = false;
  failed_ = false;
  completed_ = false;
  pending_ = false;
  hidden_ = false;
  sourceCleared_ = false;
  havePresentedFrames_ = false;
  lastPresentedFrames_ = 0u;
  totalTicks_ = 0u;
  tickCeiling_ = 0u;
  nextOrdinal_ = 1u;
  hiddenSinceTick_ = 0u;
  currentSourceGeneration_ = 1u;
  pendingAction_ = Action{};
  emittedByFamily_.fill(0u);
  appliedByFamily_.fill(0u);
  emittedByKind_.fill(0u);
  appliedByKind_.fill(0u);
}

bool Campaign::resolvePolicy(Scenario scenario, Policy& output) const noexcept {
  output = Policy{};
  switch (scenario) {
    case Scenario::Steady:
      return true;
    case Scenario::ResizeStorm:
      output.resizeActions = kResizeSpecCount;
      return true;
    case Scenario::DrawableLoss:
      output.drawableCycles = 2u;
      return true;
    case Scenario::SourceChurn:
      output.sourceCycles = 3u;
      return true;
    case Scenario::RecoveryFault:
      output.recoveryFaults = 2u;
      output.requiredRecreations = 1u;
      return true;
    case Scenario::Soak:
      output.resizeActions = 4u;
      output.drawableCycles = 2u;
      output.sourceCycles = 2u;
      output.recoveryFaults = 2u;
      output.memoryPressureActions = 2u;
      output.requiredRecreations = 1u;
      return true;
    case Scenario::MemoryPressure:
      output.memoryPressureActions = 2u;
      return true;
    case Scenario::Count:
      break;
  }
  return false;
}

bool Campaign::configure(const CampaignConfig& config,
                         std::string* diagnostic) {
  resetState();
  config_ = config;
  if (!validScenario(config.scenario)) {
    configurationDiagnostic_ = kInvalidScenario;
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }

  if (config.targetPresentedFrames < minimumTarget(config.scenario) ||
      config.targetPresentedFrames > kMaximumTargetPresentedFrames) {
    configurationDiagnostic_ = kInvalidTarget;
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }

  if (!resolvePolicy(config.scenario, policy_)) {
    configurationDiagnostic_ = kInvalidScenario;
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }

  const uint64_t actionCount =
      static_cast<uint64_t>(policy_.resizeActions) +
      static_cast<uint64_t>(policy_.drawableCycles) * 2u +
      static_cast<uint64_t>(policy_.sourceCycles) * 2u +
      static_cast<uint64_t>(policy_.recoveryFaults) +
      static_cast<uint64_t>(policy_.memoryPressureActions);
  tickCeiling_ =
      static_cast<uint64_t>(config.targetPresentedFrames) *
          kTicksPerPresentedFrame +
      actionCount * kTicksPerActionAllowance + kMinimumTickAllowance;
  ready_ = true;
  configurationDiagnostic_ = kReady;
  setDiagnostic(diagnostic, kReady);
  return true;
}

bool Campaign::candidateResize(uint32_t presentedFrames,
                               Action& output) const noexcept {
  if (policy_.resizeActions == 0u ||
      emittedByFamily_[familyIndex(ActionFamily::Resize)] >=
          policy_.resizeActions) {
    return false;
  }
  const uint32_t index =
      emittedByFamily_[familyIndex(ActionFamily::Resize)];
  const uint32_t due =
      config_.scenario == Scenario::Soak
          ? distributedDue(index, policy_.resizeActions,
                           config_.targetPresentedFrames)
          : distributedDue(index, policy_.resizeActions,
                           config_.targetPresentedFrames);
  if (presentedFrames < due) return false;

  const ResizeSpec& spec = kResizeSpecs[index % kResizeSpecCount];
  output = Action{};
  output.kind = ActionKind::Resize;
  output.duePresentedFrames = due;
  output.resizeWidth = spec.width;
  output.resizeHeight = spec.height;
  output.contentScale = spec.scale;
  return true;
}

bool Campaign::candidateDrawable(uint32_t presentedFrames,
                                 Action& output) const noexcept {
  if (policy_.drawableCycles == 0u) return false;

  const std::size_t hideIndex = kindIndex(ActionKind::Hide);
  const std::size_t showIndex = kindIndex(ActionKind::Show);
  const uint32_t hidesApplied = appliedByKind_[hideIndex];
  const uint32_t showsApplied = appliedByKind_[showIndex];
  if (hidesApplied >= policy_.drawableCycles &&
      showsApplied >= policy_.drawableCycles) {
    return false;
  }

  if (hidden_) {
    if (hidesApplied <= showsApplied ||
        totalTicks_ < hiddenSinceTick_ ||
        totalTicks_ - hiddenSinceTick_ < kShowEligibilityTicks) {
      return false;
    }
    output = Action{};
    output.kind = ActionKind::Show;
    // The show belongs to the same presented-frame boundary as its hide.  It
    // is intentionally gated by policy ticks, not by a clock or new frame.
    output.duePresentedFrames =
        config_.scenario == Scenario::Soak
            ? interiorDue(showsApplied, policy_.drawableCycles,
                          config_.targetPresentedFrames, 1u)
            : distributedDue(showsApplied, policy_.drawableCycles,
                             config_.targetPresentedFrames);
    if (presentedFrames < output.duePresentedFrames) {
      // An adapter may poll while hidden at the hide frame.  Do not emit a
      // future cycle's Show before that cycle's due frame.
      return false;
    }
    return true;
  }

  if (hidesApplied != showsApplied ||
      hidesApplied >= policy_.drawableCycles) {
    return false;
  }
  const uint32_t due =
      config_.scenario == Scenario::Soak
          ? interiorDue(hidesApplied, policy_.drawableCycles,
                        config_.targetPresentedFrames, 1u)
          : distributedDue(hidesApplied, policy_.drawableCycles,
                           config_.targetPresentedFrames);
  if (presentedFrames < due) return false;
  output = Action{};
  output.kind = ActionKind::Hide;
  output.duePresentedFrames = due;
  return true;
}

bool Campaign::candidateSource(uint32_t presentedFrames,
                               Action& output) const noexcept {
  if (policy_.sourceCycles == 0u) return false;

  const uint32_t clearsApplied =
      appliedByKind_[kindIndex(ActionKind::ClearSource)];
  const uint32_t replacesApplied =
      appliedByKind_[kindIndex(ActionKind::ReplaceSource)];
  if (clearsApplied >= policy_.sourceCycles &&
      replacesApplied >= policy_.sourceCycles) {
    return false;
  }

  const bool replace = clearsApplied > replacesApplied;
  const uint32_t cycle = replace ? replacesApplied : clearsApplied;
  if (cycle >= policy_.sourceCycles) return false;
  const uint32_t due =
      config_.scenario == Scenario::Soak
          ? interiorDue(cycle, policy_.sourceCycles,
                        config_.targetPresentedFrames, 2u)
          : distributedDue(cycle, policy_.sourceCycles,
                           config_.targetPresentedFrames);
  if (presentedFrames < due) return false;

  output = Action{};
  output.kind = replace ? ActionKind::ReplaceSource : ActionKind::ClearSource;
  output.duePresentedFrames = due;
  output.sourceGeneration =
      replace ? currentSourceGeneration_ + 1u : currentSourceGeneration_;
  return true;
}

bool Campaign::candidateRecoveryFault(uint32_t presentedFrames,
                                      Action& output) const noexcept {
  if (policy_.recoveryFaults == 0u ||
      emittedByFamily_[familyIndex(ActionFamily::RecoveryFault)] >=
          policy_.recoveryFaults) {
    return false;
  }
  const uint32_t index =
      emittedByFamily_[familyIndex(ActionFamily::RecoveryFault)];
  const uint32_t due =
      config_.scenario == Scenario::Soak
          ? interiorDue(index, policy_.recoveryFaults,
                        config_.targetPresentedFrames, 3u)
          : distributedDue(index, policy_.recoveryFaults,
                           config_.targetPresentedFrames);
  if (presentedFrames < due) return false;
  output = Action{};
  output.kind = index == 0u ? ActionKind::InjectDrawableUnavailable
                            : ActionKind::InjectPriorGpuSubmissionFailure;
  output.duePresentedFrames = due;
  return true;
}

bool Campaign::candidateMemoryPressure(uint32_t presentedFrames,
                                       Action& output) const noexcept {
  if (policy_.memoryPressureActions == 0u ||
      emittedByFamily_[familyIndex(ActionFamily::MemoryPressure)] >=
          policy_.memoryPressureActions) {
    return false;
  }
  const uint32_t index =
      emittedByFamily_[familyIndex(ActionFamily::MemoryPressure)];
  const uint32_t due =
      config_.scenario == Scenario::Soak
          ? interiorDue(index, policy_.memoryPressureActions,
                        config_.targetPresentedFrames, 4u)
          : distributedDue(index, policy_.memoryPressureActions,
                           config_.targetPresentedFrames);
  if (presentedFrames < due) return false;
  output = Action{};
  output.kind = index == 0u ? ActionKind::MemoryPressureWarning
                            : ActionKind::MemoryPressureCritical;
  output.duePresentedFrames = due;
  return true;
}

bool Campaign::candidateForSoak(uint32_t presentedFrames,
                                Action& output) const noexcept {
  // Fixed priority is part of the replay contract.  Every family has a
  // bounded count and a due point, so priority cannot starve a later family.
  if (candidateResize(presentedFrames, output)) return true;
  if (candidateDrawable(presentedFrames, output)) return true;
  if (candidateSource(presentedFrames, output)) return true;
  if (candidateRecoveryFault(presentedFrames, output)) return true;
  return candidateMemoryPressure(presentedFrames, output);
}

bool Campaign::emitCandidate(uint32_t presentedFrames,
                             Action candidate,
                             Action* outputAction,
                             std::string* diagnostic) {
  if (!outputAction) {
    setDiagnostic(diagnostic, kNullAction);
    return false;
  }
  if (!validActionKind(candidate.kind) || candidate.kind == ActionKind::None ||
      familyForKind(candidate.kind) == ActionFamily::Count) {
    setDiagnostic(diagnostic, kCampaignFailed);
    failed_ = true;
    return false;
  }
  if (nextOrdinal_ == 0u ||
      nextOrdinal_ == std::numeric_limits<uint64_t>::max()) {
    failed_ = true;
    setDiagnostic(diagnostic, kOrdinalOverflow);
    return false;
  }
  if (candidate.duePresentedFrames > presentedFrames ||
      candidate.duePresentedFrames == 0u ||
      candidate.duePresentedFrames > config_.targetPresentedFrames) {
    failed_ = true;
    setDiagnostic(diagnostic, kCampaignFailed);
    return false;
  }

  candidate.ordinal = nextOrdinal_++;
  pendingAction_ = candidate;
  pending_ = true;
  const ActionFamily family = familyForKind(candidate.kind);
  ++emittedByFamily_[familyIndex(family)];
  ++emittedByKind_[kindIndex(candidate.kind)];
  *outputAction = pendingAction_;
  setDiagnostic(diagnostic, kReady);
  return true;
}

bool Campaign::next(uint32_t presentedFrames,
                    Action* outputAction,
                    std::string* diagnostic) {
  if (!outputAction) {
    setDiagnostic(diagnostic, kNullAction);
    return false;
  }
  if (!ready_) {
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }
  if (failed_) {
    setDiagnostic(diagnostic, kCampaignFailed);
    return false;
  }
  if (completed_) {
    setDiagnostic(diagnostic, kCampaignCompleted);
    return false;
  }
  if (presentedFrames > config_.targetPresentedFrames) {
    failed_ = true;
    setDiagnostic(diagnostic, kPresentedOverTarget);
    return false;
  }
  if (havePresentedFrames_ && presentedFrames < lastPresentedFrames_) {
    failed_ = true;
    setDiagnostic(diagnostic, kPresentedRegression);
    return false;
  }

  if (pending_) {
    // Pending polls are part of the same finite campaign-wide tick budget.
    // Re-issuing the exact action is safe, but an adapter that never
    // acknowledges it must eventually fail instead of hanging forever.
    if (totalTicks_ >= tickCeiling_) {
      failed_ = true;
      setDiagnostic(diagnostic, kTickCeiling);
      return false;
    }
    ++totalTicks_;
    // Observing a newer frame while an action is in flight is harmless, but
    // the action identity and schedule remain unchanged until acknowledgement.
    if (!havePresentedFrames_ || presentedFrames > lastPresentedFrames_) {
      lastPresentedFrames_ = presentedFrames;
      havePresentedFrames_ = true;
    }
    *outputAction = pendingAction_;
    setDiagnostic(diagnostic, kPendingAction);
    return true;
  }

  if (totalTicks_ >= tickCeiling_) {
    failed_ = true;
    setDiagnostic(diagnostic, kTickCeiling);
    return false;
  }
  havePresentedFrames_ = true;
  lastPresentedFrames_ = presentedFrames;
  ++totalTicks_;

  Action candidate{};
  bool hasCandidate = false;
  if (config_.scenario == Scenario::Soak) {
    hasCandidate = candidateForSoak(presentedFrames, candidate);
  } else {
    switch (config_.scenario) {
      case Scenario::ResizeStorm:
        hasCandidate = candidateResize(presentedFrames, candidate);
        break;
      case Scenario::DrawableLoss:
        hasCandidate = candidateDrawable(presentedFrames, candidate);
        break;
      case Scenario::SourceChurn:
        hasCandidate = candidateSource(presentedFrames, candidate);
        break;
      case Scenario::RecoveryFault:
        hasCandidate = candidateRecoveryFault(presentedFrames, candidate);
        break;
      case Scenario::MemoryPressure:
        hasCandidate = candidateMemoryPressure(presentedFrames, candidate);
        break;
      case Scenario::Soak:
        // Handled by candidateForSoak above.
        break;
      case Scenario::Steady:
      case Scenario::Count:
        break;
    }
  }
  if (hasCandidate) {
    return emitCandidate(presentedFrames, candidate, outputAction, diagnostic);
  }
  *outputAction = Action{};
  setDiagnostic(diagnostic, kReady);
  return true;
}

bool Campaign::acknowledgePending(std::string* diagnostic) {
  const ActionKind kind = pendingAction_.kind;
  if (!validActionKind(kind) || kind == ActionKind::None) {
    pending_ = false;
    failed_ = true;
    setDiagnostic(diagnostic, kCampaignFailed);
    return false;
  }

  switch (kind) {
    case ActionKind::Hide:
      hidden_ = true;
      hiddenSinceTick_ = totalTicks_;
      break;
    case ActionKind::Show:
      hidden_ = false;
      break;
    case ActionKind::ClearSource:
      sourceCleared_ = true;
      break;
    case ActionKind::ReplaceSource:
      if (currentSourceGeneration_ ==
          std::numeric_limits<uint64_t>::max()) {
        pending_ = false;
        failed_ = true;
        setDiagnostic(diagnostic, kSourceGenerationOverflow);
        return false;
      }
      // The candidate payload was generated from the current generation, so
      // a successful acknowledgement advances exactly once.
      ++currentSourceGeneration_;
      sourceCleared_ = false;
      break;
    case ActionKind::Resize:
    case ActionKind::InjectDrawableUnavailable:
    case ActionKind::InjectPriorGpuSubmissionFailure:
    case ActionKind::MemoryPressureWarning:
    case ActionKind::MemoryPressureCritical:
      break;
    case ActionKind::None:
    case ActionKind::Count:
      pending_ = false;
      failed_ = true;
      setDiagnostic(diagnostic, kCampaignFailed);
      return false;
  }

  const ActionFamily family = familyForKind(kind);
  ++appliedByFamily_[familyIndex(family)];
  ++appliedByKind_[kindIndex(kind)];
  pendingAction_ = Action{};
  pending_ = false;
  setDiagnostic(diagnostic, kReady);
  return true;
}

bool Campaign::acknowledge(uint64_t ordinal,
                           bool applied,
                           std::string* diagnostic) {
  if (!ready_) {
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }
  if (failed_) {
    setDiagnostic(diagnostic, kCampaignFailed);
    return false;
  }
  if (completed_) {
    setDiagnostic(diagnostic, kCampaignCompleted);
    return false;
  }
  if (!pending_) {
    setDiagnostic(diagnostic, kNoPendingAction);
    return false;
  }
  if (ordinal == 0u || ordinal != pendingAction_.ordinal) {
    setDiagnostic(diagnostic, kAckOrdinalMismatch);
    return false;
  }
  if (!applied) {
    pending_ = false;
    pendingAction_ = Action{};
    failed_ = true;
    setDiagnostic(diagnostic, kAckNotApplied);
    return false;
  }
  return acknowledgePending(diagnostic);
}

bool Campaign::requiredActionsApplied() const noexcept {
  const auto applied = [this](ActionKind kind) {
    return appliedByKind_[kindIndex(kind)];
  };
  return applied(ActionKind::Resize) == policy_.resizeActions &&
         applied(ActionKind::Hide) == policy_.drawableCycles &&
         applied(ActionKind::Show) == policy_.drawableCycles &&
         applied(ActionKind::ClearSource) == policy_.sourceCycles &&
         applied(ActionKind::ReplaceSource) == policy_.sourceCycles &&
         applied(ActionKind::InjectDrawableUnavailable) ==
             (policy_.recoveryFaults == 0u ? 0u : 1u) &&
         applied(ActionKind::InjectPriorGpuSubmissionFailure) ==
             (policy_.recoveryFaults == 0u ? 0u : 1u) &&
         applied(ActionKind::MemoryPressureWarning) ==
             (policy_.memoryPressureActions == 0u ? 0u : 1u) &&
         applied(ActionKind::MemoryPressureCritical) ==
             (policy_.memoryPressureActions == 0u ? 0u : 1u);
}

bool Campaign::requiredFaultsObserved(
    const CompletionObservation& observation) const noexcept {
  return observation.injectedFaultsObserved == policy_.recoveryFaults &&
         observation.recoveredFaults == policy_.recoveryFaults;
}

bool Campaign::recreationCountMatches(
    const CompletionObservation& observation) const noexcept {
  return observation.runtimeRecreations == policy_.requiredRecreations;
}

bool Campaign::finish(const CompletionObservation& observation,
                      std::string* diagnostic) {
  if (!ready_) {
    setDiagnostic(diagnostic, configurationDiagnostic_);
    return false;
  }
  if (failed_) {
    setDiagnostic(diagnostic, kCampaignFailed);
    return false;
  }
  if (completed_) {
    setDiagnostic(diagnostic, kCampaignCompleted);
    return false;
  }
  if (observation.presentedFrames != config_.targetPresentedFrames ||
      !havePresentedFrames_ ||
      lastPresentedFrames_ != config_.targetPresentedFrames) {
    setDiagnostic(diagnostic, kFinishPresentedMismatch);
    return false;
  }
  if (pending_) {
    setDiagnostic(diagnostic, kFinishPending);
    return false;
  }
  if (hidden_) {
    setDiagnostic(diagnostic, kFinishHidden);
    return false;
  }
  if (!requiredActionsApplied()) {
    setDiagnostic(diagnostic, kFinishActionsIncomplete);
    return false;
  }
  if (!observation.drained) {
    setDiagnostic(diagnostic, kFinishNotDrained);
    return false;
  }
  if (!observation.completionClean) {
    setDiagnostic(diagnostic, kFinishCompletionDirty);
    return false;
  }
  if (!requiredFaultsObserved(observation)) {
    setDiagnostic(diagnostic, kFinishFaultsIncomplete);
    return false;
  }
  if (!recreationCountMatches(observation)) {
    setDiagnostic(diagnostic, kFinishRecreationMismatch);
    return false;
  }
  completed_ = true;
  setDiagnostic(diagnostic, kReady);
  return true;
}

Snapshot Campaign::snapshot() const noexcept {
  Snapshot result{};
  result.emittedByFamily = emittedByFamily_;
  result.appliedByFamily = appliedByFamily_;
  result.emittedByKind = emittedByKind_;
  result.appliedByKind = appliedByKind_;
  result.pendingOrdinal = pending_ ? pendingAction_.ordinal : 0u;
  result.pendingKind = pending_ ? pendingAction_.kind : ActionKind::None;
  result.currentSourceGeneration = currentSourceGeneration_;
  result.totalTicks = totalTicks_;
  result.tickCeiling = tickCeiling_;
  result.lastPresentedFrames = lastPresentedFrames_;
  result.targetPresentedFrames = config_.targetPresentedFrames;
  result.hidden = hidden_;
  result.failed = failed_;
  result.completed = completed_;
  return result;
}

}  // namespace ChromaspaceMetalQualification
