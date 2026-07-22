#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

// Portable policy for deterministic macOS Metal viewer qualification.
//
// This module deliberately knows nothing about Cocoa, Metal, OFX, transport,
// clocks, threads, or process state.  An adapter presents the actions to the
// platform/runtime and acknowledges the exact ordinal after applying them.
// The same policy can therefore drive a native canary, a mock adapter, or a
// future host harness without becoming a second runtime state machine.
namespace ChromaspaceMetalQualification {

enum class Scenario : uint8_t {
  Steady = 0,
  ResizeStorm,
  DrawableLoss,
  SourceChurn,
  RecoveryFault,
  Soak,
  MemoryPressure,
  Count,
};

const char* scenarioLabel(Scenario scenario) noexcept;

// Parses one of the exact stable labels returned by scenarioLabel().  A
// failed parse never changes outputScenario.
bool parseScenarioLabel(std::string_view label,
                        Scenario& outputScenario) noexcept;

enum class ActionKind : uint8_t {
  None = 0,
  Resize,
  Hide,
  Show,
  ReplaceSource,
  ClearSource,
  InjectDrawableUnavailable,
  InjectPriorGpuSubmissionFailure,
  MemoryPressureWarning,
  MemoryPressureCritical,
  Count,
};

const char* actionKindLabel(ActionKind kind) noexcept;

enum class ActionFamily : uint8_t {
  Resize = 0,
  Drawable,
  Source,
  RecoveryFault,
  MemoryPressure,
  Count,
};

constexpr std::size_t kActionFamilyCount =
    static_cast<std::size_t>(ActionFamily::Count);
constexpr std::size_t kActionKindCount =
    static_cast<std::size_t>(ActionKind::Count);

// Action is intentionally a bounded POD.  The adapter must not need to parse
// strings to apply a policy event.
struct Action final {
  uint64_t ordinal = 0u;
  ActionKind kind = ActionKind::None;
  uint64_t duePresentedFrames = 0u;
  uint32_t resizeWidth = 0u;
  uint32_t resizeHeight = 0u;
  float contentScale = 1.0f;
  uint64_t sourceGeneration = 0u;
};

static_assert(std::is_trivially_copyable<Action>::value,
              "qualification actions must remain bounded POD values");

struct CampaignConfig final {
  Scenario scenario = Scenario::Steady;
  uint32_t targetPresentedFrames = 0u;
};

struct CompletionObservation final {
  uint32_t presentedFrames = 0u;
  bool drained = false;
  bool completionClean = false;
  uint32_t runtimeRecreations = 0u;
  uint32_t injectedFaultsObserved = 0u;
  uint32_t recoveredFaults = 0u;
};

struct Snapshot final {
  std::array<uint32_t, kActionFamilyCount> emittedByFamily{};
  std::array<uint32_t, kActionFamilyCount> appliedByFamily{};
  std::array<uint32_t, kActionKindCount> emittedByKind{};
  std::array<uint32_t, kActionKindCount> appliedByKind{};
  uint64_t pendingOrdinal = 0u;
  ActionKind pendingKind = ActionKind::None;
  uint64_t currentSourceGeneration = 1u;
  uint64_t totalTicks = 0u;
  uint64_t tickCeiling = 0u;
  uint32_t lastPresentedFrames = 0u;
  uint32_t targetPresentedFrames = 0u;
  bool hidden = false;
  bool failed = false;
  bool completed = false;
};

class Campaign final {
 public:
  Campaign() noexcept = default;
  explicit Campaign(const CampaignConfig& config) noexcept;

  Campaign(const Campaign&) = delete;
  Campaign& operator=(const Campaign&) = delete;
  Campaign(Campaign&&) = default;
  Campaign& operator=(Campaign&&) = default;

  // Reinitializes the policy.  Invalid configuration leaves the policy not
  // ready and stores a stable diagnostic; no platform state is touched.
  bool configure(const CampaignConfig& config,
                 std::string* diagnostic = nullptr);

  bool ready() const noexcept { return ready_; }
  bool failed() const noexcept { return failed_; }
  bool completed() const noexcept { return completed_; }
  const char* configurationDiagnostic() const noexcept {
    return configurationDiagnostic_;
  }

  // Advances one bounded policy tick and emits at most one action.  If an
  // action is pending, the exact same action is returned without advancing
  // the schedule; adapters can safely poll until they acknowledge it.  A
  // successful call with no due action returns ActionKind::None.
  bool next(uint32_t presentedFrames,
            Action* outputAction,
            std::string* diagnostic = nullptr);

  // Acknowledgement is an exact-ordinal handshake.  Wrong, stale, duplicate,
  // or failed application never advances policy state; applied=false puts the
  // campaign into a permanent failed state.
  bool acknowledge(uint64_t ordinal,
                   bool applied,
                   std::string* diagnostic = nullptr);

  // Validates the final adapter observation.  A failed early check is
  // retryable; only an action-application failure or tick-ceiling violation
  // permanently fails the campaign.
  bool finish(const CompletionObservation& observation,
              std::string* diagnostic = nullptr);

  Snapshot snapshot() const noexcept;

 private:
  struct Policy final {
    uint32_t resizeActions = 0u;
    uint32_t drawableCycles = 0u;
    uint32_t sourceCycles = 0u;
    uint32_t recoveryFaults = 0u;
    uint32_t memoryPressureActions = 0u;
    uint32_t requiredRecreations = 0u;
  };

  void resetState() noexcept;
  bool resolvePolicy(Scenario scenario, Policy& output) const noexcept;
  bool emitCandidate(uint32_t presentedFrames,
                     Action candidate,
                     Action* outputAction,
                     std::string* diagnostic);
  bool candidateResize(uint32_t presentedFrames, Action& output) const noexcept;
  bool candidateDrawable(uint32_t presentedFrames,
                         Action& output) const noexcept;
  bool candidateSource(uint32_t presentedFrames, Action& output) const noexcept;
  bool candidateRecoveryFault(uint32_t presentedFrames,
                              Action& output) const noexcept;
  bool candidateMemoryPressure(uint32_t presentedFrames,
                               Action& output) const noexcept;
  bool candidateForSoak(uint32_t presentedFrames, Action& output) const noexcept;
  bool acknowledgePending(std::string* diagnostic);
  bool requiredActionsApplied() const noexcept;
  bool requiredFaultsObserved(const CompletionObservation& observation) const
      noexcept;
  bool recreationCountMatches(const CompletionObservation& observation) const
      noexcept;

  CampaignConfig config_{};
  Policy policy_{};
  const char* configurationDiagnostic_ = "qualification-not-configured";
  bool ready_ = false;
  bool failed_ = false;
  bool completed_ = false;
  bool pending_ = false;
  bool hidden_ = false;
  bool sourceCleared_ = false;
  bool havePresentedFrames_ = false;
  uint32_t lastPresentedFrames_ = 0u;
  uint64_t totalTicks_ = 0u;
  uint64_t tickCeiling_ = 0u;
  uint64_t nextOrdinal_ = 1u;
  uint64_t hiddenSinceTick_ = 0u;
  uint64_t currentSourceGeneration_ = 1u;
  Action pendingAction_{};
  std::array<uint32_t, kActionFamilyCount> emittedByFamily_{};
  std::array<uint32_t, kActionFamilyCount> appliedByFamily_{};
  std::array<uint32_t, kActionKindCount> emittedByKind_{};
  std::array<uint32_t, kActionKindCount> appliedByKind_{};
};

}  // namespace ChromaspaceMetalQualification
