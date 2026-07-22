#include "ChromaspaceMetalQualificationCampaign.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using ChromaspaceMetalQualification::Action;
using ChromaspaceMetalQualification::ActionFamily;
using ChromaspaceMetalQualification::ActionKind;
using ChromaspaceMetalQualification::Campaign;
using ChromaspaceMetalQualification::CampaignConfig;
using ChromaspaceMetalQualification::CompletionObservation;
using ChromaspaceMetalQualification::Scenario;
using ChromaspaceMetalQualification::Snapshot;

constexpr std::size_t index(ActionFamily family) {
  return static_cast<std::size_t>(family);
}

constexpr std::size_t index(ActionKind kind) {
  return static_cast<std::size_t>(kind);
}

struct Replay {
  Campaign campaign;
  std::vector<Action> actions;
  CompletionObservation completion{};
};

bool sameAction(const Action& a, const Action& b) {
  return a.ordinal == b.ordinal && a.kind == b.kind &&
         a.duePresentedFrames == b.duePresentedFrames &&
         a.resizeWidth == b.resizeWidth &&
         a.resizeHeight == b.resizeHeight &&
         a.contentScale == b.contentScale &&
         a.sourceGeneration == b.sourceGeneration;
}

Replay drive(Scenario scenario, uint32_t target) {
  Replay replay{};
  std::string diagnostic;
  assert(replay.campaign.configure({scenario, target}, &diagnostic));
  uint32_t presented = 0u;
  uint32_t guard = 0u;
  while (guard++ < 200000u) {
    Action action{};
    assert(replay.campaign.next(presented, &action, &diagnostic));
    if (action.kind != ActionKind::None) {
      replay.actions.push_back(action);
      if (action.kind == ActionKind::InjectDrawableUnavailable ||
          action.kind == ActionKind::InjectPriorGpuSubmissionFailure) {
        ++replay.completion.injectedFaultsObserved;
        ++replay.completion.recoveredFaults;
        if (action.kind ==
            ActionKind::InjectPriorGpuSubmissionFailure) {
          ++replay.completion.runtimeRecreations;
        }
      }
      assert(replay.campaign.acknowledge(action.ordinal, true, &diagnostic));
      continue;
    }

    const Snapshot snapshot = replay.campaign.snapshot();
    if (!snapshot.hidden && presented < target) {
      ++presented;
      continue;
    }
    if (presented == target && !snapshot.hidden &&
        snapshot.pendingOrdinal == 0u) {
      break;
    }
  }
  assert(guard < 200000u);
  replay.completion.presentedFrames = presented;
  replay.completion.drained = true;
  replay.completion.completionClean = true;
  assert(replay.campaign.finish(replay.completion, &diagnostic));
  return replay;
}

void labelsAndConfig() {
  const std::array<std::pair<Scenario, const char*>, 7> cases{{
      {Scenario::Steady, "steady"},
      {Scenario::ResizeStorm, "resize-storm"},
      {Scenario::DrawableLoss, "drawable-loss"},
      {Scenario::SourceChurn, "source-churn"},
      {Scenario::RecoveryFault, "recovery-fault"},
      {Scenario::Soak, "soak"},
      {Scenario::MemoryPressure, "memory-pressure"},
  }};
  for (const auto& item : cases) {
    assert(std::string(
               ChromaspaceMetalQualification::scenarioLabel(item.first)) ==
           item.second);
    Scenario parsed = Scenario::Count;
    assert(ChromaspaceMetalQualification::parseScenarioLabel(item.second,
                                                              parsed));
    assert(parsed == item.first);
  }
  Scenario unchanged = Scenario::SourceChurn;
  assert(!ChromaspaceMetalQualification::parseScenarioLabel("", unchanged));
  assert(unchanged == Scenario::SourceChurn);
  assert(!ChromaspaceMetalQualification::parseScenarioLabel("SOAK", unchanged));
  assert(unchanged == Scenario::SourceChurn);

  const std::array<std::pair<Scenario, uint32_t>, 7> minima{{
      {Scenario::Steady, 2u},
      {Scenario::ResizeStorm, 8u},
      {Scenario::DrawableLoss, 16u},
      {Scenario::SourceChurn, 8u},
      {Scenario::RecoveryFault, 8u},
      {Scenario::Soak, 64u},
      {Scenario::MemoryPressure, 8u},
  }};
  for (const auto& item : minima) {
    Campaign campaign;
    std::string diagnostic;
    assert(!campaign.configure({item.first, item.second - 1u}, &diagnostic));
    assert(!campaign.ready());
    assert(campaign.configure({item.first, item.second}, &diagnostic));
    assert(campaign.ready());
    assert(!campaign.configure({item.first, 10001u}, &diagnostic));
    assert(!campaign.ready());
  }
  Campaign invalid;
  assert(!invalid.configure({Scenario::Count, 64u}));
}

void deterministicReplayAndCoverage() {
  const std::array<std::pair<Scenario, uint32_t>, 7> cases{{
      {Scenario::Steady, 8u},
      {Scenario::ResizeStorm, 32u},
      {Scenario::DrawableLoss, 32u},
      {Scenario::SourceChurn, 32u},
      {Scenario::RecoveryFault, 32u},
      {Scenario::Soak, 96u},
      {Scenario::MemoryPressure, 32u},
  }};
  for (const auto& item : cases) {
    Replay first = drive(item.first, item.second);
    Replay second = drive(item.first, item.second);
    assert(first.actions.size() == second.actions.size());
    for (std::size_t i = 0; i < first.actions.size(); ++i) {
      assert(sameAction(first.actions[i], second.actions[i]));
      assert(first.actions[i].ordinal == i + 1u);
      assert(first.actions[i].duePresentedFrames < item.second);
    }
    if (item.first != Scenario::Soak) {
      for (const Action& action : first.actions) {
        assert(action.duePresentedFrames + 2u <= item.second);
      }
    }
    assert(first.campaign.snapshot().completed);
  }

  Replay resize = drive(Scenario::ResizeStorm, 32u);
  const Snapshot resizeSnapshot = resize.campaign.snapshot();
  assert(resizeSnapshot.appliedByFamily[index(ActionFamily::Resize)] >= 2u);
  bool foundOdd = false;
  bool foundRetina = false;
  for (const Action& action : resize.actions) {
    foundOdd = foundOdd || (action.resizeWidth % 2u) != 0u ||
               (action.resizeHeight % 2u) != 0u;
    foundRetina = foundRetina || action.contentScale > 1.0f;
  }
  assert(foundOdd && foundRetina);

  Replay soak = drive(Scenario::Soak, 96u);
  const Snapshot soakSnapshot = soak.campaign.snapshot();
  for (std::size_t family = 0;
       family < ChromaspaceMetalQualification::kActionFamilyCount;
       ++family) {
    assert(soakSnapshot.appliedByFamily[family] > 0u);
  }
  assert(soakSnapshot.appliedByFamily[index(ActionFamily::Resize)] == 4u);
  assert(soakSnapshot.appliedByFamily[index(ActionFamily::Drawable)] == 4u);
  assert(soakSnapshot.appliedByFamily[index(ActionFamily::Source)] == 4u);
  assert(soakSnapshot.appliedByFamily[index(ActionFamily::RecoveryFault)] ==
         2u);
  assert(soakSnapshot.appliedByFamily[index(ActionFamily::MemoryPressure)] ==
         2u);
}

void standaloneMemoryPressureContract() {
  Replay replay = drive(Scenario::MemoryPressure, 32u);
  assert(replay.actions.size() == 2u);
  assert(replay.actions[0].kind == ActionKind::MemoryPressureWarning);
  assert(replay.actions[1].kind == ActionKind::MemoryPressureCritical);
  assert(replay.actions[0].duePresentedFrames == 1u);
  assert(replay.actions[1].duePresentedFrames == 31u);
  const Snapshot snapshot = replay.campaign.snapshot();
  assert(snapshot.emittedByFamily[index(ActionFamily::MemoryPressure)] == 2u);
  assert(snapshot.appliedByFamily[index(ActionFamily::MemoryPressure)] == 2u);
  assert(snapshot.emittedByKind[index(ActionKind::MemoryPressureWarning)] ==
         1u);
  assert(snapshot.emittedByKind[index(ActionKind::MemoryPressureCritical)] ==
         1u);
  assert(snapshot.appliedByKind[index(ActionKind::MemoryPressureWarning)] ==
         1u);
  assert(snapshot.appliedByKind[index(ActionKind::MemoryPressureCritical)] ==
         1u);
  assert(replay.completion.runtimeRecreations == 0u);
}

void acknowledgementContractAndCeiling() {
  Campaign campaign({Scenario::ResizeStorm, 8u});
  Action action{};
  assert(campaign.next(1u, &action));
  assert(action.kind == ActionKind::Resize);
  assert(!campaign.acknowledge(action.ordinal + 1u, true));
  assert(campaign.snapshot().pendingOrdinal == action.ordinal);
  assert(campaign.acknowledge(action.ordinal, true));
  assert(!campaign.acknowledge(action.ordinal, true));
  assert(!campaign.failed());

  Campaign applyFailure({Scenario::ResizeStorm, 8u});
  assert(applyFailure.next(1u, &action));
  assert(action.kind == ActionKind::Resize);
  assert(!applyFailure.acknowledge(action.ordinal, false));
  assert(applyFailure.failed());
  assert(!applyFailure.next(1u, &action));

  Campaign unacknowledged({Scenario::ResizeStorm, 8u});
  assert(unacknowledged.next(1u, &action));
  assert(action.kind == ActionKind::Resize);
  const uint64_t ceiling = unacknowledged.snapshot().tickCeiling;
  bool failed = false;
  for (uint64_t tick = 0u; tick <= ceiling + 1u; ++tick) {
    if (!unacknowledged.next(1u, &action)) {
      failed = true;
      break;
    }
    assert(action.ordinal == 1u);
  }
  assert(failed);
  assert(unacknowledged.failed());

  Campaign memory({Scenario::MemoryPressure, 8u});
  assert(memory.next(1u, &action));
  assert(action.kind == ActionKind::MemoryPressureWarning);
  const Action pending = action;
  Action repeated{};
  assert(memory.next(2u, &repeated));
  assert(sameAction(repeated, pending));
  assert(!memory.acknowledge(pending.ordinal + 1u, true));
  assert(memory.snapshot().pendingOrdinal == pending.ordinal);
  assert(memory.acknowledge(pending.ordinal, true));
  assert(memory.next(2u, &action));
  assert(action.kind == ActionKind::None);

  Campaign memoryFailure({Scenario::MemoryPressure, 8u});
  assert(memoryFailure.next(1u, &action));
  assert(action.kind == ActionKind::MemoryPressureWarning);
  assert(!memoryFailure.acknowledge(action.ordinal, false));
  assert(memoryFailure.failed());
  assert(memoryFailure.snapshot().appliedByFamily[
             index(ActionFamily::MemoryPressure)] == 0u);
}

void hiddenAndSourceContracts() {
  Campaign drawable({Scenario::DrawableLoss, 16u});
  Action action{};
  uint32_t presented = 0u;
  bool sawHide = false;
  bool sawShow = false;
  for (uint32_t tick = 0u; tick < 200u && !sawShow; ++tick) {
    assert(drawable.next(presented, &action));
    if (action.kind == ActionKind::Hide) {
      sawHide = true;
      assert(drawable.acknowledge(action.ordinal, true));
    } else if (action.kind == ActionKind::Show) {
      sawShow = true;
      assert(drawable.acknowledge(action.ordinal, true));
    } else if (!drawable.snapshot().hidden) {
      ++presented;
    }
  }
  assert(sawHide && sawShow);
  assert(!drawable.snapshot().hidden);

  Replay source = drive(Scenario::SourceChurn, 32u);
  uint64_t lastGeneration = 1u;
  uint32_t clears = 0u;
  uint32_t replacements = 0u;
  bool awaitingReplacement = false;
  for (const Action& item : source.actions) {
    if (item.kind == ActionKind::ClearSource) {
      ++clears;
      assert(!awaitingReplacement);
      assert(item.sourceGeneration == lastGeneration);
      awaitingReplacement = true;
    } else if (item.kind == ActionKind::ReplaceSource) {
      ++replacements;
      assert(awaitingReplacement);
      assert(item.sourceGeneration == lastGeneration + 1u);
      lastGeneration = item.sourceGeneration;
      awaitingReplacement = false;
    }
  }
  assert(clears == 3u && replacements == 3u);
  assert(!awaitingReplacement);
  assert(source.campaign.snapshot().currentSourceGeneration == lastGeneration);
}

void faultAndFinishContracts() {
  Replay recovery = drive(Scenario::RecoveryFault, 32u);
  assert(recovery.campaign.snapshot()
             .appliedByKind[index(ActionKind::InjectDrawableUnavailable)] == 1u);
  assert(recovery.campaign.snapshot()
             .appliedByKind[
                 index(ActionKind::InjectPriorGpuSubmissionFailure)] == 1u);

  Campaign early({Scenario::Steady, 2u});
  CompletionObservation observation{};
  observation.presentedFrames = 1u;
  observation.drained = true;
  observation.completionClean = true;
  assert(!early.finish(observation));
  assert(!early.failed());
  Action action{};
  assert(early.next(2u, &action));
  observation.presentedFrames = 2u;
  observation.drained = false;
  assert(!early.finish(observation));
  observation.drained = true;
  observation.completionClean = false;
  assert(!early.finish(observation));
  observation.completionClean = true;
  assert(early.finish(observation));

  Campaign mismatch({Scenario::RecoveryFault, 8u});
  uint32_t presented = 0u;
  uint32_t faults = 0u;
  for (uint32_t guard = 0u; guard < 1000u; ++guard) {
    assert(mismatch.next(presented, &action));
    if (action.kind != ActionKind::None) {
      if (action.kind == ActionKind::InjectDrawableUnavailable ||
          action.kind == ActionKind::InjectPriorGpuSubmissionFailure) {
        ++faults;
      }
      assert(mismatch.acknowledge(action.ordinal, true));
      continue;
    }
    if (presented < 8u) {
      ++presented;
    } else {
      break;
    }
  }
  observation = {};
  observation.presentedFrames = 8u;
  observation.drained = true;
  observation.completionClean = true;
  observation.injectedFaultsObserved = faults;
  observation.recoveredFaults = faults;
  observation.runtimeRecreations = 0u;
  assert(!mismatch.finish(observation));
  observation.runtimeRecreations = 1u;
  assert(mismatch.finish(observation));

  Replay memory = drive(Scenario::MemoryPressure, 8u);
  const Snapshot memorySnapshot = memory.campaign.snapshot();
  assert(memorySnapshot.appliedByKind[
             index(ActionKind::InjectDrawableUnavailable)] == 0u);
  assert(memorySnapshot.appliedByKind[
             index(ActionKind::InjectPriorGpuSubmissionFailure)] == 0u);
  assert(memory.completion.injectedFaultsObserved == 0u);
  assert(memory.completion.recoveredFaults == 0u);
  assert(memory.completion.runtimeRecreations == 0u);
}

}  // namespace

int main() {
  labelsAndConfig();
  deterministicReplayAndCoverage();
  acknowledgementContractAndCeiling();
  hiddenAndSourceContracts();
  faultAndFinishContracts();
  return 0;
}
