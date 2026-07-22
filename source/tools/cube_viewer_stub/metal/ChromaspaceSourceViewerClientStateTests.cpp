#include "../../../src/metal/ChromaspaceSourceViewerClientState.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

using namespace ChromaspaceSourceExchange;

namespace {

Capability capability(uint8_t seed) {
  Capability value{};
  for (size_t index = 0; index < value.size(); ++index) {
    value[index] = static_cast<uint8_t>(seed + index);
  }
  return value;
}

ViewerSessionSnapshot session(uint8_t seed = 1) {
  ViewerSessionSnapshot value{};
  value.capability = capability(seed);
  value.viewerGeneration = seed;
  value.senderId = "sender";
  value.deviceRegistryId = 17;
  value.pixelFormatMask =
      kPixelFormatRGBA16Float | kPixelFormatRGBA32Float;
  value.maximumWidth = 64;
  value.maximumHeight = 64;
  value.maximumSurfaceBytes = 64 * 64 * 16;
  value.maximumRetainedBytes = value.maximumSurfaceBytes * 3;
  value.maximumSlots = 3;
  value.supportsSharedEvents = true;
  return value;
}

Publication publication(const ViewerSessionSnapshot& active,
                        uint64_t sequence,
                        uint32_t slotIndex,
                        uint64_t slotGeneration = 1,
                        uint64_t senderGeneration = 7) {
  Publication value{};
  value.capability = active.capability;
  value.key.senderId = active.senderId;
  value.key.senderGeneration = senderGeneration;
  value.key.sequence = sequence;
  value.key.slotIndex = slotIndex;
  value.key.slotGeneration = slotGeneration;
  value.deviceRegistryId = active.deviceRegistryId;
  value.width = 2;
  value.height = 2;
  value.pixelFormat = 0;
  value.bytesPerRow = 16;
  value.byteSize = 32;
  value.readyValue = sequence;
  value.contentHash = sequence + 100;
  value.semantics.sourceWidth = 2;
  value.semantics.sourceHeight = 2;
  value.semantics.sampledWidth = 2;
  value.semantics.sampledHeight = 2;
  value.semantics.authoritative = true;
  value.semantics.colorPrimaries = "scene-linear-unspecified";
  value.semantics.transferFunction = "linear";
  return value;
}

ViewerClientTransition acceptAcquired(
    ViewerClientState* state,
    const Publication& value,
    bool importSucceeded = true) {
  assert(state->beginImport(value) == ResultCode::Accepted);
  assert(state->importCompleted(value.key, importSucceeded) ==
         ResultCode::Accepted);
  const std::optional<ViewerAcknowledgement> acknowledgement =
      state->beginNextAcknowledgement();
  assert(acknowledgement.has_value());
  assert(acknowledgement->key == value.key);
  assert(acknowledgement->state ==
         ViewerAcknowledgementState::Acquired);
  return state->acknowledgementAccepted(*acknowledgement);
}

void exactAcknowledgementRetry() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication value = publication(active, 1, 0);
  assert(state.beginImport(value) == ResultCode::Accepted);
  assert(state.importCompleted(value.key, true) ==
         ResultCode::Accepted);

  const ViewerAcknowledgement first =
      *state.beginNextAcknowledgement();
  assert(!state.beginNextAcknowledgement().has_value());
  assert(state.acknowledgementTransportFailed(first) ==
         ResultCode::Accepted);
  const ViewerAcknowledgement retry =
      *state.beginNextAcknowledgement();
  assert(retry == first);
  ViewerClientTransition accepted =
      state.acknowledgementAccepted(retry);
  assert(accepted.accepted());
  assert(accepted.activated == value.key);
  assert(state.activeKey() == value.key);
}

void failedImportStillRetiresBrokerSlot() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication value = publication(active, 1, 0);
  ViewerClientTransition acquired =
      acceptAcquired(&state, value, false);
  assert(acquired.accepted());
  assert(!acquired.activated.has_value());

  const ViewerAcknowledgement retired =
      *state.beginNextAcknowledgement();
  assert(retired.key == value.key);
  assert(retired.state == ViewerAcknowledgementState::Retired);
  ViewerClientTransition released =
      state.acknowledgementAccepted(retired);
  assert(released.accepted());
  assert(released.locallyReleasable == value.key);
  assert(state.liveKeyCount() == 0);
}

void replacementWaitsForGpuDrain() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication first = publication(active, 1, 0);
  const Publication second = publication(active, 2, 1);
  assert(acceptAcquired(&state, first).activated == first.key);

  ViewerClientTransition replacement =
      acceptAcquired(&state, second);
  assert(replacement.accepted());
  assert(replacement.activated == second.key);
  assert(replacement.needsGpuDrain == first.key);
  assert(!state.beginNextAcknowledgement().has_value());

  assert(state.gpuDrainCompleted(first.key) ==
         ResultCode::Accepted);
  const ViewerAcknowledgement retired =
      *state.beginNextAcknowledgement();
  assert(retired.key == first.key);
  assert(retired.state == ViewerAcknowledgementState::Retired);
  ViewerClientTransition released =
      state.acknowledgementAccepted(retired);
  assert(released.locallyReleasable == first.key);
  assert(state.activeKey() == second.key);
}

void rejectsForeignAndStalePublications() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());

  Publication wrongCapability = publication(active, 1, 0);
  wrongCapability.capability[0] ^= 0xff;
  assert(state.beginImport(wrongCapability) ==
         ResultCode::CapabilityMismatch);

  Publication wrongDevice = publication(active, 1, 0);
  ++wrongDevice.deviceRegistryId;
  assert(state.beginImport(wrongDevice) ==
         ResultCode::DeviceMismatch);

  Publication wrongFormat = publication(active, 1, 0);
  ViewerSessionSnapshot narrow = active;
  narrow.pixelFormatMask = kPixelFormatRGBA16Float;
  assert(state.installSession(narrow).code ==
         ResultCode::InvalidRegistration);
  wrongFormat.pixelFormat = 2;
  assert(state.beginImport(wrongFormat) ==
         ResultCode::InvalidPublication);

  Publication badSemantics = publication(active, 1, 0);
  badSemantics.semantics.coverage = SourceCoverage::PartialSource;
  assert(state.beginImport(badSemantics) ==
         ResultCode::InvalidPublication);

  const Publication accepted = publication(active, 2, 0);
  assert(acceptAcquired(&state, accepted).accepted());
  assert(state.beginImport(accepted) == ResultCode::Stale);

  Publication old = publication(active, 1, 1);
  assert(state.beginImport(old) == ResultCode::Stale);

  Publication otherGeneration = publication(active, 3, 1, 1, 8);
  assert(state.beginImport(otherGeneration) == ResultCode::Stale);
}

void identicalSessionPreservesActiveAndInflightAck() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication first = publication(active, 1, 0);
  const Publication second = publication(active, 2, 1);
  assert(acceptAcquired(&state, first).accepted());
  assert(state.beginImport(second) == ResultCode::Accepted);
  assert(state.importCompleted(second.key, true) ==
         ResultCode::Accepted);
  const ViewerAcknowledgement inFlight =
      *state.beginNextAcknowledgement();

  ViewerClientTransition duplicate = state.installSession(active);
  assert(duplicate.accepted());
  assert(!duplicate.sessionChanged);
  assert(duplicate.abandoned.empty());
  assert(state.activeKey() == first.key);
  assert(state.acknowledgementTransportFailed(inFlight) ==
         ResultCode::Accepted);
  assert(*state.beginNextAcknowledgement() == inFlight);
}

void replacementAndInvalidationAbandonExactKeys() {
  ViewerClientState state;
  const ViewerSessionSnapshot firstSession = session(1);
  assert(state.installSession(firstSession).accepted());
  const Publication first = publication(firstSession, 1, 0);
  const Publication second = publication(firstSession, 2, 1);
  const Publication third = publication(firstSession, 3, 2);
  assert(acceptAcquired(&state, first).accepted());
  assert(acceptAcquired(&state, second).accepted());
  assert(state.beginImport(third) == ResultCode::Accepted);
  assert(state.liveKeyCount() == 3);

  ViewerClientTransition replaced =
      state.installSession(session(2));
  assert(replaced.accepted());
  assert(replaced.sessionChanged);
  assert(replaced.abandoned.size() == 3);
  assert(std::find(
             replaced.abandoned.begin(),
             replaced.abandoned.end(),
             first.key) != replaced.abandoned.end());
  assert(std::find(
             replaced.abandoned.begin(),
             replaced.abandoned.end(),
             second.key) != replaced.abandoned.end());
  assert(std::find(
             replaced.abandoned.begin(),
             replaced.abandoned.end(),
             third.key) != replaced.abandoned.end());

  const ViewerSessionSnapshot secondSession = session(2);
  const Publication afterReplacement =
      publication(secondSession, 1, 0);
  assert(acceptAcquired(&state, afterReplacement).accepted());
  ViewerClientTransition invalidated = state.invalidateSession();
  assert(invalidated.accepted());
  assert(invalidated.abandoned.size() == 1);
  assert(invalidated.abandoned.front() == afterReplacement.key);
  assert(state.phase() == ViewerClientPhase::Failed);
}

void boundedThreeSlotOwnership() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication first = publication(active, 1, 0);
  const Publication second = publication(active, 2, 1);
  const Publication third = publication(active, 3, 2);
  const Publication fourth = publication(active, 4, 0, 2);
  assert(acceptAcquired(&state, first).accepted());
  assert(acceptAcquired(&state, second).accepted());
  ViewerClientTransition failed =
      acceptAcquired(&state, third, false);
  assert(failed.accepted());
  assert(state.liveKeyCount() == 3);
  assert(state.beginImport(fourth) == ResultCode::ResourceLimit);
}

void clearRetiresActiveAndInFlightCandidate() {
  ViewerClientState state;
  const ViewerSessionSnapshot active = session();
  assert(state.installSession(active).accepted());
  const Publication first = publication(active, 1, 0);
  const Publication second = publication(active, 2, 1);
  assert(acceptAcquired(&state, first).activated == first.key);

  assert(state.beginImport(second) == ResultCode::Accepted);
  assert(state.importCompleted(second.key, true) ==
         ResultCode::Accepted);
  ViewerClientTransition cleared = state.clearActiveSource();
  assert(cleared.accepted());
  assert(cleared.needsGpuDrain == first.key);
  assert(!state.activeKey().has_value());

  const ViewerAcknowledgement secondAcquired =
      *state.beginNextAcknowledgement();
  ViewerClientTransition discarded =
      state.acknowledgementAccepted(secondAcquired);
  assert(discarded.accepted());
  assert(!discarded.activated.has_value());
  assert(discarded.needsGpuDrain == second.key);
  assert(!state.activeKey().has_value());

  assert(state.gpuDrainCompleted(first.key) ==
         ResultCode::Accepted);
  assert(state.gpuDrainCompleted(second.key) ==
         ResultCode::Accepted);
  for (const PublicationKey* expected : {&first.key, &second.key}) {
    const ViewerAcknowledgement retired =
        *state.beginNextAcknowledgement();
    assert(retired.key == *expected);
    assert(retired.state ==
           ViewerAcknowledgementState::Retired);
    assert(state.acknowledgementAccepted(retired).accepted());
  }
  assert(state.liveKeyCount() == 0);
}

}  // namespace

int main() {
  exactAcknowledgementRetry();
  failedImportStillRetiresBrokerSlot();
  replacementWaitsForGpuDrain();
  rejectsForeignAndStalePublications();
  identicalSessionPreservesActiveAndInflightAck();
  replacementAndInvalidationAbandonExactKeys();
  boundedThreeSlotOwnership();
  clearRetiresActiveAndInFlightCandidate();
  std::cout
      << "Chromaspace SourceExchangeV2 viewer client state tests passed\n";
  return 0;
}
