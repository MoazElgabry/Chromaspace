#include "../../../src/metal/ChromaspaceSourceProducerClientState.h"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace ChromaspaceSourceExchange;

namespace {

Capability capability(uint8_t seed) {
  Capability value{};
  for (size_t i = 0; i < value.size(); ++i) {
    value[i] = static_cast<uint8_t>(seed + i);
  }
  return value;
}

ProducerLeaseSnapshot lease(uint8_t seed = 1) {
  ProducerLeaseSnapshot value{};
  value.capability = capability(seed);
  value.viewerGeneration = seed;
  value.senderId = "sender";
  value.senderGeneration = 7;
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

ProducerPublicationSpec spec(uint64_t readyValue) {
  ProducerPublicationSpec value{};
  value.sequence = readyValue;
  value.width = 2;
  value.height = 2;
  value.pixelFormat = 0;
  value.bytesPerRow = 16;
  value.byteSize = 32;
  value.readyValue = readyValue;
  value.contentHash = readyValue + 100;
  value.semantics.sourceWidth = 2;
  value.semantics.sourceHeight = 2;
  value.semantics.sampledWidth = 2;
  value.semantics.sampledHeight = 2;
  value.semantics.authoritative = true;
  value.semantics.colorPrimaries = "scene-linear-unspecified";
  value.semantics.transferFunction = "linear";
  return value;
}

ProducerReleaseBatch releaseBatch(const ProducerLeaseSnapshot& activeLease) {
  ProducerReleaseBatch value{};
  value.capability = activeLease.capability;
  value.senderId = activeLease.senderId;
  value.senderGeneration = activeLease.senderGeneration;
  return value;
}

void exactPacketRetryAndAcceptance() {
  ProducerClientState state;
  assert(state.installLease(lease()).accepted());
  Publication publication{};
  assert(state.reserve(spec(1), &publication) == ResultCode::Accepted);
  assert(state.markSendReady(publication.key) == ResultCode::Accepted);

  std::optional<Publication> first = state.beginNextPublish();
  assert(first.has_value());
  assert(first->key == publication.key);
  assert(first->readyValue == publication.readyValue);
  assert(first->contentHash == publication.contentHash);
  assert(first->semantics == publication.semantics);
  assert(!state.beginNextPublish().has_value());

  assert(state.publishTransportFailed(publication.key) ==
         ResultCode::Accepted);
  std::optional<Publication> retry = state.beginNextPublish();
  assert(retry.has_value());
  assert(retry->key == first->key);
  assert(retry->readyValue == first->readyValue);
  assert(retry->contentHash == first->contentHash);
  assert(retry->semantics == first->semantics);
  assert(state.publishAccepted(publication.key) == ResultCode::Accepted);
  assert(state.livePublicationCount() == 1);
  assert(state.sendReadyCount() == 0);
}

void cancellationAndReleaseBeforeReply() {
  ProducerClientState state;
  const ProducerLeaseSnapshot activeLease = lease();
  assert(state.installLease(activeLease).accepted());

  Publication canceled{};
  assert(state.reserve(spec(1), &canceled) == ResultCode::Accepted);
  assert(state.cancel(canceled.key) == ResultCode::Accepted);
  assert(state.livePublicationCount() == 0);

  Publication inFlight{};
  assert(state.reserve(spec(2), &inFlight) == ResultCode::Accepted);
  assert(state.markSendReady(inFlight.key) == ResultCode::Accepted);
  assert(state.beginNextPublish()->key == inFlight.key);
  ProducerReleaseBatch batch = releaseBatch(activeLease);
  batch.throughOrdinal = 1;
  batch.events.push_back(ProducerReleaseEvent{1, inFlight.key});
  ProducerClientTransition applied = state.applyReleaseBatch(batch);
  assert(applied.accepted());
  assert(applied.released.size() == 1);
  assert(state.livePublicationCount() == 0);
  assert(state.publishAccepted(inFlight.key) == ResultCode::Stale);

  Publication reused{};
  assert(state.reserve(spec(3), &reused) == ResultCode::Accepted);
  assert(state.publishAccepted(inFlight.key) == ResultCode::Stale);
  assert(state.publication(reused.key).has_value());
}

void fixedCapacityAdmissionAndBackPressure() {
  ProducerClientState state;
  assert(state.installLease(lease()).accepted());
  Publication first{};
  Publication second{};
  Publication third{};
  assert(state.reserve(spec(10), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(20), &second) == ResultCode::Accepted);
  assert(state.reserve(spec(30), &third) == ResultCode::Accepted);
  assert(state.reserve(spec(40), nullptr) == ResultCode::SlotBusy);
  assert(state.livePublicationCount() == kMaximumSlots);
  assert(state.sendReadyCount() == 0);
}

void cancellationReusesSlotWithoutStaleMutation() {
  ProducerClientState state;
  ProducerLeaseSnapshot oneSlot = lease();
  oneSlot.maximumSlots = 1;
  assert(state.installLease(oneSlot).accepted());

  Publication first{};
  assert(state.reserve(spec(10), &first) == ResultCode::Accepted);
  assert(state.cancel(first.key) == ResultCode::Accepted);

  Publication second{};
  assert(state.reserve(spec(20), &second) == ResultCode::Accepted);
  assert(second.key.slotIndex == first.key.slotIndex);
  assert(second.key.slotGeneration > first.key.slotGeneration);
  assert(state.markSendReady(first.key) == ResultCode::Stale);
  assert(state.cancel(first.key) == ResultCode::Stale);
  assert(state.publication(second.key).has_value());
}

void releaseCursorReplayAndAtomicValidation() {
  ProducerClientState state;
  const ProducerLeaseSnapshot activeLease = lease();
  assert(state.installLease(activeLease).accepted());
  Publication first{};
  Publication second{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(2), &second) == ResultCode::Accepted);

  ProducerReleaseBatch corrupt = releaseBatch(activeLease);
  corrupt.throughOrdinal = 2;
  corrupt.events.push_back(ProducerReleaseEvent{1, first.key});
  PublicationKey unknown = second.key;
  ++unknown.slotGeneration;
  corrupt.events.push_back(ProducerReleaseEvent{2, unknown});
  assert(state.applyReleaseBatch(corrupt).code == ResultCode::Stale);
  assert(state.livePublicationCount() == 2);
  assert(state.releaseFetchCursor() == 0);

  ProducerReleaseBatch valid = releaseBatch(activeLease);
  valid.throughOrdinal = 2;
  valid.events.push_back(ProducerReleaseEvent{1, first.key});
  valid.events.push_back(ProducerReleaseEvent{2, second.key});
  ProducerClientTransition applied = state.applyReleaseBatch(valid);
  assert(applied.accepted());
  assert(applied.released.size() == 2);
  assert(applied.releaseAcknowledgementOrdinal == 2);
  assert(state.releaseFetchCursor() == 2);
  assert(state.livePublicationCount() == 0);

  ProducerClientTransition replay = state.applyReleaseBatch(valid);
  assert(replay.accepted());
  assert(replay.released.empty());
  assert(replay.releaseAcknowledgementOrdinal == 2);
  assert(state.releaseAcknowledgementAccepted(2) ==
         ResultCode::Accepted);
  assert(state.acknowledgedReleaseOrdinal() == 2);
}

void identicalLeaseAndReleaseIdentityAreStable() {
  ProducerClientState state;
  const ProducerLeaseSnapshot activeLease = lease();
  assert(state.installLease(activeLease).accepted());
  Publication publication{};
  assert(state.reserve(spec(1), &publication) == ResultCode::Accepted);
  assert(state.markSendReady(publication.key) == ResultCode::Accepted);

  assert(state.installLease(activeLease).accepted());
  assert(state.livePublicationCount() == 1);
  assert(state.sendReadyCount() == 1);

  ProducerReleaseBatch wrongCapability = releaseBatch(activeLease);
  wrongCapability.capability[0] ^= 0xff;
  wrongCapability.throughOrdinal = 1;
  wrongCapability.events.push_back(
      ProducerReleaseEvent{1, publication.key});
  assert(state.applyReleaseBatch(wrongCapability).code ==
         ResultCode::InvalidTransition);
  assert(state.livePublicationCount() == 1);

  ProducerReleaseBatch wrongSender = releaseBatch(activeLease);
  wrongSender.senderId = "other";
  wrongSender.throughOrdinal = 1;
  wrongSender.events.push_back(
      ProducerReleaseEvent{1, publication.key});
  assert(state.applyReleaseBatch(wrongSender).code ==
         ResultCode::InvalidTransition);
  assert(state.livePublicationCount() == 1);

  ProducerReleaseBatch valid = releaseBatch(activeLease);
  valid.throughOrdinal = 1;
  valid.events.push_back(ProducerReleaseEvent{1, publication.key});
  assert(state.applyReleaseBatch(valid).accepted());
  assert(state.releaseAcknowledgementAccepted(1) ==
         ResultCode::Accepted);
  assert(state.installLease(activeLease).accepted());
  assert(state.releaseFetchCursor() == 1);
  assert(state.acknowledgedReleaseOrdinal() == 1);
}

void gpuCompletionsCannotReorderPublication() {
  ProducerClientState state;
  assert(state.installLease(lease()).accepted());
  Publication first{};
  Publication second{};
  Publication third{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(2), &second) == ResultCode::Accepted);
  assert(state.reserve(spec(3), &third) == ResultCode::Accepted);

  assert(state.markSendReady(third.key) == ResultCode::Accepted);
  assert(!state.beginNextPublish().has_value());
  assert(state.markSendReady(second.key) == ResultCode::Accepted);
  assert(!state.beginNextPublish().has_value());
  assert(state.markSendReady(first.key) == ResultCode::Accepted);
  assert(state.beginNextPublish()->key == first.key);
  assert(state.publishAccepted(first.key) == ResultCode::Accepted);
  assert(state.beginNextPublish()->key == second.key);
  assert(state.publishAccepted(second.key) == ResultCode::Accepted);
  assert(state.beginNextPublish()->key == third.key);
  assert(state.publishAccepted(third.key) == ResultCode::Accepted);
  assert(!state.beginNextPublish().has_value());
}

void replacementLeaseClearsAllInlineRecords() {
  ProducerClientState state;
  assert(state.installLease(lease(1)).accepted());
  Publication first{};
  Publication second{};
  Publication third{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(2), &second) == ResultCode::Accepted);
  assert(state.reserve(spec(3), &third) == ResultCode::Accepted);
  assert(state.markSendReady(first.key) == ResultCode::Accepted);
  assert(state.beginNextPublish()->key == first.key);
  assert(state.markSendReady(second.key) == ResultCode::Accepted);

  ProducerClientTransition replacement = state.installLease(lease(2));
  assert(replacement.accepted());
  assert(replacement.leaseChanged);
  assert(replacement.abandoned.size() == 3);
  assert(state.livePublicationCount() == 0);
  assert(state.sendReadyCount() == 0);
  assert(state.publishAccepted(first.key) == ResultCode::Stale);
  assert(state.publishTransportFailed(first.key) == ResultCode::Stale);
  assert(state.markSendReady(second.key) == ResultCode::Stale);
  assert(state.cancel(third.key) == ResultCode::Stale);

  Publication replacementPublication{};
  assert(state.reserve(spec(4), &replacementPublication) ==
         ResultCode::Accepted);
  assert(replacementPublication.key.sequence > third.key.sequence);
}

void terminalFailureAbandonsWithoutCounterReuse() {
  ProducerClientState state;
  assert(state.installLease(lease(1)).accepted());
  Publication first{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  ProducerClientTransition failed = state.invalidateTransport();
  assert(failed.accepted());
  assert(failed.abandoned.size() == 1);
  assert(state.phase() == ProducerClientPhase::Failed);
  assert(state.reserve(spec(2), nullptr) == ResultCode::SessionMissing);

  assert(state.installLease(lease(2)).accepted());
  Publication afterReconnect{};
  assert(state.reserve(spec(3), &afterReconnect) == ResultCode::Accepted);
  assert(afterReconnect.key.sequence > first.key.sequence);
}

}  // namespace

int main() {
  exactPacketRetryAndAcceptance();
  cancellationAndReleaseBeforeReply();
  fixedCapacityAdmissionAndBackPressure();
  cancellationReusesSlotWithoutStaleMutation();
  releaseCursorReplayAndAtomicValidation();
  identicalLeaseAndReleaseIdentityAreStable();
  gpuCompletionsCannotReorderPublication();
  replacementLeaseClearsAllInlineRecords();
  terminalFailureAbandonsWithoutCounterReuse();
  std::cout
      << "Chromaspace SourceExchangeV2 producer client state tests passed\n";
  return 0;
}
