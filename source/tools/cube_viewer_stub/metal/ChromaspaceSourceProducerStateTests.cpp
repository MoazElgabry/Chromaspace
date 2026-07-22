#include "../../../src/metal/ChromaspaceSourceProducerState.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

using namespace ChromaspaceSourceExchange;

namespace {

Capability capability(uint8_t seed) {
  Capability value{};
  for (size_t i = 0; i < value.size(); ++i) {
    value[i] = static_cast<uint8_t>(seed + i);
  }
  return value;
}

ProducerLeaseSnapshot lease(uint8_t seed = 1,
                            uint64_t viewerGeneration = 1) {
  ProducerLeaseSnapshot value{};
  value.capability = capability(seed);
  value.viewerGeneration = viewerGeneration;
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

ProducerPublicationSpec spec(uint64_t readyValue = 1) {
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

void leaseAndSpecValidation() {
  ProducerState state;
  Publication out{};
  assert(state.reserve(spec(), &out) == ResultCode::SessionMissing);

  ProducerLeaseSnapshot invalid = lease();
  invalid.capability = Capability{};
  assert(state.installLease(invalid).code ==
         ResultCode::InvalidRegistration);
  invalid = lease();
  invalid.supportsSharedEvents = false;
  assert(state.installLease(invalid).code ==
         ResultCode::InvalidRegistration);
  invalid = lease();
  invalid.protocolMajor += 1;
  assert(state.installLease(invalid).code == ResultCode::ProtocolMismatch);

  assert(state.installLease(lease()).accepted());
  ProducerPublicationSpec bad = spec();
  bad.width = 65;
  assert(state.reserve(bad, &out) == ResultCode::InvalidPublication);
  bad = spec();
  bad.pixelFormat = 2;
  assert(state.reserve(bad, &out) == ResultCode::InvalidPublication);
  bad = spec();
  bad.height = 2;
  bad.bytesPerRow = std::numeric_limits<uint64_t>::max();
  bad.byteSize = std::numeric_limits<uint64_t>::max();
  assert(state.reserve(bad, &out) == ResultCode::InvalidPublication);
  bad = spec();
  bad.semantics.sampledWidth = 1;
  assert(state.reserve(bad, &out) == ResultCode::InvalidPublication);
  assert(state.livePublicationCount() == 0);
}

void callerSequenceAndSemanticContract() {
  ProducerState state;
  assert(state.installLease(lease()).accepted());
  ProducerPublicationSpec firstSpec = spec(4);
  firstSpec.semantics.sourceX = -2;
  firstSpec.semantics.sampledX = -2;
  Publication first{};
  assert(state.reserve(firstSpec, &first) == ResultCode::Accepted);
  assert(first.key.sequence == 4);
  assert(first.semantics == firstSpec.semantics);
  assert(state.cancelPending(first.key) == ResultCode::Accepted);

  ProducerPublicationSpec gap = spec(11);
  Publication second{};
  assert(state.reserve(gap, &second) == ResultCode::Accepted);
  assert(second.key.sequence == 11);
  assert(state.cancelPending(second.key) == ResultCode::Accepted);

  ProducerPublicationSpec stale = spec(9);
  assert(state.reserve(stale, nullptr) == ResultCode::Stale);
  assert(state.livePublicationCount() == 0);
  ProducerPublicationSpec malformed = spec(12);
  malformed.semantics.colorPrimaries =
      std::string(kMaximumSemanticIdentifierBytes + 1, 'x');
  assert(state.reserve(malformed, nullptr) ==
         ResultCode::InvalidPublication);
  assert(state.livePublicationCount() == 0);
}

void boundedRingAndRetryIdentity() {
  ProducerState state;
  const ProducerLeaseSnapshot activeLease = lease();
  assert(state.installLease(activeLease).accepted());
  assert(state.installLease(activeLease).accepted());

  Publication first{};
  Publication second{};
  Publication third{};
  Publication blocked{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(2), &second) == ResultCode::Accepted);
  assert(state.reserve(spec(3), &third) == ResultCode::Accepted);
  assert(state.reserve(spec(4), &blocked) == ResultCode::SlotBusy);
  assert(state.livePublicationCount() == 3);

  assert(state.markPublished(first.key) == ResultCode::Accepted);
  assert(state.markPublished(first.key) == ResultCode::Accepted);
  assert(state.release(first.key) == ResultCode::Accepted);
  assert(state.release(first.key) == ResultCode::Accepted);

  Publication reused{};
  assert(state.reserve(spec(5), &reused) == ResultCode::Accepted);
  assert(reused.key.sequence > third.key.sequence);
  assert(reused.key.slotIndex == first.key.slotIndex);
  assert(reused.key.slotGeneration > first.key.slotGeneration);
  assert(state.release(first.key) == ResultCode::Stale);
  assert(state.slotState(reused.key.slotIndex) ==
         ProducerSlotState::Pending);
}

void lostPublishReplyAndCancel() {
  ProducerState state;
  assert(state.installLease(lease()).accepted());
  Publication pending{};
  assert(state.reserve(spec(), &pending) == ResultCode::Accepted);
  assert(state.release(pending.key) == ResultCode::Accepted);
  assert(state.markPublished(pending.key) == ResultCode::Stale);

  Publication canceled{};
  assert(state.reserve(spec(2), &canceled) == ResultCode::Accepted);
  assert(state.cancelPending(canceled.key) == ResultCode::Accepted);
  assert(state.cancelPending(canceled.key) == ResultCode::Accepted);
  assert(state.markPublished(canceled.key) == ResultCode::Stale);
}

void replacementAndInvalidationPreserveCounters() {
  ProducerState state;
  ProducerLeaseSnapshot firstLease = lease(3, 1);
  assert(state.installLease(firstLease).accepted());
  Publication first{};
  Publication second{};
  assert(state.reserve(spec(1), &first) == ResultCode::Accepted);
  assert(state.reserve(spec(2), &second) == ResultCode::Accepted);

  ProducerLeaseSnapshot mutated = firstLease;
  mutated.maximumWidth -= 1;
  assert(state.installLease(mutated).code ==
         ResultCode::InvalidRegistration);
  assert(state.livePublicationCount() == 2);

  ProducerLeaseSnapshot replacement = lease(4, 2);
  ProducerTransition installed = state.installLease(replacement);
  assert(installed.accepted());
  assert(installed.abandoned.size() == 2);
  assert(state.livePublicationCount() == 0);

  Publication afterReplacement{};
  assert(state.reserve(spec(3), &afterReplacement) ==
         ResultCode::Accepted);
  assert(afterReplacement.key.sequence > second.key.sequence);
  if (afterReplacement.key.slotIndex == first.key.slotIndex) {
    assert(afterReplacement.key.slotGeneration >
           first.key.slotGeneration);
  }

  ProducerTransition invalidated = state.invalidateLease();
  assert(invalidated.accepted());
  assert(invalidated.abandoned.size() == 1);
  assert(!state.lease().has_value());
  assert(state.reserve(spec(4), nullptr) == ResultCode::SessionMissing);

  assert(state.installLease(lease(5, 3)).accepted());
  Publication afterInvalidation{};
  assert(state.reserve(spec(5), &afterInvalidation) ==
         ResultCode::Accepted);
  assert(afterInvalidation.key.sequence > afterReplacement.key.sequence);
}

}  // namespace

int main() {
  leaseAndSpecValidation();
  callerSequenceAndSemanticContract();
  boundedRingAndRetryIdentity();
  lostPublishReplyAndCancel();
  replacementAndInvalidationPreserveCounters();
  std::cout << "Chromaspace SourceExchangeV2 producer state tests passed\n";
  return 0;
}
