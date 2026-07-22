#include "../../../src/metal/ChromaspaceSourceExchangeState.h"

#include <cassert>
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

ViewerRegistration viewer(Capability cap,
                          uint64_t generation = 1,
                          uint64_t retainedBytes = 128) {
  ViewerRegistration value{};
  value.capability = cap;
  value.viewerGeneration = generation;
  value.deviceRegistryId = 17;
  value.pixelFormatMask =
      kPixelFormatRGBA16Float | kPixelFormatRGBA32Float;
  value.maximumWidth = 64;
  value.maximumHeight = 64;
  value.maximumSurfaceBytes = 64;
  value.maximumRetainedBytes = retainedBytes;
  value.maximumSlots = 3;
  value.supportsSharedEvents = true;
  return value;
}

ProducerRegistration producer(Capability cap, uint64_t generation = 1) {
  ProducerRegistration value{};
  value.capability = cap;
  value.senderId = "sender";
  value.senderGeneration = generation;
  value.deviceRegistryId = 17;
  return value;
}

Publication publication(Capability cap,
                        uint64_t senderGeneration,
                        uint64_t sequence,
                        uint32_t slot,
                        uint64_t slotGeneration) {
  Publication value{};
  value.capability = cap;
  value.key.senderId = "sender";
  value.key.senderGeneration = senderGeneration;
  value.key.sequence = sequence;
  value.key.slotIndex = slot;
  value.key.slotGeneration = slotGeneration;
  value.deviceRegistryId = 17;
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

void semanticValidationAndExactRetry() {
  BrokerState state;
  const Capability cap = capability(8);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication malformed = publication(cap, 1, 4, 0, 1);
  malformed.semantics.coverage = SourceCoverage::PartialSource;
  assert(state.publish(malformed).code ==
         ResultCode::InvalidPublication);
  assert(state.livePublicationCount(cap) == 0);

  Publication valid = publication(cap, 1, 7, 0, 1);
  valid.semantics.sourceX = -5;
  valid.semantics.sampledX = -5;
  assert(state.publish(valid).accepted());
  assert(state.publish(valid).accepted());
  Publication mismatchedRetry = valid;
  mismatchedRetry.semantics.colorPrimaries = "display-p3";
  assert(state.publish(mismatchedRetry).code == ResultCode::Stale);

  Publication acquired{};
  assert(state.acquireLatest(cap, "sender", 0, &acquired).accepted());
  assert(acquired.semantics == valid.semantics);
}

void happyPathAndLateAck() {
  BrokerState state;
  const Capability cap = capability(1);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());

  Publication first = publication(cap, 1, 1, 0, 1);
  assert(state.publish(first).accepted());
  assert(state.retainedBytes() == 32);

  Publication acquired{};
  assert(state.acquireLatest(cap, "sender", 0, &acquired).accepted());
  assert(acquired.key == first.key);
  assert(state.acknowledge(cap, first.key,
                           AcknowledgementState::Acquired).accepted());
  assert(state.slotState(cap, first.key) == SlotState::Acquired);
  TransitionResult retired =
      state.acknowledge(cap, first.key, AcknowledgementState::Retired);
  assert(retired.accepted() && retired.released.size() == 1);
  TransitionResult retiredRetry =
      state.acknowledge(
          cap, first.key, AcknowledgementState::Retired);
  assert(retiredRetry.accepted() && retiredRetry.released.empty());
  assert(state.retainedBytes() == 0);

  Publication second = publication(cap, 1, 2, 0, 2);
  assert(state.publish(second).accepted());
  assert(state.acknowledge(
                  cap, first.key,
                  AcknowledgementState::Retired)
             .accepted());
  assert(state.acknowledge(cap, first.key,
                           AcknowledgementState::Acquired).code ==
         ResultCode::Stale);
  assert(state.slotState(cap, second.key) == SlotState::Published);
}

void staleAndTransitionRules() {
  BrokerState state;
  const Capability cap = capability(2);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication first = publication(cap, 1, 5, 1, 5);
  assert(state.publish(first).accepted());
  assert(state.publish(first).accepted());
  assert(state.acknowledge(cap, first.key,
                           AcknowledgementState::Retired).code ==
         ResultCode::InvalidTransition);
  assert(state.acknowledge(cap, first.key,
                           AcknowledgementState::Acquired).accepted());
  assert(state.acknowledge(cap, first.key,
                           AcknowledgementState::Acquired).accepted());
}

void latestWinsReleasesSkippedPublications() {
  BrokerState state;
  const Capability cap = capability(5);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication first = publication(cap, 1, 1, 0, 1);
  Publication second = publication(cap, 1, 2, 1, 1);
  Publication third = publication(cap, 1, 3, 2, 1);
  assert(state.publish(first).accepted());
  assert(state.publish(second).accepted());
  assert(state.publish(third).accepted());

  Publication selected{};
  TransitionResult acquired = state.acquireLatest(cap, "sender", 0, &selected);
  assert(acquired.accepted());
  assert(selected.key == third.key);
  assert(acquired.released.size() == 2);
  assert(state.livePublicationCount(cap) == 1);
  assert(state.retainedBytes() == third.byteSize);
}

void limitsAndDeviceNegotiation() {
  BrokerState state;
  const Capability cap = capability(3);
  assert(state.registerViewer(viewer(cap, 1, 64)).accepted());
  ProducerRegistration wrongDevice = producer(cap);
  wrongDevice.deviceRegistryId = 99;
  assert(state.registerProducer(wrongDevice).code ==
         ResultCode::DeviceMismatch);
  assert(state.registerProducer(producer(cap)).accepted());
  assert(state.publish(publication(cap, 1, 1, 0, 1)).accepted());
  assert(state.publish(publication(cap, 1, 2, 1, 1)).accepted());
  assert(state.publish(publication(cap, 1, 3, 2, 1)).code ==
         ResultCode::ResourceLimit);

  BrokerState sessions;
  for (size_t i = 0; i < kMaximumSessions; ++i) {
    assert(sessions.registerViewer(
        viewer(capability(static_cast<uint8_t>(i + 10)))).accepted());
  }
  assert(sessions.registerViewer(viewer(capability(200))).code ==
         ResultCode::ResourceLimit);
}

void restartAndDisconnect() {
  BrokerState state;
  const Capability cap = capability(4);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication old = publication(cap, 1, 1, 0, 1);
  assert(state.publish(old).accepted());

  TransitionResult restarted = state.registerProducer(producer(cap, 2));
  assert(restarted.accepted() && restarted.released.size() == 1);
  assert(state.retainedBytes() == 0);
  assert(state.publish(old).code == ResultCode::Stale);
  Publication fresh = publication(cap, 2, 2, 0, 2);
  assert(state.publish(fresh).accepted());

  TransitionResult disconnected = state.disconnectViewer(cap);
  assert(disconnected.accepted() && disconnected.released.size() == 1);
  assert(state.sessionCount() == 0);
  assert(state.retainedBytes() == 0);
  assert(state.acquireLatest(cap, "sender", 0, nullptr).code ==
         ResultCode::SessionMissing);
}

void producerDisconnectIsScoped() {
  BrokerState state;
  const Capability cap = capability(6);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication live = publication(cap, 1, 1, 0, 1);
  assert(state.publish(live).accepted());
  assert(state.disconnectProducer(cap, "sender", 2).code ==
         ResultCode::Stale);
  TransitionResult disconnected =
      state.disconnectProducer(cap, "sender", 1);
  assert(disconnected.accepted() && disconnected.released.size() == 1);
  assert(state.sessionCount() == 1);
  assert(state.producerCount(cap) == 0);
  assert(state.retainedBytes() == 0);
}

void producerReleaseJournalIsReplaySafe() {
  BrokerState state;
  const Capability cap = capability(7);
  assert(state.registerViewer(viewer(cap)).accepted());
  assert(state.registerProducer(producer(cap)).accepted());
  Publication first = publication(cap, 1, 1, 0, 1);
  Publication second = publication(cap, 1, 2, 1, 1);
  Publication third = publication(cap, 1, 3, 2, 1);
  assert(state.publish(first).accepted());
  assert(state.publish(second).accepted());
  assert(state.publish(third).accepted());

  Publication selected{};
  assert(state.acquireLatest(cap, "sender", 0, &selected).accepted());
  assert(selected.key == third.key);
  assert(state.pendingProducerReleaseCount(cap, "sender") == 2);

  ProducerReleaseBatch batch{};
  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 0, 1, &batch)
             .accepted());
  assert(batch.events.size() == 1);
  assert(batch.events.front().ordinal == 1);
  assert(batch.events.front().key == first.key);
  assert(batch.throughOrdinal == 1);

  ProducerReleaseBatch retry{};
  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 0, 1, &retry)
             .accepted());
  assert(retry.events.size() == 1);
  assert(retry.events.front().key == first.key);
  assert(state.acknowledgeProducerReleases(
                   cap, "sender", 1, 2)
             .code == ResultCode::InvalidTransition);
  assert(state.acknowledgeProducerReleases(
                   cap, "sender", 1, 1)
             .accepted());
  assert(state.acknowledgeProducerReleases(
                   cap, "sender", 1, 1)
             .accepted());

  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 1, 8, &batch)
             .accepted());
  assert(batch.events.size() == 1);
  assert(batch.events.front().ordinal == 2);
  assert(batch.events.front().key == second.key);
  assert(state.acknowledgeProducerReleases(
                   cap, "sender", 1, 2)
             .accepted());
  assert(state.pendingProducerReleaseCount(cap, "sender") == 0);
  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 2, 8, &batch)
             .code == ResultCode::NoNewPublication);
  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 0, 8, &batch)
             .code == ResultCode::Stale);

  assert(state.acknowledge(cap, third.key,
                           AcknowledgementState::Acquired).accepted());
  assert(state.acknowledge(cap, third.key,
                           AcknowledgementState::Retired).accepted());
  assert(state.fetchProducerReleases(
                   cap, "sender", 1, 2, 8, &batch)
             .accepted());
  assert(batch.events.size() == 1);
  assert(batch.events.front().ordinal == 3);
  assert(batch.events.front().key == third.key);
}

}  // namespace

int main() {
  semanticValidationAndExactRetry();
  happyPathAndLateAck();
  staleAndTransitionRules();
  latestWinsReleasesSkippedPublications();
  limitsAndDeviceNegotiation();
  restartAndDisconnect();
  producerDisconnectIsScoped();
  producerReleaseJournalIsReplaySafe();
  std::cout << "Chromaspace SourceExchangeV2 state tests passed\n";
  return 0;
}
