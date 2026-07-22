#include "ChromaspaceSourceExchangeState.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <utility>

namespace ChromaspaceSourceExchange {
namespace {

bool validCapability(const Capability& capability) {
  return std::any_of(capability.begin(), capability.end(),
                     [](uint8_t byte) { return byte != 0; });
}

bool validSemanticIdentifier(const std::string& value) {
  if (value.empty() || value.size() > kMaximumSemanticIdentifierBytes) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '-' || c == '_' ||
           c == '.' || c == '+';
  });
}

bool intervalContained(int32_t outerStart,
                       uint32_t outerSize,
                       int32_t innerStart,
                       uint32_t innerSize) {
  if (outerSize == 0 || innerSize == 0) return false;
  const int64_t outerBegin = outerStart;
  const int64_t outerEnd = outerBegin + static_cast<int64_t>(outerSize);
  const int64_t innerBegin = innerStart;
  const int64_t innerEnd = innerBegin + static_cast<int64_t>(innerSize);
  return innerBegin >= outerBegin && innerEnd <= outerEnd;
}

bool validIdentityRange(bool enabled,
                        int32_t y1,
                        int32_t y2,
                        int32_t sourceY,
                        uint32_t sourceHeight,
                        uint32_t bandHeight) {
  if (!enabled) return y1 == 0 && y2 == 0;
  const int64_t height = static_cast<int64_t>(y2) - y1;
  return height == static_cast<int64_t>(bandHeight) &&
         intervalContained(sourceY,
                           sourceHeight,
                           y1,
                           static_cast<uint32_t>(height));
}

bool sameViewer(const ViewerRegistration& a, const ViewerRegistration& b) {
  return a.protocolMajor == b.protocolMajor &&
         a.protocolMinor == b.protocolMinor &&
         a.capability == b.capability &&
         a.viewerGeneration == b.viewerGeneration &&
         a.deviceRegistryId == b.deviceRegistryId &&
         a.pixelFormatMask == b.pixelFormatMask &&
         a.maximumWidth == b.maximumWidth &&
         a.maximumHeight == b.maximumHeight &&
         a.maximumSurfaceBytes == b.maximumSurfaceBytes &&
         a.maximumRetainedBytes == b.maximumRetainedBytes &&
         a.maximumSlots == b.maximumSlots &&
         a.supportsSharedEvents == b.supportsSharedEvents;
}

bool validViewer(const ViewerRegistration& registration) {
  constexpr uint32_t knownFormats =
      kPixelFormatRGBA16Float | kPixelFormatRGBA32Float;
  return registration.protocolMajor == kProtocolMajor &&
         registration.protocolMinor <= kProtocolMinor &&
         validCapability(registration.capability) &&
         registration.viewerGeneration != 0 &&
         registration.deviceRegistryId != 0 &&
         registration.pixelFormatMask != 0 &&
         (registration.pixelFormatMask & ~knownFormats) == 0 &&
         registration.maximumWidth > 0 &&
         registration.maximumWidth <= kMaximumDimension &&
         registration.maximumHeight > 0 &&
         registration.maximumHeight <= kMaximumDimension &&
         registration.maximumSurfaceBytes > 0 &&
         registration.maximumSurfaceBytes <= kMaximumSurfaceBytes &&
         registration.maximumRetainedBytes >=
             registration.maximumSurfaceBytes &&
         registration.maximumRetainedBytes <= kMaximumRetainedBytes &&
         registration.maximumSlots > 0 &&
         registration.maximumSlots <= kMaximumSlots &&
         registration.supportsSharedEvents;
}

uint32_t formatMask(uint32_t pixelFormat) {
  if (pixelFormat == 0) return kPixelFormatRGBA16Float;
  if (pixelFormat == 1) return kPixelFormatRGBA32Float;
  return 0;
}

bool samePublication(const Publication& a, const Publication& b) {
  return a.capability == b.capability && a.key == b.key &&
         a.deviceRegistryId == b.deviceRegistryId &&
         a.width == b.width && a.height == b.height &&
         a.pixelFormat == b.pixelFormat &&
         a.bytesPerRow == b.bytesPerRow &&
         a.byteSize == b.byteSize &&
         a.readyValue == b.readyValue &&
         a.contentHash == b.contentHash &&
         a.semantics == b.semantics;
}

}  // namespace

bool PublicationKey::operator==(const PublicationKey& other) const {
  return senderId == other.senderId &&
         senderGeneration == other.senderGeneration &&
         sequence == other.sequence &&
         slotIndex == other.slotIndex &&
         slotGeneration == other.slotGeneration;
}

bool SourceSemanticMetadata::operator==(
    const SourceSemanticMetadata& other) const {
  return sourceX == other.sourceX && sourceY == other.sourceY &&
         sourceWidth == other.sourceWidth &&
         sourceHeight == other.sourceHeight &&
         sampledX == other.sampledX && sampledY == other.sampledY &&
         sampledWidth == other.sampledWidth &&
         sampledHeight == other.sampledHeight &&
         coverage == other.coverage &&
         authoritative == other.authoritative &&
         identityStripPresent == other.identityStripPresent &&
         identityCube == other.identityCube &&
         identityRamp == other.identityRamp &&
         identityResolution == other.identityResolution &&
         identityBandHeight == other.identityBandHeight &&
         identityCubeY1 == other.identityCubeY1 &&
         identityCubeY2 == other.identityCubeY2 &&
         identityRampY1 == other.identityRampY1 &&
         identityRampY2 == other.identityRampY2 &&
         colorPrimaries == other.colorPrimaries &&
         transferFunction == other.transferFunction;
}

bool validSourceSemanticMetadata(
    const SourceSemanticMetadata& metadata) {
  if (metadata.sourceWidth == 0 ||
      metadata.sourceWidth > kMaximumDimension ||
      metadata.sourceHeight == 0 ||
      metadata.sourceHeight > kMaximumDimension ||
      !intervalContained(metadata.sourceX,
                         metadata.sourceWidth,
                         metadata.sampledX,
                         metadata.sampledWidth) ||
      !intervalContained(metadata.sourceY,
                         metadata.sourceHeight,
                         metadata.sampledY,
                         metadata.sampledHeight) ||
      !validSemanticIdentifier(metadata.colorPrimaries) ||
      !validSemanticIdentifier(metadata.transferFunction)) {
    return false;
  }

  switch (metadata.coverage) {
    case SourceCoverage::FullSource:
      if (metadata.sampledX != metadata.sourceX ||
          metadata.sampledY != metadata.sourceY ||
          metadata.sampledWidth != metadata.sourceWidth ||
          metadata.sampledHeight != metadata.sourceHeight) {
        return false;
      }
      break;
    case SourceCoverage::PartialSource:
      if (metadata.authoritative) return false;
      break;
    default:
      return false;
  }

  if (!metadata.identityStripPresent) {
    return !metadata.identityCube && !metadata.identityRamp &&
           metadata.identityResolution == 0 &&
           metadata.identityBandHeight == 0 &&
           metadata.identityCubeY1 == 0 &&
           metadata.identityCubeY2 == 0 &&
           metadata.identityRampY1 == 0 &&
           metadata.identityRampY2 == 0;
  }
  if ((!metadata.identityCube && !metadata.identityRamp) ||
      metadata.identityResolution < 2 ||
      metadata.identityResolution > kMaximumIdentityResolution ||
      metadata.identityBandHeight == 0 ||
      metadata.identityBandHeight > metadata.sourceHeight ||
      !validIdentityRange(metadata.identityCube,
                          metadata.identityCubeY1,
                          metadata.identityCubeY2,
                          metadata.sourceY,
                          metadata.sourceHeight,
                          metadata.identityBandHeight) ||
      !validIdentityRange(metadata.identityRamp,
                          metadata.identityRampY1,
                          metadata.identityRampY2,
                          metadata.sourceY,
                          metadata.sourceHeight,
                          metadata.identityBandHeight)) {
    return false;
  }
  if (metadata.identityCube && metadata.identityRamp) {
    const bool overlap =
        metadata.identityCubeY1 < metadata.identityRampY2 &&
        metadata.identityRampY1 < metadata.identityCubeY2;
    if (overlap ||
        metadata.identityBandHeight > metadata.sourceHeight / 2) {
      return false;
    }
  }
  return true;
}

struct BrokerState::Impl {
  struct Slot {
    SlotState state = SlotState::Free;
    uint64_t lastSlotGeneration = 0;
    Publication publication{};
    std::deque<PublicationKey> retiredHistory;
  };

  struct Producer {
    uint64_t generation = 0;
    uint64_t deviceRegistryId = 0;
    uint64_t lastSequence = 0;
    std::array<Slot, kMaximumSlots> slots{};
    std::deque<ProducerReleaseEvent> releases;
    uint64_t nextReleaseOrdinal = 0;
    uint64_t lastDeliveredReleaseOrdinal = 0;
    uint64_t acknowledgedReleaseOrdinal = 0;
    bool releaseJournalOverflowed = false;
  };

  struct Session {
    ViewerRegistration viewer{};
    std::map<std::string, Producer> producers;
    uint64_t retainedBytes = 0;
  };

  std::map<Capability, Session> sessions;
  uint64_t retainedBytes = 0;

  void releaseSlot(Session& session,
                   Producer& producer,
                   Slot& slot,
                   std::vector<PublicationKey>* released) {
    if (slot.state == SlotState::Free) return;
    const PublicationKey releasedKey = slot.publication.key;
    if (released) released->push_back(releasedKey);
    if (producer.nextReleaseOrdinal ==
            std::numeric_limits<uint64_t>::max() ||
        producer.releases.size() >= kMaximumProducerReleaseEvents) {
      producer.releaseJournalOverflowed = true;
    } else {
      ProducerReleaseEvent event{};
      event.ordinal = ++producer.nextReleaseOrdinal;
      event.key = slot.publication.key;
      producer.releases.push_back(std::move(event));
    }
    const uint64_t bytes = slot.publication.byteSize;
    session.retainedBytes =
        bytes <= session.retainedBytes ? session.retainedBytes - bytes : 0;
    retainedBytes = bytes <= retainedBytes ? retainedBytes - bytes : 0;
    slot.state = SlotState::Free;
    slot.publication = Publication{};
    slot.retiredHistory.push_back(releasedKey);
    if (slot.retiredHistory.size() >
        kMaximumProducerReleaseEvents) {
      slot.retiredHistory.pop_front();
    }
  }

  void releaseProducer(Session& session,
                       Producer& producer,
                       std::vector<PublicationKey>* released) {
    for (Slot& slot : producer.slots) {
      releaseSlot(session, producer, slot, released);
    }
  }
};

BrokerState::BrokerState() : impl_(std::make_unique<Impl>()) {}
BrokerState::~BrokerState() = default;
BrokerState::BrokerState(BrokerState&&) noexcept = default;
BrokerState& BrokerState::operator=(BrokerState&&) noexcept = default;

TransitionResult BrokerState::registerViewer(
    const ViewerRegistration& registration) {
  TransitionResult result{};
  if (registration.protocolMajor != kProtocolMajor ||
      registration.protocolMinor > kProtocolMinor) {
    result.code = ResultCode::ProtocolMismatch;
    return result;
  }
  if (!validViewer(registration)) {
    result.code = ResultCode::InvalidRegistration;
    return result;
  }

  auto it = impl_->sessions.find(registration.capability);
  if (it == impl_->sessions.end()) {
    if (impl_->sessions.size() >= kMaximumSessions) {
      result.code = ResultCode::ResourceLimit;
      return result;
    }
    Impl::Session session{};
    session.viewer = registration;
    impl_->sessions.emplace(registration.capability, std::move(session));
    result.code = ResultCode::Accepted;
    return result;
  }

  if (registration.viewerGeneration < it->second.viewer.viewerGeneration) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (registration.viewerGeneration == it->second.viewer.viewerGeneration) {
    result.code =
        sameViewer(registration, it->second.viewer)
            ? ResultCode::Accepted
            : ResultCode::InvalidRegistration;
    return result;
  }

  for (auto& producerEntry : it->second.producers) {
    impl_->releaseProducer(it->second, producerEntry.second, &result.released);
  }
  it->second = Impl::Session{};
  it->second.viewer = registration;
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::registerProducer(
    const ProducerRegistration& registration) {
  TransitionResult result{};
  if (!validCapability(registration.capability) ||
      registration.senderId.empty() || registration.senderId.size() > 256 ||
      registration.senderGeneration == 0 ||
      registration.deviceRegistryId == 0) {
    result.code = ResultCode::InvalidRegistration;
    return result;
  }
  auto sessionIt = impl_->sessions.find(registration.capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  Impl::Session& session = sessionIt->second;
  if (registration.deviceRegistryId != session.viewer.deviceRegistryId) {
    result.code = ResultCode::DeviceMismatch;
    return result;
  }
  auto producerIt = session.producers.find(registration.senderId);
  if (producerIt == session.producers.end()) {
    if (session.producers.size() >= kMaximumProducersPerSession) {
      result.code = ResultCode::ResourceLimit;
      return result;
    }
    Impl::Producer producer{};
    producer.generation = registration.senderGeneration;
    producer.deviceRegistryId = registration.deviceRegistryId;
    session.producers.emplace(registration.senderId, std::move(producer));
    result.code = ResultCode::Accepted;
    return result;
  }
  Impl::Producer& producer = producerIt->second;
  if (registration.senderGeneration < producer.generation) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (registration.senderGeneration == producer.generation) {
    result.code =
        registration.deviceRegistryId == producer.deviceRegistryId
            ? ResultCode::Accepted
            : ResultCode::DeviceMismatch;
    return result;
  }
  impl_->releaseProducer(session, producer, &result.released);
  producer = Impl::Producer{};
  producer.generation = registration.senderGeneration;
  producer.deviceRegistryId = registration.deviceRegistryId;
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::publish(const Publication& publication) {
  TransitionResult result{};
  auto sessionIt = impl_->sessions.find(publication.capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  Impl::Session& session = sessionIt->second;
  auto producerIt = session.producers.find(publication.key.senderId);
  if (producerIt == session.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  Impl::Producer& producer = producerIt->second;
  if (publication.key.senderGeneration != producer.generation) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (publication.deviceRegistryId != producer.deviceRegistryId ||
      publication.deviceRegistryId != session.viewer.deviceRegistryId) {
    result.code = ResultCode::DeviceMismatch;
    return result;
  }
  const uint32_t acceptedFormat = formatMask(publication.pixelFormat);
  if (acceptedFormat == 0 ||
      (session.viewer.pixelFormatMask & acceptedFormat) == 0) {
    result.code = ResultCode::UnsupportedFormat;
    return result;
  }
  const uint64_t elementBytes = publication.pixelFormat == 0 ? 8ull : 16ull;
  const bool dimensionsValid =
      publication.width > 0 &&
      publication.width <= session.viewer.maximumWidth &&
      publication.height > 0 &&
      publication.height <= session.viewer.maximumHeight;
  const bool multiplicationSafe =
      publication.height == 0 ||
      publication.bytesPerRow <=
          std::numeric_limits<uint64_t>::max() / publication.height;
  const bool sizeValid =
      dimensionsValid &&
      publication.bytesPerRow >=
          static_cast<uint64_t>(publication.width) * elementBytes &&
      multiplicationSafe &&
      publication.byteSize >=
          publication.bytesPerRow * publication.height &&
      publication.byteSize <= session.viewer.maximumSurfaceBytes &&
      publication.readyValue != 0 && publication.contentHash != 0;
  if (!sizeValid ||
      !validSourceSemanticMetadata(publication.semantics) ||
      publication.key.sequence == 0 ||
      publication.key.slotGeneration == 0 ||
      publication.key.slotIndex >= session.viewer.maximumSlots) {
    result.code = ResultCode::InvalidPublication;
    return result;
  }
  Impl::Slot& slot = producer.slots[publication.key.slotIndex];
  if (slot.state != SlotState::Free &&
      samePublication(slot.publication, publication)) {
    result.code = ResultCode::Accepted;
    return result;
  }
  if (producer.releaseJournalOverflowed ||
      producer.releases.size() >= kMaximumProducerReleaseEvents) {
    result.code = ResultCode::ResourceLimit;
    return result;
  }
  if (publication.key.sequence <= producer.lastSequence) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (slot.state != SlotState::Free) {
    result.code = ResultCode::SlotBusy;
    return result;
  }
  if (publication.key.slotGeneration <= slot.lastSlotGeneration) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (publication.byteSize >
          session.viewer.maximumRetainedBytes - session.retainedBytes ||
      publication.byteSize > kMaximumRetainedBytes - impl_->retainedBytes) {
    result.code = ResultCode::ResourceLimit;
    return result;
  }

  slot.state = SlotState::Published;
  slot.lastSlotGeneration = publication.key.slotGeneration;
  slot.publication = publication;
  producer.lastSequence = publication.key.sequence;
  session.retainedBytes += publication.byteSize;
  impl_->retainedBytes += publication.byteSize;
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::acquireLatest(const Capability& capability,
                                            const std::string& senderId,
                                            uint64_t afterSequence,
                                            Publication* outPublication) {
  TransitionResult result{};
  if (outPublication) *outPublication = Publication{};
  auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  auto producerIt = sessionIt->second.producers.find(senderId);
  if (producerIt == sessionIt->second.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  Impl::Slot* latest = nullptr;
  for (Impl::Slot& slot : producerIt->second.slots) {
    if (slot.state != SlotState::Published ||
        slot.publication.key.sequence <= afterSequence) {
      continue;
    }
    if (!latest ||
        slot.publication.key.sequence > latest->publication.key.sequence) {
      latest = &slot;
    }
  }
  if (!latest) {
    result.code = ResultCode::NoNewPublication;
    return result;
  }
  if (outPublication) *outPublication = latest->publication;
  const PublicationKey selectedKey = latest->publication.key;
  for (Impl::Slot& slot : producerIt->second.slots) {
    if (slot.state == SlotState::Published &&
        !(slot.publication.key == selectedKey) &&
        slot.publication.key.sequence < selectedKey.sequence) {
      impl_->releaseSlot(
          sessionIt->second, producerIt->second, slot, &result.released);
    }
  }
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::acknowledge(
    const Capability& capability,
    const PublicationKey& key,
    AcknowledgementState state) {
  TransitionResult result{};
  auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  auto producerIt = sessionIt->second.producers.find(key.senderId);
  if (producerIt == sessionIt->second.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  if (key.senderGeneration != producerIt->second.generation ||
      key.slotIndex >= sessionIt->second.viewer.maximumSlots) {
    result.code = ResultCode::Stale;
    return result;
  }
  Impl::Slot& slot = producerIt->second.slots[key.slotIndex];
  if (state == AcknowledgementState::Retired &&
      std::find(
          slot.retiredHistory.begin(),
          slot.retiredHistory.end(),
          key) != slot.retiredHistory.end()) {
    result.code = ResultCode::Accepted;
    return result;
  }
  if (slot.state == SlotState::Free) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (!(slot.publication.key == key)) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (state == AcknowledgementState::Acquired) {
    if (slot.state == SlotState::Acquired) {
      result.code = ResultCode::Accepted;
      return result;
    }
    if (slot.state != SlotState::Published) {
      result.code = ResultCode::InvalidTransition;
      return result;
    }
    slot.state = SlotState::Acquired;
    result.code = ResultCode::Accepted;
    return result;
  }
  if (slot.state != SlotState::Acquired) {
    result.code = ResultCode::InvalidTransition;
    return result;
  }
  impl_->releaseSlot(
      sessionIt->second, producerIt->second, slot, &result.released);
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::fetchProducerReleases(
    const Capability& capability,
    const std::string& senderId,
    uint64_t senderGeneration,
    uint64_t afterOrdinal,
    size_t maximumEvents,
    ProducerReleaseBatch* outBatch) {
  TransitionResult result{};
  if (outBatch) *outBatch = ProducerReleaseBatch{};
  if (maximumEvents == 0 ||
      maximumEvents > kMaximumProducerReleaseEvents) {
    result.code = ResultCode::InvalidRegistration;
    return result;
  }
  auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  auto producerIt = sessionIt->second.producers.find(senderId);
  if (producerIt == sessionIt->second.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  Impl::Producer& producer = producerIt->second;
  if (senderGeneration != producer.generation) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (producer.releaseJournalOverflowed) {
    result.code = ResultCode::ResourceLimit;
    return result;
  }
  if (afterOrdinal < producer.acknowledgedReleaseOrdinal ||
      afterOrdinal > producer.nextReleaseOrdinal) {
    result.code = ResultCode::Stale;
    return result;
  }

  ProducerReleaseBatch batch{};
  batch.capability = capability;
  batch.senderId = senderId;
  batch.senderGeneration = senderGeneration;
  batch.throughOrdinal = afterOrdinal;
  for (const ProducerReleaseEvent& event : producer.releases) {
    if (event.ordinal <= afterOrdinal) continue;
    batch.events.push_back(event);
    batch.throughOrdinal = event.ordinal;
    if (batch.events.size() >= maximumEvents) break;
  }
  if (batch.events.empty()) {
    if (outBatch) *outBatch = std::move(batch);
    result.code = ResultCode::NoNewPublication;
    return result;
  }
  producer.lastDeliveredReleaseOrdinal =
      std::max(producer.lastDeliveredReleaseOrdinal,
               batch.throughOrdinal);
  if (outBatch) *outBatch = std::move(batch);
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::acknowledgeProducerReleases(
    const Capability& capability,
    const std::string& senderId,
    uint64_t senderGeneration,
    uint64_t throughOrdinal) {
  TransitionResult result{};
  auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  auto producerIt = sessionIt->second.producers.find(senderId);
  if (producerIt == sessionIt->second.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  Impl::Producer& producer = producerIt->second;
  if (senderGeneration != producer.generation ||
      throughOrdinal < producer.acknowledgedReleaseOrdinal) {
    result.code = ResultCode::Stale;
    return result;
  }
  if (throughOrdinal == producer.acknowledgedReleaseOrdinal) {
    result.code = ResultCode::Accepted;
    return result;
  }
  if (throughOrdinal > producer.lastDeliveredReleaseOrdinal) {
    result.code = ResultCode::InvalidTransition;
    return result;
  }
  while (!producer.releases.empty() &&
         producer.releases.front().ordinal <= throughOrdinal) {
    producer.releases.pop_front();
  }
  producer.acknowledgedReleaseOrdinal = throughOrdinal;
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::disconnectViewer(const Capability& capability) {
  TransitionResult result{};
  auto it = impl_->sessions.find(capability);
  if (it == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  for (auto& producerEntry : it->second.producers) {
    impl_->releaseProducer(it->second, producerEntry.second, &result.released);
  }
  impl_->sessions.erase(it);
  result.code = ResultCode::Accepted;
  return result;
}

TransitionResult BrokerState::disconnectProducer(
    const Capability& capability,
    const std::string& senderId,
    uint64_t senderGeneration) {
  TransitionResult result{};
  auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  auto producerIt = sessionIt->second.producers.find(senderId);
  if (producerIt == sessionIt->second.producers.end()) {
    result.code = ResultCode::ProducerMissing;
    return result;
  }
  if (senderGeneration != producerIt->second.generation) {
    result.code = ResultCode::Stale;
    return result;
  }
  impl_->releaseProducer(
      sessionIt->second, producerIt->second, &result.released);
  sessionIt->second.producers.erase(producerIt);
  result.code = ResultCode::Accepted;
  return result;
}

size_t BrokerState::sessionCount() const {
  return impl_->sessions.size();
}

size_t BrokerState::producerCount(const Capability& capability) const {
  const auto it = impl_->sessions.find(capability);
  return it == impl_->sessions.end() ? 0 : it->second.producers.size();
}

size_t BrokerState::livePublicationCount(const Capability& capability) const {
  const auto it = impl_->sessions.find(capability);
  if (it == impl_->sessions.end()) return 0;
  size_t count = 0;
  for (const auto& producerEntry : it->second.producers) {
    for (const Impl::Slot& slot : producerEntry.second.slots) {
      if (slot.state != SlotState::Free) ++count;
    }
  }
  return count;
}

uint64_t BrokerState::retainedBytes() const {
  return impl_->retainedBytes;
}

size_t BrokerState::pendingProducerReleaseCount(
    const Capability& capability,
    const std::string& senderId) const {
  const auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) return 0;
  const auto producerIt = sessionIt->second.producers.find(senderId);
  return producerIt == sessionIt->second.producers.end()
             ? 0
             : producerIt->second.releases.size();
}

std::optional<SlotState> BrokerState::slotState(
    const Capability& capability,
    const PublicationKey& key) const {
  const auto sessionIt = impl_->sessions.find(capability);
  if (sessionIt == impl_->sessions.end()) return std::nullopt;
  const auto producerIt = sessionIt->second.producers.find(key.senderId);
  if (producerIt == sessionIt->second.producers.end() ||
      key.slotIndex >= sessionIt->second.viewer.maximumSlots) {
    return std::nullopt;
  }
  const Impl::Slot& slot = producerIt->second.slots[key.slotIndex];
  if (slot.state == SlotState::Free || !(slot.publication.key == key)) {
    return std::nullopt;
  }
  return slot.state;
}

}  // namespace ChromaspaceSourceExchange
