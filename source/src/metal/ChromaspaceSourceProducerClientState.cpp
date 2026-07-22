#include "ChromaspaceSourceProducerClientState.h"

#include <algorithm>
#include <utility>

namespace ChromaspaceSourceExchange {

ProducerClientState::SendRecord* ProducerClientState::matchingRecord(
    const PublicationKey& key) {
  if (key.slotIndex >= sends_.size()) return nullptr;
  SendRecord& record = sends_[key.slotIndex];
  if (!record.occupied || !(record.publication.key == key)) {
    return nullptr;
  }
  return &record;
}

const ProducerClientState::SendRecord*
ProducerClientState::matchingRecord(const PublicationKey& key) const {
  if (key.slotIndex >= sends_.size()) return nullptr;
  const SendRecord& record = sends_[key.slotIndex];
  if (!record.occupied || !(record.publication.key == key)) {
    return nullptr;
  }
  return &record;
}

void ProducerClientState::clearSendRecord(uint32_t slotIndex) {
  if (slotIndex < sends_.size()) sends_[slotIndex] = SendRecord{};
}

void ProducerClientState::clearSendRecords() {
  for (SendRecord& record : sends_) record = SendRecord{};
}

ProducerClientTransition ProducerClientState::installLease(
    const ProducerLeaseSnapshot& lease) {
  ProducerClientTransition result{};
  ProducerTransition transition = producer_.installLease(lease);
  result.code = transition.code;
  result.leaseChanged = transition.leaseChanged;
  result.abandoned = std::move(transition.abandoned);
  if (!transition.accepted()) return result;
  if (transition.leaseChanged) {
    clearSendRecords();
    appliedReleaseOrdinal_ = 0;
    acknowledgedReleaseOrdinal_ = 0;
  }
  phase_ = ProducerClientPhase::Ready;
  return result;
}

ProducerClientTransition ProducerClientState::invalidateTransport() {
  ProducerClientTransition result{};
  ProducerTransition transition = producer_.invalidateLease();
  result.code = transition.code;
  result.abandoned = std::move(transition.abandoned);
  clearSendRecords();
  appliedReleaseOrdinal_ = 0;
  acknowledgedReleaseOrdinal_ = 0;
  phase_ = ProducerClientPhase::Failed;
  return result;
}

ResultCode ProducerClientState::reserve(
    const ProducerPublicationSpec& spec,
    Publication* outPublication) {
  if (outPublication) *outPublication = Publication{};
  if (phase_ != ProducerClientPhase::Ready) {
    return ResultCode::SessionMissing;
  }
  Publication publication{};
  ResultCode code = producer_.reserve(spec, &publication);
  if (code != ResultCode::Accepted) return code;
  if (publication.key.slotIndex >= sends_.size()) {
    (void)producer_.cancelPending(publication.key);
    return ResultCode::InvalidTransition;
  }
  SendRecord& record = sends_[publication.key.slotIndex];
  if (record.occupied) {
    (void)producer_.cancelPending(publication.key);
    return ResultCode::InvalidTransition;
  }
  record.occupied = true;
  record.publication = std::move(publication);
  record.state = SendState::Reserved;
  if (outPublication) *outPublication = record.publication;
  return ResultCode::Accepted;
}

std::optional<Publication> ProducerClientState::publication(
    const PublicationKey& key) const {
  const SendRecord* record = matchingRecord(key);
  if (phase_ != ProducerClientPhase::Ready || record == nullptr) {
    return std::nullopt;
  }
  return record->publication;
}

ResultCode ProducerClientState::markSendReady(
    const PublicationKey& key) {
  SendRecord* record = matchingRecord(key);
  if (phase_ != ProducerClientPhase::Ready || record == nullptr ||
      record->state != SendState::Reserved) {
    return ResultCode::Stale;
  }
  record->state = SendState::Ready;
  return ResultCode::Accepted;
}

std::optional<Publication> ProducerClientState::beginNextPublish() {
  if (phase_ != ProducerClientPhase::Ready) return std::nullopt;
  SendRecord* earliest = nullptr;
  for (SendRecord& record : sends_) {
    if (!record.occupied) continue;
    if (earliest == nullptr ||
        record.publication.key.sequence <
            earliest->publication.key.sequence) {
      earliest = &record;
    }
  }
  // A later GPU completion must not overtake an earlier reserved or in-flight
  // generation even though fixed storage is indexed by ring slot.
  if (earliest == nullptr || earliest->state != SendState::Ready) {
    return std::nullopt;
  }
  earliest->state = SendState::InFlight;
  return earliest->publication;
}

ResultCode ProducerClientState::publishTransportFailed(
    const PublicationKey& key) {
  SendRecord* record = matchingRecord(key);
  if (phase_ != ProducerClientPhase::Ready || record == nullptr ||
      record->state != SendState::InFlight) {
    return ResultCode::Stale;
  }
  record->state = SendState::Ready;
  return ResultCode::Accepted;
}

ResultCode ProducerClientState::publishAccepted(
    const PublicationKey& key) {
  SendRecord* record = matchingRecord(key);
  if (phase_ != ProducerClientPhase::Ready || record == nullptr ||
      record->state != SendState::InFlight) {
    return ResultCode::Stale;
  }
  ResultCode code = producer_.markPublished(key);
  if (code == ResultCode::Accepted) clearSendRecord(key.slotIndex);
  return code;
}

ResultCode ProducerClientState::cancel(const PublicationKey& key) {
  SendRecord* record = matchingRecord(key);
  if (record == nullptr || record->state == SendState::InFlight) {
    return ResultCode::Stale;
  }
  ResultCode code = producer_.cancelPending(key);
  if (code == ResultCode::Accepted) clearSendRecord(key.slotIndex);
  return code;
}

ProducerClientTransition ProducerClientState::applyReleaseBatch(
    const ProducerReleaseBatch& batch) {
  ProducerClientTransition result{};
  const std::optional<ProducerLeaseSnapshot> activeLease =
      producer_.lease();
  if (phase_ != ProducerClientPhase::Ready || !activeLease.has_value() ||
      batch.capability != activeLease->capability ||
      batch.senderId != activeLease->senderId ||
      batch.senderGeneration != activeLease->senderGeneration ||
      batch.events.empty() ||
      batch.throughOrdinal == 0 ||
      batch.throughOrdinal != batch.events.back().ordinal) {
    result.code = ResultCode::InvalidTransition;
    return result;
  }
  if (batch.throughOrdinal <= appliedReleaseOrdinal_) {
    result.code = ResultCode::Accepted;
    result.releaseAcknowledgementOrdinal = batch.throughOrdinal;
    return result;
  }

  uint64_t expectedOrdinal = appliedReleaseOrdinal_ + 1;
  std::vector<const ProducerReleaseEvent*> newEvents;
  for (const ProducerReleaseEvent& event : batch.events) {
    if (event.ordinal <= appliedReleaseOrdinal_) continue;
    const std::optional<Publication> live =
        producer_.publication(event.key.slotIndex);
    if (event.ordinal != expectedOrdinal || !live.has_value() ||
        !(live->key == event.key)) {
      result.code = ResultCode::Stale;
      return result;
    }
    newEvents.push_back(&event);
    ++expectedOrdinal;
  }
  if (newEvents.empty() ||
      newEvents.back()->ordinal != batch.throughOrdinal) {
    result.code = ResultCode::InvalidTransition;
    return result;
  }

  for (const ProducerReleaseEvent* event : newEvents) {
    ResultCode code = producer_.release(event->key);
    if (code != ResultCode::Accepted) {
      result.code = code;
      return result;
    }
    clearSendRecord(event->key.slotIndex);
    result.released.push_back(event->key);
  }
  appliedReleaseOrdinal_ = batch.throughOrdinal;
  result.releaseAcknowledgementOrdinal = batch.throughOrdinal;
  result.code = ResultCode::Accepted;
  return result;
}

ResultCode ProducerClientState::releaseAcknowledgementAccepted(
    uint64_t throughOrdinal) {
  if (phase_ != ProducerClientPhase::Ready ||
      throughOrdinal < acknowledgedReleaseOrdinal_ ||
      throughOrdinal > appliedReleaseOrdinal_) {
    return ResultCode::Stale;
  }
  acknowledgedReleaseOrdinal_ = throughOrdinal;
  return ResultCode::Accepted;
}

size_t ProducerClientState::sendReadyCount() const {
  return static_cast<size_t>(std::count_if(
      sends_.begin(), sends_.end(), [](const SendRecord& record) {
        return record.occupied && record.state == SendState::Ready;
      }));
}

}  // namespace ChromaspaceSourceExchange
