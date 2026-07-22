#pragma once

#include "ChromaspaceSourceProducerState.h"

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace ChromaspaceSourceExchange {

enum class ProducerClientPhase {
  Disconnected,
  Ready,
  Failed,
};

struct ProducerClientTransition {
  ResultCode code = ResultCode::InvalidTransition;
  bool leaseChanged = false;
  std::vector<PublicationKey> abandoned;
  std::vector<PublicationKey> released;
  uint64_t releaseAcknowledgementOrdinal = 0;

  bool accepted() const { return code == ResultCode::Accepted; }
};

// Portable lifecycle core for the in-host producer client. The Objective-C++
// adapter owns Metal/XPC/process objects, but all reservation, exact retry,
// release ordering, and fail-closed decisions cross this one seam.
class ProducerClientState {
 public:
  ProducerClientTransition installLease(
      const ProducerLeaseSnapshot& lease);
  ProducerClientTransition invalidateTransport();

  ResultCode reserve(const ProducerPublicationSpec& spec,
                     Publication* outPublication);
  std::optional<Publication> publication(
      const PublicationKey& key) const;
  ResultCode markSendReady(const PublicationKey& key);
  std::optional<Publication> beginNextPublish();
  ResultCode publishTransportFailed(const PublicationKey& key);
  ResultCode publishAccepted(const PublicationKey& key);
  ResultCode cancel(const PublicationKey& key);

  ProducerClientTransition applyReleaseBatch(
      const ProducerReleaseBatch& batch);
  ResultCode releaseAcknowledgementAccepted(uint64_t throughOrdinal);

  ProducerClientPhase phase() const { return phase_; }
  uint64_t releaseFetchCursor() const { return appliedReleaseOrdinal_; }
  uint64_t acknowledgedReleaseOrdinal() const {
    return acknowledgedReleaseOrdinal_;
  }
  size_t sendReadyCount() const;
  size_t livePublicationCount() const {
    return producer_.livePublicationCount();
  }

 private:
  enum class SendState {
    Reserved,
    Ready,
    InFlight,
  };

  struct SendRecord {
    bool occupied = false;
    Publication publication;
    SendState state = SendState::Reserved;
  };

  SendRecord* matchingRecord(const PublicationKey& key);
  const SendRecord* matchingRecord(const PublicationKey& key) const;
  void clearSendRecord(uint32_t slotIndex);
  void clearSendRecords();

  ProducerState producer_;
  ProducerClientPhase phase_ = ProducerClientPhase::Disconnected;
  std::array<SendRecord, kMaximumSlots> sends_{};
  uint64_t appliedReleaseOrdinal_ = 0;
  uint64_t acknowledgedReleaseOrdinal_ = 0;
};

}  // namespace ChromaspaceSourceExchange
