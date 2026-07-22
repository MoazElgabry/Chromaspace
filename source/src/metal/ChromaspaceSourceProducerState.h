#pragma once

#include "ChromaspaceSourceExchangeState.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ChromaspaceSourceExchange {

struct ProducerLeaseSnapshot {
  uint32_t protocolMajor = kProtocolMajor;
  uint32_t protocolMinor = kProtocolMinor;
  Capability capability{};
  uint64_t viewerGeneration = 0;
  std::string senderId;
  uint64_t senderGeneration = 0;
  uint64_t deviceRegistryId = 0;
  uint32_t pixelFormatMask = 0;
  uint32_t maximumWidth = 0;
  uint32_t maximumHeight = 0;
  uint64_t maximumSurfaceBytes = 0;
  uint64_t maximumRetainedBytes = 0;
  uint32_t maximumSlots = 0;
  bool supportsSharedEvents = false;
};

struct ProducerPublicationSpec {
  uint64_t sequence = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  uint64_t bytesPerRow = 0;
  uint64_t byteSize = 0;
  uint64_t readyValue = 0;
  uint64_t contentHash = 0;
  SourceSemanticMetadata semantics;
};

enum class ProducerSlotState {
  Free,
  Pending,
  Published,
};

struct ProducerTransition {
  ResultCode code = ResultCode::InvalidTransition;
  bool leaseChanged = false;
  std::vector<PublicationKey> abandoned;

  bool accepted() const { return code == ResultCode::Accepted; }
};

// Owns the producer-side publication lifecycle. Metal resources and XPC
// packets remain adapter details; retries cross this seam using the exact
// Publication returned by reserve().
class ProducerState {
 public:
  ProducerTransition installLease(const ProducerLeaseSnapshot& lease);
  ProducerTransition invalidateLease();

  ResultCode reserve(const ProducerPublicationSpec& spec,
                     Publication* outPublication);
  ResultCode markPublished(const PublicationKey& key);
  ResultCode cancelPending(const PublicationKey& key);
  ResultCode release(const PublicationKey& key);

  std::optional<ProducerLeaseSnapshot> lease() const;
  std::optional<ProducerSlotState> slotState(uint32_t slotIndex) const;
  std::optional<Publication> publication(uint32_t slotIndex) const;
  size_t livePublicationCount() const;

 private:
  struct Slot {
    ProducerSlotState state = ProducerSlotState::Free;
    uint64_t lastSlotGeneration = 0;
    Publication current{};
    std::optional<PublicationKey> lastReleased;
    std::optional<PublicationKey> lastCanceled;
  };

  void abandonLive(std::vector<PublicationKey>* abandoned);

  std::optional<ProducerLeaseSnapshot> lease_;
  std::array<Slot, kMaximumSlots> slots_{};
  uint64_t lastSequence_ = 0;
  uint32_t nextSlot_ = 0;
};

}  // namespace ChromaspaceSourceExchange
