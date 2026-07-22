#include "ChromaspaceSourceProducerState.h"

#include <algorithm>
#include <limits>

namespace ChromaspaceSourceExchange {
namespace {

bool validCapability(const Capability& capability) {
  return std::any_of(capability.begin(), capability.end(),
                     [](uint8_t byte) { return byte != 0; });
}

bool validLease(const ProducerLeaseSnapshot& lease) {
  constexpr uint32_t knownFormats =
      kPixelFormatRGBA16Float | kPixelFormatRGBA32Float;
  return lease.protocolMajor == kProtocolMajor &&
         lease.protocolMinor <= kProtocolMinor &&
         validCapability(lease.capability) &&
         lease.viewerGeneration != 0 &&
         !lease.senderId.empty() && lease.senderId.size() <= 256 &&
         lease.senderGeneration != 0 &&
         lease.deviceRegistryId != 0 &&
         lease.pixelFormatMask != 0 &&
         (lease.pixelFormatMask & ~knownFormats) == 0 &&
         lease.maximumWidth > 0 &&
         lease.maximumWidth <= kMaximumDimension &&
         lease.maximumHeight > 0 &&
         lease.maximumHeight <= kMaximumDimension &&
         lease.maximumSurfaceBytes > 0 &&
         lease.maximumSurfaceBytes <= kMaximumSurfaceBytes &&
         lease.maximumRetainedBytes >= lease.maximumSurfaceBytes &&
         lease.maximumRetainedBytes <= kMaximumRetainedBytes &&
         lease.maximumSlots > 0 &&
         lease.maximumSlots <= kMaximumSlots &&
         lease.supportsSharedEvents;
}

bool sameLease(const ProducerLeaseSnapshot& a,
               const ProducerLeaseSnapshot& b) {
  return a.protocolMajor == b.protocolMajor &&
         a.protocolMinor == b.protocolMinor &&
         a.capability == b.capability &&
         a.viewerGeneration == b.viewerGeneration &&
         a.senderId == b.senderId &&
         a.senderGeneration == b.senderGeneration &&
         a.deviceRegistryId == b.deviceRegistryId &&
         a.pixelFormatMask == b.pixelFormatMask &&
         a.maximumWidth == b.maximumWidth &&
         a.maximumHeight == b.maximumHeight &&
         a.maximumSurfaceBytes == b.maximumSurfaceBytes &&
         a.maximumRetainedBytes == b.maximumRetainedBytes &&
         a.maximumSlots == b.maximumSlots &&
         a.supportsSharedEvents == b.supportsSharedEvents;
}

bool sameLeaseIdentity(const ProducerLeaseSnapshot& a,
                       const ProducerLeaseSnapshot& b) {
  return a.capability == b.capability &&
         a.viewerGeneration == b.viewerGeneration &&
         a.senderId == b.senderId &&
         a.senderGeneration == b.senderGeneration;
}

uint32_t pixelFormatMask(uint32_t pixelFormat) {
  if (pixelFormat == 0) return kPixelFormatRGBA16Float;
  if (pixelFormat == 1) return kPixelFormatRGBA32Float;
  return 0;
}

bool validSpec(const ProducerLeaseSnapshot& lease,
               const ProducerPublicationSpec& spec) {
  const uint32_t formatMask = pixelFormatMask(spec.pixelFormat);
  if (formatMask == 0 || (lease.pixelFormatMask & formatMask) == 0 ||
      spec.sequence == 0 ||
      spec.width == 0 || spec.width > lease.maximumWidth ||
      spec.height == 0 || spec.height > lease.maximumHeight ||
      spec.readyValue == 0 || spec.contentHash == 0 ||
      !validSourceSemanticMetadata(spec.semantics)) {
    return false;
  }
  const uint64_t elementBytes = spec.pixelFormat == 0 ? 8ull : 16ull;
  if (spec.width >
      std::numeric_limits<uint64_t>::max() / elementBytes) {
    return false;
  }
  const uint64_t minimumRowBytes =
      static_cast<uint64_t>(spec.width) * elementBytes;
  if (spec.bytesPerRow < minimumRowBytes ||
      spec.bytesPerRow >
          std::numeric_limits<uint64_t>::max() / spec.height) {
    return false;
  }
  const uint64_t minimumBytes =
      spec.bytesPerRow * static_cast<uint64_t>(spec.height);
  return spec.byteSize >= minimumBytes &&
         spec.byteSize <= lease.maximumSurfaceBytes &&
         spec.byteSize <= lease.maximumRetainedBytes;
}

}  // namespace

ProducerTransition ProducerState::installLease(
    const ProducerLeaseSnapshot& lease) {
  ProducerTransition result{};
  if (!validLease(lease)) {
    result.code = lease.protocolMajor != kProtocolMajor ||
                          lease.protocolMinor > kProtocolMinor
                      ? ResultCode::ProtocolMismatch
                      : ResultCode::InvalidRegistration;
    return result;
  }
  if (lease_) {
    if (sameLease(*lease_, lease)) {
      result.code = ResultCode::Accepted;
      return result;
    }
    if (sameLeaseIdentity(*lease_, lease)) {
      result.code = ResultCode::InvalidRegistration;
      return result;
    }
  }
  abandonLive(&result.abandoned);
  lease_ = lease;
  result.leaseChanged = true;
  result.code = ResultCode::Accepted;
  return result;
}

ProducerTransition ProducerState::invalidateLease() {
  ProducerTransition result{};
  abandonLive(&result.abandoned);
  lease_.reset();
  result.code = ResultCode::Accepted;
  return result;
}

ResultCode ProducerState::reserve(const ProducerPublicationSpec& spec,
                                  Publication* outPublication) {
  if (outPublication) *outPublication = Publication{};
  if (!lease_) return ResultCode::SessionMissing;
  if (!validSpec(*lease_, spec)) return ResultCode::InvalidPublication;
  if (spec.sequence <= lastSequence_) return ResultCode::Stale;

  uint32_t selected = kMaximumSlots;
  for (uint32_t offset = 0; offset < lease_->maximumSlots; ++offset) {
    const uint32_t candidate =
        (nextSlot_ + offset) % lease_->maximumSlots;
    if (slots_[candidate].state == ProducerSlotState::Free) {
      selected = candidate;
      break;
    }
  }
  if (selected == kMaximumSlots) return ResultCode::SlotBusy;
  Slot& slot = slots_[selected];
  if (slot.lastSlotGeneration == std::numeric_limits<uint64_t>::max()) {
    return ResultCode::ResourceLimit;
  }

  Publication value{};
  value.capability = lease_->capability;
  value.key.senderId = lease_->senderId;
  value.key.senderGeneration = lease_->senderGeneration;
  value.key.sequence = spec.sequence;
  value.key.slotIndex = selected;
  value.key.slotGeneration = slot.lastSlotGeneration + 1;
  value.deviceRegistryId = lease_->deviceRegistryId;
  value.width = spec.width;
  value.height = spec.height;
  value.pixelFormat = spec.pixelFormat;
  value.bytesPerRow = spec.bytesPerRow;
  value.byteSize = spec.byteSize;
  value.readyValue = spec.readyValue;
  value.contentHash = spec.contentHash;
  value.semantics = spec.semantics;

  lastSequence_ = value.key.sequence;
  slot.lastSlotGeneration = value.key.slotGeneration;
  slot.state = ProducerSlotState::Pending;
  slot.current = value;
  slot.lastReleased.reset();
  slot.lastCanceled.reset();
  nextSlot_ = (selected + 1) % lease_->maximumSlots;
  if (outPublication) *outPublication = value;
  return ResultCode::Accepted;
}

ResultCode ProducerState::markPublished(const PublicationKey& key) {
  if (key.slotIndex >= kMaximumSlots) return ResultCode::Stale;
  Slot& slot = slots_[key.slotIndex];
  if (slot.state == ProducerSlotState::Pending &&
      slot.current.key == key) {
    slot.state = ProducerSlotState::Published;
    return ResultCode::Accepted;
  }
  if (slot.state == ProducerSlotState::Published &&
      slot.current.key == key) {
    return ResultCode::Accepted;
  }
  return ResultCode::Stale;
}

ResultCode ProducerState::cancelPending(const PublicationKey& key) {
  if (key.slotIndex >= kMaximumSlots) return ResultCode::Stale;
  Slot& slot = slots_[key.slotIndex];
  if (slot.state == ProducerSlotState::Pending &&
      slot.current.key == key) {
    slot.lastCanceled = key;
    slot.state = ProducerSlotState::Free;
    slot.current = Publication{};
    return ResultCode::Accepted;
  }
  if (slot.state == ProducerSlotState::Free && slot.lastCanceled &&
      *slot.lastCanceled == key) {
    return ResultCode::Accepted;
  }
  return ResultCode::Stale;
}

ResultCode ProducerState::release(const PublicationKey& key) {
  if (key.slotIndex >= kMaximumSlots) return ResultCode::Stale;
  Slot& slot = slots_[key.slotIndex];
  if (slot.state != ProducerSlotState::Free &&
      slot.current.key == key) {
    slot.lastReleased = key;
    slot.state = ProducerSlotState::Free;
    slot.current = Publication{};
    return ResultCode::Accepted;
  }
  if (slot.state == ProducerSlotState::Free && slot.lastReleased &&
      *slot.lastReleased == key) {
    return ResultCode::Accepted;
  }
  return ResultCode::Stale;
}

std::optional<ProducerLeaseSnapshot> ProducerState::lease() const {
  return lease_;
}

std::optional<ProducerSlotState> ProducerState::slotState(
    uint32_t slotIndex) const {
  if (slotIndex >= kMaximumSlots) return std::nullopt;
  return slots_[slotIndex].state;
}

std::optional<Publication> ProducerState::publication(
    uint32_t slotIndex) const {
  if (slotIndex >= kMaximumSlots ||
      slots_[slotIndex].state == ProducerSlotState::Free) {
    return std::nullopt;
  }
  return slots_[slotIndex].current;
}

size_t ProducerState::livePublicationCount() const {
  return static_cast<size_t>(std::count_if(
      slots_.begin(), slots_.end(), [](const Slot& slot) {
        return slot.state != ProducerSlotState::Free;
      }));
}

void ProducerState::abandonLive(
    std::vector<PublicationKey>* abandoned) {
  for (Slot& slot : slots_) {
    if (slot.state == ProducerSlotState::Free) continue;
    if (abandoned) abandoned->push_back(slot.current.key);
    slot.state = ProducerSlotState::Free;
    slot.current = Publication{};
    slot.lastReleased.reset();
    slot.lastCanceled.reset();
  }
}

}  // namespace ChromaspaceSourceExchange
