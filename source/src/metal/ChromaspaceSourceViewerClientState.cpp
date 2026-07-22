#include "ChromaspaceSourceViewerClientState.h"

#include <algorithm>
#include <limits>

namespace ChromaspaceSourceExchange {
namespace {

bool capabilityValid(const Capability& capability) {
  return std::any_of(
      capability.begin(), capability.end(),
      [](uint8_t value) { return value != 0; });
}

bool sessionValid(const ViewerSessionSnapshot& session) {
  constexpr uint32_t knownFormats =
      kPixelFormatRGBA16Float | kPixelFormatRGBA32Float;
  return session.protocolMajor == kProtocolMajor &&
         session.protocolMinor <= kProtocolMinor &&
         capabilityValid(session.capability) &&
         session.viewerGeneration != 0 &&
         !session.senderId.empty() &&
         session.senderId.size() <= 256 &&
         session.deviceRegistryId != 0 &&
         session.pixelFormatMask != 0 &&
         (session.pixelFormatMask & ~knownFormats) == 0 &&
         session.maximumWidth > 0 &&
         session.maximumWidth <= kMaximumDimension &&
         session.maximumHeight > 0 &&
         session.maximumHeight <= kMaximumDimension &&
         session.maximumSurfaceBytes > 0 &&
         session.maximumSurfaceBytes <= kMaximumSurfaceBytes &&
         session.maximumRetainedBytes >=
             session.maximumSurfaceBytes &&
         session.maximumRetainedBytes <= kMaximumRetainedBytes &&
         session.maximumSlots > 0 &&
         session.maximumSlots <= kMaximumSlots &&
         session.supportsSharedEvents;
}

uint32_t formatMask(uint32_t pixelFormat) {
  if (pixelFormat == 0) return kPixelFormatRGBA16Float;
  if (pixelFormat == 1) return kPixelFormatRGBA32Float;
  return 0;
}

}  // namespace

ViewerClientState::ExactKey ViewerClientState::exactKey(
    const PublicationKey& key) {
  return {key.sequence, key.slotIndex};
}

bool ViewerClientState::sessionMatches(
    const ViewerSessionSnapshot& session) const {
  return session_ &&
         session_->protocolMajor == session.protocolMajor &&
         session_->protocolMinor == session.protocolMinor &&
         session_->capability == session.capability &&
         session_->viewerGeneration == session.viewerGeneration &&
         session_->senderId == session.senderId &&
         session_->deviceRegistryId == session.deviceRegistryId &&
         session_->pixelFormatMask == session.pixelFormatMask &&
         session_->maximumWidth == session.maximumWidth &&
         session_->maximumHeight == session.maximumHeight &&
         session_->maximumSurfaceBytes ==
             session.maximumSurfaceBytes &&
         session_->maximumRetainedBytes ==
             session.maximumRetainedBytes &&
         session_->maximumSlots == session.maximumSlots &&
         session_->supportsSharedEvents ==
             session.supportsSharedEvents;
}

bool ViewerClientState::sessionIdentityMatches(
    const ViewerSessionSnapshot& session) const {
  return session_ &&
         session_->capability == session.capability &&
         session_->viewerGeneration == session.viewerGeneration &&
         session_->senderId == session.senderId;
}

ViewerClientTransition ViewerClientState::installSession(
    const ViewerSessionSnapshot& session) {
  ViewerClientTransition result{};
  if (!sessionValid(session)) {
    result.code =
        session.protocolMajor != kProtocolMajor ||
                session.protocolMinor > kProtocolMinor
            ? ResultCode::ProtocolMismatch
            : ResultCode::InvalidRegistration;
    return result;
  }
  if (sessionMatches(session)) {
    phase_ = ViewerClientPhase::Ready;
    result.code = ResultCode::Accepted;
    return result;
  }
  if (sessionIdentityMatches(session)) {
    result.code = ResultCode::InvalidRegistration;
    return result;
  }

  abandonLive(&result.abandoned);
  session_ = session;
  senderGeneration_ = 0;
  lastObservedSequence_ = 0;
  phase_ = ViewerClientPhase::Ready;
  result.sessionChanged = true;
  result.code = ResultCode::Accepted;
  return result;
}

ViewerClientTransition ViewerClientState::invalidateSession() {
  ViewerClientTransition result{};
  abandonLive(&result.abandoned);
  session_.reset();
  senderGeneration_ = 0;
  lastObservedSequence_ = 0;
  phase_ = ViewerClientPhase::Failed;
  result.code = ResultCode::Accepted;
  return result;
}

ViewerClientTransition ViewerClientState::replaceSenderGeneration(
    uint64_t senderGeneration) {
  ViewerClientTransition result{};
  if (phase_ != ViewerClientPhase::Ready || !session_) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  if (senderGeneration == 0) {
    result.code = ResultCode::InvalidRegistration;
    return result;
  }
  if (senderGeneration_ == senderGeneration) {
    result.code = ResultCode::Accepted;
    return result;
  }
  abandonLive(&result.abandoned);
  senderGeneration_ = senderGeneration;
  lastObservedSequence_ = 0;
  result.sessionChanged = true;
  result.code = ResultCode::Accepted;
  return result;
}

ViewerClientTransition ViewerClientState::clearActiveSource() {
  ViewerClientTransition result{};
  if (phase_ != ViewerClientPhase::Ready || !session_) {
    result.code = ResultCode::SessionMissing;
    return result;
  }
  if (candidate_) {
    discardCandidateOnAcquired_ = true;
  }
  if (active_) {
    const PublicationKey old = *active_;
    retiring_[exactKey(old)] =
        RetiringRecord{old, RetiringState::AwaitingGpuDrain};
    active_.reset();
    result.needsGpuDrain = old;
  }
  result.code = ResultCode::Accepted;
  return result;
}

bool ViewerClientState::publicationValid(
    const Publication& publication) const {
  if (!session_ ||
      publication.capability != session_->capability ||
      publication.key.senderId != session_->senderId ||
      publication.key.senderGeneration == 0 ||
      publication.key.sequence == 0 ||
      publication.key.slotIndex >= session_->maximumSlots ||
      publication.key.slotGeneration == 0 ||
      publication.deviceRegistryId != session_->deviceRegistryId ||
      publication.width == 0 ||
      publication.width > session_->maximumWidth ||
      publication.height == 0 ||
      publication.height > session_->maximumHeight ||
      publication.readyValue == 0 ||
      publication.contentHash == 0 ||
      !validSourceSemanticMetadata(publication.semantics)) {
    return false;
  }
  const uint32_t mask = formatMask(publication.pixelFormat);
  if (mask == 0 || (session_->pixelFormatMask & mask) == 0) {
    return false;
  }
  const uint64_t bytesPerPixel =
      publication.pixelFormat == 0 ? 8ull : 16ull;
  if (publication.width >
      std::numeric_limits<uint64_t>::max() / bytesPerPixel) {
    return false;
  }
  const uint64_t minimumRowBytes =
      static_cast<uint64_t>(publication.width) * bytesPerPixel;
  if (publication.bytesPerRow < minimumRowBytes ||
      publication.bytesPerRow >
          std::numeric_limits<uint64_t>::max() /
              publication.height) {
    return false;
  }
  const uint64_t minimumByteSize =
      publication.bytesPerRow * publication.height;
  return publication.byteSize >= minimumByteSize &&
         publication.byteSize <= session_->maximumSurfaceBytes &&
         publication.byteSize <= session_->maximumRetainedBytes;
}

ResultCode ViewerClientState::beginImport(
    const Publication& publication) {
  if (phase_ != ViewerClientPhase::Ready || !session_) {
    return ResultCode::SessionMissing;
  }
  if (!publicationValid(publication)) {
    return publication.capability != session_->capability
               ? ResultCode::CapabilityMismatch
               : publication.deviceRegistryId !=
                         session_->deviceRegistryId
                     ? ResultCode::DeviceMismatch
                     : formatMask(publication.pixelFormat) != 0 &&
                               (session_->pixelFormatMask &
                                formatMask(publication.pixelFormat)) == 0
                           ? ResultCode::UnsupportedFormat
                           : ResultCode::InvalidPublication;
  }
  if (candidate_) return ResultCode::SlotBusy;
  if (senderGeneration_ != 0 &&
      senderGeneration_ != publication.key.senderGeneration) {
    return ResultCode::Stale;
  }
  if (publication.key.sequence <= lastObservedSequence_ ||
      keyIsLive(publication.key)) {
    return ResultCode::Stale;
  }
  if (liveKeyCount() >= session_->maximumSlots) {
    return ResultCode::ResourceLimit;
  }

  senderGeneration_ = publication.key.senderGeneration;
  lastObservedSequence_ = publication.key.sequence;
  candidate_ = Candidate{publication, CandidateState::Importing, false};
  return ResultCode::Accepted;
}

ResultCode ViewerClientState::importCompleted(
    const PublicationKey& key,
    bool importedSuccessfully) {
  if (phase_ != ViewerClientPhase::Ready || !candidate_ ||
      candidate_->state != CandidateState::Importing ||
      !(candidate_->publication.key == key)) {
    return ResultCode::Stale;
  }
  candidate_->state =
      CandidateState::AwaitingAcquiredAcknowledgement;
  candidate_->importedSuccessfully = importedSuccessfully;
  queueAcknowledgement(key, ViewerAcknowledgementState::Acquired);
  return ResultCode::Accepted;
}

std::optional<ViewerAcknowledgement>
ViewerClientState::beginNextAcknowledgement() {
  if (phase_ != ViewerClientPhase::Ready ||
      acknowledgements_.empty() ||
      acknowledgements_.front().state != SendState::Ready) {
    return std::nullopt;
  }
  acknowledgements_.front().state = SendState::InFlight;
  return acknowledgements_.front().acknowledgement;
}

ResultCode ViewerClientState::acknowledgementTransportFailed(
    const ViewerAcknowledgement& acknowledgement) {
  if (phase_ != ViewerClientPhase::Ready ||
      acknowledgements_.empty() ||
      acknowledgements_.front().state != SendState::InFlight ||
      !(acknowledgements_.front().acknowledgement ==
        acknowledgement)) {
    return ResultCode::Stale;
  }
  acknowledgements_.front().state = SendState::Ready;
  return ResultCode::Accepted;
}

ViewerClientTransition ViewerClientState::acknowledgementAccepted(
    const ViewerAcknowledgement& acknowledgement) {
  ViewerClientTransition result{};
  if (phase_ != ViewerClientPhase::Ready ||
      acknowledgements_.empty() ||
      acknowledgements_.front().state != SendState::InFlight ||
      !(acknowledgements_.front().acknowledgement ==
        acknowledgement)) {
    result.code = ResultCode::Stale;
    return result;
  }
  const bool acquired =
      acknowledgement.state ==
      ViewerAcknowledgementState::Acquired;
  RetiringRecord* retiring = nullptr;
  if (acquired) {
    if (!candidate_ ||
        candidate_->state !=
            CandidateState::AwaitingAcquiredAcknowledgement ||
        !(candidate_->publication.key == acknowledgement.key)) {
      result.code = ResultCode::Stale;
      return result;
    }
  } else {
    retiring = matchingRetiring(acknowledgement.key);
    if (retiring == nullptr ||
        retiring->state !=
            RetiringState::AwaitingRetiredAcknowledgement) {
      result.code = ResultCode::Stale;
      return result;
    }
  }
  acknowledgements_.pop_front();

  if (acquired) {
    const bool usable = candidate_->importedSuccessfully;
    const PublicationKey candidateKey = candidate_->publication.key;
    candidate_.reset();
    if (!usable) {
      discardCandidateOnAcquired_ = false;
      retiring_[exactKey(candidateKey)] =
          RetiringRecord{
              candidateKey,
              RetiringState::AwaitingRetiredAcknowledgement};
      queueAcknowledgement(
          candidateKey, ViewerAcknowledgementState::Retired);
      result.code = ResultCode::Accepted;
      return result;
    }

    if (discardCandidateOnAcquired_) {
      discardCandidateOnAcquired_ = false;
      retiring_[exactKey(candidateKey)] =
          RetiringRecord{
              candidateKey, RetiringState::AwaitingGpuDrain};
      result.needsGpuDrain = candidateKey;
      result.code = ResultCode::Accepted;
      return result;
    }

    if (active_) {
      const PublicationKey old = *active_;
      retiring_[exactKey(old)] =
          RetiringRecord{
              old, RetiringState::AwaitingGpuDrain};
      result.needsGpuDrain = old;
    }
    active_ = candidateKey;
    result.activated = candidateKey;
    result.code = ResultCode::Accepted;
    return result;
  }

  retiring_.erase(exactKey(acknowledgement.key));
  result.locallyReleasable = acknowledgement.key;
  result.code = ResultCode::Accepted;
  return result;
}

ResultCode ViewerClientState::gpuDrainCompleted(
    const PublicationKey& key) {
  if (phase_ != ViewerClientPhase::Ready) {
    return ResultCode::SessionMissing;
  }
  RetiringRecord* retiring = matchingRetiring(key);
  if (retiring == nullptr ||
      retiring->state != RetiringState::AwaitingGpuDrain) {
    return ResultCode::Stale;
  }
  retiring->state =
      RetiringState::AwaitingRetiredAcknowledgement;
  queueAcknowledgement(key, ViewerAcknowledgementState::Retired);
  return ResultCode::Accepted;
}

size_t ViewerClientState::liveKeyCount() const {
  return (candidate_ ? 1u : 0u) + (active_ ? 1u : 0u) +
         retiring_.size();
}

bool ViewerClientState::keyIsLive(
    const PublicationKey& key) const {
  if (candidate_ && candidate_->publication.key == key) return true;
  if (active_ && *active_ == key) return true;
  auto it = retiring_.find(exactKey(key));
  return it != retiring_.end() && it->second.key == key;
}

ViewerClientState::RetiringRecord*
ViewerClientState::matchingRetiring(const PublicationKey& key) {
  auto it = retiring_.find(exactKey(key));
  if (it == retiring_.end() || !(it->second.key == key)) {
    return nullptr;
  }
  return &it->second;
}

void ViewerClientState::queueAcknowledgement(
    const PublicationKey& key,
    ViewerAcknowledgementState state) {
  acknowledgements_.push_back(
      AcknowledgementRecord{
          ViewerAcknowledgement{key, state}, SendState::Ready});
}

void ViewerClientState::abandonLive(
    std::vector<PublicationKey>* abandoned) {
  if (abandoned == nullptr) return;
  if (candidate_) {
    abandoned->push_back(candidate_->publication.key);
  }
  if (active_) abandoned->push_back(*active_);
  for (const auto& entry : retiring_) {
    abandoned->push_back(entry.second.key);
  }
  candidate_.reset();
  discardCandidateOnAcquired_ = false;
  active_.reset();
  retiring_.clear();
  acknowledgements_.clear();
}

}  // namespace ChromaspaceSourceExchange
