#pragma once

#include "ChromaspaceSourceExchangeState.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ChromaspaceSourceExchange {

struct ViewerSessionSnapshot {
  uint32_t protocolMajor = kProtocolMajor;
  uint32_t protocolMinor = kProtocolMinor;
  Capability capability{};
  uint64_t viewerGeneration = 0;
  std::string senderId;
  uint64_t deviceRegistryId = 0;
  uint32_t pixelFormatMask = 0;
  uint32_t maximumWidth = 0;
  uint32_t maximumHeight = 0;
  uint64_t maximumSurfaceBytes = 0;
  uint64_t maximumRetainedBytes = 0;
  uint32_t maximumSlots = 0;
  bool supportsSharedEvents = false;
};

enum class ViewerClientPhase {
  Disconnected,
  Ready,
  Failed,
};

enum class ViewerAcknowledgementState {
  Acquired,
  Retired,
};

struct ViewerAcknowledgement {
  PublicationKey key;
  ViewerAcknowledgementState state =
      ViewerAcknowledgementState::Acquired;

  bool operator==(const ViewerAcknowledgement& other) const {
    return key == other.key && state == other.state;
  }
};

struct ViewerClientTransition {
  ResultCode code = ResultCode::InvalidTransition;
  bool sessionChanged = false;
  std::optional<PublicationKey> activated;
  std::optional<PublicationKey> needsGpuDrain;
  std::optional<PublicationKey> locallyReleasable;
  std::vector<PublicationKey> abandoned;

  bool accepted() const { return code == ResultCode::Accepted; }
};

// Portable lifecycle core for the viewer-side SourceExchange adapter. Metal
// imports, command-buffer drain detection, XPC packets, and dispatch queues
// remain adapter details.
class ViewerClientState {
 public:
  ViewerClientTransition installSession(
      const ViewerSessionSnapshot& session);
  ViewerClientTransition invalidateSession();
  ViewerClientTransition replaceSenderGeneration(
      uint64_t senderGeneration);
  ViewerClientTransition clearActiveSource();

  ResultCode beginImport(const Publication& publication);
  ResultCode importCompleted(const PublicationKey& key,
                             bool importedSuccessfully);

  std::optional<ViewerAcknowledgement> beginNextAcknowledgement();
  ResultCode acknowledgementTransportFailed(
      const ViewerAcknowledgement& acknowledgement);
  ViewerClientTransition acknowledgementAccepted(
      const ViewerAcknowledgement& acknowledgement);

  ResultCode gpuDrainCompleted(const PublicationKey& key);

  ViewerClientPhase phase() const { return phase_; }
  uint64_t lastObservedSequence() const {
    return lastObservedSequence_;
  }
  uint64_t senderGeneration() const { return senderGeneration_; }
  size_t liveKeyCount() const;
  bool importPending() const { return candidate_.has_value(); }
  bool canAcquire() const {
    return phase_ == ViewerClientPhase::Ready && session_.has_value() &&
           !candidate_.has_value() &&
           liveKeyCount() < session_->maximumSlots;
  }
  std::optional<PublicationKey> activeKey() const {
    return active_;
  }

 private:
  enum class CandidateState {
    Importing,
    AwaitingAcquiredAcknowledgement,
  };

  struct Candidate {
    Publication publication;
    CandidateState state = CandidateState::Importing;
    bool importedSuccessfully = false;
  };

  enum class RetiringState {
    AwaitingGpuDrain,
    AwaitingRetiredAcknowledgement,
  };

  struct RetiringRecord {
    PublicationKey key;
    RetiringState state = RetiringState::AwaitingGpuDrain;
  };

  enum class SendState {
    Ready,
    InFlight,
  };

  struct AcknowledgementRecord {
    ViewerAcknowledgement acknowledgement;
    SendState state = SendState::Ready;
  };

  using ExactKey = std::pair<uint64_t, uint32_t>;

  static ExactKey exactKey(const PublicationKey& key);
  bool sessionMatches(const ViewerSessionSnapshot& session) const;
  bool sessionIdentityMatches(
      const ViewerSessionSnapshot& session) const;
  bool publicationValid(const Publication& publication) const;
  bool keyIsLive(const PublicationKey& key) const;
  RetiringRecord* matchingRetiring(const PublicationKey& key);
  void queueAcknowledgement(const PublicationKey& key,
                            ViewerAcknowledgementState state);
  void abandonLive(std::vector<PublicationKey>* abandoned);

  std::optional<ViewerSessionSnapshot> session_;
  ViewerClientPhase phase_ = ViewerClientPhase::Disconnected;
  uint64_t senderGeneration_ = 0;
  uint64_t lastObservedSequence_ = 0;
  std::optional<Candidate> candidate_;
  bool discardCandidateOnAcquired_ = false;
  std::optional<PublicationKey> active_;
  std::map<ExactKey, RetiringRecord> retiring_;
  std::deque<AcknowledgementRecord> acknowledgements_;
};

}  // namespace ChromaspaceSourceExchange
