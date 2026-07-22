#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ChromaspaceSourceExchange {

constexpr uint32_t kProtocolMajor = 2;
constexpr uint32_t kProtocolMinor = 0;
constexpr size_t kCapabilityBytes = 32;
constexpr size_t kMaximumSessions = 32;
constexpr size_t kMaximumProducersPerSession = 32;
constexpr size_t kMaximumSlots = 3;
constexpr size_t kMaximumProducerReleaseEvents = 64;
constexpr uint32_t kMaximumDimension = 16384;
constexpr uint32_t kMaximumIdentityResolution = 256;
constexpr size_t kMaximumSemanticIdentifierBytes = 64;
constexpr uint64_t kMaximumSurfaceBytes = 1024ull * 1024ull * 1024ull;
constexpr uint64_t kMaximumRetainedBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr uint32_t kPixelFormatRGBA16Float = 1u << 0;
constexpr uint32_t kPixelFormatRGBA32Float = 1u << 1;

using Capability = std::array<uint8_t, kCapabilityBytes>;

enum class ResultCode {
  Accepted,
  NoNewPublication,
  SessionMissing,
  ProducerMissing,
  InvalidRegistration,
  InvalidPublication,
  ProtocolMismatch,
  CapabilityMismatch,
  DeviceMismatch,
  UnsupportedFormat,
  ResourceLimit,
  SlotBusy,
  Stale,
  InvalidTransition,
};

enum class SlotState {
  Free,
  Published,
  Acquired,
};

enum class AcknowledgementState {
  Acquired,
  Retired,
};

enum class SourceCoverage : uint32_t {
  FullSource = 0,
  PartialSource = 1,
};

// Immutable source-domain meaning for one publication. Coordinates use the
// original OFX image coordinate system; Publication width/height describe the
// transported proxy texture and may differ from the sampled source rectangle.
struct SourceSemanticMetadata {
  int32_t sourceX = 0;
  int32_t sourceY = 0;
  uint32_t sourceWidth = 0;
  uint32_t sourceHeight = 0;
  int32_t sampledX = 0;
  int32_t sampledY = 0;
  uint32_t sampledWidth = 0;
  uint32_t sampledHeight = 0;
  SourceCoverage coverage = SourceCoverage::FullSource;
  bool authoritative = false;

  bool identityStripPresent = false;
  bool identityCube = false;
  bool identityRamp = false;
  uint32_t identityResolution = 0;
  uint32_t identityBandHeight = 0;
  int32_t identityCubeY1 = 0;
  int32_t identityCubeY2 = 0;
  int32_t identityRampY1 = 0;
  int32_t identityRampY2 = 0;

  std::string colorPrimaries;
  std::string transferFunction;

  bool operator==(const SourceSemanticMetadata& other) const;
};

bool validSourceSemanticMetadata(const SourceSemanticMetadata& metadata);

struct ViewerRegistration {
  uint32_t protocolMajor = kProtocolMajor;
  uint32_t protocolMinor = kProtocolMinor;
  Capability capability{};
  uint64_t viewerGeneration = 0;
  uint64_t deviceRegistryId = 0;
  uint32_t pixelFormatMask = 0;
  uint32_t maximumWidth = 0;
  uint32_t maximumHeight = 0;
  uint64_t maximumSurfaceBytes = 0;
  uint64_t maximumRetainedBytes = 0;
  uint32_t maximumSlots = 0;
  bool supportsSharedEvents = false;
};

struct ProducerRegistration {
  Capability capability{};
  std::string senderId;
  uint64_t senderGeneration = 0;
  uint64_t deviceRegistryId = 0;
};

struct PublicationKey {
  std::string senderId;
  uint64_t senderGeneration = 0;
  uint64_t sequence = 0;
  uint32_t slotIndex = 0;
  uint64_t slotGeneration = 0;

  bool operator==(const PublicationKey& other) const;
};

struct Publication {
  Capability capability{};
  PublicationKey key;
  uint64_t deviceRegistryId = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  uint64_t bytesPerRow = 0;
  uint64_t byteSize = 0;
  uint64_t readyValue = 0;
  uint64_t contentHash = 0;
  SourceSemanticMetadata semantics;
};

struct TransitionResult {
  ResultCode code = ResultCode::InvalidTransition;
  std::vector<PublicationKey> released;

  bool accepted() const { return code == ResultCode::Accepted; }
};

struct ProducerReleaseEvent {
  uint64_t ordinal = 0;
  PublicationKey key;
};

struct ProducerReleaseBatch {
  Capability capability{};
  std::string senderId;
  uint64_t senderGeneration = 0;
  uint64_t throughOrdinal = 0;
  std::vector<ProducerReleaseEvent> events;
};

class BrokerState {
 public:
  TransitionResult registerViewer(const ViewerRegistration& registration);
  TransitionResult registerProducer(const ProducerRegistration& registration);
  TransitionResult publish(const Publication& publication);

  TransitionResult acquireLatest(const Capability& capability,
                                 const std::string& senderId,
                                 uint64_t afterSequence,
                                 Publication* outPublication);

  TransitionResult acknowledge(const Capability& capability,
                               const PublicationKey& key,
                               AcknowledgementState state);
  TransitionResult fetchProducerReleases(
      const Capability& capability,
      const std::string& senderId,
      uint64_t senderGeneration,
      uint64_t afterOrdinal,
      size_t maximumEvents,
      ProducerReleaseBatch* outBatch);
  TransitionResult acknowledgeProducerReleases(
      const Capability& capability,
      const std::string& senderId,
      uint64_t senderGeneration,
      uint64_t throughOrdinal);
  TransitionResult disconnectProducer(const Capability& capability,
                                      const std::string& senderId,
                                      uint64_t senderGeneration);
  TransitionResult disconnectViewer(const Capability& capability);

  size_t sessionCount() const;
  size_t producerCount(const Capability& capability) const;
  size_t livePublicationCount(const Capability& capability) const;
  uint64_t retainedBytes() const;
  size_t pendingProducerReleaseCount(const Capability& capability,
                                     const std::string& senderId) const;
  std::optional<SlotState> slotState(const Capability& capability,
                                     const PublicationKey& key) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

 public:
  BrokerState();
  ~BrokerState();
  BrokerState(BrokerState&&) noexcept;
  BrokerState& operator=(BrokerState&&) noexcept;
  BrokerState(const BrokerState&) = delete;
  BrokerState& operator=(const BrokerState&) = delete;
};

}  // namespace ChromaspaceSourceExchange
