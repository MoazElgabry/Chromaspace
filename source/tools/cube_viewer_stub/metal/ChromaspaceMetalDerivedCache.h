#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

// Portable policy for GPU-resident resources derived from an authoritative
// source signal. This module is platform- and host-independent; callers
// provide opaque cache IDs and
// release the IDs returned by the transaction methods.
namespace ChromaspaceMetalDerivedCache {

constexpr std::size_t kMaximumCommittedEntries = 128u;
constexpr std::size_t kMaximumStagedAcquisitions = 64u;
constexpr std::size_t kMaximumReleaseRecords =
    kMaximumCommittedEntries + kMaximumStagedAcquisitions;
constexpr std::size_t kInvalidIndex = static_cast<std::size_t>(-1);

enum class Family : uint8_t {
  Histogram = 0,
  Waveform,
  RasterPointCloud,
  GlossField,
};

// Identity is intentionally independent of a viewer window or presentation
// size.  The derivation hash must include every sampling/selection/topology
// input that changes the resident resource.
struct DerivedKey {
  uint64_t sourceId = 0u;
  uint64_t deviceRegistryId = 0u;
  uint64_t senderGeneration = 0u;
  uint64_t sequence = 0u;
  uint32_t slotIndex = 0u;
  uint64_t slotGeneration = 0u;
  uint64_t contentHash = 0u;
  Family family = Family::Histogram;
  uint64_t derivationHash = 0u;

  bool operator==(const DerivedKey& other) const noexcept {
    return sourceId == other.sourceId &&
           deviceRegistryId == other.deviceRegistryId &&
           senderGeneration == other.senderGeneration &&
           sequence == other.sequence && slotIndex == other.slotIndex &&
           slotGeneration == other.slotGeneration &&
           contentHash == other.contentHash && family == other.family &&
           derivationHash == other.derivationHash;
  }

  bool operator!=(const DerivedKey& other) const noexcept {
    return !(*this == other);
  }
};

struct Config {
  uint64_t maxResidentBytes = 512ull * 1024ull * 1024ull;
  uint64_t maxTransientBytes = 768ull * 1024ull * 1024ull;
};

enum class Status : uint8_t {
  Ok = 0,
  Success = Ok,
  Hit,
  Candidate,
  InvalidConfig,
  InvalidState,
  Shutdown,
  AlreadyShutdown,
  TransactionAlreadyActive,
  TransactionNotActive,
  InvalidEpoch,
  InvalidKey,
  InvalidEstimate,
  InvalidCandidateIndex,
  InvalidCacheId,
  CacheIdAlreadyInUse,
  CandidateAlreadyMaterialized,
  CandidateUnmaterialized,
  StagedCapacityExceeded,
  ActualSizeTooSmall,
  ResidentBudgetExceeded,
  TransientBudgetExceeded,
  CommittedCapacityExceeded,
  ReleaseOutputMissing,
  ReleaseOutputTooSmall,
  TransactionInvalid,
  ArithmeticOverflow,
};

const char* statusLabel(Status status) noexcept;
const char* familyLabel(Family family) noexcept;

enum class AcquireKind : uint8_t {
  Failure = 0,
  Hit,
  Candidate,
};

const char* acquireKindLabel(AcquireKind kind) noexcept;

struct AcquireResult {
  Status status = Status::InvalidState;
  AcquireKind kind = AcquireKind::Failure;
  std::size_t committedIndex = kInvalidIndex;
  std::size_t stagedIndex = kInvalidIndex;
  uint64_t cacheId = 0u;
  uint64_t byteSize = 0u;
  bool reused = false;

  constexpr bool succeeded() const noexcept {
    return kind != AcquireKind::Failure &&
           (status == Status::Hit || status == Status::Candidate);
  }
  constexpr explicit operator bool() const noexcept { return succeeded(); }
};

struct CandidateMetadata {
  DerivedKey key{};
  uint64_t cacheId = 0u;
  uint64_t byteSize = 0u;
  uint64_t lastUseEpoch = 0u;
  uint64_t estimatedBytes = 0u;
  bool materialized = false;
};

struct CommittedMetadata {
  DerivedKey key{};
  uint64_t cacheId = 0u;
  uint64_t byteSize = 0u;
  uint64_t lastUseEpoch = 0u;
  bool occupied = false;
};

struct ReleaseRecord {
  uint64_t cacheId = 0u;
  uint64_t byteSize = 0u;
  Family family = Family::Histogram;
};

struct ReleaseList {
  std::array<ReleaseRecord, kMaximumReleaseRecords> records{};
  std::size_t count = 0u;

  void clear() noexcept { count = 0u; }

  bool append(const ReleaseRecord& record) noexcept {
    if (count >= records.size()) return false;
    records[count] = record;
    ++count;
    return true;
  }
};

struct ReleaseResult {
  Status status = Status::InvalidState;
  ReleaseList releases{};

  constexpr operator Status() const noexcept { return status; }
  constexpr bool succeeded() const noexcept { return status == Status::Ok; }
  constexpr bool operator==(Status other) const noexcept {
    return status == other;
  }
  constexpr bool operator!=(Status other) const noexcept {
    return status != other;
  }
};

class DerivedCache final {
 public:
  explicit DerivedCache(const Config& config = Config{}) noexcept;
  ~DerivedCache() noexcept;

  DerivedCache(const DerivedCache&) = delete;
  DerivedCache& operator=(const DerivedCache&) = delete;

  static bool validateConfig(const Config& config) noexcept;

  const Config& config() const noexcept { return config_; }
  bool configValid() const noexcept { return configStatus_ == Status::Ok; }
  Status configStatus() const noexcept { return configStatus_; }
  bool shutdownRequested() const noexcept { return shutdown_; }
  bool transactionActive() const noexcept { return transactionActive_; }
  Status transactionStatus() const noexcept { return transactionStatus_; }
  uint64_t transactionEpoch() const noexcept { return transactionEpoch_; }

  // Epochs and cache IDs are opaque caller-owned non-zero tokens.  The cache
  // performs no incrementing arithmetic, so UINT64_MAX is valid and cannot
  // wrap internally.  Zero is reserved as the invalid/sentinel value.
  Status begin(uint64_t epoch) noexcept;
  AcquireResult acquire(const DerivedKey& key,
                        uint64_t estimatedBytes) noexcept;

  const CandidateMetadata* stagedAcquisition(
      std::size_t stagedIndex) const noexcept;
  const CandidateMetadata* stagedCandidate(
      std::size_t stagedIndex) const noexcept {
    return stagedAcquisition(stagedIndex);
  }

  Status materializeCandidate(std::size_t stagedIndex,
                              uint64_t cacheId,
                              uint64_t actualBytes) noexcept;

  std::size_t committedCount() const noexcept { return committedCount_; }
  std::size_t committedCount(Family family) const noexcept;
  std::size_t stagedCount() const noexcept { return stagedCount_; }
  std::size_t candidateCount() const noexcept { return stagedCount_; }
  std::size_t selectedEvictionCount() const noexcept;
  uint64_t residentByteSize() const noexcept { return residentBytes_; }
  uint64_t transientByteSize() const noexcept;
  uint64_t projectedResidentByteSize() const noexcept;

  const CommittedMetadata* committedEntry(
      std::size_t committedSlot) const noexcept;
  std::size_t committedSlotForKey(const DerivedKey& key) const noexcept;

  Status commit(ReleaseList* releases) noexcept;
  ReleaseResult commit() noexcept;
  Status commit(ReleaseRecord* records,
                std::size_t capacity,
                std::size_t* count) noexcept;

  Status abort(ReleaseList* releases) noexcept;
  ReleaseResult abort() noexcept;
  Status abort(ReleaseRecord* records,
               std::size_t capacity,
               std::size_t* count) noexcept;

  Status shutdown(ReleaseList* releases) noexcept;
  ReleaseResult shutdown() noexcept;
  Status shutdown(ReleaseRecord* records,
                  std::size_t capacity,
                  std::size_t* count) noexcept;

  // Non-terminal device/runtime recreation.  Every committed and materialized
  // staged handle is returned exactly once, then the cache can begin again.
  Status reset(ReleaseList* releases) noexcept;
  ReleaseResult reset() noexcept;

 private:
  struct Output {
    ReleaseRecord* records = nullptr;
    std::size_t capacity = 0u;
    std::size_t* count = nullptr;
  };

  static bool addExact(uint64_t left,
                       uint64_t right,
                       uint64_t* result) noexcept;
  static bool validKey(const DerivedKey& key) noexcept;

  bool findCommitted(const DerivedKey& key,
                     std::size_t* slot) const noexcept;
  bool findStaged(const DerivedKey& key,
                  std::size_t* stagedIndex) const noexcept;
  bool cacheIdInUse(uint64_t cacheId,
                    std::size_t ignoredStagedIndex = kInvalidIndex) const
      noexcept;

  bool candidateBytesWithinTransient(uint64_t candidateBytes) const noexcept;
  bool planEvictions(
      uint64_t candidateBytes,
      std::size_t finalCandidateCount,
      const std::array<bool, kMaximumCommittedEntries>& pinned,
      std::array<bool, kMaximumCommittedEntries>* proposed,
      uint64_t* projectedBytes) const noexcept;
  bool buildEvictionRecords(
      const std::array<bool, kMaximumCommittedEntries>& flags,
      Output output,
      std::size_t* emitted) const noexcept;
  bool outputCanHold(Output output, std::size_t required) const noexcept;
  void clearTransaction() noexcept;
  void clearAll() noexcept;
  Status finishAbort(Output output) noexcept;

  Config config_{};
  Status configStatus_ = Status::InvalidConfig;
  bool shutdown_ = false;
  bool transactionActive_ = false;
  uint64_t transactionEpoch_ = 0u;
  Status transactionStatus_ = Status::TransactionNotActive;

  std::array<CommittedMetadata, kMaximumCommittedEntries> committed_{};
  std::size_t committedCount_ = 0u;
  uint64_t residentBytes_ = 0u;

  std::array<CandidateMetadata, kMaximumStagedAcquisitions> staged_{};
  std::size_t stagedCount_ = 0u;
  uint64_t stagedBytes_ = 0u;
  std::array<bool, kMaximumCommittedEntries> pinned_{};
  std::array<bool, kMaximumCommittedEntries> evicted_{};
};

using ResidentDerivedCache = DerivedCache;

}  // namespace ChromaspaceMetalDerivedCache
