#include "ChromaspaceMetalDerivedCache.h"

#include <limits>

namespace ChromaspaceMetalDerivedCache {
namespace {

constexpr uint64_t kMaximumUint64 =
    (std::numeric_limits<uint64_t>::max)();

bool validFamily(Family family) noexcept {
  return family == Family::Histogram || family == Family::Waveform ||
         family == Family::RasterPointCloud || family == Family::GlossField;
}

void copyReleases(ReleaseList* destination,
                  const ReleaseList& source) noexcept {
  if (destination == nullptr) return;
  destination->count = source.count;
  for (std::size_t index = 0; index < source.count; ++index) {
    destination->records[index] = source.records[index];
  }
}

}  // namespace

const char* statusLabel(Status status) noexcept {
  switch (status) {
    case Status::Ok: return "ok";
    case Status::Hit: return "hit";
    case Status::Candidate: return "candidate";
    case Status::InvalidConfig: return "invalid-config";
    case Status::InvalidState: return "invalid-state";
    case Status::Shutdown: return "shutdown";
    case Status::AlreadyShutdown: return "already-shutdown";
    case Status::TransactionAlreadyActive:
      return "transaction-already-active";
    case Status::TransactionNotActive: return "transaction-not-active";
    case Status::InvalidEpoch: return "invalid-epoch";
    case Status::InvalidKey: return "invalid-key";
    case Status::InvalidEstimate: return "invalid-estimate";
    case Status::InvalidCandidateIndex: return "invalid-candidate-index";
    case Status::InvalidCacheId: return "invalid-cache-id";
    case Status::CacheIdAlreadyInUse: return "cache-id-already-in-use";
    case Status::CandidateAlreadyMaterialized:
      return "candidate-already-materialized";
    case Status::CandidateUnmaterialized: return "candidate-unmaterialized";
    case Status::StagedCapacityExceeded: return "staged-capacity-exceeded";
    case Status::ActualSizeTooSmall: return "actual-size-too-small";
    case Status::ResidentBudgetExceeded: return "resident-budget-exceeded";
    case Status::TransientBudgetExceeded: return "transient-budget-exceeded";
    case Status::CommittedCapacityExceeded:
      return "committed-capacity-exceeded";
    case Status::ReleaseOutputMissing: return "release-output-missing";
    case Status::ReleaseOutputTooSmall: return "release-output-too-small";
    case Status::TransactionInvalid: return "transaction-invalid";
    case Status::ArithmeticOverflow: return "arithmetic-overflow";
  }
  return "invalid-state";
}

const char* familyLabel(Family family) noexcept {
  switch (family) {
    case Family::Histogram: return "histogram";
    case Family::Waveform: return "waveform";
    case Family::RasterPointCloud: return "raster-point-cloud";
    case Family::GlossField: return "gloss-field";
  }
  return "unknown";
}

const char* acquireKindLabel(AcquireKind kind) noexcept {
  switch (kind) {
    case AcquireKind::Failure: return "failure";
    case AcquireKind::Hit: return "hit";
    case AcquireKind::Candidate: return "candidate";
  }
  return "failure";
}

DerivedCache::DerivedCache(const Config& config) noexcept
    : config_(config),
      configStatus_(validateConfig(config) ? Status::Ok
                                            : Status::InvalidConfig) {}

DerivedCache::~DerivedCache() noexcept {
  // Production callers pass a ReleaseList to release opaque native handles.
  // Destruction still drains the bounded state safely when a caller skipped
  // explicit teardown; the ignored IDs cannot be handed to a native backend.
  if (!shutdown_) {
    ReleaseList ignored{};
    (void)shutdown(&ignored);
  }
}

bool DerivedCache::validateConfig(const Config& config) noexcept {
  // During replacement, existing resident bytes and staged candidate bytes
  // coexist.  Requiring transient >= resident makes that invariant explicit.
  return config.maxResidentBytes != 0u && config.maxTransientBytes != 0u &&
         config.maxResidentBytes <= config.maxTransientBytes;
}

bool DerivedCache::addExact(uint64_t left,
                            uint64_t right,
                            uint64_t* result) noexcept {
  if (result == nullptr || right > kMaximumUint64 - left) return false;
  *result = left + right;
  return true;
}

bool DerivedCache::validKey(const DerivedKey& key) noexcept {
  // Source identity values and derivationHash use zero as an invalid/sentinel
  // value.  slotIndex is an array index and is therefore allowed to be zero.
  return key.sourceId != 0u && key.deviceRegistryId != 0u &&
         key.senderGeneration != 0u && key.sequence != 0u &&
         key.slotGeneration != 0u && key.contentHash != 0u &&
         key.derivationHash != 0u && validFamily(key.family);
}

Status DerivedCache::begin(uint64_t epoch) noexcept {
  if (shutdown_) return Status::Shutdown;
  if (!configValid()) return Status::InvalidConfig;
  if (transactionActive_) return Status::TransactionAlreadyActive;
  if (epoch == 0u) return Status::InvalidEpoch;

  transactionActive_ = true;
  transactionEpoch_ = epoch;
  transactionStatus_ = Status::Ok;
  stagedCount_ = 0u;
  stagedBytes_ = 0u;
  pinned_.fill(false);
  evicted_.fill(false);
  staged_.fill(CandidateMetadata{});
  return Status::Ok;
}

bool DerivedCache::findCommitted(const DerivedKey& key,
                                 std::size_t* slot) const noexcept {
  if (slot == nullptr) return false;
  for (std::size_t index = 0; index < committed_.size(); ++index) {
    if (committed_[index].occupied && committed_[index].key == key) {
      *slot = index;
      return true;
    }
  }
  return false;
}

bool DerivedCache::findStaged(const DerivedKey& key,
                              std::size_t* stagedIndex) const noexcept {
  if (stagedIndex == nullptr) return false;
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].key == key) {
      *stagedIndex = index;
      return true;
    }
  }
  return false;
}

bool DerivedCache::cacheIdInUse(uint64_t cacheId,
                                std::size_t ignoredStagedIndex) const noexcept {
  if (cacheId == 0u) return false;
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId == cacheId) return true;
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (index != ignoredStagedIndex && staged_[index].cacheId == cacheId) {
      return true;
    }
  }
  return false;
}

bool DerivedCache::candidateBytesWithinTransient(
    uint64_t candidateBytes) const noexcept {
  if (residentBytes_ > config_.maxTransientBytes) return false;
  return candidateBytes <= config_.maxTransientBytes - residentBytes_;
}

bool DerivedCache::planEvictions(
    uint64_t candidateBytes,
    std::size_t finalCandidateCount,
    const std::array<bool, kMaximumCommittedEntries>& pinned,
    std::array<bool, kMaximumCommittedEntries>* proposed,
    uint64_t* projectedBytes) const noexcept {
  if (proposed == nullptr || projectedBytes == nullptr) return false;

  uint64_t projected = residentBytes_;
  std::size_t selectedCount = 0u;
  for (std::size_t index = 0; index < committed_.size(); ++index) {
    if (!(*proposed)[index]) continue;
    if (!committed_[index].occupied || pinned[index]) return false;
    ++selectedCount;
    if (committed_[index].byteSize > projected) return false;
    projected -= committed_[index].byteSize;
  }
  if (!addExact(projected, candidateBytes, &projected)) return false;

  std::size_t requiredEvictions = 0u;
  if (finalCandidateCount > kMaximumCommittedEntries - committedCount_) {
    requiredEvictions = finalCandidateCount -
                        (kMaximumCommittedEntries - committedCount_);
  }

  // Deterministic unpinned LRU.  Equal epochs use the lower array slot so
  // tests and native release order are stable across runs and platforms.
  while (projected > config_.maxResidentBytes ||
         selectedCount < requiredEvictions) {
    std::size_t selected = kInvalidIndex;
    uint64_t selectedEpoch = kMaximumUint64;
    for (std::size_t index = 0; index < committed_.size(); ++index) {
      const auto& entry = committed_[index];
      if (!entry.occupied || pinned[index] || (*proposed)[index]) continue;
      if (selected == kInvalidIndex || entry.lastUseEpoch < selectedEpoch ||
          (entry.lastUseEpoch == selectedEpoch && index < selected)) {
        selected = index;
        selectedEpoch = entry.lastUseEpoch;
      }
    }
    if (selected == kInvalidIndex) return false;
    (*proposed)[selected] = true;
    ++selectedCount;
    if (committed_[selected].byteSize > projected) return false;
    projected -= committed_[selected].byteSize;
  }

  *projectedBytes = projected;
  return true;
}

AcquireResult DerivedCache::acquire(const DerivedKey& key,
                                    uint64_t estimatedBytes) noexcept {
  AcquireResult result{};
  result.status = Status::InvalidState;
  if (shutdown_) {
    result.status = Status::Shutdown;
    return result;
  }
  if (!configValid()) {
    result.status = Status::InvalidConfig;
    return result;
  }
  if (!transactionActive_) {
    result.status = Status::TransactionNotActive;
    return result;
  }
  if (transactionStatus_ != Status::Ok) {
    result.status = Status::TransactionInvalid;
    return result;
  }
  if (!validKey(key)) {
    result.status = Status::InvalidKey;
    return result;
  }
  if (estimatedBytes == 0u) {
    result.status = Status::InvalidEstimate;
    return result;
  }

  std::size_t committedSlot = kInvalidIndex;
  if (findCommitted(key, &committedSlot)) {
    auto proposedPinned = pinned_;
    auto proposedEvicted = evicted_;
    proposedPinned[committedSlot] = true;
    // A hit can reclaim a tentative eviction selected by an earlier acquire.
    proposedEvicted[committedSlot] = false;
    uint64_t projected = 0u;
    if (!candidateBytesWithinTransient(stagedBytes_) ||
        !planEvictions(stagedBytes_, stagedCount_, proposedPinned,
                       &proposedEvicted, &projected)) {
      result.status = Status::ResidentBudgetExceeded;
      return result;
    }
    pinned_ = proposedPinned;
    evicted_ = proposedEvicted;
    result.status = Status::Hit;
    result.kind = AcquireKind::Hit;
    result.committedIndex = committedSlot;
    result.cacheId = committed_[committedSlot].cacheId;
    result.byteSize = committed_[committedSlot].byteSize;
    result.reused = true;
    return result;
  }

  std::size_t stagedIndex = kInvalidIndex;
  if (findStaged(key, &stagedIndex)) {
    CandidateMetadata& candidate = staged_[stagedIndex];
    if (estimatedBytes > candidate.estimatedBytes) {
      // Once native storage has been materialized, a larger estimate would
      // invalidate the caller's actual-byte guarantee.  A smaller estimate is
      // harmless and the original reservation remains authoritative.
      if (candidate.materialized) {
        if (estimatedBytes > candidate.byteSize) {
          result.status = Status::ActualSizeTooSmall;
          return result;
        }
        candidate.estimatedBytes = estimatedBytes;
      } else {
        uint64_t nextStagedBytes = stagedBytes_;
        if (candidate.byteSize > nextStagedBytes) {
          result.status = Status::ArithmeticOverflow;
          return result;
        }
        nextStagedBytes -= candidate.byteSize;
        if (!addExact(nextStagedBytes, estimatedBytes, &nextStagedBytes)) {
          result.status = Status::ArithmeticOverflow;
          return result;
        }
        if (!candidateBytesWithinTransient(nextStagedBytes)) {
          result.status = Status::TransientBudgetExceeded;
          return result;
        }
        auto proposedEvicted = evicted_;
        uint64_t projected = 0u;
        if (!planEvictions(nextStagedBytes, stagedCount_, pinned_,
                           &proposedEvicted, &projected)) {
          result.status = stagedCount_ + committedCount_ >=
                                  kMaximumCommittedEntries
                              ? Status::CommittedCapacityExceeded
                              : Status::ResidentBudgetExceeded;
          return result;
        }
        stagedBytes_ = nextStagedBytes;
        candidate.byteSize = estimatedBytes;
        candidate.estimatedBytes = estimatedBytes;
        evicted_ = proposedEvicted;
      }
    }
    result.status = Status::Candidate;
    result.kind = AcquireKind::Candidate;
    result.stagedIndex = stagedIndex;
    result.cacheId = candidate.cacheId;
    result.byteSize = candidate.byteSize;
    result.reused = true;
    return result;
  }

  if (stagedCount_ >= kMaximumStagedAcquisitions) {
    result.status = Status::StagedCapacityExceeded;
    return result;
  }
  uint64_t nextStagedBytes = 0u;
  if (!addExact(stagedBytes_, estimatedBytes, &nextStagedBytes)) {
    result.status = Status::ArithmeticOverflow;
    return result;
  }
  if (!candidateBytesWithinTransient(nextStagedBytes)) {
    result.status = Status::TransientBudgetExceeded;
    return result;
  }

  auto proposedEvicted = evicted_;
  uint64_t projected = 0u;
  if (!planEvictions(nextStagedBytes, stagedCount_ + 1u, pinned_,
                     &proposedEvicted, &projected)) {
    result.status = committedCount_ + stagedCount_ + 1u >
                            kMaximumCommittedEntries
                        ? Status::CommittedCapacityExceeded
                        : Status::ResidentBudgetExceeded;
    return result;
  }

  stagedIndex = stagedCount_;
  CandidateMetadata candidate{};
  candidate.key = key;
  candidate.byteSize = estimatedBytes;
  candidate.lastUseEpoch = transactionEpoch_;
  candidate.estimatedBytes = estimatedBytes;
  staged_[stagedIndex] = candidate;
  ++stagedCount_;
  stagedBytes_ = nextStagedBytes;
  evicted_ = proposedEvicted;

  result.status = Status::Candidate;
  result.kind = AcquireKind::Candidate;
  result.stagedIndex = stagedIndex;
  result.byteSize = estimatedBytes;
  return result;
}

const CandidateMetadata* DerivedCache::stagedAcquisition(
    std::size_t stagedIndex) const noexcept {
  if (!transactionActive_ || stagedIndex >= stagedCount_) return nullptr;
  return &staged_[stagedIndex];
}

Status DerivedCache::materializeCandidate(std::size_t stagedIndex,
                                           uint64_t cacheId,
                                           uint64_t actualBytes) noexcept {
  if (shutdown_) return Status::Shutdown;
  if (!transactionActive_) return Status::TransactionNotActive;
  if (transactionStatus_ != Status::Ok) return Status::TransactionInvalid;
  if (stagedIndex >= stagedCount_) return Status::InvalidCandidateIndex;
  if (cacheId == 0u) return Status::InvalidCacheId;
  CandidateMetadata& candidate = staged_[stagedIndex];
  if (candidate.materialized || candidate.cacheId != 0u) {
    return Status::CandidateAlreadyMaterialized;
  }
  if (cacheIdInUse(cacheId, stagedIndex)) {
    return Status::CacheIdAlreadyInUse;
  }
  if (actualBytes == 0u || actualBytes < candidate.estimatedBytes) {
    return Status::ActualSizeTooSmall;
  }
  if (candidate.byteSize > stagedBytes_) return Status::ArithmeticOverflow;

  uint64_t nextStagedBytes = stagedBytes_ - candidate.byteSize;
  if (!addExact(nextStagedBytes, actualBytes, &nextStagedBytes)) {
    return Status::ArithmeticOverflow;
  }
  if (!candidateBytesWithinTransient(nextStagedBytes)) {
    return Status::TransientBudgetExceeded;
  }
  auto proposedEvicted = evicted_;
  uint64_t projected = 0u;
  if (!planEvictions(nextStagedBytes, stagedCount_, pinned_, &proposedEvicted,
                     &projected)) {
    return committedCount_ + stagedCount_ > kMaximumCommittedEntries
               ? Status::CommittedCapacityExceeded
               : Status::ResidentBudgetExceeded;
  }

  // Publish every field only after all validation above succeeds.  A failed
  // GPU allocation or a stale actual-byte report therefore leaves the
  // transaction unchanged and safely abortable.
  stagedBytes_ = nextStagedBytes;
  candidate.byteSize = actualBytes;
  candidate.cacheId = cacheId;
  candidate.materialized = true;
  evicted_ = proposedEvicted;
  return Status::Ok;
}

std::size_t DerivedCache::selectedEvictionCount() const noexcept {
  std::size_t count = 0u;
  for (std::size_t index = 0; index < evicted_.size(); ++index) {
    if (evicted_[index] && committed_[index].occupied) ++count;
  }
  return count;
}

std::size_t DerivedCache::committedCount(Family family) const noexcept {
  if (!validFamily(family)) return 0u;
  std::size_t count = 0u;
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.key.family == family) ++count;
  }
  return count;
}

uint64_t DerivedCache::transientByteSize() const noexcept {
  uint64_t result = 0u;
  return addExact(residentBytes_, stagedBytes_, &result) ? result
                                                        : kMaximumUint64;
}

uint64_t DerivedCache::projectedResidentByteSize() const noexcept {
  uint64_t result = residentBytes_;
  for (std::size_t index = 0; index < evicted_.size(); ++index) {
    if (!evicted_[index] || !committed_[index].occupied) continue;
    if (committed_[index].byteSize > result) return kMaximumUint64;
    result -= committed_[index].byteSize;
  }
  return addExact(result, stagedBytes_, &result) ? result : kMaximumUint64;
}

const CommittedMetadata* DerivedCache::committedEntry(
    std::size_t committedSlot) const noexcept {
  if (committedSlot >= committed_.size() ||
      !committed_[committedSlot].occupied) {
    return nullptr;
  }
  return &committed_[committedSlot];
}

std::size_t DerivedCache::committedSlotForKey(
    const DerivedKey& key) const noexcept {
  std::size_t slot = kInvalidIndex;
  return findCommitted(key, &slot) ? slot : kInvalidIndex;
}

bool DerivedCache::outputCanHold(Output output,
                                 std::size_t required) const noexcept {
  if (required == 0u) return true;
  return output.records != nullptr && output.capacity >= required;
}

bool DerivedCache::buildEvictionRecords(
    const std::array<bool, kMaximumCommittedEntries>& flags,
    Output output,
    std::size_t* emitted) const noexcept {
  if (emitted == nullptr) return false;
  *emitted = 0u;
  std::size_t required = 0u;
  for (std::size_t index = 0; index < committed_.size(); ++index) {
    if (flags[index] && committed_[index].occupied) ++required;
  }
  if (!outputCanHold(output, required)) return false;

  std::array<bool, kMaximumCommittedEntries> consumed{};
  while (*emitted < required) {
    std::size_t selected = kInvalidIndex;
    uint64_t selectedEpoch = kMaximumUint64;
    for (std::size_t index = 0; index < committed_.size(); ++index) {
      if (!flags[index] || consumed[index] || !committed_[index].occupied) {
        continue;
      }
      if (selected == kInvalidIndex ||
          committed_[index].lastUseEpoch < selectedEpoch ||
          (committed_[index].lastUseEpoch == selectedEpoch &&
           index < selected)) {
        selected = index;
        selectedEpoch = committed_[index].lastUseEpoch;
      }
    }
    if (selected == kInvalidIndex) return false;
    consumed[selected] = true;
    output.records[*emitted] =
        ReleaseRecord{committed_[selected].cacheId,
                      committed_[selected].byteSize,
                      committed_[selected].key.family};
    ++*emitted;
  }
  if (output.count != nullptr) *output.count = *emitted;
  return true;
}

void DerivedCache::clearTransaction() noexcept {
  transactionActive_ = false;
  transactionEpoch_ = 0u;
  transactionStatus_ = Status::TransactionNotActive;
  stagedCount_ = 0u;
  stagedBytes_ = 0u;
  pinned_.fill(false);
  evicted_.fill(false);
  staged_.fill(CandidateMetadata{});
}

void DerivedCache::clearAll() noexcept {
  clearTransaction();
  committed_.fill(CommittedMetadata{});
  committedCount_ = 0u;
  residentBytes_ = 0u;
}

Status DerivedCache::commit(ReleaseList* releases) noexcept {
  if (releases != nullptr) releases->clear();
  if (shutdown_) return Status::Shutdown;
  if (!transactionActive_) return Status::TransactionNotActive;
  if (!configValid()) return Status::InvalidConfig;
  if (transactionStatus_ != Status::Ok) return Status::TransactionInvalid;

  Output output{};
  if (releases != nullptr) {
    output.records = releases->records.data();
    output.capacity = releases->records.size();
    output.count = &releases->count;
  }

  for (std::size_t index = 0; index < stagedCount_; ++index) {
    const CandidateMetadata& candidate = staged_[index];
    if (candidate.cacheId == 0u || !candidate.materialized) {
      return Status::CandidateUnmaterialized;
    }
    if (candidate.byteSize == 0u ||
        candidate.byteSize < candidate.estimatedBytes) {
      return Status::ActualSizeTooSmall;
    }
    if (cacheIdInUse(candidate.cacheId, index)) {
      return Status::CacheIdAlreadyInUse;
    }
  }
  if (!candidateBytesWithinTransient(stagedBytes_)) {
    return Status::TransientBudgetExceeded;
  }

  auto proposedEvicted = evicted_;
  uint64_t projected = 0u;
  if (!planEvictions(stagedBytes_, stagedCount_, pinned_, &proposedEvicted,
                     &projected)) {
    return committedCount_ + stagedCount_ > kMaximumCommittedEntries
               ? Status::CommittedCapacityExceeded
               : Status::ResidentBudgetExceeded;
  }

  ReleaseList localReleases{};
  Output localOutput{localReleases.records.data(), localReleases.records.size(),
                     &localReleases.count};
  std::size_t evictionCount = 0u;
  if (!buildEvictionRecords(proposedEvicted, localOutput, &evictionCount)) {
    return Status::ArithmeticOverflow;
  }
  if (!outputCanHold(output, evictionCount)) {
    return output.records == nullptr ? Status::ReleaseOutputMissing
                                     : Status::ReleaseOutputTooSmall;
  }

  std::array<CommittedMetadata, kMaximumCommittedEntries> nextCommitted =
      committed_;
  std::size_t nextCount = committedCount_;
  uint64_t nextResident = residentBytes_;
  for (std::size_t index = 0; index < nextCommitted.size(); ++index) {
    if (!proposedEvicted[index] || !nextCommitted[index].occupied) continue;
    if (nextCommitted[index].byteSize > nextResident) {
      return Status::ArithmeticOverflow;
    }
    nextResident -= nextCommitted[index].byteSize;
    nextCommitted[index] = CommittedMetadata{};
    if (nextCount == 0u) return Status::ArithmeticOverflow;
    --nextCount;
  }

  for (std::size_t index = 0; index < nextCommitted.size(); ++index) {
    if (!nextCommitted[index].occupied || !pinned_[index]) continue;
    nextCommitted[index].lastUseEpoch = transactionEpoch_;
  }

  for (std::size_t stagedIndex = 0; stagedIndex < stagedCount_;
       ++stagedIndex) {
    std::size_t freeSlot = kInvalidIndex;
    for (std::size_t index = 0; index < nextCommitted.size(); ++index) {
      if (!nextCommitted[index].occupied) {
        freeSlot = index;
        break;
      }
    }
    if (freeSlot == kInvalidIndex || nextCount >= kMaximumCommittedEntries) {
      return Status::CommittedCapacityExceeded;
    }
    const CandidateMetadata& candidate = staged_[stagedIndex];
    CommittedMetadata committed{};
    committed.key = candidate.key;
    committed.cacheId = candidate.cacheId;
    committed.byteSize = candidate.byteSize;
    committed.lastUseEpoch = transactionEpoch_;
    committed.occupied = true;
    nextCommitted[freeSlot] = committed;
    ++nextCount;
    if (!addExact(nextResident, committed.byteSize, &nextResident)) {
      return Status::ArithmeticOverflow;
    }
  }
  if (nextCount > kMaximumCommittedEntries) {
    return Status::CommittedCapacityExceeded;
  }
  if (nextResident > config_.maxResidentBytes) {
    return Status::ResidentBudgetExceeded;
  }

  committed_ = nextCommitted;
  committedCount_ = nextCount;
  residentBytes_ = nextResident;
  evicted_ = proposedEvicted;
  copyReleases(releases, localReleases);
  clearTransaction();
  return Status::Ok;
}

ReleaseResult DerivedCache::commit() noexcept {
  ReleaseResult result{};
  result.status = commit(&result.releases);
  return result;
}

Status DerivedCache::commit(ReleaseRecord* records,
                            std::size_t capacity,
                            std::size_t* count) noexcept {
  if (count != nullptr) *count = 0u;
  if (shutdown_) return Status::Shutdown;
  if (!transactionActive_) return Status::TransactionNotActive;
  if (records == nullptr && capacity != 0u) return Status::ReleaseOutputMissing;

  auto proposedEvicted = evicted_;
  uint64_t projected = 0u;
  if (!planEvictions(stagedBytes_, stagedCount_, pinned_, &proposedEvicted,
                     &projected)) {
    return committedCount_ + stagedCount_ > kMaximumCommittedEntries
               ? Status::CommittedCapacityExceeded
               : Status::ResidentBudgetExceeded;
  }
  std::size_t required = 0u;
  for (std::size_t index = 0; index < proposedEvicted.size(); ++index) {
    if (proposedEvicted[index] && committed_[index].occupied) ++required;
  }
  const Output callerOutput{records, capacity, count};
  if (!outputCanHold(callerOutput, required)) {
    return records == nullptr ? Status::ReleaseOutputMissing
                              : Status::ReleaseOutputTooSmall;
  }

  ReleaseList full{};
  const Status status = commit(&full);
  if (status != Status::Ok) return status;
  if (full.count > capacity || (full.count != 0u && records == nullptr)) {
    return full.count != 0u ? Status::ReleaseOutputTooSmall
                            : Status::ReleaseOutputMissing;
  }
  for (std::size_t index = 0; index < full.count; ++index) {
    records[index] = full.records[index];
  }
  if (count != nullptr) *count = full.count;
  return Status::Ok;
}

Status DerivedCache::finishAbort(Output output) noexcept {
  if (shutdown_) return Status::Shutdown;
  if (!transactionActive_) return Status::TransactionNotActive;
  std::size_t required = 0u;
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) ++required;
  }
  if (!outputCanHold(output, required)) {
    return output.records == nullptr ? Status::ReleaseOutputMissing
                                     : Status::ReleaseOutputTooSmall;
  }

  ReleaseList local{};
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId == 0u) continue;
    local.records[local.count++] =
        ReleaseRecord{staged_[index].cacheId, staged_[index].byteSize,
                      staged_[index].key.family};
  }
  if (output.count != nullptr) *output.count = local.count;
  for (std::size_t index = 0; index < local.count; ++index) {
    output.records[index] = local.records[index];
  }
  clearTransaction();
  return Status::Ok;
}

Status DerivedCache::abort(ReleaseList* releases) noexcept {
  if (releases != nullptr) releases->clear();
  Output output{};
  if (releases != nullptr) {
    output.records = releases->records.data();
    output.capacity = releases->records.size();
    output.count = &releases->count;
  }
  return finishAbort(output);
}

ReleaseResult DerivedCache::abort() noexcept {
  ReleaseResult result{};
  result.status = abort(&result.releases);
  return result;
}

Status DerivedCache::abort(ReleaseRecord* records,
                           std::size_t capacity,
                           std::size_t* count) noexcept {
  if (count != nullptr) *count = 0u;
  if (records == nullptr && capacity != 0u) return Status::ReleaseOutputMissing;
  return finishAbort(Output{records, capacity, count});
}

Status DerivedCache::shutdown(ReleaseList* releases) noexcept {
  if (releases != nullptr) releases->clear();
  if (shutdown_) return Status::AlreadyShutdown;

  Output output{};
  if (releases != nullptr) {
    output.records = releases->records.data();
    output.capacity = releases->records.size();
    output.count = &releases->count;
  }
  std::size_t required = 0u;
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId != 0u) ++required;
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) ++required;
  }
  if (!outputCanHold(output, required)) {
    return output.records == nullptr ? Status::ReleaseOutputMissing
                                     : Status::ReleaseOutputTooSmall;
  }

  ReleaseList local{};
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId != 0u) {
      local.records[local.count++] =
          ReleaseRecord{entry.cacheId, entry.byteSize, entry.key.family};
    }
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) {
      local.records[local.count++] =
          ReleaseRecord{staged_[index].cacheId, staged_[index].byteSize,
                        staged_[index].key.family};
    }
  }
  for (std::size_t index = 0; index < local.count; ++index) {
    output.records[index] = local.records[index];
  }
  if (output.count != nullptr) *output.count = local.count;
  clearAll();
  shutdown_ = true;
  return Status::Ok;
}

ReleaseResult DerivedCache::shutdown() noexcept {
  ReleaseResult result{};
  result.status = shutdown(&result.releases);
  return result;
}

Status DerivedCache::shutdown(ReleaseRecord* records,
                              std::size_t capacity,
                              std::size_t* count) noexcept {
  if (count != nullptr) *count = 0u;
  if (records == nullptr && capacity != 0u) return Status::ReleaseOutputMissing;
  if (shutdown_) return Status::AlreadyShutdown;
  std::size_t required = 0u;
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId != 0u) ++required;
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) ++required;
  }
  if (!outputCanHold(Output{records, capacity, count}, required)) {
    return records == nullptr ? Status::ReleaseOutputMissing
                              : Status::ReleaseOutputTooSmall;
  }
  ReleaseList full{};
  const Status status = shutdown(&full);
  if (status != Status::Ok) return status;
  for (std::size_t index = 0; index < full.count; ++index) {
    records[index] = full.records[index];
  }
  if (count != nullptr) *count = full.count;
  return Status::Ok;
}

Status DerivedCache::reset(ReleaseList* releases) noexcept {
  if (releases != nullptr) releases->clear();
  if (shutdown_) return Status::Shutdown;

  Output output{};
  if (releases != nullptr) {
    output.records = releases->records.data();
    output.capacity = releases->records.size();
    output.count = &releases->count;
  }
  std::size_t required = 0u;
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId != 0u) ++required;
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) ++required;
  }
  if (!outputCanHold(output, required)) {
    return output.records == nullptr ? Status::ReleaseOutputMissing
                                     : Status::ReleaseOutputTooSmall;
  }

  ReleaseList local{};
  for (const auto& entry : committed_) {
    if (entry.occupied && entry.cacheId != 0u) {
      local.records[local.count++] =
          ReleaseRecord{entry.cacheId, entry.byteSize, entry.key.family};
    }
  }
  for (std::size_t index = 0; index < stagedCount_; ++index) {
    if (staged_[index].cacheId != 0u) {
      local.records[local.count++] =
          ReleaseRecord{staged_[index].cacheId, staged_[index].byteSize,
                        staged_[index].key.family};
    }
  }
  for (std::size_t index = 0; index < local.count; ++index) {
    output.records[index] = local.records[index];
  }
  if (output.count != nullptr) *output.count = local.count;
  clearAll();
  return Status::Ok;
}

ReleaseResult DerivedCache::reset() noexcept {
  ReleaseResult result{};
  result.status = reset(&result.releases);
  return result;
}

}  // namespace ChromaspaceMetalDerivedCache
