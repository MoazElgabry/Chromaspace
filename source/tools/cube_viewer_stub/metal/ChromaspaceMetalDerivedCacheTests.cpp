#include "ChromaspaceMetalDerivedCache.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

using namespace ChromaspaceMetalDerivedCache;

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::abort();
  }
}

DerivedKey key(uint64_t id,
               Family family = Family::Histogram,
               uint64_t derivation = 1u) {
  DerivedKey result{};
  result.sourceId = id;
  result.deviceRegistryId = 100u;
  result.senderGeneration = 200u;
  result.sequence = id + 300u;
  result.slotIndex = static_cast<uint32_t>(id % 8u);
  result.slotGeneration = id + 400u;
  result.contentHash = id + 500u;
  result.family = family;
  result.derivationHash = derivation;
  return result;
}

void materializeAndCommit(DerivedCache* cache,
                          const DerivedKey& resourceKey,
                          uint64_t cacheId,
                          uint64_t bytes,
                          uint64_t epoch) {
  expect(cache->begin(epoch) == Status::Ok, "begin helper");
  const AcquireResult acquisition = cache->acquire(resourceKey, bytes);
  expect(acquisition.status == Status::Candidate &&
             acquisition.stagedIndex != kInvalidIndex,
         "candidate helper");
  expect(cache->materializeCandidate(acquisition.stagedIndex, cacheId, bytes) ==
             Status::Ok,
         "materialize helper");
  expect(cache->commit() == Status::Ok, "commit helper");
}

void invalidConfigAndTokens() {
  Config zero{};
  zero.maxResidentBytes = 0u;
  expect(!DerivedCache::validateConfig(zero), "zero resident invalid");
  Config reversed{};
  reversed.maxResidentBytes = 200u;
  reversed.maxTransientBytes = 100u;
  expect(!DerivedCache::validateConfig(reversed),
         "transient below resident invalid");
  DerivedCache cache(zero);
  expect(cache.configStatus() == Status::InvalidConfig,
         "invalid config status");
  expect(cache.begin(1u) == Status::InvalidConfig,
         "invalid config rejects begin");

  DerivedCache valid;
  expect(valid.begin(0u) == Status::InvalidEpoch, "zero epoch invalid");
  expect(valid.acquire(key(1u), 1u).status == Status::TransactionNotActive,
         "acquire without transaction rejected");
  expect(valid.begin(1u) == Status::Ok, "valid begin");
  expect(valid.begin(2u) == Status::TransactionAlreadyActive,
         "nested transaction rejected");
  expect(valid.acquire(DerivedKey{}, 1u).status == Status::InvalidKey,
         "zero key rejected");
  DerivedKey invalidFamily = key(2u);
  invalidFamily.family = static_cast<Family>(99u);
  expect(valid.acquire(invalidFamily, 1u).status == Status::InvalidKey,
         "unknown family rejected");
  expect(valid.acquire(key(3u), 0u).status == Status::InvalidEstimate,
         "zero estimate rejected");
  expect(valid.materializeCandidate(kInvalidIndex, 1u, 1u) ==
             Status::InvalidCandidateIndex,
         "invalid staged index rejected");
  expect(valid.abort() == Status::Ok, "invalid token transaction abort");
}

void allFamiliesAndReleaseMetadata() {
  DerivedCache cache;
  expect(cache.begin(1u) == Status::Ok, "family begin");
  const Family families[] = {Family::Histogram, Family::Waveform,
                             Family::RasterPointCloud, Family::GlossField};
  for (std::size_t index = 0; index < 4u; ++index) {
    const AcquireResult acquisition =
        cache.acquire(key(10u + index, families[index]), 10u + index);
    expect(acquisition.status == Status::Candidate, "family candidate");
    expect(cache.materializeCandidate(acquisition.stagedIndex, 100u + index,
                                      10u + index) == Status::Ok,
           "family materialize");
  }
  const ReleaseResult committed = cache.commit();
  expect(committed.status == Status::Ok && committed.releases.count == 0u,
         "family commit has no evictions");
  expect(cache.committedCount() == 4u &&
             cache.committedCount(Family::Histogram) == 1u &&
             cache.committedCount(Family::GlossField) == 1u,
         "family-aware committed counts");
  const ReleaseResult reset = cache.reset();
  expect(reset.status == Status::Ok && reset.releases.count == 4u,
         "reset releases all families");
  expect(cache.committedCount(Family::GlossField) == 0u,
         "reset clears family count");
  for (std::size_t index = 0; index < reset.releases.count; ++index) {
    expect(reset.releases.records[index].family == families[index],
           "release carries family");
  }
}

void hitAndCandidateDeduplication() {
  DerivedCache cache;
  const DerivedKey resourceKey = key(20u, Family::Waveform, 0x44u);
  materializeAndCommit(&cache, resourceKey, 201u, 100u, 1u);
  expect(cache.committedCount() == 1u && cache.residentByteSize() == 100u,
         "committed metadata");

  expect(cache.begin(2u) == Status::Ok, "hit begin");
  const AcquireResult hit = cache.acquire(resourceKey, 100u);
  expect(hit.status == Status::Hit && hit.kind == AcquireKind::Hit &&
             hit.cacheId == 201u && hit.stagedIndex == kInvalidIndex,
         "committed hit");
  expect(cache.commit() == Status::Ok, "hit commit");
  const CommittedMetadata* hitEntry = cache.committedEntry(hit.committedIndex);
  expect(hitEntry != nullptr && hitEntry->lastUseEpoch == 2u,
         "hit use epoch updates only on commit");

  expect(cache.begin(3u) == Status::Ok, "dedup begin");
  const AcquireResult first = cache.acquire(key(21u), 20u);
  const AcquireResult second = cache.acquire(key(21u), 20u);
  expect(first.status == Status::Candidate && second.status == Status::Candidate &&
             first.stagedIndex == second.stagedIndex && second.reused,
         "duplicate candidate deduplicates");
  expect(cache.stagedCount() == 1u && cache.transientByteSize() == 120u,
         "dedup reserves once");
  expect(cache.materializeCandidate(first.stagedIndex, 202u, 20u) == Status::Ok,
         "dedup materialize");
  const AcquireResult materialized = cache.acquire(key(21u), 20u);
  expect(materialized.cacheId == 202u && materialized.reused,
         "dedup retains materialized metadata");
  const ReleaseResult aborted = cache.abort();
  expect(aborted.status == Status::Ok && aborted.releases.count == 1u &&
             aborted.releases.records[0].cacheId == 202u &&
             aborted.releases.records[0].family == Family::Histogram,
         "abort releases only staged candidate");
  expect(cache.committedSlotForKey(key(21u)) == kInvalidIndex,
         "abort preserves committed set");
}

void lruPinningAndTieBreak() {
  Config config{};
  config.maxResidentBytes = 300u;
  config.maxTransientBytes = 600u;
  DerivedCache cache(config);
  materializeAndCommit(&cache, key(30u), 301u, 100u, 1u);
  materializeAndCommit(&cache, key(31u), 302u, 100u, 1u);
  materializeAndCommit(&cache, key(32u), 303u, 100u, 1u);

  expect(cache.begin(2u) == Status::Ok, "lru begin");
  const AcquireResult replacement = cache.acquire(key(33u), 200u);
  expect(replacement.status == Status::Candidate &&
             cache.selectedEvictionCount() == 2u,
         "two lru evictions selected");
  expect(cache.materializeCandidate(replacement.stagedIndex, 304u, 200u) ==
             Status::Ok,
         "lru materialize");
  const ReleaseResult committed = cache.commit();
  expect(committed.status == Status::Ok && committed.releases.count == 2u &&
             committed.releases.records[0].cacheId == 301u &&
             committed.releases.records[1].cacheId == 302u,
         "lru tie break is stable slot order");
  expect(cache.committedSlotForKey(key(32u)) != kInvalidIndex,
         "newest tied entry survives");

  DerivedCache pinned(config);
  materializeAndCommit(&pinned, key(40u), 401u, 100u, 1u);
  materializeAndCommit(&pinned, key(41u), 402u, 100u, 1u);
  materializeAndCommit(&pinned, key(42u), 403u, 100u, 1u);
  expect(pinned.begin(2u) == Status::Ok, "pinned begin");
  expect(pinned.acquire(key(40u), 100u).status == Status::Hit,
         "hit pins entry");
  const AcquireResult next = pinned.acquire(key(43u), 200u);
  expect(next.status == Status::Candidate &&
             pinned.selectedEvictionCount() == 2u,
         "pinned candidate evicts only unpinned entries");
  expect(pinned.materializeCandidate(next.stagedIndex, 404u, 200u) ==
             Status::Ok,
         "pinned materialize");
  const ReleaseResult pinnedCommit = pinned.commit();
  expect(pinnedCommit.releases.count == 2u &&
             pinnedCommit.releases.records[0].cacheId == 402u &&
             pinnedCommit.releases.records[1].cacheId == 403u &&
             pinned.committedSlotForKey(key(40u)) != kInvalidIndex,
         "pinned entry survives");
}

void budgetAndActualRevalidation() {
  Config config{};
  config.maxResidentBytes = 100u;
  config.maxTransientBytes = 150u;
  DerivedCache cache(config);
  materializeAndCommit(&cache, key(50u), 501u, 100u, 1u);

  expect(cache.begin(2u) == Status::Ok, "budget begin");
  const AcquireResult candidate = cache.acquire(key(51u), 40u);
  expect(candidate.status == Status::Candidate &&
             cache.transientByteSize() == 140u,
         "transient includes resident and estimate");
  const AcquireResult transientFailure = cache.acquire(key(52u), 20u);
  expect(transientFailure.status == Status::TransientBudgetExceeded,
         "transient budget enforced");
  expect(cache.materializeCandidate(candidate.stagedIndex, 502u, 5u) ==
             Status::ActualSizeTooSmall,
         "actual below estimate rejected");
  const CandidateMetadata* unchanged =
      cache.stagedAcquisition(candidate.stagedIndex);
  expect(unchanged != nullptr && unchanged->cacheId == 0u &&
             unchanged->byteSize == 40u && cache.transientByteSize() == 140u,
         "failed materialization is transactional");
  expect(cache.materializeCandidate(candidate.stagedIndex, 502u, 40u) ==
             Status::Ok,
         "actual estimate accepted");
  expect(cache.commit() == Status::Ok, "budget commit");
  expect(cache.committedSlotForKey(key(50u)) == kInvalidIndex,
         "resident replacement evicts old resource");

  Config pinnedConfig{};
  pinnedConfig.maxResidentBytes = 100u;
  pinnedConfig.maxTransientBytes = 300u;
  DerivedCache pinned(pinnedConfig);
  materializeAndCommit(&pinned, key(60u), 601u, 50u, 1u);
  materializeAndCommit(&pinned, key(61u), 602u, 50u, 1u);
  expect(pinned.begin(2u) == Status::Ok, "pinned budget begin");
  expect(pinned.acquire(key(60u), 50u).status == Status::Hit,
         "budget pin hit");
  expect(pinned.acquire(key(62u), 70u).status ==
             Status::ResidentBudgetExceeded,
         "pinned resident budget failure");
  expect(pinned.abort() == Status::Ok, "pinned budget abort");
}

void overflowAndCapacity() {
  Config huge{};
  huge.maxResidentBytes = (std::numeric_limits<uint64_t>::max)();
  huge.maxTransientBytes = (std::numeric_limits<uint64_t>::max)();
  DerivedCache overflow(huge);
  expect(overflow.begin(1u) == Status::Ok, "overflow begin");
  const AcquireResult maxCandidate =
      overflow.acquire(key(70u), (std::numeric_limits<uint64_t>::max)());
  expect(maxCandidate.status == Status::Candidate, "maximum estimate accepted");
  expect(overflow.acquire(key(71u), 1u).status == Status::ArithmeticOverflow,
         "staged byte addition overflow rejected");
  expect(overflow.materializeCandidate(maxCandidate.stagedIndex, 701u,
                                       (std::numeric_limits<uint64_t>::max)()) ==
             Status::Ok,
         "maximum actual accepted");
  expect(overflow.abort().releases.count == 1u,
         "overflow abort releases materialized candidate");

  Config capacityConfig{};
  capacityConfig.maxResidentBytes = 128u;
  capacityConfig.maxTransientBytes = 256u;
  DerivedCache full(capacityConfig);
  for (std::size_t index = 0; index < kMaximumCommittedEntries; ++index) {
    materializeAndCommit(&full, key(1000u + index), 2000u + index, 1u,
                         static_cast<uint64_t>(index + 1u));
  }
  expect(full.committedCount() == kMaximumCommittedEntries,
         "committed capacity fills");
  expect(full.begin(1000u) == Status::Ok, "full begin");
  for (std::size_t index = 0; index < kMaximumCommittedEntries; ++index) {
    expect(full.acquire(key(1000u + index), 1u).status == Status::Hit,
           "full entry pinned");
  }
  expect(full.acquire(key(5000u), 1u).status ==
             Status::CommittedCapacityExceeded,
         "capacity reports pinned-full failure");
  expect(full.abort() == Status::Ok, "full abort");

  DerivedCache staged;
  expect(staged.begin(1u) == Status::Ok, "staged capacity begin");
  for (std::size_t index = 0; index < kMaximumStagedAcquisitions; ++index) {
    expect(staged.acquire(key(6000u + index), 1u).status ==
               Status::Candidate,
           "staged candidate accepted");
  }
  expect(staged.acquire(key(7000u), 1u).status ==
             Status::StagedCapacityExceeded,
         "staged capacity rejects overflow");
  expect(staged.abort() == Status::Ok, "staged capacity abort");
}

void abortOutputAndResetReuse() {
  DerivedCache cache;
  materializeAndCommit(&cache, key(80u), 801u, 10u, 1u);
  expect(cache.begin(2u) == Status::Ok, "abort output begin");
  const AcquireResult candidate =
      cache.acquire(key(81u, Family::RasterPointCloud), 20u);
  expect(cache.materializeCandidate(candidate.stagedIndex, 802u, 20u) ==
             Status::Ok,
         "abort output materialize");
  ReleaseRecord one{};
  std::size_t count = 0u;
  expect(cache.abort(&one, 0u, &count) == Status::ReleaseOutputTooSmall &&
             cache.transactionActive() && count == 0u,
         "small abort output preserves transaction");
  expect(cache.abort(&one, 1u, &count) == Status::Ok && count == 1u &&
             one.cacheId == 802u && one.family == Family::RasterPointCloud,
         "abort output exact release");
  expect(cache.committedSlotForKey(key(80u)) != kInvalidIndex,
         "abort preserves committed resource");

  const ReleaseResult reset = cache.reset();
  expect(reset.status == Status::Ok && reset.releases.count == 1u &&
             reset.releases.records[0].cacheId == 801u,
         "reset drains committed resource");
  expect(cache.begin(3u) == Status::Ok, "reset cache reusable");
  expect(cache.acquire(key(82u), 4u).status == Status::Candidate,
         "reusable cache candidate");
  expect(cache.abort() == Status::Ok, "reusable cache abort");
}

void shutdownIsTerminalAndIdempotent() {
  DerivedCache cache;
  materializeAndCommit(&cache, key(90u, Family::Waveform), 901u, 10u, 1u);
  expect(cache.begin(2u) == Status::Ok, "shutdown staged begin");
  const AcquireResult staged =
      cache.acquire(key(91u, Family::RasterPointCloud), 5u);
  expect(cache.materializeCandidate(staged.stagedIndex, 902u, 5u) ==
             Status::Ok,
         "shutdown staged materialize");
  const ReleaseResult first = cache.shutdown();
  expect(first.status == Status::Ok && first.releases.count == 2u,
         "shutdown releases committed and staged");
  expect(first.releases.records[0].family == Family::Waveform &&
             first.releases.records[1].family == Family::RasterPointCloud,
         "shutdown release families");
  const ReleaseResult second = cache.shutdown();
  expect(second.status == Status::AlreadyShutdown &&
             second.releases.count == 0u,
         "shutdown idempotent");
  expect(cache.begin(3u) == Status::Shutdown, "shutdown terminal begin");
  expect(cache.reset() == Status::Shutdown, "shutdown terminal reset");
  expect(cache.abort() == Status::Shutdown, "shutdown terminal abort");
}

void releaseOutputCapacityAndCacheIdRules() {
  Config config{};
  config.maxResidentBytes = 10u;
  config.maxTransientBytes = 30u;
  DerivedCache cache(config);
  materializeAndCommit(&cache, key(100u), 1001u, 10u, 1u);
  expect(cache.begin(2u) == Status::Ok, "release capacity begin");
  const AcquireResult candidate = cache.acquire(key(101u), 10u);
  expect(cache.materializeCandidate(candidate.stagedIndex, 1002u, 10u) ==
             Status::Ok,
         "release capacity materialize");
  ReleaseRecord record{};
  std::size_t count = 0u;
  expect(cache.commit(&record, 0u, &count) == Status::ReleaseOutputTooSmall &&
             cache.transactionActive() && count == 0u,
         "small commit output preserves transaction");
  expect(cache.commit(&record, 1u, &count) == Status::Ok && count == 1u &&
             record.cacheId == 1001u && record.family == Family::Histogram,
         "commit output includes evicted family");

  expect(cache.begin(3u) == Status::Ok, "cache id rules begin");
  const AcquireResult a = cache.acquire(key(102u), 2u);
  expect(cache.materializeCandidate(a.stagedIndex, 0u, 2u) ==
             Status::InvalidCacheId,
         "zero cache id rejected");
  expect(cache.materializeCandidate(a.stagedIndex, 1001u, 2u) ==
             Status::Ok,
         "released cache id can be reused");
  expect(cache.materializeCandidate(a.stagedIndex, 1001u, 2u) ==
             Status::CandidateAlreadyMaterialized,
         "candidate cannot materialize twice");
  expect(cache.abort().releases.count == 1u, "cache id abort release");
}

}  // namespace

int main() {
  invalidConfigAndTokens();
  allFamiliesAndReleaseMetadata();
  hitAndCandidateDeduplication();
  lruPinningAndTieBreak();
  budgetAndActualRevalidation();
  overflowAndCapacity();
  abortOutputAndResetReuse();
  shutdownIsTerminalAndIdempotent();
  releaseOutputCapacityAndCacheIdRules();
  std::cout << "ChromaspaceMetalDerivedCacheTests passed\n";
  return 0;
}
