#include "ChromaspaceMetalSubmissionRetention.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

using namespace ChromaspaceMetalSubmissionRetention;

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::abort();
  }
}

ResourceKey key(ResourceKind kind, std::uint64_t resourceId,
                std::uint64_t owner = 77u) {
  return ResourceKey{kind, resourceId, owner};
}

std::shared_ptr<void> resource(int value) {
  return std::static_pointer_cast<void>(std::make_shared<int>(value));
}

void moveCommitLikeHandoffKeepsStrongLifetime() {
  int destroyed = 0;
  auto object = std::shared_ptr<int>(new int(5), [&destroyed](int* pointer) {
    ++destroyed;
    delete pointer;
  });
  RetentionSet encoding(2u);
  expect(encoding.retain(key(ResourceKind::PlotSurface, 1u), object) ==
             Status::Retained,
         "encoding retains object");
  RetentionSet completion(std::move(encoding));
  expect(encoding.count() == 0u && completion.count() == 1u,
         "move transfers bounded entry");
  object.reset();
  expect(destroyed == 0, "completion handoff keeps object alive");
  completion.seal();
  expect(completion.retain(key(ResourceKind::TextAtlas, 2u), resource(2)) ==
             Status::Sealed,
         "sealed completion rejects new resource");
  completion.reset();
  expect(destroyed == 1, "completion reset releases final hold");
}

void duplicateAndConflictAreExplicit() {
  RetentionSet set(3u);
  const auto first = resource(1);
  expect(set.retain(key(ResourceKind::PlotSurface, 1u), first) ==
             Status::Retained,
         "first key accepted");
  expect(set.retain(key(ResourceKind::PlotSurface, 1u), first) ==
             Status::Duplicate,
         "identical key and pointer deduplicated");
  expect(set.count() == 1u, "duplicate does not consume capacity");
  expect(set.retain(key(ResourceKind::PlotSurface, 1u), resource(2)) ==
             Status::KeyConflict,
         "same key with different pointer rejected");
  expect(set.count() == 1u, "conflict leaves set unchanged");
}

void invalidNullAndCapacityFailuresLeaveState() {
  RetentionSet set(2u);
  const auto object = resource(1);
  expect(set.retain(ResourceKey{}, object) == Status::InvalidKey,
         "zero key rejected");
  expect(set.retain(key(ResourceKind::TextAtlas, 1u), {}) ==
             Status::NullResource,
         "null resource rejected");
  expect(set.retain(key(ResourceKind::TextAtlas, 1u), object) ==
             Status::Retained,
         "first capacity slot accepted");
  expect(set.retain(key(ResourceKind::DerivedRecord, 2u), resource(2)) ==
             Status::Retained,
         "second capacity slot accepted");
  const Snapshot before = set.snapshot();
  expect(set.retain(key(ResourceKind::PlotSurface, 3u), resource(3)) ==
             Status::CapacityExhausted,
         "capacity exhaustion rejected");
  const Snapshot after = set.snapshot();
  expect(after.count == before.count && after.capacity == before.capacity,
         "capacity rejection leaves snapshot unchanged");
  RetentionSet invalid(0u);
  expect(!invalid.valid() && invalid.retain(key(ResourceKind::PlotSurface, 9u),
                                            object) ==
             Status::InvalidCapacity,
         "zero capacity is invalid");
}

void resetAndIndependentSetsReleaseSeparately() {
  int firstDestroyed = 0;
  int secondDestroyed = 0;
  auto first = std::shared_ptr<int>(new int(1), [&firstDestroyed](int* p) {
    ++firstDestroyed;
    delete p;
  });
  auto second = std::shared_ptr<int>(new int(2), [&secondDestroyed](int* p) {
    ++secondDestroyed;
    delete p;
  });
  RetentionSet submissionA(1u);
  RetentionSet submissionB(1u);
  expect(submissionA.retain(key(ResourceKind::DerivedRecord, 1u), first) ==
             Status::Retained,
         "submission A retains independently");
  expect(submissionB.retain(key(ResourceKind::DerivedRecord, 1u), second) ==
             Status::Retained,
         "submission B accepts same ID independently");
  first.reset();
  second.reset();
  submissionA.reset();
  expect(firstDestroyed == 1 && secondDestroyed == 0,
         "reset releases only owning submission");
  submissionB.reset();
  expect(secondDestroyed == 1, "second submission releases independently");
}

void snapshotIsBoundedAndReleaseIsTransactional() {
  RetentionSet set(2u);
  const auto object = resource(4);
  const ResourceKey first = key(ResourceKind::PlotSurface, 4u);
  expect(set.retain(first, object) == Status::Retained,
         "snapshot entry accepted");
  const Snapshot snapshot = set.snapshot();
  expect(snapshot.valid && snapshot.count == 1u && snapshot.capacity == 2u &&
             snapshot.entries[0].occupied &&
             snapshot.entries[0].pointer == object.get(),
         "snapshot exposes bounded pointer and count");
  expect(set.release(key(ResourceKind::TextAtlas, 99u)) == Status::NotFound,
         "missing release is explicit");
  expect(set.release(first) == Status::Retained && set.count() == 0u,
         "release removes one entry");
}

void legitimateNativeFrameFitsAndActualBoundFailsClosed() {
  RetentionSet set;
  for (std::uint64_t id = 1u; id <= 64u; ++id) {
    expect(set.retain(key(ResourceKind::PlotSurface, id), resource(1)) ==
               Status::Retained,
           "native frame plot-surface budget accepted");
  }
  expect(set.retain(key(ResourceKind::TextAtlas, 65u), resource(2)) ==
             Status::Retained,
         "native frame atlas accepted");
  for (std::uint64_t id = 66u; id <= 129u; ++id) {
    expect(set.retain(key(ResourceKind::DerivedRecord, id), resource(3)) ==
               Status::Retained,
           "native frame derived budget accepted");
  }
  expect(set.count() == 129u, "full legitimate frame fits in fixed policy");
  for (std::uint64_t id = 130u; id <=
                              static_cast<std::uint64_t>(
                                  kMaximumRetainedResources);
       ++id) {
    expect(set.retain(key(ResourceKind::DerivedRecord, id), resource(4)) ==
               Status::Retained,
           "retention remains available through actual bound");
  }
  expect(set.count() == kMaximumRetainedResources,
         "fixed policy reaches exactly its declared bound");
  expect(set.retain(
             key(ResourceKind::DerivedRecord,
                 static_cast<std::uint64_t>(kMaximumRetainedResources) + 1u),
             resource(5)) == Status::CapacityExhausted,
         "retention fails closed beyond actual bound");
}

}  // namespace

int main() {
  moveCommitLikeHandoffKeepsStrongLifetime();
  duplicateAndConflictAreExplicit();
  invalidNullAndCapacityFailuresLeaveState();
  resetAndIndependentSetsReleaseSeparately();
  snapshotIsBoundedAndReleaseIsTransactional();
  legitimateNativeFrameFitsAndActualBoundFailsClosed();
  std::cout << "ChromaspaceMetalSubmissionRetentionTests passed\n";
  return 0;
}
