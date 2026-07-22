#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

// Portable submission-owned resource lifetime policy.  The native Metal
// backend supplies the opaque objects; this module owns only a bounded set of
// strong references and never grows a heap-backed container after construction.
namespace ChromaspaceMetalSubmissionRetention {

// Native frames can expose up to 64 visible plot surfaces, one atlas, and up
// to 64 independently resident derived records.  Candidate/committed overlap
// may add one more derived record per family, so 256 leaves bounded headroom
// without allowing an unbounded container into the render path.
constexpr std::size_t kMaximumRetainedResources = 256u;

enum class ResourceKind : std::uint8_t {
  PlotSurface = 0,
  TextAtlas,
  DerivedRecord,
};

struct ResourceKey {
  ResourceKind kind = ResourceKind::PlotSurface;
  std::uint64_t resourceId = 0u;
  std::uint64_t ownerCompositorId = 0u;
};

constexpr bool operator==(const ResourceKey& left,
                          const ResourceKey& right) noexcept {
  return left.kind == right.kind && left.resourceId == right.resourceId &&
         left.ownerCompositorId == right.ownerCompositorId;
}

constexpr bool operator!=(const ResourceKey& left,
                          const ResourceKey& right) noexcept {
  return !(left == right);
}

enum class Status : std::uint8_t {
  Retained = 0,
  Duplicate,
  InvalidKey,
  NullResource,
  CapacityExhausted,
  KeyConflict,
  Sealed,
  InvalidCapacity,
  NotFound,
};

const char* statusLabel(Status status) noexcept;

constexpr bool succeeded(Status status) noexcept {
  return status == Status::Retained || status == Status::Duplicate;
}

struct EntrySnapshot {
  ResourceKey key{};
  const void* pointer = nullptr;
  bool occupied = false;
};

struct Snapshot {
  std::array<EntrySnapshot, kMaximumRetainedResources> entries{};
  std::size_t count = 0u;
  std::size_t capacity = 0u;
  bool sealed = false;
  bool valid = false;
};

class RetentionSet final {
 public:
  // A smaller capacity is useful for focused policy tests.  Every slot still
  // lives in the fixed inline array; construction performs no allocation.
  explicit RetentionSet(
      std::size_t capacity = kMaximumRetainedResources) noexcept;
  ~RetentionSet() noexcept;

  RetentionSet(const RetentionSet&) = delete;
  RetentionSet& operator=(const RetentionSet&) = delete;
  RetentionSet(RetentionSet&& other) noexcept;
  RetentionSet& operator=(RetentionSet&& other) noexcept;

  static bool validKey(const ResourceKey& key) noexcept;

  // retain() is allocation-free after construction.  Retained and Duplicate
  // are successful outcomes; all other statuses leave the set unchanged.
  Status retain(const ResourceKey& key,
                const std::shared_ptr<void>& resource) noexcept;

  // Used only by native rollback paths that installed a resource before a
  // later transaction step failed.  Missing keys are harmlessly reported.
  Status release(const ResourceKey& key) noexcept;

  // A sealed set rejects new resources while allowing duplicate observations.
  // Native commit seals before handing the set to completion-owned state.
  void seal() noexcept { sealed_ = true; }
  bool sealed() const noexcept { return sealed_; }

  // reset() releases every strong reference and returns the set to its empty,
  // reusable state.  It is safe to call repeatedly.
  void reset() noexcept;

  std::size_t count() const noexcept { return count_; }
  std::size_t capacity() const noexcept { return capacity_; }
  bool valid() const noexcept { return valid_; }
  Snapshot snapshot() const noexcept;

 private:
  struct Entry {
    ResourceKey key{};
    std::shared_ptr<void> resource;
    bool occupied = false;

    void clear() noexcept {
      resource.reset();
      key = ResourceKey{};
      occupied = false;
    }
  };

  static constexpr std::size_t kInvalidIndex =
      static_cast<std::size_t>(-1);

  std::size_t find(const ResourceKey& key) const noexcept;
  std::size_t findEmpty() const noexcept;
  void moveFrom(RetentionSet&& other) noexcept;

  std::array<Entry, kMaximumRetainedResources> entries_{};
  std::size_t capacity_ = 0u;
  std::size_t count_ = 0u;
  bool sealed_ = false;
  bool valid_ = false;
};

using SubmissionRetention = RetentionSet;

}  // namespace ChromaspaceMetalSubmissionRetention
