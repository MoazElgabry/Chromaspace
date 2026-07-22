#include "ChromaspaceMetalSubmissionRetention.h"

namespace ChromaspaceMetalSubmissionRetention {
namespace {

bool knownKind(ResourceKind kind) noexcept {
  switch (kind) {
    case ResourceKind::PlotSurface:
    case ResourceKind::TextAtlas:
    case ResourceKind::DerivedRecord:
      return true;
  }
  return false;
}

}  // namespace

const char* statusLabel(Status status) noexcept {
  switch (status) {
    case Status::Retained: return "retained";
    case Status::Duplicate: return "duplicate";
    case Status::InvalidKey: return "invalid-key";
    case Status::NullResource: return "null-resource";
    case Status::CapacityExhausted: return "capacity-exhausted";
    case Status::KeyConflict: return "key-conflict";
    case Status::Sealed: return "sealed";
    case Status::InvalidCapacity: return "invalid-capacity";
    case Status::NotFound: return "not-found";
  }
  return "unknown";
}

RetentionSet::RetentionSet(std::size_t capacity) noexcept
    : capacity_(capacity), valid_(capacity != 0u &&
                                  capacity <= kMaximumRetainedResources) {
  if (!valid_) capacity_ = 0u;
  for (auto& entry : entries_) entry.clear();
}

RetentionSet::~RetentionSet() noexcept { reset(); }

RetentionSet::RetentionSet(RetentionSet&& other) noexcept {
  for (auto& entry : entries_) entry.clear();
  moveFrom(std::move(other));
}

RetentionSet& RetentionSet::operator=(RetentionSet&& other) noexcept {
  if (this == &other) return *this;
  reset();
  moveFrom(std::move(other));
  return *this;
}

bool RetentionSet::validKey(const ResourceKey& key) noexcept {
  return knownKind(key.kind) && key.resourceId != 0u &&
         key.ownerCompositorId != 0u;
}

std::size_t RetentionSet::find(const ResourceKey& key) const noexcept {
  for (std::size_t index = 0u; index < capacity_; ++index) {
    if (entries_[index].occupied && entries_[index].key == key) return index;
  }
  return kInvalidIndex;
}

std::size_t RetentionSet::findEmpty() const noexcept {
  for (std::size_t index = 0u; index < capacity_; ++index) {
    if (!entries_[index].occupied) return index;
  }
  return kInvalidIndex;
}

Status RetentionSet::retain(const ResourceKey& key,
                            const std::shared_ptr<void>& resource) noexcept {
  if (!valid_) return Status::InvalidCapacity;
  if (!validKey(key)) return Status::InvalidKey;
  if (!resource) return Status::NullResource;

  const std::size_t existing = find(key);
  if (existing != kInvalidIndex) {
    if (entries_[existing].resource.get() == resource.get()) {
      return Status::Duplicate;
    }
    return Status::KeyConflict;
  }
  if (sealed_) return Status::Sealed;
  const std::size_t empty = findEmpty();
  if (empty == kInvalidIndex) return Status::CapacityExhausted;

  // shared_ptr assignment is noexcept here and the inline array is already
  // constructed, so this path cannot grow a heap-backed container.
  entries_[empty].key = key;
  entries_[empty].resource = resource;
  entries_[empty].occupied = true;
  ++count_;
  return Status::Retained;
}

Status RetentionSet::release(const ResourceKey& key) noexcept {
  const std::size_t index = find(key);
  if (index == kInvalidIndex) return Status::NotFound;
  entries_[index].clear();
  if (count_ != 0u) --count_;
  return Status::Retained;
}

void RetentionSet::reset() noexcept {
  for (auto& entry : entries_) entry.clear();
  count_ = 0u;
  sealed_ = false;
}

void RetentionSet::moveFrom(RetentionSet&& other) noexcept {
  capacity_ = other.capacity_;
  count_ = other.count_;
  sealed_ = other.sealed_;
  valid_ = other.valid_;
  for (std::size_t index = 0u; index < kMaximumRetainedResources; ++index) {
    entries_[index].key = other.entries_[index].key;
    entries_[index].resource = std::move(other.entries_[index].resource);
    entries_[index].occupied = other.entries_[index].occupied;
    other.entries_[index].clear();
  }
  other.count_ = 0u;
  other.sealed_ = false;
}

Snapshot RetentionSet::snapshot() const noexcept {
  Snapshot result{};
  result.count = count_;
  result.capacity = capacity_;
  result.sealed = sealed_;
  result.valid = valid_;
  for (std::size_t index = 0u; index < kMaximumRetainedResources; ++index) {
    result.entries[index].key = entries_[index].key;
    result.entries[index].pointer = entries_[index].resource.get();
    result.entries[index].occupied = entries_[index].occupied;
  }
  return result;
}

}  // namespace ChromaspaceMetalSubmissionRetention
