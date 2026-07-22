#include "ChromaspaceMetalTransientArena.h"

#include <algorithm>
#include <limits>

namespace ChromaspaceMetalTransientArena {
namespace {

constexpr std::uint64_t kMaximumUint64 =
    (std::numeric_limits<std::uint64_t>::max)();

}  // namespace

const char* statusLabel(Status status) noexcept {
  switch (status) {
    case Status::Ok: return "ok";
    case Status::InvalidConfig: return "invalid-config";
    case Status::InvalidSubmissionId: return "invalid-submission-id";
    case Status::SubmissionAlreadyActive:
      return "submission-already-active";
    case Status::SubmissionNotFound: return "submission-not-found";
    case Status::WrongState: return "wrong-state";
    case Status::SubmissionLimitExceeded:
      return "submission-limit-exceeded";
    case Status::InFlightCapacityExceeded:
      return "in-flight-capacity-exceeded";
    case Status::SubmissionCapacityExceeded:
      return "submission-capacity-exceeded";
    case Status::ArithmeticOverflow: return "arithmetic-overflow";
    case Status::PageLimitExceeded: return "page-limit-exceeded";
    case Status::LogicalBytesExceedReserved:
      return "logical-bytes-exceed-reserved";
    case Status::CancelMismatch: return "cancel-mismatch";
  }
  return "unknown";
}

const char* stateLabel(State state) noexcept {
  switch (state) {
    case State::Empty: return "empty";
    case State::Encoding: return "encoding";
    case State::Submitted: return "submitted";
  }
  return "empty";
}

TransientArena::TransientArena(const Config& config) noexcept
    : config_(config),
      configStatus_(validateConfig(config) ? Status::Ok
                                            : Status::InvalidConfig) {
  for (auto& entry : entries_) entry.clear();
}

bool TransientArena::validateConfig(const Config& config) noexcept {
  return config.maxInFlightBytes != 0u &&
         config.maxBytesPerSubmission != 0u &&
         config.maxBytesPerSubmission <= config.maxInFlightBytes &&
         config.maxSubmissions >= 1u &&
         config.maxSubmissions <=
             static_cast<std::uint32_t>(kMaximumSubmissions);
}

bool TransientArena::addExact(std::uint64_t left,
                              std::uint64_t right,
                              std::uint64_t* result) noexcept {
  if (result == nullptr || right > kMaximumUint64 - left) return false;
  *result = left + right;
  return true;
}

std::size_t TransientArena::findSubmission(
    std::uint64_t submissionId) const noexcept {
  if (submissionId == kInvalidSubmissionId) return kInvalidIndex;
  for (std::size_t index = 0u; index < entries_.size(); ++index) {
    if (entries_[index].state != State::Empty &&
        entries_[index].id == submissionId) {
      return index;
    }
  }
  return kInvalidIndex;
}

std::size_t TransientArena::findEmptySlot() const noexcept {
  for (std::size_t index = 0u; index < entries_.size(); ++index) {
    if (entries_[index].state == State::Empty) return index;
  }
  return kInvalidIndex;
}

void TransientArena::retire(std::size_t index) noexcept {
  if (index >= entries_.size()) return;
  Entry& entry = entries_[index];

  // The arithmetic guards make underflow impossible for valid transitions.
  // Keep the defensive branches so a corrupted native caller cannot turn a
  // bookkeeping failure into a wrapped global counter.
  if (entry.reservedBytes <= inFlightReservedBytes_) {
    inFlightReservedBytes_ -= entry.reservedBytes;
  } else {
    inFlightReservedBytes_ = 0u;
  }
  if (entry.logicalBytes <= inFlightLogicalBytes_) {
    inFlightLogicalBytes_ -= entry.logicalBytes;
  } else {
    inFlightLogicalBytes_ = 0u;
  }

  if (entry.state == State::Encoding) {
    if (encodingCount_ != 0u) --encodingCount_;
  } else if (entry.state == State::Submitted) {
    if (submittedCount_ != 0u) --submittedCount_;
  }
  entry.clear();
}

Status TransientArena::begin(std::uint64_t submissionId) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  if (findSubmission(submissionId) != kInvalidIndex) {
    return Status::SubmissionAlreadyActive;
  }
  if (activeSubmissionCount() >=
      static_cast<std::size_t>(config_.maxSubmissions)) {
    return Status::SubmissionLimitExceeded;
  }

  const std::size_t slot = findEmptySlot();
  if (slot == kInvalidIndex) {
    // This should only be reachable if internal limits were changed without
    // updating Config validation.  Treat it as a bounded-capacity failure.
    return Status::SubmissionLimitExceeded;
  }
  Entry& entry = entries_[slot];
  entry.clear();
  entry.id = submissionId;
  entry.state = State::Encoding;
  ++encodingCount_;
  peakActiveSubmissionCount_ =
      (std::max)(peakActiveSubmissionCount_, activeSubmissionCount());
  return Status::Ok;
}

Status TransientArena::reservePage(std::uint64_t submissionId,
                                   std::uint64_t pageBytes) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  Entry& entry = entries_[index];
  if (entry.state != State::Encoding) return Status::WrongState;
  if (entry.pageCount >= kMaximumPageReservationsPerSubmission) {
    return Status::PageLimitExceeded;
  }

  std::uint64_t nextReserved = 0u;
  if (!addExact(entry.reservedBytes, pageBytes, &nextReserved)) {
    return Status::ArithmeticOverflow;
  }
  std::uint64_t nextInFlight = 0u;
  if (!addExact(inFlightReservedBytes_, pageBytes, &nextInFlight)) {
    return Status::ArithmeticOverflow;
  }
  if (nextInFlight > config_.maxInFlightBytes) {
    return Status::InFlightCapacityExceeded;
  }
  if (nextReserved > config_.maxBytesPerSubmission) {
    return Status::SubmissionCapacityExceeded;
  }

  entry.pageBytes[entry.pageCount] = pageBytes;
  ++entry.pageCount;
  entry.reservedBytes = nextReserved;
  inFlightReservedBytes_ = nextInFlight;
  peakInFlightReservedBytes_ =
      (std::max)(peakInFlightReservedBytes_, inFlightReservedBytes_);
  return Status::Ok;
}

Status TransientArena::cancelLastPage(std::uint64_t submissionId,
                                      std::uint64_t pageBytes) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  Entry& entry = entries_[index];
  if (entry.state != State::Encoding) return Status::WrongState;
  if (entry.pageCount == 0u ||
      entry.pageBytes[entry.pageCount - 1u] != pageBytes) {
    return Status::CancelMismatch;
  }

  const std::uint64_t lastPage = entry.pageBytes[entry.pageCount - 1u];
  // A buffer recorded against the heap cannot be made to outlive its page
  // capacity.  Refuse the rollback rather than silently corrupting totals.
  if (lastPage > entry.reservedBytes ||
      lastPage > inFlightReservedBytes_ ||
      entry.logicalBytes > entry.reservedBytes - lastPage) {
    return Status::CancelMismatch;
  }

  entry.reservedBytes -= lastPage;
  inFlightReservedBytes_ -= lastPage;
  entry.pageBytes[entry.pageCount - 1u] = 0u;
  --entry.pageCount;
  return Status::Ok;
}

Status TransientArena::recordBuffer(std::uint64_t submissionId,
                                    std::uint64_t logicalBytes) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  Entry& entry = entries_[index];
  if (entry.state != State::Encoding) return Status::WrongState;

  std::uint64_t nextLogical = 0u;
  if (!addExact(entry.logicalBytes, logicalBytes, &nextLogical)) {
    return Status::ArithmeticOverflow;
  }
  if (nextLogical > entry.reservedBytes) {
    return Status::LogicalBytesExceedReserved;
  }
  std::uint64_t nextInFlightLogical = 0u;
  if (!addExact(inFlightLogicalBytes_, logicalBytes,
                &nextInFlightLogical)) {
    return Status::ArithmeticOverflow;
  }
  if (entry.bufferCount == (std::numeric_limits<std::size_t>::max)()) {
    return Status::ArithmeticOverflow;
  }

  entry.logicalBytes = nextLogical;
  inFlightLogicalBytes_ = nextInFlightLogical;
  peakInFlightLogicalBytes_ =
      (std::max)(peakInFlightLogicalBytes_, inFlightLogicalBytes_);
  ++entry.bufferCount;
  return Status::Ok;
}

Status TransientArena::submit(std::uint64_t submissionId) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  Entry& entry = entries_[index];
  if (entry.state != State::Encoding) return Status::WrongState;
  if (encodingCount_ == 0u) return Status::WrongState;

  entry.state = State::Submitted;
  --encodingCount_;
  ++submittedCount_;
  return Status::Ok;
}

Status TransientArena::complete(std::uint64_t submissionId) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  if (entries_[index].state != State::Submitted) return Status::WrongState;
  retire(index);
  return Status::Ok;
}

Status TransientArena::abandon(std::uint64_t submissionId) noexcept {
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  const State state = entries_[index].state;
  if (state != State::Encoding && state != State::Submitted) {
    return Status::WrongState;
  }
  retire(index);
  return Status::Ok;
}

Status TransientArena::reset() noexcept {
  for (auto& entry : entries_) entry.clear();
  inFlightReservedBytes_ = 0u;
  inFlightLogicalBytes_ = 0u;
  peakInFlightReservedBytes_ = 0u;
  peakInFlightLogicalBytes_ = 0u;
  encodingCount_ = 0u;
  submittedCount_ = 0u;
  peakActiveSubmissionCount_ = 0u;
  return Status::Ok;
}

bool TransientArena::hasSubmission(std::uint64_t submissionId) const noexcept {
  return findSubmission(submissionId) != kInvalidIndex;
}

ArenaSnapshot TransientArena::snapshot() const noexcept {
  ArenaSnapshot result{};
  result.config = config_;
  result.configStatus = configStatus_;
  result.inFlightReservedBytes = inFlightReservedBytes_;
  result.inFlightLogicalBytes = inFlightLogicalBytes_;
  result.peakInFlightReservedBytes = peakInFlightReservedBytes_;
  result.peakInFlightLogicalBytes = peakInFlightLogicalBytes_;
  result.encodingCount = encodingCount_;
  result.submittedCount = submittedCount_;
  result.activeCount = activeSubmissionCount();
  result.peakActiveSubmissionCount = peakActiveSubmissionCount_;
  for (std::size_t index = 0u; index < entries_.size(); ++index) {
    result.submissions[index] = entries_[index].snapshot();
  }
  return result;
}

SubmissionSnapshot TransientArena::snapshot(
    std::uint64_t submissionId) const noexcept {
  SubmissionSnapshot result{};
  if (submissionId == kInvalidSubmissionId) return result;
  const std::size_t index = findSubmission(submissionId);
  if (index != kInvalidIndex) result = entries_[index].snapshot();
  return result;
}

Status TransientArena::snapshot(std::uint64_t submissionId,
                                SubmissionSnapshot* out) const noexcept {
  if (out == nullptr) return Status::InvalidSubmissionId;
  *out = SubmissionSnapshot{};
  if (!configValid()) return Status::InvalidConfig;
  if (submissionId == kInvalidSubmissionId) {
    return Status::InvalidSubmissionId;
  }
  const std::size_t index = findSubmission(submissionId);
  if (index == kInvalidIndex) return Status::SubmissionNotFound;
  *out = entries_[index].snapshot();
  return Status::Ok;
}

}  // namespace ChromaspaceMetalTransientArena
