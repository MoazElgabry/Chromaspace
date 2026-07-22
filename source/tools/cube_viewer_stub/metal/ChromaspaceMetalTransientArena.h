#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

// Portable accounting for private Metal transient heap pages.  The native
// backend owns the actual MTLHeap objects; this layer owns only bounded
// submission lifetime and byte-capacity policy.  All callers must serialize
// access to one arena instance on their render/runtime coordination thread.
namespace ChromaspaceMetalTransientArena {

// The native integration is intentionally limited to three in-flight
// submissions.  A fixed page ledger makes rollback of a failed heap creation
// deterministic without allocating from the render path.
constexpr std::size_t kMaximumSubmissions = 3u;
constexpr std::size_t kMaximumPageReservationsPerSubmission = 128u;
constexpr std::size_t kMaxSubmissions = kMaximumSubmissions;
constexpr std::size_t kMaxPageReservationsPerSubmission =
    kMaximumPageReservationsPerSubmission;
constexpr std::uint64_t kInvalidSubmissionId = 0u;

struct Config {
  // Defaults mirror the existing transient residency scale while leaving
  // enough room for all three normal submissions.  Values are policy only;
  // no allocation is performed by this module.
  std::uint64_t maxInFlightBytes = 768ull * 1024ull * 1024ull;
  std::uint64_t maxBytesPerSubmission = 256ull * 1024ull * 1024ull;
  std::uint32_t maxSubmissions =
      static_cast<std::uint32_t>(kMaximumSubmissions);
};

enum class State : std::uint8_t {
  Empty = 0,
  Idle = Empty,
  Encoding,
  Submitted,
};

using SubmissionState = State;

enum class Status : std::uint8_t {
  Ok = 0,
  Success = Ok,

  InvalidConfig,
  InvalidSubmissionId,
  SubmissionAlreadyActive,
  SubmissionNotFound,
  WrongState,
  SubmissionLimitExceeded,
  InFlightCapacityExceeded,
  SubmissionCapacityExceeded,
  ArithmeticOverflow,
  PageLimitExceeded,
  LogicalBytesExceedReserved,
  CancelMismatch,

  // Readable aliases retained for callers that use the policy vocabulary
  // rather than the implementation's canonical names.
  InvalidId = InvalidSubmissionId,
  DuplicateSubmission = SubmissionAlreadyActive,
  MissingSubmission = SubmissionNotFound,
  InvalidState = WrongState,
  MaxSubmissionsExceeded = SubmissionLimitExceeded,
  TooManySubmissions = SubmissionLimitExceeded,
  GlobalCapacityExceeded = InFlightCapacityExceeded,
  InFlightBytesExceeded = InFlightCapacityExceeded,
  PerSubmissionCapacityExceeded = SubmissionCapacityExceeded,
  MaxBytesPerSubmissionExceeded = SubmissionCapacityExceeded,
  PageReservationLimitExceeded = PageLimitExceeded,
  LogicalOverReserved = LogicalBytesExceedReserved,
  CancellationMismatch = CancelMismatch,
};

const char* statusLabel(Status status) noexcept;
const char* stateLabel(State state) noexcept;

constexpr bool succeeded(Status status) noexcept {
  return status == Status::Ok;
}

struct SubmissionSnapshot {
  std::uint64_t id = kInvalidSubmissionId;
  State state = State::Empty;
  std::uint64_t reservedBytes = 0u;
  std::uint64_t logicalBytes = 0u;
  std::size_t pageCount = 0u;
  std::size_t bufferCount = 0u;
  bool active = false;

  constexpr bool occupied() const noexcept { return active; }
};

struct ArenaSnapshot {
  Config config{};
  Status configStatus = Status::InvalidConfig;
  std::uint64_t inFlightReservedBytes = 0u;
  std::uint64_t inFlightLogicalBytes = 0u;
  std::uint64_t peakInFlightReservedBytes = 0u;
  std::uint64_t peakInFlightLogicalBytes = 0u;
  std::size_t encodingCount = 0u;
  std::size_t submittedCount = 0u;
  std::size_t activeCount = 0u;
  std::size_t peakActiveSubmissionCount = 0u;
  std::array<SubmissionSnapshot, kMaximumSubmissions> submissions{};
};

class TransientArena final {
 public:
  explicit TransientArena(const Config& config = Config{}) noexcept;
  ~TransientArena() noexcept = default;

  TransientArena(const TransientArena&) = delete;
  TransientArena& operator=(const TransientArena&) = delete;

  static bool validateConfig(const Config& config) noexcept;

  const Config& config() const noexcept { return config_; }
  Status configStatus() const noexcept { return configStatus_; }
  bool configValid() const noexcept { return configStatus_ == Status::Ok; }

  // A non-zero opaque ID identifies one in-flight command-buffer submission.
  // UINT64_MAX is intentionally valid; this class never increments IDs.
  Status begin(std::uint64_t submissionId) noexcept;

  // Reserve the full capacity of one private heap page.  The page count is
  // bounded by kMaximumPageReservationsPerSubmission, and failed calls leave
  // every counter and ledger entry unchanged.
  Status reservePage(std::uint64_t submissionId,
                     std::uint64_t pageBytes) noexcept;

  // Roll back only the most recently successful reservation.  This is the
  // native heap-creation rollback hook; a mismatch or wrong-state call fails
  // closed and does not mutate the ledger.
  Status cancelLastPage(std::uint64_t submissionId,
                        std::uint64_t pageBytes) noexcept;

  // Record the logical bytes used by a successfully allocated buffer.  The
  // cumulative logical total may not exceed the submission's reserved page
  // capacity.
  Status recordBuffer(std::uint64_t submissionId,
                      std::uint64_t logicalBytes) noexcept;

  Status submit(std::uint64_t submissionId) noexcept;
  Status complete(std::uint64_t submissionId) noexcept;
  Status abandon(std::uint64_t submissionId) noexcept;

  // Reset is non-terminal and drains every Encoding and Submitted record.
  // It is intentionally safe to call repeatedly.
  Status reset() noexcept;

  std::uint64_t inFlightReservedBytes() const noexcept {
    return inFlightReservedBytes_;
  }
  std::uint64_t inFlightLogicalBytes() const noexcept {
    return inFlightLogicalBytes_;
  }
  std::uint64_t peakInFlightReservedBytes() const noexcept {
    return peakInFlightReservedBytes_;
  }
  std::uint64_t peakInFlightLogicalBytes() const noexcept {
    return peakInFlightLogicalBytes_;
  }
  std::size_t peakActiveSubmissionCount() const noexcept {
    return peakActiveSubmissionCount_;
  }
  std::size_t encodingCount() const noexcept { return encodingCount_; }
  std::size_t submittedCount() const noexcept { return submittedCount_; }
  std::size_t activeSubmissionCount() const noexcept {
    return encodingCount_ + submittedCount_;
  }
  std::size_t submissionCount() const noexcept {
    return activeSubmissionCount();
  }

  // A zero-argument snapshot captures all bounded counters and slots.  The
  // one-argument form returns an Empty snapshot for a missing ID; callers
  // that need an explicit status can use the pointer overload.
  ArenaSnapshot snapshot() const noexcept;
  SubmissionSnapshot snapshot(std::uint64_t submissionId) const noexcept;
  Status snapshot(std::uint64_t submissionId,
                  SubmissionSnapshot* out) const noexcept;
  Status getSnapshot(std::uint64_t submissionId,
                     SubmissionSnapshot* out) const noexcept {
    return snapshot(submissionId, out);
  }
  Status submissionSnapshot(std::uint64_t submissionId,
                            SubmissionSnapshot* out) const noexcept {
    return snapshot(submissionId, out);
  }

  bool hasSubmission(std::uint64_t submissionId) const noexcept;

 private:
  struct Entry {
    std::uint64_t id = kInvalidSubmissionId;
    State state = State::Empty;
    std::uint64_t reservedBytes = 0u;
    std::uint64_t logicalBytes = 0u;
    std::size_t pageCount = 0u;
    std::size_t bufferCount = 0u;
    std::array<std::uint64_t, kMaximumPageReservationsPerSubmission>
        pageBytes{};

    void clear() noexcept {
      id = kInvalidSubmissionId;
      state = State::Empty;
      reservedBytes = 0u;
      logicalBytes = 0u;
      pageCount = 0u;
      bufferCount = 0u;
      pageBytes.fill(0u);
    }

    SubmissionSnapshot snapshot() const noexcept {
      SubmissionSnapshot result{};
      result.id = id;
      result.state = state;
      result.reservedBytes = reservedBytes;
      result.logicalBytes = logicalBytes;
      result.pageCount = pageCount;
      result.bufferCount = bufferCount;
      result.active = state != State::Empty;
      return result;
    }
  };

  static constexpr std::size_t kInvalidIndex =
      static_cast<std::size_t>(-1);

  static bool addExact(std::uint64_t left,
                       std::uint64_t right,
                       std::uint64_t* result) noexcept;
  std::size_t findSubmission(std::uint64_t submissionId) const noexcept;
  std::size_t findEmptySlot() const noexcept;
  void retire(std::size_t index) noexcept;

  Config config_{};
  Status configStatus_ = Status::InvalidConfig;
  std::array<Entry, kMaximumSubmissions> entries_{};
  std::uint64_t inFlightReservedBytes_ = 0u;
  std::uint64_t inFlightLogicalBytes_ = 0u;
  std::uint64_t peakInFlightReservedBytes_ = 0u;
  std::uint64_t peakInFlightLogicalBytes_ = 0u;
  std::size_t encodingCount_ = 0u;
  std::size_t submittedCount_ = 0u;
  std::size_t peakActiveSubmissionCount_ = 0u;
};

using Arena = TransientArena;
using MetalTransientArena = TransientArena;

}  // namespace ChromaspaceMetalTransientArena
