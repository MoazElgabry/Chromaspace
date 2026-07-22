#include "ChromaspaceMetalMemoryPressureMailbox.h"

#include <algorithm>

namespace ChromaspaceMetalMemoryPressure {
namespace {

constexpr uint64_t kSeverityBits = 2u;
constexpr uint64_t kCountBits = 20u;
constexpr uint64_t kSeverityMask = (1u << kSeverityBits) - 1u;
constexpr uint64_t kCountMask = (1u << kCountBits) - 1u;
constexpr uint64_t kNormalShift = kSeverityBits;
constexpr uint64_t kWarningShift = kNormalShift + kCountBits;
constexpr uint64_t kCriticalShift = kWarningShift + kCountBits;

static_assert(kCriticalShift + kCountBits <= 64u,
              "memory-pressure mailbox must fit one atomic word");

bool validSignal(Signal signal) noexcept {
  return signal == Signal::Normal || signal == Signal::Warning ||
         signal == Signal::Critical;
}

uint64_t countAt(uint64_t packed, uint64_t shift) noexcept {
  return (packed >> shift) & kCountMask;
}

uint64_t withIncrementedCount(uint64_t packed, uint64_t shift) noexcept {
  const uint64_t count = countAt(packed, shift);
  if (count == kCountMask) return packed;
  return packed + (uint64_t{1u} << shift);
}

uint64_t shiftForSignal(Signal signal) noexcept {
  switch (signal) {
    case Signal::Normal: return kNormalShift;
    case Signal::Warning: return kWarningShift;
    case Signal::Critical: return kCriticalShift;
    case Signal::None:
    case Signal::Count: break;
  }
  return 0u;
}

}  // namespace

const char* signalLabel(Signal signal) noexcept {
  switch (signal) {
    case Signal::None: return "none";
    case Signal::Normal: return "normal";
    case Signal::Warning: return "warning";
    case Signal::Critical: return "critical";
    case Signal::Count: break;
  }
  return "invalid";
}

bool Mailbox::publish(Signal signal) noexcept {
  if (!validSignal(signal)) return false;
  const uint64_t signalValue = static_cast<uint64_t>(signal);
  const uint64_t countShift = shiftForSignal(signal);
  uint64_t observed = packed_.load(std::memory_order_relaxed);
  for (;;) {
    const uint64_t strongest = observed & kSeverityMask;
    uint64_t desired = observed & ~kSeverityMask;
    desired |= std::max(strongest, signalValue);
    desired = withIncrementedCount(desired, countShift);
    if (packed_.compare_exchange_weak(observed, desired,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
      return true;
    }
  }
}

Batch Mailbox::consume() noexcept {
  const uint64_t packed = packed_.exchange(0u, std::memory_order_acq_rel);
  Batch result{};
  const uint64_t severity = packed & kSeverityMask;
  result.strongest = severity <= static_cast<uint64_t>(Signal::Critical)
                         ? static_cast<Signal>(severity)
                         : Signal::None;
  result.normalCount = static_cast<uint32_t>(countAt(packed, kNormalShift));
  result.warningCount = static_cast<uint32_t>(countAt(packed, kWarningShift));
  result.criticalCount =
      static_cast<uint32_t>(countAt(packed, kCriticalShift));
  return result;
}

}  // namespace ChromaspaceMetalMemoryPressure
