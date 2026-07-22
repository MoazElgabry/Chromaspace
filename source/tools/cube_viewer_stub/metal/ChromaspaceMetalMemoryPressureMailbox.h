#pragma once

#include <atomic>
#include <cstdint>

// Thread-safe boundary between platform memory-pressure notifications and the
// render-thread-affine Metal runtime. Producers only publish bounded scalar
// state; the viewer thread consumes one coherent batch and performs all GPU
// lifetime work itself.
namespace ChromaspaceMetalMemoryPressure {

enum class Signal : uint8_t {
  None = 0,
  Normal,
  Warning,
  Critical,
  Count,
};

const char* signalLabel(Signal signal) noexcept;

struct Batch final {
  Signal strongest = Signal::None;
  uint32_t normalCount = 0u;
  uint32_t warningCount = 0u;
  uint32_t criticalCount = 0u;

  bool empty() const noexcept { return strongest == Signal::None; }
  uint64_t eventCount() const noexcept {
    return static_cast<uint64_t>(normalCount) +
           static_cast<uint64_t>(warningCount) +
           static_cast<uint64_t>(criticalCount);
  }
};

class Mailbox final {
 public:
  Mailbox() noexcept = default;
  Mailbox(const Mailbox&) = delete;
  Mailbox& operator=(const Mailbox&) = delete;

  // Valid events are coalesced atomically. Counts saturate at a finite packed
  // maximum instead of wrapping; the strongest pending signal never
  // downgrades until consume() takes the batch.
  bool publish(Signal signal) noexcept;

  // One atomic exchange provides a coherent strongest level and per-level
  // counts. An event racing after the exchange belongs to the next batch.
  Batch consume() noexcept;

  bool empty() const noexcept {
    return packed_.load(std::memory_order_acquire) == 0u;
  }

 private:
  std::atomic<uint64_t> packed_{0u};
};

}  // namespace ChromaspaceMetalMemoryPressure
