#include "ChromaspaceMetalMemoryPressureMailbox.h"

#include <cassert>
#include <thread>
#include <vector>

namespace {

using ChromaspaceMetalMemoryPressure::Batch;
using ChromaspaceMetalMemoryPressure::Mailbox;
using ChromaspaceMetalMemoryPressure::Signal;

void testRejectsInvalidSignalsWithoutMutation() {
  Mailbox mailbox;
  assert(!mailbox.publish(Signal::None));
  assert(!mailbox.publish(Signal::Count));
  assert(!mailbox.publish(static_cast<Signal>(255u)));
  assert(mailbox.empty());
  assert(mailbox.consume().empty());
}

void testCoalescesWithoutDowngrading() {
  Mailbox mailbox;
  assert(mailbox.publish(Signal::Warning));
  assert(mailbox.publish(Signal::Normal));
  assert(mailbox.publish(Signal::Critical));
  assert(mailbox.publish(Signal::Warning));
  assert(mailbox.publish(Signal::Normal));
  const Batch batch = mailbox.consume();
  assert(batch.strongest == Signal::Critical);
  assert(batch.normalCount == 2u);
  assert(batch.warningCount == 2u);
  assert(batch.criticalCount == 1u);
  assert(batch.eventCount() == 5u);
  assert(mailbox.empty());
}

void testConsumeDefinesTheNextBatchBoundary() {
  Mailbox mailbox;
  assert(mailbox.publish(Signal::Critical));
  const Batch first = mailbox.consume();
  assert(first.strongest == Signal::Critical);
  assert(first.criticalCount == 1u);
  assert(mailbox.publish(Signal::Normal));
  const Batch second = mailbox.consume();
  assert(second.strongest == Signal::Normal);
  assert(second.normalCount == 1u);
  assert(second.warningCount == 0u);
  assert(second.criticalCount == 0u);
}

void testConcurrentPublishersPreserveExactCounts() {
  Mailbox mailbox;
  constexpr uint32_t kEventsPerThread = 1000u;
  std::vector<std::thread> threads;
  threads.emplace_back([&] {
    for (uint32_t index = 0u; index < kEventsPerThread; ++index) {
      assert(mailbox.publish(Signal::Normal));
    }
  });
  threads.emplace_back([&] {
    for (uint32_t index = 0u; index < kEventsPerThread; ++index) {
      assert(mailbox.publish(Signal::Warning));
    }
  });
  threads.emplace_back([&] {
    for (uint32_t index = 0u; index < kEventsPerThread; ++index) {
      assert(mailbox.publish(Signal::Critical));
    }
  });
  for (auto& thread : threads) thread.join();

  const Batch batch = mailbox.consume();
  assert(batch.strongest == Signal::Critical);
  assert(batch.normalCount == kEventsPerThread);
  assert(batch.warningCount == kEventsPerThread);
  assert(batch.criticalCount == kEventsPerThread);
  assert(batch.eventCount() == 3u * kEventsPerThread);
}

}  // namespace

int main() {
  testRejectsInvalidSignalsWithoutMutation();
  testCoalescesWithoutDowngrading();
  testConsumeDefinesTheNextBatchBoundary();
  testConcurrentPublishersPreserveExactCounts();
  return 0;
}
