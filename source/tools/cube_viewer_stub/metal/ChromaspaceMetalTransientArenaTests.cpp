#include "ChromaspaceMetalTransientArena.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using namespace ChromaspaceMetalTransientArena;

void expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::abort();
  }
}

void expectStatus(Status actual, Status expected, const char* message) {
  if (actual != expected) {
    std::cerr << "FAIL: " << message << " (actual=" << statusLabel(actual)
              << ", expected=" << statusLabel(expected) << ")\n";
    std::abort();
  }
}

Config config(std::uint64_t global,
              std::uint64_t perSubmission,
              std::uint32_t submissions = 3u) {
  Config result{};
  result.maxInFlightBytes = global;
  result.maxBytesPerSubmission = perSubmission;
  result.maxSubmissions = submissions;
  return result;
}

void configValidationAndIdRules() {
  expect(TransientArena::validateConfig(Config{}), "default config valid");

  Config zeroGlobal = config(0u, 0u);
  expect(!TransientArena::validateConfig(zeroGlobal),
         "zero global budget invalid");
  Config zeroPer = config(100u, 0u);
  expect(!TransientArena::validateConfig(zeroPer),
         "zero per-submission budget invalid");
  Config reversed = config(100u, 101u);
  expect(!TransientArena::validateConfig(reversed),
         "per-submission budget above global invalid");
  Config zeroSubmissions = config(100u, 100u, 0u);
  expect(!TransientArena::validateConfig(zeroSubmissions),
         "zero submission limit invalid");
  Config tooMany = config(100u, 100u, 4u);
  expect(!TransientArena::validateConfig(tooMany),
         "submission limit above fixed slots invalid");

  TransientArena invalid(zeroGlobal);
  expectStatus(invalid.configStatus(), Status::InvalidConfig,
               "invalid config status");
  expectStatus(invalid.begin(1u), Status::InvalidConfig,
               "invalid config rejects begin");

  TransientArena arena(config(100u, 100u, 3u));
  expectStatus(arena.begin(0u), Status::InvalidSubmissionId,
               "zero submission ID invalid");
  expectStatus(arena.begin((std::numeric_limits<std::uint64_t>::max)()),
               Status::Ok, "UINT64_MAX submission ID valid");
  expectStatus(arena.begin((std::numeric_limits<std::uint64_t>::max)()),
               Status::SubmissionAlreadyActive, "duplicate ID rejected");
  expect(arena.snapshot((std::numeric_limits<std::uint64_t>::max)()).active,
         "maximum ID is present");
  expectStatus(arena.abandon((std::numeric_limits<std::uint64_t>::max)()),
               Status::Ok, "maximum ID can retire");
  expectStatus(arena.abandon((std::numeric_limits<std::uint64_t>::max)()),
               Status::SubmissionNotFound, "retired ID is missing");
}

void concurrentSubmissionsAndLimits() {
  TransientArena arena(config(100u, 60u, 3u));
  expectStatus(arena.begin(1u), Status::Ok, "first concurrent begin");
  expectStatus(arena.begin(2u), Status::Ok, "second concurrent begin");
  expectStatus(arena.begin(3u), Status::Ok, "third concurrent begin");
  expect(arena.encodingCount() == 3u && arena.submittedCount() == 0u &&
             arena.activeSubmissionCount() == 3u,
         "three encoding submissions counted");
  expectStatus(arena.begin(4u), Status::SubmissionLimitExceeded,
               "fourth submission rejected");
  expect(arena.activeSubmissionCount() == 3u &&
             arena.inFlightReservedBytes() == 0u,
         "submission-limit rejection is transactional");

  expectStatus(arena.reservePage(1u, 40u), Status::Ok,
               "first page reservation");
  expectStatus(arena.reservePage(2u, 30u), Status::Ok,
               "second page reservation");
  expectStatus(arena.reservePage(3u, 20u), Status::Ok,
               "third page reservation");
  expect(arena.inFlightReservedBytes() == 90u,
         "global reserved bytes sum page capacities");
  expect(arena.peakActiveSubmissionCount() == 3u &&
             arena.peakInFlightReservedBytes() == 90u,
         "successful mutations publish active and reserved peaks");
  const ArenaSnapshot beforeGlobal = arena.snapshot();
  expectStatus(arena.reservePage(3u, 11u), Status::InFlightCapacityExceeded,
               "global capacity rejection");
  const ArenaSnapshot afterGlobal = arena.snapshot();
  expect(afterGlobal.inFlightReservedBytes ==
             beforeGlobal.inFlightReservedBytes &&
             afterGlobal.submissions[2u].reservedBytes ==
                 beforeGlobal.submissions[2u].reservedBytes &&
             afterGlobal.submissions[2u].pageCount ==
                 beforeGlobal.submissions[2u].pageCount,
         "global capacity failure leaves snapshot unchanged");
  expect(afterGlobal.peakInFlightReservedBytes ==
             beforeGlobal.peakInFlightReservedBytes &&
             afterGlobal.peakActiveSubmissionCount ==
                 beforeGlobal.peakActiveSubmissionCount,
         "capacity rejection leaves peaks unchanged");

  TransientArena per(config(200u, 60u, 1u));
  expectStatus(per.begin(10u), Status::Ok, "per-submission begin");
  expectStatus(per.reservePage(10u, 60u), Status::Ok,
               "per-submission capacity exact boundary");
  const ArenaSnapshot beforePer = per.snapshot();
  expectStatus(per.reservePage(10u, 1u), Status::SubmissionCapacityExceeded,
               "per-submission capacity rejection");
  const ArenaSnapshot afterPer = per.snapshot();
  expect(afterPer.inFlightReservedBytes == beforePer.inFlightReservedBytes &&
             afterPer.submissions[0u].reservedBytes ==
                 beforePer.submissions[0u].reservedBytes &&
             afterPer.submissions[0u].pageCount ==
                 beforePer.submissions[0u].pageCount,
         "per-submission capacity failure leaves snapshot unchanged");
}

void reserveCancelRollback() {
  TransientArena arena(config(100u, 100u, 1u));
  expectStatus(arena.begin(20u), Status::Ok, "rollback begin");
  expectStatus(arena.reservePage(20u, 10u), Status::Ok,
               "rollback first page");
  expectStatus(arena.reservePage(20u, 20u), Status::Ok,
               "rollback second page");
  expect(arena.inFlightReservedBytes() == 30u,
         "rollback reservations counted");

  const SubmissionSnapshot beforeMismatch = arena.snapshot(20u);
  expectStatus(arena.cancelLastPage(20u, 10u), Status::CancelMismatch,
               "only most recent page can be cancelled");
  const SubmissionSnapshot afterMismatch = arena.snapshot(20u);
  expect(afterMismatch.reservedBytes == beforeMismatch.reservedBytes &&
             afterMismatch.pageCount == beforeMismatch.pageCount &&
             arena.inFlightReservedBytes() == 30u,
         "cancel mismatch leaves state unchanged");
  expectStatus(arena.cancelLastPage(20u, 20u), Status::Ok,
               "most recent page cancelled");
  expect(arena.inFlightReservedBytes() == 10u &&
             arena.snapshot(20u).reservedBytes == 10u &&
             arena.snapshot(20u).pageCount == 1u,
         "cancel rolls back page and global totals");
  expectStatus(arena.cancelLastPage(20u, 20u), Status::CancelMismatch,
               "cancelled page cannot be cancelled twice");

  expectStatus(arena.recordBuffer(20u, 10u), Status::Ok,
               "logical bytes fit remaining page");
  const SubmissionSnapshot beforeLogicalCancel = arena.snapshot(20u);
  expectStatus(arena.cancelLastPage(20u, 10u), Status::CancelMismatch,
               "page with live logical bytes cannot be rolled back");
  const SubmissionSnapshot afterLogicalCancel = arena.snapshot(20u);
  expect(afterLogicalCancel.reservedBytes ==
             beforeLogicalCancel.reservedBytes &&
             afterLogicalCancel.logicalBytes ==
                 beforeLogicalCancel.logicalBytes,
         "unsafe cancel leaves state unchanged");
  expectStatus(arena.abandon(20u), Status::Ok, "rollback abandon");
}

void logicalTrackingAndWrongStates() {
  TransientArena arena(config(100u, 100u, 1u));
  expectStatus(arena.reservePage(30u, 1u), Status::SubmissionNotFound,
               "reserve missing submission");
  expectStatus(arena.begin(30u), Status::Ok, "logical begin");
  expectStatus(arena.recordBuffer(30u, 1u), Status::LogicalBytesExceedReserved,
               "logical bytes require reserved capacity");
  expect(arena.inFlightLogicalBytes() == 0u,
         "failed logical record leaves global total unchanged");
  expectStatus(arena.reservePage(30u, 40u), Status::Ok,
               "logical page reserve");
  expectStatus(arena.recordBuffer(30u, 15u), Status::Ok,
               "logical buffer record");
  expectStatus(arena.recordBuffer(30u, 30u), Status::LogicalBytesExceedReserved,
               "cumulative logical bytes bounded by reservation");
  expect(arena.inFlightLogicalBytes() == 15u &&
             arena.snapshot(30u).logicalBytes == 15u &&
             arena.snapshot(30u).bufferCount == 1u,
         "logical counters track successful records only");
  expect(arena.peakInFlightLogicalBytes() == 15u,
         "successful logical records publish a peak");
  expectStatus(arena.complete(30u), Status::WrongState,
               "complete requires submitted state");
  expectStatus(arena.submit(30u), Status::Ok, "submit transitions state");
  expect(arena.encodingCount() == 0u && arena.submittedCount() == 1u &&
             arena.snapshot(30u).state == State::Submitted,
         "submit updates state counters");
  expectStatus(arena.recordBuffer(30u, 0u), Status::WrongState,
               "record after submit rejected");
  expectStatus(arena.reservePage(30u, 0u), Status::WrongState,
               "reserve after submit rejected");
  expectStatus(arena.cancelLastPage(30u, 40u), Status::WrongState,
               "cancel after submit rejected");
  expectStatus(arena.submit(30u), Status::WrongState,
               "duplicate submit rejected");
  expectStatus(arena.complete(30u), Status::Ok, "submitted completion");
  expect(arena.inFlightReservedBytes() == 0u &&
             arena.inFlightLogicalBytes() == 0u &&
             arena.activeSubmissionCount() == 0u,
         "completion drains counters");
  expect(arena.peakInFlightReservedBytes() == 40u &&
             arena.peakInFlightLogicalBytes() == 15u &&
             arena.peakActiveSubmissionCount() == 1u,
         "completion preserves high-water marks");
  expectStatus(arena.complete(30u), Status::SubmissionNotFound,
               "duplicate completion is missing");
}

void abandonResetAndReuse() {
  TransientArena arena(config(100u, 100u, 3u));
  expectStatus(arena.begin(40u), Status::Ok, "abandon encoding begin");
  expectStatus(arena.reservePage(40u, 10u), Status::Ok,
               "abandon encoding reserve");
  expectStatus(arena.abandon(40u), Status::Ok, "abandon encoding");
  expectStatus(arena.abandon(40u), Status::SubmissionNotFound,
               "abandon encoding exactly once");

  expectStatus(arena.begin(41u), Status::Ok, "abandon submitted begin");
  expectStatus(arena.reservePage(41u, 20u), Status::Ok,
               "abandon submitted reserve");
  expectStatus(arena.submit(41u), Status::Ok, "abandon submitted transition");
  expectStatus(arena.abandon(41u), Status::Ok, "abandon submitted");
  expectStatus(arena.abandon(41u), Status::SubmissionNotFound,
               "abandon submitted exactly once");

  expectStatus(arena.begin(42u), Status::Ok, "reset encoding begin");
  expectStatus(arena.reservePage(42u, 30u), Status::Ok,
               "reset encoding reserve");
  expectStatus(arena.begin(43u), Status::Ok, "reset second begin");
  expectStatus(arena.reservePage(43u, 40u), Status::Ok,
               "reset second reserve");
  expectStatus(arena.submit(43u), Status::Ok, "reset submitted transition");
  expect(arena.activeSubmissionCount() == 2u &&
             arena.inFlightReservedBytes() == 70u,
         "reset has both states in flight");
  expectStatus(arena.reset(), Status::Ok, "reset drains all submissions");
  expect(arena.activeSubmissionCount() == 0u &&
             arena.inFlightReservedBytes() == 0u &&
             arena.inFlightLogicalBytes() == 0u &&
             arena.peakInFlightReservedBytes() == 0u &&
             arena.peakInFlightLogicalBytes() == 0u &&
             arena.peakActiveSubmissionCount() == 0u &&
             !arena.hasSubmission(42u) && !arena.hasSubmission(43u),
         "reset clears counters, peaks, and slots");
  expectStatus(arena.reset(), Status::Ok, "reset is repeatable");
  expectStatus(arena.begin(42u), Status::Ok,
               "ID reusable only after retirement");
  expect(arena.peakActiveSubmissionCount() == 1u,
         "reuse establishes a new post-reset peak");
  expectStatus(arena.abandon(42u), Status::Ok, "reuse cleanup");
}

void pageLimitAndArithmeticBoundary() {
  TransientArena pageLimited(
      config(static_cast<std::uint64_t>(kMaximumPageReservationsPerSubmission),
             static_cast<std::uint64_t>(kMaximumPageReservationsPerSubmission),
             1u));
  expectStatus(pageLimited.begin(50u), Status::Ok, "page limit begin");
  for (std::size_t index = 0u;
       index < kMaximumPageReservationsPerSubmission; ++index) {
    expectStatus(pageLimited.reservePage(50u, 1u), Status::Ok,
                 "page limit reservation accepted");
  }
  const SubmissionSnapshot beforeLimit = pageLimited.snapshot(50u);
  expectStatus(pageLimited.reservePage(50u, 1u), Status::PageLimitExceeded,
               "page limit rejection");
  const SubmissionSnapshot afterLimit = pageLimited.snapshot(50u);
  expect(afterLimit.pageCount == beforeLimit.pageCount &&
             afterLimit.reservedBytes == beforeLimit.reservedBytes &&
             pageLimited.inFlightReservedBytes() == beforeLimit.reservedBytes,
         "page limit rejection leaves state unchanged");
  expectStatus(pageLimited.abandon(50u), Status::Ok, "page limit cleanup");

  const std::uint64_t maximum =
      (std::numeric_limits<std::uint64_t>::max)();
  TransientArena arithmetic(config(maximum, maximum, 1u));
  expectStatus(arithmetic.begin(maximum), Status::Ok,
               "arithmetic maximum ID begin");
  expectStatus(arithmetic.reservePage(maximum, maximum), Status::Ok,
               "maximum page reservation accepted");
  const ArenaSnapshot beforeOverflow = arithmetic.snapshot();
  expectStatus(arithmetic.reservePage(maximum, 1u), Status::ArithmeticOverflow,
               "page addition overflow rejected");
  const ArenaSnapshot afterOverflow = arithmetic.snapshot();
  expect(afterOverflow.inFlightReservedBytes ==
             beforeOverflow.inFlightReservedBytes &&
             afterOverflow.submissions[0u].pageCount ==
                 beforeOverflow.submissions[0u].pageCount,
         "overflow failure leaves state unchanged");
  expectStatus(arithmetic.recordBuffer(maximum, maximum), Status::Ok,
               "maximum logical record accepted");
  expectStatus(arithmetic.recordBuffer(maximum, 1u),
               Status::ArithmeticOverflow,
               "logical addition overflow rejected");
  expect(arithmetic.inFlightLogicalBytes() == maximum,
         "logical overflow leaves global total unchanged");
  expectStatus(arithmetic.cancelLastPage(maximum, maximum),
               Status::CancelMismatch,
               "cancel cannot discard reserved bytes behind logical record");
  expectStatus(arithmetic.abandon(maximum), Status::Ok,
               "arithmetic cleanup");
  expect(arithmetic.inFlightReservedBytes() == 0u &&
             arithmetic.inFlightLogicalBytes() == 0u,
         "arithmetic cleanup drains maximum counters");
}

}  // namespace

int main() {
  configValidationAndIdRules();
  concurrentSubmissionsAndLimits();
  reserveCancelRollback();
  logicalTrackingAndWrongStates();
  abandonResetAndReuse();
  pageLimitAndArithmeticBoundary();
  std::cout << "ChromaspaceMetalTransientArenaTests passed\n";
  return 0;
}
