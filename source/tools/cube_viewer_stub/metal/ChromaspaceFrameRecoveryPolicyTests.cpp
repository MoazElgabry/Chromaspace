#include "ChromaspaceFrameRecoveryPolicy.h"

#include <cassert>
#include <cstdint>

namespace {

using ChromaspaceFrameRecoveryPolicy::Action;
using ChromaspaceFrameRecoveryPolicy::Config;
using ChromaspaceFrameRecoveryPolicy::FrameRecoveryPolicy;
using ChromaspaceFrameRecoveryPolicy::SurfaceVisibility;
using FailureKind = ChromaspaceMetalFrameFailure::Kind;

void testTransientBackoffAndRecreationBoundary() {
  FrameRecoveryPolicy policy;
  auto decision = policy.onFailure(FailureKind::DrawableUnavailable,
                                   SurfaceVisibility::Visible);
  assert(decision.action == Action::RetryLater);
  assert(decision.backoffMilliseconds == 8u);
  assert(decision.transientFailureCount == 1u);
  decision = policy.onFailure(FailureKind::BackpressureTimeout,
                              SurfaceVisibility::Visible);
  assert(decision.action == Action::RetryLater);
  assert(decision.backoffMilliseconds == 16u);
  decision = policy.onFailure(FailureKind::DrawableUnavailable,
                              SurfaceVisibility::Visible);
  assert(decision.action == Action::RetryLater);
  assert(decision.backoffMilliseconds == 32u);
  decision = policy.onFailure(FailureKind::BackpressureTimeout,
                              SurfaceVisibility::Visible);
  assert(decision.action == Action::RecreateRuntime);
  assert(decision.transientFailureCount == 0u);
  assert(decision.recreationAttempts == 1u);
  assert(policy.onRuntimeRecreationResult(true).action == Action::Continue);

  // Recreation alone does not reset the failure budget.  A presented frame
  // is the explicit success boundary.
  decision = policy.onFailure(FailureKind::DrawableUnavailable,
                              SurfaceVisibility::Visible);
  assert(decision.action == Action::RetryLater);
  assert(decision.recreationAttempts == 1u);
  assert(policy.onPresentedFrame().action == Action::Continue);
  assert(policy.transientFailureCount() == 0u);
  assert(policy.recreationAttempts() == 0u);
}

void testOcclusionDoesNotConsumeBudget() {
  FrameRecoveryPolicy policy;
  auto decision = policy.onFailure(FailureKind::DrawableUnavailable,
                                   SurfaceVisibility::Occluded);
  assert(decision.action == Action::SuspendUntilVisible);
  assert(decision.transientFailureCount == 0u);
  assert(decision.recreationAttempts == 0u);
  decision = policy.onFailure(FailureKind::BackpressureTimeout,
                              SurfaceVisibility::Unavailable);
  assert(decision.action == Action::SuspendUntilVisible);
  assert(policy.transientFailureCount() == 0u);
  decision = policy.onFailure(FailureKind::DrawableUnavailable,
                              SurfaceVisibility::Visible);
  assert(decision.action == Action::RetryLater);
  assert(decision.transientFailureCount == 1u);
}

void testDeviceLossRecreationExhaustion() {
  Config config{};
  config.maxRecreationAttempts = 2u;
  config.baseBackoffMilliseconds = 5u;
  config.maxBackoffMilliseconds = 6u;
  FrameRecoveryPolicy policy(config);

  auto decision = policy.onFailure(FailureKind::PriorGpuSubmissionFailure,
                                   SurfaceVisibility::Visible);
  assert(decision.action == Action::RecreateRuntime);
  assert(decision.recreationAttempts == 1u);
  assert(decision.backoffMilliseconds == 5u);
  decision = policy.onRuntimeRecreationResult(false);
  assert(decision.action == Action::RecreateRuntime);
  assert(decision.recreationAttempts == 2u);
  assert(decision.backoffMilliseconds == 6u);
  decision = policy.onRuntimeRecreationResult(false);
  assert(decision.action == Action::Terminate);
  assert(decision.recreationAttempts == 2u);
  assert(policy.onRuntimeRecreationResult(false).action == Action::Terminate);
}

void testFailClosedCategoriesAndInvalidValues() {
  FrameRecoveryPolicy policy;
  assert(policy.onFailure(FailureKind::InvalidState,
                          SurfaceVisibility::Visible)
             .action == Action::Terminate);
  assert(policy.onFailure(FailureKind::InvariantViolation,
                          SurfaceVisibility::Visible)
             .action == Action::Terminate);
  assert(policy.onFailure(FailureKind::EncodingFailure,
                          SurfaceVisibility::Visible)
             .action == Action::Terminate);
  assert(policy.onFailure(FailureKind::Unknown, SurfaceVisibility::Visible)
             .action == Action::Terminate);
  assert(policy.onFailure(FailureKind::None, SurfaceVisibility::Visible)
             .action == Action::Terminate);
  assert(policy
             .onFailure(static_cast<FailureKind>(255u),
                        SurfaceVisibility::Visible)
             .action == Action::Terminate);
}

void testLabelsAreStableTelemetryOnly() {
  assert(ChromaspaceMetalFrameFailure::label(FailureKind::DrawableUnavailable) !=
         nullptr);
  assert(ChromaspaceMetalFrameFailure::label(
             static_cast<FailureKind>(255u)) != nullptr);
  assert(ChromaspaceFrameRecoveryPolicy::actionLabel(Action::RetryLater) !=
         nullptr);
  assert(ChromaspaceFrameRecoveryPolicy::visibilityLabel(
             SurfaceVisibility::Unavailable) != nullptr);
}

}  // namespace

int main() {
  testTransientBackoffAndRecreationBoundary();
  testOcclusionDoesNotConsumeBudget();
  testDeviceLossRecreationExhaustion();
  testFailClosedCategoriesAndInvalidValues();
  testLabelsAreStableTelemetryOnly();
  return 0;
}
