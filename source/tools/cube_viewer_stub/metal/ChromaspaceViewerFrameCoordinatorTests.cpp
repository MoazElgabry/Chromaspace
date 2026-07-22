#include "ChromaspaceViewerFrameCoordinator.h"

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

using namespace ChromaspaceViewer;

ViewerFrameCoordinatorObservation observation(uint64_t now = 0u) {
  ViewerFrameCoordinatorObservation result{};
  result.monotonicTimeMilliseconds = now;
  result.renderable = true;
  return result;
}

void hiddenWaitsWithoutClose() {
  ViewerFrameCoordinator coordinator{};
  auto state = observation(1u);
  state.renderable = false;
  coordinator.observe(state);
  const auto decision = coordinator.decide(1u);
  assert(decision.kind == ViewerFrameDecisionKind::Wait);
  assert(decision.waitIndefinitely);
  assert(decision.waitMilliseconds > 0u);
  assert(!decision.shouldClose());
  assert((coordinator.pendingReasonMask() &
          viewerFrameDirtyMask(ViewerFrameDirtyReason::Initial)) != 0u);
}

void initialVisibleRendersOnce() {
  ViewerFrameCoordinator coordinator{};
  coordinator.observe(observation(10u));
  const auto decision = coordinator.decide(10u);
  assert(decision.kind == ViewerFrameDecisionKind::Render);
  assert(decision.renderTicket != 0u);
  assert((decision.reasonMask & viewerFrameDirtyMask(
                                 ViewerFrameDirtyReason::Initial)) != 0u);
  assert(coordinator.complete(decision.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  auto stable = observation(26u);
  coordinator.observe(stable);
  assert(coordinator.decide(26u).kind == ViewerFrameDecisionKind::Wait);
}

void revisionsTriggerAndCoalesce() {
  ViewerFrameCoordinator coordinator{};
  auto state = observation(0u);
  coordinator.observe(state);
  auto first = coordinator.decide(0u);
  assert(first.shouldRender());
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());

  state.monotonicTimeMilliseconds = 16u;
  state.viewportRevision = 1u;
  state.inputRevision = 1u;
  state.workspaceRevision = 1u;
  state.sourceRevision = 1u;
  coordinator.observe(state);
  const auto dirty = coordinator.decide(16u);
  assert(dirty.shouldRender());
  const uint32_t expected = viewerFrameDirtyMask(ViewerFrameDirtyReason::Viewport) |
                            viewerFrameDirtyMask(ViewerFrameDirtyReason::Input) |
                            viewerFrameDirtyMask(ViewerFrameDirtyReason::Workspace) |
                            viewerFrameDirtyMask(ViewerFrameDirtyReason::Source);
  assert((dirty.reasonMask & expected) == expected);
  // A second observation with the same revisions is coalesced.
  coordinator.observe(state);
  assert(coordinator.complete(dirty.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  assert(coordinator.decide(32u).shouldWait());
}

void hiddenDirtyRetainedThenVisible() {
  ViewerFrameCoordinator coordinator{};
  auto state = observation(0u);
  state.renderable = false;
  coordinator.observe(state);
  state.workspaceRevision = 7u;
  state.monotonicTimeMilliseconds = 5u;
  coordinator.observe(state);
  assert(coordinator.decide(5u).shouldWait());
  state.renderable = true;
  state.monotonicTimeMilliseconds = 16u;
  coordinator.observe(state);
  assert(coordinator.decide(16u).shouldRender());
}

void animationAndQualificationArePaced() {
  ViewerFrameCoordinatorConfig config{};
  config.frameIntervalMilliseconds = 16u;
  ViewerFrameCoordinator coordinator(config);
  auto state = observation(0u);
  state.animationContinuous = true;
  state.qualificationContinuous = true;
  coordinator.observe(state);
  auto first = coordinator.decide(0u);
  assert(first.shouldRender());
  assert((first.reasonMask & viewerFrameDirtyMask(ViewerFrameDirtyReason::Animation)) != 0u);
  assert((first.reasonMask & viewerFrameDirtyMask(ViewerFrameDirtyReason::Qualification)) != 0u);
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  auto early = coordinator.decide(8u);
  assert(early.shouldWait());
  assert(early.waitMilliseconds >= 8u);
  auto next = coordinator.decide(16u);
  assert(next.shouldRender());
  assert(coordinator.complete(next.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
}

void retryAndRecreationRetainDirtyWork() {
  ViewerFrameCoordinator coordinator{};
  coordinator.observe(observation(0u));
  const auto first = coordinator.decide(0u);
  assert(first.shouldRender());
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Retry)
             .accepted());
  assert(coordinator.pendingReasonMask() != 0u);
  const auto retry = coordinator.decide(16u);
  assert(retry.shouldRender());
  assert(coordinator.complete(retry.renderTicket,
                              ViewerFrameCompletionKind::RuntimeRecreated)
             .accepted());
  assert((coordinator.pendingReasonMask() &
          viewerFrameDirtyMask(ViewerFrameDirtyReason::Recovery)) != 0u);
  const auto recovered = coordinator.decide(32u);
  assert(recovered.shouldRender());
}

void sameReasonChangingInFlightSurvivesPresented() {
  ViewerFrameCoordinator coordinator{};
  auto state = observation(0u);
  coordinator.observe(state);
  const auto first = coordinator.decide(0u);
  assert(first.shouldRender());
  state.viewportRevision = 1u;
  state.monotonicTimeMilliseconds = 1u;
  coordinator.observe(state);
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  assert((coordinator.pendingReasonMask() &
          viewerFrameDirtyMask(ViewerFrameDirtyReason::Viewport)) != 0u);
  const auto next = coordinator.decide(16u);
  assert(next.shouldRender());
  assert((next.reasonMask &
          viewerFrameDirtyMask(ViewerFrameDirtyReason::Viewport)) != 0u);
}

void staleAndUnknownCompletionsAreRejected() {
  ViewerFrameCoordinator coordinator{};
  coordinator.observe(observation(0u));
  const auto first = coordinator.decide(0u);
  assert(coordinator.complete(first.renderTicket + 1u,
                              ViewerFrameCompletionKind::Presented)
             .status == ViewerFrameCompletionStatus::RejectedUnknownTicket);
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  auto changed = observation(16u);
  changed.workspaceRevision = 1u;
  coordinator.observe(changed);
  const auto second = coordinator.decide(16u);
  assert(second.shouldRender());
  assert(coordinator.complete(first.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .status == ViewerFrameCompletionStatus::RejectedStaleTicket);
  assert(coordinator.complete(second.renderTicket,
                              ViewerFrameCompletionKind::Presented)
             .accepted());
  assert(coordinator.complete(0u, ViewerFrameCompletionKind::Presented)
             .status == ViewerFrameCompletionStatus::RejectedUnknownTicket);
}

void closeWinsOverWaitAndRender() {
  ViewerFrameCoordinator coordinator{};
  auto state = observation(0u);
  coordinator.observe(state);
  state.closeRequested = true;
  coordinator.observe(state);
  assert(coordinator.decide(0u).shouldClose());
}

}  // namespace

int main() {
  hiddenWaitsWithoutClose();
  initialVisibleRendersOnce();
  revisionsTriggerAndCoalesce();
  hiddenDirtyRetainedThenVisible();
  animationAndQualificationArePaced();
  retryAndRecreationRetainDirtyWork();
  sameReasonChangingInFlightSurvivesPresented();
  staleAndUnknownCompletionsAreRejected();
  closeWinsOverWaitAndRender();
  std::cout << "Chromaspace viewer frame-coordinator tests passed\n";
  return 0;
}
