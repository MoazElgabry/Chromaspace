#include "ChromaspaceMetalQualificationFaultBackend.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using ChromaspaceMetal::FrameCompletionStats;
using ChromaspaceMetal::FrameSubmission;
using ChromaspaceMetal::FrameTransientMemoryStats;
using ChromaspaceMetalFrameExecutor::FrameBatch;
using ChromaspaceMetalFrameExecutor::FrameCompositorState;
using ChromaspaceMetalFrameExecutor::FrameExecutorBackend;
using ChromaspaceMetalFrameExecutor::FrameFailure;
using ChromaspaceMetalQualification::Action;
using ChromaspaceMetalQualification::ActionKind;
using ChromaspaceMetalQualification::CompletionExpectation;
using ChromaspaceMetalQualification::FaultBackend;
using ChromaspaceMetalQualification::RecoveryPhase;
using ChromaspaceMetalQualification::RuntimeObservation;

struct MockState final {
  void* expectedContext = nullptr;
  int createCalls = 0;
  int resizeCalls = 0;
  int drainCalls = 0;
  int destroyCalls = 0;
  int beginCalls = 0;
  int submitCalls = 0;
  int abandonCalls = 0;
  int transientStatsCalls = 0;
  int completionStatsCalls = 0;
  bool beginFailure = false;
  bool submitFailure = false;
  bool throwTransientStats = false;
  bool throwCompletionStats = false;
};

bool create(void* context,
            void* nativeWindow,
            int width,
            int height,
            float scale,
            FrameCompositorState* output,
            std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && nativeWindow != nullptr);
  assert(width == 640 && height == 360 && scale == 2.0f);
  assert(error != nullptr);
  ++state->createCalls;
  if (output != nullptr) {
    output->compositorId = 77u;
    output->drawableWidth = width;
    output->drawableHeight = height;
    output->contentsScale = scale;
  }
  return true;
}

bool resize(void* context,
            uint64_t compositorId,
            int width,
            int height,
            float scale,
            std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u);
  assert(width == 800 && height == 450 && scale == 1.5f);
  assert(error != nullptr);
  ++state->resizeCalls;
  return true;
}

bool drain(void* context,
           uint64_t compositorId,
           uint32_t timeout,
           std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u && timeout == 500u);
  assert(error != nullptr);
  ++state->drainCalls;
  return true;
}

void destroy(void* context, uint64_t compositorId) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u);
  ++state->destroyCalls;
}

bool begin(void* context,
           uint64_t compositorId,
           FrameSubmission* submission,
           std::string* error,
           FrameFailure* failure) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u);
  assert(error != nullptr && failure != nullptr);
  ++state->beginCalls;
  if (state->beginFailure) {
    *failure = FrameFailure::PriorGpuSubmissionFailure;
    return false;
  }
  if (submission != nullptr) {
    submission->submissionId = 101u;
    submission->compositorId = compositorId;
  }
  *failure = FrameFailure::None;
  return true;
}

bool submit(void* context,
            FrameSubmission* submission,
            const FrameBatch& batch,
            std::string* error,
            FrameFailure* failure) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && error != nullptr && failure != nullptr);
  (void)batch;
  ++state->submitCalls;
  if (state->submitFailure) {
    *failure = FrameFailure::EncodingFailure;
    return false;
  }
  if (submission != nullptr) *submission = FrameSubmission{};
  *failure = FrameFailure::None;
  return true;
}

void abandon(void* context, FrameSubmission* submission) noexcept {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr);
  ++state->abandonCalls;
  if (submission != nullptr) *submission = FrameSubmission{};
}

bool transientStats(void* context,
                   uint64_t compositorId,
                   FrameTransientMemoryStats* output) {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u);
  ++state->transientStatsCalls;
  if (state->throwTransientStats) throw std::runtime_error("metrics");
  if (output != nullptr) {
    output->available = true;
    output->activeSubmissionCount = 2u;
    output->peakActiveSubmissionCount = 3u;
  }
  return true;
}

bool completionStats(void* context,
                    uint64_t compositorId,
                    FrameCompletionStats* output) {
  auto* state = static_cast<MockState*>(context);
  assert(state != nullptr && compositorId == 77u);
  ++state->completionStatsCalls;
  if (state->throwCompletionStats) throw std::runtime_error("metrics");
  if (output != nullptr) {
    output->available = true;
    output->submittedSerial = 4u;
    output->completedSerial = 4u;
  }
  return true;
}

FrameExecutorBackend makeBackend(MockState* state, bool optional = true) {
  FrameExecutorBackend backend{};
  backend.context = state;
  backend.create = create;
  backend.resize = resize;
  backend.drain = drain;
  backend.destroy = destroy;
  backend.begin = begin;
  backend.submit = submit;
  backend.abandon = abandon;
  if (optional) {
    backend.transientMemoryStats = transientStats;
    backend.completionStats = completionStats;
  }
  return backend;
}

Action fault(ActionKind kind, uint64_t ordinal) {
  Action action{};
  action.kind = kind;
  action.ordinal = ordinal;
  return action;
}

void invalidBase() {
  FaultBackend nullBase(nullptr);
  assert(!nullBase.ready());
  assert(nullBase.backend() != nullptr);
  assert(std::string(nullBase.diagnostic()) ==
         "qualification-fault-backend-invalid-base");
  FrameCompositorState state{};
  std::string error;
  assert(!nullBase.backend()->create(nullBase.backend()->context, nullptr, 1,
                                     1, 1.0f, &state, &error));
  assert(state.compositorId == 0u);
  assert(!error.empty());

  FrameExecutorBackend incomplete{};
  FaultBackend invalid(&incomplete);
  assert(!invalid.ready());
  assert(!invalid.arm(fault(ActionKind::InjectDrawableUnavailable, 1u)));
}

void passThroughAndOptionalStats() {
  MockState state{};
  FrameExecutorBackend base = makeBackend(&state);
  FaultBackend wrapper(&base);
  assert(wrapper.ready());
  assert(wrapper.backend()->context == &wrapper);
  std::string error;
  FrameCompositorState compositor{};
  int nativeWindow = 1;
  assert(wrapper.backend()->create(wrapper.backend()->context, &nativeWindow,
                                   640, 360, 2.0f, &compositor, &error));
  assert(state.createCalls == 1);
  assert(wrapper.backend()->resize(wrapper.backend()->context, 77u, 800, 450,
                                   1.5f, &error));
  assert(wrapper.backend()->drain(wrapper.backend()->context, 77u, 500u,
                                  &error));
  FrameSubmission submission{};
  FrameFailure failure = FrameFailure::Unknown;
  assert(wrapper.backend()->begin(wrapper.backend()->context, 77u, &submission,
                                  &error, &failure));
  assert(submission.submissionId == 101u && failure == FrameFailure::None);
  FrameBatch batch{};
  assert(wrapper.backend()->submit(wrapper.backend()->context, &submission,
                                   batch, &error, &failure));
  assert(submission.submissionId == 0u);
  wrapper.backend()->abandon(wrapper.backend()->context, &submission);
  wrapper.backend()->destroy(wrapper.backend()->context, 77u);
  assert(state.resizeCalls == 1 && state.drainCalls == 1);
  assert(state.beginCalls == 1 && state.submitCalls == 1);
  assert(state.abandonCalls == 1 && state.destroyCalls == 1);

  FrameTransientMemoryStats transient{};
  FrameCompletionStats completion{};
  assert(wrapper.backend()->transientMemoryStats(
      wrapper.backend()->context, 77u, &transient));
  assert(wrapper.backend()->completionStats(wrapper.backend()->context, 77u,
                                            &completion));
  assert(transient.available && transient.peakActiveSubmissionCount == 3u);
  assert(completion.available && completion.completedSerial == 4u);
  assert(state.transientStatsCalls == 1 && state.completionStatsCalls == 1);

  state.throwTransientStats = true;
  state.throwCompletionStats = true;
  assert(!wrapper.backend()->transientMemoryStats(
      wrapper.backend()->context, 77u, &transient));
  assert(!transient.available);
  assert(!wrapper.backend()->completionStats(wrapper.backend()->context, 77u,
                                             &completion));
  assert(!completion.available);
}

void armAndInjectExactFailures() {
  MockState state{};
  FrameExecutorBackend base = makeBackend(&state, false);
  FaultBackend wrapper(&base);
  assert(!wrapper.arm(fault(ActionKind::Resize, 1u)));
  assert(!wrapper.arm(fault(ActionKind::InjectDrawableUnavailable, 0u)));
  Action drawable = fault(ActionKind::InjectDrawableUnavailable, 4u);
  assert(wrapper.arm(drawable));
  assert(!wrapper.arm(fault(ActionKind::InjectPriorGpuSubmissionFailure, 5u)));
  assert(!wrapper.arm(fault(ActionKind::InjectDrawableUnavailable, 3u)));

  FrameSubmission submission{999u, 999u};
  FrameFailure failure = FrameFailure::None;
  std::string error;
  assert(!wrapper.backend()->begin(wrapper.backend()->context, 77u,
                                   &submission, &error, &failure));
  assert(submission.submissionId == 0u && submission.compositorId == 0u);
  assert(state.beginCalls == 0);
  assert(failure == FrameFailure::DrawableUnavailable);
  assert(wrapper.snapshot().firedCount == 1u);
  assert(wrapper.snapshot().recoveryPhase ==
         RecoveryPhase::DrawableRetryPending);
  assert(wrapper.observe(RuntimeObservation::RetryLater, 4u));
  assert(wrapper.snapshot().recoveryPhase ==
         RecoveryPhase::DrawablePresentedPending);
  assert(wrapper.observe(RuntimeObservation::Presented, 4u));

  Action prior = fault(ActionKind::InjectPriorGpuSubmissionFailure, 9u);
  assert(wrapper.arm(prior));
  failure = FrameFailure::None;
  submission = FrameSubmission{123u, 456u};
  assert(!wrapper.backend()->begin(wrapper.backend()->context, 77u,
                                   &submission, &error, &failure));
  assert(state.beginCalls == 0);
  assert(failure == FrameFailure::PriorGpuSubmissionFailure);
  assert(wrapper.observe(RuntimeObservation::RuntimeRecreated, 9u));
  assert(wrapper.observe(RuntimeObservation::Presented, 9u));
  const auto snapshot = wrapper.snapshot();
  assert(snapshot.armedCount == 2u && snapshot.firedCount == 2u &&
         snapshot.recoveredCount == 2u);
  assert(snapshot.firedByKind[static_cast<std::size_t>(
             ActionKind::InjectDrawableUnavailable)] == 1u);
  assert(snapshot.firedByKind[static_cast<std::size_t>(
             ActionKind::InjectPriorGpuSubmissionFailure)] == 1u);
  assert(!wrapper.arm(
      fault(ActionKind::InjectDrawableUnavailable, 9u)));
  assert(!wrapper.failed());
  assert(wrapper.arm(
      fault(ActionKind::InjectDrawableUnavailable, 12u)));
  failure = FrameFailure::None;
  submission = FrameSubmission{321u, 654u};
  assert(!wrapper.backend()->begin(wrapper.backend()->context, 77u,
                                   &submission, &error, &failure));
  assert(failure == FrameFailure::DrawableUnavailable);
  assert(wrapper.observe(RuntimeObservation::RetryLater, 12u));
  assert(wrapper.observe(RuntimeObservation::Presented, 12u));
  assert(wrapper.finish(CompletionExpectation{3u, false}));
}

void wrongObservationAndFinishContracts() {
  MockState state{};
  FrameExecutorBackend base = makeBackend(&state, false);

  FaultBackend idle(&base);
  assert(idle.observe(RuntimeObservation::RetryLater, 0u));
  assert(idle.observe(RuntimeObservation::SuspendUntilVisible, 0u));
  assert(idle.observe(RuntimeObservation::RuntimeRecreated, 0u));
  assert(idle.observe(RuntimeObservation::Presented, 0u));
  assert(!idle.failed());
  assert(idle.snapshot().firedCount == 0u);
  assert(!idle.observe(RuntimeObservation::TerminalFailure, 0u));
  assert(idle.failed());

  FaultBackend invalidObservation(&base);
  assert(!invalidObservation.observe(
      static_cast<RuntimeObservation>(255u), 0u));
  assert(invalidObservation.failed());

  FaultBackend wrong(&base);
  assert(wrong.arm(fault(ActionKind::InjectDrawableUnavailable, 1u)));
  FrameSubmission submission{};
  FrameFailure failure = FrameFailure::None;
  std::string error;
  assert(!wrong.backend()->begin(wrong.backend()->context, 77u, &submission,
                                 &error, &failure));
  assert(!wrong.observe(RuntimeObservation::Presented, 1u));
  assert(wrong.failed());
  assert(!wrong.observe(RuntimeObservation::RetryLater, 1u));

  FaultBackend stale(&base);
  assert(stale.arm(fault(ActionKind::InjectDrawableUnavailable, 4u)));
  assert(!stale.observe(RuntimeObservation::RetryLater, 3u));
  assert(stale.failed());

  FaultBackend terminal(&base);
  assert(terminal.arm(fault(ActionKind::InjectDrawableUnavailable, 8u)));
  assert(!terminal.backend()->begin(terminal.backend()->context, 77u,
                                    &submission, &error, &failure));
  assert(!terminal.observe(RuntimeObservation::TerminalFailure, 8u));
  assert(terminal.failed());

  FaultBackend early(&base);
  assert(!early.finish());
  assert(!early.failed());
  assert(early.arm(fault(ActionKind::InjectDrawableUnavailable, 1u)));
  assert(!early.finish(CompletionExpectation{1u, false}));
  assert(!early.failed());
  assert(!early.arm(fault(ActionKind::InjectPriorGpuSubmissionFailure, 2u)));
  assert(!early.observe(RuntimeObservation::Presented, 1u));
  assert(early.failed());

  FaultBackend reusable(&base);
  assert(reusable.finish(CompletionExpectation{0u, false}));
  assert(!reusable.arm(fault(ActionKind::InjectDrawableUnavailable, 1u)));
}

void BaseFailurePassThrough() {
  MockState state{};
  state.beginFailure = true;
  FrameExecutorBackend base = makeBackend(&state, false);
  FaultBackend wrapper(&base);
  FrameSubmission submission{};
  FrameFailure failure = FrameFailure::None;
  std::string error;
  assert(!wrapper.backend()->begin(wrapper.backend()->context, 77u,
                                   &submission, &error, &failure));
  assert(state.beginCalls == 1);
  assert(failure == FrameFailure::PriorGpuSubmissionFailure);
  assert(!wrapper.failed());
  assert(wrapper.finish(CompletionExpectation{0u, false}));
}

}  // namespace

int main() {
  invalidBase();
  passThroughAndOptionalStats();
  armAndInjectExactFailures();
  wrongObservationAndFinishContracts();
  BaseFailurePassThrough();
  return 0;
}
