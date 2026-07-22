#include "ChromaspaceMetalQualificationFaultBackend.h"

#include <algorithm>

namespace ChromaspaceMetalQualification {
namespace {

using ChromaspaceMetalFrameExecutor::FrameCompositorState;
using ChromaspaceMetalFrameExecutor::FrameExecutorBackend;
using ChromaspaceMetalFrameExecutor::FrameFailure;

constexpr const char* kReady = "qualification-fault-backend-ready";
constexpr const char* kInvalidBase =
    "qualification-fault-backend-invalid-base";
constexpr const char* kNotReady = "qualification-fault-backend-not-ready";
constexpr const char* kAlreadyCompleted =
    "qualification-fault-backend-already-completed";
constexpr const char* kFaultAlreadyActive =
    "qualification-fault-backend-fault-already-active";
constexpr const char* kInvalidFaultKind =
    "qualification-fault-backend-invalid-fault-kind";
constexpr const char* kStaleFaultOrdinal =
    "qualification-fault-backend-stale-fault-ordinal";
constexpr const char* kFaultInjectedDrawable =
    "qualification-fault-backend-drawable-unavailable";
constexpr const char* kFaultInjectedPriorGpu =
    "qualification-fault-backend-prior-gpu-submission-failure";
constexpr const char* kExpectedDrawableRetry =
    "qualification-fault-backend-expected-drawable-retry";
constexpr const char* kExpectedDrawablePresented =
    "qualification-fault-backend-expected-drawable-presented";
constexpr const char* kExpectedPriorRecreated =
    "qualification-fault-backend-expected-prior-recreation";
constexpr const char* kExpectedPriorPresented =
    "qualification-fault-backend-expected-prior-presented";
constexpr const char* kObservationOrdinal =
    "qualification-fault-backend-observation-ordinal";
constexpr const char* kTerminalObservation =
    "qualification-fault-backend-terminal-observation";
constexpr const char* kFinishNotReady =
    "qualification-fault-backend-finish-not-ready";
constexpr const char* kFinishFailed = "qualification-fault-backend-finish-failed";
constexpr const char* kFinishActive =
    "qualification-fault-backend-finish-fault-active";
constexpr const char* kFinishCount =
    "qualification-fault-backend-finish-count-mismatch";
constexpr const char* kFinishKindCount =
    "qualification-fault-backend-finish-kind-count-mismatch";

bool validBase(const FrameExecutorBackend* base) noexcept {
  return base != nullptr && base->create != nullptr &&
         base->resize != nullptr && base->drain != nullptr &&
         base->destroy != nullptr && base->begin != nullptr &&
         base->submit != nullptr && base->abandon != nullptr;
}

void clearCompositor(FrameCompositorState* state) noexcept {
  if (state != nullptr) *state = FrameCompositorState{};
}

void clearSubmission(ChromaspaceMetal::FrameSubmission* submission) noexcept {
  if (submission != nullptr) *submission = ChromaspaceMetal::FrameSubmission{};
}

void clearTransientStats(
    ChromaspaceMetal::FrameTransientMemoryStats* stats) noexcept {
  if (stats != nullptr) *stats = ChromaspaceMetal::FrameTransientMemoryStats{};
}

void clearCompletionStats(ChromaspaceMetal::FrameCompletionStats* stats) noexcept {
  if (stats != nullptr) *stats = ChromaspaceMetal::FrameCompletionStats{};
}

void setErrorNoThrow(std::string* output, const char* value) noexcept {
  if (output == nullptr) return;
  try {
    *output = value != nullptr ? value : "qualification-fault-backend-error";
  } catch (...) {
  }
}

void setFailureNoThrow(FrameFailure* failure, FrameFailure value) noexcept {
  if (failure != nullptr) *failure = value;
}

FaultBackend* instance(void* context) noexcept {
  return static_cast<FaultBackend*>(context);
}

bool isFaultKind(ActionKind kind) noexcept {
  return kind == ActionKind::InjectDrawableUnavailable ||
         kind == ActionKind::InjectPriorGpuSubmissionFailure;
}

std::size_t kindIndex(ActionKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

}  // namespace

FaultBackend::FaultBackend(const FrameExecutorBackend* base) noexcept {
  wrappedBackend_.context = this;
  wrappedBackend_.create = &FaultBackend::createCallback;
  wrappedBackend_.resize = &FaultBackend::resizeCallback;
  wrappedBackend_.drain = &FaultBackend::drainCallback;
  wrappedBackend_.destroy = &FaultBackend::destroyCallback;
  wrappedBackend_.begin = &FaultBackend::beginCallback;
  wrappedBackend_.submit = &FaultBackend::submitCallback;
  wrappedBackend_.abandon = &FaultBackend::abandonCallback;
  if (!validBase(base)) {
    diagnostic_ = kInvalidBase;
    return;
  }
  baseBackend_ = *base;
  ready_ = true;
  diagnostic_ = kReady;
  if (baseBackend_.transientMemoryStats != nullptr) {
    wrappedBackend_.transientMemoryStats =
        &FaultBackend::transientMemoryStatsCallback;
  }
  if (baseBackend_.completionStats != nullptr) {
    wrappedBackend_.completionStats = &FaultBackend::completionStatsCallback;
  }
}

bool FaultBackend::createCallback(void* context,
                                  void* nativeWindow,
                                  int drawableWidth,
                                  int drawableHeight,
                                  float contentsScale,
                                  FrameCompositorState* outState,
                                  std::string* error) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardCreate(nativeWindow, drawableWidth, drawableHeight,
                                   contentsScale, outState, error)
             : (clearCompositor(outState), setErrorNoThrow(error, kNotReady),
                false);
}

bool FaultBackend::resizeCallback(void* context,
                                  uint64_t compositorId,
                                  int drawableWidth,
                                  int drawableHeight,
                                  float contentsScale,
                                  std::string* error) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardResize(compositorId, drawableWidth, drawableHeight,
                                   contentsScale, error)
             : (setErrorNoThrow(error, kNotReady), false);
}

bool FaultBackend::drainCallback(void* context,
                                 uint64_t compositorId,
                                 uint32_t timeoutMilliseconds,
                                 std::string* error) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardDrain(compositorId, timeoutMilliseconds, error)
             : (setErrorNoThrow(error, kNotReady), false);
}

void FaultBackend::destroyCallback(void* context,
                                   uint64_t compositorHandle) noexcept {
  FaultBackend* self = instance(context);
  if (self != nullptr) self->forwardDestroy(compositorHandle);
}

bool FaultBackend::beginCallback(void* context,
                                 uint64_t compositorId,
                                 ChromaspaceMetal::FrameSubmission* outSubmission,
                                 std::string* error,
                                 FrameFailure* failure) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardBegin(compositorId, outSubmission, error, failure)
             : (clearSubmission(outSubmission),
                setErrorNoThrow(error, kNotReady),
                setFailureNoThrow(failure, FrameFailure::InvalidState), false);
}

bool FaultBackend::submitCallback(
    void* context,
    ChromaspaceMetal::FrameSubmission* submission,
    const ChromaspaceMetalFrameExecutor::FrameBatch& batch,
    std::string* error,
    FrameFailure* failure) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardSubmit(submission, batch, error, failure)
             : (setErrorNoThrow(error, kNotReady),
                setFailureNoThrow(failure, FrameFailure::InvalidState), false);
}

void FaultBackend::abandonCallback(
    void* context,
    ChromaspaceMetal::FrameSubmission* submission) noexcept {
  FaultBackend* self = instance(context);
  if (self != nullptr) self->forwardAbandon(submission);
}

bool FaultBackend::transientMemoryStatsCallback(
    void* context,
    uint64_t compositorId,
    ChromaspaceMetal::FrameTransientMemoryStats* outStats) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardTransientMemoryStats(compositorId, outStats)
             : (clearTransientStats(outStats), false);
}

bool FaultBackend::completionStatsCallback(
    void* context,
    uint64_t compositorId,
    ChromaspaceMetal::FrameCompletionStats* outStats) noexcept {
  FaultBackend* self = instance(context);
  return self != nullptr
             ? self->forwardCompletionStats(compositorId, outStats)
             : (clearCompletionStats(outStats), false);
}

bool FaultBackend::forwardCreate(void* nativeWindow,
                                 int drawableWidth,
                                 int drawableHeight,
                                 float contentsScale,
                                 FrameCompositorState* outState,
                                 std::string* error) noexcept {
  if (!ready_) {
    clearCompositor(outState);
    setErrorNoThrow(error, kInvalidBase);
    return false;
  }
  return baseBackend_.create(baseBackend_.context, nativeWindow, drawableWidth,
                             drawableHeight, contentsScale, outState, error);
}

bool FaultBackend::forwardResize(uint64_t compositorId,
                                 int drawableWidth,
                                 int drawableHeight,
                                 float contentsScale,
                                 std::string* error) noexcept {
  if (!ready_) {
    setErrorNoThrow(error, kInvalidBase);
    return false;
  }
  return baseBackend_.resize(baseBackend_.context, compositorId, drawableWidth,
                             drawableHeight, contentsScale, error);
}

bool FaultBackend::forwardDrain(uint64_t compositorId,
                                uint32_t timeoutMilliseconds,
                                std::string* error) noexcept {
  if (!ready_) {
    setErrorNoThrow(error, kInvalidBase);
    return false;
  }
  return baseBackend_.drain(baseBackend_.context, compositorId,
                            timeoutMilliseconds, error);
}

void FaultBackend::forwardDestroy(uint64_t compositorHandle) noexcept {
  if (ready_) baseBackend_.destroy(baseBackend_.context, compositorHandle);
}

bool FaultBackend::forwardBegin(
    uint64_t compositorId,
    ChromaspaceMetal::FrameSubmission* outSubmission,
    std::string* error,
    FrameFailure* failure) noexcept {
  if (!ready_) {
    clearSubmission(outSubmission);
    setErrorNoThrow(error, kInvalidBase);
    setFailureNoThrow(failure, FrameFailure::InvalidState);
    return false;
  }
  if (recoveryPhase_ == RecoveryPhase::Armed) {
    clearSubmission(outSubmission);
    const ActionKind kind = activeAction_.kind;
    const std::size_t index = kindIndex(kind);
    if (firedCount_ != UINT32_MAX) ++firedCount_;
    if (index < firedByKind_.size() && firedByKind_[index] != UINT32_MAX) {
      ++firedByKind_[index];
    }
    recoveryPhase_ = kind == ActionKind::InjectDrawableUnavailable
                         ? RecoveryPhase::DrawableRetryPending
                         : RecoveryPhase::PriorRecreationPending;
    diagnostic_ = kind == ActionKind::InjectDrawableUnavailable
                      ? kFaultInjectedDrawable
                      : kFaultInjectedPriorGpu;
    setErrorNoThrow(error, diagnostic_);
    setFailureNoThrow(
        failure,
        kind == ActionKind::InjectDrawableUnavailable
            ? FrameFailure::DrawableUnavailable
            : FrameFailure::PriorGpuSubmissionFailure);
    return false;
  }
  return baseBackend_.begin(baseBackend_.context, compositorId, outSubmission,
                            error, failure);
}

bool FaultBackend::forwardSubmit(
    ChromaspaceMetal::FrameSubmission* submission,
    const ChromaspaceMetalFrameExecutor::FrameBatch& batch,
    std::string* error,
    FrameFailure* failure) noexcept {
  if (!ready_) {
    setErrorNoThrow(error, kInvalidBase);
    setFailureNoThrow(failure, FrameFailure::InvalidState);
    return false;
  }
  return baseBackend_.submit(baseBackend_.context, submission, batch, error,
                             failure);
}

void FaultBackend::forwardAbandon(
    ChromaspaceMetal::FrameSubmission* submission) noexcept {
  if (ready_) baseBackend_.abandon(baseBackend_.context, submission);
}

bool FaultBackend::forwardTransientMemoryStats(
    uint64_t compositorId,
    ChromaspaceMetal::FrameTransientMemoryStats* outStats) noexcept {
  if (!ready_ || baseBackend_.transientMemoryStats == nullptr) {
    clearTransientStats(outStats);
    return false;
  }
  try {
    return baseBackend_.transientMemoryStats(baseBackend_.context, compositorId,
                                             outStats);
  } catch (...) {
    clearTransientStats(outStats);
    diagnostic_ = "qualification-fault-backend-metrics-failed";
    return false;
  }
}

bool FaultBackend::forwardCompletionStats(
    uint64_t compositorId,
    ChromaspaceMetal::FrameCompletionStats* outStats) noexcept {
  if (!ready_ || baseBackend_.completionStats == nullptr) {
    clearCompletionStats(outStats);
    return false;
  }
  try {
    return baseBackend_.completionStats(baseBackend_.context, compositorId,
                                        outStats);
  } catch (...) {
    clearCompletionStats(outStats);
    diagnostic_ = "qualification-fault-backend-metrics-failed";
    return false;
  }
}

bool FaultBackend::validObservation(RuntimeObservation observation) const
    noexcept {
  const auto value = static_cast<uint8_t>(observation);
  return value < static_cast<uint8_t>(RuntimeObservation::Count);
}

bool FaultBackend::validFaultKind(ActionKind kind) const noexcept {
  const auto value = static_cast<uint8_t>(kind);
  return isFaultKind(kind) && value < static_cast<uint8_t>(ActionKind::Count);
}

bool FaultBackend::arm(const Action& action, std::string* output) noexcept {
  if (!ready_) {
    setErrorNoThrow(output, kNotReady);
    return false;
  }
  if (failed_) {
    setErrorNoThrow(output, kFinishFailed);
    return false;
  }
  if (completed_) {
    setErrorNoThrow(output, kAlreadyCompleted);
    return false;
  }
  if (recoveryPhase_ != RecoveryPhase::Idle) {
    setErrorNoThrow(output, kFaultAlreadyActive);
    return false;
  }
  if (!validFaultKind(action.kind)) {
    setErrorNoThrow(output, kInvalidFaultKind);
    return false;
  }
  if (action.ordinal == 0u || action.ordinal <= lastAcceptedOrdinal_) {
    setErrorNoThrow(output, kStaleFaultOrdinal);
    return false;
  }
  activeAction_ = action;
  lastAcceptedOrdinal_ = action.ordinal;
  recoveryPhase_ = RecoveryPhase::Armed;
  if (armedCount_ != UINT32_MAX) ++armedCount_;
  const std::size_t index = kindIndex(action.kind);
  if (index < armedByKind_.size() && armedByKind_[index] != UINT32_MAX) {
    ++armedByKind_[index];
  }
  diagnostic_ = kReady;
  setErrorNoThrow(output, kReady);
  return true;
}

bool FaultBackend::failPermanently(const char* failureDiagnostic,
                                   std::string* output) noexcept {
  failed_ = true;
  recoveryPhase_ = RecoveryPhase::Failed;
  diagnostic_ = failureDiagnostic != nullptr ? failureDiagnostic : kFinishFailed;
  setErrorNoThrow(output, diagnostic_);
  return false;
}

bool FaultBackend::observe(RuntimeObservation observation,
                           uint64_t ordinal,
                           std::string* output) noexcept {
  if (!ready_) {
    setErrorNoThrow(output, kNotReady);
    return false;
  }
  if (failed_) {
    setErrorNoThrow(output, kFinishFailed);
    return false;
  }
  if (!validObservation(observation)) {
    return failPermanently(kTerminalObservation, output);
  }
  if (recoveryPhase_ == RecoveryPhase::Idle) {
    if (ordinal != 0u) {
      return failPermanently(kObservationOrdinal, output);
    }
    if (observation != RuntimeObservation::TerminalFailure) {
      diagnostic_ = kReady;
      setErrorNoThrow(output, kReady);
      return true;
    }
    return failPermanently(kExpectedDrawablePresented, output);
  }
  if (recoveryPhase_ == RecoveryPhase::Failed) {
    setErrorNoThrow(output, kFinishFailed);
    return false;
  }
  if (recoveryPhase_ == RecoveryPhase::Armed) {
    return failPermanently(kExpectedDrawableRetry, output);
  }
  if (ordinal != activeAction_.ordinal) {
    return failPermanently(kObservationOrdinal, output);
  }

  if (recoveryPhase_ == RecoveryPhase::DrawableRetryPending) {
    if (observation != RuntimeObservation::RetryLater) {
      return failPermanently(kExpectedDrawableRetry, output);
    }
    recoveryPhase_ = RecoveryPhase::DrawablePresentedPending;
    diagnostic_ = kReady;
    setErrorNoThrow(output, kReady);
    return true;
  }
  if (recoveryPhase_ == RecoveryPhase::DrawablePresentedPending) {
    if (observation != RuntimeObservation::Presented) {
      return failPermanently(kExpectedDrawablePresented, output);
    }
  } else if (recoveryPhase_ == RecoveryPhase::PriorRecreationPending) {
    if (observation != RuntimeObservation::RuntimeRecreated) {
      return failPermanently(kExpectedPriorRecreated, output);
    }
    recoveryPhase_ = RecoveryPhase::PriorPresentedPending;
    diagnostic_ = kReady;
    setErrorNoThrow(output, kReady);
    return true;
  } else if (recoveryPhase_ == RecoveryPhase::PriorPresentedPending) {
    if (observation != RuntimeObservation::Presented) {
      return failPermanently(kExpectedPriorPresented, output);
    }
  } else {
    return failPermanently(kTerminalObservation, output);
  }

  const std::size_t index = kindIndex(activeAction_.kind);
  if (recoveredCount_ != UINT32_MAX) ++recoveredCount_;
  if (index < recoveredByKind_.size() && recoveredByKind_[index] != UINT32_MAX) {
    ++recoveredByKind_[index];
  }
  activeAction_ = Action{};
  recoveryPhase_ = RecoveryPhase::Idle;
  diagnostic_ = kReady;
  setErrorNoThrow(output, kReady);
  return true;
}

bool FaultBackend::finishWithExpectation(
    const CompletionExpectation& expectation,
    std::string* output) noexcept {
  if (!ready_) {
    setErrorNoThrow(output, kFinishNotReady);
    return false;
  }
  if (failed_) {
    setErrorNoThrow(output, kFinishFailed);
    return false;
  }
  if (completed_) {
    setErrorNoThrow(output, kAlreadyCompleted);
    return false;
  }
  if (recoveryPhase_ != RecoveryPhase::Idle) {
    setErrorNoThrow(output, kFinishActive);
    return false;
  }
  if (armedCount_ != expectation.expectedFaultCount ||
      firedCount_ != expectation.expectedFaultCount ||
      recoveredCount_ != expectation.expectedFaultCount ||
      firedCount_ != recoveredCount_) {
    setErrorNoThrow(output, kFinishCount);
    return false;
  }
  if (expectation.requireBothKinds) {
    const std::size_t drawable =
        kindIndex(ActionKind::InjectDrawableUnavailable);
    const std::size_t prior =
        kindIndex(ActionKind::InjectPriorGpuSubmissionFailure);
    if (firedByKind_[drawable] != 1u || recoveredByKind_[drawable] != 1u ||
        firedByKind_[prior] != 1u || recoveredByKind_[prior] != 1u) {
      setErrorNoThrow(output, kFinishKindCount);
      return false;
    }
  }
  completed_ = true;
  diagnostic_ = kReady;
  setErrorNoThrow(output, kReady);
  return true;
}

bool FaultBackend::finish(const CompletionExpectation& expectation,
                          std::string* output) noexcept {
  return finishWithExpectation(expectation, output);
}

bool FaultBackend::finish(std::string* output) noexcept {
  return finishWithExpectation(CompletionExpectation{}, output);
}

FaultSnapshot FaultBackend::snapshot() const noexcept {
  FaultSnapshot output{};
  output.armedByKind = armedByKind_;
  output.firedByKind = firedByKind_;
  output.recoveredByKind = recoveredByKind_;
  output.ready = ready_;
  output.failed = failed_;
  output.completed = completed_;
  output.armedCount = armedCount_;
  output.firedCount = firedCount_;
  output.recoveredCount = recoveredCount_;
  output.lastAcceptedOrdinal = lastAcceptedOrdinal_;
  output.activeOrdinal = activeAction_.ordinal;
  output.activeKind = activeAction_.kind;
  output.recoveryPhase = recoveryPhase_;
  return output;
}

}  // namespace ChromaspaceMetalQualification
