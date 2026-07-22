#include "ChromaspaceMetalFrameExecutor.h"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ChromaspaceMetalFrameExecutor;

struct MockBackendState {
  int createCalls = 0;
  int resizeCalls = 0;
  int drainCalls = 0;
  int destroyCalls = 0;
  int beginCalls = 0;
  int submitCalls = 0;
  int abandonCalls = 0;
  int transientStatsCalls = 0;
  int completionStatsCalls = 0;
  bool failCreatePartial = false;
  bool zeroCreateCompositorId = false;
  bool zeroCreateRuntimeContextId = false;
  bool zeroCreateDeviceRegistryId = false;
  bool failBegin = false;
  bool failBeginPartial = false;
  bool mismatchBeginRuntimeContextId = false;
  bool mismatchBeginDeviceRegistryId = false;
  FrameFailure beginFailure = FrameFailure::None;
  FrameFailure beginSuccessFailure = FrameFailure::None;
  bool failSubmit = false;
  FrameFailure submitFailure = FrameFailure::None;
  bool leaveSubmitToken = false;
  bool failTransientStats = false;
  bool throwTransientStats = false;
  bool failCompletionStats = false;
  uint64_t runtimeContextId = 700u;
  uint64_t deviceRegistryId = 900u;
  ChromaspaceMetal::FrameTransientMemoryStats transientStats{
      true, 2u, 1u, 1u, 96u, 72u, 192u, 144u, 3u,
      768u, 256u, 3u};
  ChromaspaceMetal::FrameCompletionStats completionStats{
      true, 8u, 8u, 0u, 8u, 0u, 0.080, 0.020, {}};
  std::vector<FramePassKind> passKinds;
};

bool mockCreate(void* context,
                void*,
                int width,
                int height,
                float scale,
                FrameCompositorState* out,
                std::string* error) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->createCalls;
  if (state->failCreatePartial && out != nullptr) {
    *out = ChromaspaceMetal::FrameCompositor{42u, width, height, scale};
    if (error) *error = "mock-partial-create";
    return false;
  }
  if (out == nullptr || width <= 0 || height <= 0 || scale <= 0.0f) {
    if (error) *error = "mock-invalid-create";
    return false;
  }
  *out = ChromaspaceMetal::FrameCompositor{
      state->zeroCreateCompositorId ? 0u : 42u,
      width,
      height,
      scale,
      state->zeroCreateRuntimeContextId ? 0u : state->runtimeContextId,
      state->zeroCreateDeviceRegistryId ? 0u : state->deviceRegistryId};
  return true;
}

bool mockResize(void* context,
                uint64_t compositorId,
                int width,
                int height,
                float scale,
                std::string* error) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->resizeCalls;
  if (compositorId != 42u || width <= 0 || height <= 0 || scale <= 0.0f) {
    if (error) *error = "mock-invalid-resize";
    return false;
  }
  return true;
}

bool mockDrain(void* context,
               uint64_t compositorId,
               uint32_t timeout,
               std::string* error) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->drainCalls;
  if (compositorId != 42u || timeout == 0u) {
    if (error) *error = "mock-invalid-drain";
    return false;
  }
  return true;
}

void mockDestroy(void* context, uint64_t compositorId) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->destroyCalls;
  assert(compositorId == 42u);
}

bool mockBegin(void* context,
               uint64_t compositorId,
               ChromaspaceMetal::FrameSubmission* out,
               std::string* error,
               FrameFailure* failure) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->beginCalls;
  if (state->failBegin || out == nullptr || compositorId != 42u) {
    if (state->failBeginPartial && out != nullptr) {
      *out = ChromaspaceMetal::FrameSubmission{
          77u, compositorId, state->runtimeContextId, state->deviceRegistryId};
    }
    if (error) *error = "mock-begin-failed";
    if (failure) *failure = state->beginFailure;
    return false;
  }
  *out = ChromaspaceMetal::FrameSubmission{
      77u,
      compositorId,
      state->mismatchBeginRuntimeContextId ? state->runtimeContextId + 1u
                                           : state->runtimeContextId,
      state->mismatchBeginDeviceRegistryId ? state->deviceRegistryId + 1u
                                           : state->deviceRegistryId};
  if (failure) *failure = state->beginSuccessFailure;
  return true;
}

bool mockSubmit(void* context,
                ChromaspaceMetal::FrameSubmission* submission,
                const FrameBatch&,
                std::string* error,
                FrameFailure* failure) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->submitCalls;
  if (state->failSubmit || submission == nullptr ||
      submission->submissionId != 77u) {
    if (error) *error = "mock-submit-failed";
    if (failure) *failure = state->submitFailure;
    return false;
  }
  if (!state->leaveSubmitToken) {
    *submission = ChromaspaceMetal::FrameSubmission{};
  }
  if (failure) *failure = FrameFailure::None;
  return true;
}

void mockAbandon(void* context,
                 ChromaspaceMetal::FrameSubmission* submission) noexcept {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->abandonCalls;
  if (submission) *submission = ChromaspaceMetal::FrameSubmission{};
}

bool mockTransientMemoryStats(
    void* context,
    uint64_t compositorId,
    ChromaspaceMetal::FrameTransientMemoryStats* outStats) {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->transientStatsCalls;
  if (state->throwTransientStats) {
    throw std::runtime_error("intentional-transient-stats-exception");
  }
  if (state->failTransientStats || compositorId != 42u || outStats == nullptr) {
    return false;
  }
  *outStats = state->transientStats;
  return true;
}

bool mockCompletionStats(
    void* context,
    uint64_t compositorId,
    ChromaspaceMetal::FrameCompletionStats* outStats) {
  auto* state = static_cast<MockBackendState*>(context);
  ++state->completionStatsCalls;
  if (state->failCompletionStats || compositorId != 42u || outStats == nullptr) {
    return false;
  }
  *outStats = state->completionStats;
  return true;
}

FrameExecutorBackend mockBackend(MockBackendState* state) {
  return {state,
          mockCreate,
          mockResize,
          mockDrain,
          mockDestroy,
          mockBegin,
          mockSubmit,
          mockAbandon,
          mockTransientMemoryStats,
          mockCompletionStats};
}

bool recordPass(const FrameExecutionContext& context,
                void* userContext,
                std::string*) {
  auto* state = static_cast<MockBackendState*>(userContext);
  assert(context.compositorId == 42u);
  assert(context.submission != nullptr && context.batch != nullptr);
  state->passKinds.push_back(FramePassKind::RenderUiText);
  return true;
}

struct PassRecorder {
  MockBackendState* state = nullptr;
  FramePassKind kind = FramePassKind::RenderUiText;
};

bool recordSpecificPass(const FrameExecutionContext& context,
                        void* userContext,
                        std::string*) {
  auto* recorder = static_cast<PassRecorder*>(userContext);
  assert(recorder != nullptr && recorder->state != nullptr);
  assert(context.submission != nullptr && context.batch != nullptr);
  recorder->state->passKinds.push_back(recorder->kind);
  return true;
}

bool failPass(const FrameExecutionContext& context,
              void*,
              std::string* error) {
  assert(context.submission != nullptr && context.batch != nullptr);
  if (error) *error = "intentional-pass-failure";
  return false;
}

bool throwPass(const FrameExecutionContext&, void*, std::string*) {
  throw std::runtime_error("intentional-pass-exception");
}

bool invalidPass(const FrameExecutionContext& context,
                 void*,
                 std::string*) {
  context.batch->compositeVectorVertices.push_back({0.0f, 0.0f, 1.0f,
                                                    1.0f, 1.0f, 1.0f});
  return true;
}

bool nestedPass(const FrameExecutionContext& context,
                void* userContext,
                std::string*) {
  auto* executor = static_cast<FrameExecutor*>(userContext);
  assert(context.submission != nullptr && executor->transactionActive());
  FrameBatch nestedBatch{};
  FramePassPlan nestedPlan{};
  nestedPlan.count = 1u;
  nestedPlan.passes[0] = {FramePassKind::RenderUiText, recordPass, nullptr};
  std::string nestedError;
  assert(!executor->execute(nestedPlan, &nestedBatch, nullptr, &nestedError));
  assert(nestedError == "frame-executor-execute-invalid-state");
  return true;
}

FramePassPlan plan(FramePassEncoder encoder, void* context = nullptr) {
  FramePassPlan result{};
  result.count = 1u;
  result.passes[0] = {FramePassKind::RenderUiText, encoder, context};
  return result;
}

FrameBatch validBatch() {
  FrameBatch batch{};
  batch.compositeItems.push_back({1u, 16, 16, 0, 0.0f, 0.0f, 16.0f, 16.0f,
                                  1.0f});
  batch.compositeOverlayRects.push_back({0.0f, 0.0f, 16.0f, 16.0f,
                                         0.1f, 0.2f, 0.3f, 1.0f});
  batch.compositeVectorVertices = {
      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},
      {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f},
  };
  batch.compositeTextVertices = {
      {0.0f, 0.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
  };
  batch.compositeTextRuns.push_back({9u, 0u, 3u, 1.0f, 1.0f, 1.0f, 1.0f,
                                     0.0f, 0.0f, 16.0f, 16.0f, 1u});
  return batch;
}

void testBatchValidation() {
  FrameBatch batch = validBatch();
  std::string error;
  assert(validateFrameBatch(batch, &error));

  FrameBatch baseline = batch;
  batch.compositeItems[0].dstW = std::numeric_limits<float>::quiet_NaN();
  assert(!validateFrameBatch(batch, &error));
  assert(error == "frame-batch-surface-item-invalid");
  batch = baseline;
  batch.compositeVectorVertices.pop_back();
  assert(!validateFrameBatch(batch, &error));
  assert(error == "frame-batch-vector-triangle-list-invalid");
  batch = baseline;
  batch.compositeTextRuns[0].vertexCount = 0u;
  assert(!validateFrameBatch(batch, &error));
  assert(error == "frame-batch-text-run-invalid");
  batch = baseline;
  batch.compositeTextRuns[0].clipEnabled = 2u;
  assert(!validateFrameBatch(batch, &error));
  assert(error == "frame-batch-text-run-invalid");
  batch = baseline;
  batch.compositeTextRuns[0].firstVertex = 2u;
  assert(!validateFrameBatch(batch, &error));
  assert(error == "frame-batch-text-run-invalid");
}

void testPlanValidation() {
  std::string error;
  FramePassPlan invalid{};
  assert(!validateFramePassPlan(invalid, &error));
  invalid.count = 2u;
  invalid.passes[0] = {FramePassKind::RenderUiText, recordPass, nullptr};
  invalid.passes[1] = {FramePassKind::RenderUiText, recordPass, nullptr};
  assert(!validateFramePassPlan(invalid, &error));
  assert(error == "frame-pass-duplicate");
  invalid.passes[1] = {FramePassKind::DerivePlotData, recordPass, nullptr};
  assert(!validateFramePassPlan(invalid, &error));
  assert(error == "frame-pass-order-invalid");
  invalid.count = 1u;
  invalid.passes[0] = {FramePassKind::RenderUiText, nullptr, nullptr};
  assert(!validateFramePassPlan(invalid, &error));
  assert(error == "frame-pass-entry-invalid");
}

void testCreateIdentityValidation() {
  struct IdentityCase {
    bool zeroCompositor = false;
    bool zeroRuntimeContext = false;
    bool zeroDeviceRegistry = false;
  };
  const std::array<IdentityCase, 3> cases{{
      {true, false, false},
      {false, true, false},
      {false, false, true},
  }};
  for (const IdentityCase& identityCase : cases) {
    MockBackendState state{};
    state.zeroCreateCompositorId = identityCase.zeroCompositor;
    state.zeroCreateRuntimeContextId = identityCase.zeroRuntimeContext;
    state.zeroCreateDeviceRegistryId = identityCase.zeroDeviceRegistry;
    const FrameExecutorBackend backend = mockBackend(&state);
    FrameExecutor executor(&backend);
    std::string error;
    int nativeWindow = 0;
    assert(!executor.create(&nativeWindow, 640, 360, 2.0f, &error));
    assert(error == "frame-executor-create-backend-identity-invalid");
    assert(state.destroyCalls == (identityCase.zeroCompositor ? 0 : 1));
    assert(!executor.ready());
    assert(executor.compositor().compositorId == 0u);
    assert(executor.compositor().runtimeContextId == 0u);
    assert(executor.compositor().deviceRegistryId == 0u);
  }
}

void testBeginIdentityValidation() {
  MockBackendState state{};
  const FrameExecutorBackend backend = mockBackend(&state);
  FrameExecutor executor(&backend);
  std::string error;
  int nativeWindow = 0;
  assert(executor.create(&nativeWindow, 640, 360, 2.0f, &error));
  assert(executor.compositor().runtimeContextId == state.runtimeContextId);
  assert(executor.compositor().deviceRegistryId == state.deviceRegistryId);

  FrameBatch batch = validBatch();
  FrameExecutionStats stats{};
  state.mismatchBeginRuntimeContextId = true;
  assert(!executor.execute(plan(recordPass, &state), &batch, &stats, &error));
  assert(error == "frame-executor-begin-token-invalid");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Begin &&
         stats.failure == FrameFailure::InvariantViolation);
  assert(state.abandonCalls == 1 && !executor.transactionActive());

  state.mismatchBeginRuntimeContextId = false;
  state.mismatchBeginDeviceRegistryId = true;
  assert(!executor.execute(plan(recordPass, &state), &batch, &stats, &error));
  assert(error == "frame-executor-begin-token-invalid");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Begin &&
         stats.failure == FrameFailure::InvariantViolation);
  assert(state.abandonCalls == 2 && !executor.transactionActive());

  executor.destroy();
  assert(executor.compositor().compositorId == 0u);
  assert(executor.compositor().runtimeContextId == 0u);
  assert(executor.compositor().deviceRegistryId == 0u);
}

void testTransactionLifecycleAndFailureAtomicity() {
  {
    MockBackendState partialState{};
    partialState.failCreatePartial = true;
    const FrameExecutorBackend partialBackend = mockBackend(&partialState);
    FrameExecutor partialExecutor(&partialBackend);
    std::string partialError;
    int partialNativeWindow = 0;
    assert(!partialExecutor.create(&partialNativeWindow, 640, 360, 2.0f,
                                   &partialError));
    assert(partialState.destroyCalls == 1 && !partialExecutor.ready());
  }
  {
    MockBackendState partialState{};
    partialState.failBegin = true;
    partialState.failBeginPartial = true;
    const FrameExecutorBackend partialBackend = mockBackend(&partialState);
    FrameExecutor partialExecutor(&partialBackend);
    std::string partialError;
    int partialNativeWindow = 0;
    assert(partialExecutor.create(&partialNativeWindow, 640, 360, 2.0f,
                                  &partialError));
    FrameBatch partialBatch = validBatch();
    FrameExecutionStats partialStats{};
    assert(!partialExecutor.execute(plan(recordPass), &partialBatch,
                                    &partialStats, &partialError));
    assert(partialState.abandonCalls == 1 &&
           partialState.transientStatsCalls == 1 &&
           partialStats.failureStage == FrameExecutionStats::FailureStage::Begin &&
           partialStats.failure == FrameFailure::Unknown &&
           partialStats.transientMemory.available &&
           !partialExecutor.transactionActive());
    partialExecutor.destroy();
  }
  MockBackendState state{};
  const FrameExecutorBackend backend = mockBackend(&state);
  FrameExecutor executor(&backend);
  std::string error;
  int nativeWindow = 0;
  assert(executor.create(&nativeWindow, 640, 360, 2.0f, &error));
  assert(!executor.create(&nativeWindow, 640, 360, 2.0f, &error));
  assert(executor.resize(800, 450, 2.0f, &error));
  assert(executor.drain(10u, &error));
  ChromaspaceMetal::FrameCompletionStats completionStats{};
  assert(executor.completionStats(&completionStats));
  assert(completionStats.available && completionStats.submittedSerial == 8u &&
         completionStats.completedSerial == 8u &&
         completionStats.timedSubmissionCount == 8u &&
         completionStats.maximumGpuSeconds == 0.020 &&
         state.completionStatsCalls == 1);

  FrameBatch batch = validBatch();
  PassRecorder firstPass{&state, FramePassKind::ImportSourceUpdate};
  PassRecorder secondPass{&state, FramePassKind::RenderPlotSurfaces};
  PassRecorder thirdPass{&state, FramePassKind::RenderUiText};
  FramePassPlan good{};
  good.count = 3u;
  good.passes[0] = {firstPass.kind, recordSpecificPass, &firstPass};
  good.passes[1] = {secondPass.kind, recordSpecificPass, &secondPass};
  good.passes[2] = {thirdPass.kind, recordSpecificPass, &thirdPass};
  FrameExecutionStats stats{};
  assert(executor.execute(good, &batch, &stats, &error));
  assert(stats.begun && stats.submitted && !stats.abandoned &&
         stats.encodedPasses == 3u && stats.failure == FrameFailure::None);
  assert(stats.transientMemory.available &&
         stats.transientMemory.activeSubmissionCount == 2u &&
         stats.transientMemory.inFlightReservedBytes == 96u &&
         stats.transientMemory.peakInFlightReservedBytes == 192u &&
         stats.transientMemory.maxInFlightBytes == 768u &&
         stats.transientMemory.maxSubmissions == 3u &&
         state.transientStatsCalls == 1);
  assert(state.passKinds.size() == 3u &&
         state.passKinds[0] == FramePassKind::ImportSourceUpdate &&
         state.passKinds[1] == FramePassKind::RenderPlotSurfaces &&
         state.passKinds[2] == FramePassKind::RenderUiText);
  assert(state.beginCalls == 1 && state.submitCalls == 1 &&
         state.abandonCalls == 0 && !executor.transactionActive());

  state.failBegin = true;
  state.beginFailure = FrameFailure::DrawableUnavailable;
  const int statsCallsBeforeBeginFailure = state.transientStatsCalls;
  assert(!executor.execute(good, &batch, &stats, &error));
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Begin &&
         stats.failure == FrameFailure::DrawableUnavailable &&
         stats.transientMemory.available &&
         state.transientStatsCalls == statsCallsBeforeBeginFailure + 1);
  state.failBegin = false;
  state.beginFailure = FrameFailure::None;

  state.beginSuccessFailure = FrameFailure::DrawableUnavailable;
  assert(!executor.execute(good, &batch, &stats, &error));
  assert(error == "frame-begin-reported-failure-on-success");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Begin &&
         stats.failure == FrameFailure::InvariantViolation &&
         state.abandonCalls == 1 && !executor.transactionActive());
  state.beginSuccessFailure = FrameFailure::None;

  FramePassPlan fail = plan(failPass);
  stats = FrameExecutionStats{};
  const int statsCallsBeforePassFailure = state.transientStatsCalls;
  assert(!executor.execute(fail, &batch, &stats, &error));
  assert(stats.begun && !stats.submitted && stats.abandoned);
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Pass);
  assert(stats.failure == FrameFailure::EncodingFailure);
  assert(stats.transientMemory.available &&
         state.transientStatsCalls == statsCallsBeforePassFailure + 1);
  assert(state.beginCalls == 4 && state.submitCalls == 1 &&
         state.abandonCalls == 2 && !executor.transactionActive());

  assert(!executor.execute(plan(throwPass), &batch, &stats, &error));
  assert(error == "frame-pass-exception");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Pass);
  assert(stats.failure == FrameFailure::EncodingFailure);
  assert(state.abandonCalls == 3 && state.submitCalls == 1 &&
         !executor.transactionActive());

  assert(!executor.execute(plan(invalidPass), &batch, &stats, &error));
  assert(error == "frame-pass-output-invalid:frame-batch-vector-triangle-list-invalid");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::Pass);
  assert(stats.failure == FrameFailure::InvariantViolation);
  assert(state.abandonCalls == 4 && state.submitCalls == 1 &&
         !executor.transactionActive());
  batch = validBatch();

  state.failSubmit = true;
  state.submitFailure = FrameFailure::BackpressureTimeout;
  const int statsCallsBeforeSubmitFailure = state.transientStatsCalls;
  assert(!executor.execute(good, &batch, &stats, &error));
  assert(state.submitCalls == 2 && state.abandonCalls == 5 &&
         !executor.transactionActive());
  assert(stats.failureStage == FrameExecutionStats::FailureStage::FinalSubmit);
  assert(stats.failure == FrameFailure::BackpressureTimeout &&
         stats.transientMemory.available &&
         state.transientStatsCalls == statsCallsBeforeSubmitFailure + 1);
  state.failSubmit = false;
  state.submitFailure = FrameFailure::None;

  state.leaveSubmitToken = true;
  assert(!executor.execute(good, &batch, &stats, &error));
  assert(error == "frame-submit-token-not-consumed");
  assert(stats.failureStage == FrameExecutionStats::FailureStage::FinalSubmit);
  assert(state.abandonCalls == 6 && !executor.transactionActive());
  state.leaveSubmitToken = false;

  assert(executor.execute(plan(nestedPass, &executor), &batch, &stats, &error));
  assert(!executor.transactionActive());
  executor.destroy();
  assert(state.destroyCalls == 1);
}

void testTransientMetricsAreBestEffort() {
  {
    MockBackendState state{};
    FrameExecutorBackend backend = mockBackend(&state);
    backend.transientMemoryStats = nullptr;
    backend.completionStats = nullptr;
    FrameExecutor executor(&backend);
    int nativeWindow = 0;
    std::string error;
    assert(executor.create(&nativeWindow, 320, 180, 1.0f, &error));
    FrameBatch batch = validBatch();
    FrameExecutionStats stats{};
    assert(executor.execute(plan(recordPass, &state), &batch, &stats, &error));
    assert(stats.submitted && !stats.transientMemory.available &&
           state.transientStatsCalls == 0 && error.empty());
    ChromaspaceMetal::FrameCompletionStats completion{};
    assert(!executor.completionStats(&completion) && !completion.available &&
           state.completionStatsCalls == 0);
  }

  {
    MockBackendState state{};
    state.throwTransientStats = true;
    const FrameExecutorBackend backend = mockBackend(&state);
    FrameExecutor executor(&backend);
    int nativeWindow = 0;
    std::string error;
    assert(executor.create(&nativeWindow, 320, 180, 1.0f, &error));
    FrameBatch batch = validBatch();
    FrameExecutionStats stats{};
    assert(executor.execute(plan(recordPass, &state), &batch, &stats, &error));
    assert(stats.submitted && !stats.transientMemory.available &&
           state.transientStatsCalls == 1 && error.empty());

    state.failBegin = true;
    state.beginFailure = FrameFailure::DrawableUnavailable;
    assert(!executor.execute(plan(recordPass, &state), &batch, &stats, &error));
    assert(stats.failureStage == FrameExecutionStats::FailureStage::Begin &&
           stats.failure == FrameFailure::DrawableUnavailable &&
           !stats.transientMemory.available &&
           state.transientStatsCalls == 2 && error == "mock-begin-failed");
  }
}

}  // namespace

int main() {
  testBatchValidation();
  testPlanValidation();
  testCreateIdentityValidation();
  testBeginIdentityValidation();
  testTransactionLifecycleAndFailureAtomicity();
  testTransientMetricsAreBestEffort();
  return 0;
}
