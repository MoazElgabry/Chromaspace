#include "ChromaspaceMetalFrameExecutor.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ChromaspaceMetalFrameExecutor {
namespace {

void setError(std::string* error, const char* message) {
  if (error) *error = message != nullptr ? message : "frame-executor-error";
}

void setError(std::string* error, const std::string& message) {
  if (error) *error = message;
}

// Backend callbacks are noexcept by contract.  Diagnostics must not turn a
// recoverable backend failure into process termination when the caller's
// allocator is exhausted.
void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "frame-executor-error";
  } catch (...) {
  }
}

bool finite(float value) noexcept { return std::isfinite(value); }

bool finiteColor(float r, float g, float b, float a) noexcept {
  return finite(r) && finite(g) && finite(b) && finite(a) && r >= 0.0f &&
         r <= 1.0f && g >= 0.0f && g <= 1.0f && b >= 0.0f && b <= 1.0f &&
         a >= 0.0f && a <= 1.0f;
}

bool finiteRect(float x, float y, float w, float h) noexcept {
  return finite(x) && finite(y) && finite(w) && finite(h) && w > 0.0f &&
         h > 0.0f;
}

bool validSurfaceItem(const SurfaceCompositeItem& item) noexcept {
  return item.surfaceId != 0u && item.surfaceWidth > 0 &&
         item.surfaceHeight > 0 &&
         (item.surfacePixelFormat == 0 || item.surfacePixelFormat == 1) &&
         finiteRect(item.dstX, item.dstY, item.dstW, item.dstH) &&
         finite(item.opacity) && item.opacity >= 0.0f && item.opacity <= 1.0f;
}

bool validOverlayRect(const OverlayRect& rect) noexcept {
  return finiteRect(rect.x, rect.y, rect.w, rect.h) &&
         finiteColor(rect.r, rect.g, rect.b, rect.a);
}

bool validVectorVertex(const VectorVertex& vertex) noexcept {
  return finite(vertex.x) && finite(vertex.y) &&
         finiteColor(vertex.r, vertex.g, vertex.b, vertex.a);
}

bool validTextVertex(const TextVertex& vertex) noexcept {
  return finite(vertex.x) && finite(vertex.y) && finite(vertex.u) &&
         finite(vertex.v);
}

bool validTextRun(const TextRun& run,
                 std::size_t textVertexCount) noexcept {
  if (run.atlasId == 0u || !finiteColor(run.r, run.g, run.b, run.a) ||
      run.vertexCount == 0u || (run.vertexCount % 3u) != 0u ||
      !finite(run.clipX) || !finite(run.clipY) || !finite(run.clipW) ||
      !finite(run.clipH) || run.clipW < 0.0f || run.clipH < 0.0f ||
      run.clipEnabled > 1u ||
      (run.clipEnabled != 0u && (run.clipW <= 0.0f || run.clipH <= 0.0f))) {
    return false;
  }
  const std::size_t first = static_cast<std::size_t>(run.firstVertex);
  const std::size_t count = static_cast<std::size_t>(run.vertexCount);
  return first <= textVertexCount && count <= textVertexCount - first;
}

bool validBackend(const FrameExecutorBackend& backend) noexcept {
  return backend.create != nullptr && backend.resize != nullptr &&
         backend.drain != nullptr && backend.destroy != nullptr &&
         backend.begin != nullptr && backend.submit != nullptr &&
         backend.abandon != nullptr;
}

bool hasCompleteIdentity(const FrameCompositorState& compositor) noexcept {
  return compositor.compositorId != 0u &&
         compositor.runtimeContextId != 0u &&
         compositor.deviceRegistryId != 0u;
}

bool hasAnyIdentity(const ChromaspaceMetal::FrameSubmission& submission) noexcept {
  return submission.submissionId != 0u || submission.compositorId != 0u ||
         submission.runtimeContextId != 0u ||
         submission.deviceRegistryId != 0u;
}

#if !defined(CHROMASPACE_METAL_EXTERNAL_DEFAULT_BACKENDS)
bool unavailableCreate(void*,
                       void*,
                       int,
                       int,
                       float,
                       FrameCompositorState*,
                       std::string* error) noexcept {
  setErrorNoThrow(error, "metal-frame-executor-backend-unavailable");
  return false;
}

bool unavailableResize(void*,
                       uint64_t,
                       int,
                       int,
                       float,
                       std::string* error) noexcept {
  setErrorNoThrow(error, "metal-frame-executor-backend-unavailable");
  return false;
}

bool unavailableDrain(void*, uint64_t, uint32_t, std::string* error) noexcept {
  setErrorNoThrow(error, "metal-frame-executor-backend-unavailable");
  return false;
}

void unavailableDestroy(void*, uint64_t) noexcept {}

bool unavailableBegin(void*,
                      uint64_t,
                      ChromaspaceMetal::FrameSubmission*,
                      std::string* error,
                      FrameFailure* failure) noexcept {
  setErrorNoThrow(error, "metal-frame-executor-backend-unavailable");
  if (failure) *failure = FrameFailure::MetalContextUnavailable;
  return false;
}

bool unavailableSubmit(void*,
                       ChromaspaceMetal::FrameSubmission*,
                       const FrameBatch&,
                       std::string* error,
                       FrameFailure* failure) noexcept {
  setErrorNoThrow(error, "metal-frame-executor-backend-unavailable");
  if (failure) *failure = FrameFailure::MetalContextUnavailable;
  return false;
}

void unavailableAbandon(void*, ChromaspaceMetal::FrameSubmission*) noexcept {}

const FrameExecutorBackend kUnavailableBackend{
    nullptr,
    unavailableCreate,
    unavailableResize,
    unavailableDrain,
    unavailableDestroy,
    unavailableBegin,
    unavailableSubmit,
    unavailableAbandon};
#endif

}  // namespace

#if !defined(CHROMASPACE_METAL_EXTERNAL_DEFAULT_BACKENDS)
const FrameExecutorBackend* defaultFrameExecutorBackend() noexcept {
  return &kUnavailableBackend;
}
#endif

bool validateFrameBatch(const FrameBatch& batch, std::string* error) {
  if (error) error->clear();
  if (batch.compositeItems.size() > kMaxSurfaceItems ||
      batch.compositeOverlayRects.size() > kMaxOverlayRects ||
      batch.compositeVectorVertices.size() > kMaxVectorVertices ||
      batch.compositeTextVertices.size() > kMaxTextVertices ||
      batch.compositeTextRuns.size() > kMaxTextRuns) {
    setError(error, "frame-batch-count-limit");
    return false;
  }
  if ((batch.compositeVectorVertices.size() % 3u) != 0u) {
    setError(error, "frame-batch-vector-triangle-list-invalid");
    return false;
  }
  if (!finiteColor(batch.clearColor[0], batch.clearColor[1],
                   batch.clearColor[2], batch.clearColor[3])) {
    setError(error, "frame-batch-clear-color-invalid");
    return false;
  }
  for (const SurfaceCompositeItem& item : batch.compositeItems) {
    if (!validSurfaceItem(item)) {
      setError(error, "frame-batch-surface-item-invalid");
      return false;
    }
  }
  for (const OverlayRect& rect : batch.compositeOverlayRects) {
    if (!validOverlayRect(rect)) {
      setError(error, "frame-batch-overlay-rect-invalid");
      return false;
    }
  }
  for (const VectorVertex& vertex : batch.compositeVectorVertices) {
    if (!validVectorVertex(vertex)) {
      setError(error, "frame-batch-vector-vertex-invalid");
      return false;
    }
  }
  for (const TextVertex& vertex : batch.compositeTextVertices) {
    if (!validTextVertex(vertex)) {
      setError(error, "frame-batch-text-vertex-invalid");
      return false;
    }
  }
  for (const TextRun& run : batch.compositeTextRuns) {
    if (!validTextRun(run, batch.compositeTextVertices.size())) {
      setError(error, "frame-batch-text-run-invalid");
      return false;
    }
  }
  return true;
}

bool validateFramePassPlan(const FramePassPlan& plan, std::string* error) {
  if (error) error->clear();
  if (plan.count == 0u || plan.count > kMaxFramePasses) {
    setError(error, "frame-pass-count-invalid");
    return false;
  }
  uint32_t seen = 0u;
  int previous = -1;
  for (std::size_t index = 0; index < plan.count; ++index) {
    const FramePass& pass = plan.passes[index];
    const int kind = static_cast<int>(pass.kind);
    if (kind < 0 || kind >= static_cast<int>(FramePassKind::Count) ||
        pass.encoder == nullptr) {
      setError(error, "frame-pass-entry-invalid");
      return false;
    }
    const uint32_t bit = 1u << static_cast<uint32_t>(kind);
    if ((seen & bit) != 0u) {
      setError(error, "frame-pass-duplicate");
      return false;
    }
    if (kind <= previous) {
      setError(error, "frame-pass-order-invalid");
      return false;
    }
    seen |= bit;
    previous = kind;
  }
  return true;
}

FrameExecutor::FrameExecutor(const FrameExecutorBackend* backend) noexcept {
  const FrameExecutorBackend* selected =
      backend != nullptr ? backend : defaultFrameExecutorBackend();
  if (selected != nullptr) backend_ = *selected;
}

FrameExecutor::~FrameExecutor() { destroy(); }

bool FrameExecutor::create(void* nativeWindow,
                           int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           std::string* error) {
  if (error) error->clear();
  if (ready() || transactionActive_) {
    setError(error, "frame-executor-already-created");
    return false;
  }
  if (!validBackend(backend_) || nativeWindow == nullptr || drawableWidth <= 0 ||
      drawableHeight <= 0 || !finite(contentsScale) || contentsScale <= 0.0f) {
    setError(error, "frame-executor-create-invalid");
    return false;
  }
  FrameCompositorState next{};
  if (!backend_.create(backend_.context, nativeWindow, drawableWidth,
                       drawableHeight, contentsScale, &next, error)) {
    if (next.compositorId != 0u) {
      backend_.destroy(backend_.context, next.compositorId);
    }
    return false;
  }
  if (!hasCompleteIdentity(next)) {
    if (next.compositorId != 0u && backend_.destroy != nullptr) {
      backend_.destroy(backend_.context, next.compositorId);
    }
    setError(error, "frame-executor-create-backend-identity-invalid");
    return false;
  }
  if (next.drawableWidth <= 0 || next.drawableHeight <= 0 ||
      !finite(next.contentsScale) || next.contentsScale <= 0.0f) {
    if (next.compositorId != 0u && backend_.destroy != nullptr) {
      backend_.destroy(backend_.context, next.compositorId);
    }
    setError(error, "frame-executor-create-backend-state-invalid");
    return false;
  }
  compositor_ = next;
  return true;
}

bool FrameExecutor::resize(int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           std::string* error) {
  if (error) error->clear();
  if (!ready() || transactionActive_ || drawableWidth <= 0 ||
      drawableHeight <= 0 || !finite(contentsScale) || contentsScale <= 0.0f) {
    setError(error, "frame-executor-resize-invalid");
    return false;
  }
  if (!backend_.resize(backend_.context, compositor_.compositorId,
                       drawableWidth, drawableHeight, contentsScale, error)) {
    return false;
  }
  compositor_.drawableWidth = drawableWidth;
  compositor_.drawableHeight = drawableHeight;
  compositor_.contentsScale = contentsScale;
  return true;
}

bool FrameExecutor::drain(uint32_t timeoutMilliseconds,
                          std::string* error) {
  if (error) error->clear();
  if (!ready() || transactionActive_ || timeoutMilliseconds == 0u) {
    setError(error, "frame-executor-drain-invalid");
    return false;
  }
  return backend_.drain(backend_.context, compositor_.compositorId,
                        timeoutMilliseconds, error);
}

bool FrameExecutor::completionStats(
    ChromaspaceMetal::FrameCompletionStats* outStats) const noexcept {
  if (outStats) *outStats = ChromaspaceMetal::FrameCompletionStats{};
  if (outStats == nullptr || !ready() ||
      backend_.completionStats == nullptr) {
    return false;
  }
  bool sampled = false;
  try {
    sampled = backend_.completionStats(
        backend_.context, compositor_.compositorId, outStats);
  } catch (...) {
    sampled = false;
  }
  if (!sampled || !outStats->available) {
    *outStats = ChromaspaceMetal::FrameCompletionStats{};
    return false;
  }
  return true;
}

void FrameExecutor::destroy() noexcept {
  if (transactionActive_) {
    if (backend_.abandon != nullptr) {
      backend_.abandon(backend_.context, &submission_);
    }
    submission_ = ChromaspaceMetal::FrameSubmission{};
    transactionActive_ = false;
  }
  if (compositor_.compositorId != 0u && backend_.destroy != nullptr) {
    backend_.destroy(backend_.context, compositor_.compositorId);
  }
  compositor_ = ChromaspaceMetal::FrameCompositor{};
}

bool FrameExecutor::execute(const FramePassPlan& plan,
                            FrameBatch* batch,
                            FrameExecutionStats* stats,
                            std::string* error) {
  if (error) error->clear();
  if (stats) *stats = FrameExecutionStats{};
  auto sampleTransientMemory = [&]() noexcept {
    if (stats == nullptr || !ready() ||
        backend_.transientMemoryStats == nullptr) {
      return;
    }
    ChromaspaceMetal::FrameTransientMemoryStats sampled{};
    bool sampledOk = false;
    try {
      sampledOk = backend_.transientMemoryStats(
          backend_.context, compositor_.compositorId, &sampled);
    } catch (...) {
      sampledOk = false;
    }
    stats->transientMemory =
        sampledOk ? sampled
                  : ChromaspaceMetal::FrameTransientMemoryStats{};
    stats->transientMemory.available =
        sampledOk && sampled.available;
  };
  auto finish = [&](bool result) noexcept {
    sampleTransientMemory();
    return result;
  };
  if (!ready() || transactionActive_ || batch == nullptr) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Preflight;
    if (stats) stats->failure = FrameFailure::InvalidState;
    setError(error, "frame-executor-execute-invalid-state");
    return finish(false);
  }
  if (!validateFramePassPlan(plan, error)) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Preflight;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    return finish(false);
  }
  if (!validateFrameBatch(*batch, error)) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Preflight;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    return finish(false);
  }
  FrameFailure backendFailure = FrameFailure::None;
  if (!backend_.begin(backend_.context, compositor_.compositorId, &submission_,
                      error, &backendFailure)) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Begin;
    if (stats) {
      stats->failure = backendFailure == FrameFailure::None
                           ? FrameFailure::Unknown
                           : backendFailure;
    }
    if (hasAnyIdentity(submission_)) {
      backend_.abandon(backend_.context, &submission_);
    }
    submission_ = ChromaspaceMetal::FrameSubmission{};
    return finish(false);
  }
  if (backendFailure != FrameFailure::None) {
    if (stats) {
      stats->failureStage = FrameExecutionStats::FailureStage::Begin;
      stats->failure = FrameFailure::InvariantViolation;
    }
    setError(error, "frame-begin-reported-failure-on-success");
    if (hasAnyIdentity(submission_)) {
      backend_.abandon(backend_.context, &submission_);
    }
    submission_ = ChromaspaceMetal::FrameSubmission{};
    return finish(false);
  }
  if (submission_.submissionId == 0u ||
      submission_.compositorId != compositor_.compositorId ||
      submission_.runtimeContextId != compositor_.runtimeContextId ||
      submission_.deviceRegistryId != compositor_.deviceRegistryId) {
    // A successful begin that returns any mismatched identity is a malformed
    // token.  Always route it through the backend abandon seam before
    // clearing the portable token, including the all-zero case.
    backend_.abandon(backend_.context, &submission_);
    submission_ = ChromaspaceMetal::FrameSubmission{};
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Begin;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    setError(error, "frame-executor-begin-token-invalid");
    return finish(false);
  }
  transactionActive_ = true;
  if (stats) stats->begun = true;
  auto fail = [&](const char* prefix) noexcept {
    if (backend_.abandon != nullptr) {
      backend_.abandon(backend_.context, &submission_);
    }
    submission_ = ChromaspaceMetal::FrameSubmission{};
    transactionActive_ = false;
    if (stats) stats->abandoned = true;
    if (error && prefix != nullptr) {
      try {
        const std::string detail = *error;
        try {
          *error = prefix;
          if (!detail.empty()) *error += ":" + detail;
        } catch (...) {
        }
      } catch (...) {
      }
    }
    return finish(false);
  };

  FrameExecutionContext context{};
  context.compositorId = compositor_.compositorId;
  context.submission = &submission_;
  context.batch = batch;
  for (std::size_t index = 0; index < plan.count; ++index) {
    const FramePass& pass = plan.passes[index];
    bool passOk = false;
    try {
      passOk = pass.encoder(context, pass.userContext, error);
    } catch (...) {
      if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Pass;
      if (stats) stats->failure = FrameFailure::EncodingFailure;
      return fail("frame-pass-exception");
    }
    if (!passOk) {
      if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Pass;
      if (stats) stats->failure = FrameFailure::EncodingFailure;
      return fail("frame-pass-failed");
    }
    bool outputValid = false;
    try {
      outputValid = validateFrameBatch(*batch, error);
    } catch (...) {
      if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Pass;
      if (stats) stats->failure = FrameFailure::InvariantViolation;
      return fail("frame-pass-validation-exception");
    }
    if (!outputValid) {
      if (stats) stats->failureStage = FrameExecutionStats::FailureStage::Pass;
      if (stats) stats->failure = FrameFailure::InvariantViolation;
      return fail("frame-pass-output-invalid");
    }
    if (stats) ++stats->encodedPasses;
  }
  bool finalBatchValid = false;
  try {
    finalBatchValid = validateFrameBatch(*batch, error);
  } catch (...) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::FinalSubmit;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    return fail("frame-submit-validation-exception");
  }
  if (!finalBatchValid) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::FinalSubmit;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    return fail("frame-submit-batch-invalid");
  }
  backendFailure = FrameFailure::None;
  if (!backend_.submit(backend_.context, &submission_, *batch, error,
                       &backendFailure)) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::FinalSubmit;
    if (stats) {
      stats->failure = backendFailure == FrameFailure::None
                           ? FrameFailure::Unknown
                           : backendFailure;
    }
    return fail("frame-submit-failed");
  }
  if (backendFailure != FrameFailure::None) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::FinalSubmit;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    setError(error, "frame-submit-reported-failure-on-success");
    return fail("frame-submit-contract-invalid");
  }
  if (hasAnyIdentity(submission_)) {
    if (stats) stats->failureStage = FrameExecutionStats::FailureStage::FinalSubmit;
    if (stats) stats->failure = FrameFailure::InvariantViolation;
    return fail("frame-submit-token-not-consumed");
  }
  submission_ = ChromaspaceMetal::FrameSubmission{};
  transactionActive_ = false;
  if (stats) stats->submitted = true;
  return finish(true);
}

}  // namespace ChromaspaceMetalFrameExecutor
