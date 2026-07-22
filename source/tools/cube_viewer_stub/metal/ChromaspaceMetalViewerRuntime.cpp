#include "ChromaspaceMetalViewerRuntime.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ChromaspaceMetalViewerRuntime {
namespace {

using FrameFailure = ChromaspaceMetalFrameFailure::Kind;
using SurfaceVisibility = ChromaspaceFrameRecoveryPolicy::SurfaceVisibility;

void setError(std::string* error, const char* message) {
  if (error) *error = message != nullptr ? message : "metal-viewer-runtime-error";
}

void setError(std::string* error, const std::string& message) {
  if (error) *error = message;
}

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "metal-viewer-runtime-error";
  } catch (...) {
  }
}

bool finite(float value) noexcept { return std::isfinite(value); }

struct PlotEncodeContext {
  ChromaspaceMetalPlotRenderer::PlotRenderer* renderer = nullptr;
  ChromaspaceMetalPlotRenderer::RenderResult* result = nullptr;
};

bool encodePlotPass(
    const ChromaspaceMetalFrameExecutor::FrameExecutionContext& context,
    void* userContext,
    std::string* error) {
  auto* pass = static_cast<PlotEncodeContext*>(userContext);
  if (!pass || !pass->renderer || !pass->result || !context.batch) {
    setError(error, "metal-viewer-runtime-plot-pass-context-invalid");
    return false;
  }
  return pass->renderer->encodePrepared(context, context.batch, pass->result,
                                        error);
}

#if !defined(__APPLE__)
bool unavailableAtlasCreate(void*,
                            uint64_t,
                            int,
                            int,
                            const unsigned char*,
                            std::size_t,
                            ChromaspaceMetal::FrameTextAtlas*,
                            std::string* error) noexcept {
  setErrorNoThrow(error, "metal-viewer-runtime-atlas-backend-unavailable");
  return false;
}

void unavailableAtlasRelease(void*, uint64_t, uint64_t) noexcept {}

const RuntimeResourceBackend kUnavailableResourceBackend{
    nullptr, unavailableAtlasCreate, unavailableAtlasRelease};
#endif

}  // namespace

#if !defined(__APPLE__)
const RuntimeResourceBackend* defaultRuntimeResourceBackend() noexcept {
  return &kUnavailableResourceBackend;
}
#endif

const char* outcomeLabel(OutcomeKind kind) noexcept {
  switch (kind) {
    case OutcomeKind::Presented: return "presented";
    case OutcomeKind::RetryLater: return "retry-later";
    case OutcomeKind::SuspendUntilVisible: return "suspend-until-visible";
    case OutcomeKind::RuntimeRecreated: return "runtime-recreated";
    case OutcomeKind::TerminalFailure: return "terminal-failure";
    case OutcomeKind::ViewportUpdated: return "viewport-updated";
  }
  return "terminal-failure";
}

const char* memoryPressureStatusLabel(MemoryPressureStatus status) noexcept {
  switch (status) {
    case MemoryPressureStatus::Accepted: return "accepted";
    case MemoryPressureStatus::InvalidLevel: return "invalid-level";
    case MemoryPressureStatus::RuntimeNotReady: return "runtime-not-ready";
    case MemoryPressureStatus::RecreationPending: return "recreation-pending";
    case MemoryPressureStatus::TransactionActive: return "transaction-active";
    case MemoryPressureStatus::RendererRejected: return "renderer-rejected";
  }
  return "unknown";
}

Runtime::Runtime(
    const ChromaspaceMetalFrameExecutor::FrameExecutorBackend* executorBackend,
    const ChromaspaceMetalPlotRenderer::RendererBackend* rendererBackend,
    const RuntimeResourceBackend* resourceBackend,
    const ChromaspaceFrameRecoveryPolicy::Config& recoveryConfig) noexcept
    : executor_(executorBackend),
      renderer_(rendererBackend),
      recovery_(recoveryConfig) {
  const RuntimeResourceBackend* selected =
      resourceBackend != nullptr ? resourceBackend : defaultRuntimeResourceBackend();
  if (selected != nullptr) resourceBackend_ = *selected;
  try {
    workingRequest_ = std::make_unique<ChromaspaceMetalPlotRenderer::FrameRequest>();
  } catch (...) {
    workingRequest_.reset();
  }
}

Runtime::~Runtime() { shutdown(); }

bool Runtime::validViewport(const Viewport& viewport) noexcept {
  return viewport.drawableWidth > 0 && viewport.drawableHeight > 0 &&
         finite(viewport.contentsScale) && viewport.contentsScale > 0.0f;
}

bool Runtime::copyAtlas(const CpuTextAtlasPayload& atlas, std::string* error) {
  if (error) error->clear();
  if ((atlas.width == 0) != (atlas.height == 0) || atlas.width < 0 ||
      atlas.height < 0 || atlas.width > kMaximumAtlasDimension ||
      atlas.height > kMaximumAtlasDimension) {
    setError(error, "metal-viewer-runtime-atlas-dimensions-invalid");
    return false;
  }
  if (atlas.width == 0 && atlas.height == 0) {
    if (atlas.byteCount != 0u || atlas.pixels != nullptr) {
      setError(error, "metal-viewer-runtime-empty-atlas-payload-invalid");
      return false;
    }
    atlasWidth_ = 0;
    atlasHeight_ = 0;
    atlasPixels_.clear();
    return true;
  }
  const std::size_t width = static_cast<std::size_t>(atlas.width);
  const std::size_t height = static_cast<std::size_t>(atlas.height);
  if (height > std::numeric_limits<std::size_t>::max() / width) {
    setError(error, "metal-viewer-runtime-atlas-size-overflow");
    return false;
  }
  const std::size_t expected = width * height;
  if (expected == 0u || expected > kMaximumAtlasBytes ||
      atlas.byteCount != expected || atlas.pixels == nullptr) {
    setError(error, "metal-viewer-runtime-atlas-payload-invalid");
    return false;
  }
  try {
    atlasPixels_.assign(atlas.pixels, atlas.pixels + expected);
  } catch (...) {
    setError(error, "metal-viewer-runtime-atlas-copy-failed");
    atlasPixels_.clear();
    return false;
  }
  atlasWidth_ = atlas.width;
  atlasHeight_ = atlas.height;
  return true;
}

bool Runtime::createResources(std::string* error,
                              uint64_t previousRuntimeContextId) {
  if (error) error->clear();
  if (nativeWindow_ == nullptr || !validViewport(latestViewport_)) {
    setError(error, "metal-viewer-runtime-create-configuration-invalid");
    return false;
  }
  if (executor_.ready() || executor_.transactionActive()) {
    setError(error, "metal-viewer-runtime-already-created");
    return false;
  }
  ChromaspaceMetal::FrameCompositor compositor{};
  std::string createError;
  if (!executor_.create(nativeWindow_, latestViewport_.drawableWidth,
                        latestViewport_.drawableHeight,
                        latestViewport_.contentsScale, &createError)) {
    if (error) *error = createError.empty() ? "metal-viewer-runtime-compositor-create-failed"
                                             : createError;
    return false;
  }
  compositor = executor_.compositor();

  // A recreated compositor must belong to a fresh backend/runtime owner.  A
  // stable device registry ID is allowed, but reusing the old context ID would
  // let stale opaque handles look valid after teardown.  Reject before any
  // dependent atlas is created and tear down the new compositor immediately.
  if (previousRuntimeContextId != 0u &&
      compositor.runtimeContextId == previousRuntimeContextId) {
    executor_.destroy();
    setError(error, "metal-viewer-runtime-recreate-context-id-reused");
    return false;
  }

  textAtlas_ = ChromaspaceMetal::FrameTextAtlas{};
  if (atlasWidth_ > 0) {
    if (resourceBackend_.createTextAtlas == nullptr ||
        resourceBackend_.releaseTextAtlas == nullptr) {
      executor_.destroy();
      setError(error, "metal-viewer-runtime-atlas-backend-invalid");
      return false;
    }
    std::string atlasError;
    bool atlasOk = false;
    try {
      atlasOk = resourceBackend_.createTextAtlas(
          resourceBackend_.context, compositor.compositorId, atlasWidth_,
          atlasHeight_, atlasPixels_.data(), atlasPixels_.size(), &textAtlas_,
          &atlasError);
    } catch (...) {
      atlasOk = false;
      atlasError = "metal-viewer-runtime-atlas-create-exception";
    }
    if (!atlasOk || textAtlas_.atlasId == 0u ||
        textAtlas_.width != atlasWidth_ || textAtlas_.height != atlasHeight_) {
      if (textAtlas_.atlasId != 0u && resourceBackend_.releaseTextAtlas != nullptr) {
        resourceBackend_.releaseTextAtlas(resourceBackend_.context,
                                          compositor.compositorId,
                                          textAtlas_.atlasId);
      }
      textAtlas_ = ChromaspaceMetal::FrameTextAtlas{};
      executor_.destroy();
      if (error) {
        *error = atlasError.empty() ? "metal-viewer-runtime-atlas-create-failed"
                                    : atlasError;
      }
      return false;
    }
  }
  if (generation_ == std::numeric_limits<uint64_t>::max()) {
    if (textAtlas_.atlasId != 0u && resourceBackend_.releaseTextAtlas != nullptr) {
      resourceBackend_.releaseTextAtlas(resourceBackend_.context,
                                        compositor.compositorId,
                                        textAtlas_.atlasId);
    }
    textAtlas_ = ChromaspaceMetal::FrameTextAtlas{};
    executor_.destroy();
    setError(error, "metal-viewer-runtime-generation-overflow");
    return false;
  }
  ++generation_;
  return true;
}

bool Runtime::destroyResources(std::string* error) noexcept {
  bool ok = true;
  std::string drainError;
  if (executor_.ready()) {
    try {
      if (!executor_.drain(2000u, &drainError)) ok = false;
    } catch (...) {
      ok = false;
      try {
        drainError = "metal-viewer-runtime-drain-exception";
      } catch (...) {
      }
    }
  }
  renderer_.shutdown();
  if (textAtlas_.atlasId != 0u && resourceBackend_.releaseTextAtlas != nullptr) {
    try {
      resourceBackend_.releaseTextAtlas(resourceBackend_.context,
                                        executor_.compositor().compositorId,
                                        textAtlas_.atlasId);
    } catch (...) {
      ok = false;
      if (drainError.empty()) {
        try {
          drainError = "metal-viewer-runtime-atlas-release-exception";
        } catch (...) {
        }
      }
    }
  }
  textAtlas_ = ChromaspaceMetal::FrameTextAtlas{};
  executor_.destroy();
  if (!ok && error != nullptr) {
    try {
      *error = drainError.empty() ? "metal-viewer-runtime-drain-failed" : drainError;
    } catch (...) {
    }
  }
  return ok;
}

bool Runtime::create(void* nativeWindow,
                     const Viewport& viewport,
                     const CpuTextAtlasPayload& atlas,
                     std::string* error) {
  if (error) error->clear();
  if (hasConfiguration_ || ready() || nativeWindow == nullptr ||
      !validViewport(viewport)) {
    setError(error, "metal-viewer-runtime-create-invalid");
    return false;
  }
  // Rendering requires the retained request scratch object.  The constructor
  // is noexcept and may have been unable to allocate it; reject creation
  // before any atlas copy or GPU resource is acquired so ready() never
  // advertises a runtime that cannot render.
  if (!workingRequest_) {
    setError(error, "metal-viewer-runtime-scratch-unavailable");
    return false;
  }
  if (!copyAtlas(atlas, error)) return false;
  nativeWindow_ = nativeWindow;
  latestViewport_ = viewport;
  hasConfiguration_ = true;
  recreationPending_ = false;
  if (!createResources(error)) {
    executor_.destroy();
    textAtlas_ = ChromaspaceMetal::FrameTextAtlas{};
    nativeWindow_ = nullptr;
    latestViewport_ = Viewport{};
    atlasWidth_ = 0;
    atlasHeight_ = 0;
    atlasPixels_.clear();
    hasConfiguration_ = false;
    return false;
  }
  return true;
}

Outcome Runtime::outcomeFromDecision(
    ChromaspaceFrameRecoveryPolicy::Decision decision,
    FrameFailure failure,
    const std::string& diagnostic) const {
  Outcome result{};
  result.waitMilliseconds = decision.backoffMilliseconds;
  result.failure = failure;
  result.runtimeGeneration = generation_;
  result.diagnostic = diagnostic;
  switch (decision.action) {
    case ChromaspaceFrameRecoveryPolicy::Action::RetryLater:
      result.kind = OutcomeKind::RetryLater;
      break;
    case ChromaspaceFrameRecoveryPolicy::Action::SuspendUntilVisible:
      result.kind = OutcomeKind::SuspendUntilVisible;
      break;
    case ChromaspaceFrameRecoveryPolicy::Action::RecreateRuntime:
      result.kind = OutcomeKind::RetryLater;
      break;
    case ChromaspaceFrameRecoveryPolicy::Action::Continue:
      result.kind = OutcomeKind::Presented;
      break;
    case ChromaspaceFrameRecoveryPolicy::Action::Terminate:
      result.kind = OutcomeKind::TerminalFailure;
      break;
  }
  return result;
}

Outcome Runtime::attemptPendingRecreation(std::string* error) {
  if (error) error->clear();
  if (!recreationPending_) {
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvalidState;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-recreation-not-pending";
    if (error) *error = result.diagnostic;
    return result;
  }
  std::string recreateError;
  if (recreate(&recreateError)) {
    recreationPending_ = false;
    recovery_.onRuntimeRecreationResult(true);
    Outcome result{};
    result.kind = OutcomeKind::RuntimeRecreated;
    result.failure = FrameFailure::None;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-recreated";
    if (error) *error = result.diagnostic;
    return result;
  }
  const auto decision = recovery_.onRuntimeRecreationResult(false);
  if (decision.action == ChromaspaceFrameRecoveryPolicy::Action::RecreateRuntime) {
    recreationPending_ = true;
    Outcome result = outcomeFromDecision(decision, FrameFailure::MetalContextUnavailable,
                                         recreateError.empty()
                                             ? "metal-viewer-runtime-recreation-failed"
                                             : recreateError);
    if (error) *error = result.diagnostic;
    return result;
  }
  recreationPending_ = false;
  Outcome result = outcomeFromDecision(
      decision, FrameFailure::MetalContextUnavailable,
      recreateError.empty() ? "metal-viewer-runtime-recreation-failed"
                            : recreateError);
  if (error) *error = result.diagnostic;
  return result;
}

Outcome Runtime::handleFailure(FrameFailure failure,
                               SurfaceVisibility visibility,
                               const std::string& diagnostic,
                               bool tryRecreate) {
  const auto decision = recovery_.onFailure(failure, visibility);
  if (decision.action == ChromaspaceFrameRecoveryPolicy::Action::RecreateRuntime &&
      tryRecreate) {
    recreationPending_ = true;
    return attemptPendingRecreation(nullptr);
  }
  return outcomeFromDecision(decision, failure, diagnostic);
}

Outcome Runtime::resize(const Viewport& viewport,
                        SurfaceVisibility visibility,
                        std::string* error) {
  if (error) error->clear();
  latestViewport_ = viewport;
  if (!hasConfiguration_ || !ready() || !validViewport(viewport)) {
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvalidState;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-resize-invalid";
    if (error) *error = result.diagnostic;
    return result;
  }
  if (recreationPending_) return attemptPendingRecreation(error);
  std::string resizeError;
  bool resized = false;
  try {
    resized = executor_.resize(viewport.drawableWidth, viewport.drawableHeight,
                               viewport.contentsScale, &resizeError);
  } catch (...) {
    resizeError = "metal-viewer-runtime-resize-exception";
    resized = false;
  }
  if (!resized) {
    Outcome result = handleFailure(
        FrameFailure::CompositorUnavailable, visibility,
        resizeError.empty() ? "metal-viewer-runtime-resize-failed" : resizeError,
        true);
    if (error) *error = result.diagnostic;
    return result;
  }
  Outcome result{};
  result.kind = OutcomeKind::ViewportUpdated;
  result.runtimeGeneration = generation_;
  result.diagnostic = "metal-viewer-runtime-resized";
  return result;
}

MemoryPressureResult Runtime::handleMemoryPressure(
    MemoryPressureLevel level) noexcept {
  MemoryPressureResult result{};
  result.level = level;

  // Validate the enum representation before inspecting runtime state.  A
  // malformed notification is therefore a pure typed rejection, even if the
  // runtime is currently rendering or awaiting recreation.
  const uint8_t levelValue = static_cast<uint8_t>(level);
  if (levelValue > static_cast<uint8_t>(MemoryPressureLevel::Critical)) {
    result.status = MemoryPressureStatus::InvalidLevel;
    return result;
  }
  if (recreationPending_) {
    result.status = MemoryPressureStatus::RecreationPending;
    return result;
  }
  if (!hasConfiguration_ || !ready()) {
    result.status = MemoryPressureStatus::RuntimeNotReady;
    return result;
  }
  if (executor_.transactionActive() || renderer_.transactionActive()) {
    result.status = MemoryPressureStatus::TransactionActive;
    return result;
  }

  if (level == MemoryPressureLevel::Normal) {
    // Normal pressure is an accepted no-op.  Still return a complete
    // renderer snapshot so callers can retain a typed accounting baseline.
    result.status = MemoryPressureStatus::Accepted;
    result.rendererTrim.status =
        ChromaspaceMetalPlotRenderer::TrimStatus::Accepted;
    result.rendererTrim.level =
        ChromaspaceMetalPlotRenderer::TrimLevel::DerivedOnly;
    result.rendererTrim.before = renderer_.residencySnapshot();
    result.rendererTrim.after = result.rendererTrim.before;
    return result;
  }

  const auto trimLevel =
      level == MemoryPressureLevel::Warning
          ? ChromaspaceMetalPlotRenderer::TrimLevel::DerivedOnly
          : ChromaspaceMetalPlotRenderer::TrimLevel::AllPlotResources;
  result.rendererTrim = renderer_.trim(trimLevel);
  if (!result.rendererTrim.accepted()) {
    result.status = MemoryPressureStatus::RendererRejected;
    return result;
  }
  result.status = MemoryPressureStatus::Accepted;
  // Only a successful critical/all-resource trim invalidates final plot
  // pixels.  Warning and Normal preserve the current presentation.
  result.redrawRequired = level == MemoryPressureLevel::Critical;
  return result;
}

Outcome Runtime::render(
    const ChromaspaceMetalPlotRenderer::FrameRequest& request,
    const ChromaspaceMetalFrameExecutor::FrameBatch& baselineBatch,
    SurfaceVisibility visibility,
    ChromaspaceMetalPlotRenderer::RenderResult* plotResult,
    ChromaspaceMetalFrameExecutor::FrameExecutionStats* executionStats,
    std::string* error) {
  if (error) error->clear();
  if (plotResult) plotResult->clear();
  if (executionStats) *executionStats = ChromaspaceMetalFrameExecutor::FrameExecutionStats{};
  if (!hasConfiguration_ || !ready() || recreationPending_) {
    if (recreationPending_) return attemptPendingRecreation(error);
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvalidState;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-render-invalid";
    if (error) *error = result.diagnostic;
    return result;
  }

  workingPlotResult_.clear();
  if (!workingRequest_) {
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvariantViolation;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-scratch-unavailable";
    if (error) *error = result.diagnostic;
    return result;
  }
  try {
    *workingRequest_ = request;
    workingBatch_ = baselineBatch;
    const std::size_t requiredItems =
        workingBatch_.compositeItems.size() + workingRequest_->commandCount;
    if (requiredItems > ChromaspaceMetalFrameExecutor::kMaxSurfaceItems) {
      Outcome result{};
      result.kind = OutcomeKind::TerminalFailure;
      result.failure = FrameFailure::InvariantViolation;
      result.runtimeGeneration = generation_;
      result.diagnostic = "metal-viewer-runtime-composite-item-limit";
      if (error) *error = result.diagnostic;
      return result;
    }
    workingBatch_.compositeItems.reserve(requiredItems);
  } catch (...) {
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvariantViolation;
    result.runtimeGeneration = generation_;
    result.diagnostic = "metal-viewer-runtime-input-copy-failed";
    if (error) *error = result.diagnostic;
    return result;
  }

  std::string prepareError;
  if (!renderer_.prepare(*workingRequest_, compositorId(), &workingPlotResult_,
                         &prepareError)) {
    const std::string diagnostic =
        prepareError.empty() ? "metal-viewer-runtime-plot-prepare-failed"
                             : prepareError;
    if (plotResult) *plotResult = workingPlotResult_;
    Outcome result{};
    result.kind = OutcomeKind::TerminalFailure;
    result.failure = FrameFailure::InvariantViolation;
    result.runtimeGeneration = generation_;
    result.diagnostic = diagnostic;
    if (error) *error = diagnostic;
    return result;
  }

  PlotEncodeContext passContext{&renderer_, &workingPlotResult_};
  ChromaspaceMetalFrameExecutor::FramePassPlan passPlan{};
  passPlan.count = 1u;
  passPlan.passes[0] = {
      ChromaspaceMetalFrameExecutor::FramePassKind::RenderPlotSurfaces,
      encodePlotPass, &passContext};
  ChromaspaceMetalFrameExecutor::FrameExecutionStats localExecutionStats{};
  std::string executeError;
  bool executeOk = false;
  try {
    executeOk = executor_.execute(passPlan, &workingBatch_, &localExecutionStats,
                                  &executeError);
  } catch (...) {
    executeError = "metal-viewer-runtime-execute-exception";
    executeOk = false;
    localExecutionStats.failureStage =
        ChromaspaceMetalFrameExecutor::FrameExecutionStats::FailureStage::FinalSubmit;
    localExecutionStats.failure = FrameFailure::Unknown;
  }

  // A prepared plot transaction must always be rolled back when execute or
  // submit fails.  finish(false) also releases newly staged surfaces/caches.
  const bool submitted = executeOk && localExecutionStats.submitted;
  const bool committed = renderer_.finish(submitted, &workingPlotResult_);
  if (plotResult) *plotResult = workingPlotResult_;
  if (executionStats) *executionStats = localExecutionStats;

  if (!executeOk || !submitted || !committed) {
    FrameFailure failure = localExecutionStats.failure;
    if (failure == FrameFailure::None) {
      failure = executeOk && submitted ? FrameFailure::InvariantViolation
                                      : FrameFailure::Unknown;
    }
    const std::string diagnostic =
        executeError.empty()
            ? (committed ? "metal-viewer-runtime-frame-submit-failed"
                         : "metal-viewer-runtime-plot-rollback-failed")
            : executeError;
    Outcome result = handleFailure(failure, visibility, diagnostic, true);
    if (error) *error = result.diagnostic;
    return result;
  }

  recovery_.onPresentedFrame();
  Outcome result{};
  result.kind = OutcomeKind::Presented;
  result.failure = FrameFailure::None;
  result.runtimeGeneration = generation_;
  result.diagnostic = "metal-viewer-runtime-presented";
  return result;
}

bool Runtime::recreate(std::string* error) {
  if (error) error->clear();
  if (!hasConfiguration_ || nativeWindow_ == nullptr ||
      !validViewport(latestViewport_)) {
    setError(error, "metal-viewer-runtime-recreate-invalid");
    return false;
  }
  const uint64_t previousRuntimeContextId = runtimeContextId();
  std::string destroyError;
  const bool drained = destroyResources(&destroyError);
  std::string createError;
  if (!createResources(&createError, previousRuntimeContextId)) {
    if (error) {
      if (!createError.empty()) *error = createError;
      else if (!destroyError.empty()) *error = destroyError;
      else *error = "metal-viewer-runtime-recreate-failed";
    }
    return false;
  }
  // A drain warning is retained for diagnostics, but a newly created
  // compositor/atlas is a successful bounded recreation.
  if (!drained && error) *error = destroyError;
  return true;
}

bool Runtime::drain(uint32_t timeoutMilliseconds, std::string* error) {
  if (error) error->clear();
  if (!ready() || recreationPending_) {
    setError(error, "metal-viewer-runtime-drain-invalid");
    return false;
  }
  try {
    return executor_.drain(timeoutMilliseconds, error);
  } catch (...) {
    setError(error, "metal-viewer-runtime-drain-exception");
    return false;
  }
}

bool Runtime::completionStats(
    ChromaspaceMetal::FrameCompletionStats* outStats) const noexcept {
  if (recreationPending_) {
    if (outStats) *outStats = ChromaspaceMetal::FrameCompletionStats{};
    return false;
  }
  return executor_.completionStats(outStats);
}

void Runtime::shutdown() noexcept {
  if (!hasConfiguration_ && !ready() && textAtlas_.atlasId == 0u) return;
  (void)destroyResources(nullptr);
  nativeWindow_ = nullptr;
  latestViewport_ = Viewport{};
  atlasWidth_ = 0;
  atlasHeight_ = 0;
  atlasPixels_.clear();
  if (workingRequest_) workingRequest_->clear();
  workingBatch_.clear();
  workingPlotResult_.clear();
  hasConfiguration_ = false;
  recreationPending_ = false;
  recovery_.reset();
}

}  // namespace ChromaspaceMetalViewerRuntime
