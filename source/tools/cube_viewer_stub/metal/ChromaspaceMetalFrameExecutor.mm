#include "ChromaspaceMetalFrameExecutor.h"

namespace ChromaspaceMetalFrameExecutor {
namespace {

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "metal-frame-executor-error";
  } catch (...) {
  }
}

bool backendCreate(void*,
                   void* nativeWindow,
                   int drawableWidth,
                   int drawableHeight,
                   float contentsScale,
                   FrameCompositorState* outState,
                   std::string* error) noexcept {
  try {
    return ChromaspaceMetal::createFrameCompositor(
        nativeWindow, drawableWidth, drawableHeight, contentsScale, outState,
        error);
  } catch (...) {
    setErrorNoThrow(error, "metal-frame-executor-create-exception");
    return false;
  }
}

bool backendResize(void*,
                   uint64_t compositorId,
                   int drawableWidth,
                   int drawableHeight,
                   float contentsScale,
                   std::string* error) noexcept {
  try {
    return ChromaspaceMetal::resizeFrameCompositor(
        compositorId, drawableWidth, drawableHeight, contentsScale, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-frame-executor-resize-exception");
    return false;
  }
}

bool backendDrain(void*,
                  uint64_t compositorId,
                  uint32_t timeoutMilliseconds,
                  std::string* error) noexcept {
  try {
    return ChromaspaceMetal::drainFrameCompositor(compositorId,
                                                  timeoutMilliseconds, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-frame-executor-drain-exception");
    return false;
  }
}

void backendDestroy(void*, uint64_t compositorId) noexcept {
  try {
    ChromaspaceMetal::releaseFrameCompositor(compositorId);
  } catch (...) {
  }
}

bool backendBegin(void*,
                  uint64_t compositorId,
                  ChromaspaceMetal::FrameSubmission* outSubmission,
                  std::string* error,
                  FrameFailure* failure) noexcept {
  try {
    return ChromaspaceMetal::beginFrameSubmission(compositorId, outSubmission,
                                                  error, failure);
  } catch (...) {
    setErrorNoThrow(error, "metal-frame-executor-begin-exception");
    if (failure) *failure = FrameFailure::Unknown;
    return false;
  }
}

bool backendSubmit(void*,
                   ChromaspaceMetal::FrameSubmission* submission,
                   const FrameBatch& batch,
                   std::string* error,
                   FrameFailure* failure) noexcept {
  try {
    return ChromaspaceMetal::submitFrameSubmissionSurfacesOverlayRectsAndText(
        submission,
        batch.compositeItems.empty() ? nullptr : batch.compositeItems.data(),
        batch.compositeItems.size(),
        batch.compositeOverlayRects.empty()
            ? nullptr
            : batch.compositeOverlayRects.data(),
        batch.compositeOverlayRects.size(),
        batch.compositeVectorVertices.empty()
            ? nullptr
            : batch.compositeVectorVertices.data(),
        batch.compositeVectorVertices.size(),
        batch.compositeTextVertices.empty() ? nullptr
                                             : batch.compositeTextVertices.data(),
        batch.compositeTextVertices.size(),
        batch.compositeTextRuns.empty() ? nullptr : batch.compositeTextRuns.data(),
        batch.compositeTextRuns.size(),
        batch.clearColor[0], batch.clearColor[1], batch.clearColor[2],
        batch.clearColor[3], error, failure);
  } catch (...) {
    setErrorNoThrow(error, "metal-frame-executor-submit-exception");
    if (failure) *failure = FrameFailure::Unknown;
    return false;
  }
}

void backendAbandon(void*, ChromaspaceMetal::FrameSubmission* submission) noexcept {
  try {
    ChromaspaceMetal::abandonFrameSubmission(submission);
  } catch (...) {
    if (submission) *submission = ChromaspaceMetal::FrameSubmission{};
  }
}

bool backendTransientMemoryStats(
    void*,
    uint64_t compositorId,
    ChromaspaceMetal::FrameTransientMemoryStats* outStats) noexcept {
  try {
    return ChromaspaceMetal::frameTransientMemoryStats(
        compositorId, outStats, nullptr);
  } catch (...) {
    if (outStats) *outStats = ChromaspaceMetal::FrameTransientMemoryStats{};
    return false;
  }
}

bool backendCompletionStats(
    void*,
    uint64_t compositorId,
    ChromaspaceMetal::FrameCompletionStats* outStats) noexcept {
  try {
    return ChromaspaceMetal::frameCompletionStats(
        compositorId, outStats, nullptr);
  } catch (...) {
    if (outStats) *outStats = ChromaspaceMetal::FrameCompletionStats{};
    return false;
  }
}

const FrameExecutorBackend kMetalBackend{
    nullptr,
    backendCreate,
    backendResize,
    backendDrain,
    backendDestroy,
    backendBegin,
    backendSubmit,
    backendAbandon,
    backendTransientMemoryStats,
    backendCompletionStats};

}  // namespace

const FrameExecutorBackend* defaultFrameExecutorBackend() noexcept {
  return &kMetalBackend;
}

}  // namespace ChromaspaceMetalFrameExecutor
