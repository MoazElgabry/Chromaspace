#pragma once

#include "ChromaspaceFrameRecoveryPolicy.h"
#include "ChromaspaceMetalFrameExecutor.h"
#include "ChromaspaceMetalPlotRenderer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ChromaspaceMetalViewerRuntime {

// Runtime-owned copies are deliberately bounded.  The caller may keep its
// shaping buffer alive only for the duration of create(); recreation uses this
// retained copy and never asks the caller to rebuild platform resources.
constexpr int kMaximumAtlasDimension = 4096;
constexpr std::size_t kMaximumAtlasBytes = 64u * 1024u * 1024u;

struct CpuTextAtlasPayload {
  int width = 0;
  int height = 0;
  const unsigned char* pixels = nullptr;
  std::size_t byteCount = 0u;
};

struct Viewport {
  int drawableWidth = 1;
  int drawableHeight = 1;
  float contentsScale = 1.0f;
};

// This is the only resource callback table owned by the coordinator.  The
// production Apple adapter is implemented in the adjacent .mm file; tests
// inject a portable mock.  No platform/windowing types cross this seam.
struct RuntimeResourceBackend {
  void* context = nullptr;
  bool (*createTextAtlas)(void* context,
                          uint64_t compositorId,
                          int width,
                          int height,
                          const unsigned char* alphaPixels,
                          std::size_t byteCount,
                          ChromaspaceMetal::FrameTextAtlas* outAtlas,
                          std::string* error) noexcept = nullptr;
  void (*releaseTextAtlas)(void* context,
                          uint64_t compositorId,
                          uint64_t atlasId) noexcept = nullptr;
};

const RuntimeResourceBackend* defaultRuntimeResourceBackend() noexcept;

enum class OutcomeKind : uint8_t {
  Presented = 0,
  RetryLater,
  SuspendUntilVisible,
  RuntimeRecreated,
  TerminalFailure,
  ViewportUpdated,
};

struct Outcome {
  OutcomeKind kind = OutcomeKind::TerminalFailure;
  uint32_t waitMilliseconds = 0u;
  ChromaspaceMetalFrameFailure::Kind failure =
      ChromaspaceMetalFrameFailure::Kind::None;
  uint64_t runtimeGeneration = 0u;
  std::string diagnostic;

  bool presented() const noexcept { return kind == OutcomeKind::Presented; }
  bool terminal() const noexcept {
    return kind == OutcomeKind::TerminalFailure;
  }
};

const char* outcomeLabel(OutcomeKind kind) noexcept;

enum class MemoryPressureLevel : uint8_t {
  Normal = 0,
  Warning = 1,
  Critical = 2,
};

enum class MemoryPressureStatus : uint8_t {
  Accepted = 0,
  InvalidLevel,
  RuntimeNotReady,
  RecreationPending,
  TransactionActive,
  RendererRejected,
};

const char* memoryPressureStatusLabel(MemoryPressureStatus status) noexcept;

struct MemoryPressureResult {
  MemoryPressureStatus status = MemoryPressureStatus::InvalidLevel;
  MemoryPressureLevel level = MemoryPressureLevel::Normal;
  ChromaspaceMetalPlotRenderer::TrimResult rendererTrim{};
  bool redrawRequired = false;

  bool accepted() const noexcept { return status == MemoryPressureStatus::Accepted; }
  bool succeeded() const noexcept { return accepted(); }
};

class Runtime final {
 public:
  // Runtime is deliberately single-owner and render-thread affine. Callers
  // must serialize lifecycle, resize, drain, and render operations on the
  // window/render thread; the coordinator does not add locks around native
  // platform/GPU backend state.
  Runtime(const ChromaspaceMetalFrameExecutor::FrameExecutorBackend*
              executorBackend = nullptr,
          const ChromaspaceMetalPlotRenderer::RendererBackend*
              rendererBackend = nullptr,
          const RuntimeResourceBackend* resourceBackend = nullptr,
          const ChromaspaceFrameRecoveryPolicy::Config& recoveryConfig =
              ChromaspaceFrameRecoveryPolicy::Config{}) noexcept;
  ~Runtime();

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  // Initial creation is intentionally a bool operation: there is no prior
  // frame for recovery to retry and no caller frame to rebuild yet.
  bool create(void* nativeWindow,
              const Viewport& viewport,
              const CpuTextAtlasPayload& atlas,
              std::string* error = nullptr);

  // Resize keeps the requested viewport even when the backend rejects it, so
  // a bounded runtime recreation can rebuild the latest drawable dimensions.
  Outcome resize(const Viewport& viewport,
                 ChromaspaceFrameRecoveryPolicy::SurfaceVisibility visibility,
                 std::string* error = nullptr);

  // The coordinator copies both inputs before PlotRenderer::prepare() and
  // FrameExecutor::execute().  A retry therefore starts from the caller's
  // baseline batch and can never duplicate plot composite items.
  Outcome render(const ChromaspaceMetalPlotRenderer::FrameRequest& request,
                 const ChromaspaceMetalFrameExecutor::FrameBatch& baselineBatch,
                 ChromaspaceFrameRecoveryPolicy::SurfaceVisibility visibility,
                 ChromaspaceMetalPlotRenderer::RenderResult* plotResult =
                     nullptr,
                  ChromaspaceMetalFrameExecutor::FrameExecutionStats* executionStats =
                      nullptr,
                  std::string* error = nullptr);

  // Memory pressure is handled on the same serialized render thread as
  // render/resize.  Normal is an accepted no-op; Warning drops only derived
  // caches; Critical drops all committed plot resources while preserving the
  // compositor, text atlas, resident source session, and runtime generation.
  MemoryPressureResult handleMemoryPressure(
      MemoryPressureLevel level) noexcept;

  bool drain(uint32_t timeoutMilliseconds, std::string* error = nullptr);
  bool completionStats(
      ChromaspaceMetal::FrameCompletionStats* outStats) const noexcept;
  void shutdown() noexcept;

  bool ready() const noexcept { return executor_.ready(); }
  bool transactionActive() const noexcept { return executor_.transactionActive(); }
  uint64_t generation() const noexcept { return generation_; }
  uint64_t textAtlasId() const noexcept { return textAtlas_.atlasId; }
  uint64_t compositorId() const noexcept {
    return executor_.compositor().compositorId;
  }
  uint64_t runtimeContextId() const noexcept {
    return executor_.compositor().runtimeContextId;
  }
  uint64_t deviceRegistryId() const noexcept {
    return executor_.compositor().deviceRegistryId;
  }
  const ChromaspaceMetal::FrameCompositor& compositor() const noexcept {
    return executor_.compositor();
  }
  const Viewport& latestViewport() const noexcept { return latestViewport_; }
  std::size_t plotResourceCount() const noexcept {
    return renderer_.resourceCount();
  }

 private:
  static bool validViewport(const Viewport& viewport) noexcept;
  bool copyAtlas(const CpuTextAtlasPayload& atlas, std::string* error);
  bool createResources(std::string* error,
                       uint64_t previousRuntimeContextId = 0u);
  // Recreate is reached only through the typed recovery policy. Keeping this
  // private prevents a caller from bypassing the bounded recovery budget.
  bool recreate(std::string* error = nullptr);
  bool destroyResources(std::string* error) noexcept;
  Outcome outcomeFromDecision(
      ChromaspaceFrameRecoveryPolicy::Decision decision,
      ChromaspaceMetalFrameFailure::Kind failure,
      const std::string& diagnostic) const;
  Outcome handleFailure(ChromaspaceMetalFrameFailure::Kind failure,
                        ChromaspaceFrameRecoveryPolicy::SurfaceVisibility
                            visibility,
                        const std::string& diagnostic,
                        bool tryRecreate);
  Outcome attemptPendingRecreation(std::string* error);

  ChromaspaceMetalFrameExecutor::FrameExecutor executor_;
  ChromaspaceMetalPlotRenderer::PlotRenderer renderer_;
  RuntimeResourceBackend resourceBackend_{};
  ChromaspaceFrameRecoveryPolicy::FrameRecoveryPolicy recovery_;
  void* nativeWindow_ = nullptr;
  Viewport latestViewport_{};
  int atlasWidth_ = 0;
  int atlasHeight_ = 0;
  std::vector<unsigned char> atlasPixels_;
  ChromaspaceMetal::FrameTextAtlas textAtlas_{};
  // Retained scratch copies reuse capacities across frames while remaining
  // isolated from caller-owned requests/batches.
  std::unique_ptr<ChromaspaceMetalPlotRenderer::FrameRequest> workingRequest_;
  ChromaspaceMetalFrameExecutor::FrameBatch workingBatch_{};
  ChromaspaceMetalPlotRenderer::RenderResult workingPlotResult_{};
  uint64_t generation_ = 0u;
  bool hasConfiguration_ = false;
  bool recreationPending_ = false;
};

}  // namespace ChromaspaceMetalViewerRuntime
