#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ChromaspaceMetal.h"

namespace ChromaspaceMetalFrameExecutor {

// These limits are deliberately finite.  A malformed host frame must fail
// closed before it can acquire a drawable, command buffer, or transient
// staging allocation.
constexpr std::size_t kMaxFramePasses = 8u;
constexpr std::size_t kMaxSurfaceItems = 4096u;
constexpr std::size_t kMaxOverlayRects = 262144u;
constexpr std::size_t kMaxVectorVertices = 1048576u;
constexpr std::size_t kMaxTextVertices = 1048576u;
constexpr std::size_t kMaxTextRuns = 65536u;

using SurfaceCompositeItem = ChromaspaceMetal::SurfaceCompositeItem;
using OverlayRect = ChromaspaceMetal::FrameOverlayRect;
using VectorVertex = ChromaspaceMetal::FrameVectorVertex;
using TextVertex = ChromaspaceMetal::FrameTextVertex;
using TextRun = ChromaspaceMetal::FrameTextRun;
using FrameFailure = ChromaspaceMetalFrameFailure::Kind;

struct FrameBatch {
  std::vector<SurfaceCompositeItem> compositeItems;
  std::vector<OverlayRect> compositeOverlayRects;
  std::vector<VectorVertex> compositeVectorVertices;
  std::vector<TextVertex> compositeTextVertices;
  std::vector<TextRun> compositeTextRuns;
  std::array<float, 4> clearColor{{0.010f, 0.011f, 0.013f, 1.0f}};

  void clear() noexcept {
    compositeItems.clear();
    compositeOverlayRects.clear();
    compositeVectorVertices.clear();
    compositeTextVertices.clear();
    compositeTextRuns.clear();
  }
};

enum class FramePassKind : uint8_t {
  ImportSourceUpdate = 0,
  DerivePlotData,
  RenderPlotSurfaces,
  RenderUiText,
  Count,
};

struct FrameExecutionContext;
using FramePassEncoder = bool (*)(const FrameExecutionContext& context,
                                  void* userContext,
                                  std::string* error);

struct FramePass {
  FramePassKind kind = FramePassKind::ImportSourceUpdate;
  FramePassEncoder encoder = nullptr;
  void* userContext = nullptr;
};

struct FramePassPlan {
  std::array<FramePass, kMaxFramePasses> passes{};
  std::size_t count = 0u;
};

struct FrameExecutionContext {
  // The compositor ID and submission token are backend handles.  They carry
  // no application-model or packaging semantics in this contract.
  uint64_t compositorId = 0u;
  ChromaspaceMetal::FrameSubmission* submission = nullptr;
  FrameBatch* batch = nullptr;
};

using FrameCompositorState = ChromaspaceMetal::FrameCompositor;

struct FrameExecutionStats {
  bool begun = false;
  bool submitted = false;
  bool abandoned = false;
  std::size_t encodedPasses = 0u;
  enum class FailureStage : uint8_t {
    None = 0,
    Preflight,
    Begin,
    Pass,
    FinalSubmit,
  } failureStage = FailureStage::None;
  FrameFailure failure = FrameFailure::None;
  ChromaspaceMetal::FrameTransientMemoryStats transientMemory{};
};

// Platform backends implement this narrow callback table.  The portable
// executor owns ordering and transaction state; the Apple implementation is
// the only code that translates these callbacks to Metal/CAMetalLayer calls.
struct FrameExecutorBackend {
  void* context = nullptr;
  bool (*create)(void* context,
                 void* nativeWindow,
                 int drawableWidth,
                 int drawableHeight,
                 float contentsScale,
                 FrameCompositorState* outState,
                 std::string* error) noexcept = nullptr;
  // resize must leave backend state unchanged when it returns false.
  bool (*resize)(void* context,
                 uint64_t compositorId,
                 int drawableWidth,
                 int drawableHeight,
                 float contentsScale,
                 std::string* error) noexcept = nullptr;
  bool (*drain)(void* context,
                uint64_t compositorId,
                uint32_t timeoutMilliseconds,
                std::string* error) noexcept = nullptr;
  void (*destroy)(void* context, uint64_t compositorHandle) noexcept = nullptr;
  bool (*begin)(void* context,
                uint64_t compositorId,
                ChromaspaceMetal::FrameSubmission* outSubmission,
                std::string* error,
                FrameFailure* failure) noexcept = nullptr;
  bool (*submit)(void* context,
                 ChromaspaceMetal::FrameSubmission* submission,
                 const FrameBatch& batch,
                 std::string* error,
                 FrameFailure* failure) noexcept = nullptr;
  void (*abandon)(void* context,
                  ChromaspaceMetal::FrameSubmission* submission) noexcept = nullptr;
  // Optional best-effort diagnostics. Metrics failure must never affect a
  // frame result, so this callback is intentionally not part of backend
  // validity and may throw; the executor isolates it.
  bool (*transientMemoryStats)(
      void* context,
      uint64_t compositorId,
      ChromaspaceMetal::FrameTransientMemoryStats* outStats) = nullptr;
  bool (*completionStats)(
      void* context,
      uint64_t compositorId,
      ChromaspaceMetal::FrameCompletionStats* outStats) = nullptr;
};

// Returns the platform implementation when one is available.  The portable
// fallback deliberately reports unavailable rather than pretending to render.
const FrameExecutorBackend* defaultFrameExecutorBackend() noexcept;

bool validateFrameBatch(const FrameBatch& batch, std::string* error);
bool validateFramePassPlan(const FramePassPlan& plan, std::string* error);

class FrameExecutor final {
 public:
  explicit FrameExecutor(const FrameExecutorBackend* backend = nullptr) noexcept;
  ~FrameExecutor();

  FrameExecutor(const FrameExecutor&) = delete;
  FrameExecutor& operator=(const FrameExecutor&) = delete;

  bool create(void* nativeWindow,
              int drawableWidth,
              int drawableHeight,
              float contentsScale,
              std::string* error = nullptr);
  bool resize(int drawableWidth,
              int drawableHeight,
              float contentsScale,
              std::string* error = nullptr);
  bool drain(uint32_t timeoutMilliseconds,
             std::string* error = nullptr);
  bool completionStats(
      ChromaspaceMetal::FrameCompletionStats* outStats) const noexcept;
  void destroy() noexcept;

  bool execute(const FramePassPlan& plan,
               FrameBatch* batch,
               FrameExecutionStats* stats = nullptr,
               std::string* error = nullptr);

  bool ready() const noexcept { return compositor_.compositorId != 0u; }
  bool transactionActive() const noexcept { return transactionActive_; }
  const FrameCompositorState& compositor() const noexcept { return compositor_; }

 private:
  FrameExecutorBackend backend_{};
  FrameCompositorState compositor_{};
  ChromaspaceMetal::FrameSubmission submission_{};
  bool transactionActive_ = false;
};

}  // namespace ChromaspaceMetalFrameExecutor
