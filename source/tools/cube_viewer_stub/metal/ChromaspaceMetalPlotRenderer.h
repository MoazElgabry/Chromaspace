#pragma once

#include "ChromaspaceMetalDerivedCache.h"
#include "ChromaspaceMetalFrameExecutor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ChromaspaceMetalPlotRenderer {

constexpr std::size_t kMaximumPlotWindows = 64u;
constexpr std::size_t kMaximumCommandVectorVertices = 4096u;
constexpr std::size_t kMaximumFrameVectorVertices = 262144u;
constexpr int kMaximumResidentRasterPoints = 1048576;
constexpr std::size_t kMaximumRendererEvents = 64u;
constexpr int kMaximumPlotDimension = 16384;
constexpr float kMaximumPlotCoordinate = 10000000.0f;
constexpr std::size_t kDefaultResidentSurfaceBudgetBytes =
    512u * 1024u * 1024u;
constexpr std::size_t kDefaultTransientSurfaceBudgetBytes =
    768u * 1024u * 1024u;
constexpr std::size_t kDefaultResidentDerivedBudgetBytes =
    512u * 1024u * 1024u;
constexpr std::size_t kDefaultTransientDerivedBudgetBytes =
    768u * 1024u * 1024u;

struct ResidencyConfig {
  std::size_t maxResidentSurfaceBytes =
      kDefaultResidentSurfaceBudgetBytes;
  std::size_t maxTransientSurfaceBytes =
      kDefaultTransientSurfaceBudgetBytes;
  std::size_t maxResidentDerivedBytes =
      kDefaultResidentDerivedBudgetBytes;
  std::size_t maxTransientDerivedBytes =
      kDefaultTransientDerivedBudgetBytes;
};

enum class PlotKind : uint8_t {
  SourceSignal = 0,
  Histogram,
  Waveform,
  GlossField2D,
  GlossProjection3D,
  ResidentRaster,
  Scaffold,
};

enum class WindowStatus : uint8_t {
  Created = 0,
  Reused,
  Resized,
  Replaced,
  Encoded,
  Scaffolded,
  Failed,
  Unavailable,
};

struct PlotRect {
  float x = 0.0f;
  float y = 0.0f;
  float width = 1.0f;
  float height = 1.0f;
};

struct PlotContentKey {
  uint64_t sourceId = 0u;
  uint64_t deviceRegistryId = 0u;
  uint64_t senderGeneration = 0u;
  uint64_t sequence = 0u;
  uint32_t slotIndex = 0u;
  uint64_t slotGeneration = 0u;
  uint64_t contentHash = 0u;
  uint64_t contentRevision = 0u;
  uint64_t derivationHash = 0u;
  PlotKind kind = PlotKind::Scaffold;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;

  bool operator==(const PlotContentKey& other) const noexcept {
    return sourceId == other.sourceId &&
           deviceRegistryId == other.deviceRegistryId &&
           senderGeneration == other.senderGeneration &&
           sequence == other.sequence && slotIndex == other.slotIndex &&
           slotGeneration == other.slotGeneration &&
           contentHash == other.contentHash &&
           contentRevision == other.contentRevision &&
           derivationHash == other.derivationHash && kind == other.kind &&
           width == other.width && height == other.height &&
           pixelFormat == other.pixelFormat;
  }
};

struct PlotCommand {
  int windowId = 0;
  // Canonical ChromaspaceViewer plot-model ID. Renderer kind alone is not
  // sufficient qualification evidence because eight distinct models share
  // the resident-raster backend.
  int plotModel = 0;
  PlotKind kind = PlotKind::Scaffold;
  PlotRect destination{};
  int targetWidth = 1;
  int targetHeight = 1;
  int targetPixelFormat = 0;
  uint64_t viewRevision = 1u;
  // Revision of every semantic input that changes the rendered plot pixels.
  // It may differ from viewRevision when shared/global state (for example a
  // synchronized Source lasso) invalidates this window.
  uint64_t contentRevision = 1u;
  uint64_t glossDerivationHash = 0u;

  ChromaspaceMetal::RasterSourceRequest raster{};
  ChromaspaceMetal::HistogramSurfaceRequest histogram{};
  ChromaspaceMetal::WaveformSurfaceRequest waveform{};
  ChromaspaceMetal::RasterPointSurfaceRequest point{};
  ChromaspaceMetal::GlossFieldRequest glossField{};
  ChromaspaceMetal::GlossFieldSurfaceRequest glossFieldSurface{};
  ChromaspaceMetal::GlossProjectionSurfaceRequest glossProjectionSurface{};

  // Optional vector primitives are encoded after the primary plot pass. This
  // supports non-clearing guides/overlays on resident point plots as well as
  // a clearing scaffold for unavailable windows.
  std::size_t vectorVertexOffset = 0u;
  std::size_t vectorVertexCount = 0u;
  bool vectorClearBeforeDraw = true;
  std::array<float, 4> vectorClearColor{{0.02f, 0.025f, 0.035f, 1.0f}};
  std::string unavailableReason;
};

struct FrameRequest {
  std::array<PlotCommand, kMaximumPlotWindows> commands{};
  std::vector<ChromaspaceMetal::FrameVectorVertex> vectorVertexArena;
  std::size_t commandCount = 0u;
  uint64_t frameRevision = 0u;
  // Exactly one source is authoritative for a frame. Plot commands refer to
  // this source implicitly; they never carry copied source descriptors.
  bool hasResidentSource = false;
  ChromaspaceMetal::ImportedSourceTexture residentSource{};

  FrameRequest();
  bool append(const PlotCommand& command) noexcept;
  bool appendVectorVertices(
      const ChromaspaceMetal::FrameVectorVertex* vertices,
      std::size_t vertexCount,
      PlotCommand* command) noexcept;
  // Kept as a descriptive compatibility alias for callers that construct
  // unavailable-window scaffolds.
  bool appendScaffoldVertices(
      const ChromaspaceMetal::FrameVectorVertex* vertices,
      std::size_t vertexCount,
      PlotCommand* command) noexcept;
  void clear() noexcept {
    for (std::size_t index = 0; index < commandCount; ++index) {
      commands[index].unavailableReason.clear();
    }
    commandCount = 0u;
    vectorVertexArena.clear();
    frameRevision = 0u;
    hasResidentSource = false;
    residentSource.sourceId = 0u;
    residentSource.senderId.clear();
    residentSource.deviceRegistryId = 0u;
    residentSource.senderGeneration = 0u;
    residentSource.sequence = 0u;
    residentSource.slotIndex = 0u;
    residentSource.slotGeneration = 0u;
    residentSource.readyValue = 0u;
    residentSource.contentHash = 0u;
    residentSource.width = 0;
    residentSource.height = 0;
    residentSource.pixelFormat = 0;
    residentSource.bytesPerRow = 0u;
    residentSource.byteSize = 0u;
    residentSource.semantics.sourceX = 0;
    residentSource.semantics.sourceY = 0;
    residentSource.semantics.sourceWidth = 0u;
    residentSource.semantics.sourceHeight = 0u;
    residentSource.semantics.sampledX = 0;
    residentSource.semantics.sampledY = 0;
    residentSource.semantics.sampledWidth = 0u;
    residentSource.semantics.sampledHeight = 0u;
    residentSource.semantics.coverage =
        ChromaspaceSourceExchange::SourceCoverage::FullSource;
    residentSource.semantics.authoritative = false;
    residentSource.semantics.identityStripPresent = false;
    residentSource.semantics.identityCube = false;
    residentSource.semantics.identityRamp = false;
    residentSource.semantics.identityResolution = 0u;
    residentSource.semantics.identityBandHeight = 0u;
    residentSource.semantics.identityCubeY1 = 0;
    residentSource.semantics.identityCubeY2 = 0;
    residentSource.semantics.identityRampY1 = 0;
    residentSource.semantics.identityRampY2 = 0;
    residentSource.semantics.colorPrimaries.clear();
    residentSource.semantics.transferFunction.clear();
  }
};

struct WindowEvent {
  int windowId = 0;
  PlotKind kind = PlotKind::Scaffold;
  WindowStatus status = WindowStatus::Failed;
  uint32_t surfaceId = 0u;
  std::string reason;
};

struct RenderResult {
  bool frameSucceeded = false;
  std::size_t commandCount = 0u;
  std::size_t compositeItemCount = 0u;
  std::size_t createdSurfaceCount = 0u;
  std::size_t reusedSurfaceCount = 0u;
  std::size_t resizedSurfaceCount = 0u;
  std::size_t replacedSurfaceCount = 0u;
  std::size_t prunedSurfaceCount = 0u;
  std::size_t residentContentHitCount = 0u;
  std::size_t residentDerivedHitCount = 0u;
  std::size_t residentDerivedCandidateCount = 0u;
  std::size_t evictedDerivedCacheCount = 0u;
  std::size_t residentSurfaceBytes = 0u;
  std::size_t transientSurfaceBytes = 0u;
  std::size_t residentDerivedBytes = 0u;
  std::size_t transientDerivedBytes = 0u;
  std::array<WindowEvent, kMaximumRendererEvents> events{};
  std::size_t eventCount = 0u;

  void clear() noexcept {
    try {
      *this = RenderResult{};
    } catch (...) {
      commandCount = 0u;
      compositeItemCount = 0u;
      eventCount = 0u;
      frameSucceeded = false;
    }
  }
};

// Memory-pressure reclamation is deliberately a typed, render-thread-affine
// operation.  The level is an enum rather than a string so an invalid caller
// value can be rejected before any renderer state is inspected or changed.
enum class TrimLevel : uint8_t {
  DerivedOnly = 0,
  AllPlotResources = 1,
};

enum class TrimStatus : uint8_t {
  Accepted = 0,
  InvalidLevel,
  TransactionActive,
  RendererUnavailable,
  DerivedCacheResetFailed,
};

const char* trimStatusLabel(TrimStatus status) noexcept;

struct ResidencySnapshot {
  std::size_t surfaceCount = 0u;
  std::size_t surfaceBytes = 0u;
  std::size_t derivedCacheCount = 0u;
  std::size_t derivedCacheBytes = 0u;
};

struct TrimResult {
  TrimStatus status = TrimStatus::InvalidLevel;
  TrimLevel level = TrimLevel::DerivedOnly;
  ResidencySnapshot before{};
  ResidencySnapshot after{};
  std::size_t releasedSurfaceCount = 0u;
  std::size_t releasedSurfaceBytes = 0u;
  std::size_t releasedDerivedCacheCount = 0u;
  std::size_t releasedDerivedCacheBytes = 0u;

  bool accepted() const noexcept { return status == TrimStatus::Accepted; }
  bool succeeded() const noexcept { return accepted(); }
};

// This is the only platform seam. The coordinator validates requests and
// owns all lifetime decisions; the backend only translates one validated
// operation to the selected GPU implementation.
struct RendererBackend {
  void* context = nullptr;
  bool (*createSurface)(void* context,
                        uint64_t compositorId,
                        int width,
                        int height,
                        int pixelFormat,
                        ChromaspaceMetal::PlotSurface* outSurface,
                        std::string* error) noexcept = nullptr;
  void (*releaseSurface)(void* context,
                         uint64_t compositorId,
                         uint32_t surfaceId) noexcept = nullptr;
  bool (*encodeSourceSignal)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      uint64_t sourceId,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeHistogram)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::HistogramSurfaceRequest& request,
      uint64_t sourceId,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeWaveform)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::WaveformSurfaceRequest& request,
      uint64_t sourceId,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeResidentRasterCached)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      ChromaspaceMetal::ResidentDerivedCache* cache,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::RasterPointSurfaceRequest& request,
      uint64_t sourceId,
      uint64_t buildSerial,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeGlossField)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      ChromaspaceMetal::GlossFieldCache* cache,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::GlossFieldRequest& request,
      uint64_t sourceId,
      uint64_t buildSerial,
      std::string* error) noexcept = nullptr;
  bool (*encodeGlossFieldSurface)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      const ChromaspaceMetal::GlossFieldCache& cache,
      const ChromaspaceMetal::GlossFieldSurfaceRequest& request,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeGlossProjectionSurface)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      const ChromaspaceMetal::GlossFieldCache& cache,
      const ChromaspaceMetal::GlossProjectionSurfaceRequest& request,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeVectors)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      const ChromaspaceMetal::FrameVectorVertex* vertices,
      std::size_t vertexCount,
      bool clearBeforeDraw,
      const std::array<float, 4>& clearColor,
      std::string* error) noexcept = nullptr;
  ChromaspaceMetal::GlossFieldCacheState (*glossCacheState)(
      void* context,
      const ChromaspaceMetal::GlossFieldCache& cache) noexcept = nullptr;
  void (*releaseGlossCache)(
      void* context,
      ChromaspaceMetal::GlossFieldCache* cache) noexcept = nullptr;
  bool (*encodeHistogramCached)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      ChromaspaceMetal::ResidentDerivedCache* cache,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::HistogramSurfaceRequest& request,
      uint64_t sourceId,
      uint64_t buildSerial,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  bool (*encodeWaveformCached)(
      void* context,
      const ChromaspaceMetal::FrameSubmission& submission,
      ChromaspaceMetal::ResidentDerivedCache* cache,
      const ChromaspaceMetal::RasterSourceRequest& rasterRequest,
      const ChromaspaceMetal::WaveformSurfaceRequest& request,
      uint64_t sourceId,
      uint64_t buildSerial,
      uint32_t outputSurfaceId,
      int outputWidth,
      int outputHeight,
      int outputPixelFormat,
      std::string* error) noexcept = nullptr;
  ChromaspaceMetal::ResidentDerivedCacheState (*derivedCacheState)(
      void* context,
      const ChromaspaceMetal::ResidentDerivedCache& cache) noexcept = nullptr;
  void (*releaseDerivedCache)(
      void* context,
      ChromaspaceMetal::ResidentDerivedCache* cache) noexcept = nullptr;
};

const RendererBackend* defaultRendererBackend() noexcept;

const char* plotKindLabel(PlotKind kind) noexcept;
const char* statusLabel(WindowStatus status) noexcept;

bool validateResidentSource(
    const ChromaspaceMetal::ImportedSourceTexture& source,
    std::string* error = nullptr);
bool validateFrameRequest(const FrameRequest& request,
                          std::string* error = nullptr);

class PlotRenderer final {
 public:
  explicit PlotRenderer(
      const RendererBackend* backend = nullptr,
      const ResidencyConfig& residencyConfig = ResidencyConfig{}) noexcept;
  ~PlotRenderer();

  PlotRenderer(const PlotRenderer&) = delete;
  PlotRenderer& operator=(const PlotRenderer&) = delete;

  bool prepare(const FrameRequest& request,
               uint64_t compositorId,
               RenderResult* result = nullptr,
               std::string* error = nullptr);
  bool encodePrepared(
      const ChromaspaceMetalFrameExecutor::FrameExecutionContext& context,
      ChromaspaceMetalFrameExecutor::FrameBatch* batch,
      RenderResult* result = nullptr,
      std::string* error = nullptr);
  // Final-submit is the commit point. Passing false aborts the staged frame
  // and preserves every previously committed resource.
  bool finish(bool submitted,
              RenderResult* result = nullptr) noexcept;
  TrimResult trim(TrimLevel level) noexcept;
  void shutdown() noexcept;

  std::size_t resourceCount() const noexcept { return resourceCount_; }
  bool transactionActive() const noexcept { return transactionActive_; }
  ResidencySnapshot residencySnapshot() const noexcept;
  std::size_t glossCacheCount() const noexcept;
  std::size_t derivedCacheCount() const noexcept {
    return derivedCache_ ? derivedCache_->committedCount() : 0u;
  }
  bool hasResource(int windowId) const noexcept;
  std::size_t residentSurfaceBytes() const noexcept {
    return residentSurfaceBytes_;
  }
  std::size_t transientSurfaceBytes() const noexcept {
    return residentSurfaceBytes_ + pendingOwnedSurfaceBytes_;
  }
  std::size_t residentDerivedBytes() const noexcept {
    return derivedCache_
               ? static_cast<std::size_t>(derivedCache_->residentByteSize())
               : 0u;
  }
  std::size_t transientDerivedBytes() const noexcept {
    return derivedCache_
               ? static_cast<std::size_t>(derivedCache_->transientByteSize())
               : 0u;
  }

 private:
  struct WindowResource {
    int windowId = 0;
    ChromaspaceMetal::PlotSurface surface{};
    PlotKind kind = PlotKind::Scaffold;
    PlotContentKey contentKey{};
    bool hasContentKey = false;
  };

  struct PendingResource {
    int windowId = 0;
    bool hadCommitted = false;
    bool ownsSurface = false;
    WindowResource previous{};
    WindowResource candidate{};
    ChromaspaceMetalDerivedCache::AcquireKind derivedAcquireKind =
        ChromaspaceMetalDerivedCache::AcquireKind::Failure;
    std::size_t derivedCacheIndex =
        ChromaspaceMetalDerivedCache::kInvalidIndex;
  };

  static bool validBackend(const RendererBackend& backend) noexcept;
  static bool sourceRequired(PlotKind kind) noexcept;
  static bool validPlotKind(PlotKind kind) noexcept;

  void addEvent(RenderResult* result,
                int windowId,
                PlotKind kind,
                WindowStatus status,
                uint32_t surfaceId,
                const std::string& reason) noexcept;
  void releaseResource(WindowResource* resource) noexcept;
  bool stageSurface(uint64_t compositorId,
                    const PlotCommand& command,
                    PendingResource* pending,
                    WindowStatus* outStatus,
                    RenderResult* result,
                    std::string* error);
  PendingResource* findPendingResource(int windowId) noexcept;
  const PendingResource* findPendingResource(int windowId) const noexcept;
  WindowResource* findResource(int windowId) noexcept;
  const WindowResource* findResource(int windowId) const noexcept;
  bool encodeCommand(const PlotCommand& command,
                     const ChromaspaceMetalFrameExecutor::FrameExecutionContext& context,
                     WindowResource* resource,
                     RenderResult* result,
                     std::string* error);
  void releaseDerivedCaches(
      const ChromaspaceMetalDerivedCache::ReleaseList& releases) noexcept;

  RendererBackend backend_{};
  ResidencyConfig residencyConfig_{};
  std::unique_ptr<ChromaspaceMetalDerivedCache::DerivedCache> derivedCache_;
  bool residencyConfigValid_ = true;
  uint64_t compositorId_ = 0u;
  uint64_t derivedBuildSerial_ = 0u;
  uint64_t derivedUseEpoch_ = 0u;
  bool transactionActive_ = false;
  std::array<PendingResource, kMaximumPlotWindows> pendingResources_{};
  std::size_t pendingResourceCount_ = 0u;
  FrameRequest pendingRequest_{};
  std::array<WindowResource, kMaximumPlotWindows> resources_{};
  std::size_t resourceCount_ = 0u;
  std::size_t residentSurfaceBytes_ = 0u;
  std::size_t pendingOwnedSurfaceBytes_ = 0u;
};

}  // namespace ChromaspaceMetalPlotRenderer
