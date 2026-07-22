#include "ChromaspaceMetalPlotRenderer.h"

#include <cassert>
#include <string>
#include <vector>

namespace {

using namespace ChromaspaceMetalPlotRenderer;
using ChromaspaceMetal::FrameSubmission;

struct MockState {
  uint32_t nextSurface = 100u;
  uint64_t nextCache = 500u;
  int createCalls = 0;
  int releaseCalls = 0;
  int sourceCalls = 0;
  int rasterCalls = 0;
  int pointMaterializations = 0;
  int glossFieldCalls = 0;
  int glossFieldSurfaceCalls = 0;
  int glossProjectionCalls = 0;
  int vectorCalls = 0;
  int glossReleaseCalls = 0;
  int scopeHistogramCalls = 0;
  int scopeWaveformCalls = 0;
  int derivedReleaseCalls = 0;
  int failRasterOnCall = 0;
  bool glossPending = false;
  bool ownerMismatch = false;
  std::vector<std::string> order;
};

void fail(std::string* error, const char* value) noexcept {
  if (error) *error = value;
}

bool createSurface(void* context, uint64_t compositorId, int width, int height,
                  int pixelFormat, ChromaspaceMetal::PlotSurface* out,
                  std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!state || compositorId == 0u || !out || width <= 0 || height <= 0 ||
      (pixelFormat != 0 && pixelFormat != 1)) {
    fail(error, "create-invalid");
    return false;
  }
  ++state->createCalls;
  *out = {state->nextSurface++, width, height, pixelFormat,
          static_cast<size_t>(width) * static_cast<size_t>(height) * 8u};
  return true;
}

void releaseSurface(void* context, uint64_t, uint32_t) noexcept {
  ++static_cast<MockState*>(context)->releaseCalls;
}

bool sourceEncode(void* context, const FrameSubmission&, uint64_t sourceId,
                  uint32_t, int, int, int, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (sourceId == 0u) {
    fail(error, "source-invalid");
    return false;
  }
  ++state->sourceCalls;
  state->order.emplace_back("source");
  return true;
}

bool rasterEncode(void* context, const FrameSubmission& submission,
                  ChromaspaceMetal::ResidentDerivedCache* cache,
                  const ChromaspaceMetal::RasterSourceRequest& raster,
                  const ChromaspaceMetal::RasterPointSurfaceRequest&,
                  uint64_t sourceId, uint64_t serial, uint32_t, int, int, int,
                  std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!cache || sourceId == 0u || serial == 0u) {
    fail(error, "raster-invalid");
    return false;
  }
  ++state->rasterCalls;
  if (state->failRasterOnCall != 0 &&
      state->rasterCalls == state->failRasterOnCall) {
    fail(error, "raster-intentional-failure");
    return false;
  }
  if (cache->cacheId == 0u) {
    cache->cacheId = state->nextCache++;
    ++state->pointMaterializations;
  }
  if (cache->ownerCompositorId != 0u &&
      cache->ownerCompositorId != submission.compositorId) {
    state->ownerMismatch = true;
    fail(error, "raster-owner-mismatch");
    return false;
  }
  cache->ownerCompositorId = submission.compositorId;
  cache->builtSerial = serial;
  cache->byteSize = static_cast<size_t>(raster.pointCount) * 28u + 16u;
  cache->family = ChromaspaceMetal::ResidentDerivedFamily::RasterPointCloud;
  cache->available = true;
  state->order.emplace_back("raster");
  return true;
}

bool histogramEncode(void*, const FrameSubmission&,
                     const ChromaspaceMetal::RasterSourceRequest&,
                     const ChromaspaceMetal::HistogramSurfaceRequest&, uint64_t,
                     uint32_t, int, int, int, std::string*) noexcept {
  return true;
}

bool waveformEncode(void*, const FrameSubmission&,
                    const ChromaspaceMetal::RasterSourceRequest&,
                    const ChromaspaceMetal::WaveformSurfaceRequest&, uint64_t,
                    uint32_t, int, int, int, std::string*) noexcept {
  return true;
}

bool glossFieldEncode(void* context, const FrameSubmission& submission,
                      ChromaspaceMetal::GlossFieldCache* cache,
                      const ChromaspaceMetal::RasterSourceRequest&,
                      const ChromaspaceMetal::GlossFieldRequest& request,
                      uint64_t sourceId, uint64_t serial,
                      std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!cache || sourceId == 0u || serial == 0u) {
    fail(error, "gloss-field-invalid");
    return false;
  }
  ++state->glossFieldCalls;
  state->order.emplace_back("gloss-field");
  if (cache->cacheId == 0u) cache->cacheId = state->nextCache++;
  if (cache->ownerCompositorId != 0u &&
      cache->ownerCompositorId != submission.compositorId) {
    state->ownerMismatch = true;
    fail(error, "gloss-field-owner-mismatch");
    return false;
  }
  cache->ownerCompositorId = submission.compositorId;
  cache->gridWidth = request.gridWidth;
  cache->gridHeight = request.gridHeight;
  cache->builtSerial = serial;
  cache->byteSize = static_cast<size_t>(request.gridWidth) *
                    static_cast<size_t>(request.gridHeight) * 84u;
  cache->available = true;
  return true;
}

bool glossFieldSurfaceEncode(
    void* context, const FrameSubmission&,
    const ChromaspaceMetal::GlossFieldCache& cache,
    const ChromaspaceMetal::GlossFieldSurfaceRequest&, uint32_t, int, int, int,
    std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (cache.cacheId == 0u) {
    fail(error, "gloss-surface-invalid");
    return false;
  }
  if (cache.ownerCompositorId != 42u) state->ownerMismatch = true;
  ++state->glossFieldSurfaceCalls;
  state->order.emplace_back("gloss-field-surface");
  return true;
}

bool glossProjectionEncode(
    void* context, const FrameSubmission&,
    const ChromaspaceMetal::GlossFieldCache& cache,
    const ChromaspaceMetal::GlossProjectionSurfaceRequest&, uint32_t, int, int,
    int, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (cache.cacheId == 0u) {
    fail(error, "gloss-projection-invalid");
    return false;
  }
  if (cache.ownerCompositorId != 42u) state->ownerMismatch = true;
  ++state->glossProjectionCalls;
  state->order.emplace_back("gloss-projection");
  return true;
}

bool vectorsEncode(void* context, const FrameSubmission&, uint32_t, int, int, int,
                   const ChromaspaceMetal::FrameVectorVertex*, std::size_t count,
                   bool clearBeforeDraw, const std::array<float, 4>&,
                   std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (count == 0u || (count % 3u) != 0u) {
    fail(error, "vectors-invalid");
    return false;
  }
  ++state->vectorCalls;
  state->order.emplace_back(clearBeforeDraw ? "vectors-clear" : "vectors-overlay");
  return true;
}

ChromaspaceMetal::GlossFieldCacheState glossState(
    void* context, const ChromaspaceMetal::GlossFieldCache& cache) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!cache.available || cache.cacheId == 0u) {
    return ChromaspaceMetal::GlossFieldCacheState::Missing;
  }
  if (cache.ownerCompositorId != 42u) state->ownerMismatch = true;
  return state->glossPending ? ChromaspaceMetal::GlossFieldCacheState::Pending
                             : ChromaspaceMetal::GlossFieldCacheState::Ready;
}

void releaseGloss(void* context, ChromaspaceMetal::GlossFieldCache* cache) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (cache && cache->cacheId != 0u && cache->ownerCompositorId != 42u) {
    state->ownerMismatch = true;
  }
  if (cache && cache->cacheId != 0u) ++state->glossReleaseCalls;
  if (cache) *cache = ChromaspaceMetal::GlossFieldCache{};
}

bool histogramCachedEncode(
    void* context, const FrameSubmission& submission,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::HistogramSurfaceRequest& request, uint64_t sourceId,
    uint64_t serial, uint32_t, int, int, int, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!cache || sourceId == 0u || serial == 0u) {
    fail(error, "scope-histogram-invalid");
    return false;
  }
  ++state->scopeHistogramCalls;
  if (cache->cacheId == 0u) cache->cacheId = state->nextCache++;
  if (cache->ownerCompositorId != 0u &&
      cache->ownerCompositorId != submission.compositorId) {
    state->ownerMismatch = true;
    fail(error, "scope-histogram-owner-mismatch");
    return false;
  }
  cache->ownerCompositorId = submission.compositorId;
  const size_t channels = request.scopeMode == 1 ? 1u : 3u;
  const size_t density = static_cast<size_t>(request.width) * channels * 4u;
  cache->builtSerial = serial;
  cache->byteSize = density * (request.showOverflow != 0 ? 2u : 1u) + 4u +
                    (request.useGpuAutoRange != 0 ? 12u : 0u);
  cache->family = ChromaspaceMetal::ResidentDerivedFamily::Histogram;
  cache->available = true;
  return true;
}

bool waveformCachedEncode(
    void* context, const FrameSubmission& submission,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::WaveformSurfaceRequest& request, uint64_t sourceId,
    uint64_t serial, uint32_t, int, int, int, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (!cache || sourceId == 0u || serial == 0u) {
    fail(error, "scope-waveform-invalid");
    return false;
  }
  ++state->scopeWaveformCalls;
  if (cache->cacheId == 0u) cache->cacheId = state->nextCache++;
  if (cache->ownerCompositorId != 0u &&
      cache->ownerCompositorId != submission.compositorId) {
    state->ownerMismatch = true;
    fail(error, "scope-waveform-owner-mismatch");
    return false;
  }
  cache->ownerCompositorId = submission.compositorId;
  const bool lumaOnly = request.scopeMode == 2;
  const bool paradeLuma = request.scopeMode == 1 && request.includeLuma != 0;
  const size_t channels = lumaOnly ? 1u : (paradeLuma ? 4u : 3u);
  const size_t density = static_cast<size_t>(request.width) *
                         static_cast<size_t>(request.height) * channels * 4u;
  cache->builtSerial = serial;
  cache->byteSize = density * (request.showOverflow != 0 ? 2u : 1u) + 4u +
                    (request.useGpuAutoRange != 0 ? 12u : 0u);
  cache->family = ChromaspaceMetal::ResidentDerivedFamily::Waveform;
  cache->available = true;
  return true;
}

ChromaspaceMetal::ResidentDerivedCacheState derivedState(
    void* context, const ChromaspaceMetal::ResidentDerivedCache& cache) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (cache.cacheId != 0u && cache.ownerCompositorId != 42u) {
    state->ownerMismatch = true;
  }
  return cache.available && cache.cacheId != 0u && cache.byteSize != 0u &&
                 cache.ownerCompositorId == 42u
             ? ChromaspaceMetal::ResidentDerivedCacheState::Ready
             : ChromaspaceMetal::ResidentDerivedCacheState::Missing;
}

void releaseDerived(void* context,
                    ChromaspaceMetal::ResidentDerivedCache* cache) noexcept {
  auto* state = static_cast<MockState*>(context);
  if (cache && cache->cacheId != 0u && cache->ownerCompositorId != 42u) {
    state->ownerMismatch = true;
  }
  if (cache && cache->cacheId != 0u) ++state->derivedReleaseCalls;
  if (cache) *cache = ChromaspaceMetal::ResidentDerivedCache{};
}

RendererBackend backend(MockState* state) {
  return {state,
          createSurface,
          releaseSurface,
          sourceEncode,
          histogramEncode,
          waveformEncode,
          rasterEncode,
          glossFieldEncode,
          glossFieldSurfaceEncode,
          glossProjectionEncode,
          vectorsEncode,
          glossState,
          releaseGloss,
          histogramCachedEncode,
          waveformCachedEncode,
          derivedState,
          releaseDerived};
}

ChromaspaceMetal::ImportedSourceTexture source() {
  ChromaspaceMetal::ImportedSourceTexture value{};
  value.sourceId = 1u;
  value.senderId = "producer";
  value.deviceRegistryId = 2u;
  value.senderGeneration = 3u;
  value.sequence = 4u;
  value.slotIndex = 0u;
  value.slotGeneration = 5u;
  value.readyValue = 6u;
  value.contentHash = 7u;
  value.width = 16;
  value.height = 16;
  value.pixelFormat = 0;
  value.bytesPerRow = 16u * 8u;
  value.byteSize = value.bytesPerRow * 16u;
  value.semantics.sourceWidth = 16u;
  value.semantics.sourceHeight = 16u;
  value.semantics.sampledWidth = 16u;
  value.semantics.sampledHeight = 16u;
  value.semantics.authoritative = true;
  value.semantics.coverage = ChromaspaceSourceExchange::SourceCoverage::FullSource;
  value.semantics.colorPrimaries = "acescg";
  value.semantics.transferFunction = "linear";
  return value;
}

PlotCommand rasterCommand(int windowId) {
  PlotCommand command{};
  command.windowId = windowId;
  command.kind = PlotKind::ResidentRaster;
  command.destination = {0.0f, 0.0f, 100.0f, 100.0f};
  command.targetWidth = 100;
  command.targetHeight = 100;
  command.raster.pointCount = 1;
  command.raster.basePointCount = 1;
  command.raster.sourceWidth = 16;
  command.raster.sourceHeight = 16;
  command.raster.sampleCountX = 1;
  command.point.pointCount = 1;
  command.point.width = 100;
  command.point.height = 100;
  return command;
}

PlotCommand histogramCommand(int windowId) {
  PlotCommand command{};
  command.windowId = windowId;
  command.kind = PlotKind::Histogram;
  command.destination = {0.0f, 0.0f, 100.0f, 100.0f};
  command.targetWidth = 100;
  command.targetHeight = 100;
  command.raster.pointCount = 16;
  command.raster.basePointCount = 16;
  command.raster.sourceWidth = 16;
  command.raster.sourceHeight = 16;
  command.raster.sampleStride = 1;
  command.raster.sampleCountX = 4;
  command.histogram.pointCount = 16;
  command.histogram.width = 64;
  command.histogram.height = 32;
  command.histogram.scopeMode = 0;
  command.histogram.showOverflow = 1;
  command.histogram.highlightOverflow = 1;
  command.contentRevision = 1u;
  return command;
}

PlotCommand glossCommand(int windowId, PlotKind kind) {
  PlotCommand command{};
  command.windowId = windowId;
  command.kind = kind;
  command.destination = {0.0f, 0.0f, 100.0f, 100.0f};
  command.targetWidth = 100;
  command.targetHeight = 100;
  command.raster.pointCount = 1;
  command.raster.basePointCount = 1;
  command.raster.sourceWidth = 16;
  command.raster.sourceHeight = 16;
  command.raster.sampleCountX = 1;
  command.glossDerivationHash = 8u;
  command.viewRevision = 9u;
  return command;
}

ChromaspaceMetalFrameExecutor::FrameExecutionContext context(
    ChromaspaceMetalFrameExecutor::FrameBatch* batch) {
  static FrameSubmission submission{9u, 42u};
  return {42u, &submission, batch};
}

FrameRequest trimRequest() {
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  assert(request.append(rasterCommand(1)));
  assert(request.append(glossCommand(2, PlotKind::GlossField2D)));
  return request;
}

void encodeAndCommit(PlotRenderer* renderer,
                     const FrameRequest& request,
                     ChromaspaceMetalFrameExecutor::FrameBatch* batch,
                     RenderResult* result,
                     std::string* error) {
  assert(renderer != nullptr && batch != nullptr && result != nullptr);
  batch->clear();
  batch->compositeItems.reserve(request.commandCount);
  assert(renderer->prepare(request, 42u, result, error));
  assert(renderer->encodePrepared(context(batch), batch, result, error));
  assert(renderer->finish(true, result));
}

void testTrimInvalidLevelAndActiveTransactionRejection() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request = trimRequest();
  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  encodeAndCommit(&renderer, request, &batch, &result, &error);
  const auto before = renderer.residencySnapshot();
  const int releaseCallsBefore = state.releaseCalls;
  const int derivedReleaseCallsBefore = state.derivedReleaseCalls;
  const int glossReleaseCallsBefore = state.glossReleaseCalls;

  const auto invalid = renderer.trim(static_cast<TrimLevel>(0xffu));
  assert(invalid.status == TrimStatus::InvalidLevel);
  assert(renderer.residencySnapshot().surfaceCount == before.surfaceCount);
  assert(renderer.residencySnapshot().derivedCacheCount ==
         before.derivedCacheCount);
  assert(state.releaseCalls == releaseCallsBefore &&
         state.derivedReleaseCalls == derivedReleaseCallsBefore &&
         state.glossReleaseCalls == glossReleaseCallsBefore);

  batch.clear();
  batch.compositeItems.reserve(request.commandCount);
  assert(renderer.prepare(request, 42u, &result, &error));
  const auto rejected = renderer.trim(TrimLevel::AllPlotResources);
  assert(rejected.status == TrimStatus::TransactionActive);
  assert(rejected.before.surfaceCount == before.surfaceCount &&
         rejected.before.surfaceBytes == before.surfaceBytes &&
         rejected.before.derivedCacheCount == before.derivedCacheCount &&
         rejected.before.derivedCacheBytes == before.derivedCacheBytes);
  assert(rejected.after.surfaceCount == rejected.before.surfaceCount &&
         rejected.after.surfaceBytes == rejected.before.surfaceBytes &&
         rejected.after.derivedCacheCount ==
             rejected.before.derivedCacheCount &&
         rejected.after.derivedCacheBytes == rejected.before.derivedCacheBytes);
  assert(state.releaseCalls == releaseCallsBefore &&
         state.derivedReleaseCalls == derivedReleaseCallsBefore &&
         state.glossReleaseCalls == glossReleaseCallsBefore);
  assert(renderer.finish(false, &result));
}

void testDerivedOnlyTrimPreservesSurfacesAndContentKeys() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request = trimRequest();
  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  encodeAndCommit(&renderer, request, &batch, &result, &error);
  const auto before = renderer.residencySnapshot();
  assert(before.surfaceCount == 2u && before.surfaceBytes == 160000u);
  assert(before.derivedCacheCount == 2u);
  const std::size_t expectedDerivedBytes = 44u + 96u * 96u * 84u;
  assert(before.derivedCacheBytes == expectedDerivedBytes);

  const auto trimmed = renderer.trim(TrimLevel::DerivedOnly);
  assert(trimmed.accepted());
  assert(trimmed.before.surfaceCount == 2u &&
         trimmed.before.surfaceBytes == 160000u &&
         trimmed.before.derivedCacheCount == 2u &&
         trimmed.before.derivedCacheBytes == expectedDerivedBytes);
  assert(trimmed.after.surfaceCount == 2u &&
         trimmed.after.surfaceBytes == 160000u &&
         trimmed.after.derivedCacheCount == 0u &&
         trimmed.after.derivedCacheBytes == 0u);
  assert(trimmed.releasedSurfaceCount == 0u &&
         trimmed.releasedSurfaceBytes == 0u &&
         trimmed.releasedDerivedCacheCount == 2u &&
         trimmed.releasedDerivedCacheBytes == expectedDerivedBytes);
  assert(state.releaseCalls == 0 && state.derivedReleaseCalls == 1 &&
         state.glossReleaseCalls == 1);
  assert(renderer.resourceCount() == 2u && renderer.derivedCacheCount() == 0u);

  // The final plot pixels and their content keys remain resident, so an
  // identical frame is a presentation cache hit even though derivation data
  // was reclaimed.
  encodeAndCommit(&renderer, request, &batch, &result, &error);
  assert(result.residentContentHitCount == 2u);
  assert(state.releaseCalls == 0 && state.derivedReleaseCalls == 1 &&
         state.glossReleaseCalls == 1);
}

void testAllTrimIsExactIdempotentAndReusable() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request = trimRequest();
  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  encodeAndCommit(&renderer, request, &batch, &result, &error);
  const auto trimmed = renderer.trim(TrimLevel::AllPlotResources);
  assert(trimmed.accepted());
  assert(trimmed.before.surfaceCount == 2u &&
         trimmed.before.surfaceBytes == 160000u &&
         trimmed.before.derivedCacheCount == 2u);
  assert(trimmed.after.surfaceCount == 0u && trimmed.after.surfaceBytes == 0u &&
         trimmed.after.derivedCacheCount == 0u &&
         trimmed.after.derivedCacheBytes == 0u);
  assert(trimmed.releasedSurfaceCount == 2u &&
         trimmed.releasedSurfaceBytes == 160000u &&
         trimmed.releasedDerivedCacheCount == 2u);
  assert(state.releaseCalls == 2 && state.derivedReleaseCalls == 1 &&
         state.glossReleaseCalls == 1);
  assert(renderer.resourceCount() == 0u && renderer.derivedCacheCount() == 0u &&
         renderer.residentSurfaceBytes() == 0u);

  const auto repeated = renderer.trim(TrimLevel::AllPlotResources);
  assert(repeated.accepted());
  assert(repeated.before.surfaceCount == 0u && repeated.before.surfaceBytes == 0u &&
         repeated.before.derivedCacheCount == 0u &&
         repeated.releasedSurfaceCount == 0u &&
         repeated.releasedDerivedCacheCount == 0u);
  assert(state.releaseCalls == 2 && state.derivedReleaseCalls == 1 &&
         state.glossReleaseCalls == 1);

  // All-resource trim is non-terminal: the same compositor can prepare and
  // submit a fresh frame without resetting renderer identity.
  encodeAndCommit(&renderer, request, &batch, &result, &error);
  assert(renderer.resourceCount() == 2u && renderer.derivedCacheCount() == 2u);
  assert(state.createCalls == 4 && state.releaseCalls == 2);
}

void testVectorOverlayOrder() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand command = rasterCommand(1);
  const ChromaspaceMetal::FrameVectorVertex vertices[3] = {
      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},
      {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f}};
  command.vectorClearBeforeDraw = false;
  assert(request.appendVectorVertices(vertices, 3u, &command));
  assert(request.append(command));
  RenderResult result;
  std::string error;
  assert(renderer.prepare(request, 42u, &result, &error));
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(1u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.order.size() == 2u && state.order[0] == "raster" &&
         state.order[1] == "vectors-overlay");
  assert(renderer.finish(true, &result));
  assert(result.frameSucceeded && renderer.resourceCount() == 1u);
}

void testResidentContentCacheAndTransactionalReplacement() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(rasterCommand(1));
  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(1u);

  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(renderer.finish(true, &result));
  assert(state.createCalls == 1 && state.rasterCalls == 1);

  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(result.residentContentHitCount == 1u && result.eventCount == 1u &&
         result.events[0].status == WindowStatus::Reused &&
         result.events[0].reason == "resident-plot-content-cache-hit");
  assert(renderer.finish(true, &result));
  assert(state.createCalls == 1 && state.rasterCalls == 1);

  // Placement-only changes re-composite the same resident pixels.
  request.commands[0].destination.x = 45.0f;
  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(result.residentContentHitCount == 1u &&
         batch.compositeItems[0].dstX == 45.0f);
  assert(renderer.finish(true, &result));
  assert(state.createCalls == 1 && state.rasterCalls == 1);

  // A content change stages a replacement surface. Aborting releases only
  // that candidate and leaves the committed resident pixels reusable.
  request.commands[0].contentRevision = 2u;
  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.replacedSurfaceCount == 1u && state.createCalls == 2);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.rasterCalls == 2);
  assert(renderer.finish(false, &result));
  assert(state.releaseCalls == 1 && renderer.hasResource(1));

  request.commands[0].contentRevision = 1u;
  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(result.residentContentHitCount == 1u && state.rasterCalls == 2);
  assert(renderer.finish(true, &result));

  request.residentSource.sequence += 1u;
  request.residentSource.contentHash += 1u;
  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.replacedSurfaceCount == 1u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(renderer.finish(true, &result));
  assert(state.rasterCalls == 3 && state.releaseCalls == 2);
}

void testSharedResidentPointCloudAndPresentationReuse() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand first = rasterCommand(1);
  PlotCommand second = rasterCommand(2);
  second.targetWidth = 120;
  second.targetHeight = 80;
  second.point.width = 120;
  second.point.height = 80;
  second.point.pointRadiusPixels = 4.0f;
  second.point.backgroundR = 0.15f;
  second.point.modelView[0] = 0.5f;
  second.contentRevision = 2u;
  assert(request.append(first));
  assert(request.append(second));

  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedCandidateCount == 1u &&
         result.residentDerivedHitCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.rasterCalls == 2 && state.pointMaterializations == 1);
  assert(renderer.finish(true, &result));
  assert(renderer.derivedCacheCount() == 1u &&
         renderer.residentDerivedBytes() == 44u);

  // Camera/radius/background/target changes require new final pixels but reuse
  // the same resident point cloud without importing or rescanning the source.
  request.commands[0].contentRevision = 3u;
  request.commands[0].point.projection[5] = 0.75f;
  request.commands[0].point.pointRadiusPixels = 6.0f;
  request.commands[0].point.backgroundB = 0.25f;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedHitCount == 1u &&
         result.residentDerivedCandidateCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.pointMaterializations == 1);
  assert(renderer.finish(true, &result));

  // A true point-derivation change stages a new native record. Abort releases
  // only that candidate and preserves the last committed point cloud.
  request.commands[0].contentRevision = 4u;
  request.commands[0].raster.colorSaturation = 0.5f;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedCandidateCount == 1u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.pointMaterializations == 2);
  assert(renderer.finish(false, &result));
  assert(state.derivedReleaseCalls == 1 &&
         renderer.derivedCacheCount() == 1u);

  renderer.shutdown();
  assert(state.derivedReleaseCalls == 2 &&
         renderer.derivedCacheCount() == 0u && !state.ownerMismatch);
}

void testSharedResidentScopeCacheAndAbortRecovery() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(histogramCommand(1));
  request.append(histogramCommand(2));
  RenderResult result;
  std::string error;
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedCandidateCount == 1u &&
         result.residentDerivedHitCount == 0u);
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(2u);
  auto frameContext = context(&batch);
  assert(renderer.encodePrepared(frameContext, &batch, &result, &error));
  assert(state.scopeHistogramCalls == 2 && state.nextCache == 501u);
  assert(renderer.finish(true, &result));
  assert(renderer.derivedCacheCount() == 1u &&
         renderer.residentDerivedBytes() > 0u);

  // Presentation-only overflow highlighting invalidates final pixels but not
  // the analytical density derivation shared by the two windows.
  request.commands[0].histogram.highlightOverflow = 0;
  request.commands[0].contentRevision = 2u;
  batch.compositeItems.clear();
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedHitCount == 1u &&
         result.residentDerivedCandidateCount == 0u);
  assert(renderer.encodePrepared(frameContext, &batch, &result, &error));
  assert(renderer.finish(true, &result));
  assert(state.nextCache == 501u && renderer.derivedCacheCount() == 1u);

  // A new authoritative source stages a distinct cache. Aborting releases
  // only that candidate and preserves the committed derivation.
  request.residentSource.contentHash = 99u;
  request.residentSource.sequence = 100u;
  request.commands[0].contentRevision = 3u;
  request.commands[1].contentRevision = 3u;
  batch.compositeItems.clear();
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(frameContext, &batch, &result, &error));
  assert(state.nextCache == 502u);
  assert(renderer.finish(false, &result));
  assert(state.derivedReleaseCalls == 1 && renderer.derivedCacheCount() == 1u);

  renderer.shutdown();
  assert(state.derivedReleaseCalls == 2 && renderer.derivedCacheCount() == 0u &&
         !state.ownerMismatch);

  // Runtime/device recreation reuses the renderer after a nonterminal reset.
  batch.compositeItems.clear();
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(frameContext, &batch, &result, &error));
  assert(renderer.finish(true, &result));
  assert(renderer.derivedCacheCount() == 1u);
  renderer.shutdown();
  assert(state.derivedReleaseCalls == 3 && !state.ownerMismatch);
}

void testResidencyBudgets() {
  MockState invalidState;
  RendererBackend invalidBackend = backend(&invalidState);
  auto invalid = std::make_unique<PlotRenderer>(
      &invalidBackend, ResidencyConfig{100000u, 99999u});
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(rasterCommand(1));
  RenderResult result;
  std::string error;
  assert(!invalid->prepare(request, 42u, &result, &error) &&
         error == "plot-residency-config-invalid");

  MockState residentState;
  RendererBackend residentBackend = backend(&residentState);
  auto residentLimited = std::make_unique<PlotRenderer>(
      &residentBackend, ResidencyConfig{79999u, 160000u});
  assert(!residentLimited->prepare(request, 42u, &result, &error) &&
         error == "plot-resident-surface-budget-exceeded" &&
         residentLimited->resourceCount() == 0u &&
         residentState.createCalls == 1 && residentState.releaseCalls == 1);

  MockState transientState;
  RendererBackend transientBackend = backend(&transientState);
  auto transientLimited = std::make_unique<PlotRenderer>(
      &transientBackend, ResidencyConfig{100000u, 120000u});
  assert(transientLimited->prepare(request, 42u, &result, &error));
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(1u);
  assert(transientLimited->encodePrepared(
      context(&batch), &batch, &result, &error));
  assert(transientLimited->finish(true, &result));
  assert(result.residentSurfaceBytes == 80000u &&
         result.transientSurfaceBytes == 80000u &&
         transientLimited->residentSurfaceBytes() == 80000u);

  request.commands[0].contentRevision = 2u;
  assert(!transientLimited->prepare(request, 42u, &result, &error) &&
         error == "plot-transient-residency-budget-exceeded" &&
         transientLimited->hasResource(1) &&
         transientLimited->residentSurfaceBytes() == 80000u &&
         transientState.createCalls == 1);
}

void testScaffoldReasonPropagation() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  PlotCommand command{};
  command.windowId = 9;
  command.kind = PlotKind::Scaffold;
  command.destination = {0.0f, 0.0f, 64.0f, 64.0f};
  command.targetWidth = 64;
  command.targetHeight = 64;
  command.unavailableReason = "resident-source-unavailable";
  const ChromaspaceMetal::FrameVectorVertex vertices[3] = {
      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f},
      {64.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f},
      {0.0f, 64.0f, 0.0f, 0.0f, 1.0f, 1.0f}};
  assert(request.appendVectorVertices(vertices, 3u, &command));
  assert(request.append(command));
  RenderResult result;
  std::string error;
  assert(renderer.prepare(request, 42u, &result, &error));
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(1u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(result.eventCount == 1u && result.events[0].status == WindowStatus::Scaffolded &&
         result.events[0].reason == "resident-source-unavailable");
  assert(renderer.finish(true, &result));
}

void testSharedGlossCacheAndTransactionalReuse() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(glossCommand(1, PlotKind::GlossField2D));
  PlotCommand projection = glossCommand(2, PlotKind::GlossProjection3D);
  projection.targetWidth = 120;
  projection.targetHeight = 80;
  projection.destination.width = 120.0f;
  projection.destination.height = 80.0f;
  projection.glossProjectionSurface.modelView[0] = 0.5f;
  projection.glossProjectionSurface.projection[5] = 0.75f;
  assert(request.append(projection));
  RenderResult result;
  std::string error;
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedCandidateCount == 1u &&
         result.residentDerivedHitCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.glossFieldCalls == 1 && state.glossFieldSurfaceCalls == 1 &&
         state.glossProjectionCalls == 1);
  assert(renderer.finish(true, &result));
  assert(renderer.glossCacheCount() == 1u &&
         renderer.residentDerivedBytes() == 96u * 96u * 84u);

  // Presentation-only controls and the coarse caller hash do not alter the
  // generalized Gloss derivation key.
  request.commands[0].viewRevision = 99u;
  request.commands[0].glossDerivationHash = 999u;
  request.commands[0].contentRevision = 2u;
  request.commands[0].glossFieldSurface.colorSaturation = 1.1f;
  request.commands[1].viewRevision = 101u;
  request.commands[1].contentRevision = 2u;
  request.commands[1].glossProjectionSurface.modelView[1] = 0.25f;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedHitCount == 1u &&
         result.residentDerivedCandidateCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.glossFieldCalls == 1 && state.glossFieldSurfaceCalls == 2 &&
         state.glossProjectionCalls == 2);
  assert(renderer.finish(true, &result));

  // Repeating the exact final-pixel request reuses both surfaces
  // transactionally and performs no field or surface encode.
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(result.residentContentHitCount == 2u &&
         state.glossFieldCalls == 1 && state.glossFieldSurfaceCalls == 2 &&
         state.glossProjectionCalls == 2);
  assert(renderer.finish(true, &result));

  // A derivation change stages a new field. Aborting releases only the new
  // candidate and preserves the committed Gloss family entry.
  request.commands[0].glossField.neighborhoodChoice = 2;
  request.commands[0].contentRevision = 2u;
  request.commands[1].glossField.neighborhoodChoice = 2;
  request.commands[1].contentRevision = 2u;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedCandidateCount == 1u &&
         result.residentDerivedHitCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(state.glossFieldCalls == 2 && state.glossFieldSurfaceCalls == 3 &&
         state.glossProjectionCalls == 3);
  assert(renderer.finish(false, &result));
  assert(state.glossReleaseCalls == 1 && renderer.glossCacheCount() == 1u &&
         renderer.residentDerivedBytes() == 96u * 96u * 84u);

  // Restore the committed derivation and verify it is still a hit.
  request.commands[0].glossField.neighborhoodChoice = 1;
  request.commands[0].contentRevision = 3u;
  request.commands[1].glossField.neighborhoodChoice = 1;
  request.commands[1].contentRevision = 3u;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(result.residentDerivedHitCount == 1u &&
         result.residentDerivedCandidateCount == 0u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(renderer.finish(true, &result));

  renderer.shutdown();
  assert(state.glossReleaseCalls == 2 && renderer.glossCacheCount() == 0u &&
         !state.ownerMismatch);
}

void testAbortAndCapacity() {
  MockState state;
  RendererBackend table = backend(&state);
  PlotRenderer renderer(&table);
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(rasterCommand(1));
  RenderResult result;
  std::string error;
  assert(renderer.prepare(request, 42u, &result, &error));
  ChromaspaceMetalFrameExecutor::FrameBatch batch;
  assert(!renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(renderer.finish(false, &result));
  assert(renderer.resourceCount() == 0u && state.releaseCalls == 1);

  // A later command failure must leave the already committed resource and
  // caller-owned batch untouched.
  request.clear();
  request.hasResidentSource = true;
  request.residentSource = source();
  request.append(rasterCommand(1));
  assert(renderer.prepare(request, 42u, &result, &error));
  batch.clear();
  batch.compositeItems.reserve(1u);
  assert(renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(renderer.finish(true, &result));
  const int releasesBeforeFailure = state.releaseCalls;

  request.clear();
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand changed = rasterCommand(1);
  changed.contentRevision = 2u;
  request.append(changed);
  request.append(rasterCommand(2));
  state.failRasterOnCall = state.rasterCalls + 2;
  batch.clear();
  batch.compositeItems.reserve(2u);
  assert(renderer.prepare(request, 42u, &result, &error));
  assert(!renderer.encodePrepared(context(&batch), &batch, &result, &error));
  assert(batch.compositeItems.empty());
  assert(renderer.finish(false, &result));
  assert(renderer.resourceCount() == 1u && renderer.hasResource(1) &&
         !renderer.hasResource(2) && state.releaseCalls > releasesBeforeFailure);
}

void testValidationRejectsMissingDerivedSamples() {
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand command = rasterCommand(1);
  command.raster.sampleCountX = 0;
  request.append(command);
  std::string error;
  assert(!validateFrameRequest(request, &error));
  assert(error == "plot-raster-request-invalid");
}

void testValidationRejectsMismatchedGpuAutoRange() {
  FrameRequest request;
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand histogram = rasterCommand(1);
  histogram.kind = PlotKind::Histogram;
  histogram.histogram.pointCount = histogram.raster.pointCount;
  histogram.histogram.width = histogram.targetWidth;
  histogram.histogram.height = histogram.targetHeight;
  histogram.histogram.useGpuAutoRange = 1;
  histogram.histogram.autoRange.pointCount = histogram.raster.pointCount;
  histogram.histogram.autoRange.waveform = 1;  // Histogram must use waveform=0.
  request.append(histogram);
  std::string error;
  assert(!validateFrameRequest(request, &error));
  assert(error == "plot-histogram-request-invalid");

  request.clear();
  request.hasResidentSource = true;
  request.residentSource = source();
  PlotCommand waveform = rasterCommand(1);
  waveform.kind = PlotKind::Waveform;
  waveform.waveform.pointCount = waveform.raster.pointCount;
  waveform.waveform.width = waveform.targetWidth;
  waveform.waveform.height = waveform.targetHeight;
  waveform.waveform.useGpuAutoRange = 1;
  waveform.waveform.autoRange.pointCount = waveform.raster.pointCount + 1;
  waveform.waveform.autoRange.waveform = 1;
  request.append(waveform);
  assert(!validateFrameRequest(request, &error));
  assert(error == "plot-waveform-request-invalid");
}

}  // namespace

int main() {
  testTrimInvalidLevelAndActiveTransactionRejection();
  testDerivedOnlyTrimPreservesSurfacesAndContentKeys();
  testAllTrimIsExactIdempotentAndReusable();
  testVectorOverlayOrder();
  testResidentContentCacheAndTransactionalReplacement();
  testSharedResidentPointCloudAndPresentationReuse();
  testSharedResidentScopeCacheAndAbortRecovery();
  testResidencyBudgets();
  testScaffoldReasonPropagation();
  testSharedGlossCacheAndTransactionalReuse();
  testAbortAndCapacity();
  testValidationRejectsMissingDerivedSamples();
  testValidationRejectsMismatchedGpuAutoRange();
  return 0;
}
