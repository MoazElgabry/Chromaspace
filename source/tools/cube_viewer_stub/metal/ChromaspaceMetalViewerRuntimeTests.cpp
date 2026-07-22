#include "ChromaspaceMetalViewerRuntime.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace ChromaspaceMetalViewerRuntime;
using namespace ChromaspaceMetalFrameExecutor;
using namespace ChromaspaceMetalPlotRenderer;
using Failure = ChromaspaceMetalFrameFailure::Kind;
using Visibility = ChromaspaceFrameRecoveryPolicy::SurfaceVisibility;

struct ExecutorState {
  int createCalls = 0;
  int resizeCalls = 0;
  int drainCalls = 0;
  int destroyCalls = 0;
  int beginCalls = 0;
  int submitCalls = 0;
  int abandonCalls = 0;
  int nextCompositorId = 100;
  uint64_t nextRuntimeContextId = 1000u;
  uint64_t lastRuntimeContextId = 0u;
  uint64_t activeRuntimeContextId = 0u;
  uint64_t deviceRegistryId = 2000u;
  bool reuseRuntimeContextId = false;
  bool failCreate = false;
  bool failResize = false;
  bool failBegin = false;
  bool failSubmit = false;
  Failure beginFailure = Failure::None;
  Failure submitFailure = Failure::None;
  ChromaspaceMetalViewerRuntime::Runtime* runtimeForPressureProbe = nullptr;
  bool probePressureDuringBegin = false;
  MemoryPressureResult pressureProbe{};
  std::vector<std::size_t> submittedItemCounts;
};

bool execCreate(void* context,
                void*,
                int width,
                int height,
                float scale,
                FrameCompositorState* out,
                std::string* error) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->createCalls;
  if (state->failCreate) {
    if (error) *error = "mock-create-failed";
    return false;
  }
  if (!out || width <= 0 || height <= 0 || scale <= 0.0f) {
    if (error) *error = "mock-create-invalid";
    return false;
  }
  const uint64_t runtimeContextId =
      state->reuseRuntimeContextId && state->lastRuntimeContextId != 0u
          ? state->lastRuntimeContextId
          : state->nextRuntimeContextId++;
  state->lastRuntimeContextId = runtimeContextId;
  state->activeRuntimeContextId = runtimeContextId;
  *out = {static_cast<uint64_t>(state->nextCompositorId++),
          width,
          height,
          scale,
          runtimeContextId,
          state->deviceRegistryId};
  return true;
}

bool execResize(void* context,
                uint64_t compositorId,
                int width,
                int height,
                float scale,
                std::string* error) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->resizeCalls;
  if (state->failResize || compositorId == 0u || width <= 0 || height <= 0 ||
      scale <= 0.0f) {
    if (error) *error = "mock-resize-failed";
    return false;
  }
  return true;
}

bool execDrain(void* context,
               uint64_t compositorId,
               uint32_t timeout,
               std::string* error) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->drainCalls;
  if (compositorId == 0u || timeout == 0u) {
    if (error) *error = "mock-drain-invalid";
    return false;
  }
  return true;
}

void execDestroy(void* context, uint64_t compositorId) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->destroyCalls;
  assert(compositorId != 0u);
}

bool execBegin(void* context,
               uint64_t compositorId,
               ChromaspaceMetal::FrameSubmission* out,
               std::string* error,
               Failure* failure) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->beginCalls;
  if (state->probePressureDuringBegin && state->runtimeForPressureProbe != nullptr) {
    state->pressureProbe =
        state->runtimeForPressureProbe->handleMemoryPressure(
            MemoryPressureLevel::Warning);
    state->probePressureDuringBegin = false;
  }
  if (state->failBegin || !out || compositorId == 0u) {
    if (error) *error = "mock-begin-failed";
    if (failure) *failure = state->beginFailure;
    return false;
  }
  *out = {static_cast<uint64_t>(state->beginCalls),
          compositorId,
          state->activeRuntimeContextId,
          state->deviceRegistryId};
  if (failure) *failure = Failure::None;
  return true;
}

bool execSubmit(void* context,
                ChromaspaceMetal::FrameSubmission* submission,
                const FrameBatch& batch,
                std::string* error,
                Failure* failure) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->submitCalls;
  if (state->failSubmit || !submission || submission->submissionId == 0u) {
    if (error) *error = "mock-submit-failed";
    if (failure) *failure = state->submitFailure;
    return false;
  }
  state->submittedItemCounts.push_back(batch.compositeItems.size());
  *submission = ChromaspaceMetal::FrameSubmission{};
  if (failure) *failure = Failure::None;
  return true;
}

void execAbandon(void* context,
                 ChromaspaceMetal::FrameSubmission* submission) noexcept {
  auto* state = static_cast<ExecutorState*>(context);
  ++state->abandonCalls;
  if (submission) *submission = ChromaspaceMetal::FrameSubmission{};
}

FrameExecutorBackend executorBackend(ExecutorState* state) {
  return {state, execCreate, execResize, execDrain, execDestroy, execBegin,
          execSubmit, execAbandon};
}

struct RendererState {
  int createCalls = 0;
  int releaseCalls = 0;
  int vectorCalls = 0;
  int nextSurface = 500;
  bool failVectors = false;
};

bool renderCreate(void* context,
                  uint64_t compositorId,
                  int width,
                  int height,
                  int pixelFormat,
                  ChromaspaceMetal::PlotSurface* out,
                  std::string* error) noexcept {
  auto* state = static_cast<RendererState*>(context);
  if (!out || compositorId == 0u || width <= 0 || height <= 0 ||
      (pixelFormat != 0 && pixelFormat != 1)) {
    if (error) *error = "mock-surface-invalid";
    return false;
  }
  ++state->createCalls;
  *out = {static_cast<uint32_t>(state->nextSurface++), width, height,
          pixelFormat, static_cast<std::size_t>(width) * height * 8u};
  return true;
}

void renderRelease(void* context, uint64_t, uint32_t) noexcept {
  ++static_cast<RendererState*>(context)->releaseCalls;
}

bool renderSource(void*, const ChromaspaceMetal::FrameSubmission&, uint64_t,
                  uint32_t, int, int, int, std::string*) noexcept {
  return true;
}
bool renderHistogram(void*, const ChromaspaceMetal::FrameSubmission&,
                     const ChromaspaceMetal::RasterSourceRequest&,
                     const ChromaspaceMetal::HistogramSurfaceRequest&, uint64_t,
                     uint32_t, int, int, int, std::string*) noexcept {
  return true;
}
bool renderWaveform(void*, const ChromaspaceMetal::FrameSubmission&,
                    const ChromaspaceMetal::RasterSourceRequest&,
                    const ChromaspaceMetal::WaveformSurfaceRequest&, uint64_t,
                    uint32_t, int, int, int, std::string*) noexcept {
  return true;
}
bool renderRaster(void*, const ChromaspaceMetal::FrameSubmission&,
                  ChromaspaceMetal::ResidentDerivedCache* cache,
                  const ChromaspaceMetal::RasterSourceRequest& request,
                  const ChromaspaceMetal::RasterPointSurfaceRequest&, uint64_t,
                  uint64_t serial, uint32_t, int, int, int,
                  std::string*) noexcept {
  if (!cache || serial == 0u) return false;
  if (cache->cacheId == 0u) cache->cacheId = 9000u + serial;
  cache->builtSerial = serial;
  cache->byteSize = static_cast<size_t>(request.pointCount) * 28u + 16u;
  cache->family = ChromaspaceMetal::ResidentDerivedFamily::RasterPointCloud;
  cache->available = true;
  return true;
}
bool renderGlossField(void*, const ChromaspaceMetal::FrameSubmission&,
                      ChromaspaceMetal::GlossFieldCache* cache,
                      const ChromaspaceMetal::RasterSourceRequest&,
                      const ChromaspaceMetal::GlossFieldRequest& request,
                      uint64_t, uint64_t serial, std::string*) noexcept {
  if (!cache || serial == 0u) return false;
  cache->cacheId = 10000u + serial;
  cache->gridWidth = request.gridWidth;
  cache->gridHeight = request.gridHeight;
  cache->builtSerial = serial;
  cache->byteSize = static_cast<std::size_t>(request.gridWidth) *
                    static_cast<std::size_t>(request.gridHeight) * 84u;
  cache->available = true;
  return true;
}
bool renderGlossSurface(void*, const ChromaspaceMetal::FrameSubmission&,
                        const ChromaspaceMetal::GlossFieldCache&,
                        const ChromaspaceMetal::GlossFieldSurfaceRequest&,
                        uint32_t, int, int, int, std::string*) noexcept {
  return true;
}
bool renderGlossProjection(void*, const ChromaspaceMetal::FrameSubmission&,
                           const ChromaspaceMetal::GlossFieldCache&,
                           const ChromaspaceMetal::GlossProjectionSurfaceRequest&,
                           uint32_t, int, int, int, std::string*) noexcept {
  return true;
}

bool renderVectors(void* context,
                   const ChromaspaceMetal::FrameSubmission&,
                   uint32_t,
                   int,
                   int,
                   int,
                   const ChromaspaceMetal::FrameVectorVertex*,
                   std::size_t count,
                   bool,
                   const std::array<float, 4>&,
                   std::string* error) noexcept {
  auto* state = static_cast<RendererState*>(context);
  ++state->vectorCalls;
  if (state->failVectors) {
    if (error) *error = "mock-vector-encode-failed";
    return false;
  }
  if (count == 0u || count % 3u != 0u) {
    if (error) *error = "mock-vector-invalid";
    return false;
  }
  return true;
}

ChromaspaceMetal::GlossFieldCacheState renderGlossState(
    void*, const ChromaspaceMetal::GlossFieldCache& cache) noexcept {
  return cache.cacheId == 0u || cache.byteSize == 0u
             ? ChromaspaceMetal::GlossFieldCacheState::Missing
             : ChromaspaceMetal::GlossFieldCacheState::Ready;
}

void renderGlossRelease(void*, ChromaspaceMetal::GlossFieldCache* cache) noexcept {
  if (cache) *cache = ChromaspaceMetal::GlossFieldCache{};
}

bool renderHistogramCached(
    void*, const ChromaspaceMetal::FrameSubmission&,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::HistogramSurfaceRequest& request, uint64_t,
    uint64_t serial, uint32_t, int, int, int, std::string*) noexcept {
  if (!cache) return false;
  if (cache->cacheId == 0u) cache->cacheId = 7000u + serial;
  const size_t channels = request.scopeMode == 1 ? 1u : 3u;
  const size_t density = static_cast<size_t>(request.width) * channels * 4u;
  cache->builtSerial = serial;
  cache->byteSize = density * (request.showOverflow != 0 ? 2u : 1u) + 4u +
                    (request.useGpuAutoRange != 0 ? 12u : 0u);
  cache->family = ChromaspaceMetal::ResidentDerivedFamily::Histogram;
  cache->available = true;
  return true;
}

bool renderWaveformCached(
    void*, const ChromaspaceMetal::FrameSubmission&,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest&,
    const ChromaspaceMetal::WaveformSurfaceRequest& request, uint64_t,
    uint64_t serial, uint32_t, int, int, int, std::string*) noexcept {
  if (!cache) return false;
  if (cache->cacheId == 0u) cache->cacheId = 8000u + serial;
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

ChromaspaceMetal::ResidentDerivedCacheState renderDerivedState(
    void*, const ChromaspaceMetal::ResidentDerivedCache& cache) noexcept {
  return cache.cacheId == 0u
             ? ChromaspaceMetal::ResidentDerivedCacheState::Missing
             : ChromaspaceMetal::ResidentDerivedCacheState::Ready;
}

void renderDerivedRelease(
    void*, ChromaspaceMetal::ResidentDerivedCache* cache) noexcept {
  if (cache) *cache = ChromaspaceMetal::ResidentDerivedCache{};
}

RendererBackend rendererBackend(RendererState* state) {
  return {state,
          renderCreate,
          renderRelease,
          renderSource,
          renderHistogram,
          renderWaveform,
          renderRaster,
          renderGlossField,
          renderGlossSurface,
          renderGlossProjection,
          renderVectors,
          renderGlossState,
          renderGlossRelease,
          renderHistogramCached,
          renderWaveformCached,
          renderDerivedState,
          renderDerivedRelease};
}

struct ResourceState {
  uint64_t nextAtlas = 9000u;
  int createCalls = 0;
  int releaseCalls = 0;
  std::vector<uint64_t> released;
};

bool resourceCreate(void* context,
                    uint64_t compositorId,
                    int width,
                    int height,
                    const unsigned char* pixels,
                    std::size_t byteCount,
                    ChromaspaceMetal::FrameTextAtlas* out,
                    std::string* error) noexcept {
  auto* state = static_cast<ResourceState*>(context);
  if (!out || compositorId == 0u || width <= 0 || height <= 0 || !pixels ||
      byteCount != static_cast<std::size_t>(width) * height) {
    if (error) *error = "mock-atlas-invalid";
    return false;
  }
  ++state->createCalls;
  *out = {state->nextAtlas++, width, height};
  return true;
}

void resourceRelease(void* context, uint64_t, uint64_t atlasId) noexcept {
  auto* state = static_cast<ResourceState*>(context);
  ++state->releaseCalls;
  state->released.push_back(atlasId);
}

RuntimeResourceBackend resourceBackend(ResourceState* state) {
  return {state, resourceCreate, resourceRelease};
}

FrameRequest scaffoldRequest() {
  FrameRequest request{};
  PlotCommand command{};
  command.windowId = 1;
  command.kind = PlotKind::Scaffold;
  command.destination = {0.0f, 0.0f, 32.0f, 24.0f};
  command.targetWidth = 32;
  command.targetHeight = 24;
  command.targetPixelFormat = 0;
  command.viewRevision = 1u;
  command.unavailableReason = "runtime-test-scaffold";
  const std::array<ChromaspaceMetal::FrameVectorVertex, 3> vertices{{
      {0.0f, 0.0f, 0.1f, 0.2f, 0.3f, 1.0f},
      {32.0f, 0.0f, 0.1f, 0.2f, 0.3f, 1.0f},
      {0.0f, 24.0f, 0.1f, 0.2f, 0.3f, 1.0f},
  }};
  assert(request.appendVectorVertices(vertices.data(), vertices.size(), &command));
  assert(request.append(command));
  return request;
}

ChromaspaceMetal::ImportedSourceTexture residentSource() {
  ChromaspaceMetal::ImportedSourceTexture value{};
  value.sourceId = 1u;
  value.senderId = "runtime-test-producer";
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

FrameRequest residentRasterRequest() {
  FrameRequest request{};
  request.hasResidentSource = true;
  request.residentSource = residentSource();
  PlotCommand command{};
  command.windowId = 11;
  command.kind = PlotKind::ResidentRaster;
  command.destination = {0.0f, 0.0f, 40.0f, 32.0f};
  command.targetWidth = 40;
  command.targetHeight = 32;
  command.targetPixelFormat = 0;
  command.viewRevision = 1u;
  command.contentRevision = 1u;
  command.raster.pointCount = 1;
  command.raster.basePointCount = 1;
  command.raster.sourceWidth = 16;
  command.raster.sourceHeight = 16;
  command.raster.sampleCountX = 1;
  command.point.pointCount = 1;
  command.point.width = 40;
  command.point.height = 32;
  assert(request.append(command));
  return request;
}

FrameBatch baselineBatch() {
  FrameBatch batch{};
  batch.compositeOverlayRects.push_back(
      {1.0f, 2.0f, 3.0f, 4.0f, 0.1f, 0.2f, 0.3f, 1.0f});
  return batch;
}

struct Fixture {
  ExecutorState executor{};
  RendererState renderer{};
  ResourceState resource{};
  FrameExecutorBackend executorTable;
  RendererBackend rendererTable;
  RuntimeResourceBackend resourceTable;
  Runtime runtime;

  explicit Fixture(uint32_t maxRecreationAttempts = 2u)
      : executorTable(executorBackend(&executor)),
        rendererTable(rendererBackend(&renderer)),
        resourceTable(resourceBackend(&resource)),
        runtime(&executorTable,
                &rendererTable,
                &resourceTable,
                recoveryConfig(maxRecreationAttempts)) {}

  static ChromaspaceFrameRecoveryPolicy::Config recoveryConfig(
      uint32_t maxRecreationAttempts) {
  ChromaspaceFrameRecoveryPolicy::Config config{};
  config.maxRecreationAttempts = maxRecreationAttempts;
  config.baseBackoffMilliseconds = 1u;
  config.maxBackoffMilliseconds = 4u;
    return config;
  }
};

void assertRequestUnchanged(const FrameRequest& before,
                            const FrameRequest& after) {
  assert(before.commandCount == after.commandCount);
  assert(before.frameRevision == after.frameRevision);
  assert(before.vectorVertexArena.size() == after.vectorVertexArena.size());
  for (std::size_t i = 0; i < before.vectorVertexArena.size(); ++i) {
    assert(before.vectorVertexArena[i].x == after.vectorVertexArena[i].x);
    assert(before.vectorVertexArena[i].y == after.vectorVertexArena[i].y);
    assert(before.vectorVertexArena[i].r == after.vectorVertexArena[i].r);
    assert(before.vectorVertexArena[i].g == after.vectorVertexArena[i].g);
    assert(before.vectorVertexArena[i].b == after.vectorVertexArena[i].b);
    assert(before.vectorVertexArena[i].a == after.vectorVertexArena[i].a);
  }
  for (std::size_t i = 0; i < before.commandCount; ++i) {
    assert(before.commands[i].windowId == after.commands[i].windowId);
    assert(before.commands[i].kind == after.commands[i].kind);
    assert(before.commands[i].vectorVertexOffset ==
           after.commands[i].vectorVertexOffset);
    assert(before.commands[i].vectorVertexCount ==
           after.commands[i].vectorVertexCount);
    assert(before.commands[i].unavailableReason ==
           after.commands[i].unavailableReason);
  }
}

void createRuntime(Runtime* runtime) {
  static const std::array<unsigned char, 16> pixels{{
      0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
      8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}};
  const CpuTextAtlasPayload atlas{4, 4, pixels.data(), pixels.size()};
  assert(runtime->create(reinterpret_cast<void*>(0x1), {64, 48, 2.0f}, atlas));
  assert(runtime->generation() == 1u);
  assert(runtime->textAtlasId() != 0u);
  assert(runtime->runtimeContextId() != 0u);
  assert(runtime->deviceRegistryId() != 0u);
}

void testMemoryPressureMappingAndRuntimeStability() {
  Fixture fixture;
  Runtime& runtime = fixture.runtime;
  const auto invalidBefore = runtime.handleMemoryPressure(
      static_cast<MemoryPressureLevel>(0xffu));
  assert(invalidBefore.status == MemoryPressureStatus::InvalidLevel);
  assert(!invalidBefore.redrawRequired && fixture.executor.createCalls == 0);
  const auto notReady = runtime.handleMemoryPressure(MemoryPressureLevel::Normal);
  assert(notReady.status == MemoryPressureStatus::RuntimeNotReady);

  createRuntime(&runtime);
  const uint64_t generation = runtime.generation();
  const uint64_t compositorId = runtime.compositorId();
  const uint64_t runtimeContextId = runtime.runtimeContextId();
  const uint64_t deviceRegistryId = runtime.deviceRegistryId();
  const uint64_t atlasId = runtime.textAtlasId();
  const int createCalls = fixture.executor.createCalls;
  const int destroyCalls = fixture.executor.destroyCalls;

  const auto normal = runtime.handleMemoryPressure(MemoryPressureLevel::Normal);
  assert(normal.accepted() && !normal.redrawRequired);
  assert(normal.rendererTrim.accepted() &&
         normal.rendererTrim.before.surfaceCount == 0u &&
         normal.rendererTrim.after.surfaceCount == 0u &&
         normal.rendererTrim.releasedSurfaceCount == 0u &&
         normal.rendererTrim.releasedDerivedCacheCount == 0u);
  assert(runtime.generation() == generation && runtime.compositorId() == compositorId &&
         runtime.runtimeContextId() == runtimeContextId &&
         runtime.deviceRegistryId() == deviceRegistryId &&
         runtime.textAtlasId() == atlasId);

  const auto warningEmpty =
      runtime.handleMemoryPressure(MemoryPressureLevel::Warning);
  assert(warningEmpty.accepted() && !warningEmpty.redrawRequired &&
         warningEmpty.rendererTrim.accepted() &&
         warningEmpty.rendererTrim.level == TrimLevel::DerivedOnly);
  assert(warningEmpty.rendererTrim.before.surfaceCount == 0u &&
         warningEmpty.rendererTrim.after.surfaceCount == 0u);

  FrameRequest request = residentRasterRequest();
  FrameBatch baseline = baselineBatch();
  auto outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented);
  assert(runtime.plotResourceCount() == 1u);
  const auto warning =
      runtime.handleMemoryPressure(MemoryPressureLevel::Warning);
  assert(warning.accepted() && !warning.redrawRequired &&
         warning.rendererTrim.level == TrimLevel::DerivedOnly);
  assert(warning.rendererTrim.before.surfaceCount == 1u &&
         warning.rendererTrim.after.surfaceCount == 1u &&
         warning.rendererTrim.before.derivedCacheCount == 1u &&
         warning.rendererTrim.after.derivedCacheCount == 0u);
  assert(runtime.plotResourceCount() == 1u);

  const auto critical =
      runtime.handleMemoryPressure(MemoryPressureLevel::Critical);
  assert(critical.accepted() && critical.redrawRequired &&
         critical.rendererTrim.level == TrimLevel::AllPlotResources);
  assert(critical.rendererTrim.before.surfaceCount == 1u &&
         critical.rendererTrim.after.surfaceCount == 0u &&
         critical.rendererTrim.releasedSurfaceCount == 1u &&
         critical.rendererTrim.releasedSurfaceBytes == 40u * 32u * 8u);
  assert(runtime.plotResourceCount() == 0u);
  assert(runtime.generation() == generation && runtime.compositorId() == compositorId &&
         runtime.runtimeContextId() == runtimeContextId &&
         runtime.deviceRegistryId() == deviceRegistryId &&
         runtime.textAtlasId() == atlasId &&
         fixture.executor.createCalls == createCalls &&
         fixture.executor.destroyCalls == destroyCalls);

  const auto repeatedCritical =
      runtime.handleMemoryPressure(MemoryPressureLevel::Critical);
  assert(repeatedCritical.accepted() && repeatedCritical.redrawRequired &&
         repeatedCritical.rendererTrim.releasedSurfaceCount == 0u &&
         repeatedCritical.rendererTrim.releasedDerivedCacheCount == 0u);

  // Critical trim is non-terminal and the same runtime can render again.
  baseline = baselineBatch();
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented && runtime.plotResourceCount() == 1u);
  runtime.shutdown();
}

void testMemoryPressureRejectsPendingRecreationAndActiveTransaction() {
  Fixture fixture;
  createRuntime(&fixture.runtime);
  fixture.executor.failSubmit = true;
  fixture.executor.submitFailure = Failure::PriorGpuSubmissionFailure;
  fixture.executor.failCreate = true;
  FrameRequest request = scaffoldRequest();
  FrameBatch baseline = baselineBatch();
  const auto recoveryOutcome =
      fixture.runtime.render(request, baseline, Visibility::Visible);
  assert(recoveryOutcome.kind == OutcomeKind::RetryLater);
  const auto pending = fixture.runtime.handleMemoryPressure(
      MemoryPressureLevel::Warning);
  assert(pending.status == MemoryPressureStatus::RecreationPending &&
         !pending.redrawRequired);

  // A renderer transaction is active during the executor begin callback.  The
  // nested typed call must reject without reclaiming the staged/committed set.
  Fixture activeFixture;
  createRuntime(&activeFixture.runtime);
  activeFixture.executor.runtimeForPressureProbe = &activeFixture.runtime;
  activeFixture.executor.probePressureDuringBegin = true;
  FrameRequest activeRequest = scaffoldRequest();
  FrameBatch activeBaseline = baselineBatch();
  const auto outcome = activeFixture.runtime.render(
      activeRequest, activeBaseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented);
  assert(activeFixture.executor.pressureProbe.status ==
         MemoryPressureStatus::TransactionActive);
  assert(!activeFixture.executor.pressureProbe.redrawRequired);
  assert(activeFixture.runtime.plotResourceCount() == 1u);
  activeFixture.runtime.shutdown();
}

void testInitialCreateRollback() {
  Fixture fixture;
  fixture.executor.failCreate = true;
  static const std::array<unsigned char, 16> pixels{{
      0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
      8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}};
  const CpuTextAtlasPayload atlas{4, 4, pixels.data(), pixels.size()};
  std::string error;
  assert(!fixture.runtime.create(reinterpret_cast<void*>(0x1), {64, 48, 2.0f},
                                 atlas, &error));
  assert(!fixture.runtime.ready());
  assert(fixture.runtime.generation() == 0u);
  fixture.executor.failCreate = false;
  assert(fixture.runtime.create(reinterpret_cast<void*>(0x1), {64, 48, 2.0f},
                                atlas, &error));
  assert(fixture.runtime.generation() == 1u);
  fixture.runtime.shutdown();

  // A partial resource seam must fail before allocation rather than creating
  // an atlas that has no corresponding release callback.
  Fixture partialFixture;
  RuntimeResourceBackend partialResource{
      &partialFixture.resource, resourceCreate, nullptr};
  Runtime partialRuntime(&partialFixture.executorTable,
                         &partialFixture.rendererTable, &partialResource,
                         Fixture::recoveryConfig(2u));
  partialFixture.executor.failCreate = false;
  assert(!partialRuntime.create(reinterpret_cast<void*>(0x1), {64, 48, 2.0f},
                                atlas, &error));
  assert(partialFixture.resource.createCalls == 0);
}

void testPresentationAndInputIsolation() {
  Fixture fixture;
  Runtime& runtime = fixture.runtime;
  ExecutorState& executor = fixture.executor;
  RendererState& renderer = fixture.renderer;
  ResourceState& resource = fixture.resource;
  createRuntime(&runtime);
  FrameRequest request = scaffoldRequest();
  FrameBatch baseline = baselineBatch();
  const FrameRequest requestBefore = request;
  const FrameBatch baselineBefore = baseline;
  auto outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented);
  assertRequestUnchanged(requestBefore, request);
  assert(baseline.compositeItems.size() == baselineBefore.compositeItems.size());
  assert(baseline.compositeOverlayRects.size() ==
         baselineBefore.compositeOverlayRects.size());
  assert(executor.submittedItemCounts.size() == 1u);
  assert(executor.submittedItemCounts.front() == 1u);
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented);
  assert(executor.submittedItemCounts.back() == 1u);
  runtime.shutdown();
  assert(resource.releaseCalls == 1);
}

void testRollbackAndTypedRecovery() {
  Fixture fixture;
  Runtime& runtime = fixture.runtime;
  ExecutorState& executor = fixture.executor;
  RendererState& renderer = fixture.renderer;
  ResourceState& resource = fixture.resource;
  createRuntime(&runtime);
  FrameRequest request = scaffoldRequest();
  FrameBatch baseline = baselineBatch();

  renderer.failVectors = true;
  auto outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::TerminalFailure);
  assert(outcome.failure == Failure::EncodingFailure);
  assert(runtime.plotResourceCount() == 0u);
  assert(renderer.releaseCalls == 1);

  renderer.failVectors = false;
  executor.failSubmit = true;
  executor.submitFailure = Failure::BackpressureTimeout;
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::RetryLater);
  assert(outcome.waitMilliseconds > 0u);
  assert(runtime.plotResourceCount() == 0u);
  assert(executor.abandonCalls >= 2);
  outcome = runtime.render(request, baseline, Visibility::Occluded);
  // The occluded transient is suspended without consuming another retry.
  assert(outcome.kind == OutcomeKind::SuspendUntilVisible);
  runtime.shutdown();
}

void testRecreationAtlasAndExternalState() {
  Fixture fixture(2u);
  Runtime& runtime = fixture.runtime;
  ExecutorState& executor = fixture.executor;
  RendererState& renderer = fixture.renderer;
  ResourceState& resource = fixture.resource;
  createRuntime(&runtime);
  const uint64_t oldGeneration = runtime.generation();
  const uint64_t oldRuntimeContextId = runtime.runtimeContextId();
  const uint64_t oldDeviceRegistryId = runtime.deviceRegistryId();
  const uint64_t oldAtlas = runtime.textAtlasId();
  int residentSessionMarker = 17;
  FrameRequest request = scaffoldRequest();
  FrameBatch baseline = baselineBatch();
  executor.failSubmit = true;
  executor.submitFailure = Failure::PriorGpuSubmissionFailure;
  auto outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::RuntimeRecreated);
  assert(runtime.generation() == oldGeneration + 1u);
  assert(runtime.runtimeContextId() != oldRuntimeContextId);
  assert(runtime.deviceRegistryId() == oldDeviceRegistryId);
  assert(runtime.textAtlasId() != oldAtlas);
  assert(residentSessionMarker == 17);
  executor.failSubmit = false;
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::Presented);

  executor.failSubmit = true;
  executor.submitFailure = Failure::PriorGpuSubmissionFailure;
  executor.failCreate = true;
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::RetryLater);
  outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::TerminalFailure);
  runtime.shutdown();
}

void testRecreationRejectsContextIdReuse() {
  Fixture fixture(1u);
  Runtime& runtime = fixture.runtime;
  createRuntime(&runtime);
  const uint64_t oldGeneration = runtime.generation();
  const uint64_t oldRuntimeContextId = runtime.runtimeContextId();
  const uint64_t oldDeviceRegistryId = runtime.deviceRegistryId();
  const int createCalls = fixture.executor.createCalls;
  const int destroyCalls = fixture.executor.destroyCalls;
  const int atlasCreateCalls = fixture.resource.createCalls;
  const int atlasReleaseCalls = fixture.resource.releaseCalls;

  fixture.executor.reuseRuntimeContextId = true;
  fixture.executor.failSubmit = true;
  fixture.executor.submitFailure = Failure::PriorGpuSubmissionFailure;
  const auto outcome = runtime.render(scaffoldRequest(), baselineBatch(),
                                     Visibility::Visible);
  assert(outcome.kind == OutcomeKind::TerminalFailure);
  assert(!runtime.ready());
  assert(runtime.generation() == oldGeneration);
  assert(runtime.runtimeContextId() == 0u);
  assert(runtime.deviceRegistryId() == 0u);
  assert(runtime.textAtlasId() == 0u);
  assert(oldRuntimeContextId != 0u && oldDeviceRegistryId != 0u);
  assert(fixture.executor.createCalls == createCalls + 1);
  assert(fixture.executor.destroyCalls == destroyCalls + 2);
  // The rejected compositor is torn down before atlas creation, so only the
  // original atlas is released and no candidate atlas leaks.
  assert(fixture.resource.createCalls == atlasCreateCalls);
  assert(fixture.resource.releaseCalls == atlasReleaseCalls + 1);
  const auto terminal = runtime.render(scaffoldRequest(), baselineBatch(),
                                       Visibility::Visible);
  assert(terminal.kind == OutcomeKind::TerminalFailure);
  runtime.shutdown();
}

void testResizeUsesTypedRecoveryAndLatestViewport() {
  Fixture fixture;
  Runtime& runtime = fixture.runtime;
  createRuntime(&runtime);
  fixture.executor.failResize = true;
  auto outcome = runtime.resize({96, 72, 2.0f}, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::RuntimeRecreated);
  assert(runtime.latestViewport().drawableWidth == 96);
  assert(runtime.latestViewport().drawableHeight == 72);
  assert(runtime.compositor().drawableWidth == 96);
  assert(runtime.compositor().drawableHeight == 72);
  fixture.executor.failResize = false;
  outcome = runtime.resize({112, 80, 2.0f}, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::ViewportUpdated);
  assert(runtime.latestViewport().drawableWidth == 112);
  assert(runtime.compositor().drawableWidth == 112);
  runtime.shutdown();
}

void testTerminalInvariant() {
  Fixture fixture;
  Runtime& runtime = fixture.runtime;
  ExecutorState& executor = fixture.executor;
  RendererState& renderer = fixture.renderer;
  ResourceState& resource = fixture.resource;
  createRuntime(&runtime);
  FrameRequest request = scaffoldRequest();
  FrameBatch baseline = baselineBatch();
  baseline.compositeVectorVertices.push_back(
      {5.0f, 6.0f, 0.1f, 0.2f, 0.3f, 1.0f});
  const auto outcome = runtime.render(request, baseline, Visibility::Visible);
  assert(outcome.kind == OutcomeKind::TerminalFailure);
  assert(outcome.failure == Failure::InvariantViolation);
  assert(executor.createCalls == 1);
  runtime.shutdown();
}

}  // namespace

int main() {
  testMemoryPressureMappingAndRuntimeStability();
  testMemoryPressureRejectsPendingRecreationAndActiveTransaction();
  testInitialCreateRollback();
  testPresentationAndInputIsolation();
  testRollbackAndTypedRecovery();
  testRecreationAtlasAndExternalState();
  testRecreationRejectsContextIdReuse();
  testResizeUsesTypedRecoveryAndLatestViewport();
  testTerminalInvariant();
  return 0;
}
