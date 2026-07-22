#pragma once

#include "../../../src/metal/ChromaspaceSourceExchangeState.h"
#include "ChromaspaceMetalFrameFailure.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceMetal {

struct ProbeResult {
  bool available = false;
  bool queueReady = false;
  const char* deviceName = "";
};

struct ResidentReadiness {
  bool deviceReady = false;
  bool queueReady = false;
  bool contextReady = false;
  uint64_t deviceRegistryId = 0;
  bool rasterSourceTextureReady = false;
  bool analyticalScopeReady = false;
  bool histogramSurfaceReady = false;
  bool waveformSurfaceReady = false;
  bool glossFieldCacheReady = false;
  bool glossFieldSurfaceReady = false;
  bool glossProjectionSurfaceReady = false;
  bool plotSurfaceReady = false;
  bool plotSurfaceVectorReady = false;
  bool rasterPointSurfaceReady = false;
  bool sourceSignalSurfaceReady = false;
  bool frameSurfaceCompositeReady = false;
  bool frameUiVectorReady = false;
  bool frameTextReady = false;
  std::string deviceName;
  std::string missing;
};

struct RemapUniforms {
  int plotMode = 0;
  int circularHsl = 0;
  int circularHsv = 0;
  int normConeNormalized = 1;
  int showOverflow = 0;
  int highlightOverflow = 1;
  int chromaticityInputTransfer = 0;
  int chromaticityReferenceBasis = 0;
  float chromaticityWhiteX = 1.0f / 3.0f;
  float chromaticityWhiteY = 1.0f / 3.0f;
  float chromaticityRgbToXyz[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
  float chromaticityXyzToRgb[9] = {
      1.0f, 0.0f, 0.0f,
      0.0f, 1.0f, 0.0f,
      0.0f, 0.0f, 1.0f};
};

struct OverlayRequest {
  int cubeSize = 25;
  int ramp = 0;
  int useInputPoints = 0;
  int pointCount = 0;
  float colorSaturation = 1.18f;
  RemapUniforms remap;
};

struct InputRequest {
  int pointCount = 0;
  int inputStride = 3;
  int glossView = 0;
  float sourceAspect = 16.0f / 9.0f;
  float glossLiftScale = 1.0f;
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  float colorSaturation = 1.18f;
  RemapUniforms remap;
};

struct RasterSourceRequest {
  int pointCount = 0;
  int basePointCount = 0;
  int sourceWidth = 0;
  int sourceHeight = 0;
  int sampleStride = 1;
  int sampleCountX = 0;
  int pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  float sourceAspect = 16.0f / 9.0f;
  float glossLiftScale = 1.0f;
  float pointAlphaScale = 1.0f;
  float denseAlphaBias = 0.0f;
  float colorSaturation = 1.18f;
  int plotLinear = 0;
  int plotLinearTransfer = 0;
  int excludeIdentityData = 0;
  int isolateIdentityData = 0;
  int readIdentityPlot = 0;
  int readGrayRamp = 0;
  int identityCubeY1 = -1;
  int identityCubeY2 = -1;
  int identityRampY1 = -1;
  int identityRampY2 = -1;
  int identityCubeAppendOffset = 0;
  int identityCubeAppendCount = 0;
  int identityCubeAppendY1 = -1;
  int identityCubeAppendY2 = -1;
  int identityCubeAppendRowStep = 1;
  int identityCubeAppendXStep = 1;
  int identityRampAppendOffset = 0;
  int identityRampAppendCount = 0;
  int identityRampAppendY1 = -1;
  int identityRampAppendY2 = -1;
  int identityRampAppendRowStep = 1;
  int identityRampAppendXStep = 1;
  int occupancyFill = 0;
  int occupancyAppendOffset = 0;
  int occupancyAppendCount = 0;
  int occupancyCandidateCount = 0;
  int lassoEnabled = 0;
  int lassoStrokeCount = 0;
  int lassoPointCount = 0;
  int lassoStrokeFirst[16] = {};
  int lassoStrokeCountPerStroke[16] = {};
  int lassoStrokeSubtract[16] = {};
  float lassoX[256] = {};
  float lassoY[256] = {};
  int cubeSlicingEnabled = 0;
  int neutralRadiusEnabled = 0;
  float neutralRadius = 1.0f;
  int cubeSliceRed = 0;
  int cubeSliceYellow = 0;
  int cubeSliceGreen = 0;
  int cubeSliceCyan = 0;
  int cubeSliceBlue = 0;
  int cubeSliceMagenta = 0;
  RemapUniforms remap;
};

struct InputSampleRequest {
  int fullPointCount = 0;
  int visiblePointCount = 0;
};

struct GlossFieldRequest {
  int gridWidth = 96;
  int gridHeight = 96;
  int showOverflow = 0;
  int neighborhoodChoice = 1;
};

struct GlossFieldResult {
  int gridWidth = 0;
  int gridHeight = 0;
  std::vector<float> occupancy;
  std::vector<float> meanRgb;
  std::vector<float> carrierY;
  std::vector<float> carrierMax;
  std::vector<float> carrierMin;
  std::vector<float> neutrality;
  std::vector<float> body;
  std::vector<float> signal;
  std::vector<float> positive;
  std::vector<float> negative;
  std::vector<float> boundary;
  std::vector<float> congruence;
  std::vector<float> confidence;
};

struct GlossFieldCache {
  uint64_t cacheId = 0;
  // Zero is reserved for the separately compiled compatibility path. Native
  // resident records are owned by one live compositor generation.
  uint64_t ownerCompositorId = 0;
  int gridWidth = 0;
  int gridHeight = 0;
  uint64_t builtSerial = 0;
  // Exact bytes retained by the native Gloss field record.  This remains a
  // compatibility-shaped handle while ownership moves to ResidentDerivedCache.
  size_t byteSize = 0;
  bool available = false;
};

enum class GlossFieldCacheState {
  Missing = 0,
  // Encoded by a frame that has not yet entered the Metal queue.
  Pending,
  // Queue-visible from a submitted frame or retained as the last completed version.
  Ready,
};

enum class ResidentDerivedFamily : uint8_t {
  Histogram = 0,
  Waveform,
  RasterPointCloud,
  GlossField,
};

// Opaque handle for GPU-derived data that is independent of any plot window
// or final render target. Family-specific native records currently own scope
// density/range buffers or compact raster point-position/attribute buffers
// plus GPU-authored indirect draw arguments. The
// renderer policy owns identity, byte budgets, eviction, and transactions.
struct ResidentDerivedCache {
  uint64_t cacheId = 0;
  // Zero is reserved for the separately compiled compatibility path. Native
  // resident records are owned by one live compositor generation.
  uint64_t ownerCompositorId = 0;
  uint64_t builtSerial = 0;
  size_t byteSize = 0;
  ResidentDerivedFamily family = ResidentDerivedFamily::Histogram;
  bool available = false;
};

enum class ResidentDerivedCacheState : uint8_t {
  Missing = 0,
  // Encoded by a frame that has not yet entered the Metal queue.
  Pending,
  // Queue-visible from a submitted frame or retained after completion.
  Ready,
};

// Compatibility names while callers migrate to the generalized derived-data
// vocabulary. These are exact aliases, not a second cache implementation.
using ScopeDerivedFamily = ResidentDerivedFamily;
using ScopeDerivedCache = ResidentDerivedCache;
using ScopeDerivedCacheState = ResidentDerivedCacheState;

struct ScopeDensityRequest {
  int pointCount = 0;
  int waveform = 1;
  int scopeMode = 0;
  int width = 768;
  int height = 512;
  float rangeMin = 0.0f;
  float invRange = 1.0f;
  int excludeOverflow = 1;
  int onlyOverflow = 0;
  int channelCount = 3;
  int lumaMethod = 0;
};

struct ScopeRangeRequest {
  int pointCount = 0;
  int waveform = 1;
  int scopeMode = 0;
  int includeRed = 1;
  int includeGreen = 1;
  int includeBlue = 1;
  int includeLuma = 0;
  int includeOverflow = 1;
  int lumaMethod = 0;
  int previousRangeValid = 0;
  float previousRangeMin = 0.0f;
  float previousRangeMax = 1.0f;
};

struct ScopeRangeResult {
  float minValue = 0.0f;
  float maxValue = 1.0f;
  uint32_t validCount = 0;
};

struct HistogramSurfaceRequest {
  int pointCount = 0;
  int scopeMode = 0;  // 0=RGB overlay, 1=luma.
  int width = 1024;
  int height = 512;
  float rangeMin = 0.0f;
  float invRange = 1.0f;
  int showOverflow = 1;
  int highlightOverflow = 1;
  int lumaMethod = 0;
  int useGpuAutoRange = 0;
  ScopeRangeRequest autoRange{};
};

struct WaveformSurfaceRequest {
  int pointCount = 0;
  int scopeMode = 0;  // 0=RGB overlay, 1=RGB parade, 2=luma.
  int width = 768;
  int height = 512;
  float rangeMin = 0.0f;
  float invRange = 1.0f;
  int showOverflow = 1;
  int highlightOverflow = 1;
  int lumaMethod = 0;
  int includeRed = 1;
  int includeGreen = 1;
  int includeBlue = 1;
  int includeLuma = 0;
  float pointBrightness = 0.4f;
  float colorSaturation = 0.75f;
  float coverageAlpha = 1.0f;
  int useGpuAutoRange = 0;
  ScopeRangeRequest autoRange{};
};

struct RasterPointSurfaceRequest {
  int pointCount = 0;
  int width = 1024;
  int height = 768;
  float pointRadiusPixels = 2.0f;
  float backgroundR = 0.035f;
  float backgroundG = 0.040f;
  float backgroundB = 0.052f;
  float backgroundA = 1.0f;
  float modelView[16] = {};
  float projection[16] = {};
};

struct GlossFieldSurfaceRequest {
  int width = 768;
  int height = 768;
  int algorithm = 0;  // 0=Candidate 1, 1=Candidate 2.
  int colorMode = 0;  // 0=Semantic signal, 1=Source hue tint.
  int debugMode = 0;  // 0=Signal, 1=max, 2=Y, 3=min, 4=neutrality.
  int diagnosticMode = 0;  // 0=Off, 1=Confidence, 2=Ambiguity.
  float colorSaturation = 2.0f;
  float glossBodyOpacity = 0.10f;
  float glossHighlightOpacity = 0.42f;
  float glossLiftScale = 1.0f;
};

struct GlossProjectionSurfaceRequest {
  int width = 1024;
  int height = 768;
  int algorithm = 0;  // 0=Candidate 1, 1=Candidate 2.
  int colorMode = 0;  // 0=Semantic signal, 1=Source hue tint.
  int debugMode = 0;  // 0=Signal, 1=max, 2=Y, 3=min, 4=neutrality.
  int diagnosticMode = 0;  // 0=Off, 1=Confidence, 2=Ambiguity.
  float sourceAspect = 16.0f / 9.0f;
  float colorSaturation = 2.0f;
  float glossBodyOpacity = 0.10f;
  float glossHighlightOpacity = 0.42f;
  float glossLiftScale = 1.0f;
  float pointRadiusPixels = 2.0f;
  float modelView[16] = {};
  float projection[16] = {};
};

struct PlotSurface {
  // Opaque viewer-runtime handle. This is never an IOSurface ID.
  uint32_t surfaceId = 0;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  size_t byteSize = 0;
};

struct FrameCompositor {
  uint64_t compositorId = 0;
  int drawableWidth = 0;
  int drawableHeight = 0;
  float contentsScale = 1.0f;
  // Opaque identity of the backend/runtime generation that owns this
  // compositor.  It is deliberately distinct from the compositor handle so
  // stale submissions cannot cross a runtime recreation boundary.
  uint64_t runtimeContextId = 0;
  // Opaque registry identity of the Metal device selected by the runtime.
  uint64_t deviceRegistryId = 0;
};

struct FrameSubmission {
  // Opaque single-use token: begin, encode zero or more passes, then submit or
  // abandon. Successful submit and abandon both reset the caller's token.
  uint64_t submissionId = 0;
  uint64_t compositorId = 0;
  uint64_t runtimeContextId = 0;
  uint64_t deviceRegistryId = 0;
};

struct FrameTransientMemoryStats {
  bool available = false;
  std::size_t activeSubmissionCount = 0u;
  std::size_t encodingCount = 0u;
  std::size_t submittedCount = 0u;
  uint64_t inFlightReservedBytes = 0u;
  uint64_t inFlightLogicalBytes = 0u;
  uint64_t peakInFlightReservedBytes = 0u;
  uint64_t peakInFlightLogicalBytes = 0u;
  std::size_t peakActiveSubmissionCount = 0u;
  uint64_t maxInFlightBytes = 0u;
  uint64_t maxBytesPerSubmission = 0u;
  uint32_t maxSubmissions = 0u;
};

struct FrameCompletionStats {
  bool available = false;
  uint64_t submittedSerial = 0u;
  uint64_t completedSerial = 0u;
  uint64_t failedSubmissionCount = 0u;
  uint64_t timedSubmissionCount = 0u;
  uint64_t untimedSubmissionCount = 0u;
  double accumulatedGpuSeconds = 0.0;
  double maximumGpuSeconds = 0.0;
  std::string lastSubmissionError;
};

struct SharedSourceImportRequest {
  // Bridged MTLSharedTextureHandle* and MTLSharedEventHandle*. They are used
  // only during import; the registry retains the reconstructed Metal objects.
  void* sharedTextureHandle = nullptr;
  void* sharedEventHandle = nullptr;
  std::string senderId;
  uint64_t deviceRegistryId = 0;
  uint64_t senderGeneration = 0;
  uint64_t sequence = 0;
  uint32_t slotIndex = 0;
  uint64_t slotGeneration = 0;
  uint64_t readyValue = 0;
  uint64_t contentHash = 0;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
  size_t bytesPerRow = 0;
  size_t byteSize = 0;
  ChromaspaceSourceExchange::SourceSemanticMetadata semantics;
};

struct ImportedSourceTexture {
  // Opaque viewer-runtime handle. This is never an IOSurface ID or Metal
  // object pointer.
  uint64_t sourceId = 0;
  std::string senderId;
  uint64_t deviceRegistryId = 0;
  uint64_t senderGeneration = 0;
  uint64_t sequence = 0;
  uint32_t slotIndex = 0;
  uint64_t slotGeneration = 0;
  uint64_t readyValue = 0;
  uint64_t contentHash = 0;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;
  size_t bytesPerRow = 0;
  size_t byteSize = 0;
  ChromaspaceSourceExchange::SourceSemanticMetadata semantics;
};

using ImportedSourceRetirementCallback = void (*)(void* context);

struct FrameTextAtlas {
  uint64_t atlasId = 0;
  int width = 0;
  int height = 0;
};

struct SurfaceCompositeItem {
  uint32_t surfaceId = 0;
  int surfaceWidth = 0;
  int surfaceHeight = 0;
  int surfacePixelFormat = 0;
  float dstX = 0.0f;
  float dstY = 0.0f;
  float dstW = 0.0f;
  float dstH = 0.0f;
  float opacity = 1.0f;
};

struct FrameOverlayRect {
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
  float a = 1.0f;
};

struct FrameVectorVertex {
  float x = 0.0f;
  float y = 0.0f;
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
  float a = 1.0f;
};

struct FrameTextVertex {
  float x = 0.0f;
  float y = 0.0f;
  float u = 0.0f;
  float v = 0.0f;
};

struct FrameTextRun {
  uint64_t atlasId = 0;
  uint32_t firstVertex = 0;
  uint32_t vertexCount = 0;
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
  float a = 1.0f;
  float clipX = 0.0f;
  float clipY = 0.0f;
  float clipW = 0.0f;
  float clipH = 0.0f;
  uint32_t clipEnabled = 0;
};

enum ModifierFlags : uint32_t {
  ModifierFlagShift = 1u << 0,
  ModifierFlagControl = 1u << 1,
  ModifierFlagAlt = 1u << 2,
  ModifierFlagSuper = 1u << 3,
};

bool activateWindow(void* nativeWindow);
uint32_t currentModifierFlags();
ProbeResult probe();
ResidentReadiness residentReadiness();
// Returns readiness for the backend context that owns the compositor.  The
// zero-argument overload remains the compatibility probe for callers that do
// not yet have a compositor handle.
ResidentReadiness residentReadiness(uint64_t compositorId);
bool createFrameCompositor(void* nativeWindow,
                           int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           FrameCompositor* outCompositor,
                           std::string* error);
bool resizeFrameCompositor(uint64_t compositorId,
                           int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           std::string* error);
// Control-plane fence used only for sender switches and orderly shutdown.
// Normal frame production must never call this.
bool drainFrameCompositor(uint64_t compositorId,
                          uint32_t timeoutMilliseconds,
                          std::string* error);
// Lock-safe, non-waiting pressure snapshot. This never changes submission
// state and is suitable for qualification/diagnostic sampling.
bool frameTransientMemoryStats(uint64_t compositorId,
                               FrameTransientMemoryStats* outStats,
                               std::string* error);
// Lock-safe completion/timing snapshot. After drain, submittedSerial must
// equal completedSerial and failedSubmissionCount must be zero for a clean
// qualification epoch.
bool frameCompletionStats(uint64_t compositorId,
                          FrameCompletionStats* outStats,
                          std::string* error);
bool clearFrameCompositor(uint64_t compositorId,
                          float r,
                          float g,
                          float b,
                          float a,
                          std::string* error);
bool beginFrameSubmission(uint64_t compositorId,
                          FrameSubmission* outSubmission,
                          std::string* error,
                          ChromaspaceMetalFrameFailure::Kind* failure = nullptr);
bool submitFrameSubmissionSurfacesOverlayRectsAndText(
    FrameSubmission* submission,
    const SurfaceCompositeItem* items,
    size_t itemCount,
    const FrameOverlayRect* overlayRects,
    size_t overlayRectCount,
    const FrameVectorVertex* vectorVertices,
    size_t vectorVertexCount,
    const FrameTextVertex* textVertices,
    size_t textVertexCount,
    const FrameTextRun* textRuns,
    size_t textRunCount,
    float clearR,
    float clearG,
    float clearB,
    float clearA,
    std::string* error,
    ChromaspaceMetalFrameFailure::Kind* failure = nullptr);
void abandonFrameSubmission(FrameSubmission* submission);
bool compositeFrameSurfaces(uint64_t compositorId,
                            const SurfaceCompositeItem* items,
                            size_t itemCount,
                            float clearR,
                            float clearG,
                            float clearB,
                            float clearA,
                            std::string* error);
bool compositeFrameSurfacesAndOverlayRects(uint64_t compositorId,
                                           const SurfaceCompositeItem* items,
                                           size_t itemCount,
                                           const FrameOverlayRect* overlayRects,
                                           size_t overlayRectCount,
                                           float clearR,
                                           float clearG,
                                           float clearB,
                                           float clearA,
                                           std::string* error);
bool compositeFrameSurfacesOverlayRectsAndText(uint64_t compositorId,
                                               const SurfaceCompositeItem* items,
                                               size_t itemCount,
                                               const FrameOverlayRect* overlayRects,
                                               size_t overlayRectCount,
                                               const FrameVectorVertex* vectorVertices,
                                               size_t vectorVertexCount,
                                               const FrameTextVertex* textVertices,
                                               size_t textVertexCount,
                                               const FrameTextRun* textRuns,
                                               size_t textRunCount,
                                               float clearR,
                                               float clearG,
                                               float clearB,
                                               float clearA,
                                               std::string* error);
void releaseFrameCompositor(uint64_t compositorId);
bool importSharedSourceTexture(const SharedSourceImportRequest& request,
                               ImportedSourceTexture* outSource,
                               std::string* error);
// Atomically prevents new frame submissions from acquiring sourceId. The
// callback runs exactly once after every already-recorded submission has
// completed or been abandoned, and may run inline. It must not call OFX host
// interfaces. Only after this callback may SourceExchange acknowledge Retired.
bool retireImportedSourceTexture(
    uint64_t sourceId,
    ImportedSourceRetirementCallback callback,
    void* callbackContext,
    std::string* error);
// Compatibility release with no notification. Cross-process transports must
// use retireImportedSourceTexture and wait for its callback.
void releaseImportedSourceTexture(uint64_t sourceId);
// Native atlases are scoped to a live compositor. Owner-zero atlases do not
// exist, and a compositor releases any atlas that its caller did not retire.
bool createFrameTextAtlas(uint64_t compositorId,
                          int width,
                          int height,
                          const unsigned char* alphaPixels,
                          FrameTextAtlas* outAtlas,
                          std::string* error);
void releaseFrameTextAtlas(uint64_t compositorId, uint64_t atlasId);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool bindIOSurfaceToOpenGLTexture(uint32_t surfaceId,
                                  int width,
                                  int height,
                                  int pixelFormat,
                                  uint32_t glTexture,
                                  std::string* error);
bool bindPlotSurfaceToOpenGLTexture(uint32_t surfaceId,
                                   int width,
                                   int height,
                                   int pixelFormat,
                                   uint32_t glTexture,
                                   std::string* error);
// Explicit IOSurface-backed compatibility target.
bool createPlotSurface(int width,
                       int height,
                       int pixelFormat,
                       PlotSurface* outSurface,
                       std::string* error);
#endif
// Normal in-process target backed by a private Metal texture. Creation,
// encoding, composition, and explicit release all require the same live
// compositor owner.
bool createPrivatePlotSurface(uint64_t compositorId,
                              int width,
                              int height,
                              int pixelFormat,
                              PlotSurface* outSurface,
                              std::string* error);
void releasePrivatePlotSurface(uint64_t compositorId, uint32_t surfaceId);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool clearPlotSurface(uint32_t surfaceId,
                      int width,
                      int height,
                      int pixelFormat,
                      float r,
                      float g,
                      float b,
                       float a,
                       std::string* error);
#endif
bool encodePlotSurfaceClear(const FrameSubmission& submission,
                            uint32_t surfaceId,
                            int width,
                            int height,
                            int pixelFormat,
                            float r,
                            float g,
                            float b,
                             float a,
                             std::string* error);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderPlotSurfaceVectorPrimitives(uint32_t surfaceId,
                                       int width,
                                       int height,
                                       int pixelFormat,
                                       const FrameVectorVertex* vertices,
                                       size_t vertexCount,
                                       bool clearBeforeDraw,
                                       float clearR,
                                       float clearG,
                                       float clearB,
                                        float clearA,
                                        std::string* error);
#endif
bool encodePlotSurfaceVectorPrimitives(const FrameSubmission& submission,
                                       uint32_t surfaceId,
                                       int width,
                                       int height,
                                       int pixelFormat,
                                       const FrameVectorVertex* vertices,
                                       size_t vertexCount,
                                       bool clearBeforeDraw,
                                       float clearR,
                                       float clearG,
                                       float clearB,
                                       float clearA,
                                       std::string* error);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderSourceSignalSurfaceFromIOSurface(uint32_t sourceSurfaceId,
                                            int sourceSurfaceWidth,
                                            int sourceSurfaceHeight,
                                            int sourceSurfacePixelFormat,
                                            uint32_t outputSurfaceId,
                                            int outputSurfaceWidth,
                                            int outputSurfaceHeight,
                                            int outputSurfacePixelFormat,
                                            std::string* error);
bool encodeSourceSignalSurfaceFromIOSurface(
    const FrameSubmission& submission,
    uint32_t sourceSurfaceId,
    int sourceSurfaceWidth,
    int sourceSurfaceHeight,
    int sourceSurfacePixelFormat,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error);
#endif
bool encodeSourceSignalSurfaceFromImportedTexture(
    const FrameSubmission& submission,
    uint64_t sourceId,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
     int outputSurfacePixelFormat,
     std::string* error);
void releasePlotSurface(uint32_t surfaceId);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildOverlayMesh(const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      std::vector<float>* outVerts,
                      std::vector<float>* outColors,
                      std::string* error);
bool buildInputMesh(const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    std::vector<float>* outVerts,
                    std::vector<float>* outColors,
                    std::string* error);
bool buildRasterSourceMesh(const RasterSourceRequest& request,
                           const void* sourceBytes,
                           size_t sourceByteCount,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error);
#endif
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildRasterSourceMeshFromIOSurface(const RasterSourceRequest& request,
                                        uint32_t surfaceId,
                                        int surfaceWidth,
                                        int surfaceHeight,
                                        int surfacePixelFormat,
                                        std::vector<float>* outVerts,
                                         std::vector<float>* outColors,
                                         std::string* error);
#endif
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildInputSampledMesh(const InputSampleRequest& request,
                           const std::vector<float>& fullVerts,
                           const std::vector<float>& fullColors,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error);
bool buildGlossField(const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     bool allowReadback,
                     GlossFieldResult* out,
                     std::string* error);
#endif
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildGlossFieldFromIOSurface(GlossFieldCache* cache,
                                  const RasterSourceRequest& rasterRequest,
                                  const GlossFieldRequest& fieldRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  uint64_t buildSerial,
                                  std::string* error);
bool encodeGlossFieldFromIOSurface(const FrameSubmission& submission,
                                   GlossFieldCache* cache,
                                   const RasterSourceRequest& rasterRequest,
                                   const GlossFieldRequest& fieldRequest,
                                   uint32_t surfaceId,
                                   int surfaceWidth,
                                   int surfaceHeight,
                                   int surfacePixelFormat,
                                   uint64_t buildSerial,
                                   std::string* error);
#endif
bool encodeGlossFieldFromImportedTexture(
    const FrameSubmission& submission,
    GlossFieldCache* cache,
    const RasterSourceRequest& rasterRequest,
    const GlossFieldRequest& fieldRequest,
    uint64_t sourceId,
    uint64_t buildSerial,
    std::string* error);
GlossFieldCacheState glossFieldCacheState(const GlossFieldCache& cache);
void releaseGlossFieldCache(GlossFieldCache* cache);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       bool allowReadback,
                       std::vector<float>* outDensity,
                       std::string* error);
#endif
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildScopeDensityFromIOSurface(const RasterSourceRequest& rasterRequest,
                                    const ScopeDensityRequest& scopeRequest,
                                    uint32_t surfaceId,
                                    int surfaceWidth,
                                    int surfaceHeight,
                                    int surfacePixelFormat,
                                    bool allowReadback,
                                    std::vector<float>* outDensity,
                                    std::string* error);
bool buildScopeRangeFromIOSurface(const RasterSourceRequest& rasterRequest,
                                  const ScopeRangeRequest& rangeRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  ScopeRangeResult* outRange,
                                  std::string* error);
#endif
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderHistogramSurfaceFromIOSurface(const RasterSourceRequest& rasterRequest,
                                         const HistogramSurfaceRequest& histogramRequest,
                                         uint32_t sourceSurfaceId,
                                         int sourceSurfaceWidth,
                                         int sourceSurfaceHeight,
                                         int sourceSurfacePixelFormat,
                                         uint32_t outputSurfaceId,
                                         int outputSurfaceWidth,
                                         int outputSurfaceHeight,
                                         int outputSurfacePixelFormat,
                                         std::string* error);
bool encodeHistogramSurfaceFromIOSurface(const FrameSubmission& submission,
                                         const RasterSourceRequest& rasterRequest,
                                         const HistogramSurfaceRequest& histogramRequest,
                                         uint32_t sourceSurfaceId,
                                         int sourceSurfaceWidth,
                                         int sourceSurfaceHeight,
                                         int sourceSurfacePixelFormat,
                                         uint32_t outputSurfaceId,
                                         int outputSurfaceWidth,
                                         int outputSurfaceHeight,
                                         int outputSurfacePixelFormat,
                                         std::string* error);
#endif
bool encodeHistogramSurfaceFromImportedTexture(
    const FrameSubmission& submission,
    const RasterSourceRequest& rasterRequest,
    const HistogramSurfaceRequest& histogramRequest,
    uint64_t sourceId,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error);
bool encodeHistogramSurfaceFromImportedTextureCached(
    const FrameSubmission& submission,
    ScopeDerivedCache* cache,
    const RasterSourceRequest& rasterRequest,
    const HistogramSurfaceRequest& histogramRequest,
    uint64_t sourceId,
    uint64_t buildSerial,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderWaveformSurfaceFromIOSurface(const RasterSourceRequest& rasterRequest,
                                        const WaveformSurfaceRequest& waveformRequest,
                                        uint32_t sourceSurfaceId,
                                        int sourceSurfaceWidth,
                                        int sourceSurfaceHeight,
                                        int sourceSurfacePixelFormat,
                                        uint32_t outputSurfaceId,
                                        int outputSurfaceWidth,
                                        int outputSurfaceHeight,
                                        int outputSurfacePixelFormat,
                                        std::string* error);
bool encodeWaveformSurfaceFromIOSurface(const FrameSubmission& submission,
                                        const RasterSourceRequest& rasterRequest,
                                        const WaveformSurfaceRequest& waveformRequest,
                                        uint32_t sourceSurfaceId,
                                        int sourceSurfaceWidth,
                                        int sourceSurfaceHeight,
                                        int sourceSurfacePixelFormat,
                                        uint32_t outputSurfaceId,
                                        int outputSurfaceWidth,
                                        int outputSurfaceHeight,
                                        int outputSurfacePixelFormat,
                                        std::string* error);
#endif
bool encodeWaveformSurfaceFromImportedTexture(
    const FrameSubmission& submission,
    const RasterSourceRequest& rasterRequest,
    const WaveformSurfaceRequest& waveformRequest,
    uint64_t sourceId,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
                                           int outputSurfacePixelFormat,
                                           std::string* error);
bool encodeWaveformSurfaceFromImportedTextureCached(
    const FrameSubmission& submission,
    ScopeDerivedCache* cache,
    const RasterSourceRequest& rasterRequest,
    const WaveformSurfaceRequest& waveformRequest,
    uint64_t sourceId,
    uint64_t buildSerial,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error);
ScopeDerivedCacheState scopeDerivedCacheState(
    const ScopeDerivedCache& cache);
void releaseScopeDerivedCache(ScopeDerivedCache* cache);
ResidentDerivedCacheState residentDerivedCacheState(
    const ResidentDerivedCache& cache);
void releaseResidentDerivedCache(ResidentDerivedCache* cache);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderRasterPointSurfaceFromIOSurface(const RasterSourceRequest& rasterRequest,
                                           const RasterPointSurfaceRequest& pointRequest,
                                           uint32_t sourceSurfaceId,
                                           int sourceSurfaceWidth,
                                           int sourceSurfaceHeight,
                                           int sourceSurfacePixelFormat,
                                           uint32_t outputSurfaceId,
                                           int outputSurfaceWidth,
                                           int outputSurfaceHeight,
                                           int outputSurfacePixelFormat,
                                           std::string* error);
bool encodeRasterPointSurfaceFromIOSurface(const FrameSubmission& submission,
                                           const RasterSourceRequest& rasterRequest,
                                           const RasterPointSurfaceRequest& pointRequest,
                                           uint32_t sourceSurfaceId,
                                           int sourceSurfaceWidth,
                                           int sourceSurfaceHeight,
                                           int sourceSurfacePixelFormat,
                                           uint32_t outputSurfaceId,
                                           int outputSurfaceWidth,
                                           int outputSurfaceHeight,
                                           int outputSurfacePixelFormat,
                                           std::string* error);
#endif
bool encodeRasterPointSurfaceFromImportedTexture(
    const FrameSubmission& submission,
    const RasterSourceRequest& rasterRequest,
    const RasterPointSurfaceRequest& pointRequest,
    uint64_t sourceId,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error);
bool encodeRasterPointSurfaceFromImportedTextureCached(
    const FrameSubmission& submission,
    ResidentDerivedCache* cache,
    const RasterSourceRequest& rasterRequest,
    const RasterPointSurfaceRequest& pointRequest,
    uint64_t sourceId,
    uint64_t buildSerial,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
     int outputSurfacePixelFormat,
     std::string* error);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderGlossFieldSurfaceFromCache(const GlossFieldCache& cache,
                                      const GlossFieldSurfaceRequest& surfaceRequest,
                                      uint32_t outputSurfaceId,
                                      int outputSurfaceWidth,
                                      int outputSurfaceHeight,
                                       int outputSurfacePixelFormat,
                                       std::string* error);
#endif
bool encodeGlossFieldSurfaceFromCache(const FrameSubmission& submission,
                                      const GlossFieldCache& cache,
                                      const GlossFieldSurfaceRequest& surfaceRequest,
                                      uint32_t outputSurfaceId,
                                      int outputSurfaceWidth,
                                      int outputSurfaceHeight,
                                       int outputSurfacePixelFormat,
                                       std::string* error);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderGlossProjectionSurfaceFromCache(const GlossFieldCache& cache,
                                           const GlossProjectionSurfaceRequest& projectionRequest,
                                           uint32_t outputSurfaceId,
                                           int outputSurfaceWidth,
                                           int outputSurfaceHeight,
                                            int outputSurfacePixelFormat,
                                            std::string* error);
#endif
bool encodeGlossProjectionSurfaceFromCache(const FrameSubmission& submission,
                                           const GlossFieldCache& cache,
                                           const GlossProjectionSurfaceRequest& projectionRequest,
                                           uint32_t outputSurfaceId,
                                           int outputSurfaceWidth,
                                           int outputSurfaceHeight,
                                           int outputSurfacePixelFormat,
                                           std::string* error);

}  // namespace ChromaspaceMetal
