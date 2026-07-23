#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
#import <IOSurface/IOSurface.h>
#import <OpenGL/gl.h>
#import <OpenGL/OpenGL.h>
#endif
#import <simd/simd.h>
#import <mach-o/dyld.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ChromaspaceMetal.h"
#include "ChromaspaceMetalSubmissionRetention.h"
#include "ChromaspaceMetalTransientArena.h"

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
#ifndef GL_TEXTURE_RECTANGLE
#define GL_TEXTURE_RECTANGLE 0x84F5
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif
#ifndef GL_HALF_FLOAT
#define GL_HALF_FLOAT 0x140B
#endif
#endif

namespace ChromaspaceMetal {
namespace {

using FrameFailureKind = ChromaspaceMetalFrameFailure::Kind;
using SubmissionRetention =
    ChromaspaceMetalSubmissionRetention::RetentionSet;
using SubmissionRetentionKey =
    ChromaspaceMetalSubmissionRetention::ResourceKey;
using SubmissionRetentionKind =
    ChromaspaceMetalSubmissionRetention::ResourceKind;
using SubmissionRetentionStatus =
    ChromaspaceMetalSubmissionRetention::Status;

void setFrameFailure(FrameFailureKind* failure, FrameFailureKind kind) noexcept {
  if (failure) *failure = kind;
}

struct MetalContext {
  uint64_t runtimeContextId = 0u;
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> overlayPipeline = nil;
  id<MTLComputePipelineState> inputPipeline = nil;
  id<MTLComputePipelineState> rasterSourcePipeline = nil;
  id<MTLComputePipelineState> rasterOccupancyCountPipeline = nil;
  id<MTLComputePipelineState> rasterSourceTexturePipeline = nil;
  id<MTLComputePipelineState> rasterOccupancyTextureCountPipeline = nil;
  id<MTLComputePipelineState> rasterOccupancyThresholdPipeline = nil;
  id<MTLComputePipelineState> rasterPointCompactLocalScanPipeline = nil;
  id<MTLComputePipelineState> rasterPointScanBlockSumsPipeline = nil;
  id<MTLComputePipelineState> rasterPointAddBlockOffsetsPipeline = nil;
  id<MTLComputePipelineState> rasterPointCompactScatterPipeline = nil;
  id<MTLComputePipelineState> rasterPointFinalizeIndirectArgsPipeline = nil;
  id<MTLComputePipelineState> inputSamplePipeline = nil;
  id<MTLComputePipelineState> scopeDensityPipeline = nil;
  id<MTLComputePipelineState> rasterScopeDensityTexturePipeline = nil;
  id<MTLComputePipelineState> rasterScopeRangeTexturePipeline = nil;
  id<MTLComputePipelineState> rasterScopeRangeHistogramTexturePipeline = nil;
  id<MTLComputePipelineState> scopeRangeHistogramPercentilePipeline = nil;
  id<MTLComputePipelineState> scopeRangeFinalizePipeline = nil;
  id<MTLComputePipelineState> histogramApplyRangePipeline = nil;
  id<MTLComputePipelineState> histogramMaxPipeline = nil;
  id<MTLComputePipelineState> histogramSurfaceRenderPipeline = nil;
  id<MTLComputePipelineState> waveformApplyRangePipeline = nil;
  id<MTLComputePipelineState> waveformMaxPipeline = nil;
  id<MTLComputePipelineState> waveformSurfaceRenderPipeline = nil;
  id<MTLComputePipelineState> glossFieldAccumulatePipeline = nil;
  id<MTLComputePipelineState> rasterGlossFieldAccumulateTexturePipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldMaxPipeline = nil;
  id<MTLComputePipelineState> glossFieldNormalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldBlurPipeline = nil;
  id<MTLComputePipelineState> glossFieldBodyPipeline = nil;
  id<MTLComputePipelineState> glossFieldRawSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldWeightedSignalPipeline = nil;
  id<MTLComputePipelineState> glossFieldMergeMaxBitsPipeline = nil;
  id<MTLComputePipelineState> glossFieldFinalNormalizePipeline = nil;
  id<MTLComputePipelineState> glossFieldLocalPercentilePipeline = nil;
  id<MTLComputePipelineState> glossFieldCandidate2RawPipeline = nil;
  id<MTLComputePipelineState> glossFieldAssembleUnifiedPipeline = nil;
  id<MTLComputePipelineState> glossFieldSurfaceRenderPipeline = nil;
  id<MTLComputePipelineState> glossProjectionSurfaceSelectPipeline = nil;
  id<MTLComputePipelineState> glossProjectionSurfaceShadePipeline = nil;
  id<MTLComputePipelineState> plotSurfaceClearPipeline = nil;
  id<MTLComputePipelineState> sourceSignalSurfacePipeline = nil;
  id<MTLRenderPipelineState> plotSurfaceVectorPipeline16 = nil;
  id<MTLRenderPipelineState> plotSurfaceVectorPipeline32 = nil;
  id<MTLRenderPipelineState> rasterPointSurfacePipeline16 = nil;
  id<MTLRenderPipelineState> rasterPointSurfacePipeline32 = nil;
  id<MTLRenderPipelineState> frameSurfaceCompositePipeline = nil;
  id<MTLRenderPipelineState> frameSolidRectPipeline = nil;
  id<MTLRenderPipelineState> frameUiVectorPipeline = nil;
  id<MTLRenderPipelineState> frameTextPipeline = nil;
  std::string deviceName;
  std::string initError;
  bool initAttempted = false;
  bool ready = false;
};

struct PlotSurfaceRecord {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
  IOSurfaceRef surface = nullptr;
#endif
  id<MTLTexture> texture = nil;
  std::shared_ptr<MetalContext> context;
  uint64_t ownerCompositorId = 0;
  int width = 0;
  int height = 0;
  int pixelFormat = 0;
  size_t byteSize = 0;

  PlotSurfaceRecord() = default;
  PlotSurfaceRecord(const PlotSurfaceRecord&) = delete;
  PlotSurfaceRecord& operator=(const PlotSurfaceRecord&) = delete;
  PlotSurfaceRecord(PlotSurfaceRecord&&) = delete;
  PlotSurfaceRecord& operator=(PlotSurfaceRecord&&) = delete;

  ~PlotSurfaceRecord() noexcept {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    // The registry owns exactly one create/lookup retain.  Shared submission
    // records keep this object alive after public-handle removal, so this is
    // the sole final CFRelease for compatibility IOSurfaces.
    if (surface != nullptr) {
      CFRelease(surface);
      surface = nullptr;
    }
#endif
  }
};

struct FrameTransientArenaState {
  std::mutex mutex;
  ChromaspaceMetalTransientArena::TransientArena policy;
};

struct FrameCompositorRecord {
  std::shared_ptr<MetalContext> context;
  NSWindow* window = nil;
  NSView* contentView = nil;
  CALayer* previousLayer = nil;
  CAMetalLayer* layer = nil;
  dispatch_semaphore_t frameSlots = nullptr;
  dispatch_group_t completionGroup = nullptr;
  std::shared_ptr<FrameTransientArenaState> transientArena;
  int drawableWidth = 0;
  int drawableHeight = 0;
  float contentsScale = 1.0f;
  uint64_t submittedSerial = 0;
  uint64_t completedSerial = 0;
  uint64_t failedSubmissionCount = 0;
  uint64_t timedSubmissionCount = 0;
  uint64_t untimedSubmissionCount = 0;
  double accumulatedGpuSeconds = 0.0;
  double maximumGpuSeconds = 0.0;
  std::string lastSubmissionError;
  std::string pendingSubmissionError;
  BOOL previousWantsLayer = NO;
};

struct GlossFieldResidentRecord;
struct ImportedSourceRecord;

struct FrameSubmissionTransactionRecord {
  std::function<void()> submitted;
  std::function<void(bool)> completed;
  std::function<void()> abandoned;
};

struct FrameSubmissionRecord {
  std::shared_ptr<MetalContext> context;
  uint64_t compositorId = 0;
  dispatch_semaphore_t frameSlots = nullptr;
  id<MTLCommandBuffer> commandBuffer = nil;
  std::shared_ptr<FrameTransientArenaState> transientArena;
  NSMutableArray<id<MTLHeap>>* transientHeaps = nil;
  std::vector<FrameSubmissionTransactionRecord> transactions;
  SubmissionRetention retainedResources{};
  std::unordered_map<uint64_t, std::shared_ptr<ImportedSourceRecord>>
      retainedImportedSources;
};

struct ImportedSourceRecord {
  ImportedSourceTexture descriptor{};
  id<MTLTexture> texture = nil;
  id<MTLSharedEvent> readyEvent = nil;
  std::mutex lifetimeMutex;
  size_t inFlightSubmissionUses = 0;
  bool retirementRequested = false;
  ImportedSourceRetirementCallback retirementCallback = nullptr;
  void* retirementContext = nullptr;
};

struct FrameTextAtlasRecord {
  id<MTLTexture> texture = nil;
  std::shared_ptr<MetalContext> context;
  uint64_t ownerCompositorId = 0;
  int width = 0;
  int height = 0;
};

struct GlossFieldResidentRecord {
  int gridWidth = 0;
  int gridHeight = 0;
  uint64_t builtSerial = 0;
  size_t byteSize = 0u;
  id<MTLBuffer> meanR = nil;
  id<MTLBuffer> meanG = nil;
  id<MTLBuffer> meanB = nil;
  id<MTLBuffer> carrierY = nil;
  id<MTLBuffer> carrierMax = nil;
  id<MTLBuffer> carrierMin = nil;
  id<MTLBuffer> neutrality = nil;
  id<MTLBuffer> body = nil;
  id<MTLBuffer> positive = nil;
  id<MTLBuffer> negative = nil;
  id<MTLBuffer> boundary = nil;
  id<MTLBuffer> congruence = nil;
  id<MTLBuffer> confidence = nil;
  id<MTLBuffer> signal = nil;
  id<MTLBuffer> body2 = nil;
  id<MTLBuffer> positive2 = nil;
  id<MTLBuffer> negative2 = nil;
  id<MTLBuffer> boundary2 = nil;
  id<MTLBuffer> congruence2 = nil;
  id<MTLBuffer> confidence2 = nil;
  id<MTLBuffer> signal2 = nil;
};

struct ScopeDerivedResidentRecord {
  ScopeDerivedFamily family = ScopeDerivedFamily::Histogram;
  uint64_t builtSerial = 0;
  size_t byteSize = 0;
  id<MTLBuffer> density = nil;
  id<MTLBuffer> overflowDensity = nil;
  id<MTLBuffer> maxDensity = nil;
  // GPU-produced [orderedMin, orderedMax, validCount]. Retained only for
  // auto-range derivations so presentation uniforms can be rebuilt without
  // rescanning the source texture.
  id<MTLBuffer> finalRange = nil;
  // RasterPointCloud records retain stable plot-space topology and attributes.
  // Camera, target size, point radius, background, and guides remain
  // presentation inputs and therefore never enter this native record.
  id<MTLBuffer> pointVertices = nil;
  id<MTLBuffer> pointColors = nil;
  // GPU-authored MTLDrawPrimitivesIndirectArguments. The vertex count is the
  // compacted visible count; no CPU readback participates in presentation.
  id<MTLBuffer> pointIndirectArguments = nil;
  NSUInteger pointCount = 0u;
  // Gloss uses the same queue-visibility/retirement registry while keeping
  // its large typed buffer set behind one deep native record.
  std::shared_ptr<GlossFieldResidentRecord> glossField;
};

enum class ScopeDerivedResidentVersionState {
  Pending = 0,
  Submitted,
};

struct ScopeDerivedResidentVersion {
  uint64_t submissionId = 0;
  ScopeDerivedResidentVersionState state =
      ScopeDerivedResidentVersionState::Pending;
  std::shared_ptr<ScopeDerivedResidentRecord> record;
};

struct ScopeDerivedResidentEntry {
  uint64_t ownerCompositorId = 0;
  std::shared_ptr<ScopeDerivedResidentRecord> committed;
  std::vector<ScopeDerivedResidentVersion> inFlight;
};

struct OverlayUniforms {
  int cubeSize;
  int ramp;
  int useInputPoints;
  int pointCount;
  float colorSaturation;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
};

struct InputUniforms {
  int pointCount;
  int inputStride;
  int glossView;
  float sourceAspect;
  float glossLiftScale;
  int showOverflow;
  int highlightOverflow;
  int plotMode;
  int circularHsl;
  int circularHsv;
  int normConeNormalized;
  int chromaticityInputTransfer;
  int chromaticityReferenceBasis;
  float chromaticityWhiteX;
  float chromaticityWhiteY;
  float chromaticityRgbToXyz[9];
  float chromaticityXyzToRgb[9];
  float pointAlphaScale;
  float denseAlphaBias;
  float colorSaturation;
};

struct RasterSourceUniforms {
  InputUniforms input;
  int basePointCount;
  int sourceWidth;
  int sourceHeight;
  int sampleStride;
  int sampleCountX;
  int pixelFormat;
  int plotLinear;
  int plotLinearTransfer;
  int excludeIdentityData;
  int isolateIdentityData;
  int readIdentityPlot;
  int readGrayRamp;
  int identityCubeY1;
  int identityCubeY2;
  int identityRampY1;
  int identityRampY2;
  int identityCubeAppendOffset;
  int identityCubeAppendCount;
  int identityCubeAppendY1;
  int identityCubeAppendY2;
  int identityCubeAppendRowStep;
  int identityCubeAppendXStep;
  int identityRampAppendOffset;
  int identityRampAppendCount;
  int identityRampAppendY1;
  int identityRampAppendY2;
  int identityRampAppendRowStep;
  int identityRampAppendXStep;
  int occupancyFill;
  int occupancyAppendOffset;
  int occupancyAppendCount;
  int occupancyCandidateCount;
  int occupancyTargetThreshold;
  int lassoEnabled;
  int lassoStrokeCount;
  int lassoPointCount;
  int lassoStrokeFirst[16];
  int lassoStrokeCountPerStroke[16];
  int lassoStrokeSubtract[16];
  float lassoX[256];
  float lassoY[256];
  int cubeSlicingEnabled;
  int neutralRadiusEnabled;
  float neutralRadius;
  int cubeSliceRed;
  int cubeSliceYellow;
  int cubeSliceGreen;
  int cubeSliceCyan;
  int cubeSliceBlue;
  int cubeSliceMagenta;
};

struct InputSampleUniforms {
  int fullPointCount;
  int visiblePointCount;
};

struct ScopeDensityUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int excludeOverflow;
  int onlyOverflow;
  int channelCount;
  int lumaMethod;
};

struct ScopeRangeUniforms {
  int pointCount;
  int waveform;
  int scopeMode;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  int includeOverflow;
  int lumaMethod;
  int previousRangeValid;
  float previousRangeMin;
  float previousRangeMax;
  int histogramBinCount;
};

struct HistogramSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
};

struct WaveformSurfaceUniforms {
  int pointCount;
  int scopeMode;
  int width;
  int height;
  float rangeMin;
  float invRange;
  int showOverflow;
  int highlightOverflow;
  int lumaMethod;
  int channelCount;
  int includeRed;
  int includeGreen;
  int includeBlue;
  int includeLuma;
  float pointBrightness;
  float colorSaturation;
  float coverageAlpha;
};

struct PlotSurfaceClearUniforms {
  float r;
  float g;
  float b;
  float a;
};

struct SourceSignalSurfaceUniforms {
  int sourceWidth;
  int sourceHeight;
  int outputWidth;
  int outputHeight;
  float backgroundR;
  float backgroundG;
  float backgroundB;
  float pad0;
};

struct RasterPointSurfaceUniforms {
  float modelView[16];
  float projection[16];
  float pointRadiusPixels;
  float surfaceWidth;
  float surfaceHeight;
  float pad0;
};

struct SurfaceCompositeUniforms {
  float dstX;
  float dstY;
  float dstW;
  float dstH;
  float drawableW;
  float drawableH;
  float opacity;
  float pad0;
};

struct FrameSolidRectUniforms {
  float dstX;
  float dstY;
  float dstW;
  float dstH;
  float drawableW;
  float drawableH;
  float r;
  float g;
  float b;
  float a;
  float pad0;
  float pad1;
};

struct FrameUiVectorUniforms {
  float drawableW;
  float drawableH;
  float pad0;
  float pad1;
};

struct FrameTextUniforms {
  float drawableW;
  float drawableH;
  float r;
  float g;
  float b;
  float a;
  float clipX;
  float clipY;
  float clipW;
  float clipH;
  float clipEnabled;
  float pad0;
  float pad1;
  float pad2;
};

struct GlossFieldAccumulateUniforms {
  int pointCount;
  int gridWidth;
  int gridHeight;
  int showOverflow;
};

struct GlossFieldCellUniforms {
  int cellCount;
  int gridWidth;
  int gridHeight;
  int neighborhoodChoice;
};

struct GlossFieldSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
};

struct GlossProjectionSurfaceUniforms {
  int gridWidth;
  int gridHeight;
  int surfaceWidth;
  int surfaceHeight;
  int algorithm;
  int colorMode;
  int debugMode;
  int diagnosticMode;
  float sourceAspect;
  float colorSaturation;
  float glossBodyOpacity;
  float glossHighlightOpacity;
  float glossLiftScale;
  float pointRadiusPixels;
  float modelView[16];
  float projection[16];
};

struct PackedFloat3 {
  float x;
  float y;
  float z;
};

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
MetalContext& context() {
  static MetalContext ctx;
  return ctx;
}
#endif

MTLPixelFormat sourceSignalMetalPixelFormat(int pixelFormat) {
  return pixelFormat == 1 ? MTLPixelFormatRGBA32Float : MTLPixelFormatRGBA16Float;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
OSType sourceSignalIOSurfacePixelFormat(int pixelFormat) {
  return pixelFormat == 1 ? static_cast<OSType>('RGBA') : static_cast<OSType>('RGhA');
}
#endif

size_t sourceSignalBytesPerElement(int pixelFormat) {
  return pixelFormat == 1 ? 16u : 8u;
}

std::mutex& plotSurfaceMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint32_t, std::shared_ptr<PlotSurfaceRecord>>&
plotSurfaceRegistry() {
  static std::unordered_map<uint32_t, std::shared_ptr<PlotSurfaceRecord>>
      registry;
  return registry;
}

// Must be called while plotSurfaceMutex() is held. Plot-surface handles form
// their own namespace and are deliberately unrelated to IOSurface IDs.
uint32_t allocatePlotSurfaceHandleLocked() {
  auto& registry = plotSurfaceRegistry();
  constexpr size_t kMaximumLiveHandles =
      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - 1u;
  if (registry.size() >= kMaximumLiveHandles) return 0;

  static uint32_t nextHandle = 1;
  if (nextHandle == 0) nextHandle = 1;
  const uint32_t firstCandidate = nextHandle;
  do {
    const uint32_t candidate = nextHandle;
    ++nextHandle;
    if (nextHandle == 0) nextHandle = 1;
    if (candidate != 0 && registry.find(candidate) == registry.end()) {
      return candidate;
    }
  } while (nextHandle != firstCandidate);
  return 0;
}

std::mutex& frameCompositorMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, FrameCompositorRecord>& frameCompositorRegistry() {
  static std::unordered_map<uint64_t, FrameCompositorRecord> registry;
  return registry;
}

// Runtime-context IDs are process-local generation identities and are never
// recycled.  A device registry ID may remain stable across recreation, while
// this ID must change whenever the device/queue/library/pipeline owner changes.
// frameCompositorMutex() serializes allocation.
uint64_t allocateRuntimeContextIdLocked() {
  static uint64_t nextId = 1u;
  if (nextId == 0u) return 0u;
  const uint64_t result = nextId;
  if (nextId == std::numeric_limits<uint64_t>::max()) {
    nextId = 0u;
  } else {
    ++nextId;
  }
  return result;
}

std::mutex& frameSubmissionMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, FrameSubmissionRecord>& frameSubmissionRegistry() {
  static std::unordered_map<uint64_t, FrameSubmissionRecord> registry;
  return registry;
}

bool contextForCompositor(uint64_t compositorId,
                          std::shared_ptr<MetalContext>* outContext,
                          std::string* error) {
  if (outContext) outContext->reset();
  if (compositorId == 0u) {
    if (error) *error = "invalid-metal-frame-compositor-handle";
    return false;
  }
  std::lock_guard<std::mutex> lock(frameCompositorMutex());
  const auto it = frameCompositorRegistry().find(compositorId);
  if (it == frameCompositorRegistry().end() || !it->second.context ||
      !it->second.context->ready ||
      it->second.context->runtimeContextId == 0u) {
    if (error) *error = "metal-runtime-context-not-found";
    return false;
  }
  if (outContext) *outContext = it->second.context;
  return true;
}

bool contextForFrameSubmission(const FrameSubmission& submission,
                               std::shared_ptr<MetalContext>* outContext,
                               std::string* error) {
  if (outContext) outContext->reset();
  if (submission.submissionId == 0u || submission.compositorId == 0u ||
      submission.runtimeContextId == 0u || submission.deviceRegistryId == 0u) {
    if (error) *error = "invalid-metal-frame-submission-context";
    return false;
  }
  std::lock_guard<std::mutex> lock(frameSubmissionMutex());
  const auto it = frameSubmissionRegistry().find(submission.submissionId);
  if (it == frameSubmissionRegistry().end() || !it->second.context ||
      it->second.compositorId != submission.compositorId ||
      it->second.context->runtimeContextId != submission.runtimeContextId ||
      it->second.context->device == nil ||
      it->second.context->device.registryID != submission.deviceRegistryId) {
    if (error) *error = "metal-frame-submission-context-mismatch";
    return false;
  }
  if (outContext) *outContext = it->second.context;
  return true;
}

bool submissionIdentityMatches(const FrameSubmissionRecord& record,
                               const FrameSubmission& submission) {
  return record.compositorId == submission.compositorId && record.context &&
         record.context->ready && record.context->runtimeContextId != 0u &&
         record.context->runtimeContextId == submission.runtimeContextId &&
         record.context->device != nil && submission.deviceRegistryId != 0u &&
         record.context->device.registryID == submission.deviceRegistryId;
}

bool contextForCommandBuffer(id<MTLCommandBuffer> commandBuffer,
                             std::shared_ptr<MetalContext>* outOwnedContext,
                             MetalContext** outContext,
                             std::string* error);

std::mutex& frameTextAtlasMutex();

std::unordered_map<uint64_t, std::shared_ptr<FrameTextAtlasRecord>>&
frameTextAtlasRegistry();

std::mutex& importedSourceMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, std::shared_ptr<ImportedSourceRecord>>&
importedSourceRegistry() {
  static std::unordered_map<uint64_t, std::shared_ptr<ImportedSourceRecord>>
      registry;
  return registry;
}

// Submission/resource lock order is intentionally centralized here:
// frameCompositorMutex() -> frameSubmissionMutex() is used only for lifecycle
// publication.  Resource acquisition takes plotSurfaceMutex(),
// frameTextAtlasMutex(), or scopeDerivedRegistryMutex() first, releases that
// lock, then takes frameSubmissionMutex() to install the strong hold.  No
// callbacks run while any registry lock is held, and no helper nests a
// resource lock under the submission lock.  This keeps teardown, encode, and
// completion paths from introducing a lock-order inversion.
const char* submissionRetentionDiagnostic(SubmissionRetentionStatus status) {
  switch (status) {
    case SubmissionRetentionStatus::InvalidKey:
      return "metal-frame-retention-invalid-key";
    case SubmissionRetentionStatus::NullResource:
      return "metal-frame-retention-null-resource";
    case SubmissionRetentionStatus::CapacityExhausted:
      return "metal-frame-retention-capacity-exhausted";
    case SubmissionRetentionStatus::KeyConflict:
      return "metal-frame-retention-key-conflict";
    case SubmissionRetentionStatus::Sealed:
      return "metal-frame-retention-sealed";
    case SubmissionRetentionStatus::InvalidCapacity:
      return "metal-frame-retention-invalid-capacity";
    case SubmissionRetentionStatus::NotFound:
      return "metal-frame-retention-not-found";
    case SubmissionRetentionStatus::Retained:
    case SubmissionRetentionStatus::Duplicate:
      return nullptr;
  }
  return "metal-frame-retention-invalid-status";
}

bool retainSubmissionResource(const FrameSubmission& submission,
                              const SubmissionRetentionKey& key,
                              const std::shared_ptr<void>& resource,
                              std::string* error,
                              bool* outAdded = nullptr) {
  if (error) error->clear();
  if (outAdded) *outAdded = false;
  if (submission.submissionId == 0u || submission.compositorId == 0u) {
    if (error) *error = "invalid-metal-frame-submission-retention";
    return false;
  }
  if (key.ownerCompositorId != submission.compositorId) {
    if (error) *error = "metal-frame-retention-owner-mismatch";
    return false;
  }
  std::lock_guard<std::mutex> lock(frameSubmissionMutex());
  auto submissionIt = frameSubmissionRegistry().find(submission.submissionId);
  if (submissionIt == frameSubmissionRegistry().end()) {
    if (error) *error = "metal-frame-submission-not-found";
    return false;
  }
  if (!submissionIdentityMatches(submissionIt->second, submission)) {
    if (error) *error = "metal-frame-submission-context-mismatch";
    return false;
  }
  const SubmissionRetentionStatus status =
      submissionIt->second.retainedResources.retain(key, resource);
  if (ChromaspaceMetalSubmissionRetention::succeeded(status)) {
    if (outAdded) *outAdded = status == SubmissionRetentionStatus::Retained;
    return true;
  }
  if (error) *error = submissionRetentionDiagnostic(status);
  return false;
}

bool releaseSubmissionResource(const FrameSubmission& submission,
                               const SubmissionRetentionKey& key) {
  if (submission.submissionId == 0u) return false;
  std::lock_guard<std::mutex> lock(frameSubmissionMutex());
  auto submissionIt = frameSubmissionRegistry().find(submission.submissionId);
  if (submissionIt == frameSubmissionRegistry().end() ||
      !submissionIdentityMatches(submissionIt->second, submission)) {
    return false;
  }
  return submissionIt->second.retainedResources.release(key) ==
         SubmissionRetentionStatus::Retained;
}

bool retainPlotSurfaceForSubmission(
    const FrameSubmission& submission,
    uint32_t surfaceId,
    std::shared_ptr<PlotSurfaceRecord>* outRecord,
    std::string* error) {
  if (outRecord) outRecord->reset();
  if (surfaceId == 0u) {
    if (error) *error = "invalid-metal-plot-surface-handle";
    return false;
  }
  std::shared_ptr<PlotSurfaceRecord> record;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    const auto it = plotSurfaceRegistry().find(surfaceId);
    if (it == plotSurfaceRegistry().end() || !it->second ||
        it->second->texture == nil) {
      if (error) *error = "metal-plot-surface-not-found";
      return false;
    }
    if (it->second->ownerCompositorId == 0u) {
      if (error) *error = "metal-plot-surface-is-compatibility-owned";
      return false;
    }
    if (it->second->ownerCompositorId != submission.compositorId) {
      if (error) *error = "metal-plot-surface-compositor-mismatch";
      return false;
    }
    record = it->second;
  }
  const SubmissionRetentionKey key{
      SubmissionRetentionKind::PlotSurface,
      static_cast<uint64_t>(surfaceId),
      submission.compositorId};
  bool added = false;
  if (!retainSubmissionResource(submission, key, record, error, &added)) {
    return false;
  }
  bool stillLive = false;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    const auto it = plotSurfaceRegistry().find(surfaceId);
    stillLive = it != plotSurfaceRegistry().end() && it->second == record &&
                it->second->ownerCompositorId == submission.compositorId &&
                it->second->texture != nil;
  }
  if (!stillLive) {
    if (added) (void)releaseSubmissionResource(submission, key);
    if (error) *error = "metal-frame-retention-retired-during-acquire";
    return false;
  }
  if (outRecord) *outRecord = std::move(record);
  return true;
}

bool retainFrameTextAtlasForSubmission(
    const FrameSubmission& submission,
    uint64_t atlasId,
    std::shared_ptr<FrameTextAtlasRecord>* outRecord,
    std::string* error) {
  if (outRecord) outRecord->reset();
  if (atlasId == 0u) {
    if (error) *error = "invalid-metal-frame-text-atlas-handle";
    return false;
  }
  std::shared_ptr<FrameTextAtlasRecord> record;
  {
    std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
    const auto it = frameTextAtlasRegistry().find(atlasId);
    if (it == frameTextAtlasRegistry().end() || !it->second ||
        it->second->texture == nil) {
      if (error) *error = "metal-frame-text-atlas-not-found";
      return false;
    }
    if (it->second->ownerCompositorId == 0u) {
      if (error) *error = "metal-frame-text-atlas-owner-zero";
      return false;
    }
    if (it->second->ownerCompositorId != submission.compositorId) {
      if (error) *error = "metal-frame-text-atlas-owner-mismatch";
      return false;
    }
    record = it->second;
  }
  const SubmissionRetentionKey key{
      SubmissionRetentionKind::TextAtlas, atlasId, submission.compositorId};
  bool added = false;
  if (!retainSubmissionResource(submission, key, record, error, &added)) {
    return false;
  }
  bool stillLive = false;
  {
    std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
    const auto it = frameTextAtlasRegistry().find(atlasId);
    stillLive = it != frameTextAtlasRegistry().end() && it->second == record &&
                it->second->ownerCompositorId == submission.compositorId &&
                it->second->texture != nil;
  }
  if (!stillLive) {
    if (added) (void)releaseSubmissionResource(submission, key);
    if (error) *error = "metal-frame-retention-retired-during-acquire";
    return false;
  }
  if (outRecord) *outRecord = std::move(record);
  return true;
}

uint64_t allocateImportedSourceIdLocked() {
  auto& registry = importedSourceRegistry();
  constexpr size_t kMaximumImportedSources = 32u * 3u;
  if (registry.size() >= kMaximumImportedSources) return 0;
  if (registry.size() >=
      static_cast<size_t>(std::numeric_limits<uint64_t>::max()) - 1u) {
    return 0;
  }
  static uint64_t nextId = 1;
  if (nextId == 0) nextId = 1;
  const uint64_t firstCandidate = nextId;
  do {
    const uint64_t candidate = nextId++;
    if (nextId == 0) nextId = 1;
    if (candidate != 0 && registry.find(candidate) == registry.end()) {
      return candidate;
    }
  } while (nextId != firstCandidate);
  return 0;
}

bool beginImportedSourceUse(
    const std::shared_ptr<ImportedSourceRecord>& record) {
  if (!record) return false;
  std::lock_guard<std::mutex> lock(record->lifetimeMutex);
  if (record->retirementRequested) return false;
  ++record->inFlightSubmissionUses;
  return true;
}

void completeImportedSourceUse(
    const std::shared_ptr<ImportedSourceRecord>& record) {
  if (!record) return;
  ImportedSourceRetirementCallback callback = nullptr;
  void* callbackContext = nullptr;
  {
    std::lock_guard<std::mutex> lock(record->lifetimeMutex);
    if (record->inFlightSubmissionUses == 0) return;
    --record->inFlightSubmissionUses;
    if (record->retirementRequested &&
        record->inFlightSubmissionUses == 0) {
      callback = record->retirementCallback;
      callbackContext = record->retirementContext;
      record->retirementCallback = nullptr;
      record->retirementContext = nullptr;
    }
  }
  if (callback != nullptr) callback(callbackContext);
}

constexpr int64_t kFrameSlotTimeoutNanoseconds = 250ll * 1000ll * 1000ll;

uint64_t allocateFrameSubmissionIdLocked() {
  auto& registry = frameSubmissionRegistry();
  constexpr size_t kMaximumLiveSubmissions =
      static_cast<size_t>(std::numeric_limits<uint64_t>::max()) - 1u;
  if (registry.size() >= kMaximumLiveSubmissions) return 0;

  static uint64_t nextId = 1;
  if (nextId == 0) nextId = 1;
  const uint64_t firstCandidate = nextId;
  do {
    const uint64_t candidate = nextId;
    ++nextId;
    if (nextId == 0) nextId = 1;
    if (candidate != 0 && registry.find(candidate) == registry.end()) {
      return candidate;
    }
  } while (nextId != firstCandidate);
  return 0;
}

bool commandBufferForFrameSubmission(const FrameSubmission& submission,
                                     id<MTLCommandBuffer>* outCommandBuffer,
                                     std::string* error,
                                     FrameFailureKind* failure = nullptr) {
  if (outCommandBuffer) *outCommandBuffer = nil;
  if (submission.submissionId == 0 || submission.compositorId == 0) {
    if (error) *error = "invalid-metal-frame-submission";
    setFrameFailure(failure, FrameFailureKind::InvalidState);
    return false;
  }
  std::lock_guard<std::mutex> lock(frameSubmissionMutex());
  auto it = frameSubmissionRegistry().find(submission.submissionId);
  if (it == frameSubmissionRegistry().end()) {
    if (error) *error = "metal-frame-submission-not-found";
    setFrameFailure(failure, FrameFailureKind::InvariantViolation);
    return false;
  }
  if (!submissionIdentityMatches(it->second, submission)) {
    if (error) *error = "metal-frame-submission-context-mismatch";
    setFrameFailure(failure, FrameFailureKind::InvariantViolation);
    return false;
  }
  if (it->second.commandBuffer == nil) {
    if (error) *error = "metal-frame-submission-command-buffer-unavailable";
    setFrameFailure(failure, FrameFailureKind::CommandBufferUnavailable);
    return false;
  }
  if (outCommandBuffer) *outCommandBuffer = it->second.commandBuffer;
  return true;
}

bool importedSourceForFrameSubmission(
    const FrameSubmission& submission,
    uint64_t sourceId,
    std::shared_ptr<ImportedSourceRecord>* outRecord,
    std::string* error) {
  if (outRecord) outRecord->reset();
  if (sourceId == 0) {
    if (error) *error = "invalid-imported-source-handle";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(frameSubmissionMutex());
    auto submissionIt =
        frameSubmissionRegistry().find(submission.submissionId);
    if (submissionIt == frameSubmissionRegistry().end()) {
      if (error) *error = "metal-frame-submission-not-found";
      return false;
    }
    FrameSubmissionRecord& submissionRecord = submissionIt->second;
    if (!submissionIdentityMatches(submissionRecord, submission)) {
      if (error) *error = "metal-frame-submission-context-mismatch";
      return false;
    }
    if (submissionRecord.commandBuffer == nil) {
      if (error) *error = "metal-frame-submission-command-buffer-unavailable";
      return false;
    }
    auto retainedIt =
        submissionRecord.retainedImportedSources.find(sourceId);
    if (retainedIt != submissionRecord.retainedImportedSources.end()) {
      if (!retainedIt->second ||
          retainedIt->second->descriptor.deviceRegistryId !=
              submission.deviceRegistryId) {
        if (error) *error = "imported-source-submission-device-mismatch";
        return false;
      }
      if (outRecord) *outRecord = retainedIt->second;
      return true;
    }
  }

  std::shared_ptr<ImportedSourceRecord> record;
  {
    std::lock_guard<std::mutex> lock(importedSourceMutex());
    auto it = importedSourceRegistry().find(sourceId);
    if (it == importedSourceRegistry().end() || !it->second ||
        it->second->texture == nil || it->second->readyEvent == nil) {
      if (error) *error = "imported-source-not-found";
      return false;
    }
    record = it->second;
  }
  if (record->descriptor.deviceRegistryId != submission.deviceRegistryId) {
    if (error) *error = "imported-source-submission-device-mismatch";
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(frameSubmissionMutex());
    auto submissionIt =
        frameSubmissionRegistry().find(submission.submissionId);
    if (submissionIt == frameSubmissionRegistry().end()) {
      if (error) *error = "metal-frame-submission-not-found";
      return false;
    }
    FrameSubmissionRecord& submissionRecord = submissionIt->second;
    if (!submissionIdentityMatches(submissionRecord, submission)) {
      if (error) *error = "metal-frame-submission-context-mismatch";
      return false;
    }
    if (submissionRecord.commandBuffer == nil) {
      if (error) *error = "metal-frame-submission-command-buffer-unavailable";
      return false;
    }
    auto retainedIt =
        submissionRecord.retainedImportedSources.find(sourceId);
    if (retainedIt != submissionRecord.retainedImportedSources.end()) {
      record = retainedIt->second;
      if (!record || record->descriptor.deviceRegistryId !=
                         submission.deviceRegistryId) {
        if (error) *error = "imported-source-submission-device-mismatch";
        return false;
      }
    } else {
      FrameSubmissionTransactionRecord transaction{};
      transaction.completed =
          [record](bool) { completeImportedSourceUse(record); };
      transaction.abandoned =
          [record]() { completeImportedSourceUse(record); };
      // Allocate every frame-owned record before acquiring the GPU lifetime
      // count. Retirement may win between the global lookup and this point;
      // beginImportedSourceUse then fails and both records roll back before
      // any Metal command references the source.
      const size_t transactionCountBefore =
          submissionRecord.transactions.size();
      try {
        submissionRecord.transactions.push_back(std::move(transaction));
        submissionRecord.retainedImportedSources.emplace(sourceId, record);
      } catch (...) {
        if (submissionRecord.transactions.size() >
            transactionCountBefore) {
          submissionRecord.transactions.resize(transactionCountBefore);
        }
        submissionRecord.retainedImportedSources.erase(sourceId);
        if (error) {
          *error = "metal-frame-imported-source-retention-allocation-failed";
        }
        return false;
      }
      if (!beginImportedSourceUse(record)) {
        submissionRecord.transactions.pop_back();
        submissionRecord.retainedImportedSources.erase(sourceId);
        if (error) *error = "imported-source-retirement-in-progress";
        return false;
      }
      [submissionRecord.commandBuffer encodeWaitForEvent:record->readyEvent
                                                   value:record->descriptor.readyValue];
    }
  }
  if (outRecord) *outRecord = std::move(record);
  return true;
}

bool validatePlotSurfaceOwnerForSubmission(const FrameSubmission& submission,
                                           uint32_t surfaceId,
                                           std::string* error) {
  return retainPlotSurfaceForSubmission(submission, surfaceId, nullptr, error);
}

bool validateResidentDerivedOwnerForSubmission(
    const FrameSubmission& submission,
    uint64_t cacheId,
    uint64_t ownerCompositorId,
    std::string* error) {
  if (cacheId == 0u && ownerCompositorId == 0u) return true;
  if (cacheId == 0u || submission.compositorId == 0u ||
      ownerCompositorId != submission.compositorId) {
    if (error) *error = "metal-resident-derived-cache-owner-mismatch";
    return false;
  }
  return true;
}

uint64_t nextFrameCompositorId() {
  static uint64_t nextId = 1;
  return nextId++;
}

std::mutex& frameTextAtlasMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, std::shared_ptr<FrameTextAtlasRecord>>&
frameTextAtlasRegistry() {
  static std::unordered_map<uint64_t, std::shared_ptr<FrameTextAtlasRecord>>
      registry;
  return registry;
}

// Must be called while frameTextAtlasMutex() is held.
uint64_t allocateFrameTextAtlasIdLocked() {
  auto& registry = frameTextAtlasRegistry();
  constexpr size_t kMaximumLiveAtlases =
      static_cast<size_t>(std::numeric_limits<uint64_t>::max()) - 1u;
  if (registry.size() >= kMaximumLiveAtlases) return 0;

  static uint64_t nextId = 1;
  if (nextId == 0) nextId = 1;
  const uint64_t firstCandidate = nextId;
  do {
    const uint64_t candidate = nextId;
    ++nextId;
    if (nextId == 0) nextId = 1;
    if (candidate != 0 && registry.find(candidate) == registry.end()) {
      return candidate;
    }
  } while (nextId != firstCandidate);
  return 0;
}

std::mutex& scopeDerivedRegistryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uint64_t, ScopeDerivedResidentEntry>&
scopeDerivedRegistry() {
  static std::unordered_map<uint64_t, ScopeDerivedResidentEntry> registry;
  return registry;
}

uint64_t allocateScopeDerivedCacheIdLocked() {
  auto& registry = scopeDerivedRegistry();
  static uint64_t nextId = 1;
  if (nextId == 0) nextId = 1;
  const uint64_t firstCandidate = nextId;
  do {
    const uint64_t candidate = nextId;
    ++nextId;
    if (nextId == 0) nextId = 1;
    if (candidate != 0 && registry.find(candidate) == registry.end()) {
      return candidate;
    }
  } while (nextId != firstCandidate);
  return 0;
}

bool scopeDerivedRecordMatchesCache(
    const ScopeDerivedResidentRecord& record,
    const ScopeDerivedCache& cache) {
  return record.family == cache.family &&
         record.byteSize == cache.byteSize;
}

bool selectScopeDerivedRecordLocked(
    uint64_t cacheId,
    uint64_t producingSubmissionId,
    bool allowOwnPending,
    const ScopeDerivedCache& cache,
    std::shared_ptr<ScopeDerivedResidentRecord>* outRecord) {
  if (outRecord) outRecord->reset();
  auto entryIt = scopeDerivedRegistry().find(cacheId);
  if (entryIt == scopeDerivedRegistry().end()) return false;
  const ScopeDerivedResidentEntry& entry = entryIt->second;
  if (entry.ownerCompositorId != cache.ownerCompositorId) return false;
  if (allowOwnPending && producingSubmissionId != 0) {
    for (const auto& version : entry.inFlight) {
      if (version.state == ScopeDerivedResidentVersionState::Pending &&
          version.submissionId == producingSubmissionId && version.record &&
          scopeDerivedRecordMatchesCache(*version.record, cache)) {
        if (outRecord) *outRecord = version.record;
        return true;
      }
    }
  }
  for (auto versionIt = entry.inFlight.rbegin();
       versionIt != entry.inFlight.rend(); ++versionIt) {
    if (versionIt->state == ScopeDerivedResidentVersionState::Submitted &&
        versionIt->record &&
        scopeDerivedRecordMatchesCache(*versionIt->record, cache)) {
      if (outRecord) *outRecord = versionIt->record;
      return true;
    }
  }
  if (entry.committed &&
      scopeDerivedRecordMatchesCache(*entry.committed, cache)) {
    if (outRecord) *outRecord = entry.committed;
    return true;
  }
  return false;
}

bool resolveScopeDerivedRecordLocked(
    uint64_t cacheId,
    uint64_t producingSubmissionId,
    bool allowOwnPending,
    const ScopeDerivedCache& cache,
    ScopeDerivedResidentRecord* outRecord) {
  std::shared_ptr<ScopeDerivedResidentRecord> selected;
  if (!selectScopeDerivedRecordLocked(cacheId,
                                      producingSubmissionId,
                                      allowOwnPending,
                                      cache,
                                      &selected)) {
    return false;
  }
  if (outRecord) *outRecord = *selected;
  return true;
}

bool scopeDerivedRegistryContainsRecordLocked(
    uint64_t cacheId,
    uint64_t ownerCompositorId,
    const std::shared_ptr<ScopeDerivedResidentRecord>& record) {
  const auto entryIt = scopeDerivedRegistry().find(cacheId);
  if (entryIt == scopeDerivedRegistry().end() ||
      entryIt->second.ownerCompositorId != ownerCompositorId || !record) {
    return false;
  }
  if (entryIt->second.committed == record) return true;
  for (const auto& version : entryIt->second.inFlight) {
    if (version.record == record) return true;
  }
  return false;
}

bool resolveScopeDerivedRecord(const ScopeDerivedCache& cache,
                               uint64_t producingSubmissionId,
                               uint64_t expectedCompositorId,
                               bool allowOwnPending,
                               ScopeDerivedResidentRecord* outRecord) {
  if (cache.cacheId == 0 || cache.byteSize == 0 || !cache.available ||
      cache.ownerCompositorId != expectedCompositorId) {
    return false;
  }
  std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
  return resolveScopeDerivedRecordLocked(cache.cacheId,
                                         producingSubmissionId,
                                         allowOwnPending,
                                         cache,
                                         outRecord);
}

bool resolveScopeDerivedRecordForSubmission(
    const FrameSubmission& submission,
    const ScopeDerivedCache& cache,
    bool allowOwnPending,
    ScopeDerivedResidentRecord* outRecord,
    std::string* error) {
  if (cache.cacheId == 0u || cache.byteSize == 0u || !cache.available) {
    return false;
  }
  if (submission.submissionId == 0u || submission.compositorId == 0u) {
    if (error) *error = "invalid-metal-frame-submission-retention";
    return false;
  }
  if (cache.ownerCompositorId != submission.compositorId) {
    if (error) *error = "metal-resident-derived-cache-owner-mismatch";
    return false;
  }

  std::shared_ptr<ScopeDerivedResidentRecord> record;
  {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    if (!selectScopeDerivedRecordLocked(cache.cacheId,
                                        submission.submissionId,
                                        allowOwnPending,
                                        cache,
                                        &record)) {
      return false;
    }
  }

  const SubmissionRetentionKey key{
      SubmissionRetentionKind::DerivedRecord,
      cache.cacheId,
      submission.compositorId};
  bool added = false;
  if (!retainSubmissionResource(submission, key, record, error, &added)) {
    return false;
  }
  bool stillLive = false;
  {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    stillLive = scopeDerivedRegistryContainsRecordLocked(
        cache.cacheId, submission.compositorId, record);
  }
  if (!stillLive) {
    if (added) (void)releaseSubmissionResource(submission, key);
    if (error) *error = "metal-frame-retention-retired-during-acquire";
    return false;
  }
  if (outRecord) *outRecord = *record;
  return true;
}

bool glossFieldRecordMatchesCache(const GlossFieldResidentRecord& record,
                                  const GlossFieldCache& cache) {
  return record.gridWidth == cache.gridWidth &&
         record.gridHeight == cache.gridHeight &&
         record.byteSize == cache.byteSize;
}

ScopeDerivedCache glossDerivedCache(const GlossFieldCache& cache) {
  ScopeDerivedCache derived{};
  derived.cacheId = cache.cacheId;
  derived.ownerCompositorId = cache.ownerCompositorId;
  derived.builtSerial = cache.builtSerial;
  derived.byteSize = cache.byteSize;
  derived.family = ScopeDerivedFamily::GlossField;
  derived.available = cache.available;
  return derived;
}

bool resolveGlossFieldRecord(const GlossFieldCache& cache,
                             uint64_t producingSubmissionId,
                             uint64_t expectedCompositorId,
                             bool allowOwnPending,
                             GlossFieldResidentRecord* outRecord) {
  if (cache.cacheId == 0 || cache.gridWidth <= 0 ||
      cache.gridHeight <= 0 || cache.builtSerial == 0 ||
      cache.byteSize == 0u || !cache.available) {
    return false;
  }
  ScopeDerivedResidentRecord derived{};
  if (!resolveScopeDerivedRecord(glossDerivedCache(cache),
                                 producingSubmissionId,
                                 expectedCompositorId,
                                 allowOwnPending,
                                 &derived) ||
      derived.family != ScopeDerivedFamily::GlossField ||
      !derived.glossField ||
      !glossFieldRecordMatchesCache(*derived.glossField, cache)) {
    return false;
  }
  if (outRecord) *outRecord = *derived.glossField;
  return true;
}

bool resolveGlossFieldRecordForSubmission(
    const FrameSubmission& submission,
    const GlossFieldCache& cache,
    bool allowOwnPending,
    GlossFieldResidentRecord* outRecord,
    std::string* error) {
  if (cache.cacheId == 0u || cache.gridWidth <= 0 ||
      cache.gridHeight <= 0 || cache.builtSerial == 0u ||
      cache.byteSize == 0u || !cache.available) {
    return false;
  }
  ScopeDerivedResidentRecord derived{};
  if (!resolveScopeDerivedRecordForSubmission(submission,
                                               glossDerivedCache(cache),
                                               allowOwnPending,
                                               &derived,
                                               error)) {
    return false;
  }
  if (derived.family != ScopeDerivedFamily::GlossField ||
      !derived.glossField ||
      !glossFieldRecordMatchesCache(*derived.glossField, cache)) {
    if (error) *error = "metal-gloss-field-cache-record-mismatch";
    return false;
  }
  if (outRecord) *outRecord = *derived.glossField;
  return true;
}

bool addFrameSubmissionTransaction(
    const FrameSubmission& submission,
    FrameSubmissionTransactionRecord transaction,
    std::string* error) {
  if (submission.submissionId == 0 || submission.compositorId == 0) {
    if (error) *error = "invalid-metal-frame-submission-transaction";
    return false;
  }
  std::lock_guard<std::mutex> lock(frameSubmissionMutex());
  auto it = frameSubmissionRegistry().find(submission.submissionId);
  if (it == frameSubmissionRegistry().end()) {
    if (error) *error = "metal-frame-submission-not-found";
    return false;
  }
  if (!submissionIdentityMatches(it->second, submission)) {
    if (error) *error = "metal-frame-submission-transaction-mismatch";
    return false;
  }
  try {
    it->second.transactions.push_back(std::move(transaction));
  } catch (...) {
    if (error) *error = "metal-frame-submission-transaction-allocation-failed";
    return false;
  }
  return true;
}

void markScopeDerivedVersionSubmitted(uint64_t cacheId,
                                      uint64_t submissionId) {
  std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
  auto entryIt = scopeDerivedRegistry().find(cacheId);
  if (entryIt == scopeDerivedRegistry().end()) return;
  for (auto& version : entryIt->second.inFlight) {
    if (version.submissionId == submissionId &&
        version.state == ScopeDerivedResidentVersionState::Pending) {
      version.state = ScopeDerivedResidentVersionState::Submitted;
      return;
    }
  }
}

void abandonScopeDerivedVersion(uint64_t cacheId,
                                uint64_t submissionId) {
  std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
  auto entryIt = scopeDerivedRegistry().find(cacheId);
  if (entryIt == scopeDerivedRegistry().end()) return;
  auto& versions = entryIt->second.inFlight;
  versions.erase(
      std::remove_if(
          versions.begin(), versions.end(),
          [&](const ScopeDerivedResidentVersion& version) {
            return version.submissionId == submissionId &&
                   version.state == ScopeDerivedResidentVersionState::Pending;
          }),
      versions.end());
  if (!entryIt->second.committed && versions.empty()) {
    scopeDerivedRegistry().erase(entryIt);
  }
}

void completeScopeDerivedVersion(uint64_t cacheId,
                                 uint64_t submissionId,
                                 bool success) {
  std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
  auto entryIt = scopeDerivedRegistry().find(cacheId);
  if (entryIt == scopeDerivedRegistry().end()) return;
  ScopeDerivedResidentEntry& entry = entryIt->second;
  auto versionIt = std::find_if(
      entry.inFlight.begin(), entry.inFlight.end(),
      [&](const ScopeDerivedResidentVersion& version) {
        return version.submissionId == submissionId &&
               version.state == ScopeDerivedResidentVersionState::Submitted;
      });
  if (versionIt == entry.inFlight.end()) return;
  if (success && versionIt->record) entry.committed = versionIt->record;
  entry.inFlight.erase(versionIt);
}

bool registerPendingScopeDerivedRecord(
    const FrameSubmission& submission,
    ScopeDerivedCache* cache,
    ScopeDerivedResidentRecord record,
    std::string* error) {
  if (!cache || submission.submissionId == 0u ||
      submission.compositorId == 0u || record.builtSerial == 0u ||
      record.byteSize == 0u ||
      (cache->ownerCompositorId != 0u &&
       cache->ownerCompositorId != submission.compositorId)) {
    if (error) *error = "invalid-metal-resident-derived-record";
    return false;
  }
  uint64_t cacheId = cache->cacheId;
  std::shared_ptr<ScopeDerivedResidentRecord> pendingRecord;
  try {
    pendingRecord =
        std::make_shared<ScopeDerivedResidentRecord>(std::move(record));
  } catch (...) {
    if (error) *error = "metal-resident-derived-cache-record-allocation-failed";
    return false;
  }
  if (cache->available && cache->byteSize != 0 &&
      cache->byteSize != pendingRecord->byteSize) {
    if (error) *error = "metal-resident-derived-cache-byte-size-changed";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    const bool cacheKnown =
        cacheId != 0 &&
        scopeDerivedRegistry().find(cacheId) != scopeDerivedRegistry().end();
    if (cacheKnown &&
        (cache->ownerCompositorId != submission.compositorId ||
         scopeDerivedRegistry().find(cacheId)->second.ownerCompositorId !=
             submission.compositorId)) {
      if (error) *error = "metal-resident-derived-cache-owner-mismatch";
      return false;
    }
    if (!cacheKnown) cacheId = allocateScopeDerivedCacheIdLocked();
    if (cacheId == 0) {
      if (error) *error = "metal-resident-derived-cache-handle-exhausted";
      return false;
    }
    ScopeDerivedResidentEntry* entry = nullptr;
    try {
      entry = &scopeDerivedRegistry()[cacheId];
    } catch (...) {
      if (error) *error = "metal-resident-derived-cache-registry-allocation-failed";
      return false;
    }
    if (entry->ownerCompositorId != 0u &&
        entry->ownerCompositorId != submission.compositorId) {
      if (error) *error = "metal-resident-derived-entry-owner-mismatch";
      return false;
    }
    if (entry->committed || !entry->inFlight.empty()) {
      if (error) *error = "metal-resident-derived-cache-already-materialized";
      return false;
    }
    entry->ownerCompositorId = submission.compositorId;
    ScopeDerivedResidentVersion version{};
    version.submissionId = submission.submissionId;
    version.record = pendingRecord;
    try {
      entry->inFlight.push_back(std::move(version));
    } catch (...) {
      if (!entry->committed && entry->inFlight.empty()) {
        scopeDerivedRegistry().erase(cacheId);
      }
      if (error) *error = "metal-resident-derived-cache-version-allocation-failed";
      return false;
    }
  }
  const SubmissionRetentionKey retentionKey{
      SubmissionRetentionKind::DerivedRecord,
      cacheId,
      submission.compositorId};
  bool retentionAdded = false;
  if (!retainSubmissionResource(submission,
                                retentionKey,
                                pendingRecord,
                                error,
                                &retentionAdded)) {
    abandonScopeDerivedVersion(cacheId, submission.submissionId);
    return false;
  }
  bool pendingStillLive = false;
  {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    pendingStillLive = scopeDerivedRegistryContainsRecordLocked(
        cacheId, submission.compositorId, pendingRecord);
  }
  if (!pendingStillLive) {
    if (retentionAdded) {
      (void)releaseSubmissionResource(submission, retentionKey);
    }
    abandonScopeDerivedVersion(cacheId, submission.submissionId);
    if (error) *error = "metal-frame-retention-retired-during-acquire";
    return false;
  }
  FrameSubmissionTransactionRecord transaction{};
  transaction.submitted = [cacheId, submissionId = submission.submissionId]() {
    markScopeDerivedVersionSubmitted(cacheId, submissionId);
  };
  transaction.completed =
      [cacheId, submissionId = submission.submissionId](bool success) {
        completeScopeDerivedVersion(cacheId, submissionId, success);
      };
  transaction.abandoned = [cacheId, submissionId = submission.submissionId]() {
    abandonScopeDerivedVersion(cacheId, submissionId);
  };
  if (!addFrameSubmissionTransaction(submission, std::move(transaction), error)) {
    abandonScopeDerivedVersion(cacheId, submission.submissionId);
    if (retentionAdded) {
      (void)releaseSubmissionResource(submission, retentionKey);
    }
    return false;
  }
  cache->cacheId = cacheId;
  cache->ownerCompositorId = submission.compositorId;
  cache->builtSerial = pendingRecord->builtSerial;
  cache->byteSize = pendingRecord->byteSize;
  cache->family = pendingRecord->family;
  cache->available = true;
  return true;
}

bool registerCommittedScopeDerivedRecord(
    ScopeDerivedCache* cache,
    ScopeDerivedResidentRecord record,
    std::string* error) {
  if (!cache || cache->ownerCompositorId != 0u ||
      record.builtSerial == 0u || record.byteSize == 0u) {
    if (error) *error = "invalid-metal-committed-derived-record";
    return false;
  }
  std::shared_ptr<ScopeDerivedResidentRecord> committedRecord;
  try {
    committedRecord =
        std::make_shared<ScopeDerivedResidentRecord>(std::move(record));
  } catch (...) {
    if (error) *error = "metal-committed-derived-record-allocation-failed";
    return false;
  }
  uint64_t cacheId = cache->cacheId;
  {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    const bool cacheKnown =
        cacheId != 0u &&
        scopeDerivedRegistry().find(cacheId) != scopeDerivedRegistry().end();
    if (!cacheKnown) cacheId = allocateScopeDerivedCacheIdLocked();
    if (cacheId == 0u) {
      if (error) *error = "metal-committed-derived-handle-exhausted";
      return false;
    }
    ScopeDerivedResidentEntry* entry = nullptr;
    try {
      entry = &scopeDerivedRegistry()[cacheId];
    } catch (...) {
      if (error) *error = "metal-committed-derived-registry-allocation-failed";
      return false;
    }
    if (entry->ownerCompositorId != 0u) {
      if (error) *error = "metal-committed-derived-owner-mismatch";
      return false;
    }
    if (!entry->inFlight.empty()) {
      if (error) *error = "metal-committed-derived-in-flight-conflict";
      return false;
    }
    entry->committed = committedRecord;
  }
  cache->cacheId = cacheId;
  cache->ownerCompositorId = 0u;
  cache->builtSerial = committedRecord->builtSerial;
  cache->byteSize = committedRecord->byteSize;
  cache->family = committedRecord->family;
  cache->available = true;
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
id<MTLTexture> makeTextureFromIOSurface(MetalContext& ctx,
                                        uint32_t surfaceId,
                                        int width,
                                        int height,
                                        int pixelFormat,
                                        std::string* error) {
  if (surfaceId == 0 || width <= 0 || height <= 0) {
    if (error) *error = "invalid-iosurface-texture-request";
    return nil;
  }
  IOSurfaceRef surface = IOSurfaceLookup(static_cast<IOSurfaceID>(surfaceId));
  if (surface == nullptr) {
    if (error) *error = "iosurface-lookup-failed";
    return nil;
  }
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sourceSignalMetalPixelFormat(pixelFormat)
                                                         width:static_cast<NSUInteger>(width)
                                                        height:static_cast<NSUInteger>(height)
                                                     mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead;
  desc.storageMode = MTLStorageModeShared;
  id<MTLTexture> texture = [ctx.device newTextureWithDescriptor:desc iosurface:surface plane:0];
  CFRelease(surface);
  if (texture == nil && error) *error = "iosurface-metal-texture-failed";
  return texture;
}
#endif



constexpr const char* kViewerMetalLibraryName = "ChromaspaceViewer.metallib";

std::string utf8String(NSString* value) {
  if (value == nil) return {};
  const char* utf8 = [value UTF8String];
  return utf8 != nullptr ? std::string(utf8) : std::string();
}

void appendUniqueMetalLibraryCandidate(std::vector<std::string>& candidates,
                                       NSString* path) {
  if (path == nil || [path length] == 0) return;
  NSString* normalized = [[path stringByExpandingTildeInPath] stringByStandardizingPath];
  const std::string candidate = utf8String(normalized);
  if (candidate.empty()) return;
  if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
    candidates.push_back(candidate);
  }
}

std::vector<std::string> viewerMetalLibraryCandidates() {
  std::vector<std::string> candidates;
  NSDictionary<NSString*, NSString*>* environment = [[NSProcessInfo processInfo] environment];
  NSString* overridePath = environment[@"CHROMASPACE_METALLIB_PATH"];
  if (overridePath != nil && [overridePath length] > 0) {
    appendUniqueMetalLibraryCandidate(candidates, overridePath);
    return candidates;
  }

  uint32_t executablePathSize = 0;
  _NSGetExecutablePath(nullptr, &executablePathSize);
  if (executablePathSize > 0) {
    std::vector<char> executablePath(executablePathSize, '\0');
    if (_NSGetExecutablePath(executablePath.data(), &executablePathSize) == 0) {
      NSString* path = [NSString stringWithUTF8String:executablePath.data()];
      NSString* directory = [[path stringByResolvingSymlinksInPath] stringByDeletingLastPathComponent];
      appendUniqueMetalLibraryCandidate(candidates,
                                        [directory stringByAppendingPathComponent:@"ChromaspaceViewer.metallib"]);
    }
  }

  NSBundle* mainBundle = [NSBundle mainBundle];
  NSURL* bundleExecutable = [mainBundle executableURL];
  if (bundleExecutable != nil) {
    appendUniqueMetalLibraryCandidate(
        candidates,
        [[[bundleExecutable URLByDeletingLastPathComponent]
            URLByAppendingPathComponent:@"ChromaspaceViewer.metallib"] path]);
  }
  appendUniqueMetalLibraryCandidate(
      candidates,
      [[mainBundle URLForResource:@"ChromaspaceViewer" withExtension:@"metallib"] path]);
  return candidates;
}

id<MTLLibrary> loadViewerMetalLibrary(id<MTLDevice> device, std::string* error) {
  const std::vector<std::string> candidates = viewerMetalLibraryCandidates();
  std::ostringstream attempts;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i > 0) attempts << "; ";
    const std::string& candidate = candidates[i];
    NSString* path = [NSString stringWithUTF8String:candidate.c_str()];
    BOOL isDirectory = NO;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path isDirectory:&isDirectory]) {
      attempts << candidate << " (not found)";
      continue;
    }
    if (isDirectory) {
      attempts << candidate << " (is a directory)";
      continue;
    }
    NSError* libraryError = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                error:&libraryError];
    if (library != nil) return library;
    attempts << candidate << " (load error: ";
    if (libraryError != nil) {
      attempts << utf8String([libraryError localizedDescription]);
    } else {
      attempts << "unknown Metal library error";
    }
    attempts << ")";
  }

  if (error != nullptr) {
    std::ostringstream message;
    message << "Failed to load precompiled Metal library " << kViewerMetalLibraryName << ". Attempts: ";
    if (candidates.empty()) {
      message << "no executable or bundle candidate could be resolved";
    } else {
      message << attempts.str();
    }
    *error = message.str();
  }
  return nil;
}

bool initializeMetalContext(MetalContext* context, std::string* error) {
  if (context == nullptr) {
    if (error) *error = "metal-context-output-missing";
    return false;
  }
  MetalContext& c = *context;
  if (c.initAttempted) {
    if (!c.ready && error) *error = c.initError;
    return c.ready;
  }
  c.initAttempted = true;
  auto initializeResources = [&]() {
    @autoreleasepool {
      c.device = MTLCreateSystemDefaultDevice();
      if (c.device == nil) {
        c.initError = "No Metal device available.";
        return;
      }
      c.deviceName = [[c.device name] UTF8String] ?: "";
      c.queue = [c.device newCommandQueue];
      if (c.queue == nil) {
        c.initError = "Failed to create Metal command queue.";
        return;
      }
      c.library = loadViewerMetalLibrary(c.device, &c.initError);
      if (c.library == nil) {
        return;
      }
      NSError* pipelineError = nil;
      id<MTLFunction> overlayFn = [c.library newFunctionWithName:@"overlayKernel"];
      if (overlayFn == nil) {
        c.initError = "Missing overlay Metal kernel.";
        return;
      }
      c.overlayPipeline = [c.device newComputePipelineStateWithFunction:overlayFn error:&pipelineError];
      if (c.overlayPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create overlay Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> inputFn = [c.library newFunctionWithName:@"inputKernel"];
      if (inputFn == nil) {
        c.initError = "Missing input Metal kernel.";
        return;
      }
      c.inputPipeline = [c.device newComputePipelineStateWithFunction:inputFn error:&pipelineError];
      if (c.inputPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create input Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterSourceFn = [c.library newFunctionWithName:@"rasterSourceKernel"];
      if (rasterSourceFn == nil) {
        c.initError = "Missing raster source Metal kernel.";
        return;
      }
      c.rasterSourcePipeline = [c.device newComputePipelineStateWithFunction:rasterSourceFn error:&pipelineError];
      if (c.rasterSourcePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster source Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterOccupancyFn = [c.library newFunctionWithName:@"rasterOccupancyCountKernel"];
      if (rasterOccupancyFn == nil) {
        c.initError = "Missing raster occupancy Metal kernel.";
        return;
      }
      c.rasterOccupancyCountPipeline = [c.device newComputePipelineStateWithFunction:rasterOccupancyFn error:&pipelineError];
      if (c.rasterOccupancyCountPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster occupancy Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterSourceTextureFn = [c.library newFunctionWithName:@"rasterSourceTextureKernel"];
      if (rasterSourceTextureFn == nil) {
        c.initError = "Missing raster source texture Metal kernel.";
        return;
      }
      c.rasterSourceTexturePipeline = [c.device newComputePipelineStateWithFunction:rasterSourceTextureFn error:&pipelineError];
      if (c.rasterSourceTexturePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster source texture Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterOccupancyTextureFn = [c.library newFunctionWithName:@"rasterOccupancyTextureCountKernel"];
      if (rasterOccupancyTextureFn == nil) {
        c.initError = "Missing raster occupancy texture Metal kernel.";
        return;
      }
      c.rasterOccupancyTextureCountPipeline = [c.device newComputePipelineStateWithFunction:rasterOccupancyTextureFn error:&pipelineError];
      if (c.rasterOccupancyTextureCountPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create raster occupancy texture Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> rasterOccupancyThresholdFn =
          [c.library newFunctionWithName:@"rasterOccupancyThresholdKernel"];
      if (rasterOccupancyThresholdFn == nil) {
        c.initError = "Missing raster occupancy threshold Metal kernel.";
        return;
      }
      c.rasterOccupancyThresholdPipeline =
          [c.device newComputePipelineStateWithFunction:rasterOccupancyThresholdFn
                                                 error:&pipelineError];
      if (c.rasterOccupancyThresholdPipeline == nil) {
        c.initError =
            pipelineError != nil
                ? [[pipelineError localizedDescription] UTF8String]
                : "Failed to create raster occupancy threshold Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> inputSampleFn = [c.library newFunctionWithName:@"inputSampleKernel"];
      if (inputSampleFn == nil) {
        c.initError = "Missing input sample Metal kernel.";
        return;
      }
      c.inputSamplePipeline = [c.device newComputePipelineStateWithFunction:inputSampleFn error:&pipelineError];
      if (c.inputSamplePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : "Failed to create input sample Metal pipeline.";
        return;
      }
      auto buildPipeline = [&](NSString* name, const char* missingMsg, const char* failMsg) -> id<MTLComputePipelineState> {
        pipelineError = nil;
        id<MTLFunction> fn = [c.library newFunctionWithName:name];
        if (fn == nil) {
          c.initError = missingMsg;
          return nil;
        }
        id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithFunction:fn error:&pipelineError];
        if (pipeline == nil) {
          c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String] : failMsg;
          return nil;
        }
        return pipeline;
      };
      c.rasterPointCompactLocalScanPipeline =
          buildPipeline(@"rasterPointCompactLocalScanKernel",
                        "Missing raster point compact local-scan Metal kernel.",
                        "Failed to create raster point compact local-scan Metal pipeline.");
      if (c.rasterPointCompactLocalScanPipeline == nil) {
        return;
      }
      c.rasterPointScanBlockSumsPipeline =
          buildPipeline(@"rasterPointScanBlockSumsKernel",
                        "Missing raster point block-scan Metal kernel.",
                        "Failed to create raster point block-scan Metal pipeline.");
      if (c.rasterPointScanBlockSumsPipeline == nil) {
        return;
      }
      c.rasterPointAddBlockOffsetsPipeline =
          buildPipeline(@"rasterPointAddBlockOffsetsKernel",
                        "Missing raster point block-offset Metal kernel.",
                        "Failed to create raster point block-offset Metal pipeline.");
      if (c.rasterPointAddBlockOffsetsPipeline == nil) {
        return;
      }
      c.rasterPointCompactScatterPipeline =
          buildPipeline(@"rasterPointCompactScatterKernel",
                        "Missing raster point compact-scatter Metal kernel.",
                        "Failed to create raster point compact-scatter Metal pipeline.");
      if (c.rasterPointCompactScatterPipeline == nil) {
        return;
      }
      c.rasterPointFinalizeIndirectArgsPipeline =
          buildPipeline(@"rasterPointFinalizeIndirectArgsKernel",
                        "Missing raster point indirect-args Metal kernel.",
                        "Failed to create raster point indirect-args Metal pipeline.");
      if (c.rasterPointFinalizeIndirectArgsPipeline == nil) {
        return;
      }
      c.scopeDensityPipeline = buildPipeline(@"scopeDensityKernel",
                                             "Missing scope density Metal kernel.",
                                             "Failed to create scope density Metal pipeline.");
      if (c.scopeDensityPipeline == nil) {
        return;
      }
      c.rasterScopeDensityTexturePipeline = buildPipeline(@"rasterScopeDensityTextureKernel",
                                                          "Missing raster scope density texture Metal kernel.",
                                                          "Failed to create raster scope density texture Metal pipeline.");
      if (c.rasterScopeDensityTexturePipeline == nil) {
        return;
      }
      c.rasterScopeRangeTexturePipeline = buildPipeline(@"rasterScopeRangeTextureKernel",
                                                        "Missing raster scope range texture Metal kernel.",
                                                        "Failed to create raster scope range texture Metal pipeline.");
      if (c.rasterScopeRangeTexturePipeline == nil) {
        return;
      }
      c.rasterScopeRangeHistogramTexturePipeline =
          buildPipeline(@"rasterScopeRangeHistogramTextureKernel",
                        "Missing raster scope range histogram texture Metal kernel.",
                        "Failed to create raster scope range histogram texture Metal pipeline.");
      if (c.rasterScopeRangeHistogramTexturePipeline == nil) {
        return;
      }
      c.scopeRangeHistogramPercentilePipeline =
          buildPipeline(@"scopeRangeHistogramPercentileKernel",
                        "Missing scope range histogram percentile Metal kernel.",
                        "Failed to create scope range histogram percentile Metal pipeline.");
      if (c.scopeRangeHistogramPercentilePipeline == nil) {
        return;
      }
      c.scopeRangeFinalizePipeline = buildPipeline(@"scopeRangeFinalizeKernel",
                                                   "Missing scope range finalize Metal kernel.",
                                                   "Failed to create scope range finalize Metal pipeline.");
      if (c.scopeRangeFinalizePipeline == nil) {
        return;
      }
      c.histogramApplyRangePipeline =
          buildPipeline(@"histogramSurfaceApplyRangeKernel",
                        "Missing histogram surface apply-range Metal kernel.",
                        "Failed to create histogram surface apply-range Metal pipeline.");
      if (c.histogramApplyRangePipeline == nil) {
        return;
      }
      c.histogramMaxPipeline = buildPipeline(@"histogramSurfaceMaxKernel",
                                             "Missing histogram surface max Metal kernel.",
                                             "Failed to create histogram surface max Metal pipeline.");
      if (c.histogramMaxPipeline == nil) {
        return;
      }
      c.histogramSurfaceRenderPipeline = buildPipeline(@"histogramSurfaceRenderKernel",
                                                       "Missing histogram surface render Metal kernel.",
                                                       "Failed to create histogram surface render Metal pipeline.");
      if (c.histogramSurfaceRenderPipeline == nil) {
        return;
      }
      c.waveformApplyRangePipeline =
          buildPipeline(@"waveformSurfaceApplyRangeKernel",
                        "Missing waveform surface apply-range Metal kernel.",
                        "Failed to create waveform surface apply-range Metal pipeline.");
      if (c.waveformApplyRangePipeline == nil) {
        return;
      }
      c.waveformMaxPipeline = buildPipeline(@"waveformSurfaceMaxKernel",
                                            "Missing waveform surface max Metal kernel.",
                                            "Failed to create waveform surface max Metal pipeline.");
      if (c.waveformMaxPipeline == nil) {
        return;
      }
      c.waveformSurfaceRenderPipeline = buildPipeline(@"waveformSurfaceRenderKernel",
                                                      "Missing waveform surface render Metal kernel.",
                                                      "Failed to create waveform surface render Metal pipeline.");
      if (c.waveformSurfaceRenderPipeline == nil) {
        return;
      }
      c.glossFieldAccumulatePipeline = buildPipeline(@"glossFieldAccumulateKernel",
                                                     "Missing gloss field accumulate Metal kernel.",
                                                     "Failed to create gloss field accumulate Metal pipeline.");
      if (c.glossFieldAccumulatePipeline == nil) {
        return;
      }
      c.rasterGlossFieldAccumulateTexturePipeline =
          buildPipeline(@"rasterGlossFieldAccumulateTextureKernel",
                        "Missing raster gloss field accumulate texture Metal kernel.",
                        "Failed to create raster gloss field accumulate texture Metal pipeline.");
      if (c.rasterGlossFieldAccumulateTexturePipeline == nil) {
        return;
      }
      c.glossFieldFinalizePipeline = buildPipeline(@"glossFieldFinalizeKernel",
                                                   "Missing gloss field finalize Metal kernel.",
                                                   "Failed to create gloss field finalize Metal pipeline.");
      if (c.glossFieldFinalizePipeline == nil) {
        return;
      }
      c.glossFieldMaxPipeline = buildPipeline(@"glossFieldMaxKernel",
                                              "Missing gloss field max Metal kernel.",
                                              "Failed to create gloss field max Metal pipeline.");
      if (c.glossFieldMaxPipeline == nil) {
        return;
      }
      c.glossFieldNormalizePipeline = buildPipeline(@"glossFieldNormalizeKernel",
                                                    "Missing gloss field normalize Metal kernel.",
                                                    "Failed to create gloss field normalize Metal pipeline.");
      if (c.glossFieldNormalizePipeline == nil) {
        return;
      }
      c.glossFieldBlurPipeline = buildPipeline(@"glossFieldBlurKernel",
                                               "Missing gloss field blur Metal kernel.",
                                               "Failed to create gloss field blur Metal pipeline.");
      if (c.glossFieldBlurPipeline == nil) {
        return;
      }
      c.glossFieldBodyPipeline = buildPipeline(@"glossFieldBodyKernel",
                                               "Missing gloss field body Metal kernel.",
                                               "Failed to create gloss field body Metal pipeline.");
      if (c.glossFieldBodyPipeline == nil) {
        return;
      }
      c.glossFieldRawSignalPipeline = buildPipeline(@"glossFieldRawSignalKernel",
                                                    "Missing gloss field raw signal Metal kernel.",
                                                    "Failed to create gloss field raw signal Metal pipeline.");
      if (c.glossFieldRawSignalPipeline == nil) {
        return;
      }
      c.glossFieldWeightedSignalPipeline = buildPipeline(@"glossFieldWeightedSignalKernel",
                                                         "Missing gloss field weighted signal Metal kernel.",
                                                         "Failed to create gloss field weighted signal Metal pipeline.");
      if (c.glossFieldWeightedSignalPipeline == nil) {
        return;
      }
      c.glossFieldMergeMaxBitsPipeline = buildPipeline(@"glossFieldMergeMaxBitsKernel",
                                                       "Missing gloss field merge max-bits Metal kernel.",
                                                       "Failed to create gloss field merge max-bits Metal pipeline.");
      if (c.glossFieldMergeMaxBitsPipeline == nil) {
        return;
      }
      c.glossFieldFinalNormalizePipeline = buildPipeline(@"glossFieldFinalNormalizeKernel",
                                                         "Missing gloss field final normalize Metal kernel.",
                                                         "Failed to create gloss field final normalize Metal pipeline.");
      if (c.glossFieldFinalNormalizePipeline == nil) {
        return;
      }
      c.glossFieldLocalPercentilePipeline = buildPipeline(@"glossFieldLocalPercentileKernel",
                                                          "Missing gloss field local percentile Metal kernel.",
                                                          "Failed to create gloss field local percentile Metal pipeline.");
      if (c.glossFieldLocalPercentilePipeline == nil) {
        return;
      }
      c.glossFieldCandidate2RawPipeline = buildPipeline(@"glossFieldCandidate2RawKernel",
                                                        "Missing gloss field candidate 2 raw Metal kernel.",
                                                        "Failed to create gloss field candidate 2 raw Metal pipeline.");
      if (c.glossFieldCandidate2RawPipeline == nil) {
        return;
      }
      c.glossFieldAssembleUnifiedPipeline = buildPipeline(@"glossFieldAssembleUnifiedKernel",
                                                          "Missing gloss field assemble unified Metal kernel.",
                                                          "Failed to create gloss field assemble unified Metal pipeline.");
      if (c.glossFieldAssembleUnifiedPipeline == nil) {
        return;
      }
      c.glossFieldSurfaceRenderPipeline = buildPipeline(@"glossFieldSurfaceRenderKernel",
                                                        "Missing gloss field surface render Metal kernel.",
                                                        "Failed to create gloss field surface render Metal pipeline.");
      if (c.glossFieldSurfaceRenderPipeline == nil) {
        return;
      }
      c.glossProjectionSurfaceSelectPipeline = buildPipeline(@"glossProjectionSurfaceSelectKernel",
                                                             "Missing gloss projection surface select Metal kernel.",
                                                             "Failed to create gloss projection surface select Metal pipeline.");
      if (c.glossProjectionSurfaceSelectPipeline == nil) {
        return;
      }
      c.glossProjectionSurfaceShadePipeline = buildPipeline(@"glossProjectionSurfaceShadeKernel",
                                                            "Missing gloss projection surface shade Metal kernel.",
                                                            "Failed to create gloss projection surface shade Metal pipeline.");
      if (c.glossProjectionSurfaceShadePipeline == nil) {
        return;
      }
      c.plotSurfaceClearPipeline = buildPipeline(@"plotSurfaceClearKernel",
                                                 "Missing plot surface clear Metal kernel.",
                                                 "Failed to create plot surface clear Metal pipeline.");
      if (c.plotSurfaceClearPipeline == nil) {
        return;
      }
      c.sourceSignalSurfacePipeline = buildPipeline(@"sourceSignalSurfaceRenderKernel",
                                                    "Missing source signal surface Metal kernel.",
                                                    "Failed to create source signal surface Metal pipeline.");
      if (c.sourceSignalSurfacePipeline == nil) {
        return;
      }
      id<MTLFunction> plotSurfaceVectorVertexFn = [c.library newFunctionWithName:@"frameUiVectorVertex"];
      id<MTLFunction> plotSurfaceVectorFragmentFn = [c.library newFunctionWithName:@"frameUiVectorFragment"];
      if (plotSurfaceVectorVertexFn == nil || plotSurfaceVectorFragmentFn == nil) {
        c.initError = "Missing plot surface vector Metal shaders.";
        return;
      }
      auto buildPlotSurfaceVectorPipeline =
          [&](MTLPixelFormat pixelFormat, const char* label, bool required) -> id<MTLRenderPipelineState> {
        NSError* renderPipelineError = nil;
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = plotSurfaceVectorVertexFn;
        desc.fragmentFunction = plotSurfaceVectorFragmentFn;
        desc.colorAttachments[0].pixelFormat = pixelFormat;
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        id<MTLRenderPipelineState> pipeline =
            [c.device newRenderPipelineStateWithDescriptor:desc error:&renderPipelineError];
        if (pipeline == nil && required) {
          if (renderPipelineError != nil) {
            c.initError = [[renderPipelineError localizedDescription] UTF8String];
          } else {
            c.initError = std::string("Failed to create ") + label + " plot surface vector pipeline.";
          }
        }
        return pipeline;
      };
      c.plotSurfaceVectorPipeline16 =
          buildPlotSurfaceVectorPipeline(MTLPixelFormatRGBA16Float, "RGBA16F", true);
      if (c.plotSurfaceVectorPipeline16 == nil) {
        return;
      }
      c.plotSurfaceVectorPipeline32 =
          buildPlotSurfaceVectorPipeline(MTLPixelFormatRGBA32Float, "RGBA32F", false);
      id<MTLFunction> rasterPointVertexFn = [c.library newFunctionWithName:@"rasterPointSurfaceVertex"];
      id<MTLFunction> rasterPointFragmentFn = [c.library newFunctionWithName:@"rasterPointSurfaceFragment"];
      if (rasterPointVertexFn == nil || rasterPointFragmentFn == nil) {
        c.initError = "Missing raster point surface Metal shaders.";
        return;
      }
      auto buildRasterPointSurfacePipeline =
          [&](MTLPixelFormat pixelFormat, const char* label, bool required) -> id<MTLRenderPipelineState> {
        NSError* renderPipelineError = nil;
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = rasterPointVertexFn;
        desc.fragmentFunction = rasterPointFragmentFn;
        desc.colorAttachments[0].pixelFormat = pixelFormat;
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        desc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        id<MTLRenderPipelineState> pipeline =
            [c.device newRenderPipelineStateWithDescriptor:desc error:&renderPipelineError];
        if (pipeline == nil && required) {
          if (renderPipelineError != nil) {
            c.initError = [[renderPipelineError localizedDescription] UTF8String];
          } else {
            c.initError = std::string("Failed to create ") + label + " raster point surface pipeline.";
          }
        }
        return pipeline;
      };
      c.rasterPointSurfacePipeline16 =
          buildRasterPointSurfacePipeline(MTLPixelFormatRGBA16Float, "RGBA16F", true);
      if (c.rasterPointSurfacePipeline16 == nil) {
        return;
      }
      c.rasterPointSurfacePipeline32 =
          buildRasterPointSurfacePipeline(MTLPixelFormatRGBA32Float, "RGBA32F", false);
      id<MTLFunction> compositeVertexFn = [c.library newFunctionWithName:@"frameSurfaceCompositeVertex"];
      id<MTLFunction> compositeFragmentFn = [c.library newFunctionWithName:@"frameSurfaceCompositeFragment"];
      if (compositeVertexFn == nil || compositeFragmentFn == nil) {
        c.initError = "Missing frame surface composite Metal shaders.";
        return;
      }
      MTLRenderPipelineDescriptor* compositeDesc = [[MTLRenderPipelineDescriptor alloc] init];
      compositeDesc.vertexFunction = compositeVertexFn;
      compositeDesc.fragmentFunction = compositeFragmentFn;
      compositeDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      compositeDesc.colorAttachments[0].blendingEnabled = YES;
      compositeDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      compositeDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      compositeDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      compositeDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      compositeDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      compositeDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      c.frameSurfaceCompositePipeline =
          [c.device newRenderPipelineStateWithDescriptor:compositeDesc error:&pipelineError];
      if (c.frameSurfaceCompositePipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String]
                                           : "Failed to create frame surface composite Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> solidRectVertexFn = [c.library newFunctionWithName:@"frameSolidRectVertex"];
      id<MTLFunction> solidRectFragmentFn = [c.library newFunctionWithName:@"frameSolidRectFragment"];
      if (solidRectVertexFn == nil || solidRectFragmentFn == nil) {
        c.initError = "Missing frame solid-rect Metal shaders.";
        return;
      }
      MTLRenderPipelineDescriptor* solidRectDesc = [[MTLRenderPipelineDescriptor alloc] init];
      solidRectDesc.vertexFunction = solidRectVertexFn;
      solidRectDesc.fragmentFunction = solidRectFragmentFn;
      solidRectDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      solidRectDesc.colorAttachments[0].blendingEnabled = YES;
      solidRectDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      solidRectDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      solidRectDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      solidRectDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      solidRectDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      solidRectDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      c.frameSolidRectPipeline =
          [c.device newRenderPipelineStateWithDescriptor:solidRectDesc error:&pipelineError];
      if (c.frameSolidRectPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String]
                                           : "Failed to create frame solid-rect Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> uiVectorVertexFn = [c.library newFunctionWithName:@"frameUiVectorVertex"];
      id<MTLFunction> uiVectorFragmentFn = [c.library newFunctionWithName:@"frameUiVectorFragment"];
      if (uiVectorVertexFn == nil || uiVectorFragmentFn == nil) {
        c.initError = "Missing frame UI-vector Metal shaders.";
        return;
      }
      MTLRenderPipelineDescriptor* uiVectorDesc = [[MTLRenderPipelineDescriptor alloc] init];
      uiVectorDesc.vertexFunction = uiVectorVertexFn;
      uiVectorDesc.fragmentFunction = uiVectorFragmentFn;
      uiVectorDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      uiVectorDesc.colorAttachments[0].blendingEnabled = YES;
      uiVectorDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      uiVectorDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      uiVectorDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      uiVectorDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      uiVectorDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      uiVectorDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      c.frameUiVectorPipeline =
          [c.device newRenderPipelineStateWithDescriptor:uiVectorDesc error:&pipelineError];
      if (c.frameUiVectorPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String]
                                           : "Failed to create frame UI-vector Metal pipeline.";
        return;
      }
      pipelineError = nil;
      id<MTLFunction> textVertexFn = [c.library newFunctionWithName:@"frameTextVertex"];
      id<MTLFunction> textFragmentFn = [c.library newFunctionWithName:@"frameTextFragment"];
      if (textVertexFn == nil || textFragmentFn == nil) {
        c.initError = "Missing frame text Metal shaders.";
        return;
      }
      MTLRenderPipelineDescriptor* textDesc = [[MTLRenderPipelineDescriptor alloc] init];
      textDesc.vertexFunction = textVertexFn;
      textDesc.fragmentFunction = textFragmentFn;
      textDesc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
      textDesc.colorAttachments[0].blendingEnabled = YES;
      textDesc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      textDesc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      textDesc.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
      textDesc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
      textDesc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      textDesc.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
      c.frameTextPipeline =
          [c.device newRenderPipelineStateWithDescriptor:textDesc error:&pipelineError];
      if (c.frameTextPipeline == nil) {
        c.initError = pipelineError != nil ? [[pipelineError localizedDescription] UTF8String]
                                           : "Failed to create frame text Metal pipeline.";
        return;
      }
      c.ready = true;
    }
  };
  try {
    initializeResources();
  } catch (...) {
    c.ready = false;
    if (c.initError.empty()) {
      c.initError = "metal-context-initialization-exception";
    }
  }
  if (!c.ready && error) *error = c.initError;
  return c.ready;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool ensureContext(std::string* error) {
  return initializeMetalContext(&context(), error);
}

bool runCompute(id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> inputBuffer,
                id<MTLBuffer> vertBuffer,
                id<MTLBuffer> colorBuffer,
                id<MTLBuffer> uniformBuffer,
                NSUInteger pointCount,
                std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || pipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal context unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal compute encoder.";
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    if (inputBuffer != nil) [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:vertBuffer offset:0 atIndex:1];
    [encoder setBuffer:colorBuffer offset:0 atIndex:2];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:3];
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(pointCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

bool runInputSampleCompute(id<MTLBuffer> srcVertBuffer,
                           id<MTLBuffer> srcColorBuffer,
                           id<MTLBuffer> dstVertBuffer,
                           id<MTLBuffer> dstColorBuffer,
                           id<MTLBuffer> uniformBuffer,
                           NSUInteger pointCount,
                           std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || ctx.inputSamplePipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal input sample pipeline unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal sample command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal sample encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.inputSamplePipeline];
    [encoder setBuffer:srcVertBuffer offset:0 atIndex:0];
    [encoder setBuffer:srcColorBuffer offset:0 atIndex:1];
    [encoder setBuffer:dstVertBuffer offset:0 atIndex:2];
    [encoder setBuffer:dstColorBuffer offset:0 atIndex:3];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:4];
    NSUInteger width = ctx.inputSamplePipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(pointCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

template <size_t N>
bool runComputeBuffers(id<MTLComputePipelineState> pipeline,
                       const std::array<id<MTLBuffer>, N>& buffers,
                       NSUInteger threadCount,
                       std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || pipeline == nil) {
    if (error && error->empty()) *error = ctx.initError.empty() ? "Metal compute pipeline unavailable." : ctx.initError;
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal compute encoder.";
      return false;
    }
    [encoder setComputePipelineState:pipeline];
    for (NSUInteger i = 0; i < N; ++i) {
      if (buffers[i] != nil) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    MTLSize threadsPerGroup = MTLSizeMake(width, 1, 1);
    MTLSize threadsPerGrid = MTLSizeMake(threadCount, 1, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}
#endif

template <size_t N>
bool encodeComputeBuffersOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLComputePipelineState> pipeline,
    const std::array<id<MTLBuffer>, N>& buffers,
    NSUInteger threadCount,
    std::string* error) {
  if (commandBuffer == nil || pipeline == nil || threadCount == 0u) {
    if (error && error->empty()) {
      *error = "Metal compute encoder unavailable.";
    }
    return false;
  }
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (encoder == nil) {
    if (error) *error = "Failed to create Metal compute encoder.";
    return false;
  }
  [encoder setComputePipelineState:pipeline];
  for (NSUInteger i = 0; i < N; ++i) {
    if (buffers[i] != nil) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
  }
  NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup;
  if (width == 0) width = 64;
  width = std::min<NSUInteger>(width, 64);
  [encoder dispatchThreads:MTLSizeMake(threadCount, 1, 1)
     threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
  return true;
}

bool encodeBufferClearOnCommandBuffer(id<MTLCommandBuffer> commandBuffer,
                                      id<MTLBuffer> buffer,
                                      std::string* error) {
  if (commandBuffer == nil || buffer == nil || [buffer length] == 0u) {
    if (error) *error = "metal-buffer-clear-encode-unavailable";
    return false;
  }
  id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
  if (encoder == nil) {
    if (error) *error = "metal-buffer-clear-encoder-failed";
    return false;
  }
  [encoder fillBuffer:buffer range:NSMakeRange(0, [buffer length]) value:0];
  [encoder endEncoding];
  return true;
}

bool encodeBufferCopyOnCommandBuffer(id<MTLCommandBuffer> commandBuffer,
                                     id<MTLBuffer> src,
                                     id<MTLBuffer> dst,
                                     NSUInteger bytes,
                                     std::string* error) {
  if (commandBuffer == nil || src == nil || dst == nil || bytes == 0u ||
      [src length] < bytes || [dst length] < bytes) {
    if (error) *error = "metal-buffer-copy-encode-unavailable";
    return false;
  }
  id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
  if (encoder == nil) {
    if (error) *error = "metal-buffer-copy-encoder-failed";
    return false;
  }
  [encoder copyFromBuffer:src
             sourceOffset:0
                 toBuffer:dst
        destinationOffset:0
                     size:bytes];
  [encoder endEncoding];
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool copyBufferOnDevice(id<MTLBuffer> src, id<MTLBuffer> dst, NSUInteger bytes, std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || src == nil || dst == nil || bytes == 0u ||
      [src length] < bytes || [dst length] < bytes) {
    if (error) *error = "metal-buffer-copy-unavailable";
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-buffer-copy-command-buffer-failed";
      return false;
    }
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-buffer-copy-encoder-failed";
      return false;
    }
    [encoder copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:bytes];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}
#endif

template <typename T>
id<MTLBuffer> makeSharedBuffer(MetalContext& ctx,
                               const T* values,
                               size_t count) {
  if (!ctx.ready) return nil;
  const NSUInteger bytes = static_cast<NSUInteger>(count * sizeof(T));
  return [ctx.device newBufferWithBytes:(values != nullptr ? values : nullptr)
                                 length:bytes
                                options:MTLResourceStorageModeShared];
}

id<MTLBuffer> makeEmptySharedBuffer(MetalContext& ctx, NSUInteger bytes) {
  if (!ctx.ready) return nil;
  return [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

id<MTLBuffer> makeEmptyPrivateBuffer(MetalContext& ctx, NSUInteger bytes) {
  if (!ctx.ready) return nil;
  return [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
template <typename T>
id<MTLBuffer> makeSharedBuffer(const T* values, size_t count) {
  return makeSharedBuffer(context(), values, count);
}

id<MTLBuffer> makeEmptySharedBuffer(NSUInteger bytes) {
  return makeEmptySharedBuffer(context(), bytes);
}

id<MTLBuffer> makeEmptyPrivateBuffer(NSUInteger bytes) {
  return makeEmptyPrivateBuffer(context(), bytes);
}
#endif

id<MTLBuffer> makeSubmissionTransientPrivateBuffer(
    id<MTLCommandBuffer> commandBuffer,
    NSUInteger bytes,
    std::string* error) {
  if (commandBuffer == nil || bytes == 0u) {
    if (error) *error = "metal-transient-buffer-request-invalid";
    return nil;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               error) ||
      contextPointer == nullptr) {
    return nil;
  }
  MetalContext& ctx = *contextPointer;
  if (!ctx.ready || ctx.device == nil) {
    if (error) *error = "metal-transient-buffer-device-unavailable";
    return nil;
  }

  constexpr MTLResourceOptions kPrivateTrackedOptions =
      static_cast<MTLResourceOptions>(
          MTLResourceStorageModePrivate |
          MTLResourceHazardTrackingModeTracked);
  const MTLSizeAndAlign requirement =
      [ctx.device heapBufferSizeAndAlignWithLength:bytes
                                           options:kPrivateTrackedOptions];
  if (requirement.size == 0u || requirement.align == 0u) {
    if (error) *error = "metal-transient-buffer-heap-requirement-invalid";
    return nil;
  }

  std::lock_guard<std::mutex> submissionLock(frameSubmissionMutex());
  uint64_t submissionId = 0u;
  FrameSubmissionRecord* submissionRecord = nullptr;
  for (auto& item : frameSubmissionRegistry()) {
    if (item.second.commandBuffer == commandBuffer) {
      submissionId = item.first;
      submissionRecord = &item.second;
      break;
    }
  }
  if (!submissionRecord || !submissionRecord->transientArena ||
      submissionRecord->transientHeaps == nil) {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    return makeEmptyPrivateBuffer(ctx, bytes);
#else
    if (error) *error = "metal-transient-buffer-submission-unregistered";
    return nil;
#endif
  }

  auto recordLogicalBuffer = [&](id<MTLBuffer> buffer) -> id<MTLBuffer> {
    if (buffer == nil) return nil;
    std::lock_guard<std::mutex> arenaLock(
        submissionRecord->transientArena->mutex);
    const auto status = submissionRecord->transientArena->policy.recordBuffer(
        submissionId, static_cast<uint64_t>(bytes));
    if (!ChromaspaceMetalTransientArena::succeeded(status)) {
      if (error) {
        *error = std::string("metal-transient-buffer-accounting-failed:") +
                 ChromaspaceMetalTransientArena::statusLabel(status);
      }
      return nil;
    }
    return buffer;
  };

  for (id<MTLHeap> heap in submissionRecord->transientHeaps) {
    if (heap != nil &&
        [heap maxAvailableSizeWithAlignment:requirement.align] >=
            requirement.size) {
      id<MTLBuffer> buffer =
          [heap newBufferWithLength:bytes options:kPrivateTrackedOptions];
      if (buffer != nil) return recordLogicalBuffer(buffer);
    }
  }

  constexpr NSUInteger kMinimumTransientHeapPageBytes = 16u * 1024u * 1024u;
  const NSUInteger requestedPageBytes =
      std::max(kMinimumTransientHeapPageBytes, requirement.size);
  MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
  descriptor.size = requestedPageBytes;
  descriptor.storageMode = MTLStorageModePrivate;
  descriptor.cpuCacheMode = MTLCPUCacheModeDefaultCache;
  descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
  descriptor.type = MTLHeapTypeAutomatic;
  id<MTLHeap> heap = [ctx.device newHeapWithDescriptor:descriptor];
  if (heap == nil || heap.size == 0u) {
    if (error) *error = "metal-transient-heap-page-allocation-failed";
    return nil;
  }

  {
    std::lock_guard<std::mutex> arenaLock(
        submissionRecord->transientArena->mutex);
    const auto reserveStatus =
        submissionRecord->transientArena->policy.reservePage(
            submissionId, static_cast<uint64_t>(heap.size));
    if (!ChromaspaceMetalTransientArena::succeeded(reserveStatus)) {
      if (error) {
        *error = std::string("metal-transient-heap-budget-rejected:") +
                 ChromaspaceMetalTransientArena::statusLabel(reserveStatus);
      }
      return nil;
    }
  }
  [submissionRecord->transientHeaps addObject:heap];
  id<MTLBuffer> buffer =
      [heap newBufferWithLength:bytes options:kPrivateTrackedOptions];
  if (buffer == nil) {
    if (error) *error = "metal-transient-heap-buffer-allocation-failed";
    return nil;
  }
  return recordLogicalBuffer(buffer);
}

void clearSharedBuffer(id<MTLBuffer> buffer) {
  if (buffer == nil) return;
  std::memset([buffer contents], 0, static_cast<size_t>([buffer length]));
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool clearBufferOnDevice(id<MTLBuffer> buffer, std::string* error) {
  MetalContext& ctx = context();
  if (!ctx.ready || buffer == nil) {
    if (error) *error = "metal-buffer-clear-unavailable";
    return false;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-buffer-clear-command-buffer-failed";
      return false;
    }
    id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-buffer-clear-encoder-failed";
      return false;
    }
    [encoder fillBuffer:buffer range:NSMakeRange(0, [buffer length]) value:0];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }
  }
  return true;
}

template <typename T>
void copySharedBuffer(id<MTLBuffer> buffer, size_t count, std::vector<float>* out) {
  if (!out) return;
  out->clear();
  if (buffer == nil || count == 0) return;
  out->resize(count * sizeof(T) / sizeof(float));
  std::memcpy(out->data(), [buffer contents], count * sizeof(T));
}
#endif

}  // namespace

void fillRasterSourceUniforms(const RasterSourceRequest& request,
                              RasterSourceUniforms* uniforms);

bool importSharedSourceTexture(const SharedSourceImportRequest& request,
                               ImportedSourceTexture* outSource,
                               std::string* error) {
  if (outSource) *outSource = ImportedSourceTexture{};
  if (error) error->clear();
  if (!outSource || request.sharedTextureHandle == nullptr ||
      request.sharedEventHandle == nullptr) {
    if (error) *error = "invalid-shared-source-import-handles";
    return false;
  }
  constexpr size_t kMaximumDimension = 16384;
  constexpr size_t kMaximumSurfaceBytes =
      1024ull * 1024ull * 1024ull;
  if (request.senderId.empty() || request.senderId.size() > 256 ||
      request.deviceRegistryId == 0 || request.senderGeneration == 0 ||
      request.sequence == 0 || request.slotIndex >= 3 ||
      request.slotGeneration == 0 || request.readyValue == 0 ||
      request.contentHash == 0 || request.width <= 0 ||
      request.height <= 0 ||
      !ChromaspaceSourceExchange::validSourceSemanticMetadata(
          request.semantics) ||
      static_cast<size_t>(request.width) > kMaximumDimension ||
      static_cast<size_t>(request.height) > kMaximumDimension ||
      (request.pixelFormat != 0 && request.pixelFormat != 1)) {
    if (error) *error = "invalid-shared-source-import-metadata";
    return false;
  }
  const size_t bytesPerPixel = request.pixelFormat == 0 ? 8u : 16u;
  if (static_cast<size_t>(request.width) >
      std::numeric_limits<size_t>::max() / bytesPerPixel) {
    if (error) *error = "shared-source-row-byte-overflow";
    return false;
  }
  const size_t minimumRowBytes =
      static_cast<size_t>(request.width) * bytesPerPixel;
  if (request.bytesPerRow < minimumRowBytes ||
      request.bytesPerRow >
          std::numeric_limits<size_t>::max() /
              static_cast<size_t>(request.height) ||
      request.byteSize <
          request.bytesPerRow * static_cast<size_t>(request.height) ||
      request.byteSize > kMaximumSurfaceBytes) {
    if (error) *error = "invalid-shared-source-import-byte-bounds";
    return false;
  }

  @autoreleasepool {
    MTLSharedTextureHandle* textureHandle =
        (__bridge MTLSharedTextureHandle*)request.sharedTextureHandle;
    MTLSharedEventHandle* eventHandle =
        (__bridge MTLSharedEventHandle*)request.sharedEventHandle;
    if (textureHandle == nil || eventHandle == nil ||
        textureHandle.device == nil ||
        textureHandle.device.registryID != request.deviceRegistryId) {
      if (error) *error = "shared-source-handle-device-mismatch";
      return false;
    }
    id<MTLDevice> importDevice = textureHandle.device;
    id<MTLTexture> texture =
        [importDevice newSharedTextureWithHandle:textureHandle];
    id<MTLSharedEvent> readyEvent =
        [importDevice newSharedEventWithHandle:eventHandle];
    const MTLPixelFormat expectedFormat =
        sourceSignalMetalPixelFormat(request.pixelFormat);
    if (texture == nil || readyEvent == nil ||
        readyEvent.device.registryID != request.deviceRegistryId) {
      if (error) *error = "shared-source-handle-reconstruction-failed";
      return false;
    }
    if (texture.textureType != MTLTextureType2D ||
        texture.width != static_cast<NSUInteger>(request.width) ||
        texture.height != static_cast<NSUInteger>(request.height) ||
        texture.depth != 1 || texture.arrayLength != 1 ||
        texture.mipmapLevelCount != 1 || texture.sampleCount != 1 ||
        texture.pixelFormat != expectedFormat ||
        (texture.usage & MTLTextureUsageShaderRead) == 0) {
      if (error) *error = "shared-source-texture-metadata-mismatch";
      return false;
    }

    auto record = std::make_shared<ImportedSourceRecord>();
    record->texture = texture;
    record->readyEvent = readyEvent;
    record->descriptor.senderId = request.senderId;
    record->descriptor.deviceRegistryId = request.deviceRegistryId;
    record->descriptor.senderGeneration = request.senderGeneration;
    record->descriptor.sequence = request.sequence;
    record->descriptor.slotIndex = request.slotIndex;
    record->descriptor.slotGeneration = request.slotGeneration;
    record->descriptor.readyValue = request.readyValue;
    record->descriptor.contentHash = request.contentHash;
    record->descriptor.width = request.width;
    record->descriptor.height = request.height;
    record->descriptor.pixelFormat = request.pixelFormat;
    record->descriptor.bytesPerRow = request.bytesPerRow;
    record->descriptor.byteSize = request.byteSize;
    record->descriptor.semantics = request.semantics;

    {
      std::lock_guard<std::mutex> lock(importedSourceMutex());
      const uint64_t sourceId = allocateImportedSourceIdLocked();
      if (sourceId == 0) {
        if (error) *error = "imported-source-handle-space-exhausted";
        return false;
      }
      record->descriptor.sourceId = sourceId;
      importedSourceRegistry().emplace(sourceId, record);
    }
    *outSource = record->descriptor;
    return true;
  }
}

void releaseImportedSourceTexture(uint64_t sourceId) {
  (void)retireImportedSourceTexture(
      sourceId, nullptr, nullptr, nullptr);
}

bool retireImportedSourceTexture(
    uint64_t sourceId,
    ImportedSourceRetirementCallback callback,
    void* callbackContext,
    std::string* error) {
  if (error) error->clear();
  if (sourceId == 0) {
    if (error) *error = "invalid-imported-source-handle";
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> record;
  {
    std::lock_guard<std::mutex> lock(importedSourceMutex());
    auto it = importedSourceRegistry().find(sourceId);
    if (it == importedSourceRegistry().end() || !it->second) {
      if (error) *error = "imported-source-not-found";
      return false;
    }
    record = it->second;
    importedSourceRegistry().erase(it);
  }

  ImportedSourceRetirementCallback callbackNow = nullptr;
  void* callbackContextNow = nullptr;
  {
    std::lock_guard<std::mutex> lock(record->lifetimeMutex);
    if (record->retirementRequested) {
      if (error) *error = "imported-source-retirement-already-requested";
      return false;
    }
    record->retirementRequested = true;
    record->retirementCallback = callback;
    record->retirementContext = callbackContext;
    if (record->inFlightSubmissionUses == 0) {
      callbackNow = record->retirementCallback;
      callbackContextNow = record->retirementContext;
      record->retirementCallback = nullptr;
      record->retirementContext = nullptr;
    }
  }
  if (callbackNow != nullptr) callbackNow(callbackContextNow);
  return true;
}

GlossFieldCacheState glossFieldCacheState(const GlossFieldCache& cache) {
  if (cache.cacheId == 0 || cache.gridWidth <= 0 || cache.gridHeight <= 0 ||
      cache.builtSerial == 0 || cache.byteSize == 0u || !cache.available) {
    return GlossFieldCacheState::Missing;
  }
  switch (residentDerivedCacheState(glossDerivedCache(cache))) {
    case ResidentDerivedCacheState::Pending:
      return GlossFieldCacheState::Pending;
    case ResidentDerivedCacheState::Ready:
      return GlossFieldCacheState::Ready;
    case ResidentDerivedCacheState::Missing:
      return GlossFieldCacheState::Missing;
  }
  return GlossFieldCacheState::Missing;
}

bool activateWindow(void* nativeWindow) {
  if (nativeWindow == nullptr) return false;
  @autoreleasepool {
    NSWindow* window = (__bridge NSWindow*)nativeWindow;
    if (window == nil) return false;
    [NSApp activateIgnoringOtherApps:YES];
    [window orderFrontRegardless];
    [window makeKeyAndOrderFront:nil];
    return [NSApp isActive] && [window isKeyWindow];
  }
}

uint32_t currentModifierFlags() {
  @autoreleasepool {
    const NSEventModifierFlags flags = [NSEvent modifierFlags];
    uint32_t out = 0;
    if ((flags & NSEventModifierFlagShift) != 0) out |= ModifierFlagShift;
    if ((flags & NSEventModifierFlagControl) != 0) out |= ModifierFlagControl;
    if ((flags & NSEventModifierFlagOption) != 0) out |= ModifierFlagAlt;
    if ((flags & NSEventModifierFlagCommand) != 0) out |= ModifierFlagSuper;
    return out;
  }
}

ProbeResult probe() {
  ProbeResult result{};
  std::string error;
  MetalContext localContext{};
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
  MetalContext& ctx = context();
#else
  MetalContext& ctx = localContext;
#endif
  result.available = initializeMetalContext(&ctx, &error);
  result.queueReady = (ctx.queue != nil);
  static thread_local std::string deviceName;
  deviceName = ctx.deviceName;
  result.deviceName = deviceName.c_str();
  return result;
}

ResidentReadiness residentReadinessFromContext(const MetalContext& ctx,
                                               const std::string& error) {
  ResidentReadiness result{};
  result.contextReady = ctx.ready;
  result.deviceReady = ctx.device != nil;
  result.queueReady = ctx.queue != nil;
  result.deviceRegistryId =
      ctx.device != nil ? ctx.device.registryID : 0;
  result.deviceName = ctx.deviceName;
  auto markMissing = [&](const char* name, bool ready) {
    if (ready) return;
    if (!result.missing.empty()) result.missing += ",";
    result.missing += name ? name : "unknown";
  };
  if (!result.contextReady) {
    result.missing = error.empty() ? (ctx.initError.empty() ? "metal-context" : ctx.initError)
                                   : error;
    return result;
  }
  result.rasterSourceTextureReady =
      ctx.rasterSourceTexturePipeline != nil &&
      ctx.rasterOccupancyTextureCountPipeline != nil &&
      ctx.rasterOccupancyThresholdPipeline != nil;
  result.analyticalScopeReady =
      ctx.rasterScopeDensityTexturePipeline != nil &&
      ctx.rasterScopeRangeTexturePipeline != nil &&
      ctx.rasterScopeRangeHistogramTexturePipeline != nil &&
      ctx.scopeRangeHistogramPercentilePipeline != nil &&
      ctx.scopeRangeFinalizePipeline != nil;
  result.histogramSurfaceReady =
      result.analyticalScopeReady &&
      ctx.histogramApplyRangePipeline != nil &&
      ctx.histogramMaxPipeline != nil &&
      ctx.histogramSurfaceRenderPipeline != nil;
  result.waveformSurfaceReady =
      result.analyticalScopeReady &&
      ctx.waveformApplyRangePipeline != nil &&
      ctx.waveformMaxPipeline != nil &&
      ctx.waveformSurfaceRenderPipeline != nil;
  result.glossFieldCacheReady =
      ctx.rasterGlossFieldAccumulateTexturePipeline != nil &&
      ctx.glossFieldFinalizePipeline != nil &&
      ctx.glossFieldMaxPipeline != nil &&
      ctx.glossFieldNormalizePipeline != nil &&
      ctx.glossFieldBlurPipeline != nil &&
      ctx.glossFieldBodyPipeline != nil &&
      ctx.glossFieldRawSignalPipeline != nil &&
      ctx.glossFieldWeightedSignalPipeline != nil &&
      ctx.glossFieldMergeMaxBitsPipeline != nil &&
      ctx.glossFieldFinalNormalizePipeline != nil &&
      ctx.glossFieldLocalPercentilePipeline != nil &&
      ctx.glossFieldCandidate2RawPipeline != nil &&
      ctx.glossFieldAssembleUnifiedPipeline != nil;
  result.glossFieldSurfaceReady = result.glossFieldCacheReady &&
                                  ctx.glossFieldSurfaceRenderPipeline != nil &&
                                  ctx.plotSurfaceClearPipeline != nil;
  result.glossProjectionSurfaceReady = result.glossFieldCacheReady &&
                                       ctx.glossProjectionSurfaceSelectPipeline != nil &&
                                       ctx.glossProjectionSurfaceShadePipeline != nil &&
                                       ctx.plotSurfaceClearPipeline != nil;
  result.plotSurfaceReady = ctx.plotSurfaceClearPipeline != nil;
  result.plotSurfaceVectorReady = ctx.plotSurfaceVectorPipeline16 != nil;
  result.rasterPointSurfaceReady =
      result.rasterSourceTextureReady &&
      ctx.rasterPointSurfacePipeline16 != nil &&
      ctx.rasterPointCompactLocalScanPipeline != nil &&
      ctx.rasterPointScanBlockSumsPipeline != nil &&
      ctx.rasterPointAddBlockOffsetsPipeline != nil &&
      ctx.rasterPointCompactScatterPipeline != nil &&
      ctx.rasterPointFinalizeIndirectArgsPipeline != nil;
  result.sourceSignalSurfaceReady = ctx.sourceSignalSurfacePipeline != nil;
  result.frameSurfaceCompositeReady =
      ctx.frameSurfaceCompositePipeline != nil && ctx.frameSolidRectPipeline != nil;
  result.frameUiVectorReady = ctx.frameUiVectorPipeline != nil;
  result.frameTextReady = ctx.frameTextPipeline != nil;
  markMissing("raster-source-texture", result.rasterSourceTextureReady);
  markMissing("analytical-scope", result.analyticalScopeReady);
  markMissing("histogram-surface", result.histogramSurfaceReady);
  markMissing("waveform-surface", result.waveformSurfaceReady);
  markMissing("gloss-field-cache", result.glossFieldCacheReady);
  markMissing("gloss-field-surface", result.glossFieldSurfaceReady);
  markMissing("gloss-projection-surface", result.glossProjectionSurfaceReady);
  markMissing("plot-surface", result.plotSurfaceReady);
  markMissing("plot-surface-vector", result.plotSurfaceVectorReady);
  markMissing("raster-point-surface", result.rasterPointSurfaceReady);
  markMissing("source-signal-surface", result.sourceSignalSurfaceReady);
  markMissing("frame-surface-composite", result.frameSurfaceCompositeReady);
  markMissing("frame-ui-vector", result.frameUiVectorReady);
  markMissing("frame-text", result.frameTextReady);
  return result;
}

ResidentReadiness residentReadiness() {
  std::string error;
  MetalContext localContext{};
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
  MetalContext& ctx = context();
#else
  MetalContext& ctx = localContext;
#endif
  (void)initializeMetalContext(&ctx, &error);
  return residentReadinessFromContext(ctx, error);
}

ResidentReadiness residentReadiness(uint64_t compositorId) {
  std::shared_ptr<MetalContext> runtimeContext;
  std::string error;
  if (!contextForCompositor(compositorId, &runtimeContext, &error) ||
      !runtimeContext) {
    ResidentReadiness result{};
    result.missing = error.empty() ? "metal-runtime-context" : error;
    return result;
  }
  return residentReadinessFromContext(*runtimeContext, {});
}

namespace {

bool contextForCommandBuffer(id<MTLCommandBuffer> commandBuffer,
                             std::shared_ptr<MetalContext>* outOwnedContext,
                             MetalContext** outContext,
                             std::string* error) {
  if (outOwnedContext) outOwnedContext->reset();
  if (outContext) *outContext = nullptr;
  if (commandBuffer == nil) {
    if (error) *error = "metal-command-buffer-context-missing";
    return false;
  }
  {
    std::lock_guard<std::mutex> lock(frameSubmissionMutex());
    for (const auto& item : frameSubmissionRegistry()) {
      const FrameSubmissionRecord& record = item.second;
      if (record.commandBuffer != commandBuffer) continue;
      if (!record.context || !record.context->ready ||
          record.context->runtimeContextId == 0u ||
          record.context->device == nil || commandBuffer.device == nil ||
          record.context->device.registryID != commandBuffer.device.registryID) {
        if (error) *error = "metal-command-buffer-context-mismatch";
        return false;
      }
      if (outOwnedContext) *outOwnedContext = record.context;
      if (outContext) *outContext = record.context.get();
      return true;
    }
  }
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) {
      *error = localError.empty() ? "metal-context-unavailable" : localError;
    }
    return false;
  }
  MetalContext& compatibility = context();
  if (compatibility.device == nil || commandBuffer.device == nil ||
      compatibility.device.registryID != commandBuffer.device.registryID) {
    if (error) *error = "metal-command-buffer-compatibility-context-mismatch";
    return false;
  }
  if (outContext) *outContext = &compatibility;
  return true;
#else
  if (error) *error = "metal-command-buffer-not-owned-by-submission";
  return false;
#endif
}

}  // namespace

std::string residentPipelineUnavailableReason(const char* stage) {
  ResidentReadiness readiness = residentReadiness();
  std::string reason = stage && stage[0] != '\0' ? stage : "metal-resident-pipeline";
  reason += "-pipeline-unavailable";
  if (!readiness.contextReady) {
    reason += ":context";
  } else if (!readiness.missing.empty()) {
    reason += ":missing=" + readiness.missing;
  }
  return reason;
}

std::string residentPipelineUnavailableReason(const MetalContext& ctx,
                                              const char* stage) {
  std::string reason =
      stage && stage[0] != '\0' ? stage : "metal-resident-pipeline";
  reason += "-pipeline-unavailable";
  if (!ctx.ready) {
    reason += ":context";
  }
  if (!ctx.initError.empty()) {
    reason += ":" + ctx.initError;
  }
  return reason;
}

bool createFrameCompositor(void* nativeWindow,
                           int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           FrameCompositor* outCompositor,
                           std::string* error) {
  if (outCompositor) *outCompositor = FrameCompositor{};
  if (error) error->clear();
  if (nativeWindow == nullptr || drawableWidth <= 0 || drawableHeight <= 0) {
    if (error) *error = "invalid-metal-frame-compositor-request";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  try {
    runtimeContext = std::make_shared<MetalContext>();
  } catch (...) {
    if (error) *error = "metal-runtime-context-allocation-failed";
    return false;
  }
  std::string localError;
  if (!initializeMetalContext(runtimeContext.get(), &localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  MetalContext& ctx = *runtimeContext;
  @autoreleasepool {
    NSWindow* window = (__bridge NSWindow*)nativeWindow;
    if (window == nil) {
      if (error) *error = "invalid-native-window";
      return false;
    }
    NSView* contentView = [window contentView];
    if (contentView == nil) {
      if (error) *error = "native-window-missing-content-view";
      return false;
    }
    const float resolvedScale =
        contentsScale > 0.0f ? contentsScale : static_cast<float>([window backingScaleFactor]);
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = ctx.device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.opaque = YES;
    layer.contentsScale = resolvedScale;
    if (@available(macOS 10.13, *)) {
      layer.maximumDrawableCount = 3;
      layer.allowsNextDrawableTimeout = YES;
    }
    layer.drawableSize =
        CGSizeMake(static_cast<CGFloat>(drawableWidth), static_cast<CGFloat>(drawableHeight));
    layer.frame = [contentView bounds];
    layer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    CALayer* previousLayer = [contentView layer];
    const BOOL previousWantsLayer = [contentView wantsLayer];
    [contentView setWantsLayer:YES];
    [contentView setLayer:layer];

    FrameCompositorRecord record{};
    record.context = runtimeContext;
    record.window = window;
    record.contentView = contentView;
    record.previousLayer = previousLayer;
    record.layer = layer;
    record.frameSlots = dispatch_semaphore_create(3);
    record.completionGroup = dispatch_group_create();
    try {
      record.transientArena = std::make_shared<FrameTransientArenaState>();
    } catch (...) {
      record.transientArena.reset();
    }
    if (record.frameSlots == nullptr ||
        record.completionGroup == nullptr || !record.transientArena ||
        !record.transientArena->policy.configValid()) {
      [contentView setLayer:previousLayer];
      [contentView setWantsLayer:previousWantsLayer];
      if (error) *error = "metal-frame-compositor-runtime-allocation-failed";
      return false;
    }
    record.drawableWidth = drawableWidth;
    record.drawableHeight = drawableHeight;
    record.contentsScale = resolvedScale;
    record.previousWantsLayer = previousWantsLayer;
    const uint64_t compositorId = nextFrameCompositorId();
    uint64_t runtimeContextId = 0u;
    bool published = false;
    {
      std::lock_guard<std::mutex> lock(frameCompositorMutex());
      runtimeContextId = allocateRuntimeContextIdLocked();
      if (runtimeContextId != 0u) {
        runtimeContext->runtimeContextId = runtimeContextId;
        try {
          published = frameCompositorRegistry()
                          .emplace(compositorId, std::move(record))
                          .second;
        } catch (...) {
          published = false;
        }
      }
    }
    if (!published) {
      [contentView setLayer:previousLayer];
      [contentView setWantsLayer:previousWantsLayer];
      if (error) {
        *error = runtimeContextId == 0u
                     ? "metal-runtime-context-handle-space-exhausted"
                     : "metal-frame-compositor-registry-allocation-failed";
      }
      return false;
    }
    if (outCompositor) {
      outCompositor->compositorId = compositorId;
      outCompositor->runtimeContextId = runtimeContextId;
      outCompositor->deviceRegistryId = ctx.device.registryID;
      outCompositor->drawableWidth = drawableWidth;
      outCompositor->drawableHeight = drawableHeight;
      outCompositor->contentsScale = resolvedScale;
    }
    return true;
  }
}

bool drainFrameCompositor(uint64_t compositorId,
                          uint32_t timeoutMilliseconds,
                          std::string* error) {
  if (error) error->clear();
  if (compositorId == 0 || timeoutMilliseconds == 0) {
    if (error) *error = "invalid-metal-frame-compositor-drain";
    return false;
  }
  dispatch_group_t completionGroup = nullptr;
  {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() ||
        it->second.completionGroup == nullptr) {
      if (error) *error = "metal-frame-compositor-not-found";
      return false;
    }
    completionGroup = it->second.completionGroup;
  }
  const int64_t timeoutNanoseconds =
      static_cast<int64_t>(timeoutMilliseconds) *
      static_cast<int64_t>(NSEC_PER_MSEC);
  if (dispatch_group_wait(
          completionGroup,
          dispatch_time(DISPATCH_TIME_NOW, timeoutNanoseconds)) !=
      0) {
    if (error) *error = "metal-frame-compositor-drain-timeout";
    return false;
  }
  return true;
}

bool frameTransientMemoryStats(uint64_t compositorId,
                               FrameTransientMemoryStats* outStats,
                               std::string* error) {
  if (outStats) *outStats = FrameTransientMemoryStats{};
  if (error) error->clear();
  if (compositorId == 0u || outStats == nullptr) {
    if (error) *error = "invalid-metal-frame-transient-stats-request";
    return false;
  }

  std::shared_ptr<FrameTransientArenaState> transientArena;
  {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    const auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() ||
        !it->second.transientArena) {
      if (error) *error = "metal-frame-compositor-not-found";
      return false;
    }
    transientArena = it->second.transientArena;
  }

  ChromaspaceMetalTransientArena::ArenaSnapshot snapshot{};
  {
    std::lock_guard<std::mutex> arenaLock(transientArena->mutex);
    snapshot = transientArena->policy.snapshot();
  }
  if (snapshot.configStatus !=
      ChromaspaceMetalTransientArena::Status::Ok) {
    if (error) *error = "metal-frame-transient-arena-invalid";
    return false;
  }

  outStats->available = true;
  outStats->activeSubmissionCount = snapshot.activeCount;
  outStats->encodingCount = snapshot.encodingCount;
  outStats->submittedCount = snapshot.submittedCount;
  outStats->inFlightReservedBytes = snapshot.inFlightReservedBytes;
  outStats->inFlightLogicalBytes = snapshot.inFlightLogicalBytes;
  outStats->peakInFlightReservedBytes =
      snapshot.peakInFlightReservedBytes;
  outStats->peakInFlightLogicalBytes = snapshot.peakInFlightLogicalBytes;
  outStats->peakActiveSubmissionCount =
      snapshot.peakActiveSubmissionCount;
  outStats->maxInFlightBytes = snapshot.config.maxInFlightBytes;
  outStats->maxBytesPerSubmission =
      snapshot.config.maxBytesPerSubmission;
  outStats->maxSubmissions = snapshot.config.maxSubmissions;
  return true;
}

bool frameCompletionStats(uint64_t compositorId,
                          FrameCompletionStats* outStats,
                          std::string* error) {
  if (outStats) *outStats = FrameCompletionStats{};
  if (error) error->clear();
  if (compositorId == 0u || outStats == nullptr) {
    if (error) *error = "invalid-metal-frame-completion-stats-request";
    return false;
  }
  std::lock_guard<std::mutex> lock(frameCompositorMutex());
  const auto it = frameCompositorRegistry().find(compositorId);
  if (it == frameCompositorRegistry().end()) {
    if (error) *error = "metal-frame-compositor-not-found";
    return false;
  }
  const FrameCompositorRecord& record = it->second;
  outStats->available = true;
  outStats->submittedSerial = record.submittedSerial;
  outStats->completedSerial = record.completedSerial;
  outStats->failedSubmissionCount = record.failedSubmissionCount;
  outStats->timedSubmissionCount = record.timedSubmissionCount;
  outStats->untimedSubmissionCount = record.untimedSubmissionCount;
  outStats->accumulatedGpuSeconds = record.accumulatedGpuSeconds;
  outStats->maximumGpuSeconds = record.maximumGpuSeconds;
  outStats->lastSubmissionError = record.lastSubmissionError;
  return true;
}

bool resizeFrameCompositor(uint64_t compositorId,
                           int drawableWidth,
                           int drawableHeight,
                           float contentsScale,
                           std::string* error) {
  if (error) error->clear();
  if (compositorId == 0 || drawableWidth <= 0 || drawableHeight <= 0) {
    if (error) *error = "invalid-metal-frame-compositor-resize";
    return false;
  }
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() || it->second.layer == nil) {
      if (error) *error = "metal-frame-compositor-not-found";
      return false;
    }
    FrameCompositorRecord& record = it->second;
    const float resolvedScale =
        contentsScale > 0.0f
            ? contentsScale
            : (record.window != nil ? static_cast<float>([record.window backingScaleFactor]) : 1.0f);
    record.layer.contentsScale = resolvedScale;
    record.layer.drawableSize =
        CGSizeMake(static_cast<CGFloat>(drawableWidth), static_cast<CGFloat>(drawableHeight));
    if (record.contentView != nil) record.layer.frame = [record.contentView bounds];
    record.drawableWidth = drawableWidth;
    record.drawableHeight = drawableHeight;
    record.contentsScale = resolvedScale;
    return true;
  }
}

bool clearFrameCompositor(uint64_t compositorId,
                          float r,
                          float g,
                          float b,
                          float a,
                          std::string* error) {
  if (error) error->clear();
  if (compositorId == 0) {
    if (error) *error = "invalid-metal-frame-compositor-clear";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  if (!contextForCompositor(compositorId, &runtimeContext, error)) {
    return false;
  }
  CAMetalLayer* layer = nil;
  {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() || it->second.layer == nil) {
      if (error) *error = "metal-frame-compositor-not-found";
      return false;
    }
    layer = it->second.layer;
  }
  MetalContext& ctx = *runtimeContext;
  @autoreleasepool {
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (drawable == nil) {
      if (error) *error = "metal-frame-compositor-next-drawable-failed";
      return false;
    }
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "metal-frame-compositor-command-buffer-failed";
      return false;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor =
        MTLClearColorMake(static_cast<double>(r),
                          static_cast<double>(g),
                          static_cast<double>(b),
                          static_cast<double>(a));
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil) {
      if (error) *error = "metal-frame-compositor-render-encoder-failed";
      return false;
    }
    [encoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
    return true;
  }
}

bool beginFrameSubmission(uint64_t compositorId,
                          FrameSubmission* outSubmission,
                          std::string* error,
                          FrameFailureKind* failure) {
  if (outSubmission) *outSubmission = FrameSubmission{};
  if (error) error->clear();
  setFrameFailure(failure, FrameFailureKind::None);
  if (!outSubmission || compositorId == 0) {
    if (error) *error = "invalid-metal-frame-submission-request";
    setFrameFailure(failure, FrameFailureKind::InvalidState);
    return false;
  }
  dispatch_semaphore_t frameSlots = nullptr;
  std::shared_ptr<FrameTransientArenaState> transientArena;
  std::shared_ptr<MetalContext> runtimeContext;
  std::string previousSubmissionError;
  {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() ||
        it->second.layer == nil ||
        it->second.frameSlots == nullptr || !it->second.context ||
        !it->second.context->ready) {
      if (error) *error = "metal-frame-compositor-not-found";
      setFrameFailure(failure, FrameFailureKind::CompositorUnavailable);
      return false;
    }
    frameSlots = it->second.frameSlots;
    transientArena = it->second.transientArena;
    runtimeContext = it->second.context;
    if (!it->second.pendingSubmissionError.empty()) {
      previousSubmissionError = it->second.pendingSubmissionError;
      it->second.pendingSubmissionError.clear();
    }
  }
  if (!transientArena || !runtimeContext || runtimeContext->queue == nil ||
      runtimeContext->runtimeContextId == 0u ||
      runtimeContext->device == nil) {
    if (error) *error = "metal-frame-transient-arena-unavailable";
    setFrameFailure(failure, FrameFailureKind::InvariantViolation);
    return false;
  }
  if (!previousSubmissionError.empty()) {
    if (error) {
      *error = std::string("metal-frame-previous-submission-failed:") +
               previousSubmissionError;
    }
    setFrameFailure(failure, FrameFailureKind::PriorGpuSubmissionFailure);
    return false;
  }

  const long slotWait =
      dispatch_semaphore_wait(frameSlots,
                              dispatch_time(DISPATCH_TIME_NOW,
                                            kFrameSlotTimeoutNanoseconds));
  if (slotWait != 0) {
    if (error) *error = "metal-frame-compositor-backpressure-timeout";
    setFrameFailure(failure, FrameFailureKind::BackpressureTimeout);
    return false;
  }

  MetalContext& ctx = *runtimeContext;
  id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
  if (commandBuffer == nil) {
    dispatch_semaphore_signal(frameSlots);
    if (error) *error = "metal-frame-submission-command-buffer-failed";
    setFrameFailure(failure, FrameFailureKind::CommandBufferUnavailable);
    return false;
  }

  uint64_t submissionId = 0;
  std::string registrationError;
  FrameFailureKind registrationFailure = FrameFailureKind::InvariantViolation;
  {
    std::scoped_lock lock(frameCompositorMutex(), frameSubmissionMutex());
    auto compositorIt = frameCompositorRegistry().find(compositorId);
    if (compositorIt == frameCompositorRegistry().end() ||
        compositorIt->second.frameSlots != frameSlots ||
        compositorIt->second.context != runtimeContext) {
      registrationError = "metal-frame-compositor-released-before-submission";
      registrationFailure = FrameFailureKind::CompositorUnavailable;
    } else {
      submissionId = allocateFrameSubmissionIdLocked();
      if (submissionId == 0) {
        registrationError = "metal-frame-submission-handle-space-exhausted";
        registrationFailure = FrameFailureKind::InvariantViolation;
      } else {
        ChromaspaceMetalTransientArena::Status arenaStatus;
        {
          std::lock_guard<std::mutex> arenaLock(transientArena->mutex);
          arenaStatus = transientArena->policy.begin(submissionId);
        }
        if (!ChromaspaceMetalTransientArena::succeeded(arenaStatus)) {
          registrationError =
              std::string("metal-frame-transient-arena-begin-failed:") +
              ChromaspaceMetalTransientArena::statusLabel(arenaStatus);
          registrationFailure = FrameFailureKind::BackpressureTimeout;
        } else {
          FrameSubmissionRecord record{};
          record.context = runtimeContext;
          record.compositorId = compositorId;
          record.frameSlots = frameSlots;
          record.commandBuffer = commandBuffer;
          record.transientArena = transientArena;
          record.transientHeaps = [NSMutableArray array];
          if (record.transientHeaps == nil) {
            std::lock_guard<std::mutex> arenaLock(transientArena->mutex);
            transientArena->policy.abandon(submissionId);
            registrationError =
                "metal-frame-transient-heap-list-allocation-failed";
            registrationFailure = FrameFailureKind::CommandBufferUnavailable;
          } else {
            try {
              frameSubmissionRegistry().emplace(submissionId,
                                                std::move(record));
            } catch (...) {
              std::lock_guard<std::mutex> arenaLock(transientArena->mutex);
              transientArena->policy.abandon(submissionId);
              registrationError =
                  "metal-frame-submission-registry-allocation-failed";
              registrationFailure = FrameFailureKind::CommandBufferUnavailable;
            }
          }
        }
      }
    }
  }
  if (!registrationError.empty()) {
    dispatch_semaphore_signal(frameSlots);
    if (error) *error = registrationError;
    setFrameFailure(failure, registrationFailure);
    return false;
  }

  outSubmission->submissionId = submissionId;
  outSubmission->compositorId = compositorId;
  outSubmission->runtimeContextId = runtimeContext->runtimeContextId;
  outSubmission->deviceRegistryId = runtimeContext->device.registryID;
  return true;
}

void abandonFrameSubmission(FrameSubmission* submission) {
  if (!submission || submission->submissionId == 0) {
    if (submission) *submission = FrameSubmission{};
    return;
  }
  dispatch_semaphore_t frameSlots = nullptr;
  std::shared_ptr<FrameTransientArenaState> transientArena;
  uint64_t transientSubmissionId = 0u;
  std::vector<FrameSubmissionTransactionRecord> transactions;
  SubmissionRetention retainedResources;
  {
    std::lock_guard<std::mutex> lock(frameSubmissionMutex());
    auto it = frameSubmissionRegistry().find(submission->submissionId);
    if (it != frameSubmissionRegistry().end() &&
        submissionIdentityMatches(it->second, *submission)) {
      frameSlots = it->second.frameSlots;
      transientArena = it->second.transientArena;
      transientSubmissionId = submission->submissionId;
      transactions = std::move(it->second.transactions);
      retainedResources = std::move(it->second.retainedResources);
      frameSubmissionRegistry().erase(it);
    }
  }
  for (auto& transaction : transactions) {
    if (transaction.abandoned) transaction.abandoned();
  }
  retainedResources.reset();
  if (transientArena && transientSubmissionId != 0u) {
    std::lock_guard<std::mutex> arenaLock(transientArena->mutex);
    transientArena->policy.abandon(transientSubmissionId);
  }
  if (frameSlots != nullptr) dispatch_semaphore_signal(frameSlots);
  *submission = FrameSubmission{};
}

bool compositeFrameSurfaces(uint64_t compositorId,
                            const SurfaceCompositeItem* items,
                            size_t itemCount,
                            float clearR,
                            float clearG,
                            float clearB,
                            float clearA,
                            std::string* error) {
  return compositeFrameSurfacesAndOverlayRects(compositorId,
                                               items,
                                               itemCount,
                                               nullptr,
                                               0,
                                               clearR,
                                               clearG,
                                               clearB,
                                               clearA,
                                               error);
}

bool compositeFrameSurfacesAndOverlayRects(uint64_t compositorId,
                                           const SurfaceCompositeItem* items,
                                           size_t itemCount,
                                           const FrameOverlayRect* overlayRects,
                                           size_t overlayRectCount,
                                           float clearR,
                                           float clearG,
                                           float clearB,
                                           float clearA,
                                           std::string* error) {
  return compositeFrameSurfacesOverlayRectsAndText(compositorId,
                                                   items,
                                                   itemCount,
                                                   overlayRects,
                                                   overlayRectCount,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   0,
                                                   nullptr,
                                                   0,
                                                   clearR,
                                                   clearG,
                                                   clearB,
                                                   clearA,
                                                   error);
}

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
    FrameFailureKind* failure) {
  if (error) error->clear();
  setFrameFailure(failure, FrameFailureKind::None);
  if (!submission ||
      submission->submissionId == 0 ||
      submission->compositorId == 0 ||
      submission->runtimeContextId == 0 ||
      submission->deviceRegistryId == 0) {
    if (error) *error = "invalid-metal-frame-submission-submit";
    setFrameFailure(failure, FrameFailureKind::InvalidState);
    return false;
  }
  const uint64_t submissionId = submission->submissionId;
  const uint64_t compositorId = submission->compositorId;
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(*submission, &commandBuffer, error,
                                        failure)) {
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  if (!contextForFrameSubmission(*submission, &runtimeContext, error)) {
    setFrameFailure(failure, FrameFailureKind::MetalContextUnavailable);
    return false;
  }
  MetalContext& ctx = *runtimeContext;
  if (ctx.frameSurfaceCompositePipeline == nil || ctx.frameSolidRectPipeline == nil) {
    if (error) *error = "metal-frame-surface-composite-pipeline-unavailable";
    setFrameFailure(failure, FrameFailureKind::EncodingFailure);
    return false;
  }
  const bool hasVectors = vectorVertices != nullptr && vectorVertexCount > 0;
  if (hasVectors && ctx.frameUiVectorPipeline == nil) {
    if (error) *error = "metal-frame-ui-vector-pipeline-unavailable";
    setFrameFailure(failure, FrameFailureKind::EncodingFailure);
    return false;
  }
  const bool hasText = textVertices != nullptr && textVertexCount > 0 &&
                       textRuns != nullptr && textRunCount > 0;
  if (hasText && ctx.frameTextPipeline == nil) {
    if (error) *error = "metal-frame-text-pipeline-unavailable";
    setFrameFailure(failure, FrameFailureKind::EncodingFailure);
    return false;
  }
  CAMetalLayer* layer = nil;
  int drawableWidth = 0;
  int drawableHeight = 0;
  {
    std::lock_guard<std::mutex> lock(frameCompositorMutex());
    auto it = frameCompositorRegistry().find(compositorId);
    if (it == frameCompositorRegistry().end() || it->second.layer == nil ||
        it->second.context != runtimeContext) {
      if (error) *error = "metal-frame-compositor-not-found";
      setFrameFailure(failure, FrameFailureKind::CompositorUnavailable);
      return false;
    }
    layer = it->second.layer;
    drawableWidth = it->second.drawableWidth;
    drawableHeight = it->second.drawableHeight;
  }
  struct DrawableItem {
    id<MTLTexture> texture = nil;
    std::shared_ptr<PlotSurfaceRecord> record;
    uint32_t surfaceId = 0u;
    SurfaceCompositeUniforms uniforms{};
  };
  struct DrawableTextRun {
    id<MTLTexture> texture = nil;
    std::shared_ptr<FrameTextAtlasRecord> record;
    uint64_t atlasId = 0u;
    FrameTextUniforms uniforms{};
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
  };
  std::vector<FrameSolidRectUniforms> rectItems;
  rectItems.reserve(overlayRectCount);
  const size_t usableOverlayRectCount = overlayRects != nullptr ? overlayRectCount : 0;
  for (size_t i = 0; i < usableOverlayRectCount; ++i) {
    const FrameOverlayRect& rect = overlayRects[i];
    if (rect.w <= 0.0f || rect.h <= 0.0f || rect.a <= 0.0f) continue;
    FrameSolidRectUniforms uniforms{};
    uniforms.dstX = rect.x;
    uniforms.dstY = rect.y;
    uniforms.dstW = rect.w;
    uniforms.dstH = rect.h;
    uniforms.drawableW = static_cast<float>(std::max(drawableWidth, 1));
    uniforms.drawableH = static_cast<float>(std::max(drawableHeight, 1));
    uniforms.r = std::max(0.0f, std::min(rect.r, 1.0f));
    uniforms.g = std::max(0.0f, std::min(rect.g, 1.0f));
    uniforms.b = std::max(0.0f, std::min(rect.b, 1.0f));
    uniforms.a = std::max(0.0f, std::min(rect.a, 1.0f));
    rectItems.push_back(uniforms);
  }
  std::vector<DrawableItem> drawItems;
  drawItems.reserve(itemCount);
  std::string resourceError;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    for (size_t i = 0; i < itemCount; ++i) {
      const SurfaceCompositeItem& item = items[i];
      if (item.surfaceId == 0 || item.dstW <= 0.0f || item.dstH <= 0.0f || item.opacity <= 0.0f) continue;
      auto surfaceIt = plotSurfaceRegistry().find(item.surfaceId);
      if (surfaceIt == plotSurfaceRegistry().end() || !surfaceIt->second ||
          surfaceIt->second->texture == nil) {
        resourceError = "metal-frame-composite-plot-surface-not-found id=" +
                        std::to_string(item.surfaceId);
        break;
      }
      if (surfaceIt->second->ownerCompositorId != compositorId ||
          surfaceIt->second->context != runtimeContext) {
        resourceError = "metal-frame-composite-plot-surface-owner-mismatch id=" +
                        std::to_string(item.surfaceId);
        break;
      }
      DrawableItem drawable{};
      drawable.surfaceId = item.surfaceId;
      drawable.record = surfaceIt->second;
      drawable.texture = drawable.record->texture;
      drawable.uniforms.dstX = item.dstX;
      drawable.uniforms.dstY = item.dstY;
      drawable.uniforms.dstW = item.dstW;
      drawable.uniforms.dstH = item.dstH;
      drawable.uniforms.drawableW = static_cast<float>(std::max(drawableWidth, 1));
      drawable.uniforms.drawableH = static_cast<float>(std::max(drawableHeight, 1));
      drawable.uniforms.opacity = std::max(0.0f, std::min(item.opacity, 1.0f));
      drawItems.push_back(drawable);
    }
  }
  if (!resourceError.empty()) {
    if (error) *error = resourceError;
    setFrameFailure(failure, FrameFailureKind::InvariantViolation);
    return false;
  }
  for (const DrawableItem& item : drawItems) {
    const SubmissionRetentionKey key{
        SubmissionRetentionKind::PlotSurface,
        static_cast<std::uint64_t>(item.surfaceId),
        compositorId};
    bool added = false;
    if (!retainSubmissionResource(*submission, key,
                                  item.record,
                                  &resourceError,
                                  &added)) {
      if (error) *error = resourceError;
      setFrameFailure(failure, FrameFailureKind::InvariantViolation);
      return false;
    }
    bool stillLive = false;
    {
      std::lock_guard<std::mutex> lock(plotSurfaceMutex());
      const auto it = plotSurfaceRegistry().find(item.surfaceId);
      stillLive = it != plotSurfaceRegistry().end() && it->second == item.record &&
                  it->second->ownerCompositorId == compositorId;
    }
    if (!stillLive) {
      if (added) (void)releaseSubmissionResource(*submission, key);
      if (error) *error = "metal-frame-retention-retired-during-acquire";
      setFrameFailure(failure, FrameFailureKind::InvariantViolation);
      return false;
    }
  }
  std::vector<DrawableTextRun> textItems;
  textItems.reserve(hasText ? textRunCount : 0);
  if (hasText) {
    std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
    for (size_t i = 0; i < textRunCount; ++i) {
      const FrameTextRun& run = textRuns[i];
      if (run.atlasId == 0 || run.vertexCount == 0 || run.a <= 0.0f) continue;
      const size_t first = static_cast<size_t>(run.firstVertex);
      const size_t count = static_cast<size_t>(run.vertexCount);
      if (first >= textVertexCount || count > textVertexCount - first) continue;
      auto atlasIt = frameTextAtlasRegistry().find(run.atlasId);
      if (atlasIt == frameTextAtlasRegistry().end() || !atlasIt->second ||
          atlasIt->second->texture == nil) {
        resourceError = "metal-frame-text-atlas-not-found id=" +
                        std::to_string(run.atlasId);
        break;
      }
      if (atlasIt->second->ownerCompositorId != compositorId ||
          atlasIt->second->context != runtimeContext) {
        resourceError = "metal-frame-text-atlas-owner-mismatch id=" +
                        std::to_string(run.atlasId);
        break;
      }
      DrawableTextRun item{};
      item.atlasId = run.atlasId;
      item.record = atlasIt->second;
      item.texture = item.record->texture;
      item.firstVertex = run.firstVertex;
      item.vertexCount = run.vertexCount;
      item.uniforms.drawableW = static_cast<float>(std::max(drawableWidth, 1));
      item.uniforms.drawableH = static_cast<float>(std::max(drawableHeight, 1));
      item.uniforms.r = std::max(0.0f, std::min(run.r, 1.0f));
      item.uniforms.g = std::max(0.0f, std::min(run.g, 1.0f));
      item.uniforms.b = std::max(0.0f, std::min(run.b, 1.0f));
      item.uniforms.a = std::max(0.0f, std::min(run.a, 1.0f));
      item.uniforms.clipX = std::max(0.0f, run.clipX);
      item.uniforms.clipY = std::max(0.0f, run.clipY);
      item.uniforms.clipW = std::max(0.0f, run.clipW);
      item.uniforms.clipH = std::max(0.0f, run.clipH);
      item.uniforms.clipEnabled = run.clipEnabled != 0 ? 1.0f : 0.0f;
      textItems.push_back(item);
    }
  }
  if (!resourceError.empty()) {
    if (error) *error = resourceError;
    setFrameFailure(failure, FrameFailureKind::InvariantViolation);
    return false;
  }
  for (size_t index = 0u; index < textItems.size(); ++index) {
    const SubmissionRetentionKey key{
        SubmissionRetentionKind::TextAtlas, textItems[index].atlasId,
        compositorId};
    bool added = false;
    if (!retainSubmissionResource(*submission, key, textItems[index].record,
                                  &resourceError, &added)) {
      if (error) *error = resourceError;
      setFrameFailure(failure, FrameFailureKind::InvariantViolation);
      return false;
    }
    bool stillLive = false;
    {
      std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
      const auto it = frameTextAtlasRegistry().find(textItems[index].atlasId);
      stillLive = it != frameTextAtlasRegistry().end() &&
                  it->second == textItems[index].record &&
                  it->second->ownerCompositorId == compositorId;
    }
    if (!stillLive) {
      if (added) (void)releaseSubmissionResource(*submission, key);
      if (error) *error = "metal-frame-retention-retired-during-acquire";
      setFrameFailure(failure, FrameFailureKind::InvariantViolation);
      return false;
    }
  }
  @autoreleasepool {
    id<MTLBuffer> vectorVertexBuffer = nil;
    if (hasVectors) {
      const size_t vectorBytes = vectorVertexCount * sizeof(FrameVectorVertex);
      vectorVertexBuffer = [ctx.device newBufferWithBytes:vectorVertices
                                                    length:static_cast<NSUInteger>(vectorBytes)
                                                   options:MTLResourceStorageModeShared];
      if (vectorVertexBuffer == nil) {
        if (error) *error = "metal-frame-ui-vector-vertex-buffer-failed";
        setFrameFailure(failure, FrameFailureKind::CommandBufferUnavailable);
        return false;
      }
    }
    id<MTLBuffer> textVertexBuffer = nil;
    if (!textItems.empty()) {
      const size_t vertexBytes = textVertexCount * sizeof(FrameTextVertex);
      textVertexBuffer = [ctx.device newBufferWithBytes:textVertices
                                                  length:static_cast<NSUInteger>(vertexBytes)
                                                 options:MTLResourceStorageModeShared];
      if (textVertexBuffer == nil) {
        if (error) *error = "metal-frame-text-vertex-buffer-failed";
        setFrameFailure(failure, FrameFailureKind::CommandBufferUnavailable);
        return false;
      }
    }
    id<CAMetalDrawable> drawable = [layer nextDrawable];
    if (drawable == nil) {
      if (error) *error = "metal-frame-compositor-next-drawable-failed";
      setFrameFailure(failure, FrameFailureKind::DrawableUnavailable);
      return false;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor =
        MTLClearColorMake(static_cast<double>(clearR),
                          static_cast<double>(clearG),
                          static_cast<double>(clearB),
                          static_cast<double>(clearA));
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil) {
      if (error) *error = "metal-frame-compositor-render-encoder-failed";
      setFrameFailure(failure, FrameFailureKind::EncodingFailure);
      return false;
    }
    [encoder setRenderPipelineState:ctx.frameSurfaceCompositePipeline];
    for (const DrawableItem& item : drawItems) {
      SurfaceCompositeUniforms uniforms = item.uniforms;
      [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:0];
      [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
      [encoder setFragmentTexture:item.texture atIndex:0];
      [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    if (!rectItems.empty()) {
      [encoder setRenderPipelineState:ctx.frameSolidRectPipeline];
      for (const FrameSolidRectUniforms& uniforms : rectItems) {
        FrameSolidRectUniforms localUniforms = uniforms;
        [encoder setVertexBytes:&localUniforms length:sizeof(localUniforms) atIndex:0];
        [encoder setFragmentBytes:&localUniforms length:sizeof(localUniforms) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
      }
    }
    if (vectorVertexBuffer != nil && vectorVertexCount > 0) {
      FrameUiVectorUniforms uniforms{};
      uniforms.drawableW = static_cast<float>(std::max(drawableWidth, 1));
      uniforms.drawableH = static_cast<float>(std::max(drawableHeight, 1));
      [encoder setRenderPipelineState:ctx.frameUiVectorPipeline];
      [encoder setVertexBuffer:vectorVertexBuffer offset:0 atIndex:0];
      [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:1];
      [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                  vertexStart:0
                  vertexCount:static_cast<NSUInteger>(vectorVertexCount)];
    }
    if (!textItems.empty()) {
      [encoder setRenderPipelineState:ctx.frameTextPipeline];
      [encoder setVertexBuffer:textVertexBuffer offset:0 atIndex:0];
      for (const DrawableTextRun& item : textItems) {
        FrameTextUniforms uniforms = item.uniforms;
        [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:1];
        [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [encoder setFragmentTexture:item.texture atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:static_cast<NSUInteger>(item.firstVertex)
                    vertexCount:static_cast<NSUInteger>(item.vertexCount)];
      }
    }
    [encoder endEncoding];
    uint64_t submissionSerial = 0;
    dispatch_semaphore_t completionSlots = nullptr;
    dispatch_group_t completionGroup = nullptr;
    std::shared_ptr<FrameTransientArenaState> completionTransientArena;
    NSArray<id<MTLHeap>>* completionTransientHeaps = nil;
    std::shared_ptr<std::vector<FrameSubmissionTransactionRecord>>
        completionTransactions;
    std::shared_ptr<SubmissionRetention> completionRetention;
    try {
      completionTransactions =
          std::make_shared<
              std::vector<FrameSubmissionTransactionRecord>>();
      completionRetention = std::make_shared<SubmissionRetention>();
    } catch (...) {
      if (error) {
        *error =
            "metal-frame-submission-completion-allocation-failed";
      }
      setFrameFailure(failure, FrameFailureKind::CommandBufferUnavailable);
      return false;
    }
    std::string consumeError;
    FrameFailureKind consumeFailure = FrameFailureKind::InvariantViolation;
    {
      std::scoped_lock lock(frameCompositorMutex(), frameSubmissionMutex());
      auto submissionIt =
          frameSubmissionRegistry().find(submissionId);
      if (submissionIt == frameSubmissionRegistry().end()) {
        consumeError = "metal-frame-submission-not-found";
        consumeFailure = FrameFailureKind::InvariantViolation;
      } else if (!submissionIdentityMatches(submissionIt->second,
                                            *submission)) {
        consumeError = "metal-frame-submission-context-mismatch";
        consumeFailure = FrameFailureKind::InvariantViolation;
      } else if (submissionIt->second.commandBuffer != commandBuffer) {
        consumeError = "metal-frame-submission-command-buffer-mismatch";
        consumeFailure = FrameFailureKind::InvariantViolation;
      } else {
        auto compositorIt = frameCompositorRegistry().find(compositorId);
        if (compositorIt == frameCompositorRegistry().end() ||
            compositorIt->second.frameSlots != submissionIt->second.frameSlots ||
            compositorIt->second.context != runtimeContext) {
          consumeError = "metal-frame-compositor-released-before-submit";
          consumeFailure = FrameFailureKind::CompositorUnavailable;
        } else {
          completionTransientArena = submissionIt->second.transientArena;
          ChromaspaceMetalTransientArena::Status transientSubmitStatus =
              ChromaspaceMetalTransientArena::Status::InvalidConfig;
          if (completionTransientArena) {
            std::lock_guard<std::mutex> arenaLock(
                completionTransientArena->mutex);
            transientSubmitStatus =
                completionTransientArena->policy.submit(submissionId);
          }
          if (!ChromaspaceMetalTransientArena::succeeded(
                  transientSubmitStatus)) {
            consumeError =
                std::string("metal-frame-transient-arena-submit-failed:") +
                ChromaspaceMetalTransientArena::statusLabel(
                    transientSubmitStatus);
            consumeFailure = FrameFailureKind::InvariantViolation;
          } else {
            completionTransientHeaps =
                [submissionIt->second.transientHeaps copy];
            completionSlots = submissionIt->second.frameSlots;
            completionGroup = compositorIt->second.completionGroup;
            submissionSerial = ++compositorIt->second.submittedSerial;
            *completionTransactions =
                std::move(submissionIt->second.transactions);
            submissionIt->second.retainedResources.seal();
            *completionRetention =
                std::move(submissionIt->second.retainedResources);
            // Publish the accepted committed-frame obligation while both
            // registries are locked. A control-plane drain cannot observe an
            // empty group after this submission has left the registry.
            dispatch_group_enter(completionGroup);
            frameSubmissionRegistry().erase(submissionIt);
          }
        }
      }
    }
    if (!consumeError.empty()) {
      if (error) *error = consumeError;
      setFrameFailure(failure, consumeFailure);
      return false;
    }
    try {
      for (auto& transaction : *completionTransactions) {
        if (transaction.submitted) transaction.submitted();
      }
    } catch (...) {
      for (auto& transaction : *completionTransactions) {
        if (transaction.abandoned) transaction.abandoned();
      }
      if (completionTransientArena) {
        std::lock_guard<std::mutex> arenaLock(
            completionTransientArena->mutex);
        completionTransientArena->policy.abandon(submissionId);
      }
      dispatch_semaphore_signal(completionSlots);
      dispatch_group_leave(completionGroup);
      if (error) {
        *error =
            "metal-frame-submission-transaction-submit-failed";
      }
      setFrameFailure(failure, FrameFailureKind::EncodingFailure);
      return false;
    }
    const std::shared_ptr<MetalContext> completionContext = runtimeContext;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completedBuffer) {
      (void)completionContext;
      (void)completionTransientHeaps;
      std::string completionError;
      NSError* commandError = completedBuffer.error;
      const double gpuStartTime = completedBuffer.GPUStartTime;
      const double gpuEndTime = completedBuffer.GPUEndTime;
      const bool gpuTimingAvailable =
          std::isfinite(gpuStartTime) && std::isfinite(gpuEndTime) &&
          gpuStartTime > 0.0 && gpuEndTime >= gpuStartTime;
      const double gpuSeconds =
          gpuTimingAvailable ? gpuEndTime - gpuStartTime : 0.0;
      if (commandError != nil) {
        NSString* description = [commandError localizedDescription];
        const char* descriptionUtf8 =
            description != nil ? [description UTF8String] : nullptr;
        completionError = descriptionUtf8 != nullptr
                              ? std::string(descriptionUtf8)
                              : std::string("unknown-metal-command-buffer-error");
      }
      if (completionTransactions) {
        const bool success = completionError.empty();
        for (auto& transaction : *completionTransactions) {
          if (transaction.completed) transaction.completed(success);
        }
      }
      if (completionTransientArena) {
        std::lock_guard<std::mutex> arenaLock(
            completionTransientArena->mutex);
        completionTransientArena->policy.complete(submissionId);
      }
      {
        std::lock_guard<std::mutex> lock(frameCompositorMutex());
        auto it = frameCompositorRegistry().find(compositorId);
        if (it != frameCompositorRegistry().end() &&
            it->second.frameSlots == completionSlots) {
          it->second.completedSerial =
              std::max(it->second.completedSerial, submissionSerial);
          if (gpuTimingAvailable && std::isfinite(gpuSeconds)) {
            ++it->second.timedSubmissionCount;
            const double nextAccumulated =
                it->second.accumulatedGpuSeconds + gpuSeconds;
            if (std::isfinite(nextAccumulated)) {
              it->second.accumulatedGpuSeconds = nextAccumulated;
            }
            it->second.maximumGpuSeconds =
                std::max(it->second.maximumGpuSeconds, gpuSeconds);
          } else {
            ++it->second.untimedSubmissionCount;
          }
          if (!completionError.empty() &&
              it->second.pendingSubmissionError.empty()) {
            it->second.pendingSubmissionError =
                std::string("serial=") + std::to_string(submissionSerial) +
                ":" + completionError;
          }
          if (!completionError.empty()) {
            ++it->second.failedSubmissionCount;
            it->second.lastSubmissionError =
                std::string("serial=") + std::to_string(submissionSerial) +
                ":" + completionError;
          }
        }
      }
      dispatch_semaphore_signal(completionSlots);
      dispatch_group_leave(completionGroup);
      // Keep completionRetention alive through all callback/statistics work;
      // releasing this final shared owner retires every native resource hold.
      if (completionRetention) completionRetention->reset();
    }];
    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];
    *submission = FrameSubmission{};
    return true;
  }
}

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
                                               std::string* error) {
  FrameSubmission submission{};
  if (!beginFrameSubmission(compositorId, &submission, error, nullptr)) {
    return false;
  }
  if (!submitFrameSubmissionSurfacesOverlayRectsAndText(
          &submission,
          items,
          itemCount,
          overlayRects,
          overlayRectCount,
          vectorVertices,
          vectorVertexCount,
          textVertices,
          textVertexCount,
          textRuns,
          textRunCount,
          clearR,
          clearG,
          clearB,
          clearA,
          error,
          nullptr)) {
    abandonFrameSubmission(&submission);
    return false;
  }
  return true;
}

void releaseFrameCompositor(uint64_t compositorId) {
  if (compositorId == 0) return;
  @autoreleasepool {
    NSView* contentView = nil;
    CALayer* layer = nil;
    CALayer* previousLayer = nil;
    dispatch_semaphore_t retainedFrameSlots = nullptr;
    BOOL previousWantsLayer = NO;
    std::vector<dispatch_semaphore_t> abandonedSlots;
    std::vector<FrameSubmissionTransactionRecord> abandonedTransactions;
    std::vector<std::pair<std::shared_ptr<FrameTransientArenaState>, uint64_t>>
        abandonedTransientSubmissions;
    {
      std::scoped_lock lock(frameCompositorMutex(), frameSubmissionMutex());
      auto compositorIt = frameCompositorRegistry().find(compositorId);
      if (compositorIt == frameCompositorRegistry().end()) return;
      contentView = compositorIt->second.contentView;
      layer = compositorIt->second.layer;
      previousLayer = compositorIt->second.previousLayer;
      retainedFrameSlots = compositorIt->second.frameSlots;
      previousWantsLayer = compositorIt->second.previousWantsLayer;
      for (auto it = frameSubmissionRegistry().begin();
           it != frameSubmissionRegistry().end();) {
        if (it->second.compositorId == compositorId) {
          if (it->second.frameSlots != nullptr) {
            abandonedSlots.push_back(it->second.frameSlots);
          }
          for (auto& transaction : it->second.transactions) {
            abandonedTransactions.push_back(std::move(transaction));
          }
          if (it->second.transientArena) {
            abandonedTransientSubmissions.emplace_back(
                it->second.transientArena, it->first);
          }
          it = frameSubmissionRegistry().erase(it);
        } else {
          ++it;
        }
      }
      frameCompositorRegistry().erase(compositorIt);
    }
    {
      std::lock_guard<std::mutex> lock(plotSurfaceMutex());
      auto& registry = plotSurfaceRegistry();
      for (auto it = registry.begin(); it != registry.end();) {
        if (it->second && it->second->ownerCompositorId == compositorId) {
          it = registry.erase(it);
        } else {
          ++it;
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
      auto& registry = frameTextAtlasRegistry();
      for (auto it = registry.begin(); it != registry.end();) {
        if (it->second && it->second->ownerCompositorId == compositorId) {
          it = registry.erase(it);
        } else {
          ++it;
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
      auto& registry = scopeDerivedRegistry();
      for (auto it = registry.begin(); it != registry.end();) {
        if (it->second.ownerCompositorId == compositorId) {
          it = registry.erase(it);
        } else {
          ++it;
        }
      }
    }
    for (auto& transaction : abandonedTransactions) {
      if (transaction.abandoned) transaction.abandoned();
    }
    for (const auto& transient : abandonedTransientSubmissions) {
      if (!transient.first || transient.second == 0u) continue;
      std::lock_guard<std::mutex> arenaLock(transient.first->mutex);
      transient.first->policy.abandon(transient.second);
    }
    if (contentView != nil && [contentView layer] == layer) {
      [contentView setLayer:previousLayer];
      [contentView setWantsLayer:previousWantsLayer];
    }
    for (dispatch_semaphore_t frameSlots : abandonedSlots) {
      dispatch_semaphore_signal(frameSlots);
    }
    (void)retainedFrameSlots;
  }
}

bool createFrameTextAtlas(uint64_t compositorId,
                          int width,
                          int height,
                          const unsigned char* alphaPixels,
                          FrameTextAtlas* outAtlas,
                          std::string* error) {
  if (outAtlas) *outAtlas = FrameTextAtlas{};
  if (error) error->clear();
  if (!outAtlas) {
    if (error) *error = "missing-frame-text-atlas-output";
    return false;
  }
  if (compositorId == 0 || width <= 0 || height <= 0 ||
      alphaPixels == nullptr) {
    if (error) *error = "invalid-frame-text-atlas-request";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  if (!contextForCompositor(compositorId, &runtimeContext, error)) {
    return false;
  }
  MetalContext& ctx = *runtimeContext;
  @autoreleasepool {
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                           width:static_cast<NSUInteger>(width)
                                                          height:static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture = [ctx.device newTextureWithDescriptor:desc];
    if (texture == nil) {
      if (error) *error = "frame-text-atlas-texture-allocation-failed";
      return false;
    }
    MTLRegion region = MTLRegionMake2D(0, 0,
                                       static_cast<NSUInteger>(width),
                                       static_cast<NSUInteger>(height));
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:alphaPixels
               bytesPerRow:static_cast<NSUInteger>(width)];
    std::shared_ptr<FrameTextAtlasRecord> record;
    try {
      record = std::make_shared<FrameTextAtlasRecord>();
    } catch (...) {
      if (error) *error = "frame-text-atlas-record-allocation-failed";
      return false;
    }
    record->texture = texture;
    record->context = runtimeContext;
    record->ownerCompositorId = compositorId;
    record->width = width;
    record->height = height;
    uint64_t atlasId = 0;
    {
      std::scoped_lock lock(frameCompositorMutex(), frameTextAtlasMutex());
      auto compositorIt = frameCompositorRegistry().find(compositorId);
      if (compositorIt == frameCompositorRegistry().end() ||
          compositorIt->second.layer == nil ||
          compositorIt->second.context != runtimeContext) {
        if (error) *error = "frame-text-atlas-compositor-not-found";
        return false;
      }
      atlasId = allocateFrameTextAtlasIdLocked();
      if (atlasId == 0) {
        if (error) *error = "frame-text-atlas-handle-space-exhausted";
        return false;
      }
      try {
        frameTextAtlasRegistry().emplace(atlasId, record);
      } catch (...) {
        if (error) *error = "frame-text-atlas-registry-allocation-failed";
        return false;
      }
    }
    outAtlas->atlasId = atlasId;
    outAtlas->width = width;
    outAtlas->height = height;
    return true;
  }
}

void releaseFrameTextAtlas(uint64_t compositorId, uint64_t atlasId) {
  if (compositorId == 0 || atlasId == 0) return;
  std::lock_guard<std::mutex> lock(frameTextAtlasMutex());
  auto atlasIt = frameTextAtlasRegistry().find(atlasId);
  if (atlasIt == frameTextAtlasRegistry().end() ||
      !atlasIt->second || atlasIt->second->ownerCompositorId != compositorId) {
    return;
  }
  frameTextAtlasRegistry().erase(atlasIt);
}

static bool createPlotSurfaceInternal(uint64_t ownerCompositorId,
                                      int width,
                                      int height,
                                      int pixelFormat,
                                      bool iosurfaceBacked,
                                      PlotSurface* outSurface,
                                      std::string* error) {
  if (outSurface) *outSurface = PlotSurface{};
  if (error) error->clear();
  if (!outSurface) {
    if (error) *error = "missing-plot-surface-output";
    return false;
  }
  if (width <= 0 || height <= 0) {
    if (error) *error = "invalid-plot-surface-size";
    return false;
  }
  if (!iosurfaceBacked && ownerCompositorId == 0) {
    if (error) *error = "private-plot-surface-owner-required";
    return false;
  }
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
  if (iosurfaceBacked) {
    if (error) *error = "iosurface-plot-surfaces-unavailable-in-native-only-mode";
    return false;
  }
#else
  if (iosurfaceBacked && ownerCompositorId != 0) {
    if (error) *error = "compatibility-plot-surface-owner-must-be-zero";
    return false;
  }
#endif
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (ownerCompositorId != 0u) {
    if (!contextForCompositor(ownerCompositorId, &runtimeContext, error)) {
      return false;
    }
    contextPointer = runtimeContext.get();
  } else {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    std::string localError;
    if (!ensureContext(&localError)) {
      if (error) {
        *error = localError.empty() ? "metal-context-unavailable" : localError;
      }
      return false;
    }
    contextPointer = &context();
#else
    if (error) *error = "private-plot-surface-owner-required";
    return false;
#endif
  }
  MetalContext& ctx = *contextPointer;
  @autoreleasepool {
    const size_t bytesPerElement = sourceSignalBytesPerElement(pixelFormat);
    const size_t bytesPerRow = static_cast<size_t>(width) * bytesPerElement;
    const size_t byteSize = bytesPerRow * static_cast<size_t>(height);
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    IOSurfaceRef surface = nullptr;
    if (iosurfaceBacked) {
      NSDictionary* surfaceProperties = @{
        (__bridge NSString*)kIOSurfaceWidth: @(width),
        (__bridge NSString*)kIOSurfaceHeight: @(height),
        (__bridge NSString*)kIOSurfaceBytesPerElement: @(bytesPerElement),
        (__bridge NSString*)kIOSurfaceBytesPerRow: @(bytesPerRow),
        (__bridge NSString*)kIOSurfaceAllocSize: @(byteSize),
        (__bridge NSString*)kIOSurfacePixelFormat: @(sourceSignalIOSurfacePixelFormat(pixelFormat)),
      };
      surface = IOSurfaceCreate((__bridge CFDictionaryRef)surfaceProperties);
      if (surface == nullptr) {
        if (error) *error = "plot-surface-iosurface-allocation-failed";
        return false;
      }
    }
#endif
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sourceSignalMetalPixelFormat(pixelFormat)
                                                           width:static_cast<NSUInteger>(width)
                                                          height:static_cast<NSUInteger>(height)
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> texture = [ctx.device newTextureWithDescriptor:desc];
#else
    desc.storageMode = iosurfaceBacked ? MTLStorageModeShared : MTLStorageModePrivate;
    id<MTLTexture> texture =
        iosurfaceBacked
            ? [ctx.device newTextureWithDescriptor:desc iosurface:surface plane:0]
            : [ctx.device newTextureWithDescriptor:desc];
#endif
    if (texture == nil) {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
      if (surface) CFRelease(surface);
#endif
      if (error) {
        *error = iosurfaceBacked ? "plot-surface-metal-iosurface-texture-failed"
                                 : "plot-surface-private-metal-texture-failed";
      }
      return false;
    }
    std::shared_ptr<PlotSurfaceRecord> record;
    try {
      record = std::make_shared<PlotSurfaceRecord>();
    } catch (...) {
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
      if (surface) CFRelease(surface);
#endif
      if (error) *error = "plot-surface-record-allocation-failed";
      return false;
    }
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    record->surface = surface;
    surface = nullptr;
#endif
    record->texture = texture;
    record->context = runtimeContext;
    record->ownerCompositorId = ownerCompositorId;
    record->width = width;
    record->height = height;
    record->pixelFormat = pixelFormat;
    record->byteSize = byteSize;
    uint32_t surfaceId = 0;
    if (ownerCompositorId == 0) {
      std::lock_guard<std::mutex> lock(plotSurfaceMutex());
      surfaceId = allocatePlotSurfaceHandleLocked();
      if (surfaceId != 0) {
        try {
          plotSurfaceRegistry().emplace(surfaceId, record);
        } catch (...) {
          surfaceId = 0;
        }
      }
    } else {
      std::scoped_lock lock(frameCompositorMutex(), plotSurfaceMutex());
      auto compositorIt = frameCompositorRegistry().find(ownerCompositorId);
      if (compositorIt == frameCompositorRegistry().end() ||
          compositorIt->second.layer == nil ||
          compositorIt->second.context != runtimeContext) {
        if (error) *error = "private-plot-surface-compositor-not-found";
        return false;
      }
      surfaceId = allocatePlotSurfaceHandleLocked();
      if (surfaceId != 0) {
        try {
          plotSurfaceRegistry().emplace(surfaceId, record);
        } catch (...) {
          surfaceId = 0;
        }
      }
    }
    if (surfaceId == 0) {
      if (error) *error = "plot-surface-handle-space-exhausted";
      return false;
    }
    if (outSurface) {
      outSurface->surfaceId = surfaceId;
      outSurface->width = width;
      outSurface->height = height;
      outSurface->pixelFormat = pixelFormat;
      outSurface->byteSize = byteSize;
    }
    return true;
  }
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool createPlotSurface(int width,
                       int height,
                       int pixelFormat,
                       PlotSurface* outSurface,
                       std::string* error) {
  return createPlotSurfaceInternal(0, width, height, pixelFormat, true, outSurface, error);
}
#endif

bool createPrivatePlotSurface(uint64_t compositorId,
                              int width,
                              int height,
                              int pixelFormat,
                              PlotSurface* outSurface,
                              std::string* error) {
  return createPlotSurfaceInternal(
      compositorId, width, height, pixelFormat, false, outSurface, error);
}

static bool encodePlotSurfaceClearOnCommandBuffer(id<MTLCommandBuffer> commandBuffer,
                                                  uint32_t surfaceId,
                                                  int width,
                                                  int height,
                                                  int pixelFormat,
                                                  float r,
                                                  float g,
                                                  float b,
                                                  float a,
                                                  std::string* error) {
  if (error) error->clear();
  if (commandBuffer == nil) {
    if (error) *error = "plot-surface-clear-command-buffer-unavailable";
    return false;
  }
  if (surfaceId == 0 || width <= 0 || height <= 0) {
    if (error) *error = "invalid-plot-surface-clear-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer, &runtimeContext,
                               &contextPointer, error) ||
      contextPointer == nullptr) {
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.plotSurfaceClearPipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "plot-surface-clear");
    }
    return false;
  }
  id<MTLTexture> texture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(surfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == width &&
        it->second->height == height &&
        it->second->pixelFormat == pixelFormat &&
        it->second->context == runtimeContext) {
      texture = it->second->texture;
    }
  }
  if (texture == nil) {
    if (error) *error = "plot-surface-clear-output-surface-missing";
    return false;
  }
  @autoreleasepool {
    PlotSurfaceClearUniforms uniforms{r, g, b, a};
    id<MTLBuffer> uniformBuffer =
        [ctx.device newBufferWithBytes:&uniforms
                                length:sizeof(uniforms)
                               options:MTLResourceStorageModeShared];
    if (uniformBuffer == nil) {
      if (error) *error = "plot-surface-clear-uniform-allocation-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "plot-surface-clear-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.plotSurfaceClearPipeline];
    [encoder setTexture:texture atIndex:0];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
    NSUInteger groupWidth = ctx.plotSurfaceClearPipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.plotSurfaceClearPipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    MTLSize threadsPerGroup = MTLSizeMake(groupWidth, groupHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(width),
                                         static_cast<NSUInteger>(height),
                                         1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    return true;
  }
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool clearPlotSurface(uint32_t surfaceId,
                      int width,
                      int height,
                      int pixelFormat,
                      float r,
                      float g,
                      float b,
                      float a,
                      std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "plot-surface-clear-command-buffer-failed";
    return false;
  }
  if (!encodePlotSurfaceClearOnCommandBuffer(
          commandBuffer, surfaceId, width, height, pixelFormat, r, g, b, a, error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}
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
                            std::string* error) {
  if (!retainPlotSurfaceForSubmission(submission, surfaceId, nullptr, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(submission, surfaceId, error)) {
    return false;
  }
  return encodePlotSurfaceClearOnCommandBuffer(
      commandBuffer, surfaceId, width, height, pixelFormat, r, g, b, a, error);
}

static bool encodePlotSurfaceVectorPrimitivesOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
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
    std::string* error) {
  if (error) error->clear();
  if (commandBuffer == nil) {
    if (error) *error = "plot-surface-vector-command-buffer-unavailable";
    return false;
  }
  if (surfaceId == 0 || width <= 0 || height <= 0) {
    if (error) *error = "invalid-plot-surface-vector-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-vector-format";
    return false;
  }
  if (vertexCount > 0 && vertices == nullptr) {
    if (error) *error = "missing-plot-surface-vector-vertices";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer, &runtimeContext,
                               &contextPointer, error) ||
      contextPointer == nullptr) {
    return false;
  }
  MetalContext& ctx = *contextPointer;
  id<MTLRenderPipelineState> pipeline =
      pixelFormat == 1 ? ctx.plotSurfaceVectorPipeline32 : ctx.plotSurfaceVectorPipeline16;
  if (pipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "plot-surface-vector");
    }
    return false;
  }
  id<MTLTexture> texture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(surfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == width &&
        it->second->height == height &&
        it->second->pixelFormat == pixelFormat &&
        it->second->context == runtimeContext) {
      texture = it->second->texture;
    }
  }
  if (texture == nil) {
    if (error) *error = "plot-surface-vector-output-surface-missing";
    return false;
  }
  @autoreleasepool {
    id<MTLBuffer> vertexBuffer = nil;
    if (vertexCount > 0) {
      const size_t vertexBytes = vertexCount * sizeof(FrameVectorVertex);
      vertexBuffer = [ctx.device newBufferWithBytes:vertices
                                             length:static_cast<NSUInteger>(vertexBytes)
                                            options:MTLResourceStorageModeShared];
      if (vertexBuffer == nil) {
        if (error) *error = "plot-surface-vector-vertex-buffer-failed";
        return false;
      }
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = texture;
    pass.colorAttachments[0].loadAction = clearBeforeDraw ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor =
        MTLClearColorMake(static_cast<double>(clearR),
                          static_cast<double>(clearG),
                          static_cast<double>(clearB),
                          static_cast<double>(clearA));
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil) {
      if (error) *error = "plot-surface-vector-render-encoder-failed";
      return false;
    }
    if (vertexBuffer != nil && vertexCount > 0) {
      FrameUiVectorUniforms uniforms{};
      uniforms.drawableW = static_cast<float>(std::max(width, 1));
      uniforms.drawableH = static_cast<float>(std::max(height, 1));
      [encoder setRenderPipelineState:pipeline];
      [encoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
      [encoder setVertexBytes:&uniforms length:sizeof(uniforms) atIndex:1];
      [encoder drawPrimitives:MTLPrimitiveTypeTriangle
                  vertexStart:0
                  vertexCount:static_cast<NSUInteger>(vertexCount)];
    }
    [encoder endEncoding];
    return true;
  }
}

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
                                       std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "plot-surface-vector-command-buffer-failed";
    return false;
  }
  if (!encodePlotSurfaceVectorPrimitivesOnCommandBuffer(commandBuffer,
                                                        surfaceId,
                                                        width,
                                                        height,
                                                        pixelFormat,
                                                        vertices,
                                                        vertexCount,
                                                        clearBeforeDraw,
                                                        clearR,
                                                        clearG,
                                                        clearB,
                                                        clearA,
                                                        error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}
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
                                       std::string* error) {
  if (!retainPlotSurfaceForSubmission(submission, surfaceId, nullptr, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(submission, surfaceId, error)) {
    return false;
  }
  return encodePlotSurfaceVectorPrimitivesOnCommandBuffer(commandBuffer,
                                                          surfaceId,
                                                          width,
                                                          height,
                                                          pixelFormat,
                                                          vertices,
                                                          vertexCount,
                                                          clearBeforeDraw,
                                                          clearR,
                                                          clearG,
                                                          clearB,
                                                          clearA,
                                                          error);
}

static bool encodeSourceSignalSurfaceFromTextureOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> sourceTexture,
    int sourceSurfaceWidth,
    int sourceSurfaceHeight,
    int sourceSurfacePixelFormat,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  if (error) error->clear();
  if (commandBuffer == nil) {
    if (error) *error = "source-signal-surface-command-buffer-unavailable";
    return false;
  }
  if (sourceTexture == nil || outputSurfaceId == 0 ||
      sourceSurfaceWidth <= 0 || sourceSurfaceHeight <= 0 ||
      outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0) {
    if (error) *error = "invalid-source-signal-surface-request";
    return false;
  }
  if ((sourceSurfacePixelFormat != 0 && sourceSurfacePixelFormat != 1) ||
      (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1)) {
    if (error) *error = "unsupported-source-signal-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer, &runtimeContext,
                               &contextPointer, error) ||
      contextPointer == nullptr) {
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.sourceSignalSurfacePipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "source-signal-surface");
    }
    return false;
  }
  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "source-signal-output-surface-missing";
    return false;
  }
  @autoreleasepool {
    SourceSignalSurfaceUniforms uniforms{};
    uniforms.sourceWidth = sourceSurfaceWidth;
    uniforms.sourceHeight = sourceSurfaceHeight;
    uniforms.outputWidth = outputSurfaceWidth;
    uniforms.outputHeight = outputSurfaceHeight;
    uniforms.backgroundR = 0.010f;
    uniforms.backgroundG = 0.011f;
    uniforms.backgroundB = 0.013f;
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(ctx, &uniforms, 1u);
    if (uniformBuffer == nil) {
      if (error) *error = "source-signal-surface-uniform-allocation-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "source-signal-surface-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.sourceSignalSurfacePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setTexture:outputTexture atIndex:1];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:0];
    NSUInteger groupWidth = ctx.sourceSignalSurfacePipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.sourceSignalSurfacePipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    MTLSize threadsPerGroup = MTLSizeMake(groupWidth, groupHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                         static_cast<NSUInteger>(outputSurfaceHeight),
                                         1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
    [encoder endEncoding];
    return true;
  }
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
static bool encodeSourceSignalSurfaceFromIOSurfaceOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    uint32_t sourceSurfaceId,
    int sourceSurfaceWidth,
    int sourceSurfaceHeight,
    int sourceSurfacePixelFormat,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) {
      *error =
          localError.empty() ? "metal-context-unavailable" : localError;
    }
    return false;
  }
  id<MTLTexture> sourceTexture =
      makeTextureFromIOSurface(context(),
                               sourceSurfaceId,
                               sourceSurfaceWidth,
                               sourceSurfaceHeight,
                               sourceSurfacePixelFormat,
                               &localError);
  if (sourceTexture == nil) {
    if (error) {
      *error = localError.empty() ? "source-signal-iosurface-import-failed"
                                  : localError;
    }
    return false;
  }
  return encodeSourceSignalSurfaceFromTextureOnCommandBuffer(
      commandBuffer,
      sourceTexture,
      sourceSurfaceWidth,
      sourceSurfaceHeight,
      sourceSurfacePixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}

bool renderSourceSignalSurfaceFromIOSurface(uint32_t sourceSurfaceId,
                                            int sourceSurfaceWidth,
                                            int sourceSurfaceHeight,
                                            int sourceSurfacePixelFormat,
                                            uint32_t outputSurfaceId,
                                            int outputSurfaceWidth,
                                            int outputSurfaceHeight,
                                            int outputSurfacePixelFormat,
                                            std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "source-signal-surface-command-buffer-failed";
    return false;
  }
  if (!encodeSourceSignalSurfaceFromIOSurfaceOnCommandBuffer(commandBuffer,
                                                             sourceSurfaceId,
                                                             sourceSurfaceWidth,
                                                             sourceSurfaceHeight,
                                                             sourceSurfacePixelFormat,
                                                             outputSurfaceId,
                                                             outputSurfaceWidth,
                                                             outputSurfaceHeight,
                                                             outputSurfacePixelFormat,
                                                             error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}

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
    std::string* error) {
  if (!retainPlotSurfaceForSubmission(
          submission, outputSurfaceId, nullptr, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  return encodeSourceSignalSurfaceFromIOSurfaceOnCommandBuffer(commandBuffer,
                                                               sourceSurfaceId,
                                                               sourceSurfaceWidth,
                                                               sourceSurfaceHeight,
                                                               sourceSurfacePixelFormat,
                                                               outputSurfaceId,
                                                               outputSurfaceWidth,
                                                               outputSurfaceHeight,
                                                               outputSurfacePixelFormat,
                                                               error);
}
#endif

bool encodeSourceSignalSurfaceFromImportedTexture(
    const FrameSubmission& submission,
    uint64_t sourceId,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  if (!retainPlotSurfaceForSubmission(
          submission, outputSurfaceId, nullptr, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  return encodeSourceSignalSurfaceFromTextureOnCommandBuffer(
      commandBuffer,
      source->texture,
      source->descriptor.width,
      source->descriptor.height,
      source->descriptor.pixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}

static bool encodeStableRasterPointCompaction(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLBuffer> sourceVertices,
    id<MTLBuffer> sourceColors,
    NSUInteger pointCount,
    id<MTLBuffer> compactVertices,
    id<MTLBuffer> compactColors,
    id<MTLBuffer> indirectArguments,
    std::string* error) {
  constexpr NSUInteger kBlockWidth = 256u;
  constexpr size_t kMaximumHierarchyLevels = 8u;
  if (commandBuffer == nil || sourceVertices == nil || sourceColors == nil ||
      compactVertices == nil || compactColors == nil ||
      indirectArguments == nil || pointCount == 0u ||
      pointCount > static_cast<NSUInteger>(std::numeric_limits<uint32_t>::max())) {
    if (error) *error = "raster-point-compaction-request-invalid";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               error) ||
      contextPointer == nullptr) {
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.rasterPointCompactLocalScanPipeline == nil ||
      ctx.rasterPointScanBlockSumsPipeline == nil ||
      ctx.rasterPointAddBlockOffsetsPipeline == nil ||
      ctx.rasterPointCompactScatterPipeline == nil ||
      ctx.rasterPointFinalizeIndirectArgsPipeline == nil ||
      ctx.rasterPointCompactLocalScanPipeline.maxTotalThreadsPerThreadgroup <
          kBlockWidth ||
      ctx.rasterPointScanBlockSumsPipeline.maxTotalThreadsPerThreadgroup <
          kBlockWidth ||
      ctx.rasterPointAddBlockOffsetsPipeline.maxTotalThreadsPerThreadgroup <
          kBlockWidth ||
      ctx.rasterPointCompactScatterPipeline.maxTotalThreadsPerThreadgroup <
          kBlockWidth) {
    if (error) *error = "raster-point-compaction-pipeline-unavailable";
    return false;
  }

  const uint32_t pointCount32 = static_cast<uint32_t>(pointCount);
  const NSUInteger firstBlockCount =
      (pointCount + kBlockWidth - 1u) / kBlockWidth;
  if (pointCount > std::numeric_limits<NSUInteger>::max() / sizeof(uint32_t) ||
      firstBlockCount >
          std::numeric_limits<NSUInteger>::max() / sizeof(uint32_t)) {
    if (error) *error = "raster-point-compaction-size-overflow";
    return false;
  }

  id<MTLBuffer> pointLocalOffsets =
      makeSubmissionTransientPrivateBuffer(
          commandBuffer, pointCount * sizeof(uint32_t), error);
  id<MTLBuffer> firstBlockSums =
      makeSubmissionTransientPrivateBuffer(
          commandBuffer, firstBlockCount * sizeof(uint32_t), error);
  if (pointLocalOffsets == nil || firstBlockSums == nil) {
    if (error && error->empty()) {
      *error = "raster-point-compaction-scratch-allocation-failed";
    }
    return false;
  }

  id<MTLComputeCommandEncoder> localScanEncoder =
      [commandBuffer computeCommandEncoder];
  if (localScanEncoder == nil) {
    if (error) *error = "raster-point-compaction-local-scan-encoder-failed";
    return false;
  }
  [localScanEncoder
      setComputePipelineState:ctx.rasterPointCompactLocalScanPipeline];
  [localScanEncoder setBuffer:sourceColors offset:0 atIndex:0];
  [localScanEncoder setBuffer:pointLocalOffsets offset:0 atIndex:1];
  [localScanEncoder setBuffer:firstBlockSums offset:0 atIndex:2];
  [localScanEncoder setBytes:&pointCount32
                       length:sizeof(pointCount32)
                      atIndex:3];
  [localScanEncoder dispatchThreadgroups:MTLSizeMake(firstBlockCount, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlockWidth, 1, 1)];
  [localScanEncoder endEncoding];

  std::array<id<MTLBuffer>, kMaximumHierarchyLevels> blockSums{};
  std::array<id<MTLBuffer>, kMaximumHierarchyLevels> blockOffsets{};
  std::array<NSUInteger, kMaximumHierarchyLevels> blockCounts{};
  blockSums[0] = firstBlockSums;
  blockCounts[0] = firstBlockCount;
  size_t hierarchyLevels = 0u;
  for (size_t level = 0u; level < kMaximumHierarchyLevels; ++level) {
    const NSUInteger count = blockCounts[level];
    const NSUInteger nextCount =
        (count + kBlockWidth - 1u) / kBlockWidth;
    if (count == 0u ||
        count > std::numeric_limits<NSUInteger>::max() / sizeof(uint32_t) ||
        nextCount >
            std::numeric_limits<NSUInteger>::max() / sizeof(uint32_t)) {
      if (error) *error = "raster-point-compaction-hierarchy-size-invalid";
      return false;
    }
    blockOffsets[level] =
        makeSubmissionTransientPrivateBuffer(
            commandBuffer, count * sizeof(uint32_t), error);
    id<MTLBuffer> nextBlockSums =
        makeSubmissionTransientPrivateBuffer(
            commandBuffer, nextCount * sizeof(uint32_t), error);
    if (blockOffsets[level] == nil || nextBlockSums == nil) {
      if (error && error->empty()) {
        *error = "raster-point-compaction-hierarchy-allocation-failed";
      }
      return false;
    }
    const uint32_t count32 = static_cast<uint32_t>(count);
    id<MTLComputeCommandEncoder> scanEncoder =
        [commandBuffer computeCommandEncoder];
    if (scanEncoder == nil) {
      if (error) *error = "raster-point-compaction-block-scan-encoder-failed";
      return false;
    }
    [scanEncoder setComputePipelineState:ctx.rasterPointScanBlockSumsPipeline];
    [scanEncoder setBuffer:blockSums[level] offset:0 atIndex:0];
    [scanEncoder setBuffer:blockOffsets[level] offset:0 atIndex:1];
    [scanEncoder setBuffer:nextBlockSums offset:0 atIndex:2];
    [scanEncoder setBytes:&count32 length:sizeof(count32) atIndex:3];
    [scanEncoder dispatchThreadgroups:MTLSizeMake(nextCount, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(kBlockWidth, 1, 1)];
    [scanEncoder endEncoding];
    hierarchyLevels = level + 1u;
    if (nextCount == 1u) break;
    if (level + 1u >= kMaximumHierarchyLevels) {
      if (error) *error = "raster-point-compaction-hierarchy-too-deep";
      return false;
    }
    blockSums[level + 1u] = nextBlockSums;
    blockCounts[level + 1u] = nextCount;
  }
  if (hierarchyLevels == 0u) {
    if (error) *error = "raster-point-compaction-hierarchy-empty";
    return false;
  }

  // The top scan is globally complete. Propagate its offsets down one level
  // at a time so the first-level block offsets address the entire point set.
  for (size_t level = hierarchyLevels; level > 1u; --level) {
    const size_t childLevel = level - 2u;
    const uint32_t count32 =
        static_cast<uint32_t>(blockCounts[childLevel]);
    id<MTLComputeCommandEncoder> addEncoder =
        [commandBuffer computeCommandEncoder];
    if (addEncoder == nil) {
      if (error) *error = "raster-point-compaction-offset-add-encoder-failed";
      return false;
    }
    [addEncoder setComputePipelineState:ctx.rasterPointAddBlockOffsetsPipeline];
    [addEncoder setBuffer:blockOffsets[childLevel] offset:0 atIndex:0];
    [addEncoder setBuffer:blockOffsets[childLevel + 1u] offset:0 atIndex:1];
    [addEncoder setBytes:&count32 length:sizeof(count32) atIndex:2];
    [addEncoder dispatchThreads:MTLSizeMake(blockCounts[childLevel], 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kBlockWidth, 1, 1)];
    [addEncoder endEncoding];
  }

  id<MTLComputeCommandEncoder> scatterEncoder =
      [commandBuffer computeCommandEncoder];
  if (scatterEncoder == nil) {
    if (error) *error = "raster-point-compaction-scatter-encoder-failed";
    return false;
  }
  [scatterEncoder setComputePipelineState:ctx.rasterPointCompactScatterPipeline];
  [scatterEncoder setBuffer:sourceVertices offset:0 atIndex:0];
  [scatterEncoder setBuffer:sourceColors offset:0 atIndex:1];
  [scatterEncoder setBuffer:pointLocalOffsets offset:0 atIndex:2];
  [scatterEncoder setBuffer:blockOffsets[0] offset:0 atIndex:3];
  [scatterEncoder setBuffer:compactVertices offset:0 atIndex:4];
  [scatterEncoder setBuffer:compactColors offset:0 atIndex:5];
  [scatterEncoder setBytes:&pointCount32
                     length:sizeof(pointCount32)
                    atIndex:6];
  [scatterEncoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlockWidth, 1, 1)];
  [scatterEncoder endEncoding];

  const uint32_t firstBlockCount32 =
      static_cast<uint32_t>(firstBlockCount);
  id<MTLComputeCommandEncoder> finalizeEncoder =
      [commandBuffer computeCommandEncoder];
  if (finalizeEncoder == nil) {
    if (error) *error = "raster-point-compaction-finalize-encoder-failed";
    return false;
  }
  [finalizeEncoder
      setComputePipelineState:ctx.rasterPointFinalizeIndirectArgsPipeline];
  [finalizeEncoder setBuffer:firstBlockSums offset:0 atIndex:0];
  [finalizeEncoder setBuffer:blockOffsets[0] offset:0 atIndex:1];
  [finalizeEncoder setBytes:&firstBlockCount32
                      length:sizeof(firstBlockCount32)
                     atIndex:2];
  [finalizeEncoder setBuffer:indirectArguments offset:0 atIndex:3];
  [finalizeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [finalizeEncoder endEncoding];
  return true;
}

static bool encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> importedSourceTexture,
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
    std::string* error,
    const ScopeDerivedResidentRecord* residentRecord = nullptr,
    ScopeDerivedResidentRecord* outRecord = nullptr,
    uint64_t buildSerial = 0) {
  if (error) error->clear();
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(rasterRequest.pointCount, 0));
  const bool useResidentRecord =
      residentRecord != nullptr &&
      residentRecord->family == ScopeDerivedFamily::RasterPointCloud &&
      residentRecord->pointVertices != nil && residentRecord->pointColors != nil &&
      residentRecord->pointIndirectArguments != nil &&
      residentRecord->pointCount == pointCount && residentRecord->byteSize != 0u;
  if (residentRecord != nullptr && !useResidentRecord) {
    if (error) *error = "raster-point-resident-record-invalid";
    return false;
  }
  if (outRecord != nullptr && buildSerial == 0u) {
    if (error) *error = "raster-point-build-serial-invalid";
    return false;
  }
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
  if ((!useResidentRecord && importedSourceTexture == nil) ||
#else
  if ((!useResidentRecord && importedSourceTexture == nil && sourceSurfaceId == 0) ||
#endif
      outputSurfaceId == 0 ||
      sourceSurfaceWidth <= 0 || sourceSurfaceHeight <= 0 ||
      outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || pointCount == 0) {
    if (error) *error = "invalid-raster-point-surface-request";
    return false;
  }
  if ((sourceSurfacePixelFormat != 0 && sourceSurfacePixelFormat != 1) ||
      (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1)) {
    if (error) *error = "unsupported-raster-point-surface-format";
    return false;
  }
  if (!useResidentRecord &&
      (sourceSurfaceWidth < rasterRequest.sourceWidth ||
       sourceSurfaceHeight < rasterRequest.sourceHeight)) {
    if (error) *error = "raster-point-source-surface-too-small";
    return false;
  }
  std::string localError;
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty() ? "raster-point-command-buffer-missing"
                                  : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if ((!useResidentRecord &&
       (ctx.rasterSourceTexturePipeline == nil ||
        ctx.rasterPointCompactLocalScanPipeline == nil ||
        ctx.rasterPointScanBlockSumsPipeline == nil ||
        ctx.rasterPointAddBlockOffsetsPipeline == nil ||
        ctx.rasterPointCompactScatterPipeline == nil ||
        ctx.rasterPointFinalizeIndirectArgsPipeline == nil)) ||
      ctx.rasterPointSurfacePipeline16 == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "raster-point-surface");
    }
    return false;
  }
  if (!useResidentRecord && rasterRequest.occupancyFill != 0 &&
      rasterRequest.occupancyAppendCount > 0 &&
      (ctx.rasterOccupancyTextureCountPipeline == nil ||
       ctx.rasterOccupancyThresholdPipeline == nil)) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "raster-occupancy-texture");
    }
    return false;
  }
  id<MTLRenderPipelineState> renderPipeline =
      outputSurfacePixelFormat == 1 ? ctx.rasterPointSurfacePipeline32 : ctx.rasterPointSurfacePipeline16;
  if (renderPipeline == nil) {
    if (error) *error = "raster-point-surface-output-format-unavailable";
    return false;
  }

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "raster-point-output-surface-missing";
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = sourceSurfacePixelFormat;

  RasterPointSurfaceUniforms pointUniforms{};
  std::copy(pointRequest.modelView, pointRequest.modelView + 16, pointUniforms.modelView);
  std::copy(pointRequest.projection, pointRequest.projection + 16, pointUniforms.projection);
  pointUniforms.pointRadiusPixels = std::max(0.5f, pointRequest.pointRadiusPixels);
  pointUniforms.surfaceWidth = static_cast<float>(std::max(outputSurfaceWidth, 1));
  pointUniforms.surfaceHeight = static_cast<float>(std::max(outputSurfaceHeight, 1));

  @autoreleasepool {
    if (pointCount > std::numeric_limits<size_t>::max() /
                         sizeof(PackedFloat3) ||
        pointCount > std::numeric_limits<size_t>::max() /
                         sizeof(simd_float4)) {
      if (error) *error = "raster-point-buffer-size-overflow";
      return false;
    }
    const size_t vertexBytes =
        static_cast<size_t>(pointCount) * sizeof(PackedFloat3);
    const size_t colorBytes =
        static_cast<size_t>(pointCount) * sizeof(simd_float4);
    constexpr size_t kIndirectArgumentBytes = 4u * sizeof(uint32_t);
    if (vertexBytes > std::numeric_limits<size_t>::max() - colorBytes ||
        vertexBytes + colorBytes >
            std::numeric_limits<size_t>::max() - kIndirectArgumentBytes) {
      if (error) *error = "raster-point-cache-size-overflow";
      return false;
    }
    const size_t residentBytes =
        vertexBytes + colorBytes + kIndirectArgumentBytes;
    if (useResidentRecord &&
        (residentRecord->byteSize != residentBytes ||
         [residentRecord->pointVertices length] != vertexBytes ||
         [residentRecord->pointColors length] != colorBytes ||
         [residentRecord->pointIndirectArguments length] !=
             kIndirectArgumentBytes)) {
      if (error) *error = "raster-point-resident-record-size-mismatch";
      return false;
    }

    id<MTLBuffer> vertBuffer =
        useResidentRecord ? residentRecord->pointVertices : nil;
    id<MTLBuffer> colorBuffer =
        useResidentRecord ? residentRecord->pointColors : nil;
    id<MTLBuffer> indirectArguments =
        useResidentRecord ? residentRecord->pointIndirectArguments : nil;
    id<MTLBuffer> pointUniformBuffer =
        makeSharedBuffer(ctx, &pointUniforms, 1u);
    if (pointUniformBuffer == nil) {
      if (error) *error = "raster-point-uniform-allocation-failed";
      return false;
    }

    if (!useResidentRecord) {
      id<MTLTexture> sourceTexture = importedSourceTexture;
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
      if (sourceTexture == nil) {
        sourceTexture = makeTextureFromIOSurface(ctx,
                                                 sourceSurfaceId,
                                                 sourceSurfaceWidth,
                                                 sourceSurfaceHeight,
                                                 sourceSurfacePixelFormat,
                                                 &localError);
      }
#endif
      id<MTLBuffer> derivedVertBuffer =
          makeSubmissionTransientPrivateBuffer(
              commandBuffer, vertexBytes, &localError);
      id<MTLBuffer> derivedColorBuffer =
          makeSubmissionTransientPrivateBuffer(
              commandBuffer, colorBytes, &localError);
      vertBuffer = makeEmptyPrivateBuffer(ctx, vertexBytes);
      colorBuffer = makeEmptyPrivateBuffer(ctx, colorBytes);
      indirectArguments = makeEmptyPrivateBuffer(ctx, kIndirectArgumentBytes);
      constexpr NSUInteger kRasterOccupancyBinCount = 18u * 18u * 18u;
      id<MTLBuffer> occupancyBuffer =
          makeSubmissionTransientPrivateBuffer(
              commandBuffer,
              kRasterOccupancyBinCount * sizeof(uint32_t),
              &localError);
      const bool useOccupancyThreshold =
          rasterRequest.occupancyFill != 0 &&
          rasterRequest.occupancyAppendCount > 0 &&
          rasterUniforms.basePointCount > 0;
      id<MTLBuffer> visibleCountBuffer =
          useOccupancyThreshold
              ? makeSubmissionTransientPrivateBuffer(
                    commandBuffer, sizeof(uint32_t), &localError)
              : nil;
      id<MTLBuffer> rasterUniformBuffer =
          makeSharedBuffer(ctx, &rasterUniforms, 1u);
      if (sourceTexture == nil || derivedVertBuffer == nil ||
          derivedColorBuffer == nil || vertBuffer == nil ||
          colorBuffer == nil || indirectArguments == nil ||
          occupancyBuffer == nil || rasterUniformBuffer == nil ||
          (useOccupancyThreshold && visibleCountBuffer == nil)) {
        if (error) {
          *error = localError.empty() ? "raster-point-buffer-allocation-failed"
                                      : localError;
        }
        return false;
      }
      id<MTLBlitCommandEncoder> clearEncoder =
          [commandBuffer blitCommandEncoder];
      if (clearEncoder == nil) {
        if (error) *error = "raster-point-occupancy-clear-encoder-failed";
        return false;
      }
      [clearEncoder fillBuffer:occupancyBuffer
                         range:NSMakeRange(0, [occupancyBuffer length])
                         value:0];
      if (visibleCountBuffer != nil) {
        [clearEncoder fillBuffer:visibleCountBuffer
                           range:NSMakeRange(0, [visibleCountBuffer length])
                           value:0];
      }
      [clearEncoder endEncoding];

      if (useOccupancyThreshold) {
        id<MTLComputeCommandEncoder> countEncoder =
            [commandBuffer computeCommandEncoder];
        if (countEncoder == nil) {
          if (error) *error = "raster-point-occupancy-encoder-failed";
          return false;
        }
        [countEncoder setComputePipelineState:ctx.rasterOccupancyTextureCountPipeline];
        [countEncoder setTexture:sourceTexture atIndex:0];
        [countEncoder setBuffer:occupancyBuffer offset:0 atIndex:0];
        [countEncoder setBuffer:visibleCountBuffer offset:0 atIndex:1];
        [countEncoder setBuffer:rasterUniformBuffer offset:0 atIndex:2];
        NSUInteger countWidth =
            ctx.rasterOccupancyTextureCountPipeline.maxTotalThreadsPerThreadgroup;
        if (countWidth == 0) countWidth = 64;
        countWidth = std::min<NSUInteger>(countWidth, 256);
        [countEncoder dispatchThreads:MTLSizeMake(
                                          static_cast<NSUInteger>(std::max(
                                              rasterUniforms.basePointCount, 0)),
                                          1,
                                          1)
                 threadsPerThreadgroup:MTLSizeMake(countWidth, 1, 1)];
        [countEncoder endEncoding];

        id<MTLComputeCommandEncoder> thresholdEncoder =
            [commandBuffer computeCommandEncoder];
        if (thresholdEncoder == nil) {
          if (error) *error = "raster-point-occupancy-threshold-encoder-failed";
          return false;
        }
        [thresholdEncoder setComputePipelineState:ctx.rasterOccupancyThresholdPipeline];
        [thresholdEncoder setBuffer:visibleCountBuffer offset:0 atIndex:0];
        [thresholdEncoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
        [thresholdEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [thresholdEncoder endEncoding];
      }

      id<MTLComputeCommandEncoder> computeEncoder =
          [commandBuffer computeCommandEncoder];
      if (computeEncoder == nil) {
        if (error) *error = "raster-point-compute-encoder-failed";
        return false;
      }
      [computeEncoder setComputePipelineState:ctx.rasterSourceTexturePipeline];
      [computeEncoder setTexture:sourceTexture atIndex:0];
      [computeEncoder setBuffer:derivedVertBuffer offset:0 atIndex:0];
      [computeEncoder setBuffer:derivedColorBuffer offset:0 atIndex:1];
      [computeEncoder setBuffer:occupancyBuffer offset:0 atIndex:2];
      [computeEncoder setBuffer:rasterUniformBuffer offset:0 atIndex:3];
      NSUInteger computeWidth =
          ctx.rasterSourceTexturePipeline.maxTotalThreadsPerThreadgroup;
      if (computeWidth == 0) computeWidth = 64;
      computeWidth = std::min<NSUInteger>(computeWidth, 256);
      [computeEncoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(computeWidth, 1, 1)];
      [computeEncoder endEncoding];
      if (!encodeStableRasterPointCompaction(commandBuffer,
                                              derivedVertBuffer,
                                              derivedColorBuffer,
                                              pointCount,
                                              vertBuffer,
                                              colorBuffer,
                                              indirectArguments,
                                              error)) {
        return false;
      }
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = outputTexture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor =
        MTLClearColorMake(static_cast<double>(pointRequest.backgroundR),
                          static_cast<double>(pointRequest.backgroundG),
                          static_cast<double>(pointRequest.backgroundB),
                          static_cast<double>(pointRequest.backgroundA));
    id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    if (renderEncoder == nil) {
      if (error) *error = "raster-point-render-encoder-failed";
      return false;
    }
    [renderEncoder setRenderPipelineState:renderPipeline];
    [renderEncoder setVertexBuffer:vertBuffer offset:0 atIndex:0];
    [renderEncoder setVertexBuffer:colorBuffer offset:0 atIndex:1];
    [renderEncoder setVertexBuffer:pointUniformBuffer offset:0 atIndex:2];
    [renderEncoder drawPrimitives:MTLPrimitiveTypePoint
                   indirectBuffer:indirectArguments
             indirectBufferOffset:0];
    [renderEncoder endEncoding];

    if (outRecord != nullptr) {
      ScopeDerivedResidentRecord encoded{};
      encoded.family = ScopeDerivedFamily::RasterPointCloud;
      encoded.builtSerial = buildSerial;
      encoded.byteSize = residentBytes;
      encoded.pointVertices = vertBuffer;
      encoded.pointColors = colorBuffer;
      encoded.pointIndirectArguments = indirectArguments;
      encoded.pointCount = pointCount;
      *outRecord = std::move(encoded);
    }
  }
  return true;
}

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
                                           std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "raster-point-command-buffer-failed";
    return false;
  }
  if (!encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          nil,
          rasterRequest,
          pointRequest,
          sourceSurfaceId,
          sourceSurfaceWidth,
          sourceSurfaceHeight,
          sourceSurfacePixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}

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
                                           std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  return encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      nil,
      rasterRequest,
      pointRequest,
      sourceSurfaceId,
      sourceSurfaceWidth,
      sourceSurfaceHeight,
      sourceSurfacePixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
          error);
}
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
    std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  return encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      source->texture,
      rasterRequest,
      pointRequest,
      0,
      source->descriptor.width,
      source->descriptor.height,
      source->descriptor.pixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}

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
    std::string* error) {
  if (!cache) {
    if (error) *error = "metal-raster-point-cache-missing";
    return false;
  }
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache->cacheId, cache->ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error) ||
      !validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }

  ScopeDerivedResidentRecord residentRecord{};
  std::string residentError;
  if (resolveScopeDerivedRecordForSubmission(
          submission, *cache, true, &residentRecord, &residentError)) {
    if (residentRecord.family != ScopeDerivedFamily::RasterPointCloud) {
      if (error) *error = "metal-raster-point-cache-family-mismatch";
      return false;
    }
    return encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
        commandBuffer,
        nil,
        rasterRequest,
        pointRequest,
        0,
        rasterRequest.sourceWidth,
        rasterRequest.sourceHeight,
        rasterRequest.pixelFormat,
        outputSurfaceId,
        outputSurfaceWidth,
        outputSurfaceHeight,
        outputSurfacePixelFormat,
        error,
        &residentRecord);
  }
  if (!residentError.empty()) {
    if (error) *error = residentError;
    return false;
  }

  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  ScopeDerivedResidentRecord encodedRecord{};
  if (!encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          source->texture,
          rasterRequest,
          pointRequest,
          0,
          source->descriptor.width,
          source->descriptor.height,
          source->descriptor.pixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error,
          nullptr,
          &encodedRecord,
          buildSerial)) {
    return false;
  }
  return registerPendingScopeDerivedRecord(
      submission, cache, std::move(encodedRecord), error);
}

void releasePlotSurface(uint32_t surfaceId) {
  if (surfaceId == 0) return;
  std::lock_guard<std::mutex> lock(plotSurfaceMutex());
  auto& registry = plotSurfaceRegistry();
  auto it = registry.find(surfaceId);
  if (it == registry.end() || !it->second ||
      it->second->ownerCompositorId != 0) return;
  registry.erase(it);
}

void releasePrivatePlotSurface(uint64_t compositorId, uint32_t surfaceId) {
  if (compositorId == 0 || surfaceId == 0) return;
  std::lock_guard<std::mutex> lock(plotSurfaceMutex());
  auto& registry = plotSurfaceRegistry();
  auto it = registry.find(surfaceId);
  if (it == registry.end() ||
      !it->second || it->second->ownerCompositorId != compositorId) {
    return;
  }
  registry.erase(it);
}

void releaseGlossFieldCache(GlossFieldCache* cache) {
  if (!cache) return;
  ScopeDerivedCache derived = glossDerivedCache(*cache);
  releaseResidentDerivedCache(&derived);
  *cache = GlossFieldCache{};
}

ResidentDerivedCacheState residentDerivedCacheState(
    const ResidentDerivedCache& cache) {
  if (cache.cacheId == 0 || cache.byteSize == 0 || !cache.available) {
    return ResidentDerivedCacheState::Missing;
  }
  std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
  auto entryIt = scopeDerivedRegistry().find(cache.cacheId);
  if (entryIt == scopeDerivedRegistry().end() ||
      entryIt->second.ownerCompositorId != cache.ownerCompositorId) {
    return ResidentDerivedCacheState::Missing;
  }
  for (const auto& version : entryIt->second.inFlight) {
    if (version.state == ScopeDerivedResidentVersionState::Pending &&
        version.record &&
        scopeDerivedRecordMatchesCache(*version.record, cache)) {
      return ResidentDerivedCacheState::Pending;
    }
  }
  return resolveScopeDerivedRecordLocked(cache.cacheId,
                                         0,
                                         false,
                                         cache,
                                         nullptr)
             ? ResidentDerivedCacheState::Ready
             : ResidentDerivedCacheState::Missing;
}

void releaseResidentDerivedCache(ResidentDerivedCache* cache) {
  if (!cache) return;
  if (cache->cacheId != 0) {
    std::lock_guard<std::mutex> lock(scopeDerivedRegistryMutex());
    const auto entryIt = scopeDerivedRegistry().find(cache->cacheId);
    if (entryIt != scopeDerivedRegistry().end() &&
        entryIt->second.ownerCompositorId == cache->ownerCompositorId) {
      scopeDerivedRegistry().erase(entryIt);
    }
  }
  *cache = ResidentDerivedCache{};
}

ScopeDerivedCacheState scopeDerivedCacheState(
    const ScopeDerivedCache& cache) {
  return residentDerivedCacheState(cache);
}

void releaseScopeDerivedCache(ScopeDerivedCache* cache) {
  releaseResidentDerivedCache(cache);
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
static bool bindIOSurfaceRefToOpenGLTexture(IOSurfaceRef surface,
                                            int width,
                                            int height,
                                            int pixelFormat,
                                            uint32_t glTexture,
                                            std::string* error) {
  if (surface == nullptr) {
    if (error) *error = "missing-iosurface-for-opengl-bind";
    return false;
  }
  CGLContextObj cgl = CGLGetCurrentContext();
  if (cgl == nullptr) {
    if (error) *error = "no-current-cgl-context";
    return false;
  }
  glBindTexture(GL_TEXTURE_RECTANGLE, static_cast<GLuint>(glTexture));
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  const GLenum internalFormat = pixelFormat == 1 ? GL_RGBA32F : GL_RGBA16F;
  const GLenum type = pixelFormat == 1 ? GL_FLOAT : GL_HALF_FLOAT;
  const CGLError cgError = CGLTexImageIOSurface2D(cgl,
                                                 GL_TEXTURE_RECTANGLE,
                                                 internalFormat,
                                                 static_cast<GLsizei>(width),
                                                 static_cast<GLsizei>(height),
                                                 GL_RGBA,
                                                 type,
                                                 surface,
                                                 0);
  glBindTexture(GL_TEXTURE_RECTANGLE, 0);
  if (cgError != kCGLNoError) {
    if (error) *error = std::string("CGLTexImageIOSurface2D failed error=") + std::to_string(cgError);
    return false;
  }
  return true;
}

bool bindIOSurfaceToOpenGLTexture(uint32_t surfaceId,
                                  int width,
                                  int height,
                                  int pixelFormat,
                                  uint32_t glTexture,
                                  std::string* error) {
  if (error) error->clear();
  if (surfaceId == 0 || width <= 0 || height <= 0 || glTexture == 0) {
    if (error) *error = "invalid-iosurface-gl-texture-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-iosurface-pixel-format";
    return false;
  }
  @autoreleasepool {
    IOSurfaceRef surface = IOSurfaceLookup(static_cast<IOSurfaceID>(surfaceId));
    if (surface == nullptr) {
      if (error) *error = "iosurface-lookup-failed";
      return false;
    }
    const bool bound =
        bindIOSurfaceRefToOpenGLTexture(surface, width, height, pixelFormat, glTexture, error);
    CFRelease(surface);
    return bound;
  }
}

bool bindPlotSurfaceToOpenGLTexture(uint32_t surfaceId,
                                   int width,
                                   int height,
                                   int pixelFormat,
                                   uint32_t glTexture,
                                   std::string* error) {
  if (error) error->clear();
  if (surfaceId == 0 || width <= 0 || height <= 0 || glTexture == 0) {
    if (error) *error = "invalid-plot-surface-gl-texture-request";
    return false;
  }
  if (pixelFormat != 0 && pixelFormat != 1) {
    if (error) *error = "unsupported-plot-surface-pixel-format";
    return false;
  }

  IOSurfaceRef surface = nullptr;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(surfaceId);
    if (it == registry.end()) {
      if (error) *error = "plot-surface-handle-not-found";
      return false;
    }
    if (!it->second || it->second->width != width ||
        it->second->height != height ||
        it->second->pixelFormat != pixelFormat) {
      if (error) *error = "plot-surface-metadata-mismatch";
      return false;
    }
    if (it->second->surface == nullptr) {
      if (error) *error = "plot-surface-not-iosurface-backed";
      return false;
    }
    CFRetain(it->second->surface);
    surface = it->second->surface;
  }

  @autoreleasepool {
    const bool bound =
        bindIOSurfaceRefToOpenGLTexture(surface, width, height, pixelFormat, glTexture, error);
    CFRelease(surface);
    return bound;
  }
}
#endif

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildOverlayMesh(const OverlayRequest& request,
                      const std::vector<float>& inputPoints,
                      std::vector<float>* outVerts,
                      std::vector<float>* outColors,
                      std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int cubeSize = request.cubeSize > 0 ? request.cubeSize : 1;
  const NSUInteger cubePoints = static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize);
  const NSUInteger rampPoints = request.ramp != 0 ? static_cast<NSUInteger>(cubeSize) * static_cast<NSUInteger>(cubeSize) : 0u;
  const NSUInteger uploadedPoints = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  const NSUInteger totalPoints = request.useInputPoints != 0 ? uploadedPoints : (cubePoints + rampPoints);
  if (totalPoints == 0) {
    if (outVerts) outVerts->clear();
    if (outColors) outColors->clear();
    return true;
  }

  MetalContext& ctx = context();
  OverlayUniforms uniforms{};
  uniforms.cubeSize = cubeSize;
  uniforms.ramp = request.ramp;
  uniforms.useInputPoints = request.useInputPoints;
  uniforms.pointCount = request.pointCount;
  uniforms.colorSaturation = request.colorSaturation;
  uniforms.plotMode = request.remap.plotMode;
  uniforms.circularHsl = request.remap.circularHsl;
  uniforms.circularHsv = request.remap.circularHsv;
  uniforms.normConeNormalized = request.remap.normConeNormalized;
  uniforms.chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms.chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms.chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms.chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms.chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms.chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }

  id<MTLBuffer> inputBuffer = nil;
  if (request.useInputPoints != 0) {
    if (inputPoints.size() < uploadedPoints * 4u) {
      if (error) *error = "Overlay Metal input point buffer is undersized.";
      return false;
    }
    inputBuffer = makeSharedBuffer(reinterpret_cast<const simd_float4*>(inputPoints.data()), uploadedPoints);
  } else {
    simd_float4 dummy = {0.0f, 0.0f, 0.0f, 0.0f};
    inputBuffer = makeSharedBuffer(&dummy, 1u);
  }
  id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(totalPoints * sizeof(PackedFloat3));
  id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(totalPoints * sizeof(simd_float4));
  id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
  if (inputBuffer == nil || vertBuffer == nil || colorBuffer == nil || uniformBuffer == nil) {
    if (error) *error = "Failed to allocate Metal overlay buffers.";
    return false;
  }

  if (!runCompute(ctx.overlayPipeline, inputBuffer, vertBuffer, colorBuffer, uniformBuffer, totalPoints, &localError)) {
    if (error) *error = localError;
    return false;
  }

  copySharedBuffer<PackedFloat3>(vertBuffer, totalPoints, outVerts);
  copySharedBuffer<simd_float4>(colorBuffer, totalPoints, outColors);
  return true;
}
#endif

void fillInputUniforms(const InputRequest& request, InputUniforms* uniforms) {
  if (!uniforms) return;
  uniforms->pointCount = request.pointCount;
  uniforms->inputStride = request.inputStride;
  uniforms->glossView = request.glossView;
  uniforms->sourceAspect = request.sourceAspect;
  uniforms->glossLiftScale = request.glossLiftScale;
  uniforms->showOverflow = request.remap.showOverflow;
  uniforms->highlightOverflow = request.remap.highlightOverflow;
  uniforms->plotMode = request.remap.plotMode;
  uniforms->circularHsl = request.remap.circularHsl;
  uniforms->circularHsv = request.remap.circularHsv;
  uniforms->normConeNormalized = request.remap.normConeNormalized;
  uniforms->chromaticityInputTransfer = request.remap.chromaticityInputTransfer;
  uniforms->chromaticityReferenceBasis = request.remap.chromaticityReferenceBasis;
  uniforms->chromaticityWhiteX = request.remap.chromaticityWhiteX;
  uniforms->chromaticityWhiteY = request.remap.chromaticityWhiteY;
  for (int i = 0; i < 9; ++i) {
    uniforms->chromaticityRgbToXyz[i] = request.remap.chromaticityRgbToXyz[i];
    uniforms->chromaticityXyzToRgb[i] = request.remap.chromaticityXyzToRgb[i];
  }
  uniforms->pointAlphaScale = request.pointAlphaScale;
  uniforms->denseAlphaBias = request.denseAlphaBias;
  uniforms->colorSaturation = request.colorSaturation;
}

void fillRasterSourceUniforms(const RasterSourceRequest& request, RasterSourceUniforms* uniforms) {
  if (!uniforms) return;
  InputRequest inputRequest{};
  inputRequest.pointCount = request.pointCount;
  inputRequest.inputStride = 3;
  inputRequest.glossView = request.remap.plotMode == 9 ? 1 : 0;
  inputRequest.sourceAspect = request.sourceAspect;
  inputRequest.glossLiftScale = request.glossLiftScale;
  inputRequest.pointAlphaScale = request.pointAlphaScale;
  inputRequest.denseAlphaBias = request.denseAlphaBias;
  inputRequest.colorSaturation = request.colorSaturation;
  inputRequest.remap = request.remap;
  fillInputUniforms(inputRequest, &uniforms->input);
  uniforms->basePointCount = request.basePointCount > 0 ? request.basePointCount : request.pointCount;
  uniforms->sourceWidth = request.sourceWidth;
  uniforms->sourceHeight = request.sourceHeight;
  uniforms->sampleStride = request.sampleStride;
  uniforms->sampleCountX = request.sampleCountX;
  uniforms->pixelFormat = request.pixelFormat;
  uniforms->plotLinear = request.plotLinear;
  uniforms->plotLinearTransfer = request.plotLinearTransfer;
  uniforms->excludeIdentityData = request.excludeIdentityData;
  uniforms->isolateIdentityData = request.isolateIdentityData;
  uniforms->readIdentityPlot = request.readIdentityPlot;
  uniforms->readGrayRamp = request.readGrayRamp;
  uniforms->identityCubeY1 = request.identityCubeY1;
  uniforms->identityCubeY2 = request.identityCubeY2;
  uniforms->identityRampY1 = request.identityRampY1;
  uniforms->identityRampY2 = request.identityRampY2;
  uniforms->identityCubeAppendOffset = request.identityCubeAppendOffset;
  uniforms->identityCubeAppendCount = request.identityCubeAppendCount;
  uniforms->identityCubeAppendY1 = request.identityCubeAppendY1;
  uniforms->identityCubeAppendY2 = request.identityCubeAppendY2;
  uniforms->identityCubeAppendRowStep = request.identityCubeAppendRowStep;
  uniforms->identityCubeAppendXStep = request.identityCubeAppendXStep;
  uniforms->identityRampAppendOffset = request.identityRampAppendOffset;
  uniforms->identityRampAppendCount = request.identityRampAppendCount;
  uniforms->identityRampAppendY1 = request.identityRampAppendY1;
  uniforms->identityRampAppendY2 = request.identityRampAppendY2;
  uniforms->identityRampAppendRowStep = request.identityRampAppendRowStep;
  uniforms->identityRampAppendXStep = request.identityRampAppendXStep;
  uniforms->occupancyFill = request.occupancyFill;
  uniforms->occupancyAppendOffset = request.occupancyAppendOffset;
  uniforms->occupancyAppendCount = request.occupancyAppendCount;
  uniforms->occupancyCandidateCount = request.occupancyCandidateCount;
  uniforms->occupancyTargetThreshold = 0;
  uniforms->lassoEnabled = request.lassoEnabled;
  uniforms->lassoStrokeCount = request.lassoStrokeCount;
  uniforms->lassoPointCount = request.lassoPointCount;
  for (int i = 0; i < 16; ++i) {
    uniforms->lassoStrokeFirst[i] = request.lassoStrokeFirst[i];
    uniforms->lassoStrokeCountPerStroke[i] = request.lassoStrokeCountPerStroke[i];
    uniforms->lassoStrokeSubtract[i] = request.lassoStrokeSubtract[i];
  }
  for (int i = 0; i < 256; ++i) {
    uniforms->lassoX[i] = request.lassoX[i];
    uniforms->lassoY[i] = request.lassoY[i];
  }
  uniforms->cubeSlicingEnabled = request.cubeSlicingEnabled;
  uniforms->neutralRadiusEnabled = request.neutralRadiusEnabled;
  uniforms->neutralRadius = request.neutralRadius;
  uniforms->cubeSliceRed = request.cubeSliceRed;
  uniforms->cubeSliceYellow = request.cubeSliceYellow;
  uniforms->cubeSliceGreen = request.cubeSliceGreen;
  uniforms->cubeSliceCyan = request.cubeSliceCyan;
  uniforms->cubeSliceBlue = request.cubeSliceBlue;
  uniforms->cubeSliceMagenta = request.cubeSliceMagenta;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildInputMesh(const InputRequest& request,
                    const std::vector<float>& rawPoints,
                    std::vector<float>* outVerts,
                    std::vector<float>* outColors,
                    std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0) {
    if (outVerts) outVerts->clear();
    if (outColors) outColors->clear();
    return true;
  }
  const size_t inputStride = static_cast<size_t>(std::max(request.inputStride, 3));
  if (rawPoints.size() < pointCount * inputStride) {
    if (error) *error = "Input Metal raw point buffer is undersized.";
    return false;
  }

  MetalContext& ctx = context();
  InputUniforms uniforms{};
  fillInputUniforms(request, &uniforms);

  id<MTLBuffer> inputBuffer = makeSharedBuffer(rawPoints.data(), rawPoints.size());
  id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
  id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
  id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
  if (inputBuffer == nil || vertBuffer == nil || colorBuffer == nil || uniformBuffer == nil) {
    if (error) *error = "Failed to allocate Metal input-cloud buffers.";
    return false;
  }

  if (!runCompute(ctx.inputPipeline, inputBuffer, vertBuffer, colorBuffer, uniformBuffer, pointCount, &localError)) {
    if (error) *error = localError;
    return false;
  }

  copySharedBuffer<PackedFloat3>(vertBuffer, pointCount, outVerts);
  copySharedBuffer<simd_float4>(colorBuffer, pointCount, outColors);
  return true;
}

bool buildRasterSourceMesh(const RasterSourceRequest& request,
                           const void* sourceBytes,
                           size_t sourceByteCount,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0 || sourceBytes == nullptr || sourceByteCount == 0 ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 || request.sampleCountX <= 0) {
    if (error) *error = "Invalid Metal raster source request.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterSourcePipeline == nil || ctx.rasterOccupancyCountPipeline == nil) {
    if (error) *error = "Metal raster source pipeline unavailable.";
    return false;
  }

  RasterSourceUniforms uniforms{};
  fillRasterSourceUniforms(request, &uniforms);

  @autoreleasepool {
    id<MTLBuffer> sourceBuffer = [ctx.device newBufferWithBytes:sourceBytes
                                                         length:static_cast<NSUInteger>(sourceByteCount)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
    id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(ctx, &uniforms, 1u);
    constexpr NSUInteger kRasterOccupancyBinCount = 18u * 18u * 18u;
    id<MTLBuffer> occupancyBuffer =
        makeEmptySharedBuffer(kRasterOccupancyBinCount * sizeof(uint32_t));
    if (sourceBuffer == nil || vertBuffer == nil || colorBuffer == nil ||
        uniformBuffer == nil || occupancyBuffer == nil) {
      if (error) *error = "Failed to allocate Metal raster source buffers.";
      return false;
    }
    clearSharedBuffer(occupancyBuffer);

    if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
      id<MTLBuffer> visibleCountBuffer = makeEmptySharedBuffer(sizeof(uint32_t));
      if (visibleCountBuffer == nil) {
        if (error) *error = "Failed to allocate Metal raster occupancy counter.";
        return false;
      }
      clearSharedBuffer(visibleCountBuffer);

      id<MTLCommandBuffer> countCommand = [ctx.queue commandBuffer];
      if (countCommand == nil) {
        if (error) *error = "Failed to create Metal raster occupancy command buffer.";
        return false;
      }
      id<MTLComputeCommandEncoder> countEncoder = [countCommand computeCommandEncoder];
      if (countEncoder == nil) {
        if (error) *error = "Failed to create Metal raster occupancy encoder.";
        return false;
      }
      [countEncoder setComputePipelineState:ctx.rasterOccupancyCountPipeline];
      [countEncoder setBuffer:sourceBuffer offset:0 atIndex:0];
      [countEncoder setBuffer:sourceBuffer offset:0 atIndex:1];
      [countEncoder setBuffer:occupancyBuffer offset:0 atIndex:2];
      [countEncoder setBuffer:visibleCountBuffer offset:0 atIndex:3];
      [countEncoder setBuffer:uniformBuffer offset:0 atIndex:4];
      NSUInteger width = ctx.rasterOccupancyCountPipeline.maxTotalThreadsPerThreadgroup;
      if (width == 0) width = 64;
      width = std::min<NSUInteger>(width, 256);
      [countEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(std::max(uniforms.basePointCount, 0)), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [countEncoder endEncoding];
      [countCommand commit];
      [countCommand waitUntilCompleted];
      NSError* countError = countCommand.error;
      if (countError != nil) {
        if (error) *error = [[countError localizedDescription] UTF8String];
        return false;
      }

      uint32_t visibleCount = 0;
      std::memcpy(&visibleCount, [visibleCountBuffer contents], sizeof(visibleCount));
      const float meanOccupancy =
          static_cast<float>(visibleCount) / static_cast<float>(kRasterOccupancyBinCount);
      uniforms.occupancyTargetThreshold =
          std::max(0, static_cast<int>(std::ceil(meanOccupancy * 0.72f)));
      std::memcpy([uniformBuffer contents], &uniforms, sizeof(uniforms));
    }

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal raster source command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal raster source encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterSourcePipeline];
    [encoder setBuffer:sourceBuffer offset:0 atIndex:0];
    [encoder setBuffer:sourceBuffer offset:0 atIndex:1];
    [encoder setBuffer:vertBuffer offset:0 atIndex:2];
    [encoder setBuffer:colorBuffer offset:0 atIndex:3];
    [encoder setBuffer:occupancyBuffer offset:0 atIndex:4];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:5];
    NSUInteger width = ctx.rasterSourcePipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 256);
    [encoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    copySharedBuffer<PackedFloat3>(vertBuffer, pointCount, outVerts);
    copySharedBuffer<simd_float4>(colorBuffer, pointCount, outColors);
  }
  return true;
}
#endif

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildRasterSourceMeshFromIOSurface(const RasterSourceRequest& request,
                                        uint32_t surfaceId,
                                        int surfaceWidth,
                                        int surfaceHeight,
                                        int surfacePixelFormat,
                                        std::vector<float>* outVerts,
                                        std::vector<float>* outColors,
                                        std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger pointCount = static_cast<NSUInteger>(std::max(request.pointCount, 0));
  if (pointCount == 0 || surfaceId == 0 || surfaceWidth <= 0 || surfaceHeight <= 0 ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 || request.sampleCountX <= 0 ||
      surfaceWidth < request.sourceWidth || surfaceHeight < request.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface raster source request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface raster source format.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterSourceTexturePipeline == nil || ctx.rasterOccupancyTextureCountPipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-raster-source");
    return false;
  }

  RasterSourceUniforms uniforms{};
  fillRasterSourceUniforms(request, &uniforms);
  uniforms.pixelFormat = surfacePixelFormat;

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> vertBuffer = makeEmptySharedBuffer(pointCount * sizeof(PackedFloat3));
    id<MTLBuffer> colorBuffer = makeEmptySharedBuffer(pointCount * sizeof(simd_float4));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    constexpr NSUInteger kRasterOccupancyBinCount = 18u * 18u * 18u;
    id<MTLBuffer> occupancyBuffer =
        makeEmptySharedBuffer(kRasterOccupancyBinCount * sizeof(uint32_t));
    if (sourceTexture == nil || vertBuffer == nil || colorBuffer == nil ||
        uniformBuffer == nil || occupancyBuffer == nil) {
      if (error) *error = localError.empty() ? "Failed to allocate Metal IOSurface raster buffers." : localError;
      return false;
    }
    clearSharedBuffer(occupancyBuffer);

    if (request.occupancyFill != 0 && request.occupancyAppendCount > 0) {
      id<MTLBuffer> visibleCountBuffer = makeEmptySharedBuffer(sizeof(uint32_t));
      if (visibleCountBuffer == nil) {
        if (error) *error = "Failed to allocate Metal IOSurface raster occupancy counter.";
        return false;
      }
      clearSharedBuffer(visibleCountBuffer);

      id<MTLCommandBuffer> countCommand = [ctx.queue commandBuffer];
      if (countCommand == nil) {
        if (error) *error = "Failed to create Metal IOSurface raster occupancy command buffer.";
        return false;
      }
      id<MTLComputeCommandEncoder> countEncoder = [countCommand computeCommandEncoder];
      if (countEncoder == nil) {
        if (error) *error = "Failed to create Metal IOSurface raster occupancy encoder.";
        return false;
      }
      [countEncoder setComputePipelineState:ctx.rasterOccupancyTextureCountPipeline];
      [countEncoder setTexture:sourceTexture atIndex:0];
      [countEncoder setBuffer:occupancyBuffer offset:0 atIndex:0];
      [countEncoder setBuffer:visibleCountBuffer offset:0 atIndex:1];
      [countEncoder setBuffer:uniformBuffer offset:0 atIndex:2];
      NSUInteger width = ctx.rasterOccupancyTextureCountPipeline.maxTotalThreadsPerThreadgroup;
      if (width == 0) width = 64;
      width = std::min<NSUInteger>(width, 256);
      [countEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(std::max(uniforms.basePointCount, 0)), 1, 1)
               threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [countEncoder endEncoding];
      [countCommand commit];
      [countCommand waitUntilCompleted];
      NSError* countError = countCommand.error;
      if (countError != nil) {
        if (error) *error = [[countError localizedDescription] UTF8String];
        return false;
      }

      uint32_t visibleCount = 0;
      std::memcpy(&visibleCount, [visibleCountBuffer contents], sizeof(visibleCount));
      const float meanOccupancy =
          static_cast<float>(visibleCount) / static_cast<float>(kRasterOccupancyBinCount);
      uniforms.occupancyTargetThreshold =
          std::max(0, static_cast<int>(std::ceil(meanOccupancy * 0.72f)));
      std::memcpy([uniformBuffer contents], &uniforms, sizeof(uniforms));
    }

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal IOSurface raster command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface raster encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterSourceTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:vertBuffer offset:0 atIndex:0];
    [encoder setBuffer:colorBuffer offset:0 atIndex:1];
    [encoder setBuffer:occupancyBuffer offset:0 atIndex:2];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:3];
    NSUInteger width = ctx.rasterSourceTexturePipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 256);
    [encoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    copySharedBuffer<PackedFloat3>(vertBuffer, pointCount, outVerts);
    copySharedBuffer<simd_float4>(colorBuffer, pointCount, outColors);
  }
  return true;
}
#endif

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildInputSampledMesh(const InputSampleRequest& request,
                           const std::vector<float>& fullVerts,
                           const std::vector<float>& fullColors,
                           std::vector<float>* outVerts,
                           std::vector<float>* outColors,
                           std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const NSUInteger fullPointCount = static_cast<NSUInteger>(std::max(request.fullPointCount, 0));
  const NSUInteger visiblePointCount = static_cast<NSUInteger>(std::max(request.visiblePointCount, 0));
  if (fullPointCount == 0 || visiblePointCount == 0 ||
      fullVerts.size() < fullPointCount * 3u || fullColors.size() < fullPointCount * 4u) {
    if (error) *error = "Invalid Metal thinning source arrays.";
    return false;
  }
  @autoreleasepool {
    id<MTLBuffer> srcVertBuffer = makeSharedBuffer(reinterpret_cast<const PackedFloat3*>(fullVerts.data()), fullPointCount);
    id<MTLBuffer> srcColorBuffer = makeSharedBuffer(reinterpret_cast<const simd_float4*>(fullColors.data()), fullPointCount);
    id<MTLBuffer> dstVertBuffer = makeEmptySharedBuffer(visiblePointCount * sizeof(PackedFloat3));
    id<MTLBuffer> dstColorBuffer = makeEmptySharedBuffer(visiblePointCount * sizeof(simd_float4));
    InputSampleUniforms uniforms{};
    uniforms.fullPointCount = request.fullPointCount;
    uniforms.visiblePointCount = request.visiblePointCount;
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1);
    if (srcVertBuffer == nil || srcColorBuffer == nil || dstVertBuffer == nil || dstColorBuffer == nil || uniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal thinning buffers.";
      return false;
    }
    if (!runInputSampleCompute(srcVertBuffer, srcColorBuffer, dstVertBuffer, dstColorBuffer, uniformBuffer, visiblePointCount, &localError)) {
      if (error) *error = localError;
      return false;
    }
    copySharedBuffer<PackedFloat3>(dstVertBuffer, visiblePointCount, outVerts);
    copySharedBuffer<simd_float4>(dstColorBuffer, visiblePointCount, outColors);
  }
  return true;
}
#endif

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildScopeDensity(const ScopeDensityRequest& request,
                       const std::vector<float>& packedSamples,
                       bool allowReadback,
                       std::vector<float>* outDensity,
                       std::string* error) {
  std::string localError;
  if (!outDensity) {
    if (error) *error = "Missing Metal scope density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "Metal compact scope density readback disabled.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(request.pointCount, 0);
  const int width = std::max(request.width, 1);
  const int height = std::max(request.height, 1);
  const int channelCount = std::max(request.channelCount, 1);
  const size_t expectedSampleFloats = static_cast<size_t>(pointCount) * 5u;
  if (pointCount == 0 || packedSamples.size() < expectedSampleFloats) {
    if (error) *error = "Invalid Metal scope density sample buffer.";
    return false;
  }
  const size_t binCount = static_cast<size_t>(width) *
                          static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  if (binCount == 0) {
    if (error) *error = "Invalid Metal scope density dimensions.";
    return false;
  }

  MetalContext& ctx = context();
  @autoreleasepool {
    ScopeDensityUniforms uniforms{};
    uniforms.pointCount = pointCount;
    uniforms.waveform = request.waveform;
    uniforms.scopeMode = request.scopeMode;
    uniforms.width = width;
    uniforms.height = height;
    uniforms.rangeMin = request.rangeMin;
    uniforms.invRange = request.invRange;
    uniforms.excludeOverflow = request.excludeOverflow;
    uniforms.onlyOverflow = request.onlyOverflow;
    uniforms.channelCount = channelCount;
    uniforms.lumaMethod = std::clamp(request.lumaMethod, 0, 3);

    id<MTLBuffer> sampleBuffer = makeSharedBuffer(packedSamples.data(), expectedSampleFloats);
    id<MTLBuffer> densityBuffer = makeEmptySharedBuffer(static_cast<NSUInteger>(binCount * sizeof(uint32_t)));
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(&uniforms, 1u);
    if (sampleBuffer == nil || densityBuffer == nil || uniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal scope density buffers.";
      return false;
    }
    clearSharedBuffer(densityBuffer);
    if (!runComputeBuffers(ctx.scopeDensityPipeline,
                           std::array<id<MTLBuffer>, 3>{sampleBuffer, densityBuffer, uniformBuffer},
                           static_cast<NSUInteger>(pointCount),
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    outDensity->assign(binCount, 0.0f);
    const uint32_t* counts = reinterpret_cast<const uint32_t*>([densityBuffer contents]);
    for (size_t i = 0; i < binCount; ++i) {
      (*outDensity)[i] = static_cast<float>(counts[i]);
    }
  }
  return true;
}
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
                                    std::string* error) {
  std::string localError;
  if (!outDensity) {
    if (error) *error = "Missing Metal IOSurface scope density output.";
    return false;
  }
  outDensity->clear();
  if (!allowReadback) {
    if (error) *error = "Metal IOSurface compact scope density readback disabled.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(scopeRequest.width, 1);
  const int height = std::max(scopeRequest.height, 1);
  const int channelCount = std::max(scopeRequest.channelCount, 1);
  const size_t binCount = static_cast<size_t>(width) *
                          static_cast<size_t>(height) *
                          static_cast<size_t>(channelCount);
  if (pointCount <= 0 || binCount == 0u || surfaceId == 0 || surfaceWidth <= 0 ||
      surfaceHeight <= 0 || rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface scope density request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface scope density format.";
    return false;
  }

  MetalContext& ctx = context();
  if (ctx.rasterScopeDensityTexturePipeline == nil) {
    if (error) *error = residentPipelineUnavailableReason("metal-iosurface-scope-density");
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeDensityUniforms scopeUniforms{};
  scopeUniforms.pointCount = pointCount;
  scopeUniforms.waveform = scopeRequest.waveform;
  scopeUniforms.scopeMode = scopeRequest.scopeMode;
  scopeUniforms.width = width;
  scopeUniforms.height = height;
  scopeUniforms.rangeMin = scopeRequest.rangeMin;
  scopeUniforms.invRange = scopeRequest.invRange;
  scopeUniforms.excludeOverflow = scopeRequest.excludeOverflow != 0 ? 1 : 0;
  scopeUniforms.onlyOverflow = scopeRequest.onlyOverflow != 0 ? 1 : 0;
  scopeUniforms.channelCount = channelCount;
  scopeUniforms.lumaMethod = std::clamp(scopeRequest.lumaMethod, 0, 3);

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> densityBuffer =
        makeEmptySharedBuffer(static_cast<NSUInteger>(binCount * sizeof(uint32_t)));
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    id<MTLBuffer> scopeUniformBuffer = makeSharedBuffer(&scopeUniforms, 1u);
    if (sourceTexture == nil || densityBuffer == nil || rasterUniformBuffer == nil ||
        scopeUniformBuffer == nil) {
      if (error) {
        *error = localError.empty() ? "Failed to allocate Metal IOSurface scope density resources."
                                    : localError;
      }
      return false;
    }
    clearSharedBuffer(densityBuffer);

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope density command buffer.";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope density encoder.";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterScopeDensityTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:densityBuffer offset:0 atIndex:0];
    [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
    [encoder setBuffer:scopeUniformBuffer offset:0 atIndex:2];
    NSUInteger threads = ctx.rasterScopeDensityTexturePipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0) threads = 64;
    threads = std::min<NSUInteger>(threads, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    outDensity->assign(binCount, 0.0f);
    const uint32_t* counts = reinterpret_cast<const uint32_t*>([densityBuffer contents]);
    for (size_t i = 0; i < binCount; ++i) {
      (*outDensity)[i] = static_cast<float>(counts[i]);
    }
  }
  return true;
}
#endif

static bool encodeScopeRangeOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> sourceTexture,
    id<MTLBuffer> rasterUniformBuffer,
    int pointCount,
    const ScopeRangeRequest& rangeRequest,
    id<MTLBuffer>* outFinalRangeBuffer,
    std::string* error) {
  if (outFinalRangeBuffer) *outFinalRangeBuffer = nil;
  if (error) error->clear();
  if (commandBuffer == nil || sourceTexture == nil || rasterUniformBuffer == nil ||
      pointCount <= 0 || outFinalRangeBuffer == nullptr) {
    if (error) *error = "invalid-metal-scope-range-encode-request";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               error) ||
      contextPointer == nullptr) {
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.rasterScopeRangeTexturePipeline == nil ||
      ctx.rasterScopeRangeHistogramTexturePipeline == nil ||
      ctx.scopeRangeHistogramPercentilePipeline == nil ||
      ctx.scopeRangeFinalizePipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(
          ctx, "metal-iosurface-scope-range");
    }
    return false;
  }

  constexpr uint32_t kRangeHistogramBins = 2048u;
  const auto orderedUintFromFloatHost = [](float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
  };

  ScopeRangeUniforms rangeUniforms{};
  rangeUniforms.pointCount = pointCount;
  rangeUniforms.waveform = rangeRequest.waveform;
  rangeUniforms.scopeMode = rangeRequest.scopeMode;
  rangeUniforms.includeRed = rangeRequest.includeRed != 0 ? 1 : 0;
  rangeUniforms.includeGreen = rangeRequest.includeGreen != 0 ? 1 : 0;
  rangeUniforms.includeBlue = rangeRequest.includeBlue != 0 ? 1 : 0;
  rangeUniforms.includeLuma = rangeRequest.includeLuma != 0 ? 1 : 0;
  rangeUniforms.includeOverflow = rangeRequest.includeOverflow != 0 ? 1 : 0;
  rangeUniforms.lumaMethod = std::clamp(rangeRequest.lumaMethod, 0, 3);
  rangeUniforms.previousRangeValid = rangeRequest.previousRangeValid != 0 ? 1 : 0;
  rangeUniforms.previousRangeMin = rangeRequest.previousRangeMin;
  rangeUniforms.previousRangeMax = rangeRequest.previousRangeMax;
  rangeUniforms.histogramBinCount = static_cast<int>(kRangeHistogramBins);

  const uint32_t initRangeBits[3] = {
      orderedUintFromFloatHost(std::numeric_limits<float>::infinity()),
      orderedUintFromFloatHost(-std::numeric_limits<float>::infinity()),
      0u,
  };
  const uint32_t initPercentiles[2] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
  };
  const uint32_t initFinalRange[3] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
      0u,
  };

  id<MTLBuffer> rangeBitsBuffer =
      makeSharedBuffer(ctx, initRangeBits, 3u);
  id<MTLBuffer> histogramBuffer =
      makeEmptySharedBuffer(
          ctx,
          static_cast<NSUInteger>(kRangeHistogramBins * sizeof(uint32_t)));
  id<MTLBuffer> percentileBuffer =
      makeSharedBuffer(ctx, initPercentiles, 2u);
  id<MTLBuffer> finalRangeBuffer =
      makeSharedBuffer(ctx, initFinalRange, 3u);
  id<MTLBuffer> rangeUniformBuffer =
      makeSharedBuffer(ctx, &rangeUniforms, 1u);
  if (rangeBitsBuffer == nil || histogramBuffer == nil || percentileBuffer == nil ||
      finalRangeBuffer == nil || rangeUniformBuffer == nil) {
    if (error) *error = "metal-scope-range-resource-allocation-failed";
    return false;
  }
  clearSharedBuffer(histogramBuffer);

  const auto dispatchPointKernel = [&](id<MTLComputePipelineState> pipeline,
                                       void (^bind)(id<MTLComputeCommandEncoder>)) -> bool {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) return false;
    [encoder setComputePipelineState:pipeline];
    bind(encoder);
    NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0) threads = 64;
    threads = std::min<NSUInteger>(threads, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    return true;
  };
  if (!dispatchPointKernel(ctx.rasterScopeRangeTexturePipeline,
                           ^(id<MTLComputeCommandEncoder> encoder) {
                             [encoder setTexture:sourceTexture atIndex:0];
                             [encoder setBuffer:rangeBitsBuffer offset:0 atIndex:0];
                             [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
                             [encoder setBuffer:rangeUniformBuffer offset:0 atIndex:2];
                           })) {
    if (error) *error = "metal-scope-range-encoder-failed";
    return false;
  }
  if (!dispatchPointKernel(ctx.rasterScopeRangeHistogramTexturePipeline,
                           ^(id<MTLComputeCommandEncoder> encoder) {
                             [encoder setTexture:sourceTexture atIndex:0];
                             [encoder setBuffer:histogramBuffer offset:0 atIndex:0];
                             [encoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
                             [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:2];
                             [encoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
                           })) {
    if (error) *error = "metal-scope-range-histogram-encoder-failed";
    return false;
  }

  id<MTLComputeCommandEncoder> percentileEncoder = [commandBuffer computeCommandEncoder];
  if (percentileEncoder == nil) {
    if (error) *error = "metal-scope-range-percentile-encoder-failed";
    return false;
  }
  [percentileEncoder setComputePipelineState:ctx.scopeRangeHistogramPercentilePipeline];
  [percentileEncoder setBuffer:histogramBuffer offset:0 atIndex:0];
  [percentileEncoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
  [percentileEncoder setBuffer:percentileBuffer offset:0 atIndex:2];
  [percentileEncoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
  [percentileEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [percentileEncoder endEncoding];

  id<MTLComputeCommandEncoder> finalizeEncoder = [commandBuffer computeCommandEncoder];
  if (finalizeEncoder == nil) {
    if (error) *error = "metal-scope-range-finalize-encoder-failed";
    return false;
  }
  [finalizeEncoder setComputePipelineState:ctx.scopeRangeFinalizePipeline];
  [finalizeEncoder setBuffer:percentileBuffer offset:0 atIndex:0];
  [finalizeEncoder setBuffer:rangeBitsBuffer offset:0 atIndex:1];
  [finalizeEncoder setBuffer:finalRangeBuffer offset:0 atIndex:2];
  [finalizeEncoder setBuffer:rangeUniformBuffer offset:0 atIndex:3];
  [finalizeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [finalizeEncoder endEncoding];

  *outFinalRangeBuffer = finalRangeBuffer;
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildScopeRangeFromIOSurface(const RasterSourceRequest& rasterRequest,
                                  const ScopeRangeRequest& rangeRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  ScopeRangeResult* outRange,
                                  std::string* error) {
  std::string localError;
  if (!outRange) {
    if (error) *error = "Missing Metal IOSurface scope range output.";
    return false;
  }
  *outRange = ScopeRangeResult{};
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  if (pointCount <= 0 || surfaceId == 0 || surfaceWidth <= 0 || surfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight) {
    if (error) *error = "Invalid Metal IOSurface scope range request.";
    return false;
  }
  if (surfacePixelFormat != 0 && surfacePixelFormat != 1) {
    if (error) *error = "Unsupported Metal IOSurface scope range format.";
    return false;
  }

  MetalContext& ctx = context();
  const auto orderedUintFromFloatHost = [](float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
  };
  const auto floatFromOrderedUintHost = [](uint32_t value) {
    uint32_t bits = (value & 0x80000000u) != 0u ? (value ^ 0x80000000u) : ~value;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  };

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;
  const uint32_t initFinalRange[3] = {
      orderedUintFromFloatHost(0.0f),
      orderedUintFromFloatHost(1.0f),
      0u,
  };

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        makeTextureFromIOSurface(ctx, surfaceId, surfaceWidth, surfaceHeight, surfacePixelFormat, &localError);
    id<MTLBuffer> rasterUniformBuffer = makeSharedBuffer(&rasterUniforms, 1u);
    if (sourceTexture == nil || rasterUniformBuffer == nil) {
      if (error) {
        *error = localError.empty() ? "Failed to allocate Metal IOSurface scope range resources."
                                    : localError;
      }
      return false;
    }

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    if (commandBuffer == nil) {
      if (error) *error = "Failed to create Metal IOSurface scope range command buffer.";
      return false;
    }
    id<MTLBuffer> finalRangeBuffer = nil;
    if (!encodeScopeRangeOnCommandBuffer(commandBuffer,
                                         sourceTexture,
                                         rasterUniformBuffer,
                                         pointCount,
                                         rangeRequest,
                                         &finalRangeBuffer,
                                         &localError)) {
      if (error) {
        *error = localError.empty() ? "Failed to encode Metal IOSurface scope range."
                                    : localError;
      }
      return false;
    }

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    NSError* cbError = commandBuffer.error;
    if (cbError != nil) {
      if (error) *error = [[cbError localizedDescription] UTF8String];
      return false;
    }

    uint32_t packed[3] = {initFinalRange[0], initFinalRange[1], initFinalRange[2]};
    std::memcpy(packed, [finalRangeBuffer contents], sizeof(packed));
    outRange->minValue = floatFromOrderedUintHost(packed[0]);
    outRange->maxValue = floatFromOrderedUintHost(packed[1]);
    outRange->validCount = packed[2];
  }
  return true;
}
#endif

static bool encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> importedSourceTexture,
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
    std::string* error,
    const ScopeDerivedResidentRecord* residentRecord = nullptr,
    ScopeDerivedResidentRecord* outEncodedRecord = nullptr,
    uint64_t buildSerial = 0) {
  if (error) error->clear();
  std::string localError;
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int binCount = std::max(histogramRequest.width, 1);
  const int channelCount = histogramRequest.scopeMode == 1 ? 1 : 3;
  const size_t densityCount = static_cast<size_t>(binCount) * static_cast<size_t>(channelCount);
  const bool useResidentRecord = residentRecord != nullptr;
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
  if (pointCount <= 0 || (!useResidentRecord && importedSourceTexture == nil) ||
#else
  if (pointCount <= 0 ||
      (!useResidentRecord && importedSourceTexture == nil && sourceSurfaceId == 0) ||
#endif
      outputSurfaceId == 0 ||
      sourceSurfaceWidth <= 0 || sourceSurfaceHeight <= 0 ||
      outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || sourceSurfaceWidth < rasterRequest.sourceWidth ||
      sourceSurfaceHeight < rasterRequest.sourceHeight || densityCount == 0u) {
    if (error) *error = "invalid-metal-histogram-surface-request";
    return false;
  }
  if ((sourceSurfacePixelFormat != 0 && sourceSurfacePixelFormat != 1) ||
      (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1)) {
    if (error) *error = "unsupported-metal-histogram-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty() ? "metal-histogram-command-buffer-invalid"
                                  : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.histogramSurfaceRenderPipeline == nil ||
      (histogramRequest.useGpuAutoRange != 0 &&
       ctx.histogramApplyRangePipeline == nil) ||
      (!useResidentRecord &&
       (ctx.rasterScopeDensityTexturePipeline == nil ||
        ctx.histogramMaxPipeline == nil))) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "metal-histogram-surface");
    }
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = sourceSurfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeDensityUniforms densityUniforms{};
  densityUniforms.pointCount = pointCount;
  densityUniforms.waveform = 0;
  densityUniforms.scopeMode = histogramRequest.scopeMode;
  densityUniforms.width = binCount;
  densityUniforms.height = 1;
  densityUniforms.rangeMin = histogramRequest.rangeMin;
  densityUniforms.invRange = histogramRequest.invRange;
  densityUniforms.excludeOverflow = 1;
  densityUniforms.onlyOverflow = 0;
  densityUniforms.channelCount = channelCount;
  densityUniforms.lumaMethod = std::clamp(histogramRequest.lumaMethod, 0, 3);
  ScopeDensityUniforms overflowUniforms = densityUniforms;
  overflowUniforms.excludeOverflow = 0;
  overflowUniforms.onlyOverflow = 1;

  HistogramSurfaceUniforms surfaceUniforms{};
  surfaceUniforms.pointCount = pointCount;
  surfaceUniforms.scopeMode = histogramRequest.scopeMode;
  surfaceUniforms.width = binCount;
  surfaceUniforms.height = std::max(histogramRequest.height, 1);
  surfaceUniforms.rangeMin = histogramRequest.rangeMin;
  surfaceUniforms.invRange = histogramRequest.invRange;
  surfaceUniforms.showOverflow = histogramRequest.showOverflow != 0 ? 1 : 0;
  surfaceUniforms.highlightOverflow = histogramRequest.highlightOverflow != 0 ? 1 : 0;
  surfaceUniforms.lumaMethod = std::clamp(histogramRequest.lumaMethod, 0, 3);
  surfaceUniforms.channelCount = channelCount;

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-histogram-output-surface-missing";
    return false;
  }

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        importedSourceTexture;
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    if (!useResidentRecord && sourceTexture == nil) {
      sourceTexture = makeTextureFromIOSurface(ctx,
                                               sourceSurfaceId,
                                               sourceSurfaceWidth,
                                               sourceSurfaceHeight,
                                               sourceSurfacePixelFormat,
                                               &localError);
    }
#endif
    id<MTLBuffer> densityBuffer =
        useResidentRecord
            ? residentRecord->density
            : makeEmptySharedBuffer(
                  ctx,
                  static_cast<NSUInteger>(densityCount * sizeof(uint32_t)));
    id<MTLBuffer> overflowDensityBuffer =
        useResidentRecord
            ? residentRecord->overflowDensity
            : (surfaceUniforms.showOverflow != 0
            ? makeEmptySharedBuffer(
                  ctx,
                  static_cast<NSUInteger>(densityCount * sizeof(uint32_t)))
            : nil);
    id<MTLBuffer> maxDensityBuffer =
        useResidentRecord ? residentRecord->maxDensity
                          : makeEmptySharedBuffer(ctx, sizeof(uint32_t));
    id<MTLBuffer> rasterUniformBuffer =
        useResidentRecord ? nil : makeSharedBuffer(ctx, &rasterUniforms, 1u);
    id<MTLBuffer> densityUniformBuffer =
        (!useResidentRecord || histogramRequest.useGpuAutoRange != 0)
            ? makeSharedBuffer(ctx, &densityUniforms, 1u)
            : nil;
    id<MTLBuffer> overflowUniformBuffer =
        (!useResidentRecord || histogramRequest.useGpuAutoRange != 0)
            ? makeSharedBuffer(ctx, &overflowUniforms, 1u)
            : nil;
    id<MTLBuffer> surfaceUniformBuffer =
        makeSharedBuffer(ctx, &surfaceUniforms, 1u);
    if ((!useResidentRecord &&
         (sourceTexture == nil || rasterUniformBuffer == nil)) ||
        (histogramRequest.useGpuAutoRange != 0 &&
         (densityUniformBuffer == nil || overflowUniformBuffer == nil)) ||
        densityBuffer == nil || maxDensityBuffer == nil ||
        surfaceUniformBuffer == nil ||
        (useResidentRecord && histogramRequest.useGpuAutoRange != 0 &&
         residentRecord->finalRange == nil) ||
        (surfaceUniforms.showOverflow != 0 && overflowDensityBuffer == nil)) {
      if (error) {
        *error = localError.empty() ? "metal-histogram-surface-allocation-failed" : localError;
      }
      return false;
    }
    if (!useResidentRecord) {
      clearSharedBuffer(densityBuffer);
      clearSharedBuffer(maxDensityBuffer);
      if (overflowDensityBuffer != nil) clearSharedBuffer(overflowDensityBuffer);
    }

    if (commandBuffer == nil) {
      if (error) *error = "metal-histogram-surface-command-buffer-failed";
      return false;
    }
    id<MTLBuffer> finalRangeBuffer =
        useResidentRecord ? residentRecord->finalRange : nil;
    if (histogramRequest.useGpuAutoRange != 0) {
      if (!useResidentRecord) {
        ScopeRangeRequest autoRangeRequest = histogramRequest.autoRange;
        autoRangeRequest.pointCount = pointCount;
        autoRangeRequest.waveform = 0;
        autoRangeRequest.scopeMode = histogramRequest.scopeMode;
        autoRangeRequest.includeOverflow = histogramRequest.showOverflow != 0 ? 1 : 0;
        autoRangeRequest.lumaMethod = std::clamp(histogramRequest.lumaMethod, 0, 3);
        if (!encodeScopeRangeOnCommandBuffer(commandBuffer,
                                             sourceTexture,
                                             rasterUniformBuffer,
                                             pointCount,
                                             autoRangeRequest,
                                             &finalRangeBuffer,
                                             &localError)) {
          if (error) {
            *error = localError.empty()
                         ? "metal-histogram-auto-range-encode-failed"
                         : localError;
          }
          return false;
        }
      }
      id<MTLComputeCommandEncoder> applyRangeEncoder = [commandBuffer computeCommandEncoder];
      if (applyRangeEncoder == nil) {
        if (error) *error = "metal-histogram-apply-range-encoder-failed";
        return false;
      }
      [applyRangeEncoder setComputePipelineState:ctx.histogramApplyRangePipeline];
      [applyRangeEncoder setBuffer:finalRangeBuffer offset:0 atIndex:0];
      [applyRangeEncoder setBuffer:densityUniformBuffer offset:0 atIndex:1];
      [applyRangeEncoder setBuffer:overflowUniformBuffer offset:0 atIndex:2];
      [applyRangeEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:3];
      [applyRangeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [applyRangeEncoder endEncoding];
    }
    auto dispatchPointKernel = [&](id<MTLComputePipelineState> pipeline,
                                   id<MTLBuffer> targetDensity,
                                   id<MTLBuffer> uniformsBuffer) -> bool {
      if (uniformsBuffer == nil) return false;
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) return false;
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:sourceTexture atIndex:0];
      [encoder setBuffer:targetDensity offset:0 atIndex:0];
      [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
      [encoder setBuffer:uniformsBuffer offset:0 atIndex:2];
      NSUInteger threads = pipeline.maxTotalThreadsPerThreadgroup;
      if (threads == 0) threads = 64;
      threads = std::min<NSUInteger>(threads, 256);
      [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
      return true;
    };

    if (!useResidentRecord) {
      if (!dispatchPointKernel(ctx.rasterScopeDensityTexturePipeline,
                               densityBuffer,
                               densityUniformBuffer)) {
        if (error) *error = "metal-histogram-density-encoder-failed";
        return false;
      }
      if (overflowDensityBuffer != nil) {
        if (!dispatchPointKernel(ctx.rasterScopeDensityTexturePipeline,
                                 overflowDensityBuffer,
                                 overflowUniformBuffer)) {
          if (error) *error = "metal-histogram-overflow-density-encoder-failed";
          return false;
        }
      }

      id<MTLComputeCommandEncoder> maxEncoder = [commandBuffer computeCommandEncoder];
      if (maxEncoder == nil) {
        if (error) *error = "metal-histogram-max-encoder-failed";
        return false;
      }
      [maxEncoder setComputePipelineState:ctx.histogramMaxPipeline];
      [maxEncoder setBuffer:densityBuffer offset:0 atIndex:0];
      [maxEncoder setBuffer:maxDensityBuffer offset:0 atIndex:1];
      [maxEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:2];
      NSUInteger maxThreads = ctx.histogramMaxPipeline.maxTotalThreadsPerThreadgroup;
      if (maxThreads == 0) maxThreads = 64;
      maxThreads = std::min<NSUInteger>(maxThreads, 256);
      [maxEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(densityCount), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(maxThreads, 1, 1)];
      [maxEncoder endEncoding];

      if (outEncodedRecord != nullptr) {
        if (buildSerial == 0) {
          if (error) *error = "metal-histogram-cache-build-serial-invalid";
          return false;
        }
        ScopeDerivedResidentRecord record{};
        record.family = ScopeDerivedFamily::Histogram;
        record.builtSerial = buildSerial;
        record.density = densityBuffer;
        record.overflowDensity = overflowDensityBuffer;
        record.maxDensity = maxDensityBuffer;
        record.finalRange = finalRangeBuffer;
        record.byteSize = static_cast<size_t>([densityBuffer length]) +
                          static_cast<size_t>([maxDensityBuffer length]) +
                          (overflowDensityBuffer != nil
                               ? static_cast<size_t>([overflowDensityBuffer length])
                               : 0u) +
                          (finalRangeBuffer != nil
                               ? static_cast<size_t>([finalRangeBuffer length])
                               : 0u);
        *outEncodedRecord = std::move(record);
      }
    }

    id<MTLComputeCommandEncoder> renderEncoder = [commandBuffer computeCommandEncoder];
    if (renderEncoder == nil) {
      if (error) *error = "metal-histogram-render-encoder-failed";
      return false;
    }
    [renderEncoder setComputePipelineState:ctx.histogramSurfaceRenderPipeline];
    [renderEncoder setTexture:outputTexture atIndex:0];
    [renderEncoder setBuffer:densityBuffer offset:0 atIndex:0];
    [renderEncoder setBuffer:overflowDensityBuffer offset:0 atIndex:1];
    [renderEncoder setBuffer:maxDensityBuffer offset:0 atIndex:2];
    [renderEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:3];
    NSUInteger groupWidth = ctx.histogramSurfaceRenderPipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.histogramSurfaceRenderPipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    [renderEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                               static_cast<NSUInteger>(outputSurfaceHeight),
                                               1)
              threadsPerThreadgroup:MTLSizeMake(groupWidth, groupHeight, 1)];
    [renderEncoder endEncoding];

  }
  return true;
}

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
                                         std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "metal-histogram-surface-command-buffer-failed";
    return false;
  }
  if (!encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          nil,
          rasterRequest,
          histogramRequest,
          sourceSurfaceId,
          sourceSurfaceWidth,
          sourceSurfaceHeight,
          sourceSurfacePixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}

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
                                         std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  return encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      nil,
      rasterRequest,
      histogramRequest,
      sourceSurfaceId,
      sourceSurfaceWidth,
      sourceSurfaceHeight,
      sourceSurfacePixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}
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
    std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  return encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      source->texture,
      rasterRequest,
      histogramRequest,
      0,
      source->descriptor.width,
      source->descriptor.height,
      source->descriptor.pixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}

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
    std::string* error) {
  if (error) error->clear();
  if (!cache || buildSerial == 0) {
    if (error) *error = "invalid-metal-histogram-scope-cache-request";
    return false;
  }
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache->cacheId, cache->ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error) ||
      !validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }

  ScopeDerivedResidentRecord residentRecord{};
  std::string residentError;
  if (resolveScopeDerivedRecordForSubmission(
          submission, *cache, true, &residentRecord, &residentError)) {
    if (residentRecord.family != ScopeDerivedFamily::Histogram) {
      if (error) *error = "metal-histogram-scope-cache-family-mismatch";
      return false;
    }
    return encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
        commandBuffer,
        nil,
        rasterRequest,
        histogramRequest,
        0,
        rasterRequest.sourceWidth,
        rasterRequest.sourceHeight,
        rasterRequest.pixelFormat,
        outputSurfaceId,
        outputSurfaceWidth,
        outputSurfaceHeight,
        outputSurfacePixelFormat,
        error,
        &residentRecord,
        nullptr,
        buildSerial);
  }
  if (!residentError.empty()) {
    if (error) *error = residentError;
    return false;
  }

  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  ScopeDerivedResidentRecord encodedRecord{};
  if (!encodeHistogramSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          source->texture,
          rasterRequest,
          histogramRequest,
          0,
          source->descriptor.width,
          source->descriptor.height,
          source->descriptor.pixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error,
          nullptr,
          &encodedRecord,
          buildSerial)) {
    return false;
  }
  return registerPendingScopeDerivedRecord(
      submission, cache, std::move(encodedRecord), error);
}

static bool encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> importedSourceTexture,
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
    std::string* error,
    const ScopeDerivedResidentRecord* residentRecord = nullptr,
    ScopeDerivedResidentRecord* outEncodedRecord = nullptr,
    uint64_t buildSerial = 0) {
  if (error) error->clear();
  std::string localError;
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int width = std::max(waveformRequest.width, 1);
  const int height = std::max(waveformRequest.height, 1);
  const bool lumaOnly = waveformRequest.scopeMode == 2;
  const bool paradeLuma = waveformRequest.scopeMode == 1 && waveformRequest.includeLuma != 0;
  const int channelCount = lumaOnly ? 1 : (paradeLuma ? 4 : 3);
  const size_t densityCount = static_cast<size_t>(width) *
                              static_cast<size_t>(height) *
                              static_cast<size_t>(channelCount);
  const bool useResidentRecord = residentRecord != nullptr;
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
  if (pointCount <= 0 || (!useResidentRecord && importedSourceTexture == nil) ||
#else
  if (pointCount <= 0 ||
      (!useResidentRecord && importedSourceTexture == nil && sourceSurfaceId == 0) ||
#endif
      outputSurfaceId == 0 ||
      sourceSurfaceWidth <= 0 || sourceSurfaceHeight <= 0 ||
      outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || sourceSurfaceWidth < rasterRequest.sourceWidth ||
      sourceSurfaceHeight < rasterRequest.sourceHeight || densityCount == 0u) {
    if (error) *error = "invalid-metal-waveform-surface-request";
    return false;
  }
  if ((sourceSurfacePixelFormat != 0 && sourceSurfacePixelFormat != 1) ||
      (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1)) {
    if (error) *error = "unsupported-metal-waveform-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty() ? "metal-waveform-command-buffer-invalid"
                                  : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.waveformSurfaceRenderPipeline == nil ||
      (waveformRequest.useGpuAutoRange != 0 &&
       ctx.waveformApplyRangePipeline == nil) ||
      (!useResidentRecord &&
       (ctx.rasterScopeDensityTexturePipeline == nil ||
        ctx.waveformMaxPipeline == nil))) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "metal-waveform-surface");
    }
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = sourceSurfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  ScopeDensityUniforms densityUniforms{};
  densityUniforms.pointCount = pointCount;
  densityUniforms.waveform = 1;
  densityUniforms.scopeMode = waveformRequest.scopeMode;
  densityUniforms.width = width;
  densityUniforms.height = height;
  densityUniforms.rangeMin = waveformRequest.rangeMin;
  densityUniforms.invRange = waveformRequest.invRange;
  densityUniforms.excludeOverflow = 1;
  densityUniforms.onlyOverflow = 0;
  densityUniforms.channelCount = channelCount;
  densityUniforms.lumaMethod = std::clamp(waveformRequest.lumaMethod, 0, 3);
  ScopeDensityUniforms overflowUniforms = densityUniforms;
  overflowUniforms.excludeOverflow = 0;
  overflowUniforms.onlyOverflow = 1;

  WaveformSurfaceUniforms surfaceUniforms{};
  surfaceUniforms.pointCount = pointCount;
  surfaceUniforms.scopeMode = waveformRequest.scopeMode;
  surfaceUniforms.width = width;
  surfaceUniforms.height = height;
  surfaceUniforms.rangeMin = waveformRequest.rangeMin;
  surfaceUniforms.invRange = waveformRequest.invRange;
  surfaceUniforms.showOverflow = waveformRequest.showOverflow != 0 ? 1 : 0;
  surfaceUniforms.highlightOverflow = waveformRequest.highlightOverflow != 0 ? 1 : 0;
  surfaceUniforms.lumaMethod = std::clamp(waveformRequest.lumaMethod, 0, 3);
  surfaceUniforms.channelCount = channelCount;
  surfaceUniforms.includeRed = waveformRequest.includeRed != 0 ? 1 : 0;
  surfaceUniforms.includeGreen = waveformRequest.includeGreen != 0 ? 1 : 0;
  surfaceUniforms.includeBlue = waveformRequest.includeBlue != 0 ? 1 : 0;
  surfaceUniforms.includeLuma = waveformRequest.includeLuma != 0 ? 1 : 0;
  surfaceUniforms.pointBrightness = std::max(0.0f, waveformRequest.pointBrightness);
  surfaceUniforms.colorSaturation = std::max(0.0f, waveformRequest.colorSaturation);
  surfaceUniforms.coverageAlpha = std::clamp(waveformRequest.coverageAlpha, 0.0f, 1.0f);

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-waveform-output-surface-missing";
    return false;
  }

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        importedSourceTexture;
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    if (!useResidentRecord && sourceTexture == nil) {
      sourceTexture = makeTextureFromIOSurface(ctx,
                                               sourceSurfaceId,
                                               sourceSurfaceWidth,
                                               sourceSurfaceHeight,
                                               sourceSurfacePixelFormat,
                                               &localError);
    }
#endif
    id<MTLBuffer> densityBuffer =
        useResidentRecord
            ? residentRecord->density
            : makeEmptySharedBuffer(
                  ctx,
                  static_cast<NSUInteger>(densityCount * sizeof(uint32_t)));
    id<MTLBuffer> overflowDensityBuffer =
        useResidentRecord
            ? residentRecord->overflowDensity
            : (surfaceUniforms.showOverflow != 0
                   ? makeEmptySharedBuffer(
                         ctx,
                         static_cast<NSUInteger>(densityCount * sizeof(uint32_t)))
                   : nil);
    id<MTLBuffer> maxDensityBuffer =
        useResidentRecord ? residentRecord->maxDensity
                          : makeEmptySharedBuffer(ctx, sizeof(uint32_t));
    id<MTLBuffer> rasterUniformBuffer =
        useResidentRecord ? nil : makeSharedBuffer(ctx, &rasterUniforms, 1u);
    id<MTLBuffer> densityUniformBuffer =
        (!useResidentRecord || waveformRequest.useGpuAutoRange != 0)
            ? makeSharedBuffer(ctx, &densityUniforms, 1u)
            : nil;
    id<MTLBuffer> overflowUniformBuffer =
        (!useResidentRecord || waveformRequest.useGpuAutoRange != 0)
            ? makeSharedBuffer(ctx, &overflowUniforms, 1u)
            : nil;
    id<MTLBuffer> surfaceUniformBuffer =
        makeSharedBuffer(ctx, &surfaceUniforms, 1u);
    if ((!useResidentRecord &&
         (sourceTexture == nil || rasterUniformBuffer == nil)) ||
        (waveformRequest.useGpuAutoRange != 0 &&
         (densityUniformBuffer == nil || overflowUniformBuffer == nil)) ||
        densityBuffer == nil || maxDensityBuffer == nil ||
        surfaceUniformBuffer == nil ||
        (useResidentRecord && waveformRequest.useGpuAutoRange != 0 &&
         residentRecord->finalRange == nil) ||
        (surfaceUniforms.showOverflow != 0 && overflowDensityBuffer == nil)) {
      if (error) {
        *error = localError.empty() ? "metal-waveform-surface-allocation-failed" : localError;
      }
      return false;
    }
    if (!useResidentRecord) {
      clearSharedBuffer(densityBuffer);
      clearSharedBuffer(maxDensityBuffer);
      if (overflowDensityBuffer != nil) clearSharedBuffer(overflowDensityBuffer);
    }

    if (commandBuffer == nil) {
      if (error) *error = "metal-waveform-surface-command-buffer-failed";
      return false;
    }
    id<MTLBuffer> finalRangeBuffer =
        useResidentRecord ? residentRecord->finalRange : nil;
    if (waveformRequest.useGpuAutoRange != 0) {
      if (!useResidentRecord) {
        ScopeRangeRequest autoRangeRequest = waveformRequest.autoRange;
        autoRangeRequest.pointCount = pointCount;
        autoRangeRequest.waveform = 1;
        autoRangeRequest.scopeMode = waveformRequest.scopeMode;
        autoRangeRequest.includeOverflow = waveformRequest.showOverflow != 0 ? 1 : 0;
        autoRangeRequest.lumaMethod = std::clamp(waveformRequest.lumaMethod, 0, 3);
        if (!encodeScopeRangeOnCommandBuffer(commandBuffer,
                                             sourceTexture,
                                             rasterUniformBuffer,
                                             pointCount,
                                             autoRangeRequest,
                                             &finalRangeBuffer,
                                             &localError)) {
          if (error) {
            *error = localError.empty()
                         ? "metal-waveform-auto-range-encode-failed"
                         : localError;
          }
          return false;
        }
      }
      id<MTLComputeCommandEncoder> applyRangeEncoder = [commandBuffer computeCommandEncoder];
      if (applyRangeEncoder == nil) {
        if (error) *error = "metal-waveform-apply-range-encoder-failed";
        return false;
      }
      [applyRangeEncoder setComputePipelineState:ctx.waveformApplyRangePipeline];
      [applyRangeEncoder setBuffer:finalRangeBuffer offset:0 atIndex:0];
      [applyRangeEncoder setBuffer:densityUniformBuffer offset:0 atIndex:1];
      [applyRangeEncoder setBuffer:overflowUniformBuffer offset:0 atIndex:2];
      [applyRangeEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:3];
      [applyRangeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [applyRangeEncoder endEncoding];
    }
    auto dispatchPointKernel = [&](id<MTLBuffer> targetDensity,
                                   id<MTLBuffer> uniformsBuffer) -> bool {
      if (uniformsBuffer == nil) return false;
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) return false;
      [encoder setComputePipelineState:ctx.rasterScopeDensityTexturePipeline];
      [encoder setTexture:sourceTexture atIndex:0];
      [encoder setBuffer:targetDensity offset:0 atIndex:0];
      [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:1];
      [encoder setBuffer:uniformsBuffer offset:0 atIndex:2];
      NSUInteger threads = ctx.rasterScopeDensityTexturePipeline.maxTotalThreadsPerThreadgroup;
      if (threads == 0) threads = 64;
      threads = std::min<NSUInteger>(threads, 256);
      [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
         threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
      return true;
    };

    if (!useResidentRecord) {
      if (!dispatchPointKernel(densityBuffer, densityUniformBuffer)) {
        if (error) *error = "metal-waveform-density-encoder-failed";
        return false;
      }
      if (overflowDensityBuffer != nil) {
        if (!dispatchPointKernel(overflowDensityBuffer, overflowUniformBuffer)) {
          if (error) *error = "metal-waveform-overflow-density-encoder-failed";
          return false;
        }
      }

      id<MTLComputeCommandEncoder> maxEncoder = [commandBuffer computeCommandEncoder];
      if (maxEncoder == nil) {
        if (error) *error = "metal-waveform-max-encoder-failed";
        return false;
      }
      [maxEncoder setComputePipelineState:ctx.waveformMaxPipeline];
      [maxEncoder setBuffer:densityBuffer offset:0 atIndex:0];
      [maxEncoder setBuffer:maxDensityBuffer offset:0 atIndex:1];
      [maxEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:2];
      NSUInteger maxThreads = ctx.waveformMaxPipeline.maxTotalThreadsPerThreadgroup;
      if (maxThreads == 0) maxThreads = 64;
      maxThreads = std::min<NSUInteger>(maxThreads, 256);
      [maxEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(densityCount), 1, 1)
             threadsPerThreadgroup:MTLSizeMake(maxThreads, 1, 1)];
      [maxEncoder endEncoding];

      if (outEncodedRecord != nullptr) {
        if (buildSerial == 0) {
          if (error) *error = "metal-waveform-cache-build-serial-invalid";
          return false;
        }
        ScopeDerivedResidentRecord record{};
        record.family = ScopeDerivedFamily::Waveform;
        record.builtSerial = buildSerial;
        record.density = densityBuffer;
        record.overflowDensity = overflowDensityBuffer;
        record.maxDensity = maxDensityBuffer;
        record.finalRange = finalRangeBuffer;
        record.byteSize = static_cast<size_t>([densityBuffer length]) +
                          static_cast<size_t>([maxDensityBuffer length]) +
                          (overflowDensityBuffer != nil
                               ? static_cast<size_t>([overflowDensityBuffer length])
                               : 0u) +
                          (finalRangeBuffer != nil
                               ? static_cast<size_t>([finalRangeBuffer length])
                               : 0u);
        *outEncodedRecord = std::move(record);
      }
    }

    id<MTLComputeCommandEncoder> renderEncoder = [commandBuffer computeCommandEncoder];
    if (renderEncoder == nil) {
      if (error) *error = "metal-waveform-render-encoder-failed";
      return false;
    }
    [renderEncoder setComputePipelineState:ctx.waveformSurfaceRenderPipeline];
    [renderEncoder setTexture:outputTexture atIndex:0];
    [renderEncoder setBuffer:densityBuffer offset:0 atIndex:0];
    [renderEncoder setBuffer:overflowDensityBuffer offset:0 atIndex:1];
    [renderEncoder setBuffer:maxDensityBuffer offset:0 atIndex:2];
    [renderEncoder setBuffer:surfaceUniformBuffer offset:0 atIndex:3];
    NSUInteger groupWidth = ctx.waveformSurfaceRenderPipeline.threadExecutionWidth;
    if (groupWidth == 0) groupWidth = 16;
    NSUInteger groupHeight = std::max<NSUInteger>(
        1, std::min<NSUInteger>(16, ctx.waveformSurfaceRenderPipeline.maxTotalThreadsPerThreadgroup / groupWidth));
    [renderEncoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                               static_cast<NSUInteger>(outputSurfaceHeight),
                                               1)
              threadsPerThreadgroup:MTLSizeMake(groupWidth, groupHeight, 1)];
    [renderEncoder endEncoding];

  }
  return true;
}

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
                                        std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "metal-waveform-surface-command-buffer-failed";
    return false;
  }
  if (!encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          nil,
          rasterRequest,
          waveformRequest,
          sourceSurfaceId,
          sourceSurfaceWidth,
          sourceSurfaceHeight,
          sourceSurfacePixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}

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
                                        std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  return encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      nil,
      rasterRequest,
      waveformRequest,
      sourceSurfaceId,
      sourceSurfaceWidth,
      sourceSurfaceHeight,
      sourceSurfacePixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}
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
    std::string* error) {
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  return encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
      commandBuffer,
      source->texture,
      rasterRequest,
      waveformRequest,
      0,
      source->descriptor.width,
      source->descriptor.height,
      source->descriptor.pixelFormat,
      outputSurfaceId,
      outputSurfaceWidth,
      outputSurfaceHeight,
      outputSurfacePixelFormat,
      error);
}

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
    std::string* error) {
  if (error) error->clear();
  if (!cache || buildSerial == 0) {
    if (error) *error = "invalid-metal-waveform-scope-cache-request";
    return false;
  }
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache->cacheId, cache->ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error) ||
      !validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }

  ScopeDerivedResidentRecord residentRecord{};
  std::string residentError;
  if (resolveScopeDerivedRecordForSubmission(
          submission, *cache, true, &residentRecord, &residentError)) {
    if (residentRecord.family != ScopeDerivedFamily::Waveform) {
      if (error) *error = "metal-waveform-scope-cache-family-mismatch";
      return false;
    }
    return encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
        commandBuffer,
        nil,
        rasterRequest,
        waveformRequest,
        0,
        rasterRequest.sourceWidth,
        rasterRequest.sourceHeight,
        rasterRequest.pixelFormat,
        outputSurfaceId,
        outputSurfaceWidth,
        outputSurfaceHeight,
        outputSurfacePixelFormat,
        error,
        &residentRecord,
        nullptr,
        buildSerial);
  }
  if (!residentError.empty()) {
    if (error) *error = residentError;
    return false;
  }

  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  ScopeDerivedResidentRecord encodedRecord{};
  if (!encodeWaveformSurfaceFromTextureSourceOnCommandBuffer(
          commandBuffer,
          source->texture,
          rasterRequest,
          waveformRequest,
          0,
          source->descriptor.width,
          source->descriptor.height,
          source->descriptor.pixelFormat,
          outputSurfaceId,
          outputSurfaceWidth,
          outputSurfaceHeight,
          outputSurfacePixelFormat,
          error,
          nullptr,
          &encodedRecord,
          buildSerial)) {
    return false;
  }
  return registerPendingScopeDerivedRecord(
      submission, cache, std::move(encodedRecord), error);
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildGlossField(const GlossFieldRequest& request,
                     const std::vector<float>& packedPoints,
                     bool allowReadback,
                     GlossFieldResult* out,
                     std::string* error) {
  std::string localError;
  if (!out) {
    if (error) *error = "Missing Metal gloss-field output.";
    return false;
  }
  *out = GlossFieldResult{};
  if (!allowReadback) {
    if (error) *error = "Metal packed gloss-field readback disabled.";
    return false;
  }
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }
  const int gridWidth = std::max(request.gridWidth, 1);
  const int gridHeight = std::max(request.gridHeight, 1);
  const NSUInteger pointCount = static_cast<NSUInteger>(packedPoints.size() / 6u);
  const NSUInteger cellCount = static_cast<NSUInteger>(gridWidth) * static_cast<NSUInteger>(gridHeight);
  if (pointCount == 0u || cellCount == 0u) {
    if (error) *error = "Invalid Metal gloss-field request.";
    return false;
  }

  MetalContext& ctx = context();
  @autoreleasepool {
    GlossFieldAccumulateUniforms accumulateUniforms{};
    accumulateUniforms.pointCount = static_cast<int>(pointCount);
    accumulateUniforms.gridWidth = gridWidth;
    accumulateUniforms.gridHeight = gridHeight;
    accumulateUniforms.showOverflow = request.showOverflow;

    GlossFieldCellUniforms cellUniforms{};
    cellUniforms.cellCount = static_cast<int>(cellCount);
    cellUniforms.gridWidth = gridWidth;
    cellUniforms.gridHeight = gridHeight;
    cellUniforms.neighborhoodChoice = request.neighborhoodChoice;

    id<MTLBuffer> inputBuffer = makeSharedBuffer(packedPoints.data(), packedPoints.size());
    id<MTLBuffer> occupancyCountsBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumRBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumGBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumBBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumYBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMaxBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumMinBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> sumNeutralityBuffer = makeEmptySharedBuffer(cellCount * sizeof(uint32_t));
    id<MTLBuffer> occupancyBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanRBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanGBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> meanBBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierYBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMaxBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> carrierMinBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> neutralityBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> occupancyNormBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> tempBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> bodyBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> rawSignalBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> positiveBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> negativeBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> boundaryBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> congruenceBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> confidenceBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> signalBuffer = makeEmptySharedBuffer(cellCount * sizeof(float));
    id<MTLBuffer> reductionBuffer = makeEmptySharedBuffer(4u * sizeof(uint32_t));
    id<MTLBuffer> accumulateUniformBuffer = makeSharedBuffer(&accumulateUniforms, 1u);
    id<MTLBuffer> cellUniformBuffer = makeSharedBuffer(&cellUniforms, 1u);
    if (inputBuffer == nil || occupancyCountsBuffer == nil || sumRBuffer == nil || sumGBuffer == nil ||
        sumBBuffer == nil || sumYBuffer == nil || sumMaxBuffer == nil || sumMinBuffer == nil ||
        sumNeutralityBuffer == nil || occupancyBuffer == nil || meanRBuffer == nil || meanGBuffer == nil ||
        meanBBuffer == nil || carrierYBuffer == nil || carrierMaxBuffer == nil || carrierMinBuffer == nil ||
        neutralityBuffer == nil || occupancyNormBuffer == nil || tempBuffer == nil || bodyBuffer == nil ||
        rawSignalBuffer == nil || positiveBuffer == nil || negativeBuffer == nil || boundaryBuffer == nil ||
        congruenceBuffer == nil || confidenceBuffer == nil || signalBuffer == nil || reductionBuffer == nil ||
        accumulateUniformBuffer == nil || cellUniformBuffer == nil) {
      if (error) *error = "Failed to allocate Metal gloss-field buffers.";
      return false;
    }

    const auto clearReduction = [&]() {
      clearSharedBuffer(reductionBuffer);
    };
    clearSharedBuffer(occupancyCountsBuffer);
    clearSharedBuffer(sumRBuffer);
    clearSharedBuffer(sumGBuffer);
    clearSharedBuffer(sumBBuffer);
    clearSharedBuffer(sumYBuffer);
    clearSharedBuffer(sumMaxBuffer);
    clearSharedBuffer(sumMinBuffer);
    clearSharedBuffer(sumNeutralityBuffer);
    clearSharedBuffer(occupancyBuffer);
    clearSharedBuffer(meanRBuffer);
    clearSharedBuffer(meanGBuffer);
    clearSharedBuffer(meanBBuffer);
    clearSharedBuffer(carrierYBuffer);
    clearSharedBuffer(carrierMaxBuffer);
    clearSharedBuffer(carrierMinBuffer);
    clearSharedBuffer(neutralityBuffer);
    clearSharedBuffer(occupancyNormBuffer);
    clearSharedBuffer(tempBuffer);
    clearSharedBuffer(bodyBuffer);
    clearSharedBuffer(rawSignalBuffer);
    clearSharedBuffer(positiveBuffer);
    clearSharedBuffer(negativeBuffer);
    clearSharedBuffer(boundaryBuffer);
    clearSharedBuffer(congruenceBuffer);
    clearSharedBuffer(confidenceBuffer);
    clearSharedBuffer(signalBuffer);
    clearReduction();

    if (!runComputeBuffers(ctx.glossFieldAccumulatePipeline,
                           std::array<id<MTLBuffer>, 10>{inputBuffer,
                                                         occupancyCountsBuffer,
                                                         sumRBuffer,
                                                         sumGBuffer,
                                                         sumBBuffer,
                                                         sumYBuffer,
                                                         sumMaxBuffer,
                                                         sumMinBuffer,
                                                         sumNeutralityBuffer,
                                                         accumulateUniformBuffer},
                           pointCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!runComputeBuffers(ctx.glossFieldFinalizePipeline,
                           std::array<id<MTLBuffer>, 17>{occupancyCountsBuffer,
                                                         sumRBuffer,
                                                         sumGBuffer,
                                                         sumBBuffer,
                                                         sumYBuffer,
                                                         sumMaxBuffer,
                                                         sumMinBuffer,
                                                         sumNeutralityBuffer,
                                                         occupancyBuffer,
                                                         meanRBuffer,
                                                         meanGBuffer,
                                                         meanBBuffer,
                                                         carrierYBuffer,
                                                         carrierMaxBuffer,
                                                         carrierMinBuffer,
                                                         neutralityBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyBuffer, occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldBlurPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, tempBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::memcpy([occupancyNormBuffer contents], [tempBuffer contents], static_cast<size_t>(cellCount) * sizeof(float));
    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !runComputeBuffers(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyNormBuffer, occupancyNormBuffer, reductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    const auto blurInPlace = [&](id<MTLBuffer> buffer) -> bool {
      if (!runComputeBuffers(ctx.glossFieldBlurPipeline,
                             std::array<id<MTLBuffer>, 3>{buffer, tempBuffer, cellUniformBuffer},
                             cellCount,
                             &localError)) {
        return false;
      }
      std::memcpy([buffer contents], [tempBuffer contents], static_cast<size_t>(cellCount) * sizeof(float));
      return true;
    };
    if (!blurInPlace(carrierYBuffer) || !blurInPlace(carrierMaxBuffer) ||
        !blurInPlace(carrierMinBuffer) || !blurInPlace(neutralityBuffer)) {
      if (error) *error = localError;
      return false;
    }

    if (!runComputeBuffers(ctx.glossFieldBodyPipeline,
                           std::array<id<MTLBuffer>, 7>{occupancyBuffer,
                                                        meanRBuffer,
                                                        meanGBuffer,
                                                        meanBBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldRawSignalPipeline,
                           std::array<id<MTLBuffer>, 6>{occupancyBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        rawSignalBuffer,
                                                        reductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::array<uint32_t, 4> reductionValues = {0u, 0u, 0u, 0u};
    std::memcpy(reductionValues.data(), [reductionBuffer contents], sizeof(uint32_t));

    clearReduction();
    if (!runComputeBuffers(ctx.glossFieldWeightedSignalPipeline,
                           std::array<id<MTLBuffer>, 11>{occupancyNormBuffer,
                                                         bodyBuffer,
                                                         rawSignalBuffer,
                                                         positiveBuffer,
                                                         negativeBuffer,
                                                         boundaryBuffer,
                                                         congruenceBuffer,
                                                         confidenceBuffer,
                                                         signalBuffer,
                                                         reductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    std::array<uint32_t, 4> weightedValues = {reductionValues[0], 0u, 0u, 0u};
    std::memcpy(weightedValues.data() + 1u, [reductionBuffer contents], 3u * sizeof(uint32_t));
    std::memcpy([reductionBuffer contents], weightedValues.data(), 4u * sizeof(uint32_t));

    if (!runComputeBuffers(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{bodyBuffer,
                                                        signalBuffer,
                                                        positiveBuffer,
                                                        negativeBuffer,
                                                        boundaryBuffer,
                                                        reductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    out->gridWidth = gridWidth;
    out->gridHeight = gridHeight;
    out->occupancy.assign(cellCount, 0.0f);
    out->meanRgb.assign(cellCount * 3u, 0.0f);
    out->carrierY.assign(cellCount, 0.0f);
    out->carrierMax.assign(cellCount, 0.0f);
    out->carrierMin.assign(cellCount, 0.0f);
    out->neutrality.assign(cellCount, 0.0f);
    out->body.assign(cellCount, 0.0f);
    out->signal.assign(cellCount, 0.0f);
    out->positive.assign(cellCount, 0.0f);
    out->negative.assign(cellCount, 0.0f);
    out->boundary.assign(cellCount, 0.0f);
    out->congruence.assign(cellCount, 0.0f);
    out->confidence.assign(cellCount, 0.0f);
    std::vector<float> meanRHost;
    std::vector<float> meanGHost;
    std::vector<float> meanBHost;
    copySharedBuffer<float>(occupancyBuffer, cellCount, &out->occupancy);
    copySharedBuffer<float>(meanRBuffer, cellCount, &meanRHost);
    copySharedBuffer<float>(meanGBuffer, cellCount, &meanGHost);
    copySharedBuffer<float>(meanBBuffer, cellCount, &meanBHost);
    copySharedBuffer<float>(carrierYBuffer, cellCount, &out->carrierY);
    copySharedBuffer<float>(carrierMaxBuffer, cellCount, &out->carrierMax);
    copySharedBuffer<float>(carrierMinBuffer, cellCount, &out->carrierMin);
    copySharedBuffer<float>(neutralityBuffer, cellCount, &out->neutrality);
    copySharedBuffer<float>(bodyBuffer, cellCount, &out->body);
    copySharedBuffer<float>(signalBuffer, cellCount, &out->signal);
    copySharedBuffer<float>(positiveBuffer, cellCount, &out->positive);
    copySharedBuffer<float>(negativeBuffer, cellCount, &out->negative);
    copySharedBuffer<float>(boundaryBuffer, cellCount, &out->boundary);
    copySharedBuffer<float>(congruenceBuffer, cellCount, &out->congruence);
    copySharedBuffer<float>(confidenceBuffer, cellCount, &out->confidence);
    for (NSUInteger idx = 0; idx < cellCount; ++idx) {
      out->meanRgb[idx * 3u + 0u] = idx < meanRHost.size() ? meanRHost[idx] : 0.0f;
      out->meanRgb[idx * 3u + 1u] = idx < meanGHost.size() ? meanGHost[idx] : 0.0f;
      out->meanRgb[idx * 3u + 2u] = idx < meanBHost.size() ? meanBHost[idx] : 0.0f;
    }
  }
  return true;
}
#endif

static bool encodeGlossFieldFromTextureSourceOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    id<MTLTexture> importedSourceTexture,
    const RasterSourceRequest& rasterRequest,
    const GlossFieldRequest& fieldRequest,
    uint32_t surfaceId,
    int surfaceWidth,
    int surfaceHeight,
    int surfacePixelFormat,
    uint64_t buildSerial,
    GlossFieldResidentRecord* outRecord,
    std::string* error) {
  if (error) error->clear();
  if (!outRecord) {
    if (error) *error = "missing-metal-gloss-field-record";
    return false;
  }
  *outRecord = GlossFieldResidentRecord{};
  std::string localError;
  const int pointCount = std::max(rasterRequest.pointCount, 0);
  const int gridWidth = std::max(fieldRequest.gridWidth, 1);
  const int gridHeight = std::max(fieldRequest.gridHeight, 1);
  const NSUInteger cellCount =
      static_cast<NSUInteger>(gridWidth) * static_cast<NSUInteger>(gridHeight);
#if defined(CHROMASPACE_METAL_NATIVE_ONLY)
  if (pointCount <= 0 || cellCount == 0u || importedSourceTexture == nil ||
#else
  if (pointCount <= 0 || cellCount == 0u ||
      (importedSourceTexture == nil && surfaceId == 0) ||
#endif
      commandBuffer == nil ||
      surfaceWidth <= 0 || surfaceHeight <= 0 ||
      rasterRequest.sourceWidth <= 0 || rasterRequest.sourceHeight <= 0 ||
      rasterRequest.sampleCountX <= 0 || surfaceWidth < rasterRequest.sourceWidth ||
      surfaceHeight < rasterRequest.sourceHeight ||
      (surfacePixelFormat != 0 && surfacePixelFormat != 1)) {
    if (error) *error = "invalid-metal-texture-gloss-field-request";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty() ? "metal-gloss-field-command-buffer-invalid"
                                  : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.rasterGlossFieldAccumulateTexturePipeline == nil ||
      ctx.glossFieldFinalizePipeline == nil ||
      ctx.glossFieldMaxPipeline == nil ||
      ctx.glossFieldNormalizePipeline == nil ||
      ctx.glossFieldBlurPipeline == nil ||
      ctx.glossFieldBodyPipeline == nil ||
      ctx.glossFieldRawSignalPipeline == nil ||
      ctx.glossFieldWeightedSignalPipeline == nil ||
      ctx.glossFieldMergeMaxBitsPipeline == nil ||
      ctx.glossFieldFinalNormalizePipeline == nil ||
      ctx.glossFieldLocalPercentilePipeline == nil ||
      ctx.glossFieldCandidate2RawPipeline == nil ||
      ctx.glossFieldAssembleUnifiedPipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "metal-texture-gloss-field");
    }
    return false;
  }

  RasterSourceUniforms rasterUniforms{};
  fillRasterSourceUniforms(rasterRequest, &rasterUniforms);
  rasterUniforms.pixelFormat = surfacePixelFormat;
  rasterUniforms.input.pointCount = pointCount;

  GlossFieldAccumulateUniforms accumulateUniforms{};
  accumulateUniforms.pointCount = pointCount;
  accumulateUniforms.gridWidth = gridWidth;
  accumulateUniforms.gridHeight = gridHeight;
  accumulateUniforms.showOverflow = fieldRequest.showOverflow;

  GlossFieldCellUniforms cellUniforms{};
  cellUniforms.cellCount = static_cast<int>(cellCount);
  cellUniforms.gridWidth = gridWidth;
  cellUniforms.gridHeight = gridHeight;
  cellUniforms.neighborhoodChoice = fieldRequest.neighborhoodChoice;
  const int neighborhoodChoice = std::clamp(fieldRequest.neighborhoodChoice, 0, 2);
  const int neighborhoodRadius = neighborhoodChoice == 0 ? 1 : (neighborhoodChoice == 2 ? 3 : 2);
  const int analysisRadius = std::max(2, neighborhoodRadius * 2);
  const float percentile50 = 50.0f;
  const float percentile35 = 35.0f;

  @autoreleasepool {
    id<MTLTexture> sourceTexture =
        importedSourceTexture;
#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
    if (sourceTexture == nil) {
      sourceTexture = makeTextureFromIOSurface(ctx,
                                               surfaceId,
                                               surfaceWidth,
                                               surfaceHeight,
                                               surfacePixelFormat,
                                               &localError);
    }
#endif
    const auto makeTransientCellUint = [&]() -> id<MTLBuffer> {
      return makeSubmissionTransientPrivateBuffer(
          commandBuffer, cellCount * sizeof(uint32_t), &localError);
    };
    const auto makeTransientCellFloat = [&]() -> id<MTLBuffer> {
      return makeSubmissionTransientPrivateBuffer(
          commandBuffer, cellCount * sizeof(float), &localError);
    };
    const auto makeTransientReduction = [&]() -> id<MTLBuffer> {
      return makeSubmissionTransientPrivateBuffer(
          commandBuffer, 4u * sizeof(uint32_t), &localError);
    };
    id<MTLBuffer> occupancyCountsBuffer = makeTransientCellUint();
    id<MTLBuffer> sumRBuffer = makeTransientCellUint();
    id<MTLBuffer> sumGBuffer = makeTransientCellUint();
    id<MTLBuffer> sumBBuffer = makeTransientCellUint();
    id<MTLBuffer> sumYBuffer = makeTransientCellUint();
    id<MTLBuffer> sumMaxBuffer = makeTransientCellUint();
    id<MTLBuffer> sumMinBuffer = makeTransientCellUint();
    id<MTLBuffer> sumNeutralityBuffer = makeTransientCellUint();
    id<MTLBuffer> occupancyBuffer = makeTransientCellFloat();
    id<MTLBuffer> meanRBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> meanGBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> meanBBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> carrierYBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> carrierMaxBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> carrierMinBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> neutralityBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> occupancyNormBuffer = makeTransientCellFloat();
    id<MTLBuffer> tempBuffer = makeTransientCellFloat();
    id<MTLBuffer> bodyBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> viewerBodyRawBuffer = makeTransientCellFloat();
    id<MTLBuffer> rawSignalBuffer = makeTransientCellFloat();
    id<MTLBuffer> positiveBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> negativeBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> boundaryBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> congruenceBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> confidenceBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> signalBuffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> bodyCoreBuffer = makeTransientCellFloat();
    id<MTLBuffer> bodyContextBuffer = makeTransientCellFloat();
    id<MTLBuffer> retinexBodyBuffer = makeTransientCellFloat();
    id<MTLBuffer> dogLowBuffer = makeTransientCellFloat();
    id<MTLBuffer> body2RawBuffer = makeTransientCellFloat();
    id<MTLBuffer> positive2RawBuffer = makeTransientCellFloat();
    id<MTLBuffer> negative2RawBuffer = makeTransientCellFloat();
    id<MTLBuffer> confidence2RawBuffer = makeTransientCellFloat();
    id<MTLBuffer> agreement2RawBuffer = makeTransientCellFloat();
    id<MTLBuffer> body2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> positive2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> negative2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> boundary2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> congruence2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> confidence2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> signal2Buffer =
        makeEmptyPrivateBuffer(ctx, cellCount * sizeof(float));
    id<MTLBuffer> bodyReductionBuffer = makeTransientReduction();
    id<MTLBuffer> weightedReductionBuffer = makeTransientReduction();
    id<MTLBuffer> finalReductionBuffer = makeTransientReduction();
    id<MTLBuffer> candidate2ReductionBuffer = makeTransientReduction();
    id<MTLBuffer> rasterUniformBuffer =
        makeSharedBuffer(ctx, &rasterUniforms, 1u);
    id<MTLBuffer> accumulateUniformBuffer =
        makeSharedBuffer(ctx, &accumulateUniforms, 1u);
    id<MTLBuffer> cellUniformBuffer =
        makeSharedBuffer(ctx, &cellUniforms, 1u);
    id<MTLBuffer> percentile50Buffer =
        makeSharedBuffer(ctx, &percentile50, 1u);
    id<MTLBuffer> percentile35Buffer =
        makeSharedBuffer(ctx, &percentile35, 1u);
    if (sourceTexture == nil || occupancyCountsBuffer == nil || sumRBuffer == nil ||
        sumGBuffer == nil || sumBBuffer == nil || sumYBuffer == nil || sumMaxBuffer == nil ||
        sumMinBuffer == nil || sumNeutralityBuffer == nil || occupancyBuffer == nil ||
        meanRBuffer == nil || meanGBuffer == nil || meanBBuffer == nil ||
        carrierYBuffer == nil || carrierMaxBuffer == nil || carrierMinBuffer == nil ||
        neutralityBuffer == nil || occupancyNormBuffer == nil || tempBuffer == nil ||
        bodyBuffer == nil || viewerBodyRawBuffer == nil || rawSignalBuffer == nil || positiveBuffer == nil ||
        negativeBuffer == nil || boundaryBuffer == nil || congruenceBuffer == nil ||
        confidenceBuffer == nil || signalBuffer == nil || bodyCoreBuffer == nil ||
        bodyContextBuffer == nil || retinexBodyBuffer == nil || dogLowBuffer == nil ||
        body2RawBuffer == nil || positive2RawBuffer == nil || negative2RawBuffer == nil ||
        confidence2RawBuffer == nil || agreement2RawBuffer == nil || body2Buffer == nil ||
        positive2Buffer == nil || negative2Buffer == nil || boundary2Buffer == nil ||
        congruence2Buffer == nil || confidence2Buffer == nil || signal2Buffer == nil ||
        bodyReductionBuffer == nil || weightedReductionBuffer == nil || finalReductionBuffer == nil ||
        candidate2ReductionBuffer == nil || rasterUniformBuffer == nil ||
        accumulateUniformBuffer == nil || cellUniformBuffer == nil ||
        percentile50Buffer == nil || percentile35Buffer == nil) {
      if (error) *error = localError.empty() ? "metal-iosurface-gloss-field-allocation-failed" : localError;
      return false;
    }
    const auto encodeCompute =
        [&](id<MTLComputePipelineState> pipeline,
            const auto& buffers,
            NSUInteger threadCount,
            std::string* encodeError) -> bool {
      return encodeComputeBuffersOnCommandBuffer(commandBuffer,
                                                  pipeline,
                                                  buffers,
                                                  threadCount,
                                                  encodeError);
    };
    const auto encodeClear =
        [&](id<MTLBuffer> buffer, std::string* encodeError) -> bool {
      return encodeBufferClearOnCommandBuffer(commandBuffer, buffer, encodeError);
    };
    const auto encodeCopy =
        [&](id<MTLBuffer> src,
            id<MTLBuffer> dst,
            NSUInteger bytes,
            std::string* encodeError) -> bool {
      return encodeBufferCopyOnCommandBuffer(commandBuffer,
                                              src,
                                              dst,
                                              bytes,
                                              encodeError);
    };
    const std::array<id<MTLBuffer>, 47> buffersToClear = {
        occupancyCountsBuffer, sumRBuffer, sumGBuffer, sumBBuffer, sumYBuffer,
        sumMaxBuffer, sumMinBuffer, sumNeutralityBuffer, occupancyBuffer, meanRBuffer,
        meanGBuffer, meanBBuffer, carrierYBuffer, carrierMaxBuffer, carrierMinBuffer,
        neutralityBuffer, occupancyNormBuffer, tempBuffer, bodyBuffer, viewerBodyRawBuffer, rawSignalBuffer,
        positiveBuffer, negativeBuffer, boundaryBuffer, congruenceBuffer, confidenceBuffer,
        signalBuffer, bodyCoreBuffer, bodyContextBuffer, retinexBodyBuffer, dogLowBuffer,
        body2RawBuffer, positive2RawBuffer, negative2RawBuffer, confidence2RawBuffer,
        agreement2RawBuffer, body2Buffer, positive2Buffer, negative2Buffer, boundary2Buffer,
        congruence2Buffer, confidence2Buffer, signal2Buffer, bodyReductionBuffer,
        weightedReductionBuffer, finalReductionBuffer, candidate2ReductionBuffer};
    id<MTLBlitCommandEncoder> initialClearEncoder =
        [commandBuffer blitCommandEncoder];
    if (initialClearEncoder == nil) {
      if (error) *error = "metal-iosurface-gloss-field-clear-encoder-failed";
      return false;
    }
    for (id<MTLBuffer> buffer : buffersToClear) {
      if (buffer != nil) {
        [initialClearEncoder fillBuffer:buffer
                                  range:NSMakeRange(0, [buffer length])
                                  value:0];
      }
    }
    [initialClearEncoder endEncoding];

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-iosurface-gloss-field-accumulate-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.rasterGlossFieldAccumulateTexturePipeline];
    [encoder setTexture:sourceTexture atIndex:0];
    [encoder setBuffer:occupancyCountsBuffer offset:0 atIndex:0];
    [encoder setBuffer:sumRBuffer offset:0 atIndex:1];
    [encoder setBuffer:sumGBuffer offset:0 atIndex:2];
    [encoder setBuffer:sumBBuffer offset:0 atIndex:3];
    [encoder setBuffer:sumYBuffer offset:0 atIndex:4];
    [encoder setBuffer:sumMaxBuffer offset:0 atIndex:5];
    [encoder setBuffer:sumMinBuffer offset:0 atIndex:6];
    [encoder setBuffer:sumNeutralityBuffer offset:0 atIndex:7];
    [encoder setBuffer:rasterUniformBuffer offset:0 atIndex:8];
    [encoder setBuffer:accumulateUniformBuffer offset:0 atIndex:9];
    NSUInteger threads = ctx.rasterGlossFieldAccumulateTexturePipeline.maxTotalThreadsPerThreadgroup;
    if (threads == 0) threads = 64;
    threads = std::min<NSUInteger>(threads, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(pointCount), 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];

    const auto clearBodyReduction = [&]() -> bool {
      return encodeBufferClearOnCommandBuffer(commandBuffer,
                                               bodyReductionBuffer,
                                               &localError);
    };
    const auto clearWeightedReduction = [&]() -> bool {
      return encodeBufferClearOnCommandBuffer(commandBuffer,
                                               weightedReductionBuffer,
                                               &localError);
    };
    const auto clearFinalReduction = [&]() -> bool {
      return encodeBufferClearOnCommandBuffer(commandBuffer,
                                               finalReductionBuffer,
                                               &localError);
    };
    if (!encodeCompute(ctx.glossFieldFinalizePipeline,
                           std::array<id<MTLBuffer>, 17>{occupancyCountsBuffer,
                                                         sumRBuffer,
                                                         sumGBuffer,
                                                         sumBBuffer,
                                                         sumYBuffer,
                                                         sumMaxBuffer,
                                                         sumMinBuffer,
                                                         sumNeutralityBuffer,
                                                         occupancyBuffer,
                                                         meanRBuffer,
                                                         meanGBuffer,
                                                         meanBBuffer,
                                                         carrierYBuffer,
                                                         carrierMaxBuffer,
                                                         carrierMinBuffer,
                                                         neutralityBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBodyReduction() ||
        !encodeCompute(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !encodeCompute(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyBuffer, occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !encodeCompute(ctx.glossFieldBlurPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, tempBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCopy(tempBuffer, occupancyNormBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBodyReduction() ||
        !encodeCompute(ctx.glossFieldMaxPipeline,
                           std::array<id<MTLBuffer>, 3>{occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError) ||
        !encodeCompute(ctx.glossFieldNormalizePipeline,
                           std::array<id<MTLBuffer>, 4>{occupancyNormBuffer, occupancyNormBuffer, bodyReductionBuffer, cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    const auto blurInPlace = [&](id<MTLBuffer> buffer) -> bool {
      if (!encodeCompute(ctx.glossFieldBlurPipeline,
                             std::array<id<MTLBuffer>, 3>{buffer, tempBuffer, cellUniformBuffer},
                             cellCount,
                             &localError)) {
        return false;
      }
      return encodeCopy(tempBuffer, buffer, cellCount * sizeof(float), &localError);
    };
    if (!blurInPlace(carrierYBuffer) || !blurInPlace(carrierMaxBuffer) ||
        !blurInPlace(carrierMinBuffer) || !blurInPlace(neutralityBuffer)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCompute(ctx.glossFieldBodyPipeline,
                           std::array<id<MTLBuffer>, 7>{occupancyBuffer,
                                                        meanRBuffer,
                                                        meanGBuffer,
                                                        meanBBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCopy(bodyBuffer, viewerBodyRawBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearBodyReduction() ||
        !encodeCompute(ctx.glossFieldRawSignalPipeline,
                           std::array<id<MTLBuffer>, 6>{occupancyBuffer,
                                                        carrierMaxBuffer,
                                                        bodyBuffer,
                                                        rawSignalBuffer,
                                                        bodyReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearWeightedReduction() ||
        !encodeCompute(ctx.glossFieldWeightedSignalPipeline,
                           std::array<id<MTLBuffer>, 11>{occupancyNormBuffer,
                                                         bodyBuffer,
                                                         rawSignalBuffer,
                                                         positiveBuffer,
                                                         negativeBuffer,
                                                         boundaryBuffer,
                                                         congruenceBuffer,
                                                         confidenceBuffer,
                                                         signalBuffer,
                                                         weightedReductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!clearFinalReduction() ||
        !encodeCompute(ctx.glossFieldMergeMaxBitsPipeline,
                           std::array<id<MTLBuffer>, 3>{bodyReductionBuffer,
                                                        weightedReductionBuffer,
                                                        finalReductionBuffer},
                           1u,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCompute(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{bodyBuffer,
                                                        signalBuffer,
                                                        positiveBuffer,
                                                        negativeBuffer,
                                                        boundaryBuffer,
                                                        finalReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    if (!encodeCompute(ctx.glossFieldLocalPercentilePipeline,
                           std::array<id<MTLBuffer>, 5>{carrierYBuffer,
                                                        occupancyBuffer,
                                                        bodyCoreBuffer,
                                                        cellUniformBuffer,
                                                        percentile50Buffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCopy(carrierYBuffer, bodyContextBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    for (int i = 0; i < std::max(2, analysisRadius * 2); ++i) {
      if (!blurInPlace(bodyContextBuffer)) {
        if (error) *error = localError;
        return false;
      }
    }
    if (!encodeCompute(ctx.glossFieldLocalPercentilePipeline,
                           std::array<id<MTLBuffer>, 5>{carrierYBuffer,
                                                        occupancyBuffer,
                                                        retinexBodyBuffer,
                                                        cellUniformBuffer,
                                                        percentile35Buffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCopy(carrierYBuffer, dogLowBuffer, cellCount * sizeof(float), &localError)) {
      if (error) *error = localError;
      return false;
    }
    for (int i = 0; i < std::max(1, analysisRadius / 2); ++i) {
      if (!blurInPlace(dogLowBuffer)) {
        if (error) *error = localError;
        return false;
      }
    }
    if (!encodeCompute(ctx.glossFieldCandidate2RawPipeline,
                           std::array<id<MTLBuffer>, 18>{occupancyBuffer,
                                                         occupancyNormBuffer,
                                                         meanRBuffer,
                                                         meanGBuffer,
                                                         meanBBuffer,
                                                         carrierYBuffer,
                                                         viewerBodyRawBuffer,
                                                         bodyCoreBuffer,
                                                         bodyContextBuffer,
                                                         retinexBodyBuffer,
                                                         dogLowBuffer,
                                                         bodyContextBuffer,
                                                         body2RawBuffer,
                                                         positive2RawBuffer,
                                                         negative2RawBuffer,
                                                         confidence2RawBuffer,
                                                         agreement2RawBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeClear(candidate2ReductionBuffer, &localError) ||
        !encodeCompute(ctx.glossFieldAssembleUnifiedPipeline,
                           std::array<id<MTLBuffer>, 14>{body2RawBuffer,
                                                         positive2RawBuffer,
                                                         negative2RawBuffer,
                                                         confidence2RawBuffer,
                                                         agreement2RawBuffer,
                                                         body2Buffer,
                                                         signal2Buffer,
                                                         positive2Buffer,
                                                         negative2Buffer,
                                                         boundary2Buffer,
                                                         congruence2Buffer,
                                                         confidence2Buffer,
                                                         candidate2ReductionBuffer,
                                                         cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }
    if (!encodeCompute(ctx.glossFieldFinalNormalizePipeline,
                           std::array<id<MTLBuffer>, 7>{body2Buffer,
                                                        signal2Buffer,
                                                        positive2Buffer,
                                                        negative2Buffer,
                                                        boundary2Buffer,
                                                        candidate2ReductionBuffer,
                                                        cellUniformBuffer},
                           cellCount,
                           &localError)) {
      if (error) *error = localError;
      return false;
    }

    constexpr size_t kGlossResidentCellBufferCount = 21u;
    constexpr size_t kGlossResidentBytesPerCell =
        kGlossResidentCellBufferCount * sizeof(float);
    if (cellCount > std::numeric_limits<size_t>::max() /
                        kGlossResidentBytesPerCell) {
      if (error) *error = "metal-gloss-field-resident-size-overflow";
      return false;
    }
    const size_t residentBytes =
        static_cast<size_t>(cellCount) * kGlossResidentBytesPerCell;

    GlossFieldResidentRecord record{};
    record.gridWidth = gridWidth;
    record.gridHeight = gridHeight;
    record.builtSerial = buildSerial;
    record.byteSize = residentBytes;
    record.meanR = meanRBuffer;
    record.meanG = meanGBuffer;
    record.meanB = meanBBuffer;
    record.carrierY = carrierYBuffer;
    record.carrierMax = carrierMaxBuffer;
    record.carrierMin = carrierMinBuffer;
    record.neutrality = neutralityBuffer;
    record.body = bodyBuffer;
    record.positive = positiveBuffer;
    record.negative = negativeBuffer;
    record.boundary = boundaryBuffer;
    record.congruence = congruenceBuffer;
    record.confidence = confidenceBuffer;
    record.signal = signalBuffer;
    record.body2 = body2Buffer;
    record.positive2 = positive2Buffer;
    record.negative2 = negative2Buffer;
    record.boundary2 = boundary2Buffer;
    record.congruence2 = congruence2Buffer;
    record.confidence2 = confidence2Buffer;
    record.signal2 = signal2Buffer;
    *outRecord = std::move(record);
  }
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool buildGlossFieldFromIOSurface(GlossFieldCache* cache,
                                  const RasterSourceRequest& rasterRequest,
                                  const GlossFieldRequest& fieldRequest,
                                  uint32_t surfaceId,
                                  int surfaceWidth,
                                  int surfaceHeight,
                                  int surfacePixelFormat,
                                  uint64_t buildSerial,
                                  std::string* error) {
  if (error) error->clear();
  if (!cache) {
    if (error) *error = "missing-metal-gloss-field-cache";
    return false;
  }
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "metal-iosurface-gloss-field-command-buffer-failed";
    return false;
  }
  GlossFieldResidentRecord record{};
  if (!encodeGlossFieldFromTextureSourceOnCommandBuffer(
          commandBuffer,
          nil,
          rasterRequest,
          fieldRequest,
          surfaceId,
          surfaceWidth,
          surfaceHeight,
          surfacePixelFormat,
          buildSerial,
          &record,
          error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  ScopeDerivedResidentRecord derivedRecord{};
  derivedRecord.family = ScopeDerivedFamily::GlossField;
  derivedRecord.builtSerial = record.builtSerial;
  derivedRecord.byteSize = record.byteSize;
  try {
    derivedRecord.glossField =
        std::make_shared<GlossFieldResidentRecord>(std::move(record));
  } catch (...) {
    if (error) *error = "metal-gloss-field-cache-record-allocation-failed";
    return false;
  }
  ScopeDerivedCache derivedCache = glossDerivedCache(*cache);
  if (!registerCommittedScopeDerivedRecord(
          &derivedCache, std::move(derivedRecord), error)) {
    return false;
  }
  cache->cacheId = derivedCache.cacheId;
  cache->ownerCompositorId = derivedCache.ownerCompositorId;
  cache->byteSize = derivedCache.byteSize;
  cache->gridWidth = std::max(fieldRequest.gridWidth, 1);
  cache->gridHeight = std::max(fieldRequest.gridHeight, 1);
  cache->builtSerial = buildSerial;
  cache->available = true;
  return true;
}
#endif

static bool encodeGlossFieldFromTextureSourceForSubmission(
    const FrameSubmission& submission,
    GlossFieldCache* cache,
    const RasterSourceRequest& rasterRequest,
    const GlossFieldRequest& fieldRequest,
    id<MTLTexture> importedSourceTexture,
    uint32_t surfaceId,
    int surfaceWidth,
    int surfaceHeight,
    int surfacePixelFormat,
    uint64_t buildSerial,
    std::string* error) {
  if (error) error->clear();
  if (!cache) {
    if (error) *error = "missing-metal-gloss-field-cache";
    return false;
  }
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache->cacheId, cache->ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  GlossFieldResidentRecord encodedRecord{};
  if (!encodeGlossFieldFromTextureSourceOnCommandBuffer(
          commandBuffer,
          importedSourceTexture,
          rasterRequest,
          fieldRequest,
          surfaceId,
          surfaceWidth,
          surfaceHeight,
          surfacePixelFormat,
          buildSerial,
          &encodedRecord,
          error)) {
    return false;
  }
  ScopeDerivedResidentRecord derivedRecord{};
  derivedRecord.family = ScopeDerivedFamily::GlossField;
  derivedRecord.builtSerial = encodedRecord.builtSerial;
  derivedRecord.byteSize = encodedRecord.byteSize;
  try {
    derivedRecord.glossField =
        std::make_shared<GlossFieldResidentRecord>(std::move(encodedRecord));
  } catch (...) {
    if (error) *error = "metal-gloss-field-cache-record-allocation-failed";
    return false;
  }
  ScopeDerivedCache derivedCache = glossDerivedCache(*cache);
  if (!registerPendingScopeDerivedRecord(
          submission, &derivedCache, std::move(derivedRecord), error)) {
    return false;
  }
  cache->cacheId = derivedCache.cacheId;
  cache->ownerCompositorId = derivedCache.ownerCompositorId;
  cache->byteSize = derivedCache.byteSize;
  cache->gridWidth = std::max(fieldRequest.gridWidth, 1);
  cache->gridHeight = std::max(fieldRequest.gridHeight, 1);
  cache->builtSerial = buildSerial;
  cache->available = true;
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool encodeGlossFieldFromIOSurface(const FrameSubmission& submission,
                                   GlossFieldCache* cache,
                                   const RasterSourceRequest& rasterRequest,
                                   const GlossFieldRequest& fieldRequest,
                                   uint32_t surfaceId,
                                   int surfaceWidth,
                                   int surfaceHeight,
                                   int surfacePixelFormat,
                                   uint64_t buildSerial,
                                   std::string* error) {
  return encodeGlossFieldFromTextureSourceForSubmission(
      submission,
      cache,
      rasterRequest,
      fieldRequest,
      nil,
      surfaceId,
      surfaceWidth,
      surfaceHeight,
      surfacePixelFormat,
      buildSerial,
      error);
}
#endif

bool encodeGlossFieldFromImportedTexture(
    const FrameSubmission& submission,
    GlossFieldCache* cache,
    const RasterSourceRequest& rasterRequest,
    const GlossFieldRequest& fieldRequest,
    uint64_t sourceId,
    uint64_t buildSerial,
    std::string* error) {
  if (error) error->clear();
  if (!cache) {
    if (error) *error = "missing-metal-gloss-field-cache";
    return false;
  }
  std::shared_ptr<ImportedSourceRecord> source;
  if (!importedSourceForFrameSubmission(
          submission, sourceId, &source, error)) {
    return false;
  }
  return encodeGlossFieldFromTextureSourceForSubmission(
      submission,
      cache,
      rasterRequest,
      fieldRequest,
      source->texture,
      0,
      source->descriptor.width,
      source->descriptor.height,
      source->descriptor.pixelFormat,
      buildSerial,
      error);
}

static bool encodeGlossFieldSurfaceFromCacheOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    const GlossFieldResidentRecord* stagedRecord,
    const GlossFieldCache& cache,
    const GlossFieldSurfaceRequest& surfaceRequest,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  if (error) error->clear();
  std::string localError;
  if (!cache.available || cache.cacheId == 0 || cache.gridWidth <= 0 || cache.gridHeight <= 0 ||
      outputSurfaceId == 0 || outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0) {
    if (error) *error = "invalid-metal-gloss-field-surface-request";
    return false;
  }
  const bool useCandidate2 = surfaceRequest.algorithm != 0;
  if (surfaceRequest.algorithm < 0 || surfaceRequest.algorithm > 1) {
    if (error) *error = "unsupported-metal-gloss-field-surface-algorithm";
    return false;
  }
  if (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1) {
    if (error) *error = "unsupported-metal-gloss-field-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty()
                   ? "metal-gloss-field-surface-command-buffer-invalid"
                   : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.glossFieldSurfaceRenderPipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(ctx, "metal-gloss-field-surface");
    }
    return false;
  }

  GlossFieldResidentRecord record{};
  if (stagedRecord == nullptr) {
    if (error) *error = "metal-gloss-field-cache-missing";
    return false;
  }
  record = *stagedRecord;
  if (record.gridWidth != cache.gridWidth || record.gridHeight != cache.gridHeight ||
      record.meanR == nil || record.meanG == nil || record.meanB == nil ||
      record.carrierY == nil || record.carrierMax == nil || record.carrierMin == nil ||
      record.neutrality == nil || record.body == nil || record.positive == nil ||
      record.negative == nil || record.boundary == nil || record.congruence == nil ||
      record.confidence == nil || record.signal == nil ||
      (useCandidate2 && (record.body2 == nil || record.positive2 == nil ||
                         record.negative2 == nil || record.boundary2 == nil ||
                         record.congruence2 == nil || record.confidence2 == nil ||
                         record.signal2 == nil))) {
    if (error) *error = "metal-gloss-field-cache-incomplete";
    return false;
  }

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-gloss-field-output-surface-missing";
    return false;
  }

  GlossFieldSurfaceUniforms uniforms{};
  uniforms.gridWidth = cache.gridWidth;
  uniforms.gridHeight = cache.gridHeight;
  uniforms.surfaceWidth = outputSurfaceWidth;
  uniforms.surfaceHeight = outputSurfaceHeight;
  uniforms.algorithm = surfaceRequest.algorithm;
  uniforms.colorMode = std::clamp(surfaceRequest.colorMode, 0, 1);
  uniforms.debugMode = std::clamp(surfaceRequest.debugMode, 0, 4);
  uniforms.diagnosticMode = std::clamp(surfaceRequest.diagnosticMode, 0, 2);
  uniforms.colorSaturation = std::clamp(surfaceRequest.colorSaturation, 0.8f, 6.0f);
  uniforms.glossBodyOpacity = std::clamp(surfaceRequest.glossBodyOpacity, 0.0f, 1.0f);
  uniforms.glossHighlightOpacity = std::clamp(surfaceRequest.glossHighlightOpacity, 0.0f, 1.0f);
  uniforms.glossLiftScale = std::max(0.01f, surfaceRequest.glossLiftScale);

  @autoreleasepool {
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(ctx, &uniforms, 1u);
    if (uniformBuffer == nil) {
      if (error) *error = "metal-gloss-field-surface-uniform-allocation-failed";
      return false;
    }
    if (commandBuffer == nil) {
      if (error) *error = "metal-gloss-field-surface-command-buffer-failed";
      return false;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-field-surface-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossFieldSurfaceRenderPipeline];
    [encoder setTexture:outputTexture atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:0];
    [encoder setBuffer:record.meanG offset:0 atIndex:1];
    [encoder setBuffer:record.meanB offset:0 atIndex:2];
    [encoder setBuffer:record.carrierY offset:0 atIndex:3];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:5];
    [encoder setBuffer:record.neutrality offset:0 atIndex:6];
    [encoder setBuffer:(useCandidate2 ? record.body2 : record.body) offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.positive2 : record.positive) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.negative2 : record.negative) offset:0 atIndex:9];
    [encoder setBuffer:(useCandidate2 ? record.boundary2 : record.boundary) offset:0 atIndex:10];
    [encoder setBuffer:(useCandidate2 ? record.congruence2 : record.congruence) offset:0 atIndex:11];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:12];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:13];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:14];
    NSUInteger width = ctx.glossFieldSurfaceRenderPipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::max<NSUInteger>(
        1,
        std::min<NSUInteger>(16, static_cast<NSUInteger>(std::sqrt(static_cast<double>(width)))));
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                         static_cast<NSUInteger>(outputSurfaceHeight),
                                         1)
       threadsPerThreadgroup:MTLSizeMake(width, width, 1)];
    [encoder endEncoding];
  }
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderGlossFieldSurfaceFromCache(const GlossFieldCache& cache,
                                      const GlossFieldSurfaceRequest& surfaceRequest,
                                      uint32_t outputSurfaceId,
                                      int outputSurfaceWidth,
                                      int outputSurfaceHeight,
                                      int outputSurfacePixelFormat,
                                      std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "metal-gloss-field-surface-command-buffer-failed";
    return false;
  }
  GlossFieldResidentRecord record{};
  if (!resolveGlossFieldRecord(cache, 0, 0, false, &record)) {
    if (error) *error = "metal-gloss-field-cache-missing";
    return false;
  }
  if (!encodeGlossFieldSurfaceFromCacheOnCommandBuffer(commandBuffer,
                                                       &record,
                                                       cache,
                                                       surfaceRequest,
                                                       outputSurfaceId,
                                                       outputSurfaceWidth,
                                                       outputSurfaceHeight,
                                                       outputSurfacePixelFormat,
                                                       error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}
#endif

bool encodeGlossFieldSurfaceFromCache(const FrameSubmission& submission,
                                      const GlossFieldCache& cache,
                                      const GlossFieldSurfaceRequest& surfaceRequest,
                                      uint32_t outputSurfaceId,
                                      int outputSurfaceWidth,
                                      int outputSurfaceHeight,
                                      int outputSurfacePixelFormat,
                                      std::string* error) {
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache.cacheId, cache.ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  GlossFieldResidentRecord stagedRecord{};
  std::string residentError;
  if (!resolveGlossFieldRecordForSubmission(
          submission, cache, true, &stagedRecord, &residentError)) {
    if (error) {
      *error = residentError.empty()
                   ? "metal-gloss-field-cache-missing"
                   : residentError;
    }
    return false;
  }
  return encodeGlossFieldSurfaceFromCacheOnCommandBuffer(commandBuffer,
                                                         &stagedRecord,
                                                         cache,
                                                         surfaceRequest,
                                                         outputSurfaceId,
                                                         outputSurfaceWidth,
                                                         outputSurfaceHeight,
                                                         outputSurfacePixelFormat,
                                                         error);
}

static bool encodeGlossProjectionSurfaceFromCacheOnCommandBuffer(
    id<MTLCommandBuffer> commandBuffer,
    const GlossFieldResidentRecord* stagedRecord,
    const GlossFieldCache& cache,
    const GlossProjectionSurfaceRequest& projectionRequest,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  if (error) error->clear();
  std::string localError;
  if (!cache.available || cache.cacheId == 0 || cache.gridWidth <= 0 || cache.gridHeight <= 0 ||
      outputSurfaceId == 0 || outputSurfaceWidth <= 0 || outputSurfaceHeight <= 0) {
    if (error) *error = "invalid-metal-gloss-projection-surface-request";
    return false;
  }
  const bool useCandidate2 = projectionRequest.algorithm != 0;
  if (projectionRequest.algorithm < 0 || projectionRequest.algorithm > 1) {
    if (error) *error = "unsupported-metal-gloss-projection-surface-algorithm";
    return false;
  }
  if (outputSurfacePixelFormat != 0 && outputSurfacePixelFormat != 1) {
    if (error) *error = "unsupported-metal-gloss-projection-surface-format";
    return false;
  }
  std::shared_ptr<MetalContext> runtimeContext;
  MetalContext* contextPointer = nullptr;
  if (!contextForCommandBuffer(commandBuffer,
                               &runtimeContext,
                               &contextPointer,
                               &localError) ||
      contextPointer == nullptr) {
    if (error) {
      *error = localError.empty()
                   ? "metal-gloss-projection-surface-command-buffer-invalid"
                   : localError;
    }
    return false;
  }
  MetalContext& ctx = *contextPointer;
  if (ctx.glossProjectionSurfaceSelectPipeline == nil ||
      ctx.glossProjectionSurfaceShadePipeline == nil) {
    if (error) {
      *error = residentPipelineUnavailableReason(
          ctx, "metal-gloss-projection-surface");
    }
    return false;
  }

  GlossFieldResidentRecord record{};
  if (stagedRecord == nullptr) {
    if (error) *error = "metal-gloss-field-cache-missing";
    return false;
  }
  record = *stagedRecord;
  if (record.gridWidth != cache.gridWidth || record.gridHeight != cache.gridHeight ||
      record.meanR == nil || record.meanG == nil || record.meanB == nil ||
      record.carrierY == nil || record.carrierMax == nil || record.carrierMin == nil ||
      record.neutrality == nil || record.body == nil || record.positive == nil ||
      record.negative == nil || record.boundary == nil || record.congruence == nil ||
      record.confidence == nil || record.signal == nil ||
      (useCandidate2 && (record.body2 == nil || record.positive2 == nil ||
                         record.negative2 == nil || record.boundary2 == nil ||
                         record.congruence2 == nil || record.confidence2 == nil ||
                         record.signal2 == nil))) {
    if (error) *error = "metal-gloss-field-cache-incomplete";
    return false;
  }

  id<MTLTexture> outputTexture = nil;
  {
    std::lock_guard<std::mutex> lock(plotSurfaceMutex());
    auto& registry = plotSurfaceRegistry();
    auto it = registry.find(outputSurfaceId);
    if (it != registry.end() && it->second &&
        it->second->width == outputSurfaceWidth &&
        it->second->height == outputSurfaceHeight &&
        it->second->pixelFormat == outputSurfacePixelFormat &&
        it->second->context == runtimeContext) {
      outputTexture = it->second->texture;
    }
  }
  if (outputTexture == nil) {
    if (error) *error = "metal-gloss-projection-output-surface-missing";
    return false;
  }
  if (!encodePlotSurfaceClearOnCommandBuffer(commandBuffer,
                                             outputSurfaceId,
                                             outputSurfaceWidth,
                                             outputSurfaceHeight,
                                             outputSurfacePixelFormat,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             &localError)) {
    if (error) *error = localError.empty() ? "metal-gloss-projection-clear-failed" : localError;
    return false;
  }

  GlossProjectionSurfaceUniforms uniforms{};
  uniforms.gridWidth = cache.gridWidth;
  uniforms.gridHeight = cache.gridHeight;
  uniforms.surfaceWidth = outputSurfaceWidth;
  uniforms.surfaceHeight = outputSurfaceHeight;
  uniforms.algorithm = projectionRequest.algorithm;
  uniforms.colorMode = std::clamp(projectionRequest.colorMode, 0, 1);
  uniforms.debugMode = std::clamp(projectionRequest.debugMode, 0, 4);
  uniforms.diagnosticMode = std::clamp(projectionRequest.diagnosticMode, 0, 2);
  uniforms.sourceAspect = std::clamp(projectionRequest.sourceAspect, 0.25f, 4.0f);
  uniforms.colorSaturation = std::clamp(projectionRequest.colorSaturation, 0.8f, 6.0f);
  uniforms.glossBodyOpacity = std::clamp(projectionRequest.glossBodyOpacity, 0.0f, 1.0f);
  uniforms.glossHighlightOpacity = std::clamp(projectionRequest.glossHighlightOpacity, 0.0f, 1.0f);
  uniforms.glossLiftScale = std::max(0.01f, projectionRequest.glossLiftScale);
  uniforms.pointRadiusPixels = std::clamp(projectionRequest.pointRadiusPixels, 1.0f, 6.0f);
  std::copy(projectionRequest.modelView, projectionRequest.modelView + 16, uniforms.modelView);
  std::copy(projectionRequest.projection, projectionRequest.projection + 16, uniforms.projection);

  @autoreleasepool {
    id<MTLBuffer> uniformBuffer = makeSharedBuffer(ctx, &uniforms, 1u);
    const size_t selectionBytes =
        static_cast<size_t>(std::max(outputSurfaceWidth, 0)) *
        static_cast<size_t>(std::max(outputSurfaceHeight, 0)) *
        sizeof(uint32_t);
    id<MTLBuffer> selectionBuffer = makeSubmissionTransientPrivateBuffer(
        commandBuffer, selectionBytes, &localError);
    if (uniformBuffer == nil || selectionBuffer == nil) {
      if (error) *error = "metal-gloss-projection-surface-allocation-failed";
      return false;
    }
    if (commandBuffer == nil) {
      if (error) *error = "metal-gloss-projection-surface-command-buffer-failed";
      return false;
    }
    id<MTLBlitCommandEncoder> clearEncoder = [commandBuffer blitCommandEncoder];
    if (clearEncoder == nil) {
      if (error) *error = "metal-gloss-projection-selection-clear-encoder-failed";
      return false;
    }
    [clearEncoder fillBuffer:selectionBuffer
                       range:NSMakeRange(0, selectionBytes)
                       value:0];
    [clearEncoder endEncoding];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-projection-surface-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossProjectionSurfaceSelectPipeline];
    [encoder setBuffer:selectionBuffer offset:0 atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:1];
    [encoder setBuffer:record.meanG offset:0 atIndex:2];
    [encoder setBuffer:record.meanB offset:0 atIndex:3];
    [encoder setBuffer:record.carrierY offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:5];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:6];
    [encoder setBuffer:record.neutrality offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:9];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:10];
    NSUInteger width = ctx.glossProjectionSurfaceSelectPipeline.maxTotalThreadsPerThreadgroup;
    if (width == 0) width = 64;
    width = std::min<NSUInteger>(width, 64);
    const NSUInteger cellCount =
        static_cast<NSUInteger>(std::max(cache.gridWidth, 0)) *
        static_cast<NSUInteger>(std::max(cache.gridHeight, 0));
    [encoder dispatchThreads:MTLSizeMake(cellCount, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];

    encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (error) *error = "metal-gloss-projection-surface-shade-encoder-failed";
      return false;
    }
    [encoder setComputePipelineState:ctx.glossProjectionSurfaceShadePipeline];
    [encoder setTexture:outputTexture atIndex:0];
    [encoder setBuffer:selectionBuffer offset:0 atIndex:0];
    [encoder setBuffer:record.meanR offset:0 atIndex:1];
    [encoder setBuffer:record.meanG offset:0 atIndex:2];
    [encoder setBuffer:record.meanB offset:0 atIndex:3];
    [encoder setBuffer:record.carrierY offset:0 atIndex:4];
    [encoder setBuffer:record.carrierMax offset:0 atIndex:5];
    [encoder setBuffer:record.carrierMin offset:0 atIndex:6];
    [encoder setBuffer:record.neutrality offset:0 atIndex:7];
    [encoder setBuffer:(useCandidate2 ? record.body2 : record.body) offset:0 atIndex:8];
    [encoder setBuffer:(useCandidate2 ? record.positive2 : record.positive) offset:0 atIndex:9];
    [encoder setBuffer:(useCandidate2 ? record.negative2 : record.negative) offset:0 atIndex:10];
    [encoder setBuffer:(useCandidate2 ? record.boundary2 : record.boundary) offset:0 atIndex:11];
    [encoder setBuffer:(useCandidate2 ? record.congruence2 : record.congruence) offset:0 atIndex:12];
    [encoder setBuffer:(useCandidate2 ? record.confidence2 : record.confidence) offset:0 atIndex:13];
    [encoder setBuffer:(useCandidate2 ? record.signal2 : record.signal) offset:0 atIndex:14];
    [encoder setBuffer:uniformBuffer offset:0 atIndex:15];
    NSUInteger shadeWidth = ctx.glossProjectionSurfaceShadePipeline.maxTotalThreadsPerThreadgroup;
    if (shadeWidth == 0) shadeWidth = 64;
    shadeWidth = std::max<NSUInteger>(
        1,
        std::min<NSUInteger>(16, static_cast<NSUInteger>(std::sqrt(static_cast<double>(shadeWidth)))));
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(outputSurfaceWidth),
                                         static_cast<NSUInteger>(outputSurfaceHeight),
                                         1)
       threadsPerThreadgroup:MTLSizeMake(shadeWidth, shadeWidth, 1)];
    [encoder endEncoding];
  }
  return true;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
bool renderGlossProjectionSurfaceFromCache(
    const GlossFieldCache& cache,
    const GlossProjectionSurfaceRequest& projectionRequest,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError.empty() ? "metal-context-unavailable" : localError;
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = [context().queue commandBuffer];
  if (commandBuffer == nil) {
    if (error) *error = "metal-gloss-projection-surface-command-buffer-failed";
    return false;
  }
  GlossFieldResidentRecord record{};
  if (!resolveGlossFieldRecord(cache, 0, 0, false, &record)) {
    if (error) *error = "metal-gloss-field-cache-missing";
    return false;
  }
  if (!encodeGlossProjectionSurfaceFromCacheOnCommandBuffer(commandBuffer,
                                                            &record,
                                                            cache,
                                                            projectionRequest,
                                                            outputSurfaceId,
                                                            outputSurfaceWidth,
                                                            outputSurfaceHeight,
                                                            outputSurfacePixelFormat,
                                                            error)) {
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* cbError = commandBuffer.error;
  if (cbError != nil) {
    if (error) *error = [[cbError localizedDescription] UTF8String];
    return false;
  }
  return true;
}
#endif

bool encodeGlossProjectionSurfaceFromCache(
    const FrameSubmission& submission,
    const GlossFieldCache& cache,
    const GlossProjectionSurfaceRequest& projectionRequest,
    uint32_t outputSurfaceId,
    int outputSurfaceWidth,
    int outputSurfaceHeight,
    int outputSurfacePixelFormat,
    std::string* error) {
  if (!validateResidentDerivedOwnerForSubmission(
          submission, cache.cacheId, cache.ownerCompositorId, error)) {
    return false;
  }
  id<MTLCommandBuffer> commandBuffer = nil;
  if (!commandBufferForFrameSubmission(submission, &commandBuffer, error)) {
    return false;
  }
  if (!validatePlotSurfaceOwnerForSubmission(
          submission, outputSurfaceId, error)) {
    return false;
  }
  GlossFieldResidentRecord stagedRecord{};
  std::string residentError;
  if (!resolveGlossFieldRecordForSubmission(
          submission, cache, true, &stagedRecord, &residentError)) {
    if (error) {
      *error = residentError.empty()
                   ? "metal-gloss-field-cache-missing"
                   : residentError;
    }
    return false;
  }
  return encodeGlossProjectionSurfaceFromCacheOnCommandBuffer(commandBuffer,
                                                              &stagedRecord,
                                                              cache,
                                                              projectionRequest,
                                                              outputSurfaceId,
                                                              outputSurfaceWidth,
                                                              outputSurfaceHeight,
                                                              outputSurfacePixelFormat,
                                                              error);
}

}  // namespace ChromaspaceMetal
