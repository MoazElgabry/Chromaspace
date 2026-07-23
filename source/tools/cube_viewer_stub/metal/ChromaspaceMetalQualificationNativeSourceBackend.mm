#include "ChromaspaceMetalQualificationNativeSourceBackend.h"

#import <Metal/Metal.h>

#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace ChromaspaceMetalQualification {
namespace {

constexpr NSUInteger kFixtureWidth = 640u;
constexpr NSUInteger kFixtureHeight = 360u;
constexpr NSUInteger kBytesPerPixel = 16u;  // RGBA32F.
constexpr NSUInteger kRowAlignment = 256u;
constexpr uint32_t kMaximumInFlight = 3u;

constexpr const char* kReady = "qualification-native-source-fixture-ready";
constexpr const char* kInvalidBackend =
    "qualification-native-source-fixture-unavailable";
constexpr const char* kInvalidArguments =
    "qualification-native-source-fixture-invalid-arguments";
constexpr const char* kDeviceMismatch =
    "qualification-native-source-fixture-device-mismatch";
constexpr const char* kGenerationRegression =
    "qualification-native-source-fixture-generation-regression";
constexpr const char* kInFlightLimit =
    "qualification-native-source-fixture-in-flight-limit";
constexpr const char* kBufferAllocation =
    "qualification-native-source-fixture-buffer-allocation-failed";
constexpr const char* kTextureAllocation =
    "qualification-native-source-fixture-texture-allocation-failed";
constexpr const char* kEventAllocation =
    "qualification-native-source-fixture-event-allocation-failed";
constexpr const char* kCommandAllocation =
    "qualification-native-source-fixture-command-allocation-failed";
constexpr const char* kBlitAllocation =
    "qualification-native-source-fixture-blit-allocation-failed";
constexpr const char* kHandleAllocation =
    "qualification-native-source-fixture-share-handle-failed";
constexpr const char* kImportFailed =
    "qualification-native-source-fixture-import-failed";
constexpr const char* kCommandException =
    "qualification-native-source-fixture-command-exception";
constexpr const char* kObjCException =
    "qualification-native-source-fixture-objective-c-exception";
constexpr const char* kCounterOverflow =
    "qualification-native-source-fixture-counter-overflow";

void clearSource(ChromaspaceMetal::ImportedSourceTexture* source) noexcept {
  if (source == nullptr) return;
  try {
    *source = ChromaspaceMetal::ImportedSourceTexture{};
  } catch (...) {
    source->sourceId = 0u;
    source->senderId.clear();
    source->deviceRegistryId = 0u;
    source->senderGeneration = 0u;
    source->sequence = 0u;
    source->slotIndex = 0u;
    source->slotGeneration = 0u;
    source->readyValue = 0u;
    source->contentHash = 0u;
    source->width = 0;
    source->height = 0;
    source->pixelFormat = 0;
    source->bytesPerRow = 0u;
    source->byteSize = 0u;
  }
}

void setErrorNoThrow(std::string* output, const char* value) noexcept {
  if (output == nullptr) return;
  try {
    *output = value != nullptr ? value : kInvalidBackend;
  } catch (...) {
  }
}

}  // namespace

struct NativeSourceFixtureBackend::Impl final {
  struct Lifetime final {
    std::atomic<uint32_t> inFlight{0u};
  };

  // The command-buffer completion handler can outlive the backend object.  A
  // shared token gives both the synchronous error path and the asynchronous
  // callback one idempotent release operation without retaining `this` or
  // `Impl` in the block.
  struct FlightToken final {
    explicit FlightToken(std::shared_ptr<Lifetime> value) noexcept
        : lifetime(std::move(value)) {}

    void release() noexcept {
      bool expected = false;
      if (released.compare_exchange_strong(expected, true,
                                            std::memory_order_acq_rel,
                                            std::memory_order_relaxed)) {
        if (lifetime != nullptr) {
          lifetime->inFlight.fetch_sub(1u, std::memory_order_relaxed);
        }
      }
    }

    std::shared_ptr<Lifetime> lifetime;
    std::atomic<bool> released{false};
  };

  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  uint64_t deviceRegistryId = 0u;
  std::shared_ptr<Lifetime> lifetime;
  std::atomic<bool> ready{false};
  std::atomic<bool> failed{false};
  std::atomic<uint32_t> createCount{0u};
  std::atomic<uint32_t> retireCount{0u};
  std::atomic<uint32_t> failureCount{0u};
  std::atomic<uint64_t> lastGeneration{0u};
  std::atomic<const char*> diagnostic{kInvalidBackend};
};

NativeSourceFixtureBackend::NativeSourceFixtureBackend() noexcept {
  backend_.context = this;
  backend_.create = &NativeSourceFixtureBackend::createCallback;
  backend_.retire = &NativeSourceFixtureBackend::retireCallback;
  try {
    impl_ = std::make_unique<Impl>();
    impl_->lifetime = std::make_shared<Impl::Lifetime>();
    @try {
      @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device == nil) {
          impl_->failed.store(true, std::memory_order_release);
          impl_->diagnostic.store(kInvalidBackend, std::memory_order_release);
          return;
        }
        impl_->queue = [impl_->device newCommandQueue];
        if (impl_->queue == nil) {
          impl_->failed.store(true, std::memory_order_release);
          impl_->diagnostic.store(kInvalidBackend, std::memory_order_release);
          return;
        }
        impl_->deviceRegistryId = impl_->device.registryID;
        if (impl_->deviceRegistryId == 0u) {
          impl_->failed.store(true, std::memory_order_release);
          impl_->diagnostic.store(kInvalidBackend, std::memory_order_release);
          return;
        }
        impl_->ready.store(true, std::memory_order_release);
        impl_->diagnostic.store(kReady, std::memory_order_release);
      }
    } @catch (id) {
      impl_->failed.store(true, std::memory_order_release);
      impl_->diagnostic.store(kObjCException, std::memory_order_release);
    }
  } catch (...) {
    if (impl_ != nullptr) {
      impl_->failed.store(true, std::memory_order_release);
      impl_->diagnostic.store(kInvalidBackend, std::memory_order_release);
    }
  }
}

NativeSourceFixtureBackend::~NativeSourceFixtureBackend() = default;

bool NativeSourceFixtureBackend::ready() const noexcept {
  return impl_ != nullptr && impl_->ready.load(std::memory_order_acquire) &&
         !impl_->failed.load(std::memory_order_acquire);
}

bool NativeSourceFixtureBackend::failed() const noexcept {
  return impl_ == nullptr || impl_->failed.load(std::memory_order_acquire);
}

const char* NativeSourceFixtureBackend::diagnostic() const noexcept {
  return impl_ != nullptr
             ? impl_->diagnostic.load(std::memory_order_acquire)
             : kInvalidBackend;
}

bool NativeSourceFixtureBackend::fail(const char* diagnostic,
                                      std::string* error) noexcept {
  if (impl_ == nullptr) {
    setErrorNoThrow(error, kInvalidBackend);
    return false;
  }
  impl_->diagnostic.store(diagnostic != nullptr ? diagnostic : kInvalidBackend,
                          std::memory_order_release);
  setErrorNoThrow(error, diagnostic != nullptr ? diagnostic : kInvalidBackend);
  uint32_t old = impl_->failureCount.load(std::memory_order_relaxed);
  while (true) {
    if (old == std::numeric_limits<uint32_t>::max()) {
      impl_->failed.store(true, std::memory_order_release);
      impl_->diagnostic.store(kCounterOverflow, std::memory_order_release);
      setErrorNoThrow(error, kCounterOverflow);
      return false;
    }
    if (impl_->failureCount.compare_exchange_weak(
            old, old + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      break;
    }
  }
  return false;
}

bool NativeSourceFixtureBackend::createCallback(
    void* context,
    const std::string& senderId,
    uint64_t deviceRegistryId,
    uint64_t sourceGeneration,
    ChromaspaceMetal::ImportedSourceTexture* outSource,
    std::string* error) noexcept {
  auto* self = static_cast<NativeSourceFixtureBackend*>(context);
  if (self == nullptr) {
    clearSource(outSource);
    setErrorNoThrow(error, kInvalidBackend);
    return false;
  }
  try {
    @try {
      return self->createInternal(senderId, deviceRegistryId, sourceGeneration,
                                  outSource, error);
    } @catch (id) {
      clearSource(outSource);
      return self->fail(kObjCException, error);
    }
  } catch (...) {
    clearSource(outSource);
    return self->fail(kCommandException, error);
  }
}

void NativeSourceFixtureBackend::retireCallback(void* context,
                                                uint64_t sourceId) noexcept {
  auto* self = static_cast<NativeSourceFixtureBackend*>(context);
  if (self == nullptr) return;
  try {
    @try {
      self->retireInternal(sourceId);
    } @catch (id) {
      (void)self->fail(kObjCException, nullptr);
    }
  } catch (...) {
    (void)self->fail(kCommandException, nullptr);
  }
}

bool NativeSourceFixtureBackend::createInternal(
    const std::string& senderId,
    uint64_t deviceRegistryId,
    uint64_t sourceGeneration,
    ChromaspaceMetal::ImportedSourceTexture* outSource,
    std::string* error) {
  clearSource(outSource);
  if (impl_ == nullptr || !ready()) {
    return fail(impl_ != nullptr ? impl_->diagnostic.load(std::memory_order_acquire)
                                 : kInvalidBackend,
                error);
  }
  if (outSource == nullptr || senderId.empty() || senderId.size() > 64u ||
      deviceRegistryId == 0u || sourceGeneration == 0u) {
    return fail(kInvalidArguments, error);
  }
  if (deviceRegistryId != impl_->deviceRegistryId) {
    return fail(kDeviceMismatch, error);
  }
  const uint64_t last = impl_->lastGeneration.load(std::memory_order_acquire);
  if (sourceGeneration <= last) {
    return fail(kGenerationRegression, error);
  }
  auto lifetime = impl_->lifetime;
  if (lifetime == nullptr) return fail(kInvalidBackend, error);
  uint32_t inFlight = lifetime->inFlight.load(std::memory_order_relaxed);
  while (true) {
    if (inFlight >= kMaximumInFlight) return fail(kInFlightLimit, error);
    if (lifetime->inFlight.compare_exchange_weak(
            inFlight, inFlight + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      break;
    }
  }
  bool flightReserved = true;
  std::shared_ptr<Impl::FlightToken> flightToken;
  auto releaseFlight = [&]() noexcept {
    if (flightToken != nullptr) {
      flightToken->release();
    } else if (flightReserved) {
      lifetime->inFlight.fetch_sub(1u, std::memory_order_relaxed);
      flightReserved = false;
    }
  };

  try {
    flightToken = std::make_shared<Impl::FlightToken>(lifetime);
  } catch (...) {
    releaseFlight();
    return fail(kCommandException, error);
  }
  flightReserved = false;

  uint32_t createCount = impl_->createCount.load(std::memory_order_relaxed);
  while (true) {
    if (createCount == std::numeric_limits<uint32_t>::max()) {
      releaseFlight();
      return fail(kCounterOverflow, error);
    }
    if (impl_->createCount.compare_exchange_weak(
            createCount, createCount + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      break;
    }
  }

  constexpr NSUInteger rowBytes =
      ((kFixtureWidth * kBytesPerPixel + (kRowAlignment - 1u)) /
       kRowAlignment) *
      kRowAlignment;
  constexpr NSUInteger byteSize = rowBytes * kFixtureHeight;
  const uint64_t readyValue = sourceGeneration;
  const uint64_t contentHash =
      0x9E3779B97F4A7C15ull ^ (sourceGeneration * 0xD6E8FEB86659FD93ull);

  @autoreleasepool {
    id<MTLBuffer> fixtureBuffer =
        [impl_->device newBufferWithLength:byteSize
                                   options:MTLResourceStorageModeShared];
    if (fixtureBuffer == nil || [fixtureBuffer contents] == nullptr) {
      releaseFlight();
      return fail(kBufferAllocation, error);
    }
    float* pixels = static_cast<float*>([fixtureBuffer contents]);
    const float generationPhase =
        static_cast<float>(sourceGeneration % 1024u) / 1024.0f;
    const NSUInteger floatsPerRow = rowBytes / sizeof(float);
    for (NSUInteger y = 0u; y < kFixtureHeight; ++y) {
      float* row = pixels + y * floatsPerRow;
      for (NSUInteger x = 0u; x < kFixtureWidth; ++x) {
        const NSUInteger offset = x * 4u;
        row[offset + 0u] =
            (static_cast<float>(x) + 0.5f) / static_cast<float>(kFixtureWidth);
        row[offset + 1u] =
            (static_cast<float>(y) + 0.5f) / static_cast<float>(kFixtureHeight);
        row[offset + 2u] = generationPhase;
        row[offset + 3u] = 1.0f;
      }
    }

    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                      width:kFixtureWidth
                                     height:kFixtureHeight
                                  mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> texture =
        [impl_->device newSharedTextureWithDescriptor:descriptor];
    if (texture == nil) {
      releaseFlight();
      return fail(kTextureAllocation, error);
    }
    id<MTLSharedEvent> readyEvent = [impl_->device newSharedEvent];
    if (readyEvent == nil) {
      releaseFlight();
      return fail(kEventAllocation, error);
    }
    id<MTLCommandBuffer> commandBuffer = [impl_->queue commandBuffer];
    if (commandBuffer == nil) {
      releaseFlight();
      return fail(kCommandAllocation, error);
    }
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (blit == nil) {
      releaseFlight();
      return fail(kBlitAllocation, error);
    }
    [blit copyFromBuffer:fixtureBuffer
            sourceOffset:0u
       sourceBytesPerRow:rowBytes
     sourceBytesPerImage:byteSize
              sourceSize:MTLSizeMake(kFixtureWidth, kFixtureHeight, 1u)
               toTexture:texture
        destinationSlice:0u
        destinationLevel:0u
       destinationOrigin:MTLOriginMake(0u, 0u, 0u)];
    [blit endEncoding];
    [commandBuffer encodeSignalEvent:readyEvent value:readyValue];

    MTLSharedTextureHandle* textureHandle = [texture newSharedTextureHandle];
    MTLSharedEventHandle* eventHandle = [readyEvent newSharedEventHandle];
    if (textureHandle == nil || eventHandle == nil) {
      releaseFlight();
      return fail(kHandleAllocation, error);
    }

    ChromaspaceMetal::SharedSourceImportRequest request{};
    request.sharedTextureHandle = (__bridge void*)textureHandle;
    request.sharedEventHandle = (__bridge void*)eventHandle;
    request.senderId = senderId;
    request.deviceRegistryId = deviceRegistryId;
    request.senderGeneration = sourceGeneration;
    request.sequence = sourceGeneration;
    request.slotIndex = static_cast<uint32_t>((sourceGeneration - 1u) % 3u);
    request.slotGeneration = sourceGeneration;
    request.readyValue = readyValue;
    request.contentHash = contentHash == 0u ? 1u : contentHash;
    request.width = static_cast<int>(kFixtureWidth);
    request.height = static_cast<int>(kFixtureHeight);
    request.pixelFormat = 1;
    request.bytesPerRow = rowBytes;
    request.byteSize = byteSize;
    request.semantics.sourceX = 0;
    request.semantics.sourceY = 0;
    request.semantics.sourceWidth = static_cast<uint32_t>(kFixtureWidth);
    request.semantics.sourceHeight = static_cast<uint32_t>(kFixtureHeight);
    request.semantics.sampledX = 0;
    request.semantics.sampledY = 0;
    request.semantics.sampledWidth = static_cast<uint32_t>(kFixtureWidth);
    request.semantics.sampledHeight = static_cast<uint32_t>(kFixtureHeight);
    request.semantics.coverage =
        ChromaspaceSourceExchange::SourceCoverage::FullSource;
    request.semantics.authoritative = true;
    request.semantics.colorPrimaries = "sRGB";
    request.semantics.transferFunction = "linear";

    ChromaspaceMetal::ImportedSourceTexture imported{};
    std::string importError;
    if (!ChromaspaceMetal::importSharedSourceTexture(request, &imported,
                                                     &importError) ||
        imported.sourceId == 0u || imported.senderId != senderId ||
        imported.deviceRegistryId != deviceRegistryId ||
        imported.senderGeneration != sourceGeneration ||
        imported.sequence != sourceGeneration ||
        imported.slotIndex != request.slotIndex ||
        imported.slotGeneration != sourceGeneration ||
        imported.readyValue != readyValue ||
        imported.contentHash != request.contentHash ||
        imported.width != static_cast<int>(kFixtureWidth) ||
        imported.height != static_cast<int>(kFixtureHeight) ||
        imported.pixelFormat != 1 || imported.bytesPerRow != rowBytes ||
        imported.byteSize != byteSize ||
        !(imported.semantics == request.semantics)) {
      if (imported.sourceId != 0u) {
        ChromaspaceMetal::releaseImportedSourceTexture(imported.sourceId);
      }
      releaseFlight();
      if (!importError.empty()) setErrorNoThrow(error, importError.c_str());
      return fail(kImportFailed, error);
    }

    try {
      @try {
        auto retainedBuffer = fixtureBuffer;
        auto retainedTexture = texture;
        auto retainedEvent = readyEvent;
        auto retainedFlight = flightToken;
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
          (void)completed;
          (void)retainedBuffer;
          (void)retainedTexture;
          (void)retainedEvent;
          retainedFlight->release();
        }];
        [commandBuffer commit];
      } @catch (id) {
        ChromaspaceMetal::releaseImportedSourceTexture(imported.sourceId);
        releaseFlight();
        return fail(kObjCException, error);
      }
    } catch (...) {
      ChromaspaceMetal::releaseImportedSourceTexture(imported.sourceId);
      releaseFlight();
      return fail(kCommandException, error);
    }
    flightReserved = false;
    impl_->lastGeneration.store(sourceGeneration, std::memory_order_release);
    *outSource = imported;
    setErrorNoThrow(error, kReady);
    return true;
  }
}

void NativeSourceFixtureBackend::retireInternal(uint64_t sourceId) {
  if (impl_ == nullptr || sourceId == 0u) {
    (void)fail(kInvalidArguments, nullptr);
    return;
  }
  ChromaspaceMetal::releaseImportedSourceTexture(sourceId);
  uint32_t old = impl_->retireCount.load(std::memory_order_relaxed);
  while (true) {
    if (old == std::numeric_limits<uint32_t>::max()) {
      (void)fail(kCounterOverflow, nullptr);
      return;
    }
    if (impl_->retireCount.compare_exchange_weak(
            old, old + 1u, std::memory_order_relaxed,
            std::memory_order_relaxed)) {
      return;
    }
  }
}

NativeSourceFixtureSnapshot NativeSourceFixtureBackend::snapshot() const
    noexcept {
  NativeSourceFixtureSnapshot output{};
  if (impl_ == nullptr) {
    output.failed = true;
    return output;
  }
  output.ready = ready();
  output.failed = failed();
  output.deviceRegistryId = impl_->deviceRegistryId;
  output.createCount = impl_->createCount.load(std::memory_order_acquire);
  output.retireCount = impl_->retireCount.load(std::memory_order_acquire);
  output.failureCount = impl_->failureCount.load(std::memory_order_acquire);
  output.inFlightCount = impl_->lifetime != nullptr
                             ? impl_->lifetime->inFlight.load(
                                   std::memory_order_acquire)
                             : 0u;
  output.lastGeneration =
      impl_->lastGeneration.load(std::memory_order_acquire);
  return output;
}

}  // namespace ChromaspaceMetalQualification
