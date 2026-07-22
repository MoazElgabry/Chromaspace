#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "ChromaspaceSourceExchangeV2.h"
#include "ChromaspaceMetal.h"
#include "ChromaspaceSourceProducerClient.h"
#include "ChromaspaceSourceProducerClientState.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <poll.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace {

constexpr int kRelayBootstrapFd = 3;
constexpr int kBootstrapTimeoutMilliseconds = 10000;
constexpr uint64_t kReleasePollNanoseconds = 50 * NSEC_PER_MSEC;
constexpr uint64_t kPublishRetryNanoseconds = 25 * NSEC_PER_MSEC;
constexpr uint32_t kMaximumPublishTransportRetries = 8;
const void* kProducerQueueSpecific = &kProducerQueueSpecific;

NSXPCInterface* hostBootstrapInterface() {
  NSXPCInterface* interface =
      [NSXPCInterface
          interfaceWithProtocol:
              @protocol(ChromaspaceSourceExchangeHostBootstrapProtocol)];
  [interface setClasses:[NSSet setWithObjects:[NSData class], nil]
            forSelector:@selector(redeemProducerRelayWithToken:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface
      setClasses:[NSSet
                     setWithObjects:[NSXPCListenerEndpoint class], nil]
      forSelector:@selector(redeemProducerRelayWithToken:withReply:)
      argumentIndex:0
      ofReply:YES];
  return interface;
}

NSXPCInterface* producerRelayInterface() {
  NSXPCInterface* interface =
      [NSXPCInterface
          interfaceWithProtocol:
              @protocol(ChromaspaceSourceExchangeProducerRelayProtocol)];
  NSSet<Class>* joinClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeProducerJoinRequest class],
                     [NSString class],
                     nil];
  NSSet<Class>* leaseClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeProducerLease class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* packetClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangePacket class],
                     [ChromaspaceSourceExchangeMetadata class],
                     [MTLSharedTextureHandle class],
                     [MTLSharedEventHandle class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* releaseClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeReleaseBatch class],
                     [ChromaspaceSourceExchangeReleaseEvent class],
                     [NSArray class],
                     [NSData class],
                     [NSString class],
                     nil];
  [interface setClasses:joinClasses
            forSelector:@selector(joinProducer:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:leaseClasses
            forSelector:@selector(joinProducer:withReply:)
          argumentIndex:0
                ofReply:YES];
  [interface setClasses:packetClasses
            forSelector:@selector(publishPacket:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:releaseClasses
            forSelector:@selector(fetchProducerReleasesAfterOrdinal:maximumEvents:withReply:)
          argumentIndex:0
                ofReply:YES];
  return interface;
}

std::string relayExecutablePath() {
  Dl_info info{};
  if (dladdr(
          reinterpret_cast<const void*>(&kProducerQueueSpecific),
          &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  std::string binary(info.dli_fname);
  const size_t separator = binary.find_last_of('/');
  if (separator == std::string::npos) return {};
  return binary.substr(0, separator + 1) +
         "Chromaspace_SourceExchangeProducerRelay";
}

bool setCloseOnExec(int fd) {
  int flags = fcntl(fd, F_GETFD);
  return flags >= 0 && fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0;
}

bool readBootstrapToken(
    int fd,
    std::atomic<bool>* stopping,
    uint8_t* token,
    size_t tokenBytes) {
  size_t received = 0;
  int waitedMilliseconds = 0;
  while (received < tokenBytes && waitedMilliseconds <
                                      kBootstrapTimeoutMilliseconds) {
    if (stopping->load(std::memory_order_acquire)) return false;
    pollfd descriptor{};
    descriptor.fd = fd;
    descriptor.events = POLLIN;
    int pollResult = poll(&descriptor, 1, 50);
    if (pollResult < 0 && errno == EINTR) continue;
    if (pollResult < 0 ||
        (descriptor.revents & (POLLERR | POLLNVAL)) != 0) {
      return false;
    }
    if (pollResult == 0) {
      waitedMilliseconds += 50;
      continue;
    }
    ssize_t count =
        read(fd, token + received, tokenBytes - received);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    received += static_cast<size_t>(count);
  }
  return received == tokenBytes;
}

bool capabilityFromData(
    NSData* data,
    ChromaspaceSourceExchange::Capability* out) {
  if (out) out->fill(0);
  if (!out || data.length !=
                  ChromaspaceSourceExchangeCapabilityBytes) {
    return false;
  }
  std::memcpy(out->data(), data.bytes, out->size());
  return true;
}

void terminateAndReap(pid_t* child) {
  if (child == nullptr || *child <= 0) return;
  const pid_t pid = *child;
  *child = -1;
  (void)kill(pid, SIGTERM);
  dispatch_async(
      dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
      });
}

ChromaspaceSourceExchange::ProducerLeaseSnapshot leaseSnapshot(
    ChromaspaceSourceExchangeProducerLease* lease) {
  ChromaspaceSourceExchange::ProducerLeaseSnapshot value{};
  value.protocolMajor = lease.protocolMajor;
  value.protocolMinor = lease.protocolMinor;
  (void)capabilityFromData(lease.sessionCapability, &value.capability);
  value.viewerGeneration = lease.viewerGeneration;
  value.senderId = lease.senderId.UTF8String ?: "";
  value.senderGeneration = lease.senderGeneration;
  value.deviceRegistryId = lease.deviceRegistryId;
  value.pixelFormatMask = lease.pixelFormatMask;
  value.maximumWidth = lease.maximumWidth;
  value.maximumHeight = lease.maximumHeight;
  value.maximumSurfaceBytes = lease.maximumSurfaceBytes;
  value.maximumRetainedBytes = lease.maximumRetainedBytes;
  value.maximumSlots = lease.maximumSlots;
  value.supportsSharedEvents = lease.supportsSharedEvents;
  return value;
}

ChromaspaceSourceExchange::PublicationKey releaseKey(
    ChromaspaceSourceExchangeReleaseEvent* event) {
  ChromaspaceSourceExchange::PublicationKey key{};
  key.senderId = event.senderId.UTF8String ?: "";
  key.senderGeneration = event.senderGeneration;
  key.sequence = event.sequence;
  key.slotIndex = event.slotIndex;
  key.slotGeneration = event.slotGeneration;
  return key;
}

}  // namespace

@class ChromaspaceSourceProducerClientImpl;

namespace {

struct SourceCompletionContext {
  __strong ChromaspaceSourceProducerClientImpl* client = nil;
  ChromaspaceSourceExchange::PublicationKey key;
};

void sourceProxyCompleted(void* context, bool completedSuccessfully);

NSData* capabilityData(
    const ChromaspaceSourceExchange::Capability& capability) {
  return [NSData dataWithBytes:capability.data()
                       length:capability.size()];
}

MTLPixelFormat sourcePixelFormat(uint32_t pixelFormat) {
  if (pixelFormat == 0) return MTLPixelFormatRGBA16Float;
  if (pixelFormat == 1) return MTLPixelFormatRGBA32Float;
  return MTLPixelFormatInvalid;
}

uint64_t sourceBytesPerPixel(uint32_t pixelFormat) {
  if (pixelFormat == 0) return 8;
  if (pixelFormat == 1) return 16;
  return 0;
}

}  // namespace

@interface ChromaspaceSourceProducerClientImpl : NSObject {
 @private
  dispatch_queue_t _queue;
  dispatch_source_t _releaseTimer;
  NSXPCConnection* _bootstrapConnection;
  NSXPCConnection* _relayConnection;
  pid_t _relayPid;
  std::atomic<bool> _stopping;
  std::atomic<bool> _started;
  std::mutex _stateMutex;
  ChromaspaceSourceExchange::ProducerClientState _state;
  std::optional<ChromaspaceSourceExchange::ProducerLeaseSnapshot>
      _activeLease;
  ChromaspaceSourceExchange::SourceProducerClientSnapshot _snapshot;
  NSString* _senderId;
  uint64_t _senderGeneration;
  uint64_t _deviceRegistryId;
  id<MTLCommandQueue> _commandQueue;
  NSArray<id<MTLTexture>>* _ringTextures;
  NSArray<MTLSharedTextureHandle*>* _ringTextureHandles;
  id<MTLSharedEvent> _sharedEvent;
  MTLSharedEventHandle* _sharedEventHandle;
  uint32_t _ringWidth;
  uint32_t _ringHeight;
  uint32_t _ringPixelFormat;
  uint32_t _requestedWidth;
  uint32_t _requestedHeight;
  uint32_t _requestedPixelFormat;
  uint64_t _nextReadyValue;
  NSMutableArray* _packetsBySlot;
  BOOL _publishInFlight;
  uint64_t _publishAttemptEpoch;
  uint32_t _publishTransportRetryCount;
  BOOL _releaseFetchInFlight;
  BOOL _bootstrapRedemptionPending;
}

- (instancetype)initWithConfiguration:
    (const ChromaspaceSourceExchange::SourceProducerClientConfiguration&)
        configuration;
- (void)start;
- (ChromaspaceSourceExchange::SourceProducerBindResult)
    bindCommandQueue:(void*)metalCommandQueue;
- (void)shutdownSynchronously;
- (ChromaspaceSourceExchange::SourceProducerClientSnapshot)snapshot;
- (void)attemptJoin;
- (void)prepareResources:
    (const ChromaspaceSourceExchange::SourceProducerResourceShape&)shape;
- (ChromaspaceSourceExchange::SourceProducerSubmitResult)
    tryEnqueueFrame:
        (const ChromaspaceSourceExchange::SourceProducerFrameRequest&)request;
- (void)sourceProxyDidComplete:
            (const ChromaspaceSourceExchange::PublicationKey&)key
                         success:(BOOL)success;
- (void)enqueueSourceProxyCompletion:
            (const ChromaspaceSourceExchange::PublicationKey&)key
                              success:(BOOL)success;
- (void)prepareRequestedResourcesOnQueue;
- (void)publishNext;

@end

@implementation ChromaspaceSourceProducerClientImpl

- (instancetype)initWithConfiguration:
    (const ChromaspaceSourceExchange::SourceProducerClientConfiguration&)
        configuration {
  self = [super init];
  if (self) {
    _queue = dispatch_queue_create(
        "com.chromaspace.SourceProducerClient", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_set_specific(
        _queue,
        kProducerQueueSpecific,
        const_cast<void*>(kProducerQueueSpecific),
        nullptr);
    _relayPid = -1;
    _stopping.store(false, std::memory_order_relaxed);
    _started.store(false, std::memory_order_relaxed);
    _senderId =
        [NSString stringWithUTF8String:configuration.senderId.c_str()];
    _senderGeneration = configuration.senderGeneration;
    _commandQueue =
        (__bridge id<MTLCommandQueue>)configuration.metalCommandQueue;
    _deviceRegistryId = _commandQueue.device.registryID;
    _nextReadyValue = 1;
    _packetsBySlot =
        [NSMutableArray arrayWithCapacity:
                            ChromaspaceSourceExchangeMaximumSlots];
    for (NSUInteger index = 0;
         index < ChromaspaceSourceExchangeMaximumSlots;
         ++index) {
      [_packetsBySlot addObject:[NSNull null]];
    }
    _snapshot.deviceRegistryId = _deviceRegistryId;
  }
  return self;
}

- (void)setHealth:
            (ChromaspaceSourceExchange::SourceProducerClientHealth)health
       diagnostic:(const std::string&)diagnostic {
  std::lock_guard<std::mutex> lock(_stateMutex);
  _snapshot.health = health;
  _snapshot.diagnostic = diagnostic;
  _snapshot.releaseFetchCursor = _state.releaseFetchCursor();
  _snapshot.acknowledgedReleaseOrdinal =
      _state.acknowledgedReleaseOrdinal();
  _snapshot.livePublicationCount = _state.livePublicationCount();
}

- (ChromaspaceSourceExchange::SourceProducerBindResult)
    bindCommandQueue:(void*)metalCommandQueue {
  using BindResult =
      ChromaspaceSourceExchange::SourceProducerBindResult;
  if (metalCommandQueue == nullptr ||
      _stopping.load(std::memory_order_acquire)) {
    return BindResult::Invalid;
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)metalCommandQueue;
  id<MTLDevice> device = queue.device;
  if (queue == nil || device == nil || device.registryID == 0) {
    return BindResult::Invalid;
  }

  bool newlyBound = false;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_deviceRegistryId != 0 &&
        _deviceRegistryId != device.registryID) {
      return BindResult::DeviceMismatch;
    }
    if (_commandQueue == nil) {
      _commandQueue = queue;
      _deviceRegistryId = device.registryID;
      _snapshot.deviceRegistryId = _deviceRegistryId;
      newlyBound = true;
    }
  }
  if (newlyBound) {
    [self start];
    return BindResult::Bound;
  }
  return BindResult::AlreadyBound;
}

- (void)fail:(const std::string&)diagnostic {
  if (_stopping.exchange(true, std::memory_order_acq_rel)) return;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    (void)_state.invalidateTransport();
    _activeLease.reset();
    _ringTextures = nil;
    _ringTextureHandles = nil;
    _sharedEvent = nil;
    _sharedEventHandle = nil;
    _ringWidth = 0;
    _ringHeight = 0;
    _ringPixelFormat = 0;
    _snapshot.health =
        ChromaspaceSourceExchange::SourceProducerClientHealth::Failed;
    _snapshot.diagnostic = diagnostic;
    _snapshot.releaseFetchCursor = 0;
    _snapshot.acknowledgedReleaseOrdinal = 0;
    _snapshot.livePublicationCount = 0;
  }
  for (NSUInteger index = 0; index < _packetsBySlot.count; ++index) {
    _packetsBySlot[index] = [NSNull null];
  }
  _publishInFlight = NO;
  _bootstrapRedemptionPending = NO;
  if (_releaseTimer != nil) {
    dispatch_source_cancel(_releaseTimer);
    _releaseTimer = nil;
  }
  [_bootstrapConnection invalidate];
  _bootstrapConnection = nil;
  [_relayConnection invalidate];
  _relayConnection = nil;
  terminateAndReap(&_relayPid);
}

- (void)prepareResources:
    (const ChromaspaceSourceExchange::SourceProducerResourceShape&)shape {
  const ChromaspaceSourceExchange::SourceProducerResourceShape requested =
      shape;
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  dispatch_async(_queue, ^{
    if (strongSelf->_stopping.load(std::memory_order_acquire)) return;
    strongSelf->_requestedWidth = requested.width;
    strongSelf->_requestedHeight = requested.height;
    strongSelf->_requestedPixelFormat = requested.pixelFormat;
    [strongSelf prepareRequestedResourcesOnQueue];
  });
}

- (void)prepareRequestedResourcesOnQueue {
  if (_stopping.load(std::memory_order_acquire) ||
      _requestedWidth == 0 || _requestedHeight == 0 ||
      _requestedPixelFormat > 1) {
    return;
  }

  ChromaspaceSourceExchange::ProducerLeaseSnapshot lease{};
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (!_activeLease.has_value() ||
        _state.livePublicationCount() != 0) {
      return;
    }
    lease = *_activeLease;
    if (_ringTextures != nil &&
        _ringWidth == _requestedWidth &&
        _ringHeight == _requestedHeight &&
        _ringPixelFormat == _requestedPixelFormat) {
      return;
    }
  }

  const uint32_t formatMask =
      _requestedPixelFormat == 0
          ? ChromaspaceSourceExchange::kPixelFormatRGBA16Float
          : ChromaspaceSourceExchange::kPixelFormatRGBA32Float;
  const uint64_t bytesPerPixel =
      sourceBytesPerPixel(_requestedPixelFormat);
  const bool multiplicationSafe =
      bytesPerPixel != 0 &&
      _requestedWidth <=
          std::numeric_limits<uint64_t>::max() / bytesPerPixel &&
      static_cast<uint64_t>(_requestedWidth) * bytesPerPixel <=
          std::numeric_limits<uint64_t>::max() / _requestedHeight;
  const uint64_t byteSize =
      multiplicationSafe
          ? static_cast<uint64_t>(_requestedWidth) *
                _requestedHeight * bytesPerPixel
          : 0;
  if (lease.maximumSlots !=
          ChromaspaceSourceExchangeMaximumSlots ||
      _requestedWidth > lease.maximumWidth ||
      _requestedHeight > lease.maximumHeight ||
      (lease.pixelFormatMask & formatMask) == 0 ||
      !lease.supportsSharedEvents || !multiplicationSafe ||
      byteSize > lease.maximumSurfaceBytes ||
      byteSize * ChromaspaceSourceExchangeMaximumSlots >
          lease.maximumRetainedBytes) {
    [self setHealth:
              ChromaspaceSourceExchange::SourceProducerClientHealth::Ready
         diagnostic:"resource-shape-not-supported-by-lease"];
    return;
  }

  std::string pipelineError;
  if (!ChromaspaceMetal::prepareSourceProxyPipeline(
          (__bridge void*)_commandQueue, &pipelineError)) {
    [self fail:pipelineError.empty()
                   ? "source-proxy-pipeline-prepare-failed"
                   : pipelineError];
    return;
  }

  id<MTLDevice> device = _commandQueue.device;
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor
          texture2DDescriptorWithPixelFormat:
              sourcePixelFormat(_requestedPixelFormat)
                                     width:_requestedWidth
                                    height:_requestedHeight
                                 mipmapped:NO];
  descriptor.storageMode = MTLStorageModePrivate;
  descriptor.usage =
      MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  NSMutableArray<id<MTLTexture>>* textures =
      [NSMutableArray
          arrayWithCapacity:ChromaspaceSourceExchangeMaximumSlots];
  NSMutableArray<MTLSharedTextureHandle*>* handles =
      [NSMutableArray
          arrayWithCapacity:ChromaspaceSourceExchangeMaximumSlots];
  for (NSUInteger index = 0;
       index < ChromaspaceSourceExchangeMaximumSlots;
       ++index) {
    id<MTLTexture> texture =
        [device newSharedTextureWithDescriptor:descriptor];
    MTLSharedTextureHandle* handle =
        [texture newSharedTextureHandle];
    if (texture == nil || handle == nil) {
      [self fail:"shared-texture-ring-allocation-failed"];
      return;
    }
    [textures addObject:texture];
    [handles addObject:handle];
  }
  id<MTLSharedEvent> event = [device newSharedEvent];
  MTLSharedEventHandle* eventHandle =
      [event newSharedEventHandle];
  if (event == nil || eventHandle == nil) {
    [self fail:"shared-event-allocation-failed"];
    return;
  }

  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_stopping.load(std::memory_order_acquire) ||
        !_activeLease.has_value() ||
        _activeLease->capability != lease.capability ||
        _activeLease->viewerGeneration != lease.viewerGeneration ||
        _state.livePublicationCount() != 0 ||
        _requestedWidth == 0 || _requestedHeight == 0) {
      return;
    }
    _ringTextures = [textures copy];
    _ringTextureHandles = [handles copy];
    _sharedEvent = event;
    _sharedEventHandle = eventHandle;
    _ringWidth = _requestedWidth;
    _ringHeight = _requestedHeight;
    _ringPixelFormat = _requestedPixelFormat;
  }
  [self setHealth:
            ChromaspaceSourceExchange::SourceProducerClientHealth::Ready
       diagnostic:"ready-resident-ring"];
}

- (ChromaspaceSourceExchange::SourceProducerSubmitResult)
    tryEnqueueFrame:
        (const ChromaspaceSourceExchange::SourceProducerFrameRequest&)request {
  using SubmitResult =
      ChromaspaceSourceExchange::SourceProducerSubmitResult;
  if (_stopping.load(std::memory_order_acquire) ||
      request.metalCommandQueue == nullptr ||
      request.sourceMetalBuffer == nullptr ||
      request.sourceWidth <= 0 || request.sourceHeight <= 0 ||
      request.sourceRowBytes == 0 || request.output.width == 0 ||
      request.output.height == 0 || request.output.pixelFormat > 1 ||
      request.contentHash == 0) {
    return SubmitResult::InvalidRequest;
  }
  id<MTLCommandQueue> submissionQueue =
      (__bridge id<MTLCommandQueue>)request.metalCommandQueue;
  if (submissionQueue == nil || submissionQueue.device == nil) {
    return SubmitResult::InvalidRequest;
  }

  std::unique_lock<std::mutex> lock(
      _stateMutex, std::try_to_lock);
  if (!lock.owns_lock()) return SubmitResult::LockBusy;
  if (_snapshot.health !=
          ChromaspaceSourceExchange::SourceProducerClientHealth::Ready ||
      !_activeLease.has_value()) {
    return SubmitResult::NotReady;
  }
  if (submissionQueue.device.registryID != _deviceRegistryId) {
    return SubmitResult::InvalidRequest;
  }
  if (_ringTextures == nil || _ringTextureHandles == nil ||
      _sharedEvent == nil || _sharedEventHandle == nil ||
      _ringWidth != request.output.width ||
      _ringHeight != request.output.height ||
      _ringPixelFormat != request.output.pixelFormat) {
    return SubmitResult::ResourceShapeMismatch;
  }
  if (_nextReadyValue == 0 ||
      _nextReadyValue == std::numeric_limits<uint64_t>::max()) {
    return SubmitResult::Failed;
  }

  const uint64_t bytesPerPixel =
      sourceBytesPerPixel(request.output.pixelFormat);
  if (request.output.width >
      std::numeric_limits<uint64_t>::max() / bytesPerPixel) {
    return SubmitResult::InvalidRequest;
  }
  const uint64_t bytesPerRow =
      static_cast<uint64_t>(request.output.width) * bytesPerPixel;
  if (request.output.height >
      std::numeric_limits<uint64_t>::max() / bytesPerRow) {
    return SubmitResult::InvalidRequest;
  }

  ChromaspaceSourceExchange::ProducerPublicationSpec spec{};
  spec.sequence = request.sequence;
  spec.width = request.output.width;
  spec.height = request.output.height;
  spec.pixelFormat = request.output.pixelFormat;
  spec.bytesPerRow = bytesPerRow;
  spec.byteSize = bytesPerRow * request.output.height;
  spec.readyValue = _nextReadyValue++;
  spec.contentHash = request.contentHash;
  spec.semantics = request.semantics;
  ChromaspaceSourceExchange::Publication publication{};
  const ChromaspaceSourceExchange::ResultCode reserved =
      _state.reserve(spec, &publication);
  if (reserved ==
      ChromaspaceSourceExchange::ResultCode::SlotBusy) {
    return SubmitResult::BackPressure;
  }
  if (reserved != ChromaspaceSourceExchange::ResultCode::Accepted ||
      publication.key.slotIndex >= _ringTextures.count) {
    return SubmitResult::Failed;
  }

  auto completion = std::unique_ptr<SourceCompletionContext>(
      new (std::nothrow) SourceCompletionContext);
  if (!completion) {
    (void)_state.cancel(publication.key);
    return SubmitResult::BackPressure;
  }
  completion->client = self;
  completion->key = publication.key;
  id<MTLTexture> texture =
      _ringTextures[publication.key.slotIndex];
  std::string encodeError;
  const bool enqueued =
      ChromaspaceMetal::enqueueSourceProxyToSharedTexture(
          request.sourceMetalBuffer,
          request.sourceWidth,
          request.sourceHeight,
          request.sourceRowBytes,
          request.sourceOriginX,
          request.sourceOriginY,
          static_cast<int>(request.output.width),
          static_cast<int>(request.output.height),
          static_cast<int>(request.output.pixelFormat),
          (__bridge void*)texture,
          (__bridge void*)_sharedEvent,
          publication.readyValue,
          (__bridge void*)submissionQueue,
          sourceProxyCompleted,
          completion.get(),
          &encodeError);
  if (!enqueued) {
    (void)_state.cancel(publication.key);
    _snapshot.diagnostic =
        encodeError.empty() ? "source-proxy-encode-failed"
                            : encodeError;
    return SubmitResult::EncodeFailed;
  }
  (void)completion.release();
  _snapshot.livePublicationCount = _state.livePublicationCount();
  return SubmitResult::Enqueued;
}

- (void)sourceProxyDidComplete:
            (const ChromaspaceSourceExchange::PublicationKey&)key
                         success:(BOOL)success {
  if (_stopping.load(std::memory_order_acquire)) return;
  if (!success) {
    [self fail:"source-proxy-command-buffer-failed"];
    return;
  }

  ChromaspaceSourceExchange::Publication publication{};
  MTLSharedTextureHandle* textureHandle = nil;
  MTLSharedEventHandle* eventHandle = nil;
  bool staleCompletion = false;
  bool slotCollision = false;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (!_activeLease.has_value() ||
        key.slotIndex >= _ringTextureHandles.count) {
      staleCompletion = true;
    } else {
      const std::optional<ChromaspaceSourceExchange::Publication> live =
          _state.publication(key);
      if (!live.has_value() ||
          _state.markSendReady(key) !=
              ChromaspaceSourceExchange::ResultCode::Accepted) {
        staleCompletion = true;
      } else {
        publication = *live;
        slotCollision =
            _packetsBySlot[key.slotIndex] != [NSNull null];
        textureHandle = _ringTextureHandles[key.slotIndex];
        eventHandle = _sharedEventHandle;
      }
    }
  }
  if (staleCompletion) return;
  if (slotCollision) {
    [self fail:"publication-packet-slot-collision"];
    return;
  }

  NSString* sender =
      [NSString stringWithUTF8String:publication.key.senderId.c_str()];
  const auto& semantics = publication.semantics;
  NSString* colorPrimaries =
      [NSString stringWithUTF8String:semantics.colorPrimaries.c_str()];
  NSString* transferFunction =
      [NSString stringWithUTF8String:semantics.transferFunction.c_str()];
  NSString* coverage =
      semantics.coverage ==
              ChromaspaceSourceExchange::SourceCoverage::FullSource
          ? @"full"
          : @"partial";
  ChromaspaceSourceExchangeMetadata* metadata =
      [[ChromaspaceSourceExchangeMetadata alloc]
          initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                  protocolMinor:ChromaspaceSourceExchangeProtocolMinor
              sessionCapability:capabilityData(publication.capability)
                       senderId:sender
               senderGeneration:publication.key.senderGeneration
                       sequence:publication.key.sequence
                      slotIndex:publication.key.slotIndex
                 slotGeneration:publication.key.slotGeneration
                     readyValue:publication.readyValue
               deviceRegistryId:publication.deviceRegistryId
                          width:publication.width
                         height:publication.height
                    pixelFormat:publication.pixelFormat
                    bytesPerRow:publication.bytesPerRow
                       byteSize:publication.byteSize
                    contentHash:publication.contentHash
                       sourceX:semantics.sourceX
                       sourceY:semantics.sourceY
                   sourceWidth:semantics.sourceWidth
                  sourceHeight:semantics.sourceHeight
                      sampledX:semantics.sampledX
                      sampledY:semantics.sampledY
                  sampledWidth:semantics.sampledWidth
                 sampledHeight:semantics.sampledHeight
                  authoritative:semantics.authoritative ? YES : NO
                       coverage:coverage
           identityStripPresent:semantics.identityStripPresent ? YES : NO
                   identityCube:semantics.identityCube ? YES : NO
                   identityRamp:semantics.identityRamp ? YES : NO
             identityResolution:semantics.identityResolution
             identityBandHeight:semantics.identityBandHeight
                 identityCubeY1:semantics.identityCubeY1
                 identityCubeY2:semantics.identityCubeY2
                 identityRampY1:semantics.identityRampY1
                 identityRampY2:semantics.identityRampY2
                 colorPrimaries:colorPrimaries
               transferFunction:transferFunction];
  ChromaspaceSourceExchangePacket* packet =
      [[ChromaspaceSourceExchangePacket alloc]
          initWithMetadata:metadata
             textureHandle:textureHandle
               eventHandle:eventHandle];
  NSError* validationError = nil;
  if (packet == nil || ![packet validate:&validationError]) {
    [self fail:"publication-packet-construction-failed"];
    return;
  }
  _packetsBySlot[key.slotIndex] = packet;
  [self publishNext];
}

- (void)enqueueSourceProxyCompletion:
            (const ChromaspaceSourceExchange::PublicationKey&)key
                              success:(BOOL)success {
  const ChromaspaceSourceExchange::PublicationKey completedKey = key;
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  dispatch_async(_queue, ^{
    [strongSelf sourceProxyDidComplete:completedKey success:success];
  });
}

- (void)publishNext {
  if (_stopping.load(std::memory_order_acquire) ||
      _relayConnection == nil || _publishInFlight) {
    return;
  }
  std::optional<ChromaspaceSourceExchange::Publication> publication;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    publication = _state.beginNextPublish();
  }
  if (!publication.has_value()) return;
  const uint32_t slot = publication->key.slotIndex;
  if (slot >= _packetsBySlot.count ||
      _packetsBySlot[slot] == [NSNull null]) {
    {
      std::lock_guard<std::mutex> lock(_stateMutex);
      (void)_state.publishTransportFailed(publication->key);
    }
    [self fail:"publication-packet-missing"];
    return;
  }
  ChromaspaceSourceExchangePacket* packet = _packetsBySlot[slot];
  const ChromaspaceSourceExchange::PublicationKey key =
      publication->key;
  _publishInFlight = YES;
  const uint64_t attemptEpoch = ++_publishAttemptEpoch;

  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  id<ChromaspaceSourceExchangeProducerRelayProtocol> relay =
      [_relayConnection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_stopping.load(
                      std::memory_order_acquire) ||
                  !strongSelf->_publishInFlight ||
                  strongSelf->_publishAttemptEpoch != attemptEpoch) {
                return;
              }
              bool retry = false;
              {
                std::lock_guard<std::mutex> lock(
                    strongSelf->_stateMutex);
                retry =
                    strongSelf->_state.publishTransportFailed(key) ==
                    ChromaspaceSourceExchange::ResultCode::Accepted;
              }
              strongSelf->_publishInFlight = NO;
              if (!retry) {
                [strongSelf publishNext];
                return;
              }
              if (++strongSelf->_publishTransportRetryCount >
                  kMaximumPublishTransportRetries) {
                [strongSelf fail:"publish-transport-retry-exhausted"];
                return;
              }
              dispatch_after(
                  dispatch_time(
                      DISPATCH_TIME_NOW,
                      kPublishRetryNanoseconds),
                  strongSelf->_queue, ^{
                    [strongSelf publishNext];
                  });
            });
          }];
  [relay
      publishPacket:packet
          withReply:^(ChromaspaceSourceExchangeStatus status,
                      NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_stopping.load(
                      std::memory_order_acquire) ||
                  !strongSelf->_publishInFlight ||
                  strongSelf->_publishAttemptEpoch != attemptEpoch) {
                return;
              }
              if (status !=
                  ChromaspaceSourceExchangeStatusAccepted) {
                [strongSelf fail:"publication-rejected"];
                return;
              }
              bool acceptedOrAlreadyReleased = false;
              {
                std::lock_guard<std::mutex> lock(
                    strongSelf->_stateMutex);
                const ChromaspaceSourceExchange::ResultCode result =
                    strongSelf->_state.publishAccepted(key);
                acceptedOrAlreadyReleased =
                    result ==
                        ChromaspaceSourceExchange::ResultCode::Accepted ||
                    result ==
                        ChromaspaceSourceExchange::ResultCode::Stale;
                strongSelf->_snapshot.livePublicationCount =
                    strongSelf->_state.livePublicationCount();
              }
              if (!acceptedOrAlreadyReleased) {
                [strongSelf fail:"publication-state-rejected"];
                return;
              }
              strongSelf->_publishInFlight = NO;
              strongSelf->_publishTransportRetryCount = 0;
              [strongSelf publishNext];
            });
          }];
}

- (BOOL)spawnRelayAndReadToken:(NSData**)outToken {
  if (outToken) *outToken = nil;
  const std::string executable = relayExecutablePath();
  if (executable.empty() || access(executable.c_str(), X_OK) != 0) {
    return NO;
  }
  int pipeFds[2] = {-1, -1};
  if (pipe(pipeFds) != 0 ||
      !setCloseOnExec(pipeFds[0]) ||
      !setCloseOnExec(pipeFds[1])) {
    if (pipeFds[0] >= 0) close(pipeFds[0]);
    if (pipeFds[1] >= 0) close(pipeFds[1]);
    return NO;
  }
  if (pipeFds[1] == kRelayBootstrapFd) {
    const int replacement =
        fcntl(pipeFds[1], F_DUPFD_CLOEXEC, kRelayBootstrapFd + 1);
    if (replacement < 0) {
      close(pipeFds[0]);
      close(pipeFds[1]);
      return NO;
    }
    close(pipeFds[1]);
    pipeFds[1] = replacement;
  }

  posix_spawn_file_actions_t actions;
  if (posix_spawn_file_actions_init(&actions) != 0) {
    close(pipeFds[0]);
    close(pipeFds[1]);
    return NO;
  }
  bool actionsReady =
      posix_spawn_file_actions_addclose(&actions, pipeFds[0]) == 0;
  actionsReady =
      actionsReady &&
      posix_spawn_file_actions_adddup2(
          &actions, pipeFds[1], kRelayBootstrapFd) == 0 &&
      posix_spawn_file_actions_addclose(&actions, pipeFds[1]) == 0;
  char fdText[16] = {};
  std::snprintf(fdText, sizeof(fdText), "%d", kRelayBootstrapFd);
  char* argv[] = {
      const_cast<char*>(executable.c_str()),
      const_cast<char*>("--bootstrap-fd"),
      fdText,
      nullptr,
  };
  pid_t child = -1;
  posix_spawnattr_t attributes;
  const bool attributesInitialized =
      posix_spawnattr_init(&attributes) == 0;
  bool attributesReady = attributesInitialized;
#if defined(POSIX_SPAWN_CLOEXEC_DEFAULT)
  if (attributesReady) {
    attributesReady =
        posix_spawnattr_setflags(
            &attributes, POSIX_SPAWN_CLOEXEC_DEFAULT) == 0;
  }
#endif
  int spawnResult =
      actionsReady && attributesReady
          ? posix_spawn(
                &child,
                executable.c_str(),
                &actions,
                &attributes,
                argv,
                environ)
          : EINVAL;
  if (attributesInitialized) posix_spawnattr_destroy(&attributes);
  posix_spawn_file_actions_destroy(&actions);
  close(pipeFds[1]);
  if (spawnResult != 0 || child <= 0) {
    close(pipeFds[0]);
    return NO;
  }
  _relayPid = child;

  uint8_t tokenBytes[ChromaspaceSourceExchangeBootstrapTokenBytes] = {};
  const bool readOk = readBootstrapToken(
      pipeFds[0],
      &_stopping,
      tokenBytes,
      sizeof(tokenBytes));
  close(pipeFds[0]);
  if (!readOk) return NO;
  if (outToken) {
    *outToken =
        [NSData dataWithBytes:tokenBytes length:sizeof(tokenBytes)];
  }
  return YES;
}

- (void)startReleasePump {
  if (_releaseTimer != nil) return;
  _releaseTimer =
      dispatch_source_create(
          DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
  dispatch_source_set_timer(
      _releaseTimer,
      dispatch_time(DISPATCH_TIME_NOW, kReleasePollNanoseconds),
      kReleasePollNanoseconds,
      5 * NSEC_PER_MSEC);
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  dispatch_source_set_event_handler(_releaseTimer, ^{
    [strongSelf pollReleases];
  });
  dispatch_resume(_releaseTimer);
}

- (void)pollReleases {
  if (_stopping.load(std::memory_order_acquire) ||
      _relayConnection == nil || _releaseFetchInFlight) {
    return;
  }
  uint64_t cursor = 0;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_snapshot.health !=
        ChromaspaceSourceExchange::SourceProducerClientHealth::Ready) {
      return;
    }
    cursor = _state.releaseFetchCursor();
  }
  _releaseFetchInFlight = YES;
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  id<ChromaspaceSourceExchangeProducerRelayProtocol> relay =
      [_relayConnection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              strongSelf->_releaseFetchInFlight = NO;
              [strongSelf fail:"release-fetch-transport-failed"];
            });
          }];
  [relay
      fetchProducerReleasesAfterOrdinal:cursor
                          maximumEvents:
                              ChromaspaceSourceExchangeMaximumReleaseEvents
                              withReply:
                                  ^(ChromaspaceSourceExchangeReleaseBatch* batch,
                                    ChromaspaceSourceExchangeStatus status,
                                    NSError* error) {
                                    (void)error;
                                    dispatch_async(strongSelf->_queue, ^{
                                      [strongSelf
                                          consumeReleaseBatch:batch
                                                       status:status];
                                    });
                                  }];
}

- (void)consumeReleaseBatch:
            (ChromaspaceSourceExchangeReleaseBatch*)batch
                       status:(ChromaspaceSourceExchangeStatus)status {
  if (_stopping.load(std::memory_order_acquire)) return;
  if (status == ChromaspaceSourceExchangeStatusNoNewPublication) {
    _releaseFetchInFlight = NO;
    return;
  }
  NSError* validationError = nil;
  if (status != ChromaspaceSourceExchangeStatusAccepted ||
      batch == nil || ![batch validate:&validationError]) {
    [self fail:"invalid-release-batch"];
    return;
  }
  ChromaspaceSourceExchange::ProducerReleaseBatch value{};
  if (!capabilityFromData(batch.sessionCapability, &value.capability)) {
    [self fail:"invalid-release-capability"];
    return;
  }
  value.senderId = batch.senderId.UTF8String ?: "";
  value.senderGeneration = batch.senderGeneration;
  value.throughOrdinal = batch.throughOrdinal;
  for (ChromaspaceSourceExchangeReleaseEvent* event in batch.events) {
    value.events.push_back(
        ChromaspaceSourceExchange::ProducerReleaseEvent{
            event.ordinal, releaseKey(event)});
  }
  uint64_t acknowledgementOrdinal = 0;
  bool releaseAccepted = false;
  std::vector<ChromaspaceSourceExchange::PublicationKey> released;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    ChromaspaceSourceExchange::ProducerClientTransition transition =
        _state.applyReleaseBatch(value);
    releaseAccepted = transition.accepted();
    if (releaseAccepted) {
      acknowledgementOrdinal =
          transition.releaseAcknowledgementOrdinal;
      released = std::move(transition.released);
      _snapshot.releaseFetchCursor = _state.releaseFetchCursor();
      _snapshot.livePublicationCount = _state.livePublicationCount();
    }
  }
  if (!releaseAccepted) {
    [self fail:"release-ordering-failed"];
    return;
  }
  for (const auto& key : released) {
    if (key.slotIndex < _packetsBySlot.count) {
      _packetsBySlot[key.slotIndex] = [NSNull null];
    }
  }
  [self prepareRequestedResourcesOnQueue];
  [self publishNext];
  if (acknowledgementOrdinal == 0) return;
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  id<ChromaspaceSourceExchangeProducerRelayProtocol> relay =
      [_relayConnection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              [strongSelf fail:"release-ack-transport-failed"];
            });
          }];
  [relay
      acknowledgeProducerReleasesThroughOrdinal:acknowledgementOrdinal
                                      withReply:
                                          ^(ChromaspaceSourceExchangeStatus ackStatus,
                                            NSError* error) {
                                            (void)error;
                                            dispatch_async(
                                                strongSelf->_queue, ^{
                                                  if (strongSelf->
                                                          _stopping.load(
                                                              std::memory_order_acquire)) {
                                                    return;
                                                  }
                                                  if (ackStatus !=
                                                      ChromaspaceSourceExchangeStatusAccepted) {
                                                    [strongSelf
                                                        fail:"release-ack-rejected"];
                                                    return;
                                                  }
                                                  bool accepted = false;
                                                  {
                                                    std::lock_guard<std::mutex>
                                                        lock(
                                                            strongSelf->
                                                                _stateMutex);
                                                    accepted =
                                                        strongSelf->
                                                            _state
                                                                .releaseAcknowledgementAccepted(
                                                                    acknowledgementOrdinal) ==
                                                        ChromaspaceSourceExchange::
                                                            ResultCode::
                                                                Accepted;
                                                    if (accepted) {
                                                      strongSelf->
                                                          _snapshot
                                                              .acknowledgedReleaseOrdinal =
                                                          strongSelf->
                                                              _state
                                                                  .acknowledgedReleaseOrdinal();
                                                    }
                                                  }
                                                  if (!accepted) {
                                                    [strongSelf
                                                        fail:"release-ack-state-failed"];
                                                    return;
                                                  }
                                                  strongSelf->
                                                      _releaseFetchInFlight =
                                                      NO;
                                                });
                                          }];
}

- (void)connectRelayEndpoint:(NSXPCListenerEndpoint*)endpoint {
  if (_stopping.load(std::memory_order_acquire) || endpoint == nil) return;
  _relayConnection =
      [[NSXPCConnection alloc] initWithListenerEndpoint:endpoint];
  _relayConnection.remoteObjectInterface = producerRelayInterface();
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  _relayConnection.invalidationHandler = ^{
    dispatch_async(strongSelf->_queue, ^{
      if (!strongSelf->_stopping.load(std::memory_order_acquire)) {
        [strongSelf fail:"relay-invalidated"];
      }
    });
  };
  _relayConnection.interruptionHandler =
      _relayConnection.invalidationHandler;
  [_relayConnection resume];
  [self attemptJoin];
}

- (void)attemptJoin {
  if (_stopping.load(std::memory_order_acquire) ||
      _relayConnection == nil) {
    return;
  }
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  ChromaspaceSourceExchangeProducerJoinRequest* request =
      [[ChromaspaceSourceExchangeProducerJoinRequest alloc]
          initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                  protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                       senderId:_senderId
               senderGeneration:_senderGeneration
               deviceRegistryId:_deviceRegistryId];
  id<ChromaspaceSourceExchangeProducerRelayProtocol> relay =
      [_relayConnection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              [strongSelf fail:"producer-join-transport-failed"];
            });
          }];
  [relay
      joinProducer:request
         withReply:
             ^(ChromaspaceSourceExchangeProducerLease* lease,
               ChromaspaceSourceExchangeStatus status,
               NSError* error) {
               dispatch_async(strongSelf->_queue, ^{
                 if (strongSelf->_stopping.load(
                         std::memory_order_acquire)) {
                   return;
                 }
                 NSError* validationError = nil;
                 if (status !=
                     ChromaspaceSourceExchangeStatusAccepted) {
                   const BOOL routeMissing =
                       [error.domain
                           isEqualToString:
                               ChromaspaceSourceExchangeErrorDomain] &&
                       error.code ==
                           ChromaspaceSourceExchangeErrorSessionMissing;
                   if (!routeMissing) {
                     [strongSelf fail:"producer-join-rejected"];
                     return;
                   }
                   [strongSelf
                       setHealth:
                           ChromaspaceSourceExchange::
                               SourceProducerClientHealth::
                                   AwaitingViewerRoute
                      diagnostic:"viewer-route-unavailable"];
                   dispatch_after(
                       dispatch_time(
                           DISPATCH_TIME_NOW,
                           250 * NSEC_PER_MSEC),
                       strongSelf->_queue, ^{
                         [strongSelf attemptJoin];
                   });
                   return;
                 }
                 if (lease == nil ||
                     ![lease validate:&validationError] ||
                     lease.deviceRegistryId !=
                         strongSelf->_deviceRegistryId) {
                   [strongSelf fail:"producer-lease-invalid"];
                   return;
                 }
                 ChromaspaceSourceExchange::ProducerClientTransition
                     installed{};
                 const ChromaspaceSourceExchange::ProducerLeaseSnapshot
                     installedLease = leaseSnapshot(lease);
                 {
                   std::lock_guard<std::mutex> lock(
                       strongSelf->_stateMutex);
                   installed = strongSelf->_state.installLease(
                       installedLease);
                   if (installed.accepted()) {
                     strongSelf->_activeLease = installedLease;
                     if (installed.leaseChanged) {
                       strongSelf->_ringTextures = nil;
                       strongSelf->_ringTextureHandles = nil;
                       strongSelf->_sharedEvent = nil;
                       strongSelf->_sharedEventHandle = nil;
                       strongSelf->_ringWidth = 0;
                       strongSelf->_ringHeight = 0;
                     }
                   }
                 }
                 if (!installed.accepted()) {
                   [strongSelf fail:"producer-lease-invalid"];
                   return;
                 }
                 [strongSelf
                     setHealth:
                         ChromaspaceSourceExchange::
                             SourceProducerClientHealth::Ready
                    diagnostic:"ready"];
                 [strongSelf startReleasePump];
                 if (installed.leaseChanged) {
                   for (NSUInteger index = 0;
                        index < strongSelf->_packetsBySlot.count;
                        ++index) {
                     strongSelf->_packetsBySlot[index] =
                         [NSNull null];
                   }
                 }
                 [strongSelf prepareRequestedResourcesOnQueue];
               });
             }];
}

- (void)startOnQueue {
  if (_stopping.load(std::memory_order_acquire) ||
      _senderId.length == 0 || _senderGeneration == 0 ||
      _deviceRegistryId == 0) {
    [self fail:"invalid-producer-configuration"];
    return;
  }
  NSData* token = nil;
  if (![self spawnRelayAndReadToken:&token] || token == nil) {
    [self fail:"relay-bootstrap-failed"];
    return;
  }
  _bootstrapConnection =
      [[NSXPCConnection alloc]
          initWithMachServiceName:
              ChromaspaceSourceExchangeBootstrapMachServiceName
                      options:0];
  _bootstrapConnection.remoteObjectInterface =
      hostBootstrapInterface();
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  _bootstrapConnection.invalidationHandler = ^{
    dispatch_async(strongSelf->_queue, ^{
      if (strongSelf->_relayConnection == nil &&
          !strongSelf->_stopping.load(std::memory_order_acquire)) {
        [strongSelf fail:"bootstrap-invalidated"];
      }
    });
  };
  _bootstrapConnection.interruptionHandler =
      _bootstrapConnection.invalidationHandler;
  [_bootstrapConnection resume];
  _bootstrapRedemptionPending = YES;
  dispatch_after(
      dispatch_time(
          DISPATCH_TIME_NOW,
          kBootstrapTimeoutMilliseconds * NSEC_PER_MSEC),
      _queue, ^{
        if (strongSelf->_bootstrapRedemptionPending &&
            !strongSelf->_stopping.load(std::memory_order_acquire)) {
          [strongSelf fail:"bootstrap-redemption-timeout"];
        }
      });
  id<ChromaspaceSourceExchangeHostBootstrapProtocol> bootstrap =
      [_bootstrapConnection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              strongSelf->_bootstrapRedemptionPending = NO;
              [strongSelf fail:"bootstrap-redemption-failed"];
            });
          }];
  [bootstrap
      redeemProducerRelayWithToken:token
                         withReply:
                             ^(NSXPCListenerEndpoint* endpoint,
                               ChromaspaceSourceExchangeStatus status,
                               NSError* error) {
                               (void)error;
                               dispatch_async(strongSelf->_queue, ^{
                                 strongSelf->
                                     _bootstrapRedemptionPending = NO;
                                 if (strongSelf->_stopping.load(
                                         std::memory_order_acquire)) {
                                   return;
                                 }
                                 if (status !=
                                         ChromaspaceSourceExchangeStatusAccepted ||
                                     endpoint == nil) {
                                   [strongSelf
                                       fail:"bootstrap-token-rejected"];
                                   return;
                                 }
                                 [strongSelf->_bootstrapConnection
                                     invalidate];
                                 strongSelf->_bootstrapConnection = nil;
                                 [strongSelf
                                     connectRelayEndpoint:endpoint];
                               });
                             }];
}

- (void)start {
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_commandQueue == nil || _deviceRegistryId == 0) return;
  }
  bool expected = false;
  if (!_started.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  [self
      setHealth:
          ChromaspaceSourceExchange::SourceProducerClientHealth::Starting
     diagnostic:"starting"];
  ChromaspaceSourceProducerClientImpl* strongSelf = self;
  dispatch_async(_queue, ^{
    [strongSelf startOnQueue];
  });
}

- (void)shutdownOnQueue {
  _stopping.store(true, std::memory_order_release);
  _bootstrapRedemptionPending = NO;
  if (_releaseTimer != nil) {
    dispatch_source_cancel(_releaseTimer);
    _releaseTimer = nil;
  }
  [_bootstrapConnection invalidate];
  _bootstrapConnection = nil;
  [_relayConnection invalidate];
  _relayConnection = nil;
  terminateAndReap(&_relayPid);
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    (void)_state.invalidateTransport();
    _activeLease.reset();
    _ringTextures = nil;
    _ringTextureHandles = nil;
    _sharedEvent = nil;
    _sharedEventHandle = nil;
    _snapshot.health =
        ChromaspaceSourceExchange::SourceProducerClientHealth::Stopped;
    _snapshot.diagnostic = "stopped";
    _snapshot.releaseFetchCursor = 0;
    _snapshot.acknowledgedReleaseOrdinal = 0;
    _snapshot.livePublicationCount = 0;
  }
  for (NSUInteger index = 0; index < _packetsBySlot.count; ++index) {
    _packetsBySlot[index] = [NSNull null];
  }
  _publishInFlight = NO;
}

- (void)shutdownSynchronously {
  _stopping.store(true, std::memory_order_release);
  if (dispatch_get_specific(kProducerQueueSpecific) != nullptr) {
    [self shutdownOnQueue];
    return;
  }
  dispatch_sync(_queue, ^{
    [self shutdownOnQueue];
  });
}

- (ChromaspaceSourceExchange::SourceProducerClientSnapshot)snapshot {
  std::lock_guard<std::mutex> lock(_stateMutex);
  return _snapshot;
}

@end

namespace {

void sourceProxyCompleted(void* context, bool completedSuccessfully) {
  std::unique_ptr<SourceCompletionContext> completion(
      static_cast<SourceCompletionContext*>(context));
  if (!completion || completion->client == nil) return;
  [completion->client
      enqueueSourceProxyCompletion:completion->key
                           success:completedSuccessfully ? YES : NO];
}

}  // namespace

namespace ChromaspaceSourceExchange {

struct SourceProducerClient {
  __strong ChromaspaceSourceProducerClientImpl* implementation = nil;
};

SourceProducerClient* createSourceProducerClient(
    const SourceProducerClientConfiguration& configuration) {
  auto* client = new SourceProducerClient;
  client->implementation =
      [[ChromaspaceSourceProducerClientImpl alloc]
          initWithConfiguration:configuration];
  if (client->implementation == nil) {
    delete client;
    return nullptr;
  }
  return client;
}

SourceProducerBindResult bindSourceProducerMetalCommandQueue(
    SourceProducerClient* client,
    void* metalCommandQueue) {
  if (client == nullptr || client->implementation == nil) {
    return SourceProducerBindResult::Failed;
  }
  return [client->implementation bindCommandQueue:metalCommandQueue];
}

void startSourceProducerClient(SourceProducerClient* client) {
  if (client == nullptr) return;
  [client->implementation start];
}

void prepareSourceProducerResources(
    SourceProducerClient* client,
    const SourceProducerResourceShape& shape) {
  if (client == nullptr || client->implementation == nil) return;
  [client->implementation prepareResources:shape];
}

SourceProducerSubmitResult tryEnqueueSourceProducerFrame(
    SourceProducerClient* client,
    const SourceProducerFrameRequest& request) {
  if (client == nullptr || client->implementation == nil) {
    return SourceProducerSubmitResult::Failed;
  }
  return [client->implementation tryEnqueueFrame:request];
}

void destroySourceProducerClient(SourceProducerClient* client) {
  if (client == nullptr) return;
  [client->implementation shutdownSynchronously];
  client->implementation = nil;
  delete client;
}

SourceProducerClientSnapshot sourceProducerClientSnapshot(
    const SourceProducerClient* client) {
  if (client == nullptr || client->implementation == nil) {
    return {};
  }
  return [client->implementation snapshot];
}

}  // namespace ChromaspaceSourceExchange
