#import <Foundation/Foundation.h>
#import <Security/Security.h>

#import "../../../src/metal/ChromaspaceSourceExchangeV2.h"
#include "../../../src/metal/ChromaspaceSourceViewerClientState.h"
#include "ChromaspaceSourceViewerClient.h"

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>

namespace {

constexpr uint64_t kAcquirePollNanoseconds = 16 * NSEC_PER_MSEC;
constexpr uint64_t kAcknowledgementRetryNanoseconds =
    25 * NSEC_PER_MSEC;
constexpr uint64_t kControlReplyDeadlineNanoseconds =
    2 * NSEC_PER_SEC;
constexpr uint64_t kAcquireReplyDeadlineNanoseconds =
    1 * NSEC_PER_SEC;
constexpr uint32_t kMaximumAcknowledgementTransportRetries = 8;
const void* kViewerQueueSpecific = &kViewerQueueSpecific;

NSXPCInterface* viewerBrokerInterface() {
  NSXPCInterface* interface =
      [NSXPCInterface
          interfaceWithProtocol:
              @protocol(ChromaspaceSourceExchangeBrokerProtocol)];
  NSSet<Class>* viewerClasses = [NSSet
      setWithObjects:
          [ChromaspaceSourceExchangeViewerRegistration class],
          [NSData class],
          nil];
  NSSet<Class>* routeClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeViewerRoute class],
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
  NSSet<Class>* acknowledgementClasses = [NSSet
      setWithObjects:
          [ChromaspaceSourceExchangeAcknowledgement class],
          [NSData class],
          [NSString class],
          nil];
  [interface setClasses:viewerClasses
            forSelector:@selector(registerViewer:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:routeClasses
            forSelector:@selector(bindViewerRoute:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface
      setClasses:packetClasses
      forSelector:@selector(acquireLatestForSession:senderId:afterSequence:withReply:)
      argumentIndex:0
      ofReply:YES];
  [interface setClasses:acknowledgementClasses
            forSelector:@selector(acknowledge:withReply:)
          argumentIndex:0
                ofReply:NO];
  return interface;
}

NSData* capabilityData(
    const ChromaspaceSourceExchange::Capability& capability) {
  return [NSData dataWithBytes:capability.data()
                       length:capability.size()];
}

bool capabilityFromData(
    NSData* data,
    ChromaspaceSourceExchange::Capability* outCapability) {
  if (outCapability) outCapability->fill(0);
  if (outCapability == nullptr ||
      data.length != ChromaspaceSourceExchangeCapabilityBytes) {
    return false;
  }
  std::memcpy(
      outCapability->data(), data.bytes, outCapability->size());
  return true;
}

ChromaspaceSourceExchange::Publication publicationFromPacket(
    ChromaspaceSourceExchangePacket* packet) {
  ChromaspaceSourceExchange::Publication result{};
  ChromaspaceSourceExchangeMetadata* metadata = packet.metadata;
  (void)capabilityFromData(
      metadata.sessionCapability, &result.capability);
  result.key.senderId = metadata.senderId.UTF8String ?: "";
  result.key.senderGeneration = metadata.senderGeneration;
  result.key.sequence = metadata.sequence;
  result.key.slotIndex = metadata.slotIndex;
  result.key.slotGeneration = metadata.slotGeneration;
  result.deviceRegistryId = metadata.deviceRegistryId;
  result.width = metadata.width;
  result.height = metadata.height;
  result.pixelFormat = metadata.pixelFormat;
  result.bytesPerRow = metadata.bytesPerRow;
  result.byteSize = metadata.byteSize;
  result.readyValue = metadata.readyValue;
  result.contentHash = metadata.contentHash;
  result.semantics.sourceX = metadata.sourceX;
  result.semantics.sourceY = metadata.sourceY;
  result.semantics.sourceWidth = metadata.sourceWidth;
  result.semantics.sourceHeight = metadata.sourceHeight;
  result.semantics.sampledX = metadata.sampledX;
  result.semantics.sampledY = metadata.sampledY;
  result.semantics.sampledWidth = metadata.sampledWidth;
  result.semantics.sampledHeight = metadata.sampledHeight;
  result.semantics.coverage =
      [metadata.coverage isEqualToString:@"full"]
          ? ChromaspaceSourceExchange::SourceCoverage::FullSource
          : ChromaspaceSourceExchange::SourceCoverage::PartialSource;
  result.semantics.authoritative = metadata.authoritative;
  result.semantics.identityStripPresent = metadata.identityStripPresent;
  result.semantics.identityCube = metadata.identityCube;
  result.semantics.identityRamp = metadata.identityRamp;
  result.semantics.identityResolution = metadata.identityResolution;
  result.semantics.identityBandHeight = metadata.identityBandHeight;
  result.semantics.identityCubeY1 = metadata.identityCubeY1;
  result.semantics.identityCubeY2 = metadata.identityCubeY2;
  result.semantics.identityRampY1 = metadata.identityRampY1;
  result.semantics.identityRampY2 = metadata.identityRampY2;
  result.semantics.colorPrimaries =
      metadata.colorPrimaries.UTF8String ?: "";
  result.semantics.transferFunction =
      metadata.transferFunction.UTF8String ?: "";
  return result;
}

using ExactKey = std::pair<uint64_t, uint32_t>;

ExactKey exactKey(
    const ChromaspaceSourceExchange::PublicationKey& key) {
  return {key.sequence, key.slotIndex};
}

struct ImportedRecord {
  ChromaspaceSourceExchange::PublicationKey key;
  ChromaspaceMetal::ImportedSourceTexture source;
};

}  // namespace

@class ChromaspaceSourceViewerClientImpl;

namespace {

struct RetirementContext {
  __strong ChromaspaceSourceViewerClientImpl* client = nil;
  ChromaspaceSourceExchange::PublicationKey key;
};

void importedSourceRetired(void* context);

}  // namespace

@interface ChromaspaceSourceViewerClientImpl : NSObject {
 @private
  dispatch_queue_t _queue;
  dispatch_source_t _acquireTimer;
  NSXPCConnection* _connection;
  std::atomic<bool> _stopping;
  std::atomic<bool> _started;
  std::mutex _stateMutex;
  ChromaspaceSourceExchange::ViewerClientState _state;
  ChromaspaceSourceExchange::SourceViewerClientSnapshot _snapshot;
  ChromaspaceSourceExchange::ViewerSessionSnapshot _session;
  NSString* _senderId;
  NSData* _capabilityData;
  std::map<ExactKey, ImportedRecord> _imported;
  BOOL _acquireInFlight;
  uint64_t _registrationAttemptEpoch;
  uint64_t _routeAttemptEpoch;
  uint64_t _acquireAttemptEpoch;
  BOOL _acknowledgementInFlight;
  uint64_t _acknowledgementAttemptEpoch;
  uint32_t _acknowledgementTransportRetryCount;
  ChromaspaceSourceExchangeAcknowledgement* _acknowledgementPacket;
}

- (instancetype)initWithConfiguration:
    (const ChromaspaceSourceExchange::SourceViewerClientConfiguration&)
        configuration;
- (void)start;
- (BOOL)clearActiveSourceOnQueue;
- (BOOL)clearActiveSourceSynchronously;
- (void)shutdownSynchronously;
- (ChromaspaceSourceExchange::SourceViewerClientSnapshot)snapshot;
- (void)fail:(const std::string&)diagnostic;
- (void)pollLatest;
- (void)consumeAcquiredPacket:
            (ChromaspaceSourceExchangePacket*)packet
                       status:(ChromaspaceSourceExchangeStatus)status;
- (void)pumpAcknowledgement;
- (void)applyAcknowledgementTransition:
    (const ChromaspaceSourceExchange::ViewerClientTransition&)
        transition;
- (void)gpuDrainCompleted:
    (const ChromaspaceSourceExchange::PublicationKey&)key;
- (void)enqueueGpuDrainCompleted:
    (const ChromaspaceSourceExchange::PublicationKey&)key;

@end

@implementation ChromaspaceSourceViewerClientImpl

- (instancetype)initWithConfiguration:
    (const ChromaspaceSourceExchange::SourceViewerClientConfiguration&)
        configuration {
  self = [super init];
  if (self) {
    _queue = dispatch_queue_create(
        "com.chromaspace.SourceViewerClient", DISPATCH_QUEUE_SERIAL);
    dispatch_queue_set_specific(
        _queue,
        kViewerQueueSpecific,
        const_cast<void*>(kViewerQueueSpecific),
        nullptr);
    _stopping.store(false, std::memory_order_relaxed);
    _started.store(false, std::memory_order_relaxed);
    _senderId =
        [NSString stringWithUTF8String:configuration.senderId.c_str()];

    _session.senderId = configuration.senderId;
    _session.deviceRegistryId = configuration.deviceRegistryId;
    _session.pixelFormatMask = configuration.pixelFormatMask;
    _session.maximumWidth = configuration.maximumWidth;
    _session.maximumHeight = configuration.maximumHeight;
    _session.maximumSurfaceBytes =
        configuration.maximumSurfaceBytes;
    _session.maximumRetainedBytes =
        configuration.maximumRetainedBytes;
    _session.maximumSlots =
        ChromaspaceSourceExchangeMaximumSlots;
    _session.supportsSharedEvents = true;
    if (SecRandomCopyBytes(
            kSecRandomDefault,
            _session.capability.size(),
            _session.capability.data()) != errSecSuccess) {
      _session.capability.fill(0);
    }
    if (SecRandomCopyBytes(
            kSecRandomDefault,
            sizeof(_session.viewerGeneration),
            reinterpret_cast<uint8_t*>(
                &_session.viewerGeneration)) != errSecSuccess) {
      _session.viewerGeneration = 0;
    }
    if (_session.viewerGeneration == 0) {
      _session.viewerGeneration = 1;
    }
    _capabilityData = capabilityData(_session.capability);
    _snapshot.viewerGeneration = _session.viewerGeneration;
  }
  return self;
}

- (void)setHealth:
            (ChromaspaceSourceExchange::SourceViewerClientHealth)health
       diagnostic:(const std::string&)diagnostic {
  std::lock_guard<std::mutex> lock(_stateMutex);
  _snapshot.health = health;
  _snapshot.diagnostic = diagnostic;
  _snapshot.lastObservedSequence =
      _state.lastObservedSequence();
  _snapshot.liveKeyCount = _state.liveKeyCount();
}

- (void)releaseAllImportedSources {
  for (const auto& entry : _imported) {
    ChromaspaceMetal::releaseImportedSourceTexture(
        entry.second.source.sourceId);
  }
  _imported.clear();
}

- (void)fail:(const std::string&)diagnostic {
  if (_stopping.exchange(true, std::memory_order_acq_rel)) return;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    (void)_state.invalidateSession();
    _snapshot.health =
        ChromaspaceSourceExchange::SourceViewerClientHealth::Failed;
    _snapshot.diagnostic = diagnostic;
    _snapshot.lastObservedSequence = 0;
    _snapshot.liveKeyCount = 0;
    _snapshot.hasActiveSource = false;
    _snapshot.activeSource =
        ChromaspaceMetal::ImportedSourceTexture{};
  }
  if (_acquireTimer != nil) {
    dispatch_source_cancel(_acquireTimer);
    _acquireTimer = nil;
  }
  _acquireInFlight = NO;
  _acknowledgementInFlight = NO;
  _acknowledgementPacket = nil;
  [self releaseAllImportedSources];
  [_connection invalidate];
  _connection = nil;
}

- (void)startAcquireTimer {
  if (_acquireTimer != nil) return;
  _acquireTimer =
      dispatch_source_create(
          DISPATCH_SOURCE_TYPE_TIMER, 0, 0, _queue);
  dispatch_source_set_timer(
      _acquireTimer,
      dispatch_time(DISPATCH_TIME_NOW, 0),
      kAcquirePollNanoseconds,
      2 * NSEC_PER_MSEC);
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  dispatch_source_set_event_handler(_acquireTimer, ^{
    [strongSelf pollLatest];
  });
  dispatch_resume(_acquireTimer);
}

- (void)pollLatest {
  if (_stopping.load(std::memory_order_acquire) ||
      _connection == nil || _acquireInFlight) {
    return;
  }
  uint64_t afterSequence = 0;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_snapshot.health !=
            ChromaspaceSourceExchange::SourceViewerClientHealth::Ready ||
        !_state.canAcquire()) {
      return;
    }
    afterSequence = _state.lastObservedSequence();
  }
  _acquireInFlight = YES;
  const uint64_t attemptEpoch = ++_acquireAttemptEpoch;
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  id<ChromaspaceSourceExchangeBrokerProtocol> broker =
      [_connection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_acquireAttemptEpoch !=
                  attemptEpoch) {
                return;
              }
              strongSelf->_acquireInFlight = NO;
              [strongSelf fail:"viewer-acquire-transport-failed"];
            });
          }];
  [broker
      acquireLatestForSession:_capabilityData
                     senderId:_senderId
               afterSequence:afterSequence
                    withReply:
                        ^(ChromaspaceSourceExchangePacket* packet,
                          ChromaspaceSourceExchangeStatus status,
                          NSError* error) {
                          (void)error;
                          dispatch_async(strongSelf->_queue, ^{
                            if (strongSelf->_acquireAttemptEpoch !=
                                attemptEpoch) {
                              return;
                            }
                            [strongSelf
                                consumeAcquiredPacket:packet
                                               status:status];
                          });
                        }];
  dispatch_after(
      dispatch_time(
          DISPATCH_TIME_NOW,
          kAcquireReplyDeadlineNanoseconds),
      _queue, ^{
        if (strongSelf->_stopping.load(
                std::memory_order_acquire) ||
            !strongSelf->_acquireInFlight ||
            strongSelf->_acquireAttemptEpoch != attemptEpoch) {
          return;
        }
        strongSelf->_acquireInFlight = NO;
        [strongSelf fail:"viewer-acquire-reply-timeout"];
      });
}

- (void)consumeAcquiredPacket:
            (ChromaspaceSourceExchangePacket*)packet
                       status:(ChromaspaceSourceExchangeStatus)status {
  if (_stopping.load(std::memory_order_acquire)) return;
  _acquireInFlight = NO;
  if (status ==
      ChromaspaceSourceExchangeStatusNoNewPublication) {
    return;
  }
  NSError* validationError = nil;
  if (status != ChromaspaceSourceExchangeStatusAccepted ||
      packet == nil || ![packet validate:&validationError]) {
    [self fail:"viewer-acquired-packet-invalid"];
    return;
  }
  const ChromaspaceSourceExchange::Publication publication =
      publicationFromPacket(packet);
  ChromaspaceSourceExchange::ViewerClientTransition generationChange{};
  bool generationChangeAttempted = false;
  bool generationReplaced = false;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    if (_state.senderGeneration() != 0 &&
        _state.senderGeneration() !=
            publication.key.senderGeneration) {
      generationChangeAttempted = true;
      generationChange = _state.replaceSenderGeneration(
          publication.key.senderGeneration);
      generationReplaced = generationChange.accepted();
      if (generationReplaced) {
        _snapshot.hasActiveSource = false;
        _snapshot.activeSource =
            ChromaspaceMetal::ImportedSourceTexture{};
        _snapshot.lastObservedSequence = 0;
        _snapshot.liveKeyCount = 0;
      }
    }
  }
  if (!generationChange.abandoned.empty()) {
    for (const auto& key : generationChange.abandoned) {
      auto it = _imported.find(exactKey(key));
      if (it != _imported.end() && it->second.key == key) {
        ChromaspaceMetal::releaseImportedSourceTexture(
            it->second.source.sourceId);
        _imported.erase(it);
      }
    }
    ++_acknowledgementAttemptEpoch;
    _acknowledgementInFlight = NO;
    _acknowledgementPacket = nil;
    _acknowledgementTransportRetryCount = 0;
  }
  if (generationChangeAttempted && !generationReplaced) {
    [self fail:"viewer-sender-generation-transition-failed"];
    return;
  }
  ChromaspaceSourceExchange::ResultCode admission;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    admission = _state.beginImport(publication);
  }
  if (admission !=
      ChromaspaceSourceExchange::ResultCode::Accepted) {
    [self fail:"viewer-publication-admission-failed"];
    return;
  }

  ChromaspaceMetal::SharedSourceImportRequest request{};
  request.sharedTextureHandle =
      (__bridge void*)packet.textureHandle;
  request.sharedEventHandle =
      (__bridge void*)packet.eventHandle;
  request.senderId = publication.key.senderId;
  request.deviceRegistryId = publication.deviceRegistryId;
  request.senderGeneration = publication.key.senderGeneration;
  request.sequence = publication.key.sequence;
  request.slotIndex = publication.key.slotIndex;
  request.slotGeneration = publication.key.slotGeneration;
  request.readyValue = publication.readyValue;
  request.contentHash = publication.contentHash;
  request.width = static_cast<int>(publication.width);
  request.height = static_cast<int>(publication.height);
  request.pixelFormat = static_cast<int>(publication.pixelFormat);
  request.bytesPerRow =
      static_cast<size_t>(publication.bytesPerRow);
  request.byteSize = static_cast<size_t>(publication.byteSize);
  request.semantics = publication.semantics;
  ChromaspaceMetal::ImportedSourceTexture source{};
  std::string importError;
  const bool imported = ChromaspaceMetal::importSharedSourceTexture(
      request, &source, &importError);
  if (imported) {
    _imported[exactKey(publication.key)] =
        ImportedRecord{publication.key, source};
  }

  ChromaspaceSourceExchange::ResultCode completed;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    completed =
        _state.importCompleted(publication.key, imported);
    _snapshot.lastObservedSequence =
        _state.lastObservedSequence();
    _snapshot.liveKeyCount = _state.liveKeyCount();
    if (!imported) {
      _snapshot.diagnostic =
          importError.empty()
              ? "viewer-source-import-failed"
              : importError;
    }
  }
  if (completed !=
      ChromaspaceSourceExchange::ResultCode::Accepted) {
    if (imported) {
      ChromaspaceMetal::releaseImportedSourceTexture(
          source.sourceId);
      _imported.erase(exactKey(publication.key));
    }
    [self fail:"viewer-import-completion-state-failed"];
    return;
  }
  [self pumpAcknowledgement];
}

- (ChromaspaceSourceExchangeAcknowledgement*)
    packetForAcknowledgement:
        (const ChromaspaceSourceExchange::ViewerAcknowledgement&)
            acknowledgement {
  const ChromaspaceSourceExchangeStatus status =
      acknowledgement.state ==
              ChromaspaceSourceExchange::
                  ViewerAcknowledgementState::Acquired
          ? ChromaspaceSourceExchangeStatusAcquired
          : ChromaspaceSourceExchangeStatusRetired;
  NSString* sender =
      [NSString
          stringWithUTF8String:
              acknowledgement.key.senderId.c_str()];
  ChromaspaceSourceExchangeAcknowledgement* packet =
      [[ChromaspaceSourceExchangeAcknowledgement alloc]
          initWithSessionCapability:_capabilityData
                            senderId:sender
                    senderGeneration:
                        acknowledgement.key.senderGeneration
                            sequence:acknowledgement.key.sequence
                           slotIndex:acknowledgement.key.slotIndex
                      slotGeneration:
                          acknowledgement.key.slotGeneration
                              status:status];
  NSError* error = nil;
  return [packet validate:&error] ? packet : nil;
}

- (void)pumpAcknowledgement {
  if (_stopping.load(std::memory_order_acquire) ||
      _connection == nil || _acknowledgementInFlight) {
    return;
  }
  std::optional<ChromaspaceSourceExchange::ViewerAcknowledgement>
      acknowledgement;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    acknowledgement = _state.beginNextAcknowledgement();
  }
  if (!acknowledgement.has_value()) return;
  if (_acknowledgementPacket == nil) {
    _acknowledgementPacket =
        [self packetForAcknowledgement:*acknowledgement];
  }
  if (_acknowledgementPacket == nil) {
    [self fail:"viewer-acknowledgement-packet-invalid"];
    return;
  }

  _acknowledgementInFlight = YES;
  const uint64_t attemptEpoch =
      ++_acknowledgementAttemptEpoch;
  const ChromaspaceSourceExchange::ViewerAcknowledgement exact =
      *acknowledgement;
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  id<ChromaspaceSourceExchangeBrokerProtocol> broker =
      [_connection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_stopping.load(
                      std::memory_order_acquire) ||
                  !strongSelf->_acknowledgementInFlight ||
                  strongSelf->_acknowledgementAttemptEpoch !=
                      attemptEpoch) {
                return;
              }
              bool restored = false;
              {
                std::lock_guard<std::mutex> lock(
                    strongSelf->_stateMutex);
                restored =
                    strongSelf->_state
                        .acknowledgementTransportFailed(exact) ==
                    ChromaspaceSourceExchange::ResultCode::Accepted;
              }
              strongSelf->_acknowledgementInFlight = NO;
              if (!restored ||
                  ++strongSelf->
                          _acknowledgementTransportRetryCount >
                      kMaximumAcknowledgementTransportRetries) {
                [strongSelf
                    fail:"viewer-acknowledgement-transport-failed"];
                return;
              }
              dispatch_after(
                  dispatch_time(
                      DISPATCH_TIME_NOW,
                      kAcknowledgementRetryNanoseconds),
                  strongSelf->_queue, ^{
                    [strongSelf pumpAcknowledgement];
                  });
            });
          }];
  [broker
      acknowledge:_acknowledgementPacket
         withReply:^(ChromaspaceSourceExchangeStatus status,
                     NSError* error) {
           (void)error;
           dispatch_async(strongSelf->_queue, ^{
             if (strongSelf->_stopping.load(
                     std::memory_order_acquire) ||
                 !strongSelf->_acknowledgementInFlight ||
                 strongSelf->_acknowledgementAttemptEpoch !=
                     attemptEpoch) {
               return;
             }
             if (status !=
                 ChromaspaceSourceExchangeStatusAccepted) {
               [strongSelf
                   fail:"viewer-acknowledgement-rejected"];
               return;
             }
             ChromaspaceSourceExchange::ViewerClientTransition
                 transition{};
             {
               std::lock_guard<std::mutex> lock(
                   strongSelf->_stateMutex);
               transition =
                   strongSelf->_state.acknowledgementAccepted(
                       exact);
               strongSelf->_snapshot.liveKeyCount =
                   strongSelf->_state.liveKeyCount();
             }
             if (!transition.accepted()) {
               [strongSelf
                   fail:"viewer-acknowledgement-state-failed"];
               return;
             }
             strongSelf->_acknowledgementInFlight = NO;
             strongSelf->_acknowledgementPacket = nil;
             strongSelf->
                 _acknowledgementTransportRetryCount = 0;
             [strongSelf applyAcknowledgementTransition:transition];
             [strongSelf pumpAcknowledgement];
           });
         }];
  dispatch_after(
      dispatch_time(
          DISPATCH_TIME_NOW,
          kControlReplyDeadlineNanoseconds),
      _queue, ^{
        if (strongSelf->_stopping.load(
                std::memory_order_acquire) ||
            !strongSelf->_acknowledgementInFlight ||
            strongSelf->_acknowledgementAttemptEpoch !=
                attemptEpoch) {
          return;
        }
        bool restored = false;
        {
          std::lock_guard<std::mutex> lock(
              strongSelf->_stateMutex);
          restored =
              strongSelf->_state
                  .acknowledgementTransportFailed(exact) ==
              ChromaspaceSourceExchange::ResultCode::Accepted;
        }
        strongSelf->_acknowledgementInFlight = NO;
        if (!restored ||
            ++strongSelf->_acknowledgementTransportRetryCount >
                kMaximumAcknowledgementTransportRetries) {
          [strongSelf
              fail:"viewer-acknowledgement-reply-timeout"];
          return;
        }
        dispatch_after(
            dispatch_time(
                DISPATCH_TIME_NOW,
                kAcknowledgementRetryNanoseconds),
            strongSelf->_queue, ^{
              [strongSelf pumpAcknowledgement];
            });
      });
}

- (void)applyAcknowledgementTransition:
    (const ChromaspaceSourceExchange::ViewerClientTransition&)
        transition {
  if (transition.activated.has_value()) {
    auto it = _imported.find(exactKey(*transition.activated));
    if (it == _imported.end() ||
        !(it->second.key == *transition.activated)) {
      [self fail:"viewer-activated-source-missing"];
      return;
    }
    std::lock_guard<std::mutex> lock(_stateMutex);
    _snapshot.hasActiveSource = true;
    _snapshot.activeSource = it->second.source;
    _snapshot.diagnostic = "ready-resident-source";
  }

  if (transition.needsGpuDrain.has_value()) {
    auto it = _imported.find(
        exactKey(*transition.needsGpuDrain));
    if (it == _imported.end() ||
        !(it->second.key == *transition.needsGpuDrain)) {
      [self fail:"viewer-draining-source-missing"];
      return;
    }
    auto* context = new (std::nothrow) RetirementContext;
    if (context == nullptr) {
      [self fail:"viewer-retirement-context-allocation-failed"];
      return;
    }
    context->client = self;
    context->key = *transition.needsGpuDrain;
    std::string error;
    if (!ChromaspaceMetal::retireImportedSourceTexture(
            it->second.source.sourceId,
            importedSourceRetired,
            context,
            &error)) {
      delete context;
      [self fail:error.empty()
                     ? "viewer-imported-source-retirement-failed"
                     : error];
      return;
    }
  }

  if (transition.locallyReleasable.has_value()) {
    _imported.erase(exactKey(*transition.locallyReleasable));
  }
}

- (void)gpuDrainCompleted:
    (const ChromaspaceSourceExchange::PublicationKey&)key {
  if (_stopping.load(std::memory_order_acquire)) return;
  ChromaspaceSourceExchange::ResultCode result;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    result = _state.gpuDrainCompleted(key);
    _snapshot.liveKeyCount = _state.liveKeyCount();
  }
  if (result != ChromaspaceSourceExchange::ResultCode::Accepted) {
    [self fail:"viewer-gpu-drain-state-failed"];
    return;
  }
  [self pumpAcknowledgement];
}

- (void)enqueueGpuDrainCompleted:
    (const ChromaspaceSourceExchange::PublicationKey&)key {
  const ChromaspaceSourceExchange::PublicationKey exact = key;
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  dispatch_async(_queue, ^{
    [strongSelf gpuDrainCompleted:exact];
  });
}

- (void)bindRoute {
  [self
      setHealth:
          ChromaspaceSourceExchange::SourceViewerClientHealth::
              BindingRoute
     diagnostic:"binding-viewer-route"];
  ChromaspaceSourceExchangeViewerRoute* route =
      [[ChromaspaceSourceExchangeViewerRoute alloc]
          initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                  protocolMinor:ChromaspaceSourceExchangeProtocolMinor
              sessionCapability:_capabilityData
               viewerGeneration:_session.viewerGeneration
                  routeRevision:1
                       senderId:_senderId];
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  const uint64_t attemptEpoch = ++_routeAttemptEpoch;
  id<ChromaspaceSourceExchangeBrokerProtocol> broker =
      [_connection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_routeAttemptEpoch !=
                  attemptEpoch) {
                return;
              }
              [strongSelf fail:"viewer-route-transport-failed"];
            });
          }];
  [broker
      bindViewerRoute:route
            withReply:^(ChromaspaceSourceExchangeStatus status,
                        NSError* error) {
              (void)error;
              dispatch_async(strongSelf->_queue, ^{
                if (strongSelf->_stopping.load(
                        std::memory_order_acquire) ||
                    strongSelf->_routeAttemptEpoch !=
                        attemptEpoch) {
                  return;
                }
                ++strongSelf->_routeAttemptEpoch;
                if (status !=
                    ChromaspaceSourceExchangeStatusAccepted) {
                  [strongSelf fail:"viewer-route-rejected"];
                  return;
                }
                ChromaspaceSourceExchange::ViewerClientTransition
                    installed{};
                {
                  std::lock_guard<std::mutex> lock(
                      strongSelf->_stateMutex);
                  installed =
                      strongSelf->_state.installSession(
                          strongSelf->_session);
                }
                if (!installed.accepted()) {
                  [strongSelf
                      fail:"viewer-session-state-invalid"];
                  return;
                }
                [strongSelf
                    setHealth:
                        ChromaspaceSourceExchange::
                            SourceViewerClientHealth::Ready
                   diagnostic:"ready-awaiting-resident-source"];
                [strongSelf startAcquireTimer];
              });
            }];
  dispatch_after(
      dispatch_time(
          DISPATCH_TIME_NOW,
          kControlReplyDeadlineNanoseconds),
      _queue, ^{
        if (strongSelf->_stopping.load(
                std::memory_order_acquire) ||
            strongSelf->_routeAttemptEpoch != attemptEpoch) {
          return;
        }
        [strongSelf fail:"viewer-route-reply-timeout"];
      });
}

- (void)registerViewer {
  ChromaspaceSourceExchangeViewerRegistration* registration =
      [[ChromaspaceSourceExchangeViewerRegistration alloc]
          initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                  protocolMinor:ChromaspaceSourceExchangeProtocolMinor
              sessionCapability:_capabilityData
               viewerGeneration:_session.viewerGeneration
               deviceRegistryId:_session.deviceRegistryId
                pixelFormatMask:_session.pixelFormatMask
                   maximumWidth:_session.maximumWidth
                  maximumHeight:_session.maximumHeight
            maximumSurfaceBytes:_session.maximumSurfaceBytes
           maximumRetainedBytes:_session.maximumRetainedBytes
                   maximumSlots:_session.maximumSlots
           supportsSharedEvents:YES];
  NSError* validationError = nil;
  if (![registration validate:&validationError]) {
    [self fail:"viewer-registration-invalid"];
    return;
  }
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  const uint64_t attemptEpoch = ++_registrationAttemptEpoch;
  id<ChromaspaceSourceExchangeBrokerProtocol> broker =
      [_connection
          remoteObjectProxyWithErrorHandler:^(NSError* error) {
            (void)error;
            dispatch_async(strongSelf->_queue, ^{
              if (strongSelf->_registrationAttemptEpoch !=
                  attemptEpoch) {
                return;
              }
              [strongSelf
                  fail:"viewer-registration-transport-failed"];
            });
          }];
  [broker
      registerViewer:registration
           withReply:^(ChromaspaceSourceExchangeStatus status,
                       NSError* error) {
             (void)error;
             dispatch_async(strongSelf->_queue, ^{
               if (strongSelf->_stopping.load(
                       std::memory_order_acquire) ||
                   strongSelf->_registrationAttemptEpoch !=
                       attemptEpoch) {
                 return;
               }
               ++strongSelf->_registrationAttemptEpoch;
               if (status !=
                   ChromaspaceSourceExchangeStatusAccepted) {
                 [strongSelf fail:"viewer-registration-rejected"];
                 return;
               }
               [strongSelf bindRoute];
             });
           }];
  dispatch_after(
      dispatch_time(
          DISPATCH_TIME_NOW,
          kControlReplyDeadlineNanoseconds),
      _queue, ^{
        if (strongSelf->_stopping.load(
                std::memory_order_acquire) ||
            strongSelf->_registrationAttemptEpoch !=
                attemptEpoch) {
          return;
        }
        [strongSelf fail:"viewer-registration-reply-timeout"];
      });
}

- (void)startOnQueue {
  if (_stopping.load(std::memory_order_acquire) ||
      _senderId.length == 0 ||
      _session.deviceRegistryId == 0 ||
      _session.pixelFormatMask == 0 ||
      _session.maximumWidth == 0 ||
      _session.maximumHeight == 0 ||
      _session.maximumSurfaceBytes == 0 ||
      _session.maximumRetainedBytes == 0 ||
      !_session.supportsSharedEvents) {
    [self fail:"invalid-viewer-client-configuration"];
    return;
  }
  _connection =
      [[NSXPCConnection alloc]
          initWithMachServiceName:
              ChromaspaceSourceExchangeMachServiceName
                      options:0];
  _connection.remoteObjectInterface = viewerBrokerInterface();
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  _connection.invalidationHandler = ^{
    dispatch_async(strongSelf->_queue, ^{
      if (!strongSelf->_stopping.load(
              std::memory_order_acquire)) {
        [strongSelf fail:"viewer-broker-invalidated"];
      }
    });
  };
  _connection.interruptionHandler = _connection.invalidationHandler;
  [_connection resume];
  [self registerViewer];
}

- (void)start {
  bool expected = false;
  if (!_started.compare_exchange_strong(
          expected, true, std::memory_order_acq_rel)) {
    return;
  }
  [self
      setHealth:
          ChromaspaceSourceExchange::SourceViewerClientHealth::
              Registering
     diagnostic:"registering-viewer"];
  ChromaspaceSourceViewerClientImpl* strongSelf = self;
  dispatch_async(_queue, ^{
    [strongSelf startOnQueue];
  });
}

- (BOOL)clearActiveSourceOnQueue {
  if (_stopping.load(std::memory_order_acquire)) return NO;
  ChromaspaceSourceExchange::ViewerClientTransition transition{};
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    transition = _state.clearActiveSource();
    if (transition.accepted()) {
      _snapshot.hasActiveSource = false;
      _snapshot.activeSource =
          ChromaspaceMetal::ImportedSourceTexture{};
      _snapshot.liveKeyCount = _state.liveKeyCount();
      _snapshot.diagnostic = "resident-source-cleared";
    }
  }
  if (!transition.accepted()) return NO;
  [self applyAcknowledgementTransition:transition];
  [self pumpAcknowledgement];
  return !_stopping.load(std::memory_order_acquire);
}

- (BOOL)clearActiveSourceSynchronously {
  if (dispatch_get_specific(kViewerQueueSpecific) != nullptr) {
    return [self clearActiveSourceOnQueue];
  }
  __block BOOL cleared = NO;
  dispatch_sync(_queue, ^{
    cleared = [self clearActiveSourceOnQueue];
  });
  return cleared;
}

- (void)shutdownOnQueue {
  _stopping.store(true, std::memory_order_release);
  if (_acquireTimer != nil) {
    dispatch_source_cancel(_acquireTimer);
    _acquireTimer = nil;
  }
  if (_connection != nil && _capabilityData != nil) {
    id<ChromaspaceSourceExchangeBrokerProtocol> broker =
        [_connection remoteObjectProxy];
    [broker
        disconnectSession:_capabilityData
                withReply:^(ChromaspaceSourceExchangeStatus status) {
                  (void)status;
                }];
  }
  [_connection invalidate];
  _connection = nil;
  {
    std::lock_guard<std::mutex> lock(_stateMutex);
    (void)_state.invalidateSession();
    _snapshot.health =
        ChromaspaceSourceExchange::SourceViewerClientHealth::Stopped;
    _snapshot.diagnostic = "stopped";
    _snapshot.lastObservedSequence = 0;
    _snapshot.liveKeyCount = 0;
    _snapshot.hasActiveSource = false;
    _snapshot.activeSource =
        ChromaspaceMetal::ImportedSourceTexture{};
  }
  [self releaseAllImportedSources];
  _acknowledgementPacket = nil;
}

- (void)shutdownSynchronously {
  _stopping.store(true, std::memory_order_release);
  if (dispatch_get_specific(kViewerQueueSpecific) != nullptr) {
    [self shutdownOnQueue];
    return;
  }
  dispatch_sync(_queue, ^{
    [self shutdownOnQueue];
  });
}

- (ChromaspaceSourceExchange::SourceViewerClientSnapshot)snapshot {
  std::lock_guard<std::mutex> lock(_stateMutex);
  return _snapshot;
}

@end

namespace {

void importedSourceRetired(void* context) {
  std::unique_ptr<RetirementContext> retirement(
      static_cast<RetirementContext*>(context));
  if (!retirement || retirement->client == nil) return;
  [retirement->client
      enqueueGpuDrainCompleted:retirement->key];
}

}  // namespace

namespace ChromaspaceSourceExchange {

struct SourceViewerClient {
  __strong ChromaspaceSourceViewerClientImpl* implementation = nil;
};

SourceViewerClient* createSourceViewerClient(
    const SourceViewerClientConfiguration& configuration) {
  auto* client = new SourceViewerClient;
  client->implementation =
      [[ChromaspaceSourceViewerClientImpl alloc]
          initWithConfiguration:configuration];
  if (client->implementation == nil) {
    delete client;
    return nullptr;
  }
  return client;
}

void startSourceViewerClient(SourceViewerClient* client) {
  if (client == nullptr || client->implementation == nil) return;
  [client->implementation start];
}

bool clearSourceViewerClient(SourceViewerClient* client) {
  if (client == nullptr || client->implementation == nil) return false;
  return [client->implementation clearActiveSourceSynchronously];
}

void destroySourceViewerClient(SourceViewerClient* client) {
  if (client == nullptr) return;
  [client->implementation shutdownSynchronously];
  client->implementation = nil;
  delete client;
}

SourceViewerClientSnapshot sourceViewerClientSnapshot(
    const SourceViewerClient* client) {
  if (client == nullptr || client->implementation == nil) return {};
  return [client->implementation snapshot];
}

}  // namespace ChromaspaceSourceExchange
