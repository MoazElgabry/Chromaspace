#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ChromaspaceSourceExchangeState.h"
#import "ChromaspaceSourceExchangeV2.h"

#include <cstring>
#include <string>
#include <unistd.h>

using ChromaspaceSourceExchange::AcknowledgementState;
using ChromaspaceSourceExchange::BrokerState;
using ChromaspaceSourceExchange::Capability;
using ChromaspaceSourceExchange::ProducerRegistration;
using ChromaspaceSourceExchange::ProducerReleaseBatch;
using ChromaspaceSourceExchange::ProducerReleaseEvent;
using ChromaspaceSourceExchange::Publication;
using ChromaspaceSourceExchange::PublicationKey;
using ChromaspaceSourceExchange::ResultCode;
using ChromaspaceSourceExchange::TransitionResult;
using ChromaspaceSourceExchange::ViewerRegistration;

namespace {

constexpr size_t kMaximumLivePeers = 64;
constexpr int64_t kRegistrationDeadlineNanoseconds =
    10 * NSEC_PER_SEC;

enum class PeerRole {
  Unregistered,
  ProducerRelay,
  Viewer,
  Producer,
};

bool capabilityFromData(NSData* data, Capability* out) {
  if (out) out->fill(0);
  if (!out || data == nil ||
      data.length != ChromaspaceSourceExchangeCapabilityBytes) {
    return false;
  }
  std::memcpy(out->data(), data.bytes, out->size());
  return true;
}

NSString* capabilityKey(const Capability& capability) {
  static const char digits[] = "0123456789abcdef";
  char text[ChromaspaceSourceExchangeCapabilityBytes * 2 + 1] = {};
  for (size_t i = 0; i < capability.size(); ++i) {
    text[i * 2] = digits[(capability[i] >> 4) & 0xf];
    text[i * 2 + 1] = digits[capability[i] & 0xf];
  }
  return [NSString stringWithUTF8String:text];
}

NSString* packetKey(const Capability& capability,
                    const PublicationKey& key) {
  return [NSString
      stringWithFormat:@"%@|%@|%llu|%llu|%u|%llu",
                       capabilityKey(capability),
                       [NSString stringWithUTF8String:key.senderId.c_str()],
                       (unsigned long long)key.senderGeneration,
                       (unsigned long long)key.sequence,
                       key.slotIndex,
                       (unsigned long long)key.slotGeneration];
}

NSString* producerPeerKey(const Capability& capability,
                          NSString* senderId) {
  return [NSString stringWithFormat:@"%@|%@",
                                    capabilityKey(capability),
                                    senderId];
}

NSError* brokerError(ResultCode code, NSString* detail) {
  ChromaspaceSourceExchangeErrorCode errorCode =
      ChromaspaceSourceExchangeErrorInvalidIdentity;
  switch (code) {
    case ResultCode::ProtocolMismatch:
      errorCode = ChromaspaceSourceExchangeErrorInvalidProtocol;
      break;
    case ResultCode::CapabilityMismatch:
      errorCode = ChromaspaceSourceExchangeErrorInvalidCapability;
      break;
    case ResultCode::DeviceMismatch:
      errorCode = ChromaspaceSourceExchangeErrorDeviceMismatch;
      break;
    case ResultCode::UnsupportedFormat:
    case ResultCode::InvalidPublication:
      errorCode = ChromaspaceSourceExchangeErrorInvalidSurface;
      break;
    case ResultCode::ResourceLimit:
    case ResultCode::SlotBusy:
      errorCode = ChromaspaceSourceExchangeErrorResourceLimit;
      break;
    case ResultCode::Stale:
    case ResultCode::InvalidTransition:
      errorCode = ChromaspaceSourceExchangeErrorStalePublication;
      break;
    case ResultCode::SessionMissing:
    case ResultCode::ProducerMissing:
      errorCode = ChromaspaceSourceExchangeErrorSessionMissing;
      break;
    default:
      break;
  }
  return [NSError
      errorWithDomain:ChromaspaceSourceExchangeErrorDomain
                 code:errorCode
             userInfo:@{
               NSLocalizedDescriptionKey :
                   detail ?: @"SourceExchangeV2 broker rejected the request."
             }];
}

ChromaspaceSourceExchangeStatus statusForResult(ResultCode code) {
  switch (code) {
    case ResultCode::Accepted:
      return ChromaspaceSourceExchangeStatusAccepted;
    case ResultCode::NoNewPublication:
      return ChromaspaceSourceExchangeStatusNoNewPublication;
    case ResultCode::Stale:
      return ChromaspaceSourceExchangeStatusStale;
    default:
      return ChromaspaceSourceExchangeStatusRejected;
  }
}

PublicationKey publicationKey(
    ChromaspaceSourceExchangeMetadata* metadata) {
  PublicationKey key{};
  key.senderId = metadata.senderId.UTF8String ?: "";
  key.senderGeneration = metadata.senderGeneration;
  key.sequence = metadata.sequence;
  key.slotIndex = metadata.slotIndex;
  key.slotGeneration = metadata.slotGeneration;
  return key;
}

ChromaspaceSourceExchange::SourceSemanticMetadata semanticsFromMetadata(
    ChromaspaceSourceExchangeMetadata* metadata) {
  ChromaspaceSourceExchange::SourceSemanticMetadata value{};
  value.sourceX = metadata.sourceX;
  value.sourceY = metadata.sourceY;
  value.sourceWidth = metadata.sourceWidth;
  value.sourceHeight = metadata.sourceHeight;
  value.sampledX = metadata.sampledX;
  value.sampledY = metadata.sampledY;
  value.sampledWidth = metadata.sampledWidth;
  value.sampledHeight = metadata.sampledHeight;
  value.coverage =
      [metadata.coverage isEqualToString:@"full"]
          ? ChromaspaceSourceExchange::SourceCoverage::FullSource
          : ChromaspaceSourceExchange::SourceCoverage::PartialSource;
  value.authoritative = metadata.authoritative;
  value.identityStripPresent = metadata.identityStripPresent;
  value.identityCube = metadata.identityCube;
  value.identityRamp = metadata.identityRamp;
  value.identityResolution = metadata.identityResolution;
  value.identityBandHeight = metadata.identityBandHeight;
  value.identityCubeY1 = metadata.identityCubeY1;
  value.identityCubeY2 = metadata.identityCubeY2;
  value.identityRampY1 = metadata.identityRampY1;
  value.identityRampY2 = metadata.identityRampY2;
  value.colorPrimaries = metadata.colorPrimaries.UTF8String ?: "";
  value.transferFunction = metadata.transferFunction.UTF8String ?: "";
  return value;
}

void releasePackets(NSMutableDictionary<NSString*,
                                          ChromaspaceSourceExchangePacket*>*
                        packets,
                    const Capability& capability,
                    const TransitionResult& result) {
  for (const PublicationKey& key : result.released) {
    [packets removeObjectForKey:packetKey(capability, key)];
  }
}

NSXPCInterface* brokerInterface() {
  NSXPCInterface* interface =
      [NSXPCInterface
          interfaceWithProtocol:
              @protocol(ChromaspaceSourceExchangeBrokerProtocol)];
  NSSet<Class>* viewerClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeViewerRegistration class],
                     [NSData class],
                     nil];
  NSSet<Class>* producerClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeProducerRegistration class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* routeClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeViewerRoute class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* bootstrapRegistrationClasses = [NSSet
      setWithObjects:
          [ChromaspaceSourceExchangeRelayBootstrapRegistration class],
          [NSXPCListenerEndpoint class],
          [NSData class],
          nil];
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
  NSSet<Class>* acknowledgementClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeAcknowledgement class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* releaseBatchClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeReleaseBatch class],
                     [ChromaspaceSourceExchangeReleaseEvent class],
                     [NSArray class],
                     [NSData class],
                     [NSString class],
                     nil];
  [interface setClasses:viewerClasses
            forSelector:@selector(registerViewer:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:producerClasses
            forSelector:@selector(registerProducer:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:bootstrapRegistrationClasses
            forSelector:@selector(registerProducerRelayBootstrap:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:routeClasses
            forSelector:@selector(bindViewerRoute:withReply:)
          argumentIndex:0
                ofReply:NO];
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
  [interface setClasses:packetClasses
            forSelector:@selector(acquireLatestForSession:senderId:afterSequence:withReply:)
          argumentIndex:0
                ofReply:YES];
  [interface setClasses:acknowledgementClasses
            forSelector:@selector(acknowledge:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:releaseBatchClasses
            forSelector:@selector(fetchProducerReleasesAfterOrdinal:maximumEvents:withReply:)
          argumentIndex:0
                ofReply:YES];
  return interface;
}

NSXPCInterface* hostBootstrapInterface() {
  NSXPCInterface* interface =
      [NSXPCInterface
          interfaceWithProtocol:
              @protocol(ChromaspaceSourceExchangeHostBootstrapProtocol)];
  NSSet<Class>* endpointClasses = [NSSet
      setWithObjects:[NSXPCListenerEndpoint class], nil];
  [interface setClasses:[NSSet setWithObjects:[NSData class], nil]
            forSelector:@selector(redeemProducerRelayWithToken:withReply:)
          argumentIndex:0
                ofReply:NO];
  [interface setClasses:endpointClasses
            forSelector:@selector(redeemProducerRelayWithToken:withReply:)
          argumentIndex:0
                ofReply:YES];
  return interface;
}

}  // namespace

@class ChromaspaceSourceExchangeBrokerCoordinator;
@class ChromaspaceSourceExchangePendingBootstrap;

@interface ChromaspaceSourceExchangeBrokerPeer
    : NSObject <ChromaspaceSourceExchangeBrokerProtocol>

@property(nonatomic, weak) NSXPCConnection* connection;
@property(nonatomic, strong)
    ChromaspaceSourceExchangeBrokerCoordinator* coordinator;
@property(nonatomic) PeerRole role;
@property(nonatomic) BOOL attached;
@property(nonatomic, copy) NSData* capability;
@property(nonatomic, copy) NSString* senderId;
@property(nonatomic) uint64_t viewerGeneration;
@property(nonatomic) uint64_t senderGeneration;
@property(nonatomic) uint64_t routeRevision;
@property(nonatomic) BOOL bootstrapRedeemed;
@property(nonatomic, copy) NSString* boundSenderId;
@property(nonatomic, strong)
    ChromaspaceSourceExchangeViewerRegistration* viewerRegistration;

@end

@interface ChromaspaceSourceExchangePendingBootstrap : NSObject

@property(nonatomic, copy) NSData* token;
@property(nonatomic, strong) NSXPCListenerEndpoint* endpoint;
@property(nonatomic, weak) ChromaspaceSourceExchangeBrokerPeer* owner;

@end

@implementation ChromaspaceSourceExchangePendingBootstrap
@end

@interface ChromaspaceSourceExchangeBrokerCoordinator : NSObject {
 @private
  dispatch_queue_t _queue;
  BrokerState _state;
  NSMutableDictionary<NSString*, ChromaspaceSourceExchangePacket*>* _packets;
  NSMutableDictionary<NSString*, ChromaspaceSourceExchangeBrokerPeer*>*
      _viewerPeers;
  NSMutableDictionary<NSString*, ChromaspaceSourceExchangeBrokerPeer*>*
      _producerPeers;
  NSMutableDictionary<NSString*, ChromaspaceSourceExchangeBrokerPeer*>*
      _viewerRoutes;
  NSMutableDictionary<NSData*, ChromaspaceSourceExchangePendingBootstrap*>*
      _pendingBootstraps;
  size_t _livePeerCount;
}

- (void)registerViewer:
            (ChromaspaceSourceExchangeViewerRegistration*)registration
                 peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                NSError* _Nullable))reply;
- (void)registerProducer:
            (ChromaspaceSourceExchangeProducerRegistration*)registration
                   peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                  reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                  NSError* _Nullable))reply;
- (void)registerProducerRelayBootstrap:
            (ChromaspaceSourceExchangeRelayBootstrapRegistration*)registration
                                      peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                     reply:
                                         (void (^)(ChromaspaceSourceExchangeStatus,
                                                   NSError* _Nullable))reply;
- (void)bindViewerRoute:(ChromaspaceSourceExchangeViewerRoute*)route
                   peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                  reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                  NSError* _Nullable))reply;
- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
               peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
              reply:
                  (void (^)(ChromaspaceSourceExchangeProducerLease* _Nullable,
                            ChromaspaceSourceExchangeStatus,
                            NSError* _Nullable))reply;
- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
                  peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                 reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                 NSError* _Nullable))reply;
- (void)acquireLatest:(NSData*)capability
             senderId:(NSString*)senderId
        afterSequence:(uint64_t)afterSequence
                 peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                reply:
                    (void (^)(ChromaspaceSourceExchangePacket* _Nullable,
                              ChromaspaceSourceExchangeStatus,
                              NSError* _Nullable))reply;
- (void)acknowledge:
            (ChromaspaceSourceExchangeAcknowledgement*)acknowledgement
               peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
              reply:(void (^)(ChromaspaceSourceExchangeStatus,
                              NSError* _Nullable))reply;
- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                     peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                    reply:
                                        (void (^)(ChromaspaceSourceExchangeReleaseBatch* _Nullable,
                                                  ChromaspaceSourceExchangeStatus,
                                                  NSError* _Nullable))reply;
- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                             peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                            reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                                            NSError* _Nullable))reply;
- (void)disconnectSession:(NSData*)capability
                      peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                     reply:(void (^)(ChromaspaceSourceExchangeStatus))reply;
- (void)peerInvalidated:(ChromaspaceSourceExchangeBrokerPeer*)peer;
- (void)redeemProducerRelayWithToken:(NSData*)token
                               reply:
                                   (void (^)(NSXPCListenerEndpoint* _Nullable,
                                             ChromaspaceSourceExchangeStatus,
                                             NSError* _Nullable))reply;
- (BOOL)tryAttachPeer:(ChromaspaceSourceExchangeBrokerPeer*)peer;
- (void)expirePeerIfUnregistered:
    (ChromaspaceSourceExchangeBrokerPeer*)peer;

@end

@implementation ChromaspaceSourceExchangeBrokerCoordinator

- (instancetype)init {
  self = [super init];
  if (self) {
    _queue = dispatch_queue_create(
        "com.chromaspace.SourceExchangeBroker.state",
        DISPATCH_QUEUE_SERIAL);
    _packets = [NSMutableDictionary dictionary];
    _viewerPeers = [NSMutableDictionary dictionary];
    _producerPeers = [NSMutableDictionary dictionary];
    _viewerRoutes = [NSMutableDictionary dictionary];
    _pendingBootstraps = [NSMutableDictionary dictionary];
  }
  return self;
}

- (void)removeViewerRouteForPeer:
    (ChromaspaceSourceExchangeBrokerPeer*)peer {
  if (peer.boundSenderId == nil) return;
  if (_viewerRoutes[peer.boundSenderId] == peer) {
    [_viewerRoutes removeObjectForKey:peer.boundSenderId];
  }
  peer.boundSenderId = nil;
}

- (void)disconnectProducerForCapability:(NSData*)capabilityData
                               senderId:(NSString*)senderId {
  Capability capability{};
  if (!capabilityFromData(capabilityData, &capability) ||
      senderId == nil) {
    return;
  }
  NSString* ownershipKey = producerPeerKey(capability, senderId);
  ChromaspaceSourceExchangeBrokerPeer* producerPeer =
      _producerPeers[ownershipKey];
  if (producerPeer == nil) return;
  TransitionResult result = _state.disconnectProducer(
      capability,
      senderId.UTF8String ?: "",
      producerPeer.senderGeneration);
  releasePackets(_packets, capability, result);
  [_producerPeers removeObjectForKey:ownershipKey];
  producerPeer.role = PeerRole::Unregistered;
  producerPeer.capability = nil;
  producerPeer.senderId = nil;
  producerPeer.senderGeneration = 0;
  [producerPeer.connection invalidate];
}

- (void)unbindProducerPeersForCapability:(NSData*)capability {
  if (capability == nil) return;
  NSMutableArray<NSString*>* keys = [NSMutableArray array];
  [_producerPeers
      enumerateKeysAndObjectsUsingBlock:
          ^(NSString* key,
            ChromaspaceSourceExchangeBrokerPeer* producerPeer,
            BOOL* stop) {
            (void)stop;
            if ([producerPeer.capability isEqualToData:capability]) {
              producerPeer.role = PeerRole::Unregistered;
              producerPeer.capability = nil;
              producerPeer.senderId = nil;
              producerPeer.senderGeneration = 0;
              [keys addObject:key];
              [producerPeer.connection invalidate];
            }
          }];
  [_producerPeers removeObjectsForKeys:keys];
}

- (void)registerViewer:
            (ChromaspaceSourceExchangeViewerRegistration*)registration
                 peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    Capability capability{};
    if ((peer.role != PeerRole::Unregistered &&
         peer.role != PeerRole::Viewer) ||
        ![registration validate:&validationError] ||
        !capabilityFromData(registration.sessionCapability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Invalid viewer registration or peer role."));
      return;
    }
    NSString* ownershipKey = capabilityKey(capability);
    ChromaspaceSourceExchangeBrokerPeer* owner = _viewerPeers[ownershipKey];
    if ((peer.role == PeerRole::Viewer &&
         ![peer.capability isEqualToData:registration.sessionCapability]) ||
        (owner != nil && owner != peer)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidRegistration,
                        @"Viewer session is already bound to another peer."));
      return;
    }
    ViewerRegistration value{};
    value.protocolMajor = registration.protocolMajor;
    value.protocolMinor = registration.protocolMinor;
    value.capability = capability;
    value.viewerGeneration = registration.viewerGeneration;
    value.deviceRegistryId = registration.deviceRegistryId;
    value.pixelFormatMask = registration.pixelFormatMask;
    value.maximumWidth = registration.maximumWidth;
    value.maximumHeight = registration.maximumHeight;
    value.maximumSurfaceBytes = registration.maximumSurfaceBytes;
    value.maximumRetainedBytes = registration.maximumRetainedBytes;
    value.maximumSlots = registration.maximumSlots;
    value.supportsSharedEvents = registration.supportsSharedEvents;
    const bool viewerGenerationChanged =
        peer.role == PeerRole::Viewer &&
        peer.viewerGeneration != registration.viewerGeneration;
    TransitionResult result = _state.registerViewer(value);
    releasePackets(_packets, capability, result);
    if (result.accepted()) {
      if (viewerGenerationChanged) {
        [self unbindProducerPeersForCapability:
                  registration.sessionCapability];
        [self removeViewerRouteForPeer:peer];
        peer.routeRevision = 0;
      }
      peer.role = PeerRole::Viewer;
      peer.capability = [registration.sessionCapability copy];
      peer.viewerGeneration = registration.viewerGeneration;
      peer.viewerRegistration = [registration copy];
      _viewerPeers[ownershipKey] = peer;
    }
    reply(statusForResult(result.code),
          result.accepted()
              ? nil
              : brokerError(result.code, @"Viewer registration rejected."));
  });
}

- (void)registerProducer:
            (ChromaspaceSourceExchangeProducerRegistration*)registration
                   peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                  reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                  NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    Capability capability{};
    if ((peer.role != PeerRole::Unregistered &&
         peer.role != PeerRole::Producer) ||
        ![registration validate:&validationError] ||
        !capabilityFromData(registration.sessionCapability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Invalid producer registration or peer role."));
      return;
    }
    NSString* ownershipKey =
        producerPeerKey(capability, registration.senderId);
    const bool peerRebind =
        peer.role == PeerRole::Producer &&
        (![peer.capability
             isEqualToData:registration.sessionCapability] ||
         ![peer.senderId isEqualToString:registration.senderId] ||
         peer.senderGeneration != registration.senderGeneration);
    ChromaspaceSourceExchangeBrokerPeer* owner =
        _producerPeers[ownershipKey];
    if (peerRebind || (owner != nil && owner != peer)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidRegistration,
                        @"Producer identity is already bound to another peer."));
      return;
    }
    ProducerRegistration value{};
    value.capability = capability;
    value.senderId = registration.senderId.UTF8String ?: "";
    value.senderGeneration = registration.senderGeneration;
    value.deviceRegistryId = registration.deviceRegistryId;
    TransitionResult result = _state.registerProducer(value);
    releasePackets(_packets, capability, result);
    if (result.accepted()) {
      peer.role = PeerRole::Producer;
      peer.capability = [registration.sessionCapability copy];
      peer.senderId = [registration.senderId copy];
      peer.senderGeneration = registration.senderGeneration;
      _producerPeers[ownershipKey] = peer;
    }
    reply(statusForResult(result.code),
          result.accepted()
              ? nil
              : brokerError(result.code, @"Producer registration rejected."));
  });
}

- (void)registerProducerRelayBootstrap:
            (ChromaspaceSourceExchangeRelayBootstrapRegistration*)registration
                                      peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                     reply:
                                         (void (^)(ChromaspaceSourceExchangeStatus,
                                                   NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    if (peer.role != PeerRole::Unregistered ||
        ![registration validate:&validationError]) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Invalid producer-relay bootstrap or peer role."));
      return;
    }
    NSData* token = [registration.bootstrapToken copy];
    if (_pendingBootstraps.count >=
            ChromaspaceSourceExchangeMaximumPendingBootstraps ||
        _pendingBootstraps[token] != nil) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::ResourceLimit,
                        @"Producer-relay bootstrap capacity is exhausted."));
      return;
    }
    ChromaspaceSourceExchangePendingBootstrap* pending =
        [[ChromaspaceSourceExchangePendingBootstrap alloc] init];
    pending.token = token;
    pending.endpoint = registration.producerRelayEndpoint;
    pending.owner = peer;
    _pendingBootstraps[token] = pending;
    peer.role = PeerRole::ProducerRelay;
    peer.bootstrapRedeemed = NO;
    reply(ChromaspaceSourceExchangeStatusAccepted, nil);

    dispatch_after(
        dispatch_time(DISPATCH_TIME_NOW, kRegistrationDeadlineNanoseconds),
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
          dispatch_async(_queue, ^{
            if (_pendingBootstraps[token] != pending) return;
            [_pendingBootstraps removeObjectForKey:token];
            [peer.connection invalidate];
          });
        });
  });
}

- (void)redeemProducerRelayWithToken:(NSData*)token
                               reply:
                                   (void (^)(NSXPCListenerEndpoint*,
                                             ChromaspaceSourceExchangeStatus,
                                             NSError*))reply {
  dispatch_async(_queue, ^{
    if (token.length != ChromaspaceSourceExchangeBootstrapTokenBytes) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::CapabilityMismatch,
                        @"Producer-relay bootstrap token is invalid."));
      return;
    }
    ChromaspaceSourceExchangePendingBootstrap* pending =
        _pendingBootstraps[token];
    ChromaspaceSourceExchangeBrokerPeer* owner = pending.owner;
    if (pending == nil || owner == nil || !owner.attached ||
        owner.role != PeerRole::ProducerRelay ||
        owner.bootstrapRedeemed) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::CapabilityMismatch,
                        @"Producer-relay bootstrap token is unavailable."));
      return;
    }
    [_pendingBootstraps removeObjectForKey:token];
    owner.bootstrapRedeemed = YES;
    reply(pending.endpoint, ChromaspaceSourceExchangeStatusAccepted, nil);

    dispatch_after(
        dispatch_time(DISPATCH_TIME_NOW, kRegistrationDeadlineNanoseconds),
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
          dispatch_async(_queue, ^{
            if (owner.attached &&
                owner.role == PeerRole::ProducerRelay &&
                owner.bootstrapRedeemed) {
              [owner.connection invalidate];
            }
          });
        });
  });
}

- (void)bindViewerRoute:(ChromaspaceSourceExchangeViewerRoute*)route
                   peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                  reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                  NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    Capability capability{};
    const bool peerMatches =
        peer.role == PeerRole::Viewer &&
        peer.viewerRegistration != nil &&
        [peer.capability isEqualToData:route.sessionCapability] &&
        peer.viewerGeneration == route.viewerGeneration;
    if (!peerMatches || ![route validate:&validationError] ||
        !capabilityFromData(route.sessionCapability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Viewer route does not match the registered viewer."));
      return;
    }
    ChromaspaceSourceExchangeBrokerPeer* owner =
        _viewerRoutes[route.senderId];
    if (owner != nil && owner != peer) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidRegistration,
                        @"Sender route is already owned by another viewer."));
      return;
    }
    if (route.routeRevision < peer.routeRevision) {
      reply(ChromaspaceSourceExchangeStatusStale,
            brokerError(ResultCode::Stale,
                        @"Viewer route revision is stale."));
      return;
    }
    if (route.routeRevision == peer.routeRevision) {
      if ([peer.boundSenderId isEqualToString:route.senderId]) {
        reply(ChromaspaceSourceExchangeStatusAccepted, nil);
      } else {
        reply(ChromaspaceSourceExchangeStatusRejected,
              brokerError(ResultCode::InvalidTransition,
                          @"Viewer route revision was reused for another sender."));
      }
      return;
    }
    if ([peer.boundSenderId isEqualToString:route.senderId]) {
      peer.routeRevision = route.routeRevision;
      reply(ChromaspaceSourceExchangeStatusAccepted, nil);
      return;
    }
    if (peer.boundSenderId != nil) {
      [self disconnectProducerForCapability:peer.capability
                                   senderId:peer.boundSenderId];
      [self removeViewerRouteForPeer:peer];
    }
    peer.boundSenderId = [route.senderId copy];
    peer.routeRevision = route.routeRevision;
    _viewerRoutes[route.senderId] = peer;
    reply(ChromaspaceSourceExchangeStatusAccepted, nil);
  });
}

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
               peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
              reply:
                  (void (^)(ChromaspaceSourceExchangeProducerLease*,
                            ChromaspaceSourceExchangeStatus,
                            NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    if ((peer.role != PeerRole::ProducerRelay &&
         peer.role != PeerRole::Producer) ||
        !peer.bootstrapRedeemed ||
        ![request validate:&validationError]) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Invalid producer join request or peer role."));
      return;
    }
    ChromaspaceSourceExchangeBrokerPeer* viewerPeer =
        _viewerRoutes[request.senderId];
    ChromaspaceSourceExchangeViewerRegistration* viewer =
        viewerPeer.viewerRegistration;
    Capability capability{};
    if (viewerPeer == nil || viewerPeer.role != PeerRole::Viewer ||
        !viewerPeer.attached || viewer == nil ||
        !capabilityFromData(viewer.sessionCapability, &capability)) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::SessionMissing,
                        @"No registered viewer route exists for this sender."));
      return;
    }
    if (request.protocolMajor != viewer.protocolMajor ||
        request.protocolMinor > viewer.protocolMinor) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::ProtocolMismatch,
                        @"Producer and viewer protocol versions do not match."));
      return;
    }
    NSString* ownershipKey =
        producerPeerKey(capability, request.senderId);
    const bool peerRebind =
        peer.role == PeerRole::Producer &&
        (![peer.capability isEqualToData:viewer.sessionCapability] ||
         ![peer.senderId isEqualToString:request.senderId] ||
         peer.senderGeneration != request.senderGeneration);
    ChromaspaceSourceExchangeBrokerPeer* owner =
        _producerPeers[ownershipKey];
    if (peerRebind || (owner != nil && owner != peer)) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidRegistration,
                        @"Producer route is already bound to another peer."));
      return;
    }

    ProducerRegistration value{};
    value.capability = capability;
    value.senderId = request.senderId.UTF8String ?: "";
    value.senderGeneration = request.senderGeneration;
    value.deviceRegistryId = request.deviceRegistryId;
    TransitionResult result = _state.registerProducer(value);
    releasePackets(_packets, capability, result);
    if (!result.accepted()) {
      reply(nil,
            statusForResult(result.code),
            brokerError(result.code, @"Producer join was rejected."));
      return;
    }

    peer.role = PeerRole::Producer;
    peer.capability = [viewer.sessionCapability copy];
    peer.senderId = [request.senderId copy];
    peer.senderGeneration = request.senderGeneration;
    _producerPeers[ownershipKey] = peer;

    ChromaspaceSourceExchangeProducerLease* lease =
        [[ChromaspaceSourceExchangeProducerLease alloc]
            initWithProtocolMajor:viewer.protocolMajor
                    protocolMinor:viewer.protocolMinor
                sessionCapability:viewer.sessionCapability
                 viewerGeneration:viewer.viewerGeneration
                         senderId:request.senderId
                 senderGeneration:request.senderGeneration
                 deviceRegistryId:viewer.deviceRegistryId
                  pixelFormatMask:viewer.pixelFormatMask
                     maximumWidth:viewer.maximumWidth
                    maximumHeight:viewer.maximumHeight
              maximumSurfaceBytes:viewer.maximumSurfaceBytes
             maximumRetainedBytes:viewer.maximumRetainedBytes
                     maximumSlots:viewer.maximumSlots
             supportsSharedEvents:viewer.supportsSharedEvents];
    NSError* leaseError = nil;
    if (![lease validate:&leaseError]) {
      TransitionResult rollback = _state.disconnectProducer(
          capability,
          request.senderId.UTF8String ?: "",
          request.senderGeneration);
      releasePackets(_packets, capability, rollback);
      [_producerPeers removeObjectForKey:ownershipKey];
      peer.role = PeerRole::ProducerRelay;
      peer.capability = nil;
      peer.senderId = nil;
      peer.senderGeneration = 0;
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            leaseError ?: brokerError(
                ResultCode::InvalidRegistration,
                @"Broker generated an invalid producer lease."));
      return;
    }
    reply(lease, ChromaspaceSourceExchangeStatusAccepted, nil);
  });
}

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
                  peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                 reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                 NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    Capability capability{};
    ChromaspaceSourceExchangeMetadata* metadata = packet.metadata;
    const bool peerMatches =
        peer.role == PeerRole::Producer &&
        [peer.capability isEqualToData:metadata.sessionCapability] &&
        [peer.senderId isEqualToString:metadata.senderId] &&
        peer.senderGeneration == metadata.senderGeneration;
    if (!peerMatches || ![packet validate:&validationError] ||
        !capabilityFromData(metadata.sessionCapability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidPublication,
                @"Packet does not match the registered producer peer."));
      return;
    }
    Publication value{};
    value.capability = capability;
    value.key = publicationKey(metadata);
    value.deviceRegistryId = metadata.deviceRegistryId;
    value.width = metadata.width;
    value.height = metadata.height;
    value.pixelFormat = metadata.pixelFormat;
    value.bytesPerRow = metadata.bytesPerRow;
    value.byteSize = metadata.byteSize;
    value.readyValue = metadata.readyValue;
    value.contentHash = metadata.contentHash;
    value.semantics = semanticsFromMetadata(metadata);
    TransitionResult result = _state.publish(value);
    releasePackets(_packets, capability, result);
    if (result.accepted()) {
      _packets[packetKey(capability, value.key)] = packet;
    }
    reply(statusForResult(result.code),
          result.accepted()
              ? nil
              : brokerError(result.code, @"Publication rejected."));
  });
}

- (void)acquireLatest:(NSData*)capabilityData
             senderId:(NSString*)senderId
        afterSequence:(uint64_t)afterSequence
                 peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                reply:
                    (void (^)(ChromaspaceSourceExchangePacket*,
                              ChromaspaceSourceExchangeStatus,
                              NSError*))reply {
  dispatch_async(_queue, ^{
    Capability capability{};
    if (peer.role != PeerRole::Viewer ||
        ![peer.capability isEqualToData:capabilityData] ||
        !capabilityFromData(capabilityData, &capability)) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::CapabilityMismatch,
                        @"Acquire does not match the registered viewer."));
      return;
    }
    Publication publication{};
    TransitionResult result =
        _state.acquireLatest(
            capability, senderId.UTF8String ?: "", afterSequence, &publication);
    releasePackets(_packets, capability, result);
    if (result.code == ResultCode::NoNewPublication) {
      reply(nil, ChromaspaceSourceExchangeStatusNoNewPublication, nil);
      return;
    }
    if (!result.accepted()) {
      reply(nil,
            statusForResult(result.code),
            brokerError(result.code, @"Acquire rejected."));
      return;
    }
    ChromaspaceSourceExchangePacket* packet =
        _packets[packetKey(capability, publication.key)];
    if (packet == nil) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidTransition,
                        @"Broker packet retention invariant failed."));
      return;
    }
    reply(packet, ChromaspaceSourceExchangeStatusAccepted, nil);
  });
}

- (void)acknowledge:
            (ChromaspaceSourceExchangeAcknowledgement*)acknowledgement
               peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
              reply:(void (^)(ChromaspaceSourceExchangeStatus,
                              NSError*))reply {
  dispatch_async(_queue, ^{
    NSError* validationError = nil;
    Capability capability{};
    if (peer.role != PeerRole::Viewer ||
        ![peer.capability isEqualToData:acknowledgement.sessionCapability] ||
        ![acknowledgement validate:&validationError] ||
        !capabilityFromData(acknowledgement.sessionCapability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::CapabilityMismatch,
                @"Acknowledgement does not match the registered viewer."));
      return;
    }
    PublicationKey key{};
    key.senderId = acknowledgement.senderId.UTF8String ?: "";
    key.senderGeneration = acknowledgement.senderGeneration;
    key.sequence = acknowledgement.sequence;
    key.slotIndex = acknowledgement.slotIndex;
    key.slotGeneration = acknowledgement.slotGeneration;
    const AcknowledgementState state =
        acknowledgement.status == ChromaspaceSourceExchangeStatusAcquired
            ? AcknowledgementState::Acquired
            : AcknowledgementState::Retired;
    TransitionResult result = _state.acknowledge(capability, key, state);
    releasePackets(_packets, capability, result);
    ChromaspaceSourceExchangeStatus status = statusForResult(result.code);
    if (result.accepted()) {
      status = state == AcknowledgementState::Acquired
                   ? ChromaspaceSourceExchangeStatusAcquired
                   : ChromaspaceSourceExchangeStatusRetired;
    }
    reply(status,
          result.accepted()
              ? nil
              : brokerError(result.code, @"Acknowledgement rejected."));
  });
}

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                     peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                    reply:
                                        (void (^)(ChromaspaceSourceExchangeReleaseBatch*,
                                                  ChromaspaceSourceExchangeStatus,
                                                  NSError*))reply {
  dispatch_async(_queue, ^{
    Capability capability{};
    if (peer.role != PeerRole::Producer ||
        !capabilityFromData(peer.capability, &capability)) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::CapabilityMismatch,
                        @"Release fetch does not match a registered producer."));
      return;
    }
    ProducerReleaseBatch batch{};
    TransitionResult result = _state.fetchProducerReleases(
        capability,
        peer.senderId.UTF8String ?: "",
        peer.senderGeneration,
        afterOrdinal,
        maximumEvents,
        &batch);
    if (result.code == ResultCode::NoNewPublication) {
      reply(nil, ChromaspaceSourceExchangeStatusNoNewPublication, nil);
      return;
    }
    if (!result.accepted()) {
      reply(nil,
            statusForResult(result.code),
            brokerError(result.code, @"Producer release fetch rejected."));
      if (result.code == ResultCode::ResourceLimit) {
        [peer.connection invalidate];
      }
      return;
    }
    NSMutableArray<ChromaspaceSourceExchangeReleaseEvent*>* events =
        [NSMutableArray arrayWithCapacity:batch.events.size()];
    for (const ProducerReleaseEvent& event : batch.events) {
      ChromaspaceSourceExchangeReleaseEvent* value =
          [[ChromaspaceSourceExchangeReleaseEvent alloc]
              initWithOrdinal:event.ordinal
                     senderId:
                         [NSString stringWithUTF8String:
                                       event.key.senderId.c_str()]
             senderGeneration:event.key.senderGeneration
                     sequence:event.key.sequence
                    slotIndex:event.key.slotIndex
               slotGeneration:event.key.slotGeneration];
      [events addObject:value];
    }
    ChromaspaceSourceExchangeReleaseBatch* value =
        [[ChromaspaceSourceExchangeReleaseBatch alloc]
            initWithSessionCapability:peer.capability
                              senderId:peer.senderId
                      senderGeneration:peer.senderGeneration
                       throughOrdinal:batch.throughOrdinal
                                events:events];
    NSError* validationError = nil;
    if (![value validate:&validationError]) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            validationError ?: brokerError(
                ResultCode::InvalidTransition,
                @"Broker generated an invalid producer release batch."));
      [peer.connection invalidate];
      return;
    }
    reply(value, ChromaspaceSourceExchangeStatusAccepted, nil);
  });
}

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                             peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                                            reply:(void (^)(ChromaspaceSourceExchangeStatus,
                                                            NSError*))reply {
  dispatch_async(_queue, ^{
    Capability capability{};
    if (peer.role != PeerRole::Producer ||
        !capabilityFromData(peer.capability, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::CapabilityMismatch,
                        @"Release acknowledgement does not match a producer."));
      return;
    }
    TransitionResult result = _state.acknowledgeProducerReleases(
        capability,
        peer.senderId.UTF8String ?: "",
        peer.senderGeneration,
        throughOrdinal);
    reply(statusForResult(result.code),
          result.accepted()
              ? nil
              : brokerError(
                    result.code,
                    @"Producer release acknowledgement rejected."));
  });
}

- (void)disconnectSession:(NSData*)capabilityData
                      peer:(ChromaspaceSourceExchangeBrokerPeer*)peer
                     reply:(void (^)(ChromaspaceSourceExchangeStatus))reply {
  dispatch_async(_queue, ^{
    Capability capability{};
    if (peer.role != PeerRole::Viewer ||
        ![peer.capability isEqualToData:capabilityData] ||
        !capabilityFromData(capabilityData, &capability)) {
      reply(ChromaspaceSourceExchangeStatusRejected);
      return;
    }
    TransitionResult result = _state.disconnectViewer(capability);
    releasePackets(_packets, capability, result);
    if (result.accepted()) {
      [_viewerPeers removeObjectForKey:capabilityKey(capability)];
      [self removeViewerRouteForPeer:peer];
      [self unbindProducerPeersForCapability:capabilityData];
      peer.role = PeerRole::Unregistered;
      peer.capability = nil;
      peer.viewerGeneration = 0;
      peer.routeRevision = 0;
      peer.viewerRegistration = nil;
    }
    reply(statusForResult(result.code));
  });
}

- (void)peerInvalidated:(ChromaspaceSourceExchangeBrokerPeer*)peer {
  dispatch_async(_queue, ^{
    if (!peer.attached) return;
    peer.attached = NO;
    if (_livePeerCount > 0) {
      --_livePeerCount;
    }
    NSMutableArray<NSData*>* bootstrapTokens =
        [NSMutableArray array];
    [_pendingBootstraps
        enumerateKeysAndObjectsUsingBlock:
            ^(NSData* token,
              ChromaspaceSourceExchangePendingBootstrap* pending,
              BOOL* stop) {
              (void)stop;
              if (pending.owner == peer) [bootstrapTokens addObject:token];
            }];
    [_pendingBootstraps removeObjectsForKeys:bootstrapTokens];
    Capability capability{};
    if (!capabilityFromData(peer.capability, &capability)) {
      peer.role = PeerRole::Unregistered;
      peer.bootstrapRedeemed = NO;
      return;
    }
    TransitionResult result{};
    if (peer.role == PeerRole::Viewer) {
      result = _state.disconnectViewer(capability);
      [_viewerPeers removeObjectForKey:capabilityKey(capability)];
      [self removeViewerRouteForPeer:peer];
      [self unbindProducerPeersForCapability:peer.capability];
    } else if (peer.role == PeerRole::Producer) {
      result = _state.disconnectProducer(
          capability,
          peer.senderId.UTF8String ?: "",
          peer.senderGeneration);
      [_producerPeers
          removeObjectForKey:producerPeerKey(capability, peer.senderId)];
    } else {
      return;
    }
    releasePackets(_packets, capability, result);
    peer.role = PeerRole::Unregistered;
    peer.capability = nil;
    peer.senderId = nil;
    peer.viewerGeneration = 0;
    peer.senderGeneration = 0;
    peer.routeRevision = 0;
    peer.bootstrapRedeemed = NO;
    peer.viewerRegistration = nil;
  });
}

- (BOOL)tryAttachPeer:(ChromaspaceSourceExchangeBrokerPeer*)peer {
  __block BOOL accepted = NO;
  dispatch_sync(_queue, ^{
    if (!peer.attached && _livePeerCount < kMaximumLivePeers) {
      peer.attached = YES;
      ++_livePeerCount;
      accepted = YES;
    }
  });
  return accepted;
}

- (void)expirePeerIfUnregistered:
    (ChromaspaceSourceExchangeBrokerPeer*)peer {
  dispatch_async(_queue, ^{
    if (peer.attached && peer.role == PeerRole::Unregistered) {
      [peer.connection invalidate];
    }
  });
}

@end

@implementation ChromaspaceSourceExchangeBrokerPeer

- (void)registerViewer:
            (ChromaspaceSourceExchangeViewerRegistration*)registration
             withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                                 NSError*))reply {
  [self.coordinator registerViewer:registration peer:self reply:reply];
}

- (void)registerProducer:
            (ChromaspaceSourceExchangeProducerRegistration*)registration
               withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                                   NSError*))reply {
  [self.coordinator registerProducer:registration peer:self reply:reply];
}

- (void)registerProducerRelayBootstrap:
            (ChromaspaceSourceExchangeRelayBootstrapRegistration*)registration
                               withReply:
                                   (void (^)(ChromaspaceSourceExchangeStatus,
                                             NSError*))reply {
  [self.coordinator registerProducerRelayBootstrap:registration
                                              peer:self
                                             reply:reply];
}

- (void)bindViewerRoute:(ChromaspaceSourceExchangeViewerRoute*)route
              withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                                  NSError*))reply {
  [self.coordinator bindViewerRoute:route peer:self reply:reply];
}

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
          withReply:
              (void (^)(ChromaspaceSourceExchangeProducerLease*,
                        ChromaspaceSourceExchangeStatus,
                        NSError*))reply {
  [self.coordinator joinProducer:request peer:self reply:reply];
}

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
            withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                                NSError*))reply {
  [self.coordinator publishPacket:packet peer:self reply:reply];
}

- (void)acquireLatestForSession:(NSData*)sessionCapability
                       senderId:(NSString*)senderId
                 afterSequence:(uint64_t)afterSequence
                      withReply:
                          (void (^)(ChromaspaceSourceExchangePacket*,
                                    ChromaspaceSourceExchangeStatus,
                                    NSError*))reply {
  [self.coordinator acquireLatest:sessionCapability
                         senderId:senderId
                    afterSequence:afterSequence
                             peer:self
                            reply:reply];
}

- (void)acknowledge:
            (ChromaspaceSourceExchangeAcknowledgement*)acknowledgement
          withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                              NSError*))reply {
  [self.coordinator acknowledge:acknowledgement peer:self reply:reply];
}

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                withReply:
                                    (void (^)(ChromaspaceSourceExchangeReleaseBatch*,
                                              ChromaspaceSourceExchangeStatus,
                                              NSError*))reply {
  [self.coordinator fetchProducerReleasesAfterOrdinal:afterOrdinal
                                        maximumEvents:maximumEvents
                                                 peer:self
                                                reply:reply];
}

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                         withReply:
                                             (void (^)(ChromaspaceSourceExchangeStatus,
                                                       NSError*))reply {
  [self.coordinator
      acknowledgeProducerReleasesThroughOrdinal:throughOrdinal
                                           peer:self
                                          reply:reply];
}

- (void)disconnectSession:(NSData*)sessionCapability
                withReply:(void (^)(ChromaspaceSourceExchangeStatus))reply {
  [self.coordinator disconnectSession:sessionCapability peer:self reply:reply];
}

@end

@interface ChromaspaceSourceExchangeListenerDelegate
    : NSObject <NSXPCListenerDelegate>

@property(nonatomic, strong)
    ChromaspaceSourceExchangeBrokerCoordinator* coordinator;

@end

@implementation ChromaspaceSourceExchangeListenerDelegate

- (BOOL)listener:(NSXPCListener*)listener
    shouldAcceptNewConnection:(NSXPCConnection*)connection {
  (void)listener;
  if (connection == nil ||
      connection.effectiveUserIdentifier != geteuid()) {
    return NO;
  }
  ChromaspaceSourceExchangeBrokerPeer* peer =
      [[ChromaspaceSourceExchangeBrokerPeer alloc] init];
  peer.connection = connection;
  peer.coordinator = self.coordinator;
  peer.role = PeerRole::Unregistered;
  if (![self.coordinator tryAttachPeer:peer]) {
    return NO;
  }
  connection.exportedInterface = brokerInterface();
  connection.exportedObject = peer;
  ChromaspaceSourceExchangeBrokerCoordinator* coordinator = self.coordinator;
  connection.invalidationHandler = ^{
    [coordinator peerInvalidated:peer];
  };
  [connection resume];
  dispatch_after(
      dispatch_time(DISPATCH_TIME_NOW,
                    kRegistrationDeadlineNanoseconds),
      dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [coordinator expirePeerIfUnregistered:peer];
      });
  return YES;
}

@end

@class ChromaspaceSourceExchangeHostBootstrapListenerDelegate;

@interface ChromaspaceSourceExchangeHostBootstrapPeer
    : NSObject <ChromaspaceSourceExchangeHostBootstrapProtocol>

@property(nonatomic, weak) NSXPCConnection* connection;
@property(nonatomic, strong)
    ChromaspaceSourceExchangeBrokerCoordinator* coordinator;
@property(nonatomic, weak)
    ChromaspaceSourceExchangeHostBootstrapListenerDelegate* listenerDelegate;
@property(nonatomic) BOOL attached;
@property(nonatomic) BOOL attemptedRedemption;

@end

@interface ChromaspaceSourceExchangeHostBootstrapListenerDelegate
    : NSObject <NSXPCListenerDelegate>

@property(nonatomic, strong)
    ChromaspaceSourceExchangeBrokerCoordinator* coordinator;
@property(nonatomic) NSUInteger livePeers;

- (void)peerInvalidated:
    (ChromaspaceSourceExchangeHostBootstrapPeer*)peer;

@end

@implementation ChromaspaceSourceExchangeHostBootstrapPeer

- (void)redeemProducerRelayWithToken:(NSData*)bootstrapToken
                           withReply:
                               (void (^)(NSXPCListenerEndpoint*,
                                         ChromaspaceSourceExchangeStatus,
                                         NSError*))reply {
  @synchronized(self) {
    if (self.attemptedRedemption) {
      reply(nil,
            ChromaspaceSourceExchangeStatusRejected,
            brokerError(ResultCode::InvalidTransition,
                        @"Bootstrap connection is one-shot."));
      return;
    }
    self.attemptedRedemption = YES;
  }
  __weak NSXPCConnection* weakConnection = self.connection;
  [self.coordinator
      redeemProducerRelayWithToken:bootstrapToken
                             reply:^(NSXPCListenerEndpoint* endpoint,
                                     ChromaspaceSourceExchangeStatus status,
                                     NSError* error) {
                               reply(endpoint, status, error);
                               dispatch_after(
                                   dispatch_time(
                                       DISPATCH_TIME_NOW,
                                       NSEC_PER_SEC),
                                   dispatch_get_global_queue(
                                       QOS_CLASS_UTILITY, 0), ^{
                                     [weakConnection invalidate];
                                   });
                             }];
}

@end

@implementation ChromaspaceSourceExchangeHostBootstrapListenerDelegate

- (BOOL)listener:(NSXPCListener*)listener
    shouldAcceptNewConnection:(NSXPCConnection*)connection {
  (void)listener;
  if (connection == nil ||
      connection.effectiveUserIdentifier != geteuid()) {
    return NO;
  }
  ChromaspaceSourceExchangeHostBootstrapPeer* peer =
      [[ChromaspaceSourceExchangeHostBootstrapPeer alloc] init];
  @synchronized(self) {
    if (self.livePeers >=
        ChromaspaceSourceExchangeMaximumPendingBootstraps) {
      return NO;
    }
    ++self.livePeers;
    peer.attached = YES;
  }
  peer.connection = connection;
  peer.coordinator = self.coordinator;
  peer.listenerDelegate = self;
  connection.exportedInterface = hostBootstrapInterface();
  connection.exportedObject = peer;
  __weak ChromaspaceSourceExchangeHostBootstrapListenerDelegate*
      weakDelegate = self;
  connection.invalidationHandler = ^{
    [weakDelegate peerInvalidated:peer];
  };
  [connection resume];
  dispatch_after(
      dispatch_time(DISPATCH_TIME_NOW, kRegistrationDeadlineNanoseconds),
      dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        [connection invalidate];
      });
  return YES;
}

- (void)peerInvalidated:
    (ChromaspaceSourceExchangeHostBootstrapPeer*)peer {
  @synchronized(self) {
    if (!peer.attached) return;
    peer.attached = NO;
    if (self.livePeers > 0) --self.livePeers;
  }
}

@end

int main(int argc, const char* argv[]) {
  (void)argc;
  (void)argv;
  @autoreleasepool {
    ChromaspaceSourceExchangeListenerDelegate*
        __attribute__((objc_precise_lifetime)) delegate =
        [[ChromaspaceSourceExchangeListenerDelegate alloc] init];
    ChromaspaceSourceExchangeBrokerCoordinator*
        __attribute__((objc_precise_lifetime)) coordinator =
        [[ChromaspaceSourceExchangeBrokerCoordinator alloc] init];
    delegate.coordinator = coordinator;
    NSXPCListener* __attribute__((objc_precise_lifetime)) listener =
        [[NSXPCListener alloc]
            initWithMachServiceName:
                ChromaspaceSourceExchangeMachServiceName];
    listener.delegate = delegate;
    [listener resume];

    ChromaspaceSourceExchangeHostBootstrapListenerDelegate*
        __attribute__((objc_precise_lifetime)) bootstrapDelegate =
        [[ChromaspaceSourceExchangeHostBootstrapListenerDelegate alloc] init];
    bootstrapDelegate.coordinator = coordinator;
    NSXPCListener* __attribute__((objc_precise_lifetime)) bootstrapListener =
        [[NSXPCListener alloc]
            initWithMachServiceName:
                ChromaspaceSourceExchangeBootstrapMachServiceName];
    bootstrapListener.delegate = bootstrapDelegate;
    [bootstrapListener resume];
    [[NSRunLoop currentRunLoop] run];
  }
  return 0;
}
