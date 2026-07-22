#pragma once

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSString* const ChromaspaceSourceExchangeErrorDomain;
FOUNDATION_EXPORT NSString* const ChromaspaceSourceExchangeMachServiceName;
FOUNDATION_EXPORT NSString* const
    ChromaspaceSourceExchangeBootstrapMachServiceName;

enum : NSUInteger {
  ChromaspaceSourceExchangeProtocolMajor = 2,
  ChromaspaceSourceExchangeProtocolMinor = 0,
  ChromaspaceSourceExchangeCapabilityBytes = 32,
  ChromaspaceSourceExchangeBootstrapTokenBytes = 32,
  ChromaspaceSourceExchangeMaximumSlots = 3,
  ChromaspaceSourceExchangeMaximumSessions = 32,
  ChromaspaceSourceExchangeMaximumPendingBootstraps = 64,
  ChromaspaceSourceExchangeMaximumReleaseEvents = 64,
  ChromaspaceSourceExchangeMaximumDimension = 16384,
  ChromaspaceSourceExchangeMaximumIdentityResolution = 256,
  ChromaspaceSourceExchangeMaximumSemanticIdentifierBytes = 64,
  ChromaspaceSourceExchangePixelFormatRGBA16Float = 1u << 0,
  ChromaspaceSourceExchangePixelFormatRGBA32Float = 1u << 1,
};

static const uint64_t ChromaspaceSourceExchangeMaximumSurfaceBytes =
    1024ull * 1024ull * 1024ull;
static const uint64_t ChromaspaceSourceExchangeMaximumRetainedBytes =
    2ull * 1024ull * 1024ull * 1024ull;

typedef NS_ERROR_ENUM(ChromaspaceSourceExchangeErrorDomain,
                      ChromaspaceSourceExchangeErrorCode) {
  ChromaspaceSourceExchangeErrorInvalidProtocol = 1,
  ChromaspaceSourceExchangeErrorInvalidCapability,
  ChromaspaceSourceExchangeErrorInvalidIdentity,
  ChromaspaceSourceExchangeErrorInvalidGeneration,
  ChromaspaceSourceExchangeErrorInvalidSlot,
  ChromaspaceSourceExchangeErrorInvalidSurface,
  ChromaspaceSourceExchangeErrorInvalidSynchronization,
  ChromaspaceSourceExchangeErrorDeviceMismatch,
  ChromaspaceSourceExchangeErrorResourceLimit,
  ChromaspaceSourceExchangeErrorStalePublication,
  ChromaspaceSourceExchangeErrorSessionMissing,
};

typedef NS_ENUM(NSInteger, ChromaspaceSourceExchangeStatus) {
  ChromaspaceSourceExchangeStatusAccepted = 0,
  ChromaspaceSourceExchangeStatusAcquired,
  ChromaspaceSourceExchangeStatusRetired,
  ChromaspaceSourceExchangeStatusNoNewPublication,
  ChromaspaceSourceExchangeStatusStale,
  ChromaspaceSourceExchangeStatusRejected,
};

@interface ChromaspaceSourceExchangeViewerRegistration
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly) uint64_t viewerGeneration;
@property(nonatomic, readonly) uint64_t deviceRegistryId;
@property(nonatomic, readonly) uint32_t pixelFormatMask;
@property(nonatomic, readonly) uint32_t maximumWidth;
@property(nonatomic, readonly) uint32_t maximumHeight;
@property(nonatomic, readonly) uint64_t maximumSurfaceBytes;
@property(nonatomic, readonly) uint64_t maximumRetainedBytes;
@property(nonatomic, readonly) uint32_t maximumSlots;
@property(nonatomic, readonly) BOOL supportsSharedEvents;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                    sessionCapability:(NSData*)sessionCapability
                     viewerGeneration:(uint64_t)viewerGeneration
                     deviceRegistryId:(uint64_t)deviceRegistryId
                      pixelFormatMask:(uint32_t)pixelFormatMask
                         maximumWidth:(uint32_t)maximumWidth
                        maximumHeight:(uint32_t)maximumHeight
                  maximumSurfaceBytes:(uint64_t)maximumSurfaceBytes
                 maximumRetainedBytes:(uint64_t)maximumRetainedBytes
                         maximumSlots:(uint32_t)maximumSlots
                 supportsSharedEvents:(BOOL)supportsSharedEvents
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeViewerRoute
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly) uint64_t viewerGeneration;
@property(nonatomic, readonly) uint64_t routeRevision;
@property(nonatomic, readonly, copy) NSString* senderId;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                    sessionCapability:(NSData*)sessionCapability
                     viewerGeneration:(uint64_t)viewerGeneration
                        routeRevision:(uint64_t)routeRevision
                             senderId:(NSString*)senderId
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

// Created by the signed relay and transferred only over its authenticated
// broker XPC connection. The endpoint contains a Mach right and therefore must
// never be flattened into a pipe, file, JSON value, or ordinary keyed archive.
@interface ChromaspaceSourceExchangeRelayBootstrapRegistration
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSData* bootstrapToken;
@property(nonatomic, readonly, strong)
    NSXPCListenerEndpoint* producerRelayEndpoint;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                       bootstrapToken:(NSData*)bootstrapToken
                producerRelayEndpoint:
                    (NSXPCListenerEndpoint*)producerRelayEndpoint
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeProducerJoinRequest
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t deviceRegistryId;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                             senderId:(NSString*)senderId
                     senderGeneration:(uint64_t)senderGeneration
                     deviceRegistryId:(uint64_t)deviceRegistryId
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeProducerLease
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly) uint64_t viewerGeneration;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t deviceRegistryId;
@property(nonatomic, readonly) uint32_t pixelFormatMask;
@property(nonatomic, readonly) uint32_t maximumWidth;
@property(nonatomic, readonly) uint32_t maximumHeight;
@property(nonatomic, readonly) uint64_t maximumSurfaceBytes;
@property(nonatomic, readonly) uint64_t maximumRetainedBytes;
@property(nonatomic, readonly) uint32_t maximumSlots;
@property(nonatomic, readonly) BOOL supportsSharedEvents;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                    sessionCapability:(NSData*)sessionCapability
                     viewerGeneration:(uint64_t)viewerGeneration
                             senderId:(NSString*)senderId
                     senderGeneration:(uint64_t)senderGeneration
                     deviceRegistryId:(uint64_t)deviceRegistryId
                      pixelFormatMask:(uint32_t)pixelFormatMask
                         maximumWidth:(uint32_t)maximumWidth
                        maximumHeight:(uint32_t)maximumHeight
                  maximumSurfaceBytes:(uint64_t)maximumSurfaceBytes
                 maximumRetainedBytes:(uint64_t)maximumRetainedBytes
                         maximumSlots:(uint32_t)maximumSlots
                 supportsSharedEvents:(BOOL)supportsSharedEvents
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeProducerRegistration
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t deviceRegistryId;

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                          deviceRegistryId:(uint64_t)deviceRegistryId
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeMetadata
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint32_t protocolMajor;
@property(nonatomic, readonly) uint32_t protocolMinor;
@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t sequence;
@property(nonatomic, readonly) uint32_t slotIndex;
@property(nonatomic, readonly) uint64_t slotGeneration;
@property(nonatomic, readonly) uint64_t readyValue;
@property(nonatomic, readonly) uint64_t deviceRegistryId;
@property(nonatomic, readonly) uint32_t width;
@property(nonatomic, readonly) uint32_t height;
@property(nonatomic, readonly) uint32_t pixelFormat;  // 0=RGBA16F, 1=RGBA32F.
@property(nonatomic, readonly) uint64_t bytesPerRow;
@property(nonatomic, readonly) uint64_t byteSize;
@property(nonatomic, readonly) uint64_t contentHash;
@property(nonatomic, readonly) int32_t sourceX;
@property(nonatomic, readonly) int32_t sourceY;
@property(nonatomic, readonly) uint32_t sourceWidth;
@property(nonatomic, readonly) uint32_t sourceHeight;
@property(nonatomic, readonly) int32_t sampledX;
@property(nonatomic, readonly) int32_t sampledY;
@property(nonatomic, readonly) uint32_t sampledWidth;
@property(nonatomic, readonly) uint32_t sampledHeight;
@property(nonatomic, readonly) BOOL authoritative;
@property(nonatomic, readonly, copy) NSString* coverage;
@property(nonatomic, readonly) BOOL identityStripPresent;
@property(nonatomic, readonly) BOOL identityCube;
@property(nonatomic, readonly) BOOL identityRamp;
@property(nonatomic, readonly) uint32_t identityResolution;
@property(nonatomic, readonly) uint32_t identityBandHeight;
@property(nonatomic, readonly) int32_t identityCubeY1;
@property(nonatomic, readonly) int32_t identityCubeY2;
@property(nonatomic, readonly) int32_t identityRampY1;
@property(nonatomic, readonly) int32_t identityRampY2;
@property(nonatomic, readonly, copy) NSString* colorPrimaries;
@property(nonatomic, readonly, copy) NSString* transferFunction;

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                    sessionCapability:(NSData*)sessionCapability
                             senderId:(NSString*)senderId
                     senderGeneration:(uint64_t)senderGeneration
                             sequence:(uint64_t)sequence
                            slotIndex:(uint32_t)slotIndex
                       slotGeneration:(uint64_t)slotGeneration
                           readyValue:(uint64_t)readyValue
                     deviceRegistryId:(uint64_t)deviceRegistryId
                                width:(uint32_t)width
                               height:(uint32_t)height
                          pixelFormat:(uint32_t)pixelFormat
                          bytesPerRow:(uint64_t)bytesPerRow
                             byteSize:(uint64_t)byteSize
                          contentHash:(uint64_t)contentHash
                             sourceX:(int32_t)sourceX
                             sourceY:(int32_t)sourceY
                         sourceWidth:(uint32_t)sourceWidth
                        sourceHeight:(uint32_t)sourceHeight
                            sampledX:(int32_t)sampledX
                            sampledY:(int32_t)sampledY
                        sampledWidth:(uint32_t)sampledWidth
                       sampledHeight:(uint32_t)sampledHeight
                        authoritative:(BOOL)authoritative
                             coverage:(NSString*)coverage
                 identityStripPresent:(BOOL)identityStripPresent
                         identityCube:(BOOL)identityCube
                         identityRamp:(BOOL)identityRamp
                   identityResolution:(uint32_t)identityResolution
                   identityBandHeight:(uint32_t)identityBandHeight
                       identityCubeY1:(int32_t)identityCubeY1
                       identityCubeY2:(int32_t)identityCubeY2
                       identityRampY1:(int32_t)identityRampY1
                       identityRampY2:(int32_t)identityRampY2
                       colorPrimaries:(NSString*)colorPrimaries
                     transferFunction:(NSString*)transferFunction
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangePacket
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly, copy) ChromaspaceSourceExchangeMetadata* metadata;
@property(nonatomic, readonly, strong) MTLSharedTextureHandle* textureHandle;
@property(nonatomic, readonly, strong) MTLSharedEventHandle* eventHandle;

- (instancetype)initWithMetadata:(ChromaspaceSourceExchangeMetadata*)metadata
                   textureHandle:(MTLSharedTextureHandle*)textureHandle
                     eventHandle:(MTLSharedEventHandle*)eventHandle
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeReleaseEvent
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly) uint64_t ordinal;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t sequence;
@property(nonatomic, readonly) uint32_t slotIndex;
@property(nonatomic, readonly) uint64_t slotGeneration;

- (instancetype)initWithOrdinal:(uint64_t)ordinal
                       senderId:(NSString*)senderId
               senderGeneration:(uint64_t)senderGeneration
                       sequence:(uint64_t)sequence
                      slotIndex:(uint32_t)slotIndex
                 slotGeneration:(uint64_t)slotGeneration
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeReleaseBatch
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t throughOrdinal;
@property(nonatomic, readonly, copy)
    NSArray<ChromaspaceSourceExchangeReleaseEvent*>* events;

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                           throughOrdinal:(uint64_t)throughOrdinal
                                    events:
                                        (NSArray<ChromaspaceSourceExchangeReleaseEvent*>*)events
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@interface ChromaspaceSourceExchangeAcknowledgement
    : NSObject <NSSecureCoding, NSCopying>

@property(nonatomic, readonly, copy) NSData* sessionCapability;
@property(nonatomic, readonly, copy) NSString* senderId;
@property(nonatomic, readonly) uint64_t senderGeneration;
@property(nonatomic, readonly) uint64_t sequence;
@property(nonatomic, readonly) uint32_t slotIndex;
@property(nonatomic, readonly) uint64_t slotGeneration;
@property(nonatomic, readonly) ChromaspaceSourceExchangeStatus status;

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                                  sequence:(uint64_t)sequence
                                 slotIndex:(uint32_t)slotIndex
                            slotGeneration:(uint64_t)slotGeneration
                                    status:(ChromaspaceSourceExchangeStatus)status
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
- (BOOL)validate:(NSError* _Nullable* _Nullable)error;

@end

@protocol ChromaspaceSourceExchangeBrokerProtocol

- (void)registerViewer:
            (ChromaspaceSourceExchangeViewerRegistration*)registration
             withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                 NSError* _Nullable error))reply;

- (void)registerProducer:
            (ChromaspaceSourceExchangeProducerRegistration*)registration
               withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                   NSError* _Nullable error))reply;

- (void)registerProducerRelayBootstrap:
            (ChromaspaceSourceExchangeRelayBootstrapRegistration*)registration
                               withReply:
                                   (void (^)(ChromaspaceSourceExchangeStatus status,
                                             NSError* _Nullable error))reply;

- (void)bindViewerRoute:(ChromaspaceSourceExchangeViewerRoute*)route
              withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                  NSError* _Nullable error))reply;

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
          withReply:
              (void (^)(ChromaspaceSourceExchangeProducerLease* _Nullable lease,
                        ChromaspaceSourceExchangeStatus status,
                        NSError* _Nullable error))reply;

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
            withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                NSError* _Nullable error))reply;

- (void)acquireLatestForSession:(NSData*)sessionCapability
                       senderId:(NSString*)senderId
                 afterSequence:(uint64_t)afterSequence
                      withReply:
                          (void (^)(ChromaspaceSourceExchangePacket* _Nullable packet,
                                    ChromaspaceSourceExchangeStatus status,
                                    NSError* _Nullable error))reply;

- (void)acknowledge:(ChromaspaceSourceExchangeAcknowledgement*)acknowledgement
          withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                              NSError* _Nullable error))reply;

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                withReply:
                                    (void (^)(ChromaspaceSourceExchangeReleaseBatch* _Nullable batch,
                                              ChromaspaceSourceExchangeStatus status,
                                              NSError* _Nullable error))reply;

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                         withReply:
                                             (void (^)(ChromaspaceSourceExchangeStatus status,
                                                       NSError* _Nullable error))reply;

- (void)disconnectSession:(NSData*)sessionCapability
                withReply:(void (^)(ChromaspaceSourceExchangeStatus status))reply;

@end

// Least-privilege interface used from the signed producer relay to the broker.
// The broker may export a protocol superset, but the relay never receives
// viewer registration, routing, acquisition, or disconnect selectors.
@protocol ChromaspaceSourceExchangeProducerBrokerProtocol

- (void)registerProducerRelayBootstrap:
            (ChromaspaceSourceExchangeRelayBootstrapRegistration*)registration
                               withReply:
                                   (void (^)(ChromaspaceSourceExchangeStatus status,
                                             NSError* _Nullable error))reply;

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
          withReply:
              (void (^)(ChromaspaceSourceExchangeProducerLease* _Nullable lease,
                        ChromaspaceSourceExchangeStatus status,
                        NSError* _Nullable error))reply;

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
            withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                NSError* _Nullable error))reply;

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                withReply:
                                    (void (^)(ChromaspaceSourceExchangeReleaseBatch* _Nullable batch,
                                              ChromaspaceSourceExchangeStatus status,
                                              NSError* _Nullable error))reply;

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                         withReply:
                                             (void (^)(ChromaspaceSourceExchangeStatus status,
                                                       NSError* _Nullable error))reply;

@end

// Private, single-host interface exported by the separately signed producer
// relay. It deliberately exposes no viewer registration, routing, acquisition,
// or session-disconnect operations.
@protocol ChromaspaceSourceExchangeProducerRelayProtocol

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
          withReply:
              (void (^)(ChromaspaceSourceExchangeProducerLease* _Nullable lease,
                        ChromaspaceSourceExchangeStatus status,
                        NSError* _Nullable error))reply;

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
            withReply:(void (^)(ChromaspaceSourceExchangeStatus status,
                                NSError* _Nullable error))reply;

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                withReply:
                                    (void (^)(ChromaspaceSourceExchangeReleaseBatch* _Nullable batch,
                                              ChromaspaceSourceExchangeStatus status,
                                              NSError* _Nullable error))reply;

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                         withReply:
                                             (void (^)(ChromaspaceSourceExchangeStatus status,
                                                       NSError* _Nullable error))reply;

@end

// Publicly discoverable but capability-minimal bootstrap surface. It returns
// only a one-shot anonymous endpoint for an exact 256-bit token and never
// exposes broker, viewer, producer, routing, or publication methods.
@protocol ChromaspaceSourceExchangeHostBootstrapProtocol

- (void)redeemProducerRelayWithToken:(NSData*)bootstrapToken
                           withReply:
                               (void (^)(NSXPCListenerEndpoint* _Nullable endpoint,
                                         ChromaspaceSourceExchangeStatus status,
                                         NSError* _Nullable error))reply;

@end

NS_ASSUME_NONNULL_END

#endif  // defined(__APPLE__)
