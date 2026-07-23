#import <Foundation/Foundation.h>

#import "ChromaspaceSourceExchangeV2.h"

#include <cstdlib>
#include <cstring>

namespace {

void require(BOOL condition, NSString* message) {
  if (condition) return;
  NSLog(@"SourceExchangeV2 contract test failed: %@", message);
  std::abort();
}

NSData* testCapability() {
  uint8_t bytes[ChromaspaceSourceExchangeCapabilityBytes] = {};
  for (NSUInteger i = 0; i < sizeof(bytes); ++i) {
    bytes[i] = static_cast<uint8_t>(i + 1);
  }
  return [NSData dataWithBytes:bytes length:sizeof(bytes)];
}

template <typename T>
T* secureRoundTrip(T* value, Class expectedClass) {
  NSError* error = nil;
  NSData* archive =
      [NSKeyedArchiver archivedDataWithRootObject:value
                           requiringSecureCoding:YES
                                           error:&error];
  require(archive != nil && error == nil, @"secure archive");
  T* decoded =
      [NSKeyedUnarchiver unarchivedObjectOfClass:expectedClass
                                       fromData:archive
                                          error:&error];
  require(decoded != nil && error == nil, @"secure unarchive");
  return decoded;
}

}  // namespace

int main() {
  @autoreleasepool {
    NSData* capability = testCapability();
    NSError* error = nil;

    ChromaspaceSourceExchangeViewerRegistration* viewer =
        [[ChromaspaceSourceExchangeViewerRegistration alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                 viewerGeneration:1
                 deviceRegistryId:0x1234
                  pixelFormatMask:
                      ChromaspaceSourceExchangePixelFormatRGBA16Float |
                      ChromaspaceSourceExchangePixelFormatRGBA32Float
                     maximumWidth:8192
                    maximumHeight:8192
              maximumSurfaceBytes:512ull * 1024ull * 1024ull
             maximumRetainedBytes:1024ull * 1024ull * 1024ull
                     maximumSlots:ChromaspaceSourceExchangeMaximumSlots
             supportsSharedEvents:YES];
    require([viewer validate:&error] && error == nil,
            @"valid viewer registration");
    viewer = secureRoundTrip(viewer,
                             [ChromaspaceSourceExchangeViewerRegistration class]);
    require([viewer validate:&error], @"viewer registration round trip");

    ChromaspaceSourceExchangeViewerRoute* route =
        [[ChromaspaceSourceExchangeViewerRoute alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                 viewerGeneration:1
                    routeRevision:1
                         senderId:@"test-sender"];
    require([route validate:&error], @"valid viewer route");
    route = secureRoundTrip(
        route, [ChromaspaceSourceExchangeViewerRoute class]);
    require([route validate:&error], @"viewer route round trip");

    NSMutableData* bootstrapToken = [NSMutableData
        dataWithLength:ChromaspaceSourceExchangeBootstrapTokenBytes];
    memset(bootstrapToken.mutableBytes, 0x5a, bootstrapToken.length);
    NSXPCListener* bootstrapListener = [NSXPCListener anonymousListener];
    ChromaspaceSourceExchangeRelayBootstrapRegistration* bootstrap =
        [[ChromaspaceSourceExchangeRelayBootstrapRegistration alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                   bootstrapToken:bootstrapToken
            producerRelayEndpoint:bootstrapListener.endpoint];
    error = nil;
    require([bootstrap validate:&error] && error == nil,
            @"valid relay bootstrap registration");
    // Listener endpoints contain Mach rights. They are intentionally not
    // passed through secureRoundTrip; production transfers them through
    // NSXPCConnection/NSXPCCoder only.

    ChromaspaceSourceExchangeProducerJoinRequest* join =
        [[ChromaspaceSourceExchangeProducerJoinRequest alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                         senderId:@"test-sender"
                 senderGeneration:2
                 deviceRegistryId:0x1234];
    require([join validate:&error], @"valid producer join");
    join = secureRoundTrip(
        join, [ChromaspaceSourceExchangeProducerJoinRequest class]);
    require([join validate:&error], @"producer join round trip");

    ChromaspaceSourceExchangeProducerLease* lease =
        [[ChromaspaceSourceExchangeProducerLease alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                 viewerGeneration:1
                         senderId:@"test-sender"
                 senderGeneration:2
                 deviceRegistryId:0x1234
                  pixelFormatMask:
                      ChromaspaceSourceExchangePixelFormatRGBA16Float |
                      ChromaspaceSourceExchangePixelFormatRGBA32Float
                     maximumWidth:8192
                    maximumHeight:8192
              maximumSurfaceBytes:512ull * 1024ull * 1024ull
             maximumRetainedBytes:1024ull * 1024ull * 1024ull
                     maximumSlots:ChromaspaceSourceExchangeMaximumSlots
             supportsSharedEvents:YES];
    require([lease validate:&error], @"valid producer lease");
    lease = secureRoundTrip(
        lease, [ChromaspaceSourceExchangeProducerLease class]);
    require([lease validate:&error], @"producer lease round trip");

    ChromaspaceSourceExchangeProducerRegistration* producer =
        [[ChromaspaceSourceExchangeProducerRegistration alloc]
            initWithSessionCapability:capability
                              senderId:@"test-sender"
                      senderGeneration:2
                      deviceRegistryId:0x1234];
    require([producer validate:&error], @"valid producer registration");
    producer = secureRoundTrip(
        producer, [ChromaspaceSourceExchangeProducerRegistration class]);
    require([producer validate:&error], @"producer registration round trip");

    ChromaspaceSourceExchangeMetadata* metadata =
        [[ChromaspaceSourceExchangeMetadata alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                         senderId:@"test-sender"
                 senderGeneration:2
                         sequence:3
                        slotIndex:1
                   slotGeneration:4
                       readyValue:5
                 deviceRegistryId:0x1234
                            width:1920
                           height:1080
                      pixelFormat:0
                      bytesPerRow:1920ull * 8ull
                         byteSize:1920ull * 1080ull * 8ull
                      contentHash:0xabcdef
                         sourceX:0
                         sourceY:0
                     sourceWidth:1920
                    sourceHeight:1080
                        sampledX:0
                        sampledY:0
                    sampledWidth:1920
                   sampledHeight:1080
                    authoritative:YES
                         coverage:@"full"
             identityStripPresent:NO
                     identityCube:NO
                     identityRamp:NO
               identityResolution:0
               identityBandHeight:0
                   identityCubeY1:0
                   identityCubeY2:0
                   identityRampY1:0
                   identityRampY2:0
                   colorPrimaries:@"source"
                 transferFunction:@"source"];
    require([metadata validate:&error], @"valid metadata");
    metadata =
        secureRoundTrip(metadata, [ChromaspaceSourceExchangeMetadata class]);
    require([metadata validate:&error], @"metadata round trip");

    ChromaspaceSourceExchangeReleaseEvent* releaseEvent =
        [[ChromaspaceSourceExchangeReleaseEvent alloc]
            initWithOrdinal:1
                   senderId:@"test-sender"
           senderGeneration:2
                   sequence:3
                  slotIndex:1
             slotGeneration:4];
    require([releaseEvent validate:&error], @"valid release event");
    releaseEvent = secureRoundTrip(
        releaseEvent, [ChromaspaceSourceExchangeReleaseEvent class]);
    require([releaseEvent validate:&error], @"release event round trip");

    ChromaspaceSourceExchangeReleaseBatch* releaseBatch =
        [[ChromaspaceSourceExchangeReleaseBatch alloc]
            initWithSessionCapability:capability
                              senderId:@"test-sender"
                      senderGeneration:2
                       throughOrdinal:1
                                events:@[ releaseEvent ]];
    require([releaseBatch validate:&error], @"valid release batch");
    releaseBatch = secureRoundTrip(
        releaseBatch, [ChromaspaceSourceExchangeReleaseBatch class]);
    require([releaseBatch validate:&error], @"release batch round trip");

    ChromaspaceSourceExchangeAcknowledgement* acknowledgement =
        [[ChromaspaceSourceExchangeAcknowledgement alloc]
            initWithSessionCapability:capability
                              senderId:@"test-sender"
                      senderGeneration:2
                              sequence:3
                             slotIndex:1
                        slotGeneration:4
                                status:ChromaspaceSourceExchangeStatusRetired];
    require([acknowledgement validate:&error], @"valid acknowledgement");
    acknowledgement = secureRoundTrip(
        acknowledgement, [ChromaspaceSourceExchangeAcknowledgement class]);
    require([acknowledgement validate:&error],
            @"acknowledgement round trip");

    ChromaspaceSourceExchangeViewerRegistration* invalidViewer =
        [[ChromaspaceSourceExchangeViewerRegistration alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor + 1
                    protocolMinor:0
                sessionCapability:capability
                 viewerGeneration:1
                 deviceRegistryId:0x1234
                  pixelFormatMask:
                      ChromaspaceSourceExchangePixelFormatRGBA16Float
                     maximumWidth:8192
                    maximumHeight:8192
              maximumSurfaceBytes:512ull * 1024ull * 1024ull
             maximumRetainedBytes:1024ull * 1024ull * 1024ull
                     maximumSlots:3
             supportsSharedEvents:YES];
    error = nil;
    require(![invalidViewer validate:&error] &&
                error.code == ChromaspaceSourceExchangeErrorInvalidProtocol,
            @"protocol rejection");

    ChromaspaceSourceExchangeViewerRoute* invalidRoute =
        [[ChromaspaceSourceExchangeViewerRoute alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                 viewerGeneration:1
                    routeRevision:0
                         senderId:@"test-sender"];
    error = nil;
    require(![invalidRoute validate:&error] &&
                error.code ==
                    ChromaspaceSourceExchangeErrorInvalidIdentity,
            @"viewer route revision rejection");

    ChromaspaceSourceExchangeRelayBootstrapRegistration* invalidBootstrap =
        [[ChromaspaceSourceExchangeRelayBootstrapRegistration alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                   bootstrapToken:[NSMutableData dataWithLength:8]
            producerRelayEndpoint:bootstrapListener.endpoint];
    error = nil;
    require(![invalidBootstrap validate:&error] &&
                error.code ==
                    ChromaspaceSourceExchangeErrorInvalidCapability,
            @"bootstrap token rejection");

    ChromaspaceSourceExchangeProducerJoinRequest* invalidJoin =
        [[ChromaspaceSourceExchangeProducerJoinRequest alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                         senderId:@""
                 senderGeneration:0
                 deviceRegistryId:0];
    error = nil;
    require(![invalidJoin validate:&error] &&
                error.code ==
                    ChromaspaceSourceExchangeErrorInvalidIdentity,
            @"producer join identity rejection");

    ChromaspaceSourceExchangeProducerLease* invalidLease =
        [[ChromaspaceSourceExchangeProducerLease alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                sessionCapability:capability
                 viewerGeneration:1
                         senderId:@"test-sender"
                 senderGeneration:2
                 deviceRegistryId:0x1234
                  pixelFormatMask:
                      ChromaspaceSourceExchangePixelFormatRGBA16Float
                     maximumWidth:8192
                    maximumHeight:8192
              maximumSurfaceBytes:512ull * 1024ull * 1024ull
             maximumRetainedBytes:1024ull * 1024ull * 1024ull
                     maximumSlots:ChromaspaceSourceExchangeMaximumSlots + 1
             supportsSharedEvents:YES];
    error = nil;
    require(![invalidLease validate:&error] &&
                error.code == ChromaspaceSourceExchangeErrorResourceLimit,
            @"producer lease limit rejection");

    ChromaspaceSourceExchangeReleaseBatch* invalidReleaseBatch =
        [[ChromaspaceSourceExchangeReleaseBatch alloc]
            initWithSessionCapability:capability
                              senderId:@"test-sender"
                      senderGeneration:2
                       throughOrdinal:2
                                events:@[ releaseEvent ]];
    error = nil;
    require(![invalidReleaseBatch validate:&error] &&
                error.code ==
                    ChromaspaceSourceExchangeErrorInvalidGeneration,
            @"release batch cursor rejection");

    ChromaspaceSourceExchangeAcknowledgement* invalidAck =
        [[ChromaspaceSourceExchangeAcknowledgement alloc]
            initWithSessionCapability:capability
                              senderId:@"test-sender"
                      senderGeneration:2
                              sequence:3
                             slotIndex:1
                        slotGeneration:4
                                status:ChromaspaceSourceExchangeStatusAccepted];
    error = nil;
    require(![invalidAck validate:&error],
            @"invalid acknowledgement state");
  }
  return 0;
}
