#import "ChromaspaceSourceExchangeV2.h"

#if defined(__APPLE__)

NSString* const ChromaspaceSourceExchangeErrorDomain =
    @"com.chromaspace.source-exchange-v2";
NSString* const ChromaspaceSourceExchangeMachServiceName =
    @"com.chromaspace.SourceExchangeBroker";
NSString* const ChromaspaceSourceExchangeBootstrapMachServiceName =
    @"com.chromaspace.SourceExchangeBootstrap";

namespace {

NSError* exchangeError(ChromaspaceSourceExchangeErrorCode code,
                       NSString* description) {
  return [NSError errorWithDomain:ChromaspaceSourceExchangeErrorDomain
                            code:code
                        userInfo:@{
                          NSLocalizedDescriptionKey :
                              description ?: @"Invalid SourceExchangeV2 object."
                        }];
}

BOOL validCapability(NSData* capability) {
  return capability != nil &&
         capability.length == ChromaspaceSourceExchangeCapabilityBytes;
}

BOOL validIdentity(NSString* value) {
  return value != nil && value.length > 0 && value.length <= 256;
}

BOOL validSemanticIdentifier(NSString* value) {
  NSData* ascii =
      [value dataUsingEncoding:NSASCIIStringEncoding
          allowLossyConversion:NO];
  if (value == nil || value.length == 0 || ascii == nil ||
      ascii.length == 0 ||
      ascii.length >
          ChromaspaceSourceExchangeMaximumSemanticIdentifierBytes) {
    return NO;
  }
  const unsigned char* bytes =
      static_cast<const unsigned char*>(ascii.bytes);
  for (NSUInteger index = 0; index < ascii.length; ++index) {
    const unsigned char c = bytes[index];
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          (c >= '0' && c <= '9') || c == '-' || c == '_' ||
          c == '.' || c == '+')) {
      return NO;
    }
  }
  return YES;
}

BOOL semanticIntervalContained(int32_t outerStart,
                               uint32_t outerSize,
                               int32_t innerStart,
                               uint32_t innerSize) {
  if (outerSize == 0 || innerSize == 0) return NO;
  const int64_t outerBegin = outerStart;
  const int64_t outerEnd = outerBegin + static_cast<int64_t>(outerSize);
  const int64_t innerBegin = innerStart;
  const int64_t innerEnd = innerBegin + static_cast<int64_t>(innerSize);
  return innerBegin >= outerBegin && innerEnd <= outerEnd;
}

BOOL validIdentityRange(BOOL enabled,
                        int32_t y1,
                        int32_t y2,
                        int32_t sourceY,
                        uint32_t sourceHeight,
                        uint32_t bandHeight) {
  if (!enabled) return y1 == 0 && y2 == 0;
  const int64_t height = static_cast<int64_t>(y2) - y1;
  return height == static_cast<int64_t>(bandHeight) &&
         semanticIntervalContained(
             sourceY, sourceHeight, y1, static_cast<uint32_t>(height));
}

}  // namespace

@implementation ChromaspaceSourceExchangeViewerRegistration

+ (BOOL)supportsSecureCoding {
  return YES;
}

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
                 supportsSharedEvents:(BOOL)supportsSharedEvents {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _sessionCapability = [sessionCapability copy];
    _viewerGeneration = viewerGeneration;
    _deviceRegistryId = deviceRegistryId;
    _pixelFormatMask = pixelFormatMask;
    _maximumWidth = maximumWidth;
    _maximumHeight = maximumHeight;
    _maximumSurfaceBytes = maximumSurfaceBytes;
    _maximumRetainedBytes = maximumRetainedBytes;
    _maximumSlots = maximumSlots;
    _supportsSharedEvents = supportsSharedEvents;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
          sessionCapability:[coder decodeObjectOfClass:[NSData class]
                                                forKey:@"sessionCapability"]
           viewerGeneration:(uint64_t)[coder decodeInt64ForKey:@"viewerGeneration"]
           deviceRegistryId:(uint64_t)[coder decodeInt64ForKey:@"deviceRegistryId"]
            pixelFormatMask:(uint32_t)[coder decodeInt64ForKey:@"pixelFormatMask"]
               maximumWidth:(uint32_t)[coder decodeInt64ForKey:@"maximumWidth"]
              maximumHeight:(uint32_t)[coder decodeInt64ForKey:@"maximumHeight"]
        maximumSurfaceBytes:(uint64_t)[coder decodeInt64ForKey:@"maximumSurfaceBytes"]
       maximumRetainedBytes:(uint64_t)[coder decodeInt64ForKey:@"maximumRetainedBytes"]
               maximumSlots:(uint32_t)[coder decodeInt64ForKey:@"maximumSlots"]
       supportsSharedEvents:[coder decodeBoolForKey:@"supportsSharedEvents"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeInt64:(int64_t)_viewerGeneration forKey:@"viewerGeneration"];
  [coder encodeInt64:(int64_t)_deviceRegistryId forKey:@"deviceRegistryId"];
  [coder encodeInt64:_pixelFormatMask forKey:@"pixelFormatMask"];
  [coder encodeInt64:_maximumWidth forKey:@"maximumWidth"];
  [coder encodeInt64:_maximumHeight forKey:@"maximumHeight"];
  [coder encodeInt64:(int64_t)_maximumSurfaceBytes forKey:@"maximumSurfaceBytes"];
  [coder encodeInt64:(int64_t)_maximumRetainedBytes forKey:@"maximumRetainedBytes"];
  [coder encodeInt64:_maximumSlots forKey:@"maximumSlots"];
  [coder encodeBool:_supportsSharedEvents forKey:@"supportsSharedEvents"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidProtocol,
                             @"Unsupported viewer protocol version.");
    }
    return NO;
  }
  if (!validCapability(_sessionCapability)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidCapability,
                             @"Viewer capability must contain exactly 32 bytes.");
    }
    return NO;
  }
  const uint32_t knownFormats =
      ChromaspaceSourceExchangePixelFormatRGBA16Float |
      ChromaspaceSourceExchangePixelFormatRGBA32Float;
  const BOOL limitsValid =
      _viewerGeneration != 0 && _deviceRegistryId != 0 &&
      (_pixelFormatMask & knownFormats) != 0 &&
      (_pixelFormatMask & ~knownFormats) == 0 &&
      _maximumWidth > 0 &&
      _maximumWidth <= ChromaspaceSourceExchangeMaximumDimension &&
      _maximumHeight > 0 &&
      _maximumHeight <= ChromaspaceSourceExchangeMaximumDimension &&
      _maximumSurfaceBytes > 0 &&
      _maximumSurfaceBytes <= ChromaspaceSourceExchangeMaximumSurfaceBytes &&
      _maximumRetainedBytes >= _maximumSurfaceBytes &&
      _maximumRetainedBytes <= ChromaspaceSourceExchangeMaximumRetainedBytes &&
      _maximumSlots > 0 &&
      _maximumSlots <= ChromaspaceSourceExchangeMaximumSlots &&
      _supportsSharedEvents;
  if (!limitsValid) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorResourceLimit,
                             @"Viewer device, format, synchronization, or limits are invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeViewerRoute

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                    sessionCapability:(NSData*)sessionCapability
                     viewerGeneration:(uint64_t)viewerGeneration
                        routeRevision:(uint64_t)routeRevision
                             senderId:(NSString*)senderId {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _sessionCapability = [sessionCapability copy];
    _viewerGeneration = viewerGeneration;
    _routeRevision = routeRevision;
    _senderId = [senderId copy];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
          sessionCapability:[coder decodeObjectOfClass:[NSData class]
                                                forKey:@"sessionCapability"]
           viewerGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"viewerGeneration"]
              routeRevision:(uint64_t)[coder
                                decodeInt64ForKey:@"routeRevision"]
                   senderId:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"senderId"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeInt64:(int64_t)_viewerGeneration forKey:@"viewerGeneration"];
  [coder encodeInt64:(int64_t)_routeRevision forKey:@"routeRevision"];
  [coder encodeObject:_senderId forKey:@"senderId"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidProtocol,
                             @"Unsupported viewer route protocol version.");
    }
    return NO;
  }
  if (!validCapability(_sessionCapability) || _viewerGeneration == 0 ||
      _routeRevision == 0 || !validIdentity(_senderId)) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidIdentity,
          @"Viewer route capability, revision, generation, or sender is invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeRelayBootstrapRegistration

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                       bootstrapToken:(NSData*)bootstrapToken
                producerRelayEndpoint:
                    (NSXPCListenerEndpoint*)producerRelayEndpoint {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _bootstrapToken = [bootstrapToken copy];
    _producerRelayEndpoint = producerRelayEndpoint;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
             bootstrapToken:[coder decodeObjectOfClass:[NSData class]
                                                forKey:@"bootstrapToken"]
      producerRelayEndpoint:
          [coder decodeObjectOfClass:[NSXPCListenerEndpoint class]
                              forKey:@"producerRelayEndpoint"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_bootstrapToken forKey:@"bootstrapToken"];
  [coder encodeObject:_producerRelayEndpoint
               forKey:@"producerRelayEndpoint"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidProtocol,
          @"Unsupported producer-relay bootstrap protocol version.");
    }
    return NO;
  }
  if (_bootstrapToken.length !=
          ChromaspaceSourceExchangeBootstrapTokenBytes ||
      _producerRelayEndpoint == nil) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidCapability,
          @"Producer-relay bootstrap requires one 256-bit token and XPC endpoint.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeProducerJoinRequest

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithProtocolMajor:(uint32_t)protocolMajor
                        protocolMinor:(uint32_t)protocolMinor
                             senderId:(NSString*)senderId
                     senderGeneration:(uint64_t)senderGeneration
                     deviceRegistryId:(uint64_t)deviceRegistryId {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _deviceRegistryId = deviceRegistryId;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
                   senderId:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"senderId"]
           senderGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"senderGeneration"]
           deviceRegistryId:(uint64_t)[coder
                                decodeInt64ForKey:@"deviceRegistryId"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_deviceRegistryId forKey:@"deviceRegistryId"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidProtocol,
                             @"Unsupported producer join protocol version.");
    }
    return NO;
  }
  if (!validIdentity(_senderId) || _senderGeneration == 0 ||
      _deviceRegistryId == 0) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidIdentity,
          @"Producer join identity, generation, or Metal device is invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeProducerLease

+ (BOOL)supportsSecureCoding {
  return YES;
}

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
                 supportsSharedEvents:(BOOL)supportsSharedEvents {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _sessionCapability = [sessionCapability copy];
    _viewerGeneration = viewerGeneration;
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _deviceRegistryId = deviceRegistryId;
    _pixelFormatMask = pixelFormatMask;
    _maximumWidth = maximumWidth;
    _maximumHeight = maximumHeight;
    _maximumSurfaceBytes = maximumSurfaceBytes;
    _maximumRetainedBytes = maximumRetainedBytes;
    _maximumSlots = maximumSlots;
    _supportsSharedEvents = supportsSharedEvents;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
          sessionCapability:[coder decodeObjectOfClass:[NSData class]
                                                forKey:@"sessionCapability"]
           viewerGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"viewerGeneration"]
                   senderId:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"senderId"]
           senderGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"senderGeneration"]
           deviceRegistryId:(uint64_t)[coder
                                decodeInt64ForKey:@"deviceRegistryId"]
            pixelFormatMask:(uint32_t)[coder
                                decodeInt64ForKey:@"pixelFormatMask"]
               maximumWidth:(uint32_t)[coder
                                decodeInt64ForKey:@"maximumWidth"]
              maximumHeight:(uint32_t)[coder
                                decodeInt64ForKey:@"maximumHeight"]
        maximumSurfaceBytes:(uint64_t)[coder
                                decodeInt64ForKey:@"maximumSurfaceBytes"]
       maximumRetainedBytes:(uint64_t)[coder
                                decodeInt64ForKey:@"maximumRetainedBytes"]
               maximumSlots:(uint32_t)[coder
                                decodeInt64ForKey:@"maximumSlots"]
       supportsSharedEvents:[coder decodeBoolForKey:@"supportsSharedEvents"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeInt64:(int64_t)_viewerGeneration forKey:@"viewerGeneration"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_deviceRegistryId forKey:@"deviceRegistryId"];
  [coder encodeInt64:_pixelFormatMask forKey:@"pixelFormatMask"];
  [coder encodeInt64:_maximumWidth forKey:@"maximumWidth"];
  [coder encodeInt64:_maximumHeight forKey:@"maximumHeight"];
  [coder encodeInt64:(int64_t)_maximumSurfaceBytes forKey:@"maximumSurfaceBytes"];
  [coder encodeInt64:(int64_t)_maximumRetainedBytes
              forKey:@"maximumRetainedBytes"];
  [coder encodeInt64:_maximumSlots forKey:@"maximumSlots"];
  [coder encodeBool:_supportsSharedEvents forKey:@"supportsSharedEvents"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidProtocol,
                             @"Unsupported producer lease protocol version.");
    }
    return NO;
  }
  const uint32_t knownFormats =
      ChromaspaceSourceExchangePixelFormatRGBA16Float |
      ChromaspaceSourceExchangePixelFormatRGBA32Float;
  const BOOL limitsValid =
      validCapability(_sessionCapability) && _viewerGeneration != 0 &&
      validIdentity(_senderId) && _senderGeneration != 0 &&
      _deviceRegistryId != 0 && (_pixelFormatMask & knownFormats) != 0 &&
      (_pixelFormatMask & ~knownFormats) == 0 && _maximumWidth > 0 &&
      _maximumWidth <= ChromaspaceSourceExchangeMaximumDimension &&
      _maximumHeight > 0 &&
      _maximumHeight <= ChromaspaceSourceExchangeMaximumDimension &&
      _maximumSurfaceBytes > 0 &&
      _maximumSurfaceBytes <= ChromaspaceSourceExchangeMaximumSurfaceBytes &&
      _maximumRetainedBytes >= _maximumSurfaceBytes &&
      _maximumRetainedBytes <= ChromaspaceSourceExchangeMaximumRetainedBytes &&
      _maximumSlots > 0 &&
      _maximumSlots <= ChromaspaceSourceExchangeMaximumSlots &&
      _supportsSharedEvents;
  if (!limitsValid) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorResourceLimit,
          @"Producer lease identity, device, or negotiated limits are invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeProducerRegistration

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                          deviceRegistryId:(uint64_t)deviceRegistryId {
  self = [super init];
  if (self) {
    _sessionCapability = [sessionCapability copy];
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _deviceRegistryId = deviceRegistryId;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithSessionCapability:[coder decodeObjectOfClass:[NSData class]
                                                    forKey:@"sessionCapability"]
                        senderId:[coder decodeObjectOfClass:[NSString class]
                                                    forKey:@"senderId"]
                senderGeneration:(uint64_t)[coder decodeInt64ForKey:@"senderGeneration"]
                deviceRegistryId:(uint64_t)[coder decodeInt64ForKey:@"deviceRegistryId"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_deviceRegistryId forKey:@"deviceRegistryId"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (!validCapability(_sessionCapability)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidCapability,
                             @"Producer capability must contain exactly 32 bytes.");
    }
    return NO;
  }
  if (!validIdentity(_senderId) || _senderGeneration == 0 ||
      _deviceRegistryId == 0) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidIdentity,
                             @"Producer identity, generation, or Metal device is invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeMetadata

+ (BOOL)supportsSecureCoding {
  return YES;
}

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
                     transferFunction:(NSString*)transferFunction {
  self = [super init];
  if (self) {
    _protocolMajor = protocolMajor;
    _protocolMinor = protocolMinor;
    _sessionCapability = [sessionCapability copy];
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _sequence = sequence;
    _slotIndex = slotIndex;
    _slotGeneration = slotGeneration;
    _readyValue = readyValue;
    _deviceRegistryId = deviceRegistryId;
    _width = width;
    _height = height;
    _pixelFormat = pixelFormat;
    _bytesPerRow = bytesPerRow;
    _byteSize = byteSize;
    _contentHash = contentHash;
    _sourceX = sourceX;
    _sourceY = sourceY;
    _sourceWidth = sourceWidth;
    _sourceHeight = sourceHeight;
    _sampledX = sampledX;
    _sampledY = sampledY;
    _sampledWidth = sampledWidth;
    _sampledHeight = sampledHeight;
    _authoritative = authoritative;
    _coverage = [coverage copy];
    _identityStripPresent = identityStripPresent;
    _identityCube = identityCube;
    _identityRamp = identityRamp;
    _identityResolution = identityResolution;
    _identityBandHeight = identityBandHeight;
    _identityCubeY1 = identityCubeY1;
    _identityCubeY2 = identityCubeY2;
    _identityRampY1 = identityRampY1;
    _identityRampY2 = identityRampY2;
    _colorPrimaries = [colorPrimaries copy];
    _transferFunction = [transferFunction copy];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithProtocolMajor:(uint32_t)[coder decodeInt64ForKey:@"protocolMajor"]
              protocolMinor:(uint32_t)[coder decodeInt64ForKey:@"protocolMinor"]
          sessionCapability:[coder decodeObjectOfClass:[NSData class]
                                                forKey:@"sessionCapability"]
                   senderId:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"senderId"]
           senderGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"senderGeneration"]
                   sequence:(uint64_t)[coder decodeInt64ForKey:@"sequence"]
                  slotIndex:(uint32_t)[coder decodeInt64ForKey:@"slotIndex"]
             slotGeneration:(uint64_t)[coder
                                decodeInt64ForKey:@"slotGeneration"]
                 readyValue:(uint64_t)[coder decodeInt64ForKey:@"readyValue"]
           deviceRegistryId:(uint64_t)[coder
                                decodeInt64ForKey:@"deviceRegistryId"]
                      width:(uint32_t)[coder decodeInt64ForKey:@"width"]
                     height:(uint32_t)[coder decodeInt64ForKey:@"height"]
                pixelFormat:(uint32_t)[coder decodeInt64ForKey:@"pixelFormat"]
                bytesPerRow:(uint64_t)[coder decodeInt64ForKey:@"bytesPerRow"]
                   byteSize:(uint64_t)[coder decodeInt64ForKey:@"byteSize"]
                contentHash:(uint64_t)[coder decodeInt64ForKey:@"contentHash"]
                   sourceX:(int32_t)[coder decodeInt64ForKey:@"sourceX"]
                   sourceY:(int32_t)[coder decodeInt64ForKey:@"sourceY"]
               sourceWidth:(uint32_t)[coder decodeInt64ForKey:@"sourceWidth"]
              sourceHeight:(uint32_t)[coder decodeInt64ForKey:@"sourceHeight"]
                  sampledX:(int32_t)[coder decodeInt64ForKey:@"sampledX"]
                  sampledY:(int32_t)[coder decodeInt64ForKey:@"sampledY"]
              sampledWidth:(uint32_t)[coder decodeInt64ForKey:@"sampledWidth"]
             sampledHeight:(uint32_t)[coder decodeInt64ForKey:@"sampledHeight"]
              authoritative:[coder decodeBoolForKey:@"authoritative"]
                   coverage:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"coverage"]
       identityStripPresent:[coder decodeBoolForKey:@"identityStripPresent"]
               identityCube:[coder decodeBoolForKey:@"identityCube"]
               identityRamp:[coder decodeBoolForKey:@"identityRamp"]
         identityResolution:(uint32_t)[coder
                                decodeInt64ForKey:@"identityResolution"]
         identityBandHeight:(uint32_t)[coder
                                decodeInt64ForKey:@"identityBandHeight"]
             identityCubeY1:(int32_t)[coder decodeInt64ForKey:@"identityCubeY1"]
             identityCubeY2:(int32_t)[coder decodeInt64ForKey:@"identityCubeY2"]
             identityRampY1:(int32_t)[coder decodeInt64ForKey:@"identityRampY1"]
             identityRampY2:(int32_t)[coder decodeInt64ForKey:@"identityRampY2"]
             colorPrimaries:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"colorPrimaries"]
           transferFunction:[coder decodeObjectOfClass:[NSString class]
                                               forKey:@"transferFunction"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:_protocolMajor forKey:@"protocolMajor"];
  [coder encodeInt64:_protocolMinor forKey:@"protocolMinor"];
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_sequence forKey:@"sequence"];
  [coder encodeInt64:_slotIndex forKey:@"slotIndex"];
  [coder encodeInt64:(int64_t)_slotGeneration forKey:@"slotGeneration"];
  [coder encodeInt64:(int64_t)_readyValue forKey:@"readyValue"];
  [coder encodeInt64:(int64_t)_deviceRegistryId forKey:@"deviceRegistryId"];
  [coder encodeInt64:_width forKey:@"width"];
  [coder encodeInt64:_height forKey:@"height"];
  [coder encodeInt64:_pixelFormat forKey:@"pixelFormat"];
  [coder encodeInt64:(int64_t)_bytesPerRow forKey:@"bytesPerRow"];
  [coder encodeInt64:(int64_t)_byteSize forKey:@"byteSize"];
  [coder encodeInt64:(int64_t)_contentHash forKey:@"contentHash"];
  [coder encodeInt64:_sourceX forKey:@"sourceX"];
  [coder encodeInt64:_sourceY forKey:@"sourceY"];
  [coder encodeInt64:_sourceWidth forKey:@"sourceWidth"];
  [coder encodeInt64:_sourceHeight forKey:@"sourceHeight"];
  [coder encodeInt64:_sampledX forKey:@"sampledX"];
  [coder encodeInt64:_sampledY forKey:@"sampledY"];
  [coder encodeInt64:_sampledWidth forKey:@"sampledWidth"];
  [coder encodeInt64:_sampledHeight forKey:@"sampledHeight"];
  [coder encodeBool:_authoritative forKey:@"authoritative"];
  [coder encodeObject:_coverage forKey:@"coverage"];
  [coder encodeBool:_identityStripPresent forKey:@"identityStripPresent"];
  [coder encodeBool:_identityCube forKey:@"identityCube"];
  [coder encodeBool:_identityRamp forKey:@"identityRamp"];
  [coder encodeInt64:_identityResolution forKey:@"identityResolution"];
  [coder encodeInt64:_identityBandHeight forKey:@"identityBandHeight"];
  [coder encodeInt64:_identityCubeY1 forKey:@"identityCubeY1"];
  [coder encodeInt64:_identityCubeY2 forKey:@"identityCubeY2"];
  [coder encodeInt64:_identityRampY1 forKey:@"identityRampY1"];
  [coder encodeInt64:_identityRampY2 forKey:@"identityRampY2"];
  [coder encodeObject:_colorPrimaries forKey:@"colorPrimaries"];
  [coder encodeObject:_transferFunction forKey:@"transferFunction"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_protocolMajor != ChromaspaceSourceExchangeProtocolMajor ||
      _protocolMinor > ChromaspaceSourceExchangeProtocolMinor) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidProtocol,
                             @"Unsupported SourceExchangeV2 protocol version.");
    }
    return NO;
  }
  if (!validCapability(_sessionCapability)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidCapability,
                             @"Session capability must contain exactly 32 bytes.");
    }
    return NO;
  }
  if (!validIdentity(_senderId)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidIdentity,
                             @"Sender identity is missing or too long.");
    }
    return NO;
  }
  if (_senderGeneration == 0 || _sequence == 0 || _slotGeneration == 0) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidGeneration,
                             @"Sender, sequence, and slot generations must be nonzero.");
    }
    return NO;
  }
  if (_slotIndex >= ChromaspaceSourceExchangeMaximumSlots) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidSlot,
                             @"Source exchange slot is outside the bounded ring.");
    }
    return NO;
  }
  if (_readyValue == 0) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidSynchronization,
          @"Producer-ready shared-event value must be nonzero.");
    }
    return NO;
  }
  const uint64_t bytesPerElement = _pixelFormat == 0 ? 8u : 16u;
  const BOOL formatValid = _pixelFormat == 0 || _pixelFormat == 1;
  const BOOL dimensionsValid =
      _width > 0 && _height > 0 &&
      _width <= ChromaspaceSourceExchangeMaximumDimension &&
      _height <= ChromaspaceSourceExchangeMaximumDimension;
  const uint64_t minimumRowBytes =
      dimensionsValid && formatValid ? (uint64_t)_width * bytesPerElement : 0;
  const BOOL sizeValid =
      _bytesPerRow >= minimumRowBytes &&
      _byteSize >= _bytesPerRow * (uint64_t)_height &&
      _byteSize <= ChromaspaceSourceExchangeMaximumSurfaceBytes;
  if (!formatValid || !dimensionsValid || !sizeValid) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidSurface,
                             @"Source texture dimensions, format, or byte bounds are invalid.");
    }
    return NO;
  }
  if (_deviceRegistryId == 0 || _contentHash == 0) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidSurface,
                             @"Device identity and content hash must be nonzero.");
    }
    return NO;
  }
  const BOOL fullCoverage = [_coverage isEqualToString:@"full"];
  const BOOL partialCoverage = [_coverage isEqualToString:@"partial"];
  const BOOL sourceGeometryValid =
      _sourceWidth > 0 &&
      _sourceWidth <= ChromaspaceSourceExchangeMaximumDimension &&
      _sourceHeight > 0 &&
      _sourceHeight <= ChromaspaceSourceExchangeMaximumDimension &&
      semanticIntervalContained(
          _sourceX, _sourceWidth, _sampledX, _sampledWidth) &&
      semanticIntervalContained(
          _sourceY, _sourceHeight, _sampledY, _sampledHeight);
  const BOOL fullGeometry =
      _sampledX == _sourceX && _sampledY == _sourceY &&
      _sampledWidth == _sourceWidth &&
      _sampledHeight == _sourceHeight;
  BOOL identityValid = NO;
  if (!_identityStripPresent) {
    identityValid =
        !_identityCube && !_identityRamp &&
        _identityResolution == 0 && _identityBandHeight == 0 &&
        _identityCubeY1 == 0 && _identityCubeY2 == 0 &&
        _identityRampY1 == 0 && _identityRampY2 == 0;
  } else {
    identityValid =
        (_identityCube || _identityRamp) &&
        _identityResolution >= 2 &&
        _identityResolution <=
            ChromaspaceSourceExchangeMaximumIdentityResolution &&
        _identityBandHeight > 0 &&
        _identityBandHeight <= _sourceHeight &&
        validIdentityRange(_identityCube,
                           _identityCubeY1,
                           _identityCubeY2,
                           _sourceY,
                           _sourceHeight,
                           _identityBandHeight) &&
        validIdentityRange(_identityRamp,
                           _identityRampY1,
                           _identityRampY2,
                           _sourceY,
                           _sourceHeight,
                           _identityBandHeight);
    if (identityValid && _identityCube && _identityRamp) {
      const BOOL overlap =
          _identityCubeY1 < _identityRampY2 &&
          _identityRampY1 < _identityCubeY2;
      identityValid =
          !overlap && _identityBandHeight <= _sourceHeight / 2;
    }
  }
  if ((!fullCoverage && !partialCoverage) ||
      (fullCoverage && !fullGeometry) ||
      (partialCoverage && _authoritative) ||
      !sourceGeometryValid || !identityValid ||
      !validSemanticIdentifier(_colorPrimaries) ||
      !validSemanticIdentifier(_transferFunction)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidSurface,
                             @"Source semantic metadata is malformed or contradictory.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangePacket

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithMetadata:(ChromaspaceSourceExchangeMetadata*)metadata
                   textureHandle:(MTLSharedTextureHandle*)textureHandle
                     eventHandle:(MTLSharedEventHandle*)eventHandle {
  self = [super init];
  if (self) {
    _metadata = [metadata copy];
    _textureHandle = textureHandle;
    _eventHandle = eventHandle;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithMetadata:[coder
                           decodeObjectOfClass:
                               [ChromaspaceSourceExchangeMetadata class]
                                        forKey:@"metadata"]
         textureHandle:[coder decodeObjectOfClass:[MTLSharedTextureHandle class]
                                           forKey:@"textureHandle"]
           eventHandle:[coder decodeObjectOfClass:[MTLSharedEventHandle class]
                                         forKey:@"eventHandle"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeObject:_metadata forKey:@"metadata"];
  [coder encodeObject:_textureHandle forKey:@"textureHandle"];
  [coder encodeObject:_eventHandle forKey:@"eventHandle"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_metadata == nil || _textureHandle == nil || _eventHandle == nil) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidSurface,
                             @"Source exchange packet is missing a required handle.");
    }
    return NO;
  }
  if (![_metadata validate:error]) return NO;
  id<MTLDevice> handleDevice = _textureHandle.device;
  if (handleDevice == nil ||
      handleDevice.registryID != _metadata.deviceRegistryId) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorDeviceMismatch,
                             @"Shared texture device does not match packet metadata.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeReleaseEvent

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithOrdinal:(uint64_t)ordinal
                       senderId:(NSString*)senderId
               senderGeneration:(uint64_t)senderGeneration
                       sequence:(uint64_t)sequence
                      slotIndex:(uint32_t)slotIndex
                 slotGeneration:(uint64_t)slotGeneration {
  self = [super init];
  if (self) {
    _ordinal = ordinal;
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _sequence = sequence;
    _slotIndex = slotIndex;
    _slotGeneration = slotGeneration;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithOrdinal:(uint64_t)[coder decodeInt64ForKey:@"ordinal"]
             senderId:[coder decodeObjectOfClass:[NSString class]
                                         forKey:@"senderId"]
     senderGeneration:(uint64_t)[coder
                          decodeInt64ForKey:@"senderGeneration"]
             sequence:(uint64_t)[coder decodeInt64ForKey:@"sequence"]
            slotIndex:(uint32_t)[coder decodeInt64ForKey:@"slotIndex"]
       slotGeneration:(uint64_t)[coder
                          decodeInt64ForKey:@"slotGeneration"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeInt64:(int64_t)_ordinal forKey:@"ordinal"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_sequence forKey:@"sequence"];
  [coder encodeInt64:_slotIndex forKey:@"slotIndex"];
  [coder encodeInt64:(int64_t)_slotGeneration forKey:@"slotGeneration"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (_ordinal == 0 || !validIdentity(_senderId) ||
      _senderGeneration == 0 || _sequence == 0 ||
      _slotIndex >= ChromaspaceSourceExchangeMaximumSlots ||
      _slotGeneration == 0) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidGeneration,
          @"Producer release event identity, ordinal, or slot is invalid.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeReleaseBatch

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                           throughOrdinal:(uint64_t)throughOrdinal
                                    events:
                                        (NSArray<ChromaspaceSourceExchangeReleaseEvent*>*)events {
  self = [super init];
  if (self) {
    _sessionCapability = [sessionCapability copy];
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _throughOrdinal = throughOrdinal;
    _events = [events copy];
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  NSSet<Class>* eventClasses = [NSSet
      setWithObjects:[NSArray class],
                     [ChromaspaceSourceExchangeReleaseEvent class],
                     [NSString class],
                     nil];
  return [self
      initWithSessionCapability:
          [coder decodeObjectOfClass:[NSData class]
                              forKey:@"sessionCapability"]
                        senderId:[coder decodeObjectOfClass:[NSString class]
                                                   forKey:@"senderId"]
                senderGeneration:(uint64_t)[coder
                                     decodeInt64ForKey:@"senderGeneration"]
                 throughOrdinal:(uint64_t)[coder
                                    decodeInt64ForKey:@"throughOrdinal"]
                          events:[coder decodeObjectOfClasses:eventClasses
                                                      forKey:@"events"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_throughOrdinal forKey:@"throughOrdinal"];
  [coder encodeObject:_events forKey:@"events"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (!validCapability(_sessionCapability) || !validIdentity(_senderId) ||
      _senderGeneration == 0 || _throughOrdinal == 0 ||
      _events.count == 0 ||
      _events.count > ChromaspaceSourceExchangeMaximumReleaseEvents) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidIdentity,
          @"Producer release batch identity, cursor, or size is invalid.");
    }
    return NO;
  }
  uint64_t priorOrdinal = 0;
  for (ChromaspaceSourceExchangeReleaseEvent* event in _events) {
    NSError* eventError = nil;
    if (![event validate:&eventError] ||
        ![event.senderId isEqualToString:_senderId] ||
        event.senderGeneration != _senderGeneration ||
        event.ordinal <= priorOrdinal) {
      if (error) {
        *error = eventError ?: exchangeError(
            ChromaspaceSourceExchangeErrorInvalidIdentity,
            @"Producer release batch contains a mismatched event.");
      }
      return NO;
    }
    priorOrdinal = event.ordinal;
  }
  if (priorOrdinal != _throughOrdinal) {
    if (error) {
      *error = exchangeError(
          ChromaspaceSourceExchangeErrorInvalidGeneration,
          @"Producer release batch cursor does not match its last event.");
    }
    return NO;
  }
  return YES;
}

@end

@implementation ChromaspaceSourceExchangeAcknowledgement

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (instancetype)initWithSessionCapability:(NSData*)sessionCapability
                                  senderId:(NSString*)senderId
                          senderGeneration:(uint64_t)senderGeneration
                                  sequence:(uint64_t)sequence
                                 slotIndex:(uint32_t)slotIndex
                            slotGeneration:(uint64_t)slotGeneration
                                    status:(ChromaspaceSourceExchangeStatus)status {
  self = [super init];
  if (self) {
    _sessionCapability = [sessionCapability copy];
    _senderId = [senderId copy];
    _senderGeneration = senderGeneration;
    _sequence = sequence;
    _slotIndex = slotIndex;
    _slotGeneration = slotGeneration;
    _status = status;
  }
  return self;
}

- (instancetype)initWithCoder:(NSCoder*)coder {
  return [self
      initWithSessionCapability:
          [coder decodeObjectOfClass:[NSData class]
                              forKey:@"sessionCapability"]
                        senderId:[coder decodeObjectOfClass:[NSString class]
                                                   forKey:@"senderId"]
                senderGeneration:(uint64_t)[coder
                                     decodeInt64ForKey:@"senderGeneration"]
                        sequence:(uint64_t)[coder decodeInt64ForKey:@"sequence"]
                       slotIndex:(uint32_t)[coder decodeInt64ForKey:@"slotIndex"]
                  slotGeneration:(uint64_t)[coder
                                     decodeInt64ForKey:@"slotGeneration"]
                          status:(ChromaspaceSourceExchangeStatus)[coder
                                     decodeIntegerForKey:@"status"]];
}

- (void)encodeWithCoder:(NSCoder*)coder {
  [coder encodeObject:_sessionCapability forKey:@"sessionCapability"];
  [coder encodeObject:_senderId forKey:@"senderId"];
  [coder encodeInt64:(int64_t)_senderGeneration forKey:@"senderGeneration"];
  [coder encodeInt64:(int64_t)_sequence forKey:@"sequence"];
  [coder encodeInt64:_slotIndex forKey:@"slotIndex"];
  [coder encodeInt64:(int64_t)_slotGeneration forKey:@"slotGeneration"];
  [coder encodeInteger:_status forKey:@"status"];
}

- (id)copyWithZone:(NSZone*)zone {
  (void)zone;
  return self;
}

- (BOOL)validate:(NSError**)error {
  if (!validCapability(_sessionCapability) || !validIdentity(_senderId)) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidIdentity,
                             @"Acknowledgement capability or sender is invalid.");
    }
    return NO;
  }
  if (_senderGeneration == 0 || _sequence == 0 || _slotGeneration == 0 ||
      _slotIndex >= ChromaspaceSourceExchangeMaximumSlots) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorInvalidGeneration,
                             @"Acknowledgement generation or slot is invalid.");
    }
    return NO;
  }
  if (_status != ChromaspaceSourceExchangeStatusAcquired &&
      _status != ChromaspaceSourceExchangeStatusRetired &&
      _status != ChromaspaceSourceExchangeStatusStale) {
    if (error) {
      *error = exchangeError(ChromaspaceSourceExchangeErrorStalePublication,
                             @"Acknowledgement status is not an acknowledgement state.");
    }
    return NO;
  }
  return YES;
}

@end

#endif  // defined(__APPLE__)
