#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Security/Security.h>

#import "ChromaspaceSourceExchangeV2.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>

namespace {

constexpr int64_t kBootstrapDeadlineNanoseconds = 10 * NSEC_PER_SEC;

NSXPCInterface* producerInterface(Protocol* protocol) {
  NSXPCInterface* interface =
      [NSXPCInterface interfaceWithProtocol:protocol];
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
  NSSet<Class>* releaseBatchClasses = [NSSet
      setWithObjects:[ChromaspaceSourceExchangeReleaseBatch class],
                     [ChromaspaceSourceExchangeReleaseEvent class],
                     [NSArray class],
                     [NSData class],
                     [NSString class],
                     nil];
  NSSet<Class>* bootstrapClasses = [NSSet
      setWithObjects:
          [ChromaspaceSourceExchangeRelayBootstrapRegistration class],
          [NSXPCListenerEndpoint class],
          [NSData class],
          nil];

  if (protocol ==
      @protocol(ChromaspaceSourceExchangeProducerBrokerProtocol)) {
    [interface setClasses:bootstrapClasses
              forSelector:@selector(registerProducerRelayBootstrap:withReply:)
            argumentIndex:0
                  ofReply:NO];
  }
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
  [interface setClasses:releaseBatchClasses
            forSelector:@selector(fetchProducerReleasesAfterOrdinal:maximumEvents:withReply:)
          argumentIndex:0
                ofReply:YES];
  return interface;
}

bool writeAll(int fd, const void* bytes, size_t byteCount) {
  const uint8_t* cursor = static_cast<const uint8_t*>(bytes);
  size_t remaining = byteCount;
  while (remaining > 0) {
    ssize_t written = write(fd, cursor, remaining);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    cursor += static_cast<size_t>(written);
    remaining -= static_cast<size_t>(written);
  }
  return true;
}

NSData* createBootstrapToken() {
  NSMutableData* token = [NSMutableData
      dataWithLength:ChromaspaceSourceExchangeBootstrapTokenBytes];
  if (SecRandomCopyBytes(kSecRandomDefault,
                         token.length,
                         static_cast<uint8_t*>(token.mutableBytes)) !=
      errSecSuccess) {
    return nil;
  }
  return token;
}

int bootstrapFileDescriptor(int argc, const char* argv[]) {
  if (argc != 3 || std::strcmp(argv[1], "--bootstrap-fd") != 0) {
    return -1;
  }
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(argv[2], &end, 10);
  if (errno != 0 || end == argv[2] || end == nullptr || *end != '\0' ||
      value < 3 || value > INT_MAX) {
    return -1;
  }
  return static_cast<int>(value);
}

}  // namespace

@interface ChromaspaceSourceExchangeProducerRelayController : NSObject

@property(nonatomic, strong) NSXPCConnection* brokerConnection;
@property(nonatomic, strong) NSXPCConnection* hostConnection;
@property(nonatomic, strong) NSXPCListener* listener;
@property(nonatomic) BOOL acceptedHost;
@property(nonatomic) BOOL deliveredBootstrapToken;
@property(nonatomic) BOOL stopping;
@property(nonatomic) int bootstrapFd;
@property(nonatomic) int exitCode;

- (void)requestStop;
- (void)requestStopWithExitCode:(int)exitCode;
- (BOOL)deliverBootstrapToken:(NSData*)token;
- (BOOL)hasDeliveredBootstrapToken;
- (BOOL)hasAcceptedHost;

@end

@interface ChromaspaceSourceExchangeProducerRelaySession
    : NSObject <ChromaspaceSourceExchangeProducerRelayProtocol>

@property(nonatomic, strong) NSXPCConnection* brokerConnection;
@property(nonatomic, weak)
    ChromaspaceSourceExchangeProducerRelayController* controller;

@end

@implementation ChromaspaceSourceExchangeProducerRelaySession

- (id<ChromaspaceSourceExchangeProducerBrokerProtocol>)brokerProxyWithErrorHandler:
    (void (^)(NSError* error))errorHandler {
  return [self.brokerConnection
      remoteObjectProxyWithErrorHandler:errorHandler];
}

- (void)joinProducer:
            (ChromaspaceSourceExchangeProducerJoinRequest*)request
          withReply:
              (void (^)(ChromaspaceSourceExchangeProducerLease*,
                        ChromaspaceSourceExchangeStatus,
                        NSError*))reply {
  id<ChromaspaceSourceExchangeProducerBrokerProtocol> broker =
      [self brokerProxyWithErrorHandler:^(NSError* error) {
        reply(nil, ChromaspaceSourceExchangeStatusRejected, error);
      }];
  [broker joinProducer:request withReply:reply];
}

- (void)publishPacket:(ChromaspaceSourceExchangePacket*)packet
            withReply:(void (^)(ChromaspaceSourceExchangeStatus,
                                NSError*))reply {
  id<ChromaspaceSourceExchangeProducerBrokerProtocol> broker =
      [self brokerProxyWithErrorHandler:^(NSError* error) {
        reply(ChromaspaceSourceExchangeStatusRejected, error);
      }];
  [broker publishPacket:packet withReply:reply];
}

- (void)fetchProducerReleasesAfterOrdinal:(uint64_t)afterOrdinal
                            maximumEvents:(uint32_t)maximumEvents
                                withReply:
                                    (void (^)(ChromaspaceSourceExchangeReleaseBatch*,
                                              ChromaspaceSourceExchangeStatus,
                                              NSError*))reply {
  id<ChromaspaceSourceExchangeProducerBrokerProtocol> broker =
      [self brokerProxyWithErrorHandler:^(NSError* error) {
        reply(nil, ChromaspaceSourceExchangeStatusRejected, error);
      }];
  [broker fetchProducerReleasesAfterOrdinal:afterOrdinal
                              maximumEvents:maximumEvents
                                  withReply:reply];
}

- (void)acknowledgeProducerReleasesThroughOrdinal:(uint64_t)throughOrdinal
                                         withReply:
                                             (void (^)(ChromaspaceSourceExchangeStatus,
                                                       NSError*))reply {
  id<ChromaspaceSourceExchangeProducerBrokerProtocol> broker =
      [self brokerProxyWithErrorHandler:^(NSError* error) {
        reply(ChromaspaceSourceExchangeStatusRejected, error);
      }];
  [broker acknowledgeProducerReleasesThroughOrdinal:throughOrdinal
                                          withReply:reply];
}

@end

@interface ChromaspaceSourceExchangeProducerRelayListenerDelegate
    : NSObject <NSXPCListenerDelegate>

@property(nonatomic, strong)
    ChromaspaceSourceExchangeProducerRelayController* controller;

@end

@implementation ChromaspaceSourceExchangeProducerRelayController

- (BOOL)hasAcceptedHost {
  @synchronized(self) {
    return self.acceptedHost;
  }
}

- (BOOL)hasDeliveredBootstrapToken {
  @synchronized(self) {
    return self.deliveredBootstrapToken;
  }
}

- (void)requestStop {
  [self requestStopWithExitCode:0];
}

- (BOOL)deliverBootstrapToken:(NSData*)token {
  @synchronized(self) {
    if (self.stopping || self.bootstrapFd < 0 ||
        token.length != ChromaspaceSourceExchangeBootstrapTokenBytes) {
      return NO;
    }
    const int fd = self.bootstrapFd;
    self.bootstrapFd = -1;
    const bool written = writeAll(fd, token.bytes, token.length);
    close(fd);
    self.deliveredBootstrapToken = written;
    return written;
  }
}

- (void)requestStopWithExitCode:(int)exitCode {
  NSXPCConnection* broker = nil;
  NSXPCConnection* host = nil;
  NSXPCListener* listener = nil;
  int bootstrapFd = -1;
  @synchronized(self) {
    if (self.stopping) return;
    if (exitCode != 0) self.exitCode = exitCode;
    self.stopping = YES;
    broker = self.brokerConnection;
    host = self.hostConnection;
    listener = self.listener;
    bootstrapFd = self.bootstrapFd;
    self.bootstrapFd = -1;
    self.brokerConnection = nil;
    self.hostConnection = nil;
    self.listener = nil;
  }
  if (bootstrapFd >= 0) close(bootstrapFd);
  [listener invalidate];
  [host invalidate];
  [broker invalidate];
  dispatch_async(dispatch_get_main_queue(), ^{
    CFRunLoopStop(CFRunLoopGetMain());
  });
}

@end

@implementation ChromaspaceSourceExchangeProducerRelayListenerDelegate

- (BOOL)listener:(NSXPCListener*)listener
    shouldAcceptNewConnection:(NSXPCConnection*)connection {
  if (connection == nil ||
      connection.effectiveUserIdentifier != geteuid()) {
    return NO;
  }
  ChromaspaceSourceExchangeProducerRelayController* controller =
      self.controller;
  @synchronized(controller) {
    if (controller.stopping || controller.acceptedHost) return NO;
    controller.acceptedHost = YES;
    controller.hostConnection = connection;
  }

  ChromaspaceSourceExchangeProducerRelaySession* session =
      [[ChromaspaceSourceExchangeProducerRelaySession alloc] init];
  session.brokerConnection = controller.brokerConnection;
  session.controller = controller;
  connection.exportedInterface =
      producerInterface(
          @protocol(ChromaspaceSourceExchangeProducerRelayProtocol));
  connection.exportedObject = session;
  __weak ChromaspaceSourceExchangeProducerRelayController* weakController =
      controller;
  connection.invalidationHandler = ^{
    [weakController requestStop];
  };
  connection.interruptionHandler = ^{
    [weakController requestStop];
  };
  [connection resume];

  // The broker-redeemed endpoint is a one-shot authority. Once its single host
  // has connected, stop accepting peers without invalidating that connection.
  [listener invalidate];
  return YES;
}

@end

int main(int argc, const char* argv[]) {
  int result = 0;
  @autoreleasepool {
    signal(SIGPIPE, SIG_IGN);
    int bootstrapFd = bootstrapFileDescriptor(argc, argv);
    if (bootstrapFd < 0) return 64;

    NSXPCConnection* brokerConnection =
        [[NSXPCConnection alloc]
            initWithMachServiceName:
                ChromaspaceSourceExchangeMachServiceName
                            options:0];
    brokerConnection.remoteObjectInterface =
        producerInterface(
            @protocol(ChromaspaceSourceExchangeProducerBrokerProtocol));
    ChromaspaceSourceExchangeProducerRelayController*
        __attribute__((objc_precise_lifetime)) controller =
        [[ChromaspaceSourceExchangeProducerRelayController alloc] init];
    controller.brokerConnection = brokerConnection;
    controller.bootstrapFd = bootstrapFd;

    __weak ChromaspaceSourceExchangeProducerRelayController* weakController =
        controller;
    brokerConnection.invalidationHandler = ^{
      [weakController requestStopWithExitCode:69];
    };
    brokerConnection.interruptionHandler = ^{
      [weakController requestStopWithExitCode:69];
    };
    [brokerConnection resume];

    ChromaspaceSourceExchangeProducerRelayListenerDelegate*
        __attribute__((objc_precise_lifetime)) delegate =
        [[ChromaspaceSourceExchangeProducerRelayListenerDelegate alloc] init];
    delegate.controller = controller;
    NSXPCListener* __attribute__((objc_precise_lifetime)) listener =
        [NSXPCListener anonymousListener];
    controller.listener = listener;
    listener.delegate = delegate;
    [listener resume];

    NSData* bootstrapToken = createBootstrapToken();
    if (bootstrapToken == nil) {
      [controller requestStopWithExitCode:70];
      return 70;
    }
    ChromaspaceSourceExchangeRelayBootstrapRegistration* registration =
        [[ChromaspaceSourceExchangeRelayBootstrapRegistration alloc]
            initWithProtocolMajor:ChromaspaceSourceExchangeProtocolMajor
                    protocolMinor:ChromaspaceSourceExchangeProtocolMinor
                   bootstrapToken:bootstrapToken
            producerRelayEndpoint:listener.endpoint];
    NSError* validationError = nil;
    if (![registration validate:&validationError]) {
      [controller requestStopWithExitCode:70];
      return 70;
    }
    id<ChromaspaceSourceExchangeProducerBrokerProtocol> broker =
        [brokerConnection
            remoteObjectProxyWithErrorHandler:^(NSError* error) {
              (void)error;
              [weakController requestStopWithExitCode:69];
            }];
    dispatch_after(
        dispatch_time(DISPATCH_TIME_NOW, kBootstrapDeadlineNanoseconds),
        dispatch_get_main_queue(), ^{
          if (![controller hasDeliveredBootstrapToken]) {
            [controller requestStopWithExitCode:75];
          }
        });
    [broker
        registerProducerRelayBootstrap:registration
                             withReply:
                                 ^(ChromaspaceSourceExchangeStatus status,
                                   NSError* error) {
                                   (void)error;
                                   if (status !=
                                           ChromaspaceSourceExchangeStatusAccepted ||
                                       ![controller
                                           deliverBootstrapToken:
                                               bootstrapToken]) {
                                     [controller
                                         requestStopWithExitCode:74];
                                     return;
                                   }
                                   dispatch_after(
                                       dispatch_time(
                                           DISPATCH_TIME_NOW,
                                           kBootstrapDeadlineNanoseconds),
                                       dispatch_get_main_queue(), ^{
                                         if (![controller
                                                 hasAcceptedHost]) {
                                           [controller
                                               requestStopWithExitCode:
                                                   75];
                                         }
                                       });
                                 }];
    [[NSRunLoop currentRunLoop] run];
    result = controller.exitCode;
  }
  return result;
}
