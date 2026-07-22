#pragma once

#include "ChromaspaceSourceExchangeState.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ChromaspaceSourceExchange {

enum class SourceProducerClientHealth {
  Stopped,
  Starting,
  AwaitingViewerRoute,
  Ready,
  Failed,
};

struct SourceProducerClientConfiguration {
  std::string senderId;
  uint64_t senderGeneration = 0;
  void* metalCommandQueue = nullptr;
};

struct SourceProducerClientSnapshot {
  SourceProducerClientHealth health =
      SourceProducerClientHealth::Stopped;
  uint64_t deviceRegistryId = 0;
  uint64_t releaseFetchCursor = 0;
  uint64_t acknowledgedReleaseOrdinal = 0;
  size_t livePublicationCount = 0;
  std::string diagnostic;
};

enum class SourceProducerBindResult {
  Bound,
  AlreadyBound,
  DeviceMismatch,
  Invalid,
  Failed,
};

struct SourceProducerResourceShape {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pixelFormat = 0;  // 0=RGBA16F, 1=RGBA32F.
};

struct SourceProducerFrameRequest {
  // One sender-domain source-generation sequence. It is allocated by the OFX
  // instance and reused by every transport representation of this generation.
  uint64_t sequence = 0;
  SourceSemanticMetadata semantics;
  // The exact host command queue for this render. Encoding on this queue
  // preserves ordering with the host-owned source buffer.
  void* metalCommandQueue = nullptr;
  const void* sourceMetalBuffer = nullptr;
  int sourceWidth = 0;
  int sourceHeight = 0;
  size_t sourceRowBytes = 0;
  int sourceOriginX = 0;
  int sourceOriginY = 0;
  SourceProducerResourceShape output;
  uint64_t contentHash = 0;
};

enum class SourceProducerSubmitResult {
  Enqueued,
  NotReady,
  LockBusy,
  ResourceShapeMismatch,
  BackPressure,
  InvalidRequest,
  EncodeFailed,
  Failed,
};

struct SourceProducerClient;

// Creation is inert. startSourceProducerClient queues all process/XPC work on
// the client's private serial queue and returns immediately.
SourceProducerClient* createSourceProducerClient(
    const SourceProducerClientConfiguration& configuration);

// Binds the first observed host Metal device and starts transport
// asynchronously. Later queues are accepted only on that exact device; each
// frame still encodes on the queue supplied in SourceProducerFrameRequest.
SourceProducerBindResult bindSourceProducerMetalCommandQueue(
    SourceProducerClient* client,
    void* metalCommandQueue);
void startSourceProducerClient(SourceProducerClient* client);

// Resource allocation and pipeline preparation run asynchronously on the
// client's private queue. Call this before the OFX render path begins
// submitting a new shape.
void prepareSourceProducerResources(
    SourceProducerClient* client,
    const SourceProducerResourceShape& shape);

// Render-safe submission seam: never waits for Metal or XPC and uses try-lock
// admission. GPU completion, packet construction, exact-retry publication,
// and release processing remain on the client's private queue.
SourceProducerSubmitResult tryEnqueueSourceProducerFrame(
    SourceProducerClient* client,
    const SourceProducerFrameRequest& request);

// Teardown never calls OFX interfaces. It may wait briefly for the private
// bootstrap worker to observe cancellation, so call it only from instance
// destruction, never from render.
void destroySourceProducerClient(SourceProducerClient* client);

SourceProducerClientSnapshot sourceProducerClientSnapshot(
    const SourceProducerClient* client);

}  // namespace ChromaspaceSourceExchange
