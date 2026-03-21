#include "ChromaspaceMetal.h"

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>

namespace ChromaspaceMetal {

namespace {

inline size_t packedRowBytesForWidth(int width) {
  return static_cast<size_t>(width) * 4u * sizeof(float);
}

inline size_t offsetForOrigin(size_t rowBytes, int originX, int originY) {
  return static_cast<size_t>(originY) * rowBytes + static_cast<size_t>(originX) * 4u * sizeof(float);
}

bool encodeCopyRows(
    id<MTLBlitCommandEncoder> blit,
    id<MTLBuffer> src,
    id<MTLBuffer> dst,
    size_t srcOffset,
    size_t dstOffset,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int width,
    int height) {
  const size_t packedRowBytes = packedRowBytesForWidth(width);
  if (packedRowBytes == 0 || height <= 0) return false;
  for (int y = 0; y < height; ++y) {
    const size_t srcRowOffset = srcOffset + static_cast<size_t>(y) * srcRowBytes;
    const size_t dstRowOffset = dstOffset + static_cast<size_t>(y) * dstRowBytes;
    [blit copyFromBuffer:src sourceOffset:srcRowOffset toBuffer:dst destinationOffset:dstRowOffset size:packedRowBytes];
  }
  return true;
}

bool validateRows(int width, int height, size_t srcRowBytes, size_t dstRowBytes) {
  if (width <= 0 || height <= 0) return false;
  const size_t packedRowBytes = packedRowBytesForWidth(width);
  return srcRowBytes >= packedRowBytes && dstRowBytes >= packedRowBytes;
}

}  // namespace

bool copyHostBuffers(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    const float* overlayPixels,
    int overlayX,
    int overlayY,
    int overlayWidth,
    int overlayHeight) {
  if (!srcMetalBuffer || !dstMetalBuffer || !metalCommandQueue) return false;
  if (!validateRows(width, height, srcRowBytes, dstRowBytes)) return false;

  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCommandQueue;
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcMetalBuffer;
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstMetalBuffer;
  if (queue == nil || src == nil || dst == nil) return false;
  id<MTLDevice> device = queue.device;
  if (device == nil) return false;

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  if (cmd == nil) return false;
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  if (blit == nil) return false;

  const size_t srcOffset = offsetForOrigin(srcRowBytes, originX, originY);
  const size_t dstOffset = offsetForOrigin(dstRowBytes, originX, originY);
  if (!encodeCopyRows(blit, src, dst, srcOffset, dstOffset, srcRowBytes, dstRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  if (overlayPixels != nullptr && overlayWidth > 0 && overlayHeight > 0) {
    const size_t overlayPackedRowBytes = packedRowBytesForWidth(overlayWidth);
    const size_t overlayBytes = overlayPackedRowBytes * static_cast<size_t>(overlayHeight);
    id<MTLBuffer> overlayBuffer =
        [device newBufferWithBytes:overlayPixels length:overlayBytes options:MTLResourceStorageModeShared];
    if (overlayBuffer == nil) {
      [blit endEncoding];
      return false;
    }
    const size_t overlayDstOffset = offsetForOrigin(dstRowBytes, overlayX, overlayY);
    if (!encodeCopyRows(blit, overlayBuffer, dst, 0, overlayDstOffset, overlayPackedRowBytes, dstRowBytes, overlayWidth, overlayHeight)) {
      [blit endEncoding];
      return false;
    }
  }
  [blit endEncoding];
  [cmd commit];
  return true;
}

bool copyHostBuffersReadback(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    void* metalCommandQueue,
    float* readbackSrc,
    size_t readbackSrcRowBytes) {
  if (!srcMetalBuffer || !dstMetalBuffer || !metalCommandQueue || !readbackSrc) return false;
  if (!validateRows(width, height, srcRowBytes, dstRowBytes)) return false;
  if (readbackSrcRowBytes < packedRowBytesForWidth(width)) return false;

  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)metalCommandQueue;
  id<MTLBuffer> src = (__bridge id<MTLBuffer>)srcMetalBuffer;
  id<MTLBuffer> dst = (__bridge id<MTLBuffer>)dstMetalBuffer;
  if (queue == nil || src == nil || dst == nil) return false;

  id<MTLDevice> device = queue.device;
  if (device == nil) return false;

  const size_t readbackBytes = readbackSrcRowBytes * static_cast<size_t>(height);
  id<MTLBuffer> readbackBuffer =
      [device newBufferWithLength:readbackBytes options:MTLResourceStorageModeShared];
  if (readbackBuffer == nil) return false;

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  if (cmd == nil) return false;
  id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
  if (blit == nil) return false;

  const size_t srcOffset = offsetForOrigin(srcRowBytes, originX, originY);
  const size_t dstOffset = offsetForOrigin(dstRowBytes, originX, originY);
  if (!encodeCopyRows(blit, src, dst, srcOffset, dstOffset, srcRowBytes, dstRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  if (!encodeCopyRows(blit, src, readbackBuffer, srcOffset, 0, srcRowBytes, readbackSrcRowBytes, width, height)) {
    [blit endEncoding];
    return false;
  }
  [blit endEncoding];
  [cmd commit];
  [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) return false;

  std::memcpy(readbackSrc, readbackBuffer.contents, readbackBytes);
  return true;
}

}  // namespace ChromaspaceMetal

#endif
