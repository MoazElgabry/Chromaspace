#pragma once

#include <cstddef>

namespace ChromaspaceMetal {

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
    int overlayHeight);

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
    size_t readbackSrcRowBytes);

}  // namespace ChromaspaceMetal
