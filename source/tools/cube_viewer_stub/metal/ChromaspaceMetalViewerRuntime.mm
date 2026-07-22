#import "ChromaspaceMetalViewerRuntime.h"

#include "ChromaspaceMetal.h"

namespace ChromaspaceMetalViewerRuntime {
namespace {

bool createTextAtlas(void*,
                     uint64_t compositorId,
                     int width,
                     int height,
                     const unsigned char* alphaPixels,
                     std::size_t,
                     ChromaspaceMetal::FrameTextAtlas* outAtlas,
                     std::string* error) noexcept {
  return ChromaspaceMetal::createFrameTextAtlas(compositorId, width, height,
                                                 alphaPixels, outAtlas, error);
}

void releaseTextAtlas(void*, uint64_t compositorId, uint64_t atlasId) noexcept {
  ChromaspaceMetal::releaseFrameTextAtlas(compositorId, atlasId);
}

const RuntimeResourceBackend kAppleResourceBackend{
    nullptr, createTextAtlas, releaseTextAtlas};

}  // namespace

const RuntimeResourceBackend* defaultRuntimeResourceBackend() noexcept {
  return &kAppleResourceBackend;
}

}  // namespace ChromaspaceMetalViewerRuntime
