#include "ChromaspaceMetalPlotRenderer.h"

#include "ChromaspaceMetal.h"

namespace ChromaspaceMetalPlotRenderer {
namespace {

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "metal-plot-renderer-error";
  } catch (...) {
  }
}

bool createSurface(void*, uint64_t compositorId, int width, int height,
                  int pixelFormat, ChromaspaceMetal::PlotSurface* outSurface,
                  std::string* error) noexcept {
  try {
    return ChromaspaceMetal::createPrivatePlotSurface(
        compositorId, width, height, pixelFormat, outSurface, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-plot-surface-create-exception");
    return false;
  }
}

void releaseSurface(void*, uint64_t compositorId, uint32_t surfaceId) noexcept {
  try {
    ChromaspaceMetal::releasePrivatePlotSurface(compositorId, surfaceId);
  } catch (...) {
  }
}

bool encodeSource(void*, const ChromaspaceMetal::FrameSubmission& submission,
                  uint64_t sourceId, uint32_t surfaceId, int width, int height,
                  int pixelFormat, std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeSourceSignalSurfaceFromImportedTexture(
        submission, sourceId, surfaceId, width, height, pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-source-surface-encode-exception");
    return false;
  }
}

bool encodeHistogram(void*, const ChromaspaceMetal::FrameSubmission& submission,
                     const ChromaspaceMetal::RasterSourceRequest& raster,
                     const ChromaspaceMetal::HistogramSurfaceRequest& request,
                     uint64_t sourceId, uint32_t surfaceId, int width, int height,
                     int pixelFormat, std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeHistogramSurfaceFromImportedTexture(
        submission, raster, request, sourceId, surfaceId, width, height,
        pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-histogram-surface-encode-exception");
    return false;
  }
}

bool encodeWaveform(void*, const ChromaspaceMetal::FrameSubmission& submission,
                    const ChromaspaceMetal::RasterSourceRequest& raster,
                    const ChromaspaceMetal::WaveformSurfaceRequest& request,
                    uint64_t sourceId, uint32_t surfaceId, int width, int height,
                    int pixelFormat, std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeWaveformSurfaceFromImportedTexture(
        submission, raster, request, sourceId, surfaceId, width, height,
        pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-waveform-surface-encode-exception");
    return false;
  }
}

bool encodeRaster(void*, const ChromaspaceMetal::FrameSubmission& submission,
                  ChromaspaceMetal::ResidentDerivedCache* cache,
                  const ChromaspaceMetal::RasterSourceRequest& raster,
                  const ChromaspaceMetal::RasterPointSurfaceRequest& point,
                  uint64_t sourceId, uint64_t buildSerial,
                  uint32_t surfaceId, int width, int height, int pixelFormat,
                  std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeRasterPointSurfaceFromImportedTextureCached(
        submission, cache, raster, point, sourceId, buildSerial, surfaceId,
        width, height, pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-raster-surface-encode-exception");
    return false;
  }
}

bool encodeGlossField(void*, const ChromaspaceMetal::FrameSubmission& submission,
                      ChromaspaceMetal::GlossFieldCache* cache,
                      const ChromaspaceMetal::RasterSourceRequest& raster,
                      const ChromaspaceMetal::GlossFieldRequest& request,
                      uint64_t sourceId, uint64_t buildSerial,
                      std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeGlossFieldFromImportedTexture(
        submission, cache, raster, request, sourceId, buildSerial, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-gloss-field-encode-exception");
    return false;
  }
}

bool encodeGlossSurface(
    void*, const ChromaspaceMetal::FrameSubmission& submission,
    const ChromaspaceMetal::GlossFieldCache& cache,
    const ChromaspaceMetal::GlossFieldSurfaceRequest& request,
    uint32_t surfaceId, int width, int height, int pixelFormat,
    std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeGlossFieldSurfaceFromCache(
        submission, cache, request, surfaceId, width, height, pixelFormat,
        error);
  } catch (...) {
    setErrorNoThrow(error, "metal-gloss-surface-encode-exception");
    return false;
  }
}

bool encodeGlossProjection(
    void*, const ChromaspaceMetal::FrameSubmission& submission,
    const ChromaspaceMetal::GlossFieldCache& cache,
    const ChromaspaceMetal::GlossProjectionSurfaceRequest& request,
    uint32_t surfaceId, int width, int height, int pixelFormat,
    std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeGlossProjectionSurfaceFromCache(
        submission, cache, request, surfaceId, width, height, pixelFormat,
        error);
  } catch (...) {
    setErrorNoThrow(error, "metal-gloss-projection-encode-exception");
    return false;
  }
}

bool encodeVectors(void*, const ChromaspaceMetal::FrameSubmission& submission,
                   uint32_t surfaceId, int width, int height, int pixelFormat,
                   const ChromaspaceMetal::FrameVectorVertex* vertices,
                   std::size_t vertexCount, bool clearBeforeDraw,
                   const std::array<float, 4>& clearColor,
                   std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodePlotSurfaceVectorPrimitives(
        submission, surfaceId, width, height, pixelFormat, vertices, vertexCount,
        clearBeforeDraw, clearColor[0], clearColor[1], clearColor[2],
        clearColor[3], error);
  } catch (...) {
    setErrorNoThrow(error, "metal-vector-surface-encode-exception");
    return false;
  }
}

ChromaspaceMetal::GlossFieldCacheState glossCacheState(
    void*, const ChromaspaceMetal::GlossFieldCache& cache) noexcept {
  try {
    return ChromaspaceMetal::glossFieldCacheState(cache);
  } catch (...) {
    return ChromaspaceMetal::GlossFieldCacheState::Missing;
  }
}

void releaseGlossCache(void*, ChromaspaceMetal::GlossFieldCache* cache) noexcept {
  try {
    ChromaspaceMetal::releaseGlossFieldCache(cache);
  } catch (...) {
  }
}

bool encodeHistogramCached(
    void*, const ChromaspaceMetal::FrameSubmission& submission,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest& raster,
    const ChromaspaceMetal::HistogramSurfaceRequest& request,
    uint64_t sourceId, uint64_t buildSerial, uint32_t surfaceId, int width,
    int height, int pixelFormat, std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeHistogramSurfaceFromImportedTextureCached(
        submission, cache, raster, request, sourceId, buildSerial, surfaceId,
        width, height, pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-histogram-cache-encode-exception");
    return false;
  }
}

bool encodeWaveformCached(
    void*, const ChromaspaceMetal::FrameSubmission& submission,
    ChromaspaceMetal::ResidentDerivedCache* cache,
    const ChromaspaceMetal::RasterSourceRequest& raster,
    const ChromaspaceMetal::WaveformSurfaceRequest& request,
    uint64_t sourceId, uint64_t buildSerial, uint32_t surfaceId, int width,
    int height, int pixelFormat, std::string* error) noexcept {
  try {
    return ChromaspaceMetal::encodeWaveformSurfaceFromImportedTextureCached(
        submission, cache, raster, request, sourceId, buildSerial, surfaceId,
        width, height, pixelFormat, error);
  } catch (...) {
    setErrorNoThrow(error, "metal-waveform-cache-encode-exception");
    return false;
  }
}

ChromaspaceMetal::ResidentDerivedCacheState derivedCacheState(
    void*, const ChromaspaceMetal::ResidentDerivedCache& cache) noexcept {
  try {
    return ChromaspaceMetal::residentDerivedCacheState(cache);
  } catch (...) {
    return ChromaspaceMetal::ResidentDerivedCacheState::Missing;
  }
}

void releaseDerivedCache(
    void*, ChromaspaceMetal::ResidentDerivedCache* cache) noexcept {
  try {
    ChromaspaceMetal::releaseResidentDerivedCache(cache);
  } catch (...) {
  }
}

const RendererBackend kBackend{
    nullptr,
    createSurface,
    releaseSurface,
    encodeSource,
    encodeHistogram,
    encodeWaveform,
    encodeRaster,
    encodeGlossField,
    encodeGlossSurface,
    encodeGlossProjection,
    encodeVectors,
    glossCacheState,
    releaseGlossCache,
    encodeHistogramCached,
    encodeWaveformCached,
    derivedCacheState,
    releaseDerivedCache};

}  // namespace

const RendererBackend* defaultRendererBackend() noexcept { return &kBackend; }

}  // namespace ChromaspaceMetalPlotRenderer
