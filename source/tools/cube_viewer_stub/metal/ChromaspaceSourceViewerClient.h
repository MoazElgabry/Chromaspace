#pragma once

#include "ChromaspaceMetal.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ChromaspaceSourceExchange {

enum class SourceViewerClientHealth {
  Stopped,
  Registering,
  BindingRoute,
  Ready,
  Failed,
};

struct SourceViewerClientConfiguration {
  std::string senderId;
  uint64_t deviceRegistryId = 0;
  uint32_t pixelFormatMask = 0;
  uint32_t maximumWidth = 0;
  uint32_t maximumHeight = 0;
  uint64_t maximumSurfaceBytes = 0;
  uint64_t maximumRetainedBytes = 0;
};

struct SourceViewerClientSnapshot {
  SourceViewerClientHealth health = SourceViewerClientHealth::Stopped;
  uint64_t viewerGeneration = 0;
  uint64_t lastObservedSequence = 0;
  size_t liveKeyCount = 0;
  bool hasActiveSource = false;
  ChromaspaceMetal::ImportedSourceTexture activeSource;
  std::string diagnostic;
};

struct SourceViewerClient;

SourceViewerClient* createSourceViewerClient(
    const SourceViewerClientConfiguration& configuration);
void startSourceViewerClient(SourceViewerClient* client);
bool clearSourceViewerClient(SourceViewerClient* client);
void destroySourceViewerClient(SourceViewerClient* client);
SourceViewerClientSnapshot sourceViewerClientSnapshot(
    const SourceViewerClient* client);

}  // namespace ChromaspaceSourceExchange
