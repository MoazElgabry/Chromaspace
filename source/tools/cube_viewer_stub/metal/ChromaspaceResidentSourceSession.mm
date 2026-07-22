#include "ChromaspaceResidentSourceSession.h"

#include "ChromaspaceSourceViewerClient.h"

namespace ChromaspaceResidentSource {
namespace {

void setErrorNoThrow(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message != nullptr ? message : "resident-source-adapter-error";
  } catch (...) {
  }
}

void resetSnapshotNoThrow(ClientSnapshot* output) noexcept {
  if (output == nullptr) return;
  output->health = ClientHealth::Stopped;
  output->viewerGeneration = 0u;
  output->lastObservedSequence = 0u;
  output->liveKeyCount = 0u;
  output->hasActiveSource = false;
  try {
    output->activeSource = ChromaspaceMetal::ImportedSourceTexture{};
  } catch (...) {
  }
  try {
    output->diagnostic.clear();
  } catch (...) {
  }
}

ClientHealth mapHealth(
    ChromaspaceSourceExchange::SourceViewerClientHealth health) noexcept {
  using ChromaspaceSourceExchange::SourceViewerClientHealth;
  switch (health) {
    case SourceViewerClientHealth::Stopped: return ClientHealth::Stopped;
    case SourceViewerClientHealth::Registering:
        return ClientHealth::Registering;
    case SourceViewerClientHealth::BindingRoute:
        return ClientHealth::BindingRoute;
    case SourceViewerClientHealth::Ready: return ClientHealth::Ready;
    case SourceViewerClientHealth::Failed: return ClientHealth::Failed;
    default: return ClientHealth::Failed;
  }
}

void* clientCreate(void*,
                   const std::string& senderId,
                   uint64_t deviceRegistryId,
                   std::string* error) noexcept {
  try {
    ChromaspaceSourceExchange::SourceViewerClientConfiguration configuration{};
    configuration.senderId = senderId;
    configuration.deviceRegistryId = deviceRegistryId;
    configuration.pixelFormatMask =
        ChromaspaceSourceExchange::kPixelFormatRGBA16Float |
        ChromaspaceSourceExchange::kPixelFormatRGBA32Float;
    configuration.maximumWidth =
        ChromaspaceSourceExchange::kMaximumDimension;
    configuration.maximumHeight =
        ChromaspaceSourceExchange::kMaximumDimension;
    configuration.maximumSurfaceBytes =
        ChromaspaceSourceExchange::kMaximumSurfaceBytes;
    configuration.maximumRetainedBytes =
        ChromaspaceSourceExchange::kMaximumRetainedBytes;
    return ChromaspaceSourceExchange::createSourceViewerClient(
        configuration);
  } catch (...) {
    setErrorNoThrow(error, "resident-source-client-create-exception");
    return nullptr;
  }
}

bool clientStart(void*, void* client, std::string* error) noexcept {
  try {
    if (client == nullptr) {
      setErrorNoThrow(error, "resident-source-client-missing");
      return false;
    }
    ChromaspaceSourceExchange::startSourceViewerClient(
        static_cast<ChromaspaceSourceExchange::SourceViewerClient*>(client));
    return true;
  } catch (...) {
    setErrorNoThrow(error, "resident-source-client-start-exception");
    return false;
  }
}

bool clientClear(void*, void* client, std::string* error) noexcept {
  try {
    if (client == nullptr) {
      setErrorNoThrow(error, "resident-source-client-missing");
      return false;
    }
    return ChromaspaceSourceExchange::clearSourceViewerClient(
        static_cast<ChromaspaceSourceExchange::SourceViewerClient*>(client));
  } catch (...) {
    setErrorNoThrow(error, "resident-source-client-clear-exception");
    return false;
  }
}

bool clientSnapshot(void*,
                    const void* client,
                    ClientSnapshot* output,
                    std::string* error) noexcept {
  if (output == nullptr) {
    setErrorNoThrow(error, "resident-source-snapshot-output-missing");
    return false;
  }
  try {
    if (client == nullptr) {
      resetSnapshotNoThrow(output);
      setErrorNoThrow(error, "resident-source-client-missing");
      return false;
    }
    const auto snapshot =
        ChromaspaceSourceExchange::sourceViewerClientSnapshot(
            static_cast<const ChromaspaceSourceExchange::SourceViewerClient*>(
                client));
    output->health = mapHealth(snapshot.health);
    output->viewerGeneration = snapshot.viewerGeneration;
    output->lastObservedSequence = snapshot.lastObservedSequence;
    output->liveKeyCount = snapshot.liveKeyCount;
    output->hasActiveSource = snapshot.hasActiveSource;
    output->activeSource = snapshot.activeSource;
    output->diagnostic = snapshot.diagnostic;
    return true;
  } catch (...) {
    resetSnapshotNoThrow(output);
    setErrorNoThrow(error, "resident-source-client-snapshot-exception");
    return false;
  }
}

void clientDestroy(void*, void* client) noexcept {
  if (client == nullptr) return;
  try {
    ChromaspaceSourceExchange::destroySourceViewerClient(
        static_cast<ChromaspaceSourceExchange::SourceViewerClient*>(client));
  } catch (...) {
  }
}

const ClientAdapter kAppleClientAdapter{
    nullptr,
    clientCreate,
    clientStart,
    clientClear,
    clientSnapshot,
    clientDestroy};

}  // namespace

const ClientAdapter* defaultResidentSourceClientAdapter() noexcept {
  return &kAppleClientAdapter;
}

}  // namespace ChromaspaceResidentSource
