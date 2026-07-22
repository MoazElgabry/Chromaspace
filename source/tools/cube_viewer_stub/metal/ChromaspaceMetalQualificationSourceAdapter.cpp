#include "ChromaspaceMetalQualificationSourceAdapter.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace ChromaspaceMetalQualification {
namespace {

using ChromaspaceResidentSource::ClientAdapter;
using ChromaspaceResidentSource::ClientHealth;
using ChromaspaceMetal::ImportedSourceTexture;

constexpr const char* kReady = "qualification-source-adapter-ready";
constexpr const char* kInvalidBackend =
    "qualification-source-adapter-invalid-backend";
constexpr const char* kNotReady = "qualification-source-adapter-not-ready";
constexpr const char* kClientAlreadyLive =
    "qualification-source-adapter-client-already-live";
constexpr const char* kClientHandleInvalid =
    "qualification-source-adapter-client-handle-invalid";
constexpr const char* kClientAlreadyStarted =
    "qualification-source-adapter-client-already-started";
constexpr const char* kClientHandleExhausted =
    "qualification-source-adapter-client-handle-exhausted";
constexpr const char* kSourceGenerationInvalid =
    "qualification-source-adapter-source-generation-invalid";
constexpr const char* kSourceCreateFailed =
    "qualification-source-adapter-source-create-failed";
constexpr const char* kSourceInvalid =
    "qualification-source-adapter-source-invalid";
constexpr const char* kSourceSenderMismatch =
    "qualification-source-adapter-source-sender-mismatch";
constexpr const char* kSourceDeviceMismatch =
    "qualification-source-adapter-source-device-mismatch";
constexpr const char* kSourceGenerationMismatch =
    "qualification-source-adapter-source-generation-mismatch";
constexpr const char* kSourceIdentityReused =
    "qualification-source-adapter-source-identity-reused";
constexpr const char* kSourceLayoutInvalid =
    "qualification-source-adapter-source-layout-invalid";
constexpr const char* kSourceSemanticInvalid =
    "qualification-source-adapter-source-semantic-invalid";
constexpr const char* kFinishNotReady =
    "qualification-source-adapter-finish-not-ready";
constexpr const char* kFinishFailed =
    "qualification-source-adapter-finish-failed";
constexpr const char* kFinishCountMismatch =
    "qualification-source-adapter-finish-count-mismatch";
constexpr const char* kFinishLiveClient =
    "qualification-source-adapter-finish-live-client";
constexpr const char* kFinishActiveSource =
    "qualification-source-adapter-finish-active-source";
constexpr const char* kMetricsOverflow =
    "qualification-source-adapter-metrics-overflow";

void setErrorNoThrow(std::string* output, const char* value) noexcept {
  if (output == nullptr) return;
  try {
    *output = value != nullptr ? value : "qualification-source-adapter-error";
  } catch (...) {
  }
}

void clearClientSnapshotNoThrow(
    ChromaspaceResidentSource::ClientSnapshot* snapshot) noexcept {
  if (snapshot == nullptr) return;
  try {
    *snapshot = ChromaspaceResidentSource::ClientSnapshot{};
  } catch (...) {
    snapshot->health = ClientHealth::Stopped;
    snapshot->viewerGeneration = 0u;
    snapshot->lastObservedSequence = 0u;
    snapshot->liveKeyCount = 0u;
    snapshot->hasActiveSource = false;
  }
}

void clearSourceNoThrow(ImportedSourceTexture* source) noexcept {
  if (source == nullptr) return;
  try {
    *source = ImportedSourceTexture{};
  } catch (...) {
    source->sourceId = 0u;
    source->senderId.clear();
    source->deviceRegistryId = 0u;
    source->senderGeneration = 0u;
    source->sequence = 0u;
    source->slotIndex = 0u;
    source->slotGeneration = 0u;
    source->readyValue = 0u;
    source->contentHash = 0u;
    source->width = 0;
    source->height = 0;
    source->pixelFormat = 0;
    source->bytesPerRow = 0u;
    source->byteSize = 0u;
  }
}

bool validNativeBackend(const NativeSourceBackend* backend) noexcept {
  return backend != nullptr && backend->create != nullptr &&
         backend->retire != nullptr;
}

}  // namespace

QualificationSourceAdapter::QualificationSourceAdapter(
    const NativeSourceBackend* backend) noexcept {
  clientAdapter_.context = this;
  clientAdapter_.create = &QualificationSourceAdapter::createClient;
  clientAdapter_.start = &QualificationSourceAdapter::startClient;
  clientAdapter_.clear = &QualificationSourceAdapter::clearClient;
  clientAdapter_.snapshot = &QualificationSourceAdapter::snapshotClient;
  clientAdapter_.destroy = &QualificationSourceAdapter::destroyClient;
  if (!validNativeBackend(backend)) {
    diagnostic_ = kInvalidBackend;
    failed_ = true;
    return;
  }
  backend_ = *backend;
  ready_ = true;
  diagnostic_ = kReady;
}

void* QualificationSourceAdapter::createClient(
    void* context,
    const std::string& senderId,
    uint64_t deviceRegistryId,
    std::string* error) noexcept {
  auto* self = static_cast<QualificationSourceAdapter*>(context);
  return self != nullptr
             ? self->createClientInternal(senderId, deviceRegistryId, error)
             : (setErrorNoThrow(error, kNotReady), nullptr);
}

bool QualificationSourceAdapter::startClient(void* context,
                                              void* client,
                                              std::string* error) noexcept {
  auto* self = static_cast<QualificationSourceAdapter*>(context);
  return self != nullptr
             ? self->startClientInternal(client, error)
             : (setErrorNoThrow(error, kNotReady), false);
}

bool QualificationSourceAdapter::clearClient(void* context,
                                              void* client,
                                              std::string* error) noexcept {
  auto* self = static_cast<QualificationSourceAdapter*>(context);
  return self != nullptr
             ? self->clearClientInternal(client, error)
             : (setErrorNoThrow(error, kNotReady), false);
}

bool QualificationSourceAdapter::snapshotClient(
    void* context,
    const void* client,
    ChromaspaceResidentSource::ClientSnapshot* snapshot,
    std::string* error) noexcept {
  auto* self = static_cast<QualificationSourceAdapter*>(context);
  return self != nullptr
             ? self->snapshotClientInternal(client, snapshot, error)
             : (clearClientSnapshotNoThrow(snapshot),
                setErrorNoThrow(error, kNotReady), false);
}

void QualificationSourceAdapter::destroyClient(void* context,
                                               void* client) noexcept {
  auto* self = static_cast<QualificationSourceAdapter*>(context);
  if (self != nullptr) self->destroyClientInternal(client);
}

bool QualificationSourceAdapter::validHandle(void* client,
                                             bool requireStarted) noexcept {
  if (client == nullptr || activeHandle_ == nullptr ||
      client != activeHandle_ || !clientLive_ || activeHandle_->owner != this ||
      !activeHandle_->live ||
      (requireStarted && (!clientStarted_ || !activeHandle_->started))) {
    return false;
  }
  return true;
}

const void* QualificationSourceAdapter::validHandle(
    const void* client, bool requireStarted) const noexcept {
  if (client == nullptr || activeHandle_ == nullptr ||
      client != activeHandle_ || !clientLive_ || activeHandle_->owner != this ||
      !activeHandle_->live ||
      (requireStarted && (!clientStarted_ || !activeHandle_->started))) {
    return nullptr;
  }
  return activeHandle_;
}

bool QualificationSourceAdapter::increment(uint32_t* counter,
                                           const char* diagnostic) noexcept {
  if (counter == nullptr || *counter == std::numeric_limits<uint32_t>::max()) {
    failed_ = true;
    diagnostic_ = diagnostic != nullptr ? diagnostic : kMetricsOverflow;
    return false;
  }
  ++*counter;
  return true;
}

bool QualificationSourceAdapter::recordFailure(const char* diagnostic,
                                               std::string* output) noexcept {
  diagnostic_ = diagnostic != nullptr ? diagnostic : kSourceInvalid;
  setErrorNoThrow(output, diagnostic_);
  return increment(&failureCount_, kMetricsOverflow);
}

bool QualificationSourceAdapter::retireSource(uint64_t sourceId) noexcept {
  if (sourceId == 0u) return true;
  if (!ready_ || backend_.retire == nullptr) {
    failed_ = true;
    diagnostic_ = kInvalidBackend;
    return false;
  }
  backend_.retire(backend_.context, sourceId);
  return increment(&retireCount_, kMetricsOverflow);
}

void QualificationSourceAdapter::clearActiveSourceNoThrow() noexcept {
  const uint64_t sourceId = hasActiveSource_ ? activeSource_.sourceId : 0u;
  if (sourceId != 0u) (void)retireSource(sourceId);
  hasActiveSource_ = false;
  clearSourceNoThrow(&activeSource_);
}

void* QualificationSourceAdapter::createClientInternal(
    const std::string& senderId,
    uint64_t deviceRegistryId,
    std::string* error) noexcept {
  if (!ready_ || failed_) {
    setErrorNoThrow(error, failed_ ? diagnostic_ : kNotReady);
    return nullptr;
  }
  if (senderId.empty() || deviceRegistryId == 0u) {
    (void)recordFailure(kClientHandleInvalid, error);
    return nullptr;
  }
  if (senderId.size() > ChromaspaceSourceExchange::kMaximumSemanticIdentifierBytes) {
    (void)recordFailure(kClientHandleInvalid, error);
    return nullptr;
  }
  if (clientLive_) {
    (void)recordFailure(kClientAlreadyLive, error);
    return nullptr;
  }
  if (!increment(&createCount_, kMetricsOverflow)) {
    setErrorNoThrow(error, diagnostic_);
    return nullptr;
  }
  ClientHandle* handle = nullptr;
  for (std::size_t offset = 0u; offset < handles_.size(); ++offset) {
    const std::size_t index = (nextHandleSlot_ + offset) % handles_.size();
    if (!handles_[index].used) {
      handle = &handles_[index];
      nextHandleSlot_ = (index + 1u) % handles_.size();
      break;
    }
  }
  if (handle == nullptr) {
    failed_ = true;
    diagnostic_ = kClientHandleExhausted;
    setErrorNoThrow(error, diagnostic_);
    return nullptr;
  }
  try {
    senderId_ = senderId;
  } catch (...) {
    (void)recordFailure("qualification-source-adapter-sender-allocation-failed",
                        error);
    return nullptr;
  }
  deviceRegistryId_ = deviceRegistryId;
  clientLive_ = true;
  clientStarted_ = false;
  handle->owner = this;
  handle->viewerGeneration = 0u;
  handle->used = true;
  handle->live = true;
  handle->started = false;
  activeHandle_ = handle;
  diagnostic_ = kReady;
  setErrorNoThrow(error, kReady);
  return activeHandle_;
}

bool QualificationSourceAdapter::startClientInternal(void* client,
                                                      std::string* error) noexcept {
  if (!ready_ || failed_) {
    setErrorNoThrow(error, failed_ ? diagnostic_ : kNotReady);
    return false;
  }
  if (!validHandle(client, false)) {
    (void)recordFailure(kClientHandleInvalid, error);
    return false;
  }
  if (clientStarted_) {
    (void)recordFailure(kClientAlreadyStarted, error);
    return false;
  }
  if (nextViewerGeneration_ == 0u ||
      nextViewerGeneration_ == std::numeric_limits<uint64_t>::max()) {
    (void)recordFailure("qualification-source-adapter-viewer-generation-overflow",
                        error);
    failed_ = true;
    return false;
  }
  if (!increment(&startCount_, kMetricsOverflow)) {
    setErrorNoThrow(error, diagnostic_);
    return false;
  }
  viewerGeneration_ = nextViewerGeneration_++;
  activeHandle_->viewerGeneration = viewerGeneration_;
  activeHandle_->started = true;
  clientStarted_ = true;
  diagnostic_ = kReady;
  setErrorNoThrow(error, kReady);
  return true;
}

bool QualificationSourceAdapter::clearClientInternal(void* client,
                                                     std::string* error) noexcept {
  if (!ready_ || failed_) {
    setErrorNoThrow(error, failed_ ? diagnostic_ : kNotReady);
    return false;
  }
  if (!validHandle(client, true)) {
    (void)recordFailure(kClientHandleInvalid, error);
    return false;
  }
  if (!increment(&clearCount_, kMetricsOverflow)) {
    setErrorNoThrow(error, diagnostic_);
    return false;
  }
  clearActiveSourceNoThrow();
  diagnostic_ = kReady;
  setErrorNoThrow(error, kReady);
  return !failed_;
}

bool QualificationSourceAdapter::snapshotClientInternal(
    const void* client,
    ChromaspaceResidentSource::ClientSnapshot* snapshot,
    std::string* error) const noexcept {
  clearClientSnapshotNoThrow(snapshot);
  if (!ready_ || failed_) {
    setErrorNoThrow(error, failed_ ? diagnostic_ : kNotReady);
    return false;
  }
  const void* valid = validHandle(client, false);
  if (valid == nullptr || snapshot == nullptr) {
    setErrorNoThrow(error, kClientHandleInvalid);
    return false;
  }
  try {
    snapshot->health = clientStarted_ ? ClientHealth::Ready
                                      : ClientHealth::Registering;
    snapshot->viewerGeneration = viewerGeneration_;
    snapshot->lastObservedSequence =
        hasActiveSource_ ? activeSource_.sequence : 0u;
    snapshot->liveKeyCount = hasActiveSource_ ? 1u : 0u;
    snapshot->hasActiveSource = hasActiveSource_;
    if (hasActiveSource_) snapshot->activeSource = activeSource_;
    snapshot->diagnostic = diagnostic_ != nullptr ? diagnostic_ : kReady;
    return true;
  } catch (...) {
    setErrorNoThrow(error, "qualification-source-adapter-snapshot-copy-failed");
    return false;
  }
}

void QualificationSourceAdapter::destroyClientInternal(void* client) noexcept {
  if (!ready_ || !validHandle(client, false)) {
    (void)recordFailure(kClientHandleInvalid, nullptr);
    return;
  }
  clearActiveSourceNoThrow();
  activeHandle_->live = false;
  activeHandle_->started = false;
  activeHandle_->owner = this;
  activeHandle_ = nullptr;
  clientLive_ = false;
  clientStarted_ = false;
  senderId_.clear();
  deviceRegistryId_ = 0u;
  if (!increment(&destroyCount_, kMetricsOverflow)) return;
  diagnostic_ = kReady;
}

bool QualificationSourceAdapter::validateSource(
    const ImportedSourceTexture& source,
    uint64_t expectedGeneration,
    std::string* error) const noexcept {
  if (source.sourceId == 0u || source.senderId.empty() ||
      source.senderId.size() >
          ChromaspaceSourceExchange::kMaximumSemanticIdentifierBytes) {
    setErrorNoThrow(error, kSourceInvalid);
    return false;
  }
  if (source.senderId != senderId_) {
    setErrorNoThrow(error, kSourceSenderMismatch);
    return false;
  }
  if (source.deviceRegistryId != deviceRegistryId_) {
    setErrorNoThrow(error, kSourceDeviceMismatch);
    return false;
  }
  if (hasActiveSource_ && source.sourceId == activeSource_.sourceId) {
    setErrorNoThrow(error, kSourceIdentityReused);
    return false;
  }
  if (expectedGeneration == 0u ||
      source.senderGeneration != expectedGeneration ||
      source.sequence != expectedGeneration ||
      source.slotGeneration != expectedGeneration) {
    setErrorNoThrow(error, kSourceGenerationMismatch);
    return false;
  }
  if (source.slotIndex >= ChromaspaceSourceExchange::kMaximumSlots ||
      source.readyValue == 0u || source.contentHash == 0u || source.width <= 0 ||
      source.height <= 0 ||
      static_cast<uint64_t>(source.width) >
          ChromaspaceSourceExchange::kMaximumDimension ||
      static_cast<uint64_t>(source.height) >
          ChromaspaceSourceExchange::kMaximumDimension ||
      (source.pixelFormat != 0 && source.pixelFormat != 1) ||
      source.bytesPerRow == 0u || source.byteSize == 0u ||
      source.byteSize > ChromaspaceSourceExchange::kMaximumSurfaceBytes) {
    setErrorNoThrow(error, kSourceInvalid);
    return false;
  }
  const uint64_t bytesPerPixel = source.pixelFormat == 0 ? 8u : 16u;
  const uint64_t width = static_cast<uint64_t>(source.width);
  const uint64_t height = static_cast<uint64_t>(source.height);
  if (width > std::numeric_limits<uint64_t>::max() / bytesPerPixel ||
      static_cast<uint64_t>(source.bytesPerRow) < width * bytesPerPixel ||
      height > std::numeric_limits<uint64_t>::max() /
                   static_cast<uint64_t>(source.bytesPerRow) ||
      height * static_cast<uint64_t>(source.bytesPerRow) >
          static_cast<uint64_t>(source.byteSize)) {
    setErrorNoThrow(error, kSourceLayoutInvalid);
    return false;
  }
  if (!ChromaspaceSourceExchange::validSourceSemanticMetadata(
          source.semantics)) {
    setErrorNoThrow(error, kSourceSemanticInvalid);
    return false;
  }
  return true;
}

bool QualificationSourceAdapter::publish(uint64_t sourceGeneration,
                                         std::string* output) noexcept {
  if (!ready_ || failed_) {
    setErrorNoThrow(output, failed_ ? diagnostic_ : kNotReady);
    return false;
  }
  if (completed_ || !clientLive_ || !clientStarted_ || activeHandle_ == nullptr) {
    (void)recordFailure(kClientHandleInvalid, output);
    return false;
  }
  if (sourceGeneration == 0u ||
      sourceGeneration <= lastAcceptedSourceGeneration_) {
    (void)recordFailure(kSourceGenerationInvalid, output);
    return false;
  }
  if (publishCount_ == std::numeric_limits<uint32_t>::max()) {
    (void)recordFailure(kMetricsOverflow, output);
    failed_ = true;
    return false;
  }

  ImportedSourceTexture candidate{};
  std::string createError;
  const bool created = backend_.create(
      backend_.context, senderId_, deviceRegistryId_, sourceGeneration,
      &candidate, &createError);
  std::string validationError;
  if (!created ||
      !validateSource(candidate, sourceGeneration, &validationError)) {
    const uint64_t candidateId = candidate.sourceId;
    if (candidateId != 0u &&
        (!hasActiveSource_ || candidateId != activeSource_.sourceId)) {
      (void)retireSource(candidateId);
    }
    const char* failure = created ? kSourceInvalid : kSourceCreateFailed;
    (void)recordFailure(failure, output);
    if (created && !validationError.empty()) {
      setErrorNoThrow(output, validationError.c_str());
    } else if (!created && !createError.empty()) {
      setErrorNoThrow(output, createError.c_str());
    }
    return false;
  }

  const uint64_t priorSourceId =
      hasActiveSource_ ? activeSource_.sourceId : 0u;
  try {
    std::swap(activeSource_, candidate);
  } catch (...) {
    const uint64_t candidateId = candidate.sourceId;
    if (candidateId != 0u && candidateId != priorSourceId) {
      (void)retireSource(candidateId);
    }
    (void)recordFailure("qualification-source-adapter-source-swap-failed",
                        output);
    return false;
  }
  hasActiveSource_ = true;
  lastAcceptedSourceGeneration_ = sourceGeneration;
  ++publishCount_;
  diagnostic_ = kReady;
  setErrorNoThrow(output, kReady);
  if (priorSourceId != 0u) (void)retireSource(priorSourceId);
  return !failed_;
}

bool QualificationSourceAdapter::finish(
    const SourceCompletionExpectation& expectation,
    std::string* output) noexcept {
  if (!ready_) {
    setErrorNoThrow(output, kFinishNotReady);
    return false;
  }
  if (failed_) {
    setErrorNoThrow(output, kFinishFailed);
    return false;
  }
  if (publishCount_ != expectation.requiredPublishCount ||
      clearCount_ != expectation.requiredClearCount) {
    setErrorNoThrow(output, kFinishCountMismatch);
    return false;
  }
  if (expectation.requireNoLiveClient && clientLive_) {
    setErrorNoThrow(output, kFinishLiveClient);
    return false;
  }
  if (expectation.requireNoActiveSource && hasActiveSource_) {
    setErrorNoThrow(output, kFinishActiveSource);
    return false;
  }
  completed_ = true;
  diagnostic_ = kReady;
  setErrorNoThrow(output, kReady);
  return true;
}

bool QualificationSourceAdapter::finish(std::string* output) noexcept {
  return finish(SourceCompletionExpectation{}, output);
}

SourceAdapterSnapshot QualificationSourceAdapter::snapshot() const noexcept {
  SourceAdapterSnapshot output{};
  try {
    output.ready = ready_ && !failed_ && clientLive_ && clientStarted_;
    output.failed = failed_;
    output.clientLive = clientLive_;
    output.clientStarted = clientStarted_;
    output.senderId = senderId_;
    output.deviceRegistryId = deviceRegistryId_;
    output.viewerGeneration = viewerGeneration_;
    output.lastAcceptedSourceGeneration = lastAcceptedSourceGeneration_;
    output.hasActiveSource = hasActiveSource_;
    if (hasActiveSource_) output.activeSource = activeSource_;
    output.createCount = createCount_;
    output.startCount = startCount_;
    output.clearCount = clearCount_;
    output.publishCount = publishCount_;
    output.retireCount = retireCount_;
    output.destroyCount = destroyCount_;
    output.failureCount = failureCount_;
  } catch (...) {
    output = SourceAdapterSnapshot{};
    output.failed = true;
  }
  return output;
}

}  // namespace ChromaspaceMetalQualification
