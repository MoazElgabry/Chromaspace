#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "ChromaspaceResidentSourceSession.h"

namespace ChromaspaceMetalQualification {

// The platform owns the actual resident texture and only returns its opaque
// descriptor.  The qualification adapter owns this callback copy, never
// fabricates a source identity, and never performs platform work itself.
struct NativeSourceBackend final {
  void* context = nullptr;
  bool (*create)(void* context,
                 const std::string& senderId,
                 uint64_t deviceRegistryId,
                 uint64_t sourceGeneration,
                 ChromaspaceMetal::ImportedSourceTexture* outSource,
                 std::string* error) noexcept = nullptr;
  void (*retire)(void* context, uint64_t sourceId) noexcept = nullptr;
};

struct SourceCompletionExpectation final {
  uint32_t requiredPublishCount = 0u;
  uint32_t requiredClearCount = 0u;
  bool requireNoLiveClient = false;
  bool requireNoActiveSource = false;
};

struct SourceAdapterSnapshot final {
  bool ready = false;
  bool failed = false;
  bool clientLive = false;
  bool clientStarted = false;
  std::string senderId;
  uint64_t deviceRegistryId = 0u;
  uint64_t viewerGeneration = 0u;
  uint64_t lastAcceptedSourceGeneration = 0u;
  bool hasActiveSource = false;
  ChromaspaceMetal::ImportedSourceTexture activeSource{};
  uint32_t createCount = 0u;
  uint32_t startCount = 0u;
  uint32_t clearCount = 0u;
  uint32_t publishCount = 0u;
  uint32_t retireCount = 0u;
  uint32_t destroyCount = 0u;
  uint32_t failureCount = 0u;
};

class QualificationSourceAdapter final {
 public:
  explicit QualificationSourceAdapter(const NativeSourceBackend* backend) noexcept;
  explicit QualificationSourceAdapter(const NativeSourceBackend& backend) noexcept
      : QualificationSourceAdapter(&backend) {}
  ~QualificationSourceAdapter() = default;

  QualificationSourceAdapter(const QualificationSourceAdapter&) = delete;
  QualificationSourceAdapter& operator=(const QualificationSourceAdapter&) = delete;
  QualificationSourceAdapter(QualificationSourceAdapter&&) = delete;
  QualificationSourceAdapter& operator=(QualificationSourceAdapter&&) = delete;

  const ChromaspaceResidentSource::ClientAdapter* clientAdapter()
      const noexcept {
    return &clientAdapter_;
  }

  bool ready() const noexcept { return ready_ && !failed_; }
  bool failed() const noexcept { return failed_; }
  const char* diagnostic() const noexcept { return diagnostic_; }

  // Publishes one strictly newer campaign generation.  The backend's
  // descriptor must represent that generation in senderGeneration, sequence,
  // and slotGeneration, which keeps stale results unambiguous.
  bool publish(uint64_t sourceGeneration,
               std::string* diagnostic = nullptr) noexcept;

  bool finish(const SourceCompletionExpectation& expectation,
              std::string* diagnostic = nullptr) noexcept;
  bool finish(std::string* diagnostic = nullptr) noexcept;

  SourceAdapterSnapshot snapshot() const noexcept;

 private:
  static void* createClient(void* context,
                            const std::string& senderId,
                            uint64_t deviceRegistryId,
                            std::string* error) noexcept;
  static bool startClient(void* context,
                          void* client,
                          std::string* error) noexcept;
  static bool clearClient(void* context,
                          void* client,
                          std::string* error) noexcept;
  static bool snapshotClient(void* context,
                             const void* client,
                             ChromaspaceResidentSource::ClientSnapshot* snapshot,
                             std::string* error) noexcept;
  static void destroyClient(void* context, void* client) noexcept;

  bool validHandle(void* client, bool requireStarted) noexcept;
  const void* validHandle(const void* client, bool requireStarted) const noexcept;
  void* createClientInternal(const std::string& senderId,
                             uint64_t deviceRegistryId,
                             std::string* error) noexcept;
  bool startClientInternal(void* client, std::string* error) noexcept;
  bool clearClientInternal(void* client, std::string* error) noexcept;
  bool snapshotClientInternal(
      const void* client,
      ChromaspaceResidentSource::ClientSnapshot* snapshot,
      std::string* error) const noexcept;
  void destroyClientInternal(void* client) noexcept;

  bool validateSource(const ChromaspaceMetal::ImportedSourceTexture& source,
                      uint64_t expectedGeneration,
                      std::string* error) const noexcept;
  bool recordFailure(const char* diagnostic,
                     std::string* output = nullptr) noexcept;
  bool increment(uint32_t* counter, const char* diagnostic) noexcept;
  bool retireSource(uint64_t sourceId) noexcept;
  void clearActiveSourceNoThrow() noexcept;
  bool validBackend(const NativeSourceBackend* backend) const noexcept;

  struct ClientHandle final {
    QualificationSourceAdapter* owner = nullptr;
    uint64_t viewerGeneration = 0u;
    bool used = false;
    bool live = false;
    bool started = false;
  };

  // Slots are never reused.  This keeps a stale void* from becoming valid
  // after a later recreate; exhaustion fails closed instead of recycling it.
  static constexpr std::size_t kHandleSlotCount = 8u;
  NativeSourceBackend backend_{};
  ChromaspaceResidentSource::ClientAdapter clientAdapter_{};
  std::array<ClientHandle, kHandleSlotCount> handles_{};
  ClientHandle* activeHandle_ = nullptr;
  std::size_t nextHandleSlot_ = 0u;
  bool ready_ = false;
  bool failed_ = false;
  bool completed_ = false;
  bool clientLive_ = false;
  bool clientStarted_ = false;
  std::string senderId_;
  uint64_t deviceRegistryId_ = 0u;
  uint64_t nextViewerGeneration_ = 1u;
  uint64_t viewerGeneration_ = 0u;
  uint64_t lastAcceptedSourceGeneration_ = 0u;
  bool hasActiveSource_ = false;
  ChromaspaceMetal::ImportedSourceTexture activeSource_{};
  uint32_t createCount_ = 0u;
  uint32_t startCount_ = 0u;
  uint32_t clearCount_ = 0u;
  uint32_t publishCount_ = 0u;
  uint32_t retireCount_ = 0u;
  uint32_t destroyCount_ = 0u;
  uint32_t failureCount_ = 0u;
  const char* diagnostic_ = "qualification-source-adapter-invalid-backend";
};

using SourceAdapter = QualificationSourceAdapter;

}  // namespace ChromaspaceMetalQualification
