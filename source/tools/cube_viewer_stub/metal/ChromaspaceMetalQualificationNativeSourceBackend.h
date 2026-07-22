#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "ChromaspaceMetalQualificationSourceAdapter.h"

namespace ChromaspaceMetalQualification {

struct NativeSourceFixtureSnapshot final {
  bool ready = false;
  bool failed = false;
  uint64_t deviceRegistryId = 0u;
  uint32_t createCount = 0u;
  uint32_t retireCount = 0u;
  uint32_t failureCount = 0u;
  uint32_t inFlightCount = 0u;
  uint64_t lastGeneration = 0u;
};

// This is a qualification-only native fixture.  The ordinary C++ header owns
// only the callback table and an opaque implementation; Objective-C and Metal
// objects exist exclusively in the .mm translation unit.
class NativeSourceFixtureBackend final {
 public:
  NativeSourceFixtureBackend() noexcept;
  ~NativeSourceFixtureBackend();

  NativeSourceFixtureBackend(const NativeSourceFixtureBackend&) = delete;
  NativeSourceFixtureBackend& operator=(const NativeSourceFixtureBackend&) = delete;
  NativeSourceFixtureBackend(NativeSourceFixtureBackend&&) = delete;
  NativeSourceFixtureBackend& operator=(NativeSourceFixtureBackend&&) = delete;

  const NativeSourceBackend* backend() const noexcept { return &backend_; }
  bool ready() const noexcept;
  bool failed() const noexcept;
  const char* diagnostic() const noexcept;
  NativeSourceFixtureSnapshot snapshot() const noexcept;

 private:
  struct Impl;

  static bool createCallback(void* context,
                             const std::string& senderId,
                             uint64_t deviceRegistryId,
                             uint64_t sourceGeneration,
                             ChromaspaceMetal::ImportedSourceTexture* outSource,
                             std::string* error) noexcept;
  static void retireCallback(void* context, uint64_t sourceId) noexcept;

  bool createInternal(const std::string& senderId,
                      uint64_t deviceRegistryId,
                      uint64_t sourceGeneration,
                      ChromaspaceMetal::ImportedSourceTexture* outSource,
                      std::string* error);
  void retireInternal(uint64_t sourceId);
  bool fail(const char* diagnostic, std::string* error = nullptr) noexcept;

  NativeSourceBackend backend_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace ChromaspaceMetalQualification
