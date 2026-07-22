#pragma once

#include "ChromaspaceViewerWorkspacePersistence.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ChromaspaceViewer {

// The store is deliberately a small, platform-neutral seam around the
// bounded/versioned workspace codec.  Callers provide the canonical primary
// path and, when migrating an older installation, an optional v1 path.
struct ViewerWorkspaceStorePaths {
  std::filesystem::path primary;
  std::filesystem::path legacy;
};

constexpr std::size_t kViewerWorkspaceStoreMaxPathBytes = 4096u;
constexpr std::size_t kViewerWorkspaceStoreMaxDiagnosticBytes = 2048u;

enum class ViewerWorkspaceStoreLoadStatus : uint8_t {
  AcceptedPrimary = 0,
  AcceptedBackup,
  AcceptedLegacy,
  DefaultsMissing,
  DefaultsRecovery,
  InvalidArgument,
  AllocationFailure,
};

struct ViewerWorkspaceStoreLoadResult {
  ViewerWorkspaceStoreLoadStatus status =
      ViewerWorkspaceStoreLoadStatus::InvalidArgument;
  ViewerWorkspaceDocument document{};
  std::string diagnostic;
  // A backup or v1 document is usable, but the canonical primary can be
  // repaired after the user has a chance to continue.  DefaultsRecovery is
  // intentionally false here: corrupt evidence must not be overwritten by a
  // clean shutdown with no durable state change.
  bool repairSuggested = false;
  bool evidencePresent = false;

  bool accepted() const noexcept {
    return status == ViewerWorkspaceStoreLoadStatus::AcceptedPrimary ||
           status == ViewerWorkspaceStoreLoadStatus::AcceptedBackup ||
           status == ViewerWorkspaceStoreLoadStatus::AcceptedLegacy ||
           status == ViewerWorkspaceStoreLoadStatus::DefaultsMissing ||
           status == ViewerWorkspaceStoreLoadStatus::DefaultsRecovery;
  }

  bool degraded() const noexcept {
    return status == ViewerWorkspaceStoreLoadStatus::DefaultsRecovery;
  }
};

enum class ViewerWorkspaceStoreSaveStatus : uint8_t {
  Saved = 0,
  InvalidArgument,
  EncodeFailed,
  DirectoryCreateFailed,
  TempCreateFailed,
  TempWriteFailed,
  TempVerificationFailed,
  BackupRotationFailed,
  CorruptPreservationFailed,
  PromotionFailed,
  RollbackFailed,
  CleanupFailed,
  AllocationFailure,
};

struct ViewerWorkspaceStoreSaveResult {
  ViewerWorkspaceStoreSaveStatus status =
      ViewerWorkspaceStoreSaveStatus::InvalidArgument;
  std::string diagnostic;
  bool changed = false;

  bool accepted() const noexcept {
    return status == ViewerWorkspaceStoreSaveStatus::Saved;
  }
};

ViewerWorkspaceStoreLoadResult loadViewerWorkspaceStore(
    const ViewerWorkspaceStorePaths& paths) noexcept;

ViewerWorkspaceStoreSaveResult saveViewerWorkspaceStore(
    const ViewerWorkspaceStorePaths& paths,
    const ViewerWorkspaceDocument& document) noexcept;

}  // namespace ChromaspaceViewer

