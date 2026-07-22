#include "ChromaspaceViewerWorkspaceStore.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <limits>
#include <new>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ChromaspaceViewer {
namespace {

constexpr std::size_t kMaxTempAttempts = 32u;

bool isNotFoundError(const std::error_code& error) noexcept {
  if (!error) return false;
  // POSIX ENOENT and Win32 ERROR_FILE_NOT_FOUND/ERROR_PATH_NOT_FOUND are
  // represented by these stable values in the filesystem error category.
  return error == std::make_error_code(std::errc::no_such_file_or_directory) ||
         error.value() == 2 || error.value() == 3;
}

void appendDiagnostic(std::string* diagnostic, std::string_view message) noexcept {
  if (!diagnostic || message.empty()) return;
  try {
    if (diagnostic->size() >= kViewerWorkspaceStoreMaxDiagnosticBytes) return;
    if (!diagnostic->empty()) {
      constexpr std::string_view separator = "; ";
      const std::size_t room =
          kViewerWorkspaceStoreMaxDiagnosticBytes - diagnostic->size();
      diagnostic->append(separator.substr(0u, std::min(room, separator.size())));
    }
    if (diagnostic->size() >= kViewerWorkspaceStoreMaxDiagnosticBytes) return;
    const std::size_t room =
        kViewerWorkspaceStoreMaxDiagnosticBytes - diagnostic->size();
    diagnostic->append(message.substr(0u, std::min(room, message.size())));
  } catch (...) {
    // Diagnostics are deliberately best effort.  They must never turn an
    // otherwise safe defaults/recovery result into an exception.
  }
}

bool pathString(const std::filesystem::path& path, std::string* output) noexcept {
  if (!output) return false;
  try {
    *output = path.string();
    return !output->empty() &&
           output->size() <= kViewerWorkspaceStoreMaxPathBytes;
  } catch (...) {
    output->clear();
    return false;
  }
}

bool pathForSuffix(const std::filesystem::path& base, std::string_view suffix,
                  std::filesystem::path* output) noexcept {
  if (!output || suffix.empty()) return false;
  std::string text;
  if (!pathString(base, &text)) return false;
  if (suffix.size() > kViewerWorkspaceStoreMaxPathBytes - text.size()) return false;
  text.append(suffix);
  try {
    *output = std::filesystem::path(text);
    return true;
  } catch (...) {
    return false;
  }
}

bool pathExists(const std::filesystem::path& path, bool* exists,
                std::string* diagnostic) noexcept {
  if (!exists) return false;
  *exists = false;
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error) {
    if (isNotFoundError(error)) return true;
    appendDiagnostic(diagnostic, "filesystem status failed");
    return false;
  }
  *exists = status.type() != std::filesystem::file_type::not_found;
  return true;
}

const char* persistenceStatusName(
    ViewerWorkspacePersistenceStatus status) noexcept {
  switch (status) {
    case ViewerWorkspacePersistenceStatus::Accepted: return "accepted";
    case ViewerWorkspacePersistenceStatus::InvalidArgument: return "invalid-argument";
    case ViewerWorkspacePersistenceStatus::EmptyInput: return "empty-input";
    case ViewerWorkspacePersistenceStatus::CapacityExceeded: return "capacity-exceeded";
    case ViewerWorkspacePersistenceStatus::Malformed: return "malformed";
    case ViewerWorkspacePersistenceStatus::UnsupportedVersion: return "unsupported-version";
    case ViewerWorkspacePersistenceStatus::IntegrityMismatch: return "integrity-mismatch";
    case ViewerWorkspacePersistenceStatus::UnknownRecord: return "unknown-record";
    case ViewerWorkspacePersistenceStatus::ValidationFailed: return "validation-failed";
    case ViewerWorkspacePersistenceStatus::AllocationFailure: return "allocation-failure";
    default: return "unknown";
  }
}

struct CandidateRead {
  bool present = false;
  bool readable = false;
  std::string bytes;
  std::string diagnostic;
};

CandidateRead readCandidate(const std::filesystem::path& path) noexcept {
  CandidateRead result{};
  try {
    std::error_code error;
    const auto status = std::filesystem::symlink_status(path, error);
    if (error) {
      if (isNotFoundError(error)) return result;
      result.present = true;
      result.diagnostic = "status failed: " + error.message();
      return result;
    }
    if (status.type() == std::filesystem::file_type::not_found) return result;
    result.present = true;
    if (status.type() != std::filesystem::file_type::regular) {
      result.diagnostic = "not a regular file";
      return result;
    }
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error) {
      result.diagnostic = "size failed: " + error.message();
      return result;
    }
    if (size > kViewerWorkspacePersistenceMaxDocumentBytes) {
      result.diagnostic = "document exceeds byte bound";
      return result;
    }
    const std::size_t boundedSize = static_cast<std::size_t>(size);
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
      result.diagnostic = "open failed";
      return result;
    }
    result.bytes.assign(boundedSize, '\0');
    if (boundedSize > 0u) {
      input.read(result.bytes.data(), static_cast<std::streamsize>(boundedSize));
      if (input.gcount() != static_cast<std::streamsize>(boundedSize)) {
        result.bytes.clear();
        result.diagnostic = "short read";
        return result;
      }
    }
    // Detect a growth race without ever retaining more than the codec bound.
    char extra = 0;
    input.read(&extra, 1);
    if (input.gcount() > 0) {
      result.bytes.clear();
      result.diagnostic = "document grew beyond byte bound";
      return result;
    }
    if (input.bad()) {
      result.bytes.clear();
      result.diagnostic = "read failed";
      return result;
    }
    result.readable = true;
    return result;
  } catch (const std::bad_alloc&) {
    result.present = true;
    result.diagnostic = "allocation failure";
    return result;
  } catch (...) {
    result.present = true;
    result.diagnostic = "filesystem read failed";
    return result;
  }
}

struct Candidate {
  const char* label = nullptr;
  std::filesystem::path path;
  ViewerWorkspaceStoreLoadStatus acceptedStatus =
      ViewerWorkspaceStoreLoadStatus::AcceptedPrimary;
  bool repairSuggested = false;
};

bool samePath(const std::filesystem::path& left,
              const std::filesystem::path& right) noexcept {
  try {
    return left.string() == right.string();
  } catch (...) {
    return false;
  }
}

std::uint64_t storeToken() noexcept {
  static std::atomic<std::uint64_t> counter{0u};
  const std::uint64_t sequence = counter.fetch_add(1u, std::memory_order_relaxed) + 1u;
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::uint64_t clockBits = static_cast<std::uint64_t>(now);
  const std::uint64_t threadBits = static_cast<std::uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  return clockBits ^ (threadBits + 0x9e3779b97f4a7c15ull + (sequence << 6u) +
                     (sequence >> 2u));
}

bool appendTokenPath(const std::filesystem::path& base, std::string_view middle,
                     std::uint64_t token, std::size_t attempt,
                     std::filesystem::path* output) noexcept {
  if (!output || middle.empty()) return false;
  std::ostringstream suffix;
  suffix << middle << std::hex << token;
  if (attempt > 0u) suffix << '.' << attempt;
  return pathForSuffix(base, suffix.str(), output);
}

bool createTempPath(const std::filesystem::path& primary,
                    std::filesystem::path* output) noexcept {
  if (!output) return false;
  const std::uint64_t token = storeToken();
  for (std::size_t attempt = 0u; attempt < kMaxTempAttempts; ++attempt) {
    std::filesystem::path candidate;
    if (!appendTokenPath(primary, ".tmp-", token, attempt, &candidate)) return false;
    bool exists = false;
    if (!pathExists(candidate, &exists, nullptr)) return false;
    if (!exists) {
      *output = std::move(candidate);
      return true;
    }
  }
  return false;
}

bool createUniqueSibling(const std::filesystem::path& base,
                         std::string_view middle,
                         std::filesystem::path* output) noexcept {
  if (!output) return false;
  const std::uint64_t token = storeToken();
  for (std::size_t attempt = 0u; attempt < kMaxTempAttempts; ++attempt) {
    std::filesystem::path candidate;
    if (!appendTokenPath(base, middle, token, attempt, &candidate)) return false;
    bool exists = false;
    if (!pathExists(candidate, &exists, nullptr)) return false;
    if (!exists) {
      *output = std::move(candidate);
      return true;
    }
  }
  return false;
}

bool createCorruptPath(const std::filesystem::path& primary,
                       std::filesystem::path* output) noexcept {
  if (!output) return false;
  std::filesystem::path canonical;
  if (!pathForSuffix(primary, ".corrupt", &canonical)) return false;
  bool exists = false;
  if (!pathExists(canonical, &exists, nullptr)) return false;
  if (!exists) {
    *output = std::move(canonical);
    return true;
  }
  return createUniqueSibling(primary, ".corrupt-", output);
}

bool removeExact(const std::filesystem::path& path, std::string* diagnostic) noexcept {
  std::error_code error;
  const bool removed = std::filesystem::remove(path, error);
  if (error) {
    appendDiagnostic(diagnostic, "owned cleanup failed: " + error.message());
    return false;
  }
  return removed || !std::filesystem::exists(path, error);
}

bool renameExact(const std::filesystem::path& from,
                 const std::filesystem::path& to,
                 std::string* diagnostic) noexcept {
  std::error_code error;
  std::filesystem::rename(from, to, error);
  if (error) {
    appendDiagnostic(diagnostic, "rename failed: " + error.message());
    return false;
  }
  return true;
}

bool validPrimaryPath(const ViewerWorkspaceStorePaths& paths,
                      std::filesystem::path* backup,
                      std::string* diagnostic) noexcept {
  std::string primaryString;
  if (!pathString(paths.primary, &primaryString) ||
      !paths.primary.has_filename()) {
    appendDiagnostic(diagnostic, "primary path is empty or exceeds bound");
    return false;
  }
  if (!pathForSuffix(paths.primary, ".bak", backup)) {
    appendDiagnostic(diagnostic, "backup path exceeds bound");
    return false;
  }
  if (!paths.legacy.empty()) {
    std::string legacyString;
    if (!pathString(paths.legacy, &legacyString) || !paths.legacy.has_filename()) {
      appendDiagnostic(diagnostic, "legacy path is invalid or exceeds bound");
      return false;
    }
  }
  return true;
}

bool writeTemp(const std::filesystem::path& path, std::string_view bytes,
               std::string* diagnostic) noexcept {
  try {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      appendDiagnostic(diagnostic, "temporary open failed");
      return false;
    }
    if (!bytes.empty()) {
      output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    output.flush();
    if (!output.good()) {
      appendDiagnostic(diagnostic, "temporary write failed");
      return false;
    }
    output.close();
    if (output.fail()) {
      appendDiagnostic(diagnostic, "temporary close failed");
      return false;
    }
    return true;
  } catch (...) {
    appendDiagnostic(diagnostic, "temporary write threw");
    return false;
  }
}

bool pathMissing(const std::filesystem::path& path) noexcept {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  return (isNotFoundError(error) ||
          (!error && status.type() == std::filesystem::file_type::not_found));
}

}  // namespace

ViewerWorkspaceStoreLoadResult loadViewerWorkspaceStore(
    const ViewerWorkspaceStorePaths& paths) noexcept {
  ViewerWorkspaceStoreLoadResult result{};
  try {
    std::filesystem::path backup;
    if (!validPrimaryPath(paths, &backup, &result.diagnostic)) return result;
    const ViewerWorkspaceDocument rawDefaults = defaultViewerWorkspaceDocument();
    ViewerWorkspaceDocument defaults{};
    if (!sanitiseViewerWorkspaceDocument(rawDefaults, &defaults) ||
        !validateViewerWorkspaceDocument(defaults)) {
      result.status = ViewerWorkspaceStoreLoadStatus::AllocationFailure;
      appendDiagnostic(&result.diagnostic, "validated defaults unavailable");
      return result;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(3u);
    candidates.push_back({"primary", paths.primary,
                          ViewerWorkspaceStoreLoadStatus::AcceptedPrimary, false});
    candidates.push_back({"backup", backup,
                          ViewerWorkspaceStoreLoadStatus::AcceptedBackup, true});
    if (!paths.legacy.empty() && !samePath(paths.legacy, paths.primary) &&
        !samePath(paths.legacy, backup)) {
      candidates.push_back({"legacy", paths.legacy,
                            ViewerWorkspaceStoreLoadStatus::AcceptedLegacy, true});
    }

    bool anyEvidence = false;
    for (const Candidate& candidate : candidates) {
      const CandidateRead read = readCandidate(candidate.path);
      if (!read.present) continue;
      anyEvidence = true;
      result.evidencePresent = true;
      if (!read.readable) {
        std::string detail = std::string(candidate.label) + " unreadable";
        if (!read.diagnostic.empty()) {
          detail += ": ";
          detail += read.diagnostic;
        }
        appendDiagnostic(&result.diagnostic, detail);
        continue;
      }
      const auto decoded = decodeViewerWorkspaceDocument(read.bytes, &defaults);
      if (!decoded.accepted() || !validateViewerWorkspaceDocument(decoded.document)) {
        std::string detail = std::string(candidate.label) + " rejected";
        detail += ": ";
        detail += persistenceStatusName(decoded.status);
        appendDiagnostic(&result.diagnostic, detail);
        continue;
      }
      result.status = candidate.acceptedStatus;
      result.document = decoded.document;
      result.repairSuggested = candidate.repairSuggested;
      appendDiagnostic(&result.diagnostic,
                       std::string("loaded ") + candidate.label);
      return result;
    }

    result.document = defaults;
    if (anyEvidence) {
      result.status = ViewerWorkspaceStoreLoadStatus::DefaultsRecovery;
      result.repairSuggested = false;
      appendDiagnostic(&result.diagnostic,
                       "using validated defaults; corrupt evidence preserved");
    } else {
      result.status = ViewerWorkspaceStoreLoadStatus::DefaultsMissing;
      result.repairSuggested = false;
      appendDiagnostic(&result.diagnostic, "no workspace candidates found");
    }
    return result;
  } catch (const std::bad_alloc&) {
    result.status = ViewerWorkspaceStoreLoadStatus::AllocationFailure;
    appendDiagnostic(&result.diagnostic, "allocation failure while loading workspace");
    return result;
  } catch (...) {
    result.status = ViewerWorkspaceStoreLoadStatus::AllocationFailure;
    appendDiagnostic(&result.diagnostic, "unexpected workspace load failure");
    return result;
  }
}

ViewerWorkspaceStoreSaveResult saveViewerWorkspaceStore(
    const ViewerWorkspaceStorePaths& paths,
    const ViewerWorkspaceDocument& document) noexcept {
  ViewerWorkspaceStoreSaveResult result{};
  std::filesystem::path backup;
  std::filesystem::path temporary;
  std::filesystem::path oldBackup;
  std::filesystem::path corrupt;
  bool temporaryOwned = false;
  bool oldBackupStashed = false;
  bool primaryMovedToBackup = false;
  bool primaryMovedToCorrupt = false;
  try {
    if (!validPrimaryPath(paths, &backup, &result.diagnostic)) return result;
    const auto encoded = encodeViewerWorkspaceV2(document);
    if (!encoded.accepted() || encoded.bytes.empty()) {
      result.status = ViewerWorkspaceStoreSaveStatus::EncodeFailed;
      appendDiagnostic(&result.diagnostic,
                       std::string("workspace encode failed: ") +
                           persistenceStatusName(encoded.status));
      return result;
    }
    const auto verify = decodeViewerWorkspaceDocument(encoded.bytes);
    if (!verify.accepted() || !validateViewerWorkspaceDocument(verify.document)) {
      result.status = ViewerWorkspaceStoreSaveStatus::TempVerificationFailed;
      appendDiagnostic(&result.diagnostic, "generated workspace failed decode verification");
      return result;
    }

    const std::filesystem::path parent = paths.primary.parent_path();
    if (!parent.empty()) {
      std::error_code directoryError;
      std::filesystem::create_directories(parent, directoryError);
      if (directoryError) {
        result.status = ViewerWorkspaceStoreSaveStatus::DirectoryCreateFailed;
        appendDiagnostic(&result.diagnostic,
                         "workspace directory creation failed: " +
                             directoryError.message());
        return result;
      }
    }
    if (!createTempPath(paths.primary, &temporary)) {
      result.status = ViewerWorkspaceStoreSaveStatus::TempCreateFailed;
      appendDiagnostic(&result.diagnostic, "could not allocate owned temporary path");
      return result;
    }
    temporaryOwned = true;
    if (!writeTemp(temporary, encoded.bytes, &result.diagnostic)) {
      result.status = ViewerWorkspaceStoreSaveStatus::TempWriteFailed;
      (void)removeExact(temporary, &result.diagnostic);
      return result;
    }
    const CandidateRead writtenTemp = readCandidate(temporary);
    if (!writtenTemp.readable || writtenTemp.bytes != encoded.bytes) {
      result.status = ViewerWorkspaceStoreSaveStatus::TempVerificationFailed;
      appendDiagnostic(&result.diagnostic,
                       "owned temporary bytes differ from encoded workspace");
      (void)removeExact(temporary, &result.diagnostic);
      return result;
    }
    const auto writtenDecode = decodeViewerWorkspaceDocument(writtenTemp.bytes);
    if (!writtenDecode.accepted() ||
        !validateViewerWorkspaceDocument(writtenDecode.document)) {
      result.status = ViewerWorkspaceStoreSaveStatus::TempVerificationFailed;
      appendDiagnostic(&result.diagnostic,
                       "owned temporary failed decode verification");
      (void)removeExact(temporary, &result.diagnostic);
      return result;
    }
    const CandidateRead primaryRead = readCandidate(paths.primary);
    const bool primaryExists = primaryRead.present;
    bool primaryValid = false;
    if (primaryRead.readable) {
      const auto decoded = decodeViewerWorkspaceDocument(primaryRead.bytes);
      primaryValid = decoded.accepted() && validateViewerWorkspaceDocument(decoded.document);
    }

    if (primaryExists && primaryValid) {
      bool backupExists = false;
      if (!pathExists(backup, &backupExists, &result.diagnostic)) {
        result.status = ViewerWorkspaceStoreSaveStatus::BackupRotationFailed;
        (void)removeExact(temporary, &result.diagnostic);
        return result;
      }
      if (backupExists) {
        if (!createUniqueSibling(backup, ".old-", &oldBackup) ||
            !renameExact(backup, oldBackup, &result.diagnostic)) {
          result.status = ViewerWorkspaceStoreSaveStatus::BackupRotationFailed;
          (void)removeExact(temporary, &result.diagnostic);
          return result;
        }
        oldBackupStashed = true;
      }
      if (!renameExact(paths.primary, backup, &result.diagnostic)) {
        if (oldBackupStashed) {
          (void)renameExact(oldBackup, backup, &result.diagnostic);
          oldBackupStashed = false;
        }
        result.status = ViewerWorkspaceStoreSaveStatus::BackupRotationFailed;
        (void)removeExact(temporary, &result.diagnostic);
        return result;
      }
      primaryMovedToBackup = true;
    } else if (primaryExists) {
      if (!createCorruptPath(paths.primary, &corrupt) ||
          !renameExact(paths.primary, corrupt, &result.diagnostic)) {
        result.status = ViewerWorkspaceStoreSaveStatus::CorruptPreservationFailed;
        (void)removeExact(temporary, &result.diagnostic);
        return result;
      }
      primaryMovedToCorrupt = true;
    }

    if (!renameExact(temporary, paths.primary, &result.diagnostic)) {
      bool rollbackOk = true;
      if (primaryMovedToBackup) {
        if (pathMissing(paths.primary)) {
          rollbackOk = renameExact(backup, paths.primary, &result.diagnostic);
        } else {
          rollbackOk = false;
        }
        if (oldBackupStashed) {
          if (!pathMissing(backup) ||
              !renameExact(oldBackup, backup, &result.diagnostic)) {
            rollbackOk = false;
          } else {
            oldBackupStashed = false;
          }
        }
      } else if (primaryMovedToCorrupt) {
        if (pathMissing(paths.primary)) {
          rollbackOk = renameExact(corrupt, paths.primary, &result.diagnostic);
        } else {
          rollbackOk = false;
        }
      }
      if (temporaryOwned) {
        (void)removeExact(temporary, &result.diagnostic);
        temporaryOwned = false;
      }
      result.status = rollbackOk ? ViewerWorkspaceStoreSaveStatus::PromotionFailed
                                  : ViewerWorkspaceStoreSaveStatus::RollbackFailed;
      return result;
    }
    temporaryOwned = false;
    result.changed = true;

    if (oldBackupStashed) {
      if (!removeExact(oldBackup, &result.diagnostic)) {
        result.status = ViewerWorkspaceStoreSaveStatus::CleanupFailed;
        return result;
      }
      oldBackupStashed = false;
    }
    result.status = ViewerWorkspaceStoreSaveStatus::Saved;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = ViewerWorkspaceStoreSaveStatus::AllocationFailure;
    appendDiagnostic(&result.diagnostic, "allocation failure while saving workspace");
  } catch (...) {
    result.status = ViewerWorkspaceStoreSaveStatus::AllocationFailure;
    appendDiagnostic(&result.diagnostic, "unexpected workspace save failure");
  }

  if (temporaryOwned) (void)removeExact(temporary, &result.diagnostic);
  // If an exception crossed the transaction after moving the old primary,
  // make one best-effort rollback.  Never remove a path that we did not create
  // in this call.
  if (primaryMovedToBackup && pathMissing(paths.primary)) {
    (void)renameExact(backup, paths.primary, &result.diagnostic);
  } else if (primaryMovedToCorrupt && pathMissing(paths.primary)) {
    (void)renameExact(corrupt, paths.primary, &result.diagnostic);
  }
  if (oldBackupStashed && pathMissing(backup)) {
    (void)renameExact(oldBackup, backup, &result.diagnostic);
  }
  return result;
}

}  // namespace ChromaspaceViewer
