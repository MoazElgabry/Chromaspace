#pragma once

#include "ChromaspaceViewerWorkspace.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ChromaspaceViewer {

// Persistence is deliberately bounded independently of the live reducer so a
// malformed file cannot make parsing proportional to untrusted input.
constexpr std::size_t kViewerWorkspacePersistenceMaxDocumentBytes = 8u * 1024u * 1024u;
constexpr std::size_t kViewerWorkspacePersistenceMaxLines =
    kViewerWorkspaceMaxWindows + 4u;
constexpr std::size_t kViewerWorkspacePersistenceMaxRuntimeStringBytes =
    kViewerWorkspaceMaxStringBytes;
constexpr std::size_t kViewerWorkspacePersistenceMaxLassoBytes =
    kViewerWorkspaceMaxSerializedLassoBytes;
constexpr uint32_t kViewerWorkspacePersistenceVersion = 2u;
constexpr std::string_view kViewerWorkspacePersistenceSchema =
    "chromaspace_viewer_workspace";
constexpr std::string_view kViewerWorkspacePersistenceV1Type =
    "chromaspace_viewer_workspace_v1";

struct ViewerWorkspacePresentationPreferences {
  bool showWorkspaceButtons = true;
  bool showSliceButtonInPlotWindows = true;
  int viewerFontSize = 0;
  int windowWidth = 720;
  int windowHeight = 600;
  int windowPosX = 0;
  int windowPosY = 0;
  bool windowPositionValid = false;
  int activeStandardLayoutIndex = 0;
  float workspaceTopNorm = 0.0f;
};

struct ViewerWorkspaceDocument {
  ViewerWorkspaceState workspace{};
  ViewerWorkspacePresentationPreferences presentation{};
};

enum class ViewerWorkspacePersistenceStatus : uint8_t {
  Accepted = 0,
  InvalidArgument,
  EmptyInput,
  CapacityExceeded,
  Malformed,
  UnsupportedVersion,
  IntegrityMismatch,
  UnknownRecord,
  ValidationFailed,
  AllocationFailure,
};

struct ViewerWorkspacePersistenceEncodeResult {
  ViewerWorkspacePersistenceStatus status =
      ViewerWorkspacePersistenceStatus::InvalidArgument;
  std::string bytes;

  bool accepted() const noexcept {
    return status == ViewerWorkspacePersistenceStatus::Accepted;
  }
};

struct ViewerWorkspacePersistenceDecodeResult {
  ViewerWorkspacePersistenceStatus status =
      ViewerWorkspacePersistenceStatus::InvalidArgument;
  ViewerWorkspaceDocument document{};

  bool accepted() const noexcept {
    return status == ViewerWorkspacePersistenceStatus::Accepted;
  }
};

// Explicit defaults are shared by the production adapter, the native canary,
// and tests.  A normal return is a valid, transient-sanitised v2 snapshot;
// an allocation failure returns an invalid empty sentinel that callers must
// validate before use.
ViewerWorkspaceDocument defaultViewerWorkspaceDocument() noexcept;

bool validateViewerWorkspaceDocument(
    const ViewerWorkspaceDocument& document) noexcept;

// Sanitise the non-durable UI/session fields and validate the resulting
// durable document.  The input is never modified on failure.
bool sanitiseViewerWorkspaceDocument(
    const ViewerWorkspaceDocument& input,
    ViewerWorkspaceDocument* output) noexcept;

ViewerWorkspacePersistenceEncodeResult encodeViewerWorkspaceV2(
    const ViewerWorkspaceDocument& document) noexcept;

// Decodes canonical v2 bytes or the legacy viewer_workspace_v1.jsonl format.
// Parsing, migration, repair, and validation happen in a temporary document;
// output is unchanged on every failure.
ViewerWorkspacePersistenceDecodeResult decodeViewerWorkspaceDocument(
    std::string_view bytes,
    const ViewerWorkspaceDocument* defaults = nullptr) noexcept;

}  // namespace ChromaspaceViewer
