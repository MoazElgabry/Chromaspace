#pragma once

#include "ChromaspaceViewerController.h"
#include "ChromaspaceViewerDomain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceViewer {

// The workspace is intentionally bounded.  Input reduction is a live-path
// operation: it may move retained domain vectors, but it must not copy lasso
// points or allocate in proportion to retained workspace data.
constexpr std::size_t kViewerWorkspaceMaxWindows = 64u;
constexpr std::size_t kViewerWorkspaceMaxLassoStrokes = 256u;
constexpr std::size_t kViewerWorkspaceMaxLassoPoints = 16384u;
constexpr std::size_t kViewerWorkspaceMaxLassoPointsPerStroke = 4096u;
constexpr std::size_t kViewerWorkspaceMaxStringBytes = 512u;
constexpr std::size_t kViewerWorkspaceMaxSerializedLassoBytes = 524288u;
constexpr std::size_t kViewerWorkspaceMaxEffects = 32u;

enum class ViewerWorkspaceToolbarPanel : uint8_t {
  None = 0,
  MainMenu,
  AddPlot,
  LayoutPreset,
};

enum class ViewerWorkspaceEffectKind : uint8_t {
  ReleaseWindowResources = 0,
  DockSourceSignalForLasso,
  SnapPreviewUpdated,
  SnapCommitted,
  RefreshReinterpret,
  RefreshResample,
  PersistSuggested,
  MainMenuOpened,
  SlicingLassoChanged,
  SlicingDrawerChanged,
  ClearSourceLasso,
};

struct ViewerWorkspaceEffect {
  ViewerWorkspaceEffectKind kind = ViewerWorkspaceEffectKind::RefreshReinterpret;
  int windowId = -1;
  PlotWindowDragMode dragMode = PlotWindowDragMode::None;
  PlotWindowRectNorm rect{};
  float pointerX = 0.0f;
  float pointerY = 0.0f;
  bool enabled = false;
};

struct ViewerWorkspaceEffectBatch {
  std::array<ViewerWorkspaceEffect, kViewerWorkspaceMaxEffects> effects{};
  std::size_t count = 0u;
  // Internal reducer diagnostic.  It remains false on accepted batches and
  // lets callers distinguish a full effect arena from an invalid command.
  bool capacityExceeded = false;

  bool empty() const noexcept { return count == 0u; }
  const ViewerWorkspaceEffect& operator[](std::size_t index) const noexcept {
    return effects[index];
  }
  bool contains(ViewerWorkspaceEffectKind kind) const noexcept;
};

struct ViewerWorkspaceState {
  // Order is the compositing/stacking order: the last window is the readable
  // top/front window when equal-area stacking is resolved.
  std::vector<PlotWindowDomainState> windows;
  int focusedWindowId = -1;
  int nextWindowId = 1;

  std::string layoutPresetSelection = "Custom";
  std::string layoutPresetBeforeSolo;
  std::string layoutPresetNameInput;

  ViewerWorkspaceToolbarPanel activeToolbarPanel =
      ViewerWorkspaceToolbarPanel::None;
  float toolbarPanelAnchorX = 0.0f;
  float toolbarPanelAnchorY = 0.0f;

  bool windowDragActive = false;
  int windowDragWindowId = -1;
  PlotWindowDragMode windowDragMode = PlotWindowDragMode::None;
  float windowDragStartX = 0.0f;
  float windowDragStartY = 0.0f;
  PlotWindowRectNorm windowDragStartRect{};

  bool sourceLassoSubtractMode = false;
  bool sourceLassoHasSelection = false;
  bool sourceLassoGlobalHasSelection = false;
  bool sourceLassoSelectionsSynced = false;
  int sourceLassoTargetWindowId = -1;
  bool sourceLassoSessionActive = false;
  uint64_t sourceLassoRevision = 0u;
  std::vector<LassoStroke> sourceLassoStrokes;

  uint64_t revision = 1u;
};

enum class ViewerWorkspaceReduceStatus : uint8_t {
  InvalidState = 0,
  InvalidCommand,
  CapacityExceeded,
  RevisionOverflow,
  AllocationFailure,
  Accepted,
};

struct ViewerWorkspaceReduceResult {
  ViewerWorkspaceReduceStatus status = ViewerWorkspaceReduceStatus::InvalidState;
  bool changed = false;
  ViewerWorkspaceEffectBatch effects{};

  bool accepted() const noexcept {
    return status == ViewerWorkspaceReduceStatus::Accepted;
  }
  bool valid() const noexcept { return accepted(); }
};

// State validation is public so persistence/adapters can reject an incomplete
// snapshot before handing it to the reducer.  It never repairs or normalizes.
bool validateViewerWorkspaceState(const ViewerWorkspaceState& state) noexcept;

enum class ViewerWorkspaceCameraUpdateStatus : uint8_t {
  InvalidArgument = 0,
  InvalidState,
  MissingWindow,
  InvalidCamera,
  RevisionOverflow,
  Accepted,
};

struct ViewerWorkspaceCameraUpdateResult {
  ViewerWorkspaceCameraUpdateStatus status =
      ViewerWorkspaceCameraUpdateStatus::InvalidArgument;
  bool changed = false;
  uint64_t workspaceRevision = 0u;
  uint64_t windowRevision = 0u;

  bool accepted() const noexcept {
    return status == ViewerWorkspaceCameraUpdateStatus::Accepted;
  }
};

// Camera interaction has high event frequency but no variable-sized domain
// effects. Commit it through this narrow workspace-owned transaction rather
// than letting platform adapters mutate retained windows directly.
ViewerWorkspaceCameraUpdateResult updateViewerWorkspaceCamera(
    ViewerWorkspaceState* state,
    int windowId,
    const CameraState& camera) noexcept;

enum class ViewerWorkspaceLassoAppendStatus : uint8_t {
  InvalidArgument = 0,
  InvalidState,
  InvalidStroke,
  MissingTarget,
  CapacityExceeded,
  RevisionOverflow,
  AllocationFailure,
  Accepted,
};

struct ViewerWorkspaceLassoAppendResult {
  ViewerWorkspaceLassoAppendStatus status =
      ViewerWorkspaceLassoAppendStatus::InvalidArgument;
  bool changed = false;
  bool globalSelection = false;
  int targetWindowId = -1;
  uint64_t lassoRevision = 0u;
  uint64_t workspaceRevision = 0u;

  bool accepted() const noexcept {
    return status == ViewerWorkspaceLassoAppendStatus::Accepted;
  }
};

// Emits the one canonical v1 payload representation used by live commits and
// persistence. Output is replaced only on success.
bool encodeCanonicalViewerLassoData(
    uint64_t revision,
    const std::vector<LassoStroke>& strokes,
    std::string* output) noexcept;

// Commits one completed Source Signal stroke to the synchronized global
// selection or the existing target window. The bounded retained stroke list,
// derived payload, lasso revision, render revision, and workspace revision are
// one transaction; allocation or validation failure leaves semantic state
// unchanged.
ViewerWorkspaceLassoAppendResult appendViewerWorkspaceLassoStroke(
    ViewerWorkspaceState* state,
    const LassoStroke& stroke) noexcept;

enum class ViewerWorkspaceLassoSessionStatus : uint8_t {
  InvalidArgument = 0,
  InvalidState,
  MissingOwner,
  UnsupportedOwner,
  CapacityExceeded,
  RevisionOverflow,
  AllocationFailure,
  Accepted,
};

struct ViewerWorkspaceLassoSessionResult {
  ViewerWorkspaceLassoSessionStatus status =
      ViewerWorkspaceLassoSessionStatus::InvalidArgument;
  bool changed = false;
  bool sourceSurfaceCreated = false;
  int sourceSurfaceWindowId = -1;
  uint64_t workspaceRevision = 0u;

  bool accepted() const noexcept {
    return status == ViewerWorkspaceLassoSessionStatus::Accepted;
  }
};

// Realizes the portable SlicingLassoChanged effect. Enabling binds selection
// ownership to the initiating non-Source plot and ensures one temporary Source
// Signal drawing surface without changing focus. Disabling closes the session
// only after the final lasso-enabled owner is gone; retained selections remain.
ViewerWorkspaceLassoSessionResult updateViewerWorkspaceSourceLassoSession(
    ViewerWorkspaceState* state,
    int ownerWindowId,
    bool enabled) noexcept;

// The six runtime slicing flags are deliberately owned here rather than by a
// platform callback.  Invalid indices are no-ops and return false.
constexpr int kViewerWorkspaceSlicingVectorCount = 6;
bool slicingVectorEnabled(const ViewerRuntimeState& state, int index) noexcept;
bool setSlicingVectorEnabled(ViewerRuntimeState* state,
                             int index,
                             bool enabled) noexcept;
bool allSlicingVectorsEnabled(const ViewerRuntimeState& state) noexcept;

// Applies the entire bounded command batch transactionally.  On rejection,
// allocation failure, invalid state, or revision/capacity overflow, the
// caller-visible state and effects are unchanged.  The implementation uses a
// fixed-size preflight shadow and no allocation proportional to lasso data.
ViewerWorkspaceReduceResult reduceViewerWorkspace(
    ViewerWorkspaceState* state,
    const ViewerControllerCommandBatch& batch) noexcept;

}  // namespace ChromaspaceViewer
