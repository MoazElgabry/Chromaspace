#pragma once

#include "ChromaspaceViewerSession.h"
#include "ChromaspaceViewerUiScene.h"
#include "ChromaspaceViewerWorkspace.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ChromaspaceViewer {

// Source Signal lasso capture is deliberately separate from the workspace.
// The workspace owns selection policy and the completed stroke is handed to
// appendViewerWorkspaceLassoStroke() by the caller after this reducer returns.
// Keeping the in-progress points here avoids allocating on pointer-move.
enum class ViewerLassoInteractionStatus : uint8_t {
  Accepted = 0,
  NoChange,
  ReplayedInput,
  InvalidState,
  InvalidInput,
  InvalidScene,
  InvalidWorkspace,
  NotAuthorized,
  SessionInactive,
  WrongControl,
  LetterboxRejected,
  InvalidSourceDimensions,
  MissingWindow,
  CaptureWindowChanged,
  CapacityExceeded,
  TooFewPoints,
  RevisionOverflow,
  AllocationFailure,
};

struct ViewerLassoSourceImageRect {
  ScreenRect rect{};
  bool valid = false;
};

// A fixed-size capture arena.  kViewerWorkspaceMaxLassoPointsPerStroke is
// the authoritative upper bound shared with retained workspace strokes.
struct ViewerLassoInteractionState {
  bool pointerCaptureActive = false;
  int pointerWindowId = -1;  // stable Source Signal surface window ID
  ViewerSessionPointerButton pointerButton =
      ViewerSessionPointerButton::Primary;
  bool strokeSubtract = false;
  double captureSourceWidth = 0.0;
  double captureSourceHeight = 0.0;
  std::array<LassoPointNorm, kViewerWorkspaceMaxLassoPointsPerStroke> points{};
  std::size_t pointCount = 0u;
  bool boundsValid = false;
  float minXNorm = 0.0f;
  float maxXNorm = 0.0f;
  float minYNorm = 0.0f;
  float maxYNorm = 0.0f;
  uint64_t lastInputSequence = 0u;
  uint64_t interactionRevision = 0u;
};

struct ViewerLassoInteractionRequest {
  const ViewerSessionReduceResult* input = nullptr;
  const ViewerSessionState* session = nullptr;
  const ViewerUiScene* scene = nullptr;
  const ViewerWorkspaceState* workspace = nullptr;

  // The controller must explicitly authorize a Source Signal PlotBody press.
  // A second, longer spelling is retained for adapters that prefer the
  // feature name in their call site; either true value is required to start.
  bool authorizeLassoStart = false;
  bool authorizeSourceLassoStart = false;

  // Dimensions of the source raster represented by the Source Signal surface.
  // They are intentionally request-owned because a source can change between
  // frames.  The reducer rejects non-finite or non-positive dimensions before
  // beginning or extending a capture.
  double sourceWidth = 0.0;
  double sourceHeight = 0.0;

  bool authorizedStart() const noexcept {
    return authorizeLassoStart || authorizeSourceLassoStart;
  }
};

struct ViewerLassoInteractionResult {
  ViewerLassoInteractionStatus status =
      ViewerLassoInteractionStatus::InvalidInput;
  bool stateChanged = false;
  bool pointAppended = false;
  bool strokeCompleted = false;
  bool strokeDiscarded = false;
  int windowId = -1;
  LassoStroke stroke{};  // populated only when strokeCompleted is true
  uint64_t interactionRevision = 0u;

  bool accepted() const noexcept {
    return status == ViewerLassoInteractionStatus::Accepted ||
           status == ViewerLassoInteractionStatus::NoChange ||
           status == ViewerLassoInteractionStatus::LetterboxRejected ||
           status == ViewerLassoInteractionStatus::TooFewPoints;
  }
};

// Computes the centered aspect-fit source image inside a Source Signal
// window's content rect.  Coordinates remain logical top-left/Y-down.
bool computeViewerLassoSourceImageRect(
    const ScreenRect& contentRect,
    double sourceWidth,
    double sourceHeight,
    ViewerLassoSourceImageRect* output) noexcept;

// Maps a logical top-left pointer to bottom-up normalized source coordinates.
// When clampOutside is false the pointer must lie in the image rect.  When it
// is true, points outside the image are clamped to its nearest edge.
bool mapViewerLassoPointerToSource(
    const ViewerLassoSourceImageRect& imageRect,
    double logicalX,
    double logicalY,
    bool clampOutside,
    LassoPointNorm* output) noexcept;

// Reduces one already-validated session result.  Invalid snapshots, replayed
// sequences, vanished capture windows, and invalid source dimensions reject
// atomically; an active capture is not silently redirected.  A too-short
// release is a deterministic discard that clears capture without emitting a
// stroke.  The returned stroke is the only allocation-producing operation.
ViewerLassoInteractionResult reduceViewerLassoInteraction(
    ViewerLassoInteractionState* state,
    const ViewerLassoInteractionRequest& request) noexcept;

enum class ViewerLassoOverlayStatus : uint8_t {
  Ready = 0,
  InvalidArgument,
  InvalidState,
  InvalidScene,
  InvalidWorkspace,
  InvalidSourceDimensions,
  MissingWindow,
  CapacityExceeded,
  AllocationFailure,
};

struct ViewerLassoOverlayResult {
  ViewerLassoOverlayStatus status = ViewerLassoOverlayStatus::InvalidArgument;
  std::size_t retainedSegments = 0u;
  std::size_t activeSegments = 0u;
  std::size_t appendedVertices = 0u;

  bool ready() const noexcept { return status == ViewerLassoOverlayStatus::Ready; }
};

// Appends triangle-list line geometry for the retained selection and active
// stroke to the portable scene vector stream. Selection scope is resolved from
// the workspace; normalized source coordinates are projected through the same
// aspect-fit transform used for hit mapping. The output vector is unchanged on
// failure.
ViewerLassoOverlayResult appendViewerLassoOverlay(
    const ViewerLassoInteractionState& interaction,
    const ViewerWorkspaceState& workspace,
    double sourceWidth,
    double sourceHeight,
    ViewerUiScene* scene) noexcept;

// Feature-explicit aliases make the seam easy to discover from Source Signal
// adapters while keeping the shorter camera-style names above.
using ViewerSourceLassoInteractionStatus = ViewerLassoInteractionStatus;
using ViewerSourceLassoSourceImageRect = ViewerLassoSourceImageRect;
using ViewerSourceLassoInteractionState = ViewerLassoInteractionState;
using ViewerSourceLassoInteractionRequest = ViewerLassoInteractionRequest;
using ViewerSourceLassoInteractionResult = ViewerLassoInteractionResult;

inline bool computeViewerSourceLassoImageRect(
    const ScreenRect& contentRect,
    double sourceWidth,
    double sourceHeight,
    ViewerSourceLassoSourceImageRect* output) noexcept {
  return computeViewerLassoSourceImageRect(contentRect, sourceWidth,
                                           sourceHeight, output);
}

inline bool mapViewerSourceLassoPointerToSource(
    const ViewerSourceLassoSourceImageRect& imageRect,
    double logicalX,
    double logicalY,
    bool clampOutside,
    LassoPointNorm* output) noexcept {
  return mapViewerLassoPointerToSource(imageRect, logicalX, logicalY,
                                       clampOutside, output);
}

inline ViewerSourceLassoInteractionResult reduceViewerSourceLassoInteraction(
    ViewerSourceLassoInteractionState* state,
    const ViewerSourceLassoInteractionRequest& request) noexcept {
  return reduceViewerLassoInteraction(state, request);
}

}  // namespace ChromaspaceViewer
