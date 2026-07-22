#pragma once

#include "ChromaspaceViewerSession.h"
#include "ChromaspaceViewerWorkspace.h"

#include <cstdint>

namespace ChromaspaceViewer {

enum class ViewerCameraInteractionMode : uint8_t {
  None = 0,
  Orbit,
  Pan,
  Zoom,
};

enum class ViewerCameraInteractionStatus : uint8_t {
  Accepted = 0,
  NoChange,
  ReplayedInput,
  InvalidState,
  InvalidInput,
  InvalidScene,
  InvalidWorkspace,
  MissingWindow,
  InvalidCamera,
  RevisionOverflow,
};

struct ViewerCameraInteractionState {
  bool pointerCaptureActive = false;
  int pointerWindowId = -1;
  ViewerCameraInteractionMode pointerMode = ViewerCameraInteractionMode::None;
  ViewerSessionPointerButton pointerButton =
      ViewerSessionPointerButton::Primary;
  float lastPointerX = 0.0f;
  float lastPointerY = 0.0f;

  bool gestureCaptureActive = false;
  int gestureWindowId = -1;
  ViewerSessionGestureKind gestureKind = ViewerSessionGestureKind::Magnify;

  uint64_t lastInputSequence = 0u;
  uint64_t interactionRevision = 0u;
};

struct ViewerCameraInteractionRequest {
  const ViewerSessionReduceResult* input = nullptr;
  const ViewerSessionState* session = nullptr;
  const ViewerUiScene* scene = nullptr;
  const ViewerWorkspaceState* workspace = nullptr;
  // True only for an unconsumed PlotBody press authorized by ViewerController.
  bool authorizeCameraStart = false;
};

struct ViewerCameraInteractionResult {
  ViewerCameraInteractionStatus status =
      ViewerCameraInteractionStatus::InvalidInput;
  bool stateChanged = false;
  bool cameraChanged = false;
  int windowId = -1;
  CameraState camera{};
  uint64_t interactionRevision = 0u;

  bool accepted() const noexcept {
    return status == ViewerCameraInteractionStatus::Accepted ||
           status == ViewerCameraInteractionStatus::NoChange;
  }
};

// Reduces one already-validated session result. Capture is bounded POD state;
// no event queue, platform object, renderer resource, or clock crosses here.
// On any rejection the caller's interaction state is unchanged.
ViewerCameraInteractionResult reduceViewerCameraInteraction(
    ViewerCameraInteractionState* state,
    const ViewerCameraInteractionRequest& request) noexcept;

}  // namespace ChromaspaceViewer
