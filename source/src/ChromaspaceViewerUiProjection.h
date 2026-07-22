#pragma once

#include "ChromaspaceViewerController.h"
#include "ChromaspaceViewerWorkspace.h"

#include <cstdint>
#include <vector>

namespace ChromaspaceViewer {

enum class ViewerUiProjectionStatus : uint8_t {
  Ready = 0,
  InvalidArgument,
  InvalidFramePlan,
  InvalidWorkspace,
  CapacityExceeded,
  DuplicatePlanWindowId,
  DuplicateWorkspaceWindowId,
  DuplicateWindowFactsId,
  WindowCountMismatch,
  MissingWorkspaceWindow,
  PlotModelMismatch,
  MissingWindowFacts,
  InvalidFacts,
  AllocationFailure,
};

struct ViewerUiProjectionSourceFacts {
  bool available = false;
  int sourceWidth = 0;
  int sourceHeight = 0;
  int displayWidth = 0;
  int displayHeight = 0;
};

struct ViewerUiProjectionWindowFacts {
  int windowId = -1;
  ViewerUiTitleMetrics titleMetrics{};
  float slicingAnimationProgress = 0.0f;
};

// A complete, immutable snapshot of the facts needed to project portable UI
// intent for one frame. Platform clocks, renderer objects, and resource
// handles deliberately stay outside this interface.
struct ViewerUiProjectionRequest {
  const ViewerFramePlan* framePlan = nullptr;
  const ViewerWorkspaceState* workspace = nullptr;
  ViewerControllerStateSnapshot controller{};

  bool showWorkspaceButtons = true;
  bool showSliceButtonInPlotWindows = true;
  float textScale = 1.0f;
  int layoutIndex = 0;

  bool hasPointer = false;
  float pointerX = 0.0f;
  float pointerY = 0.0f;

  ViewerUiProjectionSourceFacts source{};
  std::vector<ViewerUiProjectionWindowFacts> windows;
};

struct ViewerUiProjectionResult {
  ViewerUiProjectionStatus status = ViewerUiProjectionStatus::InvalidArgument;
  ViewerUiSceneInput input{};

  bool ready() const noexcept {
    return status == ViewerUiProjectionStatus::Ready;
  }
};

// Builds into temporary storage and publishes input only on Ready. The output
// follows frame-plan order, while every cross-domain join is by windowId.
ViewerUiProjectionResult projectViewerUi(
    const ViewerUiProjectionRequest& request) noexcept;

}  // namespace ChromaspaceViewer
