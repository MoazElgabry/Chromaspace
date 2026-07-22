#pragma once

#include "ChromaspaceViewerLayout.h"
#include "ChromaspaceViewerSession.h"
#include "ChromaspaceViewerUiScene.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ChromaspaceViewer {

// The controller deliberately returns a bounded batch instead of retaining an
// event queue.  A platform callback consumes the batch synchronously, then the
// next callback can observe the updated capture state.
constexpr std::size_t kViewerControllerMaxCommands = 8u;

enum class ViewerControllerCommandKind : uint8_t {
  None = 0,
  FocusWindow,
  RequestCloseWindow,
  BeginWindowDrag,
  UpdateWindowDrag,
  EndWindowDrag,
  ToolbarMenu,
  ToolbarAddPlot,
  ToolbarLayoutPreset,
  ToggleSlicingDrawer,
  SetSlicingVector,
  SoloSlicingVector,
  ToggleAllSlicingVectors,
  ToggleSlicingLasso,
  SourceLassoAdd,
  SourceLassoSubtract,
  SourceLassoClear,
  // Keyboard intent is reduced by the workspace just like pointer commands;
  // no retained lasso data is touched by the platform adapter.
  SourceLassoUndo,
  SourceSignalRestore,
};

struct ViewerControllerCommand {
  ViewerControllerCommandKind kind = ViewerControllerCommandKind::None;
  int windowId = -1;
  int controlIndex = -1;
  PlotWindowDragMode dragMode = PlotWindowDragMode::None;
  PlotWindowRectNorm rect{};
  float pointerX = 0.0f;
  float pointerY = 0.0f;
  bool enabled = false;
  bool selected = false;
};

struct ViewerControllerCommandBatch {
  std::array<ViewerControllerCommand, kViewerControllerMaxCommands> commands{};
  std::size_t count = 0u;
  bool consumed = false;
  bool continueCamera = false;
  bool continueSourceLasso = false;

  bool empty() const noexcept { return count == 0u; }
  const ViewerControllerCommand& operator[](std::size_t index) const noexcept {
    return commands[index];
  }
};

struct ViewerControllerStateSnapshot {
  int focusedWindowId = -1;
  int hoveredWindowId = -1;
  PlotWindowDragMode hoveredDragMode = PlotWindowDragMode::None;
  bool windowDragActive = false;
  int windowDragWindowId = -1;
  PlotWindowDragMode windowDragMode = PlotWindowDragMode::None;
  float windowDragStartX = 0.0f;
  float windowDragStartY = 0.0f;
  PlotWindowRectNorm windowDragStartRect{};
  bool slicingPaintActive = false;
  int slicingPaintWindowId = -1;
  bool slicingPaintDesired = false;
  int slicingPaintLastIndex = -1;
  uint64_t lastConsumedInputSequence = 0u;
};

class ViewerController final {
 public:
  ViewerController() = default;

  // Validation and reconciliation happen before replacement.  On failure the
  // previous snapshot and interaction state remain untouched.
  bool publishScene(const ViewerUiScene& scene,
                   int initialFocusedWindowId = -1);

  // The reducer result and state cross this seam together.  A result is
  // consumed at most once by acceptedInput.sequence; rejected or replayed
  // results produce an empty batch and cannot mutate the controller.
  ViewerControllerCommandBatch consume(
      const ViewerSessionReduceResult& result,
      const ViewerSessionState& session) noexcept;

  // A legacy menu or host overlay can take ownership between callbacks.  The
  // platform adapter must explicitly reconcile the controller before routing
  // that input elsewhere; this clears transient hover/capture state without
  // touching the immutable scene or the session sequence.
  void cancelInteractions() noexcept;

  const ViewerUiScene& scene() const noexcept { return scene_; }
  bool hasScene() const noexcept { return hasScene_; }
  ViewerControllerStateSnapshot state() const noexcept;

 private:
  ViewerUiScene scene_{};
  bool hasScene_ = false;
  int focusedWindowId_ = -1;
  int hoveredWindowId_ = -1;
  PlotWindowDragMode hoveredDragMode_ = PlotWindowDragMode::None;

  bool windowDragActive_ = false;
  int windowDragWindowId_ = -1;
  PlotWindowDragMode windowDragMode_ = PlotWindowDragMode::None;
  float windowDragStartX_ = 0.0f;
  float windowDragStartY_ = 0.0f;
  PlotWindowRectNorm windowDragStartRect_{};

  bool slicingPaintActive_ = false;
  int slicingPaintWindowId_ = -1;
  bool slicingPaintDesired_ = false;
  int slicingPaintLastIndex_ = -1;
  uint64_t lastConsumedInputSequence_ = 0u;
};

}  // namespace ChromaspaceViewer
