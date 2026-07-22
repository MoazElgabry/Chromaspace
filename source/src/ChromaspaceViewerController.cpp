#include "ChromaspaceViewerController.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace ChromaspaceViewer {
namespace {

bool finite(float value) noexcept { return std::isfinite(value); }

bool finiteRect(const ScreenRect& rect) noexcept {
  return finite(rect.x0) && finite(rect.y0) && finite(rect.x1) &&
         finite(rect.y1) && rect.x1 >= rect.x0 && rect.y1 >= rect.y0;
}

bool finiteNormalizedRect(const PlotWindowRectNorm& rect) noexcept {
  return finite(rect.x) && finite(rect.y) && finite(rect.w) &&
         finite(rect.h) && rect.x >= 0.0f && rect.y >= 0.0f && rect.w > 0.0f &&
         rect.h > 0.0f && rect.x + rect.w <= 1.00001f &&
         rect.y + rect.h <= 1.00001f;
}

bool finiteColor(const ViewerUiColor& color) noexcept {
  return finite(color.r) && finite(color.g) && finite(color.b) &&
         finite(color.a);
}

bool validScene(const ViewerUiScene& scene) noexcept {
  const auto present = [&scene](int windowId) noexcept {
    if (windowId <= 0) return false;
    for (const ViewerUiWindowScene& window : scene.windows) {
      if (window.windowId == windowId) return true;
    }
    return false;
  };
  if (!scene.ready() || scene.geometry.windowWidth <= 0 ||
      scene.geometry.windowHeight <= 0 || scene.geometry.framebufferWidth <= 0 ||
      scene.geometry.framebufferHeight <= 0 ||
      !finite(scene.geometry.scaleX) || !finite(scene.geometry.scaleY) ||
      scene.geometry.scaleX <= 0.0f || scene.geometry.scaleY <= 0.0f ||
      !finite(scene.geometry.reservedLeftPixels) ||
      scene.geometry.reservedLeftPixels < 0.0f ||
      scene.geometry.reservedLeftPixels >
          static_cast<float>(scene.geometry.windowWidth)) {
    return false;
  }
  for (std::size_t i = 0; i < scene.windows.size(); ++i) {
    const ViewerUiWindowScene& window = scene.windows[i];
    if (window.windowId <= 0 || !finiteRect(window.rect) ||
        !finiteRect(window.contentRect) ||
        !finiteNormalizedRect(window.normalizedRect)) {
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (scene.windows[j].windowId == window.windowId) return false;
    }
    if (window.primitiveBegin > scene.primitives.size() ||
        window.primitiveCount >
            scene.primitives.size() - window.primitiveBegin ||
        window.textBegin > scene.texts.size() ||
        window.textCount > scene.texts.size() - window.textBegin ||
        window.vectorBegin > scene.vectors.size() ||
        window.vectorCount > scene.vectors.size() - window.vectorBegin ||
        window.hitBegin > scene.hits.size() ||
        window.hitCount > scene.hits.size() - window.hitBegin) {
      return false;
    }
    for (std::size_t index = window.primitiveBegin;
         index < window.primitiveBegin + window.primitiveCount; ++index) {
      if (scene.primitives[index].windowId != window.windowId) return false;
    }
    for (std::size_t index = window.textBegin;
         index < window.textBegin + window.textCount; ++index) {
      if (scene.texts[index].windowId != window.windowId) return false;
    }
    for (std::size_t index = window.vectorBegin;
         index < window.vectorBegin + window.vectorCount; ++index) {
      if (scene.vectors[index].windowId != window.windowId) return false;
    }
    for (std::size_t index = window.hitBegin;
         index < window.hitBegin + window.hitCount; ++index) {
      if (scene.hits[index].windowId != window.windowId) return false;
    }
  }
  for (const ViewerUiSolidRect& primitive : scene.primitives) {
    if (!finiteRect(primitive.rect) || !finiteColor(primitive.color) ||
        (primitive.windowId > 0 && !present(primitive.windowId)) ||
        primitive.windowId == 0) {
      return false;
    }
  }
  for (const ViewerUiVectorVertex& vertex : scene.vectors) {
    if (!finite(vertex.x) || !finite(vertex.y) || !finiteColor(vertex.color) ||
        vertex.windowId <= 0 || !present(vertex.windowId)) {
      return false;
    }
  }
  for (const ViewerUiTextIntent& text : scene.texts) {
    if (!finiteRect(text.bounds) || !finite(text.originX) ||
        !finite(text.originY) || !finite(text.maxWidth) ||
        !finite(text.scale) || !finiteColor(text.color) || text.scale <= 0.0f ||
        text.windowId == 0 ||
        (text.windowId > 0 && !present(text.windowId))) {
      return false;
    }
  }
  for (const ViewerUiHitRegion& hit : scene.hits) {
    if (!finiteRect(hit.rect) || !finiteNormalizedRect(hit.normalizedRect) ||
        hit.windowId == 0 ||
        (hit.windowId > 0 && !present(hit.windowId))) {
      return false;
    }
  }
  return true;
}

bool windowPresent(const ViewerUiScene& scene, int windowId) noexcept {
  if (windowId < 0) return false;
  for (const ViewerUiWindowScene& window : scene.windows) {
    if (window.windowId == windowId) return true;
  }
  return false;
}

const ViewerUiWindowScene* sceneWindow(const ViewerUiScene& scene,
                                       int windowId) noexcept {
  for (const ViewerUiWindowScene& window : scene.windows) {
    if (window.windowId == windowId) return &window;
  }
  return nullptr;
}

bool validInputPoint(const ViewerSessionState& session) noexcept {
  return std::isfinite(session.pointerX) && std::isfinite(session.pointerY);
}

bool isPlotControl(ViewerUiControlKind control) noexcept {
  switch (control) {
    case ViewerUiControlKind::SlicingQuickToggle:
    case ViewerUiControlKind::SlicingVector:
    case ViewerUiControlKind::SlicingLasso:
    case ViewerUiControlKind::SourceLassoAdd:
    case ViewerUiControlKind::SourceLassoSubtract:
    case ViewerUiControlKind::SourceLassoClear:
    case ViewerUiControlKind::SourceSignalRestore:
      return true;
    default:
      return false;
  }
}

bool isPrimary(ViewerSessionPointerButton button) noexcept {
  return button == ViewerSessionPointerButton::Primary;
}

bool isSecondaryOrMiddle(ViewerSessionPointerButton button) noexcept {
  return button == ViewerSessionPointerButton::Secondary ||
         button == ViewerSessionPointerButton::Middle;
}

bool isSourceLassoUndoKey(const ViewerSessionTransientInput& input) noexcept {
  if (input.kind != ViewerSessionInputKind::Key ||
      input.key != ViewerSessionKey::Z || !input.pressed || input.repeat) {
    return false;
  }
  const ViewerSessionModifierMask disallowed =
      kViewerSessionModifierShift | kViewerSessionModifierAlt;
  if ((input.modifiers & disallowed) != 0u) return false;
  return (input.modifiers & (kViewerSessionModifierControl |
                             kViewerSessionModifierSuper)) != 0u;
}

void clearCaptures(int* windowDragWindowId,
                   PlotWindowDragMode* windowDragMode,
                   bool* windowDragActive,
                   int* slicingWindowId,
                   int* slicingLastIndex,
                   bool* slicingActive) noexcept {
  if (windowDragWindowId) *windowDragWindowId = -1;
  if (windowDragMode) *windowDragMode = PlotWindowDragMode::None;
  if (windowDragActive) *windowDragActive = false;
  if (slicingWindowId) *slicingWindowId = -1;
  if (slicingLastIndex) *slicingLastIndex = -1;
  if (slicingActive) *slicingActive = false;
}

}  // namespace

ViewerControllerStateSnapshot ViewerController::state() const noexcept {
  ViewerControllerStateSnapshot snapshot{};
  snapshot.focusedWindowId = focusedWindowId_;
  snapshot.hoveredWindowId = hoveredWindowId_;
  snapshot.hoveredDragMode = hoveredDragMode_;
  snapshot.windowDragActive = windowDragActive_;
  snapshot.windowDragWindowId = windowDragWindowId_;
  snapshot.windowDragMode = windowDragMode_;
  snapshot.windowDragStartX = windowDragStartX_;
  snapshot.windowDragStartY = windowDragStartY_;
  snapshot.windowDragStartRect = windowDragStartRect_;
  snapshot.slicingPaintActive = slicingPaintActive_;
  snapshot.slicingPaintWindowId = slicingPaintWindowId_;
  snapshot.slicingPaintDesired = slicingPaintDesired_;
  snapshot.slicingPaintLastIndex = slicingPaintLastIndex_;
  snapshot.lastConsumedInputSequence = lastConsumedInputSequence_;
  return snapshot;
}

void ViewerController::cancelInteractions() noexcept {
  clearCaptures(&windowDragWindowId_, &windowDragMode_,
                &windowDragActive_, &slicingPaintWindowId_,
                &slicingPaintLastIndex_, &slicingPaintActive_);
  hoveredWindowId_ = -1;
  hoveredDragMode_ = PlotWindowDragMode::None;
}

bool ViewerController::publishScene(const ViewerUiScene& scene,
                                    int initialFocusedWindowId) {
  if (!validScene(scene)) return false;
  try {
    ViewerUiScene next = scene;
    const bool firstPublication = !hasScene_;
    if (firstPublication && initialFocusedWindowId != -1 &&
        (initialFocusedWindowId <= 0 ||
         !windowPresent(next, initialFocusedWindowId))) {
      return false;
    }
    int nextFocused = firstPublication
                          ? (initialFocusedWindowId > 0
                                 ? initialFocusedWindowId
                                 : -1)
                          : (windowPresent(next, focusedWindowId_)
                                 ? focusedWindowId_
                                 : (initialFocusedWindowId > 0 &&
                                            windowPresent(next,
                                                         initialFocusedWindowId)
                                        ? initialFocusedWindowId
                                        : -1));
    int nextHovered = windowPresent(next, hoveredWindowId_) ? hoveredWindowId_ : -1;
    PlotWindowDragMode nextHoveredDrag =
        nextHovered >= 0 ? hoveredDragMode_ : PlotWindowDragMode::None;
    bool nextWindowDrag = windowDragActive_ &&
                          windowPresent(next, windowDragWindowId_);
    bool nextSlicingPaint = slicingPaintActive_ &&
                            windowPresent(next, slicingPaintWindowId_);
    if (!nextWindowDrag) {
      nextWindowDrag = false;
    }
    if (!nextSlicingPaint) {
      nextSlicingPaint = false;
    }
    scene_ = std::move(next);
    hasScene_ = true;
    focusedWindowId_ = nextFocused;
    hoveredWindowId_ = nextHovered;
    hoveredDragMode_ = nextHoveredDrag;
    if (!nextWindowDrag) {
      clearCaptures(&windowDragWindowId_, &windowDragMode_,
                    &windowDragActive_, nullptr, nullptr, nullptr);
    }
    if (!nextSlicingPaint) {
      clearCaptures(nullptr, nullptr, nullptr, &slicingPaintWindowId_,
                    &slicingPaintLastIndex_, &slicingPaintActive_);
    }
    return true;
  } catch (...) {
    return false;
  }
}

ViewerControllerCommandBatch ViewerController::consume(
    const ViewerSessionReduceResult& result,
    const ViewerSessionState& session) noexcept {
  ViewerControllerCommandBatch batch{};
  if (!hasScene_ || !result.accepted()) return batch;

  if (!result.inputAccepted) {
    if (result.cancelInteractions) cancelInteractions();
    return batch;
  }
  const uint64_t sequence = result.acceptedInput.sequence;
  if (sequence == 0u || sequence <= lastConsumedInputSequence_ ||
      session.lastInput.sequence != sequence) {
    return batch;
  }
  if (result.acceptedInput.kind == ViewerSessionInputKind::Cancelled) {
    cancelInteractions();
    lastConsumedInputSequence_ = sequence;
    return batch;
  }
  // Keyboard intent is portable and does not have a meaningful pointer
  // location.  Handle the one semantic shortcut before the pointer-only hit
  // path so an invalid/stale pointer cannot suppress Cmd/Ctrl+Z.
  if (result.acceptedInput.kind == ViewerSessionInputKind::Key) {
    if (isSourceLassoUndoKey(result.acceptedInput)) {
      if (batch.count < batch.commands.size()) {
        ViewerControllerCommand& command = batch.commands[batch.count++];
        command.kind = ViewerControllerCommandKind::SourceLassoUndo;
        command.windowId = -1;
        batch.consumed = true;
      }
    }
    lastConsumedInputSequence_ = sequence;
    return batch;
  }
  if (!validInputPoint(session) &&
      result.acceptedInput.kind != ViewerSessionInputKind::PointerLeft) {
    return batch;
  }

  auto append = [&batch](ViewerControllerCommandKind kind,
                         int windowId,
                         int controlIndex,
                         PlotWindowDragMode dragMode,
                         const PlotWindowRectNorm& rect,
                         float pointerX,
                         float pointerY,
                         bool enabled,
                         bool selected) noexcept {
    if (batch.count >= batch.commands.size()) return false;
    ViewerControllerCommand& command = batch.commands[batch.count++];
    command.kind = kind;
    command.windowId = windowId;
    command.controlIndex = controlIndex;
    command.dragMode = dragMode;
    command.rect = rect;
    command.pointerX = pointerX;
    command.pointerY = pointerY;
    command.enabled = enabled;
    command.selected = selected;
    return true;
  };

  const float pointerX = static_cast<float>(session.pointerX);
  const float pointerY = static_cast<float>(session.pointerY);
  const ViewerSessionInputKind kind = result.acceptedInput.kind;

  if (kind == ViewerSessionInputKind::PointerLeft) {
    hoveredWindowId_ = -1;
    hoveredDragMode_ = PlotWindowDragMode::None;
    lastConsumedInputSequence_ = sequence;
    return batch;
  }

  const ViewerUiHitResult hit = viewerUiHitTest(scene_, pointerX, pointerY);
  if (kind == ViewerSessionInputKind::PointerEntered ||
      kind == ViewerSessionInputKind::PointerMoved ||
      kind == ViewerSessionInputKind::PointerButton) {
    hoveredWindowId_ = hit.windowId;
    hoveredDragMode_ = hit.control == ViewerUiControlKind::PlotBody
                           ? hit.dragMode
                           : PlotWindowDragMode::None;
  }

  if (kind == ViewerSessionInputKind::PointerMoved) {
    if (windowDragActive_) {
      const float availableWidth = std::max(
          1.0f,
          static_cast<float>(scene_.geometry.windowWidth) -
              scene_.geometry.reservedLeftPixels);
      const float dx = (pointerX - windowDragStartX_) / availableWidth;
      const float dy = (pointerY - windowDragStartY_) /
                       static_cast<float>(std::max(1, scene_.geometry.windowHeight));
      const PlotWindowRectNorm rect = applyPlotWindowDrag(
          {windowDragStartRect_, windowDragMode_, dx, dy,
           workspaceGeometry(scene_.toolbar.visible,
                             scene_.geometry.windowHeight),
           plotWindowMinNormWidth(scene_.geometry.windowWidth),
           plotWindowMinNormHeight(scene_.geometry.windowHeight)});
      if (finiteNormalizedRect(rect)) {
        (void)append(ViewerControllerCommandKind::UpdateWindowDrag,
                     windowDragWindowId_, -1, windowDragMode_, rect, pointerX,
                     pointerY, true, false);
        batch.consumed = true;
      }
      lastConsumedInputSequence_ = sequence;
      return batch;
    }
    if (slicingPaintActive_) {
      if (hit.windowId == slicingPaintWindowId_ &&
          hit.control == ViewerUiControlKind::SlicingVector &&
          hit.controlIndex >= 0 && hit.controlIndex <
              static_cast<int>(kViewerUiSlicingVectorCount) && hit.actionable) {
        if (hit.controlIndex != slicingPaintLastIndex_) {
          (void)append(ViewerControllerCommandKind::SetSlicingVector,
                       slicingPaintWindowId_, hit.controlIndex,
                       PlotWindowDragMode::None, {}, pointerX, pointerY,
                       slicingPaintDesired_, !slicingPaintDesired_);
          slicingPaintLastIndex_ = hit.controlIndex;
        }
      }
      batch.consumed = true;
      lastConsumedInputSequence_ = sequence;
      return batch;
    }
    lastConsumedInputSequence_ = sequence;
    return batch;
  }

  if (kind != ViewerSessionInputKind::PointerButton) {
    lastConsumedInputSequence_ = sequence;
    return batch;
  }

  const ViewerSessionPointerButton button = result.acceptedInput.button;
  if (!result.acceptedInput.pressed) {
    if (isPrimary(button) && windowDragActive_) {
      const float availableWidth = std::max(
          1.0f,
          static_cast<float>(scene_.geometry.windowWidth) -
              scene_.geometry.reservedLeftPixels);
      const float dx = (pointerX - windowDragStartX_) / availableWidth;
      const float dy = (pointerY - windowDragStartY_) /
                       static_cast<float>(std::max(1, scene_.geometry.windowHeight));
      const PlotWindowRectNorm rect = applyPlotWindowDrag(
          {windowDragStartRect_, windowDragMode_, dx, dy,
           workspaceGeometry(scene_.toolbar.visible,
                             scene_.geometry.windowHeight),
           plotWindowMinNormWidth(scene_.geometry.windowWidth),
           plotWindowMinNormHeight(scene_.geometry.windowHeight)});
      if (finiteNormalizedRect(rect)) {
        (void)append(ViewerControllerCommandKind::EndWindowDrag,
                     windowDragWindowId_, -1, windowDragMode_, rect, pointerX,
                     pointerY, true, false);
      }
      clearCaptures(&windowDragWindowId_, &windowDragMode_,
                    &windowDragActive_, nullptr, nullptr, nullptr);
      batch.consumed = true;
    } else if (isPrimary(button) && slicingPaintActive_) {
      clearCaptures(nullptr, nullptr, nullptr, &slicingPaintWindowId_,
                    &slicingPaintLastIndex_, &slicingPaintActive_);
      batch.consumed = true;
    }
    lastConsumedInputSequence_ = sequence;
    return batch;
  }

  const bool primary = isPrimary(button);
  const bool secondaryOrMiddle = isSecondaryOrMiddle(button);
  if (hit.control == ViewerUiControlKind::ToolbarMenu && primary) {
    (void)append(ViewerControllerCommandKind::ToolbarMenu, -1, -1,
                 PlotWindowDragMode::None, {}, pointerX, pointerY, true, false);
    batch.consumed = true;
  } else if (hit.control == ViewerUiControlKind::ToolbarAddPlot && primary) {
    (void)append(ViewerControllerCommandKind::ToolbarAddPlot, -1, -1,
                 PlotWindowDragMode::None, {}, pointerX, pointerY, true, false);
    batch.consumed = true;
  } else if (hit.control == ViewerUiControlKind::ToolbarLayoutPreset && primary) {
    (void)append(ViewerControllerCommandKind::ToolbarLayoutPreset, -1, -1,
                 PlotWindowDragMode::None, {}, pointerX, pointerY, true, false);
    batch.consumed = true;
  } else if (hit.control == ViewerUiControlKind::PlotClose && primary &&
             hit.windowId >= 0) {
    (void)append(ViewerControllerCommandKind::RequestCloseWindow, hit.windowId,
                 -1, PlotWindowDragMode::None, {}, pointerX, pointerY, true,
                 false);
    batch.consumed = true;
  } else if (hit.control == ViewerUiControlKind::PlotBody &&
             (primary || secondaryOrMiddle) && hit.windowId >= 0) {
    focusedWindowId_ = hit.windowId;
    (void)append(ViewerControllerCommandKind::FocusWindow, hit.windowId, -1,
                 PlotWindowDragMode::None, {}, pointerX, pointerY, true, false);
    if (primary && hit.dragMode != PlotWindowDragMode::None) {
      const ViewerUiWindowScene* window = sceneWindow(scene_, hit.windowId);
      if (window != nullptr && finiteNormalizedRect(window->normalizedRect)) {
        windowDragActive_ = true;
        windowDragWindowId_ = hit.windowId;
        windowDragMode_ = hit.dragMode;
        windowDragStartX_ = pointerX;
        windowDragStartY_ = pointerY;
        windowDragStartRect_ = window->normalizedRect;
        (void)append(ViewerControllerCommandKind::BeginWindowDrag,
                     hit.windowId, -1, hit.dragMode, window->normalizedRect,
                     pointerX, pointerY, true, false);
        batch.consumed = true;
      }
    }
    if (!batch.consumed) {
      batch.continueCamera = true;
      batch.continueSourceLasso = primary;
    }
  } else if (primary && isPlotControl(hit.control) && hit.windowId >= 0 &&
             hit.actionable) {
    const bool sourceLassoControl =
        hit.control == ViewerUiControlKind::SourceLassoAdd ||
        hit.control == ViewerUiControlKind::SourceLassoSubtract ||
        hit.control == ViewerUiControlKind::SourceLassoClear ||
        hit.control == ViewerUiControlKind::SourceSignalRestore;
    if (!sourceLassoControl) {
      focusedWindowId_ = hit.windowId;
      (void)append(ViewerControllerCommandKind::FocusWindow, hit.windowId, -1,
                   PlotWindowDragMode::None, {}, pointerX, pointerY, true,
                   false);
    }
    ViewerControllerCommandKind command = ViewerControllerCommandKind::None;
    switch (hit.control) {
      case ViewerUiControlKind::SlicingQuickToggle:
        command = ViewerControllerCommandKind::ToggleSlicingDrawer;
        break;
      case ViewerUiControlKind::SlicingVector:
        if (result.acceptedInput.clickCount >= 2u) {
          command = ViewerControllerCommandKind::ToggleAllSlicingVectors;
        } else if ((result.acceptedInput.modifiers &
                    kViewerSessionModifierAlt) != 0u) {
          command = ViewerControllerCommandKind::SoloSlicingVector;
        } else {
          command = ViewerControllerCommandKind::SetSlicingVector;
        }
        break;
      case ViewerUiControlKind::SlicingLasso:
        command = ViewerControllerCommandKind::ToggleSlicingLasso;
        break;
      case ViewerUiControlKind::SourceLassoAdd:
        command = ViewerControllerCommandKind::SourceLassoAdd;
        break;
      case ViewerUiControlKind::SourceLassoSubtract:
        command = ViewerControllerCommandKind::SourceLassoSubtract;
        break;
      case ViewerUiControlKind::SourceLassoClear:
        command = ViewerControllerCommandKind::SourceLassoClear;
        break;
      case ViewerUiControlKind::SourceSignalRestore:
        command = ViewerControllerCommandKind::SourceSignalRestore;
        break;
      default:
        break;
    }
    const bool vectorPaint =
        hit.control == ViewerUiControlKind::SlicingVector &&
        result.acceptedInput.clickCount < 2u &&
        (result.acceptedInput.modifiers & kViewerSessionModifierAlt) == 0u;
    const bool desired = vectorPaint ? !hit.selected : false;
    const int commandWindowId =
        hit.control == ViewerUiControlKind::SourceSignalRestore
            ? hit.controlIndex
            : hit.windowId;
    (void)append(command, commandWindowId, hit.controlIndex,
                 PlotWindowDragMode::None, {}, pointerX, pointerY, desired,
                 hit.selected);
    if (vectorPaint &&
        hit.controlIndex >= 0 && hit.controlIndex <
            static_cast<int>(kViewerUiSlicingVectorCount)) {
      slicingPaintActive_ = true;
      slicingPaintWindowId_ = hit.windowId;
      slicingPaintDesired_ = desired;
      slicingPaintLastIndex_ = hit.controlIndex;
    }
    batch.consumed = true;
  } else if ((primary || secondaryOrMiddle) && hit.hit()) {
    batch.consumed = true;
  }

  lastConsumedInputSequence_ = sequence;
  return batch;
}

}  // namespace ChromaspaceViewer
