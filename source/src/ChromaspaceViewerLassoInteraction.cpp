#include "ChromaspaceViewerLassoInteraction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace ChromaspaceViewer {
namespace {

constexpr float kPointDistanceSquared = 1.0e-10f;

bool validSourceDimensions(double width, double height) noexcept;

bool finiteRect(const ScreenRect& rect) noexcept {
  return std::isfinite(rect.x0) && std::isfinite(rect.y0) &&
         std::isfinite(rect.x1) && std::isfinite(rect.y1) &&
         rect.x1 > rect.x0 && rect.y1 > rect.y0;
}

bool validPoint(const LassoPointNorm& point) noexcept {
  return std::isfinite(point.xNorm) && std::isfinite(point.yNorm) &&
         point.xNorm >= 0.0f && point.xNorm <= 1.0f &&
         point.yNorm >= 0.0f && point.yNorm <= 1.0f;
}

bool validState(const ViewerLassoInteractionState& state) noexcept {
  if (state.pointCount > state.points.size()) return false;
  if (state.pointerCaptureActive) {
    if (state.pointerWindowId <= 0 || state.pointCount == 0u ||
        state.pointerButton != ViewerSessionPointerButton::Primary ||
        !state.boundsValid ||
        !validSourceDimensions(state.captureSourceWidth,
                               state.captureSourceHeight)) {
      return false;
    }
  } else if (state.pointerWindowId != -1 || state.pointCount != 0u ||
             state.boundsValid || state.captureSourceWidth != 0.0 ||
             state.captureSourceHeight != 0.0) {
    return false;
  }
  for (std::size_t i = 0u; i < state.pointCount; ++i) {
    if (!validPoint(state.points[i])) return false;
  }
  if (state.boundsValid &&
      (!std::isfinite(state.minXNorm) || !std::isfinite(state.maxXNorm) ||
       !std::isfinite(state.minYNorm) || !std::isfinite(state.maxYNorm) ||
       state.minXNorm < 0.0f || state.maxXNorm > 1.0f ||
       state.minYNorm < 0.0f || state.maxYNorm > 1.0f ||
       state.minXNorm > state.maxXNorm ||
       state.minYNorm > state.maxYNorm)) {
    return false;
  }
  return true;
}

void clearCapture(ViewerLassoInteractionState* state) noexcept {
  state->pointerCaptureActive = false;
  state->pointerWindowId = -1;
  state->pointerButton = ViewerSessionPointerButton::Primary;
  state->strokeSubtract = false;
  state->captureSourceWidth = 0.0;
  state->captureSourceHeight = 0.0;
  state->pointCount = 0u;
  state->boundsValid = false;
  state->minXNorm = state->maxXNorm = 0.0f;
  state->minYNorm = state->maxYNorm = 0.0f;
}

bool bumpRevision(ViewerLassoInteractionState* state) noexcept {
  if (!state || state->interactionRevision ==
                    std::numeric_limits<uint64_t>::max()) {
    return false;
  }
  ++state->interactionRevision;
  return true;
}

const PlotWindowDomainState* workspaceWindow(
    const ViewerWorkspaceState& workspace,
    int windowId) noexcept {
  for (const auto& window : workspace.windows) {
    if (window.windowId == windowId) return &window;
  }
  return nullptr;
}

const ViewerUiWindowScene* sceneWindow(const ViewerUiScene& scene,
                                       int windowId) noexcept {
  for (const auto& window : scene.windows) {
    if (window.windowId == windowId) return &window;
  }
  return nullptr;
}

bool validSourceDimensions(double width, double height) noexcept {
  return std::isfinite(width) && std::isfinite(height) && width > 0.0 &&
         height > 0.0;
}

bool appendPoint(ViewerLassoInteractionState* state,
                 const LassoPointNorm& point,
                 bool* appended) noexcept {
  if (appended) *appended = false;
  if (!state || !validPoint(point)) return false;
  if (state->pointCount > 0u) {
    const auto& previous = state->points[state->pointCount - 1u];
    const float dx = point.xNorm - previous.xNorm;
    const float dy = point.yNorm - previous.yNorm;
    if (dx * dx + dy * dy <= kPointDistanceSquared) return true;
  }
  if (state->pointCount >= state->points.size()) return false;
  state->points[state->pointCount++] = point;
  if (!state->boundsValid) {
    state->boundsValid = true;
    state->minXNorm = state->maxXNorm = point.xNorm;
    state->minYNorm = state->maxYNorm = point.yNorm;
  } else {
    state->minXNorm = std::min(state->minXNorm, point.xNorm);
    state->maxXNorm = std::max(state->maxXNorm, point.xNorm);
    state->minYNorm = std::min(state->minYNorm, point.yNorm);
    state->maxYNorm = std::max(state->maxYNorm, point.yNorm);
  }
  if (appended) *appended = true;
  return true;
}

ViewerLassoInteractionResult rejected(
    ViewerLassoInteractionStatus status,
    const ViewerLassoInteractionState& state) noexcept {
  ViewerLassoInteractionResult result{};
  result.status = status;
  result.interactionRevision = state.interactionRevision;
  return result;
}

}  // namespace

bool computeViewerLassoSourceImageRect(
    const ScreenRect& contentRect,
    double sourceWidth,
    double sourceHeight,
    ViewerLassoSourceImageRect* output) noexcept {
  if (!output) return false;
  ViewerLassoSourceImageRect result{};
  if (!finiteRect(contentRect) ||
      !validSourceDimensions(sourceWidth, sourceHeight)) {
    *output = result;
    return false;
  }
  const double contentWidth = contentRect.x1 - contentRect.x0;
  const double contentHeight = contentRect.y1 - contentRect.y0;
  const double scale = std::min(contentWidth / sourceWidth,
                                contentHeight / sourceHeight);
  const double width = sourceWidth * scale;
  const double height = sourceHeight * scale;
  const double x0 = contentRect.x0 + (contentWidth - width) * 0.5;
  const double y0 = contentRect.y0 + (contentHeight - height) * 0.5;
  result.rect = {static_cast<float>(x0), static_cast<float>(y0),
                 static_cast<float>(x0 + width),
                 static_cast<float>(y0 + height)};
  result.valid = std::isfinite(scale) && scale > 0.0 &&
                 finiteRect(result.rect);
  *output = result;
  return result.valid;
}

bool mapViewerLassoPointerToSource(
    const ViewerLassoSourceImageRect& imageRect,
    double logicalX,
    double logicalY,
    bool clampOutside,
    LassoPointNorm* output) noexcept {
  if (!output || !imageRect.valid || !finiteRect(imageRect.rect) ||
      !std::isfinite(logicalX) || !std::isfinite(logicalY)) {
    return false;
  }
  if (!clampOutside &&
      !viewerUiRectContainsInclusive(imageRect.rect,
                                     static_cast<float>(logicalX),
                                     static_cast<float>(logicalY))) {
    return false;
  }
  const double x = std::clamp(logicalX,
                              static_cast<double>(imageRect.rect.x0),
                              static_cast<double>(imageRect.rect.x1));
  const double y = std::clamp(logicalY,
                              static_cast<double>(imageRect.rect.y0),
                              static_cast<double>(imageRect.rect.y1));
  const double width = imageRect.rect.x1 - imageRect.rect.x0;
  const double height = imageRect.rect.y1 - imageRect.rect.y0;
  LassoPointNorm point{};
  point.xNorm = static_cast<float>((x - imageRect.rect.x0) / width);
  point.yNorm = static_cast<float>(1.0 -
      (y - imageRect.rect.y0) / height);
  point.xNorm = std::clamp(point.xNorm, 0.0f, 1.0f);
  point.yNorm = std::clamp(point.yNorm, 0.0f, 1.0f);
  if (!validPoint(point)) return false;
  *output = point;
  return true;
}

ViewerLassoInteractionResult reduceViewerLassoInteraction(
    ViewerLassoInteractionState* state,
    const ViewerLassoInteractionRequest& request) noexcept {
  ViewerLassoInteractionState empty{};
  if (!state) return rejected(ViewerLassoInteractionStatus::InvalidState, empty);
  const ViewerLassoInteractionState original = *state;
  if (!validState(original)) {
    return rejected(ViewerLassoInteractionStatus::InvalidState, original);
  }
  if (!request.input || !request.session || !request.scene ||
      !request.workspace || !request.input->accepted()) {
    return rejected(ViewerLassoInteractionStatus::InvalidInput, original);
  }

  ViewerLassoInteractionState next = original;
  ViewerLassoInteractionResult output{};
  output.status = ViewerLassoInteractionStatus::NoChange;
  output.interactionRevision = original.interactionRevision;
  const auto& input = request.input->acceptedInput;

  if (request.input->cancelInteractions ||
      input.kind == ViewerSessionInputKind::Cancelled) {
    if (next.pointerCaptureActive) {
      clearCapture(&next);
      if (!bumpRevision(&next)) {
        return rejected(ViewerLassoInteractionStatus::RevisionOverflow,
                        original);
      }
      output.status = ViewerLassoInteractionStatus::Accepted;
      output.stateChanged = true;
      output.interactionRevision = next.interactionRevision;
    }
    if (input.sequence > next.lastInputSequence) {
      next.lastInputSequence = input.sequence;
    }
    *state = next;
    return output;
  }
  if (!request.input->inputAccepted ||
      input.kind == ViewerSessionInputKind::None || input.sequence == 0u) {
    return output;
  }
  if (input.sequence <= original.lastInputSequence) {
    return rejected(ViewerLassoInteractionStatus::ReplayedInput, original);
  }

  const bool press = input.kind == ViewerSessionInputKind::PointerButton &&
                     input.button == ViewerSessionPointerButton::Primary &&
                     input.pressed;
  const bool release = input.kind == ViewerSessionInputKind::PointerButton &&
                       input.button == ViewerSessionPointerButton::Primary &&
                       !input.pressed && next.pointerCaptureActive;
  const bool move = input.kind == ViewerSessionInputKind::PointerMoved &&
                    next.pointerCaptureActive;
  if (!press && !release && !move) {
    next.lastInputSequence = input.sequence;
    *state = next;
    return output;
  }
  if (press && !request.authorizedStart()) {
    return rejected(ViewerLassoInteractionStatus::NotAuthorized, original);
  }
  if (press && !request.workspace->sourceLassoSessionActive) {
    return rejected(ViewerLassoInteractionStatus::SessionInactive, original);
  }
  if (!validSourceDimensions(request.sourceWidth, request.sourceHeight)) {
    return rejected(ViewerLassoInteractionStatus::InvalidSourceDimensions,
                    original);
  }
  if (!press && next.pointerCaptureActive &&
      (request.sourceWidth != next.captureSourceWidth ||
       request.sourceHeight != next.captureSourceHeight)) {
    return rejected(ViewerLassoInteractionStatus::CaptureWindowChanged,
                    original);
  }
  if (!request.scene->ready()) {
    return rejected(ViewerLassoInteractionStatus::InvalidScene, original);
  }
  if (!validateViewerWorkspaceState(*request.workspace)) {
    return rejected(ViewerLassoInteractionStatus::InvalidWorkspace, original);
  }

  int windowId = next.pointerWindowId;
  if (press) {
    const ViewerUiHitResult hit = viewerUiHitTest(
        *request.scene, static_cast<float>(input.logicalX),
        static_cast<float>(input.logicalY));
    if (hit.control != ViewerUiControlKind::PlotBody || hit.windowId <= 0) {
      return rejected(ViewerLassoInteractionStatus::WrongControl, original);
    }
    const PlotWindowDomainState* window =
        workspaceWindow(*request.workspace, hit.windowId);
    if (!window || window->viewState.plotModel != kPlotModelSourceSignal) {
      return rejected(ViewerLassoInteractionStatus::WrongControl, original);
    }
    windowId = hit.windowId;
  }
  const PlotWindowDomainState* window =
      workspaceWindow(*request.workspace, windowId);
  const ViewerUiWindowScene* ui = sceneWindow(*request.scene, windowId);
  if (!window || !ui) {
    return rejected(ViewerLassoInteractionStatus::MissingWindow, original);
  }
  if (window->viewState.plotModel != kPlotModelSourceSignal) {
    return rejected(ViewerLassoInteractionStatus::CaptureWindowChanged,
                    original);
  }
  ViewerLassoSourceImageRect imageRect{};
  if (!computeViewerLassoSourceImageRect(ui->contentRect, request.sourceWidth,
                                         request.sourceHeight, &imageRect)) {
    return rejected(ViewerLassoInteractionStatus::InvalidSourceDimensions,
                    original);
  }
  LassoPointNorm point{};
  if (!mapViewerLassoPointerToSource(imageRect, input.logicalX, input.logicalY,
                                     !press, &point)) {
    if (press) {
      next.lastInputSequence = input.sequence;
      *state = next;
      output.status = ViewerLassoInteractionStatus::LetterboxRejected;
      return output;
    }
    return rejected(ViewerLassoInteractionStatus::InvalidInput, original);
  }

  if (press) {
    next.pointerCaptureActive = true;
    next.pointerWindowId = windowId;
    next.pointerButton = ViewerSessionPointerButton::Primary;
    next.strokeSubtract = request.workspace->sourceLassoSubtractMode;
    next.captureSourceWidth = request.sourceWidth;
    next.captureSourceHeight = request.sourceHeight;
    next.pointCount = 0u;
    next.boundsValid = false;
    bool appended = false;
    if (!appendPoint(&next, point, &appended) || !bumpRevision(&next)) {
      return rejected(ViewerLassoInteractionStatus::RevisionOverflow,
                      original);
    }
    next.lastInputSequence = input.sequence;
    *state = next;
    output.status = ViewerLassoInteractionStatus::Accepted;
    output.stateChanged = true;
    output.pointAppended = appended;
    output.windowId = windowId;
    output.interactionRevision = next.interactionRevision;
    return output;
  }

  bool appended = false;
  if (!appendPoint(&next, point, &appended)) {
    next.lastInputSequence = input.sequence;
    *state = next;
    output.status = ViewerLassoInteractionStatus::CapacityExceeded;
    output.windowId = windowId;
    return output;
  }
  if (move) {
    if (appended && !bumpRevision(&next)) {
      return rejected(ViewerLassoInteractionStatus::RevisionOverflow,
                      original);
    }
    next.lastInputSequence = input.sequence;
    *state = next;
    output.status = appended ? ViewerLassoInteractionStatus::Accepted
                             : ViewerLassoInteractionStatus::NoChange;
    output.stateChanged = appended;
    output.pointAppended = appended;
    output.windowId = windowId;
    output.interactionRevision = next.interactionRevision;
    return output;
  }

  try {
    if (next.pointCount < 3u) {
      clearCapture(&next);
      if (!bumpRevision(&next)) {
        return rejected(ViewerLassoInteractionStatus::RevisionOverflow,
                        original);
      }
      next.lastInputSequence = input.sequence;
      *state = next;
      output.status = ViewerLassoInteractionStatus::TooFewPoints;
      output.stateChanged = true;
      output.strokeDiscarded = true;
      output.windowId = windowId;
      output.interactionRevision = next.interactionRevision;
      return output;
    }
    LassoStroke stroke{};
    stroke.subtract = next.strokeSubtract;
    stroke.points.assign(next.points.begin(),
                         next.points.begin() +
                             static_cast<std::ptrdiff_t>(next.pointCount));
    stroke.boundsValid = true;
    stroke.minXNorm = next.minXNorm;
    stroke.maxXNorm = next.maxXNorm;
    stroke.minYNorm = next.minYNorm;
    stroke.maxYNorm = next.maxYNorm;
    clearCapture(&next);
    if (!bumpRevision(&next)) {
      return rejected(ViewerLassoInteractionStatus::RevisionOverflow,
                      original);
    }
    next.lastInputSequence = input.sequence;
    *state = next;
    output.status = ViewerLassoInteractionStatus::Accepted;
    output.stateChanged = true;
    output.pointAppended = appended;
    output.strokeCompleted = true;
    output.windowId = windowId;
    output.stroke = std::move(stroke);
    output.interactionRevision = next.interactionRevision;
    return output;
  } catch (...) {
    return rejected(ViewerLassoInteractionStatus::AllocationFailure, original);
  }
}

ViewerLassoOverlayResult appendViewerLassoOverlay(
    const ViewerLassoInteractionState& interaction,
    const ViewerWorkspaceState& workspace,
    double sourceWidth,
    double sourceHeight,
    ViewerUiScene* scene) noexcept {
  ViewerLassoOverlayResult result{};
  if (!scene) return result;
  if (!validState(interaction)) {
    result.status = ViewerLassoOverlayStatus::InvalidState;
    return result;
  }
  if (!scene->ready()) {
    result.status = ViewerLassoOverlayStatus::InvalidScene;
    return result;
  }
  if (!validateViewerWorkspaceState(workspace)) {
    result.status = ViewerLassoOverlayStatus::InvalidWorkspace;
    return result;
  }
  if (!validSourceDimensions(sourceWidth, sourceHeight)) {
    result.status = ViewerLassoOverlayStatus::InvalidSourceDimensions;
    return result;
  }
  if (interaction.pointerCaptureActive &&
      (sourceWidth != interaction.captureSourceWidth ||
       sourceHeight != interaction.captureSourceHeight)) {
    result.status = ViewerLassoOverlayStatus::InvalidSourceDimensions;
    return result;
  }

  int sourceWindowId = interaction.pointerCaptureActive
                           ? interaction.pointerWindowId
                           : -1;
  if (sourceWindowId <= 0) {
    for (const auto& window : workspace.windows) {
      if (window.viewState.plotModel == kPlotModelSourceSignal &&
          !window.sourceSignalDocked) {
        sourceWindowId = window.windowId;
        break;
      }
    }
  }
  const auto* sourceWindow = workspaceWindow(workspace, sourceWindowId);
  const auto* ui = sceneWindow(*scene, sourceWindowId);
  if (!interaction.pointerCaptureActive && sourceWindowId <= 0 &&
      std::any_of(workspace.windows.begin(), workspace.windows.end(),
                  [](const PlotWindowDomainState& window) {
                    return window.viewState.plotModel ==
                               kPlotModelSourceSignal &&
                           window.sourceSignalDocked;
                  })) {
    result.status = ViewerLassoOverlayStatus::Ready;
    return result;
  }
  if (!sourceWindow || !ui ||
      sourceWindow->viewState.plotModel != kPlotModelSourceSignal) {
    result.status = ViewerLassoOverlayStatus::MissingWindow;
    return result;
  }
  ViewerLassoSourceImageRect image{};
  if (!computeViewerLassoSourceImageRect(ui->contentRect, sourceWidth,
                                         sourceHeight, &image)) {
    result.status = ViewerLassoOverlayStatus::InvalidSourceDimensions;
    return result;
  }

  const std::vector<LassoStroke>* retained = nullptr;
  if (workspace.sourceLassoSelectionsSynced ||
      workspace.sourceLassoTargetWindowId <= 0) {
    retained = &workspace.sourceLassoStrokes;
  } else if (const auto* owner = workspaceWindow(
                 workspace, workspace.sourceLassoTargetWindowId)) {
    retained = &owner->viewerLassoStrokes;
  }
  std::size_t segmentCount = 0u;
  if (retained) {
    for (const auto& stroke : *retained) {
      if (stroke.points.size() >= 2u) segmentCount += stroke.points.size();
    }
  }
  if (interaction.pointerCaptureActive && interaction.pointCount >= 2u) {
    segmentCount += interaction.pointCount - 1u;
  }
  if (segmentCount > kViewerWorkspaceMaxLassoPoints +
                         kViewerWorkspaceMaxLassoPointsPerStroke ||
      segmentCount > (std::numeric_limits<std::size_t>::max() / 6u)) {
    result.status = ViewerLassoOverlayStatus::CapacityExceeded;
    return result;
  }

  try {
    std::vector<ViewerUiVectorVertex> vertices;
    vertices.reserve(segmentCount * 6u);
    auto logicalPoint = [&](const LassoPointNorm& point) {
      return std::pair<float, float>{
          image.rect.x0 + point.xNorm * (image.rect.x1 - image.rect.x0),
          image.rect.y0 + (1.0f - point.yNorm) *
                              (image.rect.y1 - image.rect.y0)};
    };
    auto segment = [&](const LassoPointNorm& a, const LassoPointNorm& b,
                       const ViewerUiColor& color) -> bool {
      const auto p0 = logicalPoint(a);
      const auto p1 = logicalPoint(b);
      const float dx = p1.first - p0.first;
      const float dy = p1.second - p0.second;
      const float length = std::sqrt(dx * dx + dy * dy);
      if (!std::isfinite(length) || length <= 1.0e-5f) return false;
      constexpr float kHalfThickness = 1.35f;
      const float nx = -dy * (kHalfThickness / length);
      const float ny = dx * (kHalfThickness / length);
      const std::pair<float, float> corners[4] = {
          {p0.first + nx, p0.second + ny},
          {p0.first - nx, p0.second - ny},
          {p1.first + nx, p1.second + ny},
          {p1.first - nx, p1.second - ny},
      };
      const int order[6] = {0, 1, 2, 2, 1, 3};
      for (int index : order) {
        ViewerUiVectorVertex vertex{};
        vertex.x = corners[index].first;
        vertex.y = corners[index].second;
        vertex.color = color;
        vertex.windowId = sourceWindowId;
        vertices.push_back(vertex);
      }
      return true;
    };

    if (retained) {
      for (const auto& stroke : *retained) {
        if (stroke.points.size() < 2u) continue;
        const ViewerUiColor color = stroke.subtract
            ? ViewerUiColor{1.0f, 0.32f, 0.18f, 0.72f}
            : ViewerUiColor{0.18f, 0.82f, 1.0f, 0.72f};
        for (std::size_t i = 1u; i < stroke.points.size(); ++i) {
          if (segment(stroke.points[i - 1u], stroke.points[i], color)) {
            ++result.retainedSegments;
          }
        }
        if (segment(stroke.points.back(), stroke.points.front(), color)) {
          ++result.retainedSegments;
        }
      }
    }
    if (interaction.pointerCaptureActive && interaction.pointCount >= 2u) {
      const ViewerUiColor color = interaction.strokeSubtract
          ? ViewerUiColor{1.0f, 0.32f, 0.18f, 0.96f}
          : ViewerUiColor{0.18f, 0.88f, 1.0f, 0.96f};
      for (std::size_t i = 1u; i < interaction.pointCount; ++i) {
        if (segment(interaction.points[i - 1u], interaction.points[i], color)) {
          ++result.activeSegments;
        }
      }
    }
    scene->vectors.insert(scene->vectors.end(), vertices.begin(), vertices.end());
    result.appendedVertices = vertices.size();
    result.status = ViewerLassoOverlayStatus::Ready;
    return result;
  } catch (...) {
    result.status = ViewerLassoOverlayStatus::AllocationFailure;
    return result;
  }
}

}  // namespace ChromaspaceViewer
