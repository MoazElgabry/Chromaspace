#include "ChromaspaceViewerLayout.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "ChromaspaceViewerState.h"

namespace ChromaspaceViewer {
namespace {

constexpr StandardLayoutDescriptor kStandardLayouts[kViewerLayoutChoiceCount] = {
    {0, "Single", 1, 720, 600},
    {1, "Split 2", 2, 1120, 640},
    {2, "Triple Columns", 3, 1320, 720},
    {3, "Triple 2 + 1", 3, 1280, 760},
    {4, "Quadrants", 4, 1280, 900},
    {5, "Six Views", 6, 1468, 1030},
    {6, "Solo", 1, 720, 600},
};

constexpr int kDisplayOrder[kViewerLayoutChoiceCount] = {
    kViewerLayoutSoloIndex, 0, 1, 2, 3, 4, 5};

int clampedLayoutIndex(int layoutIndex) noexcept {
  return std::clamp(layoutIndex, 0, kViewerLayoutChoiceCount - 1);
}

bool namesEqualNoCase(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

float clampf(float value, float low, float high) noexcept {
  // Keep the same deliberately small clamp semantics used by the viewer
  // adapter, including its behavior when a transient low/high pair crosses.
  return value < low ? low : (value > high ? high : value);
}

PlotWindowRectNorm clampRectWithMinimums(PlotWindowRectNorm rect,
                                         const WorkspaceGeometry& workspace,
                                         float minWidth,
                                         float minHeight) noexcept {
  const float maxHeight = std::max(0.0f, workspace.heightNorm);
  const float boundedMinHeight = std::min(minHeight, maxHeight);
  rect.w = clampf(rect.w, minWidth, 1.0f);
  rect.h = clampf(rect.h, boundedMinHeight, maxHeight);
  rect.x = clampf(rect.x, 0.0f, std::max(0.0f, 1.0f - rect.w));
  rect.y = clampf(rect.y,
                  workspace.topNorm,
                  std::max(workspace.topNorm, 1.0f - rect.h));
  return rect;
}

bool modeTouchesLeft(PlotWindowDragMode mode) noexcept {
  return mode == PlotWindowDragMode::ResizeLeft ||
         mode == PlotWindowDragMode::ResizeTopLeft ||
         mode == PlotWindowDragMode::ResizeBottomLeft;
}

bool modeTouchesRight(PlotWindowDragMode mode) noexcept {
  return mode == PlotWindowDragMode::ResizeRight ||
         mode == PlotWindowDragMode::ResizeTopRight ||
         mode == PlotWindowDragMode::ResizeBottomRight;
}

bool modeTouchesTop(PlotWindowDragMode mode) noexcept {
  return mode == PlotWindowDragMode::ResizeTop ||
         mode == PlotWindowDragMode::ResizeTopLeft ||
         mode == PlotWindowDragMode::ResizeTopRight;
}

bool modeTouchesBottom(PlotWindowDragMode mode) noexcept {
  return mode == PlotWindowDragMode::ResizeBottom ||
         mode == PlotWindowDragMode::ResizeBottomLeft ||
         mode == PlotWindowDragMode::ResizeBottomRight;
}

}  // namespace

WorkspaceGeometry workspaceGeometry(bool showWorkspaceToolbar,
                                     int windowHeight) noexcept {
  WorkspaceGeometry geometry{};
  geometry.topNorm = showWorkspaceToolbar
                         ? clampf(kViewerWorkspaceToolbarHeight /
                                      static_cast<float>(std::max(1, windowHeight)),
                                  0.0f,
                                  0.45f)
                         : 0.0f;
  geometry.heightNorm = std::max(0.05f, 1.0f - geometry.topNorm);
  return geometry;
}

float plotWindowMinNormWidth(int windowWidth) noexcept {
  return std::min(0.88f,
                  180.0f / static_cast<float>(std::max(1, windowWidth)));
}

float plotWindowMinNormHeight(int windowHeight) noexcept {
  return std::min(0.88f,
                  140.0f / static_cast<float>(std::max(1, windowHeight)));
}

PlotWindowRectNorm workspaceRelativeRect(const WorkspaceGeometry& workspace,
                                          float x,
                                          float y,
                                          float w,
                                          float h) noexcept {
  return {x,
          workspace.topNorm + y * workspace.heightNorm,
          w,
          h * workspace.heightNorm};
}

PlotWindowRectNorm clampPlotWindowRect(PlotWindowRectNorm rect,
                                       const WorkspaceGeometry& workspace,
                                       int windowWidth,
                                       int windowHeight) noexcept {
  return clampRectWithMinimums(rect,
                                workspace,
                                plotWindowMinNormWidth(windowWidth),
                                plotWindowMinNormHeight(windowHeight));
}

PlotWindowRectNorm reflowPlotWindowRectForWorkspaceTop(PlotWindowRectNorm rect,
                                                        float oldTop,
                                                        float newTop) noexcept {
  const float oldHeight = std::max(0.05f, 1.0f - oldTop);
  const float newHeight = std::max(0.05f, 1.0f - newTop);
  const float workspaceY = (rect.y - oldTop) / oldHeight;
  const float workspaceH = rect.h / oldHeight;
  rect.y = newTop + workspaceY * newHeight;
  rect.h = workspaceH * newHeight;
  return rect;
}

PlotWindowDragTraits plotWindowDragModeTraits(PlotWindowDragMode mode) noexcept {
  PlotWindowDragTraits traits{};
  switch (mode) {
    case PlotWindowDragMode::ResizeLeft:
      traits.isResize = true;
      traits.touchesLeft = true;
      break;
    case PlotWindowDragMode::ResizeRight:
      traits.isResize = true;
      traits.touchesRight = true;
      break;
    case PlotWindowDragMode::ResizeTop:
      traits.isResize = true;
      traits.touchesTop = true;
      break;
    case PlotWindowDragMode::ResizeBottom:
      traits.isResize = true;
      traits.touchesBottom = true;
      break;
    case PlotWindowDragMode::ResizeTopLeft:
      traits.isResize = true;
      traits.touchesTop = true;
      traits.touchesLeft = true;
      break;
    case PlotWindowDragMode::ResizeTopRight:
      traits.isResize = true;
      traits.touchesTop = true;
      traits.touchesRight = true;
      break;
    case PlotWindowDragMode::ResizeBottomLeft:
      traits.isResize = true;
      traits.touchesBottom = true;
      traits.touchesLeft = true;
      break;
    case PlotWindowDragMode::ResizeBottomRight:
      traits.isResize = true;
      traits.touchesBottom = true;
      traits.touchesRight = true;
      break;
    case PlotWindowDragMode::None:
    case PlotWindowDragMode::Move:
      break;
  }
  return traits;
}

PlotWindowDragMode plotWindowDragModeAt(
    const PlotWindowDragHitTestRequest& request) noexcept {
  const ScreenRect& screen = request.screenRect;
  const float cursorX = request.cursorX;
  const float cursorY = request.cursorY;
  if (cursorX < screen.x0 || cursorX > screen.x1 || cursorY < screen.y0 ||
      cursorY > screen.y1) {
    return PlotWindowDragMode::None;
  }

  constexpr float kResizeHandlePixels = 9.0f;
  const bool left = std::fabs(cursorX - screen.x0) <= kResizeHandlePixels;
  const bool right = std::fabs(cursorX - screen.x1) <= kResizeHandlePixels;
  const bool top = std::fabs(cursorY - screen.y0) <= kResizeHandlePixels;
  const bool bottom = std::fabs(cursorY - screen.y1) <= kResizeHandlePixels;
  if (top && left) return PlotWindowDragMode::ResizeTopLeft;
  if (top && right) return PlotWindowDragMode::ResizeTopRight;
  if (bottom && left) return PlotWindowDragMode::ResizeBottomLeft;
  if (bottom && right) return PlotWindowDragMode::ResizeBottomRight;
  if (left) return PlotWindowDragMode::ResizeLeft;
  if (right) return PlotWindowDragMode::ResizeRight;
  if (top) return PlotWindowDragMode::ResizeTop;
  if (bottom) return PlotWindowDragMode::ResizeBottom;
  if (cursorY - screen.y0 <= 26.0f && request.windowRect.w < 0.995f &&
      request.windowRect.h < 0.995f) {
    return PlotWindowDragMode::Move;
  }
  return PlotWindowDragMode::None;
}

PlotWindowRectNorm applyPlotWindowDrag(
    const PlotWindowDragRequest& request) noexcept {
  PlotWindowRectNorm rect = request.startRect;
  const float minWidth = request.minWidthNorm > 0.0f
                             ? request.minWidthNorm
                             : plotWindowMinNormWidth(1);
  const float minHeight = request.minHeightNorm > 0.0f
                              ? request.minHeightNorm
                              : plotWindowMinNormHeight(1);
  const float dx = request.deltaXNorm;
  const float dy = request.deltaYNorm;
  const auto resizeLeft = [&]() {
    const float right = rect.x + rect.w;
    rect.x = clampf(rect.x + dx, 0.0f, right - minWidth);
    rect.w = right - rect.x;
  };
  const auto resizeRight = [&]() {
    rect.w = clampf(rect.w + dx, minWidth, 1.0f - rect.x);
  };
  const auto resizeTop = [&]() {
    const float bottom = rect.y + rect.h;
    rect.y = clampf(rect.y + dy,
                    request.workspace.topNorm,
                    bottom - minHeight);
    rect.h = bottom - rect.y;
  };
  const auto resizeBottom = [&]() {
    rect.h = clampf(rect.h + dy, minHeight, 1.0f - rect.y);
  };
  switch (request.mode) {
    case PlotWindowDragMode::Move:
      rect.x = request.startRect.x + dx;
      rect.y = request.startRect.y + dy;
      break;
    case PlotWindowDragMode::ResizeLeft:
      resizeLeft();
      break;
    case PlotWindowDragMode::ResizeRight:
      resizeRight();
      break;
    case PlotWindowDragMode::ResizeTop:
      resizeTop();
      break;
    case PlotWindowDragMode::ResizeBottom:
      resizeBottom();
      break;
    case PlotWindowDragMode::ResizeTopLeft:
      resizeTop();
      resizeLeft();
      break;
    case PlotWindowDragMode::ResizeTopRight:
      resizeTop();
      resizeRight();
      break;
    case PlotWindowDragMode::ResizeBottomLeft:
      resizeBottom();
      resizeLeft();
      break;
    case PlotWindowDragMode::ResizeBottomRight:
      resizeBottom();
      resizeRight();
      break;
    case PlotWindowDragMode::None:
      break;
  }
  return clampRectWithMinimums(rect, request.workspace, minWidth, minHeight);
}

PlotWindowSnapPreviewResult computePlotWindowSnapPreview(
    const PlotWindowSnapPreviewRequest& request) noexcept {
  PlotWindowSnapPreviewResult result{};
  const float width = static_cast<float>(std::max(1, request.windowWidth));
  const float height = static_cast<float>(std::max(1, request.windowHeight));
  const float availableWidth =
      std::max(1.0f, width - request.reservedLeftPixels);
  const float nx = clampf((request.cursorX - request.reservedLeftPixels) /
                              availableWidth,
                          0.0f,
                          1.0f);
  const float ny = request.cursorY / height;
  const float workspaceTop = request.workspace.topNorm;
  const float workspaceHeight = request.workspace.heightNorm;
  const float workspaceNy = clampf((ny - workspaceTop) / workspaceHeight,
                                   0.0f,
                                   1.0f);
  const float edgeX = 22.0f / width;
  const float edgeY = 22.0f / (height * workspaceHeight);
  const float cornerX = 34.0f / width;
  const float cornerY = 34.0f / (height * workspaceHeight);
  PlotWindowRectNorm candidate = request.candidateRect;
  bool haveCandidate = false;
  const auto workspaceCandidate = [&](float x,
                                      float y,
                                      float candidateWidth,
                                      float candidateHeight) {
    return workspaceRelativeRect(request.workspace,
                                 x,
                                 y,
                                 candidateWidth,
                                 candidateHeight);
  };
  const bool nearWorkspaceEdge = nx <= edgeX || nx >= 1.0f - edgeX ||
                                 workspaceNy <= edgeY ||
                                 workspaceNy >= 1.0f - edgeY;
  const bool nearlyFullWidth = candidate.w >= 0.92f;
  const bool nearlyFullHeight = candidate.h >= workspaceHeight * 0.86f;

  if (request.singleWindowWorkspace &&
      (nearWorkspaceEdge ||
       (nearlyFullWidth &&
        (workspaceNy <= cornerY || workspaceNy >= 1.0f - cornerY)) ||
       (nearlyFullHeight &&
        (nx <= cornerX || nx >= 1.0f - cornerX)))) {
    candidate = workspaceCandidate(0.0f, 0.0f, 1.0f, 1.0f);
    haveCandidate = true;
  } else if (nx <= cornerX && workspaceNy <= cornerY) {
    candidate = workspaceCandidate(0.0f, 0.0f, 0.5f, 0.5f);
    haveCandidate = true;
  } else if (nx >= 1.0f - cornerX && workspaceNy <= cornerY) {
    candidate = workspaceCandidate(0.5f, 0.0f, 0.5f, 0.5f);
    haveCandidate = true;
  } else if (nx <= cornerX && workspaceNy >= 1.0f - cornerY) {
    candidate = workspaceCandidate(0.0f, 0.5f, 0.5f, 0.5f);
    haveCandidate = true;
  } else if (nx >= 1.0f - cornerX && workspaceNy >= 1.0f - cornerY) {
    candidate = workspaceCandidate(0.5f, 0.5f, 0.5f, 0.5f);
    haveCandidate = true;
  } else if (nx <= edgeX) {
    candidate = workspaceCandidate(0.0f, 0.0f, 0.5f, 1.0f);
    haveCandidate = true;
  } else if (nx >= 1.0f - edgeX) {
    candidate = workspaceCandidate(0.5f, 0.0f, 0.5f, 1.0f);
    haveCandidate = true;
  } else if (workspaceNy <= edgeY) {
    candidate = workspaceCandidate(0.0f, 0.0f, 1.0f, 0.5f);
    haveCandidate = true;
  } else if (workspaceNy >= 1.0f - edgeY) {
    candidate = workspaceCandidate(0.0f, 0.5f, 1.0f, 0.5f);
    haveCandidate = true;
  } else {
    constexpr float kGuideDistancePixels = 11.0f;
    float bestDxPixels = kGuideDistancePixels + 1.0f;
    float bestDyPixels = kGuideDistancePixels + 1.0f;
    float snappedX = candidate.x;
    float snappedY = candidate.y;
    float snappedW = candidate.w;
    float snappedH = candidate.h;
    const float currentLeft = candidate.x;
    const float currentRight = candidate.x + candidate.w;
    const float currentTop = candidate.y;
    const float currentBottom = candidate.y + candidate.h;
    const float minWidth = plotWindowMinNormWidth(request.windowWidth);
    const float minHeight = std::min(plotWindowMinNormHeight(request.windowHeight),
                                     workspaceHeight);
    const bool moveMode = request.dragMode == PlotWindowDragMode::Move;
    std::vector<float> xGuides = {0.0f, 1.0f, 0.5f, 1.0f / 3.0f, 2.0f / 3.0f};
    std::vector<float> yGuides = {
        workspaceTop,
        1.0f,
        workspaceTop + workspaceHeight * 0.5f,
        workspaceTop + workspaceHeight / 3.0f,
        workspaceTop + workspaceHeight * 2.0f / 3.0f};
    for (const PlotWindowRectNorm& other : request.otherWindowRects) {
      xGuides.push_back(other.x);
      xGuides.push_back(other.x + other.w);
      xGuides.push_back(other.x + other.w * 0.5f);
      yGuides.push_back(other.y);
      yGuides.push_back(other.y + other.h);
      yGuides.push_back(other.y + other.h * 0.5f);
    }
    for (const float guide : xGuides) {
      if (moveMode) {
        const float leftDeltaPixels = std::fabs(currentLeft - guide) * availableWidth;
        if (leftDeltaPixels < bestDxPixels) {
          bestDxPixels = leftDeltaPixels;
          snappedX = guide;
          snappedW = candidate.w;
        }
        const float rightDeltaPixels = std::fabs(currentRight - guide) * availableWidth;
        if (rightDeltaPixels < bestDxPixels) {
          bestDxPixels = rightDeltaPixels;
          snappedX = guide - candidate.w;
          snappedW = candidate.w;
        }
      }
      if (modeTouchesLeft(request.dragMode)) {
        const float nextWidth = currentRight - guide;
        const float leftDeltaPixels = std::fabs(currentLeft - guide) * availableWidth;
        if (nextWidth >= minWidth && leftDeltaPixels < bestDxPixels) {
          bestDxPixels = leftDeltaPixels;
          snappedX = guide;
          snappedW = nextWidth;
        }
      }
      if (modeTouchesRight(request.dragMode)) {
        const float nextWidth = guide - candidate.x;
        const float rightDeltaPixels = std::fabs(currentRight - guide) * availableWidth;
        if (nextWidth >= minWidth && rightDeltaPixels < bestDxPixels) {
          bestDxPixels = rightDeltaPixels;
          snappedX = candidate.x;
          snappedW = nextWidth;
        }
      }
    }
    for (const float guide : yGuides) {
      if (moveMode) {
        const float topDeltaPixels = std::fabs(currentTop - guide) * height;
        if (topDeltaPixels < bestDyPixels) {
          bestDyPixels = topDeltaPixels;
          snappedY = guide;
          snappedH = candidate.h;
        }
        const float bottomDeltaPixels = std::fabs(currentBottom - guide) * height;
        if (bottomDeltaPixels < bestDyPixels) {
          bestDyPixels = bottomDeltaPixels;
          snappedY = guide - candidate.h;
          snappedH = candidate.h;
        }
      }
      if (modeTouchesTop(request.dragMode)) {
        const float nextHeight = currentBottom - guide;
        const float topDeltaPixels = std::fabs(currentTop - guide) * height;
        if (nextHeight >= minHeight && topDeltaPixels < bestDyPixels) {
          bestDyPixels = topDeltaPixels;
          snappedY = guide;
          snappedH = nextHeight;
        }
      }
      if (modeTouchesBottom(request.dragMode)) {
        const float nextHeight = guide - candidate.y;
        const float bottomDeltaPixels = std::fabs(currentBottom - guide) * height;
        if (nextHeight >= minHeight && bottomDeltaPixels < bestDyPixels) {
          bestDyPixels = bottomDeltaPixels;
          snappedY = candidate.y;
          snappedH = nextHeight;
        }
      }
    }
    if (bestDxPixels <= kGuideDistancePixels) {
      candidate.x = snappedX;
      candidate.w = snappedW;
      haveCandidate = true;
    }
    if (bestDyPixels <= kGuideDistancePixels) {
      candidate.y = snappedY;
      candidate.h = snappedH;
      haveCandidate = true;
    }
  }

  if (haveCandidate) {
    result.visible = true;
    result.rect = clampPlotWindowRect(candidate,
                                      request.workspace,
                                      request.windowWidth,
                                      request.windowHeight);
  }
  return result;
}

const StandardLayoutDescriptor& standardPlotLayout(int layoutIndex) noexcept {
  return kStandardLayouts[clampedLayoutIndex(layoutIndex)];
}

const StandardLayoutDescriptor& standardPlotLayoutForDisplayRow(int row) noexcept {
  return standardPlotLayout(kDisplayOrder[std::clamp(row, 0, kViewerLayoutChoiceCount - 1)]);
}

const StandardLayoutDescriptor* findStandardPlotLayout(std::string_view name) noexcept {
  for (const StandardLayoutDescriptor& layout : kStandardLayouts) {
    if (namesEqualNoCase(name, layout.label)) return &layout;
  }
  return nullptr;
}

bool isStandardPlotLayoutNameReserved(std::string_view name) noexcept {
  return namesEqualNoCase(name, "Custom") || findStandardPlotLayout(name) != nullptr;
}

PlotWindowRectNorm standardPlotLayoutSlotRect(const StandardLayoutDescriptor& layout,
                                              int slotIndex) noexcept {
  slotIndex = std::max(0, slotIndex);
  switch (layout.index) {
    case 1:
      return {slotIndex == 0 ? 0.0f : 0.5f, 0.0f, 0.5f, 1.0f};
    case 2:
      return {static_cast<float>(std::min(slotIndex, 2)) / 3.0f,
              0.0f,
              1.0f / 3.0f,
              1.0f};
    case 3:
      if (slotIndex == 0) return {0.0f, 0.0f, 2.0f / 3.0f, 1.0f};
      return {2.0f / 3.0f,
              slotIndex == 1 ? 0.0f : 0.5f,
              1.0f / 3.0f,
              0.5f};
    case 4: {
      const int col = slotIndex % 2;
      const int row = slotIndex / 2;
      return {static_cast<float>(col) * 0.5f,
              static_cast<float>(row) * 0.5f,
              0.5f,
              0.5f};
    }
    case 5: {
      const int col = slotIndex % 3;
      const int row = slotIndex / 3;
      return {static_cast<float>(col) / 3.0f,
              static_cast<float>(row) / 2.0f,
              1.0f / 3.0f,
              0.5f};
    }
    case 0:
    case kViewerLayoutSoloIndex:
    default:
      return {0.0f, 0.0f, 1.0f, 1.0f};
  }
}

int standardPlotLayoutDefaultPlotModel(const StandardLayoutDescriptor& layout,
                                       int slotIndex) noexcept {
  if (layout.index == kViewerLayoutSoloIndex) return -1;
  static constexpr int kDefaultSlotModels[6] = {
      kPlotModelCube,
      kPlotModelHistogram,
      kPlotModelSourceSignal,
      kPlotModelHsl,
      kPlotModelJpConical,
      kPlotModelWaveform};
  return kDefaultSlotModels[std::clamp(slotIndex, 0, 5)];
}

bool plotWindowRectNear(const PlotWindowRectNorm& a,
                        const PlotWindowRectNorm& b) noexcept {
  return std::fabs(a.x - b.x) <= 0.015f &&
         std::fabs(a.y - b.y) <= 0.015f &&
         std::fabs(a.w - b.w) <= 0.015f &&
         std::fabs(a.h - b.h) <= 0.015f;
}

}  // namespace ChromaspaceViewer
