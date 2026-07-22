#include "ChromaspaceViewerUiScene.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace ChromaspaceViewer {
namespace {

constexpr ViewerUiColor kMenuFrameIdle{0.035f, 0.045f, 0.060f, 0.58f};
constexpr ViewerUiColor kMenuFrameHover{0.080f, 0.130f, 0.180f, 0.78f};
constexpr ViewerUiColor kMenuFrameActive{0.100f, 0.190f, 0.260f, 0.88f};
constexpr ViewerUiColor kAddFrameIdle{0.035f, 0.045f, 0.060f, 0.46f};
constexpr ViewerUiColor kAddFrameHover{0.080f, 0.130f, 0.180f, 0.72f};
constexpr ViewerUiColor kAddFrameActive{0.100f, 0.190f, 0.260f, 0.82f};
constexpr ViewerUiColor kLayoutFrameIdle{0.030f, 0.060f, 0.075f, 0.50f};
constexpr ViewerUiColor kLayoutFrameHover{0.060f, 0.160f, 0.200f, 0.72f};
constexpr ViewerUiColor kLayoutFrameActive{0.080f, 0.200f, 0.250f, 0.86f};
constexpr ViewerUiColor kGlyphIdle{0.820f, 0.920f, 1.000f, 0.72f};
constexpr ViewerUiColor kGlyphHover{0.820f, 0.940f, 1.000f, 0.94f};
constexpr ViewerUiColor kGlyphActive{0.820f, 0.940f, 1.000f, 0.94f};
constexpr ViewerUiColor kLayoutCell{0.140f, 0.720f, 0.880f, 0.72f};
constexpr ViewerUiColor kLayoutCellHover{0.140f, 0.720f, 0.880f, 0.96f};

bool validRect(const ScreenRect& rect) noexcept {
  return std::isfinite(rect.x0) && std::isfinite(rect.y0) &&
         std::isfinite(rect.x1) && std::isfinite(rect.y1) &&
         rect.x1 > rect.x0 && rect.y1 > rect.y0;
}

ScreenRect intersectRect(const ScreenRect& a, const ScreenRect& b) noexcept {
  const ScreenRect result{std::max(a.x0, b.x0),
                          std::max(a.y0, b.y0),
                          std::min(a.x1, b.x1),
                          std::min(a.y1, b.y1)};
  return validRect(result) ? result : ScreenRect{};
}

const ViewerUiHitRegion* toolbarRegion(
    const WorkspaceToolbarScene& scene,
    ViewerUiControlKind control) noexcept {
  for (const ViewerUiHitRegion& region : scene.controls) {
    if (region.control == control) return &region;
  }
  return nullptr;
}

ViewerUiColor toolbarFrameColor(ViewerUiControlKind control,
                                bool active,
                                bool hovered) noexcept {
  if (control == ViewerUiControlKind::ToolbarLayoutPreset) {
    if (active) return kLayoutFrameActive;
    if (hovered) return kLayoutFrameHover;
    return kLayoutFrameIdle;
  }
  if (control == ViewerUiControlKind::ToolbarAddPlot) {
    if (active) return kAddFrameActive;
    if (hovered) return kAddFrameHover;
    return kAddFrameIdle;
  }
  if (active) return kMenuFrameActive;
  if (hovered) return kMenuFrameHover;
  return kMenuFrameIdle;
}

ViewerUiColor toolbarGlyphColor(bool active, bool hovered) noexcept {
  if (active) return kGlyphActive;
  if (hovered) return kGlyphHover;
  return kGlyphIdle;
}

void appendToolbarPrimitive(WorkspaceToolbarScene* scene,
                            ViewerUiControlKind control,
                            ViewerUiPrimitiveKind kind,
                            const ScreenRect& rect,
                            ViewerUiColor color) {
  if (!scene || !validRect(rect)) return;
  const ViewerUiHitRegion* region = toolbarRegion(*scene, control);
  if (!region) return;
  const ScreenRect clipped = intersectRect(rect, region->rect);
  if (!validRect(clipped)) return;
  scene->primitives.push_back({clipped, color, control, -1, kind});
}

void appendToolbarButtonFrame(WorkspaceToolbarScene* scene,
                              ViewerUiControlKind control,
                              bool active,
                              bool hovered) {
  const ViewerUiHitRegion* region = toolbarRegion(*scene, control);
  if (!region || !validRect(region->rect)) return;
  const ScreenRect& rect = region->rect;
  appendToolbarPrimitive(scene, control, ViewerUiPrimitiveKind::SolidRect,
                         rect, toolbarFrameColor(control, active, hovered));
  const ViewerUiColor edge{0.70f,
                           0.84f,
                           0.94f,
                           active ? 0.42f : (hovered ? 0.32f : 0.20f)};
  constexpr float kLine = 1.0f;
  appendToolbarPrimitive(scene, control, ViewerUiPrimitiveKind::SolidRect,
                         {rect.x0, rect.y1 - kLine, rect.x1, rect.y1}, edge);
  appendToolbarPrimitive(scene, control, ViewerUiPrimitiveKind::SolidRect,
                         {rect.x0, rect.y0, rect.x0 + kLine, rect.y1},
                         {0.70f, 0.84f, 0.94f,
                          active ? 0.34f : (hovered ? 0.26f : 0.18f)});
  appendToolbarPrimitive(scene, control, ViewerUiPrimitiveKind::SolidRect,
                         {rect.x1 - kLine, rect.y0, rect.x1, rect.y1},
                         {0.70f, 0.84f, 0.94f,
                          active ? 0.28f : (hovered ? 0.20f : 0.14f)});
  appendToolbarPrimitive(scene, control, ViewerUiPrimitiveKind::SolidRect,
                         {rect.x0, rect.y0, rect.x1, rect.y0 + kLine},
                         {0.70f, 0.84f, 0.94f,
                          active ? 0.22f : (hovered ? 0.16f : 0.12f)});
}

void appendToolbarMenuGlyph(WorkspaceToolbarScene* scene,
                            const ScreenRect& rect,
                            bool active,
                            bool hovered) {
  if (!validRect(rect)) return;
  const ViewerUiColor color = toolbarGlyphColor(active, hovered);
  const float dotX0 = rect.x0 + 7.0f;
  const float lineX0 = rect.x0 + 13.0f;
  const float lineX1 = rect.x1 - 7.0f;
  for (float y : {rect.y0 + 9.0f, rect.y0 + 15.0f, rect.y0 + 21.0f}) {
    appendToolbarPrimitive(scene, ViewerUiControlKind::ToolbarMenu,
                           ViewerUiPrimitiveKind::Glyph,
                           {dotX0, y - 1.2f, dotX0 + 2.4f, y + 1.2f}, color);
    appendToolbarPrimitive(scene, ViewerUiControlKind::ToolbarMenu,
                           ViewerUiPrimitiveKind::Glyph,
                           {lineX0, y - 1.0f, lineX1, y + 1.0f}, color);
  }
}

void appendToolbarAddGlyph(WorkspaceToolbarScene* scene,
                           const ScreenRect& rect,
                           bool active,
                           bool hovered) {
  if (!validRect(rect)) return;
  const ViewerUiColor color = toolbarGlyphColor(active, hovered);
  const float cx = (rect.x0 + rect.x1) * 0.5f;
  const float cy = (rect.y0 + rect.y1) * 0.5f;
  appendToolbarPrimitive(scene, ViewerUiControlKind::ToolbarAddPlot,
                         ViewerUiPrimitiveKind::Glyph,
                         {cx - 7.0f, cy - 1.0f, cx + 7.0f, cy + 1.0f},
                         color);
  appendToolbarPrimitive(scene, ViewerUiControlKind::ToolbarAddPlot,
                         ViewerUiPrimitiveKind::Glyph,
                         {cx - 1.0f, cy - 7.0f, cx + 1.0f, cy + 7.0f},
                         color);
}

struct LayoutCell {
  float x = 0.0f;
  float y = 0.0f;
  float w = 0.0f;
  float h = 0.0f;
};

template <typename Emit>
void emitLayoutCells(int layoutIndex, Emit&& emit) {
  if (layoutIndex < 0) {
    emit(LayoutCell{0.00f, 0.00f, 0.58f, 0.54f});
    emit(LayoutCell{0.42f, 0.20f, 0.58f, 0.54f});
    emit(LayoutCell{0.18f, 0.58f, 0.64f, 0.42f});
    return;
  }
  switch (layoutIndex) {
    case 1:
      emit(LayoutCell{0.0f, 0.0f, 0.48f, 1.0f});
      emit(LayoutCell{0.52f, 0.0f, 0.48f, 1.0f});
      break;
    case 2:
      emit(LayoutCell{0.0f, 0.0f, 0.30f, 1.0f});
      emit(LayoutCell{0.35f, 0.0f, 0.30f, 1.0f});
      emit(LayoutCell{0.70f, 0.0f, 0.30f, 1.0f});
      break;
    case 3:
      emit(LayoutCell{0.0f, 0.0f, 0.47f, 1.0f});
      emit(LayoutCell{0.52f, 0.0f, 0.23f, 1.0f});
      emit(LayoutCell{0.78f, 0.0f, 0.22f, 1.0f});
      break;
    case 4:
      emit(LayoutCell{0.0f, 0.0f, 0.48f, 0.48f});
      emit(LayoutCell{0.52f, 0.0f, 0.48f, 0.48f});
      emit(LayoutCell{0.0f, 0.52f, 0.48f, 0.48f});
      emit(LayoutCell{0.52f, 0.52f, 0.48f, 0.48f});
      break;
    case 5:
      emit(LayoutCell{0.0f, 0.0f, 0.30f, 0.48f});
      emit(LayoutCell{0.35f, 0.0f, 0.30f, 0.48f});
      emit(LayoutCell{0.70f, 0.0f, 0.30f, 0.48f});
      emit(LayoutCell{0.0f, 0.52f, 0.30f, 0.48f});
      emit(LayoutCell{0.35f, 0.52f, 0.30f, 0.48f});
      emit(LayoutCell{0.70f, 0.52f, 0.30f, 0.48f});
      break;
    case kViewerLayoutSoloIndex:
      emit(LayoutCell{0.0f, 0.0f, 1.0f, 1.0f});
      emit(LayoutCell{0.34f, 0.34f, 0.32f, 0.32f});
      break;
    case 0:
    default:
      emit(LayoutCell{0.0f, 0.0f, 1.0f, 1.0f});
      break;
  }
}

void appendToolbarLayoutGlyph(WorkspaceToolbarScene* scene,
                              const ScreenRect& rect,
                              int layoutIndex,
                              bool active,
                              bool hovered) {
  if (!validRect(rect)) return;
  const ViewerUiColor cellColor = active || hovered ? kLayoutCellHover
                                                     : kLayoutCell;
  emitLayoutCells(layoutIndex, [&](const LayoutCell& cell) {
    const ScreenRect cellRect{
        rect.x0 + cell.x * (rect.x1 - rect.x0),
        rect.y0 + cell.y * (rect.y1 - rect.y0),
        rect.x0 + (cell.x + cell.w) * (rect.x1 - rect.x0),
        rect.y0 + (cell.y + cell.h) * (rect.y1 - rect.y0)};
    if (!validRect(cellRect)) return;
    appendToolbarPrimitive(scene,
                           ViewerUiControlKind::ToolbarLayoutPreset,
                           ViewerUiPrimitiveKind::Glyph,
                           cellRect,
                           cellColor);
  });
}

ViewerUiColor plotChromeColor(bool focused, bool hovered) noexcept {
  return {0.018f, 0.026f, 0.034f, focused ? 0.60f : (hovered ? 0.44f : 0.30f)};
}

void appendWindowPrimitive(ViewerUiScene* scene,
                           const ScreenRect& windowRect,
                           ViewerUiControlKind control,
                           int windowId,
                           ViewerUiPrimitiveKind kind,
                           const ScreenRect& rect,
                           ViewerUiColor color,
                           int controlIndex = -1,
                           bool enabled = true,
                           bool selected = false) {
  if (!scene || !validRect(rect)) return;
  const ScreenRect viewport{0.0f,
                            0.0f,
                            static_cast<float>(scene->geometry.windowWidth),
                            static_cast<float>(scene->geometry.windowHeight)};
  const ScreenRect clipped = intersectRect(intersectRect(rect, windowRect), viewport);
  if (!validRect(clipped)) return;
  ViewerUiSolidRect primitive{};
  primitive.rect = clipped;
  primitive.color = color;
  primitive.control = control;
  primitive.windowId = windowId;
  primitive.kind = kind;
  primitive.controlIndex = controlIndex;
  primitive.enabled = enabled;
  primitive.selected = selected;
  scene->primitives.push_back(primitive);
}

void appendWindowHit(ViewerUiScene* scene,
                     ViewerUiWindowScene* window,
                     ViewerUiControlKind control,
                     const ScreenRect& rect,
                     const PlotWindowRectNorm& normalizedRect,
                     bool usesPlotDragGeometry,
                     int controlIndex = -1,
                     bool enabled = true,
                     bool actionable = true,
                     bool selected = false) {
  if (!scene || !window || !validRect(rect)) return;
  const ScreenRect viewport{0.0f,
                            0.0f,
                            static_cast<float>(scene->geometry.windowWidth),
                            static_cast<float>(scene->geometry.windowHeight)};
  const ScreenRect clipped = intersectRect(intersectRect(rect, window->rect), viewport);
  if (!validRect(clipped)) return;
  ViewerUiHitRegion hit{};
  hit.control = control;
  hit.windowId = window->windowId;
  hit.rect = clipped;
  hit.normalizedRect = normalizedRect;
  hit.usesPlotDragGeometry = usesPlotDragGeometry;
  hit.controlIndex = controlIndex;
  hit.enabled = enabled;
  hit.actionable = actionable;
  hit.selected = selected;
  scene->hits.push_back(hit);
}

void appendWindowText(ViewerUiScene* scene,
                      const ScreenRect& windowRect,
                      int windowId,
                      ViewerUiTextIntent intent) {
  if (!scene || !validRect(intent.bounds) || intent.text.empty()) return;
  const ScreenRect viewport{0.0f,
                            0.0f,
                            static_cast<float>(scene->geometry.windowWidth),
                            static_cast<float>(scene->geometry.windowHeight)};
  intent.bounds = intersectRect(intersectRect(intent.bounds, windowRect), viewport);
  if (!validRect(intent.bounds)) return;
  intent.windowId = windowId;
  if (intent.control == ViewerUiControlKind::None) {
    intent.control = ViewerUiControlKind::PlotBody;
  }
  intent.maxWidth = std::max(0.0f, std::min(intent.maxWidth,
                                            intent.bounds.x1 - intent.bounds.x0));
  scene->texts.push_back(std::move(intent));
}

float clampedCoordinate(float value, float low, float high) noexcept {
  return std::clamp(std::isfinite(value) ? value : low, low, high);
}

void appendWindowVectorTriangle(ViewerUiScene* scene,
                                const ScreenRect& windowRect,
                                ViewerUiControlKind control,
                                int windowId,
                                int controlIndex,
                                bool enabled,
                                bool selected,
                                ViewerUiColor color,
                                float x0,
                                float y0,
                                float x1,
                                float y1,
                                float x2,
                                float y2) {
  if (!scene || color.a <= 0.0f) return;
  const ScreenRect viewport{0.0f,
                            0.0f,
                            static_cast<float>(scene->geometry.windowWidth),
                            static_cast<float>(scene->geometry.windowHeight)};
  const ScreenRect clip = intersectRect(windowRect, viewport);
  if (!validRect(clip)) return;
  const auto append = [&](float x, float y) {
    ViewerUiVectorVertex vertex{};
    vertex.x = clampedCoordinate(x, clip.x0, clip.x1);
    vertex.y = clampedCoordinate(y, clip.y0, clip.y1);
    vertex.color = color;
    vertex.control = control;
    vertex.windowId = windowId;
    vertex.controlIndex = controlIndex;
    vertex.enabled = enabled;
    vertex.selected = selected;
    scene->vectors.push_back(vertex);
  };
  append(x0, y0);
  append(x1, y1);
  append(x2, y2);
}

void appendWindowVectorLine(ViewerUiScene* scene,
                            const ScreenRect& windowRect,
                            ViewerUiControlKind control,
                            int windowId,
                            int controlIndex,
                            bool enabled,
                            bool selected,
                            float x0,
                            float y0,
                            float x1,
                            float y1,
                            float thickness,
                            ViewerUiColor color) {
  if (!scene || !std::isfinite(x0) || !std::isfinite(y0) ||
      !std::isfinite(x1) || !std::isfinite(y1) ||
      !std::isfinite(thickness) || thickness <= 0.0f || color.a <= 0.0f) {
    return;
  }
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float length = std::sqrt(dx * dx + dy * dy);
  if (!std::isfinite(length) || length <= 0.001f) return;
  const float half = thickness * 0.5f;
  const float nx = -dy / length * half;
  const float ny = dx / length * half;
  appendWindowVectorTriangle(scene, windowRect, control, windowId,
                             controlIndex, enabled, selected, color,
                             x0 + nx, y0 + ny, x1 + nx, y1 + ny,
                             x1 - nx, y1 - ny);
  appendWindowVectorTriangle(scene, windowRect, control, windowId,
                             controlIndex, enabled, selected, color,
                             x0 + nx, y0 + ny, x1 - nx, y1 - ny,
                             x0 - nx, y0 - ny);
}

template <std::size_t N>
void appendWindowVectorPolyline(
    ViewerUiScene* scene,
    const ScreenRect& windowRect,
    ViewerUiControlKind control,
    int windowId,
    int controlIndex,
    bool enabled,
    bool selected,
    const std::array<std::pair<float, float>, N>& points,
    float thickness,
    ViewerUiColor color) {
  static_assert(N >= 2u, "polyline requires two points");
  for (std::size_t i = 1; i < N; ++i) {
    appendWindowVectorLine(scene, windowRect, control, windowId,
                           controlIndex, enabled, selected,
                           points[i - 1].first, points[i - 1].second,
                           points[i].first, points[i].second,
                           thickness, color);
  }
}

bool validNormalizedRect(const PlotWindowRectNorm& rect) noexcept {
  return std::isfinite(rect.x) && std::isfinite(rect.y) &&
         std::isfinite(rect.w) && std::isfinite(rect.h) && rect.w > 0.0f &&
         rect.h > 0.0f;
}

float derivedTitleHeight(const ViewerUiTitleMetrics& metrics,
                         const ScreenRect& rect) noexcept {
  return viewerUiTitleBarLogicalHeight(rect.y1 - rect.y0,
                                       metrics.titleExtraHeight);
}

}  // namespace

float viewerUiTitleBarLogicalHeight(float windowHeight,
                                    float titleExtraHeight) noexcept {
  windowHeight = std::max(1.0f, std::isfinite(windowHeight) ? windowHeight : 1.0f);
  const float extra = std::isfinite(titleExtraHeight)
                          ? std::max(0.0f, titleExtraHeight)
                          : 0.0f;
  const float target = 24.0f + extra;
  if (windowHeight < 88.0f) {
    const float compactUpper = std::min({20.0f + extra,
                                         windowHeight * 0.34f,
                                         windowHeight});
    const float compactLower = std::min(12.0f, compactUpper);
    return std::min(windowHeight,
                    std::clamp(windowHeight * 0.24f,
                               compactLower,
                               compactUpper));
  }
  return std::min(windowHeight,
                  std::clamp(target,
                             20.0f,
                             std::max(20.0f, windowHeight * 0.22f)));
}

namespace {

PlotWindowDragMode interactionModeFor(const ViewerUiSceneInput& input,
                                       int windowId) noexcept {
  if (input.activeDragWindowId == windowId &&
      input.activeDragMode != PlotWindowDragMode::None) {
    return input.activeDragMode;
  }
  if (input.hoveredWindowId == windowId) return input.hoveredDragMode;
  return PlotWindowDragMode::None;
}

void appendDiagonalBlocks(ViewerUiScene* scene,
                          const ScreenRect& windowRect,
                          int windowId,
                          const ScreenRect& rect,
                          float blockSize,
                          bool reverse,
                          ViewerUiColor color) {
  const float dx = (reverse ? rect.x0 : rect.x1) -
                   (reverse ? rect.x1 : rect.x0);
  const float dy = rect.y1 - rect.y0;
  const float length = std::sqrt(dx * dx + dy * dy);
  if (length <= 0.001f) return;
  const int steps = std::max(2, static_cast<int>(std::ceil(
      length / std::max(1.0f, blockSize * 0.75f))));
  for (int i = 0; i <= steps; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(steps);
    const float x = (reverse ? rect.x1 : rect.x0) + dx * t;
    const float y = rect.y0 + dy * t;
    appendWindowPrimitive(scene,
                          windowRect,
                          ViewerUiControlKind::PlotClose,
                          windowId,
                          ViewerUiPrimitiveKind::Glyph,
                          {x - blockSize * 0.5f,
                           y - blockSize * 0.5f,
                           x + blockSize * 0.5f,
                           y + blockSize * 0.5f},
                          color);
  }
}

ScreenRect scaleRectAround(const ScreenRect& rect,
                           float anchorX,
                           float anchorY,
                           float scale) noexcept {
  return {anchorX + (rect.x0 - anchorX) * scale,
          anchorY + (rect.y0 - anchorY) * scale,
          anchorX + (rect.x1 - anchorX) * scale,
          anchorY + (rect.y1 - anchorY) * scale};
}

ScreenRect slicingQuickButtonRect(const ScreenRect& content) noexcept {
  if (!validRect(content)) return {};
  constexpr float kMargin = 10.0f;
  const float minX = content.x0 + 4.0f;
  const float maxX = std::max(minX,
                              content.x1 - kViewerUiSlicingQuickButtonSize - 4.0f);
  const float minY = content.y0 + 4.0f;
  const float maxY = std::max(minY,
                              content.y1 - kViewerUiSlicingQuickButtonSize - 4.0f);
  const float x0 = std::clamp(content.x0 + kMargin, minX, maxX);
  const float y0 = std::clamp(
      content.y1 - kViewerUiSlicingQuickButtonSize - kMargin, minY, maxY);
  return intersectRect(
      {x0,
       y0,
       x0 + kViewerUiSlicingQuickButtonSize,
       y0 + kViewerUiSlicingQuickButtonSize},
      content);
}

struct SlicingDrawerRects {
  ScreenRect drawer{};
  std::array<ScreenRect, kViewerUiSlicingVectorCount> vectors{};
  ScreenRect lasso{};
};

SlicingDrawerRects slicingDrawerRects(const ScreenRect& content,
                                       const ScreenRect& button) noexcept {
  SlicingDrawerRects out{};
  if (!validRect(content) || !validRect(button)) return out;
  constexpr float kPad = 9.0f;
  constexpr float kGap = 7.0f;
  constexpr float kPreferredItem = 24.0f;
  const float maxWidth = std::max(80.0f, content.x1 - button.x0 - 6.0f);
  const float item = std::clamp(
      (maxWidth - kPad * 2.0f - kGap * 6.0f) / 7.0f,
      14.0f,
      kPreferredItem);
  const float width = std::min(
      maxWidth, kPad * 2.0f + item * 7.0f + kGap * 6.0f);
  const float height = kPad * 2.0f + item;
  const float x0 = button.x0;
  float y1 = button.y0 - 7.0f;
  float y0 = y1 - height;
  if (y0 < content.y0 + 4.0f) {
    y0 = std::min(content.y1 - height - 4.0f, button.y1 + 7.0f);
    y1 = y0 + height;
  }
  out.drawer = intersectRect(
      {x0, y0, std::min(content.x1 - 4.0f, x0 + width), y1}, content);
  if (!validRect(out.drawer)) return out;
  const float available = std::max(
      20.0f, out.drawer.x1 - out.drawer.x0 - kPad * 2.0f);
  const float fittedItem = std::clamp(
      (available - kGap * 6.0f) / 7.0f, 12.0f, 24.0f);
  const float itemY0 = out.drawer.y0 +
                       (out.drawer.y1 - out.drawer.y0 - fittedItem) * 0.5f;
  float x = out.drawer.x0 + kPad;
  for (std::size_t i = 0; i < kViewerUiSlicingVectorCount; ++i) {
    out.vectors[i] = intersectRect(
        {x, itemY0, x + fittedItem, itemY0 + fittedItem}, out.drawer);
    x += fittedItem + kGap;
  }
  out.lasso = intersectRect(
      {x, itemY0, x + fittedItem, itemY0 + fittedItem}, out.drawer);
  return out;
}

ViewerUiColor slicingVectorColor(std::size_t index) noexcept {
  static constexpr ViewerUiColor kColors[kViewerUiSlicingVectorCount] = {
      {1.00f, 0.18f, 0.14f, 1.0f},
      {1.00f, 0.86f, 0.16f, 1.0f},
      {0.22f, 0.86f, 0.22f, 1.0f},
      {0.16f, 0.88f, 0.94f, 1.0f},
      {0.28f, 0.48f, 1.00f, 1.0f},
      {0.92f, 0.32f, 0.94f, 1.0f},
  };
  return kColors[std::min(index, kViewerUiSlicingVectorCount - 1u)];
}

void appendSlicingVectorGlyph(ViewerUiScene* scene,
                              const ScreenRect& windowRect,
                              int windowId,
                              int controlIndex,
                              const ScreenRect& rect,
                              bool enabled,
                              bool hovered,
                              float alpha) {
  if (!validRect(rect) || alpha <= 0.0f) return;
  const ViewerUiColor base = slicingVectorColor(
      static_cast<std::size_t>(std::max(0, controlIndex)));
  const float fillAlpha = enabled ? (hovered ? 0.96f : 0.82f)
                                  : (hovered ? 0.30f : 0.18f);
  appendWindowPrimitive(
      scene, windowRect, ViewerUiControlKind::SlicingVector, windowId,
      ViewerUiPrimitiveKind::SolidRect, rect,
      {base.r * (enabled ? 0.95f : 0.40f),
       base.g * (enabled ? 0.95f : 0.40f),
       base.b * (enabled ? 0.95f : 0.40f), fillAlpha * alpha},
      controlIndex, true, enabled);
  appendWindowPrimitive(
      scene, windowRect, ViewerUiControlKind::SlicingVector, windowId,
      ViewerUiPrimitiveKind::Glyph,
      {rect.x0, rect.y0, rect.x1, std::min(rect.y1, rect.y0 + 1.4f)},
      {1.0f, 1.0f, 1.0f, (enabled ? 0.36f : 0.13f) * alpha},
      controlIndex, true, enabled);
  const float outline = hovered ? 1.7f : 1.1f;
  const ViewerUiColor edge{hovered ? 0.95f : 0.52f,
                           hovered ? 0.98f : 0.64f,
                           hovered ? 1.00f : 0.72f,
                           (hovered ? 0.90f : 0.44f) * alpha};
  appendWindowPrimitive(scene, windowRect,
                        ViewerUiControlKind::SlicingVector, windowId,
                        ViewerUiPrimitiveKind::Glyph,
                        {rect.x0, rect.y0, rect.x1, rect.y0 + outline}, edge,
                        controlIndex, true, enabled);
  appendWindowPrimitive(scene, windowRect,
                        ViewerUiControlKind::SlicingVector, windowId,
                        ViewerUiPrimitiveKind::Glyph,
                        {rect.x0, rect.y1 - outline, rect.x1, rect.y1}, edge,
                        controlIndex, true, enabled);
  appendWindowPrimitive(scene, windowRect,
                        ViewerUiControlKind::SlicingVector, windowId,
                        ViewerUiPrimitiveKind::Glyph,
                        {rect.x0, rect.y0, rect.x0 + outline, rect.y1}, edge,
                        controlIndex, true, enabled);
  appendWindowPrimitive(scene, windowRect,
                        ViewerUiControlKind::SlicingVector, windowId,
                        ViewerUiPrimitiveKind::Glyph,
                        {rect.x1 - outline, rect.y0, rect.x1, rect.y1}, edge,
                        controlIndex, true, enabled);
}

void appendSlicingQuickGlyph(ViewerUiScene* scene,
                             const ScreenRect& windowRect,
                             int windowId,
                             const ScreenRect& rect,
                             bool selected,
                             bool hovered) {
  if (!validRect(rect)) return;
  appendWindowPrimitive(
      scene, windowRect, ViewerUiControlKind::SlicingQuickToggle, windowId,
      ViewerUiPrimitiveKind::SolidRect, rect,
      {hovered ? 0.07f : 0.035f, hovered ? 0.16f : 0.10f,
       hovered ? 0.22f : 0.15f, hovered ? 0.86f : 0.68f},
      -1, true, selected);
  if (selected) {
    appendWindowPrimitive(
        scene, windowRect, ViewerUiControlKind::SlicingQuickToggle, windowId,
        ViewerUiPrimitiveKind::Glyph,
        {rect.x0, rect.y0, rect.x1, std::min(rect.y1, rect.y0 + 2.0f)},
        {0.42f, 0.86f, 1.0f, 0.64f}, -1, true, true);
  }
  const float cx = (rect.x0 + rect.x1) * 0.5f;
  const float cy = (rect.y0 + rect.y1) * 0.5f;
  const float radius = std::min(rect.x1 - rect.x0, rect.y1 - rect.y0) * 0.30f;
  const float block = std::max(1.4f, radius * 0.20f);
  appendWindowPrimitive(
      scene, windowRect, ViewerUiControlKind::SlicingQuickToggle, windowId,
      ViewerUiPrimitiveKind::Glyph,
      {cx - radius * 0.42f, cy - radius * 0.42f,
       cx + radius * 0.42f, cy + radius * 0.42f},
      {selected ? 0.32f : 0.18f, selected ? 0.56f : 0.30f,
       selected ? 0.72f : 0.40f, hovered ? 0.72f : 0.54f},
      -1, true, selected);
  for (int i = 0; i < 6; ++i) {
    const float angle = static_cast<float>(i) * 6.28318530718f / 6.0f;
    const float x = cx + std::cos(angle) * radius;
    const float y = cy + std::sin(angle) * radius;
    appendWindowPrimitive(
        scene, windowRect, ViewerUiControlKind::SlicingQuickToggle, windowId,
        ViewerUiPrimitiveKind::Glyph,
        {x - block * 0.5f, y - block * 0.5f,
         x + block * 0.5f, y + block * 0.5f},
        {0.78f, 0.91f, 1.0f, hovered ? 0.92f : 0.66f},
        -1, true, selected);
    appendWindowVectorLine(
        scene, windowRect, ViewerUiControlKind::SlicingQuickToggle, windowId,
        -1, true, selected, cx, cy, x, y, std::max(1.0f, block * 0.55f),
        {0.78f, 0.91f, 1.0f, hovered ? 0.72f : 0.42f});
  }
}

void appendSlicingLassoGlyph(ViewerUiScene* scene,
                             const ScreenRect& windowRect,
                             int windowId,
                             const ScreenRect& rect,
                             bool selected,
                             bool hovered,
                             float alpha) {
  if (!validRect(rect) || alpha <= 0.0f) return;
  appendWindowPrimitive(
      scene, windowRect, ViewerUiControlKind::SlicingLasso, windowId,
      ViewerUiPrimitiveKind::SolidRect, rect,
      {selected ? 0.07f : 0.035f, selected ? 0.17f : 0.09f,
       selected ? 0.23f : 0.13f, (hovered ? 0.86f : 0.64f) * alpha},
      -1, true, selected);
  if (selected) {
    appendWindowPrimitive(
        scene, windowRect, ViewerUiControlKind::SlicingLasso, windowId,
        ViewerUiPrimitiveKind::Glyph,
        {rect.x0, rect.y0, rect.x1, std::min(rect.y1, rect.y0 + 1.5f)},
        {0.40f, 0.86f, 1.0f, 0.58f * alpha}, -1, true, true);
  }
  const float w = rect.x1 - rect.x0;
  const float h = rect.y1 - rect.y0;
  const float cx = (rect.x0 + rect.x1) * 0.5f;
  const float cy = (rect.y0 + rect.y1) * 0.5f;
  const float thickness = hovered ? 2.0f : 1.55f;
  const std::array<std::pair<float, float>, 11> body{{
      {cx - w * 0.27f, cy + h * 0.06f},
      {cx - w * 0.22f, cy + h * 0.20f},
      {cx - w * 0.06f, cy + h * 0.30f},
      {cx + w * 0.14f, cy + h * 0.25f},
      {cx + w * 0.27f, cy + h * 0.10f},
      {cx + w * 0.23f, cy - h * 0.09f},
      {cx + w * 0.05f, cy - h * 0.20f},
      {cx - w * 0.15f, cy - h * 0.16f},
      {cx - w * 0.27f, cy + h * 0.06f},
      {cx - w * 0.13f, cy - h * 0.06f},
      {cx + w * 0.22f, cy - h * 0.35f},
  }};
  appendWindowVectorPolyline(
      scene, windowRect, ViewerUiControlKind::SlicingLasso, windowId,
      -1, true, selected, body, thickness,
      {0.72f, 0.92f, 1.0f, (hovered ? 0.95f : 0.68f) * alpha});
  const std::array<std::pair<float, float>, 3> tail{{
      {cx + w * 0.16f, cy - h * 0.32f},
      {cx + w * 0.28f, cy - h * 0.38f},
      {cx + w * 0.36f, cy - h * 0.30f},
  }};
  appendWindowVectorPolyline(
      scene, windowRect, ViewerUiControlKind::SlicingLasso, windowId,
      -1, true, selected, tail, std::max(1.2f, thickness * 0.82f),
      {0.72f, 0.92f, 1.0f, (hovered ? 0.68f : 0.44f) * alpha});
}

void appendSlicingControls(ViewerUiScene* scene,
                           ViewerUiWindowScene* window,
                           const ViewerUiPlotWindowInput& requested,
                           const ViewerUiSceneInput& input) {
  if (!scene || !window || !requested.slicing.visible ||
      !validRect(window->contentRect)) {
    return;
  }
  const ScreenRect button = slicingQuickButtonRect(window->contentRect);
  if (!validRect(button)) return;
  const bool toggleSelected = requested.slicing.active ||
                              requested.slicing.drawerOpen;
  const bool toggleHovered = input.hasPointer &&
      viewerUiRectContainsInclusive(button, input.pointerX, input.pointerY);
  appendSlicingQuickGlyph(scene, window->rect, window->windowId, button,
                          toggleSelected, toggleHovered);
  appendWindowHit(scene, window, ViewerUiControlKind::SlicingQuickToggle,
                  button, window->normalizedRect, false,
                  -1, true, true, toggleSelected);
  if (!requested.slicing.drawerOpen) return;

  const SlicingDrawerRects full = slicingDrawerRects(window->contentRect, button);
  if (!validRect(full.drawer)) return;
  const float progress = requested.slicing.animationProgress;
  const float scale = 0.72f + 0.28f * progress;
  const float anchorX = (button.x0 + button.x1) * 0.5f;
  const float anchorY = (button.y0 + button.y1) * 0.5f;
  const ScreenRect drawer = intersectRect(
      scaleRectAround(full.drawer, anchorX, anchorY, scale),
      window->contentRect);
  if (!validRect(drawer)) return;
  appendWindowPrimitive(
      scene, window->rect, ViewerUiControlKind::SlicingQuickToggle,
      window->windowId, ViewerUiPrimitiveKind::SolidRect, drawer,
      {0.014f, 0.018f, 0.024f, 0.88f * progress},
      -1, true, true);
  appendWindowPrimitive(
      scene, window->rect, ViewerUiControlKind::SlicingQuickToggle,
      window->windowId, ViewerUiPrimitiveKind::Glyph,
      {drawer.x0, drawer.y0, drawer.x1,
       std::min(drawer.y1, drawer.y0 + 1.0f)},
      {0.42f, 0.76f, 0.95f, 0.32f * progress},
      -1, true, true);
  const bool controlsActionable = progress > 0.001f;
  for (std::size_t i = 0; i < kViewerUiSlicingVectorCount; ++i) {
    const ScreenRect rect = intersectRect(
        scaleRectAround(full.vectors[i], anchorX, anchorY, scale),
        window->contentRect);
    if (!validRect(rect)) continue;
    const bool hovered = input.hasPointer &&
        viewerUiRectContainsInclusive(rect, input.pointerX, input.pointerY);
    appendSlicingVectorGlyph(scene, window->rect, window->windowId,
                             static_cast<int>(i), rect,
                             requested.slicing.vectors[i], hovered, progress);
    appendWindowHit(scene, window, ViewerUiControlKind::SlicingVector,
                    rect, window->normalizedRect, false,
                    static_cast<int>(i), true, controlsActionable,
                    requested.slicing.vectors[i]);
  }
  const ScreenRect lasso = intersectRect(
      scaleRectAround(full.lasso, anchorX, anchorY, scale),
      window->contentRect);
  if (validRect(lasso)) {
    const bool hovered = input.hasPointer &&
        viewerUiRectContainsInclusive(lasso, input.pointerX, input.pointerY);
    appendSlicingLassoGlyph(scene, window->rect, window->windowId, lasso,
                            requested.slicing.lassoActive, hovered, progress);
    appendWindowHit(scene, window, ViewerUiControlKind::SlicingLasso,
                    lasso, window->normalizedRect, false,
                    -1, true, controlsActionable,
                    requested.slicing.lassoActive);
  }
}

ScreenRect sourceLassoButtonRect(const ScreenRect& content,
                                 int index) noexcept {
  constexpr float kButtonSize = 26.0f;
  constexpr float kGap = 6.0f;
  constexpr float kLeft = 10.0f;
  constexpr float kTop = 34.0f;
  const float x0 = content.x0 + kLeft +
                   static_cast<float>(index) * (kButtonSize + kGap);
  const float y0 = content.y0 + kTop;
  return intersectRect({x0, y0, x0 + kButtonSize, y0 + kButtonSize},
                       content);
}

void appendSourceLassoControls(ViewerUiScene* scene,
                               ViewerUiWindowScene* window,
                               const ViewerUiPlotWindowInput& requested,
                               const ViewerUiSceneInput& input) {
  if (!scene || !window || !requested.sourceLasso.visible ||
      !validRect(window->contentRect)) {
    return;
  }
  const ViewerUiControlKind controls[3] = {
      ViewerUiControlKind::SourceLassoAdd,
      ViewerUiControlKind::SourceLassoSubtract,
      ViewerUiControlKind::SourceLassoClear,
  };
  const char* labels[3] = {"+", "-", u8"\u21BA"};
  for (int i = 0; i < 3; ++i) {
    const ScreenRect rect = sourceLassoButtonRect(window->contentRect, i);
    if (!validRect(rect)) continue;
    const bool enabled = i != 2 || requested.sourceLasso.hasSelection;
    const bool selected = i == 0 ? !requested.sourceLasso.subtract
                                  : (i == 1 ? requested.sourceLasso.subtract
                                            : requested.sourceLasso.hasSelection);
    const bool hovered = enabled && input.hasPointer &&
        viewerUiRectContainsInclusive(rect, input.pointerX, input.pointerY);
    const float alpha = enabled ? (hovered ? 0.90f : 0.70f) : 0.30f;
    const float gain = selected ? 1.0f : 0.62f;
    appendWindowPrimitive(
        scene, window->rect, controls[i], window->windowId,
        ViewerUiPrimitiveKind::SolidRect, rect,
        {i == 2 ? 0.18f * gain : 0.07f * gain,
         i == 2 ? 0.13f * gain : 0.16f * gain,
         i == 2 ? 0.15f * gain : 0.22f * gain,
         alpha * 0.42f}, -1, enabled, selected);
    const ViewerUiColor edge{hovered ? 0.72f : 0.44f,
                             hovered ? 0.92f : 0.58f,
                             hovered ? 1.00f : 0.70f,
                             alpha * (hovered ? 0.54f : 0.28f)};
    const float outline = hovered ? 1.3f : 0.9f;
    appendWindowPrimitive(scene, window->rect, controls[i], window->windowId,
                          ViewerUiPrimitiveKind::Glyph,
                          {rect.x0, rect.y0, rect.x1, rect.y0 + outline}, edge,
                          -1, enabled, selected);
    appendWindowPrimitive(scene, window->rect, controls[i], window->windowId,
                          ViewerUiPrimitiveKind::Glyph,
                          {rect.x0, rect.y1 - outline, rect.x1, rect.y1}, edge,
                          -1, enabled, selected);
    appendWindowPrimitive(scene, window->rect, controls[i], window->windowId,
                          ViewerUiPrimitiveKind::Glyph,
                          {rect.x0, rect.y0, rect.x0 + outline, rect.y1}, edge,
                          -1, enabled, selected);
    appendWindowPrimitive(scene, window->rect, controls[i], window->windowId,
                          ViewerUiPrimitiveKind::Glyph,
                          {rect.x1 - outline, rect.y0, rect.x1, rect.y1}, edge,
                          -1, enabled, selected);

    ViewerUiTextIntent label{};
    label.visible = true;
    label.text = labels[i];
    label.bounds = rect;
    label.originX = (rect.x0 + rect.x1) * 0.5f;
    label.originY = rect.y0 + (rect.y1 - rect.y0) * 0.5f + 4.0f;
    label.maxWidth = std::max(0.0f, rect.x1 - rect.x0 - 4.0f);
    label.alignment = ViewerUiTextAlignment::Center;
    label.scale = i == 2 ? 1.06f : 1.16f;
    label.color = {enabled ? 0.92f : 0.40f,
                   enabled ? 0.98f : 0.48f,
                   enabled ? 1.00f : 0.54f,
                   alpha};
    label.control = controls[i];
    label.enabled = enabled;
    label.selected = selected;
    appendWindowText(scene, window->rect, window->windowId, std::move(label));
    appendWindowHit(scene, window, controls[i], rect, window->normalizedRect,
                    false, -1, enabled, enabled, selected);
  }
}

void appendSourceSignalRestoreControls(
    ViewerUiScene* scene,
    ViewerUiWindowScene* window,
    const ViewerUiPlotWindowInput& requested,
    const ViewerUiSceneInput& input) {
  if (!scene || !window || requested.sourceSignalRestoreWindowIds.empty() ||
      !validRect(window->contentRect)) {
    return;
  }
  constexpr float kButtonSize = 30.0f;
  constexpr float kGap = 6.0f;
  constexpr float kMargin = 10.0f;
  const float firstX = window->contentRect.x0 + kMargin +
      (requested.slicing.visible
           ? kViewerUiSlicingQuickButtonSize + kGap
           : 0.0f);
  const float y0 = std::max(
      window->contentRect.y0 + 4.0f,
      window->contentRect.y1 - kButtonSize - kMargin);
  for (std::size_t index = 0u;
       index < requested.sourceSignalRestoreWindowIds.size(); ++index) {
    const int sourceWindowId =
        requested.sourceSignalRestoreWindowIds[index];
    const float x0 = firstX + static_cast<float>(index) *
                                  (kButtonSize + kGap);
    const ScreenRect rect = intersectRect(
        {x0, y0, x0 + kButtonSize, y0 + kButtonSize},
        window->contentRect);
    if (!validRect(rect)) continue;
    const bool hovered = input.hasPointer &&
        viewerUiRectContainsInclusive(rect, input.pointerX, input.pointerY);
    appendWindowPrimitive(
        scene, window->rect, ViewerUiControlKind::SourceSignalRestore,
        window->windowId, ViewerUiPrimitiveKind::SolidRect, rect,
        {0.04f, 0.13f, 0.18f, hovered ? 0.88f : 0.68f},
        sourceWindowId, true, false);
    const float edge = hovered ? 1.4f : 1.0f;
    const ViewerUiColor edgeColor{0.28f, 0.80f, 1.0f,
                                  hovered ? 0.92f : 0.62f};
    appendWindowPrimitive(
        scene, window->rect, ViewerUiControlKind::SourceSignalRestore,
        window->windowId, ViewerUiPrimitiveKind::Glyph,
        {rect.x0, rect.y0, rect.x1, rect.y0 + edge}, edgeColor,
        sourceWindowId, true, false);
    appendWindowPrimitive(
        scene, window->rect, ViewerUiControlKind::SourceSignalRestore,
        window->windowId, ViewerUiPrimitiveKind::Glyph,
        {rect.x0, rect.y1 - edge, rect.x1, rect.y1}, edgeColor,
        sourceWindowId, true, false);
    ViewerUiTextIntent label{};
    label.visible = true;
    label.text = "SS";
    label.bounds = rect;
    label.originX = (rect.x0 + rect.x1) * 0.5f;
    label.originY = (rect.y0 + rect.y1) * 0.5f + 4.0f;
    label.maxWidth = std::max(0.0f, rect.x1 - rect.x0 - 4.0f);
    label.alignment = ViewerUiTextAlignment::Center;
    label.scale = 0.78f;
    label.color = {0.72f, 0.94f, 1.0f, hovered ? 1.0f : 0.82f};
    label.control = ViewerUiControlKind::SourceSignalRestore;
    label.controlIndex = sourceWindowId;
    appendWindowText(scene, window->rect, window->windowId,
                     std::move(label));
    appendWindowHit(scene, window,
                    ViewerUiControlKind::SourceSignalRestore, rect,
                    window->normalizedRect, false, sourceWindowId,
                    true, true, false);
  }
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

}  // namespace

bool viewerUiRectContainsInclusive(const ScreenRect& rect,
                                   float logicalX,
                                   float logicalY) noexcept {
  return validRect(rect) && std::isfinite(logicalX) &&
         std::isfinite(logicalY) && logicalX >= rect.x0 &&
         logicalX <= rect.x1 && logicalY >= rect.y0 && logicalY <= rect.y1;
}

WorkspaceToolbarScene buildWorkspaceToolbarScene(
    const WorkspaceToolbarInput& input) {
  WorkspaceToolbarScene scene{};
  scene.logicalWidth = std::max(0, input.logicalWidth);
  scene.logicalHeight = std::max(0, input.logicalHeight);
  scene.reservedLeftPixels = std::isfinite(input.reservedLeftPixels)
                                 ? std::clamp(input.reservedLeftPixels,
                                              0.0f,
                                              static_cast<float>(scene.logicalWidth))
                                 : 0.0f;
  scene.textScale = std::isfinite(input.textScale) && input.textScale > 0.0f
                        ? std::clamp(input.textScale, 0.5f, 4.0f)
                        : 1.0f;
  scene.visible = input.visible && scene.logicalWidth > 0 &&
                  scene.logicalHeight > 0;
  if (!scene.visible) return scene;

  const float left = scene.reservedLeftPixels + kWorkspaceToolbarInset;
  const float y0 = kWorkspaceToolbarInset;
  const float y1 = y0 + kWorkspaceToolbarButtonSize;
  if (y1 > static_cast<float>(scene.logicalHeight)) return scene;
  if (left + kWorkspaceToolbarButtonSize <=
      static_cast<float>(scene.logicalWidth)) {
    scene.controls[0] = {ViewerUiControlKind::ToolbarMenu,
                         -1,
                         {left, y0, left + kWorkspaceToolbarButtonSize, y1},
                         {},
                         false};
  }
  const float addX0 = left + kWorkspaceToolbarButtonSize +
                      kWorkspaceToolbarGap;
  const float addX1 = addX0 + kWorkspaceToolbarButtonSize;
  if (addX1 <= static_cast<float>(scene.logicalWidth)) {
    scene.controls[1] = {ViewerUiControlKind::ToolbarAddPlot,
                         -1,
                         {addX0, y0, addX1, y1},
                         {},
                         false};
  }

  const float layoutX = left + 2.0f *
                                   (kWorkspaceToolbarButtonSize +
                                    kWorkspaceToolbarGap);
  const float maxX = static_cast<float>(scene.logicalWidth) -
                     kWorkspaceToolbarRightMargin;
  if (layoutX + kWorkspaceToolbarButtonSize <= maxX) {
    const float layoutX1 = std::min(
        maxX, layoutX + kWorkspaceToolbarLayoutLabelWidth * scene.textScale);
    scene.controls[2] = {ViewerUiControlKind::ToolbarLayoutPreset,
                         -1,
                         {layoutX,
                          y0,
                          std::max(layoutX + kWorkspaceToolbarButtonSize, layoutX1),
                          y1},
                         {},
                         false};
  }

  const bool menuHovered = input.hasPointer && viewerUiRectContainsInclusive(
                                                  scene.controls[0].rect,
                                                  input.pointerX,
                                                  input.pointerY);
  const bool addHovered = input.hasPointer && viewerUiRectContainsInclusive(
                                                 scene.controls[1].rect,
                                                 input.pointerX,
                                                 input.pointerY);
  const bool layoutHovered = input.hasPointer && viewerUiRectContainsInclusive(
                                                    scene.controls[2].rect,
                                                    input.pointerX,
                                                    input.pointerY);
  appendToolbarButtonFrame(&scene, ViewerUiControlKind::ToolbarMenu,
                           input.menuActive, menuHovered);
  appendToolbarButtonFrame(&scene, ViewerUiControlKind::ToolbarAddPlot,
                           input.addPlotActive, addHovered);
  appendToolbarButtonFrame(&scene, ViewerUiControlKind::ToolbarLayoutPreset,
                           input.layoutActive, layoutHovered);
  appendToolbarMenuGlyph(&scene, scene.controls[0].rect, input.menuActive,
                         menuHovered);
  appendToolbarAddGlyph(&scene, scene.controls[1].rect, input.addPlotActive,
                        addHovered);

  const int layoutIndex = input.layoutIndex < -1
                              ? 0
                              : std::min(input.layoutIndex,
                                         kViewerLayoutChoiceCount - 1);
  const ScreenRect layoutRect = scene.controls[2].rect;
  if (validRect(layoutRect)) {
    appendToolbarLayoutGlyph(&scene,
                             {layoutRect.x0 + 7.0f,
                              layoutRect.y0 + 7.0f,
                              layoutRect.x0 + 28.0f,
                              layoutRect.y1 - 7.0f},
                             layoutIndex,
                             input.layoutActive,
                             layoutHovered);
    scene.layoutLabel.text = input.layoutLabel;
    scene.layoutLabel.bounds = layoutRect;
    scene.layoutLabel.originX = layoutRect.x0 + 34.0f;
    const float labelScale = 0.78f * scene.textScale;
    const float baselineOffset = scene.textScale <= 1.01f
                                     ? 8.0f
                                     : std::max(
                                           8.0f,
                                           (layoutRect.y1 - layoutRect.y0) *
                                                   0.5f -
                                               6.0f * labelScale);
    scene.layoutLabel.originY = layoutRect.y1 - baselineOffset;
    scene.layoutLabel.maxWidth = std::max(0.0f,
                                          layoutRect.x1 - scene.layoutLabel.originX -
                                              4.0f);
    scene.layoutLabel.alignment = ViewerUiTextAlignment::Left;
    scene.layoutLabel.scale = labelScale;
    scene.layoutLabel.color = {0.78f, 0.92f, 0.98f, 0.96f};
    scene.layoutLabel.control = ViewerUiControlKind::ToolbarLayoutPreset;
    scene.layoutLabel.visible = !scene.layoutLabel.text.empty();
  }
  return scene;
}

ViewerUiControlKind workspaceToolbarHitTest(
    const WorkspaceToolbarScene& scene,
    float logicalX,
    float logicalY) noexcept {
  if (!scene.visible || !std::isfinite(logicalX) ||
      !std::isfinite(logicalY)) {
    return ViewerUiControlKind::None;
  }
  for (const ViewerUiHitRegion& region : scene.controls) {
    if (viewerUiRectContainsInclusive(region.rect, logicalX, logicalY)) {
      return region.control;
    }
  }
  return ViewerUiControlKind::None;
}

ViewerUiScene buildViewerUiScene(const ViewerFramePlan& plan,
                                 const ViewerUiSceneInput& input) {
  ViewerUiScene scene{};
  scene.geometry = plan.geometry;
  if (!plan.ready()) {
    scene.status = ViewerUiSceneStatus::InvalidFramePlan;
    return scene;
  }
  if (plan.geometry.windowWidth <= 0 || plan.geometry.windowHeight <= 0 ||
      !std::isfinite(plan.geometry.scaleX) ||
      !std::isfinite(plan.geometry.scaleY) || plan.geometry.scaleX <= 0.0f ||
      plan.geometry.scaleY <= 0.0f) {
    scene.status = ViewerUiSceneStatus::InvalidViewport;
    return scene;
  }
  if (input.windows.size() != plan.windows.size()) {
    scene.status = ViewerUiSceneStatus::WindowCountMismatch;
    return scene;
  }
  for (std::size_t i = 0; i < plan.windows.size(); ++i) {
    const ViewerUiPlotWindowInput& requested = input.windows[i];
    if (requested.windowId != plan.windows[i].windowId) {
      scene.status = ViewerUiSceneStatus::WindowIdMismatch;
      return scene;
    }
    if (requested.windowId <= 0 ||
        !validNormalizedRect(plan.windows[i].normalizedRect) ||
        !std::isfinite(requested.slicing.animationProgress) ||
        requested.slicing.animationProgress < 0.0f ||
        requested.slicing.animationProgress > 1.0f) {
      scene.status = ViewerUiSceneStatus::InvalidWindowInput;
      return scene;
    }
    for (std::size_t restoreIndex = 0u;
         restoreIndex < requested.sourceSignalRestoreWindowIds.size();
         ++restoreIndex) {
      const int sourceId =
          requested.sourceSignalRestoreWindowIds[restoreIndex];
      if (sourceId <= 0 || sourceId == requested.windowId) {
        scene.status = ViewerUiSceneStatus::InvalidWindowInput;
        return scene;
      }
      for (std::size_t prior = 0u; prior < restoreIndex; ++prior) {
        if (requested.sourceSignalRestoreWindowIds[prior] == sourceId) {
          scene.status = ViewerUiSceneStatus::InvalidWindowInput;
          return scene;
        }
      }
    }
  }

  const std::size_t windowReserve = plan.windows.size();
  scene.primitives.reserve(1u + windowReserve * 80u);
  scene.texts.reserve(1u + windowReserve * 5u);
  scene.vectors.reserve(windowReserve * 96u);
  scene.hits.reserve(3u + windowReserve * 12u);

  WorkspaceToolbarInput toolbarInput = input.toolbar;
  toolbarInput.logicalWidth = plan.geometry.windowWidth;
  toolbarInput.logicalHeight = plan.geometry.windowHeight;
  toolbarInput.reservedLeftPixels = plan.geometry.reservedLeftPixels;
  toolbarInput.hasPointer = input.hasPointer;
  toolbarInput.pointerX = input.pointerX;
  toolbarInput.pointerY = input.pointerY;
  scene.toolbar = buildWorkspaceToolbarScene(toolbarInput);
  for (const ViewerUiSolidRect& primitive : scene.toolbar.primitives) {
    scene.primitives.push_back(primitive);
  }
  for (const ViewerUiHitRegion& region : scene.toolbar.controls) {
    if (region.control != ViewerUiControlKind::None && validRect(region.rect)) {
      scene.hits.push_back(region);
    }
  }
  if (scene.toolbar.layoutLabel.visible &&
      !scene.toolbar.layoutLabel.text.empty()) {
    scene.texts.push_back(scene.toolbar.layoutLabel);
  }

  const ScreenRect viewport{0.0f,
                            0.0f,
                            static_cast<float>(plan.geometry.windowWidth),
                            static_cast<float>(plan.geometry.windowHeight)};
  scene.windows.reserve(plan.windows.size());
  for (std::size_t i = 0; i < plan.windows.size(); ++i) {
    const ViewerFramePlanWindow& planned = plan.windows[i];
    const ViewerUiPlotWindowInput& requested = input.windows[i];
    const ScreenRect windowRect = intersectRect(
        {planned.logicalRect.x0,
         planned.logicalRect.y0,
         planned.logicalRect.x1,
         planned.logicalRect.y1},
        viewport);
    if (!validRect(windowRect)) {
      scene.status = ViewerUiSceneStatus::InvalidWindowInput;
      scene.windows.clear();
      scene.primitives.clear();
      scene.texts.clear();
      scene.vectors.clear();
      scene.hits.clear();
      scene.toolbar = WorkspaceToolbarScene{};
      return scene;
    }

    ViewerUiWindowScene window{};
    window.windowId = requested.windowId;
    window.rect = windowRect;
    window.normalizedRect = planned.normalizedRect;
    window.closable = requested.closable && plan.windows.size() > 1u;
    window.primitiveBegin = scene.primitives.size();
    window.textBegin = scene.texts.size();
    window.vectorBegin = scene.vectors.size();
    window.hitBegin = scene.hits.size();

    const bool focused = requested.windowId == input.focusedWindowId;
    const bool hovered = requested.windowId == input.hoveredWindowId;
    const PlotWindowDragMode interactionMode = interactionModeFor(
        input, requested.windowId);
    const bool activeDrag = input.activeDragWindowId == requested.windowId &&
                            input.activeDragMode != PlotWindowDragMode::None;
    const float titleHeight = std::clamp(
        derivedTitleHeight(requested.titleMetrics, windowRect),
        1.0f,
        std::max(1.0f, windowRect.y1 - windowRect.y0));
    window.contentRect = intersectRect(
        {windowRect.x0,
         std::min(windowRect.y1, windowRect.y0 + titleHeight),
         windowRect.x1,
         windowRect.y1},
        windowRect);
    const float border = 1.5f;
    const ViewerUiColor borderColor{0.62f, 0.82f, 1.0f,
                                    focused ? 0.86f : (hovered ? 0.58f : 0.34f)};
    appendWindowPrimitive(&scene,
                          windowRect,
                          ViewerUiControlKind::PlotBody,
                          requested.windowId,
                          ViewerUiPrimitiveKind::SolidRect,
                          {windowRect.x0,
                           windowRect.y0,
                           windowRect.x1,
                           windowRect.y0 + titleHeight},
                          plotChromeColor(focused, hovered));
    appendWindowPrimitive(&scene,
                          windowRect,
                          ViewerUiControlKind::PlotBody,
                          requested.windowId,
                          ViewerUiPrimitiveKind::SolidRect,
                          {windowRect.x0, windowRect.y0, windowRect.x1,
                           windowRect.y0 + border},
                          borderColor);
    appendWindowPrimitive(&scene,
                          windowRect,
                          ViewerUiControlKind::PlotBody,
                          requested.windowId,
                          ViewerUiPrimitiveKind::SolidRect,
                          {windowRect.x0, windowRect.y1 - border, windowRect.x1,
                           windowRect.y1},
                          {0.62f, 0.82f, 1.0f, borderColor.a * 0.72f});
    appendWindowPrimitive(&scene,
                          windowRect,
                          ViewerUiControlKind::PlotBody,
                          requested.windowId,
                          ViewerUiPrimitiveKind::SolidRect,
                          {windowRect.x0, windowRect.y0, windowRect.x0 + border,
                           windowRect.y1},
                          {0.62f, 0.82f, 1.0f, borderColor.a * 0.72f});
    appendWindowPrimitive(&scene,
                          windowRect,
                          ViewerUiControlKind::PlotBody,
                          requested.windowId,
                          ViewerUiPrimitiveKind::SolidRect,
                          {windowRect.x1 - border, windowRect.y0, windowRect.x1,
                           windowRect.y1},
                          {0.62f, 0.82f, 1.0f, borderColor.a * 0.68f});

    if (interactionMode == PlotWindowDragMode::Move) {
      const float moveAlpha = activeDrag ? 0.48f : 0.34f;
      appendWindowPrimitive(
          &scene,
          windowRect,
          ViewerUiControlKind::PlotBody,
          requested.windowId,
          ViewerUiPrimitiveKind::SolidRect,
          {windowRect.x0,
           windowRect.y0,
           windowRect.x1,
           std::min(windowRect.y1, windowRect.y0 + std::min(5.0f, titleHeight))},
          {0.20f, 0.46f, 0.64f, moveAlpha});
      const float gripX = windowRect.x0 + 10.0f;
      const float gripY = windowRect.y0 + titleHeight * 0.5f;
      appendWindowPrimitive(&scene,
                            windowRect,
                            ViewerUiControlKind::PlotBody,
                            requested.windowId,
                            ViewerUiPrimitiveKind::SolidRect,
                            {gripX - 4.0f,
                             gripY - 8.0f,
                             gripX + 24.0f,
                             gripY + 8.0f},
                            {0.05f,
                             0.13f,
                             0.18f,
                             activeDrag ? 0.66f : 0.48f});
      const float lineHeight = activeDrag ? 2.2f : 1.8f;
      for (int line = 0; line < 3; ++line) {
        const float y = gripY + static_cast<float>(line - 1) * 4.0f;
        appendWindowPrimitive(
            &scene,
            windowRect,
            ViewerUiControlKind::PlotBody,
            requested.windowId,
            ViewerUiPrimitiveKind::Glyph,
            {gripX,
             y - lineHeight * 0.5f,
             gripX + 18.0f,
             y + lineHeight * 0.5f},
            {0.82f, 0.95f, 1.0f, activeDrag ? 0.98f : 0.86f});
      }
    } else if (plotWindowDragModeTraits(interactionMode).isResize) {
      const PlotWindowDragTraits traits = plotWindowDragModeTraits(interactionMode);
      const float edgeAlpha = activeDrag ? 0.94f : 0.72f;
      const float glowAlpha = activeDrag ? 0.22f : 0.13f;
      const float thickness = activeDrag ? 5.0f : 4.0f;
      const ViewerUiColor edge{0.36f, 0.78f, 1.0f, edgeAlpha};
      const ViewerUiColor glow{0.18f, 0.56f, 0.86f, glowAlpha};
      if (traits.touchesTop) {
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y0, windowRect.x1,
                               windowRect.y0 + thickness}, edge);
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y0, windowRect.x1,
                               windowRect.y0 + 12.0f}, glow);
      }
      if (traits.touchesBottom) {
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y1 - thickness,
                               windowRect.x1, windowRect.y1}, edge);
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y1 - 12.0f,
                               windowRect.x1, windowRect.y1}, glow);
      }
      if (traits.touchesLeft) {
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y0,
                               windowRect.x0 + thickness, windowRect.y1}, edge);
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x0, windowRect.y0,
                               windowRect.x0 + 12.0f, windowRect.y1}, glow);
      }
      if (traits.touchesRight) {
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x1 - thickness, windowRect.y0,
                               windowRect.x1, windowRect.y1}, edge);
        appendWindowPrimitive(&scene, windowRect, ViewerUiControlKind::PlotBody,
                              requested.windowId, ViewerUiPrimitiveKind::SolidRect,
                              {windowRect.x1 - 12.0f, windowRect.y0,
                               windowRect.x1, windowRect.y1}, glow);
      }
    }

    if (window.closable) {
      const ScreenRect closeRect{windowRect.x1 - 23.0f,
                                 windowRect.y0 + 4.0f,
                                 windowRect.x1 - 5.0f,
                                 windowRect.y0 + 22.0f};
      appendWindowHit(&scene,
                      &window,
                      ViewerUiControlKind::PlotClose,
                      closeRect,
                      planned.normalizedRect,
                      false);
      const bool closeHovered = input.hasPointer &&
                                viewerUiRectContainsInclusive(closeRect,
                                                               input.pointerX,
                                                               input.pointerY);
      appendWindowPrimitive(&scene,
                            windowRect,
                            ViewerUiControlKind::PlotClose,
                            requested.windowId,
                            ViewerUiPrimitiveKind::SolidRect,
                            closeRect,
                            {closeHovered ? 0.20f : 0.05f,
                             closeHovered ? 0.08f : 0.06f,
                             closeHovered ? 0.09f : 0.08f,
                             closeHovered ? 0.82f : 0.46f});
      const float inset = 5.0f;
      const float block = closeHovered ? 2.4f : 2.0f;
      appendDiagonalBlocks(&scene,
                           windowRect,
                           requested.windowId,
                           {closeRect.x0 + inset,
                            closeRect.y0 + inset,
                            closeRect.x1 - inset,
                            closeRect.y1 - inset},
                           block,
                           false,
                           {0.94f, 0.92f, 0.94f, closeHovered ? 0.98f : 0.66f});
      appendDiagonalBlocks(&scene,
                           windowRect,
                           requested.windowId,
                           {closeRect.x0 + inset,
                            closeRect.y0 + inset,
                            closeRect.x1 - inset,
                            closeRect.y1 - inset},
                           block,
                           true,
                           {0.94f, 0.92f, 0.94f, closeHovered ? 0.98f : 0.66f});
    }
    appendWindowHit(&scene,
                    &window,
                    ViewerUiControlKind::PlotBody,
                    windowRect,
                    planned.normalizedRect,
                    true);

    const float requestedTextScale =
        std::isfinite(requested.titleMetrics.textScale) &&
                requested.titleMetrics.textScale > 0.0f
            ? requested.titleMetrics.textScale
            : 1.0f;
    // Small windows need a title scale that fits the title band itself. Keep
    // the lower bound readable while preventing the adapter from having to
    // apply a per-window cap after the scene has already been built.
    const float titleMaxScale = std::max(0.66f, titleHeight / 20.0f);
    const float textScale = std::clamp(requestedTextScale,
                                      0.66f,
                                      std::min(1.18f, titleMaxScale));
    const float ascent = std::max(
        1.0f,
        std::isfinite(requested.titleMetrics.fontAscent)
            ? requested.titleMetrics.fontAscent * textScale
            : 14.0f * textScale);
    const float descent = std::max(
        0.0f,
        std::isfinite(requested.titleMetrics.fontDescent)
            ? requested.titleMetrics.fontDescent * textScale
            : 4.0f * textScale);
    const float centerY = windowRect.y0 + titleHeight * 0.5f;
    const float pad = std::max(2.0f, 1.5f * textScale);
    const float centeredBaseline = centerY + (ascent - descent) * 0.5f;
    const float minBaseline = windowRect.y0 + pad + ascent;
    const float maxBaseline = windowRect.y0 + titleHeight - pad - descent;
    const float baselineY = minBaseline <= maxBaseline
                                ? std::clamp(centeredBaseline,
                                             minBaseline,
                                             maxBaseline)
                                : centeredBaseline;
    const float moveGripReserve = interactionMode == PlotWindowDragMode::Move
                                      ? 34.0f * textScale
                                      : 0.0f;
    const float closeReserve = window.closable
                                   ? std::max(24.0f,
                                              titleHeight + 6.0f * textScale)
                                   : 0.0f;
    const float leftX = windowRect.x0 + 8.0f + moveGripReserve;
    const float rightX = windowRect.x1 - 8.0f - closeReserve;
    const float available = std::max(0.0f, rightX - leftX);
    if (available >= 20.0f) {
      float metadataMax = 0.0f;
      if (!requested.metadata.empty() &&
          requested.titleMetrics.measuredMetadataWidth > 0.0f &&
          available > 170.0f * textScale) {
        metadataMax = std::min(requested.titleMetrics.measuredMetadataWidth,
                               available * 0.44f);
      }
      const float gap = metadataMax > 0.0f ? 12.0f * textScale : 0.0f;
      float titleMax = available - metadataMax - gap;
      if (titleMax < 76.0f * textScale && metadataMax > 0.0f) {
        metadataMax = 0.0f;
        titleMax = available;
      }
      if (!requested.title.empty()) {
        ViewerUiTextIntent title{};
        title.visible = requested.titleMetrics.fontAvailable;
        title.text = requested.title;
        title.bounds = {leftX,
                        windowRect.y0,
                        std::max(leftX + 1.0f, rightX - metadataMax - gap),
                        std::min(windowRect.y1, windowRect.y0 + titleHeight)};
        title.originX = leftX;
        title.originY = baselineY;
        title.maxWidth = std::max(0.0f, titleMax);
        title.alignment = ViewerUiTextAlignment::Left;
        title.scale = textScale;
        title.color = {0.88f, 0.93f, 0.98f, focused ? 0.96f : 0.74f};
        appendWindowText(&scene, windowRect, requested.windowId, std::move(title));
      }
      if (!requested.metadata.empty() && metadataMax > 0.0f) {
        ViewerUiTextIntent metadata{};
        metadata.visible = requested.titleMetrics.fontAvailable;
        metadata.text = requested.metadata;
        metadata.bounds = {std::max(leftX, rightX - metadataMax),
                           windowRect.y0,
                           rightX,
                           std::min(windowRect.y1, windowRect.y0 + titleHeight)};
        metadata.originX = rightX;
        metadata.originY = baselineY;
        metadata.maxWidth = metadataMax;
        metadata.alignment = ViewerUiTextAlignment::Right;
        metadata.scale = textScale;
        metadata.color = {0.78f, 0.88f, 0.94f, focused ? 0.82f : 0.56f};
        appendWindowText(&scene,
                         windowRect,
                         requested.windowId,
                         std::move(metadata));
      }
    }
    appendSlicingControls(&scene, &window, requested, input);
    appendSourceSignalRestoreControls(&scene, &window, requested, input);
    appendSourceLassoControls(&scene, &window, requested, input);
    window.primitiveCount = scene.primitives.size() - window.primitiveBegin;
    window.textCount = scene.texts.size() - window.textBegin;
    window.vectorCount = scene.vectors.size() - window.vectorBegin;
    window.hitCount = scene.hits.size() - window.hitBegin;
    scene.windows.push_back(window);
  }
  return scene;
}

ViewerUiHitResult viewerUiHitTest(const ViewerUiScene& scene,
                                  float logicalX,
                                  float logicalY) noexcept {
  ViewerUiHitResult result{};
  if (!scene.ready() || !std::isfinite(logicalX) ||
      !std::isfinite(logicalY)) {
    return result;
  }
  const ViewerUiControlKind toolbarControl = workspaceToolbarHitTest(
      scene.toolbar, logicalX, logicalY);
  if (toolbarControl != ViewerUiControlKind::None) {
    result.control = toolbarControl;
    result.enabled = true;
    result.actionable = true;
    return result;
  }

  for (auto windowIt = scene.windows.rbegin();
       windowIt != scene.windows.rend();
       ++windowIt) {
    const ViewerUiWindowScene& window = *windowIt;
    const std::size_t hitEnd = window.hitBegin + window.hitCount;
    for (std::size_t i = window.hitBegin; i < hitEnd && i < scene.hits.size();
         ++i) {
      const ViewerUiHitRegion& hit = scene.hits[i];
      if (hit.control != ViewerUiControlKind::PlotClose ||
          !hit.enabled || !hit.actionable ||
          !viewerUiRectContainsInclusive(hit.rect, logicalX, logicalY)) {
        continue;
      }
      result.control = ViewerUiControlKind::PlotClose;
      result.windowId = window.windowId;
      result.controlIndex = hit.controlIndex;
      result.enabled = hit.enabled;
      result.actionable = hit.actionable;
      result.selected = hit.selected;
      return result;
    }
    for (std::size_t i = window.hitBegin; i < hitEnd && i < scene.hits.size();
         ++i) {
      const ViewerUiHitRegion& hit = scene.hits[i];
      if (!isPlotControl(hit.control) || !hit.enabled || !hit.actionable ||
          !viewerUiRectContainsInclusive(hit.rect, logicalX, logicalY)) {
        continue;
      }
      result.control = hit.control;
      result.windowId = window.windowId;
      result.controlIndex = hit.controlIndex;
      result.enabled = hit.enabled;
      result.actionable = hit.actionable;
      result.selected = hit.selected;
      return result;
    }
    for (std::size_t i = window.hitBegin; i < hitEnd && i < scene.hits.size();
         ++i) {
      const ViewerUiHitRegion& hit = scene.hits[i];
      if (hit.control != ViewerUiControlKind::PlotBody ||
          !viewerUiRectContainsInclusive(hit.rect, logicalX, logicalY)) {
        continue;
      }
      result.control = ViewerUiControlKind::PlotBody;
      result.windowId = window.windowId;
      result.controlIndex = hit.controlIndex;
      result.enabled = hit.enabled;
      result.actionable = hit.actionable;
      result.selected = hit.selected;
      if (hit.usesPlotDragGeometry) {
        result.dragMode = plotWindowDragModeAt(
            {hit.normalizedRect,
             hit.rect,
             logicalX,
             logicalY});
      }
      return result;
    }
  }
  return result;
}

}  // namespace ChromaspaceViewer
