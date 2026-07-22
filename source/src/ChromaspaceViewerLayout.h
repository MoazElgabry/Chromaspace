#pragma once

#include <string_view>
#include <vector>

namespace ChromaspaceViewer {

// Normalized plot-window geometry is shared by workspace persistence,
// hit-testing, layout presets, and every presentation backend.
struct PlotWindowRectNorm {
  float x = 0.0f;
  float y = 0.0f;
  float w = 1.0f;
  float h = 1.0f;
};

// The viewer uses a top-left normalized coordinate system for workspace and
// plot-window geometry.  A workspace starts below the optional toolbar and
// extends to the bottom edge of the normalized viewer surface.
struct WorkspaceGeometry {
  float topNorm = 0.0f;
  float heightNorm = 1.0f;
};

WorkspaceGeometry workspaceGeometry(bool showWorkspaceToolbar,
                                     int windowHeight) noexcept;
float plotWindowMinNormWidth(int windowWidth) noexcept;
float plotWindowMinNormHeight(int windowHeight) noexcept;

PlotWindowRectNorm workspaceRelativeRect(const WorkspaceGeometry& workspace,
                                          float x,
                                          float y,
                                          float w,
                                          float h) noexcept;

PlotWindowRectNorm clampPlotWindowRect(PlotWindowRectNorm rect,
                                       const WorkspaceGeometry& workspace,
                                       int windowWidth,
                                       int windowHeight) noexcept;

PlotWindowRectNorm reflowPlotWindowRectForWorkspaceTop(PlotWindowRectNorm rect,
                                                        float oldTop,
                                                        float newTop) noexcept;

constexpr int kViewerLayoutChoiceCount = 7;
constexpr int kViewerLayoutSoloIndex = 6;
constexpr float kViewerWorkspaceToolbarHeight = 42.0f;

enum class PlotWindowDragMode {
  None = 0,
  Move,
  ResizeLeft,
  ResizeRight,
  ResizeTop,
  ResizeBottom,
  ResizeTopLeft,
  ResizeTopRight,
  ResizeBottomLeft,
  ResizeBottomRight
};

struct PlotWindowDragTraits {
  bool isResize = false;
  bool touchesLeft = false;
  bool touchesRight = false;
  bool touchesTop = false;
  bool touchesBottom = false;
};

PlotWindowDragTraits plotWindowDragModeTraits(PlotWindowDragMode mode) noexcept;

struct ScreenRect {
  float x0 = 0.0f;
  float y0 = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
};

struct PlotWindowDragHitTestRequest {
  PlotWindowRectNorm windowRect{};
  ScreenRect screenRect{};
  float cursorX = 0.0f;
  float cursorY = 0.0f;
};

PlotWindowDragMode plotWindowDragModeAt(
    const PlotWindowDragHitTestRequest& request) noexcept;

struct PlotWindowDragRequest {
  PlotWindowRectNorm startRect{};
  PlotWindowDragMode mode = PlotWindowDragMode::None;
  float deltaXNorm = 0.0f;
  float deltaYNorm = 0.0f;
  WorkspaceGeometry workspace{};
  float minWidthNorm = 0.0f;
  float minHeightNorm = 0.0f;
};

PlotWindowRectNorm applyPlotWindowDrag(
    const PlotWindowDragRequest& request) noexcept;

struct PlotWindowSnapPreviewRequest {
  PlotWindowRectNorm candidateRect{};
  PlotWindowDragMode dragMode = PlotWindowDragMode::None;
  WorkspaceGeometry workspace{};
  int windowWidth = 1;
  int windowHeight = 1;
  float reservedLeftPixels = 0.0f;
  float cursorX = 0.0f;
  float cursorY = 0.0f;
  bool singleWindowWorkspace = false;
  std::vector<PlotWindowRectNorm> otherWindowRects;
};

struct PlotWindowSnapPreviewResult {
  bool visible = false;
  PlotWindowRectNorm rect{};
};

PlotWindowSnapPreviewResult computePlotWindowSnapPreview(
    const PlotWindowSnapPreviewRequest& request) noexcept;

struct StandardLayoutDescriptor {
  int index = 0;
  const char* label = "Single";
  int requiredWindowCount = 1;
  int preferredWindowWidth = 720;
  int preferredWindowHeight = 600;
};

// Invalid layout indices clamp to the nearest standard layout, preserving the
// legacy integer IDs used by workspace persistence and preset files.
const StandardLayoutDescriptor& standardPlotLayout(int layoutIndex) noexcept;
const StandardLayoutDescriptor& standardPlotLayoutForDisplayRow(int row) noexcept;

// Returns nullptr when name is not one of the standard layout labels. Matching
// is case-insensitive over the same byte-wise characters used by the viewer.
const StandardLayoutDescriptor* findStandardPlotLayout(std::string_view name) noexcept;
bool isStandardPlotLayoutNameReserved(std::string_view name) noexcept;

PlotWindowRectNorm standardPlotLayoutSlotRect(const StandardLayoutDescriptor& layout,
                                              int slotIndex) noexcept;

// Returns the model ID used when a standard layout initializes a slot. Solo
// intentionally returns -1 because it preserves the focused plot state.
int standardPlotLayoutDefaultPlotModel(const StandardLayoutDescriptor& layout,
                                       int slotIndex) noexcept;

bool plotWindowRectNear(const PlotWindowRectNorm& a,
                        const PlotWindowRectNorm& b) noexcept;

}  // namespace ChromaspaceViewer
