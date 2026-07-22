#pragma once

#include "ChromaspaceViewerFramePlan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaspaceViewer {

// All UI scene coordinates are logical pixels with a top-left/Y-down origin.
// Adapters are responsible for converting them to framebuffer or backend space.
constexpr float kWorkspaceToolbarButtonSize = 30.0f;
constexpr float kWorkspaceToolbarInset = 10.0f;
constexpr float kWorkspaceToolbarGap = 7.0f;
constexpr float kWorkspaceToolbarLayoutLabelWidth = 148.0f;
constexpr float kWorkspaceToolbarRightMargin = 10.0f;
constexpr std::size_t kViewerUiSlicingVectorCount = 6u;
constexpr float kViewerUiSlicingQuickButtonSize = 34.0f;

enum class ViewerUiControlKind : uint8_t {
  None = 0,
  ToolbarMenu,
  ToolbarAddPlot,
  ToolbarLayoutPreset,
  PlotBody,
  PlotClose,
  SlicingQuickToggle,
  SlicingVector,
  SlicingLasso,
  SourceLassoAdd,
  SourceLassoSubtract,
  SourceLassoClear,
  SourceSignalRestore,
};

enum class ViewerUiPrimitiveKind : uint8_t {
  SolidRect = 0,
  Glyph,
};

enum class ViewerUiTextAlignment : uint8_t {
  Left = 0,
  Center,
  Right,
};

struct ViewerUiColor {
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
  float a = 1.0f;
};

struct ViewerUiSolidRect {
  ScreenRect rect{};
  ViewerUiColor color{};
  ViewerUiControlKind control = ViewerUiControlKind::None;
  int windowId = -1;
  ViewerUiPrimitiveKind kind = ViewerUiPrimitiveKind::SolidRect;
  int controlIndex = -1;
  bool enabled = true;
  bool selected = false;
};

struct ViewerUiHitRegion {
  ViewerUiControlKind control = ViewerUiControlKind::None;
  int windowId = -1;
  ScreenRect rect{};
  PlotWindowRectNorm normalizedRect{};
  bool usesPlotDragGeometry = false;
  int controlIndex = -1;
  bool enabled = true;
  bool actionable = true;
  bool selected = false;
};

// Backend-neutral triangle-list output in logical top-left/Y-down space.
// Every three consecutive vertices form one triangle.
struct ViewerUiVectorVertex {
  float x = 0.0f;
  float y = 0.0f;
  ViewerUiColor color{};
  ViewerUiControlKind control = ViewerUiControlKind::None;
  int windowId = -1;
  int controlIndex = -1;
  bool enabled = true;
  bool selected = false;
};

struct ViewerUiTextIntent {
  bool visible = false;
  std::string text;
  ScreenRect bounds{};
  // Origin is a left/right-aligned baseline in logical top-left/Y-down space.
  float originX = 0.0f;
  float originY = 0.0f;
  float maxWidth = 0.0f;
  ViewerUiTextAlignment alignment = ViewerUiTextAlignment::Left;
  float scale = 1.0f;
  ViewerUiColor color{0.78f, 0.92f, 0.98f, 0.96f};
  ViewerUiControlKind control = ViewerUiControlKind::None;
  int windowId = -1;
  int controlIndex = -1;
  bool enabled = true;
  bool selected = false;
};

struct ViewerUiHitResult {
  ViewerUiControlKind control = ViewerUiControlKind::None;
  int windowId = -1;
  PlotWindowDragMode dragMode = PlotWindowDragMode::None;
  int controlIndex = -1;
  bool enabled = false;
  bool actionable = false;
  bool selected = false;

  bool hit() const noexcept { return control != ViewerUiControlKind::None; }
};

// This is the portable toolbar subscene input. It contains no AppState,
// renderer, window handle, framebuffer scale, or backend resource.
struct WorkspaceToolbarInput {
  int logicalWidth = 0;
  int logicalHeight = 0;
  float reservedLeftPixels = 0.0f;
  bool visible = false;
  float textScale = 1.0f;
  bool menuActive = false;
  bool addPlotActive = false;
  bool layoutActive = false;
  bool hasPointer = false;
  float pointerX = 0.0f;
  float pointerY = 0.0f;
  int layoutIndex = 0;
  std::string layoutLabel;
};

struct WorkspaceToolbarScene {
  bool visible = false;
  int logicalWidth = 0;
  int logicalHeight = 0;
  float reservedLeftPixels = 0.0f;
  float textScale = 1.0f;
  std::array<ViewerUiHitRegion, 3> controls{};
  std::vector<ViewerUiSolidRect> primitives;
  ViewerUiTextIntent layoutLabel{};
};

WorkspaceToolbarScene buildWorkspaceToolbarScene(
    const WorkspaceToolbarInput& input);

ViewerUiControlKind workspaceToolbarHitTest(
    const WorkspaceToolbarScene& scene,
    float logicalX,
    float logicalY) noexcept;

bool viewerUiRectContainsInclusive(const ScreenRect& rect,
                                   float logicalX,
                                   float logicalY) noexcept;

struct ViewerUiTitleMetrics {
  float titleExtraHeight = 0.0f;
  float fontAscent = 14.0f;
  float fontDescent = 4.0f;
  float textScale = 1.0f;
  float measuredMetadataWidth = 0.0f;
  bool fontAvailable = true;
};

// Shared title-band policy for scene construction and legacy content adapters.
// The returned height is in logical top-left/Y-down viewer pixels.
float viewerUiTitleBarLogicalHeight(float windowHeight,
                                    float titleExtraHeight) noexcept;

struct ViewerUiPlotWindowInput {
  struct SlicingControls {
    bool visible = false;
    bool drawerOpen = false;
    bool active = false;
    std::array<bool, kViewerUiSlicingVectorCount> vectors{};
    bool lassoActive = false;
    float animationProgress = 0.0f;
  } slicing;

  struct SourceLassoControls {
    bool visible = false;
    bool subtract = false;
    bool hasSelection = false;
  } sourceLasso;

  // A lasso owner may expose more than one docked Source Signal surface.
  // Stable source IDs are carried as control indices so restoring never
  // retargets focus or selection ownership to the owner window by accident.
  std::vector<int> sourceSignalRestoreWindowIds;

  int windowId = -1;
  std::string title;
  std::string metadata;
  ViewerUiTitleMetrics titleMetrics{};
  bool closable = true;
};

struct ViewerUiSceneInput {
  WorkspaceToolbarInput toolbar{};
  bool hasPointer = false;
  float pointerX = 0.0f;
  float pointerY = 0.0f;
  int focusedWindowId = -1;
  int hoveredWindowId = -1;
  PlotWindowDragMode hoveredDragMode = PlotWindowDragMode::None;
  int activeDragWindowId = -1;
  PlotWindowDragMode activeDragMode = PlotWindowDragMode::None;
  std::vector<ViewerUiPlotWindowInput> windows;
};

enum class ViewerUiSceneStatus : uint8_t {
  Ready = 0,
  InvalidFramePlan,
  InvalidViewport,
  WindowCountMismatch,
  WindowIdMismatch,
  InvalidWindowInput,
};

struct ViewerUiWindowScene {
  int windowId = -1;
  ScreenRect rect{};
  ScreenRect contentRect{};
  PlotWindowRectNorm normalizedRect{};
  bool closable = false;
  std::size_t primitiveBegin = 0;
  std::size_t primitiveCount = 0;
  std::size_t textBegin = 0;
  std::size_t textCount = 0;
  std::size_t vectorBegin = 0;
  std::size_t vectorCount = 0;
  std::size_t hitBegin = 0;
  std::size_t hitCount = 0;
};

struct ViewerUiScene {
  ViewerUiSceneStatus status = ViewerUiSceneStatus::Ready;
  ViewerFramePlanGeometry geometry{};
  WorkspaceToolbarScene toolbar{};
  std::vector<ViewerUiWindowScene> windows;
  std::vector<ViewerUiSolidRect> primitives;
  std::vector<ViewerUiTextIntent> texts;
  std::vector<ViewerUiVectorVertex> vectors;
  std::vector<ViewerUiHitRegion> hits;

  bool valid() const noexcept { return status == ViewerUiSceneStatus::Ready; }
  bool ready() const noexcept { return valid(); }
};

ViewerUiScene buildViewerUiScene(const ViewerFramePlan& plan,
                                 const ViewerUiSceneInput& input);

ViewerUiHitResult viewerUiHitTest(const ViewerUiScene& scene,
                                  float logicalX,
                                  float logicalY) noexcept;

}  // namespace ChromaspaceViewer
