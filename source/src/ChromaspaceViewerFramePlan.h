#pragma once

#include "ChromaspaceViewerLayout.h"

#include <cstdint>
#include <vector>

namespace ChromaspaceViewer {

// Frame geometry is intentionally independent of any presentation backend.
// Logical and framebuffer coordinates use the viewer's top-left/Y-down space.
struct ViewerFramePlanRect {
  float x0 = 0.0f;
  float y0 = 0.0f;
  float x1 = 1.0f;
  float y1 = 1.0f;
};

struct ViewerFramePlanWindowInput {
  int windowId = -1;
  PlotWindowRectNorm rect{};
  int plotModel = 0;
  uint64_t viewRevision = 0;
  bool visible = true;
};

struct ViewerFramePlanRequest {
  int windowWidth = 1;
  int windowHeight = 1;
  int framebufferWidth = 1;
  int framebufferHeight = 1;
  float reservedLeftPixels = 0.0f;
  std::vector<ViewerFramePlanWindowInput> windows;
};

struct ViewerFramePlanGeometry {
  int windowWidth = 1;
  int windowHeight = 1;
  int framebufferWidth = 1;
  int framebufferHeight = 1;
  float reservedLeftPixels = 0.0f;
  float scaleX = 1.0f;
  float scaleY = 1.0f;
};

struct ViewerFramePlanWindow {
  int windowId = -1;
  int plotModel = 0;
  uint64_t viewRevision = 0;
  PlotWindowRectNorm normalizedRect{};
  ViewerFramePlanRect logicalRect{};
  ViewerFramePlanRect framebufferRect{};
  int renderTargetWidth = 1;
  int renderTargetHeight = 1;
};

enum class ViewerFramePlanStatus {
  Ready,
  InvalidWindowId,
  DuplicateWindowId,
};

struct ViewerFramePlan {
  ViewerFramePlanGeometry geometry{};
  std::vector<ViewerFramePlanWindow> windows;
  ViewerFramePlanStatus status = ViewerFramePlanStatus::Ready;
  int rejectedWindowId = 0;

  bool valid() const noexcept { return status == ViewerFramePlanStatus::Ready; }
  bool ready() const noexcept { return valid(); }
};

ViewerFramePlan buildViewerFramePlan(const ViewerFramePlanRequest& request);

}  // namespace ChromaspaceViewer
