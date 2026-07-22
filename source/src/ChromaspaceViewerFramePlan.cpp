#include "ChromaspaceViewerFramePlan.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace ChromaspaceViewer {
namespace {

int sanitizedDimension(int value) noexcept {
  return std::max(1, value);
}

float finiteOr(float value, float fallback) noexcept {
  return std::isfinite(value) ? value : fallback;
}

float finitePositive(float value) noexcept {
  return std::isfinite(value) && value >= 1.0f ? value : 1.0f;
}

int ceilRenderTargetDimension(float value) noexcept {
  if (!std::isfinite(value) || value <= 1.0f) return 1;
  constexpr float kMaxInt = static_cast<float>(std::numeric_limits<int>::max());
  if (value >= kMaxInt) return std::numeric_limits<int>::max();
  return std::max(1, static_cast<int>(std::ceil(value)));
}

}  // namespace

ViewerFramePlan buildViewerFramePlan(const ViewerFramePlanRequest& request) {
  ViewerFramePlan plan{};
  plan.geometry.windowWidth = sanitizedDimension(request.windowWidth);
  plan.geometry.windowHeight = sanitizedDimension(request.windowHeight);
  plan.geometry.framebufferWidth = sanitizedDimension(request.framebufferWidth);
  plan.geometry.framebufferHeight = sanitizedDimension(request.framebufferHeight);
  const float maxReserved = static_cast<float>(plan.geometry.windowWidth - 1);
  const float requestedReserved = finiteOr(request.reservedLeftPixels, 0.0f);
  plan.geometry.reservedLeftPixels =
      std::clamp(requestedReserved, 0.0f, maxReserved);
  plan.geometry.scaleX = static_cast<float>(plan.geometry.framebufferWidth) /
                         static_cast<float>(plan.geometry.windowWidth);
  plan.geometry.scaleY = static_cast<float>(plan.geometry.framebufferHeight) /
                         static_cast<float>(plan.geometry.windowHeight);

  const float availableLogicalWidth = std::max(
      1.0f,
      static_cast<float>(plan.geometry.windowWidth) -
          plan.geometry.reservedLeftPixels);

  // Validate the complete input sequence before emitting any visible-window
  // geometry. Hidden windows still participate in identity validation so a
  // later renderer cannot observe an ambiguous join by visibility alone.
  std::unordered_set<int> windowIds;
  windowIds.reserve(request.windows.size());
  for (const ViewerFramePlanWindowInput& input : request.windows) {
    if (input.windowId <= 0) {
      plan.status = ViewerFramePlanStatus::InvalidWindowId;
      plan.rejectedWindowId = input.windowId;
      return plan;
    }
    if (!windowIds.insert(input.windowId).second) {
      plan.status = ViewerFramePlanStatus::DuplicateWindowId;
      plan.rejectedWindowId = input.windowId;
      return plan;
    }
  }

  plan.windows.reserve(request.windows.size());
  for (const ViewerFramePlanWindowInput& input : request.windows) {
    if (!input.visible) continue;

    const float normalizedX = finiteOr(input.rect.x, 0.0f);
    const float normalizedY = finiteOr(input.rect.y, 0.0f);
    const float normalizedW = finiteOr(input.rect.w, 0.0f);
    const float normalizedH = finiteOr(input.rect.h, 0.0f);
    const float logicalX = finiteOr(
        plan.geometry.reservedLeftPixels + normalizedX * availableLogicalWidth,
        plan.geometry.reservedLeftPixels);
    const float logicalY = finiteOr(
        normalizedY * static_cast<float>(plan.geometry.windowHeight),
        0.0f);
    const float logicalW = finitePositive(normalizedW * availableLogicalWidth);
    const float logicalH = finitePositive(
        normalizedH * static_cast<float>(plan.geometry.windowHeight));
    const float logicalX1 = finiteOr(logicalX + logicalW, logicalX + 1.0f);
    const float logicalY1 = finiteOr(logicalY + logicalH, logicalY + 1.0f);

    ViewerFramePlanWindow planned{};
    planned.windowId = input.windowId;
    planned.plotModel = input.plotModel;
    planned.viewRevision = input.viewRevision;
    planned.normalizedRect = {normalizedX, normalizedY, normalizedW, normalizedH};
    planned.logicalRect = {logicalX, logicalY, logicalX1, logicalY1};

    const float framebufferX = finiteOr(logicalX * plan.geometry.scaleX, 0.0f);
    const float framebufferY = finiteOr(logicalY * plan.geometry.scaleY, 0.0f);
    const float framebufferW = finitePositive(logicalW * plan.geometry.scaleX);
    const float framebufferH = finitePositive(logicalH * plan.geometry.scaleY);
    planned.framebufferRect = {framebufferX,
                               framebufferY,
                               finiteOr(framebufferX + framebufferW, framebufferX + 1.0f),
                               finiteOr(framebufferY + framebufferH, framebufferY + 1.0f)};
    planned.renderTargetWidth = ceilRenderTargetDimension(framebufferW);
    planned.renderTargetHeight = ceilRenderTargetDimension(framebufferH);
    plan.windows.push_back(planned);
  }
  return plan;
}

}  // namespace ChromaspaceViewer
