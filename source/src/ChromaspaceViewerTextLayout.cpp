#include "ChromaspaceViewerTextLayout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace ChromaspaceViewer {
namespace {

bool finite(float value) noexcept { return std::isfinite(value); }

bool validRect(const ScreenRect& rect) noexcept {
  return finite(rect.x0) && finite(rect.y0) && finite(rect.x1) &&
         finite(rect.y1) && rect.x1 > rect.x0 && rect.y1 > rect.y0;
}

bool validColor(const ViewerUiColor& color) noexcept {
  return finite(color.r) && finite(color.g) && finite(color.b) &&
         finite(color.a) && color.r >= 0.0f && color.r <= 1.0f &&
         color.g >= 0.0f && color.g <= 1.0f && color.b >= 0.0f &&
         color.b <= 1.0f && color.a >= 0.0f && color.a <= 1.0f;
}

ViewerTextLayoutResult failed(ViewerTextLayoutStatus status) noexcept {
  ViewerTextLayoutResult result{};
  result.status = status;
  return result;
}

std::string fitText(const WorkshopText::FontAtlas& atlas,
                    const std::string& text,
                    float maxWidth,
                    float scale) {
  if (text.empty() || maxWidth <= 0.0f) return {};
  if (WorkshopText::measureTextWidth(atlas, text, scale) <= maxWidth) {
    return text;
  }
  constexpr const char* kSuffix = "...";
  if (WorkshopText::measureTextWidth(atlas, kSuffix, scale) > maxWidth) {
    return {};
  }
  std::string prefix = text;
  while (!prefix.empty()) {
    prefix.pop_back();
    const std::string candidate = prefix + kSuffix;
    if (WorkshopText::measureTextWidth(atlas, candidate, scale) <= maxWidth) {
      return candidate;
    }
  }
  return kSuffix;
}

}  // namespace

ViewerTextLayoutResult buildViewerTextLayout(
    const ViewerTextLayoutRequest& request) noexcept {
  if (!request.scene || !request.scene->ready()) {
    return failed(ViewerTextLayoutStatus::InvalidScene);
  }
  if (!request.atlas || !request.atlas->valid || request.atlas->width <= 0 ||
      request.atlas->height <= 0 || request.atlas->pixels.empty()) {
    return failed(ViewerTextLayoutStatus::InvalidAtlas);
  }
  const ViewerUiScene& scene = *request.scene;
  if (scene.geometry.windowWidth <= 0 || scene.geometry.windowHeight <= 0 ||
      request.framebufferWidth <= 0 || request.framebufferHeight <= 0) {
    return failed(ViewerTextLayoutStatus::InvalidViewport);
  }
  if (scene.texts.size() > kViewerTextLayoutMaxIntents) {
    return failed(ViewerTextLayoutStatus::CapacityExceeded);
  }
  std::size_t totalTextBytes = 0u;
  for (const ViewerUiTextIntent& intent : scene.texts) {
    if (intent.text.size() > kViewerTextLayoutMaxTextBytes - totalTextBytes) {
      return failed(ViewerTextLayoutStatus::CapacityExceeded);
    }
    totalTextBytes += intent.text.size();
    if (!intent.visible) continue;
    if (!validRect(intent.bounds) || !finite(intent.originX) ||
        !finite(intent.originY) || !finite(intent.maxWidth) ||
        intent.maxWidth <= 0.0f || !finite(intent.scale) ||
        intent.scale <= 0.0f || !validColor(intent.color) ||
        intent.alignment > ViewerUiTextAlignment::Right) {
      return failed(ViewerTextLayoutStatus::InvalidIntent);
    }
  }

  try {
    ViewerTextLayoutResult output{};
    output.vertices.reserve(std::min(
        kViewerTextLayoutMaxVertices, totalTextBytes * 6u));
    output.runs.reserve(scene.texts.size());
    const float scaleX = static_cast<float>(request.framebufferWidth) /
                         static_cast<float>(scene.geometry.windowWidth);
    const float scaleY = static_cast<float>(request.framebufferHeight) /
                         static_cast<float>(scene.geometry.windowHeight);

    for (const ViewerUiTextIntent& intent : scene.texts) {
      if (!intent.visible || intent.text.empty()) continue;
      const std::string fitted =
          fitText(*request.atlas, intent.text, intent.maxWidth, intent.scale);
      if (fitted.empty()) continue;
      const float width =
          WorkshopText::measureTextWidth(*request.atlas, fitted, intent.scale);
      float baselineX = intent.originX;
      if (intent.alignment == ViewerUiTextAlignment::Right) {
        baselineX -= width;
      } else if (intent.alignment == ViewerUiTextAlignment::Center) {
        baselineX -= width * 0.5f;
      }
      const float baselineYUp =
          static_cast<float>(scene.geometry.windowHeight) - intent.originY;
      std::vector<WorkshopText::TextQuadVertex> quads;
      WorkshopText::appendTextQuads(*request.atlas, fitted, baselineX,
                                    baselineYUp, intent.scale, &quads);
      if (quads.empty()) continue;
      if ((quads.size() % 3u) != 0u ||
          quads.size() > kViewerTextLayoutMaxVertices - output.vertices.size() ||
          output.vertices.size() >
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
          quads.size() >
              static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        return failed(ViewerTextLayoutStatus::CapacityExceeded);
      }
      ViewerTextLayoutRun run{};
      run.firstVertex = static_cast<uint32_t>(output.vertices.size());
      run.vertexCount = static_cast<uint32_t>(quads.size());
      run.color = intent.color;
      run.clip = {intent.bounds.x0 * scaleX, intent.bounds.y0 * scaleY,
                  intent.bounds.x1 * scaleX, intent.bounds.y1 * scaleY};
      for (const WorkshopText::TextQuadVertex& quad : quads) {
        output.vertices.push_back(
            {quad.x * scaleX,
             (static_cast<float>(scene.geometry.windowHeight) - quad.y) * scaleY,
             quad.u, quad.v});
      }
      output.runs.push_back(run);
    }
    output.status = ViewerTextLayoutStatus::Ready;
    return output;
  } catch (...) {
    return failed(ViewerTextLayoutStatus::AllocationFailure);
  }
}

}  // namespace ChromaspaceViewer
