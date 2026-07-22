#pragma once

#include "ChromaspaceViewerUiScene.h"
#include "text/FontRenderer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ChromaspaceViewer {

constexpr std::size_t kViewerTextLayoutMaxIntents = 4096u;
constexpr std::size_t kViewerTextLayoutMaxTextBytes = 1024u * 1024u;
constexpr std::size_t kViewerTextLayoutMaxVertices = 1024u * 1024u;

enum class ViewerTextLayoutStatus : uint8_t {
  Ready = 0,
  InvalidScene,
  InvalidAtlas,
  InvalidViewport,
  InvalidIntent,
  CapacityExceeded,
  AllocationFailure,
};

struct ViewerTextLayoutVertex {
  float x = 0.0f;
  float y = 0.0f;
  float u = 0.0f;
  float v = 0.0f;
};

struct ViewerTextLayoutRun {
  uint32_t firstVertex = 0u;
  uint32_t vertexCount = 0u;
  ViewerUiColor color{};
  ScreenRect clip{};
};

struct ViewerTextLayoutRequest {
  const ViewerUiScene* scene = nullptr;
  const WorkshopText::FontAtlas* atlas = nullptr;
  int framebufferWidth = 0;
  int framebufferHeight = 0;
};

struct ViewerTextLayoutResult {
  ViewerTextLayoutStatus status = ViewerTextLayoutStatus::InvalidScene;
  std::vector<ViewerTextLayoutVertex> vertices;
  std::vector<ViewerTextLayoutRun> runs;

  bool ready() const noexcept {
    return status == ViewerTextLayoutStatus::Ready;
  }
};

// Shapes and projects every visible scene text intent into framebuffer-space
// triangles. All output is built transactionally and published only on Ready.
ViewerTextLayoutResult buildViewerTextLayout(
    const ViewerTextLayoutRequest& request) noexcept;

}  // namespace ChromaspaceViewer
