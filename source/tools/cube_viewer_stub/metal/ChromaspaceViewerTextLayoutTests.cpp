#include "ChromaspaceViewerTextLayout.h"

#include <cassert>
#include <cmath>

namespace {

using namespace ChromaspaceViewer;

bool near(float left, float right, float epsilon = 1.0e-4f) {
  return std::fabs(left - right) <= epsilon;
}

WorkshopText::FontAtlas makeAtlas() {
  WorkshopText::FontAtlas atlas{};
  atlas.width = 128;
  atlas.height = 1;
  atlas.pixelSize = 10;
  atlas.lineHeight = 12;
  atlas.ascent = 8;
  atlas.descent = 2;
  atlas.pixels.assign(128u, 255u);
  for (std::size_t index = 0; index < atlas.glyphs.size(); ++index) {
    auto& glyph = atlas.glyphs[index];
    glyph.advanceX = 10.0f;
    glyph.advance26_6 = 640;
    glyph.bitmapTop = 8.0f;
    glyph.width = 8.0f;
    glyph.height = 10.0f;
    glyph.u0 = static_cast<float>(index) / 128.0f;
    glyph.u1 = static_cast<float>(index + 1u) / 128.0f;
    glyph.v0 = 0.0f;
    glyph.v1 = 1.0f;
    glyph.valid = true;
  }
  atlas.valid = true;
  return atlas;
}

ViewerUiTextIntent intent(const char* text,
                          float originX,
                          ViewerUiTextAlignment alignment,
                          float maxWidth = 100.0f) {
  ViewerUiTextIntent value{};
  value.visible = true;
  value.text = text;
  value.bounds = {0.0f, 0.0f, 200.0f, 50.0f};
  value.originX = originX;
  value.originY = 20.0f;
  value.maxWidth = maxWidth;
  value.alignment = alignment;
  value.scale = 1.0f;
  value.color = {0.2f, 0.4f, 0.6f, 0.8f};
  return value;
}

void testAllIntentsAlignmentFittingAndRetina() {
  const WorkshopText::FontAtlas atlas = makeAtlas();
  ViewerUiScene scene{};
  scene.geometry.windowWidth = 200;
  scene.geometry.windowHeight = 100;
  scene.geometry.framebufferWidth = 400;
  scene.geometry.framebufferHeight = 300;
  scene.texts.push_back(intent("AB", 10.0f, ViewerUiTextAlignment::Left));
  scene.texts.push_back(
      intent("ABCDEF", 100.0f, ViewerUiTextAlignment::Center, 45.0f));
  scene.texts.push_back(intent("C", 190.0f, ViewerUiTextAlignment::Right));
  ViewerUiTextIntent hidden{};
  hidden.visible = false;
  hidden.text = "ignored";
  scene.texts.push_back(hidden);

  const auto result = buildViewerTextLayout({&scene, &atlas, 400, 300});
  assert(result.ready());
  assert(result.runs.size() == 3u);
  assert(result.vertices.size() == (2u + 4u + 1u) * 6u);
  assert(result.runs[0].firstVertex == 0u &&
         result.runs[0].vertexCount == 12u);
  assert(result.runs[1].firstVertex == 12u &&
         result.runs[1].vertexCount == 24u);
  assert(near(result.vertices[0].x, 20.0f));
  // Centered fitted text is "A..." (40 logical pixels), so its baseline is 80.
  assert(near(result.vertices[12].x, 160.0f));
  // Right-aligned C starts at 180 logical pixels.
  assert(near(result.vertices[36].x, 360.0f));
  assert(near(result.runs[0].clip.x1, 400.0f));
  assert(near(result.runs[0].clip.y1, 150.0f));
  assert(result.runs[0].color.a == 0.8f);
}

void testTypedAtomicFailuresAndEmptySuccess() {
  const WorkshopText::FontAtlas atlas = makeAtlas();
  ViewerUiScene scene{};
  scene.geometry.windowWidth = 200;
  scene.geometry.windowHeight = 100;

  auto empty = buildViewerTextLayout({&scene, &atlas, 200, 100});
  assert(empty.ready() && empty.vertices.empty() && empty.runs.empty());

  ViewerUiTextIntent invalid = intent("bad", 10.0f,
                                      ViewerUiTextAlignment::Left);
  invalid.scale = 0.0f;
  scene.texts.push_back(invalid);
  const auto invalidResult =
      buildViewerTextLayout({&scene, &atlas, 200, 100});
  assert(invalidResult.status == ViewerTextLayoutStatus::InvalidIntent);
  assert(invalidResult.vertices.empty() && invalidResult.runs.empty());

  scene.texts.assign(kViewerTextLayoutMaxIntents + 1u, ViewerUiTextIntent{});
  const auto bounded = buildViewerTextLayout({&scene, &atlas, 200, 100});
  assert(bounded.status == ViewerTextLayoutStatus::CapacityExceeded);
  assert(bounded.vertices.empty() && bounded.runs.empty());

  WorkshopText::FontAtlas badAtlas{};
  const auto bad = buildViewerTextLayout({&scene, &badAtlas, 200, 100});
  assert(bad.status == ViewerTextLayoutStatus::InvalidAtlas);
}

}  // namespace

int main() {
  testAllIntentsAlignmentFittingAndRetina();
  testTypedAtomicFailuresAndEmptySuccess();
  return 0;
}
