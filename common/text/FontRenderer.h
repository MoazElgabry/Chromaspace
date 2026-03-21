#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace WorkshopText {

struct FontRuntime;

struct GlyphMetrics {
  float advanceX = 0.0f;
  float bitmapLeft = 0.0f;
  float bitmapTop = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float lsbDelta = 0.0f;
  float rsbDelta = 0.0f;
  int advance26_6 = 0;
  int lsbDelta26_6 = 0;
  int rsbDelta26_6 = 0;
  float u0 = 0.0f;
  float v0 = 0.0f;
  float u1 = 0.0f;
  float v1 = 0.0f;
  bool valid = false;
};

struct FontAtlas {
  int width = 0;
  int height = 0;
  int pixelSize = 0;
  int lineHeight = 0;
  int ascent = 0;
  int descent = 0;
  std::array<GlyphMetrics, 128> glyphs{};
  std::unordered_map<unsigned int, GlyphMetrics> glyphsByIndex;
  std::array<unsigned int, 128> glyphIndexForChar{};
  std::array<float, 128 * 128> kerningX{};
  std::array<int, 128 * 128> kerningX26_6{};
  std::vector<unsigned char> pixels;
  std::shared_ptr<FontRuntime> runtime;
  bool valid = false;
};

struct TextQuadVertex {
  float x = 0.0f;
  float y = 0.0f;
  float u = 0.0f;
  float v = 0.0f;
};

bool loadFontAtlas(const std::string& fontPath, int pixelSize, FontAtlas* out, std::string* error);
float measureTextWidth(const FontAtlas& atlas, std::string_view text, float scale = 1.0f);
void appendTextQuads(const FontAtlas& atlas,
                     std::string_view text,
                     float baselineX,
                     float baselineY,
                     float scale,
                     std::vector<TextQuadVertex>* out);

}  // namespace WorkshopText
