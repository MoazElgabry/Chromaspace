#include "FontRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <freetype/freetype.h>
#include <hb-ft.h>

namespace WorkshopText {
struct FontRuntime {
  FT_Library library = nullptr;
  FT_Face face = nullptr;
  hb_font_t* hbFont = nullptr;
  int loadFlags = FT_LOAD_DEFAULT;

  ~FontRuntime() {
    if (hbFont) hb_font_destroy(hbFont);
    if (face) FT_Done_Face(face);
    if (library) FT_Done_FreeType(library);
  }
};

namespace {

constexpr unsigned char kFirstGlyph = 32;
constexpr unsigned char kLastGlyph = 126;
constexpr int kAtlasWidth = 1024;
constexpr int kPadding = 8;
constexpr std::array<uint32_t, 3> kExtraCodepoints = {0x2026u, 0x21BAu, 0x21BBu};

struct GlyphBitmapInfo {
  unsigned int glyphIndex = 0;
  int width = 0;
  int height = 0;
  float bitmapLeft = 0.0f;
  float bitmapTop = 0.0f;
  float advanceX = 0.0f;
  float lsbDelta = 0.0f;
  float rsbDelta = 0.0f;
  std::vector<unsigned char> pixels;
};

bool shapeTextRun(const FontAtlas& atlas,
                  std::string_view text,
                  std::vector<hb_glyph_info_t>* outInfos,
                  std::vector<hb_glyph_position_t>* outPositions) {
  if (!atlas.runtime || !atlas.runtime->hbFont || !outInfos || !outPositions) return false;
  hb_buffer_t* buffer = hb_buffer_create();
  if (!buffer) return false;
  hb_buffer_add_utf8(buffer, text.data(), static_cast<int>(text.size()), 0, static_cast<int>(text.size()));
  hb_buffer_guess_segment_properties(buffer);
  // The viewer atlas only packs the base ASCII glyph set plus a few explicit extras.
  // If HarfBuzz substitutes typographic ligatures such as fi/ff/ffi, shaping can emit
  // glyph indices that are not present in the atlas, which shows up as dropped letters
  // in HUD text. Disable discretionary/standard ligature substitution for this atlas path
  // so each ASCII character maps back to a glyph we actually packed.
  const hb_feature_t features[] = {
      {HB_TAG('l', 'i', 'g', 'a'), 0, 0, static_cast<unsigned int>(-1)},
      {HB_TAG('c', 'l', 'i', 'g'), 0, 0, static_cast<unsigned int>(-1)},
      {HB_TAG('d', 'l', 'i', 'g'), 0, 0, static_cast<unsigned int>(-1)},
      {HB_TAG('h', 'l', 'i', 'g'), 0, 0, static_cast<unsigned int>(-1)},
  };
  hb_shape(atlas.runtime->hbFont, buffer, features, static_cast<unsigned int>(std::size(features)));
  unsigned int glyphCount = 0;
  const hb_glyph_info_t* infos = hb_buffer_get_glyph_infos(buffer, &glyphCount);
  const hb_glyph_position_t* positions = hb_buffer_get_glyph_positions(buffer, &glyphCount);
  outInfos->assign(infos, infos + glyphCount);
  outPositions->assign(positions, positions + glyphCount);
  hb_buffer_destroy(buffer);
  return true;
}

}  // namespace

bool loadFontAtlas(const std::string& fontPath, int pixelSize, FontAtlas* out, std::string* error) {
  if (!out) return false;
  *out = FontAtlas{};
  if (fontPath.empty()) {
    if (error) *error = "Font path is empty.";
    return false;
  }
  if (pixelSize <= 0) {
    if (error) *error = "Font pixel size must be positive.";
    return false;
  }

  std::shared_ptr<FontRuntime> runtime = std::make_shared<FontRuntime>();
  if (FT_Init_FreeType(&runtime->library) != 0) {
    if (error) *error = "FT_Init_FreeType failed.";
    return false;
  }

  if (FT_New_Face(runtime->library, fontPath.c_str(), 0, &runtime->face) != 0) {
    if (error) *error = std::string("Failed to load font face: ") + fontPath;
    return false;
  }

  FT_Select_Charmap(runtime->face, FT_ENCODING_UNICODE);

  if (FT_Set_Pixel_Sizes(runtime->face, 0, static_cast<FT_UInt>(pixelSize)) != 0) {
    if (error) *error = "FT_Set_Pixel_Sizes failed.";
    return false;
  }
  runtime->loadFlags = FT_LOAD_DEFAULT | FT_LOAD_TARGET_LIGHT;
  runtime->hbFont = hb_ft_font_create_referenced(runtime->face);
  if (!runtime->hbFont) {
    if (error) *error = "hb_ft_font_create_referenced failed.";
    return false;
  }
  hb_ft_font_set_load_flags(runtime->hbFont, runtime->loadFlags);
  hb_ft_font_set_funcs(runtime->hbFont);

  std::array<GlyphBitmapInfo, 128> glyphBitmaps{};
  std::unordered_map<unsigned int, GlyphBitmapInfo> extraGlyphBitmaps;
  int penX = kPadding;
  int penY = kPadding;
  int rowHeight = 0;
  int atlasHeight = kPadding;

  auto registerGlyphBitmap = [&](uint32_t codepoint, GlyphBitmapInfo* target) {
    if (!target) return;
    const unsigned int glyphIndex = FT_Get_Char_Index(runtime->face, codepoint);
    if (glyphIndex == 0) return;
    if (FT_Load_Glyph(runtime->face, glyphIndex, runtime->loadFlags) != 0) return;
    if (FT_Render_Glyph(runtime->face->glyph, FT_RENDER_MODE_NORMAL) != 0) return;
    const FT_GlyphSlot glyph = runtime->face->glyph;
    GlyphBitmapInfo info{};
    info.glyphIndex = glyphIndex;
    info.width = static_cast<int>(glyph->bitmap.width);
    info.height = static_cast<int>(glyph->bitmap.rows);
    info.bitmapLeft = static_cast<float>(glyph->bitmap_left);
    info.bitmapTop = static_cast<float>(glyph->bitmap_top);
    info.advanceX = static_cast<float>(glyph->advance.x) / 64.0f;
    info.lsbDelta = static_cast<float>(glyph->lsb_delta) / 64.0f;
    info.rsbDelta = static_cast<float>(glyph->rsb_delta) / 64.0f;
    const int bitmapSize = info.width * info.height;
    info.pixels.resize(static_cast<std::size_t>(std::max(bitmapSize, 0)));
    if (bitmapSize > 0) {
      for (int row = 0; row < info.height; ++row) {
        std::memcpy(info.pixels.data() + row * info.width,
                    glyph->bitmap.buffer + row * glyph->bitmap.pitch,
                    static_cast<std::size_t>(info.width));
      }
    }

    if (penX + info.width + kPadding > kAtlasWidth) {
      penX = kPadding;
      penY += rowHeight + kPadding;
      rowHeight = 0;
    }
    penX += info.width + kPadding;
    rowHeight = std::max(rowHeight, info.height);
    atlasHeight = std::max(atlasHeight, penY + rowHeight + kPadding);
    *target = std::move(info);
  };

  for (unsigned char c = kFirstGlyph; c <= kLastGlyph; ++c) {
    registerGlyphBitmap(c, &glyphBitmaps[c]);
  }
  for (uint32_t codepoint : kExtraCodepoints) {
    GlyphBitmapInfo info{};
    registerGlyphBitmap(codepoint, &info);
    if (info.glyphIndex != 0) {
      extraGlyphBitmaps[codepoint] = std::move(info);
    }
  }

  FontAtlas atlas{};
  atlas.width = kAtlasWidth;
  atlas.height = std::max(atlasHeight, pixelSize + 2 * kPadding);
  atlas.pixelSize = pixelSize;
  atlas.lineHeight = static_cast<int>(runtime->face->size->metrics.height >> 6);
  atlas.ascent = static_cast<int>(runtime->face->size->metrics.ascender >> 6);
  atlas.descent = static_cast<int>(-(runtime->face->size->metrics.descender >> 6));
  atlas.pixels.assign(static_cast<std::size_t>(atlas.width * atlas.height), 0u);
  atlas.runtime = runtime;

  penX = kPadding;
  penY = kPadding;
  rowHeight = 0;
  for (unsigned char c = kFirstGlyph; c <= kLastGlyph; ++c) {
    const GlyphBitmapInfo& info = glyphBitmaps[c];
    if (penX + info.width + kPadding > atlas.width) {
      penX = kPadding;
      penY += rowHeight + kPadding;
      rowHeight = 0;
    }

    GlyphMetrics metrics{};
    metrics.advanceX = info.advanceX;
    metrics.bitmapLeft = info.bitmapLeft;
    metrics.bitmapTop = info.bitmapTop;
    metrics.width = info.width;
    metrics.height = info.height;
    metrics.lsbDelta = info.lsbDelta;
    metrics.rsbDelta = info.rsbDelta;
    metrics.advance26_6 = static_cast<int>(std::lround(info.advanceX * 64.0f));
    metrics.lsbDelta26_6 = static_cast<int>(std::lround(info.lsbDelta * 64.0f));
    metrics.rsbDelta26_6 = static_cast<int>(std::lround(info.rsbDelta * 64.0f));
    metrics.valid = true;
    if (info.width > 0 && info.height > 0) {
      const int texY = atlas.height - penY - info.height;
      for (int row = 0; row < info.height; ++row) {
        unsigned char* dst =
            atlas.pixels.data() + (texY + (info.height - 1 - row)) * atlas.width + penX;
        const unsigned char* src = info.pixels.data() + row * info.width;
        std::memcpy(dst, src, static_cast<std::size_t>(info.width));
      }
      metrics.u0 = (static_cast<float>(penX) + 0.5f) / static_cast<float>(atlas.width);
      metrics.u1 = (static_cast<float>(penX + info.width) - 0.5f) / static_cast<float>(atlas.width);
      metrics.v0 = (static_cast<float>(texY + info.height) - 0.5f) / static_cast<float>(atlas.height);
      metrics.v1 = (static_cast<float>(texY) + 0.5f) / static_cast<float>(atlas.height);
    }
    atlas.glyphs[c] = metrics;
    atlas.glyphIndexForChar[c] = info.glyphIndex;
    atlas.glyphsByIndex[info.glyphIndex] = metrics;
    penX += info.width + kPadding;
    rowHeight = std::max(rowHeight, info.height);
  }
  for (const auto& entry : extraGlyphBitmaps) {
    const GlyphBitmapInfo& info = entry.second;
    if (penX + info.width + kPadding > atlas.width) {
      penX = kPadding;
      penY += rowHeight + kPadding;
      rowHeight = 0;
    }

    GlyphMetrics metrics{};
    metrics.advanceX = info.advanceX;
    metrics.bitmapLeft = info.bitmapLeft;
    metrics.bitmapTop = info.bitmapTop;
    metrics.width = info.width;
    metrics.height = info.height;
    metrics.lsbDelta = info.lsbDelta;
    metrics.rsbDelta = info.rsbDelta;
    metrics.advance26_6 = static_cast<int>(std::lround(info.advanceX * 64.0f));
    metrics.lsbDelta26_6 = static_cast<int>(std::lround(info.lsbDelta * 64.0f));
    metrics.rsbDelta26_6 = static_cast<int>(std::lround(info.rsbDelta * 64.0f));
    metrics.valid = true;
    if (info.width > 0 && info.height > 0) {
      const int texY = atlas.height - penY - info.height;
      for (int row = 0; row < info.height; ++row) {
        unsigned char* dst =
            atlas.pixels.data() + (texY + (info.height - 1 - row)) * atlas.width + penX;
        const unsigned char* src = info.pixels.data() + row * info.width;
        std::memcpy(dst, src, static_cast<std::size_t>(info.width));
      }
      metrics.u0 = (static_cast<float>(penX) + 0.5f) / static_cast<float>(atlas.width);
      metrics.u1 = (static_cast<float>(penX + info.width) - 0.5f) / static_cast<float>(atlas.width);
      metrics.v0 = (static_cast<float>(texY + info.height) - 0.5f) / static_cast<float>(atlas.height);
      metrics.v1 = (static_cast<float>(texY) + 0.5f) / static_cast<float>(atlas.height);
    }
    atlas.glyphsByIndex[info.glyphIndex] = metrics;
    penX += info.width + kPadding;
    rowHeight = std::max(rowHeight, info.height);
  }

  if (FT_HAS_KERNING(runtime->face)) {
    for (unsigned char left = kFirstGlyph; left <= kLastGlyph; ++left) {
      const FT_UInt leftIndex = FT_Get_Char_Index(runtime->face, left);
      if (leftIndex == 0) continue;
      for (unsigned char right = kFirstGlyph; right <= kLastGlyph; ++right) {
        const FT_UInt rightIndex = FT_Get_Char_Index(runtime->face, right);
        if (rightIndex == 0) continue;
        FT_Vector delta{};
        if (FT_Get_Kerning(runtime->face, leftIndex, rightIndex, FT_KERNING_DEFAULT, &delta) == 0) {
          atlas.kerningX[static_cast<std::size_t>(left) * 128u + static_cast<std::size_t>(right)] =
              static_cast<float>(delta.x) / 64.0f;
          atlas.kerningX26_6[static_cast<std::size_t>(left) * 128u + static_cast<std::size_t>(right)] =
              static_cast<int>(delta.x);
        }
      }
    }
  }

  atlas.valid = true;
  *out = std::move(atlas);
  return true;
}

float measureTextWidth(const FontAtlas& atlas, std::string_view text, float scale) {
  float width = 0.0f;
  if (!atlas.valid) return width;
  if (atlas.runtime && atlas.runtime->hbFont) {
    std::vector<hb_glyph_info_t> infos;
    std::vector<hb_glyph_position_t> positions;
    if (shapeTextRun(atlas, text, &infos, &positions)) {
      hb_position_t penX = 0;
      int minX = 0;
      int maxX = 0;
      for (std::size_t i = 0; i < infos.size(); ++i) {
        const auto it = atlas.glyphsByIndex.find(infos[i].codepoint);
        if (it == atlas.glyphsByIndex.end()) {
          penX += positions[i].x_advance;
          continue;
        }
        const GlyphMetrics& glyph = it->second;
        const int x0 = static_cast<int>((penX + positions[i].x_offset) >> 6) + static_cast<int>(glyph.bitmapLeft);
        const int x1 = x0 + static_cast<int>(glyph.width);
        minX = std::min(minX, x0);
        maxX = std::max(maxX, x1);
        penX += positions[i].x_advance;
      }
      maxX = std::max(maxX, static_cast<int>(penX >> 6));
      return static_cast<float>(maxX - minX) * scale;
    }
  }
  if (std::fabs(scale - 1.0f) < 1e-4f) {
    int penX26_6 = 0;
    int minX = 0;
    int maxX = 0;
    unsigned char prev = 0;
    int prevRsbDelta26_6 = 0;
    for (unsigned char c : text) {
      if (c >= atlas.glyphs.size()) continue;
      const GlyphMetrics& glyph = atlas.glyphs[c];
      if (prev != 0) {
        penX26_6 += atlas.kerningX26_6[static_cast<std::size_t>(prev) * 128u + static_cast<std::size_t>(c)];
        const int delta = prevRsbDelta26_6 - glyph.lsbDelta26_6;
        if (delta > 32) penX26_6 -= 64;
        else if (delta < -31) penX26_6 += 64;
      }
      const int x0 = (penX26_6 >> 6) + static_cast<int>(glyph.bitmapLeft);
      const int x1 = x0 + static_cast<int>(glyph.width);
      minX = std::min(minX, x0);
      maxX = std::max(maxX, x1);
      penX26_6 += glyph.advance26_6;
      prev = c;
      prevRsbDelta26_6 = glyph.rsbDelta26_6;
    }
    maxX = std::max(maxX, penX26_6 >> 6);
    return static_cast<float>(maxX - minX);
  }
  unsigned char prev = 0;
  float prevRsbDelta = 0.0f;
  for (unsigned char c : text) {
    if (c >= atlas.glyphs.size()) continue;
    if (prev != 0) {
      width += static_cast<float>(
          atlas.kerningX[static_cast<std::size_t>(prev) * 128u + static_cast<std::size_t>(c)]) * scale;
    }
    const GlyphMetrics& glyph = atlas.glyphs[c];
    if (prev != 0 && scale > 0.0f) {
      const float delta = (prevRsbDelta - glyph.lsbDelta) * scale;
      if (delta > 0.49f) width -= scale;
      else if (delta < -0.49f) width += scale;
    }
    width += static_cast<float>(glyph.advanceX) * scale;
    prev = c;
    prevRsbDelta = glyph.rsbDelta;
  }
  return width;
}

void appendTextQuads(const FontAtlas& atlas,
                     std::string_view text,
                     float baselineX,
                     float baselineY,
                     float scale,
                     std::vector<TextQuadVertex>* out) {
  if (!out || !atlas.valid) return;
  if (atlas.runtime && atlas.runtime->hbFont) {
    std::vector<hb_glyph_info_t> infos;
    std::vector<hb_glyph_position_t> positions;
    if (shapeTextRun(atlas, text, &infos, &positions)) {
      const float safeScale = scale;
      float penX = baselineX;
      const float baselineYPx = baselineY;
      for (std::size_t i = 0; i < infos.size(); ++i) {
        const auto it = atlas.glyphsByIndex.find(infos[i].codepoint);
        if (it == atlas.glyphsByIndex.end()) {
          penX += static_cast<float>(positions[i].x_advance) / 64.0f * safeScale;
          continue;
        }
        const GlyphMetrics& glyph = it->second;
        if (glyph.width > 0.0f && glyph.height > 0.0f) {
          const float x0 = penX + (static_cast<float>(positions[i].x_offset) / 64.0f +
                                   static_cast<float>(glyph.bitmapLeft)) * safeScale;
          const float top = baselineYPx + (static_cast<float>(positions[i].y_offset) / 64.0f +
                                           static_cast<float>(glyph.bitmapTop)) * safeScale;
          const float bottom = top - glyph.height * safeScale;
          const float x1 = x0 + glyph.width * safeScale;
          out->push_back({x0, bottom, glyph.u0, glyph.v1});
          out->push_back({x1, bottom, glyph.u1, glyph.v1});
          out->push_back({x1, top, glyph.u1, glyph.v0});
          out->push_back({x0, bottom, glyph.u0, glyph.v1});
          out->push_back({x1, top, glyph.u1, glyph.v0});
          out->push_back({x0, top, glyph.u0, glyph.v0});
        }
        penX += static_cast<float>(positions[i].x_advance) / 64.0f * safeScale;
      }
      return;
    }
  }
  if (std::fabs(scale - 1.0f) < 1e-4f) {
    int penX26_6 = static_cast<int>(std::lround(baselineX * 64.0f));
    const int baselineYPx = static_cast<int>(std::lround(baselineY));
    unsigned char prev = 0;
    int prevRsbDelta26_6 = 0;
    for (unsigned char c : text) {
      if (c >= atlas.glyphs.size()) continue;
      const GlyphMetrics& glyph = atlas.glyphs[c];
      if (prev != 0) {
        penX26_6 += atlas.kerningX26_6[static_cast<std::size_t>(prev) * 128u + static_cast<std::size_t>(c)];
        const int delta = prevRsbDelta26_6 - glyph.lsbDelta26_6;
        if (delta > 32) penX26_6 -= 64;
        else if (delta < -31) penX26_6 += 64;
      }
      if (glyph.width > 0.0f && glyph.height > 0.0f) {
        const float x0 = static_cast<float>((penX26_6 >> 6) + static_cast<int>(glyph.bitmapLeft));
        const float top = static_cast<float>(baselineYPx + static_cast<int>(glyph.bitmapTop));
        const float bottom = top - glyph.height;
        const float x1 = x0 + glyph.width;
        out->push_back({x0, bottom, glyph.u0, glyph.v1});
        out->push_back({x1, bottom, glyph.u1, glyph.v1});
        out->push_back({x1, top, glyph.u1, glyph.v0});
        out->push_back({x0, bottom, glyph.u0, glyph.v1});
        out->push_back({x1, top, glyph.u1, glyph.v0});
        out->push_back({x0, top, glyph.u0, glyph.v0});
      }
      penX26_6 += glyph.advance26_6;
      prev = c;
      prevRsbDelta26_6 = glyph.rsbDelta26_6;
    }
    return;
  }
  float penX = baselineX;
  unsigned char prev = 0;
  float prevRsbDelta = 0.0f;
  for (unsigned char c : text) {
    if (c >= atlas.glyphs.size()) continue;
    if (prev != 0) {
      penX += static_cast<float>(
          atlas.kerningX[static_cast<std::size_t>(prev) * 128u + static_cast<std::size_t>(c)]) * scale;
    }
    const GlyphMetrics& glyph = atlas.glyphs[c];
    if (prev != 0 && scale > 0.0f) {
      const float delta = (prevRsbDelta - glyph.lsbDelta) * scale;
      if (delta > 0.49f) penX -= scale;
      else if (delta < -0.49f) penX += scale;
    }
    if (glyph.width > 0 && glyph.height > 0) {
      const float x0 = penX + static_cast<float>(glyph.bitmapLeft) * scale;
      const float top = baselineY + static_cast<float>(glyph.bitmapTop) * scale;
      const float bottom = top - static_cast<float>(glyph.height) * scale;
      const float x1 = x0 + static_cast<float>(glyph.width) * scale;
      out->push_back({x0, bottom, glyph.u0, glyph.v1});
      out->push_back({x1, bottom, glyph.u1, glyph.v1});
      out->push_back({x1, top, glyph.u1, glyph.v0});
      out->push_back({x0, bottom, glyph.u0, glyph.v1});
      out->push_back({x1, top, glyph.u1, glyph.v0});
      out->push_back({x0, top, glyph.u0, glyph.v0});
    }
    penX += static_cast<float>(glyph.advanceX) * scale;
    prev = c;
    prevRsbDelta = glyph.rsbDelta;
  }
}

}  // namespace WorkshopText
