#include "../../../src/ChromaspaceViewerLayout.h"
#include "../../../src/ChromaspaceViewerState.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using ChromaspaceViewer::PlotWindowDragMode;
using ChromaspaceViewer::PlotWindowRectNorm;
using ChromaspaceViewer::WorkspaceGeometry;

bool near(float a, float b, float epsilon = 1e-5f) {
  return std::fabs(a - b) <= epsilon;
}

bool contains(const PlotWindowRectNorm& rect, float x, float y) {
  return x >= rect.x && x < rect.x + rect.w &&
         y >= rect.y && y < rect.y + rect.h;
}

void assertRectNear(const PlotWindowRectNorm& actual,
                    const PlotWindowRectNorm& expected,
                    float epsilon = 1e-5f) {
  assert(near(actual.x, expected.x, epsilon));
  assert(near(actual.y, expected.y, epsilon));
  assert(near(actual.w, expected.w, epsilon));
  assert(near(actual.h, expected.h, epsilon));
}

void workspaceGeometryAndClamping() {
  const WorkspaceGeometry noToolbar =
      ChromaspaceViewer::workspaceGeometry(false, 600);
  assert(near(noToolbar.topNorm, 0.0f));
  assert(near(noToolbar.heightNorm, 1.0f));

  const WorkspaceGeometry toolbar =
      ChromaspaceViewer::workspaceGeometry(true, 600);
  assert(near(toolbar.topNorm, 42.0f / 600.0f));
  assert(near(toolbar.heightNorm, 1.0f - 42.0f / 600.0f));
  const WorkspaceGeometry tinyToolbar =
      ChromaspaceViewer::workspaceGeometry(true, 1);
  assert(near(tinyToolbar.topNorm, 0.45f));
  assert(near(tinyToolbar.heightNorm, 0.55f));

  assert(near(ChromaspaceViewer::plotWindowMinNormWidth(720), 0.25f));
  assert(near(ChromaspaceViewer::plotWindowMinNormHeight(700), 0.2f));
  assert(near(ChromaspaceViewer::plotWindowMinNormWidth(1), 0.88f));
  assert(near(ChromaspaceViewer::plotWindowMinNormHeight(1), 0.88f));

  assertRectNear(ChromaspaceViewer::workspaceRelativeRect(toolbar,
                                                          0.1f,
                                                          0.2f,
                                                          0.4f,
                                                          0.5f),
                 {0.1f, toolbar.topNorm + toolbar.heightNorm * 0.2f,
                  0.4f, toolbar.heightNorm * 0.5f});

  const PlotWindowRectNorm clamped =
      ChromaspaceViewer::clampPlotWindowRect({-0.5f, -0.4f, 1.5f, 1.5f},
                                              toolbar,
                                              720,
                                              600);
  assert(clamped.x >= 0.0f && clamped.y >= toolbar.topNorm);
  assert(clamped.x + clamped.w <= 1.0f + 1e-6f);
  assert(clamped.y + clamped.h <= 1.0f + 1e-6f);
  assert(clamped.w >= ChromaspaceViewer::plotWindowMinNormWidth(720));
  assert(clamped.h >= ChromaspaceViewer::plotWindowMinNormHeight(600));

  const PlotWindowRectNorm narrow =
      ChromaspaceViewer::clampPlotWindowRect({0.1f, 0.1f, 0.1f, 0.1f},
                                              toolbar,
                                              720,
                                              600);
  assert(narrow.y >= toolbar.topNorm);
  assert(narrow.w >= 0.25f);
  assert(narrow.h >= 140.0f / 600.0f);
}

void workspaceTopReflow() {
  const PlotWindowRectNorm original{0.1f, 0.28f, 0.4f, 0.4f};
  const PlotWindowRectNorm reflowed =
      ChromaspaceViewer::reflowPlotWindowRectForWorkspaceTop(original,
                                                              0.1f,
                                                              0.2f);
  assertRectNear(reflowed,
                 {0.1f,
                  0.2f + ((0.28f - 0.1f) / 0.9f) * 0.8f,
                  0.4f,
                  0.4f / 0.9f * 0.8f});
  const PlotWindowRectNorm tiny =
      ChromaspaceViewer::reflowPlotWindowRectForWorkspaceTop(
          original, 0.99f, 0.995f);
  assert(std::isfinite(tiny.y) && std::isfinite(tiny.h));
}

void dragHitTesting() {
  const PlotWindowRectNorm rect{0.2f, 0.2f, 0.5f, 0.5f};
  const ChromaspaceViewer::ScreenRect screen{100.0f, 50.0f, 500.0f, 350.0f};
  auto hit = [&](float x, float y, const PlotWindowRectNorm& testRect) {
    return ChromaspaceViewer::plotWindowDragModeAt(
        {testRect, screen, x, y});
  };
  assert(hit(90.0f, 200.0f, rect) == PlotWindowDragMode::None);
  assert(hit(300.0f, 200.0f, rect) == PlotWindowDragMode::None);
  assert(hit(300.0f, 60.0f, rect) == PlotWindowDragMode::Move);
  assert(hit(103.0f, 200.0f, rect) == PlotWindowDragMode::ResizeLeft);
  assert(hit(497.0f, 200.0f, rect) == PlotWindowDragMode::ResizeRight);
  assert(hit(300.0f, 53.0f, rect) == PlotWindowDragMode::ResizeTop);
  assert(hit(300.0f, 347.0f, rect) == PlotWindowDragMode::ResizeBottom);
  assert(hit(103.0f, 53.0f, rect) == PlotWindowDragMode::ResizeTopLeft);
  assert(hit(497.0f, 53.0f, rect) == PlotWindowDragMode::ResizeTopRight);
  assert(hit(103.0f, 347.0f, rect) == PlotWindowDragMode::ResizeBottomLeft);
  assert(hit(497.0f, 347.0f, rect) == PlotWindowDragMode::ResizeBottomRight);
  const PlotWindowRectNorm nearlyFull{0.0f, 0.0f, 1.0f, 0.994f};
  assert(hit(300.0f, 60.0f, nearlyFull) == PlotWindowDragMode::None);
}

void dragApplication() {
  const WorkspaceGeometry workspace{0.1f, 0.9f};
  const PlotWindowRectNorm start{0.2f, 0.2f, 0.4f, 0.4f};
  const float minW = 0.2f;
  const float minH = 0.2f;
  auto apply = [&](PlotWindowDragMode mode, float dx, float dy) {
    return ChromaspaceViewer::applyPlotWindowDrag(
        {start, mode, dx, dy, workspace, minW, minH});
  };
  assertRectNear(apply(PlotWindowDragMode::Move, 0.1f, 0.1f),
                 {0.3f, 0.3f, 0.4f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::ResizeLeft, 0.1f, 0.0f),
                 {0.3f, 0.2f, 0.3f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::ResizeRight, 0.1f, 0.0f),
                 {0.2f, 0.2f, 0.5f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::ResizeTop, 0.0f, 0.1f),
                 {0.2f, 0.3f, 0.4f, 0.3f});
  assertRectNear(apply(PlotWindowDragMode::ResizeBottom, 0.0f, 0.1f),
                 {0.2f, 0.2f, 0.4f, 0.5f});
  assertRectNear(apply(PlotWindowDragMode::ResizeTopLeft, 0.1f, 0.1f),
                 {0.3f, 0.3f, 0.3f, 0.3f});
  assertRectNear(apply(PlotWindowDragMode::ResizeTopRight, 0.1f, 0.1f),
                 {0.2f, 0.3f, 0.5f, 0.3f});
  assertRectNear(apply(PlotWindowDragMode::ResizeBottomLeft, 0.1f, 0.1f),
                 {0.3f, 0.2f, 0.3f, 0.5f});
  assertRectNear(apply(PlotWindowDragMode::ResizeBottomRight, 0.1f, 0.1f),
                 {0.2f, 0.2f, 0.5f, 0.5f});

  assertRectNear(apply(PlotWindowDragMode::ResizeLeft, 0.5f, 0.0f),
                 {0.4f, 0.2f, 0.2f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::ResizeRight, -0.5f, 0.0f),
                 {0.2f, 0.2f, 0.2f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::ResizeTop, 0.0f, 0.5f),
                 {0.2f, 0.4f, 0.4f, 0.2f});
  assertRectNear(apply(PlotWindowDragMode::ResizeBottom, 0.0f, -0.5f),
                 {0.2f, 0.2f, 0.4f, 0.2f});
  assertRectNear(apply(PlotWindowDragMode::Move, -2.0f, -2.0f),
                 {0.0f, 0.1f, 0.4f, 0.4f});
  assertRectNear(apply(PlotWindowDragMode::Move, 2.0f, 2.0f),
                 {0.6f, 0.6f, 0.4f, 0.4f});
}

ChromaspaceViewer::PlotWindowSnapPreviewResult snap(
    PlotWindowRectNorm candidate,
    PlotWindowDragMode mode,
    float cursorX,
    float cursorY,
    bool single = false,
    std::vector<PlotWindowRectNorm> others = {},
    float reservedLeftPixels = 0.0f) {
  return ChromaspaceViewer::computePlotWindowSnapPreview(
      {candidate,
       mode,
       {0.1f, 0.9f},
       1000,
       800,
       reservedLeftPixels,
       cursorX,
       cursorY,
       single,
       std::move(others)});
}

void snapPreviewCandidates() {
  const PlotWindowRectNorm candidate{0.2f, 0.25f, 0.35f, 0.35f};
  assertRectNear(snap(candidate, PlotWindowDragMode::Move, 5.0f, 400.0f, true).rect,
                 {0.0f, 0.1f, 1.0f, 0.9f});

  const std::array<std::pair<float, float>, 4> corners = {{
      {10.0f, 90.0f}, {990.0f, 90.0f}, {10.0f, 790.0f}, {990.0f, 790.0f}}};
  const std::array<PlotWindowRectNorm, 4> cornerRects = {{
      {0.0f, 0.1f, 0.5f, 0.45f},
      {0.5f, 0.1f, 0.5f, 0.45f},
      {0.0f, 0.55f, 0.5f, 0.45f},
      {0.5f, 0.55f, 0.5f, 0.45f}}};
  for (std::size_t i = 0; i < corners.size(); ++i) {
    const auto result = snap(candidate,
                             PlotWindowDragMode::Move,
                             corners[i].first,
                             corners[i].second);
    assert(result.visible);
    assertRectNear(result.rect, cornerRects[i]);
  }

  const std::array<std::pair<float, float>, 4> edges = {{
      {5.0f, 400.0f}, {995.0f, 400.0f}, {500.0f, 90.0f}, {500.0f, 790.0f}}};
  const std::array<PlotWindowRectNorm, 4> edgeRects = {{
      {0.0f, 0.1f, 0.5f, 0.9f},
      {0.5f, 0.1f, 0.5f, 0.9f},
      {0.0f, 0.1f, 1.0f, 0.45f},
      {0.0f, 0.55f, 1.0f, 0.45f}}};
  for (std::size_t i = 0; i < edges.size(); ++i) {
    const auto result = snap(candidate,
                             PlotWindowDragMode::Move,
                             edges[i].first,
                             edges[i].second);
    assert(result.visible);
    assertRectNear(result.rect, edgeRects[i]);
  }

  const auto centerMove = snap({0.49f, 0.3f, 0.2f, 0.2f},
                               PlotWindowDragMode::Move,
                               500.0f,
                               400.0f);
  assert(centerMove.visible);
  assert(near(centerMove.rect.x, 0.5f));
  const auto thirdMove = snap({1.0f / 3.0f + 0.005f,
                               0.1f + 0.9f / 3.0f + 0.005f,
                               0.2f,
                               0.2f},
                              PlotWindowDragMode::Move,
                              500.0f,
                              400.0f);
  assert(thirdMove.visible);
  assert(near(thirdMove.rect.x, 1.0f / 3.0f));
  assert(near(thirdMove.rect.y, 0.1f + 0.9f / 3.0f));

  const auto neighborMove = snap({0.505f, 0.3f, 0.2f, 0.2f},
                                 PlotWindowDragMode::Move,
                                 500.0f,
                                 400.0f,
                                 false,
                                 {{0.2f, 0.2f, 0.3f, 0.3f}});
  assert(neighborMove.visible);
  assert(near(neighborMove.rect.x, 0.5f));

  const auto resizeRight = snap({0.2f, 0.3f, 0.29f, 0.2f},
                                PlotWindowDragMode::ResizeRight,
                                500.0f,
                                400.0f);
  assert(resizeRight.visible);
  assert(near(resizeRight.rect.x, 0.2f));
  assert(near(resizeRight.rect.w, 0.3f));
  const auto resizeBottom = snap({0.2f, 0.2f, 0.2f, 0.345f},
                                 PlotWindowDragMode::ResizeBottom,
                                 500.0f,
                                 400.0f);
  assert(resizeBottom.visible);
  assert(near(resizeBottom.rect.h, 0.35f));

  const auto resizeLeft = snap({0.505f, 0.3f, 0.295f, 0.2f},
                               PlotWindowDragMode::ResizeLeft,
                               500.0f,
                               400.0f);
  assert(resizeLeft.visible);
  assert(near(resizeLeft.rect.x, 0.5f));
  assert(near(resizeLeft.rect.w, 0.3f));
  const auto resizeTop = snap({0.2f, 0.545f, 0.2f, 0.255f},
                              PlotWindowDragMode::ResizeTop,
                              500.0f,
                              400.0f);
  assert(resizeTop.visible);
  assert(near(resizeTop.rect.y, 0.55f));
  assert(near(resizeTop.rect.h, 0.25f));
  const auto resizeTopLeft = snap({0.505f, 0.545f, 0.295f, 0.255f},
                                  PlotWindowDragMode::ResizeTopLeft,
                                  500.0f,
                                  400.0f);
  assert(resizeTopLeft.visible);
  assert(near(resizeTopLeft.rect.x, 0.5f));
  assert(near(resizeTopLeft.rect.w, 0.3f));
  assert(near(resizeTopLeft.rect.y, 0.55f));
  assert(near(resizeTopLeft.rect.h, 0.25f));

  const auto reservedLeft = snap(candidate,
                                 PlotWindowDragMode::Move,
                                 205.0f,
                                 400.0f,
                                 false,
                                 {},
                                 200.0f);
  assert(reservedLeft.visible);
  assertRectNear(reservedLeft.rect, {0.0f, 0.1f, 0.5f, 0.9f});

  const auto none = snap({0.123f, 0.25f, 0.234f, 0.2f},
                         PlotWindowDragMode::Move,
                         500.0f,
                         400.0f);
  assert(!none.visible);
}

void catalogAndNames() {
  static constexpr std::array<const char*, ChromaspaceViewer::kViewerLayoutChoiceCount> labels = {
      "Single", "Split 2", "Triple Columns", "Triple 2 + 1", "Quadrants", "Six Views", "Solo"};
  static constexpr std::array<int, ChromaspaceViewer::kViewerLayoutChoiceCount> displayOrder = {
      ChromaspaceViewer::kViewerLayoutSoloIndex, 0, 1, 2, 3, 4, 5};
  static constexpr std::array<int, ChromaspaceViewer::kViewerLayoutChoiceCount> required = {
      1, 2, 3, 3, 4, 6, 1};
  static constexpr std::array<std::pair<int, int>, ChromaspaceViewer::kViewerLayoutChoiceCount> preferred = {
      std::pair<int, int>{720, 600},
      std::pair<int, int>{1120, 640},
      std::pair<int, int>{1320, 720},
      std::pair<int, int>{1280, 760},
      std::pair<int, int>{1280, 900},
      std::pair<int, int>{1468, 1030},
      std::pair<int, int>{720, 600}};

  for (int index = 0; index < ChromaspaceViewer::kViewerLayoutChoiceCount; ++index) {
    const auto& layout = ChromaspaceViewer::standardPlotLayout(index);
    assert(layout.index == index);
    assert(std::string(layout.label) == labels[static_cast<std::size_t>(index)]);
    assert(layout.requiredWindowCount == required[static_cast<std::size_t>(index)]);
    const std::pair<int, int> layoutSize{layout.preferredWindowWidth,
                                         layout.preferredWindowHeight};
    assert(layoutSize == preferred[static_cast<std::size_t>(index)]);
    assert(ChromaspaceViewer::findStandardPlotLayout(layout.label) == &layout);
  }
  assert(ChromaspaceViewer::standardPlotLayout(-1).index == 0);
  assert(ChromaspaceViewer::standardPlotLayout(99).index == ChromaspaceViewer::kViewerLayoutSoloIndex);

  for (int row = 0; row < ChromaspaceViewer::kViewerLayoutChoiceCount; ++row) {
    assert(ChromaspaceViewer::standardPlotLayoutForDisplayRow(row).index ==
           displayOrder[static_cast<std::size_t>(row)]);
  }
  assert(ChromaspaceViewer::standardPlotLayoutForDisplayRow(-1).index ==
         ChromaspaceViewer::kViewerLayoutSoloIndex);
  assert(ChromaspaceViewer::standardPlotLayoutForDisplayRow(99).index == 5);

  assert(ChromaspaceViewer::findStandardPlotLayout("sInGlE")->index == 0);
  assert(ChromaspaceViewer::findStandardPlotLayout("SIX VIEWS")->index == 5);
  assert(ChromaspaceViewer::findStandardPlotLayout("missing") == nullptr);
  assert(ChromaspaceViewer::isStandardPlotLayoutNameReserved("CUSTOM"));
  assert(ChromaspaceViewer::isStandardPlotLayoutNameReserved("qUaDrAnTs"));
  assert(!ChromaspaceViewer::isStandardPlotLayoutNameReserved("Custom Layout"));
  assert(!ChromaspaceViewer::isStandardPlotLayoutNameReserved(""));
}

void slotGeometryBoundsAndCoverage() {
  for (int index = 0; index < ChromaspaceViewer::kViewerLayoutChoiceCount; ++index) {
    const auto& layout = ChromaspaceViewer::standardPlotLayout(index);
    float area = 0.0f;
    for (int slot = 0; slot < layout.requiredWindowCount; ++slot) {
      const PlotWindowRectNorm rect =
          ChromaspaceViewer::standardPlotLayoutSlotRect(layout, slot);
      assert(rect.x >= -1e-6f && rect.y >= -1e-6f);
      assert(rect.x + rect.w <= 1.0f + 1e-6f);
      assert(rect.y + rect.h <= 1.0f + 1e-6f);
      assert(rect.w > 0.0f && rect.h > 0.0f);
      area += rect.w * rect.h;
    }
    assert(near(area, 1.0f));

    // Sample cell interiors so boundaries do not count twice. Every point in
    // the normalized workspace must belong to one and only one required slot.
    for (int iy = 0; iy < 17; ++iy) {
      for (int ix = 0; ix < 17; ++ix) {
        const float x = (static_cast<float>(ix) + 0.37f) / 17.0f;
        const float y = (static_cast<float>(iy) + 0.61f) / 17.0f;
        int matches = 0;
        for (int slot = 0; slot < layout.requiredWindowCount; ++slot) {
          matches += contains(ChromaspaceViewer::standardPlotLayoutSlotRect(layout, slot), x, y) ? 1 : 0;
        }
        assert(matches == 1);
      }
    }
  }
}

void dragTraits() {
  struct Expected {
    PlotWindowDragMode mode;
    bool resize;
    bool left;
    bool right;
    bool top;
    bool bottom;
  };
  const std::array<Expected, 10> expected = {{
      {PlotWindowDragMode::None, false, false, false, false, false},
      {PlotWindowDragMode::Move, false, false, false, false, false},
      {PlotWindowDragMode::ResizeLeft, true, true, false, false, false},
      {PlotWindowDragMode::ResizeRight, true, false, true, false, false},
      {PlotWindowDragMode::ResizeTop, true, false, false, true, false},
      {PlotWindowDragMode::ResizeBottom, true, false, false, false, true},
      {PlotWindowDragMode::ResizeTopLeft, true, true, false, true, false},
      {PlotWindowDragMode::ResizeTopRight, true, false, true, true, false},
      {PlotWindowDragMode::ResizeBottomLeft, true, true, false, false, true},
      {PlotWindowDragMode::ResizeBottomRight, true, false, true, false, true},
  }};
  for (const Expected& value : expected) {
    const auto traits = ChromaspaceViewer::plotWindowDragModeTraits(value.mode);
    assert(traits.isResize == value.resize);
    assert(traits.touchesLeft == value.left);
    assert(traits.touchesRight == value.right);
    assert(traits.touchesTop == value.top);
    assert(traits.touchesBottom == value.bottom);
  }
}

void defaultModelsAndRectTolerance() {
  static constexpr std::array<int, 6> models = {
      ChromaspaceViewer::kPlotModelCube,
      ChromaspaceViewer::kPlotModelHistogram,
      ChromaspaceViewer::kPlotModelSourceSignal,
      ChromaspaceViewer::kPlotModelHsl,
      ChromaspaceViewer::kPlotModelJpConical,
      ChromaspaceViewer::kPlotModelWaveform};
  for (int index = 0; index < ChromaspaceViewer::kViewerLayoutChoiceCount; ++index) {
    const auto& layout = ChromaspaceViewer::standardPlotLayout(index);
    for (int slot = -1; slot <= 6; ++slot) {
      const int model = ChromaspaceViewer::standardPlotLayoutDefaultPlotModel(layout, slot);
      if (index == ChromaspaceViewer::kViewerLayoutSoloIndex) {
        assert(model == -1);
      } else {
        assert(model >= 0 && model < ChromaspaceViewer::kPlotModelCount);
        assert(model == models[static_cast<std::size_t>(std::clamp(slot, 0, 5))]);
      }
    }
  }

  const PlotWindowRectNorm base{0.2f, 0.3f, 0.4f, 0.5f};
  PlotWindowRectNorm candidate = base;
  candidate.x += 0.0149f;
  assert(ChromaspaceViewer::plotWindowRectNear(base, candidate));
  candidate.x += 0.0002f;
  assert(!ChromaspaceViewer::plotWindowRectNear(base, candidate));
  candidate = base;
  candidate.h -= 0.0149f;
  assert(ChromaspaceViewer::plotWindowRectNear(base, candidate));
  candidate.h -= 0.0002f;
  assert(!ChromaspaceViewer::plotWindowRectNear(base, candidate));
}

}  // namespace

int main() {
  workspaceGeometryAndClamping();
  workspaceTopReflow();
  dragHitTesting();
  dragApplication();
  snapPreviewCandidates();
  catalogAndNames();
  slotGeometryBoundsAndCoverage();
  dragTraits();
  defaultModelsAndRectTolerance();
  std::cout << "Chromaspace viewer layout tests passed\n";
  return 0;
}
