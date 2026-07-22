#include "ChromaspaceViewerUiScene.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

namespace {

using ChromaspaceViewer::PlotWindowDragMode;
using ChromaspaceViewer::PlotWindowRectNorm;
using ChromaspaceViewer::ViewerFramePlan;
using ChromaspaceViewer::ViewerFramePlanRequest;
using ChromaspaceViewer::ViewerUiControlKind;
using ChromaspaceViewer::ViewerUiHitResult;
using ChromaspaceViewer::ViewerUiHitRegion;
using ChromaspaceViewer::ViewerUiPlotWindowInput;
using ChromaspaceViewer::ViewerUiPrimitiveKind;
using ChromaspaceViewer::ViewerUiScene;
using ChromaspaceViewer::ViewerUiSceneInput;
using ChromaspaceViewer::ViewerUiSceneStatus;
using ChromaspaceViewer::ViewerUiTextAlignment;
using ChromaspaceViewer::WorkspaceToolbarInput;

bool near(float a, float b, float epsilon = 1e-4f) {
  return std::fabs(a - b) <= epsilon;
}

bool contains(const ChromaspaceViewer::ScreenRect& outer,
              const ChromaspaceViewer::ScreenRect& inner) {
  return inner.x0 >= outer.x0 && inner.y0 >= outer.y0 &&
         inner.x1 <= outer.x1 && inner.y1 <= outer.y1;
}

ViewerFramePlan makePlan(bool retina = false,
                         bool overlap = true,
                         bool fullWindow = false) {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 360;
  request.framebufferWidth = retina ? 1280 : 640;
  request.framebufferHeight = retina ? 720 : 360;
  if (fullWindow) {
    request.windows = {{1, {0.0f, 0.0f, 1.0f, 1.0f}, 3, 10, true}};
  } else if (overlap) {
    request.windows = {
        {1, {0.10f, 0.10f, 0.50f, 0.50f}, 3, 10, true},
        {2, {0.20f, 0.20f, 0.50f, 0.50f}, 4, 11, true},
    };
  } else {
    request.windows = {{1, {0.10f, 0.10f, 0.50f, 0.50f}, 3, 10, true}};
  }
  return ChromaspaceViewer::buildViewerFramePlan(request);
}

ViewerFramePlan makeDisjointPlan() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 360;
  request.framebufferWidth = 640;
  request.framebufferHeight = 360;
  request.windows = {
      {1, {0.04f, 0.08f, 0.44f, 0.82f}, 3, 10, true},
      {2, {0.52f, 0.08f, 0.44f, 0.82f}, 4, 11, true},
  };
  return ChromaspaceViewer::buildViewerFramePlan(request);
}

ViewerUiSceneInput makeInput(const ViewerFramePlan& plan) {
  ViewerUiSceneInput input{};
  input.toolbar.visible = true;
  input.toolbar.layoutLabel = "Split 2";
  input.toolbar.layoutIndex = 1;
  input.focusedWindowId = plan.windows.empty() ? -1 : plan.windows.front().windowId;
  input.windows.reserve(plan.windows.size());
  for (const auto& window : plan.windows) {
    ViewerUiPlotWindowInput requested{};
    requested.windowId = window.windowId;
    requested.title = window.windowId == 2 ? "Top Plot" : "Bottom Plot";
    requested.metadata = "source metadata";
    requested.titleMetrics.titleExtraHeight = 3.0f;
    requested.titleMetrics.fontAscent = 14.0f;
    requested.titleMetrics.fontDescent = 4.0f;
    requested.titleMetrics.textScale = 1.0f;
    requested.titleMetrics.measuredMetadataWidth = 100.0f;
    requested.titleMetrics.fontAvailable = true;
    requested.closable = true;
    input.windows.push_back(requested);
  }
  return input;
}

void assertSceneWithinViewport(const ViewerUiScene& scene) {
  const ChromaspaceViewer::ScreenRect viewport{
      0.0f,
      0.0f,
      static_cast<float>(scene.geometry.windowWidth),
      static_cast<float>(scene.geometry.windowHeight)};
  for (const auto& primitive : scene.primitives) {
    assert(contains(viewport, primitive.rect));
    if (primitive.windowId > 0) {
      bool owned = false;
      for (const auto& window : scene.windows) {
        if (window.windowId == primitive.windowId) {
          assert(contains(window.rect, primitive.rect));
          owned = true;
          break;
        }
      }
      assert(owned);
    }
  }
  for (const auto& hit : scene.hits) {
    assert(contains(viewport, hit.rect));
    if (hit.windowId > 0) {
      bool owned = false;
      for (const auto& window : scene.windows) {
        if (window.windowId == hit.windowId) {
          assert(contains(window.rect, hit.rect));
          owned = true;
          break;
        }
      }
      assert(owned);
    }
  }
  for (const auto& text : scene.texts) {
    assert(contains(viewport, text.bounds));
    assert(text.maxWidth <= text.bounds.x1 - text.bounds.x0 + 1e-4f);
    if (text.windowId > 0) {
      bool owned = false;
      for (const auto& window : scene.windows) {
        if (window.windowId == text.windowId) {
          assert(contains(window.rect, text.bounds));
          owned = true;
          break;
        }
      }
      assert(owned);
    }
  }
  assert(scene.vectors.size() % 3u == 0u);
  for (const auto& vertex : scene.vectors) {
    assert(std::isfinite(vertex.x) && std::isfinite(vertex.y));
    assert(vertex.x >= viewport.x0 && vertex.x <= viewport.x1);
    assert(vertex.y >= viewport.y0 && vertex.y <= viewport.y1);
    bool owned = false;
    for (const auto& window : scene.windows) {
      if (window.windowId == vertex.windowId) {
        assert(vertex.x >= window.rect.x0 && vertex.x <= window.rect.x1);
        assert(vertex.y >= window.rect.y0 && vertex.y <= window.rect.y1);
        owned = true;
        break;
      }
    }
    assert(owned);
  }
}

const ViewerUiHitRegion* findHit(const ViewerUiScene& scene,
                                 int windowId,
                                 ViewerUiControlKind control,
                                 int controlIndex = -1) {
  for (const auto& hit : scene.hits) {
    if (hit.windowId == windowId && hit.control == control &&
        hit.controlIndex == controlIndex) {
      return &hit;
    }
  }
  return nullptr;
}

const ChromaspaceViewer::ViewerUiSolidRect* findPrimitive(
    const ViewerUiScene& scene,
    int windowId,
    ViewerUiControlKind control,
    int controlIndex = -1) {
  for (const auto& primitive : scene.primitives) {
    if (primitive.windowId == windowId && primitive.control == control &&
        primitive.controlIndex == controlIndex) {
      return &primitive;
    }
  }
  return nullptr;
}

std::pair<float, float> center(const ViewerUiHitRegion& hit) {
  return {(hit.rect.x0 + hit.rect.x1) * 0.5f,
          (hit.rect.y0 + hit.rect.y1) * 0.5f};
}

void testInvalidPlanAndExactJoin() {
  ViewerFramePlanRequest invalid{};
  invalid.windowWidth = 640;
  invalid.windowHeight = 360;
  invalid.windows = {
      {7, {0.0f, 0.0f, 0.5f, 0.5f}, 1, 1, true},
      {7, {0.5f, 0.0f, 0.5f, 0.5f}, 2, 2, true},
  };
  const ViewerFramePlan invalidPlan =
      ChromaspaceViewer::buildViewerFramePlan(invalid);
  const ViewerUiScene invalidScene =
      ChromaspaceViewer::buildViewerUiScene(invalidPlan, {});
  assert(invalidScene.status == ViewerUiSceneStatus::InvalidFramePlan);
  assert(invalidScene.primitives.empty() && invalidScene.hits.empty());

  const ViewerFramePlan plan = makePlan();
  ViewerUiSceneInput mismatch = makeInput(plan);
  mismatch.windows.pop_back();
  const ViewerUiScene countScene =
      ChromaspaceViewer::buildViewerUiScene(plan, mismatch);
  assert(countScene.status == ViewerUiSceneStatus::WindowCountMismatch);
  assert(countScene.primitives.empty() && countScene.hits.empty());

  mismatch = makeInput(plan);
  mismatch.windows.front().windowId = 99;
  const ViewerUiScene idScene =
      ChromaspaceViewer::buildViewerUiScene(plan, mismatch);
  assert(idScene.status == ViewerUiSceneStatus::WindowIdMismatch);
  assert(idScene.primitives.empty() && idScene.hits.empty());
}

void testToolbarCoverageAndPrecedence() {
  const ViewerFramePlan plan = makePlan(false, false, true);
  ViewerUiSceneInput input = makeInput(plan);
  const ViewerUiScene scene = ChromaspaceViewer::buildViewerUiScene(plan, input);
  assert(scene.ready());
  assert(scene.toolbar.visible);
  const ViewerUiHitResult menu = ChromaspaceViewer::viewerUiHitTest(scene, 10.0f, 10.0f);
  assert(menu.control == ViewerUiControlKind::ToolbarMenu);
  assert(menu.windowId < 0);
  const ViewerUiHitResult add = ChromaspaceViewer::viewerUiHitTest(scene, 47.0f, 40.0f);
  assert(add.control == ViewerUiControlKind::ToolbarAddPlot);
  bool sawGlyph = false;
  for (const auto& primitive : scene.primitives) {
    if (primitive.control == ViewerUiControlKind::ToolbarLayoutPreset &&
        primitive.kind == ViewerUiPrimitiveKind::Glyph) {
      sawGlyph = true;
      assert(primitive.windowId < 0);
    }
  }
  assert(sawGlyph);
}

void testTitleAndCloseGeometry() {
  const ViewerFramePlan plan = makePlan();
  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(plan, makeInput(plan));
  assert(scene.ready());
  assert(scene.windows.size() == 2u);
  const auto& top = scene.windows.back();
  assert(top.windowId == 2);
  assert(near(top.rect.x0, 128.0f));
  assert(near(top.rect.y0, 72.0f));
  assert(near(top.rect.x1, 448.0f));
  assert(near(top.rect.y1, 252.0f));
  std::size_t closeCount = 0;
  for (std::size_t i = top.hitBegin; i < top.hitBegin + top.hitCount; ++i) {
    if (scene.hits[i].control == ViewerUiControlKind::PlotClose) {
      ++closeCount;
      assert(near(scene.hits[i].rect.x0, 425.0f));
      assert(near(scene.hits[i].rect.y0, 76.0f));
      assert(near(scene.hits[i].rect.x1, 443.0f));
      assert(near(scene.hits[i].rect.y1, 94.0f));
    }
  }
  assert(closeCount == 1u);

  const ViewerFramePlan onePlan = makePlan(false, false);
  const ViewerUiScene oneScene =
      ChromaspaceViewer::buildViewerUiScene(onePlan, makeInput(onePlan));
  for (const auto& hit : oneScene.hits) {
    assert(hit.control != ViewerUiControlKind::PlotClose);
  }
}

void testTitleHeightPolicy() {
  const float tiny =
      ChromaspaceViewer::viewerUiTitleBarLogicalHeight(1.0f, 3.0f);
  const float compact =
      ChromaspaceViewer::viewerUiTitleBarLogicalHeight(20.0f, 3.0f);
  const float belowBoundary =
      ChromaspaceViewer::viewerUiTitleBarLogicalHeight(87.0f, 3.0f);
  const float atBoundary =
      ChromaspaceViewer::viewerUiTitleBarLogicalHeight(88.0f, 3.0f);
  const float invalid = ChromaspaceViewer::viewerUiTitleBarLogicalHeight(
      std::numeric_limits<float>::quiet_NaN(), 3.0f);
  assert(std::isfinite(tiny) && tiny > 0.0f && tiny <= 1.0f);
  assert(std::isfinite(compact) && compact > 0.0f && compact <= 20.0f);
  assert(std::isfinite(belowBoundary) && belowBoundary > 0.0f &&
         belowBoundary <= 87.0f);
  assert(std::isfinite(atBoundary) && atBoundary > 0.0f &&
         atBoundary <= 88.0f);
  assert(std::isfinite(invalid) && near(invalid, tiny));
}

void testPortablePlotControlSemantics() {
  const ViewerFramePlan plan = makeDisjointPlan();
  ViewerUiSceneInput input = makeInput(plan);
  input.toolbar.visible = false;
  auto& slicing = input.windows[0].slicing;
  slicing.visible = true;
  slicing.drawerOpen = true;
  slicing.active = true;
  slicing.animationProgress = 1.0f;
  slicing.vectors = {true, false, true, false, true, false};
  slicing.lassoActive = true;
  input.windows[0].sourceSignalRestoreWindowIds = {2};
  auto& source = input.windows[1].sourceLasso;
  source.visible = true;
  source.subtract = true;
  source.hasSelection = false;

  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  assert(scene.ready());
  assertSceneWithinViewport(scene);

  const ViewerUiHitRegion* toggle = findHit(
      scene, 1, ViewerUiControlKind::SlicingQuickToggle);
  assert(toggle && toggle->enabled && toggle->actionable && toggle->selected);
  for (int i = 0; i < 6; ++i) {
    const ViewerUiHitRegion* vector = findHit(
        scene, 1, ViewerUiControlKind::SlicingVector, i);
    assert(vector && vector->enabled && vector->actionable);
    assert(vector->selected == (i % 2 == 0));
    const auto point = center(*vector);
    const ViewerUiHitResult hit = ChromaspaceViewer::viewerUiHitTest(
        scene, point.first, point.second);
    assert(hit.control == ViewerUiControlKind::SlicingVector);
    assert(hit.windowId == 1 && hit.controlIndex == i);
    assert(hit.enabled && hit.actionable);
    assert(hit.selected == vector->selected);
  }
  const ViewerUiHitRegion* lasso = findHit(
      scene, 1, ViewerUiControlKind::SlicingLasso);
  assert(lasso && lasso->selected);
  const ViewerUiHitRegion* restore = findHit(
      scene, 1, ViewerUiControlKind::SourceSignalRestore, 2);
  assert(restore && restore->enabled && restore->actionable);
  const auto restorePoint = center(*restore);
  const ViewerUiHitResult restoreHit = ChromaspaceViewer::viewerUiHitTest(
      scene, restorePoint.first, restorePoint.second);
  assert(restoreHit.control == ViewerUiControlKind::SourceSignalRestore &&
         restoreHit.windowId == 1 && restoreHit.controlIndex == 2);

  const ViewerUiHitRegion* add = findHit(
      scene, 2, ViewerUiControlKind::SourceLassoAdd);
  const ViewerUiHitRegion* subtract = findHit(
      scene, 2, ViewerUiControlKind::SourceLassoSubtract);
  const ViewerUiHitRegion* clear = findHit(
      scene, 2, ViewerUiControlKind::SourceLassoClear);
  assert(add && add->enabled && add->actionable && !add->selected);
  assert(subtract && subtract->enabled && subtract->actionable &&
         subtract->selected);
  assert(clear && !clear->enabled && !clear->actionable && !clear->selected);
  const auto addPoint = center(*add);
  const ViewerUiHitResult addHit = ChromaspaceViewer::viewerUiHitTest(
      scene, addPoint.first, addPoint.second);
  assert(addHit.control == ViewerUiControlKind::SourceLassoAdd);
  assert(addHit.windowId == 2 && addHit.enabled && addHit.actionable);
  const auto clearPoint = center(*clear);
  const ViewerUiHitResult clearHit = ChromaspaceViewer::viewerUiHitTest(
      scene, clearPoint.first, clearPoint.second);
  assert(clearHit.control != ViewerUiControlKind::SourceLassoClear);
  const auto* clearPrimitive = findPrimitive(
      scene, 2, ViewerUiControlKind::SourceLassoClear);
  assert(clearPrimitive && !clearPrimitive->enabled);
  bool sawDisabledClearText = false;
  for (const auto& text : scene.texts) {
    if (text.windowId == 2 &&
        text.control == ViewerUiControlKind::SourceLassoClear) {
      sawDisabledClearText = true;
      assert(!text.enabled);
    }
  }
  assert(sawDisabledClearText);

  ViewerUiSceneInput selectedInput = input;
  selectedInput.windows[1].sourceLasso.hasSelection = true;
  const ViewerUiScene selectedScene =
      ChromaspaceViewer::buildViewerUiScene(plan, selectedInput);
  const ViewerUiHitRegion* selectedClear = findHit(
      selectedScene, 2, ViewerUiControlKind::SourceLassoClear);
  assert(selectedClear && selectedClear->enabled &&
         selectedClear->actionable && selectedClear->selected);
  const auto selectedClearPoint = center(*selectedClear);
  const ViewerUiHitResult selectedClearHit =
      ChromaspaceViewer::viewerUiHitTest(
          selectedScene, selectedClearPoint.first, selectedClearPoint.second);
  assert(selectedClearHit.control == ViewerUiControlKind::SourceLassoClear);
  assert(selectedClearHit.windowId == 2 && selectedClearHit.enabled &&
         selectedClearHit.actionable && selectedClearHit.selected);

  bool sawSlicingVectors = false;
  for (const auto& vertex : scene.vectors) {
    if (vertex.windowId == 1 &&
        (vertex.control == ViewerUiControlKind::SlicingQuickToggle ||
         vertex.control == ViewerUiControlKind::SlicingLasso)) {
      sawSlicingVectors = true;
    }
  }
  assert(sawSlicingVectors);
}

void testPlotControlPrecedence() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 360;
  request.framebufferWidth = 640;
  request.framebufferHeight = 360;
  request.windows = {
      {1, {0.10f, 0.10f, 0.60f, 0.70f}, 3, 10, true},
      {2, {0.10f, 0.10f, 0.60f, 0.70f}, 4, 11, true},
  };
  const ViewerFramePlan plan = ChromaspaceViewer::buildViewerFramePlan(request);
  ViewerUiSceneInput input = makeInput(plan);
  input.toolbar.visible = false;
  for (auto& window : input.windows) {
    window.sourceLasso.visible = true;
    window.sourceLasso.hasSelection = true;
  }
  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  assert(scene.ready());
  const ViewerUiHitRegion* topAdd = findHit(
      scene, 2, ViewerUiControlKind::SourceLassoAdd);
  assert(topAdd);
  const auto controlPoint = center(*topAdd);
  const ViewerUiHitResult controlHit = ChromaspaceViewer::viewerUiHitTest(
      scene, controlPoint.first, controlPoint.second);
  assert(controlHit.control == ViewerUiControlKind::SourceLassoAdd);
  assert(controlHit.windowId == 2);

  const ViewerUiHitRegion* topClose = findHit(
      scene, 2, ViewerUiControlKind::PlotClose);
  assert(topClose);
  ViewerUiScene forcedOverlap = scene;
  for (auto& hit : forcedOverlap.hits) {
    if (hit.windowId == 2 &&
        hit.control == ViewerUiControlKind::SourceLassoAdd) {
      hit.rect = topClose->rect;
    }
  }
  const auto closePoint = center(*topClose);
  const ViewerUiHitResult closeHit = ChromaspaceViewer::viewerUiHitTest(
      forcedOverlap, closePoint.first, closePoint.second);
  assert(closeHit.control == ViewerUiControlKind::PlotClose);
  assert(closeHit.windowId == 2);
}

void testAnimatedControlGeometry() {
  const ViewerFramePlan plan = makePlan(false, false);
  ViewerUiSceneInput startInput = makeInput(plan);
  startInput.toolbar.visible = false;
  startInput.windows[0].slicing.visible = true;
  startInput.windows[0].slicing.drawerOpen = true;
  startInput.windows[0].slicing.animationProgress = 0.0f;
  const ViewerUiScene start =
      ChromaspaceViewer::buildViewerUiScene(plan, startInput);
  const ViewerUiHitRegion* startVector = findHit(
      start, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(startVector && !startVector->actionable);
  const auto startPoint = center(*startVector);
  const ViewerUiHitResult startHit = ChromaspaceViewer::viewerUiHitTest(
      start, startPoint.first, startPoint.second);
  assert(startHit.control == ViewerUiControlKind::PlotBody);

  ViewerUiSceneInput midInput = startInput;
  midInput.windows[0].slicing.animationProgress = 0.5f;
  const ViewerUiScene mid =
      ChromaspaceViewer::buildViewerUiScene(plan, midInput);
  const ViewerUiHitRegion* midVector = findHit(
      mid, 1, ViewerUiControlKind::SlicingVector, 0);
  const auto* midPrimitive = findPrimitive(
      mid, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(midVector && midPrimitive && midVector->actionable);
  assert(near(midVector->rect.x0, midPrimitive->rect.x0));
  assert(near(midVector->rect.y0, midPrimitive->rect.y0));
  assert(near(midVector->rect.x1, midPrimitive->rect.x1));
  assert(near(midVector->rect.y1, midPrimitive->rect.y1));

  ViewerUiSceneInput endInput = startInput;
  endInput.windows[0].slicing.animationProgress = 1.0f;
  const ViewerUiScene end =
      ChromaspaceViewer::buildViewerUiScene(plan, endInput);
  const ViewerUiHitRegion* endVector = findHit(
      end, 1, ViewerUiControlKind::SlicingVector, 0);
  assert(endVector && endVector->actionable);
  assert(endVector->rect.x1 - endVector->rect.x0 >
         midVector->rect.x1 - midVector->rect.x0);
  const auto endPoint = center(*endVector);
  const ViewerUiHitResult endHit = ChromaspaceViewer::viewerUiHitTest(
      end, endPoint.first, endPoint.second);
  assert(endHit.control == ViewerUiControlKind::SlicingVector);
  assert(endHit.controlIndex == 0);
}

void testClosePrecedenceAndZOrder() {
  const ViewerFramePlan plan = makePlan();
  ViewerUiSceneInput input = makeInput(plan);
  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  const ViewerUiHitResult close = ChromaspaceViewer::viewerUiHitTest(scene, 430.0f, 80.0f);
  assert(close.control == ViewerUiControlKind::PlotClose);
  assert(close.windowId == 2);

  const ViewerUiHitResult overlap = ChromaspaceViewer::viewerUiHitTest(scene, 200.0f, 150.0f);
  assert(overlap.control == ViewerUiControlKind::PlotBody);
  assert(overlap.windowId == 2);
}

void testDragSemantics() {
  const ViewerFramePlan plan = makePlan(false, false);
  ViewerUiSceneInput input = makeInput(plan);
  input.toolbar.visible = false;
  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  const ViewerUiHitResult corner = ChromaspaceViewer::viewerUiHitTest(scene, 64.0f, 36.0f);
  assert(corner.control == ViewerUiControlKind::PlotBody);
  assert(corner.dragMode == PlotWindowDragMode::ResizeTopLeft);
  const ViewerUiHitResult edge = ChromaspaceViewer::viewerUiHitTest(scene, 64.0f, 100.0f);
  assert(edge.dragMode == PlotWindowDragMode::ResizeLeft);
  const ViewerUiHitResult title = ChromaspaceViewer::viewerUiHitTest(scene, 100.0f, 50.0f);
  assert(title.dragMode == PlotWindowDragMode::Move);
  const ViewerUiHitResult body = ChromaspaceViewer::viewerUiHitTest(scene, 200.0f, 150.0f);
  assert(body.control == ViewerUiControlKind::PlotBody);
  assert(body.dragMode == PlotWindowDragMode::None);
}

void testVisualDeterminismAndText() {
  const ViewerFramePlan plan = makePlan();
  ViewerUiSceneInput input = makeInput(plan);
  input.hoveredWindowId = 2;
  input.hoveredDragMode = PlotWindowDragMode::Move;
  input.activeDragWindowId = 2;
  input.activeDragMode = PlotWindowDragMode::Move;
  input.windows[0].slicing.visible = true;
  input.windows[0].slicing.drawerOpen = true;
  input.windows[0].slicing.animationProgress = 1.0f;
  input.windows[0].slicing.lassoActive = true;
  input.hasPointer = true;
  input.pointerX = 300.0f;
  input.pointerY = 90.0f;
  const ViewerUiScene first =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  const ViewerUiScene repeat =
      ChromaspaceViewer::buildViewerUiScene(plan, input);
  assert(first.primitives.size() == repeat.primitives.size());
  assert(first.texts.size() == repeat.texts.size());
  assert(first.vectors.size() == repeat.vectors.size());
  for (std::size_t i = 0; i < first.primitives.size(); ++i) {
    assert(near(first.primitives[i].rect.x0, repeat.primitives[i].rect.x0));
    assert(near(first.primitives[i].rect.y1, repeat.primitives[i].rect.y1));
    assert(near(first.primitives[i].color.a, repeat.primitives[i].color.a));
  }
  for (std::size_t i = 0; i < first.vectors.size(); ++i) {
    assert(near(first.vectors[i].x, repeat.vectors[i].x));
    assert(near(first.vectors[i].y, repeat.vectors[i].y));
    assert(near(first.vectors[i].color.a, repeat.vectors[i].color.a));
    assert(first.vectors[i].control == repeat.vectors[i].control);
    assert(first.vectors[i].controlIndex == repeat.vectors[i].controlIndex);
  }
  bool sawMetadata = false;
  for (const auto& text : first.texts) {
    assert(text.maxWidth >= 0.0f);
    assert(text.bounds.x1 >= text.bounds.x0);
    assert(text.originY >= text.bounds.y0);
    assert(text.originY <= text.bounds.y1 + 20.0f);
    if (text.alignment == ViewerUiTextAlignment::Right) sawMetadata = true;
  }
  assert(sawMetadata);
}

void testSmallViewportClippingAndRetinaIndependence() {
  ViewerFramePlanRequest smallRequest{};
  smallRequest.windowWidth = 80;
  smallRequest.windowHeight = 60;
  smallRequest.framebufferWidth = 160;
  smallRequest.framebufferHeight = 120;
  smallRequest.windows = {{1, {-0.25f, -0.25f, 1.5f, 1.5f}, 1, 1, true}};
  const ViewerFramePlan smallPlan =
      ChromaspaceViewer::buildViewerFramePlan(smallRequest);
  ViewerUiSceneInput smallInput = makeInput(smallPlan);
  smallInput.windows[0].slicing.visible = true;
  smallInput.windows[0].slicing.drawerOpen = true;
  smallInput.windows[0].slicing.animationProgress = 1.0f;
  smallInput.windows[0].sourceLasso.visible = true;
  const ViewerUiScene small =
      ChromaspaceViewer::buildViewerUiScene(smallPlan, smallInput);
  assert(small.ready());
  assertSceneWithinViewport(small);
  assert(small.windows.size() == 1u);
  const float smallTitleHeight =
      ChromaspaceViewer::viewerUiTitleBarLogicalHeight(
          small.windows.front().rect.y1 - small.windows.front().rect.y0,
          3.0f);
  const float smallTitleMaxScale =
      std::max(0.66f, smallTitleHeight / 20.0f);
  for (const auto& text : small.texts) {
    if (text.windowId > 0 &&
        text.control == ViewerUiControlKind::PlotBody) {
      assert(text.scale <= std::min(1.18f, smallTitleMaxScale) + 1e-4f);
      assert(text.scale >= 0.66f - 1e-4f);
    }
  }

  const ViewerFramePlan oneX = makePlan(false);
  const ViewerFramePlan twoX = makePlan(true);
  ViewerUiSceneInput oneInput = makeInput(oneX);
  ViewerUiSceneInput twoInput = makeInput(twoX);
  oneInput.windows[0].slicing.visible = true;
  oneInput.windows[0].slicing.drawerOpen = true;
  oneInput.windows[0].slicing.animationProgress = 1.0f;
  twoInput.windows[0].slicing = oneInput.windows[0].slicing;
  const ViewerUiScene oneScene =
      ChromaspaceViewer::buildViewerUiScene(oneX, oneInput);
  const ViewerUiScene twoScene =
      ChromaspaceViewer::buildViewerUiScene(twoX, twoInput);
  assert(twoScene.geometry.scaleX == 2.0f && twoScene.geometry.scaleY == 2.0f);
  assert(oneScene.primitives.size() == twoScene.primitives.size());
  assert(oneScene.texts.size() == twoScene.texts.size());
  for (std::size_t i = 0; i < oneScene.primitives.size(); ++i) {
    assert(near(oneScene.primitives[i].rect.x0,
                twoScene.primitives[i].rect.x0));
    assert(near(oneScene.primitives[i].rect.y0,
                twoScene.primitives[i].rect.y0));
    assert(near(oneScene.primitives[i].rect.x1,
                twoScene.primitives[i].rect.x1));
    assert(near(oneScene.primitives[i].rect.y1,
                twoScene.primitives[i].rect.y1));
  }
  assert(oneScene.vectors.size() == twoScene.vectors.size());
  for (std::size_t i = 0; i < oneScene.vectors.size(); ++i) {
    assert(near(oneScene.vectors[i].x, twoScene.vectors[i].x));
    assert(near(oneScene.vectors[i].y, twoScene.vectors[i].y));
  }
}

void testInvalidValuesFailClosed() {
  const ViewerFramePlan plan = makePlan(false, false);
  ViewerFramePlan invalidValues = plan;
  invalidValues.windows.front().normalizedRect.w =
      std::numeric_limits<float>::quiet_NaN();
  const ViewerUiScene scene =
      ChromaspaceViewer::buildViewerUiScene(invalidValues, makeInput(plan));
  assert(scene.status == ViewerUiSceneStatus::InvalidWindowInput);
  assert(scene.primitives.empty() && scene.hits.empty());

  ViewerUiSceneInput invalidProgress = makeInput(plan);
  invalidProgress.windows.front().slicing.visible = true;
  invalidProgress.windows.front().slicing.animationProgress =
      std::numeric_limits<float>::quiet_NaN();
  const ViewerUiScene progressScene =
      ChromaspaceViewer::buildViewerUiScene(plan, invalidProgress);
  assert(progressScene.status == ViewerUiSceneStatus::InvalidWindowInput);
  assert(progressScene.primitives.empty() && progressScene.hits.empty() &&
         progressScene.vectors.empty());

  WorkspaceToolbarInput toolbar{};
  toolbar.logicalWidth = 120;
  toolbar.logicalHeight = 40;
  toolbar.visible = true;
  toolbar.layoutLabel = "Split 2";
  const auto toolbarScene =
      ChromaspaceViewer::buildWorkspaceToolbarScene(toolbar);
  assert(ChromaspaceViewer::workspaceToolbarHitTest(
             toolbarScene, 20.0f, 20.0f) ==
         ViewerUiControlKind::ToolbarMenu);
  assert(ChromaspaceViewer::workspaceToolbarHitTest(
             toolbarScene, 110.0f, 20.0f) == ViewerUiControlKind::None);
}

}  // namespace

int main() {
  testInvalidPlanAndExactJoin();
  testToolbarCoverageAndPrecedence();
  testTitleAndCloseGeometry();
  testTitleHeightPolicy();
  testPortablePlotControlSemantics();
  testPlotControlPrecedence();
  testAnimatedControlGeometry();
  testClosePrecedenceAndZOrder();
  testDragSemantics();
  testVisualDeterminismAndText();
  testSmallViewportClippingAndRetinaIndependence();
  testInvalidValuesFailClosed();
  return 0;
}
