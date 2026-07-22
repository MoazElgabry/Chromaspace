#include "../../../src/ChromaspaceViewerFramePlan.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace {

using ChromaspaceViewer::PlotWindowRectNorm;
using ChromaspaceViewer::ViewerFramePlan;
using ChromaspaceViewer::ViewerFramePlanRequest;
using ChromaspaceViewer::ViewerFramePlanStatus;
using ChromaspaceViewer::ViewerFramePlanWindowInput;

bool near(float a, float b, float epsilon = 1e-5f) {
  return std::fabs(a - b) <= epsilon;
}

void dimensionsAreSanitized() {
  ViewerFramePlanRequest request{};
  request.windowWidth = -100;
  request.windowHeight = 0;
  request.framebufferWidth = -2;
  request.framebufferHeight = 0;
  request.windows.push_back({1, {0.0f, 0.0f, 0.5f, 0.5f}, 3, 7, true});
  const ViewerFramePlan plan = ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(plan.geometry.windowWidth == 1);
  assert(plan.geometry.windowHeight == 1);
  assert(plan.geometry.framebufferWidth == 1);
  assert(plan.geometry.framebufferHeight == 1);
  assert(near(plan.geometry.scaleX, 1.0f));
  assert(near(plan.geometry.scaleY, 1.0f));
  assert(plan.windows.size() == 1u);
  assert(plan.windows.front().renderTargetWidth >= 1);
  assert(plan.windows.front().renderTargetHeight >= 1);
}

void retinaScalingAndFractionalTargets() {
  ViewerFramePlanRequest oneX{};
  oneX.windowWidth = 100;
  oneX.windowHeight = 80;
  oneX.framebufferWidth = 100;
  oneX.framebufferHeight = 80;
  oneX.windows.push_back({4, {0.1f, 0.25f, 0.333f, 0.375f}, 9, 11, true});
  const ViewerFramePlan oneXPlan =
      ChromaspaceViewer::buildViewerFramePlan(oneX);
  assert(oneXPlan.status == ViewerFramePlanStatus::Ready);
  assert(near(oneXPlan.geometry.scaleX, 1.0f));
  assert(near(oneXPlan.geometry.scaleY, 1.0f));
  assert(near(oneXPlan.windows[0].logicalRect.x0, 10.0f));
  assert(near(oneXPlan.windows[0].logicalRect.y0, 20.0f));
  assert(near(oneXPlan.windows[0].logicalRect.x1, 43.3f));
  assert(near(oneXPlan.windows[0].logicalRect.y1, 50.0f));
  assert(near(oneXPlan.windows[0].normalizedRect.x, 0.1f));
  assert(near(oneXPlan.windows[0].normalizedRect.y, 0.25f));
  assert(near(oneXPlan.windows[0].normalizedRect.w, 0.333f));
  assert(near(oneXPlan.windows[0].normalizedRect.h, 0.375f));
  assert(oneXPlan.windows[0].renderTargetWidth == 34);
  assert(oneXPlan.windows[0].renderTargetHeight == 30);

  ViewerFramePlanRequest twoX = oneX;
  twoX.framebufferWidth = 200;
  twoX.framebufferHeight = 160;
  const ViewerFramePlan twoXPlan =
      ChromaspaceViewer::buildViewerFramePlan(twoX);
  assert(twoXPlan.valid());
  assert(near(twoXPlan.geometry.scaleX, 2.0f));
  assert(near(twoXPlan.geometry.scaleY, 2.0f));
  assert(near(twoXPlan.windows[0].framebufferRect.x0, 20.0f));
  assert(near(twoXPlan.windows[0].framebufferRect.y0, 40.0f));
  assert(twoXPlan.windows[0].renderTargetWidth == 67);
  assert(twoXPlan.windows[0].renderTargetHeight == 60);

  ViewerFramePlanRequest asymmetric = oneX;
  asymmetric.framebufferWidth = 150;
  asymmetric.framebufferHeight = 200;
  const ViewerFramePlan asymmetricPlan =
      ChromaspaceViewer::buildViewerFramePlan(asymmetric);
  assert(asymmetricPlan.ready());
  assert(near(asymmetricPlan.geometry.scaleX, 1.5f));
  assert(near(asymmetricPlan.geometry.scaleY, 2.5f));
  assert(near(asymmetricPlan.windows[0].framebufferRect.x0, 15.0f));
  assert(near(asymmetricPlan.windows[0].framebufferRect.y0, 50.0f));
  assert(asymmetricPlan.windows[0].renderTargetWidth == 50);
  assert(asymmetricPlan.windows[0].renderTargetHeight == 75);
}

void reservedLeftGeometry() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 100;
  request.windowHeight = 50;
  request.framebufferWidth = 200;
  request.framebufferHeight = 100;
  request.reservedLeftPixels = 10.0f;
  request.windows.push_back({1, {0.1f, 0.2f, 0.4f, 0.5f}, 2, 3, true});
  const ViewerFramePlan plan =
      ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(near(plan.geometry.reservedLeftPixels, 10.0f));
  assert(near(plan.windows[0].logicalRect.x0, 19.0f));
  assert(near(plan.windows[0].logicalRect.y0, 10.0f));
  assert(near(plan.windows[0].logicalRect.x1, 55.0f));
  assert(near(plan.windows[0].logicalRect.y1, 35.0f));
  assert(near(plan.windows[0].framebufferRect.x0, 38.0f));
  assert(near(plan.windows[0].framebufferRect.y0, 20.0f));
  assert(near(plan.windows[0].framebufferRect.x1, 110.0f));
  assert(near(plan.windows[0].framebufferRect.y1, 70.0f));
  assert(plan.windows[0].renderTargetWidth == 72);
  assert(plan.windows[0].renderTargetHeight == 50);

  request.reservedLeftPixels = 0.0f;
  const ViewerFramePlan noReservation =
      ChromaspaceViewer::buildViewerFramePlan(request);
  assert(noReservation.ready());
  assert(near(noReservation.geometry.reservedLeftPixels, 0.0f));
  assert(near(noReservation.windows[0].logicalRect.x0, 10.0f));
  assert(near(noReservation.windows[0].logicalRect.x1, 50.0f));
}

void orderVisibilityAndPassthrough() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 640;
  request.windowHeight = 480;
  request.framebufferWidth = 640;
  request.framebufferHeight = 480;
  request.windows = {
      {42, {0.0f, 0.0f, 0.25f, 0.25f}, 17, 1001, true},
      {99, {0.25f, 0.0f, 0.25f, 0.25f}, 23, 1002, false},
      {7, {0.5f, 0.0f, 0.25f, 0.25f}, 31, 1003, true},
  };
  const ViewerFramePlan plan =
      ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(plan.windows.size() == 2u);
  assert(plan.windows[0].windowId == 42);
  assert(plan.windows[0].plotModel == 17);
  assert(plan.windows[0].viewRevision == 1001);
  assert(plan.windows[1].windowId == 7);
  assert(plan.windows[1].plotModel == 31);
  assert(plan.windows[1].viewRevision == 1003);
}

void reservedLeftAlwaysLeavesWorkspacePixel() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 10;
  request.windowHeight = 10;
  request.framebufferWidth = 20;
  request.framebufferHeight = 20;
  request.reservedLeftPixels = std::numeric_limits<float>::infinity();
  request.windows.push_back({1, {0.0f, 0.0f, 1.0f, 1.0f}, 0, 1, true});
  ViewerFramePlan plan = ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(near(plan.geometry.reservedLeftPixels, 0.0f));
  assert(near(plan.windows[0].logicalRect.x0, 0.0f));
  assert(near(plan.windows[0].logicalRect.x1, 10.0f));

  request.reservedLeftPixels = 1000000.0f;
  plan = ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(near(plan.geometry.reservedLeftPixels, 9.0f));
  assert(near(plan.windows[0].logicalRect.x0, 9.0f));
  assert(near(plan.windows[0].logicalRect.x1, 10.0f));

  request.reservedLeftPixels = -1000000.0f;
  plan = ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  assert(near(plan.geometry.reservedLeftPixels, 0.0f));
}

void emittedDimensionsStayFiniteAndPositive() {
  ViewerFramePlanRequest request{};
  request.windowWidth = 1920;
  request.windowHeight = 1080;
  request.framebufferWidth = 3840;
  request.framebufferHeight = 2160;
  request.windows.push_back({8, {0.05f, 0.07f, 0.3f, 0.4f}, 12, 44, true});
  const ViewerFramePlan plan =
      ChromaspaceViewer::buildViewerFramePlan(request);
  assert(plan.ready());
  for (const auto& entry : plan.windows) {
    const float values[] = {entry.logicalRect.x0,
                            entry.logicalRect.y0,
                            entry.logicalRect.x1,
                            entry.logicalRect.y1,
                            entry.framebufferRect.x0,
                            entry.framebufferRect.y0,
                            entry.framebufferRect.x1,
                            entry.framebufferRect.y1};
    for (float value : values) assert(std::isfinite(value));
    assert(entry.renderTargetWidth >= 1);
    assert(entry.renderTargetHeight >= 1);
  }
}

void invalidWindowIdsFailClosed() {
  ViewerFramePlanRequest invalidZero{};
  invalidZero.windows = {
      {0, {0.0f, 0.0f, 0.5f, 0.5f}, 1, 2, true},
      {7, {0.5f, 0.0f, 0.5f, 0.5f}, 1, 3, true},
  };
  ViewerFramePlan zeroPlan =
      ChromaspaceViewer::buildViewerFramePlan(invalidZero);
  assert(!zeroPlan.valid());
  assert(zeroPlan.status == ViewerFramePlanStatus::InvalidWindowId);
  assert(zeroPlan.rejectedWindowId == 0);
  assert(zeroPlan.windows.empty());

  invalidZero.windows.front().windowId = -9;
  ViewerFramePlan negativePlan =
      ChromaspaceViewer::buildViewerFramePlan(invalidZero);
  assert(negativePlan.status == ViewerFramePlanStatus::InvalidWindowId);
  assert(negativePlan.rejectedWindowId == -9);
  assert(negativePlan.windows.empty());

  ViewerFramePlanRequest duplicateVisible{};
  duplicateVisible.windows = {
      {11, {0.0f, 0.0f, 0.5f, 0.5f}, 1, 2, true},
      {11, {0.5f, 0.0f, 0.5f, 0.5f}, 2, 3, true},
  };
  ViewerFramePlan visibleDuplicatePlan =
      ChromaspaceViewer::buildViewerFramePlan(duplicateVisible);
  assert(visibleDuplicatePlan.status == ViewerFramePlanStatus::DuplicateWindowId);
  assert(visibleDuplicatePlan.rejectedWindowId == 11);
  assert(visibleDuplicatePlan.windows.empty());

  duplicateVisible.windows[1].visible = false;
  ViewerFramePlan hiddenDuplicatePlan =
      ChromaspaceViewer::buildViewerFramePlan(duplicateVisible);
  assert(hiddenDuplicatePlan.status == ViewerFramePlanStatus::DuplicateWindowId);
  assert(hiddenDuplicatePlan.rejectedWindowId == 11);
  assert(hiddenDuplicatePlan.windows.empty());
}

}  // namespace

int main() {
  dimensionsAreSanitized();
  retinaScalingAndFractionalTargets();
  reservedLeftGeometry();
  orderVisibilityAndPassthrough();
  reservedLeftAlwaysLeavesWorkspacePixel();
  emittedDimensionsStayFiniteAndPositive();
  invalidWindowIdsFailClosed();
  std::cout << "Chromaspace viewer frame-plan tests passed\n";
  return 0;
}
