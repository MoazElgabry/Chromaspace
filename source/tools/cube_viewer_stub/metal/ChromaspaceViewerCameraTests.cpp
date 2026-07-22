#include "../../../src/ChromaspaceViewerCamera.h"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

using ChromaspaceViewer::CameraState;
using ChromaspaceViewer::ViewerBounds3;
using ChromaspaceViewer::ViewerCameraMatrices;
using ChromaspaceViewer::ViewerCameraMatricesRequest;
using ChromaspaceViewer::ViewerQuaternion;

constexpr float kTolerance = 1e-5f;

bool near(float a, float b, float tolerance = kTolerance) {
  return std::fabs(a - b) <= tolerance;
}

void assertFinite(const ViewerCameraMatrices& matrices) {
  for (float value : matrices.modelView) assert(std::isfinite(value));
  for (float value : matrices.projection) assert(std::isfinite(value));
  assert(std::isfinite(matrices.zNear));
  assert(std::isfinite(matrices.zFar));
}

void invalidInputs() {
  ViewerCameraMatrices output{};
  ViewerCameraMatricesRequest request{};
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.viewportWidth = 320;
  request.viewportHeight = 240;
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.camera.distance = -1.0f;
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.camera.distance = 6.0f;
  request.verticalFovDegrees = 180.0f;
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.verticalFovDegrees = 28.0f;
  request.fitBounds.valid = true;
  request.fitBounds.minX = 2.0f;
  request.fitBounds.maxX = 1.0f;
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.fitBounds = ViewerBounds3{};
  request.fitBounds.minY = NAN;
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  request.fitBounds = ViewerBounds3{};
  request.modelOrientation = {NAN, 0.0f, 0.0f, 1.0f};
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  assert(!ChromaspaceViewer::buildViewerCameraMatrices(request, nullptr));
}

void perspectiveIdentityAndTranslation() {
  ViewerCameraMatricesRequest request{};
  request.viewportWidth = 200;
  request.viewportHeight = 100;
  request.camera.distance = 6.0f;
  request.camera.panX = 0.5f;
  request.camera.panY = -0.25f;
  ViewerCameraMatrices output{};
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  assert(near(output.modelView[0], 1.0f));
  assert(near(output.modelView[5], 1.0f));
  assert(near(output.modelView[10], 1.0f));
  assert(near(output.modelView[12], 0.5f));
  assert(near(output.modelView[13], -0.25f));
  assert(near(output.modelView[14], -6.0f));
  assert(near(output.zNear, 0.08f));
  assert(near(output.zFar, 100.0f));
  assert(near(output.projection[0], output.projection[5] * 0.5f));
  assertFinite(output);
}

void orthographicAspectAndScaling() {
  ViewerCameraMatricesRequest request{};
  request.viewportWidth = 400;
  request.viewportHeight = 200;
  request.camera.orthographic = true;
  request.camera.distance = 6.0f;
  ViewerCameraMatrices output{};
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  const float halfHeight = 6.0f * std::tan(14.0f * 3.14159265358979323846f / 180.0f);
  const float halfWidth = halfHeight * 2.0f;
  assert(near(output.projection[0], 1.0f / halfWidth, 2e-5f));
  assert(near(output.projection[5], 1.0f / halfHeight, 2e-5f));
  assert(near(output.projection[15], 1.0f));
  assertFinite(output);
}

void normalizedQuaternionAndTransformOrder() {
  ViewerCameraMatricesRequest request{};
  request.viewportWidth = 100;
  request.viewportHeight = 100;
  request.camera.distance = 4.0f;
  request.camera.qz = 2.0f;
  request.camera.qw = 2.0f;
  request.modelOrientation = {2.0f, 0.0f, 0.0f, 2.0f};
  ViewerCameraMatrices output{};
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  assert(near(output.modelView[0], 0.0f));
  assert(near(output.modelView[1], 1.0f));
  assert(near(output.modelView[4], 0.0f));
  assert(near(output.modelView[5], 0.0f));
  assert(near(output.modelView[6], 1.0f));
  assert(near(output.modelView[8], 1.0f));
  assert(near(output.modelView[9], 0.0f));
  assert(near(output.modelView[10], 0.0f));
  assert(near(output.modelView[14], -4.0f));
  assertFinite(output);

  request.modelOrientation = ViewerQuaternion{};
  request.camera.qx = request.camera.qy = request.camera.qz = 0.0f;
  request.camera.qw = 0.0f;
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  assert(near(output.modelView[0], 1.0f));
  assert(near(output.modelView[5], 1.0f));
  assert(near(output.modelView[10], 1.0f));
}

void rotatedFitBoundsAffectFarDepth() {
  ViewerCameraMatricesRequest request{};
  request.viewportWidth = 320;
  request.viewportHeight = 240;
  request.camera.distance = 6.0f;
  request.modelOrientation = {0.0f, 0.0f,
                              std::sin(22.5f * 3.14159265358979323846f / 180.0f),
                              std::cos(22.5f * 3.14159265358979323846f / 180.0f)};
  request.camera.qx = std::sin(30.0f * 3.14159265358979323846f / 180.0f);
  request.camera.qw = std::cos(30.0f * 3.14159265358979323846f / 180.0f);
  request.fitBounds = {-1.0f, -1.0f, -300.0f, 1.0f, 1.0f, -299.0f, true};
  ViewerCameraMatrices output{};
  assert(ChromaspaceViewer::buildViewerCameraMatrices(request, &output));
  assert(output.usedFitBounds);
  assert(output.zFar >= 100.0f && output.zFar <= 4000.0f);
  assert(output.zFar > 100.0f);
  assertFinite(output);
}

}  // namespace

int main() {
  invalidInputs();
  perspectiveIdentityAndTranslation();
  orthographicAspectAndScaling();
  normalizedQuaternionAndTransformOrder();
  rotatedFitBoundsAffectFarDepth();
  std::cout << "Chromaspace viewer camera tests passed\n";
  return 0;
}
