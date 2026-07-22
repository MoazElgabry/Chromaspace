#pragma once

#include "ChromaspaceViewerDomain.h"

#include <array>

namespace ChromaspaceViewer {

struct ViewerQuaternion {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 1.0f;
};

struct ViewerBounds3 {
  float minX = 0.0f;
  float minY = 0.0f;
  float minZ = 0.0f;
  float maxX = 0.0f;
  float maxY = 0.0f;
  float maxZ = 0.0f;
  bool valid = false;
};

struct ViewerCameraMatricesRequest {
  CameraState camera{};
  ViewerQuaternion modelOrientation{};
  ViewerBounds3 fitBounds{};
  int viewportWidth = 0;
  int viewportHeight = 0;
  float verticalFovDegrees = 28.0f;
  float minOrthoHalfHeight = 0.25f;
};

struct ViewerCameraMatrices {
  std::array<float, 16> modelView{};
  std::array<float, 16> projection{};
  float zNear = 0.0f;
  float zFar = 0.0f;
  bool usedFitBounds = false;
};

bool buildViewerCameraMatrices(const ViewerCameraMatricesRequest& request,
                               ViewerCameraMatrices* output) noexcept;

}  // namespace ChromaspaceViewer
