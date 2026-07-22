#include "ChromaspaceViewerCamera.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ChromaspaceViewer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinNear = 0.0018f;
constexpr float kMaxNear = 0.08f;
constexpr float kDefaultFar = 100.0f;
constexpr float kMaxFar = 4000.0f;

bool finite(float value) noexcept {
  return std::isfinite(value);
}

bool finiteQuaternion(const ViewerQuaternion& q) noexcept {
  return finite(q.x) && finite(q.y) && finite(q.z) && finite(q.w);
}

bool finiteCamera(const CameraState& camera) noexcept {
  return finite(camera.qx) && finite(camera.qy) && finite(camera.qz) &&
         finite(camera.qw) && finite(camera.distance) && finite(camera.panX) &&
         finite(camera.panY) && camera.distance > 0.0f;
}

bool validBounds(const ViewerBounds3& bounds) noexcept {
  return finite(bounds.minX) && finite(bounds.minY) && finite(bounds.minZ) &&
         finite(bounds.maxX) && finite(bounds.maxY) && finite(bounds.maxZ) &&
         bounds.minX <= bounds.maxX && bounds.minY <= bounds.maxY &&
         bounds.minZ <= bounds.maxZ;
}

ViewerQuaternion normalize(ViewerQuaternion q) noexcept {
  const float scale = std::max({std::fabs(q.x),
                                std::fabs(q.y),
                                std::fabs(q.z),
                                std::fabs(q.w)});
  if (!finite(scale) || scale <= 1e-8f) return ViewerQuaternion{};
  const float sx = q.x / scale;
  const float sy = q.y / scale;
  const float sz = q.z / scale;
  const float sw = q.w / scale;
  const float scaledLength = std::sqrt(sx * sx + sy * sy + sz * sz + sw * sw);
  if (!finite(scaledLength) || scaledLength <= 1e-8f) return ViewerQuaternion{};
  const float inverse = (1.0f / scale) / scaledLength;
  return {q.x * inverse, q.y * inverse, q.z * inverse, q.w * inverse};
}

struct Vector3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

Vector3 rotate(Vector3 value, ViewerQuaternion q) noexcept {
  q = normalize(q);
  const Vector3 qv{q.x, q.y, q.z};
  const Vector3 uv{qv.y * value.z - qv.z * value.y,
                   qv.z * value.x - qv.x * value.z,
                   qv.x * value.y - qv.y * value.x};
  const Vector3 uuv{qv.y * uv.z - qv.z * uv.y,
                    qv.z * uv.x - qv.x * uv.z,
                    qv.x * uv.y - qv.y * uv.x};
  return {value.x + 2.0f * (q.w * uv.x + uuv.x),
          value.y + 2.0f * (q.w * uv.y + uuv.y),
          value.z + 2.0f * (q.w * uv.z + uuv.z)};
}

void identity(float* matrix) noexcept {
  std::fill(matrix, matrix + 16, 0.0f);
  matrix[0] = 1.0f;
  matrix[5] = 1.0f;
  matrix[10] = 1.0f;
  matrix[15] = 1.0f;
}

void quaternionMatrix(ViewerQuaternion q, float* matrix) noexcept {
  q = normalize(q);
  const float xx = q.x * q.x;
  const float yy = q.y * q.y;
  const float zz = q.z * q.z;
  const float xy = q.x * q.y;
  const float xz = q.x * q.z;
  const float yz = q.y * q.z;
  const float wx = q.w * q.x;
  const float wy = q.w * q.y;
  const float wz = q.w * q.z;
  matrix[0] = 1.0f - 2.0f * (yy + zz);
  matrix[1] = 2.0f * (xy + wz);
  matrix[2] = 2.0f * (xz - wy);
  matrix[3] = 0.0f;
  matrix[4] = 2.0f * (xy - wz);
  matrix[5] = 1.0f - 2.0f * (xx + zz);
  matrix[6] = 2.0f * (yz + wx);
  matrix[7] = 0.0f;
  matrix[8] = 2.0f * (xz + wy);
  matrix[9] = 2.0f * (yz - wx);
  matrix[10] = 1.0f - 2.0f * (xx + yy);
  matrix[11] = 0.0f;
  matrix[12] = 0.0f;
  matrix[13] = 0.0f;
  matrix[14] = 0.0f;
  matrix[15] = 1.0f;
}

void multiply(const float* a, const float* b, float* output) noexcept {
  float result[16] = {};
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      result[row + column * 4] =
          a[row + 0 * 4] * b[0 + column * 4] +
          a[row + 1 * 4] * b[1 + column * 4] +
          a[row + 2 * 4] * b[2 + column * 4] +
          a[row + 3 * 4] * b[3 + column * 4];
    }
  }
  std::copy(result, result + 16, output);
}

void makeFrustum(float left,
                 float right,
                 float bottom,
                 float top,
                 float zNear,
                 float zFar,
                 float* output) noexcept {
  std::fill(output, output + 16, 0.0f);
  const float width = std::max(1e-6f, right - left);
  const float height = std::max(1e-6f, top - bottom);
  const float depth = std::max(1e-6f, zFar - zNear);
  output[0] = (2.0f * zNear) / width;
  output[5] = (2.0f * zNear) / height;
  output[8] = (right + left) / width;
  output[9] = (top + bottom) / height;
  output[10] = -(zFar + zNear) / depth;
  output[11] = -1.0f;
  output[14] = -(2.0f * zFar * zNear) / depth;
}

void makeOrtho(float left,
               float right,
               float bottom,
               float top,
               float zNear,
               float zFar,
               float* output) noexcept {
  std::fill(output, output + 16, 0.0f);
  const float width = std::max(1e-6f, right - left);
  const float height = std::max(1e-6f, top - bottom);
  const float depth = std::max(1e-6f, zFar - zNear);
  output[0] = 2.0f / width;
  output[5] = 2.0f / height;
  output[10] = -2.0f / depth;
  output[12] = -(right + left) / width;
  output[13] = -(top + bottom) / height;
  output[14] = -(zFar + zNear) / depth;
  output[15] = 1.0f;
}

bool finiteMatrix(const float* matrix) noexcept {
  for (int i = 0; i < 16; ++i) {
    if (!finite(matrix[i])) return false;
  }
  return true;
}

}  // namespace

bool buildViewerCameraMatrices(const ViewerCameraMatricesRequest& request,
                               ViewerCameraMatrices* output) noexcept {
  if (!output || request.viewportWidth <= 0 || request.viewportHeight <= 0 ||
      !finiteCamera(request.camera) || !finiteQuaternion(request.modelOrientation) ||
      !validBounds(request.fitBounds) || !finite(request.verticalFovDegrees) ||
      !finite(request.minOrthoHalfHeight) || request.verticalFovDegrees <= 0.0f ||
      request.verticalFovDegrees >= 180.0f || request.minOrthoHalfHeight <= 0.0f) {
    return false;
  }

  const float halfFovRadians = request.verticalFovDegrees * 0.5f * kPi / 180.0f;
  const float tanHalfFov = std::tan(halfFovRadians);
  if (!finite(tanHalfFov) || tanHalfFov <= 0.0f) return false;

  const ViewerQuaternion cameraOrientation =
      normalize({request.camera.qx, request.camera.qy, request.camera.qz, request.camera.qw});
  const ViewerQuaternion modelOrientation = normalize(request.modelOrientation);
  const float aspect = static_cast<float>(request.viewportWidth) /
                       static_cast<float>(request.viewportHeight);
  if (!finite(aspect) || aspect <= 0.0f) return false;

  float sceneMinZ = -1.0f;
  bool usedFitBounds = false;
  if (request.fitBounds.valid) {
    sceneMinZ = std::numeric_limits<float>::max();
    const ViewerBounds3& bounds = request.fitBounds;
    for (int corner = 0; corner < 8; ++corner) {
      const Vector3 source{(corner & 1) ? bounds.maxX : bounds.minX,
                           (corner & 2) ? bounds.maxY : bounds.minY,
                           (corner & 4) ? bounds.maxZ : bounds.minZ};
      const Vector3 modelPoint = rotate(source, modelOrientation);
      const Vector3 viewPoint = rotate(modelPoint, cameraOrientation);
      if (!finite(viewPoint.z)) return false;
      sceneMinZ = std::min(sceneMinZ, viewPoint.z);
    }
    usedFitBounds = true;
  }

  const float zNear = std::clamp(request.camera.distance * 0.025f,
                                 kMinNear,
                                 kMaxNear);
  float zFar = kDefaultFar;
  if (usedFitBounds) {
    const float depthToBack = std::max(0.2f,
                                      request.camera.distance - sceneMinZ + 1.5f);
    zFar = std::clamp(depthToBack, kDefaultFar, kMaxFar);
  }
  if (!finite(zNear) || !finite(zFar) || zFar <= zNear) return false;

  const float halfHeight =
      std::max(request.minOrthoHalfHeight, request.camera.distance * tanHalfFov);
  const float halfWidth = halfHeight * aspect;
  if (!finite(halfHeight) || !finite(halfWidth)) return false;

  ViewerCameraMatrices result{};
  result.zNear = zNear;
  result.zFar = zFar;
  result.usedFitBounds = usedFitBounds;
  if (request.camera.orthographic) {
    makeOrtho(-halfWidth,
              halfWidth,
              -halfHeight,
              halfHeight,
              zNear,
              zFar,
              result.projection.data());
  } else {
    const float ymax = zNear * tanHalfFov;
    const float xmax = ymax * aspect;
    if (!finite(ymax) || !finite(xmax)) return false;
    makeFrustum(-xmax,
                xmax,
                -ymax,
                ymax,
                zNear,
                zFar,
                result.projection.data());
  }

  float translation[16] = {};
  float cameraRotation[16] = {};
  float sharedModelRotation[16] = {};
  identity(translation);
  translation[12] = request.camera.panX;
  translation[13] = request.camera.panY;
  translation[14] = -request.camera.distance;
  quaternionMatrix(cameraOrientation, cameraRotation);
  quaternionMatrix(modelOrientation, sharedModelRotation);
  float cameraModelView[16] = {};
  multiply(translation, cameraRotation, cameraModelView);
  multiply(cameraModelView, sharedModelRotation, result.modelView.data());
  if (!finiteMatrix(result.modelView.data()) || !finiteMatrix(result.projection.data())) {
    return false;
  }
  *output = result;
  return true;
}

}  // namespace ChromaspaceViewer
