#include "ChromaspaceViewerCameraInteraction.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ChromaspaceViewer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kMinCameraDistance = 0.008f;
constexpr float kMaxCameraDistance = 1000.0f;
constexpr float kMinOrthoHalfHeight = 0.25f;
constexpr float kVerticalFovDegrees = 28.0f;
constexpr float kPrecisionScale = 0.22f;

struct Vec3 { float x = 0.0f; float y = 0.0f; float z = 0.0f; };
struct Quat { float x = 0.0f; float y = 0.0f; float z = 0.0f; float w = 1.0f; };

bool finite(float value) noexcept { return std::isfinite(value); }
bool finite(double value) noexcept { return std::isfinite(value); }

bool validMode(ViewerCameraInteractionMode mode) noexcept {
  return mode >= ViewerCameraInteractionMode::None &&
         mode <= ViewerCameraInteractionMode::Zoom;
}

bool validState(const ViewerCameraInteractionState& state) noexcept {
  if (!validMode(state.pointerMode) || !finite(state.lastPointerX) ||
      !finite(state.lastPointerY)) return false;
  if (state.pointerCaptureActive) {
    if (state.pointerWindowId <= 0 ||
        state.pointerMode == ViewerCameraInteractionMode::None ||
        state.pointerButton >= ViewerSessionPointerButton::Count) return false;
  } else if (state.pointerWindowId != -1 ||
             state.pointerMode != ViewerCameraInteractionMode::None) {
    return false;
  }
  if (state.gestureCaptureActive) {
    if (state.gestureWindowId <= 0 ||
        state.gestureKind >= ViewerSessionGestureKind::Count) return false;
  } else if (state.gestureWindowId != -1) {
    return false;
  }
  return true;
}

bool validCamera(const CameraState& camera) noexcept {
  const double norm = static_cast<double>(camera.qx) * camera.qx +
                      static_cast<double>(camera.qy) * camera.qy +
                      static_cast<double>(camera.qz) * camera.qz +
                      static_cast<double>(camera.qw) * camera.qw;
  return finite(camera.qx) && finite(camera.qy) && finite(camera.qz) &&
         finite(camera.qw) && finite(camera.distance) && finite(camera.panX) &&
         finite(camera.panY) && camera.distance > 0.0f && finite(norm) &&
         norm > 1.0e-12 && camera.orthographicView >= -1 &&
         camera.orthographicView <= 7;
}

const PlotWindowDomainState* workspaceWindow(const ViewerWorkspaceState& workspace,
                                              int id) noexcept {
  for (const auto& window : workspace.windows) {
    if (window.windowId == id) return &window;
  }
  return nullptr;
}

const ViewerUiWindowScene* sceneWindow(const ViewerUiScene& scene,
                                       int id) noexcept {
  for (const auto& window : scene.windows) {
    if (window.windowId == id) return &window;
  }
  return nullptr;
}

Quat normalized(Quat value) noexcept {
  const double length = std::sqrt(static_cast<double>(value.x) * value.x +
                                  static_cast<double>(value.y) * value.y +
                                  static_cast<double>(value.z) * value.z +
                                  static_cast<double>(value.w) * value.w);
  if (!finite(length) || length <= 1.0e-12) return {};
  const float inverse = static_cast<float>(1.0 / length);
  return {value.x * inverse, value.y * inverse, value.z * inverse,
          value.w * inverse};
}

Quat multiply(Quat left, Quat right) noexcept {
  return {left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
          left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
          left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
          left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z};
}

float length(Vec3 value) noexcept {
  return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vec3 normalize(Vec3 value) noexcept {
  const float size = length(value);
  if (!finite(size) || size <= 1.0e-8f) return {};
  return {value.x / size, value.y / size, value.z / size};
}

Vec3 cross(Vec3 left, Vec3 right) noexcept {
  return {left.y * right.z - left.z * right.y,
          left.z * right.x - left.x * right.z,
          left.x * right.y - left.y * right.x};
}

float dot(Vec3 left, Vec3 right) noexcept {
  return left.x * right.x + left.y * right.y + left.z * right.z;
}

Quat axisAngle(Vec3 axis, float radians) noexcept {
  axis = normalize(axis);
  const float half = radians * 0.5f;
  const float sine = std::sin(half);
  return normalized({axis.x * sine, axis.y * sine, axis.z * sine,
                     std::cos(half)});
}

Vec3 arcball(float x, float y, const ScreenRect& content) noexcept {
  const float width = std::max(1.0f, content.x1 - content.x0);
  const float height = std::max(1.0f, content.y1 - content.y0);
  const float localX = x - content.x0;
  const float localY = y - content.y0;
  const float nx = (2.0f * localX - width) / width;
  const float ny = (height - 2.0f * localY) / height;
  const float distanceSquared = nx * nx + ny * ny;
  if (distanceSquared <= 0.5f) {
    return normalize({nx, ny, std::sqrt(std::max(0.0f, 1.0f - distanceSquared))});
  }
  const float distance = std::sqrt(std::max(distanceSquared, 1.0e-8f));
  return normalize({nx, ny, 0.5f / distance});
}

float minimumDistance(const CameraState& camera) noexcept {
  if (!camera.orthographic) return kMinCameraDistance;
  const float tangent = std::tan(kVerticalFovDegrees * 0.5f * kPi / 180.0f);
  return tangent > 1.0e-6f
             ? std::max(kMinCameraDistance, kMinOrthoHalfHeight / tangent)
             : kMinCameraDistance;
}

float speed(ViewerSessionModifierMask modifiers,
            bool shiftSelectsMode = false) noexcept {
  if ((modifiers & kViewerSessionModifierControl) != 0u) return 2.0f;
  if (!shiftSelectsMode &&
      (modifiers & kViewerSessionModifierShift) != 0u) return kPrecisionScale;
  return 1.0f;
}

bool setRevision(ViewerCameraInteractionState* state) noexcept {
  if (!state || state->interactionRevision ==
                    std::numeric_limits<uint64_t>::max()) return false;
  ++state->interactionRevision;
  return true;
}

void clearPointer(ViewerCameraInteractionState* state) noexcept {
  state->pointerCaptureActive = false;
  state->pointerWindowId = -1;
  state->pointerMode = ViewerCameraInteractionMode::None;
}

void clearGesture(ViewerCameraInteractionState* state) noexcept {
  state->gestureCaptureActive = false;
  state->gestureWindowId = -1;
}

ViewerCameraInteractionResult rejected(ViewerCameraInteractionStatus status,
                                       const ViewerCameraInteractionState& state) noexcept {
  ViewerCameraInteractionResult result{};
  result.status = status;
  result.interactionRevision = state.interactionRevision;
  return result;
}

ViewerCameraInteractionMode modeForPress(
    ViewerSessionPointerButton button,
    ViewerSessionModifierMask modifiers,
    int plotModel) noexcept {
  if (plotModel == kPlotModelSourceSignal) {
    return ViewerCameraInteractionMode::None;
  }
  if (button == ViewerSessionPointerButton::Secondary) {
    return ViewerCameraInteractionMode::Zoom;
  }
  if (button == ViewerSessionPointerButton::Middle ||
      (button == ViewerSessionPointerButton::Primary &&
       ((modifiers & kViewerSessionModifierShift) != 0u ||
        plotModel == kPlotModelWaveform || plotModel == kPlotModelHistogram))) {
    return ViewerCameraInteractionMode::Pan;
  }
  return button == ViewerSessionPointerButton::Primary
             ? ViewerCameraInteractionMode::Orbit
             : ViewerCameraInteractionMode::None;
}

bool updateCameraForPointer(const ViewerCameraInteractionState& state,
                            const ViewerSessionTransientInput& input,
                            const ViewerUiWindowScene& scene,
                            CameraState* camera) noexcept {
  if (!camera) return false;
  const float x = static_cast<float>(input.logicalX);
  const float y = static_cast<float>(input.logicalY);
  const float dx = x - state.lastPointerX;
  const float dy = y - state.lastPointerY;
  if (std::abs(dx) <= 1.0e-7f && std::abs(dy) <= 1.0e-7f) return false;
  const float height = std::max(1.0f, scene.contentRect.y1 - scene.contentRect.y0);
  const float modifierSpeed = speed(
      input.modifiers,
      state.pointerButton == ViewerSessionPointerButton::Primary &&
          state.pointerMode == ViewerCameraInteractionMode::Pan);
  if (state.pointerMode == ViewerCameraInteractionMode::Zoom) {
    const float dominant = std::abs(dy) >= std::abs(dx) ? dy : -dx;
    camera->distance = std::clamp(
        camera->distance * std::exp(dominant * 0.01f * modifierSpeed),
        minimumDistance(*camera), kMaxCameraDistance);
    return true;
  }
  if (state.pointerMode == ViewerCameraInteractionMode::Pan) {
    const float distanceScale =
        std::clamp(camera->distance / 6.0f, 0.11f, 1.0f);
    const float panScale = 2.0f / height * distanceScale *
                           (modifierSpeed == 2.0f ? 2.15f : modifierSpeed);
    camera->panX += dx * panScale;
    camera->panY -= dy * panScale;
    return true;
  }
  if (state.pointerMode == ViewerCameraInteractionMode::Orbit) {
    const Vec3 before = arcball(state.lastPointerX, state.lastPointerY,
                                scene.contentRect);
    const Vec3 after = arcball(x, y, scene.contentRect);
    const Vec3 axis = cross(before, after);
    const float cosine = std::clamp(dot(before, after), -1.0f, 1.0f);
    const float angle = std::acos(cosine) * modifierSpeed;
    if (length(axis) <= 1.0e-7f || !finite(angle) || angle <= 1.0e-7f) {
      return false;
    }
    const Quat next = normalized(multiply(
        axisAngle(axis, angle),
        {camera->qx, camera->qy, camera->qz, camera->qw}));
    camera->qx = next.x;
    camera->qy = next.y;
    camera->qz = next.z;
    camera->qw = next.w;
    return true;
  }
  return false;
}

bool zoomCamera(CameraState* camera, float exponent) noexcept {
  if (!camera || !finite(exponent)) return false;
  const float before = camera->distance;
  camera->distance = std::clamp(camera->distance * std::exp(exponent),
                                minimumDistance(*camera),
                                kMaxCameraDistance);
  return camera->distance != before;
}

bool rollCamera(CameraState* camera, float radians) noexcept {
  if (!camera || !finite(radians) || std::abs(radians) <= 1.0e-8f) return false;
  const Quat next = normalized(multiply(
      axisAngle({0.0f, 0.0f, 1.0f}, -radians),
      {camera->qx, camera->qy, camera->qz, camera->qw}));
  camera->qx = next.x;
  camera->qy = next.y;
  camera->qz = next.z;
  camera->qw = next.w;
  return true;
}

}  // namespace

ViewerCameraInteractionResult reduceViewerCameraInteraction(
    ViewerCameraInteractionState* state,
    const ViewerCameraInteractionRequest& request) noexcept {
  ViewerCameraInteractionState empty{};
  if (!state) return rejected(ViewerCameraInteractionStatus::InvalidState, empty);
  const ViewerCameraInteractionState original = *state;
  if (!validState(original)) {
    return rejected(ViewerCameraInteractionStatus::InvalidState, original);
  }
  if (!request.input || !request.session || !request.scene ||
      !request.workspace || !request.input->accepted()) {
    return rejected(ViewerCameraInteractionStatus::InvalidInput, original);
  }

  ViewerCameraInteractionState next = original;
  ViewerCameraInteractionResult output{};
  output.status = ViewerCameraInteractionStatus::NoChange;
  output.interactionRevision = original.interactionRevision;
  const auto& accepted = request.input->acceptedInput;

  if (request.input->cancelInteractions ||
      accepted.kind == ViewerSessionInputKind::Cancelled) {
    if (next.pointerCaptureActive || next.gestureCaptureActive) {
      clearPointer(&next);
      clearGesture(&next);
      if (!setRevision(&next)) {
        return rejected(ViewerCameraInteractionStatus::RevisionOverflow, original);
      }
      output.status = ViewerCameraInteractionStatus::Accepted;
      output.stateChanged = true;
      output.interactionRevision = next.interactionRevision;
    }
    if (accepted.sequence > next.lastInputSequence) {
      next.lastInputSequence = accepted.sequence;
    }
    *state = next;
    return output;
  }

  if (!request.input->inputAccepted || accepted.kind == ViewerSessionInputKind::None ||
      accepted.sequence == 0u) {
    return output;
  }
  if (accepted.sequence <= original.lastInputSequence) {
    return rejected(ViewerCameraInteractionStatus::ReplayedInput, original);
  }

  const bool needsCameraContext =
      (accepted.kind == ViewerSessionInputKind::PointerButton &&
       ((accepted.pressed && request.authorizeCameraStart) ||
        (!accepted.pressed && next.pointerCaptureActive))) ||
      (accepted.kind == ViewerSessionInputKind::PointerMoved &&
       next.pointerCaptureActive) ||
      accepted.kind == ViewerSessionInputKind::Scroll ||
      accepted.kind == ViewerSessionInputKind::Gesture;
  if (needsCameraContext && !request.scene->ready()) {
    return rejected(ViewerCameraInteractionStatus::InvalidScene, original);
  }
  if (needsCameraContext &&
      !validateViewerWorkspaceState(*request.workspace)) {
    return rejected(ViewerCameraInteractionStatus::InvalidWorkspace, original);
  }
  next.lastInputSequence = accepted.sequence;

  auto publishStateChange = [&]() -> bool {
    if (!setRevision(&next)) return false;
    output.status = ViewerCameraInteractionStatus::Accepted;
    output.stateChanged = true;
    output.interactionRevision = next.interactionRevision;
    return true;
  };
  auto publishCamera = [&](int windowId, const CameraState& camera) -> bool {
    if (!validCamera(camera)) return false;
    if (!output.stateChanged && !publishStateChange()) return false;
    output.cameraChanged = true;
    output.windowId = windowId;
    output.camera = camera;
    return true;
  };

  if (accepted.kind == ViewerSessionInputKind::PointerButton) {
    if (accepted.pressed) {
      if (request.authorizeCameraStart && !next.pointerCaptureActive) {
        const ViewerUiHitResult hit = viewerUiHitTest(
            *request.scene, static_cast<float>(accepted.logicalX),
            static_cast<float>(accepted.logicalY));
        const PlotWindowDomainState* window =
            workspaceWindow(*request.workspace, hit.windowId);
        if (hit.control == ViewerUiControlKind::PlotBody && window) {
          const ViewerCameraInteractionMode mode = modeForPress(
              accepted.button, accepted.modifiers, window->viewState.plotModel);
          if (mode != ViewerCameraInteractionMode::None) {
            next.pointerCaptureActive = true;
            next.pointerWindowId = hit.windowId;
            next.pointerMode = mode;
            next.pointerButton = accepted.button;
            next.lastPointerX = static_cast<float>(accepted.logicalX);
            next.lastPointerY = static_cast<float>(accepted.logicalY);
            if (!publishStateChange()) {
              return rejected(ViewerCameraInteractionStatus::RevisionOverflow,
                              original);
            }
          }
        }
      }
    } else if (next.pointerCaptureActive &&
               accepted.button == next.pointerButton) {
      clearPointer(&next);
      if (!publishStateChange()) {
        return rejected(ViewerCameraInteractionStatus::RevisionOverflow,
                        original);
      }
    }
  } else if (accepted.kind == ViewerSessionInputKind::PointerMoved &&
             next.pointerCaptureActive) {
    const PlotWindowDomainState* window =
        workspaceWindow(*request.workspace, next.pointerWindowId);
    const ViewerUiWindowScene* ui =
        sceneWindow(*request.scene, next.pointerWindowId);
    if (!window || !ui) {
      return rejected(ViewerCameraInteractionStatus::MissingWindow, original);
    }
    if (!validCamera(window->camera)) {
      return rejected(ViewerCameraInteractionStatus::InvalidCamera, original);
    }
    CameraState camera = window->camera;
    const bool cameraChanged =
        updateCameraForPointer(next, accepted, *ui, &camera);
    next.lastPointerX = static_cast<float>(accepted.logicalX);
    next.lastPointerY = static_cast<float>(accepted.logicalY);
    if (!publishStateChange()) {
      return rejected(ViewerCameraInteractionStatus::RevisionOverflow, original);
    }
    if (cameraChanged && !publishCamera(next.pointerWindowId, camera)) {
      return rejected(ViewerCameraInteractionStatus::InvalidCamera, original);
    }
  } else if (accepted.kind == ViewerSessionInputKind::Scroll) {
    const ViewerUiHitResult hit = viewerUiHitTest(
        *request.scene, static_cast<float>(request.session->pointerX),
        static_cast<float>(request.session->pointerY));
    const PlotWindowDomainState* window =
        hit.control == ViewerUiControlKind::PlotBody
            ? workspaceWindow(*request.workspace, hit.windowId)
            : nullptr;
    if (window && window->viewState.plotModel == kPlotModelSourceSignal) {
      window = nullptr;
    }
    if (window) {
      if (!validCamera(window->camera)) {
        return rejected(ViewerCameraInteractionStatus::InvalidCamera, original);
      }
      const double delta = std::abs(accepted.deltaY) > 1.0e-8
                               ? accepted.deltaY
                               : accepted.deltaX;
      const float base = (accepted.modifiers & kViewerSessionModifierControl)
                             ? 0.20f
                             : 0.12f;
      CameraState camera = window->camera;
      if (zoomCamera(&camera, static_cast<float>(-delta) * base *
                                  speed(accepted.modifiers))) {
        if (!publishCamera(hit.windowId, camera)) {
          return rejected(ViewerCameraInteractionStatus::InvalidCamera,
                          original);
        }
      }
    }
  } else if (accepted.kind == ViewerSessionInputKind::Gesture) {
    if (accepted.gesturePhase == ViewerSessionGesturePhase::Begin) {
      const ViewerUiHitResult hit = viewerUiHitTest(
          *request.scene, static_cast<float>(request.session->pointerX),
          static_cast<float>(request.session->pointerY));
      if (hit.control == ViewerUiControlKind::PlotBody &&
          workspaceWindow(*request.workspace, hit.windowId) &&
          workspaceWindow(*request.workspace, hit.windowId)
                  ->viewState.plotModel != kPlotModelSourceSignal) {
        next.gestureCaptureActive = true;
        next.gestureWindowId = hit.windowId;
        next.gestureKind = accepted.gesture;
        if (!publishStateChange()) {
          return rejected(ViewerCameraInteractionStatus::RevisionOverflow,
                          original);
        }
      }
    }
    if (next.gestureCaptureActive &&
        accepted.gesture == next.gestureKind) {
      const PlotWindowDomainState* window =
          workspaceWindow(*request.workspace, next.gestureWindowId);
      if (!window || !sceneWindow(*request.scene, next.gestureWindowId)) {
        return rejected(ViewerCameraInteractionStatus::MissingWindow, original);
      }
      if (!validCamera(window->camera)) {
        return rejected(ViewerCameraInteractionStatus::InvalidCamera, original);
      }
      CameraState camera = window->camera;
      bool changed = false;
      if (accepted.gesture == ViewerSessionGestureKind::Magnify) {
        changed = zoomCamera(
            &camera, static_cast<float>(-accepted.gestureDelta) * 1.5f *
                         speed(accepted.modifiers));
      } else {
        changed = rollCamera(
            &camera, static_cast<float>(accepted.gestureDelta) *
                         speed(accepted.modifiers));
      }
      if (changed && !publishCamera(next.gestureWindowId, camera)) {
        return rejected(ViewerCameraInteractionStatus::InvalidCamera, original);
      }
      if (accepted.gesturePhase == ViewerSessionGesturePhase::End ||
          accepted.gesturePhase == ViewerSessionGesturePhase::Cancel) {
        clearGesture(&next);
        if (!output.stateChanged && !publishStateChange()) {
          return rejected(ViewerCameraInteractionStatus::RevisionOverflow,
                          original);
        }
      }
    }
  }

  *state = next;
  return output;
}

}  // namespace ChromaspaceViewer
