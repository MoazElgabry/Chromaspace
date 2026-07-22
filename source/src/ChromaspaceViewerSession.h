#pragma once

#include <cstdint>
#include <variant>

namespace ChromaspaceViewer {

struct ViewerSessionViewport {
  int logicalWidth = 0;
  int logicalHeight = 0;
  int framebufferWidth = 0;
  int framebufferHeight = 0;
  float contentScaleX = 0.0f;
  float contentScaleY = 0.0f;
};

enum class ViewerSessionPointerButton : uint8_t {
  Primary = 0,
  Secondary,
  Middle,
  Button4,
  Button5,
  Button6,
  Button7,
  Button8,
  Count,
};

enum class ViewerSessionKey : uint8_t {
  Unknown = 0,
  A,
  B,
  C,
  D,
  F,
  L,
  M,
  R,
  S,
  V,
  Z,
  Backspace,
  Enter,
  Escape,
  Tab,
  Count,
};

using ViewerSessionModifierMask = uint8_t;
constexpr ViewerSessionModifierMask kViewerSessionModifierShift = 1u << 0;
constexpr ViewerSessionModifierMask kViewerSessionModifierControl = 1u << 1;
constexpr ViewerSessionModifierMask kViewerSessionModifierAlt = 1u << 2;
constexpr ViewerSessionModifierMask kViewerSessionModifierSuper = 1u << 3;
constexpr ViewerSessionModifierMask kViewerSessionModifierAll =
    kViewerSessionModifierShift | kViewerSessionModifierControl |
    kViewerSessionModifierAlt | kViewerSessionModifierSuper;

enum class ViewerSessionGestureKind : uint8_t {
  Magnify = 0,
  Rotate,
  Count,
};

enum class ViewerSessionGesturePhase : uint8_t {
  Begin = 0,
  Update,
  End,
  Cancel,
  Count,
};

enum class ViewerSessionInputKind : uint8_t {
  None = 0,
  PointerEntered,
  PointerLeft,
  PointerMoved,
  PointerButton,
  Modifiers,
  Key,
  Text,
  Scroll,
  Gesture,
  // A lifecycle transition cleared one or more active input sources.  This
  // is intentionally a typed transient so inputRevision and lastInput never
  // describe a stale key/button/gesture snapshot after cancellation.
  Cancelled,
};

constexpr uint16_t viewerSessionPointerButtonBit(
    ViewerSessionPointerButton button) noexcept {
  return button < ViewerSessionPointerButton::Count
             ? static_cast<uint16_t>(1u << static_cast<uint8_t>(button))
             : 0u;
}

constexpr uint64_t viewerSessionKeyBit(ViewerSessionKey key) noexcept {
  return key > ViewerSessionKey::Unknown && key < ViewerSessionKey::Count
             ? uint64_t{1} << static_cast<uint8_t>(key)
             : 0u;
}

constexpr uint8_t viewerSessionGestureBit(
    ViewerSessionGestureKind gesture) noexcept {
  return gesture < ViewerSessionGestureKind::Count
             ? static_cast<uint8_t>(1u << static_cast<uint8_t>(gesture))
             : 0u;
}

struct ViewerSessionInitialize {
  ViewerSessionViewport viewport{};
  bool focused = true;
  bool visible = true;
  bool miniaturized = false;
};

struct ViewerSessionViewportChanged {
  ViewerSessionViewport viewport{};
};

struct ViewerSessionFocusChanged {
  bool focused = false;
};

struct ViewerSessionVisibilityChanged {
  bool visible = false;
};

struct ViewerSessionMiniaturizationChanged {
  bool miniaturized = false;
};

struct ViewerSessionCloseRequested {};

// Pointer positions are logical pixels in top-left/Y-down viewer space.
struct ViewerSessionPointerEntered {
  double logicalX = 0.0;
  double logicalY = 0.0;
};

struct ViewerSessionPointerLeft {};

struct ViewerSessionPointerMoved {
  double logicalX = 0.0;
  double logicalY = 0.0;
};

struct ViewerSessionPointerButtonChanged {
  ViewerSessionPointerButton button = ViewerSessionPointerButton::Primary;
  bool pressed = false;
  double logicalX = 0.0;
  double logicalY = 0.0;
  ViewerSessionModifierMask modifiers = 0;
  // The platform adapter normalizes consecutive primary-button clicks before
  // reducing the event.  One is the ordinary single-click value; values >= 2
  // are consecutive clicks in the same gesture family (for example a
  // double-click).  Releases carry the corresponding press count or one.
  uint8_t clickCount = 1;
};

struct ViewerSessionModifiersChanged {
  ViewerSessionModifierMask modifiers = 0;
};

struct ViewerSessionKeyChanged {
  ViewerSessionKey key = ViewerSessionKey::Unknown;
  bool pressed = false;
  bool repeat = false;
  ViewerSessionModifierMask modifiers = 0;
};

struct ViewerSessionTextInput {
  char32_t codepoint = U'\0';
};

struct ViewerSessionScroll {
  double deltaX = 0.0;
  double deltaY = 0.0;
  ViewerSessionModifierMask modifiers = 0;
};

struct ViewerSessionGesture {
  ViewerSessionGestureKind kind = ViewerSessionGestureKind::Magnify;
  ViewerSessionGesturePhase phase = ViewerSessionGesturePhase::Begin;
  // Magnify is a fractional delta. Rotate is a delta in radians.
  double delta = 0.0;
  ViewerSessionModifierMask modifiers = 0;
};

using ViewerSessionEventPayload =
    std::variant<ViewerSessionInitialize,
                 ViewerSessionViewportChanged,
                 ViewerSessionFocusChanged,
                 ViewerSessionVisibilityChanged,
                 ViewerSessionMiniaturizationChanged,
                 ViewerSessionCloseRequested,
                 ViewerSessionPointerEntered,
                 ViewerSessionPointerLeft,
                 ViewerSessionPointerMoved,
                 ViewerSessionPointerButtonChanged,
                 ViewerSessionModifiersChanged,
                 ViewerSessionKeyChanged,
                 ViewerSessionTextInput,
                 ViewerSessionScroll,
                 ViewerSessionGesture>;

struct ViewerSessionEvent {
  uint64_t sequence = 0;
  ViewerSessionEventPayload payload{};
};

struct ViewerSessionTransientInput {
  uint64_t sequence = 0;
  ViewerSessionInputKind kind = ViewerSessionInputKind::None;
  ViewerSessionPointerButton button = ViewerSessionPointerButton::Primary;
  ViewerSessionKey key = ViewerSessionKey::Unknown;
  ViewerSessionGestureKind gesture = ViewerSessionGestureKind::Magnify;
  ViewerSessionGesturePhase gesturePhase = ViewerSessionGesturePhase::Begin;
  char32_t codepoint = U'\0';
  double logicalX = 0.0;
  double logicalY = 0.0;
  double deltaX = 0.0;
  double deltaY = 0.0;
  double gestureDelta = 0.0;
  ViewerSessionModifierMask modifiers = 0;
  bool pressed = false;
  bool repeat = false;
  uint8_t clickCount = 1;
};

struct ViewerSessionState {
  uint64_t lastAcceptedSequence = 0;
  uint64_t lifecycleRevision = 0;
  uint64_t viewportRevision = 0;
  uint64_t inputRevision = 0;
  ViewerSessionViewport viewport{};
  ViewerSessionTransientInput lastInput{};
  double pointerX = 0.0;
  double pointerY = 0.0;
  uint64_t pressedKeys = 0;
  uint16_t pressedPointerButtons = 0;
  ViewerSessionModifierMask modifiers = 0;
  uint8_t activeGestures = 0;
  bool pointerPresent = false;
  bool initialized = false;
  bool focused = false;
  bool visible = false;
  bool miniaturized = false;
  bool closeRequested = false;
};

enum class ViewerSessionReduceStatus : uint8_t {
  Applied = 0,
  NoChange,
  RejectedStaleSequence,
  RejectedSequenceExhausted,
  RejectedInvalidViewport,
  RejectedInvalidPointer,
  RejectedInvalidButton,
  RejectedInvalidKey,
  RejectedInvalidModifiers,
  RejectedInvalidText,
  RejectedInvalidScroll,
  RejectedInvalidGesture,
  RejectedInvalidInputTransition,
  RejectedNotInitialized,
  RejectedAlreadyInitialized,
  RejectedClosed,
};

struct ViewerSessionReduceResult {
  ViewerSessionReduceStatus status =
      ViewerSessionReduceStatus::RejectedNotInitialized;
  ViewerSessionTransientInput acceptedInput{};
  bool changed = false;
  bool inputAccepted = false;
  bool requestFrame = false;
  bool cancelInteractions = false;
  bool shouldRender = false;
  bool shouldClose = false;

  bool accepted() const noexcept {
    return status == ViewerSessionReduceStatus::Applied ||
           status == ViewerSessionReduceStatus::NoChange;
  }
};

bool viewerSessionViewportValid(
    const ViewerSessionViewport& viewport) noexcept;

bool viewerSessionUnicodeScalarValid(char32_t codepoint) noexcept;

bool viewerSessionShouldRender(const ViewerSessionState& state) noexcept;

bool viewerSessionShouldClose(const ViewerSessionState& state) noexcept;

uint64_t viewerSessionNextSequence(const ViewerSessionState& state) noexcept;

ViewerSessionReduceResult reduceViewerSession(
    ViewerSessionState* state,
    const ViewerSessionEvent& event) noexcept;

}  // namespace ChromaspaceViewer
