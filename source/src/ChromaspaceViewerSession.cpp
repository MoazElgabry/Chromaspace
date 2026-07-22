#include "ChromaspaceViewerSession.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace ChromaspaceViewer {
namespace {

bool sameViewport(const ViewerSessionViewport& a,
                  const ViewerSessionViewport& b) noexcept {
  return a.logicalWidth == b.logicalWidth &&
         a.logicalHeight == b.logicalHeight &&
         a.framebufferWidth == b.framebufferWidth &&
         a.framebufferHeight == b.framebufferHeight &&
         a.contentScaleX == b.contentScaleX &&
         a.contentScaleY == b.contentScaleY;
}

bool validModifiers(ViewerSessionModifierMask modifiers) noexcept {
  return (modifiers & ~kViewerSessionModifierAll) == 0;
}

bool finitePoint(double x, double y) noexcept {
  return std::isfinite(x) && std::isfinite(y);
}

bool validButton(ViewerSessionPointerButton button) noexcept {
  return button < ViewerSessionPointerButton::Count;
}

bool validKey(ViewerSessionKey key) noexcept {
  return key > ViewerSessionKey::Unknown && key < ViewerSessionKey::Count;
}

bool validGesture(ViewerSessionGestureKind gesture) noexcept {
  return gesture < ViewerSessionGestureKind::Count;
}

bool validGesturePhase(ViewerSessionGesturePhase phase) noexcept {
  return phase < ViewerSessionGesturePhase::Count;
}

bool isTransientInput(const ViewerSessionEventPayload& payload) noexcept {
  return std::holds_alternative<ViewerSessionPointerEntered>(payload) ||
         std::holds_alternative<ViewerSessionPointerLeft>(payload) ||
         std::holds_alternative<ViewerSessionPointerMoved>(payload) ||
         std::holds_alternative<ViewerSessionPointerButtonChanged>(payload) ||
         std::holds_alternative<ViewerSessionModifiersChanged>(payload) ||
         std::holds_alternative<ViewerSessionKeyChanged>(payload) ||
         std::holds_alternative<ViewerSessionTextInput>(payload) ||
         std::holds_alternative<ViewerSessionScroll>(payload) ||
         std::holds_alternative<ViewerSessionGesture>(payload);
}

ViewerSessionReduceStatus validateTransientInput(
    const ViewerSessionState& state,
    const ViewerSessionEventPayload& payload) noexcept {
  if (const auto* entered =
          std::get_if<ViewerSessionPointerEntered>(&payload)) {
    return finitePoint(entered->logicalX, entered->logicalY)
               ? ViewerSessionReduceStatus::Applied
               : ViewerSessionReduceStatus::RejectedInvalidPointer;
  }
  if (std::holds_alternative<ViewerSessionPointerLeft>(payload)) {
    return ViewerSessionReduceStatus::Applied;
  }
  if (const auto* moved =
          std::get_if<ViewerSessionPointerMoved>(&payload)) {
    return finitePoint(moved->logicalX, moved->logicalY)
               ? ViewerSessionReduceStatus::Applied
               : ViewerSessionReduceStatus::RejectedInvalidPointer;
  }
  if (const auto* button =
          std::get_if<ViewerSessionPointerButtonChanged>(&payload)) {
    if (!validButton(button->button)) {
      return ViewerSessionReduceStatus::RejectedInvalidButton;
    }
    if (!finitePoint(button->logicalX, button->logicalY)) {
      return ViewerSessionReduceStatus::RejectedInvalidPointer;
    }
    if (!validModifiers(button->modifiers)) {
      return ViewerSessionReduceStatus::RejectedInvalidModifiers;
    }
    if (button->clickCount == 0u) {
      return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
    }
    const bool down =
        (state.pressedPointerButtons &
         viewerSessionPointerButtonBit(button->button)) != 0;
    if (down == button->pressed) {
      return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
    }
    return ViewerSessionReduceStatus::Applied;
  }
  if (const auto* modifiers =
          std::get_if<ViewerSessionModifiersChanged>(&payload)) {
    return validModifiers(modifiers->modifiers)
               ? ViewerSessionReduceStatus::Applied
               : ViewerSessionReduceStatus::RejectedInvalidModifiers;
  }
  if (const auto* key = std::get_if<ViewerSessionKeyChanged>(&payload)) {
    if (!validKey(key->key)) {
      return ViewerSessionReduceStatus::RejectedInvalidKey;
    }
    if (!validModifiers(key->modifiers)) {
      return ViewerSessionReduceStatus::RejectedInvalidModifiers;
    }
    if (key->repeat && !key->pressed) {
      return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
    }
    const bool down =
        (state.pressedKeys & viewerSessionKeyBit(key->key)) != 0;
    if ((key->pressed && key->repeat && !down) ||
        (key->pressed && !key->repeat && down) ||
        (!key->pressed && !down)) {
      return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
    }
    return ViewerSessionReduceStatus::Applied;
  }
  if (const auto* text = std::get_if<ViewerSessionTextInput>(&payload)) {
    return viewerSessionUnicodeScalarValid(text->codepoint)
               ? ViewerSessionReduceStatus::Applied
               : ViewerSessionReduceStatus::RejectedInvalidText;
  }
  if (const auto* scroll = std::get_if<ViewerSessionScroll>(&payload)) {
    if (!std::isfinite(scroll->deltaX) ||
        !std::isfinite(scroll->deltaY)) {
      return ViewerSessionReduceStatus::RejectedInvalidScroll;
    }
    return validModifiers(scroll->modifiers)
               ? ViewerSessionReduceStatus::Applied
               : ViewerSessionReduceStatus::RejectedInvalidModifiers;
  }
  if (const auto* gesture = std::get_if<ViewerSessionGesture>(&payload)) {
    if (!validGesture(gesture->kind) ||
        !validGesturePhase(gesture->phase) ||
        !std::isfinite(gesture->delta)) {
      return ViewerSessionReduceStatus::RejectedInvalidGesture;
    }
    if (!validModifiers(gesture->modifiers)) {
      return ViewerSessionReduceStatus::RejectedInvalidModifiers;
    }
    const bool active =
        (state.activeGestures & viewerSessionGestureBit(gesture->kind)) != 0;
    const bool begin = gesture->phase == ViewerSessionGesturePhase::Begin;
    if (begin == active) {
      return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
    }
    return ViewerSessionReduceStatus::Applied;
  }
  return ViewerSessionReduceStatus::RejectedInvalidInputTransition;
}

bool clearActiveInput(ViewerSessionState* state,
                      bool clearPointerPresence) noexcept {
  bool changed = false;
  if (state->pressedKeys != 0) {
    state->pressedKeys = 0;
    changed = true;
  }
  if (state->pressedPointerButtons != 0) {
    state->pressedPointerButtons = 0;
    changed = true;
  }
  if (state->modifiers != 0) {
    state->modifiers = 0;
    changed = true;
  }
  if (state->activeGestures != 0) {
    state->activeGestures = 0;
    changed = true;
  }
  if (clearPointerPresence && state->pointerPresent) {
    state->pointerPresent = false;
    changed = true;
  }
  return changed;
}

void incrementRevision(uint64_t* revision) noexcept {
  if (revision && *revision != std::numeric_limits<uint64_t>::max()) {
    ++(*revision);
  }
}

ViewerSessionReduceResult rejectedResult(
    const ViewerSessionState& state,
    ViewerSessionReduceStatus status) noexcept {
  ViewerSessionReduceResult result{};
  result.status = status;
  result.shouldRender = viewerSessionShouldRender(state);
  result.shouldClose = viewerSessionShouldClose(state);
  return result;
}

}  // namespace

bool viewerSessionViewportValid(
    const ViewerSessionViewport& viewport) noexcept {
  return viewport.logicalWidth > 0 && viewport.logicalHeight > 0 &&
         viewport.framebufferWidth > 0 && viewport.framebufferHeight > 0 &&
         std::isfinite(viewport.contentScaleX) &&
         std::isfinite(viewport.contentScaleY) &&
         viewport.contentScaleX > 0.0f && viewport.contentScaleY > 0.0f;
}

bool viewerSessionUnicodeScalarValid(char32_t codepoint) noexcept {
  const uint32_t value = static_cast<uint32_t>(codepoint);
  return value <= 0x10ffffu && !(value >= 0xd800u && value <= 0xdfffu);
}

bool viewerSessionShouldRender(const ViewerSessionState& state) noexcept {
  return state.initialized && viewerSessionViewportValid(state.viewport) &&
         state.visible && !state.miniaturized && !state.closeRequested;
}

bool viewerSessionShouldClose(const ViewerSessionState& state) noexcept {
  return state.closeRequested;
}

uint64_t viewerSessionNextSequence(const ViewerSessionState& state) noexcept {
  if (state.lastAcceptedSequence == std::numeric_limits<uint64_t>::max()) {
    return state.lastAcceptedSequence;
  }
  return state.lastAcceptedSequence + 1u;
}

ViewerSessionReduceResult reduceViewerSession(
    ViewerSessionState* state,
    const ViewerSessionEvent& event) noexcept {
  if (!state) {
    ViewerSessionReduceResult result{};
    result.status = ViewerSessionReduceStatus::RejectedNotInitialized;
    return result;
  }
  if (state->lastAcceptedSequence ==
      std::numeric_limits<uint64_t>::max()) {
    return rejectedResult(
        *state, ViewerSessionReduceStatus::RejectedSequenceExhausted);
  }
  if (event.sequence <= state->lastAcceptedSequence) {
    return rejectedResult(
        *state, ViewerSessionReduceStatus::RejectedStaleSequence);
  }

  const bool isInitialize =
      std::holds_alternative<ViewerSessionInitialize>(event.payload);
  if (state->closeRequested) {
    return rejectedResult(*state, ViewerSessionReduceStatus::RejectedClosed);
  }
  if (isInitialize && state->initialized) {
    return rejectedResult(
        *state, ViewerSessionReduceStatus::RejectedAlreadyInitialized);
  }
  if (!isInitialize && !state->initialized) {
    return rejectedResult(
        *state, ViewerSessionReduceStatus::RejectedNotInitialized);
  }

  if (const auto* initialize =
          std::get_if<ViewerSessionInitialize>(&event.payload)) {
    if (!viewerSessionViewportValid(initialize->viewport)) {
      return rejectedResult(
          *state, ViewerSessionReduceStatus::RejectedInvalidViewport);
    }
  }
  if (const auto* viewport =
          std::get_if<ViewerSessionViewportChanged>(&event.payload)) {
    if (!viewerSessionViewportValid(viewport->viewport)) {
      return rejectedResult(
          *state, ViewerSessionReduceStatus::RejectedInvalidViewport);
    }
  }
  const bool transientInput = isTransientInput(event.payload);
  if (transientInput) {
    const ViewerSessionReduceStatus validation =
        validateTransientInput(*state, event.payload);
    if (validation != ViewerSessionReduceStatus::Applied) {
      return rejectedResult(*state, validation);
    }
  }

  ViewerSessionState next = *state;
  ViewerSessionTransientInput acceptedInput{};
  acceptedInput.sequence = event.sequence;
  bool lifecycleChanged = false;
  bool viewportChanged = false;
  bool inputChanged = false;
  bool inputAccepted = false;
  bool cancelInteractions = false;
  bool lifecycleCancellation = false;
  std::visit(
      [&](const auto& payload) {
        using T = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<T, ViewerSessionInitialize>) {
          next.initialized = true;
          next.viewport = payload.viewport;
          next.focused = payload.focused;
          next.visible = payload.visible;
          next.miniaturized = payload.miniaturized;
          lifecycleChanged = true;
          viewportChanged = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionViewportChanged>) {
          if (!sameViewport(next.viewport, payload.viewport)) {
            next.viewport = payload.viewport;
            viewportChanged = true;
          }
        } else if constexpr (
            std::is_same_v<T, ViewerSessionFocusChanged>) {
          if (next.focused != payload.focused) {
            next.focused = payload.focused;
            lifecycleChanged = true;
            cancelInteractions = !payload.focused;
            if (!payload.focused && clearActiveInput(&next, true)) {
              inputChanged = true;
              lifecycleCancellation = true;
            }
          }
        } else if constexpr (
            std::is_same_v<T, ViewerSessionVisibilityChanged>) {
          if (next.visible != payload.visible) {
            next.visible = payload.visible;
            lifecycleChanged = true;
            cancelInteractions = !payload.visible;
            if (!payload.visible && clearActiveInput(&next, true)) {
              inputChanged = true;
              lifecycleCancellation = true;
            }
          }
        } else if constexpr (
            std::is_same_v<T, ViewerSessionMiniaturizationChanged>) {
          if (next.miniaturized != payload.miniaturized) {
            next.miniaturized = payload.miniaturized;
            lifecycleChanged = true;
            cancelInteractions = payload.miniaturized;
            if (payload.miniaturized && clearActiveInput(&next, true)) {
              inputChanged = true;
              lifecycleCancellation = true;
            }
          }
        } else if constexpr (
            std::is_same_v<T, ViewerSessionCloseRequested>) {
          next.closeRequested = true;
          lifecycleChanged = true;
          cancelInteractions = true;
          if (clearActiveInput(&next, true)) {
            inputChanged = true;
            lifecycleCancellation = true;
          }
        } else if constexpr (
            std::is_same_v<T, ViewerSessionPointerEntered>) {
          acceptedInput.kind = ViewerSessionInputKind::PointerEntered;
          acceptedInput.logicalX = payload.logicalX;
          acceptedInput.logicalY = payload.logicalY;
          next.pointerPresent = true;
          next.pointerX = payload.logicalX;
          next.pointerY = payload.logicalY;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionPointerLeft>) {
          acceptedInput.kind = ViewerSessionInputKind::PointerLeft;
          next.pointerPresent = false;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionPointerMoved>) {
          acceptedInput.kind = ViewerSessionInputKind::PointerMoved;
          acceptedInput.logicalX = payload.logicalX;
          acceptedInput.logicalY = payload.logicalY;
          next.pointerPresent = true;
          next.pointerX = payload.logicalX;
          next.pointerY = payload.logicalY;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionPointerButtonChanged>) {
          acceptedInput.kind = ViewerSessionInputKind::PointerButton;
          acceptedInput.button = payload.button;
          acceptedInput.pressed = payload.pressed;
          acceptedInput.logicalX = payload.logicalX;
          acceptedInput.logicalY = payload.logicalY;
          acceptedInput.modifiers = payload.modifiers;
          acceptedInput.clickCount = payload.clickCount;
          next.pointerPresent = true;
          next.pointerX = payload.logicalX;
          next.pointerY = payload.logicalY;
          next.modifiers = payload.modifiers;
          const uint16_t bit = viewerSessionPointerButtonBit(payload.button);
          if (payload.pressed) {
            next.pressedPointerButtons |= bit;
          } else {
            next.pressedPointerButtons &= static_cast<uint16_t>(~bit);
          }
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionModifiersChanged>) {
          acceptedInput.kind = ViewerSessionInputKind::Modifiers;
          acceptedInput.modifiers = payload.modifiers;
          next.modifiers = payload.modifiers;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionKeyChanged>) {
          acceptedInput.kind = ViewerSessionInputKind::Key;
          acceptedInput.key = payload.key;
          acceptedInput.pressed = payload.pressed;
          acceptedInput.repeat = payload.repeat;
          acceptedInput.modifiers = payload.modifiers;
          next.modifiers = payload.modifiers;
          const uint64_t bit = viewerSessionKeyBit(payload.key);
          if (payload.pressed) {
            next.pressedKeys |= bit;
          } else {
            next.pressedKeys &= ~bit;
          }
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionTextInput>) {
          acceptedInput.kind = ViewerSessionInputKind::Text;
          acceptedInput.codepoint = payload.codepoint;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionScroll>) {
          acceptedInput.kind = ViewerSessionInputKind::Scroll;
          acceptedInput.deltaX = payload.deltaX;
          acceptedInput.deltaY = payload.deltaY;
          acceptedInput.modifiers = payload.modifiers;
          next.modifiers = payload.modifiers;
          inputChanged = true;
          inputAccepted = true;
        } else if constexpr (
            std::is_same_v<T, ViewerSessionGesture>) {
          acceptedInput.kind = ViewerSessionInputKind::Gesture;
          acceptedInput.gesture = payload.kind;
          acceptedInput.gesturePhase = payload.phase;
          acceptedInput.gestureDelta = payload.delta;
          acceptedInput.modifiers = payload.modifiers;
          next.modifiers = payload.modifiers;
          const uint8_t bit = viewerSessionGestureBit(payload.kind);
          if (payload.phase == ViewerSessionGesturePhase::Begin) {
            next.activeGestures |= bit;
          } else if (payload.phase == ViewerSessionGesturePhase::End ||
                     payload.phase == ViewerSessionGesturePhase::Cancel) {
            next.activeGestures &= static_cast<uint8_t>(~bit);
          }
          cancelInteractions =
              payload.phase == ViewerSessionGesturePhase::Cancel;
          inputChanged = true;
          inputAccepted = true;
        }
      },
      event.payload);

  next.lastAcceptedSequence = event.sequence;
  if (lifecycleChanged) incrementRevision(&next.lifecycleRevision);
  if (viewportChanged) incrementRevision(&next.viewportRevision);
  if (inputChanged) incrementRevision(&next.inputRevision);
  if (lifecycleCancellation) {
    acceptedInput.kind = ViewerSessionInputKind::Cancelled;
    acceptedInput.button = ViewerSessionPointerButton::Primary;
    acceptedInput.key = ViewerSessionKey::Unknown;
    acceptedInput.gesture = ViewerSessionGestureKind::Magnify;
    acceptedInput.gesturePhase = ViewerSessionGesturePhase::Cancel;
    acceptedInput.codepoint = U'\0';
    acceptedInput.logicalX = next.pointerX;
    acceptedInput.logicalY = next.pointerY;
    acceptedInput.deltaX = 0.0;
    acceptedInput.deltaY = 0.0;
    acceptedInput.gestureDelta = 0.0;
    acceptedInput.modifiers = 0;
    acceptedInput.pressed = false;
    acceptedInput.repeat = false;
    acceptedInput.clickCount = 1;
    inputAccepted = true;
  }
  if (inputAccepted) next.lastInput = acceptedInput;
  const bool changed = lifecycleChanged || viewportChanged || inputChanged;
  *state = next;

  ViewerSessionReduceResult result{};
  result.status = changed ? ViewerSessionReduceStatus::Applied
                          : ViewerSessionReduceStatus::NoChange;
  result.acceptedInput = acceptedInput;
  result.changed = changed;
  result.inputAccepted = inputAccepted;
  result.cancelInteractions = cancelInteractions;
  result.shouldRender = viewerSessionShouldRender(*state);
  result.shouldClose = viewerSessionShouldClose(*state);
  result.requestFrame = changed && result.shouldRender;
  return result;
}

}  // namespace ChromaspaceViewer
