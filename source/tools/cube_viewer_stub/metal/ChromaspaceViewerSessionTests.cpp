#include "ChromaspaceViewerSession.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using ChromaspaceViewer::ViewerSessionCloseRequested;
using ChromaspaceViewer::ViewerSessionEvent;
using ChromaspaceViewer::ViewerSessionFocusChanged;
using ChromaspaceViewer::ViewerSessionGesture;
using ChromaspaceViewer::ViewerSessionGestureKind;
using ChromaspaceViewer::ViewerSessionGesturePhase;
using ChromaspaceViewer::ViewerSessionInitialize;
using ChromaspaceViewer::ViewerSessionInputKind;
using ChromaspaceViewer::ViewerSessionKey;
using ChromaspaceViewer::ViewerSessionKeyChanged;
using ChromaspaceViewer::ViewerSessionMiniaturizationChanged;
using ChromaspaceViewer::ViewerSessionModifiersChanged;
using ChromaspaceViewer::ViewerSessionPointerButton;
using ChromaspaceViewer::ViewerSessionPointerButtonChanged;
using ChromaspaceViewer::ViewerSessionPointerEntered;
using ChromaspaceViewer::ViewerSessionPointerLeft;
using ChromaspaceViewer::ViewerSessionPointerMoved;
using ChromaspaceViewer::ViewerSessionReduceStatus;
using ChromaspaceViewer::ViewerSessionState;
using ChromaspaceViewer::ViewerSessionScroll;
using ChromaspaceViewer::ViewerSessionTextInput;
using ChromaspaceViewer::ViewerSessionViewport;
using ChromaspaceViewer::ViewerSessionViewportChanged;
using ChromaspaceViewer::ViewerSessionVisibilityChanged;

ViewerSessionViewport viewport(int logicalWidth = 640,
                               int logicalHeight = 360,
                               int framebufferWidth = 1280,
                               int framebufferHeight = 720,
                               float scaleX = 2.0f,
                               float scaleY = 2.0f) {
  return {logicalWidth, logicalHeight, framebufferWidth, framebufferHeight,
          scaleX, scaleY};
}

ViewerSessionEvent initializeEvent(
    uint64_t sequence,
    const ViewerSessionViewport& metrics = viewport()) {
  return {sequence,
          ViewerSessionInitialize{metrics, true, true, false}};
}

void assertSameState(const ViewerSessionState& a,
                     const ViewerSessionState& b) {
  assert(a.lastAcceptedSequence == b.lastAcceptedSequence);
  assert(a.lifecycleRevision == b.lifecycleRevision);
  assert(a.viewportRevision == b.viewportRevision);
  assert(a.inputRevision == b.inputRevision);
  assert(a.lastInput.sequence == b.lastInput.sequence);
  assert(a.lastInput.kind == b.lastInput.kind);
  assert(a.lastInput.button == b.lastInput.button);
  assert(a.lastInput.key == b.lastInput.key);
  assert(a.lastInput.gesture == b.lastInput.gesture);
  assert(a.lastInput.gesturePhase == b.lastInput.gesturePhase);
  assert(a.lastInput.codepoint == b.lastInput.codepoint);
  assert(a.lastInput.logicalX == b.lastInput.logicalX);
  assert(a.lastInput.logicalY == b.lastInput.logicalY);
  assert(a.lastInput.deltaX == b.lastInput.deltaX);
  assert(a.lastInput.deltaY == b.lastInput.deltaY);
  assert(a.lastInput.gestureDelta == b.lastInput.gestureDelta);
  assert(a.lastInput.modifiers == b.lastInput.modifiers);
  assert(a.lastInput.pressed == b.lastInput.pressed);
  assert(a.lastInput.repeat == b.lastInput.repeat);
  assert(a.lastInput.clickCount == b.lastInput.clickCount);
  assert(a.pointerX == b.pointerX);
  assert(a.pointerY == b.pointerY);
  assert(a.pressedKeys == b.pressedKeys);
  assert(a.pressedPointerButtons == b.pressedPointerButtons);
  assert(a.modifiers == b.modifiers);
  assert(a.activeGestures == b.activeGestures);
  assert(a.pointerPresent == b.pointerPresent);
  assert(a.viewport.logicalWidth == b.viewport.logicalWidth);
  assert(a.viewport.logicalHeight == b.viewport.logicalHeight);
  assert(a.viewport.framebufferWidth == b.viewport.framebufferWidth);
  assert(a.viewport.framebufferHeight == b.viewport.framebufferHeight);
  assert(a.viewport.contentScaleX == b.viewport.contentScaleX);
  assert(a.viewport.contentScaleY == b.viewport.contentScaleY);
  assert(a.initialized == b.initialized);
  assert(a.focused == b.focused);
  assert(a.visible == b.visible);
  assert(a.miniaturized == b.miniaturized);
  assert(a.closeRequested == b.closeRequested);
}

void testInitializeAndAtomicViewportRevision() {
  ViewerSessionState state{};
  const auto initialized = ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(1));
  assert(initialized.status == ViewerSessionReduceStatus::Applied);
  assert(initialized.changed && initialized.requestFrame);
  assert(!initialized.cancelInteractions);
  assert(initialized.shouldRender && !initialized.shouldClose);
  assert(state.initialized && state.focused && state.visible);
  assert(!state.miniaturized && !state.closeRequested);
  assert(state.lastAcceptedSequence == 1);
  assert(state.lifecycleRevision == 1);
  assert(state.viewportRevision == 1);

  const ViewerSessionViewport retina =
      viewport(800, 450, 1600, 1125, 2.0f, 2.5f);
  const auto resized = ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionViewportChanged{retina}});
  assert(resized.status == ViewerSessionReduceStatus::Applied);
  assert(resized.changed && resized.requestFrame && resized.shouldRender);
  assert(state.viewport.logicalWidth == 800);
  assert(state.viewport.logicalHeight == 450);
  assert(state.viewport.framebufferWidth == 1600);
  assert(state.viewport.framebufferHeight == 1125);
  assert(state.viewport.contentScaleX == 2.0f);
  assert(state.viewport.contentScaleY == 2.5f);
  assert(state.viewportRevision == 2);
  assert(state.lifecycleRevision == 1);

  const auto repeated = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionViewportChanged{retina}});
  assert(repeated.status == ViewerSessionReduceStatus::NoChange);
  assert(!repeated.changed && !repeated.requestFrame);
  assert(state.lastAcceptedSequence == 3);
  assert(state.viewportRevision == 2);
}

void testStaleDuplicateAndInvalidSnapshotsDoNotMutate() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(10));
  const ViewerSessionState baseline = state;

  const auto duplicate = ChromaspaceViewer::reduceViewerSession(
      &state, {10, ViewerSessionFocusChanged{false}});
  assert(duplicate.status ==
         ViewerSessionReduceStatus::RejectedStaleSequence);
  assertSameState(state, baseline);

  const auto stale = ChromaspaceViewer::reduceViewerSession(
      &state, {9, ViewerSessionVisibilityChanged{false}});
  assert(stale.status == ViewerSessionReduceStatus::RejectedStaleSequence);
  assertSameState(state, baseline);

  ViewerSessionViewport invalid = viewport();
  invalid.framebufferWidth = 0;
  const auto zero = ChromaspaceViewer::reduceViewerSession(
      &state, {11, ViewerSessionViewportChanged{invalid}});
  assert(zero.status == ViewerSessionReduceStatus::RejectedInvalidViewport);
  assertSameState(state, baseline);

  invalid = viewport();
  invalid.contentScaleY = std::numeric_limits<float>::quiet_NaN();
  const auto nonfinite = ChromaspaceViewer::reduceViewerSession(
      &state, {12, ViewerSessionViewportChanged{invalid}});
  assert(nonfinite.status ==
         ViewerSessionReduceStatus::RejectedInvalidViewport);
  assertSameState(state, baseline);
}

void testLifecycleCancellationAndRecovery() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(1));

  const auto focusLost = ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionFocusChanged{false}});
  assert(focusLost.changed && focusLost.cancelInteractions);
  assert(focusLost.shouldRender && focusLost.requestFrame);
  assert(state.lifecycleRevision == 2);
  const auto focusRecovered = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionFocusChanged{true}});
  assert(focusRecovered.changed && !focusRecovered.cancelInteractions);
  assert(focusRecovered.shouldRender && focusRecovered.requestFrame);

  const auto hidden = ChromaspaceViewer::reduceViewerSession(
      &state, {4, ViewerSessionVisibilityChanged{false}});
  assert(hidden.changed && hidden.cancelInteractions);
  assert(!hidden.shouldRender && !hidden.requestFrame);
  const auto shown = ChromaspaceViewer::reduceViewerSession(
      &state, {5, ViewerSessionVisibilityChanged{true}});
  assert(shown.changed && shown.shouldRender && shown.requestFrame);

  const auto minimized = ChromaspaceViewer::reduceViewerSession(
      &state, {6, ViewerSessionMiniaturizationChanged{true}});
  assert(minimized.changed && minimized.cancelInteractions);
  assert(!minimized.shouldRender && !minimized.requestFrame);
  const auto restored = ChromaspaceViewer::reduceViewerSession(
      &state, {7, ViewerSessionMiniaturizationChanged{false}});
  assert(restored.changed && restored.shouldRender && restored.requestFrame);
  assert(state.lifecycleRevision == 7);
}

void testNormalizedInputStateAndCancellationMetadata() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(1));

  const auto entered = ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionPointerEntered{12.5, 24.25}});
  assert(entered.accepted() && entered.inputAccepted && entered.requestFrame);
  assert(entered.acceptedInput.kind == ViewerSessionInputKind::PointerEntered);
  assert(state.pointerPresent && state.pointerX == 12.5 &&
         state.pointerY == 24.25);
  const uint64_t revisionAfterEnter = state.inputRevision;

  const auto moved = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionPointerMoved{100.0, 80.0}});
  assert(moved.accepted() && moved.acceptedInput.kind ==
                                 ViewerSessionInputKind::PointerMoved);
  assert(state.pointerPresent && state.pointerX == 100.0 &&
         state.pointerY == 80.0);
  assert(state.inputRevision == revisionAfterEnter + 1);

  const auto buttonDown = ChromaspaceViewer::reduceViewerSession(
      &state, {4, ViewerSessionPointerButtonChanged{
                   ViewerSessionPointerButton::Primary, true, 100.0, 80.0,
                   ChromaspaceViewer::kViewerSessionModifierShift, 2}});
  assert(buttonDown.accepted() && buttonDown.inputAccepted);
  assert(buttonDown.acceptedInput.clickCount == 2u);
  assert((state.pressedPointerButtons &
          ChromaspaceViewer::viewerSessionPointerButtonBit(
              ViewerSessionPointerButton::Primary)) != 0);
  assert(state.modifiers == ChromaspaceViewer::kViewerSessionModifierShift);

  const ViewerSessionState invalidClickBaseline = state;
  const auto invalidClick = ChromaspaceViewer::reduceViewerSession(
      &state, {5, ViewerSessionPointerButtonChanged{
                   ViewerSessionPointerButton::Secondary, true, 100.0, 80.0,
                   ChromaspaceViewer::kViewerSessionModifierShift, 0}});
  assert(invalidClick.status ==
         ViewerSessionReduceStatus::RejectedInvalidInputTransition);
  assertSameState(state, invalidClickBaseline);

  const auto keyDown = ChromaspaceViewer::reduceViewerSession(
      &state, {5, ViewerSessionKeyChanged{
                   ViewerSessionKey::A, true, false,
                   ChromaspaceViewer::kViewerSessionModifierShift}});
  assert(keyDown.accepted());
  const auto keyRepeat = ChromaspaceViewer::reduceViewerSession(
      &state, {6, ViewerSessionKeyChanged{
                   ViewerSessionKey::A, true, true,
                   ChromaspaceViewer::kViewerSessionModifierShift}});
  assert(keyRepeat.accepted() && keyRepeat.acceptedInput.repeat);
  const auto keyUp = ChromaspaceViewer::reduceViewerSession(
      &state, {7, ViewerSessionKeyChanged{
                   ViewerSessionKey::A, false, false,
                   ChromaspaceViewer::kViewerSessionModifierShift}});
  assert(keyUp.accepted());
  assert((state.pressedKeys & ChromaspaceViewer::viewerSessionKeyBit(
                                  ViewerSessionKey::A)) == 0);

  const auto text = ChromaspaceViewer::reduceViewerSession(
      &state, {8, ViewerSessionTextInput{U'\U0001F600'}});
  assert(text.accepted() && text.acceptedInput.codepoint == U'\U0001F600');
  const auto scroll = ChromaspaceViewer::reduceViewerSession(
      &state, {9, ViewerSessionScroll{1.25, -2.5,
                                      ChromaspaceViewer::kViewerSessionModifierControl}});
  assert(scroll.accepted() && state.modifiers ==
                                 ChromaspaceViewer::kViewerSessionModifierControl);

  const auto gestureBegin = ChromaspaceViewer::reduceViewerSession(
      &state, {10, ViewerSessionGesture{ViewerSessionGestureKind::Magnify,
                                         ViewerSessionGesturePhase::Begin,
                                         0.25, 0}});
  const auto gestureUpdate = ChromaspaceViewer::reduceViewerSession(
      &state, {11, ViewerSessionGesture{ViewerSessionGestureKind::Magnify,
                                         ViewerSessionGesturePhase::Update,
                                         0.10, 0}});
  const auto gestureEnd = ChromaspaceViewer::reduceViewerSession(
      &state, {12, ViewerSessionGesture{ViewerSessionGestureKind::Magnify,
                                         ViewerSessionGesturePhase::End,
                                         0.05, 0}});
  assert(gestureBegin.accepted() && gestureUpdate.accepted() &&
         gestureEnd.accepted());
  assert(state.activeGestures == 0);

  // Re-establish every active source so cancellation has observable work and
  // cannot leave lastInput pointing at a stale transient.
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, {13, ViewerSessionPointerButtonChanged{
                   ViewerSessionPointerButton::Secondary, true, 40.0, 50.0, 0}});
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, {14, ViewerSessionKeyChanged{ViewerSessionKey::B, true, false, 0}});
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, {15, ViewerSessionModifiersChanged{
                   ChromaspaceViewer::kViewerSessionModifierAlt}});
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, {16, ViewerSessionGesture{ViewerSessionGestureKind::Rotate,
                                         ViewerSessionGesturePhase::Begin,
                                         0.2, 0}});
  const auto focusLost = ChromaspaceViewer::reduceViewerSession(
      &state, {17, ViewerSessionFocusChanged{false}});
  assert(focusLost.accepted() && focusLost.cancelInteractions &&
         focusLost.inputAccepted);
  assert(focusLost.acceptedInput.kind == ViewerSessionInputKind::Cancelled);
  assert(state.lastInput.kind == ViewerSessionInputKind::Cancelled);
  assert(state.lastInput.sequence == 17);
  assert(state.pressedKeys == 0 && state.pressedPointerButtons == 0 &&
         state.modifiers == 0 && state.activeGestures == 0 &&
         !state.pointerPresent);
  assert(state.inputRevision > 0 && focusLost.requestFrame);

  const ViewerSessionState cancelled = state;
  const auto stale = ChromaspaceViewer::reduceViewerSession(
      &state, {17, ViewerSessionPointerMoved{1.0, 2.0}});
  assert(stale.status == ViewerSessionReduceStatus::RejectedStaleSequence);
  assertSameState(state, cancelled);
}

void testInvalidInputTransitionsAreImmutable() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(&state, initializeEvent(1));
  const ViewerSessionState baseline = state;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto invalidPointer = ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionPointerMoved{nan, 1.0}});
  assert(invalidPointer.status == ViewerSessionReduceStatus::RejectedInvalidPointer);
  assertSameState(state, baseline);
  const auto invalidButton = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionPointerButtonChanged{
                   static_cast<ViewerSessionPointerButton>(255), true, 1.0, 1.0, 0}});
  assert(invalidButton.status == ViewerSessionReduceStatus::RejectedInvalidButton);
  assertSameState(state, baseline);
  const auto invalidModifiers = ChromaspaceViewer::reduceViewerSession(
      &state, {4, ViewerSessionModifiersChanged{0x80u}});
  assert(invalidModifiers.status == ViewerSessionReduceStatus::RejectedInvalidModifiers);
  assertSameState(state, baseline);
  const auto invalidKey = ChromaspaceViewer::reduceViewerSession(
      &state, {5, ViewerSessionKeyChanged{ViewerSessionKey::Unknown, true, false, 0}});
  assert(invalidKey.status == ViewerSessionReduceStatus::RejectedInvalidKey);
  assertSameState(state, baseline);
  const auto invalidText = ChromaspaceViewer::reduceViewerSession(
      &state, {6, ViewerSessionTextInput{static_cast<char32_t>(0xD800)}});
  assert(invalidText.status == ViewerSessionReduceStatus::RejectedInvalidText);
  assertSameState(state, baseline);
  const auto invalidScroll = ChromaspaceViewer::reduceViewerSession(
      &state, {7, ViewerSessionScroll{inf, 0.0, 0}});
  assert(invalidScroll.status == ViewerSessionReduceStatus::RejectedInvalidScroll);
  assertSameState(state, baseline);
  const auto invalidGesture = ChromaspaceViewer::reduceViewerSession(
      &state, {8, ViewerSessionGesture{ViewerSessionGestureKind::Rotate,
                                         ViewerSessionGesturePhase::Update,
                                         0.1, 0}});
  assert(invalidGesture.status == ViewerSessionReduceStatus::RejectedInvalidInputTransition);
  assertSameState(state, baseline);
  const auto invalidGestureKind = ChromaspaceViewer::reduceViewerSession(
      &state, {9, ViewerSessionGesture{
                   static_cast<ViewerSessionGestureKind>(255),
                   ViewerSessionGesturePhase::Begin, 0.1, 0}});
  assert(invalidGestureKind.status == ViewerSessionReduceStatus::RejectedInvalidGesture);
  assertSameState(state, baseline);
  const auto invalidGesturePhase = ChromaspaceViewer::reduceViewerSession(
      &state, {10, ViewerSessionGesture{
                   ViewerSessionGestureKind::Magnify,
                   static_cast<ViewerSessionGesturePhase>(255), 0.1, 0}});
  assert(invalidGesturePhase.status == ViewerSessionReduceStatus::RejectedInvalidGesture);
  assertSameState(state, baseline);
  const auto duplicateRelease = ChromaspaceViewer::reduceViewerSession(
      &state, {11, ViewerSessionPointerButtonChanged{
                   ViewerSessionPointerButton::Primary, false, 1.0, 1.0, 0}});
  assert(duplicateRelease.status == ViewerSessionReduceStatus::RejectedInvalidInputTransition);
  assertSameState(state, baseline);
}

void testRetinaLogicalCoordinatesAreIndependentOfFramebufferScale() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(1, viewport(640, 360, 1280, 720, 2.0f, 2.0f)));
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionPointerMoved{123.5, 87.25}});
  const double logicalX = state.pointerX;
  const double logicalY = state.pointerY;
  const uint64_t inputRevision = state.inputRevision;
  const auto resized = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionViewportChanged{
                   viewport(640, 360, 2560, 1440, 4.0f, 4.0f)}});
  assert(resized.accepted());
  assert(state.viewport.logicalWidth == 640 && state.viewport.logicalHeight == 360);
  assert(state.viewport.framebufferWidth == 2560 && state.viewport.framebufferHeight == 1440);
  assert(state.pointerX == logicalX && state.pointerY == logicalY);
  assert(state.inputRevision == inputRevision);
}

void testCloseIsTerminal() {
  ViewerSessionState state{};
  (void)ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(1));
  const auto closed = ChromaspaceViewer::reduceViewerSession(
      &state, {2, ViewerSessionCloseRequested{}});
  assert(closed.status == ViewerSessionReduceStatus::Applied);
  assert(closed.changed && closed.cancelInteractions);
  assert(!closed.shouldRender && closed.shouldClose);
  assert(ChromaspaceViewer::viewerSessionShouldClose(state));
  assert(!ChromaspaceViewer::viewerSessionShouldRender(state));
  const ViewerSessionState terminal = state;

  const auto attemptedRecovery = ChromaspaceViewer::reduceViewerSession(
      &state, {3, ViewerSessionVisibilityChanged{true}});
  assert(attemptedRecovery.status ==
         ViewerSessionReduceStatus::RejectedClosed);
  assertSameState(state, terminal);
}

void testInitializationAndSequenceFailurePolicies() {
  ViewerSessionState state{};
  const auto premature = ChromaspaceViewer::reduceViewerSession(
      &state, {1, ViewerSessionFocusChanged{true}});
  assert(premature.status ==
         ViewerSessionReduceStatus::RejectedNotInitialized);
  assert(state.lastAcceptedSequence == 0);

  ViewerSessionViewport invalid = viewport();
  invalid.logicalHeight = -1;
  const auto invalidInitialization = ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(2, invalid));
  assert(invalidInitialization.status ==
         ViewerSessionReduceStatus::RejectedInvalidViewport);
  assert(!state.initialized && state.lastAcceptedSequence == 0);

  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  const auto maximumInitialization = ChromaspaceViewer::reduceViewerSession(
      &state, initializeEvent(maximum));
  assert(maximumInitialization.status == ViewerSessionReduceStatus::Applied);
  const ViewerSessionState exhausted = state;
  assert(ChromaspaceViewer::viewerSessionNextSequence(state) == maximum);
  const auto overflow = ChromaspaceViewer::reduceViewerSession(
      &state, {maximum, ViewerSessionFocusChanged{false}});
  assert(overflow.status ==
         ViewerSessionReduceStatus::RejectedSequenceExhausted);
  assertSameState(state, exhausted);
}

void testDeterministicRevisions() {
  ViewerSessionState first{};
  ViewerSessionState second{};
  const ViewerSessionEvent events[] = {
      initializeEvent(1),
      {2, ViewerSessionFocusChanged{false}},
      {3, ViewerSessionVisibilityChanged{false}},
      {4, ViewerSessionVisibilityChanged{true}},
      {5, ViewerSessionMiniaturizationChanged{true}},
      {6, ViewerSessionMiniaturizationChanged{false}},
      {7, ViewerSessionViewportChanged{
              viewport(700, 400, 1400, 900, 2.0f, 2.25f)}},
  };
  for (const auto& event : events) {
    const auto a = ChromaspaceViewer::reduceViewerSession(&first, event);
    const auto b = ChromaspaceViewer::reduceViewerSession(&second, event);
    assert(a.status == b.status);
    assert(a.changed == b.changed);
  }
  assertSameState(first, second);
  assert(first.lifecycleRevision == 6);
  assert(first.viewportRevision == 2);
}

}  // namespace

int main() {
  testInitializeAndAtomicViewportRevision();
  testStaleDuplicateAndInvalidSnapshotsDoNotMutate();
  testLifecycleCancellationAndRecovery();
  testNormalizedInputStateAndCancellationMetadata();
  testInvalidInputTransitionsAreImmutable();
  testRetinaLogicalCoordinatesAreIndependentOfFramebufferScale();
  testCloseIsTerminal();
  testInitializationAndSequenceFailurePolicies();
  testDeterministicRevisions();
  return 0;
}
