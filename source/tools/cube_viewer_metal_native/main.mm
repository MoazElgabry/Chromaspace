#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "metal/ChromaspaceMetal.h"
#include "metal/ChromaspaceMetalFrameExecutor.h"
#include "metal/ChromaspaceMetalPlotRenderer.h"
#include "metal/ChromaspaceMetalPlotCompiler.h"
#include "metal/ChromaspaceMetalWorkspaceFrame.h"
#include "metal/ChromaspaceMetalViewerRuntime.h"
#include "metal/ChromaspaceMetalMemoryPressureMailbox.h"
#include "metal/ChromaspaceResidentSourceSession.h"
#include "ChromaspaceViewerCommandServer.h"
#include "ChromaspaceViewerCameraInteraction.h"
#include "ChromaspaceViewerLassoInteraction.h"
#include "ChromaspaceViewerFrameCoordinator.h"
#include "ChromaspaceViewerFramePlan.h"
#include "ChromaspaceViewerUiProjection.h"
#include "ChromaspaceViewerTextLayout.h"
#include "ChromaspaceViewerController.h"
#include "ChromaspaceViewerWorkspace.h"
#include "ChromaspaceViewerWorkspacePersistence.h"
#include "ChromaspaceViewerWorkspaceStore.h"
#include "ChromaspaceViewerSession.h"
#include "ChromaspaceViewerState.h"
#include "ChromaspaceViewerUiScene.h"

#if defined(CHROMASPACE_METAL_QUALIFICATION)
#include "metal/ChromaspaceMetalQualificationCampaign.h"
#include "metal/ChromaspaceMetalQualificationFaultBackend.h"
#include "metal/ChromaspaceMetalQualificationNativeSourceBackend.h"
#include "metal/ChromaspaceMetalQualificationSourceAdapter.h"
#include "metal/ChromaspaceMetalQualificationWorkspace.h"
#endif

struct CocoaViewerInteractionContext {
  ChromaspaceViewer::ViewerController* controller = nullptr;
  ChromaspaceViewer::ViewerWorkspaceState* workspace = nullptr;
  ChromaspaceViewer::ViewerSessionState* session = nullptr;
  ChromaspaceViewer::ViewerCameraInteractionState* camera = nullptr;
  ChromaspaceViewer::ViewerLassoInteractionState* lasso = nullptr;
  int sourceWidth = 0;
  int sourceHeight = 0;
};

void consumeCocoaViewerSessionResult(
    CocoaViewerInteractionContext* interaction,
    const ChromaspaceViewer::ViewerSessionReduceResult& result);

namespace {

// Normal operation is unbounded. Automated qualification can opt into a
// deterministic positive budget through --frames or the environment.
constexpr int kDefaultCanaryFrames = 0;
constexpr int kMaximumCanaryFrames = 10000;

// Dispatch notifications are never allowed to touch render-thread-affine GPU
// state. The handler publishes only into a shared packed-atomic mailbox; the
// viewer loop consumes it between frame transactions. The block captures no
// monitor/runtime pointer, so cancellation and late delivery cannot call into
// destroyed stack state.
class CocoaMemoryPressureMonitor final {
 public:
  CocoaMemoryPressureMonitor() noexcept = default;
  ~CocoaMemoryPressureMonitor() { stop(); }

  CocoaMemoryPressureMonitor(const CocoaMemoryPressureMonitor&) = delete;
  CocoaMemoryPressureMonitor& operator=(const CocoaMemoryPressureMonitor&) =
      delete;

  bool start(std::string* error) {
    if (error != nullptr) error->clear();
    if (source_ != nullptr || mailbox_ != nullptr) {
      if (error != nullptr) *error = "memory-pressure-monitor-already-started";
      return false;
    }
    try {
      mailbox_ =
          std::make_shared<ChromaspaceMetalMemoryPressure::Mailbox>();
    } catch (...) {
      if (error != nullptr) *error = "memory-pressure-mailbox-allocation-failed";
      return false;
    }

    const unsigned long mask = DISPATCH_MEMORYPRESSURE_NORMAL |
                               DISPATCH_MEMORYPRESSURE_WARN |
                               DISPATCH_MEMORYPRESSURE_CRITICAL;
    source_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0u,
                                     mask, dispatch_get_main_queue());
    if (source_ == nullptr) {
      mailbox_.reset();
      if (error != nullptr) *error = "memory-pressure-monitor-create-failed";
      return false;
    }

    const auto mailbox = mailbox_;
    __weak dispatch_source_t weakSource = source_;
    dispatch_source_set_event_handler(source_, ^{
      dispatch_source_t source = weakSource;
      if (source == nullptr) return;
      const unsigned long data = dispatch_source_get_data(source);
      if ((data & DISPATCH_MEMORYPRESSURE_NORMAL) != 0u) {
        (void)mailbox->publish(
            ChromaspaceMetalMemoryPressure::Signal::Normal);
      }
      if ((data & DISPATCH_MEMORYPRESSURE_WARN) != 0u) {
        (void)mailbox->publish(
            ChromaspaceMetalMemoryPressure::Signal::Warning);
      }
      if ((data & DISPATCH_MEMORYPRESSURE_CRITICAL) != 0u) {
        (void)mailbox->publish(
            ChromaspaceMetalMemoryPressure::Signal::Critical);
      }
    });
    dispatch_resume(source_);
    return true;
  }

  ChromaspaceMetalMemoryPressure::Batch consume() noexcept {
    return mailbox_ != nullptr ? mailbox_->consume()
                               : ChromaspaceMetalMemoryPressure::Batch{};
  }

  void stop() noexcept {
    if (source_ != nullptr) {
      dispatch_source_cancel(source_);
      dispatch_source_set_event_handler(source_, nil);
      source_ = nullptr;
    }
    mailbox_.reset();
  }

 private:
  dispatch_source_t source_ = nullptr;
  std::shared_ptr<ChromaspaceMetalMemoryPressure::Mailbox> mailbox_;
};

int64_t monotonicMilliseconds() noexcept {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::string heartbeatJsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (static_cast<unsigned char>(character) >= 0x20u) {
          escaped += character;
        }
        break;
    }
  }
  return escaped;
}

struct CocoaHeartbeatState {
  std::mutex mutex;
  bool visible = false;
  bool iconified = false;
  bool focused = false;
  int64_t updatedMilliseconds = 0;
  ChromaspaceResidentSource::ResidentSourceSnapshot resident{};
};

ChromaspaceViewer::ViewerWorkspaceStorePaths
cocoaViewerWorkspaceStorePaths() noexcept {
  try {
    NSArray<NSString*>* directories =
        NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory,
                                            NSUserDomainMask, YES);
    NSString* supportDirectory = directories.count > 0u ? directories[0] : nil;
    if (supportDirectory == nil) return {};
    NSString* chromaspaceDirectory =
        [supportDirectory stringByAppendingPathComponent:@"Chromaspace"];
    NSString* primary =
        [chromaspaceDirectory stringByAppendingPathComponent:
                                  @"viewer_workspace_v2.jsonl"];
    NSString* legacy =
        [chromaspaceDirectory stringByAppendingPathComponent:
                                  @"viewer_workspace_v1.jsonl"];
    const char* primaryBytes = [primary fileSystemRepresentation];
    const char* legacyBytes = [legacy fileSystemRepresentation];
    if (primaryBytes == nullptr || legacyBytes == nullptr) return {};
    return {std::filesystem::path(primaryBytes),
            std::filesystem::path(legacyBytes)};
  } catch (...) {
    return {};
  }
}

bool cocoaViewerPresentationEqual(
    const ChromaspaceViewer::ViewerWorkspacePresentationPreferences& left,
    const ChromaspaceViewer::ViewerWorkspacePresentationPreferences& right) noexcept {
  return left.showWorkspaceButtons == right.showWorkspaceButtons &&
         left.showSliceButtonInPlotWindows == right.showSliceButtonInPlotWindows &&
         left.viewerFontSize == right.viewerFontSize &&
         left.windowWidth == right.windowWidth &&
         left.windowHeight == right.windowHeight &&
         left.windowPosX == right.windowPosX && left.windowPosY == right.windowPosY &&
         left.windowPositionValid == right.windowPositionValid &&
         left.activeStandardLayoutIndex == right.activeStandardLayoutIndex &&
         left.workspaceTopNorm == right.workspaceTopNorm;
}

bool cocoaPrimaryScreenHeight(CGFloat* height) noexcept {
  if (height == nullptr) return false;
  @try {
    NSArray<NSScreen*>* screens = [NSScreen screens];
    NSScreen* primary = screens.count > 0u ? screens[0] : [NSScreen mainScreen];
    if (primary == nil) return false;
    const CGFloat value = [primary frame].size.height;
    if (!std::isfinite(static_cast<double>(value)) || value <= 0.0) {
      return false;
    }
    *height = value;
    return true;
  } @catch (NSException* exception) {
    (void)exception;
    return false;
  }
}

bool captureCocoaViewerPresentation(
    NSWindow* window,
    ChromaspaceViewer::ViewerWorkspacePresentationPreferences* presentation) noexcept {
  if (window == nil || presentation == nullptr || [window contentView] == nil) {
    return false;
  }
  try {
    const NSRect bounds = [[window contentView] bounds];
    const double width = bounds.size.width;
    const double height = bounds.size.height;
    if (std::isfinite(width) && std::isfinite(height) && width > 0.0 &&
        height > 0.0) {
      presentation->windowWidth = std::max(
          1, std::min(16384, static_cast<int>(std::lround(width))));
      presentation->windowHeight = std::max(
          1, std::min(16384, static_cast<int>(std::lround(height))));
    }
    const NSRect contentRect = [window contentRectForFrameRect:[window frame]];
    CGFloat primaryScreenHeight = 0.0;
    const double x = contentRect.origin.x;
    // The persisted contract predates the Cocoa adapter and follows the
    // legacy top-left content-coordinate convention. Preserve it so a
    // workspace migrates without moving vertically at the native cutover.
    const double y = cocoaPrimaryScreenHeight(&primaryScreenHeight)
                         ? primaryScreenHeight - contentRect.origin.y -
                               contentRect.size.height
                         : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(x) && std::isfinite(y) &&
        x >= static_cast<double>(std::numeric_limits<int>::min()) &&
        x <= static_cast<double>(std::numeric_limits<int>::max()) &&
        y >= static_cast<double>(std::numeric_limits<int>::min()) &&
        y <= static_cast<double>(std::numeric_limits<int>::max())) {
      presentation->windowPosX = static_cast<int>(std::lround(x));
      presentation->windowPosY = static_cast<int>(std::lround(y));
      presentation->windowPositionValid = true;
    } else {
      presentation->windowPositionValid = false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

struct CocoaWorkspacePersistenceTracker {
  ChromaspaceViewer::ViewerWorkspaceStorePaths paths{};
  std::string baselineBytes;
  std::string pendingBytes;
  std::string lastDiagnostic;
  ChromaspaceViewer::ViewerWorkspaceStoreLoadStatus loadStatus =
      ChromaspaceViewer::ViewerWorkspaceStoreLoadStatus::DefaultsMissing;
  bool dirty = false;
  bool repairPending = false;
  bool observed = false;
  uint64_t lastWorkspaceRevision = 0u;
  ChromaspaceViewer::ViewerWorkspacePresentationPreferences lastPresentation{};
  int64_t dirtySinceMilliseconds = -1;
  int64_t nextAttemptMilliseconds = 0;
};

void markCocoaWorkspaceDirty(CocoaWorkspacePersistenceTracker* tracker,
                             int64_t nowMilliseconds) noexcept {
  if (!tracker) return;
  tracker->dirty = true;
  if (tracker->dirtySinceMilliseconds < 0) {
    tracker->dirtySinceMilliseconds = nowMilliseconds;
  }
}

bool observeCocoaWorkspacePersistence(
    NSWindow* window,
    ChromaspaceViewer::ViewerWorkspaceState* workspace,
    ChromaspaceViewer::ViewerWorkspaceDocument* document,
    CocoaWorkspacePersistenceTracker* tracker,
    bool initial) noexcept {
  if (window == nil || workspace == nullptr || document == nullptr ||
      tracker == nullptr) {
    return false;
  }
  try {
    document->workspace = *workspace;
    (void)captureCocoaViewerPresentation(window, &document->presentation);
    const bool probeChanged =
        !tracker->observed || tracker->lastWorkspaceRevision != workspace->revision ||
        !cocoaViewerPresentationEqual(tracker->lastPresentation,
                                       document->presentation);
    if (!probeChanged) return false;
    const auto encoded = ChromaspaceViewer::encodeViewerWorkspaceV2(*document);
    if (!encoded.accepted()) {
      tracker->lastDiagnostic = "could not encode current workspace for autosave";
      tracker->lastWorkspaceRevision = workspace->revision;
      tracker->lastPresentation = document->presentation;
      tracker->observed = true;
      return false;
    }
    tracker->pendingBytes = encoded.bytes;
    if (!tracker->observed || initial) {
      // Cocoa's style mask can make the first content size differ by a few
      // pixels from the persisted request.  Treat that as this launch's
      // baseline; backup/v1 recovery remains repair-eligible.
      tracker->baselineBytes = encoded.bytes;
      if (tracker->repairPending) {
        markCocoaWorkspaceDirty(tracker, monotonicMilliseconds());
      } else {
        tracker->dirty = false;
        tracker->dirtySinceMilliseconds = -1;
      }
    } else if (encoded.bytes != tracker->baselineBytes) {
      markCocoaWorkspaceDirty(tracker, monotonicMilliseconds());
    }
    tracker->lastWorkspaceRevision = workspace->revision;
    tracker->lastPresentation = document->presentation;
    tracker->observed = true;
    return true;
  } catch (...) {
    tracker->lastDiagnostic = "workspace autosave observation failed";
    return false;
  }
}

bool maybeAutosaveCocoaWorkspace(
    CocoaWorkspacePersistenceTracker* tracker,
    const ChromaspaceViewer::ViewerWorkspaceDocument& document,
    bool force) noexcept {
  if (tracker == nullptr || !tracker->dirty) return true;
  const int64_t now = monotonicMilliseconds();
  if (!force &&
      (tracker->dirtySinceMilliseconds < 0 ||
       now - tracker->dirtySinceMilliseconds < 1500 ||
       now < tracker->nextAttemptMilliseconds)) {
    return true;
  }
  const auto save = ChromaspaceViewer::saveViewerWorkspaceStore(
      tracker->paths, document);
  if (save.accepted()) {
    tracker->baselineBytes = tracker->pendingBytes;
    tracker->dirty = false;
    tracker->repairPending = false;
    tracker->dirtySinceMilliseconds = -1;
    tracker->nextAttemptMilliseconds = 0;
    tracker->lastDiagnostic.clear();
    return true;
  }
  tracker->lastDiagnostic = save.diagnostic;
  tracker->nextAttemptMilliseconds = now + 1500;
  return false;
}

const char* residentHeartbeatStatus(
    ChromaspaceResidentSource::SessionHealth health) noexcept {
  using ChromaspaceResidentSource::SessionHealth;
  switch (health) {
    case SessionHealth::Ready: return "Ready";
    case SessionHealth::Starting: return "Waiting for source";
    case SessionHealth::Unavailable: return "Source exchange unavailable";
    case SessionHealth::Failed: return "Source exchange failed";
    case SessionHealth::Stopped:
    default: return "Waiting for Resolve";
  }
}

bool cocoaHeartbeatAck(
    const ChromaspaceViewer::ViewerCommandServerHeartbeatRequest& request,
    const ChromaspaceViewer::ViewerCommandServerSnapshot& server,
    std::string* response,
    void* context) {
  (void)server;
  if (!response || !context) return false;
  try {
    CocoaHeartbeatState snapshot{};
    auto* state = static_cast<CocoaHeartbeatState*>(context);
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      snapshot.visible = state->visible;
      snapshot.iconified = state->iconified;
      snapshot.focused = state->focused;
      snapshot.updatedMilliseconds = state->updatedMilliseconds;
      snapshot.resident = state->resident;
    }
    const int64_t age = snapshot.updatedMilliseconds > 0
                            ? std::max<int64_t>(
                                  0, monotonicMilliseconds() -
                                         snapshot.updatedMilliseconds)
                            : -1;
    std::ostringstream json;
    json << "{\"type\":\"heartbeat_ack\",\"seq\":" << request.seq
         << ",\"visible\":" << (snapshot.visible ? 1 : 0)
         << ",\"iconified\":" << (snapshot.iconified ? 1 : 0)
         << ",\"focused\":" << (snapshot.focused ? 1 : 0)
         << ",\"sourceRasterStatus\":\""
         << residentHeartbeatStatus(snapshot.resident.health)
         << "\",\"sourceRasterReason\":\""
         << heartbeatJsonEscape(snapshot.resident.diagnostic)
         << "\",\"sourceRasterSenderId\":\""
         << heartbeatJsonEscape(snapshot.resident.senderId)
         << "\",\"sourceRasterTransport\":\"source-exchange-v2\""
         << ",\"sourceRasterSeq\":"
         << snapshot.resident.lastObservedSequence
         << ",\"sourceRasterImported\":"
         << (snapshot.resident.hasActiveSource ? 1 : 0)
         << ",\"sourceRasterSurfaceId\":0"
         << ",\"sourceRasterAgeMs\":" << age << "}";
    *response = json.str();
    return true;
  } catch (...) {
    return false;
  }
}

void reportError(const char* operation, const std::string& error) {
  std::cerr << "Chromaspace Metal viewer " << operation;
  if (!error.empty()) std::cerr << ": " << error;
  std::cerr << '\n';
}

void pumpApplicationEvents(NSApplication* application,
                           NSTimeInterval initialWaitSeconds = 0.0) {
  if (application == nil) return;
  bool firstEvent = true;
  for (;;) {
    NSDate* deadline = firstEvent && initialWaitSeconds > 0.0
                           ? [NSDate dateWithTimeIntervalSinceNow:
                                         initialWaitSeconds]
                           : [NSDate distantPast];
    NSEvent* event = [application nextEventMatchingMask:NSEventMaskAny
                                              untilDate:deadline
                                                 inMode:NSDefaultRunLoopMode
                                                dequeue:YES];
    if (event == nil) break;
    firstEvent = false;
    [application sendEvent:event];
    [application updateWindows];
  }
}

void pumpRecoveryEvents(NSApplication* application, uint32_t waitMilliseconds) {
  const NSTimeInterval seconds = waitMilliseconds > 0u
                                     ? static_cast<NSTimeInterval>(waitMilliseconds) /
                                           1000.0
                                     : (1.0 / 60.0);
  pumpApplicationEvents(application, seconds);
}

bool cocoaViewerSessionVisible(NSWindow* window) {
  return window != nil && [window isVisible] &&
         (([window occlusionState] & NSWindowOcclusionStateVisible) != 0);
}

ChromaspaceFrameRecoveryPolicy::SurfaceVisibility cocoaViewerSurfaceVisibility(
    NSWindow* window) {
  if (window == nil || ![window isVisible]) {
    return ChromaspaceFrameRecoveryPolicy::SurfaceVisibility::Unavailable;
  }
  if (([window occlusionState] & NSWindowOcclusionStateVisible) == 0) {
    return ChromaspaceFrameRecoveryPolicy::SurfaceVisibility::Occluded;
  }
  return ChromaspaceFrameRecoveryPolicy::SurfaceVisibility::Visible;
}

bool cocoaViewerSessionViewport(
    NSWindow* window,
    ChromaspaceViewer::ViewerSessionViewport* viewport) {
  if (window == nil || viewport == nullptr || [window contentView] == nil) {
    return false;
  }
  NSView* contentView = [window contentView];
  const NSRect bounds = [contentView bounds];
  const NSRect backing = [contentView convertRectToBacking:bounds];
  const CGFloat fallbackScale = [window backingScaleFactor];
  viewport->logicalWidth =
      static_cast<int>(std::lround(bounds.size.width));
  viewport->logicalHeight =
      static_cast<int>(std::lround(bounds.size.height));
  viewport->framebufferWidth =
      static_cast<int>(std::lround(backing.size.width));
  viewport->framebufferHeight =
      static_cast<int>(std::lround(backing.size.height));
  viewport->contentScaleX = static_cast<float>(
      bounds.size.width > 0.0
          ? backing.size.width / bounds.size.width
          : fallbackScale);
  viewport->contentScaleY = static_cast<float>(
      bounds.size.height > 0.0
          ? backing.size.height / bounds.size.height
          : fallbackScale);
  return true;
}

template <typename Payload>
ChromaspaceViewer::ViewerSessionReduceResult applyCocoaViewerSessionEvent(
    ChromaspaceViewer::ViewerSessionState* session,
    Payload payload) {
  ChromaspaceViewer::ViewerSessionEvent event{};
  event.sequence = ChromaspaceViewer::viewerSessionNextSequence(*session);
  event.payload = std::move(payload);
  return ChromaspaceViewer::reduceViewerSession(session, event);
}

ChromaspaceViewer::ViewerSessionReduceResult
updateCocoaViewerSessionViewport(
    NSWindow* window,
    ChromaspaceViewer::ViewerSessionState* session) {
  ChromaspaceViewer::ViewerSessionViewport viewport{};
  if (!cocoaViewerSessionViewport(window, &viewport)) {
    ChromaspaceViewer::ViewerSessionReduceResult result{};
    result.status =
        ChromaspaceViewer::ViewerSessionReduceStatus::RejectedInvalidViewport;
    return result;
  }
  return applyCocoaViewerSessionEvent(
      session, ChromaspaceViewer::ViewerSessionViewportChanged{viewport});
}

ChromaspaceViewer::ViewerSessionModifierMask cocoaViewerSessionModifiers(
    NSEventModifierFlags flags) {
  using namespace ChromaspaceViewer;
  ViewerSessionModifierMask modifiers = 0;
  if ((flags & NSEventModifierFlagShift) != 0) {
    modifiers |= kViewerSessionModifierShift;
  }
  if ((flags & NSEventModifierFlagControl) != 0) {
    modifiers |= kViewerSessionModifierControl;
  }
  if ((flags & NSEventModifierFlagOption) != 0) {
    modifiers |= kViewerSessionModifierAlt;
  }
  if ((flags & NSEventModifierFlagCommand) != 0) {
    modifiers |= kViewerSessionModifierSuper;
  }
  return modifiers;
}

NSPoint cocoaViewerSessionLogicalPoint(NSView* view, NSEvent* event) {
  if (view == nil || event == nil) return NSMakePoint(0.0, 0.0);
  NSPoint point = [view convertPoint:[event locationInWindow] fromView:nil];
  if (![view isFlipped]) {
    point.y = NSHeight([view bounds]) - point.y;
  }
  return point;
}

ChromaspaceViewer::ViewerSessionKey cocoaViewerSessionKey(NSEvent* event) {
  using namespace ChromaspaceViewer;
  if (event == nil) return ViewerSessionKey::Unknown;
  switch ([event keyCode]) {
    case 0: return ViewerSessionKey::A;
    case 11: return ViewerSessionKey::B;
    case 8: return ViewerSessionKey::C;
    case 2: return ViewerSessionKey::D;
    case 3: return ViewerSessionKey::F;
    case 37: return ViewerSessionKey::L;
    case 46: return ViewerSessionKey::M;
    case 15: return ViewerSessionKey::R;
    case 1: return ViewerSessionKey::S;
    case 9: return ViewerSessionKey::V;
    case 6: return ViewerSessionKey::Z;
    case 51: return ViewerSessionKey::Backspace;
    case 36:
    case 76: return ViewerSessionKey::Enter;
    case 53: return ViewerSessionKey::Escape;
    case 48: return ViewerSessionKey::Tab;
    default: return ViewerSessionKey::Unknown;
  }
}

bool cocoaViewerSessionButton(
    NSEvent* event,
    ChromaspaceViewer::ViewerSessionPointerButton* button,
    bool rightButton = false) {
  if (!event || !button) return false;
  if (rightButton) {
    *button = ChromaspaceViewer::ViewerSessionPointerButton::Secondary;
    return true;
  }
  const NSInteger number = [event buttonNumber];
  if (number < 0 || number >= static_cast<NSInteger>(
                                  ChromaspaceViewer::ViewerSessionPointerButton::Count)) {
    return false;
  }
  *button = static_cast<ChromaspaceViewer::ViewerSessionPointerButton>(number);
  return true;
}

ChromaspaceViewer::ViewerSessionGesturePhase cocoaViewerSessionGesturePhase(
    NSEventPhase phase) {
  using namespace ChromaspaceViewer;
  if ((phase & NSEventPhaseBegan) != 0) return ViewerSessionGesturePhase::Begin;
  if ((phase & NSEventPhaseChanged) != 0) return ViewerSessionGesturePhase::Update;
  if ((phase & NSEventPhaseEnded) != 0) return ViewerSessionGesturePhase::End;
  if ((phase & NSEventPhaseCancelled) != 0) return ViewerSessionGesturePhase::Cancel;
  return ViewerSessionGesturePhase::Count;
}

void cocoaViewerSessionReduceText(
    CocoaViewerInteractionContext* interaction,
    NSString* text) {
  if (!interaction || !interaction->session || !text) return;
  const NSUInteger length = [text length];
  for (NSUInteger index = 0; index < length; ++index) {
    const unichar first = [text characterAtIndex:index];
    char32_t scalar = static_cast<char32_t>(first);
    if (first >= 0xD800u && first <= 0xDBFFu) {
      if (index + 1u < length) {
        const unichar second = [text characterAtIndex:index + 1u];
        if (second >= 0xDC00u && second <= 0xDFFFu) {
          scalar = static_cast<char32_t>(0x10000u +
                                          ((first - 0xD800u) << 10u) +
                                          (second - 0xDC00u));
          ++index;
        }
      }
    }
    const auto result = applyCocoaViewerSessionEvent(
        interaction->session, ChromaspaceViewer::ViewerSessionTextInput{scalar});
    consumeCocoaViewerSessionResult(interaction, result);
  }
}

}  // namespace

void consumeCocoaViewerSessionResult(
    CocoaViewerInteractionContext* interaction,
    const ChromaspaceViewer::ViewerSessionReduceResult& result) {
  if (interaction == nullptr || interaction->controller == nullptr ||
      interaction->workspace == nullptr || interaction->session == nullptr ||
      interaction->camera == nullptr || interaction->lasso == nullptr) return;
  auto* controller = interaction->controller;
  auto* workspace = interaction->workspace;
  auto* session = interaction->session;
  auto* cameraInteraction = interaction->camera;

  const auto batch = controller->consume(result, *session);
  const auto workspaceResult =
      ChromaspaceViewer::reduceViewerWorkspace(workspace, batch);
  if (!workspaceResult.accepted()) {
    reportError("rejected a transactional viewer workspace input", {});
    controller->cancelInteractions();
    *cameraInteraction = {};
    *interaction->lasso = {};
    return;
  }
  for (std::size_t i = 0u; i < workspaceResult.effects.count; ++i) {
    const auto& effect = workspaceResult.effects[i];
    if (effect.kind !=
        ChromaspaceViewer::ViewerWorkspaceEffectKind::SlicingLassoChanged) {
      continue;
    }
    const auto sessionResult =
        ChromaspaceViewer::updateViewerWorkspaceSourceLassoSession(
            workspace, effect.windowId, effect.enabled);
    if (!sessionResult.accepted()) {
      reportError("rejected a Source Signal lasso session transition", {});
      controller->cancelInteractions();
      *cameraInteraction = {};
      *interaction->lasso = {};
      return;
    }
  }

  if (workspace->sourceLassoSessionActive ||
      interaction->lasso->pointerCaptureActive) {
    const auto lassoResult = ChromaspaceViewer::reduceViewerLassoInteraction(
        interaction->lasso,
        {&result, session, &controller->scene(), workspace,
         batch.continueSourceLasso, false,
         static_cast<double>(interaction->sourceWidth),
         static_cast<double>(interaction->sourceHeight)});
    const bool benignLassoNoop =
        lassoResult.status ==
            ChromaspaceViewer::ViewerLassoInteractionStatus::NotAuthorized ||
        lassoResult.status ==
            ChromaspaceViewer::ViewerLassoInteractionStatus::SessionInactive ||
        lassoResult.status ==
            ChromaspaceViewer::ViewerLassoInteractionStatus::WrongControl ||
        lassoResult.status ==
            ChromaspaceViewer::ViewerLassoInteractionStatus::CapacityExceeded;
    if (!lassoResult.accepted() && !benignLassoNoop) {
      reportError("rejected a portable Source Signal lasso interaction", {});
      controller->cancelInteractions();
      *interaction->lasso = {};
      *cameraInteraction = {};
      return;
    }
    if (lassoResult.strokeCompleted) {
      const auto commit =
          ChromaspaceViewer::appendViewerWorkspaceLassoStroke(
              workspace, lassoResult.stroke);
      if (!commit.accepted()) {
        reportError("rejected a transactional Source Signal lasso stroke", {});
        controller->cancelInteractions();
        *interaction->lasso = {};
        *cameraInteraction = {};
        return;
      }
    }
  }

  const auto cameraResult = ChromaspaceViewer::reduceViewerCameraInteraction(
      cameraInteraction,
      {&result, session, &controller->scene(), workspace,
       batch.continueCamera});
  if (!cameraResult.accepted()) {
    reportError("rejected a portable camera interaction", {});
    controller->cancelInteractions();
    *cameraInteraction = {};
    *interaction->lasso = {};
    return;
  }
  if (!cameraResult.cameraChanged) return;

  const auto commit = ChromaspaceViewer::updateViewerWorkspaceCamera(
      workspace, cameraResult.windowId, cameraResult.camera);
  if (!commit.accepted()) {
    reportError("rejected a transactional camera update", {});
    controller->cancelInteractions();
    *cameraInteraction = {};
    *interaction->lasso = {};
  }
}

@interface ChromaspaceCanaryInputView : NSView {
  @private
  CocoaViewerInteractionContext* _interaction;
  ChromaspaceViewer::ViewerSessionState* _session;
  NSTrackingArea* _trackingArea;
}
- (instancetype)initWithInteraction:(CocoaViewerInteractionContext*)interaction
                           frame:(NSRect)frame;
@end

@implementation ChromaspaceCanaryInputView

- (instancetype)initWithInteraction:(CocoaViewerInteractionContext*)interaction
                           frame:(NSRect)frame {
  self = [super initWithFrame:frame];
  if (self != nil) {
    _interaction = interaction;
    _session = interaction != nullptr ? interaction->session : nullptr;
    _trackingArea = nil;
    [self setPostsFrameChangedNotifications:YES];
  }
  return self;
}

- (BOOL)isFlipped {
  return YES;
}

- (BOOL)acceptsFirstResponder {
  return YES;
}

- (void)updateTrackingAreas {
  if (_trackingArea != nil) {
    [self removeTrackingArea:_trackingArea];
    _trackingArea = nil;
  }
  const NSTrackingAreaOptions options =
      NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved |
      NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect;
  _trackingArea = [[NSTrackingArea alloc] initWithRect:NSZeroRect
                                               options:options
                                                 owner:self
                                              userInfo:nil];
  [self addTrackingArea:_trackingArea];
  [super updateTrackingAreas];
}

- (void)mouseEntered:(NSEvent*)event {
  const NSPoint point = cocoaViewerSessionLogicalPoint(self, event);
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionPointerEntered{point.x, point.y});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)mouseExited:(NSEvent*)event {
  (void)event;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionPointerLeft{});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)mouseMoved:(NSEvent*)event {
  const NSPoint point = cocoaViewerSessionLogicalPoint(self, event);
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionPointerMoved{point.x, point.y});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)sendButtonEvent:(NSEvent*)event
                 button:(ChromaspaceViewer::ViewerSessionPointerButton)button
                pressed:(BOOL)pressed {
  const NSPoint point = cocoaViewerSessionLogicalPoint(self, event);
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionPointerButtonChanged{
          button, pressed == YES, point.x, point.y,
          cocoaViewerSessionModifiers([event modifierFlags]),
          static_cast<uint8_t>(std::min<NSInteger>(
              255, std::max<NSInteger>(1, [event clickCount]))) });
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)mouseDown:(NSEvent*)event {
  [self sendButtonEvent:event
                 button:ChromaspaceViewer::ViewerSessionPointerButton::Primary
                pressed:YES];
}

- (void)mouseUp:(NSEvent*)event {
  [self sendButtonEvent:event
                 button:ChromaspaceViewer::ViewerSessionPointerButton::Primary
                pressed:NO];
}

- (void)mouseDragged:(NSEvent*)event {
  [self mouseMoved:event];
}

- (void)rightMouseDown:(NSEvent*)event {
  [self sendButtonEvent:event
                 button:ChromaspaceViewer::ViewerSessionPointerButton::Secondary
                pressed:YES];
}

- (void)rightMouseUp:(NSEvent*)event {
  [self sendButtonEvent:event
                 button:ChromaspaceViewer::ViewerSessionPointerButton::Secondary
                pressed:NO];
}

- (void)rightMouseDragged:(NSEvent*)event {
  [self mouseMoved:event];
}

- (void)otherMouseDown:(NSEvent*)event {
  ChromaspaceViewer::ViewerSessionPointerButton button{};
  if (!cocoaViewerSessionButton(event, &button)) return;
  [self sendButtonEvent:event button:button pressed:YES];
}

- (void)otherMouseUp:(NSEvent*)event {
  ChromaspaceViewer::ViewerSessionPointerButton button{};
  if (!cocoaViewerSessionButton(event, &button)) return;
  [self sendButtonEvent:event button:button pressed:NO];
}

- (void)otherMouseDragged:(NSEvent*)event {
  [self mouseMoved:event];
}

- (void)flagsChanged:(NSEvent*)event {
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionModifiersChanged{
          cocoaViewerSessionModifiers([event modifierFlags])});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)keyDown:(NSEvent*)event {
  const ChromaspaceViewer::ViewerSessionKey key =
      cocoaViewerSessionKey(event);
  if (key != ChromaspaceViewer::ViewerSessionKey::Unknown) {
    const auto result = applyCocoaViewerSessionEvent(
        _session,
        ChromaspaceViewer::ViewerSessionKeyChanged{
            key, true, static_cast<bool>([event isARepeat]),
            cocoaViewerSessionModifiers([event modifierFlags])});
    if (!result.accepted()) return;
    consumeCocoaViewerSessionResult(_interaction, result);
  }
  [self interpretKeyEvents:@[event]];
}

- (void)keyUp:(NSEvent*)event {
  const ChromaspaceViewer::ViewerSessionKey key =
      cocoaViewerSessionKey(event);
  if (key == ChromaspaceViewer::ViewerSessionKey::Unknown) return;
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionKeyChanged{
          key, false, false,
          cocoaViewerSessionModifiers([event modifierFlags])});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)insertText:(id)insertString {
  NSString* text = nil;
  if ([insertString isKindOfClass:[NSAttributedString class]]) {
    text = [insertString string];
  } else if ([insertString isKindOfClass:[NSString class]]) {
    text = insertString;
  }
  cocoaViewerSessionReduceText(_interaction, text);
}

- (void)insertText:(id)insertString replacementRange:(NSRange)replacementRange {
  (void)replacementRange;
  [self insertText:insertString];
}

- (void)doCommandBySelector:(SEL)selector {
  (void)selector;
}

- (void)scrollWheel:(NSEvent*)event {
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionScroll{
          [event scrollingDeltaX], [event scrollingDeltaY],
          cocoaViewerSessionModifiers([event modifierFlags])});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)magnifyWithEvent:(NSEvent*)event {
  const ChromaspaceViewer::ViewerSessionGesturePhase phase =
      cocoaViewerSessionGesturePhase([event phase]);
  if (phase == ChromaspaceViewer::ViewerSessionGesturePhase::Count) return;
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionGesture{
          ChromaspaceViewer::ViewerSessionGestureKind::Magnify,
          phase,
          [event magnification],
          cocoaViewerSessionModifiers([event modifierFlags])});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)rotateWithEvent:(NSEvent*)event {
  const ChromaspaceViewer::ViewerSessionGesturePhase phase =
      cocoaViewerSessionGesturePhase([event phase]);
  if (phase == ChromaspaceViewer::ViewerSessionGesturePhase::Count) return;
  constexpr double kDegreesToRadians = 0.017453292519943295769;
  const double radians = [event rotation] * kDegreesToRadians;
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionGesture{
          ChromaspaceViewer::ViewerSessionGestureKind::Rotate,
          phase,
          radians,
          cocoaViewerSessionModifiers([event modifierFlags])});
  consumeCocoaViewerSessionResult(_interaction, result);
}

@end

namespace {

bool parseCanaryFrameCount(const char* text, int* frames) {
  if (text == nullptr || text[0] == '\0' || frames == nullptr) return false;
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed <= 0 ||
      parsed > kMaximumCanaryFrames || parsed > INT_MAX) {
    return false;
  }
  *frames = static_cast<int>(parsed);
  return true;
}

#if defined(CHROMASPACE_METAL_QUALIFICATION)

struct QualificationOptions final {
  int frameBudget = kDefaultCanaryFrames;
  ChromaspaceMetalQualification::Scenario scenario =
      ChromaspaceMetalQualification::Scenario::Steady;
  bool scenarioSpecified = false;
};

constexpr const char* kQualificationWorkspaceProfile =
    ChromaspaceMetalQualificationWorkspace::kProfileName;

struct QualificationPlotTelemetry final {
  uint64_t workspaceWindows = 0u;
  uint64_t samples = 0u;
  uint64_t surfaceResidentPeakBytes = 0u;
  uint64_t surfaceTransientPeakBytes = 0u;
  uint64_t derivedResidentPeakBytes = 0u;
  uint64_t derivedTransientPeakBytes = 0u;
  uint64_t contentHits = 0u;
  uint64_t derivedHits = 0u;
  uint64_t derivedCandidates = 0u;
  uint64_t derivedEvictions = 0u;
  uint64_t surfaceCreates = 0u;
  uint64_t surfaceResizes = 0u;
  uint64_t surfaceReplacements = 0u;
  uint64_t surfacePrunes = 0u;
  uint64_t presentedCpuSamples = 0u;
  double presentedCpuTotalMilliseconds = 0.0;
  double presentedCpuMaximumMilliseconds = 0.0;
};

bool parseQualificationOptions(int argc,
                               char** argv,
                               QualificationOptions* output,
                               std::string* diagnostic) {
  if (diagnostic != nullptr) diagnostic->clear();
  if (output == nullptr) {
    if (diagnostic != nullptr) *diagnostic = "qualification-options-invalid";
    return false;
  }
  *output = QualificationOptions{};
  if (const char* environment =
          std::getenv("CHROMASPACE_METAL_CANARY_FRAME_BUDGET")) {
    if (!parseCanaryFrameCount(environment, &output->frameBudget)) {
      if (diagnostic != nullptr) {
        *diagnostic = "qualification-frame-budget-invalid";
      }
      return false;
    }
  }

  bool explicitFrameArgument = false;
  bool explicitScenarioArgument = false;
  const auto reject = [&](const char* reason) {
    if (diagnostic != nullptr) *diagnostic = reason;
    return false;
  };
  const auto parseScenario = [&](const std::string& label) {
    ChromaspaceMetalQualification::Scenario parsed =
        ChromaspaceMetalQualification::Scenario::Count;
    if (!ChromaspaceMetalQualification::parseScenarioLabel(label, parsed)) {
      if (diagnostic != nullptr) {
        *diagnostic = "qualification-scenario-unknown";
      }
      return false;
    }
    output->scenario = parsed;
    output->scenarioSpecified = true;
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i] != nullptr ? argv[i] : "";
    constexpr const char* kFramesPrefix = "--frames=";
    constexpr const char* kScenarioPrefix = "--qualification-scenario=";
    if (argument.rfind(kFramesPrefix, 0) == 0) {
      if (explicitFrameArgument ||
          !parseCanaryFrameCount(
              argument.c_str() + std::char_traits<char>::length(kFramesPrefix),
              &output->frameBudget)) {
        return reject(explicitFrameArgument
                          ? "qualification-frame-budget-duplicate"
                          : "qualification-frame-budget-invalid");
      }
      explicitFrameArgument = true;
    } else if (argument == "--frames") {
      if (explicitFrameArgument || ++i >= argc ||
          !parseCanaryFrameCount(argv[i], &output->frameBudget)) {
        return reject(explicitFrameArgument
                          ? "qualification-frame-budget-duplicate"
                          : "qualification-frame-budget-invalid");
      }
      explicitFrameArgument = true;
    } else if (argument.rfind(kScenarioPrefix, 0) == 0) {
      if (explicitScenarioArgument) {
        return reject("qualification-scenario-duplicate");
      }
      if (!parseScenario(argument.substr(
              std::char_traits<char>::length(kScenarioPrefix)))) {
        return false;
      }
      explicitScenarioArgument = true;
    } else if (argument == "--qualification-scenario") {
      if (explicitScenarioArgument || ++i >= argc) {
        return reject(explicitScenarioArgument
                          ? "qualification-scenario-duplicate"
                          : "qualification-scenario-invalid");
      }
      if (!parseScenario(argv[i] != nullptr ? argv[i] : "")) return false;
      explicitScenarioArgument = true;
    } else {
      return reject("qualification-options-unknown-argument");
    }
  }

  if (output->scenarioSpecified && output->frameBudget <= 0) {
    return reject("qualification-scenario-requires-positive-frames");
  }
  if (diagnostic != nullptr) diagnostic->clear();
  return true;
}

#else

bool canaryFrameBudget(int argc, char** argv, int* frames) {
  if (frames == nullptr) return false;
  *frames = kDefaultCanaryFrames;
  if (const char* environment =
          std::getenv("CHROMASPACE_METAL_CANARY_FRAME_BUDGET")) {
    if (!parseCanaryFrameCount(environment, frames)) return false;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i] ? argv[i] : "";
    constexpr const char* kPrefix = "--frames=";
    if (argument.rfind(kPrefix, 0) == 0) {
      if (!parseCanaryFrameCount(argument.c_str() + std::char_traits<char>::length(kPrefix),
                                 frames)) {
        return false;
      }
    } else if (argument == "--frames") {
      if (++i >= argc || !parseCanaryFrameCount(argv[i], frames)) return false;
    } else {
      return false;
    }
  }
  return true;
}

#endif

std::vector<ChromaspaceMetal::FrameOverlayRect> viewerOverlayRectsForFrame(
    const ChromaspaceViewer::ViewerUiScene& scene,
    int framebufferWidth,
    int framebufferHeight) {
  std::vector<ChromaspaceMetal::FrameOverlayRect> rects;
  if (!scene.ready() || scene.geometry.windowWidth <= 0 ||
      scene.geometry.windowHeight <= 0 ||
      framebufferWidth <= 0 || framebufferHeight <= 0) {
    return rects;
  }
  const float scaleX = static_cast<float>(framebufferWidth) /
                       static_cast<float>(scene.geometry.windowWidth);
  const float scaleY = static_cast<float>(framebufferHeight) /
                       static_cast<float>(scene.geometry.windowHeight);
  rects.reserve(scene.primitives.size());
  for (const auto& primitive : scene.primitives) {
    const auto& source = primitive.rect;
    if (source.x1 <= source.x0 || source.y1 <= source.y0) continue;
    ChromaspaceMetal::FrameOverlayRect rect{};
    rect.x = source.x0 * scaleX;
    rect.y = source.y0 * scaleY;
    rect.w = (source.x1 - source.x0) * scaleX;
    rect.h = (source.y1 - source.y0) * scaleY;
    rect.r = primitive.color.r;
    rect.g = primitive.color.g;
    rect.b = primitive.color.b;
    rect.a = primitive.color.a;
    rects.push_back(rect);
  }
  return rects;
}

std::vector<ChromaspaceMetal::FrameVectorVertex> viewerVectorVerticesForFrame(
    const ChromaspaceViewer::ViewerUiScene& scene,
    int framebufferWidth,
    int framebufferHeight) {
  std::vector<ChromaspaceMetal::FrameVectorVertex> vertices;
  if (!scene.ready() || scene.geometry.windowWidth <= 0 ||
      scene.geometry.windowHeight <= 0 || framebufferWidth <= 0 ||
      framebufferHeight <= 0) {
    return vertices;
  }
  const float scaleX = static_cast<float>(framebufferWidth) /
                       static_cast<float>(scene.geometry.windowWidth);
  const float scaleY = static_cast<float>(framebufferHeight) /
                       static_cast<float>(scene.geometry.windowHeight);
  vertices.reserve(scene.vectors.size());
  for (const auto& source : scene.vectors) {
    ChromaspaceMetal::FrameVectorVertex vertex{};
    vertex.x = source.x * scaleX;
    vertex.y = source.y * scaleY;
    vertex.r = source.color.r;
    vertex.g = source.color.g;
    vertex.b = source.color.b;
    vertex.a = source.color.a;
    vertices.push_back(vertex);
  }
  return vertices;
}

bool loadCocoaViewerFontAtlas(WorkshopText::FontAtlas* atlas,
                              std::string* diagnostic) {
  if (diagnostic) diagnostic->clear();
  if (!atlas) return false;
  *atlas = WorkshopText::FontAtlas{};
  std::vector<std::filesystem::path> candidates;
  NSURL* executableUrl = [[NSBundle mainBundle] executableURL];
  if (executableUrl != nil && [executableUrl isFileURL] &&
      [executableUrl path] != nil) {
    const char* executablePath = [[executableUrl path] fileSystemRepresentation];
    if (executablePath != nullptr) {
      candidates.push_back(
          std::filesystem::path(executablePath).parent_path() /
          "OpenSans-Regular.ttf");
    }
  }
  std::error_code currentPathError;
  const std::filesystem::path current =
      std::filesystem::current_path(currentPathError);
  if (!currentPathError) {
    candidates.push_back(current / "artifacts" / "viewer" /
                         "OpenSans-Regular.ttf");
  }
  candidates.emplace_back(
      "/System/Library/Fonts/Supplemental/Helvetica.ttc");
  candidates.emplace_back(
      "/System/Library/Fonts/Supplemental/Arial.ttf");

  std::string lastError = "no usable font candidate";
  for (const std::filesystem::path& candidate : candidates) {
    std::error_code existsError;
    if (!std::filesystem::exists(candidate, existsError) || existsError) {
      continue;
    }
    std::string loadError;
    if (WorkshopText::loadFontAtlas(candidate.string(), 18, atlas,
                                    &loadError)) {
      return true;
    }
    if (!loadError.empty()) lastError = loadError;
  }
  if (diagnostic) *diagnostic = lastError;
  return false;
}

float cocoaViewerFontScale(int preference) noexcept {
  switch (std::clamp(preference, 0, 2)) {
    case 1: return 1.12f;
    case 2: return 1.25f;
    default: return 1.0f;
  }
}

float cocoaViewerTitleExtraHeight(int preference) noexcept {
  switch (std::clamp(preference, 0, 2)) {
    case 1: return 3.0f;
    case 2: return 6.0f;
    default: return 0.0f;
  }
}

struct CocoaViewerTextBatch {
  bool ready = false;
  std::vector<ChromaspaceMetal::FrameTextVertex> vertices;
  std::vector<ChromaspaceMetal::FrameTextRun> runs;
};

CocoaViewerTextBatch viewerTextForFrame(
    const ChromaspaceViewer::ViewerUiScene& scene,
    const WorkshopText::FontAtlas& atlas,
    int framebufferWidth,
    int framebufferHeight,
    uint64_t atlasId) {
  CocoaViewerTextBatch batch;
  if (atlasId == 0u) {
    return batch;
  }
  const ChromaspaceViewer::ViewerTextLayoutResult layout =
      ChromaspaceViewer::buildViewerTextLayout(
          {&scene, &atlas, framebufferWidth, framebufferHeight});
  if (!layout.ready()) return batch;
  batch.vertices.reserve(layout.vertices.size());
  batch.runs.reserve(layout.runs.size());
  for (const auto& vertex : layout.vertices) {
    batch.vertices.push_back({vertex.x, vertex.y, vertex.u, vertex.v});
  }
  for (const auto& source : layout.runs) {
    ChromaspaceMetal::FrameTextRun run{};
    run.atlasId = atlasId;
    run.firstVertex = source.firstVertex;
    run.vertexCount = source.vertexCount;
    run.r = source.color.r;
    run.g = source.color.g;
    run.b = source.color.b;
    run.a = source.color.a;
    run.clipEnabled = 1u;
    run.clipX = source.clip.x0;
    run.clipY = source.clip.y0;
    run.clipW = source.clip.x1 - source.clip.x0;
    run.clipH = source.clip.y1 - source.clip.y0;
    batch.runs.push_back(run);
  }
  batch.ready = true;
  return batch;
}

bool cocoaResidentSourceSnapshotChanged(
    const ChromaspaceResidentSource::ResidentSourceSnapshot& before,
    const ChromaspaceResidentSource::ResidentSourceSnapshot& after) noexcept {
  if (before.health != after.health || before.senderId != after.senderId ||
      before.viewerGeneration != after.viewerGeneration ||
      before.lastObservedSequence != after.lastObservedSequence ||
      before.liveKeyCount != after.liveKeyCount ||
      before.hasActiveSource != after.hasActiveSource ||
      before.diagnostic != after.diagnostic) {
    return true;
  }
  const auto& a = before.activeSource;
  const auto& b = after.activeSource;
  return a.sourceId != b.sourceId || a.senderId != b.senderId ||
         a.deviceRegistryId != b.deviceRegistryId ||
         a.senderGeneration != b.senderGeneration || a.sequence != b.sequence ||
         a.slotIndex != b.slotIndex || a.slotGeneration != b.slotGeneration ||
         a.readyValue != b.readyValue || a.contentHash != b.contentHash ||
         a.width != b.width || a.height != b.height ||
         a.pixelFormat != b.pixelFormat || a.bytesPerRow != b.bytesPerRow ||
         a.byteSize != b.byteSize;
}

ChromaspaceViewer::ViewerFrameCoordinatorObservation
cocoaFrameCoordinatorObservation(
    NSWindow* window,
    const ChromaspaceViewer::ViewerSessionState& session,
    const ChromaspaceViewer::ViewerWorkspaceState& workspace,
    const ChromaspaceResidentSource::ResidentSourceSnapshot& resident,
    const ChromaspaceMetalViewerRuntime::Runtime& runtime,
    uint64_t sourceRevision,
    bool qualificationContinuous) noexcept {
  ChromaspaceViewer::ViewerFrameCoordinatorObservation observation{};
  const int64_t now = monotonicMilliseconds();
  observation.monotonicTimeMilliseconds =
      now >= 0 ? static_cast<uint64_t>(now) : 0u;
  observation.renderable =
      ChromaspaceViewer::viewerSessionShouldRender(session) &&
      cocoaViewerSurfaceVisibility(window) ==
          ChromaspaceFrameRecoveryPolicy::SurfaceVisibility::Visible;
  observation.closeRequested =
      ChromaspaceViewer::viewerSessionShouldClose(session);
  observation.lifecycleRevision = session.lifecycleRevision;
  observation.viewportRevision = session.viewportRevision;
  observation.inputRevision = session.inputRevision;
  observation.workspaceRevision = workspace.revision;
  observation.sourceRevision = sourceRevision != 0u
                                   ? sourceRevision
                                   : resident.lastObservedSequence;
  observation.runtimeRevision = runtime.generation();
  observation.qualificationContinuous = qualificationContinuous;
  return observation;
}

struct CocoaMemoryPressureTelemetry final {
  uint64_t warnings = 0u;
  uint64_t criticals = 0u;
  uint64_t redraws = 0u;
  uint64_t trimmedSurfaces = 0u;
  uint64_t trimmedSurfaceBytes = 0u;
  uint64_t trimmedDerived = 0u;
  uint64_t trimmedDerivedBytes = 0u;
};

#if defined(CHROMASPACE_METAL_QUALIFICATION)
uint64_t qualificationMetricValue(std::size_t value) noexcept {
  return value > static_cast<std::size_t>(std::numeric_limits<uint64_t>::max())
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(value);
}

void saturatingAddQualificationMetric(std::size_t value,
                                      uint64_t* total) noexcept {
  if (total == nullptr) return;
  const uint64_t converted = qualificationMetricValue(value);
  if (converted > std::numeric_limits<uint64_t>::max() - *total) {
    *total = std::numeric_limits<uint64_t>::max();
  } else {
    *total += converted;
  }
}
#endif

void saturatingAddMemoryMetric(std::size_t value,
                               uint64_t* total) noexcept {
  if (total == nullptr) return;
  const uint64_t converted =
      value > static_cast<std::size_t>(std::numeric_limits<uint64_t>::max())
          ? std::numeric_limits<uint64_t>::max()
          : static_cast<uint64_t>(value);
  if (converted > std::numeric_limits<uint64_t>::max() - *total) {
    *total = std::numeric_limits<uint64_t>::max();
  } else {
    *total += converted;
  }
}

bool applyCocoaMemoryPressure(
    ChromaspaceMetalViewerRuntime::MemoryPressureLevel level,
    ChromaspaceMetalViewerRuntime::Runtime* runtime,
    ChromaspaceViewer::ViewerFrameCoordinator* frameCoordinator,
    CocoaMemoryPressureTelemetry* telemetry,
    std::string* error) {
  if (error != nullptr) error->clear();
  if (runtime == nullptr || frameCoordinator == nullptr || telemetry == nullptr) {
    if (error != nullptr) *error = "memory-pressure-context-invalid";
    return false;
  }
  const auto result = runtime->handleMemoryPressure(level);
  if (!result.accepted()) {
    if (error != nullptr) {
      *error = std::string("memory-pressure-runtime-") +
               ChromaspaceMetalViewerRuntime::memoryPressureStatusLabel(
                   result.status);
    }
    return false;
  }

  if (level == ChromaspaceMetalViewerRuntime::MemoryPressureLevel::Warning) {
    saturatingAddMemoryMetric(1u, &telemetry->warnings);
  } else if (level ==
             ChromaspaceMetalViewerRuntime::MemoryPressureLevel::Critical) {
    saturatingAddMemoryMetric(1u, &telemetry->criticals);
  }
  saturatingAddMemoryMetric(result.rendererTrim.releasedSurfaceCount,
                            &telemetry->trimmedSurfaces);
  saturatingAddMemoryMetric(result.rendererTrim.releasedSurfaceBytes,
                            &telemetry->trimmedSurfaceBytes);
  saturatingAddMemoryMetric(result.rendererTrim.releasedDerivedCacheCount,
                            &telemetry->trimmedDerived);
  saturatingAddMemoryMetric(result.rendererTrim.releasedDerivedCacheBytes,
                            &telemetry->trimmedDerivedBytes);
  if (result.redrawRequired) {
    saturatingAddMemoryMetric(1u, &telemetry->redraws);
    frameCoordinator->invalidate(
        ChromaspaceViewer::ViewerFrameDirtyReason::Recovery);
  }
  return true;
}

bool applyCocoaMemoryPressureBatch(
    const ChromaspaceMetalMemoryPressure::Batch& batch,
    ChromaspaceMetalViewerRuntime::Runtime* runtime,
    ChromaspaceViewer::ViewerFrameCoordinator* frameCoordinator,
    CocoaMemoryPressureTelemetry* telemetry,
    std::string* error) {
  using Level = ChromaspaceMetalViewerRuntime::MemoryPressureLevel;
  switch (batch.strongest) {
    case ChromaspaceMetalMemoryPressure::Signal::None:
      return true;
    case ChromaspaceMetalMemoryPressure::Signal::Normal:
      return applyCocoaMemoryPressure(Level::Normal, runtime, frameCoordinator,
                                      telemetry, error);
    case ChromaspaceMetalMemoryPressure::Signal::Warning:
      return applyCocoaMemoryPressure(Level::Warning, runtime, frameCoordinator,
                                      telemetry, error);
    case ChromaspaceMetalMemoryPressure::Signal::Critical:
      return applyCocoaMemoryPressure(Level::Critical, runtime,
                                      frameCoordinator, telemetry, error);
    case ChromaspaceMetalMemoryPressure::Signal::Count:
      break;
  }
  if (error != nullptr) *error = "memory-pressure-signal-invalid";
  return false;
}

#if defined(CHROMASPACE_METAL_QUALIFICATION)

void setQualificationDiagnostic(std::string* output, const char* value) noexcept {
  if (output == nullptr) return;
  try {
    *output = value != nullptr ? value : "qualification-action-failed";
  } catch (...) {
  }
}

bool applyQualificationAction(
    const ChromaspaceMetalQualification::Action& action,
    NSApplication* application,
    NSWindow* window,
    ChromaspaceViewer::ViewerSessionState* viewerSession,
    ChromaspaceResidentSource::ResidentSourceSession* residentSourceSession,
    ChromaspaceMetalQualification::QualificationSourceAdapter*
        sourceAdapter,
    ChromaspaceMetalQualification::FaultBackend* faultBackend,
    ChromaspaceMetalViewerRuntime::Runtime* runtime,
    ChromaspaceViewer::ViewerFrameCoordinator* frameCoordinator,
    CocoaMemoryPressureTelemetry* memoryPressureTelemetry,
    const std::string& senderId,
    std::string* error) {
  setQualificationDiagnostic(error, "");
  if (application == nil || window == nil || viewerSession == nullptr ||
       residentSourceSession == nullptr || sourceAdapter == nullptr ||
       faultBackend == nullptr || runtime == nullptr ||
       frameCoordinator == nullptr || memoryPressureTelemetry == nullptr ||
       action.ordinal == 0u) {
    setQualificationDiagnostic(error, "qualification-action-invalid-context");
    return false;
  }

  try {
    @try {
      switch (action.kind) {
        case ChromaspaceMetalQualification::ActionKind::Resize: {
          if (action.resizeWidth == 0u || action.resizeHeight == 0u ||
              !std::isfinite(static_cast<double>(action.contentScale)) ||
              action.contentScale <= 0.0f) {
            setQualificationDiagnostic(error,
                                       "qualification-resize-invalid");
            return false;
          }
          // Campaign dimensions are drawable pixels. AppKit accepts logical
          // content points; use the requested scale as the sizing intent and
          // accept the actual backing scale reported by the session viewport.
          const CGFloat logicalWidth =
              static_cast<CGFloat>(action.resizeWidth) /
              static_cast<CGFloat>(action.contentScale);
          const CGFloat logicalHeight =
              static_cast<CGFloat>(action.resizeHeight) /
              static_cast<CGFloat>(action.contentScale);
          [window setContentSize:NSMakeSize(
                                    logicalWidth, logicalHeight)];
          const auto viewportResult =
              updateCocoaViewerSessionViewport(window, viewerSession);
          if (!viewportResult.accepted()) {
            setQualificationDiagnostic(error,
                                       "qualification-resize-viewport-failed");
            return false;
          }
          return true;
        }
        case ChromaspaceMetalQualification::ActionKind::Hide: {
          [window orderOut:nil];
          const auto visibilityResult = applyCocoaViewerSessionEvent(
              viewerSession,
              ChromaspaceViewer::ViewerSessionVisibilityChanged{
                  cocoaViewerSessionVisible(window)});
          if (!visibilityResult.accepted()) {
            setQualificationDiagnostic(error,
                                       "qualification-hide-visibility-failed");
            return false;
          }
          return true;
        }
        case ChromaspaceMetalQualification::ActionKind::Show: {
          [window makeKeyAndOrderFront:nil];
          [application activateIgnoringOtherApps:YES];
          const auto visibilityResult = applyCocoaViewerSessionEvent(
              viewerSession,
              ChromaspaceViewer::ViewerSessionVisibilityChanged{
                  cocoaViewerSessionVisible(window)});
          if (!visibilityResult.accepted()) {
            setQualificationDiagnostic(error,
                                       "qualification-show-visibility-failed");
            return false;
          }
          const auto viewportResult =
              updateCocoaViewerSessionViewport(window, viewerSession);
          if (!viewportResult.accepted()) {
            setQualificationDiagnostic(error,
                                       "qualification-show-viewport-failed");
            return false;
          }
          return true;
        }
        case ChromaspaceMetalQualification::ActionKind::ClearSource:
          if (!residentSourceSession->requestClear(senderId, error)) {
            if (error == nullptr || error->empty()) {
              setQualificationDiagnostic(error,
                                         "qualification-clear-failed");
            }
            return false;
          }
          return true;
        case ChromaspaceMetalQualification::ActionKind::ReplaceSource:
          if (!sourceAdapter->publish(action.sourceGeneration, error)) {
            if (error == nullptr || error->empty()) {
              setQualificationDiagnostic(error,
                                         "qualification-replace-failed");
            }
            return false;
          }
          return true;
        case ChromaspaceMetalQualification::ActionKind::
            InjectDrawableUnavailable:
        case ChromaspaceMetalQualification::ActionKind::
            InjectPriorGpuSubmissionFailure:
          if (!faultBackend->arm(action, error)) {
            if (error == nullptr || error->empty()) {
              setQualificationDiagnostic(error,
                                         "qualification-fault-arm-failed");
            }
            return false;
          }
          return true;
        case ChromaspaceMetalQualification::ActionKind::MemoryPressureWarning:
          return applyCocoaMemoryPressure(
              ChromaspaceMetalViewerRuntime::MemoryPressureLevel::Warning,
              runtime, frameCoordinator, memoryPressureTelemetry, error);
        case ChromaspaceMetalQualification::ActionKind::MemoryPressureCritical:
          return applyCocoaMemoryPressure(
              ChromaspaceMetalViewerRuntime::MemoryPressureLevel::Critical,
              runtime, frameCoordinator, memoryPressureTelemetry, error);
        case ChromaspaceMetalQualification::ActionKind::None:
        case ChromaspaceMetalQualification::ActionKind::Count:
          break;
      }
    } @catch (id) {
      setQualificationDiagnostic(error, "qualification-action-objective-c-exception");
      return false;
    }
  } catch (...) {
    setQualificationDiagnostic(error, "qualification-action-cpp-exception");
    return false;
  }
  setQualificationDiagnostic(error, "qualification-action-unsupported");
  return false;
}

ChromaspaceMetalQualification::RuntimeObservation
qualificationObservationForOutcome(
    ChromaspaceMetalViewerRuntime::OutcomeKind outcome) noexcept {
  using OutcomeKind = ChromaspaceMetalViewerRuntime::OutcomeKind;
  using RuntimeObservation =
      ChromaspaceMetalQualification::RuntimeObservation;
  switch (outcome) {
    case OutcomeKind::RetryLater:
      return RuntimeObservation::RetryLater;
    case OutcomeKind::SuspendUntilVisible:
      return RuntimeObservation::SuspendUntilVisible;
    case OutcomeKind::RuntimeRecreated:
      return RuntimeObservation::RuntimeRecreated;
    case OutcomeKind::Presented:
      return RuntimeObservation::Presented;
    case OutcomeKind::TerminalFailure:
      return RuntimeObservation::TerminalFailure;
    case OutcomeKind::ViewportUpdated:
      break;
  }
  return RuntimeObservation::TerminalFailure;
}

#endif

}  // namespace

@interface ChromaspaceCanaryWindowDelegate : NSObject <NSWindowDelegate> {
 @private
  CocoaViewerInteractionContext* _interaction;
  ChromaspaceViewer::ViewerSessionState* _session;
  __weak NSWindow* _window;
}

- (instancetype)initWithWindow:(NSWindow*)window
                    interaction:(CocoaViewerInteractionContext*)interaction;

@end

@implementation ChromaspaceCanaryWindowDelegate

- (instancetype)initWithWindow:(NSWindow*)window
                    interaction:(CocoaViewerInteractionContext*)interaction {
  self = [super init];
  if (self != nil) {
    _window = window;
    _interaction = interaction;
    _session = interaction != nullptr ? interaction->session : nullptr;
  }
  return self;
}

- (void)windowDidBecomeKey:(NSNotification*)notification {
  (void)notification;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionFocusChanged{true});
  consumeCocoaViewerSessionResult(_interaction, result);
  if (_window != nil) {
    [_window makeFirstResponder:[_window contentView]];
  }
}

- (void)windowDidResignKey:(NSNotification*)notification {
  (void)notification;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionFocusChanged{false});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)windowDidMiniaturize:(NSNotification*)notification {
  (void)notification;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionMiniaturizationChanged{true});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)windowDidDeminiaturize:(NSNotification*)notification {
  (void)notification;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionMiniaturizationChanged{false});
  consumeCocoaViewerSessionResult(_interaction, result);
  const auto viewportResult = updateCocoaViewerSessionViewport(_window, _session);
  consumeCocoaViewerSessionResult(_interaction, viewportResult);
}

- (void)windowDidResize:(NSNotification*)notification {
  (void)notification;
  const auto result = updateCocoaViewerSessionViewport(_window, _session);
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)windowDidChangeBackingProperties:(NSNotification*)notification {
  (void)notification;
  const auto result = updateCocoaViewerSessionViewport(_window, _session);
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)windowDidChangeScreen:(NSNotification*)notification {
  (void)notification;
  const auto result = updateCocoaViewerSessionViewport(_window, _session);
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (void)windowDidChangeOcclusionState:(NSNotification*)notification {
  (void)notification;
  const auto result = applyCocoaViewerSessionEvent(
      _session,
      ChromaspaceViewer::ViewerSessionVisibilityChanged{
          cocoaViewerSessionVisible(_window)});
  consumeCocoaViewerSessionResult(_interaction, result);
}

- (BOOL)windowShouldClose:(NSWindow*)sender {
  (void)sender;
  const auto result = applyCocoaViewerSessionEvent(
      _session, ChromaspaceViewer::ViewerSessionCloseRequested{});
  consumeCocoaViewerSessionResult(_interaction, result);
  return result.shouldClose ||
         ChromaspaceViewer::viewerSessionShouldClose(*_session);
}

@end

int main(int argc, char** argv) {
  @autoreleasepool {
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    QualificationOptions qualificationConfig{};
    std::string qualificationOptionsDiagnostic;
    if (!parseQualificationOptions(argc, argv, &qualificationConfig,
                                   &qualificationOptionsDiagnostic)) {
      reportError("invalid qualification options",
                  qualificationOptionsDiagnostic);
      return EXIT_FAILURE;
    }
    const int frameBudget = qualificationConfig.frameBudget;
    const bool qualificationActive = frameBudget > 0;
#else
    int frameBudget = 0;
    if (!canaryFrameBudget(argc, argv, &frameBudget)) {
      reportError(
          "requires a positive explicit --frames value no greater than 10000",
          {});
      return EXIT_FAILURE;
    }
#endif
    ChromaspaceViewer::ViewerWorkspaceStorePaths workspaceStorePaths{};
    ChromaspaceViewer::ViewerWorkspaceStoreLoadResult workspaceLoad{};
    ChromaspaceViewer::ViewerWorkspaceDocument persistenceDocument{};
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (qualificationActive) {
      const auto rawDefaults =
          ChromaspaceViewer::defaultViewerWorkspaceDocument();
      if (!ChromaspaceViewer::sanitiseViewerWorkspaceDocument(
              rawDefaults, &persistenceDocument) ||
          !ChromaspaceViewer::validateViewerWorkspaceDocument(
              persistenceDocument)) {
        reportError("could not initialize qualification workspace",
                    "qualification-default-workspace-validation-failed");
        return EXIT_FAILURE;
      }
      auto qualificationWorkspace =
          ChromaspaceMetalQualificationWorkspace::buildProfile();
      if (!qualificationWorkspace.ready()) {
        reportError("could not initialize qualification renderer profile",
                    qualificationWorkspace.diagnostic);
        return EXIT_FAILURE;
      }
      persistenceDocument.workspace =
          std::move(qualificationWorkspace.workspace);
      if (!ChromaspaceViewer::validateViewerWorkspaceDocument(
              persistenceDocument)) {
        reportError("could not validate qualification renderer profile",
                    "qualification-renderer-profile-document-invalid");
        return EXIT_FAILURE;
      }
    } else
#endif
    {
      workspaceStorePaths = cocoaViewerWorkspaceStorePaths();
      workspaceLoad = ChromaspaceViewer::loadViewerWorkspaceStore(
          workspaceStorePaths);
    }
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (!qualificationActive)
#endif
    {
      if (workspaceLoad.accepted()) {
        persistenceDocument = workspaceLoad.document;
        if (workspaceLoad.degraded()) {
          reportError("workspace restore degraded; corrupt evidence preserved",
                      workspaceLoad.diagnostic);
        }
      } else {
        // A path-resolution or allocation failure must not launch an invalid
        // workspace.  Use the same sanitized defaults as the store's recovery
        // path and keep the diagnostic visible to the operator.
        const auto rawDefaults =
            ChromaspaceViewer::defaultViewerWorkspaceDocument();
        if (!ChromaspaceViewer::sanitiseViewerWorkspaceDocument(
                rawDefaults, &persistenceDocument) ||
            !ChromaspaceViewer::validateViewerWorkspaceDocument(
                persistenceDocument)) {
          reportError("could not initialize validated viewer defaults",
                      workspaceLoad.diagnostic);
          return EXIT_FAILURE;
        }
        reportError("workspace restore unavailable; using defaults",
                    workspaceLoad.diagnostic);
      }
    }
    ChromaspaceViewer::ViewerWorkspaceState workspace =
        persistenceDocument.workspace;
    if (!ChromaspaceViewer::validateViewerWorkspaceState(workspace)) {
      reportError("could not initialize viewer workspace",
                  "restored document failed state validation");
      return EXIT_FAILURE;
    }
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (qualificationActive) {
      const bool defaultProfile =
          ChromaspaceViewer::kPlotModelCube == 0 &&
          workspace.windows.size() ==
              ChromaspaceMetalQualificationWorkspace::kExpectedWindowCount &&
          workspace.focusedWindowId == 1 && workspace.nextWindowId == 13;
      if (!defaultProfile) {
        reportError("qualification workspace profile changed",
                    "qualification-all-renderers-workspace-profile-invalid");
        return EXIT_FAILURE;
      }
    }
#endif
    NSApplication* application = [NSApplication sharedApplication];
    if (application == nil) {
      reportError("could not initialize NSApplication", {});
      return EXIT_FAILURE;
    }
    [application setActivationPolicy:NSApplicationActivationPolicyRegular];
    [application finishLaunching];

    const CGFloat restoredWidth = static_cast<CGFloat>(std::max(
        1, std::min(16384, persistenceDocument.presentation.windowWidth)));
    const CGFloat restoredHeight = static_cast<CGFloat>(std::max(
        1, std::min(16384, persistenceDocument.presentation.windowHeight)));
    const NSRect contentRect = NSMakeRect(0.0, 0.0, restoredWidth,
                                          restoredHeight);
    const NSUInteger styleMask =
        NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
        NSWindowStyleMaskResizable;
    NSWindow* window = [[NSWindow alloc] initWithContentRect:contentRect
                                                    styleMask:styleMask
                                                      backing:NSBackingStoreBuffered
                                                        defer:NO];
    if (window == nil) {
      reportError("could not create NSWindow", {});
      return EXIT_FAILURE;
    }
    [window setTitle:@"Chromaspace"];
    if (persistenceDocument.presentation.windowPositionValid) {
      CGFloat primaryScreenHeight = 0.0;
      if (cocoaPrimaryScreenHeight(&primaryScreenHeight)) {
        const CGFloat contentY =
            primaryScreenHeight -
            static_cast<CGFloat>(persistenceDocument.presentation.windowPosY) -
            restoredHeight;
        const NSRect restoredContentRect = NSMakeRect(
            static_cast<CGFloat>(persistenceDocument.presentation.windowPosX),
            contentY, restoredWidth, restoredHeight);
        const NSRect restoredFrameRect =
            [window frameRectForContentRect:restoredContentRect];
        [window setFrameOrigin:restoredFrameRect.origin];
      } else {
        [window center];
      }
    } else {
      [window center];
    }

    ChromaspaceViewer::ViewerSessionViewport initialViewport{};
    if (!cocoaViewerSessionViewport(window, &initialViewport)) {
      reportError("could not query initial viewport", {});
      [window close];
      return EXIT_FAILURE;
    }
    ChromaspaceViewer::ViewerSessionState viewerSession{};
    const auto initializeResult = applyCocoaViewerSessionEvent(
        &viewerSession,
        ChromaspaceViewer::ViewerSessionInitialize{
            initialViewport, static_cast<bool>([window isKeyWindow]),
            cocoaViewerSessionVisible(window),
            static_cast<bool>([window isMiniaturized])});
    if (!initializeResult.accepted()) {
      reportError("could not initialize viewer session", {});
      [window close];
      return EXIT_FAILURE;
    }
    ChromaspaceViewer::ViewerController viewerController{};
    ChromaspaceViewer::ViewerCameraInteractionState cameraInteraction{};
    ChromaspaceViewer::ViewerLassoInteractionState lassoInteraction{};
    CocoaViewerInteractionContext interaction{
        &viewerController, &workspace, &viewerSession, &cameraInteraction,
        &lassoInteraction, 0, 0};
    ChromaspaceCanaryWindowDelegate* windowDelegate =
        [[ChromaspaceCanaryWindowDelegate alloc]
            initWithWindow:window
               interaction:&interaction];
    ChromaspaceCanaryInputView* inputView =
        [[ChromaspaceCanaryInputView alloc]
            initWithInteraction:&interaction
                          frame:[[window contentView] bounds]];
    [window setContentView:inputView];
    [window setAcceptsMouseMovedEvents:YES];
    [window setDelegate:windowDelegate];
    const auto focusedWorkspaceWindow = std::find_if(
        workspace.windows.begin(), workspace.windows.end(),
        [&](const ChromaspaceViewer::PlotWindowDomainState& candidate) {
          return candidate.windowId == workspace.focusedWindowId;
        });
    if (focusedWorkspaceWindow != workspace.windows.end()) {
      [window setLevel:focusedWorkspaceWindow->viewState.keepOnTop
                           ? NSFloatingWindowLevel
                           : NSNormalWindowLevel];
    }
    [window makeKeyAndOrderFront:nil];
    [application activateIgnoringOtherApps:YES];
    pumpApplicationEvents(application);
    [window makeFirstResponder:inputView];
    applyCocoaViewerSessionEvent(
        &viewerSession,
        ChromaspaceViewer::ViewerSessionVisibilityChanged{
            cocoaViewerSessionVisible(window)});
    applyCocoaViewerSessionEvent(
        &viewerSession,
        ChromaspaceViewer::ViewerSessionFocusChanged{
            static_cast<bool>([window isKeyWindow])});
    if (!updateCocoaViewerSessionViewport(window, &viewerSession).accepted()) {
      reportError("could not query visible viewport", {});
      [window close];
      return EXIT_FAILURE;
    }

    CocoaWorkspacePersistenceTracker workspacePersistence{};
    workspacePersistence.paths = workspaceStorePaths;
    workspacePersistence.loadStatus = workspaceLoad.status;
    workspacePersistence.repairPending = workspaceLoad.repairSuggested;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (!qualificationActive) {
#endif
    (void)observeCocoaWorkspacePersistence(
        window, &workspace, &persistenceDocument, &workspacePersistence, true);
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    }
#endif

    const auto& initial = viewerSession.viewport;

    std::string error;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    ChromaspaceMetalQualification::Campaign qualificationCampaign(
        {qualificationConfig.scenario,
         static_cast<uint32_t>(std::max(frameBudget, 1))});
    ChromaspaceMetalQualification::NativeSourceFixtureBackend
        qualificationNativeSource{};
    ChromaspaceMetalQualification::QualificationSourceAdapter
        qualificationSourceAdapter(qualificationNativeSource.backend());
    const auto* qualificationDefaultBackend =
        ChromaspaceMetalFrameExecutor::defaultFrameExecutorBackend();
    ChromaspaceMetalQualification::FaultBackend qualificationFaultBackend(
        qualificationDefaultBackend);
    ChromaspaceMetalViewerRuntime::Runtime runtime(
        qualificationFaultBackend.backend());
#else
    ChromaspaceMetalViewerRuntime::Runtime runtime{};
#endif
    void* nativeWindow = (__bridge void*)window;
    WorkshopText::FontAtlas viewerFontAtlas{};
    if (!loadCocoaViewerFontAtlas(&viewerFontAtlas, &error)) {
      reportError("could not load viewer font atlas", error);
      [window close];
      return EXIT_FAILURE;
    }
    const ChromaspaceMetalViewerRuntime::CpuTextAtlasPayload atlasPayload{
        viewerFontAtlas.width, viewerFontAtlas.height,
        viewerFontAtlas.pixels.data(), viewerFontAtlas.pixels.size()};
    if (!runtime.create(nativeWindow,
                        {initial.framebufferWidth, initial.framebufferHeight,
                         initial.contentScaleX},
                        atlasPayload, &error)) {
      reportError("could not create CAMetalLayer compositor", error);
      [window close];
      return EXIT_FAILURE;
    }
    CocoaMemoryPressureMonitor memoryPressureMonitor{};
    if (!memoryPressureMonitor.start(&error)) {
      reportError("could not start memory-pressure monitor", error);
      runtime.shutdown();
      [window close];
      return EXIT_FAILURE;
    }

    const ChromaspaceMetal::ResidentReadiness residentReadiness =
        ChromaspaceMetal::residentReadiness(runtime.compositorId());
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (qualificationActive &&
        (!qualificationCampaign.ready() ||
         !qualificationNativeSource.ready() ||
         !qualificationSourceAdapter.ready() ||
         !qualificationFaultBackend.ready() ||
         residentReadiness.deviceRegistryId == 0u ||
         qualificationNativeSource.snapshot().deviceRegistryId !=
             residentReadiness.deviceRegistryId)) {
      reportError("qualification backend readiness mismatch",
                  qualificationNativeSource.diagnostic());
      runtime.shutdown();
      [window close];
      return EXIT_FAILURE;
    }
#endif
    const ChromaspaceResidentSource::DrainAdapter residentDrain{
        &runtime,
        [](void* context, uint32_t timeoutMilliseconds,
           std::string* drainError) noexcept {
          return context != nullptr &&
                 static_cast<ChromaspaceMetalViewerRuntime::Runtime*>(context)
                     ->drain(timeoutMilliseconds, drainError);
        }};
    ChromaspaceResidentSource::ResidentSourceSession residentSourceSession(
#if defined(CHROMASPACE_METAL_QUALIFICATION)
        qualificationActive ? qualificationSourceAdapter.clientAdapter()
                            : nullptr,
        &residentDrain);
#else
        nullptr, &residentDrain);
#endif
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    const std::string qualificationSenderId = "qualification";
    if (qualificationActive) {
      bool qualificationStartupReady = true;
      std::string qualificationStartupError;
      ChromaspaceResidentSource::TickResult startupTick{};
      ChromaspaceResidentSource::TickResult activeTick{};
      try {
        qualificationStartupReady = residentSourceSession.requestSender(
            qualificationSenderId, residentReadiness.deviceRegistryId,
            monotonicMilliseconds(), &qualificationStartupError);
        if (qualificationStartupReady) {
          startupTick = residentSourceSession.tick(monotonicMilliseconds());
          qualificationStartupReady =
              startupTick.snapshot.health ==
                  ChromaspaceResidentSource::SessionHealth::Ready &&
              startupTick.snapshot.senderId == qualificationSenderId;
          if (!qualificationStartupReady &&
              qualificationStartupError.empty()) {
            qualificationStartupError =
                "qualification-source-client-not-ready";
          }
        }
        if (qualificationStartupReady) {
          qualificationStartupReady = qualificationSourceAdapter.publish(
              1u, &qualificationStartupError);
        }
        if (qualificationStartupReady) {
          activeTick = residentSourceSession.tick(monotonicMilliseconds());
          const auto& activeSource = activeTick.snapshot.activeSource;
          qualificationStartupReady =
              activeTick.snapshot.health ==
                  ChromaspaceResidentSource::SessionHealth::Ready &&
              activeTick.snapshot.hasActiveSource &&
              activeSource.sourceId != 0u &&
              activeSource.senderId == qualificationSenderId &&
              activeSource.deviceRegistryId == residentReadiness.deviceRegistryId &&
              activeSource.senderGeneration == 1u &&
              activeSource.sequence == 1u &&
              activeSource.slotGeneration == 1u;
          if (!qualificationStartupReady &&
              qualificationStartupError.empty()) {
            qualificationStartupError =
                "qualification-source-not-active-after-publish";
          }
        }
      } catch (...) {
        qualificationStartupReady = false;
        qualificationStartupError = "qualification-source-startup-exception";
      }
      if (!qualificationStartupReady) {
        reportError("could not initialize qualification resident source",
                    qualificationStartupError);
        (void)residentSourceSession.shutdown();
        runtime.shutdown();
        [window close];
        return EXIT_FAILURE;
      }
    }
#endif
    ChromaspaceViewer::ViewerLiveCommandReducer liveCommandReducer{};
    CocoaHeartbeatState heartbeatState{};
    {
      std::lock_guard<std::mutex> lock(heartbeatState.mutex);
      heartbeatState.visible = cocoaViewerSessionVisible(window);
      heartbeatState.iconified = [window isMiniaturized];
      heartbeatState.focused = [window isKeyWindow];
      heartbeatState.updatedMilliseconds = monotonicMilliseconds();
      heartbeatState.resident = residentSourceSession.snapshot();
    }
    ChromaspaceViewer::ViewerCommandServerOptions commandServerOptions{};
    commandServerOptions.heartbeatAck = cocoaHeartbeatAck;
    commandServerOptions.heartbeatAckContext = &heartbeatState;
    ChromaspaceViewer::ViewerCommandServer commandServer(
        &liveCommandReducer, commandServerOptions);
    if (!commandServer.start()) {
      const auto commandServerSnapshot = commandServer.snapshot();
      reportError("could not start per-user viewer command server",
                  commandServerSnapshot.lastError);
      (void)residentSourceSession.shutdown();
      runtime.shutdown();
      [window close];
      return EXIT_FAILURE;
    }

    bool success = true;
    int frame = 0;
    const auto qualificationStart = std::chrono::steady_clock::now();
    bool qualificationMemoryAvailable = true;
    uint64_t qualificationPeakReservedBytes = 0u;
    uint64_t qualificationPeakLogicalBytes = 0u;
    std::size_t qualificationPeakActiveSubmissions = 0u;
    uint64_t qualificationRuntimeRecreations = 0u;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    QualificationPlotTelemetry qualificationPlotTelemetry{};
    qualificationPlotTelemetry.workspaceWindows =
        static_cast<uint64_t>(workspace.windows.size());
    ChromaspaceMetalQualificationWorkspace::Tracker
        qualificationRendererCoverage{};
#endif
    CocoaMemoryPressureTelemetry memoryPressureTelemetry{};
    uint32_t nextPumpWaitMilliseconds = 0u;
    uint64_t sourceRevision = 1u;
    auto previousResidentSnapshot = residentSourceSession.snapshot();
    ChromaspaceViewer::ViewerFrameCoordinator frameCoordinator{};
    while ((frameBudget == 0 || frame < frameBudget) &&
           !ChromaspaceViewer::viewerSessionShouldClose(viewerSession)) {
      // Pump according to the last coordinator recommendation, then process
      // live commands and the resident source before making the next frame
      // decision.  A provisional Render decision is never issued before this
      // processing point, so revisions arriving with the events are folded
      // into the same ticket.
      pumpApplicationEvents(
          application,
          nextPumpWaitMilliseconds > 0u
              ? static_cast<NSTimeInterval>(nextPumpWaitMilliseconds) / 1000.0
              : 0.0);
      nextPumpWaitMilliseconds = 0u;
      const auto memoryPressureBatch = memoryPressureMonitor.consume();
      if (!memoryPressureBatch.empty()) {
        std::string memoryPressureError;
        if (!applyCocoaMemoryPressureBatch(
                memoryPressureBatch, &runtime, &frameCoordinator,
                &memoryPressureTelemetry, &memoryPressureError)) {
          reportError("could not apply Metal memory-pressure policy",
                      memoryPressureError);
          success = false;
          break;
        }
      }
      ChromaspaceViewer::ViewerLiveCommandBatch liveCommands{};
      if (!liveCommandReducer.drain(&liveCommands)) {
        pumpRecoveryEvents(application, 1u);
        continue;
      }
      if (liveCommands.hasParams) {
        int updatedWindowId = -1;
        std::string liveStateError;
        if (!ChromaspaceMetalWorkspaceFrame::applyLiveParamsToFocusedWindow(
                liveCommands.params, &workspace, &updatedWindowId,
                &liveStateError)) {
          reportError("could not apply live viewer state", liveStateError);
        } else {
          const auto updatedWindow = std::find_if(
              workspace.windows.begin(), workspace.windows.end(),
              [updatedWindowId](
                  const ChromaspaceViewer::PlotWindowDomainState& candidate) {
                return candidate.windowId == updatedWindowId;
              });
          if (updatedWindow != workspace.windows.end()) {
            [window setLevel:updatedWindow->viewState.keepOnTop
                                 ? NSFloatingWindowLevel
                                 : NSNormalWindowLevel];
          }
          std::string sourceError;
          if (!residentSourceSession.requestSender(
                  liveCommands.params.senderId,
                  residentReadiness.deviceRegistryId,
                  monotonicMilliseconds(), &sourceError)) {
            reportError("could not select resident source sender", sourceError);
          }
        }
      }
      if (liveCommands.hasClear) {
        std::string sourceError;
        if (!residentSourceSession.requestClear(
                liveCommands.clearSenderId, &sourceError)) {
          reportError("could not clear resident source", sourceError);
        }
      }
      if (liveCommands.bringToFront) {
        [window makeKeyAndOrderFront:nil];
        [application activateIgnoringOtherApps:YES];
      }
      if (liveCommands.shutdown) {
        (void)applyCocoaViewerSessionEvent(
            &viewerSession,
            ChromaspaceViewer::ViewerSessionCloseRequested{});
      }
      const auto residentTick =
          residentSourceSession.tick(monotonicMilliseconds());
      interaction.sourceWidth = 0;
      interaction.sourceHeight = 0;
      if (residentTick.snapshot.hasActiveSource) {
        const auto& source = residentTick.snapshot.activeSource;
        interaction.sourceWidth =
            source.semantics.sourceWidth > 0u
                ? static_cast<int>(source.semantics.sourceWidth)
                : source.width;
        interaction.sourceHeight =
            source.semantics.sourceHeight > 0u
                ? static_cast<int>(source.semantics.sourceHeight)
                : source.height;
      }
      {
        std::lock_guard<std::mutex> lock(heartbeatState.mutex);
        heartbeatState.visible = cocoaViewerSessionVisible(window);
        heartbeatState.iconified = [window isMiniaturized];
        heartbeatState.focused = [window isKeyWindow];
        heartbeatState.updatedMilliseconds = monotonicMilliseconds();
        heartbeatState.resident = residentTick.snapshot;
      }
      if (cocoaResidentSourceSnapshotChanged(previousResidentSnapshot,
                                             residentTick.snapshot)) {
        if (sourceRevision != std::numeric_limits<uint64_t>::max()) {
          ++sourceRevision;
        }
        previousResidentSnapshot = residentTick.snapshot;
      }
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      if (!qualificationActive) {
#endif
        (void)observeCocoaWorkspacePersistence(
            window, &workspace, &persistenceDocument, &workspacePersistence,
            false);
        (void)maybeAutosaveCocoaWorkspace(
            &workspacePersistence, persistenceDocument, false);
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      }
#endif
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      if (qualificationActive) {
        ChromaspaceMetalQualification::Action qualificationAction{};
        std::string qualificationActionError;
        if (!qualificationCampaign.next(
                static_cast<uint32_t>(std::max(frame, 0)),
                &qualificationAction, &qualificationActionError)) {
          reportError("qualification campaign could not advance",
                      qualificationActionError);
          success = false;
          break;
        }
        if (qualificationAction.kind !=
            ChromaspaceMetalQualification::ActionKind::None) {
          const bool applied = applyQualificationAction(
              qualificationAction, application, window, &viewerSession,
              &residentSourceSession, &qualificationSourceAdapter,
              &qualificationFaultBackend, &runtime, &frameCoordinator,
              &memoryPressureTelemetry, qualificationSenderId,
              &qualificationActionError);
          std::string acknowledgementError;
          const bool acknowledged = qualificationCampaign.acknowledge(
              qualificationAction.ordinal, applied, &acknowledgementError);
          if (!applied || !acknowledged) {
            reportError(
                "qualification action failed",
                !qualificationActionError.empty()
                    ? qualificationActionError
                    : acknowledgementError);
            success = false;
            break;
          }
        }
      }
#endif
      if (ChromaspaceViewer::viewerSessionShouldClose(viewerSession)) break;
      const auto coordinatorObservation = cocoaFrameCoordinatorObservation(
          window, viewerSession, workspace, residentTick.snapshot, runtime,
          sourceRevision, frameBudget > 0);
      frameCoordinator.observe(coordinatorObservation);
      const auto frameDecision = frameCoordinator.decide(
          coordinatorObservation.monotonicTimeMilliseconds);
      if (frameDecision.kind ==
          ChromaspaceViewer::ViewerFrameDecisionKind::Close) {
        break;
      }
      if (frameDecision.kind !=
          ChromaspaceViewer::ViewerFrameDecisionKind::Render) {
        nextPumpWaitMilliseconds = frameDecision.waitMilliseconds;
        continue;
      }
      const uint64_t renderTicket = frameDecision.renderTicket;

      const auto viewport = viewerSession.viewport;
      const int nextWidth = viewport.framebufferWidth;
      const int nextHeight = viewport.framebufferHeight;
      const float nextScale = viewport.contentScaleX;
      const int nextLogicalWidth = viewport.logicalWidth;
      const int nextLogicalHeight = viewport.logicalHeight;
      const auto& compositorState = runtime.compositor();
      if (nextWidth != compositorState.drawableWidth ||
          nextHeight != compositorState.drawableHeight ||
          nextScale != compositorState.contentsScale) {
        const auto visibility = cocoaViewerSurfaceVisibility(window);
        const auto resizeOutcome = runtime.resize(
            {nextWidth, nextHeight, nextScale}, visibility, &error);
        if (resizeOutcome.kind ==
            ChromaspaceMetalViewerRuntime::OutcomeKind::RuntimeRecreated) {
          ++qualificationRuntimeRecreations;
          // The caller must rebuild text runs against the new atlas ID.
          (void)frameCoordinator.complete(
              renderTicket,
              ChromaspaceViewer::ViewerFrameCompletionKind::RuntimeRecreated);
          nextPumpWaitMilliseconds = resizeOutcome.waitMilliseconds;
          continue;
        }
        if (resizeOutcome.kind ==
                ChromaspaceMetalViewerRuntime::OutcomeKind::RetryLater ||
            resizeOutcome.kind ==
                ChromaspaceMetalViewerRuntime::OutcomeKind::SuspendUntilVisible) {
          (void)frameCoordinator.complete(
              renderTicket,
              resizeOutcome.kind ==
                      ChromaspaceMetalViewerRuntime::OutcomeKind::RetryLater
                  ? ChromaspaceViewer::ViewerFrameCompletionKind::Retry
                  : ChromaspaceViewer::ViewerFrameCompletionKind::Suspend);
          nextPumpWaitMilliseconds = resizeOutcome.waitMilliseconds;
          continue;
        }
        if (resizeOutcome.kind ==
            ChromaspaceMetalViewerRuntime::OutcomeKind::TerminalFailure) {
          reportError("could not resize CAMetalLayer compositor", error);
          success = false;
          break;
        }
      }
      ChromaspaceViewer::ViewerFramePlanRequest planRequest{};
      planRequest.windowWidth = nextLogicalWidth;
      planRequest.windowHeight = nextLogicalHeight;
      planRequest.framebufferWidth = nextWidth;
      planRequest.framebufferHeight = nextHeight;
      planRequest.reservedLeftPixels = 0.0f;
      for (const auto& workspaceWindow : workspace.windows) {
        planRequest.windows.push_back(
            {workspaceWindow.windowId, workspaceWindow.rect,
             workspaceWindow.viewState.plotModel,
             workspaceWindow.viewState.stateRevision,
             !workspaceWindow.sourceSignalDocked});
      }
      const ChromaspaceViewer::ViewerFramePlan framePlan =
          ChromaspaceViewer::buildViewerFramePlan(planRequest);
      ChromaspaceViewer::ViewerUiProjectionRequest projectionRequest{};
      projectionRequest.framePlan = &framePlan;
      projectionRequest.workspace = &workspace;
      projectionRequest.controller = viewerController.state();
      projectionRequest.showWorkspaceButtons =
          persistenceDocument.presentation.showWorkspaceButtons;
      projectionRequest.showSliceButtonInPlotWindows =
          persistenceDocument.presentation.showSliceButtonInPlotWindows;
      const float viewerFontScale = cocoaViewerFontScale(
          persistenceDocument.presentation.viewerFontSize);
      projectionRequest.textScale = viewerFontScale;
      projectionRequest.layoutIndex =
          persistenceDocument.presentation.activeStandardLayoutIndex;
      projectionRequest.hasPointer = viewerSession.pointerPresent;
      projectionRequest.pointerX = static_cast<float>(viewerSession.pointerX);
      projectionRequest.pointerY = static_cast<float>(viewerSession.pointerY);
      projectionRequest.source.available = residentTick.snapshot.hasActiveSource;
      if (residentTick.snapshot.hasActiveSource) {
        const auto& source = residentTick.snapshot.activeSource;
        projectionRequest.source.sourceWidth =
            source.semantics.sourceWidth > 0u
                ? static_cast<int>(source.semantics.sourceWidth)
                : source.width;
        projectionRequest.source.sourceHeight =
            source.semantics.sourceHeight > 0u
                ? static_cast<int>(source.semantics.sourceHeight)
                : source.height;
        projectionRequest.source.displayWidth = source.width;
        projectionRequest.source.displayHeight = source.height;
      }
      projectionRequest.windows.reserve(framePlan.windows.size());
      for (const auto& planned : framePlan.windows) {
        ChromaspaceViewer::ViewerUiProjectionWindowFacts facts{};
        facts.windowId = planned.windowId;
        facts.titleMetrics.titleExtraHeight = cocoaViewerTitleExtraHeight(
            persistenceDocument.presentation.viewerFontSize);
        facts.titleMetrics.textScale = std::clamp(
            0.88f * viewerFontScale, 0.66f, 1.18f);
        facts.titleMetrics.fontAscent =
            static_cast<float>(std::max(1, viewerFontAtlas.ascent));
        facts.titleMetrics.fontDescent =
            static_cast<float>(std::max(0, viewerFontAtlas.descent));
        facts.titleMetrics.fontAvailable = true;
        const auto workspaceWindow = std::find_if(
            workspace.windows.begin(), workspace.windows.end(),
            [&](const ChromaspaceViewer::PlotWindowDomainState& candidate) {
              return candidate.windowId == planned.windowId;
            });
        facts.slicingAnimationProgress =
            workspaceWindow != workspace.windows.end() &&
                    workspaceWindow->slicingDrawerOpen
                ? 1.0f
                : 0.0f;
        projectionRequest.windows.push_back(std::move(facts));
      }
      ChromaspaceViewer::ViewerUiProjectionResult projection =
          ChromaspaceViewer::projectViewerUi(projectionRequest);
      if (!projection.ready()) {
        reportError("could not project portable viewer UI facts", {});
        success = false;
        break;
      }
      for (auto& windowInput : projection.input.windows) {
        windowInput.titleMetrics.measuredMetadataWidth =
            windowInput.metadata.empty()
                ? 0.0f
                : WorkshopText::measureTextWidth(
                      viewerFontAtlas, windowInput.metadata,
                      windowInput.titleMetrics.textScale);
      }
      ChromaspaceViewer::ViewerUiScene uiScene =
          ChromaspaceViewer::buildViewerUiScene(framePlan, projection.input);
      if (!uiScene.ready()) {
        reportError("could not build portable viewer UI scene", {});
        success = false;
        break;
      }
      if (!viewerController.publishScene(uiScene, workspace.focusedWindowId)) {
        reportError("could not publish portable viewer UI scene", {});
        success = false;
        break;
      }
      if ((workspace.sourceLassoSessionActive ||
           lassoInteraction.pointerCaptureActive) &&
          interaction.sourceWidth > 0 && interaction.sourceHeight > 0) {
        const auto lassoOverlay =
            ChromaspaceViewer::appendViewerLassoOverlay(
                lassoInteraction, workspace,
                static_cast<double>(interaction.sourceWidth),
                static_cast<double>(interaction.sourceHeight), &uiScene);
        if (!lassoOverlay.ready()) {
          reportError("could not project Source Signal lasso overlay", {});
          success = false;
          break;
        }
      }
      const auto publishedState = viewerController.state();
      if (publishedState.focusedWindowId >= 0) {
        if (workspace.focusedWindowId != publishedState.focusedWindowId) {
          workspace.focusedWindowId = publishedState.focusedWindowId;
          if (workspace.revision != std::numeric_limits<uint64_t>::max()) {
            ++workspace.revision;
          }
        }
      }
      const auto overlayRects = viewerOverlayRectsForFrame(
          uiScene, nextWidth, nextHeight);
      const auto vectorVertices = viewerVectorVerticesForFrame(
          uiScene, nextWidth, nextHeight);
      const CocoaViewerTextBatch textBatch = viewerTextForFrame(
          uiScene, viewerFontAtlas, nextWidth, nextHeight,
          runtime.textAtlasId());
      if (vectorVertices.empty() || !textBatch.ready) {
        reportError("portable control scene projection failed", {});
        success = false;
        break;
      }

      ChromaspaceMetalFrameExecutor::FrameBatch batch{};
      batch.compositeOverlayRects = overlayRects;
      batch.compositeVectorVertices = vectorVertices;
      batch.compositeTextVertices = textBatch.vertices;
      batch.compositeTextRuns = textBatch.runs;
      batch.clearColor = {{0.015f, 0.018f, 0.025f, 1.0f}};

      ChromaspaceMetalWorkspaceFrame::CompileRequest plotCompileRequest{};
      plotCompileRequest.framePlan = &framePlan;
      plotCompileRequest.workspace = &workspace;
      plotCompileRequest.residentSource = residentTick.snapshot.hasActiveSource
                                              ? &residentTick.snapshot.activeSource
                                              : nullptr;
      plotCompileRequest.sourceDiagnostic =
          residentTick.snapshot.diagnostic.empty()
              ? "resident-source-unavailable"
              : residentTick.snapshot.diagnostic;
      plotCompileRequest.frameRevision = std::max<uint64_t>(1u, workspace.revision);
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      if (qualificationActive) {
        plotCompileRequest.glossPresentation =
            (frame & 1) == 0
                ? ChromaspaceMetalPlotCompiler::GlossPresentation::Field2D
                : ChromaspaceMetalPlotCompiler::GlossPresentation::Projection3D;
      }
#endif
      auto compiledPlots =
          ChromaspaceMetalWorkspaceFrame::compileWorkspaceFrame(
              plotCompileRequest);
      if (!compiledPlots.ready()) {
        reportError("could not compile Metal workspace frame",
                    compiledPlots.diagnostic);
        success = false;
        break;
      }
      ChromaspaceMetalFrameExecutor::FrameExecutionStats executionStats{};
      const auto visibility = cocoaViewerSurfaceVisibility(window);
      ChromaspaceMetalPlotRenderer::RenderResult plotResult{};
      const auto renderStart = std::chrono::steady_clock::now();
      const auto outcome = runtime.render(
          compiledPlots.frame, batch, visibility, &plotResult, &executionStats,
          &error);
      const auto renderEnd = std::chrono::steady_clock::now();
      const double renderCpuMilliseconds =
          std::chrono::duration<double, std::milli>(renderEnd - renderStart)
              .count();
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      if (qualificationActive) {
        const auto faultSnapshot = qualificationFaultBackend.snapshot();
        std::string observationError;
        if (!qualificationFaultBackend.observe(
                qualificationObservationForOutcome(outcome.kind),
                faultSnapshot.activeOrdinal, &observationError)) {
          reportError("qualification fault observation failed",
                      observationError);
          success = false;
          break;
        }
      }
#endif
      if (outcome.kind ==
          ChromaspaceMetalViewerRuntime::OutcomeKind::RuntimeRecreated) {
        ++qualificationRuntimeRecreations;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
        if (qualificationActive) {
          qualificationRendererCoverage.reset();
        }
#endif
        (void)frameCoordinator.complete(
            renderTicket,
            ChromaspaceViewer::ViewerFrameCompletionKind::RuntimeRecreated);
        nextPumpWaitMilliseconds = outcome.waitMilliseconds;
        continue;
      }
      if (outcome.kind ==
              ChromaspaceMetalViewerRuntime::OutcomeKind::RetryLater ||
          outcome.kind ==
              ChromaspaceMetalViewerRuntime::OutcomeKind::SuspendUntilVisible) {
        (void)frameCoordinator.complete(
            renderTicket,
            outcome.kind ==
                    ChromaspaceMetalViewerRuntime::OutcomeKind::RetryLater
                ? ChromaspaceViewer::ViewerFrameCompletionKind::Retry
                : ChromaspaceViewer::ViewerFrameCompletionKind::Suspend);
        nextPumpWaitMilliseconds = outcome.waitMilliseconds;
        continue;
      }
      if (outcome.kind !=
          ChromaspaceMetalViewerRuntime::OutcomeKind::Presented) {
        reportError("could not execute Metal frame transaction", error);
        success = false;
        break;
      }
      const auto completion = frameCoordinator.complete(
          renderTicket, ChromaspaceViewer::ViewerFrameCompletionKind::Presented);
      if (!completion.accepted()) {
        reportError("Metal frame completion ticket was stale", {});
        success = false;
        break;
      }
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      if (qualificationActive) {
        if (!plotResult.frameSucceeded) {
          reportError("qualification plot metrics frame failed",
                      "qualification-plot-render-result-failed");
          success = false;
          break;
        }
        std::string coverageDiagnostic;
        if (!qualificationRendererCoverage.observe(
                compiledPlots.frame, plotCompileRequest.glossPresentation,
                &coverageDiagnostic)) {
          reportError("qualification renderer coverage frame failed",
                      coverageDiagnostic);
          success = false;
          break;
        }
        if (!std::isfinite(renderCpuMilliseconds) ||
            renderCpuMilliseconds <= 0.0 ||
            !std::isfinite(
                qualificationPlotTelemetry.presentedCpuTotalMilliseconds) ||
            renderCpuMilliseconds >
                std::numeric_limits<double>::max() -
                    qualificationPlotTelemetry.presentedCpuTotalMilliseconds) {
          reportError("qualification CPU telemetry invalid",
                      "qualification-presented-cpu-duration-invalid");
          success = false;
          break;
        }
        ++qualificationPlotTelemetry.samples;
        qualificationPlotTelemetry.surfaceResidentPeakBytes = std::max(
            qualificationPlotTelemetry.surfaceResidentPeakBytes,
            qualificationMetricValue(plotResult.residentSurfaceBytes));
        qualificationPlotTelemetry.surfaceTransientPeakBytes = std::max(
            qualificationPlotTelemetry.surfaceTransientPeakBytes,
            qualificationMetricValue(plotResult.transientSurfaceBytes));
        qualificationPlotTelemetry.derivedResidentPeakBytes = std::max(
            qualificationPlotTelemetry.derivedResidentPeakBytes,
            qualificationMetricValue(plotResult.residentDerivedBytes));
        qualificationPlotTelemetry.derivedTransientPeakBytes = std::max(
            qualificationPlotTelemetry.derivedTransientPeakBytes,
            qualificationMetricValue(plotResult.transientDerivedBytes));
        saturatingAddQualificationMetric(plotResult.residentContentHitCount,
                                          &qualificationPlotTelemetry.contentHits);
        saturatingAddQualificationMetric(plotResult.residentDerivedHitCount,
                                          &qualificationPlotTelemetry.derivedHits);
        saturatingAddQualificationMetric(
            plotResult.residentDerivedCandidateCount,
            &qualificationPlotTelemetry.derivedCandidates);
        saturatingAddQualificationMetric(plotResult.evictedDerivedCacheCount,
                                          &qualificationPlotTelemetry.derivedEvictions);
        saturatingAddQualificationMetric(plotResult.createdSurfaceCount,
                                          &qualificationPlotTelemetry.surfaceCreates);
        saturatingAddQualificationMetric(plotResult.resizedSurfaceCount,
                                          &qualificationPlotTelemetry.surfaceResizes);
        saturatingAddQualificationMetric(
            plotResult.replacedSurfaceCount,
            &qualificationPlotTelemetry.surfaceReplacements);
        saturatingAddQualificationMetric(plotResult.prunedSurfaceCount,
                                          &qualificationPlotTelemetry.surfacePrunes);
        qualificationPlotTelemetry.presentedCpuTotalMilliseconds +=
            renderCpuMilliseconds;
        qualificationPlotTelemetry.presentedCpuMaximumMilliseconds = std::max(
            qualificationPlotTelemetry.presentedCpuMaximumMilliseconds,
            renderCpuMilliseconds);
        ++qualificationPlotTelemetry.presentedCpuSamples;
      }
#endif
      if (!executionStats.transientMemory.available) {
        qualificationMemoryAvailable = false;
      } else {
        qualificationPeakReservedBytes = std::max(
            qualificationPeakReservedBytes,
            executionStats.transientMemory.peakInFlightReservedBytes);
        qualificationPeakLogicalBytes = std::max(
            qualificationPeakLogicalBytes,
            executionStats.transientMemory.peakInFlightLogicalBytes);
        qualificationPeakActiveSubmissions = std::max(
            qualificationPeakActiveSubmissions,
            executionStats.transientMemory.peakActiveSubmissionCount);
      }
      ++frame;
    }

    if (frameBudget > 0) {
      ChromaspaceMetal::FrameCompletionStats completionStats{};
      std::string qualificationError;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      const bool drained = runtime.drain(10000u, &qualificationError);
      const bool completionAvailable =
          runtime.completionStats(&completionStats);
#else
      const bool drained = success && frame == frameBudget &&
                           runtime.drain(10000u, &qualificationError);
      const bool completionAvailable =
          drained && runtime.completionStats(&completionStats);
#endif
      const bool completionClean =
          completionAvailable && completionStats.available &&
          completionStats.submittedSerial == completionStats.completedSerial &&
          completionStats.failedSubmissionCount == 0u &&
          completionStats.timedSubmissionCount > 0u &&
          completionStats.timedSubmissionCount +
                  completionStats.untimedSubmissionCount ==
              completionStats.completedSerial;
#if defined(CHROMASPACE_METAL_QUALIFICATION)
      const auto qualificationRendererCoverageSnapshot =
          qualificationRendererCoverage.snapshot();
      bool qualificationPassed =
          success && frame == frameBudget && drained && completionClean &&
          qualificationMemoryAvailable;
      ChromaspaceMetalQualification::Snapshot campaignSnapshot{};
      ChromaspaceMetalQualification::FaultSnapshot faultSnapshot{};
      ChromaspaceMetalQualification::SourceAdapterSnapshot sourceSnapshot{};
      ChromaspaceMetalQualification::NativeSourceFixtureSnapshot nativeSnapshot{};
      bool campaignFinished = false;
      bool faultFinished = false;
      bool sourceFinished = false;
      bool sourceCountsCoherent = false;
      bool nativeCountsCoherent = false;
      bool memoryCountsCoherent = false;
      ChromaspaceResidentSource::ShutdownResult qualificationShutdown{};
      if (qualificationActive) {
        qualificationShutdown = residentSourceSession.shutdown();
        campaignSnapshot = qualificationCampaign.snapshot();
        faultSnapshot = qualificationFaultBackend.snapshot();
        sourceSnapshot = qualificationSourceAdapter.snapshot();
        nativeSnapshot = qualificationNativeSource.snapshot();

        const auto rememberError = [&](const std::string& value) {
          if (qualificationError.empty() && !value.empty()) {
            qualificationError = value;
          }
        };
        bool plotMetricsCoherent = true;
        const auto requirePlotMetric = [&](bool condition,
                                           const char* diagnostic) {
          if (!condition) {
            plotMetricsCoherent = false;
            rememberError(diagnostic);
          }
        };
        requirePlotMetric(
            std::string(kQualificationWorkspaceProfile) ==
                "qualification-all-renderers-v2" &&
                qualificationPlotTelemetry.workspaceWindows ==
                    ChromaspaceMetalQualificationWorkspace::
                        kExpectedWindowCount,
            "qualification-workspace-profile-invalid");
        requirePlotMetric(
            qualificationRendererCoverageSnapshot.complete() &&
                qualificationRendererCoverageSnapshot.coveredMask ==
                    ChromaspaceMetalQualificationWorkspace::
                        kRequiredCoverageMask &&
                qualificationRendererCoverageSnapshot.requiredMask ==
                    ChromaspaceMetalQualificationWorkspace::
                        kRequiredCoverageMask &&
                qualificationRendererCoverageSnapshot.coveredVariantCount ==
                    13u &&
                qualificationRendererCoverageSnapshot
                        .acceptedObservationCount >= 2u &&
                qualificationRendererCoverageSnapshot
                        .acceptedObservationCount <=
                    static_cast<uint32_t>(std::max(frame, 0)),
            "qualification-renderer-coverage-incomplete");
        const uint64_t presentedFrames =
            static_cast<uint64_t>(std::max(frame, 0));
        requirePlotMetric(
            qualificationPlotTelemetry.samples == presentedFrames,
            "qualification-plot-samples-mismatch");
        requirePlotMetric(
            qualificationPlotTelemetry.presentedCpuSamples == presentedFrames,
            "qualification-presented-cpu-samples-mismatch");
        requirePlotMetric(
            std::isfinite(
                qualificationPlotTelemetry.presentedCpuTotalMilliseconds) &&
                std::isfinite(
                    qualificationPlotTelemetry.presentedCpuMaximumMilliseconds) &&
                qualificationPlotTelemetry.presentedCpuTotalMilliseconds > 0.0 &&
                qualificationPlotTelemetry.presentedCpuMaximumMilliseconds > 0.0 &&
                qualificationPlotTelemetry.presentedCpuMaximumMilliseconds <=
                    qualificationPlotTelemetry.presentedCpuTotalMilliseconds,
            "qualification-presented-cpu-metrics-invalid");
        requirePlotMetric(
            qualificationPlotTelemetry.surfaceResidentPeakBytes > 0u,
            "qualification-plot-surface-resident-peak-invalid");
        requirePlotMetric(
            qualificationPlotTelemetry.surfaceTransientPeakBytes >=
                qualificationPlotTelemetry.surfaceResidentPeakBytes,
            "qualification-plot-surface-transient-peak-invalid");
        requirePlotMetric(
            qualificationPlotTelemetry.derivedResidentPeakBytes > 0u,
            "qualification-plot-derived-resident-peak-invalid");
        requirePlotMetric(
            qualificationPlotTelemetry.derivedTransientPeakBytes >=
                qualificationPlotTelemetry.derivedResidentPeakBytes,
            "qualification-plot-derived-transient-peak-invalid");
        requirePlotMetric(qualificationPlotTelemetry.surfaceCreates > 0u,
                          "qualification-plot-surface-creates-invalid");
        requirePlotMetric(qualificationPlotTelemetry.contentHits > 0u,
                          "qualification-plot-content-hits-invalid");
        requirePlotMetric(qualificationPlotTelemetry.derivedCandidates > 0u,
                          "qualification-plot-derived-candidates-invalid");
        requirePlotMetric(qualificationPlotTelemetry.derivedHits > 0u,
                          "qualification-plot-derived-hits-invalid");
        const std::size_t drawableFaultIndex = static_cast<std::size_t>(
            ChromaspaceMetalQualification::ActionKind::
                InjectDrawableUnavailable);
        const std::size_t priorFaultIndex = static_cast<std::size_t>(
            ChromaspaceMetalQualification::ActionKind::
                InjectPriorGpuSubmissionFailure);
        const uint64_t expectedFaultCount64 =
            static_cast<uint64_t>(campaignSnapshot.appliedByKind[
                drawableFaultIndex]) +
            static_cast<uint64_t>(campaignSnapshot.appliedByKind[
                priorFaultIndex]);
        if (expectedFaultCount64 <= std::numeric_limits<uint32_t>::max()) {
          const uint32_t expectedFaultCount =
              static_cast<uint32_t>(expectedFaultCount64);
          std::string finishError;
          faultFinished = qualificationFaultBackend.finish(
              {expectedFaultCount, expectedFaultCount == 2u}, &finishError);
          rememberError(finishError);
        } else {
          rememberError("qualification-fault-count-overflow");
        }

        const std::size_t replaceIndex = static_cast<std::size_t>(
            ChromaspaceMetalQualification::ActionKind::ReplaceSource);
        const std::size_t clearIndex = static_cast<std::size_t>(
            ChromaspaceMetalQualification::ActionKind::ClearSource);
        const uint64_t expectedPublishCount64 =
            1u + static_cast<uint64_t>(
                      campaignSnapshot.appliedByKind[replaceIndex]);
        const uint64_t expectedClearCount64 =
            1u + static_cast<uint64_t>(
                      campaignSnapshot.appliedByKind[clearIndex]);
        if (expectedPublishCount64 <= std::numeric_limits<uint32_t>::max() &&
            expectedClearCount64 <= std::numeric_limits<uint32_t>::max()) {
          const ChromaspaceMetalQualification::SourceCompletionExpectation
              sourceExpectation{
                  static_cast<uint32_t>(expectedPublishCount64),
                  static_cast<uint32_t>(expectedClearCount64), true, true};
          std::string finishError;
          sourceFinished = qualificationSourceAdapter.finish(
              sourceExpectation, &finishError);
          rememberError(finishError);
          sourceCountsCoherent =
              sourceSnapshot.publishCount == sourceExpectation.requiredPublishCount &&
              sourceSnapshot.clearCount == sourceExpectation.requiredClearCount;
        } else {
          rememberError("qualification-source-count-overflow");
        }

        nativeCountsCoherent =
            nativeSnapshot.createCount == sourceSnapshot.publishCount &&
            nativeSnapshot.retireCount == sourceSnapshot.retireCount &&
            nativeSnapshot.inFlightCount == 0u;
        if (!nativeCountsCoherent) {
          rememberError("qualification-native-source-count-mismatch");
        }
        const uint32_t appliedMemoryActions =
            campaignSnapshot.appliedByFamily[static_cast<std::size_t>(
                ChromaspaceMetalQualification::ActionFamily::MemoryPressure)];
        if (appliedMemoryActions == 0u) {
          memoryCountsCoherent = memoryPressureTelemetry.warnings == 0u &&
                                 memoryPressureTelemetry.criticals == 0u &&
                                 memoryPressureTelemetry.redraws == 0u &&
                                 memoryPressureTelemetry.trimmedSurfaces == 0u &&
                                 memoryPressureTelemetry.trimmedSurfaceBytes == 0u &&
                                 memoryPressureTelemetry.trimmedDerived == 0u &&
                                 memoryPressureTelemetry.trimmedDerivedBytes == 0u;
        } else if (appliedMemoryActions == 2u) {
          memoryCountsCoherent = memoryPressureTelemetry.warnings == 1u &&
                                 memoryPressureTelemetry.criticals == 1u &&
                                 memoryPressureTelemetry.redraws == 1u &&
                                 memoryPressureTelemetry.trimmedSurfaces > 0u &&
                                 memoryPressureTelemetry.trimmedSurfaceBytes > 0u &&
                                 memoryPressureTelemetry.trimmedDerived > 0u &&
                                 memoryPressureTelemetry.trimmedDerivedBytes > 0u;
        }
        if (!memoryCountsCoherent) {
          rememberError("qualification-memory-pressure-count-mismatch");
        }
        const uint64_t recreationCount64 = qualificationRuntimeRecreations;
        const bool recreationCountFits =
            recreationCount64 <= std::numeric_limits<uint32_t>::max();
        ChromaspaceMetalQualification::CompletionObservation observation{};
        observation.presentedFrames =
            static_cast<uint32_t>(std::max(frame, 0));
        observation.drained = drained && qualificationShutdown.drainSucceeded;
        observation.completionClean = completionClean;
        observation.runtimeRecreations =
            recreationCountFits ? static_cast<uint32_t>(recreationCount64)
                                : std::numeric_limits<uint32_t>::max();
        observation.injectedFaultsObserved = faultSnapshot.firedCount;
        observation.recoveredFaults = faultSnapshot.recoveredCount;
        std::string finishError;
        campaignFinished =
            qualificationCampaign.finish(observation, &finishError);
        rememberError(finishError);
        campaignSnapshot = qualificationCampaign.snapshot();
        faultSnapshot = qualificationFaultBackend.snapshot();
        sourceSnapshot = qualificationSourceAdapter.snapshot();
        nativeSnapshot = qualificationNativeSource.snapshot();
        qualificationPassed =
            qualificationPassed && campaignFinished && faultFinished &&
            sourceFinished && sourceCountsCoherent && nativeCountsCoherent &&
            memoryCountsCoherent && recreationCountFits && plotMetricsCoherent &&
            qualificationShutdown.drainAttempted &&
            qualificationShutdown.clientDestroyed &&
            !qualificationShutdown.alreadyShutdown &&
            qualificationShutdown.drainSucceeded;
      }
#else
      const bool qualificationPassed =
          success && frame == frameBudget && drained && completionClean &&
          qualificationMemoryAvailable &&
          qualificationRuntimeRecreations == 0u;
#endif
      const auto elapsedMilliseconds =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - qualificationStart)
              .count();
      std::cout << "qualification="
                << (qualificationPassed ? "pass" : "fail")
                << " frames=" << frame
                << " elapsed_ms=" << elapsedMilliseconds
                << " submitted=" << completionStats.submittedSerial
                << " completed=" << completionStats.completedSerial
                << " failed=" << completionStats.failedSubmissionCount
                << " gpu_timed=" << completionStats.timedSubmissionCount
                << " gpu_untimed=" << completionStats.untimedSubmissionCount
                << " gpu_total_ms="
                << completionStats.accumulatedGpuSeconds * 1000.0
                << " gpu_max_ms="
                << completionStats.maximumGpuSeconds * 1000.0
                << " transient_peak_reserved_bytes="
                << qualificationPeakReservedBytes
                << " transient_peak_logical_bytes="
                << qualificationPeakLogicalBytes
                << " transient_peak_submissions="
                << qualificationPeakActiveSubmissions
                << " runtime_recreations="
#if defined(CHROMASPACE_METAL_QUALIFICATION)
                << qualificationRuntimeRecreations
                << " runtime_context_id=" << runtime.runtimeContextId()
                << " device_registry_id=" << runtime.deviceRegistryId()
                << " scenario="
                << ChromaspaceMetalQualification::scenarioLabel(
                       qualificationConfig.scenario)
                << " workspace_profile=" << kQualificationWorkspaceProfile
                << " workspace_windows="
                << qualificationPlotTelemetry.workspaceWindows
                << " renderer_coverage_mask="
                << qualificationRendererCoverageSnapshot.coveredMask
                << " renderer_coverage_required_mask="
                << qualificationRendererCoverageSnapshot.requiredMask
                << " renderer_coverage_observations="
                << qualificationRendererCoverageSnapshot
                       .acceptedObservationCount
                << " renderer_variants_covered="
                << qualificationRendererCoverageSnapshot.coveredVariantCount
                << " renderer_coverage_complete="
                << (qualificationRendererCoverageSnapshot.complete() ? 1 : 0)
                << " plot_samples=" << qualificationPlotTelemetry.samples
                << " plot_surface_resident_peak_bytes="
                << qualificationPlotTelemetry.surfaceResidentPeakBytes
                << " plot_surface_transient_peak_bytes="
                << qualificationPlotTelemetry.surfaceTransientPeakBytes
                << " plot_derived_resident_peak_bytes="
                << qualificationPlotTelemetry.derivedResidentPeakBytes
                << " plot_derived_transient_peak_bytes="
                << qualificationPlotTelemetry.derivedTransientPeakBytes
                << " plot_content_hits="
                << qualificationPlotTelemetry.contentHits
                << " plot_derived_hits="
                << qualificationPlotTelemetry.derivedHits
                << " plot_derived_candidates="
                << qualificationPlotTelemetry.derivedCandidates
                << " plot_derived_evictions="
                << qualificationPlotTelemetry.derivedEvictions
                << " plot_surface_creates="
                << qualificationPlotTelemetry.surfaceCreates
                << " plot_surface_resizes="
                << qualificationPlotTelemetry.surfaceResizes
                << " plot_surface_replacements="
                << qualificationPlotTelemetry.surfaceReplacements
                << " plot_surface_prunes="
                << qualificationPlotTelemetry.surfacePrunes
                << " presented_cpu_samples="
                << qualificationPlotTelemetry.presentedCpuSamples
                << " presented_cpu_total_ms="
                << qualificationPlotTelemetry.presentedCpuTotalMilliseconds
                << " presented_cpu_max_ms="
                << qualificationPlotTelemetry.presentedCpuMaximumMilliseconds
                << " actions_emitted_resize="
                << campaignSnapshot.emittedByFamily[0u]
                << " actions_emitted_drawable="
                << campaignSnapshot.emittedByFamily[1u]
                << " actions_emitted_source="
                << campaignSnapshot.emittedByFamily[2u]
                << " actions_emitted_recovery_fault="
                << campaignSnapshot.emittedByFamily[3u]
                << " actions_emitted_memory="
                << campaignSnapshot.emittedByFamily[4u]
                << " actions_applied_resize="
                << campaignSnapshot.appliedByFamily[0u]
                << " actions_applied_drawable="
                << campaignSnapshot.appliedByFamily[1u]
                << " actions_applied_source="
                << campaignSnapshot.appliedByFamily[2u]
                << " actions_applied_recovery_fault="
                << campaignSnapshot.appliedByFamily[3u]
                << " actions_applied_memory="
                << campaignSnapshot.appliedByFamily[4u]
                << " faults_fired=" << faultSnapshot.firedCount
                << " faults_recovered=" << faultSnapshot.recoveredCount
                << " source_publishes=" << sourceSnapshot.publishCount
                << " source_clears=" << sourceSnapshot.clearCount
                << " source_retires=" << sourceSnapshot.retireCount
                << " native_creates=" << nativeSnapshot.createCount
                << " native_retires=" << nativeSnapshot.retireCount
                << " native_inflight=" << nativeSnapshot.inFlightCount
                << " memory_warnings=" << memoryPressureTelemetry.warnings
                << " memory_criticals=" << memoryPressureTelemetry.criticals
                << " memory_redraws=" << memoryPressureTelemetry.redraws
                << " memory_trimmed_surfaces="
                << memoryPressureTelemetry.trimmedSurfaces
                << " memory_trimmed_surface_bytes="
                << memoryPressureTelemetry.trimmedSurfaceBytes
                << " memory_trimmed_derived="
                << memoryPressureTelemetry.trimmedDerived
                << " memory_trimmed_derived_bytes="
                << memoryPressureTelemetry.trimmedDerivedBytes
                << "\n";
#else
                << qualificationRuntimeRecreations << "\n";
#endif
      if (!qualificationPassed) {
        reportError("Metal qualification did not reach a clean drained state",
                    qualificationError.empty()
                        ? completionStats.lastSubmissionError
                        : qualificationError);
        success = false;
      }
    }

    if (!ChromaspaceViewer::viewerSessionShouldClose(viewerSession)) {
      applyCocoaViewerSessionEvent(
          &viewerSession, ChromaspaceViewer::ViewerSessionCloseRequested{});
    }

#if defined(CHROMASPACE_METAL_QUALIFICATION)
    if (!qualificationActive) {
#endif
      (void)observeCocoaWorkspacePersistence(
          window, &workspace, &persistenceDocument, &workspacePersistence,
          false);
      if (!maybeAutosaveCocoaWorkspace(
              &workspacePersistence, persistenceDocument, true)) {
        reportError("could not persist viewer workspace",
                    workspacePersistence.lastDiagnostic);
        success = false;
      }
#if defined(CHROMASPACE_METAL_QUALIFICATION)
    }
#endif

    (void)commandServer.stopAndJoin();
    (void)residentSourceSession.shutdown();
    runtime.shutdown();
    [window orderOut:nil];
    [window close];
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  }
}
