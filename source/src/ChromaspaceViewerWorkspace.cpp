#include "ChromaspaceViewerWorkspace.h"

#include "ChromaspaceViewerLayout.h"
#include "ChromaspaceViewerState.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <type_traits>
#include <utility>

namespace ChromaspaceViewer {
namespace {

static_assert(std::is_nothrow_move_constructible<PlotWindowDomainState>::value,
              "workspace commit must move domain state without throwing");
static_assert(std::is_nothrow_move_assignable<PlotWindowDomainState>::value,
              "workspace commit must move domain state without throwing");

constexpr float kRectEpsilon = 1.0e-5f;

bool finite(float value) noexcept { return std::isfinite(value); }
bool finite(double value) noexcept { return std::isfinite(value); }

bool finiteRect(const PlotWindowRectNorm& rect) noexcept {
  return finite(rect.x) && finite(rect.y) && finite(rect.w) && finite(rect.h) &&
         rect.x >= -kRectEpsilon && rect.y >= -kRectEpsilon && rect.w > 0.0f &&
         rect.h > 0.0f && rect.x + rect.w <= 1.0f + kRectEpsilon &&
         rect.y + rect.h <= 1.0f + kRectEpsilon;
}

bool finiteCamera(const CameraState& camera) noexcept {
  return finite(camera.qx) && finite(camera.qy) && finite(camera.qz) &&
         finite(camera.qw) && finite(camera.distance) && finite(camera.panX) &&
         finite(camera.panY) && camera.distance > 0.0f &&
         camera.orthographicView >= -1 && camera.orthographicView <= 7;
}

bool validLassoStroke(const LassoStroke& stroke) noexcept {
  if (stroke.points.size() > kViewerWorkspaceMaxLassoPointsPerStroke) return false;
  if (stroke.boundsValid) {
    if (!finite(stroke.minXNorm) || !finite(stroke.maxXNorm) ||
        !finite(stroke.minYNorm) || !finite(stroke.maxYNorm) ||
        stroke.minXNorm < -kRectEpsilon || stroke.maxXNorm > 1.0f + kRectEpsilon ||
        stroke.minYNorm < -kRectEpsilon || stroke.maxYNorm > 1.0f + kRectEpsilon ||
        stroke.minXNorm > stroke.maxXNorm || stroke.minYNorm > stroke.maxYNorm) {
      return false;
    }
  }
  for (const LassoPointNorm& point : stroke.points) {
    if (!finite(point.xNorm) || !finite(point.yNorm) ||
        point.xNorm < -kRectEpsilon || point.xNorm > 1.0f + kRectEpsilon ||
        point.yNorm < -kRectEpsilon || point.yNorm > 1.0f + kRectEpsilon) {
      return false;
    }
  }
  return true;
}

bool validLassoStrokes(const std::vector<LassoStroke>& strokes,
                       std::size_t* totalPoints = nullptr) noexcept {
  if (strokes.size() > kViewerWorkspaceMaxLassoStrokes) return false;
  std::size_t points = 0u;
  for (const LassoStroke& stroke : strokes) {
    if (!validLassoStroke(stroke)) return false;
    if (stroke.points.size() > kViewerWorkspaceMaxLassoPoints - points) return false;
    points += stroke.points.size();
  }
  if (totalPoints) *totalPoints += points;
  return true;
}

bool validString(const std::string& value, bool allowEmpty = true) noexcept {
  return (allowEmpty || !value.empty()) &&
         value.size() <= kViewerWorkspaceMaxStringBytes;
}

bool validSerializedLasso(const std::string& value) noexcept {
  return value.size() <= kViewerWorkspaceMaxSerializedLassoBytes;
}

bool validPanel(ViewerWorkspaceToolbarPanel panel) noexcept {
  return static_cast<unsigned int>(panel) <=
         static_cast<unsigned int>(ViewerWorkspaceToolbarPanel::LayoutPreset);
}

bool validDragMode(PlotWindowDragMode mode) noexcept {
  return static_cast<unsigned int>(mode) <=
             static_cast<unsigned int>(PlotWindowDragMode::ResizeBottomRight) &&
         mode != PlotWindowDragMode::None;
}

bool findWindow(const ViewerWorkspaceState& state, int id,
                std::size_t* index = nullptr) noexcept {
  if (id <= 0) return false;
  for (std::size_t i = 0; i < state.windows.size(); ++i) {
    if (state.windows[i].windowId == id) {
      if (index) *index = i;
      return true;
    }
  }
  return false;
}

bool findShadow(const std::array<int, kViewerWorkspaceMaxWindows>& ids,
                std::size_t count,
                int id,
                std::size_t* index = nullptr) noexcept {
  if (id <= 0) return false;
  for (std::size_t i = 0; i < count; ++i) {
    if (ids[i] == id) {
      if (index) *index = i;
      return true;
    }
  }
  return false;
}

float rectArea(const PlotWindowRectNorm& rect) noexcept {
  return std::max(0.0f, rect.w) * std::max(0.0f, rect.h);
}

struct Shadow;
void sortShadow(Shadow* shadow) noexcept;

void sortWindows(ViewerWorkspaceState* state) noexcept {
  if (!state || state->windows.size() < 2u) return;
  for (std::size_t i = 1; i < state->windows.size(); ++i) {
    PlotWindowDomainState item = std::move(state->windows[i]);
    const float itemArea = rectArea(item.rect);
    std::size_t at = i;
    while (at > 0) {
      const auto& previous = state->windows[at - 1];
      const float previousArea = rectArea(previous.rect);
      bool before = false;
      if (std::fabs(previousArea - itemArea) > kRectEpsilon) {
        before = previousArea < itemArea;
      } else if (previous.windowId == state->focusedWindowId &&
                 item.windowId != state->focusedWindowId) {
        before = true;
      }
      if (!before) break;
      state->windows[at] = std::move(state->windows[at - 1]);
      --at;
    }
    state->windows[at] = std::move(item);
  }
}

bool bump(uint64_t* value) noexcept {
  if (!value || *value == std::numeric_limits<uint64_t>::max()) return false;
  ++(*value);
  return *value != 0u;
}

bool addEffect(ViewerWorkspaceEffectBatch* batch,
               ViewerWorkspaceEffectKind kind,
               int windowId = -1,
               PlotWindowDragMode mode = PlotWindowDragMode::None,
               PlotWindowRectNorm rect = {},
               float pointerX = 0.0f,
               float pointerY = 0.0f,
               bool enabled = false) noexcept {
  if (!batch) return false;
  if (kind == ViewerWorkspaceEffectKind::RefreshReinterpret) {
    // A resample is the stronger refresh policy and already subsumes an
    // earlier reinterpret request.
    if (batch->contains(ViewerWorkspaceEffectKind::RefreshReinterpret) ||
        batch->contains(ViewerWorkspaceEffectKind::RefreshResample)) {
      return true;
    }
  }
  if (kind == ViewerWorkspaceEffectKind::RefreshResample) {
    if (batch->contains(ViewerWorkspaceEffectKind::RefreshResample)) return true;
    for (std::size_t i = 0; i < batch->count; ++i) {
      if (batch->effects[i].kind == ViewerWorkspaceEffectKind::RefreshReinterpret) {
        batch->effects[i].kind = ViewerWorkspaceEffectKind::RefreshResample;
        return true;
      }
    }
  }
  if (kind == ViewerWorkspaceEffectKind::PersistSuggested &&
      batch->contains(kind)) {
    return true;
  }
  if (batch->count >= batch->effects.size()) {
    batch->capacityExceeded = true;
    return false;
  }
  ViewerWorkspaceEffect& effect = batch->effects[batch->count++];
  effect.kind = kind;
  effect.windowId = windowId;
  effect.dragMode = mode;
  effect.rect = rect;
  effect.pointerX = pointerX;
  effect.pointerY = pointerY;
  effect.enabled = enabled;
  return true;
}

bool isSourceSignal(const PlotWindowDomainState& window) noexcept {
  return window.viewState.plotModel == kPlotModelSourceSignal;
}

bool isValidCommandKind(ViewerControllerCommandKind kind) noexcept {
  return static_cast<unsigned int>(kind) <=
         static_cast<unsigned int>(ViewerControllerCommandKind::SourceSignalRestore);
}

struct Shadow {
  std::array<int, kViewerWorkspaceMaxWindows> ids{};
  std::array<PlotWindowRectNorm, kViewerWorkspaceMaxWindows> rects{};
  std::array<uint64_t, kViewerWorkspaceMaxWindows> revisions{};
  std::array<uint64_t, kViewerWorkspaceMaxWindows> lassoRevisions{};
  std::array<std::size_t, kViewerWorkspaceMaxWindows> lassoStrokeCounts{};
  std::array<std::array<bool, kViewerWorkspaceSlicingVectorCount>,
             kViewerWorkspaceMaxWindows>
      slicing{};
  std::array<bool, kViewerWorkspaceMaxWindows> lasso{};
  std::array<bool, kViewerWorkspaceMaxWindows> sourceSignal{};
  std::array<bool, kViewerWorkspaceMaxWindows> sourceSignalDocked{};
  std::array<int, kViewerWorkspaceMaxWindows> sourceSignalDockOwner{};
  std::array<PlotWindowRectNorm, kViewerWorkspaceMaxWindows>
      sourceSignalRestoreRect{};
  std::array<bool, kViewerWorkspaceMaxWindows> refresh{};
  std::array<bool, kViewerWorkspaceMaxWindows> clearViewerLasso{};
  std::array<bool, kViewerWorkspaceMaxWindows> drawer{};
  std::size_t count = 0u;
  int focused = -1;
  int nextId = 1;
  ViewerWorkspaceToolbarPanel panel = ViewerWorkspaceToolbarPanel::None;
  float panelX = 0.0f;
  float panelY = 0.0f;
  bool dragActive = false;
  int dragWindow = -1;
  PlotWindowDragMode dragMode = PlotWindowDragMode::None;
  float dragX = 0.0f;
  float dragY = 0.0f;
  PlotWindowRectNorm dragRect{};
  bool sourceSubtract = false;
  bool sourceHasSelection = false;
  bool globalSourceHasSelection = false;
  bool sourceSelectionsSynced = false;
  int sourceTarget = -1;
  bool sourceSessionActive = false;
  uint64_t sourceRevision = 0u;
  std::size_t sourceStrokeCount = 0u;
  uint64_t revision = 1u;
  bool changed = false;
  bool layoutCustom = false;
  ViewerWorkspaceReduceStatus failureStatus =
      ViewerWorkspaceReduceStatus::InvalidCommand;
};

void sortShadow(Shadow* shadow) noexcept {
  if (!shadow) return;
  for (std::size_t i = 1; i < shadow->count; ++i) {
    const int id = shadow->ids[i];
    const PlotWindowRectNorm rect = shadow->rects[i];
    const uint64_t revision = shadow->revisions[i];
    const uint64_t lassoRevision = shadow->lassoRevisions[i];
    const std::size_t lassoStrokeCount = shadow->lassoStrokeCounts[i];
    const auto slicing = shadow->slicing[i];
    const bool lasso = shadow->lasso[i];
    const bool sourceSignal = shadow->sourceSignal[i];
    const bool sourceSignalDocked = shadow->sourceSignalDocked[i];
    const int sourceSignalDockOwner = shadow->sourceSignalDockOwner[i];
    const PlotWindowRectNorm sourceSignalRestoreRect =
        shadow->sourceSignalRestoreRect[i];
    const bool refresh = shadow->refresh[i];
    const bool clearViewerLasso = shadow->clearViewerLasso[i];
    const bool drawer = shadow->drawer[i];
    const float area = rectArea(rect);
    std::size_t at = i;
    while (at > 0u) {
      const float previousArea = rectArea(shadow->rects[at - 1u]);
      bool before = false;
      if (std::fabs(previousArea - area) > kRectEpsilon) {
        before = previousArea < area;
      } else if (shadow->ids[at - 1u] == shadow->focused && id != shadow->focused) {
        before = true;
      }
      if (!before) break;
      shadow->ids[at] = shadow->ids[at - 1u];
      shadow->rects[at] = shadow->rects[at - 1u];
      shadow->revisions[at] = shadow->revisions[at - 1u];
      shadow->lassoRevisions[at] = shadow->lassoRevisions[at - 1u];
      shadow->lassoStrokeCounts[at] =
          shadow->lassoStrokeCounts[at - 1u];
      shadow->slicing[at] = shadow->slicing[at - 1u];
      shadow->lasso[at] = shadow->lasso[at - 1u];
      shadow->sourceSignal[at] = shadow->sourceSignal[at - 1u];
      shadow->sourceSignalDocked[at] = shadow->sourceSignalDocked[at - 1u];
      shadow->sourceSignalDockOwner[at] =
          shadow->sourceSignalDockOwner[at - 1u];
      shadow->sourceSignalRestoreRect[at] =
          shadow->sourceSignalRestoreRect[at - 1u];
      shadow->refresh[at] = shadow->refresh[at - 1u];
      shadow->clearViewerLasso[at] = shadow->clearViewerLasso[at - 1u];
      shadow->drawer[at] = shadow->drawer[at - 1u];
      --at;
    }
    shadow->ids[at] = id;
    shadow->rects[at] = rect;
    shadow->revisions[at] = revision;
    shadow->lassoRevisions[at] = lassoRevision;
    shadow->lassoStrokeCounts[at] = lassoStrokeCount;
    shadow->slicing[at] = slicing;
    shadow->lasso[at] = lasso;
    shadow->sourceSignal[at] = sourceSignal;
    shadow->sourceSignalDocked[at] = sourceSignalDocked;
    shadow->sourceSignalDockOwner[at] = sourceSignalDockOwner;
    shadow->sourceSignalRestoreRect[at] = sourceSignalRestoreRect;
    shadow->refresh[at] = refresh;
    shadow->clearViewerLasso[at] = clearViewerLasso;
    shadow->drawer[at] = drawer;
  }
}

bool buildShadow(const ViewerWorkspaceState& state, Shadow* shadow) noexcept {
  if (!shadow || state.windows.size() > kViewerWorkspaceMaxWindows) return false;
  shadow->count = state.windows.size();
  shadow->focused = state.focusedWindowId;
  shadow->nextId = state.nextWindowId;
  shadow->panel = state.activeToolbarPanel;
  shadow->panelX = state.toolbarPanelAnchorX;
  shadow->panelY = state.toolbarPanelAnchorY;
  shadow->dragActive = state.windowDragActive;
  shadow->dragWindow = state.windowDragWindowId;
  shadow->dragMode = state.windowDragMode;
  shadow->dragX = state.windowDragStartX;
  shadow->dragY = state.windowDragStartY;
  shadow->dragRect = state.windowDragStartRect;
  shadow->sourceSubtract = state.sourceLassoSubtractMode;
  shadow->sourceHasSelection = state.sourceLassoHasSelection;
  shadow->globalSourceHasSelection = state.sourceLassoGlobalHasSelection;
  shadow->sourceSelectionsSynced = state.sourceLassoSelectionsSynced;
  shadow->sourceTarget = state.sourceLassoTargetWindowId;
  shadow->sourceSessionActive = state.sourceLassoSessionActive;
  shadow->sourceRevision = state.sourceLassoRevision;
  shadow->sourceStrokeCount = state.sourceLassoStrokes.size();
  shadow->revision = state.revision;
  for (std::size_t i = 0; i < shadow->count; ++i) {
    const auto& window = state.windows[i];
    shadow->ids[i] = window.windowId;
    shadow->rects[i] = window.rect;
    shadow->revisions[i] = window.viewState.stateRevision;
    shadow->lassoRevisions[i] = window.viewerLassoRevision;
    shadow->lassoStrokeCounts[i] = window.viewerLassoStrokes.size();
    shadow->lasso[i] = window.viewState.volumeSliceLassoRegion;
    shadow->sourceSignal[i] = isSourceSignal(window);
    shadow->sourceSignalDocked[i] = window.sourceSignalDocked;
    shadow->sourceSignalDockOwner[i] = window.sourceSignalDockOwnerWindowId;
    shadow->sourceSignalRestoreRect[i] = window.sourceSignalRestoreRect;
    shadow->refresh[i] = false;
    shadow->clearViewerLasso[i] = false;
    shadow->drawer[i] = window.slicingDrawerOpen;
    shadow->slicing[i] = {{window.viewState.volumeSliceRed,
                           window.viewState.volumeSliceYellow,
                           window.viewState.volumeSliceGreen,
                           window.viewState.volumeSliceCyan,
                           window.viewState.volumeSliceBlue,
                           window.viewState.volumeSliceMagenta}};
  }
  return true;
}

bool bumpShadowRevision(Shadow* shadow) noexcept {
  if (!shadow) return false;
  if (!bump(&shadow->revision)) {
    shadow->failureStatus = ViewerWorkspaceReduceStatus::RevisionOverflow;
    return false;
  }
  shadow->changed = true;
  return true;
}

bool bumpWindowShadowRevision(Shadow* shadow, std::size_t index) noexcept {
  if (!shadow || index >= shadow->count) return false;
  if (!bump(&shadow->revisions[index])) {
    shadow->failureStatus = ViewerWorkspaceReduceStatus::RevisionOverflow;
    return false;
  }
  return bumpShadowRevision(shadow);
}

bool markCustom(Shadow* shadow) noexcept {
  if (!shadow) return false;
  shadow->layoutCustom = true;
  return bumpShadowRevision(shadow);
}

bool validPointer(float x, float y) noexcept { return finite(x) && finite(y); }

bool validateCommandFields(const ViewerControllerCommand& command,
                           const Shadow& shadow) noexcept {
  using Kind = ViewerControllerCommandKind;
  switch (command.kind) {
    case Kind::None:
    case Kind::ToolbarMenu:
    case Kind::ToolbarAddPlot:
    case Kind::ToolbarLayoutPreset:
      return command.kind == Kind::None ||
             (command.windowId == -1 && validPointer(command.pointerX, command.pointerY));
    case Kind::FocusWindow:
      return findShadow(shadow.ids, shadow.count, command.windowId);
    case Kind::RequestCloseWindow:
      return findShadow(shadow.ids, shadow.count, command.windowId);
    case Kind::BeginWindowDrag:
      return !shadow.dragActive && findShadow(shadow.ids, shadow.count, command.windowId) &&
             validDragMode(command.dragMode) && finiteRect(command.rect) &&
             validPointer(command.pointerX, command.pointerY);
    case Kind::UpdateWindowDrag:
    case Kind::EndWindowDrag:
      return shadow.dragActive && command.windowId == shadow.dragWindow &&
             command.dragMode == shadow.dragMode && finiteRect(command.rect) &&
             validPointer(command.pointerX, command.pointerY);
    case Kind::ToggleSlicingDrawer:
    case Kind::ToggleSlicingLasso:
      return findShadow(shadow.ids, shadow.count, command.windowId);
    case Kind::SetSlicingVector:
      return findShadow(shadow.ids, shadow.count, command.windowId) &&
             command.controlIndex >= 0 &&
             command.controlIndex < kViewerWorkspaceSlicingVectorCount;
    case Kind::SoloSlicingVector:
      return findShadow(shadow.ids, shadow.count, command.windowId) &&
             command.controlIndex >= 0 &&
             command.controlIndex < kViewerWorkspaceSlicingVectorCount;
    case Kind::ToggleAllSlicingVectors:
      return findShadow(shadow.ids, shadow.count, command.windowId);
    case Kind::SourceLassoAdd:
    case Kind::SourceLassoSubtract:
    case Kind::SourceLassoClear:
    case Kind::SourceLassoUndo:
    case Kind::SourceSignalRestore:
      return command.windowId == -1 ||
             findShadow(shadow.ids, shadow.count, command.windowId);
  }
  return false;
}

bool preflightCommand(const ViewerControllerCommand& command,
                      Shadow* shadow,
                      ViewerWorkspaceEffectBatch* effects) noexcept {
  using Kind = ViewerControllerCommandKind;
  if (!shadow || !effects || !isValidCommandKind(command.kind) ||
      !validateCommandFields(command, *shadow)) {
    return false;
  }
  std::size_t index = 0u;
  switch (command.kind) {
    case Kind::None:
      return true;
    case Kind::FocusWindow:
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      if (shadow->focused == command.windowId) return true;
      shadow->focused = command.windowId;
      sortShadow(shadow);
      return bumpShadowRevision(shadow);
    case Kind::RequestCloseWindow: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      if (shadow->count <= 1u) return false;
      if (shadow->sourceSessionActive && shadow->sourceSignal[index]) {
        if (shadow->sourceSignalDocked[index]) return true;
        std::size_t ownerIndex = 0u;
        int ownerId = shadow->sourceTarget;
        if (ownerId <= 0 ||
            !findShadow(shadow->ids, shadow->count, ownerId, &ownerIndex) ||
            shadow->sourceSignal[ownerIndex]) {
          ownerId = shadow->focused;
          if (ownerId <= 0 ||
              !findShadow(shadow->ids, shadow->count, ownerId, &ownerIndex) ||
              shadow->sourceSignal[ownerIndex]) {
            ownerId = -1;
            for (std::size_t candidate = 0u; candidate < shadow->count;
                 ++candidate) {
              if (!shadow->sourceSignal[candidate]) {
                ownerId = shadow->ids[candidate];
                ownerIndex = candidate;
                break;
              }
            }
          }
        }
        if (ownerId <= 0) {
          shadow->failureStatus = ViewerWorkspaceReduceStatus::InvalidState;
          return false;
        }
        shadow->sourceSignalRestoreRect[index] = shadow->rects[index];
        shadow->sourceSignalDockOwner[index] = ownerId;
        shadow->sourceSignalDocked[index] = true;
        if (!bumpShadowRevision(shadow)) return false;
        return addEffect(effects,
                         ViewerWorkspaceEffectKind::DockSourceSignalForLasso,
                         command.windowId) &&
               addEffect(effects,
                         ViewerWorkspaceEffectKind::ReleaseWindowResources,
                         command.windowId) &&
               addEffect(effects,
                         ViewerWorkspaceEffectKind::RefreshReinterpret);
      }
      for (std::size_t docked = 0u; docked < shadow->count; ++docked) {
        if (shadow->sourceSignalDocked[docked] &&
            shadow->sourceSignalDockOwner[docked] == command.windowId) {
          shadow->sourceSignalDocked[docked] = false;
          shadow->sourceSignalDockOwner[docked] = -1;
          shadow->rects[docked] = shadow->sourceSignalRestoreRect[docked];
        }
      }
      for (std::size_t j = index + 1u; j < shadow->count; ++j) {
        shadow->ids[j - 1u] = shadow->ids[j];
        shadow->rects[j - 1u] = shadow->rects[j];
        shadow->revisions[j - 1u] = shadow->revisions[j];
        shadow->lassoRevisions[j - 1u] = shadow->lassoRevisions[j];
        shadow->lassoStrokeCounts[j - 1u] = shadow->lassoStrokeCounts[j];
        shadow->slicing[j - 1u] = shadow->slicing[j];
        shadow->lasso[j - 1u] = shadow->lasso[j];
        shadow->refresh[j - 1u] = shadow->refresh[j];
        shadow->clearViewerLasso[j - 1u] = shadow->clearViewerLasso[j];
        shadow->drawer[j - 1u] = shadow->drawer[j];
        shadow->sourceSignal[j - 1u] = shadow->sourceSignal[j];
        shadow->sourceSignalDocked[j - 1u] = shadow->sourceSignalDocked[j];
        shadow->sourceSignalDockOwner[j - 1u] =
            shadow->sourceSignalDockOwner[j];
        shadow->sourceSignalRestoreRect[j - 1u] =
            shadow->sourceSignalRestoreRect[j];
      }
      --shadow->count;
      if (shadow->focused == command.windowId) {
        shadow->focused = shadow->ids[shadow->count - 1u];
      }
      if (shadow->sourceTarget == command.windowId) {
        // The target-owned lasso cannot outlive its owner.  Fall back to the
        // retained global source selection without copying any lasso points.
        shadow->sourceTarget = -1;
        shadow->sourceHasSelection = shadow->globalSourceHasSelection;
      }
      sortShadow(shadow);
      if (!markCustom(shadow)) return false;
      return addEffect(effects, ViewerWorkspaceEffectKind::ReleaseWindowResources,
                       command.windowId) &&
             addEffect(effects, ViewerWorkspaceEffectKind::PersistSuggested) &&
             addEffect(effects, ViewerWorkspaceEffectKind::RefreshResample);
    }
    case Kind::BeginWindowDrag:
      shadow->dragActive = true;
      shadow->dragWindow = command.windowId;
      shadow->dragMode = command.dragMode;
      shadow->dragX = command.pointerX;
      shadow->dragY = command.pointerY;
      shadow->dragRect = command.rect;
      return bumpShadowRevision(shadow);
    case Kind::UpdateWindowDrag:
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      shadow->rects[index] = command.rect;
      return bumpShadowRevision(shadow) &&
             addEffect(effects, ViewerWorkspaceEffectKind::SnapPreviewUpdated,
                       command.windowId, command.dragMode, command.rect,
                       command.pointerX, command.pointerY);
    case Kind::EndWindowDrag:
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      shadow->rects[index] = command.rect;
      shadow->dragActive = false;
      shadow->dragWindow = -1;
      shadow->dragMode = PlotWindowDragMode::None;
      shadow->dragX = 0.0f;
      shadow->dragY = 0.0f;
      shadow->dragRect = {};
      sortShadow(shadow);
      if (!markCustom(shadow)) return false;
      return addEffect(effects, ViewerWorkspaceEffectKind::SnapCommitted,
                       command.windowId, command.dragMode, command.rect,
                       command.pointerX, command.pointerY) &&
             addEffect(effects, ViewerWorkspaceEffectKind::PersistSuggested);
    case Kind::ToolbarMenu: {
      const auto next = shadow->panel == ViewerWorkspaceToolbarPanel::MainMenu
                            ? ViewerWorkspaceToolbarPanel::None
                            : ViewerWorkspaceToolbarPanel::MainMenu;
      if (next != shadow->panel) {
        shadow->panel = next;
        shadow->panelX = command.pointerX;
        shadow->panelY = command.pointerY;
        if (!bumpShadowRevision(shadow)) return false;
      }
      return next != ViewerWorkspaceToolbarPanel::MainMenu ||
             addEffect(effects, ViewerWorkspaceEffectKind::MainMenuOpened);
    }
    case Kind::ToolbarAddPlot: {
      const auto next = shadow->panel == ViewerWorkspaceToolbarPanel::AddPlot
                            ? ViewerWorkspaceToolbarPanel::None
                            : ViewerWorkspaceToolbarPanel::AddPlot;
      if (next != shadow->panel) {
        shadow->panel = next;
        shadow->panelX = command.pointerX;
        shadow->panelY = command.pointerY;
        if (!bumpShadowRevision(shadow)) return false;
      }
      return true;
    }
    case Kind::ToolbarLayoutPreset: {
      const auto next = shadow->panel == ViewerWorkspaceToolbarPanel::LayoutPreset
                            ? ViewerWorkspaceToolbarPanel::None
                            : ViewerWorkspaceToolbarPanel::LayoutPreset;
      if (next != shadow->panel) {
        shadow->panel = next;
        shadow->panelX = command.pointerX;
        shadow->panelY = command.pointerY;
        if (!bumpShadowRevision(shadow)) return false;
      }
      return true;
    }
    case Kind::ToggleSlicingDrawer: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      const bool next = !shadow->drawer[index];
      bool changed = false;
      for (std::size_t i = 0; i < shadow->count; ++i) {
        const bool value = i == index ? next : false;
        changed = changed || shadow->drawer[i] != value;
        shadow->drawer[i] = value;
      }
      if (changed && !bumpShadowRevision(shadow)) return false;
      return !changed ||
             addEffect(effects, ViewerWorkspaceEffectKind::SlicingDrawerChanged,
                       command.windowId, PlotWindowDragMode::None, {}, 0.0f,
                       0.0f, next);
    }
    case Kind::SetSlicingVector: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      const bool next = command.enabled;
      if (shadow->slicing[index][static_cast<std::size_t>(command.controlIndex)] == next) return true;
      shadow->slicing[index][static_cast<std::size_t>(command.controlIndex)] = next;
      shadow->refresh[index] = true;
      if (!bumpWindowShadowRevision(shadow, index)) return false;
      return addEffect(effects, ViewerWorkspaceEffectKind::RefreshReinterpret);
    }
    case Kind::SoloSlicingVector: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      bool changed = false;
      for (int i = 0; i < kViewerWorkspaceSlicingVectorCount; ++i) {
        const bool next = i == command.controlIndex;
        changed = changed || shadow->slicing[index][static_cast<std::size_t>(i)] != next;
        shadow->slicing[index][static_cast<std::size_t>(i)] = next;
      }
      if (!changed) return true;
      shadow->refresh[index] = true;
      if (!bumpWindowShadowRevision(shadow, index)) return false;
      return addEffect(effects, ViewerWorkspaceEffectKind::RefreshReinterpret);
    }
    case Kind::ToggleAllSlicingVectors: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      bool all = true;
      for (bool value : shadow->slicing[index]) all = all && value;
      const bool next = !all;
      bool changed = false;
      for (bool& value : shadow->slicing[index]) {
        changed = changed || value != next;
        value = next;
      }
      if (!changed) return true;
      shadow->refresh[index] = true;
      if (!bumpWindowShadowRevision(shadow, index)) return false;
      return addEffect(effects, ViewerWorkspaceEffectKind::RefreshReinterpret);
    }
    case Kind::ToggleSlicingLasso: {
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index)) return false;
      shadow->lasso[index] = !shadow->lasso[index];
      shadow->refresh[index] = true;
      if (!bumpWindowShadowRevision(shadow, index)) return false;
      const bool enabled = shadow->lasso[index];
      return addEffect(effects, ViewerWorkspaceEffectKind::SlicingLassoChanged,
                       command.windowId, PlotWindowDragMode::None, {}, 0.0f,
                       0.0f, enabled) &&
             addEffect(effects, enabled
                           ? ViewerWorkspaceEffectKind::RefreshResample
                           : ViewerWorkspaceEffectKind::RefreshReinterpret);
    }
    case Kind::SourceLassoAdd:
      if (shadow->sourceSubtract) {
        shadow->sourceSubtract = false;
        if (!bumpShadowRevision(shadow)) return false;
      }
      return true;
    case Kind::SourceLassoSubtract:
      if (!shadow->sourceSubtract) {
        shadow->sourceSubtract = true;
        if (!bumpShadowRevision(shadow)) return false;
      }
      return true;
    case Kind::SourceLassoClear:
      // Clearing either source scope removes the active-selection truth.  The
      // unsynchronised path clears the target window's retained lasso below,
      // while the global path advances the source revision and clears the
      // retained source strokes during commit.
      shadow->sourceHasSelection = false;
      if (shadow->sourceSelectionsSynced || shadow->sourceTarget <= 0) {
        shadow->globalSourceHasSelection = false;
        shadow->sourceStrokeCount = 0u;
        if (shadow->sourceRevision ==
                                            std::numeric_limits<uint64_t>::max()) {
          return false;
        }
        shadow->sourceHasSelection = false;
        if (!bump(&shadow->sourceRevision)) {
          shadow->failureStatus = ViewerWorkspaceReduceStatus::RevisionOverflow;
          return false;
        }
      } else {
        if (!findShadow(shadow->ids, shadow->count, shadow->sourceTarget,
                        &index)) return false;
        if (shadow->lassoRevisions[index] ==
            std::numeric_limits<uint64_t>::max()) {
          shadow->failureStatus = ViewerWorkspaceReduceStatus::RevisionOverflow;
          return false;
        }
        ++shadow->lassoRevisions[index];
        if (!bumpWindowShadowRevision(shadow, index)) return false;
        shadow->lassoStrokeCounts[index] = 0u;
        shadow->clearViewerLasso[index] = true;
      }
      if (shadow->sourceSelectionsSynced || shadow->sourceTarget <= 0) {
        if (!bumpShadowRevision(shadow)) return false;
      }
      return addEffect(effects, ViewerWorkspaceEffectKind::ClearSourceLasso,
                       shadow->sourceSelectionsSynced || shadow->sourceTarget <= 0
                           ? -1
                           : shadow->sourceTarget) &&
             addEffect(effects, ViewerWorkspaceEffectKind::RefreshReinterpret);
    case Kind::SourceLassoUndo: {
      const bool global = shadow->sourceSelectionsSynced ||
                          shadow->sourceTarget <= 0;
      if (global) {
        if (shadow->sourceStrokeCount == 0u) return true;
        if (shadow->sourceRevision == std::numeric_limits<uint64_t>::max()) {
          shadow->failureStatus =
              ViewerWorkspaceReduceStatus::RevisionOverflow;
          return false;
        }
        --shadow->sourceStrokeCount;
        ++shadow->sourceRevision;
        shadow->globalSourceHasSelection = shadow->sourceStrokeCount != 0u;
        shadow->sourceHasSelection = shadow->globalSourceHasSelection;
        if (!bumpShadowRevision(shadow)) return false;
        return addEffect(effects,
                         ViewerWorkspaceEffectKind::RefreshReinterpret);
      }

      if (!findShadow(shadow->ids, shadow->count, shadow->sourceTarget,
                      &index)) {
        shadow->failureStatus = ViewerWorkspaceReduceStatus::InvalidCommand;
        return false;
      }
      if (shadow->lassoStrokeCounts[index] == 0u) return true;
      if (shadow->lassoRevisions[index] ==
          std::numeric_limits<uint64_t>::max()) {
        shadow->failureStatus = ViewerWorkspaceReduceStatus::RevisionOverflow;
        return false;
      }
      --shadow->lassoStrokeCounts[index];
      ++shadow->lassoRevisions[index];
      shadow->sourceHasSelection = shadow->lassoStrokeCounts[index] != 0u;
      if (!bumpWindowShadowRevision(shadow, index)) return false;
      return addEffect(effects,
                       ViewerWorkspaceEffectKind::RefreshReinterpret);
    }
    case Kind::SourceSignalRestore:
      if (!findShadow(shadow->ids, shadow->count, command.windowId, &index) ||
          !shadow->sourceSignal[index] ||
          !shadow->sourceSignalDocked[index] ||
          !finiteRect(shadow->sourceSignalRestoreRect[index])) {
        return false;
      }
      shadow->sourceSignalDocked[index] = false;
      shadow->sourceSignalDockOwner[index] = -1;
      shadow->rects[index] = shadow->sourceSignalRestoreRect[index];
      if (!bumpShadowRevision(shadow)) return false;
      return addEffect(effects,
                       ViewerWorkspaceEffectKind::RefreshReinterpret);
  }
  return false;
}

void setWindowSlicingVector(ViewerRuntimeState* state,
                            int index,
                            bool enabled) noexcept {
  (void)setSlicingVectorEnabled(state, index, enabled);
}

void applyShadow(ViewerWorkspaceState* state,
                 const Shadow& shadow,
                 std::string* customSelection,
                 std::string* customNameInput,
                 bool captureCustomName) noexcept {
  state->focusedWindowId = shadow.focused;
  state->nextWindowId = shadow.nextId;
  state->activeToolbarPanel = shadow.panel;
  state->toolbarPanelAnchorX = shadow.panelX;
  state->toolbarPanelAnchorY = shadow.panelY;
  state->windowDragActive = shadow.dragActive;
  state->windowDragWindowId = shadow.dragWindow;
  state->windowDragMode = shadow.dragMode;
  state->windowDragStartX = shadow.dragX;
  state->windowDragStartY = shadow.dragY;
  state->windowDragStartRect = shadow.dragRect;
  state->sourceLassoSubtractMode = shadow.sourceSubtract;
  state->sourceLassoHasSelection = shadow.sourceHasSelection;
  state->sourceLassoGlobalHasSelection = shadow.globalSourceHasSelection;
  state->sourceLassoSelectionsSynced = shadow.sourceSelectionsSynced;
  state->sourceLassoTargetWindowId = shadow.sourceTarget;
  state->sourceLassoRevision = shadow.sourceRevision;
  if (shadow.layoutCustom && customSelection != nullptr) {
    state->layoutPresetSelection.swap(*customSelection);
    if (captureCustomName && customNameInput != nullptr) {
      state->layoutPresetNameInput.swap(*customNameInput);
    }
  }
  state->revision = shadow.revision;
}

void applySlicingShadow(ViewerWorkspaceState* state,
                        const Shadow& shadow,
                        std::array<std::string, kViewerWorkspaceMaxWindows>*
                            canonicalLassoData,
                        const std::array<bool, kViewerWorkspaceMaxWindows>&
                            replaceLassoData) noexcept {
  for (std::size_t i = 0; i < state->windows.size(); ++i) {
    std::size_t shadowIndex = 0u;
    if (!findShadow(shadow.ids, shadow.count, state->windows[i].windowId,
                    &shadowIndex)) {
      continue;
    }
    auto& runtime = state->windows[i].viewState;
    setWindowSlicingVector(&runtime, 0, shadow.slicing[shadowIndex][0]);
    setWindowSlicingVector(&runtime, 1, shadow.slicing[shadowIndex][1]);
    setWindowSlicingVector(&runtime, 2, shadow.slicing[shadowIndex][2]);
    setWindowSlicingVector(&runtime, 3, shadow.slicing[shadowIndex][3]);
    setWindowSlicingVector(&runtime, 4, shadow.slicing[shadowIndex][4]);
    setWindowSlicingVector(&runtime, 5, shadow.slicing[shadowIndex][5]);
    runtime.volumeSliceLassoRegion = shadow.lasso[shadowIndex];
    runtime.stateRevision = shadow.revisions[shadowIndex];
    if (shadow.refresh[shadowIndex]) {
      runtime.refreshPolicy = "reinterpret";
      runtime.requiresHostSamples = false;
      runtime.hostRefreshRequestedRevision = 0u;
    }
    if (state->windows[i].viewerLassoStrokes.size() !=
        shadow.lassoStrokeCounts[shadowIndex]) {
      state->windows[i].viewerLassoStrokes.resize(
          shadow.lassoStrokeCounts[shadowIndex]);
    }
    if (replaceLassoData[shadowIndex] && canonicalLassoData != nullptr) {
      state->windows[i].viewerLassoData.swap(
          (*canonicalLassoData)[shadowIndex]);
      state->windows[i].viewerLassoRevision = shadow.lassoRevisions[shadowIndex];
    }
    state->windows[i].sourceSignalDocked =
        shadow.sourceSignalDocked[shadowIndex];
    state->windows[i].sourceSignalDockOwnerWindowId =
        shadow.sourceSignalDockOwner[shadowIndex];
    state->windows[i].sourceSignalRestoreRect =
        shadow.sourceSignalRestoreRect[shadowIndex];
    if (!state->windows[i].sourceSignalDocked) {
      state->windows[i].sourceSignalDockAnimatingToDock = false;
      state->windows[i].sourceSignalDockAnimStart = -10.0;
    }
    state->windows[i].slicingDrawerOpen = shadow.drawer[shadowIndex];
    state->windows[i].selected = state->windows[i].windowId == state->focusedWindowId;
  }
}

}  // namespace

bool ViewerWorkspaceEffectBatch::contains(ViewerWorkspaceEffectKind kind) const noexcept {
  for (std::size_t i = 0; i < count; ++i) {
    if (effects[i].kind == kind) return true;
  }
  return false;
}

bool slicingVectorEnabled(const ViewerRuntimeState& state, int index) noexcept {
  switch (index) {
    case 0: return state.volumeSliceRed;
    case 1: return state.volumeSliceYellow;
    case 2: return state.volumeSliceGreen;
    case 3: return state.volumeSliceCyan;
    case 4: return state.volumeSliceBlue;
    case 5: return state.volumeSliceMagenta;
    default: return false;
  }
}

bool setSlicingVectorEnabled(ViewerRuntimeState* state,
                             int index,
                             bool enabled) noexcept {
  if (!state || index < 0 || index >= kViewerWorkspaceSlicingVectorCount) {
    return false;
  }
  bool* destination = nullptr;
  switch (index) {
    case 0: destination = &state->volumeSliceRed; break;
    case 1: destination = &state->volumeSliceYellow; break;
    case 2: destination = &state->volumeSliceGreen; break;
    case 3: destination = &state->volumeSliceCyan; break;
    case 4: destination = &state->volumeSliceBlue; break;
    case 5: destination = &state->volumeSliceMagenta; break;
    default: break;
  }
  if (!destination) return false;
  const bool changed = *destination != enabled;
  *destination = enabled;
  return changed;
}

bool allSlicingVectorsEnabled(const ViewerRuntimeState& state) noexcept {
  for (int index = 0; index < kViewerWorkspaceSlicingVectorCount; ++index) {
    if (!slicingVectorEnabled(state, index)) return false;
  }
  return true;
}

bool validateViewerWorkspaceState(const ViewerWorkspaceState& state) noexcept {
  if (state.windows.empty() || state.windows.size() > kViewerWorkspaceMaxWindows ||
      state.focusedWindowId <= 0 || state.nextWindowId <= 0 ||
      !validString(state.layoutPresetSelection, false) ||
      !validString(state.layoutPresetBeforeSolo) ||
      !validString(state.layoutPresetNameInput) || !validPanel(state.activeToolbarPanel) ||
      !validLassoStrokes(state.sourceLassoStrokes)) {
    return false;
  }
  if (!finite(state.toolbarPanelAnchorX) || !finite(state.toolbarPanelAnchorY) ||
      !finite(state.windowDragStartX) || !finite(state.windowDragStartY)) return false;
  if (state.sourceLassoTargetWindowId > 0 &&
      !findWindow(state, state.sourceLassoTargetWindowId)) {
    return false;
  }
  if (state.windowDragActive) {
    if (!findWindow(state, state.windowDragWindowId) ||
        !validDragMode(state.windowDragMode) ||
        !finiteRect(state.windowDragStartRect)) {
      return false;
    }
  } else if (state.windowDragWindowId != -1 ||
             state.windowDragMode != PlotWindowDragMode::None) {
    return false;
  }
  std::size_t totalLassoPoints = state.sourceLassoStrokes.empty() ? 0u : 0u;
  if (!validLassoStrokes(state.sourceLassoStrokes, &totalLassoPoints)) return false;
  std::size_t serializedLassoBytes = 0u;
  int maxId = 0;
  bool focusedSelected = false;
  for (std::size_t i = 0; i < state.windows.size(); ++i) {
    const auto& window = state.windows[i];
    if (window.windowId <= 0 || window.windowId > maxId) maxId = window.windowId;
    if (window.windowId <= 0 || !finiteRect(window.rect) ||
        !finiteCamera(window.camera) || window.viewState.plotModel < 0 ||
        window.viewState.plotModel >= kPlotModelCount ||
        window.viewState.stateRevision == 0u ||
        !validSerializedLasso(window.viewerLassoData) ||
        window.viewerLassoData.size() >
            kViewerWorkspaceMaxSerializedLassoBytes -
                std::min(serializedLassoBytes,
                         kViewerWorkspaceMaxSerializedLassoBytes) ||
        !validString(window.syncLabel) ||
        !validString(window.stableSyncLabel) ||
        !validLassoStrokes(window.viewerLassoStrokes, &totalLassoPoints) ||
        totalLassoPoints > kViewerWorkspaceMaxLassoPoints) {
      return false;
    }
    if (window.sourceSignalDocked) {
      std::size_t ownerIndex = 0u;
      const bool hasOwner = findWindow(
          state, window.sourceSignalDockOwnerWindowId, &ownerIndex);
      if (!state.sourceLassoSessionActive || !isSourceSignal(window) ||
          !hasOwner || isSourceSignal(state.windows[ownerIndex]) ||
          !finiteRect(window.sourceSignalRestoreRect)) {
        return false;
      }
    }
    serializedLassoBytes += window.viewerLassoData.size();
    if (window.windowId == state.focusedWindowId) {
      focusedSelected = window.selected;
    } else if (window.selected) {
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (state.windows[j].windowId == window.windowId) return false;
    }
  }
  if (!focusedSelected || state.nextWindowId <= maxId ||
      state.revision == 0u) {
    return false;
  }
  return true;
}

ViewerWorkspaceCameraUpdateResult updateViewerWorkspaceCamera(
    ViewerWorkspaceState* state,
    int windowId,
    const CameraState& camera) noexcept {
  ViewerWorkspaceCameraUpdateResult result{};
  if (!state || windowId <= 0) return result;
  if (!validateViewerWorkspaceState(*state)) {
    result.status = ViewerWorkspaceCameraUpdateStatus::InvalidState;
    return result;
  }
  std::size_t index = 0u;
  if (!findWindow(*state, windowId, &index)) {
    result.status = ViewerWorkspaceCameraUpdateStatus::MissingWindow;
    return result;
  }
  const double normSquared =
      static_cast<double>(camera.qx) * camera.qx +
      static_cast<double>(camera.qy) * camera.qy +
      static_cast<double>(camera.qz) * camera.qz +
      static_cast<double>(camera.qw) * camera.qw;
  if (!finiteCamera(camera) || !finite(normSquared) ||
      std::abs(normSquared - 1.0) > 2.0e-3) {
    result.status = ViewerWorkspaceCameraUpdateStatus::InvalidCamera;
    return result;
  }
  const CameraState& current = state->windows[index].camera;
  const bool changed = current.qx != camera.qx || current.qy != camera.qy ||
                       current.qz != camera.qz || current.qw != camera.qw ||
                       current.distance != camera.distance ||
                       current.panX != camera.panX ||
                       current.panY != camera.panY ||
                       current.orthographic != camera.orthographic ||
                       current.orthographicView != camera.orthographicView;
  if (!changed) {
    result.status = ViewerWorkspaceCameraUpdateStatus::Accepted;
    result.workspaceRevision = state->revision;
    result.windowRevision = state->windows[index].viewState.stateRevision;
    return result;
  }
  if (state->revision == std::numeric_limits<uint64_t>::max() ||
      state->windows[index].viewState.stateRevision ==
          std::numeric_limits<uint64_t>::max()) {
    result.status = ViewerWorkspaceCameraUpdateStatus::RevisionOverflow;
    return result;
  }
  state->windows[index].camera = camera;
  ++state->windows[index].viewState.stateRevision;
  ++state->revision;
  result.status = ViewerWorkspaceCameraUpdateStatus::Accepted;
  result.changed = true;
  result.workspaceRevision = state->revision;
  result.windowRevision = state->windows[index].viewState.stateRevision;
  return result;
}

namespace {

bool encodeCanonicalViewerLassoDataPrefix(
    uint64_t revision,
    const std::vector<LassoStroke>& strokes,
    std::size_t strokeCount,
    std::string* output) noexcept {
  if (!output || strokeCount > strokes.size()) return false;
  try {
    std::string encoded;
    if (strokeCount != 0u) {
      std::ostringstream stream;
      stream.imbue(std::locale::classic());
      stream << "v1|" << std::max<uint64_t>(1u, revision);
      stream << std::fixed << std::setprecision(6);
      for (std::size_t index = 0u; index < strokeCount; ++index) {
        const auto& stroke = strokes[index];
        if (stroke.points.size() < 3u || !validLassoStroke(stroke)) {
          return false;
        }
        stream << '|' << (stroke.subtract ? 's' : 'a') << ','
               << stroke.points.size();
        for (const auto& point : stroke.points) {
          stream << ',' << std::max(0.0f, std::min(1.0f, point.xNorm))
                 << ',' << std::max(0.0f, std::min(1.0f, point.yNorm));
        }
      }
      encoded = stream.str();
      if (encoded.size() > kViewerWorkspaceMaxSerializedLassoBytes) {
        return false;
      }
    }
    output->swap(encoded);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

bool encodeCanonicalViewerLassoData(
    uint64_t revision,
    const std::vector<LassoStroke>& strokes,
    std::string* output) noexcept {
  return encodeCanonicalViewerLassoDataPrefix(
      revision, strokes, strokes.size(), output);
}

ViewerWorkspaceLassoAppendResult appendViewerWorkspaceLassoStroke(
    ViewerWorkspaceState* state,
    const LassoStroke& stroke) noexcept {
  ViewerWorkspaceLassoAppendResult result{};
  if (!state) return result;
  if (!validateViewerWorkspaceState(*state)) {
    result.status = ViewerWorkspaceLassoAppendStatus::InvalidState;
    return result;
  }
  if (stroke.points.size() < 3u || !stroke.boundsValid ||
      !validLassoStroke(stroke)) {
    result.status = ViewerWorkspaceLassoAppendStatus::InvalidStroke;
    return result;
  }

  std::size_t totalPoints = 0u;
  if (!validLassoStrokes(state->sourceLassoStrokes, &totalPoints)) {
    result.status = ViewerWorkspaceLassoAppendStatus::InvalidState;
    return result;
  }
  for (const auto& window : state->windows) {
    if (!validLassoStrokes(window.viewerLassoStrokes, &totalPoints)) {
      result.status = ViewerWorkspaceLassoAppendStatus::InvalidState;
      return result;
    }
  }
  if (totalPoints > kViewerWorkspaceMaxLassoPoints ||
      stroke.points.size() > kViewerWorkspaceMaxLassoPoints - totalPoints) {
    result.status = ViewerWorkspaceLassoAppendStatus::CapacityExceeded;
    return result;
  }

  const bool global = state->sourceLassoSelectionsSynced ||
                      state->sourceLassoTargetWindowId <= 0;
  std::size_t targetIndex = 0u;
  if (!global &&
      !findWindow(*state, state->sourceLassoTargetWindowId, &targetIndex)) {
    result.status = ViewerWorkspaceLassoAppendStatus::MissingTarget;
    return result;
  }
  const auto& retained = global ? state->sourceLassoStrokes
                                : state->windows[targetIndex].viewerLassoStrokes;
  if (retained.size() >= kViewerWorkspaceMaxLassoStrokes ||
      state->revision == std::numeric_limits<uint64_t>::max()) {
    result.status = retained.size() >= kViewerWorkspaceMaxLassoStrokes
                        ? ViewerWorkspaceLassoAppendStatus::CapacityExceeded
                        : ViewerWorkspaceLassoAppendStatus::RevisionOverflow;
    return result;
  }
  const uint64_t currentLassoRevision =
      global ? state->sourceLassoRevision
             : state->windows[targetIndex].viewerLassoRevision;
  if (currentLassoRevision == std::numeric_limits<uint64_t>::max() ||
      (!global && state->windows[targetIndex].viewState.stateRevision ==
                      std::numeric_limits<uint64_t>::max())) {
    result.status = ViewerWorkspaceLassoAppendStatus::RevisionOverflow;
    return result;
  }

  try {
    std::vector<LassoStroke> next = retained;
    next.push_back(stroke);
    const uint64_t nextLassoRevision = currentLassoRevision + 1u;
    std::string nextData;
    if (!global && !encodeCanonicalViewerLassoData(
                       nextLassoRevision, next, &nextData)) {
      result.status = ViewerWorkspaceLassoAppendStatus::InvalidStroke;
      return result;
    }

    if (global) {
      state->sourceLassoStrokes.swap(next);
      state->sourceLassoRevision = nextLassoRevision;
      state->sourceLassoGlobalHasSelection = true;
      state->sourceLassoHasSelection = true;
    } else {
      auto& target = state->windows[targetIndex];
      target.viewerLassoStrokes.swap(next);
      target.viewerLassoData.swap(nextData);
      target.viewerLassoRevision = nextLassoRevision;
      ++target.viewState.stateRevision;
      state->sourceLassoHasSelection = true;
    }
    ++state->revision;
    result.status = ViewerWorkspaceLassoAppendStatus::Accepted;
    result.changed = true;
    result.globalSelection = global;
    result.targetWindowId = global ? -1 : state->sourceLassoTargetWindowId;
    result.lassoRevision = nextLassoRevision;
    result.workspaceRevision = state->revision;
    return result;
  } catch (...) {
    result.status = ViewerWorkspaceLassoAppendStatus::AllocationFailure;
    return result;
  }
}

ViewerWorkspaceLassoSessionResult updateViewerWorkspaceSourceLassoSession(
    ViewerWorkspaceState* state,
    int ownerWindowId,
    bool enabled) noexcept {
  ViewerWorkspaceLassoSessionResult result{};
  if (!state || ownerWindowId <= 0) return result;
  if (!validateViewerWorkspaceState(*state)) {
    result.status = ViewerWorkspaceLassoSessionStatus::InvalidState;
    return result;
  }
  std::size_t ownerIndex = 0u;
  if (!findWindow(*state, ownerWindowId, &ownerIndex)) {
    result.status = ViewerWorkspaceLassoSessionStatus::MissingOwner;
    return result;
  }
  const auto& owner = state->windows[ownerIndex];
  if (isSourceSignal(owner) ||
      owner.viewState.plotModel == kPlotModelGlossView) {
    result.status = ViewerWorkspaceLassoSessionStatus::UnsupportedOwner;
    return result;
  }

  bool nextActive = enabled;
  if (!enabled) {
    nextActive = std::any_of(
        state->windows.begin(), state->windows.end(),
        [](const PlotWindowDomainState& window) {
          return !isSourceSignal(window) &&
                 window.viewState.volumeSliceLassoRegion;
        });
  }

  std::size_t sourceIndex = 0u;
  bool hasSource = false;
  for (std::size_t i = 0u; i < state->windows.size(); ++i) {
    if (isSourceSignal(state->windows[i])) {
      sourceIndex = i;
      hasSource = true;
      break;
    }
  }
  const bool createSource = enabled && !hasSource;
  const bool changeSession = state->sourceLassoSessionActive != nextActive;
  const bool changeTarget = enabled &&
      state->sourceLassoTargetWindowId != ownerWindowId;
  const bool restoreDocked = !nextActive && std::any_of(
      state->windows.begin(), state->windows.end(),
      [](const PlotWindowDomainState& window) {
        return window.sourceSignalDocked;
      });
  if (!createSource && !changeSession && !changeTarget && !restoreDocked) {
    result.status = ViewerWorkspaceLassoSessionStatus::Accepted;
    result.sourceSurfaceWindowId =
        hasSource ? state->windows[sourceIndex].windowId : -1;
    result.workspaceRevision = state->revision;
    return result;
  }
  if (state->revision == std::numeric_limits<uint64_t>::max()) {
    result.status = ViewerWorkspaceLassoSessionStatus::RevisionOverflow;
    return result;
  }
  if (createSource &&
      state->windows.size() >= kViewerWorkspaceMaxWindows) {
    result.status = ViewerWorkspaceLassoSessionStatus::CapacityExceeded;
    return result;
  }
  if (createSource &&
      state->nextWindowId == std::numeric_limits<int>::max()) {
    result.status = ViewerWorkspaceLassoSessionStatus::RevisionOverflow;
    return result;
  }

  try {
    PlotWindowDomainState source{};
    if (createSource) {
      source.windowId = state->nextWindowId;
      source.rect = {
          owner.rect.x + owner.rect.w * 0.14f,
          owner.rect.y + owner.rect.h * 0.16f,
          owner.rect.w * 0.72f,
          owner.rect.h * 0.68f,
      };
      source.viewState = owner.viewState;
      source.viewState.plotModel = kPlotModelSourceSignal;
      source.viewState.volumeSliceLassoRegion = false;
      source.viewState.volumeSliceRed = false;
      source.viewState.volumeSliceYellow = false;
      source.viewState.volumeSliceGreen = false;
      source.viewState.volumeSliceCyan = false;
      source.viewState.volumeSliceBlue = false;
      source.viewState.volumeSliceMagenta = false;
      source.viewState.refreshPolicy = "reinterpret";
      source.viewState.requiresHostSamples = false;
      source.viewState.stateRevision = 1u;
      source.selected = false;
      source.sourceSignalTemporaryLassoSurface = true;
      source.sourceSignalDockOwnerWindowId = ownerWindowId;
      source.sourceSignalRestoreRect = source.rect;
      state->windows.reserve(state->windows.size() + 1u);
    }

    state->sourceLassoSessionActive = nextActive;
    if (enabled) state->sourceLassoTargetWindowId = ownerWindowId;
    if (restoreDocked) {
      for (auto& window : state->windows) {
        if (!window.sourceSignalDocked) continue;
        window.rect = window.sourceSignalRestoreRect;
        window.sourceSignalDocked = false;
        window.sourceSignalDockOwnerWindowId = -1;
        window.sourceSignalDockAnimStart = -10.0;
        window.sourceSignalDockAnimatingToDock = false;
      }
    }
    if (createSource) {
      result.sourceSurfaceWindowId = source.windowId;
      state->windows.push_back(std::move(source));
      ++state->nextWindowId;
      result.sourceSurfaceCreated = true;
    } else if (hasSource) {
      result.sourceSurfaceWindowId = state->windows[sourceIndex].windowId;
    }
    ++state->revision;
    result.status = ViewerWorkspaceLassoSessionStatus::Accepted;
    result.changed = true;
    result.workspaceRevision = state->revision;
    return result;
  } catch (...) {
    result.status = ViewerWorkspaceLassoSessionStatus::AllocationFailure;
    return result;
  }
}

ViewerWorkspaceReduceResult reduceViewerWorkspace(
    ViewerWorkspaceState* state,
    const ViewerControllerCommandBatch& batch) noexcept {
  ViewerWorkspaceReduceResult result{};
  if (!state || !validateViewerWorkspaceState(*state)) {
    result.status = ViewerWorkspaceReduceStatus::InvalidState;
    return result;
  }
  if (batch.count > batch.commands.size()) {
    result.status = ViewerWorkspaceReduceStatus::CapacityExceeded;
    return result;
  }
  try {
    // Preconstruct the only new bounded string before any state mutation.  A
    // retained lasso arena is never copied or traversed beyond validation.
    Shadow shadow{};
    if (!buildShadow(*state, &shadow)) return result;
    ViewerWorkspaceEffectBatch effects{};
    for (std::size_t i = 0; i < batch.count; ++i) {
      if (!preflightCommand(batch.commands[i], &shadow, &effects)) {
        if (effects.capacityExceeded) {
          result.status = ViewerWorkspaceReduceStatus::CapacityExceeded;
        } else {
          result.status = shadow.failureStatus;
        }
        return result;
      }
    }

    // Target-owned lasso payloads are serialized before the first semantic
    // mutation. Shrinking vectors is no-throw; swapping these prepared
    // strings makes undo/clear part of the same batch transaction.
    std::array<std::string, kViewerWorkspaceMaxWindows> canonicalLassoData{};
    std::array<bool, kViewerWorkspaceMaxWindows> replaceLassoData{};
    for (const auto& window : state->windows) {
      std::size_t shadowIndex = 0u;
      if (!findShadow(shadow.ids, shadow.count, window.windowId,
                      &shadowIndex)) {
        continue;
      }
      const std::size_t nextCount = shadow.lassoStrokeCounts[shadowIndex];
      if (nextCount == window.viewerLassoStrokes.size()) continue;
      if (nextCount > window.viewerLassoStrokes.size()) {
        result.status = ViewerWorkspaceReduceStatus::InvalidState;
        return result;
      }
      for (std::size_t strokeIndex = 0u; strokeIndex < nextCount;
           ++strokeIndex) {
        const auto& stroke = window.viewerLassoStrokes[strokeIndex];
        if (stroke.points.size() < 3u || !validLassoStroke(stroke)) {
          result.status = ViewerWorkspaceReduceStatus::InvalidState;
          return result;
        }
      }
      if (!encodeCanonicalViewerLassoDataPrefix(
              shadow.lassoRevisions[shadowIndex],
              window.viewerLassoStrokes, nextCount,
              &canonicalLassoData[shadowIndex])) {
        result.status = ViewerWorkspaceReduceStatus::AllocationFailure;
        return result;
      }
      replaceLassoData[shadowIndex] = true;
    }

    std::string customSelection = "Custom";
    std::string customNameInput;
    const bool captureCustomName = shadow.layoutCustom &&
        !isStandardPlotLayoutNameReserved(state->layoutPresetSelection);
    if (captureCustomName) customNameInput = state->layoutPresetSelection;
    for (std::size_t i = 0; i < state->windows.size(); ++i) {
      std::size_t shadowIndex = 0u;
      if (findShadow(shadow.ids, shadow.count, state->windows[i].windowId,
                     &shadowIndex) && shadow.refresh[shadowIndex]) {
        state->windows[i].viewState.refreshPolicy.reserve(11u);
      }
    }

    // Commit commands.  This phase only performs bounded, no-throw scalar
    // edits, vector erase/moves, and preflighted effect writes.
    for (std::size_t i = 0; i < batch.count; ++i) {
      const auto& command = batch.commands[i];
      std::size_t index = 0u;
      switch (command.kind) {
        case ViewerControllerCommandKind::None:
          break;
        case ViewerControllerCommandKind::FocusWindow:
          if (!findWindow(*state, command.windowId, &index)) break;
          state->focusedWindowId = command.windowId;
          for (auto& window : state->windows) window.selected = window.windowId == command.windowId;
          sortWindows(state);
          break;
        case ViewerControllerCommandKind::RequestCloseWindow:
          if (state->sourceLassoSessionActive) {
            auto* target = findWindow(*state, command.windowId, &index)
                               ? &state->windows[index]
                               : nullptr;
            if (target && isSourceSignal(*target)) break;
          }
          if (!findWindow(*state, command.windowId, &index) || state->windows.size() <= 1u) break;
          state->windows.erase(state->windows.begin() + static_cast<std::ptrdiff_t>(index));
          if (!findWindow(*state, state->focusedWindowId)) {
            state->focusedWindowId = state->windows.back().windowId;
          }
          for (auto& window : state->windows) window.selected = window.windowId == state->focusedWindowId;
          sortWindows(state);
          break;
        case ViewerControllerCommandKind::BeginWindowDrag:
          state->windowDragActive = true;
          state->windowDragWindowId = command.windowId;
          state->windowDragMode = command.dragMode;
          state->windowDragStartX = command.pointerX;
          state->windowDragStartY = command.pointerY;
          state->windowDragStartRect = command.rect;
          break;
        case ViewerControllerCommandKind::UpdateWindowDrag:
          if (!findWindow(*state, command.windowId, &index)) break;
          state->windows[index].rect = command.rect;
          break;
        case ViewerControllerCommandKind::EndWindowDrag:
          if (!findWindow(*state, command.windowId, &index)) break;
          state->windows[index].rect = command.rect;
          state->windowDragActive = false;
          state->windowDragWindowId = -1;
          state->windowDragMode = PlotWindowDragMode::None;
          state->windowDragStartX = 0.0f;
          state->windowDragStartY = 0.0f;
          state->windowDragStartRect = {};
          sortWindows(state);
          break;
        case ViewerControllerCommandKind::ToolbarMenu:
          state->activeToolbarPanel = state->activeToolbarPanel == ViewerWorkspaceToolbarPanel::MainMenu
                                          ? ViewerWorkspaceToolbarPanel::None
                                          : ViewerWorkspaceToolbarPanel::MainMenu;
          break;
        case ViewerControllerCommandKind::ToolbarAddPlot:
          state->activeToolbarPanel = state->activeToolbarPanel == ViewerWorkspaceToolbarPanel::AddPlot
                                          ? ViewerWorkspaceToolbarPanel::None
                                          : ViewerWorkspaceToolbarPanel::AddPlot;
          break;
        case ViewerControllerCommandKind::ToolbarLayoutPreset:
          state->activeToolbarPanel = state->activeToolbarPanel == ViewerWorkspaceToolbarPanel::LayoutPreset
                                          ? ViewerWorkspaceToolbarPanel::None
                                          : ViewerWorkspaceToolbarPanel::LayoutPreset;
          break;
        case ViewerControllerCommandKind::ToggleSlicingDrawer:
          if (!findWindow(*state, command.windowId, &index)) break;
          for (auto& window : state->windows) window.slicingDrawerOpen = false;
          state->windows[index].slicingDrawerOpen = !state->windows[index].slicingDrawerOpen;
          break;
        case ViewerControllerCommandKind::SetSlicingVector:
        case ViewerControllerCommandKind::SoloSlicingVector:
        case ViewerControllerCommandKind::ToggleAllSlicingVectors:
        case ViewerControllerCommandKind::ToggleSlicingLasso:
          if (!findWindow(*state, command.windowId, &index)) break;
          break;
        case ViewerControllerCommandKind::SourceLassoAdd:
          state->sourceLassoSubtractMode = false;
          break;
        case ViewerControllerCommandKind::SourceLassoSubtract:
          state->sourceLassoSubtractMode = true;
          break;
        case ViewerControllerCommandKind::SourceLassoClear:
        case ViewerControllerCommandKind::SourceLassoUndo:
        case ViewerControllerCommandKind::SourceSignalRestore:
          break;
      }
    }

    // Apply scalar shadow results and all per-window vector/revision values.
    applyShadow(state, shadow, &customSelection, &customNameInput,
                captureCustomName);
    if (state->sourceLassoStrokes.size() != shadow.sourceStrokeCount) {
      state->sourceLassoStrokes.resize(shadow.sourceStrokeCount);
    }
    applySlicingShadow(state, shadow, &canonicalLassoData,
                       replaceLassoData);
    // The close preflight shadow intentionally handles removals separately;
    // reconcile selected/focus invariants against the committed vector.
    for (auto& window : state->windows) window.selected = window.windowId == state->focusedWindowId;
    result.status = ViewerWorkspaceReduceStatus::Accepted;
    result.changed = shadow.changed;
    result.effects = effects;
    return result;
  } catch (...) {
    result.status = ViewerWorkspaceReduceStatus::AllocationFailure;
    return result;
  }
}

}  // namespace ChromaspaceViewer
