#pragma once

#include "ChromaspaceViewerState.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>

namespace ChromaspaceViewer {

// The ingress contract is intentionally bounded before it reaches either the
// GLFW viewer or the Metal-native path.  Lasso data is the largest individual
// field currently emitted by the OFX side; all other strings use the smaller
// runtime bound below.
constexpr std::size_t kViewerLiveCommandMaxLineBytes = 1u * 1024u * 1024u;
constexpr std::size_t kViewerLiveCommandMaxStringBytes = 512u;
constexpr std::size_t kViewerLiveCommandMaxLassoBytes = 524288u;
constexpr std::size_t kViewerLiveCommandMaxJsonDepth = 20u;
constexpr std::size_t kViewerLiveCommandMaxJsonMembers = 256u;
constexpr std::size_t kViewerLiveCommandMaxJsonArrayItems = 256u;
constexpr std::size_t kViewerLiveCommandMaxSenders = 64u;

// This is the portable wire/runtime value consumed by the viewer.  It is the
// complete superset of the existing ResolvedPayload; keeping it here makes
// parsing and the Apple ingress seam share one lossless value rather than
// maintaining a second, subtly different params schema.
struct ViewerLiveCommandParams {
  uint64_t seq = 0;
  uint64_t stateRevision = 1;
  std::string senderId;
  std::string sourceMode = "input";
  bool drawOnImageMode = false;
  std::string plotMode = "rgb";
  std::string cloudSettingsKey;
  bool volumeSlicingEnabled = false;
  std::string volumeSlicingMode = "hue";
  bool lassoRegionEmpty = false;
  std::string lassoData;
  bool circularHsl = false;
  bool circularHsv = false;
  bool normConeNormalized = true;
  bool plotDisplayLinear = false;
  int plotDisplayLinearTransfer = 0;
  float sourceAspect = 16.0f / 9.0f;
  bool alwaysOnTop = true;
  bool resetViewOnPlotSwitch = true;
  std::string quality = "Low";
  std::string sampling = "Balanced";
  bool occupancyFill = true;
  std::string scale = "100%";
  int resolution = 25;
  float pointSize = 1.4f;
  float pointDensity = 1.0f;
  float colorSaturation = 2.0f;
  std::string plotStyle = "Plain Scope";
  std::string pointShape = "Circle";
  int glossNeighborhood = 1;
  float glossLiftScale = 1.0f;
  bool glossSpatialInset = false;
  float glossBodyOpacity = 0.10f;
  float glossHighlightOpacity = 0.42f;
  float glossPointCrispness = 0.72f;
  bool glossHideText = false;
  bool showOverflow = false;
  bool highlightOverflow = true;
  bool cubeSlicingEnabled = false;
  bool neutralRadiusEnabled = false;
  float neutralRadius = 1.0f;
  bool cubeSliceRed = true;
  bool cubeSliceGreen = false;
  bool cubeSliceBlue = false;
  bool cubeSliceCyan = false;
  bool cubeSliceYellow = false;
  bool cubeSliceMagenta = false;
  float overflowHighlightR = 1.0f;
  float overflowHighlightG = 0.0f;
  float overflowHighlightB = 0.0f;
  float backgroundColorR = 0.08f;
  float backgroundColorG = 0.08f;
  float backgroundColorB = 0.09f;
  bool identityOverlayEnabled = false;
  bool identityOverlayRamp = false;
  bool identityOverlayAuto = true;
  int identityOverlayRequestedSize = 25;
  int identityOverlaySize = 25;
  bool readGrayRamp = false;
  bool readIdentityPlot = false;
  bool isolateIdentityData = false;
  bool excludeIdentityData = false;
  bool hasExcludeIdentityData = false;
  int identityReadResolution = 29;
  int generatedIdentityResolution = 0;
  bool generatedIdentityDrawCube = false;
  bool generatedIdentityDrawRamp = false;
  int generatedIdentityStripBandCount = 0;
  uint64_t generatedIdentityStripRevision = 0;
  int chromaticityInputPrimaries = kDefaultRec709PrimariesChoice;
  int chromaticityInputTransfer = kDefaultGamma24TransferChoice;
  int chromaticityReferenceBasis = 0;
  // Overlay primaries use the enabled-choice index (Rec.709 is 9, enabled is
  // therefore 10 in the shared color-management table).
  int chromaticityOverlayPrimaries = kDefaultRec709PrimariesChoice + 1;
  bool chromaticityPlanckianLocus = true;
  bool chromaticitySpectralLocus3D = true;
  std::string version;
  ViewerRuntimeState viewerState{};
};

enum class ViewerLiveCommandKind : uint8_t {
  Params = 0,
  ClearViewerOutput,
  Heartbeat,
  BringToFront,
  Disconnect,
  Shutdown,
  InputCloud,
  SourceSignal,
  Unknown,
  Rejected,
};

enum class ViewerLiveCommandStatus : uint8_t {
  Accepted = 0,
  Dropped,
  EmptyInput,
  Oversized,
  Malformed,
  DuplicateField,
  UnknownType,
  Invalid,
  Stale,
  InactiveSender,
  SenderCapacityExceeded,
  AllocationFailure,
};

struct ViewerLiveCommand {
  ViewerLiveCommandKind kind = ViewerLiveCommandKind::Rejected;
  ViewerLiveCommandStatus status = ViewerLiveCommandStatus::Invalid;
  uint64_t seq = 0;
  std::string senderId;
  std::string reason;
  ViewerLiveCommandParams params{};

  bool accepted() const noexcept {
    return status == ViewerLiveCommandStatus::Accepted ||
           status == ViewerLiveCommandStatus::Dropped;
  }
};

struct ViewerLiveCommandDecodeResult {
  ViewerLiveCommand command{};

  bool accepted() const noexcept { return command.accepted(); }
};

struct ViewerLiveCommandSubmitResult {
  ViewerLiveCommandKind kind = ViewerLiveCommandKind::Rejected;
  ViewerLiveCommandStatus status = ViewerLiveCommandStatus::Invalid;
  uint64_t seq = 0;
  std::string senderId;

  bool accepted() const noexcept {
    return status == ViewerLiveCommandStatus::Accepted ||
           status == ViewerLiveCommandStatus::Dropped;
  }
};

struct ViewerLiveCommandBatch {
  bool senderChanged = false;
  std::string previousSenderId;
  std::string activeSenderId;

  bool hasParams = false;
  ViewerLiveCommandParams params{};
  bool hasClear = false;
  uint64_t clearSeq = 0;
  std::string clearSenderId;
  std::string clearReason;

  bool heartbeat = false;
  std::string heartbeatSenderId;
  bool bringToFront = false;
  bool disconnected = false;
  bool shutdown = false;

  // These are diagnostics only.  input_cloud/source_signal bodies are never
  // retained by the reducer and are counted at most once per drain.
  std::size_t droppedInputCloudCount = 0;
  std::size_t droppedSourceSignalCount = 0;
  std::size_t rejectedCount = 0;

  bool empty() const noexcept {
    return !senderChanged && !hasParams && !hasClear && !heartbeat &&
           !bringToFront && !disconnected && !shutdown &&
           droppedInputCloudCount == 0u && droppedSourceSignalCount == 0u &&
           rejectedCount == 0u;
  }
};

// Decodes one complete JSON line.  The result owns only bounded decoded
// fields.  Input-cloud and source-signal commands are classified as dropped;
// their payload bodies are not copied into a retained command.
ViewerLiveCommandDecodeResult decodeViewerLiveCommand(
    std::string_view line) noexcept;

class ViewerLiveCommandReducer {
 public:
  ViewerLiveCommandReducer() noexcept = default;

  ViewerLiveCommandSubmitResult submitLine(std::string_view line) noexcept;
  bool drain(ViewerLiveCommandBatch* output) noexcept;
  void reset() noexcept;

 private:
  struct SenderWatermark {
    bool used = false;
    std::string senderId;
    uint64_t lastParamsSequence = 0;
    uint64_t lastClearSequence = 0;
  };

  SenderWatermark* findSenderLocked(std::string_view senderId) noexcept;
  const SenderWatermark* findSenderLocked(std::string_view senderId) const noexcept;
  SenderWatermark* acquireSenderLocked(std::string_view senderId) noexcept;

  mutable std::mutex mutex_;
  std::array<SenderWatermark, kViewerLiveCommandMaxSenders> senders_{};
  std::string committedSenderId_;
  std::string pendingSenderId_;
  ViewerLiveCommandParams pendingParams_{};
  bool hasPendingParams_ = false;
  uint64_t pendingClearSeq_ = 0;
  std::string pendingClearSenderId_;
  std::string pendingClearReason_;
  bool hasPendingClear_ = false;
  bool pendingHeartbeat_ = false;
  std::string pendingHeartbeatSenderId_;
  bool pendingBringToFront_ = false;
  bool pendingDisconnect_ = false;
  bool pendingShutdown_ = false;
  std::size_t pendingDroppedInputCloudCount_ = 0;
  std::size_t pendingDroppedSourceSignalCount_ = 0;
  std::size_t pendingRejectedCount_ = 0;
};

}  // namespace ChromaspaceViewer
