#include "ChromaspaceMetalQualificationWorkspace.h"

#include <array>
#include <limits>
#include <new>
#include <utility>

namespace ChromaspaceMetalQualificationWorkspace {
namespace {

using ChromaspaceMetalPlotCompiler::GlossPresentation;
using ChromaspaceMetalPlotRenderer::PlotKind;

void setDiagnostic(std::string* diagnostic, const char* value) noexcept {
  if (diagnostic == nullptr) return;
  try {
    diagnostic->assign(value != nullptr ? value : "");
  } catch (...) {
    diagnostic->clear();
  }
}

uint32_t popcount(uint32_t value) noexcept {
  uint32_t count = 0u;
  while (value != 0u) {
    count += value & 1u;
    value >>= 1u;
  }
  return count;
}

bool validPresentation(GlossPresentation presentation) noexcept {
  return presentation == GlossPresentation::Field2D ||
         presentation == GlossPresentation::Projection3D;
}

PlotKind expectedKind(int plotModel,
                      GlossPresentation presentation) noexcept {
  if (plotModel >= ChromaspaceViewer::kPlotModelCube &&
      plotModel <= ChromaspaceViewer::kPlotModelChromaticity) {
    return PlotKind::ResidentRaster;
  }
  switch (plotModel) {
    case ChromaspaceViewer::kPlotModelGlossView:
      return presentation == GlossPresentation::Projection3D
                 ? PlotKind::GlossProjection3D
                 : PlotKind::GlossField2D;
    case ChromaspaceViewer::kPlotModelWaveform:
      return PlotKind::Waveform;
    case ChromaspaceViewer::kPlotModelHistogram:
      return PlotKind::Histogram;
    case ChromaspaceViewer::kPlotModelSourceSignal:
      return PlotKind::SourceSignal;
    default:
      return PlotKind::Scaffold;
  }
}

uint32_t coverageBit(int plotModel,
                     GlossPresentation presentation) noexcept {
  if (plotModel == ChromaspaceViewer::kPlotModelGlossView &&
      presentation == GlossPresentation::Projection3D) {
    return 1u << 12u;
  }
  return 1u << static_cast<uint32_t>(plotModel);
}

}  // namespace

BuildResult buildProfile() noexcept {
  BuildResult result{};
  try {
    ChromaspaceViewer::ViewerWorkspaceState workspace{};
    workspace.windows.reserve(kExpectedWindowCount);
    constexpr int kColumns = 4;
    constexpr int kRows = 3;
    constexpr float kMargin = 0.01f;
    constexpr float kGap = 0.01f;
    constexpr float kCellWidth =
        (1.0f - 2.0f * kMargin - 3.0f * kGap) /
        static_cast<float>(kColumns);
    constexpr float kCellHeight =
        (1.0f - 2.0f * kMargin - 2.0f * kGap) /
        static_cast<float>(kRows);

    for (int model = ChromaspaceViewer::kPlotModelCube;
         model < ChromaspaceViewer::kPlotModelCount; ++model) {
      const int column = model % kColumns;
      const int row = model / kColumns;
      ChromaspaceViewer::PlotWindowDomainState window{};
      window.windowId = model + 1;
      window.rect = {
          kMargin + static_cast<float>(column) * (kCellWidth + kGap),
          kMargin + static_cast<float>(row) * (kCellHeight + kGap),
          kCellWidth,
          kCellHeight};
      window.viewState.plotModel = model;
      window.viewState.stateRevision = static_cast<uint64_t>(model + 1);
      window.selected = model == ChromaspaceViewer::kPlotModelCube;
      workspace.windows.push_back(std::move(window));
    }
    workspace.focusedWindowId = 1;
    workspace.nextWindowId = ChromaspaceViewer::kPlotModelCount + 1;
    workspace.layoutPresetSelection = "Qualification All Renderers";
    workspace.revision = 1u;
    if (workspace.windows.size() != kExpectedWindowCount ||
        !ChromaspaceViewer::validateViewerWorkspaceState(workspace)) {
      result.status = BuildStatus::InvalidProfile;
      result.workspace = {};
      setDiagnostic(&result.diagnostic,
                    "qualification-renderer-workspace-invalid");
      return result;
    }
    result.workspace = std::move(workspace);
    result.status = BuildStatus::Ready;
    return result;
  } catch (const std::bad_alloc&) {
    result.status = BuildStatus::AllocationFailure;
    result.workspace = {};
    setDiagnostic(&result.diagnostic,
                  "qualification-renderer-workspace-allocation-failed");
    return result;
  } catch (...) {
    result.status = BuildStatus::InvalidProfile;
    result.workspace = {};
    setDiagnostic(&result.diagnostic,
                  "qualification-renderer-workspace-exception");
    return result;
  }
}

const char* buildStatusLabel(BuildStatus status) noexcept {
  switch (status) {
    case BuildStatus::Ready:
      return "ready";
    case BuildStatus::InvalidProfile:
      return "invalid-profile";
    case BuildStatus::AllocationFailure:
      return "allocation-failure";
  }
  return "unknown";
}

bool Tracker::observe(
    const ChromaspaceMetalPlotRenderer::FrameRequest& frame,
    GlossPresentation glossPresentation,
    std::string* diagnostic) noexcept {
  setDiagnostic(diagnostic, "");
  if (!validPresentation(glossPresentation)) {
    setDiagnostic(diagnostic,
                  "qualification-renderer-coverage-presentation-invalid");
    return false;
  }
  if (frame.commandCount != kExpectedWindowCount ||
      frame.commandCount > frame.commands.size()) {
    setDiagnostic(diagnostic,
                  "qualification-renderer-coverage-command-count-invalid");
    return false;
  }
  if (acceptedObservationCount_ == std::numeric_limits<uint32_t>::max()) {
    setDiagnostic(diagnostic,
                  "qualification-renderer-coverage-observation-overflow");
    return false;
  }

  std::array<bool, kExpectedWindowCount> seen{};
  uint32_t observedMask = 0u;
  for (std::size_t index = 0u; index < frame.commandCount; ++index) {
    const auto& command = frame.commands[index];
    if (command.windowId <= 0 ||
        command.windowId > ChromaspaceViewer::kPlotModelCount) {
      setDiagnostic(diagnostic,
                    "qualification-renderer-coverage-window-invalid");
      return false;
    }
    const std::size_t modelIndex =
        static_cast<std::size_t>(command.windowId - 1);
    if (seen[modelIndex]) {
      setDiagnostic(diagnostic,
                    "qualification-renderer-coverage-window-duplicate");
      return false;
    }
    const int plotModel = static_cast<int>(modelIndex);
    if (command.plotModel != plotModel) {
      setDiagnostic(diagnostic,
                    "qualification-renderer-coverage-model-mismatch");
      return false;
    }
    if (command.kind == PlotKind::Scaffold ||
        command.kind != expectedKind(plotModel, glossPresentation)) {
      setDiagnostic(diagnostic,
                    "qualification-renderer-coverage-kind-mismatch");
      return false;
    }
    seen[modelIndex] = true;
    observedMask |= coverageBit(plotModel, glossPresentation);
  }
  for (bool present : seen) {
    if (!present) {
      setDiagnostic(diagnostic,
                    "qualification-renderer-coverage-window-missing");
      return false;
    }
  }

  coveredMask_ |= observedMask;
  ++acceptedObservationCount_;
  return true;
}

void Tracker::reset() noexcept {
  coveredMask_ = 0u;
  acceptedObservationCount_ = 0u;
}

CoverageSnapshot Tracker::snapshot() const noexcept {
  CoverageSnapshot result{};
  result.coveredMask = coveredMask_;
  result.requiredMask = kRequiredCoverageMask;
  result.acceptedObservationCount = acceptedObservationCount_;
  result.coveredVariantCount = popcount(coveredMask_);
  return result;
}

}  // namespace ChromaspaceMetalQualificationWorkspace
