#pragma once

#include "ChromaspaceMetalPlotCompiler.h"
#include "ChromaspaceViewerWorkspace.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ChromaspaceMetalQualificationWorkspace {

inline constexpr const char* kProfileName =
    "qualification-all-renderers-v2";
inline constexpr std::size_t kExpectedWindowCount =
    static_cast<std::size_t>(ChromaspaceViewer::kPlotModelCount);
inline constexpr uint32_t kRequiredCoverageMask = (1u << 13u) - 1u;
static_assert(ChromaspaceViewer::kPlotModelCount == 12,
              "qualification profile must be extended for new plot models");
static_assert(kExpectedWindowCount <=
                  ChromaspaceMetalPlotRenderer::kMaximumPlotWindows,
              "qualification profile exceeds renderer capacity");

enum class BuildStatus : uint8_t {
  Ready = 0,
  InvalidProfile,
  AllocationFailure,
};

struct BuildResult {
  BuildStatus status = BuildStatus::InvalidProfile;
  ChromaspaceViewer::ViewerWorkspaceState workspace{};
  std::string diagnostic;

  bool ready() const noexcept { return status == BuildStatus::Ready; }
};

BuildResult buildProfile() noexcept;
const char* buildStatusLabel(BuildStatus status) noexcept;

struct CoverageSnapshot {
  uint32_t coveredMask = 0u;
  uint32_t requiredMask = kRequiredCoverageMask;
  uint32_t acceptedObservationCount = 0u;
  uint32_t coveredVariantCount = 0u;

  bool complete() const noexcept {
    return coveredMask == requiredMask && coveredVariantCount == 13u;
  }
};

class Tracker final {
 public:
  bool observe(
      const ChromaspaceMetalPlotRenderer::FrameRequest& frame,
      ChromaspaceMetalPlotCompiler::GlossPresentation glossPresentation,
      std::string* diagnostic = nullptr) noexcept;

  void reset() noexcept;
  CoverageSnapshot snapshot() const noexcept;

 private:
  uint32_t coveredMask_ = 0u;
  uint32_t acceptedObservationCount_ = 0u;
};

}  // namespace ChromaspaceMetalQualificationWorkspace
