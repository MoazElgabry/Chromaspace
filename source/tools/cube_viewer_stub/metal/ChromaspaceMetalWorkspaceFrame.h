#pragma once

#include "ChromaspaceMetalPlotCompiler.h"
#include "ChromaspaceViewerFramePlan.h"
#include "ChromaspaceViewerLiveCommand.h"
#include "ChromaspaceViewerWorkspace.h"

#include <cstdint>
#include <string>

namespace ChromaspaceMetalWorkspaceFrame {

enum class CompileStatus : uint8_t {
  Ready = 0,
  InvalidRequest,
  InvalidWorkspace,
  InvalidFramePlan,
  MissingWorkspaceWindow,
  PlotCompileFailed,
  AllocationFailure,
};

struct CompileRequest {
  const ChromaspaceViewer::ViewerFramePlan* framePlan = nullptr;
  const ChromaspaceViewer::ViewerWorkspaceState* workspace = nullptr;
  const ChromaspaceMetal::ImportedSourceTexture* residentSource = nullptr;
  std::string sourceDiagnostic;
  uint64_t frameRevision = 1u;
  ChromaspaceMetalPlotCompiler::GlossPresentation glossPresentation =
      ChromaspaceMetalPlotCompiler::GlossPresentation::Field2D;
};

struct CompileResult {
  CompileStatus status = CompileStatus::InvalidRequest;
  ChromaspaceMetalPlotRenderer::FrameRequest frame{};
  int rejectedWindowId = -1;
  std::string diagnostic;

  bool ready() const noexcept { return status == CompileStatus::Ready; }
};

// Atomically applies the complete OFX runtime state to the focused plot
// window. Window geometry, camera, lasso and every other workspace-owned
// concern remain untouched. On failure the workspace is unchanged.
bool applyLiveParamsToFocusedWindow(
    const ChromaspaceViewer::ViewerLiveCommandParams& params,
    ChromaspaceViewer::ViewerWorkspaceState* workspace,
    int* updatedWindowId = nullptr,
    std::string* error = nullptr) noexcept;

// Joins the portable frame plan to the portable workspace and emits every
// plot command for one Metal frame. Exactly one resident source is attached
// to the resulting request; commands never carry source ownership.
CompileResult compileWorkspaceFrame(const CompileRequest& request) noexcept;

const char* compileStatusLabel(CompileStatus status) noexcept;

}  // namespace ChromaspaceMetalWorkspaceFrame
