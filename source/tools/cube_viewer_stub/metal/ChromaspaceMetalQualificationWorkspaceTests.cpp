#include "ChromaspaceMetalQualificationWorkspace.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

namespace {

using ChromaspaceMetalPlotCompiler::GlossPresentation;
using ChromaspaceMetalPlotRenderer::FrameRequest;
using ChromaspaceMetalPlotRenderer::PlotCommand;
using ChromaspaceMetalPlotRenderer::PlotKind;
using ChromaspaceMetalQualificationWorkspace::Tracker;

PlotKind expectedKind(int model, GlossPresentation presentation) {
  if (model <= ChromaspaceViewer::kPlotModelChromaticity) {
    return PlotKind::ResidentRaster;
  }
  if (model == ChromaspaceViewer::kPlotModelGlossView) {
    return presentation == GlossPresentation::Field2D
               ? PlotKind::GlossField2D
               : PlotKind::GlossProjection3D;
  }
  if (model == ChromaspaceViewer::kPlotModelWaveform) {
    return PlotKind::Waveform;
  }
  if (model == ChromaspaceViewer::kPlotModelHistogram) {
    return PlotKind::Histogram;
  }
  return PlotKind::SourceSignal;
}

std::unique_ptr<FrameRequest> frame(GlossPresentation presentation) {
  auto result = std::make_unique<FrameRequest>();
  for (int model = 0; model < ChromaspaceViewer::kPlotModelCount; ++model) {
    PlotCommand command{};
    command.windowId = model + 1;
    command.plotModel = model;
    command.kind = expectedKind(model, presentation);
    assert(result->append(command));
  }
  return result;
}

void profileIsDeterministicAndValid() {
  const auto result =
      ChromaspaceMetalQualificationWorkspace::buildProfile();
  assert(result.ready());
  assert(result.diagnostic.empty());
  assert(result.workspace.windows.size() ==
         ChromaspaceMetalQualificationWorkspace::kExpectedWindowCount);
  assert(result.workspace.focusedWindowId == 1);
  assert(result.workspace.nextWindowId == 13);
  assert(ChromaspaceViewer::validateViewerWorkspaceState(result.workspace));
  for (int model = 0; model < ChromaspaceViewer::kPlotModelCount; ++model) {
    const auto& window = result.workspace.windows[static_cast<std::size_t>(model)];
    assert(window.windowId == model + 1);
    assert(window.viewState.plotModel == model);
    assert(window.selected == (model == 0));
    assert(window.rect.x >= 0.0f && window.rect.y >= 0.0f &&
           window.rect.w > 0.0f && window.rect.h > 0.0f &&
           window.rect.x + window.rect.w <= 1.0f + 1.0e-6f &&
           window.rect.y + window.rect.h <= 1.0f + 1.0e-6f);
    for (int prior = 0; prior < model; ++prior) {
      const auto& other =
          result.workspace.windows[static_cast<std::size_t>(prior)];
      const bool disjoint =
          window.rect.x >= other.rect.x + other.rect.w ||
          other.rect.x >= window.rect.x + window.rect.w ||
          window.rect.y >= other.rect.y + other.rect.h ||
          other.rect.y >= window.rect.y + window.rect.h;
      assert(disjoint);
    }
  }
}

void bothGlossVariantsAreRequired() {
  Tracker tracker{};
  std::string diagnostic;
  const auto field = frame(GlossPresentation::Field2D);
  assert(tracker.observe(*field, GlossPresentation::Field2D, &diagnostic));
  auto snapshot = tracker.snapshot();
  assert(!snapshot.complete());
  assert(snapshot.acceptedObservationCount == 1u);
  assert(snapshot.coveredVariantCount == 12u);
  assert(snapshot.coveredMask == 4095u);

  const auto projection = frame(GlossPresentation::Projection3D);
  assert(tracker.observe(*projection, GlossPresentation::Projection3D,
                         &diagnostic));
  snapshot = tracker.snapshot();
  assert(snapshot.complete());
  assert(snapshot.coveredMask == 8191u);
  assert(snapshot.requiredMask == 8191u);
  assert(snapshot.coveredVariantCount == 13u);
  assert(snapshot.acceptedObservationCount == 2u);

  tracker.reset();
  snapshot = tracker.snapshot();
  assert(!snapshot.complete() && snapshot.coveredMask == 0u &&
         snapshot.acceptedObservationCount == 0u);
}

void rejectedFramesAreAtomic() {
  Tracker tracker{};
  std::string diagnostic;
  const auto accepted = frame(GlossPresentation::Field2D);
  assert(tracker.observe(*accepted, GlossPresentation::Field2D, &diagnostic));
  const auto before = tracker.snapshot();

  auto expectRejected = [&](std::unique_ptr<FrameRequest> candidate,
                            GlossPresentation presentation) {
    assert(candidate != nullptr);
    assert(!tracker.observe(*candidate, presentation, &diagnostic));
    assert(!diagnostic.empty());
    const auto after = tracker.snapshot();
    assert(after.coveredMask == before.coveredMask);
    assert(after.acceptedObservationCount == before.acceptedObservationCount);
  };

  auto missing = frame(GlossPresentation::Field2D);
  --missing->commandCount;
  expectRejected(std::move(missing), GlossPresentation::Field2D);

  auto duplicate = frame(GlossPresentation::Field2D);
  duplicate->commands[11].windowId = duplicate->commands[10].windowId;
  expectRejected(std::move(duplicate), GlossPresentation::Field2D);

  auto extra = frame(GlossPresentation::Field2D);
  extra->commands[extra->commandCount] = PlotCommand{};
  ++extra->commandCount;
  expectRejected(std::move(extra), GlossPresentation::Field2D);

  auto wrongKind = frame(GlossPresentation::Field2D);
  wrongKind->commands[0].kind = PlotKind::Histogram;
  expectRejected(std::move(wrongKind), GlossPresentation::Field2D);

  auto wrongModel = frame(GlossPresentation::Field2D);
  wrongModel->commands[1].plotModel = ChromaspaceViewer::kPlotModelCube;
  expectRejected(std::move(wrongModel), GlossPresentation::Field2D);

  auto scaffold = frame(GlossPresentation::Field2D);
  scaffold->commands[0].kind = PlotKind::Scaffold;
  expectRejected(std::move(scaffold), GlossPresentation::Field2D);

  auto wrongGloss = frame(GlossPresentation::Field2D);
  wrongGloss->commands[8].kind = PlotKind::GlossProjection3D;
  expectRejected(std::move(wrongGloss), GlossPresentation::Field2D);

  auto invalidPresentation = frame(GlossPresentation::Field2D);
  expectRejected(
      std::move(invalidPresentation),
      static_cast<GlossPresentation>(99));
}

}  // namespace

int main() {
  profileIsDeterministicAndValid();
  bothGlossVariantsAreRequired();
  rejectedFramesAreAtomic();
  std::cout << "Chromaspace Metal qualification workspace tests passed\n";
  return 0;
}
