#include "ChromaspaceViewerWorkspaceStore.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

using namespace ChromaspaceViewer;

namespace {

std::filesystem::path testDirectory() {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
                    ("chromaspace-workspace-store-tests-" +
                     std::to_string(static_cast<unsigned long long>(ticks)));
  std::filesystem::create_directories(root);
  return root;
}

struct ScopedTestDirectory {
  std::filesystem::path path;
  ~ScopedTestDirectory() {
    // This path is created by this test and never escapes the system temp
    // directory.  Cleanup is intentionally constrained to its exact root.
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

void writeFile(const std::filesystem::path& path, std::string_view bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  assert(output.is_open());
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output.good());
}

std::string readFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  assert(input.is_open());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

ViewerWorkspaceDocument fixture(int windowId, int plotModel) {
  const ViewerWorkspaceDocument raw = defaultViewerWorkspaceDocument();
  ViewerWorkspaceDocument document{};
  assert(sanitiseViewerWorkspaceDocument(raw, &document));
  assert(validateViewerWorkspaceDocument(document));
  auto& window = document.workspace.windows.front();
  window.windowId = windowId;
  window.viewState.plotModel = plotModel;
  window.viewState.stateRevision = 2u;
  window.rect = {0.1f, 0.1f, 0.8f, 0.8f};
  document.workspace.focusedWindowId = windowId;
  document.workspace.nextWindowId = windowId + 1;
  document.workspace.revision = 3u;
  return document;
}

std::filesystem::path backupPath(const std::filesystem::path& primary) {
  return std::filesystem::path(primary.string() + ".bak");
}

std::filesystem::path corruptPath(const std::filesystem::path& primary) {
  return std::filesystem::path(primary.string() + ".corrupt");
}

void assertWindow(const ViewerWorkspaceDocument& document, int windowId,
                  int plotModel) {
  assert(validateViewerWorkspaceDocument(document));
  assert(document.workspace.windows.size() == 1u);
  assert(document.workspace.windows.front().windowId == windowId);
  assert(document.workspace.windows.front().viewState.plotModel == plotModel);
}

const char kLegacyDocument[] =
    "{\"type\":\"chromaspace_viewer_workspace_v1\",\"focusedWindowId\":42,"
    "\"nextWindowId\":1,\"windowWidth\":1000,\"windowHeight\":700,"
    "\"workspaceTopNorm\":0.1}\n"
    "{\"type\":\"plot_window\",\"windowId\":42,\"x\":0.0,\"y\":0.0,"
    "\"w\":1.0,\"h\":1.0,\"camOrthographic\":0,\"camDistance\":6,"
    "\"camPanX\":0,\"camPanY\":0,\"camQx\":0,\"camQy\":0,"
    "\"camQz\":0,\"camQw\":1,\"camOrthographicView\":-1,\"plotModel\":9,"
    "\"viewerLassoData\":\"\"}\n";

}  // namespace

int main() {
  ScopedTestDirectory temporary{testDirectory()};
  const auto primary = temporary.path / "viewer_workspace_v2.jsonl";
  const auto backup = backupPath(primary);
  const ViewerWorkspaceStorePaths paths{primary, {}};

  // Missing files are a usable, validated defaults result.
  const auto missing = loadViewerWorkspaceStore(paths);
  assert(missing.accepted());
  assert(missing.status == ViewerWorkspaceStoreLoadStatus::DefaultsMissing);
  assert(!missing.evidencePresent);
  assertWindow(missing.document, 1, kPlotModelCube);

  const auto first = fixture(5, kPlotModelCube);
  const auto second = fixture(9, kPlotModelWaveform);
  const auto third = fixture(12, kPlotModelHistogram);

  // Save/reload and canonical primary precedence.
  const auto firstSave = saveViewerWorkspaceStore(paths, first);
  assert(firstSave.accepted());
  const auto firstLoad = loadViewerWorkspaceStore(paths);
  assert(firstLoad.status == ViewerWorkspaceStoreLoadStatus::AcceptedPrimary);
  assertWindow(firstLoad.document, 5, kPlotModelCube);

  writeFile(backup, encodeViewerWorkspaceV2(second).bytes);
  const auto precedence = loadViewerWorkspaceStore(paths);
  assert(precedence.status == ViewerWorkspaceStoreLoadStatus::AcceptedPrimary);
  assertWindow(precedence.document, 5, kPlotModelCube);

  // A valid old primary rotates to .bak, and backup recovery is explicit.
  const auto secondSave = saveViewerWorkspaceStore(paths, second);
  assert(secondSave.accepted());
  const auto rotatedBackup = loadViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{temporary.path / "missing-primary.jsonl", {}});
  (void)rotatedBackup;
  assert(std::filesystem::exists(backup));
  const auto backupDocument = decodeViewerWorkspaceDocument(readFile(backup));
  assert(backupDocument.accepted());
  assertWindow(backupDocument.document, 5, kPlotModelCube);
  std::filesystem::remove(primary);
  const auto recovered = loadViewerWorkspaceStore(paths);
  assert(recovered.status == ViewerWorkspaceStoreLoadStatus::AcceptedBackup);
  assert(recovered.repairSuggested);
  assertWindow(recovered.document, 5, kPlotModelCube);

  // Optional legacy v1 input is accepted only after canonical candidates.
  const auto legacy = temporary.path / "viewer_workspace_v1.jsonl";
  writeFile(legacy, kLegacyDocument);
  const auto legacyLoad = loadViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{temporary.path / "legacy-primary.jsonl", legacy});
  assert(legacyLoad.status == ViewerWorkspaceStoreLoadStatus::AcceptedLegacy);
  assert(legacyLoad.repairSuggested);
  assertWindow(legacyLoad.document, 42, kPlotModelWaveform);

  // Oversized and corrupt primary files fall back to a valid backup without
  // mutating the evidence.  With no valid fallback, defaults remain usable and
  // the result is degraded rather than silently clean.
  const auto fallbackPrimary = temporary.path / "fallback.jsonl";
  const auto fallbackBackup = backupPath(fallbackPrimary);
  writeFile(fallbackPrimary,
            std::string(kViewerWorkspacePersistenceMaxDocumentBytes + 1u, 'x'));
  writeFile(fallbackBackup, encodeViewerWorkspaceV2(third).bytes);
  const auto oversized = loadViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{fallbackPrimary, {}});
  assert(oversized.status == ViewerWorkspaceStoreLoadStatus::AcceptedBackup);
  assertWindow(oversized.document, 12, kPlotModelHistogram);
  assert(readFile(fallbackPrimary).size() ==
         kViewerWorkspacePersistenceMaxDocumentBytes + 1u);
  std::filesystem::remove(fallbackBackup);
  writeFile(fallbackPrimary, "not-json\n");
  const auto corruptOnly = loadViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{fallbackPrimary, {}});
  assert(corruptOnly.status == ViewerWorkspaceStoreLoadStatus::DefaultsRecovery);
  assert(corruptOnly.degraded());
  assert(corruptOnly.evidencePresent);
  assertWindow(corruptOnly.document, 1, kPlotModelCube);

  // Saving after a corrupt primary preserves it as the canonical .corrupt
  // evidence and leaves an existing recoverable backup untouched.
  writeFile(fallbackBackup, encodeViewerWorkspaceV2(first).bytes);
  const auto repairSave = saveViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{fallbackPrimary, {}}, third);
  assert(repairSave.accepted());
  const auto fallbackCorrupt = corruptPath(fallbackPrimary);
  assert(std::filesystem::exists(fallbackCorrupt));
  assert(readFile(fallbackPrimary) == encodeViewerWorkspaceV2(third).bytes);
  const auto preservedCorrupt = readFile(fallbackPrimary.string() + ".corrupt");
  assert(preservedCorrupt == "not-json\n");
  const auto preservedBackup = decodeViewerWorkspaceDocument(readFile(fallbackBackup));
  assert(preservedBackup.accepted());
  assertWindow(preservedBackup.document, 5, kPlotModelCube);

  // Invalid path failure is typed and leaves an existing primary recoverable.
  const std::string oversizedPath( kViewerWorkspaceStoreMaxPathBytes + 1u, 'p');
  const ViewerWorkspaceStorePaths invalidPaths{
      temporary.path / oversizedPath / "workspace.jsonl", {}};
  const auto invalidSave = saveViewerWorkspaceStore(invalidPaths, first);
  assert(!invalidSave.accepted());
  assert(invalidSave.status == ViewerWorkspaceStoreSaveStatus::InvalidArgument);
  const auto stillUsable = loadViewerWorkspaceStore(
      ViewerWorkspaceStorePaths{fallbackPrimary, {}});
  assert(stillUsable.status == ViewerWorkspaceStoreLoadStatus::AcceptedPrimary);
  assertWindow(stillUsable.document, 12, kPlotModelHistogram);
  return 0;
}
