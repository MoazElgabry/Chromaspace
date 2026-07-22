#include "ChromaspaceViewerWorkspacePersistence.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace ChromaspaceViewer;

static uint64_t fnv1a(std::string_view bytes) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char c : bytes) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

static std::string checksum(uint64_t value) {
  const char* digits = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = digits[value & 0xfu];
    value >>= 4u;
  }
  return out;
}

static std::string withPayload(std::string bytes, std::string payload) {
  const std::size_t newline = bytes.find('\n');
  assert(newline != std::string::npos);
  std::string header = bytes.substr(0, newline);
  const std::string payloadNeedle = "\"payloadBytes\":";
  const std::size_t payloadPos = header.find(payloadNeedle);
  assert(payloadPos != std::string::npos);
  const std::size_t payloadStart = payloadPos + payloadNeedle.size();
  const std::size_t payloadEnd = header.find(',', payloadStart);
  assert(payloadEnd != std::string::npos);
  header.replace(payloadStart, payloadEnd - payloadStart,
                 std::to_string(payload.size()));
  const std::string checksumNeedle = "\"checksum\":\"";
  const std::size_t checksumPos = header.find(checksumNeedle);
  assert(checksumPos != std::string::npos);
  const std::size_t checksumStart = checksumPos + checksumNeedle.size();
  header.replace(checksumStart, 16u, checksum(fnv1a(payload)));
  return header + "\n" + payload;
}

static void assertRejected(std::string_view bytes,
                           ViewerWorkspacePersistenceStatus status) {
  const auto result = decodeViewerWorkspaceDocument(bytes);
  assert(!result.accepted());
  assert(result.status == status);
}

static ViewerWorkspaceDocument fixture() {
  ViewerWorkspaceDocument document = defaultViewerWorkspaceDocument();
  auto& first = document.workspace.windows.front();
  first.windowId = 4;
  first.rect = {0.05f, 0.1f, 0.4f, 0.7f};
  first.viewState.plotModel = kPlotModelWaveform;
  first.viewState.sampleSettingsKey = "durable-key";
  first.viewState.sourceSessionId = "must-not-persist";
  first.viewState.refreshPolicy = "resample";
  first.viewState.requiresHostSamples = true;
  first.viewState.hostRefreshRequestedRevision = 99u;
  first.viewState.waveformSampleColumns = 1024;
  first.camera.orthographic = true;
  first.camera.distance = 6.35f;
  first.camera.panX = 0.035f;
  first.slicingDrawerOpen = true;
  LassoStroke stroke{};
  stroke.subtract = true;
  stroke.points = {{0.1f, 0.2f}, {0.3f, 0.2f}, {0.3f, 0.4f}};
  first.viewerLassoStrokes.push_back(stroke);
  first.viewerLassoRevision = 7u;
  first.viewerLassoData = "v1|7|s,3,0.1,0.2,0.3,0.2,0.3,0.4";

  LassoStroke globalStroke{};
  globalStroke.points = {{0.6f, 0.2f}, {0.8f, 0.2f}, {0.7f, 0.4f}};
  document.workspace.sourceLassoSelectionsSynced = false;
  document.workspace.sourceLassoTargetWindowId = 4;
  document.workspace.sourceLassoRevision = 8u;
  document.workspace.sourceLassoStrokes.push_back(globalStroke);
  document.workspace.sourceLassoHasSelection = true;
  document.workspace.sourceLassoGlobalHasSelection = true;

  PlotWindowDomainState second{};
  second.windowId = 9;
  second.rect = {0.5f, 0.0f, 0.5f, 1.0f};
  second.viewState.plotModel = kPlotModelHistogram;
  second.viewState.stateRevision = 3u;
  second.camera.orthographic = true;
  second.camera.distance = 6.35f;
  second.camera.panX = -0.07f;
  second.slicingDrawerOpen = false;
  document.workspace.windows.push_back(second);
  document.workspace.focusedWindowId = 9;
  document.workspace.nextWindowId = 20;
  document.workspace.layoutPresetSelection = "Custom Layout";
  document.workspace.layoutPresetBeforeSolo = "Single";
  document.workspace.layoutPresetNameInput = "Custom Layout";
  document.workspace.revision = 17u;
  document.workspace.activeToolbarPanel = ViewerWorkspaceToolbarPanel::MainMenu;
  document.workspace.sourceLassoSessionActive = true;
  document.workspace.windowDragActive = true;
  document.presentation.showWorkspaceButtons = false;
  document.presentation.viewerFontSize = 2;
  document.presentation.windowWidth = 1280;
  document.presentation.windowHeight = 900;
  document.presentation.windowPositionValid = true;
  document.presentation.windowPosX = 12;
  document.presentation.windowPosY = 34;
  document.presentation.activeStandardLayoutIndex = 2;
  document.presentation.workspaceTopNorm = 0.07f;
  return document;
}

int main() {
  const ViewerWorkspaceDocument document = fixture();
  const auto encoded = encodeViewerWorkspaceV2(document);
  assert(encoded.accepted());
  const auto encodedAgain = encodeViewerWorkspaceV2(document);
  assert(encoded.bytes == encodedAgain.bytes);
  const auto decoded = decodeViewerWorkspaceDocument(encoded.bytes);
  assert(decoded.accepted());
  assert(validateViewerWorkspaceDocument(decoded.document));
  assert(decoded.document.workspace.windows.size() == 2u);
  assert(decoded.document.workspace.windows[0].windowId == 4);
  assert(decoded.document.workspace.windows[1].windowId == 9);
  assert(decoded.document.workspace.focusedWindowId == 9);
  assert(decoded.document.workspace.nextWindowId == 20);
  assert(decoded.document.workspace.windows[0].selected == false);
  assert(decoded.document.workspace.windows[1].selected == true);
  assert(decoded.document.workspace.windows[0].viewerLassoStrokes.size() == 1u);
  assert(decoded.document.workspace.windows[0].viewerLassoData.find(
             "0.100000,0.200000") != std::string::npos);
  assert(decoded.document.workspace.windows[0].slicingDrawerOpen);
  assert(decoded.document.workspace.windows[0].viewState.sourceSessionId.empty());
  assert(decoded.document.workspace.windows[0].viewState.refreshPolicy == "none");
  assert(decoded.document.workspace.windows[0].viewState.sampleSettingsKey ==
         sampleSettingsKey(decoded.document.workspace.windows[0].viewState, false));
  assert(!decoded.document.workspace.sourceLassoSessionActive);
  assert(!decoded.document.workspace.sourceLassoSelectionsSynced);
  assert(decoded.document.workspace.sourceLassoTargetWindowId == 4);
  assert(decoded.document.workspace.sourceLassoHasSelection);
  assert(decoded.document.workspace.sourceLassoGlobalHasSelection);
  assert(decoded.document.workspace.sourceLassoStrokes.size() == 1u);
  for (const auto& window : decoded.document.workspace.windows) {
    assert(!window.viewState.sourceSyncSelections);
  }
  assert(decoded.document.presentation.viewerFontSize == 2);

  ViewerWorkspaceDocument synced = fixture();
  synced.workspace.sourceLassoSelectionsSynced = true;
  synced.workspace.sourceLassoTargetWindowId = -1;
  for (auto& window : synced.workspace.windows) {
    window.viewState.sourceSyncSelections = true;
  }
  const auto syncedEncoded = encodeViewerWorkspaceV2(synced);
  assert(syncedEncoded.accepted());
  const auto syncedDecoded = decodeViewerWorkspaceDocument(syncedEncoded.bytes);
  assert(syncedDecoded.accepted());
  assert(syncedDecoded.document.workspace.sourceLassoSelectionsSynced);
  assert(syncedDecoded.document.workspace.sourceLassoTargetWindowId == -1);
  assert(syncedDecoded.document.workspace.sourceLassoHasSelection);
  assert(syncedDecoded.document.workspace.sourceLassoGlobalHasSelection);
  for (const auto& window : syncedDecoded.document.workspace.windows) {
    assert(window.viewState.sourceSyncSelections);
  }

  std::string checksumCorrupt = encoded.bytes;
  checksumCorrupt[checksumCorrupt.find("checksum") + 12u] = '0';
  assertRejected(checksumCorrupt, ViewerWorkspacePersistenceStatus::IntegrityMismatch);
  assertRejected(encoded.bytes.substr(0, encoded.bytes.size() - 1u),
                 ViewerWorkspacePersistenceStatus::IntegrityMismatch);
  std::string unsupported = encoded.bytes;
  const std::size_t version = unsupported.find("\"version\":2");
  assert(version != std::string::npos);
  unsupported.replace(version, 11u, "\"version\":3");
  assertRejected(unsupported, ViewerWorkspacePersistenceStatus::UnsupportedVersion);

  const std::size_t payloadStart = encoded.bytes.find('\n') + 1u;
  std::string unknownRecord = encoded.bytes.substr(payloadStart);
  const std::size_t presentation = unknownRecord.find("\"presentation\"");
  assert(presentation != std::string::npos);
  unknownRecord.replace(presentation, 14u, "\"unknown_type\"");
  assertRejected(withPayload(encoded.bytes, unknownRecord),
                 ViewerWorkspacePersistenceStatus::UnknownRecord);

  std::string trailing = encoded.bytes.substr(payloadStart) + "garbage\n";
  assertRejected(withPayload(encoded.bytes, trailing),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string unknownField = encoded.bytes.substr(payloadStart);
  const std::size_t documentEnd = unknownField.find('\n');
  assert(documentEnd != std::string::npos);
  unknownField.insert(documentEnd - 1u, ",\"typo\":1");
  assertRejected(withPayload(encoded.bytes, unknownField),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string lassoMismatch = encoded.bytes.substr(payloadStart);
  const std::size_t lassoPoint = lassoMismatch.find(
      "\"data\":\"v1|7|s,3,0.100000");
  assert(lassoPoint != std::string::npos);
  lassoMismatch.replace(
      lassoPoint + std::string("\"data\":\"v1|7|s,3,").size(), 8u,
      "0.200000");
  assertRejected(withPayload(encoded.bytes, lassoMismatch),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string incompleteWire = encoded.bytes.substr(payloadStart);
  const std::size_t pointsAt = incompleteWire.find("\"points\":[");
  assert(pointsAt != std::string::npos);
  const std::size_t pointsEnd = incompleteWire.find(']', pointsAt);
  assert(pointsEnd != std::string::npos);
  incompleteWire.replace(pointsAt, pointsEnd + 1u - pointsAt,
                         "\"points\":[{\"x\":0.1,\"y\":0.2},{\"x\":0.3,\"y\":0.2}]");
  assertRejected(withPayload(encoded.bytes, incompleteWire),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string runtimePolicyMismatch = encoded.bytes.substr(payloadStart);
  const std::size_t syncFlag = runtimePolicyMismatch.find(
      "\"sourceSyncSelections\":false");
  assert(syncFlag != std::string::npos);
  runtimePolicyMismatch.replace(
      syncFlag, std::string("\"sourceSyncSelections\":false").size(),
      "\"sourceSyncSelections\":true");
  assertRejected(withPayload(encoded.bytes, runtimePolicyMismatch),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string staleSampleKey = encoded.bytes.substr(payloadStart);
  const std::string sampleKeyNeedle = "\"sampleSettingsKey\":\"";
  const std::size_t sampleKeyAt = staleSampleKey.find(sampleKeyNeedle);
  assert(sampleKeyAt != std::string::npos);
  const std::size_t sampleKeyStart = sampleKeyAt + sampleKeyNeedle.size();
  const std::size_t sampleKeyEnd = staleSampleKey.find('"', sampleKeyStart);
  assert(sampleKeyEnd != std::string::npos);
  staleSampleKey.replace(sampleKeyStart, sampleKeyEnd - sampleKeyStart, "stale");
  assertRejected(withPayload(encoded.bytes, staleSampleKey),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string sourceFlagMismatch = encoded.bytes.substr(payloadStart);
  const std::size_t sourceFlagAt = sourceFlagMismatch.find(
      "\"sourceLassoHasSelection\":true");
  assert(sourceFlagAt != std::string::npos);
  sourceFlagMismatch.replace(sourceFlagAt,
                             std::string("\"sourceLassoHasSelection\":true").size(),
                             "\"sourceLassoHasSelection\":false");
  assertRejected(withPayload(encoded.bytes, sourceFlagMismatch),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string zeroLassoTarget = encoded.bytes.substr(payloadStart);
  const std::size_t zeroTargetAt = zeroLassoTarget.find(
      "\"sourceLassoTargetWindowId\":4");
  assert(zeroTargetAt != std::string::npos);
  zeroLassoTarget.replace(zeroTargetAt,
                          std::string("\"sourceLassoTargetWindowId\":4").size(),
                          "\"sourceLassoTargetWindowId\":0");
  assertRejected(withPayload(encoded.bytes, zeroLassoTarget),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string crlf = encoded.bytes.substr(payloadStart);
  for (std::size_t at = 0u; at < crlf.size(); ++at) {
    if (crlf[at] == '\n') {
      crlf.insert(at, 1u, '\r');
      ++at;
    }
  }
  assertRejected(withPayload(encoded.bytes, crlf),
                 ViewerWorkspacePersistenceStatus::Malformed);
  std::string headerCrlf = encoded.bytes;
  headerCrlf.insert(headerCrlf.find('\n'), 1u, '\r');
  assertRejected(headerCrlf, ViewerWorkspacePersistenceStatus::Malformed);

  std::string duplicateKey = encoded.bytes.substr(payloadStart);
  const std::size_t firstClose = duplicateKey.find('}');
  assert(firstClose != std::string::npos);
  duplicateKey.insert(firstClose, ",\"revision\":1");
  assertRejected(withPayload(encoded.bytes, duplicateKey),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string badRect = encoded.bytes.substr(payloadStart);
  const std::size_t rectX = badRect.find("\"rect\":{\"x\":");
  assert(rectX != std::string::npos);
  const std::size_t rectValue = rectX + std::string("\"rect\":{\"x\":").size();
  badRect.replace(rectValue, 1u, "2");
  assertRejected(withPayload(encoded.bytes, badRect),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string duplicateWindow = encoded.bytes.substr(payloadStart);
  const std::size_t secondWindow = duplicateWindow.rfind("\"windowId\":9");
  assert(secondWindow != std::string::npos);
  duplicateWindow.replace(secondWindow, 12u, "\"windowId\":4");
  assertRejected(withPayload(encoded.bytes, duplicateWindow),
                 ViewerWorkspacePersistenceStatus::Malformed);

  std::string badCamera = encoded.bytes.substr(payloadStart);
  const std::size_t cameraDistance = badCamera.find("\"distance\":6.3499999");
  assert(cameraDistance != std::string::npos);
  const std::size_t cameraValue = cameraDistance + std::string("\"distance\":").size();
  badCamera.replace(cameraValue, 9u, "0");
  assertRejected(withPayload(encoded.bytes, badCamera),
                 ViewerWorkspacePersistenceStatus::Malformed);

  ViewerWorkspaceDocument invalid = document;
  invalid.workspace.layoutPresetNameInput.assign(
      kViewerWorkspaceMaxStringBytes + 1u, 'x');
  const auto oversized = encodeViewerWorkspaceV2(invalid);
  assert(!oversized.accepted());
  assert(oversized.status == ViewerWorkspacePersistenceStatus::ValidationFailed);

  ViewerWorkspaceDocument incompleteStroke = document;
  incompleteStroke.workspace.windows.front().viewerLassoStrokes.front().points.resize(2u);
  const auto incompleteEncoded = encodeViewerWorkspaceV2(incompleteStroke);
  assert(!incompleteEncoded.accepted());
  assert(incompleteEncoded.status == ViewerWorkspacePersistenceStatus::ValidationFailed);

  ViewerWorkspaceDocument oversizedStroke = document;
  oversizedStroke.workspace.windows.front().viewerLassoStrokes.front().points.resize(
      kViewerWorkspaceMaxLassoPointsPerStroke + 1u);
  const auto oversizedStrokeEncoded = encodeViewerWorkspaceV2(oversizedStroke);
  assert(!oversizedStrokeEncoded.accepted());
  assert(oversizedStrokeEncoded.status == ViewerWorkspacePersistenceStatus::ValidationFailed);

  ViewerWorkspaceDocument tooManyStrokes = document;
  tooManyStrokes.workspace.windows.front().viewerLassoStrokes.clear();
  for (std::size_t i = 0u; i < kViewerWorkspaceMaxLassoStrokes + 1u; ++i) {
    LassoStroke many{};
    many.points = {{0.1f, 0.1f}, {0.2f, 0.1f}, {0.1f, 0.2f}};
    tooManyStrokes.workspace.windows.front().viewerLassoStrokes.push_back(std::move(many));
  }
  const auto tooManyStrokesEncoded = encodeViewerWorkspaceV2(tooManyStrokes);
  assert(!tooManyStrokesEncoded.accepted());
  assert(tooManyStrokesEncoded.status == ViewerWorkspacePersistenceStatus::ValidationFailed);

  ViewerWorkspaceDocument tooManyPoints = document;
  tooManyPoints.workspace.windows.front().viewerLassoStrokes.clear();
  for (std::size_t i = 0u; i < 5u; ++i) {
    LassoStroke many{};
    many.points.resize(i == 4u ? 3u : kViewerWorkspaceMaxLassoPointsPerStroke);
    for (auto& point : many.points) point = {0.123456f, 0.654321f};
    tooManyPoints.workspace.windows.front().viewerLassoStrokes.push_back(std::move(many));
  }
  const auto tooManyPointsEncoded = encodeViewerWorkspaceV2(tooManyPoints);
  assert(!tooManyPointsEncoded.accepted());
  assert(tooManyPointsEncoded.status == ViewerWorkspacePersistenceStatus::ValidationFailed);

  assertRejected(std::string(kViewerWorkspacePersistenceMaxDocumentBytes + 1u, 'x'),
                 ViewerWorkspacePersistenceStatus::CapacityExceeded);

  const std::string payloadForWindowCap = encoded.bytes.substr(payloadStart);
  const std::size_t firstWindowAt = payloadForWindowCap.find("{\"type\":\"window\"");
  assert(firstWindowAt != std::string::npos);
  const std::size_t firstWindowEnd = payloadForWindowCap.find('\n', firstWindowAt);
  assert(firstWindowEnd != std::string::npos);
  const std::string windowLine =
      payloadForWindowCap.substr(firstWindowAt, firstWindowEnd + 1u - firstWindowAt);
  std::string tooManyWindows = payloadForWindowCap.substr(0u, firstWindowAt);
  for (std::size_t i = 0u; i < kViewerWorkspaceMaxWindows + 1u; ++i) {
    std::string line = windowLine;
    const std::size_t windowIdAt = line.find("\"windowId\":4");
    assert(windowIdAt != std::string::npos);
    line.replace(windowIdAt, std::string("\"windowId\":4").size(),
                 "\"windowId\":" + std::to_string(i == 0u ? 4u : 1000u + i));
    tooManyWindows += line;
  }
  assertRejected(withPayload(encoded.bytes, tooManyWindows),
                 ViewerWorkspacePersistenceStatus::CapacityExceeded);

  const std::string legacy =
      "{\"type\":\"chromaspace_viewer_workspace_v1\",\"focusedWindowId\":42,"
      "\"nextWindowId\":1,\"windowWidth\":1000,\"windowHeight\":700,"
      "\"workspaceTopNorm\":0.1}\n"
      "{\"type\":\"plot_window\",\"windowId\":42,\"x\":-0.2,\"y\":0.2,"
      "\"w\":1.2,\"h\":0.8,\"camOrthographic\":0,\"camDistance\":6,"
      "\"camPanX\":0,\"camPanY\":0,\"camQx\":0,\"camQy\":0,"
      "\"camQz\":0,\"camQw\":1,\"camOrthographicView\":-1,\"plotModel\":9,"
      "\"viewerLassoData\":\"v1|4|a,3,0.100000,0.200000,0.300000,0.200000,0.300000,0.400000\"}\n";
  const auto legacyDecoded = decodeViewerWorkspaceDocument(legacy);
  assert(legacyDecoded.accepted());
  assert(legacyDecoded.document.workspace.focusedWindowId == 42);
  assert(legacyDecoded.document.workspace.nextWindowId == 43);
  assert(legacyDecoded.document.workspace.windows.front().rect.x == 0.0f);
  assert(legacyDecoded.document.workspace.windows.front().viewState.plotModel ==
         kPlotModelWaveform);
  assert(legacyDecoded.document.workspace.windows.front().camera.orthographic);
  assert(std::abs(legacyDecoded.document.workspace.windows.front().camera.distance - 6.35f) < 1e-5f);
  assert(legacyDecoded.document.workspace.windows.front().viewerLassoRevision == 4u);
  assert(legacyDecoded.document.workspace.windows.front().viewerLassoStrokes.size() == 1u);

  const std::string malformedLegacy =
      "{\"type\":\"chromaspace_viewer_workspace_v1\"}\n"
      "{\"type\":\"plot_window\",\"windowId\":1}\n";
  assertRejected(malformedLegacy, ViewerWorkspacePersistenceStatus::Malformed);
  return 0;
}
