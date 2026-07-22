#include "ChromaspaceViewerLiveCommand.h"

#include <cassert>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace {

using namespace ChromaspaceViewer;

std::string params(uint64_t seq, const std::string& sender,
                   const std::string& extra = {}) {
  return std::string("{\"type\":\"params\",\"seq\":") +
         std::to_string(seq) + ",\"senderId\":\"" + sender +
         "\",\"stateRevision\":7,\"plotMode\":\"waveform\"," +
         "\"cloudSettingsKey\":\"quality=High\"," +
         "\"quality\":\"High\",\"pointSize\":2.5," +
         "\"waveformMode\":1,\"waveformChannelLuma\":true" +
         extra + "}";
}

void testCompleteParamsDecode() {
  const auto decoded = decodeViewerLiveCommand(
      params(11, "sender-\\u03b1",
             ",\"lassoData\":\"v1|a,3,0,0,1,0,0,1\","
             "\"sourceAspect\":1.5,\"generatedIdentityResolution\":33,"
             "\"unknownFutureField\":{\"nested\":[1,true,null]}"));
  assert(decoded.accepted());
  assert(decoded.command.kind == ViewerLiveCommandKind::Params);
  assert(decoded.command.params.seq == 11u);
  assert(decoded.command.params.senderId == "sender-\xce\xb1");
  assert(decoded.command.params.viewerState.plotModel == kPlotModelWaveform);
  assert(decoded.command.params.viewerState.quality == 2);
  assert(decoded.command.params.viewerState.waveformMode == 1);
  assert(decoded.command.params.viewerState.waveformChannelLuma);
  assert(decoded.command.params.viewerState.sourceSessionId == "sender-\xce\xb1");
  assert(decoded.command.params.generatedIdentityResolution == 33);
  assert(decoded.command.params.sourceAspect == 1.5f);
}

void testMalformedAndTypedRejections() {
  assert(decodeViewerLiveCommand("").command.status ==
         ViewerLiveCommandStatus::EmptyInput);
  assert(decodeViewerLiveCommand("{\"type\":\"params\"").command.status ==
         ViewerLiveCommandStatus::Malformed);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"type\":\"params\",\"seq\":1,"
             "\"senderId\":\"a\"}")
             .command.status == ViewerLiveCommandStatus::DuplicateField);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":0,\"senderId\":\"a\"}")
             .command.status == ViewerLiveCommandStatus::Invalid);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":1,\"senderId\":4}")
             .command.status == ViewerLiveCommandStatus::Invalid);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\","
             "\"pointSize\":\"large\"}")
             .command.status == ViewerLiveCommandStatus::Invalid);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\","
             "\"pointSize\":1e999}")
             .command.status == ViewerLiveCommandStatus::Invalid);
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\","
             "\"pointSize\":1e-999}")
             .command.status == ViewerLiveCommandStatus::Invalid);
  const auto smallFinite = decodeViewerLiveCommand(
      "{\"type\":\"params\",\"seq\":2,\"senderId\":\"a\","
      "\"pointDensity\":0.000000000000000000001}");
  assert(smallFinite.accepted());
  assert(decodeViewerLiveCommand(
             "{\"type\":\"params\",\"seq\":1,\"senderId\":\"\\ud800\"}")
             .command.status == ViewerLiveCommandStatus::Malformed);
  assert(decodeViewerLiveCommand("{\"type\":\"future\",\"seq\":1,"
                                 "\"senderId\":\"a\"}")
             .command.status == ViewerLiveCommandStatus::UnknownType);
}

void testBounds() {
  std::string oversized(kViewerLiveCommandMaxLineBytes + 1u, 'x');
  assert(decodeViewerLiveCommand(oversized).command.status ==
         ViewerLiveCommandStatus::Oversized);
  std::string sender(kViewerLiveCommandMaxStringBytes + 1u, 's');
  assert(decodeViewerLiveCommand(params(1, sender)).command.status ==
         ViewerLiveCommandStatus::Oversized);
  std::string lasso(kViewerLiveCommandMaxLassoBytes + 1u, 'p');
  assert(decodeViewerLiveCommand(
             params(1, "a", ",\"lassoData\":\"" + lasso + "\""))
             .command.status == ViewerLiveCommandStatus::Oversized);

  std::string tooMany = "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\"";
  for (std::size_t index = 0u;
       index < kViewerLiveCommandMaxJsonMembers; ++index) {
    tooMany += ",\"f" + std::to_string(index) + "\":0";
  }
  tooMany += "}";
  assert(decodeViewerLiveCommand(tooMany).command.status ==
         ViewerLiveCommandStatus::Oversized);

  std::string tooManyItems =
      "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\",\"future\":[";
  for (std::size_t index = 0u;
       index <= kViewerLiveCommandMaxJsonArrayItems; ++index) {
    if (index != 0u) tooManyItems += ',';
    tooManyItems += '0';
  }
  tooManyItems += "]}";
  assert(decodeViewerLiveCommand(tooManyItems).command.status ==
         ViewerLiveCommandStatus::Oversized);

  std::string tooDeep =
      "{\"type\":\"params\",\"seq\":1,\"senderId\":\"a\",\"future\":";
  for (std::size_t depth = 0u; depth <= kViewerLiveCommandMaxJsonDepth;
       ++depth) {
    tooDeep += '[';
  }
  tooDeep += '0';
  for (std::size_t depth = 0u; depth <= kViewerLiveCommandMaxJsonDepth;
       ++depth) {
    tooDeep += ']';
  }
  tooDeep += '}';
  assert(decodeViewerLiveCommand(tooDeep).command.status ==
         ViewerLiveCommandStatus::Oversized);
}

void testDropClassification() {
  const auto cloud = decodeViewerLiveCommand(
      "{\"type\":\"input_cloud\",\"seq\":4,\"senderId\":\"a\","
      "\"points\":\"1,2,3\"}");
  assert(cloud.accepted());
  assert(cloud.command.kind == ViewerLiveCommandKind::InputCloud);
  assert(cloud.command.status == ViewerLiveCommandStatus::Dropped);
  assert(cloud.command.params.senderId.empty());
  const auto source = decodeViewerLiveCommand(
      "{\"type\":\"source_signal\",\"seq\":5,"
      "\"senderId\":\"a\",\"surfaceId\":9}");
  assert(source.accepted());
  assert(source.command.kind == ViewerLiveCommandKind::SourceSignal);
  assert(source.command.status == ViewerLiveCommandStatus::Dropped);
}

void testReducerOrderingAndAtomicDrain() {
  ViewerLiveCommandReducer reducer;
  assert(reducer.submitLine(params(10, "a")).accepted());
  assert(reducer.submitLine(params(9, "a")).status ==
         ViewerLiveCommandStatus::Stale);
  assert(reducer.submitLine(
             "{\"type\":\"clear_viewer_output\",\"seq\":11,"
             "\"senderId\":\"b\"}")
             .status == ViewerLiveCommandStatus::InactiveSender);
  assert(reducer.submitLine(
             "{\"type\":\"clear_viewer_output\",\"seq\":11,"
             "\"senderId\":\"a\",\"reason\":\"host-stop\"}")
             .accepted());

  ViewerLiveCommandBatch first{};
  assert(reducer.drain(&first));
  assert(first.senderChanged);
  assert(first.previousSenderId.empty());
  assert(first.activeSenderId == "a");
  assert(first.hasParams && first.params.seq == 10u);
  assert(first.hasClear && first.clearSeq == 11u &&
         first.clearSenderId == "a");
  assert(first.clearReason == "host-stop");
  assert(first.rejectedCount == 2u);

  assert(reducer.submitLine(
             "{\"type\":\"clear_viewer_output\",\"seq\":100,"
             "\"senderId\":\"a\"}")
             .accepted());
  assert(reducer.submitLine(params(12, "a")).accepted());
  ViewerLiveCommandBatch independentWatermarks{};
  assert(reducer.drain(&independentWatermarks));
  assert(!independentWatermarks.senderChanged);
  assert(independentWatermarks.hasParams &&
         independentWatermarks.params.seq == 12u);
  assert(independentWatermarks.hasClear &&
         independentWatermarks.clearSeq == 100u);
  assert(reducer.submitLine(
             "{\"type\":\"clear_viewer_output\",\"seq\":100,"
             "\"senderId\":\"a\"}")
             .status == ViewerLiveCommandStatus::Stale);

  assert(reducer.submitLine(params(20, "b")).accepted());
  assert(reducer.submitLine(
             params(21, "b", ",\"viewerBackgroundColorR\":0.25"))
             .accepted());
  assert(reducer.submitLine(
             "{\"type\":\"heartbeat\",\"seq\":22,"
             "\"senderId\":\"b\"}")
             .accepted());
  assert(reducer.submitLine(
             "{\"type\":\"bring_to_front\",\"seq\":23,"
             "\"senderId\":\"b\"}")
             .accepted());
  ViewerLiveCommandBatch second{};
  assert(reducer.drain(&second));
  assert(second.senderChanged && second.previousSenderId == "a" &&
         second.activeSenderId == "b");
  assert(second.hasParams && second.params.seq == 21u);
  assert(second.params.viewerState.backgroundR == 0.25);
  assert(second.heartbeat && second.heartbeatSenderId == "b");
  assert(second.bringToFront);

  ViewerLiveCommandBatch empty{};
  assert(reducer.drain(&empty));
  assert(empty.empty());
  assert(empty.activeSenderId == "b");
  assert(!reducer.drain(nullptr));
}

void testSenderCapacity() {
  ViewerLiveCommandReducer reducer;
  for (std::size_t index = 0; index < kViewerLiveCommandMaxSenders; ++index) {
    assert(reducer.submitLine(
               params(static_cast<uint64_t>(index + 1u),
                      "sender-" + std::to_string(index)))
               .accepted());
  }
  assert(reducer.submitLine(params(1000u, "overflow-sender")).status ==
         ViewerLiveCommandStatus::SenderCapacityExceeded);
  reducer.reset();
  assert(reducer.submitLine(params(1u, "after-reset")).accepted());
}

void testControlEventsAndDropCounts() {
  ViewerLiveCommandReducer reducer;
  assert(reducer.submitLine(
             "{\"type\":\"input_cloud\",\"seq\":1,"
             "\"senderId\":\"a\",\"points\":\"x\"}")
             .accepted());
  assert(reducer.submitLine(
             "{\"type\":\"source_signal\",\"seq\":2,"
             "\"senderId\":\"a\",\"surfaceId\":2}")
             .accepted());
  assert(reducer.submitLine(
             "{\"type\":\"disconnect\",\"seq\":3,"
             "\"senderId\":\"a\"}")
             .accepted());
  assert(reducer.submitLine("{\"type\":\"shutdown\"}").accepted());
  ViewerLiveCommandBatch batch{};
  assert(reducer.drain(&batch));
  assert(batch.droppedInputCloudCount == 1u);
  assert(batch.droppedSourceSignalCount == 1u);
  assert(batch.disconnected);
  assert(batch.shutdown);
}

void testConcurrentSubmitAndDrain() {
  ViewerLiveCommandReducer reducer;
  std::atomic<bool> done{false};
  std::atomic<uint64_t> highestDrained{0u};
  std::thread producer([&] {
    for (uint64_t sequence = 1u; sequence <= 500u; ++sequence) {
      assert(reducer.submitLine(params(sequence, "threaded")).accepted());
    }
    done.store(true, std::memory_order_release);
  });
  while (!done.load(std::memory_order_acquire)) {
    ViewerLiveCommandBatch batch{};
    assert(reducer.drain(&batch));
    if (batch.hasParams) {
      const uint64_t previous = highestDrained.load(std::memory_order_relaxed);
      assert(batch.params.seq > previous);
      highestDrained.store(batch.params.seq, std::memory_order_relaxed);
    }
  }
  producer.join();
  ViewerLiveCommandBatch finalBatch{};
  assert(reducer.drain(&finalBatch));
  if (finalBatch.hasParams) {
    highestDrained.store(finalBatch.params.seq, std::memory_order_relaxed);
  }
  assert(highestDrained.load(std::memory_order_relaxed) == 500u);
}

}  // namespace

int main() {
  testCompleteParamsDecode();
  testMalformedAndTypedRejections();
  testBounds();
  testDropClassification();
  testReducerOrderingAndAtomicDrain();
  testSenderCapacity();
  testControlEventsAndDropCounts();
  testConcurrentSubmitAndDrain();
  return 0;
}
