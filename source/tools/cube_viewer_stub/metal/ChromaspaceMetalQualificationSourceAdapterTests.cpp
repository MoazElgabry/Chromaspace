#include "ChromaspaceMetalQualificationSourceAdapter.h"
#include "ChromaspaceMetalQualificationNativeSourceBackend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

static_assert(std::is_class<
                  ChromaspaceMetalQualification::NativeSourceFixtureBackend>::value,
              "the native fixture boundary must remain portable C++");
static_assert(!std::is_copy_constructible<
                  ChromaspaceMetalQualification::NativeSourceFixtureBackend>::value,
              "the native fixture must remain single-owner");
static_assert(!std::is_move_constructible<
                  ChromaspaceMetalQualification::NativeSourceFixtureBackend>::value,
              "the native fixture must not move its callback context");

namespace {

using ChromaspaceMetal::ImportedSourceTexture;
using ChromaspaceMetalQualification::NativeSourceBackend;
using ChromaspaceMetalQualification::QualificationSourceAdapter;
using ChromaspaceMetalQualification::SourceCompletionExpectation;
using ChromaspaceResidentSource::ClientAdapter;
using ChromaspaceResidentSource::ClientHealth;
using ChromaspaceResidentSource::ClientSnapshot;
using ChromaspaceResidentSource::DrainAdapter;
using ChromaspaceResidentSource::EventKind;
using ChromaspaceResidentSource::ResidentSourceSession;
using ChromaspaceResidentSource::SessionEvent;

struct NativeState final {
  uint64_t nextSourceId = 100u;
  bool failCreate = false;
  bool invalidDescriptor = false;
  bool senderMismatch = false;
  uint64_t forcedSourceId = 0u;
  std::vector<uint64_t> created;
  std::vector<uint64_t> retired;
  std::vector<std::string> operations;
};

struct DrainState final {
  std::vector<std::string> operations;
  bool fail = false;
};

void setError(std::string* error, const char* value) noexcept {
  if (!error) return;
  try {
    *error = value;
  } catch (...) {
  }
}

ImportedSourceTexture coherentSource(const std::string& sender,
                                     uint64_t device,
                                     uint64_t generation,
                                     uint64_t sourceId) {
  ImportedSourceTexture source{};
  source.sourceId = sourceId;
  source.senderId = sender;
  source.deviceRegistryId = device;
  source.senderGeneration = generation;
  source.sequence = generation;
  source.slotIndex = 0u;
  source.slotGeneration = generation;
  source.readyValue = generation + 100u;
  source.contentHash = generation + 200u;
  source.width = 4;
  source.height = 2;
  source.pixelFormat = 0;
  source.bytesPerRow = 32u;
  source.byteSize = 64u;
  source.semantics.sourceX = 0;
  source.semantics.sourceY = 0;
  source.semantics.sourceWidth = 4u;
  source.semantics.sourceHeight = 2u;
  source.semantics.sampledX = 0;
  source.semantics.sampledY = 0;
  source.semantics.sampledWidth = 4u;
  source.semantics.sampledHeight = 2u;
  source.semantics.coverage = ChromaspaceSourceExchange::SourceCoverage::FullSource;
  source.semantics.authoritative = true;
  source.semantics.colorPrimaries = "sRGB";
  source.semantics.transferFunction = "linear";
  return source;
}

bool nativeCreate(void* context,
                  const std::string& sender,
                  uint64_t device,
                  uint64_t generation,
                  ImportedSourceTexture* output,
                  std::string* error) noexcept {
  auto* state = static_cast<NativeState*>(context);
  assert(state != nullptr && output != nullptr);
  state->operations.push_back("create:" + std::to_string(generation));
  const uint64_t sourceId =
      state->forcedSourceId != 0u ? state->forcedSourceId
                                 : state->nextSourceId++;
  state->created.push_back(sourceId);
  *output = coherentSource(sender, device, generation, sourceId);
  if (state->senderMismatch) output->senderId = "foreign-sender";
  if (state->invalidDescriptor) output->bytesPerRow = 1u;
  if (state->failCreate) {
    setError(error, "native-create-failed");
    return false;
  }
  return true;
}

void nativeRetire(void* context, uint64_t sourceId) noexcept {
  auto* state = static_cast<NativeState*>(context);
  assert(state != nullptr && sourceId != 0u);
  state->operations.push_back("retire:" + std::to_string(sourceId));
  state->retired.push_back(sourceId);
}

bool drain(void* context,
           uint32_t timeoutMilliseconds,
           std::string* error) noexcept {
  auto* state = static_cast<DrainState*>(context);
  assert(state != nullptr && timeoutMilliseconds == 5000u);
  state->operations.push_back("drain");
  if (state->fail) {
    setError(error, "drain-failed");
    return false;
  }
  return true;
}

NativeSourceBackend nativeBackend(NativeState* state) {
  return {state, nativeCreate, nativeRetire};
}

DrainAdapter drainAdapter(DrainState* state) { return {state, drain}; }

void bindReady(ResidentSourceSession* session) {
  std::string error;
  assert(session->requestSender("sender-a", 7u, 0, &error));
  const auto tick = session->tick(0);
  assert(tick.snapshot.health == ChromaspaceResidentSource::SessionHealth::Ready);
  assert(tick.snapshot.viewerGeneration == 1u);
}

void invalidBackendAndClientLifecycle() {
  NativeSourceBackend invalid{};
  QualificationSourceAdapter unavailable(&invalid);
  assert(!unavailable.ready());
  assert(unavailable.failed());
  std::string error;
  assert(unavailable.clientAdapter()->create(
             unavailable.clientAdapter()->context, "sender-a", 7u, &error) ==
         nullptr);

  NativeState state{};
  const NativeSourceBackend backend = nativeBackend(&state);
  QualificationSourceAdapter adapter(backend);
  assert(adapter.ready());
  const ClientAdapter* client = adapter.clientAdapter();
  void* handle = client->create(client->context, "sender-a", 7u, &error);
  assert(handle != nullptr);
  assert(client->start(client->context, handle, &error));
  ClientSnapshot snapshot{};
  assert(client->snapshot(client->context, handle, &snapshot, &error));
  assert(snapshot.health == ClientHealth::Ready);
  assert(snapshot.viewerGeneration == 1u);
  assert(snapshot.lastObservedSequence == 0u);
  assert(!snapshot.hasActiveSource && snapshot.liveKeyCount == 0u);
}

void publishValidationAndRetirement() {
  NativeState state{};
  const NativeSourceBackend backend = nativeBackend(&state);
  QualificationSourceAdapter adapter(backend);
  const ClientAdapter* client = adapter.clientAdapter();
  std::string error;
  void* handle = client->create(client->context, "sender-a", 7u, &error);
  assert(handle != nullptr && client->start(client->context, handle, &error));

  assert(adapter.publish(1u, &error));
  const auto first = adapter.snapshot();
  assert(first.hasActiveSource && first.activeSource.sourceId == 100u);
  assert(first.lastAcceptedSourceGeneration == 1u);
  assert(!adapter.publish(1u, &error));
  assert(!adapter.publish(0u, &error));

  assert(adapter.publish(2u, &error));
  assert(state.retired.size() == 1u && state.retired[0] == 100u);
  const auto second = adapter.snapshot();
  assert(second.activeSource.sourceId == 101u);

  state.failCreate = true;
  assert(!adapter.publish(3u, &error));
  assert(state.retired.size() == 2u && state.retired.back() == 102u);
  assert(adapter.snapshot().activeSource.sourceId == 101u);
  state.failCreate = false;
  state.invalidDescriptor = true;
  assert(!adapter.publish(3u, &error));
  assert(state.retired.size() == 3u && state.retired.back() == 103u);
  assert(adapter.snapshot().activeSource.sourceId == 101u);
  state.invalidDescriptor = false;
  state.senderMismatch = true;
  assert(!adapter.publish(3u, &error));
  assert(state.retired.size() == 4u && state.retired.back() == 104u);
  assert(adapter.snapshot().activeSource.sourceId == 101u);
  state.senderMismatch = false;

  state.forcedSourceId = 101u;
  assert(!adapter.publish(3u, &error));
  assert(state.retired.size() == 4u);
  assert(adapter.snapshot().activeSource.sourceId == 101u);
  state.forcedSourceId = 0u;

  assert(client->clear(client->context, handle, &error));
  assert(!adapter.snapshot().hasActiveSource);
  assert(state.retired.size() == 5u && state.retired.back() == 101u);
  client->destroy(client->context, handle);
}

void staleHandlesAndBoundedExhaustion() {
  NativeState state{};
  QualificationSourceAdapter adapter(nativeBackend(&state));
  const ClientAdapter* client = adapter.clientAdapter();
  std::string error;
  void* stale = nullptr;
  for (int cycle = 0; cycle < 2; ++cycle) {
    void* handle = client->create(client->context, "sender-a", 7u, &error);
    assert(handle != nullptr && client->start(client->context, handle, &error));
    if (cycle == 0) stale = handle;
    client->destroy(client->context, handle);
  }
  void* current = client->create(client->context, "sender-a", 7u, &error);
  assert(current != nullptr && current != stale);
  assert(client->start(client->context, current, &error));
  ClientSnapshot output{};
  assert(!client->snapshot(client->context, stale, &output, &error));
  client->destroy(client->context, stale);
  assert(adapter.snapshot().clientLive);
  client->destroy(client->context, current);

  for (int cycle = 0; cycle < 5; ++cycle) {
    void* handle = client->create(client->context, "sender-a", 7u, &error);
    assert(handle != nullptr && client->start(client->context, handle, &error));
    client->destroy(client->context, handle);
  }
  assert(client->create(client->context, "sender-a", 7u, &error) == nullptr);
  assert(adapter.failed());
}

void residentSessionOrderingAndFinish() {
  NativeState native{};
  DrainState drainState{};
  QualificationSourceAdapter adapter(nativeBackend(&native));
  const ClientAdapter* client = adapter.clientAdapter();
  const DrainAdapter drainAdapterValue = drainAdapter(&drainState);
  ResidentSourceSession session(client, &drainAdapterValue);
  bindReady(&session);

  std::string error;
  assert(adapter.publish(1u, &error));
  auto tick = session.tick(1);
  assert(tick.snapshot.hasActiveSource && tick.snapshot.lastObservedSequence == 1u);
  assert(session.requestClear("sender-a", &error));
  tick = session.tick(2);
  assert(!tick.snapshot.hasActiveSource);
  assert(adapter.publish(2u, &error));
  tick = session.tick(3);
  assert(tick.snapshot.hasActiveSource && tick.snapshot.lastObservedSequence == 2u);

  const auto shutdown = session.shutdown();
  assert(shutdown.clientDestroyed && shutdown.drainAttempted &&
         shutdown.drainSucceeded);
  assert(!adapter.snapshot().clientLive && !adapter.snapshot().hasActiveSource);
  assert(native.retired.size() == 2u);
  assert(drainState.operations.size() == 1u);
  assert(adapter.finish(SourceCompletionExpectation{2u, 2u, true, true},
                        &error));
}

}  // namespace

int main() {
  invalidBackendAndClientLifecycle();
  publishValidationAndRetirement();
  staleHandlesAndBoundedExhaustion();
  residentSessionOrderingAndFinish();
  return 0;
}
