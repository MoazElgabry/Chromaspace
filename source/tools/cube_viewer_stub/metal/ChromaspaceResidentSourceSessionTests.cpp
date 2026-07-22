#include "ChromaspaceResidentSourceSession.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace {

using namespace ChromaspaceResidentSource;

struct MockClient {
  std::string sender;
  ClientSnapshot snapshot{};
};

struct MockState {
  std::vector<std::string> operations;
  MockClient* lastClient = nullptr;
  bool failCreate = false;
  bool failStart = false;
  bool failClear = false;
  bool failDrain = false;
  int createCalls = 0;
  int startCalls = 0;
  int clearCalls = 0;
  int destroyCalls = 0;
  int drainCalls = 0;
};

void setError(std::string* error, const char* message) noexcept {
  if (!error) return;
  try {
    *error = message;
  } catch (...) {
  }
}

void* mockCreate(void* context,
                 const std::string& sender,
                 uint64_t,
                 std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  ++state->createCalls;
  state->operations.push_back("create:" + sender);
  if (state->failCreate) {
    setError(error, "mock-create-failed");
    return nullptr;
  }
  try {
    auto* client = new MockClient;
    client->sender = sender;
    client->snapshot.health = ClientHealth::Registering;
    state->lastClient = client;
    return client;
  } catch (...) {
    setError(error, "mock-create-allocation-failed");
    return nullptr;
  }
}

bool mockStart(void* context, void* rawClient, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  auto* client = static_cast<MockClient*>(rawClient);
  ++state->startCalls;
  state->operations.push_back("start:" +
                              (client == nullptr ? "<null>" : client->sender));
  if (client == nullptr || state->failStart) {
    setError(error, "mock-start-failed");
    if (client != nullptr) client->snapshot.health = ClientHealth::Failed;
    return false;
  }
  client->snapshot.health = ClientHealth::Ready;
  return true;
}

bool mockClear(void* context, void* rawClient, std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  auto* client = static_cast<MockClient*>(rawClient);
  ++state->clearCalls;
  state->operations.push_back("clear:" +
                              (client == nullptr ? "<null>" : client->sender));
  if (client == nullptr || state->failClear) {
    setError(error, "mock-clear-failed");
    return false;
  }
  client->snapshot.hasActiveSource = false;
  client->snapshot.activeSource = ChromaspaceMetal::ImportedSourceTexture{};
  return true;
}

bool mockSnapshot(void* context,
                  const void* rawClient,
                  ClientSnapshot* output,
                  std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  const auto* client = static_cast<const MockClient*>(rawClient);
  if (client == nullptr || output == nullptr) {
    setError(error, "mock-snapshot-failed");
    return false;
  }
  try {
    *output = client->snapshot;
    return true;
  } catch (...) {
    setError(error, "mock-snapshot-copy-failed");
    return false;
  }
}

void mockDestroy(void* context, void* rawClient) noexcept {
  auto* state = static_cast<MockState*>(context);
  auto* client = static_cast<MockClient*>(rawClient);
  ++state->destroyCalls;
  state->operations.push_back("destroy:" +
                              (client == nullptr ? "<null>" : client->sender));
  if (state->lastClient == client) state->lastClient = nullptr;
  delete client;
}

bool mockDrain(void* context,
               uint32_t timeoutMilliseconds,
               std::string* error) noexcept {
  auto* state = static_cast<MockState*>(context);
  ++state->drainCalls;
  state->operations.push_back("drain:" +
                              std::to_string(timeoutMilliseconds));
  if (state->failDrain) {
    setError(error, "mock-drain-failed");
    return false;
  }
  return true;
}

ClientAdapter clientAdapter(MockState* state) {
  return {state, mockCreate, mockStart, mockClear, mockSnapshot, mockDestroy};
}

DrainAdapter drainAdapter(MockState* state) { return {state, mockDrain}; }

bool hasEvent(const TickResult& result, EventKind kind) {
  for (std::size_t index = 0; index < result.eventCount; ++index) {
    if (result.events[index].kind == kind) return true;
  }
  return false;
}

const SessionEvent* findEvent(const TickResult& result, EventKind kind) {
  for (std::size_t index = 0; index < result.eventCount; ++index) {
    if (result.events[index].kind == kind) return &result.events[index];
  }
  return nullptr;
}

uint32_t retryDelay(const TickResult& result) {
  for (std::size_t index = 0; index < result.eventCount; ++index) {
    if (result.events[index].kind == EventKind::RetryScheduled) {
      return result.events[index].retryDelayMilliseconds;
    }
  }
  return 0u;
}

std::size_t operationIndex(const MockState& state, const std::string& value) {
  const auto it = std::find(state.operations.begin(), state.operations.end(),
                            value);
  assert(it != state.operations.end());
  return static_cast<std::size_t>(it - state.operations.begin());
}

void bindReady(ResidentSourceSession* session,
               MockState* state,
               const std::string& sender,
               int64_t now = 0) {
  std::string error;
  assert(session->requestSender(sender, 7u, now, &error));
  const TickResult result = session->tick(now);
  assert(result.snapshot.health == SessionHealth::Ready);
  assert(state->lastClient != nullptr);
}

void testInitialBindAndIdempotence() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  const int creates = state.createCalls;
  const int starts = state.startCalls;
  std::string error;
  assert(session.requestSender("sender-a", 7u, 1, &error));
  const TickResult result = session.tick(1);
  assert(state.createCalls == creates && state.startCalls == starts);
  assert(result.eventCount == 0u);
}

void testRapidRequestsCoalesce() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  std::string error;
  assert(session.requestSender("sender-a", 7u, 0, &error));
  assert(session.requestSender("sender-b", 7u, 0, &error));
  const TickResult result = session.tick(0);
  assert(result.snapshot.senderId == "sender-b");
  assert(state.createCalls == 1);
  assert(state.operations[0] == "create:sender-b");
}

void testSenderSwitchDrainOrdering() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.lastClient->snapshot.hasActiveSource = true;
  state.lastClient->snapshot.activeSource.sourceId = 10u;
  state.lastClient->snapshot.activeSource.senderId = "sender-a";
  state.lastClient->snapshot.activeSource.sequence = 3u;
  state.lastClient->snapshot.activeSource.slotGeneration = 4u;
  const TickResult active = session.tick(1);
  assert(active.snapshot.hasActiveSource);
  std::string error;
  assert(session.requestSender("sender-b", 7u, 2, &error));
  const TickResult switched = session.tick(2);
  assert(!switched.snapshot.hasActiveSource);
  assert(switched.snapshot.senderId == "sender-b");
  const SessionEvent* destroyedEvent =
      findEvent(switched, EventKind::ClientDestroyed);
  assert(destroyedEvent != nullptr);
  assert(destroyedEvent->senderId == "sender-a");
  const std::size_t clear = operationIndex(state, "clear:sender-a");
  const std::size_t drainIndex = operationIndex(state, "drain:5000");
  const std::size_t destroy = operationIndex(state, "destroy:sender-a");
  const std::size_t create = operationIndex(state, "create:sender-b");
  const std::size_t start = operationIndex(state, "start:sender-b");
  assert(clear < drainIndex && drainIndex < destroy && destroy < create &&
         create < start);
}

void testPendingClearTargetsNewSender() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  std::string error;
  assert(session.requestSender("sender-b", 7u, 1, &error));
  assert(session.requestClear("sender-b", &error));
  const TickResult result = session.tick(1);
  assert(result.snapshot.senderId == "sender-b");
  assert(state.operations.back() == "clear:sender-b");
  assert(operationIndex(state, "clear:sender-a") <
         operationIndex(state, "drain:5000"));
  assert(state.clearCalls == 2);
}

void testDrainFailureStillReplacesClient() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.failDrain = true;
  std::string error;
  assert(session.requestSender("sender-b", 7u, 1, &error));
  const TickResult result = session.tick(1);
  assert(hasEvent(result, EventKind::DrainFailed));
  assert(state.destroyCalls == 1);
  assert(state.lastClient != nullptr && state.lastClient->sender == "sender-b");
}

void testStartFailureUsesDrainAndRetries() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  state.failStart = true;
  std::string error;
  assert(session.requestSender("sender-a", 7u, 0, &error));
  const TickResult failed = session.tick(0);
  assert(failed.snapshot.health == SessionHealth::Failed);
  assert(hasEvent(failed, EventKind::RetryScheduled));
  assert(state.drainCalls == 1 && state.destroyCalls == 1);
  state.failStart = false;
  const TickResult recovered = session.tick(250);
  assert(recovered.snapshot.health == SessionHealth::Ready);
  assert(state.createCalls == 2 && state.startCalls == 2);
}

void testBackoffReachesEightSecondCapAndResets() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  state.failCreate = true;
  std::string error;
  assert(session.requestSender("sender-a", 7u, 0, &error));
  int64_t now = 0;
  const uint32_t expected[] = {250u, 500u, 1000u, 2000u,
                               4000u, 8000u, 8000u};
  for (uint32_t delay : expected) {
    const TickResult failed = session.tick(now);
    assert(failed.snapshot.health == SessionHealth::Failed);
    assert(retryDelay(failed) == delay);
    now += static_cast<int64_t>(delay);
  }
  state.failCreate = false;
  const TickResult recovered = session.tick(now);
  assert(recovered.snapshot.health == SessionHealth::Ready);
  state.failCreate = true;
  assert(session.requestSender("sender-b", 7u, now + 1, &error));
  const TickResult reset = session.tick(now + 1);
  assert(retryDelay(reset) == 250u);
}

void testRetryBackoffAndHealthEdges() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  state.failCreate = true;
  std::string error;
  assert(session.requestSender("sender-a", 7u, 0, &error));
  const TickResult first = session.tick(0);
  assert(first.snapshot.health == SessionHealth::Failed);
  assert(hasEvent(first, EventKind::RetryScheduled));
  const int failedCreates = state.createCalls;
  const TickResult quiet = session.tick(249);
  assert(state.createCalls == failedCreates);
  assert(!hasEvent(quiet, EventKind::RetryScheduled));
  state.failCreate = false;
  const TickResult recovered = session.tick(250);
  assert(state.createCalls == failedCreates + 1);
  assert(recovered.snapshot.health == SessionHealth::Ready);
  assert(session.tick(251).eventCount == 0u);
}

void testActiveIdentityAndClearRouting() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.lastClient->snapshot.hasActiveSource = true;
  state.lastClient->snapshot.activeSource.sourceId = 10u;
  state.lastClient->snapshot.activeSource.senderId = "sender-a";
  const TickResult first = session.tick(1);
  assert(hasEvent(first, EventKind::SourceActivated));
  assert(session.tick(2).eventCount == 0u);
  state.lastClient->snapshot.activeSource.sourceId = 11u;
  assert(hasEvent(session.tick(3), EventKind::SourceActivated));
  std::string error;
  assert(session.requestClear("other-sender", &error));
  assert(session.tick(4).snapshot.hasActiveSource);
  assert(session.requestClear("sender-a", &error));
  const TickResult cleared = session.tick(5);
  assert(!cleared.snapshot.hasActiveSource);
  assert(state.clearCalls == 1);
}

void testForeignActiveSourceIsNeverPublished() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.lastClient->snapshot.hasActiveSource = true;
  state.lastClient->snapshot.activeSource.sourceId = 22u;
  state.lastClient->snapshot.activeSource.senderId = "foreign-sender";
  const TickResult foreign = session.tick(1);
  assert(!foreign.snapshot.hasActiveSource);
  assert(!hasEvent(foreign, EventKind::SourceActivated));
  state.lastClient->snapshot.activeSource.senderId = "sender-a";
  assert(session.tick(2).snapshot.hasActiveSource);
}

void testHealthDiagnosticEdges() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.lastClient->snapshot.diagnostic = "steady";
  const TickResult first = session.tick(1);
  assert(hasEvent(first, EventKind::HealthChanged));
  assert(session.tick(2).eventCount == 0u);
  state.lastClient->snapshot.diagnostic = "changed";
  const TickResult changed = session.tick(3);
  assert(hasEvent(changed, EventKind::HealthChanged));
  assert(session.tick(4).eventCount == 0u);
}

void testDeviceUnavailableIsEdgeTriggered() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  std::string error;
  assert(session.requestSender("sender-a", 0u, 0, &error));
  const TickResult first = session.tick(0);
  assert(first.snapshot.health == SessionHealth::Unavailable);
  assert(hasEvent(first, EventKind::DeviceUnavailable));
  assert(session.tick(1).eventCount == 0u);
  assert(session.requestSender("sender-a", 7u, 2, &error));
  assert(session.tick(2).snapshot.health == SessionHealth::Ready);
}

void testShutdownResultAndIdempotence() {
  MockState state{};
  const ClientAdapter adapter = clientAdapter(&state);
  const DrainAdapter drain = drainAdapter(&state);
  ResidentSourceSession session(&adapter, &drain);
  bindReady(&session, &state, "sender-a");
  state.lastClient->snapshot.hasActiveSource = true;
  state.lastClient->snapshot.activeSource.sourceId = 10u;
  state.lastClient->snapshot.activeSource.senderId = "sender-a";
  (void)session.tick(1);
  state.failDrain = true;
  const ShutdownResult shutdown = session.shutdown();
  assert(shutdown.drainAttempted && !shutdown.drainSucceeded);
  assert(std::string(shutdown.drainDiagnostic.data()) == "mock-drain-failed");
  assert(shutdown.clientDestroyed);
  assert(!session.snapshot().hasActiveSource);
  assert(session.snapshot().health == SessionHealth::Stopped);
  const int destroys = state.destroyCalls;
  const ShutdownResult repeated = session.shutdown();
  assert(repeated.alreadyShutdown);
  assert(state.destroyCalls == destroys);
}

}  // namespace

int main() {
  testInitialBindAndIdempotence();
  testRapidRequestsCoalesce();
  testSenderSwitchDrainOrdering();
  testPendingClearTargetsNewSender();
  testDrainFailureStillReplacesClient();
  testStartFailureUsesDrainAndRetries();
  testBackoffReachesEightSecondCapAndResets();
  testRetryBackoffAndHealthEdges();
  testActiveIdentityAndClearRouting();
  testForeignActiveSourceIsNeverPublished();
  testHealthDiagnosticEdges();
  testDeviceUnavailableIsEdgeTriggered();
  testShutdownResultAndIdempotence();
  return 0;
}
