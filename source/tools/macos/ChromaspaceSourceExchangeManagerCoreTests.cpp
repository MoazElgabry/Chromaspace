#include "ChromaspaceSourceExchangeManagerCore.h"

#include <cassert>
#include <iostream>

using namespace ChromaspaceSourceExchange;

namespace {

struct FakeService final : ManagedServiceAdapter {
  ManagedServiceStatus current = ManagedServiceStatus::NotRegistered;
  ManagedServiceStatus afterRegister = ManagedServiceStatus::Enabled;
  bool registerSucceeds = true;
  bool unregisterSucceeds = true;
  bool transitionOnRegisterFailure = false;
  int statusCalls = 0;
  int registerCalls = 0;
  int unregisterCalls = 0;
  std::string statusError;
  std::string registerError;
  std::string unregisterError;

  ManagedServiceStatus status(std::string* error) override {
    ++statusCalls;
    if (error) *error = statusError;
    return current;
  }

  bool registerService(std::string* error) override {
    ++registerCalls;
    if (error) *error = registerError;
    if (registerSucceeds || transitionOnRegisterFailure) {
      current = afterRegister;
    }
    return registerSucceeds;
  }

  bool unregisterService(std::string* error) override {
    ++unregisterCalls;
    if (error) *error = unregisterError;
    if (unregisterSucceeds) current = ManagedServiceStatus::NotRegistered;
    return unregisterSucceeds;
  }
};

struct FakeValidator final : ManagedBundleValidator {
  bool valid = true;
  int calls = 0;
  std::string validationError;

  bool validate(std::string* error) override {
    ++calls;
    if (error) *error = validationError;
    return valid;
  }
};

void registrationIsIdempotent() {
  FakeService service;
  FakeValidator validator;
  service.current = ManagedServiceStatus::Enabled;
  const auto value =
      runSourceExchangeManagerCommand("register", &service, &validator);
  assert(value.exitCode == 0);
  assert(value.output == "status=enabled");
  // The deployment adapter owns content-level idempotency so an enabled
  // service can still repair or update stale installed artifacts.
  assert(service.registerCalls == 1);
  assert(validator.calls == 1);
}

void registrationTransitionsOnce() {
  FakeService service;
  FakeValidator validator;
  const auto value =
      runSourceExchangeManagerCommand("register", &service, &validator);
  assert(value.exitCode == 0);
  assert(value.output == "status=enabled");
  assert(service.registerCalls == 1);
  assert(service.statusCalls == 2);
}

void approvalIsDistinct() {
  FakeService service;
  FakeValidator validator;
  service.current = ManagedServiceStatus::RequiresApproval;
  const auto value =
      runSourceExchangeManagerCommand("register", &service, &validator);
  assert(value.exitCode == 3);
  assert(value.output == "status=requires-approval");
  assert(service.registerCalls == 0);
}

void concurrentRegistrationRemainsIdempotent() {
  FakeService service;
  FakeValidator validator;
  service.registerSucceeds = false;
  service.transitionOnRegisterFailure = true;
  service.registerError = "already-registered";
  auto value =
      runSourceExchangeManagerCommand("register", &service, &validator);
  assert(value.exitCode == 0);
  assert(value.output == "status=enabled");
  assert(service.registerCalls == 1);
  assert(service.statusCalls == 2);

  service.current = ManagedServiceStatus::NotRegistered;
  service.afterRegister = ManagedServiceStatus::RequiresApproval;
  value = runSourceExchangeManagerCommand(
      "register", &service, &validator);
  assert(value.exitCode == 3);
  assert(value.output == "status=requires-approval");
}

void unregisterIsIdempotentAndSurvivesInvalidBundle() {
  FakeService service;
  FakeValidator validator;
  validator.valid = false;
  service.current = ManagedServiceStatus::NotRegistered;
  auto value =
      runSourceExchangeManagerCommand("unregister", &service, nullptr);
  assert(value.exitCode == 0);
  assert(service.unregisterCalls == 1);
  assert(validator.calls == 0);

  service.current = ManagedServiceStatus::Enabled;
  value = runSourceExchangeManagerCommand(
      "unregister", &service, nullptr);
  assert(value.exitCode == 0);
  assert(service.unregisterCalls == 2);
  assert(validator.calls == 0);
}

void validationNeverMutatesService() {
  FakeService service;
  FakeValidator validator;
  const auto value =
      runSourceExchangeManagerCommand("validate", nullptr, &validator);
  assert(value.exitCode == 0);
  assert(value.output == "status=valid");
  assert(service.statusCalls == 0);
  assert(service.registerCalls == 0);
  assert(service.unregisterCalls == 0);
}

void errorsAndUnknownCommandsAreObservable() {
  FakeService service;
  FakeValidator validator;
  validator.valid = false;
  validator.validationError = "foreign-bundle";
  auto value =
      runSourceExchangeManagerCommand("register", &service, &validator);
  assert(value.exitCode == 4);
  assert(value.output == "status=invalid reason=foreign-bundle");

  validator.valid = true;
  validator.calls = 0;
  value = runSourceExchangeManagerCommand("bogus", nullptr, nullptr);
  assert(value.exitCode == 2);
  assert(value.output == "status=invalid reason=unknown-command");
  assert(validator.calls == 0);
}

}  // namespace

int main() {
  registrationIsIdempotent();
  registrationTransitionsOnce();
  approvalIsDistinct();
  concurrentRegistrationRemainsIdempotent();
  unregisterIsIdempotentAndSurvivesInvalidBundle();
  validationNeverMutatesService();
  errorsAndUnknownCommandsAreObservable();
  std::cout << "Chromaspace SourceExchange manager core tests passed\n";
  return 0;
}
