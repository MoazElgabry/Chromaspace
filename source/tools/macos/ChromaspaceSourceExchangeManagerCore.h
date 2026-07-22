#pragma once

#include <string>

namespace ChromaspaceSourceExchange {

enum class ManagedServiceStatus {
  NotRegistered,
  Enabled,
  RequiresApproval,
  NotFound,
  Unknown,
};

class ManagedServiceAdapter {
 public:
  virtual ~ManagedServiceAdapter() = default;
  virtual ManagedServiceStatus status(std::string* error) = 0;
  virtual bool registerService(std::string* error) = 0;
  virtual bool unregisterService(std::string* error) = 0;
};

class ManagedBundleValidator {
 public:
  virtual ~ManagedBundleValidator() = default;
  virtual bool validate(std::string* error) = 0;
};

struct SourceExchangeManagerResult {
  int exitCode = 1;
  std::string output;
};

// Pure command policy for the GUI-less service-manager app. The adapter owns
// the current-user LaunchAgent filesystem and launchctl calls; this layer is
// deterministic and testable without mutating a real user session.
SourceExchangeManagerResult runSourceExchangeManagerCommand(
    const std::string& command,
    ManagedServiceAdapter* service,
    ManagedBundleValidator* validator);

}  // namespace ChromaspaceSourceExchange
