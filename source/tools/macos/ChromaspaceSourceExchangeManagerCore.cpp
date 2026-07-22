#include "ChromaspaceSourceExchangeManagerCore.h"

#include <cctype>

namespace ChromaspaceSourceExchange {
namespace {

constexpr int kSuccess = 0;
constexpr int kServiceError = 1;
constexpr int kUsageError = 2;
constexpr int kRequiresApproval = 3;
constexpr int kInvalidBundle = 4;

SourceExchangeManagerResult result(int exitCode,
                                   const std::string& status,
                                   const std::string& detail = {}) {
  SourceExchangeManagerResult value{};
  value.exitCode = exitCode;
  value.output = "status=" + status;
  if (!detail.empty()) {
    std::string safeDetail = detail;
    for (char& c : safeDetail) {
      if (std::isspace(static_cast<unsigned char>(c))) c = '-';
    }
    value.output += " reason=" + safeDetail;
  }
  return value;
}

const char* statusLabel(ManagedServiceStatus status) {
  switch (status) {
    case ManagedServiceStatus::NotRegistered:
      return "not-registered";
    case ManagedServiceStatus::Enabled:
      return "enabled";
    case ManagedServiceStatus::RequiresApproval:
      return "requires-approval";
    case ManagedServiceStatus::NotFound:
      return "not-found";
    case ManagedServiceStatus::Unknown:
      return "unknown";
  }
  return "unknown";
}

SourceExchangeManagerResult serviceStatusResult(
    ManagedServiceAdapter* service) {
  std::string error;
  const ManagedServiceStatus status = service->status(&error);
  if (!error.empty()) return result(kServiceError, "error", error);
  if (status == ManagedServiceStatus::RequiresApproval) {
    return result(kRequiresApproval, statusLabel(status));
  }
  if (status == ManagedServiceStatus::Unknown) {
    return result(kServiceError, statusLabel(status));
  }
  return result(kSuccess, statusLabel(status));
}

}  // namespace

SourceExchangeManagerResult runSourceExchangeManagerCommand(
    const std::string& command,
    ManagedServiceAdapter* service,
    ManagedBundleValidator* validator) {
  if (command != "register" && command != "unregister" &&
      command != "status" && command != "validate") {
    return result(kUsageError, "invalid", "unknown-command");
  }

  if (command == "unregister") {
    if (!service) {
      return result(kUsageError, "invalid", "missing-service-adapter");
    }
    std::string error;
    const ManagedServiceStatus before = service->status(&error);
    if (!error.empty()) return result(kServiceError, "error", error);
    if (!service->unregisterService(&error)) {
      return result(kServiceError,
                    "error",
                    error.empty() ? "unregister-failed" : error);
    }
    const ManagedServiceStatus after = service->status(&error);
    if (!error.empty()) return result(kServiceError, "error", error);
    if (after != ManagedServiceStatus::NotRegistered &&
        after != ManagedServiceStatus::NotFound) {
      return result(kServiceError, "error", "unregister-not-observed");
    }
    return result(kSuccess, "not-registered");
  }

  if (!validator) {
    return result(kUsageError, "invalid", "missing-bundle-validator");
  }
  std::string validationError;
  if (!validator->validate(&validationError)) {
    return result(kInvalidBundle,
                  "invalid",
                  validationError.empty() ? "bundle-contract-failed"
                                          : validationError);
  }
  if (command == "validate") return result(kSuccess, "valid");
  if (!service) {
    return result(kUsageError, "invalid", "missing-service-adapter");
  }
  if (command == "status") return serviceStatusResult(service);

  std::string error;
  const ManagedServiceStatus before = service->status(&error);
  if (!error.empty()) return result(kServiceError, "error", error);
  if (before == ManagedServiceStatus::RequiresApproval) {
    return result(kRequiresApproval, "requires-approval");
  }
  if (before == ManagedServiceStatus::Unknown) {
    return result(kServiceError, "unknown");
  }
  if (!service->registerService(&error)) {
    const std::string registerError = error;
    std::string statusError;
    const ManagedServiceStatus afterFailure =
        service->status(&statusError);
    if (statusError.empty()) {
      if (afterFailure == ManagedServiceStatus::Enabled) {
        return result(kSuccess, "enabled");
      }
      if (afterFailure == ManagedServiceStatus::RequiresApproval) {
        return result(kRequiresApproval, "requires-approval");
      }
    }
    return result(
        kServiceError,
        "error",
        !registerError.empty()
            ? registerError
            : (!statusError.empty() ? statusError : "register-failed"));
  }
  const ManagedServiceStatus after = service->status(&error);
  if (!error.empty()) return result(kServiceError, "error", error);
  if (after == ManagedServiceStatus::Enabled) {
    return result(kSuccess, "enabled");
  }
  if (after == ManagedServiceStatus::RequiresApproval) {
    return result(kRequiresApproval, "requires-approval");
  }
  return result(kServiceError, "error", "registration-not-enabled");
}

}  // namespace ChromaspaceSourceExchange
