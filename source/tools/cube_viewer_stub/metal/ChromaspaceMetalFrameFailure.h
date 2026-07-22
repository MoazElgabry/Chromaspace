#pragma once

#include <cstdint>

namespace ChromaspaceMetalFrameFailure {

// This is the transport-independent failure vocabulary shared by the Metal
// seam, the portable executor, and the recovery policy.  It deliberately does
// not carry an NSError, a diagnostic string, or a platform object.  The
// backend owns translation from its native result to this closed set; callers
// must not infer recoverability by inspecting text diagnostics.
enum class Kind : uint8_t {
  None = 0,
  DrawableUnavailable,
  BackpressureTimeout,
  PriorGpuSubmissionFailure,
  CompositorUnavailable,
  MetalContextUnavailable,
  CommandBufferUnavailable,
  InvalidState,
  InvariantViolation,
  EncodingFailure,
  Unknown,
  Count,
};

// Returns a stable diagnostic label for telemetry only.  Recovery decisions
// must switch on Kind, never on this label.
const char* label(Kind kind) noexcept;

constexpr bool isSuccess(Kind kind) noexcept { return kind == Kind::None; }

}  // namespace ChromaspaceMetalFrameFailure
