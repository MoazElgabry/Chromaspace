#include "ChromaspaceMetalFrameFailure.h"

namespace ChromaspaceMetalFrameFailure {

const char* label(Kind kind) noexcept {
  switch (kind) {
    case Kind::None: return "none";
    case Kind::DrawableUnavailable: return "drawable-unavailable";
    case Kind::BackpressureTimeout: return "backpressure-timeout";
    case Kind::PriorGpuSubmissionFailure: return "prior-gpu-submission-failure";
    case Kind::CompositorUnavailable: return "compositor-unavailable";
    case Kind::MetalContextUnavailable: return "metal-context-unavailable";
    case Kind::CommandBufferUnavailable: return "command-buffer-unavailable";
    case Kind::InvalidState: return "invalid-state";
    case Kind::InvariantViolation: return "invariant-violation";
    case Kind::EncodingFailure: return "encoding-failure";
    case Kind::Unknown: return "unknown";
    case Kind::Count: break;
  }
  return "unknown";
}

}  // namespace ChromaspaceMetalFrameFailure
