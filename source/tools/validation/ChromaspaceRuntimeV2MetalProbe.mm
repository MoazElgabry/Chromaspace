#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr NSUInteger kRasterPointCompactBlockWidth = 256u;
constexpr uint32_t kCompactionPointCount = 65537u;
constexpr size_t kMaximumHierarchyLevels = 8u;
constexpr size_t kCompactionPipelineCount = 5u;

constexpr const char* kCompactionPipelineNames[kCompactionPipelineCount] = {
    "rasterPointCompactLocalScanKernel",
    "rasterPointScanBlockSumsKernel",
    "rasterPointAddBlockOffsetsKernel",
    "rasterPointCompactScatterKernel",
    "rasterPointFinalizeIndirectArgsKernel",
};

enum ProbeExitCode : int {
  kUsageError = 2,
  kMetalDeviceUnavailable = 3,
  kMetallibLoadError = 4,
  kMissingFunction = 5,
  kPipelineError = 6,
  kAllocationError = 7,
  kCommandError = 8,
  kGpuError = 9,
  kMismatchError = 10,
};

// Metal's packed_float3 is a 12-byte value. Keep the CPU readback/reference
// type deliberately packed as three scalar floats so input/output strides are
// identical to the shader's device packed_float3 buffers.
struct PackedFloat3Value {
  float x;
  float y;
  float z;
};

struct ColorValue {
  float r;
  float g;
  float b;
  float a;
};

static_assert(sizeof(PackedFloat3Value) == 12u,
              "packed float3 validation stride must be 12 bytes");
static_assert(sizeof(ColorValue) == 16u,
              "float4 validation stride must be 16 bytes");

struct CpuCompactionReference {
  std::vector<PackedFloat3Value> inputPositions;
  std::vector<ColorValue> inputColors;
  std::vector<PackedFloat3Value> compactPositions;
  std::vector<ColorValue> compactColors;
  std::vector<uint32_t> visibleIndices;
  std::vector<uint32_t> firstBlockSums;
  std::vector<uint32_t> firstBlockOffsets;
  uint32_t expectedVisible = 0u;
};

int reportFailure(ProbeExitCode code,
                  const char* category,
                  const std::string& detail) {
  std::fprintf(stderr,
               "status=invalid error=%s detail=%s\n",
               category,
               detail.empty() ? "unknown" : detail.c_str());
  return static_cast<int>(code);
}

bool checkedMultiply(NSUInteger left, NSUInteger right, NSUInteger* result) {
  if (result == nullptr) return false;
  if (right != 0u && left > std::numeric_limits<NSUInteger>::max() / right) {
    return false;
  }
  *result = left * right;
  return true;
}

bool checkedAdd(NSUInteger left, NSUInteger right, NSUInteger* result) {
  if (result == nullptr ||
      left > std::numeric_limits<NSUInteger>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checkedBlockCount(NSUInteger pointCount, NSUInteger* blockCount) {
  if (blockCount == nullptr || pointCount == 0u) return false;
  NSUInteger withRemainder = 0u;
  if (!checkedAdd(pointCount, kRasterPointCompactBlockWidth - 1u,
                  &withRemainder)) {
    return false;
  }
  *blockCount = withRemainder / kRasterPointCompactBlockWidth;
  return *blockCount != 0u;
}

bool visibleForIndex(uint32_t index) {
  // Deliberately cover the first/last lanes around several 256-wide block
  // boundaries and the one-lane final partial block.
  if (index == 0u || index == 256u || index == 511u) return false;
  if (index == 255u || index == 512u || index == 65535u ||
      index == 65536u) {
    return true;
  }
  return (index % 7u) != 0u;
}

PackedFloat3Value encodedPosition(uint32_t index) {
  const float value = static_cast<float>(index);
  return PackedFloat3Value{value, value + 0.25f, value + 0.5f};
}

ColorValue encodedColor(uint32_t index, bool visible) {
  const float value = static_cast<float>(index);
  return ColorValue{value + 1.0f,
                    value + 2.0f,
                    value + 3.0f,
                    visible ? 1.0f : 0.0f};
}

std::vector<uint32_t> blockLocalExclusiveScan(
    const std::vector<uint32_t>& values) {
  std::vector<uint32_t> offsets(values.size(), 0u);
  for (size_t index = 0u; index < values.size(); ++index) {
    if (index % kRasterPointCompactBlockWidth == 0u) {
      continue;
    }
    offsets[index] = offsets[index - 1u] + values[index - 1u];
  }
  return offsets;
}

std::vector<uint32_t> reduceBlocks(const std::vector<uint32_t>& values) {
  const size_t blockCount =
      (values.size() + kRasterPointCompactBlockWidth - 1u) /
      kRasterPointCompactBlockWidth;
  std::vector<uint32_t> sums(blockCount, 0u);
  for (size_t index = 0u; index < values.size(); ++index) {
    sums[index / kRasterPointCompactBlockWidth] += values[index];
  }
  return sums;
}

CpuCompactionReference buildCpuReference(uint32_t pointCount) {
  CpuCompactionReference reference;
  reference.inputPositions.resize(pointCount);
  reference.inputColors.resize(pointCount);

  const size_t firstBlockCount =
      (static_cast<size_t>(pointCount) + kRasterPointCompactBlockWidth - 1u) /
      kRasterPointCompactBlockWidth;
  reference.firstBlockSums.assign(firstBlockCount, 0u);
  std::vector<uint32_t> localOffsets(pointCount, 0u);
  for (uint32_t index = 0u; index < pointCount; ++index) {
    const bool visible = visibleForIndex(index);
    reference.inputPositions[index] = encodedPosition(index);
    reference.inputColors[index] = encodedColor(index, visible);
    const size_t block = index / kRasterPointCompactBlockWidth;
    if (visible) {
      localOffsets[index] = reference.firstBlockSums[block]++;
      reference.visibleIndices.push_back(index);
    }
  }

  // Mirror the production fixed-256 recursive hierarchy on the CPU. Each
  // level scans the current block sums; the final top-level scan is propagated
  // down to the first-level offsets before stable destinations are assigned.
  std::vector<std::vector<uint32_t>> hierarchySums;
  std::vector<std::vector<uint32_t>> hierarchyOffsets;
  hierarchySums.push_back(reference.firstBlockSums);
  while (true) {
    const std::vector<uint32_t>& sums = hierarchySums.back();
    hierarchyOffsets.push_back(blockLocalExclusiveScan(sums));
    const std::vector<uint32_t> nextSums = reduceBlocks(sums);
    if (nextSums.size() == 1u) break;
    hierarchySums.push_back(nextSums);
    if (hierarchySums.size() >= kMaximumHierarchyLevels) break;
  }

  reference.firstBlockOffsets = hierarchyOffsets.front();
  for (size_t level = hierarchyOffsets.size(); level > 1u; --level) {
    const size_t childLevel = level - 2u;
    const std::vector<uint32_t>& parentOffsets = hierarchyOffsets[level - 1u];
    std::vector<uint32_t>& childOffsets = hierarchyOffsets[childLevel];
    for (size_t index = 0u; index < childOffsets.size(); ++index) {
      childOffsets[index] += parentOffsets[index / kRasterPointCompactBlockWidth];
    }
  }
  reference.firstBlockOffsets = hierarchyOffsets.front();

  reference.expectedVisible =
      static_cast<uint32_t>(reference.visibleIndices.size());
  reference.compactPositions.resize(reference.expectedVisible);
  reference.compactColors.resize(reference.expectedVisible);
  for (uint32_t index = 0u; index < pointCount; ++index) {
    if (!visibleForIndex(index)) continue;
    const size_t block = index / kRasterPointCompactBlockWidth;
    const uint32_t destination = reference.firstBlockOffsets[block] +
                                 localOffsets[index];
    reference.compactPositions[destination] = reference.inputPositions[index];
    reference.compactColors[destination] = reference.inputColors[index];
  }
  return reference;
}

id<MTLComputePipelineState> buildCompactionPipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    const char* functionName,
    std::string* error) {
  if (device == nil || library == nil || functionName == nullptr ||
      error == nullptr) {
    if (error) *error = "invalid-pipeline-request";
    return nil;
  }
  NSString* name = [NSString stringWithUTF8String:functionName];
  if (name == nil) {
    *error = std::string("invalid-function-name:") + functionName;
    return nil;
  }
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (function == nil) {
    *error = std::string("missing-function:") + functionName;
    return nil;
  }
  NSError* pipelineError = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&pipelineError];
  if (pipeline == nil) {
    const char* detail = pipelineError.localizedDescription.UTF8String;
    *error = std::string("pipeline-create-failed:") + functionName + ":" +
             (detail ? detail : "unknown");
    return nil;
  }
  if (pipeline.maxTotalThreadsPerThreadgroup <
      kRasterPointCompactBlockWidth) {
    *error = std::string("max-threads-below-256:") + functionName;
    return nil;
  }
  return pipeline;
}

int runCompactionQualification(
    id<MTLDevice> device,
    const std::array<id<MTLComputePipelineState>, kCompactionPipelineCount>&
        pipelines) {
  if (device == nil) {
    return reportFailure(kGpuError, "gpu", "metal-device-unavailable");
  }

  NSUInteger pointCount = static_cast<NSUInteger>(kCompactionPointCount);
  NSUInteger firstBlockCount = 0u;
  if (!checkedBlockCount(pointCount, &firstBlockCount) ||
      pointCount > std::numeric_limits<uint32_t>::max() ||
      firstBlockCount > std::numeric_limits<uint32_t>::max() ||
      firstBlockCount != 257u) {
    return reportFailure(kAllocationError,
                         "allocation",
                         "point-count-or-block-count-overflow");
  }

  NSUInteger positionBytes = 0u;
  NSUInteger colorBytes = 0u;
  NSUInteger pointOffsetBytes = 0u;
  NSUInteger blockSumBytes = 0u;
  if (!checkedMultiply(pointCount, sizeof(PackedFloat3Value),
                       &positionBytes) ||
      !checkedMultiply(pointCount, sizeof(ColorValue), &colorBytes) ||
      !checkedMultiply(pointCount, sizeof(uint32_t), &pointOffsetBytes) ||
      !checkedMultiply(firstBlockCount, sizeof(uint32_t), &blockSumBytes)) {
    return reportFailure(kAllocationError, "allocation", "buffer-size-overflow");
  }

  auto sharedBuffer = [&](NSUInteger length, const char* label)
      -> id<MTLBuffer> {
    if (length == 0u) return nil;
    id<MTLBuffer> buffer =
        [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (buffer == nil) {
      std::fprintf(stderr, "allocation-error:%s\n", label);
    }
    return buffer;
  };

  id<MTLBuffer> inputPositions = sharedBuffer(positionBytes, "input-positions");
  id<MTLBuffer> inputColors = sharedBuffer(colorBytes, "input-colors");
  id<MTLBuffer> localOffsets = sharedBuffer(pointOffsetBytes, "local-offsets");
  id<MTLBuffer> firstBlockSums = sharedBuffer(blockSumBytes, "first-block-sums");
  id<MTLBuffer> compactPositions =
      sharedBuffer(positionBytes, "compact-positions");
  id<MTLBuffer> compactColors = sharedBuffer(colorBytes, "compact-colors");
  id<MTLBuffer> indirectArguments =
      sharedBuffer(sizeof(uint32_t) * 4u, "indirect-arguments");
  if (inputPositions == nil || inputColors == nil || localOffsets == nil ||
      firstBlockSums == nil || compactPositions == nil || compactColors == nil ||
      indirectArguments == nil) {
    return reportFailure(kAllocationError,
                         "allocation",
                         "shared-test-buffer-allocation-failed");
  }

  const CpuCompactionReference reference =
      buildCpuReference(kCompactionPointCount);
  std::memcpy(inputPositions.contents,
              reference.inputPositions.data(),
              positionBytes);
  std::memcpy(inputColors.contents, reference.inputColors.data(), colorBytes);
  std::memset(localOffsets.contents, 0, pointOffsetBytes);
  std::memset(firstBlockSums.contents, 0, blockSumBytes);
  std::memset(compactPositions.contents, 0xA5, positionBytes);
  std::memset(compactColors.contents, 0xA5, colorBytes);
  std::memset(indirectArguments.contents, 0xA5, sizeof(uint32_t) * 4u);

  std::array<id<MTLBuffer>, kMaximumHierarchyLevels> blockSums{};
  std::array<id<MTLBuffer>, kMaximumHierarchyLevels> blockOffsets{};
  std::array<NSUInteger, kMaximumHierarchyLevels> blockCounts{};
  blockSums[0] = firstBlockSums;
  blockCounts[0] = firstBlockCount;
  size_t hierarchyLevels = 0u;

  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    return reportFailure(kCommandError, "command", "command-queue-create-failed");
  }
  id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
  if (commandBuffer == nil) {
    return reportFailure(kCommandError, "command", "command-buffer-create-failed");
  }

  const uint32_t pointCount32 = kCompactionPointCount;
  id<MTLComputeCommandEncoder> localScanEncoder =
      [commandBuffer computeCommandEncoder];
  if (localScanEncoder == nil) {
    return reportFailure(kCommandError, "command", "local-scan-encoder-create-failed");
  }
  [localScanEncoder setComputePipelineState:pipelines[0]];
  [localScanEncoder setBuffer:inputColors offset:0 atIndex:0];
  [localScanEncoder setBuffer:localOffsets offset:0 atIndex:1];
  [localScanEncoder setBuffer:firstBlockSums offset:0 atIndex:2];
  [localScanEncoder setBytes:&pointCount32 length:sizeof(pointCount32) atIndex:3];
  [localScanEncoder
      dispatchThreadgroups:MTLSizeMake(firstBlockCount, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(kRasterPointCompactBlockWidth, 1, 1)];
  [localScanEncoder endEncoding];

  for (size_t level = 0u; level < kMaximumHierarchyLevels; ++level) {
    const NSUInteger count = blockCounts[level];
    NSUInteger nextCount = 0u;
    if (count == 0u || !checkedBlockCount(count, &nextCount) ||
        count > std::numeric_limits<uint32_t>::max() ||
        nextCount > std::numeric_limits<uint32_t>::max()) {
      return reportFailure(kAllocationError,
                           "allocation",
                           "hierarchy-count-overflow");
    }
    NSUInteger offsetBytes = 0u;
    NSUInteger nextSumBytes = 0u;
    if (!checkedMultiply(count, sizeof(uint32_t), &offsetBytes) ||
        !checkedMultiply(nextCount, sizeof(uint32_t), &nextSumBytes)) {
      return reportFailure(kAllocationError,
                           "allocation",
                           "hierarchy-buffer-size-overflow");
    }
    blockOffsets[level] = sharedBuffer(offsetBytes, "block-offsets");
    id<MTLBuffer> nextBlockSums =
        sharedBuffer(nextSumBytes, "next-block-sums");
    if (blockOffsets[level] == nil || nextBlockSums == nil) {
      return reportFailure(kAllocationError,
                           "allocation",
                           "hierarchy-buffer-allocation-failed");
    }
    const uint32_t count32 = static_cast<uint32_t>(count);
    id<MTLComputeCommandEncoder> scanEncoder =
        [commandBuffer computeCommandEncoder];
    if (scanEncoder == nil) {
      return reportFailure(kCommandError,
                           "command",
                           "block-scan-encoder-create-failed");
    }
    [scanEncoder setComputePipelineState:pipelines[1]];
    [scanEncoder setBuffer:blockSums[level] offset:0 atIndex:0];
    [scanEncoder setBuffer:blockOffsets[level] offset:0 atIndex:1];
    [scanEncoder setBuffer:nextBlockSums offset:0 atIndex:2];
    [scanEncoder setBytes:&count32 length:sizeof(count32) atIndex:3];
    [scanEncoder
        dispatchThreadgroups:MTLSizeMake(nextCount, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kRasterPointCompactBlockWidth, 1, 1)];
    [scanEncoder endEncoding];
    hierarchyLevels = level + 1u;
    if (nextCount == 1u) break;
    if (level + 1u >= kMaximumHierarchyLevels) {
      return reportFailure(kAllocationError,
                           "allocation",
                           "hierarchy-too-deep");
    }
    blockSums[level + 1u] = nextBlockSums;
    blockCounts[level + 1u] = nextCount;
  }
  if (hierarchyLevels == 0u) {
    return reportFailure(kAllocationError, "allocation", "hierarchy-empty");
  }
  if (hierarchyLevels != 2u || blockCounts[0] != 257u ||
      blockCounts[1] != 2u) {
    return reportFailure(kAllocationError,
                         "allocation",
                         "unexpected-fixed-256-hierarchy-shape");
  }

  for (size_t level = hierarchyLevels; level > 1u; --level) {
    const size_t childLevel = level - 2u;
    const NSUInteger childCount = blockCounts[childLevel];
    if (childCount == 0u || childCount > std::numeric_limits<uint32_t>::max()) {
      return reportFailure(kAllocationError,
                           "allocation",
                           "offset-add-count-overflow");
    }
    const uint32_t childCount32 = static_cast<uint32_t>(childCount);
    id<MTLComputeCommandEncoder> addEncoder =
        [commandBuffer computeCommandEncoder];
    if (addEncoder == nil) {
      return reportFailure(kCommandError,
                           "command",
                           "offset-add-encoder-create-failed");
    }
    [addEncoder setComputePipelineState:pipelines[2]];
    [addEncoder setBuffer:blockOffsets[childLevel] offset:0 atIndex:0];
    [addEncoder setBuffer:blockOffsets[childLevel + 1u] offset:0 atIndex:1];
    [addEncoder setBytes:&childCount32 length:sizeof(childCount32) atIndex:2];
    [addEncoder dispatchThreads:MTLSizeMake(childCount, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kRasterPointCompactBlockWidth,
                                                 1,
                                                 1)];
    [addEncoder endEncoding];
  }

  id<MTLComputeCommandEncoder> scatterEncoder =
      [commandBuffer computeCommandEncoder];
  if (scatterEncoder == nil) {
    return reportFailure(kCommandError, "command", "scatter-encoder-create-failed");
  }
  [scatterEncoder setComputePipelineState:pipelines[3]];
  [scatterEncoder setBuffer:inputPositions offset:0 atIndex:0];
  [scatterEncoder setBuffer:inputColors offset:0 atIndex:1];
  [scatterEncoder setBuffer:localOffsets offset:0 atIndex:2];
  [scatterEncoder setBuffer:blockOffsets[0] offset:0 atIndex:3];
  [scatterEncoder setBuffer:compactPositions offset:0 atIndex:4];
  [scatterEncoder setBuffer:compactColors offset:0 atIndex:5];
  [scatterEncoder setBytes:&pointCount32 length:sizeof(pointCount32) atIndex:6];
  [scatterEncoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(kRasterPointCompactBlockWidth,
                                                 1,
                                                 1)];
  [scatterEncoder endEncoding];

  const uint32_t firstBlockCount32 = static_cast<uint32_t>(firstBlockCount);
  id<MTLComputeCommandEncoder> finalizeEncoder =
      [commandBuffer computeCommandEncoder];
  if (finalizeEncoder == nil) {
    return reportFailure(kCommandError,
                         "command",
                         "finalize-encoder-create-failed");
  }
  [finalizeEncoder setComputePipelineState:pipelines[4]];
  [finalizeEncoder setBuffer:firstBlockSums offset:0 atIndex:0];
  [finalizeEncoder setBuffer:blockOffsets[0] offset:0 atIndex:1];
  [finalizeEncoder setBytes:&firstBlockCount32
                      length:sizeof(firstBlockCount32)
                     atIndex:2];
  [finalizeEncoder setBuffer:indirectArguments offset:0 atIndex:3];
  [finalizeEncoder dispatchThreads:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [finalizeEncoder endEncoding];

  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  NSError* commandError = [commandBuffer error];
  if ([commandBuffer status] != MTLCommandBufferStatusCompleted ||
      commandError != nil) {
    const char* detail = commandError.localizedDescription.UTF8String;
    std::string message = "command-buffer-status=" +
                          std::to_string(static_cast<int>([commandBuffer status]));
    if (detail != nullptr) {
      message += ":";
      message += detail;
    }
    return reportFailure(kGpuError, "gpu", message);
  }

  const uint32_t* gpuIndirect =
      static_cast<const uint32_t*>(indirectArguments.contents);
  const std::array<uint32_t, 4> expectedIndirect = {
      reference.expectedVisible, 1u, 0u, 0u};
  for (size_t index = 0u; index < expectedIndirect.size(); ++index) {
    if (gpuIndirect[index] != expectedIndirect[index]) {
      return reportFailure(kMismatchError,
                           "mismatch",
                           "indirect-args-index=" + std::to_string(index) +
                               " expected=" +
                               std::to_string(expectedIndirect[index]) +
                               " actual=" + std::to_string(gpuIndirect[index]));
    }
  }

  const PackedFloat3Value* gpuPositions =
      static_cast<const PackedFloat3Value*>(compactPositions.contents);
  const ColorValue* gpuColors =
      static_cast<const ColorValue*>(compactColors.contents);
  for (uint32_t index = 0u; index < reference.expectedVisible; ++index) {
    if (std::memcmp(&gpuPositions[index],
                    &reference.compactPositions[index],
                    sizeof(PackedFloat3Value)) != 0) {
      return reportFailure(kMismatchError,
                           "mismatch",
                           "compact-position-index=" + std::to_string(index));
    }
    if (std::memcmp(&gpuColors[index],
                    &reference.compactColors[index],
                    sizeof(ColorValue)) != 0) {
      return reportFailure(kMismatchError,
                           "mismatch",
                           "compact-color-index=" + std::to_string(index));
    }
  }
  return 0;
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc < 3 || argv[1] == nullptr) {
      std::fprintf(stderr,
                   "usage: Chromaspace_RuntimeV2MetalProbe "
                   "METALLIB FUNCTION [FUNCTION ...]\n");
      return kUsageError;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "metal-device-unavailable\n");
      return kMetalDeviceUnavailable;
    }
    NSString* path = [NSString stringWithUTF8String:argv[1]];
    NSURL* url = [NSURL fileURLWithPath:path];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
      const char* detail = error.localizedDescription.UTF8String;
      std::fprintf(stderr,
                   "metallib-load-failed:%s\n",
                   detail ? detail : "unknown");
      return kMetallibLoadError;
    }

    bool missing = false;
    for (int index = 2; index < argc; ++index) {
      if (argv[index] == nullptr) {
        missing = true;
        continue;
      }
      NSString* name = [NSString stringWithUTF8String:argv[index]];
      if (name == nil || [library newFunctionWithName:name] == nil) {
        std::fprintf(stderr, "missing-function:%s\n", argv[index]);
        missing = true;
      }
    }
    if (missing) return kMissingFunction;

    std::array<id<MTLComputePipelineState>, kCompactionPipelineCount>
        compactionPipelines{};
    for (size_t index = 0u; index < kCompactionPipelineCount; ++index) {
      std::string pipelineError;
      compactionPipelines[index] =
          buildCompactionPipeline(device,
                                  library,
                                  kCompactionPipelineNames[index],
                                  &pipelineError);
      if (compactionPipelines[index] == nil) {
        return reportFailure(kPipelineError, "pipeline", pipelineError);
      }
    }

    const int compactionResult =
        runCompactionQualification(device, compactionPipelines);
    if (compactionResult != 0) return compactionResult;

    std::printf("status=valid device=%s functions=%d compaction=pass\n",
                device.name.UTF8String ?: "unknown",
                argc - 2);
    return 0;
  }
}
