#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parent / "verify_macos_metal_viewer_cutover.py"
SPEC = importlib.util.spec_from_file_location(
    "chromaspace_metal_viewer_cutover_policy", MODULE_PATH
)
assert SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class NativeViewerCutoverPolicyTests(unittest.TestCase):
    def test_repository_policy_passes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.assertEqual([], policy.verify(root))

    def test_non_void_metal_initializer_rejects_unwrapped_bare_return(self) -> None:
        fixture = """
bool initializeMetalContext(MetalContext* context, std::string* error) {
  try {
    @autoreleasepool {
      return;
    }
  } catch (...) {
  }
  return false;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
"""
        findings = policy._initialize_metal_context_return_findings(fixture)
        self.assertTrue(any("bare return escapes" in item for item in findings))

    def test_non_void_metal_initializer_allows_void_lambda_failure_exit(self) -> None:
        fixture = """
bool initializeMetalContext(MetalContext* context, std::string* error) {
  auto initializeResources = [&]() {
    @autoreleasepool {
      return;
    }
  };
  try {
    initializeResources();
  } catch (...) {
  }
  return false;
}

#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)
"""
        self.assertEqual(
            [], policy._initialize_metal_context_return_findings(fixture)
        )

    def test_command_buffer_context_definition_rejects_external_scope(self) -> None:
        fixture = """
namespace ChromaspaceMetal {
namespace {
bool contextForCommandBuffer(int commandBuffer);
}  // namespace

bool contextForCommandBuffer(int commandBuffer) {
  return commandBuffer != 0;
}
}  // namespace ChromaspaceMetal
"""
        findings = policy._command_buffer_context_namespace_findings(fixture)
        self.assertTrue(
            any("outside the anonymous namespace" in item for item in findings)
        )

    def test_command_buffer_context_definition_accepts_anonymous_scope(self) -> None:
        fixture = """
namespace ChromaspaceMetal {
namespace {
bool contextForCommandBuffer(int commandBuffer);
}  // namespace

namespace {
bool contextForCommandBuffer(int commandBuffer) {
  return commandBuffer != 0;
}
}  // namespace
}  // namespace ChromaspaceMetal
"""
        self.assertEqual(
            [], policy._command_buffer_context_namespace_findings(fixture)
        )

    def test_apple_compile_contract_rejects_late_helpers_and_implicit_member(self) -> None:
        fixture = """
struct FrameSubmissionRecord {
  SubmissionRetention retainedResources;
};
bool retainFrameTextAtlasForSubmission() { return true; }
std::mutex& frameTextAtlasMutex();
std::unordered_map<uint64_t, std::shared_ptr<FrameTextAtlasRecord>>&
frameTextAtlasRegistry();
static bool encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer() {
  return true;
}
void fillRasterSourceUniforms(const RasterSourceRequest& request,
                              RasterSourceUniforms* uniforms);
"""
        findings = policy._metal_apple_compile_contract_findings(fixture)
        self.assertEqual(4, len(findings))

    def test_apple_compile_contract_accepts_declared_helpers_and_member_init(self) -> None:
        fixture = """
struct FrameSubmissionRecord {
  SubmissionRetention retainedResources{};
};
std::mutex& frameTextAtlasMutex();
std::unordered_map<uint64_t, std::shared_ptr<FrameTextAtlasRecord>>&
frameTextAtlasRegistry();
bool retainFrameTextAtlasForSubmission() { return true; }
void fillRasterSourceUniforms(const RasterSourceRequest& request,
                              RasterSourceUniforms* uniforms);
static bool encodeRasterPointSurfaceFromTextureSourceOnCommandBuffer() {
  return true;
}
"""
        self.assertEqual(
            [], policy._metal_apple_compile_contract_findings(fixture)
        )

    def test_native_source_exception_contract_rejects_mismatched_definitions(
        self,
    ) -> None:
        header = """
static bool createCallback(int value) noexcept;
bool createInternal(int value);
"""
        source = """
bool NativeSourceFixtureBackend::createCallback(int value) {
  return value != 0;
}
bool NativeSourceFixtureBackend::createInternal(int value) noexcept {
  return value != 0;
}
"""
        findings = policy._native_source_exception_contract_findings(
            header, source
        )
        self.assertEqual(2, len(findings))

    def test_native_source_exception_contract_accepts_callback_boundary(
        self,
    ) -> None:
        header = """
static bool createCallback(int value) noexcept;
bool createInternal(int value);
"""
        source = """
bool NativeSourceFixtureBackend::createCallback(int value) noexcept {
  return value != 0;
}
bool NativeSourceFixtureBackend::createInternal(int value) {
  return value != 0;
}
"""
        self.assertEqual(
            [],
            policy._native_source_exception_contract_findings(header, source),
        )

    def test_qualification_campaign_switch_requires_explicit_soak_case(
        self,
    ) -> None:
        fixture = """
bool Campaign::next() {
  if (config_.scenario == Scenario::Soak) {
    candidateForSoak();
  } else {
    switch (config_.scenario) {
      case Scenario::Steady:
      case Scenario::Count:
        break;
    }
  }
}

bool Campaign::acknowledgePending() {
  return true;
}
"""
        findings = policy._qualification_campaign_switch_findings(fixture)
        self.assertTrue(any("Scenario::Soak" in item for item in findings))

    def test_source_exchange_objc_api_rejects_data_length_on_nsdata(
        self,
    ) -> None:
        fixture = """
NSData* token = [NSData dataWithLength:8];
"""
        findings = policy._source_exchange_objc_api_findings(fixture)
        self.assertEqual(1, len(findings))
        self.assertIn("NSData does not declare +dataWithLength:", findings[0])

    def test_source_exchange_objc_api_accepts_mutable_data_length(
        self,
    ) -> None:
        fixture = """
NSData* token = [NSMutableData dataWithLength:8];
"""
        self.assertEqual(
            [], policy._source_exchange_objc_api_findings(fixture)
        )

    def test_metal_default_backend_linkage_rejects_os_inferred_defaults(
        self,
    ) -> None:
        source = """
#if !defined(__APPLE__)
const Backend* defaultBackend() noexcept { return &fallback; }
#endif
"""
        cmake = """
target_compile_definitions(Chromaspace_CubeViewer PRIVATE
  CHROMASPACE_METAL_NATIVE_ONLY)
target_compile_definitions(Chromaspace_MetalViewerCutover PRIVATE
  CHROMASPACE_METAL_NATIVE_ONLY)
"""
        findings = policy._metal_default_backend_linkage_findings(
            cmake, {"frame": source, "plot": source, "runtime": source}
        )
        self.assertEqual(8, len(findings))

    def test_metal_default_backend_linkage_accepts_explicit_capability(
        self,
    ) -> None:
        macro = policy.EXTERNAL_DEFAULT_BACKENDS_MACRO
        source = f"""
#if !defined({macro})
const Backend* defaultBackend() noexcept {{ return &fallback; }}
#endif
"""
        cmake = f"""
target_compile_definitions(Chromaspace_CubeViewer PRIVATE
  CHROMASPACE_METAL_NATIVE_ONLY
  {macro})
target_compile_definitions(Chromaspace_MetalViewerCutover PRIVATE
  CHROMASPACE_METAL_NATIVE_ONLY
  {macro})
"""
        self.assertEqual(
            [],
            policy._metal_default_backend_linkage_findings(
                cmake, {"frame": source, "plot": source, "runtime": source}
            ),
        )

    def test_forbidden_token_in_executor_source_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original_read = policy._read

        def injected_read(candidate_root: Path, relative: str) -> str:
            text = original_read(candidate_root, relative)
            if relative == (
                "tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp"
            ):
                return f"{text}\n// injected forbidden token: OpenGL\n"
            return text

        with mock.patch.object(policy, "_read", side_effect=injected_read):
            findings = policy.verify(root)

        self.assertIn(
            "tools/cube_viewer_stub/metal/ChromaspaceMetalFrameExecutor.cpp: "
            "executor contains forbidden token: OpenGL",
            findings,
        )

    def test_forbidden_token_in_resident_source_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original_read = policy._read

        def injected_read(candidate_root: Path, relative: str) -> str:
            text = original_read(candidate_root, relative)
            if relative == (
                "tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp"
            ):
                return f"{text}\n// injected forbidden token: OpenGL\n"
            return text

        with mock.patch.object(policy, "_read", side_effect=injected_read):
            findings = policy.verify(root)

        self.assertIn(
            "tools/cube_viewer_stub/metal/ChromaspaceResidentSourceSession.cpp: "
            "resident source contains forbidden token: OpenGL",
            findings,
        )

    def test_retired_metal_registry_token_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original_read = policy._read

        def injected_read(candidate_root: Path, relative: str) -> str:
            text = original_read(candidate_root, relative)
            if relative == "tools/cube_viewer_stub/metal/ChromaspaceMetal.mm":
                return f"{text}\nvoid glossFieldRegistry();\n"
            return text

        with mock.patch.object(policy, "_read", side_effect=injected_read):
            findings = policy.verify(root)

        self.assertIn(
            "Metal source contains forbidden token: glossFieldRegistry",
            findings,
        )

    def test_raw_client_lifecycle_in_production_entrypoint_is_reported(self) -> None:
        root = Path(__file__).resolve().parents[2]
        original_read = policy._read

        def injected_read(candidate_root: Path, relative: str) -> str:
            text = original_read(candidate_root, relative)
            if relative == "tools/cube_viewer_stub/main.cpp":
                return f"{text}\ncreateSourceViewerClient(raw);\n"
            return text

        with mock.patch.object(policy, "_read", side_effect=injected_read):
            findings = policy.verify(root)

        self.assertIn(
            "production entrypoint contains forbidden token: "
            "createSourceViewerClient(",
            findings,
        )

    def test_legacy_token_outside_guard_fails_closed(self) -> None:
        source = "#import <OpenGL/gl.h>\n"
        violations = policy._guarded_token_violations(
            source, ["#import <OpenGL/gl.h>"]
        )
        self.assertTrue(violations)

    def test_legacy_token_inside_compatibility_guard_is_allowed(self) -> None:
        source = (
            "#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)\n"
            "#import <OpenGL/gl.h>\n"
            "#endif\n"
        )
        self.assertEqual(
            [],
            policy._guarded_token_violations(
                source, ["#import <OpenGL/gl.h>"]
            ),
        )

    def test_extra_add_executable_source_fails_closed(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover
          tools/cube_viewer_metal_native/main.mm
          src/ChromaspaceViewerSession.cpp
          tools/cube_viewer_stub/metal/ChromaspaceMetal.mm
          src/metal/ChromaspaceSourceExchangeState.cpp
          extra.mm)
        """
        findings = policy._cmake_source_closure_findings(
            fixture,
            "Chromaspace_MetalViewerCutover",
            [
                "tools/cube_viewer_metal_native/main.mm",
                "src/ChromaspaceViewerSession.cpp",
                "tools/cube_viewer_stub/metal/ChromaspaceMetal.mm",
                "src/metal/ChromaspaceSourceExchangeState.cpp",
            ],
        )
        self.assertTrue(any("source closure mismatch" in item for item in findings))

    def test_later_target_sources_addition_fails_closed(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover
          tools/cube_viewer_metal_native/main.mm
          src/ChromaspaceViewerSession.cpp
          tools/cube_viewer_stub/metal/ChromaspaceMetal.mm
          src/metal/ChromaspaceSourceExchangeState.cpp)
        target_sources(Chromaspace_MetalViewerCutover PRIVATE extra.mm)
        """
        findings = policy._cmake_source_closure_findings(
            fixture,
            "Chromaspace_MetalViewerCutover",
            [
                "tools/cube_viewer_metal_native/main.mm",
                "src/ChromaspaceViewerSession.cpp",
                "tools/cube_viewer_stub/metal/ChromaspaceMetal.mm",
                "src/metal/ChromaspaceSourceExchangeState.cpp",
            ],
        )
        self.assertTrue(any("target_sources" in item for item in findings))

    def test_canonical_source_variable_is_resolved_exactly(self) -> None:
        fixture = """
        set(CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES
          native_main.mm
          renderer.mm)
        add_executable(Chromaspace_MetalViewerCutover
          ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
        """
        self.assertEqual(
            [],
            policy._cmake_source_closure_findings(
                fixture,
                "Chromaspace_MetalViewerCutover",
                ["native_main.mm", "renderer.mm"],
                source_list_variable="CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES",
            ),
        )

    def test_canonical_source_variable_drift_fails_closed(self) -> None:
        fixture = """
        set(CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES
          native_main.mm
          renderer.mm
          legacy_main.cpp)
        add_executable(Chromaspace_MetalViewerCutover
          ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
        """
        findings = policy._cmake_source_closure_findings(
            fixture,
            "Chromaspace_MetalViewerCutover",
            ["native_main.mm", "renderer.mm"],
            source_list_variable="CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES",
        )
        self.assertTrue(any("source closure mismatch" in item for item in findings))

    def test_product_and_qualification_source_lists_compose_exactly(self) -> None:
        fixture = """
        set(PRODUCT_SOURCES native_main.mm renderer.mm)
        set(QUALIFICATION_SOURCES campaign.cpp fault.cpp)
        add_executable(Chromaspace_MetalViewerCutover
          ${PRODUCT_SOURCES}
          ${QUALIFICATION_SOURCES})
        """
        self.assertEqual(
            [],
            policy._cmake_source_closure_findings(
                fixture,
                "Chromaspace_MetalViewerCutover",
                ["native_main.mm", "renderer.mm", "campaign.cpp", "fault.cpp"],
                source_list_variables=[
                    "PRODUCT_SOURCES",
                    "QUALIFICATION_SOURCES",
                ],
            ),
        )

    def test_qualification_source_list_drift_fails_closed(self) -> None:
        fixture = """
        set(QUALIFICATION_SOURCES campaign.cpp fault.cpp shipped.cpp)
        """
        findings = policy._cmake_list_closure_findings(
            fixture,
            "QUALIFICATION_SOURCES",
            ["campaign.cpp", "fault.cpp"],
        )
        self.assertTrue(any("source closure mismatch" in item for item in findings))

    def test_apple_product_rejects_legacy_linkage(self) -> None:
        fixture = """
        if(NOT APPLE)
          find_package(OpenGL REQUIRED)
          FetchContent_MakeAvailable(glfw)
        endif()
        if(APPLE)
          set(CHROMASPACE_VIEWER_SOURCES
            ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
        endif()
        add_executable(Chromaspace_CubeViewer WIN32
          ${CHROMASPACE_VIEWER_SOURCES})
        if(APPLE)
          target_link_libraries(Chromaspace_CubeViewer PRIVATE glfw Metal)
        endif()
        """
        findings = policy._apple_product_policy_findings(
            fixture,
            "Chromaspace_CubeViewer",
            "CHROMASPACE_VIEWER_SOURCES",
            "CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES",
            [],
            ["glfw", "OpenGL::GL"],
        )
        self.assertIn(
            "Apple product mutation contains forbidden token: glfw", findings
        )

    def test_unguarded_product_legacy_linkage_fails_closed(self) -> None:
        fixture = """
        if(NOT APPLE)
          find_package(OpenGL REQUIRED)
          FetchContent_MakeAvailable(glfw)
        endif()
        if(APPLE)
          set(CHROMASPACE_VIEWER_SOURCES
            ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
        endif()
        add_executable(Chromaspace_CubeViewer WIN32
          ${CHROMASPACE_VIEWER_SOURCES})
        target_link_libraries(Chromaspace_CubeViewer PRIVATE OpenGL::GL)
        """
        findings = policy._apple_product_policy_findings(
            fixture,
            "Chromaspace_CubeViewer",
            "CHROMASPACE_VIEWER_SOURCES",
            "CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES",
            [],
            ["glfw", "OpenGL::GL"],
        )
        self.assertIn(
            "Apple product mutation contains forbidden token: OpenGL::GL",
            findings,
        )

    def test_unguarded_opengl_discovery_fails_closed(self) -> None:
        fixture = """
        find_package(OpenGL REQUIRED)
        FetchContent_MakeAvailable(glfw)
        if(APPLE)
          set(CHROMASPACE_VIEWER_SOURCES
            ${CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES})
        endif()
        add_executable(Chromaspace_CubeViewer WIN32
          ${CHROMASPACE_VIEWER_SOURCES})
        """
        findings = policy._apple_product_policy_findings(
            fixture,
            "Chromaspace_CubeViewer",
            "CHROMASPACE_VIEWER_SOURCES",
            "CHROMASPACE_METAL_NATIVE_VIEWER_SOURCES",
            [],
            ["glfw", "OpenGL::GL"],
        )
        self.assertIn(
            "OpenGL/GLFW discovery is not enclosed by NOT APPLE", findings
        )

    def test_glfw_and_nsgl_target_dependency_is_forbidden(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover main.mm)
        target_link_libraries(Chromaspace_MetalViewerCutover PRIVATE glfw NSOpenGL)
        """
        findings = policy._target_policy_findings(
            fixture,
            [],
            ["GLFW", "glfw", "NSOpenGL"],
        )
        self.assertIn("canary block contains forbidden token: glfw", findings)
        self.assertIn("canary block contains forbidden token: NSOpenGL", findings)

    def test_late_target_link_mutation_is_forbidden(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover main.mm)
        target_link_libraries(Chromaspace_MetalViewerCutover PRIVATE Metal)
        target_link_libraries(Chromaspace_MetalViewerCutover PRIVATE
          \"-framework OpenGL\")
        """
        findings = policy._target_mutation_policy_findings(
            fixture,
            "Chromaspace_MetalViewerCutover",
            ["-framework OpenGL"],
        )
        self.assertTrue(
            any(
                "target_link_libraries(Chromaspace_MetalViewerCutover)"
                in item
                and "-framework OpenGL" in item
                for item in findings
            )
        )

    def test_set_property_target_mutation_is_forbidden(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover main.mm)
        set_property(TARGET Chromaspace_MetalViewerCutover
          PROPERTY LINK_LIBRARIES \"-framework OpenGL\")
        """
        findings = policy._target_mutation_policy_findings(
            fixture,
            "Chromaspace_MetalViewerCutover",
            ["-framework OpenGL"],
        )
        self.assertTrue(
            any(
                "set_property(Chromaspace_MetalViewerCutover)" in item
                and "-framework OpenGL" in item
                for item in findings
            )
        )

    def test_unrelated_set_property_target_is_ignored(self) -> None:
        fixture = """
        add_executable(Chromaspace_MetalViewerCutover main.mm)
        set_property(TARGET AnotherTarget
          PROPERTY LINK_LIBRARIES \"-framework OpenGL\")
        """
        self.assertEqual(
            [],
            policy._target_mutation_policy_findings(
                fixture,
                "Chromaspace_MetalViewerCutover",
                ["-framework OpenGL"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
