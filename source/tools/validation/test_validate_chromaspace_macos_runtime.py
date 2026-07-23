#!/usr/bin/env python3

import importlib.util
import os
import plistlib
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "validate_chromaspace_macos_runtime.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chromaspace_macos_qualification", MODULE_PATH
)
qualification = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def write_plist(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        plistlib.dump(value, stream)


def write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class FakeNativeRunner:
    def __init__(
        self,
        *,
        signature_stdout: str = "",
        signature_stderr: str = "Signature=adhoc\nTeamIdentifier=not set\n",
        signature_returncode: int = 0,
        entitlements_stdout: str = "",
        entitlements_stderr: str = "",
        entitlements_returncode: int = 0,
        otool_stdout: str = "",
        nm_stdout: str = "",
        strings_stdout: str = "",
        lipo_stdout: str = "arm64",
    ) -> None:
        self.signature = qualification.CommandResult(
            signature_returncode, signature_stdout, signature_stderr
        )
        self.entitlements = qualification.CommandResult(
            entitlements_returncode, entitlements_stdout, entitlements_stderr
        )
        self.outputs = {
            "lipo": qualification.CommandResult(0, lipo_stdout, ""),
            "otool": qualification.CommandResult(0, otool_stdout, ""),
            "nm": qualification.CommandResult(0, nm_stdout, ""),
            "strings": qualification.CommandResult(0, strings_stdout, ""),
        }

    def run(self, args, *, timeout_seconds=30.0):
        if args[0] == "codesign":
            if args[1:3] == ["--display", "--verbose=4"]:
                return self.signature
            if args[1:4] == ["--display", "--entitlements", ":-"]:
                return self.entitlements
            raise AssertionError(f"unexpected codesign invocation: {args!r}")
        try:
            return self.outputs[args[0]]
        except KeyError as error:
            raise AssertionError(f"unexpected command: {args!r}") from error


class SpyRunner:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or qualification.CommandResult(0, "", "")

    def run(self, args, *, timeout_seconds=30.0):
        self.calls.append((tuple(args), timeout_seconds))
        return self.result


def viewer_qualification_line(**overrides) -> str:
    values = {
        "qualification": "pass",
        "frames": "3",
        "elapsed_ms": "10.5",
        "submitted": "3",
        "completed": "3",
        "failed": "0",
        "gpu_timed": "3",
        "gpu_untimed": "0",
        "gpu_total_ms": "4.5",
        "gpu_max_ms": "2.0",
        "transient_peak_reserved_bytes": "1024",
        "transient_peak_logical_bytes": "512",
        "transient_peak_submissions": "1",
        "runtime_recreations": "0",
        "runtime_context_id": "17",
        "device_registry_id": "42",
        "workspace_profile": "qualification-all-renderers-v2",
        "workspace_windows": "12",
        "renderer_coverage_mask": "8191",
        "renderer_coverage_required_mask": "8191",
        "renderer_coverage_observations": "3",
        "renderer_variants_covered": "13",
        "renderer_coverage_complete": "1",
        "plot_samples": "3",
        "plot_surface_resident_peak_bytes": "4096",
        "plot_surface_transient_peak_bytes": "6144",
        "plot_derived_resident_peak_bytes": "2048",
        "plot_derived_transient_peak_bytes": "3072",
        "plot_content_hits": "2",
        "plot_derived_hits": "2",
        "plot_derived_candidates": "1",
        "plot_derived_evictions": "0",
        "plot_surface_creates": "1",
        "plot_surface_resizes": "0",
        "plot_surface_replacements": "0",
        "plot_surface_prunes": "0",
        "presented_cpu_samples": "3",
        "presented_cpu_total_ms": "3.5",
        "presented_cpu_max_ms": "1.5",
        "scenario": "steady",
        "actions_emitted_resize": "0",
        "actions_emitted_drawable": "0",
        "actions_emitted_source": "0",
        "actions_emitted_recovery_fault": "0",
        "actions_emitted_memory": "0",
        "actions_applied_resize": "0",
        "actions_applied_drawable": "0",
        "actions_applied_source": "0",
        "actions_applied_recovery_fault": "0",
        "actions_applied_memory": "0",
        "memory_warnings": "0",
        "memory_criticals": "0",
        "memory_redraws": "0",
        "memory_trimmed_surfaces": "0",
        "memory_trimmed_surface_bytes": "0",
        "memory_trimmed_derived": "0",
        "memory_trimmed_derived_bytes": "0",
        "faults_fired": "0",
        "faults_recovered": "0",
        "source_publishes": "1",
        "source_clears": "1",
        "source_retires": "1",
        "native_creates": "1",
        "native_retires": "1",
        "native_inflight": "0",
    }
    values.update(overrides)
    return " ".join(f"{key}={value}" for key, value in values.items())


class QualificationStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.manager = root / "Chromaspace SourceExchange Manager.app"
        self.ofx = root / "Chromaspace.ofx.bundle"
        manager_contents = self.manager / "Contents"
        write_plist(
            manager_contents / "Info.plist",
            {
                "CFBundleIdentifier": qualification.MANAGER_IDENTIFIER,
                "CFBundleExecutable": qualification.MANAGER_EXECUTABLE,
                "CFBundlePackageType": "APPL",
                "LSBackgroundOnly": True,
                "LSMinimumSystemVersion": "13.0",
            },
        )
        write_executable(
            manager_contents / "MacOS" / qualification.MANAGER_EXECUTABLE
        )
        write_executable(
            manager_contents / "MacOS" / qualification.BROKER_EXECUTABLE
        )
        self.agent_path = (
            manager_contents
            / "Library"
            / "LaunchAgents"
            / qualification.AGENT_PLIST
        )
        write_plist(
            self.agent_path,
            {
                "Label": qualification.BROKER_IDENTIFIER,
                "ProgramArguments": [
                    qualification.BROKER_TEMPLATE_PROGRAM
                ],
                "MachServices": qualification.BROKER_MACH_SERVICES,
                "ProcessType": "Background",
                "ThrottleInterval": 5,
            },
        )
        write_plist(
            self.ofx / "Contents" / "Info.plist",
            {
                "CFBundleIdentifier": qualification.PLUGIN_IDENTIFIER,
                "CFBundleExecutable": qualification.PLUGIN_EXECUTABLE,
                "CFBundlePackageType": "BNDL",
            },
        )
        macos = self.ofx / "Contents" / "MacOS"
        for name in (
            qualification.PLUGIN_EXECUTABLE,
            qualification.VIEWER_EXECUTABLE,
            qualification.PRODUCER_RELAY_EXECUTABLE,
            qualification.RESOLVE_BRIDGE_EXECUTABLE,
        ):
            write_executable(macos / name)
        (macos / qualification.VIEWER_METALLIB).write_bytes(b"MTLBfixture")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, **overrides):
        values = {
            "manager_app": self.manager,
            "ofx_bundle": self.ofx,
            "skip_manager_execution": True,
        }
        values.update(overrides)
        return qualification.QualificationConfig(**values)

    def test_valid_structure_passes(self) -> None:
        report = qualification.qualify(self.config())
        self.assertTrue(report.passed)
        self.assertTrue(report.findings)

    def test_missing_metallib_fails_closed(self) -> None:
        (
            self.ofx
            / "Contents"
            / "MacOS"
            / qualification.VIEWER_METALLIB
        ).unlink()
        report = qualification.qualify(self.config())
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("ofx.viewer-metallib", failed)

    def test_agent_contract_rejects_bundle_program_key(self) -> None:
        with self.agent_path.open("rb") as stream:
            agent = plistlib.load(stream)
        agent["BundleProgram"] = "Contents/MacOS/development-broker"
        write_plist(self.agent_path, agent)
        report = qualification.qualify(self.config())
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("manager.agent-contract", failed)

    def test_agent_contract_rejects_unexpected_keys(self) -> None:
        with self.agent_path.open("rb") as stream:
            agent = plistlib.load(stream)
        agent["EnvironmentVariables"] = {"UNAPPROVED": "1"}
        write_plist(self.agent_path, agent)
        report = qualification.qualify(self.config())
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("manager.agent-contract", failed)

    def test_service_mutation_is_not_supported(self) -> None:
        report = qualification.qualify(
            self.config(service_action="register")
        )
        self.assertFalse(report.passed)
        action = [
            item for item in report.findings
            if item.name == "manager.service-action"
        ]
        self.assertEqual(len(action), 1)
        self.assertIn("mutating action forbidden", action[0].detail)

    def test_cutover_requires_macos_binary_evidence(self) -> None:
        report = qualification.qualify(self.config(mode="cutover"))
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        if os.name == "nt":
            self.assertIn("cutover.platform", failed)

    def test_exact_content_rejects_unapproved_files_and_symlinks(self) -> None:
        root = Path(self.temp.name) / "Exact.bundle"
        approved = root / "Contents" / "approved"
        approved.parent.mkdir(parents=True)
        approved.write_bytes(b"approved")
        report = qualification.Report(mode="cutover")
        qualification._validate_exact_bundle_content(
            report, "exact", root, ["Contents/approved"]
        )
        self.assertTrue(report.passed)

        (root / "Contents" / "surprise").write_bytes(b"surprise")
        report = qualification.Report(mode="cutover")
        qualification._validate_exact_bundle_content(
            report, "exact", root, ["Contents/approved"]
        )
        self.assertFalse(report.passed)
        self.assertIn("surprise", report.findings[-1].detail)

    def test_signature_metadata_is_rejected_as_unexpected_content(self) -> None:
        root = Path(self.temp.name) / "Unsigned.bundle"
        approved = root / "Contents" / "approved"
        signature = root / "Contents" / "_CodeSignature" / "CodeResources"
        approved.parent.mkdir(parents=True)
        signature.parent.mkdir(parents=True)
        approved.write_bytes(b"approved")
        signature.write_bytes(b"optional")
        report = qualification.Report(mode="cutover")
        qualification._validate_exact_bundle_content(
            report, "exact", root, ["Contents/approved"]
        )
        self.assertFalse(report.passed)
        self.assertIn("_CodeSignature", report.findings[-1].detail)

    def test_contract_lists_fail_closed(self) -> None:
        report = qualification.Report(mode="cutover")
        value = qualification._contract_string_list(
            {"required_architectures": ["arm64", "arm64"]},
            "required_architectures",
            report,
        )
        self.assertEqual(value, [])
        self.assertFalse(report.passed)

    def test_metal_probe_output_requires_compaction_marker(self) -> None:
        markers = ["status=valid", "compaction=pass"]
        report = qualification.Report(mode="cutover")
        self.assertTrue(
            qualification._check_output_markers(
                report,
                "probe",
                qualification.CommandResult(
                    0, "status=valid compaction=pass", ""
                ),
                markers,
            )
        )
        self.assertTrue(report.passed)

        report = qualification.Report(mode="cutover")
        self.assertFalse(
            qualification._check_output_markers(
                report,
                "probe",
                qualification.CommandResult(0, "status=valid", ""),
                markers,
            )
        )
        self.assertFalse(report.passed)
        self.assertIn("compaction=pass", report.findings[-1].detail)

    def test_native_binary_gate_rejects_dependency_and_marker(self) -> None:
        report = qualification.Report(mode="cutover")
        qualification._verify_native_binary(
            report,
            FakeNativeRunner(
                otool_stdout=(
                    "/System/Library/Frameworks/"
                    "ServiceManagement.framework/Versions/A/"
                    "ServiceManagement"
                ),
                strings_stdout="SMAppService",
            ),
            "fixture",
            Path("fixture"),
            ["arm64"],
            ["SMAppService"],
            ["OpenGL.framework", "ServiceManagement.framework"],
        )
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("fixture.no-forbidden-dependencies", failed)
        self.assertIn("fixture.no-forbidden-cutover-strings", failed)

    def test_native_binary_gate_accepts_identity_free_ad_hoc_without_entitlements(
        self,
    ) -> None:
        report = qualification.Report(mode="cutover")
        qualification._verify_native_binary(
            report,
            FakeNativeRunner(),
            "fixture",
            Path("fixture"),
            ["arm64"],
            [],
            [],
        )
        self.assertTrue(report.passed)

    def test_native_binary_gate_accepts_truly_unsigned_without_entitlement_query(
        self,
    ) -> None:
        runner = FakeNativeRunner(
            signature_stderr="fixture: code object is not signed at all\n",
            signature_returncode=1,
            entitlements_returncode=99,
        )
        report = qualification.Report(mode="cutover")
        qualification._verify_native_binary(
            report,
            runner,
            "fixture",
            Path("fixture"),
            ["arm64"],
            [],
            [],
        )
        self.assertTrue(report.passed)

    def test_native_binary_gate_rejects_identity_backed_signatures(self) -> None:
        signatures = {
            "developer": (
                "Signature=Developer ID Application: Example\n"
                "TeamIdentifier=ABC123\n"
            ),
            "team": "Signature=adhoc\nTeamIdentifier=ABC123\n",
            "authority": (
                "Signature=adhoc\n"
                "TeamIdentifier=not set\n"
                "Authority=Developer ID Application: Example\n"
            ),
        }
        for label, signature in signatures.items():
            with self.subTest(identity=label):
                report = qualification.Report(mode="cutover")
                qualification._verify_native_binary(
                    report,
                    FakeNativeRunner(signature_stdout=signature),
                    "fixture",
                    Path("fixture"),
                    ["arm64"],
                    [],
                    [],
                )
                self.assertFalse(report.passed)
                failed = {
                    item.name for item in report.findings if not item.passed
                }
                self.assertIn(
                    "fixture.unsigned-or-identity-free-ad-hoc", failed
                )

    def test_native_binary_gate_rejects_unclassified_codesign_failure(
        self,
    ) -> None:
        report = qualification.Report(mode="cutover")
        qualification._verify_native_binary(
            report,
            FakeNativeRunner(
                signature_stderr="codesign: internal verification error\n",
                signature_returncode=1,
            ),
            "fixture",
            Path("fixture"),
            ["arm64"],
            [],
            [],
        )
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("fixture.unsigned-or-identity-free-ad-hoc", failed)

    def test_native_binary_gate_rejects_entitlement_payload(self) -> None:
        entitlements = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
            "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
            "<plist version=\"1.0\"><dict><key>com.apple.security.app-sandbox</key>"
            "<true/></dict></plist>\n"
        )
        report = qualification.Report(mode="cutover")
        qualification._verify_native_binary(
            report,
            FakeNativeRunner(entitlements_stdout=entitlements),
            "fixture",
            Path("fixture"),
            ["arm64"],
            [],
            [],
        )
        self.assertFalse(report.passed)
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("fixture.no-entitlement-payload", failed)


class ViewerQualificationTelemetryTests(unittest.TestCase):
    REQUIRED_FIELDS = tuple(qualification.VIEWER_QUALIFICATION_FIELD_TYPES)

    def check_line(
        self,
        line: str,
        result=None,
        required_fields=None,
        expected_scenario="steady",
    ):
        report = qualification.Report(mode="cutover")
        command_result = result or qualification.CommandResult(0, line, "")
        passed = qualification._check_viewer_qualification_output(
            report,
            "viewer",
            command_result,
            3,
            required_fields if required_fields is not None else self.REQUIRED_FIELDS,
            expected_scenario,
        )
        return passed, report.findings[-1]

    def test_valid_structured_telemetry_passes(self):
        passed, finding = self.check_line(viewer_qualification_line())
        self.assertTrue(passed)
        self.assertTrue(finding.passed)

    def test_recovery_scenario_requires_exact_fault_and_recreation_counts(self):
        line = viewer_qualification_line(
            scenario="recovery-fault",
            runtime_recreations="1",
            actions_emitted_recovery_fault="2",
            actions_applied_recovery_fault="2",
            faults_fired="2",
            faults_recovered="2",
        )
        passed, finding = self.check_line(
            line, expected_scenario="recovery-fault"
        )
        self.assertTrue(passed, finding.detail)

    def test_full_scenario_matrix_accepts_only_declared_counts(self):
        suffixes = (
            "resize",
            "drawable",
            "source",
            "recovery_fault",
            "memory",
        )
        for scenario, counts in qualification.VIEWER_SCENARIO_ACTION_COUNTS.items():
            overrides = {"scenario": scenario}
            for suffix, count in zip(suffixes, counts):
                overrides[f"actions_emitted_{suffix}"] = str(count)
                overrides[f"actions_applied_{suffix}"] = str(count)
            overrides["faults_fired"] = str(counts[3])
            overrides["faults_recovered"] = str(counts[3])
            source_lifetimes = 1 + counts[2] // 2
            for field in (
                "source_publishes",
                "source_clears",
                "source_retires",
                "native_creates",
                "native_retires",
            ):
                overrides[field] = str(source_lifetimes)
            overrides["runtime_recreations"] = str(
                1 if scenario in {"recovery-fault", "soak"} else 0
            )
            if counts[4]:
                overrides.update(
                    {
                        "memory_warnings": "1",
                        "memory_criticals": "1",
                        "memory_redraws": "1",
                        "memory_trimmed_surfaces": "2",
                        "memory_trimmed_surface_bytes": "4096",
                        "memory_trimmed_derived": "3",
                        "memory_trimmed_derived_bytes": "8192",
                    }
                )
            with self.subTest(scenario=scenario):
                passed, finding = self.check_line(
                    viewer_qualification_line(**overrides),
                    expected_scenario=scenario,
                )
                self.assertTrue(passed, finding.detail)

    def test_memory_pressure_requires_positive_trim_telemetry(self):
        base = viewer_qualification_line(
            scenario="memory-pressure",
            actions_emitted_memory="2",
            actions_applied_memory="2",
            memory_warnings="1",
            memory_criticals="1",
            memory_redraws="1",
            memory_trimmed_surfaces="2",
            memory_trimmed_surface_bytes="4096",
            memory_trimmed_derived="3",
            memory_trimmed_derived_bytes="8192",
        )
        passed, finding = self.check_line(
            base, expected_scenario="memory-pressure"
        )
        self.assertTrue(passed, finding.detail)

        passed, finding = self.check_line(
            base.replace(" memory_trimmed_derived_bytes=8192", " memory_trimmed_derived_bytes=0"),
            expected_scenario="memory-pressure",
        )
        self.assertFalse(passed)
        self.assertIn("memory_trimmed_derived_bytes", finding.detail)

    def test_zero_memory_scenarios_reject_memory_telemetry(self):
        passed, finding = self.check_line(
            viewer_qualification_line(memory_redraws="1")
        )
        self.assertFalse(passed)
        self.assertIn("memory-zero-scenario", finding.detail)

    def test_negative_memory_telemetry_fails_closed(self):
        line = viewer_qualification_line(
            scenario="memory-pressure",
            actions_emitted_memory="2",
            actions_applied_memory="2",
            memory_warnings="1",
            memory_criticals="1",
            memory_redraws="1",
            memory_trimmed_surfaces="2",
            memory_trimmed_surface_bytes="4096",
            memory_trimmed_derived="3",
            memory_trimmed_derived_bytes="-1",
        )
        passed, finding = self.check_line(
            line, expected_scenario="memory-pressure"
        )
        self.assertFalse(passed)
        self.assertIn("negative=", finding.detail)

    def test_scenario_action_mismatch_fails(self):
        passed, finding = self.check_line(
            viewer_qualification_line(scenario="resize-storm"),
            expected_scenario="resize-storm",
        )
        self.assertFalse(passed)
        self.assertIn("actions=", finding.detail)

    def test_duplicate_key_fails(self):
        line = viewer_qualification_line() + " frames=3"
        passed, finding = self.check_line(line)
        self.assertFalse(passed)
        self.assertIn("duplicate key='frames'", finding.detail)

    def test_malformed_token_fails(self):
        line = viewer_qualification_line() + " malformed"
        passed, finding = self.check_line(line)
        self.assertFalse(passed)
        self.assertIn("malformed token='malformed'", finding.detail)

    def test_missing_field_fails(self):
        line = viewer_qualification_line().replace(" gpu_max_ms=2.0", "")
        passed, finding = self.check_line(line)
        self.assertFalse(passed)
        self.assertIn("gpu_max_ms", finding.detail)

    def test_incomplete_contract_fields_fail_closed_without_key_error(self):
        line = viewer_qualification_line().replace(" gpu_max_ms=2.0", "")
        passed, finding = self.check_line(
            line,
            required_fields=("qualification",),
        )
        self.assertFalse(passed)
        self.assertIn("gpu_max_ms", finding.detail)

    def test_nonfinite_field_fails(self):
        passed, finding = self.check_line(
            viewer_qualification_line(gpu_total_ms="nan")
        )
        self.assertFalse(passed)
        self.assertIn("must be finite", finding.detail)

    def test_bad_cross_field_invariant_fails(self):
        passed, finding = self.check_line(
            viewer_qualification_line(submitted="2")
        )
        self.assertFalse(passed)
        self.assertIn("serials=", finding.detail)

    def test_runtime_and_device_identity_must_be_nonzero(self):
        for field in ("runtime_context_id", "device_registry_id"):
            with self.subTest(field=field):
                passed, finding = self.check_line(
                    viewer_qualification_line(**{field: "0"})
                )
                self.assertFalse(passed)
                self.assertIn(field, finding.detail)

    def test_qualification_workspace_profile_is_exact(self):
        for field, value in (
            ("workspace_profile", "operator-workspace"),
            ("workspace_windows", "2"),
        ):
            with self.subTest(field=field):
                passed, finding = self.check_line(
                    viewer_qualification_line(**{field: value})
                )
                self.assertFalse(passed)
                self.assertIn("workspace", finding.detail)

    def test_plot_and_presented_cpu_samples_match_frames(self):
        for field in ("plot_samples", "presented_cpu_samples"):
            with self.subTest(field=field):
                passed, finding = self.check_line(
                    viewer_qualification_line(**{field: "2"})
                )
                self.assertFalse(passed)
                self.assertIn(field, finding.detail)

    def test_renderer_coverage_requires_every_model_and_both_gloss_variants(self):
        invalid = {
            "renderer_coverage_mask": "4095",
            "renderer_coverage_required_mask": "4095",
            "renderer_coverage_observations": "1",
            "renderer_variants_covered": "12",
            "renderer_coverage_complete": "0",
        }
        for field, value in invalid.items():
            with self.subTest(field=field):
                passed, finding = self.check_line(
                    viewer_qualification_line(**{field: value})
                )
                self.assertFalse(passed)
                self.assertIn("renderer_", finding.detail)

    def test_residency_peaks_and_reuse_are_required(self):
        invalid = {
            "plot_surface_resident_peak_bytes": "0",
            "plot_surface_transient_peak_bytes": "2048",
            "plot_derived_resident_peak_bytes": "0",
            "plot_derived_transient_peak_bytes": "1024",
            "plot_content_hits": "0",
            "plot_derived_hits": "0",
            "plot_derived_candidates": "0",
            "plot_surface_creates": "0",
        }
        for field, value in invalid.items():
            with self.subTest(field=field):
                passed, finding = self.check_line(
                    viewer_qualification_line(**{field: value})
                )
                self.assertFalse(passed)
                self.assertIn(field, finding.detail)

    def test_presented_cpu_metrics_are_positive_finite_and_coherent(self):
        for overrides in (
            {"presented_cpu_total_ms": "0"},
            {"presented_cpu_max_ms": "0"},
            {"presented_cpu_total_ms": "1", "presented_cpu_max_ms": "2"},
            {"presented_cpu_total_ms": "nan"},
        ):
            with self.subTest(overrides=overrides):
                passed, finding = self.check_line(
                    viewer_qualification_line(**overrides)
                )
                self.assertFalse(passed)
                self.assertIn("presented_cpu", finding.detail)

    def test_command_runner_replaces_invalid_utf8_without_losing_markers(self):
        result = qualification.CommandRunner().run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "os.write(1, b'stdout-before\\xffstdout-after'); "
                    "os.write(2, b'stderr-before\\xc3stderr-after')"
                ),
            ]
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "stdout-before\ufffdstdout-after")
        self.assertEqual(result.stderr, "stderr-before\ufffdstderr-after")

    def test_timeout_preserves_partial_output_and_fails(self):
        timeout = subprocess.TimeoutExpired(
            ["fixture"], 1, output="qualification=pass frames=3", stderr="partial"
        )
        with mock.patch.object(qualification.subprocess, "run", side_effect=timeout):
            result = qualification.CommandRunner().run(
                ["fixture"], timeout_seconds=1
            )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.returncode, -1)
        self.assertEqual(result.stdout, "qualification=pass frames=3")
        self.assertEqual(result.stderr, "partial")
        passed, finding = self.check_line(
            result.stdout,
            result=result,
        )
        self.assertFalse(passed)
        self.assertIn("timed out", finding.detail)

    def test_invalid_viewer_timeout_prevents_execution(self):
        runner = SpyRunner()
        config = QualificationConfigForTest.viewer_config(
            viewer_timeout_seconds=0
        )
        report = qualification.qualify(config, runner=runner)
        self.assertFalse(report.passed)
        self.assertEqual(runner.calls, [])
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("config.viewer-timeout-seconds", failed)

    def test_invalid_viewer_scenario_matrix_prevents_execution(self):
        runner = SpyRunner()
        config = QualificationConfigForTest.viewer_config(
            mode="cutover", viewer_scenarios=("steady", "steady")
        )
        report = qualification.qualify(config, runner=runner)
        self.assertFalse(report.passed)
        self.assertEqual(runner.calls, [])
        failed = {item.name for item in report.findings if not item.passed}
        self.assertIn("config.viewer-scenarios", failed)


class QualificationConfigForTest:
    """Small fixture factory without coupling telemetry tests to macOS paths."""

    @staticmethod
    def viewer_config(**overrides):
        root = Path("qualification-timeout-fixture")
        values = {
            "manager_app": root / "manager.app",
            "ofx_bundle": root / "Chromaspace.ofx.bundle",
            "mode": "structure",
            "skip_manager_execution": True,
        }
        values.update(overrides)
        return qualification.QualificationConfig(**values)


if __name__ == "__main__":
    unittest.main()
