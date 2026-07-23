#!/usr/bin/env python3
"""Fail-closed qualification for the Chromaspace macOS Runtime V2 artifacts.

The default ``structure`` mode is read-only. ``cutover`` adds binary,
package-payload, native-only, and unsigned-or-identity-free Mach-O checks. The
cutover accepts a genuinely unsigned Mach-O or the identity-free ad-hoc load
command an Apple linker may emit, but rejects identity-backed signatures, Team
IDs, authorities, entitlement payloads, and bundle-level ``_CodeSignature``
metadata. Notarization and distribution membership remain outside this local
validator and are not permitted deployment dependencies. The tool never
registers or unregisters a service.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import plistlib
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence


DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0
DEFAULT_VIEWER_TIMEOUT_SECONDS = 120.0
MAX_VIEWER_TIMEOUT_SECONDS = 3600.0
VIEWER_QUALIFICATION_SCENARIOS = (
    "steady",
    "resize-storm",
    "drawable-loss",
    "source-churn",
    "recovery-fault",
    "memory-pressure",
    "soak",
)
VIEWER_SCENARIO_ACTION_COUNTS = {
    "steady": (0, 0, 0, 0, 0),
    "resize-storm": (8, 0, 0, 0, 0),
    "drawable-loss": (0, 4, 0, 0, 0),
    "source-churn": (0, 0, 6, 0, 0),
    "recovery-fault": (0, 0, 0, 2, 0),
    "memory-pressure": (0, 0, 0, 0, 2),
    "soak": (4, 4, 4, 2, 2),
}

VIEWER_QUALIFICATION_FIELD_TYPES = {
    "qualification": str,
    "frames": int,
    "elapsed_ms": float,
    "submitted": int,
    "completed": int,
    "failed": int,
    "gpu_timed": int,
    "gpu_untimed": int,
    "gpu_total_ms": float,
    "gpu_max_ms": float,
    "transient_peak_reserved_bytes": int,
    "transient_peak_logical_bytes": int,
    "transient_peak_submissions": int,
    "runtime_recreations": int,
    "runtime_context_id": int,
    "device_registry_id": int,
    "scenario": str,
    "workspace_profile": str,
    "workspace_windows": int,
    "renderer_coverage_mask": int,
    "renderer_coverage_required_mask": int,
    "renderer_coverage_observations": int,
    "renderer_variants_covered": int,
    "renderer_coverage_complete": int,
    "plot_samples": int,
    "plot_surface_resident_peak_bytes": int,
    "plot_surface_transient_peak_bytes": int,
    "plot_derived_resident_peak_bytes": int,
    "plot_derived_transient_peak_bytes": int,
    "plot_content_hits": int,
    "plot_derived_hits": int,
    "plot_derived_candidates": int,
    "plot_derived_evictions": int,
    "plot_surface_creates": int,
    "plot_surface_resizes": int,
    "plot_surface_replacements": int,
    "plot_surface_prunes": int,
    "presented_cpu_samples": int,
    "presented_cpu_total_ms": float,
    "presented_cpu_max_ms": float,
    "actions_emitted_resize": int,
    "actions_emitted_drawable": int,
    "actions_emitted_source": int,
    "actions_emitted_recovery_fault": int,
    "actions_emitted_memory": int,
    "actions_applied_resize": int,
    "actions_applied_drawable": int,
    "actions_applied_source": int,
    "actions_applied_recovery_fault": int,
    "actions_applied_memory": int,
    "memory_warnings": int,
    "memory_criticals": int,
    "memory_redraws": int,
    "memory_trimmed_surfaces": int,
    "memory_trimmed_surface_bytes": int,
    "memory_trimmed_derived": int,
    "memory_trimmed_derived_bytes": int,
    "faults_fired": int,
    "faults_recovered": int,
    "source_publishes": int,
    "source_clears": int,
    "source_retires": int,
    "native_creates": int,
    "native_retires": int,
    "native_inflight": int,
}


def _command_output_text(value: object) -> str:
    """Normalize subprocess output without discarding partial timeout data."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


MANAGER_EXECUTABLE = "Chromaspace SourceExchange Manager"
MANAGER_IDENTIFIER = "com.chromaspace.SourceExchangeManager"
AGENT_PLIST = "com.chromaspace.SourceExchangeBroker.plist"
BROKER_EXECUTABLE = "com.chromaspace.SourceExchangeBroker"
BROKER_IDENTIFIER = "com.chromaspace.SourceExchangeBroker"
BROKER_TEMPLATE_PROGRAM = (
    "__CHROMASPACE_SOURCE_EXCHANGE_BROKER_EXECUTABLE__"
)
BROKER_MACH_SERVICES = {
    "com.chromaspace.SourceExchangeBroker": True,
    "com.chromaspace.SourceExchangeBootstrap": True,
}
PLUGIN_IDENTIFIER = "com.moazelgabry.chromaspace"
PLUGIN_EXECUTABLE = "Chromaspace.ofx"
VIEWER_EXECUTABLE = "Chromaspace_CubeViewer"
PRODUCER_RELAY_EXECUTABLE = "Chromaspace_SourceExchangeProducerRelay"
RESOLVE_BRIDGE_EXECUTABLE = "Chromaspace_ResolveBridge"
VIEWER_METALLIB = "ChromaspaceViewer.metallib"
LEGACY_BROKER_TEMPLATE = (
    "Contents/Resources/"
    "com.chromaspace.SourceExchangeBroker.plist.template"
)
SHIPPED_ENTITLEMENT_FILE = (
    "Contents/Resources/ChromaspaceSourceExchangeClient.entitlements"
)
FORBIDDEN_AGENT_KEYS = {
    "Program",
    "BundleProgram",
    "RunAtLoad",
    "KeepAlive",
}
FORBIDDEN_GLOBAL_IOSURFACE_SYMBOLS = {
    "IOSurfaceLookup",
    "IOSurfaceLookupFromMachPort",
    "IOSurfaceGetID",
    "kIOSurfaceIsGlobal",
}


@dataclass(frozen=True)
class Finding:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    mode: str
    findings: list[Finding] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.findings.append(Finding(name, bool(passed), detail))

    @property
    def passed(self) -> bool:
        return bool(self.findings) and all(item.passed for item in self.findings)

    def to_dict(self) -> dict:
        return {
            "schema": 1,
            "mode": self.mode,
            "passed": self.passed,
            "findings": [asdict(item) for item in self.findings],
        }


@dataclass(frozen=True)
class QualificationConfig:
    manager_app: Path
    ofx_bundle: Path
    mode: str = "structure"
    installer_pkg: Optional[Path] = None
    metal_probe: Optional[Path] = None
    viewer_canary: Optional[Path] = None
    viewer_frames: int = 300
    viewer_scenarios: tuple[str, ...] = VIEWER_QUALIFICATION_SCENARIOS
    viewer_timeout_seconds: float = DEFAULT_VIEWER_TIMEOUT_SECONDS
    contract_path: Path = (
        Path(__file__).resolve().parent / "macos_runtime_v2_contract.json"
    )
    service_action: str = "none"
    skip_manager_execution: bool = False
    expected_ofx_install_path: str = (
        "/Library/OFX/Plugins/Chromaspace.ofx.bundle"
    )


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined(self) -> str:
        return "\n".join(part for part in (self.stdout, self.stderr) if part)


class CommandRunner:
    def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> CommandResult:
        if not _valid_command_timeout(timeout_seconds):
            raise ValueError(
                f"command timeout must be finite and positive; "
                f"got={timeout_seconds!r}"
            )
        try:
            completed = subprocess.run(
                list(args),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return CommandResult(
                -1,
                _command_output_text(error.stdout),
                _command_output_text(error.stderr),
                timed_out=True,
            )
        return CommandResult(
            completed.returncode,
            _command_output_text(completed.stdout),
            _command_output_text(completed.stderr),
        )


def _load_contract(path: Path, report: Report) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != 1:
            raise ValueError("expected schema 1")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        report.add("contract", False, f"{path}: {error}")
        return {}
    report.add("contract", True, str(path))
    return value


def _load_plist(path: Path, report: Report, name: str) -> Optional[dict]:
    try:
        with path.open("rb") as stream:
            value = plistlib.load(stream)
        if not isinstance(value, dict):
            raise ValueError("top-level value is not a dictionary")
    except (OSError, plistlib.InvalidFileException, ValueError) as error:
        report.add(name, False, f"{path}: {error}")
        return None
    report.add(name, True, str(path))
    return value


def _regular_file(path: Path, executable: bool = False) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    if path.is_symlink():
        return False, f"symlink forbidden: {path}"
    if not path.is_file():
        return False, f"not a regular file: {path}"
    if executable and os.name != "nt" and not os.access(path, os.X_OK):
        return False, f"not executable: {path}"
    if path.stat().st_size <= 0:
        return False, f"empty file: {path}"
    return True, str(path)


def _check_file(
    report: Report, name: str, path: Path, executable: bool = False
) -> bool:
    passed, detail = _regular_file(path, executable)
    report.add(name, passed, detail)
    return passed


def _runner_run(
    runner: CommandRunner,
    args: Sequence[str],
    *,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> CommandResult:
    """Run a command with a bounded timeout while retaining simple test doubles.

    Older custom runners in policy tests accepted only ``run(args)``.  They do
    not launch subprocesses, so invoking them without the timeout keyword keeps
    those tests useful while the real ``CommandRunner`` bounds every process.
    """

    run = runner.run
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        signature = None
    accepts_timeout = signature is None or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "timeout_seconds"
        for parameter in signature.parameters.values()
    ) if signature is not None else True
    if accepts_timeout:
        return run(args, timeout_seconds=timeout_seconds)
    return run(args)


def _run_check(
    report: Report,
    runner: CommandRunner,
    name: str,
    args: Sequence[str],
    accepted_codes: Iterable[int] = (0,),
    required_text: Optional[str] = None,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> CommandResult:
    result = _runner_run(runner, args, timeout_seconds=timeout_seconds)
    accepted = set(accepted_codes)
    passed = not result.timed_out and result.returncode in accepted
    if required_text is not None:
        passed = passed and required_text in result.combined
    detail = (
        f"timed out after {timeout_seconds:g}s; output={result.combined!r}"
        if result.timed_out
        else result.combined or f"exit={result.returncode}"
    )
    report.add(name, passed, detail)
    return result


def _check_output_markers(
    report: Report,
    name: str,
    result: CommandResult,
    markers: Sequence[str],
) -> bool:
    missing = [marker for marker in markers if marker not in result.combined]
    passed = (
        not result.timed_out
        and result.returncode == 0
        and bool(markers)
        and not missing
    )
    report.add(
        name,
        passed,
        (
            (
                f"exit={result.returncode} timed_out={result.timed_out} "
                f"missing={missing!r}"
            )
            if missing or result.returncode != 0 or result.timed_out
            else f"markers={len(markers)}"
        ),
    )
    return passed


def _valid_viewer_timeout(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        not isinstance(value, bool)
        and math.isfinite(numeric)
        and 1.0 <= numeric <= MAX_VIEWER_TIMEOUT_SECONDS
    )


def _valid_command_timeout(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return not isinstance(value, bool) and math.isfinite(numeric) and numeric > 0.0


def _valid_viewer_scenarios(value: object) -> bool:
    if not isinstance(value, (tuple, list)) or not value:
        return False
    return (
        all(
            isinstance(item, str) and item in VIEWER_QUALIFICATION_SCENARIOS
            for item in value
        )
        and len(set(value)) == len(value)
    )


def _parse_viewer_qualification_output(
    result: CommandResult,
    requested_frames: int,
    required_fields: Sequence[str],
    expected_scenario: str = "steady",
) -> tuple[bool, str]:
    """Parse and validate the canary's one-line, machine-readable result."""

    if result.timed_out:
        return False, "timed out; partial-output-preserved"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return (
            False,
            f"expected exactly one nonempty stdout line; found={len(lines)}",
        )
    tokens = lines[0].split()
    if not tokens:
        return False, "empty qualification line"
    if not tokens[0].startswith("qualification="):
        return False, "first token must be qualification=<pass|fail>"

    values: dict[str, str] = {}
    for token in tokens:
        if token.count("=") != 1:
            return False, f"malformed token={token!r}"
        key, value = token.split("=", 1)
        if (
            not key
            or not value
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key) is None
        ):
            return False, f"malformed token={token!r}"
        if key in values:
            return False, f"duplicate key={key!r}"
        values[key] = value

    qualification = values.get("qualification")
    if qualification not in {"pass", "fail"}:
        return False, "qualification must be pass or fail"
    all_required_fields = list(
        dict.fromkeys((*required_fields, *VIEWER_QUALIFICATION_FIELD_TYPES))
    )
    missing = [field for field in all_required_fields if field not in values]
    if missing:
        return False, f"missing fields={missing!r}"

    typed: dict[str, object] = {}
    for field, field_type in VIEWER_QUALIFICATION_FIELD_TYPES.items():
        raw = values[field]
        if field_type is str:
            typed[field] = raw
            continue
        if field_type is int:
            if re.fullmatch(r"[+-]?\d+", raw) is None:
                return False, f"field={field!r} must be an integer"
            typed[field] = int(raw)
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError):
            return False, f"field={field!r} must be a number"
        if not math.isfinite(numeric):
            return False, f"field={field!r} must be finite"
        typed[field] = numeric

    if qualification != "pass":
        return False, "qualification=fail"

    reasons: list[str] = []
    integer_fields = [
        field
        for field, field_type in VIEWER_QUALIFICATION_FIELD_TYPES.items()
        if field_type is int
    ]
    negative = [field for field in integer_fields if typed[field] < 0]
    if negative:
        reasons.append(f"negative={negative!r}")
    if typed["frames"] != requested_frames:
        reasons.append(
            f"frames={typed['frames']} requested={requested_frames}"
        )
    if not (
        typed["submitted"] == requested_frames
        and typed["completed"] == requested_frames
    ):
        reasons.append(
            f"serials=submitted:{typed['submitted']} "
            f"completed:{typed['completed']} requested:{requested_frames}"
        )
    if typed["failed"] != 0:
        reasons.append(f"failed={typed['failed']}")
    if typed["gpu_timed"] <= 0:
        reasons.append(f"gpu_timed={typed['gpu_timed']}")
    if typed["gpu_timed"] + typed["gpu_untimed"] != typed["completed"]:
        reasons.append(
            f"gpu_counts={typed['gpu_timed']}+{typed['gpu_untimed']} "
            f"completed={typed['completed']}"
        )
    if typed["runtime_context_id"] <= 0:
        reasons.append(f"runtime_context_id={typed['runtime_context_id']}")
    if typed["device_registry_id"] <= 0:
        reasons.append(f"device_registry_id={typed['device_registry_id']}")
    if typed["gpu_total_ms"] <= 0 or typed["gpu_max_ms"] <= 0:
        reasons.append(
            f"gpu_ms=total:{typed['gpu_total_ms']} max:{typed['gpu_max_ms']}"
        )
    elif typed["gpu_max_ms"] > typed["gpu_total_ms"]:
        reasons.append(
            f"gpu_max_ms={typed['gpu_max_ms']} "
            f"> gpu_total_ms={typed['gpu_total_ms']}"
        )
    if typed["elapsed_ms"] < 0:
        reasons.append(f"elapsed_ms={typed['elapsed_ms']}")
    if (
        typed["transient_peak_reserved_bytes"]
        < typed["transient_peak_logical_bytes"]
    ):
        reasons.append(
            "transient reserved bytes below logical bytes"
        )
    if typed["transient_peak_submissions"] > 3:
        reasons.append(
            f"transient_peak_submissions={typed['transient_peak_submissions']}"
        )
    if typed["workspace_profile"] != "qualification-all-renderers-v2":
        reasons.append(
            f"workspace_profile={typed['workspace_profile']!r} "
            "expected='qualification-all-renderers-v2'"
        )
    if typed["workspace_windows"] != 12:
        reasons.append(
            f"workspace_windows={typed['workspace_windows']} expected=12"
        )
    expected_renderer_coverage_mask = (1 << 13) - 1
    if typed["renderer_coverage_required_mask"] != expected_renderer_coverage_mask:
        reasons.append(
            "renderer_coverage_required_mask="
            f"{typed['renderer_coverage_required_mask']} "
            f"expected={expected_renderer_coverage_mask}"
        )
    if typed["renderer_coverage_mask"] != typed["renderer_coverage_required_mask"]:
        reasons.append(
            f"renderer_coverage_mask={typed['renderer_coverage_mask']} "
            "does not match required="
            f"{typed['renderer_coverage_required_mask']}"
        )
    if typed["renderer_coverage_observations"] < 2:
        reasons.append(
            "renderer_coverage_observations="
            f"{typed['renderer_coverage_observations']} expected>=2"
        )
    elif typed["renderer_coverage_observations"] > requested_frames:
        reasons.append(
            "renderer_coverage_observations="
            f"{typed['renderer_coverage_observations']} "
            f"exceeds requested_frames={requested_frames}"
        )
    if typed["renderer_variants_covered"] != 13:
        reasons.append(
            f"renderer_variants_covered={typed['renderer_variants_covered']} "
            "expected=13"
        )
    elif typed["renderer_variants_covered"] != typed[
        "renderer_coverage_mask"
    ].bit_count():
        reasons.append(
            f"renderer_variants_covered={typed['renderer_variants_covered']} "
            "does not match coverage mask population="
            f"{typed['renderer_coverage_mask'].bit_count()}"
        )
    if typed["renderer_coverage_complete"] != 1:
        reasons.append(
            "renderer_coverage_complete="
            f"{typed['renderer_coverage_complete']} expected=1"
        )
    if typed["plot_samples"] != requested_frames:
        reasons.append(
            f"plot_samples={typed['plot_samples']} requested={requested_frames}"
        )
    if typed["presented_cpu_samples"] != requested_frames:
        reasons.append(
            "presented_cpu_samples="
            f"{typed['presented_cpu_samples']} requested={requested_frames}"
        )
    if (
        typed["presented_cpu_total_ms"] <= 0
        or typed["presented_cpu_max_ms"] <= 0
    ):
        reasons.append(
            "presented_cpu_ms="
            f"total:{typed['presented_cpu_total_ms']} "
            f"max:{typed['presented_cpu_max_ms']}"
        )
    elif typed["presented_cpu_max_ms"] > typed["presented_cpu_total_ms"]:
        reasons.append(
            f"presented_cpu_max_ms={typed['presented_cpu_max_ms']} "
            f"> presented_cpu_total_ms={typed['presented_cpu_total_ms']}"
        )
    if typed["plot_surface_resident_peak_bytes"] <= 0:
        reasons.append(
            "plot_surface_resident_peak_bytes="
            f"{typed['plot_surface_resident_peak_bytes']}"
        )
    if (
        typed["plot_surface_transient_peak_bytes"]
        < typed["plot_surface_resident_peak_bytes"]
    ):
        reasons.append(
            "plot_surface_transient_peak_bytes="
            f"{typed['plot_surface_transient_peak_bytes']} below resident="
            f"{typed['plot_surface_resident_peak_bytes']}"
        )
    if typed["plot_derived_resident_peak_bytes"] <= 0:
        reasons.append(
            "plot_derived_resident_peak_bytes="
            f"{typed['plot_derived_resident_peak_bytes']}"
        )
    if (
        typed["plot_derived_transient_peak_bytes"]
        < typed["plot_derived_resident_peak_bytes"]
    ):
        reasons.append(
            "plot_derived_transient_peak_bytes="
            f"{typed['plot_derived_transient_peak_bytes']} below resident="
            f"{typed['plot_derived_resident_peak_bytes']}"
        )
    for field in (
        "plot_content_hits",
        "plot_derived_hits",
        "plot_derived_candidates",
        "plot_surface_creates",
    ):
        if typed[field] <= 0:
            reasons.append(f"{field}={typed[field]} must be positive")
    if expected_scenario not in VIEWER_SCENARIO_ACTION_COUNTS:
        reasons.append(f"unsupported expected scenario={expected_scenario!r}")
        expected_actions = (0, 0, 0, 0, 0)
    else:
        expected_actions = VIEWER_SCENARIO_ACTION_COUNTS[expected_scenario]
    if typed["scenario"] != expected_scenario:
        reasons.append(
            f"scenario={typed['scenario']!r} expected={expected_scenario!r}"
        )
    action_suffixes = (
        "resize",
        "drawable",
        "source",
        "recovery_fault",
        "memory",
    )
    emitted = tuple(
        typed[f"actions_emitted_{suffix}"] for suffix in action_suffixes
    )
    applied = tuple(
        typed[f"actions_applied_{suffix}"] for suffix in action_suffixes
    )
    if emitted != expected_actions or applied != expected_actions:
        reasons.append(
            f"actions=emitted:{emitted!r} applied:{applied!r} "
            f"expected:{expected_actions!r}"
        )
    expected_faults = expected_actions[3]
    if (
        typed["faults_fired"] != expected_faults
        or typed["faults_recovered"] != expected_faults
    ):
        reasons.append(
            f"faults=fired:{typed['faults_fired']} "
            f"recovered:{typed['faults_recovered']} expected:{expected_faults}"
        )
    expected_source_lifetimes = 1 + expected_actions[2] // 2
    source_counts = (
        typed["source_publishes"],
        typed["source_clears"],
        typed["source_retires"],
        typed["native_creates"],
        typed["native_retires"],
    )
    if source_counts != (expected_source_lifetimes,) * 5:
        reasons.append(
            f"source_counts={source_counts!r} "
            f"expected={(expected_source_lifetimes,) * 5!r}"
        )
    if typed["native_inflight"] != 0:
        reasons.append(f"native_inflight={typed['native_inflight']}")

    memory_fields = (
        "actions_emitted_memory",
        "actions_applied_memory",
        "memory_warnings",
        "memory_criticals",
        "memory_redraws",
        "memory_trimmed_surfaces",
        "memory_trimmed_surface_bytes",
        "memory_trimmed_derived",
        "memory_trimmed_derived_bytes",
    )
    expected_memory_actions = expected_actions[4]
    # The generic integer non-negative check above runs first.  Only apply
    # scenario-specific memory semantics when no memory value is negative, so
    # a malformed negative telemetry value cannot be mistaken for a valid
    # pressure report.
    if not negative:
        if expected_memory_actions == 0:
            nonzero_memory = {
                field: typed[field]
                for field in memory_fields
                if typed[field] != 0
            }
            if nonzero_memory:
                reasons.append(f"memory-zero-scenario={nonzero_memory!r}")
        else:
            expected_memory_counts = {
                "actions_emitted_memory": 2,
                "actions_applied_memory": 2,
                "memory_warnings": 1,
                "memory_criticals": 1,
                "memory_redraws": 1,
            }
            mismatched_memory = {
                field: typed[field]
                for field, expected in expected_memory_counts.items()
                if typed[field] != expected
            }
            if mismatched_memory:
                reasons.append(f"memory-counts={mismatched_memory!r}")
            for field in (
                "memory_trimmed_surfaces",
                "memory_trimmed_surface_bytes",
                "memory_trimmed_derived",
                "memory_trimmed_derived_bytes",
            ):
                if typed[field] <= 0:
                    reasons.append(f"{field}={typed[field]} must be positive")
    expected_recreations = (
        1 if expected_scenario in {"recovery-fault", "soak"} else 0
    )
    if typed["runtime_recreations"] != expected_recreations:
        reasons.append(f"runtime_recreations={typed['runtime_recreations']}")
    if reasons:
        return False, "invariants=" + "; ".join(reasons)
    return True, f"qualification=pass frames={typed['frames']} fields={len(values)}"


def _check_viewer_qualification_output(
    report: Report,
    name: str,
    result: CommandResult,
    requested_frames: int,
    required_fields: Sequence[str],
    expected_scenario: str = "steady",
) -> bool:
    passed, detail = _parse_viewer_qualification_output(
        result, requested_frames, required_fields, expected_scenario
    )
    passed = passed and not result.timed_out and result.returncode == 0
    if result.returncode != 0 and not result.timed_out:
        detail = f"exit={result.returncode}; {detail}"
    report.add(name, passed, detail)
    return passed


def _contract_string_list(
    contract: dict, key: str, report: Report
) -> list[str]:
    value = contract.get(key)
    valid = (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item) for item in value)
        and len(set(value)) == len(value)
    )
    report.add(
        f"contract.{key}",
        valid,
        f"count={len(value) if isinstance(value, list) else 0}",
    )
    return list(value) if valid else []


def _bundle_leaf_files(root: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    forbidden_nodes: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return files, [str(root)]
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            forbidden_nodes.append(relative.as_posix())
        elif path.is_file():
            files.add(relative.as_posix())
        elif not path.is_dir():
            forbidden_nodes.append(relative.as_posix())
    return files, sorted(forbidden_nodes)


def _validate_exact_bundle_content(
    report: Report,
    name: str,
    root: Path,
    expected: Sequence[str],
) -> None:
    actual, forbidden_nodes = _bundle_leaf_files(root)
    expected_set = set(expected)
    missing = sorted(expected_set - actual)
    unexpected = sorted(actual - expected_set)
    report.add(
        name,
        not missing and not unexpected and not forbidden_nodes,
        (
            f"missing={missing!r} unexpected={unexpected!r} "
            f"forbidden-nodes={forbidden_nodes!r}"
        ),
    )


def _validate_manager_structure(
    config: QualificationConfig, report: Report, runner: CommandRunner
) -> tuple[Path, Path]:
    app = config.manager_app
    report.add(
        "manager.app",
        app.is_dir() and not app.is_symlink(),
        str(app) if app.exists() else f"missing: {app}",
    )
    contents = app / "Contents"
    info = _load_plist(contents / "Info.plist", report, "manager.info-plist")
    if info is not None:
        report.add(
            "manager.bundle-identity",
            info.get("CFBundleIdentifier") == MANAGER_IDENTIFIER
            and info.get("CFBundleExecutable") == MANAGER_EXECUTABLE
            and info.get("CFBundlePackageType") == "APPL",
            (
                f"identifier={info.get('CFBundleIdentifier')!r} "
                f"executable={info.get('CFBundleExecutable')!r}"
            ),
        )
        minimum = str(info.get("LSMinimumSystemVersion", ""))
        try:
            minimum_major = int(minimum.split(".", 1)[0])
        except ValueError:
            minimum_major = 0
        report.add(
            "manager.background-macos13",
            info.get("LSBackgroundOnly") is True and minimum_major >= 13,
            f"background={info.get('LSBackgroundOnly')!r} minimum={minimum!r}",
        )

    manager_executable = contents / "MacOS" / MANAGER_EXECUTABLE
    broker = contents / "MacOS" / BROKER_EXECUTABLE
    _check_file(
        report, "manager.executable", manager_executable, executable=True
    )
    _check_file(report, "manager.broker", broker, executable=True)

    agent = _load_plist(
        contents / "Library" / "LaunchAgents" / AGENT_PLIST,
        report,
        "manager.agent-plist",
    )
    if agent is not None:
        forbidden = sorted(FORBIDDEN_AGENT_KEYS.intersection(agent))
        expected_keys = {
            "Label",
            "ProgramArguments",
            "MachServices",
            "ProcessType",
            "ThrottleInterval",
        }
        exact = (
            set(agent) == expected_keys
            and agent.get("Label") == BROKER_IDENTIFIER
            and agent.get("ProgramArguments") == [BROKER_TEMPLATE_PROGRAM]
            and agent.get("MachServices") == BROKER_MACH_SERVICES
            and agent.get("ProcessType") == "Background"
            and agent.get("ThrottleInterval") == 5
            and not forbidden
        )
        report.add(
            "manager.agent-contract",
            exact,
            (
                f"label={agent.get('Label')!r} "
                f"programArguments={agent.get('ProgramArguments')!r} "
                f"machServices={agent.get('MachServices')!r} "
                f"forbidden={forbidden!r}"
            ),
        )

    if not config.skip_manager_execution:
        if sys.platform != "darwin":
            report.add(
                "manager.self-validation",
                False,
                "manager execution requires macOS",
            )
        elif manager_executable.exists():
            _run_check(
                report,
                runner,
                "manager.self-validation",
                [str(manager_executable), "validate"],
                required_text="status=valid",
            )

    action = config.service_action
    if action != "none":
        if action != "status":
            report.add(
                "manager.service-action",
                False,
                f"mutating action forbidden: {action}",
            )
        elif sys.platform != "darwin":
            report.add(
                "manager.service-action",
                False,
                f"{action} requires macOS",
            )
        else:
            accepted = (0, 3)
            _run_check(
                report,
                runner,
                "manager.service-action",
                [str(manager_executable), action],
                accepted_codes=accepted,
                required_text="status=",
            )
    return manager_executable, broker


def _validate_ofx_structure(
    config: QualificationConfig,
    report: Report,
    runner: CommandRunner,
    contract: dict,
) -> tuple[Path, Path, Path, Path]:
    bundle = config.ofx_bundle
    report.add(
        "ofx.bundle",
        bundle.is_dir() and not bundle.is_symlink(),
        str(bundle) if bundle.exists() else f"missing: {bundle}",
    )
    info = _load_plist(bundle / "Contents" / "Info.plist", report, "ofx.info-plist")
    if info is not None:
        report.add(
            "ofx.bundle-identity",
            info.get("CFBundleIdentifier") == PLUGIN_IDENTIFIER
            and info.get("CFBundleExecutable") == PLUGIN_EXECUTABLE
            and info.get("CFBundlePackageType") == "BNDL",
            (
                f"identifier={info.get('CFBundleIdentifier')!r} "
                f"executable={info.get('CFBundleExecutable')!r}"
            ),
        )

    macos = bundle / "Contents" / "MacOS"
    plugin = macos / PLUGIN_EXECUTABLE
    viewer = macos / VIEWER_EXECUTABLE
    relay = macos / PRODUCER_RELAY_EXECUTABLE
    resolve_bridge = macos / RESOLVE_BRIDGE_EXECUTABLE
    _check_file(report, "ofx.plugin", plugin, executable=True)
    _check_file(report, "ofx.viewer", viewer, executable=True)
    _check_file(report, "ofx.producer-relay", relay, executable=True)
    _check_file(
        report, "ofx.resolve-bridge", resolve_bridge, executable=True
    )
    metallib = macos / VIEWER_METALLIB
    metallib_valid = _check_file(
        report, "ofx.viewer-metallib", metallib
    )
    probe_markers = _contract_string_list(
        contract, "required_metal_probe_markers", report
    )
    if sys.platform == "darwin" and not config.skip_manager_execution:
        probe = config.metal_probe
        if probe is None:
            report.add(
                "ofx.viewer-metallib-probe",
                False,
                "missing --metal-probe",
            )
        elif _check_file(
            report,
            "ofx.viewer-metallib-probe-executable",
            probe,
            executable=True,
        ) and metallib_valid:
            functions = contract.get("required_metal_functions", [])
            if not isinstance(functions, list) or not functions:
                report.add(
                    "ofx.viewer-metallib-probe",
                    False,
                    "contract has no required Metal functions",
                )
            else:
                probe_result = _run_check(
                    report,
                    runner,
                    "ofx.viewer-metallib-probe",
                    [str(probe), str(metallib), *map(str, functions)],
                )
                _check_output_markers(
                    report,
                    "ofx.viewer-metallib-probe-output",
                    probe_result,
                    probe_markers,
                )
    return plugin, viewer, relay, resolve_bridge


def _verify_native_binary(
    report: Report,
    runner: CommandRunner,
    name: str,
    path: Path,
    required_architectures: Sequence[str],
    forbidden_strings: Sequence[str],
    forbidden_dependencies: Sequence[str],
) -> None:
    architectures = _runner_run(runner, ["lipo", "-archs", str(path)])
    present_architectures = set(
        re.findall(r"[A-Za-z0-9_]+", architectures.stdout)
    )
    missing_architectures = sorted(
        set(required_architectures) - present_architectures
    )
    report.add(
        f"{name}.architectures",
        architectures.returncode == 0 and not missing_architectures,
        (
            f"required={list(required_architectures)!r} "
            f"reported={architectures.stdout!r} "
            f"missing={missing_architectures!r}"
        ),
    )
    dependencies = _runner_run(runner, ["otool", "-L", str(path)])
    present_dependencies = sorted(
        marker
        for marker in forbidden_dependencies
        if marker in dependencies.combined
    )
    report.add(
        f"{name}.no-forbidden-dependencies",
        dependencies.returncode == 0 and not present_dependencies,
        f"forbidden={present_dependencies!r}",
    )
    symbols = _runner_run(runner, ["nm", "-u", str(path)])
    present = sorted(
        symbol
        for symbol in FORBIDDEN_GLOBAL_IOSURFACE_SYMBOLS
        if symbol in symbols.combined
    )
    report.add(
        f"{name}.no-global-iosurface-symbols",
        symbols.returncode == 0 and not present,
        f"forbidden={present!r}",
    )
    printable = _runner_run(runner, ["strings", "-a", str(path)])
    legacy_markers = sorted(
        marker for marker in forbidden_strings
        if marker in printable.combined
    )
    report.add(
        f"{name}.no-forbidden-cutover-strings",
        printable.returncode == 0 and not legacy_markers,
        f"forbidden={legacy_markers!r}",
    )
    signature = _runner_run(
        runner,
        ["codesign", "--display", "--verbose=4", str(path)]
    )
    signature_lines = [
        line.strip()
        for line in signature.combined.splitlines()
        if line.strip().startswith("Signature=")
    ]
    team_lines = [
        line.strip()
        for line in signature.combined.splitlines()
        if line.strip().startswith("TeamIdentifier=")
    ]
    authority_present = "Authority=" in signature.combined
    identity_free_ad_hoc = (
        signature.returncode == 0
        and signature_lines == ["Signature=adhoc"]
        and team_lines == ["TeamIdentifier=not set"]
        and not authority_present
    )
    unsigned = (
        signature.returncode != 0
        and "code object is not signed at all" in signature.combined.lower()
        and not signature_lines
        and not team_lines
        and not authority_present
    )
    report.add(
        f"{name}.unsigned-or-identity-free-ad-hoc",
        unsigned or identity_free_ad_hoc,
        (
            f"exit={signature.returncode} signature={signature_lines!r} "
            f"team={team_lines!r} "
            f"authority={'present' if authority_present else 'none'} "
            f"state={'unsigned' if unsigned else 'identity-free-ad-hoc' if identity_free_ad_hoc else 'rejected'}"
        ),
    )
    if unsigned:
        entitlement_passed = True
        entitlement_detail = "not-applicable-unsigned"
        entitlement_exit = signature.returncode
    else:
        entitlements = _runner_run(
            runner,
            ["codesign", "--display", "--entitlements", ":-", str(path)]
        )
        entitlement_stdout = entitlements.stdout.strip()
        entitlement_passed = entitlements.returncode == 0 and not entitlement_stdout
        entitlement_detail = (
            "empty"
            if not entitlement_stdout
            else repr(entitlement_stdout[:160])
        )
        entitlement_exit = entitlements.returncode
    report.add(
        f"{name}.no-entitlement-payload",
        entitlement_passed,
        (
            f"exit={entitlement_exit} "
            f"stdout={entitlement_detail}"
        ),
    )


def _normalize_payload_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def _validate_cutover(
    config: QualificationConfig,
    report: Report,
    runner: CommandRunner,
    manager_executable: Path,
    broker: Path,
    plugin: Path,
    viewer: Path,
    relay: Path,
    resolve_bridge: Path,
    contract: dict,
) -> None:
    required_architectures = _contract_string_list(
        contract, "required_architectures", report
    )
    forbidden_strings = _contract_string_list(
        contract, "forbidden_cutover_strings", report
    )
    forbidden_dependencies = _contract_string_list(
        contract, "forbidden_binary_dependencies", report
    )
    viewer_markers = _contract_string_list(
        contract, "required_viewer_qualification_markers", report
    )
    viewer_fields = _contract_string_list(
        contract, "required_viewer_qualification_fields", report
    )
    required_current_fields = set(VIEWER_QUALIFICATION_FIELD_TYPES)
    viewer_fields_complete = bool(viewer_fields) and required_current_fields.issubset(
        viewer_fields
    )
    report.add(
        "contract.required_viewer_qualification_fields-schema",
        viewer_fields_complete,
        (
            f"required={sorted(required_current_fields)!r} "
            f"declared={viewer_fields!r}"
        ),
    )
    manager_files = _contract_string_list(
        contract, "release_manager_files", report
    )
    ofx_files = _contract_string_list(
        contract, "release_ofx_files", report
    )
    _validate_exact_bundle_content(
        report,
        "cutover.manager-content",
        config.manager_app,
        manager_files,
    )
    _validate_exact_bundle_content(
        report,
        "cutover.ofx-exact-content",
        config.ofx_bundle,
        ofx_files,
    )
    report.add(
        "cutover.platform",
        sys.platform == "darwin",
        sys.platform,
    )
    if sys.platform != "darwin":
        return

    for name, path in (
        ("ofx", plugin),
        ("viewer", viewer),
        ("producer-relay", relay),
        ("resolve-bridge", resolve_bridge),
        ("broker", broker),
        ("manager-executable", manager_executable),
    ):
        _verify_native_binary(
            report,
            runner,
            name,
            path,
            required_architectures,
            forbidden_strings,
            forbidden_dependencies,
        )

    bundle = config.ofx_bundle
    forbidden_artifacts = [
        bundle / "Contents" / "MacOS" / BROKER_EXECUTABLE,
        bundle / LEGACY_BROKER_TEMPLATE,
        bundle / SHIPPED_ENTITLEMENT_FILE,
    ]
    present = [str(path) for path in forbidden_artifacts if path.exists()]
    report.add(
        "cutover.ofx-content",
        not present,
        f"forbidden-present={present!r}",
    )

    if config.skip_manager_execution:
        report.add(
            "cutover.viewer-qualification",
            True,
            "execution explicitly skipped",
        )
    elif config.viewer_canary is None:
        report.add(
            "cutover.viewer-qualification",
            False,
            "missing --viewer-canary",
        )
    elif not (1 <= config.viewer_frames <= 10000):
        report.add(
            "cutover.viewer-qualification",
            False,
            f"invalid frame count {config.viewer_frames}",
        )
    elif not _valid_viewer_scenarios(config.viewer_scenarios):
        report.add(
            "cutover.viewer-qualification",
            False,
            f"invalid scenarios {config.viewer_scenarios!r}",
        )
    elif _check_file(
        report,
        "cutover.viewer-canary-executable",
        config.viewer_canary,
        executable=True,
    ):
        for scenario in config.viewer_scenarios:
            epoch_name = f"cutover.viewer-qualification.{scenario}"
            viewer_result = _run_check(
                report,
                runner,
                epoch_name,
                [
                    str(config.viewer_canary),
                    "--frames",
                    str(config.viewer_frames),
                    "--qualification-scenario",
                    scenario,
                ],
                timeout_seconds=config.viewer_timeout_seconds,
            )
            _check_output_markers(
                report,
                f"{epoch_name}.markers",
                viewer_result,
                viewer_markers,
            )
            _check_viewer_qualification_output(
                report,
                f"{epoch_name}.output",
                viewer_result,
                config.viewer_frames,
                viewer_fields if viewer_fields_complete else [],
                scenario,
            )

    package = config.installer_pkg
    if package is None:
        report.add(
            "cutover.installer-package",
            True,
            "not supplied; per-user unsigned deployment may be used",
        )
        return
    _check_file(report, "cutover.installer-package", package)
    payload = _runner_run(runner, ["pkgutil", "--payload-files", str(package)])
    payload_paths = {
        _normalize_payload_path(line)
        for line in payload.stdout.splitlines()
        if line.strip()
    }
    expected_ofx = _normalize_payload_path(config.expected_ofx_install_path)
    ofx_present = any(
        item == expected_ofx or item.startswith(expected_ofx + "/")
        for item in payload_paths
    )
    report.add(
        "cutover.installer-payload",
        payload.returncode == 0 and ofx_present,
        f"ofx={ofx_present} entries={len(payload_paths)}",
    )


def qualify(
    config: QualificationConfig, runner: Optional[CommandRunner] = None
) -> Report:
    report = Report(mode=config.mode)
    if not _valid_viewer_timeout(config.viewer_timeout_seconds):
        report.add(
            "config.viewer-timeout-seconds",
            False,
            (
                f"must be finite and between 1 and "
                f"{MAX_VIEWER_TIMEOUT_SECONDS:g}; "
                f"got={config.viewer_timeout_seconds!r}"
            ),
        )
        return report
    if config.mode == "cutover" and not _valid_viewer_scenarios(
        config.viewer_scenarios
    ):
        report.add(
            "config.viewer-scenarios",
            False,
            f"must be unique supported scenarios; got={config.viewer_scenarios!r}",
        )
        return report
    command_runner = runner or CommandRunner()
    contract = _load_contract(config.contract_path, report)
    manager_executable, broker = _validate_manager_structure(
        config, report, command_runner
    )
    plugin, viewer, relay, resolve_bridge = _validate_ofx_structure(
        config, report, command_runner, contract
    )
    if config.mode == "cutover":
        _validate_cutover(
            config,
            report,
            command_runner,
            manager_executable,
            broker,
            plugin,
            viewer,
            relay,
            resolve_bridge,
            contract,
        )
    return report


def _viewer_timeout_argument(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("viewer timeout must be a number") from error
    if not _valid_viewer_timeout(timeout):
        raise argparse.ArgumentTypeError(
            "viewer timeout must be finite and between 1 and 3600 seconds"
        )
    return timeout


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-app", required=True, type=Path)
    parser.add_argument("--ofx-bundle", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("structure", "cutover"), default="structure"
    )
    parser.add_argument("--installer-pkg", type=Path)
    parser.add_argument("--metal-probe", type=Path)
    parser.add_argument("--viewer-canary", type=Path)
    parser.add_argument("--viewer-frames", type=int, default=300)
    parser.add_argument(
        "--viewer-scenario",
        action="append",
        choices=VIEWER_QUALIFICATION_SCENARIOS,
        help="repeat to run a subset; defaults to the full deterministic matrix",
    )
    parser.add_argument(
        "--viewer-timeout-seconds",
        type=_viewer_timeout_argument,
        default=DEFAULT_VIEWER_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=QualificationConfig.contract_path,
    )
    parser.add_argument(
        "--service-action",
        choices=("none", "status"),
        default="status",
    )
    parser.add_argument("--skip-manager-execution", action="store_true")
    parser.add_argument(
        "--expected-ofx-install-path",
        default=QualificationConfig.expected_ofx_install_path,
    )
    parser.add_argument("--report-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = QualificationConfig(
        manager_app=args.manager_app,
        ofx_bundle=args.ofx_bundle,
        mode=args.mode,
        installer_pkg=args.installer_pkg,
        metal_probe=args.metal_probe,
        viewer_canary=args.viewer_canary,
        viewer_frames=args.viewer_frames,
        viewer_scenarios=tuple(
            args.viewer_scenario or VIEWER_QUALIFICATION_SCENARIOS
        ),
        viewer_timeout_seconds=args.viewer_timeout_seconds,
        contract_path=args.contract,
        service_action=args.service_action,
        skip_manager_execution=args.skip_manager_execution,
        expected_ofx_install_path=args.expected_ofx_install_path,
    )
    report = qualify(config)
    for finding in report.findings:
        marker = "PASS" if finding.passed else "FAIL"
        print(f"[{marker}] {finding.name}: {finding.detail}")
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
