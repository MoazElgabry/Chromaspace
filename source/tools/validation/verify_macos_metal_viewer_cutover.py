#!/usr/bin/env python3
"""Fail-closed source policy for the unshipped Metal viewer cutover canary.

This is intentionally a portable source check. It does not inspect Mach-O
symbols, framework load commands, signing, or runtime behavior; those remain
macOS qualification work.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Iterable


NATIVE_ONLY_MACRO = "CHROMASPACE_METAL_NATIVE_ONLY"
TARGET_MUTATION_COMMANDS = (
    "add_executable",
    "target_link_libraries",
    "target_link_options",
    "target_compile_definitions",
    "target_compile_options",
    "target_include_directories",
    "add_dependencies",
    "target_sources",
    "set_target_properties",
    "set_property",
)


def _read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read {relative}: {exc}") from exc


def _missing(text: str, tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token not in text]


def _forbidden(text: str, tokens: Iterable[str]) -> list[str]:
    return [token for token in tokens if token in text]


def _guarded_token_violations(text: str, tokens: Iterable[str]) -> list[str]:
    """Return legacy token occurrences outside a compatibility preprocessor guard."""

    token_list = tuple(tokens)
    stack: list[bool] = []
    violations: list[str] = []
    directive = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")

    for line_number, line in enumerate(text.splitlines(), start=1):
        match = directive.match(line)
        if match:
            kind, expression = match.groups()
            if kind == "if":
                stack.append(
                    NATIVE_ONLY_MACRO in expression
                    and ("!defined" in expression or "! defined" in expression)
                )
            elif kind in {"ifdef", "ifndef"}:
                stack.append(
                    kind == "ifndef" and NATIVE_ONLY_MACRO in expression
                )
            elif kind == "elif":
                if stack:
                    stack[-1] = False
            elif kind == "else":
                if stack:
                    stack[-1] = not stack[-1]
            elif kind == "endif":
                if stack:
                    stack.pop()
                else:
                    violations.append(f"line {line_number}: unmatched #endif")
            continue

        if any(stack):
            continue
        for token in token_list:
            if token in line:
                violations.append(f"line {line_number}: {token}")

    if stack:
        violations.append("unterminated preprocessor guard")
    return violations


def _initialize_metal_context_return_findings(text: str) -> list[str]:
    """Reject bare returns that escape the non-void Metal initializer."""

    signature = (
        "bool initializeMetalContext(MetalContext* context, std::string* error)"
    )
    start = text.find(signature)
    if start < 0:
        return ["initializeMetalContext definition not found"]
    end = text.find("\n#if !defined(CHROMASPACE_METAL_NATIVE_ONLY)", start)
    if end < 0:
        return ["initializeMetalContext end marker not found"]

    region = text[start:end]
    bare_returns = list(
        re.finditer(r"(?m)^[ \t]*return;[ \t]*(?://.*)?$", region)
    )
    if not bare_returns:
        return []

    wrapper = re.search(
        r"(?s)auto\s+initializeResources\s*=\s*\[&\]\(\)\s*\{"
        r"\s*@autoreleasepool\s*\{.*?\n[ \t]*\}\s*\n[ \t]*\};"
        r"\s*try\s*\{\s*initializeResources\(\);",
        region,
    )
    findings: list[str] = []
    for match in bare_returns:
        if wrapper is not None and wrapper.start() <= match.start() < wrapper.end():
            continue
        line_number = text.count("\n", 0, start + match.start()) + 1
        findings.append(
            f"line {line_number}: bare return escapes non-void initializeMetalContext"
        )
    return findings


def _target_block(cmake: str, target: str) -> str:
    marker = f"add_executable({target}"
    start = cmake.find(marker)
    if start < 0:
        return ""
    end = cmake.find("include(CTest)", start)
    return cmake[start:] if end < 0 else cmake[start:end]


def _parse_cmake_commands(cmake: str, command: str) -> tuple[list[str], list[str]]:
    """Extract balanced command bodies and report malformed occurrences."""

    pattern = re.compile(rf"\b{re.escape(command)}\s*\(", re.IGNORECASE)
    bodies: list[str] = []
    errors: list[str] = []
    for match in pattern.finditer(cmake):
        line_start = cmake.rfind("\n", 0, match.start()) + 1
        if cmake[line_start:match.start()].lstrip().startswith("#"):
            continue
        open_index = cmake.find("(", match.start(), match.end())
        depth = 0
        quote = False
        escaped = False
        line_comment = False
        close_index = -1
        for index in range(open_index, len(cmake)):
            char = cmake[index]
            if line_comment:
                if char == "\n":
                    line_comment = False
                continue
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quote = False
                continue
            if char == '#':
                line_comment = True
            elif char == '"':
                quote = True
            elif char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    close_index = index
                    break
        if close_index < 0:
            line = cmake.count("\n", 0, match.start()) + 1
            errors.append(f"{command} at line {line} is not balanced")
        else:
            bodies.append(cmake[open_index + 1 : close_index])
    return bodies, errors


def _cmake_list_values(cmake: str, variable: str) -> tuple[list[str], list[str]]:
    """Return one literal set(VAR ...) list, failing closed on ambiguity."""

    values: list[list[str]] = []
    findings: list[str] = []
    bodies, errors = _parse_cmake_commands(cmake, "set")
    findings.extend(f"set: {error}" for error in errors)
    for body in bodies:
        arguments, error = _cmake_arguments(body)
        if error:
            findings.append(f"set: {error}")
        elif arguments and arguments[0] == variable:
            values.append(arguments[1:])
    if len(values) != 1:
        findings.append(
            f"expected exactly one set({variable} ...) command, found {len(values)}"
        )
        return [], findings
    return values[0], findings


def _cmake_is_non_apple_context(cmake: str, position: int) -> bool:
    """Return whether a position is guarded by the active non-Apple branch."""

    stack: list[tuple[str, bool]] = []
    directive = re.compile(
        r"^\s*(if|elseif|else|endif)\s*\((.*?)\)\s*(?:#.*)?$",
        re.IGNORECASE,
    )
    for line in cmake[:position].splitlines():
        match = directive.match(line)
        if match is None:
            continue
        kind = match.group(1).lower()
        expression = re.sub(r"\s+", "", match.group(2)).upper()
        if kind == "if":
            stack.append((expression, False))
        elif kind == "elseif" and stack:
            stack[-1] = (expression, False)
        elif kind == "else" and stack:
            condition, is_else = stack[-1]
            stack[-1] = (condition, not is_else)
        elif kind == "endif" and stack:
            stack.pop()
    return any(
        (condition == "NOTAPPLE" and not is_else)
        or (condition == "APPLE" and is_else)
        for condition, is_else in stack
    )


def _cmake_arguments(body: str) -> tuple[list[str], str | None]:
    try:
        return shlex.split(body, comments=True, posix=True), None
    except ValueError as exc:
        return [], f"cannot parse CMake command arguments: {exc}"


def _cmake_source_closure_findings(
    cmake: str,
    target: str,
    expected_sources: Iterable[str],
    allowed_source_variables: Iterable[str] = (),
    source_list_variable: str | None = None,
    source_list_variables: Iterable[str] | None = None,
) -> list[str]:
    findings: list[str] = []
    allowed_variables = set(allowed_source_variables)
    add_bodies, add_errors = _parse_cmake_commands(cmake, "add_executable")
    findings.extend(f"add_executable: {error}" for error in add_errors)
    target_commands: list[list[str]] = []
    for body in add_bodies:
        arguments, error = _cmake_arguments(body)
        if error:
            findings.append(f"add_executable: {error}")
            continue
        if arguments and arguments[0] == target:
            target_commands.append(arguments)
    if len(target_commands) != 1:
        findings.append(
            f"expected exactly one add_executable({target} ...) command, "
            f"found {len(target_commands)}"
        )
    else:
        declared_sources = target_commands[0][1:]
        requested_lists = (
            list(source_list_variables)
            if source_list_variables is not None
            else None
        )
        if requested_lists is not None:
            expected_references = [f"${{{value}}}" for value in requested_lists]
            if declared_sources != expected_references:
                findings.append(
                    f"{target} must use only canonical source lists "
                    f"{expected_references!r}: declared={declared_sources!r}"
                )
            declared_sources = []
            for variable in requested_lists:
                values, list_findings = _cmake_list_values(cmake, variable)
                findings.extend(list_findings)
                declared_sources.extend(values)
        elif source_list_variable is not None:
            expected_reference = f"${{{source_list_variable}}}"
            if declared_sources != [expected_reference]:
                findings.append(
                    f"{target} must use only canonical source list "
                    f"{expected_reference}: declared={declared_sources!r}"
                )
            declared_sources, list_findings = _cmake_list_values(
                cmake, source_list_variable
            )
            findings.extend(list_findings)
        expected = list(expected_sources)
        unresolved_declared = [
            source
            for source in declared_sources
            if "${" in source and source not in allowed_variables
        ]
        unresolved_expected = [
            source
            for source in expected
            if "${" in source and source not in allowed_variables
        ]
        findings.extend(
            f"{target} unresolved source variable: {source}"
            for source in unresolved_declared + unresolved_expected
        )
        if sorted(declared_sources) != sorted(expected):
            findings.append(
                f"{target} source closure mismatch: "
                f"declared={declared_sources!r} expected={expected!r}"
            )

    target_source_bodies, target_source_errors = _parse_cmake_commands(
        cmake, "target_sources"
    )
    findings.extend(f"target_sources: {error}" for error in target_source_errors)
    for body in target_source_bodies:
        arguments, error = _cmake_arguments(body)
        if error:
            findings.append(f"target_sources: {error}")
        elif arguments and arguments[0] == target:
            findings.append(
                f"separate target_sources({target} ...) command is forbidden"
            )
    return findings


def _cmake_list_closure_findings(
    cmake: str,
    variable: str,
    expected_sources: Iterable[str],
    allowed_source_variables: Iterable[str] = (),
) -> list[str]:
    """Verify one canonical source list independently of target composition."""

    declared, findings = _cmake_list_values(cmake, variable)
    expected = list(expected_sources)
    allowed = set(allowed_source_variables)
    unresolved = [
        source
        for source in declared + expected
        if "${" in source and source not in allowed
    ]
    findings.extend(
        f"{variable} unresolved source variable: {source}"
        for source in unresolved
    )
    if sorted(declared) != sorted(expected):
        findings.append(
            f"{variable} source closure mismatch: "
            f"declared={declared!r} expected={expected!r}"
        )
    return findings


def _apple_product_policy_findings(
    cmake: str,
    product_target: str,
    product_source_variable: str,
    canonical_source_variable: str,
    required_tokens: Iterable[str],
    forbidden_tokens: Iterable[str],
) -> list[str]:
    """Verify that the packaged Apple product selects the native source truth."""

    findings: list[str] = []
    add_bodies, add_errors = _parse_cmake_commands(cmake, "add_executable")
    findings.extend(f"add_executable: {error}" for error in add_errors)
    product_commands: list[list[str]] = []
    for body in add_bodies:
        arguments, error = _cmake_arguments(body)
        if error:
            findings.append(f"add_executable: {error}")
        elif arguments and arguments[0] == product_target:
            product_commands.append(arguments)
    expected_product_sources = [
        product_target,
        "WIN32",
        f"${{{product_source_variable}}}",
    ]
    if product_commands != [expected_product_sources]:
        findings.append(
            f"{product_target} source selection mismatch: "
            f"declared={product_commands!r} expected={[expected_product_sources]!r}"
        )

    apple_assignment = re.compile(
        rf"if\s*\(\s*APPLE\s*\).*?"
        rf"set\s*\(\s*{re.escape(product_source_variable)}\s+"
        rf"\$\{{{re.escape(canonical_source_variable)}\}}\s*\)",
        re.IGNORECASE | re.DOTALL,
    )
    if apple_assignment.search(cmake) is None:
        findings.append(
            f"APPLE does not assign {product_source_variable} exclusively from "
            f"${{{canonical_source_variable}}}"
        )

    product_mutations: list[str] = []
    for command in TARGET_MUTATION_COMMANDS:
        bodies, errors = _parse_cmake_commands(cmake, command)
        findings.extend(f"{command}: {error}" for error in errors)
        search_position = 0
        for body in bodies:
            body_position = cmake.find(body, search_position)
            if body_position >= 0:
                search_position = body_position + len(body)
            arguments, error = _cmake_arguments(body)
            if error:
                findings.append(f"{command}: {error}")
                continue
            affects_target = (
                len(arguments) >= 2
                and arguments[0].upper() == "TARGET"
                and arguments[1] == product_target
                if command == "set_property"
                else bool(arguments) and arguments[0] == product_target
            )
            if affects_target:
                product_mutations.append(body)
                for token in _forbidden(body, forbidden_tokens):
                    if body_position < 0 or not _cmake_is_non_apple_context(
                        cmake, body_position
                    ):
                        findings.append(
                            f"Apple product mutation contains forbidden token: {token}"
                        )
    product_policy_text = "\n".join(product_mutations)
    findings.extend(
        f"product target missing required token: {token}"
        for token in _missing(product_policy_text, required_tokens)
    )

    non_apple_dependency_block = re.compile(
        r"if\s*\(\s*NOT\s+APPLE\s*\).*?find_package\s*\(\s*OpenGL\s+REQUIRED\s*\)"
        r".*?FetchContent_MakeAvailable\s*\(\s*glfw\s*\).*?endif\s*\(\)",
        re.IGNORECASE | re.DOTALL,
    )
    if non_apple_dependency_block.search(cmake) is None:
        findings.append("OpenGL/GLFW discovery is not enclosed by NOT APPLE")
    return findings


def _target_policy_findings(
    block: str, required_tokens: Iterable[str], forbidden_tokens: Iterable[str]
) -> list[str]:
    findings: list[str] = []
    findings.extend(
        f"canary block missing required token: {token}"
        for token in _missing(block, required_tokens)
    )
    findings.extend(
        f"canary block contains forbidden token: {token}"
        for token in _forbidden(block, forbidden_tokens)
    )
    return findings


def _target_mutation_policy_findings(
    cmake: str, target: str, forbidden_tokens: Iterable[str]
) -> list[str]:
    """Check every CMake mutation command that affects the canary target."""

    findings: list[str] = []
    for command in TARGET_MUTATION_COMMANDS:
        bodies, errors = _parse_cmake_commands(cmake, command)
        findings.extend(f"{command}: {error}" for error in errors)
        for body in bodies:
            arguments, error = _cmake_arguments(body)
            if error:
                findings.append(f"{command}: {error}")
                continue
            affects_target = False
            if command == "set_property":
                affects_target = (
                    len(arguments) >= 2
                    and arguments[0].upper() == "TARGET"
                    and arguments[1] == target
                )
            else:
                affects_target = bool(arguments) and arguments[0] == target
            if not affects_target:
                continue
            for token in _forbidden(body, forbidden_tokens):
                findings.append(
                    f"{command}({target}) contains forbidden token: {token}"
                )
    return findings


def verify(root: Path) -> list[str]:
    manifest_path = root / "tools" / "validation" / "macos_metal_viewer_cutover_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest: {exc}"]

    findings: list[str] = []
    try:
        entrypoint = _read(root, manifest["entrypoint"])
        production_entrypoint = _read(
            root, manifest.get("production_entrypoint", "tools/cube_viewer_stub/main.cpp")
        )
        metal_source = _read(root, manifest["metal_source"])
        metal_header = _read(root, manifest["metal_header"])
        cmake = _read(root, manifest["cmake"])
    except (KeyError, RuntimeError) as exc:
        return [str(exc)]

    for token in _missing(entrypoint, manifest.get("entrypoint_required", [])):
        findings.append(f"entrypoint missing required token: {token}")
    for token in _forbidden(entrypoint, manifest.get("entrypoint_forbidden", [])):
        findings.append(f"entrypoint contains forbidden token: {token}")
    for token in _missing(production_entrypoint, manifest.get("production_required", [])):
        findings.append(f"production entrypoint missing required token: {token}")
    for token in _forbidden(
        production_entrypoint, manifest.get("production_forbidden", [])
    ):
        findings.append(f"production entrypoint contains forbidden token: {token}")
    for token in _missing(metal_source, manifest.get("metal_required", [])):
        findings.append(f"Metal source missing required token: {token}")
    for token in _forbidden(metal_source, manifest.get("metal_forbidden", [])):
        findings.append(f"Metal source contains forbidden token: {token}")
    findings.extend(
        f"Metal source initialization return policy: {finding}"
        for finding in _initialize_metal_context_return_findings(metal_source)
    )
    for token in manifest.get("cmake_global_required", []):
        if token not in cmake:
            findings.append(f"CMake global missing required token: {token}")

    executor_forbidden = manifest.get("executor_forbidden", [])
    for relative in manifest.get("executor_sources", []):
        try:
            executor_source = _read(root, relative)
        except RuntimeError as exc:
            findings.append(f"executor source: {exc}")
            continue
        for token in _forbidden(executor_source, executor_forbidden):
            findings.append(f"{relative}: executor contains forbidden token: {token}")

    resident_source_forbidden = manifest.get("resident_source_forbidden", [])
    for relative in manifest.get("resident_source_sources", []):
        try:
            resident_source = _read(root, relative)
        except RuntimeError as exc:
            findings.append(f"resident source: {exc}")
            continue
        for token in _forbidden(resident_source, resident_source_forbidden):
            findings.append(
                f"{relative}: resident source contains forbidden token: {token}"
            )

    plot_renderer_forbidden = manifest.get("plot_renderer_forbidden", [])
    for relative in manifest.get("plot_renderer_sources", []):
        try:
            plot_renderer_source = _read(root, relative)
        except RuntimeError as exc:
            findings.append(f"plot renderer source: {exc}")
            continue
        for token in _forbidden(plot_renderer_source, plot_renderer_forbidden):
            findings.append(
                f"{relative}: plot renderer contains forbidden token: {token}"
            )

    guarded_tokens = manifest.get("guarded_legacy_tokens", [])
    for relative, text in (
        (manifest["metal_source"], metal_source),
        (manifest["metal_header"], metal_header),
    ):
        for violation in _guarded_token_violations(text, guarded_tokens):
            findings.append(f"{relative}: legacy token outside native-only guard: {violation}")

    block = _target_block(cmake, manifest["target"])
    if not block:
        findings.append(f"CMake target block not found: {manifest['target']}")
    else:
        findings.extend(
            f"CMake {finding}"
            for finding in _target_policy_findings(
                block,
                [],
                manifest.get("cmake_forbidden", []),
            )
        )
        findings.extend(
            f"CMake canary missing required token: {token}"
            for token in _missing(cmake, manifest.get("cmake_required", []))
        )
    findings.extend(
        f"CMake source closure: {finding}"
        for finding in _cmake_source_closure_findings(
            cmake,
            manifest["target"],
            manifest["source_closure"],
            manifest.get("allowed_source_variables", []),
            manifest.get("source_list_variable"),
            manifest.get("source_list_variables"),
        )
    )
    for closure in manifest.get("canonical_source_closures", []):
        findings.extend(
            f"CMake canonical source closure: {finding}"
            for finding in _cmake_list_closure_findings(
                cmake,
                closure["variable"],
                closure["sources"],
                manifest.get("allowed_source_variables", []),
            )
        )
    findings.extend(
        f"CMake target mutation: {finding}"
        for finding in _target_mutation_policy_findings(
            cmake, manifest["target"], manifest.get("cmake_forbidden", [])
        )
    )
    findings.extend(
        f"CMake product policy: {finding}"
        for finding in _apple_product_policy_findings(
            cmake,
            manifest["product_target"],
            manifest["product_source_variable"],
            manifest.get(
                "product_canonical_source_variable",
                manifest.get("source_list_variable", ""),
            ),
            manifest.get("product_required", []),
            manifest.get("product_forbidden", []),
        )
    )

    if NATIVE_ONLY_MACRO not in metal_header:
        findings.append("Metal header does not expose the native-only membrane")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Chromaspace/source root (defaults to this script's source root)",
    )
    args = parser.parse_args(argv)
    findings = verify(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"FAIL: {finding}")
        return 1
    print("PASS: macOS Metal viewer cutover source policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
