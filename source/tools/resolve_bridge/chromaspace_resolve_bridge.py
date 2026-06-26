#!/usr/bin/env python3
"""Chromaspace Resolve bridge.

This companion watches viewer desired-state mailboxes and asks Resolve to
refresh the current frame using the same same-timecode scripting pattern used
by PhotoChemist's Film Stock Editor. It intentionally does not mutate OFX
plugin memory, OFX parameters, Resolve UI controls, or the timeline position.
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import re
import sys
import threading
import time
import atexit
from typing import Dict, Optional, Tuple


LIVE_REFRESH_INTERVAL_SECONDS = 0.08
LIVE_PULSE_STALE_SECONDS = 1.5
POLL_INTERVAL_SECONDS = 0.025
LOCK_HEARTBEAT_SECONDS = 2.0
STALE_LOCK_SECONDS = 30.0
RESOLVE_RECONNECT_SECONDS = 1.0
DAEMON_STATUS_SECONDS = 2.0
MAILBOX_ACTIVE_SECONDS = 30.0
CHROMASPACE_TOOL_TOKENS = ("chromaspace", "com.moazelgabry.chromaspace")
DRAW_LABEL_TOKENS = ("draw", "generator", "identity", "instance 1", "instance1", "upstream", "hald")
PLOT_LABEL_TOKENS = ("plot", "viewer", "view", "cube", "3d")
NODE_SELECT_METHODS = ("SetCurrentNode", "SetSelectedNode", "SelectNode", "SetActiveNode")


def sessions_root() -> pathlib.Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return pathlib.Path(base) / "Chromaspace" / "sessions"
        return pathlib.Path(os.environ.get("TEMP", ".")) / "Chromaspace" / "sessions"
    if sys.platform == "darwin":  # type: ignore[name-defined]
        return pathlib.Path.home() / "Library" / "Application Support" / "Chromaspace" / "sessions"
    return pathlib.Path.home() / ".config" / "Chromaspace" / "sessions"


def chromaspace_root() -> pathlib.Path:
    return sessions_root().parent


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            close_handle(handle)
            return True
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class SingleBridgeOwner:
    """Latest-state watcher ownership, hardened like PhotoChemist's single daemon."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        self.owns_lock = False
        self.pid = os.getpid()
        self.last_heartbeat = 0.0

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        for _ in range(2):
            try:
                fd = os.open(str(self.path), flags, 0o600)
                payload = {
                    "type": "chromaspace_resolve_bridge_lock",
                    "pid": self.pid,
                    "updatedAtUnix": time.time(),
                }
                os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
                os.close(fd)
                self.owns_lock = True
                self.write_heartbeat(force=True)
                atexit.register(self.release)
                return True
            except FileExistsError:
                if not self._remove_stale_lock():
                    return False
        return False

    def _remove_stale_lock(self) -> bool:
        try:
            text = self.path.read_text(encoding="utf-8")
            data = json.loads(text) if text.strip() else {}
        except Exception:
            data = {}
        pid = int(data.get("pid") or 0)
        updated = float(data.get("updatedAtUnix") or 0.0)
        now = time.time()
        if pid > 0 and process_alive(pid):
            return False
        if pid <= 0 and updated > 0.0 and now - updated < STALE_LOCK_SECONDS:
            return False
        try:
            self.path.unlink()
            return True
        except Exception:
            return False

    def write_heartbeat(self, force: bool = False) -> None:
        if not self.owns_lock:
            return
        now = time.time()
        if not force and now - self.last_heartbeat < LOCK_HEARTBEAT_SECONDS:
            return
        self.last_heartbeat = now
        payload = {
            "type": "chromaspace_resolve_bridge_lock",
            "pid": self.pid,
            "updatedAtUnix": now,
        }
        try:
            existing = read_json(self.path)
            existing_pid = int(existing.get("pid") or 0)
            if existing_pid not in (0, self.pid):
                self.owns_lock = False
                return
            write_json_atomic(self.path, payload)
        except Exception:
            pass

    def release(self) -> None:
        self.owns_lock = False
        try:
            data = read_json(self.path)
            if int(data.get("pid") or 0) == self.pid:
                self.path.unlink()
        except Exception:
            pass


def read_json(path: pathlib.Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json_atomic(path: pathlib.Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def import_resolve_script():
    candidates = []
    if os.name == "nt":
        program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        candidates.append(
            pathlib.Path(program_data)
            / "Blackmagic Design"
            / "DaVinci Resolve"
            / "Support"
            / "Developer"
            / "Scripting"
            / "Modules"
        )
    elif sys.platform == "darwin":
        candidates.append(
            pathlib.Path("/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules")
        )
    else:
        candidates.append(pathlib.Path("/opt/resolve/Developer/Scripting/Modules"))

    for candidate in candidates:
        if candidate.is_dir():
            text = str(candidate)
            if text not in sys.path:
                sys.path.append(text)

    import DaVinciResolveScript as resolve_script  # type: ignore

    return resolve_script


class ResolveRefreshClient:
    """Cached Resolve scripting connection with bounded reconnect attempts."""

    def __init__(self):
        self.resolve_script = None
        self.resolve = None
        self.next_connect_at = 0.0

    def _connect(self, now: float) -> Tuple[bool, str]:
        if self.resolve is not None:
            return True, "Resolve scripting connection is cached."
        if now < self.next_connect_at:
            return False, "Resolve scripting reconnect is backing off."
        try:
            if self.resolve_script is None:
                self.resolve_script = import_resolve_script()
            self.resolve = self.resolve_script.scriptapp("Resolve")
            if self.resolve is None:
                self.next_connect_at = now + RESOLVE_RECONNECT_SECONDS
                return False, "Resolve scripting module loaded, but no Resolve app object was returned."
            return True, "Resolve scripting connected."
        except Exception as exc:
            self.resolve = None
            self.next_connect_at = now + RESOLVE_RECONNECT_SECONDS
            return False, f"Resolve scripting unavailable: {exc}"

    def request_refresh(self) -> Tuple[bool, str]:
        now = time.monotonic()
        connected, message = self._connect(now)
        if not connected:
            return False, message
        try:
            page = self.resolve.GetCurrentPage()
            manager = self.resolve.GetProjectManager()
            project = manager.GetCurrentProject() if manager else None
            timeline = project.GetCurrentTimeline() if project else None
            if timeline is None:
                return False, f"Resolve scripting is reachable on page {page!r}, but no current timeline is active."

            current_timecode = timeline.GetCurrentTimecode()
            if not current_timecode:
                return False, "Resolve scripting is reachable, but the current timeline did not report a timecode."

            if timeline.SetCurrentTimecode(current_timecode):
                return True, f"Resolve same-timecode refresh request completed on page {page!r}."
            return False, "Resolve scripting is reachable, but the timeline rejected the refresh request."
        except Exception as exc:
            self.resolve = None
            self.next_connect_at = now + RESOLVE_RECONNECT_SECONDS
            return False, f"Resolve refresh failed; scripting connection reset: {exc}"


def parse_fps(value) -> int:
    try:
        fps = float(str(value).strip())
        if fps > 0:
            return max(1, int(round(fps)))
    except Exception:
        pass
    return 24


def parse_timecode_frames(timecode: str, fps: int) -> int:
    match = re.match(r"^\s*(\d+):(\d+):(\d+)[:;](\d+)\s*$", str(timecode))
    if not match:
        raise ValueError(f"Unsupported timecode format: {timecode!r}")
    hh, mm, ss, ff = (int(match.group(i)) for i in range(1, 5))
    return (((hh * 60) + mm) * 60 + ss) * fps + ff


def format_timecode_frames(frame: int, fps: int, separator: str = ":") -> str:
    frame = max(0, int(frame))
    ff = frame % fps
    total_seconds = frame // fps
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    hh = total_minutes // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}{separator}{ff:02d}"


def resolve_refresh_available() -> Tuple[bool, str]:
    return ResolveRefreshClient().request_refresh()


def write_daemon_status(
    message: str,
    *,
    refresh_unavailable: bool = False,
    refresh_requested: bool = False,
    state_revision: int = 0,
    continuous_live_refresh: bool = False,
) -> None:
    write_json_atomic(
        chromaspace_root() / "resolve_bridge_status.json",
        {
            "type": "chromaspace_resolve_bridge_daemon",
            "pid": os.getpid(),
            "singleDaemonOwner": True,
            "stateRevision": state_revision,
            "refreshUnavailable": refresh_unavailable,
            "refreshRequested": refresh_requested,
            "continuousLiveRefresh": continuous_live_refresh,
            "message": message,
            "updatedAtUnix": time.time(),
        },
    )


def safe_call(obj, method_name: str, *args):
    try:
        method = getattr(obj, method_name, None)
        if method is None:
            return None
        return method(*args)
    except Exception:
        return None


def tool_list_contains_chromaspace(tools) -> bool:
    if tools is None:
        return False
    if not isinstance(tools, (list, tuple)):
        tools = [tools]
    for tool in tools:
        text = str(tool).lower()
        if any(token in text for token in CHROMASPACE_TOOL_TOKENS):
            return True
    return False


def label_suggests_draw_instance(label: str) -> bool:
    text = str(label or "").lower()
    return any(token in text for token in DRAW_LABEL_TOKENS)


def label_suggests_plot_instance(label: str) -> bool:
    text = str(label or "").lower()
    return any(token in text for token in PLOT_LABEL_TOKENS) and not label_suggests_draw_instance(text)


def graph_node_enabled_state(graph, index: int):
    for method_name in (
        "GetNodeEnabled",
        "IsNodeEnabled",
        "GetNodeEnable",
        "GetNodeEnabledState",
    ):
        value = safe_call(graph, method_name, index)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "enabled", "on"):
            return True
        if text in ("0", "false", "no", "disabled", "off"):
            return False
    for method_name in ("GetNodeBypass", "IsNodeBypassed", "GetNodeBypassState"):
        value = safe_call(graph, method_name, index)
        if value is None:
            continue
        if isinstance(value, bool):
            return not value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "bypassed", "off"):
            return False
        if text in ("0", "false", "no", "enabled", "on"):
            return True
    return None


def scan_graph_for_chromaspace(graph, graph_name: str):
    nodes = []
    if graph is None:
        return nodes
    count = safe_call(graph, "GetNumNodes")
    try:
        count = int(count or 0)
    except Exception:
        count = 0
    for index in range(1, count + 1):
        tools = safe_call(graph, "GetToolsInNode", index)
        if not tool_list_contains_chromaspace(tools):
            continue
        label = safe_call(graph, "GetNodeLabel", index)
        nodes.append(
            {
                "graph": graph_name,
                "nodeIndex": index,
                "nodeLabel": str(label or ""),
                "tools": [str(t) for t in tools] if isinstance(tools, (list, tuple)) else [str(tools)],
                "enabled": graph_node_enabled_state(graph, index),
                "labelSuggestsDraw": label_suggests_draw_instance(str(label or "")),
                "labelSuggestsPlot": label_suggests_plot_instance(str(label or "")),
            }
        )
    return nodes


def current_color_graphs(resolve):
    graphs = []
    try:
        manager = resolve.GetProjectManager()
        project = manager.GetCurrentProject() if manager else None
        timeline = project.GetCurrentTimeline() if project else None
        if timeline is None:
            return graphs, "No current timeline is active."

        item = safe_call(timeline, "GetCurrentVideoItem")
        if item is not None:
            graphs.append(("Clip", safe_call(item, "GetNodeGraph")))
            group = safe_call(item, "GetColorGroup")
            if group is not None:
                graphs.append(("Group Pre-Clip", safe_call(group, "GetPreClipNodeGraph")))
                graphs.append(("Group Post-Clip", safe_call(group, "GetPostClipNodeGraph")))

        graphs.append(("Timeline", safe_call(timeline, "GetNodeGraph")))
        return graphs, "Resolve graph scan completed."
    except Exception as exc:
        return graphs, f"Resolve graph scan failed: {exc}"


def scan_chromaspace_draw_instance(client: ResolveRefreshClient) -> Dict:
    now = time.monotonic()
    connected, message = client._connect(now)
    if not connected:
        return {
            "type": "chromaspace_draw_instance_status",
            "available": False,
            "chromaspaceNodeCount": 0,
            "drawInstanceLikely": False,
            "drawInstanceConfirmed": False,
            "message": message,
            "nodes": [],
            "updatedAtUnix": time.time(),
        }

    graphs, graph_message = current_color_graphs(client.resolve)
    nodes = []
    for graph_name, graph in graphs:
        nodes.extend(scan_graph_for_chromaspace(graph, graph_name))

    active_nodes = [node for node in nodes if node.get("enabled") is not False]
    disabled_nodes = [node for node in nodes if node.get("enabled") is False]
    confirmed = [node for node in active_nodes if node.get("labelSuggestsDraw")]
    likely = []
    for graph_name in ("Group Pre-Clip", "Clip", "Group Post-Clip", "Timeline"):
        graph_nodes = [node for node in active_nodes if node.get("graph") == graph_name]
        if len(graph_nodes) >= 2:
            likely.append(graph_nodes[0])
    if not likely and confirmed:
        likely = confirmed

    message = graph_message
    if confirmed:
        message = "Chromaspace draw instance found by node label."
    elif likely:
        message = "Chromaspace upstream draw instance is likely present; label it Draw or Identity Generator to confirm."
    elif disabled_nodes:
        message = "Chromaspace nodes are present, but the upstream draw instance appears disabled."
    elif nodes:
        message = "Chromaspace is present, but no separate upstream draw instance was detected."
    else:
        message = "No Chromaspace draw instance was detected in the current Resolve graph."

    return {
        "type": "chromaspace_draw_instance_status",
        "available": True,
        "chromaspaceNodeCount": len(nodes),
        "disabledChromaspaceNodeCount": len(disabled_nodes),
        "drawInstanceLikely": bool(likely),
        "drawInstanceConfirmed": bool(confirmed),
        "message": message,
        "nodes": nodes,
        "updatedAtUnix": time.time(),
    }


def node_index_value(node) -> int:
    try:
        return int(node.get("nodeIndex") or 0)
    except Exception:
        return 0


def choose_chromaspace_plot_node(nodes):
    if not nodes:
        return None
    nodes = [node for node in nodes if node.get("enabled") is not False]
    if not nodes:
        return None

    # Resolve exposes Color nodes in graph order. In the intended Chromaspace
    # setup, the identity/draw generator sits upstream, while the viewer/plot
    # instance sits downstream. So the highest Chromaspace node index in the
    # active Clip graph is the strongest unlabeled plot-node lead.
    graph_order = ("Clip", "Group Post-Clip", "Timeline", "Group Pre-Clip")
    for graph_name in graph_order:
        graph_nodes = sorted(
            [node for node in nodes if node.get("graph") == graph_name],
            key=node_index_value,
        )
        if not graph_nodes:
            continue
        labelled = [node for node in graph_nodes if node.get("labelSuggestsPlot")]
        if labelled:
            target = dict(labelled[-1])
            target["plotCandidateReason"] = "plot-label-highest-index"
            return target

        non_draw = [node for node in graph_nodes if not node.get("labelSuggestsDraw")]
        if non_draw:
            target = dict(non_draw[-1])
            target["plotCandidateReason"] = "highest-non-draw-chromaspace-node-index"
            return target

        # If every Chromaspace node in this graph is explicitly draw-labelled,
        # do not focus one for lasso drawing; that would select the generator
        # overlay instead of the plotting instance the user needs.
        if len(graph_nodes) == 1 and graph_nodes[0].get("labelSuggestsDraw"):
            continue

        target = dict(graph_nodes[-1])
        target["plotCandidateReason"] = "highest-chromaspace-node-index"
        return target

    labelled = sorted([node for node in nodes if node.get("labelSuggestsPlot")], key=node_index_value)
    if labelled:
        target = dict(labelled[-1])
        target["plotCandidateReason"] = "plot-label-fallback"
        return target

    non_draw = sorted([node for node in nodes if not node.get("labelSuggestsDraw")], key=node_index_value)
    if non_draw:
        target = dict(non_draw[-1])
        target["plotCandidateReason"] = "highest-non-draw-fallback"
        return target

    return None


def try_select_node(graph, node_index: int) -> Tuple[bool, str]:
    if graph is None:
        return False, "Resolve graph object is unavailable."
    for method_name in NODE_SELECT_METHODS:
        method = getattr(graph, method_name, None)
        if method is None:
            continue
        try:
            result = method(int(node_index))
            if not result:
                continue
            return True, f"Resolve graph selected node {node_index} using {method_name}."
        except Exception:
            continue
    return False, "Resolve scripting does not expose a Color node selection method on this graph."


def focus_chromaspace_plot_node(client: ResolveRefreshClient, revision: int) -> Dict:
    now = time.monotonic()
    connected, message = client._connect(now)
    if not connected:
        return {
            "plotNodeFocusRequested": True,
            "plotNodeFocusRequestedRevision": revision,
            "plotNodeFocusAvailable": False,
            "plotNodeFocusSucceeded": False,
            "plotNodeFocusMessage": message,
            "plotNode": None,
        }

    page_changed = safe_call(client.resolve, "OpenPage", "color")
    graphs, graph_message = current_color_graphs(client.resolve)
    nodes = []
    graph_lookup = {}
    for graph_name, graph in graphs:
        graph_lookup[graph_name] = graph
        nodes.extend(scan_graph_for_chromaspace(graph, graph_name))
    target = choose_chromaspace_plot_node(nodes)
    if target is None:
        return {
            "plotNodeFocusRequested": True,
            "plotNodeFocusRequestedRevision": revision,
            "plotNodeFocusAvailable": False,
            "plotNodeFocusSucceeded": False,
            "plotNodeFocusMessage": "No Chromaspace plot node was detected in the current Resolve graph.",
            "plotNode": None,
        }

    graph = graph_lookup.get(target.get("graph"))
    selected, select_message = try_select_node(graph, int(target.get("nodeIndex") or 0))
    if not selected and page_changed:
        select_message = (
            "Resolve opened the Color page and found the Chromaspace plot node, but the public "
            "scripting graph API does not expose node selection. Select the Chromaspace plot node to draw selections."
        )
    return {
        "plotNodeFocusRequested": True,
        "plotNodeFocusRequestedRevision": revision,
        "plotNodeFocusAvailable": selected,
        "plotNodeFocusSucceeded": selected,
        "plotNodeFocusMessage": select_message if selected else f"{graph_message} {select_message}",
        "plotNode": target,
    }


def merge_plot_focus_status(status: Dict, focus_status: Optional[Dict]) -> Dict:
    if not focus_status:
        return status
    merged = dict(status)
    merged.update(focus_status)
    return merged


def write_draw_instance_status(status: Dict) -> None:
    write_json_atomic(chromaspace_root() / "draw_instance_status.json", status)


def recent_mailbox_paths(root: pathlib.Path, pattern: str):
    wall_now = time.time()
    for path in root.glob(pattern):
        try:
            age = max(0.0, wall_now - path.stat().st_mtime)
        except Exception:
            continue
        if age <= MAILBOX_ACTIVE_SECONDS:
            yield path


def active_live_refresh_pulse(root: pathlib.Path) -> bool:
    live_pulse_paths = [chromaspace_root() / "live_refresh_pulse.json"]
    live_pulse_paths.extend(root.glob("*/live_refresh_pulse.json"))
    wall_now = time.time()
    for pulse_path in live_pulse_paths:
        pulse = read_json(pulse_path)
        try:
            age = max(0.0, wall_now - pulse_path.stat().st_mtime)
        except Exception:
            continue
        if bool(pulse.get("active")) and age <= LIVE_PULSE_STALE_SECONDS:
            return True
    return False


def graph_worker_main(requests: queue.Queue) -> None:
    client = ResolveRefreshClient()
    last_plot_focus_status: Optional[Dict] = None
    while True:
        action, request_path, revision = requests.get()
        try:
            if action == "focus":
                last_plot_focus_status = focus_chromaspace_plot_node(client, revision)
                write_json_atomic(request_path.with_name("plot_node_focus_status.json"), last_plot_focus_status)
                write_draw_instance_status(
                    merge_plot_focus_status(scan_chromaspace_draw_instance(client), last_plot_focus_status)
                )
            elif action == "scan":
                status = scan_chromaspace_draw_instance(client)
                status["scanDrawInstanceRequestedRevision"] = revision
                status = merge_plot_focus_status(status, last_plot_focus_status)
                write_json_atomic(request_path.with_name("draw_instance_status.json"), status)
                write_draw_instance_status(status)
        except Exception:
            pass
        finally:
            requests.task_done()


def refresh_worker_main(requests: queue.Queue, results: queue.Queue) -> None:
    client = ResolveRefreshClient()
    while True:
        ready, continuous_live_refresh = requests.get()
        try:
            available, message = client.request_refresh()
            results.put((ready, continuous_live_refresh, available, message))
        except Exception as exc:
            results.put((ready, continuous_live_refresh, False, f"Resolve refresh worker failed: {exc}"))
        finally:
            requests.task_done()


def main() -> int:
    root = sessions_root()
    owner = SingleBridgeOwner(chromaspace_root() / "resolve_bridge.lock")
    if not owner.acquire():
        return 0
    write_daemon_status("Chromaspace Resolve bridge daemon is running.")
    observed: Dict[pathlib.Path, int] = {}
    pending: Dict[pathlib.Path, int] = {}
    sent: Dict[pathlib.Path, int] = {}
    focus_sent: Dict[pathlib.Path, int] = {}
    draw_scan_sent: Dict[pathlib.Path, int] = {}
    graph_requests: queue.Queue = queue.Queue()
    refresh_requests: queue.Queue = queue.Queue(maxsize=1)
    refresh_results: queue.Queue = queue.Queue()
    threading.Thread(target=graph_worker_main, args=(graph_requests,), daemon=True).start()
    threading.Thread(
        target=refresh_worker_main,
        args=(refresh_requests, refresh_results),
        daemon=True,
    ).start()
    # Mailboxes are durable, but bridge actions are edge-triggered. Replaying
    # every request left by old viewer sessions can block live refresh for many
    # seconds on daemon restart. Seed current revisions as observed; active
    # viewers will continue to publish their live pulse and any new requests.
    for desired in recent_mailbox_paths(root, "*/desired_state.json"):
        state = read_json(desired)
        revision = int(state.get("hostRefreshRequestedRevision") or 0)
        observed[desired] = revision
        sent[desired] = revision
        focus_sent[desired] = int(state.get("focusPlotNodeRequestedRevision") or 0)
    for request_path in recent_mailbox_paths(root, "*/draw_instance_scan_request.json"):
        request = read_json(request_path)
        draw_scan_sent[request_path] = int(request.get("revision") or 0)
    last_refresh_at = 0.0
    last_daemon_status_at = time.monotonic()
    last_live_status_at = 0.0
    refresh_in_flight = False
    while True:
        now = time.monotonic()
        owner.write_heartbeat()
        continuous_live_refresh = active_live_refresh_pulse(root)
        if continuous_live_refresh and now - last_live_status_at >= 0.5:
            last_live_status_at = now
            last_daemon_status_at = now
            write_daemon_status(
                "Chromaspace live refresh pulse is active.",
                continuous_live_refresh=True,
            )
        while True:
            try:
                completed_ready, completed_continuous, available, message = refresh_results.get_nowait()
            except queue.Empty:
                break
            refresh_in_flight = False
            latest_revision = max((revision for _, revision in completed_ready), default=0)
            for desired, revision in completed_ready:
                sent[desired] = max(sent.get(desired, 0), revision)
                if pending.get(desired) == revision:
                    pending.pop(desired, None)
                write_json_atomic(
                    desired.with_name("bridge_status.json"),
                    {
                        "type": "chromaspace_resolve_bridge_status",
                        "stateRevision": revision,
                        "refreshUnavailable": not available,
                        "refreshRequested": available,
                        "message": message,
                        "updatedAtUnix": time.time(),
                    },
                )
            write_daemon_status(
                message,
                refresh_unavailable=not available,
                refresh_requested=available,
                state_revision=latest_revision,
                continuous_live_refresh=completed_continuous,
            )
            last_daemon_status_at = now
            refresh_results.task_done()
        focus_ready = []
        draw_scan_ready = []
        for desired in recent_mailbox_paths(root, "*/desired_state.json"):
            state = read_json(desired)
            focus_revision = int(state.get("focusPlotNodeRequestedRevision") or 0)
            if focus_revision > focus_sent.get(desired, 0):
                focus_ready.append((desired, focus_revision))
            revision = int(state.get("hostRefreshRequestedRevision") or 0)
            if revision <= 0:
                observed[desired] = 0
                pending.pop(desired, None)
                continue
            if observed.get(desired) == revision:
                continue
            observed[desired] = revision
            pending[desired] = revision
        for request_path in recent_mailbox_paths(root, "*/draw_instance_scan_request.json"):
            request = read_json(request_path)
            draw_scan_revision = int(request.get("revision") or 0)
            if draw_scan_revision > draw_scan_sent.get(request_path, 0):
                draw_scan_ready.append((request_path, draw_scan_revision))

        ready = []
        for desired, revision in list(pending.items()):
            if revision <= sent.get(desired, 0):
                pending.pop(desired, None)
                continue
            ready.append((desired, revision))

        refresh_due = bool(ready) or continuous_live_refresh

        if refresh_due and now - last_refresh_at >= LIVE_REFRESH_INTERVAL_SECONDS and not refresh_in_flight:
            last_refresh_at = now
            try:
                refresh_requests.put_nowait((list(ready), continuous_live_refresh))
                refresh_in_flight = True
            except queue.Full:
                pass
        elif now - last_daemon_status_at >= DAEMON_STATUS_SECONDS:
            write_daemon_status(
                "Chromaspace Resolve bridge daemon is running.",
                continuous_live_refresh=continuous_live_refresh,
            )
            last_daemon_status_at = now
        for desired, revision in focus_ready:
            focus_sent[desired] = revision
            graph_requests.put(("focus", desired, revision))
        for desired, revision in draw_scan_ready:
            draw_scan_sent[desired] = revision
            graph_requests.put(("scan", desired, revision))
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
