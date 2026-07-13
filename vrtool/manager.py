from __future__ import annotations

import datetime as dt
import fcntl
import getpass
import ipaddress
import json
import os
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

class ManagerError(RuntimeError):
    pass


DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BRIDGE_READY_TIMEOUT_S = 35.0
QUEST_READY_TIMEOUT_S = 15.0


def _collection_runtime_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    runtime_dir = Path(str(config["paths"]["runtime_dir"]))
    return (
        runtime_dir / "collection.lease.json",
        runtime_dir / "teleop.armed.json",
        runtime_dir / "collection.abort.txt",
    )


def _pause_collection(config: Mapping[str, Any]) -> None:
    lease_path, armed_path, _abort_path = _collection_runtime_paths(config)
    lease_path.unlink(missing_ok=True)
    armed_path.unlink(missing_ok=True)


@contextmanager
def _stack_start_guard(config: Mapping[str, Any]):
    """Prevent reset/start races before the bridge owns arm_control.lock."""

    runtime_dir = Path(str(config["paths"]["runtime_dir"]))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    startup_path = runtime_dir / "stack_start.lock"
    arm_path = runtime_dir / "arm_control.lock"
    with startup_path.open("a+") as startup_handle:
        try:
            fcntl.flock(startup_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManagerError("another stack startup or arm reset is already in progress") from exc
        try:
            with arm_path.open("a+") as arm_handle:
                try:
                    fcntl.flock(arm_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ManagerError(
                        "arm control is owned by an active bridge/reset process"
                    ) from exc
                finally:
                    try:
                        fcntl.flock(arm_handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
            yield
        finally:
            fcntl.flock(startup_handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _collector_guard(config: Mapping[str, Any]):
    """Allow only one real collector to own cameras and the motion lease."""

    runtime_dir = Path(str(config["paths"]["runtime_dir"]))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / "collector.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ManagerError("another collector is already recording an episode") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _runtime_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["paths"]["runtime_dir"])) / "stack.json"


def _read_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return int(fields[21])
    except (OSError, ValueError, IndexError):
        return None


def _alive(entry: Mapping[str, Any]) -> bool:
    pid = int(entry["pid"])
    expected_ticks = int(entry.get("start_ticks", -1))
    current_ticks = _read_start_ticks(pid)
    return current_ticks is not None and (expected_ticks < 0 or current_ticks == expected_ticks)


def _managed_group_alive(entry: Mapping[str, Any]) -> bool:
    """Check the owned process group while guarding against leader PID reuse."""
    pid = int(entry["pid"])
    expected_ticks = int(entry.get("start_ticks", -1))
    current_ticks = _read_start_ticks(pid)
    if current_ticks is not None and expected_ticks >= 0 and current_ticks != expected_ticks:
        return False
    try:
        os.killpg(int(entry["pgid"]), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_state(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _runtime_path(config)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ManagerError(f"Invalid runtime state {path}: {exc}") from exc
    return state


def _write_state(config: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    path = _runtime_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, path)


def _remove_state(config: Mapping[str, Any]) -> None:
    _runtime_path(config).unlink(missing_ok=True)


def _ros_command(config: Mapping[str, Any], argv: Sequence[str]) -> list[str]:
    paths = config["paths"]
    script = 'source "$1" && source "$2" && shift 2 && exec "$@"'
    return [
        "bash",
        "-c",
        script,
        "vr-manager",
        str(paths["ros_setup"]),
        str(paths["workspace_setup"]),
        *argv,
    ]


def _spawn(
    label: str,
    command: list[str],
    log_path: Path,
    cwd: Path,
) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(0.25)
    if process.poll() is not None:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
        except OSError:
            pass
        raise ManagerError(f"{label} exited during startup (code {process.returncode}).\n{tail}")
    entry = {
        "label": label,
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": _read_start_ticks(process.pid),
        "command": command,
        "log": str(log_path),
    }
    return process, entry


def _tail_log(log_path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _wait_for_bridge_ready(
    process: subprocess.Popen[bytes],
    ready_path: Path,
    log_path: Path,
    timeout: float = BRIDGE_READY_TIMEOUT_S,
) -> None:
    """Wait for the bridge's post-initialization readiness marker.

    Merely observing a live Python process is insufficient: the bridge creates the
    marker only after the SDK connection, ROS services, UDP bind, and safety state
    have all initialized successfully.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path.is_file():
            try:
                payload = json.loads(ready_path.read_text())
                ready_pid = int(payload["pid"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # The bridge writes atomically, but tolerate a transient filesystem
                # visibility race instead of accepting an unverified marker.
                time.sleep(0.05)
                continue
            if ready_pid != process.pid:
                raise ManagerError(
                    f"Bridge readiness marker belongs to pid {ready_pid}, expected {process.pid}"
                )
            if process.poll() is None:
                return
        return_code = process.poll()
        if return_code is not None:
            tail = _tail_log(log_path)
            raise ManagerError(
                f"Bridge exited before becoming ready (code {return_code}).\n{tail}"
            )
        time.sleep(0.05)
    tail = _tail_log(log_path)
    raise ManagerError(
        f"Timed out after {timeout:.1f}s waiting for bridge readiness marker {ready_path}.\n{tail}"
    )


def _wait_for_quest_ready(
    process: subprocess.Popen[bytes],
    ready_path: Path,
    log_path: Path,
    timeout: float = QUEST_READY_TIMEOUT_S,
    listen_port: int = 5005,
) -> dict[str, Any]:
    """Wait until the current bridge receives one valid packet from Quest."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path.is_file():
            try:
                payload = json.loads(ready_path.read_text())
                ready_pid = int(payload["pid"])
                quest_ip = str(payload["quest_ip"])
                packet_size = int(payload["packet_size"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                time.sleep(0.05)
                continue
            if ready_pid != process.pid:
                raise ManagerError(
                    f"Quest readiness marker belongs to pid {ready_pid}, expected {process.pid}"
                )
            if not quest_ip or packet_size not in (24, 28, 36, 52):
                raise ManagerError(f"Invalid Quest readiness marker: {ready_path}")
            if process.poll() is None:
                return payload
        return_code = process.poll()
        if return_code is not None:
            tail = _tail_log(log_path)
            raise ManagerError(
                f"Bridge exited while waiting for Quest (code {return_code}).\n{tail}"
            )
        time.sleep(0.05)
    tail = _tail_log(log_path)
    raise ManagerError(
        f"No valid Quest UDP packet arrived on port {listen_port} within {timeout:.1f}s. "
        f"Check the Quest Wi-Fi, app target IP and UDP port.\n{tail}"
    )


def _driver_launch_argv(robot: Mapping[str, Any]) -> list[str]:
    namespace = str(robot["namespace"]).strip("/")
    if not namespace:
        raise ManagerError("robot.namespace must name a non-root ROS namespace")
    return [
        "ros2",
        "launch",
        "xarm_api",
        "uf850_driver.launch.py",
        f'robot_ip:={robot["ip"]}',
        f'report_type:={robot.get("report_type", "dev")}',
        f'add_gripper:={str(bool(robot.get("add_gripper", True))).lower()}',
        f"hw_ns:={namespace}",
    ]


def _ros_services(config: Mapping[str, Any]) -> set[str]:
    result = subprocess.run(
        _ros_command(config, ["ros2", "service", "list"]),
        text=True,
        capture_output=True,
        timeout=8.0,
        check=False,
    )
    return set(result.stdout.splitlines()) if result.returncode == 0 else set()


def _wait_for_service(config: Mapping[str, Any], name: str, timeout: float = 35.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if name in _ros_services(config):
            return
        time.sleep(0.5)
    raise ManagerError(f"Timed out waiting for ROS service: {name}")


def _hotspot_is_active(hotspot: Mapping[str, Any]) -> bool:
    iface = str(hotspot["interface"])
    ssid = str(hotspot["ssid"])
    try:
        active = subprocess.run(
            [
                "nmcli", "--wait", "3", "-g", "GENERAL.CONNECTION",
                "device", "show", iface,
            ],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    profile = next(
        (
            line.strip()
            for line in active.stdout.splitlines()
            if line.strip() and line.strip() != "--"
        ),
        "",
    ) if active.returncode == 0 else ""
    if not profile:
        return False
    try:
        details = subprocess.run(
            [
                "nmcli", "--wait", "3", "-g",
                "802-11-wireless.ssid,802-11-wireless.mode",
                "connection", "show", "id", profile,
            ],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    values = [line.strip() for line in details.stdout.splitlines()]
    return (
        details.returncode == 0
        and len(values) >= 2
        and values[0] == ssid
        and values[1].lower() == "ap"
    )


def _enable_hotspot(config: Mapping[str, Any]) -> None:
    hotspot = config["hotspot"]
    iface = str(hotspot["interface"])
    ssid = str(hotspot["ssid"])
    if _hotspot_is_active(hotspot):
        print(f"Quest hotspot {ssid} is already active on {iface}.")
        return

    env_name = str(hotspot.get("password_env", "VR_HOTSPOT_PASSWORD"))
    password = os.environ.get(env_name)
    if not password:
        password = getpass.getpass(f"Hotspot password ({env_name}): ")
    if len(password) < 8:
        raise ManagerError("Hotspot password must contain at least 8 characters")
    command = [
        "sudo",
        "nmcli",
        "device",
        "wifi",
        "hotspot",
        "ssid",
        ssid,
        "password",
        password,
        "ifname",
        iface,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise ManagerError("Failed to enable Quest hotspot")


def _interface_ipv4(interface: str) -> str | None:
    """Return the first usable IPv4 address reported by NetworkManager."""

    try:
        result = subprocess.run(
            [
                "nmcli",
                "--wait",
                "3",
                "-g",
                "IP4.ADDRESS",
                "device",
                "show",
                interface,
            ],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        candidate = line.strip().split("/", 1)[0]
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.version == 4 and not address.is_unspecified:
            return str(address)
    return None


def _wait_for_interface_ipv4(interface: str, timeout: float = 5.0) -> str | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        address = _interface_ipv4(interface)
        if address is not None:
            return address
        time.sleep(0.2)
    return _interface_ipv4(interface)


def _show_hotspot_target(config: Mapping[str, Any]) -> str:
    interface = str(config["hotspot"]["interface"])
    ssid = str(config["hotspot"]["ssid"])
    address = _wait_for_interface_ipv4(interface)
    if address is None:
        raise ManagerError(
            f"Quest hotspot is active but {interface} has no usable IPv4 address"
        )
    listen_port = int(config["network"]["listen_port"])
    feedback_port = int(config["network"]["feedback_port"])
    print(f"Quest Wi-Fi: {ssid}", flush=True)
    print(
        f"Quest app target: {address}:{listen_port} (feedback UDP {feedback_port})",
        flush=True,
    )
    return address


def prepare_quest(
    config: Mapping[str, Any],
    *,
    hotspot: bool = False,
) -> dict[str, Any]:
    """Prepare Quest networking without starting ROS or connecting to the arm."""

    statuses = stack_status(config)
    if any(alive for _label, alive, _pid, _log in statuses):
        raise ManagerError("Stop the active ROS/bridge stack before preparing Quest")

    if hotspot:
        _enable_hotspot(config)
        target_ip = _show_hotspot_target(config)
    else:
        target_ip = None
        print(
            "Quest hotspot automation is disabled; use this computer's LAN IPv4 "
            f"and UDP {int(config['network']['listen_port'])}.",
            flush=True,
        )
    return {
        "target_ip": target_ip,
        "listen_port": int(config["network"]["listen_port"]),
        "feedback_port": int(config["network"]["feedback_port"]),
    }


def _start_stack_guarded(
    config: Mapping[str, Any],
    *,
    hotspot: bool = False,
    wait_quest: bool = False,
) -> dict[str, Any]:
    existing = load_state(config)
    if existing:
        live = [
            entry for entry in existing.get("processes", []) if _managed_group_alive(entry)
        ]
        if live:
            labels = ", ".join(str(entry["label"]) for entry in live)
            raise ManagerError(f"Stack is already running: {labels}")
        _remove_state(config)

    if hotspot:
        _enable_hotspot(config)
        _show_hotspot_target(config)

    project_root = Path(str(config["_meta"]["project_root"]))
    logs_root = Path(str(config["paths"]["logs_dir"]))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_logs = logs_root / stamp
    processes: list[subprocess.Popen[bytes]] = []
    entries: list[dict[str, Any]] = []
    lease_path, armed_path, abort_path = _collection_runtime_paths(config)
    for stale_path in (lease_path, armed_path, abort_path):
        stale_path.unlink(missing_ok=True)

    try:
        description_command = _ros_command(
            config,
            [
                "ros2",
                "launch",
                "xarm_description",
                "_robot_description.launch.py",
                "robot_type:=uf850",
                "dof:=6",
                "joint_mesh_suffix:=stl",
            ],
        )
        process, entry = _spawn("description", description_command, session_logs / "description.log", project_root)
        processes.append(process)
        entries.append(entry)

        robot = config["robot"]
        driver_command = _ros_command(config, _driver_launch_argv(robot))
        process, entry = _spawn("driver", driver_command, session_logs / "driver.log", project_root)
        processes.append(process)
        entries.append(entry)

        namespace = str(robot["namespace"]).rstrip("/")
        _wait_for_service(config, f"{namespace}/vc_set_cartesian_velocity")

        ready_path = session_logs / "bridge.ready.json"
        ready_path.unlink(missing_ok=True)
        quest_ready_path = session_logs / "quest.ready.json"
        quest_ready_path.unlink(missing_ok=True)
        bridge_log_path = session_logs / "bridge.log"
        bridge_command = _ros_command(
            config,
            [
                str(config["paths"]["system_python"]),
                "-B",
                "-m",
                "vrtool.bridge",
                "--config",
                str(config["_meta"]["config_path"]),
                "--ready-file",
                str(ready_path),
                "--quest-ready-file",
                str(quest_ready_path),
                "--control-lock-file",
                str(Path(str(config["paths"]["runtime_dir"])) / "arm_control.lock"),
                "--collection-lease-file",
                str(lease_path),
                "--teleop-armed-file",
                str(armed_path),
            ],
        )
        process, entry = _spawn("bridge", bridge_command, bridge_log_path, project_root)
        processes.append(process)
        entry["ready_file"] = str(ready_path)
        entry["quest_ready_file"] = str(quest_ready_path)
        entries.append(entry)
        _wait_for_bridge_ready(process, ready_path, bridge_log_path)

        state = {
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "log_dir": str(session_logs),
            "processes": entries,
        }
        if wait_quest:
            # Persist ownership before the potentially long network wait so a
            # separate `vrctl stop` can still clean up after a manager crash.
            state["startup_in_progress"] = "waiting_for_quest"
        _write_state(config, state)

        quest_status: dict[str, Any] | None = None
        if wait_quest:
            print(
                f"Bridge is ready; confirming fresh Quest data on UDP "
                f"{int(config['network']['listen_port'])}...",
                flush=True,
            )
            quest_status = _wait_for_quest_ready(
                process,
                quest_ready_path,
                bridge_log_path,
                timeout=float(
                    config["network"].get(
                        "quest_ready_timeout_s", QUEST_READY_TIMEOUT_S
                    )
                ),
                listen_port=int(config["network"]["listen_port"]),
            )
            print(
                f"Quest control stream confirmed from {quest_status['quest_ip']}.",
                flush=True,
            )
            state.pop("startup_in_progress", None)
        if quest_status is not None:
            state["quest"] = quest_status
        _write_state(config, state)
        return state
    except BaseException as startup_error:
        survivors = _stop_entries(reversed(entries), default_timeout=5.0)
        if survivors:
            # Preserve enough ownership information for a later `vrctl stop` retry.
            failure_state = {
                "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "log_dir": str(session_logs),
                "startup_failed": True,
                "processes": survivors,
            }
            _write_state(config, failure_state)
            labels = ", ".join(str(entry.get("label", "unknown")) for entry in survivors)
            raise ManagerError(
                f"Stack startup failed and these process groups could not be stopped: {labels}. "
                "Runtime state was retained; run './vrctl stop' again."
            ) from startup_error
        _remove_state(config)
        raise


def start_stack(
    config: Mapping[str, Any],
    *,
    hotspot: bool = False,
    wait_quest: bool = False,
) -> dict[str, Any]:
    with _stack_start_guard(config):
        return _start_stack_guarded(config, hotspot=hotspot, wait_quest=wait_quest)


def _wait_group_dead(entry: Mapping[str, Any], timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not _managed_group_alive(entry):
            return True
        time.sleep(0.1)
    return not _managed_group_alive(entry)


def _signal_group(entry: Mapping[str, Any], signum: int) -> None:
    try:
        os.killpg(int(entry["pgid"]), signum)
    except ProcessLookupError:
        pass


def _stop_entry(entry: Mapping[str, Any], timeout: float) -> bool:
    """Stop one owned process group and report whether it is actually gone."""

    if not _managed_group_alive(entry):
        return True
    _signal_group(entry, signal.SIGINT)
    if _wait_group_dead(entry, timeout):
        return True
    _signal_group(entry, signal.SIGTERM)
    if _wait_group_dead(entry, 2.0):
        return True
    _signal_group(entry, signal.SIGKILL)
    return _wait_group_dead(entry, 2.0)


def _stop_entries(
    entries: Sequence[Mapping[str, Any]] | Any,
    *,
    default_timeout: float,
) -> list[dict[str, Any]]:
    survivors: list[dict[str, Any]] = []
    for entry in entries:
        timeout = 8.0 if entry.get("label") == "bridge" else default_timeout
        try:
            stopped = _stop_entry(entry, timeout=timeout)
        except (OSError, ValueError, TypeError, KeyError):
            stopped = False
        if not stopped:
            survivors.append(dict(entry))
    return survivors


def stop_stack(config: Mapping[str, Any]) -> None:
    _pause_collection(config)
    state = load_state(config)
    if not state:
        return
    priority = {"bridge": 0, "driver": 1, "description": 2}
    entries = sorted(state.get("processes", []), key=lambda item: priority.get(str(item.get("label")), 99))
    survivors = _stop_entries(entries, default_timeout=5.0)
    if survivors:
        retained = dict(state)
        retained["processes"] = survivors
        retained["stop_failed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_state(config, retained)
        labels = ", ".join(str(entry.get("label", "unknown")) for entry in survivors)
        raise ManagerError(
            f"Could not stop process groups: {labels}. Runtime state was retained for retry."
        )
    _remove_state(config)


def stack_status(config: Mapping[str, Any]) -> list[tuple[str, bool, int, str]]:
    state = load_state(config)
    if not state:
        return []
    return [
        (
            str(entry["label"]),
            _managed_group_alive(entry),
            int(entry["pid"]),
            str(entry.get("log", "")),
        )
        for entry in state.get("processes", [])
    ]


def _run_collector_guarded(
    config: Mapping[str, Any],
    *,
    dataset: str,
    task: str = "",
    mock: bool = False,
) -> int:
    if not DATASET_RE.fullmatch(dataset):
        raise ManagerError("Dataset name may contain only letters, digits, '.', '_' and '-'")
    if not mock:
        statuses = stack_status(config)
        if not statuses or not all(item[1] for item in statuses):
            raise ManagerError("The ROS/bridge stack is not healthy. Run './vrctl start' first.")

    lease_path, armed_path, abort_path = _collection_runtime_paths(config)
    for stale_path in (lease_path, armed_path, abort_path):
        stale_path.unlink(missing_ok=True)

    command = [
        str(config["paths"]["collector_python"]),
        "-B",
        "-m",
        "vrtool.collector",
        "--config",
        str(config["_meta"]["config_path"]),
        "--dataset",
        dataset,
    ]
    if task:
        command.extend(["--task", task])
    if mock:
        command.append("--mock")
    else:
        command.extend(
            [
                "--collection-lease-file",
                str(lease_path),
                "--control-abort-file",
                str(abort_path),
                "--teleop-armed-file",
                str(armed_path),
            ]
        )

    project_root = Path(str(config["_meta"]["project_root"]))
    process = subprocess.Popen(command, cwd=project_root, start_new_session=True)
    last_health_check = -1.0

    def stop_collector_child() -> int:
        _pause_collection(config)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            return process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                return process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    return process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    return 130

    try:
        while True:
            try:
                code = process.wait(timeout=0.1)
                return code
            except subprocess.TimeoutExpired:
                pass

            now = time.monotonic()
            if not mock and now - last_health_check >= 0.25:
                last_health_check = now
                statuses = stack_status(config)
                if not statuses or not all(item[1] for item in statuses):
                    _pause_collection(config)
                    abort_path.write_text(
                        "control stack stopped during collection; episode is incomplete",
                        encoding="utf-8",
                    )
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError):
                            pass
                        try:
                            process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            pass
                    return 1
    except KeyboardInterrupt:
        return stop_collector_child()
    except BaseException:
        stop_collector_child()
        raise
    finally:
        _pause_collection(config)
        abort_path.unlink(missing_ok=True)


def run_collector(
    config: Mapping[str, Any],
    *,
    dataset: str,
    task: str = "",
    mock: bool = False,
) -> int:
    with _collector_guard(config):
        return _run_collector_guarded(
            config,
            dataset=dataset,
            task=task,
            mock=mock,
        )
