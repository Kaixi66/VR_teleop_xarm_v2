from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CheckResult:
    name: str
    level: str
    detail: str


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.level != "FAIL" for check in self.checks)

    def add(self, name: str, level: str, detail: str) -> None:
        self.checks.append(CheckResult(name, level, detail))

    def render(self) -> str:
        icons = {"PASS": "OK", "WARN": "WARN", "FAIL": "FAIL"}
        return "\n".join(
            f"[{icons.get(check.level, check.level):4}] {check.name}: {check.detail}" for check in self.checks
        )


def _run(command: list[str], *, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        detail = f"command timed out after {timeout:g}s"
        return subprocess.CompletedProcess(
            command, 124, stdout=stdout, stderr=f"{stderr.rstrip()}\n{detail}".strip()
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(exc))


def _failure_detail(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    return (result.stderr or result.stdout).strip() or fallback


def _run_ros_shell(config: Mapping[str, Any], command: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    paths = config["paths"]
    script = 'source "$1" && source "$2" && shift 2 && exec "$@"'
    return _run(
        [
            "bash",
            "-c",
            script,
            "vr-doctor",
            str(paths["ros_setup"]),
            str(paths["workspace_setup"]),
            "bash",
            "-c",
            command,
        ],
        timeout=timeout,
    )


def _nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _check_imports(python: str, modules: list[str]) -> tuple[bool, str]:
    code = "\n".join(f"import {module}" for module in modules)
    result = _run([python, "-B", "-c", code])
    if result.returncode == 0:
        return True, f"{python}: {', '.join(modules)}"
    message = (result.stderr or result.stdout).strip().splitlines()
    return False, message[-1] if message else f"{python} exited {result.returncode}"


def run_doctor(config: Mapping[str, Any]) -> DoctorReport:
    report = DoctorReport()
    paths = config["paths"]

    required_paths = {
        "ROS setup": paths["ros_setup"],
        "ROS workspace": paths["workspace_setup"],
        "system Python": paths["system_python"],
        "collector Python": paths["collector_python"],
    }
    for label, value in required_paths.items():
        path = Path(str(value))
        report.add(label, "PASS" if path.exists() else "FAIL", str(path))

    collector_python = str(paths["collector_python"])
    if Path(collector_python).exists():
        ok, detail = _check_imports(collector_python, ["cv2", "numpy", "xarm.wrapper", "pyrealsense2"])
        report.add("collector imports", "PASS" if ok else "FAIL", detail)

    ros_import = _run_ros_shell(
        config,
        f'"{paths["system_python"]}" -B -c "import rclpy; import xarm_msgs.srv; import xarm_msgs.msg; import xarm.wrapper"',
    )
    report.add(
        "ROS Python imports",
        "PASS" if ros_import.returncode == 0 else "FAIL",
        "rclpy, xarm_msgs and xarm SDK available"
        if ros_import.returncode == 0
        else (ros_import.stderr.strip() or "failed"),
    )

    package_check = _run_ros_shell(config, "ros2 pkg prefix xarm_api >/dev/null && ros2 pkg prefix xarm_description")
    report.add(
        "ROS packages",
        "PASS" if package_check.returncode == 0 else "FAIL",
        "xarm_api and xarm_description" if package_check.returncode == 0 else package_check.stderr.strip(),
    )

    robot_ip = str(config["robot"]["ip"])
    ping = _run(["ping", "-c", "1", "-W", "1", robot_ip], timeout=3.0)
    report.add(
        "UF850 network",
        "PASS" if ping.returncode == 0 else "FAIL",
        robot_ip if ping.returncode == 0 else _failure_detail(ping, robot_ip),
    )

    enumerate_tool = shutil.which("rs-enumerate-devices")
    if enumerate_tool:
        devices = _run([enumerate_tool, "-s"], timeout=10.0)
        output = f"{devices.stdout}\n{devices.stderr}"
        missing = [str(camera["serial"]) for camera in config["cameras"] if str(camera["serial"]) not in output]
        if devices.returncode != 0:
            report.add("RealSense cameras", "FAIL", devices.stderr.strip() or "enumeration failed")
        elif missing:
            report.add("RealSense cameras", "FAIL", f"missing serials: {', '.join(missing)}")
        else:
            report.add("RealSense cameras", "PASS", "both configured serials detected")
    else:
        report.add("RealSense cameras", "FAIL", "rs-enumerate-devices not found")

    data_root = Path(str(paths["data_root"]))
    usage = shutil.disk_usage(_nearest_existing(data_root))
    free_gb = usage.free / (1024**3)
    minimum = float(config["recording"]["min_free_gb"])
    report.add(
        "disk space",
        "PASS" if free_gb >= minimum else "FAIL",
        f"{free_gb:.1f} GiB free; minimum {minimum:.1f} GiB",
    )

    ss = _run(["ss", "-lunH"])
    listen_port = int(config["network"]["listen_port"])
    if ss.returncode != 0:
        report.add("Quest UDP port", "FAIL", _failure_detail(ss, "could not inspect UDP sockets"))
    else:
        occupied = any(f":{listen_port} " in f"{line} " for line in ss.stdout.splitlines())
        report.add(
            "Quest UDP port",
            "WARN" if occupied else "PASS",
            f"UDP {listen_port} is {'already in use' if occupied else 'available'}",
        )

    service_name = f'{str(config["robot"]["namespace"]).rstrip("/")}/vc_set_cartesian_velocity'
    services = _run_ros_shell(config, "ros2 service list", timeout=5.0)
    service_running = services.returncode == 0 and service_name in services.stdout.splitlines()
    report.add(
        "UF850 ROS service",
        "PASS" if service_running else "WARN",
        f"{service_name} {'is available' if service_running else 'is not running yet (start will launch it)'}",
    )

    nmcli = shutil.which("nmcli")
    if nmcli:
        iface = str(config["hotspot"]["interface"])
        ssid = str(config["hotspot"]["ssid"])
        # Query the active NetworkManager profile and its saved Wi-Fi settings.
        # `device wifi list` triggers a radio scan and can block for many seconds.
        active = _run(
            [nmcli, "--wait", "3", "-g", "GENERAL.CONNECTION", "device", "show", iface],
            timeout=5.0,
        )
        if active.returncode != 0:
            report.add("Quest hotspot", "WARN", _failure_detail(active, f"could not inspect {iface}"))
        else:
            profiles = [line.strip() for line in active.stdout.splitlines() if line.strip()]
            profile = profiles[0] if profiles else ""
            if not profile or profile == "--":
                hotspot_active = False
            else:
                details = _run(
                    [
                        nmcli,
                        "--wait",
                        "3",
                        "-g",
                        "802-11-wireless.ssid,802-11-wireless.mode",
                        "connection",
                        "show",
                        "id",
                        profile,
                    ],
                    timeout=5.0,
                )
                values = [line.strip() for line in details.stdout.splitlines()]
                hotspot_active = (
                    details.returncode == 0
                    and len(values) >= 2
                    and values[0] == ssid
                    and values[1].lower() == "ap"
                )
            report.add(
                "Quest hotspot",
                "PASS" if hotspot_active else "WARN",
                f"{ssid} {'active' if hotspot_active else 'not active; use start --hotspot or connect Quest via current LAN'}",
            )
    else:
        report.add("Quest hotspot", "WARN", "nmcli not found")

    try:
        socket.inet_aton(robot_ip)
    except OSError:
        report.add("robot IP format", "FAIL", robot_ip)
    return report
