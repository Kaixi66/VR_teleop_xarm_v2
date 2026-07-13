from __future__ import annotations

import copy
import ipaddress
import math
import os
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when the collection configuration is incomplete or invalid."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
_ROS_NAMESPACE_RE = re.compile(r"^/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolve_path(value: str, base: Path) -> str:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = base / expanded
    # Keep virtual-environment interpreter symlinks intact: resolving
    # ~/robot_env/bin/python to /usr/bin/python would discard the venv's
    # site-packages (notably pyrealsense2).
    return str(expanded.absolute())


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    config = copy.deepcopy(config)
    # Validate the TOML values before path coercion so an integer or boolean in
    # a path field cannot silently become a plausible-looking string.
    validate_config(config)
    paths = config.setdefault("paths", {})
    for key in (
        "ros_setup",
        "workspace_setup",
        "system_python",
        "collector_python",
        "data_root",
        "runtime_dir",
        "logs_dir",
    ):
        if key in paths:
            paths[key] = _resolve_path(str(paths[key]), config_path.parent)

    config["_meta"] = {
        "config_path": str(config_path),
        "project_root": str(config_path.parent),
    }
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise ConfigError("Configuration root must be a TOML table")

    required_sections = (
        "robot",
        "network",
        "control",
        "safety",
        "recording",
        "paths",
        "hotspot",
    )
    sections: dict[str, Mapping[str, Any]] = {}
    missing: list[str] = []
    for name in required_sections:
        value = config.get(name)
        if not isinstance(value, Mapping):
            missing.append(name)
        else:
            sections[name] = value
    if missing:
        raise ConfigError(f"Missing or invalid configuration sections: {', '.join(missing)}")

    robot = sections["robot"]
    robot_ip = _required_string(robot, "ip", "robot")
    try:
        ipaddress.IPv4Address(robot_ip)
    except ipaddress.AddressValueError as exc:
        raise ConfigError(f"robot.ip must be a valid IPv4 address, got {robot_ip!r}") from exc
    namespace = _required_string(robot, "namespace", "robot")
    if not _ROS_NAMESPACE_RE.fullmatch(namespace):
        raise ConfigError(
            "robot.namespace must be an absolute ROS namespace such as '/ufactory'"
        )
    report_type = _required_string(robot, "report_type", "robot")
    if report_type not in {"normal", "rich", "dev"}:
        raise ConfigError("robot.report_type must be one of: normal, rich, dev")
    _required_bool(robot, "add_gripper", "robot")

    network = sections["network"]
    _required_string(network, "bind_host", "network")
    _required_int(network, "listen_port", "network", minimum=1, maximum=65535)
    _required_int(network, "feedback_port", "network", minimum=1, maximum=65535)
    _required_number(network, "feedback_hz", "network", minimum=0.0, exclusive_min=True)
    if "quest_ready_timeout_s" in network:
        _required_number(
            network,
            "quest_ready_timeout_s",
            "network",
            minimum=0.0,
            exclusive_min=True,
        )
    quest_ip = network.get("quest_ip")
    if not isinstance(quest_ip, str):
        raise ConfigError("network.quest_ip must be a string (empty means learn the first sender)")
    if quest_ip.strip():
        try:
            ipaddress.IPv4Address(quest_ip.strip())
        except ipaddress.AddressValueError as exc:
            raise ConfigError("network.quest_ip must be empty or a valid IPv4 address") from exc

    control = sections["control"]
    for key in (
        "publish_hz",
        "linear_speed_mm_s",
        "angular_speed_rad_s",
        "command_duration_s",
        "linear_accel_mm_s2",
        "angular_accel_rad_s2",
    ):
        _required_number(control, key, "control", minimum=0.0, exclusive_min=True)
    _required_number(control, "deadband", "control", minimum=0.0, maximum=1.0, exclusive_max=True)

    safety = sections["safety"]
    _required_number(safety, "watchdog_s", "safety", minimum=0.0, exclusive_min=True)
    _required_number(
        safety,
        "collection_lease_timeout_s",
        "safety",
        minimum=0.0,
        exclusive_min=True,
    )
    _required_int(safety, "neutral_packets_to_arm", "safety", minimum=1)
    _required_bool(safety, "workspace_enabled", "safety")
    for key in ("x_mm", "y_mm", "z_mm"):
        bounds = safety.get(key)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ConfigError(f"safety.{key} must be [minimum, maximum]")
        lower = _finite_number(bounds[0], f"safety.{key}[0]")
        upper = _finite_number(bounds[1], f"safety.{key}[1]")
        if lower >= upper:
            raise ConfigError(f"safety.{key} must have minimum < maximum")

    recording = sections["recording"]
    _required_number(recording, "hz", "recording", minimum=0.0, exclusive_min=True)
    for key in ("width", "height", "fps"):
        _required_int(recording, key, "recording", minimum=1)
    pixel_format = _required_string(recording, "pixel_format", "recording").lower()
    if pixel_format != "yuyv":
        raise ConfigError("recording.pixel_format must be 'yuyv'")
    _required_int(recording, "frame_buffer_size", "recording", minimum=2)
    _required_int(recording, "warmup_frames", "recording", minimum=0)
    _required_number(recording, "max_camera_skew_ms", "recording", minimum=0.0)
    _required_number(
        recording, "max_frame_age_ms", "recording", minimum=0.0, exclusive_min=True
    )
    _required_int(recording, "save_queue_size", "recording", minimum=1)
    _required_int(recording, "jpeg_quality", "recording", minimum=1, maximum=100)
    _required_number(recording, "min_free_gb", "recording", minimum=0.0)
    _required_number(
        recording, "failure_timeout_s", "recording", minimum=0.0, exclusive_min=True
    )
    _required_number(
        recording, "startup_timeout_s", "recording", minimum=0.0, exclusive_min=True
    )
    if "gripper_move_threshold" in recording:
        _required_number(recording, "gripper_move_threshold", "recording", minimum=0.0)
    if "consecutive_failure_limit" in recording:
        _required_int(recording, "consecutive_failure_limit", "recording", minimum=1)

    cameras = config.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 2:
        raise ConfigError("Exactly two [[cameras]] entries are required")
    expected = ("cam_0", "cam_1")
    names: list[str] = []
    serials: list[str] = []
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            raise ConfigError(f"cameras[{index}] must be a TOML table")
        names.append(_required_string(camera, "name", f"cameras[{index}]"))
        serials.append(_required_string(camera, "serial", f"cameras[{index}]"))
        _required_string(camera, "role", f"cameras[{index}]")
    if tuple(names) != expected:
        raise ConfigError(f"Camera order must be {expected}, got {tuple(names)}")
    if len(set(serials)) != 2:
        raise ConfigError("Camera serial numbers must be unique")

    paths = sections["paths"]
    for key in (
        "ros_setup",
        "workspace_setup",
        "system_python",
        "collector_python",
        "data_root",
        "runtime_dir",
        "logs_dir",
    ):
        _required_string(paths, key, "paths")

    hotspot = sections["hotspot"]
    ssid = _required_string(hotspot, "ssid", "hotspot")
    if len(ssid.encode("utf-8")) > 32:
        raise ConfigError("hotspot.ssid must be at most 32 UTF-8 bytes")
    interface = _required_string(hotspot, "interface", "hotspot")
    if any(character.isspace() for character in interface):
        raise ConfigError("hotspot.interface must not contain whitespace")
    password_env = _required_string(hotspot, "password_env", "hotspot")
    if not _ENV_NAME_RE.fullmatch(password_env):
        raise ConfigError("hotspot.password_env must be a valid environment variable name")


def _required_string(section: Mapping[str, Any], key: str, section_name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section_name}.{key} must be a non-empty string")
    return value.strip()


def _required_bool(section: Mapping[str, Any], key: str, section_name: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{section_name}.{key} must be true or false")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigError(f"{label} must be finite")
    return converted


def _required_number(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> float:
    label = f"{section_name}.{key}"
    if key not in section:
        raise ConfigError(f"{label} is required")
    value = _finite_number(section[key], label)
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        operator = ">" if exclusive_min else ">="
        raise ConfigError(f"{label} must be {operator} {minimum:g}")
    if maximum is not None and (value >= maximum if exclusive_max else value > maximum):
        operator = "<" if exclusive_max else "<="
        raise ConfigError(f"{label} must be {operator} {maximum:g}")
    return value


def _required_int(
    section: Mapping[str, Any],
    key: str,
    section_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    label = f"{section_name}.{key}"
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{label} must be <= {maximum}")
    return value


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable configuration snapshot without internal metadata."""
    return {key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")}
