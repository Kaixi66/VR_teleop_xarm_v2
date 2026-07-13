"""Safe Quest-to-UF850 teleoperation bridge.

The module deliberately keeps ROS 2 and xArm SDK imports inside :func:`run_bridge`.
Packet parsing and all command shaping are plain Python so they can be tested on a
development machine that has neither dependency installed.

The Quest wire format is compatible with the historical ``bridge_ghsun.py``:

* 24 bytes: six little-endian float32 velocity axes
* 28 bytes: axes plus the right trigger/gripper value
* 36 bytes: the above plus A and B
* 52 bytes: the above plus four additional buttons

Only the axes and gripper value are acted on.  Extra buttons are retained in the
parsed packet for diagnostics, but never start preset trajectories.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import math
import os
import signal
import socket
import struct
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


LOGGER = logging.getLogger("vrtool.bridge")
ZERO_COMMAND = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_PACKET_FLOATS_BY_SIZE = {24: 6, 28: 7, 36: 9, 52: 13}
MIN_TIMED_VELOCITY_FIRMWARE = (1, 8, 0)


@contextmanager
def arm_control_lock(path: Path):
    """Hold the cross-process arm-control lock for the bridge lifetime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"arm control is already owned by another bridge/reset process: {path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class PacketError(ValueError):
    """Raised when a Quest datagram is malformed or unsafe."""


@dataclass(frozen=True)
class QuestPacket:
    """A validated Quest command packet."""

    axes: tuple[float, float, float, float, float, float]
    gripper: float | None = None
    buttons: tuple[float, ...] = ()


@dataclass(frozen=True)
class Workspace:
    """Cartesian base-frame workspace in millimetres."""

    enabled: bool = True
    x_mm: tuple[float, float] = (0.0, 600.0)
    y_mm: tuple[float, float] = (0.0, 650.0)
    z_mm: tuple[float, float] = (170.0, 650.0)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return (self.x_mm, self.y_mm, self.z_mm)


@dataclass(frozen=True)
class BridgeConfig:
    """Runtime configuration for the teleoperation bridge.

    ``from_mapping`` accepts the nested mapping produced by ``tomllib``.  The
    names match the project's ``config.toml`` while a few historical aliases are
    accepted to make programmatic use less brittle.
    """

    robot_ip: str = "192.168.1.230"
    ros_namespace: str = "/ufactory"
    bind_host: str = "0.0.0.0"
    listen_port: int = 5005
    feedback_port: int = 5006
    quest_ip: str | None = None
    publish_hz: float = 50.0
    linear_speed_mm_s: float = 200.0
    angular_speed_rad_s: float = 0.3
    command_duration_s: float = 0.15
    deadband: float = 0.05
    linear_accel_mm_s2: float = 800.0
    angular_accel_rad_s2: float = 1.2
    watchdog_s: float = 0.2
    workspace: Workspace = field(default_factory=Workspace)
    feedback_interval_s: float = 0.25
    position_poll_hz: float = 20.0
    position_stale_s: float = 0.25
    service_timeout_s: float = 5.0
    state_failure_limit: int = 10
    shutdown_zero_repeats: int = 3
    collection_lease_timeout_s: float = 0.5
    neutral_packets_to_arm: int = 3
    gripper_threshold: float = 0.5
    gripper_open_position: int = 850
    gripper_closed_position: int = 0
    gripper_speed: int = 3000
    gripper_force: int = 1000

    def __post_init__(self) -> None:
        namespace = self.ros_namespace.strip()
        if not namespace:
            raise ValueError("robot.namespace must not be empty")
        if not namespace.startswith("/"):
            object.__setattr__(self, "ros_namespace", f"/{namespace}")
        elif len(namespace) > 1:
            object.__setattr__(self, "ros_namespace", namespace.rstrip("/"))

        if not self.robot_ip.strip():
            raise ValueError("robot.ip must not be empty")
        for name, port in (
            ("network.listen_port", self.listen_port),
            ("network.feedback_port", self.feedback_port),
        ):
            if not 1 <= int(port) <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
        for name, value in (
            ("control.publish_hz", self.publish_hz),
            ("control.linear_speed_mm_s", self.linear_speed_mm_s),
            ("control.angular_speed_rad_s", self.angular_speed_rad_s),
            ("control.command_duration_s", self.command_duration_s),
            ("control.linear_accel_mm_s2", self.linear_accel_mm_s2),
            ("control.angular_accel_rad_s2", self.angular_accel_rad_s2),
            ("safety.watchdog_s", self.watchdog_s),
            ("feedback_interval_s", self.feedback_interval_s),
            ("position_poll_hz", self.position_poll_hz),
            ("position_stale_s", self.position_stale_s),
            ("service_timeout_s", self.service_timeout_s),
            ("safety.collection_lease_timeout_s", self.collection_lease_timeout_s),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if not 0.0 <= self.deadband < 1.0:
            raise ValueError("control.deadband must be in [0, 1)")
        if not 0.0 <= self.gripper_threshold <= 1.0:
            raise ValueError("gripper threshold must be in [0, 1]")
        if self.state_failure_limit < 1:
            raise ValueError("state_failure_limit must be at least 1")
        if self.shutdown_zero_repeats < 1:
            raise ValueError("shutdown_zero_repeats must be at least 1")
        if self.neutral_packets_to_arm < 1:
            raise ValueError("neutral_packets_to_arm must be at least 1")
        for axis, bounds in zip("xyz", self.workspace.bounds):
            if (
                len(bounds) != 2
                or not all(math.isfinite(float(v)) for v in bounds)
                or float(bounds[0]) >= float(bounds[1])
            ):
                raise ValueError(f"safety.{axis}_mm must be [lower, upper]")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | "BridgeConfig") -> "BridgeConfig":
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("bridge config must be a mapping or BridgeConfig")

        robot = _mapping_section(config, "robot")
        network = _mapping_section(config, "network")
        control = _mapping_section(config, "control")
        safety = _mapping_section(config, "safety")
        feedback = _mapping_section(config, "feedback")
        gripper = _mapping_section(config, "gripper")

        def pick(section: Mapping[str, Any], names: Sequence[str], default: Any) -> Any:
            for name in names:
                if name in section:
                    return section[name]
            for name in names:
                if name in config:
                    return config[name]
            return default

        workspace_enabled = _as_bool(
            pick(safety, ("workspace_enabled",), Workspace.enabled)
        )
        workspace = Workspace(
            enabled=workspace_enabled,
            x_mm=_pair(pick(safety, ("x_mm", "workspace_x_mm"), Workspace.x_mm), "x_mm"),
            y_mm=_pair(pick(safety, ("y_mm", "workspace_y_mm"), Workspace.y_mm), "y_mm"),
            z_mm=_pair(pick(safety, ("z_mm", "workspace_z_mm"), Workspace.z_mm), "z_mm"),
        )
        quest_ip_value = pick(network, ("quest_ip",), None)
        quest_ip = str(quest_ip_value).strip() if quest_ip_value is not None else None
        if not quest_ip:
            quest_ip = None
        feedback_interval = pick(
            feedback, ("interval_s", "feedback_interval_s"), None
        )
        if feedback_interval is None:
            feedback_hz = float(
                pick(network, ("feedback_hz",), 1.0 / cls.feedback_interval_s)
            )
            if not math.isfinite(feedback_hz) or feedback_hz <= 0.0:
                raise ValueError("network.feedback_hz must be finite and greater than zero")
            feedback_interval = 1.0 / feedback_hz

        return cls(
            robot_ip=str(pick(robot, ("ip", "robot_ip"), cls.robot_ip)),
            ros_namespace=str(
                pick(robot, ("namespace", "ros_namespace"), cls.ros_namespace)
            ),
            bind_host=str(pick(network, ("bind_host", "host"), cls.bind_host)),
            listen_port=int(
                pick(network, ("listen_port", "command_port", "port"), cls.listen_port)
            ),
            feedback_port=int(pick(network, ("feedback_port",), cls.feedback_port)),
            quest_ip=quest_ip,
            publish_hz=float(pick(control, ("publish_hz",), cls.publish_hz)),
            linear_speed_mm_s=float(
                pick(control, ("linear_speed_mm_s", "move_gain"), cls.linear_speed_mm_s)
            ),
            angular_speed_rad_s=float(
                pick(
                    control,
                    ("angular_speed_rad_s", "rotation_gain"),
                    cls.angular_speed_rad_s,
                )
            ),
            command_duration_s=float(
                pick(control, ("command_duration_s", "duration_s"), cls.command_duration_s)
            ),
            deadband=float(pick(control, ("deadband",), cls.deadband)),
            linear_accel_mm_s2=float(
                pick(control, ("linear_accel_mm_s2",), cls.linear_accel_mm_s2)
            ),
            angular_accel_rad_s2=float(
                pick(control, ("angular_accel_rad_s2",), cls.angular_accel_rad_s2)
            ),
            watchdog_s=float(
                pick(safety, ("watchdog_s", "watchdog_timeout_s"), cls.watchdog_s)
            ),
            workspace=workspace,
            feedback_interval_s=float(feedback_interval),
            position_poll_hz=float(
                pick(safety, ("position_poll_hz",), cls.position_poll_hz)
            ),
            position_stale_s=float(
                pick(safety, ("position_stale_s",), cls.position_stale_s)
            ),
            service_timeout_s=float(
                pick(robot, ("service_timeout_s",), cls.service_timeout_s)
            ),
            state_failure_limit=int(
                pick(safety, ("state_failure_limit",), cls.state_failure_limit)
            ),
            shutdown_zero_repeats=int(
                pick(safety, ("shutdown_zero_repeats",), cls.shutdown_zero_repeats)
            ),
            collection_lease_timeout_s=float(
                pick(
                    safety,
                    ("collection_lease_timeout_s",),
                    cls.collection_lease_timeout_s,
                )
            ),
            neutral_packets_to_arm=int(
                pick(
                    safety,
                    ("neutral_packets_to_arm",),
                    cls.neutral_packets_to_arm,
                )
            ),
            gripper_threshold=float(
                pick(gripper, ("threshold",), cls.gripper_threshold)
            ),
            gripper_open_position=int(
                pick(gripper, ("open_position",), cls.gripper_open_position)
            ),
            gripper_closed_position=int(
                pick(gripper, ("closed_position",), cls.gripper_closed_position)
            ),
            gripper_speed=int(pick(gripper, ("speed",), cls.gripper_speed)),
            gripper_force=int(pick(gripper, ("force",), cls.gripper_force)),
        )


def _mapping_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"[{name}] must be a mapping")
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"safety.{name} must contain exactly two numbers")
    return (float(value[0]), float(value[1]))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"expected a boolean, got {value!r}")


def parse_packet(data: bytes) -> QuestPacket:
    """Decode and validate one supported Quest UDP datagram.

    All received floats, including currently unused buttons, must be finite.  This
    makes an invalid packet unable to acquire the peer lock or refresh the safety
    watchdog.
    """

    count = _PACKET_FLOATS_BY_SIZE.get(len(data))
    if count is None:
        expected = ", ".join(str(size) for size in _PACKET_FLOATS_BY_SIZE)
        raise PacketError(f"unsupported packet length {len(data)}; expected {expected} bytes")
    try:
        values = struct.unpack(f"<{count}f", data)
    except struct.error as exc:  # Defensive; the size check should make this impossible.
        raise PacketError(f"could not unpack Quest packet: {exc}") from exc
    if not all(math.isfinite(value) for value in values):
        raise PacketError("packet contains NaN or infinity")
    axes = tuple(float(value) for value in values[:6])
    gripper = float(values[6]) if count >= 7 else None
    buttons = tuple(float(value) for value in values[7:])
    return QuestPacket(axes=axes, gripper=gripper, buttons=buttons)  # type: ignore[arg-type]


def apply_deadband(value: float, deadband: float) -> float:
    """Zero small controller drift without changing values outside the band."""

    if not math.isfinite(value):
        raise ValueError("controller input must be finite")
    if not 0.0 <= deadband < 1.0:
        raise ValueError("deadband must be in [0, 1)")
    return 0.0 if abs(value) <= deadband else value


def sanitize_axes(values: Sequence[float], deadband: float) -> tuple[float, ...]:
    """Validate, clamp to ``[-1, 1]``, and deadband six normalized axes."""

    if len(values) != 6:
        raise ValueError("exactly six controller axes are required")
    result: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("controller axes must be finite")
        result.append(apply_deadband(max(-1.0, min(1.0, value)), deadband))
    return tuple(result)


def scale_axes(
    axes: Sequence[float], linear_speed_mm_s: float, angular_speed_rad_s: float
) -> tuple[float, ...]:
    """Scale normalized axes into the xArm Cartesian velocity units."""

    if len(axes) != 6:
        raise ValueError("exactly six controller axes are required")
    return tuple(
        float(value) * (linear_speed_mm_s if index < 3 else angular_speed_rad_s)
        for index, value in enumerate(axes)
    )


def slew_limit(
    previous: Sequence[float],
    target: Sequence[float],
    dt_s: float,
    linear_accel_mm_s2: float,
    angular_accel_rad_s2: float,
) -> tuple[float, ...]:
    """Apply per-axis acceleration limits to a six-axis velocity command."""

    if len(previous) != 6 or len(target) != 6:
        raise ValueError("previous and target commands must each contain six values")
    if not math.isfinite(dt_s) or dt_s < 0.0:
        raise ValueError("dt_s must be finite and non-negative")
    output: list[float] = []
    for index, (old_raw, new_raw) in enumerate(zip(previous, target)):
        old, new = float(old_raw), float(new_raw)
        if not math.isfinite(old) or not math.isfinite(new):
            raise ValueError("velocity commands must be finite")
        accel = linear_accel_mm_s2 if index < 3 else angular_accel_rad_s2
        max_delta = float(accel) * dt_s
        delta = max(-max_delta, min(max_delta, new - old))
        output.append(old + delta)
    return tuple(output)


def apply_workspace_guard(
    position_mm: Sequence[float],
    velocity: Sequence[float],
    workspace: Workspace,
    horizon_s: float = 0.0,
) -> tuple[float, ...]:
    """Prevent motion farther outside the configured Cartesian workspace.

    A component that points back into the workspace is always retained.  When the
    arm is inside the workspace and ``horizon_s`` is positive, a component is
    clipped so the finite-duration command cannot cross the corresponding bound.
    Rotational components are not modified.
    """

    if len(velocity) != 6:
        raise ValueError("workspace guard needs a six-axis command")
    if not workspace.enabled:
        return tuple(float(value) for value in velocity)
    if len(position_mm) < 3:
        raise ValueError("workspace guard needs an XYZ position")
    if not math.isfinite(horizon_s) or horizon_s < 0.0:
        raise ValueError("horizon_s must be finite and non-negative")
    result = [float(value) for value in velocity]
    for index, ((lower, upper), position_raw) in enumerate(
        zip(workspace.bounds, position_mm[:3])
    ):
        position = float(position_raw)
        if not math.isfinite(position):
            raise ValueError("workspace position must be finite")
        speed = result[index]
        if not math.isfinite(speed):
            raise ValueError("velocity command must be finite")
        if position <= lower and speed < 0.0:
            result[index] = 0.0
        elif position >= upper and speed > 0.0:
            result[index] = 0.0
        elif lower < position < upper and horizon_s > 0.0:
            minimum_speed = (lower - position) / horizon_s
            maximum_speed = (upper - position) / horizon_s
            result[index] = max(minimum_speed, min(maximum_speed, speed))
    return tuple(result)


@dataclass
class SafetyController:
    """Stateful command shaper with a fail-closed packet watchdog."""

    config: BridgeConfig
    desired: tuple[float, ...] = ZERO_COMMAND
    current: tuple[float, ...] = ZERO_COMMAND
    last_packet_time: float | None = None
    last_step_time: float | None = None

    def ingest(self, axes: Sequence[float], now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("packet timestamp must be finite")
        sanitized = sanitize_axes(axes, self.config.deadband)
        self.desired = scale_axes(
            sanitized,
            self.config.linear_speed_mm_s,
            self.config.angular_speed_rad_s,
        )
        self.last_packet_time = now

    def watchdog_expired(self, now: float) -> bool:
        return (
            self.last_packet_time is None
            or now - self.last_packet_time > self.config.watchdog_s
            or now < self.last_packet_time
        )

    def force_stop(self) -> None:
        """Immediately discard queued controller input and command zero."""

        self.desired = ZERO_COMMAND
        self.current = ZERO_COMMAND
        self.last_packet_time = None

    def step(
        self, now: float, position_mm: Sequence[float] | None
    ) -> tuple[float, ...]:
        if not math.isfinite(now):
            raise ValueError("step timestamp must be finite")
        default_dt = 1.0 / self.config.publish_hz
        if self.last_step_time is None or now < self.last_step_time:
            dt_s = default_dt
        else:
            # A long scheduler stall must not defeat acceleration limiting.
            dt_s = min(now - self.last_step_time, self.config.watchdog_s)
        self.last_step_time = now

        # A watchdog stop bypasses slew limiting: stale input must stop now.
        if self.watchdog_expired(now):
            self.desired = ZERO_COMMAND
            self.current = ZERO_COMMAND
            return ZERO_COMMAND
        # Workspace safety is fail closed when no trustworthy position exists.
        if self.config.workspace.enabled and position_mm is None:
            self.current = ZERO_COMMAND
            return ZERO_COMMAND

        candidate = slew_limit(
            self.current,
            self.desired,
            dt_s,
            self.config.linear_accel_mm_s2,
            self.config.angular_accel_rad_s2,
        )
        if self.config.workspace.enabled:
            candidate = apply_workspace_guard(
                position_mm or (),
                candidate,
                self.config.workspace,
                self.config.command_duration_s,
            )
        self.current = candidate
        return candidate


@dataclass
class PeerLock:
    """Locks command input to one Quest IP address."""

    configured_ip: str | None = None
    locked_ip: str | None = None

    def __post_init__(self) -> None:
        if self.configured_ip:
            self.locked_ip = self.configured_ip

    def accepts(self, ip: str) -> bool:
        if self.locked_ip is None:
            self.locked_ip = ip
            return True
        return ip == self.locked_ip


@dataclass
class GripperToggle:
    """Rising-edge toggle state for the historical Quest trigger behavior."""

    threshold: float = 0.5
    is_closed: bool = False
    previous_value: float = 0.0

    def observe(self, value: float | None) -> None:
        """Track trigger position without allowing a gripper action."""

        if value is None:
            return
        if not math.isfinite(value):
            raise ValueError("gripper input must be finite")
        self.previous_value = value

    def update(self, value: float | None) -> bool | None:
        """Return the new closed state on a rising edge, otherwise ``None``."""

        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("gripper input must be finite")
        toggled: bool | None = None
        if value > self.threshold and self.previous_value <= self.threshold:
            self.is_closed = not self.is_closed
            toggled = self.is_closed
        self.previous_value = value
        return toggled


@dataclass
class CollectionLease:
    """Fail-closed permission gate for motion during an active recording.

    The collector refreshes a monotonic timestamp only after the robot and both
    cameras are ready.  A missing/stale lease pauses Cartesian and gripper
    commands while the bridge continues receiving Quest packets and feedback.
    A new collector must also present neutral controls for a few packets before
    motion is armed.
    """

    path: Path | None
    timeout_s: float
    neutral_packets_required: int
    gripper_threshold: float
    armed: bool = False
    collector_pid: int | None = None
    neutral_packets: int = 0

    def _reset(self) -> None:
        self.armed = False
        self.collector_pid = None
        self.neutral_packets = 0

    def _fresh_collector(self, now: float) -> int | None:
        if self.path is None:
            return os.getpid()
        try:
            payload = json.loads(self.path.read_text())
            collector_pid = int(payload["pid"])
            heartbeat_s = int(payload["monotonic_ns"]) / 1e9
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        age_s = now - heartbeat_s
        if collector_pid <= 0 or age_s < 0.0 or age_s > self.timeout_s:
            return None
        return collector_pid

    def observe(
        self,
        axes: Sequence[float],
        gripper_value: float | None,
        now: float,
    ) -> bool:
        collector_pid = self._fresh_collector(now)
        if collector_pid is None:
            self._reset()
            return False
        if self.collector_pid != collector_pid:
            self.armed = False
            self.collector_pid = collector_pid
            self.neutral_packets = 0
        if self.armed:
            return True
        neutral = all(float(value) == 0.0 for value in axes) and (
            gripper_value is None or float(gripper_value) <= self.gripper_threshold
        )
        self.neutral_packets = self.neutral_packets + 1 if neutral else 0
        if self.neutral_packets >= self.neutral_packets_required:
            self.armed = True
        return self.armed

    def enabled(self, now: float) -> bool:
        collector_pid = self._fresh_collector(now)
        if collector_pid is None or collector_pid != self.collector_pid:
            self._reset()
            return False
        return self.armed


def populate_velocity_request(request: Any, speeds: Sequence[float], duration_s: float) -> Any:
    """Populate an xarm_msgs/MoveVelocity request (kept generic for unit tests)."""

    if len(speeds) != 6 or not all(math.isfinite(float(v)) for v in speeds):
        raise ValueError("MoveVelocity requires six finite speeds")
    request.speeds = [float(value) for value in speeds]
    request.is_sync = False
    request.is_tool_coord = False
    request.duration = float(duration_s)
    return request


def pack_feedback(position_mm: Sequence[float], gripper_closed: bool) -> bytes:
    """Encode XYZ metres and binary gripper state for Quest UDP port 5006."""

    if len(position_mm) < 3 or not all(
        math.isfinite(float(value)) for value in position_mm[:3]
    ):
        raise ValueError("feedback requires three finite position values")
    return struct.pack(
        "<ffff",
        float(position_mm[0]) / 1000.0,
        float(position_mm[1]) / 1000.0,
        float(position_mm[2]) / 1000.0,
        1.0 if gripper_closed else 0.0,
    )


def issue_safe_stop(
    send_velocity: Callable[[Sequence[float]], Any],
    enter_stop: Callable[[], Any],
    repeats: int = 3,
) -> None:
    """Issue repeated zero commands followed by the xArm STOP state."""

    if repeats < 1:
        raise ValueError("safe stop requires at least one zero command")
    first_error: BaseException | None = None
    for _ in range(repeats):
        try:
            send_velocity(ZERO_COMMAND)
        except BaseException as exc:  # Continue trying every independent stop action.
            if first_error is None:
                first_error = exc
    try:
        enter_stop()
    except BaseException as exc:
        if first_error is None:
            first_error = exc
    if first_error is not None:
        raise first_error


class _RateLimitedLog:
    def __init__(self, interval_s: float = 2.0) -> None:
        self.interval_s = interval_s
        self._last: dict[str, float] = {}

    def warning(self, key: str, message: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last.get(key, -math.inf) >= self.interval_s:
            self._last[key] = now
            LOGGER.warning(message, *args)


class _RosVelocityPublisher:
    """Bounded asynchronous publisher for the xArm velocity service."""

    def __init__(self, client: Any, request_type: Any, duration_s: float) -> None:
        self.client = client
        self.request_type = request_type
        self.duration_s = duration_s
        self.pending: deque[Any] = deque()
        self.max_pending = 16

    def _reap(self) -> None:
        while self.pending and self.pending[0].done():
            future = self.pending.popleft()
            error = future.exception()
            if error is not None:
                raise RuntimeError(f"velocity service call failed: {error}") from error
            response = future.result()
            if response is not None and getattr(response, "ret", 0) != 0:
                raise RuntimeError(
                    "velocity service rejected command: "
                    f"ret={getattr(response, 'ret', None)} "
                    f"message={getattr(response, 'message', '')!r}"
                )

    def send(self, speeds: Sequence[float]) -> Any:
        self._reap()
        if len(self.pending) >= self.max_pending:
            raise RuntimeError("velocity service has 16 outstanding calls")
        request = populate_velocity_request(
            self.request_type.Request(), speeds, self.duration_s
        )
        future = self.client.call_async(request)
        self.pending.append(future)
        return future


def _call_service_sync(
    rclpy: Any,
    node: Any,
    client: Any,
    request: Any,
    name: str,
    timeout_s: float,
) -> Any:
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise RuntimeError(f"required ROS service {name} is unavailable")
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise RuntimeError(f"ROS service {name} timed out after {timeout_s:.1f}s")
    error = future.exception()
    if error is not None:
        raise RuntimeError(f"ROS service {name} failed: {error}") from error
    response = future.result()
    if response is not None and getattr(response, "ret", 0) != 0:
        raise RuntimeError(
            f"ROS service {name} returned ret={getattr(response, 'ret', None)}: "
            f"{getattr(response, 'message', '')}"
        )
    return response


def _sdk_code(result: Any, operation: str) -> None:
    code = result[0] if isinstance(result, tuple) else result
    if code not in (None, 0):
        raise RuntimeError(f"xArm SDK {operation} failed with code {code}")


def require_timed_velocity_firmware(arm: Any) -> tuple[int, int, int]:
    """Reject firmware that cannot enforce MoveVelocity.duration."""

    raw = getattr(arm, "version_number", None)
    try:
        version = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("could not determine xArm firmware version") from exc
    if len(version) != 3:
        raise RuntimeError(f"invalid xArm firmware version: {raw!r}")
    if version < MIN_TIMED_VELOCITY_FIRMWARE:
        rendered = ".".join(str(value) for value in version)
        raise RuntimeError(
            f"xArm firmware {rendered} is unsafe for timed velocity control; "
            "version 1.8.0 or newer is required"
        )
    return version  # type: ignore[return-value]


def _read_position(arm: Any) -> tuple[float, ...]:
    result = arm.get_position()
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(f"xArm SDK returned malformed position result: {result!r}")
    code, position = result[0], result[1]
    if code != 0 or position is None or len(position) < 3:
        raise RuntimeError(f"xArm SDK get_position failed with code {code}")
    values = tuple(float(value) for value in position)
    if not all(math.isfinite(value) for value in values[:3]):
        raise RuntimeError("xArm SDK returned non-finite position")
    return values


def _configure_gripper(arm: Any, config: BridgeConfig) -> None:
    """Configure the standard xArm gripper across SDK variants.

    xarm-python-sdk 1.17.3 has no independent ``set_gripper_force`` API for
    the standard 0-850 pulse gripper.  Some historical/custom SDKs expose it,
    so apply force only when that exact method is available.
    """

    _sdk_code(arm.set_gripper_enable(True), "set_gripper_enable")
    _sdk_code(arm.set_gripper_mode(0), "set_gripper_mode")
    _sdk_code(arm.set_gripper_speed(config.gripper_speed), "set_gripper_speed")
    force_setter = getattr(arm, "set_gripper_force", None)
    if callable(force_setter):
        _sdk_code(force_setter(config.gripper_force), "set_gripper_force")
    else:
        LOGGER.info(
            "xArm SDK has no set_gripper_force for the standard gripper; "
            "continuing with position and speed control"
        )


def _create_arm(config: BridgeConfig) -> tuple[Any, bool]:
    try:
        from xarm.wrapper import XArmAPI
    except ImportError as exc:
        raise RuntimeError(
            "xArm Python SDK is required by the bridge for workspace safety and gripper control"
        ) from exc

    arm = XArmAPI(config.robot_ip)
    if hasattr(arm, "connected") and not arm.connected:
        raise RuntimeError(f"could not connect xArm SDK to {config.robot_ip}")
    try:
        require_timed_velocity_firmware(arm)
        _sdk_code(arm.clean_error(), "clean_error")
        _configure_gripper(arm, config)
        initial_closed = False
        try:
            result = arm.get_gripper_position()
            if isinstance(result, tuple) and len(result) >= 2 and result[0] == 0:
                initial_closed = float(result[1]) < (
                    config.gripper_open_position + config.gripper_closed_position
                ) / 2.0
        except Exception:
            LOGGER.warning("Could not read initial gripper position; assuming open")
        return arm, initial_closed
    except Exception:
        try:
            arm.disconnect()
        finally:
            raise


def _write_ready_file(path: Path, **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "pid": os.getpid(),
        "ready_at_monotonic_s": time.monotonic(),
        **metadata,
    }
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)


def _send_velocity_sync(
    rclpy: Any,
    node: Any,
    client: Any,
    request_type: Any,
    speeds: Sequence[float],
    duration_s: float,
    timeout_s: float,
    name: str,
) -> Any:
    request = populate_velocity_request(request_type.Request(), speeds, duration_s)
    return _call_service_sync(rclpy, node, client, request, name, timeout_s)


def _drain_udp(
    sock: socket.socket,
    peer: PeerLock,
    controller: SafetyController,
    gripper: GripperToggle,
    arm: Any,
    now: float,
    config: BridgeConfig,
    warning_log: _RateLimitedLog,
    on_first_valid_packet: Callable[[str, int, int], None] | None = None,
    collection_lease: CollectionLease | None = None,
) -> None:
    # Bound work per control tick so a datagram flood cannot starve the 50 Hz loop.
    for _ in range(64):
        try:
            data, address = sock.recvfrom(1024)
        except BlockingIOError:
            return
        try:
            packet = parse_packet(data)
            sanitized = sanitize_axes(packet.axes, config.deadband)
        except (PacketError, ValueError) as exc:
            warning_log.warning("invalid_packet", "Ignoring invalid Quest packet: %s", exc)
            continue
        if not peer.accepts(address[0]):
            warning_log.warning(
                "wrong_peer",
                "Ignoring Quest packet from %s; bridge is locked to %s",
                address[0],
                peer.locked_ip,
            )
            continue
        if not getattr(peer, "_announced", False):
            if on_first_valid_packet is not None:
                on_first_valid_packet(address[0], int(address[1]), len(data))
            LOGGER.info(
                "Quest peer locked to %s; feedback destination is %s:%d",
                peer.locked_ip,
                peer.locked_ip,
                config.feedback_port,
            )
            setattr(peer, "_announced", True)
        control_enabled = (
            collection_lease.observe(sanitized, packet.gripper, now)
            if collection_lease is not None
            else True
        )
        if control_enabled:
            controller.ingest(sanitized, now)
            new_closed = gripper.update(packet.gripper)
        else:
            controller.force_stop()
            gripper.observe(packet.gripper)
            new_closed = None
        if new_closed is not None:
            target = (
                config.gripper_closed_position
                if new_closed
                else config.gripper_open_position
            )
            result = arm.set_gripper_position(target, wait=False)
            code = result[0] if isinstance(result, tuple) else result
            if code not in (None, 0):
                warning_log.warning(
                    "gripper_error", "xArm gripper command failed with code %s", code
                )


def _create_clean_error_call(
    node: Any, prefix: str, call_type: Any
) -> tuple[Any, Any]:
    """Create the xArm ``clean_error`` client with its empty Call request.

    Unlike ``set_mode`` and ``set_state``, ``clean_error`` uses
    ``xarm_msgs/srv/Call`` rather than ``SetInt16`` in xarm_ros2.
    """

    return (
        node.create_client(call_type, f"{prefix}/clean_error"),
        call_type.Request(),
    )


def run_bridge(
    config: Mapping[str, Any] | BridgeConfig,
    *,
    stop_event: threading.Event | None = None,
    ready_file: Path | None = None,
    quest_ready_file: Path | None = None,
    collection_lease_file: Path | None = None,
    teleop_armed_file: Path | None = None,
) -> None:
    """Run the real UF850 bridge until stopped or an unrecoverable error occurs."""

    cfg = BridgeConfig.from_mapping(config)
    stop_event = stop_event or threading.Event()
    if quest_ready_file is not None:
        quest_ready_file.unlink(missing_ok=True)
    if teleop_armed_file is not None:
        teleop_armed_file.unlink(missing_ok=True)

    # Delayed imports keep unit tests and `vrctl doctor` usable without ROS sourced.
    try:
        import rclpy
        from xarm_msgs.srv import Call, MoveVelocity, SetInt16, SetInt16ById
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python and xarm_msgs are required; source the ROS workspace first"
        ) from exc

    arm: Any = None
    sock: socket.socket | None = None
    node: Any = None
    velocity: _RosVelocityPublisher | None = None
    velocity_client: Any = None
    state_client: Any = None
    ros_initialized = False
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        LOGGER.info("Received signal %s; requesting safe shutdown", signum)
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)

    try:
        arm, initial_closed = _create_arm(cfg)
        position = _read_position(arm)
        position_time = time.monotonic()

        rclpy.init(args=None)
        ros_initialized = True
        node = rclpy.create_node("uf850_vr_teleop_bridge")
        prefix = cfg.ros_namespace
        velocity_client = node.create_client(
            MoveVelocity, f"{prefix}/vc_set_cartesian_velocity"
        )
        clean_error_client, clean_error_request = _create_clean_error_call(
            node, prefix, Call
        )
        motion_enable_client = node.create_client(
            SetInt16ById, f"{prefix}/motion_enable"
        )
        mode_client = node.create_client(SetInt16, f"{prefix}/set_mode")
        state_client = node.create_client(SetInt16, f"{prefix}/set_state")

        _call_service_sync(
            rclpy,
            node,
            clean_error_client,
            clean_error_request,
            f"{prefix}/clean_error",
            cfg.service_timeout_s,
        )
        _call_service_sync(
            rclpy,
            node,
            motion_enable_client,
            SetInt16ById.Request(id=8, data=1),
            f"{prefix}/motion_enable",
            cfg.service_timeout_s,
        )
        _call_service_sync(
            rclpy,
            node,
            mode_client,
            SetInt16.Request(data=5),
            f"{prefix}/set_mode",
            cfg.service_timeout_s,
        )
        _call_service_sync(
            rclpy,
            node,
            state_client,
            SetInt16.Request(data=0),
            f"{prefix}/set_state",
            cfg.service_timeout_s,
        )
        velocity = _RosVelocityPublisher(
            velocity_client, MoveVelocity, cfg.command_duration_s
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((cfg.bind_host, cfg.listen_port))
        sock.setblocking(False)

        if threading.current_thread() is threading.main_thread():
            for sig in previous_handlers:
                signal.signal(sig, request_stop)

        controller = SafetyController(cfg)
        collection_lease = (
            CollectionLease(
                collection_lease_file,
                cfg.collection_lease_timeout_s,
                cfg.neutral_packets_to_arm,
                cfg.gripper_threshold,
            )
            if collection_lease_file is not None
            else None
        )
        teleop_armed = collection_lease is None
        peer = PeerLock(cfg.quest_ip)
        gripper = GripperToggle(cfg.gripper_threshold, initial_closed)
        warning_log = _RateLimitedLog()
        period_s = 1.0 / cfg.publish_hz
        position_period_s = 1.0 / cfg.position_poll_hz
        next_tick = time.monotonic()
        last_position_poll = -math.inf
        last_feedback = -math.inf
        state_failures = 0

        LOGGER.info(
            "Bridge ready: UDP %s:%d, ROS namespace %s, %.1f Hz, duration %.3fs",
            cfg.bind_host,
            cfg.listen_port,
            cfg.ros_namespace,
            cfg.publish_hz,
            cfg.command_duration_s,
        )
        if ready_file is not None:
            _write_ready_file(ready_file)

        def mark_quest_ready(ip: str, source_port: int, packet_size: int) -> None:
            if quest_ready_file is not None:
                _write_ready_file(
                    quest_ready_file,
                    quest_ip=ip,
                    source_port=source_port,
                    packet_size=packet_size,
                )

        while rclpy.ok() and not stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                rclpy.spin_once(node, timeout_sec=min(0.002, next_tick - now))
                remaining = next_tick - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(0.001, remaining))
                continue

            _drain_udp(
                sock,
                peer,
                controller,
                gripper,
                arm,
                now,
                cfg,
                warning_log,
                mark_quest_ready,
                collection_lease,
            )

            lease_enabled = (
                collection_lease.enabled(now)
                if collection_lease is not None
                else True
            )
            if collection_lease is not None and lease_enabled != teleop_armed:
                teleop_armed = lease_enabled
                if teleop_armed:
                    LOGGER.info(
                        "TELEOP ARMED for collector pid %s",
                        collection_lease.collector_pid,
                    )
                    if teleop_armed_file is not None:
                        _write_ready_file(
                            teleop_armed_file,
                            collector_pid=collection_lease.collector_pid,
                        )
                else:
                    LOGGER.info("TELEOP PAUSED; sending zero velocity")
                    controller.force_stop()
                    if teleop_armed_file is not None:
                        teleop_armed_file.unlink(missing_ok=True)

            if now - last_position_poll >= position_period_s:
                last_position_poll = now
                try:
                    position = _read_position(arm)
                    position_time = time.monotonic()
                    state_failures = 0
                except Exception as exc:
                    state_failures += 1
                    warning_log.warning(
                        "position_error",
                        "xArm state read failed (%d/%d): %s",
                        state_failures,
                        cfg.state_failure_limit,
                        exc,
                    )
                    if state_failures >= cfg.state_failure_limit:
                        raise RuntimeError(
                            f"xArm state failed {state_failures} consecutive times"
                        ) from exc

            safe_position: Sequence[float] | None = position
            if now - position_time > cfg.position_stale_s:
                safe_position = None
            command = (
                controller.step(now, safe_position)
                if lease_enabled
                else ZERO_COMMAND
            )
            velocity.send(command)

            if (
                peer.locked_ip
                and now - last_feedback >= cfg.feedback_interval_s
                and safe_position is not None
            ):
                last_feedback = now
                try:
                    sock.sendto(
                        pack_feedback(safe_position, gripper.is_closed),
                        (peer.locked_ip, cfg.feedback_port),
                    )
                except OSError as exc:
                    warning_log.warning(
                        "feedback_error", "Quest feedback send failed: %s", exc
                    )

            rclpy.spin_once(node, timeout_sec=0.0)
            next_tick += period_s
            if now - next_tick >= period_s:
                warning_log.warning(
                    "rate_overrun", "Bridge control loop missed its %.1f Hz deadline", cfg.publish_hz
                )
                next_tick = now + period_s
    finally:
        if velocity_client is not None and node is not None and state_client is not None:
            LOGGER.info("Sending repeated zero velocity and entering xArm STOP state")

            def send_zero(command: Sequence[float]) -> None:
                _send_velocity_sync(
                    rclpy,
                    node,
                    velocity_client,
                    MoveVelocity,
                    command,
                    cfg.command_duration_s,
                    min(cfg.service_timeout_s, 1.0),
                    f"{cfg.ros_namespace}/vc_set_cartesian_velocity(zero)",
                )

            def enter_stop() -> None:
                _call_service_sync(
                    rclpy,
                    node,
                    state_client,
                    SetInt16.Request(data=4),
                    f"{cfg.ros_namespace}/set_state(STOP)",
                    min(cfg.service_timeout_s, 2.0),
                )

            try:
                issue_safe_stop(
                    send_zero, enter_stop, repeats=cfg.shutdown_zero_repeats
                )
            except Exception:
                LOGGER.exception("Safe shutdown encountered an error")
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        if quest_ready_file is not None:
            quest_ready_file.unlink(missing_ok=True)
        if teleop_armed_file is not None:
            teleop_armed_file.unlink(missing_ok=True)
        if sock is not None:
            sock.close()
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                LOGGER.exception("xArm SDK disconnect failed")
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                LOGGER.exception("ROS node cleanup failed")
        if ros_initialized:
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                LOGGER.exception("ROS shutdown failed")
        for sig, old_handler in previous_handlers.items():
            try:
                signal.signal(sig, old_handler)
            except (ValueError, OSError):
                pass


def load_config(path: str | Path) -> BridgeConfig:
    """Load bridge settings from TOML using only the Python standard library."""

    try:
        import tomllib
    except ImportError as exc:  # pragma: no cover - Python 3.11+ is used on the robot.
        raise RuntimeError("Python 3.11 or newer is required to read config.toml") from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        return BridgeConfig.from_mapping(tomllib.load(stream))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safe Quest-to-UF850 ROS 2 bridge")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config.toml",
        help="TOML configuration path (default: project config.toml)",
    )
    parser.add_argument(
        "--ready-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quest-ready-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--control-lock-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--collection-lease-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--teleop-armed-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    lock_path = args.control_lock_file
    if lock_path is None:
        lock_path = args.config.expanduser().resolve().parent / ".runtime" / "arm_control.lock"
    try:
        with arm_control_lock(lock_path):
            run_bridge(
                load_config(args.config),
                ready_file=args.ready_file,
                quest_ready_file=args.quest_ready_file,
                collection_lease_file=args.collection_lease_file,
                teleop_armed_file=args.teleop_armed_file,
            )
    except KeyboardInterrupt:
        # SIGINT normally reaches the stop event, but retain a clean CLI fallback.
        return 130
    except Exception:
        LOGGER.exception("Bridge stopped because of an unrecoverable error")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
