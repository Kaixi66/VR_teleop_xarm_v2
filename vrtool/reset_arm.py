"""Standalone, guarded UF850 joint-space reset.

The target pose comes from the proven legacy/OFT data-collection reset.  This
module deliberately imports the xArm SDK only for a real reset so ``--dry-run``
and unit tests cannot connect to hardware by accident.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .config import DEFAULT_CONFIG_PATH, ConfigError, load_config
from .manager import ManagerError, stack_status


class ResetError(RuntimeError):
    """Raised when reset cannot be performed or verified safely."""


@dataclass(frozen=True)
class ResetSettings:
    robot_ip: str
    joint_angles_deg: tuple[float, float, float, float, float, float]
    speed_deg_s: float
    acceleration_deg_s2: float
    timeout_s: float
    pause_s: float
    countdown_s: float
    tolerance_deg: float
    open_gripper: bool
    gripper_open_position: int
    gripper_speed: int
    runtime_dir: Path

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ResetSettings":
        robot = _mapping(config.get("robot"), "robot")
        reset = _mapping(config.get("reset"), "reset")
        paths = _mapping(config.get("paths"), "paths")

        robot_ip = robot.get("ip")
        if not isinstance(robot_ip, str) or not robot_ip.strip():
            raise ResetError("robot.ip must be a non-empty string")

        raw_angles = reset.get("joint_angles_deg")
        if not isinstance(raw_angles, list) or len(raw_angles) != 6:
            raise ResetError("reset.joint_angles_deg must contain exactly 6 numbers")
        angles = tuple(
            _number(value, f"reset.joint_angles_deg[{index}]", minimum=-360.0, maximum=360.0)
            for index, value in enumerate(raw_angles)
        )

        open_gripper = reset.get("open_gripper")
        if not isinstance(open_gripper, bool):
            raise ResetError("reset.open_gripper must be true or false")
        gripper_position = _integer(
            reset.get("gripper_open_position"),
            "reset.gripper_open_position",
            minimum=0,
            maximum=850,
        )
        gripper_speed = _integer(
            reset.get("gripper_speed"),
            "reset.gripper_speed",
            minimum=1,
            maximum=5000,
        )

        runtime_value = paths.get("runtime_dir")
        if not isinstance(runtime_value, (str, os.PathLike)):
            raise ResetError("paths.runtime_dir must be a path string")

        return cls(
            robot_ip=robot_ip.strip(),
            joint_angles_deg=angles,  # type: ignore[arg-type]
            speed_deg_s=_number(
                reset.get("speed_deg_s"),
                "reset.speed_deg_s",
                minimum=0.0,
                maximum=30.0,
                exclusive_min=True,
            ),
            acceleration_deg_s2=_number(
                reset.get("acceleration_deg_s2"),
                "reset.acceleration_deg_s2",
                minimum=0.0,
                maximum=500.0,
                exclusive_min=True,
            ),
            timeout_s=_number(
                reset.get("timeout_s"),
                "reset.timeout_s",
                minimum=1.0,
                maximum=120.0,
            ),
            pause_s=_number(
                reset.get("pause_s"), "reset.pause_s", minimum=0.0, maximum=30.0
            ),
            countdown_s=_number(
                reset.get("countdown_s"),
                "reset.countdown_s",
                minimum=0.0,
                maximum=30.0,
            ),
            tolerance_deg=_number(
                reset.get("tolerance_deg"),
                "reset.tolerance_deg",
                minimum=0.0,
                maximum=10.0,
                exclusive_min=True,
            ),
            open_gripper=open_gripper,
            gripper_open_position=gripper_position,
            gripper_speed=gripper_speed,
            runtime_dir=Path(runtime_value).expanduser().absolute(),
        )


@dataclass(frozen=True)
class ResetResult:
    initial_angles_deg: tuple[float, ...]
    final_angles_deg: tuple[float, ...]
    max_error_deg: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResetError(f"{label} must be a TOML table")
    return value


def _number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
    exclusive_min: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResetError(f"{label} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ResetError(f"{label} must be finite")
    if converted < minimum or (exclusive_min and converted == minimum):
        operator = ">" if exclusive_min else ">="
        raise ResetError(f"{label} must be {operator} {minimum:g}")
    if converted > maximum:
        raise ResetError(f"{label} must be <= {maximum:g}")
    return converted


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResetError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ResetError(f"{label} must be between {minimum} and {maximum}")
    return value


def _check_code(result: Any, operation: str) -> None:
    if isinstance(result, bool) or not isinstance(result, int):
        raise ResetError(f"{operation} returned malformed result: {result!r}")
    if result != 0:
        raise ResetError(f"{operation} failed with xArm code {result}")


def _checked_pair(result: Any, operation: str) -> Any:
    if not isinstance(result, (tuple, list)) or len(result) < 2:
        raise ResetError(f"{operation} returned malformed result: {result!r}")
    _check_code(result[0], operation)
    return result[1]


def _angles(value: Any, operation: str) -> tuple[float, ...]:
    try:
        converted = tuple(float(item) for item in value[:6])
    except (TypeError, ValueError, IndexError) as exc:
        raise ResetError(f"{operation} returned invalid joint angles: {value!r}") from exc
    if len(converted) != 6 or not all(math.isfinite(item) for item in converted):
        raise ResetError(f"{operation} must return 6 finite joint angles")
    return converted


def _get_angles(arm: Any) -> tuple[float, ...]:
    return _angles(
        _checked_pair(arm.get_servo_angle(is_radian=False), "get_servo_angle"),
        "get_servo_angle",
    )


def _get_health(arm: Any) -> tuple[int, int, int]:
    state = _checked_pair(arm.get_state(), "get_state")
    if isinstance(state, bool) or not isinstance(state, int):
        raise ResetError(f"get_state returned invalid state: {state!r}")
    err_warn = _checked_pair(arm.get_err_warn_code(), "get_err_warn_code")
    if not isinstance(err_warn, (tuple, list)) or len(err_warn) != 2:
        raise ResetError(f"get_err_warn_code returned invalid values: {err_warn!r}")
    error_code, warning_code = err_warn
    if any(isinstance(value, bool) or not isinstance(value, int) for value in err_warn):
        raise ResetError(f"get_err_warn_code returned invalid values: {err_warn!r}")
    return state, error_code, warning_code


def _require_healthy_idle(arm: Any, *, stage: str) -> int:
    state, error_code, warning_code = _get_health(arm)
    if error_code != 0 or warning_code != 0:
        raise ResetError(
            f"controller is not healthy {stage}: error={error_code}, warning={warning_code}; "
            "inspect the robot before resetting instead of clearing errors automatically"
        )
    if state not in (0, 2, 4):
        raise ResetError(
            f"controller state {state} is not safe to reset {stage}; expected ready/sleep/STOP"
        )
    return state


def _require_uf850(arm: Any) -> None:
    axis = getattr(arm, "axis", None)
    if axis is None:
        raise ResetError("xArm SDK did not report an axis count; refusing non-verified hardware")
    try:
        axis_count = int(axis)
    except (TypeError, ValueError) as exc:
        raise ResetError(f"xArm SDK returned an invalid axis count: {axis!r}") from exc
    if axis_count != 6:
        raise ResetError(f"expected a 6-axis UF850, SDK reports {axis} axes")
    is_850 = getattr(arm, "is_850", None)
    device_type = getattr(arm, "device_type", None)
    device_type_is_850 = (
        not isinstance(device_type, bool)
        and isinstance(device_type, int)
        and device_type == 12
    )
    if is_850 is not True and not device_type_is_850:
        raise ResetError(
            f"connected device is not verified as UF850 (axis={axis}, device_type={device_type!r})"
        )


def _wait_for_uf850_report(arm: Any, sleeper: Callable[[float], None]) -> None:
    """Allow the SDK rich-report thread up to two seconds to identify hardware."""

    last_error: ResetError | None = None
    for attempt in range(41):
        try:
            _require_uf850(arm)
            return
        except ResetError as exc:
            last_error = exc
        if attempt < 40:
            sleeper(0.05)
    assert last_error is not None
    raise last_error


def _check_joint_limit(arm: Any, target: Sequence[float]) -> None:
    checker = getattr(arm, "is_joint_limit", None)
    if not callable(checker):
        raise ResetError("xArm SDK does not provide is_joint_limit; refusing unchecked motion")
    limited = _checked_pair(checker(list(target), is_radian=False), "is_joint_limit")
    if not isinstance(limited, bool):
        raise ResetError(f"is_joint_limit returned invalid limit flag: {limited!r}")
    if limited:
        raise ResetError("configured reset target violates the controller joint limits")


def _countdown(seconds: float, sleeper: Callable[[float], None], output: Callable[[str], None]) -> None:
    remaining = int(math.ceil(seconds))
    for value in range(remaining, 0, -1):
        output(f"Reset motion starts in {value} second(s). Press Ctrl+C to cancel.")
        sleeper(1.0)


def execute_reset(
    arm: Any,
    settings: ResetSettings,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    output: Callable[[str], None] = print,
) -> ResetResult:
    """Execute and verify one reset using an already connected arm object."""

    if hasattr(arm, "connected") and not bool(arm.connected):
        raise ResetError("xArm SDK is not connected")
    _wait_for_uf850_report(arm, sleeper)
    _require_healthy_idle(arm, stage="before reset")

    target = settings.joint_angles_deg
    initial = _get_angles(arm)
    _check_joint_limit(arm, target)
    max_move = max(abs(current - desired) for current, desired in zip(initial, target))

    output("Current joints (deg): " + ", ".join(f"{value:.3f}" for value in initial))
    output("Target joints  (deg): " + ", ".join(f"{value:.3f}" for value in target))
    output(f"Largest joint change: {max_move:.3f} deg")
    if settings.open_gripper:
        output(
            f"WARNING: the gripper will open to {settings.gripper_open_position} before motion; "
            "a held object may drop."
        )
    _countdown(settings.countdown_s, sleeper, output)

    _check_code(arm.motion_enable(enable=True), "motion_enable")
    _check_code(arm.set_mode(0), "set_mode(0)")
    _check_code(arm.set_state(0), "set_state(0)")
    sleeper(0.5)

    if settings.open_gripper:
        _check_code(arm.set_gripper_enable(True), "set_gripper_enable")
        _check_code(arm.set_gripper_mode(0), "set_gripper_mode(0)")
        _check_code(arm.set_gripper_speed(settings.gripper_speed), "set_gripper_speed")
        _check_code(
            arm.set_gripper_position(
                settings.gripper_open_position,
                wait=True,
                timeout=settings.timeout_s,
            ),
            "set_gripper_position",
        )

    _check_code(
        arm.set_servo_angle(
            angle=list(target),
            speed=settings.speed_deg_s,
            mvacc=settings.acceleration_deg_s2,
            is_radian=False,
            wait=False,
        ),
        "set_servo_angle(reset)",
    )
    deadline = clock() + settings.timeout_s
    final = initial
    max_error = max_move
    while True:
        state, error_code, warning_code = _get_health(arm)
        if error_code != 0 or warning_code != 0:
            raise ResetError(
                f"controller fault during reset: error={error_code}, warning={warning_code}"
            )
        if state not in (0, 1, 2):
            raise ResetError(f"controller entered unsafe state {state} during reset")
        final = _get_angles(arm)
        errors = tuple(abs(actual - desired) for actual, desired in zip(final, target))
        max_error = max(errors)
        if max_error <= settings.tolerance_deg and state != 1:
            break
        now = clock()
        if now >= deadline:
            raise ResetError(
                f"reset timed out after {settings.timeout_s:.1f}s; "
                f"last max joint error was {max_error:.3f} deg"
            )
        sleeper(min(0.05, deadline - now))

    _require_healthy_idle(arm, stage="after reset")
    if settings.pause_s:
        sleeper(settings.pause_s)
    output("Final joints   (deg): " + ", ".join(f"{value:.3f}" for value in final))
    output(f"Reset pose verified; maximum joint error is {max_error:.3f} deg.")
    return ResetResult(initial, final, max_error)


def _safe_stop_and_disconnect(
    arm: Any, output: Callable[[str], None], *, disable_gripper: bool
) -> ResetError | None:
    failures: list[str] = []
    try:
        result = arm.set_state(4)
        if isinstance(result, bool) or not isinstance(result, int) or result != 0:
            failures.append(f"set_state(4) returned {result!r}")
    except BaseException as exc:
        failures.append(f"set_state(4) raised {exc}")
    gripper_disable_method = getattr(arm, "set_gripper_enable", None)
    if disable_gripper and callable(gripper_disable_method):
        try:
            result = gripper_disable_method(False)
            if isinstance(result, bool) or not isinstance(result, int) or result != 0:
                failures.append(f"set_gripper_enable(False) returned {result!r}")
        except BaseException as exc:
            failures.append(f"set_gripper_enable(False) raised {exc}")
    try:
        arm.disconnect()
    except BaseException as exc:
        failures.append(f"disconnect raised {exc}")
    if failures:
        return ResetError("cleanup failed: " + "; ".join(failures))
    output("Arm is in STOP state and disconnected.")
    return None


def run_hardware_reset(
    settings: ResetSettings,
    *,
    arm_factory: Callable[[str], Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    output: Callable[[str], None] = print,
) -> ResetResult:
    """Connect, reset, and guarantee STOP plus disconnect on every exit path."""

    if arm_factory is None:
        try:
            from xarm.wrapper import XArmAPI
        except ModuleNotFoundError as exc:
            raise ResetError("xarm-python-sdk is not installed") from exc
        arm_factory = lambda ip: XArmAPI(ip, is_radian=False, report_type="rich")

    arm: Any = None
    try:
        arm = arm_factory(settings.robot_ip)
        if hasattr(arm, "connected") and not bool(arm.connected):
            arm.connect()
        if hasattr(arm, "connected") and not bool(arm.connected):
            raise ResetError(f"failed to connect to xArm at {settings.robot_ip}")
        result = execute_reset(
            arm, settings, sleeper=sleeper, clock=clock, output=output
        )
    except BaseException as exc:
        if arm is not None:
            cleanup_error = _safe_stop_and_disconnect(
                arm, output, disable_gripper=settings.open_gripper
            )
            if cleanup_error is not None:
                output(f"WARNING: {cleanup_error}")
        if isinstance(exc, (KeyboardInterrupt, SystemExit, ResetError)):
            raise
        raise ResetError(f"xArm reset operation raised: {exc}") from exc

    cleanup_error = _safe_stop_and_disconnect(
        arm, output, disable_gripper=settings.open_gripper
    )
    if cleanup_error is not None:
        raise cleanup_error
    return result


KNOWN_CONTROLLER_MARKERS = (
    "vrtool.bridge",
    "bridge_ghsun.py",
    "inference_oft_xarm.py",
    "keyboard_control.py",
)


def find_external_controllers(proc_root: Path = Path("/proc")) -> list[tuple[int, str]]:
    """Find known local arm controllers which are outside vrctl state tracking."""

    matches: list[tuple[int, str]] = []
    own_pid = os.getpid()
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return matches
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        command = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        if command and any(marker in command for marker in KNOWN_CONTROLLER_MARKERS):
            matches.append((int(entry.name), command))
    return sorted(matches)


def ensure_exclusive_control(
    config: Mapping[str, Any],
    *,
    status_fn: Callable[[Mapping[str, Any]], list[tuple[str, bool, int, str]]] = stack_status,
    controller_scan: Callable[[], list[tuple[int, str]]] = find_external_controllers,
) -> None:
    live = [item for item in status_fn(config) if item[1]]
    if live:
        labels = ", ".join(f"{label}(pid={pid})" for label, _alive, pid, _log in live)
        raise ResetError(
            f"managed VR/ROS stack is still running: {labels}. Run './vrctl stop' first."
        )
    external = controller_scan()
    if external:
        summary = ", ".join(f"pid={pid} {command}" for pid, command in external)
        raise ResetError(f"another known arm controller is running: {summary}")


@contextmanager
def reset_lock(runtime_dir: Path) -> Iterator[None]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    arm_path = runtime_dir / "arm_control.lock"
    startup_path = runtime_dir / "stack_start.lock"
    with arm_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResetError("another bridge or reset process already owns arm control") from exc
        try:
            with startup_path.open("a+") as startup_handle:
                try:
                    fcntl.flock(
                        startup_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                except BlockingIOError as exc:
                    raise ResetError("a VR/ROS stack startup is already in progress") from exc
                try:
                    yield
                finally:
                    fcntl.flock(startup_handle.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install_signal_handlers() -> None:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, signal.default_int_handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move UF850 to the configured legacy/OFT reset joint pose"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the reset plan without importing xArm SDK or connecting to hardware",
    )
    return parser


def _print_plan(settings: ResetSettings, output: Callable[[str], None] = print) -> None:
    output("UF850 standalone reset")
    output("======================")
    output(f"Robot IP: {settings.robot_ip}")
    output("Target joints (deg): " + ", ".join(f"{v:.6f}" for v in settings.joint_angles_deg))
    output(f"Speed: {settings.speed_deg_s:g} deg/s")
    output(f"Acceleration: {settings.acceleration_deg_s2:g} deg/s^2")
    output(f"Timeout: {settings.timeout_s:g} s")
    if settings.open_gripper:
        output(
            f"Gripper: open to {settings.gripper_open_position} at speed "
            f"{settings.gripper_speed} before joint motion"
        )
    else:
        output("Gripper: unchanged")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    install_signal_handlers()
    try:
        raw_config = load_config(args.config)
        settings = ResetSettings.from_mapping(raw_config)
        _print_plan(settings)
        if args.dry_run:
            print("DRY RUN: no SDK import, robot connection, or motion was performed.")
            return 0
        with reset_lock(settings.runtime_dir):
            ensure_exclusive_control(raw_config)
            run_hardware_reset(settings)
        print("RESET COMPLETE")
        return 0
    except KeyboardInterrupt:
        print("RESET CANCELLED; STOP cleanup was requested.", file=sys.stderr)
        return 130
    except (ConfigError, ManagerError, ResetError, OSError) as exc:
        print(f"RESET FAILED: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "ResetError",
    "ResetResult",
    "ResetSettings",
    "ensure_exclusive_control",
    "execute_reset",
    "find_external_controllers",
    "install_signal_handlers",
    "main",
    "reset_lock",
    "run_hardware_reset",
]


if __name__ == "__main__":
    raise SystemExit(main())
