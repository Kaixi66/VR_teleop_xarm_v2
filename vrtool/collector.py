"""Reliable, backwards-compatible UF850 teleoperation data collection.

The module deliberately imports the xArm and RealSense SDKs only when real
hardware sources are started.  This keeps validation, unit tests, and mock
collection usable on machines without either SDK installed.
"""

from __future__ import annotations

import dataclasses
import argparse
import itertools
import json
import math
import os
import queue
import re
import signal
import shutil
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "2.0"
_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EPISODE_RE = re.compile(r"^episode_(\d+)$")
_IN_PROGRESS_RE = re.compile(r"^\.episode_(\d+)\.inprogress$")
_SENTINEL = object()


class CollectionError(RuntimeError):
    """A collection failure, optionally carrying the incomplete result."""

    def __init__(self, message: str, result: "CollectionResult | None" = None):
        super().__init__(message)
        self.result = result


class RobotReadError(CollectionError):
    """The robot SDK returned an error or malformed state."""


class StorageError(CollectionError):
    """An episode or step could not be persisted safely."""


@dataclass(frozen=True)
class CollectorConfig:
    """All collector settings after resolving ``config.toml`` defaults."""

    robot_ip: str = "192.168.1.230"
    data_root: Path = Path("data")
    # Real deployments must provide both serials in config.toml.  Keeping
    # placeholders here prevents a public source checkout from silently
    # targeting one particular pair of cameras.
    camera_serials: tuple[str, str] = ("WRIST_CAMERA_SERIAL", "EXTERNAL_CAMERA_SERIAL")
    camera_roles: tuple[str, str] = ("wrist", "external")
    hz: float = 10.0
    width: int = 1920
    height: int = 1080
    camera_fps: int = 30
    frame_buffer_size: int = 8
    warmup_frames: int = 30
    max_camera_skew_ms: float = 50.0
    max_frame_age_ms: float = 150.0
    save_queue_size: int = 16
    jpeg_quality: int = 95
    min_free_gb: float = 5.0
    gripper_move_threshold: float = 10.0
    consecutive_failure_limit: int = 10
    startup_timeout_s: float = 5.0
    writer_close_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser())
        if len(self.camera_serials) != 2:
            raise ValueError("exactly two camera serials are required")
        if len(self.camera_roles) != 2:
            raise ValueError("exactly two camera roles are required")
        if self.hz <= 0 or self.camera_fps <= 0:
            raise ValueError("recording hz and camera fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.frame_buffer_size < 2:
            raise ValueError("frame_buffer_size must be at least 2")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")
        if self.save_queue_size < 1:
            raise ValueError("save_queue_size must be positive")
        if self.max_camera_skew_ms < 0 or self.max_frame_age_ms <= 0:
            raise ValueError("camera timing limits must be non-negative")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.min_free_gb < 0:
            raise ValueError("min_free_gb must be non-negative")
        if self.consecutive_failure_limit < 1:
            raise ValueError("consecutive_failure_limit must be positive")
        if self.writer_close_timeout_s <= 0:
            raise ValueError("writer_close_timeout_s must be positive")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None
    ) -> "CollectorConfig":
        """Build from the project's ``[robot]``, ``[recording]`` and paths.

        Unknown keys are intentionally ignored so bridge/orchestration settings
        can live in the same TOML file.
        """

        robot = _mapping(raw.get("robot"))
        recording = _mapping(raw.get("recording"))
        paths = _mapping(raw.get("paths"))
        camera_rows = raw.get("cameras", ())
        if not isinstance(camera_rows, Sequence) or isinstance(camera_rows, (str, bytes)):
            raise ValueError("config 'cameras' must be an array of tables")

        serials: list[str] = []
        roles: list[str] = []
        for index, row_value in enumerate(camera_rows):
            row = _mapping(row_value)
            serial = row.get("serial", row.get("serial_number"))
            if serial is None:
                raise ValueError(f"cameras[{index}] is missing 'serial'")
            serials.append(str(serial))
            roles.append(str(row.get("role", f"cam_{index}")))
        if not serials:
            serials = list(cls.camera_serials)
            roles = list(cls.camera_roles)

        output_value = recording.get(
            "output_root", paths.get("data_root", paths.get("output_root", cls.data_root))
        )
        output_path = Path(output_value).expanduser()
        if not output_path.is_absolute() and base_dir is not None:
            output_path = Path(base_dir).expanduser() / output_path

        hz = float(recording.get("hz", cls.hz))
        failure_limit_value = recording.get("consecutive_failure_limit")
        if failure_limit_value is None:
            failure_timeout_s = float(recording.get("failure_timeout_s", 1.0))
            failure_limit_value = max(1, math.ceil(failure_timeout_s * hz))

        return cls(
            robot_ip=str(robot.get("ip", robot.get("robot_ip", cls.robot_ip))),
            data_root=output_path,
            camera_serials=tuple(serials),  # type: ignore[arg-type]
            camera_roles=tuple(roles),  # type: ignore[arg-type]
            hz=hz,
            width=int(recording.get("width", cls.width)),
            height=int(recording.get("height", cls.height)),
            camera_fps=int(recording.get("fps", cls.camera_fps)),
            frame_buffer_size=int(recording.get("frame_buffer_size", cls.frame_buffer_size)),
            warmup_frames=int(recording.get("warmup_frames", cls.warmup_frames)),
            max_camera_skew_ms=float(
                recording.get("max_camera_skew_ms", cls.max_camera_skew_ms)
            ),
            max_frame_age_ms=float(recording.get("max_frame_age_ms", cls.max_frame_age_ms)),
            save_queue_size=int(recording.get("save_queue_size", cls.save_queue_size)),
            jpeg_quality=int(recording.get("jpeg_quality", cls.jpeg_quality)),
            min_free_gb=float(recording.get("min_free_gb", cls.min_free_gb)),
            gripper_move_threshold=float(
                recording.get("gripper_move_threshold", cls.gripper_move_threshold)
            ),
            consecutive_failure_limit=int(failure_limit_value),
            startup_timeout_s=float(recording.get("startup_timeout_s", cls.startup_timeout_s)),
            writer_close_timeout_s=float(
                recording.get("writer_close_timeout_s", cls.writer_close_timeout_s)
            ),
        )

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> "CollectorConfig":
        path = Path(path).expanduser().resolve()
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:
                raise RuntimeError("Python 3.11+ or the 'tomli' package is required") from exc
        with path.open("rb") as handle:
            return cls.from_mapping(tomllib.load(handle), base_dir=path.parent)

    def snapshot(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value["data_root"] = str(self.data_root)
        value["camera_serials"] = list(self.camera_serials)
        value["camera_roles"] = list(self.camera_roles)
        return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class CameraFrame:
    camera_index: int
    image: Any
    host_monotonic_ns: int
    host_wall_time_ns: int
    device_timestamp_ms: float
    frame_number: int


@dataclass(frozen=True)
class FrameSelection:
    frames: tuple[CameraFrame, ...]
    skew_ms: float
    max_age_ms: float


@dataclass(frozen=True)
class RobotState:
    ee_pos: tuple[float, float, float, float, float, float]
    joint_pos: tuple[float, float, float, float, float, float]
    gripper_pos: float
    sample_monotonic_ns: int
    sample_time_unix_ns: int
    read_duration_ms: float


@dataclass(frozen=True)
class Sample:
    robot: RobotState
    frames: tuple[CameraFrame, CameraFrame]
    camera_skew_ms: float
    camera_max_age_ms: float


@dataclass(frozen=True)
class CollectionResult:
    dataset: str
    episode_index: int
    path: Path
    complete: bool
    termination_reason: str
    steps_written: int
    samples_captured: int
    error_counts: dict[str, int]


class FrameBuffer:
    """Small, thread-safe, immutable-snapshot camera frame buffer."""

    def __init__(self, maxlen: int):
        self._frames: deque[CameraFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: CameraFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def snapshot(self) -> tuple[CameraFrame, ...]:
        with self._lock:
            return tuple(self._frames)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


def select_synchronized_frames(
    frame_sets: Sequence[Sequence[CameraFrame]],
    target_monotonic_ns: int,
    *,
    max_skew_ms: float,
    max_age_ms: float,
) -> FrameSelection | None:
    """Choose the freshest frame combination satisfying age and skew limits."""

    if not frame_sets or any(not frames for frames in frame_sets):
        return None
    max_age_ns = int(max_age_ms * 1_000_000)
    max_skew_ns = int(max_skew_ms * 1_000_000)
    candidates: list[tuple[CameraFrame, ...]] = []
    for frames in frame_sets:
        fresh = tuple(
            frame
            for frame in frames
            if 0 <= target_monotonic_ns - frame.host_monotonic_ns <= max_age_ns
        )
        if not fresh:
            return None
        candidates.append(fresh)

    best: tuple[CameraFrame, ...] | None = None
    best_score: tuple[int, int, int] | None = None
    for combination in itertools.product(*candidates):
        times = [frame.host_monotonic_ns for frame in combination]
        skew_ns = max(times) - min(times)
        if skew_ns > max_skew_ns:
            continue
        ages = [target_monotonic_ns - stamp for stamp in times]
        score = (max(ages), skew_ns, sum(ages))
        if best_score is None or score < best_score:
            best = combination
            best_score = score

    if best is None:
        return None
    times = [frame.host_monotonic_ns for frame in best]
    return FrameSelection(
        frames=best,
        skew_ms=(max(times) - min(times)) / 1_000_000.0,
        max_age_ms=max(target_monotonic_ns - stamp for stamp in times) / 1_000_000.0,
    )


def wrapped_angle_delta(current: float, previous: float) -> float:
    """Shortest signed degree delta in [-180, 180)."""

    return (current - previous + 180.0) % 360.0 - 180.0


def build_step_data(
    previous: Sample,
    current: Sample,
    *,
    step: int,
    total_gripper_moves: int,
) -> dict[str, Any]:
    """Create the legacy observation/action schema plus v2 timing metadata."""

    delta_ee = [b - a for a, b in zip(previous.robot.ee_pos, current.robot.ee_pos)]
    delta_joint = [b - a for a, b in zip(previous.robot.joint_pos, current.robot.joint_pos)]
    dt_s = (current.robot.sample_monotonic_ns - previous.robot.sample_monotonic_ns) / 1e9
    return {
        "observations": {
            "ee_pos": list(previous.robot.ee_pos),
            "joint_pos": list(previous.robot.joint_pos),
            "gripper_pos": previous.robot.gripper_pos,
        },
        "action": {
            "delta_ee_pos": delta_ee,
            "delta_joint_pos": delta_joint,
            "delta_gripper": current.robot.gripper_pos - previous.robot.gripper_pos,
        },
        "meta": {
            "step": step,
            "total_gripper_moves": total_gripper_moves,
            "schema_version": SCHEMA_VERSION,
            "sample_time_unix_ns": previous.robot.sample_time_unix_ns,
            "sample_monotonic_ns": previous.robot.sample_monotonic_ns,
            "dt_s": dt_s,
            "camera_timestamps_ms": [
                float(frame.device_timestamp_ms) for frame in previous.frames
            ],
            "camera_frame_numbers": [int(frame.frame_number) for frame in previous.frames],
            "camera_host_monotonic_ns": [
                int(frame.host_monotonic_ns) for frame in previous.frames
            ],
            "camera_skew_ms": previous.camera_skew_ms,
            "camera_max_age_ms": previous.camera_max_age_ms,
            "state_read_duration_ms": previous.robot.read_duration_ms,
            "delta_ee_rotation_wrapped": [
                wrapped_angle_delta(current.robot.ee_pos[index], previous.robot.ee_pos[index])
                for index in range(3, 6)
            ],
        },
    }


def validate_dataset_name(name: str) -> str:
    if not _DATASET_RE.fullmatch(name):
        raise ValueError(
            "dataset must start with an ASCII letter/digit and contain only "
            "letters, digits, '.', '_' or '-'"
        )
    return name


def next_episode_index(dataset_dir: Path) -> int:
    indices: list[int] = []
    if dataset_dir.exists():
        for child in dataset_dir.iterdir():
            match = _EPISODE_RE.fullmatch(child.name) or _IN_PROGRESS_RE.fullmatch(child.name)
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


class ArmSource(Protocol):
    def start(self) -> None: ...

    def read(self) -> RobotState: ...

    def close(self) -> None: ...


class CameraSource(Protocol):
    camera_index: int
    serial: str
    buffer: FrameBuffer
    fatal_error: BaseException | None

    def start(self) -> None: ...

    def close(self) -> None: ...


class XArmStateSource:
    """Read-only state source. ``xarm`` is imported only in :meth:`start`."""

    def __init__(self, ip: str):
        self.ip = ip
        self._arm: Any = None

    def start(self) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ModuleNotFoundError as exc:  # pragma: no cover - hardware environment
            raise CollectionError("xArm SDK is not installed in this Python environment") from exc
        self._arm = XArmAPI(self.ip)
        if hasattr(self._arm, "connect") and not getattr(self._arm, "connected", True):
            result = self._arm.connect()
            if isinstance(result, int) and result != 0:
                raise CollectionError(f"xArm connect failed with code {result}")

    @staticmethod
    def _checked(result: Any, operation: str) -> Any:
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise RobotReadError(f"{operation} returned malformed result: {result!r}")
        code, value = result[0], result[1]
        if code != 0:
            raise RobotReadError(f"{operation} failed with xArm code {code}")
        if value is None:
            raise RobotReadError(f"{operation} returned no value")
        return value

    def read(self) -> RobotState:
        if self._arm is None:
            raise RobotReadError("xArm source has not been started")
        started_ns = time.monotonic_ns()
        ee_raw = self._checked(self._arm.get_position(), "get_position")
        joints_raw = self._checked(self._arm.get_servo_angle(), "get_servo_angle")
        gripper_raw = self._checked(
            self._arm.get_gripper_position(), "get_gripper_position"
        )
        completed_ns = time.monotonic_ns()
        ee = _finite_tuple(ee_raw, 6, "end-effector pose")
        joints = _finite_tuple(joints_raw, 6, "joint pose")
        try:
            gripper = float(gripper_raw)
        except (TypeError, ValueError) as exc:
            raise RobotReadError(f"invalid gripper position: {gripper_raw!r}") from exc
        if not math.isfinite(gripper):
            raise RobotReadError("gripper position is not finite")
        return RobotState(
            ee_pos=ee,
            joint_pos=joints,
            gripper_pos=gripper,
            sample_monotonic_ns=completed_ns,
            sample_time_unix_ns=time.time_ns(),
            read_duration_ms=(completed_ns - started_ns) / 1_000_000.0,
        )

    def close(self) -> None:
        if self._arm is not None:
            try:
                self._arm.disconnect()
            finally:
                self._arm = None


def _finite_tuple(value: Any, length: int, label: str) -> tuple[Any, ...]:
    try:
        converted = tuple(float(item) for item in value[:length])
    except (TypeError, ValueError, IndexError) as exc:
        raise RobotReadError(f"invalid {label}: {value!r}") from exc
    if len(converted) != length or not all(math.isfinite(item) for item in converted):
        raise RobotReadError(f"invalid {label}: expected {length} finite values")
    return converted


class RealSenseCameraSource:
    """One RealSense YUYV camera with a bounded converted-frame buffer."""

    def __init__(
        self,
        camera_index: int,
        serial: str,
        config: CollectorConfig,
        *,
        error_callback: Callable[[str], None] | None = None,
    ):
        self.camera_index = camera_index
        self.serial = serial
        self.config = config
        self.buffer = FrameBuffer(config.frame_buffer_size)
        self.fatal_error: BaseException | None = None
        self._error_callback = error_callback
        self._pipeline: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            import cv2
            import numpy as np
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:  # pragma: no cover - hardware environment
            raise CollectionError(
                "RealSense collection requires cv2, numpy, and pyrealsense2"
            ) from exc

        self._cv2 = cv2
        self._np = np
        self._rs = rs
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs_config.enable_device(self.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.yuyv,
            self.config.camera_fps,
        )
        try:
            pipeline.start(rs_config)
        except Exception as exc:
            raise CollectionError(f"failed to start camera {self.serial}: {exc}") from exc
        self._pipeline = pipeline
        self._thread = threading.Thread(
            target=self._worker,
            name=f"realsense-{self.camera_index}",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:  # pragma: no cover - exercised with hardware
        consecutive_errors = 0
        warmup_remaining = self.config.warmup_frames
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError("frameset contained no color frame")
                if warmup_remaining > 0:
                    warmup_remaining -= 1
                    consecutive_errors = 0
                    continue
                acquired_monotonic_ns = time.monotonic_ns()
                acquired_wall_ns = time.time_ns()
                raw = self._np.asanyarray(color.get_data())
                # RealSense YUYV is two bytes per pixel.  View bytes explicitly;
                # never infer the color format from numpy's dtype.
                yuyv = raw.view(self._np.uint8).reshape(
                    self.config.height, self.config.width, 2
                )
                image = self._cv2.cvtColor(yuyv, self._cv2.COLOR_YUV2BGR_YUYV)
                self.buffer.append(
                    CameraFrame(
                        camera_index=self.camera_index,
                        image=image,
                        host_monotonic_ns=acquired_monotonic_ns,
                        host_wall_time_ns=acquired_wall_ns,
                        device_timestamp_ms=float(color.get_timestamp()),
                        frame_number=int(color.get_frame_number()),
                    )
                )
                consecutive_errors = 0
            except Exception as exc:
                if self._stop.is_set():
                    break
                consecutive_errors += 1
                if self._error_callback is not None:
                    self._error_callback(f"camera_{self.camera_index}_read_error")
                if consecutive_errors >= self.config.consecutive_failure_limit:
                    self.fatal_error = CollectionError(
                        f"camera {self.serial} failed {consecutive_errors} times: {exc}"
                    )
                    break

    def close(self) -> None:
        self._stop.set()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._pipeline = None


class MockArmSource:
    """Deterministic arm state source for no-hardware end-to-end runs."""

    def __init__(self):
        self._count = 0

    def start(self) -> None:
        self._count = 0

    def read(self) -> RobotState:
        started = time.monotonic_ns()
        index = self._count
        self._count += 1
        completed = time.monotonic_ns()
        return RobotState(
            ee_pos=(300.0 + index, 200.0, 350.0, 179.0 + index * 2.0, 0.0, 0.0),
            joint_pos=tuple(float(index + joint) for joint in range(6)),  # type: ignore[arg-type]
            gripper_pos=float(850 if (index // 10) % 2 == 0 else 0),
            sample_monotonic_ns=completed,
            sample_time_unix_ns=time.time_ns(),
            read_duration_ms=(completed - started) / 1_000_000.0,
        )

    def close(self) -> None:
        return None


class MockCameraSource:
    """Synthetic BGR camera which follows the same buffering contract."""

    def __init__(self, camera_index: int, serial: str, config: CollectorConfig):
        self.camera_index = camera_index
        self.serial = serial
        self.config = config
        self.buffer = FrameBuffer(config.frame_buffer_size)
        self.fatal_error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise CollectionError("mock collection requires numpy") from exc
        self._np = np
        self._thread = threading.Thread(
            target=self._worker, name=f"mock-camera-{self.camera_index}", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        interval = 1.0 / self.config.camera_fps
        frame_number = 0
        deadline = time.monotonic()
        while not self._stop.is_set():
            value = (frame_number + self.camera_index * 80) % 256
            image = self._np.full(
                (self.config.height, self.config.width, 3), value, dtype=self._np.uint8
            )
            now_ns = time.monotonic_ns()
            self.buffer.append(
                CameraFrame(
                    camera_index=self.camera_index,
                    image=image,
                    host_monotonic_ns=now_ns,
                    host_wall_time_ns=time.time_ns(),
                    device_timestamp_ms=frame_number * interval * 1000.0,
                    frame_number=frame_number,
                )
            )
            frame_number += 1
            deadline += interval
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


@dataclass(frozen=True)
class _StepPayload:
    step: int
    images: tuple[Any, Any]
    data: dict[str, Any]


class AtomicEpisodeWriter:
    """Asynchronous bounded writer for one hidden in-progress episode."""

    def __init__(
        self,
        config: CollectorConfig,
        dataset: str,
        *,
        episode_index: int | None = None,
        error_callback: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.dataset = validate_dataset_name(dataset)
        self.dataset_dir = config.data_root / self.dataset
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.episode_index = (
            next_episode_index(self.dataset_dir) if episode_index is None else episode_index
        )
        self.episode_name = f"episode_{self.episode_index:03d}"
        self.final_path = self.dataset_dir / self.episode_name
        self.in_progress_path = self.dataset_dir / f".{self.episode_name}.inprogress"
        if self.final_path.exists() or self.in_progress_path.exists():
            raise StorageError(f"episode path already exists: {self.episode_name}")
        self._ensure_disk_space()
        self.in_progress_path.mkdir()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=config.save_queue_size)
        self._thread: threading.Thread | None = None
        self._fatal_error: BaseException | None = None
        self._error_callback = error_callback
        self.steps_written = 0
        self.queue_high_water_mark = 0
        self._closed = False

    @property
    def fatal_error(self) -> BaseException | None:
        return self._fatal_error

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def _ensure_disk_space(self) -> None:
        probe = self.dataset_dir if self.dataset_dir.exists() else self.config.data_root
        probe.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(probe).free / (1024**3)
        if free_gb < self.config.min_free_gb:
            raise StorageError(
                f"only {free_gb:.2f} GiB free; minimum is {self.config.min_free_gb:.2f} GiB"
            )

    def start(self) -> None:
        # A Python thread blocked inside a filesystem call cannot be cancelled.
        # Keep it daemonized so the collector process can still return control to
        # the orchestration layer and stop the robot if storage becomes wedged.
        # Normal shutdown still drains the queue before returning.
        self._thread = threading.Thread(target=self._worker, name="episode-writer", daemon=True)
        self._thread.start()

    def submit(self, payload: _StepPayload) -> None:
        if self._fatal_error is not None:
            raise StorageError(f"writer failed: {self._fatal_error}")
        try:
            self._queue.put_nowait(payload)
            self.queue_high_water_mark = max(self.queue_high_water_mark, self._queue.qsize())
        except queue.Full as exc:
            if self._error_callback is not None:
                self._error_callback("save_queue_full")
            raise StorageError(
                f"save queue reached its {self.config.save_queue_size}-step limit"
            ) from exc

    def _worker(self) -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            self._fatal_error = StorageError("saving images requires cv2")
            cv2 = None  # type: ignore[assignment]

        while True:
            item = self._queue.get()
            try:
                if item is _SENTINEL:
                    return
                if self._fatal_error is not None:
                    continue
                assert isinstance(item, _StepPayload)
                self._write_step(item, cv2)
                self.steps_written += 1
            except BaseException as exc:
                self._fatal_error = exc
                if self._error_callback is not None:
                    self._error_callback("step_write_error")
            finally:
                self._queue.task_done()

    def _write_step(self, payload: _StepPayload, cv2: Any) -> None:
        self._ensure_disk_space()
        final = self.in_progress_path / f"step_{payload.step:05d}"
        temporary = self.in_progress_path / f".step_{payload.step:05d}.{uuid.uuid4().hex}.tmp"
        if final.exists():
            raise StorageError(f"duplicate step path: {final}")
        temporary.mkdir()
        try:
            for index, image in enumerate(payload.images):
                image_path = temporary / f"cam_{index}.jpg"
                ok = cv2.imwrite(
                    str(image_path),
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality],
                )
                if not ok:
                    raise StorageError(f"JPEG encoding failed for {image_path.name}")
            json_path = temporary / "data.json"
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload.data, handle, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.rename(final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def close(self, *, timeout_s: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return

        timeout = self.config.writer_close_timeout_s if timeout_s is None else timeout_s
        if timeout <= 0:
            raise ValueError("writer close timeout must be positive")
        deadline = time.monotonic() + timeout

        # queue.put() and queue.join() have no useful cancellation semantics.
        # Poll both operations against one deadline so a full queue or a stuck
        # image/filesystem write cannot indefinitely block safety cleanup.
        sentinel_enqueued = False
        while self._thread.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.put(_SENTINEL, timeout=min(0.05, remaining))
                sentinel_enqueued = True
                break
            except queue.Full:
                continue

        while self._thread.is_alive() and time.monotonic() < deadline:
            self._thread.join(timeout=min(0.05, max(0.0, deadline - time.monotonic())))

        if self._thread.is_alive():
            detail = "before the shutdown marker could be queued" if not sentinel_enqueued else "while draining pending steps"
            raise StorageError(f"writer shutdown timed out {detail}")
        if self._queue.unfinished_tasks:
            raise StorageError(
                f"writer stopped with {self._queue.unfinished_tasks} unfinished queue item(s)"
            )
        if self._fatal_error is not None:
            raise StorageError(f"writer failed: {self._fatal_error}") from self._fatal_error

    def finish(self, metadata: Mapping[str, Any], *, complete: bool) -> Path:
        if not self._closed:
            self.close()
        _atomic_json_write(self.in_progress_path / "episode_meta.json", metadata)
        if complete:
            self.in_progress_path.rename(self.final_path)
            return self.final_path
        return self.in_progress_path


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class TeleopCollector:
    """Coordinate state sampling, soft camera sync, and atomic persistence."""

    def __init__(
        self,
        config: CollectorConfig,
        dataset: str,
        *,
        task: str = "",
        mock: bool = False,
        arm_source: ArmSource | None = None,
        camera_sources: Sequence[CameraSource] | None = None,
        stop_event: threading.Event | None = None,
        collection_lease_file: Path | None = None,
        control_abort_file: Path | None = None,
        teleop_armed_file: Path | None = None,
    ):
        self.config = config
        self.dataset = validate_dataset_name(dataset)
        self.task = task
        self.mock = mock
        self.stop_event = stop_event or threading.Event()
        self.collection_lease_file = collection_lease_file
        self.control_abort_file = control_abort_file
        self.teleop_armed_file = teleop_armed_file
        self.error_counts: Counter[str] = Counter()
        self.arm_source: ArmSource = arm_source or (
            MockArmSource() if mock else XArmStateSource(config.robot_ip)
        )
        if camera_sources is None:
            camera_type = MockCameraSource if mock else RealSenseCameraSource
            self.camera_sources = [
                camera_type(index, serial, config)  # type: ignore[call-arg]
                for index, serial in enumerate(config.camera_serials)
            ]
            for source in self.camera_sources:
                if isinstance(source, RealSenseCameraSource):
                    source._error_callback = self._increment_error
        else:
            self.camera_sources = list(camera_sources)
        if len(self.camera_sources) != 2:
            raise ValueError("collector requires exactly two camera sources")

    def _increment_error(self, key: str) -> None:
        self.error_counts[key] += 1

    def stop(self) -> None:
        self.stop_event.set()

    def _refresh_collection_lease(self) -> None:
        if self.collection_lease_file is None:
            return
        path = self.collection_lease_file
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps({"pid": os.getpid(), "monotonic_ns": time.monotonic_ns()}),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _release_collection_lease(self) -> None:
        if self.collection_lease_file is not None:
            self.collection_lease_file.unlink(missing_ok=True)

    def _raise_if_control_aborted(self) -> None:
        if self.control_abort_file is None or not self.control_abort_file.exists():
            return
        try:
            reason = self.control_abort_file.read_text(encoding="utf-8").strip()
        except OSError:
            reason = "the ROS/bridge control stack stopped"
        raise CollectionError(reason or "the ROS/bridge control stack stopped")

    def _wait_for_teleop_armed(self) -> None:
        if self.teleop_armed_file is None:
            return
        while not self.stop_event.is_set():
            self._raise_if_control_aborted()
            self._refresh_collection_lease()
            try:
                payload = json.loads(self.teleop_armed_file.read_text(encoding="utf-8"))
                if int(payload.get("collector_pid", -1)) == os.getpid():
                    return
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            self.stop_event.wait(0.05)
        raise CollectionError("collection stopped before VR teleoperation was armed")

    def run(self, *, max_steps: int | None = None) -> CollectionResult:
        """Record one episode.

        ``max_steps`` exists for mock/acceptance runs; normal collection stops on
        Ctrl+C or :meth:`stop`.  Controlled user stops finalize the episode,
        whereas hardware/storage failures deliberately leave the hidden episode
        for ``vrctl validate`` to report.
        """

        writer: AtomicEpisodeWriter | None = None
        started_wall = datetime.now(timezone.utc)
        started_monotonic_ns = time.monotonic_ns()
        samples_captured = 0
        gripper_moves = 0
        previous: Sample | None = None
        previous_gripper: float | None = None
        termination_reason = "stopped"
        terminal_error: BaseException | None = None
        writer_close_error: BaseException | None = None

        try:
            writer = AtomicEpisodeWriter(
                self.config, self.dataset, error_callback=self._increment_error
            )
            writer.start()
            self.arm_source.start()
            for source in self.camera_sources:
                source.start()
            self._wait_for_cameras()
            self._refresh_collection_lease()
            print(
                "COLLECTOR READY: robot connection and both cameras are ready. "
                "Open/keep open the Quest app, hold the controllers still until TELEOP ARMED.",
                flush=True,
            )
            self._wait_for_teleop_armed()
            started_wall = datetime.now(timezone.utc)
            started_monotonic_ns = time.monotonic_ns()
            print(
                "TELEOP ARMED: Quest controls are live and recording has started.",
                flush=True,
            )

            interval_s = 1.0 / self.config.hz
            deadline = time.monotonic()
            consecutive_failures = 0
            last_disk_check = 0.0
            while not self.stop_event.is_set():
                self._raise_if_control_aborted()
                self._refresh_collection_lease()
                for source in self.camera_sources:
                    if source.fatal_error is not None:
                        raise CollectionError(str(source.fatal_error)) from source.fatal_error
                if writer.fatal_error is not None:
                    raise StorageError(f"writer failed: {writer.fatal_error}")

                now = time.monotonic()
                if now - last_disk_check >= 1.0:
                    writer._ensure_disk_space()
                    last_disk_check = now

                # Read robot state first and use its completed-read timestamp as
                # the synchronization target.  Selecting before these network
                # calls understated image age by the complete state-read delay.
                try:
                    robot = self.arm_source.read()
                except RobotReadError:
                    self._increment_error("robot_read_error")
                    consecutive_failures += 1
                    if consecutive_failures >= self.config.consecutive_failure_limit:
                        raise
                else:
                    selection = select_synchronized_frames(
                        [source.buffer.snapshot() for source in self.camera_sources],
                        robot.sample_monotonic_ns,
                        max_skew_ms=self.config.max_camera_skew_ms,
                        max_age_ms=self.config.max_frame_age_ms,
                    )
                    if selection is None:
                        self._increment_error("camera_sync_miss")
                        consecutive_failures += 1
                        if consecutive_failures >= self.config.consecutive_failure_limit:
                            raise CollectionError(
                                "no synchronized fresh camera pair for "
                                f"{consecutive_failures} consecutive samples"
                            )
                    else:
                        consecutive_failures = 0
                        sample = Sample(
                            robot=robot,
                            frames=(selection.frames[0], selection.frames[1]),
                            camera_skew_ms=selection.skew_ms,
                            camera_max_age_ms=selection.max_age_ms,
                        )
                        samples_captured += 1
                        if previous_gripper is not None and abs(
                            robot.gripper_pos - previous_gripper
                        ) > self.config.gripper_move_threshold:
                            gripper_moves += 1
                        previous_gripper = robot.gripper_pos

                        if previous is not None:
                            step_index = samples_captured - 2
                            data = build_step_data(
                                previous,
                                sample,
                                step=step_index,
                                total_gripper_moves=gripper_moves,
                            )
                            writer.submit(
                                _StepPayload(
                                    step=step_index,
                                    images=(previous.frames[0].image, previous.frames[1].image),
                                    data=data,
                                )
                            )
                        previous = sample
                        if max_steps is not None and samples_captured - 1 >= max_steps:
                            termination_reason = "max_steps"
                            break

                deadline += interval_s
                wait_s = deadline - time.monotonic()
                if wait_s <= -interval_s:
                    self._increment_error("sampling_deadline_miss")
                    deadline = time.monotonic()
                    wait_s = 0.0
                self.stop_event.wait(max(0.0, wait_s))
        except KeyboardInterrupt:
            termination_reason = "keyboard_interrupt"
            self.stop_event.set()
        except BaseException as exc:
            terminal_error = exc
            termination_reason = _termination_reason(exc)
            self.stop_event.set()
        finally:
            # Pause VR motion before camera shutdown and writer flush.
            self._release_collection_lease()
            for source in reversed(self.camera_sources):
                try:
                    source.close()
                except BaseException:
                    self._increment_error("camera_close_error")
            try:
                self.arm_source.close()
            except BaseException:
                self._increment_error("robot_close_error")
            if writer is not None:
                try:
                    writer.close(timeout_s=self.config.writer_close_timeout_s)
                except BaseException as exc:
                    writer_close_error = exc
                    self._increment_error("writer_close_error")

        if terminal_error is None and writer_close_error is not None:
            terminal_error = writer_close_error
            termination_reason = "storage_error"

        if terminal_error is None and writer is not None and writer.steps_written == 0:
            terminal_error = CollectionError("episode contains no complete observation/action step")
            termination_reason = "no_steps"

        ended_wall = datetime.now(timezone.utc)
        ended_monotonic_ns = time.monotonic_ns()
        complete = terminal_error is None
        if writer is None:
            raise CollectionError(str(terminal_error or "collector failed before episode creation"))

        metadata = self._episode_metadata(
            writer=writer,
            started_wall=started_wall,
            ended_wall=ended_wall,
            duration_s=(ended_monotonic_ns - started_monotonic_ns) / 1e9,
            termination_reason=termination_reason,
            complete=complete,
            samples_captured=samples_captured,
            gripper_moves=gripper_moves,
            terminal_error=terminal_error,
        )
        if writer_close_error is not None:
            # The daemon writer may still be blocked inside the filesystem.  Do
            # not race it by writing episode metadata into the same hidden tree.
            path = writer.in_progress_path
        else:
            try:
                path = writer.finish(metadata, complete=complete)
            except BaseException as exc:
                terminal_error = terminal_error or exc
                complete = False
                path = writer.in_progress_path

        result = CollectionResult(
            dataset=self.dataset,
            episode_index=writer.episode_index,
            path=path,
            complete=complete,
            termination_reason=termination_reason,
            steps_written=writer.steps_written,
            samples_captured=samples_captured,
            error_counts=dict(self.error_counts),
        )
        if terminal_error is not None:
            raise CollectionError(str(terminal_error), result=result) from terminal_error
        return result

    def _wait_for_cameras(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            for source in self.camera_sources:
                if source.fatal_error is not None:
                    raise CollectionError(str(source.fatal_error)) from source.fatal_error
            if all(len(source.buffer) > 0 for source in self.camera_sources):
                return
            if self.stop_event.wait(0.02):
                return
        missing = [source.serial for source in self.camera_sources if len(source.buffer) == 0]
        raise CollectionError(f"camera startup timed out; no frames from: {', '.join(missing)}")

    def _episode_metadata(
        self,
        *,
        writer: AtomicEpisodeWriter,
        started_wall: datetime,
        ended_wall: datetime,
        duration_s: float,
        termination_reason: str,
        complete: bool,
        samples_captured: int,
        gripper_moves: int,
        terminal_error: BaseException | None,
    ) -> dict[str, Any]:
        steps = writer.steps_written
        actual_hz = steps / duration_s if duration_s > 0 else 0.0
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "episode_index": writer.episode_index,
            "task": self.task,
            "started_at_utc": started_wall.isoformat(),
            "ended_at_utc": ended_wall.isoformat(),
            "duration_s": duration_s,
            "termination_reason": termination_reason,
            "complete": complete,
            "steps_written": steps,
            "samples_captured": samples_captured,
            "total_gripper_moves": gripper_moves,
            "cameras": [
                {
                    "index": index,
                    "serial": serial,
                    "role": self.config.camera_roles[index],
                    "width": self.config.width,
                    "height": self.config.height,
                    "fps": self.config.camera_fps,
                    "format": "YUYV" if not self.mock else "BGR-mock",
                }
                for index, serial in enumerate(self.config.camera_serials)
            ],
            "config": self.config.snapshot(),
            "error_counts": dict(self.error_counts),
            "stats": {
                "actual_step_hz": actual_hz,
                "save_queue_high_water_mark": writer.queue_high_water_mark,
            },
            "error": str(terminal_error) if terminal_error is not None else None,
        }


def _termination_reason(exc: BaseException) -> str:
    if isinstance(exc, StorageError):
        message = str(exc).lower()
        return "disk_full" if "free" in message or "disk" in message else "storage_error"
    if isinstance(exc, RobotReadError):
        return "robot_error"
    message = str(exc).lower()
    if "camera" in message:
        return "camera_error"
    return "collection_error"


def collect_episode(
    config: CollectorConfig | Mapping[str, Any],
    dataset: str,
    *,
    task: str = "",
    mock: bool = False,
    stop_event: threading.Event | None = None,
    max_steps: int | None = None,
    collection_lease_file: Path | None = None,
    control_abort_file: Path | None = None,
    teleop_armed_file: Path | None = None,
) -> CollectionResult:
    """Convenience API used by ``vrctl collect`` and ``vrctl run``."""

    resolved = config if isinstance(config, CollectorConfig) else CollectorConfig.from_mapping(config)
    return TeleopCollector(
        resolved,
        dataset,
        task=task,
        mock=mock,
        stop_event=stop_event,
        collection_lease_file=collection_lease_file,
        control_abort_file=control_abort_file,
        teleop_armed_file=teleop_armed_file,
    ).run(max_steps=max_steps)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record one UF850 VR teleoperation episode")
    parser.add_argument("--config", required=True, help="project config.toml path")
    parser.add_argument("--dataset", required=True, help="dataset name below paths.data_root")
    parser.add_argument("--task", default="", help="task text saved in episode metadata")
    parser.add_argument("--mock", action="store_true", help="use synthetic robot and cameras")
    parser.add_argument("--max-steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--collection-lease-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--control-abort-file", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--teleop-armed-file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def install_collection_signal_handlers() -> None:
    """Make all normal termination signals follow the controlled finalize path.

    Non-interactive shells commonly start background jobs with ``SIGINT`` ignored,
    and ignored dispositions survive ``exec``.  Explicitly restoring Python's
    default interrupt handler here ensures the manager's SIGINT is never lost.
    SIGTERM and SIGHUP deliberately use the same ``KeyboardInterrupt`` path so a
    service stop or terminal disconnect also drains and finalizes the episode.
    """

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)


def main(argv: Sequence[str] | None = None) -> int:
    install_collection_signal_handlers()
    args = _argument_parser().parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        print("ERROR: --max-steps must be positive", file=__import__("sys").stderr)
        return 2
    try:
        from .config import load_config

        raw_config = load_config(args.config)
        config = CollectorConfig.from_mapping(raw_config)
        print(
            f"Recording dataset '{args.dataset}' at {config.hz:g} Hz. "
            "Press Ctrl+C to finalize the episode.",
            flush=True,
        )
        result = collect_episode(
            config,
            args.dataset,
            task=args.task,
            mock=args.mock,
            max_steps=args.max_steps,
            collection_lease_file=args.collection_lease_file,
            control_abort_file=args.control_abort_file,
            teleop_armed_file=args.teleop_armed_file,
        )
    except (CollectionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        if isinstance(exc, CollectionError) and exc.result is not None:
            print(f"Incomplete episode retained at: {exc.result.path}", file=__import__("sys").stderr)
        return 1
    print(
        f"Saved {result.steps_written} steps to {result.path} "
        f"({result.termination_reason})."
    )
    return 0


__all__ = [
    "AtomicEpisodeWriter",
    "CameraFrame",
    "CollectionError",
    "CollectionResult",
    "CollectorConfig",
    "FrameBuffer",
    "FrameSelection",
    "MockArmSource",
    "MockCameraSource",
    "RealSenseCameraSource",
    "RobotReadError",
    "RobotState",
    "Sample",
    "StorageError",
    "TeleopCollector",
    "XArmStateSource",
    "build_step_data",
    "collect_episode",
    "install_collection_signal_handlers",
    "main",
    "next_episode_index",
    "select_synchronized_frames",
    "validate_dataset_name",
    "wrapped_angle_delta",
]


if __name__ == "__main__":
    raise SystemExit(main())
