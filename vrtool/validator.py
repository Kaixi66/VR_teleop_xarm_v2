"""Read-only validation for UF850 VR teleoperation datasets.

JPEGs are fully decoded by default so damaged entropy data cannot pass merely
because its header and final marker look plausible.  ``--fast`` retains a
header-only mode for quick scans or environments without OpenCV.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


EPISODE_RE = re.compile(r"^episode_(\d{3,})$")
STEP_RE = re.compile(r"^step_(\d{5,})$")
INPROGRESS_EPISODE_RE = re.compile(r"^\.episode_(\d{3,})\.inprogress$")
TEMP_STEP_RE = re.compile(r"^\.step_(\d{5,})\..+\.tmp$")

V2_SCHEMA_VERSION = "2.0"
V2_META_FIELDS = (
    "schema_version",
    "sample_time_unix_ns",
    "sample_monotonic_ns",
    "dt_s",
    "camera_timestamps_ms",
    "camera_frame_numbers",
    "camera_host_monotonic_ns",
    "camera_skew_ms",
    "camera_max_age_ms",
    "state_read_duration_ms",
    "delta_ee_rotation_wrapped",
)
V2_EPISODE_META_FIELDS = (
    "schema_version",
    "dataset",
    "episode_index",
    "task",
    "started_at_utc",
    "ended_at_utc",
    "duration_s",
    "termination_reason",
    "complete",
    "steps_written",
    "samples_captured",
    "total_gripper_moves",
    "cameras",
    "config",
    "error_counts",
    "stats",
)


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding."""

    severity: str
    code: str
    path: str
    message: str


@dataclass
class ValidationResult:
    """Structured result returned by :func:`validate_dataset`."""

    dataset: str
    episodes_checked: int = 0
    steps_checked: int = 0
    legacy_episodes: int = 0
    v2_episodes: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def ok(self) -> bool:
        return self.error_count == 0

    @property
    def valid(self) -> bool:
        """Alias retained for callers that prefer ``result.valid``."""

        return self.ok

    def __bool__(self) -> bool:
        return self.ok

    def add(self, severity: str, code: str, path: str, message: str) -> None:
        self.issues.append(ValidationIssue(severity, code, path, message))

    def error(self, code: str, path: str, message: str) -> None:
        self.add("error", code, path, message)

    def warning(self, code: str, path: str, message: str) -> None:
        self.add("warning", code, path, message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "ok": self.ok,
            "episodes_checked": self.episodes_checked,
            "steps_checked": self.steps_checked,
            "legacy_episodes": self.legacy_episodes,
            "v2_episodes": self.v2_episodes,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class _StepRecord:
    number: int
    path: Path
    data: dict[str, Any] | None = None
    v2: bool = False


def _relative(path: Path, root: Path) -> str:
    try:
        value = str(path.relative_to(root))
        return value or "."
    except ValueError:
        return str(path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _missing_ranges(numbers: Sequence[int]) -> str:
    if not numbers:
        return ""
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = number
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _check_contiguous(
    numbers: Sequence[int],
    *,
    result: ValidationResult,
    code: str,
    path: str,
    label: str,
) -> None:
    if not numbers:
        return
    unique = sorted(set(numbers))
    expected = set(range(unique[-1] + 1))
    missing = sorted(expected.difference(unique))
    if missing:
        result.error(
            code,
            path,
            f"{label} numbering must be continuous from 0; missing {_missing_ranges(missing)}",
        )
    if len(unique) != len(numbers):
        result.error(code, path, f"{label} contains duplicate numeric indices")


def _read_json(path: Path, result: ValidationResult, root: Path) -> Any | None:
    rel = _relative(path, root)
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError:
        result.error("missing_json", rel, "required JSON file is missing")
    except UnicodeDecodeError as exc:
        result.error("invalid_json_encoding", rel, f"JSON is not UTF-8: {exc}")
    except json.JSONDecodeError as exc:
        result.error(
            "invalid_json",
            rel,
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )
    except OSError as exc:
        result.error("unreadable_file", rel, f"cannot read file: {exc}")
    return None


def _jpeg_info(path: Path) -> tuple[int, int, int]:
    """Return ``(width, height, components)`` from a JPEG SOF segment.

    A Start-of-Scan and terminal End-of-Image marker are also required.  This
    catches truncated and mislabeled files without loading multi-megabyte image
    data or requiring OpenCV/Pillow.
    """

    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            raise ValueError("missing JPEG SOI marker")
        stream.seek(0, 2)
        size = stream.tell()
        if size < 6:
            raise ValueError("JPEG is too short")
        stream.seek(-2, 2)
        if stream.read(2) != b"\xff\xd9":
            raise ValueError("missing terminal JPEG EOI marker")
        stream.seek(2)

        dimensions: tuple[int, int, int] | None = None
        saw_scan = False
        while stream.tell() < size - 2:
            prefix = stream.read(1)
            if prefix != b"\xff":
                raise ValueError("invalid marker prefix before JPEG scan")
            marker_byte = stream.read(1)
            while marker_byte == b"\xff":
                marker_byte = stream.read(1)
            if not marker_byte:
                raise ValueError("truncated JPEG marker")
            marker = marker_byte[0]
            if marker == 0x00:
                raise ValueError("unexpected stuffed marker before JPEG scan")
            if marker == 0xD9:
                break
            if marker == 0xDA:
                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    raise ValueError("truncated JPEG SOS segment")
                length = int.from_bytes(length_bytes, "big")
                if length < 2 or stream.tell() + length - 2 > size:
                    raise ValueError("invalid JPEG SOS segment length")
                saw_scan = True
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue

            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment length")
            length = int.from_bytes(length_bytes, "big")
            if length < 2 or stream.tell() + length - 2 > size:
                raise ValueError("invalid JPEG segment length")
            payload_size = length - 2
            if marker in sof_markers:
                payload = stream.read(payload_size)
                if len(payload) < 6:
                    raise ValueError("truncated JPEG SOF segment")
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                components = payload[5]
                if width <= 0 or height <= 0 or components <= 0:
                    raise ValueError("invalid JPEG dimensions/components")
                dimensions = (width, height, components)
            else:
                stream.seek(payload_size, 1)

        if dimensions is None:
            raise ValueError("missing JPEG SOF segment")
        if not saw_scan:
            raise ValueError("missing JPEG SOS segment")
        return dimensions


def _validate_image(
    path: Path,
    result: ValidationResult,
    root: Path,
    expected_size: tuple[int, int],
    *,
    fast: bool,
) -> None:
    rel = _relative(path, root)
    if not path.is_file():
        result.error("missing_camera", rel, "required camera JPEG is missing")
        return
    try:
        width, height, components = _jpeg_info(path)
    except (OSError, ValueError) as exc:
        result.error("invalid_jpeg", rel, str(exc))
        return
    if (width, height) != expected_size:
        result.error(
            "image_dimensions",
            rel,
            f"expected {expected_size[0]}x{expected_size[1]}, got {width}x{height}",
        )
    if components != 3:
        result.error(
            "image_channels", rel, f"expected 3 JPEG components, got {components}"
        )
    if fast:
        return

    try:
        import cv2
    except ModuleNotFoundError:
        result.error(
            "jpeg_decoder_unavailable",
            rel,
            "full JPEG validation requires OpenCV; install cv2 or use --fast",
        )
        return
    try:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        result.error("invalid_jpeg", rel, f"JPEG decoder failed: {exc}")
        return
    if image is None:
        result.error("invalid_jpeg", rel, "JPEG cannot be fully decoded")
        return
    decoded_height, decoded_width = image.shape[:2]
    decoded_components = 1 if image.ndim == 2 else image.shape[2]
    if (decoded_width, decoded_height, decoded_components) != (width, height, components):
        result.error(
            "jpeg_decode_mismatch",
            rel,
            "decoded image shape does not match the JPEG header "
            f"({decoded_width}x{decoded_height}x{decoded_components} versus "
            f"{width}x{height}x{components})",
        )


def _validate_vector(
    value: Any,
    length: int,
    *,
    result: ValidationResult,
    code: str,
    path: str,
    field_name: str,
) -> bool:
    if not isinstance(value, list) or len(value) != length:
        result.error(code, path, f"{field_name} must be a list of length {length}")
        return False
    if not all(_is_finite_number(item) for item in value):
        result.error(code, path, f"{field_name} must contain only finite numbers")
        return False
    return True


def _validate_numeric_list(
    value: Any,
    length: int,
    *,
    integers: bool,
    nonnegative: bool,
) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    predicate = _is_int if integers else _is_finite_number
    if not all(predicate(item) for item in value):
        return False
    return not nonnegative or all(item >= 0 for item in value)


def _validate_v2_meta(
    meta: dict[str, Any],
    *,
    result: ValidationResult,
    rel: str,
    max_camera_skew_ms: float,
    max_frame_age_ms: float,
) -> None:
    missing = [name for name in V2_META_FIELDS if name not in meta]
    if missing:
        result.error(
            "v2_meta_missing", rel, f"v2 meta is missing: {', '.join(missing)}"
        )

    if meta.get("schema_version") != V2_SCHEMA_VERSION:
        result.error(
            "schema_version",
            rel,
            f"schema_version must be {V2_SCHEMA_VERSION!r}",
        )

    for name in ("sample_time_unix_ns", "sample_monotonic_ns"):
        value = meta.get(name)
        if not _is_int(value) or value <= 0:
            result.error("v2_meta_type", rel, f"meta.{name} must be a positive integer")

    dt_s = meta.get("dt_s")
    if not _is_finite_number(dt_s) or dt_s <= 0:
        result.error("v2_meta_type", rel, "meta.dt_s must be a positive finite number")

    timestamps = meta.get("camera_timestamps_ms")
    if not _validate_numeric_list(timestamps, 2, integers=False, nonnegative=True):
        result.error(
            "v2_meta_type",
            rel,
            "meta.camera_timestamps_ms must contain two nonnegative finite numbers",
        )
    frame_numbers = meta.get("camera_frame_numbers")
    if not _validate_numeric_list(frame_numbers, 2, integers=True, nonnegative=True):
        result.error(
            "v2_meta_type",
            rel,
            "meta.camera_frame_numbers must contain two nonnegative integers",
        )
    host_times = meta.get("camera_host_monotonic_ns")
    if not _validate_numeric_list(host_times, 2, integers=True, nonnegative=True):
        result.error(
            "v2_meta_type",
            rel,
            "meta.camera_host_monotonic_ns must contain two nonnegative integers",
        )

    for name in ("camera_skew_ms", "camera_max_age_ms", "state_read_duration_ms"):
        value = meta.get(name)
        if not _is_finite_number(value) or value < 0:
            result.error(
                "v2_meta_type", rel, f"meta.{name} must be a nonnegative finite number"
            )

    skew = meta.get("camera_skew_ms")
    if _is_finite_number(skew) and skew > max_camera_skew_ms:
        result.warning(
            "camera_skew",
            rel,
            f"camera skew {skew:.3f} ms exceeds {max_camera_skew_ms:.3f} ms",
        )
    age = meta.get("camera_max_age_ms")
    if _is_finite_number(age) and age > max_frame_age_ms:
        result.warning(
            "camera_frame_age",
            rel,
            f"camera frame age {age:.3f} ms exceeds {max_frame_age_ms:.3f} ms",
        )

    if _validate_numeric_list(host_times, 2, integers=True, nonnegative=True) and _is_finite_number(
        skew
    ):
        computed_skew = abs(host_times[0] - host_times[1]) / 1_000_000.0
        if not math.isclose(float(skew), computed_skew, rel_tol=1e-4, abs_tol=0.05):
            result.error(
                "camera_skew_value",
                rel,
                f"camera_skew_ms={skew:.6g} does not match host timestamps "
                f"({computed_skew:.6g} ms)",
            )

    sample_ns = meta.get("sample_monotonic_ns")
    recorded_age = meta.get("camera_max_age_ms")
    if (
        _is_int(sample_ns)
        and _validate_numeric_list(host_times, 2, integers=True, nonnegative=True)
        and _is_finite_number(recorded_age)
    ):
        ages_ns = [sample_ns - stamp for stamp in host_times]
        if any(age < 0 for age in ages_ns):
            result.error(
                "camera_age_value",
                rel,
                "camera host timestamp must not be later than the robot sample timestamp",
            )
        else:
            computed_age = max(ages_ns) / 1_000_000.0
            if not math.isclose(
                float(recorded_age), computed_age, rel_tol=1e-4, abs_tol=0.05
            ):
                result.error(
                    "camera_age_value",
                    rel,
                    f"camera_max_age_ms={recorded_age:.6g} does not match robot/camera "
                    f"timestamps ({computed_age:.6g} ms)",
                )

    _validate_vector(
        meta.get("delta_ee_rotation_wrapped"),
        3,
        result=result,
        code="v2_meta_type",
        path=rel,
        field_name="meta.delta_ee_rotation_wrapped",
    )


def _validate_step_json(
    record: _StepRecord,
    *,
    result: ValidationResult,
    root: Path,
    max_camera_skew_ms: float,
    max_frame_age_ms: float,
) -> None:
    json_path = record.path / "data.json"
    rel = _relative(json_path, root)
    data = _read_json(json_path, result, root)
    if data is None:
        return
    if not isinstance(data, dict):
        result.error("json_shape", rel, "data.json root must be an object")
        return
    record.data = data

    observations = data.get("observations")
    action = data.get("action")
    meta = data.get("meta")
    if not isinstance(observations, dict):
        result.error("json_shape", rel, "observations must be an object")
        observations = {}
    if not isinstance(action, dict):
        result.error("json_shape", rel, "action must be an object")
        action = {}
    if not isinstance(meta, dict):
        result.error("json_shape", rel, "meta must be an object")
        meta = {}

    _validate_vector(
        observations.get("ee_pos"),
        6,
        result=result,
        code="observation_shape",
        path=rel,
        field_name="observations.ee_pos",
    )
    _validate_vector(
        observations.get("joint_pos"),
        6,
        result=result,
        code="observation_shape",
        path=rel,
        field_name="observations.joint_pos",
    )
    if not _is_finite_number(observations.get("gripper_pos")):
        result.error(
            "observation_shape", rel, "observations.gripper_pos must be a finite number"
        )

    _validate_vector(
        action.get("delta_ee_pos"),
        6,
        result=result,
        code="action_shape",
        path=rel,
        field_name="action.delta_ee_pos",
    )
    _validate_vector(
        action.get("delta_joint_pos"),
        6,
        result=result,
        code="action_shape",
        path=rel,
        field_name="action.delta_joint_pos",
    )
    if not _is_finite_number(action.get("delta_gripper")):
        result.error("action_shape", rel, "action.delta_gripper must be a finite number")

    step_value = meta.get("step")
    if not _is_int(step_value) or step_value != record.number:
        result.error(
            "meta_step",
            rel,
            f"meta.step must equal directory step index {record.number}",
        )
    moves = meta.get("total_gripper_moves")
    if not _is_int(moves) or moves < 0:
        result.error(
            "meta_gripper_moves",
            rel,
            "meta.total_gripper_moves must be a nonnegative integer",
        )

    record.v2 = "schema_version" in meta
    if record.v2:
        _validate_v2_meta(
            meta,
            result=result,
            rel=rel,
            max_camera_skew_ms=max_camera_skew_ms,
            max_frame_age_ms=max_frame_age_ms,
        )


def _numeric_vector(data: dict[str, Any], group: str, field_name: str, length: int) -> list[float] | None:
    group_value = data.get(group)
    if not isinstance(group_value, dict):
        return None
    value = group_value.get(field_name)
    if not isinstance(value, list) or len(value) != length:
        return None
    if not all(_is_finite_number(item) for item in value):
        return None
    return [float(item) for item in value]


def _numeric_scalar(data: dict[str, Any], group: str, field_name: str) -> float | None:
    group_value = data.get(group)
    if not isinstance(group_value, dict):
        return None
    value = group_value.get(field_name)
    return float(value) if _is_finite_number(value) else None


def _compare_action(
    current: _StepRecord,
    following: _StepRecord,
    *,
    result: ValidationResult,
    root: Path,
    tolerance: float,
) -> None:
    if current.data is None or following.data is None:
        return
    rel = _relative(current.path / "data.json", root)
    comparisons = (
        ("ee_pos", "delta_ee_pos", 6),
        ("joint_pos", "delta_joint_pos", 6),
    )
    for observation_name, action_name, length in comparisons:
        before = _numeric_vector(current.data, "observations", observation_name, length)
        after = _numeric_vector(following.data, "observations", observation_name, length)
        actual = _numeric_vector(current.data, "action", action_name, length)
        if before is None or after is None or actual is None:
            continue
        expected = [right - left for left, right in zip(before, after)]
        max_error = max(abs(got - want) for got, want in zip(actual, expected))
        if max_error > tolerance:
            result.error(
                "action_alignment",
                rel,
                f"action.{action_name} does not equal next-current "
                f"(maximum absolute error {max_error:.6g})",
            )

    before_gripper = _numeric_scalar(current.data, "observations", "gripper_pos")
    after_gripper = _numeric_scalar(following.data, "observations", "gripper_pos")
    actual_gripper = _numeric_scalar(current.data, "action", "delta_gripper")
    if before_gripper is not None and after_gripper is not None and actual_gripper is not None:
        expected_gripper = after_gripper - before_gripper
        error = abs(actual_gripper - expected_gripper)
        if error > tolerance:
            result.error(
                "action_alignment",
                rel,
                "action.delta_gripper does not equal next-current "
                f"(absolute error {error:.6g})",
            )


def _validate_time_sequence(
    current: _StepRecord,
    following: _StepRecord,
    *,
    result: ValidationResult,
    root: Path,
) -> None:
    if not current.v2 or not following.v2 or current.data is None or following.data is None:
        return
    current_meta = current.data.get("meta")
    following_meta = following.data.get("meta")
    if not isinstance(current_meta, dict) or not isinstance(following_meta, dict):
        return
    rel = _relative(current.path / "data.json", root)
    current_ns = current_meta.get("sample_monotonic_ns")
    following_ns = following_meta.get("sample_monotonic_ns")
    if _is_int(current_ns) and _is_int(following_ns):
        if following_ns <= current_ns:
            result.error(
                "timestamp_order", rel, "sample_monotonic_ns must strictly increase"
            )
        else:
            expected_dt = (following_ns - current_ns) / 1_000_000_000.0
            actual_dt = current_meta.get("dt_s")
            if _is_finite_number(actual_dt) and not math.isclose(
                float(actual_dt), expected_dt, rel_tol=0.02, abs_tol=0.002
            ):
                result.error(
                    "dt_alignment",
                    rel,
                    f"meta.dt_s={actual_dt:.6g} does not match the next sample "
                    f"timestamp delta ({expected_dt:.6g} s)",
                )

    current_frames = current_meta.get("camera_frame_numbers")
    following_frames = following_meta.get("camera_frame_numbers")
    if _validate_numeric_list(current_frames, 2, integers=True, nonnegative=True) and _validate_numeric_list(
        following_frames, 2, integers=True, nonnegative=True
    ):
        for camera_index, (before, after) in enumerate(zip(current_frames, following_frames)):
            if after < before:
                result.error(
                    "camera_frame_order",
                    rel,
                    f"camera {camera_index} frame number decreases from {before} to {after}",
                )
            elif after == before:
                result.warning(
                    "camera_frame_reuse",
                    rel,
                    f"camera {camera_index} reuses frame number {before} in adjacent steps",
                )


def _validate_episode_meta(
    episode_path: Path,
    episode_number: int,
    records: Sequence[_StepRecord],
    *,
    result: ValidationResult,
    root: Path,
    has_v2: bool,
) -> None:
    path = episode_path / "episode_meta.json"
    if not path.exists():
        if has_v2:
            result.error(
                "missing_episode_meta",
                _relative(path, root),
                "v2 episode is missing episode_meta.json",
            )
        return
    data = _read_json(path, result, root)
    rel = _relative(path, root)
    if data is None:
        return
    if not isinstance(data, dict):
        result.error("episode_meta_shape", rel, "episode_meta.json root must be an object")
        return
    if not has_v2:
        return

    missing = [field_name for field_name in V2_EPISODE_META_FIELDS if field_name not in data]
    if missing:
        result.error(
            "episode_meta_missing", rel, f"episode metadata is missing: {', '.join(missing)}"
        )
    if data.get("schema_version") != V2_SCHEMA_VERSION:
        result.error(
            "schema_version", rel, f"schema_version must be {V2_SCHEMA_VERSION!r}"
        )
    if data.get("episode_index") != episode_number:
        result.error(
            "episode_meta_index",
            rel,
            f"episode_index must equal directory episode index {episode_number}",
        )
    if data.get("dataset") != root.name:
        result.error(
            "episode_meta_dataset",
            rel,
            f"dataset must equal dataset directory name {root.name!r}",
        )
    if data.get("complete") is not True:
        result.error("episode_incomplete", rel, "finalized episode must have complete=true")
    if data.get("steps_written") != len(records):
        result.error(
            "episode_step_count",
            rel,
            f"steps_written must equal {len(records)}",
        )
    samples = data.get("samples_captured")
    if not _is_int(samples) or samples != len(records) + 1:
        result.error(
            "episode_sample_count",
            rel,
            f"samples_captured must equal steps_written + 1 ({len(records) + 1})",
        )
    for field_name in ("duration_s",):
        value = data.get(field_name)
        if not _is_finite_number(value) or value < 0:
            result.error(
                "episode_meta_type", rel, f"{field_name} must be a nonnegative finite number"
            )
    moves = data.get("total_gripper_moves")
    if not _is_int(moves) or moves < 0:
        result.error(
            "episode_meta_type", rel, "total_gripper_moves must be a nonnegative integer"
        )
    step_moves: list[int] = []
    for record in records:
        if record.data is None or not isinstance(record.data.get("meta"), dict):
            continue
        value = record.data["meta"].get("total_gripper_moves")
        if _is_int(value) and value >= 0:
            step_moves.append(value)
    if any(after < before for before, after in zip(step_moves, step_moves[1:])):
        result.error(
            "gripper_move_order", rel, "step total_gripper_moves must not decrease"
        )
    if step_moves and _is_int(moves) and moves != step_moves[-1]:
        result.error(
            "episode_gripper_moves",
            rel,
            "episode total_gripper_moves must equal the final step value",
        )
    for field_name in ("dataset", "task", "started_at_utc", "ended_at_utc", "termination_reason"):
        value = data.get(field_name)
        if not isinstance(value, str):
            result.error("episode_meta_type", rel, f"{field_name} must be a string")
    cameras = data.get("cameras")
    if not isinstance(cameras, list) or len(cameras) != 2:
        result.error("episode_meta_type", rel, "cameras must be a list with two entries")
    for field_name in ("config", "error_counts", "stats"):
        if not isinstance(data.get(field_name), dict):
            result.error("episode_meta_type", rel, f"{field_name} must be an object")


def _collect_numbered_directories(
    parent: Path,
    pattern: re.Pattern[str],
    prefix: str,
    *,
    result: ValidationResult,
    root: Path,
    malformed_code: str,
) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    try:
        entries = list(parent.iterdir())
    except OSError as exc:
        result.error("unreadable_directory", _relative(parent, root), str(exc))
        return found
    for entry in entries:
        match = pattern.fullmatch(entry.name)
        if match:
            if entry.is_dir():
                found.append((int(match.group(1)), entry))
            else:
                result.error(
                    malformed_code,
                    _relative(entry, root),
                    f"{entry.name} must be a directory",
                )
        elif entry.name.startswith(prefix):
            result.error(
                malformed_code,
                _relative(entry, root),
                f"malformed {prefix.rstrip('_')} name",
            )
    return sorted(found, key=lambda item: item[0])


def _declared_image_size(episode_path: Path) -> tuple[int, int] | None:
    """Best-effort size lookup; normal metadata validation reports bad JSON/types."""

    try:
        data = json.loads((episode_path / "episode_meta.json").read_text(encoding="utf-8"))
        cameras = data.get("cameras")
        sizes = {
            (camera.get("width"), camera.get("height"))
            for camera in cameras
            if isinstance(camera, dict)
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
        return None
    if len(sizes) != 1:
        return None
    width, height = next(iter(sizes))
    if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
        return None
    return width, height


def validate_dataset(
    path: str | Path,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    action_tolerance: float = 1e-5,
    max_camera_skew_ms: float = 50.0,
    max_frame_age_ms: float = 150.0,
    strict_v2: bool = False,
    fast: bool = False,
) -> ValidationResult:
    """Validate a dataset without changing it.

    Legacy episodes (without ``meta.schema_version`` and
    ``episode_meta.json``) remain valid.  Once a step declares a schema
    version, all v2 metadata is required and mixed legacy/v2 steps within one
    episode are rejected.
    """

    root = Path(path).expanduser().resolve()
    result = ValidationResult(dataset=str(root))
    if not root.exists():
        result.error("dataset_missing", str(root), "dataset path does not exist")
        return result
    if not root.is_dir():
        result.error("dataset_not_directory", str(root), "dataset path is not a directory")
        return result
    if (expected_width is None) != (expected_height is None):
        raise ValueError("expected image width and height must be supplied together")
    if expected_width is not None and (expected_width <= 0 or expected_height <= 0):
        raise ValueError("expected image dimensions must be positive")
    if action_tolerance < 0 or max_camera_skew_ms < 0 or max_frame_age_ms < 0:
        raise ValueError("validation tolerances must be nonnegative")

    for entry in root.iterdir():
        if INPROGRESS_EPISODE_RE.fullmatch(entry.name):
            result.error(
                "inprogress_episode",
                _relative(entry, root),
                "unfinished episode directory remains from an interrupted collection",
            )

    episodes = _collect_numbered_directories(
        root,
        EPISODE_RE,
        "episode_",
        result=result,
        root=root,
        malformed_code="malformed_episode",
    )
    if not episodes:
        result.error("no_episodes", ".", "dataset contains no finalized episodes")
        return result
    _check_contiguous(
        [number for number, _ in episodes],
        result=result,
        code="episode_sequence",
        path=".",
        label="episode",
    )

    for episode_number, episode_path in episodes:
        result.episodes_checked += 1
        expected_size = (
            (expected_width, expected_height)
            if expected_width is not None and expected_height is not None
            else (_declared_image_size(episode_path) or (1920, 1080))
        )
        for entry in episode_path.iterdir():
            if TEMP_STEP_RE.fullmatch(entry.name):
                result.error(
                    "temporary_step",
                    _relative(entry, root),
                    "unfinished temporary step remains in finalized episode",
                )
        steps = _collect_numbered_directories(
            episode_path,
            STEP_RE,
            "step_",
            result=result,
            root=root,
            malformed_code="malformed_step",
        )
        if not steps:
            result.error(
                "no_steps", _relative(episode_path, root), "episode contains no finalized steps"
            )
            _validate_episode_meta(
                episode_path,
                episode_number,
                [],
                result=result,
                root=root,
                has_v2=False,
            )
            continue
        _check_contiguous(
            [number for number, _ in steps],
            result=result,
            code="step_sequence",
            path=_relative(episode_path, root),
            label="step",
        )

        records = [_StepRecord(number=number, path=step_path) for number, step_path in steps]
        for record in records:
            result.steps_checked += 1
            _validate_image(
                record.path / "cam_0.jpg",
                result,
                root,
                expected_size,
                fast=fast,
            )
            _validate_image(
                record.path / "cam_1.jpg",
                result,
                root,
                expected_size,
                fast=fast,
            )
            _validate_step_json(
                record,
                result=result,
                root=root,
                max_camera_skew_ms=max_camera_skew_ms,
                max_frame_age_ms=max_frame_age_ms,
            )

        v2_count = sum(record.v2 for record in records)
        if v2_count and v2_count != len(records):
            result.error(
                "mixed_schema",
                _relative(episode_path, root),
                "episode mixes legacy and v2 step metadata",
            )
        if v2_count:
            result.v2_episodes += 1
        else:
            result.legacy_episodes += 1
            if strict_v2:
                result.error(
                    "legacy_schema",
                    _relative(episode_path, root),
                    "strict v2 validation requires schema_version metadata",
                )

        for current, following in zip(records, records[1:]):
            if following.number != current.number + 1:
                continue
            _compare_action(
                current,
                following,
                result=result,
                root=root,
                tolerance=action_tolerance,
            )
            _validate_time_sequence(current, following, result=result, root=root)

        _validate_episode_meta(
            episode_path,
            episode_number,
            records,
            result=result,
            root=root,
            has_v2=bool(v2_count),
        )

    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a UF850 VR dataset (read-only)")
    parser.add_argument("path", type=Path, help="dataset directory containing episode_NNN")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--width", type=int, default=None, help="override expected image width")
    parser.add_argument("--height", type=int, default=None, help="override expected image height")
    parser.add_argument(
        "--action-tolerance", type=float, default=1e-5, help="absolute action tolerance"
    )
    parser.add_argument(
        "--max-camera-skew-ms", type=float, default=50.0, help="camera skew warning limit"
    )
    parser.add_argument(
        "--max-frame-age-ms", type=float, default=150.0, help="camera age warning limit"
    )
    parser.add_argument(
        "--strict-v2", action="store_true", help="reject legacy episodes without v2 metadata"
    )
    parser.add_argument(
        "--fast", action="store_true", help="check JPEG structure without fully decoding images"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = validate_dataset(
            args.path,
            expected_width=args.width,
            expected_height=args.height,
            action_tolerance=args.action_tolerance,
            max_camera_skew_ms=args.max_camera_skew_ms,
            max_frame_age_ms=args.max_frame_age_ms,
            strict_v2=args.strict_v2,
            fast=args.fast,
        )
    except ValueError as exc:
        print(f"validate: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        status = "PASS" if result.ok else "FAIL"
        print(
            f"[{status}] {result.dataset}: {result.episodes_checked} episode(s), "
            f"{result.steps_checked} step(s), {result.error_count} error(s), "
            f"{result.warning_count} warning(s)"
        )
        for issue in result.issues:
            print(f"{issue.severity.upper():7} {issue.code}: {issue.path}: {issue.message}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
