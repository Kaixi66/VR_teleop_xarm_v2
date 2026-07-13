from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrtool.validator import main, validate_dataset


def _write_jpeg(path: Path, width: int = 1920, height: int = 1080, components: int = 3) -> None:
    import cv2
    import numpy as np

    shape = (height, width) if components == 1 else (height, width, components)
    ok, encoded = cv2.imencode(".jpg", np.zeros(shape, dtype=np.uint8))
    if not ok:
        raise RuntimeError("test JPEG encoding failed")
    path.write_bytes(encoded.tobytes())


def _write_header_only_jpeg(path: Path, width: int = 1920, height: int = 1080) -> None:
    component_spec = b"".join(bytes((index + 1, 0x11, 0)) for index in range(3))
    sof_payload = bytes((8,)) + height.to_bytes(2, "big") + width.to_bytes(2, "big")
    sof_payload += bytes((3,)) + component_spec
    sof = b"\xff\xc0" + (len(sof_payload) + 2).to_bytes(2, "big") + sof_payload
    sos_payload = bytes((3,)) + b"\x01\x00\x02\x00\x03\x00\x00\x3f\x00"
    sos = b"\xff\xda" + (len(sos_payload) + 2).to_bytes(2, "big") + sos_payload
    path.write_bytes(b"\xff\xd8" + sof + sos + b"\x00\xff\xd9")


def _sample(
    step: int,
    value: float,
    next_value: float,
    *,
    v2: bool = False,
    monotonic_ns: int = 1_000_000_000,
    frame_number: int = 1,
) -> dict:
    delta = next_value - value
    meta = {"step": step, "total_gripper_moves": 0}
    if v2:
        meta.update(
            {
                "schema_version": "2.0",
                "sample_time_unix_ns": 1_700_000_000_000_000_000 + monotonic_ns,
                "sample_monotonic_ns": monotonic_ns,
                "dt_s": 0.1,
                "camera_timestamps_ms": [1000.0 + step, 1000.5 + step],
                "camera_frame_numbers": [frame_number, frame_number],
                "camera_host_monotonic_ns": [monotonic_ns - 10_000_000, monotonic_ns - 9_000_000],
                "camera_skew_ms": 1.0,
                "camera_max_age_ms": 10.0,
                "state_read_duration_ms": 3.0,
                "delta_ee_rotation_wrapped": [delta, delta, delta],
            }
        )
    return {
        "observations": {
            "ee_pos": [value] * 6,
            "joint_pos": [value] * 6,
            "gripper_pos": value,
        },
        "action": {
            "delta_ee_pos": [delta] * 6,
            "delta_joint_pos": [delta] * 6,
            "delta_gripper": delta,
        },
        "meta": meta,
    }


def _write_step(episode: Path, step: int, data: dict, *, width: int = 1920) -> Path:
    step_path = episode / f"step_{step:05d}"
    step_path.mkdir(parents=True)
    _write_jpeg(step_path / "cam_0.jpg", width=width)
    _write_jpeg(step_path / "cam_1.jpg", width=width)
    (step_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    return step_path


def _write_episode_meta(
    episode: Path, steps: int, *, width: int = 1920, height: int = 1080
) -> None:
    payload = {
        "schema_version": "2.0",
        "dataset": episode.parent.name,
        "episode_index": int(episode.name.split("_")[1]),
        "task": "validator test",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "ended_at_utc": "2026-01-01T00:00:01Z",
        "duration_s": 1.0,
        "termination_reason": "ctrl_c",
        "complete": True,
        "steps_written": steps,
        "samples_captured": steps + 1,
        "total_gripper_moves": 0,
        "cameras": [
            {"serial": "a", "width": width, "height": height},
            {"serial": "b", "width": width, "height": height},
        ],
        "config": {},
        "error_counts": {},
        "stats": {},
    }
    (episode / "episode_meta.json").write_text(json.dumps(payload), encoding="utf-8")


class ValidatorTests(unittest.TestCase):
    def test_default_decodes_jpeg_while_fast_mode_is_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            step = _write_step(episode, 0, _sample(0, 1.0, 2.0))
            _write_header_only_jpeg(step / "cam_0.jpg")

            self.assertIn("invalid_jpeg", {issue.code for issue in validate_dataset(root).errors})
            self.assertTrue(validate_dataset(root, fast=True).ok)

    def test_valid_legacy_dataset_and_action_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            _write_step(episode, 0, _sample(0, 1.0, 2.0))
            _write_step(episode, 1, _sample(1, 2.0, 99.0))

            result = validate_dataset(root)

            self.assertTrue(result.ok, result.to_dict())
            self.assertEqual(result.legacy_episodes, 1)
            self.assertEqual(result.v2_episodes, 0)
            self.assertEqual(result.steps_checked, 2)

            strict_result = validate_dataset(root, strict_v2=True)
            self.assertFalse(strict_result.ok)
            self.assertIn("legacy_schema", {issue.code for issue in strict_result.errors})

    def test_valid_v2_dataset_checks_episode_metadata_and_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            _write_step(
                episode,
                0,
                _sample(0, 1.0, 2.0, v2=True, monotonic_ns=1_000_000_000, frame_number=10),
            )
            _write_step(
                episode,
                1,
                _sample(1, 2.0, 3.0, v2=True, monotonic_ns=1_100_000_000, frame_number=13),
            )
            _write_episode_meta(episode, 2)

            result = validate_dataset(root)

            self.assertTrue(result.ok, result.to_dict())
            self.assertEqual(result.v2_episodes, 1)

    def test_v2_uses_declared_image_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            _write_step(
                episode,
                0,
                _sample(0, 1.0, 2.0, v2=True),
                width=64,
            )
            # Replace both images with the declared non-default size.
            _write_jpeg(episode / "step_00000" / "cam_0.jpg", width=64, height=48)
            _write_jpeg(episode / "step_00000" / "cam_1.jpg", width=64, height=48)
            _write_episode_meta(episode, 1, width=64, height=48)
            self.assertTrue(validate_dataset(root).ok)

    def test_v2_recomputes_camera_age_and_warns_on_frame_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            first = _sample(
                0, 1.0, 2.0, v2=True, monotonic_ns=1_000_000_000, frame_number=10
            )
            first["meta"]["camera_max_age_ms"] = 1.0
            _write_step(episode, 0, first)
            _write_step(
                episode,
                1,
                _sample(
                    1, 2.0, 3.0, v2=True, monotonic_ns=1_100_000_000, frame_number=10
                ),
            )
            _write_episode_meta(episode, 2)

            result = validate_dataset(root)
            self.assertIn("camera_age_value", {issue.code for issue in result.errors})
            self.assertIn("camera_frame_reuse", {issue.code for issue in result.warnings})

    def test_reports_sequences_files_shapes_finite_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_001"
            first = _sample(0, 1.0, 9.0)
            first["observations"]["joint_pos"][2] = math.inf
            step = _write_step(episode, 0, first, width=640)
            (step / "cam_1.jpg").unlink()
            (episode / "step_00002").mkdir()
            _write_jpeg(episode / "step_00002" / "cam_0.jpg")
            _write_jpeg(episode / "step_00002" / "cam_1.jpg")
            (episode / "step_00002" / "data.json").write_text("{broken", encoding="utf-8")

            result = validate_dataset(root)
            codes = {issue.code for issue in result.errors}

            self.assertFalse(result.ok)
            self.assertIn("episode_sequence", codes)
            self.assertIn("step_sequence", codes)
            self.assertIn("image_dimensions", codes)
            self.assertIn("missing_camera", codes)
            self.assertIn("observation_shape", codes)
            self.assertIn("invalid_json", codes)

    def test_reports_action_timestamps_skew_and_crash_residue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".episode_001.inprogress").mkdir()
            episode = root / "episode_000"
            first = _sample(0, 1.0, 2.0, v2=True, monotonic_ns=2_000_000_000, frame_number=10)
            first["action"]["delta_ee_pos"] = [7.0] * 6
            first["meta"]["dt_s"] = 0.1
            first["meta"]["camera_host_monotonic_ns"] = [1_900_000_000, 1_980_000_000]
            first["meta"]["camera_skew_ms"] = 80.0
            first["meta"]["camera_max_age_ms"] = 200.0
            _write_step(episode, 0, first)
            _write_step(
                episode,
                1,
                _sample(1, 2.0, 3.0, v2=True, monotonic_ns=1_900_000_000, frame_number=9),
            )
            (episode / ".step_00002.deadbeef.tmp").mkdir()
            _write_episode_meta(episode, 2)

            result = validate_dataset(root)
            error_codes = {issue.code for issue in result.errors}
            warning_codes = {issue.code for issue in result.warnings}

            self.assertIn("inprogress_episode", error_codes)
            self.assertIn("temporary_step", error_codes)
            self.assertIn("action_alignment", error_codes)
            self.assertIn("timestamp_order", error_codes)
            self.assertIn("camera_frame_order", error_codes)
            self.assertIn("camera_skew", warning_codes)
            self.assertIn("camera_frame_age", warning_codes)

    def test_cli_json_and_missing_dataset_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000"
            _write_step(episode, 0, _sample(0, 1.0, 2.0))
            output = io.StringIO()
            with redirect_stdout(output):
                return_code = main([str(root), "--json"])
            payload = json.loads(output.getvalue())
            self.assertEqual(return_code, 0)
            self.assertTrue(payload["ok"])

            with redirect_stdout(io.StringIO()):
                missing_return_code = main([str(root / "absent")])
            self.assertEqual(missing_return_code, 1)


if __name__ == "__main__":
    unittest.main()
