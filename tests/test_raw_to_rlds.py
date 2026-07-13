from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vrtool.raw_to_rlds import (
    ConversionError,
    binary_gripper,
    is_noop,
    scan_dataset,
    transformed_state_and_action,
    wrap_degrees,
)


def _step_payload(
    *,
    xyz=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    gripper=800.0,
) -> dict:
    return {
        "observations": {
            "ee_pos": [200.0, 300.0, 400.0, 180.0, 0.0, 0.0],
            "joint_pos": [0.0, 90.0, -90.0, 180.0, -180.0, 45.0],
            "gripper_pos": gripper,
        },
        "action": {
            "delta_ee_pos": [*xyz, *rotation],
            "delta_joint_pos": [0.0] * 6,
            "delta_gripper": 0.0,
        },
        "meta": {
            "step": 0,
            "delta_ee_rotation_wrapped": wrap_degrees(rotation).tolist(),
        },
    }


class TransformTests(unittest.TestCase):
    def test_wrap_degrees(self):
        actual = wrap_degrees([359.0, -359.0, 180.0, -180.0, 540.0])
        np.testing.assert_allclose(actual, [-1.0, 1.0, -180.0, -180.0, -180.0])

    def test_binary_gripper_sign_and_threshold(self):
        self.assertEqual(binary_gripper(0.0), 1.0)
        self.assertEqual(binary_gripper(425.0), 1.0)
        self.assertEqual(binary_gripper(425.001), -1.0)
        self.assertEqual(binary_gripper(850.0), -1.0)

    def test_state_and_action_units(self):
        payload = _step_payload(
            xyz=(10.0, -20.0, 30.0),
            rotation=(359.0, -359.0, 180.0),
            gripper=425.0,
        )
        state, action, gripper = transformed_state_and_action(
            payload, path=Path("step/data.json")
        )
        np.testing.assert_allclose(
            state,
            np.deg2rad([0.0, 90.0, -90.0, 180.0, -180.0, 45.0]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            action,
            [1.0, -2.0, 3.0, -np.pi / 180.0, np.pi / 180.0, -np.pi, 1.0],
            atol=1e-6,
        )
        self.assertEqual(gripper, 1.0)

    def test_rejects_inconsistent_wrapped_rotation(self):
        payload = _step_payload(rotation=(359.0, 0.0, 0.0))
        payload["meta"]["delta_ee_rotation_wrapped"] = [0.0, 0.0, 0.0]
        with self.assertRaises(ConversionError):
            transformed_state_and_action(payload, path=Path("bad.json"))

    def test_noop_filter_boundaries_and_gripper_change(self):
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0
        self.assertTrue(is_noop(action, previous_gripper=-1.0))

        action[0] = 0.02
        self.assertFalse(is_noop(action, previous_gripper=-1.0))

        action[:] = 0.0
        action[6] = 1.0
        self.assertFalse(is_noop(action, previous_gripper=-1.0))


class ScanTests(unittest.TestCase):
    def _write_episode(self, root: Path, payloads: list[dict]) -> None:
        episode = root / "episode_000"
        episode.mkdir(parents=True)
        (episode / "episode_meta.json").write_text(
            json.dumps({"task": "Do the test task."})
        )
        for index, payload in enumerate(payloads):
            step = episode / f"step_{index:05d}"
            step.mkdir()
            payload["meta"]["step"] = index
            (step / "data.json").write_text(json.dumps(payload))
            (step / "cam_0.jpg").write_bytes(b"present")
            (step / "cam_1.jpg").write_bytes(b"present")

    def test_scans_json_before_images_and_filters_noops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "task"
            root.mkdir()
            self._write_episode(
                root,
                [
                    _step_payload(),
                    _step_payload(xyz=(1.0, 0.0, 0.0)),
                    _step_payload(gripper=0.0),
                    _step_payload(gripper=0.0),
                ],
            )
            result = scan_dataset(root)
            self.assertEqual(result.raw_steps, 4)
            self.assertEqual(result.kept_steps, 2)
            self.assertEqual(result.dropped_steps, 2)
            self.assertEqual(result.fallback_episodes, 0)
            self.assertEqual(
                [step.source_index for step in result.episodes[0].kept_steps], [1, 2]
            )

    def test_falls_back_if_filter_would_leave_fewer_than_two_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "task"
            root.mkdir()
            self._write_episode(root, [_step_payload(), _step_payload(), _step_payload()])
            result = scan_dataset(root)
            self.assertEqual(result.kept_steps, 3)
            self.assertEqual(result.fallback_episodes, 1)


if __name__ == "__main__":
    unittest.main()
