from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import call, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vrtool.collector import (  # noqa: E402
    AtomicEpisodeWriter,
    CameraFrame,
    CollectionError,
    CollectorConfig,
    FrameBuffer,
    RobotReadError,
    RobotState,
    Sample,
    StorageError,
    TeleopCollector,
    XArmStateSource,
    build_step_data,
    next_episode_index,
    install_collection_signal_handlers,
    select_synchronized_frames,
    validate_dataset_name,
    _StepPayload,
    _argument_parser,
)


def _frame(camera: int, stamp_ns: int, number: int = 1, image=None) -> CameraFrame:
    return CameraFrame(
        camera_index=camera,
        image=image,
        host_monotonic_ns=stamp_ns,
        host_wall_time_ns=stamp_ns + 100,
        device_timestamp_ms=float(number) * 10.0,
        frame_number=number,
    )


def _state(stamp_ns: int, ee, joints, gripper: float) -> RobotState:
    return RobotState(
        ee_pos=tuple(ee),
        joint_pos=tuple(joints),
        gripper_pos=gripper,
        sample_monotonic_ns=stamp_ns,
        sample_time_unix_ns=stamp_ns + 1_000_000_000,
        read_duration_ms=2.5,
    )


class ConfigTests(unittest.TestCase):
    def test_collector_cli_uses_task_as_canonical_instruction(self):
        args = _argument_parser().parse_args(
            [
                "--config",
                "config.toml",
                "--dataset",
                "demo",
                "--task",
                "open the drawer using the left handle",
            ]
        )
        self.assertEqual(args.task, "open the drawer using the left handle")

    def test_collection_signals_use_python_controlled_interrupt_handler(self):
        with patch("vrtool.collector.signal.signal") as install:
            install_collection_signal_handlers()
        expected = [
            call(signal.SIGINT, signal.default_int_handler),
            call(signal.SIGTERM, signal.default_int_handler),
        ]
        if hasattr(signal, "SIGHUP"):
            expected.append(call(signal.SIGHUP, signal.default_int_handler))
        self.assertEqual(install.call_args_list, expected)

    def test_from_project_mapping_and_relative_output(self):
        config = CollectorConfig.from_mapping(
            {
                "robot": {"ip": "10.0.0.9"},
                "recording": {
                    "hz": 12,
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                    "output_root": "captures",
                    "save_queue_size": 4,
                    "warmup_frames": 12,
                    "failure_timeout_s": 2.0,
                },
                "cameras": [
                    {"serial": "wrist-sn", "role": "wrist"},
                    {"serial": "room-sn", "role": "external"},
                ],
            },
            base_dir="/tmp/config-dir",
        )
        self.assertEqual(config.robot_ip, "10.0.0.9")
        self.assertEqual(config.camera_serials, ("wrist-sn", "room-sn"))
        self.assertEqual(config.data_root, Path("/tmp/config-dir/captures"))
        self.assertEqual(config.hz, 12)
        self.assertEqual(config.save_queue_size, 4)
        self.assertEqual(config.warmup_frames, 12)
        self.assertEqual(config.consecutive_failure_limit, 24)

    def test_dataset_name_cannot_escape_data_root(self):
        for invalid in ("../escape", "nested/name", ".hidden", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_dataset_name(invalid)


class SynchronizationTests(unittest.TestCase):
    def test_frame_history_is_bounded(self):
        buffer = FrameBuffer(maxlen=4)
        for number in range(1000):
            buffer.append(_frame(0, number, number=number))
        snapshot = buffer.snapshot()
        self.assertEqual(len(snapshot), 4)
        self.assertEqual([frame.frame_number for frame in snapshot], [996, 997, 998, 999])

    def test_selects_freshest_pair_within_skew(self):
        target = 1_000_000_000
        selection = select_synchronized_frames(
            [
                [_frame(0, target - 80_000_000, 1), _frame(0, target - 10_000_000, 2)],
                [_frame(1, target - 75_000_000, 1), _frame(1, target - 15_000_000, 2)],
            ],
            target,
            max_skew_ms=10,
            max_age_ms=100,
        )
        self.assertIsNotNone(selection)
        assert selection is not None
        self.assertEqual([frame.frame_number for frame in selection.frames], [2, 2])
        self.assertEqual(selection.skew_ms, 5.0)
        self.assertEqual(selection.max_age_ms, 15.0)

    def test_rejects_stale_or_unsynchronized_frames(self):
        target = 1_000_000_000
        self.assertIsNone(
            select_synchronized_frames(
                [[_frame(0, target - 200_000_000)], [_frame(1, target - 10_000_000)]],
                target,
                max_skew_ms=50,
                max_age_ms=150,
            )
        )
        self.assertIsNone(
            select_synchronized_frames(
                [[_frame(0, target - 80_000_000)], [_frame(1, target - 10_000_000)]],
                target,
                max_skew_ms=50,
                max_age_ms=150,
            )
        )


class SchemaTests(unittest.TestCase):
    def test_preserves_legacy_delta_and_adds_wrapped_rotation(self):
        previous = Sample(
            robot=_state(1_000_000_000, [1, 2, 3, 179, 0, 90], [0] * 6, 850),
            frames=(_frame(0, 990_000_000), _frame(1, 995_000_000)),
            camera_skew_ms=5.0,
            camera_max_age_ms=10.0,
        )
        current = Sample(
            robot=_state(1_100_000_000, [2, 4, 6, -179, 1, 89], [1] * 6, 800),
            frames=(_frame(0, 1_090_000_000), _frame(1, 1_095_000_000)),
            camera_skew_ms=5.0,
            camera_max_age_ms=10.0,
        )
        data = build_step_data(previous, current, step=7, total_gripper_moves=2)

        self.assertEqual(data["observations"]["ee_pos"], [1, 2, 3, 179, 0, 90])
        self.assertEqual(data["action"]["delta_ee_pos"], [1, 2, 3, -358, 1, -1])
        self.assertEqual(data["action"]["delta_joint_pos"], [1] * 6)
        self.assertEqual(data["action"]["delta_gripper"], -50)
        self.assertEqual(data["meta"]["delta_ee_rotation_wrapped"], [2.0, 1.0, -1.0])
        self.assertAlmostEqual(data["meta"]["dt_s"], 0.1)
        self.assertEqual(data["meta"]["step"], 7)
        self.assertEqual(data["meta"]["schema_version"], "2.0")

    def test_xarm_return_codes_are_not_ignored(self):
        with self.assertRaises(RobotReadError):
            XArmStateSource._checked((3, None), "get_position")
        with self.assertRaises(RobotReadError):
            XArmStateSource._checked(None, "get_position")


class EpisodeTests(unittest.TestCase):
    def test_collection_lease_is_written_for_this_process_and_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            lease_path = Path(temporary) / "run" / "collection.lease.json"
            collector = TeleopCollector(
                CollectorConfig(width=16, height=12, min_free_gb=0),
                "lease-test",
                mock=True,
                collection_lease_file=lease_path,
            )

            with patch("vrtool.collector.time.monotonic_ns", return_value=123456789):
                collector._refresh_collection_lease()

            payload = json.loads(lease_path.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"pid": os.getpid(), "monotonic_ns": 123456789})
            self.assertFalse(list(lease_path.parent.glob(".*.tmp")))

            collector._release_collection_lease()
            self.assertFalse(lease_path.exists())

    def test_writer_close_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = CollectorConfig(
                data_root=Path(temporary), save_queue_size=1, min_free_gb=0
            )
            writer = AtomicEpisodeWriter(config, "timeout")
            entered = threading.Event()
            release = threading.Event()

            def blocked_write(payload, cv2):
                entered.set()
                release.wait(2.0)

            writer._write_step = blocked_write
            writer.start()
            writer.submit(_StepPayload(0, (None, None), {}))
            self.assertTrue(entered.wait(1.0))
            started = time.monotonic()
            with self.assertRaisesRegex(StorageError, "shutdown timed out"):
                writer.close(timeout_s=0.05)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(writer._thread.daemon)
            release.set()
            writer._thread.join(1.0)

    def test_index_reserves_complete_and_in_progress_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "episode_002").mkdir()
            (root / ".episode_004.inprogress").mkdir()
            (root / "not-an-episode").mkdir()
            self.assertEqual(next_episode_index(root), 5)

    def test_mock_episode_is_atomic_and_old_schema_readable(self):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("mock end-to-end test needs cv2 and numpy")

        with tempfile.TemporaryDirectory() as temporary:
            config = CollectorConfig(
                data_root=Path(temporary),
                camera_serials=("mock-wrist", "mock-room"),
                camera_roles=("wrist", "external"),
                hz=20,
                width=32,
                height=24,
                camera_fps=60,
                frame_buffer_size=4,
                max_camera_skew_ms=100,
                max_frame_age_ms=500,
                save_queue_size=16,
                jpeg_quality=80,
                min_free_gb=0,
                startup_timeout_s=2,
            )
            result = TeleopCollector(
                config,
                "unit-test",
                task="抓取蓝色方块并放进托盘",
                mock=True,
            ).run(max_steps=3)

            self.assertTrue(result.complete)
            self.assertEqual(result.steps_written, 3)
            self.assertEqual(result.samples_captured, 4)
            self.assertEqual(result.path.name, "episode_000")
            self.assertFalse(result.path.with_name(".episode_000.inprogress").exists())

            steps = sorted(result.path.glob("step_*"))
            self.assertEqual([path.name for path in steps], [
                "step_00000",
                "step_00001",
                "step_00002",
            ])
            for path in steps:
                self.assertTrue((path / "cam_0.jpg").is_file())
                self.assertTrue((path / "cam_1.jpg").is_file())
                with (path / "data.json").open(encoding="utf-8") as handle:
                    data = json.load(handle)
                self.assertEqual(set(data), {"observations", "action", "meta"})
                self.assertNotIn("instruction", data["meta"])
                self.assertNotIn("prompt", data["meta"])
                self.assertEqual(len(data["observations"]["ee_pos"]), 6)
                self.assertEqual(len(data["action"]["delta_joint_pos"]), 6)

            with (result.path / "episode_meta.json").open(encoding="utf-8") as handle:
                episode_meta = json.load(handle)
            self.assertTrue(episode_meta["complete"])
            self.assertEqual(episode_meta["termination_reason"], "max_steps")
            self.assertEqual(episode_meta["task"], "抓取蓝色方块并放进托盘")
            self.assertNotIn("instruction", episode_meta)
            self.assertNotIn("prompt", episode_meta)
            self.assertIn(
                "抓取蓝色方块并放进托盘",
                (result.path / "episode_meta.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(episode_meta["steps_written"], 3)
            self.assertFalse(list(result.path.glob(".step_*.tmp")))

    def test_task_defaults_to_empty_string(self):
        collector = TeleopCollector(
            CollectorConfig(width=16, height=12, min_free_gb=0),
            "defaults",
            mock=True,
        )
        self.assertEqual(collector.task, "")

    def test_empty_stopped_episode_is_not_finalized(self):
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("mock end-to-end test needs cv2 and numpy")

        with tempfile.TemporaryDirectory() as temporary:
            stop = threading.Event()
            stop.set()
            config = CollectorConfig(
                data_root=Path(temporary),
                width=16,
                height=12,
                min_free_gb=0,
                startup_timeout_s=1,
            )
            with self.assertRaisesRegex(CollectionError, "no complete"):
                TeleopCollector(config, "empty", mock=True, stop_event=stop).run()
            self.assertFalse((Path(temporary) / "empty" / "episode_000").exists())
            self.assertTrue(
                (Path(temporary) / "empty" / ".episode_000.inprogress").exists()
            )


if __name__ == "__main__":
    unittest.main()
