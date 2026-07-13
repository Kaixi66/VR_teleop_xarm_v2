from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from vrtool.reset_arm import (
    ResetError,
    ResetSettings,
    ensure_exclusive_control,
    find_external_controllers,
    install_signal_handlers,
    reset_lock,
    run_hardware_reset,
)


TARGET = (55.399232, 7.733498, -48.980042, -1.039517, -57.38115, -0.614669)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def settings(**changes) -> ResetSettings:
    values = dict(
        robot_ip="192.168.1.230",
        joint_angles_deg=TARGET,
        speed_deg_s=30.0,
        acceleration_deg_s2=100.0,
        timeout_s=15.0,
        pause_s=0.0,
        countdown_s=0.0,
        tolerance_deg=2.0,
        open_gripper=True,
        gripper_open_position=800,
        gripper_speed=1000,
        runtime_dir=Path("/tmp/reset-test"),
    )
    values.update(changes)
    return ResetSettings(**values)


class FakeArm:
    def __init__(
        self, _ip: str, *, failure: str | None = None, final=None,
        limited=False, state=0, error=0, warning=0,
    ):
        self.connected = True
        self.axis = 6
        self.device_type = 12
        self.is_850 = True
        self.failure = failure
        self.final = tuple(final or TARGET)
        self.limited = limited
        self.controller_state = state
        self.error = error
        self.warning = warning
        self.calls: list[tuple[str, object]] = []
        self._angle_reads = 0

    def _result(self, name, payload=None):
        self.calls.append((name, payload))
        return 7 if self.failure == name else 0

    def connect(self):
        self.calls.append(("connect", None))
        self.connected = True

    def disconnect(self):
        self.calls.append(("disconnect", None))
        self.connected = False

    def get_state(self):
        self.calls.append(("get_state", None))
        if self.failure == "get_state":
            return 7, None
        return 0, self.controller_state

    def get_err_warn_code(self):
        self.calls.append(("get_err_warn_code", None))
        if self.failure == "get_err_warn_code":
            return 7, None
        return 0, [self.error, self.warning]

    def get_servo_angle(self, **kwargs):
        self.calls.append(("get_servo_angle", kwargs))
        self._angle_reads += 1
        if self.failure == "get_servo_angle":
            return 7, None
        return 0, ([0.0] * 6 if self._angle_reads == 1 else list(self.final))

    def is_joint_limit(self, target, **kwargs):
        self.calls.append(("is_joint_limit", (target, kwargs)))
        if self.failure == "is_joint_limit":
            return 7, None
        return 0, self.limited

    def clean_error(self): return self._result("clean_error")
    def clean_warn(self): return self._result("clean_warn")
    def motion_enable(self, **kwargs): return self._result("motion_enable", kwargs)
    def set_mode(self, mode): return self._result("set_mode", mode)
    def set_state(self, state):
        result = self._result("set_state", state)
        if result == 0:
            self.controller_state = state
        return result
    def set_gripper_enable(self, enabled): return self._result("set_gripper_enable", enabled)
    def set_gripper_mode(self, mode): return self._result("set_gripper_mode", mode)
    def set_gripper_speed(self, speed): return self._result("set_gripper_speed", speed)
    def set_gripper_position(self, position, **kwargs):
        return self._result("set_gripper_position", (position, kwargs))
    def set_servo_angle(self, **kwargs): return self._result("set_servo_angle", kwargs)


class ResetSettingsTests(unittest.TestCase):
    def test_project_reset_configuration(self):
        from vrtool.config import load_config

        config = load_config(PROJECT_ROOT / "config.example.toml")
        parsed = ResetSettings.from_mapping(config)
        self.assertEqual(parsed.joint_angles_deg, TARGET)
        self.assertEqual(parsed.gripper_open_position, 800)
        self.assertEqual(parsed.speed_deg_s, 30.0)
        self.assertEqual(parsed.acceleration_deg_s2, 100.0)

    def test_invalid_configuration_is_rejected(self):
        base = {
            "robot": {"ip": "192.168.1.230"},
            "paths": {"runtime_dir": "/tmp/runtime"},
            "reset": {
                "joint_angles_deg": list(TARGET),
                "speed_deg_s": 30.0,
                "acceleration_deg_s2": 100.0,
                "timeout_s": 15.0,
                "pause_s": 2.0,
                "countdown_s": 3.0,
                "tolerance_deg": 2.0,
                "open_gripper": True,
                "gripper_open_position": 800,
                "gripper_speed": 1000,
            },
        }
        for key, value in (
            ("joint_angles_deg", [1, 2]),
            ("speed_deg_s", 31.0),
            ("acceleration_deg_s2", float("nan")),
            ("gripper_open_position", 851),
        ):
            candidate = {name: dict(section) for name, section in base.items()}
            candidate["reset"][key] = value
            with self.subTest(key=key), self.assertRaises(ResetError):
                ResetSettings.from_mapping(candidate)


class ResetExecutionTests(unittest.TestCase):
    def test_public_xarm_shape_without_is_850_uses_device_type_12(self):
        arm = FakeArm("ignored")
        del arm.is_850
        result = run_hardware_reset(
            settings(), arm_factory=lambda _ip: arm,
            sleeper=lambda _seconds: None, output=lambda _line: None,
        )
        self.assertEqual(result.max_error_deg, 0.0)

    def test_missing_joint_limit_api_fails_closed(self):
        arm = FakeArm("ignored")
        arm.is_joint_limit = None
        with self.assertRaisesRegex(ResetError, "unchecked motion"):
            run_hardware_reset(
                settings(), arm_factory=lambda _ip: arm,
                sleeper=lambda _seconds: None, output=lambda _line: None,
            )
        self.assertNotIn("motion_enable", [name for name, _ in arm.calls])

    def test_non_uf850_or_unhealthy_controller_is_rejected_before_motion(self):
        cases = [
            (dict(), "not verified"),
            (dict(error=23), "not healthy"),
            (dict(warning=11), "not healthy"),
            (dict(state=1), "not safe"),
        ]
        for arm_kwargs, message in cases:
            arm = FakeArm("ignored", **arm_kwargs)
            if not arm_kwargs:
                arm.is_850 = False
                arm.device_type = 9
            with self.subTest(arm_kwargs=arm_kwargs), self.assertRaisesRegex(ResetError, message):
                run_hardware_reset(
                    settings(), arm_factory=lambda _ip, arm=arm: arm,
                    sleeper=lambda _seconds: None, output=lambda _line: None,
                )
            self.assertNotIn("motion_enable", [name for name, _ in arm.calls])

    def test_success_has_expected_order_parameters_and_final_stop(self):
        arm = FakeArm("ignored")
        result = run_hardware_reset(
            settings(), arm_factory=lambda _ip: arm, sleeper=lambda _seconds: None, output=lambda _line: None
        )
        names = [name for name, _payload in arm.calls]
        self.assertLess(names.index("set_gripper_position"), names.index("set_servo_angle"))
        self.assertEqual(names[-3:], ["set_state", "set_gripper_enable", "disconnect"])
        move = next(payload for name, payload in arm.calls if name == "set_servo_angle")
        self.assertEqual(move["angle"], list(TARGET))
        self.assertEqual(move["speed"], 30.0)
        self.assertEqual(move["mvacc"], 100.0)
        self.assertFalse(move["is_radian"])
        self.assertFalse(move["wait"])
        self.assertNotIn("timeout", move)
        self.assertEqual(result.max_error_deg, 0.0)

    def test_each_setup_failure_stops_and_disconnects(self):
        for failure in (
            "get_state", "get_err_warn_code", "get_servo_angle", "is_joint_limit",
            "motion_enable", "set_mode", "set_state",
            "set_gripper_enable", "set_gripper_mode", "set_gripper_speed",
            "set_gripper_position", "set_servo_angle",
        ):
            arm = FakeArm("ignored", failure=failure)
            with self.subTest(failure=failure), self.assertRaises(ResetError):
                run_hardware_reset(
                    settings(), arm_factory=lambda _ip, arm=arm: arm,
                    sleeper=lambda _seconds: None, output=lambda _line: None,
                )
            self.assertEqual(
                [name for name, _ in arm.calls][-3:],
                ["set_state", "set_gripper_enable", "disconnect"],
            )

    def test_joint_limit_is_rejected_before_enabling_motion(self):
        arm = FakeArm("ignored", limited=True)
        with self.assertRaisesRegex(ResetError, "joint limits"):
            run_hardware_reset(
                settings(), arm_factory=lambda _ip: arm,
                sleeper=lambda _seconds: None, output=lambda _line: None,
            )
        names = [name for name, _ in arm.calls]
        self.assertNotIn("clean_error", names)
        self.assertEqual(names[-3:], ["set_state", "set_gripper_enable", "disconnect"])

    def test_verification_failure_stops_and_disconnects(self):
        arm = FakeArm("ignored", final=[0.0] * 6)
        class Clock:
            now = 0.0
            def __call__(self): return self.now
            def sleep(self, seconds): self.now += seconds
        clock = Clock()
        with self.assertRaisesRegex(ResetError, "timed out"):
            run_hardware_reset(
                settings(), arm_factory=lambda _ip: arm,
                sleeper=clock.sleep, clock=clock, output=lambda _line: None,
            )
        self.assertEqual(
            [name for name, _ in arm.calls][-3:],
            ["set_state", "set_gripper_enable", "disconnect"],
        )

    def test_sdk_exception_is_wrapped_after_stop_and_disconnect(self):
        arm = FakeArm("ignored")
        def fail(**_kwargs):
            raise RuntimeError("network lost")
        arm.motion_enable = fail
        with self.assertRaisesRegex(ResetError, "network lost"):
            run_hardware_reset(
                settings(), arm_factory=lambda _ip: arm,
                sleeper=lambda _seconds: None, output=lambda _line: None,
            )
        self.assertEqual(
            [name for name, _ in arm.calls][-3:],
            ["set_state", "set_gripper_enable", "disconnect"],
        )

    def test_keyboard_interrupt_stops_and_disconnects(self):
        arm = FakeArm("ignored")
        def interrupt(_seconds):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_hardware_reset(
                settings(countdown_s=1), arm_factory=lambda _ip: arm,
                sleeper=interrupt, output=lambda _line: None,
            )
        self.assertEqual(
            [name for name, _ in arm.calls][-3:],
            ["set_state", "set_gripper_enable", "disconnect"],
        )

    def test_keyboard_interrupt_inside_joint_command_stops_and_disconnects(self):
        arm = FakeArm("ignored")
        def interrupt_move(**_kwargs):
            arm.calls.append(("set_servo_angle", "interrupt"))
            raise KeyboardInterrupt
        arm.set_servo_angle = interrupt_move
        with self.assertRaises(KeyboardInterrupt):
            run_hardware_reset(
                settings(), arm_factory=lambda _ip: arm,
                sleeper=lambda _seconds: None, output=lambda _line: None,
            )
        self.assertEqual(
            [name for name, _ in arm.calls][-3:],
            ["set_state", "set_gripper_enable", "disconnect"],
        )


class ExclusivityTests(unittest.TestCase):
    def test_reset_and_stack_start_guards_are_race_free(self):
        from vrtool.manager import ManagerError, _stack_start_guard

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            config = {"paths": {"runtime_dir": str(runtime)}}
            with reset_lock(runtime):
                with self.assertRaisesRegex(ManagerError, "reset"):
                    with _stack_start_guard(config):
                        self.fail("stack startup unexpectedly entered during reset")
            with _stack_start_guard(config):
                with self.assertRaisesRegex(ResetError, "startup"):
                    with reset_lock(runtime):
                        self.fail("reset unexpectedly entered during stack startup")

    def test_reset_and_bridge_share_the_same_control_lock(self):
        from vrtool.bridge import arm_control_lock

        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            with arm_control_lock(runtime / "arm_control.lock"):
                with self.assertRaisesRegex(ResetError, "owns arm control"):
                    with reset_lock(runtime):
                        self.fail("reset unexpectedly acquired bridge control lock")

    def test_managed_or_external_controller_is_rejected(self):
        with self.assertRaisesRegex(ResetError, "managed"):
            ensure_exclusive_control(
                {}, status_fn=lambda _cfg: [("bridge", True, 123, "bridge.log")],
                controller_scan=lambda: [],
            )
        with self.assertRaisesRegex(ResetError, "another known"):
            ensure_exclusive_control(
                {}, status_fn=lambda _cfg: [],
                controller_scan=lambda: [(321, "python inference_oft_xarm.py")],
            )

    def test_proc_controller_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / "12345"
            process.mkdir()
            (process / "cmdline").write_bytes(b"python\0bridge_ghsun.py\0")
            self.assertEqual(find_external_controllers(root)[0][0], 12345)

    def test_signal_handlers_use_controlled_interrupt(self):
        with patch("vrtool.reset_arm.signal.signal") as install:
            install_signal_handlers()
        self.assertEqual(
            install.call_args_list,
            [
                call(signal.SIGINT, signal.default_int_handler),
                call(signal.SIGTERM, signal.default_int_handler),
                call(signal.SIGHUP, signal.default_int_handler),
            ],
        )


class ResetLauncherTests(unittest.TestCase):
    def _launcher_tree(self, root: Path) -> tuple[Path, Path, Path]:
        launcher = root / "reset_arm"
        launcher.write_text(
            (PROJECT_ROOT / "reset_arm").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        launcher.chmod(0o755)

        capture = root / "calls.txt"
        vrctl = root / "vrctl"
        vrctl.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'vrctl %s\\n' \"$1\" >> \"$CAPTURE\"\n"
            "if [[ $1 == status ]]; then printf 'bridge RUNNING\\n'; exit 0; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        vrctl.chmod(0o755)

        fake_python = root / "python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'python %s\\n' \"$*\" >> \"$CAPTURE\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        return launcher, fake_python, capture

    def test_paused_stack_is_stopped_automatically_before_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher, fake_python, capture = self._launcher_tree(Path(temporary))
            result = subprocess.run(
                [str(launcher)],
                text=True,
                capture_output=True,
                env=dict(
                    os.environ,
                    RESET_PYTHON=str(fake_python),
                    CAPTURE=str(capture),
                ),
                check=False,
            )
            calls = capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[:2], ["vrctl status", "vrctl stop"])
        self.assertTrue(calls[2].startswith("python -B -m vrtool.reset_arm"))
        self.assertIn("stopping it safely", result.stdout)

    def test_active_collector_prevents_stack_stop_and_reset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher, fake_python, capture = self._launcher_tree(root)
            runtime = root / ".runtime"
            runtime.mkdir()
            with (runtime / "collector.lock").open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [str(launcher)],
                    text=True,
                    capture_output=True,
                    env=dict(
                        os.environ,
                        RESET_PYTHON=str(fake_python),
                        CAPTURE=str(capture),
                    ),
                    check=False,
                )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("still recording or saving", result.stderr)
        self.assertFalse(capture.exists())


if __name__ == "__main__":
    unittest.main()
