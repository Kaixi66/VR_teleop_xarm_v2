from __future__ import annotations

import json
import math
import os
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vrtool.bridge import (  # noqa: E402
    ZERO_COMMAND,
    BridgeConfig,
    CollectionLease,
    GripperToggle,
    PacketError,
    PeerLock,
    SafetyController,
    Workspace,
    arm_control_lock,
    _create_clean_error_call,
    _configure_gripper,
    _drain_udp,
    apply_deadband,
    apply_workspace_guard,
    issue_safe_stop,
    pack_feedback,
    parse_packet,
    populate_velocity_request,
    sanitize_axes,
    scale_axes,
    slew_limit,
    require_timed_velocity_firmware,
    _send_velocity_sync,
    _write_ready_file,
)


def _write_collection_lease(path: Path, pid: int, monotonic_s: float) -> None:
    path.write_text(
        json.dumps({"pid": pid, "monotonic_ns": int(monotonic_s * 1e9)}),
        encoding="utf-8",
    )


class PacketParsingTests(unittest.TestCase):
    def test_all_historical_packet_shapes(self) -> None:
        cases = {
            6: (None, ()),
            7: (6.0, ()),
            9: (6.0, (7.0, 8.0)),
            13: (6.0, (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)),
        }
        for count, (expected_gripper, expected_buttons) in cases.items():
            with self.subTest(bytes=count * 4):
                packet = parse_packet(struct.pack(f"<{count}f", *range(count)))
                self.assertEqual(packet.axes, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
                self.assertEqual(packet.gripper, expected_gripper)
                self.assertEqual(packet.buttons, expected_buttons)

    def test_unknown_packet_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(PacketError, "unsupported packet length"):
            parse_packet(b"\0" * 32)

    def test_non_finite_axis_or_unused_button_is_rejected(self) -> None:
        for bad_index in (0, 12):
            values = [0.0] * 13
            values[bad_index] = math.nan
            with self.subTest(index=bad_index), self.assertRaisesRegex(
                PacketError, "NaN or infinity"
            ):
                parse_packet(struct.pack("<13f", *values))


class ControlLockTests(unittest.TestCase):
    def test_second_arm_controller_cannot_take_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "arm_control.lock"
            with arm_control_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already owned"):
                    with arm_control_lock(lock_path):
                        self.fail("second controller unexpectedly acquired the lock")


class CollectionLeaseTests(unittest.TestCase):
    def make_lease(self, path: Path) -> CollectionLease:
        return CollectionLease(
            path=path,
            timeout_s=0.5,
            neutral_packets_required=2,
            gripper_threshold=0.5,
        )

    def test_missing_stale_and_malformed_lease_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.lease.json"
            lease = self.make_lease(path)

            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.0))

            _write_collection_lease(path, pid=123, monotonic_s=9.0)
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.0))

            path.write_text("not-json", encoding="utf-8")
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.0))
            self.assertFalse(lease.armed)
            self.assertIsNone(lease.collector_pid)
            self.assertEqual(lease.neutral_packets, 0)

    def test_small_future_heartbeat_race_is_fresh_but_large_future_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.lease.json"
            lease = self.make_lease(path)

            # The collector refreshed the file 10 ms after the bridge captured
            # the tick timestamp. This is a normal cross-process scheduling race.
            _write_collection_lease(path, pid=123, monotonic_s=10.01)
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.0))
            self.assertTrue(lease.observe(ZERO_COMMAND, 0.0, now=10.0))

            # A timestamp 100 ms ahead is not attributable to the normal race
            # and must still pause control.
            _write_collection_lease(path, pid=123, monotonic_s=10.10)
            self.assertFalse(lease.enabled(now=10.0))
            self.assertFalse(lease.armed)

    def test_new_collector_pid_requires_a_new_neutral_packet_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.lease.json"
            lease = self.make_lease(path)

            _write_collection_lease(path, pid=101, monotonic_s=10.0)
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.1))
            self.assertTrue(lease.observe(ZERO_COMMAND, 0.0, now=10.2))

            _write_collection_lease(path, pid=202, monotonic_s=10.2)
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.3))
            self.assertEqual(lease.collector_pid, 202)
            self.assertEqual(lease.neutral_packets, 1)
            self.assertTrue(lease.observe(ZERO_COMMAND, 0.0, now=10.4))

    def test_nonneutral_packet_resets_neutral_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.lease.json"
            lease = self.make_lease(path)
            _write_collection_lease(path, pid=123, monotonic_s=10.0)

            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.1))
            self.assertEqual(lease.neutral_packets, 1)
            self.assertFalse(lease.observe((0.1, 0, 0, 0, 0, 0), 0.0, now=10.2))
            self.assertEqual(lease.neutral_packets, 0)
            self.assertFalse(lease.observe(ZERO_COMMAND, 0.0, now=10.3))

    def test_paused_udp_still_announces_quest_but_suppresses_gripper(self) -> None:
        class FakeSocket:
            def __init__(self) -> None:
                self.packets = [
                    (
                        struct.pack("<7f", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                        ("10.0.0.9", 3000),
                    )
                ]

            def recvfrom(self, _size: int):
                if self.packets:
                    return self.packets.pop(0)
                raise BlockingIOError

        with tempfile.TemporaryDirectory() as temporary:
            missing_path = Path(temporary) / "collection.lease.json"
            lease = self.make_lease(missing_path)
            callback = Mock()
            arm = Mock()
            controller = SafetyController(
                BridgeConfig(workspace=Workspace(enabled=False))
            )
            controller.current = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            _drain_udp(
                FakeSocket(),
                PeerLock(),
                controller,
                GripperToggle(),
                arm,
                10.0,
                BridgeConfig(workspace=Workspace(enabled=False)),
                Mock(),
                callback,
                lease,
            )

            callback.assert_called_once_with("10.0.0.9", 3000, 28)
            arm.set_gripper_position.assert_not_called()
            self.assertEqual(controller.current, ZERO_COMMAND)
            self.assertIsNone(controller.last_packet_time)


class GripperCompatibilityTests(unittest.TestCase):
    class StandardGripper:
        def __init__(self):
            self.calls = []

        def set_gripper_enable(self, value):
            self.calls.append(("enable", value))
            return 0

        def set_gripper_mode(self, value):
            self.calls.append(("mode", value))
            return 0

        def set_gripper_speed(self, value):
            self.calls.append(("speed", value))
            return 0

    def test_standard_sdk_without_force_api_still_initializes(self) -> None:
        arm = self.StandardGripper()
        with self.assertLogs("vrtool.bridge", level="INFO") as messages:
            _configure_gripper(arm, BridgeConfig())
        self.assertEqual(
            arm.calls,
            [("enable", True), ("mode", 0), ("speed", 3000)],
        )
        self.assertIn("no set_gripper_force", "\n".join(messages.output))

    def test_custom_sdk_force_api_is_used_and_checked(self) -> None:
        class ForceGripper(self.StandardGripper):
            def set_gripper_force(self, value):
                self.calls.append(("force", value))
                return 0

        arm = ForceGripper()
        _configure_gripper(arm, BridgeConfig())
        self.assertEqual(arm.calls[-1], ("force", 1000))

        arm.set_gripper_force = lambda _value: 9
        with self.assertRaisesRegex(RuntimeError, "set_gripper_force"):
            _configure_gripper(arm, BridgeConfig())


class ConfigurationTests(unittest.TestCase):
    def test_project_toml_shape_is_supported(self) -> None:
        config = BridgeConfig.from_mapping(
            {
                "robot": {"ip": "10.0.0.9", "namespace": "ufactory"},
                "network": {
                    "bind_host": "127.0.0.1",
                    "listen_port": 6005,
                    "feedback_port": 6006,
                    "feedback_hz": 5.0,
                    "quest_ip": "10.0.0.4",
                },
                "control": {
                    "publish_hz": 50,
                    "linear_speed_mm_s": 50,
                    "angular_speed_rad_s": 0.2,
                    "command_duration_s": 0.15,
                    "deadband": 0.05,
                    "linear_accel_mm_s2": 500,
                    "angular_accel_rad_s2": 1.0,
                },
                "safety": {
                    "watchdog_s": 0.2,
                    "workspace_enabled": True,
                    "x_mm": [1, 2],
                    "y_mm": [3, 4],
                    "z_mm": [5, 6],
                },
            }
        )
        self.assertEqual(config.robot_ip, "10.0.0.9")
        self.assertEqual(config.ros_namespace, "/ufactory")
        self.assertEqual(config.listen_port, 6005)
        self.assertEqual(config.quest_ip, "10.0.0.4")
        self.assertEqual(config.feedback_interval_s, 0.2)
        self.assertEqual(config.linear_speed_mm_s, 50.0)
        self.assertEqual(config.workspace.bounds, ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))

    def test_empty_quest_ip_means_learn_first_valid_sender(self) -> None:
        config = BridgeConfig.from_mapping({"network": {"quest_ip": ""}})
        self.assertIsNone(config.quest_ip)

    def test_invalid_workspace_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "x_mm"):
            BridgeConfig.from_mapping({"safety": {"x_mm": [600, 0]}})


class CommandShapingTests(unittest.TestCase):
    def test_deadband_is_inclusive(self) -> None:
        self.assertEqual(apply_deadband(0.05, 0.05), 0.0)
        self.assertEqual(apply_deadband(-0.05, 0.05), 0.0)
        self.assertEqual(apply_deadband(0.051, 0.05), 0.051)

    def test_axes_are_finite_clamped_and_deadbanded(self) -> None:
        self.assertEqual(
            sanitize_axes((2.0, -2.0, 0.04, -0.05, 0.5, -0.5), 0.05),
            (1.0, -1.0, 0.0, 0.0, 0.5, -0.5),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            sanitize_axes((0, 0, 0, 0, 0, math.inf), 0.05)

    def test_linear_and_angular_axes_use_separate_gains(self) -> None:
        self.assertEqual(
            scale_axes((1, -1, 0.5, 1, -1, 0.5), 200, 0.3),
            (200.0, -200.0, 100.0, 0.3, -0.3, 0.15),
        )

    def test_slew_limit_uses_acceleration_and_elapsed_time(self) -> None:
        result = slew_limit(ZERO_COMMAND, (100, -100, 5, 1, -1, 0.2), 0.02, 800, 1.2)
        self.assertEqual(result[:3], (16.0, -16.0, 5.0))
        self.assertAlmostEqual(result[3], 0.024)
        self.assertAlmostEqual(result[4], -0.024)
        self.assertAlmostEqual(result[5], 0.024)

    def test_workspace_blocks_outward_but_allows_return_motion(self) -> None:
        workspace = Workspace()
        outward = apply_workspace_guard(
            (600, 0, 650), (25, -25, 25, 0.1, 0.2, 0.3), workspace, 0.15
        )
        self.assertEqual(outward[:3], (0.0, 0.0, 0.0))
        self.assertEqual(outward[3:], (0.1, 0.2, 0.3))
        returning = apply_workspace_guard(
            (610, -10, 660), (-25, 25, -25, 0, 0, 0), workspace, 0.15
        )
        self.assertEqual(returning[:3], (-25.0, 25.0, -25.0))

    def test_workspace_clips_command_before_crossing_boundary(self) -> None:
        result = apply_workspace_guard(
            (599, 300, 300), (200, 0, 0, 0, 0, 0), Workspace(), 0.15
        )
        self.assertAlmostEqual(result[0], 1.0 / 0.15)

    def test_disabled_workspace_does_not_need_a_valid_position(self) -> None:
        velocity = (1, 2, 3, 4, 5, 6)
        self.assertEqual(
            apply_workspace_guard((), velocity, Workspace(enabled=False)), velocity
        )


class SafetyControllerTests(unittest.TestCase):
    def config(self, **overrides: object) -> BridgeConfig:
        defaults: dict[str, object] = {
            "workspace": Workspace(enabled=False),
            "linear_accel_mm_s2": 1000.0,
            "angular_accel_rad_s2": 10.0,
        }
        defaults.update(overrides)
        return BridgeConfig(**defaults)

    def test_valid_input_is_slew_limited_at_publish_rate(self) -> None:
        controller = SafetyController(self.config())
        controller.ingest((1, 0, 0, 0, 0, 0), now=1.0)
        command = controller.step(now=1.0, position_mm=None)
        self.assertEqual(command[0], 20.0)  # 1000 mm/s^2 / 50 Hz

    def test_watchdog_bypasses_slew_and_stops_immediately(self) -> None:
        controller = SafetyController(self.config(watchdog_s=0.2))
        controller.ingest((1, 0, 0, 0, 0, 0), now=1.0)
        self.assertGreater(controller.step(1.0, None)[0], 0.0)
        self.assertEqual(controller.step(1.201, None), ZERO_COMMAND)
        self.assertEqual(controller.current, ZERO_COMMAND)

    def test_no_packet_is_always_zero(self) -> None:
        self.assertEqual(SafetyController(self.config()).step(10.0, None), ZERO_COMMAND)

    def test_workspace_enabled_fails_closed_without_position(self) -> None:
        controller = SafetyController(self.config(workspace=Workspace(enabled=True)))
        controller.ingest((1, 0, 0, 0, 0, 0), 1.0)
        self.assertEqual(controller.step(1.0, None), ZERO_COMMAND)

    def test_outward_residual_slew_is_also_blocked(self) -> None:
        controller = SafetyController(
            self.config(workspace=Workspace(enabled=True), command_duration_s=0.15)
        )
        controller.current = (100, 0, 0, 0, 0, 0)
        controller.ingest((0, 0, 0, 0, 0, 0), 1.0)
        self.assertEqual(controller.step(1.0, (600, 300, 300))[0], 0.0)


class StateHelpersTests(unittest.TestCase):
    def test_clean_error_uses_empty_call_service(self) -> None:
        class Call:
            class Request:
                pass

        node = Mock()
        client = object()
        node.create_client.return_value = client
        actual_client, request = _create_clean_error_call(
            node, "/ufactory", Call
        )
        node.create_client.assert_called_once_with(
            Call, "/ufactory/clean_error"
        )
        self.assertIs(actual_client, client)
        self.assertIsInstance(request, Call.Request)

    def test_timed_velocity_requires_firmware_1_8_or_newer(self) -> None:
        self.assertEqual(
            require_timed_velocity_firmware(SimpleNamespace(version_number=(1, 8, 0))),
            (1, 8, 0),
        )
        with self.assertRaisesRegex(RuntimeError, "1.8.0 or newer"):
            require_timed_velocity_firmware(SimpleNamespace(version_number=(1, 7, 99)))

    def test_ready_file_contains_current_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bridge.ready.json"
            _write_ready_file(path, quest_ip="10.42.0.8", packet_size=24)
            import json
            import os

            payload = json.loads(path.read_text())
            self.assertEqual(payload["pid"], os.getpid())
            self.assertEqual(payload["quest_ip"], "10.42.0.8")

    def test_only_first_valid_accepted_packet_announces_quest(self) -> None:
        class FakeSocket:
            def __init__(self):
                self.packets = [
                    (b"bad", ("10.0.0.9", 3000)),
                    (struct.pack("<6f", *([0.0] * 6)), ("10.0.0.8", 3001)),
                    (struct.pack("<6f", *([0.0] * 6)), ("10.0.0.9", 3002)),
                    (struct.pack("<6f", *([0.0] * 6)), ("10.0.0.9", 3003)),
                ]

            def recvfrom(self, _size):
                if self.packets:
                    return self.packets.pop(0)
                raise BlockingIOError

        config = BridgeConfig(workspace=Workspace(enabled=False))
        controller = SafetyController(config)
        callback = Mock()
        arm = Mock()
        _drain_udp(
            FakeSocket(),
            PeerLock("10.0.0.9"),
            controller,
            GripperToggle(),
            arm,
            1.0,
            config,
            Mock(),
            callback,
        )
        callback.assert_called_once_with("10.0.0.9", 3002, 24)
        self.assertEqual(controller.last_packet_time, 1.0)
        arm.set_gripper_position.assert_not_called()

    def test_synchronous_zero_waits_for_successful_service_response(self) -> None:
        class Request:
            pass

        future = Mock()
        future.done.return_value = True
        future.exception.return_value = None
        future.result.return_value = SimpleNamespace(ret=0, message="")
        client = Mock()
        client.wait_for_service.return_value = True
        client.call_async.return_value = future
        rclpy = Mock()
        _send_velocity_sync(
            rclpy,
            object(),
            client,
            SimpleNamespace(Request=Request),
            ZERO_COMMAND,
            0.15,
            1.0,
            "/ufactory/vc_set_cartesian_velocity(zero)",
        )
        request = client.call_async.call_args.args[0]
        self.assertEqual(request.speeds, [0.0] * 6)
        self.assertEqual(request.duration, 0.15)
        rclpy.spin_until_future_complete.assert_called_once()

    def test_peer_is_locked_to_first_valid_ip(self) -> None:
        peer = PeerLock()
        self.assertTrue(peer.accepts("10.0.0.1"))
        self.assertFalse(peer.accepts("10.0.0.2"))
        self.assertTrue(peer.accepts("10.0.0.1"))

    def test_configured_peer_never_accepts_another_ip(self) -> None:
        peer = PeerLock("10.0.0.9")
        self.assertFalse(peer.accepts("10.0.0.1"))
        self.assertTrue(peer.accepts("10.0.0.9"))

    def test_gripper_only_toggles_on_rising_edge(self) -> None:
        toggle = GripperToggle(threshold=0.5)
        self.assertIsNone(toggle.update(None))
        self.assertTrue(toggle.update(0.8))
        self.assertIsNone(toggle.update(0.9))
        self.assertIsNone(toggle.update(0.1))
        self.assertFalse(toggle.update(0.8))

    def test_velocity_request_sets_finite_duration_and_base_frame(self) -> None:
        request = SimpleNamespace()
        populate_velocity_request(request, (1, 2, 3, 4, 5, 6), 0.15)
        self.assertEqual(request.speeds, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertFalse(request.is_sync)
        self.assertFalse(request.is_tool_coord)
        self.assertEqual(request.duration, 0.15)

    def test_feedback_wire_format_is_four_little_endian_floats(self) -> None:
        payload = pack_feedback((100, 200, 300), True)
        self.assertEqual(len(payload), 16)
        decoded = struct.unpack("<ffff", payload)
        for actual, expected in zip(decoded, (0.1, 0.2, 0.3, 1.0)):
            self.assertAlmostEqual(actual, expected)

    def test_safe_stop_repeats_zero_then_enters_stop(self) -> None:
        calls: list[object] = []
        issue_safe_stop(lambda command: calls.append(tuple(command)), lambda: calls.append("STOP"), 3)
        self.assertEqual(calls, [ZERO_COMMAND, ZERO_COMMAND, ZERO_COMMAND, "STOP"])

    def test_safe_stop_attempts_stop_even_when_zero_send_fails(self) -> None:
        calls: list[str] = []

        def bad_send(_command: object) -> None:
            calls.append("zero")
            raise RuntimeError("transport down")

        with self.assertRaisesRegex(RuntimeError, "transport down"):
            issue_safe_stop(bad_send, lambda: calls.append("STOP"), 2)
        self.assertEqual(calls, ["zero", "zero", "STOP"])


if __name__ == "__main__":
    unittest.main()
