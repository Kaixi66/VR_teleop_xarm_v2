from __future__ import annotations

import os
import tempfile
import unittest
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vrtool.config import load_config
from vrtool.cli import _parser as cli_parser
from vrtool.manager import (
    ManagerError,
    _read_start_ticks,
    _driver_launch_argv,
    _enable_hotspot,
    _hotspot_is_active,
    _interface_ipv4,
    _wait_for_bridge_ready,
    _wait_for_quest_ready,
    _write_state,
    load_state,
    prepare_quest,
    run_collector,
    stack_status,
    stop_stack,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG_PATH = PROJECT_ROOT / "config.example.toml"


class ManagerTests(unittest.TestCase):
    def test_hotspot_active_detection_and_idempotent_enable(self) -> None:
        hotspot = {
            "interface": "wifi0",
            "ssid": "MyRobotHotspot",
            "password_env": "VR_HOTSPOT_PASSWORD",
        }
        active = SimpleNamespace(returncode=0, stdout="Hotspot profile\n")
        details = SimpleNamespace(returncode=0, stdout="MyRobotHotspot\nap\n")
        with patch("vrtool.manager.subprocess.run", side_effect=[active, details]) as run:
            self.assertTrue(_hotspot_is_active(hotspot))
        self.assertEqual(run.call_count, 2)

        with patch("vrtool.manager._hotspot_is_active", return_value=True), patch(
            "vrtool.manager.subprocess.run"
        ) as run, patch("vrtool.manager.getpass.getpass") as password:
            _enable_hotspot({"hotspot": hotspot})
        run.assert_not_called()
        password.assert_not_called()

    def test_inactive_hotspot_uses_environment_password(self) -> None:
        config = {
            "hotspot": {
                "interface": "wifi0",
                "ssid": "MyRobotHotspot",
                "password_env": "VR_HOTSPOT_PASSWORD",
            }
        }
        result = SimpleNamespace(returncode=0)
        with patch("vrtool.manager._hotspot_is_active", return_value=False), patch.dict(
            os.environ, {"VR_HOTSPOT_PASSWORD": "secret123"}
        ), patch("vrtool.manager.subprocess.run", return_value=result) as run:
            _enable_hotspot(config)
        command = run.call_args.args[0]
        self.assertIn("hotspot", command)
        self.assertEqual(command[command.index("password") + 1], "secret123")

    def test_collect_and_run_cli_use_task_as_canonical_instruction(self) -> None:
        for command in ("collect", "run"):
            with self.subTest(command=command):
                args = cli_parser().parse_args(
                    [
                        command,
                        "--dataset",
                        "demo",
                        "--task",
                        "lift the object gently",
                    ]
                )
                self.assertEqual(args.task, "lift the object gently")

    def test_prepare_and_wait_quest_cli_flags(self) -> None:
        prepare = cli_parser().parse_args(["prepare", "--hotspot"])
        self.assertTrue(prepare.hotspot)
        run = cli_parser().parse_args(
            ["run", "--dataset", "demo", "--wait-quest"]
        )
        self.assertTrue(run.wait_quest)

    def test_driver_receives_configured_namespace(self) -> None:
        argv = _driver_launch_argv(
            {"ip": "10.0.0.2", "namespace": "/custom", "add_gripper": True}
        )
        self.assertIn("hw_ns:=custom", argv)

    def test_bridge_ready_marker_must_match_child_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready.json"
            log = Path(temporary) / "bridge.log"
            log.write_text("")
            ready.write_text(json.dumps({"pid": 1234}))
            process = SimpleNamespace(pid=1234, poll=lambda: None)
            _wait_for_bridge_ready(process, ready, log, timeout=0.1)

            ready.write_text(json.dumps({"pid": 9999}))
            with self.assertRaisesRegex(ManagerError, "belongs to pid"):
                _wait_for_bridge_ready(process, ready, log, timeout=0.1)

    def test_quest_ready_marker_requires_current_bridge_and_valid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "quest.ready.json"
            log = Path(temporary) / "bridge.log"
            log.write_text("")
            process = SimpleNamespace(pid=1234, poll=lambda: None)
            ready.write_text(
                json.dumps(
                    {"pid": 1234, "quest_ip": "10.42.0.8", "packet_size": 28}
                )
            )
            payload = _wait_for_quest_ready(process, ready, log, timeout=0.1)
            self.assertEqual(payload["quest_ip"], "10.42.0.8")

            ready.write_text(
                json.dumps(
                    {"pid": 9999, "quest_ip": "10.42.0.8", "packet_size": 28}
                )
            )
            with self.assertRaisesRegex(ManagerError, "belongs to pid"):
                _wait_for_quest_ready(process, ready, log, timeout=0.1)

            ready.write_text(
                json.dumps(
                    {"pid": 1234, "quest_ip": "10.42.0.8", "packet_size": 32}
                )
            )
            with self.assertRaisesRegex(ManagerError, "Invalid Quest"):
                _wait_for_quest_ready(process, ready, log, timeout=0.1)

    def test_hotspot_ipv4_is_parsed_without_cidr_suffix(self) -> None:
        result = SimpleNamespace(returncode=0, stdout="10.42.0.1/24\n")
        with patch("vrtool.manager.subprocess.run", return_value=result):
            self.assertEqual(_interface_ipv4("wifi0"), "10.42.0.1")

    def test_prepare_quest_only_prepares_network(self) -> None:
        config = load_config(TEST_CONFIG_PATH)
        with patch("vrtool.manager.stack_status", return_value=[]), patch(
            "vrtool.manager._enable_hotspot"
        ) as enable, patch(
            "vrtool.manager._show_hotspot_target", return_value="10.42.0.1"
        ) as show:
            result = prepare_quest(config, hotspot=True)
        enable.assert_called_once_with(config)
        show.assert_called_once_with(config)
        self.assertEqual(result["target_ip"], "10.42.0.1")
        self.assertEqual(result["listen_port"], 5005)

    def test_prepare_quest_rejects_an_active_stack(self) -> None:
        config = load_config(TEST_CONFIG_PATH)
        with patch(
            "vrtool.manager.stack_status",
            return_value=[("bridge", True, 123, "bridge.log")],
        ), self.assertRaisesRegex(ManagerError, "Stop the active"):
            prepare_quest(config, hotspot=True)

    def test_stop_retains_runtime_state_when_group_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(TEST_CONFIG_PATH)
            config["paths"]["runtime_dir"] = temporary
            entry = {"label": "bridge", "pid": 12, "pgid": 12, "start_ticks": 1}
            _write_state(config, {"processes": [entry]})
            with patch("vrtool.manager._stop_entries", return_value=[entry]):
                with self.assertRaisesRegex(ManagerError, "Runtime state was retained"):
                    stop_stack(config)
            self.assertEqual(load_state(config)["processes"], [entry])

    def test_collector_is_interrupted_and_exception_is_reraised(self) -> None:
        class ParentTermination(BaseException):
            pass

        config = load_config(TEST_CONFIG_PATH)
        process = Mock(pid=1234)
        process.wait.side_effect = [ParentTermination(), 0]
        with patch("vrtool.manager.subprocess.Popen", return_value=process), patch(
            "vrtool.manager.os.getpgid", return_value=4321
        ), patch("vrtool.manager.os.killpg") as killpg:
            with self.assertRaises(ParentTermination):
                run_collector(config, dataset="mock", mock=True)
        killpg.assert_called_once_with(4321, signal.SIGINT)

    def test_ctrl_c_finalizes_only_collector_and_keeps_stack_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(TEST_CONFIG_PATH)
            config["paths"]["runtime_dir"] = temporary
            process = Mock(pid=1234)
            process.wait.side_effect = [KeyboardInterrupt(), 0]
            healthy_stack = [
                ("description", True, 1001, "description.log"),
                ("driver", True, 1002, "driver.log"),
                ("bridge", True, 1003, "bridge.log"),
            ]
            with patch(
                "vrtool.manager.stack_status", return_value=healthy_stack
            ), patch(
                "vrtool.manager.subprocess.Popen", return_value=process
            ), patch(
                "vrtool.manager.os.getpgid", return_value=4321
            ), patch(
                "vrtool.manager.os.killpg"
            ) as killpg, patch(
                "vrtool.manager.stop_stack"
            ) as stop_stack:
                code = run_collector(config, dataset="demo")

        self.assertEqual(code, 0)
        killpg.assert_called_once_with(4321, signal.SIGINT)
        stop_stack.assert_not_called()

    def test_bridge_failure_aborts_collector_and_returns_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(TEST_CONFIG_PATH)
            config["paths"]["runtime_dir"] = temporary
            process = Mock(pid=1234)
            process.wait.side_effect = [
                subprocess.TimeoutExpired("collector", 0.1),
                1,
            ]
            healthy_stack = [
                ("description", True, 1001, "description.log"),
                ("driver", True, 1002, "driver.log"),
                ("bridge", True, 1003, "bridge.log"),
            ]
            failed_stack = [
                ("description", True, 1001, "description.log"),
                ("driver", True, 1002, "driver.log"),
                ("bridge", False, 1003, "bridge.log"),
            ]
            with patch(
                "vrtool.manager.stack_status",
                side_effect=[healthy_stack, failed_stack],
            ), patch(
                "vrtool.manager.subprocess.Popen", return_value=process
            ):
                code = run_collector(config, dataset="demo")

            self.assertEqual(code, 1)
            self.assertFalse((Path(temporary) / "collection.lease.json").exists())
            self.assertFalse((Path(temporary) / "collection.abort.txt").exists())

    def test_dataset_path_traversal_is_rejected_before_launch(self) -> None:
        config = load_config(TEST_CONFIG_PATH)
        with self.assertRaises(ManagerError):
            run_collector(config, dataset="../escape", mock=True)

    def test_collector_command_forwards_only_canonical_task(self) -> None:
        config = load_config(TEST_CONFIG_PATH)
        process = Mock(pid=1234)
        process.wait.return_value = 0
        with patch("vrtool.manager.subprocess.Popen", return_value=process) as popen:
            code = run_collector(
                config,
                dataset="mock",
                task="legacy task",
                mock=True,
            )
        self.assertEqual(code, 0)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--task") + 1], "legacy task")
        self.assertNotIn("--instruction", command)
        self.assertNotIn("--prompt", command)

    def test_runtime_state_uses_pid_start_time_to_avoid_pid_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(TEST_CONFIG_PATH)
            config["paths"]["runtime_dir"] = temporary
            ticks = _read_start_ticks(os.getpid())
            self.assertIsNotNone(ticks)
            state = {
                "processes": [
                    {
                        "label": "self-test",
                        "pid": os.getpid(),
                        "pgid": os.getpgid(os.getpid()),
                        "start_ticks": ticks,
                        "log": str(Path(temporary) / "test.log"),
                    }
                ]
            }
            _write_state(config, state)
            self.assertTrue(stack_status(config)[0][1])

            state["processes"][0]["start_ticks"] = int(ticks or 0) + 1
            _write_state(config, state)
            self.assertFalse(stack_status(config)[0][1])


if __name__ == "__main__":
    unittest.main()
