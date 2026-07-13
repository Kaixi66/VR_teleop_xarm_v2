from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EasyCollectTests(unittest.TestCase):
    def test_public_launcher_uses_direct_settings_without_a_hardcoded_password(self) -> None:
        source = (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8")
        self.assertIn('# ======================== USER SETTINGS ========================', source)
        self.assertIn('DATASET_NAME="example_task"', source)
        self.assertIn('TASK_TEXT="Describe the task in one English sentence."', source)
        self.assertNotIn("user_settings.env", source)
        self.assertIsNone(
            re.search(r'HOTSPOT_PASSWORD="[0-9]{8,}"', source),
            "public launcher must not contain a numeric hotspot password",
        )

    def test_real_mode_prepares_starts_then_collects_with_persistent_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            launcher.write_text(
                (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n"
                "[hotspot]\nssid = \"LabQuestNet\"\n",
                encoding="utf-8",
            )
            capture = root / "args.txt"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\n"
                "{ printf 'CALL'; printf '\\t%s' \"$@\"; printf '\\n'; } >> \"$CAPTURE\"\n"
                "if [[ $1 == status ]]; then exit 1; fi\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            env = dict(os.environ, CAPTURE=str(capture))
            result = subprocess.run(
                [str(launcher)],
                input="START\n",
                text=True,
                capture_output=True,
                env=env,
                timeout=5.0,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                line.split("\t")[1:]
                for line in capture.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(calls[0], ["doctor"])
            self.assertEqual(calls[1], ["status"])
            self.assertEqual(calls[2], ["prepare", "--hotspot"])
            self.assertEqual(calls[3], ["start", "--hotspot"])
            self.assertEqual(calls[4][0], "collect")
            self.assertTrue(any(value.startswith("--dataset=") for value in calls[4]))
            self.assertTrue(any(value.startswith("--task=") for value in calls[4]))
            self.assertLess(result.stdout.index("Quest network is ready"), result.stdout.index("START"))
            self.assertIn("Quest Wi-Fi: LabQuestNet", result.stdout)
            self.assertIn("an already-open App may remain open", result.stdout)

    def test_real_mode_reuses_a_healthy_active_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            launcher.write_text(
                (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n", encoding="utf-8"
            )
            capture = root / "args.txt"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\n"
                "{ printf 'CALL'; printf '\\t%s' \"$@\"; printf '\\n'; } >> \"$CAPTURE\"\n"
                "if [[ $1 == status ]]; then printf 'bridge RUNNING\\n'; fi\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            result = subprocess.run(
                [str(launcher)],
                input="START\n",
                text=True,
                capture_output=True,
                env=dict(os.environ, CAPTURE=str(capture)),
                timeout=5.0,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [
                line.split("\t")[1:]
                for line in capture.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([call[0] for call in calls], ["doctor", "status", "collect"])
            self.assertIn("existing ROS/bridge session is healthy", result.stdout)

    def test_non_start_confirmation_never_runs_robot_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            launcher.write_text(
                (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n", encoding="utf-8"
            )
            capture = root / "args.txt"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$1\" >> \"$CAPTURE\"\n"
                "if [[ $1 == status ]]; then exit 1; fi\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            result = subprocess.run(
                [str(launcher)],
                input="not-start\n",
                text=True,
                capture_output=True,
                env=dict(os.environ, CAPTURE=str(capture)),
                timeout=5.0,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                ["doctor", "status", "prepare"],
            )

    def test_prepare_failure_never_runs_robot_stack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            launcher.write_text(
                (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n", encoding="utf-8"
            )
            capture = root / "args.txt"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$1\" >> \"$CAPTURE\"\n"
                "if [[ $1 == status || $1 == prepare ]]; then exit 1; fi\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            result = subprocess.run(
                [str(launcher)],
                text=True,
                capture_output=True,
                env=dict(os.environ, CAPTURE=str(capture)),
                timeout=5.0,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(
                capture.read_text(encoding="utf-8").splitlines(),
                ["doctor", "status", "prepare"],
            )

    def test_mock_mode_skips_doctor_hotspot_and_quest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            source = (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8")
            launcher.write_text(
                source.replace('MODE="real"', 'MODE="mock"', 1),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n", encoding="utf-8"
            )
            capture = root / "args.txt"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            result = subprocess.run(
                [str(launcher)],
                text=True,
                capture_output=True,
                env=dict(os.environ, CAPTURE=str(capture)),
                timeout=5.0,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[0], "collect")
            self.assertIn("--mock", arguments)
            self.assertNotIn("--hotspot", arguments)
            self.assertNotIn("--wait-quest", arguments)

    def test_parent_term_is_forwarded_to_passive_prepare(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "easy_collect"
            launcher.write_text(
                (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            launcher.chmod(0o755)
            (root / "config.toml").write_text(
                "[control]\nlinear_speed_mm_s = 50.0\n", encoding="utf-8"
            )
            child_pid_file = root / "prepare.pid"
            fake_vrctl = root / "vrctl"
            fake_vrctl.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $1 == status ]]; then exit 1; fi\n"
                "if [[ $1 == prepare ]]; then\n"
                "  printf '%s\\n' \"$$\" > \"$CHILD_PID_FILE\"\n"
                "  trap 'exit 0' INT TERM HUP\n"
                "  while true; do sleep 0.05; done\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            process = subprocess.Popen(
                [str(launcher)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, CHILD_PID_FILE=str(child_pid_file)),
            )
            deadline = time.monotonic() + 2.0
            while not child_pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(child_pid_file.exists(), "prepare child did not start")
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))

            os.kill(process.pid, signal.SIGTERM)
            _stdout, stderr = process.communicate(timeout=3.0)
            self.assertEqual(process.returncode, 143, stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
