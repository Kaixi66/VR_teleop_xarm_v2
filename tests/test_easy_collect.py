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
    def test_launcher_uses_direct_settings_with_the_local_hotspot_password(self) -> None:
        # This launcher is the local robot-PC copy, so the lab hotspot password
        # is stored inline on purpose. Keep it a plain literal so the operator
        # can see and edit it, and keep the environment override working.
        source = (PROJECT_ROOT / "easy_collect").read_text(encoding="utf-8")
        self.assertIn('# ======================== USER SETTINGS ========================', source)
        self.assertIn('DATASET_NAME="put_blue_bowl_in_second_drawer"', source)
        self.assertIn('TASK_TEXT="put the blue bowl in the second drawer"', source)
        self.assertNotIn("user_settings.env", source)
        self.assertIsNotNone(
            re.search(r'^HOTSPOT_PASSWORD="[^"]{8,}"$', source, re.M),
            "launcher must carry the local hotspot password",
        )
        self.assertIn("VR_HOTSPOT_PASSWORD", source)

    def test_confirmation_accepts_s_in_either_case(self) -> None:
        for answer in ("s", "S", "start", "START"):
            with self.subTest(answer=answer):
                calls = self._run_real_launcher(answer)
                self.assertIn("collect", [call[0] for call in calls])
        for answer in ("", "n", "sta", "yes"):
            with self.subTest(answer=answer):
                calls = self._run_real_launcher(answer)
                self.assertNotIn("collect", [call[0] for call in calls])

    def test_reset_arm_runs_after_the_collector_exits(self) -> None:
        calls = self._run_real_launcher("s", with_reset=True)
        self.assertEqual([call[0] for call in calls][-2:], ["collect", "reset_arm"])

    def test_reset_arm_still_runs_when_the_collector_is_interrupted(self) -> None:
        # Ctrl+C reaches the collector; the launcher must survive to reset.
        calls = self._run_real_launcher("s", with_reset=True, collect_exit=130)
        self.assertEqual([call[0] for call in calls][-2:], ["collect", "reset_arm"])

    def test_missing_reset_arm_warns_instead_of_failing_the_run(self) -> None:
        calls, result = self._run_real_launcher("s", with_reset=False, want_result=True)
        self.assertEqual([call[0] for call in calls][-1], "collect")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reset script not found", result.stderr)

    def test_stop_key_ends_the_episode_without_any_signal(self) -> None:
        # The collector runs in its own session, so the launcher reads the key
        # and hands it over as a file.  Nothing is signalled, so no in-flight
        # image or metadata write can be interrupted.
        import pty
        import select

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
            capture = root / "calls.txt"
            (root / "vrctl").write_text(
                "#!/usr/bin/env bash\n"
                f'echo "$1" >> {capture}\n'
                "if [[ $1 == status ]]; then printf 'bridge RUNNING\\n'; fi\n"
                "if [[ $1 == collect ]]; then\n"
                '  echo "COLLECTOR RECORDING"\n'
                "  for i in $(seq 200); do\n"
                f"    [[ -f {root}/stopflag ]] && {{ echo 'COLLECTOR SAVED'; exit 0; }}\n"
                "    sleep 0.05\n"
                "  done\n"
                "  exit 9\n"
                "fi\n"
                f"if [[ $1 == stop-episode ]]; then touch {root}/stopflag; fi\n",
                encoding="utf-8",
            )
            (root / "vrctl").chmod(0o755)
            (root / "reset_arm").write_text(
                f'#!/usr/bin/env bash\necho reset_arm >> {capture}\n', encoding="utf-8"
            )
            (root / "reset_arm").chmod(0o755)

            pid, fd = pty.fork()
            if pid == 0:  # pragma: no cover - child replaces itself
                os.chdir(root)
                os.execv(str(launcher), [str(launcher)])
                os._exit(1)

            transcript = b""

            def pump(seconds: float) -> None:
                nonlocal transcript
                end = time.time() + seconds
                while time.time() < end:
                    ready, _, _ = select.select([fd], [], [], 0.05)
                    if ready:
                        try:
                            chunk = os.read(fd, 65536)
                        except OSError:
                            return
                        if not chunk:
                            return
                        transcript += chunk

            try:
                pump(1.5)
                os.write(fd, b"s\n")
                pump(3.0)
                self.assertIn(b"COLLECTOR RECORDING", transcript)
                os.write(fd, b"r")  # a bare keypress, no Enter
                pump(5.0)
                _, status = os.waitpid(pid, 0)
            finally:
                os.close(fd)

            self.assertEqual(os.waitstatus_to_exitcode(status), 0, transcript)
            self.assertIn(b"COLLECTOR SAVED", transcript)
            self.assertIn(b"[R pressed]", transcript)
            self.assertEqual(
                capture.read_text(encoding="utf-8").split(),
                ["doctor", "status", "collect", "stop-episode", "reset_arm"],
            )

    def _run_real_launcher(
        self,
        answer: str,
        *,
        with_reset: bool = False,
        collect_exit: int = 0,
        want_result: bool = False,
    ):
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
                "if [[ $1 == status ]]; then printf 'bridge RUNNING\\n'; fi\n"
                f"if [[ $1 == collect ]]; then exit {collect_exit}; fi\n",
                encoding="utf-8",
            )
            fake_vrctl.chmod(0o755)
            if with_reset:
                fake_reset = root / "reset_arm"
                fake_reset.write_text(
                    "#!/usr/bin/env bash\n"
                    "{ printf 'CALL'; printf '\\treset_arm'; printf '\\n'; } >> \"$CAPTURE\"\n",
                    encoding="utf-8",
                )
                fake_reset.chmod(0o755)
            result = subprocess.run(
                [str(launcher)],
                input=f"{answer}\n",
                text=True,
                capture_output=True,
                env=dict(os.environ, CAPTURE=str(capture)),
                timeout=10.0,
                check=False,
            )
            calls = [
                line.split("\t")[1:]
                for line in capture.read_text(encoding="utf-8").splitlines()
            ]
            return (calls, result) if want_result else calls

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
