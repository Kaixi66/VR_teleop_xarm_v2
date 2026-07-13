from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from vrtool.config import load_config
from vrtool.doctor import _run, run_doctor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG_PATH = PROJECT_ROOT / "config.example.toml"


class DoctorTests(unittest.TestCase):
    def test_command_timeout_and_missing_binary_become_results(self) -> None:
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["slow"], 1)):
            self.assertEqual(_run(["slow"], timeout=1).returncode, 124)
        with mock.patch("subprocess.run", side_effect=FileNotFoundError("missing")):
            self.assertEqual(_run(["missing"]).returncode, 127)

    def test_hotspot_check_uses_active_profile_without_wifi_scan(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            if "GENERAL.CONNECTION" in command:
                return subprocess.CompletedProcess(command, 0, "Hotspot\n", "")
            if "802-11-wireless.ssid,802-11-wireless.mode" in command:
                return subprocess.CompletedProcess(command, 0, "MyRobotHotspot\nap\n", "")
            if command[:2] == ["ss", "-lunH"]:
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 1, "", "unavailable")

        config = load_config(TEST_CONFIG_PATH)
        disk = mock.Mock(free=100 * 1024**3)
        with (
            mock.patch("vrtool.doctor._run", side_effect=fake_run),
            mock.patch("vrtool.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}"),
            mock.patch("vrtool.doctor.shutil.disk_usage", return_value=disk),
        ):
            report = run_doctor(config)

        hotspot = next(check for check in report.checks if check.name == "Quest hotspot")
        self.assertEqual(hotspot.level, "PASS")
        nmcli_commands = [command for command in commands if command[0].endswith("nmcli")]
        self.assertTrue(nmcli_commands)
        self.assertFalse(any("wifi" in command or "list" in command for command in nmcli_commands))


if __name__ == "__main__":
    unittest.main()
