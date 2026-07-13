from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vrtool.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG_PATH = PROJECT_ROOT / "config.example.toml"


class ConfigTests(unittest.TestCase):
    def test_public_config_template_has_camera_order(self) -> None:
        config = load_config(TEST_CONFIG_PATH)
        self.assertEqual([camera["name"] for camera in config["cameras"]], ["cam_0", "cam_1"])
        self.assertEqual(
            config["cameras"][0]["serial"], "REPLACE_WITH_WRIST_REALSENSE_SERIAL"
        )
        self.assertEqual(
            config["cameras"][1]["serial"], "REPLACE_WITH_EXTERNAL_REALSENSE_SERIAL"
        )
        self.assertEqual(config["recording"]["hz"], 10.0)
        self.assertEqual(config["network"]["quest_ready_timeout_s"], 15.0)
        self.assertEqual(
            config["paths"]["collector_python"],
            str(Path.home() / "robot_env/bin/python"),
        )

    def test_missing_sections_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.toml"
            path.write_text('[robot]\nip = "192.168.1.230"\n')
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_critical_schema_and_ranges_raise_config_error(self) -> None:
        original = TEST_CONFIG_PATH.read_text()
        cases = {
            "bad control type": ('publish_hz = 50.0', 'publish_hz = "fast"'),
            "bad port": ("listen_port = 5005", "listen_port = 70000"),
            "bad Quest timeout": (
                "quest_ready_timeout_s = 15.0",
                "quest_ready_timeout_s = 0.0",
            ),
            "bad deadband": ("deadband = 0.05", "deadband = 1.0"),
            "bad frame buffer": ("frame_buffer_size = 4", "frame_buffer_size = 1"),
            "unsupported format": ('pixel_format = "yuyv"', 'pixel_format = "rgb8"'),
            "empty path": ('collector_python = "~/robot_env/bin/python"', 'collector_python = ""'),
        }
        for label, (old, new) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.toml"
                path.write_text(original.replace(old, new, 1))
                with self.assertRaises(ConfigError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
