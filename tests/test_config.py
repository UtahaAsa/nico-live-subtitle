import json
import tempfile
import unittest
from pathlib import Path

from nico_live_subtitle.config import AppConfig


class AppConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = AppConfig.load(None)
        self.assertEqual("small", config.recognition.model)
        self.assertEqual("google", config.translation.backend)

    def test_partial_config_keeps_defaults(self) -> None:
        with tempfile.TemporaryDirectory(dir="work") as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"translation": {"backend": "none"}}),
                encoding="utf-8",
            )
            config = AppConfig.load(path)
        self.assertEqual("none", config.translation.backend)
        self.assertEqual(48_000, config.audio.sample_rate)

    def test_unknown_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="work") as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"audio": {"surprise": True}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "未知字段"):
                AppConfig.load(path)

    def test_invalid_timing_is_rejected(self) -> None:
        config = AppConfig()
        config.audio.max_utterance_ms = config.audio.partial_ms
        with self.assertRaisesRegex(ValueError, "max_utterance_ms"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
