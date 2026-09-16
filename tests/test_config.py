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

    def test_hunyuan_translation_config_is_valid(self) -> None:
        config = AppConfig()
        config.translation.backend = "hunyuan"
        config.translation.model_path = "model.gguf"
        config.translation.context_lines = 3
        config.validate()

    def test_invalid_translation_context_is_rejected(self) -> None:
        config = AppConfig()
        config.translation.context_lines = 9
        with self.assertRaisesRegex(ValueError, "context_lines"):
            config.validate()

    def test_live_model_must_not_be_empty(self) -> None:
        config = AppConfig()
        config.recognition.live_model = ""
        with self.assertRaisesRegex(ValueError, "live_model"):
            config.validate()

    def test_silero_requires_32_ms_blocks(self) -> None:
        config = AppConfig()
        config.audio.vad_mode = "silero"
        with self.assertRaisesRegex(ValueError, "32"):
            config.validate()

    def test_lexicon_profile_uses_safe_identifier(self) -> None:
        config = AppConfig()
        config.lexicon.profile = "../outside"
        with self.assertRaisesRegex(ValueError, "lexicon.profile"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
