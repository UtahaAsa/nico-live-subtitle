import tempfile
import unittest
from pathlib import Path

from nico_live_subtitle.translation import HunyuanTranslator, build_hunyuan_prompt


class FakeModel:
    def __init__(self, **kwargs: object) -> None:
        self.init_kwargs = kwargs
        self.call_kwargs: dict[str, object] = {}

    def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.call_kwargs = kwargs
        return {"choices": [{"message": {"content": "  今天天气很好。  "}}]}


class HunyuanTranslatorTest(unittest.TestCase):
    def test_prompt_uses_context_only_as_reference(self) -> None:
        prompt = build_hunyuan_prompt(
            "今日はいい天気ですね。",
            ["こんにちは。", "散歩に行きましょう。"],
            "スバル=昴，エミリア＝爱蜜莉雅",
        )
        self.assertIn("スバル 翻译成 昴", prompt)
        self.assertIn("こんにちは。", prompt)
        self.assertIn("不需要翻译上文", prompt)
        self.assertTrue(prompt.endswith("今日はいい天気ですね。"))

    def test_translator_passes_official_sampling_parameters(self) -> None:
        with tempfile.TemporaryDirectory(dir="work") as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            models: list[FakeModel] = []

            def factory(**kwargs: object) -> FakeModel:
                model = FakeModel(**kwargs)
                models.append(model)
                return model

            translator = HunyuanTranslator(
                str(model_path), n_gpu_layers=-1, model_factory=factory
            )
            translated = translator.translate("今日はいい天気ですね。")

        self.assertEqual("今天天气很好。", translated)
        self.assertEqual(-1, models[0].init_kwargs["n_gpu_layers"])
        self.assertEqual(0.7, models[0].call_kwargs["temperature"])
        self.assertEqual(1.05, models[0].call_kwargs["repeat_penalty"])


if __name__ == "__main__":
    unittest.main()
