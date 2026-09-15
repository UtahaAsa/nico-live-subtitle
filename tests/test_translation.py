import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nico_live_subtitle.translation import (
    HunyuanTranslator,
    LocalLlmTranslator,
    OpenAICompatibleTranslator,
    build_hunyuan_prompt,
    clean_llm_output,
)


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
            [("こんにちは。", "你好。"), ("散歩に行きましょう。", "去散步吧。")],
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

    def test_local_llm_uses_paired_context_and_cleans_thinking(self) -> None:
        class ThinkingModel(FakeModel):
            def create_chat_completion(self, **kwargs: object) -> dict[str, object]:
                self.call_kwargs = kwargs
                return {
                    "choices": [
                        {"message": {"content": "<think>不要显示</think>\n自然译文"}}
                    ]
                }

        with tempfile.TemporaryDirectory(dir="work") as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            models: list[ThinkingModel] = []

            def factory(**kwargs: object) -> ThinkingModel:
                model = ThinkingModel(**kwargs)
                models.append(model)
                return model

            translator = LocalLlmTranslator(str(model_path), model_factory=factory)
            translated = translator.translate("次の台詞", [("前の台詞", "上一句")])

        messages = models[0].call_kwargs["messages"]
        self.assertEqual("自然译文", translated)
        self.assertEqual("前の台詞", messages[1]["content"])
        self.assertEqual("上一句", messages[2]["content"])

    def test_incomplete_thinking_is_hidden(self) -> None:
        self.assertEqual("", clean_llm_output("<think>处理中", allow_incomplete=True))

    def test_local_openai_compatible_endpoint_needs_no_secret(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeCompletions:
            def create(self, **kwargs: object) -> object:
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="兼容接口译文")
                        )
                    ]
                )

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.chat = SimpleNamespace(completions=FakeCompletions())

        translator = OpenAICompatibleTranslator(
            "http://127.0.0.1:11434/v1",
            "local-model",
            "MISSING_TEST_KEY",
            client_factory=FakeClient,
        )
        translated = translator.translate("こんにちは")
        self.assertEqual("兼容接口译文", translated)
        self.assertEqual("local-model", calls[0]["model"])


if __name__ == "__main__":
    unittest.main()
