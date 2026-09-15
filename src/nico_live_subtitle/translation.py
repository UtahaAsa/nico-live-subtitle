from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


TranslationContext = tuple[str, str]


class Translator(Protocol):
    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str: ...


class NoopTranslator:
    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        return ""


class GoogleTranslator:
    def __init__(self) -> None:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator

        self._translator = DeepGoogleTranslator(source="ja", target="zh-CN")

    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        translated = self._translator.translate(text)
        return translated.strip() if translated else ""


class ArgosTranslator:
    def __init__(self, packages_dir: str | None = None) -> None:
        if packages_dir:
            resolved_packages = Path(packages_dir).resolve()
            argos_root = resolved_packages.parent
            os.environ["ARGOS_PACKAGES_DIR"] = str(resolved_packages)
            os.environ.setdefault("XDG_DATA_HOME", str(argos_root / "data"))
            os.environ.setdefault("XDG_CONFIG_HOME", str(argos_root / "config"))
            os.environ.setdefault("XDG_CACHE_HOME", str(argos_root / "cache"))
        os.environ.setdefault("ARGOS_DEVICE_TYPE", "cpu")
        try:
            from argostranslate import translate
        except ImportError as error:
            raise RuntimeError(
                "未安装 Argos Translate；请安装项目的 local-translate 可选依赖"
            ) from error

        languages = translate.get_installed_languages()
        source = next((item for item in languages if item.code == "ja"), None)
        target = next((item for item in languages if item.code.startswith("zh")), None)
        if source is None or target is None:
            raise RuntimeError("Argos Translate 中未安装日语到中文语言包")
        try:
            self._translation = source.get_translation(target)
        except Exception as error:
            raise RuntimeError("Argos Translate 中未安装日语到中文语言包") from error

    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        return str(self._translation.translate(text)).strip()


class HunyuanTranslator:
    def __init__(
        self,
        model_path: str | None,
        glossary: str = "",
        n_gpu_layers: int = -1,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_path:
            raise RuntimeError("HY-MT 模型路径不能为空")
        resolved_model = Path(model_path).resolve()
        if not resolved_model.is_file():
            raise RuntimeError(f"找不到 HY-MT 模型：{resolved_model}")
        if model_factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise RuntimeError(
                    "未安装 llama-cpp-python；请按 README 安装 local-llm 可选依赖"
                ) from error
            model_factory = Llama
        self._model = model_factory(
            model_path=str(resolved_model),
            n_ctx=2_048,
            n_batch=256,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._glossary = glossary

    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        prompt = build_hunyuan_prompt(text, context, self._glossary)
        result = self._model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_k=20,
            top_p=0.6,
            repeat_penalty=1.05,
            max_tokens=256,
        )
        try:
            translated = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("HY-MT 返回了无法解析的结果") from error
        return str(translated).strip()


def build_hunyuan_prompt(
    text: str,
    context: Sequence[TranslationContext] = (),
    glossary: str = "",
) -> str:
    sections: list[str] = []
    terms = _parse_glossary(glossary)
    if terms:
        term_lines = "\n".join(f"{source} 翻译成 {target}" for source, target in terms)
        sections.append(f"参考下面的翻译：\n{term_lines}")

    clean_context = [source.strip() for source, _ in context if source.strip()]
    if clean_context:
        sections.append("\n".join(clean_context))
        instruction = (
            "参考上面的信息，把下面的文本翻译成简体中文，"
            "注意不需要翻译上文，也不要额外解释："
        )
    else:
        instruction = (
            "将以下文本翻译为简体中文，注意只需要输出翻译后的结果，"
            "不要额外解释："
        )
    sections.append(f"{instruction}\n\n{text.strip()}")
    return "\n\n".join(sections)


class LocalLlmTranslator:
    def __init__(
        self,
        model_path: str | None,
        glossary: str = "",
        n_gpu_layers: int = -1,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model_path:
            raise RuntimeError("本地 LLM 模型路径不能为空")
        resolved_model = Path(model_path).resolve()
        if not resolved_model.is_file():
            raise RuntimeError(f"找不到本地 LLM 模型：{resolved_model}")
        if model_factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as error:
                raise RuntimeError(
                    "未安装 llama-cpp-python；请按 README 安装 local-llm 可选依赖"
                ) from error
            model_factory = Llama
        self._model = model_factory(
            model_path=str(resolved_model),
            n_ctx=4_096,
            n_batch=256,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self._system_prompt = build_anime_system_prompt(glossary)

    def warmup(self) -> None:
        self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": "テスト /no_think"},
            ],
            temperature=0.2,
            max_tokens=1,
        )

    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        result = self._model.create_chat_completion(
            messages=self._messages(text, context),
            temperature=0.2,
            top_k=20,
            top_p=0.8,
            max_tokens=192,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("本地 LLM 返回了无法解析的结果") from error
        return clean_llm_output(str(content))

    def translate_iter(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> Any:
        stream = self._model.create_chat_completion(
            messages=self._messages(text, context),
            temperature=0.2,
            top_k=20,
            top_p=0.8,
            max_tokens=192,
            stream=True,
        )
        yield from _stream_cleaned_content(stream)

    def _messages(
        self, text: str, context: Sequence[TranslationContext]
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt}]
        for source, translated in context:
            if source and translated:
                messages.append({"role": "user", "content": source})
                messages.append({"role": "assistant", "content": translated})
        messages.append({"role": "user", "content": f"{text} /no_think"})
        return messages


class OpenAICompatibleTranslator:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key_env: str,
        glossary: str = "",
        timeout_sec: int = 20,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not api_base.strip() or not model.strip():
            raise RuntimeError("OpenAI 兼容接口地址和模型名称不能为空")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        hostname = (urlparse(api_base).hostname or "").casefold()
        if not api_key and hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"环境变量 {api_key_env} 未配置")
        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "未安装 openai；请安装项目的 llm-api 可选依赖"
                ) from error
            client_factory = OpenAI
        self._client = client_factory(
            base_url=api_base.rstrip("/"),
            api_key=api_key or "local",
            timeout=timeout_sec,
        )
        self._model = model
        self._system_prompt = build_anime_system_prompt(glossary)

    def translate(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(text, context),
            temperature=0.2,
            max_tokens=192,
        )
        return clean_llm_output(response.choices[0].message.content or "")

    def translate_iter(
        self, text: str, context: Sequence[TranslationContext] = ()
    ) -> Any:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(text, context),
            temperature=0.2,
            max_tokens=192,
            stream=True,
        )
        raw = ""
        last = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if not content:
                continue
            raw += content
            cleaned = clean_llm_output(raw, allow_incomplete=True)
            if cleaned and cleaned != last:
                last = cleaned
                yield cleaned

    def _messages(
        self, text: str, context: Sequence[TranslationContext]
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt}]
        for source, translated in context:
            if source and translated:
                messages.append({"role": "user", "content": source})
                messages.append({"role": "assistant", "content": translated})
        messages.append({"role": "user", "content": text})
        return messages


def build_anime_system_prompt(glossary: str = "") -> str:
    prompt = (
        "你是实时动画字幕翻译器。把日语台词翻译成自然、简洁、符合人物语气的"
        "简体中文。结合对话前文修正明显的语音识别错误。只输出当前台词的一种"
        "最佳译文，不解释、不提供备选、不输出思考过程。"
    )
    terms = _parse_glossary(glossary)
    if terms:
        term_lines = "；".join(f"{source} 必须译为 {target}" for source, target in terms)
        prompt += f"\n术语表：{term_lines}。"
    return prompt


def clean_llm_output(text: str, allow_incomplete: bool = False) -> str:
    cleaned = text.strip()
    if allow_incomplete and "<think>" in cleaned and "</think>" not in cleaned:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "").strip()
    return cleaned


def _stream_cleaned_content(stream: Any) -> Any:
    raw = ""
    last = ""
    for chunk in stream:
        try:
            content = chunk["choices"][0]["delta"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            continue
        raw += str(content)
        cleaned = clean_llm_output(raw, allow_incomplete=True)
        if cleaned and cleaned != last:
            last = cleaned
            yield cleaned


def _parse_glossary(glossary: str) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    for item in glossary.replace("，", ",").split(","):
        separator = "=" if "=" in item else "＝" if "＝" in item else None
        if separator is None:
            continue
        source, target = (part.strip() for part in item.split(separator, 1))
        if source and target:
            terms.append((source, target))
    return terms


def create_translator(
    backend: str,
    packages_dir: str | None = None,
    model_path: str | None = None,
    glossary: str = "",
    n_gpu_layers: int = -1,
    api_base: str = "http://127.0.0.1:11434/v1",
    api_model: str = "qwen3:8b",
    api_key_env: str = "NICO_SUBTITLE_API_KEY",
    timeout_sec: int = 20,
) -> Translator:
    if backend == "none":
        return NoopTranslator()
    if backend == "google":
        return GoogleTranslator()
    if backend == "argos":
        return ArgosTranslator(packages_dir)
    if backend == "hunyuan":
        return HunyuanTranslator(model_path, glossary, n_gpu_layers)
    if backend == "local_llm":
        return LocalLlmTranslator(model_path, glossary, n_gpu_layers)
    if backend == "openai_compatible":
        return OpenAICompatibleTranslator(
            api_base, api_model, api_key_env, glossary, timeout_sec
        )
    raise ValueError(f"不支持的翻译后端：{backend}")
