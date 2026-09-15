from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol


class Translator(Protocol):
    def translate(self, text: str, context: Sequence[str] = ()) -> str: ...


class NoopTranslator:
    def translate(self, text: str, context: Sequence[str] = ()) -> str:
        return ""


class GoogleTranslator:
    def __init__(self) -> None:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator

        self._translator = DeepGoogleTranslator(source="ja", target="zh-CN")

    def translate(self, text: str, context: Sequence[str] = ()) -> str:
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

    def translate(self, text: str, context: Sequence[str] = ()) -> str:
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

    def translate(self, text: str, context: Sequence[str] = ()) -> str:
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
    text: str, context: Sequence[str] = (), glossary: str = ""
) -> str:
    sections: list[str] = []
    terms = _parse_glossary(glossary)
    if terms:
        term_lines = "\n".join(f"{source} 翻译成 {target}" for source, target in terms)
        sections.append(f"参考下面的翻译：\n{term_lines}")

    clean_context = [line.strip() for line in context if line.strip()]
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
) -> Translator:
    if backend == "none":
        return NoopTranslator()
    if backend == "google":
        return GoogleTranslator()
    if backend == "argos":
        return ArgosTranslator(packages_dir)
    if backend == "hunyuan":
        return HunyuanTranslator(model_path, glossary, n_gpu_layers)
    raise ValueError(f"不支持的翻译后端：{backend}")
