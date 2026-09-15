from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class NoopTranslator:
    def translate(self, text: str) -> str:
        return ""


class GoogleTranslator:
    def __init__(self) -> None:
        from deep_translator import GoogleTranslator as DeepGoogleTranslator

        self._translator = DeepGoogleTranslator(source="ja", target="zh-CN")

    def translate(self, text: str) -> str:
        translated = self._translator.translate(text)
        return translated.strip() if translated else ""


class ArgosTranslator:
    def __init__(self) -> None:
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

    def translate(self, text: str) -> str:
        return str(self._translation.translate(text)).strip()


def create_translator(backend: str) -> Translator:
    if backend == "none":
        return NoopTranslator()
    if backend == "google":
        return GoogleTranslator()
    if backend == "argos":
        return ArgosTranslator()
    raise ValueError(f"不支持的翻译后端：{backend}")

