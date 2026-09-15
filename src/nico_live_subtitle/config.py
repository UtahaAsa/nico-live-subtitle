from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, TypeVar


@dataclass
class AudioConfig:
    device: str | None = None
    sample_rate: int = 48_000
    block_ms: int = 100
    energy_threshold: float = 0.008
    silence_ms: int = 650
    partial_ms: int = 1_800
    max_utterance_ms: int = 12_000
    pre_roll_ms: int = 300


@dataclass
class RecognitionConfig:
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 1
    hotwords: str = ""


@dataclass
class TranslationConfig:
    backend: str = "google"
    packages_dir: str | None = None
    model_path: str | None = None
    context_lines: int = 2
    glossary: str = ""
    n_gpu_layers: int = -1


@dataclass
class OverlayConfig:
    font_size: int = 24
    opacity: float = 0.82
    max_lines: int = 3
    width: int = 1_000
    click_through: bool = False


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        if path is None:
            config = cls()
        else:
            config_path = Path(path)
            with config_path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
            if not isinstance(raw, Mapping):
                raise ValueError("配置文件顶层必须是 JSON 对象")
            config = cls(
                audio=_load_section(AudioConfig, raw.get("audio")),
                recognition=_load_section(RecognitionConfig, raw.get("recognition")),
                translation=_load_section(TranslationConfig, raw.get("translation")),
                overlay=_load_section(OverlayConfig, raw.get("overlay")),
            )
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        audio = self.audio
        if audio.sample_rate < 16_000:
            raise ValueError("audio.sample_rate 不能低于 16000")
        if not 20 <= audio.block_ms <= 1_000:
            raise ValueError("audio.block_ms 必须在 20 到 1000 之间")
        if not 0.0001 <= audio.energy_threshold <= 1.0:
            raise ValueError("audio.energy_threshold 必须在 0.0001 到 1.0 之间")
        if audio.silence_ms < audio.block_ms:
            raise ValueError("audio.silence_ms 不能小于 audio.block_ms")
        if audio.partial_ms < audio.block_ms:
            raise ValueError("audio.partial_ms 不能小于 audio.block_ms")
        if audio.max_utterance_ms <= audio.partial_ms:
            raise ValueError("audio.max_utterance_ms 必须大于 audio.partial_ms")
        if audio.pre_roll_ms < 0:
            raise ValueError("audio.pre_roll_ms 不能为负数")

        recognition = self.recognition
        if recognition.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("recognition.device 只能是 auto、cpu 或 cuda")
        if recognition.beam_size < 1:
            raise ValueError("recognition.beam_size 必须大于等于 1")

        translation = self.translation
        if translation.backend not in {"google", "argos", "hunyuan", "none"}:
            raise ValueError(
                "translation.backend 只能是 google、argos、hunyuan 或 none"
            )
        if not 0 <= translation.context_lines <= 8:
            raise ValueError("translation.context_lines 必须在 0 到 8 之间")
        if translation.n_gpu_layers < -1:
            raise ValueError("translation.n_gpu_layers 不能小于 -1")

        overlay = self.overlay
        if not 12 <= overlay.font_size <= 72:
            raise ValueError("overlay.font_size 必须在 12 到 72 之间")
        if not 0.2 <= overlay.opacity <= 1.0:
            raise ValueError("overlay.opacity 必须在 0.2 到 1.0 之间")
        if not 1 <= overlay.max_lines <= 8:
            raise ValueError("overlay.max_lines 必须在 1 到 8 之间")
        if not 480 <= overlay.width <= 3_840:
            raise ValueError("overlay.width 必须在 480 到 3840 之间")


SectionType = TypeVar(
    "SectionType", AudioConfig, RecognitionConfig, TranslationConfig, OverlayConfig
)


def _load_section(
    section_type: type[SectionType], raw: object | None
) -> SectionType:
    if raw is None:
        return section_type()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{section_type.__name__} 配置必须是 JSON 对象")

    allowed = set(section_type.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown:
        unknown_text = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"{section_type.__name__} 包含未知字段：{unknown_text}")
    return section_type(**dict(raw))
