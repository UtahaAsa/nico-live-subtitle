from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RecognitionConfig


@dataclass(frozen=True)
class RecognitionRuntime:
    device: str
    compute_type: str


class JapaneseRecognizer:
    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config
        runtime = resolve_runtime(config)

        from faster_whisper import WhisperModel

        self._model_type = WhisperModel
        try:
            self._model = self._model_type(
                config.model,
                device=runtime.device,
                compute_type=runtime.compute_type,
            )
        except Exception:
            if config.device != "auto" or runtime.device != "cuda":
                raise
            runtime = RecognitionRuntime(device="cpu", compute_type="int8")
            self._model = self._model_type(
                config.model,
                device=runtime.device,
                compute_type=runtime.compute_type,
            )
        self.runtime = runtime

    def transcribe(self, samples: np.ndarray) -> str:
        try:
            return self._transcribe_once(samples)
        except RuntimeError:
            # CTranslate2 可能检测到 NVIDIA 显卡，但直到第一次推理才发现
            # CUDA/cuDNN DLL 不完整。auto 模式在这里安全地重试 CPU；显式
            # 选择 cuda 时仍保留原始错误，避免违背用户设置。
            if self._config.device != "auto" or self.runtime.device != "cuda":
                raise
            self.runtime = RecognitionRuntime(device="cpu", compute_type="int8")
            self._model = self._model_type(
                self._config.model,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
            )
            return self._transcribe_once(samples)

    def _transcribe_once(self, samples: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            language="ja",
            beam_size=self._config.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
            hotwords=self._config.hotwords or None,
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return "".join(parts)


def resolve_runtime(config: RecognitionConfig) -> RecognitionRuntime:
    device = config.device
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"

    if config.compute_type != "auto":
        compute_type = config.compute_type
    else:
        compute_type = "float16" if device == "cuda" else "int8"
    return RecognitionRuntime(device=device, compute_type=compute_type)
