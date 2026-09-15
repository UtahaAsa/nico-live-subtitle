from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import AudioConfig


@dataclass(frozen=True)
class AudioSegment:
    utterance_id: int
    samples: np.ndarray
    is_final: bool


class SpeechProbability(Protocol):
    def score(self, samples: np.ndarray) -> float: ...


class EnergySpeechProbability:
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def score(self, samples: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        return min(1.0, rms / (self._threshold * 2.0))


class SileroSpeechProbability:
    def __init__(self, model_path: str | None, sample_rate: int = 16_000) -> None:
        if not model_path:
            raise RuntimeError("Silero VAD 模型路径不能为空")
        resolved_model = Path(model_path).resolve()
        if not resolved_model.is_file():
            raise RuntimeError(f"找不到 Silero VAD 模型：{resolved_model}")
        import torch

        self._torch = torch
        self._sample_rate = sample_rate
        self._model = torch.jit.load(str(resolved_model), map_location="cpu")
        self._model.eval()
        self._model.reset_states()

    def score(self, samples: np.ndarray) -> float:
        if samples.size != 512:
            raise ValueError("Silero VAD 每次必须接收 512 个 16kHz 采样点")
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.float32)
        )
        with self._torch.inference_mode():
            return float(self._model(tensor, self._sample_rate).item())


class SpeechSegmenter:
    """使用语音概率、预录音和自适应停顿切分连续音频。"""

    def __init__(
        self,
        config: AudioConfig,
        sample_rate: int = 16_000,
        speech_probability: SpeechProbability | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        if speech_probability is not None:
            self._speech_probability = speech_probability
            self._threshold = config.vad_threshold
        elif config.vad_mode == "silero":
            self._speech_probability = SileroSpeechProbability(
                config.silero_model, sample_rate
            )
            self._threshold = config.vad_threshold
        else:
            self._speech_probability = EnergySpeechProbability(
                config.energy_threshold
            )
            self._threshold = 0.5
        self._base_silence_frames = round(
            sample_rate * config.silence_ms / 1_000
        )
        self._min_speech_frames = round(
            sample_rate * config.min_speech_ms / 1_000
        )
        self._partial_frames = round(sample_rate * config.partial_ms / 1_000)
        self._max_frames = round(sample_rate * config.max_utterance_ms / 1_000)
        self._pre_roll_frames = round(sample_rate * config.pre_roll_ms / 1_000)
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_confidences: deque[float] = deque()
        self._pre_roll_size = 0
        self._active_chunks: list[np.ndarray] = []
        self._confidence_history: list[float] = []
        self._active_size = 0
        self._trailing_silence = 0
        self._next_partial_size = self._partial_frames
        self._utterance_id = 0

    @property
    def active(self) -> bool:
        return bool(self._active_chunks)

    def push(self, samples: np.ndarray) -> list[AudioSegment]:
        chunk = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return []
        confidence = self._speech_probability.score(chunk)
        is_speech = confidence >= self._threshold

        if not self.active:
            if not is_speech:
                self._append_pre_roll(chunk, confidence)
                return []
            self._utterance_id += 1
            self._active_chunks = [*self._pre_roll, chunk]
            self._confidence_history = [*self._pre_roll_confidences, confidence]
            self._active_size = sum(item.size for item in self._active_chunks)
            self._pre_roll.clear()
            self._pre_roll_confidences.clear()
            self._pre_roll_size = 0
            self._trailing_silence = 0
            self._next_partial_size = self._partial_frames
        else:
            self._active_chunks.append(chunk)
            self._confidence_history.append(confidence)
            self._active_size += chunk.size
            self._trailing_silence = 0 if is_speech else self._trailing_silence + chunk.size

        if self._active_size >= self._max_frames:
            return [self._split_at_best_pause()]
        if self._trailing_silence >= self._effective_silence_frames():
            voiced_frames = sum(
                item.size
                for item, score in zip(
                    self._active_chunks, self._confidence_history, strict=True
                )
                if score >= self._threshold
            )
            if voiced_frames < self._min_speech_frames:
                self._reset_active()
                return []
            return [self._finish()]
        if self._active_size >= self._next_partial_size:
            self._next_partial_size += self._partial_frames
            return [self._snapshot(is_final=False)]
        return []

    def flush(self) -> AudioSegment | None:
        if not self.active:
            return None
        return self._finish()

    def _append_pre_roll(self, chunk: np.ndarray, confidence: float) -> None:
        if self._pre_roll_frames <= 0:
            return
        self._pre_roll.append(chunk)
        self._pre_roll_confidences.append(confidence)
        self._pre_roll_size += chunk.size
        while self._pre_roll and self._pre_roll_size > self._pre_roll_frames:
            excess = self._pre_roll_size - self._pre_roll_frames
            first = self._pre_roll[0]
            if first.size <= excess:
                self._pre_roll.popleft()
                self._pre_roll_confidences.popleft()
                self._pre_roll_size -= first.size
            else:
                self._pre_roll[0] = first[excess:]
                self._pre_roll_size -= excess

    def _snapshot(self, is_final: bool) -> AudioSegment:
        samples = np.concatenate(self._active_chunks).astype(np.float32, copy=False)
        return AudioSegment(self._utterance_id, samples, is_final)

    def _finish(self) -> AudioSegment:
        result = self._snapshot(is_final=True)
        self._reset_active()
        return result

    def _reset_active(self) -> None:
        self._active_chunks = []
        self._confidence_history = []
        self._active_size = 0
        self._trailing_silence = 0
        self._next_partial_size = self._partial_frames

    def _effective_silence_frames(self) -> int:
        duration = self._active_size / self._sample_rate
        if duration >= 6.0:
            multiplier = 0.25
        elif duration >= 3.0:
            multiplier = 0.5
        else:
            multiplier = 1.0
        minimum = round(self._sample_rate * 0.2)
        return max(minimum, round(self._base_silence_frames * multiplier))

    def _split_at_best_pause(self) -> AudioSegment:
        split_index = self._best_pause_index()
        if split_index <= 0:
            return self._finish()

        first_chunks = self._active_chunks[:split_index]
        remaining_chunks = self._active_chunks[split_index:]
        remaining_scores = self._confidence_history[split_index:]
        result = AudioSegment(
            self._utterance_id,
            np.concatenate(first_chunks).astype(np.float32, copy=False),
            True,
        )
        self._utterance_id += 1
        self._active_chunks = remaining_chunks
        self._confidence_history = remaining_scores
        self._active_size = sum(item.size for item in remaining_chunks)
        self._trailing_silence = 0
        for item, score in zip(
            reversed(remaining_chunks), reversed(remaining_scores), strict=True
        ):
            if score >= self._threshold:
                break
            self._trailing_silence += item.size
        self._next_partial_size = self._active_size + self._partial_frames
        return result

    def _best_pause_index(self) -> int:
        count = len(self._confidence_history)
        if count < 4:
            return -1
        window = min(5, count // 2)
        smoothed = []
        for index in range(count):
            start = max(0, index - window // 2)
            end = min(count, index + window // 2 + 1)
            values = self._confidence_history[start:end]
            smoothed.append(sum(values) / len(values))
        search_start = max(1, count * 3 // 10)
        candidates = smoothed[search_start:-1]
        if not candidates:
            return -1
        minimum = min(candidates)
        split_index = search_start + max(
            index for index, value in enumerate(candidates) if value == minimum
        )
        average = sum(candidates) / len(candidates)
        if minimum < self._threshold or minimum < average * 0.8:
            return split_index
        return -1
