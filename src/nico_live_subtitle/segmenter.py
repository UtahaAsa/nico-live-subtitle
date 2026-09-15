from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import AudioConfig


@dataclass(frozen=True)
class AudioSegment:
    utterance_id: int
    samples: np.ndarray
    is_final: bool


class SpeechSegmenter:
    """使用简单能量门限把连续音频切成可流式更新的语句。"""

    def __init__(self, config: AudioConfig, sample_rate: int = 16_000) -> None:
        self._threshold = config.energy_threshold
        self._silence_frames = round(sample_rate * config.silence_ms / 1_000)
        self._partial_frames = round(sample_rate * config.partial_ms / 1_000)
        self._max_frames = round(sample_rate * config.max_utterance_ms / 1_000)
        self._pre_roll_frames = round(sample_rate * config.pre_roll_ms / 1_000)
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_size = 0
        self._active_chunks: list[np.ndarray] = []
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
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))
        is_speech = rms >= self._threshold

        if not self.active:
            if not is_speech:
                self._append_pre_roll(chunk)
                return []
            self._utterance_id += 1
            self._active_chunks = [*self._pre_roll, chunk]
            self._active_size = sum(item.size for item in self._active_chunks)
            self._pre_roll.clear()
            self._pre_roll_size = 0
            self._trailing_silence = 0
            self._next_partial_size = self._partial_frames
        else:
            self._active_chunks.append(chunk)
            self._active_size += chunk.size
            self._trailing_silence = 0 if is_speech else self._trailing_silence + chunk.size

        if self._active_size >= self._max_frames:
            return [self._finish()]
        if self._trailing_silence >= self._silence_frames:
            return [self._finish()]
        if self._active_size >= self._next_partial_size:
            self._next_partial_size += self._partial_frames
            return [self._snapshot(is_final=False)]
        return []

    def flush(self) -> AudioSegment | None:
        if not self.active:
            return None
        return self._finish()

    def _append_pre_roll(self, chunk: np.ndarray) -> None:
        if self._pre_roll_frames <= 0:
            return
        self._pre_roll.append(chunk)
        self._pre_roll_size += chunk.size
        while self._pre_roll and self._pre_roll_size > self._pre_roll_frames:
            excess = self._pre_roll_size - self._pre_roll_frames
            first = self._pre_roll[0]
            if first.size <= excess:
                self._pre_roll.popleft()
                self._pre_roll_size -= first.size
            else:
                self._pre_roll[0] = first[excess:]
                self._pre_roll_size -= excess

    def _snapshot(self, is_final: bool) -> AudioSegment:
        samples = np.concatenate(self._active_chunks).astype(np.float32, copy=False)
        return AudioSegment(self._utterance_id, samples, is_final)

    def _finish(self) -> AudioSegment:
        result = self._snapshot(is_final=True)
        self._active_chunks = []
        self._active_size = 0
        self._trailing_silence = 0
        self._next_partial_size = self._partial_frames
        return result
