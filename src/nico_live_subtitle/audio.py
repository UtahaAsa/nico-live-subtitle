from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import AudioConfig


TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class LoopbackDevice:
    id: str
    name: str
    is_default: bool


def list_loopback_devices() -> list[LoopbackDevice]:
    import soundcard as sc

    default_speaker = sc.default_speaker()
    default_id = str(getattr(default_speaker, "id", ""))
    default_name = str(getattr(default_speaker, "name", ""))
    devices: list[LoopbackDevice] = []
    for microphone in sc.all_microphones(include_loopback=True):
        if not bool(getattr(microphone, "isloopback", False)):
            continue
        device_id = str(getattr(microphone, "id", ""))
        name = str(getattr(microphone, "name", microphone))
        is_default = bool(
            (default_id and device_id == default_id)
            or (default_name and default_name.casefold() in name.casefold())
        )
        devices.append(LoopbackDevice(device_id, name, is_default))
    devices.sort(key=lambda item: (not item.is_default, item.name.casefold()))
    return devices


def resample_mono(
    samples: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1, dtype=np.float32)
    elif array.ndim != 1:
        raise ValueError("音频数组必须是一维或二维")

    if source_rate == target_rate or array.size == 0:
        return np.ascontiguousarray(array, dtype=np.float32)

    target_size = max(1, round(array.size * target_rate / source_rate))
    source_positions = np.arange(array.size, dtype=np.float64)
    target_positions = np.linspace(0, array.size - 1, target_size, dtype=np.float64)
    result = np.interp(target_positions, source_positions, array)
    return np.ascontiguousarray(result, dtype=np.float32)


class SystemAudioCapture:
    def __init__(self, config: AudioConfig) -> None:
        self._config = config

    def run(
        self,
        stop_event: threading.Event,
        on_audio: Callable[[np.ndarray], None],
        on_status: Callable[[str], None],
    ) -> None:
        import soundcard as sc

        microphone = self._select_microphone(sc)
        name = str(getattr(microphone, "name", microphone))
        on_status(f"正在监听：{name}")
        frame_count = max(
            1, round(self._config.sample_rate * self._config.block_ms / 1_000)
        )

        # Windows/WASAPI 的单声道回环存在兼容问题，因此不请求单声道，
        # 而是在录制完成后统一混音。
        with microphone.recorder(
            samplerate=self._config.sample_rate,
            blocksize=frame_count * 2,
        ) as recorder:
            while not stop_event.is_set():
                frames = recorder.record(numframes=frame_count)
                if frames is None:
                    continue
                mono = resample_mono(
                    np.asarray(frames), self._config.sample_rate, TARGET_SAMPLE_RATE
                )
                if mono.size:
                    on_audio(mono)

    def _select_microphone(self, sc: object) -> object:
        requested = self._config.device
        if requested:
            try:
                return sc.get_microphone(requested, include_loopback=True)
            except Exception as error:
                raise RuntimeError(f"找不到音频回环设备：{requested}") from error

        default_speaker = sc.default_speaker()
        candidates = sc.all_microphones(include_loopback=True)
        loopbacks = [item for item in candidates if getattr(item, "isloopback", False)]
        if not loopbacks:
            raise RuntimeError("没有找到 WASAPI 回环设备")

        speaker_id = str(getattr(default_speaker, "id", ""))
        speaker_name = str(getattr(default_speaker, "name", ""))
        for microphone in loopbacks:
            if speaker_id and str(getattr(microphone, "id", "")) == speaker_id:
                return microphone
        for microphone in loopbacks:
            name = str(getattr(microphone, "name", ""))
            if speaker_name and speaker_name.casefold() in name.casefold():
                return microphone
        return loopbacks[0]

