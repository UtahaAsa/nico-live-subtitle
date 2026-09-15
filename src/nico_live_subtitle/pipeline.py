from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass

from PySide6 import QtCore

from .asr import JapaneseRecognizer
from .audio import SystemAudioCapture
from .config import AppConfig
from .segmenter import AudioSegment, SpeechSegmenter
from .translation import create_translator


@dataclass(frozen=True)
class TranscriptUpdate:
    utterance_id: int
    text: str
    is_final: bool


@dataclass(frozen=True)
class TranslationJob:
    utterance_id: int
    text: str
    is_final: bool


@dataclass(frozen=True)
class TranslationUpdate:
    utterance_id: int
    source_text: str
    text: str
    is_final: bool


class SubtitlePipeline(QtCore.QObject):
    status_changed = QtCore.Signal(str)
    transcript_ready = QtCore.Signal(object)
    translation_ready = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._stop_event = threading.Event()
        self._asr_queue: queue.Queue[AudioSegment | None] = queue.Queue(maxsize=4)
        self._translation_queue: queue.Queue[TranslationJob | None] = queue.Queue(
            maxsize=8
        )
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(
                target=self._capture_worker, name="audio-capture", daemon=True
            ),
            threading.Thread(target=self._asr_worker, name="asr", daemon=True),
            threading.Thread(
                target=self._translation_worker, name="translation", daemon=True
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._offer_sentinel(self._asr_queue)
        self._offer_sentinel(self._translation_queue)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        self.status_changed.emit("已停止")

    def _capture_worker(self) -> None:
        segmenter = SpeechSegmenter(self._config.audio)
        capture = SystemAudioCapture(self._config.audio)

        def on_audio(samples: object) -> None:
            for segment in segmenter.push(samples):
                self._enqueue_asr(segment)

        try:
            capture.run(self._stop_event, on_audio, self.status_changed.emit)
        except Exception as error:
            if not self._stop_event.is_set():
                self._emit_failure("系统音频采集失败", error)
        finally:
            pending = segmenter.flush()
            if pending is not None and not self._stop_event.is_set():
                self._enqueue_asr(pending)

    def _asr_worker(self) -> None:
        try:
            self.status_changed.emit("正在加载语音模型…")
            recognizer = JapaneseRecognizer(self._config.recognition)
            if self._stop_event.is_set():
                return
            self.status_changed.emit(
                f"语音模型就绪：{recognizer.runtime.device} / "
                f"{recognizer.runtime.compute_type}"
            )
        except Exception as error:
            if not self._stop_event.is_set():
                self._emit_failure("语音模型加载失败", error)
            return

        while not self._stop_event.is_set():
            try:
                segment = self._asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if segment is None:
                return
            try:
                text = recognizer.transcribe(segment.samples)
            except Exception as error:
                self._emit_failure("日语识别失败", error)
                continue
            if not text:
                continue

            update = TranscriptUpdate(segment.utterance_id, text, segment.is_final)
            self.transcript_ready.emit(update)
            self._enqueue_translation(
                TranslationJob(segment.utterance_id, text, segment.is_final)
            )

    def _translation_worker(self) -> None:
        try:
            translator = create_translator(self._config.translation.backend)
        except Exception as error:
            if not self._stop_event.is_set():
                self._emit_failure("翻译后端初始化失败", error)
            return

        while not self._stop_event.is_set():
            try:
                job = self._translation_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                translated = translator.translate(job.text)
            except Exception as error:
                self._emit_failure("翻译失败，日语识别仍会继续", error)
                continue
            self.translation_ready.emit(
                TranslationUpdate(
                    job.utterance_id, job.text, translated, job.is_final
                )
            )

    def _enqueue_asr(self, segment: AudioSegment) -> None:
        if segment.is_final:
            self._discard_queued_partials(self._asr_queue, segment.utterance_id)
        try:
            self._asr_queue.put_nowait(segment)
        except queue.Full:
            if segment.is_final:
                self._drop_oldest(self._asr_queue)
                self._asr_queue.put_nowait(segment)

    def _enqueue_translation(self, job: TranslationJob) -> None:
        if job.is_final:
            self._discard_queued_partials(self._translation_queue, job.utterance_id)
        try:
            self._translation_queue.put_nowait(job)
        except queue.Full:
            self._drop_oldest(self._translation_queue)
            try:
                self._translation_queue.put_nowait(job)
            except queue.Full:
                pass

    @staticmethod
    def _discard_queued_partials(
        target_queue: queue.Queue[object], utterance_id: int
    ) -> None:
        retained: list[object] = []
        while True:
            try:
                item = target_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                retained.append(item)
                continue
            if (
                getattr(item, "utterance_id", None) == utterance_id
                and not getattr(item, "is_final", False)
            ):
                continue
            retained.append(item)
        for item in retained:
            try:
                target_queue.put_nowait(item)
            except queue.Full:
                break

    @staticmethod
    def _drop_oldest(target_queue: queue.Queue[object]) -> None:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass

    @staticmethod
    def _offer_sentinel(target_queue: queue.Queue[object]) -> None:
        try:
            target_queue.put_nowait(None)
        except queue.Full:
            SubtitlePipeline._drop_oldest(target_queue)
            try:
                target_queue.put_nowait(None)
            except queue.Full:
                pass

    def _emit_failure(self, context: str, error: Exception) -> None:
        detail = "".join(traceback.format_exception_only(type(error), error)).strip()
        self.failed.emit(f"{context}：{detail}")
