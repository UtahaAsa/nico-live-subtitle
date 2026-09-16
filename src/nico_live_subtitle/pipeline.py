from __future__ import annotations

import queue
import threading
import traceback
from collections import deque
from dataclasses import dataclass, replace

from PySide6 import QtCore

from .asr import create_recognizer
from .audio import SystemAudioCapture
from .config import AppConfig
from .lexicon import LexiconBundle, build_lexicon_bundle
from .segmenter import AudioSegment, SpeechSegmenter
from .text import TranscriptStabilizer
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
        self._lexicon_lock = threading.Lock()
        self._lexicon_bundle: LexiconBundle | None = None
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
            lexicon = self._get_lexicon_bundle()
            recognition_config = replace(
                self._config.recognition, hotwords=lexicon.hotwords
            )
            recognizer = create_recognizer(recognition_config)
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

        stabilizers: dict[int, TranscriptStabilizer] = {}
        pending_ids: dict[int, int] = {}
        next_display_id = 0
        while not self._stop_event.is_set():
            try:
                segment = self._asr_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if segment is None:
                return
            previous_runtime = recognizer.runtime
            try:
                text = recognizer.transcribe(segment.samples)
            except Exception as error:
                self._emit_failure("日语识别失败", error)
                continue
            if recognizer.runtime != previous_runtime:
                self.status_changed.emit(
                    f"CUDA 不可用，已切换：{recognizer.runtime.device} / "
                    f"{recognizer.runtime.compute_type}"
                )
            if not text or not any(character.isalnum() for character in text):
                continue

            stabilizer = stabilizers.setdefault(
                segment.utterance_id, TranscriptStabilizer()
            )
            complete, pending = stabilizer.push(text, segment.is_final)
            for sentence in complete:
                display_id = pending_ids.pop(segment.utterance_id, None)
                if display_id is None:
                    next_display_id += 1
                    display_id = next_display_id
                self.transcript_ready.emit(
                    TranscriptUpdate(display_id, sentence, True)
                )
                if self._config.translation.backend != "none":
                    self._enqueue_translation(
                        TranslationJob(display_id, sentence, True)
                    )
            if pending and not segment.is_final:
                display_id = pending_ids.get(segment.utterance_id)
                if display_id is None:
                    next_display_id += 1
                    display_id = next_display_id
                    pending_ids[segment.utterance_id] = display_id
                self.transcript_ready.emit(
                    TranscriptUpdate(display_id, pending, False)
                )
            if segment.is_final:
                stabilizers.pop(segment.utterance_id, None)
                pending_ids.pop(segment.utterance_id, None)

    def _translation_worker(self) -> None:
        try:
            lexicon = self._get_lexicon_bundle()
            translator = create_translator(
                self._config.translation.backend,
                self._config.translation.packages_dir,
                self._config.translation.model_path,
                lexicon.glossary,
                self._config.translation.n_gpu_layers,
                self._config.translation.api_base,
                self._config.translation.api_model,
                self._config.translation.api_key_env,
                self._config.translation.timeout_sec,
            )
            if self._stop_event.is_set():
                return
            warmup = getattr(translator, "warmup", None)
            if callable(warmup):
                self.status_changed.emit("正在预热本地翻译模型…")
                warmup()
                if self._stop_event.is_set():
                    return
                self.status_changed.emit("本地翻译模型就绪")
        except Exception as error:
            if not self._stop_event.is_set():
                self._emit_failure("翻译后端初始化失败", error)
            return

        context: deque[tuple[str, str]] = deque(
            maxlen=self._config.translation.context_lines or None
        )
        while not self._stop_event.is_set():
            try:
                job = self._translation_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                translate_iter = getattr(translator, "translate_iter", None)
                if callable(translate_iter):
                    results = translate_iter(job.text, tuple(context))
                else:
                    results = iter([translator.translate(job.text, tuple(context))])
                previous = ""
                for translated in results:
                    translated = str(translated).strip()
                    if not translated or translated == previous:
                        continue
                    if previous:
                        self.translation_ready.emit(
                            TranslationUpdate(
                                job.utterance_id,
                                job.text,
                                previous,
                                False,
                            )
                        )
                    previous = translated
                if not previous:
                    raise RuntimeError("翻译后端返回了空结果")
            except Exception as error:
                self._emit_failure("翻译失败，日语识别仍会继续", error)
                continue
            self.translation_ready.emit(
                TranslationUpdate(
                    job.utterance_id, job.text, previous, job.is_final
                )
            )
            if job.is_final and self._config.translation.context_lines:
                context.append((job.text, previous))

    def _get_lexicon_bundle(self) -> LexiconBundle:
        with self._lexicon_lock:
            if self._lexicon_bundle is None:
                self._lexicon_bundle = build_lexicon_bundle(
                    self._config.lexicon,
                    self._config.recognition.hotwords,
                    self._config.translation.glossary,
                )
            return self._lexicon_bundle

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
