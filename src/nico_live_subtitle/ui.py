from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from .audio import LoopbackDevice, list_loopback_devices
from .config import AppConfig
from .lexicon import AnimeLexicon, load_lexicon_catalog
from .pipeline import SubtitlePipeline, TranscriptUpdate, TranslationUpdate


@dataclass
class SubtitleLine:
    japanese: str = ""
    chinese: str = ""
    is_final: bool = False


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, config: AppConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("实时字幕设置")
        self.setMinimumWidth(560)
        self._config = config
        self._devices: list[LoopbackDevice] = []

        self.device_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("刷新设备")
        device_row = QtWidgets.QHBoxLayout()
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_button)

        self.asr_engine_combo = QtWidgets.QComboBox()
        self.asr_engine_combo.addItem("Anime-Whisper（动画日语）", "anime_whisper")
        self.asr_engine_combo.addItem("Faster-Whisper（通用）", "faster_whisper")
        asr_index = self.asr_engine_combo.findData(config.recognition.engine)
        self.asr_engine_combo.setCurrentIndex(max(0, asr_index))
        self.model_edit = QtWidgets.QLineEdit(config.recognition.model)
        self.model_edit.setPlaceholderText("例如 small，或本地 CTranslate2 模型目录")
        self.live_model_edit = QtWidgets.QLineEdit(config.recognition.live_model)
        self.live_model_edit.setPlaceholderText(
            "例如 work/models/whisper-large-v3-turbo"
        )
        self.runtime_combo = QtWidgets.QComboBox()
        self.runtime_combo.addItems(["auto", "cpu", "cuda"])
        self.runtime_combo.setCurrentText(config.recognition.device)
        self.compute_combo = QtWidgets.QComboBox()
        self.compute_combo.addItems(["auto", "int8", "float16", "int8_float16"])
        if config.recognition.compute_type not in {
            self.compute_combo.itemText(index)
            for index in range(self.compute_combo.count())
        }:
            self.compute_combo.addItem(config.recognition.compute_type)
        self.compute_combo.setCurrentText(config.recognition.compute_type)
        self.hotwords_edit = QtWidgets.QLineEdit(config.recognition.hotwords)
        self.hotwords_edit.setPlaceholderText("只填写词库中没有的名字，用空格分隔")
        self.hotwords_edit.setToolTip(
            "Faster-Whisper 会直接使用；Anime-Whisper 中由翻译模型结合词库纠错"
        )

        self._lexicons: dict[str, AnimeLexicon] = {}
        self.lexicon_combo = QtWidgets.QComboBox()
        self.lexicon_combo.setEditable(True)
        self.lexicon_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.lexicon_combo.setMaxVisibleItems(16)
        completer = self.lexicon_combo.completer()
        if completer is not None:
            completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(
                QtWidgets.QCompleter.CompletionMode.PopupCompletion
            )
        self.lexicon_refresh_button = QtWidgets.QPushButton("刷新词库")
        lexicon_row = QtWidgets.QHBoxLayout()
        lexicon_row.addWidget(self.lexicon_combo, 1)
        lexicon_row.addWidget(self.lexicon_refresh_button)
        self.lexicon_info_label = QtWidgets.QLabel()
        self.lexicon_info_label.setWordWrap(True)
        self.lexicon_info_label.setStyleSheet("color: #666;")

        self.translation_combo = QtWidgets.QComboBox()
        self.translation_combo.addItem("本地动画 LLM（推荐离线）", "local_llm")
        self.translation_combo.addItem("OpenAI 兼容接口（推荐质量）", "openai_compatible")
        self.translation_combo.addItem("HY-MT 离线翻译（轻量）", "hunyuan")
        self.translation_combo.addItem("Google 在线翻译", "google")
        self.translation_combo.addItem("Argos 离线翻译", "argos")
        self.translation_combo.addItem("不翻译", "none")
        translation_index = self.translation_combo.findData(config.translation.backend)
        self.translation_combo.setCurrentIndex(max(0, translation_index))
        self.argos_dir_edit = QtWidgets.QLineEdit(
            config.translation.packages_dir or ""
        )
        self.argos_dir_edit.setPlaceholderText(
            "例如 work/argos/packages；仅 Argos 模式使用"
        )
        self.translation_model_edit = QtWidgets.QLineEdit(
            config.translation.model_path or ""
        )
        self.translation_model_edit.setPlaceholderText(
            "本地 LLM 或 HY-MT 的 .gguf 模型路径"
        )
        self.context_lines_spin = QtWidgets.QSpinBox()
        self.context_lines_spin.setRange(0, 8)
        self.context_lines_spin.setValue(config.translation.context_lines)
        self.glossary_edit = QtWidgets.QLineEdit(config.translation.glossary)
        self.glossary_edit.setPlaceholderText("只填写个人修正，例如 昵称=固定译名")
        self.api_base_edit = QtWidgets.QLineEdit(config.translation.api_base)
        self.api_base_edit.setPlaceholderText("例如 http://127.0.0.1:11434/v1")
        self.api_model_edit = QtWidgets.QLineEdit(config.translation.api_model)
        self.api_key_env_edit = QtWidgets.QLineEdit(config.translation.api_key_env)
        self.api_key_env_edit.setPlaceholderText("只填写环境变量名，不填写密钥")

        self.vad_combo = QtWidgets.QComboBox()
        self.vad_combo.addItem("Silero 神经网络 VAD", "silero")
        self.vad_combo.addItem("能量阈值 VAD", "energy")
        vad_index = self.vad_combo.findData(config.audio.vad_mode)
        self.vad_combo.setCurrentIndex(max(0, vad_index))
        self.silero_model_edit = QtWidgets.QLineEdit(config.audio.silero_model or "")
        self.silero_model_edit.setPlaceholderText("Silero VAD 的 .jit 模型路径")
        self.vad_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.vad_threshold_spin.setRange(0.05, 0.95)
        self.vad_threshold_spin.setSingleStep(0.05)
        self.vad_threshold_spin.setValue(config.audio.vad_threshold)

        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 0.2)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.001)
        self.threshold_spin.setValue(config.audio.energy_threshold)
        self.silence_spin = QtWidgets.QSpinBox()
        self.silence_spin.setRange(100, 3_000)
        self.silence_spin.setSuffix(" ms")
        self.silence_spin.setValue(config.audio.silence_ms)
        self.partial_spin = QtWidgets.QSpinBox()
        self.partial_spin.setRange(500, 6_000)
        self.partial_spin.setSuffix(" ms")
        self.partial_spin.setValue(config.audio.partial_ms)

        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(12, 72)
        self.font_spin.setSuffix(" px")
        self.font_spin.setValue(config.overlay.font_size)
        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0.2, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(config.overlay.opacity)
        self.lines_spin = QtWidgets.QSpinBox()
        self.lines_spin.setRange(1, 8)
        self.lines_spin.setValue(config.overlay.max_lines)
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(480, 3_840)
        self.width_spin.setSuffix(" px")
        self.width_spin.setValue(config.overlay.width)

        form = QtWidgets.QFormLayout()
        form.addRow("系统音频设备", device_row)
        form.addRow("语音切分方式", self.vad_combo)
        form.addRow("Silero VAD 模型", self.silero_model_edit)
        form.addRow("VAD 语音阈值", self.vad_threshold_spin)
        form.addRow("识别引擎", self.asr_engine_combo)
        form.addRow("动画语音模型", self.model_edit)
        form.addRow("日英直播模型", self.live_model_edit)
        form.addRow("运行设备", self.runtime_combo)
        form.addRow("计算类型", self.compute_combo)
        form.addRow("内容模式 / 词库", lexicon_row)
        form.addRow("词库内容", self.lexicon_info_label)
        form.addRow("自定义热词", self.hotwords_edit)
        form.addRow("翻译方式", self.translation_combo)
        form.addRow("本地翻译模型", self.translation_model_edit)
        form.addRow("参考前文句数", self.context_lines_spin)
        form.addRow("自定义术语", self.glossary_edit)
        form.addRow("兼容接口地址", self.api_base_edit)
        form.addRow("兼容接口模型", self.api_model_edit)
        form.addRow("密钥环境变量名", self.api_key_env_edit)
        form.addRow("Argos 模型目录", self.argos_dir_edit)
        form.addRow("语音能量阈值", self.threshold_spin)
        form.addRow("确认停顿", self.silence_spin)
        form.addRow("临时字幕间隔", self.partial_spin)
        form.addRow("字幕字号", self.font_spin)
        form.addRow("背景透明度", self.opacity_spin)
        form.addRow("保留句数", self.lines_spin)
        form.addRow("窗口宽度", self.width_spin)

        hint = QtWidgets.QLabel(
            "设置在确认后生效；运行中修改会重启识别。首次使用模型名称会触发模型下载。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        self.refresh_button.clicked.connect(self._refresh_devices)
        self.lexicon_refresh_button.clicked.connect(self._refresh_lexicons)
        self.lexicon_combo.currentIndexChanged.connect(self._update_lexicon_info)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)
        self._refresh_lexicons()
        self._refresh_devices()

    def apply(self) -> None:
        self._config.audio.device = self.device_combo.currentData()
        self._config.audio.vad_mode = str(self.vad_combo.currentData())
        self._config.audio.block_ms = (
            32 if self._config.audio.vad_mode == "silero" else 100
        )
        self._config.audio.silero_model = (
            self.silero_model_edit.text().strip() or None
        )
        self._config.audio.vad_threshold = self.vad_threshold_spin.value()
        self._config.audio.energy_threshold = self.threshold_spin.value()
        self._config.audio.silence_ms = self.silence_spin.value()
        self._config.audio.partial_ms = self.partial_spin.value()
        self._config.recognition.engine = str(self.asr_engine_combo.currentData())
        self._config.recognition.model = self.model_edit.text().strip()
        self._config.recognition.live_model = self.live_model_edit.text().strip()
        self._config.recognition.device = self.runtime_combo.currentText()
        self._config.recognition.compute_type = self.compute_combo.currentText()
        self._config.recognition.hotwords = self.hotwords_edit.text().strip()
        self._config.lexicon.profile = str(self.lexicon_combo.currentData() or "")
        self._config.translation.backend = str(self.translation_combo.currentData())
        self._config.translation.packages_dir = (
            self.argos_dir_edit.text().strip() or None
        )
        self._config.translation.model_path = (
            self.translation_model_edit.text().strip() or None
        )
        self._config.translation.context_lines = self.context_lines_spin.value()
        self._config.translation.glossary = self.glossary_edit.text().strip()
        self._config.translation.api_base = self.api_base_edit.text().strip()
        self._config.translation.api_model = self.api_model_edit.text().strip()
        self._config.translation.api_key_env = self.api_key_env_edit.text().strip()
        self._config.overlay.font_size = self.font_spin.value()
        self._config.overlay.opacity = self.opacity_spin.value()
        self._config.overlay.max_lines = self.lines_spin.value()
        self._config.overlay.width = self.width_spin.value()
        self._config.validate()

    def _refresh_devices(self) -> None:
        selected = self._config.audio.device
        self.device_combo.clear()
        self.device_combo.addItem("默认播放设备", None)
        try:
            self._devices = list_loopback_devices()
        except Exception as error:
            self.device_combo.addItem(f"读取失败：{error}", "")
            return
        for device in self._devices:
            suffix = "（默认）" if device.is_default else ""
            self.device_combo.addItem(f"{device.name}{suffix}", device.id)
        index = self.device_combo.findData(selected)
        self.device_combo.setCurrentIndex(max(0, index))

    def _refresh_lexicons(self) -> None:
        selected = (
            self.lexicon_combo.currentData()
            if self.lexicon_combo.count()
            else self._config.lexicon.profile
        )
        self.lexicon_combo.clear()
        self._lexicons = {}
        self.lexicon_combo.addItem("自动判断（动画 / 日英直播）", "auto")
        self.lexicon_combo.addItem("通用日英直播", "live")
        self.lexicon_combo.addItem("通用日语动画", "")
        try:
            catalog = load_lexicon_catalog(self._config.lexicon.directory)
        except ValueError as error:
            self.lexicon_combo.addItem(f"读取失败：{error}", None)
            self._update_lexicon_info()
            return
        self._lexicons = {item.id: item for item in catalog}
        for lexicon in catalog:
            if lexicon.id in {"anime-common", "live-common"}:
                continue
            self.lexicon_combo.addItem(lexicon.title, lexicon.id)
        index = self.lexicon_combo.findData(selected or "")
        if index < 0 and selected:
            self.lexicon_combo.addItem(f"找不到：{selected}", None)
            index = self.lexicon_combo.count() - 1
        self.lexicon_combo.setCurrentIndex(max(0, index))
        self._update_lexicon_info()

    def _update_lexicon_info(self) -> None:
        profile_id = self.lexicon_combo.currentData()
        if profile_id == "auto":
            self.lexicon_info_label.setText(
                "开始时匹配窗口标题；已知作品加载专属词库，其他动画/日英直播自动分流"
            )
            return
        common = self._lexicons.get("anime-common")
        common_count = len(common.terms) if common is not None else 0
        if profile_id == "live":
            live = self._lexicons.get("live-common")
            live_count = len(live.terms) if live is not None else 0
            self.lexicon_info_label.setText(
                f"日语/英语自动识别，附带 {live_count} 条直播口语固定译法"
            )
            return
        profile = self._lexicons.get(str(profile_id)) if profile_id else None
        profile_count = len(profile.terms) if profile is not None else 0
        if profile_id is None and self.lexicon_combo.currentIndex() > 0:
            self.lexicon_info_label.setText("词库不可用，请刷新或重新选择")
            return
        self.lexicon_info_label.setText(
            f"自动叠加通用 {common_count} 条 + 本作品 {profile_count} 条；"
            "输入框只用于个人补充"
        )

    def _validate_and_accept(self) -> None:
        if not self.model_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "设置错误", "动画语音模型不能为空")
            return
        if not self.live_model_edit.text().strip():
            QtWidgets.QMessageBox.warning(self, "设置错误", "日英直播模型不能为空")
            return
        if self.device_combo.currentData() == "":
            QtWidgets.QMessageBox.warning(self, "设置错误", "请先选择有效音频设备")
            return
        lexicon_index = self.lexicon_combo.findText(
            self.lexicon_combo.currentText(), QtCore.Qt.MatchFlag.MatchFixedString
        )
        if lexicon_index < 0 or self.lexicon_combo.itemData(lexicon_index) is None:
            QtWidgets.QMessageBox.warning(
                self, "设置错误", "请选择有效的内容模式 / 词库"
            )
            return
        self.lexicon_combo.setCurrentIndex(lexicon_index)
        if (
            self.translation_combo.currentData() in {"hunyuan", "local_llm"}
            and not self.translation_model_edit.text().strip()
        ):
            QtWidgets.QMessageBox.warning(self, "设置错误", "本地翻译模型路径不能为空")
            return
        if (
            self.vad_combo.currentData() == "silero"
            and not self.silero_model_edit.text().strip()
        ):
            QtWidgets.QMessageBox.warning(self, "设置错误", "Silero VAD 模型路径不能为空")
            return
        try:
            self.apply()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "设置错误", str(error))
            return
        self.accept()


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._pipeline: SubtitlePipeline | None = None
        self._lines: OrderedDict[int, SubtitleLine] = OrderedDict()
        self._drag_offset: QtCore.QPoint | None = None
        self._running = False

        self.setWindowTitle("Nico Live Subtitle")
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        self.card = QtWidgets.QFrame()
        self.japanese_label = QtWidgets.QLabel("播放动画或日英直播后点击“开始”")
        self.chinese_label = QtWidgets.QLabel("中文字幕将在这里显示")
        for label in (self.japanese_label, self.chinese_label):
            label.setWordWrap(True)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setStyleSheet("color: #c8c8c8;")
        self.start_button = QtWidgets.QPushButton("开始")
        self.settings_button = QtWidgets.QPushButton("设置")
        self.pin_button = QtWidgets.QPushButton("点击穿透")
        self.close_button = QtWidgets.QPushButton("退出")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.status_label, 1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.settings_button)
        controls.addWidget(self.pin_button)
        controls.addWidget(self.close_button)

        card_layout = QtWidgets.QVBoxLayout(self.card)
        card_layout.setContentsMargins(22, 14, 22, 12)
        card_layout.setSpacing(6)
        card_layout.addWidget(self.japanese_label)
        card_layout.addWidget(self.chinese_label)
        card_layout.addLayout(controls)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.card)

        self.start_button.clicked.connect(self.toggle_running)
        self.settings_button.clicked.connect(self.open_settings)
        self.pin_button.clicked.connect(lambda: self.set_click_through(True))
        self.close_button.clicked.connect(QtWidgets.QApplication.instance().quit)
        self._build_tray_icon()
        self._apply_overlay_style()
        self.resize(config.overlay.width, self.sizeHint().height())
        self._move_to_bottom_center()

        if config.overlay.click_through:
            QtCore.QTimer.singleShot(0, lambda: self.set_click_through(True))

    def toggle_running(self) -> None:
        if self._running:
            self.stop_pipeline()
        else:
            self.start_pipeline()

    def start_pipeline(self) -> None:
        if self._running:
            return
        self._pipeline = SubtitlePipeline(self._config)
        self._pipeline.status_changed.connect(self._set_status)
        self._pipeline.transcript_ready.connect(self._on_transcript)
        self._pipeline.translation_ready.connect(self._on_translation)
        self._pipeline.failed.connect(self._on_failure)
        self._running = True
        self.start_button.setText("停止")
        self._set_status("正在启动…")
        self._pipeline.start()

    def stop_pipeline(self) -> None:
        if self._pipeline is not None:
            pipeline = self._pipeline
            self._pipeline = None
            pipeline.status_changed.disconnect(self._set_status)
            pipeline.transcript_ready.disconnect(self._on_transcript)
            pipeline.translation_ready.disconnect(self._on_translation)
            pipeline.failed.disconnect(self._on_failure)
            pipeline.stop()
        self._running = False
        self.start_button.setText("开始")
        self._set_status("已停止")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self._config, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        was_running = self._running
        if was_running:
            self.stop_pipeline()
        self._apply_overlay_style()
        self._resize_to_content_preserving_anchor()
        self._trim_lines()
        self._render_lines()
        if was_running:
            self.start_pipeline()

    def set_click_through(self, enabled: bool) -> None:
        self._config.overlay.click_through = enabled
        self.setWindowFlag(QtCore.Qt.WindowType.WindowTransparentForInput, enabled)
        self.show()
        self.click_through_action.setChecked(enabled)
        if not enabled:
            self.raise_()
            self.activateWindow()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_pipeline()
        self.tray_icon.hide()
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_offset = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._drag_offset is not None
            and event.buttons() & QtCore.Qt.MouseButton.LeftButton
        ):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def _on_transcript(self, update: TranscriptUpdate) -> None:
        line = self._lines.setdefault(update.utterance_id, SubtitleLine())
        if line.japanese != update.text:
            line.chinese = ""
        line.japanese = update.text
        line.is_final = update.is_final
        self._lines.move_to_end(update.utterance_id)
        self._trim_lines()
        self._render_lines()

    def _on_translation(self, update: TranslationUpdate) -> None:
        line = self._lines.get(update.utterance_id)
        if line is None or line.japanese != update.source_text:
            return
        line.chinese = update.text
        line.is_final = update.is_final
        self._render_lines()

    def _on_failure(self, message: str) -> None:
        self._set_status(message)
        self.tray_icon.showMessage(
            "Nico Live Subtitle",
            message,
            QtWidgets.QSystemTrayIcon.MessageIcon.Warning,
            4_000,
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _trim_lines(self) -> None:
        while len(self._lines) > self._config.overlay.max_lines:
            self._lines.popitem(last=False)

    def _render_lines(self) -> None:
        japanese = "\n".join(line.japanese for line in self._lines.values())
        chinese = "\n".join(line.chinese for line in self._lines.values() if line.chinese)
        self.japanese_label.setText(japanese or "正在等待日语或英语语音…")
        if self._config.translation.backend == "none":
            self.chinese_label.hide()
        else:
            self.chinese_label.show()
            self.chinese_label.setText(chinese or "正在等待翻译…")
        self._resize_to_content_preserving_anchor()

    def _resize_to_content_preserving_anchor(self) -> None:
        """调整字幕框高度时保持用户拖动后的底部中心位置不变。"""
        current = self.frameGeometry()
        anchor_x = current.center().x()
        anchor_y = current.bottom()
        target_width = self._config.overlay.width

        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        target_height = self.heightForWidth(target_width)
        if target_height < 0:
            target_height = self.sizeHint().height()
        self.resize(target_width, max(1, target_height))
        self.move(
            anchor_x - (self.width() - 1) // 2,
            anchor_y - self.height() + 1,
        )

    def _apply_overlay_style(self) -> None:
        alpha = round(self._config.overlay.opacity * 255)
        font_size = self._config.overlay.font_size
        self.card.setStyleSheet(
            f"QFrame {{ background-color: rgba(12, 12, 16, {alpha}); "
            "border-radius: 14px; }} "
            "QPushButton { color: white; background: rgba(255,255,255,35); "
            "border: 1px solid rgba(255,255,255,55); border-radius: 5px; "
            "padding: 4px 10px; } "
            "QPushButton:hover { background: rgba(255,255,255,65); }"
        )
        self.japanese_label.setStyleSheet(
            f"color: #ffffff; font-size: {font_size}px; font-weight: 600; "
            "background: transparent;"
        )
        self.chinese_label.setStyleSheet(
            f"color: #ffe08a; font-size: {font_size + 2}px; font-weight: 700; "
            "background: transparent;"
        )

    def _build_tray_icon(self) -> None:
        self.tray_icon = QtWidgets.QSystemTrayIcon(self)
        icon = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaVolume)
        self.tray_icon.setIcon(icon)
        menu = QtWidgets.QMenu()
        toggle_action = menu.addAction("开始 / 停止")
        toggle_action.triggered.connect(self.toggle_running)
        show_action = menu.addAction("显示字幕窗")
        show_action.triggered.connect(self._restore_window)
        self.click_through_action = menu.addAction("点击穿透")
        self.click_through_action.setCheckable(True)
        self.click_through_action.toggled.connect(self.set_click_through)
        menu.addSeparator()
        quit_action = menu.addAction("退出")
        quit_action.triggered.connect(QtWidgets.QApplication.instance().quit)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

    def _restore_window(self) -> None:
        self.show()
        self.raise_()
        if not self._config.overlay.click_through:
            self.activateWindow()

    def _on_tray_activated(
        self, reason: QtWidgets.QSystemTrayIcon.ActivationReason
    ) -> None:
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self._restore_window()

    def _move_to_bottom_center(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        x = available.x() + (available.width() - self.width()) // 2
        y = available.bottom() - self.height() - 48
        self.move(max(available.x(), x), max(available.y(), y))
