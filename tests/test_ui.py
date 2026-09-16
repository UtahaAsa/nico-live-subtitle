import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from nico_live_subtitle.config import AppConfig
from nico_live_subtitle.pipeline import TranscriptUpdate, TranslationUpdate
from nico_live_subtitle.ui import OverlayWindow


class OverlayWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QtWidgets.QApplication.instance()
        if cls.application is None:
            cls.application = QtWidgets.QApplication([])

    def test_content_growth_preserves_bottom_center_anchor(self) -> None:
        config = AppConfig()
        config.overlay.width = 600
        window = OverlayWindow(config)
        window.move(240, 500)
        original = window.frameGeometry()
        anchor = (original.center().x(), original.bottom())

        source = "これはとても長いアニメの台詞です。" * 30
        window._on_transcript(TranscriptUpdate(1, source, True))
        window._on_translation(
            TranslationUpdate(1, source, "这是一段很长的动画台词。" * 30, True)
        )
        self.application.processEvents()

        resized = window.frameGeometry()
        self.assertGreater(resized.height(), original.height())
        self.assertEqual(anchor, (resized.center().x(), resized.bottom()))
        window.close()


if __name__ == "__main__":
    unittest.main()
