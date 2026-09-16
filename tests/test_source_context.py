import unittest

from nico_live_subtitle.config import LexiconConfig
from nico_live_subtitle.lexicon import load_lexicon_catalog
from nico_live_subtitle.source_context import detect_source_context


class SourceContextTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_lexicon_catalog("lexicons")

    def test_auto_matches_anime_from_browser_title(self) -> None:
        context = detect_source_context(
            LexiconConfig(profile="auto"),
            self.catalog,
            ["Re:ゼロから始める異世界生活 4th season - ニコニコ動画"],
        )
        self.assertEqual("anime", context.kind)
        self.assertEqual("re-zero", context.profile)

    def test_auto_falls_back_to_bilingual_live_mode(self) -> None:
        context = detect_source_context(
            LexiconConfig(profile="auto"),
            self.catalog,
            ["English gaming stream - YouTube"],
        )
        self.assertEqual("live", context.kind)
        self.assertEqual("", context.profile)

    def test_auto_detects_unlisted_niconico_anime_as_generic_anime(self) -> None:
        context = detect_source_context(
            LexiconConfig(profile="auto"),
            self.catalog,
            ["まだ詞庫にない作品 第7話 - ニコニコ動画"],
        )
        self.assertEqual("anime", context.kind)
        self.assertEqual("", context.profile)

    def test_live_hint_wins_over_generic_niconico_hint(self) -> None:
        context = detect_source_context(
            LexiconConfig(profile="auto"),
            self.catalog,
            ["雑談生放送 - ニコニコ動画"],
        )
        self.assertEqual("live", context.kind)

    def test_manual_live_mode_skips_title_matching(self) -> None:
        context = detect_source_context(
            LexiconConfig(profile="live"),
            self.catalog,
            ["Re:ゼロから始める異世界生活"],
        )
        self.assertEqual("live", context.kind)


if __name__ == "__main__":
    unittest.main()
