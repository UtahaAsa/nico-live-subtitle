import json
import tempfile
import unittest
from pathlib import Path

from nico_live_subtitle.config import LexiconConfig
from nico_live_subtitle.lexicon import build_lexicon_bundle, load_lexicon_catalog


class AnimeLexiconTest(unittest.TestCase):
    def test_built_in_re_zero_profile_merges_common_terms(self) -> None:
        catalog = load_lexicon_catalog("lexicons")
        self.assertIn("re-zero", {item.id for item in catalog})

        bundle = build_lexicon_bundle(
            LexiconConfig(profile="re-zero"),
            custom_hotwords="追加名",
            custom_glossary="エミリア=艾米莉亚",
        )

        self.assertEqual("Re:从零开始的异世界生活", bundle.title)
        self.assertIn("ナツキ・スバル", bundle.hotwords)
        self.assertIn("追加名", bundle.hotwords)
        self.assertIn("異世界=异世界", bundle.glossary)
        self.assertIn("エミリア=艾米莉亚", bundle.glossary)
        self.assertNotIn("エミリア=爱蜜莉雅", bundle.glossary)

    def test_invalid_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir="work") as directory:
            path = Path(directory) / "broken.json"
            path.write_text(
                json.dumps({"schema_version": 2, "id": "broken"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_lexicon_catalog(directory)


if __name__ == "__main__":
    unittest.main()
