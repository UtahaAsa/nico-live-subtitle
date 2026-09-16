import unittest

from nico_live_subtitle.text import TranscriptStabilizer, split_japanese_sentences


class JapaneseSentenceTest(unittest.TestCase):
    def test_split_keeps_closing_quote_with_sentence(self) -> None:
        self.assertEqual(
            ["「行くぞ！」", "まだ終わってない"],
            split_japanese_sentences("「行くぞ！」まだ終わってない"),
        )

    def test_partial_only_commits_sentences_before_unstable_tail(self) -> None:
        stabilizer = TranscriptStabilizer()
        complete, pending = stabilizer.push("今日は晴れだ。散歩に", False)
        self.assertEqual(["今日は晴れだ。"], complete)
        self.assertEqual("散歩に", pending)

        complete, pending = stabilizer.push(
            "今日は晴れだ。散歩に行きましょう。", True
        )
        self.assertEqual(["散歩に行きましょう。"], complete)
        self.assertEqual("", pending)

    def test_overlap_tail_is_removed(self) -> None:
        stabilizer = TranscriptStabilizer()
        stabilizer.push("これは前の文。次の", False)
        complete, _ = stabilizer.push("前の文。次の文です。", True)
        self.assertEqual(["次の文です。"], complete)

    def test_small_recognition_change_is_fuzzy_deduplicated(self) -> None:
        stabilizer = TranscriptStabilizer()
        stabilizer.push("今日はとてもいい天気です。続き", False)
        complete, _ = stabilizer.push("今日はとってもいい天気です。次です。", True)
        self.assertEqual(["次です。"], complete)

    def test_english_period_commits_before_unstable_tail(self) -> None:
        stabilizer = TranscriptStabilizer()
        complete, pending = stabilizer.push(
            "Thanks for joining. Today we're playing", False
        )
        self.assertEqual(["Thanks for joining."], complete)
        self.assertEqual("Today we're playing", pending)

    def test_english_abbreviation_is_not_split(self) -> None:
        self.assertEqual(
            ["Dr. Smith is here.", "Welcome!"],
            split_japanese_sentences("Dr. Smith is here. Welcome!"),
        )


if __name__ == "__main__":
    unittest.main()
