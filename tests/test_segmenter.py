import unittest

import numpy as np

from nico_live_subtitle.config import AudioConfig
from nico_live_subtitle.segmenter import SpeechSegmenter


class SpeechSegmenterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AudioConfig(
            sample_rate=1_000,
            block_ms=100,
            energy_threshold=0.01,
            silence_ms=200,
            partial_ms=300,
            max_utterance_ms=1_000,
            pre_roll_ms=100,
        )
        self.silence = np.zeros(100, dtype=np.float32)
        self.speech = np.full(100, 0.1, dtype=np.float32)

    def test_emits_partial_then_final_after_silence(self) -> None:
        segmenter = SpeechSegmenter(self.config, sample_rate=1_000)
        self.assertEqual([], segmenter.push(self.silence))
        self.assertEqual([], segmenter.push(self.speech))
        partial = segmenter.push(self.speech)
        self.assertEqual(1, len(partial))
        self.assertFalse(partial[0].is_final)
        self.assertEqual(300, partial[0].samples.size)

        self.assertEqual([], segmenter.push(self.silence))
        final = segmenter.push(self.silence)
        self.assertEqual(1, len(final))
        self.assertTrue(final[0].is_final)
        self.assertEqual(partial[0].utterance_id, final[0].utterance_id)

    def test_force_splits_long_utterance(self) -> None:
        segmenter = SpeechSegmenter(self.config, sample_rate=1_000)
        events = []
        for _ in range(10):
            events.extend(segmenter.push(self.speech))
        self.assertTrue(events[-1].is_final)
        self.assertEqual(1_000, events[-1].samples.size)

    def test_flush_finishes_active_utterance(self) -> None:
        segmenter = SpeechSegmenter(self.config, sample_rate=1_000)
        segmenter.push(self.speech)
        final = segmenter.flush()
        self.assertIsNotNone(final)
        assert final is not None
        self.assertTrue(final.is_final)
        self.assertIsNone(segmenter.flush())

    def test_probability_provider_controls_segmentation(self) -> None:
        class FakeProbability:
            def __init__(self) -> None:
                self.values = iter([0.1, 0.9, 0.9, 0.1, 0.1])

            def score(self, samples: np.ndarray) -> float:
                return next(self.values)

        self.config.vad_threshold = 0.5
        segmenter = SpeechSegmenter(
            self.config,
            sample_rate=1_000,
            speech_probability=FakeProbability(),
        )
        events = []
        for _ in range(5):
            events.extend(segmenter.push(self.speech))
        self.assertTrue(events[-1].is_final)

    def test_pre_roll_silence_does_not_count_as_speech(self) -> None:
        self.config.min_speech_ms = 200
        segmenter = SpeechSegmenter(self.config, sample_rate=1_000)

        self.assertEqual([], segmenter.push(self.silence))
        self.assertEqual([], segmenter.push(self.speech))
        partial = segmenter.push(self.silence)
        self.assertEqual(1, len(partial))
        self.assertFalse(partial[0].is_final)
        self.assertEqual([], segmenter.push(self.silence))
        self.assertFalse(segmenter.active)


if __name__ == "__main__":
    unittest.main()
