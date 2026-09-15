import unittest

import numpy as np

from nico_live_subtitle.audio import resample_mono


class ResampleMonoTest(unittest.TestCase):
    def test_stereo_is_mixed_and_resampled(self) -> None:
        left = np.linspace(-1.0, 1.0, 4_800, dtype=np.float32)
        stereo = np.column_stack((left, -left))
        result = resample_mono(stereo, 48_000, 16_000)
        self.assertEqual(1_600, result.size)
        self.assertEqual(np.float32, result.dtype)
        self.assertLess(float(np.max(np.abs(result))), 1e-6)

    def test_invalid_rank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "一维或二维"):
            resample_mono(np.zeros((2, 2, 2), dtype=np.float32), 48_000)


if __name__ == "__main__":
    unittest.main()

