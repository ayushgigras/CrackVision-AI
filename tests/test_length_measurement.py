import unittest
import numpy as np

from src.measurement import measure_crack_lengths, compute_skeleton


class TestCrackLength(unittest.TestCase):

    def test_empty_crack_mask(self):
        """An empty mask should have zero length and zero components."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = measure_crack_lengths(mask)

        self.assertEqual(result["total_length_px"], 0.0)
        self.assertEqual(result["component_count"], 0)

    def test_horizontal_line_length(self):
        """Automatically skeletonized horizontal crack should have positive length."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[49:52, 10:90] = 255

        result = measure_crack_lengths(
            mask,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        self.assertGreater(measured, 0.0)

    def test_vertical_line_length(self):
        """Automatically skeletonized vertical crack should have positive length."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 49:52] = 255

        skeleton = compute_skeleton(mask)

        ys, xs = np.where(skeleton > 0)

        # The generated skeleton should be non-empty and vertical.
        self.assertGreater(len(xs), 0)
        self.assertTrue(np.all(xs == 50))
        self.assertEqual(len(np.unique(ys)), len(ys))

        result = measure_crack_lengths(
            mask,
            skeleton=skeleton,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        # A continuous straight path of N pixels has N - 1 pixel steps.
        expected_length = float(len(ys) - 1)
        self.assertAlmostEqual(measured, expected_length, places=2)

    def test_horizontal_known_skeleton_length(self):
        """A supplied horizontal skeleton should measure exactly 79 pixels."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[49:52, 10:90] = 255

        skeleton = np.zeros_like(mask)
        skeleton[50, 10:90] = 255

        result = measure_crack_lengths(
            mask,
            skeleton=skeleton,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        self.assertAlmostEqual(measured, 79.0, places=2)

    def test_vertical_known_skeleton_length(self):
        """A supplied vertical skeleton should measure exactly 79 pixels."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 49:52] = 255

        skeleton = np.zeros_like(mask)
        skeleton[10:90, 50] = 255

        result = measure_crack_lengths(
            mask,
            skeleton=skeleton,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        self.assertAlmostEqual(measured, 79.0, places=2)


if __name__ == "__main__":
    unittest.main()