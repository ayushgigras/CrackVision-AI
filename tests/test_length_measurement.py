import unittest
import numpy as np

from src.measurement import measure_crack_lengths, compute_skeleton


class TestCrackLength(unittest.TestCase):

    def test_empty_crack_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = measure_crack_lengths(mask)

        self.assertEqual(result["total_length_px"], 0.0)
        self.assertEqual(result["component_count"], 0)

    def test_horizontal_line_length(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[49:52, 10:90] = 255

        result = measure_crack_lengths(
            mask,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        print("\n--- Horizontal Crack Evaluation ---")
        print(f"Measured length: {measured:.2f} px")

        self.assertGreater(measured, 0)

    def test_vertical_line_length(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 49:52] = 255

        # Inspect the automatically generated skeleton.
        skeleton = compute_skeleton(mask)

        ys, xs = np.where(skeleton > 0)

        print("\n--- Vertical Skeleton Debug ---")
        print(f"Skeleton pixels: {len(xs)}")
        print(f"Y range: {ys.min()} to {ys.max()}")
        print(f"X range: {xs.min()} to {xs.max()}")

        result = measure_crack_lengths(
            mask,
            skeleton=skeleton,
            enable_gap_closing=False
        )

        measured = result["total_length_px"]

        print("\n--- Vertical Crack Evaluation ---")
        print(f"Measured length: {measured:.2f} px")

        self.assertGreater(measured, 0)

    def test_horizontal_known_skeleton_length(self):
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

        print("\n--- Known Horizontal Skeleton ---")
        print("Expected: 79.00 px")
        print(f"Measured: {measured:.2f} px")

        self.assertAlmostEqual(measured, 79.0, places=2)

    def test_vertical_known_skeleton_length(self):
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

        print("\n--- Known Vertical Skeleton ---")
        print("Expected: 79.00 px")
        print(f"Measured: {measured:.2f} px")

        self.assertAlmostEqual(measured, 79.0, places=2)


if __name__ == "__main__":
    unittest.main()