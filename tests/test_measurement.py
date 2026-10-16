import unittest
import numpy as np

from src.measurement import measure_crack_widths


class TestCrackWidth(unittest.TestCase):

    def test_empty_crack_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = measure_crack_widths(mask)

        self.assertEqual(result["max_width_px"], 0.0)
        self.assertEqual(result["mean_width_px"], 0.0)
        self.assertEqual(result["median_width_px"], 0.0)
        self.assertEqual(result["skeleton_points"], 0)

    def test_horizontal_cracks_different_widths(self):
        thicknesses = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        print("\n--- Horizontal Crack Width Evaluation ---")

        for thickness in thicknesses:
            with self.subTest(thickness=thickness):
                mask = np.zeros((100, 100), dtype=np.uint8)

                start_row = (100 - thickness) // 2
                mask[start_row:start_row + thickness, 10:90] = 255

                result = measure_crack_widths(mask)

                measured = result["median_width_px"]
                error = abs(thickness - measured)

                print(
                    f"Expected: {thickness:2d} px | "
                    f"Measured: {measured:.2f} px | "
                    f"Error: {error:.2f} px"
                )

                # Sanity check: the detected crack should have positive width.
                self.assertGreater(measured, 0)

    def test_vertical_crack_width(self):
        mask = np.zeros((100, 100), dtype=np.uint8)

        mask[10:90, 48:53] = 255

        result = measure_crack_widths(mask)

        measured = result["median_width_px"]
        error = abs(5 - measured)

        print("\n--- Vertical Crack Width Evaluation ---")
        print(f"Expected: 5 px | Measured: {measured:.2f} px")
        print(f"Absolute Error: {error:.2f} px")

        self.assertGreater(measured, 0)
        self.assertLessEqual(error, 1.0)


if __name__ == "__main__":
    unittest.main()