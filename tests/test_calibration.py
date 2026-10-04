import unittest
import numpy as np
import cv2

from src.calibration import (
    calculate_mm_per_pixel,
    calibrate_measurements,
    get_calibration_scale,
    detect_aruco_calibration,
    rectify_aruco_marker,
)


class TestCalibration(unittest.TestCase):
    def test_calculate_mm_per_pixel_valid(self):
        scale = calculate_mm_per_pixel(50.0, 1000.0)
        self.assertAlmostEqual(scale, 0.05)

    def test_calculate_mm_per_pixel_invalid(self):
        with self.assertRaises(ValueError):
            calculate_mm_per_pixel(0, 100)
        with self.assertRaises(ValueError):
            calculate_mm_per_pixel(50, 0)
        with self.assertRaises(ValueError):
            calculate_mm_per_pixel(-10, 100)

    def test_calibrate_measurements(self):
        mock_meas = {
            "max_width_px": 10.0,
            "mean_width_px": 5.0,
            "median_width_px": 4.0,
            "std_width_px": 1.0,
            "total_length_px": 200.0,
            "components": [
                {"id": 1, "length_px": 120.0},
                {"id": 2, "length_px": 80.0},
            ],
        }
        scale = 0.05
        calibrated = calibrate_measurements(mock_meas, scale)

        self.assertAlmostEqual(calibrated["max_width_mm"], 0.5)
        self.assertAlmostEqual(calibrated["mean_width_mm"], 0.25)
        self.assertAlmostEqual(calibrated["median_width_mm"], 0.20)
        self.assertAlmostEqual(calibrated["std_width_mm"], 0.05)
        self.assertAlmostEqual(calibrated["total_length_mm"], 10.0)

        # Ensure original pixel values are preserved
        self.assertEqual(calibrated["max_width_px"], 10.0)
        self.assertEqual(calibrated["total_length_px"], 200.0)

        # Check metadata
        self.assertTrue(calibrated["calibration"]["is_calibrated"])
        self.assertAlmostEqual(calibrated["calibration"]["mm_per_pixel"], 0.05)
        self.assertAlmostEqual(calibrated["calibration"]["pixels_per_mm"], 20.0)

        # Check components
        self.assertAlmostEqual(calibrated["components"][0]["length_mm"], 6.0)
        self.assertAlmostEqual(calibrated["components"][1]["length_mm"], 4.0)

    def test_get_calibration_scale(self):
        self.assertIsNone(get_calibration_scale())
        self.assertAlmostEqual(get_calibration_scale(mm_per_pixel=0.05), 0.05)
        self.assertAlmostEqual(
            get_calibration_scale(reference_length_mm=100.0, reference_length_px=2000.0),
            0.05,
        )

        with self.assertRaises(ValueError):
            get_calibration_scale(mm_per_pixel=0.05, reference_length_mm=100.0)

        with self.assertRaises(ValueError):
            get_calibration_scale(reference_length_mm=100.0)

        with self.assertRaises(ValueError):
            get_calibration_scale(mm_per_pixel=-1.0)

    def test_aruco_detection_and_rectification(self):
        # Generate marker with white border for robust detection
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        raw_marker = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)
        bordered = cv2.copyMakeBorder(raw_marker, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

        res = detect_aruco_calibration(bordered, marker_size_mm=50.0)
        self.assertTrue(res["is_calibrated"])
        self.assertEqual(res["marker_id"], 0)
        self.assertIsNotNone(res["mm_per_pixel"])
        self.assertAlmostEqual(res["marker_size_mm"], 50.0)
        self.assertEqual(res["rectified_marker"].shape, (600, 600))


if __name__ == "__main__":
    unittest.main()
