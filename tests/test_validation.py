import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.validation import (
    aggregate_measurement_errors,
    absolute_error,
    dice_coefficient,
    f1_score,
    intersection_over_union,
    measurement_error,
    precision_score,
    recall_score,
    relative_error_percentage,
    segmentation_metrics,
    save_validation_csv,
)


class TestSegmentationMetrics(unittest.TestCase):
    def test_metrics_for_matching_masks(self):
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        metrics = segmentation_metrics(mask, mask)

        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_metrics_for_partial_overlap(self):
        predicted = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        ground_truth = np.array([[255, 0], [255, 0]], dtype=np.uint8)

        self.assertAlmostEqual(intersection_over_union(predicted, ground_truth), 1 / 3)
        self.assertAlmostEqual(dice_coefficient(predicted, ground_truth), 0.5)
        self.assertAlmostEqual(precision_score(predicted, ground_truth), 0.5)
        self.assertAlmostEqual(recall_score(predicted, ground_truth), 0.5)
        self.assertAlmostEqual(f1_score(predicted, ground_truth), 0.5)

    def test_metrics_for_completely_different_non_empty_masks(self):
        predicted = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        ground_truth = np.array([[0, 0], [0, 255]], dtype=np.uint8)

        metrics = segmentation_metrics(predicted, ground_truth)

        self.assertEqual(metrics["iou"], 0.0)
        self.assertEqual(metrics["dice"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)

    def test_empty_masks_are_a_safe_perfect_match(self):
        empty = np.zeros((2, 2), dtype=np.uint8)

        metrics = segmentation_metrics(empty, empty)

        for name in ("iou", "dice", "precision", "recall", "f1"):
            self.assertEqual(metrics[name], 1.0)

    def test_one_empty_mask_has_zero_scores_without_division_errors(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        non_empty = np.array([[255, 0], [0, 0]], dtype=np.uint8)

        metrics = segmentation_metrics(non_empty, empty)

        for name in ("iou", "dice", "precision", "recall", "f1"):
            self.assertEqual(metrics[name], 0.0)

    def test_mask_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching dimensions"):
            segmentation_metrics(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((3, 2), dtype=np.uint8),
            )


class TestMeasurementErrors(unittest.TestCase):
    def test_single_measurement_errors(self):
        self.assertEqual(absolute_error(12.0, 10.0), 2.0)
        self.assertEqual(relative_error_percentage(12.0, 10.0), 20.0)
        self.assertIsNone(relative_error_percentage(2.0, 0.0))
        self.assertEqual(
            measurement_error(12.0, 10.0)["absolute_error"],
            2.0,
        )

    def test_aggregate_errors(self):
        result = aggregate_measurement_errors([11.0, 18.0], [10.0, 20.0])

        self.assertEqual(result["count"], 2)
        self.assertAlmostEqual(result["mae"], 1.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(result["mean_relative_error_percent"], 10.0)

    def test_aggregate_errors_handles_zero_references(self):
        result = aggregate_measurement_errors([0.0, 2.0], [0.0, 1.0])

        self.assertEqual(result["zero_reference_count"], 1)
        self.assertEqual(result["mean_relative_error_percent"], 100.0)

    def test_measurement_inputs_are_validated(self):
        with self.assertRaises(ValueError):
            absolute_error(-1.0, 2.0)
        with self.assertRaises(ValueError):
            aggregate_measurement_errors([1.0], [1.0, 2.0])

    def test_validation_csv_contains_metrics_and_pixel_errors(self):
        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.csv"
            save_validation_csv(
                output_path,
                {"iou": 0.5, "dice": 2 / 3},
                width_error=measurement_error(12.0, 10.0),
                length_error=measurement_error(90.0, 100.0),
                measurement_unit="px",
            )

            content = output_path.read_text(encoding="utf-8")

        self.assertIn("iou,dice", content)
        self.assertIn("predicted_width_px", content)
        self.assertIn("reference_length_px", content)
        self.assertIn("width_absolute_error_px", content)
        self.assertIn("length_relative_error_percent", content)
        self.assertIn("12.0", content)


if __name__ == "__main__":
    unittest.main()
