"""Validation metrics for segmentation masks and crack measurements.

This module only evaluates supplied predictions against supplied references.
It does not infer ground truth from synthetic images or alter the production
preprocessing, segmentation, or measurement pipelines.

Command-line usage:
    python -m src.validation --prediction predicted_mask.png \
        --ground-truth ground_truth_mask.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _as_binary_mask(mask: np.ndarray, name: str) -> np.ndarray:
    """Validate and normalize an image-like mask to a boolean array."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D single-channel mask")
    if mask.size == 0:
        raise ValueError(f"{name} must not be empty")
    return mask > 0


def _validated_masks(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate mask types, dimensions, and shape compatibility."""
    predicted = _as_binary_mask(predicted_mask, "predicted_mask")
    ground_truth = _as_binary_mask(ground_truth_mask, "ground_truth_mask")
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            "predicted_mask and ground_truth_mask must have matching dimensions "
            f"(got {predicted.shape} and {ground_truth.shape})"
        )
    return predicted, ground_truth


def segmentation_metrics(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> dict[str, float | int]:
    """Calculate binary segmentation metrics for two masks.

    Non-zero pixels are treated as crack pixels. If both masks are empty,
    they are considered a perfect match and all metrics are 1.0. If only one
    mask is empty, the metrics are 0.0. This makes zero-denominator cases
    explicit and avoids fabricated or undefined scores.
    """
    predicted, ground_truth = _validated_masks(predicted_mask, ground_truth_mask)

    true_positive = int(np.count_nonzero(predicted & ground_truth))
    false_positive = int(np.count_nonzero(predicted & ~ground_truth))
    false_negative = int(np.count_nonzero(~predicted & ground_truth))
    true_negative = int(np.count_nonzero(~predicted & ~ground_truth))

    union = true_positive + false_positive + false_negative
    predicted_positive = true_positive + false_positive
    ground_truth_positive = true_positive + false_negative

    if union == 0:
        iou = dice = precision = recall = f1 = 1.0
    else:
        iou = true_positive / union
        dice_denominator = 2 * true_positive + false_positive + false_negative
        dice = (
            2 * true_positive / dice_denominator
            if dice_denominator
            else 0.0
        )
        precision = (
            true_positive / predicted_positive
            if predicted_positive
            else 0.0
        )
        recall = (
            true_positive / ground_truth_positive
            if ground_truth_positive
            else 0.0
        )
        f1_denominator = precision + recall
        f1 = (
            2 * precision * recall / f1_denominator
            if f1_denominator
            else 0.0
        )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def intersection_over_union(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Return intersection over union for two binary crack masks."""
    return float(segmentation_metrics(predicted_mask, ground_truth_mask)["iou"])


def dice_coefficient(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Return the Dice coefficient for two binary crack masks."""
    return float(segmentation_metrics(predicted_mask, ground_truth_mask)["dice"])


def precision_score(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Return pixel-level precision for two binary crack masks."""
    return float(segmentation_metrics(predicted_mask, ground_truth_mask)["precision"])


def recall_score(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Return pixel-level recall for two binary crack masks."""
    return float(segmentation_metrics(predicted_mask, ground_truth_mask)["recall"])


def f1_score(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float:
    """Return pixel-level F1 score for two binary crack masks."""
    return float(segmentation_metrics(predicted_mask, ground_truth_mask)["f1"])


def absolute_error(predicted: float, reference: float) -> float:
    """Return the absolute error between a prediction and reference value."""
    _validate_measurement(predicted, "predicted")
    _validate_measurement(reference, "reference")
    return abs(float(predicted) - float(reference))


def relative_error_percentage(
    predicted: float,
    reference: float,
) -> float | None:
    """Return absolute error as a percentage, or ``None`` for zero reference."""
    _validate_measurement(predicted, "predicted")
    _validate_measurement(reference, "reference")
    if reference == 0:
        return None
    return absolute_error(predicted, reference) / float(reference) * 100.0


def measurement_error(
    predicted: float,
    reference: float,
) -> dict[str, float | None]:
    """Return absolute and relative errors for one width or length value."""
    return {
        "predicted": _validate_measurement(predicted, "predicted"),
        "reference": _validate_measurement(reference, "reference"),
        "absolute_error": absolute_error(predicted, reference),
        "relative_error_percent": relative_error_percentage(predicted, reference),
    }


def aggregate_measurement_errors(
    predicted_values: Iterable[float],
    reference_values: Iterable[float],
) -> dict[str, float | int | None]:
    """Calculate MAE, RMSE, and relative errors over paired measurements."""
    predicted = _validated_measurements(predicted_values, "predicted_values")
    reference = _validated_measurements(reference_values, "reference_values")
    if len(predicted) != len(reference):
        raise ValueError(
            "predicted_values and reference_values must contain the same number "
            f"of measurements (got {len(predicted)} and {len(reference)})"
        )
    if not predicted:
        raise ValueError("at least one measurement pair is required")

    errors = np.asarray(predicted) - np.asarray(reference)
    absolute_errors = np.abs(errors)
    nonzero_references = [
        absolute_error_value / reference_value * 100.0
        for absolute_error_value, reference_value in zip(absolute_errors, reference)
        if reference_value != 0
    ]

    return {
        "count": len(predicted),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean_relative_error_percent": (
            float(np.mean(nonzero_references))
            if nonzero_references
            else None
        ),
        "zero_reference_count": len(reference) - len(nonzero_references),
    }


def _validate_measurement(value: float, name: str) -> float:
    """Validate a non-negative finite scalar measurement."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    if numeric_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric_value


def _validated_measurements(
    values: Iterable[float],
    name: str,
) -> list[float]:
    """Materialize and validate a sequence of measurements."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real numbers")
    return [_validate_measurement(value, name) for value in values]


def validate_mask_files(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
) -> dict[str, float | int]:
    """Load two grayscale mask files and calculate segmentation metrics."""
    prediction = cv2.imread(str(prediction_path), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(str(ground_truth_path), cv2.IMREAD_GRAYSCALE)
    if prediction is None:
        raise FileNotFoundError(f"Could not read prediction mask: {prediction_path}")
    if ground_truth is None:
        raise FileNotFoundError(
            f"Could not read ground-truth mask: {ground_truth_path}"
        )
    return segmentation_metrics(prediction, ground_truth)


def save_validation_csv(
    output_path: str | Path,
    segmentation: dict[str, float | int],
    width_error: dict[str, float | None] | None = None,
    length_error: dict[str, float | None] | None = None,
    measurement_unit: str = "px",
) -> None:
    """Save segmentation metrics and optional measurement errors as one CSV row.

    Measurement values are not computed here. If supplied, they must already
    be compared in the same unit, such as pixels (``px``) or millimetres
    (``mm``). This prevents calibrated and uncalibrated values being mixed.
    """
    if measurement_unit not in {"px", "mm"}:
        raise ValueError("measurement_unit must be 'px' or 'mm'")

    row: dict[str, float | int | str | None] = dict(segmentation)
    for label, errors in (("width", width_error), ("length", length_error)):
        if errors is None:
            continue
        row[f"predicted_{label}_{measurement_unit}"] = errors["predicted"]
        row[f"reference_{label}_{measurement_unit}"] = errors["reference"]
        row[f"{label}_absolute_error_{measurement_unit}"] = errors["absolute_error"]
        row[f"{label}_relative_error_percent"] = errors["relative_error_percent"]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Run mask validation from the command line and save CSV metrics."""
    parser = argparse.ArgumentParser(
        description="Compare a predicted crack mask with a ground-truth mask."
    )
    parser.add_argument("--prediction", required=True, help="Predicted mask path")
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Ground-truth binary mask path",
    )
    parser.add_argument(
        "--output-csv",
        default="data/outputs/validation_metrics.csv",
        help="CSV output path (default: data/outputs/validation_metrics.csv)",
    )
    parser.add_argument(
        "--predicted-width",
        type=float,
        help="Predicted crack width, in the unit selected by --measurement-unit",
    )
    parser.add_argument(
        "--reference-width",
        type=float,
        help="Reference crack width, in the unit selected by --measurement-unit",
    )
    parser.add_argument(
        "--predicted-length",
        type=float,
        help="Predicted crack length, in the unit selected by --measurement-unit",
    )
    parser.add_argument(
        "--reference-length",
        type=float,
        help="Reference crack length, in the unit selected by --measurement-unit",
    )
    parser.add_argument(
        "--measurement-unit",
        choices=("px", "mm"),
        default="px",
        help="Unit for supplied width/length values (default: px)",
    )
    args = parser.parse_args()
    metrics = validate_mask_files(args.prediction, args.ground_truth)

    def paired_measurement(
        predicted: float | None,
        reference: float | None,
        name: str,
    ) -> dict[str, float | None] | None:
        if (predicted is None) != (reference is None):
            parser.error(
                f"--predicted-{name} and --reference-{name} must be supplied together"
            )
        if predicted is None:
            return None
        return measurement_error(predicted, reference)

    width_error = paired_measurement(
        args.predicted_width, args.reference_width, "width"
    )
    length_error = paired_measurement(
        args.predicted_length, args.reference_length, "length"
    )
    save_validation_csv(
        args.output_csv,
        metrics,
        width_error=width_error,
        length_error=length_error,
        measurement_unit=args.measurement_unit,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Validation metrics saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
