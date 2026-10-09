"""Create a visual comparison of a predicted mask and ground-truth mask.

Run from the repository root:
    python tools/visualize_validation.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation import segmentation_metrics


ORIGINAL_PATH = ROOT / "data" / "real_images" / "00001.jpg"
PREDICTION_PATH = ROOT / "data" / "outputs" / "00001_08_final_crack_mask.png"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth" / "00001_mask.png"
OUTPUT_PATH = ROOT / "data" / "outputs" / "00001_validation_comparison.png"


def load_validation_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the original image and both grayscale masks."""
    original = cv2.imread(str(ORIGINAL_PATH), cv2.IMREAD_COLOR)
    predicted = cv2.imread(str(PREDICTION_PATH), cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(str(GROUND_TRUTH_PATH), cv2.IMREAD_GRAYSCALE)

    missing = [
        str(path)
        for path, image in (
            (ORIGINAL_PATH, original),
            (PREDICTION_PATH, predicted),
            (GROUND_TRUTH_PATH, ground_truth),
        )
        if image is None
    ]
    if missing:
        raise FileNotFoundError(f"Could not load required file(s): {', '.join(missing)}")

    assert original is not None
    assert predicted is not None
    assert ground_truth is not None
    original_shape = original.shape[:2]
    if predicted.shape != original_shape or ground_truth.shape != original_shape:
        raise ValueError(
            "Original image, predicted mask, and ground-truth mask must have "
            f"matching dimensions; got {original_shape}, {predicted.shape}, "
            f"and {ground_truth.shape}"
        )
    return original, predicted, ground_truth


def build_comparison_overlay(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> np.ndarray:
    """Return an RGB TP/FP/FN overlay using green, red, and blue."""
    if predicted_mask.shape != ground_truth_mask.shape:
        raise ValueError("Predicted and ground-truth masks must have matching dimensions")

    predicted = predicted_mask > 0
    ground_truth = ground_truth_mask > 0
    overlay = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    overlay[predicted & ground_truth] = (0, 255, 0)
    overlay[predicted & ~ground_truth] = (255, 0, 0)
    overlay[~predicted & ground_truth] = (0, 0, 255)
    return overlay


def create_comparison_figure(
    original_bgr: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    metrics: dict[str, float | int],
) -> plt.Figure:
    """Create the four-panel validation comparison figure."""
    overlay = build_comparison_overlay(predicted_mask, ground_truth_mask)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle("CrackGauge Validation Comparison", fontsize=16, fontweight="bold")

    panels = (
        (axes[0, 0], cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), "Original image", None),
        (axes[0, 1], ground_truth_mask, "Ground-truth mask", "gray"),
        (axes[1, 0], predicted_mask, "Predicted mask", "gray"),
        (axes[1, 1], overlay, "TP / FP / FN comparison", None),
    )
    for axis, image, title, cmap in panels:
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")

    legend = [
        Patch(facecolor="green", label=f"TP ({metrics['true_positive']})"),
        Patch(facecolor="red", label=f"FP ({metrics['false_positive']})"),
        Patch(facecolor="blue", label=f"FN ({metrics['false_negative']})"),
    ]
    axes[1, 1].legend(handles=legend, loc="lower right", framealpha=0.9)
    figure.tight_layout()
    return figure


def main() -> None:
    """Generate the real-image validation comparison and print metrics."""
    original, predicted, ground_truth = load_validation_inputs()
    metrics = segmentation_metrics(predicted, ground_truth)
    figure = create_comparison_figure(original, predicted, ground_truth, metrics)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Validation comparison saved to: {OUTPUT_PATH}")
    for name in (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
    ):
        print(f"{name}: {metrics[name]}")


if __name__ == "__main__":
    main()
