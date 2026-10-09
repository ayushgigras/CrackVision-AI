"""Compare baseline and recovery segmentation profiles against a supplied mask.

Run from the repository root:
    python tools/compare_segmentation_profiles.py
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

from src.preprocessing import run_preprocessing_pipeline
from src.segmentation import run_segmentation_pipeline
from src.validation import segmentation_metrics


IMAGE_PATH = ROOT / "data" / "real_images" / "00001.jpg"
GROUND_TRUTH_PATH = ROOT / "data" / "ground_truth" / "00001_mask.png"
OUTPUT_PATH = ROOT / "data" / "outputs" / "00001_segmentation_profiles.png"


def _load_inputs() -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)
    ground_truth = cv2.imread(str(GROUND_TRUTH_PATH), cv2.IMREAD_GRAYSCALE)
    if image is None or ground_truth is None:
        raise FileNotFoundError("Could not load the image or ground-truth mask")
    if image.shape[:2] != ground_truth.shape:
        raise ValueError("Image and ground-truth mask must have matching dimensions")
    return image, ground_truth


def _overlay(predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    predicted_binary = predicted > 0
    ground_truth_binary = ground_truth > 0
    overlay = np.zeros((*predicted.shape, 3), dtype=np.uint8)
    overlay[predicted_binary & ground_truth_binary] = (0, 255, 0)
    overlay[predicted_binary & ~ground_truth_binary] = (255, 0, 0)
    overlay[~predicted_binary & ground_truth_binary] = (0, 0, 255)
    return overlay


def _run_profile(
    preprocessed: dict,
    ground_truth: np.ndarray,
    profile: str,
) -> tuple[np.ndarray, dict]:
    result = run_segmentation_pipeline(
        preprocessed,
        output_dir=None,
        segmentation_profile=profile,
        save_steps=False,
    )
    return result["final_mask"], segmentation_metrics(result["final_mask"], ground_truth)


def _print_metrics(name: str, metrics: dict[str, float | int]) -> None:
    print(name)
    for key in (
        "true_positive", "false_positive", "false_negative",
        "precision", "recall", "dice", "f1", "iou",
    ):
        print(f"  {key}: {metrics[key]}")


def main() -> None:
    original, ground_truth = _load_inputs()
    preprocessed = run_preprocessing_pipeline(
        str(IMAGE_PATH), output_dir=None, save_steps=False
    )
    baseline, baseline_metrics = _run_profile(preprocessed, ground_truth, "baseline")
    recovery, recovery_metrics = _run_profile(
        preprocessed, ground_truth, "adaptive_recovery"
    )
    _print_metrics("baseline", baseline_metrics)
    _print_metrics("adaptive_recovery", recovery_metrics)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = (
        (axes[0, 0], cv2.cvtColor(original, cv2.COLOR_BGR2RGB), "Original image", None),
        (axes[0, 1], baseline, "Baseline mask", "gray"),
        (axes[0, 2], recovery, "Adaptive recovery mask", "gray"),
        (axes[1, 0], _overlay(baseline, ground_truth), "Baseline TP / FP / FN", None),
        (axes[1, 1], _overlay(recovery, ground_truth), "Recovery TP / FP / FN", None),
        (axes[1, 2], ground_truth, "Ground-truth mask", "gray"),
    )
    for axis, image, title, cmap in panels:
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="green", label="TP"),
            Patch(facecolor="red", label="FP"),
            Patch(facecolor="blue", label="FN"),
        ],
        loc="lower right",
    )
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Comparison saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
