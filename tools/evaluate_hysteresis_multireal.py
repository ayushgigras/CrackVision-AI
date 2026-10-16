#!/usr/bin/env python3
"""Multi-image validation of CrackGauge's experimental hysteresis_recovery profile.

Evaluates baseline (Otsu, min_area=80) vs. hysteresis_recovery (low=25, high=48, min_area=200)
across real concrete images from data/real_images/ and sample image data/sample_images/crack_synthetic_01.jpg.
Excludes the tuning image (concrete-temperature-crack.jpg).

Calculates:
  - Crack pixels, coverage percentage
  - Connected component count
  - Maximum and mean crack widths
  - Total length (both without gap closing and with gap closing)
  - Ground-truth segmentation metrics (only where verified ground truth exists)
  - Qualitative metrics (continuity, texture false-positive detection)

Saves all outputs to data/outputs/evaluation_hysteresis_multireal/.
Preserves all previous experiment artifacts.

Usage:
    python -m tools.evaluate_hysteresis_multireal
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.preprocessing import run_preprocessing_pipeline
from src.segmentation import (
    binarize_ridge,
    create_crack_overlay,
    enhance_ridges_frangi,
    filter_crack_components,
    select_segmentation_profile,
)
from src.measurement import (
    measure_crack_lengths,
    measure_crack_widths,
)
from src.validation import segmentation_metrics

# Directory paths
REAL_IMAGES_DIR = _project_root / "data" / "real_images"
SAMPLE_IMAGES_DIR = _project_root / "data" / "sample_images"
GROUND_TRUTH_DIR = _project_root / "data" / "ground_truth"
OUTPUT_DIR = _project_root / "data" / "outputs" / "evaluation_hysteresis_multireal"

# Target parameters (strictly matching specification)
LOW_THRESH = 25
HIGH_THRESH = 48
HYST_MIN_AREA = 200
BASELINE_MIN_AREA = 80


def discover_evaluation_images() -> list[tuple[Path, Path | None]]:
    """Collect evaluation image list and matching ground-truth paths.

    Excludes the tuning image 'concrete-temperature-crack.jpg'.
    """
    image_list: list[Path] = []

    # 1. Real concrete images
    if REAL_IMAGES_DIR.is_dir():
        for p in sorted(REAL_IMAGES_DIR.glob("*.jpg")):
            image_list.append(p)

    # 2. Sample images (excluding tuning image)
    if SAMPLE_IMAGES_DIR.is_dir():
        for p in sorted(SAMPLE_IMAGES_DIR.glob("*.jpg")):
            if p.name != "concrete-temperature-crack.jpg":
                image_list.append(p)

    pairs: list[tuple[Path, Path | None]] = []
    for img_path in image_list:
        gt_path = GROUND_TRUTH_DIR / f"{img_path.stem}_mask.png"
        if gt_path.is_file():
            pairs.append((img_path, gt_path))
        else:
            pairs.append((img_path, None))

    return pairs


def compute_metrics(mask: np.ndarray, label: str) -> dict:
    """Compute width, length, and topological metrics for a binary mask."""
    crack_pixels = int(np.sum(mask > 0))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    coverage_pct = (crack_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    components = max(0, num_labels - 1)

    width_info = measure_crack_widths(mask)
    length_nogap = measure_crack_lengths(mask, enable_gap_closing=False)
    length_gap = measure_crack_lengths(mask, enable_gap_closing=True)

    # Find size of largest component to measure continuity
    largest_comp_px = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if components > 0 else 0
    continuity_ratio = round(largest_comp_px / crack_pixels, 4) if crack_pixels > 0 else 0.0

    return {
        "label": label,
        "crack_pixels": crack_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 4),
        "connected_components": components,
        "largest_component_pixels": largest_comp_px,
        "continuity_ratio": continuity_ratio,
        "max_width_px": round(float(width_info["max_width_px"]), 2),
        "mean_width_px": round(float(width_info["mean_width_px"]), 2),
        "total_length_no_gap_px": round(float(length_nogap["total_length_px"]), 2),
        "length_components_no_gap": int(length_nogap["component_count"]),
        "total_length_with_gap_px": round(float(length_gap["total_length_px"]), 2),
        "length_components_with_gap": int(length_gap["component_count"]),
    }


def evaluate_single_image(image_path: Path, gt_path: Path | None) -> dict:
    """Run baseline and hysteresis evaluation on one image."""
    image_name = image_path.name
    stem = image_path.stem

    # 1. Preprocessing
    preproc = run_preprocessing_pipeline(str(image_path), save_steps=False)
    original = preproc["original"]
    enhanced = preproc["enhanced"]
    adaptive_mask = preproc.get("crack_mask")

    # 2. Frangi Filter (shared)
    _, ridge_norm = enhance_ridges_frangi(
        enhanced, sigmas=(1, 2, 3, 4), black_ridges=True
    )

    # 3. Baseline profile (Otsu + min_area=80)
    ridge_binary_otsu = binarize_ridge(ridge_norm, method="otsu")
    cleaned_otsu, _ = filter_crack_components(
        ridge_binary_otsu, min_area=BASELINE_MIN_AREA
    )
    baseline_mask = select_segmentation_profile(
        cleaned_otsu, adaptive_mask, profile="baseline"
    )
    baseline_overlay = create_crack_overlay(
        original, baseline_mask, color=(0, 0, 255), alpha=0.65
    )
    baseline_metrics = compute_metrics(baseline_mask, "baseline")

    # 4. Hysteresis profile (low=25, high=48, min_area=200)
    hyst_mask = select_segmentation_profile(
        cleaned_otsu,
        adaptive_mask,
        profile="hysteresis_recovery",
        ridge_norm=ridge_norm,
        hysteresis_low=LOW_THRESH,
        hysteresis_high=HIGH_THRESH,
        hysteresis_min_area=HYST_MIN_AREA,
    )
    hyst_overlay = create_crack_overlay(
        original, hyst_mask, color=(0, 200, 0), alpha=0.65
    )
    hyst_metrics = compute_metrics(
        hyst_mask, f"hysteresis_low{LOW_THRESH}_high{HIGH_THRESH}_area{HYST_MIN_AREA}"
    )

    # 5. Difference visualization
    baseline_bool = baseline_mask > 0
    hyst_bool = hyst_mask > 0
    only_baseline = baseline_bool & ~hyst_bool
    only_hyst = hyst_bool & ~baseline_bool
    common = baseline_bool & hyst_bool

    diff_vis = original.copy()
    diff_vis[common] = (0, 255, 255)       # Yellow: detected by both
    diff_vis[only_baseline] = (0, 0, 255)  # Red: baseline only
    diff_vis[only_hyst] = (0, 255, 0)      # Green: hysteresis only

    # 6. Ground truth comparison if available
    gt_metrics_baseline = None
    gt_metrics_hysteresis = None
    if gt_path is not None and gt_path.is_file():
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is not None and gt_mask.shape == baseline_mask.shape:
            gt_metrics_baseline = segmentation_metrics(baseline_mask, gt_mask)
            gt_metrics_hysteresis = segmentation_metrics(hyst_mask, gt_mask)

    # 7. Save image outputs
    img_out_dir = OUTPUT_DIR / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_out_dir / "baseline_mask.png"), baseline_mask)
    cv2.imwrite(str(img_out_dir / "baseline_overlay.jpg"), baseline_overlay)
    cv2.imwrite(str(img_out_dir / "hysteresis_mask.png"), hyst_mask)
    cv2.imwrite(str(img_out_dir / "hysteresis_overlay.jpg"), hyst_overlay)
    cv2.imwrite(str(img_out_dir / "diff_visualization.jpg"), diff_vis)

    # Compute comparison deltas
    pixel_delta = hyst_metrics["crack_pixels"] - baseline_metrics["crack_pixels"]
    length_nogap_delta = round(
        hyst_metrics["total_length_no_gap_px"] - baseline_metrics["total_length_no_gap_px"], 2
    )
    length_gap_delta = round(
        hyst_metrics["total_length_with_gap_px"] - baseline_metrics["total_length_with_gap_px"], 2
    )

    return {
        "image_filename": image_name,
        "image_stem": stem,
        "has_ground_truth": gt_path is not None,
        "ground_truth_filename": gt_path.name if gt_path else None,
        "baseline": baseline_metrics,
        "hysteresis": hyst_metrics,
        "difference": {
            "common_pixels": int(np.sum(common)),
            "baseline_only_pixels": int(np.sum(only_baseline)),
            "hysteresis_only_pixels": int(np.sum(only_hyst)),
            "pixel_delta": pixel_delta,
            "length_no_gap_delta_px": length_nogap_delta,
            "length_with_gap_delta_px": length_gap_delta,
        },
        "ground_truth_metrics": {
            "baseline": gt_metrics_baseline,
            "hysteresis": gt_metrics_hysteresis,
        } if gt_metrics_baseline else None,
    }


def run_evaluation() -> None:
    """Run evaluation over all discovered images."""
    pairs = discover_evaluation_images()
    print("=" * 75)
    print("  CRACKGAUGE MULTI-IMAGE VALIDATION: HYSTERESIS vs BASELINE")
    print(f"  Images Found  : {len(pairs)} (excluding tuning image)")
    print(f"  Configuration : low={LOW_THRESH}, high={HIGH_THRESH}, min_area={HYST_MIN_AREA}")
    print(f"  Output Dir    : {OUTPUT_DIR}")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for idx, (img_path, gt_path) in enumerate(pairs, start=1):
        gt_info = f" (GT: {gt_path.name})" if gt_path else " (No GT)"
        print(f"[{idx:02d}/{len(pairs):02d}] Evaluating {img_path.name}{gt_info}...")
        res = evaluate_single_image(img_path, gt_path)
        results.append(res)

    # Save summary JSON
    summary_json_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(
            {
                "configuration": {
                    "hysteresis_low": LOW_THRESH,
                    "hysteresis_high": HIGH_THRESH,
                    "hysteresis_min_area": HYST_MIN_AREA,
                    "baseline_min_area": BASELINE_MIN_AREA,
                },
                "total_images_evaluated": len(results),
                "images_with_ground_truth": sum(1 for r in results if r["has_ground_truth"]),
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )

    # Save CSV summary table
    summary_csv_path = OUTPUT_DIR / "evaluation_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Has GT",
            "Baseline Pixels", "Hysteresis Pixels", "Pixel Delta",
            "Baseline Comps", "Hysteresis Comps",
            "Baseline Length (NoGap)", "Hysteresis Length (NoGap)", "Length Delta (NoGap)",
            "Baseline Length (Gap)", "Hysteresis Length (Gap)", "Length Delta (Gap)",
            "Baseline MeanWidth", "Hysteresis MeanWidth",
            "GT Dice (Baseline)", "GT Dice (Hysteresis)",
            "GT IoU (Baseline)", "GT IoU (Hysteresis)",
        ])
        for r in results:
            b = r["baseline"]
            h = r["hysteresis"]
            diff = r["difference"]
            gt = r["ground_truth_metrics"]
            gt_dice_b = f"{gt['baseline']['dice']:.4f}" if gt else "N/A"
            gt_dice_h = f"{gt['hysteresis']['dice']:.4f}" if gt else "N/A"
            gt_iou_b = f"{gt['baseline']['iou']:.4f}" if gt else "N/A"
            gt_iou_h = f"{gt['hysteresis']['iou']:.4f}" if gt else "N/A"

            writer.writerow([
                r["image_filename"],
                "Yes" if r["has_ground_truth"] else "No",
                b["crack_pixels"], h["crack_pixels"], diff["pixel_delta"],
                b["connected_components"], h["connected_components"],
                b["total_length_no_gap_px"], h["total_length_no_gap_px"], diff["length_no_gap_delta_px"],
                b["total_length_with_gap_px"], h["total_length_with_gap_px"], diff["length_with_gap_delta_px"],
                b["mean_width_px"], h["mean_width_px"],
                gt_dice_b, gt_dice_h,
                gt_iou_b, gt_iou_h,
            ])

    print("\n" + "=" * 75)
    print(f"  EVALUATION COMPLETE! Output saved to: {OUTPUT_DIR}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Summary CSV : {summary_csv_path}")
    print("=" * 75)


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
