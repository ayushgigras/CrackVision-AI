#!/usr/bin/env python3
"""Controlled experiment comparing three hysteresis post-filtering strategies.

Strategies:
  Strategy A: Hysteresis min_area=200 (area cutoff only)
  Strategy B: Hysteresis min_area=80 (area cutoff only, no shape filter)
  Strategy C: Hysteresis min_area=80 + conservative shape filter
              (keeps area >= 200 unconditionally; for 80 <= area < 200,
               evaluates oriented aspect ratio and skeleton length with
               boundary-aware relaxed threshold)

Evaluates on the 20 test images:
  - 19 real images from data/real_images/
  - 1 sample image data/sample_images/crack_synthetic_01.jpg
  (concrete-temperature-crack.jpg is excluded from primary evaluation)

Ground truth metrics (Dice/F1, IoU, Precision, Recall) calculated for 00001.jpg.
Saves all artifacts in data/outputs/experiment_hysteresis_shape_filter/.

Usage:
    python -m tools.experiment_hysteresis_shape_filter
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

REAL_IMAGES_DIR = _project_root / "data" / "real_images"
SAMPLE_IMAGES_DIR = _project_root / "data" / "sample_images"
GROUND_TRUTH_DIR = _project_root / "data" / "ground_truth"
OUTPUT_DIR = _project_root / "data" / "outputs" / "experiment_hysteresis_shape_filter"

LOW_THRESH = 25
HIGH_THRESH = 48


def discover_evaluation_images() -> list[tuple[Path, Path | None]]:
    """Collect 20 evaluation images and any matching ground-truth."""
    image_list: list[Path] = []
    if REAL_IMAGES_DIR.is_dir():
        for p in sorted(REAL_IMAGES_DIR.glob("*.jpg")):
            image_list.append(p)
    if SAMPLE_IMAGES_DIR.is_dir():
        for p in sorted(SAMPLE_IMAGES_DIR.glob("*.jpg")):
            if p.name != "concrete-temperature-crack.jpg":
                image_list.append(p)

    pairs: list[tuple[Path, Path | None]] = []
    for img_path in image_list:
        gt_path = GROUND_TRUTH_DIR / f"{img_path.stem}_mask.png"
        pairs.append((img_path, gt_path if gt_path.is_file() else None))
    return pairs


def compute_metrics(mask: np.ndarray, label: str) -> dict:
    """Compute width, length, and component metrics for a binary mask."""
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

    return {
        "label": label,
        "crack_pixels": crack_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 4),
        "connected_components": components,
        "max_width_px": round(float(width_info["max_width_px"]), 2),
        "mean_width_px": round(float(width_info["mean_width_px"]), 2),
        "total_length_no_gap_px": round(float(length_nogap["total_length_px"]), 2),
        "length_components_no_gap": int(length_nogap["component_count"]),
        "total_length_with_gap_px": round(float(length_gap["total_length_px"]), 2),
        "length_components_with_gap": int(length_gap["component_count"]),
    }


def run_experiment() -> None:
    pairs = discover_evaluation_images()
    print("=" * 75)
    print("  CONTROLLED EXPERIMENT: HYSTERESIS FILTERING STRATEGIES")
    print(f"  Images Evaluated : {len(pairs)}")
    print(f"  Output Directory : {OUTPUT_DIR}")
    print("  Strategy A : min_area=200 (Area cutoff)")
    print("  Strategy B : min_area=80  (No shape filter)")
    print("  Strategy C : min_area=80  + shape-aware filter (boundary-aware)")
    print("=" * 75)

    dir_strat_a = OUTPUT_DIR / "strategy_A_min_area_200"
    dir_strat_b = OUTPUT_DIR / "strategy_B_min_area_80"
    dir_strat_c = OUTPUT_DIR / "strategy_C_shape_filter"

    for d in [dir_strat_a, dir_strat_b, dir_strat_c]:
        d.mkdir(parents=True, exist_ok=True)

    results_all = []

    for idx, (img_path, gt_path) in enumerate(pairs, start=1):
        stem = img_path.stem
        gt_info = f" (GT: {gt_path.name})" if gt_path else " (No GT)"
        print(f"[{idx:02d}/{len(pairs):02d}] Evaluating {img_path.name}{gt_info}...")

        preproc = run_preprocessing_pipeline(str(img_path), save_steps=False)
        original = preproc["original"]
        enhanced = preproc["enhanced"]
        adaptive_mask = preproc.get("crack_mask")

        # Shared Frangi ridge enhancement
        _, ridge_norm = enhance_ridges_frangi(
            enhanced, sigmas=(1, 2, 3, 4), black_ridges=True
        )

        # Baseline Otsu (reference)
        ridge_binary_otsu = binarize_ridge(ridge_norm, method="otsu")
        cleaned_otsu, _ = filter_crack_components(ridge_binary_otsu, min_area=80)
        baseline_mask = select_segmentation_profile(
            cleaned_otsu, adaptive_mask, profile="baseline"
        )
        base_metrics = compute_metrics(baseline_mask, "baseline")

        # Strategy A: min_area=200
        mask_a = select_segmentation_profile(
            cleaned_otsu, adaptive_mask, profile="hysteresis_recovery",
            ridge_norm=ridge_norm, hysteresis_low=LOW_THRESH, hysteresis_high=HIGH_THRESH,
            hysteresis_min_area=200, hysteresis_use_shape_filter=False,
        )
        overlay_a = create_crack_overlay(original, mask_a, color=(0, 200, 0), alpha=0.65)
        metrics_a = compute_metrics(mask_a, "strategy_A_min_area_200")

        # Strategy B: min_area=80 (no shape filter)
        mask_b = select_segmentation_profile(
            cleaned_otsu, adaptive_mask, profile="hysteresis_recovery",
            ridge_norm=ridge_norm, hysteresis_low=LOW_THRESH, hysteresis_high=HIGH_THRESH,
            hysteresis_min_area=80, hysteresis_use_shape_filter=False,
        )
        overlay_b = create_crack_overlay(original, mask_b, color=(255, 128, 0), alpha=0.65)
        metrics_b = compute_metrics(mask_b, "strategy_B_min_area_80")

        # Strategy C: min_area=80 + shape filter
        mask_c = select_segmentation_profile(
            cleaned_otsu, adaptive_mask, profile="hysteresis_recovery",
            ridge_norm=ridge_norm, hysteresis_low=LOW_THRESH, hysteresis_high=HIGH_THRESH,
            hysteresis_min_area=80, hysteresis_use_shape_filter=True,
            hysteresis_large_area_thresh=200,
            hysteresis_interior_aspect_thresh=2.2,
            hysteresis_boundary_aspect_thresh=1.6,
            hysteresis_min_skeleton_length=20,
        )
        overlay_c = create_crack_overlay(original, mask_c, color=(0, 255, 0), alpha=0.65)
        metrics_c = compute_metrics(mask_c, "strategy_C_shape_filter")

        # Difference visualization between Strategy C and Strategy A:
        # Yellow: detected by both A and C
        # Green: recovered by C over A (e.g. preserved elongated branches)
        # Red: in A but not in C (usually 0 since C is superset of A)
        diff_ca = original.copy()
        common_ca = (mask_c > 0) & (mask_a > 0)
        c_only = (mask_c > 0) & (mask_a == 0)
        a_only = (mask_a > 0) & (mask_c == 0)
        diff_ca[common_ca] = (0, 255, 255)
        diff_ca[c_only] = (0, 255, 0)
        diff_ca[a_only] = (0, 0, 255)

        # Ground truth evaluation for matching pairs
        gt_metrics = None
        if gt_path is not None and gt_path.is_file():
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None and gt_mask.shape == mask_a.shape:
                gt_metrics = {
                    "baseline": segmentation_metrics(baseline_mask, gt_mask),
                    "strategy_A": segmentation_metrics(mask_a, gt_mask),
                    "strategy_B": segmentation_metrics(mask_b, gt_mask),
                    "strategy_C": segmentation_metrics(mask_c, gt_mask),
                }

        # Save images
        cv2.imwrite(str(dir_strat_a / f"{stem}_mask.png"), mask_a)
        cv2.imwrite(str(dir_strat_a / f"{stem}_overlay.jpg"), overlay_a)

        cv2.imwrite(str(dir_strat_b / f"{stem}_mask.png"), mask_b)
        cv2.imwrite(str(dir_strat_b / f"{stem}_overlay.jpg"), overlay_b)

        cv2.imwrite(str(dir_strat_c / f"{stem}_mask.png"), mask_c)
        cv2.imwrite(str(dir_strat_c / f"{stem}_overlay.jpg"), overlay_c)
        cv2.imwrite(str(dir_strat_c / f"{stem}_diff_vs_strategy_A.jpg"), diff_ca)

        row = {
            "image_filename": img_path.name,
            "has_ground_truth": gt_path is not None,
            "baseline": base_metrics,
            "strategy_A": metrics_a,
            "strategy_B": metrics_b,
            "strategy_C": metrics_c,
            "c_vs_a_delta": {
                "pixels": metrics_c["crack_pixels"] - metrics_a["crack_pixels"],
                "components": metrics_c["connected_components"] - metrics_a["connected_components"],
                "length_no_gap_px": round(metrics_c["total_length_no_gap_px"] - metrics_a["total_length_no_gap_px"], 2),
                "length_with_gap_px": round(metrics_c["total_length_with_gap_px"] - metrics_a["total_length_with_gap_px"], 2),
            },
            "ground_truth_metrics": gt_metrics,
        }
        results_all.append(row)

    # Save summary JSON
    summary_json = OUTPUT_DIR / "comparative_summary.json"
    with open(summary_json, "w") as f:
        json.dump(
            {
                "experiment": "Hysteresis Post-Filtering Comparison (A vs B vs C)",
                "images_evaluated": len(results_all),
                "parameters": {
                    "hysteresis_low": LOW_THRESH,
                    "hysteresis_high": HIGH_THRESH,
                    "strategy_A": {"min_area": 200, "shape_filter": False},
                    "strategy_B": {"min_area": 80, "shape_filter": False},
                    "strategy_C": {
                        "min_area": 80,
                        "large_area_thresh": 200,
                        "interior_aspect_thresh": 2.2,
                        "boundary_aspect_thresh": 1.6,
                        "min_skeleton_length": 20,
                        "shape_filter": True,
                    },
                },
                "results": results_all,
            },
            f,
            indent=2,
            default=str,
        )

    # Save summary CSV
    summary_csv = OUTPUT_DIR / "comparative_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Has GT",
            "Strat A Pixels", "Strat B Pixels", "Strat C Pixels",
            "Strat A Comps", "Strat B Comps", "Strat C Comps",
            "Strat A Len (NoGap)", "Strat B Len (NoGap)", "Strat C Len (NoGap)",
            "Strat A Len (Gap)", "Strat B Len (Gap)", "Strat C Len (Gap)",
            "Strat C vs A Pixel Delta", "Strat C vs A Len Delta (NoGap)",
            "GT Dice A", "GT Dice B", "GT Dice C",
            "GT IoU A", "GT IoU B", "GT IoU C",
        ])
        for r in results_all:
            a = r["strategy_A"]
            b = r["strategy_B"]
            c = r["strategy_C"]
            diff = r["c_vs_a_delta"]
            gt = r["ground_truth_metrics"]
            gt_dice_a = f"{gt['strategy_A']['dice']:.4f}" if gt else "N/A"
            gt_dice_b = f"{gt['strategy_B']['dice']:.4f}" if gt else "N/A"
            gt_dice_c = f"{gt['strategy_C']['dice']:.4f}" if gt else "N/A"
            gt_iou_a = f"{gt['strategy_A']['iou']:.4f}" if gt else "N/A"
            gt_iou_b = f"{gt['strategy_B']['iou']:.4f}" if gt else "N/A"
            gt_iou_c = f"{gt['strategy_C']['iou']:.4f}" if gt else "N/A"

            writer.writerow([
                r["image_filename"],
                "Yes" if r["has_ground_truth"] else "No",
                a["crack_pixels"], b["crack_pixels"], c["crack_pixels"],
                a["connected_components"], b["connected_components"], c["connected_components"],
                a["total_length_no_gap_px"], b["total_length_no_gap_px"], c["total_length_no_gap_px"],
                a["total_length_with_gap_px"], b["total_length_with_gap_px"], c["total_length_with_gap_px"],
                diff["pixels"], diff["length_no_gap_px"],
                gt_dice_a, gt_dice_b, gt_dice_c,
                gt_iou_a, gt_iou_b, gt_iou_c,
            ])

    print("\n" + "=" * 75)
    print(f"  EXPERIMENT COMPLETE! Output saved to: {OUTPUT_DIR}")
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary CSV : {summary_csv}")
    print("=" * 75)


def main() -> None:
    run_experiment()


if __name__ == "__main__":
    main()
