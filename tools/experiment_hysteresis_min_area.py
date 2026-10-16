#!/usr/bin/env python3
"""Controlled follow-up experiment for hysteresis_recovery min_area filtering.

Compares min_area=80 (control), min_area=150, and min_area=200 on
data/sample_images/concrete-temperature-crack.jpg.

Evaluates:
  - Retained crack pixels and coverage
  - Connected component count
  - Status of suspected satellite artifacts (upper-right and lower-left)
  - Maximum and mean crack widths
  - Total crack length with and without gap closing
  - Preservation of main crack branches

Usage (from repository root):
    python -m tools.experiment_hysteresis_min_area
"""

from __future__ import annotations

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

TARGET_IMAGE = os.path.join(
    _project_root, "data", "sample_images", "concrete-temperature-crack.jpg"
)
BASE_OUTPUT_DIR = os.path.join(
    _project_root, "data", "outputs", "experiment_hysteresis_min_area"
)

LOW_THRESH = 25
HIGH_THRESH = 48
MIN_AREA_CANDIDATES = [80, 150, 200]


def _compute_metrics(mask: np.ndarray, label: str) -> dict:
    """Compute comprehensive metrics for a binary mask."""
    crack_pixels = int(np.sum(mask > 0))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    coverage_pct = crack_pixels / total_pixels * 100.0

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


def _check_suspected_artifacts(mask: np.ndarray) -> dict:
    """Check whether known suspected satellite artifacts are present in the mask.

    - Upper-right component: around x in [292, 346], y in [57, 75]
    - Lower-left component: around x in [62, 74], y in [294, 328]
    """
    # Crop regions
    ur_crop = mask[57:76, 292:347]
    ll_crop = mask[294:329, 62:75]

    ur_pixels = int(np.sum(ur_crop > 0))
    ll_pixels = int(np.sum(ll_crop > 0))

    # Also check if any connected component centroid falls inside the artifact bounding boxes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    ur_comp_found = False
    ur_comp_area = 0
    ll_comp_found = False
    ll_comp_area = 0

    for i in range(1, num_labels):
        cx, cy = centroids[i]
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Upper-right centroid ~ (316, 70)
        if 280 <= cx <= 360 and 50 <= cy <= 85:
            ur_comp_found = True
            ur_comp_area = area
        # Lower-left centroid ~ (67, 309)
        if 50 <= cx <= 85 and 285 <= cy <= 335:
            ll_comp_found = True
            ll_comp_area = area

    return {
        "upper_right_artifact": {
            "description": "Upper-right texture artifact at bbox ~ [292-346, 57-75]",
            "pixels_in_bbox": int(ur_pixels),
            "component_retained": bool(ur_comp_found),
            "component_area": int(ur_comp_area),
        },
        "lower_left_artifact": {
            "description": "Lower-left pit/void artifact at bbox ~ [62-74, 294-328]",
            "pixels_in_bbox": int(ll_pixels),
            "component_retained": bool(ll_comp_found),
            "component_area": int(ll_comp_area),
        },
    }


def _check_main_branches(mask: np.ndarray) -> dict:
    """Check preservation of key structural crack branches."""
    # Main trunk spans full height (y in 0..370, x in 0..205)
    # Bottom horizontal branch (x in 202..370, y in 267..318)
    # Mid-left horizontal branch (x in 1..65, y in 152..180)
    # Mid-right parallel fissure (x in 170..196, y in 156..200)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    main_trunk_pixels = 0
    bottom_branch_pixels = 0
    mid_left_branch_pixels = 0
    parallel_fissure_pixels = 0

    for i in range(1, num_labels):
        x, y, w, h, a = stats[i]
        cx, cy = centroids[i]
        if a >= 3000:  # Main spanning trunk
            main_trunk_pixels = int(a)
        elif 200 <= cx <= 350 and 260 <= cy <= 320:
            bottom_branch_pixels = int(a)
        elif cx <= 65 and 140 <= cy <= 185:
            mid_left_branch_pixels = int(a)
        elif 160 <= cx <= 200 and 150 <= cy <= 200:
            parallel_fissure_pixels = int(a)

    return {
        "main_vertical_trunk": {
            "retained": bool(main_trunk_pixels > 0),
            "pixels": int(main_trunk_pixels),
        },
        "bottom_right_branch": {
            "retained": bool(bottom_branch_pixels > 0),
            "pixels": int(bottom_branch_pixels),
        },
        "mid_left_branch": {
            "retained": bool(mid_left_branch_pixels > 0),
            "pixels": int(mid_left_branch_pixels),
        },
        "mid_right_parallel_fissure": {
            "retained": bool(parallel_fissure_pixels > 0),
            "pixels": int(parallel_fissure_pixels),
        },
    }


def run_experiments() -> dict:
    """Execute the controlled follow-up experiment."""
    print("=" * 75)
    print("  CONTROLLED FOLLOW-UP EXPERIMENT: Hysteresis min_area Tuning")
    print(f"  Target Image: {TARGET_IMAGE}")
    print(f"  Conditions  : min_area in {MIN_AREA_CANDIDATES}")
    print(f"  Thresholds  : low={LOW_THRESH}, high={HIGH_THRESH}")
    print("=" * 75)

    if not os.path.isfile(TARGET_IMAGE):
        raise FileNotFoundError(f"Target image not found: {TARGET_IMAGE}")

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 1. Preprocessing (shared across all conditions)
    print("\n[Step 1] Running preprocessing...")
    preproc = run_preprocessing_pipeline(TARGET_IMAGE, save_steps=False)
    enhanced = preproc["enhanced"]
    adaptive_mask = preproc.get("crack_mask")

    # 2. Frangi ridge enhancement (shared)
    print("[Step 2] Computing Frangi ridge filter...")
    _, ridge_norm = enhance_ridges_frangi(
        enhanced, sigmas=(1, 2, 3, 4), black_ridges=True
    )

    # 3. Baseline segmentation (fixed reference)
    print("[Step 3] Computing baseline (Otsu) reference...")
    ridge_binary_otsu = binarize_ridge(ridge_norm, method="otsu")
    cleaned_otsu, _ = filter_crack_components(ridge_binary_otsu, min_area=80)
    baseline_mask = select_segmentation_profile(
        cleaned_otsu, adaptive_mask, profile="baseline"
    )
    baseline_overlay = create_crack_overlay(
        preproc["original"], baseline_mask, color=(0, 0, 255), alpha=0.65
    )
    baseline_metrics = _compute_metrics(baseline_mask, "baseline_otsu_min_area_80")

    results_by_area = {}
    control_80_metrics = None

    for min_area in MIN_AREA_CANDIDATES:
        cond_dir = os.path.join(BASE_OUTPUT_DIR, f"min_area_{min_area}")
        os.makedirs(cond_dir, exist_ok=True)
        print(f"\n[Step 4] Running condition: min_area={min_area} -> {cond_dir}")

        # Hysteresis mask with specific min_area
        hyst_mask = select_segmentation_profile(
            cleaned_otsu,
            adaptive_mask,
            profile="hysteresis_recovery",
            ridge_norm=ridge_norm,
            hysteresis_low=LOW_THRESH,
            hysteresis_high=HIGH_THRESH,
            hysteresis_min_area=min_area,
        )
        hyst_overlay = create_crack_overlay(
            preproc["original"], hyst_mask, color=(0, 200, 0), alpha=0.65
        )
        hyst_metrics = _compute_metrics(
            hyst_mask, f"hysteresis_recovery_min_area_{min_area}"
        )

        if min_area == 80:
            control_80_metrics = hyst_metrics

        # Difference visualization against baseline
        baseline_bool = baseline_mask > 0
        hyst_bool = hyst_mask > 0
        only_baseline = baseline_bool & ~hyst_bool
        only_hyst = hyst_bool & ~baseline_bool
        common = baseline_bool & hyst_bool

        diff_vis = preproc["original"].copy()
        diff_vis[common] = (0, 255, 255)       # Yellow: detected by both
        diff_vis[only_baseline] = (0, 0, 255)  # Red: baseline only
        diff_vis[only_hyst] = (0, 255, 0)      # Green: hysteresis only

        diff_stats = {
            "common_pixels": int(np.sum(common)),
            "baseline_only_pixels": int(np.sum(only_baseline)),
            "hysteresis_only_pixels": int(np.sum(only_hyst)),
        }

        # Artifacts and branch checks
        artifacts = _check_suspected_artifacts(hyst_mask)
        branches = _check_main_branches(hyst_mask)

        # Retention vs min_area=80 control
        ref_pix = control_80_metrics["crack_pixels"] if control_80_metrics else hyst_metrics["crack_pixels"]
        ref_l_nogap = control_80_metrics["total_length_no_gap_px"] if control_80_metrics else hyst_metrics["total_length_no_gap_px"]
        ref_l_gap = control_80_metrics["total_length_with_gap_px"] if control_80_metrics else hyst_metrics["total_length_with_gap_px"]

        retention = {
            "pixel_retention_vs_80_pct": round(hyst_metrics["crack_pixels"] / ref_pix * 100.0, 2),
            "length_no_gap_retention_vs_80_pct": round(hyst_metrics["total_length_no_gap_px"] / ref_l_nogap * 100.0, 2),
            "length_with_gap_retention_vs_80_pct": round(hyst_metrics["total_length_with_gap_px"] / ref_l_gap * 100.0, 2),
        }

        # Save condition artifacts
        cv2.imwrite(os.path.join(cond_dir, "baseline_mask.png"), baseline_mask)
        cv2.imwrite(os.path.join(cond_dir, "baseline_overlay.jpg"), baseline_overlay)
        cv2.imwrite(os.path.join(cond_dir, "hysteresis_mask.png"), hyst_mask)
        cv2.imwrite(os.path.join(cond_dir, "hysteresis_overlay.jpg"), hyst_overlay)
        cv2.imwrite(os.path.join(cond_dir, "diff_visualization.jpg"), diff_vis)

        cond_report = {
            "min_area": min_area,
            "hysteresis_thresholds": {"low": LOW_THRESH, "high": HIGH_THRESH},
            "baseline": baseline_metrics,
            "hysteresis": hyst_metrics,
            "difference_vs_baseline": diff_stats,
            "retention_vs_min_area_80": retention,
            "suspected_artifacts": artifacts,
            "main_branches": branches,
        }

        report_file = os.path.join(cond_dir, "experiment_report.json")
        with open(report_file, "w") as f:
            json.dump(cond_report, f, indent=2, default=str)

        results_by_area[min_area] = cond_report

    # Save comparative summary
    summary_file = os.path.join(BASE_OUTPUT_DIR, "comparative_summary.json")
    summary = {
        "image": os.path.basename(TARGET_IMAGE),
        "conditions_tested": MIN_AREA_CANDIDATES,
        "baseline_reference": baseline_metrics,
        "hysteresis_conditions": {
            f"min_area_{k}": v for k, v in results_by_area.items()
        },
    }
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 75)
    print("  EXPERIMENT COMPLETE! Output directories:")
    for min_area in MIN_AREA_CANDIDATES:
        print(f"   - data/outputs/experiment_hysteresis_min_area/min_area_{min_area}")
    print(f"   - Comparative summary: {summary_file}")
    print("=" * 75)

    return summary


def main() -> None:
    run_experiments()


if __name__ == "__main__":
    main()
