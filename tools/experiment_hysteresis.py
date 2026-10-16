#!/usr/bin/env python3
"""Hysteresis segmentation experiment: baseline vs. hysteresis_recovery.

Runs the full preprocessing + segmentation pipeline on the target image
using both the ``baseline`` (Otsu) profile and the new ``hysteresis_recovery``
profile, then compares crack pixels, coverage, connected components,
maximum/mean width, total length, and saves all artifacts to a dedicated
output directory.

Usage (from project root):
    python -m tools.experiment_hysteresis
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
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
    binarize_ridge_hysteresis,
    create_crack_overlay,
    enhance_ridges_frangi,
    filter_crack_components,
    select_segmentation_profile,
)
from src.measurement import (
    compute_skeleton,
    measure_crack_lengths,
    measure_crack_widths,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

TARGET_IMAGE = os.path.join(
    _project_root, "data", "sample_images", "concrete-temperature-crack.jpg"
)
OUTPUT_DIR = os.path.join(
    _project_root, "data", "outputs", "experiment_hysteresis"
)

# Default hysteresis thresholds – adjusted below if the Frangi response
# distribution shows they are unsuitable.
DEFAULT_LOW = 25
DEFAULT_HIGH = 48


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mask_metrics(mask: np.ndarray, label: str) -> dict:
    """Compute summary metrics from a binary mask."""
    crack_pixels = int(np.sum(mask > 0))
    total_pixels = int(mask.shape[0] * mask.shape[1])
    coverage_pct = crack_pixels / total_pixels * 100.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    components = max(0, num_labels - 1)

    width_info = measure_crack_widths(mask)
    length_info = measure_crack_lengths(mask, enable_gap_closing=False)

    return {
        "label": label,
        "crack_pixels": crack_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 4),
        "connected_components": components,
        "max_width_px": round(width_info["max_width_px"], 2),
        "mean_width_px": round(width_info["mean_width_px"], 2),
        "total_length_px": round(length_info["total_length_px"], 2),
        "length_components": length_info["component_count"],
    }


def _frangi_distribution(ridge_norm: np.ndarray) -> dict:
    """Summarise the ridge_norm (0–255) distribution."""
    nonzero = ridge_norm[ridge_norm > 0]
    return {
        "min": int(ridge_norm.min()),
        "max": int(ridge_norm.max()),
        "mean": round(float(ridge_norm.mean()), 2),
        "std": round(float(ridge_norm.std()), 2),
        "nonzero_count": int(nonzero.size),
        "nonzero_mean": round(float(nonzero.mean()), 2) if nonzero.size > 0 else 0,
        "nonzero_std": round(float(nonzero.std()), 2) if nonzero.size > 0 else 0,
        "p10": int(np.percentile(ridge_norm, 10)),
        "p25": int(np.percentile(ridge_norm, 25)),
        "p50_median": int(np.percentile(ridge_norm, 50)),
        "p75": int(np.percentile(ridge_norm, 75)),
        "p90": int(np.percentile(ridge_norm, 90)),
        "p95": int(np.percentile(ridge_norm, 95)),
        "p99": int(np.percentile(ridge_norm, 99)),
    }


def _derive_thresholds(dist: dict) -> tuple[int, int]:
    """Derive sensible hysteresis thresholds from the Frangi distribution.

    Strategy:
    - high_thresh ≈ top of the nonzero distribution – strong ridge certainty.
      We target ~ nonzero_mean + 1.0 * nonzero_std, clamped to [30, 120].
    - low_thresh ≈ lower edge – weak ridge boundary.
      We target ~ nonzero_mean - 0.5 * nonzero_std, clamped to [10, high - 5].

    If DEFAULT_LOW / DEFAULT_HIGH fall within a reasonable band of the derived
    values (±40% relative) we keep the defaults to match the diagnosis.
    """
    nz_mean = dist["nonzero_mean"]
    nz_std = dist["nonzero_std"]

    derived_high = int(np.clip(nz_mean + 1.0 * nz_std, 30, 120))
    derived_low = int(np.clip(nz_mean - 0.5 * nz_std, 10, derived_high - 5))

    # Check if defaults are compatible
    defaults_ok = (
        DEFAULT_LOW >= max(5, derived_low - 15)
        and DEFAULT_LOW <= derived_low + 20
        and DEFAULT_HIGH >= max(20, derived_high - 25)
        and DEFAULT_HIGH <= derived_high + 30
    )
    if defaults_ok:
        return DEFAULT_LOW, DEFAULT_HIGH
    else:
        return derived_low, derived_high


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("  EXPERIMENT: Hysteresis Segmentation vs Baseline (Otsu)")
    print("=" * 70)

    if not os.path.isfile(TARGET_IMAGE):
        print(f"[ERROR] Target image not found: {TARGET_IMAGE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Preprocessing ─────────────────────────────────────────────────
    print("\n── Step 1: Preprocessing ──")
    preproc = run_preprocessing_pipeline(
        TARGET_IMAGE, output_dir=OUTPUT_DIR, save_steps=True
    )
    enhanced = preproc["enhanced"]
    adaptive_mask = preproc.get("crack_mask")

    # ── 2. Frangi ridge enhancement ──────────────────────────────────────
    print("\n── Step 2: Frangi ridge enhancement ──")
    ridge_float, ridge_norm = enhance_ridges_frangi(
        enhanced, sigmas=(1, 2, 3, 4), black_ridges=True
    )

    # ── 3. Probe Frangi response distribution ────────────────────────────
    print("\n── Step 3: Frangi response distribution ──")
    dist = _frangi_distribution(ridge_norm)
    for k, v in dist.items():
        print(f"    {k:20s}: {v}")

    low, high = _derive_thresholds(dist)
    if (low, high) == (DEFAULT_LOW, DEFAULT_HIGH):
        print(f"\n    ✓ Default thresholds ({DEFAULT_LOW}/{DEFAULT_HIGH}) are "
              f"compatible with distribution. Using defaults.")
    else:
        print(f"\n    ⚠ Default thresholds ({DEFAULT_LOW}/{DEFAULT_HIGH}) fall "
              f"outside the compatible band.")
        print(f"    → Using derived thresholds: low={low}, high={high}")

    # Save histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(ridge_norm.ravel(), bins=256, range=(0, 256), color="#2196F3",
                alpha=0.7, log=True)
        ax.axvline(low, color="orange", linestyle="--", label=f"low={low}")
        ax.axvline(high, color="red", linestyle="--", label=f"high={high}")
        ax.set_title("Normalized Frangi Response Histogram (log scale)")
        ax.set_xlabel("ridge_norm value (0–255)")
        ax.set_ylabel("Pixel count (log)")
        ax.legend()
        fig.tight_layout()
        hist_path = os.path.join(OUTPUT_DIR, "frangi_histogram.png")
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        print(f"    Histogram saved: {hist_path}")
    except ImportError:
        print("    (matplotlib not available – skipping histogram)")

    # ── 4. Baseline segmentation (Otsu) ──────────────────────────────────
    print("\n── Step 4: Baseline segmentation (Otsu) ──")
    ridge_binary_otsu = binarize_ridge(ridge_norm, method="otsu")
    cleaned_otsu, _ = filter_crack_components(ridge_binary_otsu, min_area=80)
    baseline_mask = select_segmentation_profile(
        cleaned_otsu, adaptive_mask, profile="baseline"
    )
    baseline_overlay = create_crack_overlay(
        preproc["original"], baseline_mask, color=(0, 0, 255), alpha=0.65
    )
    baseline_metrics = _mask_metrics(baseline_mask, "baseline")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "baseline_mask.png"), baseline_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "baseline_overlay.jpg"), baseline_overlay)

    # ── 5. Hysteresis segmentation ───────────────────────────────────────
    print("\n── Step 5: Hysteresis segmentation ──")
    hyst_mask = select_segmentation_profile(
        cleaned_otsu,          # ridge_mask (unused directly by hysteresis, but validated)
        adaptive_mask,
        profile="hysteresis_recovery",
        ridge_norm=ridge_norm,
        hysteresis_low=low,
        hysteresis_high=high,
    )
    hyst_overlay = create_crack_overlay(
        preproc["original"], hyst_mask, color=(0, 200, 0), alpha=0.65
    )
    hyst_metrics = _mask_metrics(hyst_mask, f"hysteresis (low={low}, high={high})")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "hysteresis_mask.png"), hyst_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "hysteresis_overlay.jpg"), hyst_overlay)

    # ── 6. Difference visualisation ──────────────────────────────────────
    print("\n── Step 6: Difference analysis ──")
    baseline_bool = baseline_mask > 0
    hyst_bool = hyst_mask > 0
    only_baseline = baseline_bool & ~hyst_bool
    only_hyst = hyst_bool & ~baseline_bool
    common = baseline_bool & hyst_bool

    diff_vis = preproc["original"].copy()
    diff_vis[common] = (0, 255, 255)       # Yellow – detected by both
    diff_vis[only_baseline] = (0, 0, 255)  # Red – baseline only
    diff_vis[only_hyst] = (0, 255, 0)      # Green – hysteresis only
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diff_visualization.jpg"), diff_vis)

    diff_stats = {
        "common_pixels": int(np.sum(common)),
        "baseline_only_pixels": int(np.sum(only_baseline)),
        "hysteresis_only_pixels": int(np.sum(only_hyst)),
    }

    # ── 7. Report ────────────────────────────────────────────────────────
    report = {
        "image": os.path.basename(TARGET_IMAGE),
        "frangi_distribution": dist,
        "thresholds_used": {"low": low, "high": high},
        "thresholds_are_defaults": (low, high) == (DEFAULT_LOW, DEFAULT_HIGH),
        "baseline": baseline_metrics,
        "hysteresis": hyst_metrics,
        "difference": diff_stats,
    }

    report_path = os.path.join(OUTPUT_DIR, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  EXPERIMENT RESULTS")
    print("=" * 70)
    fmt = "  {:<30s}  {:>15s}  {:>15s}"
    print(fmt.format("Metric", "Baseline", "Hysteresis"))
    print("  " + "-" * 64)
    for key in [
        "crack_pixels",
        "coverage_pct",
        "connected_components",
        "max_width_px",
        "mean_width_px",
        "total_length_px",
    ]:
        bv = baseline_metrics[key]
        hv = hyst_metrics[key]
        print(fmt.format(key, str(bv), str(hv)))

    print()
    print(f"  Common pixels       : {diff_stats['common_pixels']:,}")
    print(f"  Baseline-only pixels: {diff_stats['baseline_only_pixels']:,}")
    print(f"  Hysteresis-only px  : {diff_stats['hysteresis_only_pixels']:,}")
    print()
    print(f"  Artifacts saved to  : {OUTPUT_DIR}")
    print(f"  Full report         : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
