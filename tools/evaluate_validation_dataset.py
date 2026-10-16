"""Evaluate segmentation profiles across manually annotated image-mask pairs.

Filename convention:
    data/real_images/00001.jpg
    data/ground_truth/00001_mask.png

Run from the repository root:
    python tools/evaluate_validation_dataset.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import run_preprocessing_pipeline
from src.segmentation import SEGMENTATION_PROFILES, run_segmentation_pipeline
from src.validation import segmentation_metrics
from tools.annotate_mask import validate_mask_contract


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = [
    "image_filename",
    "ground_truth_filename",
    "profile",
    "status",
    "error_message",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "dice",
    "f1",
    "iou",
]

HYSTERESIS_STRATEGIES = (
    "baseline",
    "strategy_A_min_area_200",
    "strategy_B_min_area_80",
    "strategy_C_shape_filter",
)
VALID_EVALUATION_MODES = ("profiles", "hysteresis_strategies")


def discover_image_mask_pairs(
    image_dir: Path,
    ground_truth_dir: Path,
) -> tuple[list[tuple[Path, Path | None]], list[Path]]:
    """Discover images and exact ``<stem>_mask`` annotation matches."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {ground_truth_dir}"
        )

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    masks_by_stem = {
        path.stem.removesuffix("_mask"): path
        for path in ground_truth_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.stem.endswith("_mask")
    }
    pairs = [(image, masks_by_stem.get(image.stem)) for image in images]
    matched_stems = {image.stem for image, mask in pairs if mask is not None}
    unmatched_masks = sorted(
        path for stem, path in masks_by_stem.items() if stem not in matched_stems
    )
    return pairs, unmatched_masks


def _skipped_row(
    image_path: Path,
    mask_path: Path | None,
    error_message: str,
) -> dict[str, object]:
    return {
        "image_filename": image_path.name,
        "ground_truth_filename": mask_path.name if mask_path else "",
        "profile": "",
        "status": "skipped",
        "error_message": error_message,
    }


def _metric_row(
    image_path: Path,
    mask_path: Path,
    profile: str,
    metrics: dict[str, float | int],
) -> dict[str, object]:
    return {
        "image_filename": image_path.name,
        "ground_truth_filename": mask_path.name,
        "profile": profile,
        "status": "evaluated",
        "error_message": "",
        **metrics,
    }


def dispatch_segmentation(
    preprocessed: dict,
    target: str,
    segment_fn: Callable = run_segmentation_pipeline,
) -> np.ndarray:
    """Execute segmentation for either a standard profile or an isolated hysteresis strategy."""
    if target == "strategy_A_min_area_200":
        try:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                hysteresis_min_area=200,
                hysteresis_use_shape_filter=False,
                save_steps=False,
            )
        except TypeError:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                save_steps=False,
            )
        return result["final_mask"]
    elif target == "strategy_B_min_area_80":
        try:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                hysteresis_min_area=80,
                hysteresis_use_shape_filter=False,
                save_steps=False,
            )
        except TypeError:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                save_steps=False,
            )
        return result["final_mask"]
    elif target == "strategy_C_shape_filter":
        try:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                hysteresis_min_area=80,
                hysteresis_use_shape_filter=True,
                save_steps=False,
            )
        except TypeError:
            result = segment_fn(
                preprocessed,
                output_dir=None,
                segmentation_profile="hysteresis_recovery",
                save_steps=False,
            )
        return result["final_mask"]
    else:
        result = segment_fn(
            preprocessed,
            output_dir=None,
            segmentation_profile=target,
            save_steps=False,
        )
        return result["final_mask"]


def compute_aggregate_metrics_from_rows(
    rows: list[dict[str, object]],
    target: str,
) -> dict[str, float | int]:
    """Compute aggregate metrics by summing integer TP, FP, FN, and TN across valid pairs."""
    evaluated = [
        r for r in rows
        if r.get("profile") == target and r.get("status") == "evaluated"
    ]
    if not evaluated:
        return {}

    tp = sum(int(r["true_positive"]) for r in evaluated)
    fp = sum(int(r["false_positive"]) for r in evaluated)
    fn = sum(int(r["false_negative"]) for r in evaluated)
    tn = sum(int(r["true_negative"]) for r in evaluated)

    union = tp + fp + fn
    pred_pos = tp + fp
    gt_pos = tp + fn

    if union == 0:
        iou = dice = precision = recall = f1 = 1.0
    else:
        iou = tp / union
        dice_denom = 2 * tp + fp + fn
        dice = (2 * tp / dice_denom) if dice_denom else 0.0
        precision = (tp / pred_pos) if pred_pos else 0.0
        recall = (tp / gt_pos) if gt_pos else 0.0
        f1_denom = precision + recall
        f1 = (2 * precision * recall / f1_denom) if f1_denom else 0.0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "f1": f1,
        "iou": iou,
    }


def evaluate_dataset(
    image_dir: Path,
    ground_truth_dir: Path,
    output_csv: Path,
    profiles: tuple[str, ...] | None = None,
    evaluation_mode: str = "profiles",
    preprocess: Callable = run_preprocessing_pipeline,
    segment: Callable = run_segmentation_pipeline,
    allow_empty_masks: bool = False,
) -> dict[str, object]:
    """Evaluate every valid image-mask pair and write per-image metrics."""
    if profiles is not None:
        targets = tuple(profiles)
    else:
        if evaluation_mode == "hysteresis_strategies":
            targets = HYSTERESIS_STRATEGIES
        elif evaluation_mode == "profiles":
            targets = SEGMENTATION_PROFILES
        else:
            raise ValueError(
                f"Unknown evaluation_mode: '{evaluation_mode}'. Valid modes: {VALID_EVALUATION_MODES}"
            )

    allowed_targets = set(SEGMENTATION_PROFILES) | set(HYSTERESIS_STRATEGIES)
    invalid_targets = set(targets) - allowed_targets
    if invalid_targets:
        raise ValueError(
            f"Unknown evaluation target(s): {', '.join(sorted(invalid_targets))}"
        )

    pairs, unmatched_masks = discover_image_mask_pairs(image_dir, ground_truth_dir)
    rows: list[dict[str, object]] = []
    valid_pair_count = 0

    for image_path, mask_path in pairs:
        if mask_path is None:
            rows.append(_skipped_row(image_path, None, "Matching ground-truth mask not found"))
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rows.append(_skipped_row(image_path, mask_path, "Image could not be read"))
            continue
        if ground_truth is None:
            rows.append(_skipped_row(image_path, mask_path, "Ground-truth mask could not be read"))
            continue

        # Validate mask against CrackGauge ground-truth contract
        is_valid, reason, _stats = validate_mask_contract(
            image, ground_truth, allow_empty=allow_empty_masks
        )
        if not is_valid:
            print(f"[WARN] Skipping candidate pair {image_path.name} / {mask_path.name}: {reason}")
            rows.append(_skipped_row(image_path, mask_path, reason))
            continue

        try:
            preprocessed = preprocess(str(image_path), output_dir=None, save_steps=False)
            for target in targets:
                predicted = dispatch_segmentation(preprocessed, target, segment_fn=segment)
                metrics = segmentation_metrics(predicted, ground_truth)
                rows.append(_metric_row(image_path, mask_path, target, metrics))
            valid_pair_count += 1
        except Exception as error:
            rows.append(_skipped_row(image_path, mask_path, f"Evaluation failed: {error}"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS} for row in rows
        )

    # Compute aggregate metrics by directly summing TP, FP, FN, and TN across evaluated pairs
    aggregate_metrics: dict[str, dict[str, float | int]] = {}
    for target in targets:
        agg = compute_aggregate_metrics_from_rows(rows, target)
        if agg:
            aggregate_metrics[target] = agg

    skipped_rows = [row for row in rows if row["status"] == "skipped"]
    return {
        "rows": rows,
        "aggregate_metrics": aggregate_metrics,
        "valid_pair_count": valid_pair_count,
        "skipped_count": len(skipped_rows),
        "unmatched_masks": unmatched_masks,
        "output_csv": output_csv,
        "evaluation_mode": evaluation_mode,
        "targets": targets,
    }


def _print_report(report: dict[str, object]) -> None:
    """Print evaluated-pair counts, skipped candidates, and aggregate metrics."""
    print("=" * 65)
    print(f"  CrackGauge Ground-Truth Evaluation ({report.get('evaluation_mode', 'profiles')} mode)")
    print("=" * 65)
    print(f"Valid image-mask pairs evaluated : {report['valid_pair_count']}")
    print(f"Skipped image candidates         : {report['skipped_count']}")
    unmatched_masks = report["unmatched_masks"]
    if unmatched_masks:
        print(f"Unmatched ground-truth masks     : {len(unmatched_masks)}")
        for path in unmatched_masks:
            print(f"  - {path.name}")
    print(f"Per-image CSV saved to           : {report['output_csv']}")
    print("-" * 65)
    print("Aggregate Confusion Matrix & Metrics (Summed Pixels):")

    aggregate_metrics = report["aggregate_metrics"]
    for target, metrics in aggregate_metrics.items():
        print(f"\n[{target}]")
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "precision",
            "recall",
            "dice",
            "f1",
            "iou",
        ):
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:<15}: {val:.4f} ({val*100:.2f}%)")
            else:
                print(f"  {key:<15}: {val:,} px")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CrackGauge segmentation profiles or hysteresis strategies against manual masks"
    )
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data" / "real_images")
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=ROOT / "data" / "ground_truth",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "outputs" / "multi_image_validation.csv",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_EVALUATION_MODES,
        default="profiles",
        help="Evaluation mode: 'profiles' (default) or 'hysteresis_strategies'",
    )
    parser.add_argument(
        "--allow-empty-masks",
        action="store_true",
        help="Allow masks with 0 foreground crack pixels instead of skipping them with a warning",
    )
    args = parser.parse_args()
    report = evaluate_dataset(
        args.image_dir,
        args.ground_truth_dir,
        args.output_csv,
        evaluation_mode=args.mode,
        allow_empty_masks=args.allow_empty_masks,
    )
    _print_report(report)


if __name__ == "__main__":
    main()
