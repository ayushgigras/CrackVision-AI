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


def evaluate_dataset(
    image_dir: Path,
    ground_truth_dir: Path,
    output_csv: Path,
    profiles: tuple[str, ...] = SEGMENTATION_PROFILES,
    preprocess: Callable = run_preprocessing_pipeline,
    segment: Callable = run_segmentation_pipeline,
) -> dict[str, object]:
    """Evaluate every valid image-mask pair and write per-image metrics."""
    invalid_profiles = set(profiles) - set(SEGMENTATION_PROFILES)
    if invalid_profiles:
        raise ValueError(
            f"Unknown segmentation profile(s): {', '.join(sorted(invalid_profiles))}"
        )

    pairs, unmatched_masks = discover_image_mask_pairs(image_dir, ground_truth_dir)
    rows: list[dict[str, object]] = []
    masks_by_profile: dict[str, list[np.ndarray]] = {profile: [] for profile in profiles}
    ground_truth_by_profile: dict[str, list[np.ndarray]] = {
        profile: [] for profile in profiles
    }
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
        if image.shape[:2] != ground_truth.shape:
            rows.append(
                _skipped_row(
                    image_path,
                    mask_path,
                    "Image and ground-truth mask dimensions do not match "
                    f"({image.shape[:2]} vs {ground_truth.shape})",
                )
            )
            continue

        try:
            preprocessed = preprocess(str(image_path), output_dir=None, save_steps=False)
            profile_results = {}
            for profile in profiles:
                result = segment(
                    preprocessed,
                    output_dir=None,
                    segmentation_profile=profile,
                    save_steps=False,
                )
                predicted = result["final_mask"]
                metrics = segmentation_metrics(predicted, ground_truth)
                rows.append(_metric_row(image_path, mask_path, profile, metrics))
                profile_results[profile] = predicted
            valid_pair_count += 1
            for profile in profiles:
                masks_by_profile[profile].append(profile_results[profile])
                ground_truth_by_profile[profile].append(ground_truth)
        except Exception as error:
            rows.append(_skipped_row(image_path, mask_path, f"Evaluation failed: {error}"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in CSV_FIELDS} for row in rows
        )

    aggregate_metrics: dict[str, dict[str, float | int]] = {}
    for profile in profiles:
        if not masks_by_profile[profile]:
            continue
        predicted = np.concatenate(
            [mask.reshape(-1) for mask in masks_by_profile[profile]]
        ).reshape(1, -1)
        ground_truth = np.concatenate(
            [mask.reshape(-1) for mask in ground_truth_by_profile[profile]]
        ).reshape(1, -1)
        aggregate_metrics[profile] = segmentation_metrics(predicted, ground_truth)

    skipped_rows = [row for row in rows if row["status"] == "skipped"]
    return {
        "rows": rows,
        "aggregate_metrics": aggregate_metrics,
        "valid_pair_count": valid_pair_count,
        "skipped_count": len(skipped_rows),
        "unmatched_masks": unmatched_masks,
        "output_csv": output_csv,
    }


def _print_report(report: dict[str, object]) -> None:
    """Print evaluated-pair counts, skipped candidates, and aggregate metrics."""
    print(f"Valid image-mask pairs evaluated: {report['valid_pair_count']}")
    print(f"Skipped image candidates: {report['skipped_count']}")
    unmatched_masks = report["unmatched_masks"]
    if unmatched_masks:
        print("Unmatched ground-truth masks:")
        for path in unmatched_masks:
            print(f"  - {path.name}")
    print(f"Per-image CSV saved to: {report['output_csv']}")

    aggregate_metrics = report["aggregate_metrics"]
    for profile, metrics in aggregate_metrics.items():
        print(profile)
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "dice",
            "f1",
            "iou",
        ):
            print(f"  {key}: {metrics[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CrackGauge segmentation profiles against manual masks"
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
    args = parser.parse_args()
    report = evaluate_dataset(args.image_dir, args.ground_truth_dir, args.output_csv)
    _print_report(report)


if __name__ == "__main__":
    main()
