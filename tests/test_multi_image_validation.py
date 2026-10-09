import csv

import cv2
import numpy as np

from tools.evaluate_validation_dataset import (
    discover_image_mask_pairs,
    evaluate_dataset,
)


def test_discover_pairs_uses_stem_mask_convention(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    (image_dir / "00001.jpg").write_bytes(b"image")
    (image_dir / "00002.jpg").write_bytes(b"image")
    (mask_dir / "00001_mask.png").write_bytes(b"mask")
    (mask_dir / "other_mask.png").write_bytes(b"mask")

    pairs, unmatched = discover_image_mask_pairs(image_dir, mask_dir)

    assert [(image.name, mask.name if mask else None) for image, mask in pairs] == [
        ("00001.jpg", "00001_mask.png"),
        ("00002.jpg", None),
    ]
    assert [path.name for path in unmatched] == ["other_mask.png"]


def test_evaluate_dataset_writes_metrics_and_skips_missing_masks(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    ground_truth = np.zeros((3, 3), dtype=np.uint8)
    ground_truth[0, 0] = 255
    cv2.imwrite(str(image_dir / "00001.jpg"), image)
    cv2.imwrite(str(mask_dir / "00001_mask.png"), ground_truth)
    (image_dir / "00002.jpg").write_text("not an image", encoding="utf-8")

    def fake_preprocess(image_path, output_dir=None, save_steps=False):
        return {"image_path": image_path}

    def fake_segment(preprocessed, output_dir=None, segmentation_profile="baseline", save_steps=False):
        predicted = np.zeros((3, 3), dtype=np.uint8)
        predicted[0, 0] = 255
        return {"final_mask": predicted}

    output_csv = tmp_path / "results.csv"
    report = evaluate_dataset(
        image_dir,
        mask_dir,
        output_csv,
        profiles=("baseline", "adaptive_recovery", "guided_recovery"),
        preprocess=fake_preprocess,
        segment=fake_segment,
    )

    assert report["valid_pair_count"] == 1
    assert report["skipped_count"] == 1
    with output_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 4
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    assert len(evaluated) == 3
    assert all(row["iou"] == "1.0" for row in evaluated)
    assert any(row["error_message"] == "Matching ground-truth mask not found" for row in rows)


def test_evaluate_dataset_skips_dimension_mismatch(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    cv2.imwrite(str(image_dir / "00001.jpg"), np.zeros((3, 3, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_dir / "00001_mask.png"), np.zeros((2, 3), dtype=np.uint8))

    report = evaluate_dataset(
        image_dir,
        mask_dir,
        tmp_path / "results.csv",
        profiles=("baseline",),
        preprocess=lambda *args, **kwargs: {},
        segment=lambda *args, **kwargs: {},
    )

    assert report["valid_pair_count"] == 0
    assert report["skipped_count"] == 1
    assert "dimensions do not match" in report["rows"][0]["error_message"]
