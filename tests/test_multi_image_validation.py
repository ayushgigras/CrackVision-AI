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


def test_validate_mask_contract():
    from tools.annotate_mask import validate_mask_contract

    img = np.zeros((10, 10, 3), dtype=np.uint8)

    # Valid mask
    valid_mask = np.zeros((10, 10), dtype=np.uint8)
    valid_mask[2:5, 2:5] = 255
    ok, reason, stats = validate_mask_contract(img, valid_mask)
    assert ok is True
    assert reason == ""
    assert stats["crack_pixels"] == 9

    # Dimension mismatch
    bad_dim_mask = np.zeros((8, 10), dtype=np.uint8)
    ok, reason, _ = validate_mask_contract(img, bad_dim_mask)
    assert ok is False
    assert "dimensions do not match" in reason

    # Non-binary mask
    non_bin = np.zeros((10, 10), dtype=np.uint8)
    non_bin[0, 0] = 128
    ok, reason, _ = validate_mask_contract(img, non_bin)
    assert ok is False
    assert "Non-binary pixel values detected" in reason

    # Empty mask default (rejected) vs allowed
    empty_mask = np.zeros((10, 10), dtype=np.uint8)
    ok, reason, _ = validate_mask_contract(img, empty_mask, allow_empty=False)
    assert ok is False
    assert "0 foreground crack pixels" in reason

    ok, reason, _ = validate_mask_contract(img, empty_mask, allow_empty=True)
    assert ok is True


def test_load_existing_mask_safe_resume(tmp_path):
    from tools.annotate_mask import load_existing_mask

    mask_file = tmp_path / "valid_mask.png"
    arr = np.zeros((20, 20), dtype=np.uint8)
    arr[5, 5] = 255
    cv2.imwrite(str(mask_file), arr)

    # Valid load
    loaded = load_existing_mask(mask_file, (20, 20))
    assert loaded.shape == (20, 20)
    assert loaded[5, 5] == 255

    # Dimension mismatch raises ValueError
    import pytest
    with pytest.raises(ValueError, match=r"do not match expected image dimensions"):
        load_existing_mask(mask_file, (25, 25))

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_existing_mask(tmp_path / "missing.png", (20, 20))


def test_evaluate_dataset_hysteresis_strategies_mode(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[2, 2] = 255
    cv2.imwrite(str(image_dir / "00001.jpg"), image)
    cv2.imwrite(str(mask_dir / "00001_mask.png"), gt)

    dispatched = []

    def mock_segment(preprocessed, output_dir=None, **kwargs):
        dispatched.append(kwargs)
        res = np.zeros((10, 10), dtype=np.uint8)
        res[2, 2] = 255
        return {"final_mask": res}

    output_csv = tmp_path / "strategies_results.csv"
    report = evaluate_dataset(
        image_dir,
        mask_dir,
        output_csv,
        evaluation_mode="hysteresis_strategies",
        preprocess=lambda *a, **k: {"dummy": True},
        segment=mock_segment,
    )

    assert report["valid_pair_count"] == 1
    assert report["skipped_count"] == 0
    assert report["evaluation_mode"] == "hysteresis_strategies"
    assert len(report["aggregate_metrics"]) == 4

    # Verify parameters dispatched to mock_segment
    profiles_called = [d["segmentation_profile"] for d in dispatched]
    assert profiles_called == [
        "baseline",
        "hysteresis_recovery",
        "hysteresis_recovery",
        "hysteresis_recovery",
    ]

    # Strategy A: min_area=200, shape_filter=False
    strat_a_call = dispatched[1]
    assert strat_a_call["hysteresis_min_area"] == 200
    assert strat_a_call["hysteresis_use_shape_filter"] is False

    # Strategy B: min_area=80, shape_filter=False
    strat_b_call = dispatched[2]
    assert strat_b_call["hysteresis_min_area"] == 80
    assert strat_b_call["hysteresis_use_shape_filter"] is False

    # Strategy C: min_area=80, shape_filter=True
    strat_c_call = dispatched[3]
    assert strat_c_call["hysteresis_min_area"] == 80
    assert strat_c_call["hysteresis_use_shape_filter"] is True


def test_aggregate_metrics_summed_confusion_matrix():
    from tools.evaluate_validation_dataset import compute_aggregate_metrics_from_rows

    # Image 1: 10 TP, 0 FP, 0 FN (Precision=1.0, Recall=1.0)
    # Image 2: 1 TP, 9 FP, 9 FN (Precision=0.1, Recall=0.1)
    # Average of Precision = (1.0 + 0.1)/2 = 0.55
    # Summed TP = 11, Summed FP = 9 -> Precision = 11 / 20 = 0.55
    # Summed FN = 9 -> Recall = 11 / 20 = 0.55
    # But now consider asymmetric:
    # Image 1: 100 TP, 10 FP, 0 FN (Precision = 100/110 = 0.909)
    # Image 2: 1 TP, 10 FP, 0 FN (Precision = 1/11 = 0.0909)
    # Macro-average = (0.909 + 0.091) / 2 = 0.50
    # Summed TP = 101, Summed FP = 20 -> Summed Precision = 101 / 121 = 0.8347
    rows = [
        {
            "profile": "baseline",
            "status": "evaluated",
            "true_positive": 100,
            "false_positive": 10,
            "false_negative": 0,
            "true_negative": 1000,
        },
        {
            "profile": "baseline",
            "status": "evaluated",
            "true_positive": 1,
            "false_positive": 10,
            "false_negative": 0,
            "true_negative": 1000,
        },
    ]

    agg = compute_aggregate_metrics_from_rows(rows, "baseline")
    assert agg["true_positive"] == 101
    assert agg["false_positive"] == 20
    assert agg["false_negative"] == 0
    assert agg["true_negative"] == 2000
    expected_prec = 101 / 121
    assert abs(agg["precision"] - expected_prec) < 1e-6
    # Confirms it is NOT macro-averaged to ~0.50
    assert agg["precision"] > 0.80


def test_evaluate_dataset_skips_empty_mask_by_default(tmp_path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((5, 5, 3), dtype=np.uint8)
    empty_gt = np.zeros((5, 5), dtype=np.uint8)
    cv2.imwrite(str(image_dir / "00001.jpg"), image)
    cv2.imwrite(str(mask_dir / "00001_mask.png"), empty_gt)

    # Default skips empty mask
    report = evaluate_dataset(
        image_dir,
        mask_dir,
        tmp_path / "out.csv",
        profiles=("baseline",),
        preprocess=lambda *a, **k: {},
        segment=lambda *a, **k: {"final_mask": np.zeros((5, 5), dtype=np.uint8)},
    )
    assert report["valid_pair_count"] == 0
    assert report["skipped_count"] == 1
    assert "0 foreground crack pixels" in report["rows"][0]["error_message"]

    # With allow_empty_masks=True, it evaluates
    report2 = evaluate_dataset(
        image_dir,
        mask_dir,
        tmp_path / "out2.csv",
        profiles=("baseline",),
        preprocess=lambda *a, **k: {},
        segment=lambda *a, **k: {"final_mask": np.zeros((5, 5), dtype=np.uint8)},
        allow_empty_masks=True,
    )
    assert report2["valid_pair_count"] == 1
    assert report2["skipped_count"] == 0


def test_production_default_isolation():
    from src.segmentation import SEGMENTATION_PROFILES, select_segmentation_profile

    # Baseline remains the default profile
    ridge = np.zeros((10, 10), dtype=np.uint8)
    res = select_segmentation_profile(ridge, None)
    assert isinstance(res, np.ndarray)
    assert res.shape == (10, 10)

    # SEGMENTATION_PROFILES contains baseline first
    assert SEGMENTATION_PROFILES[0] == "baseline"
