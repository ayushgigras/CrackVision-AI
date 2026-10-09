import csv
import inspect

import cv2
import numpy as np

from main import SUMMARY_FIELDS, discover_images, process_single_image, write_summary_csv


def test_discover_images_finds_supported_files_only(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"image")
    (tmp_path / "a.PNG").write_bytes(b"image")
    (tmp_path / "ignore.txt").write_text("not an image", encoding="utf-8")

    paths = discover_images(str(tmp_path), {".jpg", ".png"})

    assert [path.split("\\")[-1] for path in paths] == ["a.PNG", "b.jpg"]


def test_write_summary_csv_has_required_columns_and_rows(tmp_path):
    output = tmp_path / "summary.csv"
    row = {
        "image_filename": "00001.jpg",
        "crack_coverage_percent": 1.25,
        "component_count": 2,
        "total_length_px": 42.0,
        "max_width_px": 4.0,
        "mean_width_px": 2.0,
        "dominant_orientation_deg": 30.0,
        "status": "success",
        "error_message": "",
    }

    write_summary_csv(str(output), [row])

    with output.open(newline="", encoding="utf-8") as csv_file:
        records = list(csv.DictReader(csv_file))
    with output.open(encoding="utf-8") as csv_file:
        assert csv.DictReader(csv_file).fieldnames == SUMMARY_FIELDS
    assert records[0]["image_filename"] == "00001.jpg"
    assert records[0]["status"] == "success"


def test_failed_image_is_readable_as_a_failed_batch_candidate(tmp_path):
    invalid = tmp_path / "broken.jpg"
    invalid.write_text("not an image", encoding="utf-8")
    discovered = discover_images(str(tmp_path), {".jpg"})

    assert discovered == [str(invalid)]
    assert cv2.imread(discovered[0]) is None


def test_single_image_keeps_legacy_visualization_default():
    signature = inspect.signature(process_single_image)

    assert signature.parameters["save_legacy_visualization"].default is True
