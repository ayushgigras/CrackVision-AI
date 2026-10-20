"""Unit and integration tests for persistent analysis history, PDF reports, and CSV exports."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api import app
from src.db import delete_analysis, get_analysis, init_db, list_analyses, save_analysis
from src.report import (
    generate_analysis_csv,
    generate_history_csv,
    generate_pdf_report,
)


def _make_sample_result(
    analysis_id: str = "test-analysis-uuid-1",
    is_calibrated: bool = True,
    client_id: str = "default",
) -> dict:
    """Create a mock analysis result matching pipeline_service schema."""
    # Synthetic small image
    img = np.full((64, 64, 3), 180, dtype=np.uint8)
    img[30:34, 10:50] = (20, 20, 20)
    _, buf = cv2.imencode(".png", img)
    import base64
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"

    summary = {
        "crack_coverage_percent": 1.25,
        "crack_pixels": 160,
        "max_width_px": 4.0,
        "mean_width_px": 3.2,
        "median_width_px": 3.0,
        "std_width_px": 0.4,
        "total_length_px": 40.0,
        "component_count": 1,
        "raw_component_count": 1,
        "dominant_angle_deg": 0.0,
        "dominant_type": "Horizontal",
        "calibration": {
            "is_calibrated": is_calibrated,
            "status": "manual_scale" if is_calibrated else "not_calibrated",
            "mm_per_pixel": 0.15 if is_calibrated else None,
            "marker_id": None,
            "marker_size_mm": None,
        },
    }
    if is_calibrated:
        summary["max_width_mm"] = 0.60
        summary["mean_width_mm"] = 0.48
        summary["median_width_mm"] = 0.45
        summary["total_length_mm"] = 6.0

    return {
        "analysis_id": analysis_id,
        "client_id": client_id,
        "image_name": "test_crack_image",
        "timestamp": "2026-10-20T12:00:00Z",
        "segmentation_profile": "baseline",
        "dimensions": {"width": 64, "height": 64},
        "summary": summary,
        "components": [
            {
                "id": 1,
                "length_px": 40.0,
                "length_mm": 6.0 if is_calibrated else None,
                "mask_area_px": 160,
                "bbox_x": 10,
                "bbox_y": 30,
                "bbox_w": 40,
                "bbox_h": 4,
            }
        ],
        "visuals": {
            "original": data_url,
            "overlay": data_url,
            "crack_mask": data_url,
            "width_heatmap": data_url,
            "length_map": data_url,
            "orientation_map": data_url,
        },
    }


class TestDatabasePersistence(unittest.TestCase):
    """Tests for SQLite database operations in src/db.py."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_crackgauge.db")
        init_db(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_and_get_analysis(self):
        data = _make_sample_result("test-record-1")
        aid = save_analysis(data, client_id="operator_a", db_path=self.db_path)
        self.assertEqual(aid, "test-record-1")

        retrieved = get_analysis(aid, db_path=self.db_path)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["analysis_id"], aid)
        self.assertEqual(retrieved["client_id"], "operator_a")
        self.assertEqual(retrieved["image_name"], "test_crack_image")
        self.assertIn("visuals", retrieved)
        self.assertIn("overlay", retrieved["visuals"])
        self.assertTrue(retrieved["summary"]["calibration"]["is_calibrated"])

    def test_list_analyses_with_client_ownership(self):
        rec_a = _make_sample_result("rec-client-a", client_id="client_a")
        rec_b = _make_sample_result("rec-client-b", client_id="client_b")
        save_analysis(rec_a, client_id="client_a", db_path=self.db_path)
        save_analysis(rec_b, client_id="client_b", db_path=self.db_path)

        all_items, total = list_analyses(db_path=self.db_path)
        self.assertEqual(total, 2)
        self.assertEqual(len(all_items), 2)

        items_a, total_a = list_analyses(client_id="client_a", db_path=self.db_path)
        self.assertEqual(total_a, 1)
        self.assertEqual(items_a[0]["analysis_id"], "rec-client-a")

    def test_delete_analysis(self):
        data = _make_sample_result("rec-to-delete")
        save_analysis(data, db_path=self.db_path)

        deleted = delete_analysis("rec-to-delete", db_path=self.db_path)
        self.assertTrue(deleted)
        self.assertIsNone(get_analysis("rec-to-delete", db_path=self.db_path))

    def test_validate_id_rejects_unsafe_chars(self):
        with self.assertRaises(ValueError):
            get_analysis("drop table analyses;--", db_path=self.db_path)


class TestReportGenerators(unittest.TestCase):
    """Tests for PDF and CSV generation in src/report.py."""

    def test_generate_pdf_report_calibrated(self):
        data = _make_sample_result("pdf-test-cal", is_calibrated=True)
        pdf_bytes = generate_pdf_report(data)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_generate_pdf_report_uncalibrated(self):
        data = _make_sample_result("pdf-test-uncal", is_calibrated=False)
        pdf_bytes = generate_pdf_report(data)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))
        self.assertGreater(len(pdf_bytes), 2000)

    def test_generate_analysis_csv(self):
        data = _make_sample_result("csv-test-1", is_calibrated=True)
        csv_str = generate_analysis_csv(data)
        self.assertIn("analysis_id,image_name,timestamp", csv_str)
        self.assertIn("csv-test-1", csv_str)
        self.assertIn("test_crack_image", csv_str)
        self.assertIn("0.6", csv_str)  # max_width_mm

    def test_generate_history_csv(self):
        items = [
            {
                "analysis_id": "h1",
                "image_name": "img1.png",
                "timestamp": "2026-10-20T10:00:00Z",
                "segmentation_profile": "baseline",
                "is_calibrated": True,
                "calibration_status": "manual_scale",
                "mm_per_pixel": 0.15,
                "dimensions": {"width": 100, "height": 100},
                "max_width_px": 5.0,
                "max_width_mm": 0.75,
                "mean_width_px": 3.0,
                "mean_width_mm": 0.45,
                "total_length_px": 50.0,
                "total_length_mm": 7.5,
                "dominant_angle_deg": 12.0,
                "dominant_type": "Horizontal",
                "crack_coverage_percent": 1.1,
                "crack_pixels": 110,
                "component_count": 1,
            }
        ]
        csv_str = generate_history_csv(items)
        self.assertIn("analysis_id", csv_str)
        self.assertIn("h1", csv_str)
        self.assertIn("img1.png", csv_str)


class TestApiHistoryAndReportingEndpoints(unittest.TestCase):
    """Integration tests for history, PDF, and CSV endpoints in src/api.py."""

    def setUp(self):
        self.client = TestClient(app)

    def test_history_list_endpoint(self):
        response = self.client.get("/api/history")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("analyses", data)
        self.assertIn("total", data)

    def test_post_report_pdf_endpoint(self):
        payload = _make_sample_result("direct-pdf-report")
        response = self.client.post("/api/report/pdf", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "application/pdf")
        self.assertTrue(response.content.startswith(b"%PDF"))

    def test_export_all_history_csv_endpoint(self):
        response = self.client.get("/api/history/export/csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        self.assertIn("analysis_id", response.text)
