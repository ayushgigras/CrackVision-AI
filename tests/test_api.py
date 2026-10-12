"""Integration tests for CrackGauge FastAPI endpoints (src/api.py).

Verifies:
- GET /api/health returns 200 and capability info.
- POST /api/analyze processes valid images and returns 200 with JSON result.
- POST /api/analyze validates image formats, sizes, and configuration errors (415, 422).
- POST /api/analyze handles calibration parameters.
"""

from __future__ import annotations

import io
import os
import sys
import unittest

import cv2
import numpy as np

# Ensure project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi.testclient import TestClient
from src.api import app


def _create_synthetic_image_bytes(h: int = 128, w: int = 128, ext: str = ".png") -> bytes:
    """Create encoded bytes of a synthetic image with a horizontal crack."""
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    mid = h // 2
    img[mid - 2 : mid + 2, 10 : w - 10] = (30, 30, 30)
    success, buf = cv2.imencode(ext, img)
    assert success, "Failed to encode synthetic test image"
    return buf.tobytes()


class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /api/health."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_returns_200_and_expected_fields(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("version", data)
        self.assertIn("supported_segmentation_profiles", data)
        self.assertIn("supported_bin_methods", data)
        self.assertIn("supported_aruco_dictionaries", data)
        self.assertIn("aruco_available", data)


class TestAnalyzeEndpoint(unittest.TestCase):
    """Tests for POST /api/analyze."""

    def setUp(self):
        self.client = TestClient(app)

    def test_analyze_valid_png_baseline(self):
        img_bytes = _create_synthetic_image_bytes(128, 128, ".png")
        response = self.client.post(
            "/api/analyze",
            files={"file": ("test_crack.png", io.BytesIO(img_bytes), "image/png")},
            data={"segmentation_profile": "baseline"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check top-level structure
        self.assertEqual(data["image_name"], "test_crack")
        self.assertIn("dimensions", data)
        self.assertEqual(data["dimensions"]["width"], 128)
        self.assertEqual(data["dimensions"]["height"], 128)
        self.assertIn("summary", data)
        self.assertIn("components", data)
        self.assertIn("visuals", data)

        # Check visuals are base64 data URLs
        visuals = data["visuals"]
        for key in ("overlay", "width_heatmap", "length_map", "orientation_map"):
            self.assertIn(key, visuals)
            self.assertTrue(visuals[key].startswith("data:image/png;base64,"))

        # Check summary measurements
        summary = data["summary"]
        self.assertGreater(summary["total_length_px"], 0.0)
        self.assertGreater(summary["max_width_px"], 0.0)
        self.assertFalse(summary["calibration"]["is_calibrated"])

    def test_analyze_with_direct_mm_per_pixel(self):
        img_bytes = _create_synthetic_image_bytes(128, 128, ".png")
        response = self.client.post(
            "/api/analyze",
            files={"file": ("test_crack.png", io.BytesIO(img_bytes), "image/png")},
            data={
                "segmentation_profile": "baseline",
                "mm_per_pixel": "0.15",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        summary = data["summary"]
        self.assertTrue(summary["calibration"]["is_calibrated"])
        self.assertAlmostEqual(summary["calibration"]["mm_per_pixel"], 0.15, places=3)
        self.assertIn("max_width_mm", summary)
        self.assertIn("total_length_mm", summary)

    def test_analyze_rejects_unsupported_extension(self):
        response = self.client.post(
            "/api/analyze",
            files={"file": ("document.txt", io.BytesIO(b"dummy text"), "text/plain")},
        )
        self.assertEqual(response.status_code, 415)

    def test_analyze_rejects_empty_file(self):
        response = self.client.post(
            "/api/analyze",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_rejects_corrupted_image(self):
        response = self.client.post(
            "/api/analyze",
            files={"file": ("corrupt.png", io.BytesIO(b"not a valid png stream"), "image/png")},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_rejects_invalid_profile(self):
        img_bytes = _create_synthetic_image_bytes(64, 64, ".png")
        response = self.client.post(
            "/api/analyze",
            files={"file": ("crack.png", io.BytesIO(img_bytes), "image/png")},
            data={"segmentation_profile": "non_existent_profile"},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_rejects_contradictory_calibration(self):
        img_bytes = _create_synthetic_image_bytes(64, 64, ".png")
        response = self.client.post(
            "/api/analyze",
            files={"file": ("crack.png", io.BytesIO(img_bytes), "image/png")},
            data={
                "mm_per_pixel": "0.1",
                "marker_size_mm": "50.0",
            },
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
