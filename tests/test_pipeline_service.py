"""Tests for src/pipeline_service.py.

These tests are fully self-contained: they generate small synthetic BGR
images in-memory rather than reading from disk.  No measurement *accuracy*
claims are made; we only verify that the service returns well-formed,
JSON-serializable structures without crashing.

Run with:
    python -m unittest tests/test_pipeline_service.py   # requires only stdlib
    pytest tests/test_pipeline_service.py               # requires pytest
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import unittest

import cv2
import numpy as np

# Ensure project root is on path so 'src.*' imports work from any cwd.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.pipeline_service import (
    PipelineConfig,
    _SUPPORTED_ARUCO_DICTS,
    _SUPPORTED_BIN_METHODS,
    _SUPPORTED_PROFILES,
    _bgr_to_png_dataurl,
    _gray_to_png_dataurl,
    _safe_float,
    _safe_int,
    analyze_image_array,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_color_image(h: int = 64, w: int = 64, color_bgr=(128, 128, 128)) -> np.ndarray:
    """Return a small solid-color uint8 BGR image."""
    img = np.full((h, w, 3), color_bgr, dtype=np.uint8)
    return img


def _crack_image(h: int = 128, w: int = 128) -> np.ndarray:
    """Return a synthetic grey concrete image with a simple horizontal crack."""
    img = np.full((h, w, 3), (180, 180, 180), dtype=np.uint8)
    # Draw a dark horizontal crack of thickness ~4 px across the middle
    mid = h // 2
    img[mid - 2 : mid + 2, 10 : w - 10] = (30, 30, 30)
    return img


def _is_valid_png_dataurl(value: str) -> bool:
    """Return True if *value* is a valid data:image/png;base64,… URL."""
    if not isinstance(value, str):
        return False
    if not value.startswith("data:image/png;base64,"):
        return False
    b64 = value[len("data:image/png;base64,"):]
    try:
        decoded = base64.b64decode(b64)
        return decoded[:4] == b"\x89PNG"
    except Exception:
        return False


def _result_is_json_serializable(result: dict) -> bool:
    """Return True if *result* can be round-tripped through json.dumps/loads."""
    try:
        json.loads(json.dumps(result))
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Tests: _safe_float / _safe_int helpers
# ---------------------------------------------------------------------------

class TestSafeConversions(unittest.TestCase):
    def test_safe_float_normal(self):
        self.assertAlmostEqual(_safe_float(3.14), 3.14)
        self.assertAlmostEqual(_safe_float(np.float32(2.5)), 2.5)
        self.assertAlmostEqual(_safe_float(0), 0.0)

    def test_safe_float_nan_returns_none(self):
        self.assertIsNone(_safe_float(math.nan))
        self.assertIsNone(_safe_float(float("nan")))

    def test_safe_float_inf_returns_none(self):
        self.assertIsNone(_safe_float(math.inf))
        self.assertIsNone(_safe_float(-math.inf))

    def test_safe_float_none_returns_none(self):
        self.assertIsNone(_safe_float(None))

    def test_safe_float_string_returns_none(self):
        self.assertIsNone(_safe_float("not a number"))

    def test_safe_int_normal(self):
        self.assertEqual(_safe_int(42), 42)
        self.assertEqual(_safe_int(np.int32(7)), 7)

    def test_safe_int_none_returns_none(self):
        self.assertIsNone(_safe_int(None))


# ---------------------------------------------------------------------------
# Tests: image encoding helpers
# ---------------------------------------------------------------------------

class TestImageEncoding(unittest.TestCase):
    def test_bgr_to_png_dataurl_produces_valid_url(self):
        img = _solid_color_image()
        url = _bgr_to_png_dataurl(img)
        self.assertTrue(_is_valid_png_dataurl(url), "Expected valid PNG data URL")

    def test_bgr_to_png_dataurl_with_max_edge_downscales(self):
        img = _solid_color_image(h=400, w=600)
        url = _bgr_to_png_dataurl(img, max_edge=100)
        # Decode and check size
        b64 = url[len("data:image/png;base64,"):]
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        h, w = decoded.shape[:2]
        self.assertLessEqual(max(h, w), 100, "Longest edge should be ≤ max_edge")

    def test_bgr_to_png_dataurl_without_resize_keeps_size(self):
        img = _solid_color_image(h=64, w=64)
        url = _bgr_to_png_dataurl(img)  # no max_edge
        b64 = url[len("data:image/png;base64,"):]
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (64, 64, 3))

    def test_gray_to_png_dataurl_produces_valid_url(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        url = _gray_to_png_dataurl(gray)
        self.assertTrue(_is_valid_png_dataurl(url))

    def test_bgr_to_png_dataurl_rejects_empty_array(self):
        with self.assertRaises(ValueError):
            _bgr_to_png_dataurl(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_bgr_to_png_dataurl_rejects_none(self):
        with self.assertRaises((ValueError, AttributeError)):
            _bgr_to_png_dataurl(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests: PipelineConfig validation
# ---------------------------------------------------------------------------

class TestPipelineConfigValidation(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = PipelineConfig()
        cfg.validate()  # must not raise

    def test_invalid_segmentation_profile_raises(self):
        cfg = PipelineConfig(segmentation_profile="nonexistent")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_bin_method_raises(self):
        cfg = PipelineConfig(bin_method="median")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_invalid_fusion_mode_raises(self):
        cfg = PipelineConfig(fusion="unknown_mode")
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_negative_min_area_raises(self):
        cfg = PipelineConfig(min_area=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_zero_gap_closing_iters_raises(self):
        cfg = PipelineConfig(gap_closing_iters=0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_multiple_calibration_sources_raises(self):
        cfg = PipelineConfig(mm_per_pixel=0.05, marker_size_mm=50.0)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_unpaired_reference_length_mm_raises(self):
        cfg = PipelineConfig(reference_length_mm=100.0)  # missing px
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_unpaired_reference_length_px_raises(self):
        cfg = PipelineConfig(reference_length_px=2000.0)  # missing mm
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_negative_mm_per_pixel_raises(self):
        cfg = PipelineConfig(mm_per_pixel=-0.01)
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_paired_reference_lengths_are_valid(self):
        cfg = PipelineConfig(reference_length_mm=100.0, reference_length_px=2000.0)
        cfg.validate()  # must not raise

    def test_all_segmentation_profiles_accepted(self):
        for profile in _SUPPORTED_PROFILES:
            cfg = PipelineConfig(segmentation_profile=profile)
            cfg.validate()

    def test_all_bin_methods_accepted(self):
        for method in _SUPPORTED_BIN_METHODS:
            cfg = PipelineConfig(bin_method=method)
            cfg.validate()

    def test_invalid_aruco_dict_raises(self):
        cfg = PipelineConfig(aruco_dictionary="DICT_UNKNOWN")
        with self.assertRaises(ValueError):
            cfg.validate()


# ---------------------------------------------------------------------------
# Tests: analyze_image_array — input validation
# ---------------------------------------------------------------------------

class TestAnalyzeInputValidation(unittest.TestCase):
    def test_none_image_raises_value_error(self):
        with self.assertRaises((ValueError, AttributeError)):
            analyze_image_array(None)  # type: ignore[arg-type]

    def test_2channel_image_raises_value_error(self):
        bad = np.zeros((64, 64, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyze_image_array(bad)

    def test_wrong_dtype_raises_value_error(self):
        bad = np.zeros((64, 64, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            analyze_image_array(bad)

    def test_too_small_image_raises_value_error(self):
        tiny = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            analyze_image_array(tiny)

    def test_invalid_config_raises_value_error(self):
        img = _solid_color_image()
        cfg = PipelineConfig(segmentation_profile="bad_profile")
        with self.assertRaises(ValueError):
            analyze_image_array(img, cfg)


# ---------------------------------------------------------------------------
# Tests: analyze_image_array — result structure and serialization
# ---------------------------------------------------------------------------

class TestAnalyzeResultStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Run the pipeline once on a small synthetic crack image.

        Using setUpClass keeps the suite fast: the Frangi filter is the
        expensive step and runs only once per test class.
        """
        cls.image = _crack_image(h=128, w=128)
        cls.result = analyze_image_array(cls.image, image_name="test_crack")

    def test_top_level_keys_present(self):
        for key in ("image_name", "dimensions", "summary", "components", "visuals"):
            self.assertIn(key, self.result, f"Missing top-level key: '{key}'")

    def test_image_name_is_correct(self):
        self.assertEqual(self.result["image_name"], "test_crack")

    def test_dimensions_match_input(self):
        dims = self.result["dimensions"]
        self.assertEqual(dims["width"], 128)
        self.assertEqual(dims["height"], 128)

    def test_summary_measurement_keys_present(self):
        summary = self.result["summary"]
        for key in (
            "crack_coverage_percent",
            "crack_pixels",
            "max_width_px",
            "mean_width_px",
            "median_width_px",
            "std_width_px",
            "skeleton_points",
            "total_length_px",
            "component_count",
            "raw_component_count",
            "dominant_angle_deg",
            "dominant_type",
            "calibration",
        ):
            self.assertIn(key, summary, f"Missing summary key: '{key}'")

    def test_no_mm_fields_without_calibration(self):
        summary = self.result["summary"]
        for key in ("max_width_mm", "mean_width_mm", "total_length_mm"):
            self.assertNotIn(key, summary, f"mm key '{key}' present without calibration")

    def test_calibration_is_not_calibrated_by_default(self):
        cal = self.result["summary"]["calibration"]
        self.assertFalse(cal["is_calibrated"])
        self.assertIsNone(cal["mm_per_pixel"])

    def test_all_visual_keys_present(self):
        visuals = self.result["visuals"]
        for key in (
            "original", "crack_mask", "overlay",
            "width_heatmap", "length_map", "orientation_map",
        ):
            self.assertIn(key, visuals, f"Missing visual key: '{key}'")

    def test_all_visuals_are_valid_png_data_urls(self):
        for key, value in self.result["visuals"].items():
            if value is None:
                continue  # rectified_marker is None when not calibrated
            self.assertTrue(
                _is_valid_png_dataurl(value),
                f"Visual '{key}' is not a valid PNG data URL",
            )

    def test_result_is_json_serializable(self):
        self.assertTrue(
            _result_is_json_serializable(self.result),
            "Result cannot be round-tripped through JSON",
        )

    def test_no_numpy_arrays_in_result(self):
        """Walk the entire result tree and ensure no NumPy arrays leaked."""
        def _walk(obj):
            if isinstance(obj, np.ndarray):
                return False
            if isinstance(obj, dict):
                return all(_walk(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return all(_walk(v) for v in obj)
            return True

        self.assertTrue(_walk(self.result), "NumPy array found in result tree")

    def test_no_nan_or_inf_in_result(self):
        """Walk floats and confirm none are NaN or ±Inf."""
        def _walk(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return False
            if isinstance(obj, dict):
                return all(_walk(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return all(_walk(v) for v in obj)
            return True

        self.assertTrue(_walk(self.result), "NaN or Inf found in result tree")

    def test_components_are_list(self):
        self.assertIsInstance(self.result["components"], list)

    def test_component_keys_when_present(self):
        for comp in self.result["components"]:
            for key in ("id", "length_px", "skeleton_pixels", "mask_area_px",
                        "centroid_x", "centroid_y",
                        "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
                self.assertIn(key, comp, f"Component missing key '{key}'")


# ---------------------------------------------------------------------------
# Tests: calibration behavior
# ---------------------------------------------------------------------------

class TestCalibrationBehavior(unittest.TestCase):
    def setUp(self):
        self.image = _crack_image(h=128, w=128)

    def test_mm_per_pixel_calibration_adds_mm_fields(self):
        cfg = PipelineConfig(mm_per_pixel=0.05)
        result = analyze_image_array(self.image, cfg)
        summary = result["summary"]
        cal = summary["calibration"]
        self.assertTrue(cal["is_calibrated"])
        self.assertAlmostEqual(cal["mm_per_pixel"], 0.05, places=6)
        # At least total_length_mm should appear when calibrated
        # (it may be 0.0 if no cracks detected in the tiny image)
        self.assertIn("total_length_mm", summary)

    def test_reference_length_calibration(self):
        cfg = PipelineConfig(reference_length_mm=50.0, reference_length_px=1000.0)
        result = analyze_image_array(self.image, cfg)
        cal = result["summary"]["calibration"]
        self.assertTrue(cal["is_calibrated"])
        self.assertAlmostEqual(cal["mm_per_pixel"], 0.05, places=6)

    def test_no_calibration_source_leaves_uncalibrated(self):
        cfg = PipelineConfig()
        result = analyze_image_array(self.image, cfg)
        cal = result["summary"]["calibration"]
        self.assertFalse(cal["is_calibrated"])
        self.assertIsNone(cal["mm_per_pixel"])

    def test_contradictory_calibration_rejected_by_validate(self):
        cfg = PipelineConfig(mm_per_pixel=0.05, reference_length_mm=50.0)
        with self.assertRaises(ValueError):
            analyze_image_array(self.image, cfg)


# ---------------------------------------------------------------------------
# Tests: segmentation profiles
# ---------------------------------------------------------------------------

class TestSegmentationProfiles(unittest.TestCase):
    def setUp(self):
        self.image = _crack_image(h=128, w=128)

    def test_baseline_profile_runs(self):
        cfg = PipelineConfig(segmentation_profile="baseline")
        result = analyze_image_array(self.image, cfg)
        self.assertIn("summary", result)

    def test_adaptive_recovery_profile_runs(self):
        cfg = PipelineConfig(segmentation_profile="adaptive_recovery")
        result = analyze_image_array(self.image, cfg)
        self.assertIn("summary", result)

    def test_guided_recovery_profile_runs(self):
        cfg = PipelineConfig(segmentation_profile="guided_recovery")
        result = analyze_image_array(self.image, cfg)
        self.assertIn("summary", result)


# ---------------------------------------------------------------------------
# Tests: empty / featureless images
# ---------------------------------------------------------------------------

class TestEdgeCaseImages(unittest.TestCase):
    def test_blank_white_image_does_not_crash(self):
        """A pure white image has no cracks; all measurements should be zero."""
        img = np.full((128, 128, 3), 255, dtype=np.uint8)
        result = analyze_image_array(img)
        summary = result["summary"]
        self.assertEqual(summary["component_count"], 0)

    def test_blank_black_image_does_not_crash(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        result = analyze_image_array(img)
        self.assertIn("summary", result)

    def test_minimum_valid_size_image(self):
        """16×16 is the declared minimum; must not raise."""
        img = _solid_color_image(h=16, w=16)
        result = analyze_image_array(img)
        self.assertEqual(result["dimensions"]["width"], 16)

    def test_gap_closing_disabled(self):
        """Setting gap_closing_ksize=0 must not raise."""
        img = _crack_image()
        cfg = PipelineConfig(gap_closing_ksize=0)
        result = analyze_image_array(img, cfg)
        self.assertIn("summary", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
