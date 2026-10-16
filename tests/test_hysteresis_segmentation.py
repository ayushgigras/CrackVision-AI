"""Focused tests for the hysteresis_recovery segmentation profile.

Covers:
  - Connected weak-ridge pixels anchored to strong seeds are accepted.
  - Disconnected weak-ridge islands (no strong pixel) are rejected.
  - Threshold validation (low >= high, out-of-range).
  - Profile integration via select_segmentation_profile.
  - Existing profiles are unaffected.
"""

import cv2
import numpy as np
import pytest

from src.segmentation import (
    binarize_ridge_hysteresis,
    select_segmentation_profile,
)


# ──────────────────────────────────────────────────────────────────────────────
# binarize_ridge_hysteresis – unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestBinarizeRidgeHysteresis:
    """Direct tests for the dual-threshold hysteresis function."""

    def test_connected_weak_pixels_accepted(self):
        """Weak pixels 8-connected to a strong pixel are accepted."""
        img = np.zeros((5, 5), dtype=np.uint8)
        # Strong pixel at centre
        img[2, 2] = 60
        # Weak pixels connected to it (diagonal + adjacent)
        img[1, 1] = 30
        img[2, 3] = 35
        img[3, 3] = 28

        result = binarize_ridge_hysteresis(img, low_thresh=25, high_thresh=50)

        # All four should be accepted
        assert result[2, 2] == 255, "Strong pixel must be accepted"
        assert result[1, 1] == 255, "Connected weak pixel must be accepted"
        assert result[2, 3] == 255, "Adjacent weak pixel must be accepted"
        assert result[3, 3] == 255, "Chain-connected weak pixel must be accepted"

    def test_disconnected_weak_pixels_rejected(self):
        """Weak pixels NOT connected to any strong pixel are rejected."""
        img = np.zeros((10, 10), dtype=np.uint8)
        # Strong cluster in top-left
        img[0, 0] = 60
        img[0, 1] = 30

        # Disconnected weak island in bottom-right (no strong neighbour)
        img[8, 8] = 30
        img[8, 9] = 35
        img[9, 8] = 28

        result = binarize_ridge_hysteresis(img, low_thresh=25, high_thresh=50)

        # Top-left cluster accepted
        assert result[0, 0] == 255
        assert result[0, 1] == 255

        # Bottom-right island rejected
        assert result[8, 8] == 0, "Disconnected weak pixel must be rejected"
        assert result[8, 9] == 0, "Disconnected weak pixel must be rejected"
        assert result[9, 8] == 0, "Disconnected weak pixel must be rejected"

    def test_below_low_threshold_always_rejected(self):
        """Pixels below low_thresh are never accepted, even if adjacent to strong."""
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 2] = 60   # strong
        img[2, 3] = 20   # below low_thresh

        result = binarize_ridge_hysteresis(img, low_thresh=25, high_thresh=50)

        assert result[2, 2] == 255
        assert result[2, 3] == 0, "Below-low pixel must never be accepted"

    def test_all_below_low_yields_empty(self):
        """If no pixel reaches low_thresh the output is entirely black."""
        img = np.full((4, 4), 10, dtype=np.uint8)
        result = binarize_ridge_hysteresis(img, low_thresh=20, high_thresh=50)
        assert np.sum(result) == 0

    def test_all_strong_yields_full_mask(self):
        """If every pixel is >= high_thresh, all should be accepted."""
        img = np.full((3, 3), 200, dtype=np.uint8)
        result = binarize_ridge_hysteresis(img, low_thresh=25, high_thresh=50)
        assert np.all(result == 255)

    def test_chain_connectivity(self):
        """A chain of weak pixels connecting two strong seeds forms one region."""
        img = np.zeros((1, 10), dtype=np.uint8)
        img[0, 0] = 60   # strong
        img[0, 1] = 30   # weak
        img[0, 2] = 28   # weak
        img[0, 3] = 35   # weak
        img[0, 4] = 26   # weak
        img[0, 5] = 60   # strong
        # gap (=0) at 6
        img[0, 7] = 30   # disconnected weak
        img[0, 8] = 30   # disconnected weak

        result = binarize_ridge_hysteresis(img, low_thresh=25, high_thresh=50)

        # Connected chain accepted
        for col in range(6):
            assert result[0, col] == 255, f"Column {col} should be accepted"

        # Disconnected tail rejected
        assert result[0, 7] == 0, "Disconnected weak at col 7 must be rejected"
        assert result[0, 8] == 0, "Disconnected weak at col 8 must be rejected"


class TestBinarizeRidgeHysteresisValidation:
    """Threshold validation edge cases."""

    def test_low_equals_high_raises(self):
        with pytest.raises(ValueError, match="low_thresh"):
            binarize_ridge_hysteresis(
                np.zeros((2, 2), dtype=np.uint8), low_thresh=50, high_thresh=50
            )

    def test_low_greater_than_high_raises(self):
        with pytest.raises(ValueError, match="low_thresh"):
            binarize_ridge_hysteresis(
                np.zeros((2, 2), dtype=np.uint8), low_thresh=80, high_thresh=50
            )

    def test_negative_low_raises(self):
        with pytest.raises(ValueError, match="low_thresh"):
            binarize_ridge_hysteresis(
                np.zeros((2, 2), dtype=np.uint8), low_thresh=-1, high_thresh=50
            )

    def test_high_exceeds_255_raises(self):
        with pytest.raises(ValueError, match="low_thresh"):
            binarize_ridge_hysteresis(
                np.zeros((2, 2), dtype=np.uint8), low_thresh=10, high_thresh=256
            )


# ──────────────────────────────────────────────────────────────────────────────
# select_segmentation_profile – hysteresis_recovery integration
# ──────────────────────────────────────────────────────────────────────────────


class TestHysteresisProfileIntegration:
    """Test the profile dispatch wiring for hysteresis_recovery."""

    def _make_ridge_norm(self):
        """Create a ridge_norm with a strong+weak pattern large enough to
        survive the min_area=80 component filtering inside the profile.

        Layout (50x50):
          - A horizontal stripe of 100 strong pixels (rows 20-21, cols 0-49)
          - A connected weak row just below (row 22, cols 0-49)
          - A disconnected weak pixel at (0, 0)
        """
        rn = np.zeros((50, 50), dtype=np.uint8)
        rn[20, :] = 60   # strong row
        rn[21, :] = 60   # strong row (100 px total – survives min_area=80)
        rn[22, :] = 30   # weak row – 8-connected to strong rows above
        rn[0, 0] = 30    # weak – disconnected from strong
        return rn

    def test_hysteresis_profile_produces_correct_mask(self):
        rn = self._make_ridge_norm()
        ridge_mask = np.zeros_like(rn, dtype=np.uint8)

        result = select_segmentation_profile(
            ridge_mask,
            None,
            profile="hysteresis_recovery",
            ridge_norm=rn,
            hysteresis_low=25,
            hysteresis_high=50,
        )

        # Strong + connected weak rows all accepted (component area = 150)
        assert result[20, 25] == 255, "Strong pixel must survive"
        assert result[22, 25] == 255, "Connected weak pixel must survive"
        # Disconnected weak at (0,0): only 1 pixel, no strong neighbour,
        # rejected by hysteresis AND by min_area filter.
        assert result[0, 0] == 0, "Disconnected weak pixel must be rejected"

    def test_hysteresis_profile_requires_ridge_norm(self):
        ridge_mask = np.zeros((3, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="ridge_norm"):
            select_segmentation_profile(
                ridge_mask, None, profile="hysteresis_recovery"
            )

    def test_baseline_unaffected(self):
        """Baseline profile still behaves exactly as before."""
        ridge = np.array([[255, 0], [0, 0]], dtype=np.uint8)
        adaptive = np.array([[0, 255], [0, 0]], dtype=np.uint8)
        result = select_segmentation_profile(ridge, adaptive)
        np.testing.assert_array_equal(result, ridge)

    def test_guided_recovery_unaffected(self):
        """Guided recovery still keeps only adaptive components near ridges."""
        ridge = np.zeros((7, 7), dtype=np.uint8)
        ridge[3, 3] = 255
        adaptive = np.zeros_like(ridge)
        adaptive[3, 4] = 255
        adaptive[0, 0] = 255

        result = select_segmentation_profile(ridge, adaptive, "guided_recovery")

        assert result[3, 3] == 255
        assert result[3, 4] == 255
        assert result[0, 0] == 0

    def test_hysteresis_custom_min_area_filtering(self):
        """Verify that hysteresis_min_area filters components by size."""
        # 100x100 canvas with two distinct strong components:
        # Comp A: 120 pixels (e.g. 12x10 rect)
        # Comp B: 180 pixels (e.g. 18x10 rect)
        rn = np.zeros((100, 100), dtype=np.uint8)
        rn[10:22, 10:20] = 60   # 12 * 10 = 120 pixels
        rn[50:68, 50:60] = 60   # 18 * 10 = 180 pixels
        ridge_mask = np.zeros_like(rn)

        # min_area=80: both 120 and 180 survive
        res_80 = select_segmentation_profile(
            ridge_mask, None, profile="hysteresis_recovery",
            ridge_norm=rn, hysteresis_min_area=80
        )
        assert np.sum(res_80[10:22, 10:20] > 0) == 120
        assert np.sum(res_80[50:68, 50:60] > 0) == 180

        # min_area=150: 120 filtered out, 180 survives
        res_150 = select_segmentation_profile(
            ridge_mask, None, profile="hysteresis_recovery",
            ridge_norm=rn, hysteresis_min_area=150
        )
        assert np.sum(res_150[10:22, 10:20] > 0) == 0
        assert np.sum(res_150[50:68, 50:60] > 0) == 180

        # min_area=200: both filtered out
        res_200 = select_segmentation_profile(
            ridge_mask, None, profile="hysteresis_recovery",
            ridge_norm=rn, hysteresis_min_area=200
        )
        assert np.sum(res_200 > 0) == 0

    def test_hysteresis_rejects_negative_min_area(self):
        rn = np.zeros((10, 10), dtype=np.uint8)
        ridge_mask = np.zeros_like(rn)
        with pytest.raises(ValueError, match="hysteresis_min_area"):
            select_segmentation_profile(
                ridge_mask, None, profile="hysteresis_recovery",
                ridge_norm=rn, hysteresis_min_area=-1
            )

    def test_run_segmentation_pipeline_hysteresis_profile(self):
        """Verify run_segmentation_pipeline works with hysteresis_recovery."""
        from src.segmentation import run_segmentation_pipeline
        # Create a synthetic preprocessed dict
        img = np.full((60, 60), 200, dtype=np.uint8)
        # Draw a dark crack line
        img[30, 10:50] = 30
        preproc = {
            'image_name': 'test_hyst',
            'original': cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
            'enhanced': img,
            'crack_mask': None,
        }
        res = run_segmentation_pipeline(
            preproc,
            segmentation_profile="hysteresis_recovery",
            hysteresis_min_area=20,
            save_steps=False,
        )
        assert "final_mask" in res
        assert res["final_mask"].shape == (60, 60)

    def test_filter_crack_components_by_shape(self):
        """Test unit behavior of shape-aware component filtering."""
        from src.segmentation import filter_crack_components_by_shape
        # Create 150x150 canvas
        canvas = np.zeros((150, 150), dtype=np.uint8)

        # 1. Large component: 15x16 = 240 px (aspect ~ 1.0, but area >= 200) -> must survive
        canvas[10:25, 10:26] = 255

        # 2. Compact interior component: 10x10 = 100 px (aspect 1.0, interior) -> must be pruned
        canvas[50:60, 50:60] = 255

        # 3. Elongated interior component: 3x35 = 105 px (aspect > 10, skel > 30) -> must survive
        canvas[80:83, 20:55] = 255

        # 4. Moderately elongated boundary component: 6x16 = 96 px at x=0 (touches boundary) -> survives under boundary threshold
        canvas[110:116, 0:16] = 255

        # 5. Micro component: 5x5 = 25 px (< 80) -> must be pruned
        canvas[130:135, 130:135] = 255

        cleaned, kept = filter_crack_components_by_shape(
            canvas,
            min_area=80,
            large_area_thresh=200,
            interior_aspect_thresh=2.2,
            boundary_aspect_thresh=1.6,
            min_skeleton_length=10,
        )

        assert kept == 3, f"Expected 3 components kept, got {kept}"
        # Large component survives
        assert np.sum(cleaned[10:25, 10:26] > 0) == 240
        # Compact interior pruned
        assert np.sum(cleaned[50:60, 50:60] > 0) == 0
        # Elongated interior survives
        assert np.sum(cleaned[80:83, 20:55] > 0) == 105
        # Boundary component survives
        assert np.sum(cleaned[110:116, 0:16] > 0) == 96
        # Micro component pruned
        assert np.sum(cleaned[130:135, 130:135] > 0) == 0

    def test_select_segmentation_profile_shape_filter_integration(self):
        """Test profile selection with hysteresis_use_shape_filter=True."""
        rn = np.zeros((100, 100), dtype=np.uint8)
        # Compact component: 10x10 strong pixels = 100 px (aspect 1.0)
        rn[10:20, 10:20] = 60
        # Elongated component: 3x35 strong pixels = 105 px (aspect ~ 11)
        rn[50:53, 20:55] = 60
        ridge_mask = np.zeros_like(rn)

        # With standard area filtering (min_area=80), BOTH survive
        res_standard = select_segmentation_profile(
            ridge_mask, None, profile="hysteresis_recovery",
            ridge_norm=rn, hysteresis_min_area=80, hysteresis_use_shape_filter=False
        )
        assert np.sum(res_standard[10:20, 10:20] > 0) == 100
        assert np.sum(res_standard[50:53, 20:55] > 0) == 105

        # With shape filter (min_area=80, use_shape_filter=True), compact is pruned, elongated survives
        res_shape = select_segmentation_profile(
            ridge_mask, None, profile="hysteresis_recovery",
            ridge_norm=rn, hysteresis_min_area=80, hysteresis_use_shape_filter=True,
            hysteresis_min_skeleton_length=10,
        )
        assert np.sum(res_shape[10:20, 10:20] > 0) == 0
        assert np.sum(res_shape[50:53, 20:55] > 0) == 105
