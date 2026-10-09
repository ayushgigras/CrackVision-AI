import numpy as np
import pytest

from src.segmentation import select_segmentation_profile


def test_baseline_profile_preserves_frangi_primary_behavior():
    ridge = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    adaptive = np.array([[0, 255], [0, 0]], dtype=np.uint8)

    result = select_segmentation_profile(ridge, adaptive)

    np.testing.assert_array_equal(result, ridge)


def test_adaptive_recovery_uses_cleaned_adaptive_mask():
    ridge = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    adaptive = np.array([[0, 255], [0, 128]], dtype=np.uint8)

    result = select_segmentation_profile(ridge, adaptive, "adaptive_recovery")

    np.testing.assert_array_equal(result, np.array([[0, 255], [0, 255]], dtype=np.uint8))


def test_guided_recovery_keeps_only_adaptive_components_near_ridges():
    ridge = np.zeros((7, 7), dtype=np.uint8)
    ridge[3, 3] = 255
    adaptive = np.zeros_like(ridge)
    adaptive[3, 4] = 255
    adaptive[0, 0] = 255

    result = select_segmentation_profile(ridge, adaptive, "guided_recovery")

    assert result[3, 3] == 255
    assert result[3, 4] == 255
    assert result[0, 0] == 0


def test_invalid_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown segmentation profile"):
        select_segmentation_profile(np.zeros((2, 2), dtype=np.uint8), None, "unknown")


@pytest.mark.parametrize(
    "ridge, adaptive, message",
    [
        (np.zeros((2, 2, 1), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), "ridge_mask"),
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8), "matching dimensions"),
        (np.zeros((2, 2), dtype=np.uint8), None, "requires an adaptive mask"),
    ],
)
def test_profile_masks_are_validated(ridge, adaptive, message):
    profile = "guided_recovery" if adaptive is None else "adaptive_recovery"
    expected_error = TypeError if ridge.dtype == np.float32 else ValueError
    with pytest.raises(expected_error, match=message if expected_error is ValueError else "dtype"):
        select_segmentation_profile(ridge, adaptive, profile)


def test_profile_masks_reject_unsupported_dtype():
    with pytest.raises(TypeError, match="dtype"):
        select_segmentation_profile(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.uint8),
        )
