import numpy as np

from tools.visualize_validation import build_comparison_overlay


def test_comparison_overlay_colors_and_dimensions():
    predicted = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    ground_truth = np.array([[255, 0], [255, 0]], dtype=np.uint8)

    overlay = build_comparison_overlay(predicted, ground_truth)

    assert overlay.shape == (2, 2, 3)
    assert tuple(overlay[0, 0]) == (0, 255, 0)  # TP
    assert tuple(overlay[0, 1]) == (255, 0, 0)  # FP
    assert tuple(overlay[1, 0]) == (0, 0, 255)  # FN
    assert tuple(overlay[1, 1]) == (0, 0, 0)  # background
