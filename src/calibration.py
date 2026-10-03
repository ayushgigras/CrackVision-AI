"""Step 5: Convert pixel measurements to millimetres.

The calibration scale is defined as millimetres per pixel. It can be supplied
directly or calculated from a known reference length measured in the image.
"""

from copy import deepcopy

import cv2
import numpy as np


def calculate_mm_per_pixel(reference_length_mm: float,
                           reference_length_px: float) -> float:
    """Calculate a physical scale from a reference object in an image."""
    if reference_length_mm <= 0:
        raise ValueError("reference_length_mm must be greater than zero")
    if reference_length_px <= 0:
        raise ValueError("reference_length_px must be greater than zero")
    return float(reference_length_mm) / float(reference_length_px)


def calibrate_measurements(measurements: dict,
                            mm_per_pixel: float) -> dict:
    """Add millimetre measurements while preserving the original pixel values."""
    if mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be greater than zero")

    calibrated = deepcopy(measurements)
    pixel_fields = (
        'max_width_px',
        'mean_width_px',
        'median_width_px',
        'std_width_px',
        'total_length_px',
    )
    for field in pixel_fields:
        if field in calibrated:
            calibrated[field.replace('_px', '_mm')] = float(calibrated[field]) * mm_per_pixel

    calibrated['calibration'] = {
        'mm_per_pixel': float(mm_per_pixel),
        'pixels_per_mm': 1.0 / float(mm_per_pixel),
        'is_calibrated': True,
    }

    calibrated['components'] = [
        {
            **component,
            'length_mm': float(component['length_px']) * mm_per_pixel,
        }
        for component in calibrated.get('components', [])
    ]
    return calibrated


def _get_aruco_dictionary(dictionary_name: str):
    """Resolve an OpenCV predefined ArUco dictionary by name."""
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError("OpenCV ArUco support is unavailable; install opencv-contrib-python")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _marker_side_lengths(corners: np.ndarray) -> np.ndarray:
    """Return the four detected marker side lengths in pixels."""
    return np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)


def rectify_aruco_marker(image: np.ndarray,
                         corners: np.ndarray,
                         output_size: int = 600) -> np.ndarray:
    """Perspective-correct a detected marker into a square image."""
    destination = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, homography, (output_size, output_size))


def detect_aruco_calibration(image: np.ndarray,
                             marker_size_mm: float,
                             dictionary_name: str = 'DICT_4X4_50') -> dict:
    """Detect one ArUco marker and calculate its physical image scale.

    The marker's physical side length must be supplied by the caller. No scale
    is returned when detection fails.
    """
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be greater than zero")

    dictionary = _get_aruco_dictionary(dictionary_name)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    if ids is None or not corners:
        return {
            'is_calibrated': False,
            'status': 'marker_not_detected',
            'mm_per_pixel': None,
            'marker_id': None,
            'marker_corners': None,
            'rectified_marker': None,
        }

    marker_corners = np.asarray(corners[0], dtype=np.float32).reshape(4, 2)
    side_lengths_px = _marker_side_lengths(marker_corners)
    marker_width_px = float(np.mean(side_lengths_px))
    mm_per_pixel = calculate_mm_per_pixel(marker_size_mm, marker_width_px)

    return {
        'is_calibrated': True,
        'status': 'calibrated',
        'mm_per_pixel': mm_per_pixel,
        'marker_id': int(ids[0][0]),
        'marker_corners': marker_corners,
        'side_lengths_px': side_lengths_px,
        'marker_width_px': marker_width_px,
        'marker_size_mm': float(marker_size_mm),
        'dictionary': dictionary_name,
        'rectified_marker': rectify_aruco_marker(image, marker_corners),
    }


def get_calibration_scale(mm_per_pixel: float = None,
                          reference_length_mm: float = None,
                          reference_length_px: float = None) -> float:
    """Resolve direct or reference-based calibration CLI inputs."""
    if mm_per_pixel is not None:
        if reference_length_mm is not None or reference_length_px is not None:
            raise ValueError("Use either --mm-per-pixel or both reference lengths, not both")
        if mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be greater than zero")
        return float(mm_per_pixel)

    if (reference_length_mm is None) != (reference_length_px is None):
        raise ValueError("Both reference length values are required for calibration")
    if reference_length_mm is None:
        return None
    return calculate_mm_per_pixel(reference_length_mm, reference_length_px)