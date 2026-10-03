"""Step 5: Convert pixel measurements to millimetres.

The calibration scale is defined as millimetres per pixel. It can be supplied
directly or calculated from a known reference length measured in the image.
"""

from copy import deepcopy


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