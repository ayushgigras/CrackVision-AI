"""CrackGauge In-Memory Pipeline Service (Step 1-5, no CLI dependency).

This module is the single reusable entry point for:
  - The FastAPI web server (src/api.py)
  - Future mobile/desktop integrations

It DOES NOT import or depend on main.py, argparse, or any CLI scaffolding.
All existing src/* modules are called with their real typed arguments so that
existing unit tests for those modules remain unaffected.

Usage example::

    import cv2
    from src.pipeline_service import PipelineConfig, analyze_image_array

    image = cv2.imread("data/sample_images/crack_synthetic_01.jpg")
    config = PipelineConfig(segmentation_profile="baseline")
    result = analyze_image_array(image, config)
    print(result["summary"])
    # result["visuals"]["overlay"] is a data: URL safe for an <img> tag
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

import cv2
import numpy as np

# Ensure project root is on sys.path when this module is imported directly.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.preprocessing import run_preprocessing_pipeline
from src.segmentation import SEGMENTATION_PROFILES, run_segmentation_pipeline
from src.measurement import (
    create_length_visualization_image,
    create_orientation_visualization_image,
    create_width_visualization_image,
    measure_crack_lengths,
    measure_crack_orientations,
    measure_crack_widths,
    compute_skeleton,
    compute_distance_transform,
    run_measurement_pipeline,
)
from src.calibration import (
    calibrate_measurements,
    detect_aruco_calibration,
    get_calibration_scale,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_PROFILES = tuple(SEGMENTATION_PROFILES)
_SUPPORTED_BIN_METHODS = ("otsu", "cutoff")
_SUPPORTED_FUSION_MODES = ("frangi_primary", "intersection", "union")
_SUPPORTED_ARUCO_DICTS = (
    "DICT_4X4_50",
    "DICT_5X5_50",
    "DICT_6X6_50",
    "DICT_7X7_50",
)

# Max long-edge (pixels) for preview images sent in JSON.
# Full-resolution images are used for all measurements.
_PREVIEW_MAX_EDGE = 1280


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Typed configuration for one crack-analysis run.

    Attributes
    ----------
    segmentation_profile:
        One of ``"baseline"``, ``"adaptive_recovery"``, or
        ``"guided_recovery"``.  Default is ``"baseline"``.
    bin_method:
        Ridge binarisation method: ``"otsu"`` or ``"cutoff"``.
    fusion:
        Mask fusion mode used only by the ``baseline`` profile.
    min_area:
        Minimum component pixel area for the Frangi filter step.
    gap_closing_ksize:
        Morphological closing kernel size for Step 4A length measurement.
        Set to 0 or 1 to disable.
    gap_closing_iters:
        Number of closing iterations.
    marker_size_mm:
        Physical side-length of the ArUco marker visible in the image (mm).
        When provided, automatic ArUco calibration is attempted.
    mm_per_pixel:
        Direct physical scale (mm/px).  Mutually exclusive with the
        *reference_length_* pair and with ``marker_size_mm``.
    reference_length_mm:
        Physical length of a known reference object (mm).  Must be paired
        with ``reference_length_px``.
    reference_length_px:
        Pixel length of the same reference object.
    aruco_dictionary:
        OpenCV ArUco dictionary name.  Default is ``"DICT_4X4_50"``.
    save_to_disk:
        When *True* and ``output_dir`` is set, intermediate step images are
        written to disk (mirrors the CLI behaviour).  Defaults to *False*.
    output_dir:
        Filesystem directory for saved artifacts; ignored when
        ``save_to_disk`` is *False*.
    """

    segmentation_profile: Literal[
        "baseline", "adaptive_recovery", "guided_recovery"
    ] = "baseline"
    bin_method: Literal["otsu", "cutoff"] = "otsu"
    fusion: Literal["frangi_primary", "intersection", "union"] = "frangi_primary"
    min_area: int = 80
    gap_closing_ksize: int = 7
    gap_closing_iters: int = 1
    marker_size_mm: Optional[float] = None
    mm_per_pixel: Optional[float] = None
    reference_length_mm: Optional[float] = None
    reference_length_px: Optional[float] = None
    aruco_dictionary: str = "DICT_4X4_50"
    save_to_disk: bool = False
    output_dir: Optional[str] = None

    def validate(self) -> None:
        """Raise ``ValueError`` for contradictory or out-of-range settings."""
        if self.segmentation_profile not in _SUPPORTED_PROFILES:
            raise ValueError(
                f"segmentation_profile must be one of {_SUPPORTED_PROFILES}; "
                f"got {self.segmentation_profile!r}"
            )
        if self.bin_method not in _SUPPORTED_BIN_METHODS:
            raise ValueError(
                f"bin_method must be one of {_SUPPORTED_BIN_METHODS}; "
                f"got {self.bin_method!r}"
            )
        if self.fusion not in _SUPPORTED_FUSION_MODES:
            raise ValueError(
                f"fusion must be one of {_SUPPORTED_FUSION_MODES}; "
                f"got {self.fusion!r}"
            )
        if self.min_area < 1:
            raise ValueError("min_area must be a positive integer")
        if self.gap_closing_ksize < 0:
            raise ValueError("gap_closing_ksize must be >= 0")
        if self.gap_closing_iters < 1:
            raise ValueError("gap_closing_iters must be >= 1")
        if self.aruco_dictionary not in _SUPPORTED_ARUCO_DICTS:
            raise ValueError(
                f"aruco_dictionary must be one of {_SUPPORTED_ARUCO_DICTS}; "
                f"got {self.aruco_dictionary!r}"
            )

        # Calibration mutual-exclusion rules (mirrors get_calibration_scale logic)
        n_calibration_sources = sum(
            x is not None
            for x in (
                self.marker_size_mm,
                self.mm_per_pixel,
                self.reference_length_mm,
            )
        )
        if n_calibration_sources > 1:
            raise ValueError(
                "Specify at most one calibration source: marker_size_mm, "
                "mm_per_pixel, or the (reference_length_mm + reference_length_px) pair."
            )
        if (self.reference_length_mm is None) != (self.reference_length_px is None):
            raise ValueError(
                "reference_length_mm and reference_length_px must both be "
                "provided together or both omitted."
            )
        if self.mm_per_pixel is not None and self.mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be positive")
        if self.marker_size_mm is not None and self.marker_size_mm <= 0:
            raise ValueError("marker_size_mm must be positive")
        if self.reference_length_mm is not None and self.reference_length_mm <= 0:
            raise ValueError("reference_length_mm must be positive")
        if self.reference_length_px is not None and self.reference_length_px <= 0:
            raise ValueError("reference_length_px must be positive")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Convert a NumPy scalar or Python number to a JSON-safe float.

    Returns *None* for NaN, Infinity, or non-numeric types.
    Never raises.
    """
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    """Convert a NumPy integer or Python int to a native Python int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bgr_to_png_dataurl(
    image_bgr: np.ndarray,
    max_edge: Optional[int] = None,
) -> str:
    """Encode a BGR NumPy array as a ``data:image/png;base64,…`` URL.

    Parameters
    ----------
    image_bgr:
        OpenCV BGR image array (uint8).
    max_edge:
        When given, down-scales the image so its longest edge equals this
        value (aspect-ratio preserved).  Used for preview thumbnails.

    Returns
    -------
    str
        RFC 2397 data URL; safe to assign directly to an ``<img src>``
        attribute in React.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Cannot encode empty or None image array")

    img = image_bgr.copy()
    if max_edge is not None:
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_edge:
            scale = max_edge / long_edge
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed: could not encode image as PNG")
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _gray_to_png_dataurl(
    image_gray: np.ndarray,
    max_edge: Optional[int] = None,
) -> str:
    """Encode a single-channel grayscale array as a PNG data URL."""
    if image_gray is None or image_gray.size == 0:
        raise ValueError("Cannot encode empty or None grayscale array")
    bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    return _bgr_to_png_dataurl(bgr, max_edge=max_edge)


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def analyze_image_array(
    image_bgr: np.ndarray,
    config: Optional[PipelineConfig] = None,
    *,
    image_name: str = "upload",
) -> dict:
    """Run the full CrackGauge pipeline on an in-memory BGR image.

    This function is the single entry point used by the API layer and tests.
    It does **not** read from or write to disk unless
    ``config.save_to_disk`` is *True* and ``config.output_dir`` is set.

    Parameters
    ----------
    image_bgr:
        Decoded BGR image as returned by ``cv2.imdecode`` or ``cv2.imread``.
        Must be a 3-channel, uint8 array.
    config:
        Pipeline configuration.  If *None*, all defaults apply.
    image_name:
        Human-readable identifier used in terminal progress prints and in the
        response ``image_name`` field.

    Returns
    -------
    dict
        A JSON-serializable dictionary.  All NumPy arrays have been replaced
        by base64 PNG data URLs.  All NumPy scalars have been cast to native
        Python types.  NaN and ±Infinity values are replaced with *None*.

        Top-level keys
        ~~~~~~~~~~~~~~
        * ``image_name``   – str
        * ``dimensions``   – {width: int, height: int}
        * ``summary``      – measurement + calibration scalars
        * ``components``   – per-crack-component list
        * ``visuals``      – base64 data URLs for all visual artifacts

    Raises
    ------
    ValueError
        For invalid images or contradictory configuration.
    RuntimeError
        If OpenCV cannot encode an output image.
    """
    if config is None:
        config = PipelineConfig()
    config.validate()

    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "image_bgr must be a 3-channel BGR uint8 NumPy array "
            f"(received shape {getattr(image_bgr, 'shape', None)})"
        )
    if image_bgr.dtype != np.uint8:
        raise ValueError(
            f"image_bgr must have dtype uint8; got {image_bgr.dtype}"
        )

    h, w = image_bgr.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(
            f"Image is too small for analysis ({w}×{h} px). "
            "Minimum supported size is 16×16 px."
        )

    # Resolve disk-saving options
    output_dir = config.output_dir if config.save_to_disk else None

    # ------------------------------------------------------------------
    # Steps 1-4: Preprocessing → Segmentation → Measurement
    # ------------------------------------------------------------------
    # We cannot call run_preprocessing_pipeline directly because it reads a
    # file from disk.  We replicate what it does internally without saving to
    # disk, using the same sub-functions it calls.
    # ------------------------------------------------------------------
    from src.preprocessing import (
        to_grayscale,
        reduce_noise,
        enhance_contrast,
        adaptive_threshold,
        morphological_cleanup,
    )

    # Step 1 sub-steps (in-memory)
    gray = to_grayscale(image_bgr)
    blurred = reduce_noise(gray, ksize=9)
    enhanced = enhance_contrast(blurred, clip_limit=3.0)
    binary = adaptive_threshold(enhanced, block_size=21, C=5)
    crack_mask_prep = morphological_cleanup(binary)

    prep_results = {
        "image_name": image_name,
        "image_path": "",          # no real path in API context
        "original": image_bgr,
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "binary": binary,
        "crack_mask": crack_mask_prep,
    }

    # Step 2: Segmentation
    seg_results = run_segmentation_pipeline(
        preprocessed_results=prep_results,
        output_dir=output_dir,
        sigmas=(1, 2, 3, 4),
        binarize_method=config.bin_method,
        min_area=config.min_area,
        fusion_mode=config.fusion,
        segmentation_profile=config.segmentation_profile,
        save_steps=config.save_to_disk,
    )

    # Steps 3, 4A, 4B: Width, Length, Orientation
    meas_results = run_measurement_pipeline(
        segmentation_results=seg_results,
        output_dir=output_dir,
        gap_closing_ksize=config.gap_closing_ksize,
        gap_closing_iters=config.gap_closing_iters,
        enable_gap_closing=(config.gap_closing_ksize > 1),
        save_steps=config.save_to_disk,
    )

    # ------------------------------------------------------------------
    # Step 5: Calibration
    # ------------------------------------------------------------------
    aruco_result = {
        "is_calibrated": False,
        "status": "marker_not_configured",
        "mm_per_pixel": None,
        "marker_id": None,
        "marker_size_mm": None,
        "rectified_marker_data_url": None,
    }

    calibration_scale: Optional[float] = None

    if config.marker_size_mm is not None:
        # Automatic ArUco calibration
        raw_aruco = detect_aruco_calibration(
            image_bgr,
            marker_size_mm=config.marker_size_mm,
            dictionary_name=config.aruco_dictionary,
        )
        aruco_result = {
            "is_calibrated": raw_aruco.get("is_calibrated", False),
            "status": raw_aruco.get("status", "unknown"),
            "mm_per_pixel": _safe_float(raw_aruco.get("mm_per_pixel")),
            "marker_id": _safe_int(raw_aruco.get("marker_id")),
            "marker_size_mm": _safe_float(config.marker_size_mm),
            "rectified_marker_data_url": None,
        }
        if raw_aruco.get("is_calibrated") and raw_aruco.get("rectified_marker") is not None:
            aruco_result["rectified_marker_data_url"] = _gray_to_png_dataurl(
                raw_aruco["rectified_marker"], max_edge=_PREVIEW_MAX_EDGE
            )
        calibration_scale = aruco_result["mm_per_pixel"]
    else:
        # Manual / reference-based calibration
        calibration_scale = get_calibration_scale(
            mm_per_pixel=config.mm_per_pixel,
            reference_length_mm=config.reference_length_mm,
            reference_length_px=config.reference_length_px,
        )
        if calibration_scale is not None:
            aruco_result = {
                "is_calibrated": True,
                "status": "manual_scale",
                "mm_per_pixel": _safe_float(calibration_scale),
                "marker_id": None,
                "marker_size_mm": None,
                "rectified_marker_data_url": None,
            }

    if calibration_scale is not None:
        meas_results = calibrate_measurements(meas_results, calibration_scale)
        if config.marker_size_mm is not None:
            # Merge ArUco detection metadata into the calibration sub-dict
            meas_results.setdefault("calibration", {}).update(aruco_result)
        else:
            meas_results.setdefault("calibration", {})["status"] = "manual_scale"

    # ------------------------------------------------------------------
    # Build JSON-safe response
    # ------------------------------------------------------------------
    calibration_sub = meas_results.get("calibration", aruco_result)

    calibration_out = {
        "is_calibrated": bool(calibration_sub.get("is_calibrated", False)),
        "status": str(calibration_sub.get("status", "not_calibrated")),
        "mm_per_pixel": _safe_float(calibration_sub.get("mm_per_pixel")),
        "marker_id": _safe_int(calibration_sub.get("marker_id")),
        "marker_size_mm": _safe_float(calibration_sub.get("marker_size_mm")),
    }

    # Per-component list
    # Length pipeline: components have keys id, length_px, skeleton_pixels,
    #                  mask_area, centroid, bbox
    # calibrate_measurements adds length_mm when calibrated.
    components_out = []
    for comp in meas_results.get("components", []):
        comp_dict: dict = {
            "id": _safe_int(comp.get("id")),
            "length_px": _safe_float(comp.get("length_px")),
            "skeleton_pixels": _safe_int(comp.get("skeleton_pixels")),
            "mask_area_px": _safe_int(comp.get("mask_area")),
            "centroid_x": _safe_float(comp["centroid"][0]) if comp.get("centroid") else None,
            "centroid_y": _safe_float(comp["centroid"][1]) if comp.get("centroid") else None,
            "bbox_x": _safe_int(comp["bbox"][0]) if comp.get("bbox") else None,
            "bbox_y": _safe_int(comp["bbox"][1]) if comp.get("bbox") else None,
            "bbox_w": _safe_int(comp["bbox"][2]) if comp.get("bbox") else None,
            "bbox_h": _safe_int(comp["bbox"][3]) if comp.get("bbox") else None,
        }
        if "length_mm" in comp:
            comp_dict["length_mm"] = _safe_float(comp["length_mm"])
        components_out.append(comp_dict)

    summary: dict = {
        # Segmentation
        "crack_coverage_percent": _safe_float(seg_results.get("crack_percent")),
        "crack_pixels": _safe_int(seg_results.get("crack_pixels")),
        # Width (Step 3)
        "max_width_px": _safe_float(meas_results.get("max_width_px")),
        "mean_width_px": _safe_float(meas_results.get("mean_width_px")),
        "median_width_px": _safe_float(meas_results.get("median_width_px")),
        "std_width_px": _safe_float(meas_results.get("std_width_px")),
        "skeleton_points": _safe_int(meas_results.get("skeleton_points")),
        # Length (Step 4A)
        "total_length_px": _safe_float(meas_results.get("total_length_px")),
        "component_count": _safe_int(meas_results.get("component_count")),
        "raw_component_count": _safe_int(meas_results.get("raw_component_count")),
        # Orientation (Step 4B)
        "dominant_angle_deg": _safe_float(meas_results.get("dominant_angle_deg")),
        "dominant_type": str(meas_results.get("dominant_type", "Undetermined")),
        # Calibration: mm values only present when calibrated
        "calibration": calibration_out,
    }

    # Add _mm fields only when they actually exist in meas_results
    for px_key in ("max_width", "mean_width", "median_width", "std_width", "total_length"):
        mm_key = f"{px_key}_mm"
        if mm_key in meas_results:
            summary[mm_key] = _safe_float(meas_results[mm_key])

    # ------------------------------------------------------------------
    # Visual artifacts: encode as PNG data URLs
    # All visualizations are produced at the original image resolution.
    # Previews sent via JSON are scaled to ≤ PREVIEW_MAX_EDGE on their
    # longest side so payload size stays manageable.
    # ------------------------------------------------------------------
    overlay_bgr = seg_results.get("overlay")          # BGR, full resolution
    final_mask = seg_results.get("final_mask")         # grayscale
    width_vis_bgr = meas_results.get("width_vis")     # BGR heatmap
    length_vis_bgr = meas_results.get("length_vis")   # BGR annotated
    orient_vis_bgr = meas_results.get("orientation_vis")  # BGR PCA

    visuals: dict = {
        "original": _bgr_to_png_dataurl(image_bgr, max_edge=_PREVIEW_MAX_EDGE),
        "crack_mask": _gray_to_png_dataurl(final_mask, max_edge=_PREVIEW_MAX_EDGE),
        "overlay": _bgr_to_png_dataurl(overlay_bgr, max_edge=_PREVIEW_MAX_EDGE),
        "width_heatmap": _bgr_to_png_dataurl(width_vis_bgr, max_edge=_PREVIEW_MAX_EDGE),
        "length_map": _bgr_to_png_dataurl(length_vis_bgr, max_edge=_PREVIEW_MAX_EDGE),
        "orientation_map": _bgr_to_png_dataurl(orient_vis_bgr, max_edge=_PREVIEW_MAX_EDGE),
        "rectified_marker": aruco_result.get("rectified_marker_data_url"),
    }

    return {
        "image_name": image_name,
        "dimensions": {"width": w, "height": h},
        "summary": summary,
        "components": components_out,
        "visuals": visuals,
    }
