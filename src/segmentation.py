"""
CrackGauge - Step 2: Crack Segmentation Module
================================================
Implements Hessian Matrix / Frangi Ridge Filtering for accurate crack segmentation.

Pipeline:
    Preprocessed Image (from Step 1)
        -> Frangi (Hessian Matrix) Ridge Filter (detects continuous crack lines)
        -> Ridge Response Normalization
        -> Ridge Binarization (Otsu / Adaptive Cutoff)
        -> Fusion with Preprocessing Mask (optional / configurable)
        -> Morphological Component Cleanup (area filtering)
        -> Crack Overlay Generation (Red highlight on original image)

Author: CrackGauge Team (Manish, Sachin, Ayush)
"""

import os
import sys
import cv2
import numpy as np
from skimage.filters import frangi

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def enhance_ridges_frangi(image_gray: np.ndarray,
                           sigmas: tuple = (1, 2, 3, 4),
                           black_ridges: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Frangi (Hessian matrix eigenvalue) filter to enhance continuous crack lines.
    
    Why Frangi Filter?
        Standard thresholding treats dark surface pits or shadows as cracks.
        The Frangi filter examines the 2nd derivatives (Hessian matrix) at multiple
        scales (sigmas) to detect elongated, ridge-like tubular structures (cracks).
        
    Args:
        image_gray:   Grayscale or CLAHE-enhanced image (uint8)
        sigmas:       Scales of crack widths to detect (e.g., (1, 2, 3, 4))
        black_ridges: True if cracks are darker than background (concrete cracks)
        
    Returns:
        (ridge_float, ridge_norm_uint8):
            ridge_float:      Raw filter response in range [0, 1]
            ridge_norm_uint8: Normalized 8-bit image [0, 255]
    """
    # Normalize input to [0, 1] float for skimage
    img_float = image_gray.astype(np.float32) / 255.0
    
    # Run Frangi filter
    ridge_float = frangi(img_float, sigmas=sigmas, black_ridges=black_ridges)
    
    # Normalize to 0-255 uint8
    r_min, r_max = float(ridge_float.min()), float(ridge_float.max())
    if r_max - r_min > 1e-8:
        ridge_norm = ((ridge_float - r_min) / (r_max - r_min) * 255.0).astype(np.uint8)
    else:
        ridge_norm = np.zeros_like(image_gray, dtype=np.uint8)
        
    print(f"[OK] Frangi ridge filter computed (sigmas={sigmas}, max_response={r_max:.4f})")
    return ridge_float, ridge_norm


def binarize_ridge(ridge_norm: np.ndarray,
                   method: str = "otsu",
                   cutoff_factor: float = 2.0) -> np.ndarray:
    """
    Convert the continuous ridge response to a binary crack mask.
    
    Args:
        ridge_norm:    Normalized 8-bit ridge response
        method:        'otsu' or 'cutoff' (mean + cutoff_factor * std)
        cutoff_factor: Multiplier for standard deviation when method='cutoff'
        
    Returns:
        Binary mask (uint8) with cracks as 255 (white) and background as 0 (black).
    """
    if method == "otsu":
        thresh_val, binary = cv2.threshold(
            ridge_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print(f"[OK] Ridge binarization (Otsu threshold={thresh_val:.1f})")
    elif method == "cutoff":
        mean_val = float(np.mean(ridge_norm))
        std_val = float(np.std(ridge_norm))
        thresh_val = int(np.clip(mean_val + cutoff_factor * std_val, 15, 80))
        _, binary = cv2.threshold(ridge_norm, thresh_val, 255, cv2.THRESH_BINARY)
        print(f"[OK] Ridge binarization (Cutoff threshold={thresh_val})")
    else:
        raise ValueError(f"Unknown binarization method: {method}")
        
    return binary

def binarize_ridge_hysteresis(
    ridge_norm: np.ndarray,
    low_thresh: int = 25,
    high_thresh: int = 48,
) -> np.ndarray:
    """Dual-threshold hysteresis binarization of the normalized Frangi response.

    Pixels with ridge_norm >= high_thresh are marked as *strong* seeds.
    Pixels with low_thresh <= ridge_norm < high_thresh are *weak* candidates.
    Weak pixels are accepted only when they are 8-connected to at least one
    strong pixel (directly or through other accepted weak pixels).
    All other pixels are background.

    This avoids the Otsu single-threshold problem where an aggressive cutoff
    discards faint-but-connected crack ridge pixels while a permissive cutoff
    admits broad concrete texture.

    Args:
        ridge_norm:  Normalized 8-bit Frangi response (uint8, 0–255).
        low_thresh:  Lower hysteresis threshold (inclusive).
        high_thresh: Upper hysteresis threshold (inclusive).

    Returns:
        Binary mask (uint8) with accepted crack pixels as 255.

    Raises:
        ValueError: If thresholds are out of range or low >= high.
    """
    if not (0 <= low_thresh < high_thresh <= 255):
        raise ValueError(
            f"Require 0 <= low_thresh < high_thresh <= 255; "
            f"got low_thresh={low_thresh}, high_thresh={high_thresh}"
        )

    strong = ridge_norm >= high_thresh
    weak = (ridge_norm >= low_thresh) & ~strong

    # Label connected components across the union of strong + weak pixels.
    candidate = (strong | weak).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate, connectivity=8)

    # Accept only those components that contain at least one strong pixel.
    strong_labels = set(np.unique(labels[strong])) - {0}
    result = np.zeros_like(ridge_norm, dtype=np.uint8)
    for lbl in strong_labels:
        result[labels == lbl] = 255

    accepted = int(np.sum(result > 0))
    total_strong = int(np.sum(strong))
    total_weak = int(np.sum(weak))
    print(
        f"[OK] Hysteresis binarization (low={low_thresh}, high={high_thresh}): "
        f"{total_strong} strong, {total_weak} weak, {accepted} accepted"
    )
    return result


def filter_crack_components(binary_mask: np.ndarray,
                            min_area: int = 80,
                            connectivity: int = 8) -> tuple[np.ndarray, int]:
    """
    Remove isolated noise components smaller than min_area pixels.
    
    Args:
        binary_mask:  Input binary image (0 or 255)
        min_area:     Minimum pixel area for a valid crack component
        connectivity: Pixel connectivity (4 or 8)
        
    Returns:
        (cleaned_mask, kept_count): Cleaned binary mask and count of kept components
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity
    )
    
    cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
    kept = 0
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255
            kept += 1
            
    print(f"[OK] Component filtering: {num_labels - 1} found, {kept} kept (area >= {min_area}px)")
    return cleaned, kept


def filter_crack_components_by_shape(
    binary_mask: np.ndarray,
    min_area: int = 80,
    large_area_thresh: int = 200,
    interior_aspect_thresh: float = 2.2,
    boundary_aspect_thresh: float = 1.6,
    min_skeleton_length: int = 20,
    connectivity: int = 8,
) -> tuple[np.ndarray, int]:
    """Filter binary components by area and elongation with boundary-aware handling.

    - Components with area < min_area are discarded as micro-noise.
    - Components with area >= large_area_thresh are unconditionally preserved
      as primary crack structures.
    - Intermediate components (min_area <= area < large_area_thresh) are evaluated
      for crack-like elongation via oriented minimum-area bounding box aspect ratio
      and medial axis skeleton length.
    - For boundary-touching components (which are physically truncated by the image frame),
      the aspect ratio threshold is relaxed to boundary_aspect_thresh.

    Args:
        binary_mask: Binary input mask (0 or 255)
        min_area: Minimum area to be considered (default: 80)
        large_area_thresh: Area at or above which components are always kept (default: 200)
        interior_aspect_thresh: Minimum oriented aspect ratio for interior components (default: 2.2)
        boundary_aspect_thresh: Minimum oriented aspect ratio for boundary components (default: 1.6)
        min_skeleton_length: Minimum medial axis skeleton pixels (default: 20)
        connectivity: Pixel connectivity (4 or 8)

    Returns:
        (cleaned_mask, kept_count): Filtered binary mask and count of kept components
    """
    if binary_mask is None or np.sum(binary_mask > 0) == 0:
        empty = np.zeros((1, 1), dtype=np.uint8) if binary_mask is None else np.zeros_like(binary_mask)
        return empty, 0

    from skimage.morphology import skeletonize

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=connectivity
    )
    h_img, w_img = binary_mask.shape[:2]

    cleaned = np.zeros_like(binary_mask, dtype=np.uint8)
    kept = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Tier 1: Large components kept unconditionally
        if area >= large_area_thresh:
            cleaned[labels == label] = 255
            kept += 1
            continue

        # Tier 2: Intermediate components (min_area <= area < large_area_thresh)
        touches_boundary = (x == 0) or (y == 0) or (x + w >= w_img) or (y + h >= h_img)
        comp_mask = (labels == label).astype(np.uint8) * 255

        # Oriented bounding box aspect ratio (rotation-invariant)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0 and len(contours[0]) >= 3:
            rect = cv2.minAreaRect(contours[0])
            rw, rh = rect[1]
            oriented_aspect = max(rw, rh) / max(1e-3, min(rw, rh))
        else:
            oriented_aspect = max(w, h) / max(1, min(w, h))

        # Medial axis skeleton length
        skel = skeletonize(comp_mask > 0)
        skel_len = int(np.sum(skel))

        req_aspect = boundary_aspect_thresh if touches_boundary else interior_aspect_thresh
        if oriented_aspect >= req_aspect and skel_len >= min_skeleton_length:
            cleaned[labels == label] = 255
            kept += 1

    print(f"[OK] Shape-aware component filtering: {num_labels - 1} found, {kept} kept")
    return cleaned, kept


def fuse_masks(ridge_mask: np.ndarray,
               adaptive_mask: np.ndarray,
               mode: str = "frangi_primary") -> np.ndarray:
    """
    Combine Frangi ridge mask with the adaptive threshold mask from Step 1.
    
    Modes:
        'frangi_primary': Uses Frangi mask directly (cleanest, minimal noise)
        'intersection':   Bitwise AND (only pixels detected by both methods)
        'union':          Bitwise OR (combines detections from both methods)
    """
    if mode == "frangi_primary":
        return ridge_mask
    elif mode == "intersection":
        fused = cv2.bitwise_and(ridge_mask, adaptive_mask)
        print("[OK] Fused mask via Intersection (AND)")
        return fused
    elif mode == "union":
        fused = cv2.bitwise_or(ridge_mask, adaptive_mask)
        print("[OK] Fused mask via Union (OR)")
        return fused
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


SEGMENTATION_PROFILES = ("baseline", "adaptive_recovery", "guided_recovery", "hysteresis_recovery")


def select_segmentation_profile(
    ridge_mask: np.ndarray,
    adaptive_mask: np.ndarray | None,
    profile: str = "baseline",
    fusion_mode: str = "frangi_primary",
    guided_kernel_size: int = 5,
    *,
    ridge_norm: np.ndarray | None = None,
    hysteresis_low: int = 25,
    hysteresis_high: int = 48,
    hysteresis_min_area: int = 80,
    hysteresis_use_shape_filter: bool = False,
    hysteresis_large_area_thresh: int = 200,
    hysteresis_interior_aspect_thresh: float = 2.2,
    hysteresis_boundary_aspect_thresh: float = 1.6,
    hysteresis_min_skeleton_length: int = 20,
) -> np.ndarray:
    """Select a final mask while preserving the existing baseline path.

    ``baseline`` delegates to the existing fusion behavior. ``adaptive_recovery``
    uses the cleaned adaptive mask. ``guided_recovery`` keeps the ridge mask and
    adds only adaptive components that overlap a small dilation of a ridge seed.
    ``hysteresis_recovery`` applies dual-threshold hysteresis on the normalized
    Frangi response (``ridge_norm``), recovering weak-ridge pixels that are
    8-connected to strong-ridge seeds without admitting disconnected texture.
    """
    if profile not in SEGMENTATION_PROFILES:
        raise ValueError(
            f"Unknown segmentation profile: {profile}. "
            f"Expected one of: {', '.join(SEGMENTATION_PROFILES)}"
        )
    _validate_profile_mask(ridge_mask, "ridge_mask")
    if adaptive_mask is not None:
        _validate_profile_mask(adaptive_mask, "adaptive_mask")
        if adaptive_mask.shape != ridge_mask.shape:
            raise ValueError(
                "ridge_mask and adaptive_mask must have matching dimensions"
            )
    if profile in ("adaptive_recovery", "guided_recovery") and adaptive_mask is None:
        raise ValueError(
            f"Segmentation profile '{profile}' requires an adaptive mask"
        )
    if profile == "hysteresis_recovery" and ridge_norm is None:
        raise ValueError(
            "Segmentation profile 'hysteresis_recovery' requires ridge_norm "
            "(the normalized uint8 Frangi response)"
        )

    if profile == "baseline":
        return fuse_masks(ridge_mask, adaptive_mask, mode=fusion_mode) if adaptive_mask is not None else ridge_mask
    if profile == "adaptive_recovery":
        return np.where(adaptive_mask > 0, 255, 0).astype(np.uint8)

    if profile == "hysteresis_recovery":
        if hysteresis_min_area < 0:
            raise ValueError(
                f"hysteresis_min_area must be non-negative, got {hysteresis_min_area}"
            )
        hysteresis_mask = binarize_ridge_hysteresis(
            ridge_norm, low_thresh=hysteresis_low, high_thresh=hysteresis_high
        )
        if hysteresis_use_shape_filter:
            cleaned, _ = filter_crack_components_by_shape(
                hysteresis_mask,
                min_area=hysteresis_min_area,
                large_area_thresh=hysteresis_large_area_thresh,
                interior_aspect_thresh=hysteresis_interior_aspect_thresh,
                boundary_aspect_thresh=hysteresis_boundary_aspect_thresh,
                min_skeleton_length=hysteresis_min_skeleton_length,
            )
        else:
            # Apply component filtering (default min_area=80)
            cleaned, _ = filter_crack_components(
                hysteresis_mask, min_area=hysteresis_min_area
            )
        return cleaned

    # guided_recovery
    if guided_kernel_size < 1 or guided_kernel_size % 2 == 0:
        raise ValueError("guided_kernel_size must be a positive odd integer")
    ridge_binary = ridge_mask > 0
    adaptive_binary = adaptive_mask > 0
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (guided_kernel_size, guided_kernel_size)
    )
    ridge_seed = cv2.dilate(ridge_binary.astype(np.uint8), kernel) > 0
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        adaptive_binary.astype(np.uint8), connectivity=8
    )
    recovered_adaptive = np.zeros_like(ridge_mask, dtype=np.uint8)
    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & ridge_seed):
            recovered_adaptive[component] = 255
    return np.where(ridge_binary | (recovered_adaptive > 0), 255, 0).astype(np.uint8)


def _validate_profile_mask(mask: np.ndarray, name: str) -> None:
    """Validate a binary mask supplied to a segmentation profile."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"{name} must be a 2D numpy array")
    if mask.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise TypeError(f"{name} must use bool or uint8 dtype")


def create_crack_overlay(original_bgr: np.ndarray,
                         crack_mask: np.ndarray,
                         color: tuple = (0, 0, 255),
                         alpha: float = 0.65) -> np.ndarray:
    """
    Overlay detected cracks on top of the original BGR image.
    
    Args:
        original_bgr: Original BGR image
        crack_mask:   Binary crack mask (cracks = 255)
        color:        BGR color for highlighting cracks (default: Red (0, 0, 255))
        alpha:        Transparency factor for the overlay (0.0 - 1.0)
        
    Returns:
        BGR image with red overlay on crack regions.
    """
    overlay = original_bgr.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(original_bgr)
    colored_mask[crack_mask > 0] = color
    
    # Alpha blend where crack is present
    crack_pixels = crack_mask > 0
    overlay[crack_pixels] = cv2.addWeighted(
        original_bgr, 1.0 - alpha, colored_mask, alpha, 0
    )[crack_pixels]
    
    # Add thin contour outline for sharp edge contrast
    contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)  # Yellow border
    
    print("[OK] Crack overlay generated (Red mask with Yellow contour)")
    return overlay


def run_segmentation_pipeline(preprocessed_results: dict,
                              output_dir: str = None,
                              sigmas: tuple = (1, 2, 3, 4),
                              binarize_method: str = "otsu",
                              min_area: int = 80,
                              fusion_mode: str = "frangi_primary",
                              segmentation_profile: str = "baseline",
                              save_steps: bool = True,
                              *,
                              hysteresis_low: int = 25,
                              hysteresis_high: int = 48,
                              hysteresis_min_area: int | None = None,
                              hysteresis_use_shape_filter: bool = False,
                              hysteresis_large_area_thresh: int = 200,
                              hysteresis_interior_aspect_thresh: float = 2.2,
                              hysteresis_boundary_aspect_thresh: float = 1.6,
                              hysteresis_min_skeleton_length: int = 20) -> dict:
    """
    Execute the full Step 2 Segmentation Pipeline on preprocessed results.
    
    Args:
        preprocessed_results: Dict output from preprocessing.py
        output_dir:           Directory to save output files
        sigmas:               Scale parameters for Frangi filter
        binarize_method:      'otsu' or 'cutoff'
        min_area:             Minimum component size in pixels
        fusion_mode:          'frangi_primary', 'intersection', or 'union'
        segmentation_profile: 'baseline', 'adaptive_recovery', or 'guided_recovery'
        save_steps:           Whether to write step images to disk
        
    Returns:
        dict containing ridge outputs, final mask, overlay, and metrics.
    """
    print("\n" + "="*60)
    print("  CrackGauge Step 2: Crack Segmentation (Hessian/Frangi)")
    print("="*60)
    
    original = preprocessed_results['original']
    enhanced = preprocessed_results['enhanced']
    adaptive_mask = preprocessed_results.get('crack_mask', None)
    
    # 1. Frangi Ridge Enhancement
    ridge_float, ridge_norm = enhance_ridges_frangi(
        enhanced, sigmas=sigmas, black_ridges=True
    )
    
    # 2. Binarization
    ridge_binary = binarize_ridge(ridge_norm, method=binarize_method)
    
    # 3. Component Filtering
    cleaned_ridge, kept_count = filter_crack_components(ridge_binary, min_area=min_area)
    
    # 4. Fusion (if adaptive_mask exists)
    final_mask = select_segmentation_profile(
        cleaned_ridge,
        adaptive_mask,
        profile=segmentation_profile,
        fusion_mode=fusion_mode,
        ridge_norm=ridge_norm,
        hysteresis_low=hysteresis_low,
        hysteresis_high=hysteresis_high,
        hysteresis_min_area=hysteresis_min_area if hysteresis_min_area is not None else min_area,
        hysteresis_use_shape_filter=hysteresis_use_shape_filter,
        hysteresis_large_area_thresh=hysteresis_large_area_thresh,
        hysteresis_interior_aspect_thresh=hysteresis_interior_aspect_thresh,
        hysteresis_boundary_aspect_thresh=hysteresis_boundary_aspect_thresh,
        hysteresis_min_skeleton_length=hysteresis_min_skeleton_length,
    )
        
    # 5. Overlay
    overlay = create_crack_overlay(original, final_mask, color=(0, 0, 255), alpha=0.65)
    
    # Metrics
    crack_pixels = int(np.sum(final_mask > 0))
    total_pixels = int(final_mask.shape[0] * final_mask.shape[1])
    crack_percent = (crack_pixels / total_pixels) * 100.0
    
    base_name = preprocessed_results.get('image_name', 'crack_result')
    
    results = {
        'image_name':      base_name,
        'original':        original,
        'enhanced':        enhanced,
        'adaptive_mask':   adaptive_mask,
        'ridge_float':     ridge_float,
        'ridge_norm':      ridge_norm,
        'ridge_binary':    ridge_binary,
        'final_mask':      final_mask,
        'overlay':         overlay,
        'kept_components': kept_count,
        'crack_pixels':    crack_pixels,
        'crack_percent':   crack_percent,
    }
    
    # Save step images if requested
    if save_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = preprocessed_results.get('image_name', 'crack_result')
            
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_06_ridge_norm.jpg"), ridge_norm)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_07_ridge_binary.jpg"), ridge_binary)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_08_final_crack_mask.jpg"), final_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_09_crack_overlay.jpg"), overlay)
        print(f"[OK] Segmentation step images saved to: {output_dir}")
        
    print(f"\n{'='*60}")
    print(f"  Segmentation Summary:")
    print(f"  Crack Components : {kept_count}")
    print(f"  Crack Pixels     : {crack_pixels:,} px")
    print(f"  Crack Coverage   : {crack_percent:.2f}%")
    print(f"{'='*60}\n")
    
    return results


def visualize_segmentation(results: dict, save_path: str = None):
    """
    Create a 6-panel visualization comparing Preprocessing, Frangi Ridge,
    Final Binary Mask, and Overlay.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("CrackGauge - Step 2: Crack Segmentation & Ridge Enhancement",
                 fontsize=16, fontweight='bold', y=0.98)
                 
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    plots = [
        ("1. Original Image",                 cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB), None),
        ("2. Step 1: CLAHE Enhanced",        results['enhanced'],                                   'gray'),
        ("3. Frangi Ridge Filter Response",  results['ridge_norm'],                                 'inferno'),
        ("4. Ridge Binary Mask",             results['ridge_binary'],                               'gray'),
        ("5. Final Cleaned Crack Mask",      results['final_mask'],                                 'gray'),
        ("6. Crack Overlay on Concrete",     cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB),   None),
    ]
    
    axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
    
    for ax, (title, img, cmap) in zip(axes, plots):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
    # Highlight final overlay panel
    axes[5].set_title("6. Crack Overlay on Concrete [DETECTED]",
                      fontsize=11, fontweight='bold', color='#c0392b', pad=8)
                      
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[OK] Segmentation visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()
