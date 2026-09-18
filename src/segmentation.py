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
                              save_steps: bool = True) -> dict:
    """
    Execute the full Step 2 Segmentation Pipeline on preprocessed results.
    
    Args:
        preprocessed_results: Dict output from preprocessing.py
        output_dir:           Directory to save output files
        sigmas:               Scale parameters for Frangi filter
        binarize_method:      'otsu' or 'cutoff'
        min_area:             Minimum component size in pixels
        fusion_mode:          'frangi_primary', 'intersection', or 'union'
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
    if adaptive_mask is not None:
        final_mask = fuse_masks(cleaned_ridge, adaptive_mask, mode=fusion_mode)
    else:
        final_mask = cleaned_ridge
        
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
