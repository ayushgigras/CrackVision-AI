"""
CrackGauge - Step 3: Crack Width Measurement Module
====================================================
Estimates local crack width in pixels along the medial axis (skeleton).

Methodology:
    1. Skeletonization:
       Extracts a 1-pixel wide central skeleton (medial axis) of the crack mask.
    2. Distance Transform:
       Computes Euclidean distance from each crack pixel to the nearest background boundary.
    3. Local Width Estimation:
       Along each point on the skeleton, local width is calculated as:
           width_pixels = 2 * distance_to_boundary
    4. Statistical Metrics:
       Calculates Maximum, Average (Mean), and Median crack widths in pixels.
    5. Visualization:
       Generates a width map / heatmap along the skeleton with max width highlighted.

Note:
    Ground-truth calibration (pixels -> mm) is not applied at this stage.
    All measurements are strictly reported in PIXELS.

Author: CrackGauge Team (Manish, Sachin, Ayush)
"""

import os
import sys
import cv2
import numpy as np
from skimage.morphology import skeletonize

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_skeleton(crack_mask: np.ndarray) -> np.ndarray:
    """
    Extract a 1-pixel wide skeleton (medial axis) of the binary crack mask.
    
    Args:
        crack_mask: Binary mask where crack pixels are non-zero (uint8)
        
    Returns:
        Binary skeleton mask (uint8, values 0 or 255)
    """
    if crack_mask is None or np.sum(crack_mask > 0) == 0:
        print("[WARNING] Empty crack mask received for skeletonization.")
        return np.zeros_like(crack_mask if crack_mask is not None else np.zeros((1, 1), dtype=np.uint8))
        
    # skeletonize requires boolean input
    mask_bool = crack_mask > 0
    skeleton_bool = skeletonize(mask_bool)
    skeleton_uint8 = (skeleton_bool * 255).astype(np.uint8)
    
    skeleton_pts = int(np.sum(skeleton_bool))
    print(f"[OK] Skeletonization completed ({skeleton_pts:,} skeleton pixels)")
    return skeleton_uint8


def compute_distance_transform(crack_mask: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance transform (L2) of the crack mask.
    
    For every crack pixel, computes distance to the nearest boundary (0 pixel).
    Along the medial axis (skeleton), this distance equals the inscribed radius r.
    
    Args:
        crack_mask: Binary mask (uint8)
        
    Returns:
        Floating-point distance transform image (float32)
    """
    if crack_mask is None or np.sum(crack_mask > 0) == 0:
        return np.zeros((1, 1), dtype=np.float32) if crack_mask is None else np.zeros_like(crack_mask, dtype=np.float32)
        
    # Ensure binary uint8 with 0 and 255
    binary = np.zeros_like(crack_mask, dtype=np.uint8)
    binary[crack_mask > 0] = 255
    
    dist_transform = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=5)
    max_dist = float(dist_transform.max())
    print(f"[OK] Distance transform computed (max inscribed radius: {max_dist:.2f} px)")
    return dist_transform


def measure_crack_widths(crack_mask: np.ndarray,
                         skeleton: np.ndarray = None,
                         dist_transform: np.ndarray = None) -> dict:
    """
    Calculate crack widths in pixels along the skeleton.
    
    Formula:
        local_width = 2.0 * distance_transform[skeleton_point]
        
    Args:
        crack_mask:     Binary crack mask (uint8)
        skeleton:       Optional precomputed skeleton (uint8)
        dist_transform: Optional precomputed distance transform (float32)
        
    Returns:
        dict containing:
            'max_width_px':      float, maximum width in pixels
            'mean_width_px':     float, average width in pixels
            'median_width_px':   float, median width in pixels
            'std_width_px':      float, standard deviation of width
            'skeleton_points':   int, number of skeleton points sampled
            'max_width_coords':  tuple (y, x) location of maximum width point
            'width_map':         np.ndarray (float32), 2D map of local widths on skeleton
            'widths_array':      np.ndarray, 1D array of all measured widths
    """
    if crack_mask is None or np.sum(crack_mask > 0) == 0:
        print("[INFO] No crack pixels found. Width measurements set to 0.0 px.")
        empty_map = np.zeros((1, 1), dtype=np.float32) if crack_mask is None else np.zeros_like(crack_mask, dtype=np.float32)
        return {
            'max_width_px': 0.0,
            'mean_width_px': 0.0,
            'median_width_px': 0.0,
            'std_width_px': 0.0,
            'skeleton_points': 0,
            'max_width_coords': (0, 0),
            'width_map': empty_map,
            'widths_array': np.array([], dtype=np.float32),
        }
        
    if skeleton is None:
        skeleton = compute_skeleton(crack_mask)
        
    if dist_transform is None:
        dist_transform = compute_distance_transform(crack_mask)
        
    skeleton_mask = skeleton > 0
    skeleton_count = int(np.sum(skeleton_mask))
    
    if skeleton_count == 0:
        print("[INFO] Skeleton is empty. Width measurements set to 0.0 px.")
        return {
            'max_width_px': 0.0,
            'mean_width_px': 0.0,
            'median_width_px': 0.0,
            'std_width_px': 0.0,
            'skeleton_points': 0,
            'max_width_coords': (0, 0),
            'width_map': np.zeros_like(crack_mask, dtype=np.float32),
            'widths_array': np.array([], dtype=np.float32),
        }
        
    # Local width = 2 * radius (distance to boundary)
    width_map = np.zeros_like(dist_transform, dtype=np.float32)
    width_map[skeleton_mask] = 2.0 * dist_transform[skeleton_mask]
    
    widths_array = width_map[skeleton_mask]
    
    max_w = float(np.max(widths_array))
    mean_w = float(np.mean(widths_array))
    median_w = float(np.median(widths_array))
    std_w = float(np.std(widths_array))
    
    # Location of maximum width
    max_idx = np.argmax(widths_array)
    y_coords, x_coords = np.where(skeleton_mask)
    max_coords = (int(y_coords[max_idx]), int(x_coords[max_idx]))
    
    print(f"[OK] Width calculations complete:")
    print(f"     Max Width    : {max_w:.2f} px at (x={max_coords[1]}, y={max_coords[0]})")
    print(f"     Mean Width   : {mean_w:.2f} px")
    print(f"     Median Width : {median_w:.2f} px")
    
    return {
        'max_width_px': max_w,
        'mean_width_px': mean_w,
        'median_width_px': median_w,
        'std_width_px': std_w,
        'skeleton_points': skeleton_count,
        'max_width_coords': max_coords,
        'width_map': width_map,
        'widths_array': widths_array,
    }


def create_width_visualization_image(original_bgr: np.ndarray,
                                     skeleton: np.ndarray,
                                     width_map: np.ndarray,
                                     max_coords: tuple) -> np.ndarray:
    """
    Create a detailed visualization of width measurements:
    Draws the skeleton with line thickness / color according to local width,
    and marks the maximum width location with an indicator and text.
    """
    vis = original_bgr.copy()
    
    skeleton_mask = skeleton > 0
    if not np.any(skeleton_mask):
        return vis
        
    max_w = float(np.max(width_map))
    if max_w < 1e-6:
        return vis
        
    # Normalize widths to [0, 255] for colormap
    norm_widths = np.clip((width_map / max_w) * 255.0, 0, 255).astype(np.uint8)
    color_mapped = cv2.applyColorMap(norm_widths, cv2.COLORMAP_TURBO)
    
    # Dilate skeleton slightly for visibility in visualization
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_skeleton = cv2.dilate(skeleton, kernel, iterations=1) > 0
    
    # Dilate color map to match
    color_dilated = cv2.dilate(color_mapped, kernel, iterations=1)
    
    # Blend onto original image
    vis[dilated_skeleton] = color_dilated[dilated_skeleton]
    
    # Highlight max width location with marker
    my, mx = max_coords
    cv2.circle(vis, (mx, my), 9, (0, 0, 255), 2)  # Red outer ring
    cv2.circle(vis, (mx, my), 3, (0, 255, 255), -1)  # Yellow center dot
    
    # Annotation text
    label = f"Max: {max_w:.1f}px"
    # Position text so it stays inside frame
    tx = min(mx + 12, original_bgr.shape[1] - 110)
    ty = max(my - 10, 25)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    return vis


def run_measurement_pipeline(segmentation_results: dict,
                             output_dir: str = None,
                             save_steps: bool = True) -> dict:
    """
    Execute Step 3 Width Measurement Pipeline on segmentation results.
    
    Args:
        segmentation_results: Output dict from segmentation.py
        output_dir:           Directory to save output files
        save_steps:           Whether to save images to disk
        
    Returns:
        dict containing measurement metrics, skeleton, distance transform, and width image.
    """
    print("\n" + "="*60)
    print("  CrackGauge Step 3: Crack Width Measurement (Pixels)")
    print("="*60)
    
    original = segmentation_results['original']
    final_mask = segmentation_results['final_mask']
    
    # 1. Skeletonize
    skeleton = compute_skeleton(final_mask)
    
    # 2. Distance Transform
    dist_transform = compute_distance_transform(final_mask)
    
    # 3. Measure Widths
    measure_stats = measure_crack_widths(final_mask, skeleton, dist_transform)
    
    # 4. Width Heatmap / Visualization Image
    width_vis = create_width_visualization_image(
        original, skeleton, measure_stats['width_map'], measure_stats['max_width_coords']
    )
    
    results = {
        'original':          original,
        'final_mask':        final_mask,
        'skeleton':          skeleton,
        'dist_transform':    dist_transform,
        'width_map':         measure_stats['width_map'],
        'width_vis':         width_vis,
        'max_width_px':      measure_stats['max_width_px'],
        'mean_width_px':     measure_stats['mean_width_px'],
        'median_width_px':   measure_stats['median_width_px'],
        'std_width_px':      measure_stats['std_width_px'],
        'skeleton_points':   measure_stats['skeleton_points'],
        'max_width_coords':  measure_stats['max_width_coords'],
    }
    
    # Save outputs if requested
    if save_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = segmentation_results.get('image_name', 'crack_result')
        
        # Save skeleton
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_10_skeleton.jpg"), skeleton)
        
        # Normalize distance transform for visual file
        dist_max = float(dist_transform.max())
        if dist_max > 0:
            dist_vis = (dist_transform / dist_max * 255.0).astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_VIRIDIS)
        else:
            dist_color = np.zeros_like(original)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_11_distance_transform.jpg"), dist_color)
        
        # Save width visualization
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_12_width_measurement.jpg"), width_vis)
        print(f"[OK] Measurement step images saved to: {output_dir}")
        
    print(f"\n{'='*60}")
    print(f"  Width Measurement Summary (PIXELS):")
    print(f"  Max Width    : {measure_stats['max_width_px']:.2f} px")
    print(f"  Mean Width   : {measure_stats['mean_width_px']:.2f} px")
    print(f"  Median Width : {measure_stats['median_width_px']:.2f} px")
    print(f"  Sample Points: {measure_stats['skeleton_points']:,} px")
    print(f"{'='*60}\n")
    
    return results


def visualize_measurement(results: dict, save_path: str = None):
    """
    Create a 4-panel visual comparison:
      Panel 1: Original Concrete Image
      Panel 2: Final Segmented Crack Mask
      Panel 3: Crack Skeleton (Medial Axis)
      Panel 4: Width Measurement Map (with Max Width Callout)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("CrackGauge - Step 3: Crack Width Measurement (Medial Axis & Distance Transform)",
                 fontsize=15, fontweight='bold', y=0.98)
                 
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.20)
    
    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Original Image", fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')
    
    # Panel 2: Crack Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['final_mask'], cmap='gray')
    ax2.set_title("2. Final Crack Mask (Step 2)", fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')
    
    # Panel 3: Skeleton
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(results['skeleton'], cmap='hot')
    ax3.set_title(f"3. Crack Skeleton ({results['skeleton_points']:,} px)", fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Panel 4: Width measurement
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cv2.cvtColor(results['width_vis'], cv2.COLOR_BGR2RGB))
    max_w = results['max_width_px']
    mean_w = results['mean_width_px']
    med_w = results['median_width_px']
    ax4.set_title(f"4. Width Measurement Map [Max: {max_w:.2f}px | Mean: {mean_w:.2f}px | Med: {med_w:.2f}px]",
                  fontsize=11, fontweight='bold', color='#c0392b', pad=8)
    ax4.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[OK] Measurement visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()
