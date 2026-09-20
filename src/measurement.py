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
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        print("[WARNING] Empty crack mask received for skeletonization.")
        return np.zeros_like(crack_mask if crack_mask is not None else np.zeros((1, 1), dtype=np.uint8))
        
    # skeletonize requires boolean input
    mask_bool = crack_mask > 127
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
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        return np.zeros((1, 1), dtype=np.float32) if crack_mask is None else np.zeros_like(crack_mask, dtype=np.float32)
        
    # Ensure binary uint8 with 0 and 255
    binary = np.zeros_like(crack_mask, dtype=np.uint8)
    binary[crack_mask > 127] = 255
    
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


# ==============================================================================
# STEP 4A: CRACK LENGTH MEASUREMENT ONLY
# ==============================================================================

def measure_crack_lengths(crack_mask: np.ndarray,
                          skeleton: np.ndarray = None,
                          gap_closing_ksize: int = 7,
                          gap_closing_iters: int = 1,
                          enable_gap_closing: bool = True) -> dict:
    """
    Calculate total crack length and per-component crack lengths in pixels from the skeleton.
    
    Morphological Preprocessing (Gap-Closing):
        Before connected component labeling, applies a conservative morphological closing
        operation (ellipse kernel) to join artificial micro-gaps in the synthetic segmentation mask.
        The skeleton is then recomputed on the closed mask to provide a continuous medial path.
        
    Path Length Calculation:
        For each connected component, adjacent 8-connected skeleton pixels are traversed:
          - Orthogonal neighbor step: distance = 1.0 px
          - Diagonal neighbor step:   distance = sqrt(2) approx 1.414 px
        Undirected neighbor edges are summed to obtain the true path length in pixels.
        
    Args:
        crack_mask:          Binary crack mask (uint8)
        skeleton:            Optional precomputed 1-pixel wide skeleton (uint8)
        gap_closing_ksize:   Kernel size for closing (default: 7, ellipse). Set <= 1 to disable.
        gap_closing_iters:   Iterations of morphological closing (default: 1)
        enable_gap_closing:  Flag to enable/disable gap closing (default: True)
        
    Returns:
        dict containing:
            'total_length_px':      float, sum of all component lengths in pixels
            'component_count':      int, number of crack components after gap closing
            'raw_component_count':  int, number of crack components before gap closing
            'components':           list of dicts for each component
            'closed_mask':          np.ndarray, binary mask after gap closing
            'length_skeleton':      np.ndarray, recalculated skeleton on closed mask
            'gap_closing_ksize':    int
            'gap_closing_iters':    int
    """
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        print("[INFO] No crack pixels found for length measurement. Total length set to 0.0 px.")
        empty_mask = np.zeros((1, 1), dtype=np.uint8) if crack_mask is None else np.zeros_like(crack_mask)
        return {
            'total_length_px': 0.0,
            'component_count': 0,
            'raw_component_count': 0,
            'components': [],
            'closed_mask': empty_mask,
            'length_skeleton': empty_mask,
            'gap_closing_ksize': gap_closing_ksize if enable_gap_closing else 0,
            'gap_closing_iters': gap_closing_iters if enable_gap_closing else 0,
        }
        
    raw_mask_bin = (crack_mask > 127).astype(np.uint8)
    
    # Count raw components before gap closing
    num_raw_labels, _, _, _ = cv2.connectedComponentsWithStats(raw_mask_bin, connectivity=8)
    raw_component_count = max(0, num_raw_labels - 1)
    
    # Morphological gap-closing (conservative)
    if enable_gap_closing and gap_closing_ksize > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_closing_ksize, gap_closing_ksize))
        closed_mask = cv2.morphologyEx(raw_mask_bin * 255, cv2.MORPH_CLOSE, kernel, iterations=gap_closing_iters)
        # Recalculate skeleton on the closed mask
        recalculated_skeleton = compute_skeleton(closed_mask)
        eval_mask_bin = (closed_mask > 127).astype(np.uint8)
    else:
        closed_mask = crack_mask.copy()
        recalculated_skeleton = skeleton if skeleton is not None else compute_skeleton(crack_mask)
        eval_mask_bin = raw_mask_bin
        
    skel_bool = recalculated_skeleton > 0
    
    # Label connected components on the evaluated mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eval_mask_bin, connectivity=8)
    
    # 4 forward directions for undirected edge counting: (dy, dx, distance)
    fwd_dirs = [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 1, 1.41421356),
        (1, -1, 1.41421356),
    ]
    
    components = []
    total_length = 0.0
    comp_idx = 1
    
    for label in range(1, num_labels):
        comp_skel_mask = (labels == label) & skel_bool
        ys, xs = np.where(comp_skel_mask)
        skel_pts_count = len(ys)
        
        if skel_pts_count == 0:
            continue
            
        pts_set = set(zip(ys, xs))
        comp_len = 0.0
        edges_found = 0
        
        if skel_pts_count == 1:
            comp_len = 1.0
        else:
            for y, x in zip(ys, xs):
                for dy, dx, weight in fwd_dirs:
                    if (y + dy, x + dx) in pts_set:
                        comp_len += weight
                        edges_found += 1
            if edges_found == 0:
                comp_len = float(skel_pts_count)
                
        total_length += comp_len
        
        bx = int(stats[label, cv2.CC_STAT_LEFT])
        by = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[label][0]), float(centroids[label][1])
        
        components.append({
            'id': comp_idx,
            'length_px': comp_len,
            'skeleton_pixels': skel_pts_count,
            'mask_area': area,
            'centroid': (cx, cy),
            'bbox': (bx, by, bw, bh),
        })
        comp_idx += 1
        
    if enable_gap_closing and gap_closing_ksize > 1:
        print(f"[INFO] Step 4A Gap Closing: {raw_component_count} raw component(s) -> {len(components)} continuous component(s) (kernel={gap_closing_ksize}x{gap_closing_ksize}, iters={gap_closing_iters})")
    print(f"[OK] Length calculation completed: {len(components)} components | Total Length: {total_length:.2f} px")
    
    return {
        'total_length_px': total_length,
        'component_count': len(components),
        'raw_component_count': raw_component_count,
        'components': components,
        'closed_mask': closed_mask,
        'length_skeleton': recalculated_skeleton,
        'gap_closing_ksize': gap_closing_ksize if enable_gap_closing else 0,
        'gap_closing_iters': gap_closing_iters if enable_gap_closing else 0,
    }


def create_length_visualization_image(original_bgr: np.ndarray,
                                      crack_mask: np.ndarray,
                                      skeleton: np.ndarray,
                                      components: list,
                                      total_length_px: float,
                                      raw_component_count: int = None) -> np.ndarray:
    """
    Generate an annotated image highlighting each crack component's path in distinct colors
    with non-overlapping component ID and length callout badges, connected via leader lines.
    """
    vis = original_bgr.copy()
    img_h, img_w = vis.shape[:2]
    
    if not components or skeleton is None or np.sum(skeleton > 0) == 0:
        return vis
        
    # High-contrast color palette (BGR) for distinguishing components
    PALETTE = [
        (0, 255, 255),    # Yellow
        (0, 255, 0),      # Lime Green
        (255, 128, 0),    # Blue-Cyan / Sky
        (255, 0, 255),    # Magenta
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Violet
        (0, 215, 255),    # Gold
        (50, 205, 50),    # Forest Lime
        (255, 192, 203),  # Pink
        (220, 20, 60),    # Crimson
        (0, 128, 255),    # Amber
        (240, 230, 140),  # Khaki
    ]
    
    mask_bin = (crack_mask > 127).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    skel_bool = skeleton > 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # 1. Draw colored skeleton paths
    actual_idx = 1
    for label in range(1, num_labels):
        comp_skel_mask = (labels == label) & skel_bool
        if not np.any(comp_skel_mask):
            continue
            
        color = PALETTE[(actual_idx - 1) % len(PALETTE)]
        dilated = cv2.dilate(comp_skel_mask.astype(np.uint8), kernel, iterations=1) > 0
        vis[dilated] = color
        actual_idx += 1
        
    # 2. Collision-free badge placement with leader lines
    placed_boxes = []  # list of (x1, y1, x2, y2)
    
    candidate_offsets = [
        (14, -14),
        (14, 20),
        (-14, -14),
        (-14, 20),
        (0, -32),
        (0, 32),
        (35, -5),
        (-35, -5),
        (35, 25),
        (-35, 25),
        (50, -20),
        (-50, -20),
        (50, 40),
        (-50, 40),
        (0, -50),
        (0, 50),
        (70, 0),
        (-70, 0),
        (70, 35),
        (-70, 35),
    ]
    
    for c in components:
        idx = c['id']
        color = PALETTE[(idx - 1) % len(PALETTE)]
        cx, cy = c['centroid']
        clen = c['length_px']
        
        badge_text = f"#{idx}: {clen:.1f}px"
        font_scale = 0.42
        thickness = 1
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bw = tw + 10
        bh = th + 8
        
        best_box = None
        best_score = float('inf')
        
        for ox, oy in candidate_offsets:
            if ox < 0:
                bx = int(cx + ox - bw)
            elif ox == 0:
                bx = int(cx - bw // 2)
            else:
                bx = int(cx + ox)
                
            if oy < 0:
                by = int(cy + oy - bh)
            else:
                by = int(cy + oy)
                
            # Clamp inside image boundaries (reserve top y=45 for HUD banner)
            bx = max(10, min(bx, img_w - bw - 10))
            by = max(45, min(by, img_h - bh - 10))
            box = (bx, by, bx + bw, by + bh)
            
            # Check overlap area with already placed badges (plus 3px padding)
            overlap_area = 0
            for pbox in placed_boxes:
                ix1 = max(box[0] - 3, pbox[0] - 3)
                iy1 = max(box[1] - 3, pbox[1] - 3)
                ix2 = min(box[2] + 3, pbox[2] + 3)
                iy2 = min(box[3] + 3, pbox[3] + 3)
                if ix2 > ix1 and iy2 > iy1:
                    overlap_area += (ix2 - ix1) * (iy2 - iy1)
                    
            box_cx = bx + bw / 2.0
            box_cy = by + bh / 2.0
            dist = np.hypot(box_cx - cx, box_cy - cy)
            
            score = (overlap_area * 1000.0) + dist
            if score < best_score:
                best_score = score
                best_box = box
                
        if best_box is None:
            bx = max(10, min(int(cx), img_w - bw - 10))
            by = max(45, min(int(cy), img_h - bh - 10))
            best_box = (bx, by, bx + bw, by + bh)
            
        placed_boxes.append(best_box)
        bx, by, bx2, by2 = best_box
        
        # Leader line from centroid to badge anchor
        anchor_x = int(np.clip(cx, bx, bx2))
        anchor_y = int(np.clip(cy, by, by2))
        
        # Centroid marker dot
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (20, 20, 20), -1)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 2, color, -1)
        
        # Draw leader line if badge is offset from centroid
        if np.hypot(anchor_x - cx, anchor_y - cy) > 5:
            cv2.line(vis, (int(round(cx)), int(round(cy))), (anchor_x, anchor_y), (40, 40, 40), 2, cv2.LINE_AA)
            cv2.line(vis, (int(round(cx)), int(round(cy))), (anchor_x, anchor_y), color, 1, cv2.LINE_AA)
            
        # Draw badge box
        cv2.rectangle(vis, (bx, by), (bx2, by2), (20, 20, 20), -1)
        cv2.rectangle(vis, (bx, by), (bx2, by2), color, 1, cv2.LINE_AA)
        
        # Badge text
        tx = bx + 5
        ty = by + th + 4
        cv2.putText(vis, badge_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    # Top HUD banner with overall length summary
    if raw_component_count is not None and raw_component_count != len(components):
        hud_text = f"Total Crack Length: {total_length_px:.1f} px  |  Components: {len(components)} (Raw: {raw_component_count}, Gap-Closed)"
    else:
        hud_text = f"Total Crack Length: {total_length_px:.1f} px  |  Components: {len(components)}"
        
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.rectangle(vis, (10, 8), (10 + hw + 20, 8 + hh + 16), (20, 20, 20), -1)
    cv2.rectangle(vis, (10, 8), (10 + hw + 20, 8 + hh + 16), (0, 200, 255), 2)
    cv2.putText(vis, hud_text, (20, 8 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
    
    return vis


# ==============================================================================
# STEP 4B: CRACK ORIENTATION MEASUREMENT ONLY (PCA)
# ==============================================================================

def classify_orientation_angle(angle_deg: float) -> str:
    """
    Classify orientation angle according to structural civil engineering conventions.
    
    Angle is in [0, 180) degrees relative to horizontal (X-axis):
      - 0° ± 15° (or 180° ± 15°): Horizontal (e.g. beam flexural/tension cracks)
      - 90° ± 15°: Vertical (e.g. column compression / shrinkage cracks)
      - Otherwise: Diagonal / Oblique (e.g. shear cracks)
    """
    norm_angle = angle_deg % 180.0
    if norm_angle <= 15.0 or norm_angle >= 165.0:
        return "Horizontal"
    elif 75.0 <= norm_angle <= 105.0:
        return "Vertical"
    else:
        return "Diagonal / Oblique"


def compute_pca_orientation(points_xy: np.ndarray) -> dict:
    """
    Apply Principal Component Analysis (PCA) on 2D coordinates of crack pixels.
    
    Args:
        points_xy: Nx2 numpy array of (x, y) coordinates
        
    Returns:
        dict containing:
            'center': (cx, cy) mean coordinate
            'angle_deg': float, angle in [0, 180) degrees
            'orientation_type': str ('Horizontal', 'Vertical', 'Diagonal / Oblique')
            'eigenvector': 1D array of 2 floats (dx, dy)
            'eigenvalues': (lambda1, lambda2)
            'endpoints': (p1, p2) coordinates for plotting the principal axis line
    """
    if points_xy is None or len(points_xy) < 3:
        return {
            'center': (0.0, 0.0),
            'angle_deg': 0.0,
            'orientation_type': 'Undetermined',
            'eigenvector': np.array([1.0, 0.0], dtype=np.float32),
            'eigenvalues': (0.0, 0.0),
            'endpoints': ((0, 0), (0, 0)),
        }
        
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points_xy.astype(np.float32), mean=None)
    
    cx, cy = float(mean[0, 0]), float(mean[0, 1])
    center = (cx, cy)
    v1 = eigenvectors[0]  # principal eigenvector (dx, dy)
    
    # Angle in radians and degrees relative to horizontal X-axis
    angle_rad = np.arctan2(v1[1], v1[0])
    angle_deg = float(np.degrees(angle_rad) % 180.0)
    
    lam1 = float(eigenvalues[0, 0])
    lam2 = float(eigenvalues[1, 0]) if eigenvalues.shape[0] > 1 else 0.0
    
    orient_type = classify_orientation_angle(angle_deg)
    
    # Axis length proportional to 2 * sqrt(eigenvalue)
    axis_len = max(25.0, 2.0 * np.sqrt(max(0.0, lam1)))
    p1 = (int(round(cx - v1[0] * axis_len)), int(round(cy - v1[1] * axis_len)))
    p2 = (int(round(cx + v1[0] * axis_len)), int(round(cy + v1[1] * axis_len)))
    
    return {
        'center': center,
        'angle_deg': angle_deg,
        'orientation_type': orient_type,
        'eigenvector': v1,
        'eigenvalues': (lam1, lam2),
        'endpoints': (p1, p2),
    }


def measure_crack_orientations(crack_mask: np.ndarray,
                               min_points: int = 30) -> dict:
    """
    Measure orientation for each meaningful crack component using PCA.
    
    Args:
        crack_mask: Binary crack mask (uint8)
        min_points: Minimum component pixel count to be considered meaningful (default: 30)
        
    Returns:
        dict containing:
            'components': list of dicts for each evaluated component
            'dominant_angle_deg': float, overall PCA angle across all crack pixels
            'dominant_type': str, overall orientation classification
            'evaluated_component_count': int
    """
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        print("[INFO] No crack pixels found for orientation analysis.")
        return {
            'components': [],
            'dominant_angle_deg': 0.0,
            'dominant_type': 'Undetermined',
            'evaluated_component_count': 0,
        }
        
    mask_bin = (crack_mask > 127).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    
    components = []
    comp_idx = 1
    
    all_xs = []
    all_ys = []
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_points:
            continue
            
        ys, xs = np.where(labels == label)
        pts = np.column_stack((xs, ys))
        
        all_xs.extend(xs)
        all_ys.extend(ys)
        
        pca_res = compute_pca_orientation(pts)
        
        components.append({
            'id': comp_idx,
            'angle_deg': pca_res['angle_deg'],
            'orientation_type': pca_res['orientation_type'],
            'center': pca_res['center'],
            'endpoints': pca_res['endpoints'],
            'points_count': len(pts),
        })
        comp_idx += 1
        
    # Dominant orientation across all crack pixels (PCA on full crack mask)
    if len(all_xs) >= 3:
        all_pts = np.column_stack((all_xs, all_ys))
        overall_pca = compute_pca_orientation(all_pts)
        dom_angle = overall_pca['angle_deg']
        dom_type = overall_pca['orientation_type']
        dom_center = overall_pca['center']
        dom_eigenvector = overall_pca['eigenvector']
        dom_eigenvalues = overall_pca['eigenvalues']
    else:
        dom_angle = 0.0
        dom_type = 'Undetermined'
        dom_center = (0.0, 0.0)
        dom_eigenvector = np.array([1.0, 0.0], dtype=np.float32)
        dom_eigenvalues = (0.0, 0.0)

    # Debug: per-component summary (terminal only, not on visualization)
    print(f"[OK] PCA orientation analysis complete: {len(components)} meaningful components evaluated.")
    for c in components:
        print(f"     Comp #{c['id']:2d}: {c['angle_deg']:.1f} deg ({c['orientation_type']}) | {c['points_count']} px")
    print(f"     Dominant Orientation: {dom_angle:.1f} deg ({dom_type})")

    return {
        'components': components,
        'dominant_angle_deg': dom_angle,
        'dominant_type': dom_type,
        'dominant_center': dom_center,
        'dominant_eigenvector': dom_eigenvector,
        'dominant_eigenvalues': dom_eigenvalues,
        'evaluated_component_count': len(components),
    }


def create_orientation_visualization_image(original_bgr: np.ndarray,
                                           crack_mask: np.ndarray,
                                           skeleton: np.ndarray,
                                           orientations_data: dict) -> np.ndarray:
    """
    Generate a clean visualization of the DOMINANT PCA orientation only.

    Design:
    - Draws a subtle red overlay on all crack pixels.
    - Draws ONE prominent orientation axis representing the full-image dominant PCA result.
    - Axis runs across the entire image (not just the component bounding box) so it is
      readable at 800x600 resolution without any per-component clutter.
    - A single center-point marker at the PCA centroid.
    - Top HUD banner: dominant angle, structural type, and component count.
    - No per-component axes or badge labels are drawn on the image.
      (Per-component results are available in terminal output.)
    """
    vis = original_bgr.copy()
    img_h, img_w = vis.shape[:2]

    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        return vis

    # --- Crack mask overlay (subtle red tint) ---
    mask_bin = crack_mask > 127
    vis[mask_bin] = cv2.addWeighted(
        original_bgr[mask_bin], 0.35,
        np.full_like(original_bgr[mask_bin], (0, 0, 255)), 0.65, 0
    )

    dom_angle = orientations_data.get('dominant_angle_deg', 0.0)
    dom_type  = orientations_data.get('dominant_type', 'Undetermined')
    n_comps   = orientations_data.get('evaluated_component_count', 0)

    # Retrieve pre-computed dominant eigenvector and centroid from measure_crack_orientations
    dom_center     = orientations_data.get('dominant_center', None)
    dom_eigenvec   = orientations_data.get('dominant_eigenvector', None)

    # --- Dominant PCA axis across the whole image ---
    if dom_center is not None and dom_eigenvec is not None:
        cx, cy = float(dom_center[0]), float(dom_center[1])
        vx, vy = float(dom_eigenvec[0]), float(dom_eigenvec[1])

        # Extend axis until it hits the image boundary
        # parametric line: (cx + t*vx, cy + t*vy) — solve for t at all 4 edges
        t_candidates = []
        eps = 1e-9
        if abs(vx) > eps:
            t_candidates.append((0 - cx) / vx)        # left edge
            t_candidates.append((img_w - 1 - cx) / vx)  # right edge
        if abs(vy) > eps:
            t_candidates.append((0 - cy) / vy)        # top edge
            t_candidates.append((img_h - 1 - cy) / vy)  # bottom edge

        if t_candidates:
            t_min = min(t_candidates)
            t_max = max(t_candidates)
            ax_p1 = (int(round(cx + t_min * vx)), int(round(cy + t_min * vy)))
            ax_p2 = (int(round(cx + t_max * vx)), int(round(cy + t_max * vy)))
        else:
            # Fallback: short fixed-length axis
            half = min(img_w, img_h) // 3
            ax_p1 = (int(round(cx - vx * half)), int(round(cy - vy * half)))
            ax_p2 = (int(round(cx + vx * half)), int(round(cy + vy * half)))

        # Dark shadow for contrast, then bright color on top
        AXIS_COLOR = (0, 220, 255)   # Amber/Gold
        cv2.line(vis, ax_p1, ax_p2, (10, 10, 10), 5, cv2.LINE_AA)   # shadow
        cv2.line(vis, ax_p1, ax_p2, AXIS_COLOR,   3, cv2.LINE_AA)   # axis

        # Arrowhead at both ends to indicate bi-directional principal axis
        cv2.arrowedLine(vis, ax_p2, ax_p1, AXIS_COLOR, 2, cv2.LINE_AA, tipLength=0.02)
        cv2.arrowedLine(vis, ax_p1, ax_p2, AXIS_COLOR, 2, cv2.LINE_AA, tipLength=0.02)

        # Centroid marker
        cx_i, cy_i = int(round(cx)), int(round(cy))
        cv2.circle(vis, (cx_i, cy_i), 9,  (10, 10, 10),  -1)
        cv2.circle(vis, (cx_i, cy_i), 7,  AXIS_COLOR,     -1)
        cv2.circle(vis, (cx_i, cy_i), 3,  (255, 255, 255), -1)

        # Angle badge next to centroid (single, non-overlapping)
        badge = f"{dom_angle:.1f}deg"
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx = int(np.clip(cx_i + 14, 10, img_w - bw - 14))
        by = int(np.clip(cy_i - 14, 40, img_h - bh - 14))
        cv2.rectangle(vis, (bx - 4, by - bh - 4), (bx + bw + 4, by + 4), (10, 10, 10), -1)
        cv2.rectangle(vis, (bx - 4, by - bh - 4), (bx + bw + 4, by + 4), AXIS_COLOR, 1, cv2.LINE_AA)
        cv2.putText(vis, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Top HUD banner ---
    hud_text = f"Dominant Orientation: {dom_angle:.1f}deg ({dom_type})  |  PCA Comps: {n_comps}"
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(vis, (10, 8), (10 + hw + 20, 8 + hh + 16), (10, 10, 10), -1)
    cv2.rectangle(vis, (10, 8), (10 + hw + 20, 8 + hh + 16), (255, 128, 0), 2)
    cv2.putText(vis, hud_text, (20, 8 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    return vis


def run_measurement_pipeline(segmentation_results: dict,
                             output_dir: str = None,
                             min_orientation_points: int = 30,
                             gap_closing_ksize: int = 7,
                             gap_closing_iters: int = 1,
                             enable_gap_closing: bool = True,
                             save_steps: bool = True) -> dict:
    """
    Execute Step 3 (Width), Step 4A (Length), and Step 4B (Orientation) Pipeline on segmentation results.
    
    Args:
        segmentation_results:    Output dict from segmentation.py
        output_dir:              Directory to save output files
        min_orientation_points:  Minimum component points for PCA orientation analysis
        gap_closing_ksize:       Kernel size for Step 4A morphological closing (default: 7)
        gap_closing_iters:       Iterations of Step 4A morphological closing (default: 1)
        enable_gap_closing:      Whether to enable Step 4A gap closing (default: True)
        save_steps:              Whether to save images to disk
        
    Returns:
        dict containing width, length, and orientation metrics, skeleton, and visual outputs.
    """
    print("\n" + "="*60)
    print("  CrackGauge Steps 3, 4A, 4B: Width, Length & Orientation")
    print("="*60)
    
    original = segmentation_results['original']
    final_mask = segmentation_results['final_mask']
    
    # 1. Skeletonize (Step 3 Medial Axis)
    skeleton = compute_skeleton(final_mask)
    
    # 2. Distance Transform (Step 3)
    dist_transform = compute_distance_transform(final_mask)
    
    # 3. Measure Widths (Step 3) - Strictly preserves existing calculations
    width_stats = measure_crack_widths(final_mask, skeleton, dist_transform)
    
    # 4. Measure Lengths (Step 4A) - With configurable morphological gap-closing & re-skeletonization
    length_stats = measure_crack_lengths(
        final_mask,
        skeleton,
        gap_closing_ksize=gap_closing_ksize,
        gap_closing_iters=gap_closing_iters,
        enable_gap_closing=enable_gap_closing
    )
    
    # 5. Measure Orientation via PCA (Step 4B) - Strictly preserves existing calculations
    orientation_stats = measure_crack_orientations(final_mask, min_points=min_orientation_points)
    
    # 6. Width Heatmap Image (Step 3)
    width_vis = create_width_visualization_image(
        original, skeleton, width_stats['width_map'], width_stats['max_width_coords']
    )
    
    # 7. Length Component Image (Step 4A, non-overlapping badges on closed mask & recomputed skeleton)
    length_vis = create_length_visualization_image(
        original,
        length_stats['closed_mask'],
        length_stats['length_skeleton'],
        length_stats['components'],
        length_stats['total_length_px'],
        raw_component_count=length_stats['raw_component_count']
    )
    
    # 8. Orientation PCA Image (Step 4B)
    orientation_vis = create_orientation_visualization_image(
        original, final_mask, skeleton, orientation_stats
    )
    
    results = {
        'original':                   original,
        'final_mask':                 final_mask,
        'closed_mask':                length_stats['closed_mask'],
        'skeleton':                   skeleton,
        'length_skeleton':            length_stats['length_skeleton'],
        'dist_transform':             dist_transform,
        'width_map':                  width_stats['width_map'],
        'width_vis':                  width_vis,
        'length_vis':                 length_vis,
        'orientation_vis':            orientation_vis,
        'max_width_px':               width_stats['max_width_px'],
        'mean_width_px':              width_stats['mean_width_px'],
        'median_width_px':            width_stats['median_width_px'],
        'std_width_px':               width_stats['std_width_px'],
        'skeleton_points':            width_stats['skeleton_points'],
        'length_skeleton_points':     int(np.sum(length_stats['length_skeleton'] > 0)),
        'max_width_coords':           width_stats['max_width_coords'],
        'total_length_px':            length_stats['total_length_px'],
        'component_count':            length_stats['component_count'],
        'raw_component_count':        length_stats['raw_component_count'],
        'components':                 length_stats['components'],
        'dominant_angle_deg':         orientation_stats['dominant_angle_deg'],
        'dominant_type':              orientation_stats['dominant_type'],
        'component_orientations':     orientation_stats['components'],
    }
    
    # Save outputs if requested
    if save_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = segmentation_results.get('image_name', 'crack_result')
        
        # Step 3 files
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_10_skeleton.jpg"), skeleton)
        
        dist_max = float(dist_transform.max())
        if dist_max > 0:
            dist_vis = (dist_transform / dist_max * 255.0).astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_VIRIDIS)
        else:
            dist_color = np.zeros_like(original)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_11_distance_transform.jpg"), dist_color)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_12_width_measurement.jpg"), width_vis)
        
        # Step 4A file
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_13_length_measurement.jpg"), length_vis)
        
        # Step 4B file
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_14_orientation_measurement.jpg"), orientation_vis)
        print(f"[OK] Measurement step images saved to: {output_dir}")
        
    print(f"\n{'='*60}")
    print(f"  Measurement Summary (PIXELS & DEGREES):")
    print(f"  Total Crack Length  : {length_stats['total_length_px']:.2f} px")
    n_raw = length_stats['raw_component_count']
    n_closed = length_stats['component_count']
    if n_raw != n_closed:
        print(f"  Crack Components    : {n_closed} (reduced from {n_raw} raw fragments via gap closing)")
    else:
        print(f"  Crack Components    : {n_closed}")
    print(f"  Max Width           : {width_stats['max_width_px']:.2f} px")
    print(f"  Mean Width          : {width_stats['mean_width_px']:.2f} px")
    print(f"  Dominant Orientation: {orientation_stats['dominant_angle_deg']:.1f} deg ({orientation_stats['dominant_type']})")
    print(f"{'='*60}\n")
    
    return results



def visualize_measurement(results: dict, save_path: str = None):
    """
    Create a 4-panel visual comparison for Width Measurement:
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
        print(f"[OK] Width measurement visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_length_measurement(results: dict, save_path: str = None):
    """
    Create a 4-panel visual comparison for Length Measurement:
      Panel 1: Original Concrete Image
      Panel 2: Final Segmented Crack Mask
      Panel 3: Crack Skeleton with Colored Components
      Panel 4: Length Measurement Map (with Component Badges & Total Length)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("CrackGauge - Step 4A: Crack Length Measurement (Skeleton Path & Components)",
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
    len_skel = results.get('length_skeleton', results['skeleton'])
    ax3.imshow(len_skel, cmap='hot')
    tot_len = results.get('total_length_px', 0.0)
    n_comp = results.get('component_count', 0)
    n_raw = results.get('raw_component_count', n_comp)
    skel_pts = results.get('length_skeleton_points', results.get('skeleton_points', 0))
    if n_raw != n_comp:
        panel3_title = f"3. Recalculated Skeleton ({skel_pts:,} px | {n_raw} -> {n_comp} Components)"
    else:
        panel3_title = f"3. Crack Skeleton ({skel_pts:,} px | {n_comp} Components)"
    ax3.set_title(panel3_title, fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Panel 4: Length Measurement Vis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cv2.cvtColor(results['length_vis'], cv2.COLOR_BGR2RGB))
    ax4.set_title(f"4. Length Measurement Map [Total Length: {tot_len:.1f} px | {n_comp} Comps]",
                  fontsize=11, fontweight='bold', color='#27ae60', pad=8)
    ax4.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[OK] Length measurement visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_orientation_measurement(results: dict, save_path: str = None):
    """
    Create a 4-panel visual comparison for Orientation Measurement (PCA):
      Panel 1: Original Concrete Image
      Panel 2: Final Segmented Crack Mask
      Panel 3: Crack Skeleton with Centroids
      Panel 4: PCA Orientation Map (Principal Axes & Direction Vectors)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("CrackGauge - Step 4B: Crack Orientation Measurement (PCA Principal Direction)",
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
    dom_ang = results.get('dominant_angle_deg', 0.0)
    dom_type = results.get('dominant_type', 'Undetermined')
    n_pts = results.get('skeleton_points', 0)
    ax3.set_title(f"3. Crack Skeleton ({n_pts:,} px)",
                  fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    
    # Panel 4: Orientation Map Vis
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(cv2.cvtColor(results['orientation_vis'], cv2.COLOR_BGR2RGB))
    ax4.set_title(f"4. PCA Orientation Map [Dominant: {dom_ang:.1f}deg ({dom_type})]",
                  fontsize=11, fontweight='bold', color='#8e44ad', pad=8)
    ax4.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"[OK] Orientation measurement visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()


