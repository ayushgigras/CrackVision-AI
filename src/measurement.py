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
                          skeleton: np.ndarray = None) -> dict:
    """
    Calculate total crack length and per-component crack lengths in pixels from the skeleton.
    
    Path Length Calculation:
        For each connected component, adjacent 8-connected skeleton pixels are traversed:
          - Orthogonal neighbor step: distance = 1.0 px
          - Diagonal neighbor step:   distance = sqrt(2) approx 1.414 px
        Undirected neighbor edges are summed to obtain the true path length in pixels.
        
    Args:
        crack_mask: Binary crack mask (uint8)
        skeleton:   Optional precomputed 1-pixel wide skeleton (uint8)
        
    Returns:
        dict containing:
            'total_length_px':  float, sum of all component lengths in pixels
            'component_count':  int, number of crack components
            'components':       list of dicts for each component:
                {
                    'id':              int (1-based),
                    'length_px':       float,
                    'skeleton_pixels': int,
                    'mask_area':       int,
                    'centroid':        tuple (cx, cy),
                    'bbox':            tuple (x, y, w, h)
                }
    """
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        print("[INFO] No crack pixels found for length measurement. Total length set to 0.0 px.")
        return {
            'total_length_px': 0.0,
            'component_count': 0,
            'components': [],
        }
        
    if skeleton is None:
        skeleton = compute_skeleton(crack_mask)
        
    mask_bin = (crack_mask > 127).astype(np.uint8)
    skel_bool = skeleton > 0
    
    # Label connected components on mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    
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
        
    print(f"[OK] Length calculation completed: {len(components)} components | Total Length: {total_length:.2f} px")
    return {
        'total_length_px': total_length,
        'component_count': len(components),
        'components': components,
    }


def create_length_visualization_image(original_bgr: np.ndarray,
                                      crack_mask: np.ndarray,
                                      skeleton: np.ndarray,
                                      components: list,
                                      total_length_px: float) -> np.ndarray:
    """
    Generate an annotated image highlighting each crack component's path in distinct colors
    with component ID and length callout badges.
    """
    vis = original_bgr.copy()
    
    if not components or skeleton is None or np.sum(skeleton > 0) == 0:
        return vis
        
    # High-contrast color palette (BGR) for distinguishing components
    PALETTE = [
        (0, 255, 255),    # Yellow
        (0, 255, 0),      # Lime Green
        (255, 128, 0),    # Orange/Cyan
        (255, 0, 255),    # Magenta
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Violet
        (0, 215, 255),    # Gold
        (50, 205, 50),    # Lime
        (255, 192, 203),  # Pink
        (220, 20, 60),    # Crimson
        (0, 128, 255),    # Amber
        (240, 230, 140),  # Khaki
    ]
    
    mask_bin = (crack_mask > 127).astype(np.uint8)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    skel_bool = skeleton > 0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    comp_map = {c['id']: c for c in components}
    actual_idx = 1
    
    for label in range(1, num_labels):
        comp_skel_mask = (labels == label) & skel_bool
        if not np.any(comp_skel_mask):
            continue
            
        color = PALETTE[(actual_idx - 1) % len(PALETTE)]
        
        # Dilate component skeleton for crisp visibility
        dilated = cv2.dilate(comp_skel_mask.astype(np.uint8), kernel, iterations=1) > 0
        vis[dilated] = color
        
        # Add badge text near centroid if component is significant
        if actual_idx in comp_map:
            cdata = comp_map[actual_idx]
            cx, cy = cdata['centroid']
            clen = cdata['length_px']
            
            # Badge text
            badge = f"#{actual_idx}: {clen:.1f}px"
            tx = int(np.clip(cx + 8, 10, original_bgr.shape[1] - 120))
            ty = int(np.clip(cy - 8, 20, original_bgr.shape[0] - 10))
            
            # Small dark background for text readability
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (20, 20, 20), -1)
            cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, 1)
            cv2.putText(vis, badge, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        actual_idx += 1
        
    # Top HUD banner with overall length summary
    hud_text = f"Total Crack Length: {total_length_px:.1f} px  |  Components: {len(components)}"
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(vis, (10, 10), (10 + hw + 20, 10 + hh + 16), (20, 20, 20), -1)
    cv2.rectangle(vis, (10, 10), (10 + hw + 20, 10 + hh + 16), (0, 200, 255), 2)
    cv2.putText(vis, hud_text, (20, 10 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    
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
        
    # Dominant orientation across all crack pixels
    if len(all_xs) >= 3:
        all_pts = np.column_stack((all_xs, all_ys))
        overall_pca = compute_pca_orientation(all_pts)
        dom_angle = overall_pca['angle_deg']
        dom_type = overall_pca['orientation_type']
    else:
        dom_angle = 0.0
        dom_type = 'Undetermined'
        
    print(f"[OK] PCA orientation analysis complete: {len(components)} meaningful components evaluated.")
    print(f"     Dominant Orientation: {dom_angle:.1f} deg ({dom_type})")
    
    return {
        'components': components,
        'dominant_angle_deg': dom_angle,
        'dominant_type': dom_type,
        'evaluated_component_count': len(components),
    }


def create_orientation_visualization_image(original_bgr: np.ndarray,
                                           crack_mask: np.ndarray,
                                           skeleton: np.ndarray,
                                           orientations_data: dict) -> np.ndarray:
    """
    Generate an annotated image drawing PCA principal axes and orientation angle badges
    on the original image with cracks.
    """
    vis = original_bgr.copy()
    
    if crack_mask is None or np.sum(crack_mask > 127) == 0:
        return vis
        
    # Subtle overlay of crack mask
    mask_bin = crack_mask > 127
    vis[mask_bin] = cv2.addWeighted(
        original_bgr[mask_bin], 0.35, np.full_like(original_bgr[mask_bin], (0, 0, 255)), 0.65, 0
    )
    
    # Color palette for axes and badges
    PALETTE = [
        (0, 255, 255),    # Yellow
        (0, 255, 0),      # Lime Green
        (255, 128, 0),    # Orange/Cyan
        (255, 0, 255),    # Magenta
        (0, 165, 255),    # Orange
        (255, 255, 0),    # Cyan
        (128, 0, 255),    # Violet
        (0, 215, 255),    # Gold
    ]
    
    components = orientations_data.get('components', [])
    
    for c in components:
        idx = c['id']
        color = PALETTE[(idx - 1) % len(PALETTE)]
        p1, p2 = c['endpoints']
        cx, cy = int(round(c['center'][0])), int(round(c['center'][1]))
        
        # Draw PCA Principal Axis line (thick with dark outline for contrast)
        cv2.line(vis, p1, p2, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.line(vis, p1, p2, color, 2, cv2.LINE_AA)
        
        # Draw centroid marker
        cv2.circle(vis, (cx, cy), 6, (20, 20, 20), -1)
        cv2.circle(vis, (cx, cy), 4, color, -1)
        
        # Badge annotation: "#ID: XX.X° (Type)"
        badge = f"#{idx}: {c['angle_deg']:.1f}deg ({c['orientation_type']})"
        tx = int(np.clip(cx + 10, 10, original_bgr.shape[1] - 220))
        ty = int(np.clip(cy - 10, 25, original_bgr.shape[0] - 10))
        
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (20, 20, 20), -1)
        cv2.rectangle(vis, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), color, 1)
        cv2.putText(vis, badge, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
    # Top HUD banner
    dom_angle = orientations_data.get('dominant_angle_deg', 0.0)
    dom_type = orientations_data.get('dominant_type', 'Undetermined')
    hud_text = f"Dominant Orientation: {dom_angle:.1f}deg ({dom_type})  |  PCA Comps: {len(components)}"
    (hw, hh), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.rectangle(vis, (10, 10), (10 + hw + 20, 10 + hh + 16), (20, 20, 20), -1)
    cv2.rectangle(vis, (10, 10), (10 + hw + 20, 10 + hh + 16), (255, 128, 0), 2)
    cv2.putText(vis, hud_text, (20, 10 + hh + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
    
    return vis


def run_measurement_pipeline(segmentation_results: dict,
                             output_dir: str = None,
                             min_orientation_points: int = 30,
                             save_steps: bool = True) -> dict:
    """
    Execute Step 3 (Width), Step 4A (Length), and Step 4B (Orientation) Pipeline on segmentation results.
    
    Args:
        segmentation_results:    Output dict from segmentation.py
        output_dir:              Directory to save output files
        min_orientation_points:  Minimum component points for PCA orientation analysis
        save_steps:              Whether to save images to disk
        
    Returns:
        dict containing width, length, and orientation metrics, skeleton, and visual outputs.
    """
    print("\n" + "="*60)
    print("  CrackGauge Steps 3, 4A, 4B: Width, Length & Orientation")
    print("="*60)
    
    original = segmentation_results['original']
    final_mask = segmentation_results['final_mask']
    
    # 1. Skeletonize
    skeleton = compute_skeleton(final_mask)
    
    # 2. Distance Transform
    dist_transform = compute_distance_transform(final_mask)
    
    # 3. Measure Widths (Step 3)
    width_stats = measure_crack_widths(final_mask, skeleton, dist_transform)
    
    # 4. Measure Lengths (Step 4A)
    length_stats = measure_crack_lengths(final_mask, skeleton)
    
    # 5. Measure Orientation via PCA (Step 4B)
    orientation_stats = measure_crack_orientations(final_mask, min_points=min_orientation_points)
    
    # 6. Width Heatmap Image
    width_vis = create_width_visualization_image(
        original, skeleton, width_stats['width_map'], width_stats['max_width_coords']
    )
    
    # 7. Length Component Image
    length_vis = create_length_visualization_image(
        original, final_mask, skeleton, length_stats['components'], length_stats['total_length_px']
    )
    
    # 8. Orientation PCA Image
    orientation_vis = create_orientation_visualization_image(
        original, final_mask, skeleton, orientation_stats
    )
    
    results = {
        'original':                   original,
        'final_mask':                 final_mask,
        'skeleton':                   skeleton,
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
        'max_width_coords':           width_stats['max_width_coords'],
        'total_length_px':            length_stats['total_length_px'],
        'component_count':            length_stats['component_count'],
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
    print(f"  Crack Components    : {length_stats['component_count']}")
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
    ax3.imshow(results['skeleton'], cmap='hot')
    tot_len = results.get('total_length_px', 0.0)
    n_comp = results.get('component_count', 0)
    ax3.set_title(f"3. Crack Skeleton ({results['skeleton_points']:,} px | {n_comp} Components)",
                  fontsize=11, fontweight='bold', pad=8)
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


