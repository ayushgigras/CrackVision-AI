"""
CrackGauge - Synthetic Sample Image Generator
==============================================
Generates multiple realistic synthetic concrete crack images for pipeline testing:
  1. crack_synthetic_01.jpg: Diagonal branching crack (already generated)
  2. crack_synthetic_02.jpg: Horizontal flexural / beam crack
  3. crack_synthetic_03.jpg: Multi-hairline shrinkage crack network
  4. crack_synthetic_04.jpg: Vertical longitudinal shear crack on rough concrete
"""

import os
import sys
import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def make_concrete_texture(height=600, width=800, mean_intensity=160, noise_std=25, seed=None):
    """Generate a realistic textured concrete background."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base random texture
    base = np.random.normal(mean_intensity, noise_std, (height, width)).astype(np.float32)
    
    # Large scale roughness (pores and shade variations)
    roughness = cv2.GaussianBlur(base, (31, 31), sigmaX=10)
    
    # Fine aggregate noise
    fine_noise = np.random.normal(0, 10, (height, width)).astype(np.float32)
    fine_noise = cv2.GaussianBlur(fine_noise, (3, 3), sigmaX=1)
    
    # Combine
    concrete = cv2.addWeighted(roughness, 0.7, fine_noise, 0.3, mean_intensity * 0.3)
    concrete = np.clip(concrete, 40, 230).astype(np.uint8)
    
    # Convert to 3-channel BGR
    return cv2.cvtColor(concrete, cv2.COLOR_GRAY2BGR)


def generate_image_02(output_path: str):
    """Horizontal flexural crack (typical beam tension crack)."""
    img = make_concrete_texture(height=600, width=800, mean_intensity=170, seed=101)
    crack_color = (40, 40, 40)
    
    # Horizontal wavy crack across the middle
    points = []
    x, y = 80, 290
    np.random.seed(202)
    while x < 720:
        x += np.random.randint(4, 9)
        y += np.random.randint(-3, 4)
        points.append((x, int(np.clip(y, 100, 500))))
    
    for i in range(1, len(points)):
        # Thicker in the middle, tapering at edges
        dist_from_mid = abs(points[i][0] - 400) / 400.0
        thickness = max(1, int(4 * (1.0 - dist_from_mid * 0.6)))
        cv2.line(img, points[i-1], points[i], crack_color, thickness)
    
    # Small branch
    b_points = []
    bx, by = points[len(points)//2]
    for _ in range(40):
        bx += np.random.randint(1, 4)
        by += np.random.randint(2, 5)
        b_points.append((bx, by))
    for i in range(1, len(b_points)):
        cv2.line(img, b_points[i-1], b_points[i], crack_color, 1)

    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.4)
    cv2.imwrite(output_path, img)
    print(f"[OK] Generated: {os.path.basename(output_path)}")


def generate_image_03(output_path: str):
    """Hairline crack network (shrinkage cracks)."""
    img = make_concrete_texture(height=600, width=800, mean_intensity=155, seed=303)
    crack_color = (55, 55, 55)
    np.random.seed(404)
    
    # 3 interconnected thin hairline crack paths
    starts = [(200, 150), (450, 180), (320, 350)]
    for sx, sy in starts:
        x, y = sx, sy
        pts = [(x, y)]
        for _ in range(90):
            x += np.random.randint(-3, 6)
            y += np.random.randint(2, 6)
            pts.append((int(np.clip(x, 20, 780)), int(np.clip(y, 20, 580))))
        for i in range(1, len(pts)):
            cv2.line(img, pts[i-1], pts[i], crack_color, 1)
            
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.3)
    cv2.imwrite(output_path, img)
    print(f"[OK] Generated: {os.path.basename(output_path)}")


def generate_image_04(output_path: str):
    """Vertical structural crack on darker aged concrete."""
    img = make_concrete_texture(height=600, width=800, mean_intensity=135, noise_std=35, seed=505)
    crack_color = (25, 25, 25)
    np.random.seed(606)
    
    points = []
    x, y = 380, 50
    while y < 550:
        x += np.random.randint(-4, 5)
        y += np.random.randint(4, 9)
        points.append((int(np.clip(x, 100, 700)), y))
        
    for i in range(1, len(points)):
        thickness = np.random.randint(2, 5)
        cv2.line(img, points[i-1], points[i], crack_color, thickness)
        
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.4)
    cv2.imwrite(output_path, img)
    print(f"[OK] Generated: {os.path.basename(output_path)}")


def generate_all_samples(output_dir="data/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    img2 = os.path.join(output_dir, "crack_synthetic_02.jpg")
    img3 = os.path.join(output_dir, "crack_synthetic_03.jpg")
    img4 = os.path.join(output_dir, "crack_synthetic_04.jpg")
    
    generate_image_02(img2)
    generate_image_03(img3)
    generate_image_04(img4)
    print(f"\n[OK] All synthetic test images ready in {output_dir}/")


if __name__ == "__main__":
    generate_all_samples()
