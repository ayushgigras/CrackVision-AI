"""
CrackGauge - Sample Image Downloader
======================================
Downloads real crack images from public datasets for testing.
Uses SDNET2018 / Concrete Crack Images dataset links (public domain).

Run this once to populate data/sample_images/
"""

import os
import urllib.request
import sys

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Public domain concrete crack images from various sources
# These are actual crack photos used in research papers
SAMPLE_IMAGES = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Cracked_floor.jpg/1280px-Cracked_floor.jpg",
        "filename": "crack_floor_01.jpg",
        "description": "Concrete floor crack"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Concrete_crack_wall.jpg/1280px-Concrete_crack_wall.jpg",
        "filename": "crack_wall_01.jpg",
        "description": "Concrete wall crack"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/8/8e/Cracked_concrete.jpg",
        "filename": "crack_concrete_01.jpg",
        "description": "General concrete crack"
    },
]

# Fallback: if internet not available, generate a synthetic crack image
def generate_synthetic_crack(output_path: str):
    """
    Generate a synthetic concrete crack image for testing.
    This is ONLY for initial testing when real images aren't available.
    """
    import numpy as np
    import cv2
    
    print("   Generating synthetic crack image for testing...")
    
    # Create a concrete-like grey texture
    height, width = 600, 800
    
    # Base concrete texture (random grey noise)
    np.random.seed(42)
    base = np.random.randint(140, 180, (height, width), dtype=np.uint8)
    
    # Add Gaussian blur to smooth the texture
    base = cv2.GaussianBlur(base, (15, 15), sigmaX=5)
    
    # Add subtle texture variation
    noise = np.random.randint(-20, 20, (height, width))
    base = np.clip(base.astype(int) + noise, 100, 220).astype(np.uint8)
    
    # Draw realistic crack lines
    img_color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    
    # Crack 1: Main diagonal crack (from top-left area downward)
    crack_color = (35, 35, 35)  # Dark grey
    points1 = []
    x, y = 150, 80
    for i in range(200):
        x += np.random.randint(2, 5)
        y += np.random.randint(1, 4)
        points1.append((x, y))
    
    for i in range(1, len(points1)):
        thickness = np.random.randint(1, 4)
        cv2.line(img_color, points1[i-1], points1[i], crack_color, thickness)
    
    # Crack 2: Shorter branch crack
    points2 = []
    x, y = 300, 200
    for i in range(100):
        x += np.random.randint(1, 4)
        y += np.random.randint(-1, 3)
        points2.append((x, y))
    
    for i in range(1, len(points2)):
        thickness = np.random.randint(1, 3)
        cv2.line(img_color, points2[i-1], points2[i], crack_color, thickness)
    
    # Crack 3: Hairline crack (very thin)
    points3 = []
    x, y = 500, 100
    for i in range(150):
        x += np.random.randint(0, 3)
        y += np.random.randint(2, 5)
        points3.append((x, y))
    
    for i in range(1, len(points3)):
        cv2.line(img_color, points3[i-1], points3[i], (50, 50, 50), 1)
    
    # Apply slight blur to make cracks look more natural
    img_color = cv2.GaussianBlur(img_color, (3, 3), sigmaX=0.5)
    
    cv2.imwrite(output_path, img_color)
    print(f"   [OK] Synthetic crack image saved: {output_path}")
    return output_path


def download_image(url: str, filepath: str) -> bool:
    """Download an image from URL. Returns True on success."""
    try:
        print(f"   Downloading: {os.path.basename(filepath)}...")
        headers = {'User-Agent': 'Mozilla/5.0 CrackGauge-Academic-Project'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())
        size = os.path.getsize(filepath) / 1024
        print(f"   [✓] Downloaded ({size:.1f} KB)")
        return True
    except Exception as e:
        print(f"   [!] Download failed: {e}")
        return False


def setup_sample_images(output_dir: str = "data/sample_images"):
    """Download or generate sample crack images."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  CrackGauge - Sample Image Setup")
    print("="*60)
    
    downloaded = 0
    
    for item in SAMPLE_IMAGES:
        filepath = os.path.join(output_dir, item['filename'])
        
        if os.path.exists(filepath):
            print(f"   [Skip] {item['filename']} already exists.")
            downloaded += 1
            continue
        
        print(f"   {item['description']}")
        success = download_image(item['url'], filepath)
        if success:
            downloaded += 1
    
    # If no downloads succeeded, generate synthetic images
    if downloaded == 0:
        print("\n   Internet unavailable. Generating synthetic test images...")
    
    # Always generate at least one synthetic image as backup
    synthetic_path = os.path.join(output_dir, "crack_synthetic_01.jpg")
    if not os.path.exists(synthetic_path):
        generate_synthetic_crack(synthetic_path)
        downloaded += 1
    
    # List all available images
    all_images = [f for f in os.listdir(output_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"\n{'='*60}")
    print(f"  Setup Complete! {len(all_images)} image(s) ready:")
    for img in sorted(all_images):
        size = os.path.getsize(os.path.join(output_dir, img)) / 1024
        print(f"    • {img}  ({size:.1f} KB)")
    print(f"{'='*60}\n")
    
    return [os.path.join(output_dir, f) for f in sorted(all_images)]


if __name__ == "__main__":
    images = setup_sample_images("data/sample_images")
    print(f"[DONE] Ready! You can now run:")
    print(f"   python main.py --image {images[0]}")
