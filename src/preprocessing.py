"""
CrackGauge - Preprocessing Pipeline (Step 1)
=============================================
Takes a raw concrete image and produces a clean binary crack mask.

Pipeline:
  Input Image
      → Grayscale
      → Gaussian Blur (noise reduction)
      → CLAHE (contrast enhancement)
      → Adaptive Thresholding
      → Morphological Cleanup
      → Crack Mask (binary)

Author: CrackGauge Team (Manish, Sachin, Ayush)
"""

import cv2
import numpy as np
import os
import sys

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk.
    Returns the original BGR image (as OpenCV loads it).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image (unsupported format?): {image_path}")
    
    print(f"[OK] Loaded image: {os.path.basename(image_path)} | Shape: {img.shape}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    Grayscale is all we need for crack detection — color info not needed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"[OK] Grayscale conversion done | Shape: {gray.shape}")
    return gray


def reduce_noise(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency noise.
    
    Args:
        gray:  Grayscale image
        ksize: Kernel size (must be odd). Higher = more blur. Default 5.
    
    Why Gaussian?
        Concrete surfaces have texture noise. Gaussian blur smooths it
        without destroying crack edges (unlike median blur).
    """
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=0)
    print(f"[OK] Gaussian blur applied (kernel={ksize}x{ksize})")
    return blurred


def enhance_contrast(gray: np.ndarray,
                     clip_limit: float = 3.0,
                     tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Why CLAHE?
        Concrete images often have uneven lighting (shadows, bright spots).
        CLAHE enhances contrast locally — so cracks in dark areas become
        visible, and bright areas don't get over-amplified.
    
    Args:
        gray:           Grayscale input
        clip_limit:     Threshold for contrast limiting. Higher = more contrast.
        tile_grid_size: Size of grid for local histogram equalization.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    print(f"[OK] CLAHE contrast enhancement done (clip={clip_limit})")
    return enhanced


def adaptive_threshold(enhanced: np.ndarray,
                       block_size: int = 21,
                       C: int = 5) -> np.ndarray:
    """
    Apply Adaptive Thresholding to binarize the image.
    
    Why Adaptive (not Global)?
        Global thresholding fails when lighting is uneven across the image.
        Adaptive thresholding computes a LOCAL threshold for each pixel's
        neighborhood — much better for real-world concrete images.
    
    Args:
        enhanced:   Contrast-enhanced grayscale image
        block_size: Size of local neighborhood (must be odd). Default 11.
        C:          Constant subtracted from mean. Helps fine-tune.
    
    Returns:
        Binary image where cracks appear as WHITE (255) on BLACK (0).
        Note: cv2 adaptive threshold gives dark=crack initially, so we invert.
    """
    # THRESH_BINARY_INV makes cracks WHITE (bright areas = non-crack → black)
    binary = cv2.adaptiveThreshold(
        enhanced,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=C
    )
    print(f"[OK] Adaptive thresholding done (block={block_size}, C={C})")
    return binary


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Clean up the binary mask using morphological operations.
    
    Steps:
        1. Opening (erosion → dilation): Removes small noise dots (pepper noise)
        2. Closing (dilation → erosion): Fills tiny gaps in crack regions
        3. Remove small connected components (area filter)
    
    Why?
        After thresholding, there are many small white blobs that are NOT cracks
        (surface texture, dust, stains). We remove them here.
    """
    # Step 1: Opening — remove isolated noise pixels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # small: removes noise dots
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Step 2: Closing — fill small gaps within crack
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # larger: fills gaps in crack
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    
    # Step 3: Remove small connected components (area < min_area are noise)
    mask = _remove_small_components(closed, min_area=300)  # 300px² is roughly a 17x17 region
    
    print(f"[OK] Morphological cleanup done")
    return mask


def _remove_small_components(binary: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove connected components (blobs) smaller than min_area pixels.
    These are almost always noise, not actual cracks.
    """
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create output mask
    output = np.zeros_like(binary)
    
    # Keep components with area >= min_area (label 0 = background, skip it)
    kept = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == label] = 255
            kept += 1
    
    print(f"   -> Connected components: {num_labels - 1} found, {kept} kept (area >= {min_area}px)")
    return output


def run_preprocessing_pipeline(image_path: str,
                                output_dir: str = None,
                                save_steps: bool = True) -> dict:
    """
    Run the full preprocessing pipeline on a single image.
    
    Args:
        image_path: Path to input concrete image
        output_dir: Directory to save step-by-step output images
        save_steps: If True, saves each intermediate step as an image
    
    Returns:
        dict with keys:
            'original'  : Original BGR image
            'gray'      : Grayscale image
            'blurred'   : Noise-reduced image
            'enhanced'  : Contrast-enhanced image
            'binary'    : Thresholded binary image
            'crack_mask': Final cleaned crack mask
    """
    print("\n" + "="*60)
    print("  CrackGauge Preprocessing Pipeline")
    print("="*60)
    
    # --- Pipeline Steps ---
    original  = load_image(image_path)
    gray      = to_grayscale(original)
    blurred   = reduce_noise(gray, ksize=9)
    enhanced  = enhance_contrast(blurred, clip_limit=3.0)
    binary    = adaptive_threshold(enhanced, block_size=21, C=5)
    crack_mask = morphological_cleanup(binary)
    
    results = {
        'original':   original,
        'gray':       gray,
        'blurred':    blurred,
        'enhanced':   enhanced,
        'binary':     binary,
        'crack_mask': crack_mask,
    }
    
    # --- Save outputs ---
    if save_steps and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_01_gray.jpg"),      gray)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_02_blurred.jpg"),   blurred)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_03_enhanced.jpg"),  enhanced)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_04_binary.jpg"),    binary)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_05_crack_mask.jpg"), crack_mask)
        print(f"\n[OK] All step images saved to: {output_dir}")
    
    # --- Summary ---
    crack_pixels = np.sum(crack_mask > 0)
    total_pixels = crack_mask.shape[0] * crack_mask.shape[1]
    crack_percent = (crack_pixels / total_pixels) * 100
    
    print(f"\n{'='*60}")
    print(f"  Pipeline Complete!")
    print(f"  Image size      : {original.shape[1]}x{original.shape[0]} px")
    print(f"  Crack pixels    : {crack_pixels:,}")
    print(f"  Crack coverage  : {crack_percent:.2f}%")
    print(f"{'='*60}\n")
    
    return results


def visualize_results(results: dict, save_path: str = None):
    """
    Create a side-by-side comparison visualization of pipeline steps.
    Saves to file if save_path is given, otherwise shows on screen.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("CrackGauge - Preprocessing Pipeline Results", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    # Plot config: (title, image, colormap)
    plots = [
        ("1. Original Image",         cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB), None),
        ("2. Grayscale",               results['gray'],       'gray'),
        ("3. Noise Reduction (Blur)",  results['blurred'],    'gray'),
        ("4. CLAHE Enhancement",       results['enhanced'],   'gray'),
        ("5. Adaptive Threshold",      results['binary'],     'gray'),
        ("6. Crack Mask (Final)",      results['crack_mask'], 'gray'),
    ]
    
    axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
    
    for ax, (title, img, cmap) in zip(axes, plots):
        if cmap:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Add image dimensions as subtitle
        h, w = img.shape[:2]
        ax.text(0.5, -0.04, f"{w}×{h} px", transform=ax.transAxes,
                ha='center', fontsize=9, color='gray')
    
    # Highlight the final crack mask panel
    axes[5].set_title("6. Crack Mask (Final)", 
                       fontsize=11, fontweight='bold', color='#e74c3c', pad=8)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"[OK] Visualization saved: {save_path}")
        plt.close()
    else:
        plt.show()


# ── Standalone test (run this file directly) ──────────────────────────────────
if __name__ == "__main__":
    import sys
    import glob
    
    # Try to find a sample image automatically
    search_dirs = [
        "data/sample_images",
        "../data/sample_images",
        ".",
    ]
    
    test_image = None
    
    # Check if image path given as argument
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Auto-search for any image in sample_images dir
        for d in search_dirs:
            patterns = [f"{d}/*.jpg", f"{d}/*.jpeg", f"{d}/*.png", f"{d}/*.bmp"]
            for pattern in patterns:
                found = glob.glob(pattern)
                if found:
                    test_image = found[0]
                    break
            if test_image:
                break
    
    if test_image is None:
        print("No image found! Please provide an image path:")
        print("  python preprocessing.py path/to/crack_image.jpg")
        print("\nOr place images in: data/sample_images/")
        sys.exit(1)
    
    print(f"Using image: {test_image}")
    
    # Run pipeline
    results = run_preprocessing_pipeline(
        image_path=test_image,
        output_dir="data/outputs",
        save_steps=True
    )
    
    # Save visualization
    visualize_results(results, save_path="data/outputs/pipeline_result.png")
    print("Done! Check data/outputs/ folder for results.")
