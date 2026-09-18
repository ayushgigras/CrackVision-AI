"""
CrackGauge - Main Entry Point
==============================
Run the full preprocessing pipeline from command line.

Usage:
    python main.py                              # auto-finds image in data/sample_images/
    python main.py --image path/to/crack.jpg   # specific image
    python main.py --image path/to/crack.jpg --show   # also show plots
"""

import argparse
import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import run_preprocessing_pipeline, visualize_results
from segmentation import run_segmentation_pipeline, visualize_segmentation
from measurement import run_measurement_pipeline, visualize_measurement, visualize_length_measurement


def process_single_image(image_path: str, output_dir: str, args):
    """Process a single crack image through Step 1 (Preprocessing), Step 2 (Segmentation), and Step 3/4A (Width & Length Measurement)."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # --- Step 1: Preprocessing ---
    prep_results = run_preprocessing_pipeline(
        image_path=image_path,
        output_dir=output_dir,
        save_steps=True
    )
    
    # Save Step 1 Visualization
    viz_prep_path = os.path.join(output_dir, f"{base_name}_preprocessing_viz.png")
    visualize_results(prep_results, save_path=viz_prep_path)
    # Also save standard pipeline_visualization.png for backward compatibility
    legacy_viz = os.path.join(output_dir, "pipeline_visualization.png")
    visualize_results(prep_results, save_path=legacy_viz)
    
    # --- Step 2: Crack Segmentation (Hessian/Frangi) ---
    seg_results = run_segmentation_pipeline(
        preprocessed_results=prep_results,
        output_dir=output_dir,
        sigmas=(1, 2, 3, 4),
        binarize_method=args.bin_method,
        min_area=args.min_area,
        fusion_mode=args.fusion,
        save_steps=True
    )
    
    # Save Step 2 Visualization
    viz_seg_path = os.path.join(output_dir, f"{base_name}_segmentation_viz.png")
    visualize_segmentation(seg_results, save_path=viz_seg_path)
    
    # --- Step 3 & 4A: Width & Length Measurement (Medial Axis & Distance Transform) ---
    meas_results = run_measurement_pipeline(
        segmentation_results=seg_results,
        output_dir=output_dir,
        save_steps=True
    )
    
    # Save Step 3 Width Visualization
    viz_meas_path = os.path.join(output_dir, f"{base_name}_measurement_viz.png")
    visualize_measurement(meas_results, save_path=viz_meas_path)
    
    # Save Step 4A Length Visualization
    viz_len_path = os.path.join(output_dir, f"{base_name}_length_viz.png")
    visualize_length_measurement(meas_results, save_path=viz_len_path)
    
    return {
        'image_name': base_name,
        'image_path': image_path,
        'prep':       prep_results,
        'seg':        seg_results,
        'meas':       meas_results,
        'prep_viz':   viz_prep_path,
        'seg_viz':    viz_seg_path,
        'meas_viz':   viz_meas_path,
        'len_viz':    viz_len_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CrackGauge - Computer Vision Crack Measurement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --all
  python main.py --image data/sample_images/crack_synthetic_01.jpg
  python main.py --image data/sample_images/crack_synthetic_01.jpg --show
        """
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to the input crack image'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Process all images found in data/sample_images/'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/outputs',
        help='Directory to save output images (default: data/outputs)'
    )
    parser.add_argument(
        '--min-area',
        type=int,
        default=80,
        help='Minimum crack component area in pixels (default: 80)'
    )
    parser.add_argument(
        '--bin-method',
        type=str,
        choices=['otsu', 'cutoff'],
        default='otsu',
        help='Ridge binarization method: otsu or cutoff (default: otsu)'
    )
    parser.add_argument(
        '--fusion',
        type=str,
        choices=['frangi_primary', 'intersection', 'union'],
        default='frangi_primary',
        help='Mask fusion mode (default: frangi_primary)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display visualization window (requires display)'
    )
    
    args = parser.parse_args()
    
    import glob
    
    # Collect target images
    target_images = []
    
    if args.all:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            target_images.extend(glob.glob(os.path.join("data/sample_images", ext)))
        target_images = sorted(list(set(target_images)))
        if not target_images:
            print("[ERROR] No images found in data/sample_images/")
            sys.exit(1)
        print(f"[INFO] Found {len(target_images)} images to process.")
    elif args.image:
        if not os.path.exists(args.image):
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        target_images = [args.image]
    else:
        # Auto-search first available image
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            found = glob.glob(os.path.join("data/sample_images", ext))
            if found:
                target_images = [sorted(found)[0]]
                print(f"[Auto] Using image: {target_images[0]}")
                break
                
    if not target_images:
        print("[ERROR] No image found!")
        print("   Place a crack image in: data/sample_images/")
        print("   Or run: python main.py --image your_image.jpg")
        sys.exit(1)
        
    # --- Execute Pipeline on all target images ---
    all_reports = []
    for img_path in target_images:
        res = process_single_image(img_path, args.output, args)
        all_reports.append(res)
        
    # --- Print Final Summary Report ---
    print("\n" + "="*105)
    print("  CrackGauge Pipeline Execution Summary (Steps 1, 2, 3, 4A)")
    print("="*105)
    print(f"{'Image Name':<22} | {'Crack %':<8} | {'Comps':<6} | {'Total Length':<13} | {'Max Width':<11} | {'Mean Width':<11}")
    print("-" * 105)
    for rep in all_reports:
        name = rep['image_name']
        cp_pct = rep['seg']['crack_percent']
        tot_len = rep['meas']['total_length_px']
        n_comps = rep['meas']['component_count']
        max_w = rep['meas']['max_width_px']
        mean_w = rep['meas']['mean_width_px']
        print(f"{name:<22} | {cp_pct:>7.2f}% | {n_comps:>6d} | {tot_len:>10.1f} px | {max_w:>9.2f} px | {mean_w:>9.2f} px")
    print("="*105)
    print("  * Note: All measurements are reported in PIXELS (ground-truth calibration pending).")
    
    # Detailed Component Breakdown
    print("\n" + "-"*75)
    print("  Per-Component Crack Length Breakdown (PIXELS):")
    print("-" * 75)
    for rep in all_reports:
        name = rep['image_name']
        comps = rep['meas']['components']
        print(f"\n  [{name}] - {len(comps)} component(s):")
        for c in comps:
            print(f"    - Comp #{c['id']:2d}: Length = {c['length_px']:>7.1f} px | Skeleton Pts = {c['skeleton_pixels']:>4d} | Mask Area = {c['mask_area']:>4d} px")
    print("-" * 75)
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
        
    print(f"\n[SUCCESS] Pipeline complete! Output directory: {args.output}/")
    print("Steps verified: Step 1 (Preprocessing) -> Step 2 (Segmentation) -> Step 3 (Width) -> Step 4A (Length).")


if __name__ == "__main__":
    main()

