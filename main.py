"""
CrackGauge - Main Entry Point
==============================
Run the full crack-measurement pipeline on any input image.

Usage:
    python main.py                                                # interactive: prompts for image path
    python main.py --image path/to/crack.jpg                     # direct CLI argument
    python main.py --image path/to/crack.jpg --show              # also display plots
    python main.py --all                                          # process all images in data/sample_images/

Supported formats: .jpg  .jpeg  .png  .bmp  .webp
"""

import argparse
import os
import sys
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure project root is on sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import run_preprocessing_pipeline, visualize_results
from src.segmentation import run_segmentation_pipeline, visualize_segmentation
from src.measurement import (
    run_measurement_pipeline,
    visualize_measurement,
    visualize_length_measurement,
    visualize_orientation_measurement,
)


def process_single_image(image_path: str, output_dir: str, args):
    """Process a single crack image through Step 1 (Preprocessing), Step 2 (Segmentation), and Step 3/4A/4B (Width, Length & Orientation)."""
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
    
    # --- Step 3, 4A, 4B: Width, Length & Orientation Measurement ---
    meas_results = run_measurement_pipeline(
        segmentation_results=seg_results,
        output_dir=output_dir,
        gap_closing_ksize=getattr(args, 'gap_closing_ksize', 7),
        gap_closing_iters=getattr(args, 'gap_closing_iters', 1),
        enable_gap_closing=(getattr(args, 'gap_closing_ksize', 7) > 1),
        save_steps=True
    )
    
    # Save Step 3 Width Visualization
    viz_meas_path = os.path.join(output_dir, f"{base_name}_measurement_viz.png")
    visualize_measurement(meas_results, save_path=viz_meas_path)
    
    # Save Step 4A Length Visualization
    viz_len_path = os.path.join(output_dir, f"{base_name}_length_viz.png")
    visualize_length_measurement(meas_results, save_path=viz_len_path)
    
    # Save Step 4B Orientation Visualization (PCA)
    viz_orient_path = os.path.join(output_dir, f"{base_name}_orientation_viz.png")
    visualize_orientation_measurement(meas_results, save_path=viz_orient_path)
    
    return {
        'image_name':  base_name,
        'image_path':  image_path,
        'prep':        prep_results,
        'seg':         seg_results,
        'meas':        meas_results,
        'prep_viz':    viz_prep_path,
        'seg_viz':     viz_seg_path,
        'meas_viz':    viz_meas_path,
        'len_viz':     viz_len_path,
        'orient_viz':  viz_orient_path,
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
        '--gap-closing-ksize',
        type=int,
        default=7,
        help='Kernel size for Step 4A morphological gap-closing (default: 7, 0 or 1 to disable)'
    )
    parser.add_argument(
        '--gap-closing-iters',
        type=int,
        default=1,
        help='Iterations for Step 4A morphological gap-closing (default: 1)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display visualization window (requires display)'
    )
    
    args = parser.parse_args()
    
    import glob
    import cv2

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def validate_image_path(path: str) -> str:
        """Normalise, validate existence, extension, and OpenCV readability.
        Returns the normalised path or calls sys.exit on error."""
        # Normalise: strip surrounding quotes that shells / users sometimes leave in
        path = path.strip().strip('"').strip("'")
        # Convert to OS-native separators so Windows paths work correctly
        path = os.path.normpath(path)

        if not os.path.isfile(path):
            print(f"\n[ERROR] File not found: {path}")
            print("        Please check the path and try again.")
            sys.exit(1)

        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            print(f"\n[ERROR] Unsupported file format: '{ext}'")
            print(f"        Supported formats: {', '.join(sorted(SUPPORTED_EXTS))}")
            sys.exit(1)

        img_check = cv2.imread(path)
        if img_check is None:
            print(f"\n[ERROR] OpenCV could not read the image: {path}")
            print("        The file may be corrupted or in an unsupported colour space.")
            sys.exit(1)

        return path

    # ------------------------------------------------------------------
    # Collect target images
    # ------------------------------------------------------------------
    target_images = []

    if args.all:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            target_images.extend(glob.glob(os.path.join("data/sample_images", ext)))
        target_images = sorted(list(set(target_images)))
        if not target_images:
            print("[ERROR] No images found in data/sample_images/")
            sys.exit(1)
        print(f"[INFO] Found {len(target_images)} image(s) to process.")

    elif args.image:
        # --- OPTION 1: path provided via --image / -i argument ---
        validated = validate_image_path(args.image)
        target_images = [validated]

    else:
        # --- OPTION 2: no argument — interactive prompt ---
        print()
        print("CrackGauge - Interactive Image Input")
        print("-" * 40)
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTS))}")
        print()
        raw_path = input("Enter input image path: ").strip()
        if not raw_path:
            print("\n[ERROR] No path entered. Exiting.")
            sys.exit(1)
        validated = validate_image_path(raw_path)
        target_images = [validated]

    if not target_images:
        print("[ERROR] No image found!")
        print("   Place a crack image in: data/sample_images/")
        print("   Or run: python main.py --image your_image.jpg")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Execute Pipeline on all target images
    # ------------------------------------------------------------------
    all_reports = []
    for img_path in target_images:
        print()
        print("=" * 60)
        print(f"[INPUT] Image: {img_path}")
        print("=" * 60)
        print("Step 1 --> Preprocessing")
        print("Step 2 --> Segmentation (Frangi / Hessian)")
        print("Step 3 --> Width Measurement")
        print("Step 4A --> Length Measurement")
        print("Step 4B --> Orientation (PCA)")
        print("=" * 60)
        res = process_single_image(img_path, args.output, args)
        all_reports.append(res)
        
    # --- Print Final Summary Report ---
    print("\n" + "="*125)
    print("  CrackGauge Pipeline Execution Summary (Steps 1, 2, 3, 4A, 4B)")
    print("="*125)
    print(f"{'Image Name':<20} | {'Crack %':<8} | {'Comps (Raw)':<13} | {'Total Length':<13} | {'Max Width':<10} | {'Dominant Angle':<15} | {'Classification':<18}")
    print("-" * 125)
    for rep in all_reports:
        name = rep['image_name']
        cp_pct = rep['seg']['crack_percent']
        tot_len = rep['meas']['total_length_px']
        n_comps = rep['meas']['component_count']
        n_raw = rep['meas'].get('raw_component_count', n_comps)
        comps_str = f"{n_comps} (raw {n_raw})" if n_raw != n_comps else f"{n_comps}"
        max_w = rep['meas']['max_width_px']
        dom_ang = rep['meas']['dominant_angle_deg']
        dom_type = rep['meas']['dominant_type']
        print(f"{name:<20} | {cp_pct:>7.2f}% | {comps_str:>13} | {tot_len:>10.1f} px | {max_w:>8.2f} px | {dom_ang:>12.1f} deg | {dom_type:<18}")
    print("="*125)
    print("  * Note: Width/Length in PIXELS, Orientation in DEGREES relative to horizontal [0, 180). Calibration pending Step 5.")
    
    # Detailed Component Breakdown (Length + PCA Orientation)
    print("\n" + "-"*85)
    print("  Per-Component Crack Measurement Breakdown (PIXELS & DEGREES):")
    print("-" * 85)
    for rep in all_reports:
        name = rep['image_name']
        comps = rep['meas']['components']
        n_raw = rep['meas'].get('raw_component_count', len(comps))
        raw_note = f" (reduced from {n_raw} raw fragments via gap-closing)" if n_raw != len(comps) else ""
        orients = {c['id']: c for c in rep['meas'].get('component_orientations', [])}
        print(f"\n  [{name}] - {len(comps)} component(s){raw_note}:")
        for c in comps:
            cid = c['id']
            clen = c['length_px']
            pts = c['skeleton_pixels']
            area = c['mask_area']
            if cid in orients:
                o_ang = f"{orients[cid]['angle_deg']:.1f} deg"
                o_type = f"({orients[cid]['orientation_type']})"
            else:
                o_ang = "N/A"
                o_type = "(Too small for PCA)"
            print(f"    - Comp #{cid:2d}: Length = {clen:>7.1f} px | PCA Angle = {o_ang:>9} {o_type:<20} | Skel = {pts:>4d} px | Area = {area:>4d} px")
    print("-" * 85)
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
        
    print(f"\n[SUCCESS] Pipeline complete! Output directory: {args.output}/")
    print("Steps verified: Step 1 (Preprocessing) -> Step 2 (Segmentation) -> Step 3 (Width) -> Step 4A (Length) -> Step 4B (Orientation).")



if __name__ == "__main__":
    main()

