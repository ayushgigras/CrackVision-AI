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
import csv
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
from src.segmentation import (
    SEGMENTATION_PROFILES,
    run_segmentation_pipeline,
    visualize_segmentation,
)
from src.measurement import (
    run_measurement_pipeline,
    visualize_measurement,
    visualize_length_measurement,
    visualize_orientation_measurement,
)
from src.calibration import (
    calibrate_measurements,
    detect_aruco_calibration,
    get_calibration_scale,
)


SUMMARY_FIELDS = [
    'image_filename',
    'crack_coverage_percent',
    'component_count',
    'total_length_px',
    'max_width_px',
    'mean_width_px',
    'dominant_orientation_deg',
    'status',
    'error_message',
]


def discover_images(input_dir: str, supported_exts: set[str]) -> list[str]:
    """Return readable-path candidates in an input directory, sorted by name."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name)[1].lower() in supported_exts
    )


def write_summary_csv(summary_path: str, rows: list[dict]) -> None:
    """Write one machine-readable summary row per attempted image."""
    parent = os.path.dirname(summary_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(summary_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, '') for field in SUMMARY_FIELDS} for row in rows)


def process_single_image(
    image_path: str,
    output_dir: str,
    args,
    save_legacy_visualization: bool = True,
):
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
    # Preserve the legacy output for single-image runs, but avoid replacing it
    # repeatedly during directory processing.
    if save_legacy_visualization:
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
        segmentation_profile=args.segmentation_profile,
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

    # --- Step 5: ArUco / Physical Scale Calibration ---
    aruco_calibration = {
        'is_calibrated': False,
        'status': 'marker_not_configured',
        'mm_per_pixel': None,
    }
    marker_size_mm = getattr(args, 'marker_size_mm', None)
    if marker_size_mm is not None:
        aruco_calibration = detect_aruco_calibration(
            prep_results['original'],
            marker_size_mm=marker_size_mm,
            dictionary_name=getattr(args, 'aruco_dictionary', 'DICT_4X4_50'),
        )
        if aruco_calibration['is_calibrated']:
            rectified_path = os.path.join(output_dir, f"{base_name}_aruco_rectified.png")
            import cv2
            cv2.imwrite(rectified_path, aruco_calibration['rectified_marker'])
            print(
                f"[OK] ArUco marker {aruco_calibration['marker_id']} detected: "
                f"{aruco_calibration['marker_width_px']:.2f} px = {marker_size_mm:.2f} mm"
            )
        else:
            print("[WARNING] Calibration marker not detected.")
            print("Width/Length reported in pixels only.")

    calibration_scale = aruco_calibration.get('mm_per_pixel')
    if calibration_scale is None and marker_size_mm is None:
        # Manual calibration is only used when automatic marker calibration
        # was not requested.
        calibration_scale = get_calibration_scale(
            mm_per_pixel=getattr(args, 'mm_per_pixel', None),
            reference_length_mm=getattr(args, 'reference_length_mm', None),
            reference_length_px=getattr(args, 'reference_length_px', None),
        )
    if calibration_scale is not None:
        meas_results = calibrate_measurements(meas_results, calibration_scale)
        if marker_size_mm is not None:
            meas_results['calibration'].update(aruco_calibration)
        else:
            meas_results['calibration']['status'] = 'manual_scale'
        print(f"[OK] Step 5 calibration applied: {calibration_scale:.6f} mm/px")
    else:
        meas_results['calibration'] = aruco_calibration
    
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
        '--input-dir',
        type=str,
        default=None,
        help='Process all supported images found in this directory'
    )
    parser.add_argument(
        '--summary-csv',
        type=str,
        default=None,
        help='Write per-image batch results to this CSV file'
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
        '--segmentation-profile',
        type=str,
        choices=SEGMENTATION_PROFILES,
        default='baseline',
        help='Segmentation profile: baseline, adaptive_recovery, or guided_recovery'
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
    parser.add_argument(
        '--mm-per-pixel',
        type=float,
        default=None,
        help='Direct physical calibration scale in millimetres per pixel'
    )
    parser.add_argument(
        '--reference-length-mm',
        type=float,
        default=None,
        help='Physical length of known reference object in millimetres'
    )
    parser.add_argument(
        '--reference-length-px',
        type=float,
        default=None,
        help='Pixel length of known reference object in the image'
    )
    parser.add_argument(
        '--marker-size-mm',
        type=float,
        default=None,
        help='Physical side length of the ArUco marker in millimetres'
    )
    parser.add_argument(
        '--aruco-dictionary',
        choices=['DICT_4X4_50', 'DICT_5X5_50', 'DICT_6X6_50', 'DICT_7X7_50'],
        default='DICT_4X4_50',
        help='Predefined ArUco dictionary used for calibration'
    )
    
    args = parser.parse_args()
    try:
        get_calibration_scale(
            mm_per_pixel=args.mm_per_pixel,
            reference_length_mm=args.reference_length_mm,
            reference_length_px=args.reference_length_px,
        )
    except ValueError as error:
        parser.error(str(error))
    
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

    batch_mode = args.input_dir is not None or args.all
    if args.input_dir:
        try:
            target_images = discover_images(args.input_dir, SUPPORTED_EXTS)
        except FileNotFoundError as error:
            parser.error(str(error))
        if not target_images:
            parser.error(f"No supported images found in {args.input_dir}")
        print(f"[INFO] Found {len(target_images)} image(s) in {args.input_dir}.")
    elif args.all:
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
        print("Step 5 --> Calibration (ArUco / Physical Scale)")
        print("=" * 60)
        image_name = os.path.basename(img_path)
        try:
            if batch_mode:
                import cv2
                if not os.path.isfile(img_path):
                    raise FileNotFoundError(f"File not found: {img_path}")
                if cv2.imread(img_path) is None:
                    raise ValueError(f"OpenCV could not read the image: {img_path}")
            res = process_single_image(
                img_path,
                args.output,
                args,
                save_legacy_visualization=not batch_mode,
            )
            all_reports.append(res)
        except Exception as error:
            if not batch_mode:
                raise
            print(f"[WARNING] Skipping {image_name}: {error}")
            all_reports.append({
                'image_name': os.path.splitext(image_name)[0],
                'image_path': img_path,
                'error_message': str(error),
            })

    if args.summary_csv:
        summary_rows = []
        for rep in all_reports:
            if 'error_message' in rep:
                summary_rows.append({
                    'image_filename': os.path.basename(rep['image_path']),
                    'status': 'failed',
                    'error_message': rep['error_message'],
                })
                continue
            summary_rows.append({
                'image_filename': os.path.basename(rep['image_path']),
                'crack_coverage_percent': rep['seg']['crack_percent'],
                'component_count': rep['meas']['component_count'],
                'total_length_px': rep['meas']['total_length_px'],
                'max_width_px': rep['meas']['max_width_px'],
                'mean_width_px': rep['meas']['mean_width_px'],
                'dominant_orientation_deg': rep['meas']['dominant_angle_deg'],
                'status': 'success',
                'error_message': '',
            })
        write_summary_csv(args.summary_csv, summary_rows)
        print(f"[OK] Batch summary CSV saved to: {args.summary_csv}")

    if not all_reports:
        print("[ERROR] No images were processed successfully.")
        return

    successful_reports = [report for report in all_reports if 'error_message' not in report]
    if not successful_reports:
        print("[ERROR] No images were processed successfully.")
        return
        
    # --- Print Final Summary Report ---
    print("\n" + "="*125)
    print("  CrackGauge Pipeline Execution Summary (Steps 1, 2, 3, 4A, 4B, 5)")
    print("="*125)
    print(f"{'Image Name':<20} | {'Crack %':<8} | {'Comps (Raw)':<13} | {'Total Length':<13} | {'Max Width':<10} | {'Dominant Angle':<15} | {'Classification':<18}")
    print("-" * 125)
    for rep in successful_reports:
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
        if 'total_length_mm' in rep['meas']:
            print(f"{'':<20} | {'':<8} | {'':<13} | {rep['meas']['total_length_mm']:>10.2f} mm | {rep['meas']['max_width_mm']:>8.2f} mm | {'':>12} | calibrated")
    print("="*125)
    calibration_info = successful_reports[0]['meas'].get('calibration', {})
    calibration_active = calibration_info.get('is_calibrated', False)
    if calibration_active:
        scale = calibration_info['mm_per_pixel']
        marker_id = calibration_info.get('marker_id')
        if marker_id is not None:
            print(f"  * Calibration: ArUco marker {marker_id} | {scale:.6f} mm/px | Width/Length available in millimetres.")
        else:
            print(f"  * Calibration: manual scale | {scale:.6f} mm/px | Width/Length available in millimetres.")
    else:
        print("  * Calibration: unavailable | Width/Length reported in pixels only.")
    
    # Detailed Component Breakdown (Length + PCA Orientation)
    print("\n" + "-"*85)
    print("  Per-Component Crack Measurement Breakdown (PIXELS, MILLIMETRES & DEGREES):")
    print("-" * 85)
    for rep in successful_reports:
        name = rep['image_name']
        comps = rep['meas']['components']
        n_raw = rep['meas'].get('raw_component_count', len(comps))
        raw_note = f" (reduced from {n_raw} raw fragments via gap-closing)" if n_raw != len(comps) else ""
        orients = {c['id']: c for c in rep['meas'].get('component_orientations', [])}
        print(f"\n  [{name}] - {len(comps)} component(s){raw_note}:")
        for c in comps:
            cid = c['id']
            clen = c['length_px']
            clen_mm = c.get('length_mm')
            pts = c['skeleton_pixels']
            area = c['mask_area']
            if cid in orients:
                o_ang = f"{orients[cid]['angle_deg']:.1f} deg"
                o_type = f"({orients[cid]['orientation_type']})"
            else:
                o_ang = "N/A"
                o_type = "(Too small for PCA)"
            length_text = f"{clen:>7.1f} px"
            if clen_mm is not None:
                length_text += f" / {clen_mm:>7.2f} mm"
            print(f"    - Comp #{cid:2d}: Length = {length_text} | PCA Angle = {o_ang:>9} {o_type:<20} | Skel = {pts:>4d} px | Area = {area:>4d} px")
    print("-" * 85)
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
        
    print(f"\n[SUCCESS] Pipeline complete! Output directory: {args.output}/")
    print("Steps verified: Step 1 (Preprocessing) -> Step 2 (Segmentation) -> Step 3 (Width) -> Step 4A (Length) -> Step 4B (Orientation) -> Step 5 (ArUco Calibration).")



if __name__ == "__main__":
    main()
