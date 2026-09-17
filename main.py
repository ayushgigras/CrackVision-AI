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

# Add src/ to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import run_preprocessing_pipeline, visualize_results


def main():
    parser = argparse.ArgumentParser(
        description="CrackGauge — Computer Vision Crack Measurement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --image data/sample_images/crack1.jpg
  python main.py --image data/sample_images/crack1.jpg --show
        """
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        default=None,
        help='Path to the input crack image'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/outputs',
        help='Directory to save output images (default: data/outputs)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display visualization window (requires display)'
    )
    
    args = parser.parse_args()
    
    # --- Find input image ---
    image_path = args.image
    
    if image_path is None:
        # Auto-search in sample_images
        import glob
        patterns = [
            "data/sample_images/*.jpg",
            "data/sample_images/*.jpeg",
            "data/sample_images/*.png",
            "data/sample_images/*.bmp",
        ]
        for pattern in patterns:
            found = glob.glob(pattern)
            if found:
                image_path = found[0]
                print(f"[Auto] Using image: {image_path}")
                break
    
    if image_path is None:
        print("❌ No image found!")
        print("   Place a crack image in:  data/sample_images/")
        print("   Or run: python main.py --image your_image.jpg")
        sys.exit(1)
    
    # --- Run Pipeline ---
    results = run_preprocessing_pipeline(
        image_path=image_path,
        output_dir=args.output,
        save_steps=True
    )
    
    # --- Visualize ---
    viz_path = os.path.join(args.output, "pipeline_visualization.png")
    visualize_results(results, save_path=viz_path)
    
    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Results saved to: {args.output}/")
    print(f"   Files created:")
    for f in os.listdir(args.output):
        size = os.path.getsize(os.path.join(args.output, f))
        print(f"     • {f}  ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
