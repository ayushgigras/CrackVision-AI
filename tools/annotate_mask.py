"""Interactive manual crack-mask annotation tool.

The tool starts with an empty mask and never reads prediction outputs.

Example:
    python tools/annotate_mask.py ^
        --image data/real_images/00001.jpg ^
        --output data/ground_truth/00001_mask.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np


WINDOW_NAME = "CrackGauge Manual Annotation"
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 101
BRUSH_STEP = 2


def parse_args() -> argparse.Namespace:
    """Parse annotation input and output paths and options."""
    parser = argparse.ArgumentParser(
        description="Manually draw or edit a binary crack ground-truth mask."
    )
    parser.add_argument("--image", required=True, help="Original image path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output binary PNG mask path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load and edit an existing mask at --output (fails if mask is corrupted or dimensions mismatch)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force overwrite of an existing mask at --output starting with a blank canvas",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the existing mask at --output against the image contract without launching the GUI",
    )
    return parser.parse_args()


def validate_mask_contract(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    allow_empty: bool = False,
) -> tuple[bool, str, dict[str, object]]:
    """Validate a ground-truth mask against CrackGauge's strict ground-truth contract.

    Contract requirements:
      1. Image and mask must be valid numpy ndarrays.
      2. Mask must be 2D single-channel uint8 matching image.shape[:2].
      3. Pixel values must be strictly binary: subset of {0, 255}.
      4. Unless allow_empty is True, mask must contain at least 1 crack pixel (non-empty).

    Returns:
      (is_valid, reason, stats_dict)
    """
    if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
        return False, "Image and mask must be numpy ndarrays", {}

    if mask.ndim != 2:
        return False, f"Mask must be 2D single-channel, got shape {mask.shape}", {}

    if image.shape[:2] != mask.shape:
        return (
            False,
            f"Image and ground-truth mask dimensions do not match (image is {image.shape[:2]}, mask is {mask.shape})",
            {},
        )

    unique_vals = set(np.unique(mask).tolist())
    if not unique_vals.issubset({0, 255}):
        return (
            False,
            f"Non-binary pixel values detected: {sorted(unique_vals)} (must be subset of {{0, 255}})",
            {"unique_values": sorted(unique_vals)},
        )

    crack_pixels = int(np.count_nonzero(mask > 0))
    total_pixels = int(mask.size)
    coverage_pct = (crack_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    stats: dict[str, object] = {
        "dimensions": mask.shape,
        "crack_pixels": crack_pixels,
        "total_pixels": total_pixels,
        "coverage_pct": round(coverage_pct, 4),
        "unique_values": sorted(unique_vals),
    }

    if crack_pixels == 0 and not allow_empty:
        return False, "Ground-truth mask contains 0 foreground crack pixels (empty mask)", stats

    return True, "", stats


def load_existing_mask(mask_path: str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
    """Safely load an existing binary mask, ensuring dimension and integrity match.

    Raises FileNotFoundError, ValueError, or OSError on any mismatch or corruption.
    Never silently fails or returns zeros on read errors.
    """
    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Existing mask not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode image file as grayscale mask: {path}")

    if mask.shape != expected_shape:
        raise ValueError(
            f"Existing mask dimensions {mask.shape} do not match expected image dimensions {expected_shape}"
        )

    # Strictly ensure binary values {0, 255}
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def load_original_image(image_path: str) -> np.ndarray:
    """Load and validate the original BGR image."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    return image


def display_dimensions(image: np.ndarray) -> tuple[int, int]:
    """Return display dimensions while preserving the original aspect ratio."""
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def save_binary_mask(mask: np.ndarray, output_path: str) -> None:
    """Save a strictly binary mask, creating its parent directory if needed."""
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    parent = Path(output_path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(output_path, binary_mask):
        raise OSError(f"Could not save annotation mask: {output_path}")


class AnnotationSession:
    """Manage display rendering, mouse coordinate mapping, and mask editing."""

    def __init__(
        self,
        image: np.ndarray,
        output_path: str,
        initial_mask: np.ndarray | None = None,
    ) -> None:
        self.image = image
        if initial_mask is not None:
            if initial_mask.shape != image.shape[:2]:
                raise ValueError("initial_mask dimensions do not match image")
            self.mask = np.where(initial_mask > 0, 255, 0).astype(np.uint8)
            crack_px = int(np.count_nonzero(self.mask > 0))
            self.status = f"Resumed mask ({crack_px:,} crack px). Left draw | Right erase."
        else:
            self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
            self.status = "Draw cracks with left mouse button."

        self.output_path = output_path
        self.display_width, self.display_height = display_dimensions(image)
        self.brush_size = 9
        self.drawing = False
        self.erase = False
        self.reset_pending = False

    def map_display_point(self, x: int, y: int) -> tuple[int, int]:
        """Map a display-window point to an original-image pixel."""
        original_height, original_width = self.mask.shape
        original_x = int(x * original_width / self.display_width)
        original_y = int(y * original_height / self.display_height)
        return (
            min(max(original_x, 0), original_width - 1),
            min(max(original_y, 0), original_height - 1),
        )

    def paint(self, x: int, y: int) -> None:
        """Paint or erase a brush stroke at a display coordinate."""
        original_x, original_y = self.map_display_point(x, y)
        color = 0 if self.erase else 255
        radius = max(0, self.brush_size // 2)
        cv2.circle(self.mask, (original_x, original_y), radius, color, -1)

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _data) -> None:
        """Handle drawing and erasing mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erase = False
            self.paint(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erase = True
            self.paint(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.paint(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False

    def render(self) -> np.ndarray:
        """Render the original image with the annotation overlay."""
        displayed = cv2.resize(
            self.image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        display_mask = cv2.resize(
            self.mask,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        overlay = displayed.copy()
        overlay[display_mask > 0] = (0, 0, 255)
        rendered = cv2.addWeighted(displayed, 0.7, overlay, 0.3, 0)
        cv2.putText(
            rendered,
            f"Brush: {self.brush_size}px | {self.status}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            "Left draw | Right erase | [ ] brush | S save | R reset | Y/N confirm | Q/Esc exit",
            (10, self.display_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return rendered

    def run(self) -> bool:
        """Run the GUI loop and return whether the mask was saved."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.display_width, self.display_height)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        saved = False
        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == ord("["):
                self.brush_size = max(MIN_BRUSH_SIZE, self.brush_size - BRUSH_STEP)
                self.status = f"Brush size: {self.brush_size}px"
            elif key == ord("]"):
                self.brush_size = min(MAX_BRUSH_SIZE, self.brush_size + BRUSH_STEP)
                self.status = f"Brush size: {self.brush_size}px"
            elif key == ord("r") and not self.reset_pending:
                self.reset_pending = True
                self.status = "Reset annotation? Press Y to confirm or N to cancel."
            elif self.reset_pending and key == ord("y"):
                self.mask.fill(0)
                self.reset_pending = False
                self.status = "Annotation reset."
            elif self.reset_pending and key == ord("n"):
                self.reset_pending = False
                self.status = "Reset cancelled."
            elif key == ord("s"):
                save_binary_mask(self.mask, self.output_path)
                saved = True
                self.status = f"Saved: {self.output_path}"
                print(f"[OK] Annotation mask saved: {self.output_path}")
            elif key in (ord("q"), 27):
                break

        cv2.destroyWindow(WINDOW_NAME)
        return saved


def main() -> None:
    """Load an image, run annotation or validation, and preserve strict mask integrity."""
    args = parse_args()
    image = load_original_image(args.image)

    # 1. Validation-only mode
    if args.validate_only:
        if not os.path.isfile(args.output):
            print(f"[FAIL] Mask file does not exist: {args.output}")
            sys.exit(1)
        mask = cv2.imread(args.output, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[FAIL] Could not decode mask file: {args.output}")
            sys.exit(1)
        is_valid, reason, stats = validate_mask_contract(image, mask)
        if is_valid:
            print(f"[OK] Mask contract VALID: {args.output}")
            print(f"     Dimensions : {stats['dimensions']}")
            print(f"     Crack Pixels: {stats['crack_pixels']:,} px ({stats['coverage_pct']}%)")
            print(f"     Values      : {stats['unique_values']}")
            sys.exit(0)
        else:
            print(f"[FAIL] Mask contract INVALID: {args.output}")
            print(f"       Reason: {reason}")
            if stats:
                print(f"       Stats : {stats}")
            sys.exit(1)

    # 2. Safe loading / Overwrite protection
    initial_mask = None
    if os.path.isfile(args.output):
        if args.resume:
            try:
                initial_mask = load_existing_mask(args.output, image.shape[:2])
                print(f"[OK] Resumed from existing mask: {args.output} ({int(np.count_nonzero(initial_mask > 0)):,} crack px)")
            except Exception as e:
                print(f"[ERROR] Failed to safely load existing mask for resume: {e}")
                print("[ABORT] Refusing to overwrite or corrupt existing file.")
                sys.exit(1)
        elif args.overwrite:
            print(f"[WARN] Overwriting existing mask at {args.output} with a fresh blank canvas.")
        else:
            print(f"[ERROR] Target mask file already exists: {args.output}")
            print("        To avoid accidental data loss, CrackGauge requires explicit intent:")
            print("        - Pass --resume to load and edit the existing mask.")
            print("        - Pass --overwrite to discard the existing mask and start fresh.")
            sys.exit(1)

    session = AnnotationSession(image, args.output, initial_mask=initial_mask)
    saved = session.run()
    if not saved:
        print("[INFO] Exited without saving an annotation mask.")


if __name__ == "__main__":
    main()
