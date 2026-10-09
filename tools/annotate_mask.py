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

import cv2
import numpy as np


WINDOW_NAME = "CrackGauge Manual Annotation"
MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800
MIN_BRUSH_SIZE = 1
MAX_BRUSH_SIZE = 101
BRUSH_STEP = 2


def parse_args() -> argparse.Namespace:
    """Parse annotation input and output paths."""
    parser = argparse.ArgumentParser(
        description="Manually draw a binary crack ground-truth mask."
    )
    parser.add_argument("--image", required=True, help="Original image path")
    parser.add_argument(
        "--output",
        required=True,
        help="Output binary PNG mask path",
    )
    return parser.parse_args()


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

    def __init__(self, image: np.ndarray, output_path: str) -> None:
        self.image = image
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.output_path = output_path
        self.display_width, self.display_height = display_dimensions(image)
        self.brush_size = 9
        self.drawing = False
        self.erase = False
        self.reset_pending = False
        self.status = "Draw cracks with left mouse button."

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
    """Load an image, run annotation, and save only manually drawn pixels."""
    args = parse_args()
    image = load_original_image(args.image)
    session = AnnotationSession(image, args.output)
    saved = session.run()
    if not saved:
        print("[INFO] Exited without saving an annotation mask.")


if __name__ == "__main__":
    main()
