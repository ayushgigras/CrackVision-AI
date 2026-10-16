# CrackGauge Ground-Truth Annotation & Validation Guideline

This document defines the ground-truth annotation specification, file conventions, validation contract, and evaluation procedures for the CrackGauge concrete inspection system.

---

## 1. Ground-Truth Mask Contract

To ensure scientific validity and reproducibility, every ground-truth mask must strictly adhere to the following contract:

1. **Exact 1-to-1 File Naming & Location**:
   - Source Image: `data/real_images/<stem>.<ext>` (e.g. `00001.jpg`, `00055.jpg`)
   - Ground-Truth Mask: `data/ground_truth/<stem>_mask.png` (e.g. `00001_mask.png`, `00055_mask.png`)
   - The stem must match after stripping `_mask`. Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
2. **Dimension Matching**:
   - The mask dimensions $(H, W)$ must be bitwise identical to the source image height and width.
   - Any shape mismatch will cause the evaluation pipeline to reject the pair with an explicit warning.
3. **Encoding & Channel Depth**:
   - Format: Single-channel 8-bit PNG (`cv2.IMREAD_GRAYSCALE`, `dtype=uint8`, `ndim=2`).
4. **Strict Binary Conventions**:
   - `0` (Black): Background concrete, aggregate, shadows, surface blemishes, or intact substrate.
   - `255` (White): Foreground crack fissure pixels.
   - No anti-aliased or intermediate gray values (e.g., 1–254) are permitted.
5. **Foreground Integrity**:
   - A valid ground-truth mask must contain at least 1 crack pixel ($\sum > 0$), unless explicitly run in empty-mask mode.
6. **Independence Principle**:
   - Ground-truth masks must be manually drawn or reviewed by human visual inspection from the original raw image.
   - Masks must **never** be generated or seeded from algorithmic predictions (Otsu, Frangi, or recovery masks), as doing so introduces circular evaluation bias.

---

## 2. Interactive Annotation Tool (`tools/annotate_mask.py`)

CrackGauge provides an interactive OpenCV annotation tool that operates directly on the original image canvas.

### Starting a New Annotation
```powershell
python tools/annotate_mask.py `
  --image data/real_images/00055.jpg `
  --output data/ground_truth/00055_mask.png
```
*Note: If the target output file already exists, the tool will refuse to overwrite it without an explicit flag to prevent accidental data loss.*

### Safely Resuming / Editing an Existing Mask
To continue work on a previously saved mask:
```powershell
python tools/annotate_mask.py `
  --image data/real_images/00055.jpg `
  --output data/ground_truth/00055_mask.png `
  --resume
```
The tool validates the existing mask's dimensions and binary integrity before loading. If loading fails or dimensions mismatch, the tool immediately aborts without modifying the file.

### Overwriting an Existing Mask
To discard an existing mask and start fresh:
```powershell
python tools/annotate_mask.py `
  --image data/real_images/00055.jpg `
  --output data/ground_truth/00055_mask.png `
  --overwrite
```

### Standalone Mask Contract Verification
To verify that an existing mask satisfies the contract without opening a GUI window:
```powershell
python tools/annotate_mask.py `
  --image data/real_images/00001.jpg `
  --output data/ground_truth/00001_mask.png `
  --validate-only
```
Output:
```text
[OK] Mask contract VALID: data/ground_truth/00001_mask.png
     Dimensions : (227, 227)
     Crack Pixels: 3,127 px (6.0684%)
     Values      : [0, 255]
```

### GUI Interactive Controls
- **Left Mouse Button (Click & Drag)**: Draw white crack pixels (`255`).
- **Right Mouse Button (Click & Drag)**: Erase pixels back to background (`0`).
- **`[` / `]`**: Decrease / increase brush diameter (step: 2 px, range: 1–101 px).
- **`s`**: Save the binary PNG mask to disk.
- **`r` then `y`**: Reset/clear the canvas after confirmation (`n` to cancel).
- **`q` or `Esc`**: Exit the tool.

---

## 3. Dataset Evaluation Pipeline (`tools/evaluate_validation_dataset.py`)

The evaluation pipeline automatically pairs images in `data/real_images/` with masks in `data/ground_truth/`.

### Evaluation Modes

#### Mode 1: Standard Profiles (`--mode profiles`, Default)
Evaluates production and candidate profiles (`baseline`, `adaptive_recovery`, `guided_recovery`, `hysteresis_recovery`):
```powershell
python tools/evaluate_validation_dataset.py --mode profiles
```

#### Mode 2: Hysteresis Strategies (`--mode hysteresis_strategies`)
Compares the baseline against the three isolated hysteresis filtering strategies on identical preprocessed inputs:
- **`baseline`**: Standard Frangi ridge filter + Otsu binarization.
- **`strategy_A_min_area_200`**: Hysteresis low=25, high=48, $min\_area = 200$ px (area cutoff only).
- **`strategy_B_min_area_80`**: Hysteresis low=25, high=48, $min\_area = 80$ px (area cutoff only).
- **`strategy_C_shape_filter`**: Hysteresis low=25, high=48, $min\_area = 80$ px + conservative oriented aspect ratio & skeleton length filtering with boundary relaxation.

Command:
```powershell
python tools/evaluate_validation_dataset.py `
  --mode hysteresis_strategies `
  --output-csv data/outputs/multi_image_validation_strategies.csv
```

---

## 4. Metric Aggregation & Scientific Integrity

1. **Summed Pixel Confusion Matrices (Micro-Averaging)**:
   - Aggregate metrics (Precision, Recall, F1/Dice, and IoU) are computed by summing integer `TP`, `FP`, `FN`, and `TN` counts across all valid evaluated pairs, rather than averaging per-image percentage scores.
   - This ensures larger cracks contribute proportionally to the aggregate benchmark and avoids skew from tiny fragments.
2. **Strict Handling of Unannotated Images**:
   - Discovered images that lack matching ground-truth masks are recorded with `status: skipped` and reason `"Matching ground-truth mask not found"`.
   - Skipped candidates are **never** treated as negative examples or assigned accuracy scores.
   - With 1 annotated image (`00001.jpg`), reported aggregate metrics reflect exactly 1 image-mask pair. Dataset-wide accuracy must never be claimed for unannotated images.
3. **Empty Mask Protection**:
   - By default, ground-truth masks with 0 foreground crack pixels are flagged with a warning and skipped (`"Ground-truth mask contains 0 foreground crack pixels (empty mask)"`).
   - If evaluation of confirmed negative control images is required, pass `--allow-empty-masks`.
