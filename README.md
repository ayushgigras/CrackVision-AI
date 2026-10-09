# CrackGauge 🔍
**Computer Vision Based Structural Concrete Crack Measurement Gauge**

> 5th Semester Mini Project — BTCSE  
> Team: Manish Saini | Sachin Chauhan | Ayush Gigras  
> Mentor: Ms. Anuska Shukla

---

## 📌 What is CrackGauge?

CrackGauge is a low-cost, portable, camera-based system that:
1. Captures concrete surface images
2. Detects and segments crack regions (Computer Vision)
3. Measures crack **width**, **length**, and **orientation**
4. Converts pixel measurements to **millimeters** via calibration
5. Generates a **digital inspection report** (PDF)

---

## 🏗️ Project Phases

| Phase | Status | Description |
|-------|--------|-------------|
| P1 | ✅ Done (Presentation) | Ideation, architecture, PPT |
| P2 | 🔄 **Current** | Crack Segmentation (preprocessing pipeline) |
| P3 | ⏳ Upcoming | Width/Length/Orientation measurement |
| P4 | ⏳ Upcoming | Physical calibration (pixels → mm) |
| P5 | ⏳ Upcoming | Raspberry Pi prototype |
| P6 | ⏳ Upcoming | Dashboard + Database |
| P7 | ⏳ Upcoming | Validation & testing |

---

## 📁 Project Structure

```
CrackGauge/
├── src/                    # Source code
│   ├── preprocessing.py    # Step 1: Image preprocessing pipeline
│   ├── segmentation.py     # Step 2: Crack segmentation
│   ├── measurement.py      # Step 3: Width/Length/Orientation
│   ├── calibration.py      # Step 5: Pixel -> mm conversion
│   └── report.py           # Step 6: PDF report generation
├── data/
│   ├── sample_images/      # Input crack images
│   └── outputs/            # Pipeline output images
├── notebooks/              # Jupyter exploration notebooks
├── tests/                  # Unit tests
├── docs/                   # Project documents
│   ├── CrackGauge_Presentation.pptx
│   └── miniprojectsynopsisss.docx
├── main.py                 # Entry point - run full pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone / navigate to repo
```bash
cd d:\MINI_PROJECT_2\CrackGauge
```

### 2. Activate virtual environment
```bash
# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run pipeline
```bash
python main.py --image data/sample_images/crack1.jpg

# Direct calibration scale
python main.py --image data/sample_images/crack1.jpg --mm-per-pixel 0.05

# Reference-based calibration (known reference length / measured pixels)
python main.py --image data/sample_images/crack1.jpg --reference-length-mm 100 --reference-length-px 820

# Automatic ArUco calibration (marker must be visible in the image)
python main.py --image data/sample_images/crack1.jpg --marker-size-mm 50

# Preserve the current Frangi-primary segmentation (default)
python main.py --image data/real_images/00001.jpg --segmentation-profile baseline

# Recover the existing cleaned adaptive-threshold mask
python main.py --image data/real_images/00001.jpg --segmentation-profile adaptive_recovery

# Conservatively add adaptive components near Frangi detections
python main.py --image data/real_images/00001.jpg --segmentation-profile guided_recovery
```

Segmentation profiles do not change the default pipeline: `baseline` is the
default and preserves the existing Frangi-primary behavior. `adaptive_recovery`
uses the already-computed, cleaned adaptive-threshold mask. `guided_recovery`
keeps the baseline Frangi mask and adds only adaptive connected components that
overlap a small dilation of a Frangi detection. These profiles are alternatives
for comparison, not claims of general accuracy; evaluate them on multiple
independently annotated images before selecting a production default.

For automatic calibration, place a printed ArUco marker in the same plane as
the concrete surface and pass its measured physical side length with
`--marker-size-mm`. The default dictionary is `DICT_4X4_50`; select another
supported dictionary with `--aruco-dictionary`. If the marker is not detected,
the pipeline does not calculate a scale and reports width and length in pixels
only.

### Validation against a ground-truth mask

Validation is separate from the production pipeline and requires a supplied
ground-truth binary mask. Non-zero pixels are treated as crack pixels. It does
not treat the synthetic input images or generated prediction masks as
ground truth.

Run segmentation validation from the repository root:

```bash
python -m src.validation ^
  --prediction data/outputs/crack_synthetic_01_08_final_crack_mask.jpg ^
  --ground-truth path/to/ground_truth_mask.png ^
  --output-csv data/outputs/validation_metrics.csv
```

The prediction and ground-truth masks must have the same width and height.
They must be single-channel image files readable by OpenCV, such as PNG,
JPEG, or BMP. Non-zero pixels are treated as crack pixels and zero pixels as
background. The ground-truth mask must be manually annotated or otherwise
independently measured; synthetic input images and generated prediction masks
are not ground truth.

The command prints the segmentation metrics as JSON and writes one CSV row to
`data/outputs/validation_metrics.csv` by default. Use `--output-csv` to choose
another path. The CSV contains IoU, Dice, precision, recall, F1, and confusion
counts. No accuracy percentage is invented when a ground-truth mask is absent.

Optional width and length references can be included when both the predicted
and independently measured reference values are known:

```bash
python -m src.validation ^
  --prediction path/to/predicted_mask.png ^
  --ground-truth path/to/ground_truth_mask.png ^
  --predicted-width 12.0 ^
  --reference-width 10.0 ^
  --predicted-length 90.0 ^
  --reference-length 100.0 ^
  --measurement-unit px ^
  --output-csv data/outputs/validation_metrics.csv
```

Use `--measurement-unit mm` only when both predicted and reference
measurements are calibrated millimetre values. Use `px` for raw pixel
measurements; the validation command does not perform calibration. Width and
length error columns contain absolute error in the selected unit and relative
error percentage. A zero reference has an undefined relative percentage,
stored as an empty CSV field. The `--predicted-*` and `--reference-*` options
must be supplied as pairs.

The reusable Python APIs are in `src.validation`, including
`segmentation_metrics` for masks and `aggregate_measurement_errors` for
paired width or length reference measurements.

### Batch processing

To process every supported image in `data/real_images/` and write a
machine-readable summary:

```powershell
python main.py `
  --input-dir data\real_images `
  --output data\outputs `
  --summary-csv data\outputs\batch_summary.csv
```

Supported input extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.
Each successful row contains crack coverage, component count, total length,
maximum and mean width, and dominant orientation in pixels/degrees. Unreadable
or failed images are not allowed to stop the batch; they receive
`status=failed` and an `error_message` in the CSV. Per-image visualization
filenames are preserved, and the generic `pipeline_visualization.png` is not
rewritten during directory processing.

The profile can also be used during batch processing:

```powershell
python main.py `
  --input-dir data\real_images `
  --segmentation-profile adaptive_recovery `
  --summary-csv data\outputs\adaptive_batch_summary.csv
```

The summary CSV schema is unchanged. It reports measurements generated from the
selected segmentation profile.

### Comparing profiles against an annotation

For the annotated `00001` example, run:

```powershell
python tools\compare_segmentation_profiles.py
```

This runs the `baseline` and `adaptive_recovery` profiles against
`data/ground_truth/00001_mask.png`, prints TP, FP, FN, precision, recall,
Dice/F1, and IoU for both, and saves:

```text
data/outputs/00001_segmentation_profiles.png
```

The comparison is a single-image diagnostic. It must not be interpreted as
real-world accuracy without additional independently annotated images.

### Multi-image validation

Manual masks follow this exact filename convention:

```text
data/real_images/00001.jpg
data/ground_truth/00001_mask.png
```

The image stem must match the mask stem after removing `_mask`. Supported
extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

Run the multi-image evaluator from the repository root:

```powershell
python tools\evaluate_validation_dataset.py
```

The default command discovers images in `data\real_images`, matching masks in
`data\ground_truth`, evaluates `baseline`, `adaptive_recovery`, and
`guided_recovery`, and writes:

```text
data/outputs/multi_image_validation.csv
```

The CSV contains one row per evaluated image/profile and explicit `skipped`
rows for images with missing or unreadable masks, unreadable images, dimension
mismatches, or evaluation failures. Skipped candidates are never treated as
negative examples. The command prints aggregate TP, FP, FN, precision, recall,
Dice/F1, and IoU for each profile across valid evaluated pairs.

Use custom directories or an output path when needed:

```powershell
python tools\evaluate_validation_dataset.py `
  --image-dir data\real_images `
  --ground-truth-dir data\ground_truth `
  --output-csv data\outputs\multi_image_validation.csv
```

The reported sample count is the number of valid image-mask pairs, not the
number of images discovered. With only one annotated image, the aggregate
results are one-image results and must not be described as dataset-wide
accuracy. Add more independently annotated masks before making dataset-level
claims.

### Manual ground-truth mask annotation

The project includes a standalone OpenCV annotation tool for creating
manually labeled masks. It starts with an empty mask and never uses a
predicted segmentation to fill or generate the annotation.

For the first real image, run this from the repository root:

```powershell
python tools\annotate_mask.py `
  --image data\real_images\00001.jpg `
  --output data\ground_truth\00001_mask.png
```

The tool opens the original image in a window. The output mask keeps the
original image width and height and contains only:

```text
0   = background
255 = manually annotated crack
```

Controls:

- Left mouse button: draw white crack pixels
- Right mouse button: erase pixels back to black
- `[` / `]`: decrease / increase brush size
- `s`: save the binary PNG mask
- `r`, then `y`: clear the annotation after confirmation
- `r`, then `n`: cancel reset
- `q` or `Esc`: exit

The display may be resized to fit the screen, but mouse coordinates are mapped
back to the original image dimensions before painting. Saving creates the
output directory if necessary. This tool requires a desktop session with GUI
support and does not modify the original image or the detection pipeline.


---

## 🔬 Vision Pipeline

```
Input Image
    ↓
Grayscale Conversion
    ↓
Noise Reduction (Gaussian Blur)
    ↓
Contrast Enhancement (CLAHE)
    ↓
Adaptive Thresholding
    ↓
Morphological Processing (cleanup)
    ↓
Crack Mask (Binary)
    ↓
[Width | Length | Orientation]
    ↓
Digital Report (PDF)
```

---

## 👥 Team Roles

| Member | Role |
|--------|------|
| **Manish Saini** | Detection & Segmentation accuracy |
| **Sachin Chauhan** | Measurement accuracy (width/length) |
| **Ayush Gigras** | Hardware, calibration, field testing |

---

## 📚 Tech Stack
- **Python 3.13** — Core language
- **OpenCV** — Image processing & computer vision
- **scikit-image** — Skeletonization, morphology
- **NumPy / SciPy** — Numerical computing
- **Matplotlib** — Visualization
- **ReportLab** — PDF report generation
- **Raspberry Pi 4** — Hardware platform (Phase 5)
