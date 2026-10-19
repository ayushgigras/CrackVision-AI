# CrackGauge

**Automated Computer Vision System for Structural Concrete Crack Detection, Segmentation, and Geometric Profiling**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8.svg)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF.svg)](https://vite.dev/)
[![Tests Passing](https://img.shields.io/badge/pytest-129%20passed-brightgreen.svg)](tests/)

---

## Overview

**CrackGauge** is a modular, deterministic computer-vision system engineered to detect, segment, and quantify surface cracks in concrete civil infrastructure (bridges, retaining walls, pavements, and building structures).

Rather than relying on opaque deep-learning models that require large labeled training corpuses and GPU infrastructure, CrackGauge implements an explainable classical computer-vision pipeline. By combining multi-scale **Hessian matrix ridge filtering (Frangi filter)**, **topological skeletonization**, and **Euclidean distance transforms**, the system delivers reproducible geometric measurements (width variation, continuous curve length, and spatial orientation) alongside physical pixel-to-millimetre calibration.

The project exposes a **headless Python service**, an interactive **FastAPI REST API**, and a mobile-ready **React 19 / Vite single-page dashboard** for live camera capture and field inspection.

---

## Key Features

- **Explainable Vision Pipeline:** Multi-stage image preprocessing incorporating Gaussian smoothing ($9\times 9$), Contrast Limited Adaptive Histogram Equalization (CLAHE), and locally adaptive Gaussian thresholding to isolate fine fissures against noisy aggregate concrete textures.
- **Multi-Scale Ridge Filtering:** Evaluates the eigenvalues of the local Hessian matrix across configurable scales ($\sigma \in \{1, 2, 3, 4\}$) to isolate tubular fissure structures from surface blemishes and shadows.
- **Selectable Segmentation Profiles:** Supports 4 decoupled segmentation strategies (`baseline`, `adaptive_recovery`, `guided_recovery`, and `hysteresis_recovery`) without altering the default Frangi-primary execution path.
- **Medial Axis Width Profiling:** Extracts the topological medial axis via Zhang-Suen skeletonization and maps $L_2$ Euclidean distance transforms to calculate local width ($w = 2 \times \text{distance}$) at every skeleton pixel, reporting maximum, mean, median, and standard deviation.
- **Continuous Length Traversal:** Applies morphological gap closing ($7\times 7$ ellipse) to reconnect micro-fragmented crack segments and calculates true curve length via undirected 8-connected graph traversal.
- **PCA Dominant Orientation:** Extracts 2D coordinate point clouds of segmented crack pixels and computes the principal eigenvector via OpenCV `PCACompute2` to measure dominant orientation angle in degrees ($0^\circ - 180^\circ$), categorized by angular thresholds into Horizontal ($\le 15^\circ$ or $\ge 165^\circ$), Vertical ($75^\circ - 105^\circ$), or Diagonal / Oblique.
- **Multi-Source Calibration:** Computes physical millimetres per pixel ($mm/px$) via direct scale factor specification, reference length ratios ($L_{\text{ref, mm}} / L_{\text{ref, px}}$), or automated OpenCV ArUco fiducial marker corner detection.
- **Modern Decoupled Architecture:** Core vision routines operate as pure, headless functions in `src/pipeline_service.py` returning JSON-safe metrics and base64 PNG data URLs, consumed independently by CLI scripts or the web application.
- **Interactive Web Interface:** Mobile-first React dashboard supporting device camera capture (`capture="environment"`), image gallery uploads, calibration configuration, real-time metric cards, and multi-tab visual output inspection.
- **Batch Processing & Tooling:** Includes a robust batch directory processor with fault tolerance and CSV export, a standalone OpenCV interactive ground-truth mask annotation GUI, and a dataset evaluation harness.
- **Comprehensive Automated Test Suite:** 129 unit and integration tests passing across 11 test modules with zero failures, covering the REST API, pipeline service, geometric measurements, and binarization strategies.

---

## System Architecture

```text
[ Concrete Surface Image ] (.jpg / .png / .webp)
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Preprocessing (src/preprocessing.py)                     │
│    • Grayscale conversion & Gaussian spatial blur (9x9)     │
│    • Local contrast enhancement (CLAHE: clip=3.0, grid=8x8) │
│    • Locally adaptive Gaussian thresholding (block=21, C=5) │
│    • Morphological cleanup (Opening 3x3, Closing 7x7)       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Crack Segmentation (src/segmentation.py)                 │
│    • Multi-scale Frangi Hessian ridge filter (σ=1, 2, 3, 4) │
│    • Ridge response normalization (0-255 uint8)             │
│    • Binarization: Otsu / Cutoff / Dual-threshold Hysteresis│
│    • Component area & shape aspect-ratio filtering          │
│    • Selectable profiles: baseline, adaptive, guided,       │
│      hysteresis recovery                                    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Geometric Quantification (src/measurement.py)            │
│    • Medial axis skeletonization (Zhang-Suen)               │
│    • L2 Euclidean distance transform (cv2.distanceTransform)│
│    • Local width calculation: width = 2.0 * distance        │
│    • Morphological gap closing & continuous skeleton re-eval│
│    • 8-connected graph traversal for total path length      │
│    • PCA eigenvector analysis for dominant orientation &    │
│      angular category                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Physical Scale Calibration (src/calibration.py)          │
│    • Direct scale input (--mm-per-pixel)                    │
│    • Reference ratio (--reference-length-mm / -px)          │
│    • Automated ArUco fiducial marker corner detection       │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│ CLI & Batch Processing       │    │ Service & Web Interface      │
│ • main.py (--image / --all)  │    │ • Headless pipeline_service  │
│ • batch_summary.csv export   │    │ • FastAPI backend (src/api)  │
│ • tools/evaluate_validation  │    │ • React 19 / Vite Dashboard  │
└──────────────────────────────┘    └──────────────────────────────┘
```

---

## Repository Structure

```text
CrackGauge/
├── src/                                  # Core computer vision & service library
│   ├── __init__.py                       # Package exports and __version__ = "0.1.0"
│   ├── preprocessing.py                  # Step 1: Grayscale, Gaussian blur, CLAHE, adaptive thresholding
│   ├── segmentation.py                   # Step 2: Frangi ridge filtering, hysteresis, profile selection
│   ├── measurement.py                    # Step 3 & 4: Skeleton, distance transform, length, PCA orientation
│   ├── calibration.py                    # Step 5: Direct, reference, and ArUco marker scaling
│   ├── pipeline_service.py               # In-memory pipeline service decoupling vision logic from CLI
│   ├── api.py                            # FastAPI REST application exposing /api/health and /api/analyze
│   ├── validation.py                     # Mathematical evaluation of confusion matrices, IoU, Dice, errors
│   ├── download_samples.py               # Utility to retrieve sample concrete images
│   └── generate_synthetic_samples.py     # Procedural concrete texture and synthetic crack generator
│
├── frontend/                             # Single-page web inspection application
│   ├── index.html                        # HTML entry point with mobile viewport meta tags
│   ├── package.json                      # React 19, Vite, and frontend toolchain dependencies
│   ├── vite.config.js                    # Vite bundler configuration with backend reverse proxy
│   ├── src/
│   │   ├── main.jsx                      # React application mount point
│   │   ├── App.jsx                       # Master dashboard component (camera, upload, results viewer)
│   │   ├── App.css                       # Application stylesheet with dark modern UI theme
│   │   ├── index.css                     # Global reset, typography, and CSS design system tokens
│   │   └── assets/                       # UI static vector and image assets
│   └── dist/                             # [Generated build artifact, gitignored] Compiled production bundle
│
├── tests/                                # Automated Pytest test suite (129 tests)
│   ├── test_api.py                       # FastAPI endpoint integration & error handling tests
│   ├── test_batch_processing.py          # Batch directory scanning and summary CSV export tests
│   ├── test_calibration.py               # Scaling math, reference length, and ArUco detection tests
│   ├── test_hysteresis_segmentation.py   # Dual-threshold hysteresis & shape filtering unit tests
│   ├── test_length_measurement.py        # 8-connected skeleton graph traversal length tests
│   ├── test_measurement.py               # Distance transform width profiling & subtest thickness tests
│   ├── test_multi_image_validation.py    # Image-mask pairing, contract validation, and skip tests
│   ├── test_pipeline_service.py          # In-memory service validation, NaN/Inf safety, and data URL tests
│   ├── test_segmentation_profiles.py     # Profile isolation and mask contract validation tests
│   ├── test_validation.py                # IoU, Dice, precision, recall, and error math tests
│   └── test_validation_visualization.py  # Diagnostic false-color comparison overlay tests
│
├── tools/                                # Developer CLI evaluation and annotation tools
│   ├── annotate_mask.py                  # Standalone OpenCV interactive manual mask annotation GUI
│   ├── compare_segmentation_profiles.py  # Diagnostic profile comparison on specimen 00001
│   ├── evaluate_validation_dataset.py    # Multi-image evaluation harness (profiles & hysteresis)
│   ├── evaluate_hysteresis_multireal.py  # Multi-image comparative batch evaluator
│   ├── experiment_hysteresis.py          # Hysteresis parameter sweep experiment
│   ├── experiment_hysteresis_min_area.py # Minimum area cutoff study
│   ├── experiment_hysteresis_shape_filter.py # Shape elongation aspect-ratio study
│   ├── generate_aruco.py                 # Generates 50 mm ArUco marker image
│   └── visualize_validation.py           # False-color TP/FP/FN diagnostic visualization
│
├── data/                                 # Datasets and outputs
│   ├── calibration/                      # Calibration targets (aruco_marker_50mm.png)
│   ├── ground_truth/                     # Verified ground-truth binary masks (00001_mask.png)
│   ├── real_images/                      # 19 real concrete surface images (00001.jpg – 00062.jpg)
│   ├── sample_images/                    # Tracked sample images (synthetic & temperature crack)
│   └── outputs/                          # [Generated output directory, gitignored] Saved masks, overlays, CSV reports
│
├── docs/                                 # Technical & project documentation
│   ├── ANNOTATION_GUIDELINE.md           # Ground-truth binary mask contract and annotation manual
│   ├── CrackGauge_Presentation.pptx      # Project presentation slides
│   └── miniprojectsynopsisss.docx        # Approved project synopsis document
├── main.py                               # CLI entry point for single-image and batch execution
├── requirements.txt                      # Production Python dependencies
├── requirements-dev.txt                  # Test and development dependencies (pytest, httpx)
└── README.md                             # Project documentation
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10 to 3.13
- **Node.js:** 18.0+ and **npm** (for the web frontend)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/ayushgigras/CrackVision-AI.git
cd CrackVision-AI
```

### 2. Configure Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Windows (CMD):
.\venv\Scripts\activate.bat
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 3. Configure Frontend Web Dashboard
```bash
cd frontend
npm install
cd ..
```

---

## Running the Web Application

The system includes a decoupled web architecture: a FastAPI backend serving computer-vision analysis endpoints, and a Vite-powered React dashboard.

### 1. Launch FastAPI Backend Service
```bash
# Start backend server on http://127.0.0.1:8000
python -m uvicorn src.api:app --reload --port 8000
```
- **API Health Probe:** `http://127.0.0.1:8000/api/health`
- **Interactive OpenAPI Documentation (Swagger UI):** `http://127.0.0.1:8000/api/docs`
- **ReDoc:** `http://127.0.0.1:8000/api/redoc`

### 2. Launch React Frontend Development Server
In a second terminal:
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser. The frontend automatically connects to the FastAPI backend, displaying live connection telemetry (`API Online v0.1.0`), camera scan actions, and the interactive visual results inspector.

---

## Command-Line Usage

### 1. Analyze a Single Image

```bash
# Run baseline pipeline on a real concrete specimen
python main.py --image data/real_images/00001.jpg

# Display interactive matplotlib diagnostic plots
python main.py --image data/real_images/00001.jpg --show

# Save step-by-step intermediate images and plots to a custom directory
python main.py --image data/real_images/00001.jpg --output data/outputs/my_run
```

### 2. Physical Scale Calibration Options

```bash
# Option A: Direct millimetre-per-pixel scale
python main.py --image data/real_images/00001.jpg --mm-per-pixel 0.125

# Option B: Reference object ratio (known physical length / measured pixel length)
python main.py --image data/real_images/00001.jpg --reference-length-mm 100.0 --reference-length-px 820.0

# Option C: Automated ArUco fiducial marker calibration (marker must be in image plane)
python main.py --image data/sample_images/crack_synthetic_01.jpg --marker-size-mm 50.0 --aruco-dictionary DICT_4X4_50
```

### 3. Selectable Segmentation Profiles

```bash
# Baseline: Multi-scale Frangi filter + Otsu binarization (default)
python main.py --image data/real_images/00001.jpg --segmentation-profile baseline

# Adaptive Recovery: Uses cleaned CLAHE adaptive threshold mask
python main.py --image data/real_images/00001.jpg --segmentation-profile adaptive_recovery

# Guided Recovery: Adds adaptive components overlapping dilated Frangi seeds
python main.py --image data/real_images/00001.jpg --segmentation-profile guided_recovery

# Hysteresis Recovery: Dual-threshold hysteresis on normalized Frangi response
python main.py --image data/real_images/00001.jpg --segmentation-profile hysteresis_recovery
```

### 4. Batch Directory Processing

Process an entire directory of concrete images with fault tolerance and aggregate summary CSV export:

```bash
python main.py \
  --input-dir data/real_images \
  --output data/outputs/batch_run \
  --summary-csv data/outputs/batch_run/batch_summary.csv
```
Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Corrupted images are logged with error messages and do not interrupt batch processing.

---

## Evaluation & Annotation Tools

### 1. Interactive Ground-Truth Annotation GUI
The repository includes a standalone OpenCV manual labeling utility that starts with an empty canvas and does not bias ground truth with algorithmic predictions:

```bash
# Launch interactive labeling tool
python tools/annotate_mask.py \
  --image data/real_images/00001.jpg \
  --output data/ground_truth/00001_mask.png

# Resume editing an existing binary mask
python tools/annotate_mask.py \
  --image data/real_images/00001.jpg \
  --output data/ground_truth/00001_mask.png \
  --resume

# Validate an existing mask against the strict contract without opening GUI
python tools/annotate_mask.py \
  --image data/real_images/00001.jpg \
  --output data/ground_truth/00001_mask.png \
  --validate-only
```
**GUI Controls:**
- `Left Click + Drag`: Draw white crack pixels ($255$)
- `Right Click + Drag`: Erase pixels back to background ($0$)
- `[` / `]`: Decrease / increase brush diameter ($1\text{ to }101\text{ px}$)
- `s`: Save binary PNG mask
- `r` followed by `y`: Clear canvas
- `q` or `Esc`: Exit

### 2. Multi-Image Dataset Evaluation Harness
Evaluates segmentation profiles across all matched image-mask pairs conforming to `<stem>.jpg` and `<stem>_mask.png`:

```bash
# Evaluate segmentation profiles across dataset
python tools/evaluate_validation_dataset.py --mode profiles

# Evaluate isolated hysteresis component filtering strategies
python tools/evaluate_validation_dataset.py --mode hysteresis_strategies
```
Aggregate metrics (Precision, Recall, Dice/F1, IoU) are computed strictly from **summed integer confusion matrices** (micro-averaging). Images lacking matching ground truth are recorded as `skipped` and are never counted as negative examples.

---

## Segmentation Profiles Explained

> [!NOTE]
> The performance metrics below reflect single-image diagnostic evaluation on the pilot specimen (`data/real_images/00001.jpg`, $227\times 227\text{ px}$) against `00001_mask.png`. They illustrate trade-offs observed on this specific specimen, not generalized dataset-wide performance.

| Profile Name | Underlying Algorithmic Mechanism | Target Use Case | Observed Performance on Specimen `00001.jpg` |
|---|---|---|---|
| **`baseline`** *(default)* | Multi-scale Frangi Hessian filter ($\sigma \in \{1, 2, 3, 4\}$) binarized via Otsu's optimal thresholding with connected component area filter ($\ge 80\text{ px}$). | Clean concrete surfaces with high-contrast, well-defined cracks. | High Precision ($79.75\%$), Low Recall ($26.19\%$), Dice $0.3943$. Suppresses surface noise but drops faint fissure extremities. |
| **`adaptive_recovery`** | Evaluates the morphological adaptive threshold mask from Step 1 (Gaussian C, block size $21$, offset $C=5$, component filter $\ge 300\text{ px}$). | Low-contrast concrete, faint surface crazing, or uneven illumination. | Higher Recall ($70.29\%$), Precision $83.16\%$, Dice $0.7619$. Recovers faint crack branches with slightly higher false-positive area. |
| **`guided_recovery`** | Preserves baseline Frangi ridge detections as structural seeds and selectively admits adaptive threshold components that overlap a small dilation ($5\times 5$ ellipse) of the seeds. | Discontinuous fissures where aggregate bridging breaks tubular continuity. | Balanced Trade-off: Precision $82.73\%$, Recall $70.64\%$, Dice $0.7621$, IoU $0.6157$. |
| **`hysteresis_recovery`** | Dual-threshold hysteresis (`low=25, high=48`) on normalized Frangi response with optional shape-aware elongation filtering (aspect ratio $\ge 2.2$). | Rough concrete textures with surface aggregate pitting. | Intermediate Recovery: Precision $82.28\%$, Recall $45.73\%$, Dice $0.5879$. Anchors weak-ridge pixels to strong seeds. |

---

## Testing & Quality Assurance

The repository enforces test verification across all pipeline stages, validation math, service contracts, and REST API endpoints:

```bash
# Run complete automated test suite
pytest -v

# Collect test inventory summary
pytest --collect-only -q
```

### Verified Test Suite Results:
- **Total Tests:** **129 passed in 17.54s** across 11 modules with 0 failures.
- **Coverage Areas:**
  - `test_api.py` (6 tests): Health probe, upload size limits ($20\text{ MB}$), MIME type validation, structured error codes ($413, 415, 422$).
  - `test_pipeline_service.py` (41 tests): In-memory execution, NaN/Inf guards, JSON serializability, base64 data URL formatting, blank image handling.
  - `test_hysteresis_segmentation.py` (32 tests): Seed connection, threshold boundaries, shape-aware elongation filtering.
  - `test_measurement.py` (3 tests + 9 subtests): Medial axis distance transform width verification across bar thicknesses from $2\text{ to }10\text{ px}$.
  - `test_length_measurement.py` (5 tests): 8-connected skeleton traversal on horizontal and vertical lines ($79.00\text{ px}$ measured vs. $79.00\text{ px}$ expected).
  - `test_calibration.py` (5 tests): Scale math, unit conversions, ArUco corner detection and rectification.
  - `test_multi_image_validation.py` (9 tests): Image-mask pairing, mask contract enforcement, safe resumption.
  - `test_segmentation_profiles.py` (8 tests): Profile isolation and mask type validation.
  - `test_batch_processing.py` (4 tests): Directory discovery, error tolerance on corrupted images, CSV export.
  - `test_validation.py` & `test_validation_visualization.py` (12 tests): IoU, Dice, confusion matrix calculations, edge-case zero handling.

---

## Empirical Benchmark & Scientific Disclosures

To maintain scientific integrity and prevent fabricated performance claims:

### 1. Ground-Truth Dataset Scope
- The repository contains **19 real concrete crack images** ($227\times 227\text{ px}$) in `data/real_images/`.
- **Exactly ONE image (`00001.jpg`) currently possesses a verified, independently annotated ground-truth mask (`data/ground_truth/00001_mask.png`, 3,127 crack pixels).**
- The remaining 18 images are unannotated and are explicitly logged as `skipped` by the evaluation harness. **Performance on `00001.jpg` represents single-specimen diagnostic validation and must not be cited as dataset-wide accuracy.**

### 2. Measured Benchmark on Specimen `00001.jpg`
Evaluated via `python tools/evaluate_validation_dataset.py`:

| Profile | True Positive | False Positive | False Negative | True Negative | Precision | Recall | Dice / F1 | IoU |
|---|---|---|---|---|---|---|---|---|
| **`baseline`** | 819 px | 208 px | 2,308 px | 48,194 px | **79.75%** | **26.19%** | **0.3943** | **0.2456** |
| **`adaptive_recovery`** | 2,198 px | 445 px | 929 px | 47,957 px | **83.16%** | **70.29%** | **0.7619** | **0.6153** |
| **`guided_recovery`** | 2,209 px | 461 px | 918 px | 47,941 px | **82.73%** | **70.64%** | **0.7621** | **0.6157** |
| **`hysteresis_recovery`** | 1,430 px | 308 px | 1,697 px | 48,094 px | **82.28%** | **45.73%** | **0.5879** | **0.4163** |

### 3. Physical Validation Status (MAE / RMSE)
- Geometric measurements are verified in pixel space on synthetic geometric targets (measuring known 2–10 px lines with absolute error $\le 1.0\text{ px}$).
- **Zero physical laboratory crack measurements from feeler gauges, optical comparators, or calipers exist in the repository.**
- While error aggregation functions (`absolute_error`, `relative_error_percentage`, `aggregate_measurement_errors`) exist in `src/validation.py`, **Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) in millimetres are currently unavailable** pending physical specimen testing.

### 4. Classical Computer Vision (No Deep Learning)
CrackGauge does not utilize trained deep neural networks (e.g., U-Net, YOLO, Mask R-CNN). The system is built purely on deterministic image processing algorithms, ensuring mathematical explainability, zero GPU dependencies, and low-latency execution on commodity hardware.

---

## Current Development Status & Roadmap

### Implemented in Current Release
- [x] Step 1 Preprocessing (Grayscale, Gaussian Blur, CLAHE, Adaptive Thresholding, Morphological Cleanup)
- [x] Step 2 Segmentation (Multi-scale Frangi filter, Otsu, Hysteresis, Shape Filtering, 4 Profiles)
- [x] Step 3 Width Estimation (Medial axis skeleton, $L_2$ Euclidean distance transform, statistics)
- [x] Step 4A Length Estimation (Morphological gap closing, 8-connected graph traversal)
- [x] Step 4B Orientation Analysis (PCA eigenvector calculation, dominant angle & angular category)
- [x] Step 5 Physical Calibration (Direct $mm/px$ scale, reference length ratios, ArUco marker detection)
- [x] Headless In-Memory Pipeline Service (`src/pipeline_service.py`)
- [x] FastAPI REST Web Service with OpenAPI documentation (`src/api.py`)
- [x] React 19 / Vite Web Inspection Dashboard with Camera Capture (`frontend/`)
- [x] Interactive OpenCV Ground-Truth Annotation GUI (`tools/annotate_mask.py`)
- [x] Multi-Image Ground-Truth Evaluation Engine (`tools/evaluate_validation_dataset.py`)
- [x] 129 Automated Unit & Integration Tests

### Planned for Future Milestones
- [ ] **Expanded Ground Truth:** Annotate the remaining 18 real concrete images using `tools/annotate_mask.py`.
- [ ] **Physical Caliper Benchmarking:** Conduct laboratory test specimen measurements with feeler gauges to report physical MAE and RMSE in millimetres.
- [ ] **ArUco Generator Border Update:** Add automatic white quiet margin padding to `tools/generate_aruco.py` for out-of-the-box detection.
- [ ] **PDF Report Export:** Implement automated digital inspection report generation using ReportLab (`src/report.py`).
- [ ] **Inspection Database:** Implement SQLite persistence for longitudinal crack growth monitoring over time.
- [ ] **Full-Scene Rectification:** Extend perspective homography from the marker patch to the entire concrete wall plane.

---

## Tech Stack Summary

- **Core Runtime:** Python 3.10+
- **Computer Vision & Processing:** OpenCV (`opencv-python`), scikit-image, NumPy, SciPy
- **Data & Visualization:** Matplotlib, Pandas
- **Backend Web API:** FastAPI, Uvicorn, python-multipart
- **Frontend Dashboard:** React 19, Vite, Vanilla CSS Design System
- **Testing Toolchain:** Pytest, HTTPX
