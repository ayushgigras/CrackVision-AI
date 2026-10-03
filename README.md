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
```

For automatic calibration, place a printed ArUco marker in the same plane as
the concrete surface and pass its measured physical side length with
`--marker-size-mm`. The default dictionary is `DICT_4X4_50`; select another
supported dictionary with `--aruco-dictionary`. If the marker is not detected,
the pipeline does not calculate a scale and reports width and length in pixels
only.

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
