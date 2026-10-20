"""CrackGauge Professional Inspection PDF Report Generator.

Generates an industrial-grade structural inspection PDF report containing:
- Original concrete surface image
- Crack segmentation overlay
- Quantitative geometric measurements (width, length, orientation, coverage)
- Segmentation algorithm profile
- Analysis timestamp
- Physical calibration status (reporting mm strictly when calibrated)
"""

from __future__ import annotations

import base64
import csv
import io
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _decode_data_url(data_url: Optional[str]) -> Optional[io.BytesIO]:
    """Decode a base64 PNG data URL to an in-memory BytesIO buffer."""
    if not data_url or not isinstance(data_url, str):
        return None
    try:
        raw = data_url.split(",", 1)[1] if "," in data_url else data_url
        data = base64.b64decode(raw)
        return io.BytesIO(data)
    except Exception:
        return None


def generate_pdf_report(analysis_data: Dict[str, Any]) -> bytes:
    """Generate a PDF report buffer from an analysis result dictionary.

    Parameters
    ----------
    analysis_data:
        Analysis result dictionary (from ``analyze_image_array`` or database).

    Returns
    -------
    bytes
        Raw PDF document bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom palette matching CrackGauge dark navy / industrial slate
    primary_color = colors.HexColor("#0f172a")     # Deep Slate
    accent_amber = colors.HexColor("#d97706")      # Amber
    border_color = colors.HexColor("#cbd5e1")      # Slate border
    header_bg = colors.HexColor("#f1f5f9")         # Light slate table header
    subtle_bg = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=2,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=accent_amber,
        spaceAfter=10,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=6,
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
    )

    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=cell_style,
        fontName="Helvetica-Bold",
    )

    cell_header = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=primary_color,
    )

    caption_style = ParagraphStyle(
        "CaptionStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,  # Centered
        textColor=colors.HexColor("#475569"),
        spaceBefore=4,
    )

    footer_style = ParagraphStyle(
        "FooterStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        alignment=1,
        textColor=colors.HexColor("#64748b"),
        spaceBefore=12,
    )

    story = []

    # -----------------------------------------------------------------------
    # Document Header
    # -----------------------------------------------------------------------
    story.append(Paragraph("CRACKGAUGE STRUCTURAL INSPECTION REPORT", title_style))
    story.append(
        Paragraph(
            "AUTOMATED COMPUTER VISION CRACK QUANTIFICATION &amp; GEOMETRIC PROFILING",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_amber, spaceAfter=10))

    # Extract analysis fields
    analysis_id = analysis_data.get("analysis_id", "N/A")
    image_name = analysis_data.get("image_name", "unnamed_capture")
    timestamp_raw = analysis_data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        timestamp_str = str(timestamp_raw)

    profile = str(analysis_data.get("segmentation_profile", "baseline"))
    dimensions = analysis_data.get("dimensions", {})
    w_px = dimensions.get("width", "—")
    h_px = dimensions.get("height", "—")

    summary = analysis_data.get("summary", {})
    calibration = summary.get("calibration", {})
    is_calibrated = bool(calibration.get("is_calibrated", False))
    mm_per_px = calibration.get("mm_per_pixel")

    if is_calibrated and mm_per_px is not None:
        calib_str = f"Calibrated ({mm_per_px:.4f} mm/px)"
    else:
        calib_str = "Uncalibrated (Pixel measurements only)"

    # Overview metadata table
    meta_data = [
        [
            Paragraph("Inspection ID", cell_header),
            Paragraph(str(analysis_id), cell_style),
            Paragraph("Date / Time", cell_header),
            Paragraph(timestamp_str, cell_style),
        ],
        [
            Paragraph("Target Image", cell_header),
            Paragraph(f"{image_name} ({w_px} &times; {h_px} px)", cell_style),
            Paragraph("Algorithm Profile", cell_header),
            Paragraph(profile.replace("_", " ").title(), cell_style),
        ],
        [
            Paragraph("Calibration Status", cell_header),
            Paragraph(calib_str, cell_bold if is_calibrated else cell_style),
            Paragraph("Dominant Pattern", cell_header),
            Paragraph(
                f"{summary.get('dominant_type', 'Undetermined')} ({summary.get('dominant_angle_deg', 0):.1f}&deg;)",
                cell_style,
            ),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[1.4 * inch, 2.3 * inch, 1.4 * inch, 2.3 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), subtle_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # Visual Evidence Section (Original vs Segmented Overlay)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Visual Inspection Evidence", section_heading))

    visuals = analysis_data.get("visuals", {})
    orig_stream = _decode_data_url(visuals.get("original"))
    overlay_stream = _decode_data_url(visuals.get("overlay"))

    # Desired box dimensions for side-by-side images
    img_max_w = 3.5 * inch
    img_max_h = 2.4 * inch

    img_cells = []
    if orig_stream and overlay_stream:
        img_orig = Image(orig_stream, width=img_max_w, height=img_max_h, kind="proportional")
        img_overlay = Image(overlay_stream, width=img_max_w, height=img_max_h, kind="proportional")
        img_cells = [
            [img_orig, img_overlay],
            [
                Paragraph("Figure 1: Original Concrete Surface", caption_style),
                Paragraph("Figure 2: Crack Detection &amp; Segmentation Overlay", caption_style),
            ],
        ]
    elif overlay_stream:
        img_overlay = Image(overlay_stream, width=img_max_w * 1.5, height=img_max_h * 1.5, kind="proportional")
        img_cells = [
            [img_overlay],
            [Paragraph("Crack Segmentation Overlay", caption_style)],
        ]
    elif orig_stream:
        img_orig = Image(orig_stream, width=img_max_w * 1.5, height=img_max_h * 1.5, kind="proportional")
        img_cells = [
            [img_orig],
            [Paragraph("Original Concrete Surface", caption_style)],
        ]

    if img_cells:
        cols = [3.7 * inch, 3.7 * inch] if len(img_cells[0]) == 2 else [7.4 * inch]
        img_table = Table(img_cells, colWidths=cols)
        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(img_table)
    story.append(Spacer(1, 10))

    # -----------------------------------------------------------------------
    # Geometric Quantification Results Table
    # -----------------------------------------------------------------------
    story.append(Paragraph("Quantitative Geometric Measurements", section_heading))

    max_w_px = summary.get("max_width_px")
    mean_w_px = summary.get("mean_width_px")
    med_w_px = summary.get("median_width_px")
    tot_l_px = summary.get("total_length_px")

    max_w_mm = summary.get("max_width_mm") if is_calibrated else None
    mean_w_mm = summary.get("mean_width_mm") if is_calibrated else None
    med_w_mm = summary.get("median_width_mm") if is_calibrated else None
    tot_l_mm = summary.get("total_length_mm") if is_calibrated else None

    def _val(val: Optional[float], unit: str, decimals: int = 2) -> str:
        if val is None or val == "":
            return "N/A (Uncalibrated)" if unit == "mm" else "—"
        return f"{val:.{decimals}f} {unit}"

    meas_rows = [
        [
            Paragraph("Geometric Parameter", cell_header),
            Paragraph("Measured Value (Pixel)", cell_header),
            Paragraph("Physical Dimension (Millimetre)", cell_header),
            Paragraph("Measurement Basis &amp; Method", cell_header),
        ],
        [
            Paragraph("Maximum Crack Width", cell_bold),
            Paragraph(_val(max_w_px, "px", 1), cell_style),
            Paragraph(_val(max_w_mm, "mm", 2), cell_bold if is_calibrated else cell_style),
            Paragraph("Medial axis distance transform (w = 2 &times; dist)", cell_style),
        ],
        [
            Paragraph("Mean Crack Width", cell_style),
            Paragraph(_val(mean_w_px, "px", 1), cell_style),
            Paragraph(_val(mean_w_mm, "mm", 2), cell_style),
            Paragraph("Arithmetic mean of widths across skeleton", cell_style),
        ],
        [
            Paragraph("Median Crack Width", cell_style),
            Paragraph(_val(med_w_px, "px", 1), cell_style),
            Paragraph(_val(med_w_mm, "mm", 2), cell_style),
            Paragraph("50th percentile width along crack skeleton", cell_style),
        ],
        [
            Paragraph("Total Crack Length", cell_bold),
            Paragraph(_val(tot_l_px, "px", 1), cell_style),
            Paragraph(_val(tot_l_mm, "mm", 1), cell_bold if is_calibrated else cell_style),
            Paragraph("8-connected graph traversal along medial axis", cell_style),
        ],
        [
            Paragraph("Dominant Orientation", cell_style),
            Paragraph(f"{summary.get('dominant_angle_deg', 0):.1f}&deg;", cell_style),
            Paragraph(str(summary.get("dominant_type", "Undetermined")), cell_style),
            Paragraph("Principal Component Analysis (PCA eigenvector)", cell_style),
        ],
        [
            Paragraph("Surface Crack Coverage", cell_style),
            Paragraph(f"{summary.get('crack_coverage_percent', 0):.2f}%", cell_style),
            Paragraph(f"{summary.get('crack_pixels', 0):,} pixels", cell_style),
            Paragraph("Total segmented fissure pixels vs total frame", cell_style),
        ],
        [
            Paragraph("Connected Components", cell_style),
            Paragraph(f"{summary.get('component_count', 0)} branch(es)", cell_style),
            Paragraph(
                f"{summary.get('raw_component_count', summary.get('component_count', 0))} raw components",
                cell_style,
            ),
            Paragraph("Continuous fissures after morphological gap closing", cell_style),
        ],
    ]

    meas_table = Table(
        meas_rows,
        colWidths=[1.8 * inch, 1.6 * inch, 1.8 * inch, 2.2 * inch],
    )
    meas_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(meas_table)
    story.append(Spacer(1, 8))

    # -----------------------------------------------------------------------
    # Continuous Crack Branches (if multiple)
    # -----------------------------------------------------------------------
    components = analysis_data.get("components", [])
    if components and len(components) > 1:
        story.append(Paragraph(f"Continuous Branch Breakdown ({len(components)} branches)", section_heading))
        comp_rows = [
            [
                Paragraph("Branch #", cell_header),
                Paragraph("Length (px)", cell_header),
                Paragraph("Length (mm)", cell_header),
                Paragraph("Mask Area (px)", cell_header),
                Paragraph("Bounding Box (x, y, w, h)", cell_header),
            ]
        ]
        for comp in components[:8]:  # Up to top 8 branches for clean fit
            comp_l_px = comp.get("length_px", 0.0)
            comp_l_mm = comp.get("length_mm") if is_calibrated else None
            bbox_str = f"({comp.get('bbox_x', 0)}, {comp.get('bbox_y', 0)}, {comp.get('bbox_w', 0)}, {comp.get('bbox_h', 0)})"
            comp_rows.append(
                [
                    Paragraph(f"Branch #{comp.get('id', 1)}", cell_style),
                    Paragraph(f"{comp_l_px:.1f} px", cell_style),
                    Paragraph(f"{comp_l_mm:.1f} mm" if comp_l_mm is not None else "N/A", cell_style),
                    Paragraph(f"{comp.get('mask_area_px', 0):,} px", cell_style),
                    Paragraph(bbox_str, cell_style),
                ]
            )

        comp_table = Table(comp_rows, colWidths=[1.2 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch, 2.0 * inch])
        comp_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                    ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ]
            )
        )
        story.append(comp_table)

    # -----------------------------------------------------------------------
    # Footer Notice
    # -----------------------------------------------------------------------
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "This report is generated deterministically by CrackGauge Computer Vision Analysis System. "
            "Measurements are computed via Hessian ridge detection, medial axis skeletonization, and Euclidean distance transform.",
            footer_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()


_CSV_COLUMNS = [
    "analysis_id",
    "image_name",
    "timestamp",
    "segmentation_profile",
    "is_calibrated",
    "calibration_status",
    "mm_per_pixel",
    "dimensions_width",
    "dimensions_height",
    "max_width_px",
    "max_width_mm",
    "mean_width_px",
    "mean_width_mm",
    "median_width_px",
    "median_width_mm",
    "total_length_px",
    "total_length_mm",
    "dominant_angle_deg",
    "dominant_type",
    "crack_coverage_percent",
    "crack_pixels",
    "component_count",
]


def generate_analysis_csv(analysis_data: Dict[str, Any]) -> str:
    """Generate a CSV string for a single analysis record."""
    summary = analysis_data.get("summary", {})
    calibration = summary.get("calibration", {})
    dimensions = analysis_data.get("dimensions", {})
    is_calibrated = bool(calibration.get("is_calibrated", False))

    row = {
        "analysis_id": analysis_data.get("analysis_id", ""),
        "image_name": analysis_data.get("image_name", ""),
        "timestamp": analysis_data.get("timestamp", ""),
        "segmentation_profile": analysis_data.get("segmentation_profile", "baseline"),
        "is_calibrated": is_calibrated,
        "calibration_status": calibration.get("status", "not_calibrated"),
        "mm_per_pixel": calibration.get("mm_per_pixel", "") if is_calibrated else "",
        "dimensions_width": dimensions.get("width", ""),
        "dimensions_height": dimensions.get("height", ""),
        "max_width_px": summary.get("max_width_px", ""),
        "max_width_mm": summary.get("max_width_mm", "") if is_calibrated else "",
        "mean_width_px": summary.get("mean_width_px", ""),
        "mean_width_mm": summary.get("mean_width_mm", "") if is_calibrated else "",
        "median_width_px": summary.get("median_width_px", ""),
        "median_width_mm": summary.get("median_width_mm", "") if is_calibrated else "",
        "total_length_px": summary.get("total_length_px", ""),
        "total_length_mm": summary.get("total_length_mm", "") if is_calibrated else "",
        "dominant_angle_deg": summary.get("dominant_angle_deg", ""),
        "dominant_type": summary.get("dominant_type", ""),
        "crack_coverage_percent": summary.get("crack_coverage_percent", ""),
        "crack_pixels": summary.get("crack_pixels", ""),
        "component_count": summary.get("component_count", ""),
    }

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


def generate_history_csv(history_list: List[Dict[str, Any]]) -> str:
    """Generate a combined CSV string for multiple historical analyses."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_COLUMNS)
    writer.writeheader()
    for item in history_list:
        is_cal = bool(item.get("is_calibrated", False))
        dims = item.get("dimensions", {})
        writer.writerow(
            {
                "analysis_id": item.get("analysis_id", ""),
                "image_name": item.get("image_name", ""),
                "timestamp": item.get("timestamp", ""),
                "segmentation_profile": item.get("segmentation_profile", "baseline"),
                "is_calibrated": is_cal,
                "calibration_status": item.get("calibration_status", "not_calibrated"),
                "mm_per_pixel": item.get("mm_per_pixel", "") if is_cal else "",
                "dimensions_width": dims.get("width", ""),
                "dimensions_height": dims.get("height", ""),
                "max_width_px": item.get("max_width_px", ""),
                "max_width_mm": item.get("max_width_mm", "") if is_cal else "",
                "mean_width_px": item.get("mean_width_px", ""),
                "mean_width_mm": item.get("mean_width_mm", "") if is_cal else "",
                "median_width_px": item.get("median_width_px", ""),
                "median_width_mm": item.get("median_width_mm", "") if is_cal else "",
                "total_length_px": item.get("total_length_px", ""),
                "total_length_mm": item.get("total_length_mm", "") if is_cal else "",
                "dominant_angle_deg": item.get("dominant_angle_deg", ""),
                "dominant_type": item.get("dominant_type", ""),
                "crack_coverage_percent": item.get("crack_coverage_percent", ""),
                "crack_pixels": item.get("crack_pixels", ""),
                "component_count": item.get("component_count", ""),
            }
        )
    return output.getvalue()

