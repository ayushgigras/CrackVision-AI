"""CrackGauge FastAPI web server.

Endpoints
---------
GET  /api/health      — Liveness probe; returns server version and capabilities.
POST /api/analyze     — Upload one image, receive full crack-measurement results.

Start the server
----------------
    pip install -r requirements.txt       # installs fastapi + uvicorn
    uvicorn src.api:app --reload --port 8000

Or from the project root:
    python -m uvicorn src.api:app --reload --port 8000

Sample cURL request
-------------------
    curl -X POST http://localhost:8000/api/analyze \\
         -F "file=@data/sample_images/crack_synthetic_01.jpg" \\
         -F "segmentation_profile=baseline"
"""

from __future__ import annotations

import io
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

import cv2
import numpy as np

# Ensure project root is on sys.path so ``src.*`` imports resolve when uvicorn
# is launched from inside or outside the project directory.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.db import delete_analysis, get_analysis, list_analyses, save_analysis
from src.pipeline_service import (
    PipelineConfig,
    _SUPPORTED_ARUCO_DICTS,
    _SUPPORTED_BIN_METHODS,
    _SUPPORTED_PROFILES,
    analyze_image_array,
)
from src.report import (
    generate_analysis_csv,
    generate_history_csv,
    generate_pdf_report,
)
from src.segmentation import SEGMENTATION_PROFILES

logger = logging.getLogger("crackvision.api")

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CrackGauge API",
    description=(
        "Computer Vision crack-measurement pipeline exposed as a REST API.\n\n"
        "Upload a concrete-surface image and receive crack width, length, "
        "orientation and (optionally) physical calibration in millimetres."
    ),
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS: allow all origins during development.
# Restrict to specific origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 20 MB
_ALLOWED_CONTENT_TYPES = frozenset(
    ("image/jpeg", "image/png", "image/bmp", "image/webp")
)
_ALLOWED_EXTENSIONS = frozenset(
    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
)
_MIN_IMAGE_EDGE = 16    # px  — matches pipeline_service lower bound
_MAX_IMAGE_EDGE = 8000  # px  — practical sanity cap for web uploads


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _decode_upload(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decode raw upload bytes to a BGR NumPy array.

    Raises HTTPException(422) for corrupted or unreadable files.
    """
    buf = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "image_decode_failed",
                "message": (
                    f"The uploaded file '{filename}' could not be decoded as an image. "
                    "Ensure the file is a valid JPEG, PNG, BMP, or WebP image."
                ),
            },
        )
    return image


def _validate_upload(upload: UploadFile, raw_bytes: bytes) -> None:
    """Validate file size, extension, and reported content-type.

    Raises HTTPException(413) or HTTPException(415) on rejection.
    """
    # File size
    n_bytes = len(raw_bytes)
    if n_bytes == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "empty_file", "message": "Uploaded file is empty."},
        )
    if n_bytes > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "file_too_large",
                "message": (
                    f"File size {n_bytes / 1024 / 1024:.1f} MB exceeds the "
                    f"{_MAX_UPLOAD_BYTES // 1024 // 1024} MB limit."
                ),
            },
        )
    # Extension
    if upload.filename:
        ext = os.path.splitext(upload.filename)[-1].lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "error": "unsupported_extension",
                    "message": (
                        f"File extension '{ext}' is not supported. "
                        f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
                    ),
                },
            )
    # Content-type header (advisory; we still try to decode regardless)
    if upload.content_type and upload.content_type not in _ALLOWED_CONTENT_TYPES:
        logger.warning(
            "Upload '%s' has unexpected content-type '%s'; "
            "proceeding with pixel-level decode.",
            upload.filename,
            upload.content_type,
        )


def _validate_image_dimensions(image: np.ndarray) -> None:
    """Raise HTTPException(422) if decoded image dimensions are out of range."""
    h, w = image.shape[:2]
    if h < _MIN_IMAGE_EDGE or w < _MIN_IMAGE_EDGE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "image_too_small",
                "message": (
                    f"Decoded image ({w}×{h} px) is too small. "
                    f"Minimum supported size is {_MIN_IMAGE_EDGE}×{_MIN_IMAGE_EDGE} px."
                ),
            },
        )
    if h > _MAX_IMAGE_EDGE or w > _MAX_IMAGE_EDGE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "image_too_large",
                "message": (
                    f"Decoded image ({w}×{h} px) exceeds the "
                    f"{_MAX_IMAGE_EDGE}×{_MAX_IMAGE_EDGE} px limit."
                ),
            },
        )


def _build_config(
    segmentation_profile: str,
    bin_method: str,
    fusion: str,
    min_area: int,
    gap_closing_ksize: int,
    gap_closing_iters: int,
    marker_size_mm: Optional[float],
    mm_per_pixel: Optional[float],
    reference_length_mm: Optional[float],
    reference_length_px: Optional[float],
    aruco_dictionary: str,
) -> PipelineConfig:
    """Construct and validate a PipelineConfig; raise HTTPException(422) on error."""
    try:
        cfg = PipelineConfig(
            segmentation_profile=segmentation_profile,
            bin_method=bin_method,
            fusion=fusion,
            min_area=min_area,
            gap_closing_ksize=gap_closing_ksize,
            gap_closing_iters=gap_closing_iters,
            marker_size_mm=marker_size_mm,
            mm_per_pixel=mm_per_pixel,
            reference_length_mm=reference_length_mm,
            reference_length_px=reference_length_px,
            aruco_dictionary=aruco_dictionary,
            save_to_disk=False,
        )
        cfg.validate()
        return cfg
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "invalid_configuration", "message": str(exc)},
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/health",
    summary="Health check",
    tags=["Utility"],
)
async def health() -> dict:
    """Return server liveness and capability information.

    This endpoint never calls the pipeline and never fails unless the
    server process itself is unhealthy.
    """
    try:
        aruco_available = hasattr(cv2, "aruco")
    except Exception:
        aruco_available = False

    return {
        "status": "healthy",
        "version": __version__,
        "supported_segmentation_profiles": list(_SUPPORTED_PROFILES),
        "supported_bin_methods": list(_SUPPORTED_BIN_METHODS),
        "supported_aruco_dictionaries": list(_SUPPORTED_ARUCO_DICTS),
        "aruco_available": aruco_available,
        "max_upload_bytes": _MAX_UPLOAD_BYTES,
        "allowed_extensions": sorted(_ALLOWED_EXTENSIONS),
    }


@app.post(
    "/api/analyze",
    summary="Analyze a crack image",
    tags=["Analysis"],
    status_code=status.HTTP_200_OK,
)
async def analyze(
    file: Annotated[UploadFile, File(description="Concrete surface image (JPEG/PNG/BMP/WebP, ≤20 MB)")],
    segmentation_profile: Annotated[
        str,
        Form(description=f"Segmentation profile. One of: {', '.join(SEGMENTATION_PROFILES)}"),
    ] = "baseline",
    bin_method: Annotated[
        str,
        Form(description="Ridge binarisation method: 'otsu' or 'cutoff'"),
    ] = "otsu",
    fusion: Annotated[
        str,
        Form(description="Mask fusion mode for baseline profile: 'frangi_primary', 'intersection', or 'union'"),
    ] = "frangi_primary",
    min_area: Annotated[
        int,
        Form(description="Minimum crack component area in pixels (default 80)", ge=1),
    ] = 80,
    gap_closing_ksize: Annotated[
        int,
        Form(description="Morphological closing kernel size for length measurement (0 = disabled, default 7)", ge=0),
    ] = 7,
    gap_closing_iters: Annotated[
        int,
        Form(description="Morphological closing iterations (default 1)", ge=1),
    ] = 1,
    marker_size_mm: Annotated[
        Optional[float],
        Form(description="Physical side-length of the ArUco marker in mm (enables auto ArUco calibration)"),
    ] = None,
    mm_per_pixel: Annotated[
        Optional[float],
        Form(description="Direct physical scale (mm per pixel). Mutually exclusive with marker_size_mm and reference_length_* pair"),
    ] = None,
    reference_length_mm: Annotated[
        Optional[float],
        Form(description="Physical length of a reference object (mm). Must be paired with reference_length_px"),
    ] = None,
    reference_length_px: Annotated[
        Optional[float],
        Form(description="Pixel length of the same reference object. Must be paired with reference_length_mm"),
    ] = None,
    aruco_dictionary: Annotated[
        str,
        Form(description=f"ArUco dictionary name. One of: {', '.join(_SUPPORTED_ARUCO_DICTS)}"),
    ] = "DICT_4X4_50",
    x_client_id: Annotated[
        Optional[str],
        Header(description="Optional client / operator identifier for history ownership"),
    ] = None,
) -> JSONResponse:
    """Upload a concrete-surface image and receive crack measurements.

    **Processing steps**

    1. Preprocessing (grayscale → Gaussian blur → CLAHE → adaptive threshold → morphological cleanup)
    2. Segmentation (Frangi ridge filter → binarisation → component filtering → profile selection)
    3. Width measurement (skeletonisation + distance transform)
    4A. Length measurement (skeleton path traversal with optional gap-closing)
    4B. Orientation measurement (PCA on crack pixels)
    5. Calibration (ArUco marker or manual scale → mm conversion)
    6. Persistence (Save analysis metadata and results to SQLite history)

    **Response**

    All image artifacts are returned as `data:image/png;base64,…` URLs ready
    for assignment to React `<img src>` attributes.  Pixel measurements are
    always present; millimetre measurements are only present when calibration
    succeeds.  Never returns NumPy arrays or internal filesystem paths.

    **Calibration rules**

    Specify at most **one** of:
    - `marker_size_mm`  → automatic ArUco detection
    - `mm_per_pixel`    → direct scale
    - `reference_length_mm` + `reference_length_px`  → reference-object scale

    Specifying more than one source is rejected with HTTP 422.
    """
    # 1. Read upload bytes
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "read_failed", "message": "Could not read uploaded file."},
        ) from exc

    # 2. Validate upload metadata + byte size
    _validate_upload(file, raw_bytes)

    # 3. Validate configuration before spending compute on decoding
    config = _build_config(
        segmentation_profile=segmentation_profile,
        bin_method=bin_method,
        fusion=fusion,
        min_area=min_area,
        gap_closing_ksize=gap_closing_ksize,
        gap_closing_iters=gap_closing_iters,
        marker_size_mm=marker_size_mm,
        mm_per_pixel=mm_per_pixel,
        reference_length_mm=reference_length_mm,
        reference_length_px=reference_length_px,
        aruco_dictionary=aruco_dictionary,
    )

    # 4. Decode image (pure bytes → NumPy array; no disk I/O)
    image_name = os.path.splitext(file.filename or "upload")[0]
    image_bgr = _decode_upload(raw_bytes, file.filename or "upload")
    _validate_image_dimensions(image_bgr)

    # 5. Run CPU-bound pipeline in thread pool so async event loop stays free
    try:
        result = await run_in_threadpool(
            analyze_image_array,
            image_bgr,
            config,
            image_name=image_name,
        )
    except ValueError as exc:
        # Configuration / image errors raised inside the service
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "pipeline_error", "message": str(exc)},
        ) from exc
    except Exception as exc:
        # Unexpected errors: log detail internally, return opaque 500
        logger.exception("Unexpected pipeline error for file '%s'", file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred during analysis. "
                           "Please check server logs.",
            },
        ) from exc

    # 6. Assign persistent identity and save to SQLite history
    analysis_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    result["analysis_id"] = analysis_id
    result["timestamp"] = timestamp
    result["segmentation_profile"] = segmentation_profile

    try:
        save_analysis(result, client_id=x_client_id or "default")
    except Exception as exc:
        logger.warning("Could not persist analysis %s to SQLite: %s", analysis_id, exc)

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# History & Reporting Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/api/history",
    summary="List analysis history",
    tags=["History"],
)
async def get_history(
    limit: Annotated[int, Query(description="Maximum records to return", ge=1, le=200)] = 50,
    offset: Annotated[int, Query(description="Number of records to skip", ge=0)] = 0,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    """Return past analysis records without heavy base64 visual blobs."""
    try:
        items, total = list_analyses(client_id=x_client_id, limit=limit, offset=offset)
        return {"analyses": items, "total": total, "limit": limit, "offset": offset}
    except Exception as exc:
        logger.exception("Failed to query analysis history: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "history_query_failed", "message": str(exc)},
        ) from exc


@app.get(
    "/api/history/export/csv",
    summary="Export all analysis history as CSV",
    tags=["Reporting"],
)
async def export_all_history_csv(
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> Response:
    """Export summary metrics of all saved analyses as a combined CSV spreadsheet."""
    items, _ = list_analyses(client_id=x_client_id, limit=200)
    csv_content = generate_history_csv(items)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="crackgauge_history.csv"'},
    )


@app.get(
    "/api/history/{analysis_id}",
    summary="Get full analysis record by ID",
    tags=["History"],
)
async def get_history_item(
    analysis_id: str,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    """Retrieve full analysis result including base64 visual overlays."""
    try:
        record = get_analysis(analysis_id, client_id=x_client_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_id", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Failed to fetch analysis %s: %s", analysis_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "fetch_failed", "message": str(exc)},
        ) from exc

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Analysis '{analysis_id}' not found."},
        )
    return record


@app.delete(
    "/api/history/{analysis_id}",
    summary="Delete an analysis record",
    tags=["History"],
)
async def delete_history_item(
    analysis_id: str,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> dict:
    """Safely delete an analysis record from persistent history."""
    try:
        deleted = delete_analysis(analysis_id, client_id=x_client_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_id", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Failed to delete analysis %s: %s", analysis_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "delete_failed", "message": str(exc)},
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Analysis '{analysis_id}' not found."},
        )
    return {"deleted": True, "analysis_id": analysis_id}


@app.get(
    "/api/history/{analysis_id}/pdf",
    summary="Download PDF report for an existing analysis",
    tags=["Reporting"],
)
async def download_history_pdf(
    analysis_id: str,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> Response:
    """Generate and download a professional PDF inspection report."""
    try:
        record = get_analysis(analysis_id, client_id=x_client_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "fetch_failed", "message": str(exc)},
        ) from exc

    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Analysis '{analysis_id}' not found."},
        )

    try:
        pdf_bytes = await run_in_threadpool(generate_pdf_report, record)
    except Exception as exc:
        logger.exception("Failed to generate PDF for %s: %s", analysis_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "pdf_generation_failed", "message": str(exc)},
        ) from exc

    filename = f"crackgauge_report_{record.get('image_name', analysis_id)}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post(
    "/api/report/pdf",
    summary="Generate PDF report from current analysis payload",
    tags=["Reporting"],
)
async def generate_pdf_from_payload(
    payload: Annotated[Dict[str, Any], Body(description="Full analysis JSON payload")],
) -> Response:
    """Generate a PDF report directly from the current analysis payload."""
    try:
        pdf_bytes = await run_in_threadpool(generate_pdf_report, payload)
    except Exception as exc:
        logger.exception("Failed to generate PDF from payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "pdf_generation_failed", "message": str(exc)},
        ) from exc

    safe_name = payload.get("image_name") or payload.get("analysis_id") or "scan"
    filename = f"crackgauge_report_{safe_name}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get(
    "/api/history/{analysis_id}/csv",
    summary="Download CSV for an analysis record",
    tags=["Reporting"],
)
async def download_history_csv(
    analysis_id: str,
    x_client_id: Annotated[Optional[str], Header()] = None,
) -> Response:
    """Export quantitative measurement values for an analysis as CSV."""
    record = get_analysis(analysis_id, client_id=x_client_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": f"Analysis '{analysis_id}' not found."},
        )

    csv_content = generate_analysis_csv(record)
    filename = f"crackgauge_{record.get('image_name', analysis_id)}_measurements.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


