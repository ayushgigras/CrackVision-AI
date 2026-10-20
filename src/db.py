"""CrackGauge SQLite Persistence Layer for Analysis History.

Provides lightweight, thread-safe, and persistent storage of crack analysis
results with safe identifier validation, parameterized queries, and client
ownership rules.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "crackgauge.db")

_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_id(identifier: str, name: str = "Identifier") -> None:
    """Ensure identifier matches safe alphanumeric/dash/underscore rules."""
    if not identifier or not _ID_PATTERN.match(identifier):
        raise ValueError(
            f"Invalid {name}: must be 1-64 alphanumeric characters, hyphens, or underscores."
        )


def get_db_path(custom_path: Optional[str] = None) -> str:
    """Return the absolute path to the SQLite database file."""
    if custom_path:
        return os.path.abspath(custom_path)
    env_path = os.getenv("CRACKGAUGE_DB_PATH")
    if env_path:
        return os.path.abspath(env_path)
    return _DEFAULT_DB_PATH


import contextlib


@contextlib.contextmanager
def _get_connection(db_path: Optional[str] = None):
    """Yield an open sqlite3 connection and guarantee it is closed on exit."""
    resolved_path = get_db_path(db_path)
    parent_dir = os.path.dirname(resolved_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(resolved_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except Exception:
        pass
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    """Create necessary tables and indices if they do not exist."""
    with _get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                image_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                segmentation_profile TEXT NOT NULL,
                is_calibrated INTEGER NOT NULL,
                calibration_status TEXT NOT NULL,
                mm_per_pixel REAL,
                max_width_px REAL NOT NULL,
                max_width_mm REAL,
                mean_width_px REAL NOT NULL,
                mean_width_mm REAL,
                median_width_px REAL,
                median_width_mm REAL,
                std_width_px REAL,
                total_length_px REAL NOT NULL,
                total_length_mm REAL,
                component_count INTEGER NOT NULL,
                dominant_angle_deg REAL,
                dominant_type TEXT,
                crack_coverage_percent REAL,
                crack_pixels INTEGER,
                dimensions_width INTEGER NOT NULL,
                dimensions_height INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                components_json TEXT NOT NULL,
                visuals_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_client ON analyses(client_id, timestamp DESC);"
        )
        conn.commit()


def save_analysis(
    data: Dict[str, Any],
    client_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> str:
    """Save an analysis dictionary into SQLite and return its analysis ID.

    Parameters
    ----------
    data:
        Result dictionary produced by ``analyze_image_array`` or REST API.
    client_id:
        Optional client / operator identifier for ownership scoping.
        Defaults to ``"default"``.
    db_path:
        Optional SQLite database file path.
    """
    init_db(db_path)

    analysis_id = data.get("analysis_id") or uuid.uuid4().hex
    _validate_id(analysis_id, "analysis_id")

    safe_client_id = (client_id or "default").strip()
    if not safe_client_id:
        safe_client_id = "default"
    _validate_id(safe_client_id, "client_id")

    image_name = str(data.get("image_name") or "upload").strip()
    # Strip directory traversal characters
    image_name = os.path.basename(image_name) or "upload"

    timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
    summary = data.get("summary") or {}
    dimensions = data.get("dimensions") or {"width": 0, "height": 0}
    components = data.get("components") or []
    visuals = data.get("visuals") or {}
    calibration = summary.get("calibration") or {}

    is_calibrated = 1 if calibration.get("is_calibrated") else 0
    calibration_status = str(calibration.get("status") or "not_calibrated")
    mm_per_pixel = calibration.get("mm_per_pixel")
    segmentation_profile = str(data.get("segmentation_profile") or "baseline")

    with _get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses (
                id, client_id, image_name, timestamp, segmentation_profile,
                is_calibrated, calibration_status, mm_per_pixel,
                max_width_px, max_width_mm,
                mean_width_px, mean_width_mm,
                median_width_px, median_width_mm,
                std_width_px,
                total_length_px, total_length_mm,
                component_count, dominant_angle_deg, dominant_type,
                crack_coverage_percent, crack_pixels,
                dimensions_width, dimensions_height,
                summary_json, components_json, visuals_json
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?,
                ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?
            );
            """,
            (
                analysis_id,
                safe_client_id,
                image_name,
                timestamp,
                segmentation_profile,
                is_calibrated,
                calibration_status,
                mm_per_pixel,
                summary.get("max_width_px", 0.0),
                summary.get("max_width_mm"),
                summary.get("mean_width_px", 0.0),
                summary.get("mean_width_mm"),
                summary.get("median_width_px"),
                summary.get("median_width_mm"),
                summary.get("std_width_px"),
                summary.get("total_length_px", 0.0),
                summary.get("total_length_mm"),
                summary.get("component_count", 0),
                summary.get("dominant_angle_deg"),
                summary.get("dominant_type"),
                summary.get("crack_coverage_percent"),
                summary.get("crack_pixels"),
                int(dimensions.get("width", 0)),
                int(dimensions.get("height", 0)),
                json.dumps(summary),
                json.dumps(components),
                json.dumps(visuals),
            ),
        )
        conn.commit()

    return analysis_id


def list_analyses(
    client_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return lightweight summary records for history listing (without base64 blobs)."""
    init_db(db_path)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    where_clause = ""
    params: List[Any] = []
    if client_id and client_id.strip():
        safe_client = client_id.strip()
        _validate_id(safe_client, "client_id")
        where_clause = "WHERE client_id = ?"
        params.append(safe_client)

    with _get_connection(db_path) as conn:
        count_query = f"SELECT COUNT(*) FROM analyses {where_clause};"
        total = conn.execute(count_query, params).fetchone()[0]

        query = f"""
            SELECT
                id, client_id, image_name, timestamp, segmentation_profile,
                is_calibrated, calibration_status, mm_per_pixel,
                max_width_px, max_width_mm,
                mean_width_px, mean_width_mm,
                median_width_px, median_width_mm,
                total_length_px, total_length_mm,
                component_count, dominant_angle_deg, dominant_type,
                crack_coverage_percent, crack_pixels,
                dimensions_width, dimensions_height
            FROM analyses
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
        """
        rows = conn.execute(query, params + [limit, offset]).fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "analysis_id": r["id"],
                "client_id": r["client_id"],
                "image_name": r["image_name"],
                "timestamp": r["timestamp"],
                "segmentation_profile": r["segmentation_profile"],
                "is_calibrated": bool(r["is_calibrated"]),
                "calibration_status": r["calibration_status"],
                "mm_per_pixel": r["mm_per_pixel"],
                "max_width_px": r["max_width_px"],
                "max_width_mm": r["max_width_mm"],
                "mean_width_px": r["mean_width_px"],
                "mean_width_mm": r["mean_width_mm"],
                "median_width_px": r["median_width_px"],
                "median_width_mm": r["median_width_mm"],
                "total_length_px": r["total_length_px"],
                "total_length_mm": r["total_length_mm"],
                "component_count": r["component_count"],
                "dominant_angle_deg": r["dominant_angle_deg"],
                "dominant_type": r["dominant_type"],
                "crack_coverage_percent": r["crack_coverage_percent"],
                "crack_pixels": r["crack_pixels"],
                "dimensions": {
                    "width": r["dimensions_width"],
                    "height": r["dimensions_height"],
                },
            }
        )
    return results, total


def get_analysis(
    analysis_id: str,
    client_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the complete analysis result matching pipeline structure."""
    init_db(db_path)
    _validate_id(analysis_id, "analysis_id")

    query = "SELECT * FROM analyses WHERE id = ?"
    params: List[Any] = [analysis_id]
    if client_id and client_id.strip():
        safe_client = client_id.strip()
        _validate_id(safe_client, "client_id")
        query += " AND client_id = ?"
        params.append(safe_client)

    with _get_connection(db_path) as conn:
        row = conn.execute(query, params).fetchone()

    if not row:
        return None

    try:
        summary = json.loads(row["summary_json"])
    except Exception:
        summary = {}
    try:
        components = json.loads(row["components_json"])
    except Exception:
        components = []
    try:
        visuals = json.loads(row["visuals_json"])
    except Exception:
        visuals = {}

    return {
        "analysis_id": row["id"],
        "client_id": row["client_id"],
        "image_name": row["image_name"],
        "timestamp": row["timestamp"],
        "segmentation_profile": row["segmentation_profile"],
        "dimensions": {
            "width": row["dimensions_width"],
            "height": row["dimensions_height"],
        },
        "summary": summary,
        "components": components,
        "visuals": visuals,
    }


def delete_analysis(
    analysis_id: str,
    client_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """Delete an analysis record. Returns True if deleted, False if not found."""
    init_db(db_path)
    _validate_id(analysis_id, "analysis_id")

    query = "DELETE FROM analyses WHERE id = ?"
    params: List[Any] = [analysis_id]
    if client_id and client_id.strip():
        safe_client = client_id.strip()
        _validate_id(safe_client, "client_id")
        query += " AND client_id = ?"
        params.append(safe_client)

    with _get_connection(db_path) as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0
