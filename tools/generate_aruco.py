import cv2
from pathlib import Path

output_dir = Path("data/calibration")
output_dir.mkdir(parents=True, exist_ok=True)

marker_size_px = 800
marker_id = 0

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

marker = cv2.aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    marker_size_px
)

output_path = output_dir / "aruco_marker_50mm.png"

cv2.imwrite(str(output_path), marker)

print(f"[OK] ArUco marker generated: {output_path}")
print(f"[INFO] Marker ID: {marker_id}")
print("[INFO] Print this marker at exactly 50 mm × 50 mm.")