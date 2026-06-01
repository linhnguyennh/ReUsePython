import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.vision.realsense_frame import realsense_init, realsense_get_frame

# === Board Config ===
BOARD_COLS = 4
BOARD_ROWS = 3
SQUARE_LENGTH = 0.030
MARKER_LENGTH = 0.022

# === Camera Config ===
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# === Setup ChArUco ===
#aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
)
detector = cv2.aruco.CharucoDetector(board)

max_corners = (BOARD_COLS - 1) * (BOARD_ROWS - 1)

# === Setup RealSense ===
print("Starting D435i...")
try:
    rs_cfg = realsense_init(CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
except RuntimeError as e:
    print(f"Failed to start camera: {e}")
    print("Check that the D435i is connected via USB 3.0")
    sys.exit(1)

intrinsics = rs_cfg.color_intrinsics
print(f"Resolution:  {intrinsics.width}×{intrinsics.height}")
print(f"fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
print(f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
print(f"Distortion:  {intrinsics.coeffs}")
print(f"\nBoard:       {BOARD_COLS}×{BOARD_ROWS}, {SQUARE_LENGTH*1000:.0f}mm squares")
print(f"Max corners: {max_corners}")

# realsense_init already does a 5-frame warm-up; add more if needed
print("Warming up (25 more frames)...")
for _ in range(25):
    realsense_get_frame(rs_cfg)

print("\nShowing live feed. Press Q to quit.\n")

K = np.array([
    [intrinsics.fx, 0,            intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1             ]
])
D = np.array(intrinsics.coeffs)

# === Main Loop ===
try:
    while True:
        color_frame, _ = realsense_get_frame(rs_cfg)
        if color_frame is None:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(gray)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

            n = len(charuco_corners)
            ratio = n / max_corners

            if ratio > 0.8:
                color, status = (0, 255, 0),   "EXCELLENT"
            elif ratio > 0.5:
                color, status = (0, 255, 255), "OK"
            else:
                color, status = (0, 100, 255), "LOW"

            cv2.putText(display, f"Corners: {n}/{max_corners}  [{status}]",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"Markers: {len(marker_ids)}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            if n >= 6:
                obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)

                if success:
                    dist_mm = np.linalg.norm(tvec) * 1000
                    cv2.putText(display, f"Distance: {dist_mm:.0f} mm",
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
                    cv2.drawFrameAxes(display, K, D, rvec, tvec, SQUARE_LENGTH * 2)
        else:
            cv2.putText(display, "NO BOARD DETECTED",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("D435i ChArUco Detection Test", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    rs_cfg.pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")