import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.vision.realsense_frame import realsense_init, realsense_get_frame

# === Marker Config ===
MARKER_LENGTH = 0.022      # meters — physical side length of your printed marker
TARGET_MARKER_ID = 2       # which marker ID to track

# === Camera Config ===
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# === Setup ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# 3D corners of the marker in marker frame (z=0 plane)
# Order: top-left, top-right, bottom-right, bottom-left
half = MARKER_LENGTH / 2.0

#CENTER
# obj_pts = np.array([
#     [-half,  half, 0],
#     [ half,  half, 0],
#     [ half, -half, 0],
#     [-half, -half, 0],
# ], dtype=np.float32)

#CORNER0
obj_pts = np.array([
    [0,            0,            0],  # top-left  → origin (corner 0, red dot)
    [MARKER_LENGTH, 0,            0],  # top-right
    [MARKER_LENGTH, MARKER_LENGTH, 0],  # bottom-right
    [0,            MARKER_LENGTH, 0],  # bottom-left
], dtype=np.float32)

# === Setup RealSense ===
print("Starting D435i...")
try:
    rs_cfg = realsense_init(CAM_WIDTH, CAM_HEIGHT, CAM_FPS)
except RuntimeError as e:
    print(f"Failed to start camera: {e}")
    sys.exit(1)

intrinsics = rs_cfg.color_intrinsics
K = np.array([
    [intrinsics.fx, 0,             intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1             ]
], dtype=np.float64)
D = np.array(intrinsics.coeffs, dtype=np.float64)

print(f"fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
print(f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
print("\nShowing live feed. Press Q to quit.\n")

# === Main Loop ===
try:
    while True:
        color_frame, depth_frame = realsense_get_frame(rs_cfg)
        if color_frame is None or depth_frame is None:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

            # Find our target marker
            matches = np.where(marker_ids.flatten() == TARGET_MARKER_ID)[0]
            if len(matches) > 0:
                corners = marker_corners[matches[0]].reshape(4, 2).astype(np.float32)

                #center = corners.mean(axis=0).astype(int)  # pixel center of marker
                corner0 = corners[0].astype(int)

                depth_value = np.asanyarray(depth_frame.get_data())[corner0[1], corner0[0]]
                depth_mm = depth_value * rs_cfg.depth_scale * 1000

                success, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    x, y, z = tvec.flatten() * 1000  # convert to mm
                    cv2.drawFrameAxes(display, K, D, rvec, tvec, MARKER_LENGTH)

                    cv2.putText(display, f"ID {TARGET_MARKER_ID}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(display, f"X: {x:+.1f} mm",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                    cv2.putText(display, f"Y: {y:+.1f} mm",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                    cv2.putText(display, f"Z: {z:+.1f} mm",
                                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
                    cv2.putText(display, f"Depth (sensor): {depth_mm:.1f} mm",
                                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
                    cv2.putText(display, f"({x:+.1f}, {y:+.1f}, {z:+.1f}) mm",
                                (corner0[0] + 10, corner0[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                    cv2.drawMarker(display, tuple(corner0), (100, 200, 255),
                                    cv2.MARKER_CROSS, 20, 2)
            else:
                cv2.putText(display, f"Marker {TARGET_MARKER_ID} not found",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
        else:
            cv2.putText(display, "NO MARKERS DETECTED",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2
        cv2.line(display, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(display, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        cv2.imshow("ArUco Marker Pose", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    rs_cfg.pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")