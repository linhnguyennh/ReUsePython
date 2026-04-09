import pyrealsense2 as rs
import numpy as np
import cv2
import sys

from src.vision.realsense_frame import realsense_init, realsense_get_frame

# === Board Config (must match what you printed) ===
BOARD_COLS = 7
BOARD_ROWS = 5
SQUARE_LENGTH = 0.030     # meters — USE YOUR MEASURED VALUE
MARKER_LENGTH = 0.022     # meters

# === Camera Config ===
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# === Setup ChArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
)
detector = cv2.aruco.CharucoDetector(board)

max_corners = (BOARD_COLS - 1) * (BOARD_ROWS - 1)

# === Setup RealSense ===
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS)

pipeline,_,_ = realsense_init(CAM_WIDTH, CAM_HEIGHT, CAM_FPS)



print("Starting D435i...")
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"Failed to start camera: {e}")
    print("Check that the D435i is connected via USB 3.0")
    sys.exit(1)

# Get and display factory intrinsics for reference
color_stream = profile.get_stream(rs.stream.color)
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
print(f"Resolution:  {intrinsics.width}×{intrinsics.height}")
print(f"fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}")
print(f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
print(f"Distortion:  {intrinsics.coeffs}")
print(f"\nBoard:       {BOARD_COLS}×{BOARD_ROWS}, {SQUARE_LENGTH*1000:.0f}mm squares")
print(f"Max corners: {max_corners}")

# Warm up — let auto-exposure settle
print("Warming up (30 frames)...")
for _ in range(30):
    pipeline.wait_for_frames()

print("\nShowing live feed. Press Q to quit.\n")

# === Main Loop ===
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Detect
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(gray)

        # Draw ArUco markers (green boxes)
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        # Draw ChArUco corners (red dots)
        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(
                display, charuco_corners, charuco_ids
            )

            n = len(charuco_corners)
            ratio = n / max_corners

            # Color: green if >80%, yellow if >50%, red otherwise
            if ratio > 0.8:
                color = (0, 255, 0)
                status = "EXCELLENT"
            elif ratio > 0.5:
                color = (0, 255, 255)
                status = "OK"
            else:
                color = (0, 100, 255)
                status = "LOW"

            cv2.putText(
                display,
                f"Corners: {n}/{max_corners}  [{status}]",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2
            )
            cv2.putText(
                display,
                f"Markers: {len(marker_ids)}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
            )

            # Show estimated distance to board (rough, using factory intrinsics)
            if n >= 6:
                K = np.array([
                    [intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]
                ])
                D = np.array(intrinsics.coeffs)

                obj_pts, img_pts = board.matchImagePoints(
                    charuco_corners, charuco_ids
                )
                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)

                if success:
                    dist_mm = np.linalg.norm(tvec) * 1000
                    cv2.putText(
                        display,
                        f"Distance: {dist_mm:.0f} mm",
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 200, 100), 2
                    )

                    # Draw axis on the board
                    cv2.drawFrameAxes(
                        display, K, D, rvec, tvec, SQUARE_LENGTH * 2
                    )

        else:
            cv2.putText(
                display,
                "NO BOARD DETECTED",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )

        cv2.imshow("D435i ChArUco Detection Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Camera stopped.")