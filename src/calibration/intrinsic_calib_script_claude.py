# calibrate_intrinsics.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json
import time

# ============================================================
# CONFIGURATION — must match your printed board
# ============================================================
BOARD_COLS = 7
BOARD_ROWS = 5
SQUARE_LENGTH = 0.030     # meters — USE YOUR MEASURED VALUE WITH CALIPERS
MARKER_LENGTH = 0.022     # meters

CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

MIN_CORNERS = 6           # minimum corners to accept a frame
MIN_IMAGES = 20           # minimum images before allowing calibration
TARGET_IMAGES = 40        # aim for this many

SAVE_DIR = "GP7_intrinsic_images"

# ============================================================
# SETUP
# ============================================================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard(
    (BOARD_COLS, BOARD_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
)
detector = cv2.aruco.CharucoDetector(board)
max_corners = (BOARD_COLS - 1) * (BOARD_ROWS - 1)


# ============================================================
# STEP 1: CAPTURE IMAGES
# ============================================================
def capture_images():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    profile = pipeline.start(config)

    # Print factory intrinsics for reference
    stream = profile.get_stream(rs.stream.color)
    intr = stream.as_video_stream_profile().get_intrinsics()
    print(f"Factory intrinsics:")
    print(f"  fx={intr.fx:.1f}  fy={intr.fy:.1f}")
    print(f"  cx={intr.ppx:.1f}  cy={intr.ppy:.1f}")

    # Warm up
    print("Warming up camera...")
    for _ in range(30):
        pipeline.wait_for_frames()

    count = 0
    all_corner_positions = []  # for coverage visualization

    print("\n" + "=" * 60)
    print("INTRINSIC CALIBRATION — IMAGE CAPTURE")
    print("=" * 60)
    print(f"Board:       {BOARD_COLS}×{BOARD_ROWS}, "
          f"{SQUARE_LENGTH*1000:.0f}mm squares")
    print(f"Max corners: {max_corners}")
    print(f"Target:      {TARGET_IMAGES} images")
    print()
    print("Controls:")
    print("  SPACE  — capture image")
    print("  C      — show corner coverage heatmap")
    print("  Q      — finish and proceed to calibration")
    print()
    print("Tips:")
    print("  • Hold the board STILL (rolling shutter!)")
    print("  • Move to all 9 regions of the image")
    print("  • Tilt the board ±30° in different directions")
    print("  • Vary distance (250mm to 600mm)")
    print("  • Board should fill 30-80% of the frame")
    print("=" * 60)

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        # Detect board
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)

        # Draw detections
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)

        if corners is not None and len(corners) >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            n = len(corners)

            color = (0, 255, 0) if n > max_corners * 0.7 else (0, 255, 255)
            cv2.putText(display, f"Corners: {n}/{max_corners} — READY",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(display, "Board not detected (or too few corners)",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show capture count and coverage guide
        cv2.putText(display, f"Captured: {count}/{TARGET_IMAGES}",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Draw 3x3 grid overlay as placement guide
        h, w = display.shape[:2]
        for i in range(1, 3):
            cv2.line(display, (w * i // 3, 0), (w * i // 3, h),
                    (50, 50, 50), 1)
            cv2.line(display, (0, h * i // 3), (w, h * i // 3),
                    (50, 50, 50), 1)

        cv2.imshow("Intrinsic Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if corners is None or len(corners) < MIN_CORNERS:
                print("  ✗ Not enough corners — move closer or adjust angle")
                continue

            # Save image
            fname = f"{SAVE_DIR}/intrinsic_{count:03d}.png"
            cv2.imwrite(fname, frame)

            # Track corner positions for coverage
            for c in corners:
                all_corner_positions.append(c[0])

            print(f"  ✓ [{count:2d}] Saved — {len(corners)} corners detected")
            count += 1

        elif key == ord('c'):
            # Show coverage heatmap
            if len(all_corner_positions) > 0:
                show_coverage(all_corner_positions, (CAM_WIDTH, CAM_HEIGHT))

        elif key == ord('q'):
            if count < MIN_IMAGES:
                print(f"  ⚠ Only {count} images — need at least {MIN_IMAGES}")
                print(f"    Keep capturing or press Q again to force quit")
                # Allow a second Q press to force quit
                cv2.waitKey(0)
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\nCaptured {count} images in '{SAVE_DIR}/'")
    return count


# ============================================================
# COVERAGE VISUALIZATION
# ============================================================
def show_coverage(corner_positions, image_size):
    """Show where corners have been detected across the frame."""
    w, h = image_size
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pt in corner_positions:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(heatmap, (x, y), 20, 1.0, -1)

    heatmap = cv2.GaussianBlur(heatmap, (61, 61), 0)

    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Check edge coverage
    margin = 80
    edges = {
        "top":    heatmap[:margin, :].max() > 30,
        "bottom": heatmap[-margin:, :].max() > 30,
        "left":   heatmap[:, :margin].max() > 30,
        "right":  heatmap[:, -margin:].max() > 30,
    }

    y_offset = 30
    for edge_name, covered in edges.items():
        status = "✓" if covered else "✗ NEED MORE HERE"
        color = (0, 255, 0) if covered else (0, 0, 255)
        cv2.putText(heatmap_color, f"{edge_name}: {status}",
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_offset += 30

    cv2.imshow("Corner Coverage (press any key)", heatmap_color)
    cv2.waitKey(0)
    cv2.destroyWindow("Corner Coverage (press any key)")

    all_covered = all(edges.values())
    if all_covered:
        print("  ✓ All edges covered — good distribution")
    else:
        missing = [k for k, v in edges.items() if not v]
        print(f"  ⚠ Missing coverage at: {', '.join(missing)}")
        print(f"    Place the board near the {', '.join(missing)} of the image")


# ============================================================
# STEP 2: RUN CALIBRATION
# ============================================================
def calibrate():
    import glob

    images = sorted(glob.glob(f"{SAVE_DIR}/*.png"))
    print(f"\nProcessing {len(images)} images...")

    all_corners = []
    all_ids = []
    image_size = None
    corner_counts = []
    used_images = []

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        corners, ids, _, _ = detector.detectBoard(gray)

        if corners is not None and len(corners) >= MIN_CORNERS:
            all_corners.append(corners)
            all_ids.append(ids)
            corner_counts.append(len(corners))
            used_images.append(fname)
        else:
            print(f"  Skipping {fname} — insufficient corners")

    print(f"Using {len(all_corners)} / {len(images)} images")
    print(f"Corners per image: min={min(corner_counts)}, "
          f"max={max(corner_counts)}, "
          f"mean={np.mean(corner_counts):.1f}")

    if len(all_corners) < 10:
        print("ERROR: Not enough valid images. Need at least 10.")
        return None, None

    # --- Calibrate ---
    print("\nRunning calibration...")
    flags = 0
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO   # uncomment if fx≈fy expected
    # flags |= cv2.CALIB_FIX_K3             # uncomment to simplify distortion

    #OLD API
    # ret, K, D, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    #     all_corners, all_ids, board, image_size, None, None,
    #     flags=flags
    # )

    all_obj_points = []
    all_img_points = []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        all_obj_points.append(obj_pts)
        all_img_points.append(img_pts)

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, image_size, None, None
    )

    print(f"\n{'='*60}")
    print(f"CALIBRATION RESULTS")
    print(f"{'='*60}")
    print(f"Reprojection error: {ret:.4f} px")
    print(f"\nCamera matrix K:")
    print(f"  fx = {K[0,0]:.2f}")
    print(f"  fy = {K[1,1]:.2f}")
    print(f"  cx = {K[0,2]:.2f}")
    print(f"  cy = {K[1,2]:.2f}")
    print(f"\nDistortion D: {D.flatten()}")
    print(f"  k1 = {D[0,0]:.6f}")
    print(f"  k2 = {D[0,1]:.6f}")
    print(f"  p1 = {D[0,2]:.6f}")
    print(f"  p2 = {D[0,3]:.6f}")
    print(f"  k3 = {D[0,4]:.6f}")

    # --- Quality assessment ---
    print(f"\n{'='*60}")
    print(f"QUALITY ASSESSMENT")
    print(f"{'='*60}")

    if ret < 0.3:
        print(f"  Reprojection error: {ret:.4f} px — EXCELLENT ✓")
    elif ret < 0.5:
        print(f"  Reprojection error: {ret:.4f} px — GOOD ✓")
    elif ret < 1.0:
        print(f"  Reprojection error: {ret:.4f} px — ACCEPTABLE ○")
    else:
        print(f"  Reprojection error: {ret:.4f} px — POOR ✗")
        print(f"  Check: square size measurement, board flatness, focus")

    # Compare with factory
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    rs_profile = pipeline.start(rs_config)
    stream = rs_profile.get_stream(rs.stream.color)
    factory = stream.as_video_stream_profile().get_intrinsics()
    pipeline.stop()

    print(f"\n  Comparison with factory intrinsics:")
    print(f"  {'':20s} {'Custom':>10s} {'Factory':>10s} {'Diff':>10s}")
    print(f"  {'fx':20s} {K[0,0]:>10.2f} {factory.fx:>10.2f} "
          f"{abs(K[0,0]-factory.fx):>10.2f}")
    print(f"  {'fy':20s} {K[1,1]:>10.2f} {factory.fy:>10.2f} "
          f"{abs(K[1,1]-factory.fy):>10.2f}")
    print(f"  {'cx':20s} {K[0,2]:>10.2f} {factory.ppx:>10.2f} "
          f"{abs(K[0,2]-factory.ppx):>10.2f}")
    print(f"  {'cy':20s} {K[1,2]:>10.2f} {factory.ppy:>10.2f} "
          f"{abs(K[1,2]-factory.ppy):>10.2f}")

    # --- Per-image reprojection error ---
    print(f"\n  Per-image reprojection errors:")
    per_image_errors = []
    for i in range(len(all_corners)):
        obj_pts, img_pts = board.matchImagePoints(all_corners[i], all_ids[i])
        projected, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], K, D)
        err = cv2.norm(img_pts, projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(err)

    per_image_errors = np.array(per_image_errors)
    for i, err in enumerate(per_image_errors):
        flag = " ⚠ OUTLIER" if err > ret * 3 else ""
        print(f"    Image {i:2d}: {err:.4f} px{flag}")

    # Suggest removing outliers
    outliers = np.where(per_image_errors > ret * 3)[0]
    if len(outliers) > 0:
        print(f"\n  ⚠ Consider removing outlier images and recalibrating:")
        for idx in outliers:
            print(f"    {used_images[idx]}")

    # --- Save ---
    np.savez("GP7_intrinsics.npz",
        K=K,
        D=D,
        reprojection_error=ret,
        image_width=CAM_WIDTH,
        image_height=CAM_HEIGHT,
        board_cols=BOARD_COLS,
        board_rows=BOARD_ROWS,
        square_length=SQUARE_LENGTH,
        marker_length=MARKER_LENGTH,
        num_images=len(all_corners)
    )
    print(f"\nSaved to intrinsics.npz")

    return K, D


# ============================================================
# STEP 3: VERIFY WITH UNDISTORTION
# ============================================================
def verify():
    """Visual check — show undistorted live feed."""
    data = np.load("GP7_intrinsics.npz")
    K = data["K"]
    D = data["D"]

    # Compute undistortion maps once (faster than undistort per frame)
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (CAM_WIDTH, CAM_HEIGHT), 1, (CAM_WIDTH, CAM_HEIGHT)
    )
    mapx, mapy = cv2.initUndistortRectifyMap(
        K, D, None, new_K, (CAM_WIDTH, CAM_HEIGHT), cv2.CV_32FC1
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    pipeline.start(config)

    for _ in range(30):
        pipeline.wait_for_frames()

    print("\nShowing original (left) vs undistorted (right)")
    print("Look for straight lines near edges — they should be straighter")
    print("Press Q to quit")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # Draw grid overlay to see distortion correction
        for img in [frame, undistorted]:
            for x in range(0, CAM_WIDTH, 80):
                cv2.line(img, (x, 0), (x, CAM_HEIGHT), (0, 50, 0), 1)
            for y in range(0, CAM_HEIGHT, 80):
                cv2.line(img, (0, y), (CAM_WIDTH, y), (0, 50, 0), 1)

        cv2.putText(frame, "Original", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(undistorted, "Undistorted", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined = np.hstack([
            cv2.resize(frame, (640, 360)),
            cv2.resize(undistorted, (640, 360))
        ])

        cv2.imshow("Verification", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        print("Usage:")
        print("  python calibrate_intrinsics.py capture    — capture images")
        print("  python calibrate_intrinsics.py calibrate  — run calibration")
        print("  python calibrate_intrinsics.py verify     — visual check")
        print("  python calibrate_intrinsics.py all        — do everything")
        sys.exit(0)

    if command in ("capture", "all"):
        capture_images()

    if command in ("calibrate", "all"):
        calibrate()

    if command in ("verify", "all"):
        verify()