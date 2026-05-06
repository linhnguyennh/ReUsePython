# handeye_calibration.py

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
import json
import sys
from scipy.spatial.transform import Rotation

# ============================================================
# CONFIGURATION
# ============================================================

# Board — must match what you printed
BOARD_COLS = 7
BOARD_ROWS = 5
SQUARE_LENGTH = 0.030     # meters — YOUR MEASURED VALUE
MARKER_LENGTH = 0.022     # meters

# Camera
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# Paths
INTRINSICS_FILE = "GP7_intrinsics.npz"
POSES_CSV = "GP7_poses.csv"
IMAGE_DIR = "GP7_handeye_images"
RESULT_FILE = "GP7_handeye_result.npz"

# Detection
MIN_CORNERS = 6


# ============================================================
# YASKAWA POSE CONVERSION
# ============================================================

def yaskawa_to_Rt(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg):
    """
    Convert Yaskawa YRC1000 pendant readings to rotation matrix and 
    translation vector.
    
    Yaskawa convention:
      - Position in mm (Base coordinate)
      - Rotation in degrees, Euler ZYX intrinsic
        R = Rz(rz) · Ry(ry) · Rx(rx)
    
    Returns:
      R:  3×3 rotation matrix
      t:  3×1 translation vector in METERS
    """
    # mm to meters
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64).reshape(3, 1) / 1000.0
    
    # Euler ZYX intrinsic (uppercase = intrinsic in scipy)
    # Order of arguments: [rz, ry, rx] for 'ZYX'
    R = Rotation.from_euler(
        'ZYX', [rz_deg, ry_deg, rx_deg], degrees=True
    ).as_matrix()
    
    return R, t


def load_poses_from_csv(filepath):
    """
    Load robot poses from CSV file.
    Format: X(mm), Y(mm), Z(mm), Rx(deg), Ry(deg), Rz(deg)
    Lines starting with # are comments.
    """
    poses = []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty lines and comments
            if not row or row[0].strip().startswith('#'):
                continue
            
            vals = [float(v.strip()) for v in row]
            if len(vals) != 6:
                print(f"  ⚠ Skipping malformed row: {row}")
                continue
            
            x, y, z, rx, ry, rz = vals
            R, t = yaskawa_to_Rt(x, y, z, rx, ry, rz)
            poses.append({
                'R': R,
                't': t,
                'raw': vals
            })
    
    print(f"Loaded {len(poses)} poses from {filepath}")
    return poses


# ============================================================
# BOARD SETUP
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
    """
    Interactive image capture.
    Move the robot to each pose, then press SPACE to capture.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    pipeline.start(config)
    
    # Warm up
    for _ in range(30):
        pipeline.wait_for_frames()
    
    count = 0
    
    print("\n" + "=" * 60)
    print("HAND-EYE CALIBRATION — IMAGE CAPTURE")
    print("=" * 60)
    print()
    print("SETUP:")
    print("  1. Fix the ChArUco board rigidly in the workspace")
    print("     (clamp it — it must NOT move)")
    print("  2. Board should be visible from the robot's workspace")
    print()
    print("FOR EACH POSE:")
    print("  1. Move robot to a new pose (vary rotation!)")
    print("  2. Wait for robot to be completely still")
    print("  3. Check that the board is detected on screen")
    print("  4. Press SPACE to capture")
    print("  5. Write down the pendant values (X,Y,Z,Rx,Ry,Rz)")
    print()
    print("  SPACE = capture    Q = finish")
    print("=" * 60)
    print()
    print("Pose suggestions:")
    print("  Pose 1-3:   Face board straight, vary distance")
    print("  Pose 4-6:   Tilt left/right 15-25°")
    print("  Pose 7-9:   Tilt up/down 15-25°")
    print("  Pose 10-12: Rotate in-plane ±20-30°")
    print("  Pose 13-15: Combined: tilt + shift")
    print("  Pose 16-20: Maximum variety, different quadrants")
    print("=" * 60)
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
        # Detect
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        
        if corners is not None and len(corners) >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            n = len(corners)
            color = (0, 255, 0) if n > max_corners * 0.5 else (0, 255, 255)
            cv2.putText(display, f"Corners: {n}/{max_corners} — READY",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(display, "BOARD NOT DETECTED",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display, f"Captured: {count} poses",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Reminder text
        if count < 15:
            cv2.putText(display, f"Need at least 15 (have {count})",
                       (10, CAM_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (100, 100, 255), 2)
        else:
            cv2.putText(display, f"Have {count} — good to finish (Q)",
                       (10, CAM_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (100, 255, 100), 2)
        
        cv2.imshow("Hand-Eye Capture", display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if corners is None or len(corners) < MIN_CORNERS:
                print(f"  ✗ Not enough corners — adjust robot pose")
                continue
            
            fname = f"{IMAGE_DIR}/pose_{count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"  ✓ Pose {count:2d} captured ({len(corners)} corners)")
            print(f"    → Write down pendant values for this pose!")
            count += 1
        
        elif key == ord('q'):
            if count < 10:
                print(f"\n  ⚠ Only {count} poses — need at least 10")
                print(f"    Press Q again to force quit, or keep capturing")
            else:
                break
    
    pipeline.stop()
    cv2.destroyAllWindows()
    
    print(f"\nCaptured {count} images in '{IMAGE_DIR}/'")
    print(f"\nNEXT STEPS:")
    print(f"  1. Create '{POSES_CSV}' with your pendant readings")
    print(f"  2. One line per pose: X, Y, Z, Rx, Ry, Rz")
    print(f"  3. Run: python {sys.argv[0]} calibrate")
    
    return count


# ============================================================
# STEP 2: INTERACTIVE CAPTURE WITH LIVE POSE ENTRY
# ============================================================

def capture_interactive():
    """
    Capture images and enter robot poses at the same time.
    No CSV file needed — saves everything to JSON.
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    pipeline.start(config)
    
    for _ in range(30):
        pipeline.wait_for_frames()
    
    poses_data = []
    count = 0
    
    print("\n" + "=" * 60)
    print("INTERACTIVE HAND-EYE CAPTURE")
    print("=" * 60)
    print("For each pose:")
    print("  1. Move robot, wait for it to stop")
    print("  2. Press SPACE when board is detected")
    print("  3. Switch to terminal and type pendant values")
    print("  4. Repeat")
    print()
    print("Controls:")
    print("  SPACE  — capture image + enter pose")
    print("  D      — delete last pose (undo)")
    print("  L      — list all captured poses")
    print("  Q      — finish")
    print("=" * 60)
    
    def get_pendant_values(pose_number):
        """
        Prompt user for pendant values with re-entry option.
        Returns (R, t, raw_values) or None if skipped.
        """
        while True:
            print(f"\n--- Pose {pose_number} ---")
            print("Enter pendant values (Base coord):")
            print("  Type 'skip' to discard this capture")
            
            try:
                x_input = input(f"  X  (mm):  ").strip()
                if x_input.lower() == 'skip':
                    return None
                x = float(x_input)
                
                y_input = input(f"  Y  (mm):  ").strip()
                if y_input.lower() == 'skip':
                    return None
                y = float(y_input)
                
                z_input = input(f"  Z  (mm):  ").strip()
                if z_input.lower() == 'skip':
                    return None
                z = float(z_input)
                
                rx_input = input(f"  Rx (deg): ").strip()
                if rx_input.lower() == 'skip':
                    return None
                rx = float(rx_input)
                
                ry_input = input(f"  Ry (deg): ").strip()
                if ry_input.lower() == 'skip':
                    return None
                ry = float(ry_input)
                
                rz_input = input(f"  Rz (deg): ").strip()
                if rz_input.lower() == 'skip':
                    return None
                rz = float(rz_input)
                
            except ValueError:
                print("  ✗ Invalid number — let's try again")
                continue
            
            # Show summary and ask for confirmation
            print(f"\n  Entered values:")
            print(f"    X:  {x:>10.3f} mm")
            print(f"    Y:  {y:>10.3f} mm")
            print(f"    Z:  {z:>10.3f} mm")
            print(f"    Rx: {rx:>10.3f}°")
            print(f"    Ry: {ry:>10.3f}°")
            print(f"    Rz: {rz:>10.3f}°")
            
            # Show resulting tool orientation for quick sanity check
            R, t = yaskawa_to_Rt(x, y, z, rx, ry, rz)
            z_axis = R[:, 2]
            print(f"\n  Tool Z-axis in base: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
            
            while True:
                confirm = input(f"\n  Accept? [y]es / [r]e-enter / [s]kip: ").strip().lower()
                
                if confirm in ('y', 'yes', ''):
                    return R, t, [x, y, z, rx, ry, rz]
                elif confirm in ('r', 're-enter', 'redo'):
                    print("  → Re-entering values...")
                    break  # break inner loop, continue outer loop
                elif confirm in ('s', 'skip'):
                    return None
                else:
                    print("  Type 'y', 'r', or 's'")
    
    
    def print_poses_summary():
        """Print all captured poses so far."""
        if not poses_data:
            print("\n  No poses captured yet.")
            return
        
        print(f"\n  {'#':>3}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
              f"{'Rx':>8}  {'Ry':>8}  {'Rz':>8}  Image")
        print("  " + "-" * 75)
        
        for i, p in enumerate(poses_data):
            raw = p['raw_mm_deg']
            print(f"  {i:>3}  {raw[0]:>8.1f}  {raw[1]:>8.1f}  {raw[2]:>8.1f}  "
                  f"{raw[3]:>8.1f}  {raw[4]:>8.1f}  {raw[5]:>8.1f}  "
                  f"{os.path.basename(p['image'])}")
    
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        if corners is not None and len(corners) >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            cv2.putText(display, f"Corners: {len(corners)}/{max_corners} — READY",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "BOARD NOT DETECTED",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display, f"Captured: {len(poses_data)} poses",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Show controls
        cv2.putText(display, "SPACE:capture  D:delete last  L:list  Q:finish",
                   (10, CAM_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
        
        if len(poses_data) < 15:
            cv2.putText(display, f"Need at least 15 (have {len(poses_data)})",
                       (10, CAM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (100, 100, 255), 2)
        else:
            cv2.putText(display, f"Have {len(poses_data)} — good to finish (Q)",
                       (10, CAM_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (100, 255, 100), 2)
        
        cv2.imshow("Hand-Eye Capture", display)
        key = cv2.waitKey(1) & 0xFF
        
        # === CAPTURE ===
        if key == ord(' '):
            if corners is None or len(corners) < MIN_CORNERS:
                print(f"  ✗ Board not detected properly — adjust pose")
                continue
            
            # Save image
            fname = f"{IMAGE_DIR}/pose_{count:03d}.png"
            cv2.imwrite(fname, frame)
            
            # Get pendant values (with re-entry option)
            result = get_pendant_values(count)
            
            if result is None:
                # User chose to skip
                os.remove(fname)
                print(f"  → Pose skipped, image deleted")
                continue
            
            R, t, raw = result
            
            poses_data.append({
                'image': fname,
                'raw_mm_deg': raw,
                'R_gripper2base': R.tolist(),
                't_gripper2base': t.tolist()
            })
            
            print(f"  ✓ Pose {count} saved [{raw[0]:.1f}, {raw[1]:.1f}, "
                  f"{raw[2]:.1f}, {raw[3]:.1f}, {raw[4]:.1f}, {raw[5]:.1f}]")
            count += 1
        
        # === DELETE LAST ===
        elif key == ord('d'):
            if not poses_data:
                print("  Nothing to delete")
                continue
            
            removed = poses_data.pop()
            raw = removed['raw_mm_deg']
            
            # Delete the image file too
            if os.path.exists(removed['image']):
                os.remove(removed['image'])
            
            print(f"\n  ✗ Deleted pose {len(poses_data)}: "
                  f"[{raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f}, "
                  f"{raw[3]:.1f}, {raw[4]:.1f}, {raw[5]:.1f}]")
            print(f"    {len(poses_data)} poses remaining")
        
        # === LIST ALL ===
        elif key == ord('l'):
            print_poses_summary()
        
        # === EDIT SPECIFIC POSE ===
        elif key == ord('e'):
            if not poses_data:
                print("  No poses to edit")
                continue
            
            print_poses_summary()
            try:
                idx_input = input(f"\n  Enter pose number to edit (0-{len(poses_data)-1}): ").strip()
                idx = int(idx_input)
                
                if idx < 0 or idx >= len(poses_data):
                    print(f"  ✗ Invalid index")
                    continue
                
                old_raw = poses_data[idx]['raw_mm_deg']
                print(f"\n  Current values for pose {idx}:")
                print(f"    X:  {old_raw[0]:.3f}  Y:  {old_raw[1]:.3f}  Z:  {old_raw[2]:.3f}")
                print(f"    Rx: {old_raw[3]:.3f}  Ry: {old_raw[4]:.3f}  Rz: {old_raw[5]:.3f}")
                
                result = get_pendant_values(idx)
                
                if result is None:
                    print(f"  → Edit cancelled, keeping original values")
                    continue
                
                R, t, raw = result
                
                # Keep the same image, update the pose
                poses_data[idx]['raw_mm_deg'] = raw
                poses_data[idx]['R_gripper2base'] = R.tolist()
                poses_data[idx]['t_gripper2base'] = t.tolist()
                
                print(f"  ✓ Pose {idx} updated")
                
            except (ValueError, IndexError):
                print(f"  ✗ Invalid input")
        
        # === QUIT ===
        elif key == ord('q'):
            if len(poses_data) < 10:
                print(f"\n  ⚠ Only {len(poses_data)} poses — need at least 10")
                confirm = input("  Quit anyway? [y/n]: ").strip().lower()
                if confirm != 'y':
                    continue
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()
    
    if not poses_data:
        print("\nNo poses captured.")
        return poses_data
    
    # Final review
    print(f"\n{'='*60}")
    print(f"FINAL REVIEW — {len(poses_data)} poses")
    print(f"{'='*60}")
    print_poses_summary()
    
    # Ask for final confirmation
    print()
    confirm = input("  Save and proceed? [y]es / [e]dit a pose / [q]uit without saving: ").strip().lower()
    
    if confirm in ('e', 'edit'):
        # One more chance to edit
        try:
            idx = int(input(f"  Pose number to edit: "))
            result = get_pendant_values(idx)
            if result is not None:
                R, t, raw = result
                poses_data[idx]['raw_mm_deg'] = raw
                poses_data[idx]['R_gripper2base'] = R.tolist()
                poses_data[idx]['t_gripper2base'] = t.tolist()
                print(f"  ✓ Pose {idx} updated")
        except (ValueError, IndexError):
            print(f"  ✗ Invalid — saving as-is")
    
    elif confirm in ('q', 'quit'):
        print("  Discarded all data.")
        return []
    
    # Save to JSON
    with open("handeye_poses.json", "w") as f:
        json.dump(poses_data, f, indent=2)
    
    print(f"\n✓ Saved {len(poses_data)} poses to handeye_poses.json")
    return poses_data

# ============================================================
# STEP 3: CALIBRATE
# ============================================================

def calibrate(source='csv'):
    """
    Run hand-eye calibration.
    source: 'csv' (from robot_poses.csv) or 'json' (from interactive capture)
    """
    # --- Load intrinsics ---
    if not os.path.exists(INTRINSICS_FILE):
        print(f"ERROR: {INTRINSICS_FILE} not found!")
        print(f"Run intrinsic calibration first.")
        return
    
    intr = np.load(INTRINSICS_FILE)
    K = intr["K"]
    D = intr["D"]
    print(f"Loaded intrinsics from {INTRINSICS_FILE}")
    print(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    
    # --- Load robot poses ---
    if source == 'csv':
        poses = load_poses_from_csv(POSES_CSV)
        # Match with images by index
        images = sorted([
            f for f in os.listdir(IMAGE_DIR) 
            if f.endswith('.png')
        ])
        
        if len(poses) != len(images):
            print(f"\n  ⚠ WARNING: {len(poses)} poses but {len(images)} images!")
            print(f"    They must match 1:1 in order.")
            print(f"    Pose 1 in CSV → {IMAGE_DIR}/pose_000.png")
            print(f"    Pose 2 in CSV → {IMAGE_DIR}/pose_001.png")
            print(f"    etc.")
            min_count = min(len(poses), len(images))
            print(f"    Using first {min_count} of each.")
            poses = poses[:min_count]
            images = images[:min_count]
        
        pose_image_pairs = []
        for i, (pose, img_name) in enumerate(zip(poses, images)):
            pose_image_pairs.append({
                'R': pose['R'],
                't': pose['t'],
                'image': os.path.join(IMAGE_DIR, img_name),
                'raw': pose['raw']
            })
    
    elif source == 'json':
        with open("handeye_poses.json") as f:
            data = json.load(f)
        
        pose_image_pairs = []
        for entry in data:
            pose_image_pairs.append({
                'R': np.array(entry['R_gripper2base']),
                't': np.array(entry['t_gripper2base']),
                'image': entry['image'],
                'raw': entry['raw_mm_deg']
            })
        
        print(f"Loaded {len(pose_image_pairs)} poses from handeye_poses.json")
    
    # --- Process each pose: detect board + solvePnP ---
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    used_indices = []
    
    print(f"\nProcessing {len(pose_image_pairs)} pose-image pairs...")
    
    for i, pair in enumerate(pose_image_pairs):
        img = cv2.imread(pair['image'])
        if img is None:
            print(f"  ✗ Pose {i}: could not read {pair['image']}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        
        if corners is None or len(corners) < MIN_CORNERS:
            print(f"  ✗ Pose {i}: not enough corners ({0 if corners is None else len(corners)})")
            continue
        
        obj_pts, img_pts = board.matchImagePoints(corners, ids)
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D)
        
        if not success:
            print(f"  ✗ Pose {i}: solvePnP failed")
            continue
        
        R_t2c, _ = cv2.Rodrigues(rvec)
        
        R_gripper2base_list.append(pair['R'])
        t_gripper2base_list.append(pair['t'])
        R_target2cam_list.append(R_t2c)
        t_target2cam_list.append(tvec)
        used_indices.append(i)
        
        raw = pair['raw']
        dist = np.linalg.norm(tvec) * 1000
        print(f"  ✓ Pose {i}: {len(corners)} corners, "
              f"board at {dist:.0f}mm, "
              f"robot [{raw[0]:.1f}, {raw[1]:.1f}, {raw[2]:.1f}, "
              f"{raw[3]:.1f}, {raw[4]:.1f}, {raw[5]:.1f}]")
    
    n_valid = len(R_gripper2base_list)
    print(f"\nValid pose pairs: {n_valid} / {len(pose_image_pairs)}")
    
    if n_valid < 5:
        print("ERROR: Not enough valid poses. Need at least 5 (recommend 15+).")
        return
    
    # --- Solve with all methods ---
    methods = {
        "TSAI":       cv2.CALIB_HAND_EYE_TSAI,
        "PARK":       cv2.CALIB_HAND_EYE_PARK,
        "HORAUD":     cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF":    cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    
    results = {}
    
    print(f"\n{'='*70}")
    print(f"HAND-EYE CALIBRATION RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Method':<12} {'tx(mm)':>8} {'ty(mm)':>8} {'tz(mm)':>8} "
          f"{'rot(°)':>8} {'det(R)':>8}")
    print("-" * 62)
    
    for name, method in methods.items():
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            method=method
        )
        
        rvec_result, _ = cv2.Rodrigues(R_c2g)
        angle = np.linalg.norm(rvec_result) * 180 / np.pi
        t_mm = t_c2g.flatten() * 1000
        det = np.linalg.det(R_c2g)
        
        print(f"{name:<12} {t_mm[0]:>8.1f} {t_mm[1]:>8.1f} {t_mm[2]:>8.1f} "
              f"{angle:>8.2f} {det:>8.4f}")
        
        results[name] = (R_c2g.copy(), t_c2g.copy())
    
    # --- Check consistency ---
    translations = np.array([r[1].flatten() for r in results.values()]) * 1000
    t_spread = translations.max(axis=0) - translations.min(axis=0)
    
    print(f"\nTranslation spread across methods:")
    print(f"  Δx = {t_spread[0]:.1f} mm")
    print(f"  Δy = {t_spread[1]:.1f} mm")
    print(f"  Δz = {t_spread[2]:.1f} mm")
    
    if np.all(t_spread < 5.0):
        print("  ✓ Methods agree — data quality is GOOD")
    elif np.all(t_spread < 15.0):
        print("  ○ Methods partially agree — acceptable but could be better")
        print("    Consider: more poses, more rotation variety, check Euler convention")
    else:
        print("  ✗ Methods DISAGREE — likely issue with:")
        print("    1. Euler angle convention (most common)")
        print("    2. Not enough rotation variety in poses")
        print("    3. Base vs World coordinate on pendant")
        print("    4. Wrong tool frame active on pendant")
    
    # --- Select best result (PARK is generally robust) ---
    R_cam2gripper, t_cam2gripper = results["PARK"]
    
    # Build 4x4
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    print(f"\n{'='*70}")
    print(f"SELECTED RESULT (PARK)")
    print(f"{'='*70}")
    print(f"\nTranslation (camera to TCP):")
    t_mm = t_cam2gripper.flatten() * 1000
    print(f"  x = {t_mm[0]:>8.2f} mm")
    print(f"  y = {t_mm[1]:>8.2f} mm")
    print(f"  z = {t_mm[2]:>8.2f} mm")
    print(f"  distance = {np.linalg.norm(t_mm):>8.2f} mm")
    
    euler = Rotation.from_matrix(R_cam2gripper).as_euler('ZYX', degrees=True)
    print(f"\nRotation (camera to TCP, Euler ZYX):")
    print(f"  Rz = {euler[0]:>8.2f}°")
    print(f"  Ry = {euler[1]:>8.2f}°")
    print(f"  Rx = {euler[2]:>8.2f}°")
    
    print(f"\n4×4 Transform:\n{T_cam2gripper}")
    
    print(f"\n→ Does the translation roughly match where the camera")
    print(f"  is physically mounted relative to the TCP flange?")
    
    # --- Save ---
    np.savez(RESULT_FILE,
        R=R_cam2gripper,
        t=t_cam2gripper,
        T=T_cam2gripper,
        method="PARK",
        num_poses=n_valid,
        board_cols=BOARD_COLS,
        board_rows=BOARD_ROWS,
        square_length=SQUARE_LENGTH
    )
    print(f"\nSaved to {RESULT_FILE}")
    
    # --- Run validation ---
    validate(R_cam2gripper, t_cam2gripper,
             R_gripper2base_list, t_gripper2base_list,
             R_target2cam_list, t_target2cam_list)
    
    return T_cam2gripper


# ============================================================
# STEP 4: VALIDATE
# ============================================================

def validate(R_c2g, t_c2g,
             R_g2b_list, t_g2b_list,
             R_t2c_list, t_t2c_list):
    """
    Validation: the target is fixed, so its position in base frame
    should be the SAME regardless of which robot pose we compute it from.
    """
    
    def to_T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.array(t).flatten()
        return T
    
    T_c2g = to_T(R_c2g, t_c2g)
    
    target_positions = []
    target_rotations = []
    
    for i in range(len(R_g2b_list)):
        T_g2b = to_T(R_g2b_list[i], t_g2b_list[i])
        T_t2c = to_T(R_t2c_list[i], t_t2c_list[i])
        
        # Chain: base ← gripper ← camera ← target
        T_target2base = T_g2b @ T_c2g @ T_t2c
        
        target_positions.append(T_target2base[:3, 3])
        target_rotations.append(T_target2base[:3, :3])
    
    target_positions = np.array(target_positions)
    mean_pos = target_positions.mean(axis=0)
    errors_mm = np.linalg.norm(target_positions - mean_pos, axis=1) * 1000
    
    print(f"\n{'='*70}")
    print(f"VALIDATION — Target Position Consistency")
    print(f"{'='*70}")
    print(f"\nTarget position in base frame (should be constant):")
    print(f"  Mean:  [{mean_pos[0]*1000:.1f}, {mean_pos[1]*1000:.1f}, "
          f"{mean_pos[2]*1000:.1f}] mm")
    
    print(f"\nPer-pose error from mean:")
    for i, err in enumerate(errors_mm):
        flag = " ⚠" if err > errors_mm.mean() * 2.5 else ""
        print(f"  Pose {i:2d}: {err:.2f} mm{flag}")
    
    print(f"\nSummary:")
    print(f"  Mean error:  {errors_mm.mean():.2f} mm")
    print(f"  Max error:   {errors_mm.max():.2f} mm")
    print(f"  Std dev:     {errors_mm.std():.2f} mm")
    
    if errors_mm.mean() < 1.0:
        print(f"  → EXCELLENT calibration ✓")
    elif errors_mm.mean() < 2.0:
        print(f"  → GOOD calibration ✓")
    elif errors_mm.mean() < 5.0:
        print(f"  → ACCEPTABLE calibration ○")
    else:
        print(f"  → POOR calibration ✗")
        print(f"    Check Euler convention, pose variety, board flatness")
    
    # Rotation consistency
    mean_euler = []
    for R in target_rotations:
        e = Rotation.from_matrix(R).as_euler('ZYX', degrees=True)
        mean_euler.append(e)
    mean_euler = np.array(mean_euler)
    euler_std = mean_euler.std(axis=0)
    
    print(f"\nRotation consistency (std dev):")
    print(f"  Rz: {euler_std[0]:.3f}°")
    print(f"  Ry: {euler_std[1]:.3f}°")
    print(f"  Rx: {euler_std[2]:.3f}°")


# ============================================================
# STEP 5: VERIFY WITH LIVE FEED
# ============================================================

def verify_live():
    """
    Live visualization: project board axes using the calibration.
    Move the robot around — the target position in base frame
    should stay constant.
    """
    if not os.path.exists(RESULT_FILE):
        print(f"ERROR: {RESULT_FILE} not found. Run calibration first.")
        return
    
    result = np.load(RESULT_FILE)
    T_c2g = result["T"]
    
    intr = np.load(INTRINSICS_FILE)
    K, D_coeffs = intr["K"], intr["D"]
    
    print("\n" + "=" * 60)
    print("LIVE VERIFICATION")
    print("=" * 60)
    print("Move the robot around while keeping the board visible.")
    print("The 'Target in Base' position should stay CONSTANT.")
    print("Press Q to quit.")
    print("=" * 60)
    print()
    print("You need to enter the CURRENT robot pose.")
    print("For a proper live test, you'd need MotoROS2.")
    print("For now, enter the starting pose and observe detection.")
    
    # For demo purposes — in practice connect to MotoROS2 for live poses
    print("\nEnter current pendant values:")
    try:
        x  = float(input("  X  (mm):  "))
        y  = float(input("  Y  (mm):  "))
        z  = float(input("  Z  (mm):  "))
        rx = float(input("  Rx (deg): "))
        ry = float(input("  Ry (deg): "))
        rz = float(input("  Rz (deg): "))
    except ValueError:
        print("Invalid input")
        return
    
    R_g2b, t_g2b = yaskawa_to_Rt(x, y, z, rx, ry, rz)
    T_g2b = np.eye(4)
    T_g2b[:3, :3] = R_g2b
    T_g2b[:3, 3] = t_g2b.flatten()
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, CAM_WIDTH, CAM_HEIGHT, rs.format.bgr8, CAM_FPS
    )
    pipeline.start(config)
    
    for _ in range(30):
        pipeline.wait_for_frames()
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        
        if corners is not None and len(corners) >= MIN_CORNERS:
            cv2.aruco.drawDetectedCornersCharuco(display, corners, ids)
            
            obj_pts, img_pts = board.matchImagePoints(corners, ids)
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D_coeffs)
            
            if success:
                cv2.drawFrameAxes(display, K, D_coeffs, rvec, tvec,
                                 SQUARE_LENGTH * 2)
                
                R_t2c, _ = cv2.Rodrigues(rvec)
                T_t2c = np.eye(4)
                T_t2c[:3, :3] = R_t2c
                T_t2c[:3, 3] = tvec.flatten()
                
                T_target_in_base = T_g2b @ T_c2g @ T_t2c
                pos = T_target_in_base[:3, 3] * 1000
                
                cv2.putText(display,
                    f"Target in Base: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                dist = np.linalg.norm(tvec) * 1000
                cv2.putText(display, f"Board distance: {dist:.0f} mm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        else:
            cv2.putText(display, "Board not detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Live Verification", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    pipeline.stop()
    cv2.destroyAllWindows()


# ============================================================
# EULER CONVENTION VERIFICATION
# ============================================================

def verify_euler():
    """
    Run this FIRST to make sure the Euler convention is correct.
    """
    print("\n" + "=" * 60)
    print("EULER CONVENTION VERIFICATION")
    print("=" * 60)
    print()
    print("Test 1: Move robot so TCP points straight down (typical home)")
    print("Pendant should show something like: Rx≈180, Ry≈0, Rz≈0")
    print()
    
    rx, ry, rz = 180, 0, 0
    R, _ = yaskawa_to_Rt(0, 0, 0, rx, ry, rz)
    print(f"For Rx={rx}°, Ry={ry}°, Rz={rz}°:")
    print(f"R =\n{R}")
    print(f"  → Z-axis of tool: {R[:, 2]} (should be [0, 0, -1] for pointing down)")
    
    print()
    print("Test 2: Now rotate the tool 90° around Z on the pendant")
    print("(Rz should change by ~90°)")
    
    rz = 90
    R90, _ = yaskawa_to_Rt(0, 0, 0, 180, 0, 90)
    print(f"\nFor Rx=180°, Ry=0°, Rz=90°:")
    print(f"R =\n{R90}")
    print(f"  → X-axis of tool: {R90[:, 0]}")
    print(f"  → Should have rotated 90° in XY plane")
    
    print()
    print("Compare these with what you physically see on the robot.")
    print("If they DON'T match, the Euler convention needs adjusting.")
    print()
    print("Common alternatives to try:")
    print("  Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True)  ← current")
    print("  Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)  ← equivalent")
    print("  Rotation.from_euler('XYZ', [rx, ry, rz], degrees=True)  ← different!")
    print("  Rotation.from_euler('zyx', [rz, ry, rx], degrees=True)  ← different!")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    commands = {
        'capture':     ('Capture images (enter poses in CSV later)', capture_images),
        'interactive': ('Capture images + enter poses live',         capture_interactive),
        'calibrate':   ('Run calibration from CSV',                  lambda: calibrate('csv')),
        'calibrate_json': ('Run calibration from JSON',              lambda: calibrate('json')),
        'verify_live': ('Live verification',                         verify_live),
        'verify_euler': ('Verify Euler convention',                  verify_euler),
    }
    
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("Usage:")
        print(f"  python {sys.argv[0]} <command>\n")
        print("Commands:")
        for cmd, (desc, _) in commands.items():
            print(f"  {cmd:<18s} — {desc}")
        print()
        print("Typical workflow:")
        print(f"  1. python {sys.argv[0]} verify_euler      ← do this FIRST")
        print(f"  2. python {sys.argv[0]} interactive       ← capture + enter poses")
        print(f"  3. python {sys.argv[0]} calibrate_json   ← run calibration")
        print(f"  4. python {sys.argv[0]} verify_live      ← check result")
        print()
        print("OR:")
        print(f"  1. python {sys.argv[0]} verify_euler")
        print(f"  2. python {sys.argv[0]} capture          ← capture images only")
        print(f"  3. Fill in {POSES_CSV} with pendant values")
        print(f"  4. python {sys.argv[0]} calibrate        ← run calibration")
        sys.exit(0)
    
    cmd = sys.argv[1]
    commands[cmd][1]()