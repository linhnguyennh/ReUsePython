import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame


def get_line_median_depth(depth_data, x1, y1, x2, y2, depth_scale, num_samples=20):
    xs = np.linspace(x1, x2, num_samples).astype(int)
    ys = np.linspace(y1, y2, num_samples).astype(int)
    h, w = depth_data.shape
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    depths = depth_data[ys[mask], xs[mask]] * depth_scale
    valid = depths[(depths > 0.1) & (depths < 5.0)]
    return float(np.median(valid)) if len(valid) >= num_samples // 2 else None


def line_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def line_midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def line_distance(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    mid1 = np.array([(x1 + x2)/2, (y1 + y2)/2])
    mid2 = np.array([(x3 + x4)/2, (y3 + y4)/2])

    return np.linalg.norm(mid1 - mid2)


def merge_lines(lines, angle_thresh=10.0, distance_thresh=20.0):
    """
    Merge lines that are nearly parallel and spatially close.
    Groups lines by similar angle, then clusters by midpoint proximity.
    Returns list of merged (x1, y1, x2, y2).
    """
    if not lines:
        return []

    segments = []
    for x1, y1, x2, y2 in lines:
        angle = line_angle(x1, y1, x2, y2)
        mid = line_midpoint(x1, y1, x2, y2)
        segments.append({'pts': (x1, y1, x2, y2), 'angle': angle, 'mid': mid})

    used = [False] * len(segments)
    merged = []

    for i, seg in enumerate(segments):
        if used[i]:
            continue
        group = [seg]
        used[i] = True
        for j, other in enumerate(segments):
            if used[j]:
                continue
            angle_diff = abs(seg['angle'] - other['angle'])
            angle_diff = min(angle_diff, 180 - angle_diff)
            if angle_diff > angle_thresh:
                continue
            if point_distance(seg['mid'], other['mid']) > distance_thresh:
                continue
            group.append(other)
            used[j] = True

        # Merge group: project all endpoints onto the dominant direction
        all_pts = np.array([s['pts'] for s in group])
        all_xy = all_pts.reshape(-1, 2).astype(np.float32)  # (N*2, 2)

        mean_pt = all_xy.mean(axis=0)
        centered = all_xy - mean_pt
        _, _, vt = np.linalg.svd(centered)
        direction = vt[0]  # principal axis

        projections = centered @ direction
        p_min = mean_pt + direction * projections.min()
        p_max = mean_pt + direction * projections.max()

        merged.append((int(p_min[0]), int(p_min[1]), int(p_max[0]), int(p_max[1])))

    return merged

def group_lines(lines, angle_thresh=10.0, distance_thresh=20.0):
    if not lines:
        return []

    segments = []
    for x1, y1, x2, y2 in lines:
        angle = line_angle(x1, y1, x2, y2)
        mid = line_midpoint(x1, y1, x2, y2)
        segments.append((x1, y1, x2, y2, angle, mid))

    used = [False] * len(segments)
    groups = []

    for i, s in enumerate(segments):
        if used[i]:
            continue

        group = [(s[0], s[1], s[2], s[3])]
        used[i] = True

        for j, o in enumerate(segments):
            if used[j]:
                continue

            angle_diff = abs(s[4] - o[4])
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff > angle_thresh:
                continue

            if np.linalg.norm(np.array(s[5]) - np.array(o[5])) > distance_thresh:
                continue

            group.append((o[0], o[1], o[2], o[3]))
            used[j] = True

        groups.append(group)

    return groups

def merge_group(group):
    pts = []
    dirs = []

    for x1, y1, x2, y2 in group:
        v = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-6)

        pts.append([x1, y1])
        pts.append([x2, y2])
        dirs.append(v)

    mean_dir = np.mean(dirs, axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)

    if np.linalg.norm(mean_dir) < 1e-6:
        return group[0]
    pts = np.array(pts)

    center = np.mean(pts, axis=0)

    projections = (pts - center) @ mean_dir

    p1 = center + mean_dir * projections.min()
    p2 = center + mean_dir * projections.max()

    return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])

def select_roi(frame):
    roi = cv2.selectROI("Select ROI — ENTER to confirm, C to cancel", frame, showCrosshair=True)
    cv2.destroyWindow("Select ROI — ENTER to confirm, C to cancel")
    return roi

def match_line(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2

    # midpoints
    m1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    m2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])

    spatial_dist = np.linalg.norm(m1 - m2)

    # direction vectors
    v1 = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    v2 = np.array([x4 - x3, y4 - y3], dtype=np.float32)

    v1 /= (np.linalg.norm(v1) + 1e-6)
    v2 /= (np.linalg.norm(v2) + 1e-6)

    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_diff = np.arccos(cos_angle)  # radians
    # normalized cost
    return spatial_dist + 100 * angle_diff

def main():
    DEPTH_MIN = 0.2
    DEPTH_MAX = 0.5
    ANGLE_THRESH = 15.0      # degrees — lines within this are "parallel"
    DISTANCE_THRESH = 50.0   # pixels — midpoints within this get merged

    config = realsense_init(width=640, height=480, fps=30,
                            enable_spatial=True, enable_temporal=True)
    print("RealSense initialized.")

    prev_lines = []
    ALPHA = 0.7  # higher = more stable, lower = more responsive

    while True:
        color_frame, depth_frame = realsense_get_frame(config)
        if color_frame is not None:
            break
    first_frame = np.asanyarray(color_frame.get_data())

    roi = select_roi(first_frame)
    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        rx, ry, rw, rh = 0, 0, first_frame.shape[1], first_frame.shape[0]
    print(f"ROI: x={rx} y={ry} w={rw} h={rh}. Press 'r' to reselect, 'q' to quit.")

    try:
        while True:
            color_frame, depth_frame = realsense_get_frame(config)
            if color_frame is None or depth_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            crop_color = frame[ry:ry+rh, rx:rx+rw]
            crop_depth = depth_data[ry:ry+rh, rx:rx+rw]

            gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
            
            #constrast normalization
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

           
            #Adaptive canny
            
            #blurred = cv2.GaussianBlur(gray, (5, 5), 1.5) #uniform blur
            blurred = cv2.bilateralFilter(gray, 7, 50, 50) #spatial aware

            v = np.median(blurred)

            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))

            edges = cv2.Canny(blurred, lower, upper)

            kernel = np.ones((3,3), np.uint8)
            #edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
            #edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                        threshold=60, minLineLength=40, maxLineGap=15)

            # Depth filter raw lines first
            depth_filtered = []
            if raw_lines is not None:
                for x1, y1, x2, y2 in raw_lines[:, 0]:
                    depth = get_line_median_depth(crop_depth, x1, y1, x2, y2, config.depth_scale)
                    if depth is not None and DEPTH_MIN <= depth <= DEPTH_MAX:
                        depth_filtered.append((x1, y1, x2, y2))

            # Merge neighboring lines
           # merged = merge_lines(depth_filtered, ANGLE_THRESH, DISTANCE_THRESH)

            groups = group_lines(depth_filtered, ANGLE_THRESH, DISTANCE_THRESH)

            merged = []
            for g in groups:
                merged.append(merge_group(g))

            smoothed_lines = []

            for line in merged:
                best_prev = None
                best_score = 1e9

                for prev in prev_lines:
                    score = match_line(line, prev)
                    if score < best_score:
                        best_score = score
                        best_prev = prev

                if best_prev is None or best_score > 80:
                    # new line
                    smoothed_lines.append(line)
                else:
                    # EMA smoothing
                    x1 = int(ALPHA * best_prev[0] + (1 - ALPHA) * line[0])
                    y1 = int(ALPHA * best_prev[1] + (1 - ALPHA) * line[1])
                    x2 = int(ALPHA * best_prev[2] + (1 - ALPHA) * line[2])
                    y2 = int(ALPHA * best_prev[3] + (1 - ALPHA) * line[3])

                    smoothed_lines.append((x1, y1, x2, y2))

            prev_lines = smoothed_lines



            overlay = crop_color.copy()
            for x1, y1, x2, y2 in smoothed_lines:
                depth = get_line_median_depth(crop_depth, x1, y1, x2, y2, config.depth_scale)
                label = f"{depth:.2f}m" if depth else "?"

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Endpoints
                cv2.circle(overlay, (x1, y1), 5, (0, 255, 255), -1)
                cv2.circle(overlay, (x2, y2), 5, (255, 0, 255), -1)
                cv2.putText(overlay, f"({x1},{y1})", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(overlay, f"({x2},{y2})", (x2 + 5, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.putText(overlay, label, mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(overlay, f"raw={len(depth_filtered)} merged={len(merged)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            context = frame.copy()
            cv2.rectangle(context, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)

            fh = frame.shape[0]
            def resize_to_height(img, h):
                scale = h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * scale), h))

            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display = np.hstack([context, resize_to_height(edges_color, fh), resize_to_height(overlay, fh)])
            cv2.imshow("Full | Canny | Merged Lines", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                roi = select_roi(frame)
                rx, ry, rw, rh = roi
                if rw == 0 or rh == 0:
                    rx, ry, rw, rh = 0, 0, frame.shape[1], frame.shape[0]
                print(f"ROI updated: x={rx} y={ry} w={rw} h={rh}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()