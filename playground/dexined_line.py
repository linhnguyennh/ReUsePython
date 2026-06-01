import cv2
import numpy as np
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame

try:
    from kornia.contrib.edge_detection import EdgeDetectorBuilder
except ImportError:
    raise SystemExit("kornia not found. Run: pip install kornia")


# ---------------------------------------------------------------------------
# DexiNed wrapper
# ---------------------------------------------------------------------------

class DexiNedEdges:
    def __init__(self, image_size=352, threshold=0.3, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.threshold = threshold  # float [0, 1]

        print(f"Loading DexiNed on {self.device}...")
        self.model = EdgeDetectorBuilder.build(
            model_name="dexined",
            pretrained=True,
            image_size=image_size,
        ).to(self.device)
        self.model.eval()
        print("DexiNed ready.")

    @torch.no_grad()
    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """
        Run DexiNed on a BGR uint8 crop.
        Returns uint8 HxW edge map in the original crop resolution.
        """
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(resized).float().div(255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)   # 1x3xSxS

        out = self.model(t)
        if isinstance(out, (list, tuple)):
            out = out[-1]                                       # fused output

        edge = out.squeeze().cpu().numpy()                      # HxW float [0,1]
        edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
        return (edge * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry helpers  (unchanged from original)
# ---------------------------------------------------------------------------

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
    pts, dirs = [], []
    for x1, y1, x2, y2 in group:
        v = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-6)
        pts += [[x1, y1], [x2, y2]]
        dirs.append(v)

    mean_dir = np.mean(dirs, axis=0)
    mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    projections = (pts - center) @ mean_dir
    p1 = center + mean_dir * projections.min()
    p2 = center + mean_dir * projections.max()
    return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])


def match_line(l1, l2):
    m1 = np.array([(l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2])
    m2 = np.array([(l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2])
    spatial_dist = np.linalg.norm(m1 - m2)
    v1 = np.array([l1[2] - l1[0], l1[3] - l1[1]], dtype=np.float32)
    v2 = np.array([l2[2] - l2[0], l2[3] - l2[1]], dtype=np.float32)
    v1 /= (np.linalg.norm(v1) + 1e-6)
    v2 /= (np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return spatial_dist + 100 * np.arccos(cos_angle)


def select_roi(frame):
    roi = cv2.selectROI("Select ROI — ENTER to confirm, C to cancel",
                        frame, showCrosshair=True)
    cv2.destroyWindow("Select ROI — ENTER to confirm, C to cancel")
    return roi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    DEPTH_MIN       = 0.2
    DEPTH_MAX       = 0.5
    ANGLE_THRESH    = 15.0
    DISTANCE_THRESH = 50.0
    ALPHA           = 0.7     # EMA smoothing (higher = more stable)

    # DexiNed settings — tune threshold to taste
    DEXINED_SIZE      = 352
    DEXINED_THRESHOLD = 128   # uint8 [0-255] binarisation threshold

    # Hough settings for DexiNed edges (slightly looser than Canny — edges are
    # already thin and probabilistic, so gap tolerance can be reduced)
    HOUGH_THRESHOLD   = 40
    HOUGH_MIN_LENGTH  = 30
    HOUGH_MAX_GAP     = 10

    detector = DexiNedEdges(image_size=DEXINED_SIZE)
    device_str = str(detector.device)

    config = realsense_init(width=640, height=480, fps=30,
                            enable_spatial=True, enable_temporal=True)
    print("RealSense initialized.")

    # Grab first frame for ROI selection
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
    print(f"     't'/'T' to decrease/increase edge threshold (current={DEXINED_THRESHOLD})")

    prev_lines = []

    try:
        while True:
            color_frame, depth_frame = realsense_get_frame(config)
            if color_frame is None or depth_frame is None:
                continue

            frame      = np.asanyarray(color_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            crop_color = frame[ry:ry+rh, rx:rx+rw]
            crop_depth = depth_data[ry:ry+rh, rx:rx+rw]

            # ----------------------------------------------------------------
            # Edge detection — DexiNed replaces CLAHE + bilateral + Canny
            # ----------------------------------------------------------------
            edge_map = detector.detect(crop_color)          # uint8 HxW [0-255]
            _, edges = cv2.threshold(edge_map, DEXINED_THRESHOLD,
                                     255, cv2.THRESH_BINARY)

            # ----------------------------------------------------------------
            # Hough line detection on DexiNed edges
            # ----------------------------------------------------------------
            raw_lines = cv2.HoughLinesP(
                edges,
                rho=1, theta=np.pi / 180,
                threshold=HOUGH_THRESHOLD,
                minLineLength=HOUGH_MIN_LENGTH,
                maxLineGap=HOUGH_MAX_GAP,
            )

            # Depth filter
            depth_filtered = []
            if raw_lines is not None:
                for x1, y1, x2, y2 in raw_lines[:, 0]:
                    depth = get_line_median_depth(crop_depth, x1, y1, x2, y2,
                                                  config.depth_scale)
                    if depth is not None and DEPTH_MIN <= depth <= DEPTH_MAX:
                        depth_filtered.append((x1, y1, x2, y2))

            # Group + merge
            groups = group_lines(depth_filtered, ANGLE_THRESH, DISTANCE_THRESH)
            merged = [merge_group(g) for g in groups]

            # EMA temporal smoothing
            smoothed_lines = []
            for line in merged:
                best_prev, best_score = None, 1e9
                for prev in prev_lines:
                    score = match_line(line, prev)
                    if score < best_score:
                        best_score, best_prev = score, prev

                if best_prev is None or best_score > 80:
                    smoothed_lines.append(line)
                else:
                    sx1 = int(ALPHA * best_prev[0] + (1 - ALPHA) * line[0])
                    sy1 = int(ALPHA * best_prev[1] + (1 - ALPHA) * line[1])
                    sx2 = int(ALPHA * best_prev[2] + (1 - ALPHA) * line[2])
                    sy2 = int(ALPHA * best_prev[3] + (1 - ALPHA) * line[3])
                    smoothed_lines.append((sx1, sy1, sx2, sy2))

            prev_lines = smoothed_lines

            # ----------------------------------------------------------------
            # Visualisation  (same layout as original)
            # ----------------------------------------------------------------
            overlay = crop_color.copy()
            for x1, y1, x2, y2 in smoothed_lines:
                depth = get_line_median_depth(crop_depth, x1, y1, x2, y2,
                                              config.depth_scale)
                label = f"{depth:.2f}m" if depth else "?"

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(overlay, (x1, y1), 5, (0, 255, 255), -1)
                cv2.circle(overlay, (x2, y2), 5, (255, 0, 255), -1)
                cv2.putText(overlay, f"({x1},{y1})", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.putText(overlay, f"({x2},{y2})", (x2 + 5, y2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.putText(overlay, label, mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.putText(overlay,
                        f"raw={len(depth_filtered)} merged={len(merged)} "
                        f"thr={DEXINED_THRESHOLD} [{device_str}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            context = frame.copy()
            cv2.rectangle(context, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)

            fh = frame.shape[0]
            def resize_h(img, h):
                s = h / img.shape[0]
                return cv2.resize(img, (int(img.shape[1] * s), h))

            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            display = np.hstack([
                context,
                resize_h(edges_color, fh),
                resize_h(overlay, fh),
            ])
            cv2.imshow("Full | DexiNed edges | Merged Lines", display)

            # ----------------------------------------------------------------
            # Key handling
            # ----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                roi = select_roi(frame)
                rx, ry, rw, rh = roi
                if rw == 0 or rh == 0:
                    rx, ry, rw, rh = 0, 0, frame.shape[1], frame.shape[0]
                prev_lines = []
                print(f"ROI updated: x={rx} y={ry} w={rw} h={rh}")
            elif key == ord('t'):
                DEXINED_THRESHOLD = max(0, DEXINED_THRESHOLD - 10)
                print(f"Edge threshold: {DEXINED_THRESHOLD}")
            elif key == ord('T'):
                DEXINED_THRESHOLD = min(255, DEXINED_THRESHOLD + 10)
                print(f"Edge threshold: {DEXINED_THRESHOLD}")

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()