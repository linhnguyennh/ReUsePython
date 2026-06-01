import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame


# ----------------------------
# DEPTH UTIL
# ----------------------------
def get_line_median_depth(depth_data, x1, y1, x2, y2, depth_scale, num_samples=20):
    xs = np.linspace(x1, x2, num_samples).astype(int)
    ys = np.linspace(y1, y2, num_samples).astype(int)

    h, w = depth_data.shape
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

    depths = depth_data[ys[mask], xs[mask]] * depth_scale
    valid = depths[(depths > 0.1) & (depths < 5.0)]

    return float(np.median(valid)) if len(valid) > num_samples // 2 else None

def resize_to_height(img, h):
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))

# ----------------------------
# CONTOUR → LINE (PCA)
# ----------------------------
def contour_to_line(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)

    if len(pts) < 10:
        return None

    mean = np.mean(pts, axis=0)
    centered = pts - mean

    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]

    projections = centered @ direction

    p1 = mean + direction * projections.min()
    p2 = mean + direction * projections.max()

    return int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])


# ----------------------------
# ROI
# ----------------------------
def select_roi(frame):
    roi = cv2.selectROI(
        "Select ROI — ENTER to confirm, C to cancel",
        frame,
        showCrosshair=True
    )
    cv2.destroyWindow("Select ROI — ENTER to confirm, C to cancel")
    return roi


# ----------------------------
# MAIN
# ----------------------------
def main():

    DEPTH_MIN = 0.2
    DEPTH_MAX = 0.5

    config = realsense_init(
        width=640,
        height=480,
        fps=30,
        enable_spatial=True,
        enable_temporal=True
    )

    print("RealSense initialized.")

    # warm start
    while True:
        color_frame, _ = realsense_get_frame(config)
        if color_frame is not None:
            break

    frame = np.asanyarray(color_frame.get_data())
    roi = select_roi(frame)

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        rx, ry, rw, rh = 0, 0, frame.shape[1], frame.shape[0]

    print(f"ROI set: {rx},{ry},{rw},{rh}")

    try:
        while True:

            color_frame, depth_frame = realsense_get_frame(config)
            if color_frame is None or depth_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            crop = frame[ry:ry+rh, rx:rx+rw]
            crop_depth = depth[ry:ry+rh, rx:rx+rw]

            # ----------------------------
            # PREPROCESS
            # ----------------------------
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(2.0, (8, 8))
            gray = clahe.apply(gray)

            gray = cv2.bilateralFilter(gray, 7, 50, 50)

            v = np.median(gray)
            lower = int(max(0, 0.66 * v))
            upper = int(min(255, 1.33 * v))

            edges = cv2.Canny(gray, lower, upper)

            # NO MORPHOLOGY (important for contours)

            # ----------------------------
            # CONTOURS
            # ----------------------------
            contours, _ = cv2.findContours(
                edges,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            overlay = crop.copy()

            cutting_lines = []

            # ----------------------------
            # FILTER + CONVERT TO LINES
            # ----------------------------
            for c in contours:

                area = cv2.contourArea(c)
                if area < 100:
                    continue

                line = contour_to_line(c)
                if line is None:
                    continue

                x1, y1, x2, y2 = line

                depth_val = get_line_median_depth(
                    crop_depth,
                    x1, y1, x2, y2,
                    config.depth_scale
                )

                if depth_val is None:
                    continue

                if not (DEPTH_MIN <= depth_val <= DEPTH_MAX):
                    continue

                cutting_lines.append((x1, y1, x2, y2, depth_val))

            # ----------------------------
            # DRAW RESULT
            # ----------------------------
            for x1, y1, x2, y2, d in cutting_lines:

                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

                mid = ((x1 + x2) // 2, (y1 + y2) // 2)

                cv2.putText(
                    overlay,
                    f"{d:.2f}m",
                    mid,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

                cv2.circle(overlay, (x1, y1), 4, (0, 255, 255), -1)
                cv2.circle(overlay, (x2, y2), 4, (255, 0, 255), -1)

            # ----------------------------
            # VISUALIZATION
            # ----------------------------
            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            context = frame.copy()
            cv2.rectangle(context, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)

            target_h = context.shape[0]

            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            display = np.hstack([
                context,
                resize_to_height(edges_vis, target_h),
                resize_to_height(overlay, target_h)
            ])

            cv2.imshow("Contour Cutting Pipeline", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()