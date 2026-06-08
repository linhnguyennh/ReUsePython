"""
Seam detection for battery cell presented to a fixed spindle.

Finds the horizontal seam offset relative to a taught spindle centerline
using a 1D vertical Sobel profile — no learned model needed.

Usage:
    python seam_detect.py

Controls:
    t       - teach spindle centerline (set y_spindle = current cut target)
    r       - reset taught position
    s       - save current frame
    m       - toggle long/short edge mode at runtime
    q/ESC   - quit

Output (printed each frame):
    cut delta px  : cut target offset from spindle in pixels
    cut delta mm  : cut target offset in mm (requires MM_PER_PIXEL calibration)
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.realsense_frame import realsense_init, realsense_get_frame
from src.utils.draw_rs import draw_depth
# ---------------------------------------------------------------------------
# Mode switch — True = long edge, False = short edge
# ---------------------------------------------------------------------------

IS_LONG_EDGE = True

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

SEARCH_BAND_PX  = 40      # px above/below y_spindle to search
MM_PER_PIXEL    = None    # set after calibration; None = report px only
SMOOTH_ALPHA    = 0.6     # EMA smoothing on seam row (0=no smooth, 1=frozen)
PROFILE_WIDTH   = 200     # width of the profile panel in the display
DEPTH_MIN = 0.1
DEPTH_MAX = 0.30

# Cut offset: how many px ABOVE the detected seam the cut target should be.
# Positive = shift upward in image (toward smaller row index = toward casing body).
# Set to 0 to cut exactly at the seam.
CUT_OFFSET_PX   = 20      # px above seam → on the casing body

# ---------------------------------------------------------------------------
# Per-mode ROI config
# ---------------------------------------------------------------------------

ROI_CONFIG = {
    "long": {
        "x_start": 100,
        "x_end":   540,
        "label":   "LONG EDGE",
        "color":   (50, 50, 200),   # blue search band
    },
    "short": {
        "x_start": 220,
        "x_end":   420,
        "label":   "SHORT EDGE",
        "color":   (200, 50, 50),   # red search band
    },
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_seam(gray_roi: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Run vertical Sobel on gray ROI, collapse to 1D profile, return peak row.

    Args:
        gray_roi: grayscale crop of shape (2*SEARCH_BAND, width)
    Returns:
        (peak_row_in_roi, profile_1d, sobel_raw)
    """
    sobel = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.abs(sobel)
    sobel_raw = sobel.copy()

    # Suppress weak responses (noise floor)
    sobel = np.where(sobel > sobel.max() * 0.1, sobel, 0.0)

    profile = sobel.mean(axis=1)   # shape: (2*SEARCH_BAND,)
    peak = int(np.argmax(profile))
    return peak, profile, sobel_raw


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_profile_panel(profile: np.ndarray, peak_row: int,
                       panel_h: int, panel_w: int = PROFILE_WIDTH) -> np.ndarray:
    """Render 1D profile as a horizontal bar chart panel."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    if profile.max() < 1e-6:
        return panel

    norm = profile / profile.max()

    for row_i, val in enumerate(norm):
        bar_len = int(val * (panel_w - 10))
        y      = int(row_i       * panel_h / len(norm))
        y_next = int((row_i + 1) * panel_h / len(norm))
        cv2.rectangle(panel, (0, y), (bar_len, y_next - 1), (0, 200, 255), -1)

    # Seam peak line
    peak_y = int(peak_row * panel_h / len(profile))
    cv2.line(panel, (0, peak_y), (panel_w, peak_y), (0, 255, 0), 2)
    cv2.putText(panel, "seam", (4, max(peak_y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Cut offset line
    cut_row = peak_row - CUT_OFFSET_PX
    if 0 <= cut_row < len(profile):
        cut_y = int(cut_row * panel_h / len(profile))
        cv2.line(panel, (0, cut_y), (panel_w, cut_y), (0, 100, 255), 1)
        cv2.putText(panel, "cut", (4, max(cut_y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    return panel


def build_sobel_vis(sobel_raw: np.ndarray, peak_local: int) -> np.ndarray:
    if sobel_raw.max() > 0:
        vis = (255 * sobel_raw / sobel_raw.max()).astype(np.uint8)
    else:
        vis = np.zeros_like(sobel_raw, dtype=np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, peak_local), (vis.shape[1], peak_local), (0, 255, 0), 2)
    cut_row = peak_local - CUT_OFFSET_PX
    if 0 <= cut_row < vis.shape[0]:
        cv2.line(vis, (0, cut_row), (vis.shape[1], cut_row), (0, 100, 255), 2)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global IS_LONG_EDGE

    config = realsense_init(width=640, height=480, fps=30,
                            enable_spatial=True, enable_temporal=True)
    print("RealSense initialized.")

    # Grab first frame
    while True:
        color_frame, depth_frame = realsense_get_frame(config)
        if color_frame is not None:
            break
    first = np.asanyarray(color_frame.get_data())
    fh, fw = first.shape[:2]

    y_spindle       = fh // 2 + 100
    seam_row_smooth = float(y_spindle)
    save_idx        = 0

    print(f"Initial spindle y={y_spindle}.")
    print(f"Mode: {'LONG' if IS_LONG_EDGE else 'SHORT'} EDGE  |  "
          f"Cut offset: {CUT_OFFSET_PX}px above seam")
    print("Press 't' to teach, 'm' to toggle mode, 'q' to quit.")

    try:
        while True:
            color_frame, depth_frame = realsense_get_frame(config)
            if color_frame is None or depth_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Active ROI config
            mode_key = "long" if IS_LONG_EDGE else "short"
            cfg      = ROI_CONFIG[mode_key]
            x0 = max(0,  cfg["x_start"])
            x1 = min(fw, cfg["x_end"])
            y0 = max(0,  y_spindle - SEARCH_BAND_PX)
            y1 = min(fh, y_spindle + SEARCH_BAND_PX)

            roi_color = frame[y0:y1, x0:x1]
            roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

            peak_local, profile, sobel_raw = detect_seam(roi_gray)

            seam_row_global = y0 + peak_local

            # EMA temporal smoothing on seam row
            seam_row_smooth = (SMOOTH_ALPHA * seam_row_smooth
                               + (1 - SMOOTH_ALPHA) * seam_row_global)
            seam_row_int = int(round(seam_row_smooth))

            depth_data = np.asanyarray(depth_frame.get_data())

            

            # Cut target = seam shifted upward by CUT_OFFSET_PX
            cut_row_int = seam_row_int - CUT_OFFSET_PX
            
            #Filter depth
            cut_depths = depth_data[cut_row_int, x0:x1] * config.depth_scale
            valid_mask = (cut_depths > DEPTH_MIN) & (cut_depths < DEPTH_MAX)

            # Find leftmost and rightmost valid depth pixel on the seam row
            valid_cols = np.where(valid_mask)[0]
            if len(valid_cols) > 0:
                x_left  = x0 + valid_cols[0]
                x_right = x0 + valid_cols[-1]


            # Delta: how far cut target is from spindle centerline
            delta_px = cut_row_int - y_spindle
            delta_mm = (delta_px * MM_PER_PIXEL) if MM_PER_PIXEL else None

            # ----------------------------------------------------------------
            # Visualisation
            # ----------------------------------------------------------------
            vis = frame.copy()

            depth_panel = draw_depth(depth_data, fw, fh)

            # Search band rectangle
            cv2.rectangle(vis, (x0, y0), (x1, y1), cfg["color"], 1)

            # Mode label
            cv2.putText(vis, cfg["label"], (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, cfg["color"], 1)

            # Spindle centerline
            cv2.line(vis, (x0, y_spindle), (x1, y_spindle), (0, 255, 255), 1)
            cv2.putText(vis, "spindle", (x1 + 4, y_spindle + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Detected seam
            cv2.line(vis, (x0, seam_row_int), (x1, seam_row_int), (0, 255, 0), 2)
            cv2.putText(vis, "seam", (x1 + 4, seam_row_int + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Cut target line
            cv2.line(vis, (x_left, cut_row_int), (x_right, cut_row_int), (0, 100, 255), 2)
            cv2.putText(vis, f"cut  ({CUT_OFFSET_PX}px offset)",
                        (x1 + 4, cut_row_int + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

            # Offset arrow: spindle → cut target
            if abs(delta_px) > 2:
                cv2.arrowedLine(vis,
                                ((x0 + x1) // 2, y_spindle),
                                ((x0 + x1) // 2, cut_row_int),
                                (0, 100, 255), 2, tipLength=0.3)

            # HUD
            offset_str = (f"cut delta: {delta_px:+d}px  {delta_mm:+.2f}mm"
                          if delta_mm is not None
                          else f"cut delta: {delta_px:+d}px")
            cv2.putText(vis, offset_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis,
                        f"spindle y={y_spindle}  seam y={seam_row_int}  "
                        f"cut y={cut_row_int}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.putText(vis,
                        "[t]teach [r]reset [m]toggle mode [s]save [q]quit",
                        (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

            # Profile panel — embedded next to ROI
            roi_h = y1 - y0
            profile_panel = draw_profile_panel(profile, peak_local, roi_h)
            panel_x = x1 + 2
            if panel_x + PROFILE_WIDTH <= fw:
                vis[y0:y1, panel_x:panel_x + PROFILE_WIDTH] = profile_panel
            else:
                pw = fw - panel_x
                if pw > 10:
                    vis[y0:y1, panel_x:fw] = profile_panel[:, :pw]

            # Sobel window
            sobel_vis = build_sobel_vis(sobel_raw, peak_local)

            cv2.imshow("Seam Detection", vis)
            cv2.imshow("Sobel ROI", sobel_vis)
            cv2.imshow("Depth", depth_panel)

            print(f"\r[{cfg['label']}]  {offset_str}  "
                  f"seam={seam_row_int}  cut={cut_row_int}",
                  end="", flush=True)

            # ----------------------------------------------------------------
            # Keys
            # ----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('t'):
                # Teach spindle to the current cut target position
                y_spindle = cut_row_int
                seam_row_smooth = float(seam_row_int)
                print(f"\nTaught spindle y={y_spindle} (cut target)")
            elif key == ord('r'):
                y_spindle = fh // 2
                seam_row_smooth = float(y_spindle)
                print(f"\nReset spindle y={y_spindle}")
            elif key == ord('m'):
                IS_LONG_EDGE = not IS_LONG_EDGE
                seam_row_smooth = float(y_spindle)   # reset smooth on mode change
                print(f"\nMode → {'LONG' if IS_LONG_EDGE else 'SHORT'} EDGE")
            elif key == ord('s'):
                fname = f"seam_{save_idx:04d}.png"
                cv2.imwrite(fname, vis)
                print(f"\nSaved {fname}")
                save_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()
        print()


if __name__ == "__main__":
    main()