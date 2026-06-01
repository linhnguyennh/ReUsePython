"""
Seam detection for battery cell presented to a fixed spindle.

Finds the horizontal seam offset relative to a taught spindle centerline
using a 1D vertical Sobel profile — no learned model needed.

Usage:
    python seam_detect.py

Controls:
    t       - teach spindle centerline (set y_spindle = current seam estimate)
    r       - reset taught position
    s       - save current frame + profile plot
    q/ESC   - quit

Output (printed each frame):
    delta_px  : seam offset from spindle in pixels
    delta_mm  : seam offset in mm (requires mm_per_pixel calibration)
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.realsense_frame import realsense_init, realsense_get_frame

# ---------------------------------------------------------------------------
# Config — tune these for your setup
# ---------------------------------------------------------------------------

SEARCH_BAND_PX  = 40      # px above/below y_spindle to search
X_START         = 100     # left crop boundary (exclude cell edges)
X_END           = 540     # right crop boundary
MM_PER_PIXEL    = None    # set after calibration; None = report px only
SMOOTH_ALPHA    = 0.6     # EMA smoothing on seam row (0=no smooth, 1=frozen)
PROFILE_WIDTH   = 200     # width of the profile panel in the display


def detect_seam(gray_roi: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Run vertical Sobel on gray ROI, collapse to 1D profile, return peak row.

    Args:
        gray_roi: grayscale crop of shape (2*SEARCH_BAND, width)
    Returns:
        (peak_row_in_roi, profile_1d)
    """
    sobel = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.abs(sobel)

    sobel_raw = sobel.copy()
    # Optional: suppress weak responses (noise floor)
    sobel = np.where(sobel > sobel.max() * 0.1, sobel, 0.0)


    profile = sobel.mean(axis=1)   # shape: (2*SEARCH_BAND,)
    peak = int(np.argmax(profile))
    return peak, profile, sobel_raw


def draw_profile_panel(profile: np.ndarray, peak_row: int,
                       panel_h: int, panel_w: int = PROFILE_WIDTH) -> np.ndarray:
    """Render 1D profile as a horizontal bar chart panel."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    if profile.max() < 1e-6:
        return panel

    norm = profile / profile.max()
    bar_color = (0, 200, 255)

    for row_i, val in enumerate(norm):
        bar_len = int(val * (panel_w - 10))
        y = int(row_i * panel_h / len(norm))
        y_next = int((row_i + 1) * panel_h / len(norm))
        cv2.rectangle(panel, (0, y), (bar_len, y_next - 1), bar_color, -1)

    # Mark peak
    peak_y = int(peak_row * panel_h / len(profile))
    cv2.line(panel, (0, peak_y), (panel_w, peak_y), (0, 255, 0), 2)
    cv2.putText(panel, "peak", (4, max(peak_y - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return panel


def main():
    config = realsense_init(width=640, height=480, fps=30,
                            enable_spatial=True, enable_temporal=True)
    print("RealSense initialized.")

    # Grab first frame
    while True:
        color_frame, _ = realsense_get_frame(config)
        if color_frame is not None:
            break
    first = np.asanyarray(color_frame.get_data())
    fh, fw = first.shape[:2]

    # Initial spindle position = image center row
    y_spindle = fh // 2
    seam_row_smooth = float(y_spindle)
    save_idx = 0

    print(f"Initial spindle y={y_spindle}. Press 't' to teach after positioning.")

    try:
        while True:
            color_frame, _ = realsense_get_frame(config)
            if color_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # Search band clipped to frame bounds
            #Define ROI
            y0 = max(0, y_spindle - SEARCH_BAND_PX)
            y1 = min(fh, y_spindle + SEARCH_BAND_PX)
            x0 = max(0, X_START)
            x1 = min(fw, X_END)

            roi_color = frame[y0:y1, x0:x1]
            roi_gray  = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

            peak_local, profile, sobel_raw = detect_seam(roi_gray)

            #Visualization of sobel
            if sobel_raw.max() > 0:
                sobel_vis = (
                    255 * sobel_raw / sobel_raw.max()
                ).astype(np.uint8)
            else:
                sobel_vis = np.zeros_like(sobel_raw, dtype=np.uint8)

            sobel_vis = cv2.cvtColor(
                sobel_vis,
                cv2.COLOR_GRAY2BGR
            )

            cv2.line(
                sobel_vis,
                (0, peak_local),
                (sobel_vis.shape[1], peak_local),
                (0, 255, 0),
                2
            )

            


            seam_row_global = y0 + peak_local

            # EMA temporal smoothing
            seam_row_smooth = (SMOOTH_ALPHA * seam_row_smooth
                               + (1 - SMOOTH_ALPHA) * seam_row_global)
            seam_row_int = int(round(seam_row_smooth))

            delta_px = seam_row_int - y_spindle
            delta_mm = (delta_px * MM_PER_PIXEL) if MM_PER_PIXEL else None

            # ----------------------------------------------------------------
            # Visualisation
            # ----------------------------------------------------------------
            vis = frame.copy()

            # Search band
            cv2.rectangle(vis, (x0, y0), (x1, y1), (50, 50, 200), 1)

            # Spindle centerline (taught reference)
            cv2.line(vis, (x0, y_spindle), (x1, y_spindle), (0, 255, 255), 1)
            cv2.putText(vis, "spindle", (x1 + 4, y_spindle + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Detected seam
            cv2.line(vis, (x0, seam_row_int), (x1, seam_row_int), (0, 255, 0), 2)
            cv2.putText(vis, "seam", (x1 + 4, seam_row_int + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Offset arrow
            if abs(delta_px) > 2:
                cv2.arrowedLine(vis,
                                ((x0 + x1) // 2, y_spindle),
                                ((x0 + x1) // 2, seam_row_int),
                                (0, 100, 255), 2, tipLength=0.3)

            # HUD
            offset_str = (f"delta: {delta_px:+d}px  {delta_mm:+.2f}mm"
                          if delta_mm is not None else f"delta: {delta_px:+d}px")
            cv2.putText(vis, offset_str, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"spindle y={y_spindle}  seam y={seam_row_int}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(vis, "[t]teach [r]reset [s]save [q]quit",
                        (10, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1)

            # Profile panel
            roi_h = y1 - y0
            profile_panel = draw_profile_panel(profile, peak_local, roi_h)

            # Composite: embed profile panel next to ROI region
            panel_x = x1 + 2
            if panel_x + PROFILE_WIDTH <= fw:
                vis[y0:y1, panel_x:panel_x + PROFILE_WIDTH] = profile_panel
            else:
                # fallback: overlay on right edge
                pw = fw - panel_x
                if pw > 10:
                    vis[y0:y1, panel_x:fw] = profile_panel[:, :pw]

            cv2.imshow("Seam Detection", vis)
            cv2.imshow("Sobel", sobel_vis)

            print(f"\r{offset_str}", end="", flush=True)

            # ----------------------------------------------------------------
            # Keys
            # ----------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('t'):
                y_spindle = seam_row_int
                seam_row_smooth = float(y_spindle)
                print(f"\nTaught spindle y={y_spindle}")
            elif key == ord('r'):
                y_spindle = fh // 2
                seam_row_smooth = float(y_spindle)
                print(f"\nReset spindle y={y_spindle}")
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