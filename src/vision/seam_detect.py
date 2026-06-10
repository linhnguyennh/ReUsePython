import cv2
import numpy as np
import pyrealsense2 as rs

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
                       panel_h: int, panel_w: int, cut_row : int) -> np.ndarray:
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
    if 0 <= cut_row < len(profile):
        cut_y = int(cut_row * panel_h / len(profile))
        cv2.line(panel, (0, cut_y), (panel_w, cut_y), (0, 100, 255), 1)
        cv2.putText(panel, "cut", (4, max(cut_y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    return panel


def build_sobel_vis(sobel_raw: np.ndarray, peak_local: int, cut_row : int) -> np.ndarray:
    if sobel_raw.max() > 0:
        vis = (255 * sobel_raw / sobel_raw.max()).astype(np.uint8)
    else:
        vis = np.zeros_like(sobel_raw, dtype=np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (0, peak_local), (vis.shape[1], peak_local), (0, 255, 0), 2)
    if 0 <= cut_row < vis.shape[0]:
        cv2.line(vis, (0, cut_row), (vis.shape[1], cut_row), (0, 100, 255), 2)
    return vis