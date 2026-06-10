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
ROI_Y_OFFSET = 100
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