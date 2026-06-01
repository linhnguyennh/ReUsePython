"""
Depth Anything 3 - RealSense Center Point Depth Test
=====================================================
Streams from RealSense, runs DA3 monocular inference each frame,
and displays RGB + INFERNO depth map with the center pixel depth in meters.

Model choices (set MODEL_ID below):
  "depth-anything/DA3METRIC-LARGE"          → metric depth, Apache 2.0, ~0.35B params
  "depth-anything/DA3NESTED-GIANT-LARGE-1.1"→ metric depth (nested), best quality, CC BY-NC 4.0
  "depth-anything/DA3MONO-LARGE"            → relative depth only, Apache 2.0

For DA3METRIC-LARGE the formula is:  depth_m = focal_px * raw_output / 300
For DA3NESTED the output is already in meters.
"""

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from depth_anything_3.api import DepthAnything3

# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_ID = "depth-anything/DA3METRIC-LARGE"
#MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"   # best quality, heavier // very heavy
# MODEL_ID = "depth-anything/DA3MONO-LARGE"               # relative only

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RS_W, RS_H, RS_FPS = 640, 480, 30

# ─── Load model ───────────────────────────────────────────────────────────────

print(f"[·] Loading {MODEL_ID} on {DEVICE} …")
model = DepthAnything3.from_pretrained(MODEL_ID).to(device=DEVICE).eval()
print("[✓] Model ready")

IS_METRIC   = "METRIC" in MODEL_ID or "NESTED" in MODEL_ID
IS_NESTED   = "NESTED" in MODEL_ID   # already in meters, no focal scaling needed

# ─── RealSense setup ──────────────────────────────────────────────────────────

pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16,  RS_FPS)
profile  = pipeline.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()         # meters per raw unit

# Read intrinsics for focal-length metric scaling (DA3METRIC)
color_profile  = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics     = color_profile.get_intrinsics()
fx, fy         = intrinsics.fx, intrinsics.fy
focal_px       = ((fx + fy) / 2.0) / 2.0

align = rs.align(rs.stream.color)
print(f"[✓] RealSense started  (depth_scale={depth_scale:.5f} m/unit, focal={focal_px:.1f} px)")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def colorize(depth: np.ndarray) -> np.ndarray:
    """Map float depth → INFERNO BGR image."""
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def draw_crosshair(img, cx, cy, color, size=18, thickness=2):
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, color, -1, cv2.LINE_AA)


def put_label(img, text, x, y, color, scale=0.65, thickness=2):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def colorbar_strip(h: int, d_min: float, d_max: float, width: int = 28) -> np.ndarray:
    bar   = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    strip = cv2.applyColorMap(np.tile(bar, (1, width)), cv2.COLORMAP_INFERNO)
    unit  = "m" if IS_METRIC else ""
    put_label(strip, f"{d_max:.2f}{unit}", 2, 14,  (255, 255, 255), scale=0.35, thickness=1)
    put_label(strip, f"{d_min:.2f}{unit}", 2, h-4, (255, 255, 255), scale=0.35, thickness=1)
    return strip

# ─── Main loop ────────────────────────────────────────────────────────────────

print("Press  Q  to quit.")

while True:
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)
    c_frame = aligned.get_color_frame()
    d_frame = aligned.get_depth_frame()
    if not c_frame or not d_frame:
        continue

    rgb_bgr  = np.asanyarray(c_frame.get_data())          # (H,W,3) BGR  uint8
    rs_depth = np.asanyarray(d_frame.get_data())          # (H,W)   uint16

    h, w = rgb_bgr.shape[:2]
    cx, cy = w // 2, h // 2

    # --- DA3 inference (expects RGB image or path) ----------------------------
    rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)    # (H,W,3) RGB uint8

    with torch.no_grad():
        prediction = model.inference([rgb_rgb])            # list of np arrays OK

    raw_depth = prediction.depth[0]                        # (H,W) float32

    # --- Convert to metric meters --------------------------------------------
    if IS_NESTED:
        depth_m = raw_depth                                # already meters
    elif IS_METRIC:
        # DA3METRIC: depth_m = focal * net_output / 300
        depth_m = focal_px * raw_depth / 300.0
    else:
        depth_m = raw_depth                                # relative, not meters

    dm_h, dm_w = depth_m.shape
    dm_cx = int(cx * dm_w / w)
    dm_cy = int(cy * dm_h / h)
    center_depth = float(depth_m[dm_cy, dm_cx])

    # RealSense ground-truth at center for comparison
    rs_center_m  = float(rs_depth[cy, cx]) * depth_scale

    # --- Build panels ---------------------------------------------------------
    rgb_ann   = rgb_bgr.copy()
    depth_vis = colorize(depth_m)

    GREEN, WHITE = (0, 220, 0), (255, 255, 255)

    # DA3 resizes internally — scale crosshair coords to depth_vis resolution
    dh_vis, dw_vis = depth_vis.shape[:2]
    dcx = int(cx * dw_vis / w)
    dcy = int(cy * dh_vis / h)

    # Crosshairs
    draw_crosshair(rgb_ann,   cx,  cy,  GREEN)
    draw_crosshair(depth_vis, dcx, dcy, WHITE)

    # Depth label
    if IS_METRIC:
        label = f"{center_depth:.3f} m  (RS: {rs_center_m:.3f} m)"
    else:
        label = f"rel={center_depth:.4f}  (RS: {rs_center_m:.3f} m)"

    put_label(rgb_ann,   label, cx  + 14, cy  - 10, GREEN)
    put_label(depth_vis, label, dcx + 14, dcy - 10, WHITE)

    # Colorbar — use depth_vis actual height (DA3 may resize internally)
    dh, dw = depth_vis.shape[:2]
    d_min_v, d_max_v = depth_m.min(), depth_m.max()
    cbar = colorbar_strip(dh, d_min_v, d_max_v)
    depth_panel = np.hstack([depth_vis, cbar])
    depth_panel = cv2.resize(depth_panel, (w, h))

    # Title banner
    banner = np.zeros((30, w * 2, 3), dtype=np.uint8)
    cv2.putText(banner,
                f"DA3 [{MODEL_ID.split('/')[-1]}]  |  {DEVICE}  |  center: {label}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    display = np.vstack([banner, np.hstack([rgb_ann, depth_panel])])
    cv2.imshow("Depth Anything 3 — Live", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ─── Cleanup ──────────────────────────────────────────────────────────────────
pipeline.stop()
cv2.destroyAllWindows()
print("Done.")