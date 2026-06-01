"""
DexiNed live edge detection via kornia + OpenCV webcam.
 
Requirements:
    pip install kornia torch torchvision opencv-python
 
Usage:
    python dexined_live.py                  # default webcam (index 0)
    python dexined_live.py --source 1       # webcam index 1
    python dexined_live.py --source video.mp4
    python dexined_live.py --size 352       # inference resolution (default 352)
    python dexined_live.py --overlay        # blend edges over original frame
 
Controls:
    q / ESC  - quit
    s        - save current edge frame as PNG
    o        - toggle overlay mode
    +/-      - increase/decrease edge threshold
"""
 
import argparse
import time
from pathlib import Path
 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
 
# kornia lazy import with friendly error
try:
    import kornia
    from kornia.contrib.edge_detection import EdgeDetectorBuilder
except ImportError:
    raise SystemExit("kornia not found. Run: pip install kornia")
 
 
def parse_args():
    p = argparse.ArgumentParser(description="DexiNed live edge detector")
    p.add_argument("--source", default="0",
                   help="Camera index (int) or video file path")
    p.add_argument("--size", type=int, default=352,
                   help="Inference resolution (square). Default: 352")
    p.add_argument("--overlay", action="store_true",
                   help="Start in overlay mode")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Edge binarisation threshold (0-1). Default: 0.3")
    return p.parse_args()
 
 
def open_source(source_str):
    try:
        src = int(source_str)
    except ValueError:
        src = source_str
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {source_str}")
    return cap
 
 
def preprocess(frame_bgr: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """BGR uint8 HxWx3 → float32 tensor 1x3xSxS on device."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().div(255.0)   # HxWx3
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)    # 1x3xHxW
    return t
 
 
def postprocess(edge_tensor: torch.Tensor, out_hw: tuple) -> np.ndarray:
    """1x1xHxW float32 tensor → uint8 HxW numpy edge map."""
    edge = edge_tensor.squeeze().cpu().numpy()         # HxW float [0,1]
    edge = cv2.resize(edge, (out_hw[1], out_hw[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (edge * 255).clip(0, 255).astype(np.uint8)
 
 
def overlay_edges(frame_bgr: np.ndarray, edges_gray: np.ndarray,
                  threshold: float) -> np.ndarray:
    """Overlay thresholded edges (green) on the original frame."""
    _, mask = cv2.threshold(edges_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
    out = frame_bgr.copy()
    out[mask > 0] = (0, 220, 60)   # green edges
    return out
 
 
def build_display(frame_bgr, edges_gray, overlay_mode, threshold, fps):
    h, w = frame_bgr.shape[:2]
    if overlay_mode:
        vis = overlay_edges(frame_bgr, edges_gray, threshold)
    else:
        vis = cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR)
 
    # HUD
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)
    mode_str = "overlay" if overlay_mode else "edges"
    cv2.putText(vis, f"mode: {mode_str}  thr: {threshold:.2f}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(vis, "[o]toggle  [+/-]thr  [s]save  [q]quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return vis
 
 
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    print("Loading DexiNed (kornia)...")
    model = EdgeDetectorBuilder.build(
        model_name="dexined",
        pretrained=True,
        image_size=args.size,
    )
    model = model.to(device)
    model.eval()
    print("Model ready.")
 
    cap = open_source(args.source)
    overlay_mode = args.overlay
    threshold = args.threshold
    save_idx = 0
 
    prev_time = time.time()
    fps = 0.0
 
    print("Starting live inference. Press 'q' or ESC to quit.")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or read error.")
            break
 
        h, w = frame.shape[:2]
 
        # --- Inference ---
        with torch.no_grad():
            inp = preprocess(frame, args.size, device)
            # kornia EdgeDetector returns a tensor directly
            out = model(inp)                        # 1x1xSxS
            # kornia may return list or tensor depending on version
            if isinstance(out, (list, tuple)):
                out = out[-1]
 
        edges = postprocess(out, (h, w))
 
        # --- FPS ---
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
 
        # --- Display ---
        vis = build_display(frame, edges, overlay_mode, threshold, fps)
        cv2.imshow("DexiNed Live", vis)
 
        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):           # q or ESC
            break
        elif key == ord('o'):
            overlay_mode = not overlay_mode
        elif key == ord('s'):
            fname = f"edge_{save_idx:04d}.png"
            cv2.imwrite(fname, edges)
            print(f"Saved {fname}")
            save_idx += 1
        elif key == ord('+') or key == ord('='):
            threshold = min(threshold + 0.05, 1.0)
        elif key == ord('-'):
            threshold = max(threshold - 0.05, 0.0)
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 