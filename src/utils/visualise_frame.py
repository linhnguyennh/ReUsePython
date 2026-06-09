import numpy as np
import cv2

def draw_depth(depth_frame, width, height):
    depth_arr = np.asarray(depth_frame).astype(np.float32)
    depth_vis = np.clip(depth_arr, 0, 2000)
    depth_vis = (depth_vis / 2000.0 * 255).astype(np.uint8)
    depth_panel = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    if depth_panel.shape[:2] != (height, width):
        depth_panel = cv2.resize(depth_panel, (width, height))

    return depth_panel
