import cv2
import numpy as np
import sys
import os
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultralytics import YOLO
from src.vision.realsense_frame import realsense_init, realsense_get_frame


# -------------------------
# Temporal buffer
# -------------------------
mask_buffer = deque(maxlen=3)


def smooth_mask(mask_buffer, new_mask, alpha=0.7):
    if len(mask_buffer) == 0:
        return new_mask

    prev = mask_buffer[-1]
    return alpha * new_mask + (1 - alpha) * prev


def main():
    model = YOLO(r"models\focus1\260421\train\weights\best.pt")
    segmentation = True

    print("Initializing RealSense camera...")
    config = realsense_init(
        width=640,
        height=480,
        fps=30,
        enable_decimation=False,
        enable_spatial=False,
        enable_temporal=False
    )

    print("RealSense camera initialized successfully!")

    try:
        while True:
            color_frame, depth_frame = realsense_get_frame(config)

            if color_frame is None:
                continue

            frame = np.asanyarray(color_frame.get_data())

            results = model(frame, conf=0.6)
            result = results[0]

            annotated = result.plot()
            img = frame.copy()

            if segmentation and result.masks is not None:

                masks = result.masks.data.cpu().numpy()

                for mask in masks:

                    # -------------------------
                    # 1. convert to float mask
                    # -------------------------
                    mask = (mask > 0.5).astype(np.float32)

                    # -------------------------
                    # 2. morphological cleanup (SPATIAL FIRST)
                    # -------------------------
                    kernel = np.ones((5, 5), np.uint8)
                    mask_uint8 = mask.astype(np.uint8)

                    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

                    mask = mask_uint8.astype(np.float32)

                    # -------------------------
                    # 3. temporal smoothing
                    # -------------------------
                    mask_buffer.append(mask)
                    smoothed_mask = smooth_mask(mask_buffer, mask, alpha=0.7)

                    binary_mask = (smoothed_mask > 0.5).astype(np.uint8)

                    # -------------------------
                    # 4. visualization
                    # -------------------------
                    colored = np.zeros_like(img)
                    colored[:, :, 1] = binary_mask * 255

                    img = cv2.addWeighted(img, 1.0, colored, 0.5, 0)

            cv2.imshow("mask_overlay", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")


if __name__ == "__main__":
    main()