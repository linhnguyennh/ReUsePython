import cv2
import numpy as np
import os
from datetime import datetime
import sys

# adjust import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame


def create_save_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def save_frame(rgb, save_dir, idx):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(save_dir, f"{idx}_{timestamp}.png")

    cv2.imwrite(path, rgb)
    print(f"[SAVED] {path}")


def main():
    config = realsense_init(
        width=640,
        height=480,
        fps=30,
        enable_decimation=False,
        enable_spatial=False,
        enable_temporal=False
    )

    save_dir = create_save_dir("data/morrow/dataset/morrow_overall_260421")

    # prevent index reset issue
    idx = len(os.listdir(save_dir))

    try:
        while True:
            color_frame, _ = realsense_get_frame(config)

            if color_frame is None:
                continue

            rgb = np.asanyarray(color_frame.get_data())

            cv2.imshow("RGB", rgb)

            key = cv2.waitKey(1) & 0xFF

            # SPACE BAR to save
            if key == 32:
                save_frame(rgb, save_dir, idx)
                idx += 1

            # QUIT
            elif key == ord('q'):
                break

    finally:
        config.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()