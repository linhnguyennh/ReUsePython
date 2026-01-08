import os
import cv2
from datetime import datetime
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.vision.realsense_frame import realsense_init,realsense_get_frame

class FrameSaver:
    def __init__(self, root_dir="data/yolo_retrain_obb_260108", prefix="morrow", ext=".png"):
        """
        Args:
            root_dir (str): Base directory for saving images
            prefix (str): Filename prefix
            ext (str): Image extension (.png recommended for lossless storage)
        """

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join(root_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.prefix = prefix
        self.ext = ext
        self.frame_idx = 0

        print(f"[INFO] Saving images to: {self.session_dir}")

    def save(self, color_image):
        """
        Save a BGR color frame to disk.

        Args:
            color_frame (np.ndarray): (H, W, 3) uint8 BGR image
        """

        filename = f"{self.prefix}_{self.frame_idx:06d}{self.ext}"
        filepath = os.path.join(self.session_dir, filename)

        cv2.imwrite(filepath, color_image)
        self.frame_idx += 1

def main():
    pipeline, _, _ = realsense_init(width=640,height=480)
    saver = FrameSaver()

    print("[INFO] Press SPACE to save frame | Press Q to quit")

    try:
        while True:
            color_frame, _ = realsense_get_frame(pipeline)
            if color_frame is None:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("RealSense Color", color_image)

            key = cv2.waitKey(1) & 0xFF

            # Save frame on SPACE
            if key == 32:  # SPACE
                saver.save(color_image)
                print(f"[SAVED] Frame {saver.frame_idx - 1}")

            # Quit on Q
            elif key == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()