import cv2
import numpy as np
import sys
import os
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultralytics import YOLO
from src.vision.realsense_frame import realsense_init, realsense_get_frame

def main():
    # Load your trained model
    model = YOLO(r"models\focus1\260421\train\weights\best.pt")

    # Initialize RealSense camera
    print("Initializing RealSense camera...")
    config = realsense_init(width=1280, height=720, fps=30,
                           enable_decimation=False, enable_spatial=False, enable_temporal=False)

    print("RealSense camera initialized successfully!")

    try:
        while True:
            # Get RealSense frames
            color_frame, depth_frame = realsense_get_frame(config)

            if color_frame is None or depth_frame is None:
                print("Failed to get frames, skipping...")
                continue

            # Convert color frame to numpy array for YOLO
            frame = np.asanyarray(color_frame.get_data())

            # Run inference
            results = model(frame, conf=0.4)
            #results[0].show()
            # Visualize results (built-in)
            annotated = results[0].plot()

            # Show
            cv2.imshow("Battery Segmentation", annotated)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Stop the RealSense pipeline
        config.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")

if __name__ == "__main__":
    main()