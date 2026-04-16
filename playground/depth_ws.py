import cv2
import logging
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.realsense_frame import realsense_init, realsense_get_frame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_depth_camera():
    """
    Test script for Intel RealSense D435i depth camera.
    
    Captures frames from the camera and displays color and depth streams with filtering.
    Press 'q' to exit.
    """
    logger.info("Initializing RealSense D435i camera with depth filters...")
    config = realsense_init(width=640, height=480, fps=30)
    logger.info("Camera initialized successfully with filters enabled")
    
    try:
        frame_count = 0
        while True:
            # Get frames (filters are applied automatically in realsense_get_frame)
            frames = realsense_get_frame(config)
            
            if frames[0] is None or frames[1] is None:
                logger.warning("Failed to get frames")
                continue
            
            color_frame, depth_frame = frames
            
            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert depth to 8-bit for display (normalize for visualization)
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Display both frames
            cv2.imshow('Color Stream', color_image)
            cv2.imshow('Depth Stream (Filtered)', depth_colormap)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exiting...")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        logger.info("Cleaning up...")
        config.pipeline.stop()
        cv2.destroyAllWindows()
        logger.info("Camera stopped and windows closed")

if __name__ == "__main__":
    test_depth_camera()