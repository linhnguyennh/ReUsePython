import numpy as np
import sys
import logging
from pathlib import Path
import time

from src.vision.realsense_stream import RealSenseStream
from config.common import CAM_WIDTH, CAM_HEIGHT, CAM_FPS
from config.gp7_cut_config import ROI_CONFIG
from src.vision.seam_detect import detect_seam, build_sobel_vis, draw_profile_panel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def select_roi(frame, x0, y0, x1, y1):
    return frame[y0:y1,x0:x1]

def process_seam(frame):
    pass

def visualise_seam():
    pass

def main():
    rs_stream = RealSenseStream(
        width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS,
        enable_decimation=True,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=True
    )
    frame_queue = rs_stream.frame_queue
    rs_stream.start()

    # PUT THREAD INIT HERE #
    
    #Display thread
    #PLC OPCUA thread
    try:
        while True:
            # PUT TASK CODE HERE #
            #PIPELINE: Select ROI (Base on SHORT OR LONG) --> Detect EDGE + Smoothing --> Filter by depth --> Generate Endpoints --> TODO: GENERATE MOVEMENT VECTOR --> SEND TO ROBOT 

            #NEED: Vector from first point to marker tip --> MOVE
            
            #NEED: Vector from current position (marker tip) to end point --> MOVE
            
            #REPEAT AFTER ROTATION FROM JOB
            
            #AFTER 4 ROTATION END
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt triggered...")
        rs_stream.stop()
        logger.info("All threads terminated")

if __name__ == '__main__':
    main()