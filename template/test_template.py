import numpy as np
import logging
import time
from dataclasses import dataclass, field
import cv2
import pyrealsense2 as rs
from queue import Empty
#CUSTOM

from src.vision.realsense_stream import RealSenseStream
from config.common import CAM_WIDTH, CAM_HEIGHT, CAM_FPS


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

    try:
        while True:
            try:
                rgb_frame, depth_frame = frame_queue.get()
            except Empty:
                continue
            
            rgb_data = np.asanyarray(rgb_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())
            cv2.imshow("D435i RGB", rgb_data)
            # PUT TASK CODE HERE #
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt triggered...")
        rs_stream.stop()
        cv2.destroyAllWindows()
        logger.info("All threads terminated")

if __name__ == '__main__':
    main()