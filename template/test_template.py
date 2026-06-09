import numpy as np
import sys
import logging
from pathlib import Path
import time

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
            # PUT TASK CODE HERE #
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt triggered...")
        rs_stream.stop()
        logger.info("All threads terminated")

if __name__ == '__main__':
    main()