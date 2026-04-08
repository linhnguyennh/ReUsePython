import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
import numpy as np
import sys
import os
import logging
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker, DisplayWorker
from src.communication.opcua_device import PLCClient, Yaskawa_YRC1000
from src.vision.visual_controller import calc_control_val

def main():
    # --- Control parameters ---
    Kp = 0.2
    Kd = 0.4
    ALPHA = 0.32       # smoothing factor (0–1)
    DEADBAND_M = 0.02  # 10 mm deadband
    LOOP_HZ = 20
    LOOP_DT = 1.0 / LOOP_HZ

    # --- Connection URLs ---
    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"
    postcp_index = 0
    # --- Start camera and model ---
    camera = RealSenseStream(fps=30, width=640, height=480)
    camera.start()

    model = YOLO(r"models\focus1\retrain_obb_BIGMAP_260108\train\weights\morrow_obb_260108.pt")
    #model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.pt")
    #model = YOLO(r"models\focus1\retrain\train3\weights\best_morrow_251020.pt")

    
    #model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.onnx")
    detection_worker = DetectionWorker(
        model=model,
        camera=camera,
        max_queue_size=1,
        obb=True,
        conf=0.85,
        imgsz=(480,640),
        task='obb'
    )
    display_worker = DisplayWorker(
        camera=camera,
        detections_queue=detection_worker.detections_queue,
        obb=True,
        limit_box=True
    )

    detection_worker.start()
    display_worker.start()

    try:
        logging.info("[Main] Visual servo control loop started.")
        while display_worker.running:
            try:
                detections = detection_worker.detections_queue.get(timeout=0.2)
            except Empty:
                continue
        
            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("[Main] Keyboard interrupt detected. Stopping...")

    finally:
        logging.info("[Main] Cleaning up...")
        detection_worker.stop()
        camera.stop()
        logging.info("[Main] All threads stopped cleanly.")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
