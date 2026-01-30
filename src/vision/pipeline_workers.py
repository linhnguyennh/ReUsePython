import threading
from queue import Queue, Empty, Full
import logging
import time
import numpy as np
import cv2
from asyncua.sync import Client, ThreadLoop

from src.vision.detection_fn import detection_xyz, detection_xyz_obb, draw_detection, draw_detection_obb, colorize_depth
from src.vision.realsense_stream import RealSenseStream
from ..utils.queue_helper import put_latest

# -------------------------
# Detection Worker
# -------------------------

class DetectionWorker(threading.Thread):
    def __init__(self, model, camera: RealSenseStream, max_queue_size=1, obb = False, **yolo_args):
        super().__init__(daemon=True)
        self.model = model
        self._camera = camera
        
        self._detections_queue = Queue(maxsize=max_queue_size)
        self._obb = obb

        self.running = False
        self.yolo_args = yolo_args

        self.det_logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        self.running = True
        
        intrinsics = self._camera.depth_intrinsics
        width, height = self._camera.width, self._camera.height

        self.det_logger.info("Detection Thread started")
       
        while self.running:
            try:
                frame = self._camera.get_latest_frame()
                if frame is None:
                    continue
                color_frame, depth_frame = frame

            except Empty:
                continue

            if not self._obb:
                detections = detection_xyz(
                    self.model,
                    color_frame,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )
            
            if self._obb:
                detections = detection_xyz_obb(
                    self.model,
                    color_frame,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )

            put_latest(self._detections_queue, detections)

        self.det_logger.info("Detection stop")

    def stop(self):
        self.running = False
        self.join()

    @property
    def detections_queue(self):
        return self._detections_queue


class DisplayWorker(threading.Thread):
    def __init__(self, camera : RealSenseStream, detections_queue : Queue, obb = False, limit_box=True, depth=False):
        super().__init__(daemon=True)
        self.running = False
        
        self._camera = camera
        
        self._detections_queue = detections_queue

        self._obb = obb
        self._depth = depth
        self._limit_box = limit_box

        self.display_logger = logging.getLogger(self.__class__.__name__)

        
    def run(self):
        self.running = True
        self.display_logger.info("Display Thread start")
        width, height = self._camera.width, self._camera.height
        while self.running:
            try:
                frame = self._camera.get_latest_frame()
                if frame is None:
                    continue
                color_frame, depth_frame = frame
                color_image = np.asanyarray(color_frame.get_data())

            except Empty:
                continue

            detections = self._detections_queue.get()
                
            if not self._obb:
                color_annotated = draw_detection(color_image, detections, self._limit_box, camera_width=width, camera_height=height)
                

            if self._obb:
                color_annotated = draw_detection_obb(color_image, detections, self._limit_box, camera_width=width, camera_height=height)
            
            if self._depth:
                depth_colored = colorize_depth(depth_frame=depth_frame, depth_scale=self._camera.depth_scale)
                cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("YOLO Detections with XYZ coordinate", color_annotated)
            
            
            if cv2.waitKey(1) == 27:  # ESC
                self.running = False
                break
            
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.join()
        self.display_logger.info("Thread stop")