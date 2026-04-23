import threading
from queue import Queue, Empty, Full
import logging
import time
import numpy as np
import cv2
from asyncua.sync import Client, ThreadLoop

from src.vision.detection_fn import detection_xyz, detection_xyz_obb, draw_detection, draw_detection_obb, colorize_depth
from src.vision.realsense_stream import RealSenseStream
from src.vision.segmentation_fn import segment_object
from ..utils.queue_helper import put_latest

# -------------------------
# Detection Worker
# -------------------------

class DetectionWorker(threading.Thread):
    def __init__(self, model, width, height, depth_intrinsics, frame_queue : Queue, max_queue_size=1, obb = False, **yolo_args):
        super().__init__(daemon=True)
        self.model = model
        
        self._results_queue = Queue(maxsize=max_queue_size)
        self._detections_queue = Queue(maxsize=max_queue_size)
        self._obb = obb

        self.running = False
        
        self._width = width
        self._height = height
        self._depth_intrinsics = depth_intrinsics

        self._frame_queue = frame_queue

        self.yolo_args = yolo_args
        

        self.det_logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        self.running = True
        
        intrinsics = self._depth_intrinsics
        width, height = self._width, self._height

        self.det_logger.info("Detection Thread started")
       
        while self.running:
            try:
                frame = self._frame_queue.get_nowait()
                if frame is None:
                    continue
                color_frame, depth_frame = frame
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

            except Empty:
                continue

            if not self._obb:
                detections = detection_xyz(
                    self.model,
                    color_data,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )
            
            if self._obb:
                detections = detection_xyz_obb(
                    self.model,
                    color_data,
                    depth_frame,
                    intrinsics=intrinsics,
                    img_width=width,
                    img_height=height,
                    **self.yolo_args
                )

            put_latest(self._results_queue, (color_data, depth_data, detections))
            put_latest(self._detections_queue, detections)

        self.det_logger.info("Detection stop")

    def stop(self):
        self.running = False
        self.join()

    @property
    def results_queue(self):
        return self._results_queue

    @property
    def detections_queue(self):
        return self._detections_queue

class SegmentationWorker(threading.Thread):
    def __init__(self, model, frame_queue : Queue, max_queue_size=1, **yolo_args):
        super().__init__(daemon=True)
        self.running = False

        self._model = model
        self._frame_queue = frame_queue
        self._mask_queue = Queue(maxsize=max_queue_size) 
        self._yolo_args = yolo_args

        self.seg_logger = logging.getLogger(self.__class__.__name__)
    def run(self):
        self.running = True
        self.seg_logger.info("Segmentation Thread started")
        while self.running:
            try:
                frame = self._frame_queue.get_nowait()
                if frame is None:
                    continue

                rgb_frame, depth_frame = frame
            except Empty:
                continue
           
            #Run inference
            rgb_data = np.asanyarray(rgb_frame.get_data())
            mask, _ = segment_object(self._model, rgb_data, **self._yolo_args)
            put_latest(self._mask_queue, (rgb_frame, depth_frame, mask))

        self.seg_logger.info("Segmentation Worker stop") 
    def stop(self):
        self.running = False

    @property
    def mask_queue(self):
        return self._mask_queue
class DisplayWorker(threading.Thread):
    def __init__(self, width: int, height: int, depth_scale: float, results_queue : Queue, obb = False, limit_box=True, depth=False):
        super().__init__(daemon=True)
        self.running = False
        
        self._width = width
        self._height = height
        self._depth_scale = depth_scale
        
        self._results_queue = results_queue

        self._obb = obb
        self._depth = depth
        self._limit_box = limit_box

        self.display_logger = logging.getLogger(self.__class__.__name__)

        
    def run(self):
        self.running = True
        self.display_logger.info("Display Thread start")
        while self.running:
            try:
                color_data, depth_data, detections = self._results_queue.get()
                color_image = color_data

            except Empty:
                continue
                
            if not self._obb:
                color_annotated = draw_detection(color_image, detections, self._limit_box, camera_width=self._width, camera_height=self._height)
                

            if self._obb:
                color_annotated = draw_detection_obb(color_image, detections, self._limit_box, camera_width=self._width, camera_height=self._height)
            
            if self._depth:
                depth_colored = colorize_depth(depth_data, depth_scale=self._depth_scale)
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