import threading
import logging
from queue import Queue, Full, Empty
from .realsense_frame import realsense_get_frame, realsense_init
from ..utils.queue_helper import put_latest

class RealSenseStream:
    """
    A threaded RealSense camera stream class for capturing and queuing frames.

    This class initializes a RealSense pipeline, starts a background thread for frame capture,
    and provides methods to retrieve the latest frames from a queue.

    Attributes:
        pipeline (rs.pipeline): The started RealSense pipeline.
        depth_scale (float): The depth scale factor.
        depth_intrinsics (rs.intrinsics): The depth stream intrinsics.
        color_intrinsics (rs.intrinsics): The color stream intrinsics.
        width (int): The frame width.
        height (int): The frame height.
        running (bool): Flag indicating if the capture thread is running.
    """

    def __init__(self, width=640, height=480, fps=15, enable_imu=False, max_queue_size=1,
                 enable_decimation=False, enable_spatial=False, enable_temporal=False):
        """
        Initialize the RealSense stream.

        Args:
            width (int): Frame width. Defaults to 640.
            height (int): Frame height. Defaults to 480.
            fps (int): Frame rate. Defaults to 15.
            enable_imu (bool): Whether to enable IMU streams. Defaults to False.
            max_queue_size (int): Maximum size of the frame queue. Defaults to 1.
            enable_decimation (bool): Whether to enable decimation filter. Defaults to False.
            enable_spatial (bool): Whether to enable spatial filter. Defaults to False.
            enable_temporal (bool): Whether to enable temporal filter. Defaults to False.
        """
        self._config = realsense_init(width, height, fps, enable_imu, 
                                      enable_decimation, enable_spatial, enable_temporal)
        self._frame_queue = Queue(maxsize=max_queue_size)
        self._width = width
        self._height = height
        self.running = False
        self.cam_logger = logging.getLogger(self.__class__.__name__)

    def _capture_loop(self):
        """
        Internal capture loop running in a separate thread to continuously capture frames.
        """
        self.cam_logger.info("Capture loop started")
        while self.running:
            color_frame, depth_frame = realsense_get_frame(self._config)

            if color_frame is not None and depth_frame is not None:
                try:
                    put_latest(self.frame_queue, (color_frame, depth_frame))
                except Exception as e:
                    logging.info(f'Exception from RealSenseStream capture loop: {e}')
                    continue

    def start(self):
        """
        Start the capture thread.
        """
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        self.cam_logger.info("Thread started")

    def stop(self):
        """
        Stop the capture thread and the pipeline.
        """
        self.running = False
        self.thread.join()
        self._config.pipeline.stop()
        self.cam_logger.info("Thread stopped")

    def get_latest_frame(self):
        """
        Retrieve the latest frame from the queue.

        Returns:
            tuple or None: The latest (color_frame, depth_frame) tuple, or None if no frames are available.
        """
        latest = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except Empty:
                break
        return latest

    @property
    def depth_scale(self):
        """
        Get the depth scale factor.

        Returns:
            float: The depth scale.
        """
        return self._config.depth_scale

    @property
    def depth_intrinsics(self):
        """
        Get the depth stream intrinsics.

        Returns:
            rs.intrinsics: The depth intrinsics.
        """
        return self._config.depth_intrinsics

    @property
    def color_intrinsics(self):
        """
        Get the color stream intrinsics.

        Returns:
            rs.intrinsics: The color intrinsics.
        """
        return self._config.color_intrinsics

    @property
    def width(self):
        """
        Get the frame width.

        Returns:
            int: The width.
        """
        return self._width

    @property
    def height(self):
        """
        Get the frame height.

        Returns:
            int: The height.
        """
        return self._height

    @property
    def frame_queue(self):
        """
        Get the frame queue.

        Returns:
            Queue: The internal frame queue.
        """
        return self._frame_queue
