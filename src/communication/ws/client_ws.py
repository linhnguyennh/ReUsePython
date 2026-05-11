import cv2
import numpy as np
import threading
import logging
import sys
from pathlib import Path
from queue import Queue, Empty
from websockets.sync.client import connect
from typing import Callable, Optional
from ultralytics import YOLO
import time
from math import pi

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.communication.ws.ws_helper import encode_frame, decode_pose, TYPE_POSE
from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import SegmentationWorker
from src.pose.pose_process_fn import transform_pose, get_rotation_matrix, get_x_axis, get_y_axis, get_z_axis, euler_angles, axis_angle, align_axis

logger = logging.getLogger(__name__)


class StreamClientWebSocket:
    def __init__(
        self,
        ws_url: str,
        frame_and_mask_queue: Queue,
        encoder: Optional[Callable[..., bytes]] = None,
    ):
        self.ws_url      = ws_url
        self._frame_and_mask_queue = frame_and_mask_queue
        self.encoder     = encoder or self._default_encoder
        self.ws          = None
        self._running    = False
        self.T_cam_obj = None

        self._pose_queue = Queue(maxsize=1)

    # ---------- default encoder ----------
    @staticmethod
    def _default_encoder(rgb, depth=None, mask=None) -> bytes:
        return encode_frame(rgb, depth, mask)

    # ---------- setup ----------
    def _connect_ws(self):
        self.ws = connect(self.ws_url)

    # ---------- send loop ----------
    def _send_loop(self):
        logger.info("Send loop started")
        while self._running:
            try:
                rgb_frame, depth_frame, mask = self._frame_and_mask_queue.get(timeout=1.0)
            except Empty:
                continue
            except ValueError:
                logger.error("Expected (rgb, depth, mask) tuple in queue")
                continue
            
            rgb_data   = np.asanyarray(rgb_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            if mask is not None:
                mask = mask.astype(np.uint8)
            try:
                packet = self.encoder(rgb_data, depth_data, mask)
                self.ws.send(packet)
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

        logger.info("Send loop stopped")

    def _pose_loop(self):
        logger.info("Pose receive loop started")
        while self._running:
            try:
                data = self.ws.recv(timeout=1.0)
                if isinstance(data, bytes):
                    result = decode_pose(data)
                    if result["type"] == TYPE_POSE:
                        #self.T_cam_obj = result["pose"]
                        obj_pose_cam_frame = result["pose"]
                        # PUT POSE DATA INTO A QUEUE AND EXPOSE IT FOR DOWNSTREAM PROCESSING
                        if self._pose_queue.full():
                            try:
                                self._pose_queue.get_nowait()
                            except Empty:
                                pass
                        self._pose_queue.put_nowait(obj_pose_cam_frame)

                        #gripper_obj_pose = T_cam_to_gripper @ self.T_obj_to_cam
                        #logger.info(f"Gripper Pose Matrix: {np.round(obj_pose_cam_frame,2)}")
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Pose receive error: {e}")
                break

        logger.info("Pose receive loop stopped")



    # ---------- run ----------
    def start(self):
        self._connect_ws()
        self._running = True

        send_thread = threading.Thread(target=self._send_loop, daemon=True)
        pose_thread = threading.Thread(target=self._pose_loop, daemon=True)

        send_thread.start()
        pose_thread.start()

        try:
            while self._running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("KeyboardInterrupt received → stopping client")
            self._running = False

        finally:
            send_thread.join(timeout=2)
            pose_thread.join(timeout=2)

    def stop(self):
        self._running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
        logger.info("Client closed")

    @property
    def pose_queue(self):
        return self._pose_queue





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)


    T_cam_to_gripper =  np.array([[0.0,     -1.0,   -0.3,   0.088],
                          [1.0,      0.0,  0.013,  -0.035],
                          [-0.01,   -0.3,   0.95,  -0.041],
                          [0.0,      0.0,    0.0,     1.0]])

    


    rs_stream = RealSenseStream(
        width=640, height=480, fps=30,
        enable_decimation=False,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=False
    )
    
    frame_queue = rs_stream.frame_queue
    seg_model = YOLO(r"models\focus1\260421\train\weights\best.pt")

    segmentor = SegmentationWorker(
        model = seg_model,
        frame_queue = frame_queue,
        max_queue_size=1,
        conf=0.6,
        device='cuda',
        verbose = False
    )
    
    segmentor_out_queue = segmentor.mask_queue


    

    client = StreamClientWebSocket(
        ws_url="ws://localhost:8000/ws", #port 1000 for local windows cross-script, port 8000 for script-to-container
        frame_and_mask_queue=segmentor_out_queue,
    )

    pose_queue = client.pose_queue

    running = True

    def test_pose_process_loop():
        global running
        global T_cam_to_gripper
        robot_z = np.array([0,0,1])
        while running:
            try:
                pose_obj_to_cam = pose_queue.get(timeout=0.5)

                pose_obj_to_gripper = transform_pose(pose_obj_to_cam)#, T_cam_to_gripper)

                obj_to_gripper_rotation_matrix = get_rotation_matrix(pose_obj_to_gripper)

                pose_x_axis = get_x_axis(obj_to_gripper_rotation_matrix)
                pose_y_axis = get_y_axis(obj_to_gripper_rotation_matrix)
                pose_z_axis = get_z_axis(obj_to_gripper_rotation_matrix)


                print(f"Transformed posed: \n {pose_obj_to_gripper}")
                print(f"Rotation matrix: \n {obj_to_gripper_rotation_matrix}")
                print(f"X axis: {pose_x_axis}")
                print(f"Y axis: {pose_y_axis}")
                print(f"Z axis: {pose_z_axis}")

                axis, angle = axis_angle(robot_z, pose_y_axis)
                print(f"Axis: {axis}, Angle: {angle*180/pi}")
                rz,ry,rx = align_axis(robot_z, pose_y_axis)
                print(f"RX: {rx:.2f}, RY: {ry:.2f}, RZ:{rz:.2f}")


                dx = np.dot(robot_z,pose_x_axis)
                dy = np.dot(robot_z,pose_y_axis)
                dz = np.dot(robot_z,pose_z_axis)

                print(f"dotX: {dx:.2f}, dotY: {dy:.2f}, dotZ: {dz:.2f}")



                time.sleep(1)
            except Empty:
                continue
    pose_process_t = threading.Thread(target=test_pose_process_loop, daemon=True)    
    # POSE QUEUE START

    try:
        rs_stream.start()
        segmentor.start()
        
        pose_process_t.start()
        client.start()
    finally:
        running = False
        pose_process_t.join()
        segmentor.stop()
        rs_stream.stop()
        client.stop()
        