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

root_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root_dir))

from src.communication.ws.ws_helper import encode_frame, decode_pose, TYPE_POSE
from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import SegmentationWorker
from src.pose.pose_process_fn import *

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

        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._pose_thread = threading.Thread(target=self._pose_loop, daemon=True)

        self._send_thread.start()
        self._pose_thread.start()

        logger.info("Client started")

    def stop(self):
        self._running = False

        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing websocket: {e}")
        
        self._send_thread.join(timeout=2)
        self._pose_thread.join(timeout=2)

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

    T_cam_to_cam = np.eye(4)
    


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
        robot_x = np.array([1,0,0])
        robot_z = np.array([0,0,1])
        while running:
            try:
                pose_obj_to_cam = pose_queue.get(timeout=0.5) #POSE IS a 4x4 matrix of obj in cam frame
            
                pose_obj_to_gripper = transform_pose(pose_obj_to_cam, T_cam_to_gripper)
                
                obj_to_gripper_rotation_matrix = get_rotation_matrix(pose_obj_to_gripper)

                #Unpacking
                pose_x_axis = get_x_axis(obj_to_gripper_rotation_matrix) #Along long side (thinest) BLUE
                pose_y_axis = get_y_axis(obj_to_gripper_rotation_matrix) #Towards Terminal GREEN
                pose_z_axis = get_z_axis(obj_to_gripper_rotation_matrix) #RED
                pose_z_axis = -pose_z_axis #RED AXIS IS FLIPPED HERE IDK WHY
                # PUT POSE IN LIST/DICT and THEN USE INDEX 
                pose_position = get_position(pose_obj_to_gripper)

                print("===== TRANSFORMED MATRICES =====")
                print(f"Transformed posed: \n {pose_obj_to_gripper}")
                print(f"Rotation matrix: \n {obj_to_gripper_rotation_matrix}")

                # Generate which axis is best (which one is the most aligned with z-axis)
                pose_axis_to_align, dot_to_align = compare_dot_product(pose_y_axis, pose_x_axis, robot_z) #Generate the axis and its dot product with reference for alignment, we dont care about the z axis because it's not a graspable axis

                #Check alignment direction (towards (dot<0) or away (dot>0) from camera)
                if dot_to_align >= 0:
                    is_away = True
                else:
                    is_away = False

                if not is_away: #Enforce is pointing away from gripper Z
                    pose_axis_to_align = -pose_axis_to_align

                print("===== DOT PRODUCTS =====")
                dx = np.dot(robot_z,pose_x_axis)
                dy = np.dot(robot_z,pose_y_axis)
                dz = np.dot(robot_z,pose_z_axis) 
                
                print(f"dotBLUE_X: {dx:.2f}, dotGREEN_Y: {dy:.2f}, dotRED_Z: {dz:.2f}")                


                print("===== ANGLE TO ROTATE =====")
                s_angle = signed_angle(robot_z, pose_axis_to_align, robot_x)
                sym_angle  = symmetric_angle(s_angle)
                print(f"Signed Angle: {s_angle:.2f}, Sym. Angle: {sym_angle:.2f}")

                rz,ry,rx = align_axis(robot_z, pose_axis_to_align)
                print(f"RX: {rx:.2f}, RY: {ry:.2f}, RZ:{rz:.2f}")

                print("===== OBJECT POSITION AND PREGRASP IN GRIPPER FRAME =====")
                print(f"Object POSITION: {np.round(pose_position, 2)}")
                pre_grasp_position, approach_vector = pre_grasp_xyz(pose_position, pose_axis_to_align, 0.140, 0.6) #A point 14cm on the line of the axis to-be-aligned away from object
                print(f"Pregrasp: {np.round(pre_grasp_position,2)}, Approach: {np.round(approach_vector,2)}")



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

# def test_pose_process_loop():
#         global running
#         global T_cam_to_gripper
#         robot_z = np.array([0,0,1])
#         while running:
#             try:
#                 pose_obj_to_cam = pose_queue.get(timeout=0.5) #POSE IS a 4x4 matrix of obj in cam frame

#                 #pose_obj_to_gripper = transform_pose(pose_obj_to_cam)#, T_cam_to_gripper)
#                 pose_obj_to_gripper = transform_pose(pose_obj_to_cam, T_cam_to_cam)
#                 obj_to_gripper_rotation_matrix = get_rotation_matrix(pose_obj_to_gripper)

#                 pose_x_axis = get_x_axis(obj_to_gripper_rotation_matrix)
#                 pose_y_axis = get_y_axis(obj_to_gripper_rotation_matrix)
#                 pose_z_axis = get_z_axis(obj_to_gripper_rotation_matrix)
#                 # PUT POSE IN LIST/DICT and THEN USE INDEX 

#                 print("===== MATRICES =====")
#                 print(f"Transformed posed: \n {pose_obj_to_gripper}")
#                 print(f"Rotation matrix: \n {obj_to_gripper_rotation_matrix}")
#                 print(f"X axis: {pose_x_axis}")
#                 print(f"Y axis: {pose_y_axis}")
#                 print(f"Z axis: {pose_z_axis}")

#                 print("===== ANGLE TO AXIS =====")
#                 axis, angle = axis_angle(robot_z, pose_y_axis)
#                 print(f"Axis: {axis}, Angle: {angle*180/pi}")
#                 rz,ry,rx = align_axis(robot_z, pose_y_axis)
#                 print(f"RX: {rx:.2f}, RY: {ry:.2f}, RZ:{rz:.2f}")

#                 print("===== DOT PRODUCTS =====")
#                 dx = np.dot(robot_z,pose_x_axis)
#                 dy = np.dot(robot_z,pose_y_axis)
#                 dz = np.dot(robot_z,-pose_z_axis) #RED AXIS IS FLIPPED HERE IDK WHY

#                 print(f"dotBLUE_X: {dx:.2f}, dotGREEN_Y: {dy:.2f}, dotRED_Z: {dz:.2f}")
#                 # print("===== DIRECTION CHECK =====")
#                 # is_away_x = is_pointing_away(robot_z, pose_x_axis)
#                 # is_away_y = is_pointing_away(robot_z, pose_y_axis)
#                 # is_away_z = is_pointing_away(robot_z, pose_z_axis)

#                 # COMPARE BLUE AND GREEN ONLY (Based on absolute value) --> MAX OF THESE TWO GETS THE WIN
#                 # WHICH EVER WINS: DIRECTION CHECK --> pointing towards then vector +, else vector -
#                 # THEN GENERATE PREGRASP XYZ and rotation
#                 # THEN MOVE IN DELTA DIRECTION
#                 # THEN GRASP

#                 # # if is_away_x:
#                 # #     print("X pointing same direction")
#                 # # else:
#                 # #     print("X pointing opposite")

#                 # # if is_away_y:
#                 # #     print("Y pointing same direction")
#                 # # else:
#                 # #     print("Y pointing opposite")

#                 # # if is_away_z:
#                 # #     print("Z pointing same direction")
#                 # # else:
#                 # #     print("Z pointing opposite")


#                 time.sleep(1)
#             except Empty:
#                 continue 