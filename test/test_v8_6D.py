from pathlib import Path
import sys
from ultralytics import YOLO
from queue import Queue, Empty
import time
import numpy as np
import logging
import threading
from enum import IntEnum

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))

from src.communication.ws.client_ws import StreamClientWebSocket
from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import SegmentationWorker
from src.utils.logger_helper import log_title
from src.pose.pose_process_fn import *
from src.communication.opcua.opcua_client import PLCClient, PLCInterface, PLCNodeMap, OPCUAClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



def process_pose(
        pose_obj_to_cam : np.ndarray,
        T_cam_to_gripper : np.ndarray,
        robot_z : np.ndarray,
        offset_approach_meter : float,
        gripper_depth : float,
        debug : bool = False
):
    try:
        #INPUT POSE --> TRANSFORM TO GRIPPER FRAME
        pose_obj_to_gripper = transform_pose(pose_obj_to_cam, T_cam_to_gripper)
        #GET R MATRIX
        obj_to_gripper_R_matrix = get_rotation_matrix(pose_obj_to_gripper)
        
        #GET p VECTOR
        pose_position = get_position(pose_obj_to_gripper)

        #GET XYZ AXES
        pose_x_axis = get_x_axis(obj_to_gripper_R_matrix)
        pose_y_axis = get_y_axis(obj_to_gripper_R_matrix)
        pose_z_axis = get_z_axis(obj_to_gripper_R_matrix)
        pose_z_axis = -pose_z_axis #FLIP Z AXIS (NOT SURE WHY THO BUT IT WORKS)

        #GENERATE BEST AXIS TO ALIGN ROBOT Z WITH (MAX DOT PRODUCT)
        pose_axis_to_align, dot_to_align = compare_dot_product(pose_y_axis, pose_x_axis, robot_z)

        #CHECK ALIGNMENT DIRECTION (dot <0 towards, >0 away)
        if dot_to_align >= 0:
            is_away = True
        else:
            is_away = False
        
        if not is_away:
            pose_axis_to_align = -pose_axis_to_align

        #GENERATE RX RY RZ FOR ROBOT in RZRYRX SEQUENCE
        rz, ry, rx = align_axis(robot_z, pose_axis_to_align)
        #CALCULATE PREGRASP POSITION AND APPROACH VECTOR
        pre_grasp_position, approach_vector = pre_grasp_xyz(pose_position, pose_axis_to_align, offset_approach_meter, gripper_depth)
        
        if debug:
            logger.info("Transformed pose: \n%s", pose_obj_to_gripper)
            log_title(logger, "DOT PRODUCT OF ROBOT Z AND AXES")
            dot_x = np.dot(robot_z,pose_x_axis)
            dot_y = np.dot(robot_z,pose_y_axis)
            dot_z = np.dot(robot_z,pose_z_axis)
            logger.info("dot_BLUE_X: %.2f, dot_GREEN_Y: %.2f, dot_RED_Z: %.2f", dot_x, dot_y, dot_z)
            log_title(logger, "ANGLE TO ROTATE") 
            logger.info("RX: %.2f, RY: %.2f, RZ: %.2f", rx,ry,rz)
            log_title(logger, "OBJECT POS, PREGRASP, APPROACH")
            logger.info("OBJECT POS: %s", pose_position)
            logger.info("PREGRASP: %s \nAPPROACH: %s" ,pre_grasp_position, approach_vector)
            log_title(logger,"")
        return pre_grasp_position, np.array([rx, ry, rz]), approach_vector
    except Exception as e:
        logger.exception("Pose processing failed: %s", e)

def pose_thread(
        pose_queue : Queue,
        stop_event : threading.Event,
        pose_process_function,
        *arg

):
    while not stop_event.is_set():  
        try:
            pose_obj_to_cam = pose_queue.get(timeout=0.5)
        except Empty:
            continue
        try:
            pre_grasp_position, (rx, ry, rz), approach_vector = pose_process_function(pose_obj_to_cam, *arg)
        except Exception as e:
            logger.exception("Pose thread failed: %s", e)

class MotionState(IntEnum):
    IDLE = 0
    MOVE_PREGRASP = 1 
    MOVE_WRIST = 2
    MOVE_APPROACH = 3
    GRASP = 4
    STATE_5 = 5
def main():
    T_cam_to_gripper =  np.array([
        [0.0,     -1.0,   -0.3,   0.088],
        [1.0,      0.0,  0.013,  -0.035],
        [-0.01,   -0.3,   0.95,  -0.041],
        [0.0,      0.0,    0.0,     1.0]
    ])

    ROBOT_X = np.array([1,0,0])
    ROBOT_Z = np.array([0,0,1])

    #INIT
    #REALSENSE THREAD
    rs_stream = RealSenseStream(
        width=640, height=480, fps=30,
        enable_decimation=False,
        enable_spatial=True,
        enable_temporal=True,
        enable_hole_filling=False
    )

    frame_queue = rs_stream.frame_queue

    seg_model = YOLO(r"models\focus1\260421\train\weights\best.pt")
    #SEGMENTATION THREAD
    segmentor = SegmentationWorker(
        model = seg_model,
        frame_queue = frame_queue,
        max_queue_size=1,
        conf=0.6,
        device='cuda',
        verbose = False
    )

    segmentor_out_queue = segmentor.mask_queue

    #POSE RECEIVER THREAD
    ws_client = StreamClientWebSocket(
        ws_url="ws://localhost:8000/ws", #port 1000 for local windows cross-script, port 8000 for script-to-container
        frame_and_mask_queue=segmentor_out_queue,
    )

    pose_queue = ws_client.pose_queue

    #OPCUA CLIENTS
    plc_url = "opc.tcp://192.168.0.1:4840"
    plc_client = OPCUAClient(plc_url)

    node_map = PLCNodeMap(plc_client, r"C:\Users\lin40269\Desktop\Linh (Desktop)\01_Python\realsense\config\plc_opcua_nodes.yaml")
    
    plc_io = PLCInterface(node_map,plc_client)

    #START THREADS
    rs_stream.start()
    segmentor.start()

    ws_client_thread = threading.Thread(
        target=ws_client.start,
        daemon=True
    )
    ws_client_thread.start()

    #KEEP MAIN ALIVE
    running = True
    try:
        while running:
            state_motion = MotionState(plc_io.get_state_motion())
            match state_motion:
                case MotionState.IDLE:
                    try:
                        pose_obj_to_cam = pose_queue.get(timeout=0.5)
                    except Empty:
                        continue

                    pre_grasp_position, rxryrz, approach_vector = process_pose(pose_obj_to_cam, T_cam_to_gripper, ROBOT_Z, 0.14, 0.06, True)

                    #Pad value
                    pre_grasp_position = pre_grasp_position*1000.0 #Scale to mm
                    approach_vector = approach_vector*1000.0

                    pre_grasp_position = np.pad(pre_grasp_position, (0, 8 - len(pre_grasp_position)), mode='constant')
                    

                    rx_only = rxryrz[0]
                    wrist_rotation = np.zeros(8)

                    wrist_rotation[3] = rx_only

                    approach_vector = np.pad(approach_vector, (0, 8 - len(approach_vector)), mode='constant')
                    if plc_io.get_bool_6D_pose_data():
                        plc_io.set_pregrasp_tcp(pre_grasp_position.tolist())
                        logger.info("PREGRASP SENT")
                        plc_io.set_wrist_rotation_tcp(wrist_rotation.tolist())
                        logger.info("WRIST ROTATION SENT")
                        plc_io.set_approach_tcp(approach_vector.tolist())
                    time.sleep(2)

            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt triggered...")
        plc_client.stop_communication()
        segmentor.stop()
        rs_stream.stop()
        ws_client_thread.join()
        logger.info("All threads terminated")


if __name__ == "__main__":
    main()