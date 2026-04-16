import time
import threading
from queue import Queue, Empty
from ultralytics import YOLO
import numpy as np
import sys
import os
import logging
import csv
# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.realsense_stream import RealSenseStream
from src.vision.pipeline_workers import DetectionWorker, DisplayWorker
from src.communication.opcua_device import PLCClient, Yaskawa_YRC1000
from src.vision.visual_controller import calc_control_val, check_stability, tf_camera_to_gripper

def main():
    # --- Control parameters ---
    # Kp: 0.2, Kd: 0.4, ALPHA: 0.32
    Kp = 0.3
    Kd = 0.0
    ALPHA = 0.5       # smoothing factor (0–1)
    DEADBAND_M = 0.001  # 10 mm deadband
    LOOP_HZ = 30
    LOOP_DT = 1.0 / LOOP_HZ

    STABILITY_THRESHOLD = 0.005   # 5 mm in meters
    STABILITY_TIME = 2.0          # seconds required for stability

    stable_timer_start = None
    is_stable = False

    # --- Connection URLs ---
    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"
    postcp_index = 0
    # --- Start camera and model ---
    try:
        camera = RealSenseStream(fps=30, width=640, height=480)
        camera.start()
    except Exception as e:
        print(e)

    #model = YOLO(r"models/focus1/retrain/train3/weights/best_morrow_251020.pt")
    #model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.pt")
    model = YOLO("./model/morrow_obb_260108_ncnn_model", task='obb')
    is_obb = True
    try:
        detection_worker = DetectionWorker(
            model=model,
            camera=camera,
            max_queue_size=1,
            obb=is_obb,
            conf=0.5,
            imgsz=640
        )
        display_worker = DisplayWorker(
            width=camera.width,
            height=camera.height,
            depth_scale=camera.depth_scale,
            results_queue=detection_worker.results_queue,
            obb=is_obb,
            limit_box=True
        )

        detection_worker.start()
        display_worker.start()
    except Exception as e:
        print(e)
    with PLCClient(plc_url) as plc:
        try:
            logging.info("[Main] Visual servo control loop started.")
            # --- Initialize control state ---
            last_error = np.zeros(3)
            smoothed_error = np.zeros(3)
            stable_timer_start = None
            is_stable = False
            plc.set_breakloop(False)
            control_xz = np.zeros(2)   # Predefine
            error = np.zeros(3)
            error_raw = np.zeros(3)
            angle_degree = 0

            # ===== LOGGING STATE (ADDED) =====
            logging_active = False
            csv_file = None
            csv_writer = None
            log_start_time = None
            log_frame_idx = 0
            trial_id = 1
            # =================================

            while display_worker.running:
                
                
                ###------ DETECTION ------###
                try:
                    detections = detection_worker.detections_queue.get(timeout=0.2)

                    # Filter detection for battery housing
                    housing_detection = next(
                    (det for det in detections if det["class_name"] == "battery_housing"),
                    None
                )
                except Empty:
                    continue

                
                

                if housing_detection:
                    cx, cy = housing_detection["center_2d"]

                    # Check if within limit box
                    if (camera.width//2 - 200) < cx < (camera.width//2 + 200) and (camera.height//2 - 125) < cy < (camera.height//2 + 125):
                        # Object offset in gripper frame (error signal)
                        error = np.array(housing_detection["xyz_gripper_frame"]) #Error of camera frame but rotated to gripper frame, no shift
                        error_raw = np.array(housing_detection["xyz"])
                        last_error, control = calc_control_val(error, last_error,ALPHA,Kp,Kd)
                        control_xz = np.array([control[0], control[2]])
                        #control_y = error[1]
                        error_xz = np.array([error[0], error[2]])
                        error_mag_xz = np.linalg.norm(error_xz)
                        angle_degree = housing_detection["long_side_normalized"]
                        
                        stable_timer_start, is_stable = check_stability(
                            error_mag_xz,
                            STABILITY_THRESHOLD,
                            STABILITY_TIME,
                            stable_timer_start,
                            is_stable
                                ) 
                        print(is_stable) 
                        print(error_mag_xz)
                    else:
                        control_xz = np.array([0, 0])
                        
                ###------ ------ ------###

                if is_stable:
                    plc.set_breakloop(True)
                    plc.set_trigger(True)


                state_job = plc.get_state_job()
                
                ###
                        # Put error logging here:
                        # log actual error: error_xz or error[0] and error[2]
                        # log controller output: control_xz or control[0] and control[2]
                        # log magnitude: error_mag_xz
                        # log is_stable
                ###

                # ===== START LOGGING ON SERVO STATES =====
                if state_job in [11, 12, 13] and not logging_active:
                    os.makedirs("logs", exist_ok=True)
                    csv_file = open(
                        f"logs/trial_{trial_id:03d}_visualservo.csv",
                        "w",
                        newline=""
                    )
                    csv_writer = csv.DictWriter(csv_file, fieldnames=[
                        "t",
                        "frame",
                        "error_x_m",
                        "error_z_m",
                        "err_mag",
                        "control_x",
                        "control_z",
                        "stable"
                    ])
                    csv_writer.writeheader()

                    log_start_time = time.monotonic()
                    log_frame_idx = 0
                    logging_active = True
                # =======================================


                match state_job:
                    case s if s in [11,12,13]:

                        loop_start = time.time() 

                        # ===== LOG SERVO DATA =====
                        if logging_active:
                            csv_writer.writerow({
                                "t": time.monotonic() - log_start_time,
                                "frame": log_frame_idx,
                                "error_x_m": error[0], #error x and z in meters from camera
                                "error_z_m": error[2],
                                "err_mag": error_mag_xz,
                                "control_x": control_xz[0],
                                "control_z": control_xz[1],
                                "stable": int(is_stable)
                            })
                            log_frame_idx += 1
                        # =========================

                        if abs(np.linalg.norm(control_xz)) > DEADBAND_M:
                            #dx, dy, dz = control.tolist() #Delta xyz
                            dx, dz = control_xz.tolist()
                            # Clamp movement per step (e.g. ≤ 20 mm)
                            dx = np.clip(dx, -0.02, 0.02)
                            dz = np.clip(dz, -0.02, 0.02)

                            if postcp_index == 0:
                                plc.send_coordinates0(
                                    x=dx * 1000,  # m → mm
                                    y=0,
                                    z=dz * 1000
                                )
                            if postcp_index == 1:
                                plc.send_coordinates1(
                                    x=dx * 1000,  # m → mm
                                    y=0,
                                    z=dz * 1000
                                )
                            if postcp_index == 2:
                                plc.send_coordinates2(
                                    x=dx * 1000,  # m → mm
                                    y=0,
                                    z=dz * 1000
                                )

                            postcp_index += 1
                            if postcp_index > 3:
                                postcp_index = 0



                            # Short, safe trigger pulse
                            plc.set_trigger(True)
                            #time.sleep(0.02)
                            plc.set_trigger(False)

                        # --- Maintain control frequency ---
                        elapsed = time.time() - loop_start
                        time.sleep(max(0.0, LOOP_DT - elapsed))
                    case 14:
                        
                        if logging_active:
                            csv_writer.writerow({
                                "t": time.monotonic() - log_start_time,
                                "frame": log_frame_idx,
                                "error_x_m": error[0], #error x and z in meters from camera
                                "error_z_m": error[2],
                                "err_mag": error_mag_xz,
                                "control_x": control_xz[0],
                                "control_z": control_xz[1],
                                "stable": int(is_stable)
                            })
                            log_frame_idx += 1
                        # =========================

                        error_shifted = tf_camera_to_gripper(error_raw)
                        dx,dy,dz = error_shifted.tolist()
                        plc.send_coordinates3(
                            x=dx*1000,
                            y=dy*1000,
                            z=dz*1000,
                            ry=angle_degree if angle_degree else 0)
                        plc.set_stepz(True)
                        plc.set_stepz(False)
                    case 15:
                        plc.set_closegripper(True)
                        plc.set_closegripper(False)
                        time.sleep(1.5)
                    case 16:
                        plc.set_trigger(False)
                        with Yaskawa_YRC1000(robot_url) as robot:
                            robot.set_servo(True)
                            robot.start_job('CAM_HOME', block=True)
                            robot.start_job('BATTERY_PLACE_MORROW', block=True)
                        plc.set_trigger(True)
                        plc.set_trigger(False)
                    case 17:
                        plc.set_opengripper(True)
                        plc.set_opengripper(False)
                        time.sleep(1.5)
                    case 18:
                        plc.set_trigger(False)
                        with Yaskawa_YRC1000(robot_url) as robot:
                            robot.set_servo(True)
                            robot.start_job('BATTERY_RESET_MORROW', block=True)
                            robot.start_job('CAM_HOME', block=True)
                        plc.set_trigger(True)
                        plc.set_trigger(False)

                # ===== STOP LOGGING WHEN SERVOING ENDS =====
                if logging_active and state_job not in [11, 12, 13]:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                    logging_active = False
                    trial_id += 1
                # ========================================
        except KeyboardInterrupt:
            logging.info("[Main] Keyboard interrupt detected. Stopping...")

        finally:
            logging.info("[Main] Cleaning up...")
            plc.send_coordinates0(x=0, y=0, z=0)
            plc.send_coordinates1(x=0, y=0, z=0)
            plc.send_coordinates2(x=0, y=0, z=0)
            robot.set_servo(False)
            detection_worker.stop()
            camera.stop()
            logging.info("[Main] All threads stopped cleanly.")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
