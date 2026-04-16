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
from src.communication.opcua.opcua_device import PLCClient, Yaskawa_YRC1000
from src.vision.visual_controller import calc_control_val, check_stability, tf_camera_to_gripper

def main():
    # --- Control parameters ---
    Kp = 0.2
    Kd = 0.4
    ALPHA = 0.32       # smoothing factor (0–1)
    DEADBAND_M = 0.001  # 10 mm deadband
    LOOP_HZ = 15
    LOOP_DT = 1.0 / LOOP_HZ

    STABILITY_THRESHOLD = 0.003   # 5 mm in meters
    STABILITY_TIME = 2.0          # seconds required for stability

    stable_timer_start = None
    is_stable = False

    # --- Connection URLs ---
    plc_url = "opc.tcp://192.168.0.1:4840"
    robot_url = "opc.tcp://192.168.0.20:16448"
    postcp_index = 0
    # --- Start camera and model ---
    camera = RealSenseStream(fps=30, width=1280, height=720)
    camera.start()

    #model = YOLO(r"models/focus1/retrain/train3/weights/best_morrow_251020.pt")
    model = YOLO(r"models\focus1\retrain_obb_BIGMAP_251203\train2\weights\morrow_obb_251203.pt")
    is_obb = True
    detection_worker = DetectionWorker(
        model=model,
        camera=camera,
        max_queue_size=1,
        obb=is_obb,
        conf=0.85,
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
    # Yaskawa_YRC1000(robot_url) as robot,
    with  PLCClient(plc_url) as plc:
        try:
            logging.info("[Main] Visual servo control loop started.")
            # --- Initialize control state ---
            last_error = np.zeros(3)
            smoothed_error = np.zeros(3)
            stable_timer_start = None
            is_stable = False

            while display_worker.running:
                loop_start = time.time()
                try:
                    detections = detection_worker.detections_queue.get(timeout=0.2)
                except Empty:
                    continue

                # Filter detection for battery housing
                housing_detection = next(
                    (det for det in detections if det["class_name"] == "battery_housing"),
                    None
                )

                if housing_detection:
                    cx, cy = housing_detection["center_2d"]

                    # Check if within limit box
                    if (640 - 400) < cx < (640 + 400) and (360 - 250) < cy < (360 + 250):
                        # Object offset in gripper frame (error signal)
                        error = np.array(housing_detection["xyz_gripper_frame"])
                        last_error, control = calc_control_val(error, last_error,ALPHA,Kp,Kd)
                        control_xz = np.array([control[0], control[2]])
                        control_y = error[1]
                        error_xz = np.array([error[0], error[2]])
                        error_mag_xz = np.linalg.norm(error_xz)
                        
                        # --- Stability detection logic (X/Z only) ---
                        

                        stable_timer_start, is_stable = check_stability(
                            error_mag_xz,
                            STABILITY_THRESHOLD,
                            STABILITY_TIME,
                            stable_timer_start,
                            is_stable
                        )

                        if is_stable:
                            dy = control_y
                            plc.set_trigger(True)
                            plc.set_trigger(False)
                            plc.set_breakloop(True)
                            plc.send_coordinates3(
                                x = 0,
                                y = dy,
                                z = 0
                            )
                            plc.set_stepz(True)
                            plc.set_stepz(False)



                        # --- Deadband check ---

                        if not is_stable and np.linalg.norm(control_xz) > DEADBAND_M:
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
                            
                        # plc.set_stepz(True)
                            
                        # plc.set_stepz(False)
                        # Update memory
                        #last_error = smoothed_error

                # --- Maintain control frequency ---
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, LOOP_DT - elapsed))

        except KeyboardInterrupt:
            logging.info("[Main] Keyboard interrupt detected. Stopping...")

        finally:
            logging.info("[Main] Cleaning up...")
            plc.send_coordinates0(x=0, y=0, z=0)
            plc.send_coordinates1(x=0, y=0, z=0)
            plc.send_coordinates2(x=0, y=0, z=0)
            detection_worker.stop()
            camera.stop()
            logging.info("[Main] All threads stopped cleanly.")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
