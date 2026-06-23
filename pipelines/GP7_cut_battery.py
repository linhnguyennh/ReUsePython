import numpy as np
import logging
import sys
import time
from dataclasses import dataclass, field
import cv2
import pyrealsense2 as rs
from queue import Empty
#CUSTOM

from config.common import CAM_WIDTH, CAM_HEIGHT, CAM_FPS, PLC_URL, ROBOT_URL, NODE_MAP_PATH
from config.gp7_cut_config import ROI_CONFIG, SEARCH_BAND_PX, ROI_Y_OFFSET, SMOOTH_ALPHA, CUT_OFFSET_PX, DEPTH_MIN, DEPTH_MAX, PROFILE_WIDTH, ENABLE_PLC_CONNECTION, ENABLE_ROBOT_CONNECTION
from config.vectors_matrices import MARKER_TIP_IN_CAM, R_CAM_TO_BASE_GP7_SPINDLE, EXTEND_MARKER_TIP

from src.vision.realsense_stream import RealSenseStream
from src.vision.seam_detect import detect_seam, build_sobel_vis, draw_profile_panel
from src.utils.visualise_frame import draw_depth
from src.vector.vector_helper import make_3d_vector, rotate_vector_to_frame
from src.communication.opcua.opcua_client import Yaskawa_YRC1000, OPCUAClient, PLCInterface, PLCNodeMap


log_handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

@dataclass
class ROI:
    mode_label : str
    x0 : int
    y0 : int = field(init=False)
    x1 : int
    y1 : int = field(init=False)
    roi_color : tuple[int,int,int]
    roi_y_mid : int = CAM_HEIGHT // 2 + ROI_Y_OFFSET
    
    def __post_init__(self):
        self.y0 = max(0,self.roi_y_mid - SEARCH_BAND_PX)
        self.y1 = min(CAM_HEIGHT,self.roi_y_mid + SEARCH_BAND_PX)

@dataclass
class Seam:
    roi : ROI #x0,x1,y0,y1,color,label
    y_seam : int
    y_cut : int
    x_cut_left : int
    x_cut_right : int
    p3d_left : np.ndarray
    p3d_right : np.ndarray
    peak_local : int


def select_roi(rgb_data : np.ndarray, roi : ROI):
    return rgb_data[roi.y0:roi.y1,roi.x0:roi.x1]

def process_seam(rgb_data : np.ndarray, depth_data : np.ndarray, cam_config, roi : ROI, y_seam_smooth : float = None):
    roi_rgb = select_roi(rgb_data, roi)
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2GRAY)

    peak_local, profile, sobel_raw = detect_seam(roi_gray)
    
    y_seam_global = roi.y0 + peak_local
    
    #EMA temporal filtering on the seam
    if y_seam_smooth is None:
        y_seam_smooth = float(roi.roi_y_mid)
    else:
        y_seam_smooth = (SMOOTH_ALPHA * y_seam_smooth
                                + (1 - SMOOTH_ALPHA) * y_seam_global)
    
    y_seam = int(round(y_seam_smooth))
    # Cut target = seam shifted upward by CUT_OFFSET_PX
    y_cut = y_seam - CUT_OFFSET_PX
    #y_cut = np.clip(y_cut, 0, depth_data.shape[0] - 1)
    #Filter via depth
    cut_depths = depth_data[y_cut, roi.x0:roi.x1] * cam_config.depth_scale
    valid_mask = (cut_depths > DEPTH_MIN) & (cut_depths < DEPTH_MAX)

    # Find leftmost and rightmost valid depth pixel on the seam row
    valid_cols = np.where(valid_mask)[0]

    # FIX: initialise with safe fallback values before the branch so Seam
    # construction never references an unbound variable when valid_cols is empty
    x_cut_left  = roi.x0
    x_cut_right = roi.x1
    p3d_left    = None
    p3d_right   = None

    if len(valid_cols) > 0:
        x_cut_left  = roi.x0 + valid_cols[0]
        x_cut_right = roi.x0 + valid_cols[-1]

        #--> Pixel value at end points
        # Get 3D by deprojecting with depth (in meter)
        #NOTE: USE COLOR INTRINSIC BECAUSE DEPTH CAMERA IS ALIGNED TO COLOR
        p3d_left = rs.rs2_deproject_pixel_to_point(cam_config.color_intrinsics,[x_cut_left, y_cut], cut_depths[valid_cols[0]])

        p3d_right = rs.rs2_deproject_pixel_to_point(cam_config.color_intrinsics,[x_cut_right, y_cut], cut_depths[valid_cols[-1]])

        p3d_left = np.array(p3d_left, dtype=np.float32) * 1000.0 #Convert to mm
        p3d_right = np.array(p3d_right, dtype=np.float32) * 1000.0 #Convert to mm
    else:
        p3d_left = None
        p3d_right = None


    return Seam(roi, y_seam, y_cut, x_cut_left, x_cut_right, p3d_left, p3d_right, peak_local), profile, sobel_raw, y_seam_smooth


def visualise_seam(rgb_data : np.ndarray, depth_data : np.ndarray, seam : Seam, profile, sobel_raw):
    vis = rgb_data.copy()

    depth_panel = draw_depth(depth_data, CAM_WIDTH, CAM_HEIGHT)
    # Profile panel — embedded next to ROI
    roi_h = seam.roi.y1 - seam.roi.y0
    cut_row_local = seam.y_cut - seam.roi.y0
    profile_panel = draw_profile_panel(profile, seam.peak_local, roi_h, PROFILE_WIDTH, cut_row_local)
    sobel_vis = build_sobel_vis(sobel_raw, seam.peak_local, seam.y_cut)

    panel_x = seam.roi.x1 + 2
    if panel_x + PROFILE_WIDTH <= CAM_WIDTH:
        vis[seam.roi.y0:seam.roi.y1, panel_x:panel_x + PROFILE_WIDTH] = profile_panel
    else:
        pw = CAM_WIDTH - panel_x
        if pw > 10:
            vis[seam.roi.y0:seam.roi.y1, panel_x:CAM_WIDTH] = profile_panel[:, :pw]
    # Search band rectangle
    cv2.rectangle(vis, (seam.roi.x0, seam.roi.y0), (seam.roi.x1, seam.roi.y1), seam.roi.roi_color, 1)

    # Mode label
    cv2.putText(vis, seam.roi.mode_label, (seam.roi.x0, seam.roi.y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, seam.roi.roi_color, 1)

    # Spindle centerline
    cv2.line(vis, (seam.roi.x0, seam.roi.roi_y_mid), (seam.roi.x1, seam.roi.roi_y_mid), (0, 255, 255), 1)
    cv2.putText(vis, "ROI midline", (seam.roi.x1 + 4, seam.roi.roi_y_mid + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Detected seam
    cv2.line(vis, (seam.roi.x0, seam.y_seam), (seam.roi.x1, seam.y_seam), (0, 255, 0), 2)
    cv2.putText(vis, "seam", (seam.roi.x1 + 4, seam.y_seam + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Cut target line
    cv2.line(vis, (seam.x_cut_left, seam.y_cut), (seam.x_cut_right, seam.y_cut), (0, 100, 255), 2)
    cv2.putText(vis, f"cut  ({CUT_OFFSET_PX}px offset)",
                (seam.roi.x1 + 4, seam.y_cut + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    # Format left 3D coordinate text (Converting meters to mm for readability)
    if seam.p3d_left is not None and seam.p3d_right is not None:
        left_str = f"L: ({seam.p3d_left[0]:.0f}, {seam.p3d_left[1]:.0f}, {seam.p3d_left[2]:.0f})mm"
        cv2.putText(vis, left_str, (seam.x_cut_left - 50, seam.y_cut - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

        right_str = f"R: ({seam.p3d_right[0]:.0f}, {seam.p3d_right[1]:.0f}, {seam.p3d_right[2]:.0f})mm"
        cv2.putText(vis, right_str, (seam.x_cut_right - 10, seam.y_cut - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
    print(f"\r[{seam.roi.mode_label}]  "
                  f"seam={seam.y_seam}  cut={seam.y_cut}",
                  end="", flush=True)
    
    return vis, depth_panel, sobel_vis

def main():
    #CAM INIT
    rs_stream = RealSenseStream(
        width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS,
        enable_decimation = False, enable_spatial = True, enable_temporal = True, enable_hole_filling = True, enable_depth_to_disparity = True, enable_disparity_to_depth = True
    )
    
    cam_config = rs_stream.config
    frame_queue = rs_stream.frame_queue

    roi_long = ROI(
        mode_label= 'LONG EDGE',
        x0=max(0,ROI_CONFIG['long']['x_start']),
        x1=min(CAM_WIDTH,ROI_CONFIG['long']['x_end']),
        roi_color=ROI_CONFIG['long']['color']
    )
    roi_short = ROI(
        mode_label= 'SHORT EDGE',
        x0=ROI_CONFIG['short']['x_start'],
        x1=ROI_CONFIG['short']['x_end'],
        roi_color=ROI_CONFIG['short']['color']
    )
    
    if ENABLE_ROBOT_CONNECTION:
        try:
            #ROBOT OPCUA INIT
            robot_GP7 = Yaskawa_YRC1000(ROBOT_URL)
            robot_GP7.set_servo(True)
        except Exception as e:
            logger.error(f"{e}")
    else:
        robot_GP7 = None
    
    if ENABLE_PLC_CONNECTION:
        try:
            #PLC OPCUA INIT
            plc_client = OPCUAClient(PLC_URL)
            node_map = PLCNodeMap(plc_client,NODE_MAP_PATH)
            plc_io = PLCInterface(node_map,plc_client)
        except Exception as e:
            logger.error(f"{e}")
    else:
        plc_client = None
    
    #CONSTANTS
    global IS_LONG_EDGE 
    IS_LONG_EDGE = True
    y_seam_smooth = None
    
    # START CAM THREAD
    rs_stream.start()

    # PUT THREAD INIT HERE #
    try:
        while True:
            try:
                rgb_frame, depth_frame = frame_queue.get()
            except Empty:
                continue
            
            rgb_data = np.asanyarray(rgb_frame.get_data())
            depth_data = np.asanyarray(depth_frame.get_data())

            roi = roi_long if IS_LONG_EDGE else roi_short

            # PUT TASK CODE HERE #
            #PIPELINE: Select ROI (Base on SHORT OR LONG) --> Detect EDGE + Smoothing --> Filter by depth --> Generate Endpoints
            seam, profile, sobel_raw, y_seam_smooth = process_seam(rgb_data,depth_data,cam_config,roi, y_seam_smooth)

            vis, depth_panel, sobel_vis = visualise_seam(rgb_data, depth_data, seam, profile, sobel_raw)

            #--> TODO: GENERATE MOVEMENT VECTOR --> SEND TO ROBOT 

            #NEED: Vector from first point (p3d_right) to marker tip --> MOVE
            #GOING FROM RIGHT EDGE TO REACH MARKER TIP 
            if plc_client is not None and robot_GP7 is not None:
                if plc_io.get_bool_up():
                    time.sleep(0.2)
                    robot_GP7.start_job('MARKER_SHORT_UP', block=True)
                if plc_io.get_bool_down():
                    time.sleep(0.2)
                    robot_GP7.start_job('MARKER_SHORT_DOWN', block=True)
                if plc_io.get_bool_left():
                    time.sleep(0.2)
                    robot_GP7.start_job('MARKER_LONG_LEFT', block=True)
                if plc_io.get_bool_right():
                    time.sleep(0.2)
                    robot_GP7.start_job('MARKER_LONG_RIGHT', block=True)
                if plc_io.get_bool_findedge():
                    try:
                        point_to_marker = make_3d_vector(seam.p3d_right, EXTEND_MARKER_TIP) #Vector from cutting line endpoint to marker tip
                        ptm_in_base = rotate_vector_to_frame(point_to_marker, R_CAM_TO_BASE_GP7_SPINDLE)

                        ptm_arr_float = np.zeros(8)
                        ptm_arr_float[:3] = ptm_in_base
                        ptm_arr_float = ptm_arr_float.tolist()
                        
                        # print(f"p3d_right: {seam.p3d_right}")
                        # print(f"Converted p3d_right: {ptm_arr_float}")
                        plc_io.set_point_to_marker(ptm_arr_float)
                    except Exception as e:
                        logger.error(f"{e}")

                    #GOING FROM RIGHT EDGE TO LEFT EDGE
                    try:
                        #start is left and end is right because robot control
                        right_to_left = make_3d_vector(seam.p3d_left, seam.p3d_right) #Vector from cutting line endpoint to marker tip
                        rtl_in_base = rotate_vector_to_frame(right_to_left, R_CAM_TO_BASE_GP7_SPINDLE)

                        rtl_arr_float = np.zeros(8)
                        rtl_arr_float[:3] = rtl_in_base
                        rtl_arr_float = rtl_arr_float.tolist()
                        
                        plc_io.set_right_to_left(rtl_arr_float)
                    except Exception as e:
                        logger.error(f"{e}")

            #NEED: Vector from current position (marker tip) to end point --> MOVE
            
            #TRANSITION: MOVEMENT DONE --> CALL JOB

            #REPEAT AFTER ROTATION FROM JOB
            #AFTER 4 ROTATION END

            cv2.imshow("Seam Detection", vis)
            cv2.imshow("Sobel ROI", sobel_vis)
            cv2.imshow("Depth", depth_panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('m'):
                IS_LONG_EDGE = not IS_LONG_EDGE
                y_seam_smooth = float(roi.roi_y_mid)   # reset smooth on mode change
                print(f"\nMode → {'LONG' if IS_LONG_EDGE else 'SHORT'} EDGE")
            
            #time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt triggered...")
        rs_stream.stop()
        if robot_GP7 is not None:
            robot_GP7.stop_communication()
        if plc_client is not None:
            plc_client.stop_communication()
        cv2.destroyAllWindows()
        logger.info("All threads terminated")
    finally:
        if robot_GP7 is not None:
            robot_GP7.stop_communication()
        if plc_client is not None:
            plc_client.stop_communication()

if __name__ == '__main__':
    main()