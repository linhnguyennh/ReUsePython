import pyrealsense2 as rs
import cv2
import numpy as np

def realsense_init(width = 1280, height = 720, fps = 15, enable_imu = False) -> rs.pipeline:
    """
    Initialize RealSense pipeline with params

    Args:
        width (int): Frame width
        height (int): Frame height
        fps (int): (Max) Frame rate
        enable_imu (bool): Enable accel and gyro streams
    Returns:
        pipeline: Started RealSense pipeline
        depth_scale:
        depth_intrinsics: 
    """

    pipeline = rs.pipeline()
    config = rs.config()

    # Depth
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps) 
    
    # Color
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Accel & Gyro
    if enable_imu:
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    return pipeline, depth_scale, depth_intrinsics




def realsense_get_frame(pipeline):
    """
    Get frame from Intel Realsense camera via started pipeline object

    Args:
        pipeline (rs.pipeline) : started pipeline

    Returns:
        depth_Frame (np.ndarray) : depth frame of shape (H,W) uint16;
        color_frame (np.ndarray) : RGB color frame of shape (H,W,3) uint8
    """

    #Grab frame
    align_to = rs.stream.color
    align = rs.align(align_to)

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not color_frame or not depth_frame:
        return None, None

    
    return color_frame, depth_frame