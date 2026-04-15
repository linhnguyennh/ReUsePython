import pyrealsense2 as rs
import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class RealSenseConfig:
    pipeline: rs.pipeline
    align: rs.align
    depth_scale: float
    color_intrinsics: rs.intrinsics
    depth_intrinsics: rs.intrinsics
    depth_to_color_extrinsics: rs.extrinsics
    decimation_filter: rs.decimation_filter = None
    spatial_filter: rs.spatial_filter = None
    temporal_filter: rs.temporal_filter = None

def realsense_init(width = 1280, height = 720, fps = 15, enable_imu = False, 
                   enable_decimation = False, enable_spatial = False, enable_temporal = False):
    """
    Initialize RealSense pipeline with specified parameters and optional depth filters.

    Args:
        width (int): Frame width. Defaults to 1280.
        height (int): Frame height. Defaults to 720.
        fps (int): Frame rate. Defaults to 15.
        enable_imu (bool): Whether to enable accelerometer and gyroscope streams. Defaults to False.
        enable_decimation (bool): Whether to enable decimation filter for depth. Defaults to False.
        enable_spatial (bool): Whether to enable spatial filter for depth. Defaults to False.
        enable_temporal (bool): Whether to enable temporal filter for depth. Defaults to False.

    Returns:
        RealSenseConfig: A dataclass containing all pipeline configuration, intrinsics, and optional filters.
    """

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    if enable_imu:
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        raise RuntimeError(f"Failed to start pipeline: {e}")

    # Warm-up
    for _ in range(5):
        pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Optional tuning
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()
    depth_to_color_extrinsics = depth_profile.get_extrinsics_to(color_profile)

    # Initialize filters if enabled
    decimation_filter = None
    spatial_filter = None
    temporal_filter = None

    if enable_decimation:
        decimation_filter = rs.decimation_filter()
        decimation_filter.set_option(rs.option.filter_magnitude, 2)

    if enable_spatial:
        spatial_filter = rs.spatial_filter()
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

    if enable_temporal:
        temporal_filter = rs.temporal_filter()
        temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

    return RealSenseConfig(
        pipeline=pipeline,
        align=align,
        depth_scale=depth_scale,
        color_intrinsics=color_intrinsics,
        depth_intrinsics=depth_intrinsics,
        depth_to_color_extrinsics=depth_to_color_extrinsics,
        decimation_filter=decimation_filter,
        spatial_filter=spatial_filter,
        temporal_filter=temporal_filter
    )




def realsense_get_frame(config):
    """
    Get aligned color and depth frames from the RealSense camera.
    
    Applies enabled depth filters if they were initialized.

    Args:
        config (RealSenseConfig): RealSense configuration containing pipeline, alignment, and optional filters.

    Returns:
        tuple: A tuple containing:
            - color_frame (rs.frame or None): BGR8 color frame of shape (H, W, 3) as uint8, or None if frames are not available.
            - depth_frame (rs.frame or None): Depth frame of shape (H, W) as uint16, or None if frames are not available.
    """

    frames = config.pipeline.wait_for_frames(timeout_ms=1000)
    aligned_frames = config.align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not color_frame or not depth_frame:
        return None, None

    # Apply depth filters if enabled
    if config.decimation_filter:
        depth_frame = config.decimation_filter.process(depth_frame)
    
    if config.spatial_filter:
        depth_frame = config.spatial_filter.process(depth_frame)
    
    if config.temporal_filter:
        depth_frame = config.temporal_filter.process(depth_frame)
    
    return color_frame, depth_frame