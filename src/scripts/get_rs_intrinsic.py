import sys
import os
import pyrealsense2 as rs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame

config = realsense_init(width=640, height=480, fps=30)


profile = config.pipeline.get_active_profile()

color_intrinsics = profile.get_stream(rs.stream.color)\
                          .as_video_stream_profile()\
                          .get_intrinsics()

print(color_intrinsics)