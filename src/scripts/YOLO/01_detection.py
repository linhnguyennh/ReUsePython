import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import threading
from ultralytics import YOLO
import time


# === Global shared frame storage ===
latest_frames = {"color": None, "depth": None}
frame_lock = threading.Lock()
running = True

# === Initialize RealSense Pipeline ===
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale:", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# === Frame capture thread ===
def frame_capture():
    global latest_frames
    while running:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()

        if not depth or not color:
            continue

        with frame_lock:
            latest_frames["depth"] = depth
            latest_frames["color"] = color

# === Start the capture thread ===
thread = threading.Thread(target=frame_capture, daemon=True)
thread.start()

# === Load YOLOv8 model ===
#model = YOLO(r"..\..\models\focus1\Focus1_YOLO11n_x1024_14112024_ncnn_model")
model = YOLO(r"..\..\models\focus1\Focus1_YOLO11n_x1024_14112024.pt")
#model.fuse()
#model.to("cuda" if torch.cuda.is_available() else "cpu")

try:
    # === Main loop for detection and visualization ===
    start_time = time.time()
    frame_count = 0
    
    while True:
        with frame_lock:
            depth_frame = latest_frames["depth"]
            color_frame = latest_frames["color"]

        if depth_frame is None or color_frame is None:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference
        #results = model.predict(source=color_image, verbose=False, conf=0.7)[0]

        results = model(color_image, conf=0.7, device='cpu')[0]
        # Compute FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
        else:
            fps = 0


        # Display FPS
        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Get depth at bounding box center
            depth = depth_frame.get_distance(cx, cy)

            # Deproject to 3D point
            intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)

            label = f"{model.names[cls]} {depth:.2f}m"
            print(f"[{model.names[cls]}] at ({cx}, {cy}) → {depth:.2f}m → 3D: {np.round(point_3d, 2)}")

            # Draw detections
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Show output
        cv2.imshow("YOLOv8 + RealSense", color_image)
        if cv2.waitKey(1) == 27:  # ESC key
            break

finally:
    running = False
    thread.join()
    pipeline.stop()
    cv2.destroyAllWindows()
