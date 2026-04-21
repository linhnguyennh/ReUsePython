import cv2
import numpy as np
import sys
import os

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vision.realsense_frame import realsense_init, realsense_get_frame

def main():
    # Initialize RealSense camera
    print("Initializing RealSense camera...")
    config = realsense_init(width=640, height=480, fps=30,
                           enable_decimation=True, enable_spatial=True, enable_temporal=True)

    print("RealSense camera initialized successfully!")
    try:
        while True:
            # Get RealSense frames
            color_frame, depth_frame = realsense_get_frame(config)

            if color_frame is None:
                print("Failed to get color frame, skipping...")
                continue

            # Convert color frame to numpy array
            frame = np.asanyarray(color_frame.get_data())
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Blur to suppress specular noise first
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blurred, 30, 100)

            # Dilate to close gaps in the edge contour
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)

            # Find external contours and select only those within 30cm depth
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            if contours and depth_frame is not None:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_height, depth_width = depth_image.shape
                frame_height, frame_width = gray.shape
                depth_scale = config.depth_scale

                valid_contours = []
                for contour in contours:
                    if len(contour) < 5:
                        continue

                    x, y, w, h = cv2.boundingRect(contour)
                    center_x = x + w // 2
                    center_y = y + h // 2

                    if center_x < 0 or center_y < 0 or center_x >= frame_width or center_y >= frame_height:
                        continue

                    depth_x = int(center_x * depth_width / frame_width)
                    depth_y = int(center_y * depth_height / frame_height)
                    if depth_x < 0 or depth_y < 0 or depth_x >= depth_width or depth_y >= depth_height:
                        continue

                    depth_value = depth_image[depth_y, depth_x]
                    depth_meters = float(depth_value) * depth_scale
                    if 0.0 < depth_meters <= 0.30:
                        valid_contours.append(contour)

                if valid_contours:
                    largest = max(valid_contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(largest)
                    box = cv2.boxPoints(rect)
                    box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [box], (255,))

            edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Combine images for display
            top_row = np.hstack([frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])
            bottom_row = np.hstack([edges_color, mask_color])
            combined = np.vstack([top_row, bottom_row])

            # Add labels
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Gray", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Edges", (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Mask", (650, 510), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show
            cv2.imshow("RealSense Canny Edge and Largest Contour Mask", combined)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Stop the RealSense pipeline
        config.pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped and windows closed.")

if __name__ == "__main__":
    main()