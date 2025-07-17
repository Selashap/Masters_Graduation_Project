import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO


model = YOLO("model_4_blocks.pt")

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #stream for depth len on the camera
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #stream for the coloured image.

pipeline.start(config)

# Align depth to colour
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get intrinsics for deprojection
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        # Convert images to numpy
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model.predict(source=color_image, imgsz=640, conf=0.5, verbose=False)
        annotated = results[0].plot()

        if len(results[0].boxes) > 0:
            print("Object detected!")

            # Get the first detection
            xyxy = results[0].boxes[0].xyxy[0].cpu().numpy()
            x_pixel = int((xyxy[0] + xyxy[2]) / 2)
            y_pixel = int((xyxy[1] + xyxy[3]) / 2)

            # Get depth at pixel (in mm)
            depth = depth_image[y_pixel, x_pixel]

            # Convert pixel + depth to real-world coordinates
            x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, [x_pixel, y_pixel], depth)
            x, y, z = x / 1000, y / 1000, z / 1000  # Convert mm to meters

            print(f"Object center in image: ({x_pixel}, {y_pixel})")
            print(f"Real-world coordinates: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")

            # Draw data on image
            cv2.circle(annotated, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
            cv2.putText(annotated, f"({x:.2f}, {y:.2f}, {z:.2f}) m", (x_pixel+10, y_pixel),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO + RealSense (3D)", annotated)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
