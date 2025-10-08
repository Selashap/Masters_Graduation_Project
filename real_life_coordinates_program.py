import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("model_4_colours.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

try:
    print("Press ESC to quit.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Get camera intrinsics
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf)
                cls = int(box.cls)

                # Calculate center
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Get depth at center
                depth = depth_frame.get_distance(cx, cy)
                if depth == 0:
                    continue  # Skip invalid readings

                # Convert pixel + depth to 3D coordinates (in meters)
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

                label = f"{model.names[cls]} ({X:.2f}, {Y:.2f}, {Z:.2f})m"
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(color_image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show the image
        cv2.imshow("YOLOv8 + RealSense (3D)", color_image)
        if cv2.waitKey(1) == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

