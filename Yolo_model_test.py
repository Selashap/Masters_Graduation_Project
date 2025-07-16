import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("model_4_blocks.pt")

# Setting up the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #coloured stream
pipeline.start(config)

try:
    while True:
        # Get colour frame from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        results = model.predict(source=color_image, imgsz=640, conf=0.5, verbose=False)

        annotated_frame = results[0].plot()

        # Show image
        cv2.imshow("YOLOv8 + RealSense", annotated_frame)

        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


