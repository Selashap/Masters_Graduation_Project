import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
from pyniryo import NiryoRobot, JointsPosition

robot = NiryoRobot('10.10.10.10')
print("[INFO] Moving to scan start position...")
robot.move(JointsPosition(0.024, 0.452, -0.452, -0.047, -1.500, 0.078))
time.sleep(2)

model = YOLO("new_model_4_blocks.pt")

# === RealSense setup ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# === Get intrinsics ===
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("[INFO] Starting scan...")

try:
    for angle_deg in range(0, 360, 10):
        angle_rad = angle_deg * np.pi / 180.0
        print(f"[SCANNING] Rotating base to {angle_deg}°")

        # Move robot base joint
        current_joints = robot.get_joints()
        new_joints = list(current_joints)
        new_joints[0] = float(angle_rad)
        robot.move(JointsPosition(*new_joints))
        time.sleep(0.2)

        # Capture aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy image
        color_image = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        results = model(color_image, imgsz=640, conf=0.5, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                conf = float(box.conf)
                cls = int(box.cls)

                # Camera image center
                center_x = color_image.shape[1] // 2
                center_y = color_image.shape[0] // 2
                dx = abs(cx - center_x)
                dy = abs(cy - center_y)

                print(f"[CENTER CHECK] Offset dx={dx}, dy={dy}")

                # If shape is close to center stop scan
                if dx < 90 and dy < 90:
                    depth = depth_frame.get_distance(cx, cy)
                    if depth == 0:
                        print(f"[WARNING] No valid depth at center ({cx}, {cy})")
                        continue

                    X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

                    print(f"[DETECTED] {model.names[cls]} at pixel ({cx}, {cy})")
                    print(f"[3D COORDS] X={X:.3f}, Y={Y:.3f}, Z={Z:.3f} m")

                    # Visualise
                    label = f"{model.names[cls]} ({X:.2f}, {Y:.2f}, {Z:.2f})m"
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(color_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Stop scanning
                    print("[INFO] Target shape is centered. Stopping.")
                    raise KeyboardInterrupt

except KeyboardInterrupt:
    print("[INFO] Scanning stopped.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    robot.close_connection()
    print("[INFO] Robot disconnected.")
