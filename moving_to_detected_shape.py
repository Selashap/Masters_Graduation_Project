import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO
from pyniryo import NiryoRobot, JointsPosition, PoseObject
import math
from scipy.spatial.transform import Rotation as R

X_OFFSET = -0.03
Y_OFFSET = 0.05
Z_OFFSET = 0.0

# Transform matrix from flange to camera frame (translation in meters)
FLANGE_TO_CAM_TRANSFORM = np.array([
    [-0.192962029,  0.265886842,  0.944494491, -0.0644849896],
    [-0.0165481335, 0.961566339, -0.274073593, -0.00642201071],
    [-0.981066672, -0.0685154176, -0.181145861,  0.0167684892],
    [0, 0, 0, 1]
])

CAMERA_MATRIX = np.array([
    [621.26335487, 0., 316.0555719],
    [0., 618.08176042, 249.83024965],
    [0., 0., 1.]
])
DIST_COEFFS = np.array([[0.11752553, 0.0824063, 0.00995174, 0.00314283, -1.33768341]])

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def pose_to_matrix(pose: PoseObject):
    rotation = R.from_euler('xyz', [pose.roll, pose.pitch, pose.yaw])
    rot_matrix = rotation.as_matrix()
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot_matrix
    transform_matrix[:3, 3] = [pose.x, pose.y, pose.z]
    return transform_matrix

def transform_point(point_in_camera, transform_matrix):
    p_cam_homogeneous = np.append(point_in_camera, 1)
    p_base_homogeneous = transform_matrix @ p_cam_homogeneous
    return p_base_homogeneous[0], p_base_homogeneous[1], p_base_homogeneous[2]

# TCP positions (x, y, z) guidelines in detection space
known_positions = np.array([
    [0.132, 0.147, 0.1],
    [0.015, 0.174, 0.1],
    [-0.118, 0.130, 0.1],
    [-0.173, 0.048, 0.1]
])

def is_near_known_position(current_pos, known_positions, threshold=0.1):  # slightly bigger tolerance
    distances = np.linalg.norm(known_positions - current_pos, axis=1)
    min_dist = np.min(distances)
    if min_dist <= threshold:
        return known_positions[np.argmin(distances)]
    else:
        return None

# --- Main Program ---
robot = NiryoRobot('10.10.10.10')
robot.clear_collision_detected()
robot.move(JointsPosition(0.075, 0.172, -0.195, -0.041, -1.611, 0.081))
time.sleep(2)

model = YOLO("model_4_blocks.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# Warm up camera
for _ in range(10):
    pipeline.wait_for_frames()

print("[INFO] Starting scan...")

try:
    for angle_deg in range(0, 360, 10):
        angle_rad = math.radians(angle_deg)
        print(f"[SCANNING] Rotating base to {angle_deg}°")

        current_joints = robot.get_joints()
        new_joints = list(current_joints)
        new_joints[0] = angle_rad
        robot.move(JointsPosition(*new_joints))
        time.sleep(3)  # allow vibrations to settle

        # Capture multiple frames and use the last for stability
        for _ in range(5):
            frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            print("[WARNING] Frames not available, skipping this angle.")
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Run detection with slightly lower confidence
        results = model(color_image, imgsz=640, conf=0.25, verbose=False)

        if not results or len(results[0].boxes) == 0:
            print("[INFO] No detection at this angle.")
            continue

        # Debug: print all detections
        for i, box in enumerate(results[0].boxes):
            print(f"  Det {i}: class={int(box.cls)}, conf={float(box.conf):.2f}")

        # Pick best detection
        boxes = results[0].boxes
        best_box_idx = boxes.conf.argmax()
        best_detection = boxes[best_box_idx]

        x1, y1, x2, y2 = map(int, best_detection.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf = best_detection.conf.item()
        print(f"[DETECTED] Confidence: {conf:.2f}, Center pixel: ({cx},{cy})")

        depth = depth_frame.get_distance(cx, cy)
        if depth == 0:
            print("[WARNING] No depth data at detection pixel, skipping.")
            continue  # FIXED indentation bug

        # Draw on image for debugging
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
        cv2.imshow("Detection Debug", color_image)
        cv2.waitKey(1)

        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        point_in_camera = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth)

        # Get current robot flange pose as matrix
        current_pose_obj = robot.get_pose()
        T_base_flange = pose_to_matrix(current_pose_obj)

        # Compute transformation base to camera
        T_base_cam = T_base_flange @ FLANGE_TO_CAM_TRANSFORM

        # Transform detected point to robot base frame
        x_base, y_base, z_base = transform_point(point_in_camera, T_base_cam)
        print(f"[DETECTED] Original base frame (x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f})")

        x_detected = x_base + X_OFFSET
        y_detected = y_base + Y_OFFSET
        z_detected = z_base + Z_OFFSET
        print(f"[ADJUSTED] Corrected base frame (x={x_detected:.3f}, y={y_detected:.3f}, z={z_detected:.3f})")

        detected_pos = np.array([x_detected, y_detected, z_detected])
        close_pos = is_near_known_position(detected_pos, known_positions)

        if close_pos is not None:
            print(f"[INFO] Detected shape near known position {close_pos}, descending to pick...")

            TABLE_Z = 0.078
            OBJECT_HEIGHT = 0.03
            HOVER_OFFSET = 0.05
            z_touch = TABLE_Z + OBJECT_HEIGHT
            z_hover = z_touch + HOVER_OFFSET
            z_touch = clamp(z_touch, 0.05, 0.3)
            z_hover = clamp(z_hover, 0.1, 0.35)

            pick_orientation = [-0.262, 1.523, 1.228]  # roll, pitch, yaw

            hover_pose = PoseObject(x=close_pos[0], y=close_pos[1], z=z_hover,
                                    roll=pick_orientation[0], pitch=pick_orientation[1], yaw=pick_orientation[2])
            touch_pose = PoseObject(x=close_pos[0], y=close_pos[1], z=z_touch,
                                    roll=pick_orientation[0], pitch=pick_orientation[1], yaw=pick_orientation[2])

            print("[INFO] Moving to hover pose...")
            robot.move(hover_pose)
            time.sleep(2)

            print("[INFO] Moving to touch pose...")
            robot.move(touch_pose)
            time.sleep(2)

            print("[INFO] Moving back to hover pose...")
            robot.move(hover_pose)
            time.sleep(2)

            print("[INFO] Returning to home position...")
            robot.move(JointsPosition(0.075, 0.172, -0.195, -0.041, -1.611, 0.081))
            break
        else:
            print("[INFO] Detected shape not near any known position, continue scanning.")

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

except Exception as e:
    print(f"[ERROR] {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    robot.close_connection()
    print("[INFO] Robot disconnected.")
