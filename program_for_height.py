import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from pyniryo import NiryoRobot, JointsPosition

robot = NiryoRobot('10.10.10.10')
robot.clear_collision_detected()

# Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Align depth to colour
align_to = rs.stream.color
align = rs.align(align_to)

pipeline.start(config)

target_positions = [
    {'name': 'Target A', 'joints': JointsPosition(1.509, 0.339, 0.007, -0.020, -1.915, 0.078)},
    {'name': 'Target B', 'joints': JointsPosition(1.600, 0.313, -0.102, -0.077, -1.795, 0.078)},
    {'name': 'Target C', 'joints': JointsPosition(1.590, 0.390, -0.351, -0.072, -1.675, 0.078)},
    {'name': 'Target D', 'joints': JointsPosition(1.590, 0.409, -0.658, -0.109, -1.375, 0.078)},
    {'name': 'Target E', 'joints': JointsPosition(1.565, 0.359, -0.766, -0.084, -1.220, 0.078)},
]

depth_results = []

def move_robot_to(target):
    print(f"[ACTION] Moving to {target['name']}")
    robot.move(target['joints'])
    time.sleep(3) 

def get_center_depth(depth_frame):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    x, y = w // 2, h // 2
    depth = depth_frame.get_distance(x, y)
    if depth == 0 or depth > 2.0:
        return None
    return depth

try:
    print("[INFO] Starting measurements...")

    for target in target_positions:
        move_robot_to(target)

        # Wait for stable frame
        for _ in range(10):
            frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()

        depth = get_center_depth(depth_frame)
        if depth is None:
            print(f"[WARNING] No valid depth data at {target['name']}")
            depth = 0

        print(f"[RESULT] {target['name']}: Depth = {depth:.3f} meters")
        depth_results.append((target['name'], depth))

    # Plot the depth results
    names = [name for name, _ in depth_results]
    depths = [depth for _, depth in depth_results]

    plt.figure(figsize=(8, 5))
    plt.plot(names, depths, marker='o', linestyle='-', color='blue')
    plt.ylabel("Depth (m)")
    plt.title("Depth Measurement at Each Target")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

finally:
    print("[INFO] Stopping camera and disconnecting robot.")
    pipeline.stop()
    robot.close_connection()
