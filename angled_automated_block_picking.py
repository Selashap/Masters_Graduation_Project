import json
import time
from pyniryo import NiryoRobot, PoseObject, JointsPosition
import math
import numpy as np
import cv2
import pyrealsense2 as rs

# Heights for picking and moving.
PICK_HOVER_HEIGHT = 0.22
SAFE_TRANSIT_HEIGHT = 0.20

DROP_ZONE_POSE = PoseObject(
    x=0.198, y=0.186, z=0.308,
    roll=2.928, pitch=1.450, yaw=-2.634
)

BASE_PICK_ORIENTATION = {"pitch": math.pi / 2, "yaw": 0.0}

def load_object_list(filename="object_list.json"):
    #Loads the list of detected objects from the JSON file.
    try:
        with open(filename, "r") as f:
            object_list = json.load(f)
        print(f"[INFO] Successfully loaded object list from {filename}")
        return object_list
    except FileNotFoundError:
        print(f"[ERROR] The file '{filename}' was not found. Please run the scanning script first.")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading the object list: {e}")
        return None

def save_object_list(object_list, filename="object_list.json"):
    #Saves the updated object list back to the JSON file.
    try:
        with open(filename, "w") as f:
            json.dump(object_list, f, indent=4)
        print(f"[INFO] Successfully saved updated object list to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save the updated list: {e}")

# Vision Function for Angle Detection
def get_block_angle_at_center(image):
    h, w, _ = image.shape
    image_center = np.array([w / 2, h / 2])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not contours:
        return None
    closest_contour = min(contours, key=lambda c: cv2.norm(cv2.minEnclosingCircle(c)[0] - image_center))
    rect = cv2.minAreaRect(closest_contour)
    angle = rect[2]
    width, height = rect[1]
    if width < height:
        angle += 90
    if angle > 90:
        angle -= 180
    return float(angle)


def pick_and_place_object(robot: NiryoRobot, position, pipeline):
    target_x = position['x']
    target_y = position['y']
    pick_target_height = position['z']

    try:
        hover_pose = PoseObject(x=target_x, y=target_y, z=PICK_HOVER_HEIGHT, roll=0.0, **BASE_PICK_ORIENTATION)
        print(f"  -> Moving to hover over block at ({target_x:.3f}, {target_y:.3f})")
        robot.move(hover_pose)
        time.sleep(2) 

        print("  -> Analyzing block orientation...")
        for _ in range(15):
            frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("  -> [WARNING] Could not get camera frame. Aborting this pick.")
            return False
            
        image = np.asanyarray(color_frame.get_data())
        
        detected_angle_deg = get_block_angle_at_center(image)
        
        if detected_angle_deg is None:
            print("  -> [WARNING] Could not detect block angle clearly. Using neutral grip.")
            detected_angle_deg = 0.0

        print(f"  -> Detected angle: {detected_angle_deg:.2f} degrees.")
        
        target_roll_rad = math.radians(detected_angle_deg)

        hover_pose.roll = target_roll_rad
        pick_pose = PoseObject(x=target_x, y=target_y, z=pick_target_height, roll=target_roll_rad, **BASE_PICK_ORIENTATION)
        
        print("  -> Aligning gripper...")
        robot.move(hover_pose)
        time.sleep(1.5)

        robot.open_gripper()
        time.sleep(0.5)
        print(f"  -> Descending to pick block at height {pick_target_height:.3f}m...")
        robot.move(pick_pose)
        time.sleep(1.5)
        robot.close_gripper()
        time.sleep(1)
        print("  -> Lifting block...")
        robot.move(hover_pose)
        time.sleep(1)
        
        print("  -> Moving to drop-off zone...")
        robot.move(DROP_ZONE_POSE)
        time.sleep(2)
        
        time.sleep(1)

        safe_drop_pose = PoseObject(
            x=DROP_ZONE_POSE.x, y=DROP_ZONE_POSE.y, z=SAFE_TRANSIT_HEIGHT,
            roll=DROP_ZONE_POSE.roll, pitch=DROP_ZONE_POSE.pitch, yaw=DROP_ZONE_POSE.yaw
        )
        robot.move(safe_drop_pose)
        time.sleep(1)
        
        robot.open_gripper()

        robot.move(DROP_ZONE_POSE)
        time.sleep(2)

        print("  -> Pick and place sequence complete for this block.")
        return True

    except Exception as e:
        print(f"[ERROR] An error occurred during the pick and place sequence: {e}")
        robot.set_learning_mode(True) 
        return False

if __name__ == "__main__":
    robot = None
    pipeline = None
    try:
        object_list = load_object_list()
        if object_list is None:
            exit()

        target_color = input("\nEnter the color of the blocks you want to pick (e.g., 'red', 'blue'): ").strip().lower()

        target_objects = [obj for obj in object_list if obj.get("color", "").lower() == target_color]

        if not target_objects:
            print(f"\n[INFO] No blocks with the color '{target_color}' were found in the list.")
            exit()

        print(f"\n[INFO] Found {len(target_objects)} '{target_color}' block(s).")
        
        print("\n[INFO] Connecting to robot and starting camera...")
        robot = NiryoRobot("10.10.10.10")
        robot.calibrate_auto()
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        time.sleep(1)
        
        successfully_picked_objects = []
        for i, obj in enumerate(target_objects):
            print(f"\n--- Processing block {i+1}/{len(target_objects)} ---")
            
            success = pick_and_place_object(robot, obj['position'], pipeline)
            
            if success:
                successfully_picked_objects.append(obj)
            else:
                print("[FATAL] Halting program due to a pick-and-place error.")
                break

        if successfully_picked_objects:
            print(f"\n[INFO] Updating object list for {len(successfully_picked_objects)} picked block(s)...")
            remaining_objects = [obj for obj in object_list if obj not in successfully_picked_objects]
            save_object_list(remaining_objects)

        print("\n[INFO] Program finished. Returning to home position.")
        home_joints = JointsPosition(0.0, 0.0, 0.0, 0.0, -1.57, 0.0)
        robot.move(home_joints)
        robot.close_gripper()
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
    finally:
        if pipeline:
            pipeline.stop()
        if robot:
            robot.close_connection()
            print("[INFO] Disconnected from robot.")