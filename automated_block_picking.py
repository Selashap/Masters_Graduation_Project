import json
import time
from pyniryo import NiryoRobot, PoseObject, JointsPosition
import math
import numpy as np

# --- Configuration Constants ---

# Heights for picking and moving.
PICK_HOVER_HEIGHT = 0.15
# NOTE: The target height will now come directly from the JSON file.
SAFE_TRANSIT_HEIGHT = 0.20

# Your calibrated drop-off zone pose.
DROP_ZONE_POSE = PoseObject(
    x=0.198, y=0.186, z=0.308,
    roll=2.928, pitch=1.450, yaw=-2.634
)

# A standard, reliable orientation for picking up blocks from a flat surface.
PICK_ORIENTATION = {"roll": 2.993, "pitch": 1.430, "yaw": 1.506}

# --- Helper Functions (MODIFIED for object_list.json) ---

def load_object_list(filename="object_list.json"):
    """Loads the list of detected objects from the JSON file."""
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
    """Saves the updated object list back to the JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(object_list, f, indent=4)
        print(f"[INFO] Successfully saved updated object list to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save the updated list: {e}")

# --- Main Action Function (MODIFIED) ---

def pick_and_place_object(robot: NiryoRobot, position):
    """
    Executes the full sequence to pick a block at a specific 3D coordinate
    and move it to the drop zone.
    """
    target_x = position['x']
    target_y = position['y']
    # Use the accurate Z-height from the JSON file for the pick target
    pick_target_height = position['z']

    try:
        hover_pose = PoseObject(x=target_x, y=target_y, z=PICK_HOVER_HEIGHT, **PICK_ORIENTATION)
        print(f"  -> Moving to hover over block at ({target_x:.3f}, {target_y:.3f})")
        robot.move(hover_pose)
        time.sleep(1.5)

        robot.open_gripper()
        time.sleep(0.5)

        pick_pose = PoseObject(x=target_x, y=target_y, z=pick_target_height, **PICK_ORIENTATION)
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

        safe_drop_pose = PoseObject(
            x=DROP_ZONE_POSE.x, y=DROP_ZONE_POSE.y, z=SAFE_TRANSIT_HEIGHT,
            roll=DROP_ZONE_POSE.roll, pitch=DROP_ZONE_POSE.pitch, yaw=DROP_ZONE_POSE.yaw
        )
        robot.move(safe_drop_pose)
        time.sleep(1)
        
        robot.open_gripper()
        time.sleep(1)

        robot.move(DROP_ZONE_POSE)
        time.sleep(2)

        print("  -> Pick and place sequence complete for this block.")
        return True

    except Exception as e:
        print(f"[ERROR] An error occurred during the pick and place sequence: {e}")
        robot.set_learning_mode(True) 
        return False

# --- Main Program (MODIFIED for object_list.json) ---

if __name__ == "__main__":
    robot = None
    try:
        # 1. Load the list of available objects
        object_list = load_object_list()
        if object_list is None:
            exit()

        # 2. Get user input for which color to pick
        target_color = input("\nEnter the color of the blocks you want to pick (e.g., 'red', 'blue'): ").strip().lower()

        # 3. Find all objects of the target color in the list
        target_objects = []
        for obj in object_list:
            if obj.get("color", "").lower() == target_color:
                target_objects.append(obj)

        if not target_objects:
            print(f"\n[INFO] No blocks with the color '{target_color}' were found in the list.")
            exit()

        print(f"\n[INFO] Found {len(target_objects)} '{target_color}' block(s).")
        
        # 4. Connect to the robot
        print("\n[INFO] Connecting to robot...")
        robot = NiryoRobot("10.10.10.10")
        robot.calibrate_auto()
        
        successfully_picked_objects = []
        for i, obj in enumerate(target_objects):
            print(f"\n--- Processing block {i+1}/{len(target_objects)} ---")
            
            success = pick_and_place_object(robot, obj['position'])
            
            if success:
                # Keep track of objects we successfully picked
                successfully_picked_objects.append(obj)
            else:
                print("[FATAL] Halting program due to a pick-and-place error.")
                break # Stop if any sequence fails

        # 5. Update and save the object list only once at the end
        if successfully_picked_objects:
            print(f"\n[INFO] Updating object list for {len(successfully_picked_objects)} picked block(s)...")
            
            # Create a new list containing only the objects that were NOT picked
            remaining_objects = [obj for obj in object_list if obj not in successfully_picked_objects]
            
            save_object_list(remaining_objects)

        print("\n[INFO] Program finished. Returning to home position.")
        robot.move_joints([0, 0, 0, 0, -1.57, 0])

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
    finally:
        if robot:
            robot.close_connection()
            print("[INFO] Disconnected from robot.")