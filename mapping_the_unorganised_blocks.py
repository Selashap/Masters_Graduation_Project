import json
import time
import math
import numpy as np
import cv2
from pyniryo import NiryoRobot, PoseObject, JointsPosition
from ultralytics import YOLO
import pyrealsense2 as rs
import threading


WORKSPACE_CORNERS = {
    "top_left":     np.array([0.121, -0.438]), 
    "top_right":    np.array([-0.080, -0.423]), 
    "bottom_left":  np.array([0.134, -0.214]), 
    "bottom_right": np.array([-0.05, -0.195])
}

BLOCK_SURFACE_Z = 0.085

# Observation pose
OBSERVATION_POSE = PoseObject(
    x=0.01, y=-0.253, z=0.386,
    roll=-2.819, pitch=1.530, yaw=1.950
)

_mouse_click_pos = None

# --- Helper Functions ---

def mouse_callback(event, x, y, flags, param):
    global _mouse_click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click_pos = (x, y)
        print(f"    -> Pixel selected at ({x}, {y})")

def get_user_click(pipeline, instruction_text):
    global _mouse_click_pos
    _mouse_click_pos = None
    window_name = "Workspace Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print(f"    -> INSTRUCTION: {instruction_text}")
    while _mouse_click_pos is None:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: continue
        
        image = np.asanyarray(color_frame.get_data())
        cv2.putText(image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, image)
        
        if cv2.waitKey(1) & 0xFF == 27: # Allow ESC to abort
            cv2.destroyAllWindows()
            return None
            
    cv2.destroyAllWindows()
    return _mouse_click_pos

# --- Main Scanning Function ---

def scan_and_map_workspace(robot: NiryoRobot, model: YOLO):
    pipeline = None
    detected_objects = []

    try:
        #Start camera
        print("[INFO] Starting RealSense camera for calibration...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) 
        pipeline.start(config)
        time.sleep(1) 

        #Move to observation pose
        print("[INFO] Moving to observation pose for calibration...")
        robot.move(OBSERVATION_POSE)
        time.sleep(3.5)

        # Perform 4-Point Interactive Calibration
        print("\n--- Starting Workspace Calibration ---")
        print("Please click the four corners of your physical workspace in the camera window.")
        
        pixel_corners = {
            "top_left":     get_user_click(pipeline, "Click the TOP-LEFT corner of the workspace"),
            "top_right":    get_user_click(pipeline, "Click the TOP-RIGHT corner of the workspace"),
            "bottom_left":  get_user_click(pipeline, "Click the BOTTOM-LEFT corner of the workspace"),
            "bottom_right": get_user_click(pipeline, "Click the BOTTOM-RIGHT corner of the workspace")
        }

        if any(v is None for v in pixel_corners.values()):
            print("[ERROR] Calibration aborted by user. Exiting.")
            return

        print("--- Calibration Complete ---")
        
        #Calculate the Perspective Transform Matrix
        pixel_src = np.float32(list(pixel_corners.values()))
        world_dst = np.float32(list(WORKSPACE_CORNERS.values()))
        
        perspective_matrix = cv2.getPerspectiveTransform(pixel_src, world_dst)
        print("[INFO] Perspective transformation matrix calculated.")

        print("\n[INFO] Capturing final image for object detection...")
        for _ in range(30):
            pipeline.wait_for_frames()
        
        color_frame = pipeline.wait_for_frames().get_color_frame()
        if not color_frame:
            print("[ERROR] Could not get final frame from camera.")
            return
            
        color_image = np.asanyarray(color_frame.get_data())

        print("[INFO] Running object detection...")
        results = model(color_image, conf=0.5, verbose=False)
        debug_image = color_image.copy()

        if not results or len(results[0].boxes) == 0:
            print("[INFO] No objects were detected.")
            return

        print(f"[INFO] Found {len(results[0].boxes)} objects. Calculating positions...")
        
        detected_pixels = []
        box_info = []
        for box in results[0].boxes:
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detected_pixels.append([[float(cx), float(cy)]])
            box_info.append({"name": class_name, "box": (x1, y1, x2, y2)})

        detected_pixels_np = np.array(detected_pixels, dtype=np.float32)
        transformed_points = cv2.perspectiveTransform(detected_pixels_np, perspective_matrix)

        for i, point in enumerate(transformed_points):
            real_x, real_y = point[0]
            
            #Convert NumPy float32 to standard Python float
            obj_data = {
                "color": box_info[i]["name"],
                "position": {
                    "x": round(float(real_x), 4),
                    "y": round(float(real_y), 4),
                    "z": BLOCK_SURFACE_Z
                }
            }
            detected_objects.append(obj_data)
            print(f"  -> Detected '{obj_data['color']}' at X={obj_data['position']['x']:.3f}, Y={obj_data['position']['y']:.3f}, Z={obj_data['position']['z']:.3f}")

            x1, y1, x2, y2 = box_info[i]["box"]
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{obj_data['color']} (X:{real_x:.2f}, Y:{real_y:.2f})"
            cv2.putText(debug_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Detection Debug View", debug_image)
        print("\n[INFO] Displaying detection view. Press any key to continue...")
        cv2.waitKey(0)

        #Save the results to a JSON file
        try:
            with open("angled_object_list.json", "w") as f:
                json.dump(detected_objects, f, indent=4)
            print("\n[SUCCESS] Object list saved to 'angled_object_list.json'")
        except Exception as e:
            print(f"\n[ERROR] Could not save the file: {e}")

    finally:
        if pipeline:
            pipeline.stop()
            print("[INFO] Camera stopped.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    robot = None
    try:
        print("[INFO] Loading YOLO model...")
        model = YOLO("model_4_colours.pt")

        print("[INFO] Connecting to robot...")
        robot = NiryoRobot("10.10.10.10")
        robot.calibrate_auto()
        
        scan_and_map_workspace(robot, model)

        print("\n[INFO] Program finished. Returning to home position.")
        home_joints = JointsPosition(0.0, 0.0, 0.0, 0.0, -1.57, 0.0)
        robot.move(home_joints)

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
    finally:
        if robot:
            robot.close_connection()
            print("[INFO] Disconnected from robot.")
        cv2.destroyAllWindows()

