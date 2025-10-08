import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
from ultralytics import YOLO
from pyniryo import NiryoRobot, PoseObject, JointsPosition
import math
from scipy.spatial.transform import Rotation as R


GRID_ROWS = 4
GRID_COLS = 4

#pose for observing the blocks.
OBSERVATION_POSE = PoseObject(
    x=0.01, y=-0.253, z=0.386,
    roll=-2.819, pitch=1.530, yaw=1.950
)

# grid four point calibration 
pixel_map_corners = {
    "top_left": None,
    "top_right": None,
    "bottom_left": None,
    "bottom_right": None
}

def find_closest_cell_pixel_space(detected_pixel, grid_pixel_centers):
    rows, cols, _ = grid_pixel_centers.shape
    all_centers = grid_pixel_centers.reshape(-1, 2)
    distances = np.linalg.norm(all_centers - detected_pixel, axis=1)
    closest_index = np.argmin(distances)
    closest_row = closest_index // cols
    closest_col = closest_index % cols
    return closest_row, closest_col

def print_grid_map(grid_map, class_names):
    print("\n--- Detected Grid Map ---")
    max_len = max(len(name) for name in class_names.values()) if class_names else 10
    for r in range(len(grid_map)):
        row_str = "|"
        for c in range(len(grid_map[r])):
            item = grid_map[r][c]
            cell_content = "Empty".center(max_len) if item is None else item.center(max_len)
            row_str += f" {cell_content} |"
        print(row_str)
    print("-------------------------\n")


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['click_pos'] = (x, y)
        print(f"    -> Pixel selected at ({x}, {y})")

def calibrate_pixel_map_4_corners(robot: NiryoRobot, pipeline):
    print("\n--- Starting 2D Pixel Calibration (4-Point) ---")
    
    print("[CALIBRATION] Moving to observation pose...")
    robot.move(OBSERVATION_POSE)
    time.sleep(3.5)
    
    pixel_map_corners["top_left"] = get_user_click(pipeline, "Click the CENTER of the TOP-LEFT block (0,0)")
    pixel_map_corners["top_right"] = get_user_click(pipeline, "Click the CENTER of the TOP-RIGHT block (0,3)")
    pixel_map_corners["bottom_left"] = get_user_click(pipeline, "Click the CENTER of the BOTTOM-LEFT block (3,0)")
    pixel_map_corners["bottom_right"] = get_user_click(pipeline, "Click the CENTER of the BOTTOM-RIGHT block (3,3)")
    
    print("--- Calibration Complete ---")
    return pixel_map_corners

def get_user_click(pipeline, instruction_text):
    mouse_param = {'click_pos': None}
    window_name = "Pixel Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, mouse_param)
    
    print(f"    -> INSTRUCTION: {instruction_text}")
    while mouse_param['click_pos'] is None:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        cv2.putText(image, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27: # Allow ESC to exit
            break
    
    cv2.destroyWindow(window_name)
    return mouse_param['click_pos']

def generate_grid_centers_from_4_corners(rows, cols, corners):
    #Calculates the pixel centers for all grid cells using a perspective transform.
    if any(v is None for v in corners.values()):
        raise ValueError("Calibration failed. Four corners were not selected. Please restart the script.")
        
    # Define the four source points
    src_points = np.float32([corners["top_left"], corners["top_right"], 
                             corners["bottom_left"], corners["bottom_right"]])
    
    dst_size = 500
    dst_points = np.float32([[0, 0], [dst_size, 0], [0, dst_size], [dst_size, dst_size]])
    
    # Calculate the perspective transform matrix from the ideal grid to the camera's view
    matrix = cv2.getPerspectiveTransform(dst_points, src_points)
    
    grid_pixels = np.zeros((rows, cols, 2), dtype=int)
    
    # Create the ideal grid points in the destination space
    ideal_points = []
    for r in range(rows):
        for c in range(cols):
            px = np.interp(c, [0, cols - 1], [0, dst_size])
            py = np.interp(r, [0, rows - 1], [0, dst_size])
            ideal_points.append([px, py])
    
    ideal_points = np.float32(ideal_points).reshape(-1, 1, 2)
    
    # Transform the ideal points to find real positions in the distorted image
    transformed_points = cv2.perspectiveTransform(ideal_points, matrix)
    
    grid_pixels = transformed_points.reshape(rows, cols, 2).astype(int)
            
    return grid_pixels

def scan_grid_in_pixel_space(robot: NiryoRobot, model: YOLO, pipeline, grid_pixel_centers):
    #Moves to the observation pose and maps detections using the pixel grid.
    rows, cols, _ = grid_pixel_centers.shape
    grid_map = [[None for _ in range(cols)] for _ in range(rows)]

    print("\n[INFO] Capturing final image for analysis...")
    for _ in range(30): 
        pipeline.wait_for_frames()
    
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("[ERROR] Could not capture final image.")
        return None
    
    color_image = np.asanyarray(color_frame.get_data())
    
    print("[INFO] Running YOLO detection...")
    results = model(color_image, conf=0.4, verbose=False)
    
    debug_image = color_image.copy()

    if not results or len(results[0].boxes) == 0:
        print("[INFO] No objects detected.")
        return grid_map

    for box in results[0].boxes:
        class_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        row, col = find_closest_cell_pixel_space((cx, cy), grid_pixel_centers)

        if grid_map[row][col] is None:
            grid_map[row][col] = class_name
            print(f"[MAPPED] Found '{class_name}' at Row: {row}, Col: {col}")
        
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(debug_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw the calculated pixel grid on the image
    for r in range(rows):
        for c in range(cols):
            px, py = grid_pixel_centers[r, c]
            cv2.circle(debug_image, (px, py), 5, (255, 0, 0), -1)
            cv2.putText(debug_image, f"{r},{c}", (px + 5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection with Pixel Grid", debug_image)
    print("\n[INFO] Displaying final detection view. Press any key to continue.")
    cv2.waitKey(0)

    return grid_map

if __name__ == "__main__":
    robot = None
    pipeline = None
    try:
        print("[INFO] Connecting to robot...")
        robot = NiryoRobot("10.10.10.10")
        robot.clear_collision_detected()
        robot.calibrate_auto()
        robot.update_tool()

        print("[INFO] Loading YOLO model...")
        model = YOLO("model_4_colours.pt")
        
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        corners = calibrate_pixel_map_4_corners(robot, pipeline)
        grid_pixel_centers = generate_grid_centers_from_4_corners(GRID_ROWS, GRID_COLS, corners)
        final_grid_map = scan_grid_in_pixel_space(robot, model, pipeline, grid_pixel_centers)

        if final_grid_map is not None:
            print_grid_map(final_grid_map, model.names)
            
            # saving the map as Json file
            try:
                with open("organised_grid_map.json", "w") as f:
                    json.dump(final_grid_map, f, indent=4)
                print("\n[SUCCESS] Grid map has been saved to organised_grid_map.json")
            except Exception as e:
                print(f"\n[ERROR] Failed to save grid map to file: {e}")

        print("[INFO] Program finished. Returning to home position.")
        home_joints = JointsPosition(0.0, 0.0, 0.0, 0.0, -1.57, 0.0)
        robot.move(home_joints)

    except Exception as e:
        print(f"\n[FATAL ERROR] An exception occurred: {e}")
    finally:
        if pipeline:
            pipeline.stop()
        if robot:
            robot.close_connection()
        cv2.destroyAllWindows()

