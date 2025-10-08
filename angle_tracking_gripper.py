import time
from pyniryo import NiryoRobot, JointsPosition, PoseObject
import math
import cv2
import numpy as np
import pyrealsense2 as rs


ROBOT_IP = "10.10.10.10"
YAW_CALIBRATION_OFFSET = math.radians(0) 
MOVEMENT_THRESHOLD_DEG = 3.0

def analyze_all_shapes(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
    )
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for contour in contours:
        # Filter out contours that are too small to be a block
        if cv2.contourArea(contour) < 500:
            continue

        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        width, height = rect[1]

        if width < height:
            angle += 90
        if angle > 90:
            angle -= 180
        
        box_points = np.intp(cv2.boxPoints(rect))
        
        detections.append({
            'angle': float(angle),
            'box': box_points,
            'area': cv2.contourArea(contour)
        })
        
    return detections

# --- Main Robot Control Program ---
def run_integrated_tracking():
    try:
        robot = NiryoRobot(ROBOT_IP)
        robot.calibrate_auto()
        robot.set_arm_max_velocity(80)
    except Exception as e:
        print(f"Error connecting to robot: {e}")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    print("Camera and Robot Initialized.")

    print(f"Moving to the starting pose...")
    robot.clear_collision_detected()
    robot.move(JointsPosition(-1.529, 0.228, -0.375, 0.000, -1.357, 0.003))
    print("Starting pose reached. Ready to track.")

    last_commanded_angle = None

    try:
        # --- Main Loop ---
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            display_image = image.copy()

            all_detections = analyze_all_shapes(image)

            if all_detections:
                # --- Find the largest detected shape to be the target ---
                largest_detection = max(all_detections, key=lambda d: d['area'])
                current_angle_deg = largest_detection['angle']

                should_move = False
                if last_commanded_angle is None:
                    should_move = True
                else:
                    angle_difference = abs(current_angle_deg - last_commanded_angle)
                    if angle_difference > MOVEMENT_THRESHOLD_DEG:
                        should_move = True

                if should_move:
                    print(f"Largest block detected at: {current_angle_deg:.1f}°. Moving gripper.")
                    current_pose = robot.get_pose()
                    
                    target_pose = PoseObject(
                        x=current_pose.x, y=current_pose.y, z=current_pose.z,
                        roll=math.radians(current_angle_deg) + YAW_CALIBRATION_OFFSET,
                        pitch=current_pose.pitch, yaw=current_pose.yaw
                    )
                    
                    robot.clear_collision_detected()
                    robot.move(target_pose)
                    last_commanded_angle = current_angle_deg


                for detection in all_detections:
                    box = detection['box']
                    angle = detection['angle']
                    cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)
                    text_position = (box[1][0], box[1][1] - 10)
                    cv2.putText(display_image, f"Angle: {angle:.1f}",
                                text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Live View", display_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping program.")

    finally:
        print("Returning to home position.")
        home_pose = JointsPosition(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        robot.clear_collision_detected()
        robot.move(home_pose)
        robot.close_connection()
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Disconnected from robot.")

if __name__ == '__main__':
    run_integrated_tracking()




 