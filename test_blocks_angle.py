import cv2
import numpy as np
import pyrealsense2 as rs

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
        
        box_points = np.intp(cv2.boxPoints(rect))

        detection_data = {
            'angle': angle,
            'box': box_points
        }

        detections.append(detection_data)

    return detections

# --- Main Program ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
print("Starting RealSense camera stream... Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()
        
        all_detections = analyze_all_shapes(color_image)

        if all_detections:
            for detection in all_detections:
                angle = detection['angle']
                box = detection['box']
                
                # Draw the rotated bounding box for the current shape
                cv2.drawContours(display_image, [box], 0, (0, 255, 0), 2)

                text_position = (box[1][0], box[1][1] - 10)
                cv2.putText(
                    display_image,
                    f"Angle: {angle:.2f}",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("RealSense Live Angle Detection", display_image)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

finally:
    print("Stopping stream.")
    pipeline.stop()
    cv2.destroyAllWindows()    

