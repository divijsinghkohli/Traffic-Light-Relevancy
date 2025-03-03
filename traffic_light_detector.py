import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (pre-trained on the COCO dataset)
model = YOLO("yolov8s.pt")  

# Traffic light class ID in the COCO dataset
TRAFFIC_LIGHT_CLASS_ID = 9

def is_within_roi(x1, y1, x2, y2, roi):
    """
    Check if the detected traffic light is within the region of interest (ROI).
    roi is a polygon (array of points) defining the region.
    """
    # Define the center of the bounding box
    box_center = [(x1 + x2) // 2, (y1 + y2) // 2]
    
    # Convert the box to a tuple, I got this from StackOverflow: https://stackoverflow.com/questions/13786088/determine-if-a-point-is-inside-or-outside-of-a-shape-with-opencv
    box_center = tuple(box_center)
    
    # Check if the center point is inside the polygon (ROI)
    return cv2.pointPolygonTest(roi, box_center, False) >= 0

def is_traffic_light_relevant(light_bbox, lane_region, frame_shape):
    """
    Determine if a traffic light is relevant based on its position relative to the lane.
    """
    x1, y1, x2, y2 = light_bbox # get the coordinates from the bounding box
    height, width = frame_shape[:2] # get the height and the width
    
    # Calculate the center of the traffic light
    light_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    # Calculate the vertical position ratio (0 at top, 1 at bottom) => gives us the "severity" of the location of the light
    vertical_ratio = y1 / height
    
    # Define stricter position checks
    if vertical_ratio > 0.7:  # Ignore lights in bottom 30% of frame
        return False
    
    if vertical_ratio < 0.1:  # Ignore lights in top 10% of frame
        return False
    
    # Calculate lane center at the light's vertical position
    lane_points = lane_region[0]
    top_center = (lane_points[1][0] + lane_points[2][0]) // 2
    bottom_center = (lane_points[0][0] + lane_points[3][0]) // 2
    
    # Interpolate lane center at light's vertical position
    lane_center_at_light = bottom_center + (top_center - bottom_center) * (1 - vertical_ratio)
    
    # Calculate horizontal distance from lane center
    distance_from_center = abs(light_center[0] - lane_center_at_light)
    
    # Make the allowed distance proportional to the height (stricter at bottom, more forgivin at top since this would mean the light is further away)
    max_allowed_distance = width * (0.15 + 0.15 * (1 - vertical_ratio))
    
    # Check if light is within allowed distance
    if distance_from_center > max_allowed_distance:
        return False
    
    # Check if the light is within or very close to the lane region
    distance_to_region = cv2.pointPolygonTest(lane_region, light_center, True)
    if distance_to_region < -30:  # Only allow lights very close to the region
        return False
    
    # For debugging
    #print(f"Light detected at {light_center}, distance from center: {distance_from_center:.1f}px, max allowed: {max_allowed_distance:.1f}px")
    
    return True

def detect_relevant_traffic_lights(frame, lane_region):
    """
    Detect and filter relevant traffic lights based on lane position.
    """
    # Resize frame for better detection of small objects
    height, width = frame.shape[:2]
    scale = 1.5
    frame_resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
    
    # Perform YOLO detection on resized frame with verbose=False
    results = model(frame_resized, verbose=False)
    
    relevant_lights = []
    
    # Draw lane region first (so it's under the traffic light boxes)
    cv2.polylines(frame, [lane_region], True, (0, 255, 255), 2)
    
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(coord / scale) for coord in coords]
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            if class_id == TRAFFIC_LIGHT_CLASS_ID and confidence > 0.25:  # Slightly increased confidence threshold
                if is_traffic_light_relevant((x1, y1, x2, y2), lane_region, frame.shape):
                    relevant_lights.append((confidence, (x1, y1, x2, y2)))
                else:
                    # Draw rejected lights in red (for debugging)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    # Draw relevant lights
    for confidence, bbox in relevant_lights:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Relevant {confidence:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    # Define the Region of Interest (ROI) as a polygon (adjust coordinates based on lane and vehicle position)
    roi = np.array([[400, 100], [1500, 100], [1500, 800], [400, 800]]) 

    # Video path (hardcoded at the moment but need to integrate with camera feed eventually)
    video_path = "/Users/divijkohli/Desktop/Clubs/Wisconsin Autonomous/traffic_light_relevance/data/v1.mp4"  
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect relevant traffic lights in the region of interest (ROI)
        frame = detect_relevant_traffic_lights(frame, roi)

        # Show the frame with detected traffic lights
        cv2.imshow('Relevant Traffic Light Detection', frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
