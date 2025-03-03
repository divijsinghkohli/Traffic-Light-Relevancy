import cv2
import numpy as np
from lane_detection import process_frame
from traffic_light_detector import detect_relevant_traffic_lights

def main():
    # Hardcode the video path here
    video_path = "/Users/divijkohli/Desktop/Clubs/Wisconsin Autonomous/traffic_light_relevance/data/v1.mp4"  # Replace with your video file path

    # Define Region of Interest (ROI) if needed
    roi = np.array([[400, 100], [1500, 100], [1500, 800], [400, 800]])  # Example rectangular ROI (adjust as needed)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Process the video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame to get lane region
        processed_frame, lane_region = process_frame(frame)
        
        # Detect relevant traffic lights using lane region
        processed_frame = detect_relevant_traffic_lights(processed_frame, lane_region)
        
        # Show the processed frame
        cv2.imshow('Lane and Traffic Light Detection', processed_frame)

        # Exit the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
