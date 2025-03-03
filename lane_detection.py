import cv2
import numpy as np

def region_of_interest(img, vertices):
    """
    Apply a mask to the image, keeping only the region of interest (ROI).
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def lane_detection(frame):
    """
    Detect lanes in the given frame using Hough Transform.
    Returns the processed frame, detected lines, and lane boundaries
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define the region of interest (ROI)
    height, width = edges.shape
    roi_vertices = np.array([[
        (0, height), 
        (width // 2 - 100, height // 2 + 50), 
        (width // 2 + 100, height // 2 + 50), 
        (width, height)
    ]], dtype=np.int32)
    
    # Mask the edges image to focus on the region of interest
    masked_edges = region_of_interest(edges, roi_vertices)

    # Use Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    # Separate left and right lane lines
    left_lines = []
    right_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            
            # Filter out horizontal lines and categorize based on slope
            if abs(slope) > 0.3:  # Minimum slope threshold
                if slope < 0:  # Left lane
                    left_lines.append(line)
                else:  # Right lane
                    right_lines.append(line)

    # Calculate average lane boundaries
    left_boundary = None
    right_boundary = None
    
    if left_lines:
        left_points = np.concatenate(left_lines)
        left_boundary = np.mean(left_points, axis=0).astype(int)
    
    if right_lines:
        right_points = np.concatenate(right_lines)
        right_boundary = np.mean(right_points, axis=0).astype(int)

    # Draw lane boundaries
    if left_boundary is not None:
        cv2.line(frame, (left_boundary[0], left_boundary[1]), 
                (left_boundary[2], left_boundary[3]), (255, 0, 0), 3)
    if right_boundary is not None:
        cv2.line(frame, (right_boundary[0], right_boundary[1]), 
                (right_boundary[2], right_boundary[3]), (255, 0, 0), 3)

    return frame, left_boundary, right_boundary

def calculate_lane_region(left_boundary, right_boundary, frame_shape):
    """
    Calculate the region that represents the current driving lane.
    """
    height, width = frame_shape[:2]
    
    if left_boundary is None or right_boundary is None:
        # Default lane region if boundaries not detected
        center_x = width // 2
        lane_width = 400  # Fixed lane width when no boundaries detected
        
        return np.array([[
            (center_x - lane_width//2, height),
            (center_x - lane_width//3, height // 4),  # Narrower at top
            (center_x + lane_width//3, height // 4),  # Narrower at top
            (center_x + lane_width//2, height)
        ]], dtype=np.int32)
    
    # Create a polygon around the detected lane with fixed margins
    base_margin = 75  # Fixed margin at the bottom
    top_margin = 50   # Fixed margin at the top
    
    # Calculate the center line of the lane
    center_bottom = (left_boundary[0] + right_boundary[0]) // 2
    center_top = (left_boundary[2] + right_boundary[2]) // 2
    
    lane_region = np.array([[
        (left_boundary[0] - base_margin, height),
        (center_top - top_margin * 2, height // 4),  # Use center point at top
        (center_top + top_margin * 2, height // 4),  # Use center point at top
        (right_boundary[0] + base_margin, height)
    ]], dtype=np.int32)
    
    return lane_region

def process_frame(frame):
    """
    Process each frame, detect lanes and calculate lane region.
    """
    # Perform lane detection
    processed_frame, left_boundary, right_boundary = lane_detection(frame)
    
    # Calculate lane region
    lane_region = calculate_lane_region(left_boundary, right_boundary, frame.shape)
    
    return processed_frame, lane_region
