import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(image_path):
    """
    Read an image from the given path.
    
    Args:
    - image_path: Path to the image file.
    
    Returns:
    - image: The loaded image.
    """
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    """
    Preprocess the image for lane detection.
    
    Args:
    - image: Input image in BGR format.
    
    Returns:
    - frame: Preprocessed image.
    """
    # Convert the image from BGR to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply Gaussian blur to reduce noise and detail in the image
    frame = cv2.GaussianBlur(image_rgb, (5, 5), 0)
    return frame

def group_lines(lines):
    """
    Group lines based on similar slopes and close center points.
    
    Args:
    - lines: List of lines.
    
    Returns:
    - grouped_lines: Grouped lines.
    """
    if not lines:
        return []
    
    grouped_lines = []
    # Sort lines based on their slopes
    lines_sorted = sorted(lines, key=lambda x: x[0])
    current_group = [lines_sorted[0]]
    
    # Iterate through sorted lines to group them
    for i in range(1, len(lines_sorted)):
        if abs(lines_sorted[i][0] - current_group[-1][0]) < 0.3:
            x1_cur, y1_cur, x2_cur, y2_cur = current_group[-1][2].reshape(4)
            x1_new, y1_new, x2_new, y2_new = lines_sorted[i][2].reshape(4)
            # Calculate the center points of the current and new lines
            cx_cur, cy_cur = (x1_cur + x2_cur) / 2, (y1_cur + y2_cur) / 2
            cx_new, cy_new = (x1_new + x2_new) / 2, (y1_new + y2_new) / 2
            # Group lines if they have similar slopes and close center points
            if np.sqrt((cx_cur - cx_new) ** 2 + (cy_cur - cy_new) ** 2) < 90:
                current_group.append(lines_sorted[i])
            else:
                grouped_lines.append(current_group)
                current_group = [lines_sorted[i]]
        else:
            grouped_lines.append(current_group)
            current_group = [lines_sorted[i]]
    
    grouped_lines.append(current_group)
    return grouped_lines

def detect_lane_center(image):
    """
    Detect the center of the lane in the image.
    
    Args:
    - image: Preprocessed image with lanes detected.
    
    Returns:
    - center_lane_x, center_lane_y: Coordinates of the center of the lane.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Define the range of yellow color in HSV
    low_yellow = np.array([10, 94, 140])
    up_yellow = np.array([48, 255, 255])
    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    # Detect edges in the masked image
    edges = cv2.Canny(mask, 75, 150)
    
    # Detect lines using the Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=30, maxLineGap=70)
    points = []
    
    if lines is not None:
        positive_slopes = []
        negative_slopes = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if x2 - x1 == 0:  # Prevent division by zero
                continue
            slope_line = (y2 - y1) / (x2 - x1)
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if slope_line > 0:
                positive_slopes.append((slope_line, line_length, line))
            elif slope_line < 0:
                negative_slopes.append((slope_line, line_length, line))

        # Group positive and negative slope lines separately
        positive_groups = group_lines(positive_slopes)
        negative_groups = group_lines(negative_slopes)

        # Handle case with only negative lines
        if not positive_groups and negative_groups:
            negative_lines = [line for group in negative_groups for line in group]
            x_coords = [x for line in negative_lines for x in line[2].reshape(4)[::2]]
            if len(x_coords) > 1 and (max(x_coords) - min(x_coords)) > 400:
                center_lane_x = (max(x_coords) + min(x_coords)) / 2
                center_lane_y = image.shape[0] / 2  # Just an estimation
                return center_lane_x, center_lane_y
            else:
                return -1, -1
        
        # Handle case with only positive lines
        if not negative_groups and positive_groups:
            positive_lines = [line for group in positive_groups for line in group]
            x_coords = [x for line in positive_lines for x in line[2].reshape(4)[::2]]
            if len(x_coords) > 1 and (max(x_coords) - min(x_coords)) > 400:
                center_lane_x = (max(x_coords) + min(x_coords)) / 2
                center_lane_y = image.shape[0] / 2  # Just an estimation
                return center_lane_x, center_lane_y
            else:
                return 1, 1

        # Draw the longest line from each group
        for group in positive_groups:
            max_line = max(group, key=lambda x: x[1], default=None)
            if max_line is not None:
                x1, y1, x2, y2 = max_line[2].reshape(4)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                points.append((x1, y1))
                points.append((x2, y2))
        for group in negative_groups:
            max_line = max(group, key=lambda x: x[1], default=None)
            if max_line is not None:
                x1, y1, x2, y2 = max_line[2].reshape(4)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
                points.append((x1, y1))
                points.append((x2, y2))
    
    if len(points) < 2:
        return None, None

    # Find the furthest left and right points and calculate the center point between them
    furthest_points = max(points, key=lambda point: point[0]), min(points, key=lambda point: point[0])
    center_lane_x = (furthest_points[0][0] + furthest_points[1][0]) / 2
    center_lane_y = (furthest_points[0][1] + furthest_points[1][1]) / 2
    
    return center_lane_x, center_lane_y

def calculate_steering_angle(image_width, lane_center_x, max_steering_angle):
    """
    Calculate the steering angle based on the lane center.
    
    Args:
    - image_width: Width of the image.
    - lane_center_x: X-coordinate of the lane center.
    - max_steering_angle: Maximum steering angle of the car.
    
    Returns:
    - steering_angle: Calculated steering angle.
    """
    # Calculate the deviation of the lane center from the image center
    deviation = lane_center_x - image_width / 2
    # Normalize the deviation and scale it to the maximum steering angle
    steering_angle = deviation / (image_width / 2) * max_steering_angle
    return steering_angle

def draw_center_point(image, center_x, center_y):
    """
    Draw the center point of the lane on the image.
    
    Args:
    - image: Image to draw on.
    - center_x, center_y: Coordinates of the center point.
    
    Returns:
    - image_with_center: Image with center point drawn.
    """
    # Draw a circle at the center point of the lane
    image_with_center = cv2.circle(image, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)
    return image_with_center

def main():
    # Path to the folder containing images
    folder_path = r'C:\Users\Wasseem\Desktop\lane_test\test'
    
    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    
    for image_file in image_files:
        # Read image
        image_path = os.path.join(folder_path, image_file)
        image = read_image(image_path)
        
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        
        # Detect center of lane
        center_lane_x, center_lane_y = detect_lane_center(preprocessed_image)
        print("Center point of the lane in", image_file, ":", (center_lane_x, center_lane_y))
        
        # Assume example values for image width and maximum steering angle
        image_width = 315  # Width of the image (example value)
        max_steering_angle = 70  # Maximum steering angle supported by the car
        
        # Calculate the steering angle based on the lane center
        steering_angle = calculate_steering_angle(image_width, center_lane_x, max_steering_angle)
        print("Steering angle:", steering_angle)
        
        # Draw the center point on the image
        output_image = draw_center_point(preprocessed_image, center_lane_x, center_lane_y)
        
        # Display the output image
        plt.imshow(output_image)
        plt.title(f'Lane detection for {image_file}')
        plt.show()

if __name__ == "__main__":
    main()
