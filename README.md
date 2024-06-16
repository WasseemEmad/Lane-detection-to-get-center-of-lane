# Lane-detection-to-get-center-of-lane
This repository contains code for detecting lanes and calculating steering angles for autonomous vehicles. The code employs computer vision techniques to process images, identify lane markers, and determine the center of the lane. It is designed to assist in the development of self-driving car technology by providing accurate lane detection and steering angle calculation.

# Features

## Image Preprocessing
The image preprocessing module handles the conversion of images to the RGB color space and applies Gaussian blur to reduce noise and enhance the detection of lane markers.

## Lane Detection
Lane detection is performed using the Hough Transform, which identifies lines within edge-detected images. This technique is crucial for detecting the straight segments of lanes in the input images.

## Line Grouping
Detected lines are grouped based on their slopes and proximity. This step ensures that lines belonging to the same lane are grouped together, differentiating between multiple lanes if present.

## Center Calculation
The center calculation module determines the center point of the detected lane. This is done by finding the mid-point between the furthest left and right points of the grouped lines, which represents the center of the lane.

## Steering Angle Calculation
Based on the calculated lane center, the steering angle is computed to keep the vehicle centered within the lane. This steering angle is essential for guiding the autonomous vehicle.

## Visualization
The code includes visualization capabilities to draw the detected lane center on the image, providing a visual representation of the lane detection and steering guidance.

# Dependencies
- **OpenCV:** For image processing and computer vision tasks.
- **NumPy:** For numerical operations and array manipulation.
- **Matplotlib:** For visualization of the processed images and detected lanes.
  
# Results:
![image](https://github.com/WasseemEmad/Lane-detection-to-get-center-of-lane/assets/159874318/9b5ed4b7-4e1c-4884-8529-b57f4606b70b)
![image](https://github.com/WasseemEmad/Lane-detection-to-get-center-of-lane/assets/159874318/4b61d5ca-1bfe-44b4-b0f5-a97eabd35f0c)
