# Vehicle-_speed-using-python-
This Python project detects vehicles in a video using the YOLOv8 object detection model and estimates their speed based on their movement between frames. It’s a practical demonstration of real-time vehicle tracking and rough speed approximation using computer vision.
.

🎯 Features

-Detects and tracks vehicles (default: cars) in video

-Estimates speed in km/h using bounding box displacement

-Annotates bounding boxes with speed and object IDs

-Uses YOLOv8 from Ultralytics

-Works on pre-recorded video footage

🛠 Requirements

-Python 3.8+

-Ultralytics YOLO

-OpenCV

-NumPy

#Install dependencies:

-bash

-Copy code

-pip install ultralytics opencv-python

▶ How to Run

1)Clone this repository.

2)Place a sample vehicle video in the project folder and update the path in vehicle_speed.py.

3)Run the script:

4)python vehicle_speed.py

5)Press q to stop the video.
