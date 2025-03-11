Object Detection using YOLOv8 (Computer Vision)

Overview

This project focuses on real-time object detection using the YOLOv8 model in Google Colab. Since Colab does not support direct webcam streaming, we utilize JavaScript to capture video frames and process them in Python. The goal is to detect objects in a live webcam feed and display results with bounding boxes.

Features:-

•	Real-time object detection using the YOLOv8 nano model.

•	Webcam capture in Colab using JavaScript.

•	Seamless integration of OpenCV, PyTorch, and Google Colab utilities.

•	Bounding box visualization for detected objects.

•	Efficient and scalable cloud-based solution.


Tech Stack:_

•	Python (for model processing)

•	JavaScript (for webcam access)

•	YOLOv8 (Ultralytics) (for object detection)

•	OpenCV (for image processing)

•	Google Colab (for execution)

•	PyTorch (for deep learning operations)

Installation & Setup:_

1.	Install required dependencies:
                     !pip install ultralytics opencv-python-headless torch torchvision torchaudio
2.	Import necessary libraries in Python:
          import cv2
          import torch
          from ultralytics import YOLO
          from google.colab.patches import cv2_imshow
3.	Load the YOLOv8 model:
                       model = YOLO("yolov8n.pt")
4.	Enable webcam in Colab:-
           from IPython.display import display, Javascript
                       from google.colab.output import eval_js
  	
6.	Capture and process video frames:-
   
      Use JavaScript to capture webcam frames.
  	
      Convert them into a format usable by YOLOv8.
  	
      Perform object detection and visualize results.

Usage:

•	Run the Colab notebook and execute the script.

•	The webcam feed captures images in real-time.

•	The YOLOv8 model processes each frame and displays detected objects with bounding boxes.

Results

The project successfully implements real-time object detection in Google Colab, overcoming webcam limitations. The model accurately detects objects, making it a scalable solution for cloud-based computer vision applications.

Future Enhancements

•	Implement multi-class detection with enhanced visualization.

•	Optimize frame processing speed for better real-time performance.

•	Expand support for custom object detection datasets.
________________________________________
Developed with YOLOv8 & Google Colab 🚀

