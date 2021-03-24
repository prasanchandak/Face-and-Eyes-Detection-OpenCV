# Face and Eyes Detection Program

Problem Statement: Implement Face Detection and Eyes Detection using HaarCascade Classifier in a Video Frame.

The programs included detect Face and Eyes using HaarCascade Classifier in a Video Frame. The video could be in the form of a 
live video or a video file. These programs utilize the OpenCV library in Python.

## Tools Used

Python: An interpreted, high-level and general-purpose programming language.

NumPy is a Python library that provides a simple yet powerful data structure: the n-dimensional array. 

OpenCV (Open Source Computer Vision Library: http://opencv.org) is an open-source BSD-licensed library that includes 
several hundreds of computer vision algorithms. It is a cross platform library and in this program it is used in Python.

Haar Cascade Classifiers : We will implement our use case using the Haar Cascade classifier. Haar Cascade classifier 
is an effective object detection approach which was proposed by Paul Viola and Michael Jones in their paper, “Rapid 
Object Detection using a Boosted Cascade of Simple Features” in 2001.

## Installation and Operating Instructions

Installing NumPy

````
pip install numpy
````

Installing the OpenCV library in Python

````
pip install opencv-python
````

Importing the NumPy and OpenCV libraries 

````python
import numpy as np
import cv2

````

## Files Included

FaceEyes_Detection_Live_Video_Feed.py: This program detects face and eyes in a live video feed and outputs it to a new file.

FaceEyes_Detection_Video_File.py: This program detects face and eyes in an already existing video file and outputs it to a new file.

## References

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

https://towardsdatascience.com/computer-vision-detecting-objects-using-haar-cascade-classifier-4585472829a9

https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
 