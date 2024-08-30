Ball Tracking Using Computer Vision

Project Name: Ball Tracking and Quadrant Detection

Description

This Python project implements a computer vision system for tracking colored balls in a video and recording their entry and exit events within defined quadrants.

Features
    Color-based Ball Detection: Utilizes OpenCV library to identify balls based on a specified color range in the HSV color space.
    Quadrant Detection: Divides the video frame into four equally sized quadrants.
    Entry/Exit Tracking: Monitors the balls' movement across quadrants and logs their entry and exit times with corresponding quadrants.
    Visualization: Overlays circles on detected balls and displays quadrant labels for visual clarity.
    Output Generation: Saves a processed video file showcasing the tracked balls and quadrants. Additionally, creates a text file recording entry/exit events with timestamps, quadrant numbers, color information ("Color" for consistency), and entry/exit labels.

Requirements
     Python 3.x
    OpenCV library (pip install opencv-python)
    NumPy library (pip install numpy)
