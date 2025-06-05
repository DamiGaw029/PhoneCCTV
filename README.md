# YOLO-Person-Detection-CCTV

## Overview

This project demonstrates a simple and efficient way to repurpose an old smartphone as a wireless CCTV camera using object detection powered by YOLOv8.

By leveraging the DroidCam mobile app to stream video over Wi-Fi and running a Python-based detection script on a computer, the system detects people in real time using the YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics).

## Features

- Uses an old phone as a wireless IP camera (via DroidCam)
- Runs real-time object detection with YOLOv8
- Filters detections to show only persons
- Displays bounding boxes around detected people

## Future Plans

- Save frames to disk only when a person is detected
- Archive detection logs with timestamps
- Optional web API (FastAPI) for remote access or integration

## Requirements

- Python 3.8 or higher
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV
- DroidCam app (Android or iOS) running on your phone
