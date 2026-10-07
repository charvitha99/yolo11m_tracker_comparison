# YOLO11m PT vs ONNX + Deep SORT Object Tracking

A complete computer vision and video analytics project using **YOLO11m** for object detection, **ONNX** for model-format comparison, and **Deep SORT** for multi-object tracking.

The project processes the same input video using the YOLO11m PyTorch model and its ONNX version, compares their measured performance, and separately applies Deep SORT to generate a tracked output video with object IDs.

The project is also organized into separate Python modules to make the code easier to understand, maintain, test, and extend.

---

# 1. Project Overview

This project demonstrates a complete object detection and tracking pipeline:

1. Load the YOLO11m PyTorch model.
2. Run YOLO11m on the provided video.
3. Convert YOLO11m from PyTorch (`.pt`) to ONNX (`.onnx`).
4. Run the ONNX model on the same video.
5. Compare the PyTorch and ONNX results.
6. Select **Deep SORT** as the object tracker.
7. Use YOLO11m detections as input to Deep SORT.
8. Track objects across video frames.
9. Assign unique tracking IDs to objects.
10. Generate a tracked output video.
11. Save detection information in JSON format.
12. Save tracking and performance metrics in JSON format.
13. Generate a processing log.
14. Modularize the tracking pipeline into separate Python files.
15. Provide Docker and Docker Compose support.
16. Maintain the project using Git and GitHub.

---

# 2. Project Objectives

The main objectives of this project are:

- Understand YOLO11m object detection.
- Work with a PyTorch YOLO model.
- Convert a PyTorch model to ONNX.
- Understand model-format differences between PT and ONNX.
- Measure model inference performance.
- Compare PT and ONNX on the same video.
- Understand object tracking.
- Implement Deep SORT.
- Maintain object identities across frames.
- Generate a tracking video.
- Store detection and tracking information.
- Measure processing performance and resource usage.
- Organize the application using modular Python files.
- Containerize the project using Docker.
- Provide reproducible project setup and execution.

---

# 3. Technologies Used

## Programming Language

- Python 3.9.13

## Object Detection

- YOLO11m
- Ultralytics

## Model Formats

- PyTorch `.pt`
- ONNX `.onnx`

## Model Runtime

- PyTorch
- ONNX Runtime

## Object Tracking

- Deep SORT
- `deep-sort-realtime`

## Computer Vision

- OpenCV

## Performance Monitoring

- psutil

## Video Processing

- OpenCV
- FFmpeg

## Containerization

- Docker
- Docker Compose

## Version Control

- Git
- GitHub

---

# 4. Environment

The project was developed and tested using:

```text
Operating System: Windows 10
Python: 3.9.13
CPU: Intel Core i5-7300U
Execution Device: CPU
Docker Desktop: 29.6.2