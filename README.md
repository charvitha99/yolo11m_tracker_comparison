# YOLO11m PT vs ONNX + Deep SORT Object Tracking

## 1. Project Overview

This project performs object detection using the **YOLO11m model** in two formats:

- PyTorch (`.pt`)
- ONNX (`.onnx`)

Both models are tested using the same input video.

The project also implements **Deep SORT** for multi-object tracking using YOLO11m.

The complete Deep SORT pipeline is also **Dockerized** to provide a reproducible execution environment.

---

# 2. Project Objectives

The main objectives of this project are:

1. Run YOLO11m using the PyTorch model.
2. Convert YOLO11m from PyTorch to ONNX format.
3. Run the ONNX model on the same video.
4. Compare PT and ONNX performance.
5. Implement Deep SORT for object tracking.
6. Generate a tracked output video.
7. Store detection and tracking information in JSON format.
8. Collect performance and resource metrics.
9. Dockerize the complete application.
10. Maintain a reproducible project environment.

---

# 3. Technologies Used

- Python 3.9
- YOLO11m
- Ultralytics
- PyTorch
- ONNX
- ONNX Runtime
- OpenCV
- Deep SORT
- deep-sort-realtime
- NumPy
- SciPy
- psutil
- FFmpeg
- Docker
- Git
- GitHub

---

# 4. Input Video

The project uses:

```text
PNNL_Parking_LOT(1).avi