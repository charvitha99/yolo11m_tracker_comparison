# YOLO11m PT vs ONNX and Deep SORT Object Tracking

## 📌 Project Overview

This project evaluates the YOLO11m object detection model in two formats:

- YOLO11m PyTorch (`.pt`)
- YOLO11m ONNX (`.onnx`)

The project also implements **Deep SORT** for multi-object tracking using YOLO11m detections.

The work is divided into two independent parts:

1. **YOLO11m PT vs ONNX model comparison**
2. **YOLO11m + Deep SORT object tracking**

---

## 🎯 Objectives

- Run YOLO11m using the PyTorch model.
- Convert YOLO11m from PyTorch to ONNX.
- Run the ONNX model on the same video.
- Compare PT and ONNX performance.
- Measure detection and inference performance.
- Implement Deep SORT for multi-object tracking.
- Generate a tracked output video with bounding boxes, class labels and track IDs.

---

## 🧰 Technologies Used

- Python 3.9.13
- YOLO11m
- Ultralytics
- PyTorch
- ONNX
- ONNX Runtime
- OpenCV
- Deep SORT
- `deep-sort-realtime`
- FFmpeg
- Git / GitHub

---

## 📂 Project Structure

```text
yolo11m_tracker_comparison/
│
├── pt_baseline.py
├── onnx_baseline.py
├── tracker_deepsort.py
├── compare_results.py
│
├── pt_metrics.json
├── onnx_metrics.json
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── results/
│   ├── deepsort_metrics.json
│   ├── deepsort_tracking.mp4
│   ├── deepsort_tracking_h264.mp4
│   └── yolo11m_detections.json
│
├── runs/
│
├── yolo11m.pt
├── yolo11m.onnx
├── PNNL_Parking_LOT(1).avi
└── venv/