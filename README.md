# YOLO11m Model Performance Comparison: PyTorch vs ONNX vs TensorRT

## Overview

This project benchmarks the YOLO11m object detection model using three inference formats:

- **PyTorch (.pt)** — Original YOLO11m model
- **ONNX (.onnx)** — ONNX model format
- **TensorRT (.engine)** — TensorRT engine built from the existing ONNX model for GPU inference

The project compares inference speed, processing FPS, detection counts, confidence scores, and class-wise detection results using the same parking-lot video.

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLO11m
- ONNX
- NVIDIA TensorRT
- CUDA-enabled NVIDIA GPU
- OpenCV
- JSON

## Project Structure

```text
yolo11m_tracker_comparison/
├── yolo11m.pt
├── yolo11m.onnx
├── tensorrt_baseline.py
├── pt_baseline.py
├── onnx_baseline.py
├── compare_results.py
├── pt_metrics.json
├── onnx_metrics.json
├── comparison_results.json
├── comparison_results.txt
├── results/
│   ├── deepsort_metrics.json
│   └── tensorrt_metrics.json
├── PNNL_Parking_LOT(1).avi
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

The generated `yolo11m.engine` file and GPU virtual environment are not included in the Git repository.

## Model Conversion

The existing ONNX model was converted into a TensorRT engine using the TensorRT Python API.

**Conversion details:**
- Input model: `yolo11m.onnx`
- Output engine: `yolo11m.engine`
- Input shape: `(1, 3, 640, 640)`
- Input data type: FP32
- Output tensor shape: `(1, 84, 8400)`
- TensorRT version: 11.4.0.106

The generated engine was successfully loaded and tested on an NVIDIA GeForce RTX 3060 GPU.

## Benchmark Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11m |
| Input video | `PNNL_Parking_LOT(1).avi` |
| Total video frames | 998 |
| Image size | 640 × 640 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.70 |
| TensorRT execution device | NVIDIA GPU |
| TensorRT engine precision | FP32 |

## Performance Results

The following results were collected from the benchmark runs.

| Metric | PyTorch | ONNX | TensorRT |
|---|---:|---:|---:|
| Total frames | 998 | 998 | 998 |
| Total detections | 13,053 | 13,291 | 13,287 |
| Processing time (seconds) | 602.52 | 938.28 | 22.02 |
| Processing FPS | 1.66 | 1.06 | 45.33 |
| Average inference time (ms) | 550.33 | 880.39 | 10.30 |
| Inference FPS | 1.82 | 1.14 | 97.07 |
| Average confidence | 0.7178 | 0.7287 | 0.7287 |

**Important:** The PyTorch and ONNX results were collected on the CPU, while TensorRT was executed on the GPU. These measurements demonstrate the observed performance in the recorded runs, but they are not a controlled, same-hardware comparison.

## TensorRT Results

The TensorRT benchmark processed 998 frames with the following results:

- Processing time: **22.02 seconds**
- Processing speed: **45.33 FPS**
- Average inference time: **10.30 ms/frame**
- Inference speed: **97.07 FPS**
- Total detections: **13,287**
- Average confidence: **0.7287**

The benchmark saves its metrics in `results/tensorrt_metrics.json`.

## Running the Benchmarks

### 1. Install dependencies

Create and activate a Python virtual environment, then install the dependencies supported by your system.

```bash
pip install -r requirements.txt
```

For TensorRT benchmarking, make sure that a compatible NVIDIA GPU driver, CUDA environment, TensorRT installation, and GPU-enabled PyTorch/Ultralytics environment are available.

### 2. Run the PyTorch benchmark

```bash
python pt_baseline.py
```

### 3. Run the ONNX benchmark

```bash
python onnx_baseline.py
```

### 4. Generate the TensorRT engine

The TensorRT engine was generated from the existing ONNX model using the TensorRT Python API. The engine is hardware- and software-environment dependent, so generate or transfer a compatible engine for the target system before running the TensorRT benchmark.

The generated engine is named:

```text
yolo11m.engine
```

### 5. Run the TensorRT benchmark

```bash
python tensorrt_baseline.py
```

The benchmark stores its metrics in:

```text
results/tensorrt_metrics.json
```

### 6. Compare all model results

```bash
python compare_results.py
```

To save the displayed comparison to a text file:

```bash
python compare_results.py > comparison_results.txt
```

The combined JSON metrics are stored in:

```text
comparison_results.json
```

This file contains the recorded metrics from PyTorch, ONNX, and TensorRT.

## Evaluation Metrics

The comparison includes:

- Total detections
- Average, minimum, and maximum confidence
- Total processing time
- Processing FPS
- Average inference time
- Inference FPS
- Class-wise detection counts

Precision, recall, F1-score, mAP@50, mAP@50–95, and IoU are currently reported as unavailable because ground-truth annotations were not provided for this benchmark.

## Notes and Limitations

- All three benchmark runs use the same input video and nominal image size.
- PyTorch and ONNX were benchmarked on CPU; TensorRT was benchmarked on GPU.
- For a fair performance comparison, run all three formats on the same hardware with consistent preprocessing and benchmark settings.
- Class names in TensorRT output should be mapped consistently with the original YOLO11m model before comparing class-wise counts.
- TensorRT engine compatibility depends on the target GPU and software environment.
- The generated engine and virtual environment are excluded from the repository.

## Future Improvements

- Benchmark PyTorch and ONNX on the same GPU.
- Standardize class names across all model formats.
- Evaluate FP16 TensorRT inference against FP32.
- Add ground-truth annotations to calculate precision, recall, F1-score, and mAP.
- Generate plots and structured benchmark reports.

## Author
P.Charvitha
