import json


# ==========================================
# File paths
# ==========================================

PT_FILE = "pt_metrics.json"
ONNX_FILE = "onnx_metrics.json"


# ==========================================
# Load JSON files
# ==========================================

def load_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


pt = load_json(PT_FILE)
onnx = load_json(ONNX_FILE)


# ==========================================
# Extract PT metrics
# ==========================================

pt_frames = pt["video_information"]["total_frames"]

pt_detections = pt["detection"]["total_detections"]

pt_confidence = pt["detection"]["average_confidence"]

pt_min_confidence = pt["detection"]["minimum_confidence"]

pt_max_confidence = pt["detection"]["maximum_confidence"]

pt_time = pt["performance"]["total_processing_time_seconds"]

pt_fps = pt["performance"]["average_processing_fps"]

pt_inference_time = pt["performance"]["average_inference_time_ms"]

pt_inference_fps = pt["performance"]["inference_fps"]


# ==========================================
# Extract ONNX metrics
# ==========================================

onnx_frames = onnx["video_information"]["total_frames"]

onnx_detections = onnx["detection"]["total_detections"]

onnx_confidence = onnx["detection"]["average_confidence"]

onnx_min_confidence = onnx["detection"]["minimum_confidence"]

onnx_max_confidence = onnx["detection"]["maximum_confidence"]

onnx_time = onnx["performance"]["total_processing_time_seconds"]

onnx_fps = onnx["performance"]["average_processing_fps"]

onnx_inference_time = onnx["performance"]["average_inference_time_ms"]

onnx_inference_fps = onnx["performance"]["inference_fps"]


# ==========================================
# Header
# ==========================================

print()
print("=" * 75)
print("YOLO11m PT vs ONNX COMPARISON")
print("=" * 75)

print(
    f"{'Metric':<40}"
    f"{'PT':>15}"
    f"{'ONNX':>15}"
)

print("-" * 75)


# ==========================================
# Comparison table
# ==========================================

comparison = [

    (
        "Total Frames",
        pt_frames,
        onnx_frames
    ),

    (
        "Total Detections",
        pt_detections,
        onnx_detections
    ),

    (
        "Average Confidence",
        pt_confidence,
        onnx_confidence
    ),

    (
        "Minimum Confidence",
        pt_min_confidence,
        onnx_min_confidence
    ),

    (
        "Maximum Confidence",
        pt_max_confidence,
        onnx_max_confidence
    ),

    (
        "Processing Time (seconds)",
        pt_time,
        onnx_time
    ),

    (
        "Processing FPS",
        pt_fps,
        onnx_fps
    ),

    (
        "Average Inference Time (ms)",
        pt_inference_time,
        onnx_inference_time
    ),

    (
        "Inference FPS",
        pt_inference_fps,
        onnx_inference_fps
    )
]


for metric, pt_value, onnx_value in comparison:

    print(
        f"{metric:<40}"
        f"{str(pt_value):>15}"
        f"{str(onnx_value):>15}"
    )


print("-" * 75)


# ==========================================
# Evaluation metrics
# ==========================================

print()
print("Evaluation Metrics")
print("-" * 75)

evaluation_metrics = [
    "Precision",
    "Recall",
    "F1-score",
    "mAP@50",
    "mAP@50-95",
    "IoU"
]

for metric in evaluation_metrics:

    print(
        f"{metric:<40}"
        f"{'N/A':>15}"
        f"{'N/A':>15}"
        f"  Ground truth required"
    )


print()
print("=" * 75)