
import json
from pathlib import Path

# ==========================================
# File paths
# ==========================================

PT_FILE = "pt_metrics.json"
ONNX_FILE = "onnx_metrics.json"
TENSORRT_FILE = "results/tensorrt_metrics.json"


# ==========================================
# Load JSON files
# ==========================================

def load_json(filename):
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {path}\n"
            "Run the corresponding model benchmark first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


models = {
    "PT": load_json(PT_FILE),
    "ONNX": load_json(ONNX_FILE),
    "TensorRT": load_json(TENSORRT_FILE),
}


# ==========================================
# Extract metrics safely
# ==========================================

def get_metric(data, section, key):
    return data.get(section, {}).get(key, "N/A")


# ==========================================
# Display comparison
# ==========================================

print()
print("=" * 100)
print("YOLO11m: PYTORCH vs ONNX vs TENSORRT")
print("=" * 100)

print(
    f"{'Metric':<38}"
    f"{'PT':>18}"
    f"{'ONNX':>18}"
    f"{'TensorRT':>22}"
)

print("-" * 100)

comparison = [
    ("Total Frames", "video_information", "total_frames"),
    ("Total Detections", "detection", "total_detections"),
    ("Average Confidence", "detection", "average_confidence"),
    ("Minimum Confidence", "detection", "minimum_confidence"),
    ("Maximum Confidence", "detection", "maximum_confidence"),
    ("Processing Time (seconds)", "performance",
     "total_processing_time_seconds"),
    ("Processing FPS", "performance", "average_processing_fps"),
    ("Average Inference Time (ms)", "performance",
     "average_inference_time_ms"),
    ("Inference FPS", "performance", "inference_fps"),
]

for label, section, key in comparison:
    values = [
        get_metric(models[name], section, key)
        for name in ("PT", "ONNX", "TensorRT")
    ]

    print(
        f"{label:<38}"
        f"{str(values[0]):>18}"
        f"{str(values[1]):>18}"
        f"{str(values[2]):>22}"
    )

print("-" * 100)


# ==========================================
# Class-wise detection counts
# ==========================================

print("\nClass-wise Detection Counts")
print("-" * 100)

class_names = sorted({
    class_name
    for data in models.values()
    for class_name in data.get("detection", {}).get(
        "class_counts", {}
    )
})

for class_name in class_names:
    values = [
        data.get("detection", {}).get(
            "class_counts", {}
        ).get(class_name, 0)
        for data in models.values()
    ]

    print(
        f"{class_name:<38}"
        f"{str(values[0]):>18}"
        f"{str(values[1]):>18}"
        f"{str(values[2]):>22}"
    )


# ==========================================
# Evaluation metrics
# ==========================================

print("\nEvaluation Metrics")
print("-" * 100)

evaluation_keys = [
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-score", "f1_score"),
    ("mAP@50", "map50"),
    ("mAP@50-95", "map50_95"),
    ("IoU", "iou"),
]

for label, key in evaluation_keys:
    values = [
        data.get("evaluation", {}).get(key)
        for data in models.values()
    ]

    values = [
        "N/A" if value is None else value
        for value in values
    ]

    print(
        f"{label:<38}"
        f"{str(values[0]):>18}"
        f"{str(values[1]):>18}"
        f"{str(values[2]):>22}"
    )

print("-" * 100)
print("N/A means evaluation ground truth is unavailable.")
print("=" * 100)
