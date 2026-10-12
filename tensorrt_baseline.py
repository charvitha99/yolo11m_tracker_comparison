
from ultralytics import YOLO
import time
import json
from pathlib import Path

MODEL_PATH = "yolo11m.engine"
VIDEO_PATH = "PNNL_Parking_LOT(1).avi"
OUTPUT_FILE = "tensorrt_metrics.json"

# Check required files
if not Path(MODEL_PATH).is_file():
    raise FileNotFoundError(
        f"TensorRT engine not found: {MODEL_PATH}. "
        "Export yolo11m.onnx to TensorRT first."
    )

if not Path(VIDEO_PATH).is_file():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

# Load TensorRT engine
model = YOLO(MODEL_PATH, task="detect")

# Run inference on GPU
start_time = time.perf_counter()

results = model.predict(
    source=VIDEO_PATH,
    imgsz=640,
    conf=0.25,
    iou=0.70,
    device=0,
    save=True,
    project="runs/tensorrt",
    name="prediction",
    exist_ok=True,
    verbose=True
)

total_processing_time = time.perf_counter() - start_time

# Basic metrics
total_frames = len(results)
average_fps = (
    total_frames / total_processing_time
    if total_processing_time > 0 else 0
)

# Detection metrics
confidence_scores = []
total_detections = 0
class_counts = {}

for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.cpu().tolist()
        confidence_scores.extend(confidences)
        total_detections += len(confidences)

        class_ids = result.boxes.cls.cpu().tolist()

        for class_id in class_ids:
            class_name = model.names[int(class_id)]
            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

# Confidence statistics
if confidence_scores:
    average_confidence = sum(confidence_scores) / len(confidence_scores)
    minimum_confidence = min(confidence_scores)
    maximum_confidence = max(confidence_scores)
else:
    average_confidence = 0
    minimum_confidence = 0
    maximum_confidence = 0

# Per-frame inference time reported by Ultralytics
inference_times = [
    result.speed["inference"]
    for result in results
    if result.speed and "inference" in result.speed
]

average_inference_time = (
    sum(inference_times) / len(inference_times)
    if inference_times else 0
)

inference_fps = (
    1000 / average_inference_time
    if average_inference_time > 0 else 0
)

# Ground-truth annotations are required for these metrics
evaluation_metrics = {
    "precision": None,
    "recall": None,
    "f1_score": None,
    "map50": None,
    "map50_95": None,
    "iou": None
}

# Keep the same JSON structure as PT and ONNX
metrics = {
    "model": MODEL_PATH,
    "format": "TensorRT",
    "video": VIDEO_PATH,

    "inference_settings": {
        "imgsz": 640,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.70,
        "device": "cuda:0"
    },

    "video_information": {
        "total_frames": total_frames
    },

    "performance": {
        "total_processing_time_seconds": round(
            total_processing_time, 2
        ),
        "average_processing_fps": round(average_fps, 2),
        "average_inference_time_ms": round(
            average_inference_time, 2
        ),
        "inference_fps": round(inference_fps, 2)
    },

    "detection": {
        "total_detections": total_detections,
        "average_confidence": round(average_confidence, 4),
        "minimum_confidence": round(minimum_confidence, 4),
        "maximum_confidence": round(maximum_confidence, 4),
        "class_counts": class_counts
    },

    "evaluation": evaluation_metrics
}

# Save metrics
Path("results").mkdir(parents=True, exist_ok=True)
output_path = Path("results") / OUTPUT_FILE

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# Print summary
print("\n========== TENSORRT BASELINE ==========")
print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Total frames: {total_frames}")
print(f"Total detections: {total_detections}")
print(f"Processing time: {total_processing_time:.2f} seconds")
print(f"Processing FPS: {average_fps:.2f}")
print(f"Average inference time: {average_inference_time:.2f} ms/frame")
print(f"Inference FPS: {inference_fps:.2f}")
print(f"Average confidence: {average_confidence:.4f}")
print(f"Minimum confidence: {minimum_confidence:.4f}")
print(f"Maximum confidence: {maximum_confidence:.4f}")
print(f"Class counts: {class_counts}")
print(f"Metrics saved to: {output_path}")
print("========================================")
