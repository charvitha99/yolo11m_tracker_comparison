from ultralytics import YOLO
import time
import json

MODEL_PATH = "yolo11m.onnx"
VIDEO_PATH = "PNNL_Parking_LOT(1).avi"

# Load ONNX model
model = YOLO(MODEL_PATH)

# Start total processing timer
start_time = time.perf_counter()

# Run ONNX inference
results = model.predict(
    source=VIDEO_PATH,
    imgsz=640,
    conf=0.25,
    iou=0.70,
    device="cpu",
    save=True,
    verbose=True
)

# End total processing timer
end_time = time.perf_counter()

# -----------------------------
# Basic metrics
# -----------------------------

total_processing_time = end_time - start_time
total_frames = len(results)

average_fps = total_frames / total_processing_time

# -----------------------------
# Detection metrics
# -----------------------------

confidence_scores = []
total_detections = 0
class_counts = {}

for result in results:

    if result.boxes is not None and len(result.boxes) > 0:

        # Confidence scores
        confidences = result.boxes.conf.cpu().tolist()
        confidence_scores.extend(confidences)

        # Number of detections
        total_detections += len(confidences)

        # Class IDs
        class_ids = result.boxes.cls.cpu().tolist()

        for class_id in class_ids:

            class_id = int(class_id)
            class_name = model.names[class_id]

            if class_name not in class_counts:
                class_counts[class_name] = 0

            class_counts[class_name] += 1

# -----------------------------
# Confidence statistics
# -----------------------------

if confidence_scores:

    average_confidence = (
        sum(confidence_scores) / len(confidence_scores)
    )

    maximum_confidence = max(confidence_scores)
    minimum_confidence = min(confidence_scores)

else:

    average_confidence = 0
    maximum_confidence = 0
    minimum_confidence = 0

# -----------------------------
# Inference speed
# -----------------------------

inference_times = []

for result in results:

    if hasattr(result, "speed") and result.speed:

        if "inference" in result.speed:

            inference_times.append(
                result.speed["inference"]
            )

if inference_times:

    average_inference_time = (
        sum(inference_times) / len(inference_times)
    )

    inference_fps = 1000 / average_inference_time

else:

    average_inference_time = 0
    inference_fps = 0

# -----------------------------
# Evaluation metrics
# -----------------------------
# Ground-truth annotations are required.

evaluation_metrics = {

    "precision": None,
    "recall": None,
    "f1_score": None,
    "map50": None,
    "map50_95": None,
    "iou": None

}

# -----------------------------
# Create metrics dictionary
# -----------------------------

metrics = {

    "model": MODEL_PATH,

    "format": "ONNX",

    "video": VIDEO_PATH,

    "inference_settings": {

        "imgsz": 640,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.70,
        "device": "cpu"

    },

    "video_information": {

        "total_frames": total_frames

    },

    "performance": {

        "total_processing_time_seconds":
            round(total_processing_time, 2),

        "average_processing_fps":
            round(average_fps, 2),

        "average_inference_time_ms":
            round(average_inference_time, 2),

        "inference_fps":
            round(inference_fps, 2)

    },

    "detection": {

        "total_detections":
            total_detections,

        "average_confidence":
            round(average_confidence, 4),

        "minimum_confidence":
            round(minimum_confidence, 4),

        "maximum_confidence":
            round(maximum_confidence, 4),

        "class_counts":
            class_counts

    },

    "evaluation": evaluation_metrics

}

# -----------------------------
# Save metrics
# -----------------------------

output_file = "onnx_metrics.json"

with open(output_file, "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )

# -----------------------------
# Print results
# -----------------------------

print("\n========== ONNX BASELINE ==========")

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")

print(f"Total frames: {total_frames}")
print(f"Total detections: {total_detections}")

print(
    f"Total processing time: "
    f"{total_processing_time:.2f} seconds"
)

print(
    f"Average processing FPS: "
    f"{average_fps:.2f}"
)

print(
    f"Average inference time: "
    f"{average_inference_time:.2f} ms/frame"
)

print(
    f"Inference FPS: "
    f"{inference_fps:.2f}"
)

print(
    f"Average confidence: "
    f"{average_confidence:.4f}"
)

print(
    f"Minimum confidence: "
    f"{minimum_confidence:.4f}"
)

print(
    f"Maximum confidence: "
    f"{maximum_confidence:.4f}"
)

print(f"Class counts: {class_counts}")

print(f"\nMetrics saved to: {output_file}")

print("====================================")