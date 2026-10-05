from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import cv2
import time
import json
import os
import statistics
import psutil


# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "yolo11m.pt"
VIDEO_PATH = "PNNL_Parking_LOT(1).avi"

OUTPUT_DIR = "results"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "deepsort_tracking.mp4"
)

OUTPUT_DETECTIONS = os.path.join(
    OUTPUT_DIR,
    "yolo11m_detections.json"
)

OUTPUT_METRICS = os.path.join(
    OUTPUT_DIR,
    "deepsort_metrics.json"
)

OUTPUT_LOG = os.path.join(
    OUTPUT_DIR,
    "deepsort.log"
)

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70


# ==========================================
# Create output directory
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================
# Load YOLO model
# ==========================================

print("Loading YOLO11m...")

model = YOLO(MODEL_PATH)

print("YOLO11m loaded successfully.")


# ==========================================
# Initialize Deep SORT
# ==========================================

print("Initializing Deep SORT...")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=None,
    embedder="mobilenet",
    half=False,
    bgr=True,
    embedder_gpu=False
)

print("Deep SORT initialized successfully.")


# ==========================================
# Open video
# ==========================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ==========================================
# Video information
# ==========================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


print("\n========== VIDEO INFORMATION ==========")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print("=======================================\n")


# ==========================================
# Output video writer
# ==========================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )


# ==========================================
# Storage
# ==========================================

all_detections = []

unique_track_ids = set()

frame_track_counts = []

total_detections = 0
total_tracks = 0

confidence_scores = []

frame_number = 0

# Track lifetime storage
track_first_frame = {}
track_last_frame = {}

# Timing storage
frame_latencies = []

# Resource monitoring
process = psutil.Process(
    os.getpid()
)

process_cpu_values = []
process_ram_values = []

system_cpu_values = []

max_active_tracks = 0


# ==========================================
# Start timer
# ==========================================

start_time = time.perf_counter()


# ==========================================
# Process video
# ==========================================

while True:

    frame_start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1


    # --------------------------------------
    # YOLO detection
    # --------------------------------------

    results = model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device="cpu",
        verbose=False
    )

    result = results[0]

    detections_for_tracker = []

    frame_detection_data = []


    if result.boxes is not None:

        for box in result.boxes:

            # Bounding box
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0].cpu()
            )

            class_id = int(
                box.cls[0].cpu()
            )

            class_name = model.names[
                class_id
            ]

            width_box = x2 - x1
            height_box = y2 - y1


            # Deep SORT format
            detections_for_tracker.append(
                (
                    [
                        x1,
                        y1,
                        width_box,
                        height_box
                    ],
                    confidence,
                    class_name
                )
            )


            # Save detection information
            frame_detection_data.append(
                {
                    "class": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                }
            )


            total_detections += 1

            confidence_scores.append(
                confidence
            )


    # --------------------------------------
    # Save detections for this frame
    # --------------------------------------

    all_detections.append(
        {
            "frame": frame_number,
            "detections": frame_detection_data
        }
    )


    # --------------------------------------
    # Deep SORT tracking
    # --------------------------------------

    tracks = tracker.update_tracks(
        detections_for_tracker,
        frame=frame
    )

    current_frame_tracks = 0


    # --------------------------------------
    # Draw tracking results
    # --------------------------------------

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        if ltrb is None:
            continue

        left, top, right, bottom = map(
            int,
            ltrb
        )


        unique_track_ids.add(
            track_id
        )

        current_frame_tracks += 1
        total_tracks += 1


        # Track lifetime
        if track_id not in track_first_frame:

            track_first_frame[
                track_id
            ] = frame_number

        track_last_frame[
            track_id
        ] = frame_number


        # Get class name
        class_name = track.get_det_class()

        if class_name is None:
            class_name = "object"


        # Draw bounding box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )


        # Track label
        label = (
            f"ID {track_id} "
            f"| {class_name}"
        )


        cv2.putText(
            frame,
            label,
            (
                left,
                max(top - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    frame_track_counts.append(
        current_frame_tracks
    )


    # Maximum active tracks
    if current_frame_tracks > max_active_tracks:

        max_active_tracks = (
            current_frame_tracks
        )


    # --------------------------------------
    # Frame information
    # --------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Active Tracks: {current_frame_tracks}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # --------------------------------------
    # Write output frame
    # --------------------------------------

    writer.write(frame)


    # --------------------------------------
    # Frame latency
    # --------------------------------------

    frame_end_time = time.perf_counter()

    frame_latency = (
        frame_end_time -
        frame_start_time
    )

    frame_latencies.append(
        frame_latency
    )


    # --------------------------------------
    # Resource monitoring
    # --------------------------------------

    try:

        process_cpu = (
            process.cpu_percent(
                interval=None
            )
        )

        process_ram = (
            process.memory_info()
            .rss / (1024 * 1024)
        )

        system_cpu = psutil.cpu_percent(
            interval=None
        )

        process_cpu_values.append(
            process_cpu
        )

        process_ram_values.append(
            process_ram
        )

        system_cpu_values.append(
            system_cpu
        )

    except Exception:
        pass


    # --------------------------------------
    # Progress
    # --------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames} frames"
        )


# ==========================================
# Stop timer
# ==========================================

end_time = time.perf_counter()

total_processing_time = (
    end_time -
    start_time
)


# ==========================================
# Release resources
# ==========================================

cap.release()
writer.release()


# ==========================================
# Calculate metrics
# ==========================================

average_processing_fps = (
    total_frames /
    total_processing_time
    if total_processing_time > 0
    else 0
)


average_tracks_per_frame = (
    sum(frame_track_counts) /
    len(frame_track_counts)
    if frame_track_counts
    else 0
)


average_confidence = (
    sum(confidence_scores) /
    len(confidence_scores)
    if confidence_scores
    else 0
)


# ==========================================
# Latency metrics
# ==========================================

latency_ms = [
    value * 1000
    for value in frame_latencies
]

average_latency_ms = (
    sum(latency_ms) /
    len(latency_ms)
    if latency_ms
    else 0
)


if latency_ms:

    sorted_latencies = sorted(
        latency_ms
    )

    p95_index = int(
        0.95 *
        (len(sorted_latencies) - 1)
    )

    p95_latency_ms = (
        sorted_latencies[p95_index]
    )

else:

    p95_latency_ms = 0


# ==========================================
# Track lifetime metrics
# ==========================================

track_lifetimes = []

for track_id in unique_track_ids:

    if track_id in track_first_frame:

        lifetime = (
            track_last_frame[track_id]
            -
            track_first_frame[track_id]
            +
            1
        )

        track_lifetimes.append(
            lifetime
        )


average_track_lifetime = (
    sum(track_lifetimes) /
    len(track_lifetimes)
    if track_lifetimes
    else 0
)


longest_track_lifetime = (
    max(track_lifetimes)
    if track_lifetimes
    else 0
)


# ==========================================
# Resource metrics
# ==========================================

average_process_cpu = (
    sum(process_cpu_values) /
    len(process_cpu_values)
    if process_cpu_values
    else 0
)

peak_process_cpu = (
    max(process_cpu_values)
    if process_cpu_values
    else 0
)

average_process_ram = (
    sum(process_ram_values) /
    len(process_ram_values)
    if process_ram_values
    else 0
)

peak_process_ram = (
    max(process_ram_values)
    if process_ram_values
    else 0
)

average_system_cpu = (
    sum(system_cpu_values) /
    len(system_cpu_values)
    if system_cpu_values
    else 0
)

peak_system_cpu = (
    max(system_cpu_values)
    if system_cpu_values
    else 0
)


# ==========================================
# Absolute output paths
# ==========================================

absolute_video_path = os.path.abspath(
    OUTPUT_VIDEO
)

absolute_json_path = os.path.abspath(
    OUTPUT_METRICS
)

absolute_log_path = os.path.abspath(
    OUTPUT_LOG
)

absolute_model_path = os.path.abspath(
    MODEL_PATH
)

absolute_video_input = os.path.abspath(
    VIDEO_PATH
)


# ==========================================
# Tracking metrics JSON
# ==========================================

tracking_metrics = {

    "project": {

        "name":
            "YOLO11m + Deep SORT Object Tracking",

        "tracker":
            "Deep SORT"
    },


    "model": {

        "name":
            "YOLO11m",

        "format":
            "pt",

        "path":
            absolute_model_path
    },


    "configuration": {

        "confidence_threshold":
            CONF_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "tracker_config":
            "Deep SORT",

        "device":
            "CPU"
    },


    "video": {

        "input":
            absolute_video_input,

        "width":
            width,

        "height":
            height,

        "fps":
            fps,

        "total_frames":
            total_frames
    },


    "tracking_statistics": {

        "frame_count":
            total_frames,

        "total_detections":
            total_detections,

        "unique_track_ids":
            len(unique_track_ids),

        "average_active_tracks":
            round(
                average_tracks_per_frame,
                2
            ),

        "maximum_active_tracks":
            max_active_tracks,

        "average_track_lifetime":
            round(
                average_track_lifetime,
                2
            ),

        "longest_track_lifetime":
            longest_track_lifetime,

        "average_fps":
            round(
                average_processing_fps,
                2
            ),

        "average_latency_ms":
            round(
                average_latency_ms,
                2
            ),

        "p95_latency_ms":
            round(
                p95_latency_ms,
                2
            ),

        "total_processing_time_seconds":
            round(
                total_processing_time,
                2
            )
    },


    "resources": {

        "process": {

            "average_cpu_percent":
                round(
                    average_process_cpu,
                    2
                ),

            "peak_cpu_percent":
                round(
                    peak_process_cpu,
                    2
                ),

            "average_ram_mb":
                round(
                    average_process_ram,
                    2
                ),

            "peak_ram_mb":
                round(
                    peak_process_ram,
                    2
                )
        },


        "system": {

            "average_cpu_percent":
                round(
                    average_system_cpu,
                    2
                ),

            "peak_cpu_percent":
                round(
                    peak_system_cpu,
                    2
                )
        }
    },


    "outputs": {

        "video":
            absolute_video_path,

        "json":
            absolute_json_path,

        "log":
            absolute_log_path
    }
}


# ==========================================
# Save detection JSON
# ==========================================

with open(
    OUTPUT_DETECTIONS,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_detections,
        file,
        indent=2
    )


# ==========================================
# Save tracking metrics JSON
# ==========================================

with open(
    OUTPUT_METRICS,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        tracking_metrics,
        file,
        indent=4
    )


# ==========================================
# Generate Deep SORT log automatically
# ==========================================

log_content = f"""
========== DEEP SORT LOG ==========

Project: YOLO11m + Deep SORT Object Tracking
Tracker: Deep SORT

========== MODEL ==========
Model: YOLO11m
Format: pt
Path: {absolute_model_path}

========== CONFIGURATION ==========
Confidence threshold: {CONF_THRESHOLD}
IoU threshold: {IOU_THRESHOLD}
Tracker config: Deep SORT
Device: CPU

========== VIDEO ==========
Input: {absolute_video_input}
Resolution: {width} x {height}
FPS: {fps}
Total frames: {total_frames}

========== TRACKING STATISTICS ==========
Frame count: {total_frames}
Total detections: {total_detections}
Unique track IDs: {len(unique_track_ids)}
Average active tracks: {average_tracks_per_frame}
Maximum active tracks: {max_active_tracks}
Average track lifetime: {average_track_lifetime}
Longest track lifetime: {longest_track_lifetime}
Average FPS: {average_processing_fps}
Average latency: {average_latency_ms} ms
P95 latency: {p95_latency_ms} ms
Total processing time: {total_processing_time} seconds

========== RESOURCES ==========
Average process CPU: {average_process_cpu}%
Peak process CPU: {peak_process_cpu}%
Average process RAM: {average_process_ram} MB
Peak process RAM: {peak_process_ram} MB
Average system CPU: {average_system_cpu}%
Peak system CPU: {peak_system_cpu}%

========== OUTPUTS ==========
Video: {absolute_video_path}
JSON: {absolute_json_path}
Log: {absolute_log_path}

===================================
"""


with open(
    OUTPUT_LOG,
    "w",
    encoding="utf-8"
) as log_file:

    log_file.write(
        log_content
    )


# ==========================================
# Final output
# ==========================================

print(
    "\n========== DEEP SORT RESULTS =========="
)

print(
    f"Total frames: "
    f"{total_frames}"
)

print(
    f"Total detections: "
    f"{total_detections}"
)

print(
    f"Unique track IDs: "
    f"{len(unique_track_ids)}"
)

print(
    f"Average active tracks: "
    f"{average_tracks_per_frame:.2f}"
)

print(
    f"Maximum active tracks: "
    f"{max_active_tracks}"
)

print(
    f"Average track lifetime: "
    f"{average_track_lifetime:.2f} frames"
)

print(
    f"Longest track lifetime: "
    f"{longest_track_lifetime} frames"
)

print(
    f"Average FPS: "
    f"{average_processing_fps:.2f}"
)

print(
    f"Average latency: "
    f"{average_latency_ms:.2f} ms"
)

print(
    f"P95 latency: "
    f"{p95_latency_ms:.2f} ms"
)

print(
    f"Total processing time: "
    f"{total_processing_time:.2f} seconds"
)

print(
    f"Average process CPU: "
    f"{average_process_cpu:.2f}%"
)

print(
    f"Peak process CPU: "
    f"{peak_process_cpu:.2f}%"
)

print(
    f"Average process RAM: "
    f"{average_process_ram:.2f} MB"
)

print(
    f"Peak process RAM: "
    f"{peak_process_ram:.2f} MB"
)

print(
    f"Average system CPU: "
    f"{average_system_cpu:.2f}%"
)

print(
    f"Peak system CPU: "
    f"{peak_system_cpu:.2f}%"
)

print(
    f"\nTracked video: "
    f"{absolute_video_path}"
)

print(
    f"Detections JSON: "
    f"{OUTPUT_DETECTIONS}"
)

print(
    f"Metrics JSON: "
    f"{absolute_json_path}"
)

print(
    f"Log file: "
    f"{absolute_log_path}"
)

print(
    "======================================"
)