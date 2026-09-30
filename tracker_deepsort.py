from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
import json
import os


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

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70

# ==========================================
# Create output directory
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ==========================================
# Video information
# ==========================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
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

# ==========================================
# Start timer
# ==========================================

start_time = time.perf_counter()


# ==========================================
# Process video
# ==========================================

while True:

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

            # Deep SORT format:
            # ([left, top, width, height],
            #  confidence,
            #  class_name)

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
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    frame_track_counts.append(
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
    end_time - start_time
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
# Tracking metrics
# ==========================================

tracking_metrics = {

    "model": MODEL_PATH,

    "tracker": "Deep SORT",

    "video": VIDEO_PATH,

    "configuration": {

        "imgsz": IMG_SIZE,

        "confidence_threshold":
            CONF_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "device": "cpu"

    },

    "video_information": {

        "total_frames":
            total_frames,

        "width":
            width,

        "height":
            height,

        "video_fps":
            fps

    },

    "detection": {

        "total_detections":
            total_detections,

        "average_confidence":
            round(
                average_confidence,
                4
            )

    },

    "tracking": {

        "unique_track_ids":
            len(unique_track_ids),

        "total_track_observations":
            total_tracks,

        "average_tracks_per_frame":
            round(
                average_tracks_per_frame,
                2
            )

    },

    "performance": {

        "total_processing_time_seconds":
            round(
                total_processing_time,
                2
            ),

        "average_processing_fps":
            round(
                average_processing_fps,
                2
            )

    },

    "evaluation": {

        "tracking_accuracy":
            None,

        "id_switches":
            None,

        "mota":
            None,

        "idf1":
            None,

        "hota":
            None

    }

}


# ==========================================
# Save detection JSON
# ==========================================

with open(
    OUTPUT_DETECTIONS,
    "w"
) as file:

    json.dump(
        all_detections,
        file,
        indent=2
    )


# ==========================================
# Save tracking metrics
# ==========================================

with open(
    OUTPUT_METRICS,
    "w"
) as file:

    json.dump(
        tracking_metrics,
        file,
        indent=4
    )


# ==========================================
# Final output
# ==========================================

print("\n========== DEEP SORT RESULTS ==========")

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
    f"Average tracks/frame: "
    f"{average_tracks_per_frame:.2f}"
)

print(
    f"Average confidence: "
    f"{average_confidence:.4f}"
)

print(
    f"Total processing time: "
    f"{total_processing_time:.2f} seconds"
)

print(
    f"Processing FPS: "
    f"{average_processing_fps:.2f}"
)

print(
    f"\nTracked video: "
    f"{OUTPUT_VIDEO}"
)

print(
    f"Detections JSON: "
    f"{OUTPUT_DETECTIONS}"
)

print(
    f"Metrics JSON: "
    f"{OUTPUT_METRICS}"
)

print("======================================")