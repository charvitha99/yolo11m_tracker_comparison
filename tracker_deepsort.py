import os
import json
import time
import statistics

import cv2
import psutil

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "PNNL_Parking_LOT(1).avi"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11m.pt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


TRACKED_VIDEO_PATH = os.path.join(
    RESULTS_DIR,
    "deepsort_tracking.mp4"
)

DETECTIONS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "yolo11m_detections.json"
)

METRICS_JSON_PATH = os.path.join(
    RESULTS_DIR,
    "deepsort_metrics.json"
)

LOG_PATH = os.path.join(
    RESULTS_DIR,
    "deepsort.log"
)


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70
DEVICE = "cpu"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # ========================================================
    # START TIME
    # ========================================================

    total_start_time = time.perf_counter()

    print("Loading YOLO11m...")

    model = YOLO(MODEL_PATH)

    print("YOLO11m loaded successfully.")


    # ========================================================
    # INITIALIZE DEEP SORT
    # ========================================================

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


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )


    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    print()
    print("========== VIDEO INFORMATION ==========")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {video_fps}")
    print(f"Total frames: {total_frames}")
    print("=======================================")
    print()


    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        TRACKED_VIDEO_PATH,
        fourcc,
        video_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {TRACKED_VIDEO_PATH}"
        )


    # ========================================================
    # TRACKING STATISTICS
    # ========================================================

    total_detections = 0

    unique_track_ids = set()

    active_tracks_per_frame = []

    track_lifetimes = {}

    confidence_scores = []

    frame_detections = {}

    latency_values = []

    process_cpu_values = []
    process_ram_values = []

    system_cpu_values = []


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    frame_number = 0

    process = psutil.Process(os.getpid())

    while True:

        frame_start = time.perf_counter()

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # ====================================================
        # YOLO DETECTION
        # ====================================================

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False
        )

        result = results[0]


        # ====================================================
        # PREPARE DEEP SORT DETECTIONS
        # ====================================================

        detections_for_tracker = []

        frame_detection_data = []


        if result.boxes is not None:

            for box in result.boxes:

                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(
                    float,
                    xyxy
                )

                confidence = float(
                    box.conf[0].cpu().item()
                )

                class_id = int(
                    box.cls[0].cpu().item()
                )

                class_name = (
                    model.names[class_id]
                    if class_id in model.names
                    else str(class_id)
                )


                width_box = x2 - x1
                height_box = y2 - y1


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


                frame_detection_data.append(
                    {
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    }
                )


                total_detections += 1
                confidence_scores.append(confidence)


        frame_detections[str(frame_number)] = (
            frame_detection_data
        )


        # ====================================================
        # DEEP SORT TRACKING
        # ====================================================

        tracks = tracker.update_tracks(
            detections_for_tracker,
            frame=frame
        )


        active_track_count = 0


        for track in tracks:

            if not track.is_confirmed():
                continue

            if track.time_since_update > 1:
                continue


            active_track_count += 1


            track_id = track.track_id

            unique_track_ids.add(track_id)


            # Track lifetime
            if track_id not in track_lifetimes:

                track_lifetimes[track_id] = {
                    "start_frame": frame_number,
                    "end_frame": frame_number
                }

            else:

                track_lifetimes[track_id][
                    "end_frame"
                ] = frame_number


            # Get bounding box
            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                ltrb
            )


            # Get class
            class_name = track.get_det_class()

            if class_name is None:
                class_name = "object"


            # Confidence
            track_confidence = track.det_conf

            if track_confidence is None:
                track_confidence = 0.0


            # =================================================
            # DRAW TRACK
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            label = (
                f"ID: {track_id} "
                f"{class_name} "
                f"{track_confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        active_tracks_per_frame.append(
            active_track_count
        )


        # ====================================================
        # FRAME INFORMATION
        # ====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Active Tracks: {active_track_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ====================================================
        # WRITE FRAME
        # ====================================================

        writer.write(frame)


        # ====================================================
        # RESOURCE METRICS
        # ====================================================

        process_cpu = process.cpu_percent(
            interval=None
        )

        process_memory = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        system_cpu = psutil.cpu_percent(
            interval=None
        )


        process_cpu_values.append(
            process_cpu
        )

        process_ram_values.append(
            process_memory
        )

        system_cpu_values.append(
            system_cpu
        )


        # ====================================================
        # FRAME LATENCY
        # ====================================================

        frame_latency = (
            time.perf_counter()
            - frame_start
        ) * 1000

        latency_values.append(
            frame_latency
        )


        # ====================================================
        # PROGRESS
        # ====================================================

        if frame_number % 50 == 0:

            print(
                f"Processed "
                f"{frame_number}/{total_frames} frames"
            )


    # ========================================================
    # RELEASE RESOURCES
    # ========================================================

    cap.release()
    writer.release()


    # ========================================================
    # TOTAL PROCESSING TIME
    # ========================================================

    total_processing_time = (
        time.perf_counter()
        - total_start_time
    )


    average_fps = (
        total_frames / total_processing_time
        if total_processing_time > 0
        else 0
    )


    average_latency = (
        statistics.mean(latency_values)
        if latency_values
        else 0
    )


    # ========================================================
    # P95 LATENCY
    # ========================================================

    if latency_values:

        sorted_latencies = sorted(
            latency_values
        )

        index = int(
            0.95 * (len(sorted_latencies) - 1)
        )

        p95_latency = sorted_latencies[index]

    else:

        p95_latency = 0


    # ========================================================
    # TRACK LIFETIME
    # ========================================================

    lifetime_values = []

    for track_id, lifetime in track_lifetimes.items():

        lifetime_frames = (
            lifetime["end_frame"]
            - lifetime["start_frame"]
            + 1
        )

        lifetime_values.append(
            lifetime_frames
        )


    average_track_lifetime = (
        statistics.mean(lifetime_values)
        if lifetime_values
        else 0
    )


    longest_track_lifetime = (
        max(lifetime_values)
        if lifetime_values
        else 0
    )


    # ========================================================
    # ACTIVE TRACK STATISTICS
    # ========================================================

    average_active_tracks = (
        statistics.mean(active_tracks_per_frame)
        if active_tracks_per_frame
        else 0
    )


    maximum_active_tracks = (
        max(active_tracks_per_frame)
        if active_tracks_per_frame
        else 0
    )


    # ========================================================
    # CONFIDENCE STATISTICS
    # ========================================================

    if confidence_scores:

        average_confidence = statistics.mean(
            confidence_scores
        )

        minimum_confidence = min(
            confidence_scores
        )

        maximum_confidence = max(
            confidence_scores
        )

    else:

        average_confidence = 0
        minimum_confidence = 0
        maximum_confidence = 0


    # ========================================================
    # RESOURCE STATISTICS
    # ========================================================

    average_process_cpu = (
        statistics.mean(process_cpu_values)
        if process_cpu_values
        else 0
    )

    peak_process_cpu = (
        max(process_cpu_values)
        if process_cpu_values
        else 0
    )

    average_process_ram = (
        statistics.mean(process_ram_values)
        if process_ram_values
        else 0
    )

    peak_process_ram = (
        max(process_ram_values)
        if process_ram_values
        else 0
    )

    average_system_cpu = (
        statistics.mean(system_cpu_values)
        if system_cpu_values
        else 0
    )

    peak_system_cpu = (
        max(system_cpu_values)
        if system_cpu_values
        else 0
    )


    # ========================================================
    # ABSOLUTE OUTPUT PATHS
    # ========================================================

    absolute_video_path = os.path.abspath(
        TRACKED_VIDEO_PATH
    )

    absolute_json_path = os.path.abspath(
        METRICS_JSON_PATH
    )

    absolute_log_path = os.path.abspath(
        LOG_PATH
    )


    # ========================================================
    # METRICS JSON
    # ========================================================

    metrics = {

        "project": {
            "name": "YOLO11m + Deep SORT Object Tracking",
            "tracker": "Deep SORT"
        },

        "model": {
            "name": "YOLO11m",
            "format": "pt",
            "path": os.path.abspath(MODEL_PATH)
        },

        "configuration": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "tracker_config": "Deep SORT",
            "device": "CPU"
        },

        "video": {
            "input": os.path.abspath(VIDEO_PATH),
            "width": width,
            "height": height,
            "fps": video_fps,
            "total_frames": total_frames
        },

        "tracking_statistics": {

            "frame_count": total_frames,

            "total_detections": total_detections,

            "unique_track_ids": len(
                unique_track_ids
            ),

            "average_active_tracks": round(
                average_active_tracks,
                2
            ),

            "maximum_active_tracks": maximum_active_tracks,

            "average_track_lifetime": round(
                average_track_lifetime,
                2
            ),

            "longest_track_lifetime": longest_track_lifetime,

            "average_fps": round(
                average_fps,
                2
            ),

            "average_latency_ms": round(
                average_latency,
                2
            ),

            "p95_latency_ms": round(
                p95_latency,
                2
            ),

            "total_processing_time_seconds": round(
                total_processing_time,
                2
            )
        },

        "resources": {

            "process": {

                "average_cpu_percent": round(
                    average_process_cpu,
                    2
                ),

                "peak_cpu_percent": round(
                    peak_process_cpu,
                    2
                ),

                "average_ram_mb": round(
                    average_process_ram,
                    2
                ),

                "peak_ram_mb": round(
                    peak_process_ram,
                    2
                )
            },

            "system": {

                "average_cpu_percent": round(
                    average_system_cpu,
                    2
                ),

                "peak_cpu_percent": round(
                    peak_system_cpu,
                    2
                )
            }
        },

        "outputs": {

            "video": absolute_video_path,

            "json": absolute_json_path,

            "log": absolute_log_path
        }
    }


    # ========================================================
    # SAVE METRICS
    # ========================================================

    with open(
        METRICS_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


    # ========================================================
    # SAVE DETECTIONS
    # ========================================================

    detection_output = {

        "project": "YOLO11m + Deep SORT",

        "video": os.path.abspath(
            VIDEO_PATH
        ),

        "total_frames": total_frames,

        "detections": frame_detections
    }


    with open(
        DETECTIONS_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detection_output,
            file,
            indent=4
        )


    # ========================================================
    # SAVE LOG
    # ========================================================

    log_content = f"""
========== DEEP SORT LOG ==========

Project: YOLO11m + Deep SORT Object Tracking
Tracker: Deep SORT

========== VIDEO ==========
Input: {os.path.abspath(VIDEO_PATH)}
Resolution: {width} x {height}
FPS: {video_fps}
Total Frames: {total_frames}

========== TRACKING ==========
Total Detections: {total_detections}
Unique Track IDs: {len(unique_track_ids)}
Average Active Tracks: {average_active_tracks:.2f}
Maximum Active Tracks: {maximum_active_tracks}
Average Track Lifetime: {average_track_lifetime:.2f}
Longest Track Lifetime: {longest_track_lifetime}

========== PERFORMANCE ==========
Average FPS: {average_fps:.2f}
Average Latency: {average_latency:.2f} ms
P95 Latency: {p95_latency:.2f} ms
Total Processing Time: {total_processing_time:.2f} seconds

========== RESOURCES ==========
Average Process CPU: {average_process_cpu:.2f}%
Peak Process CPU: {peak_process_cpu:.2f}%
Average Process RAM: {average_process_ram:.2f} MB
Peak Process RAM: {peak_process_ram:.2f} MB

Average System CPU: {average_system_cpu:.2f}%
Peak System CPU: {peak_system_cpu:.2f}%

========== OUTPUTS ==========
Video: {absolute_video_path}
JSON: {absolute_json_path}
Log: {absolute_log_path}

===================================
"""


    with open(
        LOG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(log_content)


    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print()
    print("========== DEEP SORT RESULTS ==========")

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"Total detections: {total_detections}"
    )

    print(
        f"Unique track IDs: "
        f"{len(unique_track_ids)}"
    )

    print(
        f"Average active tracks: "
        f"{average_active_tracks:.2f}"
    )

    print(
        f"Maximum active tracks: "
        f"{maximum_active_tracks}"
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
        f"{average_fps:.2f}"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{p95_latency:.2f} ms"
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

    print()

    print(
        f"Tracked video: "
        f"{absolute_video_path}"
    )

    print(
        f"Detections JSON: "
        f"{os.path.abspath(DETECTIONS_JSON_PATH)}"
    )

    print(
        f"Metrics JSON: "
        f"{absolute_json_path}"
    )

    print(
        f"Log file: "
        f"{absolute_log_path}"
    )

    print("======================================")


# ============================================================
# IMPORTANT
# ============================================================

if __name__ == "__main__":
    main()