import os
import time
import statistics

import cv2
import psutil

from config import (
    MODEL_PATH,
    VIDEO_PATH,
    OUTPUT_DIR,
    OUTPUT_VIDEO,
    OUTPUT_DETECTIONS,
    OUTPUT_METRICS,
    OUTPUT_LOG,
    IMG_SIZE,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    DEVICE,
    DEEP_SORT_MAX_AGE,
    DEEP_SORT_N_INIT,
    DEEP_SORT_NMS_MAX_OVERLAP,
    DEEP_SORT_MAX_COSINE_DISTANCE,
    DEEP_SORT_NN_BUDGET,
    DEEP_SORT_EMBEDDER,
    DEEP_SORT_HALF,
    DEEP_SORT_BGR,
    DEEP_SORT_EMBEDDER_GPU,
)

from detector import YOLODetector
from tracker import DeepSORTTracker
from output_manager import OutputManager


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("YOLO11m + Deep SORT Object Tracking")
    print("MODULAR FULL VIDEO RUN")
    print("=" * 60)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Initialize YOLO detector
    # --------------------------------------------------------

    detector = YOLODetector(
        MODEL_PATH,
        IMG_SIZE,
        CONF_THRESHOLD,
        IOU_THRESHOLD,
        DEVICE
    )

    # --------------------------------------------------------
    # Initialize Deep SORT tracker
    # --------------------------------------------------------

    tracker = DeepSORTTracker(
        DEEP_SORT_MAX_AGE,
        DEEP_SORT_N_INIT,
        DEEP_SORT_NMS_MAX_OVERLAP,
        DEEP_SORT_MAX_COSINE_DISTANCE,
        DEEP_SORT_NN_BUDGET,
        DEEP_SORT_EMBEDDER,
        DEEP_SORT_HALF,
        DEEP_SORT_BGR,
        DEEP_SORT_EMBEDDER_GPU
    )

    # --------------------------------------------------------
    # Initialize Output Manager
    # --------------------------------------------------------

    output_manager = OutputManager(
        OUTPUT_VIDEO,
        OUTPUT_DETECTIONS,
        OUTPUT_METRICS
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video: {VIDEO_PATH}"
        )

    video_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    video_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    video_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("Video information:")
    print(f"Width: {video_width}")
    print(f"Height: {video_height}")
    print(f"FPS: {video_fps}")
    print(f"Total frames: {total_video_frames}")
    print()

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        video_fps,
        (video_width, video_height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Unable to create output video: {OUTPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    frame_number = 0

    total_detections = 0

    unique_track_ids = set()

    active_tracks_per_frame = []

    track_lifetimes = {}

    confidence_scores = []

    frame_latencies = []

    process_cpu_values = []

    process_ram_values = []

    system_cpu_values = []

    process = psutil.Process(
        os.getpid()
    )

    start_time = time.time()

    # ========================================================
    # PROCESS COMPLETE VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        frame_start = time.time()

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        (
            detections_for_tracker,
            frame_detection_data
        ) = detector.detect(frame)

        total_detections += len(
            frame_detection_data
        )

        for detection in frame_detection_data:

            confidence_scores.append(
                detection["confidence"]
            )

        # ----------------------------------------------------
        # Store detections
        # ----------------------------------------------------

        output_manager.add_frame_detections(
            frame_number,
            frame_detection_data
        )

        # ----------------------------------------------------
        # Deep SORT tracking
        # ----------------------------------------------------

        tracks = tracker.update(
            detections_for_tracker,
            frame
        )

        active_track_count = 0

        for track in tracks:

            if not track.is_confirmed():
                continue

            if track.time_since_update > 0:
                continue

            active_track_count += 1

            track_id = track.track_id

            unique_track_ids.add(
                track_id
            )

            # ------------------------------------------------
            # Track lifetime
            # ------------------------------------------------

            if track_id not in track_lifetimes:
                track_lifetimes[track_id] = 0

            track_lifetimes[track_id] += 1

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                ltrb
            )

            # ------------------------------------------------
            # Class name
            # ------------------------------------------------

            class_name = "object"

            if track.det_class is not None:
                class_name = str(
                    track.det_class
                )

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Draw ID and class
            # ------------------------------------------------

            label = (
                f"ID: {track_id} "
                f"{class_name}"
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

        # ----------------------------------------------------
        # Frame information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_video_frames}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Active Tracks: {active_track_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Write output frame
        # ----------------------------------------------------

        writer.write(frame)

        # ----------------------------------------------------
        # Performance measurements
        # ----------------------------------------------------

        frame_latency = (
            time.time() - frame_start
        ) * 1000

        frame_latencies.append(
            frame_latency
        )

        process_cpu_values.append(
            process.cpu_percent()
        )

        process_ram_values.append(
            process.memory_info().rss
            / (1024 * 1024)
        )

        system_cpu_values.append(
            psutil.cpu_percent()
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Processed frame "
            f"{frame_number}/{total_video_frames}"
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    total_processing_time = (
        time.time() - start_time
    )

    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    average_fps = (
        frame_number / total_processing_time
        if total_processing_time > 0
        else 0
    )

    average_latency = (
        statistics.mean(frame_latencies)
        if frame_latencies
        else 0
    )

    # --------------------------------------------------------
    # P95 latency
    # --------------------------------------------------------

    if frame_latencies:

        sorted_latencies = sorted(
            frame_latencies
        )

        p95_index = int(
            0.95 * (len(sorted_latencies) - 1)
        )

        p95_latency = sorted_latencies[
            p95_index
        ]

    else:
        p95_latency = 0

    # --------------------------------------------------------
    # Active tracks
    # --------------------------------------------------------

    average_active_tracks = (
        statistics.mean(
            active_tracks_per_frame
        )
        if active_tracks_per_frame
        else 0
    )

    maximum_active_tracks = (
        max(active_tracks_per_frame)
        if active_tracks_per_frame
        else 0
    )

    # --------------------------------------------------------
    # Track lifetime
    # --------------------------------------------------------

    average_track_lifetime = (
        statistics.mean(
            track_lifetimes.values()
        )
        if track_lifetimes
        else 0
    )

    longest_track_lifetime = (
        max(
            track_lifetimes.values()
        )
        if track_lifetimes
        else 0
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    average_confidence = (
        statistics.mean(
            confidence_scores
        )
        if confidence_scores
        else 0
    )

    minimum_confidence = (
        min(confidence_scores)
        if confidence_scores
        else 0
    )

    maximum_confidence = (
        max(confidence_scores)
        if confidence_scores
        else 0
    )

    # --------------------------------------------------------
    # Process CPU
    # --------------------------------------------------------

    average_process_cpu = (
        statistics.mean(
            process_cpu_values
        )
        if process_cpu_values
        else 0
    )

    peak_process_cpu = (
        max(process_cpu_values)
        if process_cpu_values
        else 0
    )

    # --------------------------------------------------------
    # Process RAM
    # --------------------------------------------------------

    average_process_ram = (
        statistics.mean(
            process_ram_values
        )
        if process_ram_values
        else 0
    )

    peak_process_ram = (
        max(process_ram_values)
        if process_ram_values
        else 0
    )

    # --------------------------------------------------------
    # System CPU
    # --------------------------------------------------------

    average_system_cpu = (
        statistics.mean(
            system_cpu_values
        )
        if system_cpu_values
        else 0
    )

    peak_system_cpu = (
        max(system_cpu_values)
        if system_cpu_values
        else 0
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
            "path": os.path.abspath(
                MODEL_PATH
            )
        },

        "configuration": {
            "confidence_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "tracker_config": "Deep SORT",
            "device": DEVICE
        },

        "video": {
            "input": os.path.abspath(
                VIDEO_PATH
            ),
            "width": video_width,
            "height": video_height,
            "fps": video_fps,
            "total_frames": frame_number
        },

        "tracking_statistics": {

            "frame_count": frame_number,

            "total_detections": total_detections,

            "unique_track_ids": len(
                unique_track_ids
            ),

            "average_active_tracks": round(
                average_active_tracks,
                2
            ),

            "maximum_active_tracks":
                maximum_active_tracks,

            "average_track_lifetime": round(
                average_track_lifetime,
                2
            ),

            "longest_track_lifetime":
                longest_track_lifetime,

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

            "total_processing_time_seconds":
                round(
                    total_processing_time,
                    2
                ),

            "average_confidence": round(
                average_confidence,
                4
            ),

            "minimum_confidence": round(
                minimum_confidence,
                4
            ),

            "maximum_confidence": round(
                maximum_confidence,
                4
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

            "video": os.path.abspath(
                OUTPUT_VIDEO
            ),

            "json": os.path.abspath(
                OUTPUT_METRICS
            ),

            "log": os.path.abspath(
                OUTPUT_LOG
            )
        }
    }

    # ========================================================
    # SAVE DETECTION JSON
    # ========================================================

    output_manager.save_detection_json()

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    output_manager.save_metrics(
        metrics
    )

    # ========================================================
    # SAVE LOG
    # ========================================================

    log_content = f"""
========== DEEP SORT LOG ==========

Project: YOLO11m + Deep SORT Object Tracking
Tracker: Deep SORT

========== VIDEO ==========

Input:
{os.path.abspath(VIDEO_PATH)}

Frames processed: {frame_number}
Video FPS: {video_fps}

========== TRACKING ==========

Total detections: {total_detections}
Unique track IDs: {len(unique_track_ids)}

Average active tracks:
{average_active_tracks:.2f}

Maximum active tracks:
{maximum_active_tracks}

Average track lifetime:
{average_track_lifetime:.2f} frames

Longest track lifetime:
{longest_track_lifetime} frames

========== PERFORMANCE ==========

Average FPS:
{average_fps:.2f}

Average latency:
{average_latency:.2f} ms

P95 latency:
{p95_latency:.2f} ms

Total processing time:
{total_processing_time:.2f} seconds

========== CONFIDENCE ==========

Average confidence:
{average_confidence:.4f}

Minimum confidence:
{minimum_confidence:.4f}

Maximum confidence:
{maximum_confidence:.4f}

========== RESOURCES ==========

Average process CPU:
{average_process_cpu:.2f}%

Peak process CPU:
{peak_process_cpu:.2f}%

Average process RAM:
{average_process_ram:.2f} MB

Peak process RAM:
{peak_process_ram:.2f} MB

Average system CPU:
{average_system_cpu:.2f}%

Peak system CPU:
{peak_system_cpu:.2f}%

========== OUTPUTS ==========

Video:
{os.path.abspath(OUTPUT_VIDEO)}

Detection JSON:
{os.path.abspath(OUTPUT_DETECTIONS)}

Metrics JSON:
{os.path.abspath(OUTPUT_METRICS)}

Log:
{os.path.abspath(OUTPUT_LOG)}

===================================
"""

    with open(
        OUTPUT_LOG,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(log_content)

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("========== DEEP SORT RESULTS ==========")
    print("=" * 60)

    print(
        f"Total frames: {frame_number}"
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
        f"Average confidence: "
        f"{average_confidence:.4f}"
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

    print(
        f"Tracked video: "
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

    print(
        f"Log file: "
        f"{OUTPUT_LOG}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()