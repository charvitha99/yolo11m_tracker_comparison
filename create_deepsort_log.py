import json
import os


# ==========================================
# Paths
# ==========================================

RESULTS_DIR = "results"

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "deepsort_metrics.json"
)

LOG_FILE = os.path.join(
    RESULTS_DIR,
    "deepsort.log"
)


# ==========================================
# Read existing metrics
# ==========================================

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8"
) as file:

    metrics = json.load(file)


# ==========================================
# Extract values
# ==========================================

project = metrics["project"]
model = metrics["model"]
configuration = metrics["configuration"]
video = metrics["video"]
tracking = metrics["tracking_statistics"]
resources = metrics["resources"]


# ==========================================
# Create log
# ==========================================

with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as log:

    log.write(
        "========== DEEP SORT LOG ==========\n\n"
    )

    log.write(
        f"Project: {project['name']}\n"
    )

    log.write(
        f"Tracker: {project['tracker']}\n\n"
    )

    log.write(
        "========== MODEL ==========\n"
    )

    log.write(
        f"Model: {model['name']}\n"
    )

    log.write(
        f"Format: {model['format']}\n"
    )

    log.write(
        f"Path: {model['path']}\n\n"
    )

    log.write(
        "========== CONFIGURATION ==========\n"
    )

    log.write(
        f"Confidence threshold: "
        f"{configuration['confidence_threshold']}\n"
    )

    log.write(
        f"IoU threshold: "
        f"{configuration['iou_threshold']}\n"
    )

    log.write(
        f"Tracker config: "
        f"{configuration['tracker_config']}\n"
    )

    log.write(
        f"Device: "
        f"{configuration['device']}\n\n"
    )

    log.write(
        "========== VIDEO ==========\n"
    )

    log.write(
        f"Input: {video['input']}\n"
    )

    log.write(
        f"Resolution: "
        f"{video['width']} x {video['height']}\n"
    )

    log.write(
        f"FPS: {video['fps']}\n"
    )

    log.write(
        f"Total frames: "
        f"{video['total_frames']}\n\n"
    )

    log.write(
        "========== TRACKING STATISTICS ==========\n"
    )

    log.write(
        f"Frame count: "
        f"{tracking['frame_count']}\n"
    )

    log.write(
        f"Total detections: "
        f"{tracking['total_detections']}\n"
    )

    log.write(
        f"Unique track IDs: "
        f"{tracking['unique_track_ids']}\n"
    )

    log.write(
        f"Average active tracks: "
        f"{tracking['average_active_tracks']}\n"
    )

    log.write(
        f"Maximum active tracks: "
        f"{tracking['maximum_active_tracks']}\n"
    )

    log.write(
        f"Average track lifetime: "
        f"{tracking['average_track_lifetime']}\n"
    )

    log.write(
        f"Longest track lifetime: "
        f"{tracking['longest_track_lifetime']}\n"
    )

    log.write(
        f"Average FPS: "
        f"{tracking['average_fps']}\n"
    )

    log.write(
        f"Average latency: "
        f"{tracking['average_latency_ms']} ms\n"
    )

    log.write(
        f"P95 latency: "
        f"{tracking['p95_latency_ms']} ms\n"
    )

    log.write(
        f"Total processing time: "
        f"{tracking['total_processing_time_seconds']} seconds\n\n"
    )

    log.write(
        "========== RESOURCES ==========\n"
    )

    log.write(
        f"Average process CPU: "
        f"{resources['process']['average_cpu_percent']}%\n"
    )

    log.write(
        f"Peak process CPU: "
        f"{resources['process']['peak_cpu_percent']}%\n"
    )

    log.write(
        f"Average process RAM: "
        f"{resources['process']['average_ram_mb']} MB\n"
    )

    log.write(
        f"Peak process RAM: "
        f"{resources['process']['peak_ram_mb']} MB\n"
    )

    log.write(
        f"Average system CPU: "
        f"{resources['system']['average_cpu_percent']}%\n"
    )

    log.write(
        f"Peak system CPU: "
        f"{resources['system']['peak_cpu_percent']}%\n\n"
    )

    log.write(
        "========== OUTPUTS ==========\n"
    )

    log.write(
        f"Video: "
        f"{os.path.abspath('results/deepsort_tracking.mp4')}\n"
    )

    log.write(
        f"JSON: "
        f"{os.path.abspath('results/deepsort_metrics.json')}\n"
    )

    log.write(
        f"Log: "
        f"{os.path.abspath('results/deepsort.log')}\n"
    )

    log.write(
        "\n===================================\n"
    )


print("Deep SORT log created successfully.")
print(
    f"Log file: {os.path.abspath(LOG_FILE)}"
)