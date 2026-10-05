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

VIDEO_FILE = os.path.join(
    RESULTS_DIR,
    "deepsort_tracking.mp4"
)

LOG_FILE = os.path.join(
    RESULTS_DIR,
    "deepsort.log"
)


# ==========================================
# Create absolute paths
# ==========================================

video_path = os.path.abspath(
    VIDEO_FILE
)

json_path = os.path.abspath(
    METRICS_FILE
)

log_path = os.path.abspath(
    LOG_FILE
)


# ==========================================
# Read existing metrics JSON
# ==========================================

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8"
) as file:

    metrics = json.load(file)


# ==========================================
# Update output paths
# ==========================================

metrics["outputs"] = {
    "video": video_path,
    "json": json_path,
    "log": log_path
}


# ==========================================
# Save updated JSON
# ==========================================

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================
# Create log file
# ==========================================

log_content = f"""
========== DEEP SORT LOG ==========

Project: YOLO11m + Deep SORT Object Tracking
Tracker: Deep SORT

========== OUTPUTS ==========
Video: {video_path}
JSON: {json_path}
Log: {log_path}

===================================
"""


with open(
    LOG_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(log_content)


print("===================================")
print("Deep SORT output files updated.")
print()
print(f"JSON: {json_path}")
print(f"Log : {log_path}")
print("===================================")