import os


# ==========================================
# Project Configuration
# ==========================================

MODEL_PATH = "yolo11m.pt"

VIDEO_PATH = "PNNL_Parking_LOT(1).avi"


# ==========================================
# Output Configuration
# ==========================================

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


# ==========================================
# YOLO Configuration
# ==========================================

IMG_SIZE = 640

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.70

DEVICE = "cpu"


# ==========================================
# Deep SORT Configuration
# ==========================================

DEEP_SORT_MAX_AGE = 30

DEEP_SORT_N_INIT = 3

DEEP_SORT_NMS_MAX_OVERLAP = 1.0

DEEP_SORT_MAX_COSINE_DISTANCE = 0.2

DEEP_SORT_NN_BUDGET = None

DEEP_SORT_EMBEDDER = "mobilenet"

DEEP_SORT_HALF = False

DEEP_SORT_BGR = True

DEEP_SORT_EMBEDDER_GPU = False