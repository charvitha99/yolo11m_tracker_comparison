from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORTTracker:

    def __init__(
        self,
        max_age,
        n_init,
        nms_max_overlap,
        max_cosine_distance,
        nn_budget,
        embedder,
        half,
        bgr,
        embedder_gpu
    ):

        print("Initializing Deep SORT...")

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            half=half,
            bgr=bgr,
            embedder_gpu=embedder_gpu
        )

        print("Deep SORT initialized successfully.")

    # ==========================================
    # Update Tracks
    # ==========================================

    def update(self, detections, frame):

        tracks = self.tracker.update_tracks(
            detections,
            frame=frame
        )

        return tracks