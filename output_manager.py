import json
import os


class OutputManager:

    def __init__(
        self,
        output_video,
        output_detections,
        output_metrics
    ):

        self.output_video = output_video
        self.output_detections = output_detections
        self.output_metrics = output_metrics

        self.all_detections = []

    # ==========================================
    # Save Frame Detection Data
    # ==========================================

    def add_frame_detections(
        self,
        frame_number,
        frame_detection_data
    ):

        self.all_detections.append(
            {
                "frame": frame_number,
                "detections": frame_detection_data
            }
        )

    # ==========================================
    # Save Detection JSON
    # ==========================================

    def save_detection_json(self):

        with open(
            self.output_detections,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.all_detections,
                file,
                indent=2
            )

    # ==========================================
    # Save Metrics JSON
    # ==========================================

    def save_metrics(self, metrics):

        with open(
            self.output_metrics,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

    # ==========================================
    # Get Detection Data
    # ==========================================

    def get_all_detections(self):

        return self.all_detections