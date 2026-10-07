from ultralytics import YOLO


class YOLODetector:

    def __init__(
        self,
        model_path,
        img_size,
        confidence_threshold,
        iou_threshold,
        device
    ):

        print("Loading YOLO11m...")

        self.model = YOLO(model_path)

        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        print("YOLO11m loaded successfully.")


    # ==========================================
    # Run YOLO Detection
    # ==========================================

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        result = results[0]

        detections_for_tracker = []

        frame_detection_data = []

        if result.boxes is not None:

            for box in result.boxes:

                # --------------------------------------
                # Bounding box
                # --------------------------------------

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                # --------------------------------------
                # Confidence
                # --------------------------------------

                confidence = float(
                    box.conf[0].cpu()
                )

                # --------------------------------------
                # Class ID
                # --------------------------------------

                class_id = int(
                    box.cls[0].cpu()
                )

                # --------------------------------------
                # Class name
                # --------------------------------------

                class_name = self.model.names[
                    class_id
                ]

                # --------------------------------------
                # Bounding box width and height
                # --------------------------------------

                width_box = x2 - x1
                height_box = y2 - y1

                # --------------------------------------
                # Deep SORT detection format
                # --------------------------------------

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

                # --------------------------------------
                # Detection information for JSON
                # --------------------------------------

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

        return detections_for_tracker, frame_detection_data