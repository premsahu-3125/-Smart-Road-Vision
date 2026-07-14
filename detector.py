"""
detector.py

Wraps the Ultralytics YOLOv5 model behind our own small class instead of
calling their detect.py directly. RoadObjectDetector is the only thing the
rest of the app talks to - it doesn't need to know torch.hub exists.
"""

import time

import cv2
import torch

from config import (
    YOLO_REPO,
    YOLO_WEIGHTS,
    MODEL_CACHE_DIR,
    ROAD_RELEVANT_CLASSES,
    CLASS_COLORS,
    DEFAULT_BOX_COLOR,
    DEFAULT_CONFIDENCE,
)


class ModelLoadError(Exception):
    """Raised when the YOLO weights can't be fetched or initialized."""
    pass


class Detection:
    """Plain data holder for a single detected object in a frame."""

    __slots__ = ("label", "confidence", "box")

    def __init__(self, label, confidence, box):
        self.label = label
        self.confidence = confidence  # 0.0 - 1.0
        self.box = box  # (x1, y1, x2, y2)


class RoadObjectDetector:
    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE):
        self.model = None
        self.confidence_threshold = confidence_threshold
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize_model(self):
        """Downloads (first run only) and loads YOLOv5s via torch.hub.

        Wrapped in its own try/except so a network hiccup or a missing
        weights file produces a clear ModelLoadError instead of a raw
        traceback the GUI can't do anything useful with.
        """
        try:
            torch.hub.set_dir(MODEL_CACHE_DIR)
            self.model = torch.hub.load(
                YOLO_REPO, YOLO_WEIGHTS, pretrained=True, verbose=False
            )
            self.model.to(self._device)
            self.model.conf = self.confidence_threshold
            return True
        except Exception as err:
            raise ModelLoadError(
                f"Failed to load YOLOv5 weights ({YOLO_WEIGHTS}). "
                f"Check your internet connection or cached weights. Details: {err}"
            )

    def set_confidence(self, value: float):
        self.confidence_threshold = value
        if self.model is not None:
            self.model.conf = value

    def detect_objects(self, frame) -> list:
        """Runs inference on a single BGR frame and returns a filtered list
        of Detection objects, keeping only classes relevant to a road scene.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized - call initialize_model() first.")

        # YOLOv5 expects RGB, OpenCV gives us BGR by default.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(rgb_frame, size=640)

        detections = []
        for *box, conf, cls_id in results.xyxy[0].tolist():
            label = self.model.names[int(cls_id)]

            if label not in ROAD_RELEVANT_CLASSES:
                continue
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            detections.append(Detection(label, float(conf), (x1, y1, x2, y2)))

        return detections

    def draw_boxes(self, frame, detections: list):
        """Draws bounding boxes + label/confidence text directly onto frame
        (in place) and returns it for chaining.
        """
        for det in detections:
            x1, y1, x2, y2 = det.box
            color = CLASS_COLORS.get(det.label, DEFAULT_BOX_COLOR)
            caption = f"{det.label} {det.confidence * 100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (text_w, text_h), baseline = cv2.getTextSize(
                caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_bg_y1 = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(frame, (x1, label_bg_y1), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(
                frame, caption, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        return frame

    def process_frame(self, frame):
        """Convenience wrapper: detect + draw + timing, used heavily by the
        webcam/video loops in gui.py so they don't repeat this every time.
        """
        start = time.time()
        detections = self.detect_objects(frame)
        annotated = self.draw_boxes(frame, detections)
        elapsed = time.time() - start
        return annotated, detections, elapsed
