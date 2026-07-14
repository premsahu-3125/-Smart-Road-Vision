"""
report.py

Builds the human-readable .txt report that gets written to reports/ after
every completed session (image, video, or webcam run). Separate from
logger.py because this is meant to be read by a person, not parsed by code.
"""

import os
from dataclasses import dataclass, field

from config import REPORTS_DIR
from utils import current_date_str, current_time_str, safe_filename, format_seconds


@dataclass
class SessionStats:
    """Accumulates numbers over a session so gui.py doesn't have to juggle
    a dozen separate counters by hand.
    """
    input_source: str = "Unknown"
    total_objects: int = 0
    confidence_sum: float = 0.0
    class_counts: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    frames_processed: int = 0

    def register_detections(self, detections):
        for det in detections:
            self.total_objects += 1
            self.confidence_sum += det.confidence
            self.class_counts[det.label] = self.class_counts.get(det.label, 0) + 1

    @property
    def average_confidence(self) -> float:
        if self.total_objects == 0:
            return 0.0
        return (self.confidence_sum / self.total_objects) * 100

    @property
    def unique_classes(self) -> int:
        return len(self.class_counts)

    @property
    def processing_time(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def average_fps(self) -> float:
        if self.processing_time <= 0:
            return 0.0
        return self.frames_processed / self.processing_time

    def detected_labels(self) -> list:
        return list(self.class_counts.keys())


def generate_report(stats: SessionStats) -> str:
    """Writes a formatted .txt report and returns the file path.

    Kept as plain text rather than anything fancier since it's meant to be
    quickly skimmed or pasted into an assignment submission.
    """
    filename = safe_filename("session_report", "txt")
    path = os.path.join(REPORTS_DIR, filename)

    lines = [
        "=" * 55,
        "SMART ROAD VISION - SESSION REPORT",
        "=" * 55,
        f"Date            : {current_date_str()}",
        f"Time            : {current_time_str()}",
        f"Input Source    : {stats.input_source}",
        "-" * 55,
        f"Total Objects   : {stats.total_objects}",
        f"Unique Classes  : {stats.unique_classes}",
        f"Avg Confidence  : {stats.average_confidence:.2f}%",
        f"Processing Time : {format_seconds(stats.processing_time)}",
        f"Average FPS     : {stats.average_fps:.1f}",
        "-" * 55,
        "Class-wise Breakdown:",
    ]

    if stats.class_counts:
        for label, count in sorted(stats.class_counts.items(), key=lambda x: -x[1]):
            lines.append(f"   - {label:<15}: {count}")
    else:
        lines.append("   No objects detected in this session.")

    lines.append("=" * 55)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except OSError as err:
        raise IOError(f"Could not write report file: {err}")

    return path
