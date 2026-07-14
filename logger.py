"""
logger.py

Two responsibilities live here:
  1. Appending one row per completed session to detection_history.csv
  2. Feeding timestamped text lines to the GUI's scrollable log window

Kept separate from report.py because a CSV row is a permanent structured
record, while the text report in report.py is a one-off human-readable
summary someone might attach to an email.
"""

import csv
import os
from typing import Callable, Optional

from config import DETECTION_HISTORY_CSV, CSV_HEADERS, LOGS_DIR
from utils import current_date_str, current_time_str


class DetectionLogger:
    """Handles both the CSV history file and live GUI log callbacks."""

    def __init__(self, gui_log_callback: Optional[Callable[[str], None]] = None):
        # gui_log_callback lets the GUI register a function that receives
        # every log line, so this class doesn't need to know Tkinter exists.
        self.gui_log_callback = gui_log_callback
        self._ensure_csv_exists()

    def _ensure_csv_exists(self):
        if not os.path.exists(DETECTION_HISTORY_CSV):
            with open(DETECTION_HISTORY_CSV, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def log(self, message: str):
        """Send a line to the GUI log window (and stdout, for terminal runs)."""
        line = f"[{current_time_str()}] {message}"
        print(line)
        if self.gui_log_callback:
            try:
                self.gui_log_callback(line)
            except Exception:
                # A GUI hiccup logging a message should never crash detection.
                pass

    def append_history(self, input_source: str, detected_objects: list,
                        total_objects: int, average_confidence: float):
        """Write one row summarizing a finished detection session."""
        objects_str = ", ".join(sorted(set(detected_objects))) if detected_objects else "None"

        row = [
            current_date_str(),
            current_time_str(),
            input_source,
            objects_str,
            total_objects,
            f"{average_confidence:.2f}%",
        ]

        try:
            with open(DETECTION_HISTORY_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.log(f"History row saved -> {os.path.basename(DETECTION_HISTORY_CSV)}")
        except OSError as err:
            self.log(f"Could not write to history CSV: {err}")

    def export_csv(self, destination_path: str) -> bool:
        """Lets the user save a copy of the history file somewhere else,
        e.g. for attaching to an assignment submission.
        """
        try:
            with open(DETECTION_HISTORY_CSV, "r", encoding="utf-8") as src:
                content = src.read()
            with open(destination_path, "w", encoding="utf-8") as dst:
                dst.write(content)
            self.log(f"Exported detection history to {destination_path}")
            return True
        except OSError as err:
            self.log(f"Export failed: {err}")
            return False
