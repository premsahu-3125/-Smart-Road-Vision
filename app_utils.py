"""
utils.py

Small helpers that didn't belong inside any single module - an FPS tracker,
file/extension validation, and a couple of timestamp formatters used by both
the logger and the report generator.
"""

import os
import time
from datetime import datetime

from config import SUPPORTED_IMAGE_EXT, SUPPORTED_VIDEO_EXT


class FPSTracker:
    """Rolling FPS estimate based on a short window of recent frame times.

    A naive 1/(t_now - t_prev) reading jumps around too much frame to frame
    to be readable in the GUI, so this keeps the last N timestamps and
    averages the gaps instead.
    """

    def __init__(self, window_size=20):
        self.window_size = window_size
        self._timestamps = []

    def tick(self):
        """Call once per processed frame; returns the current smoothed FPS."""
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)

        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self):
        self._timestamps.clear()


def is_supported_image(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_IMAGE_EXT)


def is_supported_video(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_VIDEO_EXT)


def validate_input_path(path: str) -> tuple[bool, str]:
    """Basic sanity checks before we hand a path to OpenCV.

    Returns (is_valid, reason). Keeping the reason string means the GUI can
    show a specific status-bar message instead of a generic "invalid file".
    """
    if not path:
        return False, "No file selected."

    if not os.path.exists(path):
        return False, f"File not found: {path}"

    if os.path.getsize(path) == 0:
        return False, "File appears to be empty or corrupted."

    if not (is_supported_image(path) or is_supported_video(path)):
        ext = os.path.splitext(path)[1] or "(no extension)"
        return False, f"Unsupported format '{ext}'."

    return True, ""


def current_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def current_time_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def clock_display_str() -> str:
    """Used by the GUI's live clock label."""
    return datetime.now().strftime("%d %b %Y  |  %H:%M:%S")


def safe_filename(prefix: str, extension: str) -> str:
    """Builds a collision-safe filename using a timestamp, e.g.

    screenshot_20260706_143210.jpg
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{extension.lstrip('.')}"


def format_seconds(seconds: float) -> str:
    """Human readable duration for the summary panel and reports."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.1f}s"
