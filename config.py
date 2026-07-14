"""
config.py

Everything that other modules would otherwise hardcode lives here instead -
folder paths, the subset of COCO classes we actually care about for a road
scene, default thresholds, and the color table used when drawing boxes.

Keeping this separate meant that when I changed the confidence default
from 0.35 to 0.45 halfway through testing, I only had to touch one line.
"""

import os

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

REQUIRED_DIRS = [ASSETS_DIR, OUTPUT_DIR, REPORTS_DIR, SCREENSHOTS_DIR, LOGS_DIR]


def ensure_project_folders():
    """Create the working folders on first launch if they aren't already there.

    Doing this at startup instead of assuming the folders exist saves a
    confusing FileNotFoundError the first time someone clones the repo.
    """
    for folder in REQUIRED_DIRS:
        os.makedirs(folder, exist_ok=True)


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
# Using the small variant - good enough accuracy for a demo and it loads in
# a couple of seconds on a laptop CPU, which matters when you're re-running
# this twenty times while debugging the GUI.
YOLO_REPO = "ultralytics/yolov5"
YOLO_WEIGHTS = "yolov5s"
MODEL_CACHE_DIR = os.path.join(BASE_DIR, ".torch_cache")

DEFAULT_CONFIDENCE = 0.45
MIN_CONFIDENCE = 0.10
MAX_CONFIDENCE = 0.95

# ---------------------------------------------------------------------------
# Classes relevant to a road / driving scene
# ---------------------------------------------------------------------------
# YOLOv5's default weights are trained on the full 80-class COCO set. We only
# want the ones that matter on a road, so detections outside this list get
# filtered out before they ever reach the GUI. Names must match COCO exactly.
ROAD_RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
}

# A rough color per class so the same object type is always drawn the same
# color across frames - makes the output easier to read than random colors.
CLASS_COLORS = {
    "person": (66, 135, 245),        # blue
    "bicycle": (255, 196, 0),        # amber
    "car": (52, 199, 89),            # green
    "motorcycle": (255, 149, 0),     # orange
    "bus": (191, 90, 242),           # purple
    "truck": (255, 69, 58),          # red
    "traffic light": (255, 214, 10), # yellow
    "stop sign": (220, 20, 60),      # crimson
    "dog": (0, 199, 190),            # teal
    "cat": (0, 199, 190),
    "horse": (0, 199, 190),
    "cow": (0, 199, 190),
    "sheep": (0, 199, 190),
}
DEFAULT_BOX_COLOR = (0, 255, 255)  # fallback if a class slips through without a color

# ---------------------------------------------------------------------------
# File support
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
SUPPORTED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ---------------------------------------------------------------------------
# GUI appearance (dark theme)
# ---------------------------------------------------------------------------
APP_TITLE = "Smart Road Vision - Real-Time Road Object Detection"
APP_MIN_WIDTH = 1180
APP_MIN_HEIGHT = 720

COLOR_BG = "#1b1e26"
COLOR_PANEL = "#242832"
COLOR_ACCENT = "#3ba3ff"
COLOR_ACCENT_HOVER = "#5cb4ff"
COLOR_TEXT = "#e6e9ef"
COLOR_SUBTEXT = "#9aa1b1"
COLOR_SUCCESS = "#34c759"
COLOR_WARNING = "#ff9500"
COLOR_DANGER = "#ff453a"

FONT_FAMILY = "Segoe UI"

# ---------------------------------------------------------------------------
# Keyboard shortcuts (used by gui.py's key bindings)
# ---------------------------------------------------------------------------
KEY_PAUSE = "<space>"
KEY_SCREENSHOT = "s"
KEY_QUIT = "q"

# ---------------------------------------------------------------------------
# Logging / history
# ---------------------------------------------------------------------------
DETECTION_HISTORY_CSV = os.path.join(LOGS_DIR, "detection_history.csv")
CSV_HEADERS = [
    "Date",
    "Time",
    "Input Source",
    "Detected Objects",
    "Total Objects",
    "Average Confidence",
]
