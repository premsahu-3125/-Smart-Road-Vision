# Smart Road Vision

Real-time road object detection using YOLOv5 - built as a desktop app with a
Tkinter GUI, supporting images, videos, and live webcam input.

## Project Overview

Smart Road Vision detects objects relevant to autonomous/assisted driving -
cars, motorcycles, buses, trucks, pedestrians, cyclists, traffic lights,
stop signs, and animals - from a still image, a video file, or a live
webcam feed, and presents the results through a dark-themed desktop UI.

## Problem Statement

Self-driving and driver-assistance systems depend on reliably recognizing
the objects around a vehicle in real time. This project demonstrates a
lightweight, self-contained version of that pipeline using a pretrained
YOLOv5 model, without requiring a full autonomous-driving dataset or
custom training run.

## Objectives

- Detect road-relevant objects in real time from multiple input types
- Present bounding boxes, class labels, and confidence scores clearly
- Track and log detection history across sessions
- Provide a usable, configurable GUI rather than a command-line-only tool

## Features

**Core**
- Load a single image or an entire folder of images
- Load a video file or use a live webcam feed
- Adjustable confidence threshold via slider
- Bounding boxes, class labels, and confidence scores drawn on output
- Live FPS counter during video/webcam sessions
- Automatic saving of annotated output frames

**Extra**
- Detection history saved to `logs/detection_history.csv`
- One-click screenshot capture
- Dark mode interface throughout
- End-of-session summary (total objects, unique classes, processing time, average FPS)
- Automatic `.txt` report generated per session in `reports/`
- Folder batch-processing with a progress bar
- Pause / Resume for video and webcam sessions
- Keyboard shortcuts: `Space` = Pause/Resume, `S` = Screenshot, `Q` = Exit
- Live clock displayed in the header

## Technology Stack

| Component        | Technology          |
|------------------|----------------------|
| Language         | Python 3.10+          |
| Detection model  | YOLOv5 (Ultralytics, via torch.hub) |
| Deep learning    | PyTorch                |
| Image processing | OpenCV, Pillow          |
| GUI              | Tkinter                  |
| Numerical ops    | NumPy                    |

## Folder Structure

```
SmartRoadVision/
├── main.py          Entry point
├── gui.py           Tkinter application (all UI + capture threads)
├── detector.py       YOLOv5 wrapper (init, inference, drawing)
├── report.py         Session stats + .txt report generation
├── logger.py         CSV detection history + activity log
├── utils.py           FPS tracker, validation, timestamp helpers
├── config.py           Paths, class list, colors, thresholds, theme
├── requirements.txt
├── README.md
├── assets/            Icons / static assets
├── output/            Auto-saved annotated frames
├── reports/            Auto-generated .txt session reports
├── screenshots/         Manual screenshot captures
└── logs/                detection_history.csv + runtime logs
```

## Installation

1. Make sure Python 3.10 or newer is installed.
2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

The first run will download the YOLOv5s weights automatically via
`torch.hub` (requires an internet connection once; cached afterward).

## How to Run

```
python main.py
```

Then use the left-hand panel to load an image, load a video, start the
webcam, or process an entire folder of images. Adjust the confidence
slider to control how strict detections are.

## Screenshots

*(Add screenshots of the running application here before submission.)*

- `screenshots/main_window.png`
- `screenshots/detection_example.png`

## Future Improvements

- Fine-tune YOLOv5 on an actual road-scene dataset (e.g. BDD100K) for
  better accuracy on Indian traffic conditions specifically
- Add multi-camera support for surround-view detection
- Export session reports as PDF instead of plain text
- Add object tracking (e.g. DeepSORT) instead of per-frame-only detection

## Limitations

- Uses a general-purpose pretrained model (COCO classes), not one trained
  specifically on road/traffic datasets, so accuracy on things like
  region-specific signage is limited
- Webcam performance depends heavily on the host machine's CPU/GPU
- No persistence of settings between runs (confidence threshold resets
  to default on restart)

## Conclusion

Smart Road Vision demonstrates a complete, modular real-time detection
pipeline - from raw input through inference, visualization, logging, and
reporting - wrapped in a usable desktop interface rather than a bare
command-line script.
