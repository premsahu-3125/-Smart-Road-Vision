"""
gui.py

The Tkinter front end. Video/webcam capture runs on a background thread so
the UI never freezes; frames get pushed onto a queue and the main thread
pulls from it on a timer via root.after(). This is the pattern that avoids
the classic "Tkinter window not responding" freeze you get from running
cv2.VideoCapture reads directly on the main thread.
"""

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import config
from detector import RoadObjectDetector, ModelLoadError
from logger import DetectionLogger
from report import SessionStats, generate_report
from app_utils import (
    FPSTracker,
    validate_input_path,
    is_supported_image,
    clock_display_str,
    safe_filename,
)


class SmartRoadVisionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(config.APP_TITLE)
        self.root.minsize(config.APP_MIN_WIDTH, config.APP_MIN_HEIGHT)
        self.root.configure(bg=config.COLOR_BG)

        config.ensure_project_folders()

        # --- core state -----------------------------------------------
        self.detector = RoadObjectDetector()
        self.logger = DetectionLogger(gui_log_callback=self._append_log_line)
        self.model_ready = False

        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()

        self.current_source_label = "None"
        self.session_stats = SessionStats()
        self.last_frame_bgr = None  # kept around for the screenshot button

        self._build_layout()
        self._bind_shortcuts()
        self._tick_clock()

        # Model load happens after the window is visible so the user isn't
        # staring at a blank frame while YOLOv5 downloads on first run.
        self.root.after(200, self._load_model_async)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", background=config.COLOR_ACCENT,
                         troughcolor=config.COLOR_PANEL)

        header = tk.Frame(self.root, bg=config.COLOR_PANEL, height=56)
        header.pack(side="top", fill="x")

        tk.Label(
            header, text="  Smart Road Vision", bg=config.COLOR_PANEL,
            fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 16, "bold")
        ).pack(side="left", padx=8, pady=8)

        self.clock_label = tk.Label(
            header, text="", bg=config.COLOR_PANEL, fg=config.COLOR_SUBTEXT,
            font=(config.FONT_FAMILY, 10)
        )
        self.clock_label.pack(side="right", padx=16)

        body = tk.Frame(self.root, bg=config.COLOR_BG)
        body.pack(side="top", fill="both", expand=True)

        left_panel = tk.Frame(body, bg=config.COLOR_PANEL, width=230)
        left_panel.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left_panel.pack_propagate(False)

        center_panel = tk.Frame(body, bg=config.COLOR_BG)
        center_panel.pack(side="left", fill="both", expand=True, padx=4, pady=8)

        right_panel = tk.Frame(body, bg=config.COLOR_PANEL, width=280)
        right_panel.pack(side="left", fill="y", padx=(4, 8), pady=8)
        right_panel.pack_propagate(False)

        self._build_controls(left_panel)
        self._build_preview(center_panel)
        self._build_summary_and_log(right_panel)
        self._build_status_bar()

    def _make_button(self, parent, text, command):
        btn = tk.Button(
            parent, text=text, command=command,
            bg=config.COLOR_ACCENT, fg="white", activebackground=config.COLOR_ACCENT_HOVER,
            font=(config.FONT_FAMILY, 10, "bold"), relief="flat", cursor="hand2",
            padx=8, pady=8,
        )
        return btn

    def _build_controls(self, parent):
        tk.Label(
            parent, text="CONTROLS", bg=config.COLOR_PANEL, fg=config.COLOR_SUBTEXT,
            font=(config.FONT_FAMILY, 9, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self._make_button(parent, "Load Image", self.load_image).pack(fill="x", padx=10, pady=4)
        self._make_button(parent, "Load Video", self.load_video).pack(fill="x", padx=10, pady=4)
        self._make_button(parent, "Start Webcam", self.start_webcam).pack(fill="x", padx=10, pady=4)
        self._make_button(parent, "Process Folder", self.process_folder).pack(fill="x", padx=10, pady=4)

        self.pause_btn = self._make_button(parent, "Pause (Space)", self.toggle_pause)
        self.pause_btn.pack(fill="x", padx=10, pady=4)

        self._make_button(parent, "Screenshot (S)", self.take_screenshot).pack(fill="x", padx=10, pady=4)
        self._make_button(parent, "Stop", self.stop_current).pack(fill="x", padx=10, pady=4)

        tk.Label(
            parent, text="CONFIDENCE THRESHOLD", bg=config.COLOR_PANEL,
            fg=config.COLOR_SUBTEXT, font=(config.FONT_FAMILY, 9, "bold")
        ).pack(anchor="w", padx=10, pady=(20, 2))

        self.conf_var = tk.DoubleVar(value=config.DEFAULT_CONFIDENCE)
        self.conf_slider = tk.Scale(
            parent, from_=config.MIN_CONFIDENCE, to=config.MAX_CONFIDENCE,
            resolution=0.05, orient="horizontal", variable=self.conf_var,
            command=self._on_confidence_change, bg=config.COLOR_PANEL,
            fg=config.COLOR_TEXT, troughcolor=config.COLOR_BG,
            highlightthickness=0, activebackground=config.COLOR_ACCENT,
        )
        self.conf_slider.pack(fill="x", padx=10)

        exit_btn = tk.Button(
            parent, text="Exit (Q)", command=self.on_exit,
            bg=config.COLOR_DANGER, fg="white", relief="flat",
            font=(config.FONT_FAMILY, 10, "bold"), cursor="hand2", pady=8,
        )
        exit_btn.pack(fill="x", padx=10, pady=(30, 10), side="bottom")

    def _build_preview(self, parent):
        self.preview_label = tk.Label(parent, bg="black")
        self.preview_label.pack(fill="both", expand=True)
        self._show_placeholder("Load an image, video, or start the webcam to begin.")

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 0))

    def _build_summary_and_log(self, parent):
        tk.Label(
            parent, text="DETECTION SUMMARY", bg=config.COLOR_PANEL,
            fg=config.COLOR_SUBTEXT, font=(config.FONT_FAMILY, 9, "bold")
        ).pack(anchor="w", padx=10, pady=(10, 4))

        self.summary_text = tk.Label(
            parent, text=self._summary_placeholder(), justify="left", anchor="nw",
            bg=config.COLOR_PANEL, fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 10),
        )
        self.summary_text.pack(fill="x", padx=10, pady=4)

        tk.Label(
            parent, text="LIVE DETECTIONS", bg=config.COLOR_PANEL, fg=config.COLOR_SUBTEXT,
            font=(config.FONT_FAMILY, 9, "bold")
        ).pack(anchor="w", padx=10, pady=(16, 4))

        self.detections_list = tk.Listbox(
            parent, bg=config.COLOR_BG, fg=config.COLOR_TEXT, height=8,
            relief="flat", highlightthickness=0, font=(config.FONT_FAMILY, 10),
        )
        self.detections_list.pack(fill="x", padx=10)

        tk.Label(
            parent, text="ACTIVITY LOG", bg=config.COLOR_PANEL, fg=config.COLOR_SUBTEXT,
            font=(config.FONT_FAMILY, 9, "bold")
        ).pack(anchor="w", padx=10, pady=(16, 4))

        log_frame = tk.Frame(parent, bg=config.COLOR_PANEL)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side="right", fill="y")

        self.log_box = tk.Text(
            log_frame, bg=config.COLOR_BG, fg=config.COLOR_SUBTEXT, wrap="word",
            height=10, yscrollcommand=scrollbar.set, relief="flat",
            font=(config.FONT_FAMILY, 9), state="disabled",
        )
        self.log_box.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_box.yview)

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg=config.COLOR_PANEL, height=26)
        bar.pack(side="bottom", fill="x")

        self.status_label = tk.Label(
            bar, text="Status: Initializing model...", bg=config.COLOR_PANEL,
            fg=config.COLOR_SUBTEXT, font=(config.FONT_FAMILY, 9), anchor="w",
        )
        self.status_label.pack(side="left", padx=10, pady=4)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model_async(self):
        def worker():
            try:
                self.detector.initialize_model()
                self.model_ready = True
                self.root.after(0, lambda: self._set_status("Model loaded. Ready."))
                self.logger.log("YOLOv5 model initialized successfully.")
            except ModelLoadError as err:
              error_msg = str(err)
              self.root.after(0, lambda: self._set_status("Model load failed - see log."))
              self.logger.log(error_msg)
              self.root.after(0, lambda msg=error_msg: messagebox.showerror("Model Error", msg))

        threading.Thread(target=worker, daemon=True).start()

    def _require_model(self) -> bool:
        if not self.model_ready:
            messagebox.showwarning("Please wait", "The detection model is still loading.")
            return False
        return True

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    def load_image(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return

        valid, reason = validate_input_path(path)
        if not valid:
            messagebox.showerror("Invalid file", reason)
            return

        self.session_stats = SessionStats(input_source=os.path.basename(path))
        self.session_stats.start_time = time.time()

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Read error", "Could not read image - it may be corrupted.")
            return

        annotated, detections, elapsed = self.detector.process_frame(frame)
        self.session_stats.register_detections(detections)
        self.session_stats.frames_processed = 1
        self.session_stats.end_time = time.time()

        self.last_frame_bgr = annotated
        self._display_frame(annotated)
        self._update_live_detections(detections)
        self._update_summary()
        self._save_output_frame(annotated, prefix="image_result")
        self._finish_session()

    def process_folder(self):
        if not self._require_model():
            return
        folder = filedialog.askdirectory(title="Select a folder of images")
        if not folder:
            return

        images = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if is_supported_image(f)
        ]
        if not images:
            messagebox.showinfo("No images", "No supported image files found in that folder.")
            return

        self.session_stats = SessionStats(input_source=f"Folder: {os.path.basename(folder)}")
        self.session_stats.start_time = time.time()
        self.progress["maximum"] = len(images)
        self.progress["value"] = 0

        def worker():
            for idx, img_path in enumerate(images, start=1):
                frame = cv2.imread(img_path)
                if frame is None:
                    self.logger.log(f"Skipped unreadable file: {os.path.basename(img_path)}")
                    continue

                annotated, detections, _ = self.detector.process_frame(frame)
                self.session_stats.register_detections(detections)
                self.session_stats.frames_processed += 1
                self._save_output_frame(annotated, prefix=f"folder_{idx:03d}")

                self.root.after(0, self._display_frame, annotated.copy())
                self.root.after(0, self._update_live_detections, detections)
                self.root.after(0, lambda i=idx: self.progress.configure(value=i))

            self.session_stats.end_time = time.time()
            self.root.after(0, self._update_summary)
            self.root.after(0, self._finish_session)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Video / webcam handling (shared background-thread capture loop)
    # ------------------------------------------------------------------
    def load_video(self):
        if not self._require_model():
            return
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not path:
            return

        valid, reason = validate_input_path(path)
        if not valid:
            messagebox.showerror("Invalid file", reason)
            return

        self._start_capture_loop(source=path, label=os.path.basename(path))

    def start_webcam(self):
        if not self._require_model():
            return
        self._start_capture_loop(source=0, label="Webcam")

    def _start_capture_loop(self, source, label):
        self.stop_current()  # make sure nothing else is running first

        self.current_source_label = label
        self.session_stats = SessionStats(input_source=label)
        self.session_stats.start_time = time.time()

        self.stop_flag.clear()
        self.pause_flag.clear()
        self.pause_btn.config(text="Pause (Space)")

        self.capture_thread = threading.Thread(
            target=self._capture_worker, args=(source,), daemon=True
        )
        self.capture_thread.start()
        self._set_status(f"Running: {label}")
        self.root.after(30, self._pump_frame_queue)

    def _capture_worker(self, source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.logger.log(f"Could not open source: {source}")
            self.root.after(0, lambda: messagebox.showerror(
                "Camera/Video unavailable",
                "Could not access the webcam or open the video file."
            ))
            return

        fps_tracker = FPSTracker()

        while not self.stop_flag.is_set():
            if self.pause_flag.is_set():
                time.sleep(0.1)
                continue

            ok, frame = cap.read()
            if not ok:
                # End of video file, or webcam read failure.
                break

            try:
                annotated, detections, _ = self.detector.process_frame(frame)
            except Exception as err:
                self.logger.log(f"Frame processing error: {err}")
                continue

            fps = fps_tracker.tick()
            self.session_stats.register_detections(detections)
            self.session_stats.frames_processed += 1

            payload = (annotated, detections, fps)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(payload)

        cap.release()
        self.session_stats.end_time = time.time()
        self.root.after(0, self._update_summary)
        self.root.after(0, self._finish_session)
        self.root.after(0, lambda: self._set_status("Session finished."))

    def _pump_frame_queue(self):
        """Runs on the main thread; pulls the latest processed frame and
        updates the GUI. Re-schedules itself while capture is active.
        """
        try:
            annotated, detections, fps = self.frame_queue.get_nowait()
            self.last_frame_bgr = annotated
            self._display_frame(annotated)
            self._update_live_detections(detections)
            self._set_status(f"Running: {self.current_source_label}  |  FPS: {fps:.1f}")
        except queue.Empty:
            pass

        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.root.after(30, self._pump_frame_queue)

    def toggle_pause(self):
        if self.capture_thread is None or not self.capture_thread.is_alive():
            return
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.pause_btn.config(text="Pause (Space)")
            self._set_status(f"Resumed: {self.current_source_label}")
        else:
            self.pause_flag.set()
            self.pause_btn.config(text="Resume (Space)")
            self._set_status("Paused")

    def stop_current(self):
        self.stop_flag.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=1.0)
        self.capture_thread = None

    # ------------------------------------------------------------------
    # Screenshot / output saving
    # ------------------------------------------------------------------
    def take_screenshot(self):
        if self.last_frame_bgr is None:
            messagebox.showinfo("Nothing to capture", "There is no active frame yet.")
            return
        filename = safe_filename("screenshot", "jpg")
        path = os.path.join(config.SCREENSHOTS_DIR, filename)
        cv2.imwrite(path, self.last_frame_bgr)
        self.logger.log(f"Screenshot saved -> {filename}")
        self._set_status(f"Screenshot saved: {filename}")

    def _save_output_frame(self, frame, prefix):
        filename = safe_filename(prefix, "jpg")
        path = os.path.join(config.OUTPUT_DIR, filename)
        cv2.imwrite(path, frame)

    # ------------------------------------------------------------------
    # Session finishing (report + CSV history)
    # ------------------------------------------------------------------
    def _finish_session(self):
        if self.session_stats.frames_processed == 0:
            return  # nothing to report, avoid noise in the log/history

        try:
            report_path = generate_report(self.session_stats)
            self.logger.log(f"Report generated -> {os.path.basename(report_path)}")
        except IOError as err:
            self.logger.log(str(err))

        self.logger.append_history(
            input_source=self.session_stats.input_source,
            detected_objects=self.session_stats.detected_labels(),
            total_objects=self.session_stats.total_objects,
            average_confidence=self.session_stats.average_confidence,
        )

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
    def _display_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        max_w = self.preview_label.winfo_width() or 800
        max_h = self.preview_label.winfo_height() or 500
        image.thumbnail((max_w, max_h))

        photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # keep a reference, Tkinter needs this

    def _show_placeholder(self, message):
        self.preview_label.configure(image="", text=message, fg=config.COLOR_SUBTEXT,
                                      font=(config.FONT_FAMILY, 12))

    def _update_live_detections(self, detections):
        self.detections_list.delete(0, tk.END)
        for det in detections:
            self.detections_list.insert(
                tk.END, f"{det.label}  ({det.confidence * 100:.0f}%)"
            )

    def _summary_placeholder(self):
        return (
            "Total Objects   : -\n"
            "Unique Classes  : -\n"
            "Processing Time : -\n"
            "Average FPS     : -"
        )

    def _update_summary(self):
        s = self.session_stats
        self.summary_text.config(text=(
            f"Total Objects   : {s.total_objects}\n"
            f"Unique Classes  : {s.unique_classes}\n"
            f"Avg Confidence  : {s.average_confidence:.1f}%\n"
            f"Processing Time : {s.processing_time:.1f}s\n"
            f"Average FPS     : {s.average_fps:.1f}"
        ))

    def _append_log_line(self, line):
        self.log_box.config(state="normal")
        self.log_box.insert(tk.END, line + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")

    def _set_status(self, text):
        self.status_label.config(text=f"Status: {text}")

    def _tick_clock(self):
        self.clock_label.config(text=clock_display_str())
        self.root.after(1000, self._tick_clock)

    def _on_confidence_change(self, value):
        self.detector.set_confidence(float(value))

    # ------------------------------------------------------------------
    # Shortcuts / exit
    # ------------------------------------------------------------------
    def _bind_shortcuts(self):
        self.root.bind(config.KEY_PAUSE, lambda e: self.toggle_pause())
        self.root.bind(config.KEY_SCREENSHOT, lambda e: self.take_screenshot())
        self.root.bind(config.KEY_QUIT, lambda e: self.on_exit())

    def on_exit(self):
        self.stop_current()
        self.root.destroy()
