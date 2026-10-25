"""Gradio demo: detect behaviors on an image, a video, or a live webcam stream."""

import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from .config import resolve_path


def display_names(class_names):
    return [n.replace("_", " ").title() for n in class_names]


class PerformanceTracker:
    """Rolling FPS and per-class detection counts, rendered as a short text report."""

    def __init__(self, names):
        self.names = names
        self.fps = deque(maxlen=30)
        self.history = deque(maxlen=100)
        self.start = time.time()
        self.frames = 0

    def update_fps(self, fps):
        self.fps.append(fps)
        self.frames += 1

    def update_detections(self, class_ids):
        counts = defaultdict(int)
        for c in class_ids:
            if 0 <= c < len(self.names):
                counts[self.names[c]] += 1
        self.history.append(dict(counts))

    def avg_fps(self):
        return sum(self.fps) / len(self.fps) if self.fps else 0.0

    def summary(self):
        totals = defaultdict(int)
        for frame in self.history:
            for name, n in frame.items():
                totals[name] += n
        lines = [
            f"Session: {int(time.time() - self.start)}s | Frames: {self.frames} | Avg FPS: {self.avg_fps():.1f}",
            "",
            "Detected behaviors",
            "=" * 40,
        ]
        for name in self.names:
            if totals.get(name):
                per_frame = totals[name] / len(self.history)
                lines.append(f"{name:15} | {totals[name]:4} total | {per_frame:5.2f}/frame {'#' * min(int(per_frame * 10), 20)}")
        if not totals:
            lines.append("  No detections yet")
        return "\n".join(lines)


def _class_ids(result):
    return [int(c) for c in result.boxes.cls.cpu().numpy()]


def build_demo(cfg: dict, weights=None):
    import gradio as gr
    from ultralytics import YOLO

    weights = Path(weights) if weights else resolve_path(cfg, "weights")
    if not weights.exists():
        raise FileNotFoundError(
            f"No weights at {weights}. Train a model (`behavior-detection train`) or pass --weights "
            "(e.g. models/best_legacy_run.pt)."
        )
    model = YOLO(str(weights))
    names = display_names(cfg["classes"])
    default_conf, default_iou = cfg["app"]["conf"], cfg["app"]["iou"]
    webcam_tracker = PerformanceTracker(names)

    def process_image(image, conf, iou):
        if image is None:
            return None, "No image provided"
        result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]
        counts = defaultdict(int)
        for c in _class_ids(result):
            counts[names[c]] += 1
        text = f"Detections: {len(result.boxes)}\n\n" + "\n".join(f"  {n}: {counts[n]}" for n in names if counts[n])
        return cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), text

    def process_video(path, conf, iou, progress=gr.Progress()):
        if path is None:
            return None, "No video provided"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None, "Could not open the video"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        tracker, n = PerformanceTracker(names), 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
            tracker.update_detections(_class_ids(result))
            writer.write(result.plot())
            n += 1
            if n % 10 == 0:
                progress(n / total, desc=f"Frame {n}/{total}")
        cap.release()
        writer.release()
        tracker.frames = n
        return out_path, tracker.summary()

    def process_webcam(frame, conf, iou):
        if frame is None:
            return None, "Waiting for the webcam..."
        t0 = time.time()
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        result = model.predict(small, conf=conf, iou=iou, verbose=False, max_det=50)[0]
        out = cv2.cvtColor(result.plot(line_width=3), cv2.COLOR_BGR2RGB)
        if scale < 1:
            out = cv2.resize(out, (w, h))
        webcam_tracker.update_detections(_class_ids(result))
        latency = time.time() - t0
        webcam_tracker.update_fps(1 / latency if latency > 0 else 0)
        for i, text in enumerate((f"FPS: {webcam_tracker.avg_fps():.1f}", f"Detections: {len(result.boxes)}", f"Latency: {int(latency * 1000)}ms")):
            cv2.putText(out, text, (15, 40 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        return out, webcam_tracker.summary()

    def reset_webcam():
        webcam_tracker.__init__(names)
        return None, "Statistics reset"

    def sliders(conf):
        return (
            gr.Slider(0.1, 0.9, value=conf, step=0.05, label="Confidence threshold"),
            gr.Slider(0.1, 0.9, value=default_iou, step=0.05, label="IoU threshold (NMS)"),
        )

    with gr.Blocks(title="Student Behavior Detection", theme=gr.themes.Soft(primary_hue="green")) as demo:
        gr.Markdown(f"# Student Behavior Detection (YOLOv8)\nClasses: {' | '.join(names)}")
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        img_in = gr.Image(type="numpy", label="Input image")
                        img_conf, img_iou = sliders(default_conf)
                        img_btn = gr.Button("Analyze", variant="primary")
                    with gr.Column():
                        img_out = gr.Image(label="Detections")
                        img_stats = gr.Textbox(label="Stats", lines=8)
                img_btn.click(process_image, [img_in, img_conf, img_iou], [img_out, img_stats])

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        vid_in = gr.Video(label="Input video")
                        vid_conf, vid_iou = sliders(default_conf)
                        vid_btn = gr.Button("Process", variant="primary")
                    with gr.Column():
                        vid_out = gr.Video(label="Annotated video")
                        vid_stats = gr.Textbox(label="Stats", lines=8)
                vid_btn.click(process_video, [vid_in, vid_conf, vid_iou], [vid_out, vid_stats])

            with gr.Tab("Webcam"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cam_conf, cam_iou = sliders(0.3)
                        reset_btn = gr.Button("Reset statistics")
                    with gr.Column(scale=2):
                        cam = gr.Image(sources=["webcam"], streaming=True, height=600, label="Webcam")
                    with gr.Column(scale=1):
                        cam_stats = gr.Textbox(label="Live stats", lines=18, interactive=False)
                cam.stream(process_webcam, [cam, cam_conf, cam_iou], [cam, cam_stats], show_progress=False, time_limit=None)
                reset_btn.click(reset_webcam, outputs=[cam, cam_stats])
    return demo


def launch(cfg: dict, weights=None, share=False, port=7860):
    build_demo(cfg, weights).queue().launch(share=share, server_port=port)
