"""Bounded local video jobs, cancellation and browser-compatible exports."""
import copy
import json
import math
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

MAX_DURATION = 600
MAX_SAMPLES = 6000
MAX_JOBS = 8
JOB_TTL = 3600
TERMINAL = {"complete", "error", "cancelled"}


class VideoJobs:
    def __init__(self, root, detector):
        self.root = Path(root)
        self.detector = detector
        self.jobs = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="behavior-video")

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job["cancel"].set()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _prune(self):
        expired = [key for key, job in self.jobs.items()
                   if job["state"] in TERMINAL and time.monotonic() - job["touched"] > JOB_TTL]
        for key in expired:
            shutil.rmtree(self.root / key, ignore_errors=True)
            del self.jobs[key]

    def create(self, source, name, sample_fps, iou, export_conf):
        with self.lock:
            self._prune()
            if sum(j["state"] not in TERMINAL for j in self.jobs.values()) >= 2:
                raise ValueError("Deux vidéos sont déjà en cours. Annulez une analyse ou attendez sa fin.")
            if len(self.jobs) >= MAX_JOBS:
                oldest = next((key for key, j in self.jobs.items() if j["state"] in TERMINAL), None)
                if oldest:
                    shutil.rmtree(self.root / oldest, ignore_errors=True)
                    del self.jobs[oldest]
            key = uuid.uuid4().hex
            folder = self.root / key
            folder.mkdir()
            target = folder / ("source" + source.suffix.lower())
            shutil.move(str(source), str(target))
            self.jobs[key] = {"id": key, "name": name, "state": "queued", "progress": 0,
                "processed_frames": 0, "total_frames": 0, "samples": 0, "error": None,
                "cancel": threading.Event(), "touched": time.monotonic()}
            self.pool.submit(self._run, key, target, sample_fps, iou, export_conf)
            return self.status(key)

    def status(self, key):
        with self.lock:
            self._prune()
            job = self.jobs.get(key)
            if not job:
                raise KeyError(key)
            job["touched"] = time.monotonic()
            return copy.deepcopy({k: v for k, v in job.items() if k not in {"cancel", "touched"}})

    def cancel(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if not job:
                raise KeyError(key)
            job["cancel"].set()
            return self.status(key)

    def _update(self, key, **values):
        with self.lock:
            self.jobs[key].update(values, touched=time.monotonic())

    @staticmethod
    def _encoder(destination, width, height, fps):
        import imageio_ffmpeg
        return subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _run(self, key, source, sample_fps, iou, export_conf):
        folder = source.parent
        cap = None
        encoders = []
        cancelled = self.jobs[key]["cancel"]
        try:
            if cancelled.is_set():
                self._update(key, state="cancelled")
                return
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError("Vidéo illisible. Essayez un fichier MP4 ou WebM valide.")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width, height = (int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or not 0 < fps <= 240 or frames <= 0 or min(width, height) <= 0:
                raise ValueError("Les informations de cette vidéo sont invalides.")
            if frames / fps > MAX_DURATION or width * height > 25_000_000:
                raise ValueError("Limite vidéo : 10 minutes et 25 mégapixels par image.")
            stride = max(1, round(fps / sample_fps))
            if math.ceil(frames / stride) > MAX_SAMPLES:
                raise ValueError("Trop d’images à analyser. Réduisez la fréquence d’analyse.")
            scale = min(1, 1280 / max(width, height))
            out_w, out_h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
            for name in ("preview.mp4", "annotated.mp4"):
                encoders.append(self._encoder(folder / name, out_w, out_h, fps))
            self._update(key, state="processing", total_frames=frames, duration=round(frames / fps, 3))
            samples, detections, index = [], [], 0
            started = time.perf_counter()
            while True:
                if cancelled.is_set():
                    self._update(key, state="cancelled")
                    return
                ok, frame = cap.read()
                if not ok:
                    break
                if index / fps > MAX_DURATION:
                    raise ValueError("La vidéo dépasse la limite de 10 minutes.")
                frame = cv2.resize(frame, (out_w, out_h))
                if index % stride == 0:
                    result = self.detector.predict(frame, iou=iou)
                    detections = result["detections"]
                    samples.append({"time": round(index / fps, 4), "frame": index, **result})
                    if len(samples) > MAX_SAMPLES:
                        raise ValueError("La vidéo dépasse la limite d’images analysées.")
                encoders[0].stdin.write(frame.tobytes())
                annotated = frame.copy()
                for detection in detections:
                    if detection["confidence"] < export_conf:
                        continue
                    x1, y1, x2, y2 = [int(v * size) for v, size in zip(
                        detection["box"], [out_w, out_h, out_w, out_h])]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (121, 199, 229), 2)
                    label = f'{detection["class_name"]} {detection["confidence"]:.0%}'
                    cv2.putText(annotated, label, (x1, max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (121, 199, 229), 1, cv2.LINE_AA)
                encoders[1].stdin.write(annotated.tobytes())
                index += 1
                if index % max(1, int(fps)) == 0:
                    self._update(key, processed_frames=index, samples=len(samples), progress=min(0.99, index / frames))
            if not index:
                raise ValueError("Aucune image lisible dans cette vidéo.")
            if index < frames * 0.95:
                raise ValueError("La lecture s’est interrompue : la vidéo semble tronquée.")
            for encoder in encoders:
                encoder.stdin.close()
                if encoder.wait(timeout=60) != 0:
                    raise ValueError("Échec de l’encodage MP4. Essayez une vidéo plus courte.")
            if cancelled.is_set():
                self._update(key, state="cancelled")
                return
            payload = {"name": self.jobs[key]["name"], "fps": fps, "sample_fps": fps / stride,
                "duration": index / fps, "width": out_w, "height": out_h, "total_frames": index,
                "export_confidence": export_conf, "iou": iou, "audio": False, "frames": samples,
                "processing_seconds": round(time.perf_counter() - started, 2)}
            (folder / "results.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            self._update(key, state="complete", progress=1, processed_frames=index, samples=len(samples),
                         processing_seconds=payload["processing_seconds"])
        except Exception as exc:
            self._update(key, state="error", error=str(exc) or "Le traitement a échoué.")
        finally:
            if cap is not None:
                cap.release()
            for encoder in encoders:
                if encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                if encoder.stdin and not encoder.stdin.closed:
                    try:
                        encoder.stdin.close()
                    except OSError:
                        pass
            source.unlink(missing_ok=True)
            if self.jobs[key]["state"] != "complete":
                for name in ("preview.mp4", "annotated.mp4", "results.json"):
                    (folder / name).unlink(missing_ok=True)
