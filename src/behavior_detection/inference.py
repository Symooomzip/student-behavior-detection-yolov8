"""Shared, serialized YOLO inference for image, video and camera requests."""
import threading
import time
from pathlib import Path

import numpy as np

LABELS = {"distracted": "Distrait", "concentrating": "Concentré", "hand_raised": "Main levée",
          "phone": "Téléphone", "sleeping": "Sommeil"}
MIN_CONFIDENCE = 0.1


class Detector:
    def __init__(self, weights, classes, device=None):
        self.weights = Path(weights)
        self.classes = classes
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def predict(self, frame, iou=0.45):
        with self._lock:
            if self._model is None:
                if not self.weights.is_file():
                    raise FileNotFoundError("Modèle introuvable. Vérifiez le chemin --weights.")
                from ultralytics import YOLO
                model = YOLO(str(self.weights))
                if [model.names[i] for i in range(len(model.names))] != self.classes:
                    raise ValueError("Les classes du modèle ne correspondent pas à configs/default.yaml.")
                self._model = model
            started = time.perf_counter()
            result = self._model.predict(frame, conf=MIN_CONFIDENCE, iou=iou, imgsz=640,
                                         max_det=100, verbose=False, device=self.device)[0]
            detections = []
            height, width = frame.shape[:2]
            for index, row in enumerate(result.boxes.data.cpu().numpy()):
                x1, y1, x2, y2, confidence, class_id = row[:6]
                class_id = int(class_id)
                detections.append({"id": index, "class_id": class_id, "class_name": self.classes[class_id],
                    "confidence": round(float(confidence), 4),
                    "box": [round(float(v), 6) for v in np.clip(
                        [x1 / width, y1 / height, x2 / width, y2 / height], 0, 1)]})
            return {"width": width, "height": height, "detections": detections,
                    "inference_ms": round((time.perf_counter() - started) * 1000, 1)}
