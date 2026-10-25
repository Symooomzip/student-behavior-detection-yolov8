"""Evaluate a weights file on the val and test splits."""

import json
from pathlib import Path

from .config import resolve_path


def evaluate(cfg: dict, weights=None, device=None) -> dict:
    from ultralytics import YOLO

    weights = Path(weights) if weights else resolve_path(cfg, "weights")
    data_yaml = resolve_path(cfg, "prepared") / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run `behavior-detection prepare` first.")

    model = YOLO(str(weights))
    ev, results = cfg["evaluate"], {}
    for split in ("val", "test"):
        m = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=cfg["train"]["imgsz"],
            batch=cfg["train"]["batch"],
            conf=ev["conf"],
            iou=ev["iou"],
            device=device,
            plots=False,
            project=str(resolve_path(cfg, "runs")),
            name=f"eval_{split}",
            exist_ok=True,
        )
        results[split] = {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
        }
        print(f"{split:5} " + "  ".join(f"{k}={v}" for k, v in results[split].items()))

    out = resolve_path(cfg, "runs") / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": str(weights), **results}, indent=2), encoding="utf-8")
    return results
