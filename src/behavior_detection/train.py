"""Fine-tune YOLOv8 on the prepared dataset."""

import shutil

from .config import resolve_path


def train(cfg: dict, device=None) -> str:
    from ultralytics import YOLO  # imported here so the data tooling works without torch

    data_yaml = resolve_path(cfg, "prepared") / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Run `behavior-detection prepare` first.")

    hyp = dict(cfg["train"])
    model = YOLO(hyp.pop("model"))
    name = hyp.pop("name")
    model.train(
        data=str(data_yaml),
        project=str(resolve_path(cfg, "runs")),
        name=name,
        exist_ok=True,
        pretrained=True,
        device=device,
        **hyp,
    )

    best = resolve_path(cfg, "runs") / name / "weights" / "best.pt"
    target = resolve_path(cfg, "weights")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    return str(target)
