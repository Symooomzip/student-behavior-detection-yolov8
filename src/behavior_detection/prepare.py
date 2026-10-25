"""raw images -> data/prepared (split first, then balance the train split only)."""

import json
import shutil

import yaml

from .augment import balance_train
from .config import resolve_path
from .dataset import class_stats, format_label, load_samples
from .split import SPLITS, split_samples


def prepare(cfg: dict) -> dict:
    """Build the YOLO training layout and return a summary dict (also written to split.json)."""
    names = cfg["classes"]
    raw, out = resolve_path(cfg, "raw"), resolve_path(cfg, "prepared")

    samples, report = load_samples(raw, names)
    for line in report.problems:
        print(f"[warn] skipped invalid label line: {line}")
    if report.unlabeled:
        print(f"[warn] {len(report.unlabeled)} image(s) without usable labels were skipped: {report.unlabeled}")

    splits = split_samples(samples, cfg["split"]["ratios"], cfg["split"]["seed"])

    if out.exists():  # always rebuild from scratch: no stale files, no augment-of-augment
        shutil.rmtree(out)
    for kind in ("images", "labels"):
        for split in SPLITS:
            (out / kind / split).mkdir(parents=True)

    for split, items in splits.items():
        for s in items:
            shutil.copy2(s.image, out / "images" / split / s.image.name)
            (out / "labels" / split / f"{s.stem}.txt").write_text(
                "\n".join(format_label(b) for b in s.boxes) + "\n", encoding="utf-8"
            )

    generated = {}
    if cfg["balance"]["enabled"]:
        generated = balance_train(
            splits["train"],
            len(names),
            cfg["augmentation"],
            cfg["balance"]["target_ratio"],
            cfg["balance"]["seed"],
            out / "images" / "train",
            out / "labels" / "train",
        )

    # absolute `path` so ultralytics finds the data wherever it is launched from
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")

    summary = {
        "seed": cfg["split"]["seed"],
        "ratios": cfg["split"]["ratios"],
        "original_images": {k: len(v) for k, v in splits.items()},
        "augmented_train_images": {names[c]: n for c, n in generated.items()},
        "class_stats_originals": {k: class_stats(v, names) for k, v in splits.items()},
        "files": {k: [s.image.name for s in v] for k, v in splits.items()},
    }
    manifest = resolve_path(cfg, "manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
