"""Reading and validating a YOLO-format dataset (one .txt per image: `class cx cy w h`, normalized)."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

Box = Tuple[int, float, float, float, float]  # class_id, cx, cy, w, h


@dataclass
class Sample:
    image: Path
    label: Optional[Path]
    boxes: List[Box] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.image.stem

    @property
    def classes(self) -> set:
        return {b[0] for b in self.boxes}


@dataclass
class LoadReport:
    unlabeled: List[str] = field(default_factory=list)  # image has no label file / no valid box
    problems: List[str] = field(default_factory=list)   # malformed lines that were skipped


def parse_label_file(path: Path, class_names: List[str]) -> Tuple[List[Box], List[str]]:
    """Parse one label file. Accepts numeric ids ('3' or '3.0') or class names ('phone').

    Returns the valid boxes and a list of human-readable problems for the rest.
    """
    boxes, problems = [], []
    names = {n: i for i, n in enumerate(class_names)}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) < 5:
                raise ValueError("expected 5 fields")
            cls = names[parts[0]] if parts[0] in names else int(float(parts[0]))
            cx, cy, w, h = (float(p) for p in parts[1:5])
            if not 0 <= cls < len(class_names):
                raise ValueError(f"class {cls} out of range")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("coordinates not normalized to [0, 1]")
        except (ValueError, KeyError) as e:
            problems.append(f"{path.name}:{n}: {e}")
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes, problems


def load_samples(raw_dir: Path, class_names: List[str]) -> Tuple[List[Sample], LoadReport]:
    """Load every labeled image under raw_dir/{images,labels}. Sorted, so the result is deterministic."""
    images_dir, labels_dir = Path(raw_dir) / "images", Path(raw_dir) / "labels"
    report = LoadReport()
    samples = []
    for image in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label = labels_dir / f"{image.stem}.txt"
        if not label.exists():
            report.unlabeled.append(image.name)
            continue
        boxes, problems = parse_label_file(label, class_names)
        report.problems.extend(problems)
        if not boxes:
            report.unlabeled.append(image.name)
            continue
        samples.append(Sample(image=image, label=label, boxes=boxes))
    return samples, report


def class_stats(samples: List[Sample], class_names: List[str]) -> dict:
    """Per-class count of boxes and of images containing the class."""
    instances, images = Counter(), Counter()
    for s in samples:
        for b in s.boxes:
            instances[b[0]] += 1
        for c in s.classes:
            images[c] += 1
    return {
        name: {"instances": instances[i], "images": images[i]}
        for i, name in enumerate(class_names)
    }


def format_label(box: Box) -> str:
    cls, cx, cy, w, h = box
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
