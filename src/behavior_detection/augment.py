"""Offline augmentation used to balance the classes of the TRAIN split."""

import random
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2

from .dataset import Sample, format_label


def build_pipeline(aug: dict) -> A.Compose:
    return A.Compose(
        [
            # lighting variation (different rooms / times of day)
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=aug["brightness_contrast"],
                        contrast_limit=aug["brightness_contrast"],
                        p=1.0,
                    ),
                    A.RandomGamma(gamma_limit=tuple(aug["gamma"]), p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=aug["hsv"]["hue"],
                        sat_shift_limit=aug["hsv"]["sat"],
                        val_shift_limit=aug["hsv"]["val"],
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            # sensor noise / blur (cheap classroom cameras)
            A.OneOf(
                [
                    A.GaussNoise(std_range=tuple(aug["noise_std"]), p=1.0),
                    A.GaussianBlur(blur_limit=tuple(aug["blur_limit"]), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
                ],
                p=0.3,
            ),
            # small camera shifts. No flip: it would mirror the scene.
            A.Affine(
                translate_percent={"x": (-aug["shift"], aug["shift"]), "y": (-aug["shift"], aug["shift"])},
                scale=(1 - aug["scale"], 1 + aug["scale"]),
                rotate=(-aug["rotate"], aug["rotate"]),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.4,
            ),
            # video compression artefacts
            A.ImageCompression(quality_range=tuple(aug["jpeg_quality"]), p=0.3),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.4),
    )


def augment_sample(sample: Sample, pipeline: A.Compose, out_image: Path, out_label: Path) -> bool:
    """Write one augmented copy of `sample`. Returns False if the image is unreadable
    or every box was lost to the transform."""
    image = cv2.imread(str(sample.image))
    if image is None:
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = pipeline(
        image=image,
        bboxes=[list(b[1:]) for b in sample.boxes],
        class_labels=[b[0] for b in sample.boxes],
    )
    boxes = []
    for (cx, cy, w, h), cls in zip(out["bboxes"], out["class_labels"]):
        cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
        if w > 0 and h > 0:
            boxes.append((int(cls), cx, cy, w, h))
    if not boxes:
        return False
    cv2.imwrite(str(out_image), cv2.cvtColor(out["image"], cv2.COLOR_RGB2BGR))
    out_label.write_text("\n".join(format_label(b) for b in boxes) + "\n", encoding="utf-8")
    return True


def balance_train(
    train: List[Sample],
    n_classes: int,
    aug: dict,
    target_ratio: float,
    seed: int,
    out_images: Path,
    out_labels: Path,
) -> Dict[int, int]:
    """Bring every class up to target_ratio x (images of the most common class) by augmenting
    train images that contain it. Sources are always original images, never earlier copies.

    Returns {class_id: number of augmented images generated for it}.
    """
    rng = random.Random(seed)
    # fix numpy/albumentations randomness too, so a re-run reproduces the same files
    pipeline = build_pipeline(aug)
    pipeline.set_random_seed(seed)

    by_class = {c: [s for s in train if c in s.classes] for c in range(n_classes)}
    present = [len(v) for v in by_class.values() if v]
    if not present:
        return {}
    target = int(max(present) * target_ratio)

    generated: Dict[int, int] = {}
    counter = 0
    for cls, sources in by_class.items():
        shortage = target - len(sources)
        if shortage <= 0 or not sources:
            continue
        made, attempts = 0, 0
        while made < shortage and attempts < shortage * 3:
            attempts += 1
            src = rng.choice(sources)
            name = f"{src.stem}_aug{counter}"
            if augment_sample(src, pipeline, out_images / f"{name}{src.image.suffix}", out_labels / f"{name}.txt"):
                made += 1
                counter += 1
        generated[cls] = made
    return generated
