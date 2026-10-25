import copy
import re

import cv2
import numpy as np
import pytest

from behavior_detection.config import load_config
from behavior_detection.dataset import load_samples, parse_label_file
from behavior_detection.prepare import prepare
from behavior_detection.split import split_samples

NAMES = ["distracted", "concentrating", "hand_raised", "phone", "sleeping"]


def test_parse_label_file_accepts_ids_and_names_and_reports_bad_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text(
        "3.0 0.5 0.5 0.2 0.2\n"        # float class id
        "phone 0.4 0.4 0.1 0.1\n"      # class name
        "9 0.5 0.5 0.2 0.2\n"          # class out of range
        "1 1.5 0.5 0.2 0.2\n"          # not normalized
        "1 0.5 0.5\n"                  # too short
    )
    boxes, problems = parse_label_file(f, NAMES)
    assert [b[0] for b in boxes] == [3, 3]
    assert len(problems) == 3


def make_raw(tmp_path, n=40):
    images, labels = tmp_path / "raw" / "images", tmp_path / "raw" / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(images / f"img{i:03d}.jpg"), rng.integers(0, 255, (96, 128, 3), dtype=np.uint8))
        # class 0 is common, class 4 is rare, so balancing has work to do
        cls = 0 if i % 4 else 4
        (labels / f"img{i:03d}.txt").write_text(f"{cls} 0.5 0.5 0.4 0.4\n")
    return tmp_path


def test_split_is_deterministic_and_disjoint(tmp_path):
    samples, _ = load_samples(make_raw(tmp_path) / "raw", NAMES)
    a = split_samples(samples, [0.7, 0.2, 0.1], seed=1)
    b = split_samples(samples, [0.7, 0.2, 0.1], seed=1)
    assert {k: [s.stem for s in v] for k, v in a.items()} == {k: [s.stem for s in v] for k, v in b.items()}
    seen = [s.stem for v in a.values() for s in v]
    assert len(seen) == len(set(seen)) == len(samples)


def test_prepare_never_leaks_augmented_copies_into_val_or_test(tmp_path):
    root = make_raw(tmp_path)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["_root"] = root
    cfg["paths"].update(raw="raw", prepared="prepared", manifest="split.json")
    prepare(cfg)

    prepared = root / "prepared"
    names = {s: {p.stem for p in (prepared / "images" / s).iterdir()} for s in ("train", "val", "test")}
    assert not any("_aug" in n for n in names["val"] | names["test"])

    # every augmented train image comes from an ORIGINAL train image
    origin = lambda n: re.sub(r"_aug\d+$", "", n)
    train_originals = {n for n in names["train"] if "_aug" not in n}
    assert {origin(n) for n in names["train"] if "_aug" in n} <= train_originals
    # and no source image appears in two splits
    originals = {s: {origin(n) for n in v} for s, v in names.items()}
    assert not (originals["train"] & originals["val"] or originals["train"] & originals["test"] or originals["val"] & originals["test"])
    # every image has a matching label file
    for s in names:
        assert names[s] == {p.stem for p in (prepared / "labels" / s).iterdir()}
