"""Train / val / test split at the level of source images."""

import random
from typing import Dict, List, Sequence

from .dataset import Sample

SPLITS = ("train", "val", "test")


def split_samples(samples: List[Sample], ratios: Sequence[float], seed: int) -> Dict[str, List[Sample]]:
    """Shuffle (seeded) and cut. Call this on ORIGINAL images, before any augmentation:
    that is what keeps augmented copies of a training image out of val/test."""
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError(f"ratios must be 3 values summing to 1, got {list(ratios)}")
    ordered = sorted(samples, key=lambda s: s.stem)
    random.Random(seed).shuffle(ordered)
    n_train = int(len(ordered) * ratios[0])
    n_val = int(len(ordered) * ratios[1])
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train:n_train + n_val],
        "test": ordered[n_train + n_val:],
    }
