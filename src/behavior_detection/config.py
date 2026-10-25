"""Loads configs/default.yaml and resolves the paths it declares."""

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
if not DEFAULT_CONFIG.exists():  # non-editable install: fall back to the working directory
    DEFAULT_CONFIG = Path.cwd() / "configs" / "default.yaml"


def load_config(path=None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # paths in the file are relative to the repo root (the parent of configs/)
    cfg["_root"] = path.resolve().parent.parent
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Absolute path for an entry of cfg['paths']."""
    return (cfg["_root"] / cfg["paths"][key]).resolve()
