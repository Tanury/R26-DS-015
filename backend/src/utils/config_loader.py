from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Supports recursive ``_base_`` inheritance: if present, the base config is
    resolved first and the current file's keys are merged on top. This allows
    chains such as synthetic_test -> training -> preprocessing.

    Example training.yaml:
        _base_: configs/preprocessing.yaml
        training:
          epochs: 20
          ...
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "_base_" in cfg:
        base_path = cfg.pop("_base_")
        base_cfg_path = Path(base_path)
        if not base_cfg_path.is_absolute():
            base_cfg_path = project_root() / base_cfg_path
        base = load_config(base_cfg_path)
        base.pop("_config_path", None)
        cfg = _deep_merge(base, cfg)

    cfg["_config_path"] = str(cfg_path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p


def resolve_data_file(cfg: dict[str, Any], data_key: str, base_path_key: str) -> Path:
    """Resolve a configured data file through its Modal/local artifact directory.

    Configs keep portable defaults such as ``data/splits/foo.csv`` while Modal
    notebooks commonly override ``paths.splits_dir`` to ``/mnt/.../splits``.
    For non-absolute data files, the filename is resolved under the configured
    artifact directory instead of under the repository root.
    """
    p = Path(cfg["data"][data_key])
    if p.is_absolute():
        return p
    return resolve_path(Path(cfg["paths"][base_path_key]) / p.name)


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
