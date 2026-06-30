"""Artifact persistence helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def ensure_run_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_config(output_dir: str, config: Mapping[str, object]) -> Path:
    path = ensure_run_dir(output_dir) / "run_config.json"
    path.write_text(json.dumps(dict(config), indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_model(model: nn.Module, output_dir: str, filename: str = "final_model.pt") -> Path:
    path = ensure_run_dir(output_dir) / filename
    torch.save(model.state_dict(), path)
    return path
