from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REQUIRED_PACKET_KEYS = {
    "teacher_visible_depth",
    "teacher_visible_confidence",
    "teacher_hidden_depth",
    "teacher_hidden_confidence",
}

OPTIONAL_PACKET_KEYS = {
    "teacher_visible_normals",
    "teacher_rendered_target",
    "teacher_hidden_confidence_raw",
    "teacher_hidden_count",
    "teacher_hidden_gap",
}


def build_teacher_packet_path(
    root: str | Path,
    split: str,
    scene_id: str,
    context_timestamp: int | str,
    target_timestamp: int | str,
) -> Path:
    root = Path(root)
    filename = f"{context_timestamp}_{target_timestamp}.pt"
    return root / split / scene_id / filename


def load_teacher_packet(path: str | Path) -> dict[str, Any]:
    packet = torch.load(path, map_location="cpu")
    missing = REQUIRED_PACKET_KEYS.difference(packet.keys())
    if missing:
        raise KeyError(f"Teacher packet {path} is missing keys: {sorted(missing)}")
    return packet
