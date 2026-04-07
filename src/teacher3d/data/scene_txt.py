from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from teacher3d.teacher_packets import build_teacher_packet_path, load_teacher_packet


@dataclass(frozen=True)
class FrameRecord:
    timestamp: int
    intrinsics: torch.Tensor
    world_to_camera: torch.Tensor


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _read_image(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    return _pil_to_tensor(image)


def _parse_metadata_file(path: Path, skip_first_line: bool = True) -> list[FrameRecord]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if skip_first_line:
        lines = lines[1:]
    frames: list[FrameRecord] = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        timestamp = int(parts[0])
        values = list(map(float, parts[1:]))
        if len(values) < 18:
            raise ValueError(f"Expected at least 18 values in {path}, got {len(values)}")
        fx, fy, cx, cy = values[:4]
        intrinsics = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        extrinsics = torch.tensor(values[6:18], dtype=torch.float32).view(3, 4)
        world_to_camera = torch.eye(4, dtype=torch.float32)
        world_to_camera[:3, :4] = extrinsics
        frames.append(FrameRecord(timestamp=timestamp, intrinsics=intrinsics, world_to_camera=world_to_camera))
    if not frames:
        raise ValueError(f"No frames parsed from metadata file: {path}")
    return frames


class TxtMetadataSceneDataset(Dataset):
    """Single-image scene dataset backed by per-scene jpg folders and txt camera metadata."""

    def __init__(self, config) -> None:
        self.config = config
        self.data_root = Path(config.data.data_root)
        self.metadata_root = Path(getattr(config.data, "metadata_root", self.data_root))
        self.split = config.data.split
        self.width = int(config.data.image_width)
        self.height = int(config.data.image_height)
        self.min_frame_gap = int(config.data.min_frame_gap)
        self.max_frame_gap = int(config.data.max_frame_gap)
        self.max_scenes = int(getattr(config.data, "max_scenes", -1))
        self.max_samples = int(getattr(config.data, "max_samples", -1))
        self.samples_per_scene = int(getattr(config.data, "samples_per_scene", 4))
        self.skip_first_metadata_line = bool(getattr(config.data, "skip_first_metadata_line", True))
        self.teacher_target_root = getattr(config.data, "teacher_target_root", None)
        self.require_teacher_targets = bool(getattr(config.data, "require_teacher_targets", False))
        self._metadata_cache: dict[str, list[FrameRecord]] = {}

        self.image_split_root = self.data_root / self.split
        self.metadata_split_root = self.metadata_root / self.split
        if not self.image_split_root.exists():
            raise FileNotFoundError(f"Image split root not found: {self.image_split_root}")
        if not self.metadata_split_root.exists():
            raise FileNotFoundError(f"Metadata split root not found: {self.metadata_split_root}")

        split_file_value = getattr(config.data, "split_file", None)
        self.split_file = None if split_file_value in {None, "", "null"} else Path(split_file_value)

        self.scene_ids = self._load_scene_ids()
        if self.max_scenes > 0:
            self.scene_ids = self.scene_ids[: self.max_scenes]
        if not self.scene_ids:
            raise ValueError(f"No scenes found under {self.image_split_root}")

        if self.split_file is not None:
            self.samples = self._load_split_samples(self.split_file)
        else:
            desired = len(self.scene_ids) * self.samples_per_scene
            if self.max_samples > 0:
                desired = min(desired, self.max_samples)
            self.samples = [None] * max(desired, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_scene_ids(self) -> list[str]:
        image_scenes = {path.name for path in self.image_split_root.iterdir() if path.is_dir()}
        metadata_scenes = {path.stem for path in self.metadata_split_root.glob("*.txt")}
        return sorted(image_scenes.intersection(metadata_scenes))

    def _load_metadata(self, scene_id: str) -> list[FrameRecord]:
        if scene_id not in self._metadata_cache:
            metadata_path = self.metadata_split_root / f"{scene_id}.txt"
            self._metadata_cache[scene_id] = _parse_metadata_file(
                metadata_path,
                skip_first_line=self.skip_first_metadata_line,
            )
        return self._metadata_cache[scene_id]

    def _load_split_samples(self, split_file: Path) -> list[dict[str, Any]]:
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        samples: list[dict[str, Any]] = []
        for line in split_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            scene_id = parts[0]
            if scene_id not in self.scene_ids:
                continue
            context_index = int(parts[1])
            target_indices = [int(p) for p in parts[2:]]
            samples.append(
                {
                    "scene_id": scene_id,
                    "context_index": context_index,
                    "target_indices": target_indices,
                }
            )
        if self.max_samples > 0:
            samples = samples[: self.max_samples]
        return samples

    def _choose_pair_for_scene(self, scene_id: str, seed: int) -> tuple[int, int]:
        frames = self._load_metadata(scene_id)
        if len(frames) < 2:
            raise ValueError(f"Scene {scene_id} has fewer than 2 frames")
        rng = random.Random(seed)
        context_index = rng.randrange(len(frames))
        candidate_indices = [
            i for i in range(len(frames))
            if self.min_frame_gap <= abs(i - context_index) <= self.max_frame_gap
        ]
        if not candidate_indices:
            candidate_indices = [i for i in range(len(frames)) if i != context_index]
        target_index = rng.choice(candidate_indices)
        return context_index, target_index

    def _resolve_sample(self, index: int) -> tuple[str, int, int]:
        if self.split_file is not None:
            sample = self.samples[index]
            scene_id = sample["scene_id"]
            target_indices = sample["target_indices"]
            if len(target_indices) == 1:
                target_index = target_indices[0]
            else:
                chooser = random.Random(index)
                target_index = chooser.choice(target_indices)
            return scene_id, sample["context_index"], target_index
        scene_id = self.scene_ids[index % len(self.scene_ids)]
        return scene_id, *self._choose_pair_for_scene(scene_id, seed=index)

    def _image_path(self, scene_id: str, timestamp: int) -> Path:
        return self.image_split_root / scene_id / f"{timestamp}.jpg"

    def _maybe_load_teacher_packet(self, scene_id: str, context_timestamp: int, target_timestamp: int) -> dict[str, Any]:
        if not self.teacher_target_root:
            return {"teacher_available": torch.tensor(0.0, dtype=torch.float32)}
        packet_path = build_teacher_packet_path(
            self.teacher_target_root,
            self.split,
            scene_id,
            context_timestamp,
            target_timestamp,
        )
        if not packet_path.exists():
            if self.require_teacher_targets:
                raise FileNotFoundError(f"Teacher packet not found: {packet_path}")
            return {
                "teacher_available": torch.tensor(0.0, dtype=torch.float32),
                "teacher_path": str(packet_path),
            }
        packet = load_teacher_packet(packet_path)
        packet["teacher_available"] = torch.tensor(1.0, dtype=torch.float32)
        packet["teacher_path"] = str(packet_path)
        return packet

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id, context_index, target_index = self._resolve_sample(index)
        frames = self._load_metadata(scene_id)
        context_frame = frames[context_index]
        target_frame = frames[target_index]

        context_image_path = self._image_path(scene_id, context_frame.timestamp)
        target_image_path = self._image_path(scene_id, target_frame.timestamp)
        image = _read_image(context_image_path, self.width, self.height)
        target_image = _read_image(target_image_path, self.width, self.height)

        sample = {
            "image": image,
            "target_image": target_image,
            "scene_id": scene_id,
            "split": self.split,
            "context_index": torch.tensor(context_index, dtype=torch.long),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "context_timestamp": torch.tensor(context_frame.timestamp, dtype=torch.long),
            "target_timestamp": torch.tensor(target_frame.timestamp, dtype=torch.long),
            "context_intrinsics": context_frame.intrinsics.clone(),
            "target_intrinsics": target_frame.intrinsics.clone(),
            "context_w2c": context_frame.world_to_camera.clone(),
            "target_w2c": target_frame.world_to_camera.clone(),
            "context_image_path": str(context_image_path),
            "target_image_path": str(target_image_path),
        }
        sample.update(
            self._maybe_load_teacher_packet(
                scene_id=scene_id,
                context_timestamp=context_frame.timestamp,
                target_timestamp=target_frame.timestamp,
            )
        )
        return sample
