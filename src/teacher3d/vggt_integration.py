from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from teacher3d.teacher_packets import build_teacher_packet_path


@dataclass
class VGGTConfig:
    repo_path: Path
    device: str = "cuda"
    hf_repo: str = "facebook/VGGT-1B"
    weights_path: str | None = None
    conf_threshold: float = 1.5
    depth_margin: float = 0.05


class VGGTTeacherRunner:
    def __init__(self, config: VGGTConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._load_upstream()
        self.model = self._build_model().to(self.device).eval()

    def _load_upstream(self) -> None:
        repo_path = self.config.repo_path.resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"VGGT repo not found: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from vggt.models.vggt import VGGT  # type: ignore
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore

        self._VGGT = VGGT
        self._pose_encoding_to_extri_intri = pose_encoding_to_extri_intri

    def _build_model(self):
        if self.config.weights_path:
            model = self._VGGT()
            state_dict = torch.load(self.config.weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            return model
        try:
            return self._VGGT.from_pretrained(self.config.hf_repo)
        except Exception:
            url = f"https://huggingface.co/{self.config.hf_repo}/resolve/main/model.pt"
            model = self._VGGT()
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            model.load_state_dict(state_dict)
            return model

    @staticmethod
    def _load_and_resize(image_path: str | Path, width: int, height: int) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height), Image.LANCZOS)
        tensor = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32") / 255.0).permute(2, 0, 1)
        return tensor.contiguous()

    @staticmethod
    def _pad_to_multiple(images: torch.Tensor, multiple: int = 14) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, height, width = images.shape
        pad_h = (multiple - height % multiple) % multiple
        pad_w = (multiple - width % multiple) % multiple
        if pad_h == 0 and pad_w == 0:
            return images, (0, 0)
        padded = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=1.0)
        return padded, (pad_h, pad_w)

    @staticmethod
    def _unpad_tensor(tensor: torch.Tensor, pad_hw: tuple[int, int]) -> torch.Tensor:
        pad_h, pad_w = pad_hw
        if tensor.ndim < 2 or (pad_h == 0 and pad_w == 0):
            return tensor
        h_slice = slice(None, -pad_h) if pad_h > 0 else slice(None)
        w_slice = slice(None, -pad_w) if pad_w > 0 else slice(None)
        return tensor[..., h_slice, w_slice]

    def predict_pair(self, context_image_path: str, target_image_path: str, width: int, height: int) -> dict[str, torch.Tensor]:
        context = self._load_and_resize(context_image_path, width, height)
        target = self._load_and_resize(target_image_path, width, height)
        images = torch.stack([context, target], dim=0)
        images, pad_hw = self._pad_to_multiple(images)
        images = images.to(self.device)

        with torch.inference_mode():
            if self.device.type == "cuda":
                capability = torch.cuda.get_device_capability(self.device)
                dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    predictions = self.model(images)
            else:
                predictions = self.model(images)

        extrinsic, intrinsic = self._pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        result = {
            "depth": predictions["depth"].detach().float().cpu()[0],
            "depth_conf": predictions["depth_conf"].detach().float().cpu()[0],
            "world_points": predictions["world_points"].detach().float().cpu()[0],
            "world_points_conf": predictions["world_points_conf"].detach().float().cpu()[0],
            "extrinsic": extrinsic.detach().float().cpu()[0],
            "intrinsic": intrinsic.detach().float().cpu()[0],
            "pad_hw": torch.tensor(pad_hw, dtype=torch.long),
        }

        for key in ["depth", "depth_conf", "world_points", "world_points_conf"]:
            value = result[key]
            if key == "world_points":
                value = value.permute(0, 3, 1, 2)
                value = self._unpad_tensor(value, pad_hw).permute(0, 2, 3, 1).contiguous()
            elif key == "depth":
                value = value.permute(0, 3, 1, 2)
                value = self._unpad_tensor(value, pad_hw).permute(0, 2, 3, 1).contiguous()
            else:
                value = self._unpad_tensor(value, pad_hw)
            result[key] = value
        return result


def compute_normals_from_points(world_points: torch.Tensor) -> torch.Tensor:
    # world_points: [H, W, 3]
    dx = world_points[:, 1:, :] - world_points[:, :-1, :]
    dy = world_points[1:, :, :] - world_points[:-1, :, :]
    dx = F.pad(dx.permute(2, 0, 1), (0, 1, 0, 0), mode="replicate").permute(1, 2, 0)
    dy = F.pad(dy.permute(2, 0, 1), (0, 0, 0, 1), mode="replicate").permute(1, 2, 0)
    normals = torch.cross(dx, dy, dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return normals


def project_world_points(world_points: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = world_points.reshape(-1, 3)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    cam = (extrinsic @ points_h.t()).t()
    z = cam[:, 2]
    xy = cam[:, :2] / z.unsqueeze(-1).clamp_min(1e-6)
    uv_h = torch.cat([xy, torch.ones_like(z.unsqueeze(-1))], dim=1)
    pixel = (intrinsic @ uv_h.t()).t()[:, :2]
    return pixel, z


def build_vggt_teacher_packet(
    predictions: dict[str, torch.Tensor],
    conf_threshold: float = 1.5,
    depth_margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    depth = predictions["depth"]  # [2, H, W, 1]
    depth_conf = predictions["depth_conf"]  # [2, H, W]
    world_points = predictions["world_points"]  # [2, H, W, 3]
    world_points_conf = predictions["world_points_conf"]  # [2, H, W]
    extrinsic = predictions["extrinsic"]  # [2, 3, 4]
    intrinsic = predictions["intrinsic"]  # [2, 3, 3]

    context_depth = depth[0, ..., 0]
    context_conf = (depth_conf[0] > conf_threshold).float()
    context_normals = compute_normals_from_points(world_points[0]).permute(2, 0, 1).contiguous()

    target_points = world_points[1]
    target_conf = world_points_conf[1]
    pixel, cam_z = project_world_points(target_points, extrinsic[0], intrinsic[0])

    h, w = context_depth.shape
    u = torch.round(pixel[:, 0]).long()
    v = torch.round(pixel[:, 1]).long()
    valid = torch.isfinite(cam_z)
    valid &= torch.isfinite(pixel).all(dim=1)
    valid &= cam_z > 1e-6
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

    flat_idx = v[valid] * w + u[valid]
    z_vals = cam_z[valid]
    conf_vals = target_conf.reshape(-1)[valid]
    vis_depth_vals = context_depth[v[valid], u[valid]]
    hidden_valid = z_vals > (vis_depth_vals + depth_margin)

    hidden_depth_flat = torch.full((h * w,), float("inf"), dtype=torch.float32)
    hidden_conf_flat = torch.zeros((h * w,), dtype=torch.float32)
    hidden_count_flat = torch.zeros((h * w,), dtype=torch.float32)
    if hidden_valid.any():
        hidden_idx = flat_idx[hidden_valid]
        hidden_z = z_vals[hidden_valid].float()
        hidden_c = conf_vals[hidden_valid].float()
        hidden_depth_flat.scatter_reduce_(0, hidden_idx, hidden_z, reduce="amin", include_self=True)
        hidden_conf_flat.scatter_reduce_(0, hidden_idx, hidden_c, reduce="amax", include_self=True)
        hidden_count_flat.scatter_add_(0, hidden_idx, torch.ones_like(hidden_z))

    hidden_conf_raw = hidden_conf_flat.view(1, h, w)
    hidden_count = hidden_count_flat.view(1, h, w)
    hidden_conf_mask = (hidden_conf_raw > conf_threshold).float()
    hidden_depth = hidden_depth_flat.view(1, h, w)
    hidden_gap = torch.where(
        hidden_count > 0.0,
        (hidden_depth - context_depth.unsqueeze(0)).clamp_min(0.0),
        torch.zeros_like(hidden_depth),
    )
    fallback_depth = (context_depth + 1.0).unsqueeze(0)
    hidden_depth = torch.where(torch.isfinite(hidden_depth), hidden_depth, fallback_depth)

    packet = {
        "teacher_visible_depth": context_depth.unsqueeze(0).float(),
        "teacher_visible_normals": context_normals.float(),
        "teacher_visible_confidence": context_conf.unsqueeze(0).float(),
        "teacher_hidden_depth": hidden_depth.float(),
        "teacher_hidden_confidence": hidden_conf_mask.float(),
        "teacher_hidden_confidence_raw": hidden_conf_raw.float(),
        "teacher_hidden_count": hidden_count.float(),
        "teacher_hidden_gap": hidden_gap.float(),
        "teacher_packet_type": "vggt_pair_projection",
        "teacher_context_extrinsic": extrinsic[0].float(),
        "teacher_context_intrinsic": intrinsic[0].float(),
        "teacher_target_extrinsic": extrinsic[1].float(),
        "teacher_target_intrinsic": intrinsic[1].float(),
    }
    return packet


def save_vggt_packet(packet_root: str | Path, split: str, scene_id: str, context_timestamp: int, target_timestamp: int, packet: dict[str, Any]) -> Path:
    path = build_teacher_packet_path(packet_root, split, scene_id, context_timestamp, target_timestamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packet, path)
    return path
