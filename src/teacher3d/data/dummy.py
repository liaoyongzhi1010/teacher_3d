from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader, Dataset


class DummySceneDataset(Dataset):
    def __init__(self, length: int, image_size: int, hidden_proposals: int) -> None:
        self.length = length
        self.image_size = image_size
        self.hidden_proposals = hidden_proposals

    def __len__(self) -> int:
        return self.length

    def _make_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.linspace(-1.0, 1.0, self.image_size)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return xx, yy

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        xx, yy = self._make_grid()
        phase = 0.17 * index
        visible_depth = 1.3 + 0.25 * torch.sin(2.0 * math.pi * (xx + phase))
        visible_depth += 0.15 * torch.cos(2.0 * math.pi * (yy - 0.5 * phase))
        visible_depth = visible_depth.unsqueeze(0)

        grad_x = torch.gradient(visible_depth.squeeze(0), dim=1)[0]
        grad_y = torch.gradient(visible_depth.squeeze(0), dim=0)[0]
        normals = torch.stack((-grad_x, -grad_y, torch.ones_like(grad_x)), dim=0)
        normals = normals / normals.norm(dim=0, keepdim=True).clamp_min(1e-6)

        visible_mask = ((xx.pow(2) + yy.pow(2)) < 0.9).float().unsqueeze(0)
        hidden_mask = ((xx > -0.1) & (yy < 0.35)).float().unsqueeze(0) * (1.0 - visible_mask)

        hidden_depth = visible_depth + 0.6 + 0.1 * torch.sin(3.0 * xx).unsqueeze(0)
        hidden_depth = hidden_depth.repeat(self.hidden_proposals, 1, 1)

        base_rgb = torch.stack(
            [
                0.5 + 0.5 * torch.sin(3.0 * xx + phase),
                0.5 + 0.5 * torch.cos(3.0 * yy - phase),
                visible_mask.squeeze(0),
            ],
            dim=0,
        )
        hidden_rgb = torch.stack(
            [
                0.5 + 0.5 * torch.cos(5.0 * xx - phase),
                0.5 + 0.5 * torch.sin(5.0 * yy + phase),
                0.25 + 0.75 * hidden_mask.squeeze(0),
            ],
            dim=0,
        )
        image = base_rgb * visible_mask + 0.05 * (1.0 - visible_mask)
        target = 0.8 * image + 0.2 * hidden_rgb * hidden_mask

        teacher_visible_confidence = visible_mask.clone()
        teacher_hidden_confidence = hidden_mask.repeat(self.hidden_proposals, 1, 1)

        return {
            "image": image.float(),
            "target_image": target.float(),
            "visible_depth": visible_depth.float(),
            "visible_normals": normals.float(),
            "visible_mask": visible_mask.float(),
            "hidden_depth": hidden_depth.float(),
            "hidden_mask": teacher_hidden_confidence.float(),
            "teacher_visible_depth": visible_depth.float(),
            "teacher_visible_normals": normals.float(),
            "teacher_visible_confidence": teacher_visible_confidence.float(),
            "teacher_hidden_depth": hidden_depth.float(),
            "teacher_hidden_confidence": teacher_hidden_confidence.float(),
        }


def build_dataloader(config, shuffle: bool = True) -> DataLoader:
    dataset = DummySceneDataset(
        length=config.data.length,
        image_size=config.data.image_size,
        hidden_proposals=config.data.hidden_proposals,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
    )
