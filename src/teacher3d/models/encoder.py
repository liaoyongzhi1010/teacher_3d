from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageEncoder(nn.Module):
    def __init__(self, channels: list[int]) -> None:
        super().__init__()
        dims = [3, *channels]
        blocks = []
        for in_channels, out_channels in zip(dims[:-1], dims[1:]):
            blocks.append(ConvBlock(in_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.blocks(image)
