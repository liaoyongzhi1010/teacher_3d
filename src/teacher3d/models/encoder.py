from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
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
        self.out_channels = channels[-1]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.blocks(image)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class DINOv3ConvNeXtBackbone(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        depths: tuple[int, int, int, int] = (3, 3, 27, 3),
        dims: tuple[int, int, int, int] = (128, 256, 512, 1024),
    ) -> None:
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0]),
            )
        )
        for idx in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[idx]),
                    nn.Conv2d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList(
            [nn.Sequential(*[ConvNeXtBlock(dims[idx]) for _ in range(depths[idx])]) for idx in range(4)]
        )
        self.norm = LayerNorm2d(dims[-1])
        self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        state_dict = torch.load(Path(checkpoint_path), map_location='cpu')
        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_unexpected = {'norms.3.weight', 'norms.3.bias'}
        unexpected = set(incompatible.unexpected_keys) - allowed_unexpected
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                'Failed to load DINOv3 ConvNeXt checkpoint cleanly. '
                f'missing={incompatible.missing_keys}, unexpected={sorted(unexpected)}'
            )

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        features.append(x)
        for idx in range(1, 4):
            x = self.downsample_layers[idx](x)
            x = self.stages[idx](x)
            if idx == 3:
                x = self.norm(x)
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)[-1]


class DINOv3ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        checkpoint_path: str | Path,
        fpn_dim: int = 256,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = DINOv3ConvNeXtBackbone(checkpoint_path=checkpoint_path)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        stage_dims = [128, 256, 512, 1024]
        self.lateral_convs = nn.ModuleList([nn.Conv2d(dim, fpn_dim, kernel_size=1) for dim in stage_dims])
        self.smooth_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                )
                for _ in stage_dims
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(stage_dims), fpn_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fpn_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.out_channels = hidden_dim
        self.register_buffer('pixel_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        normalized = (image - self.pixel_mean) / self.pixel_std
        pyramid = [
            proj(feature)
            for proj, feature in zip(self.lateral_convs, self.backbone.forward_features(normalized))
        ]
        pyramid[-1] = self.smooth_convs[-1](pyramid[-1])
        for idx in range(len(pyramid) - 2, -1, -1):
            pyramid[idx] = self.smooth_convs[idx](
                pyramid[idx]
                + F.interpolate(
                    pyramid[idx + 1],
                    size=pyramid[idx].shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            )

        target_size = pyramid[0].shape[-2:]
        fused = torch.cat(
            [
                pyramid[0],
                *[
                    F.interpolate(level, size=target_size, mode='bilinear', align_corners=False)
                    for level in pyramid[1:]
                ],
            ],
            dim=1,
        )
        fused = self.fuse(fused)
        fused = F.interpolate(fused, size=image.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(fused)


def build_image_encoder(config) -> nn.Module:
    hidden_dim = int(config.model.hidden_dim)
    backbone_config = getattr(config.model, 'backbone', None)
    if backbone_config is None:
        channels = list(getattr(config.model, 'encoder_channels', [32, 64, 96]))
        return ImageEncoder(channels)

    name = str(getattr(backbone_config, 'name', 'simple_conv')).lower()
    if name in {'simple_conv', 'conv'}:
        channels = list(getattr(config.model, 'encoder_channels', [32, 64, 96]))
        return ImageEncoder(channels)
    if name == 'dinov3_convnext':
        checkpoint_path = getattr(backbone_config, 'checkpoint_path', None)
        if not checkpoint_path:
            raise ValueError('model.backbone.checkpoint_path must be set for dinov3_convnext')
        return DINOv3ConvNeXtEncoder(
            hidden_dim=hidden_dim,
            checkpoint_path=checkpoint_path,
            fpn_dim=int(getattr(backbone_config, 'fpn_dim', max(256, hidden_dim))),
            freeze_backbone=bool(getattr(backbone_config, 'freeze', False)),
        )
    raise ValueError(f'Unsupported backbone: {name}')
