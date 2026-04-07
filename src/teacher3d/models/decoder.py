from __future__ import annotations

import torch
from torch import nn


class LayeredGaussianDecoder(nn.Module):
    """Proxy decoder for the V1 scaffold.

    This is not a full 3DGS rasterizer yet. It keeps the visible/hidden split explicit
    and provides a renderable output for smoke tests and early ablations.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.rgb_head = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + 5, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
        visible_features: torch.Tensor,
        hidden_features: torch.Tensor,
        visible_alpha: torch.Tensor,
        hidden_alpha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden_alpha_mean = hidden_alpha.mean(dim=1, keepdim=True)
        decoder_input = torch.cat(
            [image, visible_features, hidden_features, visible_alpha, hidden_alpha_mean],
            dim=1,
        )
        rendered = self.rgb_head(decoder_input)
        return {
            "rendered_target": rendered,
            "hidden_alpha": hidden_alpha_mean,
        }
