from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class VisibleBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        enable_confidence_bias: bool = False,
        enable_separate_render_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.enable_confidence_bias = enable_confidence_bias
        self.enable_separate_render_alpha = enable_separate_render_alpha
        self.depth_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.normal_head = nn.Conv2d(hidden_dim, 3, kernel_size=1)
        self.confidence_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        if self.enable_confidence_bias:
            self.confidence_bias_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
            )
            nn.init.zeros_(self.confidence_bias_head[1].weight)
            nn.init.zeros_(self.confidence_bias_head[1].bias)
        else:
            self.confidence_bias_head = None
        self.alpha_head = nn.Conv2d(hidden_dim, 1, kernel_size=1) if self.enable_separate_render_alpha else None
        self.feature_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        visible_depth = F.softplus(self.depth_head(features)) + 1e-3
        visible_normals = self.normal_head(features)
        visible_normals = visible_normals / visible_normals.norm(dim=1, keepdim=True).clamp_min(1e-6)
        visible_confidence_logits = self.confidence_head(features)
        if self.confidence_bias_head is None:
            visible_confidence_bias = torch.zeros_like(visible_confidence_logits[:, :, :1, :1])
        else:
            visible_confidence_bias = self.confidence_bias_head(features)
        visible_confidence = torch.sigmoid(visible_confidence_logits + visible_confidence_bias)
        visible_alpha = torch.sigmoid(self.alpha_head(features)) if self.alpha_head is not None else visible_confidence
        visible_features = self.feature_head(features)
        return {
            'visible_depth': visible_depth,
            'visible_normals': visible_normals,
            'visible_confidence': visible_confidence,
            'visible_confidence_logits': visible_confidence_logits,
            'visible_confidence_bias': visible_confidence_bias,
            'visible_alpha': visible_alpha,
            'visible_features': visible_features,
        }


class HiddenBranch(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_proposals: int,
        enable_confidence_bias: bool = False,
        enable_separate_render_alpha: bool = False,
        enable_gap_conditioned_confidence: bool = False,
        enable_boundary_conditioned_confidence: bool = False,
        enable_local_confidence_residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_proposals = hidden_proposals
        self.enable_confidence_bias = enable_confidence_bias
        self.enable_separate_render_alpha = enable_separate_render_alpha
        self.enable_gap_conditioned_confidence = enable_gap_conditioned_confidence
        self.enable_boundary_conditioned_confidence = enable_boundary_conditioned_confidence
        self.enable_local_confidence_residual = enable_local_confidence_residual
        self.offset_head = nn.Conv2d(hidden_dim, hidden_proposals, kernel_size=1)
        base_confidence_in_channels = hidden_dim + hidden_proposals if self.enable_gap_conditioned_confidence else hidden_dim
        self.confidence_head = nn.Conv2d(base_confidence_in_channels, hidden_proposals, kernel_size=1)
        if self.enable_local_confidence_residual:
            local_in_channels = hidden_dim
            if self.enable_gap_conditioned_confidence:
                local_in_channels += hidden_proposals
            if self.enable_boundary_conditioned_confidence:
                local_in_channels += 1
            local_hidden_dim = max(hidden_dim // 2, hidden_proposals)
            self.local_confidence_head = nn.Sequential(
                nn.Conv2d(local_in_channels, local_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(local_hidden_dim, hidden_proposals, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.local_confidence_head[-1].weight)
            nn.init.zeros_(self.local_confidence_head[-1].bias)
        else:
            self.local_confidence_head = None
        if self.enable_confidence_bias:
            self.confidence_bias_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_proposals, kernel_size=1),
            )
            nn.init.zeros_(self.confidence_bias_head[1].weight)
            nn.init.zeros_(self.confidence_bias_head[1].bias)
        else:
            self.confidence_bias_head = None
        self.alpha_head = nn.Conv2d(hidden_dim, hidden_proposals, kernel_size=1) if self.enable_separate_render_alpha else None
        self.feature_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    @staticmethod
    def _compute_visible_boundary(visible_depth: torch.Tensor, visible_confidence: torch.Tensor | None) -> torch.Tensor:
        visible_depth = visible_depth.detach()
        visible_confidence = visible_confidence.detach() if visible_confidence is not None else None
        depth_boundary = torch.zeros_like(visible_depth)
        horiz_diff = (visible_depth[:, :, :, 1:] - visible_depth[:, :, :, :-1]).abs()
        horiz_scale = torch.maximum(
            torch.minimum(visible_depth[:, :, :, 1:], visible_depth[:, :, :, :-1]).clamp_min(1e-3) * 0.05,
            torch.full_like(horiz_diff, 0.02),
        )
        horiz_score = (horiz_diff / horiz_scale.clamp_min(1e-6)).clamp(0.0, 1.0)
        depth_boundary[:, :, :, 1:] = torch.maximum(depth_boundary[:, :, :, 1:], horiz_score)
        depth_boundary[:, :, :, :-1] = torch.maximum(depth_boundary[:, :, :, :-1], horiz_score)

        vert_diff = (visible_depth[:, :, 1:, :] - visible_depth[:, :, :-1, :]).abs()
        vert_scale = torch.maximum(
            torch.minimum(visible_depth[:, :, 1:, :], visible_depth[:, :, :-1, :]).clamp_min(1e-3) * 0.05,
            torch.full_like(vert_diff, 0.02),
        )
        vert_score = (vert_diff / vert_scale.clamp_min(1e-6)).clamp(0.0, 1.0)
        depth_boundary[:, :, 1:, :] = torch.maximum(depth_boundary[:, :, 1:, :], vert_score)
        depth_boundary[:, :, :-1, :] = torch.maximum(depth_boundary[:, :, :-1, :], vert_score)

        if visible_confidence is None:
            return depth_boundary

        conf_boundary = torch.zeros_like(visible_depth)
        horiz_conf = (visible_confidence[:, :, :, 1:] - visible_confidence[:, :, :, :-1]).abs().clamp(0.0, 1.0)
        conf_boundary[:, :, :, 1:] = torch.maximum(conf_boundary[:, :, :, 1:], horiz_conf)
        conf_boundary[:, :, :, :-1] = torch.maximum(conf_boundary[:, :, :, :-1], horiz_conf)
        vert_conf = (visible_confidence[:, :, 1:, :] - visible_confidence[:, :, :-1, :]).abs().clamp(0.0, 1.0)
        conf_boundary[:, :, 1:, :] = torch.maximum(conf_boundary[:, :, 1:, :], vert_conf)
        conf_boundary[:, :, :-1, :] = torch.maximum(conf_boundary[:, :, :-1, :], vert_conf)
        return torch.maximum(depth_boundary, conf_boundary) * visible_confidence.clamp(0.0, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        visible_depth: torch.Tensor,
        visible_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_offset = F.softplus(self.offset_head(features))
        hidden_depth = visible_depth + 0.1 + hidden_offset
        if self.enable_gap_conditioned_confidence:
            confidence_features = torch.cat([features, hidden_offset], dim=1)
        else:
            confidence_features = features
        hidden_confidence_logits = self.confidence_head(confidence_features)

        if self.local_confidence_head is not None:
            local_features = [features]
            if self.enable_gap_conditioned_confidence:
                local_features.append(hidden_offset)
            if self.enable_boundary_conditioned_confidence:
                visible_boundary = self._compute_visible_boundary(visible_depth, visible_confidence)
                local_features.append(visible_boundary)
            hidden_confidence_logits = hidden_confidence_logits + self.local_confidence_head(torch.cat(local_features, dim=1))

        if self.confidence_bias_head is None:
            hidden_confidence_bias = torch.zeros_like(hidden_confidence_logits[:, :, :1, :1])
        else:
            hidden_confidence_bias = self.confidence_bias_head(features)
        hidden_confidence = torch.sigmoid(hidden_confidence_logits + hidden_confidence_bias)
        hidden_alpha = torch.sigmoid(self.alpha_head(features)) if self.alpha_head is not None else hidden_confidence
        hidden_features = self.feature_head(features)
        return {
            'hidden_depth': hidden_depth,
            'hidden_confidence': hidden_confidence,
            'hidden_confidence_logits': hidden_confidence_logits,
            'hidden_confidence_bias': hidden_confidence_bias,
            'hidden_alpha': hidden_alpha,
            'hidden_features': hidden_features,
        }
