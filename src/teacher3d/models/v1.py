from __future__ import annotations

import torch
from torch import nn

from teacher3d.models.branches import HiddenBranch, VisibleBranch
from teacher3d.models.decoder import LayeredGaussianDecoder
from teacher3d.models.encoder import build_image_encoder


class ConfidenceCalibrator(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        enable_visible_calibration: bool = True,
        enable_hidden_calibration: bool = True,
        enable_dual_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.enable_visible_calibration = enable_visible_calibration
        self.enable_hidden_calibration = enable_hidden_calibration
        self.enable_dual_temperature = enable_dual_temperature
        self.pool = nn.AdaptiveAvgPool2d(1)

        bias_channels = int(enable_visible_calibration) + int(enable_hidden_calibration)
        if bias_channels > 0:
            self.bias_head = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, bias_channels, kernel_size=1),
            )
            nn.init.zeros_(self.bias_head[-1].weight)
            nn.init.zeros_(self.bias_head[-1].bias)
        else:
            self.bias_head = None

        temp_channels = bias_channels if enable_dual_temperature else 1
        self.temperature_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, temp_channels, kernel_size=1),
        )
        nn.init.zeros_(self.temperature_head[-1].weight)
        nn.init.zeros_(self.temperature_head[-1].bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.pool(features)
        outputs: dict[str, torch.Tensor] = {}

        if self.bias_head is not None:
            bias = self.bias_head(pooled)
            offset = 0
            if self.enable_visible_calibration:
                outputs['visible_scene_bias'] = bias[:, offset : offset + 1]
                offset += 1
            if self.enable_hidden_calibration:
                outputs['hidden_scene_bias'] = bias[:, offset : offset + 1]

        log_temperature = self.temperature_head(pooled).clamp(-1.0, 1.0)
        if self.enable_dual_temperature:
            offset = 0
            if self.enable_visible_calibration:
                visible_log_temperature = log_temperature[:, offset : offset + 1]
                outputs['visible_confidence_log_temperature'] = visible_log_temperature
                outputs['visible_confidence_temperature'] = torch.exp(visible_log_temperature)
                offset += 1
            if self.enable_hidden_calibration:
                hidden_log_temperature = log_temperature[:, offset : offset + 1]
                outputs['hidden_confidence_log_temperature'] = hidden_log_temperature
                outputs['hidden_confidence_temperature'] = torch.exp(hidden_log_temperature)
        else:
            outputs['shared_confidence_log_temperature'] = log_temperature
            outputs['shared_confidence_temperature'] = torch.exp(log_temperature)

        return outputs


class Teacher3DV1(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        hidden_dim = int(config.model.hidden_dim)
        self.hidden_dim = hidden_dim
        self.hidden_proposals = int(getattr(config.model, 'hidden_proposals', 1))
        self.enable_hidden_branch = bool(getattr(config.model, 'enable_hidden_branch', True))
        self.enable_visible_confidence_bias = bool(getattr(config.model, 'enable_visible_confidence_bias', False))
        self.enable_hidden_confidence_bias = bool(getattr(config.model, 'enable_hidden_confidence_bias', False))
        self.enable_separate_render_alpha = bool(getattr(config.model, 'enable_separate_render_alpha', False))
        self.enable_gap_conditioned_hidden_confidence = bool(getattr(config.model, 'enable_gap_conditioned_hidden_confidence', False))
        self.enable_boundary_conditioned_hidden_confidence = bool(getattr(config.model, 'enable_boundary_conditioned_hidden_confidence', False))
        self.enable_hidden_local_confidence_residual = bool(getattr(config.model, 'enable_hidden_local_confidence_residual', False))
        self.enable_shared_confidence_calibration = bool(getattr(config.model, 'enable_shared_confidence_calibration', False))
        default_calibration_flag = self.enable_shared_confidence_calibration
        self.enable_visible_confidence_calibration = bool(
            getattr(config.model, 'enable_visible_confidence_calibration', default_calibration_flag)
        )
        self.enable_hidden_confidence_calibration = bool(
            getattr(config.model, 'enable_hidden_confidence_calibration', default_calibration_flag)
        )
        self.enable_dual_confidence_temperature = bool(
            getattr(config.model, 'enable_dual_confidence_temperature', False)
        )

        self.encoder = build_image_encoder(config)
        encoder_out_channels = int(getattr(self.encoder, 'out_channels', hidden_dim))
        self.bridge = nn.Identity() if encoder_out_channels == hidden_dim else nn.Conv2d(encoder_out_channels, hidden_dim, kernel_size=1)
        self.visible_branch = VisibleBranch(
            hidden_dim,
            enable_confidence_bias=self.enable_visible_confidence_bias,
            enable_separate_render_alpha=self.enable_separate_render_alpha,
        )
        self.hidden_branch = (
            HiddenBranch(
                hidden_dim,
                self.hidden_proposals,
                enable_confidence_bias=self.enable_hidden_confidence_bias,
                enable_separate_render_alpha=self.enable_separate_render_alpha,
                enable_gap_conditioned_confidence=self.enable_gap_conditioned_hidden_confidence,
                enable_boundary_conditioned_confidence=self.enable_boundary_conditioned_hidden_confidence,
                enable_local_confidence_residual=self.enable_hidden_local_confidence_residual,
            )
            if self.enable_hidden_branch
            else None
        )
        if self.enable_visible_confidence_calibration or self.enable_hidden_confidence_calibration:
            self.confidence_calibrator = ConfidenceCalibrator(
                hidden_dim,
                enable_visible_calibration=self.enable_visible_confidence_calibration,
                enable_hidden_calibration=self.enable_hidden_confidence_calibration,
                enable_dual_temperature=self.enable_dual_confidence_temperature,
            )
        else:
            self.confidence_calibrator = None
        self.decoder = LayeredGaussianDecoder(hidden_dim)

    def _empty_hidden(self, features: torch.Tensor, visible_depth: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, height, width = features.shape
        zeros_feature = torch.zeros((batch, self.hidden_dim, height, width), device=features.device, dtype=features.dtype)
        zeros_conf = torch.zeros((batch, self.hidden_proposals, height, width), device=features.device, dtype=features.dtype)
        hidden_depth = visible_depth.repeat(1, self.hidden_proposals, 1, 1) + 0.1
        zeros_bias = torch.zeros((batch, self.hidden_proposals, 1, 1), device=features.device, dtype=features.dtype)
        return {
            'hidden_depth': hidden_depth,
            'hidden_confidence': zeros_conf,
            'hidden_confidence_logits': zeros_conf,
            'hidden_confidence_bias': zeros_bias,
            'hidden_alpha': zeros_conf,
            'hidden_features': zeros_feature,
        }

    @staticmethod
    def _select_temperature(calibration: dict[str, torch.Tensor], prefix: str) -> torch.Tensor | None:
        specific = calibration.get(f'{prefix}_confidence_temperature')
        if specific is not None:
            return specific
        return calibration.get('shared_confidence_temperature')

    def _apply_confidence_calibration(
        self,
        visible: dict[str, torch.Tensor],
        hidden: dict[str, torch.Tensor],
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.confidence_calibrator is None:
            return {}

        calibration = self.confidence_calibrator(features)

        if self.enable_visible_confidence_calibration:
            visible_bias = visible.get(
                'visible_confidence_bias',
                torch.zeros_like(visible['visible_confidence_logits'][:, :, :1, :1]),
            )
            visible_temperature = self._select_temperature(calibration, 'visible')
            visible_scene_bias = calibration.get('visible_scene_bias', torch.zeros_like(visible_bias))
            visible_logits = (visible['visible_confidence_logits'] + visible_bias + visible_scene_bias) / visible_temperature
            visible['visible_confidence_logits'] = visible_logits
            visible['visible_confidence_bias'] = torch.zeros_like(visible_bias)
            visible['visible_confidence'] = torch.sigmoid(visible_logits)
            if not self.enable_separate_render_alpha:
                visible['visible_alpha'] = visible['visible_confidence']

        if self.enable_hidden_confidence_calibration:
            hidden_bias = hidden.get(
                'hidden_confidence_bias',
                torch.zeros_like(hidden['hidden_confidence_logits'][:, :, :1, :1]),
            )
            hidden_temperature = self._select_temperature(calibration, 'hidden')
            hidden_scene_bias = calibration.get('hidden_scene_bias', torch.zeros_like(hidden_bias))
            hidden_logits = (hidden['hidden_confidence_logits'] + hidden_bias + hidden_scene_bias) / hidden_temperature
            hidden['hidden_confidence_logits'] = hidden_logits
            hidden['hidden_confidence_bias'] = torch.zeros_like(hidden_bias)
            hidden['hidden_confidence'] = torch.sigmoid(hidden_logits)
            if not self.enable_separate_render_alpha:
                hidden['hidden_alpha'] = hidden['hidden_confidence']

        return calibration

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.bridge(self.encoder(image))
        visible = self.visible_branch(features)
        if self.hidden_branch is None:
            hidden = self._empty_hidden(features, visible['visible_depth'])
        else:
            hidden = self.hidden_branch(features, visible['visible_depth'], visible.get('visible_confidence'))

        calibration = self._apply_confidence_calibration(visible, hidden, features)

        decoded = self.decoder(
            image=image,
            visible_features=visible['visible_features'],
            hidden_features=hidden['hidden_features'],
            visible_alpha=visible['visible_alpha'],
            hidden_alpha=hidden['hidden_alpha'],
        )
        outputs = {}
        outputs.update(visible)
        outputs.update(hidden)
        outputs.update(calibration)
        outputs.update(decoded)
        return outputs
