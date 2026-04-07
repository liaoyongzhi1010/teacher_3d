from __future__ import annotations

import math

import torch


def _get(config, name: str, default):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def safe_logit(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(prob) - torch.log1p(-prob)


def apply_affine_confidence(prob: torch.Tensor, bias: float = 0.0, temperature: float = 1.0) -> torch.Tensor:
    if abs(float(bias)) < 1e-12 and abs(float(temperature) - 1.0) < 1e-12:
        return prob
    logit = safe_logit(prob)
    scaled = (logit + float(bias)) / max(float(temperature), 1e-6)
    return torch.sigmoid(scaled)


def apply_eval_calibration(outputs: dict[str, torch.Tensor], eval_config=None) -> dict[str, torch.Tensor]:
    if eval_config is None:
        return outputs
    calibrated = dict(outputs)
    if 'visible_confidence' in outputs:
        calibrated['visible_confidence'] = apply_affine_confidence(
            outputs['visible_confidence'],
            bias=float(_get(eval_config, 'visible_confidence_bias', 0.0)),
            temperature=float(_get(eval_config, 'visible_confidence_temperature', 1.0)),
        )
    if 'hidden_confidence' in outputs:
        calibrated['hidden_confidence'] = apply_affine_confidence(
            outputs['hidden_confidence'],
            bias=float(_get(eval_config, 'hidden_confidence_bias', 0.0)),
            temperature=float(_get(eval_config, 'hidden_confidence_temperature', 1.0)),
        )
    return calibrated


def get_binary_threshold(eval_config, prefix: str) -> float:
    specific = _get(eval_config, f'{prefix}_threshold', None)
    if specific is not None:
        return float(specific)
    return float(_get(eval_config, 'binary_threshold', 0.5))


def threshold_to_bias(threshold: float) -> float:
    threshold = min(max(float(threshold), 1e-6), 1.0 - 1e-6)
    return -math.log(threshold / (1.0 - threshold))
