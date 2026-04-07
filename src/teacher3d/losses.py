from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LossComputer:
    def __init__(self, config) -> None:
        self.weights = config.loss
        self.visibility_aware_teacher = bool(getattr(config.loss, 'visibility_aware_teacher', True))
        self.visible_pos_weight = float(getattr(config.loss, 'visible_pos_weight', 1.0))
        self.visible_focal_gamma = float(getattr(config.loss, 'visible_focal_gamma', 0.0))
        self.hidden_pos_weight = float(getattr(config.loss, 'hidden_pos_weight', 1.0))
        self.hidden_focal_gamma = float(getattr(config.loss, 'hidden_focal_gamma', 0.0))
        self.hidden_gap_min = float(
            getattr(config.loss, 'hidden_gap_min', getattr(getattr(config, 'teacher', {}), 'depth_margin', 0.05))
        )
        self.hidden_conf_margin_min = float(getattr(config.loss, 'hidden_conf_margin_min', 0.10))
        self.visible_logit_margin_min = float(getattr(config.loss, 'visible_logit_margin_min', 0.20))
        self.hidden_logit_margin_min = float(getattr(config.loss, 'hidden_logit_margin_min', 0.20))
        self.hidden_pixel_contrastive_margin = float(getattr(config.loss, 'hidden_pixel_contrastive_margin', 0.20))
        self.hidden_pixel_contrastive_samples = int(getattr(config.loss, 'hidden_pixel_contrastive_samples', 64))
        self.hidden_logit_band_positive_prob = float(getattr(config.loss, 'hidden_logit_band_positive_prob', 0.55))
        self.hidden_logit_band_negative_prob = float(getattr(config.loss, 'hidden_logit_band_negative_prob', 0.35))
        self.hidden_logit_band_samples = int(getattr(config.loss, 'hidden_logit_band_samples', 128))
        self.hidden_ambiguous_low_prob = float(getattr(config.loss, 'hidden_ambiguous_low_prob', 0.20))
        self.hidden_ambiguous_high_prob = float(getattr(config.loss, 'hidden_ambiguous_high_prob', 0.50))

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return F.l1_loss(pred, target)
        weight = mask.expand_as(pred).float()
        denom = weight.sum().clamp_min(1.0)
        return ((pred - target).abs() * weight).sum() / denom

    @staticmethod
    def _binary_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        pos_weight: float = 1.0,
        focal_gamma: float = 0.0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        target = target.float().expand_as(pred)
        loss = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        if pos_weight != 1.0:
            loss = loss * (1.0 + (pos_weight - 1.0) * target)
        if focal_gamma > 0.0:
            pt = pred * target + (1.0 - pred) * (1.0 - target)
            loss = loss * (1.0 - pt).pow(focal_gamma)
        if mask is None:
            return loss.mean()
        weight = mask.expand_as(pred).float()
        denom = weight.sum().clamp_min(1.0)
        return (loss * weight).sum() / denom

    @staticmethod
    def _coverage_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_cov = pred.mean(dim=(1, 2, 3))
        target_cov = target.float().expand_as(pred).mean(dim=(1, 2, 3))
        return F.l1_loss(pred_cov, target_cov)

    @staticmethod
    def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        target = target.float().expand_as(pred)
        intersection = (pred * target).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice.mean()

    @staticmethod
    def _masked_mean_per_sample(pred: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.expand_as(pred).float()
        denom = weight.sum(dim=(1, 2, 3))
        mean = (pred * weight).sum(dim=(1, 2, 3)) / denom.clamp_min(1.0)
        valid = denom > 0.0
        return mean, valid

    @staticmethod
    def _pixel_contrastive_loss(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        margin: float,
        max_samples: int,
    ) -> torch.Tensor | None:
        flat_logits = logits.reshape(logits.shape[0], -1)
        flat_pos = positive_mask.expand_as(logits).reshape(logits.shape[0], -1) > 0.5
        flat_neg = negative_mask.expand_as(logits).reshape(logits.shape[0], -1) > 0.5
        losses = []
        for idx in range(flat_logits.shape[0]):
            pos_vals = flat_logits[idx][flat_pos[idx]]
            neg_vals = flat_logits[idx][flat_neg[idx]]
            if pos_vals.numel() == 0 or neg_vals.numel() == 0:
                continue
            if max_samples > 0 and pos_vals.numel() > max_samples:
                pos_vals = torch.topk(pos_vals, k=max_samples, largest=False).values
            if max_samples > 0 and neg_vals.numel() > max_samples:
                neg_vals = torch.topk(neg_vals, k=max_samples, largest=True).values
            pairwise = neg_vals[:, None] - pos_vals[None, :] + margin
            losses.append(F.softplus(pairwise).mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    @staticmethod
    def _logit(prob: float) -> float:
        prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        return math.log(prob / (1.0 - prob))

    @classmethod
    def _logit_band_loss(
        cls,
        logits: torch.Tensor,
        mask: torch.Tensor,
        target_prob: float,
        direction: str,
        max_samples: int,
    ) -> torch.Tensor | None:
        flat_logits = logits.reshape(logits.shape[0], -1)
        flat_mask = mask.expand_as(logits).reshape(logits.shape[0], -1) > 0.5
        target_logit = cls._logit(target_prob)
        losses = []
        for idx in range(flat_logits.shape[0]):
            vals = flat_logits[idx][flat_mask[idx]]
            if vals.numel() == 0:
                continue
            if direction == 'positive':
                violation = target_logit - vals
            elif direction == 'negative':
                violation = vals - target_logit
            else:
                raise ValueError(f'Unsupported direction: {direction}')
            if max_samples > 0 and violation.numel() > max_samples:
                violation = torch.topk(violation, k=max_samples, largest=True).values
            losses.append(F.relu(violation).mean())
        if not losses:
            return None
        return torch.stack(losses).mean()

    @staticmethod
    def _prob_interval_loss(
        pred: torch.Tensor,
        mask: torch.Tensor,
        low_prob: float,
        high_prob: float,
    ) -> torch.Tensor | None:
        low_prob = float(max(0.0, min(1.0, low_prob)))
        high_prob = float(max(low_prob, min(1.0, high_prob)))
        weight = mask.expand_as(pred).float()
        denom = weight.sum()
        if float(denom.detach().cpu()) <= 0.0:
            return None
        pred = pred.clamp(1e-6, 1.0 - 1e-6)
        violation = F.relu(low_prob - pred) + F.relu(pred - high_prob)
        return (violation * weight).sum() / denom.clamp_min(1.0)

    @staticmethod
    def _zero(device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def __call__(self, outputs, batch, teacher_targets):
        device = batch['image'].device
        losses = {}

        visible_teacher_mask = teacher_targets.get('visible_confidence') if self.visibility_aware_teacher else None
        hidden_teacher_mask = teacher_targets.get('hidden_confidence') if self.visibility_aware_teacher else None

        if 'visible_depth' in teacher_targets:
            visible_geometry = self._masked_l1(
                outputs['visible_depth'],
                teacher_targets['visible_depth'],
                visible_teacher_mask,
            )
            if 'visible_normals' in teacher_targets:
                cosine = 1.0 - F.cosine_similarity(
                    outputs['visible_normals'],
                    teacher_targets['visible_normals'],
                    dim=1,
                )
                if visible_teacher_mask is not None:
                    denom = visible_teacher_mask.sum().clamp_min(1.0)
                    cosine = (cosine * visible_teacher_mask[:, 0]).sum() / denom
                else:
                    cosine = cosine.mean()
                visible_geometry = visible_geometry + 0.5 * cosine
            losses['visible_geometry'] = visible_geometry
        else:
            losses['visible_geometry'] = self._zero(device)

        visible_confidence_logits = outputs['visible_confidence_logits'] + outputs.get(
            'visible_confidence_bias',
            torch.zeros_like(outputs['visible_confidence_logits'][:, :, :1, :1]),
        )
        hidden_confidence_logits = outputs['hidden_confidence_logits'] + outputs.get(
            'hidden_confidence_bias',
            torch.zeros_like(outputs['hidden_confidence_logits'][:, :, :1, :1]),
        )

        if 'visible_confidence' in teacher_targets:
            losses['visible_confidence'] = self._binary_loss(
                outputs['visible_confidence'],
                teacher_targets['visible_confidence'],
                pos_weight=self.visible_pos_weight,
                focal_gamma=self.visible_focal_gamma,
            )
            losses['visible_coverage'] = self._coverage_l1(
                outputs['visible_confidence'],
                teacher_targets['visible_confidence'],
            )
            render_mask = teacher_targets['visible_confidence'] if self.visibility_aware_teacher else torch.ones_like(outputs['visible_confidence'])
        else:
            losses['visible_confidence'] = self._zero(device)
            losses['visible_coverage'] = self._zero(device)
            render_mask = torch.ones_like(outputs['visible_confidence'])

        losses['visible_render'] = ((outputs['rendered_target'] - batch['image']).abs() * render_mask).mean()

        if 'hidden_depth' in teacher_targets:
            losses['hidden_geometry'] = self._masked_l1(
                outputs['hidden_depth'],
                teacher_targets['hidden_depth'],
                hidden_teacher_mask,
            )
        else:
            losses['hidden_geometry'] = self._zero(device)

        pred_gap = outputs['hidden_depth'] - outputs['visible_depth'].expand_as(outputs['hidden_depth'])
        if hidden_teacher_mask is not None and 'hidden_depth' in teacher_targets and 'visible_depth' in teacher_targets:
            teacher_gap = teacher_targets['hidden_depth'] - teacher_targets['visible_depth'].expand_as(outputs['hidden_depth'])
            teacher_gap = teacher_gap.clamp_min(self.hidden_gap_min)
            losses['hidden_ranking'] = self._masked_l1(pred_gap, teacher_gap, hidden_teacher_mask)
        else:
            target_gap = torch.full_like(pred_gap, self.hidden_gap_min)
            weight = outputs['hidden_confidence'].detach().expand_as(pred_gap)
            denom = weight.sum().clamp_min(1.0)
            losses['hidden_ranking'] = (F.relu(target_gap - pred_gap) * weight).sum() / denom

        losses['hidden_sparse'] = outputs['hidden_confidence'].mean()

        if 'hidden_confidence' in teacher_targets:
            losses['hidden_coverage'] = self._coverage_l1(
                outputs['hidden_confidence'],
                teacher_targets['hidden_confidence'],
            )
            positive_mask = teacher_targets['hidden_confidence'].expand_as(outputs['hidden_confidence'])
            losses['hidden_positive'] = self._binary_loss(
                outputs['hidden_confidence'],
                torch.ones_like(outputs['hidden_confidence']),
                mask=positive_mask,
            )
            losses['hidden_dice'] = self._soft_dice_loss(
                outputs['hidden_confidence'],
                teacher_targets['hidden_confidence'],
            )
        else:
            losses['hidden_coverage'] = self._zero(device)
            losses['hidden_positive'] = self._zero(device)
            losses['hidden_dice'] = self._zero(device)

        if 'hidden_local_support' in teacher_targets:
            losses['hidden_local_support'] = self._binary_loss(
                outputs['hidden_confidence'],
                teacher_targets['hidden_local_support'],
                mask=(teacher_targets['hidden_local_support'] > 0).float(),
            )
        else:
            losses['hidden_local_support'] = self._zero(device)

        if 'hidden_deep_support' in teacher_targets:
            deep_mask = (teacher_targets['hidden_deep_support'] > 0).float()
            losses['hidden_deep_support'] = self._binary_loss(
                outputs['hidden_confidence'],
                teacher_targets['hidden_deep_support'],
                mask=deep_mask,
            )
            ambiguous_band = self._prob_interval_loss(
                outputs['hidden_confidence'],
                deep_mask,
                low_prob=self.hidden_ambiguous_low_prob,
                high_prob=self.hidden_ambiguous_high_prob,
            )
            losses['hidden_ambiguous_band'] = ambiguous_band if ambiguous_band is not None else self._zero(device)
        else:
            losses['hidden_deep_support'] = self._zero(device)
            losses['hidden_ambiguous_band'] = self._zero(device)

        if 'hidden_interior_negative' in teacher_targets:
            losses['hidden_interior_negative'] = self._binary_loss(
                outputs['hidden_confidence'],
                torch.zeros_like(outputs['hidden_confidence']),
                mask=teacher_targets['hidden_interior_negative'],
            )
        else:
            losses['hidden_interior_negative'] = self._zero(device)

        if 'hidden_confidence_mined_mask' in teacher_targets and 'hidden_confidence_mined_target' in teacher_targets:
            losses['hidden_mined_confidence'] = self._binary_loss(
                outputs['hidden_confidence'],
                teacher_targets['hidden_confidence_mined_target'],
                mask=teacher_targets['hidden_confidence_mined_mask'],
            )
            losses['hidden_mined_geometry'] = self._masked_l1(
                outputs['hidden_depth'],
                teacher_targets['hidden_depth_mined'],
                teacher_targets['hidden_confidence_mined_mask'],
            )
        else:
            losses['hidden_mined_confidence'] = self._zero(device)
            losses['hidden_mined_geometry'] = self._zero(device)

        negative_mask = None
        if 'visible_confidence' in teacher_targets:
            negative_mask = teacher_targets['visible_confidence'].expand_as(outputs['hidden_confidence'])
            if 'hidden_confidence' in teacher_targets:
                negative_mask = negative_mask * (1.0 - teacher_targets['hidden_confidence'].expand_as(outputs['hidden_confidence']))
            losses['hidden_hard_negative'] = self._binary_loss(
                outputs['hidden_confidence'],
                torch.zeros_like(outputs['hidden_confidence']),
                mask=negative_mask,
            )
        else:
            losses['hidden_hard_negative'] = self._zero(device)

        if visible_teacher_mask is not None:
            visible_negative_mask = 1.0 - visible_teacher_mask.expand_as(outputs['visible_confidence'])
            vis_pos_mean, vis_pos_valid = self._masked_mean_per_sample(visible_confidence_logits, visible_teacher_mask)
            vis_neg_mean, vis_neg_valid = self._masked_mean_per_sample(visible_confidence_logits, visible_negative_mask)
            vis_valid = vis_pos_valid & vis_neg_valid
            if vis_valid.any():
                vis_margin = vis_pos_mean[vis_valid] - vis_neg_mean[vis_valid]
                losses['visible_logit_margin'] = F.relu(self.visible_logit_margin_min - vis_margin).mean()
            else:
                losses['visible_logit_margin'] = self._zero(device)
        else:
            losses['visible_logit_margin'] = self._zero(device)

        if hidden_teacher_mask is not None and negative_mask is not None:
            pos_mean, pos_valid = self._masked_mean_per_sample(outputs['hidden_confidence'], hidden_teacher_mask)
            neg_mean, neg_valid = self._masked_mean_per_sample(outputs['hidden_confidence'], negative_mask)
            valid = pos_valid & neg_valid
            if valid.any():
                margin = pos_mean[valid] - neg_mean[valid]
                losses['hidden_conf_margin'] = F.relu(self.hidden_conf_margin_min - margin).mean()
            else:
                losses['hidden_conf_margin'] = self._zero(device)

            logit_pos_mean, logit_pos_valid = self._masked_mean_per_sample(hidden_confidence_logits, hidden_teacher_mask)
            logit_neg_mean, logit_neg_valid = self._masked_mean_per_sample(hidden_confidence_logits, negative_mask)
            logit_valid = logit_pos_valid & logit_neg_valid
            if logit_valid.any():
                logit_margin = logit_pos_mean[logit_valid] - logit_neg_mean[logit_valid]
                losses['hidden_logit_margin'] = F.relu(self.hidden_logit_margin_min - logit_margin).mean()
            else:
                losses['hidden_logit_margin'] = self._zero(device)

            pixel_contrastive = self._pixel_contrastive_loss(
                hidden_confidence_logits,
                hidden_teacher_mask,
                negative_mask,
                margin=self.hidden_pixel_contrastive_margin,
                max_samples=self.hidden_pixel_contrastive_samples,
            )
            if pixel_contrastive is not None:
                losses['hidden_pixel_contrastive'] = pixel_contrastive
            else:
                losses['hidden_pixel_contrastive'] = self._zero(device)

            pos_band = self._logit_band_loss(
                hidden_confidence_logits,
                hidden_teacher_mask,
                target_prob=self.hidden_logit_band_positive_prob,
                direction='positive',
                max_samples=self.hidden_logit_band_samples,
            )
            neg_band = self._logit_band_loss(
                hidden_confidence_logits,
                negative_mask,
                target_prob=self.hidden_logit_band_negative_prob,
                direction='negative',
                max_samples=self.hidden_logit_band_samples,
            )
            band_terms = [term for term in (pos_band, neg_band) if term is not None]
            if band_terms:
                losses['hidden_logit_band'] = torch.stack(band_terms).mean()
            else:
                losses['hidden_logit_band'] = self._zero(device)
        else:
            losses['hidden_conf_margin'] = self._zero(device)
            losses['hidden_logit_margin'] = self._zero(device)
            losses['hidden_pixel_contrastive'] = self._zero(device)
            losses['hidden_logit_band'] = self._zero(device)

        losses['novel_view'] = F.l1_loss(outputs['rendered_target'], batch['target_image'])

        calibration_terms = []
        for key in ('shared_confidence_log_temperature', 'visible_confidence_log_temperature', 'hidden_confidence_log_temperature'):
            if key in outputs:
                calibration_terms.append(outputs[key].pow(2).mean())
        for key in ('visible_scene_bias', 'hidden_scene_bias'):
            if key in outputs:
                calibration_terms.append(0.25 * outputs[key].pow(2).mean())
        if calibration_terms:
            losses['shared_calibration_reg'] = torch.stack(calibration_terms).sum()
        else:
            losses['shared_calibration_reg'] = self._zero(device)

        if 'hidden_confidence' in teacher_targets:
            losses['teacher_consistency'] = self._binary_loss(
                outputs['hidden_confidence'],
                teacher_targets['hidden_confidence'],
                pos_weight=self.hidden_pos_weight,
                focal_gamma=self.hidden_focal_gamma,
            )
        else:
            losses['teacher_consistency'] = self._zero(device)

        dx_conf = outputs['visible_confidence'][..., 1:] - outputs['visible_confidence'][..., :-1]
        dx_depth = outputs['visible_depth'][..., 1:] - outputs['visible_depth'][..., :-1]
        losses['boundary'] = (dx_conf.abs() * dx_depth.abs()).mean()

        total = torch.zeros((), device=device)
        for key, value in losses.items():
            if key == 'visibility_aware_teacher':
                continue
            total = total + float(self.weights.get(key, 0.0)) * value
        losses['total'] = total
        return losses
