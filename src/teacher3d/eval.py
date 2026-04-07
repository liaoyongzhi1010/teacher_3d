from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from teacher3d.calibration import apply_eval_calibration, get_binary_threshold
from teacher3d.config import load_config
from teacher3d.data import build_dataloader
from teacher3d.models import Teacher3DV1
from teacher3d.teacher import build_teacher


def move_batch(batch, device: torch.device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.l1_loss(pred, target)
    weight = mask.expand_as(pred).float()
    denom = weight.sum().clamp_min(1.0)
    return ((pred - target).abs() * weight).sum() / denom


def _masked_mean(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(pred).float()
    denom = weight.sum().clamp_min(1.0)
    return (pred * weight).sum() / denom


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


def _binary_stats(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float().expand_as(pred_binary)
    true_positive = (pred_binary * target_binary).sum()
    pred_positive = pred_binary.sum().clamp_min(1.0)
    target_positive = target_binary.sum().clamp_min(1.0)
    union = ((pred_binary + target_binary) > 0).float().sum().clamp_min(1.0)
    precision = true_positive / pred_positive
    recall = true_positive / target_positive
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    iou = true_positive / union
    return precision, recall, f1, iou


def _confidence_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float().expand_as(pred_binary)
    return (pred_binary == target_binary).float().mean()


def compute_metrics(outputs, batch, teacher_targets=None, eval_config=None) -> dict[str, torch.Tensor]:
    outputs = apply_eval_calibration(outputs, eval_config)
    visible_threshold = get_binary_threshold(eval_config, 'visible')
    hidden_threshold = get_binary_threshold(eval_config, 'hidden')

    metrics: dict[str, torch.Tensor] = {}
    pred_target = outputs['rendered_target'].clamp(0.0, 1.0)
    target_image = batch['target_image']

    metrics['novel_view_l1'] = F.l1_loss(pred_target, target_image)
    metrics['novel_view_psnr'] = _psnr(pred_target, target_image)
    metrics['hidden_alpha_mean'] = outputs['hidden_alpha'].mean()
    metrics['hidden_confidence_mean'] = outputs['hidden_confidence'].mean()
    metrics['hidden_active_ratio'] = (outputs['hidden_confidence'] > hidden_threshold).float().mean()
    metrics['visible_confidence_mean'] = outputs['visible_confidence'].mean()

    visible_mask = batch.get('teacher_visible_confidence')
    if 'teacher_visible_depth' in batch:
        metrics['visible_depth_l1'] = _masked_l1(outputs['visible_depth'], batch['teacher_visible_depth'], visible_mask)
    if 'teacher_visible_normals' in batch:
        visible_cos = 1.0 - F.cosine_similarity(outputs['visible_normals'], batch['teacher_visible_normals'], dim=1)
        if visible_mask is not None:
            denom = visible_mask.sum().clamp_min(1.0)
            visible_cos = (visible_cos * visible_mask[:, 0]).sum() / denom
        else:
            visible_cos = visible_cos.mean()
        metrics['visible_normal_cos'] = visible_cos
    if visible_mask is not None:
        metrics['visible_confidence_acc'] = _confidence_accuracy(outputs['visible_confidence'], visible_mask, visible_threshold)
        visible_precision, visible_recall, visible_f1, visible_iou = _binary_stats(outputs['visible_confidence'], visible_mask, visible_threshold)
        metrics['visible_precision'] = visible_precision
        metrics['visible_recall'] = visible_recall
        metrics['visible_f1'] = visible_f1
        metrics['visible_iou'] = visible_iou
        metrics['visible_teacher_coverage'] = visible_mask.mean()

    hidden_mask = batch.get('teacher_hidden_confidence')
    if 'teacher_hidden_depth' in batch:
        metrics['hidden_depth_l1'] = _masked_l1(outputs['hidden_depth'], batch['teacher_hidden_depth'], hidden_mask)
    if hidden_mask is not None:
        metrics['hidden_confidence_acc'] = _confidence_accuracy(outputs['hidden_confidence'], hidden_mask, hidden_threshold)
        hidden_precision, hidden_recall, hidden_f1, hidden_iou = _binary_stats(outputs['hidden_confidence'], hidden_mask, hidden_threshold)
        metrics['hidden_precision'] = hidden_precision
        metrics['hidden_recall'] = hidden_recall
        metrics['hidden_f1'] = hidden_f1
        metrics['hidden_iou'] = hidden_iou
        metrics['hidden_teacher_coverage'] = hidden_mask.mean()

    if hidden_mask is not None and 'teacher_visible_depth' in batch and 'teacher_hidden_depth' in batch:
        pred_gap = outputs['hidden_depth'] - outputs['visible_depth'].expand_as(outputs['hidden_depth'])
        teacher_gap = batch['teacher_hidden_depth'] - batch['teacher_visible_depth'].expand_as(outputs['hidden_depth'])
        metrics['hidden_gap_l1'] = _masked_l1(pred_gap, teacher_gap, hidden_mask)

    if teacher_targets is not None and 'hidden_local_support' in teacher_targets:
        hidden_local_mask = (teacher_targets['hidden_local_support'] > 0).float()
        metrics['hidden_local_teacher_coverage'] = hidden_local_mask.mean()
        local_precision, local_recall, local_f1, local_iou = _binary_stats(outputs['hidden_confidence'], hidden_local_mask, hidden_threshold)
        metrics['hidden_local_precision'] = local_precision
        metrics['hidden_local_recall'] = local_recall
        metrics['hidden_local_f1'] = local_f1
        metrics['hidden_local_iou'] = local_iou

    if teacher_targets is not None and 'hidden_deep_support' in teacher_targets:
        hidden_deep_mask = (teacher_targets['hidden_deep_support'] > 0).float()
        metrics['hidden_deep_teacher_coverage'] = hidden_deep_mask.mean()
        if float(hidden_deep_mask.sum().detach().cpu()) > 0.0:
            metrics['hidden_on_deep_mean'] = _masked_mean(outputs['hidden_confidence'], hidden_deep_mask)
            metrics['hidden_on_deep_active'] = _masked_mean((outputs['hidden_confidence'] > hidden_threshold).float(), hidden_deep_mask)

    if teacher_targets is not None and 'hidden_interior_negative' in teacher_targets:
        interior_mask = teacher_targets['hidden_interior_negative'].float()
        metrics['hidden_interior_negative_coverage'] = interior_mask.mean()
        if float(interior_mask.sum().detach().cpu()) > 0.0:
            metrics['hidden_on_interior_negative_mean'] = _masked_mean(outputs['hidden_confidence'], interior_mask)
            metrics['hidden_on_interior_negative_active'] = _masked_mean((outputs['hidden_confidence'] > hidden_threshold).float(), interior_mask)

    if visible_mask is not None:
        visible_negative_mask = visible_mask.expand_as(outputs['hidden_confidence'])
        if hidden_mask is not None:
            visible_negative_mask = visible_negative_mask * (1.0 - hidden_mask.expand_as(outputs['hidden_confidence']))
        metrics['hidden_on_visible_mean'] = _masked_mean(outputs['hidden_confidence'], visible_negative_mask)
        metrics['hidden_on_visible_active'] = _masked_mean((outputs['hidden_confidence'] > hidden_threshold).float(), visible_negative_mask)

    if 'teacher_available' in batch:
        metrics['teacher_available_ratio'] = batch['teacher_available'].float().mean()

    return metrics


def evaluate(config_path: str, checkpoint_path: str | None = None, device_override: str | None = None, max_steps: int | None = None):
    config = load_config(config_path)
    checkpoint = Path(checkpoint_path) if checkpoint_path else Path(config.output_dir) / 'model.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    eval_config = getattr(config, 'eval', None)
    eval_device = getattr(eval_config, 'device', None) if eval_config is not None else None
    device_name = device_override or eval_device or config.train.device
    device = torch.device(device_name)

    loader = build_dataloader(config, shuffle=False)
    teacher = build_teacher(config)
    model = Teacher3DV1(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    totals: dict[str, float] = {}
    steps = 0
    total_samples = 0

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            batch = move_batch(batch, device)
            teacher_targets = teacher(batch)
            outputs = model(batch['image'])
            metrics = compute_metrics(outputs, batch, teacher_targets=teacher_targets, eval_config=eval_config)
            batch_size = int(batch['image'].shape[0])
            total_samples += batch_size
            steps += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size

    if total_samples == 0:
        raise RuntimeError('No samples were evaluated.')

    averaged = {key: value / total_samples for key, value in totals.items()}
    averaged['num_steps'] = steps
    averaged['num_samples'] = total_samples
    averaged['checkpoint'] = str(checkpoint)
    averaged['config'] = str(config_path)
    averaged['device'] = str(device)
    return averaged


def main(config_path: str, checkpoint_path: str | None = None, output_path: str | None = None, device_override: str | None = None, max_steps: int | None = None) -> None:
    metrics = evaluate(config_path=config_path, checkpoint_path=checkpoint_path, device_override=device_override, max_steps=max_steps)
    if output_path:
        output_file = Path(output_path)
    else:
        config = load_config(config_path)
        output_file = Path(config.output_dir) / 'eval.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f'saved eval to {output_file}')
