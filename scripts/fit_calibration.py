import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.calibration import threshold_to_bias
from teacher3d.config import load_config
from teacher3d.data import build_dataloader
from teacher3d.models import Teacher3DV1
from teacher3d.train import move_batch


def _threshold_grid(threshold_min: float, threshold_max: float, threshold_step: float):
    thresholds = []
    value = threshold_min
    while value <= threshold_max + 1e-9:
        thresholds.append(round(value, 6))
        value += threshold_step
    return thresholds


def _accumulate_binary_stats(stats, pred: torch.Tensor, target: torch.Tensor, prefix: str, thresholds):
    target = (target > 0.5).expand_as(pred)
    for threshold in thresholds:
        pred_binary = pred > threshold
        tp = float((pred_binary & target).sum().item())
        pp = float(pred_binary.sum().item())
        gt = float(target.sum().item())
        stats[prefix][threshold]['tp'] += tp
        stats[prefix][threshold]['pp'] += pp
        stats[prefix][threshold]['gt'] += gt


def _summarize(stats, thresholds):
    summary = {}
    for prefix in ['visible', 'hidden']:
        best = None
        curve = []
        for threshold in thresholds:
            tp = stats[prefix][threshold]['tp']
            pp = max(stats[prefix][threshold]['pp'], 1.0)
            gt = max(stats[prefix][threshold]['gt'], 1.0)
            precision = tp / pp
            recall = tp / gt
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
            row = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'equivalent_bias': threshold_to_bias(threshold),
            }
            curve.append(row)
            if best is None or row['f1'] > best['f1']:
                best = row
        summary[prefix] = {'best': best, 'curve': curve}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--threshold-min', type=float, default=0.05)
    parser.add_argument('--threshold-max', type=float, default=0.5)
    parser.add_argument('--threshold-step', type=float, default=0.01)
    parser.add_argument('--output')
    args = parser.parse_args()

    thresholds = _threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    config = load_config(args.config)
    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else pathlib.Path(config.output_dir) / 'model.pt'
    device = torch.device(args.device)

    loader = build_dataloader(config, shuffle=False)
    model = Teacher3DV1(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()

    stats = {
        'visible': {threshold: {'tp': 0.0, 'pp': 0.0, 'gt': 0.0} for threshold in thresholds},
        'hidden': {threshold: {'tp': 0.0, 'pp': 0.0, 'gt': 0.0} for threshold in thresholds},
    }

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if args.max_steps is not None and step >= args.max_steps:
                break
            batch = move_batch(batch, device)
            outputs = model(batch['image'])
            _accumulate_binary_stats(stats, outputs['visible_confidence'], batch['teacher_visible_confidence'], 'visible', thresholds)
            _accumulate_binary_stats(stats, outputs['hidden_confidence'], batch['teacher_hidden_confidence'], 'hidden', thresholds)

    summary = {
        'config': args.config,
        'checkpoint': str(checkpoint),
        'threshold_min': args.threshold_min,
        'threshold_max': args.threshold_max,
        'threshold_step': args.threshold_step,
    }
    summary.update(_summarize(stats, thresholds))

    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        print(f'saved calibration to {output_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
