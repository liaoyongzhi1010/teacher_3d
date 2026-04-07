import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.config import load_config
from teacher3d.data import build_dataloader
from teacher3d.models import Teacher3DV1
from teacher3d.teacher import build_teacher
from teacher3d.eval import move_batch, compute_metrics


def parse_thresholds(spec: str):
    return [float(item) for item in spec.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--visible-thresholds', default='0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--hidden-thresholds', default='0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--teacher-target-root')
    parser.add_argument('--enable-layered-eval', action='store_true')
    parser.add_argument('--layered-boundary-min-score', type=float, default=0.30)
    parser.add_argument('--layered-local-gap-max', type=float, default=0.12)
    parser.add_argument('--layered-raw-confidence-scale', type=float, default=3.0)
    parser.add_argument('--layered-raw-count-scale', type=float, default=4.0)
    parser.add_argument('--max-visible-bleed', type=float, default=None)
    parser.add_argument('--min-visible-f1', type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.teacher_target_root:
        config.data.teacher_target_root = args.teacher_target_root
    if args.enable_layered_eval:
        config.teacher.enable_hidden_layered_targets = True
        config.teacher.hidden_target_boundary_radius = 1
        config.teacher.hidden_target_boundary_threshold = 0.05
        config.teacher.hidden_target_boundary_rel_threshold = 0.02
        config.teacher.hidden_layered_boundary_min_score = args.layered_boundary_min_score
        config.teacher.hidden_layered_local_gap_max = args.layered_local_gap_max
        config.teacher.hidden_layered_use_raw_support = True
        config.teacher.hidden_layered_raw_confidence_scale = args.layered_raw_confidence_scale
        config.teacher.hidden_layered_raw_count_scale = args.layered_raw_count_scale

    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else pathlib.Path(config.output_dir) / 'model.pt'
    output_path = pathlib.Path(args.output)
    visible_thresholds = parse_thresholds(args.visible_thresholds)
    hidden_thresholds = parse_thresholds(args.hidden_thresholds)

    device = torch.device(config.train.device)
    loader = build_dataloader(config, shuffle=False)
    teacher = build_teacher(config)
    model = Teacher3DV1(config).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    pair_totals = {}
    pair_counts = {}
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            teacher_targets = teacher(batch)
            outputs = model(batch['image'])
            batch_size = int(batch['image'].shape[0])
            for vt in visible_thresholds:
                for ht in hidden_thresholds:
                    key = (vt, ht)
                    metrics = compute_metrics(
                        outputs,
                        batch,
                        teacher_targets=teacher_targets,
                        eval_config={'visible_threshold': vt, 'hidden_threshold': ht},
                    )
                    totals = pair_totals.setdefault(key, {})
                    pair_counts[key] = pair_counts.get(key, 0) + batch_size
                    for metric_name, value in metrics.items():
                        totals[metric_name] = totals.get(metric_name, 0.0) + float(value.detach().cpu()) * batch_size

        results = []
        for vt in visible_thresholds:
            for ht in hidden_thresholds:
                key = (vt, ht)
                total_samples = pair_counts[key]
                totals = pair_totals[key]
                averaged = {metric_name: value / total_samples for metric_name, value in totals.items()}
                averaged['visible_threshold'] = vt
                averaged['hidden_threshold'] = ht
                results.append(averaged)

    payload = {
        'config': str(args.config),
        'checkpoint': str(checkpoint),
        'results': results,
        'best_by_hidden_f1': max(results, key=lambda x: x.get('hidden_f1', -1.0)),
        'best_by_hidden_local_f1': max(results, key=lambda x: x.get('hidden_local_f1', -1.0)),
    }

    constrained = results
    if args.min_visible_f1 is not None:
        constrained = [row for row in constrained if row.get('visible_f1', 0.0) >= args.min_visible_f1]
    if args.max_visible_bleed is not None:
        constrained = [row for row in constrained if row.get('hidden_on_visible_active', 1e9) <= args.max_visible_bleed]
    if constrained:
        payload['best_constrained_by_hidden_f1'] = max(constrained, key=lambda x: x.get('hidden_f1', -1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        'best_by_hidden_f1': payload['best_by_hidden_f1'],
        'best_by_hidden_local_f1': payload['best_by_hidden_local_f1'],
        'best_constrained_by_hidden_f1': payload.get('best_constrained_by_hidden_f1'),
    }, indent=2, sort_keys=True))
    print(f'saved dual sweep to {output_path}')


if __name__ == '__main__':
    main()
