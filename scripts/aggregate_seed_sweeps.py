import argparse
import json
import pathlib
from statistics import mean, pstdev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--key', default='best_constrained_by_hidden_f1')
    parser.add_argument('--output')
    args = parser.parse_args()

    rows = json.loads(pathlib.Path(args.manifest).read_text())
    loaded = []
    for row in rows:
        dual = row.get('dual_sweep')
        if not dual:
            continue
        obj = json.loads(pathlib.Path(dual).read_text())
        entry = obj.get(args.key) or obj.get('best_by_hidden_f1')
        loaded.append({
            'seed': row['seed'],
            'visible_f1': entry.get('visible_f1'),
            'hidden_f1': entry.get('hidden_f1'),
            'hidden_local_f1': entry.get('hidden_local_f1'),
            'hidden_on_visible_active': entry.get('hidden_on_visible_active'),
            'novel_view_l1': entry.get('novel_view_l1'),
        })
    metrics = ['visible_f1', 'hidden_f1', 'hidden_local_f1', 'hidden_on_visible_active', 'novel_view_l1']
    summary = {'per_seed': loaded, 'mean_std': {}}
    for metric in metrics:
        vals = [row[metric] for row in loaded if row.get(metric) is not None]
        if not vals:
            continue
        summary['mean_std'][metric] = {'mean': mean(vals), 'std': pstdev(vals) if len(vals) > 1 else 0.0}
    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(out)
    else:
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
