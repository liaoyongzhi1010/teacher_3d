import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.config import load_config


def render_config(template_path: pathlib.Path, seed: int, suffix: str) -> pathlib.Path:
    config = load_config(template_path)
    config["seed"] = seed
    config["output_dir"] = f"{config.output_dir}_seed{seed}{suffix}"
    out_path = ROOT / 'configs' / 'generated' / f"{template_path.stem}_seed{seed}{suffix}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = pathlib.Path(template_path).read_text().splitlines()
    updated = []
    for line in lines:
        if line.startswith('seed:'):
            updated.append(f'seed: {seed}')
        elif line.startswith('output_dir:'):
            updated.append(f'output_dir: {config.output_dir}')
        else:
            updated.append(line)
    out_path.write_text("\n".join(updated) + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template-config', required=True)
    parser.add_argument('--seeds', default='7,11,17')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dual-sweep', action='store_true')
    parser.add_argument('--teacher-target-root')
    parser.add_argument('--enable-layered-eval', action='store_true')
    parser.add_argument('--min-visible-f1', type=float, default=0.30)
    parser.add_argument('--max-visible-bleed', type=float, default=0.20)
    args = parser.parse_args()

    template = pathlib.Path(args.template_config)
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    rows = []
    py = str(ROOT / '.venv' / 'bin' / 'python')
    for seed in seeds:
        cfg_path = render_config(template, seed, args.suffix)
        row = {'seed': seed, 'config': str(cfg_path)}
        if args.train:
            subprocess.run([py, str(ROOT / 'scripts' / 'train.py'), '--config', str(cfg_path)], check=True)
        if args.eval:
            out = ROOT / 'outputs' / 'seed_sweeps' / f'{cfg_path.stem}_eval.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([py, str(ROOT / 'scripts' / 'eval.py'), '--config', str(cfg_path), '--output', str(out)], check=True)
            row['eval'] = str(out)
        if args.dual_sweep:
            out = ROOT / 'outputs' / 'seed_sweeps' / f'{cfg_path.stem}_dual.json'
            cmd = [py, str(ROOT / 'scripts' / 'dual_threshold_sweep.py'), '--config', str(cfg_path), '--output', str(out), '--min-visible-f1', str(args.min_visible_f1), '--max-visible-bleed', str(args.max_visible_bleed)]
            if args.teacher_target_root:
                cmd.extend(['--teacher-target-root', args.teacher_target_root])
            if args.enable_layered_eval:
                cmd.append('--enable-layered-eval')
            subprocess.run(cmd, check=True)
            row['dual_sweep'] = str(out)
        rows.append(row)
    manifest = ROOT / 'outputs' / 'seed_sweeps' / f'{template.stem}{args.suffix}_manifest.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2))
    print(manifest)


if __name__ == '__main__':
    main()
