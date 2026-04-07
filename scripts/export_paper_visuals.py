import argparse
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT / '.venv' / 'bin' / 'python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    if len(args.configs) != len(args.names):
        raise ValueError('configs and names must have the same length')
    out_root = pathlib.Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, config in zip(args.names, args.configs):
        out_dir = out_root / name
        subprocess.run([str(PY), str(ROOT / 'scripts' / 'visualize.py'), '--config', config, '--output-dir', str(out_dir), '--max-samples', str(args.max_samples)], check=True)
        manifest.append({'name': name, 'config': config, 'output_dir': str(out_dir)})
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(out_root / 'manifest.json')


if __name__ == '__main__':
    main()
