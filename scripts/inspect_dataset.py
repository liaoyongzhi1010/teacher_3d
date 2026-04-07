import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.config import load_config
from teacher3d.data import build_dataloader


def main(config_path: str) -> None:
    config = load_config(config_path)
    loader = build_dataloader(config, shuffle=False)
    batch = next(iter(loader))
    summary = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            summary[key] = list(value.shape)
        elif isinstance(value, list):
            summary[key] = value[:2]
        else:
            summary[key] = str(value)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
