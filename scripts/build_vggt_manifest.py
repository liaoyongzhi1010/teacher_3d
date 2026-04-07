import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.config import load_config
from teacher3d.data import build_dataset
from teacher3d.teacher_packets import build_teacher_packet_path


def main(config_path: str, limit: int) -> None:
    config = load_config(config_path)
    dataset = build_dataset(config)
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        scene_id = sample["scene_id"]
        split = sample["split"]
        context_timestamp = _as_int(sample["context_timestamp"])
        target_timestamp = _as_int(sample["target_timestamp"])
        packet_path = build_teacher_packet_path(
            config.data.teacher_target_root,
            split,
            scene_id,
            context_timestamp,
            target_timestamp,
        )
        print(f"{index}\t{scene_id}\t{context_timestamp}\t{target_timestamp}\t{packet_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()
    main(args.config, args.limit)
