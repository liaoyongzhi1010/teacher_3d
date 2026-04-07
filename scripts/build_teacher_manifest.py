import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.config import load_config
from teacher3d.data import build_dataset
from teacher3d.teacher_packets import build_teacher_packet_path


def main(config_path: str, limit: int, output_path: str | None) -> None:
    config = load_config(config_path)
    dataset = build_dataset(config)
    teacher_root = getattr(config.data, "teacher_target_root", None)
    if not teacher_root:
        raise ValueError("config.data.teacher_target_root must be set to build a teacher manifest")

    rows = []
    for index in range(min(limit, len(dataset))):
        sample = dataset[index]
        scene_id = sample["scene_id"]
        split = sample["split"]
        context_timestamp = int(sample["context_timestamp"].item())
        target_timestamp = int(sample["target_timestamp"].item())
        packet_path = build_teacher_packet_path(
            teacher_root,
            split,
            scene_id,
            context_timestamp,
            target_timestamp,
        )
        row = {
            "index": index,
            "scene_id": scene_id,
            "split": split,
            "context_timestamp": context_timestamp,
            "target_timestamp": target_timestamp,
            "packet_path": str(packet_path),
        }
        rows.append(row)

    if output_path is None:
        print(json.dumps(rows, indent=2))
        return

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.config, args.limit, args.output)
