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
from teacher3d.vggt_integration import VGGTConfig, VGGTTeacherRunner, build_vggt_teacher_packet, save_vggt_packet


def _as_int(value):
    return int(value.item()) if hasattr(value, "item") else int(value)


def main(args) -> None:
    config = load_config(args.config)
    config.data.require_teacher_targets = False
    dataset = build_dataset(config)

    teacher_cfg = VGGTConfig(
        repo_path=pathlib.Path(getattr(config.teacher, "repo_path", ROOT / "third_party" / "vggt")),
        device=args.device,
        hf_repo=getattr(config.teacher, "hf_repo", "facebook/VGGT-1B"),
        weights_path=getattr(config.teacher, "weights_path", None),
        conf_threshold=float(getattr(config.teacher, "conf_threshold", 1.5)),
        depth_margin=float(getattr(config.teacher, "depth_margin", 0.05)),
    )
    runner = VGGTTeacherRunner(teacher_cfg)

    limit = len(dataset) if args.limit < 0 else min(args.limit, len(dataset))
    stats = []
    for index in range(limit):
        sample = dataset[index]
        scene_id = sample["scene_id"]
        split = sample["split"]
        context_timestamp = _as_int(sample["context_timestamp"])
        target_timestamp = _as_int(sample["target_timestamp"])
        out_path = pathlib.Path(config.data.teacher_target_root) / split / scene_id / f"{context_timestamp}_{target_timestamp}.pt"
        if out_path.exists() and not args.overwrite:
            print(json.dumps({"index": index, "status": "skip_exists", "path": str(out_path)}))
            continue

        predictions = runner.predict_pair(
            context_image_path=sample["context_image_path"],
            target_image_path=sample["target_image_path"],
            width=int(config.data.image_width),
            height=int(config.data.image_height),
        )
        packet = build_vggt_teacher_packet(
            predictions,
            conf_threshold=teacher_cfg.conf_threshold,
            depth_margin=teacher_cfg.depth_margin,
        )
        packet["scene_id"] = scene_id
        packet["split"] = split
        packet["context_timestamp"] = context_timestamp
        packet["target_timestamp"] = target_timestamp
        path = save_vggt_packet(
            packet_root=config.data.teacher_target_root,
            split=split,
            scene_id=scene_id,
            context_timestamp=context_timestamp,
            target_timestamp=target_timestamp,
            packet=packet,
        )
        row = {
            "index": index,
            "status": "saved",
            "scene_id": scene_id,
            "context_timestamp": context_timestamp,
            "target_timestamp": target_timestamp,
            "path": str(path),
        }
        stats.append(row)
        print(json.dumps(row))

    print(json.dumps({"saved": len(stats), "limit": limit}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args)
