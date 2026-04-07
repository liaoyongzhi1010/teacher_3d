import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.eval import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_override=args.device,
        max_steps=args.max_steps,
    )
