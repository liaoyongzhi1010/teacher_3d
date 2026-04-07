import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from teacher3d.train import main


if __name__ == "__main__":
    main(str(ROOT / "configs" / "v1_dummy.yaml"))
