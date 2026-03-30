from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from maixsense.realtime_pose import run


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else "COM8"
    run(port=port)
