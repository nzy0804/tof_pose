import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tof_pose.realtime_pose import run


def _list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except Exception:
        return []

    ports = []
    for p in list_ports.comports():
        if getattr(p, "device", None):
            ports.append(str(p.device))
    return ports


def main() -> int:
    parser = argparse.ArgumentParser(description="ToF pose serial realtime pipeline")
    parser.add_argument("--port", default="COM8", help="serial port (e.g., COM8)")
    args = parser.parse_args()

    print(f"[tof_pose_cli] pid={os.getpid()} port={args.port}", flush=True)
    try:
        run(port=args.port)
    except Exception as exc:
        ports = _list_serial_ports()
        if ports:
            print(f"[tof_pose_cli] available ports: {', '.join(ports)}", flush=True)
        print(f"[tof_pose_cli] failed: {exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
