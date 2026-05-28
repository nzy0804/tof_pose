#!/usr/bin/env python3
"""Monitor a process' CPU and memory usage in real-time.

Use cases
- Monitor gRPC server / tof_pose_cli runtime CPU% and RSS MB.
- Either attach to an existing PID, or launch a command and monitor it.

Examples
  # Attach to a running process
  python scripts/monitor_process_resources.py --pid 12345

  # Launch then monitor (note the -- separator)
  python scripts/monitor_process_resources.py --interval 1 -- python scripts/tof_pose_cli.py COM8

  # Monitor the packaged Linux binary (on Linux)
  python scripts/monitor_process_resources.py --interval 1 -- ./maixsense-grpc-server --host 0.0.0.0 --port 50052
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import psutil


def _fmt_mb(x_bytes: float) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def monitor_pid(pid: int, interval_s: float, duration_s: float | None) -> int:
    proc = psutil.Process(pid)

    # Prime cpu_percent calculation.
    proc.cpu_percent(interval=None)

    start = time.time()
    peak_rss_mb = 0.0

    print("timestamp,cpu_percent,rss_mb,vms_mb,threads", flush=True)

    while True:
        if duration_s is not None and (time.time() - start) >= duration_s:
            break

        try:
            cpu = proc.cpu_percent(interval=interval_s)
            mem = proc.memory_info()
            rss_mb = _fmt_mb(mem.rss)
            vms_mb = _fmt_mb(mem.vms)
            threads = proc.num_threads()
            peak_rss_mb = max(peak_rss_mb, rss_mb)
            print(f"{time.strftime('%H:%M:%S')},{cpu:.1f},{rss_mb:.1f},{vms_mb:.1f},{threads}", flush=True)
        except psutil.NoSuchProcess:
            break

    print(f"peak_rss_mb={peak_rss_mb:.1f}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor process CPU% and memory RSS in real-time")
    parser.add_argument("--pid", type=int, default=None, help="PID to attach")
    parser.add_argument("--interval", type=float, default=1.0, help="sampling interval seconds")
    parser.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="command to launch after --")
    args = parser.parse_args()

    if args.pid is None and not args.cmd:
        parser.error("Provide --pid <pid> or a command after --")

    if args.pid is not None and args.cmd:
        parser.error("Use either --pid or command, not both")

    if args.pid is not None:
        return monitor_pid(args.pid, float(args.interval), args.duration)

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("Empty command")

    child = subprocess.Popen(cmd)
    try:
        return monitor_pid(child.pid, float(args.interval), args.duration)
    finally:
        # If duration ended, leave the child running; user can Ctrl+C to stop.
        pass


if __name__ == "__main__":
    raise SystemExit(main())
