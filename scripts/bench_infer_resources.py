#!/usr/bin/env python3
"""Benchmark local resource usage of RealtimePoseEngine.infer().

Why this exists
- gRPC adds networking/serialization overhead; for sizing CPU/RAM, the dominant cost is model inference.
- This script measures per-call latency + process CPU time + RSS memory (peak) in one process.

Example
  python scripts/bench_infer_resources.py dataset/images/val/xxx.png --iters 50 --warmup 5

Notes
- First call is usually slower (model warmup, cache).
- RSS includes native memory (PyTorch/OpenCV) when psutil is available.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure repository root and src/ are on sys.path so tof_pose can be imported when running from repo.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

from tof_pose.realtime_service import RealtimePoseEngine


@dataclass
class Sample:
    wall_ms: float
    cpu_ms: float
    rss_mb: float | None


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return float(values_sorted[f])
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return float(d0 + d1)


def _rss_mb(process: "psutil.Process") -> float:
    return float(process.memory_info().rss) / (1024.0 * 1024.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RealtimePoseEngine.infer() CPU/RAM usage")
    parser.add_argument("input", help="input PNG image path (depth gray) or any image cv2 can decode")
    parser.add_argument("--frame-id", default="bench_frame", help="frame_id to pass into infer()")
    parser.add_argument("--iters", type=int, default=50, help="measured iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations (not counted)")
    parser.add_argument("--fps-target", type=float, default=None, help="optional target FPS to estimate required CPU cores")
    parser.add_argument("--model-path", type=str, default=None, help="override seg model path")
    parser.add_argument("--pose-model-path", type=str, default=None, help="override pose model path")
    args = parser.parse_args()

    img_path = Path(args.input)
    if not img_path.exists():
        raise SystemExit(f"input not found: {img_path}")

    image_bytes = img_path.read_bytes()

    proc = psutil.Process(os.getpid()) if psutil is not None else None
    rss0 = _rss_mb(proc) if proc is not None else None

    t_init0 = time.perf_counter()
    engine = RealtimePoseEngine(
        model_path=Path(args.model_path) if args.model_path else None,
        pose_model_path=Path(args.pose_model_path) if args.pose_model_path else None,
    )
    t_init1 = time.perf_counter()

    rss1 = _rss_mb(proc) if proc is not None else None

    # Warmup
    for i in range(max(0, int(args.warmup))):
        engine.infer(f"{args.frame_id}_warmup_{i}", image_bytes)

    samples: list[Sample] = []
    rss_peak = rss1

    cpu0 = time.process_time()
    wall0 = time.perf_counter()

    for i in range(max(1, int(args.iters))):
        w0 = time.perf_counter()
        c0 = time.process_time()
        engine.infer(f"{args.frame_id}_{i}", image_bytes)
        c1 = time.process_time()
        w1 = time.perf_counter()

        rss = _rss_mb(proc) if proc is not None else None
        if rss is not None:
            rss_peak = rss if rss_peak is None else max(rss_peak, rss)

        samples.append(
            Sample(
                wall_ms=(w1 - w0) * 1000.0,
                cpu_ms=(c1 - c0) * 1000.0,
                rss_mb=rss,
            )
        )

    wall1 = time.perf_counter()
    cpu1 = time.process_time()

    wall_ms_list = [s.wall_ms for s in samples]
    cpu_ms_list = [s.cpu_ms for s in samples]

    total_wall_s = max(wall1 - wall0, 1e-9)
    total_cpu_s = max(cpu1 - cpu0, 0.0)

    mean_wall = statistics.mean(wall_ms_list)
    p50_wall = _pct(wall_ms_list, 50)
    p95_wall = _pct(wall_ms_list, 95)
    max_wall = max(wall_ms_list)

    mean_cpu = statistics.mean(cpu_ms_list)
    p50_cpu = _pct(cpu_ms_list, 50)
    p95_cpu = _pct(cpu_ms_list, 95)

    print("=== bench_infer_resources ===")
    print(f"input: {img_path}")
    print(f"init_ms: {(t_init1 - t_init0) * 1000.0:.1f}")
    if rss0 is not None and rss1 is not None:
        print(f"rss_mb: before_init={rss0:.1f} after_init={rss1:.1f} peak={rss_peak:.1f}")
    else:
        print("rss_mb: (psutil not available; install via: pip install psutil)")

    print(f"iters: warmup={args.warmup} measured={len(samples)}")
    print(f"wall_ms: mean={mean_wall:.1f} p50={p50_wall:.1f} p95={p95_wall:.1f} max={max_wall:.1f}")
    print(f"cpu_ms : mean={mean_cpu:.1f} p50={p50_cpu:.1f} p95={p95_cpu:.1f}")

    # Average CPU cores consumed by this process during benchmark window.
    avg_cores = total_cpu_s / total_wall_s if total_wall_s > 0 else 0.0
    cpu_s_per_call = total_cpu_s / max(len(samples), 1)
    wall_s_per_call = total_wall_s / max(len(samples), 1)
    print(f"avg_cores_used: {avg_cores:.2f}")
    print(f"per_call: wall_s={wall_s_per_call:.4f} cpu_s={cpu_s_per_call:.4f}")

    if args.fps_target is not None and args.fps_target > 0:
        cores_needed = cpu_s_per_call * float(args.fps_target)
        print(f"cores_needed_for_{args.fps_target:g}fps (est): {cores_needed:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
