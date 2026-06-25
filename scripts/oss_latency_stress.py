"""Stress test object storage latency without gRPC or model inference.

Examples:
    python scripts/oss_latency_stress.py --env-file maixsense-oss.env --mode upload --duration-sec 120 --workers 8
    python scripts/oss_latency_stress.py --env-file maixsense-oss.env --mode download --prefix frames/sim-device-001/ --duration-sec 120 --workers 8
    python scripts/oss_latency_stress.py --env-file maixsense-oss.env --mode download --seed-count 200 --duration-sec 120 --workers 8
    python scripts/oss_latency_stress.py --env-file maixsense-oss.env --mode batch --seed-count 200 --duration-sec 120
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import random
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tof_pose.object_storage import ObjectStorageConfig, create_object_storage_client


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    last_modified: int
    size: int


class LatencyStats:
    def __init__(self, *, slow_threshold_ms: int, slow_log_limit: int) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, list[int]] = {
            "upload": [],
            "download": [],
            "batch_download": [],
            "batch_upload": [],
            "batch_total": [],
        }
        self._bytes: dict[str, int] = {
            "upload": 0,
            "download": 0,
            "batch_download": 0,
            "batch_upload": 0,
            "batch_total": 0,
        }
        self._errors: dict[str, int] = {
            "upload": 0,
            "download": 0,
            "batch_download": 0,
            "batch_upload": 0,
            "batch_total": 0,
        }
        self._slow_events: list[tuple[str, int, int, str]] = []
        self._first_errors: list[str] = []
        self._slow_threshold_ms = max(0, int(slow_threshold_ms))
        self._slow_log_limit = max(0, int(slow_log_limit))

    def add_success(self, *, op: str, duration_ms: int, byte_count: int, key: str) -> None:
        duration_ms = max(0, int(duration_ms))
        byte_count = max(0, int(byte_count))
        print_slow = False
        with self._lock:
            self._durations.setdefault(op, []).append(duration_ms)
            self._bytes[op] = self._bytes.get(op, 0) + byte_count
            if self._slow_threshold_ms > 0 and duration_ms >= self._slow_threshold_ms:
                if len(self._slow_events) < self._slow_log_limit:
                    self._slow_events.append((op, duration_ms, byte_count, key))
                    print_slow = True
        if print_slow:
            print(f"SLOW op={op} ms={duration_ms} bytes={byte_count} key={key}", flush=True)

    def add_error(self, *, op: str, duration_ms: int, key: str, error: BaseException) -> None:
        message = f"ERROR op={op} ms={max(0, int(duration_ms))} key={key} error={type(error).__name__}: {error}"
        with self._lock:
            self._errors[op] = self._errors.get(op, 0) + 1
            if len(self._first_errors) < 10:
                self._first_errors.append(message)
        print(message, flush=True)

    def total_count(self) -> int:
        with self._lock:
            return sum(len(values) for values in self._durations.values())

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        with self._lock:
            durations = {op: list(values) for op, values in self._durations.items()}
            byte_counts = dict(self._bytes)
            errors = dict(self._errors)
        return {
            op: _summarize(values, byte_counts.get(op, 0), errors.get(op, 0))
            for op, values in durations.items()
        }

    def slow_events(self) -> list[tuple[str, int, int, str]]:
        with self._lock:
            return list(self._slow_events)

    def first_errors(self) -> list[str]:
        with self._lock:
            return list(self._first_errors)


class OpLimiter:
    def __init__(self, max_ops: int) -> None:
        self._max_ops = int(max_ops)
        self._count = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        if self._max_ops <= 0:
            return True
        with self._lock:
            if self._count >= self._max_ops:
                return False
            self._count += 1
            return True


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = int(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    return int(round(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction))


def _summarize(durations_ms: list[int], byte_count: int, errors: int) -> dict[str, int | float]:
    sorted_durations = sorted(durations_ms)
    count = len(sorted_durations)
    total_ms = sum(sorted_durations)
    return {
        "count": count,
        "errors": int(errors),
        "bytes": int(byte_count),
        "avg_ms": round(total_ms / count, 2) if count else 0,
        "p50_ms": _percentile(sorted_durations, 0.50),
        "p90_ms": _percentile(sorted_durations, 0.90),
        "p95_ms": _percentile(sorted_durations, 0.95),
        "p99_ms": _percentile(sorted_durations, 0.99),
        "max_ms": sorted_durations[-1] if sorted_durations else 0,
    }


def load_env_file(path: str | None) -> None:
    if not path:
        return
    env_path = Path(path).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _aliyun_bucket(config: ObjectStorageConfig):
    import oss2

    if config.security_token:
        auth = oss2.StsAuth(config.access_key_id, config.access_key_secret, config.security_token)
    else:
        auth = oss2.Auth(config.access_key_id, config.access_key_secret)
    session = None
    if hasattr(oss2, "Session"):
        try:
            session = oss2.Session(pool_size=config.max_pool_connections)
        except TypeError:
            session = oss2.Session()
    if session is not None:
        return oss2.Bucket(auth, config.endpoint, config.bucket, session=session)
    return oss2.Bucket(auth, config.endpoint, config.bucket)


def list_aliyun_objects(config: ObjectStorageConfig, prefix: str, max_keys: int) -> list[ObjectInfo]:
    import oss2

    bucket = _aliyun_bucket(config)
    objects: list[ObjectInfo] = []
    for item in oss2.ObjectIterator(bucket, prefix=prefix):
        key = str(getattr(item, "key", "") or "")
        if not key or key.endswith("/"):
            continue
        objects.append(
            ObjectInfo(
                key=key,
                last_modified=int(getattr(item, "last_modified", 0) or 0),
                size=int(getattr(item, "size", 0) or 0),
            )
        )
        if max_keys > 0 and len(objects) >= max_keys:
            break
    return objects


def list_s3_objects(config: ObjectStorageConfig, prefix: str, max_keys: int) -> list[ObjectInfo]:
    import boto3
    from botocore.config import Config

    kwargs = {
        "service_name": "s3",
        "endpoint_url": config.endpoint,
        "aws_access_key_id": config.access_key_id,
        "aws_secret_access_key": config.access_key_secret,
        "config": Config(max_pool_connections=config.max_pool_connections),
    }
    if config.region:
        kwargs["region_name"] = config.region
    if config.security_token:
        kwargs["aws_session_token"] = config.security_token
    client = boto3.client(**kwargs)

    objects: list[ObjectInfo] = []
    token: str | None = None
    while True:
        request = {"Bucket": config.bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            request["ContinuationToken"] = token
        response = client.list_objects_v2(**request)
        for item in response.get("Contents", []):
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            last_modified = int(item.get("LastModified").timestamp()) if item.get("LastModified") else 0
            objects.append(ObjectInfo(key=key, last_modified=last_modified, size=int(item.get("Size") or 0)))
            if max_keys > 0 and len(objects) >= max_keys:
                return objects
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            break
    return objects


def list_objects(config: ObjectStorageConfig, prefix: str, max_keys: int) -> list[ObjectInfo]:
    if config.provider == "aliyun":
        objects = list_aliyun_objects(config, prefix, max_keys)
    elif config.provider == "s3":
        objects = list_s3_objects(config, prefix, max_keys)
    else:
        raise ValueError(f"unsupported object storage provider: {config.provider}")
    return sorted(objects, key=lambda item: (item.last_modified, item.key))


def load_keys_file(path: str | None) -> list[str]:
    if not path:
        return []
    keys_path = Path(path).expanduser()
    return [
        line.strip()
        for line in keys_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def make_payload(size_kb: int) -> bytes:
    size = max(1, int(size_kb)) * 1024
    pattern = uuid.uuid4().bytes
    return (pattern * ((size // len(pattern)) + 1))[:size]


def seed_download_objects(client, *, prefix: str, count: int, payload: bytes, workers: int) -> list[str]:
    if count <= 0:
        return []
    run_id = uuid.uuid4().hex[:12]
    keys = [f"{prefix.rstrip('/')}/seed-{run_id}-{index:06d}.bin" for index in range(count)]
    print(f"Seeding download objects: count={count} workers={workers} prefix={prefix}", flush=True)

    def put_one(key: str) -> str:
        client.put_bytes(key, payload, content_type="application/octet-stream")
        return key

    with ThreadPoolExecutor(max_workers=max(1, min(workers, count))) as executor:
        futures = [executor.submit(put_one, key) for key in keys]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % max(1, count // 10) == 0 or completed == count:
                print(f"Seed progress: {completed}/{count}", flush=True)
    return keys


class GlobalLimiter:
    def __init__(self, max_workers: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max_workers) if max_workers > 0 else None

    def run(self, func):
        if self._semaphore is None:
            return func()
        self._semaphore.acquire()
        try:
            return func()
        finally:
            self._semaphore.release()


def run_concurrent_items(items: list, workers: int, func) -> None:
    if not items:
        return
    if len(items) <= 1 or workers <= 1:
        for item in items:
            func(item)
        return
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(items))) as executor:
        futures = [executor.submit(func, item) for item in items]
        for future in as_completed(futures):
            future.result()


def run_one_batch(
    *,
    batch_index: int,
    client,
    limiter: GlobalLimiter,
    stats: LatencyStats,
    download_keys: list[str],
    upload_payload: bytes,
    upload_prefix: str,
    batch_inputs: int,
    uploads_per_input: int,
    download_workers: int,
    upload_workers: int,
) -> None:
    rng = random.Random((batch_index + 1) * 9176)
    selected_download_keys = [download_keys[rng.randrange(len(download_keys))] for _ in range(max(0, batch_inputs))]
    batch_upload_count = max(0, batch_inputs) * max(0, uploads_per_input)
    upload_keys = [
        f"{upload_prefix.rstrip('/')}/batch-{batch_index:06d}/result-{index:04d}.bin"
        for index in range(batch_upload_count)
    ]
    batch_started = time.perf_counter()
    downloaded_bytes = 0
    uploaded_bytes = 0
    downloaded_bytes_lock = threading.Lock()
    uploaded_bytes_lock = threading.Lock()

    def download_one(key: str) -> None:
        nonlocal downloaded_bytes
        object_started = time.perf_counter()
        try:
            data = limiter.run(lambda: client.get_bytes(key))
            duration_ms = int((time.perf_counter() - object_started) * 1000)
            with downloaded_bytes_lock:
                downloaded_bytes += len(data)
            stats.add_success(op="download", duration_ms=duration_ms, byte_count=len(data), key=key)
        except BaseException as exc:
            stats.add_error(
                op="download",
                duration_ms=int((time.perf_counter() - object_started) * 1000),
                key=key,
                error=exc,
            )

    download_started = time.perf_counter()
    run_concurrent_items(selected_download_keys, download_workers, download_one)
    download_ms = int((time.perf_counter() - download_started) * 1000)
    stats.add_success(
        op="batch_download",
        duration_ms=download_ms,
        byte_count=downloaded_bytes,
        key=f"batch-{batch_index:06d}",
    )

    def upload_one(key: str) -> None:
        nonlocal uploaded_bytes
        object_started = time.perf_counter()
        try:
            limiter.run(lambda: client.put_bytes(key, upload_payload, content_type="application/octet-stream"))
            duration_ms = int((time.perf_counter() - object_started) * 1000)
            with uploaded_bytes_lock:
                uploaded_bytes += len(upload_payload)
            stats.add_success(op="upload", duration_ms=duration_ms, byte_count=len(upload_payload), key=key)
        except BaseException as exc:
            stats.add_error(
                op="upload",
                duration_ms=int((time.perf_counter() - object_started) * 1000),
                key=key,
                error=exc,
            )

    upload_started = time.perf_counter()
    run_concurrent_items(upload_keys, upload_workers, upload_one)
    upload_ms = int((time.perf_counter() - upload_started) * 1000)
    stats.add_success(
        op="batch_upload",
        duration_ms=upload_ms,
        byte_count=uploaded_bytes,
        key=f"batch-{batch_index:06d}",
    )
    stats.add_success(
        op="batch_total",
        duration_ms=int((time.perf_counter() - batch_started) * 1000),
        byte_count=downloaded_bytes + uploaded_bytes,
        key=f"batch-{batch_index:06d}",
    )


def run_batch_worker(
    *,
    worker_id: int,
    client,
    limiter: GlobalLimiter,
    upload_payload: bytes,
    upload_prefix: str,
    download_keys: list[str],
    stop_at: float,
    op_limiter: OpLimiter,
    stats: LatencyStats,
    batch_inputs: int,
    uploads_per_input: int,
    download_workers: int,
    upload_workers: int,
) -> None:
    batch_index = worker_id
    while time.perf_counter() < stop_at and op_limiter.take():
        run_one_batch(
            batch_index=batch_index,
            client=client,
            limiter=limiter,
            stats=stats,
            download_keys=download_keys,
            upload_payload=upload_payload,
            upload_prefix=upload_prefix,
            batch_inputs=batch_inputs,
            uploads_per_input=uploads_per_input,
            download_workers=download_workers,
            upload_workers=upload_workers,
        )
        batch_index += 100000


def format_summary(label: str, summary: dict[str, int | float], elapsed_sec: float) -> str:
    count = int(summary["count"])
    byte_count = int(summary["bytes"])
    qps = count / elapsed_sec if elapsed_sec > 0 else 0.0
    mibps = (byte_count / 1024 / 1024) / elapsed_sec if elapsed_sec > 0 else 0.0
    return (
        f"{label}: count={count} errors={summary['errors']} qps={qps:.2f} mibps={mibps:.2f} "
        f"avg_ms={summary['avg_ms']} p50_ms={summary['p50_ms']} p90_ms={summary['p90_ms']} "
        f"p95_ms={summary['p95_ms']} p99_ms={summary['p99_ms']} max_ms={summary['max_ms']} "
        f"bytes={byte_count}"
    )


def run_worker(
    *,
    worker_id: int,
    mode: str,
    client,
    upload_payload: bytes,
    upload_prefix: str,
    download_keys: list[str],
    stop_at: float,
    limiter: OpLimiter,
    stats: LatencyStats,
) -> None:
    rng = random.Random((worker_id + 1) * 1000003)
    upload_index = 0
    run_id = uuid.uuid4().hex[:12]
    while time.perf_counter() < stop_at and limiter.take():
        if mode == "mixed":
            op = "download" if download_keys and rng.random() < 0.5 else "upload"
        else:
            op = mode

        if op == "download":
            if not download_keys:
                raise ValueError("download mode requires keys from --keys-file, --prefix, or --seed-count")
            key = download_keys[rng.randrange(len(download_keys))]
            started = time.perf_counter()
            try:
                data = client.get_bytes(key)
                stats.add_success(
                    op="download",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    byte_count=len(data),
                    key=key,
                )
            except BaseException as exc:
                stats.add_error(
                    op="download",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    key=key,
                    error=exc,
                )
            continue

        key = f"{upload_prefix.rstrip('/')}/worker-{worker_id:03d}-{run_id}-{upload_index:09d}.bin"
        upload_index += 1
        started = time.perf_counter()
        try:
            client.put_bytes(key, upload_payload, content_type="application/octet-stream")
            stats.add_success(
                op="upload",
                duration_ms=int((time.perf_counter() - started) * 1000),
                byte_count=len(upload_payload),
                key=key,
            )
        except BaseException as exc:
            stats.add_error(
                op="upload",
                duration_ms=int((time.perf_counter() - started) * 1000),
                key=key,
                error=exc,
            )


def print_progress(stats: LatencyStats, started_at: float, stop_event: threading.Event, interval_sec: float) -> None:
    if interval_sec <= 0:
        return
    while not stop_event.wait(interval_sec):
        elapsed = max(0.001, time.perf_counter() - started_at)
        snapshot = stats.snapshot()
        print(format_summary("progress upload", snapshot["upload"], elapsed), flush=True)
        print(format_summary("progress download", snapshot["download"], elapsed), flush=True)
        print(format_summary("progress batch_download", snapshot["batch_download"], elapsed), flush=True)
        print(format_summary("progress batch_upload", snapshot["batch_upload"], elapsed), flush=True)
        print(format_summary("progress batch_total", snapshot["batch_total"], elapsed), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stress test OSS upload/download latency only.")
    parser.add_argument("--env-file", default=None, help="optional env file containing OSS_* settings")
    parser.add_argument("--mode", choices=["upload", "download", "mixed", "batch"], default="upload")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--concurrent-batches", type=int, default=1)
    parser.add_argument("--batch-inputs", type=int, default=20)
    parser.add_argument("--uploads-per-input", type=int, default=4)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--upload-workers", type=int, default=8)
    parser.add_argument("--global-workers", type=int, default=8)
    parser.add_argument("--max-ops", type=int, default=0, help="0 means no operation-count limit")
    parser.add_argument("--object-size-kb", type=int, default=64)
    parser.add_argument("--download-object-size-kb", type=int, default=None)
    parser.add_argument("--upload-object-size-kb", type=int, default=None)
    parser.add_argument("--upload-prefix", default=None, help="prefix for uploaded stress-test objects")
    parser.add_argument("--prefix", default=None, help="prefix used to list existing download objects")
    parser.add_argument("--keys-file", default=None, help="file containing one object key per line for download tests")
    parser.add_argument("--list-max-keys", type=int, default=1000)
    parser.add_argument("--seed-count", type=int, default=0, help="upload this many objects before measuring download")
    parser.add_argument("--seed-workers", type=int, default=8)
    parser.add_argument("--slow-threshold-ms", type=int, default=1000)
    parser.add_argument("--slow-log-limit", type=int, default=50)
    parser.add_argument("--report-interval-sec", type=float, default=10.0)
    parser.add_argument("--oss-provider", default=None)
    parser.add_argument("--oss-endpoint", default=None)
    parser.add_argument("--oss-bucket", default=None)
    parser.add_argument("--oss-access-key-id", default=None)
    parser.add_argument("--oss-access-key-secret", default=None)
    parser.add_argument("--oss-region", default=None)
    parser.add_argument("--oss-security-token", default=None)
    parser.add_argument("--oss-output-prefix", default=None)
    parser.add_argument("--oss-max-pool-connections", default=None, type=int)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    config = ObjectStorageConfig.from_values(
        provider=args.oss_provider,
        endpoint=args.oss_endpoint,
        bucket=args.oss_bucket,
        access_key_id=args.oss_access_key_id,
        access_key_secret=args.oss_access_key_secret,
        region=args.oss_region,
        security_token=args.oss_security_token,
        output_prefix=args.oss_output_prefix,
        max_pool_connections=args.oss_max_pool_connections,
    )
    if config is None:
        raise RuntimeError("OSS config is missing; provide OSS_* env vars, --env-file, or CLI options")
    client = create_object_storage_client(config)
    if client is None:
        raise RuntimeError("failed to create object storage client")

    upload_prefix = args.upload_prefix or f"{config.output_prefix.rstrip('/')}/oss-latency-stress/{uuid.uuid4().hex[:12]}"
    download_object_size_kb = args.download_object_size_kb or args.object_size_kb
    upload_object_size_kb = args.upload_object_size_kb or args.object_size_kb
    download_payload = make_payload(download_object_size_kb)
    upload_payload = make_payload(upload_object_size_kb)
    download_keys = load_keys_file(args.keys_file)

    if args.prefix:
        objects = list_objects(config, args.prefix, args.list_max_keys)
        download_keys.extend(item.key for item in objects)
        print(f"Listed download objects: prefix={args.prefix} count={len(objects)}", flush=True)

    if args.seed_count > 0:
        seed_prefix = args.prefix or upload_prefix
        download_keys.extend(
            seed_download_objects(
                client,
                prefix=seed_prefix,
                count=args.seed_count,
                payload=download_payload,
                workers=args.seed_workers,
            )
        )

    if args.mode in {"download", "mixed", "batch"} and not download_keys:
        raise RuntimeError("download keys are empty; use --keys-file, --prefix, or --seed-count")

    print(
        "Starting OSS latency stress: "
        f"provider={config.provider} endpoint={config.endpoint} bucket={config.bucket} "
        f"mode={args.mode} workers={args.workers} duration_sec={args.duration_sec} "
        f"concurrent_batches={args.concurrent_batches} batch_inputs={args.batch_inputs} "
        f"uploads_per_input={args.uploads_per_input} download_workers={args.download_workers} "
        f"upload_workers={args.upload_workers} global_workers={args.global_workers} "
        f"download_object_size_kb={download_object_size_kb} upload_object_size_kb={upload_object_size_kb} "
        f"pool_connections={config.max_pool_connections} "
        f"download_keys={len(download_keys)} upload_prefix={upload_prefix}",
        flush=True,
    )

    stats = LatencyStats(slow_threshold_ms=args.slow_threshold_ms, slow_log_limit=args.slow_log_limit)
    limiter = OpLimiter(args.max_ops)
    started_at = time.perf_counter()
    stop_at = started_at + max(0.1, float(args.duration_sec))
    stop_progress = threading.Event()
    progress_thread = threading.Thread(
        target=print_progress,
        args=(stats, started_at, stop_progress, float(args.report_interval_sec)),
        daemon=True,
    )
    progress_thread.start()

    if args.mode == "batch":
        global_limiter = GlobalLimiter(args.global_workers)
        futures = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.concurrent_batches))) as executor:
            futures = [
                executor.submit(
                    run_batch_worker,
                    worker_id=worker_id,
                    client=client,
                    limiter=global_limiter,
                    upload_payload=upload_payload,
                    upload_prefix=upload_prefix,
                    download_keys=download_keys,
                    stop_at=stop_at,
                    op_limiter=limiter,
                    stats=stats,
                    batch_inputs=args.batch_inputs,
                    uploads_per_input=args.uploads_per_input,
                    download_workers=args.download_workers,
                    upload_workers=args.upload_workers,
                )
                for worker_id in range(max(1, int(args.concurrent_batches)))
            ]
            for future in as_completed(futures):
                future.result()
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(
                    run_worker,
                    worker_id=worker_id,
                    mode=args.mode,
                    client=client,
                    upload_payload=upload_payload,
                    upload_prefix=upload_prefix,
                    download_keys=download_keys,
                    stop_at=stop_at,
                    limiter=limiter,
                    stats=stats,
                )
                for worker_id in range(max(1, int(args.workers)))
            ]
        for future in as_completed(futures):
            future.result()

    stop_progress.set()
    progress_thread.join(timeout=2.0)
    elapsed = max(0.001, time.perf_counter() - started_at)
    snapshot = stats.snapshot()
    print("Final summary", flush=True)
    print(format_summary("upload", snapshot["upload"], elapsed), flush=True)
    print(format_summary("download", snapshot["download"], elapsed), flush=True)
    print(format_summary("batch_download", snapshot["batch_download"], elapsed), flush=True)
    print(format_summary("batch_upload", snapshot["batch_upload"], elapsed), flush=True)
    print(format_summary("batch_total", snapshot["batch_total"], elapsed), flush=True)
    if stats.slow_events():
        print("Slow events captured:", flush=True)
        for op, duration_ms, byte_count, key in stats.slow_events():
            print(f"  op={op} ms={duration_ms} bytes={byte_count} key={key}", flush=True)
    if stats.first_errors():
        print("First errors:", flush=True)
        for message in stats.first_errors():
            print(f"  {message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
