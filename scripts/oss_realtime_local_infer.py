"""Poll OSS images for one device and run the local RealtimePoseEngine.

Example:
    python scripts/oss_realtime_local_infer.py \
        --device-id sim-device-001 \
        --prefix frames/sim-device-001/ \
        --batch-size 20 \
        --device cuda:0 \
        --output-format jpeg \
        --jpeg-quality 60 \
        --start oldest \
        --max-batches 1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tof_pose.object_storage import ObjectStorageConfig, create_object_storage_client
from tof_pose.realtime_service import CPU_WORKER_MODE_PROCESS, CPU_WORKER_MODE_THREAD, INPUT_MODALITIES, INPUT_MODALITY_DEPTH, RealtimePoseEngine


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".bin")


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    last_modified: int
    size: int


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
        last_modified = int(getattr(item, "last_modified", 0) or 0)
        size = int(getattr(item, "size", 0) or 0)
        objects.append(ObjectInfo(key=key, last_modified=last_modified, size=size))
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
            size = int(item.get("Size") or 0)
            objects.append(ObjectInfo(key=key, last_modified=last_modified, size=size))
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
        raise ValueError(f"unsupported provider: {config.provider}")
    return sorted(objects, key=lambda item: (item.last_modified, item.key))


def should_keep_object(key: str, args: argparse.Namespace) -> bool:
    lowered = key.lower()
    if args.image_suffix and not lowered.endswith(tuple(args.image_suffix)):
        return False
    if args.filter_device_id and args.device_id not in key:
        return False
    return True


def frame_id_from_key(key: str) -> str:
    stem = Path(key).name
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem or key.replace("/", "_")


def download_batch(client, objects: list[ObjectInfo], workers: int) -> list[tuple[str, bytes, str]]:
    def download_one(item: ObjectInfo) -> tuple[str, bytes, str]:
        return frame_id_from_key(item.key), client.get_bytes(item.key), item.key

    if len(objects) <= 1 or workers <= 1:
        return [download_one(item) for item in objects]

    ordered: dict[str, tuple[str, bytes, str]] = {}
    with ThreadPoolExecutor(max_workers=min(max(1, workers), len(objects))) as executor:
        futures = {executor.submit(download_one, item): item.key for item in objects}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    return [ordered[item.key] for item in objects]


def write_result_images(output_dir: Path, batch_index: int, results: list[dict]) -> None:
    batch_dir = output_dir / f"batch_{batch_index:06d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for idx, result in enumerate(results):
        pseudo_format = str(result.get("pseudo_color_image_format") or "png")
        skeleton_format = str(result.get("skeleton_contour_image_format") or "png")
        frame_id = str(result.get("frame_id") or f"result_{idx:04d}").replace("/", "_")
        (batch_dir / f"{idx:04d}_{frame_id}_pseudo.{pseudo_format}").write_bytes(
            result.get("pseudo_color_image", b"")
        )
        (batch_dir / f"{idx:04d}_{frame_id}_skeleton_contour.{skeleton_format}").write_bytes(
            result.get("skeleton_contour_image", b"")
        )


def summarize_results(results: list[dict]) -> dict:
    return {
        "result_count": len(results),
        "person_counts": [int(item.get("person_count", 0) or 0) for item in results],
        "person_status": [str(item.get("person_status", "") or "") for item in results],
        "person_distance": [str(item.get("person_distance", "") or "") for item in results],
        "action_level": [str(item.get("action_level", "") or "") for item in results],
    }


def build_engine(args: argparse.Namespace) -> RealtimePoseEngine:
    engine = RealtimePoseEngine(
        model_path=Path(args.model_path) if args.model_path else None,
        pose_model_path=Path(args.pose_model_path) if args.pose_model_path else None,
        stateless=args.stateless,
        pose_only=args.pose_only,
        pose_validate_seg=not args.no_pose_validate,
        pose_fallback=not args.no_pose_fallback,
        seg_conf_threshold=args.seg_conf,
        pose_conf_threshold=args.pose_conf,
        pose_kpt_conf_threshold=args.pose_kpt_conf,
        pose_gate_kpt_conf_threshold=args.pose_gate_kpt_conf,
        pose_kpt_min_points=args.pose_kpt_min_points,
        mask_threshold=args.mask_threshold,
        mask_min_area_ratio=args.mask_min_area_ratio,
        mask_max_area_ratio=args.mask_max_area_ratio,
        contour_new_track_conf_threshold=args.contour_new_conf,
        contour_existing_track_conf_threshold=args.contour_existing_conf,
        device=args.device,
        render_workers=args.render_workers,
        decode_workers=args.decode_workers,
        png_compression=args.png_compression,
        output_format=args.output_format,
        jpeg_quality=args.jpeg_quality,
        input_modality=args.input_modality,
        ir_preprocess=args.ir_preprocess,
        cpu_worker_mode=args.cpu_worker_mode,
        cpu_process_start_method=args.cpu_process_start_method,
        instance_name="local-oss",
    )
    if not args.no_warmup:
        engine.warmup(batch_size=args.warmup_batch_size)
    return engine


def process_batch(
    *,
    batch_index: int,
    client,
    engine: RealtimePoseEngine,
    objects: list[ObjectInfo],
    output_dir: Path,
    download_workers: int,
    save_images: bool,
) -> None:
    download_start = time.perf_counter()
    downloaded = download_batch(client, objects, download_workers)
    download_ms = int((time.perf_counter() - download_start) * 1000)

    infer_start = time.perf_counter()
    frames = [(frame_id, data) for frame_id, data, _key in downloaded]
    results = engine.infer_batch(frames)
    infer_ms = int((time.perf_counter() - infer_start) * 1000)

    if save_images:
        write_result_images(output_dir, batch_index, results)

    summary = summarize_results(results)
    summary.update(
        {
            "batch_index": batch_index,
            "input_count": len(objects),
            "download_ms": download_ms,
            "infer_ms": infer_ms,
            "first_key": objects[0].key if objects else "",
            "last_key": objects[-1].key if objects else "",
            "saved_dir": str(output_dir / f"batch_{batch_index:06d}") if save_images else "",
        }
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def iter_new_objects(
    *,
    config: ObjectStorageConfig,
    args: argparse.Namespace,
    seen: set[str],
) -> list[ObjectInfo]:
    objects = list_objects(config, args.prefix, args.list_max_keys)
    filtered = [item for item in objects if should_keep_object(item.key, args)]
    new_items = [item for item in filtered if item.key not in seen]
    for item in new_items:
        seen.add(item.key)
    return new_items


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime local inference from OSS images for one device.")
    parser.add_argument("--device-id", required=True, help="device_id to test; object keys are filtered by this value by default")
    parser.add_argument("--prefix", default=None, help="OSS object prefix to poll; defaults to OSS_INPUT_PREFIX or device_id")
    parser.add_argument("--env-file", default=None, help="optional env file containing OSS_* variables")
    parser.add_argument("--start", choices=("latest", "oldest"), default="latest", help="latest skips existing objects, oldest processes from the first listed object")
    parser.add_argument("--initial-backfill", type=int, default=0, help="process the newest N existing objects before realtime polling")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--batch-timeout", type=float, default=1.0, help="flush partial batch after this many seconds; 0 disables partial flush")
    parser.add_argument("--max-batches", type=int, default=0, help="stop after N processed batches; 0 means run forever")
    parser.add_argument("--list-max-keys", type=int, default=1000)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--filter-device-id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-suffix", action="append", default=None, help="allowed suffix; may be repeated. Default: png/jpg/jpeg/bmp/bin")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "oss_realtime_local_infer"))
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--oss-provider", default=None, choices=("aliyun", "s3"))
    parser.add_argument("--oss-endpoint", default=None)
    parser.add_argument("--oss-bucket", default=None)
    parser.add_argument("--oss-access-key-id", default=None)
    parser.add_argument("--oss-access-key-secret", default=None)
    parser.add_argument("--oss-region", default=None)
    parser.add_argument("--oss-security-token", default=None)
    parser.add_argument("--oss-max-pool-connections", default=None, type=int)

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--pose-model-path", default=None)
    parser.add_argument("--seg-conf", default=None, type=float)
    parser.add_argument("--pose-conf", default=None, type=float)
    parser.add_argument("--pose-kpt-conf", default=None, type=float)
    parser.add_argument("--pose-gate-kpt-conf", default=None, type=float)
    parser.add_argument("--pose-kpt-min-points", default=4, type=int)
    parser.add_argument("--mask-threshold", default=0.5, type=float)
    parser.add_argument("--mask-min-area-ratio", default=None, type=float)
    parser.add_argument("--mask-max-area-ratio", default=None, type=float)
    parser.add_argument("--contour-new-conf", default=None, type=float)
    parser.add_argument("--contour-existing-conf", default=None, type=float)
    parser.add_argument("--device", default=None, help="local inference device, for example cuda:0 or cpu")
    parser.add_argument("--render-workers", default=1, type=int)
    parser.add_argument("--decode-workers", default=1, type=int)
    parser.add_argument("--cpu-worker-mode", default=CPU_WORKER_MODE_THREAD, choices=(CPU_WORKER_MODE_THREAD, CPU_WORKER_MODE_PROCESS))
    parser.add_argument("--cpu-process-start-method", default="auto", choices=("auto", "fork", "spawn", "forkserver"))
    parser.add_argument("--png-compression", default=1, type=int)
    parser.add_argument("--output-format", default="jpeg", choices=("png", "jpeg", "jpg"))
    parser.add_argument("--jpeg-quality", default=60, type=int)
    parser.add_argument("--input-modality", default=INPUT_MODALITY_DEPTH, choices=INPUT_MODALITIES)
    parser.add_argument("--ir-preprocess", action="store_true", help="enable median filtering plus CLAHE for infrared grayscale inputs")
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument("--pose-only", action="store_true")
    parser.add_argument("--no-pose-validate", action="store_true")
    parser.add_argument("--no-pose-fallback", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-batch-size", default=20, type=int)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(args.env_file)
    if args.prefix is None:
        args.prefix = os.environ.get("OSS_INPUT_PREFIX") or args.device_id
    if args.image_suffix is None:
        args.image_suffix = list(IMAGE_SUFFIXES)
    else:
        args.image_suffix = [item.lower() if item.startswith(".") else f".{item.lower()}" for item in args.image_suffix]

    config = ObjectStorageConfig.from_values(
        provider=args.oss_provider,
        endpoint=args.oss_endpoint,
        bucket=args.oss_bucket,
        access_key_id=args.oss_access_key_id,
        access_key_secret=args.oss_access_key_secret,
        region=args.oss_region,
        security_token=args.oss_security_token,
        max_pool_connections=args.oss_max_pool_connections,
    )
    if config is None:
        raise SystemExit("OSS config missing. Set OSS_* env vars or pass --oss-* arguments.")

    client = create_object_storage_client(config)
    if client is None:
        raise SystemExit("failed to create object storage client")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine(args)
    seen: set[str] = set()
    pending: list[ObjectInfo] = []
    pending_since: float | None = None
    batch_index = 0

    existing = [item for item in list_objects(config, args.prefix, args.list_max_keys) if should_keep_object(item.key, args)]
    if args.start == "latest":
        if args.initial_backfill > 0:
            pending.extend(existing[-args.initial_backfill :])
            seen.update(item.key for item in existing)
            pending_since = time.perf_counter() if pending else None
        else:
            seen.update(item.key for item in existing)
            print(json.dumps({"event": "primed", "seen": len(seen), "prefix": args.prefix}, ensure_ascii=False), flush=True)
    else:
        pending.extend(existing)
        seen.update(item.key for item in existing)
        pending_since = time.perf_counter() if pending else None

    while True:
        if args.max_batches and batch_index >= args.max_batches:
            return 0

        if not pending or len(pending) < args.batch_size:
            new_items = iter_new_objects(config=config, args=args, seen=seen)
            if new_items:
                pending.extend(new_items)
                if pending_since is None:
                    pending_since = time.perf_counter()

        should_flush = len(pending) >= args.batch_size
        if (
            not should_flush
            and pending
            and args.batch_timeout > 0
            and pending_since is not None
            and (time.perf_counter() - pending_since) >= args.batch_timeout
        ):
            should_flush = True

        if should_flush:
            batch_objects = pending[: args.batch_size]
            del pending[: len(batch_objects)]
            if pending:
                pending_since = time.perf_counter()
            else:
                pending_since = None
            batch_index += 1
            process_batch(
                batch_index=batch_index,
                client=client,
                engine=engine,
                objects=batch_objects,
                output_dir=output_dir,
                download_workers=args.download_workers,
                save_images=args.save_images,
            )
            continue

        time.sleep(max(0.1, float(args.poll_interval)))


if __name__ == "__main__":
    raise SystemExit(main())
