from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
DEFAULT_SPLITS = (0.7, 0.2, 0.1)


@dataclass
class VideoAssignment:
    source: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从视频抽帧并生成 tof_pose 训练目录。"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="输入视频文件或目录，可传多个。",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="数据集根目录，默认是 dataset。",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=5,
        help="每隔多少帧抽一张，默认 5。",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="每个视频最多抽多少张，0 表示不限制。",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        choices=[".jpg", ".png"],
        help="输出图像扩展名。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于划分数据集。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLITS[0],
        help="训练集比例，默认 0.7。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLITS[1],
        help="验证集比例，默认 0.2。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLITS[2],
        help="测试集比例，默认 0.1。",
    )
    parser.add_argument(
        "--touch-labels",
        action="store_true",
        help="为每张抽帧图像自动创建同名空标签文件。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标图像已存在则覆盖。",
    )
    return parser.parse_args()


def collect_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            videos.append(path.resolve())
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
                    videos.append(candidate.resolve())
    unique_videos = sorted(dict.fromkeys(videos))
    return unique_videos


def ensure_dataset_layout(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> dict[str, int]:
    split_names = ("train", "val", "test")
    raw = [r * total for r in ratios]
    base = [math.floor(v) for v in raw]
    remainder = total - sum(base)

    ranked = sorted(
        enumerate(raw),
        key=lambda item: item[1] - math.floor(item[1]),
        reverse=True,
    )
    for idx, _ in ranked[:remainder]:
        base[idx] += 1

    if total >= 3:
        for idx in range(3):
            if ratios[idx] > 0 and base[idx] == 0:
                donor = max(range(3), key=lambda i: base[i])
                if base[donor] > 1:
                    base[donor] -= 1
                    base[idx] += 1

    return {name: count for name, count in zip(split_names, base)}


def assign_videos_to_splits(
    videos: list[Path],
    ratios: tuple[float, float, float],
    seed: int,
) -> list[VideoAssignment]:
    rng = random.Random(seed)
    shuffled = videos[:]
    rng.shuffle(shuffled)
    counts = allocate_counts(len(shuffled), ratios)

    assignments: list[VideoAssignment] = []
    cursor = 0
    for split in ("train", "val", "test"):
        for _ in range(counts[split]):
            assignments.append(VideoAssignment(source=shuffled[cursor], split=split))
            cursor += 1
    return assignments


def write_dataset_yaml(dataset_root: Path) -> None:
    content = """path: .
train: images/train
val: images/val
test: images/test

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
"""
    (dataset_root / "dataset.yaml").write_text(content, encoding="utf-8")


def sanitize_name(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in path.stem)


def extract_frames(
    assignment: VideoAssignment,
    dataset_root: Path,
    sample_every: int,
    max_frames_per_video: int,
    image_ext: str,
    touch_labels: bool,
    overwrite: bool,
) -> list[dict[str, str]]:
    cap = cv2.VideoCapture(str(assignment.source))
    if not cap.isOpened():
        print(f"[prepare_dataset] 无法打开视频: {assignment.source}")
        return []

    split = assignment.split
    output_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    stem = sanitize_name(assignment.source)
    rows: list[dict[str, str]] = []
    saved_count = 0
    frame_index = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % sample_every != 0:
            frame_index += 1
            continue

        image_name = f"{stem}_f{frame_index:06d}{image_ext}"
        image_path = output_dir / image_name
        label_path = label_dir / f"{stem}_f{frame_index:06d}.txt"

        if image_path.exists() and not overwrite:
            print(f"[prepare_dataset] 跳过已存在文件: {image_path}")
        else:
            cv2.imwrite(str(image_path), frame)

        if touch_labels:
            label_path.touch(exist_ok=True)

        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        rows.append(
            {
                "split": split,
                "source_video": str(assignment.source),
                "frame_index": str(frame_index),
                "timestamp_sec": f"{timestamp_sec:.4f}",
                "image_path": str(image_path.relative_to(dataset_root)),
                "label_path": str(label_path.relative_to(dataset_root)),
            }
        )

        saved_count += 1
        frame_index += 1
        if max_frames_per_video > 0 and saved_count >= max_frames_per_video:
            break

    cap.release()
    print(
        f"[prepare_dataset] 已处理 {assignment.source.name} -> {split}，抽帧 {saved_count} 张"
    )
    return rows


def write_manifest(dataset_root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = dataset_root / "meta" / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split",
                "source_video",
                "frame_index",
                "timestamp_sec",
                "image_path",
                "label_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r < 0 for r in ratios):
        raise ValueError("划分比例不能为负数。")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("划分比例之和必须大于 0。")
    return tuple(r / total for r in ratios)


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    videos = collect_videos(args.input)
    if not videos:
        raise SystemExit("未找到可处理的视频文件。")

    dataset_root = Path(args.dataset_root).resolve()
    ensure_dataset_layout(dataset_root)
    write_dataset_yaml(dataset_root)

    assignments = assign_videos_to_splits(videos, ratios, args.seed)
    print(f"[prepare_dataset] 共发现 {len(videos)} 个视频文件")
    for item in assignments:
        print(f"[prepare_dataset] {item.source.name} -> {item.split}")

    manifest_rows: list[dict[str, str]] = []
    for assignment in assignments:
        manifest_rows.extend(
            extract_frames(
                assignment=assignment,
                dataset_root=dataset_root,
                sample_every=max(1, args.sample_every),
                max_frames_per_video=max(0, args.max_frames_per_video),
                image_ext=args.image_ext,
                touch_labels=args.touch_labels,
                overwrite=args.overwrite,
            )
        )

    write_manifest(dataset_root, manifest_rows)
    print(f"[prepare_dataset] 完成，manifest 已写入: {dataset_root / 'meta' / 'manifest.csv'}")
    print(f"[prepare_dataset] 训练配置文件: {dataset_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
