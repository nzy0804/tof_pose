from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_SPLIT = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 CVAT 导出的 COCO Keypoints 数据转换为 YOLO pose 格式。"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="CVAT 导出的 COCO Keypoints 解压目录。",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="目标数据集根目录，默认 dataset。",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "val", "test"],
        help="导入到哪个数据集划分，默认 train。",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="复制图像到 dataset/images/<split>。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的图像和标签。",
    )
    return parser.parse_args()


def ensure_layout(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (dataset_root / "meta").mkdir(parents=True, exist_ok=True)
    dataset_yaml = dataset_root / "dataset.yaml"
    if not dataset_yaml.exists():
        dataset_yaml.write_text(
            """path: .
train: images/train
val: images/val
test: images/test

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person
""",
            encoding="utf-8",
        )


def load_coco_export(input_dir: Path) -> tuple[dict, Path]:
    annotation_files = sorted((input_dir / "annotations").glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError("未找到 annotations/*.json 标注文件。")
    annotation_path = annotation_files[0]
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    image_root = input_dir / "images"
    if not image_root.exists():
        raise FileNotFoundError("未找到 images 目录。")
    return data, image_root


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (
        (x + w / 2.0) / width,
        (y + h / 2.0) / height,
        w / width,
        h / height,
    )


def coco_keypoints_to_yolo(keypoints: list[float], width: int, height: int) -> list[float]:
    values: list[float] = []
    for idx in range(0, len(keypoints), 3):
        x, y, v = keypoints[idx : idx + 3]
        values.extend([x / width, y / height, v])
    return values


def build_label_line(annotation: dict, image_info: dict, class_id: int = 0) -> str:
    width = image_info["width"]
    height = image_info["height"]
    x_center, y_center, box_w, box_h = coco_bbox_to_yolo(annotation["bbox"], width, height)
    kpt_values = coco_keypoints_to_yolo(annotation["keypoints"], width, height)

    parts = [
        str(class_id),
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{box_w:.6f}",
        f"{box_h:.6f}",
    ]
    for idx, value in enumerate(kpt_values):
        if idx % 3 == 2:
            parts.append(str(int(value)))
        else:
            parts.append(f"{value:.6f}")
    return " ".join(parts)


def find_image_path(image_root: Path, file_name: str) -> Path:
    candidates = list(image_root.rglob(file_name))
    if not candidates:
        raise FileNotFoundError(f"未找到图像文件: {file_name}")
    return candidates[0]


def convert_dataset(
    input_dir: Path,
    dataset_root: Path,
    split: str,
    copy_images: bool,
    overwrite: bool,
) -> None:
    data, image_root = load_coco_export(input_dir)
    images = {item["id"]: item for item in data.get("images", [])}

    categories = {item["id"]: item for item in data.get("categories", [])}
    category_to_class = {
        category_id: idx for idx, category_id in enumerate(sorted(categories.keys()))
    }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    image_output_dir = dataset_root / "images" / split
    label_output_dir = dataset_root / "labels" / split

    converted_count = 0
    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        source_image = find_image_path(image_root, file_name)
        target_image = image_output_dir / file_name
        target_label = label_output_dir / f"{Path(file_name).stem}.txt"

        if copy_images:
            if target_image.exists() and not overwrite:
                pass
            else:
                shutil.copy2(source_image, target_image)

        if target_label.exists() and not overwrite:
            print(f"[convert_coco] 跳过已存在标签: {target_label}")
            continue

        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            class_id = category_to_class.get(annotation["category_id"], 0)
            lines.append(build_label_line(annotation, image_info, class_id))

        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted_count += 1

    print(f"[convert_coco] 已转换 {converted_count} 张图像到 split={split}")
    if copy_images:
        print(f"[convert_coco] 图像已复制到: {image_output_dir}")
    print(f"[convert_coco] 标签输出目录: {label_output_dir}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    ensure_layout(dataset_root)
    convert_dataset(
        input_dir=input_dir,
        dataset_root=dataset_root,
        split=args.split,
        copy_images=args.copy_images,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
