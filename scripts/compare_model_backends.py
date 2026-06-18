#!/usr/bin/env python3
"""Compare PT and TensorRT backend outputs for the MaixSense realtime engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import cv2
import numpy as np

try:
    import tensorrt as _trt

    if not hasattr(_trt, "__version__"):
        _trt.__version__ = "10.0.0"
    if hasattr(_trt, "Runtime") and not hasattr(_trt.Runtime, "__enter__"):
        _trt.Runtime.__enter__ = lambda self: self
        _trt.Runtime.__exit__ = lambda self, exc_type, exc, tb: None
    if not hasattr(_trt, "nptype"):
        _trt_dtype_map = {
            getattr(_trt, "float32", None): np.float32,
            getattr(_trt, "float16", None): np.float16,
            getattr(_trt, "int8", None): np.int8,
            getattr(_trt, "int32", None): np.int32,
            getattr(_trt, "int64", None): np.int64,
            getattr(_trt, "uint8", None): np.uint8,
            getattr(_trt, "bool", None): np.bool_,
        }
        _trt_dtype_map.pop(None, None)
        _trt.nptype = lambda dtype: _trt_dtype_map[dtype]
except Exception:
    pass

from tof_pose.realtime_service import (
    CONF_THRESHOLD,
    POSE_INFER_IMGSZ,
    SEG_INFER_IMGSZ,
    TRACKER_CONFIG,
    RealtimePoseEngine,
)


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


def _result_count(result) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    try:
        return int(len(boxes))
    except Exception:
        return 0


def _summarize_seg_result(result) -> dict:
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    count = _result_count(result)
    box_values: list[list[float]] = []
    conf_values: list[float] = []
    mask_areas: list[int] = []
    track_ids: list[int] = []

    if boxes is not None and count > 0:
        if getattr(boxes, "xyxy", None) is not None:
            box_values = np.round(boxes.xyxy.cpu().numpy(), 2).tolist()
        if getattr(boxes, "conf", None) is not None:
            conf_values = np.round(boxes.conf.cpu().numpy(), 4).tolist()
        if getattr(boxes, "id", None) is not None:
            track_ids = [int(v) for v in boxes.id.int().cpu().tolist()]
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.cpu().numpy()
        mask_areas = [int(np.count_nonzero(mask > 0.5)) for mask in mask_data[:count]]

    return {
        "count": count,
        "conf": conf_values,
        "boxes": box_values,
        "track_ids": track_ids,
        "mask_areas": mask_areas,
        "has_masks": bool(masks is not None),
    }


def _summarize_pose_result(result) -> dict:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    count = _result_count(result)
    box_values: list[list[float]] = []
    conf_values: list[float] = []
    kpt_conf_counts_020: list[int] = []
    kpt_conf_counts_035: list[int] = []

    if boxes is not None and count > 0:
        if getattr(boxes, "xyxy", None) is not None:
            box_values = np.round(boxes.xyxy.cpu().numpy(), 2).tolist()
        if getattr(boxes, "conf", None) is not None:
            conf_values = np.round(boxes.conf.cpu().numpy(), 4).tolist()
    if keypoints is not None and getattr(keypoints, "conf", None) is not None:
        kpt_conf = keypoints.conf.cpu().numpy()
        kpt_conf_counts_020 = [int(np.sum(row >= 0.20)) for row in kpt_conf]
        kpt_conf_counts_035 = [int(np.sum(row >= 0.35)) for row in kpt_conf]

    return {
        "count": count,
        "conf": conf_values,
        "boxes": box_values,
        "kpt_conf_points_ge_020": kpt_conf_counts_020,
        "kpt_conf_points_ge_035": kpt_conf_counts_035,
        "has_keypoints": bool(keypoints is not None),
    }


def _suffix_for_format(fmt: str) -> str:
    normalized = (fmt or "png").lower()
    if normalized == "jpeg":
        return "jpg"
    return normalized


def _write_pipeline_images(output_dir: Path, backend_name: str, results: list[dict]) -> None:
    backend_dir = output_dir / backend_name / "pipeline_images"
    backend_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        frame_id = str(result.get("frame_id", f"frame_{index:04d}"))
        pseudo_format = str(result.get("pseudo_color_image_format") or "png")
        skeleton_format = str(result.get("skeleton_contour_image_format") or "png")
        pseudo_suffix = _suffix_for_format(pseudo_format)
        skeleton_suffix = _suffix_for_format(skeleton_format)
        (backend_dir / f"{index:04d}_{frame_id}_pseudo.{pseudo_suffix}").write_bytes(
            bytes(result.get("pseudo_color_image") or b"")
        )
        (backend_dir / f"{index:04d}_{frame_id}_skeleton_contour.{skeleton_suffix}").write_bytes(
            bytes(result.get("skeleton_contour_image") or b"")
        )


def _run_backend(
    *,
    name: str,
    seg_path: Path,
    pose_path: Path,
    device: str,
    frames: list[tuple[str, bytes]],
    output_dir: Path,
    save_images: bool,
) -> dict:
    start = time.perf_counter()
    engine = RealtimePoseEngine(
        model_path=seg_path,
        pose_model_path=pose_path,
        stateless=True,
        device=device,
        render_workers=1,
        decode_workers=1,
        output_format="png",
        instance_name=name,
    )
    load_ms = int((time.perf_counter() - start) * 1000)

    decode_start = time.perf_counter()
    decoded_frames = engine._decode_frames(frames)
    source_frames = [
        {"frame_id": frame_id, "input_index": index, "depth": depth}
        for index, (frame_id, depth) in enumerate(decoded_frames)
    ]
    decode_ms = int((time.perf_counter() - decode_start) * 1000)

    prepare_start = time.perf_counter()
    color_imgs = engine._prepare_model_color_images(source_frames)
    prepare_ms = int((time.perf_counter() - prepare_start) * 1000)

    seg_start = time.perf_counter()
    seg_results = engine.seg_model.track(
        color_imgs,
        conf=CONF_THRESHOLD,
        persist=False,
        tracker=TRACKER_CONFIG,
        classes=[0],
        imgsz=SEG_INFER_IMGSZ,
        device=device,
        verbose=False,
    )
    seg_ms = int((time.perf_counter() - seg_start) * 1000)

    pose_start = time.perf_counter()
    pose_results = engine.pose_model.predict(
        color_imgs,
        conf=CONF_THRESHOLD,
        classes=[0],
        imgsz=POSE_INFER_IMGSZ,
        device=device,
        verbose=False,
    )
    pose_ms = int((time.perf_counter() - pose_start) * 1000)

    pipeline_start = time.perf_counter()
    pipeline_results = engine.infer_batch(frames)
    pipeline_ms = int((time.perf_counter() - pipeline_start) * 1000)

    if save_images:
        _write_pipeline_images(output_dir, name, pipeline_results)

    frame_summaries = []
    for index, (frame_id, _data) in enumerate(frames):
        current_outputs = [
            result
            for result in pipeline_results
            if int(result.get("input_index", -1)) == index and result.get("result_kind") == "current"
        ]
        frame_summaries.append(
            {
                "frame_id": frame_id,
                "seg": _summarize_seg_result(seg_results[index]),
                "pose": _summarize_pose_result(pose_results[index]),
                "pipeline_current_person_count": int(current_outputs[0].get("person_count", 0)) if current_outputs else 0,
            }
        )

    return {
        "name": name,
        "seg_path": str(seg_path),
        "pose_path": str(pose_path),
        "device": device,
        "timing_ms": {
            "load": load_ms,
            "decode": decode_ms,
            "prepare": prepare_ms,
            "seg": seg_ms,
            "pose": pose_ms,
            "pipeline": pipeline_ms,
        },
        "frames": frame_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PT and TensorRT MaixSense backends on the same images.")
    parser.add_argument("--input", action="append", required=True, help="input PNG/JPEG path; repeat for multiple frames")
    parser.add_argument("--seg-pt", required=True)
    parser.add_argument("--pose-pt", required=True)
    parser.add_argument("--seg-engine", required=True)
    parser.add_argument("--pose-engine", required=True)
    parser.add_argument("--pt-device", default="cpu")
    parser.add_argument("--engine-device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = [(Path(path).stem, _read_bytes(Path(path))) for path in args.input]

    report = {
        "inputs": [str(path) for path in args.input],
        "backends": [],
    }
    report["backends"].append(
        _run_backend(
            name="pt",
            seg_path=Path(args.seg_pt),
            pose_path=Path(args.pose_pt),
            device=args.pt_device,
            frames=frames,
            output_dir=output_dir,
            save_images=args.save_images,
        )
    )
    report["backends"].append(
        _run_backend(
            name="engine",
            seg_path=Path(args.seg_engine),
            pose_path=Path(args.pose_engine),
            device=args.engine_device,
            frames=frames,
            output_dir=output_dir,
            save_images=args.save_images,
        )
    )

    report_path = output_dir / "backend_compare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
