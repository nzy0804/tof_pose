from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tof_pose.paths import DEFAULT_MODEL_PATH, DEFAULT_POSE_MODEL_PATH
from tof_pose.person_distance import estimate_person_distance_from_mask
from tof_pose.pose_drawing import draw_stick_figure


CONF_THRESHOLD = 0.25
TRACKER_CONFIG = "botsort.yaml"
SEG_INFER_IMGSZ = 320
POSE_INFER_IMGSZ = 320
POSE_INFER_INTERVAL = 2
MASK_BLEND_ALPHA = 0.25
DISPLAY_SIZE = (320, 320)
DISPLAY_SCALE = 3

DISPLAY_MODE_BOTH = "both"
DISPLAY_MODE_CONTOUR_ONLY = "contour"
DISPLAY_MODE_SKELETON_ONLY = "skeleton"
DISPLAY_MODE_RAW_ONLY = "raw"


TRACK_COLORS = [
    (40, 210, 255),
    (120, 220, 80),
    (255, 180, 50),
    (255, 110, 170),
    (180, 130, 255),
    (100, 245, 210),
    (255, 120, 80),
    (90, 170, 255),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return TRACK_COLORS[abs(int(track_id)) % len(TRACK_COLORS)]


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in box_b[:4]]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return inter_area / denom


def _box_center_distance(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_a = np.array([(box_a[0] + box_a[2]) * 0.5, (box_a[1] + box_a[3]) * 0.5], dtype=np.float32)
    center_b = np.array([(box_b[0] + box_b[2]) * 0.5, (box_b[1] + box_b[3]) * 0.5], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))


def _match_pose_to_seg_tracks(
    seg_boxes: list[np.ndarray],
    seg_track_ids: list[int],
    pose_boxes: list[np.ndarray],
) -> dict[int, int]:
    if not seg_boxes or not seg_track_ids or not pose_boxes:
        return {}

    candidates: list[tuple[float, int, int]] = []
    for pose_idx, pose_box in enumerate(pose_boxes):
        for seg_idx, seg_box in enumerate(seg_boxes):
            iou = _box_iou(pose_box, seg_box)
            center_distance = _box_center_distance(pose_box, seg_box)
            if iou < 0.05 and center_distance > 90.0:
                continue
            score = iou - center_distance / 320.0
            candidates.append((score, pose_idx, seg_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    assigned_pose: set[int] = set()
    assigned_seg: set[int] = set()
    matched: dict[int, int] = {}
    for score, pose_idx, seg_idx in candidates:
        if score <= -0.25:
            continue
        if pose_idx in assigned_pose or seg_idx in assigned_seg:
            continue
        matched[pose_idx] = seg_track_ids[seg_idx]
        assigned_pose.add(pose_idx)
        assigned_seg.add(seg_idx)
    return matched


def _compute_pairwise_distances(
    records: list[tuple[int, np.ndarray, float | None]],
    frame_width: int,
) -> tuple[str, list[tuple[int, int, float]]]:
    pairs: list[tuple[int, int, float]] = []
    for idx in range(len(records)):
        track_id_a, box_a, distance_a = records[idx]
        if distance_a is None:
            continue
        center_x_a = float((box_a[0] + box_a[2]) * 0.5)
        normalized_x_a = (center_x_a - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
        lateral_x_a = float(distance_a) * float(np.tan(np.deg2rad(35.0) * normalized_x_a))

        for jdx in range(idx + 1, len(records)):
            track_id_b, box_b, distance_b = records[jdx]
            if distance_b is None:
                continue
            center_x_b = float((box_b[0] + box_b[2]) * 0.5)
            normalized_x_b = (center_x_b - frame_width * 0.5) / max(frame_width * 0.5, 1.0)
            lateral_x_b = float(distance_b) * float(np.tan(np.deg2rad(35.0) * normalized_x_b))
            spacing = float(np.hypot(lateral_x_a - lateral_x_b, float(distance_a) - float(distance_b)))
            if 0.0 < spacing <= 400.0:
                pairs.append((track_id_a, track_id_b, spacing))

    if not pairs:
        return "Pair Dist: N/A", []

    nearest = min(pairs, key=lambda item: item[2])
    return f"Pair Dist: {nearest[0]}-{nearest[1]} ~{nearest[2]:.1f}", pairs


@dataclass
class _FrameViewSet:
    output_image_S21: bytes
    output_image_S22: bytes
    output_image_S23: bytes
    output_image_S24: bytes
    person_count: int


class RealtimePoseEngine:
    def __init__(
        self,
        model_path: Path | None = None,
        pose_model_path: Path | None = None,
        *,
        stateless: bool = False,
        persist_tracks: bool | None = None,
        pose_only: bool = False,
        pose_conf_threshold: float | None = None,
        pose_kpt_conf_threshold: float | None = None,
        pose_kpt_min_points: int = 4,
    ) -> None:
        model_file = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        pose_model_file = Path(pose_model_path) if pose_model_path else DEFAULT_POSE_MODEL_PATH

        self.seg_model = YOLO(str(model_file))
        self.pose_model = YOLO(str(pose_model_file))
        self._lock = threading.Lock()
        self._stateless = bool(stateless)
        self._persist_tracks = (not self._stateless) if persist_tracks is None else bool(persist_tracks)
        self._pose_only = bool(pose_only)
        if pose_conf_threshold is None and self._pose_only:
            self._pose_conf_threshold = 0.15
        else:
            self._pose_conf_threshold = float(pose_conf_threshold) if pose_conf_threshold is not None else float(CONF_THRESHOLD)

        # Pose-only person counting heuristics.
        self._pose_kpt_conf_threshold = float(pose_kpt_conf_threshold) if pose_kpt_conf_threshold is not None else 0.20
        self._pose_kpt_min_points = int(pose_kpt_min_points)
        self._frame_idx = 0
        self._cached_pose_boxes: list[np.ndarray] = []
        self._cached_kpt_xy: np.ndarray | None = None
        self._cached_kpt_conf: np.ndarray | None = None
        self._warned_no_masks = False
        self._warned_no_keypoints = False
        self._last_views: tuple[bytes, bytes, bytes, bytes] | None = None

    def reset(self) -> None:
        """Reset per-stream caches so next infer behaves like the first frame."""
        self._frame_idx = 0
        self._cached_pose_boxes = []
        self._cached_kpt_xy = None
        self._cached_kpt_conf = None
        self._last_views = None

    def _decode_image(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cannot decode image")
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _analyze_frame(self, depth_gray: np.ndarray) -> dict:
        self._frame_idx += 1
        width, height = DISPLAY_SIZE
        depth_up = cv2.resize(depth_gray, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)
        color_img = cv2.applyColorMap(depth_up, cv2.COLORMAP_MAGMA)

        # Pose-only mode: do not rely on segmentation model outputs.
        if self._pose_only:
            pose_results = self.pose_model.predict(
                color_img,
                conf=self._pose_conf_threshold,
                classes=[0],
                imgsz=POSE_INFER_IMGSZ,
                verbose=False,
            )
            pose_result = pose_results[0] if pose_results else None
            kpt_xy_np = None
            kpt_conf_np = None
            if pose_result is not None and pose_result.keypoints is not None:
                keypoints_xy = pose_result.keypoints.xy
                keypoints_conf = pose_result.keypoints.conf
                if keypoints_xy is not None and keypoints_conf is not None:
                    kpt_xy_np = keypoints_xy.cpu().numpy()
                    kpt_conf_np = keypoints_conf.cpu().numpy()

            pose_draw_indices: list[int] = []
            if kpt_conf_np is not None:
                for idx in range(len(kpt_conf_np)):
                    confident_points = int(np.sum(kpt_conf_np[idx] >= self._pose_kpt_conf_threshold))
                    if confident_points >= self._pose_kpt_min_points:
                        pose_draw_indices.append(idx)

            return {
                "depth_up": depth_up,
                "color_img": color_img,
                "records": [],
                "person_count": int(len(pose_draw_indices)),
                "pair_text": "Pair Dist: N/A",
                "pair_stats": [],
                "tracked_labels": [],
                "kpt_xy_np": kpt_xy_np,
                "kpt_conf_np": kpt_conf_np,
                "pose_to_track": {},
                "validated_track_ids": set(),
                "pose_only": True,
                "pose_draw_indices": pose_draw_indices,
                "width": width,
                "height": height,
            }

        results = self.seg_model.track(
            color_img,
            conf=CONF_THRESHOLD,
            persist=self._persist_tracks,
            tracker=TRACKER_CONFIG,
            classes=[0],
            imgsz=SEG_INFER_IMGSZ,
            verbose=False,
        )
        result = results[0]

        run_pose_now = (self._frame_idx % POSE_INFER_INTERVAL == 0) or (self._cached_kpt_xy is None)
        pose_result = None
        if run_pose_now:
            pose_results = self.pose_model.predict(
                color_img,
                conf=CONF_THRESHOLD,
                classes=[0],
                imgsz=POSE_INFER_IMGSZ,
                verbose=False,
            )
            pose_result = pose_results[0] if pose_results else None
            if pose_result is not None:
                if pose_result.keypoints is not None:
                    keypoints_xy = pose_result.keypoints.xy
                    keypoints_conf = pose_result.keypoints.conf
                    if keypoints_xy is not None and keypoints_conf is not None:
                        self._cached_kpt_xy = keypoints_xy.cpu().numpy()
                        self._cached_kpt_conf = keypoints_conf.cpu().numpy()
                elif not self._warned_no_keypoints:
                    print(
                        "[tof_pose] 当前姿态模型没有输出 keypoints，请改用 YOLO pose 权重，例如 yolo11n-pose.pt。",
                        flush=True,
                    )
                    self._warned_no_keypoints = True

                if pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    self._cached_pose_boxes = [box.copy() for box in pose_result.boxes.xyxy.cpu().numpy()]

        pose_boxes_np = self._cached_pose_boxes
        kpt_xy_np = self._cached_kpt_xy
        kpt_conf_np = self._cached_kpt_conf

        if result.boxes is not None and len(result.boxes) > 0 and result.masks is None and not self._warned_no_masks:
            print(
                "[tof_pose] 当前模型没有输出 segmentation masks，请改用 YOLO segment 权重，例如 yolo11n-seg.pt。",
                flush=True,
            )
            self._warned_no_masks = True

        records: list[dict] = []
        person_count = 0
        tracked_labels: list[str] = []
        pair_records: list[tuple[int, np.ndarray, float | None]] = []
        pose_to_track: dict[int, int] = {}
        validated_track_ids: set[int] = set()

        if result.boxes is not None and len(result.boxes) > 0 and result.masks is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            track_ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else list(range(1, len(boxes_xyxy) + 1))
            )
            masks_data = result.masks.data.cpu().numpy()

            match_count = min(len(boxes_xyxy), len(masks_data), len(track_ids))
            seg_boxes_for_match = [boxes_xyxy[idx].copy() for idx in range(match_count)]
            seg_track_ids_for_match = [int(track_ids[idx]) for idx in range(match_count)]

            if pose_boxes_np:
                pose_to_track = _match_pose_to_seg_tracks(seg_boxes_for_match, seg_track_ids_for_match, pose_boxes_np)
                if kpt_conf_np is not None:
                    for pose_idx, track_id in pose_to_track.items():
                        if pose_idx >= len(kpt_conf_np):
                            continue
                        confident_points = int(np.sum(kpt_conf_np[pose_idx] >= 0.35))
                        if confident_points >= 4:
                            validated_track_ids.add(int(track_id))

            for idx in range(match_count):
                box = boxes_xyxy[idx]
                track_id = int(track_ids[idx])
                if track_id not in validated_track_ids:
                    continue
                track_color = _track_color(track_id)
                mask = (masks_data[idx] > 0.5).astype(np.uint8)
                estimate = estimate_person_distance_from_mask(depth_up, box, mask)
                person_count += 1

                if estimate.distance is None:
                    tracked_labels.append(f"{track_id}:N/A")
                    label = f"ID {track_id} Dist=N/A"
                else:
                    tracked_labels.append(f"{track_id}:{estimate.distance:.1f}")
                    label = f"ID {track_id} Dist~{estimate.distance:.1f}"

                pair_records.append((track_id, box.copy(), estimate.distance))
                records.append(
                    {
                        "track_id": track_id,
                        "box": box.copy(),
                        "mask": mask,
                        "contour": estimate.contour,
                        "anchor": getattr(estimate, "anchor", None),
                        "distance": estimate.distance,
                        "label": label,
                        "track_color": track_color,
                    }
                )

        if kpt_xy_np is not None and kpt_conf_np is not None:
            for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                matched_track = pose_to_track.get(idx)
                if matched_track is None or matched_track not in validated_track_ids:
                    continue

        pair_text, pair_stats = _compute_pairwise_distances(pair_records, width)
        return {
            "depth_up": depth_up,
            "color_img": color_img,
            "records": records,
            "person_count": person_count,
            "pair_text": pair_text,
            "pair_stats": pair_stats,
            "tracked_labels": tracked_labels,
            "kpt_xy_np": kpt_xy_np,
            "kpt_conf_np": kpt_conf_np,
            "pose_to_track": pose_to_track,
            "validated_track_ids": validated_track_ids,
            "pose_only": False,
            "pose_draw_indices": [],
            "width": width,
            "height": height,
        }

    def _render_display(self, analysis: dict, display_mode: str) -> np.ndarray:
        display = analysis["color_img"].copy()
        records = analysis["records"]
        kpt_xy_np = analysis["kpt_xy_np"]
        kpt_conf_np = analysis["kpt_conf_np"]
        pose_to_track = analysis["pose_to_track"]
        validated_track_ids = analysis["validated_track_ids"]
        pose_only = bool(analysis.get("pose_only", False))
        pose_draw_indices = analysis.get("pose_draw_indices") or []
        width = analysis["width"]
        height = analysis["height"]

        if display_mode in (DISPLAY_MODE_BOTH, DISPLAY_MODE_CONTOUR_ONLY):
            for record in records:
                box = record["box"]
                track_color = record["track_color"]
                contour = record["contour"]
                mask = record["mask"]
                label = record["label"]

                if contour is not None and display_mode != DISPLAY_MODE_SKELETON_ONLY:
                    anchor = record.get("anchor")
                    if anchor is not None and len(anchor) >= 2:
                        offset_x, offset_y = int(anchor[0]), int(anchor[1])
                    else:
                        offset_x, offset_y = int(round(box[0])), int(round(box[1]))
                    shifted_contour = contour + np.array([[[offset_x, offset_y]]])
                    cv2.drawContours(display, [shifted_contour], -1, track_color, 2, cv2.LINE_AA)

                mask_overlay = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                selected = mask_overlay > 0
                if np.any(selected):
                    blended = (
                        display[selected].astype(np.float32) * (1.0 - MASK_BLEND_ALPHA)
                        + np.array(track_color, dtype=np.float32) * MASK_BLEND_ALPHA
                    )
                    display[selected] = blended.astype(np.uint8)

                x1 = max(0, int(round(box[0])))
                y1 = max(18, int(round(box[1])) - 8)

        if display_mode in (DISPLAY_MODE_BOTH, DISPLAY_MODE_SKELETON_ONLY) and kpt_xy_np is not None and kpt_conf_np is not None:
            if pose_only:
                for idx in pose_draw_indices:
                    if idx < 0 or idx >= len(kpt_xy_np) or idx >= len(kpt_conf_np):
                        continue
                    color_override = _track_color(idx + 1)
                    draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=color_override)
            else:
                for idx in range(min(len(kpt_xy_np), len(kpt_conf_np))):
                    matched_track = pose_to_track.get(idx)
                    if matched_track is None or matched_track not in validated_track_ids:
                        continue
                    color_override = _track_color(matched_track)
                    draw_stick_figure(display, kpt_xy_np[idx], kpt_conf_np[idx], color_override=color_override)

        return display

    def _render_gray_depth(self, analysis: dict) -> np.ndarray:
        """Return 320x320 uint8 grayscale depth image."""
        depth_up = analysis.get("depth_up")
        if depth_up is None:
            raise RuntimeError("missing depth_up")
        if not isinstance(depth_up, np.ndarray):
            raise RuntimeError("invalid depth_up")
        if depth_up.dtype != np.uint8:
            depth_up = depth_up.astype(np.uint8, copy=False)
        return depth_up

    def _render_color_depth(self, analysis: dict) -> np.ndarray:
        """Return 320x320 BGR colormap depth image."""
        color_img = analysis.get("color_img")
        if color_img is None:
            raise RuntimeError("missing color_img")
        return color_img

    def _encode_png(self, img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("failed to encode PNG")
        return buf.tobytes()

    def infer(self, frame_id: str, image_bytes: bytes) -> dict:
        with self._lock:
            if self._stateless:
                self.reset()

            start = time.time()
            depth = self._decode_image(image_bytes)
            analysis = self._analyze_frame(depth)

            # Match IoT doc semantics/order:
            # S1: gray depth, S2: color depth, S3: skeleton, S4: contour.
            current_views = (
                self._encode_png(self._render_gray_depth(analysis)),
                self._encode_png(self._render_color_depth(analysis)),
                self._encode_png(self._render_display(analysis, DISPLAY_MODE_SKELETON_ONLY)),
                self._encode_png(self._render_display(analysis, DISPLAY_MODE_CONTOUR_ONLY)),
            )

            prev_views = self._last_views or (b"", b"", b"", b"")
            self._last_views = current_views

            elapsed = int((time.time() - start) * 1000)
            return {
                "frame_id": frame_id,
                "output_image_S11": prev_views[0],
                "output_image_S12": prev_views[1],
                "output_image_S13": prev_views[2],
                "output_image_S14": prev_views[3],
                "output_image_S21": current_views[0],
                "output_image_S22": current_views[1],
                "output_image_S23": current_views[2],
                "output_image_S24": current_views[3],
                "person_count": int(analysis["person_count"]),
                "processing_time_ms": elapsed,
            }
