from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


LOG_PREFIX = "[offline_pseudo_depth_video]"

COLORMAPS = {
    "redblue": None,
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "bone": cv2.COLORMAP_BONE,
}


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_s, height_s = value.lower().split("x", 1)
        width = int(width_s)
        height = int(height_s)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expected WIDTHxHEIGHT, for example 320x320") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def _validate_odd_kernel(k: int, *, name: str) -> int:
    if k <= 0:
        return 0
    if k < 3:
        raise ValueError(f"{name} must be 0 or >= 3")
    if k % 2 == 0:
        k += 1
    return int(k)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        return gray

    gray_f = gray.astype(np.float32, copy=False)
    min_val = float(np.min(gray_f)) if gray_f.size else 0.0
    max_val = float(np.max(gray_f)) if gray_f.size else 0.0
    if max_val <= min_val:
        return np.zeros_like(gray_f, dtype=np.uint8)
    return cv2.normalize(gray_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _prepare_gray(
    gray: np.ndarray,
    *,
    median_k: int,
    clahe: cv2.CLAHE | None,
    output_size: tuple[int, int] | None,
) -> np.ndarray:
    if output_size is not None:
        gray = cv2.resize(gray, output_size, interpolation=cv2.INTER_LINEAR)
    if median_k > 0:
        gray = cv2.medianBlur(gray, median_k)
    if clahe is not None:
        gray = clahe.apply(gray)
    return gray


def _estimate_global_range(
    cap: cv2.VideoCapture,
    *,
    median_k: int,
    clahe: cv2.CLAHE | None,
    output_size: tuple[int, int] | None,
    low_percentile: float,
    high_percentile: float,
    sample_stride: int,
    max_frames: int,
) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    frame_idx = 0
    sample_stride = max(1, int(sample_stride))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        if frame_idx % sample_stride == 0:
            gray = _prepare_gray(
                _to_gray(frame),
                median_k=median_k,
                clahe=clahe,
                output_size=output_size,
            )
            samples.append(gray[::4, ::4].reshape(-1))

        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not samples:
        return 0.0, 255.0

    values = np.concatenate(samples).astype(np.float32, copy=False)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        return 0.0, 255.0
    return low, high


def _normalize_gray(gray: np.ndarray, *, low: float, high: float) -> np.ndarray:
    scale = 255.0 / max(float(high) - float(low), 1e-6)
    normalized = (gray.astype(np.float32) - float(low)) * scale
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _make_pseudo_depth(
    gray: np.ndarray,
    *,
    colormap: int | None,
    invert: bool,
    low: float,
    high: float,
) -> np.ndarray:
    normalized = _normalize_gray(gray, low=low, high=high)
    if invert:
        normalized = 255 - normalized
    if colormap is None:
        red = normalized
        green = np.zeros_like(normalized)
        blue = 255 - normalized
        return cv2.merge([blue, green, red])
    return cv2.applyColorMap(normalized, colormap)


def process_video(
    *,
    input_path: Path,
    output_path: Path,
    colormap_name: str,
    normalize_mode: str,
    invert_depth: bool,
    median_k: int,
    clahe_clip: float,
    clahe_grid: int,
    disable_clahe: bool,
    low_percentile: float,
    high_percentile: float,
    sample_stride: int,
    output_size: tuple[int, int] | None,
    side_by_side: bool,
    max_frames: int,
    fourcc_name: str,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    median_k = _validate_odd_kernel(int(median_k), name="--median-k")
    clahe = None
    if not disable_clahe:
        grid = max(1, int(clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(grid, grid))

    if output_size is None:
        output_size = (source_width, source_height)
    width, height = output_size

    low, high = 0.0, 255.0
    if normalize_mode == "global":
        low, high = _estimate_global_range(
            cap,
            median_k=median_k,
            clahe=clahe,
            output_size=output_size,
            low_percentile=float(low_percentile),
            high_percentile=float(high_percentile),
            sample_stride=int(sample_stride),
            max_frames=int(max_frames),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer_width = width * 2 if side_by_side else width
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        fps,
        (writer_width, height),
        True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open VideoWriter: {output_path}")

    print(f"{LOG_PREFIX} input={input_path}")
    print(f"{LOG_PREFIX} output={output_path}")
    print(f"{LOG_PREFIX} fps={fps:.3f} source_size={source_width}x{source_height} output_size={writer_width}x{height}")
    if colormap_name == "redblue":
        direction = "high-value-red/low-value-blue" if not invert_depth else "low-value-red/high-value-blue"
    else:
        direction = "high-value-bright/low-value-dark" if not invert_depth else "low-value-bright/high-value-dark"
    print(f"{LOG_PREFIX} frames={total_frames} normalize={normalize_mode} range=({low:.2f},{high:.2f}) colormap={colormap_name} direction={direction}")

    frame_idx = 0
    colormap = COLORMAPS[colormap_name]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        prepared = _prepare_gray(
            _to_gray(frame),
            median_k=median_k,
            clahe=clahe,
            output_size=output_size,
        )
        if normalize_mode == "frame":
            low = float(np.percentile(prepared, low_percentile))
            high = float(np.percentile(prepared, high_percentile))
            if high <= low:
                low, high = 0.0, 255.0

        pseudo = _make_pseudo_depth(prepared, colormap=colormap, invert=invert_depth, low=low, high=high)
        if side_by_side:
            original = cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
            frame_out = np.hstack([original, pseudo])
        else:
            frame_out = pseudo

        writer.write(frame_out)

        frame_idx += 1
        if frame_idx % 30 == 0:
            if total_frames > 0:
                percent = frame_idx / total_frames * 100.0
                print(f"{LOG_PREFIX} {frame_idx}/{total_frames} ({percent:.1f}%)", end="\r")
            else:
                print(f"{LOG_PREFIX} processed={frame_idx}", end="\r")

    cap.release()
    writer.release()
    print(f"\n{LOG_PREFIX} done frames={frame_idx} output={output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an offline infrared/grayscale video to pseudo-depth visualization.",
    )
    parser.add_argument("input", type=Path, help="Input offline video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/videos/<stem>_pseudo_depth.mp4)",
    )
    parser.add_argument(
        "--colormap",
        choices=sorted(COLORMAPS),
        default="magma",
        help="Colormap used for pseudo-depth visualization. Default magma maps larger normalized values to brighter colors.",
    )
    parser.add_argument(
        "--normalize",
        choices=("global", "frame"),
        default="global",
        help="global avoids flicker; frame maximizes contrast per frame.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert normalized values before applying the colormap. Use this only when near objects are lower-valued.",
    )
    parser.add_argument("--far-bright", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--median-k", type=int, default=5, help="Median blur kernel size, 0 to disable. Default: 5.")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE local contrast enhancement.")
    parser.add_argument("--clahe-clip", type=float, default=2.0, help="CLAHE clip limit. Default: 2.0.")
    parser.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tile grid size. Default: 8.")
    parser.add_argument("--low-percentile", type=float, default=1.0, help="Lower percentile for contrast clipping.")
    parser.add_argument("--high-percentile", type=float, default=99.0, help="Upper percentile for contrast clipping.")
    parser.add_argument("--sample-stride", type=int, default=5, help="Frame stride used for global range estimation.")
    parser.add_argument("--size", type=_parse_size, default=None, help="Resize output to WIDTHxHEIGHT, e.g. 320x320.")
    parser.add_argument("--side-by-side", action="store_true", help="Write grayscale input and pseudo-depth side by side.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process at most this many frames, 0 for all.")
    parser.add_argument("--fourcc", default="mp4v", help="VideoWriter fourcc, default: mp4v.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{LOG_PREFIX} input not found: {input_path}")

    output_path = Path(args.output) if args.output else Path("outputs") / "videos" / f"{input_path.stem}_pseudo_depth.mp4"
    fourcc_name = str(args.fourcc)
    if len(fourcc_name) != 4:
        raise SystemExit(f"{LOG_PREFIX} --fourcc must be exactly 4 characters")

    process_video(
        input_path=input_path,
        output_path=output_path,
        colormap_name=args.colormap,
        normalize_mode=args.normalize,
        invert_depth=bool(args.invert),
        median_k=int(args.median_k),
        clahe_clip=float(args.clahe_clip),
        clahe_grid=int(args.clahe_grid),
        disable_clahe=bool(args.no_clahe),
        low_percentile=float(args.low_percentile),
        high_percentile=float(args.high_percentile),
        sample_stride=int(args.sample_stride),
        output_size=args.size,
        side_by_side=bool(args.side_by_side),
        max_frames=int(args.max_frames),
        fourcc_name=fourcc_name,
    )


if __name__ == "__main__":
    main()
