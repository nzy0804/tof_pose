from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


LOG_PREFIX = "[video_binarize]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binarize a video and write the binary video output.")
    parser.add_argument("input", type=Path, help="Input video path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: outputs/videos/<stem>_binary.mp4)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Binary threshold in [0,255]. If 0, uses Otsu.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert binary output (swap 0 and 255).",
    )
    parser.add_argument(
        "--median-k",
        type=int,
        default=5,
        help="Median blur kernel size for salt-and-pepper denoise before thresholding (0 to disable). Default: 5.",
    )
    parser.add_argument(
        "--morph-k",
        type=int,
        default=0,
        help="Morphology kernel size applied after binarization (open then close). 0 to disable. Default: 0.",
    )
    parser.add_argument(
        "--morph-iter",
        type=int,
        default=1,
        help="Morphology iterations (only if --morph-k > 0). Default: 1.",
    )
    return parser.parse_args()


def _validate_odd_kernel(k: int, *, name: str) -> int:
    if k <= 0:
        return 0
    if k < 3:
        raise ValueError(f"{name} must be 0 or >= 3")
    if k % 2 == 0:
        k += 1
    return int(k)


def binarize_video(
    *,
    input_path: Path,
    output_path: Path,
    threshold: int,
    invert: bool,
    median_k: int,
    morph_k: int,
    morph_iter: int,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,  # write BGR frames
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open VideoWriter: {output_path}")

    print(f"{LOG_PREFIX} input={input_path}")
    print(f"{LOG_PREFIX} output={output_path}")
    print(f"{LOG_PREFIX} fps={fps} size={width}x{height} frames={total_frames}")
    median_k = _validate_odd_kernel(int(median_k), name="--median-k")
    morph_k = _validate_odd_kernel(int(morph_k), name="--morph-k")
    morph_iter = max(1, int(morph_iter))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if median_k > 0:
            gray = cv2.medianBlur(gray, median_k)

        if threshold <= 0:
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, 0, 255, thresh_type | cv2.THRESH_OTSU)
        else:
            t = int(np.clip(threshold, 0, 255))
            thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, t, 255, thresh_type)

        if morph_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        out.write(binary_bgr)

        frame_idx += 1
        if frame_idx % 30 == 0 and total_frames > 0:
            percent = (frame_idx / total_frames) * 100
            print(f"{LOG_PREFIX} {frame_idx}/{total_frames} ({percent:.1f}%)", end="\r")

    cap.release()
    out.release()
    print(f"\n{LOG_PREFIX} done: {output_path}")


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"{LOG_PREFIX} input not found: {input_path}")

    output_path = Path(args.output) if args.output else Path("outputs") / "videos" / f"{input_path.stem}_binary.mp4"

    binarize_video(
        input_path=input_path,
        output_path=output_path,
        threshold=int(args.threshold),
        invert=bool(args.invert),
        median_k=int(args.median_k),
        morph_k=int(args.morph_k),
        morph_iter=int(args.morph_iter),
    )


if __name__ == "__main__":
    main()
