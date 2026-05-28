#!/usr/bin/env python3
"""Compute per-pixel absolute difference between two images.

This script is intentionally simple: it reads two images, checks shape compatibility,
computes `abs(a - b)` per pixel, then writes diff images and prints summary stats.

Examples:
  python scripts/image_diff.py img_a.png img_b.png
  python scripts/image_diff.py img_a.png img_b.png --outdir outputs/image_diff --prefix sample

Outputs:
  <outdir>/<prefix>_diff.png       : per-channel abs diff (same shape as input)
  <outdir>/<prefix>_diff_gray.png  : single-channel visualization for quick view
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    # Normalize (H,W,1) -> (H,W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]

    return img


def _to_vis_gray(diff: np.ndarray) -> np.ndarray:
    """Convert diff image to a single-channel uint8 visualization."""
    if diff.ndim == 2:
        gray = diff
    else:
        # For multi-channel, use max across channels so small changes are visible.
        gray = diff.max(axis=2)

    if gray.dtype == np.uint8:
        return gray

    # Best-effort conversion for 16-bit or other types.
    gray_f = gray.astype(np.float32)
    maxv = float(np.max(gray_f)) if gray_f.size else 0.0
    if maxv <= 0.0:
        return np.zeros_like(gray_f, dtype=np.uint8)
    scaled = np.clip(gray_f * (255.0 / maxv), 0.0, 255.0).astype(np.uint8)
    return scaled


def main() -> int:
    parser = argparse.ArgumentParser(description="Image absolute difference (absdiff)")
    parser.add_argument("image_a", help="path to first image")
    parser.add_argument("image_b", help="path to second image")
    parser.add_argument("--outdir", default=str(Path("outputs") / "image_diff"))
    parser.add_argument(
        "--prefix",
        default=None,
        help="output filename prefix (default: <a_stem>__<b_stem>)",
    )
    args = parser.parse_args()

    a_path = Path(args.image_a)
    b_path = Path(args.image_b)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = _load(a_path)
    b = _load(b_path)

    if a.shape != b.shape:
        raise SystemExit(
            "Image shapes do not match. "
            f"a={a.shape} ({a_path}) b={b.shape} ({b_path}). "
            "Please pre-resize/crop them to the same size first."
        )

    # absdiff supports uint8/uint16/float etc.
    diff = cv2.absdiff(a, b)
    diff_gray = _to_vis_gray(diff)

    prefix = args.prefix or f"{a_path.stem}__{b_path.stem}"
    diff_path = outdir / f"{prefix}_diff.png"
    diff_gray_path = outdir / f"{prefix}_diff_gray.png"

    # Save
    if not cv2.imwrite(str(diff_path), diff):
        raise SystemExit(f"failed to write {diff_path}")
    if not cv2.imwrite(str(diff_gray_path), diff_gray):
        raise SystemExit(f"failed to write {diff_gray_path}")

    # Stats
    diff_f = diff.astype(np.float32)
    mean = float(np.mean(diff_f))
    maxv = float(np.max(diff_f)) if diff_f.size else 0.0
    nonzero = int(np.count_nonzero(diff))
    total = int(diff.size)

    print("image_a:", str(a_path))
    print("image_b:", str(b_path))
    print("shape:", a.shape, "dtype:", a.dtype)
    print("diff_mean:", f"{mean:.4f}")
    print("diff_max:", f"{maxv:.4f}")
    print("nonzero_pixels:", f"{nonzero}/{total}")
    print("wrote:", str(diff_path))
    print("wrote:", str(diff_gray_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
