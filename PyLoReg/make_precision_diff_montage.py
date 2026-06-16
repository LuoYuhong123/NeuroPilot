#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a visual montage for the largest precision-output differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff
from PIL import Image, ImageDraw


def _open_stack(path: Path) -> np.ndarray:
    try:
        return tiff.memmap(str(path))
    except Exception:
        return tiff.imread(str(path))


def _to_u8(frame: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1.0
    out = (frame.astype(np.float32) - lo) / (hi - lo)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--chunk-frames", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    a = _open_stack(args.baseline)
    b = _open_stack(args.candidate)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    means = np.zeros((a.shape[0],), dtype=np.float64)
    for start in range(0, a.shape[0], int(args.chunk_frames)):
        aa = np.asarray(a[start:start + int(args.chunk_frames)], dtype=np.int32)
        bb = np.asarray(b[start:start + int(args.chunk_frames)], dtype=np.int32)
        means[start:start + aa.shape[0]] = np.abs(bb - aa).mean(axis=(1, 2))

    selected = np.argsort(means)[-int(args.frames):][::-1]
    H, W = int(a.shape[1]), int(a.shape[2])
    label_h = 24
    pad = 8
    cell_w = W
    cell_h = H + label_h
    canvas = Image.new("RGB", (3 * cell_w + 4 * pad, len(selected) * cell_h + (len(selected) + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)

    for row, idx in enumerate(selected):
        af = np.asarray(a[int(idx)], dtype=np.float32)
        bf = np.asarray(b[int(idx)], dtype=np.float32)
        diff = np.abs(bf - af)

        lo, hi = np.percentile(af, [1, 99])
        diff_hi = max(1.0, float(np.percentile(diff, 99)))
        panels = [
            ("full precision", _to_u8(af, lo, hi)),
            ("uint8 flow", _to_u8(bf, lo, hi)),
            (f"abs diff x, mean={means[int(idx)]:.1f}", _to_u8(diff, 0, diff_hi)),
        ]

        y0 = pad + row * (cell_h + pad)
        for col, (title, arr) in enumerate(panels):
            x0 = pad + col * (cell_w + pad)
            draw.text((x0, y0), f"frame {int(idx)} | {title}", fill=(0, 0, 0))
            img = Image.fromarray(arr, mode="L").convert("RGB")
            canvas.paste(img, (x0, y0 + label_h))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out)
    print("selected_frames:", ",".join(str(int(i)) for i in selected))
    print("selected_mean_abs_diff:", ",".join(f"{means[int(i)]:.3f}" for i in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
