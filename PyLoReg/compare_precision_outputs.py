#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two PyLoReg output TIFF stacks for precision experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile as tiff


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _open_stack(path: Path) -> np.ndarray:
    try:
        return tiff.memmap(str(path))
    except Exception:
        return tiff.imread(str(path))


def _percentiles_from_hist(hist: np.ndarray, total: int, qs: list[float]) -> dict[str, int]:
    cs = np.cumsum(hist)
    out: dict[str, int] = {}
    for q in qs:
        target = int(np.ceil(total * q / 100.0))
        out[str(q)] = int(np.searchsorted(cs, target, side="left"))
    return out


def compare_stacks(a_path: Path, b_path: Path, chunk_frames: int, sample_count: int) -> dict:
    a = _open_stack(a_path)
    b = _open_stack(b_path)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 3:
        raise ValueError(f"Expected T,H,W stack, got {a.shape}")

    hist = np.zeros(65536, dtype=np.int64)
    total = 0
    sum_abs = 0
    sum_sq = 0
    nonzero = 0
    max_abs = 0

    T = int(a.shape[0])
    for start in range(0, T, int(chunk_frames)):
        aa = np.asarray(a[start:start + int(chunk_frames)], dtype=np.int32)
        bb = np.asarray(b[start:start + int(chunk_frames)], dtype=np.int32)
        d = np.abs(bb - aa)
        hist += np.bincount(d.ravel(), minlength=65536)
        total += int(d.size)
        sum_abs += int(d.sum(dtype=np.int64))
        sum_sq += int((d.astype(np.int64) ** 2).sum(dtype=np.int64))
        nonzero += int(np.count_nonzero(d))
        max_abs = max(max_abs, int(d.max()))

    sample_idx = np.linspace(0, T - 1, max(1, int(sample_count)), dtype=int)
    corrs = []
    grad_ratios = []
    mean_abs_samples = []
    p99_samples = []
    for i in sample_idx:
        aa = np.asarray(a[int(i)], dtype=np.float32)
        bb = np.asarray(b[int(i)], dtype=np.float32)
        da = aa - float(aa.mean())
        db = bb - float(bb.mean())
        denom = float(np.sqrt((da * da).mean() * (db * db).mean()) + 1e-12)
        corrs.append(float((da * db).mean() / denom))

        abs_diff = np.abs(bb - aa)
        mean_abs_samples.append(float(abs_diff.mean()))
        p99_samples.append(float(np.percentile(abs_diff, 99)))

        gy_a, gx_a = np.gradient(aa)
        gy_b, gx_b = np.gradient(bb)
        mag_a = np.hypot(gx_a, gy_a)
        mag_b = np.hypot(gx_b, gy_b)
        grad_ratios.append(float(np.mean(np.abs(mag_b - mag_a)) / (np.mean(mag_a) + 1e-6)))

    return {
        "baseline": str(a_path.resolve()),
        "candidate": str(b_path.resolve()),
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "exact_equal": bool(nonzero == 0),
        "total_pixels": int(total),
        "nonzero_pixels": int(nonzero),
        "nonzero_fraction": float(nonzero / total),
        "max_abs_diff": int(max_abs),
        "mean_abs_diff": float(sum_abs / total),
        "rmse": float((sum_sq / total) ** 0.5),
        "abs_diff_percentiles": _percentiles_from_hist(hist, total, [50, 90, 95, 99, 99.9, 99.99]),
        "sampled_frames": sample_idx.tolist(),
        "sample_ncc_min": float(np.min(corrs)),
        "sample_ncc_mean": float(np.mean(corrs)),
        "sample_ncc_median": float(np.median(corrs)),
        "sample_grad_mag_diff_ratio_mean": float(np.mean(grad_ratios)),
        "sample_grad_mag_diff_ratio_max": float(np.max(grad_ratios)),
        "sample_mean_abs_diff_mean": float(np.mean(mean_abs_samples)),
        "sample_p99_abs_diff_max": float(np.max(p99_samples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-frames", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=31)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(_repo_root()))
    summary = compare_stacks(
        args.baseline,
        args.candidate,
        chunk_frames=int(args.chunk_frames),
        sample_count=int(args.sample_count),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
