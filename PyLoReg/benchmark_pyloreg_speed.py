#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark PyLoReg inference speed and final TIFF equivalence.

The benchmark compares:
  1) a baseline source file, by default ../test_PyLoReg.py;
  2) the project module PyLoReg.pylog_inference.

It intentionally disables auxiliary mask/flow/template QC writes by default so
the timing focuses on registration and final TIFF generation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile as tiff


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("pyloreg_baseline_for_benchmark", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import baseline module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_case(label: str, module, kwargs: dict, out_path: Path) -> dict:
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()
    module.demotion_PyLoReg_infer2stack(
        **kwargs,
        stack_save_path=str(out_path),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    arr = tiff.imread(str(out_path))
    return {
        "label": label,
        "path": str(out_path),
        "seconds": elapsed,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


def _compare_tiffs(path_a: Path, path_b: Path) -> dict:
    a = tiff.imread(str(path_a))
    b = tiff.imread(str(path_b))
    if a.shape != b.shape:
        return {
            "same_shape": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }

    diff = b.astype(np.int32) - a.astype(np.int32)
    abs_diff = np.abs(diff)
    return {
        "same_shape": True,
        "same_dtype": str(a.dtype) == str(b.dtype),
        "exact_equal": bool(np.array_equal(a, b)),
        "max_abs_diff": int(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "nonzero_pixels": int(np.count_nonzero(diff)),
        "total_pixels": int(diff.size),
    }


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    default_img = root / "demo_data" / "spine" / "ju2df_5day_freemoving-male1-5day-image-pain 0.tif"
    default_baseline = root.parent / "test_PyLoReg.py"
    default_out = root / "runs" / "pyloreg_benchmark" / datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img", type=Path, default=default_img, help="Motion-corrected/input image stack.")
    parser.add_argument("--raw", type=Path, default=None, help="Raw stack. Defaults to --img for demo benchmarking.")
    parser.add_argument("--baseline-source", type=Path, default=default_baseline)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument("--frames", type=int, default=12, help="First N frames to benchmark.")
    parser.add_argument("--crop-size", type=int, default=192, help="Center crop size. Use 0 to keep full HxW.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--net-iters", type=int, default=1, help="Network rounds after Iter1 rigid.")
    parser.add_argument("--use-feature-num", type=int, default=4)
    parser.add_argument("--robust-template-pool-size", type=int, default=2000)
    parser.add_argument("--robust-template-keep-ratio", type=float, default=0.20)
    parser.add_argument("--save-mask-flow", action="store_true", help="Also write mask/flow auxiliaries.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    sys.path.insert(0, str(root))

    raw_path = args.raw if args.raw is not None else args.img
    if not args.img.is_file():
        raise FileNotFoundError(args.img)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not args.baseline_source.is_file():
        raise FileNotFoundError(args.baseline_source)

    baseline = _load_module_from_path(args.baseline_source)
    optimized = importlib.import_module("PyLoReg.pylog_inference")

    common_kwargs = {
        "img_stack_path": str(args.img),
        "raw_stack_path": str(raw_path),
        "model_root": str(root / "PyLoReg" / "PyLoReg_model"),
        "feature_channels": 128,
        "use_feature_num": args.use_feature_num,
        "max_frames": args.frames,
        "crop_size": None if args.crop_size <= 0 else args.crop_size,
        "batch_size": args.batch_size,
        "save_mask_flow": bool(args.save_mask_flow),
        "save_templates": False,
        "save_robust_template_qc": False,
        "net_iter_num": args.net_iters,
        "robust_template_pool_size": args.robust_template_pool_size,
        "robust_template_keep_ratio": args.robust_template_keep_ratio,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_out = args.out_dir / "baseline_output.tif"
    optimized_out = args.out_dir / "optimized_output.tif"

    print("[Benchmark] baseline:", args.baseline_source)
    baseline_result = _run_case("baseline", baseline, common_kwargs, baseline_out)
    print("[Benchmark] optimized:", root / "PyLoReg" / "pylog_inference.py")
    optimized_result = _run_case("optimized", optimized, common_kwargs, optimized_out)
    comparison = _compare_tiffs(baseline_out, optimized_out)

    summary = {
        "inputs": {
            "img": str(args.img),
            "raw": str(raw_path),
            "frames": args.frames,
            "crop_size": None if args.crop_size <= 0 else args.crop_size,
            "batch_size": args.batch_size,
            "net_iters": args.net_iters,
            "save_mask_flow": bool(args.save_mask_flow),
        },
        "baseline": baseline_result,
        "optimized": optimized_result,
        "speedup": baseline_result["seconds"] / optimized_result["seconds"]
        if optimized_result["seconds"] > 0 else None,
        "comparison": comparison,
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("[Benchmark] summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
