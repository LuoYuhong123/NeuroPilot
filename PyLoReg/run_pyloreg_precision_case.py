#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one PyLoReg precision case and write a compact runtime summary."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tifffile as tiff


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--crop-size", type=int, default=0, help="0 keeps full HxW.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--net-iters", type=int, default=2)
    parser.add_argument("--flow-storage-dtype", choices=("uint16", "uint8"), default="uint16")
    parser.add_argument("--flow-scale", type=float, default=None)
    parser.add_argument("--flow-offset", type=float, default=None)
    parser.add_argument("--save-mask-flow", action="store_true")
    parser.add_argument("--model-root", type=Path, default=root / "PyLoReg" / "PyLoReg_model")
    return parser.parse_args()


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root))

    from PyLoReg.pylog_inference import demotion_PyLoReg_infer2stack

    args = parse_args()
    raw_path = args.raw if args.raw is not None else args.img
    if not args.img.is_file():
        raise FileNotFoundError(args.img)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "img_stack_path": str(args.img),
        "raw_stack_path": str(raw_path),
        "stack_save_path": str(args.out),
        "model_root": str(args.model_root),
        "feature_channels": 128,
        "use_feature_num": 4,
        "max_frames": int(args.frames),
        "crop_size": None if int(args.crop_size) <= 0 else int(args.crop_size),
        "batch_size": int(args.batch_size),
        "save_mask_flow": bool(args.save_mask_flow),
        "save_templates": False,
        "save_robust_template_qc": False,
        "net_iter_num": int(args.net_iters),
        "flow_storage_dtype": str(args.flow_storage_dtype),
    }
    if args.flow_scale is not None:
        kwargs["flow_scale"] = float(args.flow_scale)
    if args.flow_offset is not None:
        kwargs["flow_offset"] = float(args.flow_offset)

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        torch = None

    start = time.perf_counter()
    demotion_PyLoReg_infer2stack(**kwargs)
    if "torch" in locals() and torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    arr = tiff.imread(str(args.out))
    summary = {
        "path": str(args.out.resolve()),
        "seconds": float(elapsed),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "params": {
            "img": str(args.img.resolve()),
            "raw": str(raw_path.resolve()),
            "frames": int(args.frames),
            "crop_size": None if int(args.crop_size) <= 0 else int(args.crop_size),
            "batch_size": int(args.batch_size),
            "net_iters": int(args.net_iters),
            "flow_storage_dtype": str(args.flow_storage_dtype),
            "flow_scale": args.flow_scale,
            "flow_offset": args.flow_offset,
            "save_mask_flow": bool(args.save_mask_flow),
        },
    }
    summary_path = args.out.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("[Summary]", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
