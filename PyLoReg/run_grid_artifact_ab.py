#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B test PyLoReg grid artifacts on a selected frame window."""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile as tiff


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_module(module_file: Path, package_root: Path | None):
    if package_root is not None:
        for name in list(sys.modules):
            if name == "PyLoReg" or name.startswith("PyLoReg."):
                del sys.modules[name]
        sys.path.insert(0, str(package_root))
    module_name = f"PyLoReg._ab_module_{abs(hash(str(module_file)))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_window(raw_path: Path, out_path: Path, start: int, end_inclusive: int) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        arr = tiff.imread(str(out_path))
    else:
        raw = tiff.memmap(str(raw_path))
        arr = np.asarray(raw[int(start):int(end_inclusive) + 1])
        tiff.imwrite(str(out_path), arr)
    return {
        "path": str(out_path.resolve()),
        "source": str(raw_path.resolve()),
        "start_frame": int(start),
        "end_frame_inclusive": int(end_inclusive),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _run_case(case: dict, input_tif: Path, out_dir: Path) -> dict:
    label = case["label"]
    case_dir = out_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    out_tif = case_dir / f"{label}.tif"

    package_root = Path(case["package_root"]).resolve() if case.get("package_root") else None
    module_file = Path(case["module_file"]).resolve()
    module = _load_module(module_file, package_root)
    fn = getattr(module, case["function"])

    kwargs = {
        "img_stack_path": str(input_tif),
        "raw_stack_path": str(input_tif),
        "stack_save_path": str(out_tif),
        "model_root": str(Path(case["model_root"]).resolve()),
        "feature_channels": 128,
        "use_feature_num": 4,
        "max_frames": None,
        "crop_size": None,
        "batch_size": int(case.get("batch_size", 2)),
        "save_mask_flow": False,
        "save_templates": False,
        "warp_mode": str(case.get("warp_mode", "nearest")),
    }
    kwargs.update(case.get("extra_kwargs", {}))
    signature = inspect.signature(fn)
    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    fn(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    arr = tiff.imread(str(out_tif))
    return {
        "label": label,
        "path": str(out_tif.resolve()),
        "seconds": float(elapsed),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "module_file": str(module_file),
        "function": case["function"],
        "warp_mode": str(case.get("warp_mode", "nearest")),
        "extra_kwargs": case.get("extra_kwargs", {}),
    }


def _motion_summary(stack: np.ndarray) -> dict:
    from input_metrics import estimate_raw_motion_rigid

    rigid, _ = estimate_raw_motion_rigid(stack)
    s = rigid.get("rigid_motion_summary", {})
    j = rigid.get("frame_to_frame_jitter", {})
    c = rigid.get("registration_confidence", {})
    return {
        "motion_mean_px": s.get("motion_mean_px"),
        "motion_p95_px": s.get("motion_p95_px"),
        "motion_max_px": s.get("motion_max_px"),
        "jitter_mean_px": j.get("jitter_mean_px"),
        "jitter_p95_px": j.get("jitter_p95_px"),
        "jitter_max_px": j.get("jitter_max_px"),
        "corr_mean": c.get("corr_mean"),
        "corr_min": c.get("corr_min"),
    }


def _periodic_boundary_scores(stack: np.ndarray, periods: tuple[int, ...] = (8, 16, 32)) -> dict:
    x = stack.astype(np.float32, copy=False)
    dx = np.abs(np.diff(x, axis=2))
    dy = np.abs(np.diff(x, axis=1))
    base_x = float(dx.mean() + 1e-6)
    base_y = float(dy.mean() + 1e-6)
    out = {}
    for p in periods:
        cols = [c - 1 for c in range(p, x.shape[2], p)]
        rows = [r - 1 for r in range(p, x.shape[1], p)]
        out[f"period_{p}_x_ratio"] = float(dx[:, :, cols].mean() / base_x) if cols else None
        out[f"period_{p}_y_ratio"] = float(dy[:, rows, :].mean() / base_y) if rows else None
    return out


def _case_quality(case_result: dict) -> dict:
    stack = tiff.imread(case_result["path"])
    return {
        "motion": _motion_summary(stack),
        "periodic_boundary": _periodic_boundary_scores(stack),
    }


def _compare_to_raw(raw_stack: np.ndarray, case_result: dict) -> dict:
    stack = tiff.imread(case_result["path"])
    diff = np.abs(stack.astype(np.int32) - raw_stack.astype(np.int32))
    return {
        "mean_abs_diff_from_window_raw": float(diff.mean()),
        "p95_abs_diff_from_window_raw": float(np.percentile(diff, 95)),
        "p99_abs_diff_from_window_raw": float(np.percentile(diff, 99)),
        "max_abs_diff_from_window_raw": int(diff.max()),
    }


def _make_montage(raw_stack: np.ndarray, case_results: list[dict], out_path: Path, frame_index: int, source_frame: int):
    from PIL import Image, ImageDraw

    panels = [("raw", raw_stack[frame_index].astype(np.float32))]
    for result in case_results:
        arr = tiff.memmap(result["path"])[frame_index].astype(np.float32)
        panels.append((result["label"], arr))

    lo, hi = np.percentile(raw_stack[frame_index], [1, 99])
    hi = max(float(hi), float(lo) + 1.0)

    def to_u8(x):
        return np.clip((x - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    H, W = raw_stack.shape[1:]
    pad = 8
    label_h = 26
    cols = len(panels)
    canvas = Image.new("RGB", (cols * W + (cols + 1) * pad, H + label_h + 2 * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for col, (title, arr) in enumerate(panels):
        x0 = pad + col * (W + pad)
        y0 = pad
        draw.text((x0, y0), f"frame {source_frame} | {title}", fill=(0, 0, 0))
        canvas.paste(Image.fromarray(to_u8(arr), mode="L").convert("RGB"), (x0, y0 + label_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=root / "runs" / "pyloreg_grid_ab")
    parser.add_argument("--start-frame", type=int, default=2300)
    parser.add_argument("--end-frame", type=int, default=2380)
    parser.add_argument("--focus-frame", type=int, default=2349)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    sys.path.insert(0, str(root))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_tif = args.out_dir / f"input_frames_{args.start_frame}_{args.end_frame}.tif"
    input_info = _prepare_window(args.raw, input_tif, args.start_frame, args.end_frame)

    old_root = Path(r"D:\0_NurMar_shared_files\2_code\NeuroPilot_LLM")
    current_model_root = root / "PyLoReg" / "PyLoReg_model"
    old_model_root = old_root / "PyLoReg" / "PyLoReg_model"
    cases = [
        {
            "label": "current_nearest",
            "module_file": str(root / "PyLoReg" / "pylog_inference.py"),
            "package_root": str(root),
            "function": "demotion_PyLoReg_infer2stack",
            "model_root": str(current_model_root),
            "warp_mode": "nearest",
            "extra_kwargs": {"net_iter_num": 2, "flow_storage_dtype": "uint16"},
        },
        {
            "label": "current_bilinear",
            "module_file": str(root / "PyLoReg" / "pylog_inference.py"),
            "package_root": str(root),
            "function": "demotion_PyLoReg_infer2stack",
            "model_root": str(current_model_root),
            "warp_mode": "bilinear",
            "extra_kwargs": {"net_iter_num": 2, "flow_storage_dtype": "uint16"},
        },
        {
            "label": "old_v3_nearest",
            "module_file": str(old_root / "PyLoReg" / "pylog_inference.py"),
            "package_root": str(old_root),
            "function": "demotion_PyLoReg_infer2stack_less_save_acc_v3",
            "model_root": str(old_model_root),
            "warp_mode": "nearest",
            "extra_kwargs": {"net_iter_num": 2},
        },
        {
            "label": "old_v2_template_nearest",
            "module_file": str(old_root / "PyLoReg" / "test_PyLoReg_v2.py"),
            "package_root": str(old_root),
            "function": "demotion_PyLoReg_infer2stack_less_save_acc",
            "model_root": str(old_model_root),
            "warp_mode": "nearest",
            "extra_kwargs": {"iteration_num": 2, "template_mode": "mean"},
        },
    ]

    results = []
    for case in cases:
        print(f"\n[AB] Running {case['label']}")
        result = _run_case(case, input_tif, args.out_dir)
        results.append(result)
        print(json.dumps(result, indent=2))

    raw_stack = tiff.imread(str(input_tif))
    qualities = {}
    for result in results:
        qualities[result["label"]] = _case_quality(result)
        qualities[result["label"]].update(_compare_to_raw(raw_stack, result))

    focus_idx = int(args.focus_frame) - int(args.start_frame)
    focus_idx = max(0, min(focus_idx, raw_stack.shape[0] - 1))
    montage_path = args.out_dir / f"montage_frame_{args.focus_frame}.png"
    _make_montage(raw_stack, results, montage_path, focus_idx, int(args.focus_frame))

    summary = {
        "input": input_info,
        "cases": results,
        "quality": qualities,
        "montage": str(montage_path.resolve()),
    }
    summary_path = args.out_dir / "ab_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("[AB] Summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
