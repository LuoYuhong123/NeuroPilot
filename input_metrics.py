#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute input metrics for a raw calcium-imaging movie stack with shape (T, H, W).

Outputs:
1) JSON summary containing:
   - basic metadata / shape
   - intensity statistics
   - bleaching trend summary
   - SNR estimate
   - rigid raw-motion estimate
2) STD projection over time
3) MIP (maximum intensity projection) over time

Example:
    python input_metrics.py ^
        --input path/to/movie.tif ^
        --output-dir path/to/output ^
        --fps 10 ^
        --pixel-size-um 0.65
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile as tiff


def load_tif_anyshape(tif_path: str | Path) -> np.ndarray:
    """
    Load a TIFF-like stack and return a 3D movie with shape (T, H, W).

    Supported cases:
    - (T, H, W)
    - (H, W, T)  via shape heuristic
    - (A, B, H, W) -> (A*B, H, W)
    - (H, W, A, B) -> (A*B, H, W)
    """
    arr = tiff.imread(str(tif_path))

    if arr.ndim == 3:
        if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))
        return np.asarray(arr)

    if arr.ndim == 4:
        if arr.shape[2] >= 16 and arr.shape[3] >= 16:
            a_dim, b_dim, height, width = arr.shape
            return np.asarray(arr).reshape(a_dim * b_dim, height, width)
        if arr.shape[0] >= 16 and arr.shape[1] >= 16:
            height, width, a_dim, b_dim = arr.shape
            arr = np.transpose(arr, (2, 3, 0, 1))
            return np.asarray(arr).reshape(a_dim * b_dim, height, width)
        a_dim, b_dim, height, width = arr.shape
        return np.asarray(arr).reshape(a_dim * b_dim, height, width)

    raise ValueError(f"Unsupported TIFF dims={arr.ndim}, shape={arr.shape}")


def moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or x.size <= 1:
        return x.astype(np.float64, copy=True)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(x.astype(np.float64), (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def percentile_float(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def build_snr_reference(stack: np.ndarray, roi_percentile: float = 90.0) -> dict[str, Any]:
    stack_f = stack.astype(np.float32, copy=False)
    std_map = np.std(stack_f, axis=0)
    roi_thr = float(np.percentile(std_map, roi_percentile))
    roi = std_map > roi_thr
    if not np.any(roi):
        roi = std_map >= roi_thr
    return {
        "roi_mask": roi.astype(bool, copy=False),
        "roi_threshold": roi_thr,
        "roi_percentile": float(roi_percentile),
        "roi_pixel_count": int(np.count_nonzero(roi)),
        "roi_pixel_fraction": float(np.mean(roi)),
    }


def _robust_mad_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size < 2:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


def save_projection_tifs(stack: np.ndarray, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    mip = np.max(stack, axis=0)
    std_map = np.std(stack.astype(np.float32), axis=0)

    mip_path = output_dir / f"{stem}_MIP.tif"
    std_path = output_dir / f"{stem}_STD.tif"

    tiff.imwrite(str(mip_path), mip)
    tiff.imwrite(str(std_path), std_map.astype(np.float32))

    return {
        "mip_tif": str(mip_path),
        "std_tif": str(std_path),
    }


def compute_basic_stats(
    stack: np.ndarray,
    fps: float | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, Any]:
    flat = stack.reshape(-1).astype(np.float64)
    dtype = stack.dtype

    observed_min = float(np.min(flat))
    observed_max = float(np.max(flat))
    observed_range = observed_max - observed_min
    p1 = percentile_float(flat, 1)
    p50 = percentile_float(flat, 50)
    p99 = percentile_float(flat, 99)
    robust_range = p99 - p1

    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        dtype_min = float(dtype_info.min)
        dtype_max = float(dtype_info.max)
        full_scale_range = dtype_max - dtype_min
        range_utilization = robust_range / full_scale_range if full_scale_range > 0 else 0.0
        saturation_ratio = float(np.mean(flat >= dtype_max))
        dark_ratio = float(np.mean(flat <= dtype_min))
    else:
        dtype_min = None
        dtype_max = None
        full_scale_range = None
        range_utilization = robust_range / observed_range if observed_range > 0 else 0.0
        saturation_ratio = float(np.mean(flat >= observed_max))
        dark_ratio = float(np.mean(flat <= observed_min))

    num_frames, height, width = stack.shape

    return {
        "shape_thw": [int(num_frames), int(height), int(width)],
        "num_frames": int(num_frames),
        "height_px": int(height),
        "width_px": int(width),
        "dtype": str(dtype),
        "frame_rate_hz": float(fps) if fps is not None else None,
        "pixel_size_um": float(pixel_size_um) if pixel_size_um is not None else None,
        "intensity_statistics": {
            "min": observed_min,
            "max": observed_max,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p01": p1,
            "p50": p50,
            "p99": p99,
        },
        "dynamic_range": {
            "observed_range": observed_range,
            "robust_range_p99_minus_p01": robust_range,
            "dtype_min": dtype_min,
            "dtype_max": dtype_max,
            "full_scale_range": full_scale_range,
            "utilization_ratio": float(range_utilization),
        },
        "extreme_pixel_ratio": {
            "saturation_ratio": saturation_ratio,
            "dark_ratio": dark_ratio,
        },
    }


def compute_bleaching_summary(stack: np.ndarray, fps: float | None = None) -> dict[str, Any]:
    """
    Estimate bleaching / brightness drift using whole-frame mean intensity over time.

    We summarize:
    - start vs end brightness
    - linear slope over the smoothed trace
    - relative drop percentage
    - a simple boolean flag for obvious downward drift
    """
    stack_f64 = stack.astype(np.float64)
    frame_mean = stack_f64.mean(axis=(1, 2))
    frame_median = np.median(stack_f64, axis=(1, 2))

    smooth_window = max(5, int(round(stack.shape[0] * 0.05)))
    smooth_window = min(smooth_window, max(1, stack.shape[0]))
    frame_mean_smooth = moving_average_1d(frame_mean, smooth_window)

    if stack.shape[0] >= 2:
        x = np.arange(stack.shape[0], dtype=np.float64)
        slope_per_frame, intercept = np.polyfit(x, frame_mean_smooth, deg=1)
    else:
        slope_per_frame = 0.0
        intercept = float(frame_mean_smooth[0])

    segment = max(3, int(round(stack.shape[0] * 0.1)))
    segment = min(segment, stack.shape[0])
    start_mean = float(np.median(frame_mean[:segment]))
    end_mean = float(np.median(frame_mean[-segment:]))
    relative_change = (end_mean - start_mean) / max(abs(start_mean), 1e-8)
    relative_drop_pct = -100.0 * relative_change

    if fps is not None and fps > 0:
        slope_per_second = float(slope_per_frame * fps)
    else:
        slope_per_second = None

    obvious_bleaching = bool((slope_per_frame < 0) and (relative_drop_pct >= 10.0))

    return {
        "method": "whole_frame_mean_intensity_vs_time",
        "smoothing_window_frames": int(smooth_window),
        "start_segment_frames": int(segment),
        "start_mean_intensity": start_mean,
        "end_mean_intensity": end_mean,
        "relative_change_ratio": float(relative_change),
        "relative_drop_percent": float(relative_drop_pct),
        "linear_fit_slope_per_frame": float(slope_per_frame),
        "linear_fit_intercept": float(intercept),
        "linear_fit_slope_per_second": slope_per_second,
        "mean_trace_summary": {
            "mean": float(np.mean(frame_mean)),
            "std": float(np.std(frame_mean)),
            "min": float(np.min(frame_mean)),
            "max": float(np.max(frame_mean)),
        },
        "median_trace_summary": {
            "mean": float(np.mean(frame_median)),
            "std": float(np.std(frame_median)),
            "min": float(np.min(frame_median)),
            "max": float(np.max(frame_median)),
        },
        "obvious_bleaching_flag": obvious_bleaching,
    }


def compute_snr_metric(
    stack: np.ndarray,
    roi_percentile: float = 90.0,
    roi_mask: np.ndarray | None = None,
    roi_source: str = "self",
    roi_reference_label: str | None = None,
) -> dict[str, Any]:
    """
    Robust SNR proxy for before/after comparisons.

    Signal:
    - p95(trace) - p20(trace) over a high-activity ROI

    Noise:
    - 1.4826 * MAD(diff(trace)) / sqrt(2), a robust frame-to-frame noise estimate

    Comparison-friendly behavior:
    - when a reference ROI mask is provided, raw and processed stacks are evaluated
      on the same spatial support instead of re-selecting the ROI independently
    """
    stack_f = stack.astype(np.float32, copy=False)
    if roi_mask is None:
        ref = build_snr_reference(stack_f, roi_percentile=roi_percentile)
        roi = ref["roi_mask"]
        roi_thr = float(ref["roi_threshold"])
        roi_source = "self"
    else:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != stack_f.shape[1:]:
            raise ValueError(
                f"SNR ROI mask shape mismatch: expected {stack_f.shape[1:]}, got {roi.shape}"
            )
        if not np.any(roi):
            raise ValueError("SNR ROI mask is empty.")
        roi_thr = None

    ts = stack_f[:, roi].mean(axis=1).astype(np.float32, copy=False)
    p20 = float(np.percentile(ts, 20))
    p95 = float(np.percentile(ts, 95))
    signal_amplitude = max(0.0, float(p95 - p20))

    diff_ts = np.diff(ts)
    diff_noise_mad = _robust_mad_std(diff_ts)
    noise = float(diff_noise_mad / np.sqrt(2.0)) if diff_ts.size else 0.0

    if noise <= 1e-12:
        baseline_samples = ts[ts <= p20]
        noise = _robust_mad_std(baseline_samples)
    if noise <= 1e-12:
        noise = float(np.std(ts)) if ts.size >= 2 else 0.0

    snr_value = None if noise <= 1e-12 else float(signal_amplitude / noise)

    return {
        "method": "fixed_roi_trace_p95_minus_p20_over_diff_mad_noise",
        "roi_source": str(roi_source),
        "roi_reference_label": roi_reference_label,
        "std_roi_percentile_threshold": float(roi_percentile),
        "roi_threshold_from_reference_std": roi_thr,
        "roi_pixel_count": int(np.count_nonzero(roi)),
        "roi_pixel_fraction": float(np.mean(roi)),
        "trace_p20": p20,
        "trace_p95": p95,
        "signal_amplitude_p95_minus_p20": signal_amplitude,
        "noise_diff_mad_std": float(diff_noise_mad),
        "noise_std": float(noise),
        "snr": snr_value,
    }


def preprocess_for_registration(stack: np.ndarray) -> np.ndarray:
    stack_f = stack.astype(np.float32, copy=False)
    processed = np.empty_like(stack_f, dtype=np.float32)

    for idx in range(stack_f.shape[0]):
        frame = stack_f[idx]
        lo = float(np.percentile(frame, 1))
        hi = float(np.percentile(frame, 99))
        frame_clip = np.clip(frame, lo, hi)
        frame_centered = frame_clip - float(np.median(frame_clip))
        scale = float(np.std(frame_centered))
        if scale > 1e-8:
            frame_norm = frame_centered / scale
        else:
            frame_norm = frame_centered
        processed[idx] = frame_norm

    return processed


def make_hanning_window(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height)
    wx = np.hanning(width)
    return np.outer(wy, wx).astype(np.float32)


def phase_correlation_shift(frame: np.ndarray, template: np.ndarray) -> tuple[float, float, float]:
    """
    Return shift-to-apply to the frame in order to align it to the template.

    Output convention:
    - dx > 0: shift frame to the right
    - dy > 0: shift frame downward
    """
    height, width = frame.shape
    window = make_hanning_window(height, width)

    a = frame * window
    b = template * window

    fft_a = np.fft.fft2(a)
    fft_b = np.fft.fft2(b)
    cross_power = fft_a * np.conjugate(fft_b)
    cross_power /= np.maximum(np.abs(cross_power), 1e-8)

    corr = np.abs(np.fft.ifft2(cross_power))
    peak_y, peak_x = np.unravel_index(np.argmax(corr), corr.shape)
    peak_value = float(corr[peak_y, peak_x])

    if peak_y > height // 2:
        peak_y -= height
    if peak_x > width // 2:
        peak_x -= width

    shift_y_to_apply = float(-peak_y)
    shift_x_to_apply = float(-peak_x)

    return shift_x_to_apply, shift_y_to_apply, peak_value


def estimate_raw_motion_rigid(stack: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    stack_proc = preprocess_for_registration(stack)

    num_template_frames = min(100, stack_proc.shape[0])
    template = np.median(stack_proc[:num_template_frames], axis=0)

    shifts = []
    corr_scores = []

    for idx in range(stack_proc.shape[0]):
        frame = stack_proc[idx]
        dx, dy, score = phase_correlation_shift(frame, template)
        shifts.append([dx, dy])
        corr_scores.append(score)

    shifts = np.asarray(shifts, dtype=np.float64)
    corr_scores = np.asarray(corr_scores, dtype=np.float64)

    dx = shifts[:, 0]
    dy = shifts[:, 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)

    ddx = np.diff(dx, prepend=dx[0])
    ddy = np.diff(dy, prepend=dy[0])
    dmag = np.sqrt(ddx ** 2 + ddy ** 2)

    motion_json = {
        "method": "phase_correlation_to_median_template_after_framewise_normalization",
        "template_frame_count": int(num_template_frames),
        "rigid_motion_summary": {
            "motion_mean_px": float(np.mean(mag)),
            "motion_median_px": float(np.median(mag)),
            "motion_p95_px": float(np.percentile(mag, 95)),
            "motion_max_px": float(np.max(mag)),
        },
        "x_shift_summary": {
            "dx_mean_px": float(np.mean(dx)),
            "dx_std_px": float(np.std(dx)),
            "dx_min_px": float(np.min(dx)),
            "dx_max_px": float(np.max(dx)),
        },
        "y_shift_summary": {
            "dy_mean_px": float(np.mean(dy)),
            "dy_std_px": float(np.std(dy)),
            "dy_min_px": float(np.min(dy)),
            "dy_max_px": float(np.max(dy)),
        },
        "frame_to_frame_jitter": {
            "jitter_mean_px": float(np.mean(dmag)),
            "jitter_p95_px": float(np.percentile(dmag, 95)),
            "jitter_max_px": float(np.max(dmag)),
        },
        "registration_confidence": {
            "corr_mean": float(np.mean(corr_scores)),
            "corr_median": float(np.median(corr_scores)),
            "corr_min": float(np.min(corr_scores)),
        },
    }

    return motion_json, shifts


def build_metrics_json(
    stack: np.ndarray,
    input_path: Path,
    output_dir: Path,
    fps: float | None,
    pixel_size_um: float | None,
) -> dict[str, Any]:
    basic = compute_basic_stats(stack, fps=fps, pixel_size_um=pixel_size_um)
    bleaching = compute_bleaching_summary(stack, fps=fps)
    snr = compute_snr_metric(stack)
    motion_summary, shifts = estimate_raw_motion_rigid(stack)

    shift_artifact = output_dir / f"{input_path.stem}_rigid_shifts.npy"
    np.save(str(shift_artifact), shifts.astype(np.float32))

    projection_paths = save_projection_tifs(stack, output_dir, input_path.stem)

    return {
        "input_file": str(input_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_summary": basic,
        "bleaching_trend": bleaching,
        "snr_metric": snr,
        "raw_motion_metric": motion_summary,
        "artifacts": {
            **projection_paths,
            "rigid_shifts_npy": str(shift_artifact),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute input movie metrics for NeuroPilot.")
    parser.add_argument("--input", required=True, type=str, help="Path to input TIFF stack.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Default: <input_stem>_input_metrics beside input file.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Frame rate in Hz.")
    parser.add_argument("--pixel-size-um", type=float, default=None, help="Pixel size in um.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if args.output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_input_metrics"
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stack = load_tif_anyshape(input_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected a 3D stack after loading, got shape={stack.shape}")

    metrics = build_metrics_json(
        stack=stack,
        input_path=input_path,
        output_dir=output_dir,
        fps=args.fps,
        pixel_size_um=args.pixel_size_um,
    )

    json_path = output_dir / f"{input_path.stem}_input_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[OK] Metrics JSON: {json_path}")
    print(f"[OK] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
