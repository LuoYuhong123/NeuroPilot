#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from input_metrics import (
    build_snr_reference,
    compute_basic_stats,
    compute_bleaching_summary,
    compute_snr_metric,
    estimate_raw_motion_rigid,
    load_tif_anyshape,
    save_projection_tifs,
)
from report_figures import create_comparison_png_assets


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def _extract_motion_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    motion = metrics.get("rigid_motion_metric", {})
    return {
        "motion_mean_px": _safe_float(motion.get("rigid_motion_summary", {}).get("motion_mean_px")),
        "motion_median_px": _safe_float(motion.get("rigid_motion_summary", {}).get("motion_median_px")),
        "motion_p95_px": _safe_float(motion.get("rigid_motion_summary", {}).get("motion_p95_px")),
        "motion_max_px": _safe_float(motion.get("rigid_motion_summary", {}).get("motion_max_px")),
        "jitter_mean_px": _safe_float(motion.get("frame_to_frame_jitter", {}).get("jitter_mean_px")),
        "jitter_p95_px": _safe_float(motion.get("frame_to_frame_jitter", {}).get("jitter_p95_px")),
        "jitter_max_px": _safe_float(motion.get("frame_to_frame_jitter", {}).get("jitter_max_px")),
        "corr_mean": _safe_float(motion.get("registration_confidence", {}).get("corr_mean")),
        "corr_median": _safe_float(motion.get("registration_confidence", {}).get("corr_median")),
        "corr_min": _safe_float(motion.get("registration_confidence", {}).get("corr_min")),
    }


def compute_metrics_for_tif(
    input_path: str | Path,
    output_dir: str | Path,
    fps: float | None = None,
    pixel_size_um: float | None = None,
    snr_reference_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute reusable stack metrics for a TIFF stack and persist artifacts.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF not found: {input_path}")

    stack = load_tif_anyshape(input_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (T,H,W), got shape={stack.shape}")

    basic = compute_basic_stats(stack, fps=fps, pixel_size_um=pixel_size_um)
    bleaching = compute_bleaching_summary(stack, fps=fps)
    snr_reference_label = str(input_path)
    snr_roi_mask: np.ndarray | None = None
    snr_roi_percentile = 90.0
    snr_roi_source = "self"
    if snr_reference_metrics:
        snr_roi_percentile = float(
            snr_reference_metrics.get("snr_metric", {}).get("std_roi_percentile_threshold", 90.0)
        )
        ref_artifacts = snr_reference_metrics.get("artifacts", {})
        ref_mask_path = ref_artifacts.get("snr_roi_mask_npy")
        if ref_mask_path and Path(ref_mask_path).exists():
            snr_roi_mask = np.load(str(ref_mask_path)).astype(bool)
        else:
            ref_input_file = snr_reference_metrics.get("input_file")
            if ref_input_file and Path(ref_input_file).exists():
                ref_stack = load_tif_anyshape(ref_input_file)
                snr_roi_mask = build_snr_reference(ref_stack, roi_percentile=snr_roi_percentile)["roi_mask"]
        if snr_roi_mask is not None:
            snr_roi_source = "reference_metrics"
            snr_reference_label = str(
                ref_artifacts.get("metrics_json")
                or snr_reference_metrics.get("input_file")
                or input_path
            )
    if snr_roi_mask is None:
        snr_ref = build_snr_reference(stack, roi_percentile=snr_roi_percentile)
        snr_roi_mask = snr_ref["roi_mask"]
    snr = compute_snr_metric(
        stack,
        roi_percentile=snr_roi_percentile,
        roi_mask=snr_roi_mask,
        roi_source=snr_roi_source,
        roi_reference_label=snr_reference_label,
    )
    rigid_motion, shifts = estimate_raw_motion_rigid(stack)

    projection_tif_paths = save_projection_tifs(stack, output_dir=output_dir, stem=input_path.stem)
    shifts_path = output_dir / f"{input_path.stem}_rigid_shifts.npy"
    snr_roi_mask_path = output_dir / f"{input_path.stem}_snr_roi_mask.npy"
    np.save(str(shifts_path), shifts.astype(np.float32))
    np.save(str(snr_roi_mask_path), np.asarray(snr_roi_mask, dtype=np.uint8))

    metrics = {
        "schema_version": "pipeline_metrics.v2",
        "input_file": str(input_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_summary": basic,
        "bleaching_trend": bleaching,
        "snr_metric": snr,
        "rigid_motion_metric": rigid_motion,
        "artifacts": {
            "mip_tif": projection_tif_paths["mip_tif"],
            "std_tif": projection_tif_paths["std_tif"],
            "rigid_shifts_npy": str(shifts_path),
            "snr_roi_mask_npy": str(snr_roi_mask_path),
        },
    }

    metrics_json_path = output_dir / f"{input_path.stem}_metrics.json"
    metrics["artifacts"]["metrics_json"] = _write_json(metrics_json_path, metrics)
    return metrics


def compare_two_metrics(raw_metrics: dict[str, Any], final_metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Compare two metric dicts (typically raw vs final) and return a deterministic delta summary.
    """
    raw_shape = raw_metrics.get("data_summary", {}).get("shape_thw")
    final_shape = final_metrics.get("data_summary", {}).get("shape_thw")

    raw_snr = _safe_float(raw_metrics.get("snr_metric", {}).get("snr"))
    final_snr = _safe_float(final_metrics.get("snr_metric", {}).get("snr"))

    raw_bleach = raw_metrics.get("bleaching_trend", {})
    final_bleach = final_metrics.get("bleaching_trend", {})

    raw_motion = _extract_motion_summary(raw_metrics)
    final_motion = _extract_motion_summary(final_metrics)

    comparison = {
        "schema_version": "pipeline_metrics_comparison.v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metadata_delta": {
            "raw_shape_thw": raw_shape,
            "final_shape_thw": final_shape,
            "shape_changed": bool(raw_shape != final_shape),
            "fps_raw": _safe_float(raw_metrics.get("data_summary", {}).get("frame_rate_hz")),
            "fps_final": _safe_float(final_metrics.get("data_summary", {}).get("frame_rate_hz")),
            "pixel_size_um_raw": _safe_float(raw_metrics.get("data_summary", {}).get("pixel_size_um")),
            "pixel_size_um_final": _safe_float(final_metrics.get("data_summary", {}).get("pixel_size_um")),
        },
        "snr_before_after": {
            "method": final_metrics.get("snr_metric", {}).get("method") or raw_metrics.get("snr_metric", {}).get("method"),
            "roi_source_final": final_metrics.get("snr_metric", {}).get("roi_source"),
            "roi_reference_final": final_metrics.get("snr_metric", {}).get("roi_reference_label"),
            "raw_snr": raw_snr,
            "final_snr": final_snr,
            "delta_snr": None if (raw_snr is None or final_snr is None) else float(final_snr - raw_snr),
            "ratio_final_over_raw": None if (raw_snr is None or raw_snr == 0 or final_snr is None) else float(final_snr / raw_snr),
        },
        "bleaching_before_after": {
            "raw_relative_drop_percent": _safe_float(raw_bleach.get("relative_drop_percent")),
            "final_relative_drop_percent": _safe_float(final_bleach.get("relative_drop_percent")),
            "raw_obvious_bleaching_flag": bool(raw_bleach.get("obvious_bleaching_flag", False)),
            "final_obvious_bleaching_flag": bool(final_bleach.get("obvious_bleaching_flag", False)),
            "delta_relative_drop_percent": (
                None
                if (_safe_float(raw_bleach.get("relative_drop_percent")) is None or _safe_float(final_bleach.get("relative_drop_percent")) is None)
                else float(_safe_float(final_bleach.get("relative_drop_percent")) - _safe_float(raw_bleach.get("relative_drop_percent")))
            ),
        },
        "motion_before_after": {
            "raw": raw_motion,
            "final": final_motion,
            "delta_motion_mean_px": (
                None
                if (raw_motion["motion_mean_px"] is None or final_motion["motion_mean_px"] is None)
                else float(final_motion["motion_mean_px"] - raw_motion["motion_mean_px"])
            ),
            "delta_motion_p95_px": (
                None
                if (raw_motion["motion_p95_px"] is None or final_motion["motion_p95_px"] is None)
                else float(final_motion["motion_p95_px"] - raw_motion["motion_p95_px"])
            ),
            "delta_jitter_mean_px": (
                None
                if (raw_motion["jitter_mean_px"] is None or final_motion["jitter_mean_px"] is None)
                else float(final_motion["jitter_mean_px"] - raw_motion["jitter_mean_px"])
            ),
            "delta_jitter_p95_px": (
                None
                if (raw_motion["jitter_p95_px"] is None or final_motion["jitter_p95_px"] is None)
                else float(final_motion["jitter_p95_px"] - raw_motion["jitter_p95_px"])
            ),
        },
        "artifact_paths": {
            "raw_artifacts": raw_metrics.get("artifacts", {}),
            "final_artifacts": final_metrics.get("artifacts", {}),
            "comparison_assets": {},
        },
    }
    return comparison


def run_raw_final_metrics_comparison(
    raw_tif_path: str | Path,
    final_tif_path: str | Path,
    output_dir: str | Path,
    fps: float | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, Any]:
    """
    Convenience function for one-shot deterministic output:
    - raw metrics JSON
    - final metrics JSON
    - comparison JSON
    - report-ready PNG assets
    """
    output_dir = Path(output_dir).expanduser().resolve()
    raw_out = output_dir / "metrics_raw"
    final_out = output_dir / "metrics_final"
    assets_out = output_dir / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_metrics = compute_metrics_for_tif(raw_tif_path, raw_out, fps=fps, pixel_size_um=pixel_size_um)
    final_metrics = compute_metrics_for_tif(
        final_tif_path,
        final_out,
        fps=fps,
        pixel_size_um=pixel_size_um,
        snr_reference_metrics=raw_metrics,
    )

    comparison = compare_two_metrics(raw_metrics, final_metrics)
    png_assets = create_comparison_png_assets(raw_metrics, final_metrics, assets_out)
    comparison["artifact_paths"]["comparison_assets"] = png_assets

    raw_metrics_path = output_dir / "raw_metrics.json"
    final_metrics_path = output_dir / "final_metrics.json"
    comparison_path = output_dir / "comparison.json"

    _write_json(raw_metrics_path, raw_metrics)
    _write_json(final_metrics_path, final_metrics)
    _write_json(comparison_path, comparison)

    return {
        "raw_metrics": raw_metrics,
        "final_metrics": final_metrics,
        "comparison": comparison,
        "paths": {
            "raw_metrics_json": str(raw_metrics_path),
            "final_metrics_json": str(final_metrics_path),
            "comparison_json": str(comparison_path),
            "assets_dir": str(assets_out),
        },
    }
