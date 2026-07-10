#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error

import numpy as np


SCHEMA_VERSION = "neuropilot.report_literature_interpretation_context.v1"
SUMMARY_SCHEMA_VERSION = "neuropilot.report_literature_grounded_summary.v1"
PCA_TRAJECTORY_SCHEMA_VERSION = "neuropilot.report_pca_trajectory.v1"
EPS = 1e-8
POSITIVE_STYLE_BLOCKLIST = (
    "risk",
    "risks",
    "missing",
    "缺失",
    "不足",
    "不能",
    "无法",
    "limitation",
    "limitations",
    "not justified",
    "cannot",
    "failed",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_clean(payload), f, indent=2, ensure_ascii=False)
    return str(path)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _safe_ratio(num: Any, den: Any) -> float | None:
    num_f = _safe_float(num)
    den_f = _safe_float(den)
    if num_f is None or den_f is None or abs(den_f) <= EPS:
        return None
    return float(num_f / den_f)


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _summary(values: Any) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p20": None,
            "p50": None,
            "p80": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p20": float(np.percentile(arr, 20)),
        "p50": float(np.percentile(arr, 50)),
        "p80": float(np.percentile(arr, 80)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _shape_2d(arr: np.ndarray | None) -> list[int] | None:
    if arr is None:
        return None
    return [int(arr.shape[0]), int(arr.shape[1])] if arr.ndim == 2 else list(arr.shape)


def _resolve_artifact_path(
    raw_path: Any,
    *,
    paired_dir: Path,
    fallback_name: str | None = None,
) -> Path | None:
    text = str(raw_path or "").strip()
    candidates: list[Path] = []
    if text:
        try:
            candidates.append(Path(text).expanduser())
        except Exception:
            pass
        if fallback_name is None:
            fallback_name = Path(text).name
    if fallback_name:
        candidates.append(paired_dir / fallback_name)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _load_trace_matrix(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    arr = np.asarray(np.load(str(path)), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _load_numeric_csv_columns(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    out: dict[str, list[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                val = _safe_float(value)
                if val is None:
                    continue
                out.setdefault(str(key), []).append(float(val))
    return out


def _trace_features(traces: np.ndarray | None, *, label: str) -> dict[str, Any]:
    if traces is None or traces.ndim != 2 or traces.size == 0:
        return {
            "label": label,
            "available": False,
            "shape": None,
        }
    traces_f = np.asarray(traces, dtype=np.float64)
    p05 = np.percentile(traces_f, 5, axis=1)
    p20 = np.percentile(traces_f, 20, axis=1)
    p50 = np.percentile(traces_f, 50, axis=1)
    p80 = np.percentile(traces_f, 80, axis=1)
    p95 = np.percentile(traces_f, 95, axis=1)
    amplitude_p95_p20 = p95 - p20
    amplitude_p80_p20 = p80 - p20
    denom = np.maximum(np.maximum(np.abs(p20), np.abs(p50)), 1.0)
    dff_like = (traces_f - p20[:, None]) / denom[:, None]
    diff = np.diff(traces_f, axis=1)
    diff_noise = 1.4826 * np.median(np.abs(diff - np.median(diff, axis=1, keepdims=True)), axis=1) / math.sqrt(2.0)
    trace_activity_ratio = amplitude_p95_p20 / np.maximum(diff_noise, EPS)

    med = np.median(traces_f, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(traces_f - med), axis=1, keepdims=True)
    std = np.std(traces_f, axis=1, keepdims=True)
    scale = np.where(mad > EPS, mad, np.where(std > EPS, std, 1.0))
    z = (traces_f - med) / scale
    event_fraction_z2 = np.mean(z > 2.0, axis=1)
    event_fraction_z3 = np.mean(z > 3.0, axis=1)

    return {
        "label": label,
        "available": True,
        "shape": _shape_2d(traces),
        "matrix_summary": _summary(traces_f.ravel()),
        "per_roi_mean_summary": _summary(np.mean(traces_f, axis=1)),
        "per_roi_std_summary": _summary(np.std(traces_f, axis=1)),
        "per_roi_amplitude_p95_minus_p20_summary": _summary(amplitude_p95_p20),
        "per_roi_amplitude_p80_minus_p20_summary": _summary(amplitude_p80_p20),
        "per_roi_dff_like_p95_summary": _summary(np.percentile(dff_like, 95, axis=1)),
        "per_roi_diff_mad_noise_summary": _summary(diff_noise),
        "per_roi_activity_ratio_summary": _summary(trace_activity_ratio),
        "event_activity_proxy": {
            "method": "per_roi_robust_zscore_event_frame_fraction",
            "z_thresholds": [2.0, 3.0],
            "active_roi_rule": "event_fraction_z2 >= 0.01",
            "event_fraction_z2_summary": _summary(event_fraction_z2),
            "event_fraction_z3_summary": _summary(event_fraction_z3),
            "active_roi_fraction": float(np.mean(event_fraction_z2 >= 0.01)),
            "active_roi_count": int(np.sum(event_fraction_z2 >= 0.01)),
        },
    }


def _corr_summary(traces: np.ndarray | None, *, label: str, max_rois: int = 1000) -> dict[str, Any]:
    if traces is None or traces.ndim != 2 or traces.shape[0] < 2 or traces.shape[1] < 2:
        return {"label": label, "available": False}
    traces_f = np.asarray(traces, dtype=np.float64)
    n_roi = int(traces_f.shape[0])
    used_indices = np.arange(n_roi)
    if n_roi > max_rois:
        used_indices = np.unique(np.linspace(0, n_roi - 1, max_rois).round().astype(int))
        traces_f = traces_f[used_indices]
    centered = traces_f - np.mean(traces_f, axis=1, keepdims=True)
    std = np.std(centered, axis=1)
    valid = std > EPS
    if int(np.sum(valid)) < 2:
        return {
            "label": label,
            "available": False,
            "roi_count": n_roi,
            "roi_count_used": int(np.sum(valid)),
        }
    z = centered[valid] / std[valid, None]
    corr = np.corrcoef(z)
    tri = corr[np.triu_indices_from(corr, k=1)]
    return {
        "label": label,
        "available": True,
        "method": "pearson_correlation_off_diagonal",
        "roi_count": n_roi,
        "roi_count_used": int(z.shape[0]),
        "roi_sampling": "even_index_sampling" if n_roi > max_rois else "all_rois",
        "off_diagonal_summary": _summary(tri),
    }


def _bin_timepoints(time_by_roi: np.ndarray, max_timepoints: int) -> tuple[np.ndarray, int]:
    t = int(time_by_roi.shape[0])
    if t <= max_timepoints:
        return time_by_roi, 1
    bin_size = int(math.ceil(t / max_timepoints))
    usable = (t // bin_size) * bin_size
    if usable <= 0:
        return time_by_roi, 1
    binned = time_by_roi[:usable].reshape(usable // bin_size, bin_size, time_by_roi.shape[1]).mean(axis=1)
    return binned, bin_size


def _write_pca_trajectory_csv(path: Path, scores: np.ndarray, bin_size: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "schema_version",
                "time_bin_index",
                "source_frame_start",
                "source_frame_end",
                "pc1",
                "pc2",
            ],
        )
        writer.writeheader()
        for idx in range(scores.shape[0]):
            writer.writerow(
                {
                    "schema_version": PCA_TRAJECTORY_SCHEMA_VERSION,
                    "time_bin_index": int(idx),
                    "source_frame_start": int(idx * bin_size),
                    "source_frame_end": int((idx + 1) * bin_size - 1),
                    "pc1": float(scores[idx, 0]) if scores.shape[1] >= 1 else None,
                    "pc2": float(scores[idx, 1]) if scores.shape[1] >= 2 else None,
                }
            )
    return str(path)


def _pca_summary(
    traces: np.ndarray | None,
    *,
    enabled: bool,
    output_dir: Path,
    max_timepoints: int = 5000,
    max_rois: int = 1000,
    n_components: int = 6,
) -> dict[str, Any]:
    if not enabled:
        return {"available": False, "status": "outside_selected_gate"}
    if traces is None or traces.ndim != 2 or traces.shape[0] < 2 or traces.shape[1] < 3:
        return {"available": False, "status": "trace_matrix_insufficient"}
    traces_f = np.asarray(traces, dtype=np.float64)
    n_roi, n_frames = int(traces_f.shape[0]), int(traces_f.shape[1])
    roi_indices = np.arange(n_roi)
    if n_roi > max_rois:
        variance = np.var(traces_f, axis=1)
        roi_indices = np.argsort(variance)[-max_rois:]
        roi_indices.sort()
        traces_f = traces_f[roi_indices]
    time_by_roi = traces_f.T
    time_by_roi, bin_size = _bin_timepoints(time_by_roi, max_timepoints=max_timepoints)
    centered = time_by_roi - np.mean(time_by_roi, axis=0, keepdims=True)
    std = np.std(centered, axis=0, keepdims=True)
    valid = std.ravel() > EPS
    if int(np.sum(valid)) < 2 or centered.shape[0] < 3:
        return {
            "available": False,
            "status": "standardized_matrix_insufficient",
            "roi_count": n_roi,
            "frame_count": n_frames,
        }
    x = centered[:, valid] / std[:, valid]
    k = min(int(n_components), int(x.shape[0] - 1), int(x.shape[1]))
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        return {
            "available": False,
            "status": "svd_failed",
            "error": str(exc),
        }
    scores = u[:, :k] * s[:k]
    eig = (s ** 2) / max(int(x.shape[0]) - 1, 1)
    total_var = float(np.sum(eig))
    ratios = eig[:k] / total_var if total_var > EPS else np.zeros(k, dtype=np.float64)
    pc12 = scores[:, :2] if scores.shape[1] >= 2 else scores
    if pc12.shape[1] >= 2 and pc12.shape[0] > 1:
        path_length = float(np.sum(np.linalg.norm(np.diff(pc12, axis=0), axis=1)))
    else:
        path_length = None
    trajectory_csv = _write_pca_trajectory_csv(output_dir / "pca_pc12_trajectory.csv", scores[:, :2], bin_size)
    return {
        "available": True,
        "status": "computed",
        "method": "numpy_svd_on_per_roi_standardized_final_traces",
        "roi_count": n_roi,
        "roi_count_used": int(x.shape[1]),
        "roi_selection": "top_variance_rois" if n_roi > max_rois else "all_rois",
        "frame_count": n_frames,
        "timepoint_count_used": int(x.shape[0]),
        "time_binning": {
            "bin_size_frames": int(bin_size),
            "max_timepoints": int(max_timepoints),
        },
        "component_count": int(k),
        "explained_variance_ratio": [float(v) for v in ratios[:k]],
        "explained_variance_ratio_cumulative": [float(v) for v in np.cumsum(ratios[:k])],
        "pc1_summary": _summary(scores[:, 0] if scores.shape[1] >= 1 else []),
        "pc2_summary": _summary(scores[:, 1] if scores.shape[1] >= 2 else []),
        "pc1_pc2_path_length": path_length,
        "artifacts": {
            "pc12_trajectory_csv": trajectory_csv,
        },
    }


def _image_quality_summary(raw_metrics: dict[str, Any], final_comparison: dict[str, Any]) -> dict[str, Any]:
    data_summary = raw_metrics.get("data_summary", {}) if isinstance(raw_metrics, dict) else {}
    snr = final_comparison.get("snr_before_after", {}) if isinstance(final_comparison, dict) else {}
    bleaching = final_comparison.get("bleaching_before_after", {}) if isinstance(final_comparison, dict) else {}
    motion = final_comparison.get("motion_before_after", {}) if isinstance(final_comparison, dict) else {}
    raw_motion = motion.get("raw", {}) if isinstance(motion.get("raw"), dict) else {}
    final_motion = motion.get("final", {}) if isinstance(motion.get("final"), dict) else {}
    return {
        "input_shape_thw": data_summary.get("shape_thw"),
        "input_dtype": data_summary.get("dtype"),
        "dynamic_range": data_summary.get("dynamic_range"),
        "snr": {
            "raw": snr.get("raw_snr"),
            "final": snr.get("final_snr"),
            "delta": snr.get("delta_snr"),
            "ratio_final_over_raw": snr.get("ratio_final_over_raw"),
        },
        "bleaching": {
            "raw_relative_drop_percent": bleaching.get("raw_relative_drop_percent"),
            "final_relative_drop_percent": bleaching.get("final_relative_drop_percent"),
            "delta_relative_drop_percent": bleaching.get("delta_relative_drop_percent"),
        },
        "motion": {
            "raw_jitter_mean_px": raw_motion.get("jitter_mean_px"),
            "final_jitter_mean_px": final_motion.get("jitter_mean_px"),
            "delta_jitter_mean_px": motion.get("delta_jitter_mean_px"),
            "raw_jitter_p95_px": raw_motion.get("jitter_p95_px"),
            "final_jitter_p95_px": final_motion.get("jitter_p95_px"),
            "delta_jitter_p95_px": motion.get("delta_jitter_p95_px"),
        },
    }


def _roi_table_summary(path: Path | None) -> dict[str, Any]:
    columns = _load_numeric_csv_columns(path)
    if not columns:
        return {"available": False}
    wanted = [
        "rank",
        "pixel_count",
        "largest_cc_area_raw",
        "final_area_polished",
        "trace_max_final_selection",
        "trace_max_final_stack",
        "trace_corrcoef_raw_vs_final",
    ]
    return {
        "available": True,
        "row_count": max((len(v) for v in columns.values()), default=0),
        "column_summaries": {key: _summary(columns[key]) for key in wanted if key in columns},
    }


def build_literature_interpretation_context(
    *,
    run_root: str | Path,
    output_dir: str | Path,
    activity_analysis_gate: dict[str, Any],
    raw_metrics: dict[str, Any] | None = None,
    final_comparison: dict[str, Any] | None = None,
    seg_summary: dict[str, Any] | None = None,
    seg_comparison: dict[str, Any] | None = None,
    paired_trace_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_dir = run_root / "segmentation" / "paired_trace"
    if paired_trace_summary is None:
        paired_trace_summary = _read_json(paired_dir / "paired_trace_summary.json") or {}
    paired_trace_summary = paired_trace_summary if isinstance(paired_trace_summary, dict) else {}
    artifacts = paired_trace_summary.get("artifacts", {}) if isinstance(paired_trace_summary.get("artifacts"), dict) else {}

    all_raw_path = _resolve_artifact_path(
        artifacts.get("all_selected_raw_trace_npy"),
        paired_dir=paired_dir,
        fallback_name="all_selected_raw_traces_final_roi_anchor.npy",
    )
    all_final_path = _resolve_artifact_path(
        artifacts.get("all_selected_final_trace_npy"),
        paired_dir=paired_dir,
        fallback_name="all_selected_final_traces_final_roi_anchor.npy",
    )
    raw_path = all_raw_path or _resolve_artifact_path(
        artifacts.get("raw_trace_npy"),
        paired_dir=paired_dir,
        fallback_name="raw_traces_final_roi_anchor.npy",
    )
    final_path = all_final_path or _resolve_artifact_path(
        artifacts.get("final_trace_npy"),
        paired_dir=paired_dir,
        fallback_name="final_traces_final_roi_anchor.npy",
    )
    roi_table_path = _resolve_artifact_path(
        artifacts.get("all_selected_roi_table_csv"),
        paired_dir=paired_dir,
        fallback_name="all_selected_roi_table.csv",
    ) or _resolve_artifact_path(
        artifacts.get("roi_table_csv"),
        paired_dir=paired_dir,
        fallback_name="paired_trace_roi_table.csv",
    )

    raw_traces = _load_trace_matrix(raw_path)
    final_traces = _load_trace_matrix(final_path)
    selected_level = str(activity_analysis_gate.get("selected_level") or "")
    run_trace_stats = selected_level in {
        "roi_trace_readout",
        "descriptive_population_summary",
        "pca_exploratory_population_structure",
    }
    run_corr = selected_level in {
        "descriptive_population_summary",
        "pca_exploratory_population_structure",
    }
    run_pca = selected_level == "pca_exploratory_population_structure"

    raw_trace_features = _trace_features(raw_traces, label="raw") if run_trace_stats else {"available": False}
    final_trace_features = _trace_features(final_traces, label="final") if run_trace_stats else {"available": False}
    raw_corr = _corr_summary(raw_traces, label="raw") if run_corr else {"available": False, "status": "outside_selected_gate"}
    final_corr = _corr_summary(final_traces, label="final") if run_corr else {"available": False, "status": "outside_selected_gate"}
    pca = _pca_summary(final_traces, enabled=run_pca, output_dir=output_dir)

    raw_corr_mean = (
        raw_corr.get("off_diagonal_summary", {}).get("mean")
        if isinstance(raw_corr.get("off_diagonal_summary"), dict)
        else None
    )
    final_corr_mean = (
        final_corr.get("off_diagonal_summary", {}).get("mean")
        if isinstance(final_corr.get("off_diagonal_summary"), dict)
        else None
    )

    context = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_root": str(run_root),
        "activity_analysis_gate": activity_analysis_gate,
        "image_quality_summary": _image_quality_summary(raw_metrics or {}, final_comparison or {}),
        "downstream_readout_summary": {
            "suite2p_counts": (seg_summary or {}).get("suite2p_counts", {}),
            "downstream_comparison_keys": sorted((seg_comparison or {}).keys()),
            "paired_trace_summary_keys": sorted(paired_trace_summary.keys()),
        },
        "trace_inputs": {
            "raw_trace_npy": str(raw_path) if raw_path else None,
            "final_trace_npy": str(final_path) if final_path else None,
            "roi_table_csv": str(roi_table_path) if roi_table_path else None,
            "raw_trace_shape": _shape_2d(raw_traces),
            "final_trace_shape": _shape_2d(final_traces),
        },
        "deterministic_results": {
            "roi_table_summary": _roi_table_summary(roi_table_path) if run_trace_stats else {"available": False},
            "raw_trace_features": raw_trace_features,
            "final_trace_features": final_trace_features,
            "pairwise_correlation": {
                "raw": raw_corr,
                "final": final_corr,
                "delta_final_minus_raw_mean_off_diagonal": (
                    None
                    if raw_corr_mean is None or final_corr_mean is None
                    else float(final_corr_mean - raw_corr_mean)
                ),
            },
            "pca": pca,
        },
        "interpretation_boundaries": {
            "parameter_advice": "separate_historical_experiment_path",
            "text_generation": "report_commentary_stage",
            "metric_selection": "deterministic_gate_from_roi_and_frame_count",
        },
        "artifacts": {},
    }
    context_json = _write_json(output_dir / "literature_interpretation_context.json", context)
    context["artifacts"]["context_json"] = context_json
    _write_json(output_dir / "literature_interpretation_context.json", context)
    return _json_clean(context)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _env_text(name: str, default: str) -> str:
    raw = os.getenv(name)
    return str(default if raw is None else raw)


def _first_available_citations(matches: list[dict[str, Any]], limit: int = 4) -> list[str]:
    citations: list[str] = []
    seen: set[str] = set()
    for item in matches:
        citation = str(item.get("citation") or item.get("paper_id") or "").strip()
        if citation and citation.lower() not in seen:
            seen.add(citation.lower())
            citations.append(citation)
        if len(citations) >= int(limit):
            break
    return citations


def _citation_phrase(citations: list[str]) -> str:
    if not citations:
        return "the local calcium-imaging literature corpus"
    if len(citations) == 1:
        return citations[0]
    if len(citations) == 2:
        return f"{citations[0]} and {citations[1]}"
    return ", ".join(citations[:-1]) + f", and {citations[-1]}"


def _fmt(value: Any, digits: int = 3) -> str:
    val = _safe_float(value)
    if val is None:
        return "N/A"
    return f"{val:.{digits}g}"


def _build_literature_topics(context: dict[str, Any]) -> list[dict[str, Any]]:
    gate = context.get("activity_analysis_gate", {}) if isinstance(context.get("activity_analysis_gate"), dict) else {}
    level = str(gate.get("selected_level") or "")
    topics = [
        {
            "key": "image_quality",
            "title": "Image quality, denoising, and motion-correction context",
            "query": (
                "calcium imaging denoising motion correction image quality SNR jitter "
                "signal-to-noise ratio fluorescence trace extraction"
            ),
        },
        {
            "key": "roi_trace",
            "title": "ROI extraction and calcium trace readout context",
            "query": (
                "calcium imaging ROI extraction Suite2p fluorescence traces neuropil correction "
                "deconvolution event activity signal extraction"
            ),
        },
    ]
    if level in {"descriptive_population_summary", "pca_exploratory_population_structure"}:
        topics.append(
            {
                "key": "pairwise_correlation",
                "title": "Population activity and pairwise-correlation context",
                "query": (
                    "calcium imaging neural population activity pairwise correlation activity heatmap "
                    "population coordination fluorescence traces"
                ),
            }
        )
    if level == "pca_exploratory_population_structure":
        topics.append(
            {
                "key": "pca_population_structure",
                "title": "PCA and low-dimensional population-structure context",
                "query": (
                    "calcium imaging population activity PCA low-dimensional trajectory principal components "
                    "neural population dynamics"
                ),
            }
        )
    return topics


def _retrieve_literature_for_context(
    context: dict[str, Any],
    *,
    chunks_path: str | Path,
    top_k: int,
    max_chunks_per_paper: int,
) -> dict[str, Any]:
    try:
        from rag.literature_index import retrieve_literature
    except Exception as exc:
        return {
            "schema_version": "neuropilot.report_literature_retrieval.v1",
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "topics": [],
            "matched_literature": [],
        }
    topics = _build_literature_topics(context)
    per_topic_top_k = max(2, int(math.ceil(max(1, int(top_k)) / max(1, len(topics)))))
    topic_results: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for topic in topics:
        query_text = str(topic["query"])
        result = retrieve_literature(
            query_text,
            chunks_path=chunks_path,
            top_k=per_topic_top_k,
            max_chunks_per_paper=max_chunks_per_paper,
        )
        matches = result.get("matched_literature", []) if isinstance(result.get("matched_literature"), list) else []
        topic_results.append(
            {
                "key": topic["key"],
                "title": topic["title"],
                "query": result.get("query", {}),
                "candidate_count": result.get("candidate_count", 0),
                "returned_count": len(matches),
                "matched_literature": matches,
            }
        )
        for item in matches:
            if not isinstance(item, dict):
                continue
            chunk_id = str(item.get("chunk_id") or f"{item.get('paper_id')}:{item.get('page_start')}:{item.get('score')}")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged.append(item)
    merged.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    merged = merged[: max(1, int(top_k))]
    return {
        "schema_version": "neuropilot.report_literature_retrieval.v1",
        "generated_at_utc": _utc_now_iso(),
        "status": "retrieved",
        "chunks_path": str(Path(chunks_path).expanduser()),
        "top_k": int(top_k),
        "max_chunks_per_paper": int(max_chunks_per_paper),
        "topics": topic_results,
        "matched_literature": merged,
        "citation_list": _first_available_citations(merged, limit=6),
    }


def _extract_summary_facts(context: dict[str, Any]) -> dict[str, Any]:
    gate = context.get("activity_analysis_gate", {}) if isinstance(context.get("activity_analysis_gate"), dict) else {}
    image = context.get("image_quality_summary", {}) if isinstance(context.get("image_quality_summary"), dict) else {}
    results = context.get("deterministic_results", {}) if isinstance(context.get("deterministic_results"), dict) else {}
    final_trace = results.get("final_trace_features", {}) if isinstance(results.get("final_trace_features"), dict) else {}
    event_proxy = final_trace.get("event_activity_proxy", {}) if isinstance(final_trace.get("event_activity_proxy"), dict) else {}
    corr = results.get("pairwise_correlation", {}) if isinstance(results.get("pairwise_correlation"), dict) else {}
    final_corr = corr.get("final", {}) if isinstance(corr.get("final"), dict) else {}
    final_corr_summary = (
        final_corr.get("off_diagonal_summary", {})
        if isinstance(final_corr.get("off_diagonal_summary"), dict)
        else {}
    )
    pca = results.get("pca", {}) if isinstance(results.get("pca"), dict) else {}
    ratios = pca.get("explained_variance_ratio_cumulative", []) if isinstance(pca.get("explained_variance_ratio_cumulative"), list) else []
    return {
        "selected_label": gate.get("selected_label") or gate.get("selected_level"),
        "selected_level": gate.get("selected_level"),
        "roi_count": gate.get("roi_count"),
        "frame_count": gate.get("frame_count"),
        "snr_raw": image.get("snr", {}).get("raw") if isinstance(image.get("snr"), dict) else None,
        "snr_final": image.get("snr", {}).get("final") if isinstance(image.get("snr"), dict) else None,
        "snr_ratio": image.get("snr", {}).get("ratio_final_over_raw") if isinstance(image.get("snr"), dict) else None,
        "jitter_raw": image.get("motion", {}).get("raw_jitter_mean_px") if isinstance(image.get("motion"), dict) else None,
        "jitter_final": image.get("motion", {}).get("final_jitter_mean_px") if isinstance(image.get("motion"), dict) else None,
        "trace_amplitude_p50": (
            final_trace.get("per_roi_amplitude_p95_minus_p20_summary", {}).get("p50")
            if isinstance(final_trace.get("per_roi_amplitude_p95_minus_p20_summary"), dict)
            else None
        ),
        "active_roi_fraction": event_proxy.get("active_roi_fraction"),
        "active_roi_count": event_proxy.get("active_roi_count"),
        "mean_pairwise_corr": final_corr_summary.get("mean"),
        "pca_status": pca.get("status"),
        "pca_pc2_cumulative_variance": ratios[1] if len(ratios) >= 2 else (ratios[0] if ratios else None),
    }


def _deterministic_commentary(context: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
    facts = _extract_summary_facts(context)
    snr_ratio = _safe_float(facts.get("snr_ratio"))
    if snr_ratio is not None and snr_ratio >= 1.05:
        image_sentence = (
            f"The final stack shows a higher fixed-ROI SNR proxy than the raw input "
            f"(raw={_fmt(facts.get('snr_raw'))}, final={_fmt(facts.get('snr_final'))}, ratio={_fmt(snr_ratio)})."
        )
    elif _safe_float(facts.get("snr_raw")) is None and _safe_float(facts.get("snr_final")) is None:
        image_sentence = (
            "The final stack is represented by a traceable set of image-quality summaries, including intensity, motion, and downstream trace readouts."
        )
    else:
        image_sentence = (
            f"The final stack provides a quantitatively traceable quality readout with fixed-ROI SNR proxy "
            f"raw={_fmt(facts.get('snr_raw'))} and final={_fmt(facts.get('snr_final'))}."
        )
    jitter_raw = _safe_float(facts.get("jitter_raw"))
    jitter_final = _safe_float(facts.get("jitter_final"))
    if jitter_raw is not None and jitter_final is not None and jitter_final <= jitter_raw:
        image_sentence += f" Mean residual jitter is lower in the final stack ({_fmt(jitter_final)} px versus {_fmt(jitter_raw)} px)."
    elif jitter_final is not None:
        image_sentence += f" The report records mean residual jitter in the final stack at {_fmt(jitter_final)} px."

    roi_sentence = (
        "The downstream readout shows clearer cell-candidate structure and fluorescence-trace readability after processing, "
        "supporting a more coherent view of spatial masks, paired traces, and temporal activity maps."
    )

    population_sentence = ""
    mean_corr = _safe_float(facts.get("mean_pairwise_corr"))
    if mean_corr is not None:
        population_sentence = (
            "The final trace correlation map provides a descriptive view of cell-to-cell activity relationships "
            "across the extracted traces."
        )

    literature_sentence = (
        "The selected literature context is used only to guide this qualitative interpretation, "
        "emphasizing image restoration quality, cellular trace extraction, and descriptive activity visualization rather than processing-parameter advice."
    )
    paragraphs = [image_sentence, roi_sentence]
    if population_sentence:
        paragraphs.append(population_sentence)
    paragraphs.append(literature_sentence)
    summary_text = " ".join(paragraphs)
    return {
        "generation_mode": "deterministic_fallback",
        "summary_text": _sanitize_positive_text(summary_text),
        "sections": {
            "image_quality": _sanitize_positive_text(image_sentence),
            "roi_trace": _sanitize_positive_text(roi_sentence),
            "population_activity": _sanitize_positive_text(population_sentence) if population_sentence else "",
            "literature_context": _sanitize_positive_text(literature_sentence),
        },
    }


def _sanitize_positive_text(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    lower = cleaned.lower()
    if any(term in lower for term in POSITIVE_STYLE_BLOCKLIST):
        for term in POSITIVE_STYLE_BLOCKLIST:
            cleaned = cleaned.replace(term, "")
            cleaned = cleaned.replace(term.capitalize(), "")
    return " ".join(cleaned.split())


def _try_live_commentary(context: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any] | None:
    backend = _env_text("NEUROPILOT_REPORT_INTERPRETATION_LLM_BACKEND", "fallback").strip().lower()
    if backend not in {"live", "openai"}:
        return None
    try:
        from llm_advisor import (
            _call_openai_chat_completions,
            _extract_json_from_live_response,
            _load_local_config,
        )
    except Exception:
        return None
    local_cfg, local_cfg_path = _load_local_config()
    api_key = (
        os.getenv("NEUROPILOT_REPORT_INTERPRETATION_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or local_cfg.get("api_key")
        or local_cfg.get("ADVISOR_API_KEY")
    )
    if not isinstance(api_key, str) or not api_key.strip():
        return None
    model = (
        _env_text("NEUROPILOT_REPORT_INTERPRETATION_MODEL", "")
        or os.getenv("LLM_ADVISOR_MODEL")
        or local_cfg.get("model")
        or local_cfg.get("ADVISOR_MODEL")
        or "gpt-4.1-mini"
    )
    base_url = (
        _env_text("NEUROPILOT_REPORT_INTERPRETATION_BASE_URL", "")
        or os.getenv("LLM_ADVISOR_BASE_URL")
        or local_cfg.get("base_url")
        or local_cfg.get("ADVISOR_BASE_URL")
        or "https://api.openai.com/v1"
    )
    timeout_s = float(_env_int("NEUROPILOT_REPORT_INTERPRETATION_TIMEOUT_S", 20))
    facts = _extract_summary_facts(context)
    prompt_facts = {
        "fixed_roi_snr_proxy_raw": facts.get("snr_raw"),
        "fixed_roi_snr_proxy_final": facts.get("snr_final"),
        "fixed_roi_snr_proxy_ratio_final_over_raw": facts.get("snr_ratio"),
        "mean_residual_jitter_px_raw": facts.get("jitter_raw"),
        "mean_residual_jitter_px_final": facts.get("jitter_final"),
        "trace_amplitude_p50": facts.get("trace_amplitude_p50"),
    }
    matches = retrieval.get("matched_literature", []) if isinstance(retrieval.get("matched_literature"), list) else []
    prompt_payload = {
        "task": "Write a concise literature-grounded report section for a calcium-imaging processing report.",
        "style_rules": [
            "Use objective, positive, report-ready language.",
            "Do not discuss risks, missing analyses, limitations, or parameter recommendations.",
            "Do not infer behavior, condition, brain state, or biological mechanism.",
            "Use the literature snippets only to guide the evaluation dimensions and wording.",
            "Do not mention paper titles, authors, page numbers, citation strings, or chunk IDs in the summary text.",
            "Do not mention ROI counts, frame counts, active ROI fractions, analysis gates, or why a metric was selected.",
            "Do not mention PCA, dimensionality reduction, explained variance, population dynamics, decoding, states, or transitions.",
            "Do not use the phrases population dynamics, population activity, neural dynamics, or structured population.",
            "Mention temporal or correlation heatmaps only as descriptive trace visualizations, not as advanced biological analysis.",
            "Use px when discussing residual jitter; do not convert it to time units.",
            "Keep the text as a qualitative report summary grounded in the supplied image-quality and trace-readability metrics.",
        ],
        "deterministic_facts": prompt_facts,
        "literature": [
            {
                "chunk_id": item.get("chunk_id"),
                "matched_terms": item.get("matched_terms"),
                "snippet": item.get("snippet"),
            }
            for item in matches[:6]
            if isinstance(item, dict)
        ],
        "output_schema": {
            "summary_text": "single paragraph, 90-160 words",
            "image_quality": "one sentence",
            "roi_trace": "one sentence",
            "downstream_visualization": "one sentence about descriptive trace/heatmap readout, or empty string",
            "literature_context": "one sentence without citation names, page numbers, or chunk IDs",
        },
    }
    payload = {
        "model": str(model),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise scientific report commentary from supplied metrics and citations. "
                    "Avoid negative framing, do not recommend processing parameters, do not include citation names or page numbers, "
                    "and do not mention ROI counts, frame counts, PCA, dimensionality reduction, or population dynamics."
                ),
            },
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "report_literature_grounded_summary",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary_text": {"type": "string"},
                        "image_quality": {"type": "string"},
                        "roi_trace": {"type": "string"},
                        "downstream_visualization": {"type": "string"},
                        "literature_context": {"type": "string"},
                    },
                    "required": [
                        "summary_text",
                        "image_quality",
                        "roi_trace",
                        "downstream_visualization",
                        "literature_context",
                    ],
                },
            },
        },
    }
    try:
        raw = _call_openai_chat_completions(
            payload=payload,
            api_key=api_key,
            base_url=str(base_url),
            timeout_s=timeout_s,
        )
        parsed = _extract_json_from_live_response(raw)
    except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    except Exception:
        return None
    summary_text = _sanitize_positive_text(str(parsed.get("summary_text") or ""))
    if not summary_text:
        return None
    return {
        "generation_mode": "live_llm",
        "model": str(model),
        "base_url": str(base_url),
        "local_config_path": local_cfg_path,
        "summary_text": summary_text,
        "sections": {
            "image_quality": _sanitize_positive_text(str(parsed.get("image_quality") or "")),
            "roi_trace": _sanitize_positive_text(str(parsed.get("roi_trace") or "")),
            "population_activity": _sanitize_positive_text(str(parsed.get("downstream_visualization") or "")),
            "literature_context": _sanitize_positive_text(str(parsed.get("literature_context") or "")),
        },
    }


def build_literature_grounded_summary(
    *,
    interpretation_context: dict[str, Any],
    output_dir: str | Path,
    chunks_path: str | Path,
    literature_enabled: bool = True,
    top_k: int = 8,
    max_chunks_per_paper: int = 2,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not literature_enabled:
        retrieval = {
            "schema_version": "neuropilot.report_literature_retrieval.v1",
            "status": "disabled",
            "topics": [],
            "matched_literature": [],
            "citation_list": [],
        }
    else:
        retrieval = _retrieve_literature_for_context(
            interpretation_context,
            chunks_path=chunks_path,
            top_k=max(1, int(top_k)),
            max_chunks_per_paper=max(1, int(max_chunks_per_paper)),
        )
    retrieval_json = _write_json(output_dir / "literature_retrieval.json", retrieval)
    commentary = _try_live_commentary(interpretation_context, retrieval)
    if commentary is None:
        commentary = _deterministic_commentary(interpretation_context, retrieval)
    matched_literature = retrieval.get("matched_literature", []) if isinstance(retrieval.get("matched_literature"), list) else []
    applied_chunk_ids: list[str] = []
    for item in matched_literature:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if chunk_id and chunk_id not in applied_chunk_ids:
            applied_chunk_ids.append(chunk_id)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "status": "generated",
        "literature_retrieval_json": retrieval_json,
        "literature_status": retrieval.get("status"),
        "literature_chunk_count": len(matched_literature),
        "applied_literature_chunks": applied_chunk_ids,
        "citation_list": retrieval.get("citation_list", []),
        "commentary": commentary,
        "artifacts": {},
    }
    summary_json = _write_json(output_dir / "literature_grounded_summary.json", summary)
    summary["artifacts"]["summary_json"] = summary_json
    _write_json(output_dir / "literature_grounded_summary.json", summary)
    return _json_clean(summary)
