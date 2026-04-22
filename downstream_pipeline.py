#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
import subprocess
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tifffile as tiff

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _summary_arr(x: np.ndarray | None) -> dict[str, Any]:
    if x is None:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None, "max": None}
    x = np.asarray(x)
    if x.size == 0:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def _fingerprint(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _materialize_final_stack(final_stack_source: Path, final_dir: Path, source_semantic: str) -> tuple[Path, Path]:
    final_dir.mkdir(parents=True, exist_ok=True)
    final_stack_path = final_dir / "final_stack.tif"
    shutil.copy2(final_stack_source, final_stack_path)
    sidecar_path = final_dir / "final_stack_sidecar.json"
    _write_json(
        sidecar_path,
        {
            "final_stack_path": str(final_stack_path),
            "final_stack_source": str(final_stack_source),
            "source_semantic": source_semantic,
            "materialized_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return final_stack_path, sidecar_path


def _resolve_external_python(runner_cfg: dict | None) -> Path:
    runner_cfg = dict(runner_cfg or {})
    python_executable = runner_cfg.get("python_executable")
    if python_executable:
        py_path = Path(str(python_executable)).expanduser().resolve()
        if py_path.exists():
            return py_path
        raise FileNotFoundError(f"Configured downstream python not found: {py_path}")

    env_name = str(runner_cfg.get("env_name") or "").strip()
    if not env_name:
        raise ValueError("runner_config requires either python_executable or env_name")

    exe_name = "python.exe" if os.name == "nt" else "python"
    candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix).resolve().parent / env_name / exe_name)

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(Path(conda_exe).resolve().parents[1] / "envs" / env_name / exe_name)

    current_python = Path(sys.executable).resolve()
    if current_python.parent.name.lower() == env_name.lower():
        candidates.append(current_python)
    elif current_python.parent.parent.name.lower() == "envs":
        candidates.append(current_python.parent.parent / env_name / exe_name)

    home_conda = Path.home() / "anaconda3" / "envs" / env_name / exe_name
    candidates.append(home_conda)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not resolve python for downstream env '{env_name}'. Checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _extract_counts(seg: dict | None, sel: dict | None) -> dict:
    seg_counts = (seg or {}).get("counts", {})
    sel_counts = (sel or {}).get("counts", {})
    return {
        "plane0_total": int(seg_counts.get("plane0_total", 0)),
        "after_intensity_filter": int(seg_counts.get("after_intensity_filter", 0)),
        "after_cc_area_filter": int(sel_counts.get("analysis_after_cc_area_filter", 0)),
        "display_selected_count": int(sel_counts.get("display_selected_count", 0)),
    }


def _extract_trace_summary(sel: dict | None) -> dict:
    trace_parse = (sel or {}).get("trace_parse", {})
    return {
        "trace_count": int(trace_parse.get("trace_count", 0)),
        "trace_max_summary": trace_parse.get("trace_max_summary"),
        "F_summary": trace_parse.get("F_summary"),
        "Fneu_summary": trace_parse.get("Fneu_summary"),
        "spks_summary": trace_parse.get("spks_summary"),
    }


def _normalize_stack_to_thw(arr: np.ndarray, expect_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape == expect_hw:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[1:] == expect_hw:
            return arr
        if arr.shape[:2] == expect_hw:
            return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 4:
        if arr.shape[2:] == expect_hw:
            a, b, h, w = arr.shape
            return arr.reshape(a * b, h, w)
        if arr.shape[:2] == expect_hw:
            h, w, a, b = arr.shape
            return np.transpose(arr, (2, 3, 0, 1)).reshape(a * b, h, w)
    raise ValueError(f"Unsupported TIFF shape {arr.shape} for expected HW={expect_hw}")


def _load_final_roi_rows(
    roi_csv_path: str | Path,
    top_n: int | None = None,
    subset: str = "display",
) -> list[dict[str, Any]]:
    subset_key_map = {
        "display": "in_display_subset",
        "analysis": "in_analysis_set",
        "all": None,
    }
    subset_key = subset_key_map.get(str(subset).strip().lower())
    rows: list[dict[str, Any]] = []
    with open(Path(roi_csv_path).resolve(), "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if subset_key is not None and int(row.get(subset_key, 1)) != 1:
                continue
            rows.append(
                {
                    "rank": int(row["rank"]),
                    "rid": int(row["rid"]),
                    "largest_cc_area_raw": int(row["largest_cc_area_raw"]),
                    "final_area_polished": int(row["final_area_polished"]),
                    "trace_max": float(row["trace_max"]),
                }
            )
    rows.sort(key=lambda x: int(x["rank"]))
    if top_n is None:
        return rows
    return rows[: max(0, int(top_n))]


def _load_rank_masks_from_labelmask(labelmask_path: str | Path, roi_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labelmask = np.asarray(tiff.imread(str(Path(labelmask_path).resolve())))
    labelmask = np.squeeze(labelmask)
    if labelmask.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got shape={labelmask.shape}")

    out: list[dict[str, Any]] = []
    for row in roi_rows:
        rank = int(row["rank"])
        mask = labelmask == rank
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count <= 0:
            raise RuntimeError(f"Rank {rank} not found in final display labelmask")
        item = dict(row)
        item["pixel_count"] = pixel_count
        item["mask"] = mask
        out.append(item)
    return out


def _merge_roi_rows_with_labelmask_ranks(
    labelmask_path: str | Path,
    roi_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    labelmask = np.asarray(tiff.imread(str(Path(labelmask_path).resolve())))
    labelmask = np.asarray(np.squeeze(labelmask), dtype=np.int32)
    if labelmask.ndim != 2:
        raise ValueError(f"Expected 2D label mask, got shape={labelmask.shape}")

    label_ranks = sorted(int(x) for x in np.unique(labelmask) if int(x) > 0)
    row_by_rank = {int(row["rank"]): dict(row) for row in (roi_rows or [])}
    merged: list[dict[str, Any]] = []
    for rank in label_ranks:
        item = dict(row_by_rank.get(rank, {}))
        item["rank"] = int(rank)
        item["rid"] = int(item.get("rid", rank))
        item["largest_cc_area_raw"] = int(item.get("largest_cc_area_raw", 0) or 0)
        item["final_area_polished"] = int(item.get("final_area_polished", 0) or 0)
        item["trace_max"] = float(item.get("trace_max", 0.0) or 0.0)
        merged.append(item)
    return merged


def _extract_mean_traces_from_tif(
    tif_path: str | Path,
    roi_defs: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int]]:
    tif_path = Path(tif_path).resolve()
    if not roi_defs:
        return np.zeros((0, 0), dtype=np.float32), {"pages_total": 0, "frames_used": 0, "pages_skipped": 0}

    h, w = roi_defs[0]["mask"].shape
    mask_indices = [np.flatnonzero(np.asarray(d["mask"]).reshape(-1)) for d in roi_defs]
    traces_by_frame: list[list[float]] = []
    pages_total = 0
    frames_used = 0
    pages_skipped = 0

    with tiff.TiffFile(str(tif_path)) as tf:
        if len(tf.pages) == 1:
            stack = _normalize_stack_to_thw(tf.asarray(), (h, w))
            pages_total = int(stack.shape[0])
            for frame in stack:
                flat = frame.reshape(-1).astype(np.float32, copy=False)
                traces_by_frame.append([float(np.mean(flat[idx])) for idx in mask_indices])
            frames_used = int(stack.shape[0])
        else:
            for pg in tf.pages:
                pages_total += 1
                frame = np.squeeze(pg.asarray())
                if frame.shape != (h, w):
                    pages_skipped += 1
                    continue
                flat = frame.reshape(-1).astype(np.float32, copy=False)
                traces_by_frame.append([float(np.mean(flat[idx])) for idx in mask_indices])
                frames_used += 1

    if not traces_by_frame:
        raise RuntimeError(f"No valid frames found for paired trace extraction: {tif_path}")

    trace_arr = np.asarray(traces_by_frame, dtype=np.float32).T
    return trace_arr, {
        "pages_total": int(pages_total),
        "frames_used": int(frames_used),
        "pages_skipped": int(pages_skipped),
    }


def _write_trace_matrix_csv(path: Path, traces: np.ndarray, roi_defs: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["frame_idx"] + [f"rank_{int(d['rank']):04d}_rid_{int(d['rid']):04d}" for d in roi_defs]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        n_frames = int(traces.shape[1]) if traces.ndim == 2 else 0
        for frame_idx in range(n_frames):
            row = [frame_idx] + [float(traces[i, frame_idx]) for i in range(traces.shape[0])]
            writer.writerow(row)


def _write_roi_table_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_trace_for_plot(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.size == 0:
        return trace.astype(np.float32)
    centered = trace - float(np.percentile(trace, 10))
    scale = max(float(np.std(centered)), 1e-6)
    return (centered / scale).astype(np.float32)


def _summary_rowwise_max(traces: np.ndarray) -> dict[str, Any]:
    traces = np.asarray(traces)
    if traces.ndim != 2 or traces.shape[0] == 0 or traces.shape[1] == 0:
        return _summary_arr(np.array([]))
    return _summary_arr(np.max(traces, axis=1))


def _compute_trace_corr_matrix(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 2 or traces.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)

    n = int(traces.shape[0])
    corr = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(n):
        corr[i, i] = 1.0
        for j in range(i + 1, n):
            value = _safe_corrcoef(traces[i], traces[j])
            if value is None:
                continue
            corr[i, j] = float(value)
            corr[j, i] = float(value)
    return corr


def _bin_trace_matrix(traces: np.ndarray, target_bin_count: int = 1000) -> tuple[np.ndarray, int]:
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim != 2:
        raise ValueError(f"Expected trace matrix with shape (N, T), got {traces.shape}")
    if traces.shape[1] <= 0:
        return np.zeros((traces.shape[0], 0), dtype=np.float32), 0

    bin_count = min(max(1, int(target_bin_count)), int(traces.shape[1]))
    frame_bins = np.array_split(np.arange(int(traces.shape[1]), dtype=np.int32), bin_count)
    binned = np.zeros((traces.shape[0], bin_count), dtype=np.float32)
    for idx, frame_idx in enumerate(frame_bins):
        binned[:, idx] = np.mean(traces[:, frame_idx], axis=1, dtype=np.float64).astype(np.float32)
    return binned, int(bin_count)


def _zscore_trace_rows(traces: np.ndarray) -> np.ndarray:
    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError(f"Expected trace matrix with shape (N, T), got {traces.shape}")
    out = np.zeros(traces.shape, dtype=np.float32)
    for i in range(traces.shape[0]):
        row = traces[i]
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        row_valid = row[finite]
        mean = float(np.mean(row_valid))
        std = float(np.std(row_valid))
        if std <= 1e-12:
            continue
        out[i, finite] = ((row_valid - mean) / std).astype(np.float32)
    return out


def _heatmap_tick_positions(count: int) -> tuple[np.ndarray, list[int]]:
    if count <= 0:
        return np.asarray([], dtype=np.int32), []
    ticks = np.arange(0, count, dtype=np.int32)
    return ticks, [int(x) for x in ticks]


def _heatmap_tick_fontsize(count: int) -> float:
    if count <= 20:
        return 8.0
    if count <= 40:
        return 7.0
    if count <= 80:
        return 6.0
    if count <= 140:
        return 5.0
    return 4.0


def _save_trace_corr_heatmap_png(
    corr_matrix: np.ndarray,
    roi_defs: list[dict[str, Any]],
    out_png: Path,
    title: str,
) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if len(roi_defs) == 0 or corr_matrix.size == 0:
        fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=140)
        ax.text(0.5, 0.5, "No ROI correlation heatmap available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(out_png), dpi=140)
        plt.close(fig)
        return str(out_png)

    n = len(roi_defs)
    size = float(np.clip(0.12 * n + 6.0, 6.2, 18.0))
    tick_fontsize = _heatmap_tick_fontsize(n)
    fig, ax = plt.subplots(figsize=(size, size), dpi=140)
    cmap = plt.get_cmap("bwr")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    if hasattr(cmap, "set_bad"):
        cmap.set_bad("#cfcfcf")
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest", origin="upper")
    rank_labels = [int(roi["rank"]) for roi in roi_defs]
    ticks, tick_idx = _heatmap_tick_positions(n)
    if len(ticks) > 0:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(rank_labels[i]) for i in tick_idx], rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels([str(rank_labels[i]) for i in tick_idx], fontsize=tick_fontsize)
    ax.set_xlabel("Selected ROI (final rank)")
    ax.set_ylabel("Selected ROI (final rank)")
    ax.set_title(title)
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)
    return str(out_png)


def _save_trace_temporal_heatmap_png(
    traces: np.ndarray,
    roi_defs: list[dict[str, Any]],
    out_png: Path,
    title: str,
    target_bin_count: int = 1000,
) -> tuple[str, int]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if len(roi_defs) == 0 or np.asarray(traces).size == 0:
        fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=140)
        ax.text(0.5, 0.5, "No temporal trace heatmap available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(out_png), dpi=140)
        plt.close(fig)
        return str(out_png), 0

    binned, bin_count = _bin_trace_matrix(traces, target_bin_count=target_bin_count)
    zscore = _zscore_trace_rows(binned)
    scale = min(0.12, 20.0 / max(bin_count, 1), 14.0 / max(len(roi_defs), 1))
    fig_w = max(7.5, float(bin_count) * scale + 1.6)
    fig_h = max(4.5, float(len(roi_defs)) * scale + 1.6)
    tick_fontsize = _heatmap_tick_fontsize(len(roi_defs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    cmap = plt.get_cmap("RdBu_r")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    if hasattr(cmap, "set_bad"):
        cmap.set_bad("#cfcfcf")
    im = ax.imshow(zscore, cmap=cmap, vmin=-3.0, vmax=3.0, aspect="equal", interpolation="nearest", origin="upper")
    rank_labels = [int(roi["rank"]) for roi in roi_defs]
    ticks, tick_idx = _heatmap_tick_positions(len(roi_defs))
    if len(ticks) > 0:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(rank_labels[i]) for i in tick_idx], fontsize=tick_fontsize)
    ax.set_xlabel(f"Time bin (count={bin_count})")
    ax.set_ylabel("Selected ROI (final rank)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Per-cell Z-score")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)
    return str(out_png), int(bin_count)


def _save_paired_trace_plot(
    raw_traces: np.ndarray,
    final_traces: np.ndarray,
    roi_defs: list[dict[str, Any]],
    per_roi_corr: list[float | None],
    out_png: Path,
) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if len(roi_defs) == 0:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=140)
        ax.text(0.5, 0.5, "No paired trace ROI available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(out_png), dpi=140)
        plt.close(fig)
        return str(out_png)

    n = len(roi_defs)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5 * n, 3.5)), dpi=140, sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        roi = roi_defs[i]
        raw_norm = _normalize_trace_for_plot(raw_traces[i])
        final_norm = _normalize_trace_for_plot(final_traces[i])
        corr_txt = "NA" if per_roi_corr[i] is None else f"{per_roi_corr[i]:.3f}"
        ax.plot(raw_norm, color="#d55e00", linewidth=0.9, alpha=0.9, label="raw")
        ax.plot(final_norm, color="#0072b2", linewidth=0.9, alpha=0.9, label="final")
        ax.set_title(
            f"rank={int(roi['rank'])} rid={int(roi['rid'])} px={int(roi['pixel_count'])} corr={corr_txt}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Paired Trace Curves (final ROI anchor, per-ROI normalized)", fontsize=12)
    fig.supxlabel("Frame")
    fig.supylabel("Normalized trace")
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=140)
    plt.close(fig)
    return str(out_png)


def _crop_mask_with_pad(mask: np.ndarray, pad: int = 8) -> np.ndarray:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    return mask[y0:y1, x0:x1].astype(np.uint8)


def _save_paired_mask_artifacts(
    roi_defs: list[dict[str, Any]],
    mask_png_path: Path,
    labelmask_tif_path: Path,
) -> dict[str, str]:
    mask_png_path.parent.mkdir(parents=True, exist_ok=True)
    labelmask_tif_path.parent.mkdir(parents=True, exist_ok=True)

    if not roi_defs:
        blank = np.zeros((1, 1), dtype=np.uint16)
        tiff.imwrite(str(labelmask_tif_path), blank, photometric="minisblack", metadata=None)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=140)
        ax.text(0.5, 0.5, "No paired ROI mask available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(mask_png_path), dpi=140)
        plt.close(fig)
        return {
            "paired_mask_png": str(mask_png_path),
            "paired_labelmask_tif": str(labelmask_tif_path),
        }

    h, w = roi_defs[0]["mask"].shape
    labelmask = np.zeros((h, w), dtype=np.uint16)
    for roi in roi_defs:
        labelmask[np.asarray(roi["mask"], dtype=bool)] = np.uint16(int(roi["rank"]))
    tiff.imwrite(str(labelmask_tif_path), labelmask, photometric="minisblack", metadata=None)

    n = len(roi_defs)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), dpi=140)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    flat_axes = list(axes_arr.flatten())

    for ax, roi in zip(flat_axes, roi_defs):
        crop = _crop_mask_with_pad(np.asarray(roi["mask"], dtype=bool), pad=8)
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"rank={int(roi['rank'])} rid={int(roi['rid'])}\npx={int(roi['pixel_count'])}",
            fontsize=9,
        )
        ax.set_axis_off()

    for ax in flat_axes[len(roi_defs):]:
        ax.set_axis_off()

    fig.suptitle("Paired Trace ROI Masks (final ROI anchor)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(mask_png_path), dpi=140)
    plt.close(fig)
    return {
        "paired_mask_png": str(mask_png_path),
        "paired_labelmask_tif": str(labelmask_tif_path),
    }


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return None
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _run_paired_trace_extraction(
    raw_stack_path: str | Path,
    final_stack_path: str | Path,
    final_sel: dict,
    output_dir: str | Path,
    top_n_final_rois: int = 5,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    paired_dir = output_dir / "paired_trace"
    paired_dir.mkdir(parents=True, exist_ok=True)

    artifacts = final_sel.get("artifacts", {})
    all_roi_seed_rows = _load_final_roi_rows(artifacts["roi_summary_csv"], top_n=None, subset="analysis")
    all_roi_rows = _merge_roi_rows_with_labelmask_ranks(artifacts["analysis_mask_uint16_tif"], all_roi_seed_rows)
    all_roi_defs = _load_rank_masks_from_labelmask(artifacts["analysis_mask_uint16_tif"], all_roi_rows)
    raw_traces_all, raw_meta = _extract_mean_traces_from_tif(raw_stack_path, all_roi_defs)
    final_traces_all, final_meta = _extract_mean_traces_from_tif(final_stack_path, all_roi_defs)

    display_roi_rows = _load_final_roi_rows(artifacts["roi_summary_csv"], top_n=int(top_n_final_rois), subset="display")
    roi_defs_display = _load_rank_masks_from_labelmask(artifacts["display_labelmask_uint16_tif"], display_roi_rows)
    all_rank_to_idx = {int(roi["rank"]): idx for idx, roi in enumerate(all_roi_defs)}
    selected_pairs = [
        (roi, all_rank_to_idx[int(roi["rank"])])
        for roi in roi_defs_display
        if int(roi["rank"]) in all_rank_to_idx
    ]
    roi_defs = [roi for roi, _ in selected_pairs]
    selected_indices = [idx for _, idx in selected_pairs]
    raw_traces = np.asarray(raw_traces_all[selected_indices], dtype=np.float32)
    final_traces = np.asarray(final_traces_all[selected_indices], dtype=np.float32)

    raw_npy = paired_dir / "raw_traces_final_roi_anchor.npy"
    final_npy = paired_dir / "final_traces_final_roi_anchor.npy"
    raw_csv = paired_dir / "raw_traces_final_roi_anchor.csv"
    final_csv = paired_dir / "final_traces_final_roi_anchor.csv"
    roi_csv = paired_dir / "paired_trace_roi_table.csv"
    trace_plot_png = paired_dir / "paired_trace_curves.png"
    mask_png = paired_dir / "paired_trace_cell_masks.png"
    labelmask_tif = paired_dir / "paired_trace_anchor_labelmask_uint16.tif"
    all_roi_csv = paired_dir / "all_selected_roi_table.csv"
    all_raw_npy = paired_dir / "all_selected_raw_traces_final_roi_anchor.npy"
    all_final_npy = paired_dir / "all_selected_final_traces_final_roi_anchor.npy"
    all_raw_csv = paired_dir / "all_selected_raw_traces_final_roi_anchor.csv"
    all_final_csv = paired_dir / "all_selected_final_traces_final_roi_anchor.csv"
    raw_corr_png = paired_dir / "all_selected_raw_trace_correlation_heatmap.png"
    final_corr_png = paired_dir / "all_selected_final_trace_correlation_heatmap.png"
    raw_temporal_png = paired_dir / "all_selected_raw_trace_temporal_heatmap.png"
    final_temporal_png = paired_dir / "all_selected_final_trace_temporal_heatmap.png"
    summary_json = paired_dir / "paired_trace_summary.json"

    np.save(raw_npy, raw_traces)
    np.save(final_npy, final_traces)
    _write_trace_matrix_csv(raw_csv, raw_traces, roi_defs)
    _write_trace_matrix_csv(final_csv, final_traces, roi_defs)
    np.save(all_raw_npy, raw_traces_all)
    np.save(all_final_npy, final_traces_all)
    _write_trace_matrix_csv(all_raw_csv, raw_traces_all, all_roi_defs)
    _write_trace_matrix_csv(all_final_csv, final_traces_all, all_roi_defs)

    common_frames = min(int(raw_traces.shape[1]), int(final_traces.shape[1]))
    delta = final_traces[:, :common_frames] - raw_traces[:, :common_frames]
    per_roi_corr = [
        _safe_corrcoef(raw_traces[i, :common_frames], final_traces[i, :common_frames])
        for i in range(len(roi_defs))
    ]
    corr_vals = np.asarray([v for v in per_roi_corr if v is not None], dtype=np.float32)
    trace_plot_path = _save_paired_trace_plot(raw_traces, final_traces, roi_defs, per_roi_corr, trace_plot_png)
    mask_artifacts = _save_paired_mask_artifacts(roi_defs, mask_png, labelmask_tif)

    raw_corr_matrix_all = _compute_trace_corr_matrix(raw_traces_all)
    final_corr_matrix_all = _compute_trace_corr_matrix(final_traces_all)
    raw_corr_heatmap_path = _save_trace_corr_heatmap_png(
        raw_corr_matrix_all,
        all_roi_defs,
        raw_corr_png,
        "Raw ROI Correlation Heatmap (final analysis ROI order)",
    )
    final_corr_heatmap_path = _save_trace_corr_heatmap_png(
        final_corr_matrix_all,
        all_roi_defs,
        final_corr_png,
        "Final ROI Correlation Heatmap (final analysis ROI order)",
    )
    raw_temporal_heatmap_path, raw_temporal_bins = _save_trace_temporal_heatmap_png(
        raw_traces_all,
        all_roi_defs,
        raw_temporal_png,
        "Raw ROI Temporal Heatmap (per-cell Z-score)",
    )
    final_temporal_heatmap_path, final_temporal_bins = _save_trace_temporal_heatmap_png(
        final_traces_all,
        all_roi_defs,
        final_temporal_png,
        "Final ROI Temporal Heatmap (per-cell Z-score)",
    )

    roi_rows_out = []
    for i, roi in enumerate(roi_defs):
        roi_rows_out.append(
            {
                "rank": int(roi["rank"]),
                "rid": int(roi["rid"]),
                "pixel_count": int(roi["pixel_count"]),
                "largest_cc_area_raw": int(roi["largest_cc_area_raw"]),
                "final_area_polished": int(roi["final_area_polished"]),
                "trace_max_final_selection": float(roi["trace_max"]),
                "trace_mean_raw_stack": float(np.mean(raw_traces[i])),
                "trace_mean_final_stack": float(np.mean(final_traces[i])),
                "trace_max_raw_stack": float(np.max(raw_traces[i])),
                "trace_max_final_stack": float(np.max(final_traces[i])),
                "trace_corrcoef_raw_vs_final": per_roi_corr[i],
            }
        )

    fieldnames = [
        "rank",
        "rid",
        "pixel_count",
        "largest_cc_area_raw",
        "final_area_polished",
        "trace_max_final_selection",
        "trace_mean_raw_stack",
        "trace_mean_final_stack",
        "trace_max_raw_stack",
        "trace_max_final_stack",
        "trace_corrcoef_raw_vs_final",
    ]
    _write_roi_table_csv(roi_csv, roi_rows_out, fieldnames=fieldnames)

    all_roi_rows_out = []
    all_common_frames = min(int(raw_traces_all.shape[1]), int(final_traces_all.shape[1]))
    all_per_roi_corr = [
        _safe_corrcoef(raw_traces_all[i, :all_common_frames], final_traces_all[i, :all_common_frames])
        for i in range(len(all_roi_defs))
    ]
    for i, roi in enumerate(all_roi_defs):
        all_roi_rows_out.append(
            {
                "rank": int(roi["rank"]),
                "rid": int(roi["rid"]),
                "pixel_count": int(roi["pixel_count"]),
                "largest_cc_area_raw": int(roi["largest_cc_area_raw"]),
                "final_area_polished": int(roi["final_area_polished"]),
                "trace_max_final_selection": float(roi["trace_max"]),
                "trace_mean_raw_stack": float(np.mean(raw_traces_all[i])) if raw_traces_all.shape[1] > 0 else None,
                "trace_mean_final_stack": float(np.mean(final_traces_all[i])) if final_traces_all.shape[1] > 0 else None,
                "trace_max_raw_stack": float(np.max(raw_traces_all[i])) if raw_traces_all.shape[1] > 0 else None,
                "trace_max_final_stack": float(np.max(final_traces_all[i])) if final_traces_all.shape[1] > 0 else None,
                "trace_corrcoef_raw_vs_final": all_per_roi_corr[i],
            }
        )
    _write_roi_table_csv(all_roi_csv, all_roi_rows_out, fieldnames=fieldnames)

    summary = {
        "execution_status": "executed",
        "anchor_source": "final_display_subset",
        "selected_count": int(len(roi_defs)),
        "top_n_final_rois": int(top_n_final_rois),
        "roi_ranks": [int(r["rank"]) for r in roi_defs],
        "roi_ids": [int(r["rid"]) for r in roi_defs],
        "all_selected_count": int(len(all_roi_defs)),
        "all_selected_roi_ranks": [int(r["rank"]) for r in all_roi_defs],
        "all_selected_roi_ids": [int(r["rid"]) for r in all_roi_defs],
        "all_analysis_count": int(len(all_roi_defs)),
        "all_analysis_roi_ranks": [int(r["rank"]) for r in all_roi_defs],
        "all_analysis_roi_ids": [int(r["rid"]) for r in all_roi_defs],
        "raw_trace_shape": list(raw_traces.shape),
        "final_trace_shape": list(final_traces.shape),
        "all_selected_raw_trace_shape": list(raw_traces_all.shape),
        "all_selected_final_trace_shape": list(final_traces_all.shape),
        "all_analysis_raw_trace_shape": list(raw_traces_all.shape),
        "all_analysis_final_trace_shape": list(final_traces_all.shape),
        "common_frames_compared": int(common_frames),
        "raw_stack_read_meta": raw_meta,
        "final_stack_read_meta": final_meta,
        "raw_trace_summary": _summary_arr(raw_traces.flatten()),
        "final_trace_summary": _summary_arr(final_traces.flatten()),
        "raw_trace_max_summary": _summary_rowwise_max(raw_traces),
        "final_trace_max_summary": _summary_rowwise_max(final_traces),
        "delta_final_minus_raw_summary": _summary_arr(delta.flatten()),
        "corrcoef_summary": _summary_arr(corr_vals),
        "heatmap_config": {
            "roi_order": "final_analysis_rank_ascending",
            "roi_scope": "final_analysis_set",
            "temporal_bin_target": 1000,
            "temporal_bin_count_used": int(final_temporal_bins if final_temporal_bins == raw_temporal_bins else min(raw_temporal_bins, final_temporal_bins)),
            "temporal_bin_count_used_by_stack": {
                "raw": int(raw_temporal_bins),
                "final": int(final_temporal_bins),
            },
            "temporal_normalization": "per_cell_zscore_after_binning",
            "corr_metric": "pearson",
        },
        "artifacts": {
            "roi_table_csv": str(roi_csv),
            "raw_trace_npy": str(raw_npy),
            "final_trace_npy": str(final_npy),
            "raw_trace_csv": str(raw_csv),
            "final_trace_csv": str(final_csv),
            "paired_trace_plot_png": str(trace_plot_path),
            "paired_mask_png": mask_artifacts["paired_mask_png"],
            "paired_labelmask_tif": mask_artifacts["paired_labelmask_tif"],
            "all_selected_roi_table_csv": str(all_roi_csv),
            "all_selected_raw_trace_npy": str(all_raw_npy),
            "all_selected_final_trace_npy": str(all_final_npy),
            "all_selected_raw_trace_csv": str(all_raw_csv),
            "all_selected_final_trace_csv": str(all_final_csv),
            "raw_trace_corr_heatmap_png": str(raw_corr_heatmap_path),
            "final_trace_corr_heatmap_png": str(final_corr_heatmap_path),
            "raw_trace_temporal_heatmap_png": str(raw_temporal_heatmap_path),
            "final_trace_temporal_heatmap_png": str(final_temporal_heatmap_path),
            "summary_json": str(summary_json),
        },
    }
    _write_json(summary_json, summary)
    return summary


def run_downstream_analysis(
    raw_stack_path: str | Path,
    final_stack_path: str | Path,
    output_dir: str | Path,
    dataset_profile: str,
    config: dict | None = None,
) -> dict:
    cfg = {
        "run_raw": True,
        "source_files_used": ["STEP0_seg.py", "STEP1_display_v3.py"],
        "backend_name": "suite2p_step0_step1_adapter",
        "backend_version_or_source": "local_repo",
        "segmentation_config": {},
        "selection_config": {
            "analysis_use_top_percent": False,
            "asset_prefix": "final",
        },
        "paired_trace_config": {
            "enabled": True,
            "top_n_final_rois": 5,
        },
    }
    if config:
        cfg.update(config)

    raw_stack_path = Path(raw_stack_path).resolve()
    final_stack_path = Path(final_stack_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_status_path = output_dir / "backend_status.json"
    run_status_path = output_dir / "run_status.json"
    summary_path = output_dir / "summary.json"
    cmp_path = output_dir / "downstream_comparison.json"

    backend_status = {
        "backend_name": cfg["backend_name"],
        "backend_version_or_source": cfg["backend_version_or_source"],
        "source_files_used": cfg["source_files_used"],
        "import_mode": "direct_python_import",
        "dataset_profile": str(dataset_profile),
        "available": False,
        "status": "placeholder",
    }

    if str(dataset_profile).lower() != "neuronal":
        backend_status["status"] = "placeholder"
        backend_status["available"] = False
        _write_json(backend_status_path, backend_status)
        run_status = {
            "dataset_profile": dataset_profile,
            "execution_status": "skipped",
            "reason": "dataset_profile_not_neuronal",
            "raw_stack_path": str(raw_stack_path),
            "final_stack_path": str(final_stack_path),
            "final_stack_source": "final/final_stack.tif",
            "suite2p_registration_used": None,
            "config_snapshot_path": str(output_dir / "downstream_config_snapshot.json"),
        }
        _write_json(output_dir / "downstream_config_snapshot.json", cfg)
        _write_json(run_status_path, run_status)
        _write_json(
            summary_path,
            {
                "dataset_profile": dataset_profile,
                "backend_name": cfg["backend_name"],
                "raw_executed": False,
                "final_executed": False,
                "suite2p_counts": {},
                "notes": ["Downstream skipped for non-neuronal profile."],
                "unavailable_fields": ["suite2p outputs", "roi selection", "trace parse"],
            },
        )
        _write_json(
            cmp_path,
            {
                "dataset_profile": dataset_profile,
                "backend_name": cfg["backend_name"],
                "executed_on_raw": False,
                "executed_on_final": False,
                "notes": ["Skipped due to dataset profile."],
                "unavailable_fields": ["roi_count_raw", "roi_count_final", "trace summaries"],
            },
        )
        return {
            "backend_status_path": str(backend_status_path),
            "run_status_path": str(run_status_path),
            "summary_path": str(summary_path),
            "comparison_path": str(cmp_path),
        }

    try:
        from STEP0_seg import run_suite2p_segmentation
        from STEP1_display_v3 import select_rois_and_build_artifacts
        backend_status["available"] = True
        backend_status["status"] = "available"
    except Exception as exc:
        backend_status["available"] = False
        backend_status["status"] = "placeholder"
        backend_status["reason"] = f"import_failed: {type(exc).__name__}: {exc}"
        _write_json(backend_status_path, backend_status)
        _write_json(output_dir / "downstream_config_snapshot.json", cfg)
        _write_json(
            run_status_path,
            {
                "dataset_profile": dataset_profile,
                "execution_status": "placeholder",
                "reason": backend_status["reason"],
                "raw_stack_path": str(raw_stack_path),
                "final_stack_path": str(final_stack_path),
                "final_stack_source": "final/final_stack.tif",
                "suite2p_registration_used": None,
                "config_snapshot_path": str(output_dir / "downstream_config_snapshot.json"),
            },
        )
        _write_json(
            summary_path,
            {
                "dataset_profile": dataset_profile,
                "backend_name": cfg["backend_name"],
                "raw_executed": False,
                "final_executed": False,
                "suite2p_counts": {},
                "notes": [backend_status["reason"]],
                "unavailable_fields": ["suite2p", "selection", "trace parse"],
            },
        )
        _write_json(
            cmp_path,
            {
                "dataset_profile": dataset_profile,
                "backend_name": cfg["backend_name"],
                "executed_on_raw": False,
                "executed_on_final": False,
                "notes": [backend_status["reason"]],
                "unavailable_fields": ["roi counts", "trace summaries"],
            },
        )
        return {
            "backend_status_path": str(backend_status_path),
            "run_status_path": str(run_status_path),
            "summary_path": str(summary_path),
            "comparison_path": str(cmp_path),
        }

    _write_json(backend_status_path, backend_status)
    _write_json(output_dir / "downstream_config_snapshot.json", cfg)

    raw_seg = None
    raw_sel = None
    final_seg = None
    final_sel = None

    final_root = output_dir / "final"
    raw_root = output_dir / "raw"
    final_seg = run_suite2p_segmentation(final_stack_path, final_root, cfg.get("segmentation_config", {}))
    sel_cfg_final = dict(cfg.get("selection_config", {}))
    sel_cfg_final["asset_prefix"] = "final"
    final_sel = select_rois_and_build_artifacts(
        plane0_dir=Path(final_seg["artifacts"]["plane0_intensityfilt_dir"]),
        output_root=final_root / "roi_selection",
        selection_config=sel_cfg_final,
    )

    raw_executed = False
    if bool(cfg.get("run_raw", True)):
        raw_seg = run_suite2p_segmentation(raw_stack_path, raw_root, cfg.get("segmentation_config", {}))
        sel_cfg_raw = dict(cfg.get("selection_config", {}))
        sel_cfg_raw["asset_prefix"] = "raw"
        raw_sel = select_rois_and_build_artifacts(
            plane0_dir=Path(raw_seg["artifacts"]["plane0_intensityfilt_dir"]),
            output_root=raw_root / "roi_selection",
            selection_config=sel_cfg_raw,
        )
        raw_executed = True

    run_status = {
        "dataset_profile": dataset_profile,
        "execution_status": "executed",
        "reason": "downstream_executed_neuronal_profile",
        "raw_stack_path": str(raw_stack_path),
        "final_stack_path": str(final_stack_path),
        "final_stack_source": "final/final_stack.tif",
        "suite2p_registration_used": bool(final_seg.get("suite2p_registration_used")),
        "config_snapshot_path": str(output_dir / "downstream_config_snapshot.json"),
    }
    _write_json(run_status_path, run_status)

    raw_counts = _extract_counts(raw_seg, raw_sel)
    final_counts = _extract_counts(final_seg, final_sel)
    paired_trace_cfg = dict(cfg.get("paired_trace_config", {}) or {})
    paired_trace_result = None
    if bool(paired_trace_cfg.get("enabled", True)):
        paired_trace_result = _run_paired_trace_extraction(
            raw_stack_path=raw_stack_path,
            final_stack_path=final_stack_path,
            final_sel=final_sel,
            output_dir=output_dir,
            top_n_final_rois=int(paired_trace_cfg.get("top_n_final_rois", 5)),
        )

    summary = {
        "dataset_profile": dataset_profile,
        "backend_name": cfg["backend_name"],
        "raw_executed": raw_executed,
        "final_executed": True,
        "suite2p_counts": {
            "raw_plane0_total": raw_counts["plane0_total"] if raw_executed else None,
            "raw_after_intensity_filter": raw_counts["after_intensity_filter"] if raw_executed else None,
            "raw_after_cc_area_filter": raw_counts["after_cc_area_filter"] if raw_executed else None,
            "raw_display_selected_count": raw_counts["display_selected_count"] if raw_executed else None,
            "final_plane0_total": final_counts["plane0_total"],
            "final_after_intensity_filter": final_counts["after_intensity_filter"],
            "final_after_cc_area_filter": final_counts["after_cc_area_filter"],
            "final_display_selected_count": final_counts["display_selected_count"],
        },
        "selection_thresholds_used": {
            "intensity_thr_max": (cfg.get("segmentation_config", {}) or {}).get("int_thr_max"),
            "intensity_thr_std": (cfg.get("segmentation_config", {}) or {}).get("int_thr_std"),
            "min_largest_cc_area": (cfg.get("selection_config", {}) or {}).get("min_largest_cc_area"),
            "top_percent": (cfg.get("selection_config", {}) or {}).get("top_percent"),
            "polish_selected_masks": (cfg.get("selection_config", {}) or {}).get("polish_selected_masks"),
            "erode_pixels": (cfg.get("selection_config", {}) or {}).get("erode_pixels"),
        },
        "paired_trace": paired_trace_result,
        "notes": [],
        "unavailable_fields": [],
    }
    _write_json(summary_path, summary)

    raw_trace = _extract_trace_summary(raw_sel)
    final_trace = _extract_trace_summary(final_sel)
    cmp = {
        "dataset_profile": dataset_profile,
        "backend_name": cfg["backend_name"],
        "executed_on_raw": raw_executed,
        "executed_on_final": True,
        "roi_count_raw": raw_counts["plane0_total"] if raw_executed else None,
        "roi_count_final": final_counts["plane0_total"],
        "accepted_roi_count_raw": raw_counts["after_cc_area_filter"] if raw_executed else None,
        "accepted_roi_count_final": final_counts["after_cc_area_filter"],
        "display_selected_count_raw": raw_counts["display_selected_count"] if raw_executed else None,
        "display_selected_count_final": final_counts["display_selected_count"],
        "trace_count_raw": raw_trace["trace_count"] if raw_executed else None,
        "trace_count_final": final_trace["trace_count"],
        "trace_max_summary_raw": raw_trace["trace_max_summary"] if raw_executed else None,
        "trace_max_summary_final": final_trace["trace_max_summary"],
        "F_summary_raw": raw_trace["F_summary"] if raw_executed else None,
        "F_summary_final": final_trace["F_summary"],
        "Fneu_summary_raw": raw_trace["Fneu_summary"] if raw_executed else None,
        "Fneu_summary_final": final_trace["Fneu_summary"],
        "spks_summary_raw": raw_trace["spks_summary"] if raw_executed else None,
        "spks_summary_final": final_trace["spks_summary"],
        "paired_trace": paired_trace_result,
        "artifact_paths": {
            "raw": raw_sel.get("artifacts", {}) if raw_executed and raw_sel else {},
            "final": final_sel.get("artifacts", {}) if final_sel else {},
            "paired_trace": (paired_trace_result or {}).get("artifacts", {}),
        },
        "notes": [],
        "unavailable_fields": [],
    }
    _write_json(cmp_path, cmp)

    return {
        "backend_status_path": str(backend_status_path),
        "run_status_path": str(run_status_path),
        "summary_path": str(summary_path),
        "comparison_path": str(cmp_path),
        "paired_trace": paired_trace_result,
        "raw": {"segmentation": raw_seg, "selection": raw_sel},
        "final": {"segmentation": final_seg, "selection": final_sel},
    }


def materialize_final_and_run_downstream(
    raw_stack_path: str | Path,
    final_stack_source_path: str | Path,
    output_root: str | Path,
    dataset_profile: str,
    downstream_config: dict | None,
    final_source_semantic: str = "last_iter_denoised_output",
) -> dict:
    output_root = Path(output_root).resolve()
    final_dir = output_root / "final"
    segmentation_dir = output_root / "segmentation"
    downstream_config = dict(downstream_config or {})
    final_stack_path, sidecar_path = _materialize_final_stack(
        final_stack_source=Path(final_stack_source_path).resolve(),
        final_dir=final_dir,
        source_semantic=final_source_semantic,
    )
    runner_cfg = dict(downstream_config.get("runner_config", {}) or {})
    runner_mode = str(runner_cfg.get("mode", "direct")).lower()
    dataset_profile_norm = str(dataset_profile or "").strip().lower()
    if dataset_profile_norm != "neuronal":
        # Non-cell datasets should not require a separate downstream env just to
        # emit the explicit placeholder artifacts expected by the report layer.
        downstream_out = run_downstream_analysis(
            raw_stack_path=raw_stack_path,
            final_stack_path=final_stack_path,
            output_dir=segmentation_dir,
            dataset_profile=dataset_profile,
            config=downstream_config,
        )
    elif runner_mode == "external_python":
        runner_python = _resolve_external_python(runner_cfg)
        runner_log_path = segmentation_dir / "downstream_subprocess.log"
        runner_cfg_path = segmentation_dir / "downstream_subprocess_config.json"
        runner_result_path = segmentation_dir / "downstream_subprocess_result.json"
        _write_json(runner_cfg_path, downstream_config)
        cmd = [
            str(runner_python),
            str(Path(__file__).resolve()),
            "--raw-stack-path",
            str(Path(raw_stack_path).resolve()),
            "--final-stack-path",
            str(final_stack_path),
            "--output-dir",
            str(segmentation_dir),
            "--dataset-profile",
            str(dataset_profile),
            "--config-json",
            str(runner_cfg_path),
            "--result-json-path",
            str(runner_result_path),
        ]
        segmentation_dir.mkdir(parents=True, exist_ok=True)
        with open(runner_log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                cwd=str(Path(__file__).resolve().parent),
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"External downstream subprocess failed with code {proc.returncode}. "
                f"See log: {runner_log_path}"
            )
        if not runner_result_path.exists():
            raise RuntimeError(
                f"External downstream subprocess completed but result json was not created: {runner_result_path}"
            )
        with open(runner_result_path, "r", encoding="utf-8") as f:
            downstream_out = json.load(f)
        downstream_out["downstream_subprocess_log_path"] = str(runner_log_path)
        downstream_out["downstream_subprocess_result_path"] = str(runner_result_path)
        downstream_out["downstream_subprocess_python"] = str(runner_python)
    else:
        downstream_out = run_downstream_analysis(
            raw_stack_path=raw_stack_path,
            final_stack_path=final_stack_path,
            output_dir=segmentation_dir,
            dataset_profile=dataset_profile,
            config=downstream_config,
        )
    return {
        "final_stack_path": str(final_stack_path),
        "final_stack_sidecar_path": str(sidecar_path),
        "segmentation_output_dir": str(segmentation_dir),
        **downstream_out,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run downstream segmentation + ROI selection pipeline as a standalone script."
    )
    parser.add_argument("--raw-stack-path", required=True, help="Path to raw stack TIFF")
    parser.add_argument("--final-stack-path", required=True, help="Path to final stack TIFF")
    parser.add_argument("--output-dir", required=True, help="Output directory for downstream artifacts")
    parser.add_argument(
        "--dataset-profile",
        default="neuronal",
        help='Dataset profile (default: "neuronal")',
    )
    parser.add_argument(
        "--config-json",
        default=None,
        help="Optional path to downstream config JSON",
    )
    parser.add_argument(
        "--result-json-path",
        default=None,
        help="Optional path to write final result JSON for subprocess integration",
    )
    return parser.parse_args()


def _load_config(config_json: str | None) -> dict | None:
    if not config_json:
        return None
    cfg_path = Path(config_json).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    args = _parse_args()
    cfg = _load_config(args.config_json)
    result = run_downstream_analysis(
        raw_stack_path=args.raw_stack_path,
        final_stack_path=args.final_stack_path,
        output_dir=args.output_dir,
        dataset_profile=args.dataset_profile,
        config=cfg,
    )
    if args.result_json_path:
        _write_json(Path(args.result_json_path).resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
