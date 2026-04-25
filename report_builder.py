#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import shutil
import string
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import tifffile as tiff

RAW_COLOR = "#2f2f2f"
DENOISE_COLOR = "#d57d2a"
MOTION_COLOR = "#1f9d98"
FINAL_COLOR = "#2e8b57"
BORDER_COLOR = "#d9d9d9"
TEXT_MUTED = "#5f5f5f"
DISPLAY_FINAL_NAME = "NeuroPilot"
REPORT_DISPLAY_TOP_PERCENT = 0.60
REPORT_CROP_SCALE_FACTOR = 0.55
DEFAULT_IMAGING_MODALITY = "2p"
DEFAULT_PIXEL_SIZE_UM = 0.645
DEFAULT_FPS_HZ = 10.0
REPORT_TITLE = "NeuroPilot Processing Analysis Report"
REPORT_KYMOGRAPH_LINE_COUNT = 2
READER_LABELS = {
    "raw": "Raw",
    "denoise": "Denoised",
    "motion": "Motion-corrected",
    "final": DISPLAY_FINAL_NAME,
}

CONDITION_COLORS = {
    "raw": RAW_COLOR,
    "denoise": DENOISE_COLOR,
    "motion": MOTION_COLOR,
    "final": FINAL_COLOR,
}
MODALITY_CMAPS = {
    "2p": LinearSegmentedColormap.from_list(
        "rp_2p",
        [
            (0.00, "#000000"),
            (0.18, "#001408"),
            (0.42, "#003314"),
            (0.68, "#007a20"),
            (0.88, "#00d92f"),
            (1.00, "#5cff4a"),
        ],
    ),
    "1p": LinearSegmentedColormap.from_list("rp_1p", ["#000000", "#ffffff"]),
    "3p": plt.get_cmap("afmhot"),
}

def _available_report_font_stack() -> list[str]:
    preferred = ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]
    try:
        available = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception:
        available = set()
    chosen = [name for name in preferred if name in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
    return chosen


REPORT_FONT_FAMILY = _available_report_font_stack()
REPORT_FONT_PRIMARY = REPORT_FONT_FAMILY[0]
REPORT_FONT_CSS = ",".join(
    [f"'{name}'" if " " in name else name for name in REPORT_FONT_FAMILY] + ["sans-serif"]
)

plt.rcParams.update(
    {
        "font.family": [REPORT_FONT_PRIMARY],
        "font.sans-serif": REPORT_FONT_FAMILY,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path)


def _read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def _summary_array(x: np.ndarray | None) -> dict[str, Any]:
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


def _reader_label(key: str | None) -> str:
    return READER_LABELS.get(str(key or "").strip().lower(), str(key or "N/A"))


def _normalize_modality(modality: str | None) -> str:
    modality_key = str(modality or DEFAULT_IMAGING_MODALITY).strip().lower()
    return modality_key if modality_key in MODALITY_CMAPS else DEFAULT_IMAGING_MODALITY


def _colormap_label(modality: str | None) -> str:
    modality_key = _normalize_modality(modality)
    if modality_key == "2p":
        return "black-to-green"
    if modality_key == "3p":
        return "afmhot"
    return "grayscale"


def _coerce_bool_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off"}:
            return False
    return None


def _infer_is_cell_data(manifest: dict[str, Any], dataset_profile: str | None) -> bool | None:
    direct = _coerce_bool_flag(manifest.get("is_cell_data"))
    if direct is not None:
        return direct
    downstream = manifest.get("downstream", {}) if isinstance(manifest.get("downstream"), dict) else {}
    downstream_flag = _coerce_bool_flag(downstream.get("is_cell_data"))
    if downstream_flag is not None:
        return downstream_flag
    profile = str(dataset_profile or "").strip().lower()
    if profile == "neuronal":
        return True
    if profile:
        return False
    return None


def _yes_no_label(value: bool | None) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def _stringify_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return _fmt(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ": "))
        except Exception:
            return str(value)
    return str(value)


def _truncate_middle(text: Any, max_chars: int = 90) -> str:
    s = _stringify_value(text)
    if len(s) <= max_chars:
        return s
    head = max(18, int(max_chars * 0.55))
    tail = max(14, max_chars - head - 1)
    return f"{s[:head]}...{s[-tail:]}"


def _normalize_stack_to_thw(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] <= arr.shape[-1]:
            return arr
        return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 4:
        if arr.shape[2] >= 16 and arr.shape[3] >= 16:
            return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
        arr = np.transpose(arr, (2, 3, 0, 1))
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    raise ValueError(f"Unsupported TIFF shape: {arr.shape}")


def _primary_tiff_series(tf: tiff.TiffFile):
    if getattr(tf, "series", None):
        return tf.series[0]
    return None


def _primary_series_pages(series) -> Any:
    if series is None:
        return None
    try:
        pages = series.pages
    except Exception:
        return None
    if len(pages) <= 0:
        return None
    try:
        frame0 = np.asarray(np.squeeze(pages[0].asarray()))
    except Exception:
        return None
    if frame0.ndim != 2:
        return None
    return pages


def _stack_info(stack_path: str | Path) -> dict[str, Any]:
    stack_path = Path(stack_path).expanduser().resolve()
    with tiff.TiffFile(str(stack_path)) as tf:
        series = _primary_tiff_series(tf)
        pages = _primary_series_pages(series)
        if pages is not None:
            frame0 = np.asarray(np.squeeze(pages[0].asarray()))
            return {
                "num_frames": int(len(pages)),
                "height_px": int(frame0.shape[0]),
                "width_px": int(frame0.shape[1]),
                "dtype": str(frame0.dtype),
            }
        stack = _normalize_stack_to_thw(series.asarray() if series is not None else tf.asarray())
        return {
            "num_frames": int(stack.shape[0]),
            "height_px": int(stack.shape[1]),
            "width_px": int(stack.shape[2]),
            "dtype": str(stack.dtype),
        }


def _iter_frames(stack_path: str | Path):
    stack_path = Path(stack_path).expanduser().resolve()
    with tiff.TiffFile(str(stack_path)) as tf:
        series = _primary_tiff_series(tf)
        pages = _primary_series_pages(series)
        if pages is not None:
            for idx, page in enumerate(pages):
                yield idx, np.asarray(np.squeeze(page.asarray()))
            return
        stack = _normalize_stack_to_thw(series.asarray() if series is not None else tf.asarray())
        for idx in range(int(stack.shape[0])):
            yield idx, np.asarray(stack[idx])


def _read_frame(stack_path: str | Path, frame_index: int) -> np.ndarray:
    info = _stack_info(stack_path)
    frame_index = int(np.clip(int(frame_index), 0, max(info["num_frames"] - 1, 0)))
    stack_path = Path(stack_path).expanduser().resolve()
    with tiff.TiffFile(str(stack_path)) as tf:
        series = _primary_tiff_series(tf)
        pages = _primary_series_pages(series)
        if pages is not None:
            return np.asarray(np.squeeze(pages[frame_index].asarray()))
        stack = _normalize_stack_to_thw(series.asarray() if series is not None else tf.asarray())
        return np.asarray(stack[frame_index])


def _read_2d_tif(path: str | Path) -> np.ndarray:
    arr = np.asarray(tiff.imread(str(Path(path).expanduser().resolve())))
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got shape={arr.shape}")
    return arr


def _optional_existing_file(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    text = str(path_like).strip()
    if not text:
        return None
    try:
        path = Path(text).expanduser().resolve()
    except Exception:
        return None
    if path.is_file():
        return path
    return None


def _normalize_image(img: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    img_f = np.asarray(img, dtype=np.float32)
    finite = img_f[np.isfinite(img_f)]
    if finite.size == 0:
        return np.zeros_like(img_f, dtype=np.float32)
    lo = float(np.percentile(finite, lo_q))
    hi = float(np.percentile(finite, hi_q))
    if hi <= lo + 1e-8:
        return np.zeros_like(img_f, dtype=np.float32)
    return np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)


def _modality_display_image(img: np.ndarray, imaging_modality: str | None = None) -> np.ndarray:
    modality = _normalize_modality(imaging_modality)
    normalized = _normalize_image(img)
    if modality == "2p":
        return np.power(normalized, 0.88).astype(np.float32, copy=False)
    return normalized


def _modality_rgb(img: np.ndarray, imaging_modality: str | None = None) -> np.ndarray:
    modality = _normalize_modality(imaging_modality)
    cmap = MODALITY_CMAPS[modality]
    return cmap(_modality_display_image(img, modality))[..., :3]


def _choose_scalebar_px(shape_hw: Sequence[int]) -> int:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    target = max(16, int(round(min(h, w) * 0.16)))
    options = np.asarray([16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192], dtype=np.int32)
    return int(options[np.argmin(np.abs(options - target))])


def _scalebar_label(px_len: int, pixel_size_um: float | None) -> str:
    if pixel_size_um is None or pixel_size_um <= 0:
        return f"{int(px_len)} px"
    physical = float(px_len) * float(pixel_size_um)
    if physical >= 10:
        return f"{physical:.0f} um"
    if physical >= 1:
        return f"{physical:.1f} um"
    return f"{physical:.2f} um"


def _add_scalebar(ax, image_shape: Sequence[int], pixel_size_um: float | None = None):
    h, w = int(image_shape[0]), int(image_shape[1])
    px_len = _choose_scalebar_px((h, w))
    margin_x = max(6, int(round(w * 0.05)))
    margin_y = max(6, int(round(h * 0.06)))
    x0 = w - margin_x - px_len
    x1 = w - margin_x
    y = h - margin_y
    ax.plot([x0, x1], [y, y], color="white", linewidth=2.2, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2.0,
        y - max(8, h * 0.03),
        _scalebar_label(px_len, pixel_size_um),
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": (0, 0, 0, 0.38), "edgecolor": "none", "pad": 1.2},
    )


def _panel_label(index: int) -> str:
    alphabet = string.ascii_uppercase
    out = ""
    value = int(index)
    while True:
        out = alphabet[value % 26] + out
        value = value // 26 - 1
        if value < 0:
            break
    return out


def _save_figure(fig, out_path: str | Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return str(out_path)


def save_unavailable_panel(output_png: str | Path, title: str, message: str = "N/A") -> str:
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=180)
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=10, color=TEXT_MUTED)
    ax.set_axis_off()
    return _save_figure(fig, output_png)


def select_representative_frame(raw_stack_path: str | Path, roi_percentile: float = 90.0) -> dict[str, Any]:
    info = _stack_info(raw_stack_path)
    t = int(info["num_frames"])
    h = int(info["height_px"])
    w = int(info["width_px"])
    sum_img = np.zeros((h, w), dtype=np.float64)
    sumsq_img = np.zeros((h, w), dtype=np.float64)
    global_trace = np.zeros((t,), dtype=np.float64)

    for idx, frame in _iter_frames(raw_stack_path):
        frame_f = np.asarray(frame, dtype=np.float32)
        sum_img += frame_f
        sumsq_img += frame_f * frame_f
        global_trace[idx] = float(np.mean(frame_f))

    mean_img = sum_img / max(t, 1)
    var_img = np.maximum(sumsq_img / max(t, 1) - mean_img * mean_img, 0.0)
    std_map = np.sqrt(var_img, dtype=np.float64).astype(np.float32)

    threshold = float(np.percentile(std_map, roi_percentile))
    roi_mask = std_map > threshold
    if not np.any(roi_mask):
        roi_mask = std_map >= threshold

    roi_trace = np.full((t,), np.nan, dtype=np.float64)
    if np.any(roi_mask):
        for idx, frame in _iter_frames(raw_stack_path):
            roi_trace[idx] = float(np.mean(np.asarray(frame, dtype=np.float32)[roi_mask]))

    if np.all(np.isnan(roi_trace)):
        frame_index = int(np.argmax(global_trace)) if global_trace.size else 0
        rule = "fallback_global_mean_peak"
        peak_value = _safe_float(global_trace[frame_index]) if global_trace.size else None
    else:
        frame_index = int(np.nanargmax(roi_trace))
        rule = "std_roi_trace_peak"
        peak_value = _safe_float(roi_trace[frame_index])

    return {
        "frame_index": int(frame_index),
        "selection_rule": rule,
        "roi_percentile": float(roi_percentile),
        "roi_threshold_value": float(threshold),
        "roi_pixel_count": int(np.count_nonzero(roi_mask)),
        "roi_pixel_fraction": float(np.mean(roi_mask)),
        "peak_value": peak_value,
        "global_mean_peak_frame": int(np.argmax(global_trace)) if global_trace.size else 0,
        "global_mean_peak_value": _safe_float(np.max(global_trace)) if global_trace.size else None,
        "_std_map": std_map,
        "_roi_mask": roi_mask,
    }


def select_motion_burst_triplet(raw_shifts_npy: str | Path, num_frames: int, percentile: float = 95.0, base_delta: int = 5) -> dict[str, Any]:
    shifts = np.asarray(np.load(str(Path(raw_shifts_npy).expanduser().resolve())), dtype=np.float32)
    if shifts.ndim != 2 or shifts.shape[1] < 2:
        raise ValueError(f"Invalid rigid shifts shape for motion burst selection: {shifts.shape}")
    mag = np.sqrt(shifts[:, 0] ** 2 + shifts[:, 1] ** 2)
    peak_frame = int(np.argmax(mag)) if mag.size else 0
    threshold = float(np.percentile(mag, percentile)) if mag.size else 0.0
    hi = np.flatnonzero(mag >= threshold)

    center = peak_frame
    rule = "max_motion_magnitude"
    if hi.size > 0:
        breaks = np.where(np.diff(hi) > 1)[0] + 1
        groups = np.split(hi, breaks)
        scores = [float(np.mean(mag[g])) for g in groups]
        chosen = groups[int(np.argmax(scores))]
        if peak_frame in chosen:
            chosen = chosen
        center = int(round(float(np.mean(chosen))))
        rule = "top_p95_motion_cluster_center"

    if num_frames <= 1:
        delta = 0
    else:
        delta = int(min(max(1, int(base_delta)), center, max(num_frames - 1 - center, 0)))
    triplet = [int(max(0, center - delta)), int(center), int(min(num_frames - 1, center + delta))]

    return {
        "center_frame": int(center),
        "delta": int(delta),
        "triplet_frames": triplet,
        "selection_rule": rule,
        "percentile_threshold": float(percentile),
        "motion_threshold_value": float(threshold),
        "peak_frame": int(peak_frame),
        "peak_motion_magnitude_px": _safe_float(np.max(mag)) if mag.size else None,
        "_motion_magnitude": mag,
    }


def _select_zoom_regions(
    std_map: np.ndarray,
    representative_frame: np.ndarray,
    roi_mask: np.ndarray | None = None,
    max_regions: int = 2,
    scale_factor: float = REPORT_CROP_SCALE_FACTOR,
) -> list[dict[str, Any]]:
    h, w = std_map.shape
    crop_size = int(max(96, min(160, round(min(h, w) * 0.25))))
    half = crop_size // 2
    if h <= crop_size + 8 or w <= crop_size + 8:
        return [{"name": "Zoom 1", "x0": 0, "y0": 0, "x1": w, "y1": h, "score": None}]

    std_norm = _normalize_image(std_map)
    gy, gx = np.gradient(_normalize_image(representative_frame))
    contrast = _normalize_image(np.hypot(gx, gy))
    roi_float = np.asarray(roi_mask, dtype=np.float32) if roi_mask is not None else np.zeros_like(std_norm, dtype=np.float32)

    margin = max(half + 8, int(round(min(h, w) * 0.12)))
    step = max(18, crop_size // 5)
    candidates: list[dict[str, Any]] = []
    for cy in range(margin, h - margin, step):
        for cx in range(margin, w - margin, step):
            y0 = max(cy - half, 0)
            y1 = min(cy + half, h)
            x0 = max(cx - half, 0)
            x1 = min(cx + half, w)
            score = (
                float(np.mean(std_norm[y0:y1, x0:x1]))
                + 0.55 * float(np.mean(contrast[y0:y1, x0:x1]))
                + 0.75 * float(np.mean(roi_float[y0:y1, x0:x1]))
            )
            candidates.append({"cx": cx, "cy": cy, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": score})

    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen: list[dict[str, Any]] = []
    for cand in candidates:
        too_close = False
        for prev in chosen:
            if abs(cand["cx"] - prev["cx"]) < crop_size and abs(cand["cy"] - prev["cy"]) < crop_size:
                too_close = True
                break
        if too_close:
            continue
        chosen.append(cand)
        if len(chosen) >= max_regions:
            break

    if not chosen and candidates:
        chosen.append(candidates[0])

    out = []
    for idx, item in enumerate(chosen, start=1):
        width = int(item["x1"]) - int(item["x0"])
        height = int(item["y1"]) - int(item["y0"])
        shrink_w = max(48, int(round(width * float(scale_factor))))
        shrink_h = max(48, int(round(height * float(scale_factor))))
        cx = int(round((int(item["x0"]) + int(item["x1"])) / 2.0))
        cy = int(round((int(item["y0"]) + int(item["y1"])) / 2.0))
        x0 = max(0, min(w - shrink_w, cx - shrink_w // 2))
        y0 = max(0, min(h - shrink_h, cy - shrink_h // 2))
        x1 = min(w, x0 + shrink_w)
        y1 = min(h, y0 + shrink_h)
        out.append(
            {
                "name": f"Zoom {idx}",
                "x0": int(x0),
                "y0": int(y0),
                "x1": int(x1),
                "y1": int(y1),
                "score": float(item["score"]),
                "selection_rule": "combined_std_roi_density_local_contrast",
                "shrink_factor": float(scale_factor),
            }
        )
    return out


def _select_line_coords(region: dict[str, Any]) -> dict[str, Any]:
    y_mid = int(round((int(region["y0"]) + int(region["y1"])) / 2.0))
    return {
        "x0": int(region["x0"]),
        "y0": int(y_mid),
        "x1": int(region["x1"]) - 1,
        "y1": int(y_mid),
        "num_samples": max(2, int(region["x1"]) - int(region["x0"])),
    }


def _compute_temporal_mean_projection(stack_path: str | Path) -> np.ndarray:
    info = _stack_info(stack_path)
    acc = np.zeros((int(info["height_px"]), int(info["width_px"])), dtype=np.float64)
    count = 0
    for _, frame in _iter_frames(stack_path):
        acc += np.asarray(frame, dtype=np.float32)
        count += 1
    if count <= 0:
        return np.zeros_like(acc, dtype=np.float32)
    return (acc / float(count)).astype(np.float32)


def _compute_std_and_mip_projection(stack_path: str | Path) -> dict[str, np.ndarray]:
    info = _stack_info(stack_path)
    h = int(info["height_px"])
    w = int(info["width_px"])
    sum_img = np.zeros((h, w), dtype=np.float64)
    sumsq_img = np.zeros((h, w), dtype=np.float64)
    mip = np.full((h, w), -np.inf, dtype=np.float32)
    count = 0
    for _, frame in _iter_frames(stack_path):
        frame_f = np.asarray(frame, dtype=np.float32)
        sum_img += frame_f
        sumsq_img += frame_f * frame_f
        mip = np.maximum(mip, frame_f)
        count += 1
    if count <= 0:
        return {"std": np.zeros((h, w), dtype=np.float32), "mip": np.zeros((h, w), dtype=np.float32)}
    mean_img = sum_img / float(count)
    var_img = np.maximum(sumsq_img / float(count) - mean_img * mean_img, 0.0)
    return {"std": np.sqrt(var_img).astype(np.float32), "mip": mip.astype(np.float32)}


def _compute_frame_to_mean_projection_correlation_curve(stack_path: str | Path) -> dict[str, Any]:
    mean_proj = _compute_temporal_mean_projection(stack_path)
    mean_centered = np.asarray(mean_proj, dtype=np.float32) - float(np.mean(mean_proj))
    mean_norm = float(np.sqrt(np.sum(mean_centered * mean_centered)))
    num_frames = int(_stack_info(stack_path)["num_frames"])
    curve = np.full((num_frames,), np.nan, dtype=np.float32)
    if mean_norm <= 1e-12:
        return {"temporal_mean_projection": mean_proj, "curve": curve}
    for idx, frame in _iter_frames(stack_path):
        frame_f = np.asarray(frame, dtype=np.float32)
        frame_centered = frame_f - float(np.mean(frame_f))
        denom = float(np.sqrt(np.sum(frame_centered * frame_centered)) * mean_norm)
        if denom > 1e-12:
            curve[idx] = float(np.sum(frame_centered * mean_centered) / denom)
    return {"temporal_mean_projection": mean_proj, "curve": curve}

def save_single_frame_png(
    stack_path: str | Path,
    frame_index: int,
    output_png: str | Path,
    title: str | None = None,
    condition: str = "raw",
    crop: dict[str, int] | None = None,
    pixel_size_um: float | None = None,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    frame = _read_frame(stack_path, frame_index)
    if crop:
        frame = frame[int(crop["y0"]):int(crop["y1"]), int(crop["x0"]):int(crop["x1"])]
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=180)
    ax.imshow(_modality_rgb(frame, imaging_modality), interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=4)
    _add_scalebar(ax, frame.shape, pixel_size_um=pixel_size_um)
    return _save_figure(fig, output_png)


def save_projection_png(
    image_source: str | Path,
    output_png: str | Path,
    title: str | None = None,
    condition: str = "raw",
    pixel_size_um: float | None = None,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    try:
        img = _read_2d_tif(image_source)
    except Exception:
        return save_unavailable_panel(output_png, title or Path(str(image_source)).stem, "Projection unavailable")
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=180)
    ax.imshow(_modality_rgb(img, imaging_modality), interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=4)
    _add_scalebar(ax, img.shape, pixel_size_um=pixel_size_um)
    return _save_figure(fig, output_png)


def save_zoom_crops_png(
    image_sources: Sequence[dict[str, Any]],
    crop_regions: Sequence[dict[str, Any]],
    output_png: str | Path,
    frame_index: int,
    title: str | None = None,
    pixel_size_um: float | None = None,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    if not image_sources or not crop_regions:
        return save_unavailable_panel(output_png, title or "Zoom crops", "No crop or image available")

    frames: list[np.ndarray | None] = []
    for source in image_sources:
        try:
            frames.append(_read_frame(source["path"], frame_index))
        except Exception:
            frames.append(None)

    rows = len(crop_regions)
    cols = len(image_sources)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows), dpi=180)
    axes_arr = np.atleast_2d(axes)
    if rows == 1:
        axes_arr = axes_arr.reshape(1, cols)
    if cols == 1:
        axes_arr = axes_arr.reshape(rows, 1)

    for r, crop in enumerate(crop_regions):
        for c, source in enumerate(image_sources):
            ax = axes_arr[r, c]
            frame = frames[c]
            if frame is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_axis_off()
                continue
            y0, y1 = int(crop["y0"]), int(crop["y1"])
            x0, x1 = int(crop["x0"]), int(crop["x1"])
            crop_img = frame[y0:y1, x0:x1]
            ax.imshow(_modality_rgb(crop_img, imaging_modality), interpolation="nearest")
            ax.set_axis_off()
            if r == 0:
                ax.set_title(str(source.get("title") or source.get("condition", "raw")), pad=4)
            if c == 0:
                ax.text(-0.03, 0.5, str(crop.get("name", f"Zoom {r+1}")), transform=ax.transAxes, rotation=90, ha="right", va="center")
            _add_scalebar(ax, crop_img.shape, pixel_size_um=pixel_size_um)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    return _save_figure(fig, output_png)


def save_rgb_motion_composite_png(
    stack_path: str | Path,
    frame_triplet: Sequence[int],
    output_png: str | Path,
    title: str | None = None,
    crop: dict[str, int] | None = None,
    pixel_size_um: float | None = None,
) -> str:
    if len(frame_triplet) != 3:
        return save_unavailable_panel(output_png, title or "Motion composite", "Triplet unavailable")
    frames = []
    for idx in frame_triplet:
        frame = _read_frame(stack_path, int(idx))
        if crop:
            frame = frame[int(crop["y0"]):int(crop["y1"]), int(crop["x0"]):int(crop["x1"])]
        frames.append(np.asarray(frame, dtype=np.float32))
    stack = np.stack(frames, axis=0)
    lo = float(np.percentile(stack, 1))
    hi = float(np.percentile(stack, 99))
    rgb = np.zeros((*frames[0].shape, 3), dtype=np.float32)
    if hi > lo + 1e-8:
        rgb = np.clip((np.transpose(stack, (1, 2, 0)) - lo) / (hi - lo), 0.0, 1.0)
    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=180)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=4)
    _add_scalebar(ax, frames[0].shape, pixel_size_um=pixel_size_um)
    ax.text(0.02, 0.02, f"R={int(frame_triplet[0])}  G={int(frame_triplet[1])}  B={int(frame_triplet[2])}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="white", bbox={"facecolor": (0, 0, 0, 0.42), "edgecolor": "none", "pad": 1.2})
    return _save_figure(fig, output_png)


def _sample_line(img: np.ndarray, line_coords: dict[str, Any]) -> np.ndarray:
    x0 = float(line_coords["x0"])
    y0 = float(line_coords["y0"])
    x1 = float(line_coords["x1"])
    y1 = float(line_coords["y1"])
    n = int(line_coords.get("num_samples", max(abs(int(round(x1 - x0))), abs(int(round(y1 - y0))), 2) + 1))
    xs = np.clip(np.round(np.linspace(x0, x1, n)).astype(np.int32), 0, img.shape[1] - 1)
    ys = np.clip(np.round(np.linspace(y0, y1, n)).astype(np.int32), 0, img.shape[0] - 1)
    return np.asarray(img[ys, xs], dtype=np.float32)


def save_kymograph_png(
    stack_sources: Sequence[dict[str, Any]],
    line_coords: dict[str, Any],
    output_png: str | Path,
    title: str | None = None,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    if not stack_sources:
        return save_unavailable_panel(output_png, title or "Kymograph", "No stack available")
    rows = len(stack_sources)
    fig, axes = plt.subplots(rows, 1, figsize=(7.6, max(2.5 * rows, 3.0)), dpi=180)
    axes_arr = np.atleast_1d(axes)
    for ax, source in zip(axes_arr, stack_sources):
        try:
            lines = []
            for _, frame in _iter_frames(source["path"]):
                lines.append(_sample_line(np.asarray(frame), line_coords))
            kymo = np.asarray(lines, dtype=np.float32)
            ax.imshow(
                _modality_display_image(kymo.T, imaging_modality),
                aspect="auto",
                interpolation="nearest",
                cmap=MODALITY_CMAPS[_normalize_modality(imaging_modality)],
                origin="lower",
            )
            ax.set_box_aspect(0.5)
            ax.set_title(str(source.get("title") or source.get("condition", "raw")), loc="left", pad=4)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Position along line")
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_axis_off()
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    return _save_figure(fig, output_png)


def save_correlation_curve_panel(
    raw_curve: np.ndarray,
    final_curve: np.ndarray,
    output_png: str | Path,
    title: str | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7.8, 3.2), dpi=180)
    has_curve = False
    x_raw = np.arange(len(raw_curve), dtype=np.int32)
    x_final = np.arange(len(final_curve), dtype=np.int32)
    if len(x_raw) > 0 and np.any(np.isfinite(raw_curve)):
        ax.plot(x_raw, raw_curve, color=RAW_COLOR, linewidth=1.2, label=_reader_label("raw"))
        has_curve = True
    if len(x_final) > 0 and np.any(np.isfinite(final_curve)):
        ax.plot(x_final, final_curve, color=FINAL_COLOR, linewidth=1.2, label=DISPLAY_FINAL_NAME)
        has_curve = True
    if has_curve:
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.05, 1.02)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(title or "Frame-to-mean projection correlation", loc="left", pad=4)
    return _save_figure(fig, output_png)


def save_line_profile_png(image_sources: Sequence[dict[str, Any]], frame_index: int, line_coords: dict[str, Any], output_png: str | Path, title: str | None = None) -> str:
    if not image_sources:
        return save_unavailable_panel(output_png, title or "Line profile", "No image available")
    fig, ax = plt.subplots(figsize=(6.4, 3.1), dpi=180)
    has_curve = False
    for source in image_sources:
        try:
            frame = _read_frame(source["path"], frame_index)
            profile = _normalize_image(_sample_line(frame, line_coords))
            ax.plot(profile, color=CONDITION_COLORS.get(source.get("condition", "raw"), RAW_COLOR), linewidth=1.4, label=str(source.get("title") or source.get("condition", "raw")))
            has_curve = True
        except Exception:
            continue
    if has_curve:
        ax.legend(loc="upper right", frameon=False)
        ax.set_xlabel("Line position")
        ax.set_ylabel("Normalized intensity")
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()
    if title:
        ax.set_title(title, loc="left", pad=4)
    return _save_figure(fig, output_png)


def save_metric_comparison_panel(metric_specs: Sequence[dict[str, Any]], output_png: str | Path, title: str | None = None) -> str:
    metric_specs = list(metric_specs or [])
    if not metric_specs:
        return save_unavailable_panel(output_png, title or "Metric panel", "No metric available")
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(3.0 * len(metric_specs), 3.3), dpi=180)
    axes_arr = np.atleast_1d(axes)
    for ax, spec in zip(axes_arr, metric_specs):
        labels = list(spec.get("labels") or [_reader_label("raw"), spec.get("candidate_label", DISPLAY_FINAL_NAME)])
        values = list(spec.get("values") or [spec.get("raw"), spec.get("candidate")])
        colors = list(spec.get("colors") or [RAW_COLOR, spec.get("candidate_color", FINAL_COLOR)])[: len(values)]
        numeric = [0.0 if _safe_float(v) is None else float(v) for v in values]
        bars = ax.bar(range(len(values)), numeric, color=colors, width=0.62)
        max_numeric = max([0.0] + numeric)
        offset = 0.04 * max(1.0, max_numeric)
        for bar, value in zip(bars, values):
            y = float(bar.get_height())
            if _safe_float(value) is None:
                bar.set_alpha(0.35)
                bar.set_hatch("//")
                ax.text(bar.get_x() + bar.get_width() / 2.0, y + offset, "N/A", ha="center", va="bottom")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2.0, y + offset, str(spec.get("value_fmt", "{:.3g}")).format(float(value)), ha="center", va="bottom")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(str(spec.get("label", "metric")), pad=4)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylim(0, max(max_numeric * 1.28 + offset, 1.0))
        delta_text = spec.get("delta_text")
        if delta_text:
            ax.text(0.5, -0.22, str(delta_text), transform=ax.transAxes, ha="center", va="top", fontsize=8)
        unit = str(spec.get("unit") or "").strip()
        if unit:
            ax.set_ylabel(unit)
    if title:
        fig.suptitle(title, fontsize=11, y=1.03)
    return _save_figure(fig, output_png)


def save_downstream_count_panel(
    count_rows: Sequence[dict[str, Any]],
    output_png: str | Path,
    title: str | None = None,
    final_display_name: str = DISPLAY_FINAL_NAME,
) -> str:
    count_rows = list(count_rows or [])
    if not count_rows:
        return save_unavailable_panel(output_png, title or "Downstream counts", "No count available")
    labels = [str(row["label"]) for row in count_rows]
    raw_vals = [0.0 if _safe_float(row.get("raw")) is None else float(row.get("raw")) for row in count_rows]
    final_vals = [0.0 if _safe_float(row.get("final")) is None else float(row.get("final")) for row in count_rows]
    x = np.arange(len(labels), dtype=np.float32)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=180)
    raw_bars = ax.bar(x - width / 2.0, raw_vals, width=width, color=RAW_COLOR, label=_reader_label("raw"))
    final_bars = ax.bar(x + width / 2.0, final_vals, width=width, color=FINAL_COLOR, label=final_display_name)
    for bars in (raw_bars, final_bars):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(1.0, 0.02 * max(raw_vals + final_vals + [1.0])), f"{int(round(bar.get_height()))}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper right", frameon=False)
    if title:
        ax.set_title(title, loc="left", pad=4)
    return _save_figure(fig, output_png)


def assemble_scientific_panel_layout(panel_specs: Sequence[dict[str, Any]], output_png: str | Path, title: str, subtitle: str | None = None, columns: int = 2) -> str:
    panel_specs = list(panel_specs or [])
    if not panel_specs:
        return save_unavailable_panel(output_png, title, "No panel available")
    columns = max(1, int(columns))
    rows = int(math.ceil(len(panel_specs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.0 * columns, 4.7 * rows), dpi=180)
    axes_arr = np.atleast_1d(axes).reshape(rows, columns)
    flat_axes = list(axes_arr.ravel())
    for ax, spec in zip(flat_axes, panel_specs):
        img_path = spec.get("image_path")
        if img_path and Path(img_path).exists():
            ax.imshow(plt.imread(str(img_path)))
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()
        label = str(spec.get("panel_label") or "")
        title_txt = str(spec.get("title") or Path(str(img_path)).stem)
        ax.set_title(f"{label}  {title_txt}".strip(), loc="left", pad=4)
    for ax in flat_axes[len(panel_specs):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=13, y=1.01)
    if subtitle:
        fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=9, color=TEXT_MUTED)
    return _save_figure(fig, output_png)


def save_intensity_grid_png(
    image_rows: Sequence[Sequence[dict[str, Any]]],
    output_png: str | Path,
    title: str,
    row_labels: Sequence[str] | None = None,
    pixel_size_um: float | None = None,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    rows = len(image_rows)
    cols = max((len(row) for row in image_rows), default=0)
    if rows <= 0 or cols <= 0:
        return save_unavailable_panel(output_png, title, "No image available")
    fig, axes = plt.subplots(rows, cols, figsize=(3.35 * cols, 3.15 * rows), dpi=180)
    axes_arr = np.atleast_2d(axes).reshape(rows, cols)
    for r in range(rows):
        for c in range(cols):
            ax = axes_arr[r, c]
            if c >= len(image_rows[r]):
                ax.set_axis_off()
                continue
            spec = image_rows[r][c]
            img = np.asarray(spec["image"], dtype=np.float32)
            ax.imshow(_modality_rgb(img, imaging_modality), interpolation="nearest")
            ax.set_axis_off()
            ax.set_title(str(spec.get("title") or ""), loc="left", pad=3, fontsize=9)
            for rect_idx, rect in enumerate(spec.get("rectangles") or [], start=1):
                x0 = float(rect["x0"])
                y0 = float(rect["y0"])
                width = float(rect["x1"]) - x0
                height = float(rect["y1"]) - y0
                edgecolor = str(rect.get("edgecolor") or "#ffffff")
                linewidth = float(rect.get("linewidth") or 1.8)
                patch = Rectangle((x0, y0), width, height, fill=False, edgecolor=edgecolor, linewidth=linewidth)
                ax.add_patch(patch)
                label = rect.get("label")
                if label:
                    ax.text(
                        x0 + max(3.0, width * 0.05),
                        y0 + max(10.0, height * 0.10),
                        str(label),
                        color=edgecolor,
                        ha="left",
                        va="top",
                        fontsize=8,
                        fontweight="bold",
                        bbox={"facecolor": (0, 0, 0, 0.40), "edgecolor": "none", "pad": 1.0},
                    )
            _add_scalebar(ax, img.shape, pixel_size_um=pixel_size_um)
            if row_labels and c == 0 and r < len(row_labels):
                ax.text(
                    -0.08,
                    0.5,
                    str(row_labels[r]),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    fig.suptitle(title, fontsize=12, y=1.01)
    return _save_figure(fig, output_png)


def _compute_kymograph_array(stack_path: str | Path, line_coords: dict[str, Any]) -> np.ndarray:
    lines = []
    for _, frame in _iter_frames(stack_path):
        lines.append(_sample_line(np.asarray(frame), line_coords))
    if not lines:
        return np.zeros((1, 1), dtype=np.float32)
    return np.asarray(lines, dtype=np.float32).T


def save_kymograph_bundle_png(
    neuropilot_std_image: np.ndarray,
    raw_stack_path: str | Path,
    final_stack_path: str | Path,
    line_coords_list: Sequence[dict[str, Any]],
    output_png: str | Path,
    pixel_size_um: float | None = None,
    fps_hz: float = DEFAULT_FPS_HZ,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
) -> str:
    if not line_coords_list:
        return save_unavailable_panel(output_png, "Kymograph bundle", "No line coordinates available")
    fig = plt.figure(figsize=(12.6, 6.3), dpi=180)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.02, 1.28, 1.28], hspace=0.22, wspace=0.18)
    ax_overlay = fig.add_subplot(gs[:, 0])
    ax_overlay.imshow(_modality_rgb(neuropilot_std_image, imaging_modality), interpolation="nearest")
    ax_overlay.set_axis_off()
    ax_overlay.set_title(f"{DISPLAY_FINAL_NAME} STD with kymograph lines", loc="left", pad=4, fontsize=10)
    _add_scalebar(ax_overlay, neuropilot_std_image.shape, pixel_size_um=pixel_size_um)

    line_colors = ["#f39c12", "#15aabf", "#8e44ad", "#2ecc71"]

    def _add_kymo_scalebar(ax, kymo_shape: Sequence[int]):
        kh, kw = int(kymo_shape[0]), int(kymo_shape[1])
        margin_x = max(10, int(round(kw * 0.06)))
        margin_y = max(3, int(round(kh * 0.07)))
        spatial_px = _choose_scalebar_px((kh, kh))
        if fps_hz > 0:
            seconds_target = 5.0 if kw / fps_hz >= 5.0 else max(1.0, round((kw / fps_hz) * 0.2, 1))
            frame_options = np.asarray([10, 20, 30, 50, 75, 100, 150, 200, 300], dtype=np.int32)
            time_frames = int(frame_options[np.argmin(np.abs(frame_options - seconds_target * fps_hz))])
        else:
            time_frames = max(10, int(round(kw * 0.15)))
        time_frames = int(min(time_frames, max(10, kw - margin_x - 2)))
        x1 = kw - margin_x
        x0 = max(2, x1 - time_frames)
        y0 = max(2, margin_y)
        y1 = min(kh - 2, y0 + spatial_px)
        ax.plot([x0, x1], [y0, y0], color="white", linewidth=2.0, solid_capstyle="butt")
        ax.plot([x0, x0], [y0, y1], color="white", linewidth=2.0, solid_capstyle="butt")
        time_label = f"{time_frames / fps_hz:.0f} s" if fps_hz > 0 else f"{time_frames} fr"
        spatial_label = _scalebar_label(spatial_px, pixel_size_um)
        ax.text(
            (x0 + x1) / 2.0,
            y0 + max(2, kh * 0.06),
            time_label,
            color="white",
            ha="center",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": (0, 0, 0, 0.35), "edgecolor": "none", "pad": 1.0},
        )
        ax.text(
            x0 + max(2, kw * 0.02),
            (y0 + y1) / 2.0,
            spatial_label,
            color="white",
            ha="left",
            va="center",
            rotation=90,
            fontsize=7,
            bbox={"facecolor": (0, 0, 0, 0.35), "edgecolor": "none", "pad": 1.0},
        )

    for idx, line_coords in enumerate(line_coords_list, start=1):
        color = line_colors[(idx - 1) % len(line_colors)]
        ax_overlay.plot(
            [line_coords["x0"], line_coords["x1"]],
            [line_coords["y0"], line_coords["y1"]],
            color=color,
            linewidth=2.2,
            solid_capstyle="round",
        )
        ax_overlay.text(
            float(line_coords["x0"] + line_coords["x1"]) / 2.0,
            float(line_coords["y0"] + line_coords["y1"]) / 2.0,
            str(idx),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "white", "linewidth": 0.8, "boxstyle": "circle,pad=0.25"},
        )

    for idx, line_coords in enumerate(line_coords_list[:2], start=1):
        raw_ax = fig.add_subplot(gs[idx - 1, 1])
        final_ax = fig.add_subplot(gs[idx - 1, 2])
        raw_kymo = _compute_kymograph_array(raw_stack_path, line_coords)
        final_kymo = _compute_kymograph_array(final_stack_path, line_coords)
        for ax, kymo, ttl in [
            (raw_ax, raw_kymo, f"Line {idx} - {_reader_label('raw')}"),
            (final_ax, final_kymo, f"Line {idx} - {DISPLAY_FINAL_NAME}"),
        ]:
            ax.imshow(
                _modality_display_image(kymo, imaging_modality),
                aspect="auto",
                interpolation="nearest",
                cmap=MODALITY_CMAPS[_normalize_modality(imaging_modality)],
                origin="lower",
            )
            ax.set_box_aspect(0.5)
            ax.set_title(ttl, loc="left", pad=3, fontsize=9)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Position along line")
            _add_kymo_scalebar(ax, kymo.shape)
    fig.suptitle("Kymograph comparison", fontsize=12, y=0.995)
    return _save_figure(fig, output_png)


def _load_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(Path(csv_path).expanduser().resolve(), "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_ranked_roi_rows_from_csv(
    csv_path: str | Path,
    subset: str = "analysis",
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    subset_key_map = {
        "analysis": "in_analysis_set",
        "display": "in_display_subset",
        "all": None,
    }
    subset_key = subset_key_map.get(str(subset).strip().lower())
    rows: list[dict[str, Any]] = []
    for row in _load_csv_rows(csv_path):
        if subset_key is not None:
            try:
                if int(float(row.get(subset_key, "0") or 0)) != 1:
                    continue
            except Exception:
                continue
        try:
            rows.append(
                {
                    "rank": int(float(row["rank"])),
                    "rid": int(float(row["rid"])),
                    "largest_cc_area_raw": int(float(row.get("largest_cc_area_raw", 0) or 0)),
                    "final_area_polished": int(float(row.get("final_area_polished", 0) or 0)),
                    "trace_max": float(row.get("trace_max", 0.0) or 0.0),
                }
            )
        except Exception:
            continue
    rows.sort(key=lambda item: int(item["rank"]))
    if top_n is None:
        return rows
    return rows[: max(0, int(top_n))]


def _load_rank_masks_from_labelmask(labelmask_path: str | Path, roi_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    labelmask = np.asarray(tiff.imread(str(Path(labelmask_path).expanduser().resolve())))
    labelmask = np.squeeze(labelmask)
    out: list[dict[str, Any]] = []
    for row in roi_rows:
        rank = int(row["rank"])
        mask = labelmask == rank
        if int(np.count_nonzero(mask)) <= 0:
            continue
        item = dict(row)
        item["mask"] = mask
        item["pixel_count"] = int(np.count_nonzero(mask))
        out.append(item)
    return out


def _merge_roi_rows_with_labelmask_ranks(
    labelmask_path: str | Path,
    roi_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    labelmask = np.asarray(_read_2d_tif(labelmask_path), dtype=np.int32)
    label_ranks = sorted(int(x) for x in np.unique(labelmask) if int(x) > 0)
    row_by_rank = {int(row["rank"]): dict(row) for row in (roi_rows or [])}
    merged: list[dict[str, Any]] = []
    for rank in label_ranks:
        item = dict(row_by_rank.get(rank, {}))
        item["rank"] = int(rank)
        item["rid"] = int(item.get("rid", rank))
        merged.append(item)
    return merged


def _extract_rank_mean_traces_from_stack(
    stack_path: str | Path,
    labelmask_path: str | Path,
    roi_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    labelmask = np.asarray(_read_2d_tif(labelmask_path), dtype=np.int32)
    if labelmask.size == 0 or len(roi_rows) == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    label_flat = labelmask.reshape(-1)
    max_rank = int(max(int(np.max(label_flat)), max(int(row["rank"]) for row in roi_rows)))
    pixel_counts = np.bincount(label_flat, minlength=max_rank + 1).astype(np.float64, copy=False)

    valid_rows: list[dict[str, Any]] = []
    rank_ids: list[int] = []
    for row in roi_rows:
        rank = int(row["rank"])
        if rank < 0 or rank >= pixel_counts.shape[0]:
            continue
        pixel_count = int(pixel_counts[rank])
        if pixel_count <= 0:
            continue
        item = dict(row)
        item["pixel_count"] = pixel_count
        valid_rows.append(item)
        rank_ids.append(rank)

    if not valid_rows:
        return [], np.zeros((0, 0), dtype=np.float32)

    info = _stack_info(stack_path)
    traces = np.zeros((len(valid_rows), int(info["num_frames"])), dtype=np.float32)
    rank_arr = np.asarray(rank_ids, dtype=np.int32)
    denom = pixel_counts[rank_arr]

    frame_write_idx = 0
    for _, frame in _iter_frames(stack_path):
        flat = np.asarray(frame, dtype=np.float64).reshape(-1)
        sums = np.bincount(label_flat, weights=flat, minlength=max_rank + 1)
        traces[:, frame_write_idx] = (sums[rank_arr] / denom).astype(np.float32)
        frame_write_idx += 1

    if frame_write_idx != traces.shape[1]:
        traces = traces[:, :frame_write_idx]
    return valid_rows, traces


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return mask
    inner = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        inner[1:-1, 1:-1] = (
            mask[1:-1, 1:-1]
            & mask[:-2, 1:-1]
            & mask[2:, 1:-1]
            & mask[1:-1, :-2]
            & mask[1:-1, 2:]
        )
    return mask & ~inner


def _report_labels_for_overlay(labelmask: np.ndarray, roi_summary_csv: str | Path | None, report_top_percent: float) -> list[int]:
    labels = [int(x) for x in np.unique(labelmask) if int(x) > 0]
    labels.sort()
    csv_ranks: list[int] = []
    if roi_summary_csv and Path(roi_summary_csv).exists():
        for row in _load_csv_rows(roi_summary_csv):
            try:
                csv_ranks.append(int(float(row["rank"])))
            except Exception:
                continue
        csv_ranks = sorted(set(csv_ranks))
    source_labels = labels
    if len(csv_ranks) >= len(labels):
        source_labels = csv_ranks
    keep = max(1, int(math.ceil(len(source_labels) * float(report_top_percent)))) if source_labels else 0
    return source_labels[:keep]


def save_segmentation_overlay_png(
    background_img: np.ndarray,
    labelmask_path: str | Path,
    roi_summary_csv: str | Path | None,
    output_png: str | Path,
    title: str,
    report_top_percent: float,
    pixel_size_um: float | None = None,
) -> dict[str, Any]:
    labelmask = np.asarray(tiff.imread(str(Path(labelmask_path).expanduser().resolve())))
    labelmask = np.squeeze(labelmask)
    selected_labels = _report_labels_for_overlay(labelmask, roi_summary_csv, report_top_percent)
    if labelmask.ndim != 2 or not selected_labels:
        return {
            "png": save_unavailable_panel(output_png, title, "Segmentation overview unavailable"),
            "selected_count": 0,
            "selected_ranks": [],
        }
    bg = _normalize_image(background_img)
    rgb = np.repeat(bg[..., None], 3, axis=2)
    cmap = plt.get_cmap("tab20")
    selected_ranks: list[int] = []
    label_annotations: list[tuple[float, float, np.ndarray, int]] = []
    for idx, rank in enumerate(selected_labels):
        mask = labelmask == rank
        if not np.any(mask):
            continue
        color = np.asarray(cmap(idx % 20)[:3], dtype=np.float32)
        boundary = _mask_boundary(mask)
        rgb[mask] = 0.84 * rgb[mask] + 0.16 * color
        rgb[boundary] = color
        selected_ranks.append(rank)
        cx, cy = _mask_label_position(mask)
        label_annotations.append((cx, cy, color, int(rank)))
    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=180)
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
    for cx, cy, color, rank in label_annotations:
        ax.text(
            cx,
            cy,
            str(rank),
            color="white",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "white", "linewidth": 0.6, "boxstyle": "round,pad=0.18"},
        )
    ax.set_axis_off()
    ax.set_title(title, loc="left", pad=4)
    _add_scalebar(ax, labelmask.shape, pixel_size_um=pixel_size_um)
    ax.text(
        0.02,
        0.02,
        f"Report display subset: top {int(round(report_top_percent * 100))}% of analysis ROI",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        bbox={"facecolor": (0, 0, 0, 0.45), "edgecolor": "none", "pad": 1.2},
    )
    return {
        "png": _save_figure(fig, output_png),
        "selected_count": len(selected_ranks),
        "selected_ranks": selected_ranks,
    }


def _normalize_trace_for_plot(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    if trace.size == 0:
        return trace.astype(np.float32)
    centered = trace - float(np.percentile(trace, 10))
    scale = max(float(np.std(centered)), 1e-6)
    return (centered / scale).astype(np.float32)


def _resample_trace_to_length(trace: np.ndarray, target_len: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if trace.size == target_len:
        return trace.astype(np.float32, copy=False)
    if trace.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if trace.size == 1:
        return np.full((target_len,), float(trace[0]), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, int(trace.size), dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    return np.interp(dst_x, src_x, trace.astype(np.float64, copy=False)).astype(np.float32)


def _align_traces_for_corr(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0:
        return a, b
    target_len = min(int(a.size), int(b.size))
    return _resample_trace_to_length(a, target_len), _resample_trace_to_length(b, target_len)


def _trace_plot_xcoords(trace: np.ndarray, axis_len: int) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)
    axis_len = max(int(axis_len), 1)
    if trace.size <= 1:
        return np.zeros((trace.size,), dtype=np.float32)
    if trace.size == axis_len:
        return np.arange(axis_len, dtype=np.float32)
    return np.linspace(0.0, float(axis_len - 1), int(trace.size), dtype=np.float32)


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
        valid = row[finite]
        mean = float(np.mean(valid))
        std = float(np.std(valid))
        if std <= 1e-12:
            continue
        out[i, finite] = ((valid - mean) / std).astype(np.float32)
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


def _mask_label_position(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return 0.0, 0.0
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    coords = np.column_stack((ys, xs)).astype(np.float64, copy=False)
    nearest_idx = int(np.argmin((coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2))
    anchor = coords[nearest_idx]
    return float(anchor[1]), float(anchor[0])


def _save_paired_trace_plot(raw_traces: np.ndarray, final_traces: np.ndarray, roi_defs: Sequence[dict[str, Any]], output_png: str | Path) -> str:
    if len(roi_defs) == 0:
        return save_unavailable_panel(output_png, "Paired trace curves", "No paired ROI available")
    n = len(roi_defs)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, max(2.2 * n, 3.2)), dpi=180, sharex=True)
    axes_arr = np.atleast_1d(axes)
    roi_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.95, max(n, 2)))
    for i, ax in enumerate(axes_arr):
        raw_norm = _normalize_trace_for_plot(raw_traces[i])
        final_norm = _normalize_trace_for_plot(final_traces[i])
        corr_raw, corr_final = _align_traces_for_corr(raw_norm, final_norm)
        corr = _safe_corrcoef(corr_raw, corr_final)
        corr_txt = "N/A" if corr is None else f"{corr:.3f}"
        axis_len = max(int(raw_norm.size), int(final_norm.size), 1)
        raw_x = _trace_plot_xcoords(raw_norm, axis_len)
        final_x = _trace_plot_xcoords(final_norm, axis_len)
        ax.axhline(0.0, color="#d8d8d8", linewidth=0.7, zorder=0)
        ax.plot(raw_x, raw_norm, color=RAW_COLOR, linewidth=0.95, label=_reader_label("raw"))
        ax.plot(final_x, final_norm, color=FINAL_COLOR, linewidth=0.95, label=DISPLAY_FINAL_NAME)
        ax.text(
            -0.07,
            0.5,
            str(i + 1),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox={"facecolor": roi_colors[i], "edgecolor": "none", "boxstyle": "round,pad=0.25"},
        )
        title_suffix = ""
        if raw_norm.size != final_norm.size:
            title_suffix = f"  len(raw/final)={int(raw_norm.size)}/{int(final_norm.size)}"
        ax.set_title(f"ROI {i + 1}  rid={int(roi_defs[i]['rid'])}  corr={corr_txt}{title_suffix}", fontsize=9)
        ax.grid(True, alpha=0.22)
        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"Paired trace curves ({DISPLAY_FINAL_NAME} ROI anchor)", fontsize=11)
    fig.supxlabel("Frame (progress-aligned when lengths differ)")
    fig.supylabel("Normalized trace")
    return _save_figure(fig, output_png)


def _save_paired_mask_plot(
    roi_defs: Sequence[dict[str, Any]],
    output_png: str | Path,
    labelmask_out: str | Path,
    background_img: np.ndarray | None = None,
) -> dict[str, str]:
    if len(roi_defs) == 0:
        blank = np.zeros((1, 1), dtype=np.uint16)
        tiff.imwrite(str(labelmask_out), blank, photometric="minisblack", metadata=None)
        return {"mask_png": save_unavailable_panel(output_png, "Paired ROI map", "No paired ROI available"), "labelmask_tif": str(labelmask_out)}
    h, w = roi_defs[0]["mask"].shape
    labelmask = np.zeros((h, w), dtype=np.uint16)
    for idx, roi in enumerate(roi_defs, start=1):
        labelmask[np.asarray(roi["mask"], dtype=bool)] = np.uint16(idx)
    tiff.imwrite(str(labelmask_out), labelmask, photometric="minisblack", metadata=None)
    bg = np.zeros((h, w), dtype=np.float32) if background_img is None else _normalize_image(background_img)
    rgb = np.repeat(bg[..., None], 3, axis=2)
    roi_colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.95, max(len(roi_defs), 2)))
    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=180)
    for idx, roi in enumerate(roi_defs, start=1):
        mask = np.asarray(roi["mask"], dtype=bool)
        color = np.asarray(roi_colors[idx - 1][:3], dtype=np.float32)
        boundary = _mask_boundary(mask)
        rgb[mask] = 0.82 * rgb[mask] + 0.18 * color
        rgb[boundary] = color
        cx, cy = _mask_label_position(mask)
        ax.text(
            cx,
            cy,
            str(idx),
            color="white",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": color, "edgecolor": "white", "linewidth": 0.7, "boxstyle": "circle,pad=0.28"},
        )
    ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(f"Paired ROI map ({DISPLAY_FINAL_NAME} anchor)", loc="left", pad=4)
    return {"mask_png": _save_figure(fig, output_png), "labelmask_tif": str(labelmask_out)}


def _save_trace_corr_heatmap_png(
    corr_matrix: np.ndarray,
    roi_defs: Sequence[dict[str, Any]],
    output_png: str | Path,
    title: str,
) -> str:
    if len(roi_defs) == 0 or np.asarray(corr_matrix).size == 0:
        return save_unavailable_panel(output_png, title, "No ROI correlation heatmap available")

    n = len(roi_defs)
    size = float(np.clip(0.12 * n + 6.0, 6.2, 18.0))
    tick_fontsize = _heatmap_tick_fontsize(n)
    fig, ax = plt.subplots(figsize=(size, size), dpi=180)
    cmap = plt.get_cmap("bwr")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    if hasattr(cmap, "set_bad"):
        cmap.set_bad("#cfcfcf")
    im = ax.imshow(np.asarray(corr_matrix, dtype=np.float32), cmap=cmap, vmin=-1.0, vmax=1.0, interpolation="nearest", origin="upper")
    rank_labels = [int(roi["rank"]) for roi in roi_defs]
    ticks, tick_idx = _heatmap_tick_positions(n)
    if len(ticks) > 0:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(rank_labels[i]) for i in tick_idx], rotation=90, fontsize=tick_fontsize)
        ax.set_yticklabels([str(rank_labels[i]) for i in tick_idx], fontsize=tick_fontsize)
    ax.set_xlabel("Selected ROI (final rank)")
    ax.set_ylabel("Selected ROI (final rank)")
    ax.set_title(title, loc="left", pad=4)
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    return _save_figure(fig, output_png)


def _save_trace_temporal_heatmap_png(
    traces: np.ndarray,
    roi_defs: Sequence[dict[str, Any]],
    output_png: str | Path,
    title: str,
    target_bin_count: int = 1000,
    fps_hz: float | None = None,
) -> tuple[str, int]:
    traces = np.asarray(traces, dtype=np.float32)
    if len(roi_defs) == 0 or traces.size == 0:
        return save_unavailable_panel(output_png, title, "No temporal trace heatmap available"), 0

    binned, bin_count = _bin_trace_matrix(traces, target_bin_count=target_bin_count)
    zscore = _zscore_trace_rows(binned)
    scale = min(0.12, 20.0 / max(bin_count, 1), 14.0 / max(len(roi_defs), 1))
    fig_w = max(7.5, float(bin_count) * scale + 1.6)
    fig_h = max(4.5, float(len(roi_defs)) * scale + 1.6)
    tick_fontsize = _heatmap_tick_fontsize(len(roi_defs))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    cmap = plt.get_cmap("RdBu_r")
    cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    if hasattr(cmap, "set_bad"):
        cmap.set_bad("#cfcfcf")
    duration_s: float | None = None
    fps_value = 0.0
    if fps_hz is not None:
        try:
            fps_value = float(fps_hz)
        except (TypeError, ValueError):
            fps_value = 0.0
        if math.isfinite(fps_value) and fps_value > 0:
            duration_s = float(traces.shape[1]) / fps_value
    im = ax.imshow(zscore, cmap=cmap, vmin=-3.0, vmax=3.0, aspect="equal", interpolation="nearest", origin="upper")
    if duration_s is not None:
        xtick_count = max(2, min(6, bin_count))
        xticks = np.unique(np.round(np.linspace(0, max(bin_count - 1, 0), num=xtick_count)).astype(np.int32))
        ax.set_xticks(xticks)
        sec_per_bin = float(traces.shape[1]) / float(bin_count) / fps_value
        ax.set_xticklabels([f"{float(t) * sec_per_bin:.1f}" for t in xticks])
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xlabel(f"Time bin (count={bin_count})")
    rank_labels = [int(roi["rank"]) for roi in roi_defs]
    ticks, tick_idx = _heatmap_tick_positions(len(roi_defs))
    if len(ticks) > 0:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(rank_labels[i]) for i in tick_idx], fontsize=tick_fontsize)
    ax.set_ylabel("Selected ROI (final rank)")
    ax.set_title(title, loc="left", pad=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Per-cell Z-score")
    return _save_figure(fig, output_png), int(bin_count)


def _generate_paired_trace_assets(
    seg_dir: Path,
    assets_dir: Path,
    unavailable: list[str],
    background_img: np.ndarray | None = None,
    fps_hz: float | None = None,
    final_stack_path: str | Path | None = None,
) -> dict[str, Any]:
    paired_dir = seg_dir / "paired_trace"
    summary = _read_json(paired_dir / "paired_trace_summary.json")
    if not summary:
        unavailable.append("paired_trace_summary_missing")
        return {
            "summary": None,
            "curve_png": None,
            "mask_png": None,
            "labelmask_tif": None,
            "raw_corr_png": None,
            "final_corr_png": None,
            "raw_trace_heatmap_png": None,
            "final_trace_heatmap_png": None,
            "displayed_count": 0,
            "all_selected_count": 0,
        }

    curve_src_raw = str(summary.get("artifacts", {}).get("paired_trace_plot_png", "")).strip()
    mask_src_raw = str(summary.get("artifacts", {}).get("paired_mask_png", "")).strip()
    labelmask_src_raw = str(summary.get("artifacts", {}).get("paired_labelmask_tif", "")).strip()
    curve_src = Path(curve_src_raw) if curve_src_raw else None
    mask_src = Path(mask_src_raw) if mask_src_raw else None
    labelmask_src = Path(labelmask_src_raw) if labelmask_src_raw else None
    curve_dst = assets_dir / "paired_trace_curves.png"
    mask_dst = assets_dir / "paired_trace_numbered_map.png"
    labelmask_dst = assets_dir / "paired_trace_anchor_labelmask_uint16.tif"
    raw_corr_dst = assets_dir / "raw_trace_corr_heatmap.png"
    final_corr_dst = assets_dir / "final_trace_corr_heatmap.png"
    raw_temporal_dst = assets_dir / "raw_trace_temporal_heatmap.png"
    final_temporal_dst = assets_dir / "final_trace_temporal_heatmap.png"

    artifacts = dict(summary.get("artifacts", {}) or {})
    result = {
        "summary": summary,
        "curve_png": None,
        "mask_png": None,
        "labelmask_tif": None,
        "raw_corr_png": _copy_if_exists(artifacts.get("raw_trace_corr_heatmap_png"), raw_corr_dst),
        "final_corr_png": _copy_if_exists(artifacts.get("final_trace_corr_heatmap_png"), final_corr_dst),
        "raw_trace_heatmap_png": _copy_if_exists(artifacts.get("raw_trace_temporal_heatmap_png"), raw_temporal_dst),
        "final_trace_heatmap_png": _copy_if_exists(artifacts.get("final_trace_temporal_heatmap_png"), final_temporal_dst),
        "displayed_count": int(summary.get("selected_count") or summary.get("n_selected_final_rois") or summary.get("trace_count") or 0),
        "all_selected_count": int(summary.get("all_selected_count") or 0),
    }

    if (
        curve_src is not None and curve_src.is_file()
        and mask_src is not None and mask_src.is_file()
        and labelmask_src is not None and labelmask_src.is_file()
        and background_img is None
    ):
        shutil.copy2(curve_src, curve_dst)
        shutil.copy2(mask_src, mask_dst)
        shutil.copy2(labelmask_src, labelmask_dst)
        result["curve_png"] = str(curve_dst)
        result["mask_png"] = str(mask_dst)
        result["labelmask_tif"] = str(labelmask_dst)
    else:
        roi_csv = Path(str(artifacts.get("roi_table_csv", paired_dir / "paired_trace_roi_table.csv")))
        raw_npy = Path(str(artifacts.get("raw_trace_npy", paired_dir / "raw_traces_final_roi_anchor.npy")))
        final_npy = Path(str(artifacts.get("final_trace_npy", paired_dir / "final_traces_final_roi_anchor.npy")))
        final_labelmask = seg_dir / "final" / "roi_selection" / "display_labelmask_uint16.tif"
        if roi_csv.exists() and raw_npy.exists() and final_npy.exists() and final_labelmask.exists():
            roi_rows = []
            for row in _load_csv_rows(roi_csv):
                roi_rows.append({"rank": int(row["rank"]), "rid": int(row["rid"]), "pixel_count": int(float(row.get("pixel_count", 0)))})
            roi_defs = _load_rank_masks_from_labelmask(final_labelmask, roi_rows)
            raw_traces = np.asarray(np.load(str(raw_npy)), dtype=np.float32)
            final_traces = np.asarray(np.load(str(final_npy)), dtype=np.float32)
            result["curve_png"] = _save_paired_trace_plot(raw_traces, final_traces, roi_defs, curve_dst)
            mask_artifacts = _save_paired_mask_plot(roi_defs, mask_dst, labelmask_dst, background_img=background_img)
            result["mask_png"] = mask_artifacts["mask_png"]
            result["labelmask_tif"] = mask_artifacts["labelmask_tif"]
            result["displayed_count"] = int(len(roi_defs))
        else:
            unavailable.append("paired_trace_visual_assets_missing")

    final_analysis_roi_csv = seg_dir / "final" / "roi_selection" / "roi_summary.csv"
    final_analysis_labelmask = seg_dir / "final" / "roi_selection" / "analysis_mask_uint16.tif"
    if final_stack_path is not None and final_analysis_labelmask.exists():
        try:
            heatmap_seed_rows = _load_ranked_roi_rows_from_csv(final_analysis_roi_csv, subset="analysis") if final_analysis_roi_csv.exists() else []
            heatmap_rows = _merge_roi_rows_with_labelmask_ranks(final_analysis_labelmask, heatmap_seed_rows)
            heatmap_roi_defs, final_heatmap_traces = _extract_rank_mean_traces_from_stack(
                final_stack_path,
                final_analysis_labelmask,
                heatmap_rows,
            )
            if len(heatmap_roi_defs) > 0 and final_heatmap_traces.size > 0:
                result["final_corr_png"] = _save_trace_corr_heatmap_png(
                    _compute_trace_corr_matrix(final_heatmap_traces),
                    heatmap_roi_defs,
                    final_corr_dst,
                    f"{DISPLAY_FINAL_NAME} cell correlation heatmap",
                )
                result["final_trace_heatmap_png"], _ = _save_trace_temporal_heatmap_png(
                    final_heatmap_traces,
                    heatmap_roi_defs,
                    final_temporal_dst,
                    f"{DISPLAY_FINAL_NAME} cell temporal heatmap",
                    fps_hz=fps_hz,
                )
                result["all_selected_count"] = int(len(heatmap_roi_defs))
        except Exception as exc:
            unavailable.append(f"paired_trace_final_heatmap_regen_failed:{type(exc).__name__}")

    final_labelmask = seg_dir / "final" / "roi_selection" / "display_labelmask_uint16.tif"
    need_heatmap_fallback = any(
        result[key] is None
        for key in ["raw_corr_png", "final_corr_png", "raw_trace_heatmap_png", "final_trace_heatmap_png"]
    )
    all_roi_csv = Path(str(artifacts.get("all_selected_roi_table_csv", paired_dir / "all_selected_roi_table.csv")))
    all_raw_npy = Path(str(artifacts.get("all_selected_raw_trace_npy", paired_dir / "all_selected_raw_traces_final_roi_anchor.npy")))
    all_final_npy = Path(str(artifacts.get("all_selected_final_trace_npy", paired_dir / "all_selected_final_traces_final_roi_anchor.npy")))
    if need_heatmap_fallback:
        if all_roi_csv.exists() and all_raw_npy.exists() and all_final_npy.exists() and final_labelmask.exists():
            roi_rows = []
            for row in _load_csv_rows(all_roi_csv):
                roi_rows.append({"rank": int(row["rank"]), "rid": int(row["rid"]), "pixel_count": int(float(row.get("pixel_count", 0)))})
            all_roi_defs = _load_rank_masks_from_labelmask(final_labelmask, roi_rows)
            raw_traces_all = np.asarray(np.load(str(all_raw_npy)), dtype=np.float32)
            final_traces_all = np.asarray(np.load(str(all_final_npy)), dtype=np.float32)
            if result["raw_corr_png"] is None:
                result["raw_corr_png"] = _save_trace_corr_heatmap_png(
                    _compute_trace_corr_matrix(raw_traces_all),
                    all_roi_defs,
                    raw_corr_dst,
                    "Raw cell correlation heatmap",
                )
            if result["final_corr_png"] is None:
                result["final_corr_png"] = _save_trace_corr_heatmap_png(
                    _compute_trace_corr_matrix(final_traces_all),
                    all_roi_defs,
                    final_corr_dst,
                    f"{DISPLAY_FINAL_NAME} cell correlation heatmap",
                )
            if result["raw_trace_heatmap_png"] is None:
                result["raw_trace_heatmap_png"], _ = _save_trace_temporal_heatmap_png(
                    raw_traces_all,
                    all_roi_defs,
                    raw_temporal_dst,
                    "Raw cell temporal heatmap",
                )
            if result["final_trace_heatmap_png"] is None:
                result["final_trace_heatmap_png"], _ = _save_trace_temporal_heatmap_png(
                    final_traces_all,
                    all_roi_defs,
                    final_temporal_dst,
                    f"{DISPLAY_FINAL_NAME} cell temporal heatmap",
                    fps_hz=fps_hz,
                )
            result["all_selected_count"] = max(int(result.get("all_selected_count") or 0), int(len(all_roi_defs)))
        elif need_heatmap_fallback:
            unavailable.append("paired_trace_heatmap_assets_missing")

    return result

def _save_motion_curve_png(
    raw_shifts_npy: str | Path | None,
    target_shifts_npy: str | Path | None,
    output_png: str | Path,
    target_label: str,
    pixel_size_um: float | None = None,
    target_display_name: str | None = None,
) -> str:
    fig, ax = plt.subplots(figsize=(7.6, 3.1), dpi=180)
    has_curve = False
    unit_label = "Motion magnitude (px)"
    scale = _safe_float(pixel_size_um)
    if scale is not None and scale > 0:
        unit_label = "Motion magnitude (μm)"
    if raw_shifts_npy and Path(raw_shifts_npy).exists():
        raw = np.asarray(np.load(str(Path(raw_shifts_npy))), dtype=np.float32)
        if raw.ndim == 2 and raw.shape[1] >= 2:
            raw_mag = np.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2)
            if scale is not None and scale > 0:
                raw_mag = raw_mag * float(scale)
            ax.plot(raw_mag, color=RAW_COLOR, linewidth=1.2, label=_reader_label("raw"))
            has_curve = True
    if target_shifts_npy and Path(target_shifts_npy).exists():
        target = np.asarray(np.load(str(Path(target_shifts_npy))), dtype=np.float32)
        if target.ndim == 2 and target.shape[1] >= 2:
            target_mag = np.sqrt(target[:, 0] ** 2 + target[:, 1] ** 2)
            if scale is not None and scale > 0:
                target_mag = target_mag * float(scale)
            label = target_display_name or _reader_label(target_label)
            ax.plot(target_mag, color=MOTION_COLOR if target_label == "motion" else FINAL_COLOR, linewidth=1.2, label=label)
            has_curve = True
    if has_curve:
        ax.legend(loc="upper right", frameon=False)
        ax.set_xlabel("Frame")
        ax.set_ylabel(unit_label)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Rigid motion magnitude across frames", loc="left", pad=4)
    return _save_figure(fig, output_png)


def _copy_if_exists(src: str | Path | None, dst: str | Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def _rel(path: str | Path | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def _fmt(value: Any, fmt: str = "{:.3g}") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    try:
        return fmt.format(float(value))
    except Exception:
        return str(value)


def _kv(label: str, value: Any) -> dict[str, Any]:
    return {"label": label, "value": value}


def _px_to_um(value_px: Any, pixel_size_um: float | None) -> float | None:
    value = _safe_float(value_px)
    scale = _safe_float(pixel_size_um)
    if value is None or scale is None:
        return None
    return float(value) * float(scale)


def _load_stage_metrics_from_comparison(comparison: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if not comparison:
        return None, None
    metrics_json = comparison.get("artifact_paths", {}).get("final_artifacts", {}).get("metrics_json")
    if not metrics_json:
        return None, None
    return _read_json(metrics_json), metrics_json


def _collect_inventory(run_root: Path, manifest: dict[str, Any], report_assets: dict[str, Any]) -> dict[str, Any]:
    required = {
        "pipeline_manifest": run_root / "manifests" / "pipeline_manifest.json",
        "final_used_params": run_root / "final_used_params.json",
        "final_stack_sidecar": run_root / "final" / "final_stack_sidecar.json",
        "input_metrics": Path(str(manifest.get("raw_metrics_json") or "")),
        "iter0_raw_vs_denoise": run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_denoise.json",
        "iter0_raw_vs_motion": run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_motion.json",
        "iter1_raw_vs_final": run_root / "iterations" / "iter_1" / "metrics" / "comparison_raw_vs_final.json",
        "seg_backend_status": run_root / "segmentation" / "backend_status.json",
        "seg_run_status": run_root / "segmentation" / "run_status.json",
        "seg_summary": run_root / "segmentation" / "summary.json",
        "seg_comparison": run_root / "segmentation" / "downstream_comparison.json",
    }
    required_jsons = {key: {"path": str(path), "found": bool(path and Path(path).exists())} for key, path in required.items()}
    optional = {key: {"path": str(value) if value else None, "found": bool(value and Path(value).exists())} for key, value in report_assets.items()}
    return {"required_jsons": required_jsons, "report_assets": optional}


def _file_to_data_uri(path: str | Path) -> str | None:
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    suffix = path_obj.suffix.lower()
    mime = "image/png"
    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    raw = path_obj.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _render_panel_html(panel: dict[str, Any], report_dir: Path, embed_assets: bool) -> str:
    title = html.escape(str(panel.get("title", "Panel")))
    label = html.escape(str(panel.get("label", "")))
    note = html.escape(str(panel.get("note", ""))) if panel.get("note") else ""
    source_raw = str(panel.get("source", "")) if panel.get("source") else ""
    source = html.escape(_truncate_middle(source_raw, max_chars=84)) if source_raw else ""
    source_title = html.escape(source_raw) if source_raw else ""
    path = panel.get("path")
    if path and Path(report_dir / path).exists():
        img_src = _file_to_data_uri(report_dir / path) if embed_assets else html.escape(path)
        img_html = f'<img src="{img_src}" alt="{title}">' if img_src else '<div class="panel-na">N/A</div>'
    else:
        img_html = '<div class="panel-na">N/A</div>'
    meta_bits = []
    if note:
        meta_bits.append(f'<div class="panel-note">{note}</div>')
    if source:
        meta_bits.append(f'<div class="panel-source" title="{source_title}">Source: {source}</div>')
    return f'<figure class="panel"><div class="panel-label">{label}</div>{img_html}<figcaption><div class="panel-title">{title}</div>{"".join(meta_bits)}</figcaption></figure>'


def _render_kv_grid(items: Sequence[dict[str, Any]], extra_class: str = "") -> str:
    blocks = []
    for item in items:
        full_value = _stringify_value(item["value"])
        shown_value = _truncate_middle(full_value, max_chars=int(item.get("max_chars", 96)))
        blocks.append(
            f'<div class="kv-item"><div class="kv-label">{html.escape(str(item["label"]))}</div>'
            f'<div class="kv-value" title="{html.escape(full_value)}">{html.escape(shown_value)}</div></div>'
        )
    cls = f"kv-grid {extra_class}".strip()
    return f'<div class="{cls}">' + "".join(blocks) + "</div>"


def _render_section_html(section: dict[str, Any], report_dir: Path, embed_assets: bool) -> str:
    groups_html = []
    for group in section.get("panel_groups", []):
        cls = html.escape(str(group.get("layout", "grid-2")))
        groups_html.append(f'<div class="panel-grid {cls}">' + ''.join(_render_panel_html(panel, report_dir, embed_assets=embed_assets) for panel in group.get("panels", [])) + '</div>')
    metric_style = str(section.get("metric_style") or "").strip()
    metrics_html = _render_kv_grid(section.get("mini_metrics", []), extra_class=metric_style) if section.get("mini_metrics") else ''
    caption_html = f'<div class="section-caption">{html.escape(str(section.get("caption", "")))}</div>' if section.get("caption") else ''
    return f'<section class="report-section"><div class="section-header"><div class="section-kicker">{html.escape(str(section.get("kicker", "Figure section")))}</div><h2>{html.escape(str(section.get("title", "Section")))}</h2></div>{"".join(groups_html)}{metrics_html}{caption_html}</section>'


def _render_report_html(report_data: dict[str, Any], report_dir: Path, print_mode: bool = False) -> str:
    title = html.escape(str(report_data.get("report_title") or REPORT_TITLE))
    embed_assets = bool(report_data.get("asset_embedding_status", {}).get("assets_embedded", False))
    sections_html = ''.join(_render_section_html(section, report_dir, embed_assets=embed_assets) for section in report_data.get("sections", []))
    header_kvs = _render_kv_grid([
        _kv("Folder tag", report_data["dataset_identity"].get("folder_tag", "N/A")),
        _kv("Cell-type data", _yes_no_label(_coerce_bool_flag(report_data["dataset_identity"].get("is_cell_data")))),
        _kv("Dataset profile", report_data["dataset_identity"].get("dataset_profile", "N/A")),
        _kv("Representative frame", report_data["representative_frames"].get("raw_peak_activity_frame", "N/A")),
        _kv("Imaging modality", report_data.get("imaging_modality_used", "N/A")),
        _kv("Colormap", report_data.get("colormap_used", "N/A")),
        _kv("Pixel size", f"{_fmt(report_data.get('pixel_size_um_used'))} um/pixel"),
        _kv("Frame rate", f"{_fmt(report_data.get('fps_hz_used'))} Hz"),
    ], extra_class="header-strip")
    body_class = "print-mode" if print_mode else "web-mode"
    defaults_note = html.escape(str(report_data.get("defaults_note") or ""))
    style = f"""
    :root {{--raw:{RAW_COLOR};--denoise:{DENOISE_COLOR};--motion:{MOTION_COLOR};--final:{FINAL_COLOR};--border:{BORDER_COLOR};--muted:{TEXT_MUTED};--bg:#ffffff;}}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; padding:0; background:var(--bg); color:#111111; font-family:{REPORT_FONT_CSS}; }}
    .report, .report * {{ overflow-wrap:anywhere; word-break:break-word; white-space:normal; }}
    .report {{ width:min(1420px,95vw); margin:0 auto; padding:18px 18px 28px; }}
    .report-header {{ border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:20px; }}
    .report-header h1 {{ margin:0 0 6px; font-size:30px; line-height:1.12; font-weight:700; }}
    .report-sub {{ color:var(--muted); font-size:13px; margin-bottom:10px; }}
    .report-note {{ margin-top:10px; color:#333333; font-size:12px; line-height:1.4; }}
    .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; font-family:{REPORT_FONT_CSS}; font-size:12px; }}
    .legend span::before {{ content:''; display:inline-block; width:14px; height:14px; margin-right:6px; vertical-align:-2px; border:1px solid rgba(0,0,0,0.08); }}
    .legend .raw::before {{ background:var(--raw); }} .legend .denoise::before {{ background:var(--denoise); }} .legend .motion::before {{ background:var(--motion); }} .legend .final::before {{ background:var(--final); }}
    .kv-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:14px 0 0; }}
    .kv-item {{ border:1px solid var(--border); padding:8px 10px; min-height:58px; background:#fff; }}
    .kv-label {{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px; font-family:{REPORT_FONT_CSS}; }}
    .kv-value {{ font-size:14px; line-height:1.28; font-family:{REPORT_FONT_CSS}; }}
    .header-strip .kv-item {{ min-height:64px; }}
    .metric-hero {{ grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
    .metric-hero .kv-item {{ min-height:84px; padding:12px 14px; }}
    .metric-hero .kv-label {{ font-size:11px; }}
    .metric-hero .kv-value {{ font-size:19px; font-weight:600; line-height:1.18; }}
    .metric-compact .kv-item {{ min-height:56px; }}
    .metric-compact .kv-value {{ font-size:13px; }}
    .report-section {{ padding:16px 0 18px; border-bottom:1px solid var(--border); break-inside:avoid; page-break-inside:avoid; }}
    .section-header {{ margin-bottom:10px; }}
    .section-kicker {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; font-family:{REPORT_FONT_CSS}; }}
    .section-header h2 {{ margin:4px 0 0; font-size:23px; line-height:1.12; font-weight:700; }}
    .panel-grid {{ display:grid; gap:10px; margin-top:10px; align-items:start; }} .grid-3 {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .grid-2 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid-1 {{ grid-template-columns:repeat(1,minmax(0,1fr)); }}
    .panel {{ margin:0; border:1px solid var(--border); padding:7px; position:relative; background:#ffffff; }}
    .panel img,.panel .panel-na {{ width:100%; display:block; background:#f7f7f7; }}
    .panel-na {{ min-height:240px; display:flex; align-items:center; justify-content:center; color:var(--muted); font-family:{REPORT_FONT_CSS}; }}
    .panel-label {{ position:absolute; top:9px; left:10px; font-family:{REPORT_FONT_CSS}; font-weight:700; font-size:15px; background:rgba(255,255,255,0.95); padding:1px 6px; border:1px solid var(--border); }}
    .panel-title {{ font-family:{REPORT_FONT_CSS}; font-size:13px; font-weight:700; margin-top:7px; line-height:1.25; }}
    .panel-note,.panel-source {{ font-family:{REPORT_FONT_CSS}; font-size:11px; color:var(--muted); margin-top:3px; line-height:1.35; }}
    .section-caption {{ margin-top:10px; font-size:12.5px; line-height:1.45; color:#1b1b1b; font-family:{REPORT_FONT_CSS}; }}
    details.provenance-block {{ margin-top:10px; border:1px solid var(--border); padding:8px 10px; }}
    details.provenance-block summary {{ cursor:pointer; font-weight:700; }}
    .print-mode .report {{ width:100%; padding:7mm 6mm 9mm; }}
    .print-mode .report-section {{ padding:12px 0 16px; }}
    @media (max-width: 1050px) {{ .grid-3,.grid-2 {{ grid-template-columns:1fr; }} .report {{ width:min(98vw,860px); padding:18px 14px 26px; }} }}
    @media print {{ .report {{ width:100%; padding:7mm 6mm 9mm; }} .report-section {{ page-break-inside:avoid; break-inside:avoid; }} }}
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{style}</style></head><body class='{body_class}'><div class='report'><header class='report-header'><h1>{REPORT_TITLE}</h1><div class='report-sub'>{html.escape(str(report_data['dataset_identity'].get('folder_name', '')))}</div><div class='legend'><span class='raw'>{_reader_label('raw')}</span><span class='final'>{DISPLAY_FINAL_NAME}</span></div>{header_kvs}<div class='report-note'>{defaults_note}</div></header>{sections_html}</div></body></html>"

def _try_generate_pdf(print_html_path: Path, pdf_out: Path) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    try:
        import weasyprint  # type: ignore

        weasyprint.HTML(filename=str(print_html_path)).write_pdf(str(pdf_out))
        if pdf_out.exists():
            return {"generated": True, "path": str(pdf_out), "engine": "weasyprint", "attempts": attempts}
    except Exception as exc:
        attempts.append({"engine": "weasyprint", "error": str(exc)})
    candidates = []
    for name in ["msedge", "chrome", "google-chrome", "chromium"]:
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates.extend([
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ])
    for candidate in candidates:
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            continue
        user_data_dir = pdf_out.parent / "edge_pdf_profile"
        user_data_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(candidate_path),
            "--headless",
            "--disable-gpu",
            "--allow-file-access-from-files",
            "--no-pdf-header-footer",
            f"--user-data-dir={user_data_dir}",
            f"--print-to-pdf={pdf_out}",
            print_html_path.resolve().as_uri(),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            if completed.returncode == 0 and pdf_out.exists():
                return {
                    "generated": True,
                    "path": str(pdf_out),
                    "engine": str(candidate_path),
                    "stderr": completed.stderr.strip(),
                    "attempts": attempts,
                }
            attempts.append(
                {
                    "engine": str(candidate_path),
                    "returncode": int(completed.returncode),
                    "stderr": completed.stderr.strip(),
                }
            )
        except Exception as exc:
            attempts.append({"engine": str(candidate_path), "error": str(exc)})
    try:
        wk = shutil.which("wkhtmltopdf")
        if wk:
            completed = subprocess.run(
                [wk, str(print_html_path), str(pdf_out)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
            if completed.returncode == 0 and pdf_out.exists():
                return {"generated": True, "path": str(pdf_out), "engine": wk, "stderr": completed.stderr.strip(), "attempts": attempts}
            attempts.append({"engine": wk, "returncode": int(completed.returncode), "stderr": completed.stderr.strip()})
    except Exception as exc:
        attempts.append({"engine": "wkhtmltopdf", "error": str(exc)})
    return {
        "generated": False,
        "path": None,
        "engine": None,
        "error": "no_pdf_engine_succeeded",
        "attempts": attempts,
        "install_hint": "Install WeasyPrint or ensure Microsoft Edge/Chrome headless print-to-pdf is accessible.",
    }


def build_deterministic_report(
    run_root: str | Path,
    output_dir: str | Path | None = None,
    try_pdf: bool = True,
    generate_overview_pngs: bool = False,
    imaging_modality: str = DEFAULT_IMAGING_MODALITY,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    fps_hz: float = DEFAULT_FPS_HZ,
    display_name_final: str = DISPLAY_FINAL_NAME,
    report_embed_assets: bool = True,
    report_inline_css: bool = True,
    report_generate_pdf: bool = True,
    report_crop_scale_factor: float = REPORT_CROP_SCALE_FACTOR,
    report_kymograph_line_count: int = REPORT_KYMOGRAPH_LINE_COUNT,
    report_use_intermediate_sections: bool = False,
) -> dict[str, Any]:
    del generate_overview_pngs
    run_root = Path(run_root).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else (run_root / "report")
    assets_dir = report_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    unavailable: list[str] = []

    imaging_modality_used = _normalize_modality(imaging_modality)
    display_name_final = str(display_name_final or DISPLAY_FINAL_NAME)
    report_generate_pdf = bool(report_generate_pdf and try_pdf)
    report_kymograph_line_count = max(1, int(report_kymograph_line_count))

    manifest = _read_json(run_root / "manifests" / "pipeline_manifest.json") or {}
    final_used_params = _read_json(run_root / "final_used_params.json") or {}
    final_sidecar = _read_json(run_root / "final" / "final_stack_sidecar.json") or {}
    raw_metrics = _read_json(manifest.get("raw_metrics_json") or "") if manifest.get("raw_metrics_json") else None
    if raw_metrics is None:
        unavailable.append("input_metrics_missing")
        raw_metrics = {}
    comp_denoise = _read_json(run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_denoise.json") or {}
    comp_motion = _read_json(run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_motion.json") or {}
    comp_final = _read_json(run_root / "iterations" / "iter_1" / "metrics" / "comparison_raw_vs_final.json") or {}
    seg_summary = _read_json(run_root / "segmentation" / "summary.json") or {}
    seg_comp = _read_json(run_root / "segmentation" / "downstream_comparison.json") or {}
    raw_sel = _read_json(run_root / "segmentation" / "raw" / "roi_selection" / "selection_summary.json") or {}
    final_sel = _read_json(run_root / "segmentation" / "final" / "roi_selection" / "selection_summary.json") or {}

    final_metrics, _ = _load_stage_metrics_from_comparison(comp_final)
    raw_stack_path = Path(str(raw_metrics.get("input_file") or ""))
    final_stack_path = Path(str(final_sidecar.get("final_stack_path") or (final_metrics or {}).get("input_file") or ""))
    if not raw_stack_path.exists():
        raise FileNotFoundError(f"Raw stack not found for report build: {raw_stack_path}")
    if not final_stack_path.exists():
        raise FileNotFoundError(f"Final stack not found for report build: {final_stack_path}")

    def _maybe_read_projection(path_like: str | Path | None, fallback: np.ndarray | None = None) -> np.ndarray:
        try:
            if path_like and Path(path_like).exists():
                return _read_2d_tif(path_like)
        except Exception:
            pass
        if fallback is not None:
            return np.asarray(fallback, dtype=np.float32)
        raise FileNotFoundError(f"Projection image unavailable: {path_like}")

    rep_info = select_representative_frame(raw_stack_path)
    rep_frame = int(rep_info["frame_index"])
    raw_rep_frame = _read_frame(raw_stack_path, rep_frame)
    final_rep_frame = _read_frame(final_stack_path, rep_frame)
    raw_corr_info = _compute_frame_to_mean_projection_correlation_curve(raw_stack_path)
    final_corr_info = _compute_frame_to_mean_projection_correlation_curve(final_stack_path)

    raw_projection_fallback = {"std": np.asarray(rep_info["_std_map"], dtype=np.float32), "mip": None}
    final_projection_fallback = None
    try:
        raw_std_img = _maybe_read_projection(raw_metrics.get("artifacts", {}).get("std_tif"), fallback=raw_projection_fallback["std"])
    except Exception:
        raw_std_img = np.asarray(rep_info["_std_map"], dtype=np.float32)
        unavailable.append("raw_std_projection_fallback")
    try:
        raw_mip_img = _maybe_read_projection(raw_metrics.get("artifacts", {}).get("mip_tif"))
    except Exception:
        raw_projection_fallback = _compute_std_and_mip_projection(raw_stack_path)
        raw_mip_img = np.asarray(raw_projection_fallback["mip"], dtype=np.float32)
        unavailable.append("raw_mip_projection_fallback")
    try:
        final_std_img = _maybe_read_projection(comp_final.get("artifact_paths", {}).get("final_artifacts", {}).get("std_tif"))
    except Exception:
        final_projection_fallback = _compute_std_and_mip_projection(final_stack_path)
        final_std_img = np.asarray(final_projection_fallback["std"], dtype=np.float32)
        unavailable.append("final_std_projection_fallback")
    try:
        final_mip_img = _maybe_read_projection(comp_final.get("artifact_paths", {}).get("final_artifacts", {}).get("mip_tif"))
    except Exception:
        if final_projection_fallback is None:
            final_projection_fallback = _compute_std_and_mip_projection(final_stack_path)
        final_mip_img = np.asarray(final_projection_fallback["mip"], dtype=np.float32)
        unavailable.append("final_mip_projection_fallback")

    downstream_manifest = manifest.get("downstream", {}) if isinstance(manifest.get("downstream"), dict) else {}
    dataset_profile = seg_summary.get("dataset_profile") or downstream_manifest.get("dataset_profile") or manifest.get("dataset_profile")
    is_cell_data = _infer_is_cell_data(manifest, dataset_profile)
    cell_downstream_enabled = _coerce_bool_flag(downstream_manifest.get("cell_downstream_enabled"))
    include_downstream_section = True
    downstream_section_omitted_reason = None
    if is_cell_data is False:
        include_downstream_section = False
        downstream_section_omitted_reason = "non_cell_data"
    elif cell_downstream_enabled is False:
        include_downstream_section = False
        downstream_section_omitted_reason = "downstream_disabled"

    actual_pixel_size = _safe_float(raw_metrics.get("data_summary", {}).get("pixel_size_um"))
    pixel_size_used = float(actual_pixel_size) if actual_pixel_size is not None else float(pixel_size_um)
    pixel_size_default_used = actual_pixel_size is None
    actual_fps = _safe_float(raw_metrics.get("data_summary", {}).get("frame_rate_hz"))
    fps_hz_used = float(actual_fps) if actual_fps is not None else float(fps_hz)
    fps_default_used = actual_fps is None
    defaults_note_parts = [f"Imaging modality: {imaging_modality_used} ({_colormap_label(imaging_modality_used)} colormap for intensity panels)."]
    if pixel_size_default_used:
        defaults_note_parts.append(f"Pixel size defaulted to {pixel_size_used:.3f} μm/pixel.")
    if fps_default_used:
        defaults_note_parts.append(f"Frame rate defaulted to {fps_hz_used:.1f} Hz.")
    defaults_note = " ".join(defaults_note_parts)

    roi_mask = None
    analysis_mask_path = _optional_existing_file(final_sel.get("artifacts", {}).get("analysis_mask_uint16_tif"))
    if bool(is_cell_data) and analysis_mask_path is not None:
        roi_mask = _read_2d_tif(analysis_mask_path) > 0
    zoom_regions = _select_zoom_regions(
        rep_info["_std_map"],
        raw_rep_frame,
        roi_mask,
        max_regions=max(2, report_kymograph_line_count),
        scale_factor=float(report_crop_scale_factor),
    )
    if not zoom_regions:
        full_h, full_w = raw_rep_frame.shape
        zoom_regions = [{"name": "Main crop", "x0": 0, "y0": 0, "x1": full_w, "y1": full_h, "selection_rule": "full_frame_fallback", "shrink_factor": float(report_crop_scale_factor)}]
    roi_crop_region = dict(zoom_regions[0])
    roi_crop_region["name"] = "Main comparison crop"

    line_regions = [dict(region) for region in zoom_regions[:report_kymograph_line_count]]
    while len(line_regions) < report_kymograph_line_count:
        base = dict(roi_crop_region)
        height = int(base["y1"]) - int(base["y0"])
        shift = int(round(height * 0.35 * len(line_regions)))
        y0 = max(0, min(raw_rep_frame.shape[0] - height, int(base["y0"]) + shift))
        base["y0"] = int(y0)
        base["y1"] = int(y0 + height)
        base["name"] = f"Derived line region {len(line_regions)+1}"
        line_regions.append(base)
    kymograph_lines = []
    for idx, region in enumerate(line_regions[:report_kymograph_line_count], start=1):
        coords = _select_line_coords(region)
        coords["line_index"] = idx
        coords["selection_rule"] = f"horizontal_midline_of_{region.get('name', f'region_{idx}') }"
        kymograph_lines.append(coords)

    if raw_metrics.get("artifacts", {}).get("rigid_shifts_npy"):
        motion_info = select_motion_burst_triplet(raw_metrics["artifacts"]["rigid_shifts_npy"], _stack_info(raw_stack_path)["num_frames"])
    else:
        motion_info = {"center_frame": None, "delta": None, "triplet_frames": [], "selection_rule": "unavailable"}
        unavailable.append("raw_rigid_shifts_missing")

    panel_index = 0

    def panel(title: str, path: str | None, source: str | None = None, note: str | None = None) -> dict[str, Any]:
        nonlocal panel_index
        payload = {
            "label": _panel_label(panel_index),
            "title": title,
            "path": _rel(path, report_dir) if path else None,
            "source": source,
            "note": note,
        }
        panel_index += 1
        return payload

    input_dir = assets_dir / "input"
    integrated_dir = assets_dir / "integrated_final"
    down_dir = assets_dir / "downstream"
    for d in [input_dir, integrated_dir, down_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def _crop_array(img: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
        return np.asarray(img[int(crop["y0"]):int(crop["y1"]), int(crop["x0"]):int(crop["x1"])], dtype=np.float32)

    crop_rectangle = {
        "x0": int(roi_crop_region["x0"]),
        "y0": int(roi_crop_region["y0"]),
        "x1": int(roi_crop_region["x1"]),
        "y1": int(roi_crop_region["y1"]),
        "edgecolor": "#ffffff",
        "linewidth": 0.9,
    }

    raw_input_overview_png = save_intensity_grid_png(
        [[
            {"image": raw_rep_frame, "title": f"{_reader_label('raw')} single frame", "rectangles": [crop_rectangle]},
            {"image": raw_std_img, "title": f"{_reader_label('raw')} STD", "rectangles": [crop_rectangle]},
            {"image": raw_mip_img, "title": f"{_reader_label('raw')} MIP", "rectangles": [crop_rectangle]},
        ]],
        input_dir / "raw_input_overview_panel.png",
        title="Raw input overview",
        pixel_size_um=pixel_size_used,
        imaging_modality=imaging_modality_used,
    )
    main_comparison_png = save_intensity_grid_png(
        [
            [
                {"image": raw_rep_frame, "title": f"{_reader_label('raw')} single frame", "rectangles": [crop_rectangle]},
                {"image": raw_std_img, "title": f"{_reader_label('raw')} STD", "rectangles": [crop_rectangle]},
                {"image": raw_mip_img, "title": f"{_reader_label('raw')} MIP", "rectangles": [crop_rectangle]},
            ],
            [
                {"image": final_rep_frame, "title": f"{display_name_final} single frame", "rectangles": [crop_rectangle]},
                {"image": final_std_img, "title": f"{display_name_final} STD", "rectangles": [crop_rectangle]},
                {"image": final_mip_img, "title": f"{display_name_final} MIP", "rectangles": [crop_rectangle]},
            ],
        ],
        integrated_dir / "raw_vs_neuropilot_main_grid.png",
        title=f"{_reader_label('raw')} vs {display_name_final} main comparison",
        row_labels=[_reader_label("raw"), display_name_final],
        pixel_size_um=pixel_size_used,
        imaging_modality=imaging_modality_used,
    )
    crop_comparison_png = save_intensity_grid_png(
        [
            [
                {"image": _crop_array(raw_rep_frame, roi_crop_region), "title": f"{_reader_label('raw')} single crop"},
                {"image": _crop_array(raw_std_img, roi_crop_region), "title": f"{_reader_label('raw')} STD crop"},
                {"image": _crop_array(raw_mip_img, roi_crop_region), "title": f"{_reader_label('raw')} MIP crop"},
            ],
            [
                {"image": _crop_array(final_rep_frame, roi_crop_region), "title": f"{display_name_final} single crop"},
                {"image": _crop_array(final_std_img, roi_crop_region), "title": f"{display_name_final} STD crop"},
                {"image": _crop_array(final_mip_img, roi_crop_region), "title": f"{display_name_final} MIP crop"},
            ],
        ],
        integrated_dir / "raw_vs_neuropilot_crop_grid.png",
        title="Matched ROI crop comparison",
        row_labels=[_reader_label("raw"), display_name_final],
        pixel_size_um=pixel_size_used,
        imaging_modality=imaging_modality_used,
    )
    kymograph_bundle_png = save_kymograph_bundle_png(
        final_std_img,
        raw_stack_path,
        final_stack_path,
        kymograph_lines[:2],
        integrated_dir / "kymograph_bundle.png",
        pixel_size_um=pixel_size_used,
        fps_hz=fps_hz_used,
        imaging_modality=imaging_modality_used,
    )
    corr_curve_csv = integrated_dir / "integrated_correlation_curves.csv"
    with open(corr_curve_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "raw_correlation", "neuropilot_correlation"])
        for idx in range(max(len(raw_corr_info["curve"]), len(final_corr_info["curve"]))):
            raw_v = raw_corr_info["curve"][idx] if idx < len(raw_corr_info["curve"]) else np.nan
            fin_v = final_corr_info["curve"][idx] if idx < len(final_corr_info["curve"]) else np.nan
            writer.writerow([idx, "" if not np.isfinite(raw_v) else float(raw_v), "" if not np.isfinite(fin_v) else float(fin_v)])
    corr_panel_png = save_correlation_curve_panel(
        np.asarray(raw_corr_info["curve"], dtype=np.float32),
        np.asarray(final_corr_info["curve"], dtype=np.float32),
        integrated_dir / "integrated_correlation_panel.png",
        title=f"{_reader_label('raw')} vs {display_name_final} frame-to-mean correlation",
    )
    metric_bar_specs = [
        {
            "label": "SNR proxy",
            "raw": comp_final.get("snr_before_after", {}).get("raw_snr"),
            "candidate": comp_final.get("snr_before_after", {}).get("final_snr"),
            "candidate_label": display_name_final,
            "candidate_color": FINAL_COLOR,
            "unit": "SNR",
            "value_fmt": "{:.2f}",
        },
        {
            "label": "Motion mean",
            "raw": _px_to_um(comp_final.get("motion_before_after", {}).get("raw", {}).get("motion_mean_px"), pixel_size_used),
            "candidate": _px_to_um(comp_final.get("motion_before_after", {}).get("final", {}).get("motion_mean_px"), pixel_size_used),
            "candidate_label": display_name_final,
            "candidate_color": FINAL_COLOR,
            "unit": "μm",
            "value_fmt": "{:.2f}",
            "delta_text": "Converted from rigid shift px",
        },
        {
            "label": "Motion p95",
            "raw": _px_to_um(comp_final.get("motion_before_after", {}).get("raw", {}).get("motion_p95_px"), pixel_size_used),
            "candidate": _px_to_um(comp_final.get("motion_before_after", {}).get("final", {}).get("motion_p95_px"), pixel_size_used),
            "candidate_label": display_name_final,
            "candidate_color": FINAL_COLOR,
            "unit": "μm",
            "value_fmt": "{:.2f}",
            "delta_text": "Converted from rigid shift px",
        },
        {
            "label": "Bleaching drop",
            "raw": comp_final.get("bleaching_before_after", {}).get("raw_relative_drop_percent"),
            "candidate": comp_final.get("bleaching_before_after", {}).get("final_relative_drop_percent"),
            "candidate_label": display_name_final,
            "candidate_color": FINAL_COLOR,
            "unit": "%",
            "value_fmt": "{:.2f}",
        },
    ]
    metric_bar_png = save_metric_comparison_panel(
        metric_bar_specs,
        integrated_dir / "integrated_metric_bar_panel.png",
        title=f"{_reader_label('raw')} vs {display_name_final} quantitative comparison",
    )
    raw_shifts_npy = comp_final.get("artifact_paths", {}).get("raw_artifacts", {}).get("rigid_shifts_npy")
    final_shifts_npy = comp_final.get("artifact_paths", {}).get("final_artifacts", {}).get("rigid_shifts_npy")
    motion_curve_png = _save_motion_curve_png(
        raw_shifts_npy,
        final_shifts_npy,
        integrated_dir / "integrated_motion_curve_um.png",
        target_label="final",
        pixel_size_um=pixel_size_used,
        target_display_name=display_name_final,
    )

    raw_analysis_mask = run_root / "segmentation" / "raw" / "roi_selection" / "analysis_mask_uint16.tif"
    final_analysis_mask = run_root / "segmentation" / "final" / "roi_selection" / "analysis_mask_uint16.tif"
    raw_roi_csv = run_root / "segmentation" / "raw" / "roi_selection" / "roi_summary.csv"
    final_roi_csv = run_root / "segmentation" / "final" / "roi_selection" / "roi_summary.csv"
    raw_overlay_info = {"png": None, "selected_count": 0, "selected_ranks": []}
    final_overlay_info = {"png": None, "selected_count": 0, "selected_ranks": []}
    paired_assets = {
        "summary": None,
        "curve_png": None,
        "mask_png": None,
        "labelmask_tif": None,
        "raw_corr_png": None,
        "final_corr_png": None,
        "raw_trace_heatmap_png": None,
        "final_trace_heatmap_png": None,
        "displayed_count": 0,
        "all_selected_count": 0,
    }
    down_count_png = None
    if include_downstream_section:
        raw_overlay_info = save_segmentation_overlay_png(
            raw_std_img,
            raw_analysis_mask,
            raw_roi_csv,
            down_dir / "segmentation_overlay_raw.png",
            "Raw segmentation overview (STD background)",
            REPORT_DISPLAY_TOP_PERCENT,
            pixel_size_um=pixel_size_used,
        ) if raw_analysis_mask.exists() else {"png": save_unavailable_panel(down_dir / "segmentation_overlay_raw.png", "Raw segmentation overview", "Analysis mask missing"), "selected_count": 0, "selected_ranks": []}
        final_overlay_info = save_segmentation_overlay_png(
            final_std_img,
            final_analysis_mask,
            final_roi_csv,
            down_dir / "segmentation_overlay_NeuroPilot.png",
            f"{display_name_final} segmentation overview (STD background)",
            REPORT_DISPLAY_TOP_PERCENT,
            pixel_size_um=pixel_size_used,
        ) if final_analysis_mask.exists() else {"png": save_unavailable_panel(down_dir / "segmentation_overlay_NeuroPilot.png", f"{display_name_final} segmentation overview", "Analysis mask missing"), "selected_count": 0, "selected_ranks": []}
        paired_assets = _generate_paired_trace_assets(
            run_root / "segmentation",
            down_dir,
            unavailable,
            background_img=final_std_img,
            fps_hz=fps_hz_used,
            final_stack_path=final_stack_path,
        )
        down_count_png = save_downstream_count_panel(
            [
                {"label": "plane0 total", "raw": seg_summary.get("suite2p_counts", {}).get("raw_plane0_total"), "final": seg_summary.get("suite2p_counts", {}).get("final_plane0_total")},
                {"label": "accepted ROI", "raw": seg_summary.get("suite2p_counts", {}).get("raw_after_cc_area_filter"), "final": seg_summary.get("suite2p_counts", {}).get("final_after_cc_area_filter")},
                {"label": "display selected", "raw": seg_summary.get("suite2p_counts", {}).get("raw_display_selected_count"), "final": seg_summary.get("suite2p_counts", {}).get("final_display_selected_count")},
                {"label": "trace count", "raw": seg_comp.get("trace_count_raw"), "final": seg_comp.get("trace_count_final")},
            ],
            down_dir / "downstream_count_panel.png",
            title="Downstream quantitative comparison",
            final_display_name=display_name_final,
        )

    report_assets = {
        "raw_input_overview_png": raw_input_overview_png,
        "main_comparison_png": main_comparison_png,
        "crop_comparison_png": crop_comparison_png,
        "kymograph_bundle_png": kymograph_bundle_png,
        "metric_bar_png": metric_bar_png,
        "motion_curve_png": motion_curve_png,
        "corr_panel_png": corr_panel_png,
        "raw_seg_png": raw_overlay_info["png"],
        "final_seg_png": final_overlay_info["png"],
        "down_count_png": down_count_png,
        "paired_curve_png": paired_assets.get("curve_png"),
        "paired_numbered_map_png": paired_assets.get("mask_png"),
        "paired_labelmask_tif": paired_assets.get("labelmask_tif"),
        "raw_corr_heatmap_png": paired_assets.get("raw_corr_png"),
        "final_corr_heatmap_png": paired_assets.get("final_corr_png"),
        "raw_trace_heatmap_png": paired_assets.get("raw_trace_heatmap_png"),
        "final_trace_heatmap_png": paired_assets.get("final_trace_heatmap_png"),
        "corr_curve_csv": str(corr_curve_csv),
    }

    sections = [
        {
            "kicker": "Section A",
            "title": "A. Input Data Overview",
            "panel_groups": [
                {"layout": "grid-1", "panels": [panel("Raw single frame / STD / MIP", raw_input_overview_png, source=str(raw_stack_path), note=f"Representative frame = {rep_frame}; ROI box marks the crop shown below; {_colormap_label(imaging_modality_used)} colormap")]},
            ],
            "metric_style": "metric-hero",
            "mini_metrics": [
                _kv("Shape", raw_metrics.get("data_summary", {}).get("shape_thw", "N/A")),
                _kv("dtype", raw_metrics.get("data_summary", {}).get("dtype", "N/A")),
                _kv("fps", f"{_fmt(fps_hz_used)} Hz"),
                _kv("pixel size", f"{_fmt(pixel_size_used)} μm/pixel"),
                _kv("dynamic range", _fmt(raw_metrics.get("data_summary", {}).get("dynamic_range", {}).get("robust_range_p99_minus_p01"))),
                _kv("bleaching drop %", _fmt(raw_metrics.get("bleaching_trend", {}).get("relative_drop_percent"))),
                _kv("SNR proxy", _fmt(raw_metrics.get("snr_metric", {}).get("snr"))),
                _kv("motion mean px", _fmt(raw_metrics.get("rigid_motion_metric", {}).get("rigid_motion_summary", {}).get("motion_mean_px"))),
            ],
            "caption": "Raw input quality is summarized with the representative single frame, STD, and MIP. The ROI box shows the crop location reused later in the comparison panel. Scale bars use the configured pixel size and all intensity displays follow the selected imaging modality colormap.",
        },
        {
            "kicker": "Section B",
            "title": f"B. Integrated Improvement: Raw vs {display_name_final}",
            "panel_groups": [
                {"layout": "grid-1", "panels": [panel(f"{_reader_label('raw')} vs {display_name_final} main grid (2x3)", main_comparison_png, source=str(final_stack_path), note="Top row = Raw; bottom row = NeuroPilot; white ROI boxes mark the crop location")]},
                {"layout": "grid-1", "panels": [panel("Matched ROI crop grid (2x3)", crop_comparison_png, source=str(final_stack_path), note=f"Crop coordinates = ({roi_crop_region['x0']},{roi_crop_region['y0']})-({roi_crop_region['x1']},{roi_crop_region['y1']})")]},
                {"layout": "grid-1", "panels": [panel("Kymograph group", kymograph_bundle_png, source=str(final_stack_path), note="NeuroPilot STD line map + Raw/NeuroPilot kymographs for line 1 and line 2")]},
                {"layout": "grid-2", "panels": [
                    panel("Raw vs NeuroPilot quantitative bars", metric_bar_png, source=str(run_root / "iterations" / "iter_1" / "metrics" / "comparison_raw_vs_final.json"), note="Fixed-ROI SNR proxy, motion mean, motion p95 and bleaching are shown as paired bars"),
                    panel("Motion curve", motion_curve_png, source=str(final_shifts_npy or raw_shifts_npy or ""), note="Frame-wise rigid motion magnitude converted to μm using the configured pixel size"),
                ]},
                {"layout": "grid-1", "panels": [panel("Correlation curve", corr_panel_png, source=str(corr_curve_csv), note="frame-to-temporal-mean projection correlation")]},
            ],
            "metric_style": "metric-hero",
            "mini_metrics": [],
            "caption": f"The central comparison uses only Raw vs {display_name_final}, not intermediate denoise-only or registration-only displays. The same representative frame and deterministic crop are reused across all six images; quantitative differences are now shown as paired bar charts, while frame-wise rigid motion is plotted in μm and the correlation panel remains only in this integrated section.",
        },
        {
            "kicker": "Section C",
            "title": "C. Downstream Improvement",
            "panel_groups": [
                {"layout": "grid-2", "panels": [
                    panel("Raw segmentation map", raw_overlay_info["png"], source=str(raw_analysis_mask), note=f"Report display top percent = {int(round(REPORT_DISPLAY_TOP_PERCENT * 100))}%; background = STD; labels = ROI rank"),
                    panel(f"{display_name_final} segmentation map", final_overlay_info["png"], source=str(final_analysis_mask), note=f"Report display top percent = {int(round(REPORT_DISPLAY_TOP_PERCENT * 100))}%; background = STD; labels = ROI rank"),
                ]},
                {"layout": "grid-1", "panels": [panel("Downstream count comparison", down_count_png, source=str(run_root / 'segmentation' / 'summary.json'), note="Counts preserve analysis summary semantics")]},
                {"layout": "grid-2", "panels": [
                    panel("Paired trace numbered map", paired_assets.get("mask_png"), source=str(paired_assets.get("labelmask_tif") or ""), note="Numbering matches trace panel"),
                    panel("Paired trace curves", paired_assets.get("curve_png"), source=str(run_root / 'segmentation' / 'paired_trace' / 'paired_trace_summary.json'), note="Raw vs NeuroPilot traces on identical ROI anchors"),
                ]},
                {"layout": "grid-1", "panels": [
                    panel(f"{display_name_final} cell correlation heatmap", paired_assets.get("final_corr_png"), source=str(run_root / 'segmentation' / 'paired_trace' / 'paired_trace_summary.json'), note="All final analysis ROI shown; ROI order = final analysis rank ascending; value = Pearson correlation; colorbar = [-1, 1]"),
                ]},
                {"layout": "grid-1", "panels": [
                    panel(f"{display_name_final} cell temporal heatmap", paired_assets.get("final_trace_heatmap_png"), source=str(run_root / 'segmentation' / 'paired_trace' / 'paired_trace_summary.json'), note="All final analysis ROI shown; ROI order = final analysis rank ascending; x-axis labels = time in seconds; colormap = RdBu_r; bin count = min(1000, frames); value = per-cell Z-score after binning; cells are rendered with square heatmap elements"),
                ]},
            ],
            "metric_style": "metric-compact",
            "mini_metrics": [
                _kv("raw plane0 total", seg_summary.get("suite2p_counts", {}).get("raw_plane0_total", "N/A")),
                _kv(f"{display_name_final} plane0 total", seg_summary.get("suite2p_counts", {}).get("final_plane0_total", "N/A")),
                _kv("raw accepted ROI", seg_summary.get("suite2p_counts", {}).get("raw_after_cc_area_filter", "N/A")),
                _kv(f"{display_name_final} accepted ROI", seg_summary.get("suite2p_counts", {}).get("final_after_cc_area_filter", "N/A")),
                _kv("raw display selected", seg_summary.get("suite2p_counts", {}).get("raw_display_selected_count", "N/A")),
                _kv(f"{display_name_final} display selected", seg_summary.get("suite2p_counts", {}).get("final_display_selected_count", "N/A")),
                _kv("paired trace displayed count", paired_assets.get("displayed_count", "N/A")),
                _kv("heatmap ROI count", paired_assets.get("all_selected_count", "N/A")),
            ],
            "caption": f"Downstream visualization keeps the analysis counts unchanged while using a larger deterministic report-only subset for segmentation overview on STD backgrounds with ROI-rank labels. Paired traces remain a compact top-ROI view, while the correlation and temporal heatmaps now show the full final analysis ROI set in rank order, with the temporal axis labeled in seconds and the temporal heatmap rendered with square cells.",
        },
        {
            "kicker": "Section D",
            "title": "D. Provenance / Reproducibility",
            "panel_groups": [],
            "metric_style": "metric-compact",
            "mini_metrics": [
                _kv("folder tag", manifest.get("folder_tag", "N/A")),
                _kv("pipeline LLM mode", manifest.get("pipeline_llm_mode", "N/A")),
                _kv("advisor backend", manifest.get("advisor_backend", "N/A")),
                _kv("final_stack semantic", final_sidecar.get("source_semantic", "N/A")),
                _kv("raw stack path", str(raw_stack_path)),
                _kv("final stack path", str(final_stack_path)),
                _kv("iter params", final_used_params.get("iterations", [])),
                _kv("missing summary", ", ".join(unavailable) if unavailable else "none"),
            ],
            "caption": "This section records the frozen final_stack semantic, effective iteration parameters, and key artifact provenance consumed by the deterministic report builder.",
        },
    ]

    section_specs = [
        {
            "key": "input_overview",
            "title_body": "Input Data Overview",
            "panel_groups": sections[0]["panel_groups"],
            "metric_style": sections[0]["metric_style"],
            "mini_metrics": sections[0]["mini_metrics"],
            "caption": sections[0]["caption"],
        },
        {
            "key": "integrated_improvement",
            "title_body": f"Integrated Improvement: Raw vs {display_name_final}",
            "panel_groups": sections[1]["panel_groups"],
            "metric_style": sections[1]["metric_style"],
            "mini_metrics": sections[1]["mini_metrics"],
            "caption": sections[1]["caption"],
        },
    ]
    if include_downstream_section:
        section_specs.append(
            {
                "key": "downstream_improvement",
                "title_body": "Downstream Improvement",
                "panel_groups": sections[2]["panel_groups"],
                "metric_style": sections[2]["metric_style"],
                "mini_metrics": sections[2]["mini_metrics"],
                "caption": sections[2]["caption"],
            }
        )
    provenance_source = sections[3]
    section_specs.append(
        {
            "key": "provenance",
            "title_body": "Provenance / Reproducibility",
            "panel_groups": provenance_source["panel_groups"],
            "metric_style": provenance_source["metric_style"],
            "mini_metrics": provenance_source["mini_metrics"],
            "caption": provenance_source["caption"],
        }
    )
    sections = []
    section_structure = {}
    for idx, spec in enumerate(section_specs):
        letter = _panel_label(idx)
        section = {
            "kicker": f"Section {letter}",
            "title": f"{letter}. {spec['title_body']}",
            "panel_groups": spec["panel_groups"],
            "metric_style": spec["metric_style"],
            "mini_metrics": spec["mini_metrics"],
            "caption": spec["caption"],
        }
        sections.append(section)
        section_structure[str(spec["key"])] = section["title"]

    layout_notes = [
        f"Intensity panels use {_colormap_label(imaging_modality_used)} colormap based on imaging_modality={imaging_modality_used}.",
        f"The main crop uses scale_factor={float(report_crop_scale_factor):.2f} and is reused across Raw/{display_name_final} single-frame, STD and MIP views.",
        f"Kymograph line count = {report_kymograph_line_count}.",
        f"Intermediate denoise/registration sections removed = {not bool(report_use_intermediate_sections)}.",
        "All HTML assets are embedded via data URI when report_embed_assets=true.",
    ]
    if not include_downstream_section:
        layout_notes.append(
            f"Downstream section omitted because this dataset is marked as {downstream_section_omitted_reason or 'downstream_not_applicable'}."
        )

    report_data = {
        "schema_version": "deterministic_report_data.v3",
        "report_title": REPORT_TITLE,
        "dataset_identity": {
            "folder_name": manifest.get("folder_name"),
            "folder_tag": manifest.get("folder_tag"),
            "dataset_profile": dataset_profile,
            "is_cell_data": is_cell_data,
            "run_root": str(run_root),
            "raw_stack_path": str(raw_stack_path),
            "final_stack_path": str(final_stack_path),
        },
        "imaging_modality_used": imaging_modality_used,
        "colormap_used": _colormap_label(imaging_modality_used),
        "pixel_size_um_used": pixel_size_used,
        "fps_hz_used": fps_hz_used,
        "defaults_note": defaults_note,
        "representative_frames": {
            "raw_peak_activity_frame": rep_frame,
            "raw_peak_activity_rule": rep_info.get("selection_rule"),
            "raw_peak_activity_roi_percentile": rep_info.get("roi_percentile"),
            "motion_burst_center_frame": motion_info.get("center_frame"),
            "motion_burst_delta": motion_info.get("delta"),
            "motion_burst_triplet": motion_info.get("triplet_frames"),
        },
        "main_comparison_panel_layout": {
            "main_grid": "2x3",
            "main_grid_rows": [_reader_label("raw"), display_name_final],
            "main_grid_columns": ["single frame", "STD", "MIP"],
            "crop_grid": "2x3",
            "kymograph_bundle": "STD line map + 2 x (Raw vs NeuroPilot kymograph)",
    "quantitative_bar_panel": "Raw vs NeuroPilot grouped bar charts for fixed-ROI SNR proxy, motion mean, motion p95 and bleaching",
            "motion_curve_panel": "Frame-wise rigid motion magnitude in μm",
        },
        "roi_crop_regions": [roi_crop_region],
        "zoom_regions": zoom_regions,
        "kymograph_lines": kymograph_lines,
        "motion_burst_config": {
            "center_frame": motion_info.get("center_frame"),
            "delta": motion_info.get("delta"),
            "triplet_frames": motion_info.get("triplet_frames"),
            "selection_rule": motion_info.get("selection_rule"),
        },
        "correlation_curve": {
            "source": "frame_to_temporal_mean_projection",
            "x_axis": "frame",
            "y_axis": "correlation",
            "artifact_csv": str(corr_curve_csv),
            "raw_summary": _summary_array(np.asarray(raw_corr_info["curve"], dtype=np.float32)),
            "neuropilot_summary": _summary_array(np.asarray(final_corr_info["curve"], dtype=np.float32)),
        },
        "motion_curve": {
            "source": "rigid_shift_magnitude",
            "x_axis": "frame",
            "y_axis": "motion_magnitude_um",
            "pixel_size_um_used": pixel_size_used,
            "raw_shifts_npy": str(raw_shifts_npy) if raw_shifts_npy else None,
            "neuropilot_shifts_npy": str(final_shifts_npy) if final_shifts_npy else None,
            "raw_motion_mean_um": _px_to_um(comp_final.get("motion_before_after", {}).get("raw", {}).get("motion_mean_px"), pixel_size_used),
            "neuropilot_motion_mean_um": _px_to_um(comp_final.get("motion_before_after", {}).get("final", {}).get("motion_mean_px"), pixel_size_used),
            "raw_motion_p95_um": _px_to_um(comp_final.get("motion_before_after", {}).get("raw", {}).get("motion_p95_px"), pixel_size_used),
            "neuropilot_motion_p95_um": _px_to_um(comp_final.get("motion_before_after", {}).get("final", {}).get("motion_p95_px"), pixel_size_used),
            "artifact_png": str(motion_curve_png),
        },
        "label_alias": {"final_display_name": display_name_final},
        "downstream_display_config": {
            "section_included": bool(include_downstream_section),
            "omitted_reason": downstream_section_omitted_reason,
            "report_display_top_percent": REPORT_DISPLAY_TOP_PERCENT,
            "paired_trace_numbering_rule": "paired_trace_roi_table_rank_order_to_labels_1..N",
            "heatmap_roi_order": "final_analysis_rank_ascending",
            "heatmap_cell_display": "all_final_analysis_cells",
            "segmentation_background": "std_projection",
            "temporal_bin_target": 1000,
            "temporal_axis_unit": "seconds_using_fps",
            "temporal_heatmap_colormap": "RdBu_r",
            "temporal_heatmap_cell_aspect": "square",
        },
        "section_structure": section_structure,
        "asset_embedding_status": {
            "assets_embedded": bool(report_embed_assets),
            "inline_css": bool(report_inline_css),
            "standalone_html": bool(report_embed_assets and report_inline_css),
        },
        "metrics": {
            "input_quality": {
                "shape_thw": raw_metrics.get("data_summary", {}).get("shape_thw"),
                "dtype": raw_metrics.get("data_summary", {}).get("dtype"),
                "dynamic_range": raw_metrics.get("data_summary", {}).get("dynamic_range"),
                "bleaching": raw_metrics.get("bleaching_trend"),
                "snr": raw_metrics.get("snr_metric"),
                "rigid_motion": raw_metrics.get("rigid_motion_metric"),
            },
            "integrated_final_improvement": {
                **comp_final,
                "comparison_target_display_name": display_name_final,
                "metric_bar_panel_png": str(metric_bar_png),
                "motion_curve_png": str(motion_curve_png),
                "correlation_curve_csv": str(corr_curve_csv),
                "used_intermediate_sections": bool(report_use_intermediate_sections),
                "intermediate_references": {
                    "raw_vs_denoise_json": str(run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_denoise.json"),
                    "raw_vs_motion_json": str(run_root / "iterations" / "iter_0" / "metrics" / "comparison_raw_vs_motion.json"),
                },
            },
            "downstream_improvement": {
                "section_included": bool(include_downstream_section),
                "omitted_reason": downstream_section_omitted_reason,
                "summary": seg_summary if include_downstream_section else None,
                "comparison": seg_comp if include_downstream_section else None,
                "paired_trace": paired_assets.get("summary"),
            },
        },
        "artifact_inventory": _collect_inventory(run_root, manifest, report_assets),
        "unavailable_metrics": unavailable,
        "layout_notes": layout_notes,
        "provenance": {
            "pipeline_manifest_path": str(run_root / "manifests" / "pipeline_manifest.json"),
            "final_used_params_path": str(run_root / "final_used_params.json"),
            "final_stack_sidecar_path": str(run_root / "final" / "final_stack_sidecar.json"),
            "effective_params_per_iteration": final_used_params.get("iterations", []),
            "downstream_paths": manifest.get("downstream", {}),
        },
        "correlation_panel_source": "frame_to_temporal_mean_projection",
        "sections": sections,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    report_data_path = report_dir / "report_data.json"
    report_manifest_path = report_dir / "report_manifest.json"
    report_html_path = report_dir / "report.html"
    report_print_html_path = report_dir / "report_print.html"
    _write_json(report_data_path, report_data)
    report_html_path.write_text(_render_report_html(report_data, report_dir, print_mode=False), encoding="utf-8")
    report_print_html_path.write_text(_render_report_html(report_data, report_dir, print_mode=True), encoding="utf-8")

    pdf_result = {"generated": False, "path": None, "engine": None, "error": "disabled"}
    if report_generate_pdf:
        pdf_result = _try_generate_pdf(report_print_html_path, report_dir / "report.pdf")
    report_data["pdf_status"] = pdf_result
    _write_json(report_data_path, report_data)

    manifest_payload = {
        "schema_version": "deterministic_report.v3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "report_title": REPORT_TITLE,
        "report_data_json": str(report_data_path),
        "report_html": str(report_html_path),
        "report_print_html": str(report_print_html_path),
        "report_pdf": pdf_result.get("path"),
        "pdf_status": pdf_result,
        "assets_embedded": bool(report_embed_assets),
        "pdf_generated": bool(pdf_result.get("generated")),
        "pdf_engine_used": pdf_result.get("engine"),
        "colormap_mode_used": _colormap_label(imaging_modality_used),
        "imaging_modality_used": imaging_modality_used,
        "pixel_size_um_used": pixel_size_used,
        "fps_hz_used": fps_hz_used,
        "unavailable_metrics": unavailable,
    }
    _write_json(report_manifest_path, manifest_payload)

    return {
        "report_data_json": str(report_data_path),
        "report_manifest_json": str(report_manifest_path),
        "report_html": str(report_html_path),
        "report_print_html": str(report_print_html_path),
        "report_pdf": pdf_result.get("path"),
        "pdf_status": pdf_result,
        "representative_frame": rep_frame,
        "motion_burst_center": motion_info.get("center_frame"),
        "motion_burst_delta": motion_info.get("delta"),
        "zoom_regions": zoom_regions,
        "kymograph_lines": kymograph_lines,
        "correlation_panel_source": "frame_to_temporal_mean_projection",
        "standalone_html_generated": bool(report_embed_assets and report_inline_css),
        "unavailable_metrics": unavailable,
    }


def main():
    parser = argparse.ArgumentParser(description="Build deterministic scientific report from existing NeuMar artifacts.")
    parser.add_argument("--run-root", required=True, help="Pipeline run root that already contains metrics/final/segmentation artifacts")
    parser.add_argument("--output-dir", default=None, help="Optional explicit report output directory")
    parser.add_argument("--no-pdf", action="store_true", help="Skip optional PDF export")
    parser.add_argument("--imaging-modality", default=DEFAULT_IMAGING_MODALITY, choices=["1p", "2p", "3p"], help="Imaging modality used to choose the intensity colormap")
    parser.add_argument("--pixel-size-um", type=float, default=DEFAULT_PIXEL_SIZE_UM, help="Fallback pixel size in μm/pixel when input metrics do not provide one")
    parser.add_argument("--fps-hz", type=float, default=DEFAULT_FPS_HZ, help="Fallback frame rate in Hz when input metrics do not provide one")
    parser.add_argument("--no-embed-assets", action="store_true", help="Keep report HTML referencing external asset files instead of embedding them")
    parser.add_argument("--crop-scale-factor", type=float, default=REPORT_CROP_SCALE_FACTOR, help="Deterministic scale factor for the report crop relative to the initial high-information region")
    parser.add_argument("--kymograph-line-count", type=int, default=REPORT_KYMOGRAPH_LINE_COUNT, help="Number of deterministic kymograph lines to select")
    args = parser.parse_args()
    result = build_deterministic_report(
        args.run_root,
        output_dir=args.output_dir,
        try_pdf=not args.no_pdf,
        imaging_modality=args.imaging_modality,
        pixel_size_um=args.pixel_size_um,
        fps_hz=args.fps_hz,
        report_embed_assets=not args.no_embed_assets,
        report_crop_scale_factor=args.crop_scale_factor,
        report_kymograph_line_count=args.kymograph_line_count,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
