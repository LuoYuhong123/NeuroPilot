#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import tifffile as tiff

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except Exception:
    cv2 = None


DEFAULT_PIXEL_SIZE_UM = 0.645
DEFAULT_FPS_HZ = 10.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _render_scale_label(pixel_size_um: float | None) -> str:
    if pixel_size_um is None or pixel_size_um <= 0:
        return "pixel size unavailable"
    return f"{pixel_size_um:.3f} um/pixel"


def load_tif_pages_to_thw(tif_path: str, expect_hw: tuple[int, int] | None = None) -> np.ndarray:
    with tiff.TiffFile(str(tif_path)) as tf:
        page_shapes = [pg.shape for pg in tf.pages]

        if expect_hw is None:
            (h, w), _ = Counter(page_shapes).most_common(1)[0]
        else:
            h, w = expect_hw

        frames = []
        skipped = 0
        for pg in tf.pages:
            if pg.shape != (h, w):
                skipped += 1
                continue
            arr = np.squeeze(pg.asarray())
            if arr.shape == (h, w):
                frames.append(arr)
            else:
                skipped += 1

        if not frames:
            raise RuntimeError(f"No valid frames found in {tif_path}")

        stack = np.stack(frames, axis=0)
        print(
            f"[INFO] {Path(tif_path).name}: "
            f"pages={len(tf.pages)}, used={len(frames)}, skipped={skipped}, shape={stack.shape}"
        )
        return stack


def write_proj_max_std(stack_thw: np.ndarray, out_dir: Path, dtype_out=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dtype_out is None:
        dtype_out = stack_thw.dtype
    img_max = stack_thw.max(axis=0)
    tiff.imwrite(str(out_dir / "MAX.tif"), img_max.astype(dtype_out), bigtiff=False)
    img_std = stack_thw.astype(np.float32).std(axis=0)
    tiff.imwrite(str(out_dir / "STD.tif"), img_std.astype(np.float32), bigtiff=False)


def normalize_robust_01(img: np.ndarray, p_low=5.0, p_high=99.5, eps=1e-6) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _pick_representative_frame(stack: np.ndarray) -> tuple[int, np.ndarray]:
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[0] <= 0:
        raise ValueError(f"Expected stack with shape (T,H,W), got {stack.shape}")
    score = stack.astype(np.float32).reshape(stack.shape[0], -1).std(axis=1)
    idx = int(np.argmax(score))
    return idx, np.asarray(stack[idx], dtype=np.float32)


def _prepare_candidate_score(
    max_img: np.ndarray,
    std_img: np.ndarray,
    norm_p_low: float,
    norm_p_high: float,
) -> dict[str, np.ndarray]:
    max_norm = normalize_robust_01(max_img, p_low=norm_p_low, p_high=norm_p_high)
    std_norm = normalize_robust_01(std_img, p_low=norm_p_low, p_high=norm_p_high)
    score = np.clip(0.6 * max_norm + 0.4 * std_norm, 0.0, 1.0).astype(np.float32)
    if cv2 is not None:
        score_blur = cv2.GaussianBlur(score, (0, 0), sigmaX=1.2, sigmaY=1.2)
    else:
        score_blur = score
    return {
        "max_norm": max_norm.astype(np.float32),
        "std_norm": std_norm.astype(np.float32),
        "score": score.astype(np.float32),
        "score_blur": np.asarray(score_blur, dtype=np.float32),
    }


def _ring_mean(img: np.ndarray, mask: np.ndarray, dilate_px: int = 5) -> float:
    if cv2 is None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return float(np.mean(img))
        y0 = max(0, int(np.min(ys)) - dilate_px)
        y1 = min(img.shape[0], int(np.max(ys)) + dilate_px + 1)
        x0 = max(0, int(np.min(xs)) - dilate_px)
        x1 = min(img.shape[1], int(np.max(xs)) + dilate_px + 1)
        patch = np.asarray(img[y0:y1, x0:x1], dtype=np.float32)
        patch_mask = np.asarray(mask[y0:y1, x0:x1], dtype=bool)
        ring = patch[~patch_mask]
        return float(np.mean(ring)) if ring.size else float(np.mean(patch))

    kernel_size = max(3, int(dilate_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dilated, ~mask)
    if np.count_nonzero(ring) <= 0:
        return float(np.mean(img))
    return float(np.mean(np.asarray(img, dtype=np.float32)[ring]))


def _detect_candidates(
    score_blur: np.ndarray,
    max_norm: np.ndarray,
    std_norm: np.ndarray,
) -> dict[str, Any]:
    h, w = score_blur.shape
    thr_base = max(0.22, float(np.percentile(score_blur, 94)))
    thr_lo = max(0.18, thr_base * 0.92)
    binary = score_blur >= thr_lo

    if cv2 is not None:
        binary_u8 = (binary.astype(np.uint8) * 255)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, open_kernel)
        binary_u8 = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, close_kernel)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    else:
        labels = np.zeros_like(binary, dtype=np.int32)
        stats = np.zeros((1, 5), dtype=np.int32)
        centroids = np.zeros((1, 2), dtype=np.float32)
        current = 0
        for y in range(h):
            for x in range(w):
                if not binary[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                coords: list[tuple[int, int]] = []
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for ny in range(max(0, cy - 1), min(h, cy + 2)):
                        for nx in range(max(0, cx - 1), min(w, cx + 2)):
                            if binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current
                                stack.append((ny, nx))
                ys = [c[0] for c in coords]
                xs = [c[1] for c in coords]
                stats = np.vstack(
                    [
                        stats,
                        np.array(
                            [
                                min(xs),
                                min(ys),
                                max(xs) - min(xs) + 1,
                                max(ys) - min(ys) + 1,
                                len(coords),
                            ],
                            dtype=np.int32,
                        ),
                    ]
                )
                centroids = np.vstack(
                    [
                        centroids,
                        np.array([[float(np.mean(xs)), float(np.mean(ys))]], dtype=np.float32),
                    ]
                )

    all_candidates: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    overlay_mask = np.zeros_like(score_blur, dtype=np.int32)
    total_labels = int(labels.max())

    for label in range(1, total_labels + 1):
        mask = labels == label
        area = int(np.count_nonzero(mask))
        if area <= 0:
            continue
        ys, xs = np.where(mask)
        x0 = int(np.min(xs))
        x1 = int(np.max(xs))
        y0 = int(np.min(ys))
        y1 = int(np.max(ys))
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        aspect = float(max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h)))
        eq_diameter = float(math.sqrt(4.0 * area / math.pi))
        mean_score = float(np.mean(score_blur[mask]))
        mean_max = float(np.mean(max_norm[mask]))
        mean_std = float(np.mean(std_norm[mask]))
        border_touch = bool(x0 <= 1 or y0 <= 1 or x1 >= (w - 2) or y1 >= (h - 2))
        border_sides = int(x0 <= 1) + int(y0 <= 1) + int(x1 >= (w - 2)) + int(y1 >= (h - 2))
        ring_mean = _ring_mean(score_blur, mask, dilate_px=5)
        local_contrast = float(max(0.0, mean_score - ring_mean))
        candidate = {
            "label": int(label),
            "area_px": area,
            "bbox": {"x0": x0, "y0": y0, "x1": x1 + 1, "y1": y1 + 1},
            "bbox_w": int(bbox_w),
            "bbox_h": int(bbox_h),
            "aspect_ratio": aspect,
            "equivalent_diameter_px": eq_diameter,
            "mean_score": mean_score,
            "mean_max_norm": mean_max,
            "mean_std_norm": mean_std,
            "local_contrast": local_contrast,
            "border_touch": border_touch,
            "border_sides": border_sides,
            "centroid_xy": [float(np.mean(xs)), float(np.mean(ys))],
        }
        all_candidates.append(candidate)

        border_crop_soma_like = (
            border_touch
            and border_sides <= 1
            and area >= 80
            and eq_diameter >= 10.0
            and aspect <= 2.8
            and local_contrast >= 0.12
            and mean_score >= max(0.30, thr_lo)
        )
        keep = (
            area >= 18
            and area <= 2000
            and eq_diameter >= 6.0
            and eq_diameter <= 40.0
            and aspect <= 3.5
            and local_contrast >= 0.03
            and (not border_touch or border_crop_soma_like)
        )
        if keep:
            selected_candidates.append(candidate)
            overlay_mask[mask] = len(selected_candidates)

    basis_candidates = selected_candidates if len(selected_candidates) >= 5 else [
        c for c in all_candidates if c["area_px"] >= 18 and c["equivalent_diameter_px"] >= 6.0
    ]

    selected_label_values = {int(c["label"]) for c in selected_candidates}
    selected_labelmask = np.zeros_like(labels, dtype=np.uint16)
    next_selected_rank = 1
    for label in range(1, total_labels + 1):
        if int(label) not in selected_label_values:
            continue
        selected_labelmask[labels == label] = np.uint16(next_selected_rank)
        next_selected_rank += 1

    basis_label_values = {int(c["label"]) for c in basis_candidates}
    basis_labelmask = np.zeros_like(labels, dtype=np.uint16)
    next_basis_rank = 1
    for label in range(1, total_labels + 1):
        if int(label) not in basis_label_values:
            continue
        basis_labelmask[labels == label] = np.uint16(next_basis_rank)
        next_basis_rank += 1

    centroids = np.asarray([c["centroid_xy"] for c in basis_candidates], dtype=np.float32)
    nn_distances: list[float] = []
    if centroids.shape[0] >= 2:
        diff = centroids[:, None, :] - centroids[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        dist[dist == 0] = np.inf
        nn_distances = dist.min(axis=1).astype(np.float64).tolist()

    density_per_128 = float(len(basis_candidates) / max(1e-6, (h * w) / float(128 * 128)))
    elongated_fraction = float(
        np.mean([c["aspect_ratio"] > 2.5 for c in basis_candidates], dtype=np.float64)
    ) if basis_candidates else 0.0
    border_touch_fraction = float(
        np.mean([c["border_touch"] for c in all_candidates], dtype=np.float64)
    ) if all_candidates else 0.0

    stats_summary = {
        "all_candidate_count": int(len(all_candidates)),
        "selected_candidate_count": int(len(selected_candidates)),
        "basis_candidate_count": int(len(basis_candidates)),
        "density_per_128x128": density_per_128,
        "border_touch_fraction": border_touch_fraction,
        "elongated_fraction": elongated_fraction,
        "equivalent_diameter_px": _summary([c["equivalent_diameter_px"] for c in basis_candidates]),
        "aspect_ratio": _summary([c["aspect_ratio"] for c in basis_candidates]),
        "local_contrast": _summary([c["local_contrast"] for c in basis_candidates]),
        "mean_score": _summary([c["mean_score"] for c in basis_candidates]),
        "nearest_neighbor_px": _summary(nn_distances),
    }
    return {
        "all_candidates": all_candidates,
        "selected_candidates": selected_candidates,
        "basis_candidates": basis_candidates,
        "overlay_mask": overlay_mask,
        "selected_labelmask": selected_labelmask,
        "basis_labelmask": basis_labelmask,
        "nearest_neighbor_values": nn_distances,
        "score_threshold_used": thr_lo,
        "stats": stats_summary,
    }


def _suggest_config(
    existing_cfg: dict[str, Any] | None,
    profile_stats: dict[str, Any],
    pixel_size_um: float | None,
    fps_hz: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(existing_cfg or {})
    diameter_stats = profile_stats.get("equivalent_diameter_px", {}) or {}
    aspect_stats = profile_stats.get("aspect_ratio", {}) or {}
    contrast_stats = profile_stats.get("local_contrast", {}) or {}
    nn_stats = profile_stats.get("nearest_neighbor_px", {}) or {}

    base_diameter = _safe_float(diameter_stats.get("median"))
    if base_diameter is None:
        base_diameter = 16.0
    suggested_diameter = int(round(_clamp(base_diameter * 1.10, 10.0, 24.0)))

    threshold_scaling = 1.00
    elongated_fraction = _safe_float(profile_stats.get("elongated_fraction")) or 0.0
    density = _safe_float(profile_stats.get("density_per_128x128")) or 0.0
    local_contrast_median = _safe_float(contrast_stats.get("median")) or 0.0
    if elongated_fraction > 0.30:
        threshold_scaling += 0.10
    if density > 10.0:
        threshold_scaling += 0.05
    if local_contrast_median < 0.06:
        threshold_scaling -= 0.05
    threshold_scaling = round(_clamp(threshold_scaling, 0.85, 1.15), 2)

    aspect_p90 = _safe_float(aspect_stats.get("p90"))
    if aspect_p90 is None:
        suggested_aspect = 2.0
    else:
        suggested_aspect = round(_clamp(aspect_p90, 1.6, 2.5), 2)

    nn_median = _safe_float(nn_stats.get("median"))
    if nn_median is None:
        max_overlap = 0.45
    else:
        spacing_ratio = nn_median / max(1.0, float(suggested_diameter))
        if spacing_ratio >= 1.8:
            max_overlap = 0.35
        elif spacing_ratio >= 1.4:
            max_overlap = 0.45
        else:
            max_overlap = 0.55

    min_area_px = int(
        round(
            _clamp(
                0.35 * math.pi * ((float(suggested_diameter) / 2.0) ** 2),
                25.0,
                120.0,
            )
        )
    )

    suggested = dict(cfg)
    suggested.update(
        {
            "fs_hz": int(round(_safe_float(cfg.get("fs_hz")) or fps_hz or DEFAULT_FPS_HZ)),
            "do_registration": False,
            "nonrigid": False,
            "diameter_px": int(suggested_diameter),
            "threshold_scaling": float(threshold_scaling),
            "aspect_max": float(suggested_aspect),
            "max_overlap": float(max_overlap),
            "min_area_px": int(min_area_px),
        }
    )

    suggestion_meta = {
        "basis": {
            "pixel_size_um": pixel_size_um,
            "fps_hz": fps_hz,
            "diameter_median_px": base_diameter,
            "aspect_p90": aspect_p90,
            "local_contrast_median": local_contrast_median,
            "density_per_128x128": density,
            "elongated_fraction": elongated_fraction,
            "nearest_neighbor_median_px": nn_median,
        },
        "rationale": {
            "do_registration": "Disabled because the upstream final stack is already motion-corrected and pre-segmentation profiling targets direct ROI detection on the final artifact.",
            "diameter_px": "Set from the robust median equivalent diameter of pre-segmentation blob candidates, with a small upward margin to better cover soma boundaries.",
            "threshold_scaling": "Starts from a conservative soma-first default and is adjusted upward when elongated/background candidates are common, or downward when contrast is weak.",
            "aspect_max": "Tightened toward soma-like compactness using the p90 aspect ratio of candidate blobs rather than neurite-friendly defaults.",
            "max_overlap": "Mapped from candidate nearest-neighbor spacing so well-separated soma proposals do not over-merge.",
            "min_area_px": "Derived from the suggested diameter to reject small debris before later ROI selection stages.",
        },
    }
    return suggested, suggestion_meta


def _save_profile_overview_png(
    out_path: Path,
    representative_frame: np.ndarray,
    max_norm: np.ndarray,
    std_norm: np.ndarray,
    score: np.ndarray,
    overlay_mask: np.ndarray,
    pixel_size_um: float | None,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = [
        (representative_frame, "Representative frame"),
        (max_norm, "MAX projection (normalized)"),
        (std_norm, "STD projection (normalized)"),
        (score, "Candidate score with selected blobs"),
    ]
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    overlay_ax = axes[1, 1]
    if np.count_nonzero(overlay_mask) > 0:
        overlay_ax.contour(overlay_mask > 0, levels=[0.5], colors=["#5ec8ff"], linewidths=1.0)
    fig.suptitle(f"Pre-segmentation profiling overview ({_render_scale_label(pixel_size_um)})", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(out_path)


def _save_profile_histograms_png(
    out_path: Path,
    nearest_neighbor_values: list[float],
    basis_candidates: list[dict[str, Any]],
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    arrays = {
        "Equivalent diameter (px)": [c["equivalent_diameter_px"] for c in basis_candidates],
        "Aspect ratio": [c["aspect_ratio"] for c in basis_candidates],
        "Local contrast": [c["local_contrast"] for c in basis_candidates],
        "Nearest-neighbor distance (px)": list(nearest_neighbor_values or []),
    }
    for ax, (title, values) in zip(axes.flat, arrays.items()):
        values = list(values or [])
        if values:
            ax.hist(values, bins=min(20, max(5, len(values) // 2)), color="#3a86ff", alpha=0.85)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2, linestyle="--")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    return str(out_path)


def _html_table(rows: list[tuple[str, str]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in rows
    )
    return f"<table>{body}</table>"


def _save_preseg_report_html(
    out_path: Path,
    profile: dict[str, Any],
    suggested_config: dict[str, Any],
    overview_png_rel: str,
    hist_png_rel: str,
) -> str:
    stats = profile.get("stats", {}) or {}
    diameter_stats = stats.get("equivalent_diameter_px", {}) or {}
    aspect_stats = stats.get("aspect_ratio", {}) or {}
    contrast_stats = stats.get("local_contrast", {}) or {}
    nn_stats = stats.get("nearest_neighbor_px", {}) or {}
    pixel_size_um = _safe_float(profile.get("pixel_size_um"))

    summary_rows = [
        ("Target mode", profile.get("target_mode") or "soma"),
        ("Pixel size", _render_scale_label(pixel_size_um)),
        ("Frame rate", f"{_safe_float(profile.get('fps_hz')) or DEFAULT_FPS_HZ:.1f} Hz"),
        ("Image size", f"{profile.get('height_px')} x {profile.get('width_px')} px"),
        ("Frames", str(profile.get("frame_count"))),
        ("Candidate basis count", str(stats.get("basis_candidate_count"))),
        ("Candidate density / 128x128", f"{(_safe_float(stats.get('density_per_128x128')) or 0.0):.2f}"),
        ("Border-touch fraction", f"{(_safe_float(stats.get('border_touch_fraction')) or 0.0):.2f}"),
        ("Elongated fraction", f"{(_safe_float(stats.get('elongated_fraction')) or 0.0):.2f}"),
        ("Median candidate diameter", f"{(_safe_float(diameter_stats.get('median')) or 0.0):.2f} px"),
        ("Median aspect ratio", f"{(_safe_float(aspect_stats.get('median')) or 0.0):.2f}"),
        ("Median local contrast", f"{(_safe_float(contrast_stats.get('median')) or 0.0):.3f}"),
        ("Median nearest neighbor", f"{(_safe_float(nn_stats.get('median')) or 0.0):.2f} px"),
    ]
    suggestion_rows = [
        ("do_registration", str(bool(suggested_config.get("do_registration"))).lower()),
        ("nonrigid", str(bool(suggested_config.get("nonrigid"))).lower()),
        ("diameter_px", str(suggested_config.get("diameter_px"))),
        ("threshold_scaling", str(suggested_config.get("threshold_scaling"))),
        ("aspect_max", str(suggested_config.get("aspect_max"))),
        ("max_overlap", str(suggested_config.get("max_overlap"))),
        ("min_area_px", str(suggested_config.get("min_area_px"))),
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NeuroPilot Pre-segmentation Profiling Report</title>
  <style>
    body {{
      font-family: Georgia, "Times New Roman", serif;
      margin: 28px;
      color: #e8edf3;
      background: #0b0f14;
      line-height: 1.45;
    }}
    h1, h2 {{
      color: #dcecff;
      margin-bottom: 0.4rem;
    }}
    .brand {{
      color: #79b8ff;
    }}
    .lead {{
      color: #b6c4d6;
      max-width: 960px;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }}
    .panel {{
      background: #101722;
      border: 1px solid #1f2b3b;
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.25);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #243244;
      vertical-align: top;
    }}
    th {{
      width: 42%;
      color: #98c1ff;
      font-weight: 600;
    }}
    img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid #203041;
      background: #ffffff;
    }}
    .note {{
      color: #a9b6c7;
      font-size: 0.95rem;
    }}
  </style>
</head>
<body>
  <h1><span class="brand">NeuroPilot</span> Pre-segmentation Profiling Report</h1>
  <p class="lead">This profiling pass inspects the final motion-corrected stack before suite2p, estimates soma-like blob statistics from MAX/STD projections, and recommends a conservative segmentation configuration with suite2p registration disabled.</p>
  <div class="grid">
    <section class="panel">
      <h2>Profile Summary</h2>
      { _html_table(summary_rows) }
    </section>
    <section class="panel">
      <h2>Suggested Suite2p Config</h2>
      { _html_table(suggestion_rows) }
      <p class="note">These values are intended for soma-first segmentation on the final stack and are applied before suite2p starts.</p>
    </section>
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Projection Overview</h2>
      <img src="{html.escape(overview_png_rel)}" alt="Pre-segmentation overview">
    </section>
    <section class="panel">
      <h2>Candidate Distributions</h2>
      <img src="{html.escape(hist_png_rel)}" alt="Pre-segmentation distributions">
    </section>
  </div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return str(out_path)


def build_presegmentation_profile(
    final_stack_path: str | Path,
    output_root: str | Path,
    segmentation_config: dict[str, Any] | None = None,
    pixel_size_um: float | None = None,
    fps_hz: float | None = None,
    target_mode: str = "soma",
    norm_p_low: float = 5.0,
    norm_p_high: float = 99.5,
) -> dict[str, Any]:
    final_stack_path = Path(final_stack_path).resolve()
    output_root = Path(output_root).resolve()
    assets_dir = output_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    stack = load_tif_pages_to_thw(str(final_stack_path))
    frame_count = int(stack.shape[0])
    rep_idx, representative_frame = _pick_representative_frame(stack)
    write_proj_max_std(stack, output_root)
    max_img = np.asarray(stack.max(axis=0), dtype=np.float32)
    std_img = np.asarray(stack.astype(np.float32).std(axis=0), dtype=np.float32)
    del stack

    prepared = _prepare_candidate_score(max_img, std_img, norm_p_low=norm_p_low, norm_p_high=norm_p_high)
    tiff.imwrite(str(output_root / "MAX_norm.tif"), prepared["max_norm"].astype(np.float32), bigtiff=False)
    tiff.imwrite(str(output_root / "STD_norm.tif"), prepared["std_norm"].astype(np.float32), bigtiff=False)
    tiff.imwrite(str(output_root / "candidate_score.tif"), prepared["score"].astype(np.float32), bigtiff=False)

    detection = _detect_candidates(
        score_blur=prepared["score_blur"],
        max_norm=prepared["max_norm"],
        std_norm=prepared["std_norm"],
    )
    tiff.imwrite(
        str(output_root / "selected_candidate_labelmask_uint16.tif"),
        np.asarray(detection["selected_labelmask"], dtype=np.uint16),
        photometric="minisblack",
        metadata=None,
    )
    tiff.imwrite(
        str(output_root / "basis_candidate_labelmask_uint16.tif"),
        np.asarray(detection["basis_labelmask"], dtype=np.uint16),
        photometric="minisblack",
        metadata=None,
    )
    suggested_config, suggestion_meta = _suggest_config(
        existing_cfg=segmentation_config,
        profile_stats=detection["stats"],
        pixel_size_um=pixel_size_um,
        fps_hz=fps_hz,
    )

    overview_png = _save_profile_overview_png(
        assets_dir / "preseg_overview.png",
        representative_frame=representative_frame,
        max_norm=prepared["max_norm"],
        std_norm=prepared["std_norm"],
        score=prepared["score"],
        overlay_mask=detection["overlay_mask"],
        pixel_size_um=pixel_size_um,
    )
    hist_png = _save_profile_histograms_png(
        assets_dir / "preseg_candidate_distributions.png",
        nearest_neighbor_values=detection["nearest_neighbor_values"],
        basis_candidates=detection["basis_candidates"],
    )

    profile = {
        "target_mode": str(target_mode or "soma"),
        "input_tif_path": str(final_stack_path),
        "pixel_size_um": _safe_float(pixel_size_um) or DEFAULT_PIXEL_SIZE_UM,
        "fps_hz": _safe_float(fps_hz) or DEFAULT_FPS_HZ,
        "frame_count": frame_count,
        "representative_frame_index": int(rep_idx),
        "height_px": int(representative_frame.shape[0]),
        "width_px": int(representative_frame.shape[1]),
        "normalization": {
            "norm_p_low": float(norm_p_low),
            "norm_p_high": float(norm_p_high),
        },
        "score_threshold_used": float(detection["score_threshold_used"]),
        "stats": detection["stats"],
        "candidate_preview": detection["basis_candidates"][:32],
        "suggestion_meta": suggestion_meta,
    }
    profile_json_path = output_root / "preseg_profile.json"
    suggested_json_path = output_root / "preseg_suggested_config.json"
    report_html_path = output_root / "preseg_report.html"
    _write_json(profile_json_path, profile)
    _write_json(
        suggested_json_path,
        {
            "target_mode": profile["target_mode"],
            "input_tif_path": str(final_stack_path),
            "suggested_segmentation_config": suggested_config,
            "suggestion_meta": suggestion_meta,
        },
    )
    report_html = _save_preseg_report_html(
        report_html_path,
        profile=profile,
        suggested_config=suggested_config,
        overview_png_rel=f"assets/{Path(overview_png).name}",
        hist_png_rel=f"assets/{Path(hist_png).name}",
    )
    return {
        "profile_json": str(profile_json_path),
        "suggested_config_json": str(suggested_json_path),
        "report_html": str(report_html),
        "assets_dir": str(assets_dir),
        "artifacts": {
            "max_tif": str(output_root / "MAX.tif"),
            "std_tif": str(output_root / "STD.tif"),
            "max_norm_tif": str(output_root / "MAX_norm.tif"),
            "std_norm_tif": str(output_root / "STD_norm.tif"),
            "candidate_score_tif": str(output_root / "candidate_score.tif"),
            "selected_candidate_labelmask_tif": str(output_root / "selected_candidate_labelmask_uint16.tif"),
            "basis_candidate_labelmask_tif": str(output_root / "basis_candidate_labelmask_uint16.tif"),
            "overview_png": str(overview_png),
            "candidate_distributions_png": str(hist_png),
        },
        "suggested_segmentation_config": suggested_config,
        "stats": detection["stats"],
    }
