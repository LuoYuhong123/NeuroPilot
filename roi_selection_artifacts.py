#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter all Suite2p ROIs by largest connected-component area,
sort by trace max descending, and combine TOP X% into one colored mask.

NEW:
For the FINAL selected top X% ROIs only, polish each single mask before composing:
1) fill enclosed holes inside the mask
2) erode the mask boundary by 1 pixel
This makes each ROI look cleaner and slightly thinner.

Outputs:
1) colored PNG mask of top-ranked ROIs WITH IDs
2) colored PNG mask of top-ranked ROIs WITHOUT IDs
3) HD colored PNG WITH IDs
4) HD colored PNG WITHOUT IDs
5) uint16 labeled TIFF mask (0 bg, 1..N for selected ROIs)
6) CSV summary table
7) folder of all area-filtered masks (each ROI -> one tif)
8) folder of selected top-percent masks (each ROI -> one tif)

Author: ChatGPT
"""

from __future__ import annotations
import os
from pathlib import Path
import csv
import math
import json

import numpy as np
import matplotlib.pyplot as plt

try:
    import tifffile as tiff
except ImportError:
    raise ImportError("pip install tifffile")

try:
    import cv2
except ImportError:
    raise ImportError("pip install opencv-python")


# ============================================================
# CONFIG
# ============================================================

PLANE0_DIR_RAW = os.getenv("STEP1_PLANE0_DIR", "").strip()
PLANE0_DIR: Path | None = Path(PLANE0_DIR_RAW).expanduser() if PLANE0_DIR_RAW else None
OUT_ROOT = Path(os.getenv("STEP1_OUT_ROOT", "roi_selection_outputs")).expanduser()

# 面积过滤阈值：
# 注意这里是“最大连通域面积”阈值，不是原始 ypix 数量
MIN_LARGEST_CC_AREA = 60

# 取前多少比例
TOP_PERCENT = 1  # 0.10 = top 10%

# trace 文件
TRACE_NPY_NAME = "F.npy"

# 彩图参数
CMAP_NAME = "tab20"
DRAW_IDS = True

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.45
LABEL_THICKNESS = 1
LABEL_TEXT_COLOR = (0, 0, 0)       # black in BGR
LABEL_BG_COLOR = (255, 255, 255)   # white
LABEL_BG_ALPHA = 0.65

DRAW_TOTAL_N = True
TOTAL_TEXT_COLOR = (0, 0, 0)
TOTAL_BG_COLOR = (255, 255, 255)
TOTAL_BG_ALPHA = 0.75

# 高清彩色 mask 放大倍数
HD_UPSAMPLE = 6

# 单独 mask tif 的像素值
SINGLE_MASK_TIF_VALUE = 255   # 255 或 1 都可以

# ========= NEW: polish selected masks only =========
POLISH_SELECTED_MASKS = True
FILL_HOLES_BEFORE_ERODE = True
ERODE_PIXELS = 1
ERODE_KERNEL_SHAPE = "ellipse"   # "ellipse" / "rect" / "cross"

# 输出文件名
OUT_COLOR_PNG_WITH_IDS    = "top_percent_color_mask_with_ids.png"
OUT_COLOR_PNG_NO_IDS      = "top_percent_color_mask_no_ids.png"
OUT_COLOR_HD_PNG_WITH_IDS = "top_percent_color_mask_hd_with_ids.png"
OUT_COLOR_HD_PNG_NO_IDS   = "top_percent_color_mask_hd_no_ids.png"

OUT_LABEL_TIF = "top_percent_labelmask_uint16.tif"
OUT_CSV       = "top_percent_roi_summary.csv"

# 输出文件夹名
ALL_FILTERED_MASK_DIR = "all_filtered_masks_each_roi"
SELECTED_MASK_DIR     = "selected_top_masks_each_roi"


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def roi_centroid_from_mask(mask01: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask01 > 0)
    if ys.size == 0:
        return 0.0, 0.0
    return float(np.mean(ys)), float(np.mean(xs))

def draw_text_with_bg(
    img_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    font,
    font_scale: float,
    thickness: int,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    bg_alpha: float,
):
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 2

    x0 = max(0, x - pad)
    y0 = max(0, y - th - pad)
    x1 = min(img_bgr.shape[1] - 1, x + tw + pad)
    y1 = min(img_bgr.shape[0] - 1, y + baseline + pad)

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, thickness=-1)
    img_bgr[:] = cv2.addWeighted(overlay, bg_alpha, img_bgr, 1.0 - bg_alpha, 0.0)

    cv2.putText(
        img_bgr, text, (x, y),
        font, font_scale, text_color, thickness, lineType=cv2.LINE_AA
    )

def build_full_mask_from_stat(stat_entry, Ly: int, Lx: int) -> np.ndarray:
    """
    Build full-frame binary mask (uint8, 0/1) from Suite2p stat entry.
    """
    mask = np.zeros((Ly, Lx), dtype=np.uint8)
    yy = np.asarray(stat_entry["ypix"], dtype=np.int32)
    xx = np.asarray(stat_entry["xpix"], dtype=np.int32)
    ok = (yy >= 0) & (yy < Ly) & (xx >= 0) & (xx < Lx)
    mask[yy[ok], xx[ok]] = 1
    return mask

def largest_connected_component(mask01: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Return:
      largest_cc_mask01
      largest_cc_area
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.max() == 0:
        return np.zeros_like(mask01, dtype=np.uint8), 0

    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if nlab <= 1:
        return np.zeros_like(mask01, dtype=np.uint8), 0

    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = int(np.argmax(areas)) + 1
    best_area = int(stats[best_idx, cv2.CC_STAT_AREA])

    out = (lab == best_idx).astype(np.uint8)
    return out, best_area

def safe_trace_max_per_roi(F: np.ndarray) -> np.ndarray:
    if F is None:
        return None
    F = np.asarray(F)
    if F.ndim != 2:
        return None
    return np.nanmax(F, axis=1)

def rank_roi_ids_by_trace_max(roi_ids: np.ndarray, trace_max_all: np.ndarray) -> np.ndarray:
    roi_ids = np.asarray(roi_ids, dtype=int)
    scores = trace_max_all[roi_ids].astype(np.float64)
    scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.lexsort((roi_ids, -scores))  # primary: -score, tie: rid
    return roi_ids[order]

def make_hd_image_nearest(img: np.ndarray, upsample: int) -> np.ndarray:
    if upsample <= 1:
        return img.copy()
    h, w = img.shape[:2]
    return cv2.resize(img, (w * upsample, h * upsample), interpolation=cv2.INTER_NEAREST)

def save_single_mask_tif(mask01: np.ndarray, out_path: Path, value: int = 255):
    m = (mask01 > 0).astype(np.uint8)
    if value == 255:
        m = m * 255
    elif value == 1:
        m = m
    else:
        raise ValueError("SINGLE_MASK_TIF_VALUE must be 255 or 1")
    tiff.imwrite(str(out_path), m, photometric="minisblack", metadata=None)

def get_erode_kernel(shape_name: str = "ellipse") -> np.ndarray:
    shape_name = str(shape_name).lower()
    if shape_name == "ellipse":
        shape = cv2.MORPH_ELLIPSE
    elif shape_name == "rect":
        shape = cv2.MORPH_RECT
    elif shape_name == "cross":
        shape = cv2.MORPH_CROSS
    else:
        raise ValueError('ERODE_KERNEL_SHAPE must be "ellipse", "rect", or "cross"')
    return cv2.getStructuringElement(shape, (3, 3))

def fill_enclosed_holes(mask01: np.ndarray) -> np.ndarray:
    """
    Fill only enclosed holes inside a binary object.
    Holes connected to image border will remain background.
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.max() == 0:
        return mask01

    # invert: object->0, background->1
    inv = (1 - mask01).astype(np.uint8)

    # flood fill from border on the inverted image
    flood = inv.copy()
    h, w = flood.shape
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=2)

    # pixels still ==1 are enclosed holes
    holes = (flood == 1).astype(np.uint8)

    filled = ((mask01 > 0) | (holes > 0)).astype(np.uint8)
    return filled

def polish_single_mask(
    mask01: np.ndarray,
    fill_holes_before_erode: bool = True,
    erode_pixels: int = 1,
    erode_kernel_shape: str = "ellipse",
) -> np.ndarray:
    """
    For selected masks only:
    1) fill enclosed holes
    2) erode boundary by ERODE_PIXELS
    """
    m = (mask01 > 0).astype(np.uint8)
    if m.max() == 0:
        return m

    if bool(fill_holes_before_erode):
        m = fill_enclosed_holes(m)

    if int(erode_pixels) > 0:
        kernel = get_erode_kernel(str(erode_kernel_shape))
        eroded = cv2.erode(m, kernel, iterations=int(erode_pixels))

        # avoid returning empty mask after erosion
        if eroded.max() > 0:
            m = eroded

    return m.astype(np.uint8)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _summary_of_array(x: np.ndarray) -> dict:
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


def _save_trace_examples_png(F: np.ndarray, roi_ids_ranked: list[int], out_png: Path, top_n: int = 12) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if F is None or F.ndim != 2 or len(roi_ids_ranked) == 0:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        ax.text(0.5, 0.5, "No trace examples available", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(out_png), dpi=120)
        plt.close(fig)
        return str(out_png)

    sel = roi_ids_ranked[:max(1, int(top_n))]
    traces = F[np.asarray(sel, dtype=int)]
    traces = np.nan_to_num(traces, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)
    offset = 0.0
    for i, tr in enumerate(traces):
        tr0 = tr.astype(np.float64)
        tr0 = tr0 - np.percentile(tr0, 10)
        scale = max(np.std(tr0), 1e-6)
        trn = tr0 / scale
        ax.plot(trn + offset, linewidth=0.9, alpha=0.9, label=f"rid={sel[i]}")
        offset += 3.0
    ax.set_title("Trace Examples (ranked by trace max)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Normalized + offset")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=120)
    plt.close(fig)
    return str(out_png)


def select_rois_and_build_artifacts(
    plane0_dir: str | Path,
    output_root: str | Path,
    selection_config: dict | None = None,
) -> dict:
    """
    Importable adapter from STEP1 logic.
    Analysis ROI set:
      intensity-filtered iscell -> largest connected component area filter.
    Display subset:
      top_percent by trace max from analysis set.
    """
    cfg = {
        "min_largest_cc_area": MIN_LARGEST_CC_AREA,
        "top_percent": TOP_PERCENT,
        "analysis_only_iscell": True,
        "analysis_use_top_percent": False,
        "polish_selected_masks": POLISH_SELECTED_MASKS,
        "fill_holes_before_erode": FILL_HOLES_BEFORE_ERODE,
        "erode_pixels": ERODE_PIXELS,
        "erode_kernel_shape": ERODE_KERNEL_SHAPE,
        "single_mask_tif_value": SINGLE_MASK_TIF_VALUE,
        "hd_upsample": HD_UPSAMPLE,
        "trace_examples_top_n": 12,
        "asset_prefix": "final",
    }
    if selection_config:
        cfg.update(selection_config)

    plane0_dir = Path(plane0_dir).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir = output_root.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    roi_selection_dir = output_root

    stat = np.load(plane0_dir / "stat.npy", allow_pickle=True)
    ops = np.load(plane0_dir / "ops.npy", allow_pickle=True).item()
    iscell_path = plane0_dir / "iscell.npy"
    iscell = np.load(iscell_path, allow_pickle=True) if iscell_path.exists() else None
    F = np.load(plane0_dir / TRACE_NPY_NAME) if (plane0_dir / TRACE_NPY_NAME).exists() else None
    Fneu = np.load(plane0_dir / "Fneu.npy") if (plane0_dir / "Fneu.npy").exists() else None
    spks = np.load(plane0_dir / "spks.npy") if (plane0_dir / "spks.npy").exists() else None

    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    n_total = int(len(stat))
    trace_max_all = safe_trace_max_per_roi(F) if F is not None else np.full(n_total, np.nan, dtype=np.float32)

    candidate_ids = np.arange(n_total, dtype=int)
    if bool(cfg["analysis_only_iscell"]) and iscell is not None and iscell.shape[0] == n_total:
        candidate_ids = np.where(iscell[:, 0].astype(bool))[0]

    analysis_info = []
    for rid in candidate_ids:
        full_mask = build_full_mask_from_stat(stat[rid], Ly, Lx)
        largest_cc_mask, largest_cc_area = largest_connected_component(full_mask)
        if largest_cc_area >= int(cfg["min_largest_cc_area"]):
            analysis_info.append(
                {
                    "rid": int(rid),
                    "largest_cc_area": int(largest_cc_area),
                    "largest_cc_mask": largest_cc_mask,
                    "trace_max": float(trace_max_all[rid]) if trace_max_all is not None else float("nan"),
                }
            )

    if len(analysis_info) == 0:
        summary = {
            "execution_status": "executed",
            "counts": {
                "plane0_total": n_total,
                "candidate_count": int(len(candidate_ids)),
                "analysis_after_cc_area_filter": 0,
                "display_selected_count": 0,
            },
            "artifacts": {},
            "trace_parse": {
                "F_summary": _summary_of_array(np.array([])),
                "Fneu_summary": _summary_of_array(np.array([])),
                "spks_summary": _summary_of_array(np.array([])),
            },
            "config_used": cfg,
            "notes": ["No ROI survived CC area filtering."],
        }
        _write_json(roi_selection_dir / "selection_summary.json", summary)
        return summary

    kept_rids = np.array([d["rid"] for d in analysis_info], dtype=int)
    ranked_rids = rank_roi_ids_by_trace_max(kept_rids, trace_max_all)
    info_map = {int(d["rid"]): d for d in analysis_info}
    ranked_info = [info_map[int(rid)] for rid in ranked_rids]
    for rank, d in enumerate(ranked_info, start=1):
        d["rank"] = rank

    n_kept = len(ranked_info)
    n_select = max(1, int(math.ceil(n_kept * float(cfg["top_percent"]))))
    selected_info = ranked_info[:n_select]

    if bool(cfg["analysis_use_top_percent"]):
        analysis_final = selected_info
    else:
        analysis_final = ranked_info

    if bool(cfg["polish_selected_masks"]):
        for d in selected_info:
            polished = polish_single_mask(
                d["largest_cc_mask"],
                fill_holes_before_erode=bool(cfg["fill_holes_before_erode"]),
                erode_pixels=int(cfg["erode_pixels"]),
                erode_kernel_shape=str(cfg["erode_kernel_shape"]),
            )
            d["selected_mask_final"] = polished
            d["selected_mask_final_area"] = int(np.count_nonzero(polished))
    else:
        for d in selected_info:
            raw_mask = (d["largest_cc_mask"] > 0).astype(np.uint8)
            d["selected_mask_final"] = raw_mask
            d["selected_mask_final_area"] = int(np.count_nonzero(raw_mask))

    all_filtered_dir = ensure_dir(roi_selection_dir / ALL_FILTERED_MASK_DIR)
    selected_dir = ensure_dir(roi_selection_dir / SELECTED_MASK_DIR)

    analysis_label_mask = np.zeros((Ly, Lx), dtype=np.uint16)
    for rank, d in enumerate(analysis_final, start=1):
        cc_mask = d["largest_cc_mask"] > 0
        analysis_label_mask[cc_mask] = np.uint16(rank)
        out_name = (
            f"rank_{rank:04d}_rid_{int(d['rid']):04d}_area_{int(d['largest_cc_area']):05d}"
            f"_tracemax_{float(d['trace_max']):.4f}.tif"
        )
        save_single_mask_tif(
            d["largest_cc_mask"],
            all_filtered_dir / out_name,
            value=int(cfg["single_mask_tif_value"]),
        )

    display_label_mask = np.zeros((Ly, Lx), dtype=np.uint16)
    color_img_no_ids = np.zeros((Ly, Lx, 3), dtype=np.uint8)
    cmap = plt.get_cmap(CMAP_NAME)
    denom = max(len(selected_info), 1)

    rows_for_csv = []
    selected_rids = []
    for rank, d in enumerate(selected_info, start=1):
        rid = int(d["rid"])
        selected_rids.append(rid)
        final_mask = d["selected_mask_final"]
        display_label_mask[final_mask > 0] = np.uint16(rank)
        if hasattr(cmap, "N"):
            r, g, b, _ = cmap((rank - 1) % cmap.N)
        else:
            r, g, b, _ = cmap((rank - 1) / denom)
        rgb_u8 = np.clip(np.array([r, g, b]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        bgr = (int(rgb_u8[2]), int(rgb_u8[1]), int(rgb_u8[0]))
        color_img_no_ids[final_mask > 0] = bgr

        out_name = (
            f"rank_{rank:04d}_rid_{rid:04d}_rawarea_{int(d['largest_cc_area']):05d}"
            f"_finalarea_{int(d['selected_mask_final_area']):05d}_tracemax_{float(d['trace_max']):.4f}.tif"
        )
        save_single_mask_tif(
            final_mask,
            selected_dir / out_name,
            value=int(cfg["single_mask_tif_value"]),
        )
        rows_for_csv.append(
            {
                "rank": rank,
                "rid": rid,
                "largest_cc_area_raw": int(d["largest_cc_area"]),
                "final_area_polished": int(d["selected_mask_final_area"]),
                "trace_max": float(d["trace_max"]),
                "in_analysis_set": 1,
                "in_display_subset": 1,
            }
        )

    color_img_with_ids = color_img_no_ids.copy()
    for row, d in zip(rows_for_csv, selected_info):
        cy, cx = roi_centroid_from_mask(d["selected_mask_final"])
        x = int(np.clip(int(round(cx)), 0, Lx - 1))
        y = int(np.clip(int(round(cy)), 0, Ly - 1))
        draw_text_with_bg(
            color_img_with_ids,
            str(int(row["rank"])),
            (x + 2, y - 2),
            font=LABEL_FONT,
            font_scale=LABEL_FONT_SCALE,
            thickness=LABEL_THICKNESS,
            text_color=LABEL_TEXT_COLOR,
            bg_color=LABEL_BG_COLOR,
            bg_alpha=LABEL_BG_ALPHA,
        )

    color_img_hd_with_ids = make_hd_image_nearest(color_img_with_ids, int(cfg["hd_upsample"]))
    color_img_hd_no_ids = make_hd_image_nearest(color_img_no_ids, int(cfg["hd_upsample"]))

    asset_prefix = str(cfg.get("asset_prefix", "final"))
    overlay_with_ids = assets_dir / f"segmentation_overlay_{asset_prefix}.png"
    overlay_no_ids = assets_dir / f"segmentation_overlay_{asset_prefix}_no_ids.png"
    overlay_hd = assets_dir / f"segmentation_overlay_{asset_prefix}_hd.png"
    trace_png = assets_dir / f"trace_examples_{asset_prefix}.png"

    cv2.imwrite(str(overlay_with_ids), color_img_with_ids)
    cv2.imwrite(str(overlay_no_ids), color_img_no_ids)
    cv2.imwrite(str(overlay_hd), color_img_hd_with_ids)

    analysis_mask_path = roi_selection_dir / "analysis_mask_uint16.tif"
    display_mask_path = roi_selection_dir / "display_labelmask_uint16.tif"
    roi_csv_path = roi_selection_dir / "roi_summary.csv"
    tiff.imwrite(str(analysis_mask_path), analysis_label_mask, photometric="minisblack", metadata=None)
    tiff.imwrite(str(display_mask_path), display_label_mask, photometric="minisblack", metadata=None)

    with open(roi_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank", "rid", "largest_cc_area_raw", "final_area_polished", "trace_max",
                "in_analysis_set", "in_display_subset"
            ],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    trace_path = _save_trace_examples_png(F, [int(d["rid"]) for d in selected_info], trace_png, top_n=int(cfg["trace_examples_top_n"]))

    trace_parse = {
        "trace_count": int(F.shape[0]) if isinstance(F, np.ndarray) and F.ndim == 2 else 0,
        "trace_selection_rule": f"top trace max among display subset, top_n={int(cfg['trace_examples_top_n'])}",
        "trace_max_summary": _summary_of_array(np.asarray(trace_max_all[selected_rids])) if len(selected_rids) > 0 else _summary_of_array(np.array([])),
        "F_summary": _summary_of_array(F.flatten()) if isinstance(F, np.ndarray) else _summary_of_array(np.array([])),
        "Fneu_summary": _summary_of_array(Fneu.flatten()) if isinstance(Fneu, np.ndarray) else _summary_of_array(np.array([])),
        "spks_summary": _summary_of_array(spks.flatten()) if isinstance(spks, np.ndarray) else _summary_of_array(np.array([])),
    }

    summary = {
        "execution_status": "executed",
        "counts": {
            "plane0_total": n_total,
            "candidate_count": int(len(candidate_ids)),
            "analysis_after_cc_area_filter": int(len(ranked_info)),
            "analysis_final_count": int(len(analysis_final)),
            "display_selected_count": int(len(selected_info)),
        },
        "thresholds_used": {
            "min_largest_cc_area": int(cfg["min_largest_cc_area"]),
            "top_percent": float(cfg["top_percent"]),
            "polish_selected_masks": bool(cfg["polish_selected_masks"]),
            "erode_pixels": int(cfg["erode_pixels"]),
            "analysis_use_top_percent": bool(cfg["analysis_use_top_percent"]),
        },
        "trace_parse": trace_parse,
        "artifacts": {
            "analysis_mask_uint16_tif": str(analysis_mask_path),
            "display_labelmask_uint16_tif": str(display_mask_path),
            "roi_summary_csv": str(roi_csv_path),
            "all_filtered_masks_dir": str(all_filtered_dir),
            "selected_top_masks_dir": str(selected_dir),
            "segmentation_overlay_with_ids_png": str(overlay_with_ids),
            "segmentation_overlay_no_ids_png": str(overlay_no_ids),
            "segmentation_overlay_hd_png": str(overlay_hd),
            "trace_examples_png": str(trace_path),
        },
        "config_used": cfg,
    }
    _write_json(roi_selection_dir / "selection_summary.json", summary)
    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    if PLANE0_DIR is None:
        raise ValueError(
            "STEP1_PLANE0_DIR is not configured. "
            "Set STEP1_PLANE0_DIR to the suite2p plane0_intensityfilt folder before running this script."
        )
    if not PLANE0_DIR.exists():
        raise FileNotFoundError(f"Configured STEP1_PLANE0_DIR does not exist: {PLANE0_DIR}")

    root_name = PLANE0_DIR.parents[1].name if len(PLANE0_DIR.parents) >= 2 else PLANE0_DIR.name
    out_dir = ensure_dir(OUT_ROOT / root_name)

    print(f"[INFO] PLANE0_DIR = {PLANE0_DIR}")
    print(f"[INFO] OUT_DIR    = {out_dir}")

    stat = np.load(PLANE0_DIR / "stat.npy", allow_pickle=True)
    ops  = np.load(PLANE0_DIR / "ops.npy", allow_pickle=True).item()

    Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
    n_total = len(stat)

    print(f"[INFO] Ly={Ly}, Lx={Lx}, total ROIs={n_total}")

    trace_path = PLANE0_DIR / TRACE_NPY_NAME
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    F = np.load(trace_path)
    trace_max_all = safe_trace_max_per_roi(F)
    if trace_max_all is None or trace_max_all.shape[0] != n_total:
        raise ValueError(
            f"Trace shape mismatch: F.shape={getattr(F, 'shape', None)}, n_total={n_total}"
        )

    print(f"[INFO] Loaded trace: {trace_path}, shape={F.shape}")

    # --------------------------------------------------------
    # Step 1: compute largest CC area for every ROI
    # --------------------------------------------------------
    kept_info = []

    print("[INFO] Computing largest connected-component area for each ROI ...")
    for rid in range(n_total):
        full_mask = build_full_mask_from_stat(stat[rid], Ly, Lx)
        largest_cc_mask, largest_cc_area = largest_connected_component(full_mask)

        if largest_cc_area >= MIN_LARGEST_CC_AREA:
            kept_info.append({
                "rid": rid,
                "largest_cc_area": int(largest_cc_area),
                "largest_cc_mask": largest_cc_mask,   # raw largest-CC mask
                "trace_max": float(trace_max_all[rid]),
            })

    n_kept = len(kept_info)
    print(f"[INFO] Kept after largest-CC-area filter (>= {MIN_LARGEST_CC_AREA}) = {n_kept}")

    if n_kept == 0:
        print("[WARN] No ROI survived filtering. Exit.")
        return

    # --------------------------------------------------------
    # Step 2: sort by trace max descending
    # --------------------------------------------------------
    kept_rids = np.array([d["rid"] for d in kept_info], dtype=int)
    ranked_rids = rank_roi_ids_by_trace_max(kept_rids, trace_max_all)

    info_map = {int(d["rid"]): d for d in kept_info}
    ranked_info = [info_map[int(rid)] for rid in ranked_rids]

    # 给所有过滤后的 ROI 加 rank
    for rank, d in enumerate(ranked_info, start=1):
        d["rank"] = rank

    # --------------------------------------------------------
    # Step 3: take top X%
    # --------------------------------------------------------
    n_select = max(1, int(math.ceil(n_kept * TOP_PERCENT)))
    selected_info = ranked_info[:n_select]

    print(f"[INFO] TOP_PERCENT = {TOP_PERCENT:.3f}")
    print(f"[INFO] Selected top N = {n_select}")

    # --------------------------------------------------------
    # Step 3.5: polish selected masks only
    # --------------------------------------------------------
    if POLISH_SELECTED_MASKS:
        print("[INFO] Polishing selected masks: fill enclosed holes + erode boundary ...")
        for d in selected_info:
            raw_mask = d["largest_cc_mask"]
            polished = polish_single_mask(raw_mask)
            d["selected_mask_final"] = polished
            d["selected_mask_final_area"] = int(np.count_nonzero(polished))
    else:
        for d in selected_info:
            raw_mask = (d["largest_cc_mask"] > 0).astype(np.uint8)
            d["selected_mask_final"] = raw_mask
            d["selected_mask_final_area"] = int(np.count_nonzero(raw_mask))

    # --------------------------------------------------------
    # Step 4A: save all filtered masks (raw largest-CC version)
    # --------------------------------------------------------
    all_filtered_dir = ensure_dir(out_dir / ALL_FILTERED_MASK_DIR)
    print(f"[INFO] Saving all filtered masks to: {all_filtered_dir}")

    for d in ranked_info:
        rid = int(d["rid"])
        rank = int(d["rank"])
        area = int(d["largest_cc_area"])
        trace_max = float(d["trace_max"])
        cc_mask = d["largest_cc_mask"]

        out_name = (
            f"rank_{rank:04d}_rid_{rid:04d}"
            f"_area_{area:05d}"
            f"_tracemax_{trace_max:.4f}.tif"
        )
        save_single_mask_tif(
            cc_mask,
            all_filtered_dir / out_name,
            value=SINGLE_MASK_TIF_VALUE
        )

    # --------------------------------------------------------
    # Step 4B: save selected top masks (POLISHED version)
    # --------------------------------------------------------
    selected_dir = ensure_dir(out_dir / SELECTED_MASK_DIR)
    print(f"[INFO] Saving selected top masks to: {selected_dir}")

    for rank, d in enumerate(selected_info, start=1):
        rid = int(d["rid"])
        raw_area = int(d["largest_cc_area"])
        final_area = int(d["selected_mask_final_area"])
        trace_max = float(d["trace_max"])
        final_mask = d["selected_mask_final"]

        out_name = (
            f"rank_{rank:04d}_rid_{rid:04d}"
            f"_rawarea_{raw_area:05d}"
            f"_finalarea_{final_area:05d}"
            f"_tracemax_{trace_max:.4f}.tif"
        )
        save_single_mask_tif(
            final_mask,
            selected_dir / out_name,
            value=SINGLE_MASK_TIF_VALUE
        )

    # --------------------------------------------------------
    # Step 5: build color PNG + label TIFF using POLISHED masks
    # --------------------------------------------------------
    color_img_no_ids = np.zeros((Ly, Lx, 3), dtype=np.uint8)   # BGR for cv2
    label_mask = np.zeros((Ly, Lx), dtype=np.uint16)

    cmap = plt.get_cmap(CMAP_NAME)
    denom = max(n_select, 1)

    rows_for_csv = []

    for rank, d in enumerate(selected_info, start=1):
        rid = int(d["rid"])
        cc_mask = d["selected_mask_final"]   # use polished mask here
        trace_max = float(d["trace_max"])
        raw_area = int(d["largest_cc_area"])
        final_area = int(d["selected_mask_final_area"])

        if hasattr(cmap, "N"):
            r, g, b, _ = cmap((rank - 1) % cmap.N)
        else:
            r, g, b, _ = cmap((rank - 1) / denom)

        rgb_u8 = np.clip(np.array([r, g, b]) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        bgr = (int(rgb_u8[2]), int(rgb_u8[1]), int(rgb_u8[0]))

        color_img_no_ids[cc_mask > 0] = bgr
        label_mask[cc_mask > 0] = np.uint16(rank)

        rows_for_csv.append({
            "rank": rank,
            "rid": rid,
            "largest_cc_area_raw": raw_area,
            "final_area_polished": final_area,
            "trace_max": trace_max,
        })

    # 再复制出“有编号版本”
    color_img_with_ids = color_img_no_ids.copy()

    # draw ids only on with_ids image
    if DRAW_IDS:
        for row, d in zip(rows_for_csv, selected_info):
            rank = int(row["rank"])
            cc_mask = d["selected_mask_final"]
            cy, cx = roi_centroid_from_mask(cc_mask)

            x = int(np.clip(int(round(cx)), 0, Lx - 1))
            y = int(np.clip(int(round(cy)), 0, Ly - 1))
            org = (x + 2, y - 2)

            draw_text_with_bg(
                color_img_with_ids,
                str(rank),
                org,
                font=LABEL_FONT,
                font_scale=LABEL_FONT_SCALE,
                thickness=LABEL_THICKNESS,
                text_color=LABEL_TEXT_COLOR,
                bg_color=LABEL_BG_COLOR,
                bg_alpha=LABEL_BG_ALPHA,
            )

    if DRAW_TOTAL_N:
        total_txt = f"top {TOP_PERCENT*100:.1f}%  N = {n_select}"

        draw_text_with_bg(
            color_img_with_ids,
            total_txt,
            (10, 30),
            font=LABEL_FONT,
            font_scale=0.9,
            thickness=2,
            text_color=TOTAL_TEXT_COLOR,
            bg_color=TOTAL_BG_COLOR,
            bg_alpha=TOTAL_BG_ALPHA,
        )

    # 高清图
    color_img_hd_no_ids = make_hd_image_nearest(color_img_no_ids, HD_UPSAMPLE)
    color_img_hd_with_ids = make_hd_image_nearest(color_img_with_ids, HD_UPSAMPLE)

    # --------------------------------------------------------
    # Step 6: save
    # --------------------------------------------------------
    pct_tag = int(round(TOP_PERCENT * 100))

    out_png_with_ids = out_dir / f"{Path(OUT_COLOR_PNG_WITH_IDS).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct.png"
    out_png_no_ids   = out_dir / f"{Path(OUT_COLOR_PNG_NO_IDS).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct.png"

    out_hd_png_with_ids = out_dir / f"{Path(OUT_COLOR_HD_PNG_WITH_IDS).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct_x{HD_UPSAMPLE}.png"
    out_hd_png_no_ids   = out_dir / f"{Path(OUT_COLOR_HD_PNG_NO_IDS).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct_x{HD_UPSAMPLE}.png"

    out_tif = out_dir / f"{Path(OUT_LABEL_TIF).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct.tif"
    out_csv = out_dir / f"{Path(OUT_CSV).stem}_minCC{MIN_LARGEST_CC_AREA}_top{pct_tag:02d}pct.csv"

    cv2.imwrite(str(out_png_with_ids), color_img_with_ids)
    cv2.imwrite(str(out_png_no_ids), color_img_no_ids)
    cv2.imwrite(str(out_hd_png_with_ids), color_img_hd_with_ids)
    cv2.imwrite(str(out_hd_png_no_ids), color_img_hd_no_ids)

    tiff.imwrite(str(out_tif), label_mask, photometric="minisblack", metadata=None)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "rid", "largest_cc_area_raw", "final_area_polished", "trace_max"]
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f"[OK] saved color PNG with IDs    : {out_png_with_ids}")
    print(f"[OK] saved color PNG no IDs      : {out_png_no_ids}")
    print(f"[OK] saved HD color PNG with IDs : {out_hd_png_with_ids}")
    print(f"[OK] saved HD color PNG no IDs   : {out_hd_png_no_ids}")
    print(f"[OK] saved label TIFF            : {out_tif}")
    print(f"[OK] saved CSV                   : {out_csv}")
    print(f"[OK] saved all filtered masks    : {all_filtered_dir}")
    print(f"[OK] saved selected top masks    : {selected_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()
