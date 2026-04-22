#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


def _normalize_image_for_png(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32, copy=False)
    lo = float(np.percentile(img_f, 1))
    hi = float(np.percentile(img_f, 99))
    if hi <= lo + 1e-8:
        return np.zeros_like(img_f, dtype=np.float32)
    out = (img_f - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _save_gray_png_from_array(img: np.ndarray, out_path: Path, title: str | None = None) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_n = _normalize_image_for_png(img)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(img_n, cmap="gray", interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout(pad=0.2)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return str(out_path)


def save_projection_pngs(std_tif: str | Path, mip_tif: str | Path, output_dir: str | Path, prefix: str) -> dict[str, str]:
    output_dir = Path(output_dir)
    std_arr = tiff.imread(str(std_tif))
    mip_arr = tiff.imread(str(mip_tif))

    std_png = output_dir / f"{prefix}_STD.png"
    mip_png = output_dir / f"{prefix}_MIP.png"

    std_png_path = _save_gray_png_from_array(std_arr, std_png, title=f"{prefix.upper()} STD")
    mip_png_path = _save_gray_png_from_array(mip_arr, mip_png, title=f"{prefix.upper()} MIP")

    return {
        f"{prefix}_std_png": std_png_path,
        f"{prefix}_mip_png": mip_png_path,
    }


def save_motion_curve_png(
    raw_shifts_npy: str | Path | None,
    final_shifts_npy: str | Path | None,
    output_png: str | Path,
) -> str:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=120)

    has_curve = False

    if raw_shifts_npy is not None and Path(raw_shifts_npy).exists():
        raw_shifts = np.load(str(raw_shifts_npy))
        raw_mag = np.sqrt(raw_shifts[:, 0] ** 2 + raw_shifts[:, 1] ** 2)
        ax.plot(raw_mag, label="raw", linewidth=1.2)
        has_curve = True

    if final_shifts_npy is not None and Path(final_shifts_npy).exists():
        final_shifts = np.load(str(final_shifts_npy))
        final_mag = np.sqrt(final_shifts[:, 0] ** 2 + final_shifts[:, 1] ** 2)
        ax.plot(final_mag, label="final", linewidth=1.2)
        has_curve = True

    if has_curve:
        ax.legend(loc="upper right")
        ax.set_ylabel("Shift magnitude (px)")
        ax.set_xlabel("Frame index")
    else:
        ax.text(0.5, 0.5, "No motion shift curves available", ha="center", va="center")
        ax.set_axis_off()

    ax.set_title("Rigid Motion Curve")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=120)
    plt.close(fig)

    return str(output_png)


def save_snr_comparison_png(raw_snr: float | None, final_snr: float | None, output_png: str | Path) -> str:
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels = ["raw", "final"]
    values = [
        None if raw_snr is None else float(raw_snr),
        None if final_snr is None else float(final_snr),
    ]
    colors = ["#4C78A8", "#59A14F"]

    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=120)
    numeric_values = [0.0 if v is None else v for v in values]
    bars = ax.bar(labels, numeric_values, color=colors, width=0.6)

    max_numeric = max([v for v in numeric_values] + [0.0])
    text_offset = 0.03 * max(1.0, max_numeric)
    for bar, val in zip(bars, values):
        y = bar.get_height()
        if val is None:
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + text_offset, "N/A", ha="center", va="bottom")
            bar.set_hatch("//")
            bar.set_alpha(0.45)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2.0, y + text_offset, f"{val:.3g}", ha="center", va="bottom")

    ax.set_ylabel("SNR proxy (N/A shown as hatched bar)")
    ax.set_title("Fixed-ROI SNR Proxy Comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max_numeric * 1.25 + text_offset)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=120)
    plt.close(fig)

    return str(output_png)


def create_comparison_png_assets(raw_metrics: dict[str, Any], final_metrics: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_artifacts = raw_metrics.get("artifacts", {})
    final_artifacts = final_metrics.get("artifacts", {})

    raw_pngs = save_projection_pngs(
        std_tif=raw_artifacts["std_tif"],
        mip_tif=raw_artifacts["mip_tif"],
        output_dir=output_dir,
        prefix="raw",
    )
    final_pngs = save_projection_pngs(
        std_tif=final_artifacts["std_tif"],
        mip_tif=final_artifacts["mip_tif"],
        output_dir=output_dir,
        prefix="final",
    )

    motion_curve = save_motion_curve_png(
        raw_shifts_npy=raw_artifacts.get("rigid_shifts_npy"),
        final_shifts_npy=final_artifacts.get("rigid_shifts_npy"),
        output_png=output_dir / "motion_curve.png",
    )

    raw_snr = raw_metrics.get("snr_metric", {}).get("snr")
    final_snr = final_metrics.get("snr_metric", {}).get("snr")
    snr_png = save_snr_comparison_png(raw_snr=raw_snr, final_snr=final_snr, output_png=output_dir / "snr_comparison.png")

    return {
        "raw_STD_png": raw_pngs["raw_std_png"],
        "raw_MIP_png": raw_pngs["raw_mip_png"],
        "final_STD_png": final_pngs["final_std_png"],
        "final_MIP_png": final_pngs["final_mip_png"],
        "motion_curve_png": motion_curve,
        "snr_comparison_png": snr_png,
    }
