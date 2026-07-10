#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Suite2p downstream segmentation pipeline with precomputed projections,
compact-cell-body defaults, and intensity-based post-filtering on MAX/STD.

What this script does:
1) Read a TIFF stack (supports 3D: (T,H,W) or (H,W,T); supports 4D: (A,B,H,W) or (H,W,A,B)).
2) Save a working movie "img_cat.tif" under output folder "<parent_folder>_seg".
3) Save MAX and STD projections (MAX.tif / STD.tif) for QC.
4) Run suite2p with NeuroPilot downstream detection parameters.
5) Post-filter suite2p "cell" ROIs (iscell==1 by default):
   - Normalize MAX/STD with robust percentile background subtraction -> [0,1].
   - For each ROI, compute mean intensity on normalized MAX and normalized STD.
   - Drop ROI if it's dim on BOTH MAX and STD (roi_mean_max<thr_max AND roi_mean_std<thr_std).
   - Save filtered results into a NEW folder: suite2p/plane0_intensityfilt (copy plane0; replace iscell.npy).
   - Print how many cells were dropped and how many remain.
6) Optionally delete the temporary "img_cat.tif".

Notes:
- The MAX/STD intensity filter only removes ROIs that are dim on both projections,
  so it stays conservative even when the upstream profiling stage tightens soma-first detection.
- Thresholds are defined on normalized images in [0,1], so INT_THR_MAX/STD are directly interpretable.
"""

from __future__ import annotations

import os
import shutil
import json
import hashlib
import importlib
import inspect
import pkgutil
import random
from pathlib import Path
from collections import Counter
import numpy as np
import tifffile as tiff

# Windows pip wheels for torch/suite2p dependencies may pull in multiple
# OpenMP runtimes. Allowing the duplicate runtime keeps suite2p importable in
# the dedicated downstream environment used by the public pipeline.
if os.name == "nt" and not os.environ.get("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_seed_env = os.environ.get("NEUROPILOT_RANDOM_SEED", "").strip()
if _seed_env:
    try:
        _seed = int(_seed_env)
        random.seed(_seed)
        np.random.seed(_seed)
    except ValueError:
        pass

try:
    import suite2p
except Exception:
    suite2p = None

def _iter_suite2p_modules():
    if suite2p is None:
        return
    seen = set()
    base_name = getattr(suite2p, "__name__", "suite2p")

    def _yield_mod(mod_name: str):
        if mod_name in seen:
            return
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            return
        seen.add(mod_name)
        yield mod

    for m in _yield_mod(base_name):
        yield m

    static_candidates = [
        f"{base_name}.run_s2p",
        f"{base_name}.ops",
        f"{base_name}.main",
        f"{base_name}.pipeline",
    ]
    for name in static_candidates:
        for m in _yield_mod(name):
            yield m

    pkg_path = getattr(suite2p, "__path__", None)
    if pkg_path is not None:
        for modinfo in pkgutil.iter_modules(pkg_path):
            if ("run" in modinfo.name) or (modinfo.name in {"ops", "main", "pipeline"}):
                for m in _yield_mod(f"{base_name}.{modinfo.name}"):
                    yield m


def _suite2p_default_ops() -> dict:
    """
    Resolve suite2p default ops across different package layouts.
    Compatible with variants where default_ops is exposed as:
    - suite2p.default_ops
    - suite2p.run_s2p.default_ops
    """
    if suite2p is None:
        raise ImportError("suite2p is not available in current environment")

    for mod in _iter_suite2p_modules():
        cand = getattr(mod, "default_ops", None)
        if callable(cand):
            try:
                return cand()
            except Exception:
                continue

    # Last-resort fallback: proceed with an empty ops dict and let caller fill
    # the fields used by this pipeline. This avoids hard failure when suite2p
    # package layout changes and default_ops symbol is not exported.
    return {}


def _suite2p_run(ops: dict):
    """
    Run suite2p across package layouts where run_s2p may be:
    - suite2p.run_s2p (callable)
    - suite2p.run_s2p.run_s2p (module + function)
    """
    if suite2p is None:
        raise ImportError("suite2p is not available in current environment")

    candidates = []
    for mod in _iter_suite2p_modules():
        for name in ("run_s2p", "run"):
            fn = getattr(mod, name, None)
            if callable(fn):
                candidates.append(fn)

    seen_ids = set()
    uniq_candidates = []
    for fn in candidates:
        fid = id(fn)
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        uniq_candidates.append(fn)

    for fn in uniq_candidates:
        try:
            sig = inspect.signature(fn)
            param_names = list(sig.parameters.keys())
        except Exception:
            param_names = []

        try:
            if "ops" in param_names and "db" in param_names:
                return fn(ops=ops, db={})
            if "ops" in param_names:
                return fn(ops=ops)
            if len(param_names) == 1:
                return fn(ops)

            # Fallback attempts for unknown signatures.
            try:
                return fn(ops=ops)
            except TypeError:
                return fn(ops)
        except TypeError:
            continue

    mod_file = getattr(suite2p, "__file__", None)
    attrs = sorted([a for a in dir(suite2p) if not a.startswith("_")])
    preview = attrs[:30]
    raise AttributeError(
        f"suite2p run entry not found or incompatible; module={mod_file}; attrs_preview={preview}"
    )


# ============================================================
# ====================== USER SETTINGS =======================
# ============================================================

# Input stack
# tif_path = "/mnt/md0/guoxun/03_reg/NeuroPilot_1210/2_validation_data/2p_1_whole_out_complex_0.75_highSNR/highSNR.tif"
# tif_path = "/mnt/md0/guoxun/03_reg/NeuroPilot_1210/2_validation_data/2p_1_whole_out_complex_0.75/cropped_warped.tif"
tif_path = "/mnt/md0/guoxun/03_reg/NeuroPilot_1210/0_MF4_spinal_cord_multi_dataset/ju_SNL_model_1hao/results_deepcad/ju_SNL_model_1hao_day1_iter1_DeepCAD"\
"/ju_SNL_model_1hao_day1.tif"

# Acquisition
FS_HZ = 10
NPLANES = 1
NCHANNELS = 1

# Registration stays off here because NeuroPilot final stacks are already
# motion-corrected upstream; downstream suite2p should focus on ROI detection.
DO_REGISTRATION = False
NONRIGID = False

# Soma-first detection defaults. A pre-segmentation profiling stage can refine
# these before suite2p runs, but the fallback defaults should already prefer
# compact cell bodies rather than neurite-like elongated regions.
DIAMETER_PX = 16
THRESH_SCALING = 1.0
ASPECT_MAX = 2.0
MAX_OVERLAP = 0.45
MIN_AREA_PX = 40

# Neuropil tuned for small ROIs
NEUROPIL_EXTRACT = True
INNER_NEUROPIL_RADIUS = 2
MIN_NEUROPIL_PIXELS = 80  # try 50~150

# Baseline for long recordings
BASELINE_MODE = "maximin"
WIN_BASELINE_SEC = 60
SIG_BASELINE = 10

# Save temporary concatenated movie?
DELETE_TEMP_MOVIE = True

# -------------------------
# Intensity-based post-filter (KEY REQUEST)
# We normalize MAX/STD to [0,1], so thresholds are absolute in [0,1].
# Drop ROI if dim in BOTH normalized MAX and normalized STD:
#   (roi_mean_max_norm < INT_THR_MAX) AND (roi_mean_std_norm < INT_THR_STD)
# -------------------------
if 'higSNR' in tif_path:
    INT_THR_MAX = 0.10       # try 0.05~0.15
    INT_THR_STD = 0.10       # neurites may have low std; try 0.03~0.10
else:
    INT_THR_MAX = 0.1
    INT_THR_STD = 0.1

INT_ONLY_ISCELL = True   # True: only filter those marked iscell==1; False: filter all ROIs

# Robust normalization for background / outliers
NORM_P_LOW = 5.0         # background percentile
NORM_P_HIGH = 99.5       # saturation percentile (avoid a few extreme bright pixels)


# ============================================================
# --------------------------- Helpers ------------------------
# ============================================================

def load_tif_anyshape(tif_path: str) -> np.ndarray:
    """
    Load tif and return a 3D stack shaped (T,H,W).
    Supports:
      - (T,H,W)
      - (H,W,T)  (heuristic)
      - (A,B,H,W) -> (A*B,H,W)
      - (H,W,A,B) -> (A*B,H,W)

    WARNING: loads into RAM. For huge stacks, switch to memmap/streaming.
    """
    arr = tiff.imread(tif_path)

    if arr.ndim == 3:
        # Heuristic: if last dim is smallest, assume (H,W,T)
        if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
            arr = np.transpose(arr, (2, 0, 1))  # (T,H,W)
        return arr  # (T,H,W)

    if arr.ndim == 4:
        # common: (A,B,H,W)
        if arr.shape[2] >= 16 and arr.shape[3] >= 16:
            A, B, H, W = arr.shape
            return arr.reshape(A * B, H, W)
        # less common: (H,W,A,B)
        if arr.shape[0] >= 16 and arr.shape[1] >= 16:
            H, W, A, B = arr.shape
            arr = np.transpose(arr, (2, 3, 0, 1))  # (A,B,H,W)
            return arr.reshape(A * B, H, W)

        # fallback
        A, B, H, W = arr.shape
        return arr.reshape(A * B, H, W)

    raise ValueError(f"Unsupported tif dims {arr.ndim}, shape={arr.shape} for {tif_path}")


def write_proj_max_std(stack_thw: np.ndarray, out_dir: Path, dtype_out=None):
    """
    Save MAX and STD images from (T,H,W) stack.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if dtype_out is None:
        dtype_out = stack_thw.dtype

    img_max = stack_thw.max(axis=0)
    tiff.imwrite(str(out_dir / "MAX.tif"), img_max.astype(dtype_out), bigtiff=False)

    img_std = stack_thw.astype(np.float32).std(axis=0)
    tiff.imwrite(str(out_dir / "STD.tif"), img_std.astype(np.float32), bigtiff=False)


def load_tif_pages_to_thw(tif_path: str, expect_hw: tuple[int, int] | None = None) -> np.ndarray:
    with tiff.TiffFile(str(tif_path)) as tf:
        page_shapes = [pg.shape for pg in tf.pages]

        if expect_hw is None:
            (H, W), _ = Counter(page_shapes).most_common(1)[0]
        else:
            H, W = expect_hw

        frames = []
        skipped = 0
        for pg in tf.pages:
            if pg.shape != (H, W):
                skipped += 1
                continue
            arr = np.squeeze(pg.asarray())
            if arr.shape == (H, W):
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



def normalize_robust_01(img: np.ndarray, p_low=5.0, p_high=99.5, eps=1e-6) -> np.ndarray:
    """
    Robust normalization to [0,1] using percentiles.
    Good when there is background offset and a few very bright pixels.
    """
    x = img.astype(np.float32, copy=False)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return y


def intensity_filter_suite2p_plane0_no_overwrite(
    plane0_dir: Path,
    max_tif: Path,
    std_tif: Path,
    out_plane0_dir: Path,
    thr_max: float = 0.10,
    thr_std: float = 0.10,
    only_iscell: bool = True,
    norm_p_low: float = 5.0,
    norm_p_high: float = 99.5,
):
    """
    Filter suite2p ROIs by intensity on MAX and STD projections.

    Steps:
      1) Read MAX/STD
      2) Robust percentile normalize each to [0,1]
      3) For each ROI compute mean intensity on normalized MAX and normalized STD
      4) Drop ROI if (mean_on_MAX < thr_max) AND (mean_on_STD < thr_std)
      5) Save a NEW plane0 folder with filtered iscell.npy + diagnostics

    Writes filtered results to out_plane0_dir (copy everything, overwrite only iscell.npy there).
    Also saves diagnostics:
      - roi_mean_on_MAX_norm.npy
      - roi_mean_on_STD_norm.npy
      - roi_keep_by_max_std_norm.npy
      - MAX_norm.tif / STD_norm.tif (for QC)
      - intensity_filter_report.txt
    """
    plane0_dir = Path(plane0_dir)
    out_plane0_dir = Path(out_plane0_dir)
    max_tif = Path(max_tif)
    std_tif = Path(std_tif)

    if not (plane0_dir / "stat.npy").exists():
        raise FileNotFoundError(f"Missing stat.npy in {plane0_dir}")
    if not (plane0_dir / "iscell.npy").exists():
        raise FileNotFoundError(f"Missing iscell.npy in {plane0_dir}")
    if not max_tif.exists():
        raise FileNotFoundError(f"Missing MAX.tif: {max_tif}")
    if not std_tif.exists():
        raise FileNotFoundError(f"Missing STD.tif: {std_tif}")

    # ---- prepare output folder (copy plane0 outputs) ----
    if out_plane0_dir.exists():
        shutil.rmtree(out_plane0_dir)
    out_plane0_dir.mkdir(parents=True, exist_ok=True)

    for p in plane0_dir.iterdir():
        if p.is_file():
            shutil.copy2(p, out_plane0_dir / p.name)

    # ---- load from output dir (operate only on copy) ----
    stat = np.load(out_plane0_dir / "stat.npy", allow_pickle=True)
    iscell = np.load(out_plane0_dir / "iscell.npy", allow_pickle=True)

    img_max_raw = tiff.imread(str(max_tif)).astype(np.float32)
    img_std_raw = tiff.imread(str(std_tif)).astype(np.float32)

    # Robust background-aware normalization
    img_max = normalize_robust_01(img_max_raw, p_low=norm_p_low, p_high=norm_p_high)
    img_std = normalize_robust_01(img_std_raw, p_low=norm_p_low, p_high=norm_p_high)

    # Save normalized projections for QC
    tiff.imwrite(str(out_plane0_dir / "MAX_norm.tif"), img_max.astype(np.float32), bigtiff=False)
    tiff.imwrite(str(out_plane0_dir / "STD_norm.tif"), img_std.astype(np.float32), bigtiff=False)

    roi_mean_max = np.full(len(stat), np.nan, dtype=np.float32)
    roi_mean_std = np.full(len(stat), np.nan, dtype=np.float32)

    for i, s in enumerate(stat):
        yp = s["ypix"].astype(np.int32)
        xp = s["xpix"].astype(np.int32)
        if yp.size == 0:
            continue
        roi_mean_max[i] = float(img_max[yp, xp].mean())
        roi_mean_std[i] = float(img_std[yp, xp].mean())

    if only_iscell:
        idx = np.where(iscell[:, 0].astype(bool))[0]
        before = int(iscell[:, 0].sum())
    else:
        idx = np.arange(len(stat))
        before = int(len(stat))

    keep = np.ones(len(stat), dtype=bool)

    # drop if dim in BOTH normalized projections
    dim_both = (roi_mean_max < float(thr_max)) & (roi_mean_std < float(thr_std))
    keep[idx] = ~dim_both[idx]

    iscell_new = iscell.copy()
    iscell_new[~keep, 0] = 0

    after = int(iscell_new[:, 0].sum()) if only_iscell else int(keep.sum())
    dropped = before - after

    # ---- save diagnostics ----
    np.save(out_plane0_dir / "iscell.npy", iscell_new)
    np.save(out_plane0_dir / "roi_mean_on_MAX_norm.npy", roi_mean_max)
    np.save(out_plane0_dir / "roi_mean_on_STD_norm.npy", roi_mean_std)
    np.save(out_plane0_dir / "roi_keep_by_max_std_norm.npy", keep)

    with open(out_plane0_dir / "intensity_filter_report.txt", "w", encoding="utf-8") as f:
        f.write("=== Intensity filter (normalized MAX/STD) ===\n")
        f.write(f"norm_p_low  = {norm_p_low}\n")
        f.write(f"norm_p_high = {norm_p_high}\n")
        f.write(f"thr_max (on MAX_norm) = {thr_max}\n")
        f.write(f"thr_std (on STD_norm) = {thr_std}\n")
        f.write(f"only_iscell = {only_iscell}\n")
        f.write(f"before = {before}\n")
        f.write(f"after  = {after}\n")
        f.write(f"dropped = {dropped}\n")
        f.write("\n--- raw projection stats ---\n")
        f.write(f"MAX_raw: min={float(img_max_raw.min()):.6g}, max={float(img_max_raw.max()):.6g}\n")
        f.write(f"STD_raw: min={float(img_std_raw.min()):.6g}, max={float(img_std_raw.max()):.6g}\n")

    print("\n[Intensity filter by MAX+STD (normalized)]")
    print(f"  norm: p_low={norm_p_low}, p_high={norm_p_high}  -> [0,1]")
    print(f"  thr_max = {thr_max:.3f}  (on MAX_norm)")
    print(f"  thr_std = {thr_std:.3f}  (on STD_norm)")
    print(f"  Dropped: {dropped}")
    print(f"  Remaining: {after}")
    print(f"  Saved filtered plane0 to: {out_plane0_dir}\n")

    return out_plane0_dir, dropped, after


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json_if_exists(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fingerprint(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _save_effective_ops_snapshot(plane0_dir: Path, out_path: Path) -> str | None:
    ops_path = plane0_dir / "ops.npy"
    if not ops_path.exists():
        return None
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
    except Exception:
        return None
    keys = [
        "fs",
        "nplanes",
        "nchannels",
        "do_registration",
        "nonrigid",
        "diameter",
        "threshold_scaling",
        "aspect",
        "max_overlap",
        "min_area",
        "allow_overlap",
        "inner_neuropil_radius",
        "min_neuropil_pixels",
        "maxregshift",
        "block_size",
    ]
    payload = {k: ops.get(k) for k in keys}
    _write_json(out_path, payload)
    return str(out_path)


def run_suite2p_segmentation(
    input_tif_path: str | Path,
    output_root: str | Path,
    config: dict | None = None,
) -> dict:
    """
    Importable pipeline adapter for STEP0 behavior.
    """
    cfg = {
        "fs_hz": FS_HZ,
        "nplanes": NPLANES,
        "nchannels": NCHANNELS,
        "do_registration": DO_REGISTRATION,
        "nonrigid": NONRIGID,
        "diameter_px": DIAMETER_PX,
        "threshold_scaling": THRESH_SCALING,
        "aspect_max": ASPECT_MAX,
        "max_overlap": MAX_OVERLAP,
        "min_area_px": MIN_AREA_PX,
        "neuropil_extract": NEUROPIL_EXTRACT,
        "inner_neuropil_radius": INNER_NEUROPIL_RADIUS,
        "min_neuropil_pixels": MIN_NEUROPIL_PIXELS,
        "baseline_mode": BASELINE_MODE,
        "win_baseline_sec": WIN_BASELINE_SEC,
        "sig_baseline": SIG_BASELINE,
        "delete_temp_movie": DELETE_TEMP_MOVIE,
        "int_thr_max": 0.10,
        "int_thr_std": 0.10,
        "int_only_iscell": INT_ONLY_ISCELL,
        "norm_p_low": NORM_P_LOW,
        "norm_p_high": NORM_P_HIGH,
    }
    if config:
        cfg.update(config)

    input_tif_path = Path(input_tif_path).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    params_payload = {
        "input_tif_path": str(input_tif_path),
        "output_root": str(output_root),
        "config": cfg,
    }
    params_fp = _fingerprint(params_payload)
    sidecar_path = output_root / "segmentation.params.json"
    existing_sidecar = _read_json_if_exists(sidecar_path)
    plane0_dir = output_root / "suite2p" / "plane0"
    plane0_intensity_dir = output_root / "suite2p" / "plane0_intensityfilt"

    if (
        existing_sidecar is not None
        and existing_sidecar.get("config_fingerprint") == params_fp
        and plane0_intensity_dir.exists()
        and (plane0_intensity_dir / "iscell.npy").exists()
    ):
        stat = np.load(plane0_dir / "stat.npy", allow_pickle=True) if (plane0_dir / "stat.npy").exists() else []
        iscell = np.load(plane0_intensity_dir / "iscell.npy", allow_pickle=True)
        effective_ops_path = _save_effective_ops_snapshot(plane0_dir, output_root / "suite2p_effective_ops.json")
        return {
            "execution_status": "skipped",
            "skip_reason": "matched_fingerprint",
            "config_fingerprint": params_fp,
            "sidecar_path": str(sidecar_path),
            "input_tif_path": str(input_tif_path),
            "output_root": str(output_root),
            "suite2p_registration_used": bool(cfg["do_registration"]),
            "artifacts": {
                "plane0_dir": str(plane0_dir),
                "plane0_intensityfilt_dir": str(plane0_intensity_dir),
                "max_tif": str(output_root / "MAX.tif"),
                "std_tif": str(output_root / "STD.tif"),
                "effective_ops_json": effective_ops_path,
            },
            "counts": {
                "plane0_total": int(len(stat)),
                "after_intensity_filter": int(np.sum(iscell[:, 0] > 0)),
            },
        }

    if suite2p is None:
        raise ImportError("suite2p is not available in current environment")

    stack = load_tif_pages_to_thw(str(input_tif_path))
    img_cat_name = str(output_root / "img_cat.tif")
    tiff.imwrite(img_cat_name, stack, bigtiff=True)
    write_proj_max_std(stack, output_root)
    del stack

    save_root = str(output_root) + "/"
    ops = _suite2p_default_ops()
    ops["data_path"] = [str(Path(img_cat_name).parent)]
    ops["tiff_list"] = [Path(img_cat_name).name]
    ops["save_path0"] = save_root
    ops["fs"] = float(cfg["fs_hz"])
    ops["nplanes"] = int(cfg["nplanes"])
    ops["nchannels"] = int(cfg["nchannels"])
    ops["bin_size"] = 1
    ops["nbinned"] = 20000
    ops["max_iterations"] = 40
    ops["highpass_time"] = 0
    ops["use_builtin_classifier"] = False
    ops["do_registration"] = bool(cfg["do_registration"])
    ops["nonrigid"] = bool(cfg["nonrigid"])
    ops["block_size"] = [128, 128]
    ops["maxregshift"] = 0.1
    ops["diameter"] = int(cfg["diameter_px"])
    ops["threshold_scaling"] = float(cfg["threshold_scaling"])
    ops["aspect"] = float(cfg["aspect_max"])
    ops["max_overlap"] = float(cfg["max_overlap"])
    ops["min_area"] = float(cfg["min_area_px"])
    ops["allow_overlap"] = True
    ops["neuropil_extract"] = bool(cfg["neuropil_extract"])
    ops["inner_neuropil_radius"] = int(cfg["inner_neuropil_radius"])
    ops["min_neuropil_pixels"] = int(cfg["min_neuropil_pixels"])
    ops["spike_deconvolution"] = True
    ops["baseline"] = str(cfg["baseline_mode"])
    ops["win_baseline"] = int(cfg["win_baseline_sec"])
    ops["sig_baseline"] = int(cfg["sig_baseline"])
    ops["keep_good_only"] = False
    ops["combined"] = False
    ops["save_mat"] = False

    _suite2p_run(ops=ops)

    max_tif = output_root / "MAX.tif"
    std_tif = output_root / "STD.tif"
    intensity_filter_suite2p_plane0_no_overwrite(
        plane0_dir=plane0_dir,
        max_tif=max_tif,
        std_tif=std_tif,
        out_plane0_dir=plane0_intensity_dir,
        thr_max=float(cfg["int_thr_max"]),
        thr_std=float(cfg["int_thr_std"]),
        only_iscell=bool(cfg["int_only_iscell"]),
        norm_p_low=float(cfg["norm_p_low"]),
        norm_p_high=float(cfg["norm_p_high"]),
    )

    if bool(cfg["delete_temp_movie"]):
        try:
            os.remove(img_cat_name)
        except Exception:
            pass

    stat = np.load(plane0_dir / "stat.npy", allow_pickle=True) if (plane0_dir / "stat.npy").exists() else []
    iscell = np.load(plane0_intensity_dir / "iscell.npy", allow_pickle=True)
    effective_ops_path = _save_effective_ops_snapshot(plane0_dir, output_root / "suite2p_effective_ops.json")

    _write_json(
        sidecar_path,
        {
            "config_fingerprint": params_fp,
            "params": params_payload,
        },
    )

    return {
        "execution_status": "executed",
        "config_fingerprint": params_fp,
        "sidecar_path": str(sidecar_path),
        "input_tif_path": str(input_tif_path),
        "output_root": str(output_root),
        "suite2p_registration_used": bool(cfg["do_registration"]),
        "artifacts": {
            "plane0_dir": str(plane0_dir),
            "plane0_intensityfilt_dir": str(plane0_intensity_dir),
            "max_tif": str(max_tif),
            "std_tif": str(std_tif),
            "effective_ops_json": effective_ops_path,
        },
        "counts": {
            "plane0_total": int(len(stat)),
            "after_intensity_filter": int(np.sum(iscell[:, 0] > 0)),
        },
        "config_used": cfg,
    }


# ============================================================
# ----------------------------- Main -------------------------
# ============================================================

def main():
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Missing file: {tif_path}")

    p = Path(tif_path).resolve()
    folder_name = p.parent.name
    save_root_path = folder_name + "_seg"
    os.makedirs(save_root_path, exist_ok=True)

    print(f"[INFO] Input: {tif_path}")
    print(f"[INFO] Output root: {save_root_path}")

    # --- Load stack (RAM) ---
    # stack = load_tif_anyshape(tif_path)  # (T,H,W)
    stack = load_tif_pages_to_thw(tif_path)
    print(f"[INFO] Loaded stack shape (T,H,W) = {stack.shape}, dtype={stack.dtype}")
    if 0:
        stack = stack[:, ::2, ::2]
        print(f"[INFO] Downsampled stack shape (T,H,W) = {stack.shape}")
    # --- Save a working movie for suite2p ---
    img_cat_name = str(Path(save_root_path) / "img_cat.tif")
    tiff.imwrite(img_cat_name, stack, bigtiff=True)
    print(f"[OK] Saved working movie: {img_cat_name}")

    # --- Save quick QC projections ---
    write_proj_max_std(stack, Path(save_root_path))
    print(f"[OK] Saved MAX/STD into: {save_root_path}")

    # Free RAM if desired (suite2p will read from disk)
    del stack

    # -------------------------
    # Suite2p ops
    # -------------------------
    save_root = str(Path("./") / save_root_path) + "/"

    ops = _suite2p_default_ops()

    # I/O
    ops["data_path"] = [str(Path(img_cat_name).parent)]
    ops["tiff_list"] = [Path(img_cat_name).name]
    ops["save_path0"] = save_root

    # Acquisition
    ops["fs"] = FS_HZ
    ops["nplanes"] = NPLANES
    ops["nchannels"] = NCHANNELS
    ops["bin_size"] = 1
    ops["nbinned"] = 20000 
    ops["max_iterations"] = 40
    ops["highpass_time"] = 0
    ops['use_builtin_classifier'] = False
    # Registration
    ops["do_registration"] = DO_REGISTRATION
    ops["nonrigid"] = NONRIGID
    ops["block_size"] = [128, 128]
    ops["maxregshift"] = 0.1

    # -------- NeuroPilot downstream detection --------
    ops["diameter"] = int(DIAMETER_PX)
    ops["threshold_scaling"] = float(THRESH_SCALING)
    ops["aspect"] = float(ASPECT_MAX)
    ops["max_overlap"] = float(MAX_OVERLAP)
    ops["min_area"] = float(MIN_AREA_PX)

    # Extraction
    ops["allow_overlap"] = True

    # Neuropil (small ROI friendly)
    ops["neuropil_extract"] = bool(NEUROPIL_EXTRACT)
    ops["inner_neuropil_radius"] = int(INNER_NEUROPIL_RADIUS)
    ops["min_neuropil_pixels"] = int(MIN_NEUROPIL_PIXELS)

    # Deconvolution (optional)
    ops["spike_deconvolution"] = True

    # Long recording robustness
    ops["baseline"] = BASELINE_MODE
    ops["win_baseline"] = WIN_BASELINE_SEC
    ops["sig_baseline"] = SIG_BASELINE

    # Keep everything (we'll filter later)
    ops["keep_good_only"] = False
    ops["combined"] = False
    ops["save_mat"] = False

    print("\n[INFO] Running suite2p with NeuroPilot downstream ops:")
    for k in [
        "do_registration", "nonrigid", "diameter", "threshold_scaling", "aspect",
        "max_overlap", "min_area", "allow_overlap", "inner_neuropil_radius",
        "min_neuropil_pixels"
    ]:
        print(f"  ops[{k!r}] = {ops.get(k)}")
    print()

    # Run suite2p
    _suite2p_run(ops=ops)

    # -------------------------
    # Intensity-based post-filter (NEW FOLDER, non-overwrite)
    # -------------------------
    save_root_p = Path(save_root_path).resolve()  # e.g. xxx_seg/
    plane0_dir = (Path(save_root) / "suite2p" / "plane0").resolve()

    max_tif = save_root_p / "MAX.tif"
    std_tif = save_root_p / "STD.tif"

    out_plane0_dir = plane0_dir.parent / "plane0_intensityfilt"  # .../suite2p/plane0_intensityfilt

    intensity_filter_suite2p_plane0_no_overwrite(
        plane0_dir=plane0_dir,
        max_tif=max_tif,
        std_tif=std_tif,
        out_plane0_dir=out_plane0_dir,
        thr_max=INT_THR_MAX,
        thr_std=INT_THR_STD,
        only_iscell=INT_ONLY_ISCELL,
        norm_p_low=NORM_P_LOW,
        norm_p_high=NORM_P_HIGH,
    )

    # -------------------------
    # Optional: delete temporary movie
    # -------------------------
    if DELETE_TEMP_MOVIE:
        try:
            os.remove(img_cat_name)
            print(f"[OK] Deleted temp movie: {img_cat_name}")
        except Exception as e:
            print(f"[WARN] Failed to delete temp movie: {img_cat_name} ({e})")


if __name__ == "__main__":
    main()
