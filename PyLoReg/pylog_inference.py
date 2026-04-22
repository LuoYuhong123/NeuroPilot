#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
import numpy as np
import tifffile as tiff
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tifffile import TiffWriter

from PyLoReg.PyLoRegNet.PyLoRegNet import PyLoRegNet


# =========================================================
# 0) Robust checkpoint loader
# =========================================================

def load_weights_safely(model, ckpt_path, device='cuda', strict=False, verbose=True):
    """
    Robust checkpoint loader for single / DP / DDP / torch.compile-wrapped models.

    Returns:
      (model, ckpt, info_dict)
    """
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    map_dev = device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=map_dev)

    # -------- extract state_dict --------
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
            state = ckpt['state_dict']
        elif 'model' in ckpt:
            m = ckpt['model']
            state = m.state_dict() if hasattr(m, 'state_dict') else (m if isinstance(m, dict) else None)
            if state is None:
                state = ckpt
        else:
            state = ckpt
    else:
        state = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: extracted state is {type(state)}")

    def strip_prefix(d, prefix: str):
        if not prefix:
            return d
        out = {}
        for k, v in d.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    def add_prefix(d, prefix: str):
        if not prefix:
            return d
        return {(prefix + k): v for k, v in d.items()}

    def normalize_known_prefixes(d):
        out = d
        changed = True
        while changed:
            changed = False
            for p in ("module.", "_orig_mod."):
                keys = list(out.keys())
                if keys and sum(k.startswith(p) for k in keys) > 0.6 * len(keys):
                    out = strip_prefix(out, p)
                    changed = True
        return out

    def summarize_diff(missing, unexpected, head=20):
        if not verbose:
            return
        if missing:
            print(f"[load] Missing keys ({len(missing)}):")
            for k in list(missing)[:head]:
                print("   ", k)
            if len(missing) > head:
                print("   ...")
        if unexpected:
            print(f"[load] Unexpected keys ({len(unexpected)}):")
            for k in list(unexpected)[:head]:
                print("   ", k)
            if len(unexpected) > head:
                print("   ...")

    # target model (unwrap DP/DDP)
    target = model.module if hasattr(model, "module") else model

    model_keys = list(target.state_dict().keys())
    model_expects_orig_mod = (model_keys and sum(k.startswith("_orig_mod.") for k in model_keys) > 0.6 * len(model_keys))

    base_state = normalize_known_prefixes(state)

    missing, unexpected = target.load_state_dict(base_state, strict=strict)

    def is_good(missing, unexpected):
        return (len(missing) == 0 and len(unexpected) == 0) or (not strict and len(missing) < 10 and len(unexpected) < 10)

    if is_good(missing, unexpected):
        if verbose:
            print("[load] Done. strict =", strict)
        info = {"strategy": "direct(normalized)", "missing": missing, "unexpected": unexpected}
        return model, ckpt, info

    candidates = []
    if model_expects_orig_mod:
        candA = add_prefix(base_state, "_orig_mod.")
        candidates.append(("add_orig_mod_prefix_to_ckpt", candA))

    best = None
    best_score = None
    best_missing = None
    best_unexpected = None

    for name, cand in candidates:
        missing2, unexpected2 = target.load_state_dict(cand, strict=strict)
        score = len(missing2) + len(unexpected2)
        if best_score is None or score < best_score:
            best = (name, cand)
            best_score = score
            best_missing = missing2
            best_unexpected = unexpected2

    if best is not None and best_score is not None and best_score < (len(missing) + len(unexpected)):
        name, cand = best
        missing3, unexpected3 = target.load_state_dict(cand, strict=strict)
        if verbose:
            print(f"[load] Applied strategy: {name}")
            summarize_diff(missing3, unexpected3)
            print("[load] Done. strict =", strict)
        info = {"strategy": name, "missing": missing3, "unexpected": unexpected3}
        return model, ckpt, info

    if verbose:
        summarize_diff(missing, unexpected)
        print("[load] Done (with mismatches). strict =", strict)

    info = {"strategy": "direct(normalized)_no_fix", "missing": missing, "unexpected": unexpected}
    return model, ckpt, info


# =========================================================
# 1) Template saver (per-iter single-frame writer)
# =========================================================

@dataclass
class TemplateSaver:
    stack_save_path: str
    template_save_dir: str
    template_save_dtype: str = "uint16"  # "uint16" | "float32"

    def open(self, it: int):
        out_dir = Path(self.template_save_dir) if self.template_save_dir else self._default_dir()
        out_dir.mkdir(exist_ok=True, parents=True)

        stem = Path(self.stack_save_path).stem
        out_path = out_dir / f"{stem}_template_iter{it+1:02d}.tif"

        # Windows path-length fallback: shorten file/dir to avoid FileNotFoundError.
        if os.name == "nt":
            max_len = 240
            if len(str(out_path)) > max_len:
                stem_hash = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
                short_stem = stem[:24]
                short_name = f"{short_stem}_{stem_hash}_tpl_i{it+1:02d}.tif"
                out_path = out_dir / short_name

            if len(str(out_path)) > max_len:
                fallback_dir = Path(self.stack_save_path).parent / "_tpl"
                fallback_dir.mkdir(exist_ok=True, parents=True)
                stem_hash = hashlib.md5(stem.encode("utf-8")).hexdigest()[:8]
                out_path = fallback_dir / f"{stem_hash}_tpl_i{it+1:02d}.tif"

        writer = TiffWriter(str(out_path), bigtiff=True)
        return out_path, writer

    def _default_dir(self):
        stack_path = Path(self.stack_save_path)
        folder = stack_path.parent
        return folder.with_name(folder.name + "_template")

    def cast(self, im_hw: np.ndarray) -> np.ndarray:
        if self.template_save_dtype == "float32":
            return im_hw.astype(np.float32, copy=False)
        if self.template_save_dtype == "uint16":
            return np.clip(im_hw * 65535.0, 0, 65535).astype(np.uint16, copy=False)
        raise ValueError(f"template_save_dtype must be 'uint16' or 'float32', got {self.template_save_dtype}")


# =========================================================
# 2) Flow quant (uint16) helpers
# =========================================================

@dataclass
class FlowQuantizer:
    flow_scale: float
    flow_offset: float

    def encode_u16(self, flow_hw2: np.ndarray) -> np.ndarray:
        q = np.rint(flow_hw2 * float(self.flow_scale) + float(self.flow_offset))
        q = np.clip(q, 0, 65535).astype(np.uint16)
        return q

    def decode_u16(self, q_hw2: np.ndarray) -> np.ndarray:
        return (q_hw2.astype(np.float32) - float(self.flow_offset)) / float(self.flow_scale)


# =========================================================
# 3) Grid cache + flow->grid
# =========================================================

class GridCache:
    def __init__(self, cache_base_grid: bool = True):
        self.cache_base_grid = bool(cache_base_grid)
        self._cache = {}

    def base_xy_1hw2(self, H: int, W: int, dev: torch.device, dtype: torch.dtype):
        if not self.cache_base_grid:
            ys = torch.arange(H, device=dev, dtype=dtype)
            xs = torch.arange(W, device=dev, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            return torch.stack([xx, yy], dim=-1).unsqueeze(0)

        key = ("base_xy", H, W, dev.type, dev.index, str(dtype))
        base = self._cache.get(key, None)
        if base is None or base.device != dev or base.dtype != dtype:
            ys = torch.arange(H, device=dev, dtype=dtype)
            xs = torch.arange(W, device=dev, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            base = torch.stack([xx, yy], dim=-1).unsqueeze(0)
            self._cache[key] = base
        return base

    def flow_to_grid(self, flow_bhw2: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = flow_bhw2.shape
        base = self.base_xy_1hw2(H, W, flow_bhw2.device, flow_bhw2.dtype)
        coords = base + flow_bhw2

        grid = torch.empty_like(coords)
        if W > 1:
            grid[..., 0] = coords[..., 0] * (2.0 / (W - 1)) - 1.0
        else:
            grid[..., 0] = 0
        if H > 1:
            grid[..., 1] = coords[..., 1] * (2.0 / (H - 1)) - 1.0
        else:
            grid[..., 1] = 0
        return grid


# =========================================================
# 4) Warp & compose helpers
# =========================================================

def warp_two_batches_once_with_mask(
    raw_b1hw: torch.Tensor,
    img_b1hw: torch.Tensor,
    flow_bhw2: torch.Tensor,
    grid_cache: GridCache,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
):
    grid = grid_cache.flow_to_grid(flow_bhw2).to(dtype=raw_b1hw.dtype)

    x_b2hw = torch.cat([raw_b1hw, img_b1hw], dim=1)
    warped_b2hw = F.grid_sample(x_b2hw, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    warped_raw = warped_b2hw[:, 0:1]
    warped_img = warped_b2hw[:, 1:2]

    mask = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)).to(raw_b1hw.dtype)
    mask = mask.unsqueeze(1)
    return warped_raw, warped_img, mask


def compose_flow_batch(prev_bhw2: torch.Tensor, new_bhw2: torch.Tensor, grid_cache: GridCache) -> torch.Tensor:
    """
    prev/new: [B,H,W,2] pixel flow
    return composed: prev + sample(new at prev coords)
    """
    target_dtype = new_bhw2.dtype
    prev_bhw2 = prev_bhw2.to(target_dtype)

    new_b2hw = new_bhw2.permute(0, 3, 1, 2).contiguous()
    grid = grid_cache.flow_to_grid(prev_bhw2).to(dtype=target_dtype)

    sampled_new = F.grid_sample(new_b2hw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled_new_bhw2 = sampled_new.permute(0, 2, 3, 1).contiguous()
    return prev_bhw2 + sampled_new_bhw2


# =========================================================
# 5) Padding utils
# =========================================================

def pad_to_multiple(tensor, multiple=16, mode="reflect"):
    B, C, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    pad_top, pad_bottom = 0, pad_h
    pad_left, pad_right = 0, pad_w
    pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, pad_tuple, mode=mode)
    return tensor, pad_tuple


def unpad(tensor, pad_tuple):
    pad_left, pad_right, pad_top, pad_bottom = pad_tuple
    if tensor.ndim == 4:
        _, _, H, W = tensor.shape
        h_start = pad_top
        h_end = H - pad_bottom if pad_bottom > 0 else H
        w_start = pad_left
        w_end = W - pad_right if pad_right > 0 else W
        return tensor[:, :, h_start:h_end, w_start:w_end]
    if tensor.ndim == 3:
        H, W, C = tensor.shape
        h_start = pad_top
        h_end = H - pad_bottom if pad_bottom > 0 else H
        w_start = pad_left
        w_end = W - pad_right if pad_right > 0 else W
        return tensor[h_start:h_end, w_start:w_end, :]
    raise ValueError(f"unpad only supports 3D(HWC) or 4D(BCHW) tensors, got ndim={tensor.ndim}")


# =========================================================
# 6) Fast corr score (for best-frames refine)
# =========================================================

def _center_crop_stride(x_b1hw: torch.Tensor, crop=256, stride=2):
    B, C, H, W = x_b1hw.shape
    c = int(min(crop, H, W))
    y0 = (H - c) // 2
    x0 = (W - c) // 2
    x = x_b1hw[:, :, y0:y0+c, x0:x0+c]
    if stride > 1:
        x = x[:, :, ::stride, ::stride]
    return x


def batch_ncc_corr(a_b1hw: torch.Tensor, b_b1hw: torch.Tensor, eps=1e-6):
    # normalized cross-correlation per batch element
    B = a_b1hw.shape[0]
    a = a_b1hw.reshape(B, -1)
    b = b_b1hw.reshape(B, -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).mean(dim=1)
    den = a.std(dim=1) * b.std(dim=1) + eps
    return num / den


# =========================================================
# 7) Main function
# =========================================================

def demotion_PyLoReg_infer2stack_less_save_acc_v3_1(
    img_stack_path,
    raw_stack_path,
    stack_save_path,
    model_root="PyLoReg//PyLoReg_model",
    model_name="GM3_fn5_202511301551",
    feature_channels=128,
    use_feature_num=4,
    iteration_num=2,
    batch_size=2,
    substack_len=1000,          # (kept for backward compat; not used in new template strategy)
    template_mode="mean",       # (kept; new strategy uses global mean template)
    gaussian_sigma=0.0,         # (kept)
    max_frames=1000,
    crop_size=None,
    save_mask_flow=True,
    # ===== flow uint16 quant params =====
    flow_scale=100.0,
    flow_offset=65535.0 / 2.0,
    # ===== extra speed knobs =====
    use_amp=True,
    cache_base_grid=True,
    cache_prefix_mean=True,     # (kept)
    # ===== warp mode =====
    warp_mode="nearest",
    # ===== old knobs (kept, but template selection no longer uses motion_thr) =====
    drop_large_motion_in_template=True,
    motion_thr_px=20.0,
    motion_metric="l2mean",
    # ===== save templates =====
    save_templates=False,
    template_save_dir=None,
    template_save_dtype="uint16",
):
    import warnings
    warnings.filterwarnings("ignore")
    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)

    # ---------- helpers ----------
    grid_cache = GridCache(cache_base_grid=cache_base_grid)
    fq = FlowQuantizer(flow_scale=flow_scale, flow_offset=flow_offset)
    tpl_saver = TemplateSaver(stack_save_path, template_save_dir, template_save_dtype)

    # ---------- build/load model ----------
    base_model = PyLoRegNet(feature_channels=feature_channels).to(device)
    model = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    model = torch.compile(model, mode="reduce-overhead")

    model_path = os.path.join(model_root, model_name, "gmflow_latest.pt")
    print("[Model] Loading:", model_path)
    model, _, info = load_weights_safely(model, model_path, device="cuda", strict=False, verbose=True)
    if isinstance(info, dict):
        print("[ModelLoad] strategy:", info.get("strategy", "unknown"))

    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    # ---------- read stacks ----------
    img_u16 = tiff.imread(img_stack_path)
    raw_u16 = tiff.imread(raw_stack_path)
    print("[Img] :", img_u16.shape)
    print("[Raw] :", raw_u16.shape)

    T = min(len(img_u16), len(raw_u16))
    if max_frames is not None:
        T = min(T, max_frames)
    img_u16 = img_u16[:T]
    raw_u16 = raw_u16[:T]
    print(f"[Use] first {T} frames")

    img_f32 = img_u16.astype(np.float32, copy=False)
    raw_f32 = raw_u16.astype(np.float32, copy=False)
    vmin_img, vmax_img = float(img_f32.min()), float(img_f32.max())
    vmin_raw, vmax_raw = float(raw_f32.min()), float(raw_f32.max())

    img_stack = (img_f32 - vmin_img) / max(vmax_img - vmin_img, 1e-12)
    raw_stack = (raw_f32 - vmin_raw) / max(vmax_raw - vmin_raw, 1e-12)

    # crop
    T0, H0, W0 = img_stack.shape
    if crop_size is not None and (H0 > crop_size or W0 > crop_size):
        h0, w0 = (H0 - crop_size) // 2, (W0 - crop_size) // 2
        img_stack = img_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        raw_stack = raw_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        print("[Crop] →", img_stack.shape)

    T, H, W = img_stack.shape

    # ---------- containers ----------
    if save_mask_flow:
        flow_stack = np.zeros((T, H, W, 2), dtype=np.uint16)
        mask_accum_stack = np.full((T, H, W), 255, dtype=np.uint8)

        stack_path = Path(stack_save_path)
        folder = stack_path.parent
        stem = stack_path.stem
        mask_folder = folder.with_name(folder.name + "_mask")
        flow_folder = folder.with_name(folder.name + "_flow")
        mask_folder.mkdir(exist_ok=True)
        flow_folder.mkdir(exist_ok=True)
    else:
        flow_stack = None
        mask_accum_stack = None

    # =========================================================
    # NEW: Global reference frame pool + refine schedule
    # =========================================================
    ref_pool_size = min(T, 2000)      # time-cost knob: 1000~3000 typical
    pool_mode = "uniform"             # "uniform" or "random"
    pool_seed = 0

    # refine ratios: Iter1 uses ALL pool; then 10%,20%,30%...
    # You can cap it to avoid template being dragged by drift
    refine_step = 0.10
    refine_cap  = 0.40               # max 40% of pool used for template

    # corr computation knobs (time-cost)
    corr_crop = 256
    corr_stride = 2

    def make_pool_indices(T_, n, mode="uniform", seed=0):
        if n >= T_:
            return np.arange(T_, dtype=np.int64)
        if mode == "random":
            rng = np.random.default_rng(seed)
            idx = rng.choice(T_, size=n, replace=False)
            idx.sort()
            return idx.astype(np.int64)
        idx = np.linspace(0, T_ - 1, num=n, dtype=np.int64)
        idx = np.unique(idx)
        return idx.astype(np.int64)

    pool_idx = make_pool_indices(T, ref_pool_size, mode=pool_mode, seed=pool_seed)
    print(f"[RefPool] mode={pool_mode} size={len(pool_idx)} / T={T}")

    corr_score_prev = None
    motion_score_prev = None  # keep for debug/printing if you want

    # ---------- main loop ----------
    with torch.inference_mode():
        for it in range(int(iteration_num)):
            print(f"\n[Iter {it+1}/{iteration_num}] ...")

            # =========================================================
            # Build ONE global template per iteration (suite2p-like refine)
            # =========================================================
            if corr_score_prev is None:
                sel_idx = pool_idx
                print(f"[TemplateSelect] Iter{it+1}: init from ALL pool frames ({len(sel_idx)})")
            else:
                keep_ratio = float(np.clip(refine_step * it, 0.0, refine_cap))  # it=1 -> 10%
                P = len(pool_idx)
                k = max(int(np.ceil(P * keep_ratio)), 10)
                k = min(k, P)

                q = corr_score_prev[pool_idx].astype(np.float32, copy=False)  # higher is better
                thr = np.partition(q, P - k)[P - k]
                sel_idx = pool_idx[q >= thr]
                print(f"[TemplateSelect] Iter{it+1}: top {keep_ratio*100:.1f}% in pool => {len(sel_idx)} frames")

            # template mean on CPU
            global_template = img_stack[sel_idx].mean(axis=0).astype(np.float32, copy=False)  # [H,W]
            global_template_t = torch.from_numpy(global_template).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]

            if bool(save_templates):
                tpl_path, tpl_writer = tpl_saver.open(it)
                tpl_writer.write(tpl_saver.cast(global_template), photometric="minisblack")
                tpl_writer.close()
                print(f"[Template] Saved global template → {tpl_path}")

            # scores for this iteration
            corr_score_this = np.zeros((T,), dtype=np.float32)
            motion_score_this = np.zeros((T,), dtype=np.float32)  # optional debug

            for start in tqdm(
                range(0, T, int(batch_size)),
                desc="🌀 Demotion [PyLoReg]",
                ncols=100, unit="batch",
                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
            ):
                end = min(start + int(batch_size), T)
                idx = slice(start, end)
                B = end - start

                img_batch = torch.from_numpy(img_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
                raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)

                # ONE GT for all frames in this batch
                GT = global_template_t.expand(B, 1, H, W).contiguous().to(device, non_blocking=True)

                # corr score (higher is better): use cheap center crop + stride
                img_q = _center_crop_stride(img_batch, crop=corr_crop, stride=corr_stride)
                gt_q  = _center_crop_stride(GT,        crop=corr_crop, stride=corr_stride)
                corr_b = batch_ncc_corr(img_q, gt_q).float()
                corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)

                img_p, pad = pad_to_multiple(img_batch, 16)
                GT_p, _ = pad_to_multiple(GT, 16)

                if amp_enabled:
                    with torch.cuda.amp.autocast(True):
                        out = model(GT_p, img_p, use_feature_num)
                else:
                    out = model(GT_p, img_p, use_feature_num)

                flow_b2hw = unpad(out["flow_preds"][-1], pad)  # [B,2,H,W]
                flow_bhw2 = flow_b2hw.permute(0, 2, 3, 1).contiguous()

                # motion score (optional debug)
                if str(motion_metric) == "absmean":
                    score_b = flow_b2hw.abs().mean(dim=(1, 2, 3))
                else:
                    score_b = torch.sqrt(flow_b2hw[:, 0].pow(2) + flow_b2hw[:, 1].pow(2)).mean(dim=(1, 2))
                motion_score_this[start:end] = score_b.detach().float().cpu().numpy()

                # warp (one grid_sample)
                warped_raw_b1hw, warped_img_b1hw, mask_b1hw = warp_two_batches_once_with_mask(
                    raw_batch, img_batch, flow_bhw2,
                    grid_cache=grid_cache,
                    mode=str(warp_mode),
                    padding_mode="reflection", # "zeros",
                    align_corners=True
                )

                raw_stack[idx] = warped_raw_b1hw.squeeze(1).cpu().numpy()
                img_stack[idx] = warped_img_b1hw.squeeze(1).cpu().numpy()

                if save_mask_flow:
                    prev_mask_u8 = mask_accum_stack[idx]
                    prev_mask = (
                        torch.from_numpy(prev_mask_u8).unsqueeze(1).to(device, non_blocking=True)
                        .float().div_(255.0)
                    )
                    grid = grid_cache.flow_to_grid(flow_bhw2).to(dtype=prev_mask.dtype)
                    warped_prev = F.grid_sample(prev_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                    new_mask = (warped_prev * mask_b1hw).clamp_(0.0, 1.0)
                    mask_accum_stack[idx] = new_mask.mul(255.0).round().to(torch.uint8).squeeze(1).cpu().numpy()

                    if it == 0:
                        flow_np = flow_bhw2.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(flow_np)
                    else:
                        prev_np = fq.decode_u16(flow_stack[idx])
                        prev_t = torch.from_numpy(prev_np).to(device, non_blocking=True)
                        composed = compose_flow_batch(prev_t, flow_bhw2, grid_cache=grid_cache)
                        composed_np = composed.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(composed_np)

                del img_batch, raw_batch, GT, img_p, GT_p, out, flow_b2hw, flow_bhw2
                del warped_raw_b1hw, warped_img_b1hw, mask_b1hw, img_q, gt_q, corr_b

            # update for next iter
            corr_score_prev = corr_score_this
            motion_score_prev = motion_score_this

            # stats
            try:
                med = float(np.percentile(corr_score_prev[pool_idx], 50))
                p90 = float(np.percentile(corr_score_prev[pool_idx], 90))
                p99 = float(np.percentile(corr_score_prev[pool_idx], 99))
                print(f"[Iter {it+1}] corr(pool) median={med:.4f}, p90={p90:.4f}, p99={p99:.4f}")
            except Exception:
                pass

            try:
                p95m = float(np.percentile(motion_score_prev, 95))
                p99m = float(np.percentile(motion_score_prev, 99))
                print(f"[Iter {it+1}] motion_score p95={p95m:.3f}, p99={p99m:.3f}")
            except Exception:
                pass

    # ---------- save mask/flow ----------
    if save_mask_flow:
        tiff.imwrite(mask_folder / f"{stem}_mask_final.tif", mask_accum_stack)
        tiff.imwrite(flow_folder / f"{stem}_flow_u_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif", flow_stack[..., 0])
        tiff.imwrite(flow_folder / f"{stem}_flow_v_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif", flow_stack[..., 1])
        print("[Saved] Mask & Flow")

    # ---------- save final stack ----------
    final = (raw_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
    tiff.imwrite(stack_save_path, final)
    print("\n[Done] Demotion saved →", stack_save_path)
    return stack_save_path




def demotion_PyLoReg_infer2stack_less_save_acc_v3_2(
    img_stack_path,
    raw_stack_path,
    stack_save_path,
    model_root="PyLoReg//PyLoReg_model",
    model_name="GM3_fn5_202511301551",
    feature_channels=128,
    use_feature_num=4,
    iteration_num=2,
    batch_size=2,
    substack_len=1000,
    template_mode="mean",
    gaussian_sigma=0.0,
    max_frames=1000,
    crop_size=None,
    save_mask_flow=True,
    # ===== flow uint16 quant params =====
    flow_scale=100.0,
    flow_offset=65535.0 / 2.0,
    # ===== extra speed knobs =====
    use_amp=True,
    cache_base_grid=True,
    cache_prefix_mean=True,
    # ===== warp mode =====
    warp_mode="nearest",
    # ===== old knobs =====
    drop_large_motion_in_template=True,
    motion_thr_px=20.0,
    motion_metric="l2mean",
    # ===== save templates =====
    save_templates=False,
    template_save_dir=None,
    template_save_dtype="uint16",

    # =========================================================
    # RIGID-only Iter1
    # =========================================================
    rigid_max_shift_ratio=0.5,  # 画面大小（min(H,W)）的 50%
    rigid_crop=None,
    rigid_stride=1,

    # =========================================================
    # NEW: save intermediate (after Iter1 rigid correction)
    # =========================================================
    save_iter1_stack=False,
    iter1_save_path=None,       # None => auto: <final>_iter1_rigid.tif
    save_iter1_img_stack=False, # also save img_stack after iter1 for QC
):
    import os
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import tifffile as tiff
    from pathlib import Path
    from tqdm import tqdm

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)

    # ---------- helpers (assumed existing in your project) ----------
    grid_cache = GridCache(cache_base_grid=cache_base_grid)
    fq = FlowQuantizer(flow_scale=flow_scale, flow_offset=flow_offset)
    tpl_saver = TemplateSaver(stack_save_path, template_save_dir, template_save_dtype)

    # =========================================================
    # RIGID (translation-only) helper functions
    # =========================================================
    def _center_crop_stride_2d(x_b1hw: torch.Tensor, crop: int, stride: int = 1) -> torch.Tensor:
        if crop is None:
            y = x_b1hw
        else:
            B, C, H_, W_ = x_b1hw.shape
            c = int(crop)
            c = min(c, H_, W_)
            h0 = (H_ - c) // 2
            w0 = (W_ - c) // 2
            y = x_b1hw[:, :, h0:h0 + c, w0:w0 + c]
        s = max(int(stride), 1)
        if s > 1:
            y = y[:, :, ::s, ::s]
        return y

    def _phase_corr_shift_batched(x_b1hw: torch.Tensor, ref_b1hw: torch.Tensor, eps: float = 1e-9):
        x = x_b1hw - x_b1hw.mean(dim=(-2, -1), keepdim=True)
        r = ref_b1hw - ref_b1hw.mean(dim=(-2, -1), keepdim=True)

        X = torch.fft.rfft2(x, dim=(-2, -1))
        R = torch.fft.rfft2(r, dim=(-2, -1))

        CPS = X * torch.conj(R)
        CPS = CPS / (torch.abs(CPS) + eps)

        cc = torch.fft.irfft2(CPS, s=x.shape[-2:], dim=(-2, -1))  # [B,1,H,W]
        cc = cc.squeeze(1)

        B, H_, W_ = cc.shape
        flat = cc.reshape(B, -1)
        idx = flat.argmax(dim=1)
        peak_y = (idx // W_).to(torch.int64)
        peak_x = (idx % W_).to(torch.int64)

        dy = peak_y.to(torch.float32)
        dx = peak_x.to(torch.float32)

        dy = torch.where(dy > (H_ // 2), dy - H_, dy)
        dx = torch.where(dx > (W_ // 2), dx - W_, dx)
        return dy, dx

    def _warp_translation_batched(
        x_b1hw: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        align_corners: bool = True,
    ):
        B, C, H_, W_ = x_b1hw.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H_, device=x_b1hw.device),
            torch.linspace(-1, 1, W_, device=x_b1hw.device),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

        sx = (2.0 * dx) / max(W_ - 1, 1)
        sy = (2.0 * dy) / max(H_ - 1, 1)

        base[..., 0] = base[..., 0] - sx.view(B, 1, 1)
        base[..., 1] = base[..., 1] - sy.view(B, 1, 1)

        y = F.grid_sample(x_b1hw, base, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return y

    def _shift_to_flow_field(dy: torch.Tensor, dx: torch.Tensor, H_: int, W_: int, device_) -> torch.Tensor:
        B = dy.numel()
        flow = torch.zeros((B, H_, W_, 2), device=device_, dtype=torch.float32)
        flow[..., 0] = dx.view(B, 1, 1)
        flow[..., 1] = dy.view(B, 1, 1)
        return flow

    # ---------- build/load model (ONLY used for Iter2+) ----------
    base_model = PyLoRegNet(feature_channels=feature_channels).to(device)
    model = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
    model = torch.compile(model, mode="reduce-overhead")

    model_path = os.path.join(model_root, model_name, "gmflow_latest.pt")
    print("[Model] Loading:", model_path)
    model, _, info = load_weights_safely(model, model_path, device="cuda", strict=False, verbose=True)
    if isinstance(info, dict):
        print("[ModelLoad] strategy:", info.get("strategy", "unknown"))

    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    # ---------- read stacks ----------
    img_u16 = tiff.imread(img_stack_path)
    raw_u16 = tiff.imread(raw_stack_path)
    print("[Img] :", img_u16.shape)
    print("[Raw] :", raw_u16.shape)

    T = min(len(img_u16), len(raw_u16))
    if max_frames is not None:
        T = min(T, int(max_frames))
    img_u16 = img_u16[:T]
    raw_u16 = raw_u16[:T]
    print(f"[Use] first {T} frames")

    img_f32 = img_u16.astype(np.float32, copy=False)
    raw_f32 = raw_u16.astype(np.float32, copy=False)
    vmin_img, vmax_img = float(img_f32.min()), float(img_f32.max())
    vmin_raw, vmax_raw = float(raw_f32.min()), float(raw_f32.max())

    img_stack = (img_f32 - vmin_img) / max(vmax_img - vmin_img, 1e-12)
    raw_stack = (raw_f32 - vmin_raw) / max(vmax_raw - vmin_raw, 1e-12)

    # crop
    T0, H0, W0 = img_stack.shape
    if crop_size is not None and (H0 > crop_size or W0 > crop_size):
        h0, w0 = (H0 - crop_size) // 2, (W0 - crop_size) // 2
        img_stack = img_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        raw_stack = raw_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        print("[Crop] →", img_stack.shape)

    T, H, W = img_stack.shape

    # rigid reject threshold = 50% FOV (min side)
    rigid_max_shift_px = float(rigid_max_shift_ratio) * float(min(H, W))
    print(f"[Rigid] max allowed shift = {rigid_max_shift_px:.1f} px "
          f"({rigid_max_shift_ratio*100:.0f}% of min(H,W)={min(H,W)})")

    # ---------- containers ----------
    if save_mask_flow:
        flow_stack = np.zeros((T, H, W, 2), dtype=np.uint16)
        mask_accum_stack = np.full((T, H, W), 255, dtype=np.uint8)

        stack_path = Path(stack_save_path)
        folder = stack_path.parent
        stem = stack_path.stem
        mask_folder = folder.with_name(folder.name + "_mask")
        flow_folder = folder.with_name(folder.name + "_flow")
        mask_folder.mkdir(exist_ok=True)
        flow_folder.mkdir(exist_ok=True)
    else:
        flow_stack = None
        mask_accum_stack = None

    # =========================================================
    # Global reference pool for template
    # =========================================================
    ref_pool_size = min(T, 2000)
    pool_mode = "uniform"
    pool_seed = 0

    refine_step = 0.10
    refine_cap = 0.40

    corr_crop = 256
    corr_stride = 2

    def make_pool_indices(T_, n, mode="uniform", seed=0):
        if n >= T_:
            return np.arange(T_, dtype=np.int64)
        if mode == "random":
            rng = np.random.default_rng(seed)
            idx = rng.choice(T_, size=n, replace=False)
            idx.sort()
            return idx.astype(np.int64)
        idx = np.linspace(0, T_ - 1, num=n, dtype=np.int64)
        idx = np.unique(idx)
        return idx.astype(np.int64)

    pool_idx = make_pool_indices(T, ref_pool_size, mode=pool_mode, seed=pool_seed)
    print(f"[RefPool] mode={pool_mode} size={len(pool_idx)} / T={T}")

    corr_score_prev = None
    motion_score_prev = None

    # ---------- main loop ----------
    with torch.inference_mode():
        for it in range(int(iteration_num)):
            print(f"\n[Iter {it+1}/{iteration_num}] ...")

            # template select
            if corr_score_prev is None:
                sel_idx = pool_idx
                print(f"[TemplateSelect] Iter{it+1}: init from ALL pool frames ({len(sel_idx)})")
            else:
                keep_ratio = float(np.clip(refine_step * it, 0.0, refine_cap))  # it=1 -> 10%
                P = len(pool_idx)
                k = max(int(np.ceil(P * keep_ratio)), 10)
                k = min(k, P)

                q = corr_score_prev[pool_idx].astype(np.float32, copy=False)
                thr = np.partition(q, P - k)[P - k]
                sel_idx = pool_idx[q >= thr]
                print(f"[TemplateSelect] Iter{it+1}: top {keep_ratio*100:.1f}% in pool => {len(sel_idx)} frames")

            global_template = img_stack[sel_idx].mean(axis=0).astype(np.float32, copy=False)  # [H,W]
            global_template_t = torch.from_numpy(global_template).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]

            if bool(save_templates):
                tpl_path, tpl_writer = tpl_saver.open(it)
                tpl_writer.write(tpl_saver.cast(global_template), photometric="minisblack")
                tpl_writer.close()
                print(f"[Template] Saved global template → {tpl_path}")

            corr_score_this = np.zeros((T,), dtype=np.float32)
            motion_score_this = np.zeros((T,), dtype=np.float32)

            if it == 0:
                print(f"[Iter1] RIGID ONLY (translation) | reject if |shift| > {rigid_max_shift_px:.1f}px")

                GT1 = global_template_t.to(device, non_blocking=True)

                for start in tqdm(
                    range(0, T, int(batch_size)),
                    desc="🧱 RigidReg (phase-corr)",
                    ncols=100, unit="batch",
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
                ):
                    end = min(start + int(batch_size), T)
                    idx = slice(start, end)
                    B = end - start

                    img_batch = torch.from_numpy(img_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
                    raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)

                    img_q = _center_crop_stride_2d(img_batch, crop=rigid_crop, stride=rigid_stride)
                    gt_q  = _center_crop_stride_2d(GT1.expand(B, 1, H, W), crop=rigid_crop, stride=rigid_stride)

                    dy_s, dx_s = _phase_corr_shift_batched(img_q, gt_q)
                    s = max(int(rigid_stride), 1)
                    dy = dy_s * s
                    dx = dx_s * s

                    mag = torch.sqrt(dy * dy + dx * dx)
                    ok = (mag <= rigid_max_shift_px)

                    dy_apply = dy.clone()
                    dx_apply = dx.clone()
                    dy_apply[~ok] = 0.0
                    dx_apply[~ok] = 0.0

                    img_for_corr = _warp_translation_batched(
                        img_batch, dy_apply, dx_apply,
                        mode="bilinear", padding_mode="reflection", align_corners=True
                    )

                    img_cc = _center_crop_stride(img_for_corr, crop=corr_crop, stride=corr_stride)
                    gt_cc  = _center_crop_stride(GT1.expand(B, 1, H, W), crop=corr_crop, stride=corr_stride)
                    corr_b = batch_ncc_corr(img_cc, gt_cc).float()
                    corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)

                    warped_raw = _warp_translation_batched(
                        raw_batch, dy_apply, dx_apply,
                        mode=str(warp_mode), padding_mode="reflection", align_corners=True
                    )
                    warped_img = _warp_translation_batched(
                        img_batch, dy_apply, dx_apply,
                        mode=str(warp_mode), padding_mode="reflection", align_corners=True
                    )

                    raw_stack[idx] = warped_raw.squeeze(1).detach().cpu().numpy()
                    img_stack[idx] = warped_img.squeeze(1).detach().cpu().numpy()

                    motion_score_this[start:end] = mag.detach().float().cpu().numpy()

                    if save_mask_flow:
                        ok_mask = ok.to(torch.float32).view(B, 1, 1, 1).expand(B, 1, H, W)

                        prev_mask_u8 = mask_accum_stack[idx]
                        prev_mask = (
                            torch.from_numpy(prev_mask_u8).unsqueeze(1).to(device, non_blocking=True)
                            .float().div_(255.0)
                        )

                        warped_prev = _warp_translation_batched(
                            prev_mask, dy_apply, dx_apply,
                            mode="bilinear", padding_mode="zeros", align_corners=True
                        )

                        new_mask = (warped_prev * ok_mask).clamp_(0.0, 1.0)
                        mask_accum_stack[idx] = new_mask.mul(255.0).round().to(torch.uint8).squeeze(1).cpu().numpy()

                        flow_rigid = _shift_to_flow_field(dy_apply, dx_apply, H, W, device_=device)
                        flow_np = flow_rigid.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(flow_np)

                    del img_batch, raw_batch, img_q, gt_q, dy_s, dx_s, dy, dx, mag, ok, corr_b
                    del dy_apply, dx_apply, img_for_corr, img_cc, gt_cc, warped_raw, warped_img

                # =========================================================
                # NEW: save intermediate result after Iter1
                # =========================================================
                if bool(save_iter1_stack):
                    final_path = Path(stack_save_path)
                    if iter1_save_path is None:
                        iter1_path = final_path.with_name(final_path.stem + "_iter1_rigid" + final_path.suffix)
                    else:
                        iter1_path = Path(iter1_save_path)

                    inter_raw_u16 = (raw_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
                    tiff.imwrite(str(iter1_path), inter_raw_u16)
                    print(f"[Saved] Iter1 rigid stack → {iter1_path}")

                    if bool(save_iter1_img_stack):
                        iter1_img_path = iter1_path.with_name(iter1_path.stem + "_IMG" + iter1_path.suffix)
                        inter_img_u16 = (img_stack * (vmax_img - vmin_img) + vmin_img).clip(0, 65535).astype("uint16")
                        tiff.imwrite(str(iter1_img_path), inter_img_u16)
                        print(f"[Saved] Iter1 rigid img_stack → {iter1_img_path}")

            else:
                print("[Iter2+] NETWORK FLOW REGISTRATION")

                for start in tqdm(
                    range(0, T, int(batch_size)),
                    desc="🌀 Demotion [PyLoReg]",
                    ncols=100, unit="batch",
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
                ):
                    end = min(start + int(batch_size), T)
                    idx = slice(start, end)
                    B = end - start

                    img_batch = torch.from_numpy(img_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
                    raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)

                    GT = global_template_t.expand(B, 1, H, W).contiguous().to(device, non_blocking=True)

                    img_q = _center_crop_stride(img_batch, crop=corr_crop, stride=corr_stride)
                    gt_q  = _center_crop_stride(GT,        crop=corr_crop, stride=corr_stride)
                    corr_b = batch_ncc_corr(img_q, gt_q).float()
                    corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)

                    img_p, pad = pad_to_multiple(img_batch, 16)
                    GT_p, _ = pad_to_multiple(GT, 16)

                    if amp_enabled:
                        with torch.cuda.amp.autocast(True):
                            out = model(GT_p, img_p, use_feature_num)
                    else:
                        out = model(GT_p, img_p, use_feature_num)

                    flow_b2hw = unpad(out["flow_preds"][-1], pad)  # [B,2,H,W]
                    flow_bhw2 = flow_b2hw.permute(0, 2, 3, 1).contiguous()  # [B,H,W,2]

                    if str(motion_metric) == "absmean":
                        score_b = flow_b2hw.abs().mean(dim=(1, 2, 3))
                    else:
                        score_b = torch.sqrt(flow_b2hw[:, 0].pow(2) + flow_b2hw[:, 1].pow(2)).mean(dim=(1, 2))
                    motion_score_this[start:end] = score_b.detach().float().cpu().numpy()

                    warped_raw_b1hw, warped_img_b1hw, mask_b1hw = warp_two_batches_once_with_mask(
                        raw_batch, img_batch, flow_bhw2,
                        grid_cache=grid_cache,
                        mode=str(warp_mode),
                        padding_mode="reflection",
                        align_corners=True
                    )

                    raw_stack[idx] = warped_raw_b1hw.squeeze(1).cpu().numpy()
                    img_stack[idx] = warped_img_b1hw.squeeze(1).cpu().numpy()

                    if save_mask_flow:
                        prev_mask_u8 = mask_accum_stack[idx]
                        prev_mask = (
                            torch.from_numpy(prev_mask_u8).unsqueeze(1).to(device, non_blocking=True)
                            .float().div_(255.0)
                        )
                        grid = grid_cache.flow_to_grid(flow_bhw2).to(dtype=prev_mask.dtype)
                        warped_prev = F.grid_sample(prev_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                        new_mask = (warped_prev * mask_b1hw).clamp_(0.0, 1.0)
                        mask_accum_stack[idx] = new_mask.mul(255.0).round().to(torch.uint8).squeeze(1).cpu().numpy()

                        prev_np = fq.decode_u16(flow_stack[idx])
                        prev_t = torch.from_numpy(prev_np).to(device, non_blocking=True)
                        composed = compose_flow_batch(prev_t, flow_bhw2, grid_cache=grid_cache)
                        composed_np = composed.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(composed_np)

                    del img_batch, raw_batch, GT, img_p, GT_p, out, flow_b2hw, flow_bhw2
                    del warped_raw_b1hw, warped_img_b1hw, mask_b1hw, img_q, gt_q, corr_b

            corr_score_prev = corr_score_this
            motion_score_prev = motion_score_this

            try:
                med = float(np.percentile(corr_score_prev[pool_idx], 50))
                p90 = float(np.percentile(corr_score_prev[pool_idx], 90))
                p99 = float(np.percentile(corr_score_prev[pool_idx], 99))
                print(f"[Iter {it+1}] corr(pool) median={med:.4f}, p90={p90:.4f}, p99={p99:.4f}")
            except Exception:
                pass

            try:
                p95m = float(np.percentile(motion_score_prev, 95))
                p99m = float(np.percentile(motion_score_prev, 99))
                print(f"[Iter {it+1}] motion_score p95={p95m:.3f}, p99={p99m:.3f}")
            except Exception:
                pass

    # ---------- save mask/flow ----------
    if save_mask_flow:
        tiff.imwrite(mask_folder / f"{stem}_mask_final.tif", mask_accum_stack)
        tiff.imwrite(flow_folder / f"{stem}_flow_u_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif", flow_stack[..., 0])
        tiff.imwrite(flow_folder / f"{stem}_flow_v_u_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif", flow_stack[..., 1])
        print("[Saved] Mask & Flow")

    # ---------- save final stack ----------
    final = (raw_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
    tiff.imwrite(stack_save_path, final)
    print("\n[Done] Demotion saved →", stack_save_path)
    return stack_save_path




def demotion_PyLoReg_infer2stack_less_save_acc_v3(
    img_stack_path,
    raw_stack_path,
    stack_save_path,
    model_root="PyLoReg//PyLoReg_model",
    model_name="GM3_fn5_202511301551",
    feature_channels=128,
    use_feature_num=4,
    iteration_num=2,            # kept for backward compat; NOT used to control net iters anymore
    batch_size=1,
    substack_len=1000,
    template_mode="mean",
    gaussian_sigma=0.0,
    max_frames=1000,
    crop_size=None,
    save_mask_flow=True,
    # ===== flow uint16 quant params =====
    flow_scale=100.0,
    flow_offset=65535.0 / 2.0,
    # ===== extra speed knobs =====
    use_amp=True,
    cache_base_grid=True,
    cache_prefix_mean=True,
    # ===== warp mode =====
    warp_mode="nearest",
    # ===== old knobs =====
    drop_large_motion_in_template=True,
    motion_thr_px=20.0,
    motion_metric="l2mean",
    # ===== save templates =====
    save_templates=False,
    template_save_dir=None,
    template_save_dtype="uint16",

    # =========================================================
    # RIGID-only Iter1
    # =========================================================
    rigid_max_shift_ratio=0.2,  # 画面大小（min(H,W)）的比例
    rigid_crop=None,
    rigid_stride=1,
    # =========================================================
    # NEW: Iter1 template strategy (YOU CONTROL)
    # =========================================================
    iter1_template_policy="prev_with_fallback",  # "prev" | "anchor" | "prev_with_fallback"
    iter1_anchor_every=50,                       # 每隔 N 帧强制用 anchor 重新锚定（防 drift）
    iter1_fallback_corr_thr=0.25,                # prev 对齐后 corr 低于阈值 -> fallback 到 anchor 再试
    iter1_anchor_source="global_mean",           # "global_mean" | "running_mean"
    iter1_running_mean_alpha=0.05,               # running mean 更新速率（仅当 anchor_source="running_mean"）
    iter1_force_sequential=True,                 # prev-frame 策略建议 True（保证 t-1 已更新）
    # =========================================================
    # NEW: network iterations (Iter2, Iter3, ...)
    # net_iter_num=2 => Iter2 + Iter3 (two NN rounds)
    # =========================================================
    net_iter_num=2,
    # =========================================================
    # save intermediate (after Iter1 rigid correction)
    # =========================================================
    save_iter1_stack=False,
    iter1_save_path=None,       # None => auto: <final>_iter1_rigid.tif
    save_iter1_img_stack=False,  # also save img_stack after iter1 for QC
):
    import os
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import tifffile as tiff
    from pathlib import Path
    from tqdm import tqdm

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.autograd.set_detect_anomaly(False)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device] Using:", device)

    # ---------- helpers (assumed existing in your project) ----------
    # GridCache, FlowQuantizer, TemplateSaver, PyLoRegNet,
    # load_weights_safely, pad_to_multiple, unpad,
    # warp_two_batches_once_with_mask, batch_ncc_corr,
    # compose_flow_batch
    grid_cache = GridCache(cache_base_grid=cache_base_grid)
    fq = FlowQuantizer(flow_scale=flow_scale, flow_offset=flow_offset)
    tpl_saver = TemplateSaver(stack_save_path, template_save_dir, template_save_dtype)

    # =========================================================
    # RIGID (translation-only) helper functions
    # =========================================================
    def _center_crop_stride_2d(x_b1hw: torch.Tensor, crop: int, stride: int = 1) -> torch.Tensor:
        if crop is None:
            y = x_b1hw
        else:
            B, C, H_, W_ = x_b1hw.shape
            c = int(crop)
            c = min(c, H_, W_)
            h0 = (H_ - c) // 2
            w0 = (W_ - c) // 2
            y = x_b1hw[:, :, h0:h0 + c, w0:w0 + c]
        s = max(int(stride), 1)
        if s > 1:
            y = y[:, :, ::s, ::s]
        return y

    def _phase_corr_shift_batched(x_b1hw: torch.Tensor, ref_b1hw: torch.Tensor, eps: float = 1e-9):
        """
        Return dy, dx (float32) for each batch item.
        """
        x = x_b1hw - x_b1hw.mean(dim=(-2, -1), keepdim=True)
        r = ref_b1hw - ref_b1hw.mean(dim=(-2, -1), keepdim=True)

        X = torch.fft.rfft2(x, dim=(-2, -1))
        R = torch.fft.rfft2(r, dim=(-2, -1))

        CPS = X * torch.conj(R)
        CPS = CPS / (torch.abs(CPS) + eps)

        cc = torch.fft.irfft2(CPS, s=x.shape[-2:], dim=(-2, -1))  # [B,1,H,W]
        cc = cc.squeeze(1)  # [B,H,W]

        B, H_, W_ = cc.shape
        flat = cc.reshape(B, -1)
        idx = flat.argmax(dim=1)
        peak_y = (idx // W_).to(torch.int64)
        peak_x = (idx % W_).to(torch.int64)

        dy = peak_y.to(torch.float32)
        dx = peak_x.to(torch.float32)

        dy = torch.where(dy > (H_ // 2), dy - H_, dy)
        dx = torch.where(dx > (W_ // 2), dx - W_, dx)
        return dy, dx

    def _warp_translation_batched(
        x_b1hw: torch.Tensor,
        dy: torch.Tensor,
        dx: torch.Tensor,
        mode: str = "bilinear",
        padding_mode: str = "reflection",
        align_corners: bool = True,
    ):
        B, C, H_, W_ = x_b1hw.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H_, device=x_b1hw.device),
            torch.linspace(-1, 1, W_, device=x_b1hw.device),
            indexing="ij"
        )
        base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H,W,2]

        sx = (2.0 * dx) / max(W_ - 1, 1)
        sy = (2.0 * dy) / max(H_ - 1, 1)

        base[..., 0] = base[..., 0] - sx.view(B, 1, 1)
        base[..., 1] = base[..., 1] - sy.view(B, 1, 1)

        y = F.grid_sample(x_b1hw, base, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return y

    def _shift_to_flow_field(dy: torch.Tensor, dx: torch.Tensor, H_: int, W_: int, device_) -> torch.Tensor:
        B = dy.numel()
        flow = torch.zeros((B, H_, W_, 2), device=device_, dtype=torch.float32)
        flow[..., 0] = dx.view(B, 1, 1)
        flow[..., 1] = dy.view(B, 1, 1)
        return flow

    # =========================================================
    # Build/load model (ONLY used for Iter2+)
    # =========================================================
    net_iter_num = int(net_iter_num)
    model = None
    amp_enabled = bool(use_amp and device.type == "cuda")

    # Resolve checkpoint path relative to repo root, not current working directory.
    # This makes the function robust when NeuMar_DCPYDC_valid.py is launched from
    # a different CWD.
    repo_root = Path(__file__).resolve().parents[1]  # .../NeuroPilot_0302
    model_root_path = Path(model_root)
    if not model_root_path.is_absolute():
        model_root_path = repo_root / model_root_path
    model_path = str(model_root_path / model_name / "gmflow_latest.pt")
    if net_iter_num > 0:
        if not os.path.isfile(model_path):
            # Fallback: checkpoint missing => skip Iter2+ network rounds.
            print(f"[WARN] PyLoReg checkpoint not found: {model_path}. Skip network iters (net_iter_num=0).")
            net_iter_num = 0
        else:
            base_model = PyLoRegNet(feature_channels=feature_channels).to(device)
            model = nn.DataParallel(base_model) if torch.cuda.device_count() > 1 else base_model
            # model = torch.compile(model, mode="reduce-overhead")
            import platform
            if platform.system() != "Windows":
                model = torch.compile(model, mode="reduce-overhead")
            else:
                print("[INFO] torch.compile disabled on Windows")

            print("[Model] Loading:", model_path)
            model, _, info = load_weights_safely(model, model_path, device="cuda", strict=False, verbose=True)
            if isinstance(info, dict):
                print("[ModelLoad] strategy:", info.get("strategy", "unknown"))
            model.eval()

    # =========================================================
    # Read stacks
    # =========================================================
    img_u16 = tiff.imread(img_stack_path)
    raw_u16 = tiff.imread(raw_stack_path)
    print("[Img] :", img_u16.shape)
    print("[Raw] :", raw_u16.shape)

    T = min(len(img_u16), len(raw_u16))
    if max_frames is not None:
        T = min(T, int(max_frames))
    img_u16 = img_u16[:T]
    raw_u16 = raw_u16[:T]
    print(f"[Use] first {T} frames")

    img_f32 = img_u16.astype(np.float32, copy=False)
    raw_f32 = raw_u16.astype(np.float32, copy=False)
    vmin_img, vmax_img = float(img_f32.min()), float(img_f32.max())
    vmin_raw, vmax_raw = float(raw_f32.min()), float(raw_f32.max())

    img_stack = (img_f32 - vmin_img) / max(vmax_img - vmin_img, 1e-12)
    raw_stack = (raw_f32 - vmin_raw) / max(vmax_raw - vmin_raw, 1e-12)

    # crop
    T0, H0, W0 = img_stack.shape
    if crop_size is not None and (H0 > crop_size or W0 > crop_size):
        h0, w0 = (H0 - crop_size) // 2, (W0 - crop_size) // 2
        img_stack = img_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        raw_stack = raw_stack[:, h0:h0 + crop_size, w0:w0 + crop_size]
        print("[Crop] →", img_stack.shape)

    T, H, W = img_stack.shape

    # rigid reject threshold
    rigid_max_shift_px = float(rigid_max_shift_ratio) * float(min(H, W))
    print(f"[Rigid] max allowed shift = {rigid_max_shift_px:.1f} px "
          f"({rigid_max_shift_ratio*100:.0f}% of min(H,W)={min(H,W)})")

    # =========================================================
    # Containers for mask/flow
    # =========================================================
    if save_mask_flow:
        flow_stack = np.zeros((T, H, W, 2), dtype=np.uint16)
        mask_accum_stack = np.full((T, H, W), 255, dtype=np.uint8)

        stack_path = Path(stack_save_path)
        folder = stack_path.parent
        folder.mkdir(exist_ok=True, parents=True)
        stem = stack_path.stem
        mask_folder = folder.with_name(folder.name + "_mask")
        flow_folder = folder.with_name(folder.name + "_flow")
        mask_file_name = f"{stem}_mask_final.tif"
        flow_u_file_name = f"{stem}_flow_u_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif"
        flow_v_file_name = f"{stem}_flow_v_u_u16_x{int(flow_scale)}_off{int(flow_offset)}.tif"

        def _resolve_aux_dir(default_dir: Path, sample_name: str, short_root_name: str) -> Path:
            max_len = 240 if os.name == "nt" else 4096
            candidate = default_dir / sample_name
            if len(str(candidate)) <= max_len:
                default_dir.mkdir(exist_ok=True, parents=True)
                return default_dir

            short_root = folder.parent / short_root_name
            short_root.mkdir(exist_ok=True, parents=True)
            short_key = hashlib.md5(str(folder).encode("utf-8")).hexdigest()[:8]
            short_dir = short_root / short_key
            short_dir.mkdir(exist_ok=True, parents=True)
            print(
                f"[PATH-SHORTEN] auxiliary output redirected: "
                f"{default_dir} -> {short_dir} (sample_len={len(str(candidate))})"
            )
            return short_dir

        mask_folder = _resolve_aux_dir(mask_folder, mask_file_name, "_mask_short")
        flow_folder = _resolve_aux_dir(flow_folder, flow_u_file_name, "_flow_short")
    else:
        flow_stack = None
        mask_accum_stack = None
        stack_path = Path(stack_save_path)
        stack_path.parent.mkdir(exist_ok=True, parents=True)
        stem = stack_path.stem

    # =========================================================
    # Global reference pool for template (used for template building each iter)
    # =========================================================
    ref_pool_size = min(T, 2000)
    pool_mode = "uniform"
    pool_seed = 0

    refine_step = 0.10
    refine_cap = 0.40

    corr_crop = 256
    corr_stride = 2

    def make_pool_indices(T_, n, mode="uniform", seed=0):
        if n >= T_:
            return np.arange(T_, dtype=np.int64)
        if mode == "random":
            rng = np.random.default_rng(seed)
            idx = rng.choice(T_, size=n, replace=False)
            idx.sort()
            return idx.astype(np.int64)
        idx = np.linspace(0, T_ - 1, num=n, dtype=np.int64)
        idx = np.unique(idx)
        return idx.astype(np.int64)

    pool_idx = make_pool_indices(T, ref_pool_size, mode=pool_mode, seed=pool_seed)
    print(f"[RefPool] mode={pool_mode} size={len(pool_idx)} / T={T}")

    corr_score_prev = None
    motion_score_prev = None

    # =========================================================
    # Total iterations: Iter1 rigid + net_iter_num NN rounds
    # =========================================================
    total_iters = 1 + int(net_iter_num)
    print(f"[Iters] total={total_iters} (Iter1 rigid + net_iter_num={int(net_iter_num)} network rounds)")

    # =========================================================
    # Main loop
    # =========================================================
    with torch.inference_mode():
        for it in range(total_iters):
            print(f"\n[Iter {it+1}/{total_iters}] ...")

            # ---------------- template select (for building global_template only) ----------------
            if corr_score_prev is None:
                sel_idx = pool_idx
                print(f"[TemplateSelect] Iter{it+1}: init from ALL pool frames ({len(sel_idx)})")
            else:
                # it starts at 1 for Iter2; keep_ratio uses 'it' like original intent
                keep_ratio = float(np.clip(refine_step * it, 0.0, refine_cap))  # it=1 -> 10%
                P = len(pool_idx)
                k = max(int(np.ceil(P * keep_ratio)), 10)
                k = min(k, P)

                q = corr_score_prev[pool_idx].astype(np.float32, copy=False)
                thr = np.partition(q, P - k)[P - k]
                sel_idx = pool_idx[q >= thr]
                print(f"[TemplateSelect] Iter{it+1}: top {keep_ratio*100:.1f}% in pool => {len(sel_idx)} frames")

            global_template = img_stack[sel_idx].mean(axis=0).astype(np.float32, copy=False)  # [H,W]
            global_template_t = torch.from_numpy(global_template).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]

            if bool(save_templates):
                tpl_path, tpl_writer = tpl_saver.open(it)
                tpl_writer.write(tpl_saver.cast(global_template), photometric="minisblack")
                tpl_writer.close()
                print(f"[Template] Saved global template → {tpl_path}")

            corr_score_this = np.zeros((T,), dtype=np.float32)
            motion_score_this = np.zeros((T,), dtype=np.float32)

            if it == 0:
                # =========================================================
                # Iter1: RIGID ONLY, template policy
                # =========================================================
                print(
                    f"[Iter1] RIGID ONLY | policy={iter1_template_policy} "
                    f"| anchor_every={iter1_anchor_every} | fallback_corr_thr={iter1_fallback_corr_thr} "
                    f"| anchor_source={iter1_anchor_source} | reject if |shift| > {rigid_max_shift_px:.1f}px"
                )

                # ---- prev policy recommended sequential to ensure t-1 already updated ----
                iter1_bs = int(batch_size)
                if iter1_force_sequential and ("prev" in str(iter1_template_policy)):
                    if iter1_bs != 1:
                        print(f"[Iter1] NOTE: iter1_force_sequential=True, override batch_size {iter1_bs} -> 1 (strict prev-template).")
                    iter1_bs = 1

                # ---- anchor template state ----
                running_anchor = global_template.copy()  # [H,W] float32
                running_anchor_t = torch.from_numpy(running_anchor).unsqueeze(0).unsqueeze(0).to(device)

                def _get_anchor_template_b1hw(B=1):
                    if str(iter1_anchor_source) == "running_mean":
                        return running_anchor_t.expand(B, 1, H, W).contiguous()
                    else:
                        return global_template_t.expand(B, 1, H, W).contiguous().to(device, non_blocking=True)

                def _rigid_register_to_ref(img_batch, raw_batch, ref_batch):
                    """
                    Return warped_raw, warped_img, dy_apply, dx_apply, mag, corr_b, ok
                    """
                    img_q = _center_crop_stride_2d(img_batch, crop=rigid_crop, stride=rigid_stride)
                    ref_q = _center_crop_stride_2d(ref_batch, crop=rigid_crop, stride=rigid_stride)
                    dy_s, dx_s = _phase_corr_shift_batched(img_q, ref_q)

                    s = max(int(rigid_stride), 1)
                    dy = dy_s * s
                    dx = dx_s * s

                    mag = torch.sqrt(dy * dy + dx * dx)
                    ok = (mag <= rigid_max_shift_px)

                    dy_apply = dy.clone()
                    dx_apply = dx.clone()
                    dy_apply[~ok] = 0.0
                    dx_apply[~ok] = 0.0

                    # warp current toward ref
                    warped_img_for_corr = _warp_translation_batched(
                        img_batch, dy_apply, dx_apply,
                        mode="bilinear", padding_mode="reflection", align_corners=True
                    )

                    # corr score: NCC(warped_img, ref)
                    img_cc = _center_crop_stride_2d(warped_img_for_corr, crop=corr_crop, stride=corr_stride)
                    ref_cc = _center_crop_stride_2d(ref_batch,            crop=corr_crop, stride=corr_stride)
                    corr_b = batch_ncc_corr(img_cc, ref_cc).float()  # [B]

                    warped_raw = _warp_translation_batched(
                        raw_batch, dy_apply, dx_apply,
                        mode=str(warp_mode), padding_mode="reflection", align_corners=True
                    )
                    warped_img = _warp_translation_batched(
                        img_batch, dy_apply, dx_apply,
                        mode=str(warp_mode), padding_mode="reflection", align_corners=True
                    )
                    return warped_raw, warped_img, dy_apply, dx_apply, mag, corr_b, ok

                for start in tqdm(
                    range(0, T, iter1_bs),
                    desc="🧱 Iter1 RigidReg (policy)",
                    ncols=100, unit="batch",
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
                ):
                    end = min(start + iter1_bs, T)
                    idx = slice(start, end)
                    B = end - start

                    img_batch = torch.from_numpy(img_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
                    raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)

                    # ----- choose ref -----
                    policy = str(iter1_template_policy)

                    if policy == "anchor":
                        ref_batch = _get_anchor_template_b1hw(B)
                    else:
                        # "prev" or "prev_with_fallback"
                        if start == 0:
                            ref_batch = _get_anchor_template_b1hw(B)
                        else:
                            ref_np = img_stack[np.clip(np.arange(start, end) - 1, 0, T - 1)]
                            ref_batch = torch.from_numpy(ref_np).unsqueeze(1).to(device, non_blocking=True)

                        # periodic re-anchor
                        if int(iter1_anchor_every) > 0 and iter1_bs == 1:
                            t = start
                            if (t % int(iter1_anchor_every)) == 0:
                                ref_batch = _get_anchor_template_b1hw(B)

                    # ----- first pass -----
                    warped_raw, warped_img, dy_apply, dx_apply, mag, corr_b, ok = _rigid_register_to_ref(
                        img_batch, raw_batch, ref_batch
                    )

                    # ----- fallback (prev_with_fallback only) -----
                    if policy == "prev_with_fallback":
                        fb = (corr_b < float(iter1_fallback_corr_thr))
                        if torch.any(fb):
                            ref_anchor = _get_anchor_template_b1hw(B)
                            warped_raw2, warped_img2, dy2, dx2, mag2, corr2, ok2 = _rigid_register_to_ref(
                                img_batch, raw_batch, ref_anchor
                            )

                            # choose per-item better result (only for fb items)
                            better = (corr2 > corr_b)
                            choose2 = (fb & better)

                            if torch.any(choose2):
                                choose2_v = choose2.view(B, 1, 1, 1)
                                warped_raw = torch.where(choose2_v, warped_raw2, warped_raw)
                                warped_img = torch.where(choose2_v, warped_img2, warped_img)
                                dy_apply = torch.where(choose2, dy2, dy_apply)
                                dx_apply = torch.where(choose2, dx2, dx_apply)
                                mag = torch.where(choose2, mag2, mag)
                                corr_b = torch.where(choose2, corr2, corr_b)
                                ok = torch.where(choose2, ok2, ok)

                            del ref_anchor, warped_raw2, warped_img2, dy2, dx2, mag2, corr2, ok2, better, choose2

                    # ----- write back -----
                    raw_stack[idx] = warped_raw.squeeze(1).detach().cpu().numpy()
                    img_stack[idx] = warped_img.squeeze(1).detach().cpu().numpy()

                    corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)
                    motion_score_this[start:end] = mag.detach().float().cpu().numpy()

                    # ----- update running anchor (optional) -----
                    if str(iter1_anchor_source) == "running_mean":
                        alpha = float(iter1_running_mean_alpha)
                        if alpha > 0:
                            cur_mean = img_stack[idx].mean(axis=0).astype(np.float32, copy=False)
                            running_anchor = (1.0 - alpha) * running_anchor + alpha * cur_mean
                            running_anchor_t = torch.from_numpy(running_anchor).unsqueeze(0).unsqueeze(0).to(device)

                    # ----- save mask/flow -----
                    if save_mask_flow:
                        ok_mask = ok.to(torch.float32).view(B, 1, 1, 1).expand(B, 1, H, W)

                        prev_mask_u8 = mask_accum_stack[idx]
                        prev_mask = (
                            torch.from_numpy(prev_mask_u8).unsqueeze(1).to(device, non_blocking=True)
                            .float().div_(255.0)
                        )

                        warped_prev = _warp_translation_batched(
                            prev_mask, dy_apply, dx_apply,
                            mode="bilinear", padding_mode="zeros", align_corners=True
                        )

                        new_mask = (warped_prev * ok_mask).clamp_(0.0, 1.0)
                        mask_accum_stack[idx] = new_mask.mul(255.0).round().to(torch.uint8).squeeze(1).cpu().numpy()

                        flow_rigid = _shift_to_flow_field(dy_apply, dx_apply, H, W, device_=device)
                        flow_np = flow_rigid.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(flow_np)

                    del img_batch, raw_batch, ref_batch
                    del warped_raw, warped_img, dy_apply, dx_apply, mag, corr_b, ok

                # =========================================================
                # save intermediate result after Iter1
                # =========================================================
                if bool(save_iter1_stack):
                    final_path = Path(stack_save_path)
                    if iter1_save_path is None:
                        iter1_path = final_path.with_name(final_path.stem + "_iter1_rigid" + final_path.suffix)
                    else:
                        iter1_path = Path(iter1_save_path)

                    inter_raw_u16 = (raw_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
                    tiff.imwrite(str(iter1_path), inter_raw_u16)
                    print(f"[Saved] Iter1 rigid stack → {iter1_path}")

                    if bool(save_iter1_img_stack):
                        iter1_img_path = iter1_path.with_name(iter1_path.stem + "_IMG" + iter1_path.suffix)
                        inter_img_u16 = (img_stack * (vmax_img - vmin_img) + vmin_img).clip(0, 65535).astype("uint16")
                        tiff.imwrite(str(iter1_img_path), inter_img_u16)
                        print(f"[Saved] Iter1 rigid img_stack → {iter1_img_path}")

            else:
                # =========================================================
                # Iter2+: NETWORK FLOW REGISTRATION (global template)
                # =========================================================
                print(f"[Iter{it+1}] NETWORK FLOW REGISTRATION (global template)")

                for start in tqdm(
                    range(0, T, int(batch_size)),
                    desc="🌀 Demotion [PyLoReg]",
                    ncols=100, unit="batch",
                    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
                ):
                    end = min(start + int(batch_size), T)
                    idx = slice(start, end)
                    B = end - start

                    img_batch = torch.from_numpy(img_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
                    raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)

                    GT = global_template_t.expand(B, 1, H, W).contiguous().to(device, non_blocking=True)

                    # corr score for template refine
                    img_q = _center_crop_stride_2d(img_batch, crop=corr_crop, stride=corr_stride)
                    gt_q  = _center_crop_stride_2d(GT,        crop=corr_crop, stride=corr_stride)
                    corr_b = batch_ncc_corr(img_q, gt_q).float()
                    corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)

                    img_p, pad = pad_to_multiple(img_batch, 16)
                    GT_p, _ = pad_to_multiple(GT, 16)

                    if amp_enabled:
                        with torch.cuda.amp.autocast(True):
                            out = model(GT_p, img_p, use_feature_num)
                    else:
                        out = model(GT_p, img_p, use_feature_num)

                    flow_b2hw = unpad(out["flow_preds"][-1], pad)  # [B,2,H,W]
                    flow_bhw2 = flow_b2hw.permute(0, 2, 3, 1).contiguous()  # [B,H,W,2]

                    if str(motion_metric) == "absmean":
                        score_b = flow_b2hw.abs().mean(dim=(1, 2, 3))
                    else:
                        score_b = torch.sqrt(flow_b2hw[:, 0].pow(2) + flow_b2hw[:, 1].pow(2)).mean(dim=(1, 2))
                    motion_score_this[start:end] = score_b.detach().float().cpu().numpy()

                    warped_raw_b1hw, warped_img_b1hw, mask_b1hw = warp_two_batches_once_with_mask(
                        raw_batch, img_batch, flow_bhw2,
                        grid_cache=grid_cache,
                        mode=str(warp_mode),
                        padding_mode="reflection",
                        align_corners=True
                    )

                    raw_stack[idx] = warped_raw_b1hw.squeeze(1).cpu().numpy()
                    img_stack[idx] = warped_img_b1hw.squeeze(1).cpu().numpy()

                    if save_mask_flow:
                        prev_mask_u8 = mask_accum_stack[idx]
                        prev_mask = (
                            torch.from_numpy(prev_mask_u8).unsqueeze(1).to(device, non_blocking=True)
                            .float().div_(255.0)
                        )
                        grid = grid_cache.flow_to_grid(flow_bhw2).to(dtype=prev_mask.dtype)
                        warped_prev = F.grid_sample(prev_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                        new_mask = (warped_prev * mask_b1hw).clamp_(0.0, 1.0)
                        mask_accum_stack[idx] = new_mask.mul(255.0).round().to(torch.uint8).squeeze(1).cpu().numpy()

                        prev_np = fq.decode_u16(flow_stack[idx])
                        prev_t = torch.from_numpy(prev_np).to(device, non_blocking=True)
                        composed = compose_flow_batch(prev_t, flow_bhw2, grid_cache=grid_cache)
                        composed_np = composed.detach().cpu().numpy().astype(np.float32, copy=False)
                        flow_stack[idx] = fq.encode_u16(composed_np)

                    del img_batch, raw_batch, GT, img_p, GT_p, out, flow_b2hw, flow_bhw2
                    del warped_raw_b1hw, warped_img_b1hw, mask_b1hw, img_q, gt_q, corr_b

            # ---------------- end of iteration bookkeeping ----------------
            corr_score_prev = corr_score_this
            motion_score_prev = motion_score_this

            try:
                med = float(np.percentile(corr_score_prev[pool_idx], 50))
                p90 = float(np.percentile(corr_score_prev[pool_idx], 90))
                p99 = float(np.percentile(corr_score_prev[pool_idx], 99))
                print(f"[Iter {it+1}] corr(pool) median={med:.4f}, p90={p90:.4f}, p99={p99:.4f}")
            except Exception:
                pass

            try:
                p95m = float(np.percentile(motion_score_prev, 95))
                p99m = float(np.percentile(motion_score_prev, 99))
                print(f"[Iter {it+1}] motion_score p95={p95m:.3f}, p99={p99m:.3f}")
            except Exception:
                pass

    # =========================================================
    # save mask/flow
    # =========================================================
    if save_mask_flow:
        tiff.imwrite(mask_folder / mask_file_name, mask_accum_stack)
        tiff.imwrite(flow_folder / flow_u_file_name, flow_stack[..., 0])
        tiff.imwrite(flow_folder / flow_v_file_name, flow_stack[..., 1])
        print("[Saved] Mask & Flow")

    # =========================================================
    # save final stack
    # =========================================================
    final = (raw_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
    tiff.imwrite(stack_save_path, final)
    print("\n[Done] Demotion saved →", stack_save_path)
    return stack_save_path


# If you want to run directly:
# if __name__ == "__main__":
#     demotion_PyLoReg_infer2stack_less_save_acc_v3(
#         img_stack_path="img.tif",
#         raw_stack_path="raw.tif",
#         stack_save_path="out.tif",
#         iteration_num=3,
#         batch_size=2,
#         max_frames=2000,
#         save_templates=True,
#     )
