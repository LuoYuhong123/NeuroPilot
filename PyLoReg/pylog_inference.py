#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaned PyLoReg inference module.

Main behavior:
    1) Iteratively estimates motion on img_stack.
    2) Warps img_stack during each iteration for template refinement.
    3) Composes all rigid/dense flows into one cumulative flow.
    4) Warps raw_stack only once at the end.

This file intentionally removes older duplicated v3 code and the unused
single-image helper from the pasted version. Compatibility aliases are kept
at the bottom for existing scripts.
"""

import os
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
# 2) Flow quant helpers
# =========================================================

@dataclass
class FlowQuantizer:
    flow_scale: float | None
    flow_offset: float | None
    storage_dtype: str = "uint16"

    def __post_init__(self):
        dtype_name = str(self.storage_dtype).lower()
        if dtype_name in {"uint16", "u16"}:
            self.storage_dtype = "uint16"
            self.np_dtype = np.dtype(np.uint16)
        elif dtype_name in {"uint8", "u8"}:
            self.storage_dtype = "uint8"
            self.np_dtype = np.dtype(np.uint8)
        else:
            raise ValueError(f"flow_storage_dtype must be 'uint16' or 'uint8', got {self.storage_dtype!r}")

        self.max_value = float(np.iinfo(self.np_dtype).max)
        if self.flow_scale is None:
            self.flow_scale = 100.0 if self.storage_dtype == "uint16" else 1.0
        self.flow_scale = float(self.flow_scale)
        if self.flow_scale <= 0:
            raise ValueError(f"flow_scale must be > 0, got {self.flow_scale}")

        if self.flow_offset is None:
            self.flow_offset = self.max_value / 2.0
        self.flow_offset = float(self.flow_offset)

    def encode(self, flow_hw2: np.ndarray) -> np.ndarray:
        q = np.rint(flow_hw2 * float(self.flow_scale) + float(self.flow_offset))
        q = np.clip(q, 0, self.max_value).astype(self.np_dtype)
        return q

    def decode(self, q_hw2: np.ndarray) -> np.ndarray:
        return (q_hw2.astype(np.float32) - float(self.flow_offset)) / float(self.flow_scale)

    # Backward-compatible method names for older local callers.
    def encode_u16(self, flow_hw2: np.ndarray) -> np.ndarray:
        return self.encode(flow_hw2)

    def decode_u16(self, q_hw2: np.ndarray) -> np.ndarray:
        return self.decode(q_hw2)


class _TensorRTFlowRunner:
    """Run the exported flow-only PyLoReg TensorRT engine on existing CUDA tensors."""

    def __init__(self, engine_path, device):
        if str(device).split(":")[0] != "cuda":
            raise RuntimeError("TensorRT runner requires CUDA tensors.")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT is not installed in this Python environment.") from exc

        self.trt = trt
        self.engine_path = str(engine_path)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        serialized = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name for name in self.tensor_names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if set(self.input_names) != {"GT", "img"} or len(self.output_names) != 1:
            raise RuntimeError(
                f"Unexpected TensorRT IO. inputs={self.input_names}, outputs={self.output_names}"
            )
        self.output_name = self.output_names[0]
        self.input_shapes = {
            name: tuple(int(v) for v in self.engine.get_tensor_shape(name))
            for name in self.input_names
        }
        self.output_shape = tuple(int(v) for v in self.engine.get_tensor_shape(self.output_name))

    def __call__(self, gt_b1hw: torch.Tensor, img_b1hw: torch.Tensor) -> torch.Tensor:
        gt_b1hw = gt_b1hw.contiguous()
        img_b1hw = img_b1hw.contiguous()
        if tuple(gt_b1hw.shape) != self.input_shapes["GT"]:
            raise ValueError(f"TensorRT GT shape mismatch: got {tuple(gt_b1hw.shape)}, expected {self.input_shapes['GT']}")
        if tuple(img_b1hw.shape) != self.input_shapes["img"]:
            raise ValueError(f"TensorRT img shape mismatch: got {tuple(img_b1hw.shape)}, expected {self.input_shapes['img']}")

        out = torch.empty(self.output_shape, device=gt_b1hw.device, dtype=torch.float32)
        tensors = {"GT": gt_b1hw, "img": img_b1hw, self.output_name: out}
        for name in self.tensor_names:
            tensor = tensors[name]
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream(gt_b1hw.device)
        ok = self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        return out


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
    Compose two backward-sampling flow fields used by grid_sample.

    Convention used everywhere in this file:
        warped(p) = image(p + flow(p))

    If previous cumulative flow is prev and the newly estimated flow on the
    already-warped image is new, then:

        I1(p) = I0(p + prev(p))
        I2(p) = I1(p + new(p))
              = I0(p + new(p) + prev(p + new(p)))

    Therefore the correct one-shot flow from I0 to I2 is:

        composed(p) = new(p) + sample(prev, p + new(p))

    This is NOT the same as prev + sample(new, p + prev(p)).
    """
    target_dtype = new_bhw2.dtype
    prev_bhw2 = prev_bhw2.to(target_dtype)
    new_bhw2 = new_bhw2.to(target_dtype)

    # Sample the previous cumulative flow at coordinates p + new(p).
    prev_b2hw = prev_bhw2.permute(0, 3, 1, 2).contiguous()
    grid = grid_cache.flow_to_grid(new_bhw2).to(dtype=target_dtype)

    sampled_prev = F.grid_sample(
        prev_b2hw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled_prev_bhw2 = sampled_prev.permute(0, 2, 3, 1).contiguous()

    return new_bhw2 + sampled_prev_bhw2


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


# =========================================================
# 7) Main stack registration function
# =========================================================

def demotion_PyLoReg_infer2stack_one_shot_raw_warp(
    img_stack_path,
    raw_stack_path,
    stack_save_path,
    model_root="PyLoReg//PyLoReg_model",
    model_name="GM3_fn5_202511301551",
    feature_channels=128,
    use_feature_num=4,
    iteration_num=2,            # kept for backward compat; NOT used to control net iters anymore
    batch_size=2,
    substack_len=1000,
    template_mode="mean",
    gaussian_sigma=0.0,
    max_frames=1000,
    crop_size=None,
    save_mask_flow=True,
    # ===== flow quant params =====
    flow_storage_dtype="uint16",
    flow_scale=None,
    flow_offset=None,
    # ===== extra speed knobs =====
    use_amp=True,
    network_backend="tensorrt",
    tensorrt_engine_path=None,
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
    # NEW: robust template by stack-internal high-correlation frames
    # 每一轮都会在当前 img_stack 中重新做 frame quality selection，
    # 找到和“大多数帧/consensus”最相似的 high-correlation frames，
    # 然后只用这些帧平均/中位数生成本轮 global template。
    # =========================================================
    robust_template_each_iter=True,
    robust_template_pool_size=2000,       # 每轮最多抽多少帧用于计算 correlation，越大越稳但越慢
    robust_template_keep_ratio=0.20,      # 保留相关性最高的比例
    robust_template_min_frames=50,        # 至少保留多少帧，防止平均帧太少
    robust_template_crop=256,             # 只用中心 crop 算 correlation，加速且减少边缘影响；None=全图
    robust_template_stride=4,             # correlation 计算时下采样倍数；建议 4 或 8
    robust_template_batch=256,            # 分批提取特征，控制内存
    robust_template_stat="mean",          # "mean" | "median"，median 更抗异常但稍慢
    robust_template_use_gpu=False,        # 默认 CPU，避免占用配准网络显存；小图可改 True
    robust_template_eps=1e-6,
    save_robust_template_qc=True,         # 保存每轮模板/选中帧/scores，方便检查

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
    network_backend = str(network_backend).lower()
    if network_backend not in {"torch", "tensorrt"}:
        raise ValueError(f"network_backend must be 'torch' or 'tensorrt', got {network_backend!r}")

    print("========== PyLoReg cleaned: one-shot raw warp ==========")
    print(f"[Network] backend={network_backend}")
    # ---------- helpers (assumed existing in your project) ----------
    # GridCache, FlowQuantizer, TemplateSaver, PyLoRegNet,
    # load_weights_safely, pad_to_multiple, unpad,
    # warp_two_batches_once_with_mask, batch_ncc_corr,
    # compose_flow_batch
    grid_cache = GridCache(cache_base_grid=cache_base_grid)
    fq = FlowQuantizer(
        flow_scale=flow_scale,
        flow_offset=flow_offset,
        storage_dtype=flow_storage_dtype,
    )
    print(
        f"[Flow] storage_dtype={fq.storage_dtype}, "
        f"scale={float(fq.flow_scale):g}, offset={float(fq.flow_offset):g}, "
        f"step={1.0 / float(fq.flow_scale):.4g}px"
    )
    tpl_saver = TemplateSaver(stack_save_path, template_save_dir, template_save_dtype)
    net_iter_num = max(int(net_iter_num), 0)
    _translation_grid_cache = {}

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
        cache_key = (H_, W_, str(x_b1hw.device))
        base_1hw2 = _translation_grid_cache.get(cache_key)
        if base_1hw2 is None:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H_, device=x_b1hw.device),
                torch.linspace(-1, 1, W_, device=x_b1hw.device),
                indexing="ij"
            )
            base_1hw2 = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1,H,W,2]
            _translation_grid_cache[cache_key] = base_1hw2

        base = base_1hw2.expand(B, -1, -1, -1).clone()  # [B,H,W,2]

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

    def _encode_constant_translation_flow(dy_flow: torch.Tensor, dx_flow: torch.Tensor, H_: int, W_: int):
        """Encode a spatially constant translation flow without materializing it on GPU."""
        dy_np = dy_flow.detach().cpu().numpy().astype(np.float32, copy=False)
        dx_np = dx_flow.detach().cpu().numpy().astype(np.float32, copy=False)
        B = dy_np.shape[0]
        flow_np = np.empty((B, H_, W_, 2), dtype=np.float32)
        flow_np[..., 0] = dx_np[:, None, None]
        flow_np[..., 1] = dy_np[:, None, None]
        return fq.encode(flow_np)

    def _warp_stack_once_with_flow(
        x_b1hw: torch.Tensor,
        flow_bhw2: torch.Tensor,
        mode: str = "nearest",
        padding_mode: str = "reflection",
        align_corners: bool = True,
        return_valid_mask: bool = False,
    ):
        """
        Warp one stack/batch once with an already-composed flow field.

        This is used to avoid repeatedly interpolating raw data. During iterative
        registration we warp img_stack for refinement, but raw_stack is kept
        untouched. After all flows are composed, raw_stack is passed here once.
        """
        grid = grid_cache.flow_to_grid(flow_bhw2).to(dtype=x_b1hw.dtype)
        y = F.grid_sample(
            x_b1hw, grid,
            mode=str(mode),
            padding_mode=str(padding_mode),
            align_corners=align_corners,
        )
        if not return_valid_mask:
            return y

        ones = torch.ones_like(x_b1hw)
        valid_mask = F.grid_sample(
            ones, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        ).clamp_(0.0, 1.0)
        return y, valid_mask

    # =========================================================
    # Build/load model (ONLY used for Iter2+)
    # =========================================================
    model = None
    trt_runner = None
    amp_enabled = bool(use_amp and device.type == "cuda")
    model_root_path = Path(model_root)
    if not model_root_path.is_absolute():
        cwd_candidate = Path.cwd() / model_root_path
        file_candidate = Path(__file__).resolve().parents[1] / model_root_path
        model_root_path = cwd_candidate if cwd_candidate.exists() else file_candidate
    model_path = str(model_root_path / model_name / "gmflow_latest.pt")

    if net_iter_num > 0:
        if network_backend == "tensorrt":
            if device.type != "cuda":
                raise RuntimeError("TensorRT backend requires CUDA.")
            if tensorrt_engine_path is None:
                tensorrt_engine_path = model_root_path / model_name / "pyloreg_flow_b2_480x480_fp16_trt108.engine"
            if not Path(tensorrt_engine_path).is_file():
                raise FileNotFoundError(
                    "TensorRT engine not found. Pass tensorrt_engine_path explicitly or rebuild the default "
                    f"engine at: {tensorrt_engine_path}"
                )
            trt_runner = _TensorRTFlowRunner(tensorrt_engine_path, device=device)
            print("[Model] TensorRT engine:", tensorrt_engine_path)
            print("[Model] TensorRT expected inputs:", trt_runner.input_shapes)
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
    else:
        print("[Model] Skipped: net_iter_num=0 (rigid-only mode)")

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
    # Containers for cumulative mask/flow
    # IMPORTANT:
    #   - img_stack is still warped every registration iteration so templates/refinement work.
    #   - raw_stack is NOT warped inside the iterations anymore.
    #   - flow_stack stores the composed/cumulative transform from original raw frame
    #     to the current corrected coordinate system. At the end, raw_stack is warped once.
    # =========================================================
    stack_path = Path(stack_save_path)
    folder = stack_path.parent
    stem = stack_path.stem

    # We always need cumulative flow for the final one-shot raw warp,
    # even when save_mask_flow=False. Quantized storage reduces memory.
    flow_stack = np.empty((T, H, W, 2), dtype=fq.np_dtype)
    zero_flow = fq.encode(np.zeros((1, H, W, 2), dtype=np.float32))[0]
    flow_stack[...] = zero_flow

    if save_mask_flow:
        mask_accum_stack = np.full((T, H, W), 255, dtype=np.uint8)
        mask_folder = folder.with_name(folder.name + "_mask")
        flow_folder = folder.with_name(folder.name + "_flow")
        mask_folder.mkdir(exist_ok=True)
        flow_folder.mkdir(exist_ok=True)
    else:
        mask_accum_stack = None
        mask_folder = None
        flow_folder = None

    # =========================================================
    # Global reference pool for robust template building
    # =========================================================
    # 注意：原版第一轮是直接 pool 全部帧平均，这里改成每一轮都重新做：
    # 当前 img_stack -> crop/stride 特征 -> 每帧 z-score/L2 norm -> consensus -> corr score -> top frames -> template
    ref_pool_size = min(T, int(robust_template_pool_size)) if bool(robust_template_each_iter) else min(T, 2000)
    pool_mode = "uniform"
    pool_seed = 0

    refine_step = 0.10  # kept only for fallback original behavior
    refine_cap = 0.40   # kept only for fallback original behavior

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

    def _center_crop_stride_np(stack_thw: np.ndarray, indices: np.ndarray, crop, stride: int):
        """
        Fast feature extraction for template-quality scoring.
        Return cropped/downsampled frames [B,h,w] as float32.
        """
        _, H_, W_ = stack_thw.shape
        if crop is None:
            y0, x0, c_h, c_w = 0, 0, H_, W_
        else:
            c = int(min(crop, H_, W_))
            y0 = (H_ - c) // 2
            x0 = (W_ - c) // 2
            c_h = c_w = c
        s_ = max(int(stride), 1)
        return stack_thw[indices, y0:y0 + c_h:s_, x0:x0 + c_w:s_].astype(np.float32, copy=False)

    def compute_robust_highcorr_template(
        stack_thw: np.ndarray,
        pool_idx_: np.ndarray,
        keep_ratio: float = 0.20,
        min_frames: int = 50,
        crop: int = 256,
        stride: int = 4,
        batch: int = 256,
        template_stat: str = "mean",
        use_gpu: bool = False,
        eps: float = 1e-6,
    ):
        """
        Compute one robust template from the CURRENT stack.

        Algorithm:
            1) uniformly sampled pool frames are represented by center-crop + stride features;
            2) each frame feature is z-scored, then L2-normalized;
            3) consensus = mean normalized feature of the whole pool;
            4) score(frame) = dot(feature, consensus), equivalent to correlation/cosine similarity;
            5) keep top high-correlation frames;
            6) build full-resolution template from these selected frames.

        Why this is faster:
            - correlation is computed on crop/stride features, e.g. 256x256 stride 4 -> 64x64 = 4096 dims;
            - only one consensus vector is used, not all-pair correlations O(N^2);
            - full-resolution averaging is done only for selected top frames.
        """
        assert stack_thw.ndim == 3, f"Expected stack [T,H,W], got {stack_thw.shape}"
        T_, H_, W_ = stack_thw.shape

        pool_idx_ = np.asarray(pool_idx_, dtype=np.int64)
        pool_idx_ = pool_idx_[(pool_idx_ >= 0) & (pool_idx_ < T_)]
        pool_idx_ = np.unique(pool_idx_)
        if len(pool_idx_) == 0:
            raise ValueError("pool_idx_ is empty; cannot compute robust template.")

        batch = max(int(batch), 1)
        keep_ratio = float(np.clip(keep_ratio, 0.001, 1.0))
        k = int(np.ceil(len(pool_idx_) * keep_ratio))
        k = max(k, int(min_frames))
        k = min(k, len(pool_idx_))

        # ---------------- CPU implementation: stable and memory-friendly ----------------
        if not bool(use_gpu):
            sum_vec = None
            n_total = 0

            for st in range(0, len(pool_idx_), batch):
                ed = min(st + batch, len(pool_idx_))
                fr = _center_crop_stride_np(stack_thw, pool_idx_[st:ed], crop=crop, stride=stride)
                B = fr.shape[0]
                feat = fr.reshape(B, -1)
                feat = feat - feat.mean(axis=1, keepdims=True)
                feat = feat / (feat.std(axis=1, keepdims=True) + eps)
                feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + eps)

                if sum_vec is None:
                    sum_vec = feat.sum(axis=0, dtype=np.float64)
                else:
                    sum_vec += feat.sum(axis=0, dtype=np.float64)
                n_total += B

            consensus = (sum_vec / max(n_total, 1)).astype(np.float32, copy=False)
            consensus = consensus / (np.linalg.norm(consensus) + eps)

            scores = np.empty((len(pool_idx_),), dtype=np.float32)
            for st in range(0, len(pool_idx_), batch):
                ed = min(st + batch, len(pool_idx_))
                fr = _center_crop_stride_np(stack_thw, pool_idx_[st:ed], crop=crop, stride=stride)
                B = fr.shape[0]
                feat = fr.reshape(B, -1)
                feat = feat - feat.mean(axis=1, keepdims=True)
                feat = feat / (feat.std(axis=1, keepdims=True) + eps)
                feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + eps)
                scores[st:ed] = feat @ consensus

        # ---------------- GPU implementation: optional; useful if crop is larger ----------------
        else:
            dev_ = device
            sum_vec_t = None
            n_total = 0
            for st in range(0, len(pool_idx_), batch):
                ed = min(st + batch, len(pool_idx_))
                fr_np = _center_crop_stride_np(stack_thw, pool_idx_[st:ed], crop=crop, stride=stride)
                feat = torch.from_numpy(fr_np.reshape(fr_np.shape[0], -1)).to(dev_, non_blocking=True)
                feat = feat - feat.mean(dim=1, keepdim=True)
                feat = feat / (feat.std(dim=1, keepdim=True) + eps)
                feat = feat / (torch.linalg.norm(feat, dim=1, keepdim=True) + eps)

                cur_sum = feat.sum(dim=0, dtype=torch.float64)
                sum_vec_t = cur_sum if sum_vec_t is None else (sum_vec_t + cur_sum)
                n_total += feat.shape[0]
                del feat, cur_sum

            consensus_t = (sum_vec_t / max(n_total, 1)).float()
            consensus_t = consensus_t / (torch.linalg.norm(consensus_t) + eps)

            scores = np.empty((len(pool_idx_),), dtype=np.float32)
            for st in range(0, len(pool_idx_), batch):
                ed = min(st + batch, len(pool_idx_))
                fr_np = _center_crop_stride_np(stack_thw, pool_idx_[st:ed], crop=crop, stride=stride)
                feat = torch.from_numpy(fr_np.reshape(fr_np.shape[0], -1)).to(dev_, non_blocking=True)
                feat = feat - feat.mean(dim=1, keepdim=True)
                feat = feat / (feat.std(dim=1, keepdim=True) + eps)
                feat = feat / (torch.linalg.norm(feat, dim=1, keepdim=True) + eps)
                scores[st:ed] = (feat @ consensus_t).detach().cpu().numpy().astype(np.float32, copy=False)
                del feat
            del consensus_t, sum_vec_t

        # Select top-k frames by correlation to consensus.
        top_local = np.argpartition(scores, len(scores) - k)[len(scores) - k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        selected_idx = pool_idx_[top_local].astype(np.int64, copy=False)

        # Build full-resolution template only from high-correlation frames.
        stat = str(template_stat).lower()
        if stat == "median":
            template = np.median(stack_thw[selected_idx], axis=0).astype(np.float32, copy=False)
        elif stat == "mean":
            template = stack_thw[selected_idx].mean(axis=0).astype(np.float32, copy=False)
        else:
            raise ValueError(f"robust_template_stat must be 'mean' or 'median', got {template_stat}")

        info = {
            "pool_idx": pool_idx_,
            "selected_idx": selected_idx,
            "scores": scores,
            "selected_scores": scores[top_local],
            "k": int(k),
            "score_median": float(np.median(scores)),
            "score_p75": float(np.percentile(scores, 75)),
            "score_p90": float(np.percentile(scores, 90)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "selected_score_min": float(scores[top_local].min()),
            "selected_score_median": float(np.median(scores[top_local])),
            "selected_score_max": float(scores[top_local].max()),
        }
        return template, selected_idx, info

    pool_idx = make_pool_indices(T, ref_pool_size, mode=pool_mode, seed=pool_seed)
    print(f"[RefPool] mode={pool_mode} size={len(pool_idx)} / T={T} | robust_template_each_iter={bool(robust_template_each_iter)}")

    if bool(save_robust_template_qc):
        robust_tpl_dir = folder.with_name(folder.name + "_robust_template")
        robust_tpl_dir.mkdir(exist_ok=True, parents=True)
    else:
        robust_tpl_dir = None

    corr_score_prev = None
    motion_score_prev = None

    # =========================================================
    # Total iterations: Iter1 rigid + net_iter_num NN rounds
    # =========================================================
    total_iters = 1 + net_iter_num
    print(f"[Iters] total={total_iters} (Iter1 rigid + net_iter_num={net_iter_num} network rounds)")

    # =========================================================
    # Main loop
    # =========================================================
    with torch.inference_mode():
        for it in range(total_iters):
            print(f"\n[Iter {it+1}/{total_iters}] ...")

            # ---------------- robust template select/build ----------------
            if bool(robust_template_each_iter):
                # 每一轮都从当前 img_stack 重新计算：
                # high-correlation frames -> robust template。
                # Iter1 用原始 img_stack；Iter2/3/... 用上一轮 warp/refine 后的 img_stack。
                global_template, sel_idx, tpl_info = compute_robust_highcorr_template(
                    img_stack,
                    pool_idx_=pool_idx,
                    keep_ratio=robust_template_keep_ratio,
                    min_frames=robust_template_min_frames,
                    crop=robust_template_crop,
                    stride=robust_template_stride,
                    batch=robust_template_batch,
                    template_stat=robust_template_stat,
                    use_gpu=robust_template_use_gpu,
                    eps=robust_template_eps,
                )
                print(
                    f"[TemplateSelect] Iter{it+1}: robust high-corr template | "
                    f"selected={len(sel_idx)}/{len(pool_idx)} | "
                    f"score median={tpl_info['score_median']:.4f}, "
                    f"p90={tpl_info['score_p90']:.4f}, "
                    f"p95={tpl_info['score_p95']:.4f}, "
                    f"selected min/med/max="
                    f"{tpl_info['selected_score_min']:.4f}/"
                    f"{tpl_info['selected_score_median']:.4f}/"
                    f"{tpl_info['selected_score_max']:.4f}"
                )
            else:
                # Original behavior, kept as fallback.
                if corr_score_prev is None:
                    sel_idx = pool_idx
                    print(f"[TemplateSelect] Iter{it+1}: init from ALL pool frames ({len(sel_idx)})")
                else:
                    keep_ratio = float(np.clip(refine_step * it, 0.0, refine_cap))
                    P = len(pool_idx)
                    k = max(int(np.ceil(P * keep_ratio)), 10)
                    k = min(k, P)

                    q = corr_score_prev[pool_idx].astype(np.float32, copy=False)
                    thr = np.partition(q, P - k)[P - k]
                    sel_idx = pool_idx[q >= thr]
                    print(f"[TemplateSelect] Iter{it+1}: top {keep_ratio*100:.1f}% in pool => {len(sel_idx)} frames")

                global_template = img_stack[sel_idx].mean(axis=0).astype(np.float32, copy=False)
                tpl_info = None

            global_template_t = torch.from_numpy(global_template).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            global_template_dev = global_template_t.to(device, non_blocking=True)

            if bool(save_templates):
                tpl_path, tpl_writer = tpl_saver.open(it)
                tpl_writer.write(tpl_saver.cast(global_template), photometric="minisblack")
                tpl_writer.close()
                print(f"[Template] Saved global template → {tpl_path}")

            if bool(save_robust_template_qc) and robust_tpl_dir is not None:
                # 保存每一轮模板、选中帧和 correlation scores，方便检查模板是否被坏帧污染。
                robust_template_path = robust_tpl_dir / f"{stem}_robust_template_iter{it+1:02d}.tif"
                if template_save_dtype == "float32":
                    tiff.imwrite(str(robust_template_path), global_template.astype(np.float32, copy=False))
                else:
                    tiff.imwrite(str(robust_template_path), np.clip(global_template * 65535.0, 0, 65535).astype(np.uint16))

                np.save(str(robust_tpl_dir / f"{stem}_robust_selected_indices_iter{it+1:02d}.npy"), sel_idx)
                if tpl_info is not None:
                    np.savez_compressed(
                        str(robust_tpl_dir / f"{stem}_robust_scores_iter{it+1:02d}.npz"),
                        pool_idx=tpl_info["pool_idx"],
                        selected_idx=tpl_info["selected_idx"],
                        scores=tpl_info["scores"],
                        selected_scores=tpl_info["selected_scores"],
                    )

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
                        return global_template_dev.expand(B, 1, H, W).contiguous()

                def _rigid_register_to_ref(img_batch, ref_batch):
                    """
                    Return warped_img, dy_apply, dx_apply, mag, corr_b, ok.
                    raw_stack is intentionally not warped here.
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

                    warped_img = _warp_translation_batched(
                        img_batch, dy_apply, dx_apply,
                        mode=str(warp_mode), padding_mode="reflection", align_corners=True
                    )
                    return warped_img, dy_apply, dx_apply, mag, corr_b, ok

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
                    warped_img, dy_apply, dx_apply, mag, corr_b, ok = _rigid_register_to_ref(
                        img_batch, ref_batch
                    )

                    # ----- fallback (prev_with_fallback only) -----
                    if policy == "prev_with_fallback":
                        fb = (corr_b < float(iter1_fallback_corr_thr))
                        if torch.any(fb):
                            ref_anchor = _get_anchor_template_b1hw(B)
                            warped_img2, dy2, dx2, mag2, corr2, ok2 = _rigid_register_to_ref(
                                img_batch, ref_anchor
                            )

                            # choose per-item better result (only for fb items)
                            better = (corr2 > corr_b)
                            choose2 = (fb & better)

                            if torch.any(choose2):
                                choose2_v = choose2.view(B, 1, 1, 1)
                                warped_img = torch.where(choose2_v, warped_img2, warped_img)
                                dy_apply = torch.where(choose2, dy2, dy_apply)
                                dx_apply = torch.where(choose2, dx2, dx_apply)
                                mag = torch.where(choose2, mag2, mag)
                                corr_b = torch.where(choose2, corr2, corr_b)
                                ok = torch.where(choose2, ok2, ok)

                            del ref_anchor, warped_img2, dy2, dx2, mag2, corr2, ok2, better, choose2

                    # ----- write back -----
                    # Only img_stack is warped during iterative registration.
                    # raw_stack remains original and will be warped once at the end.
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

                    # ----- cumulative flow update -----
                    # IMPORTANT about sign:
                    # _warp_translation_batched() internally samples image at p - shift,
                    # while the dense-flow warper in this file uses warped(p)=image(p+flow(p)).
                    # Therefore the equivalent flow field for this rigid correction is -shift.
                    # Iter1 has no previous non-zero transform, so cumulative flow is just this rigid flow.
                    flow_stack[idx] = _encode_constant_translation_flow(-dy_apply, -dx_apply, H, W)

                    del img_batch, ref_batch
                    del warped_img, dy_apply, dx_apply, mag, corr_b, ok

                # =========================================================
                # save intermediate result after Iter1
                # =========================================================
                if bool(save_iter1_stack):
                    final_path = Path(stack_save_path)
                    if iter1_save_path is None:
                        iter1_path = final_path.with_name(final_path.stem + "_iter1_rigid" + final_path.suffix)
                    else:
                        iter1_path = Path(iter1_save_path)

                    # raw_stack is intentionally untouched until the final one-shot warp.
                    # Therefore this optional QC save writes the iter1-corrected img_stack,
                    # not a partially warped raw_stack.
                    inter_img_u16 = (img_stack * (vmax_img - vmin_img) + vmin_img).clip(0, 65535).astype("uint16")
                    tiff.imwrite(str(iter1_path), inter_img_u16)
                    print(f"[Saved] Iter1 rigid img_stack QC → {iter1_path}")

                    if bool(save_iter1_img_stack):
                        iter1_img_path = iter1_path.with_name(iter1_path.stem + "_IMG" + iter1_path.suffix)
                        tiff.imwrite(str(iter1_img_path), inter_img_u16)
                        print(f"[Saved] Iter1 rigid img_stack copy → {iter1_img_path}")

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

                    GT = global_template_dev.expand(B, 1, H, W).contiguous()

                    # corr score for template refine
                    img_q = _center_crop_stride_2d(img_batch, crop=corr_crop, stride=corr_stride)
                    gt_q  = _center_crop_stride_2d(GT,        crop=corr_crop, stride=corr_stride)
                    corr_b = batch_ncc_corr(img_q, gt_q).float()
                    corr_score_this[start:end] = corr_b.detach().cpu().numpy().astype(np.float32, copy=False)

                    img_p, pad = pad_to_multiple(img_batch, 16)
                    GT_p, _ = pad_to_multiple(GT, 16)

                    if trt_runner is not None:
                        flow_b2hw = trt_runner(GT_p, img_p)
                    elif amp_enabled:
                        with torch.cuda.amp.autocast(True):
                            out = model(GT_p, img_p, use_feature_num)
                        flow_b2hw = unpad(out["flow_preds"][-1], pad)  # [B,2,H,W]
                    else:
                        out = model(GT_p, img_p, use_feature_num)
                        flow_b2hw = unpad(out["flow_preds"][-1], pad)  # [B,2,H,W]
                    flow_bhw2 = flow_b2hw.permute(0, 2, 3, 1).contiguous()  # [B,H,W,2]

                    if str(motion_metric) == "absmean":
                        score_b = flow_b2hw.abs().mean(dim=(1, 2, 3))
                    else:
                        score_b = torch.sqrt(flow_b2hw[:, 0].pow(2) + flow_b2hw[:, 1].pow(2)).mean(dim=(1, 2))
                    motion_score_this[start:end] = score_b.detach().float().cpu().numpy()

                    # Warp only img_stack during iterative registration.
                    # raw_stack remains untouched and will be warped once after all flows are composed.
                    if save_mask_flow:
                        warped_img_b1hw, mask_b1hw = _warp_stack_once_with_flow(
                            img_batch, flow_bhw2,
                            mode=str(warp_mode),
                            padding_mode="reflection",
                            align_corners=True,
                            return_valid_mask=True,
                        )
                    else:
                        warped_img_b1hw = _warp_stack_once_with_flow(
                            img_batch, flow_bhw2,
                            mode=str(warp_mode),
                            padding_mode="reflection",
                            align_corners=True,
                            return_valid_mask=False,
                        )
                        mask_b1hw = None

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

                    # ----- cumulative flow update -----
                    # Compose previous cumulative flow with this iteration's new flow.
                    # This keeps a single transform that maps original raw -> final corrected coordinates.
                    prev_np = fq.decode(flow_stack[idx])
                    prev_t = torch.from_numpy(prev_np).to(device, non_blocking=True)
                    composed = compose_flow_batch(prev_t, flow_bhw2, grid_cache=grid_cache)
                    composed_np = composed.detach().cpu().numpy().astype(np.float32, copy=False)
                    flow_stack[idx] = fq.encode(composed_np)

                    for _name in (
                        "img_batch", "GT", "img_p", "GT_p", "out", "flow_b2hw", "flow_bhw2",
                        "warped_img_b1hw", "mask_b1hw", "img_q", "gt_q", "corr_b", "prev_t", "composed",
                    ):
                        if _name in locals():
                            del locals()[_name]

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
    # Final one-shot warp of ORIGINAL raw_stack using composed/cumulative flow
    # =========================================================
    print("\n[FinalWarp] Applying composed cumulative flow to raw_stack ONCE ...")
    final_stack = np.empty_like(raw_stack, dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(
            range(0, T, int(batch_size)),
            desc="🎯 Final one-shot raw warp",
            ncols=100, unit="batch",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"
        ):
            end = min(start + int(batch_size), T)
            idx = slice(start, end)

            raw_batch = torch.from_numpy(raw_stack[idx]).unsqueeze(1).to(device, non_blocking=True)
            flow_np = fq.decode(flow_stack[idx])
            flow_bhw2 = torch.from_numpy(flow_np).to(device, non_blocking=True)

            warped_raw_b1hw = _warp_stack_once_with_flow(
                raw_batch, flow_bhw2,
                mode=str(warp_mode),
                padding_mode="reflection",
                align_corners=True,
                return_valid_mask=False,
            )
            final_stack[idx] = warped_raw_b1hw.squeeze(1).detach().cpu().numpy()

            del raw_batch, flow_bhw2, warped_raw_b1hw

    # =========================================================
    # save mask/flow
    # =========================================================
    if save_mask_flow:
        scale_label = ("%g" % float(fq.flow_scale)).replace(".", "p")
        offset_label = ("%g" % float(fq.flow_offset)).replace(".", "p")
        flow_label = f"{fq.storage_dtype}_x{scale_label}_off{offset_label}"
        tiff.imwrite(mask_folder / f"{stem}_mask_final.tif", mask_accum_stack)
        tiff.imwrite(flow_folder / f"{stem}_flow_u_{flow_label}.tif", flow_stack[..., 0])
        tiff.imwrite(flow_folder / f"{stem}_flow_v_{flow_label}.tif", flow_stack[..., 1])
        print("[Saved] Mask & cumulative Flow")

    # =========================================================
    # save final stack
    # =========================================================
    final = (final_stack * (vmax_raw - vmin_raw) + vmin_raw).clip(0, 65535).astype("uint16")
    tiff.imwrite(stack_save_path, final)
    print("\n[Done] Demotion saved →", stack_save_path)
    return stack_save_path



# =========================================================
# 8) Backward-compatible aliases
# =========================================================

# Preferred name for new code.
demotion_PyLoReg_infer2stack = demotion_PyLoReg_infer2stack_one_shot_raw_warp

# Keep old names so existing pipeline calls still work.
# NOTE: both now use the cleaned one-shot-raw-warp behavior.
demotion_PyLoReg_infer2stack_less_save_acc_v4 = demotion_PyLoReg_infer2stack_one_shot_raw_warp
demotion_PyLoReg_infer2stack_less_save_acc_v3 = demotion_PyLoReg_infer2stack_one_shot_raw_warp
