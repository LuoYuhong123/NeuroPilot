#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import time
import random
import datetime
import warnings
from pathlib import Path

import numpy as np
import yaml
import tifffile as tiff

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import socket
import torch.multiprocessing as mp
from skimage import io

from .network import Network_3D_Unet
from .data_process import trainset
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")


TIF_MANIFEST_NAME = "tif_manifest.json"


def _load_tif_paths_from_manifest(datasets_path):
    root = Path(str(datasets_path)).expanduser()
    manifest_path = root if root.is_file() else root / TIF_MANIFEST_NAME
    if not manifest_path.is_file():
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    raw_entries = payload.get("tifs") or payload.get("selected_tifs") or []
    tif_files = []
    for entry in raw_entries:
        raw_path = entry.get("path") if isinstance(entry, dict) else entry
        if not raw_path:
            continue
        candidate = Path(str(raw_path)).expanduser()
        if candidate.is_file() and candidate.suffix.lower() in (".tif", ".tiff"):
            tif_files.append(str(candidate))
    return sorted(tif_files, key=lambda p: os.path.basename(p).lower())


def _resolve_training_tif_files(datasets_path):
    manifest_tifs = _load_tif_paths_from_manifest(datasets_path)
    if manifest_tifs:
        return manifest_tifs
    return sorted(
        glob.glob(os.path.join(datasets_path, '*.tif'))
        + glob.glob(os.path.join(datasets_path, '*.tiff')),
        key=lambda p: os.path.basename(p).lower(),
    )


def _select_ddp_backend() -> str:
    if dist.is_available():
        # Avoid probing NCCL on Windows and silence the PyTorch warning that
        # appears when the current wheel was built without NCCL support.
        if os.name != "nt" and hasattr(dist, "is_nccl_available"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="PyTorch is not compiled with NCCL support",
                )
                if dist.is_nccl_available():
                    return "nccl"
        if hasattr(dist, "is_gloo_available") and dist.is_gloo_available():
            return "gloo"
    return "gloo"



def _find_free_port():
    """找一个空闲端口给 DDP 用，避免端口冲突。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ddp_worker_train_deepcad(
    rank: int,
    world_size: int,
    gpu_list: list,
    train_dict: dict,
    master_addr: str,
    master_port: int,
):
    """
    子进程入口：每个 rank 绑定一张 GPU，init_process_group，然后跑 training_class.run()
    """
    # 1) 绑定当前进程可见 GPU（关键）
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))

    # 2) rank 对应到 visible device index（0..world_size-1）
    local_rank = rank
    torch.cuda.set_device(local_rank)

    # 3) 初始化 DDP 通信环境
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)

    backend = _select_ddp_backend()
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )

    # 4) 把 DDP runtime 信息塞进 train_dict，让 training_class 用
    #    （你要的是“输入输出不变”，所以 train_dict 仍然是一个 dict）
    train_dict = dict(train_dict)  # copy
    train_dict["_ddp_enable"] = True
    train_dict["_ddp_rank"] = rank
    train_dict["_ddp_world_size"] = world_size
    train_dict["_ddp_local_rank"] = local_rank

    # 5) 运行训练（DDP版 training_class）
    tc = training_class(train_dict)
    tc.run()

    # 6) 清理
    dist.barrier()
    dist.destroy_process_group()


def train_deepcad(
    datasets_path,
    pth_dir="./pth_deepcad",
    pth_name=None,
    n_epochs=1,
    # ---- formerly CONFIG ----
    patch_xy=192,
    patch_t=512,
    overlap_factor=0.5,
    gpu="0,1",
    num_workers=0,
    fmap=8,
    scale_factor=1,
    # ---- others ----
    train_datasets_size=1000,
    select_img_num=100000,
    lr=1e-4,
    b1=0.5,
    b2=0.999,
    sample_mode='T',
    visualize_images_per_epoch=False,
    save_test_images_per_epoch=False,
    batch_size=4,
    # ---- ddp knobs ----
    use_ddp=True,                 # ✅ 新增：是否启用 DDP
    master_addr="127.0.0.1",      # 单机默认即可
    master_port=None,             # None -> 自动找空闲端口
):
    """
    ✅ 保持你原来的函数调用方式不变：
       train_deepcad(...) 直接跑
    ✅ 如果 use_ddp=True 且 gpu 包含多张卡，则内部 spawn 多进程 DDP
    ✅ 输出/保存目录行为保持：只由 rank0 写文件
    """
    # -------------------------
    # 1) 组装 train_dict（保持你原逻辑）
    # -------------------------
    train_dict = {
        "batch_size": batch_size,
        "sample_mode": sample_mode,
        "patch_x": patch_xy,
        "patch_y": patch_xy,
        "patch_t": patch_t,
        "overlap_factor": overlap_factor,
        "scale_factor": scale_factor,
        "select_img_num": select_img_num,
        "train_datasets_size": train_datasets_size,
        "datasets_path": datasets_path,

        "pth_dir": pth_dir,
        "pth_name": pth_name,
        "n_epochs": n_epochs,

        "lr": lr,
        "b1": b1,
        "b2": b2,

        "fmap": fmap,
        "GPU": gpu,                 # 仍然保留
        "num_workers": num_workers,

        "visualize_images_per_epoch": visualize_images_per_epoch,
        "save_test_images_per_epoch": save_test_images_per_epoch,
    }

    # -------------------------
    # 2) 解析 GPU 列表
    # -------------------------
    gpu_list = [int(x) for x in str(gpu).split(",") if str(x).strip() != ""]
    world_size = len(gpu_list)

    # -------------------------
    # 3) 单卡 / 不启用 DDP -> 走你原来的 DP/单卡逻辑
    # -------------------------
    if (not use_ddp) or world_size <= 1:
        tc = training_class(train_dict)
        tc.run()
        return tc.pth_path

    # -------------------------
    # 4) 多卡 DDP：spawn 多进程
    # -------------------------
    if master_port is None:
        master_port = _find_free_port()

    # 为了避免 Windows/交互环境问题，推荐用 spawn
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        _ddp_worker_train_deepcad,
        args=(world_size, gpu_list, train_dict, master_addr, master_port),
        nprocs=world_size,
        join=True,
    )

    # rank0 会写 pth_path，路径规则与 training_class.prepare_file 一致
    # 我们在这里重新构造一次得到 pth_path（不需要拿到 tc 实例）
    datasets_name = datasets_path.rstrip("/").split("/")[-1]
    pth_name_final = datasets_name if pth_name is None else pth_name
    pth_path = os.path.join(pth_dir, pth_name_final)
    return pth_path


# =========================
#   Utility Functions
# =========================

def split_patch_XY(raw_patch: torch.Tensor):
    assert raw_patch.ndim == 5, "输入必须是 5 维 (B, C, T, X, Y)"
    mode = 'row'
    if mode == 'row':
        patch1 = raw_patch[:, :, :, :, 0::2]
        patch2 = raw_patch[:, :, :, :, 1::2]
    else:
        patch1 = raw_patch[:, :, :, 0::2, :]
        patch2 = raw_patch[:, :, :, 1::2, :]
    return patch1, patch2


def split_patch_T(raw_patch: torch.Tensor):
    assert raw_patch.ndim == 5, "输入必须是 5 维 (B, C, T, X, Y)"
    _, _, T, _, _ = raw_patch.shape
    assert T >= 2, "时间维 T 必须 ≥ 2 才能拆分"
    patch1 = raw_patch[:, :, 0::2, :, :]
    patch2 = raw_patch[:, :, 1::2, :, :]
    return patch1, patch2


def save_tiff_image(image_tensor, image_path):
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.detach().cpu().numpy()
    elif isinstance(image_tensor, np.ndarray):
        image = np.squeeze(image_tensor)
    else:
        raise TypeError(f"Unsupported image type: {type(image_tensor)}")

    save_tiff_path = image_path.replace('.png', '.tif')
    folder = os.path.dirname(save_tiff_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    io.imsave(save_tiff_path, image.astype(np.float32))


def _fast_stack_meta(tif_path: str):
    try:
        with tiff.TiffFile(tif_path) as tf:
            T = len(tf.pages)
            sh = tf.pages[0].shape
            if len(sh) >= 2:
                H, W = int(sh[-2]), int(sh[-1])
            else:
                arr0 = tf.pages[0].asarray()
                H, W = int(arr0.shape[-2]), int(arr0.shape[-1])
            return int(T), int(H), int(W)
    except Exception:
        arr = tiff.imread(tif_path)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected [T,H,W] tif stack, got {arr.shape} for {tif_path}")
        T, H, W = arr.shape
        return int(T), int(H), int(W)


def _ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    return dist.get_rank() if _ddp_is_initialized() else 0


def _get_world_size():
    return dist.get_world_size() if _ddp_is_initialized() else 1


def _is_main_process():
    return _get_rank() == 0


def seed_everything(base_seed: int, rank: int):
    # 让每个 rank 的随机序列不同，但可复现
    seed = int(base_seed) + int(rank) * 100003
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
#   Training Class (DDP)
# =========================

class training_class:
    """
    DDP 版本：输入输出尽量保持一致，仅把模型/数据并行化。
    """

    def __init__(self, params_dict):
        # ---- 基本配置 ----
        self.datasets_path = ''
        self.output_dir = None
        self.pth_dir = './pth'
        self.pth_name = None

        # ---- 训练参数 ----
        self.n_epochs = 20
        self.fmap = 16
        self.batch_size = 1
        self.lr = 1e-5
        self.b1 = 0.5
        self.b2 = 0.999

        # GPU 相关
        self.GPU = '0'   # DDP 下建议不要依赖这个；torchrun 会分配 local_rank
        self.ngpu = 1

        # patch / 采样相关
        self.sample_mode = 'XY'  # 'XY' 或 'T'
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.overlap_factor = 0.5
        self.gap_x = 115
        self.gap_y = 115
        self.gap_t = 115

        self.scale_factor = 1.0
        self.train_datasets_size = 2000
        self.select_img_num = 1000
        self.num_workers = 4

        # speed knobs
        self.use_amp = False
        self.cudnn_benchmark = False
        self.allow_tf32 = False
        self.log_interval = 20
        self.save_vis_every_stack = True
        self.save_vis_every_n_epoch = 1

        self.eta_ema_alpha = 0.05

        # reproducibility
        self.seed = 1234

        # 预训练权重
        if self.sample_mode == 'XY':
            self.pretrained_pth = 'pretrained_pth//deepcad_XY//Self_SL_30.pth'
        elif self.sample_mode == 'T':
            self.pretrained_pth = 'pretrained_pth//deepcad_T//Self_SL_30.pth'
        else:
            self.pretrained_pth = None

        # 内部变量
        self.img_name_list = []
        self.img_dict = {}
        self.stack_num = 0

        self._preprocess_logged = False
        self.local_model = None
        self.pth_path = None

        # cache
        self._stack_meta = {}
        self._valid_tif_files = []

        # DDP runtime
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = None

        self.set_params(params_dict)

    # -------------------------
    #  对外主入口
    # -------------------------
    def run(self):
        self._ddp_setup()

        if _is_main_process():
            print("\033[95m[Step 1] Preparing files...\033[0m")
        self.prepare_file()

        if _is_main_process():
            print("\033[94m[Step 2] Scanning stacks (FAST meta) + preprocessing coords...\033[0m")
        self._scan_stacks_and_cache_meta()
        self.train_preprocess_lessMemoryMulStacks()

        if _is_main_process():
            print("\033[96m[Step 3] Saving training parameters to YAML...\033[0m")
        if _is_main_process():
            self.save_yaml_train()

        if _is_main_process():
            print("\033[92m[Step 4] Initializing network...\033[0m")
        self.initialize_network()

        if _is_main_process():
            print("\033[93m[Step 5] Wrapping model with DDP...\033[0m")
        self.distribute_GPU_ddp()

        if _is_main_process():
            print("\033[91m[Step 6] Starting training...\033[0m")
        self.train()

        self._ddp_cleanup()

    # -------------------------
    #  DDP init / cleanup
    # -------------------------
    def _ddp_setup(self):
        if self.world_size > 1 and not _ddp_is_initialized():
            dist.init_process_group(backend=_select_ddp_backend(), init_method="env://")

        self.rank = _get_rank()
        self.world_size = _get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        # backend knobs
        torch.backends.cudnn.benchmark = bool(self.cudnn_benchmark)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(self.allow_tf32)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(self.allow_tf32)

        seed_everything(self.seed, self.rank)

        if _is_main_process():
            print(
                f"[DDP] backend={_select_ddp_backend()} "
                f"world_size={self.world_size}, rank={self.rank}, "
                f"local_rank={self.local_rank}, device={self.device}"
            )

    def _ddp_cleanup(self):
        if _ddp_is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    # -------------------------
    #  配置与路径
    # -------------------------
    def set_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))

        # DDP 下 ngpu/world_size 来自 torchrun；这里仅保留兼容打印
        self.ngpu = str(self.GPU).count(',') + 1

        if _is_main_process():
            print('\033[1;31mTraining parameters -----> \033[0m')
            print(self.__dict__)

    def prepare_file(self):
        if self.datasets_path.endswith('/'):
            self.datasets_name = self.datasets_path.rstrip('/').split('/')[-1]
        else:
            self.datasets_name = self.datasets_path.split('/')[-1]

        pth_name = self.datasets_name if self.pth_name is None else self.pth_name

        self.pth_path = os.path.join(self.pth_dir, pth_name)
        if _is_main_process():
            os.makedirs(self.pth_path, exist_ok=True)
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------
    #  模型与 DDP
    # -------------------------
    def initialize_network(self):
        self.local_model = Network_3D_Unet(
            in_channels=1,
            out_channels=1,
            f_maps=self.fmap,
            final_sigmoid=False
        )

    def distribute_GPU_ddp(self):
        if torch.cuda.is_available():
            self.local_model = self.local_model.to(self.device)

        # load pretrained on ALL ranks (must be same weights)
        if self.pretrained_pth is not None and os.path.exists(self.pretrained_pth):
            state = torch.load(self.pretrained_pth, map_location=self.device)
            self.local_model.load_state_dict(state)
            if _is_main_process():
                print(f"\033[1;31m{'!' * 60}\033[0m")
                print(f"\033[1;31m[MODEL LOADED] {self.pretrained_pth}\033[0m")
                print(f"\033[1;31m{'!' * 60}\033[0m")

        # Wrap with DDP if multi-GPU
        if self.world_size > 1 and torch.cuda.is_available():
            self.local_model = nn.parallel.DistributedDataParallel(
                self.local_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,  # 用 GN/LN 更常见；BN 才需要 buffer 同步
                find_unused_parameters=False
            )
        else:
            # 单卡也保持接口一致
            pass

    # -------------------------
    #  FAST: 扫描 stack 元信息（只做一次）
    # -------------------------
    def _scan_stacks_and_cache_meta(self):
        patch_t2 = self.patch_t

        tif_files = _resolve_training_tif_files(self.datasets_path)

        if _is_main_process() and (not self._preprocess_logged):
            print(f"\033[1;31m[PATH] {self.datasets_path}\033[0m")
            print(f"\033[1;31m[FILES] {tif_files}\033[0m")
            self._preprocess_logged = True

        valid, meta = [], {}
        for tif_file in tif_files:
            T, H, W = _fast_stack_meta(tif_file)
            if T > patch_t2:
                valid.append(tif_file)
                meta[tif_file] = (int(T), int(H), int(W))

        self._valid_tif_files = valid
        self._stack_meta = meta
        self.stack_num = len(self._valid_tif_files)

        if self.stack_num == 0:
            raise RuntimeError("No valid tif/tiff stacks found for training.")

    # -------------------------
    #  预处理：采样 patch 坐标（每个 epoch 重采样）
    # -------------------------
    def train_preprocess_lessMemoryMulStacks(self):
        patch_t2 = self.patch_t

        self.img_name_list = []
        self.img_dict = {}

        valid_tif_files = self._valid_tif_files
        self.stack_num = len(valid_tif_files)
        if self.stack_num == 0:
            raise RuntimeError("No valid tif/tiff stacks found for training.")

        per_stack_datasize = self.train_datasets_size // self.stack_num

        # ✅ DDP：每个 rank 采样不同坐标（靠 seed_everything 已保证随机流不同）
        for im_dir in valid_tif_files:
            per_img_dict = {}
            im_name = os.path.basename(im_dir)
            im_name_no_ext = os.path.splitext(im_name)[0]

            self.img_name_list.append(im_name_no_ext)
            per_img_dict['img_path'] = im_dir

            T0, H0, W0 = self._stack_meta.get(im_dir, _fast_stack_meta(im_dir))

            whole_t = int(min(T0, self.select_img_num))
            whole_y = int(H0)
            whole_x = int(W0)

            coordinate_list = []
            for _ in range(per_stack_datasize):
                init_h = random.randint(0, whole_y - self.patch_y - 1)
                end_h = init_h + self.patch_y

                init_w = random.randint(0, whole_x - self.patch_x - 1)
                end_w = init_w + self.patch_x

                init_s = random.randint(0, whole_t - patch_t2 - 1)
                end_s = init_s + patch_t2

                coordinate_list.append(dict(
                    init_h=init_h, end_h=end_h,
                    init_w=init_w, end_w=end_w,
                    init_s=init_s, end_s=end_s
                ))

            per_img_dict['coordinate_list'] = coordinate_list
            self.img_dict[im_name_no_ext] = per_img_dict

    def save_yaml_train(self):
        yaml_name = os.path.join(self.pth_path, 'para.yaml')
        para = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                para[k] = v
            elif isinstance(v, (list, tuple, dict)):
                try:
                    yaml.safe_dump(v)
                    para[k] = v
                except Exception:
                    para[k] = repr(v)
            else:
                para[k] = repr(v)
        with open(yaml_name, 'w') as f:
            yaml.dump(para, f)

    # -------------------------
    #  训练主循环（DDP）
    # -------------------------
    def train(self):
        torch.set_grad_enabled(True)

        # get raw module for optimizer
        model_for_optim = self.local_model.module if isinstance(self.local_model, nn.parallel.DistributedDataParallel) else self.local_model

        optimizer_G = torch.optim.Adam(
            model_for_optim.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

        use_cuda = torch.cuda.is_available()

        L1_pixelwise = torch.nn.L1Loss().to(self.device) if use_cuda else torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss().to(self.device) if use_cuda else torch.nn.MSELoss()

        use_amp = bool(self.use_amp) and use_cuda
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        ema_step_time = None
        ema_alpha = float(self.eta_ema_alpha)

        time_start = time.time()

        for epoch in range(self.n_epochs):
            # 每个 epoch 重采样坐标（逻辑不变）
            self.train_preprocess_lessMemoryMulStacks()

            # ✅ epoch 总步数估计（每个 rank 自己算自己的数据量）
            epoch_total_steps = 0
            for img_name in self.img_name_list:
                coords = self.img_dict[img_name]["coordinate_list"]
                n_samples = len(coords)
                n_batches = int(np.ceil(n_samples / max(1, int(self.batch_size))))
                epoch_total_steps += max(1, n_batches)

            steps_done_in_epoch = 0
            prev_step_t = time.time()

            for img_name in self.img_name_list:
                per_img_dict = self.img_dict[img_name]
                im_dir = per_img_dict['img_path']
                coordinate_list = per_img_dict['coordinate_list']

                # 每个 rank 都读一次 stack（不改变 I/O 行为；要更快可做 memmap/共享，但你要求输入输出不变）
                noise_im = tiff.imread(im_dir).astype(np.float32)
                if noise_im.shape[0] > self.select_img_num:
                    noise_im = noise_im[:self.select_img_num, :, :]
                noise_im = noise_im - noise_im.mean()

                if _is_main_process():
                    print('Training image shape ::: ', noise_im.shape)

                train_data = trainset(noise_im, coordinate_list)

                # ✅ DDP sampler：把 coordinate_list 对应样本切给各 rank
                sampler = None
                if self.world_size > 1:
                    sampler = DistributedSampler(
                        train_data,
                        num_replicas=self.world_size,
                        rank=self.rank,
                        shuffle=True,
                        drop_last=False
                    )

                dl_kwargs = dict(
                    batch_size=self.batch_size,
                    shuffle=(sampler is None),  # 有 sampler 时不能再 shuffle
                    sampler=sampler,
                    num_workers=self.num_workers,
                )
                if use_cuda:
                    dl_kwargs["pin_memory"] = True
                if self.num_workers and int(self.num_workers) > 0:
                    dl_kwargs["persistent_workers"] = True
                    dl_kwargs["prefetch_factor"] = 4

                trainloader = DataLoader(train_data, **dl_kwargs)

                # DDP 习惯：每个 epoch 需要 set_epoch，保证 shuffle 不同
                if sampler is not None:
                    sampler.set_epoch(epoch)

                for iteration, raw_patch in enumerate(trainloader):
                    if self.sample_mode == 'XY':
                        input_, target = split_patch_XY(raw_patch)
                    elif self.sample_mode == 'T':
                        input_, target = split_patch_T(raw_patch)
                    else:
                        raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

                    input_ = input_.to(self.device, non_blocking=True) if use_cuda else input_
                    target = target.to(self.device, non_blocking=True) if use_cuda else target

                    optimizer_G.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            fake_B = self.local_model(input_)
                            L1_loss = L1_pixelwise(fake_B, target)
                            L2_loss = L2_pixelwise(fake_B, target)
                            Total_loss = 0.5 * L1_loss + 0.5 * L2_loss
                        scaler.scale(Total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()
                    else:
                        fake_B = self.local_model(input_)
                        L1_loss = L1_pixelwise(fake_B, target)
                        L2_loss = L2_pixelwise(fake_B, target)
                        Total_loss = 0.5 * L1_loss + 0.5 * L2_loss
                        Total_loss.backward()
                        optimizer_G.step()

                    # ===== ETA update (EMA over step time) =====
                    t_now = time.time()
                    step_time = t_now - prev_step_t
                    prev_step_t = t_now

                    if ema_step_time is None:
                        ema_step_time = step_time
                    else:
                        ema_step_time = (1.0 - ema_alpha) * ema_step_time + ema_alpha * step_time

                    steps_done_in_epoch += 1
                    steps_left_in_epoch = max(0, epoch_total_steps - steps_done_in_epoch)
                    time_left = datetime.timedelta(seconds=int(steps_left_in_epoch * ema_step_time))

                    # ✅ 只在 rank0 打 log（避免刷屏）
                    if _is_main_process() and self.log_interval and (iteration % int(self.log_interval) == 0):
                        time_cost = datetime.timedelta(seconds=int(time.time() - time_start))
                        print(
                            '\r\033[2K'
                            '\033[92m'
                            '[Denoise TRAIN][DDP] [Epoch %d/%d] [Batch %d/%d] '
                            '[Total loss: %.4f] [ETA: %s] [Time cost: %s]'
                            '\033[0m'
                            % (
                                epoch + 1, self.n_epochs,
                                steps_done_in_epoch, epoch_total_steps,
                                Total_loss.item(), time_left, time_cost
                            ),
                            end='',
                            flush=True
                        )

                # ✅ 保存模型/可视化：只让 rank0 做（否则多进程写同一路径会冲突）
                if _is_main_process():
                    print('\n', end=' ')
                    self.save_model(epoch)

                    if bool(self.save_vis_every_stack) and ((epoch + 1) % int(self.save_vis_every_n_epoch) == 0):
                        output_input_folder = os.path.join(self.pth_path, 'input')
                        os.makedirs(output_input_folder, exist_ok=True)
                        output_input_path = os.path.join(output_input_folder, f'{epoch}_{img_name}_input.tif')
                        save_tiff_image(input_, output_input_path)

                        output_target_folder = os.path.join(self.pth_path, 'target')
                        os.makedirs(output_target_folder, exist_ok=True)
                        output_target_path = os.path.join(output_target_folder, f'{epoch}_{img_name}_target.tif')
                        save_tiff_image(target, output_target_path)

                        output_pred_folder = os.path.join(self.pth_path, 'pred')
                        os.makedirs(output_pred_folder, exist_ok=True)
                        output_pred_path = os.path.join(output_pred_folder, f'{epoch}_{img_name}_pred.tif')
                        save_tiff_image(fake_B, output_pred_path)

            if _ddp_is_initialized():
                dist.barrier()

        if _is_main_process():
            print("\n")

    # -------------------------
    #  模型保存（DDP）
    # -------------------------
    def save_model(self, epoch: int):
        model_save_name = os.path.join(self.pth_path, f'Self_SL_{epoch + 1}.pth')
        if isinstance(self.local_model, nn.parallel.DistributedDataParallel):
            torch.save(self.local_model.module.state_dict(), model_save_name)
        elif isinstance(self.local_model, nn.DataParallel):
            torch.save(self.local_model.module.state_dict(), model_save_name)
        else:
            torch.save(self.local_model.state_dict(), model_save_name)
