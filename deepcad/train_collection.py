#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import time
import random
import datetime
import warnings
import functools

import numpy as np
import yaml
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage import io

from .network import Network_3D_Unet
from .data_process import trainset

warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")


def seed_everything(seed: int, deterministic: bool = False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int, base_seed: int, deterministic: bool = False):
    seed_everything(int(base_seed) + int(worker_id) + 1, bool(deterministic))


# =========================
#   Utility Functions
# =========================

def split_patch_XY(raw_patch: torch.Tensor):
    """
    将 5D patch (B, C, T, X, Y) 沿空间维度做隔行 / 隔列拆分
    返回 (patch1, patch2)，保证形状一致，仅 X 或 Y 尺度减半。
    """
    assert raw_patch.ndim == 5, "输入必须是 5 维 (B, C, T, X, Y)"
    mode = 'row'
    if mode == 'row':
        patch1 = raw_patch[:, :, :, :, 0::2]
        patch2 = raw_patch[:, :, :, :, 1::2]
    else:  # 'col'
        patch1 = raw_patch[:, :, :, 0::2, :]
        patch2 = raw_patch[:, :, :, 1::2, :]
    return patch1, patch2


def split_patch_T(raw_patch: torch.Tensor):
    """
    将 5D patch (B, C, T, X, Y) 沿时间维做隔帧拆分。
    """
    assert raw_patch.ndim == 5, "输入必须是 5 维 (B, C, T, X, Y)"
    _, _, T, _, _ = raw_patch.shape
    assert T >= 2, "时间维 T 必须 ≥ 2 才能拆分"
    patch1 = raw_patch[:, :, 0::2, :, :]
    patch2 = raw_patch[:, :, 1::2, :, :]
    return patch1, patch2


def split_patch_PixelDrop(raw_patch: torch.Tensor, drop_ratio: float = 0.5, drop_value: float = 0.0):
    """
    NEW MODE:
      input_  = raw_patch 随机把一部分像素置为 drop_value（默认 0）
      target  = raw_patch 自身（干净目标）
    适用于 5D patch: (B, C, T, X, Y)

    drop_ratio:
      - 0.0 -> 不丢
      - 0.5 -> 丢一半像素（随机伯努利 mask）
      - 1.0 -> 全丢（不建议）
    """
    assert raw_patch.ndim == 5, "输入必须是 5 维 (B, C, T, X, Y)"
    r = float(drop_ratio)
    if not (0.0 <= r < 1.0):
        raise ValueError(f"drop_ratio must be in [0.0, 1.0), got {drop_ratio}")

    target = raw_patch
    # keep probability
    keep_p = 1.0 - r

    # 生成 keep mask：True 表示保留像素
    # 注意：在 CPU 上做也行，后面会 cuda(non_blocking=True) 传上去
    mask = (torch.rand_like(raw_patch) < keep_p)

    # input = raw_patch，但随机丢弃像素置 0
    if drop_value == 0.0:
        input_ = raw_patch * mask.to(raw_patch.dtype)
    else:
        input_ = raw_patch.clone()
        input_[~mask] = drop_value

    return input_, target


def save_tiff_image(image_tensor, image_path):
    """
    将 torch.Tensor 或 numpy.ndarray 保存为 tif。
    自动把 '.png' 后缀替换为 '.tif'。
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.detach().cpu().numpy()
    elif isinstance(image_tensor, np.ndarray):
        image = np.squeeze(image_tensor)
    else:
        raise TypeError(
            f"Unsupported image type: {type(image_tensor)}. "
            "Expected torch.Tensor or np.ndarray."
        )

    save_tiff_path = image_path.replace('.png', '.tif')
    folder = os.path.dirname(save_tiff_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    io.imsave(save_tiff_path, image.astype(np.float32))


def _fast_stack_meta(tif_path: str):
    """
    FAST: 只读元数据，不整栈 imread。
    返回 (T, H, W)；若无法解析则回退到一次 imread（保正确性）。
    """
    try:
        with tiff.TiffFile(tif_path) as tf:
            # NOTE:
            # 有些 tif 文件把“每帧”拆成多个 page（例如 pages=1395，但实际 stack
            # 的时间维只有 T=698）。用 len(tf.pages) 会导致“时间索引越界”，从而生成空 patch。
            #
            # 因此优先用 tf.series[0].shape 来推断 (T,H,W)。
            if getattr(tf, "series", None) and len(tf.series) > 0:
                sh = tf.series[0].shape  # e.g. (T, H, W)
                if len(sh) == 3:
                    T, H, W = int(sh[0]), int(sh[1]), int(sh[2])
                    return T, H, W
                if len(sh) >= 2:
                    # 兜底：至少保证 H/W 正确
                    H, W = int(sh[-2]), int(sh[-1])
                    # 尽量取第一维作为 T；若不是 3D stack，则 fallback 到 pages/imread
                    T = int(sh[0]) if len(sh) >= 3 else None
                    if T is not None:
                        return T, H, W

            # fallback: pages 可能和真实 T 不一致，但仍可提供一个近似
            T = len(tf.pages)
            sh = tf.pages[0].shape
            if len(sh) >= 2:
                H, W = int(sh[-2]), int(sh[-1])
            else:
                arr0 = tf.pages[0].asarray()
                H, W = int(arr0.shape[-2]), int(arr0.shape[-1])
            return T, H, W
    except Exception:
        arr = tiff.imread(tif_path)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected [T,H,W] tif stack, got {arr.shape} for {tif_path}")
        T, H, W = arr.shape
        return int(T), int(H), int(W)


# =========================
#   Training Class
# =========================

class training_class:
    """
    3D U-Net 自监督降噪训练类（多 stack，低显存）。

    ✅ 新增训练模式：
      sample_mode = 'PIXELDROP'
        - input_  = raw_patch 随机丢弃一部分像素置 0（比例由 pixel_drop_ratio 控制）
        - target  = raw_patch 自身
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
        self.GPU = '0'     # 例如 '0' 或 '0,1'
        self.ngpu = 1

        # patch / 采样相关
        # 'XY' / 'T' / 'PIXELDROP'
        self.sample_mode = 'XY'
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.overlap_factor = 0.5
        self.gap_x = 115
        self.gap_y = 115
        self.gap_t = 115

        self.scale_factor = 1.0
        self.train_datasets_size = 2000  # 总 patch 数
        self.select_img_num = 1000       # 每个 stack 最多选前 N 帧
        self.num_workers = 4

        # ===== speed knobs (默认不改变训练数值行为) =====
        self.use_amp = False
        self.cudnn_benchmark = False
        self.allow_tf32 = False
        self.log_interval = 20
        self.save_vis_every_stack = True
        self.save_vis_every_n_epoch = 1
        self.seed = None
        self.deterministic_training = False

        # ETA knobs
        self.eta_ema_alpha = 0.05  # EMA 平滑系数（0.03~0.1 都可以）

        # ===== NEW: PixelDrop knobs =====
        self.pixel_drop_ratio = 0.5   # 丢弃比例（默认一半）
        self.pixel_drop_value = 0.0   # 丢弃值（默认置 0）

        # 预训练权重（如果不需要，可以保持为 None）
        if self.sample_mode == 'XY':
            self.pretrained_pth = 'pretrained_pth//deepcad_XY//Self_SL_30.pth'
        elif self.sample_mode == 'T' or self.sample_mode == 'N2V':
            self.pretrained_pth = 'pretrained_pth//deepcad_T//Self_SL_30.pth'
        else:
            # PIXELDROP 默认不强制加载，你也可以自己指定 pretrained_pth
            self.pretrained_pth = None

        # 内部变量
        self.img_name_list = []
        self.img_dict = {}
        self.stack_num = 0
        self.whole_x = 0
        self.whole_y = 0
        self.whole_t = 0

        self._preprocess_logged = False
        self.local_model = None
        self.pth_path = None

        # cache
        self._stack_meta = {}        # {im_dir: (T,H,W)}
        self._valid_tif_files = []   # list of paths

        # 根据外部字典覆盖参数
        self.set_params(params_dict)


    # -------------------------
    #  对外主入口
    # -------------------------
    def run(self):
        if self.seed is not None:
            seed_everything(int(self.seed), bool(self.deterministic_training))

        print("\033[95m[Step 1] Preparing files...\033[0m")
        self.prepare_file()

        print("\033[94m[Step 2] Scanning stacks (FAST meta) + preprocessing coords...\033[0m")
        self._scan_stacks_and_cache_meta()          # 只做一次
        self.train_preprocess_lessMemoryMulStacks() # 仍会采样坐标

        print("\033[96m[Step 3] Saving training parameters to YAML...\033[0m")
        self.save_yaml_train()

        print("\033[92m[Step 4] Initializing network...\033[0m")
        self.initialize_network()

        print("\033[93m[Step 5] Distributing model to GPU(s)...\033[0m")
        self.distribute_GPU()

        print("\033[91m[Step 6] Starting training...\033[0m")
        self.train()

    # -------------------------
    #  配置与路径
    # -------------------------
    def set_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # ------------------------------------------------------------
        # DeepCAD 的 3D UNet 在 Encoder 中会做 3 次 MaxPool3d
        # （pool_kernel_size 默认 (2,2,2)），因此进入第二/第三次池化前，
        # 时间/空间维度都至少需要 >= 2。
        #
        # 为了避免 patch_t 太小导致某一层出现 output size == 0，
        # 这里对 patch 尺寸做最小保护：
        #   - 3 次下采样 => 有效维度需要 >= 2^3 = 8
        #   - sample_mode='XY' 时，split_patch_XY 会把最后一个空间维减半，
        #     而当前 data_process 返回的原始 raw_patch 维度顺序是 (B,C,T,H,W)，
        #     因此“被减半”的是 patch_x（对应 W）。
        # ------------------------------------------------------------
        if str(self.sample_mode).upper() == "XY":
            # 有效 D = patch_t
            self.patch_t = max(int(self.patch_t), 8)
            # 有效 W = patch_x / 2
            self.patch_x = max(int(self.patch_x), 16)
            # 有效 H = patch_y
            self.patch_y = max(int(self.patch_y), 8)
        elif str(self.sample_mode).upper() == "T":
            # split_patch_T 会把时间维减半 => 有效 D = patch_t / 2
            self.patch_t = max(int(self.patch_t), 16)
            self.patch_x = max(int(self.patch_x), 8)
            self.patch_y = max(int(self.patch_y), 8)
        elif str(self.sample_mode).upper() == "N2V":
            self.patch_t = max(int(self.patch_t), 8)
            self.patch_x = max(int(self.patch_x), 8)
            self.patch_y = max(int(self.patch_y), 8)
        else:
            # 让后续逻辑抛出更明确错误
            pass

        # gap 计算（保持原逻辑）
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))

        # GPU 数量（保持原逻辑）
        self.ngpu = str(self.GPU).count(',') + 1

        # 根据 sample_mode 自动选默认预训练（除非用户显式传 pretrained_pth）
        if "pretrained_pth" not in params_dict:
            if self.sample_mode == 'XY':
                self.pretrained_pth = 'pretrained_pth//deepcad_XY//Self_SL_30.pth'
            elif self.sample_mode == 'T':
                self.pretrained_pth = 'pretrained_pth//deepcad_T//Self_SL_30.pth'
            else:
                self.pretrained_pth = None

        print('\033[1;31mTraining parameters -----> \033[0m')
        print(self.__dict__)


    def prepare_file(self):
        if self.datasets_path.endswith('/'):
            self.datasets_name = self.datasets_path.rstrip('/').split('/')[-1]
        else:
            self.datasets_name = self.datasets_path.split('/')[-1]

        if self.pth_name is None:
            pth_name = self.datasets_name
        else:
            pth_name = self.pth_name

        self.pth_path = os.path.join(self.pth_dir, pth_name)
        os.makedirs(self.pth_path, exist_ok=True)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------
    #  模型与 GPU
    # -------------------------
    def initialize_network(self):
        self.local_model = Network_3D_Unet(
            in_channels=1,
            out_channels=1,
            f_maps=self.fmap,
            final_sigmoid=False
        )

    def distribute_GPU(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)

        torch.backends.cudnn.benchmark = bool(self.cudnn_benchmark)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = bool(self.allow_tf32)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(self.allow_tf32)

        if not torch.cuda.is_available():
            print("\033[93m[Warning] CUDA is not available, training will run on CPU.\033[0m")
            return

        self.local_model = self.local_model.cuda()

        if self.pretrained_pth is not None and os.path.exists(self.pretrained_pth):
            state = torch.load(self.pretrained_pth, map_location="cuda")
            self.local_model.load_state_dict(state)
            RED = "\033[1;31m"
            RESET = "\033[0m"
            # NOTE: keep console output ASCII-only (Windows terminals may use gbk)
            print(f"{RED}{'!' * 60}")
            print(f"!!!   [MODEL LOADED] {self.pretrained_pth}   !!!")
            print(f"{'!' * 60}{RESET}")

        if self.ngpu > 1:
            device_ids = list(range(self.ngpu))
        else:
            device_ids = [0]
        self.local_model = nn.DataParallel(self.local_model, device_ids=device_ids)

    # -------------------------
    #  FAST: 扫描 stack 元信息（只做一次）
    # -------------------------
    def _scan_stacks_and_cache_meta(self):
        patch_t2 = self.patch_t

        tif_files = glob.glob(os.path.join(self.datasets_path, '*.tif')) \
                    + glob.glob(os.path.join(self.datasets_path, '*.tiff'))

        if not self._preprocess_logged:
            # NOTE: keep console output ASCII-only (Windows terminals may use gbk)
            print(f"\033[1;31m[WARN][WARN][WARN] PATH: {self.datasets_path} [WARN][WARN][WARN]\033[0m")
            print(f"\033[1;31m[WARN][WARN][WARN] FILES: {tif_files} [WARN][WARN][WARN]\033[0m")
            self._preprocess_logged = True

        valid = []
        meta = {}
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

            self.whole_t, self.whole_y, self.whole_x = whole_t, whole_y, whole_x

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
    #  训练主循环（修正 ETA）
    # -------------------------
    def train(self):
        if self.seed is not None:
            seed_everything(int(self.seed), bool(self.deterministic_training))
            data_generator = torch.Generator()
            data_generator.manual_seed(int(self.seed))
        else:
            data_generator = None

        torch.set_grad_enabled(True)

        optimizer_G = torch.optim.Adam(
            self.local_model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )

        use_cuda = torch.cuda.is_available()

        L1_pixelwise = torch.nn.L1Loss()
        L2_pixelwise = torch.nn.MSELoss()
        if use_cuda:
            L1_pixelwise = L1_pixelwise.cuda()
            L2_pixelwise = L2_pixelwise.cuda()

        use_amp = bool(self.use_amp) and use_cuda
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # === ETA: global + EMA ===
        ema_step_time = None
        ema_alpha = float(self.eta_ema_alpha)

        time_start = time.time()

        for epoch in range(self.n_epochs):
            # 每个 epoch 重采样坐标（逻辑不变）
            self.train_preprocess_lessMemoryMulStacks()

            # ✅ 计算本 epoch 总 batch 数（跨所有 stack）
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

                noise_im = tiff.imread(im_dir).astype(np.float32)
                if noise_im.shape[0] > self.select_img_num:
                    noise_im = noise_im[:self.select_img_num, :, :]
                noise_im = noise_im - noise_im.mean()

                print('Training image shape ::: ', noise_im.shape)
                train_data = trainset(noise_im, coordinate_list)

                dl_kwargs = dict(
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers
                )
                if data_generator is not None:
                    dl_kwargs["generator"] = data_generator
                    dl_kwargs["worker_init_fn"] = functools.partial(
                        seed_worker,
                        base_seed=int(self.seed),
                        deterministic=bool(self.deterministic_training),
                    )
                if use_cuda:
                    dl_kwargs["pin_memory"] = True
                if self.num_workers and int(self.num_workers) > 0:
                    dl_kwargs["persistent_workers"] = True
                    dl_kwargs["prefetch_factor"] = 4

                trainloader = DataLoader(train_data, **dl_kwargs)

                for iteration, raw_patch in enumerate(trainloader):
                    # data split / corruption
                    if self.sample_mode == 'XY':
                        input_, target = split_patch_XY(raw_patch)
                    elif self.sample_mode == 'T':
                        input_, target = split_patch_T(raw_patch)
                    elif self.sample_mode == 'N2V':
                        input_, target = split_patch_PixelDrop(
                            raw_patch,
                            drop_ratio=float(self.pixel_drop_ratio),
                            drop_value=float(self.pixel_drop_value)
                        )
                    else:
                        raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

                    if use_cuda:
                        input_ = input_.cuda(non_blocking=True)
                        target = target.cuda(non_blocking=True)

                    real_A = Variable(input_)
                    real_B = Variable(target)

                    optimizer_G.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            fake_B = self.local_model(real_A)
                            L1_loss = L1_pixelwise(fake_B, real_B)
                            L2_loss = L2_pixelwise(fake_B, real_B)
                            Total_loss = 0.5 * L1_loss + 0.5 * L2_loss
                        scaler.scale(Total_loss).backward()
                        scaler.step(optimizer_G)
                        scaler.update()
                    else:
                        fake_B = self.local_model(real_A)
                        L1_loss = L1_pixelwise(fake_B, real_B)
                        L2_loss = L2_pixelwise(fake_B, real_B)
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

                    if self.log_interval and (iteration % int(self.log_interval) == 0):
                        time_cost = datetime.timedelta(seconds=int(time.time() - time_start))
                        print(
                            '\r\033[2K'
                            '\033[92m'
                            '[Denoise TRAIN] [Epoch %d/%d] [Batch %d/%d] '
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

                # 每个 stack 训练完保存一次（原逻辑）
                print('\n', end=' ')
                self.save_model(epoch)

                # 保存可视化（原逻辑）
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

        print("\n")

    # -------------------------
    #  模型保存
    # -------------------------
    def save_model(self, epoch: int):
        model_save_name = os.path.join(self.pth_path, f'Self_SL_{epoch + 1}.pth')
        if isinstance(self.local_model, nn.DataParallel):
            torch.save(self.local_model.module.state_dict(), model_save_name)
        else:
            torch.save(self.local_model.state_dict(), model_save_name)
