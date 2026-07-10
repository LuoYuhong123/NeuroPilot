import os
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import sys

from skimage import io
from .network import Network_3D_Unet
from .data_process import (
    test_preprocess_chooseOne,
    testset,
    multibatch_test_save,
    singlebatch_test_save,
)
from deepcad.movie_display import test_img_display


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def sec2hms(sec: int) -> str:
    sec = int(max(0, sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


class testing_class:
    """
    Testing class for 3D U-Net DeepCAD model. (FAST)
    """

    def __init__(self, params_dict):
        # -------- 基本参数 --------
        self.datasets_path = ""
        self.output_dir = "./results"
        self.pth_dir = ""
        self.denoise_model = ""
        self.output_folder = None

        # -------- Patch / 重叠 --------
        self.overlap_factor = 0.5
        self.patch_t = 150
        self.patch_x = 150
        self.patch_y = 150
        self.gap_y = 115
        self.gap_x = 115
        self.gap_t = 115

        # -------- 网络 / 测试参数 --------
        self.fmap = 16
        self.batch_size = 1              # ✅ 现在允许用户设置 >1
        self.num_workers = 0
        self.scale_factor = 1
        self.test_datasize = 400

        # -------- GPU --------
        self.GPU = "0"
        self.ngpu = 1

        # -------- 可视化 --------
        self.visualize_images_per_epoch = False
        self.colab_display = False
        self.result_display = ""

        # ====== 加速开关 ======
        self.use_amp = False            # True 会更快（推理），但可能有极小数值差异
        self.cudnn_benchmark = True     # True 通常更快（固定输入形状）
        self.print_every = 10           # 刷新频率（patch 进度）

        # 内部变量
        self.img_list = []
        self.model_list = None
        self.model_list_length = 1
        self.datasets_name = ""
        self.output_path = ""
        self.local_model = None

        self.set_params(params_dict)

    # -------------------------
    #   主入口
    # -------------------------
    def run(self):
        self.prepare_file()
        self.read_modellist()
        self.read_imglist()
        self.save_yaml_test()
        self.initialize_network()
        self.distribute_GPU()
        self.test()

    # -------------------------
    #   参数与路径
    # -------------------------
    def set_params(self, params_dict):
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # 根据 overlap_factor 更新 gap
        self.gap_x = int(self.patch_x * (1 - self.overlap_factor))
        self.gap_y = int(self.patch_y * (1 - self.overlap_factor))
        self.gap_t = int(self.patch_t * (1 - self.overlap_factor))

        # 统计 GPU 数量
        self.ngpu = str(self.GPU).count(",") + 1

        print("\033[1;31mTesting parameters -----> \033[0m")
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                print(f"  {k}: {v}")

    def prepare_file(self):
        if self.datasets_path.endswith("/"):
            self.datasets_name = self.datasets_path.rstrip("/").split("/")[-1]
        else:
            self.datasets_name = self.datasets_path.split("/")[-1]

        os.makedirs(self.output_dir, exist_ok=True)

        if self.output_folder is None:
            self.output_folder = self.denoise_model

        self.output_path = os.path.join(self.output_dir, f"{self.output_folder}_DeepCAD")
        os.makedirs(self.output_path, exist_ok=True)

    # -------------------------
    #   图像 & 模型列表
    # -------------------------
    def read_imglist(self):
        im_folder = self.datasets_path
        self.img_list = [
            f
            for f in list(os.walk(im_folder, topdown=False))[-1][-1]
            if f.lower().endswith((".tif", ".tiff"))
        ]
        self.img_list.sort()

        print("\033[1;31mStacks for processing -----> \033[0m")
        for img in self.img_list:
            print(img)

    @staticmethod
    def _pick_latest_model(models):
        import re
        extracted = []
        for m in models:
            match = re.search(r"(\d+)\.pth$", m)
            if match:
                num = int(match.group(1))
                extracted.append((num, m))
        if not extracted:
            raise RuntimeError("No valid .pth file (with trailing number) found in model list.")
        latest = max(extracted, key=lambda x: x[0])[1]
        return latest

    def read_modellist(self):
        model_path = os.path.join(self.pth_dir, self.denoise_model)
        print("model_path ---> ", model_path)

        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model directory not found: {model_path}")

        model_file_list = list(os.walk(model_path, topdown=False))[-1][-1]
        model_list = [item for item in model_file_list if item.lower().endswith(".pth")]
        model_list.sort()
        if not model_list:
            raise RuntimeError("There is no .pth file in the models directory!")

        self.model_list = self._pick_latest_model(model_list)
        self.model_list_length = 1

    # -------------------------
    #   网络 & GPU
    # -------------------------
    def initialize_network(self):
        denoise_generator = Network_3D_Unet(
            in_channels=1,
            out_channels=1,
            f_maps=self.fmap,
            final_sigmoid=True,
        )
        self.local_model = denoise_generator

    def distribute_GPU(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.GPU)

        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()

            # ✅ 关键：单 GPU 不要 DataParallel（很多情况下更快）
            n_dev = torch.cuda.device_count()
            if n_dev > 1:
                device_ids = list(range(n_dev))
                self.local_model = nn.DataParallel(self.local_model, device_ids=device_ids)
                print(f"\033[1;31mUsing {n_dev} GPU(s) (DataParallel) -----> \033[0m")
            else:
                print("\033[1;31mUsing 1 GPU (no DataParallel) -----> \033[0m")
        else:
            print("\033[93m[Warning] CUDA is not available, testing will run on CPU.\033[0m")

    # -------------------------
    #   YAML 保存
    # -------------------------
    def save_yaml_test(self):
        yaml_name = os.path.join(self.output_path, "para.yaml")
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

        with open(yaml_name, "w") as f:
            yaml.dump(para, f)

    # -------------------------
    #   测试主流程（FAST）
    # -------------------------
    @torch.inference_mode()
    def test(self):
        pth_count = 0
        print(f"\033[1;31m[MODEL] {self.model_list}\033[0m")

        if not self.model_list or not self.model_list.endswith(".pth"):
            print("\033[1;31mNo valid .pth model found to test.\033[0m")
            return

        pth_count += 1
        pth_name = self.model_list

        output_path_name = self.output_path
        os.makedirs(output_path_name, exist_ok=True)

        # load model weights
        model_name = os.path.join(self.pth_dir, self.denoise_model, pth_name)
        state = torch.load(model_name, map_location="cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(self.local_model, nn.DataParallel):
            self.local_model.module.load_state_dict(state)
        else:
            self.local_model.load_state_dict(state)

        self.local_model.eval()
        if torch.cuda.is_available():
            self.local_model = self.local_model.cuda()

        print("Testing the last / latest model by default:")

        use_amp = bool(self.use_amp) and torch.cuda.is_available()
        autocast_ctx = torch.cuda.amp.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast

        # loop stacks
        for N in range(len(self.img_list)):
            (
                name_list,
                noise_img,
                coordinate_list,
                test_im_name,
                img_mean,
                input_data_type,
            ) = test_preprocess_chooseOne(self, N)

            time_start = time.time()
            last_tick = time.time()

            denoise_img = np.zeros_like(noise_img, dtype=np.float32)
            input_img = np.zeros_like(noise_img, dtype=np.float32)

            test_data = testset(name_list, coordinate_list, noise_img)

            # ✅ DataLoader 工程参数（worker>0 才生效）
            nw = int(self.num_workers)
            dl_kwargs = dict(
                batch_size=int(self.batch_size),   # ✅ 允许 >1
                shuffle=False,
                num_workers=nw,
                pin_memory=True if torch.cuda.is_available() else False,
            )
            if nw > 0:
                dl_kwargs.update(
                    persistent_workers=True,
                    prefetch_factor=4,
                )

            testloader = DataLoader(test_data, **dl_kwargs)
            total_batches = len(testloader)

            # --- patch loop ---
            for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                if torch.cuda.is_available():
                    noise_patch = noise_patch.cuda(non_blocking=True)

                # forward (AMP optional)
                if use_amp:
                    with autocast_ctx(dtype=torch.float16):
                        fake_B = self.local_model(noise_patch)
                else:
                    fake_B = self.local_model(noise_patch)

                # ETA (avoid heavy prints every iter)
                if (iteration % max(1, int(self.print_every)) == 0) or (iteration + 1 == total_batches):
                    elapsed = time.time() - time_start
                    # use smoothed per-iter time
                    now = time.time()
                    dt = max(1e-6, now - last_tick)
                    last_tick = now
                    batches_left = total_batches - (iteration + 1)
                    eta = int(batches_left * dt)
                    msg = (
                        f'\033[95m'
                        f'[⏱️  Denoise TEST {pth_count}/{self.model_list_length}｜{pth_name}] '
                        f'[Stack {N + 1}/{len(self.img_list)}｜{self.img_list[N]}] '
                        f'[Patch {iteration + 1}/{total_batches}] '
                        f'[Time {sec2hms(int(elapsed))}] '
                        f'[ETA {sec2hms(int(eta))}]\033[0m'
                    )
                    sys.stdout.write('\r\033[2K' + msg)
                    sys.stdout.flush()

                # ---- move output to numpy once per batch ----
                output_np = fake_B.detach().cpu().numpy()
                raw_np = noise_patch.detach().cpu().numpy()

                # output_np / raw_np could be:
                #  - [B,1,T,H,W] or [B,T,H,W] depending on net/data
                # keep your original logic but loop over batch dimension
                if output_np.ndim == 5:  # [B,1,*,*,*]
                    output_np = np.squeeze(output_np, axis=1)  # [B,?, ?, ?]
                if raw_np.ndim == 5:
                    raw_np = np.squeeze(raw_np, axis=1)

                # coordinates: could be tensor/list; make it indexable per item
                # many datasets return coordinates as list-like length B
                if isinstance(single_coordinate, torch.Tensor):
                    coord_batch = single_coordinate
                else:
                    coord_batch = single_coordinate

                B = output_np.shape[0] if output_np.ndim >= 4 else 1

                # When B==1, keep compatibility
                for bi in range(B):
                    output_image = output_np[bi] if B > 1 else np.squeeze(output_np)
                    raw_image = raw_np[bi] if B > 1 else np.squeeze(raw_np)

                    # coordinate item
                    coord_i = coord_batch[bi] if B > 1 else coord_batch

                    if output_image.ndim == 3:
                        postprocess_turn = 1
                    else:
                        postprocess_turn = output_image.shape[0]

                    if postprocess_turn > 1:
                        for idx in range(postprocess_turn):
                            (
                                output_patch,
                                raw_patch,
                                stack_start_w,
                                stack_end_w,
                                stack_start_h,
                                stack_end_h,
                                stack_start_s,
                                stack_end_s,
                            ) = multibatch_test_save(coord_i, idx, output_image, raw_image)

                            denoise_img[
                                stack_start_s:stack_end_s,
                                stack_start_h:stack_end_h,
                                stack_start_w:stack_end_w,
                            ] = output_patch
                            input_img[
                                stack_start_s:stack_end_s,
                                stack_start_h:stack_end_h,
                                stack_start_w:stack_end_w,
                            ] = raw_patch
                    else:
                        (
                            output_patch,
                            raw_patch,
                            stack_start_w,
                            stack_end_w,
                            stack_start_h,
                            stack_end_h,
                            stack_start_s,
                            stack_end_s,
                        ) = singlebatch_test_save(coord_i, output_image, raw_image)

                        denoise_img[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = output_patch
                        input_img[
                            stack_start_s:stack_end_s,
                            stack_start_h:stack_end_h,
                            stack_start_w:stack_end_w,
                        ] = raw_patch

            # end patch loop: finish status line
            sys.stdout.write("\n")
            sys.stdout.flush()

            # merge done
            output_img = denoise_img.squeeze().astype(np.float32) * self.scale_factor + img_mean
            del denoise_img

            # visualize (optional)
            if self.visualize_images_per_epoch:
                print("Displaying the denoised file ----->")
                test_img_display(
                    output_img,
                    display_length=200,
                    norm_min_percent=1,
                    norm_max_percent=99,
                )

            # save uint16 (same as your logic)
            output_img = output_img - output_img.min()
            output_img = np.clip(output_img, 0, 65535).astype("uint16")

            result_name = os.path.join(output_path_name, self.img_list[N])
            print("Denoising TEST SAVE ::: ", result_name)
            io.imsave(result_name, output_img, check_contrast=False)

            if pth_count == self.model_list_length and self.colab_display:
                self.result_display = result_name
