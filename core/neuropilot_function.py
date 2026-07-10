import os
import sys
import time
import json
from pathlib import Path


def _ensure_windows_torch_openmp_alias():
    if os.name != "nt":
        return
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    if not prefix:
        return
    bin_dir = Path(prefix) / "Library" / "bin"
    src = bin_dir / "libomp.dll"
    dst = bin_dir / "libomp140.x86_64.dll"
    if src.exists() and not dst.exists():
        try:
            dst.write_bytes(src.read_bytes())
        except OSError:
            pass


if os.name == "nt" and not os.environ.get("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ensure_windows_torch_openmp_alias()

from deepcad.train_collection import training_class
from deepcad.test_collection import testing_class

from demotion import OF_options
from demotion.compensate_recording import compensate_recording

from PyLoReg.pylog_inference import (demotion_PyLoReg_infer2stack_less_save_acc_v3)


def _pyloreg_backend_from_env() -> str:
    backend = os.environ.get("NEUROPILOT_PYLOREG_BACKEND", "torch").strip().lower()
    if backend not in {"torch", "tensorrt"}:
        print(f"[WARN] Unsupported NEUROPILOT_PYLOREG_BACKEND={backend!r}; using torch.")
        backend = "torch"
    return backend


def get_subfolder_names(directory_path):
    with os.scandir(directory_path) as entries:
        subfolder_names = [entry.name for entry in entries if entry.is_dir()]
    return subfolder_names


def get_single_tif_path(datasets_path: str) -> Path:
    root = Path(datasets_path)
    tifs = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')]
    if len(tifs) == 0:
        raise FileNotFoundError(f"没有找到tif文件：{root}")
    if len(tifs) > 1:
        raise RuntimeError(f"找到多个tif文件（{len(tifs)}个），请确认只有一个：{root}")
    return tifs[0]


# ============================================================
# 1) DeepCAD TEST: 全部从参数传入（不再读 CONFIG）
# ============================================================
def test_deepcad(
    datasets_path,
    pth_dir="./pth_deepcad",
    denoise_model=None,
    output_dir="./results_deepcad",
    output_folder=None,
    # ---- formerly CONFIG ----
    patch_xy=192,
    patch_t=512,
    overlap_factor=0.5,
    gpu="0,1",
    num_workers=0,
    fmap=8,
    scale_factor=1,
    # ---- others ----
    test_datasize=100000,
    visualize_images_per_epoch=False,
):
    folder_name = os.path.basename(os.path.normpath(datasets_path))
    if denoise_model is None:
        denoise_model = folder_name

    # 你原本传进来的 output_folder 在 __main__ 里没给，会导致 json_path 拼接出错
    # 这里给一个更稳的默认：用 folder_name
    if output_folder is None:
        output_folder = folder_name

    all_path = {
        "datasets_path": str(datasets_path),
        "denoise_model": str(denoise_model),
        "output_dir": str(output_dir),
        "output_folder": str(output_folder),
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{output_folder}_DeepCAD_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_path, f, indent=4, ensure_ascii=False)

    test_dict = {
        # dataset dependent parameters
        "patch_x": patch_xy,
        "patch_y": patch_xy,
        "patch_t": patch_t,
        "overlap_factor": overlap_factor,
        "scale_factor": scale_factor,
        "test_datasize": test_datasize,
        "datasets_path": datasets_path,
        "pth_dir": pth_dir,
        "denoise_model": denoise_model,
        "output_dir": output_dir,
        "output_folder": output_folder,

        # network related parameters
        "fmap": fmap,
        "GPU": gpu,
        "num_workers": num_workers,
        "visualize_images_per_epoch": visualize_images_per_epoch,
    }

    tc = testing_class(test_dict)
    tc.run()
    return tc.output_path


# ============================================================
# 2) DeepCAD TRAIN: 全部从参数传入（不再读 CONFIG）
# ============================================================
def train_deepcad(
    datasets_path,
    pth_dir="./pth_deepcad",
    pth_name=None,
    n_epochs=1,
    # ---- formerly CONFIG ----
    patch_x=192,
    patch_y=192,
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
    sample_mode = 'T',
    visualize_images_per_epoch=False,
    save_test_images_per_epoch=False,
    batch_size=4,
    seed=None,
    deterministic_training=False,
):
    train_dict = {
        # dataset dependent parameters
        "batch_size":batch_size,
        "sample_mode": sample_mode, #"XY",  # T
        "patch_x": patch_x,
        "patch_y": patch_y,
        "patch_t": patch_t,
        "overlap_factor": overlap_factor,
        "scale_factor": scale_factor,
        "select_img_num": select_img_num,
        "train_datasets_size": train_datasets_size,
        "datasets_path": datasets_path,

        # save / train
        "pth_dir": pth_dir,
        "pth_name": pth_name,
        "n_epochs": n_epochs,

        # optim
        "lr": lr,
        "b1": b1,
        "b2": b2,

        # network
        "fmap": fmap,
        "GPU": gpu,
        "num_workers": num_workers,
        "seed": seed,
        "deterministic_training": deterministic_training,

        # viz
        "visualize_images_per_epoch": visualize_images_per_epoch,
        "save_test_images_per_epoch": save_test_images_per_epoch,
    }

    tc = training_class(train_dict)
    tc.run()
    return tc.pth_path



def train_deepcad_ddp(
    datasets_path,
    pth_dir="./pth_deepcad",
    pth_name=None,
    n_epochs=1,
    # ---- formerly CONFIG ----
    patch_x=192,
    patch_y=192,
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
        "patch_x": patch_x,
        "patch_y": patch_y,
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
    from deepcad import train_collection_ddp as deepcad_ddp

    if master_port is None:
        master_port = deepcad_ddp._find_free_port()

    # 为了避免 Windows/交互环境问题，推荐用 spawn
    deepcad_ddp.mp.set_start_method("spawn", force=True)

    deepcad_ddp.mp.spawn(
        deepcad_ddp._ddp_worker_train_deepcad,
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

###########################################################################################
###########################################################################################
# ============================================================
# 3) demotion_flow: GPU 等从参数传入（不再读 CONFIG）
# ============================================================
def save_paths_to_json(input_dir, raw_dir, output_dir, output_path):
    all_path = {
        "input_dir": str(input_dir),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    json_path = os.path.join(output_path, "data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_path, f, indent=4, ensure_ascii=False)
    print(f"✅ 已保存到 {json_path}")


def demotion_flow(
    datasets_path,
    input_path=None,
    output_path=None,
    # ---- formerly CONFIG GPU ----
    gpu="0,1",
    # ---- original OF params (也都做成参数，方便你以后也从外部传) ----
    output_format="TIFF",
    alpha=1.5,
    sigma=(2, 2, 0.1),
    quality_setting="balanced",
    bin_size=1,
    buffer_size=200,
    reference_frames=tuple(range(700, 900)),
):
    raw_dir = get_single_tif_path(datasets_path)
    tif_name = os.path.basename(os.path.normpath(raw_dir))
    print('\033[96mRAW data :\033[0m', raw_dir)

    folder_name = os.path.basename(os.path.normpath(datasets_path))

    if input_path is None:
        input_root = f".//results_deepcad//{folder_name}"  # _DeepCAD
        input_dir = get_single_tif_path(input_root)
    else:
        input_dir = get_single_tif_path(input_path)
    print('\033[96mEST motion data :\033[0m', input_dir)

    if output_path is None:
        output_root = f".//results_demotion//{folder_name}_demotion"
        os.makedirs(output_root, exist_ok=True)
        output_dir = os.path.join(output_root, tif_name)
    else:
        os.makedirs(output_path, exist_ok=True)
        output_dir = os.path.join(output_path, tif_name)
    print('\033[96mDE motion data :\033[0m', output_dir)

    # 如果 demotion 里也需要 GPU 环境变量，这里设置一下（保持你原来逻辑一致）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    save_paths_to_json(input_dir, raw_dir, output_dir, output_path or output_root)

    options = OF_options.OFOptions(
        input_file=input_dir,
        raw_images_dir=raw_dir,
        output_file_name=output_dir,
        output_format=output_format,
        alpha=alpha,
        sigma=list(sigma),
        quality_setting=quality_setting,
        bin_size=bin_size,
        buffer_size=buffer_size,
        reference_frames=list(reference_frames),
    )

    st_time = time.time()
    compensate_recording(options)
    ed_time = time.time()
    print(f"Time taken for nonrigid motion correction: {ed_time - st_time:.2f} seconds")


# ============================================================
# 4) demotion_PyLoReg: GPU 等从参数传入（不再读 CONFIG）
# ============================================================
def demotion_PyLoReg(
    datasets_path,
    input_path=None,
    output_path=None,
    # ---- formerly CONFIG GPU ----
    gpu="0,1",
    # ---- pyloreg infer params ----
    iteration_num=2,
    max_frames=None,
):
    datasets_path = str(datasets_path)
    folder_name = os.path.basename(os.path.normpath(datasets_path))

    raw_root = Path(datasets_path)
    raw_tifs = sorted([
        p for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    ])
    if len(raw_tifs) == 0:
        raise FileNotFoundError(f"No tif files found under: {datasets_path}")

    print(f"\033[96mFound {len(raw_tifs)} raw tifs under:\033[0m {datasets_path}")

    if input_path is None:
        input_root = os.path.join(".//results_deepcad", folder_name)
    else:
        input_root = str(input_path)

    input_root_path = Path(input_root)
    input_map = {}

    if input_root_path.is_dir():
        denoised_tifs = sorted([
            p for p in input_root_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
        ])
        input_map = {p.name: p for p in denoised_tifs}
        print(f"\033[96mSearch denoised tifs under:\033[0m {input_root}")
        print(f"\033[96mFound {len(denoised_tifs)} denoised tifs.\033[0m")
    elif input_root_path.is_file():
        if len(raw_tifs) > 1:
            raise RuntimeError(
                f"input_path is a single file ({input_root}), "
                f"but there are {len(raw_tifs)} raw tifs. Cannot match them."
            )
        input_map[raw_tifs[0].name] = input_root_path
        print(f"\033[96mEST motion data (single file):\033[0m {input_root_path}")
    else:
        raise FileNotFoundError(f"Input path not found: {input_root}")

    if output_path is None:
        output_root = os.path.join(".//results_demotion", folder_name + "_demotion")
    else:
        output_root = str(output_path)
    os.makedirs(output_root, exist_ok=True)
    output_root_path = Path(output_root)
    print(f"\033[96mDemotion results will be saved under:\033[0m {output_root_path}")

    # set GPU (保持你原逻辑)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    save_paths = []
    for raw_path in raw_tifs:
        raw_path = raw_path.resolve()
        tif_name = raw_path.name

        print("\n\033[96mRAW data :\033[0m", raw_path)

        if tif_name not in input_map:
            print(f"\033[93m[WARN] No matched denoised tif for:\033[0m {tif_name}  (skip)")
            continue

        input_dir = input_map[tif_name].resolve()
        print("\033[96mEST motion data :\033[0m", input_dir)

        output_dir = (output_root_path / tif_name)
        print("\033[96mDE motion data  :\033[0m", output_dir)

        # demotion_PyLoReg_infer2stack_less_save_acc
        # demotion_PyLoReg_infer2stack_less_save
        # demotion_PyLoReg_infer2stack_less_save_accacc
        # demotion_PyLoReg_infer2stack_less_save_acc_v3
        stack_save_path = demotion_PyLoReg_infer2stack_less_save_acc_v3(
            img_stack_path=str(input_dir),
            raw_stack_path=str(raw_path),
            stack_save_path=str(output_dir),
            iteration_num=int(iteration_num),
            max_frames=max_frames,
            save_mask_flow=True,
            save_templates=True,
            network_backend=_pyloreg_backend_from_env(),
        )
        save_paths.append(stack_save_path)

    print("\n\033[92mPyLoReg demotion finished for all matched tifs.\033[0m")
    print("\033[96mWarped stacks:\033[0m")
    for p in save_paths:
        print("  ", p)

    return save_paths



###########################################################################################
###########################################################################################


def _require_deepinterpolation_runtime():
    """
    Load DeepInterpolation lazily so the main pipeline can be imported without
    bundling that optional backend.
    """
    try:
        from deepinterpolation.di_train import DITrainer
        from deepinterpolation.di_test import DITester
    except ImportError as exc:
        raise ImportError(
            "DeepInterpolation runtime is not available. "
            "Install/include the deepinterpolation package before calling "
            "train_deepinter or test_deepinter."
        ) from exc
    return DITrainer, DITester


CONFIG = {
    "PATCH_XY": 192,
    "PATCH_T": 128,
    "OVERLAP_FACTOR": 0.5,
    "GPU": "0,1",
    "NUM_WORKERS": 0,      
    "FMAP": 8,
    "PTH_DIR": "./pth_deepcad",
    "RESULT_DIR": "./results_deepcad",
    "SCALE_FACTOR": 1,
}


def train_deepinter(datasets_path, pth_dir=None, pth_name=None, n_epochs = 1,
                    patch_t = 128, train_datasets_size=1000, GPU='0'  ,batch_size=4):    
    DITrainer, _ = _require_deepinterpolation_runtime()
    params = dict(
        datasets_path=datasets_path,
        output_dir=pth_dir,
        output_folder=pth_name,

        GPU=GPU,
        batch_size=batch_size,
        num_workers=4,
        lr=1e-4,
        epochs=n_epochs,
        # 64 128
        Npre=patch_t,
        Npost=patch_t,
        base_channels=8,

        train_samples_per_movie=train_datasets_size,
        val_fraction=0.05,
        seed=0,
    )
    DITrainer(params).run()



def test_deepinter(datasets_path, pth_dir=None, denoise_model=None, 
                    output_dir=None, output_folder=None,
                    patch_t = 128,  GPU='0', batch_size=4  ):
    _, DITester = _require_deepinterpolation_runtime()
    params = dict(
        datasets_path=datasets_path,
        output_dir=output_dir,
        output_folder=output_folder,

        ckpt_path=pth_dir+'//'+denoise_model,

        GPU=GPU,
        batch_size=batch_size,
        num_workers=4,

        Npre=patch_t,
        Npost=patch_t,
        base_channels=8,

        # NEW: you can choose reflect/edge/constant
        pad_mode="reflect",   # or "edge"
        pad_constant=0.0,

        save_uint16=True,
        denoise_suffix="",
    )
    DITester(params).run()
