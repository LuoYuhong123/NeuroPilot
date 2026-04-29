#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import os
import sys
import json
import hashlib
import shutil
import time
from pathlib import Path
import tifffile as tiff


def _ensure_windows_torch_openmp_alias() -> None:
    """
    Some Windows PyTorch wheels load fbgemm.dll, which may look specifically for
    libomp140.x86_64.dll even when a compatible libomp.dll is already present in
    the conda environment. Create the filename alias eagerly before importing any
    module that pulls in torch.
    """
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
            # If the alias cannot be created, torch import will raise the usual
            # loader error and the troubleshooting docs cover the manual fix.
            pass


if os.name == "nt" and not os.environ.get("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_ensure_windows_torch_openmp_alias()

from NeuMar_function import (
    train_deepcad, train_deepcad_ddp, test_deepcad,
    get_subfolder_names, demotion_PyLoReg
)

from utils import (
    ensure_dir, now_tag,
    has_pth_file, has_valid_tif,
    safe_write_exception
)
from pipeline_metrics import compute_metrics_for_tif, compare_two_metrics
from llm_advisor import get_llm_suggestion, has_configured_api_key
from downstream_pipeline import materialize_final_and_run_downstream
from report_builder import build_deterministic_report

# =============================================================================
# 0) CONFIG
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT_DIR / "input_data"
DEFAULT_RESULTS_PATH = ROOT_DIR / "tmp_pipeline_metrics" / "demo_run"


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _bootstrap_local_env() -> None:
    # Load .env first, then .env.example as a local fallback convenience source.
    _load_env_file(ROOT_DIR / ".env", override=False)
    _load_env_file(ROOT_DIR / ".env.example", override=False)


_bootstrap_local_env()


def _env_text(name: str, default: str | Path) -> str:
    value = os.getenv(name)
    if value is None:
        return str(default)
    value = value.strip()
    return value or str(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return _parse_subfolder_list(raw)


def _parse_subfolder_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _has_nonempty_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    value = value.strip()
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {'"', "'"}
    ):
        value = value[1:-1]
    return bool(value.strip())


def _discover_root_level_tifs(input_root: str | Path) -> list[str]:
    root = Path(input_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )


def _discover_subfolders_safe(input_root: str | Path) -> list[str]:
    root = Path(input_root).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return get_subfolder_names(str(root))


def _refresh_runtime_paths() -> None:
    global test_output_path, train_pth_dir, demotion_path, logs_dir
    test_output_path = os.path.join(RESULTS_PATH, "results_deepcad")
    train_pth_dir = os.path.join(RESULTS_PATH, "pth_deepcad")
    demotion_path = os.path.join(RESULTS_PATH, "results_demotion")
    logs_dir = os.path.join(RESULTS_PATH, "logs")


def _folder_output_name(folder_name: str) -> str:
    name = Path(str(folder_name)).name.strip()
    if not name:
        raise ValueError(f"Invalid folder name for output layout: {folder_name!r}")
    return name


def _folder_results_root(folder_name: str) -> Path:
    return Path(RESULTS_PATH).expanduser() / _folder_output_name(folder_name)


def _folder_runtime_paths(folder_name: str) -> dict[str, Path]:
    run_root = _folder_results_root(folder_name)
    return {
        "run_root": run_root,
        "results_deepcad": run_root / "results_deepcad",
        "pth_deepcad": run_root / "pth_deepcad",
        "results_demotion": run_root / "results_demotion",
        "logs": run_root / "logs",
    }


def _refresh_dataset_profile() -> None:
    global DATASET_PROFILE
    DATASET_PROFILE = CELL_DATASET_PROFILE if IS_CELL_DATA else NON_CELL_DATASET_PROFILE


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NeuroPilot calcium-imaging pipeline."
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        default=None,
        help="Root directory containing dataset subfolders. Each child folder may contain one or more tif/tiff files.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Root directory where one result subfolder will be created per dataset child folder, with shared and per-stack outputs inside.",
    )
    parser.add_argument(
        "--subfolders",
        dest="subfolders",
        default=None,
        help="Optional subset of dataset subfolders to process. Supports folder1,folder2 or ['folder1','folder2'].",
    )
    parser.add_argument(
        "--llm-mode",
        dest="llm_mode",
        choices=("off", "shadow", "apply"),
        default=None,
        help="Override pipeline LLM mode for this run.",
    )
    parser.add_argument(
        "--cell-data",
        dest="cell_data",
        action="store_true",
        help="Treat the dataset as cell-type imaging data and enable downstream segmentation.",
    )
    parser.add_argument(
        "--non-cell-data",
        dest="cell_data",
        action="store_false",
        help="Treat the dataset as non-cell data and skip cell-style downstream segmentation.",
    )
    parser.add_argument(
        "--GPU",
        "--gpu",
        dest="gpu",
        default=None,
        help="GPU index or comma-separated GPU indices, for example 0 or 0,1.",
    )
    parser.add_argument(
        "--um-per-pixel",
        dest="um_per_pixel",
        type=float,
        default=None,
        help="Microns per pixel used by report generation and downstream pre-segmentation profiling.",
    )
    parser.add_argument(
        "--frame-rate",
        dest="frame_rate",
        type=float,
        default=None,
        help="Frame rate in Hz used by report generation and downstream segmentation settings.",
    )
    parser.add_argument(
        "--downstream-env",
        dest="downstream_env",
        default=None,
        help="Optional conda environment name for cell-data downstream segmentation. Defaults to suite2p.",
    )
    parser.set_defaults(cell_data=None)
    return parser.parse_args()


directory_path = str(Path(_env_text("NEUROPILOT_INPUT_DIR", DEFAULT_INPUT_DIR)).expanduser())
subfolder_list = _env_list("NEUROPILOT_SUBFOLDERS") or _discover_subfolders_safe(directory_path)

RESULTS_PATH = str(Path(_env_text("NEUROPILOT_RESULTS_DIR", DEFAULT_RESULTS_PATH)).expanduser())
test_output_path = ""
train_pth_dir = ""
demotion_path = ""
logs_dir = ""
_refresh_runtime_paths()

GPU = _env_text("NEUROPILOT_GPU", "0")

data_low_SNR = 0
if data_low_SNR:
    PATCH_XY, PATCH_T = 128, 256
    PATCH_XY_TEST, PATCH_T_TEST = 128, 512
else:
    PATCH_XY, PATCH_T = 128, 128
    PATCH_XY_TEST, PATCH_T_TEST = 256, 128

OVERLAP_FACTOR = 0.5
NUM_WORKERS    = 0
FMAP           = 8
SCALE_FACTOR   = 1
TRAIN_BATCH_SIZE_DEFAULT = 6
TRAIN_BATCH_SIZE_BOUNDS = (1, 16)

TRAIN_DATASETS_SIZE = 1000
SELECT_IMG_NUM      = 100000
TEST_DATASIZE       = 100000
TRAIN_MAX_TIFS_FOR_MODEL = max(0, _env_int("NEUROPILOT_TRAIN_MAX_TIFS", 4))

iter_num = 2  # iter0 + iter1

# pipeline mode: off | shadow | apply
PIPELINE_LLM_MODE = _env_text("NEUROPILOT_PIPELINE_LLM_MODE", "shadow")
# advisor backend: mock | live
ADVISOR_BACKEND = _env_text("LLM_ADVISOR_BACKEND", "mock")
# Leave this unset in code; llm_advisor resolves OPENAI_API_KEY or local config at runtime.
ADVISOR_API_KEY = None
ADVISOR_MODEL = _env_text("LLM_ADVISOR_MODEL", "gpt-4.1-mini")
ADVISOR_BASE_URL = _env_text("LLM_ADVISOR_BASE_URL", "https://api.openai.com/v1")
ADVISOR_TIMEOUT_S = _env_float("LLM_ADVISOR_TIMEOUT_S", 20.0)

# Dataset type switch:
# - True: treat current input as cell-type imaging data and run downstream segmentation/trace extraction
# - False: skip downstream cell segmentation/trace extraction and label the report accordingly
IS_CELL_DATA = _env_bool("NEUROPILOT_IS_CELL_DATA", True)
CELL_DATASET_PROFILE = _env_text("NEUROPILOT_CELL_DATASET_PROFILE", "neuronal")
NON_CELL_DATASET_PROFILE = _env_text("NEUROPILOT_NON_CELL_DATASET_PROFILE", "unknown")
DATASET_PROFILE = CELL_DATASET_PROFILE if IS_CELL_DATA else NON_CELL_DATASET_PROFILE

# Optional deterministic test overrides for advisor mock response.
# Key = iter_index, Value = partial/full suggestion dict passed to llm_advisor(mock_response=...).
# Keep empty by default so live backend never depends on this symbol existing.
FORCE_MOCK_SUGGESTIONS_BY_ITER = {}

# Downstream runner:
# - mode="external_python": switch to another env/python for suite2p downstream
# - env_name: conda env name to resolve automatically
# - python_executable: optional explicit interpreter path, overrides env_name when set
DOWNSTREAM_RUNNER_MODE = _env_text("NEUROPILOT_DOWNSTREAM_RUNNER_MODE", "external_python")
DOWNSTREAM_ENV_NAME = _env_text("NEUROPILOT_DOWNSTREAM_ENV_NAME", "suite2p")
DOWNSTREAM_PYTHON_EXECUTABLE_RAW = os.getenv("NEUROPILOT_DOWNSTREAM_PYTHON", "").strip()
DOWNSTREAM_PYTHON_EXECUTABLE = DOWNSTREAM_PYTHON_EXECUTABLE_RAW or None

# Report config (Step 8, deterministic report layer only)
REPORT_IMAGING_MODALITY = _env_text("NEUROPILOT_REPORT_IMAGING_MODALITY", "2p")
REPORT_PIXEL_SIZE_UM = _env_float("NEUROPILOT_REPORT_PIXEL_SIZE_UM", 0.645)
REPORT_FPS_HZ = _env_float("NEUROPILOT_REPORT_FPS_HZ", 10.0)
REPORT_DISPLAY_NAME_FINAL = _env_text("NEUROPILOT_REPORT_DISPLAY_NAME", "NeuroPilot")
REPORT_EMBED_ASSETS = _env_bool("NEUROPILOT_REPORT_EMBED_ASSETS", True)
REPORT_INLINE_CSS = _env_bool("NEUROPILOT_REPORT_INLINE_CSS", True)
REPORT_GENERATE_PDF = _env_bool("NEUROPILOT_REPORT_GENERATE_PDF", False)
REPORT_CROP_SCALE_FACTOR = _env_float("NEUROPILOT_REPORT_CROP_SCALE_FACTOR", 0.55)
REPORT_KYMOGRAPH_LINE_COUNT = int(_env_float("NEUROPILOT_REPORT_KYMOGRAPH_LINE_COUNT", 2))
REPORT_USE_INTERMEDIATE_SECTIONS = _env_bool("NEUROPILOT_REPORT_USE_INTERMEDIATE_SECTIONS", False)

REPORT_TRY_PDF = _env_bool("NEUROPILOT_REPORT_TRY_PDF", False)
REPORT_GENERATE_OVERVIEW_PNGS = _env_bool("NEUROPILOT_REPORT_GENERATE_OVERVIEW_PNGS", True)

DOWNSTREAM_CONFIG = {
    "run_raw": True,
    "runner_config": {
        "mode": DOWNSTREAM_RUNNER_MODE,
        "env_name": DOWNSTREAM_ENV_NAME,
        "python_executable": DOWNSTREAM_PYTHON_EXECUTABLE,
    },
    "segmentation_config": {
        "fs_hz": REPORT_FPS_HZ,
        "nplanes": 1,
        "nchannels": 1,
        "do_registration": False,
        "nonrigid": False,
        "diameter_px": 16,
        "threshold_scaling": 1.0,
        "aspect_max": 2.0,
        "max_overlap": 0.45,
        "min_area_px": 40,
        "neuropil_extract": True,
        "inner_neuropil_radius": 2,
        "min_neuropil_pixels": 80,
        "int_thr_max": 0.10,
        "int_thr_std": 0.10,
        "norm_p_low": 5.0,
        "norm_p_high": 99.5,
        "delete_temp_movie": True,
    },
    "presegmentation_config": {
        "enabled": True,
        "target_mode": "soma",
        "pixel_size_um": REPORT_PIXEL_SIZE_UM,
        "fps_hz": REPORT_FPS_HZ,
        "norm_p_low": 5.0,
        "norm_p_high": 99.5,
    },
    "selection_config": {
        "min_largest_cc_area": 60,
        "top_percent": 0.10,
        "polish_selected_masks": True,
        "erode_pixels": 1,
        "analysis_use_top_percent": False,
    },
}




# =============================================================================
# 1) SMALL HELPERS
# =============================================================================

def get_patch_used(sample_mode: str):
    if sample_mode == "XY":
        patch_x = PATCH_XY * 2
        patch_y = PATCH_XY
        patch_t = PATCH_T
    elif sample_mode == "T":
        patch_x = PATCH_XY
        patch_y = PATCH_XY
        patch_t = PATCH_T * 2
    elif sample_mode == "N2V":
        patch_x = PATCH_XY
        patch_y = PATCH_XY
        patch_t = PATCH_T
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")
    return patch_x, patch_y, patch_t


def get_default_iter_params(iter_index: int) -> dict:
    sample_mode = "XY" if int(iter_index) == 0 else "T"
    n_epochs = 15 if int(iter_index) == 0 else 30
    patch_x, patch_y, patch_t = get_patch_used(sample_mode)
    return {
        "sample_mode": sample_mode,
        "n_epochs": int(n_epochs),
        "patch_x": int(patch_x),
        "patch_y": int(patch_y),
        "patch_t": int(patch_t),
        "batch_size": int(TRAIN_BATCH_SIZE_DEFAULT),
    }


def get_patch_minimums(sample_mode: str) -> tuple[int, int, int]:
    mode = str(sample_mode).strip().upper()
    if mode == "XY":
        return 16, 8, 8
    if mode == "T":
        return 8, 8, 16
    if mode == "N2V":
        return 8, 8, 8
    raise ValueError(f"Unknown sample_mode: {sample_mode}")


def _normalize_shape_to_thw(shape) -> tuple[int, int, int] | None:
    if shape is None:
        return None
    if len(shape) == 2:
        return 2, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    return int(shape[-3]), int(shape[-2]), int(shape[-1])


def _coerce_int(value, fallback: int):
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _clamp_int(value: int, low: int, high: int) -> int:
    low = int(low)
    high = int(high)
    if high < low:
        high = low
    return max(low, min(int(value), high))


def _prefer_even(value: int, low: int, high: int) -> int:
    value = int(value)
    low = int(low)
    high = int(high)
    if value % 2 == 0:
        return value
    if value - 1 >= low:
        return value - 1
    if value + 1 <= high:
        return value + 1
    return value


def get_patch_bounds(sample_mode: str, datasets_path: str) -> dict:
    default_x, default_y, default_t = get_patch_used(sample_mode)
    min_x, min_y, min_t = get_patch_minimums(sample_mode)
    shape_thw = _normalize_shape_to_thw(get_first_tif_shape(datasets_path))
    if shape_thw is None:
        max_x = max(default_x * 2, default_x)
        max_y = max(default_y * 2, default_y)
        max_t = max(default_t * 2, default_t)
    else:
        whole_t, whole_y, whole_x = shape_thw
        max_x = max(1, int(whole_x) - 1)
        max_y = max(1, int(whole_y) - 1)
        max_t = max(1, int(whole_t) - 1)
    return {
        "patch_x": [min(min_x, max_x), max_x],
        "patch_y": [min(min_y, max_y), max_y],
        "patch_t": [min(min_t, max_t), max_t],
    }


def get_batch_size_bounds(patch_x: int, patch_y: int, patch_t: int) -> tuple[int, int]:
    low, high = int(TRAIN_BATCH_SIZE_BOUNDS[0]), int(TRAIN_BATCH_SIZE_BOUNDS[1])
    baseline_volume = max(1, int(PATCH_XY) * int(PATCH_XY) * int(PATCH_T) * 2)
    current_volume = max(1, int(patch_x) * int(patch_y) * int(patch_t))
    memory_cap = max(low, int(round(TRAIN_BATCH_SIZE_DEFAULT * baseline_volume / current_volume)))
    return low, min(high, memory_cap)


def clamp_patch_to_data(
    patch_x: int,
    patch_y: int,
    patch_t: int,
    datasets_path: str,
    sample_mode: str,
):
    bounds = get_patch_bounds(sample_mode=sample_mode, datasets_path=datasets_path)
    safe_x = _clamp_int(patch_x, bounds["patch_x"][0], bounds["patch_x"][1])
    safe_y = _clamp_int(patch_y, bounds["patch_y"][0], bounds["patch_y"][1])
    safe_t = _clamp_int(patch_t, bounds["patch_t"][0], bounds["patch_t"][1])

    mode = str(sample_mode).strip().upper()
    if mode == "XY":
        safe_x = _prefer_even(safe_x, bounds["patch_x"][0], bounds["patch_x"][1])
    elif mode == "T":
        safe_t = _prefer_even(safe_t, bounds["patch_t"][0], bounds["patch_t"][1])

    return int(safe_x), int(safe_y), int(safe_t)


def clamp_test_patch_to_data(
    patch_xy: int,
    patch_t: int,
    datasets_path: str,
) -> tuple[int, int]:
    shape_thw = _normalize_shape_to_thw(get_first_tif_shape(datasets_path))
    if shape_thw is None:
        return int(patch_xy), int(patch_t)

    whole_t, whole_y, whole_x = shape_thw
    max_xy = max(2, min(int(whole_x), int(whole_y)) - 1)
    max_t = max(2, int(whole_t) - 1)

    safe_xy = _prefer_even(_clamp_int(patch_xy, 2, max_xy), 2, max_xy)
    safe_t = _prefer_even(_clamp_int(patch_t, 2, max_t), 2, max_t)
    return int(safe_xy), int(safe_t)


def sanitize_train_params(params: dict, datasets_path: str) -> dict:
    sample_mode = str(params["sample_mode"]).strip().upper()
    n_epochs = _clamp_int(_coerce_int(params.get("n_epochs"), 30), 1, 300)
    patch_x, patch_y, patch_t = clamp_patch_to_data(
        patch_x=_coerce_int(params.get("patch_x"), get_patch_used(sample_mode)[0]),
        patch_y=_coerce_int(params.get("patch_y"), get_patch_used(sample_mode)[1]),
        patch_t=_coerce_int(params.get("patch_t"), get_patch_used(sample_mode)[2]),
        datasets_path=datasets_path,
        sample_mode=sample_mode,
    )
    batch_low, batch_high = get_batch_size_bounds(patch_x=patch_x, patch_y=patch_y, patch_t=patch_t)
    batch_size = _clamp_int(_coerce_int(params.get("batch_size"), TRAIN_BATCH_SIZE_DEFAULT), batch_low, batch_high)
    return {
        "sample_mode": sample_mode,
        "n_epochs": int(n_epochs),
        "patch_x": int(patch_x),
        "patch_y": int(patch_y),
        "patch_t": int(patch_t),
        "batch_size": int(batch_size),
    }


def get_advisor_target(target_defaults: dict, datasets_path: str) -> dict:
    target_defaults = sanitize_train_params(target_defaults, datasets_path)
    bounds = get_patch_bounds(sample_mode=target_defaults["sample_mode"], datasets_path=datasets_path)
    batch_bounds = get_batch_size_bounds(
        patch_x=target_defaults["patch_x"],
        patch_y=target_defaults["patch_y"],
        patch_t=target_defaults["patch_t"],
    )
    return {
        "sample_mode_locked": str(target_defaults["sample_mode"]).strip().upper(),
        "default_params": target_defaults,
        "bounds": {
            "n_epochs": [1, 300],
            "patch_x": bounds["patch_x"],
            "patch_y": bounds["patch_y"],
            "patch_t": bounds["patch_t"],
            "batch_size": [int(batch_bounds[0]), int(batch_bounds[1])],
        },
    }


def validate_apply_fields(suggestion: dict, target_defaults: dict, datasets_path: str):
    warnings = []
    accepted_fields = set()
    resolved = dict(target_defaults)

    def _maybe_parse_int(field_name: str):
        raw_value = suggestion.get(field_name)
        if raw_value is None:
            return None
        try:
            parsed = int(raw_value)
        except Exception:
            warnings.append(f"invalid {field_name}={raw_value}, fallback to default")
            return None
        accepted_fields.add(field_name)
        return parsed

    n_epochs_val = _maybe_parse_int("n_epochs")
    if n_epochs_val is not None:
        clamped = _clamp_int(n_epochs_val, 1, 300)
        if clamped != n_epochs_val:
            warnings.append(f"n_epochs={n_epochs_val} out of range, clamped to {clamped}")
        resolved["n_epochs"] = clamped

    patch_x_val = _maybe_parse_int("patch_x")
    patch_y_val = _maybe_parse_int("patch_y")
    patch_t_val = _maybe_parse_int("patch_t")
    desired_patch_x = resolved["patch_x"] if patch_x_val is None else patch_x_val
    desired_patch_y = resolved["patch_y"] if patch_y_val is None else patch_y_val
    desired_patch_t = resolved["patch_t"] if patch_t_val is None else patch_t_val
    safe_patch_x, safe_patch_y, safe_patch_t = clamp_patch_to_data(
        patch_x=desired_patch_x,
        patch_y=desired_patch_y,
        patch_t=desired_patch_t,
        datasets_path=datasets_path,
        sample_mode=resolved["sample_mode"],
    )
    if patch_x_val is not None and safe_patch_x != patch_x_val:
        warnings.append(f"patch_x={patch_x_val} adjusted to {safe_patch_x}")
    if patch_y_val is not None and safe_patch_y != patch_y_val:
        warnings.append(f"patch_y={patch_y_val} adjusted to {safe_patch_y}")
    if patch_t_val is not None and safe_patch_t != patch_t_val:
        warnings.append(f"patch_t={patch_t_val} adjusted to {safe_patch_t}")
    resolved["patch_x"] = safe_patch_x
    resolved["patch_y"] = safe_patch_y
    resolved["patch_t"] = safe_patch_t

    batch_size_val = _maybe_parse_int("batch_size")
    desired_batch_size = resolved["batch_size"] if batch_size_val is None else batch_size_val
    batch_low, batch_high = get_batch_size_bounds(
        patch_x=resolved["patch_x"],
        patch_y=resolved["patch_y"],
        patch_t=resolved["patch_t"],
    )
    safe_batch_size = _clamp_int(desired_batch_size, batch_low, batch_high)
    if batch_size_val is not None and safe_batch_size != batch_size_val:
        warnings.append(
            f"batch_size={batch_size_val} adjusted to {safe_batch_size} "
            f"for patch_volume={resolved['patch_x'] * resolved['patch_y'] * resolved['patch_t']}"
        )
    resolved["batch_size"] = safe_batch_size

    return resolved, warnings, accepted_fields


def get_forced_mock_response(iter_index: int):
    if not isinstance(FORCE_MOCK_SUGGESTIONS_BY_ITER, dict):
        return None
    payload = FORCE_MOCK_SUGGESTIONS_BY_ITER.get(iter_index)
    return payload if isinstance(payload, dict) else None


def normalize_pipeline_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in ("off", "shadow", "apply"):
        return "off"
    return mode


def normalize_backend_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in ("mock", "live"):
        return "mock"
    return mode


def resolve_advisor_backend(pipeline_mode: str, configured_backend: str) -> str:
    pipeline_mode = normalize_pipeline_mode(pipeline_mode)
    if pipeline_mode == "off":
        return "off"
    backend = normalize_backend_mode(configured_backend)
    if has_configured_api_key(api_key=ADVISOR_API_KEY):
        return "live"
    return backend


def list_tif_paths(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")],
        key=lambda p: p.name.lower(),
    )


def get_single_tif_path_strict(folder: str | Path) -> Path:
    tifs = list_tif_paths(folder)
    if len(tifs) == 0:
        raise FileNotFoundError(f"No tif found under: {folder}")
    if len(tifs) > 1:
        raise RuntimeError(f"Expected one tif under {folder}, found: {[p.name for p in tifs]}")
    return tifs[0]


def choose_training_tifs(tif_paths: list[Path], max_count: int) -> list[Path]:
    if not tif_paths:
        raise RuntimeError("Cannot select a denoise training subset from an empty tif list.")
    if max_count <= 0 or len(tif_paths) <= max_count:
        return list(tif_paths)
    if max_count == 1:
        return [tif_paths[0]]

    n = len(tif_paths)
    chosen_indexes = []
    seen = set()
    for i in range(max_count):
        idx = int(round(i * (n - 1) / (max_count - 1)))
        if idx not in seen:
            chosen_indexes.append(idx)
            seen.add(idx)
    for idx in range(n):
        if len(chosen_indexes) >= max_count:
            break
        if idx not in seen:
            chosen_indexes.append(idx)
            seen.add(idx)
    return [tif_paths[idx] for idx in sorted(chosen_indexes)]


def tif_identity(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def tif_selection_fingerprint(paths: list[Path]) -> str:
    payload = [tif_identity(path) for path in paths]
    return compute_fingerprint({"tifs": payload})[:12]


def link_or_copy_tif(src: str | Path, dst: str | Path) -> Path:
    src = Path(src).resolve()
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return dst
            dst.unlink()
        except OSError:
            dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def materialize_tif_subset(src_paths: list[Path], target_dir: str | Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        link_or_copy_tif(src, target_dir / src.name)
    return target_dir


def get_first_tif_shape(datasets_path: str | Path):
    try:
        tif_list = list_tif_paths(datasets_path)
    except FileNotFoundError:
        return None
    if not tif_list:
        return None
    with tiff.TiffFile(str(tif_list[0])) as tf:
        shape = tf.series[0].shape
    return shape


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def read_json_if_exists(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_fingerprint(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_artifact_tag(folder_name: str, max_prefix: int = 24) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(folder_name))
    safe = safe.strip("_") or "folder"
    short = safe[:max_prefix]
    digest = hashlib.sha1(str(folder_name).encode("utf-8")).hexdigest()[:8]
    return f"{short}_{digest}"


def save_stage_sidecar(stage_dir: str | Path, sidecar_name: str, stage_name: str, params: dict):
    payload = {
        "stage": stage_name,
        "params": params,
        "config_fingerprint": compute_fingerprint(params),
    }
    write_json(Path(stage_dir) / sidecar_name, payload)


def can_skip_by_sidecar(stage_dir: str | Path, sidecar_name: str, expected_params: dict) -> bool:
    payload = read_json_if_exists(Path(stage_dir) / sidecar_name)
    if payload is None:
        return False
    return payload.get("config_fingerprint") == compute_fingerprint(expected_params)


def inspect_sidecar_match(stage_dir: str | Path, sidecar_name: str, expected_params: dict) -> dict:
    sidecar_path = Path(stage_dir) / sidecar_name
    payload = read_json_if_exists(sidecar_path)
    expected_fp = compute_fingerprint(expected_params)
    if payload is None:
        return {
            "sidecar_path": str(sidecar_path),
            "sidecar_exists": False,
            "expected_fingerprint": expected_fp,
            "sidecar_fingerprint": None,
            "fingerprint_match": False,
            "reason": "sidecar_missing",
        }
    sidecar_fp = payload.get("config_fingerprint")
    match = isinstance(sidecar_fp, str) and sidecar_fp == expected_fp
    return {
        "sidecar_path": str(sidecar_path),
        "sidecar_exists": True,
        "expected_fingerprint": expected_fp,
        "sidecar_fingerprint": sidecar_fp,
        "fingerprint_match": bool(match),
        "reason": "match" if match else "fingerprint_mismatch",
    }


def train_one_model_if_needed(
    denoise_datasets_path: str,
    folder_name: str,
    iiii: int,
    effective_params: dict,
    train_pth_dir: str | Path,
    training_selection: list[dict] | None = None,
) -> dict:
    sample_mode = str(effective_params["sample_mode"])
    n_epochs = int(effective_params["n_epochs"])
    patch_x = int(effective_params["patch_x"])
    patch_y = int(effective_params["patch_y"])
    patch_t = int(effective_params["patch_t"])
    batch_size = int(effective_params["batch_size"])
    pth_name = (
        f"{folder_name}_iter{iiii}_{sample_mode}"
        f"_e{n_epochs}_x{patch_x}_y{patch_y}_t{patch_t}_b{batch_size}"
    )
    train_pth_dir = Path(train_pth_dir)
    train_model_dir = train_pth_dir / pth_name
    runtime_path = train_model_dir / "train.runtime.json"
    print(f"\n\033[1;31m[PTH NAME]\033[0m \033[1;31m{train_model_dir}\033[0m")

    train_params = {
        "datasets_path": str(denoise_datasets_path),
        "iter_index": int(iiii),
        "sample_mode": str(sample_mode),
        "n_epochs": int(n_epochs),
        "patch_x": int(patch_x),
        "patch_y": int(patch_y),
        "patch_t": int(patch_t),
        "batch_size": int(batch_size),
        "gpu": str(GPU),
        "training_selection": training_selection or [],
    }

    has_pth = has_pth_file(str(train_model_dir))
    train_skip_info = inspect_sidecar_match(train_model_dir, "train.params.json", train_params)
    print(
        f"[TRAIN-CHECK] iter={iiii} has_pth={has_pth} "
        f"sidecar_exists={train_skip_info['sidecar_exists']} "
        f"fp_match={train_skip_info['fingerprint_match']} reason={train_skip_info['reason']}"
    )
    if has_pth and train_skip_info["fingerprint_match"]:
        print(f"\033[93m[SKIP]\033[0m Found existing .pth with matching sidecar in: {train_model_dir}")
        runtime_summary = read_json_if_exists(runtime_path) or {
            "schema_version": "stage_runtime.v1",
            "stage": "train_deepcad",
            "executed": False,
            "skipped": True,
            "reuse_reason": "matched_train_sidecar",
            "params": train_params,
        }
        return {
            "pth_name": pth_name,
            "pth_dir": str(train_pth_dir),
            "train_params": train_params,
            "runtime_summary": runtime_summary,
        }

    train_start = time.time()
    _pth_path = train_deepcad(
        datasets_path=denoise_datasets_path,
        pth_dir=str(train_pth_dir),
        pth_name=pth_name,
        n_epochs=n_epochs,
        patch_x=patch_x,
        patch_y=patch_y,
        patch_t=patch_t,
        overlap_factor=OVERLAP_FACTOR,
        gpu=GPU,
        num_workers=NUM_WORKERS,
        fmap=FMAP,
        scale_factor=SCALE_FACTOR,
        train_datasets_size=TRAIN_DATASETS_SIZE,
        select_img_num=SELECT_IMG_NUM,
        sample_mode=sample_mode,
        batch_size=batch_size,
    )
    train_wall_time_sec = float(time.time() - train_start)
    save_stage_sidecar(train_model_dir, "train.params.json", "train_deepcad", train_params)
    runtime_summary = {
        "schema_version": "stage_runtime.v1",
        "stage": "train_deepcad",
        "executed": True,
        "skipped": False,
        "wall_time_sec": train_wall_time_sec,
        "seconds_per_epoch": train_wall_time_sec / max(1, n_epochs),
        "params": train_params,
        "pth_path": str(_pth_path),
    }
    write_json(runtime_path, runtime_summary)
    return {
        "pth_name": pth_name,
        "pth_dir": str(train_pth_dir),
        "train_params": train_params,
        "runtime_summary": runtime_summary,
    }


# =============================================================================
# 2) MAIN PIPELINE FOR ONE FOLDER
# =============================================================================

def clone_llm_artifacts(shared_llm_dir: str | Path, target_llm_dir: str | Path) -> None:
    shared_llm_dir = Path(shared_llm_dir)
    target_llm_dir = Path(target_llm_dir)
    target_llm_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "mode.json",
        "request.json",
        "response_raw.json",
        "suggestion_validated.json",
        "applied_diff.json",
        "continue_iteration_record.json",
        "apply_warnings.json",
        "forced_mock_response.json",
        "forced_mock_response_disabled.json",
    ):
        src = shared_llm_dir / name
        if src.exists():
            shutil.copy2(src, target_llm_dir / name)


def init_stack_state(
    folder_name: str,
    folder_tag: str,
    raw_tif_path: Path,
    folder_results_root: Path,
    all_raw_tifs: list[Path],
    training_raw_tifs: list[Path],
    pipeline_mode: str,
    advisor_backend: str,
) -> dict:
    stack_tag = make_artifact_tag(raw_tif_path.stem, max_prefix=32)
    run_root = folder_results_root / stack_tag
    raw_input_dir = run_root / "raw_input"
    materialize_tif_subset([raw_tif_path], raw_input_dir)

    manifests_dir = run_root / "manifests"
    iterations_root = run_root / "iterations"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    iterations_root.mkdir(parents=True, exist_ok=True)

    raw_metrics_dir = run_root / "metrics" / "input"
    raw_metrics_json_path = raw_metrics_dir / f"{raw_tif_path.stem}_metrics.json"

    def _infer_tiff_shape_thw(tif_path: Path) -> list[int] | None:
        try:
            with tiff.TiffFile(str(tif_path)) as tf:
                series = tf.series[0] if getattr(tf, "series", None) else None
                if series is None:
                    return None
                pages = getattr(series, "pages", None)
                if pages is not None and len(pages) > 1:
                    frame0 = pages[0].asarray()
                    frame0 = frame0.squeeze()
                    if frame0.ndim == 2:
                        return [int(len(pages)), int(frame0.shape[0]), int(frame0.shape[1])]
                shape = tuple(int(x) for x in series.shape)
                axes = str(getattr(series, "axes", "") or "").upper()
                if len(shape) == len(axes) and "Y" in axes and "X" in axes:
                    y_idx = axes.index("Y")
                    x_idx = axes.index("X")
                    frames = 1
                    for idx, dim in enumerate(shape):
                        if idx not in {y_idx, x_idx}:
                            frames *= int(dim)
                    return [int(frames), int(shape[y_idx]), int(shape[x_idx])]
                if len(shape) == 3 and shape[1] >= 16 and shape[2] >= 16:
                    return [int(shape[0]), int(shape[1]), int(shape[2])]
        except Exception:
            return None
        return None

    if raw_metrics_json_path.exists():
        raw_metrics = read_json_if_exists(raw_metrics_json_path)
        raw_metrics_mask_path = Path(
            str((raw_metrics or {}).get("artifacts", {}).get("snr_roi_mask_npy") or "")
        )
        expected_shape_thw = _infer_tiff_shape_thw(raw_tif_path)
        cached_shape_thw = (raw_metrics or {}).get("data_summary", {}).get("shape_thw")
        try:
            cached_shape_norm = [int(x) for x in cached_shape_thw] if isinstance(cached_shape_thw, (list, tuple)) else None
        except Exception:
            cached_shape_norm = None
        shape_mismatch = (
            expected_shape_thw is not None
            and cached_shape_norm != expected_shape_thw
        )
        if (
            raw_metrics is None
            or raw_metrics.get("schema_version") != "pipeline_metrics.v2"
            or not raw_metrics_mask_path.exists()
            or shape_mismatch
        ):
            raw_metrics = compute_metrics_for_tif(raw_tif_path, raw_metrics_dir)
    else:
        raw_metrics = compute_metrics_for_tif(raw_tif_path, raw_metrics_dir)

    training_policy = {
        "scope": "input_subfolder_shared_model",
        "selection_policy": "sorted_even_subset",
        "max_tifs_for_model": int(TRAIN_MAX_TIFS_FOR_MODEL),
        "all_tifs": [tif_identity(path) for path in all_raw_tifs],
        "selected_tifs": [tif_identity(path) for path in training_raw_tifs],
        "same_modality_recommendation": (
            "All tif files in the same input subfolder should be as similar as possible "
            "in data type, imaging modality, noise profile, and acquisition settings."
        ),
    }

    pipeline_manifest = {
        "schema_version": "pipeline_manifest.v2",
        "folder_name": folder_name,
        "folder_tag": folder_tag,
        "stack_name": raw_tif_path.name,
        "stack_tag": stack_tag,
        "raw_tif_path": str(raw_tif_path),
        "is_cell_data": bool(IS_CELL_DATA),
        "dataset_profile": DATASET_PROFILE,
        "pipeline_llm_mode": pipeline_mode,
        "advisor_backend": advisor_backend,
        "iter_num": int(iter_num),
        "output_root": str(run_root),
        "output_subfolder_name": stack_tag,
        "raw_metrics_json": str(raw_metrics_json_path),
        "denoise_training_policy": training_policy,
        "iterations": [],
    }
    final_used_params = {
        "schema_version": "final_used_params.v2",
        "folder_name": folder_name,
        "folder_tag": folder_tag,
        "stack_name": raw_tif_path.name,
        "stack_tag": stack_tag,
        "is_cell_data": bool(IS_CELL_DATA),
        "dataset_profile": DATASET_PROFILE,
        "pipeline_llm_mode": pipeline_mode,
        "advisor_backend": advisor_backend,
        "output_root": str(run_root),
        "output_subfolder_name": stack_tag,
        "denoise_training_policy": training_policy,
        "iterations": [],
    }
    return {
        "folder_name": folder_name,
        "folder_tag": folder_tag,
        "raw_tif_path": raw_tif_path,
        "raw_tif_name": raw_tif_path.name,
        "stack_tag": stack_tag,
        "run_root": run_root,
        "manifests_dir": manifests_dir,
        "iterations_root": iterations_root,
        "current_input_dir": raw_input_dir,
        "last_iter_denoise_tif_path": None,
        "raw_metrics": raw_metrics,
        "pipeline_manifest": pipeline_manifest,
        "final_used_params": final_used_params,
        "iteration_contexts": {},
    }


def run_one_folder(folder_name: str) -> None:
    raw_datasets_path = Path(directory_path).expanduser() / folder_name
    raw_tif_paths = list_tif_paths(raw_datasets_path)
    if not raw_tif_paths:
        raise FileNotFoundError(f"No tif found under: {raw_datasets_path}")

    folder_tag = make_artifact_tag(folder_name)
    folder_results_root = _folder_results_root(folder_name)
    shared_root = folder_results_root / "_shared"
    shared_train_pth_dir = shared_root / "pth_deepcad"
    shared_iterations_root = shared_root / "iterations"
    shared_train_pth_dir.mkdir(parents=True, exist_ok=True)
    shared_iterations_root.mkdir(parents=True, exist_ok=True)

    pipeline_mode = normalize_pipeline_mode(PIPELINE_LLM_MODE)
    advisor_backend = resolve_advisor_backend(pipeline_mode, ADVISOR_BACKEND)
    training_raw_tifs = choose_training_tifs(raw_tif_paths, TRAIN_MAX_TIFS_FOR_MODEL)
    training_names = {path.name for path in training_raw_tifs}

    print(
        f"[PIPELINE-MODE] folder={folder_name} folder_tag={folder_tag} "
        f"pipeline_mode={pipeline_mode} is_cell_data={IS_CELL_DATA} "
        f"dataset_profile={DATASET_PROFILE} advisor_backend={advisor_backend} "
        f"iter_num={iter_num} tif_count={len(raw_tif_paths)} "
        f"train_tif_count={len(training_raw_tifs)}"
    )
    print("[TRAIN-TIF-SELECTION] =>", [path.name for path in training_raw_tifs])
    print(
        "[TRAIN-TIF-GUIDANCE] TIFF files within the same input subfolder should be as similar "
        "as possible in modality/type, acquisition settings, and noise profile."
    )

    states = [
        init_stack_state(
            folder_name=folder_name,
            folder_tag=folder_tag,
            raw_tif_path=raw_tif_path,
            folder_results_root=folder_results_root,
            all_raw_tifs=raw_tif_paths,
            training_raw_tifs=training_raw_tifs,
            pipeline_mode=pipeline_mode,
            advisor_backend=advisor_backend,
        )
        for raw_tif_path in raw_tif_paths
    ]
    control_state = next((state for state in states if state["raw_tif_name"] in training_names), states[0])
    next_iter_overrides = {}

    for iiii in range(iter_num):
        selected_stage_tifs = [
            get_single_tif_path_strict(state["current_input_dir"])
            for state in states
            if state["raw_tif_name"] in training_names
        ]
        subset_hash = tif_selection_fingerprint(selected_stage_tifs)
        train_subset_dir = shared_root / "train_inputs" / f"iter_{iiii}_{subset_hash}"
        materialize_tif_subset(selected_stage_tifs, train_subset_dir)
        training_selection = [tif_identity(path) for path in selected_stage_tifs]

        default_params = sanitize_train_params(get_default_iter_params(iiii), train_subset_dir)
        effective_params = dict(default_params)
        param_source = "default"
        if iiii in next_iter_overrides:
            ov = next_iter_overrides[iiii]
            effective_params.update({k: v for k, v in ov.items() if k in effective_params})
            effective_params = sanitize_train_params(effective_params, train_subset_dir)
            param_source = ov.get("param_source", "llm_apply")
            print(
                f"[APPLY] iter={iiii} using shared prior suggestion override "
                f"sample_mode={effective_params['sample_mode']} "
                f"n_epochs={effective_params['n_epochs']} "
                f"patch=({effective_params['patch_x']},{effective_params['patch_y']},{effective_params['patch_t']}) "
                f"batch_size={effective_params['batch_size']}"
            )

        current_params = dict(effective_params)
        print(
            f"[ITER-PARAMS] folder={folder_name} iter={iiii} "
            f"sample_mode={current_params['sample_mode']} "
            f"n_epochs={current_params['n_epochs']} "
            f"patch=({current_params['patch_x']},{current_params['patch_y']},{current_params['patch_t']}) "
            f"batch_size={current_params['batch_size']} param_source={param_source} "
            f"train_subset={train_subset_dir}"
        )

        train_result = train_one_model_if_needed(
            denoise_datasets_path=str(train_subset_dir),
            folder_name=folder_tag,
            iiii=iiii,
            effective_params=current_params,
            train_pth_dir=shared_train_pth_dir,
            training_selection=training_selection,
        )
        test_deepcad_model = train_result["pth_name"]
        train_runtime_summary = train_result["runtime_summary"]
        test_pth_path = Path(train_result["pth_dir"])

        for state in states:
            run_root = Path(state["run_root"])
            iter_dir = state["iterations_root"] / f"iter_{iiii}"
            iter_metrics_dir = iter_dir / "metrics"
            iter_llm_dir = iter_dir / "llm"
            iter_metrics_dir.mkdir(parents=True, exist_ok=True)
            iter_llm_dir.mkdir(parents=True, exist_ok=True)
            write_json(iter_metrics_dir / "train_runtime.json", train_runtime_summary)

            state["final_used_params"]["iterations"].append(
                {
                    "iter_index": iiii,
                    "effective_params": current_params,
                    "param_source": param_source,
                    "shared_train_subset_dir": str(train_subset_dir),
                    "shared_train_model": str(test_pth_path / test_deepcad_model),
                }
            )

            test_output_root = run_root / "results_deepcad"
            demotion_root = run_root / "results_demotion"
            ensure_dir(str(test_output_root))
            ensure_dir(str(demotion_root))

            test_patch_xy, test_patch_t = clamp_test_patch_to_data(
                patch_xy=int(PATCH_XY_TEST),
                patch_t=int(PATCH_T_TEST),
                datasets_path=str(state["current_input_dir"]),
            )
            if test_patch_xy != int(PATCH_XY_TEST) or test_patch_t != int(PATCH_T_TEST):
                print(
                    f"[DENOISE-PATCH] stack={state['raw_tif_name']} iter={iiii} "
                    f"adjusted test patch from ({int(PATCH_XY_TEST)},{int(PATCH_T_TEST)}) "
                    f"to ({test_patch_xy},{test_patch_t})"
                )

            test_output_folder = f"{state['stack_tag']}_iter{iiii}"
            test_denoise_dir = test_output_root / f"{test_output_folder}_DeepCAD"
            denoise_stage_params = {
                "datasets_path": str(state["current_input_dir"]),
                "pth_dir": str(test_pth_path),
                "denoise_model": str(test_deepcad_model),
                "output_dir": str(test_output_root),
                "output_folder": str(test_output_folder),
                "patch_xy": int(test_patch_xy),
                "patch_t": int(test_patch_t),
                "overlap_factor": float(OVERLAP_FACTOR),
                "gpu": str(GPU),
                "num_workers": int(NUM_WORKERS),
                "fmap": int(FMAP),
                "scale_factor": int(SCALE_FACTOR),
                "test_datasize": int(TEST_DATASIZE),
                "shared_train_subset_dir": str(train_subset_dir),
                "raw_tif_name": state["raw_tif_name"],
            }

            denoise_skip_info = inspect_sidecar_match(test_denoise_dir, "denoise.params.json", denoise_stage_params)
            denoise_has_tif = has_valid_tif(test_denoise_dir)
            denoise_skip_decision_reason = "execute_missing_output_or_sidecar_mismatch"
            if denoise_has_tif and denoise_skip_info["fingerprint_match"]:
                denoise_skip_decision_reason = "skip_safe_reuse_matched_fingerprint"
            write_json(iter_metrics_dir / "denoise_skip_check.json", {
                "has_valid_tif": denoise_has_tif,
                "skip_decision_reason": denoise_skip_decision_reason,
                **denoise_skip_info,
            })
            print(
                f"[DENOISE-CHECK] stack={state['raw_tif_name']} iter={iiii} "
                f"has_tif={denoise_has_tif} sidecar_exists={denoise_skip_info['sidecar_exists']} "
                f"fp_match={denoise_skip_info['fingerprint_match']} reason={denoise_skip_info['reason']}"
            )
            if denoise_has_tif and denoise_skip_info["fingerprint_match"]:
                print(f"\033[93m[SKIP]\033[0m Found existing denoised tif with matching sidecar in: {test_denoise_dir}")
                output_path = str(test_denoise_dir)
            else:
                print(f"[EXECUTE] stack={state['raw_tif_name']} iter={iiii} run test_deepcad")
                output_path = test_deepcad(
                    datasets_path=str(state["current_input_dir"]),
                    pth_dir=str(test_pth_path),
                    denoise_model=test_deepcad_model,
                    output_dir=str(test_output_root),
                    output_folder=test_output_folder,
                    patch_xy=test_patch_xy,
                    patch_t=test_patch_t,
                    overlap_factor=OVERLAP_FACTOR,
                    gpu=GPU,
                    num_workers=NUM_WORKERS,
                    fmap=FMAP,
                    scale_factor=SCALE_FACTOR,
                    test_datasize=TEST_DATASIZE,
                )
                save_stage_sidecar(test_denoise_dir, "denoise.params.json", "test_deepcad", denoise_stage_params)

            denoise_tif_path = get_single_tif_path_strict(output_path)
            state["last_iter_denoise_tif_path"] = denoise_tif_path
            denoise_metrics = compute_metrics_for_tif(
                denoise_tif_path,
                iter_metrics_dir / "denoise",
                snr_reference_metrics=state["raw_metrics"],
            )
            raw_vs_denoise = compare_two_metrics(state["raw_metrics"], denoise_metrics)
            write_json(iter_metrics_dir / "comparison_raw_vs_denoise.json", raw_vs_denoise)

            current_stage_metrics = denoise_metrics
            current_stage_name = "denoise"
            raw_vs_current_comparison = raw_vs_denoise

            if iiii < iter_num - 1:
                demotion_input_path = output_path
                demotion_output_path = demotion_root / f"{state['stack_tag']}_iter{iiii}_demotion"
                ensure_dir(str(demotion_output_path))
                print(
                    f"[PATH-LEN] stack={state['raw_tif_name']} iter={iiii} "
                    f"demotion_out_len={len(str(demotion_output_path))} "
                    f"demotion_mask_len={len(str(demotion_output_path) + '_mask')}"
                )

                motion_stage_params = {
                    "datasets_path": str(state["current_input_dir"]),
                    "input_path": str(demotion_input_path),
                    "output_path": str(demotion_output_path),
                    "upstream_denoise_fingerprint": compute_fingerprint(denoise_stage_params),
                    "gpu": str(GPU),
                    "iteration_num": 2,
                    "max_frames": None,
                    "raw_tif_name": state["raw_tif_name"],
                }

                motion_skip_info = inspect_sidecar_match(demotion_output_path, "motion.params.json", motion_stage_params)
                motion_has_tif = has_valid_tif(demotion_output_path)
                motion_skip_decision_reason = "execute_missing_output_or_sidecar_mismatch"
                if motion_has_tif and motion_skip_info["fingerprint_match"]:
                    motion_skip_decision_reason = "skip_safe_reuse_matched_fingerprint"
                write_json(iter_metrics_dir / "motion_skip_check.json", {
                    "has_valid_tif": motion_has_tif,
                    "skip_decision_reason": motion_skip_decision_reason,
                    **motion_skip_info,
                })
                print(
                    f"[MOTION-CHECK] stack={state['raw_tif_name']} iter={iiii} "
                    f"has_tif={motion_has_tif} sidecar_exists={motion_skip_info['sidecar_exists']} "
                    f"fp_match={motion_skip_info['fingerprint_match']} reason={motion_skip_info['reason']}"
                )
                if motion_has_tif and motion_skip_info["fingerprint_match"]:
                    print(f"\033[93m[SKIP]\033[0m motion output exists with matching sidecar: {demotion_output_path}")
                else:
                    print(f"[EXECUTE] stack={state['raw_tif_name']} iter={iiii} run demotion_PyLoReg")
                    demotion_PyLoReg(
                        datasets_path=str(state["current_input_dir"]),
                        input_path=str(demotion_input_path),
                        output_path=str(demotion_output_path),
                        gpu=GPU,
                        iteration_num=2,
                        max_frames=None,
                    )
                    save_stage_sidecar(demotion_output_path, "motion.params.json", "demotion_PyLoReg", motion_stage_params)

                motion_tif_path = get_single_tif_path_strict(demotion_output_path)
                motion_metrics = compute_metrics_for_tif(
                    motion_tif_path,
                    iter_metrics_dir / "motion_corrected",
                    snr_reference_metrics=state["raw_metrics"],
                )
                raw_vs_motion = compare_two_metrics(state["raw_metrics"], motion_metrics)
                write_json(iter_metrics_dir / "comparison_raw_vs_motion.json", raw_vs_motion)

                current_stage_metrics = motion_metrics
                current_stage_name = "motion_corrected"
                raw_vs_current_comparison = raw_vs_motion
                state["current_input_dir"] = demotion_output_path

            if iiii == iter_num - 1:
                raw_vs_final = compare_two_metrics(state["raw_metrics"], current_stage_metrics)
                write_json(iter_metrics_dir / "comparison_raw_vs_final.json", raw_vs_final)

            state["iteration_contexts"][iiii] = {
                "iter_metrics_dir": iter_metrics_dir,
                "iter_llm_dir": iter_llm_dir,
                "current_stage_metrics": current_stage_metrics,
                "current_stage_name": current_stage_name,
                "raw_vs_current_comparison": raw_vs_current_comparison,
            }

        target_iter_index = iiii + 1 if iiii < iter_num - 1 else None
        target_datasets_path = control_state["current_input_dir"] if target_iter_index is not None else train_subset_dir
        target_defaults = sanitize_train_params(
            get_default_iter_params(iiii if target_iter_index is None else target_iter_index),
            target_datasets_path,
        )
        advisor_target = get_advisor_target(target_defaults=target_defaults, datasets_path=target_datasets_path)
        advisor_target["target_iter"] = target_iter_index
        advisor_target["mode_policy"] = {
            "iter_0": "XY",
            "iter_ge_1": "T",
        }

        shared_llm_dir = shared_iterations_root / f"iter_{iiii}" / "llm"
        shared_llm_dir.mkdir(parents=True, exist_ok=True)
        llm_mode_payload = {
            "pipeline_mode": pipeline_mode,
            "advisor_backend": advisor_backend,
            "backend_effective": advisor_backend,
            "advisor_model_config": ADVISOR_MODEL,
            "advisor_base_url_config": ADVISOR_BASE_URL,
            "iter_index": iiii,
            "shared_training": {
                "enabled": True,
                "train_subset_dir": str(train_subset_dir),
                "selected_tifs": training_selection,
                "reference_stack_for_llm": control_state["raw_tif_name"],
            },
            "execution_policy": (
                "v2 apply-only-for(n_epochs,patch_x,patch_y,patch_t,batch_size); "
                "sample_mode locked to default schedule(iter0=XY,iter>=1=T); "
                "continue_iteration is record-only"
            ),
        }
        write_json(shared_llm_dir / "mode.json", llm_mode_payload)
        print(
            f"[LLM-CHECK] iter={iiii} pipeline_mode={pipeline_mode} "
            f"advisor_backend={advisor_backend} "
            f"(shared training: one suggestion per input subfolder; v2 apply-only fields unchanged)"
        )

        control_iter_context = control_state["iteration_contexts"][iiii]
        llm_context = {
            "folder_name": folder_name,
            "stack_name": control_state["raw_tif_name"],
            "iter_index": iiii,
            "current_params": current_params,
            "current_runtime": {
                "train": train_runtime_summary,
            },
            "suggestion_target": advisor_target,
            "metrics": control_iter_context["current_stage_metrics"],
            "comparisons": {
                f"raw_vs_{control_iter_context['current_stage_name']}": control_iter_context["raw_vs_current_comparison"],
            },
            "multi_tif_training": {
                "enabled": True,
                "all_tifs": [path.name for path in raw_tif_paths],
                "selected_training_tifs": [path.name for path in training_raw_tifs],
                "reference_stack_for_llm": control_state["raw_tif_name"],
            },
        }
        if iiii == iter_num - 1:
            llm_context["comparisons"]["raw_vs_final"] = compare_two_metrics(
                control_state["raw_metrics"],
                control_iter_context["current_stage_metrics"],
            )

        if pipeline_mode == "off":
            off_defaults = advisor_target["default_params"]
            llm_suggestion = {
                "schema_version": "llm_advisor.suggestion.v2",
                "n_epochs": off_defaults["n_epochs"],
                "patch_x": off_defaults["patch_x"],
                "patch_y": off_defaults["patch_y"],
                "patch_t": off_defaults["patch_t"],
                "batch_size": off_defaults["batch_size"],
                "continue_iteration": True,
                "reason": "pipeline llm mode off",
                "meta": {
                    "advisor_mode": "off",
                    "used_fallback": False,
                    "fallback_reason": None,
                    "validation_errors": [],
                    "timestamp_utc": now_tag(),
                },
            }
            write_json(shared_llm_dir / "request.json", {"mode": "off", "context": llm_context})
            write_json(shared_llm_dir / "response_raw.json", {"status": "off_mode", "reason": "advisor not called"})
            write_json(shared_llm_dir / "suggestion_validated.json", llm_suggestion)
        else:
            forced_mock_response = get_forced_mock_response(iiii) if advisor_backend == "mock" else None
            if advisor_backend == "live" and FORCE_MOCK_SUGGESTIONS_BY_ITER:
                write_json(
                    shared_llm_dir / "forced_mock_response_disabled.json",
                    {
                        "backend_effective": "live",
                        "reason": "force mock suggestions disabled when advisor backend is live",
                    },
                )
            if forced_mock_response is not None:
                print(f"[LLM-MOCK-OVERRIDE] iter={iiii} using forced mock suggestion override")
                write_json(shared_llm_dir / "forced_mock_response.json", forced_mock_response)
            llm_suggestion = get_llm_suggestion(
                context=llm_context,
                mode=advisor_backend,
                api_key=ADVISOR_API_KEY,
                model=ADVISOR_MODEL,
                base_url=ADVISOR_BASE_URL,
                timeout_s=ADVISOR_TIMEOUT_S,
                logs_dir=shared_llm_dir,
                mock_response=forced_mock_response,
            )
        print(
            f"[LLM-RESULT] folder={folder_name} iter={iiii} "
            f"reference_stack={control_state['raw_tif_name']} "
            f"suggested_n_epochs={llm_suggestion.get('n_epochs')} "
            f"suggested_patch=({llm_suggestion.get('patch_x')},{llm_suggestion.get('patch_y')},"
            f"{llm_suggestion.get('patch_t')}) "
            f"suggested_batch_size={llm_suggestion.get('batch_size')} "
            f"continue_iteration(record_only)={llm_suggestion.get('continue_iteration')}"
        )

        applied_changes = {}
        suggestion_target_iter = None
        applied_to_iter = None
        not_applied_reason = None
        apply_warnings = []
        if pipeline_mode == "apply" and iiii < iter_num - 1:
            suggestion_target_iter = iiii + 1
            next_default = dict(advisor_target["default_params"])
            next_effective, warnings, accepted_fields = validate_apply_fields(
                llm_suggestion,
                target_defaults=next_default,
                datasets_path=target_datasets_path,
            )
            apply_warnings.extend(warnings)
            next_source = "llm_apply" if accepted_fields else "fallback"
            next_iter_overrides[iiii + 1] = {
                **next_effective,
                "param_source": next_source,
            }

            for field_name in ("n_epochs", "patch_x", "patch_y", "patch_t", "batch_size"):
                if next_effective[field_name] != next_default[field_name]:
                    applied_changes[field_name] = {
                        "from": next_default[field_name],
                        "to": next_effective[field_name],
                        "target_iter": iiii + 1,
                    }
            if applied_changes:
                applied_to_iter = iiii + 1
            else:
                not_applied_reason = "suggestion_matches_default_or_fallback_to_default"
            if apply_warnings:
                print(f"\033[93m[WARN]\033[0m iter={iiii} apply warnings: {' | '.join(apply_warnings)}")
            print(
                f"[APPLY-PLAN] from iter={iiii} -> iter={iiii+1} "
                f"default={next_default} effective={next_effective}"
            )
        elif pipeline_mode == "apply" and iiii == iter_num - 1:
            suggestion_target_iter = None
            applied_to_iter = None
            not_applied_reason = "no_next_iteration"
            print(f"[APPLY-PLAN] iter={iiii} is last iter; suggestion recorded but no next-iter apply.")
        else:
            not_applied_reason = "pipeline_mode_not_apply"

        applied_diff = {
            "suggestion_target_iter": suggestion_target_iter,
            "applied_to_iter": applied_to_iter,
            "not_applied_reason": not_applied_reason,
            "changes": applied_changes,
        }

        write_json(shared_llm_dir / "applied_diff.json", applied_diff)
        write_json(shared_llm_dir / "continue_iteration_record.json", {"continue_iteration": llm_suggestion.get("continue_iteration")})
        if apply_warnings:
            write_json(shared_llm_dir / "apply_warnings.json", {"warnings": apply_warnings})

        for state in states:
            iter_context = state["iteration_contexts"][iiii]
            clone_llm_artifacts(shared_llm_dir, iter_context["iter_llm_dir"])
            state["pipeline_manifest"]["iterations"].append(
                {
                    "iter_index": iiii,
                    "current_params": current_params,
                    "param_source": param_source,
                    "train_runtime_summary": train_runtime_summary,
                    "shared_train_subset_dir": str(train_subset_dir),
                    "shared_train_model": str(test_pth_path / test_deepcad_model),
                    "llm_target_default_params": advisor_target["default_params"],
                    "current_stage": iter_context["current_stage_name"],
                    "metrics_dir": str(iter_context["iter_metrics_dir"]),
                    "llm_dir": str(iter_context["iter_llm_dir"]),
                    "shared_llm_dir": str(shared_llm_dir),
                    "llm_reference_stack": control_state["raw_tif_name"],
                    "llm_continue_iteration_recorded": llm_suggestion.get("continue_iteration"),
                    "suggestion_target_iter": suggestion_target_iter,
                    "applied_to_iter": applied_to_iter,
                    "not_applied_reason": not_applied_reason,
                }
            )

    for state in states:
        if state["last_iter_denoise_tif_path"] is None:
            raise RuntimeError(f"No final denoise tif found for {state['raw_tif_name']} to materialize final stack.")

        run_root = Path(state["run_root"])
        downstream_result = materialize_final_and_run_downstream(
            raw_stack_path=state["raw_tif_path"],
            final_stack_source_path=state["last_iter_denoise_tif_path"],
            output_root=run_root,
            dataset_profile=DATASET_PROFILE,
            downstream_config=DOWNSTREAM_CONFIG,
            final_source_semantic="last_iter_denoised_output",
        )
        state["pipeline_manifest"]["downstream"] = {
            "is_cell_data": bool(IS_CELL_DATA),
            "dataset_profile": DATASET_PROFILE,
            "cell_downstream_enabled": bool(IS_CELL_DATA),
            "final_stack_path": downstream_result.get("final_stack_path"),
            "final_stack_sidecar_path": downstream_result.get("final_stack_sidecar_path"),
            "segmentation_output_dir": downstream_result.get("segmentation_output_dir"),
            "downstream_subprocess_python": downstream_result.get("downstream_subprocess_python"),
            "downstream_subprocess_log_path": downstream_result.get("downstream_subprocess_log_path"),
            "downstream_subprocess_result_path": downstream_result.get("downstream_subprocess_result_path"),
            "backend_status_path": downstream_result.get("backend_status_path"),
            "run_status_path": downstream_result.get("run_status_path"),
            "summary_path": downstream_result.get("summary_path"),
            "comparison_path": downstream_result.get("comparison_path"),
            "requested_config_path": downstream_result.get("requested_config_path"),
            "effective_config_path": downstream_result.get("effective_config_path"),
            "presegmentation_profile_json": downstream_result.get("presegmentation_profile_json"),
            "presegmentation_suggested_config_json": downstream_result.get("presegmentation_suggested_config_json"),
            "presegmentation_report_html": downstream_result.get("presegmentation_report_html"),
        }

        write_json(state["manifests_dir"] / "pipeline_manifest.json", state["pipeline_manifest"])
        write_json(run_root / "final_used_params.json", state["final_used_params"])

        report_result = build_deterministic_report(
            run_root=run_root,
            try_pdf=REPORT_TRY_PDF,
            generate_overview_pngs=REPORT_GENERATE_OVERVIEW_PNGS,
            imaging_modality=REPORT_IMAGING_MODALITY,
            pixel_size_um=REPORT_PIXEL_SIZE_UM,
            fps_hz=REPORT_FPS_HZ,
            display_name_final=REPORT_DISPLAY_NAME_FINAL,
            report_embed_assets=REPORT_EMBED_ASSETS,
            report_inline_css=REPORT_INLINE_CSS,
            report_generate_pdf=REPORT_GENERATE_PDF,
            report_crop_scale_factor=REPORT_CROP_SCALE_FACTOR,
            report_kymograph_line_count=REPORT_KYMOGRAPH_LINE_COUNT,
            report_use_intermediate_sections=REPORT_USE_INTERMEDIATE_SECTIONS,
        )
        state["pipeline_manifest"]["report"] = {
            "report_data_json": report_result.get("report_data_json"),
            "report_manifest_json": report_result.get("report_manifest_json"),
            "report_html": report_result.get("report_html"),
            "report_print_html": report_result.get("report_print_html"),
            "report_pdf": report_result.get("report_pdf"),
            "overview_page1_png": report_result.get("overview_page1_png"),
            "overview_page2_png": report_result.get("overview_page2_png"),
            "config": {
                "imaging_modality": REPORT_IMAGING_MODALITY,
                "pixel_size_um": REPORT_PIXEL_SIZE_UM,
                "fps_hz": REPORT_FPS_HZ,
                "display_name_final": REPORT_DISPLAY_NAME_FINAL,
                "report_embed_assets": REPORT_EMBED_ASSETS,
                "report_inline_css": REPORT_INLINE_CSS,
                "report_generate_pdf": REPORT_GENERATE_PDF,
                "report_crop_scale_factor": REPORT_CROP_SCALE_FACTOR,
                "report_kymograph_line_count": REPORT_KYMOGRAPH_LINE_COUNT,
                "report_use_intermediate_sections": REPORT_USE_INTERMEDIATE_SECTIONS,
            },
        }
        write_json(state["manifests_dir"] / "pipeline_manifest.json", state["pipeline_manifest"])


# =============================================================================
# 3) MAIN LOOP WITH TRY/EXCEPT
# =============================================================================

def main():
    global directory_path, subfolder_list, RESULTS_PATH, PIPELINE_LLM_MODE
    global IS_CELL_DATA, GPU, DOWNSTREAM_ENV_NAME, REPORT_PIXEL_SIZE_UM, REPORT_FPS_HZ

    args = parse_cli_args()
    if args.input_dir:
        directory_path = str(Path(args.input_dir).expanduser())
    if args.output_dir:
        RESULTS_PATH = str(Path(args.output_dir).expanduser())
        _refresh_runtime_paths()
    if args.subfolders:
        subfolder_list = _parse_subfolder_list(args.subfolders)
    else:
        subfolder_list = _env_list("NEUROPILOT_SUBFOLDERS") or _discover_subfolders_safe(directory_path)
    if args.llm_mode:
        PIPELINE_LLM_MODE = str(args.llm_mode).strip().lower()
    if args.cell_data is not None:
        IS_CELL_DATA = bool(args.cell_data)
        _refresh_dataset_profile()
    if args.gpu:
        GPU = str(args.gpu).strip()
        os.environ["NEUROPILOT_GPU"] = GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    if args.um_per_pixel is not None:
        if float(args.um_per_pixel) <= 0:
            raise ValueError("--um-per-pixel must be positive.")
        REPORT_PIXEL_SIZE_UM = float(args.um_per_pixel)
        os.environ["NEUROPILOT_REPORT_PIXEL_SIZE_UM"] = str(REPORT_PIXEL_SIZE_UM)
        DOWNSTREAM_CONFIG.setdefault("presegmentation_config", {})["pixel_size_um"] = REPORT_PIXEL_SIZE_UM
    if args.frame_rate is not None:
        if float(args.frame_rate) <= 0:
            raise ValueError("--frame-rate must be positive.")
        REPORT_FPS_HZ = float(args.frame_rate)
        os.environ["NEUROPILOT_REPORT_FPS_HZ"] = str(REPORT_FPS_HZ)
        DOWNSTREAM_CONFIG.setdefault("segmentation_config", {})["fs_hz"] = REPORT_FPS_HZ
        DOWNSTREAM_CONFIG.setdefault("presegmentation_config", {})["fps_hz"] = REPORT_FPS_HZ
    if args.downstream_env:
        DOWNSTREAM_ENV_NAME = str(args.downstream_env).strip()
        os.environ["NEUROPILOT_DOWNSTREAM_ENV_NAME"] = DOWNSTREAM_ENV_NAME
        DOWNSTREAM_CONFIG.setdefault("runner_config", {})["env_name"] = DOWNSTREAM_ENV_NAME

    ensure_dir(str(Path(RESULTS_PATH).expanduser()))

    input_root = Path(directory_path).expanduser()
    pipeline_mode = normalize_pipeline_mode(PIPELINE_LLM_MODE)
    resolved_backend = resolve_advisor_backend(pipeline_mode, ADVISOR_BACKEND)
    print("[INPUT_ROOT] =>", input_root)
    print("[RESULTS_ROOT] =>", Path(RESULTS_PATH).expanduser())
    print("[SUBFOLDERS] =>", subfolder_list)
    print("[GPU] =>", GPU)
    print("[UM_PER_PIXEL] =>", REPORT_PIXEL_SIZE_UM)
    print("[FRAME_RATE_HZ] =>", REPORT_FPS_HZ)
    print("[DOWNSTREAM_ENV] =>", DOWNSTREAM_CONFIG.get("runner_config", {}).get("env_name"))
    print("[LLM_MODE] =>", pipeline_mode)
    print("[LLM_BACKEND_EFFECTIVE] =>", resolved_backend)

    if not input_root.exists():
        raise FileNotFoundError(
            f"Configured input root does not exist: {input_root}. "
            "Set NEUROPILOT_INPUT_DIR to your dataset root before running."
        )
    if not subfolder_list:
        loose_tifs = _discover_root_level_tifs(input_root)
        if loose_tifs:
            raise RuntimeError(
                "No dataset subfolders were found under the input root. "
                "The pipeline expects one child folder per dataset, with one or more TIFF files inside each child folder. "
                f"Found loose TIFF files at the root instead: {loose_tifs[:5]}. "
                "Run `python prepare_input_tiffs.py --input-dir <your_tif_folder>` first to rewrite and organize them into subfolders."
            )
        raise RuntimeError(
            "No input subfolders were selected. "
            "Set NEUROPILOT_SUBFOLDERS=folder_a,folder_b or place datasets under NEUROPILOT_INPUT_DIR."
        )

    for folder_name in subfolder_list:
        print("\n" + "#" * 90)
        print(f"[RUN] folder_name = {folder_name}")
        print("#" * 90)

        try:
            run_one_folder(folder_name=folder_name)
            print(f"\033[92m[DONE]\033[0m {folder_name}")

        except Exception as e:
            folder_logs_root = _folder_runtime_paths(folder_name)["logs"]
            ensure_dir(str(folder_logs_root))
            log_file = folder_logs_root / f"ERROR_{_folder_output_name(folder_name)}_{now_tag()}.log"
            safe_write_exception(str(log_file), e)
            print(f"\033[91m[ERROR]\033[0m {folder_name} failed. Log saved to: {log_file}")
            continue


if __name__ == "__main__":
    main()

