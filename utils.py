#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def model_exists_loose(pth_dir: str, pth_name: str) -> bool:
    """
    宽松检查：pth_dir 下存在同名文件/文件夹，或者任何以 pth_name 开头的项即认为存在。
    """
    p = Path(pth_dir)
    if not p.exists():
        return False
    for x in p.iterdir():
        if x.name == pth_name:
            return True
        if x.name.startswith(pth_name):
            return True
    return False



def has_pth_file(pth_dir: str) -> bool:
    if not pth_dir or not os.path.isdir(pth_dir):
        return False
    for f in os.listdir(pth_dir):
        if f.endswith(".pth"):
            return True
    return False


def has_valid_tif(folder: str) -> bool:
    """
    判断输出目录里是否已经存在有效 tif/tiff 结果。
    - 只要有任意一个 tif/tiff 且 size > 0，就认为已经生成过，可跳过。
    """
    if not folder or (not os.path.isdir(folder)):
        return False
    tifs = glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff"))
    for f in tifs:
        try:
            if os.path.getsize(f) > 0:
                return True
        except OSError:
            pass
    return False


def safe_write_exception(log_file: str, e: BaseException) -> None:
    """
    把异常和 traceback 写入 log_file
    """
    ensure_dir(str(Path(log_file).parent))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{now_tag()}] EXCEPTION:\n")
        f.write(repr(e) + "\n\n")
        f.write(traceback.format_exc())
        f.write("\n" + "=" * 80 + "\n")
