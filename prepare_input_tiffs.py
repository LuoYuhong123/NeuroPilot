#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile as tiff


def discover_tifs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
    )


def is_page_stack(path: Path) -> tuple[bool, tuple[int, ...] | None, int]:
    with tiff.TiffFile(path) as tf:
        shape = tf.series[0].shape
        n_pages = len(tf.pages)
    if shape is not None and len(shape) >= 3 and n_pages == int(shape[0]):
        return True, tuple(shape), n_pages
    return False, tuple(shape) if shape is not None else None, n_pages


def rewrite_or_copy_to_dataset_folder(
    tif_path: Path,
    output_root: Path,
    overwrite: bool,
    bigtiff: bool,
) -> dict:
    dataset_name = tif_path.stem
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / tif_path.name

    already_page_stack, shape, n_pages = is_page_stack(tif_path)
    record = {
        "source": str(tif_path),
        "dataset_name": dataset_name,
        "output_path": str(output_path),
        "original_shape": list(shape) if shape is not None else None,
        "original_page_count": int(n_pages),
        "already_page_stack": bool(already_page_stack),
        "action": None,
    }

    if output_path.exists() and not overwrite:
        record["action"] = "skip_existing"
        return record

    if already_page_stack:
        shutil.copy2(tif_path, output_path)
        record["action"] = "copied_page_stack"
        return record

    with tiff.TiffFile(tif_path) as tf:
        stack = np.stack([page.asarray() for page in tf.pages], axis=0)

    tiff.imwrite(
        output_path,
        stack,
        dtype=stack.dtype,
        bigtiff=bigtiff,
        photometric="minisblack",
    )
    record["action"] = "rewritten_to_page_stack"
    record["rewritten_shape"] = list(stack.shape)
    return record


def prepare_input_tiffs(
    input_dir: Path,
    output_dir: Path | None = None,
    recursive: bool = False,
    overwrite: bool = True,
    bigtiff: bool = True,
) -> dict:
    input_dir = input_dir.resolve()
    output_root = (output_dir.resolve() if output_dir is not None else input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    tif_files = discover_tifs(input_dir, recursive=recursive)
    if not tif_files:
        raise FileNotFoundError(f"No tif/tiff files found under: {input_dir}")

    results = []
    for tif_path in tif_files:
        results.append(
            rewrite_or_copy_to_dataset_folder(
                tif_path=tif_path,
                output_root=output_root,
                overwrite=overwrite,
                bigtiff=bigtiff,
            )
        )

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_root),
        "recursive": bool(recursive),
        "overwrite": bool(overwrite),
        "files_found": len(tif_files),
        "results": results,
    }

    summary_path = output_root / "prepare_input_tiffs.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a flat folder of TIFF files for NeuroPilot. "
            "Each TIFF is checked, rewritten to page-stack format when needed, "
            "and placed into its own same-name dataset subfolder."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder containing loose TIFF files before NeuroPilot preprocessing.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output root. Defaults to preparing in place under --input-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan nested folders under --input-dir.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing prepared outputs and skip files whose destination already exists.",
    )
    args = parser.parse_args()

    summary = prepare_input_tiffs(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        recursive=bool(args.recursive),
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
