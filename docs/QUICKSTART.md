# Quickstart

## Input Layout

The main pipeline expects one child folder per dataset:

```text
/path/to/datasets/
  sample_a/
    sample_a_fov01.tif
    sample_a_fov02.tif
  sample_b/
    sample_b.tif
```

Each first-level dataset folder may contain one or more primary TIFF movies.

Within the same dataset folder, keep TIFF files as similar as possible in modality/type, acquisition settings, and noise profile. The pipeline trains one shared denoise model per dataset folder using a deterministic subset of its TIFF files, then processes each TIFF separately through denoising, registration, final-stack materialization, downstream analysis, and report generation.

By default, at most `4` TIFF files per dataset folder are used for the shared denoise training stage. Set `NEUROPILOT_TRAIN_MAX_TIFS=0` if you want to use all TIFF files in that folder for denoise training.

If your TIFF files are currently flat in one large folder, prepare them first:

```bash
python prepare_input_tiffs.py --input-dir /path/to/flat_tif_folder
```

## Repository Demo Data

The current demo layout uses two first-level dataset folders:

```text
demo_data/
  24h/
    CellVideo 01-1.tif
  spine/
    ju2df_5day_freemoving-male1-5day-image-pain 0.tif
```

The TIFF payloads are large, so the repository tracks the folder layout and distributes the movies outside normal git history.

From the repository root, a minimal demo run is:

```bash
python neuropilot_pipeline.py \
  --input-dir demo_data \
  --subfolders 24h,spine \
  --output-dir runs/demo_data \
  --non-cell-data \
  --llm-mode off \
  --GPU 0
```

The larger example movies are intended to be distributed as GitHub Release assets rather than committed into the repository history.

Release downloads:
- `https://github.com/LuoYuhong123/NeuroPilot/releases`

## Cell-Data Run

Run the main command from the `neuropilot` environment. If your downstream segmentation environment keeps the default name `suite2p`, you can omit `--downstream-env`.

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --cell-data \
  --llm-mode shadow \
  --downstream-env suite2p \
  --GPU 0
```

## Cell-Data Run With LLM Apply

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --cell-data \
  --llm-mode apply \
  --downstream-env suite2p \
  --GPU 0
```

Before using `--llm-mode shadow` or `--llm-mode apply`:
- set a real `OPENAI_API_KEY` in local `.env`, local `.env.example`, or your shell
- set a usable GPU index in `.env` with `NEUROPILOT_GPU=0` or pass `--GPU 0`
- no extra `.env` LLM mode variable is required
- if your downstream cell-data environment is not named `suite2p`, pass `--downstream-env your_env_name`

## Non-Cell Run

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --non-cell-data \
  --llm-mode off \
  --GPU 0
```

This non-cell path does not require the downstream segmentation environment.

## Restrict To Selected Datasets

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --subfolders "['sample_a','sample_b']" \
  --cell-data \
  --llm-mode off \
  --GPU 0
```

## What To Expect

The pipeline writes one dataset root under `--output-dir/<dataset_folder_name>/`, with shared training artifacts in `_shared/` and one per-TIFF run folder for metrics, iteration artifacts, final outputs, manifests, and an HTML report.

By default:
- all child folders are scanned if `--subfolders` is omitted
- loose TIFF files directly under `--input-dir` should be preprocessed first
- reports are generated as HTML, not PDF
- downstream segmentation is enabled unless `--non-cell-data` is used or `NEUROPILOT_IS_CELL_DATA=false` is set
- even for the full two-environment workflow, the main command is run from `neuropilot`
