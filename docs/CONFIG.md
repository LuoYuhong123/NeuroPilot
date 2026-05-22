# Config

## Main Entry Parameters

- `--input-dir`: root folder containing one child folder per dataset; each child folder may contain one or more TIFF files
- `--output-dir`: root folder where run artifacts will be written
- `--subfolders`: optional dataset subset, written as `sample_a,sample_b` or `['sample_a','sample_b']`
- `--llm-mode`: `off`, `shadow`, or `apply`
- `--cell-data`: enable downstream cell-style analysis; this is the published default behavior
- `--non-cell-data`: skip downstream cell-style analysis
- `--GPU`: GPU index or comma-separated GPU indices, for example `0` or `0,1`
- `--downstream-env`: downstream conda environment name for cell-data segmentation; default is `suite2p`

## Input Folder Rule

The main pipeline does not expect all TIFF files to be placed directly under `--input-dir`.

Instead:
- `--input-dir` should contain multiple child folders
- each child folder may contain one or more TIFF files
- TIFF files within the same child folder should be as similar as possible in imaging modality, acquisition settings, and noise profile
- the pipeline deterministically selects a subset of TIFF files in that child folder as the shared denoise training set
- after shared training, each TIFF continues through denoising, registration, final-stack materialization, downstream analysis, and reporting as its own per-stack run
- if your TIFF files are flat in one folder, first run `prepare_input_tiffs.py`

## Shared Denoise Training Subset

- `NEUROPILOT_TRAIN_MAX_TIFS` controls the maximum number of TIFF files per input child folder used to train the shared denoise model
- default: `4`
- set `NEUROPILOT_TRAIN_MAX_TIFS=0` to use all TIFF files in the child folder
- selection is deterministic and based on sorted filenames

## LLM Practical Behavior

- if `--llm-mode off` is used, the pipeline does not call the advisor
- if `--llm-mode shadow` or `--llm-mode apply` is used and a real `OPENAI_API_KEY` is available in local `.env`, local `.env.example`, or the shell, the advisor switches to live mode automatically
- you do not need to set `NEUROPILOT_PIPELINE_LLM_MODE` in `.env` when the CLI already provides `--llm-mode`

## RAG Practical Behavior

- `NEUROPILOT_RAG_ENABLED`: enables local retrieval context; default `true`
- `NEUROPILOT_RAG_TOP_K`: number of prior experiment records to pass to the advisor; default `5`
- `NEUROPILOT_RAG_RUNS_ROOTS`: optional semicolon-separated runs roots
- `NEUROPILOT_RAG_LITERATURE_ENABLED`: enables local literature retrieval; default `true`
- `NEUROPILOT_RAG_LITERATURE_CHUNKS`: optional path to `literature/index/literature_chunks.jsonl`
- `NEUROPILOT_RAG_LITERATURE_TOP_K`: number of literature chunks to pass to the advisor; default `5`
- `NEUROPILOT_RAG_LITERATURE_MAX_CHUNKS_PER_PAPER`: per-paper cap for literature evidence; default `2`

## Downstream Practical Behavior

- if neither `--cell-data` nor `--non-cell-data` is passed, the published default is `cell-data`
- full cell-data processing expects two environments, but the main command is still launched from `neumar`
- `--downstream-env` only matters for `--cell-data`
- if `--non-cell-data` is used, the pipeline skips the downstream cell-analysis stage and does not require the second environment
