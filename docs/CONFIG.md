# Config

## Main Entry Parameters

- `--input-dir`: root folder containing one child folder per dataset
- `--output-dir`: root folder where run artifacts will be written
- `--subfolders`: optional dataset subset, written as `sample_a,sample_b` or `['sample_a','sample_b']`
- `--llm-mode`: `off`, `shadow`, or `apply`
- `--cell-data`: enable downstream cell-style analysis
- `--non-cell-data`: skip downstream cell-style analysis
- `--GPU`: GPU index or comma-separated GPU indices, for example `0` or `0,1`
- `--downstream-env`: downstream conda environment name for cell-data segmentation; default is `suite2p`

## Input Folder Rule

The main pipeline does not expect all TIFF files to be placed directly under `--input-dir`.

Instead:
- `--input-dir` should contain multiple child folders
- each child folder should contain exactly one TIFF file
- if your TIFF files are flat in one folder, first run `prepare_input_tiffs.py`

## LLM Practical Behavior

- if `--llm-mode off` is used, the pipeline does not call the advisor
- if `--llm-mode shadow` or `--llm-mode apply` is used and `OPENAI_API_KEY` is available in local `.env` or the shell, the advisor switches to live mode automatically
- you do not need to set `NEUROPILOT_PIPELINE_LLM_MODE` in `.env` when the CLI already provides `--llm-mode`

## Downstream Practical Behavior

- full cell-data processing expects two environments, but the main command is still launched from `neumar`
- `--downstream-env` only matters for `--cell-data`
- if `--non-cell-data` is used, the pipeline skips the downstream cell-analysis stage and does not require the second environment
