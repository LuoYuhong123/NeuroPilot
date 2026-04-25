# NeuroPilot

NeuroPilot is a calcium-imaging processing pipeline built around iterative denoising and registration, followed by optional downstream segmentation / trace extraction and deterministic-first HTML reporting.

## Highlights

- iterative denoising-registration workflow rather than a single standalone registration step
- single-command pipeline entry through `neuropilot_pipeline.py`
- deterministic metrics, manifests, final stack materialization, and HTML reports
- optional live LLM advisor in `off`, `shadow`, or `apply` mode
- optional cell-style downstream analysis through a separate `suite2p` environment
- frozen `final_stack` semantic: the final stack is the last-iteration denoised output

## Repository Scope

This public snapshot is intentionally limited to code, configuration templates, required runtime assets, and usage documentation needed to install and run the pipeline.

Included:
- pipeline orchestration, denoising, registration, metrics, and report-generation code
- configuration templates and environment files
- optional downstream adapters and related scripts
- the default public PyLoReg checkpoint used by the registration stage
- one repository-tracked quick demo movie under `demo_data/quick_demo/quick_demo.tif`

Excluded:
- internal experiment outputs and temporary run folders
- manuscript materials and internal documentation drafts
- DeepCAD denoising pretrained weights
- private API credentials
- the optional `deepinterpolation/` backend
- large example TIFF movies, which should be distributed as GitHub Release assets rather than committed into git history

## Main Entry

```bash
python neuropilot_pipeline.py --input-dir /path/to/datasets --output-dir /path/to/run_output
```

Important input-layout rule:
- `--input-dir` should contain multiple child folders
- each child folder may contain one or more `.tif` or `.tiff` files
- TIFF files within the same child folder should be as similar as possible in modality/type, acquisition settings, and noise profile
- the pipeline deterministically selects a subset of TIFF files in each child folder as the shared denoise training set
- after shared denoise training, each TIFF in that child folder continues through denoising, registration, final-stack materialization, downstream analysis, and report generation as its own per-stack run
- if `--subfolders` is omitted, the pipeline scans and processes all first-level child folders
- if `--subfolders` is provided, only the named child folders are processed
- results are grouped under `--output-dir/<child_folder_name>/`, with `_shared/` training artifacts plus one per-TIFF run folder

Supported `--subfolders` forms:

```bash
--subfolders sample_a,sample_b
```

or

```bash
--subfolders "['sample_a','sample_b']"
```

The pipeline does not expect all TIFF files to be placed directly under `--input-dir` without subfolders. If your TIFF files are currently flat in one large folder, first run:

```bash
python prepare_input_tiffs.py --input-dir /path/to/flat_tif_folder
```

That helper script checks TIFF layout, rewrites non-page-stack TIFFs when needed, and places each TIFF into its own same-name subfolder as a safe default before the main pipeline is launched. You may later regroup similar TIFFs into the same child folder when you want them to share denoise training.

## Environments

- `environment-neumar.yml`: main GPU-enabled `neumar` environment for iterative denoising, registration, metrics, HTML report generation, and optional LLM advisor logic
- `environment-suite2p.yml`: default downstream segmentation environment for cell-data analysis
- the same environment files are intended for both Windows and Linux users; Linux-specific setup notes are listed below

For the complete cell-data pipeline, two environments are expected:
- `neumar` for the main pipeline entry
- a second downstream environment for segmentation / ROI selection

All user-facing pipeline commands are still launched from the `neumar` environment. The main entry switches to the downstream interpreter only when cell-data downstream analysis is needed.

## Installation

Linux shell examples are shown below. The same environment files can also be created on Windows with the same `conda env create -f ...` commands.

Main environment:

```bash
conda env create -f environment-neumar.yml
conda activate neumar
```

The published `neumar` environment uses mirrored conda channels for the scientific stack and installs the CUDA 12.1 PyTorch wheel through pip.

If `conda env create -f environment-neumar.yml` fails while mentioning `torchaudio` or another package that does not appear in `environment-neumar.yml`, first confirm that your local `environment-neumar.yml` really matches the published one, then inspect hidden conda constraints and retry with a clean cache or the classic solver:

```bash
grep -nE 'torchaudio|torchvision|pytorch|cuda|python' environment-neumar.yml
cat "$CONDA_PREFIX/conda-meta/pinned" 2>/dev/null
conda config --show-sources
conda config --show create_default_packages
conda config --show pinned_packages
conda clean --index-cache --tarballs --packages -y
CONDA_SOLVER=classic conda env create -f environment-neumar.yml
```

The published environment files intentionally avoid the Windows-only `pkgs/msys2` channel. If your Linux machine still shows unstable solving behavior after the checks above, replace the explicit mirror URLs with your preferred `conda-forge` / `pkgs/main` channels and retry.

Linux notes:
- tested target: Ubuntu-like systems with Python 3.10 and an NVIDIA driver compatible with CUDA 12.x
- if `cv2` later fails with `libGL.so.1` or `libgthread-2.0.so.0`, install:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

- if you are outside the default mirror region, you may replace the explicit mirror URLs in the environment files with your preferred `conda-forge` / `pkgs/main` channels before creating the environments
- after activation, a minimal sanity check is:

```bash
python -c "import torch, cv2, skimage, tifffile; print(torch.__version__); print(torch.cuda.is_available())"
```

On Windows, the published main environment and entry script apply two compatibility workarounds:
- set `KMP_DUPLICATE_LIB_OK=TRUE` to tolerate duplicate OpenMP runtimes from mixed scientific wheels
- bootstrap a `libomp140.x86_64.dll` filename alias inside the active conda environment when `libomp.dll` is already available

Default downstream environment:

```bash
conda env create -f environment-suite2p.yml
conda activate suite2p
```

If you prefer another environment name, create it with that name and pass it later through `--downstream-env your_env_name`.

Linux users may optionally point the main pipeline to an explicit downstream interpreter with:

```bash
export NEUROPILOT_DOWNSTREAM_PYTHON=/full/path/to/python
```

On Windows, the published downstream environment sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the common `libomp.dll` / `libiomp5md.dll` conflict seen when importing `suite2p` together with its pip-installed dependencies.

## Quick Start

Cell-data example first:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --cell-data \
  --llm-mode shadow \
  --downstream-env suite2p \
  --GPU 0
```

Cell-data with LLM apply:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --cell-data \
  --llm-mode apply \
  --downstream-env suite2p \
  --GPU 0
```

Before running `--llm-mode apply` or `--llm-mode shadow`:
- place a real `OPENAI_API_KEY` in local `.env`, local `.env.example`, or your shell environment
- set an available GPU index in `.env` through `NEUROPILOT_GPU=0` or pass `--GPU 0` directly
- you do not need to set `NEUROPILOT_PIPELINE_LLM_MODE` in `.env`
- if `--llm-mode off` is used, the pipeline ignores those LLM runtime settings and does not call the advisor
- if neither `--cell-data` nor `--non-cell-data` is passed, the published default is `cell-data`
- for cell-data runs, keep the downstream environment installed and pass `--downstream-env` only if its name is not the default `suite2p`

Minimal non-cell run:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --non-cell-data \
  --llm-mode off \
  --GPU 0
```

This non-cell path runs entirely inside `neumar` and does not require the downstream segmentation environment.

Restrict to specific datasets:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --subfolders "['sample_a','sample_b']" \
  --cell-data \
  --llm-mode off \
  --GPU 0
```

Expected input layout:

```text
/path/to/datasets/
  sample_a/
    sample_a_fov01.tif
    sample_a_fov02.tif
  sample_b/
    sample_b.tif
```

By default, the pipeline trains one shared denoise model per child folder using a deterministic subset of its TIFF files, then processes each TIFF separately. The default subset cap is `4` files per child folder; set `NEUROPILOT_TRAIN_MAX_TIFS=0` to use all TIFF files in that child folder for denoise training.

## Demo Data

This repository keeps one small quick demo in git for smoke testing:
- `demo_data/quick_demo/quick_demo.tif`

The current quick demo is an `80 x 448 x 448` center crop from `demo_data/raw_LEC/raw_LEC.tif`, chosen to preserve more field of view than the older 256x256 sample and reduce obvious blank borders after processing.

You can run that bundled demo directly from the repository root with:

```bash
python neuropilot_pipeline.py \
  --input-dir demo_data \
  --subfolders quick_demo \
  --output-dir runs/quick_demo \
  --non-cell-data \
  --llm-mode off \
  --GPU 0
```

The larger example movies should be uploaded as GitHub Release assets instead of tracked files in the repository history:
- `raw_CA1.tif`
- `raw_LEC.tif`

GitHub Releases page:
- `https://github.com/LuoYuhong123/NeuroPilot/releases`

After downloading those release assets, place them into the same dataset-subfolder layout used by the pipeline, for example:

```text
demo_data/
  raw_CA1/
    raw_CA1.tif
  raw_LEC/
    raw_LEC.tif
  quick_demo/
    quick_demo.tif
```

See `demo_data/README.md` for the quick-demo and release-asset convention.

## Configuration Notes

The main entry accepts these user-facing parameters:
- `--input-dir`: root folder containing one child folder per dataset; each child folder may contain one or more TIFF files
- `--output-dir`: root folder where dataset-specific result folders will be written
- `--subfolders`: optional dataset subset, written as `sample_a,sample_b` or `['sample_a','sample_b']`
- `--llm-mode`: `off`, `shadow`, or `apply`
- `--cell-data` / `--non-cell-data`: whether to enable downstream cell-style analysis; the published default is `cell-data`
- `--GPU`: GPU index or comma-separated GPU indices, for example `0` or `0,1`
- `--downstream-env`: optional downstream conda environment name for cell-data runs; default is `suite2p`

Useful environment override:
- `NEUROPILOT_TRAIN_MAX_TIFS`: maximum number of TIFF files per input child folder used for shared denoise training; default `4`, set to `0` to use all TIFF files

## Weights

The public snapshot includes the default PyLoReg registration checkpoint used by the iterative registration stage:
- `PyLoReg/PyLoReg_model/GM3_fn5_202511301551/gmflow_latest.pt`

DeepCAD denoising pretrained weights are intentionally not distributed here, because the denoising stage is retrained for each input dataset within the pipeline.

## Outputs

Key outputs are grouped per dataset under `--output-dir/<child_folder_name>/`:
- `_shared/pth_deepcad/`, `_shared/train_inputs/`, and `_shared/iterations/` for child-folder-level shared denoise-training artifacts
- one per-TIFF run folder, typically named after the TIFF stem
- `<stack_tag>/manifests/pipeline_manifest.json`
- `<stack_tag>/final_used_params.json`
- `<stack_tag>/iterations/iter_*/metrics/`
- `<stack_tag>/iterations/iter_*/llm/`
- `<stack_tag>/final/final_stack.tif`
- `<stack_tag>/final/final_stack_sidecar.json`
- `<stack_tag>/segmentation/` for cell-data runs
- `<stack_tag>/report/report.html`
- `<stack_tag>/report/report_print.html`
- `<stack_tag>/report/report_data.json`
- `<stack_tag>/report/report_manifest.json`

See `docs/OUTPUTS.md` for a fuller directory walkthrough.

## Included Docs

- `docs/INSTALL.md`
- `docs/QUICKSTART.md`
- `docs/CONFIG.md`
- `docs/OUTPUTS.md`
- `docs/TROUBLESHOOTING.md`
- `docs/THIRD_PARTY_NOTICES.md`

## Notes

- `deepinterpolation/` is not bundled here. The retained compatibility functions in `NeuMar_function.py` raise a clear import error if that optional backend is invoked.
- The public snapshot defaults to HTML reports only; PDF export is disabled unless you explicitly re-enable it via environment variables.

## License

This repository is distributed under the GNU General Public License v3.0 (GPL-3.0).

The repository-level GPL choice reflects the current public composition of the project:
- denoising components in `deepcad/` and related wrappers are derived from and modified from [DeepCAD-RT](https://github.com/cabooster/DeepCAD-RT), which is published under GPL-3.0
- optional downstream segmentation / trace extraction is designed to work with [suite2p](https://github.com/MouseLand/suite2p), which is also published under GPL-3.0
- original NeuroPilot registration, orchestration, metrics, and reporting code is released under the same repository license for compatibility with the redistributed GPL components

See `docs/THIRD_PARTY_NOTICES.md` for provenance notes and file-level caveats.

## Acknowledgements

NeuroPilot builds on the ideas and codebases of the calcium-imaging community, especially DeepCAD-RT for denoising and suite2p for downstream segmentation workflows. We are grateful to those upstream authors and retain their required notices in this repository.
