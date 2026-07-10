# Install

## Requirements

The public snapshot is designed around a conda-based workflow.

Recommended assumptions:
- Python 3.10
- an NVIDIA GPU for the main `neuropilot` environment
- a separate downstream environment for cell-style analysis when `--cell-data` is used

Supported installation targets:
- Windows 10/11 with an NVIDIA GPU
- Linux workstations or servers, especially Ubuntu-like systems with CUDA-capable NVIDIA drivers

## Linux System Packages

The conda environments cover Python-level dependencies. On Linux, you may still need a few system libraries for OpenCV and image I/O:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

If your Linux machine does not use `apt`, install the equivalent packages through your distribution package manager.

## Main Environment

```bash
conda env create -f environment-neuropilot.yml
conda activate neuropilot
```

This environment file is tuned for mirrored conda channels plus a pip-installed CUDA 12.1 PyTorch wheel, which helps avoid the flaky `nvidia` conda channel on some Windows setups.

Important solver note:
- `environment-neuropilot.yml` does not request conda-side `torchaudio`
- if the solver reports `torchaudio` or another package that is absent from the environment file, check the current local YAML copy first
- other common causes include hidden conda pins, injected defaults, stale solver caches, or a `libmamba`-specific solve issue

Check with:

```bash
grep -nE 'torchaudio|torchvision|pytorch|cuda|python' environment-neuropilot.yml
cat "$CONDA_PREFIX/conda-meta/pinned" 2>/dev/null
conda config --show-sources
conda config --show create_default_packages
conda config --show pinned_packages
```

Then clear the cache and retry with the classic solver:

```bash
conda clean --index-cache --tarballs --packages -y
CONDA_SOLVER=classic conda env create -f environment-neuropilot.yml
```

Linux notes:
- the same `environment-neuropilot.yml` file is intended to work on Linux
- if you are outside the default mirror region, replace the explicit mirror URLs with your preferred `conda-forge` / `pkgs/main` channels before creating the environment
- the published environment files intentionally omit the Windows-only `pkgs/msys2` channel
- after activation, a quick sanity check is:

```bash
python -c "import torch, cv2, skimage, tifffile; print(torch.__version__); print(torch.cuda.is_available())"
```

On Windows, the published main environment and entry script apply two compatibility workarounds:
- set `KMP_DUPLICATE_LIB_OK=TRUE` to tolerate duplicate OpenMP runtimes from mixed scientific wheels
- create a local `libomp140.x86_64.dll` filename alias inside the active conda environment when only `libomp.dll` is present

Together these workarounds address the common `fbgemm.dll` and duplicate-OpenMP import failures seen with some PyTorch wheel combinations.

Main environment responsibilities:
- iterative denoising
- iterative registration
- input metrics and comparisons
- LLM advisor integration
- deterministic HTML report generation
- local browser UI launcher through `neuropilot_local_ui.py`

## Optional Suite2p Environment

```bash
conda env create -f environment-suite2p.yml
conda activate suite2p
```

Use this second environment for cell-data downstream segmentation / ROI selection.

On Windows, this environment template sets `KMP_DUPLICATE_LIB_OK=TRUE` because `suite2p` plus its pip-installed dependencies can otherwise raise the `libomp.dll` / `libiomp5md.dll` duplicate-runtime error during import.

The published examples use the environment name `suite2p`, but the name is not fixed. If you create the downstream environment under another name, pass it to the main entry with:

```bash
python neuropilot_pipeline.py --downstream-env your_env_name ...
```

Even for the full two-environment workflow, the user-facing main command is still launched from `neuropilot`. The main entry switches to the downstream interpreter only for the cell-data downstream stage.

If automatic environment switching is not enough on your machine, point the pipeline to an explicit interpreter:

```bash
set NEUROPILOT_DOWNSTREAM_PYTHON=/full/path/to/python
```

Linux shell form:

```bash
export NEUROPILOT_DOWNSTREAM_PYTHON=/full/path/to/python
```

## Local Configuration

For local machine-specific overrides, you may keep settings in `.env` next to `neuropilot_pipeline.py`. The runtime loads `.env` first and also accepts `.env.example` as a local fallback convenience source.

Typical pattern:

```text
NEUROPILOT_GPU=0
OPENAI_API_KEY=...
```

Linux users normally store the same values in `.env`, `.env.example`, or export them from the shell before running the pipeline.

When `--llm-mode shadow` or `--llm-mode apply` is passed on the command line, a real `OPENAI_API_KEY` in local `.env`, local `.env.example`, or the shell is enough to let the advisor use the live backend automatically.

The command-line flag `--GPU` overrides the default GPU selection from `.env`.

If you run with `--non-cell-data`, the pipeline stays inside `neuropilot` and does not require the second downstream environment to be installed.

## Report Output

The public snapshot defaults to HTML report generation.

Expected report artifacts:
- `report/report.html`
- `report/report_print.html`
- `report/report_data.json`
- `report/report_manifest.json`

PDF export is disabled by default and is not required for the published workflow.

## Local Browser UI

The repository also includes a local-only browser UI:

```bash
python neuropilot_local_ui.py --open-browser
```

Default address:

```text
http://127.0.0.1:8008
```

The UI:
- validates that `input-dir` uses the expected dataset-subfolder layout
- lets the user set the main environment name, downstream environment name, GPU, subfolders, and LLM mode
- launches the pipeline through the selected main conda environment
- scans `output-dir` and embeds generated `report.html` files in the browser

See [LOCAL_UI.md](LOCAL_UI.md) for the full local-UI workflow.

## Bundled And Excluded Weights

Bundled:
- the default PyLoReg checkpoint used by the registration stage

Not bundled:
- DeepCAD denoising pretrained weights, because the denoising stage is retrained for each input dataset
