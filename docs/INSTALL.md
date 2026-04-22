# Install

## Requirements

The public snapshot is designed around a conda-based workflow.

Recommended assumptions:
- Python 3.10
- an NVIDIA GPU for the main `neumar` environment
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
conda env create -f environment-neumar.yml
conda activate neumar
```

This environment file is tuned for mirrored conda channels plus a pip-installed CUDA 12.1 PyTorch wheel, which helps avoid the flaky `nvidia` conda channel on some Windows setups.

Linux notes:
- the same `environment-neumar.yml` file is intended to work on Linux
- if you are outside the default mirror region, replace the explicit mirror URLs with your preferred `conda-forge` / `pkgs/main` channels before creating the environment
- after activation, a quick sanity check is:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
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

Even for the full two-environment workflow, the user-facing main command is still launched from `neumar`. The main entry switches to the downstream interpreter only for the cell-data downstream stage.

If automatic environment switching is not enough on your machine, point the pipeline to an explicit interpreter:

```bash
set NEUROPILOT_DOWNSTREAM_PYTHON=/full/path/to/python
```

Linux shell form:

```bash
export NEUROPILOT_DOWNSTREAM_PYTHON=/full/path/to/python
```

## Local Configuration

Copy `.env.example` to a local `.env` next to `neuropilot_pipeline.py` if you want private machine-specific overrides. The published `.env.example` file is a template only and is not loaded at runtime.

Typical pattern:

```text
NEUROPILOT_GPU=0
OPENAI_API_KEY=...
```

Linux users normally store the same values in `.env` or export them from the shell before running the pipeline.

When `--llm-mode shadow` or `--llm-mode apply` is passed on the command line, a non-empty `OPENAI_API_KEY` in local `.env` or the shell is enough to let the advisor use the live backend automatically.

The command-line flag `--GPU` overrides the default GPU selection from `.env`.

If you run with `--non-cell-data`, the pipeline stays inside `neumar` and does not require the second downstream environment to be installed.

## Report Output

The public snapshot defaults to HTML report generation.

Expected report artifacts:
- `report/report.html`
- `report/report_print.html`
- `report/report_data.json`
- `report/report_manifest.json`

PDF export is disabled by default and is not required for the published workflow.

## Bundled And Excluded Weights

Bundled:
- the default PyLoReg checkpoint used by the registration stage

Not bundled:
- DeepCAD denoising pretrained weights, because the denoising stage is retrained for each input dataset
