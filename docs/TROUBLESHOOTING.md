# Troubleshooting

## No Input Subfolders Were Selected

If the pipeline reports that no input subfolders were selected:
- check `--input-dir`
- ensure each dataset sits in its own first-level child folder
- use `--subfolders sample_a,sample_b` to force a subset
- or set `NEUROPILOT_SUBFOLDERS`

If your TIFF files are flat in one folder instead of already separated into child folders, run:

```bash
python prepare_input_tiffs.py --input-dir /path/to/flat_tif_folder
```

## Live LLM Still Looks Like Mock

Check all of the following:
- `--llm-mode` is not `off`
- `OPENAI_API_KEY` is present in your shell or local `.env`
- if `--llm-mode shadow` or `--llm-mode apply` is used and `OPENAI_API_KEY` is present, the public snapshot should switch to live advisor mode automatically
- you are reading the latest run output, not an older run directory

## Downstream Environment Not Found

For cell-data runs:
- confirm the second downstream environment is installed
- if it is not named `suite2p`, pass `--downstream-env your_env_name`
- if automatic environment resolution still fails, set `NEUROPILOT_DOWNSTREAM_PYTHON` to an explicit interpreter path

For non-cell runs:
- `--non-cell-data` should complete without the downstream segmentation environment
- if a non-cell run still complains about the downstream interpreter, re-check that the command really used `--non-cell-data`

## Downstream Suite2p Stage Not Running

Check:
- you used `--cell-data`
- the `suite2p` environment exists
- `NEUROPILOT_DOWNSTREAM_ENV_NAME=suite2p` or `NEUROPILOT_DOWNSTREAM_PYTHON` points to the intended interpreter
- suite2p is importable inside that interpreter

## Suite2p Import Fails With libomp.dll / libiomp5md.dll

If `python -c "import suite2p"` fails on Windows with:
- `OMP: Error #15`
- `Initializing libomp.dll, but found libiomp5md.dll already initialized`

then your environment is hitting the common duplicate OpenMP runtime conflict from pip-installed dependencies.

Immediate workaround for an existing environment:

```bash
conda env config vars set -n suite2p KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate suite2p
```

The published `environment-suite2p.yml` now includes this variable by default for future installs.

## Report PDF Is Missing

This is expected in the public snapshot.

Default public settings are HTML-only:
- `NEUROPILOT_REPORT_GENERATE_PDF=false`
- `NEUROPILOT_REPORT_TRY_PDF=false`

Use `report/report.html` as the primary report artifact.

## Optional DeepInterpolation Backend Is Missing

This is also expected.

The public repository does not bundle `deepinterpolation/`. Any compatibility function that still points to that backend will raise a clear import error if invoked.

## Environment Creation Fails On GPU Packages

The main environment assumes an NVIDIA GPU-oriented setup. If `pytorch-cuda=12.1` does not match your machine:
- adjust the CUDA package selection for your driver stack
- or create a local derivative environment file for your hardware
- on Linux outside the default mirror region, replace the explicit mirror URLs in the environment files with your preferred conda channels before retrying

## Conda Solve Mentions torchaudio But The YAML Does Not

If `conda env create -f environment-neumar.yml` fails with a solver error that mentions `torchaudio` or another package not present in the environment file, the problem is usually not the repository YAML itself.

Check the following before editing the repository YAML:
- confirm the current local `environment-neumar.yml` really is the file you intended to use
- search the file itself for `torchaudio`, `torchvision`, `pytorch`, or other stale constraints
- inspect hidden conda pins and global config
- clear the solver cache and retry with the classic solver

Check with:

```bash
grep -nE 'torchaudio|torchvision|pytorch|cuda|python' environment-neumar.yml
cat "$CONDA_PREFIX/conda-meta/pinned" 2>/dev/null
conda config --show-sources
conda config --show create_default_packages
conda config --show pinned_packages
echo "$CONDA_PINNED_PACKAGES"
echo "$CONDA_CREATE_DEFAULT_PACKAGES"
```

Then retry with:

```bash
conda clean --index-cache --tarballs --packages -y
CONDA_SOLVER=classic \
conda env create -f environment-neumar.yml
```

This repository's `environment-neumar.yml` intentionally installs the main PyTorch wheel through pip and does not require conda-side `torchaudio`.

If Linux solving still looks fragile after the steps above, replace the explicit mirror URLs in the environment file with your preferred `conda-forge` / `pkgs/main` channels and retry.

## Linux cv2 Import Fails With libGL.so.1

If the main environment imports fail on Linux with messages mentioning `libGL.so.1` or `libglib-2.0.so.0`, install the missing system packages:

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

Then reactivate the environment and retry the import.

## Linux CUDA Is Not Visible To PyTorch

If `torch.cuda.is_available()` is `False` on Linux:
- confirm the NVIDIA driver is installed and working through `nvidia-smi`
- verify that the machine's driver is compatible with CUDA 12.x
- reactivate the `neumar` environment after driver changes
- if needed, replace the pip-installed torch wheel with one that matches your local CUDA stack

## Torch Import Fails With fbgemm.dll On Windows

If the main environment fails before the pipeline starts, with an error like:
- `Error loading ... torch\\lib\\fbgemm.dll`
- `WinError 127`

then the active conda environment is usually missing the `libomp140.x86_64.dll` filename that some Windows PyTorch wheels expect.

Immediate fix for an existing environment:

```bash
copy %CONDA_PREFIX%\\Library\\bin\\libomp.dll %CONDA_PREFIX%\\Library\\bin\\libomp140.x86_64.dll
```

The published main entry now tries to create this alias automatically before importing torch-backed modules. Recreating the `neumar` environment from the updated `environment-neumar.yml` is still recommended for a clean setup.

## Torch Or DeepCAD Import Fails With Duplicate OpenMP On Windows

If the main environment fails while importing `deepcad` or the pipeline entry, with a message like:
- `OMP: Error #15`
- `Initializing libomp.dll, but found libiomp5md.dll already initialized`

then the active process is seeing both LLVM OpenMP and Intel OpenMP runtimes.

Immediate fix for an existing `neumar` environment:

```bash
conda env config vars set -n neumar KMP_DUPLICATE_LIB_OK=TRUE
conda deactivate
conda activate neumar
```

The published `environment-neumar.yml` now includes this variable by default for future installs.

## Output Looks Incomplete

Check whether the run was:
- non-cell data, which skips segmentation outputs
- LLM `off`, which skips advisor artifacts
- HTML-only, which skips PDF output

## PyLoReg Checkpoint Not Found

The public snapshot is expected to bundle the default PyLoReg checkpoint used by the registration stage.

If you still see a missing-checkpoint warning:
- check that `PyLoReg/PyLoReg_model/GM3_fn5_202511301551/gmflow_latest.pt` exists
- check that you did not remove or ignore the bundled checkpoint during packaging
