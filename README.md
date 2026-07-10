# NeuroPilot

NeuroPilot is a local calcium-imaging pipeline for iterative denoising, motion correction, optional downstream segmentation, and deterministic HTML reporting.

![NeuroPilot scope overview](docs/assets/readme/neuroPilot_scope_overview.png)

## What It Does

- runs iterative denoising and registration from a single main entry
- optionally runs cell-data downstream analysis in a separate `suite2p` environment
- writes a per-stack `report.html` for browser-based review
- supports a local browser UI for parameter entry, job launch, monitoring, and report viewing

![NeuroPilot restoration and analysis workflow](docs/assets/readme/neuroPilot_workflow.png)

## Prerequisites

- Run all commands from the repository root unless a command explicitly says otherwise.
- Use Conda or Mamba to create the environments.
- A CUDA-capable NVIDIA GPU is recommended for full denoising and registration runs.
- On Windows, use Anaconda Prompt or PowerShell after `conda init`.
- Keep output paths reasonably short on Windows, for example `runs/demo`, to avoid long-path issues in generated artifacts.

## Environments

Create and activate the main environment:

```bash
conda env create -f environment-neuropilot.yml
conda activate neuropilot
```

If you want `cell-data` downstream analysis, also create the downstream environment:

```bash
conda env create -f environment-suite2p.yml
```

Default environment names:

- main pipeline: `neuropilot`
- downstream segmentation: `suite2p`

Verify that the entry points are visible after activating the main environment:

```bash
python neuropilot_local_ui.py --help
python neuropilot_pipeline.py --help
python report_builder.py --help
```

For the downstream helper, use this command in the source repository layout:

```bash
python downstream_pipeline.py --help
```

In the published layout, the same helper is stored under `core/`:

```bash
python core/downstream_pipeline.py --help
```

## Input Layout

`--input-dir` must contain first-level dataset subfolders. Each subfolder may contain one or more `.tif` or `.tiff` files.

```text
/path/to/datasets/
  sample_a/
    sample_a_01.tif
    sample_a_02.tif
  sample_b/
    sample_b_01.tif
```

If your TIFF files are flat at the input root, prepare them first:

```bash
python prepare_input_tiffs.py --input-dir /path/to/flat_tif_folder --output-dir /path/to/prepared_datasets
```

The helper copies or organizes TIFF files into first-level subfolders that the pipeline can scan. Use a separate `--output-dir` when you want to keep the original input folder unchanged.

## Local UI

The local UI is the recommended starting point for most runs because it validates paths, scans subfolders, launches the same pipeline entry point, streams logs, and opens reports from the output directory.

Launch it from the main environment:

```bash
conda activate neuropilot
python neuropilot_local_ui.py --open-browser
```

By default, the UI binds to `127.0.0.1:8008`. If that port is already in use, choose another port:

```bash
python neuropilot_local_ui.py --port 8010 --open-browser
```

![NeuroPilot local UI](docs/assets/readme/local_ui_screenshot.png)

The UI provides:

- input and output directory selection
- main environment and downstream environment selection
- GPU, `um-per-pixel`, and `frame-rate` entry
- subfolder scanning and validation
- in-place preparation of flat TIFF inputs when needed
- live job logs
- direct embedded viewing of generated `report.html`

Recommended UI workflow:

1. Enter `input-dir` and `output-dir`.
2. Set the main environment name, GPU, `um-per-pixel`, and `frame-rate`.
3. Choose `cell-data` or `non-cell-data`.
4. If using `cell-data`, confirm the downstream environment name.
5. Click `Scan input directory`.
6. If flat TIFF files are detected, use the prepare button or run `prepare_input_tiffs.py` manually.
7. Select the valid subfolders to process.
8. Launch the pipeline and review `report.html` in the built-in viewer.

## Main Entry

Use the command-line entry point when you want a reproducible scripted run.

Minimal run:

```bash
conda activate neuropilot
python neuropilot_pipeline.py --input-dir /path/to/datasets --output-dir /path/to/run_output --non-cell-data
```

Common options:

- `--input-dir`: dataset root containing first-level subfolders
- `--output-dir`: run output root
- `--subfolders`: optional subset such as `sample_a,sample_b`
- `--GPU`: GPU index or comma-separated indices, default `0`
- `--um-per-pixel`: microns per pixel used by reports and pre-segmentation profiling, default `0.645`
- `--frame-rate`: frame rate in Hz used by reports and downstream settings, default `10`
- `--llm-mode`: `off`, `shadow`, or `apply`
- `--cell-data`: enable downstream segmentation, trace extraction, and Downstream Improvement metrics
- `--non-cell-data`: skip cell-data downstream analysis and generate the restoration-focused report
- `--downstream-env`: downstream conda environment name for `cell-data`, default `suite2p`

Use `--cell-data` only for cell-type calcium-imaging inputs where downstream ROI and trace metrics are expected. Use `--non-cell-data` for restoration-only runs or non-cellular imaging data.

Cell-data example:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --cell-data \
  --downstream-env suite2p \
  --llm-mode off \
  --GPU 0 \
  --um-per-pixel 0.645 \
  --frame-rate 10
```

Non-cell-data example:

```bash
python neuropilot_pipeline.py \
  --input-dir /path/to/datasets \
  --output-dir /path/to/run_output \
  --non-cell-data \
  --llm-mode off \
  --GPU 0 \
  --um-per-pixel 0.645 \
  --frame-rate 10
```

If `--subfolders` is omitted, all valid first-level subfolders are processed.

## LLM Apply Mode

`--llm-mode apply` uses previous experiment records to guide concrete pipeline parameters. Literature chunks are reserved for report interpretation and text summarization, not direct parameter advice.

For live model calls, set an OpenAI-compatible API key in the process that starts NeuroPilot. The default environment variable name is exactly `OPENAI_API_KEY`.

PowerShell, current terminal only:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

PowerShell, persistent user environment:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")
```

After setting a persistent user environment variable, restart the terminal or Codex app so the new process can read it.

Windows cmd:

```bat
setx OPENAI_API_KEY "sk-..."
```

macOS/Linux shell:

```bash
export OPENAI_API_KEY="sk-..."
```

Optional overrides are `LLM_ADVISOR_MODEL`, `LLM_ADVISOR_BASE_URL`, and `LLM_ADVISOR_TIMEOUT_S`. If they are not set, NeuroPilot uses `gpt-4.1-mini`, `https://api.openai.com/v1`, and the built-in timeout. You can also create a local `.env` file in the repository root with the same key names. Do not commit real API keys.

Mode summary:

- `off`: run without LLM parameter advice
- `shadow`: request advice and record it without applying suggested parameters
- `apply`: request advice and apply accepted suggestions during the iterative pipeline

## Outputs

NeuroPilot writes outputs under:

```text
<output-dir>/<dataset-subfolder>/<stack-tag>/
```

Typical per-stack artifacts include:

- `final/final_stack.tif`
- `manifests/pipeline_manifest.json`
- `report/report.html`
- `report/report_data.json`
- `report/report_manifest.json`
- `segmentation/` for cell-data downstream outputs

When `--cell-data` completes, downstream artifacts such as `segmentation/summary.json`, `downstream_comparison.json`, and paired-trace summaries populate the Downstream Improvement section in the report.

The UI discovers `report.html` recursively under the chosen output directory.

To rebuild only the report from an existing run root:

```bash
python report_builder.py --run-root /path/to/run_output/sample_a/sample_a_tag --no-pdf
```

## Demo

The current demo data layout uses two first-level dataset folders:

```text
demo_data/
  24h/
    CellVideo 01-1.tif
  spine/
    ju2df_5day_freemoving-male1-5day-image-pain 0.tif
```

The TIFF payloads are large and are distributed outside normal git history; place them at the paths above before running the demo.

Run it from the repository root:

```bash
python neuropilot_pipeline.py \
  --input-dir demo_data \
  --subfolders 24h,spine \
  --output-dir runs/demo_data \
  --non-cell-data \
  --llm-mode off \
  --GPU 0 \
  --um-per-pixel 0.645 \
  --frame-rate 10
```

Or start the UI and use:

- input directory: `demo_data`
- output directory: `runs/local_ui_demo`
- subfolders: `24h`, `spine`

## Runtime Notes

- `cell-data` uses the downstream environment only for the downstream stage and can take longer than restoration-only runs.
- `NEUROPILOT_PYLOREG_BACKEND` defaults to `torch`; set it to `tensorrt` only when TensorRT and the matching engine files are available.
- The final report is generated after the final stack and downstream artifacts are materialized.
- The report summary uses local literature chunks for text interpretation and appends the applied chunk list at the end of the report.
