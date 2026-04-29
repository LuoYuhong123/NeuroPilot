# Local UI

## Purpose

The local UI is a browser-based control panel for running NeuroPilot on the same machine where the repository is deployed.

It is designed for the published pipeline layout:
- the UI binds to `127.0.0.1` by default
- the core pipeline is still executed through the existing CLI entry
- the UI does not import the pipeline as a Python module for execution
- the UI launches the main pipeline through `conda run -n <main_env_name> ...`
- the UI renders generated `report.html` files directly inside the browser

## Launch

From the repository root:

```bash
python neuropilot_local_ui.py --open-browser
```

Default local address:

```text
http://127.0.0.1:8008
```

Optional flags:
- `--host`: local bind host, default `127.0.0.1`
- `--port`: local port, default `8008`
- `--open-browser`: open the UI automatically after startup

Because the UI uses only Python standard-library modules, it does not add a separate web-framework dependency. The actual NeuroPilot run is still launched from the selected conda environment.

## Parameters Exposed In The UI

- `input-dir`
- `output-dir`
- main environment name
- downstream environment name
- GPU index
- dataset type: `cell-data` or `non-cell-data`
- LLM mode: `off` or `apply`
- `OPENAI_API_KEY` when `llm-mode=apply`
- selected dataset subfolders under `input-dir`

Important runtime behavior:
- the main environment name is consumed by the UI launcher, not passed as a NeuroPilot CLI flag
- the downstream environment name maps to `--downstream-env`
- the UI always passes an explicit `--subfolders` list based on the selected valid dataset folders
- only one job can run at a time through the UI

## Input Directory Validation

Before launch, the UI scans `input-dir` and checks:
- the directory exists
- first-level child folders are present
- selected child folders contain TIFF files
- no loose `.tif` or `.tiff` files are present directly under `input-dir`

If loose root-level TIFF files are found, reorganize them first with:

```bash
python prepare_input_tiffs.py --input-dir /path/to/flat_tif_folder
```

## Report Viewing

After the pipeline completes, the UI scans `output-dir` and looks for:

```text
<output-dir>/<dataset-folder>/<stack-name>/report/report.html
```

The report viewer is intended for the repository defaults where HTML assets are embedded and CSS is inlined.

The UI groups reports by dataset folder and stack folder, then opens the selected `report.html` inside an iframe on the same local server.

## Notes

- the UI is local-only and does not add authentication
- environment availability is checked with `conda run -n <env> python -c "import sys; print(sys.executable)"`
- if the downstream path is `non-cell-data`, the downstream environment field is hidden and not used
- if `llm-mode=off`, the UI removes `OPENAI_API_KEY` from the launched subprocess environment
