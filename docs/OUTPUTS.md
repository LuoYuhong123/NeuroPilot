# Outputs

## Run Root Overview

When multiple datasets are processed from one `--input-dir`, the pipeline creates one result root per input child folder:

```text
run_output/
  sample_a/
    manifests/
    iterations/
    final/
    segmentation/
    report/
    results_deepcad/
    results_demotion/
    pth_deepcad/
    logs/
    final_used_params.json
  sample_b/
    ...
```

Each dataset keeps its own run root under `--output-dir/<input_child_folder_name>/`. Not every directory is populated in every run. For example, `segmentation/` is meaningful only for cell-data workflows.

## Key Files

### Manifests

- `manifests/pipeline_manifest.json`: top-level run metadata, iteration summaries, downstream summary, report summary for one dataset
- `final_used_params.json`: effective parameter record for each iteration for one dataset

### Iteration Artifacts

For each iteration:

```text
iterations/
  iter_0/
    metrics/
    llm/
  iter_1/
    metrics/
    llm/
```

Common contents:
- metric json files for raw / denoise / motion / final comparisons
- LLM request / response / validated suggestion artifacts when LLM modes are enabled
- apply records and warnings when apply mode is used

### Final Output

- `final/final_stack.tif`
- `final/final_stack_sidecar.json`

The public v1 semantic is frozen:
- `final_stack = last_iter_denoised_output`

### Downstream Output

When `--cell-data` is enabled, the downstream stage materializes artifacts under `segmentation/`, including:
- backend status
- run status
- summary files
- ROI overlays
- paired trace outputs

### Reports

The published workflow generates HTML report artifacts under `report/`:
- `report.html`
- `report_print.html`
- `report_data.json`
- `report_manifest.json`

Because public defaults disable PDF export, `report.pdf` is normally absent unless you explicitly re-enable PDF generation.

## Files Used By The Report Builder

The deterministic report builder reads from:
- `manifests/pipeline_manifest.json`
- `final_used_params.json`
- `final/final_stack_sidecar.json`
- iteration comparison metrics under `iterations/iter_*/metrics/`
- downstream summary files when the downstream section is available
