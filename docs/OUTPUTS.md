# Outputs

## Run Root Overview

When multiple datasets are processed from one `--input-dir`, the pipeline creates one result root per input child folder:

```text
run_output/
  sample_a/
    _shared/
      pth_deepcad/
      train_inputs/
      iterations/
    sample_a_fov01/
      manifests/
      iterations/
      final/
      segmentation/
      report/
      results_deepcad/
      results_demotion/
      metrics/
      final_used_params.json
    sample_a_fov02/
      ...
  sample_b/
    _shared/
      ...
    sample_b/
      ...
```

Each dataset keeps its own dataset root under `--output-dir/<input_child_folder_name>/`. The `_shared/` directory stores artifacts reused by all TIFF files in that dataset folder, including the shared denoise-training subset, checkpoints, and shared LLM artifacts. Each TIFF file then gets its own per-stack run root, typically named after the TIFF stem. Not every directory is populated in every run. For example, `segmentation/` is meaningful only for cell-data workflows.

## Key Files

### Shared Dataset-Level Artifacts

- `_shared/pth_deepcad/`: shared denoise model checkpoints trained for the input child folder
- `_shared/train_inputs/iter_*`: materialized TIFF subset used to train each iteration's shared denoise model
- `_shared/iterations/iter_*/llm/`: shared LLM request / response / validation artifacts for that dataset folder and iteration

### Per-Stack Manifests

- `<stack_tag>/manifests/pipeline_manifest.json`: top-level run metadata, iteration summaries, downstream summary, and report summary for one TIFF stack
- `<stack_tag>/final_used_params.json`: effective parameter record for each iteration for one TIFF stack, including the shared denoise-training policy

### Iteration Artifacts

For each iteration inside one per-stack run:

```text
<stack_tag>/
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

- `<stack_tag>/final/final_stack.tif`
- `<stack_tag>/final/final_stack_sidecar.json`

The public v1 semantic is frozen:
- `final_stack = last_iter_denoised_output`

### Downstream Output

When `--cell-data` is enabled, the downstream stage materializes artifacts under `<stack_tag>/segmentation/`, including:
- backend status
- run status
- summary files
- ROI overlays
- paired trace outputs

### Reports

The published workflow generates HTML report artifacts under `<stack_tag>/report/`:
- `report.html`
- `report_print.html`
- `report_data.json`
- `report_manifest.json`

Because public defaults disable PDF export, `report.pdf` is normally absent unless you explicitly re-enable PDF generation.

## Files Used By The Report Builder

The deterministic report builder reads from one per-stack run root:
- `manifests/pipeline_manifest.json`
- `final_used_params.json`
- `final/final_stack_sidecar.json`
- iteration comparison metrics under `iterations/iter_*/metrics/`
- downstream summary files when the downstream section is available
