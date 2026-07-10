# NeuroPilot Published Layout

This published tree keeps user-facing entry points in the repository root and
moves implementation helpers into focused subdirectories.

## Root Entry Points

- `neuropilot_pipeline.py`: main end-to-end processing pipeline.
- `neuropilot_local_ui.py`: local browser UI launcher.
- `report_builder.py`: rebuild a report from an existing run root.
- `prepare_input_tiffs.py`: helper for converting a flat TIFF folder into
  dataset subfolders.

## Internal Modules

- `core/`: shared implementation modules used by the pipeline, report builder,
  downstream segmentation, RAG, and LLM advisor.
- `deepcad/`: DeepCAD denoising implementation.
- `demotion/`: optical-flow motion correction helpers.
- `PyLoReg/`: PyLoReg inference module, model definition, and model artifacts.
- `local_ui/static/`: browser assets for the local UI.
- `literature/index/` and `literature/manifest/`: local literature knowledge
  base metadata used by report interpretation.

Generated outputs such as `runs/`, caches, temporary figures, raw literature
PDFs, and ad hoc experimental scripts are intentionally excluded from this
published layout.
