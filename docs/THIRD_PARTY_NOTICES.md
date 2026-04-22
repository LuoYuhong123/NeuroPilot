# Third-Party Notices

This repository combines original NeuroPilot code with redistributed and modified research code from upstream projects. Unless a file carries a more specific notice, the repository as a whole is distributed under the GNU General Public License v3.0 (GPL-3.0). Embedded notices inside individual files remain in effect and must be preserved.

## Repository-Level Licensing Decision

The top-level repository license is GPL-3.0.

This is the safest repository-level choice for the current public snapshot because:
- the denoising stack redistributed in `deepcad/` is derived from and modified from DeepCAD-RT, which is published under GPL-3.0
- the optional downstream segmentation workflow is designed around suite2p, which is published under GPL-3.0
- NeuroPilot-original modules are distributed under the same repository license for compatibility with the redistributed GPL components

## NeuroPilot-Original Components

The pipeline orchestration, metrics aggregation, report generation, and most surrounding glue code in files such as `neuropilot_pipeline.py`, `NeuMar_function.py`, `pipeline_metrics.py`, `report_builder.py`, and `report_figures.py` are original NeuroPilot work released in this repository under GPL-3.0.

## DeepCAD-RT-Derived Denoising Components

- Upstream project: DeepCAD-RT
- Upstream repository: https://github.com/cabooster/DeepCAD-RT
- Upstream license: GPL-3.0
- Local use in this snapshot: `deepcad/` and NeuroPilot wrappers that configure or invoke the denoising stack

These components were modified for NeuroPilot. Because this repository redistributes modified DeepCAD-RT-derived code, redistribution of this snapshot follows GPL-3.0.

## suite2p Downstream Dependency

- Upstream project: suite2p
- Upstream repository: https://github.com/MouseLand/suite2p
- Upstream license: GPL-3.0
- Local use in this snapshot: optional downstream segmentation / trace extraction via `STEP0_seg.py`, `downstream_pipeline.py`, and `environment-suite2p.yml`

The suite2p source code is not redistributed in this repository. Users install suite2p separately and remain responsible for complying with the upstream suite2p license when installing, modifying, or redistributing it.

## Registration / Motion-Correction Stack (`demotion/`)

Based on the current repository history and local project notes, the registration / motion-correction stack redistributed under `demotion/` is treated as NeuroPilot / NeuMar-original code, except where a file carries a separate embedded notice.

Known bundled third-party notice inside this directory:
- `demotion/flow_viz.py` explicitly states that the flow-visualization code comes from Tom Runia's OpticalFlow_Visualization project and carries an MIT notice in the file itself

That file-level MIT notice must remain intact.

## PyLoReg Runtime Subset (`PyLoReg/`)

This public snapshot redistributes a reduced runtime-oriented subset of `PyLoReg/`, mainly `pylog_inference.py` and the `PyLoRegNet/` network files needed by the current pipeline.

The public snapshot also bundles the default runtime checkpoint currently referenced by the registration code:
- `PyLoReg/PyLoReg_model/GM3_fn5_202511301551/gmflow_latest.pt`

From the bundled files themselves, the following file-level provenance is explicit:
- `PyLoReg/PyLoRegNet/position.py` cites Facebook DETR: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
- `PyLoReg/PyLoRegNet/trident_conv.py` cites Detectron2 TridentNet: https://github.com/facebookresearch/detectron2/blob/main/projects/TridentNet/tridentnet/trident_conv.py

Additional local clues in the bundled code indicate a GMFlow-style architecture or checkpoint naming convention, but this is an inference from comments and filenames inside the snapshot rather than a separate upstream license statement.

Because the public release redistributes only a reduced PyLoReg runtime subset, future releases should keep reviewing and documenting the provenance of any newly added PyLoReg files before expanding this directory.

## Practical Guidance For Future Releases

Before adding new third-party code to the public snapshot, record:
- the exact upstream repository or publication source
- the upstream license identifier
- whether the local copy is redistributed verbatim or modified
- any file-level notices that must remain intact

If additional upstream components are redistributed later, update this file and the top-level `README.md` at the same time.
