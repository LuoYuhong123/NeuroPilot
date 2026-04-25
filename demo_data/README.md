# Demo Data Layout

This directory follows the same input layout expected by the main pipeline: one child folder per dataset, and each child folder may contain one or more related TIFF files.

```text
demo_data/
  quick_demo/
    quick_demo.tif
  raw_CA1/
    raw_CA1.tif
  raw_LEC/
    raw_LEC.tif
```

## Repository-Tracked Quick Demo

The repository keeps one small smoke-test movie in git:
- `quick_demo/quick_demo.tif`

This file is intentionally kept below the normal GitHub repository single-file limit so users can clone the repository and run a minimal example immediately.

Current selection details:
- source movie: `raw_LEC/raw_LEC.tif`
- extracted window: frames `440:520`, center crop `(x=32:480, y=32:480)`
- final quick-demo shape: `80 x 448 x 448`
- rationale: keep a larger field of view than the old 256x256 quick demo while reducing the risk of obvious blank registration borders in the processed outputs


## Release-Only Large Demo Movies

The larger example movies are intended to be distributed as GitHub Release assets, not committed into git history:
- `raw_CA1/raw_CA1.tif`
- `raw_LEC/raw_LEC.tif`

GitHub Releases page:
- `https://github.com/LuoYuhong123/NeuroPilot/releases`

The root `.gitignore` excludes those larger TIFFs so they stay local unless you upload them separately as release assets.

After downloading the release assets, place them back into the folder layout shown above. If you place multiple TIFF files into the same child folder, keep them as similar as possible in modality/type, acquisition settings, and noise profile so they can share denoise training sensibly.
