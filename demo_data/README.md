# Demo Data Layout

This directory follows the same input layout expected by the main pipeline:

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

## Release-Only Large Demo Movies

The larger example movies are intended to be distributed as GitHub Release assets, not committed into git history:
- `raw_CA1/raw_CA1.tif`
- `raw_LEC/raw_LEC.tif`

The root `.gitignore` excludes those larger TIFFs so they stay local unless you upload them separately as release assets.

After downloading the release assets, place them back into the folder layout shown above.
