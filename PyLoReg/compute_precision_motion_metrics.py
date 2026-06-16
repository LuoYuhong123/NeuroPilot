#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute pipeline motion metrics for two precision experiment TIFFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _extract_motion(metrics: dict) -> dict:
    rigid = metrics.get("rigid_motion_metric", {})
    summary = rigid.get("rigid_motion_summary", {})
    jitter = rigid.get("frame_to_frame_jitter", {})
    conf = rigid.get("registration_confidence", {})
    return {
        "motion_mean_px": summary.get("motion_mean_px"),
        "motion_median_px": summary.get("motion_median_px"),
        "motion_p95_px": summary.get("motion_p95_px"),
        "motion_max_px": summary.get("motion_max_px"),
        "jitter_mean_px": jitter.get("jitter_mean_px"),
        "jitter_p95_px": jitter.get("jitter_p95_px"),
        "jitter_max_px": jitter.get("jitter_max_px"),
        "corr_mean": conf.get("corr_mean"),
        "corr_median": conf.get("corr_median"),
        "corr_min": conf.get("corr_min"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    sys.path.insert(0, str(_repo_root()))
    from pipeline_metrics import compute_metrics_for_tif

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = compute_metrics_for_tif(args.baseline, args.out_dir / "baseline_metrics")
    candidate_metrics = compute_metrics_for_tif(args.candidate, args.out_dir / "candidate_metrics")
    summary = {
        "baseline": {
            "path": str(args.baseline.resolve()),
            "motion": _extract_motion(baseline_metrics),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "motion": _extract_motion(candidate_metrics),
        },
    }
    out_path = args.out_dir / "motion_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
