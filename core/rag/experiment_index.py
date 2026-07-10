#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INDEX_SCHEMA_VERSION = "neuropilot_rag.experiment_index.v1"
BUNDLE_SCHEMA_VERSION = "neuropilot_rag.retrieval_bundle.v1"
DEFAULT_TOP_K = 5
DEFAULT_LITERATURE_TOP_K = 5
DEFAULT_LITERATURE_MAX_CHUNKS_PER_PAPER = 2
STACK_RUN_REQUIRED_DIRS = ("raw_input", "results_deepcad", "results_demotion", "metrics", "iterations")
QUALITY_COMPLETE_MANIFEST = "complete_manifest"
QUALITY_RECOVERED_REPORT = "recovered_report"
QUALITY_PARTIAL_ITERATION = "partial_iteration_only"


def _print_json(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


try:
    from rag.literature_index import retrieve_literature
except Exception:
    retrieve_literature = None  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_file():
            return None
        with open(path_obj, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path_obj)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _default_literature_chunks_path() -> Path:
    return Path(__file__).resolve().parents[1] / "literature" / "index" / "literature_chunks.jsonl"


def _resolve_literature_chunks_path(path: str | Path | None) -> Path:
    if path:
        return Path(path).expanduser()
    env_path = os.getenv("NEUROPILOT_RAG_LITERATURE_CHUNKS") or os.getenv("NEUROPILOT_LITERATURE_CHUNKS")
    if env_path:
        return Path(env_path).expanduser()
    return _default_literature_chunks_path()


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _safe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_shape_thw(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    vals = [_safe_int(item) for item in value[:3]]
    if any(item is None or item <= 0 for item in vals):
        return None
    return [int(vals[0]), int(vals[1]), int(vals[2])]


def _tokenize(*parts: Any) -> set[str]:
    text = " ".join(str(part or "") for part in parts)
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9_]+", text) if len(tok) >= 2}


def _first_json(paths: Iterable[str | Path | None]) -> dict[str, Any] | None:
    for path in paths:
        data = _read_json(path)
        if data is not None:
            return data
    return None


def _existing_jsons(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(path for path in root.glob(pattern) if path.is_file())
    except Exception:
        return []


def _rglob_jsons(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(path for path in root.rglob(pattern) if path.is_file())
    except Exception:
        return []


def _existing_tifs(root: Path) -> list[Path]:
    try:
        return sorted(
            [path for path in root.glob("*.tif") if path.is_file()]
            + [path for path in root.glob("*.tiff") if path.is_file()]
        )
    except Exception:
        return []


def _looks_like_stack_run(run_root: Path) -> bool:
    return all((run_root / name).is_dir() for name in STACK_RUN_REQUIRED_DIRS)


def _discover_stack_run_roots(root_path: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        raw_dirs = sorted(path for path in root_path.rglob("raw_input") if path.is_dir())
    except Exception:
        raw_dirs = []
    for raw_dir in raw_dirs:
        run_root = raw_dir.parent
        if _looks_like_stack_run(run_root):
            candidates.append(run_root)
    return candidates


def _read_report_data(run_root: Path) -> dict[str, Any] | None:
    return _read_json(run_root / "report" / "report_data.json")


def _report_identity(report_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report_data, dict):
        return {}
    identity = report_data.get("dataset_identity")
    return identity if isinstance(identity, dict) else {}


def _truthy_or_none(value: Any) -> Any:
    return value if value not in ("", [], {}, None) else None


def _infer_stack_name(run_root: Path, manifest: dict[str, Any], report_data: dict[str, Any] | None) -> str | None:
    identity = _report_identity(report_data)
    raw_stack_path = identity.get("raw_stack_path")
    if raw_stack_path:
        try:
            return Path(str(raw_stack_path)).name
        except Exception:
            pass
    raw_tifs = _existing_tifs(run_root / "raw_input")
    if raw_tifs:
        return raw_tifs[0].name
    stack_tag = str(_truthy_or_none(manifest.get("stack_tag")) or run_root.name)
    inferred = re.sub(r"_[0-9a-fA-F]{6,}$", "", stack_tag).strip("_")
    return f"{inferred}.tif" if inferred else None


def _infer_identity(run_root: Path, manifest: dict[str, Any], report_data: dict[str, Any] | None) -> dict[str, Any]:
    identity = _report_identity(report_data)
    downstream = _read_downstream_summary(run_root)
    downstream_summary = downstream.get("summary", {}) if isinstance(downstream.get("summary"), dict) else {}
    folder_name = (
        _truthy_or_none(manifest.get("folder_name"))
        or _truthy_or_none(identity.get("folder_name"))
        or run_root.parent.name
    )
    stack_name = _truthy_or_none(manifest.get("stack_name")) or _infer_stack_name(run_root, manifest, report_data)
    dataset_profile = (
        _truthy_or_none(manifest.get("dataset_profile"))
        or _truthy_or_none(identity.get("dataset_profile"))
        or _truthy_or_none(downstream_summary.get("dataset_profile"))
    )
    is_cell_data = manifest.get("is_cell_data")
    if is_cell_data is None and "is_cell_data" in identity:
        is_cell_data = identity.get("is_cell_data")
    return {
        "folder_name": folder_name,
        "folder_tag": _truthy_or_none(manifest.get("folder_tag")) or _truthy_or_none(identity.get("folder_tag")) or run_root.parent.name,
        "stack_name": stack_name,
        "stack_tag": _truthy_or_none(manifest.get("stack_tag")) or run_root.name,
        "dataset_profile": dataset_profile,
        "is_cell_data": is_cell_data,
    }


def _record_quality(run_root: Path, manifest: dict[str, Any], final_used_params: dict[str, Any] | None, report_data: dict[str, Any] | None) -> dict[str, Any]:
    expected = {
        "manifests/pipeline_manifest.json": bool(manifest),
        "report/report_data.json": isinstance(report_data, dict),
        "report/report.html": (run_root / "report" / "report.html").is_file(),
        "final_used_params.json": isinstance(final_used_params, dict),
        "final/final_stack_sidecar.json": (run_root / "final" / "final_stack_sidecar.json").is_file(),
        "metrics/input/*_metrics.json": bool(_existing_jsons(run_root / "metrics" / "input", "*_metrics.json")),
        "iterations/iter_*/metrics": (run_root / "iterations").is_dir() and bool(list((run_root / "iterations").glob("iter_*/metrics"))),
    }
    missing = [name for name, present in expected.items() if not present]
    if manifest:
        level = QUALITY_COMPLETE_MANIFEST
        confidence = 1.0
        method = "manifest"
    elif expected["report/report_data.json"] or expected["final/final_stack_sidecar.json"]:
        level = QUALITY_RECOVERED_REPORT
        confidence = 0.85
        method = "structure_recovered_report"
    else:
        level = QUALITY_PARTIAL_ITERATION
        confidence = 0.55
        method = "structure_partial_iteration"
    return {
        "level": level,
        "confidence": confidence,
        "discovery_method": method,
        "has_manifest": bool(manifest),
        "has_report_data": expected["report/report_data.json"],
        "has_report_html": expected["report/report.html"],
        "has_final_used_params": expected["final_used_params.json"],
        "has_final_stack_sidecar": expected["final/final_stack_sidecar.json"],
        "missing_expected_files": missing,
    }


def _metric_summary(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics if isinstance(metrics, dict) else {}
    rigid = metrics.get("rigid_motion_metric", {}) if isinstance(metrics.get("rigid_motion_metric"), dict) else {}
    rigid_summary = rigid.get("rigid_motion_summary", {}) if isinstance(rigid.get("rigid_motion_summary"), dict) else {}
    jitter = rigid.get("frame_to_frame_jitter", {}) if isinstance(rigid.get("frame_to_frame_jitter"), dict) else {}
    return {
        "shape_thw": _as_shape_thw(metrics.get("data_summary", {}).get("shape_thw") if isinstance(metrics.get("data_summary"), dict) else None),
        "dtype": metrics.get("data_summary", {}).get("dtype") if isinstance(metrics.get("data_summary"), dict) else None,
        "snr": _safe_float(metrics.get("snr_metric", {}).get("snr") if isinstance(metrics.get("snr_metric"), dict) else None),
        "motion_mean_px": _safe_float(rigid_summary.get("motion_mean_px")),
        "motion_p95_px": _safe_float(rigid_summary.get("motion_p95_px")),
        "motion_max_px": _safe_float(rigid_summary.get("motion_max_px")),
        "jitter_mean_px": _safe_float(jitter.get("jitter_mean_px")),
        "jitter_p95_px": _safe_float(jitter.get("jitter_p95_px")),
        "bleaching_drop_percent": _safe_float(
            metrics.get("bleaching_trend", {}).get("relative_drop_percent")
            if isinstance(metrics.get("bleaching_trend"), dict)
            else None
        ),
    }


def _comparison_summary(comparison: dict[str, Any] | None) -> dict[str, Any]:
    comparison = comparison if isinstance(comparison, dict) else {}
    snr = comparison.get("snr_before_after", {}) if isinstance(comparison.get("snr_before_after"), dict) else {}
    bleaching = (
        comparison.get("bleaching_before_after", {})
        if isinstance(comparison.get("bleaching_before_after"), dict)
        else {}
    )
    motion = comparison.get("motion_before_after", {}) if isinstance(comparison.get("motion_before_after"), dict) else {}
    return {
        "raw_snr": _safe_float(snr.get("raw_snr")),
        "final_snr": _safe_float(snr.get("final_snr")),
        "delta_snr": _safe_float(snr.get("delta_snr")),
        "snr_ratio_final_over_raw": _safe_float(snr.get("ratio_final_over_raw")),
        "raw_bleaching_drop_percent": _safe_float(bleaching.get("raw_relative_drop_percent")),
        "final_bleaching_drop_percent": _safe_float(bleaching.get("final_relative_drop_percent")),
        "delta_motion_mean_px": _safe_float(motion.get("delta_motion_mean_px")),
        "delta_motion_p95_px": _safe_float(motion.get("delta_motion_p95_px")),
    }


def _read_raw_metrics(run_root: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[str | Path | None] = []
    candidates.append(manifest.get("raw_metrics_json"))
    candidates.extend(_existing_jsons(run_root / "metrics" / "input", "*_metrics.json"))
    report_data = _read_json(run_root / "report" / "report_data.json")
    if isinstance(report_data, dict):
        input_quality = report_data.get("metrics", {}).get("input_quality")
        if isinstance(input_quality, dict):
            return {
                "data_summary": {
                    "shape_thw": input_quality.get("shape_thw"),
                    "dtype": input_quality.get("dtype"),
                    "dynamic_range": input_quality.get("dynamic_range"),
                },
                "bleaching_trend": input_quality.get("bleaching"),
                "snr_metric": input_quality.get("snr"),
                "rigid_motion_metric": input_quality.get("rigid_motion"),
            }
    return _first_json(candidates)


def _read_final_comparison(run_root: Path) -> dict[str, Any] | None:
    report_data = _read_json(run_root / "report" / "report_data.json")
    if isinstance(report_data, dict):
        integrated = report_data.get("metrics", {}).get("integrated_final_improvement")
        if isinstance(integrated, dict):
            return integrated
    candidates = _rglob_jsons(run_root / "iterations", "comparison_raw_vs_final.json")
    if candidates:
        return _read_json(candidates[-1])
    return None


def _read_downstream_summary(run_root: Path) -> dict[str, Any]:
    report_data = _read_json(run_root / "report" / "report_data.json")
    if isinstance(report_data, dict):
        downstream = report_data.get("metrics", {}).get("downstream_improvement", {})
        if isinstance(downstream, dict):
            summary = downstream.get("summary")
            comparison = downstream.get("comparison")
            if isinstance(summary, dict) or isinstance(comparison, dict):
                return {"summary": summary if isinstance(summary, dict) else {}, "comparison": comparison if isinstance(comparison, dict) else {}}
    candidates = [
        run_root / "segmentation" / "segmentation_summary.json",
        run_root / "segmentation" / "downstream_summary.json",
        run_root / "segmentation" / "paired_trace" / "paired_trace_summary.json",
    ]
    out: dict[str, Any] = {}
    for path in candidates:
        data = _read_json(path)
        if data:
            out[path.stem] = data
    return out


def _read_advisor_summaries(run_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for llm_dir in sorted((run_root / "iterations").glob("iter_*/llm")):
        match = re.search(r"iter_(\d+)", llm_dir.as_posix())
        iter_index = int(match.group(1)) if match else None
        suggestion = _read_json(llm_dir / "suggestion_validated.json") or {}
        applied = _read_json(llm_dir / "applied_diff.json") or {}
        retrieval = _read_json(llm_dir / "retrieval_bundle.json") or {}
        if suggestion or applied or retrieval:
            out.append(
                {
                    "iter_index": iter_index,
                    "suggested_params": {
                        key: suggestion.get(key)
                        for key in ("n_epochs", "patch_x", "patch_y", "patch_t", "batch_size", "continue_iteration")
                        if key in suggestion
                    },
                    "reason": suggestion.get("reason"),
                    "advisor_mode": suggestion.get("meta", {}).get("advisor_mode") if isinstance(suggestion.get("meta"), dict) else None,
                    "used_fallback": suggestion.get("meta", {}).get("used_fallback") if isinstance(suggestion.get("meta"), dict) else None,
                    "applied_diff": applied,
                    "retrieval_match_count": len(retrieval.get("matched_experiments", [])) if isinstance(retrieval, dict) else 0,
                }
            )
    return out


def _params_from_item(item: dict[str, Any]) -> dict[str, Any]:
    params = item.get("effective_params") or item.get("params") or item.get("current_params") or item.get("train_params") or {}
    return params if isinstance(params, dict) else {}


def _summarize_param_item(item: dict[str, Any], *, param_source: str | None = None) -> dict[str, Any]:
    params = _params_from_item(item)
    return {
        "iter_index": item.get("iter_index"),
        "param_source": item.get("param_source") or param_source,
        "sample_mode": params.get("sample_mode") or item.get("sample_mode"),
        "n_epochs": params.get("n_epochs") or item.get("n_epochs"),
        "patch_x": params.get("patch_x") or item.get("patch_x"),
        "patch_y": params.get("patch_y") or item.get("patch_y"),
        "patch_t": params.get("patch_t") or item.get("patch_t"),
        "batch_size": params.get("batch_size") or item.get("batch_size"),
    }


def _iteration_params_from_llm_requests(run_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for llm_dir in sorted((run_root / "iterations").glob("iter_*/llm")):
        match = re.search(r"iter_(\d+)", llm_dir.as_posix())
        iter_index = int(match.group(1)) if match else None
        request = _read_json(llm_dir / "request.json") or {}
        context = request.get("context") if isinstance(request.get("context"), dict) else {}
        current_params = context.get("current_params") if isinstance(context.get("current_params"), dict) else {}
        runtime_train = context.get("current_runtime", {}).get("train", {}) if isinstance(context.get("current_runtime"), dict) else {}
        runtime_params = runtime_train.get("params") if isinstance(runtime_train.get("params"), dict) else {}
        params = current_params or runtime_params
        if params:
            out.append(_summarize_param_item({"iter_index": iter_index, "params": params}, param_source="llm_request_context"))
    return out


def _iteration_params_from_train_runtime(run_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for runtime_path in sorted((run_root / "iterations").glob("iter_*/metrics/train_runtime.json")):
        match = re.search(r"iter_(\d+)", runtime_path.as_posix())
        iter_index = int(match.group(1)) if match else None
        runtime = _read_json(runtime_path) or {}
        params = runtime.get("params") if isinstance(runtime.get("params"), dict) else {}
        if params:
            out.append(_summarize_param_item({"iter_index": iter_index, "params": params}, param_source="train_runtime"))
    return out


def _iteration_param_summary(
    manifest: dict[str, Any],
    final_used_params: dict[str, Any] | None,
    *,
    run_root: Path | None = None,
    report_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    source = final_used_params if isinstance(final_used_params, dict) else manifest
    iterations = source.get("iterations", []) if isinstance(source.get("iterations"), list) else []
    if not iterations and isinstance(report_data, dict):
        provenance = report_data.get("provenance")
        if isinstance(provenance, dict) and isinstance(provenance.get("effective_params_per_iteration"), list):
            iterations = provenance.get("effective_params_per_iteration") or []
    out: list[dict[str, Any]] = []
    for item in iterations:
        if not isinstance(item, dict):
            continue
        out.append(_summarize_param_item(item))
    if not out and run_root is not None:
        out = _iteration_params_from_llm_requests(run_root)
    if not out and run_root is not None:
        out = _iteration_params_from_train_runtime(run_root)
    return out


def summarize_run(run_root: str | Path) -> dict[str, Any] | None:
    run_root = Path(run_root).expanduser()
    manifest = _read_json(run_root / "manifests" / "pipeline_manifest.json") or {}
    if not manifest and not _looks_like_stack_run(run_root):
        return None
    final_used_params = _read_json(run_root / "final_used_params.json")
    report_data = _read_report_data(run_root)
    raw_metrics = _read_raw_metrics(run_root, manifest)
    final_comparison = _read_final_comparison(run_root)
    raw_summary = _metric_summary(raw_metrics)
    comparison_summary = _comparison_summary(final_comparison)
    downstream = _read_downstream_summary(run_root)
    rel_report = run_root / "report" / "report.html"
    identity = _infer_identity(run_root, manifest, report_data)
    dataset_profile = identity.get("dataset_profile")
    folder_name = identity.get("folder_name")
    stack_name = identity.get("stack_name")
    quality = _record_quality(run_root, manifest, final_used_params, report_data)
    return {
        "schema_version": "neuropilot_rag.experiment_summary.v1",
        "run_root": str(run_root.resolve()),
        "folder_name": folder_name,
        "folder_tag": identity.get("folder_tag"),
        "stack_name": stack_name,
        "stack_tag": identity.get("stack_tag"),
        "is_cell_data": identity.get("is_cell_data"),
        "dataset_profile": dataset_profile,
        "pipeline_llm_mode": manifest.get("pipeline_llm_mode"),
        "advisor_backend": manifest.get("advisor_backend"),
        "record_quality": quality,
        "raw_metrics": raw_summary,
        "final_outcome": comparison_summary,
        "downstream": downstream,
        "iteration_params": _iteration_param_summary(
            manifest,
            final_used_params,
            run_root=run_root,
            report_data=report_data,
        ),
        "advisor_summaries": _read_advisor_summaries(run_root),
        "report_html": str(rel_report.resolve()) if rel_report.exists() else None,
        "tokens": sorted(_tokenize(folder_name, stack_name, dataset_profile, identity.get("folder_tag"), identity.get("stack_tag"))),
    }


def discover_experiment_summaries(runs_roots: Iterable[str | Path]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for root in runs_roots:
        if not root:
            continue
        root_path = Path(root).expanduser()
        if not root_path.exists():
            continue
        candidate_roots: list[Path] = []
        try:
            candidate_roots.extend(manifest_path.parent.parent for manifest_path in sorted(root_path.rglob("manifests/pipeline_manifest.json")))
        except Exception:
            pass
        candidate_roots.extend(_discover_stack_run_roots(root_path))
        for run_root in candidate_roots:
            key = str(run_root.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            summary = summarize_run(run_root)
            if summary:
                summaries.append(summary)
    return summaries


def build_experiment_index(runs_roots: Iterable[str | Path]) -> dict[str, Any]:
    roots = [str(Path(root).expanduser().resolve()) for root in runs_roots if str(root or "").strip()]
    records = discover_experiment_summaries(roots)
    quality_counts: dict[str, int] = {}
    for record in records:
        quality = record.get("record_quality", {}) if isinstance(record.get("record_quality"), dict) else {}
        level = str(quality.get("level") or "unknown")
        quality_counts[level] = quality_counts.get(level, 0) + 1
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "runs_roots": roots,
        "record_count": len(records),
        "record_quality_counts": quality_counts,
        "records": records,
    }


def _infer_runs_root_from_run(run_root: Path | None) -> Path | None:
    if run_root is None:
        return None
    for parent in [run_root, *run_root.parents]:
        if parent.name.lower() == "runs":
            return parent
    return None


def parse_runs_roots(raw: str | None, *, current_run_root: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if raw:
        parts = [part.strip() for part in re.split(r"[;|]", raw) if part.strip()]
        roots.extend(Path(part).expanduser() for part in parts)
    env_roots = os.getenv("NEUROPILOT_RAG_RUNS_ROOTS")
    if env_roots and not raw:
        roots.extend(Path(part.strip()).expanduser() for part in re.split(r"[;|]", env_roots) if part.strip())
    if not roots:
        inferred = _infer_runs_root_from_run(Path(current_run_root).expanduser() if current_run_root else None)
        if inferred is not None:
            roots.append(inferred)
        else:
            roots.append(Path.cwd() / "runs")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except Exception:
            key = str(root).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def build_context_from_run_root(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).expanduser()
    manifest = _read_json(run_root / "manifests" / "pipeline_manifest.json") or {}
    report_data = _read_report_data(run_root)
    identity = _infer_identity(run_root, manifest, report_data)
    raw_metrics = _read_raw_metrics(run_root, manifest) or {}
    final_comparison = _read_final_comparison(run_root) or {}
    final_used_params = _read_json(run_root / "final_used_params.json") or {}
    iterations = _iteration_param_summary(
        manifest,
        final_used_params,
        run_root=run_root,
        report_data=report_data,
    )
    current_params = {}
    if iterations:
        last = iterations[-1] if isinstance(iterations[-1], dict) else {}
        current_params = _params_from_item(last) or last
        current_params = current_params if isinstance(current_params, dict) else {}
    return {
        "folder_name": identity.get("folder_name"),
        "stack_name": identity.get("stack_name"),
        "run_root": str(run_root.resolve()),
        "dataset_profile": identity.get("dataset_profile"),
        "is_cell_data": identity.get("is_cell_data"),
        "current_params": current_params,
        "metrics": raw_metrics,
        "comparisons": {"raw_vs_final": final_comparison},
    }


def _query_summary(context: dict[str, Any]) -> dict[str, Any]:
    metrics = context.get("metrics", {}) if isinstance(context.get("metrics"), dict) else {}
    comparisons = context.get("comparisons", {}) if isinstance(context.get("comparisons"), dict) else {}
    final_comparison = None
    for key, value in comparisons.items():
        if "raw_vs_final" in str(key) and isinstance(value, dict):
            final_comparison = value
            break
    if final_comparison is None:
        for value in comparisons.values():
            if isinstance(value, dict):
                final_comparison = value
                break
    current_params = context.get("current_params", {}) if isinstance(context.get("current_params"), dict) else {}
    return {
        "folder_name": context.get("folder_name"),
        "stack_name": context.get("stack_name"),
        "run_root": context.get("run_root"),
        "dataset_profile": context.get("dataset_profile"),
        "is_cell_data": context.get("is_cell_data"),
        "current_params": {
            key: current_params.get(key)
            for key in ("sample_mode", "n_epochs", "patch_x", "patch_y", "patch_t", "batch_size")
            if key in current_params
        },
        "raw_metrics": _metric_summary(metrics),
        "current_outcome": _comparison_summary(final_comparison),
        "tokens": sorted(_tokenize(context.get("folder_name"), context.get("stack_name"), context.get("dataset_profile"))),
    }


def _distance_score(a: float | None, b: float | None, scale: float) -> float:
    if a is None or b is None:
        return 0.0
    return max(0.0, 1.0 - min(abs(a - b) / max(scale, 1e-9), 1.0))


def _shape_score(a: list[int] | None, b: list[int] | None) -> float:
    if not a or not b:
        return 0.0
    scores = []
    for av, bv in zip(a, b):
        if av <= 0 or bv <= 0:
            continue
        scores.append(1.0 - min(abs(math.log(av / bv)), 1.0))
    return sum(scores) / len(scores) if scores else 0.0


def score_experiment(query: dict[str, Any], record: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    query_profile = query.get("dataset_profile")
    record_profile = record.get("dataset_profile")
    if query_profile and record_profile and str(query_profile).lower() == str(record_profile).lower():
        score += 1.2
        reasons.append(f"same dataset_profile={record_profile}")
    if query.get("is_cell_data") is not None and query.get("is_cell_data") == record.get("is_cell_data"):
        score += 0.8
        reasons.append(f"same cell-data flag={record.get('is_cell_data')}")
    token_overlap = set(query.get("tokens", [])) & set(record.get("tokens", []))
    if token_overlap:
        score += min(1.0, 0.25 * len(token_overlap))
        reasons.append("name/profile token overlap: " + ",".join(sorted(token_overlap)[:4]))

    q_metrics = query.get("raw_metrics", {}) if isinstance(query.get("raw_metrics"), dict) else {}
    r_metrics = record.get("raw_metrics", {}) if isinstance(record.get("raw_metrics"), dict) else {}
    shape = _shape_score(q_metrics.get("shape_thw"), r_metrics.get("shape_thw"))
    if shape > 0:
        score += 1.5 * shape
        reasons.append(f"similar shape score={shape:.2f}")
    snr_score = _distance_score(q_metrics.get("snr"), r_metrics.get("snr"), scale=80.0)
    if snr_score > 0:
        score += 1.0 * snr_score
        reasons.append(f"similar raw SNR score={snr_score:.2f}")
    motion_score = _distance_score(q_metrics.get("motion_p95_px"), r_metrics.get("motion_p95_px"), scale=5.0)
    if motion_score > 0:
        score += 0.7 * motion_score
        reasons.append(f"similar motion p95 score={motion_score:.2f}")

    q_outcome = query.get("current_outcome", {}) if isinstance(query.get("current_outcome"), dict) else {}
    r_outcome = record.get("final_outcome", {}) if isinstance(record.get("final_outcome"), dict) else {}
    ratio_score = _distance_score(
        q_outcome.get("snr_ratio_final_over_raw"),
        r_outcome.get("snr_ratio_final_over_raw"),
        scale=2.0,
    )
    if ratio_score > 0:
        score += 0.8 * ratio_score
        reasons.append(f"similar final/raw SNR ratio score={ratio_score:.2f}")
    if r_outcome.get("delta_snr") is not None:
        score += 0.3
        reasons.append("has final improvement outcome")
    if record.get("advisor_summaries"):
        score += 0.2
        reasons.append("has advisor history")
    quality = record.get("record_quality", {}) if isinstance(record.get("record_quality"), dict) else {}
    confidence = _safe_float(quality.get("confidence"))
    if confidence is not None and confidence < 1.0:
        score *= max(0.0, min(confidence, 1.0))
        reasons.append(f"record quality multiplier={confidence:.2f} ({quality.get('level')})")
    return float(score), reasons


def retrieve_similar_experiments(
    context: dict[str, Any],
    *,
    runs_roots: Iterable[str | Path],
    current_run_root: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    index = build_experiment_index(runs_roots)
    query = _query_summary(context)
    current_key = None
    if current_run_root:
        try:
            current_key = str(Path(current_run_root).expanduser().resolve()).lower()
        except Exception:
            current_key = str(current_run_root).lower()
    matches: list[dict[str, Any]] = []
    excluded_current = 0
    for record in index.get("records", []):
        record_key = str(record.get("run_root") or "").lower()
        if current_key and record_key == current_key:
            excluded_current += 1
            continue
        score, reasons = score_experiment(query, record)
        if score <= 0:
            continue
        matches.append({"score": score, "matched_reasons": reasons, "record": record})
    matches.sort(key=lambda item: item["score"], reverse=True)
    compact: list[dict[str, Any]] = []
    for rank, item in enumerate(matches[: max(0, int(top_k))], start=1):
        record = item["record"]
        compact.append(
            {
                "rank": rank,
                "score": round(float(item["score"]), 4),
                "run_root": record.get("run_root"),
                "folder_name": record.get("folder_name"),
                "stack_name": record.get("stack_name"),
                "dataset_profile": record.get("dataset_profile"),
                "is_cell_data": record.get("is_cell_data"),
                "record_quality": record.get("record_quality"),
                "raw_metrics": record.get("raw_metrics"),
                "final_outcome": record.get("final_outcome"),
                "iteration_params": record.get("iteration_params", []),
                "advisor_summaries": record.get("advisor_summaries", []),
                "report_html": record.get("report_html"),
                "matched_reasons": item["matched_reasons"],
            }
        )
    meta = {
        "candidate_count": len(index.get("records", [])),
        "excluded_current_run_count": excluded_current,
        "returned_count": len(compact),
    }
    return compact, query, {"index": index, "meta": meta}


def _literature_query_from_context(context: dict[str, Any], experiment_query: dict[str, Any]) -> str:
    parts: list[str] = [
        "neural imaging",
        "calcium imaging",
        "denoising",
        "motion correction",
        "source extraction",
    ]
    folder_name = str(experiment_query.get("folder_name") or context.get("folder_name") or "")
    stack_name = str(experiment_query.get("stack_name") or context.get("stack_name") or "")
    dataset_profile = str(experiment_query.get("dataset_profile") or context.get("dataset_profile") or "")
    if dataset_profile:
        parts.append(dataset_profile)
    if experiment_query.get("is_cell_data") or context.get("is_cell_data"):
        parts.extend(["cell data", "cell segmentation", "neuronal signal extraction"])
    name_text = f"{folder_name} {stack_name}".lower()
    if "24" in name_text or "24h" in name_text:
        parts.extend(["longitudinal imaging", "24-hour recording", "time series"])
    if any(term in name_text for term in ("miniscope", "snl", "deep", "hypothalamus", "vmh", "mpoa")):
        parts.extend(["miniscope", "deep brain imaging", "microendoscopy"])

    raw_metrics = experiment_query.get("raw_metrics", {}) if isinstance(experiment_query.get("raw_metrics"), dict) else {}
    snr = _safe_float(raw_metrics.get("snr"))
    motion_p95 = _safe_float(raw_metrics.get("motion_p95_px"))
    jitter_p95 = _safe_float(raw_metrics.get("jitter_p95_px"))
    bleaching = _safe_float(raw_metrics.get("bleaching_drop_percent"))
    shape = raw_metrics.get("shape_thw")
    if isinstance(shape, list) and shape:
        frames = _safe_int(shape[0])
        if frames is not None and frames >= 1000:
            parts.extend(["long recording", "large calcium imaging stack"])
    if snr is not None:
        parts.append("low SNR" if snr < 15 else "SNR quality")
    if (motion_p95 is not None and motion_p95 >= 20) or (jitter_p95 is not None and jitter_p95 >= 20):
        parts.extend(["large motion", "motion artifact", "non-rigid registration"])
    if bleaching is not None and bleaching >= 5:
        parts.extend(["photobleaching", "bleaching correction"])

    params = experiment_query.get("current_params", {}) if isinstance(experiment_query.get("current_params"), dict) else {}
    sample_mode = str(params.get("sample_mode") or "").upper()
    if sample_mode == "T":
        parts.extend(["temporal denoising", "temporal patch", "patch_t"])
    elif sample_mode == "XY":
        parts.extend(["spatial denoising", "spatial patch"])
    if params:
        parts.extend(["training parameters", "n_epochs", "batch size"])

    current_outcome = experiment_query.get("current_outcome", {}) if isinstance(experiment_query.get("current_outcome"), dict) else {}
    if _safe_float(current_outcome.get("delta_snr")) is not None:
        parts.extend(["SNR improvement", "restoration quality"])
    if _safe_float(current_outcome.get("delta_motion_p95_px")) is not None:
        parts.extend(["motion reduction", "registration quality"])

    tokens = experiment_query.get("tokens")
    if isinstance(tokens, list):
        for token in tokens:
            token_text = str(token)
            if len(token_text) >= 3 and not token_text.isdigit():
                parts.append(token_text)

    parts.extend(["DeepCAD", "NeuroPilot", "closed-loop advisor"])
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = str(part).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return " ".join(deduped)


def build_advisor_retrieval_bundle(
    context: dict[str, Any],
    *,
    runs_roots: Iterable[str | Path] | None = None,
    current_run_root: str | Path | None = None,
    top_k: int = DEFAULT_TOP_K,
    index_output_path: str | Path | None = None,
    literature_enabled: bool | None = None,
    literature_chunks_path: str | Path | None = None,
    literature_top_k: int | None = None,
    literature_max_chunks_per_paper: int | None = None,
) -> dict[str, Any]:
    if runs_roots is None:
        runs_roots = parse_runs_roots(None, current_run_root=current_run_root)
    roots = [Path(root).expanduser() for root in runs_roots]
    matches, query, retrieval_state = retrieve_similar_experiments(
        context,
        runs_roots=roots,
        current_run_root=current_run_root,
        top_k=top_k,
    )
    if index_output_path:
        _write_json(index_output_path, retrieval_state["index"])
    lit_enabled = _env_bool("NEUROPILOT_RAG_LITERATURE_ENABLED", True) if literature_enabled is None else bool(literature_enabled)
    lit_top_k = (
        max(0, _env_int("NEUROPILOT_RAG_LITERATURE_TOP_K", DEFAULT_LITERATURE_TOP_K))
        if literature_top_k is None
        else max(0, int(literature_top_k))
    )
    lit_max_per_paper = (
        max(1, _env_int("NEUROPILOT_RAG_LITERATURE_MAX_CHUNKS_PER_PAPER", DEFAULT_LITERATURE_MAX_CHUNKS_PER_PAPER))
        if literature_max_chunks_per_paper is None
        else max(1, int(literature_max_chunks_per_paper))
    )
    lit_path = _resolve_literature_chunks_path(literature_chunks_path)
    literature: list[dict[str, Any]] = []
    literature_retrieval: dict[str, Any] = {
        "enabled": lit_enabled,
        "chunks_path": str(lit_path.resolve()) if lit_path.exists() else str(lit_path),
        "top_k": lit_top_k,
        "max_chunks_per_paper": lit_max_per_paper,
        "status": "disabled" if not lit_enabled or lit_top_k <= 0 else "not_run",
    }
    if lit_enabled and lit_top_k > 0:
        if retrieve_literature is None:
            literature_retrieval.update(
                {
                    "status": "unavailable",
                    "error_type": "ImportError",
                    "error": "rag.literature_index.retrieve_literature is not available",
                }
            )
        elif not lit_path.exists():
            literature_retrieval.update(
                {
                    "status": "missing_chunks",
                    "error_type": "FileNotFoundError",
                    "error": f"literature chunks file not found: {lit_path}",
                }
            )
        else:
            try:
                literature_query = _literature_query_from_context(context, query)
                lit_result = retrieve_literature(
                    literature_query,
                    chunks_path=lit_path,
                    top_k=lit_top_k,
                    max_chunks_per_paper=lit_max_per_paper,
                )
                literature = lit_result.get("matched_literature", [])
                literature_retrieval.update(
                    {
                        "status": "retrieved",
                        "schema_version": lit_result.get("schema_version"),
                        "retrieval_mode": lit_result.get("retrieval_mode"),
                        "candidate_count": lit_result.get("candidate_count", 0),
                        "returned_count": len(literature),
                        "query": lit_result.get("query", {}),
                    }
                )
            except Exception as exc:
                literature_retrieval.update(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    retrieval_mode = "local_experiment_history"
    if literature_retrieval.get("status") == "retrieved":
        retrieval_mode = "local_experiment_history+local_literature_bm25"
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "retrieval_mode": retrieval_mode,
        "query": query,
        "sources": {
            "runs_roots": [str(root.resolve()) for root in roots if root.exists()],
            "index_output_path": str(Path(index_output_path).resolve()) if index_output_path else None,
            "literature_chunks_path": literature_retrieval.get("chunks_path"),
        },
        "candidate_count": retrieval_state["meta"]["candidate_count"],
        "excluded_current_run_count": retrieval_state["meta"]["excluded_current_run_count"],
        "matched_experiments": matches,
        "literature": literature,
        "literature_retrieval": literature_retrieval,
        "notes": [
            "M1/M2 retrieves local historical NeuroPilot runs.",
            "M3.4 retrieves local literature chunks when literature retrieval is enabled and the chunks index is available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the local NeuroPilot experiment index.")
    parser.add_argument("--runs-root", action="append", default=None, help="Runs root to scan. Can be passed more than once.")
    parser.add_argument("--output", default=None, help="Write the generated experiment index JSON here.")
    parser.add_argument("--query-run-root", default=None, help="Build an advisor retrieval bundle for this run root.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--literature-chunks", default=None, help="M3 literature chunks JSONL path.")
    parser.add_argument("--literature-top-k", type=int, default=None, help="Number of literature chunks to return.")
    parser.add_argument(
        "--literature-max-chunks-per-paper",
        type=int,
        default=None,
        help="Maximum returned chunks per paper.",
    )
    parser.add_argument("--disable-literature", action="store_true", help="Disable M3 literature retrieval for this query.")
    args = parser.parse_args()

    runs_roots = args.runs_root or [str(path) for path in parse_runs_roots(None, current_run_root=args.query_run_root)]
    if args.query_run_root:
        context = build_context_from_run_root(args.query_run_root)
        bundle = build_advisor_retrieval_bundle(
            context,
            runs_roots=runs_roots,
            current_run_root=args.query_run_root,
            top_k=args.top_k,
            index_output_path=args.output,
            literature_enabled=not args.disable_literature,
            literature_chunks_path=args.literature_chunks,
            literature_top_k=args.literature_top_k,
            literature_max_chunks_per_paper=args.literature_max_chunks_per_paper,
        )
        _print_json(bundle)
        return

    index = build_experiment_index(runs_roots)
    if args.output:
        _write_json(args.output, index)
    _print_json(index)


if __name__ == "__main__":
    main()
