#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ALLOWED_SAMPLE_MODES = ("XY", "T", "N2V")
DEFAULT_EPOCH_RANGE = (1, 300)
DEFAULT_BATCH_SIZE_RANGE = (1, 16)
OPTIMIZATION_PRIORITIES = (
    "balanced",
    "snr_priority",
    "motion_priority",
    "segmentation_priority",
)
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
LOCAL_CONFIG_CANDIDATES = ("advisor_config.json", "configs/advisor_config.json")
ENV_FILE_CANDIDATES = (".env", ".env.example")
API_KEY_PLACEHOLDER_MARKERS = (
    "your_openai_api_key",
    "replace_me",
    "example",
    "<api_key>",
    "sk-xxxx",
)


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {'"', "'"}
        ):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _bootstrap_local_env() -> None:
    roots = [
        Path.cwd(),
        Path(__file__).resolve().parent,
    ]
    seen: set[str] = set()
    for root in roots:
        root_key = str(root.resolve()).lower()
        if root_key in seen:
            continue
        seen.add(root_key)
        for file_name in ENV_FILE_CANDIDATES:
            _load_env_file(root / file_name, override=False)


_bootstrap_local_env()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _to_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _clamp_int(value: int, low: int, high: int) -> int:
    low = int(low)
    high = int(high)
    if high < low:
        high = low
    return max(low, min(int(value), high))


def _prefer_even(value: int, low: int, high: int) -> int:
    value = int(value)
    low = int(low)
    high = int(high)
    if value % 2 == 0:
        return value
    if value - 1 >= low:
        return value - 1
    if value + 1 <= high:
        return value + 1
    return value


def _load_local_config() -> tuple[dict[str, Any], str | None]:
    env_path = os.getenv("LLM_ADVISOR_CONFIG_PATH")
    candidates = [env_path] if env_path else list(LOCAL_CONFIG_CANDIDATES)
    for p in candidates:
        if not p:
            continue
        path = Path(p)
        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data, str(path.resolve())
            except Exception:
                continue
    return {}, None


def _has_real_api_key_candidate(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in API_KEY_PLACEHOLDER_MARKERS)


def has_configured_api_key(api_key: str | None = None, api_key_env: str = "OPENAI_API_KEY") -> bool:
    explicit_api_key = api_key if _has_real_api_key_candidate(api_key) else None
    local_cfg, _ = _load_local_config()
    cfg_api_key = local_cfg.get("api_key") or local_cfg.get("ADVISOR_API_KEY")
    env_api_key = os.getenv(api_key_env)
    return any(
        _has_real_api_key_candidate(candidate)
        for candidate in (explicit_api_key, cfg_api_key, env_api_key)
    )


def _normalize_base_url(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _normalize_sample_mode(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s in ALLOWED_SAMPLE_MODES else None


def _to_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
    if isinstance(x, str):
        v = x.strip().lower()
        if v in ("true", "1", "yes", "y"):
            return True
        if v in ("false", "0", "no", "n"):
            return False
    return None


def _to_int(x: Any) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(round(x))
    if isinstance(x, str):
        m = re.search(r"-?\d+", x.strip())
        if m:
            return int(m.group(0))
    return None


def _extract_default_params(context: dict[str, Any]) -> dict[str, Any]:
    ctx = context if isinstance(context, dict) else {}
    current = ctx.get("current_params", {}) if isinstance(ctx.get("current_params"), dict) else {}
    suggestion_target = ctx.get("suggestion_target", {}) if isinstance(ctx.get("suggestion_target"), dict) else {}
    default_params = suggestion_target.get("default_params", {}) if isinstance(suggestion_target.get("default_params"), dict) else {}
    base = default_params if default_params else current
    sample_mode = (
        _normalize_sample_mode(base.get("sample_mode"))
        or _normalize_sample_mode(suggestion_target.get("sample_mode_locked"))
        or _normalize_sample_mode(current.get("sample_mode"))
        or "T"
    )
    return {
        "sample_mode": sample_mode,
        "n_epochs": _to_int(base.get("n_epochs")) or _to_int(current.get("n_epochs")) or 30,
        "patch_x": _to_int(base.get("patch_x")) or _to_int(current.get("patch_x")) or 128,
        "patch_y": _to_int(base.get("patch_y")) or _to_int(current.get("patch_y")) or 128,
        "patch_t": _to_int(base.get("patch_t")) or _to_int(current.get("patch_t")) or 128,
        "batch_size": _to_int(base.get("batch_size")) or _to_int(current.get("batch_size")) or 6,
    }


def _normalize_bounds(
    raw_bounds: Any,
    fallback: tuple[int, int],
    field_name: str,
    errors: list[str],
) -> tuple[int, int]:
    if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
        errors.append(f"invalid {field_name}; fallback to default bounds")
        return int(fallback[0]), int(fallback[1])
    low = _to_int(raw_bounds[0])
    high = _to_int(raw_bounds[1])
    if low is None or high is None:
        errors.append(f"non-integer {field_name}; fallback to default bounds")
        return int(fallback[0]), int(fallback[1])
    if low > high:
        low, high = high, low
        errors.append(f"{field_name} reversed; auto-corrected")
    return int(low), int(high)


def _extract_bounds(context: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, tuple[int, int]], list[str]]:
    ctx = context if isinstance(context, dict) else {}
    suggestion_target = ctx.get("suggestion_target", {}) if isinstance(ctx.get("suggestion_target"), dict) else {}
    raw_bounds = suggestion_target.get("bounds", {}) if isinstance(suggestion_target.get("bounds"), dict) else {}
    errors: list[str] = []
    bounds = {
        "n_epochs": _normalize_bounds(
            raw_bounds.get("n_epochs", ctx.get("n_epochs_bounds")),
            DEFAULT_EPOCH_RANGE,
            "n_epochs_bounds",
            errors,
        ),
        "patch_x": _normalize_bounds(
            raw_bounds.get("patch_x"),
            (1, max(1, int(defaults["patch_x"]))),
            "patch_x_bounds",
            errors,
        ),
        "patch_y": _normalize_bounds(
            raw_bounds.get("patch_y"),
            (1, max(1, int(defaults["patch_y"]))),
            "patch_y_bounds",
            errors,
        ),
        "patch_t": _normalize_bounds(
            raw_bounds.get("patch_t"),
            (1, max(1, int(defaults["patch_t"]))),
            "patch_t_bounds",
            errors,
        ),
        "batch_size": _normalize_bounds(
            raw_bounds.get("batch_size"),
            DEFAULT_BATCH_SIZE_RANGE,
            "batch_size_bounds",
            errors,
        ),
    }
    return bounds, errors


def _sanitize_field(
    field_name: str,
    suggestion: dict[str, Any],
    defaults: dict[str, Any],
    bounds: dict[str, tuple[int, int]],
    errors: list[str],
) -> int:
    raw_value = suggestion.get(field_name)
    parsed = _to_int(raw_value)
    if parsed is None:
        errors.append(f"invalid or missing {field_name}; fallback to default {field_name}")
        parsed = int(defaults[field_name])
    low, high = bounds[field_name]
    clamped = _clamp_int(parsed, low, high)
    if clamped != parsed:
        errors.append(f"{field_name} out of bounds; clamped")
    return int(clamped)


def _extract_epoch_options(context: dict[str, Any], defaults: dict[str, Any]) -> list[int]:
    ctx = context if isinstance(context, dict) else {}
    suggestion_target = ctx.get("suggestion_target", {}) if isinstance(ctx.get("suggestion_target"), dict) else {}
    raw_options = suggestion_target.get("epoch_options")
    options: list[int] = []
    if isinstance(raw_options, (list, tuple)):
        for item in raw_options:
            parsed = _to_int(item)
            if parsed is not None and parsed > 0 and parsed not in options:
                options.append(int(parsed))
    if not options:
        options = [int(defaults["n_epochs"])]
    return sorted(options)


def _extract_field_options(
    context: dict[str, Any],
    field_name: str,
    defaults: dict[str, Any],
) -> list[int]:
    ctx = context if isinstance(context, dict) else {}
    suggestion_target = ctx.get("suggestion_target", {}) if isinstance(ctx.get("suggestion_target"), dict) else {}
    raw_options = None
    if field_name == "batch_size":
        raw_options = suggestion_target.get("batch_options")
    else:
        patch_options = suggestion_target.get("patch_options")
        if isinstance(patch_options, dict):
            raw_options = patch_options.get(field_name)
    options: list[int] = []
    if isinstance(raw_options, (list, tuple)):
        for item in raw_options:
            parsed = _to_int(item)
            if parsed is not None and parsed > 0 and parsed not in options:
                options.append(int(parsed))
    if not options and field_name in defaults:
        options = [int(defaults[field_name])]
    return sorted(options)


def _snap_to_options(value: int, options: list[int]) -> int:
    if not options:
        return int(value)
    return int(min(options, key=lambda option: (abs(int(option) - int(value)), int(option))))


def _sanitize_priority(value: Any, errors: list[str]) -> str:
    priority = str(value or "").strip().lower()
    if priority not in OPTIMIZATION_PRIORITIES:
        if priority:
            errors.append(f"invalid optimization_priority={priority}; fallback to balanced")
        else:
            errors.append("missing optimization_priority; fallback to balanced")
        return "balanced"
    return priority


def _sanitize_text(value: Any, field_name: str, default: str, errors: list[str], max_len: int = 4000) -> str:
    if value is None:
        errors.append(f"missing {field_name}; default applied")
        return default
    text = str(value).strip()
    if not text:
        errors.append(f"empty {field_name}; default applied")
        return default
    return text[:max_len]


def _build_fallback_suggestion(context: dict[str, Any], reason: str) -> dict[str, Any]:
    defaults = _extract_default_params(context)
    return {
        "n_epochs": defaults["n_epochs"],
        "patch_x": defaults["patch_x"],
        "patch_y": defaults["patch_y"],
        "patch_t": defaults["patch_t"],
        "batch_size": defaults["batch_size"],
        "continue_iteration": True,
        "optimization_priority": "balanced",
        "stop_reason": f"fallback; continue conservatively: {reason}",
        "reason": f"fallback: {reason}",
    }


def validate_and_sanitize_suggestion(
    suggestion: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    ctx = context or {}
    suggestion = suggestion if isinstance(suggestion, dict) else {}
    defaults = _extract_default_params(ctx)
    bounds, errors = _extract_bounds(ctx, defaults)

    n_epochs = _sanitize_field("n_epochs", suggestion, defaults, bounds, errors)
    patch_x = _sanitize_field("patch_x", suggestion, defaults, bounds, errors)
    patch_y = _sanitize_field("patch_y", suggestion, defaults, bounds, errors)
    patch_t = _sanitize_field("patch_t", suggestion, defaults, bounds, errors)
    batch_size = _sanitize_field("batch_size", suggestion, defaults, bounds, errors)
    epoch_options = _extract_epoch_options(ctx, defaults)
    snapped_n_epochs = _snap_to_options(n_epochs, epoch_options)
    if snapped_n_epochs != n_epochs:
        errors.append(f"n_epochs snapped to allowed epoch option {snapped_n_epochs}")
        n_epochs = snapped_n_epochs

    for field_name, value in (
        ("patch_x", patch_x),
        ("patch_y", patch_y),
        ("patch_t", patch_t),
        ("batch_size", batch_size),
    ):
        options = _extract_field_options(ctx, field_name, defaults)
        snapped = _snap_to_options(value, options)
        if snapped != value:
            errors.append(f"{field_name} snapped to allowed option {snapped}")
        if field_name == "patch_x":
            patch_x = snapped
        elif field_name == "patch_y":
            patch_y = snapped
        elif field_name == "patch_t":
            patch_t = snapped
        elif field_name == "batch_size":
            batch_size = snapped

    if defaults["sample_mode"] == "XY":
        adjusted = _prefer_even(patch_x, bounds["patch_x"][0], bounds["patch_x"][1])
        if adjusted != patch_x:
            errors.append("patch_x adjusted to nearest even value for XY mode")
            patch_x = adjusted
    elif defaults["sample_mode"] == "T":
        adjusted = _prefer_even(patch_t, bounds["patch_t"][0], bounds["patch_t"][1])
        if adjusted != patch_t:
            errors.append("patch_t adjusted to nearest even value for T mode")
            patch_t = adjusted

    continue_iteration = _to_bool(suggestion.get("continue_iteration"))
    if continue_iteration is None:
        continue_iteration = True
        errors.append("invalid or missing continue_iteration; fallback to True")

    optimization_priority = _sanitize_priority(suggestion.get("optimization_priority"), errors)
    stop_reason = _sanitize_text(
        suggestion.get("stop_reason"),
        "stop_reason",
        "continue by default; no stop reason provided",
        errors,
    )
    reason = _sanitize_text(suggestion.get("reason"), "reason", "no reason provided", errors)

    sanitized = {
        "schema_version": "llm_advisor.suggestion.v2",
        "n_epochs": int(n_epochs),
        "patch_x": int(patch_x),
        "patch_y": int(patch_y),
        "patch_t": int(patch_t),
        "batch_size": int(batch_size),
        "continue_iteration": bool(continue_iteration),
        "optimization_priority": optimization_priority,
        "stop_reason": stop_reason,
        "reason": reason[:4000],
    }
    return sanitized, errors


def _build_mock_suggestion(context: dict[str, Any]) -> dict[str, Any]:
    defaults = _extract_default_params(context)
    bounds, _ = _extract_bounds(context, defaults)
    metrics = context.get("metrics", {}) if isinstance(context, dict) else {}
    runtime = {}
    if isinstance(context.get("current_runtime"), dict):
        runtime = context.get("current_runtime", {}).get("train", {})
    snr_val = metrics.get("snr_metric", {}).get("snr")
    motion_p95_val = metrics.get("rigid_motion_metric", {}).get("rigid_motion_summary", {}).get("motion_p95_px")
    bleaching_drop_val = metrics.get("bleaching_trend", {}).get("relative_drop_percent")
    sec_per_epoch_val = runtime.get("seconds_per_epoch") if isinstance(runtime, dict) else None

    snr = _to_float(snr_val)
    mp95 = _to_float(motion_p95_val)
    bdrop = _to_float(bleaching_drop_val)
    sec_per_epoch = _to_float(sec_per_epoch_val)

    n_epochs = int(defaults["n_epochs"])
    patch_x = int(defaults["patch_x"])
    patch_y = int(defaults["patch_y"])
    patch_t = int(defaults["patch_t"])
    batch_size = int(defaults["batch_size"])
    reason_parts = ["mock heuristic"]

    if snr is not None and snr < 3.0:
        n_epochs += 10
        patch_x = int(round(patch_x * 1.25))
        patch_y = int(round(patch_y * 1.25))
        reason_parts.append("low snr -> more epochs and slightly larger spatial patches")
    elif bdrop is not None and bdrop > 20.0:
        n_epochs += 5
        reason_parts.append("strong bleaching -> moderate epoch increase")

    if mp95 is not None and mp95 >= 5.0:
        patch_t = int(round(patch_t * 1.25))
        reason_parts.append("high motion p95 -> slightly larger temporal patch")

    if sec_per_epoch is not None:
        if sec_per_epoch > 120.0:
            batch_size -= 2
            reason_parts.append("slow training speed -> lower batch size")
        elif sec_per_epoch < 45.0:
            batch_size += 2
            reason_parts.append("fast training speed -> higher batch size")

    return {
        "n_epochs": _clamp_int(n_epochs, bounds["n_epochs"][0], bounds["n_epochs"][1]),
        "patch_x": _clamp_int(patch_x, bounds["patch_x"][0], bounds["patch_x"][1]),
        "patch_y": _clamp_int(patch_y, bounds["patch_y"][0], bounds["patch_y"][1]),
        "patch_t": _clamp_int(patch_t, bounds["patch_t"][0], bounds["patch_t"][1]),
        "batch_size": _clamp_int(batch_size, bounds["batch_size"][0], bounds["batch_size"][1]),
        "continue_iteration": True,
        "optimization_priority": "balanced",
        "stop_reason": "mock mode continues conservatively",
        "reason": "; ".join(reason_parts),
    }


def _context_without_literature_evidence(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict):
        return context
    retrieval = context.get("retrieval_context")
    if not isinstance(retrieval, dict):
        return context
    cleaned_retrieval = dict(retrieval)
    cleaned_retrieval.pop("literature", None)
    cleaned_retrieval.pop("literature_retrieval", None)
    if str(cleaned_retrieval.get("retrieval_mode") or "").endswith("+local_literature_bm25"):
        cleaned_retrieval["retrieval_mode"] = "local_experiment_history"
    notes = cleaned_retrieval.get("notes")
    cleaned_retrieval["notes"] = [
        *(notes if isinstance(notes, list) else []),
        "Literature evidence is excluded from parameter-advisor requests.",
    ]
    cleaned_context = dict(context)
    cleaned_context["retrieval_context"] = cleaned_retrieval
    return cleaned_context


def _build_live_request_payload(
    context: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    context_for_advisor = _context_without_literature_evidence(context)
    system_prompt = (
        "You are a strict next-iteration training parameter advisor for an imaging pipeline. "
        "sample_mode is locked by the pipeline and must not be suggested. "
        "Only use epoch values from context.suggestion_target.epoch_options. "
        "Only use patch values from context.suggestion_target.patch_options and batch values from "
        "context.suggestion_target.batch_options. "
        "When retrieval_context is present, use only matched_experiments as local historical evidence. "
        "Do not use literature chunks or paper citations to justify parameter values. "
        "In reason, cite the most relevant run names when they influence your recommendation. "
        "Set continue_iteration=false only when enough iterations have completed and current metrics suggest additional "
        "training is likely marginal or risky. "
        "Return valid JSON only. No markdown."
    )
    user_prompt = {
        "task": "Suggest next-iteration training parameters for the unlocked fields only.",
        "constraints": {
            "locked_by_pipeline": {
                "sample_mode": "Do not suggest sample_mode. Use the pipeline default schedule already provided in context.",
            },
            "retrieval_context": {
                "usage": (
                    "Use matched_experiments as supporting evidence only. "
                    "Do not copy settings blindly when metrics, dataset profile, or shape differ. "
                    "Prefer current metrics and pipeline bounds when evidence conflicts."
                ),
                "experiment_citation_requirement": (
                    "If matched_experiments are present and relevant, mention the key run_root or stack_name evidence in reason."
                ),
                "literature_policy": "Literature evidence is intentionally excluded from parameter advice.",
            },
            "allowed_fields": [
                "n_epochs",
                "patch_x",
                "patch_y",
                "patch_t",
                "batch_size",
                "continue_iteration",
                "optimization_priority",
                "stop_reason",
                "reason",
            ],
            "note": (
                "continue_iteration may be used by the pipeline control layer after minimum iteration guardrails. "
                "n_epochs, patch_x, patch_y, patch_t, and batch_size are snapped to the finite options supplied in context.suggestion_target. "
                "Medium-risk mode allows finite patch and batch buckets; sample_mode remains locked. "
                "optimization_priority must be one of balanced, snr_priority, motion_priority, segmentation_priority. "
                "stop_reason is required even when continue_iteration=true; explain the stop or continue decision."
            ),
        },
        "context": context_for_advisor,
        "output_requirement": {
            "must_be_json_object": True,
            "keys": [
                "n_epochs",
                "patch_x",
                "patch_y",
                "patch_t",
                "batch_size",
                "continue_iteration",
                "optimization_priority",
                "stop_reason",
                "reason",
            ],
        },
    }

    json_schema = {
        "name": "llm_advisor_suggestion_v2",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "n_epochs": {"type": "integer"},
                "patch_x": {"type": "integer"},
                "patch_y": {"type": "integer"},
                "patch_t": {"type": "integer"},
                "batch_size": {"type": "integer"},
                "continue_iteration": {"type": "boolean"},
                "optimization_priority": {
                    "type": "string",
                    "enum": list(OPTIMIZATION_PRIORITIES),
                },
                "stop_reason": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": [
                "n_epochs",
                "patch_x",
                "patch_y",
                "patch_t",
                "batch_size",
                "continue_iteration",
                "optimization_priority",
                "stop_reason",
                "reason",
            ],
        },
    }

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }


def _call_openai_chat_completions(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    endpoint = _normalize_base_url(base_url)
    req = request.Request(
        url=endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _extract_json_from_live_response(response_obj: dict[str, Any]) -> dict[str, Any]:
    choices = response_obj.get("choices", [])
    if not choices:
        raise ValueError("live response missing choices")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("live response missing message.content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("live response content is not a JSON object")
    return parsed


def get_llm_suggestion(
    context: dict[str, Any],
    mode: str = "mock",
    *,
    api_key: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    model: str | None = None,
    base_url: str | None = None,
    timeout_s: float = 20.0,
    logs_dir: str | Path | None = None,
    mock_response: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main API:
      - mode="mock": deterministic local suggestion
      - mode="live": call OpenAI API
    Always returns validated/sanitized Python dict.
    Falls back on timeout/network/JSON/validation errors.
    """
    mode_norm = str(mode).strip().lower()
    if mode_norm not in ("mock", "live"):
        mode_norm = "mock"

    logs_path = None if logs_dir is None else Path(logs_dir)
    if logs_path is not None:
        logs_path.mkdir(parents=True, exist_ok=True)

    local_cfg, local_cfg_path = _load_local_config()

    # Resolution precedence:
    # explicit arg > local config > environment variable > default
    explicit_model = model if (isinstance(model, str) and model.strip()) else None
    explicit_base_url = base_url if (isinstance(base_url, str) and base_url.strip()) else None
    explicit_api_key = api_key if (isinstance(api_key, str) and api_key.strip()) else None
    # With signature timeout_s=20.0, we treat 20.0 as "not explicitly overridden"
    # so config/env can still provide timeout.
    explicit_timeout = _to_float(timeout_s)
    timeout_is_explicit = explicit_timeout is not None and float(explicit_timeout) != 20.0

    cfg_model = local_cfg.get("model") or local_cfg.get("ADVISOR_MODEL")
    cfg_base_url = local_cfg.get("base_url") or local_cfg.get("ADVISOR_BASE_URL")
    cfg_api_key = local_cfg.get("api_key") or local_cfg.get("ADVISOR_API_KEY")
    cfg_timeout = local_cfg.get("timeout_s") or local_cfg.get("ADVISOR_TIMEOUT_S")

    env_model = os.getenv("LLM_ADVISOR_MODEL")
    env_base_url = os.getenv("LLM_ADVISOR_BASE_URL")
    env_api_key = os.getenv(api_key_env)
    env_timeout = os.getenv("LLM_ADVISOR_TIMEOUT_S")

    model_used = (
        explicit_model
        or (cfg_model if isinstance(cfg_model, str) and cfg_model.strip() else None)
        or (env_model if isinstance(env_model, str) and env_model.strip() else None)
        or DEFAULT_MODEL
    )
    model_source = (
        "explicit" if explicit_model else
        "config" if (isinstance(cfg_model, str) and cfg_model.strip()) else
        "env" if (isinstance(env_model, str) and env_model.strip()) else
        "default"
    )

    base_url_used = (
        explicit_base_url
        or (cfg_base_url if isinstance(cfg_base_url, str) and cfg_base_url.strip() else None)
        or (env_base_url if isinstance(env_base_url, str) and env_base_url.strip() else None)
        or DEFAULT_BASE_URL
    )
    base_url_source = (
        "explicit" if explicit_base_url else
        "config" if (isinstance(cfg_base_url, str) and cfg_base_url.strip()) else
        "env" if (isinstance(env_base_url, str) and env_base_url.strip()) else
        "default"
    )

    timeout_used = (
        explicit_timeout
        if timeout_is_explicit else
        (_to_float(cfg_timeout) if _to_float(cfg_timeout) is not None else None)
    )
    if timeout_used is None:
        timeout_used = _to_float(env_timeout)
    if timeout_used is None:
        timeout_used = 20.0
    timeout_source = (
        "explicit" if timeout_is_explicit else
        "config" if _to_float(cfg_timeout) is not None else
        "env" if _to_float(env_timeout) is not None else
        "default"
    )

    api_key_used = (
        explicit_api_key
        or (cfg_api_key if isinstance(cfg_api_key, str) and cfg_api_key.strip() else None)
        or (env_api_key if isinstance(env_api_key, str) and env_api_key.strip() else None)
        or None
    )
    api_key_source = (
        "explicit" if explicit_api_key else
        "config" if (isinstance(cfg_api_key, str) and cfg_api_key.strip()) else
        "env" if (isinstance(env_api_key, str) and env_api_key.strip()) else
        "missing"
    )

    request_payload: dict[str, Any] | None = None
    raw_response_obj: Any = None
    candidate: dict[str, Any] | None = None
    fallback_reason: str | None = None

    if mode_norm == "mock":
        candidate = mock_response if isinstance(mock_response, dict) else _build_mock_suggestion(context)
    else:
        request_payload = _build_live_request_payload(context=context, model=model_used)
        if not api_key_used:
            fallback_reason = f"missing api key (resolved source={api_key_source})"
        else:
            try:
                raw_response_obj = _call_openai_chat_completions(
                    payload=request_payload,
                    api_key=api_key_used,
                    base_url=base_url_used,
                    timeout_s=float(timeout_used),
                )
                candidate = _extract_json_from_live_response(raw_response_obj)
            except (error.HTTPError, error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                fallback_reason = f"live call failed: {type(exc).__name__}: {exc}"
            except Exception as exc:  # defensive catch
                fallback_reason = f"unexpected live error: {type(exc).__name__}: {exc}"

    if candidate is None:
        candidate = _build_fallback_suggestion(context, reason=fallback_reason or "unknown error")

    validated, validation_errors = validate_and_sanitize_suggestion(candidate, context=context)

    result = {
        **validated,
        "meta": {
            "advisor_mode": mode_norm,
            "backend_effective": mode_norm,
            "model_used": model_used,
            "model_source": model_source,
            "base_url_used": base_url_used,
            "base_url_source": base_url_source,
            "timeout_s_used": timeout_used,
            "timeout_source": timeout_source,
            "api_key_source": api_key_source,
            "local_config_path": local_cfg_path,
            "used_fallback": bool(fallback_reason is not None),
            "fallback_reason": fallback_reason,
            "validation_errors": validation_errors,
            "timestamp_utc": _utc_now_iso(),
        },
    }

    if logs_path is not None:
        if request_payload is not None:
            _safe_write_json(
                logs_path / "request.json",
                {
                    "backend_effective": mode_norm,
                    "model_used": model_used,
                    "base_url_used": base_url_used,
                    "api_key_source": api_key_source,
                    "timeout_s_used": timeout_used,
                    "request_payload": request_payload,
                },
            )
        else:
            _safe_write_json(
                logs_path / "request.json",
                {
                    "mode": mode_norm,
                    "backend_effective": mode_norm,
                    "model_used": model_used,
                    "base_url_used": base_url_used,
                    "api_key_source": api_key_source,
                    "timeout_s_used": timeout_used,
                    "request_source": "local_mock_or_fallback",
                    "context": context,
                },
            )
        if raw_response_obj is not None:
            _safe_write_json(logs_path / "response_raw.json", raw_response_obj)
        elif mode_norm == "mock":
            _safe_write_json(
                logs_path / "response_raw.json",
                {
                    "backend_effective": mode_norm,
                    "status": "mock_mode",
                    "live_call": "no_live_call",
                    "reason": "mock mode does not call remote API",
                },
            )
        else:
            _safe_write_json(
                logs_path / "response_raw.json",
                {
                    "backend_effective": mode_norm,
                    "status": "no_live_response",
                    "reason": fallback_reason,
                    "model_used": model_used,
                    "base_url_used": base_url_used,
                    "api_key_source": api_key_source,
                },
            )
        _safe_write_json(logs_path / "suggestion_validated.json", result)

    return result


if __name__ == "__main__":
    demo_context = {
        "folder_name": "bloodvessel_A62",
        "iter_index": 0,
        "current_params": {
            "sample_mode": "XY",
            "n_epochs": 15,
            "patch_x": 256,
            "patch_y": 128,
            "patch_t": 128,
            "batch_size": 6,
        },
        "current_runtime": {
            "train": {
                "seconds_per_epoch": 32.0,
            },
        },
        "suggestion_target": {
            "target_iter": 1,
            "sample_mode_locked": "T",
            "default_params": {
                "sample_mode": "T",
                "n_epochs": 30,
                "patch_x": 128,
                "patch_y": 128,
                "patch_t": 256,
                "batch_size": 6,
            },
            "bounds": {
                "n_epochs": [1, 100],
                "patch_x": [8, 256],
                "patch_y": [8, 256],
                "patch_t": [16, 512],
                "batch_size": [1, 12],
            },
        },
        "metrics": {
            "snr_metric": {"snr": 2.1},
            "bleaching_trend": {"relative_drop_percent": 12.0},
            "rigid_motion_metric": {"rigid_motion_summary": {"motion_p95_px": 6.4}},
        },
    }
    out = get_llm_suggestion(
        demo_context,
        mode="mock",
        logs_dir=Path(__file__).resolve().parent / "tmp_pipeline_metrics" / "llm_demo",
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


