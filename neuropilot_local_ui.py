from __future__ import annotations

import argparse
import codecs
import json
import locale
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import webbrowser
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "local_ui" / "static"
PIPELINE_ENTRY = ROOT_DIR / "neuropilot_pipeline.py"
TIFF_SUFFIXES = {".tif", ".tiff"}
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def server_context(host_header: str | None = None) -> dict[str, Any]:
    return {
        "access_scope": "server_filesystem",
        "host_header": host_header or "",
        "server_root": str(ROOT_DIR),
        "server_cwd": str(Path.cwd().resolve()),
        "path_note": (
            "Input/output paths are resolved on the machine running neuropilot_local_ui.py. "
            "When this UI is exposed as a web page, browser-local paths are not available to the server."
        ),
    }


def server_path_note() -> str:
    return (
        "This path was checked on the UI server, not on the browser user's computer. "
        "For web deployment, first copy/upload/mount TIFF data on the server, then enter that server-side path."
    )


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _candidate_log_encodings() -> list[str]:
    candidates = [
        "utf-8",
        locale.getpreferredencoding(False),
        os.device_encoding(1),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = str(candidate).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


LOG_ENCODING_CANDIDATES = _candidate_log_encodings()


def decode_log_bytes(raw: bytes) -> str:
    for encoding in LOG_ENCODING_CANDIDATES:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def resolve_user_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Path cannot be empty.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (ROOT_DIR / candidate).resolve()
    return candidate.resolve()


def list_tiffs(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def is_loose_tif_prepared_in_place(input_dir: Path, tif_path: Path) -> bool:
    prepared_copy = input_dir / tif_path.stem / tif_path.name
    return prepared_copy.is_file()


def inspect_input_layout(input_dir: Path) -> dict[str, Any]:
    prepare_command = f'python prepare_input_tiffs.py --input-dir "{input_dir}"'
    result: dict[str, Any] = {
        "input_dir": str(input_dir),
        "server_resolved_input_dir": str(input_dir),
        "server_root": str(ROOT_DIR),
        "server_cwd": str(Path.cwd().resolve()),
        "path_access_scope": "server_filesystem",
        "exists": input_dir.exists(),
        "is_dir": input_dir.is_dir(),
        "loose_tifs": [],
        "prepared_loose_tifs": [],
        "unprepared_loose_tifs": [],
        "subfolders": [],
        "valid_subfolders": [],
        "invalid_subfolders": [],
        "default_selected_subfolders": [],
        "messages": [],
        "prepare_command": prepare_command,
        "should_prepare_input": False,
        "total_tif_count": 0,
        "can_run": False,
    }
    if not input_dir.exists():
        result["messages"].append("Input directory does not exist.")
        result["messages"].append(server_path_note())
        return result
    if not input_dir.is_dir():
        result["messages"].append("Input path is not a directory.")
        result["messages"].append(server_path_note())
        return result

    loose_tifs = list_tiffs(input_dir)
    result["loose_tifs"] = [path.name for path in loose_tifs]
    result["total_tif_count"] += len(loose_tifs)
    prepared_loose_tifs = [path.name for path in loose_tifs if is_loose_tif_prepared_in_place(input_dir, path)]
    unprepared_loose_tifs = [path.name for path in loose_tifs if path.name not in prepared_loose_tifs]
    result["prepared_loose_tifs"] = prepared_loose_tifs
    result["unprepared_loose_tifs"] = unprepared_loose_tifs
    if unprepared_loose_tifs:
        result["should_prepare_input"] = True
        result["messages"].append(
            "Loose TIFF files were found at the input root. NeuroPilot expects dataset subfolders under input-dir."
        )
        result["messages"].append(
            f"To reorganize flat TIFF inputs into dataset subfolders, run: {prepare_command}"
        )

    child_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not child_dirs:
        if not loose_tifs:
            result["messages"].append("No tif/tiff files were found under input-dir.")
        result["messages"].append("No first-level dataset subfolders were found under input-dir.")
        return result

    for child_dir in child_dirs:
        tif_files = list_tiffs(child_dir)
        result["total_tif_count"] += len(tif_files)
        item = {
            "name": child_dir.name,
            "path": str(child_dir),
            "tif_count": len(tif_files),
            "sample_tifs": [path.name for path in tif_files[:10]],
            "has_tiffs": bool(tif_files),
        }
        result["subfolders"].append(item)
        if tif_files:
            result["valid_subfolders"].append(child_dir.name)
            result["default_selected_subfolders"].append(child_dir.name)
        else:
            result["invalid_subfolders"].append(child_dir.name)

    if result["invalid_subfolders"]:
        result["messages"].append(
            "Some child folders do not contain TIFF files and will be skipped unless you reorganize the input directory."
        )

    if loose_tifs and not unprepared_loose_tifs:
        result["messages"].append(
            "Root-level TIFF files are still present, but matching prepared dataset subfolders were detected. The original flat TIFF files can be retained; the pipeline will run from the prepared subfolders."
        )

    if result["total_tif_count"] == 0:
        result["messages"].append("No tif/tiff files were found under input-dir.")
        result["messages"].append(server_path_note())

    result["can_run"] = (not unprepared_loose_tifs) and bool(result["valid_subfolders"])
    if result["can_run"]:
        result["messages"].append(
            "Input directory passed the required root-level structure check."
        )
    return result


def discover_reports(output_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "output_dir": str(output_dir),
        "exists": output_dir.exists(),
        "datasets": [],
        "messages": [],
    }
    if not output_dir.exists():
        data["messages"].append("Output directory does not exist yet.")
        return data
    if not output_dir.is_dir():
        data["messages"].append("Output path is not a directory.")
        return data

    for dataset_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        dataset_item = {
            "dataset_name": dataset_dir.name,
            "dataset_path": str(dataset_dir),
            "errors": [],
            "stacks": [],
        }
        logs_dir = dataset_dir / "logs"
        if logs_dir.is_dir():
            dataset_item["errors"] = [
                str(path.resolve())
                for path in sorted(logs_dir.glob("ERROR_*.log"))
            ]

        for stack_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if stack_dir.name in {"_shared", "logs", "results_deepcad", "results_demotion", "pth_deepcad"}:
                continue
            report_path = stack_dir / "report" / "report.html"
            if not report_path.is_file():
                continue
            manifest_path = stack_dir / "manifests" / "pipeline_manifest.json"
            final_stack_path = stack_dir / "final" / "final_stack.tif"
            manifest = None
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = None
            dataset_item["stacks"].append(
                {
                    "stack_name": stack_dir.name,
                    "stack_path": str(stack_dir),
                    "report_path": str(report_path.resolve()),
                    "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
                    "final_stack_path": str(final_stack_path.resolve()) if final_stack_path.is_file() else None,
                    "report_generated": bool(manifest and manifest.get("report")),
                }
            )
        if dataset_item["stacks"] or dataset_item["errors"]:
            data["datasets"].append(dataset_item)

    if not data["datasets"]:
        data["messages"].append("No report.html files were found under output-dir yet.")
    return data


def ensure_report_path(report_path: Path, output_dir: Path) -> Path:
    report_path = report_path.resolve(strict=True)
    output_dir = output_dir.resolve(strict=True)
    if report_path.name != "report.html":
        raise ValueError("Only report.html can be rendered in the embedded viewer.")
    if report_path.is_dir():
        raise ValueError("Report path must point to a file.")
    if not report_path.is_relative_to(output_dir):
        raise ValueError("Report path must stay under the selected output directory.")
    return report_path


def prepare_flat_input_dir(input_dir: Path) -> dict[str, Any]:
    scan = inspect_input_layout(input_dir)
    if not scan["exists"]:
        raise FileNotFoundError("Input directory does not exist.")
    if not scan["is_dir"]:
        raise NotADirectoryError("Input path must be a directory.")
    if scan["total_tif_count"] == 0:
        raise FileNotFoundError("No tif/tiff files were found under input-dir, so the prepare helper is not applicable.")
    if not scan["should_prepare_input"]:
        raise ValueError("No unprepared loose root-level TIFF files were found. The prepare helper is only shown for flat input layouts that still need preparation.")

    from prepare_input_tiffs import prepare_input_tiffs

    summary = prepare_input_tiffs(
        input_dir=input_dir,
        output_dir=None,
        recursive=False,
        overwrite=False,
        bigtiff=True,
    )
    post_scan = inspect_input_layout(input_dir)
    return {
        "summary": summary,
        "scan": post_scan,
    }


def resolve_conda_executable() -> str:
    candidates = [
        os.getenv("CONDA_EXE", "").strip(),
        shutil.which("conda"),
        shutil.which("conda.exe"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise RuntimeError(
        "Could not find a usable conda executable. Launch the UI from a shell where conda is available."
    )


def run_conda_python_check(conda_executable: str, env_name: str) -> str:
    command = [
        conda_executable,
        "run",
        "-n",
        env_name,
        "python",
        "-c",
        "import sys; print(sys.executable)",
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Conda environment '{env_name}' could not be used. {stderr or 'No error output was returned.'}"
        )
    return (completed.stdout or "").strip()


@dataclass
class JobConfig:
    input_dir: Path
    output_dir: Path
    main_env_name: str
    downstream_env_name: str | None
    gpu: str
    um_per_pixel: float
    frame_rate: float
    llm_mode: str
    openai_api_key: str | None
    dataset_type: str
    subfolders: list[str]

    def public_dict(self) -> dict[str, Any]:
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "main_env_name": self.main_env_name,
            "downstream_env_name": self.downstream_env_name,
            "gpu": self.gpu,
            "um_per_pixel": self.um_per_pixel,
            "frame_rate": self.frame_rate,
            "llm_mode": self.llm_mode,
            "dataset_type": self.dataset_type,
            "subfolders": list(self.subfolders),
            "api_key_present": bool(self.openai_api_key),
        }


def normalize_job_config(payload: dict[str, Any]) -> JobConfig:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    llm_mode = str(payload.get("llm_mode") or "off").strip().lower()
    if llm_mode not in {"off", "apply"}:
        raise ValueError("llm_mode must be either 'off' or 'apply'.")

    dataset_type = str(payload.get("dataset_type") or "cell-data").strip().lower()
    if dataset_type not in {"cell-data", "non-cell-data"}:
        raise ValueError("dataset_type must be 'cell-data' or 'non-cell-data'.")

    main_env_name = str(payload.get("main_env_name") or "").strip()
    if not main_env_name:
        raise ValueError("main_env_name is required.")

    downstream_env_name = str(payload.get("downstream_env_name") or "").strip() or None
    if dataset_type == "cell-data" and not downstream_env_name:
        raise ValueError("downstream_env_name is required when dataset_type is cell-data.")

    api_key = str(payload.get("openai_api_key") or "").strip() or None
    if llm_mode == "apply" and not api_key:
        raise ValueError("OPENAI_API_KEY is required when llm_mode is apply.")

    raw_subfolders = payload.get("subfolders") or []
    if not isinstance(raw_subfolders, list):
        raise ValueError("subfolders must be a JSON array of folder names.")
    subfolders = sorted({str(item).strip() for item in raw_subfolders if str(item).strip()})

    gpu = str(payload.get("gpu") or "0").strip() or "0"
    try:
        um_per_pixel = float(payload.get("um_per_pixel") or 0.645)
    except (TypeError, ValueError):
        raise ValueError("um_per_pixel must be a positive number.")
    if um_per_pixel <= 0:
        raise ValueError("um_per_pixel must be a positive number.")

    try:
        frame_rate = float(payload.get("frame_rate") or 10)
    except (TypeError, ValueError):
        raise ValueError("frame_rate must be a positive number.")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be a positive number.")

    return JobConfig(
        input_dir=resolve_user_path(payload.get("input_dir") or ""),
        output_dir=resolve_user_path(payload.get("output_dir") or ""),
        main_env_name=main_env_name,
        downstream_env_name=downstream_env_name,
        gpu=gpu,
        um_per_pixel=um_per_pixel,
        frame_rate=frame_rate,
        llm_mode=llm_mode,
        openai_api_key=api_key,
        dataset_type=dataset_type,
        subfolders=subfolders,
    )


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = "idle"
        self._message = "No job has been started yet."
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._config: JobConfig | None = None
        self._command: list[str] = []
        self._logs: deque[str] = deque(maxlen=5000)
        self._live_log_text: str | None = None
        self._live_log_entry: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._returncode: int | None = None
        self._worker: threading.Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            config = self._config.public_dict() if self._config else None
            return {
                "status": self._status,
                "message": self._message,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "command": list(self._command),
                "returncode": self._returncode,
                "pid": self._process.pid if self._process else None,
                "config": config,
                "logs": list(self._logs) + ([self._live_log_entry] if self._live_log_entry else []),
                "is_running": self._status in {"validating", "running"},
            }

    def start(self, config: JobConfig) -> dict[str, Any]:
        with self._lock:
            if self._status in {"validating", "running"}:
                raise RuntimeError("Another NeuroPilot job is already running in the local UI.")
            self._status = "validating"
            self._message = "Validating input directory, environments, and launch command."
            self._started_at = utc_now_iso()
            self._finished_at = None
            self._config = config
            self._command = []
            self._logs = deque(maxlen=5000)
            self._live_log_text = None
            self._live_log_entry = None
            self._returncode = None
            self._process = None
            self._append_log_locked("Job accepted by UI. Starting validation.")
            self._worker = threading.Thread(target=self._run_job, args=(config,), daemon=True)
            self._worker.start()
        return self.snapshot()

    def _append_log_locked(self, text: str) -> None:
        cleaned = strip_ansi(text).strip()
        if not cleaned:
            return
        if self._live_log_text is not None:
            if cleaned == self._live_log_text:
                self._live_log_text = None
                self._live_log_entry = None
            else:
                self._flush_live_log_locked()
        self._logs.append(f"[{utc_now_iso()}] {cleaned}")

    def _set_live_log_locked(self, text: str) -> None:
        cleaned = strip_ansi(text).strip()
        if not cleaned:
            return
        self._live_log_text = cleaned
        self._live_log_entry = f"[{utc_now_iso()}] {cleaned}"

    def _flush_live_log_locked(self) -> None:
        if not self._live_log_text or not self._live_log_entry:
            self._live_log_text = None
            self._live_log_entry = None
            return
        self._logs.append(self._live_log_entry)
        self._live_log_text = None
        self._live_log_entry = None

    def _append_log(self, text: str) -> None:
        with self._lock:
            self._append_log_locked(text)

    def _set_live_log(self, text: str) -> None:
        with self._lock:
            self._set_live_log_locked(text)

    def _mark_failed(self, message: str) -> None:
        with self._lock:
            self._flush_live_log_locked()
            self._status = "failed"
            self._message = message
            self._finished_at = utc_now_iso()
            self._returncode = self._returncode if self._returncode is not None else 1
            self._append_log_locked(message)
            self._process = None

    def _mark_success(self, message: str) -> None:
        with self._lock:
            self._flush_live_log_locked()
            self._status = "success"
            self._message = message
            self._finished_at = utc_now_iso()
            self._append_log_locked(message)
            self._process = None

    def _consume_process_output(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buffer = ""

        while True:
            raw_chunk = process.stdout.read(4096)
            if not raw_chunk:
                if process.poll() is not None:
                    break
                continue

            buffer += decoder.decode(raw_chunk)

            while True:
                newline_idx = buffer.find("\n")
                carriage_idx = buffer.find("\r")
                split_candidates = [idx for idx in (newline_idx, carriage_idx) if idx != -1]
                if not split_candidates:
                    break

                split_idx = min(split_candidates)
                separator = buffer[split_idx]
                segment = buffer[:split_idx]
                buffer = buffer[split_idx + 1:]

                if separator == "\r":
                    if segment.strip():
                        self._set_live_log(segment)
                    continue

                if segment.strip():
                    self._append_log(segment)
                else:
                    with self._lock:
                        self._flush_live_log_locked()

        buffer += decoder.decode(b"", final=True)
        trailing = strip_ansi(buffer).strip()
        with self._lock:
            if trailing:
                self._append_log_locked(trailing)
            self._flush_live_log_locked()

    def _terminate_process_if_running(self) -> None:
        with self._lock:
            process = self._process
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
        finally:
            with self._lock:
                self._process = None

    def _validate_input_and_expand_subfolders(self, config: JobConfig) -> JobConfig:
        scan = inspect_input_layout(config.input_dir)
        if not scan["exists"]:
            raise RuntimeError("Input directory does not exist.")
        if not scan["is_dir"]:
            raise RuntimeError("Input path must be a directory.")
        if scan["unprepared_loose_tifs"]:
            raise RuntimeError(
                "Unprepared loose TIFF files were found directly under input-dir. "
                f"Reorganize them into child folders first, for example by running: {scan['prepare_command']}"
            )
        valid_names = set(scan["valid_subfolders"])
        if not valid_names:
            raise RuntimeError("No valid dataset subfolders with TIFF files were found under input-dir.")

        requested = config.subfolders or scan["default_selected_subfolders"]
        missing = [name for name in requested if name not in valid_names]
        if missing:
            raise RuntimeError(
                f"Selected subfolders are not valid dataset folders with TIFF files: {', '.join(missing)}"
            )

        return JobConfig(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            main_env_name=config.main_env_name,
            downstream_env_name=config.downstream_env_name,
            gpu=config.gpu,
            um_per_pixel=config.um_per_pixel,
            frame_rate=config.frame_rate,
            llm_mode=config.llm_mode,
            openai_api_key=config.openai_api_key,
            dataset_type=config.dataset_type,
            subfolders=requested,
        )

    def _build_command(self, conda_executable: str, config: JobConfig) -> list[str]:
        command = [
            conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            config.main_env_name,
            "python",
            "-u",
            str(PIPELINE_ENTRY),
            "--input-dir",
            str(config.input_dir),
            "--output-dir",
            str(config.output_dir),
            "--subfolders",
            ",".join(config.subfolders),
            "--um-per-pixel",
            str(config.um_per_pixel),
            "--frame-rate",
            str(config.frame_rate),
            "--llm-mode",
            config.llm_mode,
            "--GPU",
            config.gpu,
        ]
        if config.dataset_type == "cell-data":
            command.extend(["--cell-data", "--downstream-env", str(config.downstream_env_name)])
        else:
            command.append("--non-cell-data")
        return command

    def _run_job(self, config: JobConfig) -> None:
        try:
            config = self._validate_input_and_expand_subfolders(config)
            self._append_log("Input directory passed validation.")
            config.output_dir.mkdir(parents=True, exist_ok=True)
            self._append_log(f"Output directory ready: {config.output_dir}")

            conda_executable = resolve_conda_executable()
            self._append_log(f"Using conda executable: {conda_executable}")

            main_python = run_conda_python_check(conda_executable, config.main_env_name)
            self._append_log(f"Main environment '{config.main_env_name}' resolved to: {main_python}")

            if config.dataset_type == "cell-data" and config.downstream_env_name:
                downstream_python = run_conda_python_check(conda_executable, config.downstream_env_name)
                self._append_log(
                    f"Downstream environment '{config.downstream_env_name}' resolved to: {downstream_python}"
                )

            command = self._build_command(conda_executable, config)
            env = os.environ.copy()
            if config.llm_mode == "apply" and config.openai_api_key:
                env["OPENAI_API_KEY"] = config.openai_api_key
            elif config.llm_mode == "off":
                env.pop("OPENAI_API_KEY", None)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")

            with self._lock:
                self._status = "running"
                self._message = "Pipeline is running."
                self._config = config
                self._command = command
                self._append_log_locked("Launching NeuroPilot pipeline subprocess.")

            process = subprocess.Popen(
                command,
                cwd=str(ROOT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            with self._lock:
                self._process = process

            self._consume_process_output(process)

            returncode = process.wait()
            with self._lock:
                self._returncode = returncode

            if returncode == 0:
                self._mark_success("Pipeline finished successfully. Reports can now be viewed in the UI.")
            else:
                self._mark_failed(f"Pipeline exited with return code {returncode}. Check the logs and output-dir artifacts.")

        except Exception as exc:
            self._terminate_process_if_running()
            self._mark_failed(f"UI launch failed: {exc}")


JOB_MANAGER = JobManager()


class NeuroPilotUIHandler(BaseHTTPRequestHandler):
    server_version = "NeuroPilotLocalUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_static_file(STATIC_DIR / "index.html", content_type="text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            relative = parsed.path.removeprefix("/static/")
            target = (STATIC_DIR / relative).resolve()
            if not target.is_file() or not target.is_relative_to(STATIC_DIR.resolve()):
                self._send_error_json(HTTPStatus.NOT_FOUND, "Static asset not found.")
                return
            content_type, _ = mimetypes.guess_type(str(target))
            self._serve_static_file(target, content_type=content_type or "application/octet-stream")
            return
        if parsed.path == "/api/job":
            self._send_json(HTTPStatus.OK, JOB_MANAGER.snapshot())
            return
        if parsed.path == "/api/server-info":
            self._send_json(HTTPStatus.OK, server_context(self.headers.get("Host", "")))
            return
        if parsed.path == "/api/reports":
            query = parse_qs(parsed.query)
            output_dir_raw = query.get("output_dir", [""])[0]
            try:
                output_dir = resolve_user_path(output_dir_raw)
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, discover_reports(output_dir))
            return
        if parsed.path == "/report":
            query = parse_qs(parsed.query)
            output_dir_raw = query.get("output_dir", [""])[0]
            report_path_raw = query.get("report_path", [""])[0]
            try:
                output_dir = resolve_user_path(output_dir_raw)
                report_path = ensure_report_path(resolve_user_path(report_path_raw), output_dir)
            except Exception as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._serve_static_file(report_path, content_type="text/html; charset=utf-8")
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Route not found.")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
            return

        if parsed.path == "/api/scan-input":
            try:
                input_dir = resolve_user_path(payload.get("input_dir") or "")
            except ValueError as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, inspect_input_layout(input_dir))
            return

        if parsed.path == "/api/prepare-input":
            if JOB_MANAGER.snapshot().get("is_running"):
                self._send_error_json(
                    HTTPStatus.CONFLICT,
                    "A pipeline job is currently running. Wait for it to finish before preparing the input directory.",
                )
                return
            try:
                input_dir = resolve_user_path(payload.get("input_dir") or "")
                result = prepare_flat_input_dir(input_dir)
            except Exception as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if parsed.path == "/api/start-job":
            try:
                config = normalize_job_config(payload)
                snapshot = JOB_MANAGER.start(config)
            except Exception as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.ACCEPTED, snapshot)
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, "Route not found.")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        length_text = self.headers.get("Content-Length", "").strip()
        if not length_text:
            return {}
        try:
            length = int(length_text)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header.") from exc
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid UTF-8 JSON.") from exc

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"error": message})

    def _serve_static_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self._send_error_json(HTTPStatus.NOT_FOUND, "Requested file was not found.")
            return
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the local NeuroPilot UI for configuring and monitoring pipeline runs."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8008, help="Port to bind. Defaults to 8008.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local UI in your default browser after the server starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), NeuroPilotUIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[LOCAL_UI] Serving NeuroPilot UI at {url}")
    print(f"[LOCAL_UI] Repository root: {ROOT_DIR}")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[LOCAL_UI] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
