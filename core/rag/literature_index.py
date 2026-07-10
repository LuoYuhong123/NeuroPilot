#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MANIFEST_SCHEMA_VERSION = "neuropilot_rag.literature_manifest.v1"
CHUNK_SCHEMA_VERSION = "neuropilot_rag.literature_chunk.v1"
CHUNK_SUMMARY_SCHEMA_VERSION = "neuropilot_rag.literature_chunk_summary.v1"
RETRIEVAL_SCHEMA_VERSION = "neuropilot_rag.literature_retrieval.v1"
DEFAULT_CHUNK_WORDS = 900
DEFAULT_CHUNK_OVERLAP_WORDS = 120
DEFAULT_LITERATURE_TOP_K = 8
DEFAULT_MAX_CHUNKS_PER_PAPER = 2
MIN_CHUNK_CHARS = 160
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*|\d+(?:\.\d+)?")


def _print_json(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        print(text)
    except UnicodeEncodeError:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "using",
    "with",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    path_obj = Path(path)
    if not path_obj.exists():
        return []
    records: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                records.append(data)
    return records


def _write_json(path: str | Path, payload: dict[str, Any]) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(path_obj)


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path_obj)


def _safe_resolve(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


def _slug(value: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug[:max_len].rstrip("_") or "document").lower()


def _title_from_stem(stem: str) -> str:
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    return re.sub(r"[_\-]+", " ", stem).strip().title()


def _year_from_text(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False


def _path_key(path: str | Path | None) -> str | None:
    resolved = _safe_resolve(path)
    return resolved.lower() if resolved else None


def _topic_tags_for_path(path: Path, existing: dict[str, Any] | None = None) -> list[str]:
    tags: list[str] = []
    if existing:
        tags.extend(str(tag) for tag in existing.get("topic_tags", []) if str(tag).strip())
    rel_text = f"{path.parent.name} {path.name}".lower()
    inferred = {
        "Neuropilot": "neuropilot" in rel_text,
        "workflow": "neuropilot" in rel_text or "pipeline" in rel_text,
        "supplementary": "supp" in rel_text,
        "calcium imaging": "calcium" in rel_text or "neuro" in rel_text,
        "miniscope": "miniscope" in rel_text or "microendoscopy" in rel_text,
        "motion correction": "motion" in rel_text or "normcorre" in rel_text,
        "segmentation": "cellpose" in rel_text or "segmentation" in rel_text,
        "DeepCAD": "deepcad" in rel_text,
    }
    for tag, enabled in inferred.items():
        if enabled:
            tags.append(tag)
    out: list[str] = []
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return out


def _infer_neuropilot_record(pdf_path: Path) -> dict[str, Any]:
    stem = pdf_path.stem
    is_supp = "supp" in stem.lower()
    return {
        "paper_id": _slug(stem, 64),
        "title": "NeuroPilot supplementary materials" if is_supp else "NeuroPilot main manuscript",
        "authors": None,
        "year": _year_from_text(stem),
        "venue": "manuscript",
        "doi": None,
        "source_group": "Neuropilot",
        "topic_tags": _topic_tags_for_path(pdf_path),
        "document_type": "supplementary_materials" if is_supp else "main_manuscript",
        "is_core_neuropilot": True,
    }


def _infer_pdf_record(pdf_path: Path, raw_root: Path, existing: dict[str, Any] | None) -> dict[str, Any]:
    if existing:
        record = dict(existing)
    elif "neuropilot" in str(pdf_path).lower():
        record = _infer_neuropilot_record(pdf_path)
    else:
        try:
            source_group = pdf_path.relative_to(raw_root).parts[0]
        except Exception:
            source_group = pdf_path.parent.name
        record = {
            "paper_id": _slug(pdf_path.stem, 64),
            "title": _title_from_stem(pdf_path.stem),
            "authors": None,
            "year": _year_from_text(pdf_path.stem),
            "venue": None,
            "doi": None,
            "source_group": source_group,
            "topic_tags": _topic_tags_for_path(pdf_path),
            "document_type": "article",
            "is_core_neuropilot": False,
        }

    source_group = record.get("source_group")
    if not source_group:
        try:
            source_group = pdf_path.relative_to(raw_root).parts[0]
        except Exception:
            source_group = pdf_path.parent.name
    local_text = f"{source_group} {pdf_path.name}".lower()
    is_core = bool(record.get("is_core_neuropilot")) or "neuropilot" in local_text
    document_type = record.get("document_type")
    if not document_type:
        document_type = "supplementary_materials" if "supp" in pdf_path.stem.lower() else "article"
        if is_core and document_type == "article":
            document_type = "main_manuscript"

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "paper_id": record.get("paper_id") or _slug(pdf_path.stem, 64),
        "title": record.get("title") or _title_from_stem(pdf_path.stem),
        "authors": record.get("authors"),
        "year": record.get("year") or _year_from_text(pdf_path.stem),
        "venue": record.get("venue"),
        "doi": record.get("doi"),
        "source_group": source_group,
        "topic_tags": _topic_tags_for_path(pdf_path, record),
        "document_type": document_type,
        "is_core_neuropilot": is_core,
        "status": "downloaded" if _is_pdf(pdf_path) else "invalid_pdf",
        "pdf_path": str(pdf_path.resolve()),
        "relative_path": None,
        "bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "sha256": _sha256_file(pdf_path) if pdf_path.exists() else None,
        "urls": record.get("urls", []),
        "landing_page": record.get("landing_page"),
        "download_url": record.get("download_url"),
        "metadata_note": record.get("metadata_note"),
        "open_version_note": record.get("open_version_note"),
    }


def build_literature_manifest(raw_root: str | Path, metadata_jsonl: str | Path | None = None) -> dict[str, Any]:
    raw_root_path = Path(raw_root).expanduser().resolve()
    metadata_records = _read_jsonl(metadata_jsonl)
    by_path: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for record in metadata_records:
        if record.get("paper_id"):
            by_id[str(record["paper_id"])] = record
        key = _path_key(record.get("pdf_path"))
        if key:
            by_path[key] = record

    records_by_id: dict[str, dict[str, Any]] = {}
    pdf_paths = sorted(path for path in raw_root_path.rglob("*.pdf") if path.is_file())
    for pdf_path in pdf_paths:
        key = _path_key(pdf_path)
        existing = by_path.get(key or "")
        record = _infer_pdf_record(pdf_path, raw_root_path, existing)
        try:
            record["relative_path"] = str(pdf_path.relative_to(raw_root_path.parent.resolve()))
        except Exception:
            record["relative_path"] = str(pdf_path)
        records_by_id[str(record["paper_id"])] = record

    for old in metadata_records:
        paper_id = old.get("paper_id")
        if not paper_id or paper_id in records_by_id:
            continue
        record = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "paper_id": paper_id,
            "title": old.get("title"),
            "authors": old.get("authors"),
            "year": old.get("year"),
            "venue": old.get("venue"),
            "doi": old.get("doi"),
            "source_group": old.get("source_group") or "metadata_only",
            "topic_tags": old.get("topic_tags", []),
            "document_type": old.get("document_type") or "metadata_only",
            "is_core_neuropilot": bool(old.get("is_core_neuropilot")),
            "status": old.get("download_status") or "metadata_only",
            "pdf_path": None,
            "relative_path": None,
            "bytes": 0,
            "sha256": None,
            "urls": old.get("urls", []),
            "landing_page": old.get("landing_page"),
            "download_url": old.get("download_url"),
            "metadata_note": old.get("metadata_note"),
            "open_version_note": old.get("open_version_note"),
        }
        records_by_id[str(paper_id)] = record

    records = sorted(
        records_by_id.values(),
        key=lambda item: (
            0 if item.get("is_core_neuropilot") else 1,
            str(item.get("source_group") or ""),
            int(item.get("year") or 0),
            str(item.get("paper_id") or ""),
        ),
    )
    status_counts = Counter(str(record.get("status") or "unknown") for record in records)
    group_counts = Counter(str(record.get("source_group") or "unknown") for record in records)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "raw_root": str(raw_root_path),
        "metadata_jsonl": _safe_resolve(metadata_jsonl),
        "records": records,
        "summary": {
            "record_count": len(records),
            "pdf_count": sum(1 for record in records if record.get("pdf_path")),
            "core_neuropilot_count": sum(1 for record in records if record.get("is_core_neuropilot")),
            "status_counts": dict(sorted(status_counts.items())),
            "source_group_counts": dict(sorted(group_counts.items())),
        },
    }


def _clean_text(text: str) -> str:
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_token(token: str) -> str:
    token = token.lower().strip("_-+")
    token = token.replace("-", "_")
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    return token


def _tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text or ""):
        token = _normalize_token(raw)
        if len(token) < 2 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _flatten_context_for_query(value: Any, prefix: str = "") -> list[str]:
    parts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text:
                parts.append(key_text)
            parts.extend(_flatten_context_for_query(item, key_text))
    elif isinstance(value, list):
        for item in value[:50]:
            parts.extend(_flatten_context_for_query(item, prefix))
    elif value is not None and not isinstance(value, bool):
        parts.append(str(value))
    return parts


def build_literature_query(context_or_text: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(context_or_text, str):
        query_text = context_or_text.strip()
        source = "text"
    else:
        parts = _flatten_context_for_query(context_or_text)
        query_text = " ".join(part for part in parts if part).strip()
        source = "context"
    terms = sorted(set(_tokenize_text(query_text)))
    return {
        "source": source,
        "query_text": query_text,
        "terms": terms,
        "term_count": len(terms),
    }


def _extract_with_pypdf(pdf_path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        from pypdf import PdfReader  # type: ignore

        backend = "pypdf"
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore

            backend = "PyPDF2"
        except Exception as exc:
            raise RuntimeError("pypdf/PyPDF2 is not available") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": index, "text": _clean_text(text)})
    return pages, backend


def _extract_with_pdftotext(pdf_path: Path) -> tuple[list[dict[str, Any]], str]:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is not available")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "document.txt"
        subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(out_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        text = out_path.read_text(encoding="utf-8", errors="replace")
    pages = [
        {"page_number": index, "text": _clean_text(page_text)}
        for index, page_text in enumerate(text.split("\f"), start=1)
        if _clean_text(page_text)
    ]
    return pages, "pdftotext"


def extract_pdf_pages(pdf_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    path = Path(pdf_path)
    errors: list[str] = []
    for extractor in (_extract_with_pypdf, _extract_with_pdftotext):
        try:
            return extractor(path)
        except Exception as exc:
            errors.append(f"{extractor.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _word_windows(words: list[str], chunk_words: int, overlap_words: int) -> Iterable[tuple[int, int, list[str]]]:
    if len(words) <= chunk_words:
        yield 0, len(words), words
        return
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + chunk_words)
        yield start, end, words[start:end]
        if end >= len(words):
            break
        start += step


def _make_chunks_for_page(
    record: dict[str, Any],
    page: dict[str, Any],
    *,
    chunk_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    page_number = int(page.get("page_number") or 0)
    text = _clean_text(str(page.get("text") or ""))
    if len(text) < MIN_CHUNK_CHARS:
        return []
    words = text.split()
    chunks: list[dict[str, Any]] = []
    for chunk_index, (word_start, word_end, window_words) in enumerate(
        _word_windows(words, chunk_words=chunk_words, overlap_words=overlap_words)
    ):
        chunk_text = " ".join(window_words).strip()
        if len(chunk_text) < MIN_CHUNK_CHARS:
            continue
        chunk_id = f"{record['paper_id']}:p{page_number:04d}:c{chunk_index:03d}"
        chunks.append(
            {
                "schema_version": CHUNK_SCHEMA_VERSION,
                "chunk_id": chunk_id,
                "paper_id": record.get("paper_id"),
                "title": record.get("title"),
                "source_group": record.get("source_group"),
                "document_type": record.get("document_type"),
                "is_core_neuropilot": bool(record.get("is_core_neuropilot")),
                "topic_tags": record.get("topic_tags", []),
                "pdf_path": record.get("pdf_path"),
                "page_start": page_number,
                "page_end": page_number,
                "chunk_index_on_page": chunk_index,
                "word_start_on_page": word_start,
                "word_end_on_page": word_end,
                "word_count": len(window_words),
                "char_count": len(chunk_text),
                "citation": f"{record.get('title') or record.get('paper_id')}, p. {page_number}",
                "text": chunk_text,
            }
        )
    return chunks


def build_literature_chunks(
    manifest_records: Iterable[dict[str, Any]],
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    backend_counts: Counter[str] = Counter()
    for record in manifest_records:
        pdf_path = record.get("pdf_path")
        if not pdf_path:
            documents.append(
                {
                    "paper_id": record.get("paper_id"),
                    "title": record.get("title"),
                    "status": "skipped_no_pdf",
                    "chunk_count": 0,
                }
            )
            continue
        path = Path(str(pdf_path))
        try:
            pages, backend = extract_pdf_pages(path)
            backend_counts[backend] += 1
            doc_chunks: list[dict[str, Any]] = []
            for page in pages:
                doc_chunks.extend(
                    _make_chunks_for_page(
                        record,
                        page,
                        chunk_words=chunk_words,
                        overlap_words=overlap_words,
                    )
                )
            chunks.extend(doc_chunks)
            documents.append(
                {
                    "paper_id": record.get("paper_id"),
                    "title": record.get("title"),
                    "status": "extracted" if doc_chunks else "empty_text",
                    "backend": backend,
                    "page_count": len(pages),
                    "chunk_count": len(doc_chunks),
                    "word_count": sum(int(chunk.get("word_count") or 0) for chunk in doc_chunks),
                    "is_core_neuropilot": bool(record.get("is_core_neuropilot")),
                    "pdf_path": str(path),
                }
            )
        except Exception as exc:
            documents.append(
                {
                    "paper_id": record.get("paper_id"),
                    "title": record.get("title"),
                    "status": "extract_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "chunk_count": 0,
                    "is_core_neuropilot": bool(record.get("is_core_neuropilot")),
                    "pdf_path": str(path),
                }
            )
    status_counts = Counter(str(item.get("status")) for item in documents)
    return {
        "schema_version": CHUNK_SUMMARY_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "chunking": {
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "min_chunk_chars": MIN_CHUNK_CHARS,
        },
        "chunks": chunks,
        "summary": {
            "document_count": len(documents),
            "extracted_document_count": sum(1 for item in documents if item.get("status") == "extracted"),
            "chunk_count": len(chunks),
            "core_neuropilot_chunk_count": sum(1 for chunk in chunks if chunk.get("is_core_neuropilot")),
            "status_counts": dict(sorted(status_counts.items())),
            "backend_counts": dict(sorted(backend_counts.items())),
            "total_words": sum(int(chunk.get("word_count") or 0) for chunk in chunks),
        },
        "documents": documents,
    }


def _chunk_index_text(chunk: dict[str, Any]) -> str:
    topic_tags = " ".join(str(tag) for tag in chunk.get("topic_tags", []) if str(tag).strip())
    title = str(chunk.get("title") or "")
    source_group = str(chunk.get("source_group") or "")
    document_type = str(chunk.get("document_type") or "")
    text = str(chunk.get("text") or "")
    # Repeat metadata fields so short method names and document tags are retrievable even
    # when the exact phrase is sparse in a page chunk.
    return " ".join(
        [
            text,
            title,
            title,
            title,
            topic_tags,
            topic_tags,
            topic_tags,
            source_group,
            source_group,
            document_type,
        ]
    )


def _load_chunks(chunks_path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(chunks_path)


def _bm25_rank_chunks(
    chunks: list[dict[str, Any]],
    query_terms: list[str],
    *,
    core_neuropilot_boost: float = 1.08,
) -> list[dict[str, Any]]:
    if not chunks or not query_terms:
        return []
    query_term_set = set(query_terms)
    tokenized_docs: list[list[str]] = []
    term_frequencies: list[Counter[str]] = []
    document_frequencies: Counter[str] = Counter()
    lengths: list[int] = []

    for chunk in chunks:
        tokens = _tokenize_text(_chunk_index_text(chunk))
        counts = Counter(tokens)
        tokenized_docs.append(tokens)
        term_frequencies.append(counts)
        lengths.append(len(tokens))
        for term in query_term_set:
            if counts.get(term, 0) > 0:
                document_frequencies[term] += 1

    n_docs = len(chunks)
    avg_len = sum(lengths) / n_docs if n_docs else 1.0
    k1 = 1.4
    b = 0.72
    idf = {
        term: math.log(1.0 + (n_docs - document_frequencies.get(term, 0) + 0.5) / (document_frequencies.get(term, 0) + 0.5))
        for term in query_term_set
    }

    ranked: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        counts = term_frequencies[index]
        doc_len = max(1, lengths[index])
        matched_terms = sorted(term for term in query_term_set if counts.get(term, 0) > 0)
        if not matched_terms:
            continue
        bm25 = 0.0
        for term in matched_terms:
            tf = counts[term]
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_len))
            bm25 += idf[term] * ((tf * (k1 + 1.0)) / denom)
        title_tokens = set(_tokenize_text(str(chunk.get("title") or "")))
        tag_tokens = set(_tokenize_text(" ".join(str(tag) for tag in chunk.get("topic_tags", []))))
        title_hits = len(query_term_set & title_tokens)
        tag_hits = len(query_term_set & tag_tokens)
        metadata_boost = min(0.35, title_hits * 0.08 + tag_hits * 0.06)
        core_boost = core_neuropilot_boost if chunk.get("is_core_neuropilot") else 1.0
        score = (bm25 * (1.0 + metadata_boost)) * core_boost
        ranked.append(
            {
                "chunk": chunk,
                "score": score,
                "score_components": {
                    "bm25": bm25,
                    "metadata_boost": metadata_boost,
                    "core_neuropilot_boost": core_boost,
                    "matched_term_count": len(matched_terms),
                },
                "matched_terms": matched_terms,
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _make_snippet(text: str, matched_terms: list[str], *, max_chars: int = 520) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= max_chars:
        return clean
    lower = clean.lower()
    positions = [lower.find(term.replace("_", " ")) for term in matched_terms]
    positions += [lower.find(term) for term in matched_terms]
    positions = [pos for pos in positions if pos >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(clean), start + max_chars)
    start = max(0, end - max_chars)
    snippet = clean[start:end].strip()
    if start > 0:
        snippet = "... " + snippet
    if end < len(clean):
        snippet = snippet + " ..."
    return snippet


def _compact_literature_match(
    ranked_item: dict[str, Any],
    *,
    max_snippet_chars: int = 520,
    max_matched_terms: int = 12,
) -> dict[str, Any]:
    chunk = ranked_item["chunk"]
    matched_terms = ranked_item["matched_terms"][:max_matched_terms]
    return {
        "chunk_id": chunk.get("chunk_id"),
        "paper_id": chunk.get("paper_id"),
        "title": chunk.get("title"),
        "source_group": chunk.get("source_group"),
        "document_type": chunk.get("document_type"),
        "is_core_neuropilot": bool(chunk.get("is_core_neuropilot")),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "citation": chunk.get("citation"),
        "pdf_path": chunk.get("pdf_path"),
        "score": round(float(ranked_item["score"]), 6),
        "score_components": ranked_item["score_components"],
        "matched_terms": matched_terms,
        "snippet": _make_snippet(str(chunk.get("text") or ""), matched_terms, max_chars=max_snippet_chars),
    }


def retrieve_literature(
    context_or_text: dict[str, Any] | str,
    *,
    chunks: list[dict[str, Any]] | None = None,
    chunks_path: str | Path = "literature/index/literature_chunks.jsonl",
    top_k: int = DEFAULT_LITERATURE_TOP_K,
    max_chunks_per_paper: int = DEFAULT_MAX_CHUNKS_PER_PAPER,
) -> dict[str, Any]:
    if chunks is None:
        chunks = _load_chunks(chunks_path)
    query = build_literature_query(context_or_text)
    ranked = _bm25_rank_chunks(chunks, query["terms"])
    per_paper: Counter[str] = Counter()
    matches: list[dict[str, Any]] = []
    for item in ranked:
        chunk = item["chunk"]
        paper_id = str(chunk.get("paper_id") or "")
        if max_chunks_per_paper > 0 and per_paper[paper_id] >= max_chunks_per_paper:
            continue
        matches.append(_compact_literature_match(item))
        per_paper[paper_id] += 1
        if len(matches) >= top_k:
            break
    max_score = max((match["score"] for match in matches), default=0.0)
    if max_score > 0:
        for match in matches:
            match["normalized_score"] = round(match["score"] / max_score, 6)
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "retrieval_mode": "local_literature_bm25",
        "query": {
            **query,
            "top_k": top_k,
            "max_chunks_per_paper": max_chunks_per_paper,
        },
        "sources": {
            "chunks_path": _safe_resolve(chunks_path),
            "chunk_count": len(chunks),
        },
        "candidate_count": len(ranked),
        "returned_count": len(matches),
        "matched_literature": matches,
        "notes": [
            "M3.3 uses local BM25-style lexical retrieval over M3.2 literature chunks.",
            "Scores combine BM25 term relevance with small title/tag and NeuroPilot-core boosts; they are for ranking, not statistical confidence.",
        ],
    }


def validate_outputs(
    manifest: dict[str, Any],
    chunk_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = manifest.get("records", [])
    records = records if isinstance(records, list) else []
    pdf_records = [record for record in records if record.get("pdf_path")]
    core_records = [record for record in records if record.get("is_core_neuropilot")]
    invalid_pdf_records = [
        record.get("paper_id")
        for record in pdf_records
        if not record.get("pdf_path") or not _is_pdf(Path(str(record.get("pdf_path"))))
    ]
    validation: dict[str, Any] = {
        "schema_version": "neuropilot_rag.literature_validation.v1",
        "generated_at_utc": _utc_now_iso(),
        "manifest": {
            "record_count": len(records),
            "pdf_record_count": len(pdf_records),
            "core_neuropilot_record_count": len(core_records),
            "invalid_pdf_record_ids": invalid_pdf_records,
            "has_neuropilot_main": any(
                record.get("is_core_neuropilot") and record.get("document_type") == "main_manuscript"
                for record in records
            ),
            "has_neuropilot_supplement": any(
                record.get("is_core_neuropilot") and record.get("document_type") == "supplementary_materials"
                for record in records
            ),
        },
    }
    if chunk_result is not None:
        documents = chunk_result.get("documents", [])
        documents = documents if isinstance(documents, list) else []
        by_id = {str(item.get("paper_id")): item for item in documents}
        extracted_ids = {str(item.get("paper_id")) for item in documents if item.get("status") == "extracted"}
        validation["chunks"] = {
            "document_count": len(documents),
            "chunk_count": len(chunk_result.get("chunks", [])),
            "extracted_document_count": sum(1 for item in documents if item.get("status") == "extracted"),
            "extract_failed_ids": [
                item.get("paper_id") for item in documents if item.get("status") == "extract_failed"
            ],
            "downloaded_pdf_without_chunks": [
                record.get("paper_id") for record in pdf_records if str(record.get("paper_id")) not in extracted_ids
            ],
            "core_neuropilot_chunks_by_document": {
                str(record.get("paper_id")): int(by_id.get(str(record.get("paper_id")), {}).get("chunk_count") or 0)
                for record in core_records
            },
        }
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the local NeuroPilot literature corpus.")
    parser.add_argument("--raw-root", default="literature/raw", help="Folder containing literature PDFs.")
    parser.add_argument("--metadata-jsonl", default="literature/manifest/papers.jsonl", help="Existing paper metadata JSONL.")
    parser.add_argument("--manifest-output", default="literature/manifest/literature_manifest.jsonl")
    parser.add_argument("--manifest-summary-output", default="literature/manifest/literature_manifest_summary.json")
    parser.add_argument("--chunks-input", default=None, help="Existing literature chunks JSONL to query. Defaults to --chunks-output.")
    parser.add_argument("--chunks-output", default="literature/index/literature_chunks.jsonl")
    parser.add_argument("--chunks-summary-output", default="literature/index/literature_chunks_summary.json")
    parser.add_argument("--validation-output", default="literature/index/literature_validation.json")
    parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    parser.add_argument("--chunk-overlap-words", type=int, default=DEFAULT_CHUNK_OVERLAP_WORDS)
    parser.add_argument("--query", default=None, help="Run a literature retrieval query after building or against existing chunks.")
    parser.add_argument("--query-context-json", default=None, help="JSON file containing an advisor context to convert into a query.")
    parser.add_argument("--retrieval-output", default=None, help="Write retrieval result JSON here.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_LITERATURE_TOP_K)
    parser.add_argument("--max-chunks-per-paper", type=int, default=DEFAULT_MAX_CHUNKS_PER_PAPER)
    parser.add_argument("--skip-build", action="store_true", help="Do not rebuild manifest/chunks; only run retrieval when a query is provided.")
    parser.add_argument("--skip-chunks", action="store_true", help="Only build the literature manifest.")
    args = parser.parse_args()

    query_payload: dict[str, Any] | str | None = None
    if args.query_context_json:
        with Path(args.query_context_json).open("r", encoding="utf-8") as f:
            loaded_context = json.load(f)
        query_payload = loaded_context if isinstance(loaded_context, dict) else str(loaded_context)
    elif args.query:
        query_payload = args.query

    manifest: dict[str, Any] | None = None
    chunk_result = None
    result_payload: dict[str, Any] = {}
    if not args.skip_build:
        manifest = build_literature_manifest(args.raw_root, metadata_jsonl=args.metadata_jsonl)
        _write_jsonl(args.manifest_output, manifest["records"])
        manifest_summary = {key: value for key, value in manifest.items() if key != "records"}
        _write_json(args.manifest_summary_output, manifest_summary)

        if not args.skip_chunks:
            chunk_result = build_literature_chunks(
                manifest["records"],
                chunk_words=max(100, args.chunk_words),
                overlap_words=max(0, min(args.chunk_overlap_words, args.chunk_words - 1)),
            )
            _write_jsonl(args.chunks_output, chunk_result["chunks"])
            chunk_summary = {key: value for key, value in chunk_result.items() if key != "chunks"}
            _write_json(args.chunks_summary_output, chunk_summary)

        validation = validate_outputs(manifest, chunk_result)
        _write_json(args.validation_output, validation)
        result_payload = {"manifest": manifest["summary"], "validation": validation}

    if query_payload is not None:
        chunks_path = args.chunks_input or args.chunks_output
        chunks = chunk_result["chunks"] if chunk_result and "chunks" in chunk_result else None
        retrieval = retrieve_literature(
            query_payload,
            chunks=chunks,
            chunks_path=chunks_path,
            top_k=max(1, args.top_k),
            max_chunks_per_paper=max(1, args.max_chunks_per_paper),
        )
        if args.retrieval_output:
            _write_json(args.retrieval_output, retrieval)
        if result_payload:
            result_payload["retrieval"] = retrieval
        else:
            result_payload = retrieval

    _print_json(result_payload)


if __name__ == "__main__":
    main()
