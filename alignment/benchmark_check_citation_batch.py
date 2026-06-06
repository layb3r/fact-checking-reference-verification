"""
Benchmark runner for batch citation verification from JSON instances.

Expected input: a JSON file containing a list of objects like:
{
  "claim_text": "... [CITATION] ...",
  "surrounding_context": "...",
  "citation_metadata": {
    "title": "...",
    "authors": ["..."],
    "venue": "...",
    "year": 2023,
    "identifiers": {
      "doi": null,
      "arxiv_id": null,
      "url": null
    }
  },
  "true_outputs": {
        "true_alignment": 0,
    ...
  }
}

Workflow:
1) Resolve paper text from arXiv first (prefer explicit arXiv ID, otherwise title search via arXiv API)
2) Fall back to Semantic Scholar title search when arXiv does not resolve
3) Download PDF and extract text
4) Run check_reference_batch over resolved claim/reference pairs
5) Compute classification metrics (precision, recall, F1, accuracy)

Examples:
  python alignment/benchmark_check_citation_batch.py --input data/my_benchmark.json
  python alignment/benchmark_check_citation_batch.py --input data/my_benchmark.json --max-instances 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import dotenv
import pymupdf
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIGNMENT_DIR = Path(__file__).resolve().parent

for path in (ALIGNMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

dotenv.load_dotenv(PROJECT_ROOT / ".env")

from claimcheck import check_reference_batch


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_ABS_PREFIX = "https://arxiv.org/abs/"
ARXIV_PDF_PREFIX = "https://arxiv.org/pdf/"
ARXIV_USER_AGENT = "fact-checking-reference-verification/1.0"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "title,authors,year,venue,externalIds,abstract,openAccessPdf,url"
SEMANTIC_SCHOLAR_USER_AGENT = "fact-checking-reference-verification/1.0"

LABEL_TO_INDEX = {
    "SUPPORTED": 0,
    "REFUTED": 1,
    "NEI": 2,
}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

_ARXIV_TITLE_CACHE: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
_SEMANTIC_SCHOLAR_TITLE_CACHE: Dict[Tuple[str, int], List[Dict[str, str]]] = {}


def json_default(value: Any) -> Any:
    """Convert non-standard scalar/container values to JSON-serializable types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)

    # Handles numpy scalar types such as np.float32/np.int64 without importing numpy.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_title(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_arxiv_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None

    value = value.strip()
    # Handles formats like:
    # - 2301.01234
    # - arXiv:2301.01234v2
    # - https://arxiv.org/abs/2301.01234v2
    # - https://arxiv.org/pdf/2301.01234v2.pdf
    match = re.search(r"([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", value)
    if match:
        return match.group(1)

    return None


def arxiv_id_to_pdf_url(arxiv_id: str) -> str:
    clean = arxiv_id.strip()
    if clean.endswith(".pdf"):
        clean = clean[:-4]
    return f"{ARXIV_PDF_PREFIX}{clean}.pdf"


def extract_arxiv_id_from_identifiers(citation_metadata: Dict[str, Any]) -> Optional[str]:
    identifiers = (citation_metadata or {}).get("identifiers") or {}

    candidate_fields: Sequence[Optional[str]] = (
        identifiers.get("arxiv_id"),
        identifiers.get("url"),
        citation_metadata.get("url"),
    )

    for value in candidate_fields:
        arxiv_id = parse_arxiv_id(value)
        if arxiv_id:
            return arxiv_id

    return None


def query_arxiv_by_title(title: str, max_results: int = 5, timeout: int = 20) -> List[Dict[str, str]]:
    title = (title or "").strip()
    if not title:
        return []

    cache_key = (normalize_title(title), max_results)
    if cache_key in _ARXIV_TITLE_CACHE:
        return list(_ARXIV_TITLE_CACHE[cache_key])

    query = f'ti:"{title}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }

    headers = {"User-Agent": ARXIV_USER_AGENT}
    response = None
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = requests.get(ARXIV_API_URL, params=params, timeout=timeout, headers=headers)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(30, 2 ** attempt)
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(min(30, 2 ** attempt))

    if response is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to query arXiv API")

    root = ET.fromstring(response.text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entries: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        entry_title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        entry_id_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()

        arxiv_id = parse_arxiv_id(entry_id_url)
        pdf_url = ""
        for link in entry.findall("atom:link", ns):
            title_attr = link.attrib.get("title", "")
            href = link.attrib.get("href", "")
            if title_attr == "pdf" and href:
                pdf_url = href
                break

        if arxiv_id and not pdf_url:
            pdf_url = arxiv_id_to_pdf_url(arxiv_id)

        entries.append(
            {
                "title": entry_title,
                "id_url": entry_id_url,
                "arxiv_id": arxiv_id or "",
                "pdf_url": pdf_url,
                "summary": summary,
            }
        )

    _ARXIV_TITLE_CACHE[cache_key] = list(entries)
    return entries


def query_semantic_scholar_by_title(title: str, max_results: int = 5, timeout: int = 20) -> List[Dict[str, str]]:
    title = (title or "").strip()
    if not title:
        return []

    cache_key = (normalize_title(title), max_results)
    if cache_key in _SEMANTIC_SCHOLAR_TITLE_CACHE:
        return list(_SEMANTIC_SCHOLAR_TITLE_CACHE[cache_key])

    params = {
        "query": title,
        "fields": SEMANTIC_SCHOLAR_FIELDS,
        "limit": min(max_results, 100),
    }

    headers = {"User-Agent": SEMANTIC_SCHOLAR_USER_AGENT}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    response = None
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            response = requests.get(SEMANTIC_SCHOLAR_API_URL, params=params, timeout=timeout, headers=headers)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(30, 2 ** attempt)
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 3:
                raise
            time.sleep(min(30, 2 ** attempt))

    if response is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to query Semantic Scholar API")

    entries: List[Dict[str, str]] = []
    for item in response.json().get("data", []):
        ext_ids = item.get("externalIds") or {}
        open_access_pdf = item.get("openAccessPdf") or {}
        arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("Arxiv") or ""
        pdf_url = open_access_pdf.get("url") or ""
        if arxiv_id and not pdf_url:
            pdf_url = arxiv_id_to_pdf_url(arxiv_id)

        entries.append(
            {
                "title": (item.get("title") or "").strip(),
                "paper_id": str(item.get("paperId") or ""),
                "arxiv_id": str(arxiv_id or ""),
                "pdf_url": str(pdf_url or ""),
                "url": str(item.get("url") or ""),
                "summary": (item.get("abstract") or "").strip(),
            }
        )

    _SEMANTIC_SCHOLAR_TITLE_CACHE[cache_key] = list(entries)
    return entries


def pick_best_arxiv_match(target_title: str, candidates: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not candidates:
        return None

    target_norm = normalize_title(target_title)
    best_score = -1
    best_item: Optional[Dict[str, str]] = None

    for item in candidates:
        cand_norm = normalize_title(item.get("title", ""))
        if not cand_norm:
            continue

        # Token-overlap score for a lightweight robust title match.
        target_tokens = set(target_norm.split())
        cand_tokens = set(cand_norm.split())
        if not target_tokens or not cand_tokens:
            score = 0
        else:
            overlap = len(target_tokens & cand_tokens)
            union = len(target_tokens | cand_tokens)
            score = int((overlap / union) * 1000)

        # Strong boost for near exact normalized title.
        if target_norm == cand_norm:
            score += 1000

        if score > best_score:
            best_score = score
            best_item = item

    return best_item


def pick_best_semantic_scholar_match(target_title: str, candidates: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    return pick_best_arxiv_match(target_title=target_title, candidates=candidates)


def download_file(url: str, output_path: Path, timeout: int = 45) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)


def extract_pdf_text(pdf_path: Path) -> str:
    text_chunks: List[str] = []
    document = pymupdf.open(pdf_path)
    try:
        for page in document:
            text_chunks.append(page.get_text())
    finally:
        document.close()

    return "\n".join(text_chunks).strip()


def safe_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "paper"
    return value[:max_len]


def normalize_encoded_label(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value) if int(value) in INDEX_TO_LABEL else None
    if isinstance(value, int):
        return value if value in INDEX_TO_LABEL else None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned.isdigit():
            parsed = int(cleaned)
            return parsed if parsed in INDEX_TO_LABEL else None
        upper = cleaned.upper()
        if upper in LABEL_TO_INDEX:
            return LABEL_TO_INDEX[upper]
    return None


def classification_to_label_index(classification: str) -> Optional[int]:
    label = (classification or "").strip().upper()
    if label in LABEL_TO_INDEX:
        return LABEL_TO_INDEX[label]
    if label.isdigit():
        parsed = int(label)
        return parsed if parsed in INDEX_TO_LABEL else None
    return None


def compute_multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    labels = [0, 1, 2]
    confusion_matrix = {true_label: {pred_label: 0 for pred_label in labels} for true_label in labels}

    for truth, pred in zip(y_true, y_pred):
        if truth in confusion_matrix and pred in confusion_matrix[truth]:
            confusion_matrix[truth][pred] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1_scores: List[float] = []

    for label in labels:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in labels if other != label)
        fn = sum(confusion_matrix[label][other] for other in labels if other != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[str(label)] = {
            "label": INDEX_TO_LABEL[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion_matrix[label].values()),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = sum(confusion_matrix[label][label] for label in labels) / len(y_true) if y_true else 0.0

    return {
        "count": len(y_true),
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def load_instances(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("instances"), list):
            return data["instances"]
        if isinstance(data.get("data"), list):
            return data["data"]

    raise ValueError("Input JSON must be a list, or a dict containing an 'instances'/'data' list.")


def select_instances_with_arxiv_id(
    instances: Sequence[Dict[str, Any]],
    max_instances: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []

    for instance in instances:
        citation_metadata = instance.get("citation_metadata") or {}
        arxiv_id = extract_arxiv_id_from_identifiers(citation_metadata)
        if not arxiv_id:
            continue

        selected.append(instance)
        if max_instances > 0 and len(selected) >= max_instances:
            break

    return selected


def resolve_reference_from_arxiv(instance: Dict[str, Any], cache_dir: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    citation_metadata = instance.get("citation_metadata") or {}
    title = (citation_metadata.get("title") or "").strip()

    details: Dict[str, Any] = {
        "title": title,
        "source": None,
        "arxiv_id": None,
        "paper_id": None,
        "pdf_url": None,
        "status": "unresolved",
        "error": None,
    }

    try:
        arxiv_id = extract_arxiv_id_from_identifiers(citation_metadata)
        pdf_url: Optional[str] = None

        if arxiv_id:
            pdf_url = arxiv_id_to_pdf_url(arxiv_id)
            details["source"] = "identifier"
            details["arxiv_id"] = arxiv_id
            details["pdf_url"] = pdf_url
        else:
            try:
                candidates = query_arxiv_by_title(title=title)
                best = pick_best_arxiv_match(target_title=title, candidates=candidates)
                if best:
                    arxiv_id = best.get("arxiv_id") or None
                    pdf_url = best.get("pdf_url") or None
                    details["source"] = "title_search"
                    details["arxiv_id"] = arxiv_id
                    details["pdf_url"] = pdf_url
            except Exception as exc:
                details["arxiv_title_search_error"] = str(exc)

        if not pdf_url:
            try:
                candidates = query_semantic_scholar_by_title(title=title)
                best = pick_best_semantic_scholar_match(target_title=title, candidates=candidates)
                if best:
                    arxiv_id = best.get("arxiv_id") or arxiv_id
                    pdf_url = best.get("pdf_url") or None
                    details["source"] = "semantic_scholar_title_search"
                    details["paper_id"] = best.get("paper_id") or None
                    details["arxiv_id"] = arxiv_id
                    details["pdf_url"] = pdf_url
            except Exception as exc:
                details["semantic_scholar_title_search_error"] = str(exc)

        if not pdf_url:
            details["status"] = "unresolved"
            details["error"] = "No arXiv or Semantic Scholar match found"
            return None, details

        file_stem = safe_filename(arxiv_id or details.get("paper_id") or title or "paper")
        pdf_path = cache_dir / f"{file_stem}.pdf"

        if not pdf_path.exists():
            download_file(pdf_url, pdf_path)

        reference_text = extract_pdf_text(pdf_path)
        if not reference_text:
            details["status"] = "error"
            details["error"] = "Extracted empty text from PDF"
            return None, details

        details["status"] = "resolved"
        details["cached_pdf"] = str(pdf_path)
        details["reference_chars"] = len(reference_text)
        return reference_text, details

    except Exception as exc:
        details["status"] = "error"
        details["error"] = str(exc)
        return None, details


def build_batch_for_checker(
    instances: Sequence[Dict[str, Any]],
    cache_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []

    for index, instance in enumerate(instances):
        claim_text = (instance.get("claim_text") or "").strip()
        if not claim_text:
            mapping.append(
                {
                    "index": index,
                    "status": "skipped",
                    "error": "Missing claim_text",
                    "resolved_reference": None,
                }
            )
            continue

        reference_text, ref_details = resolve_reference_from_arxiv(instance, cache_dir)
        if not reference_text:
            mapping.append(
                {
                    "index": index,
                    "status": "skipped",
                    "error": ref_details.get("error") or "Could not resolve reference text",
                    "resolved_reference": ref_details,
                }
            )
            continue

        citation_metadata = instance.get("citation_metadata") or {}
        metadata_blob = {
            "title": citation_metadata.get("title"),
            "authors": citation_metadata.get("authors"),
            "venue": citation_metadata.get("venue"),
            "year": citation_metadata.get("year"),
            "identifiers": citation_metadata.get("identifiers"),
            # "surrounding_context": instance.get("surrounding_context"),
        }

        pair_id = f"instance-{index}"
        batch.append(
            {
                "pair_id": pair_id,
                "citation": claim_text,
                "reference_text": reference_text,
                "metadata": json.dumps(metadata_blob, ensure_ascii=False),
            }
        )

        mapping.append(
            {
                "index": index,
                "status": "ready",
                "pair_id": pair_id,
                "resolved_reference": ref_details,
            }
        )

    return batch, mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark batch citation verification using arXiv or Semantic Scholar PDFs.")
    parser.add_argument("--input", required=True, help="Path to input JSON containing benchmark instances.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON report path. Defaults to benchmark/results_<timestamp>.json",
    )
    parser.add_argument("--max-instances", type=int, default=0, help="Limit number of instances (0 means all).")
    parser.add_argument(
        "--max-arxiv-id-instances",
        type=int,
        default=0,
        help="Process only instances that already have an arXiv ID, capped at this many examples (0 means all matching instances).",
    )
    parser.add_argument(
        "--target-label-field",
        default="true_alignment",
        help="Field under true_outputs to compare against predicted classification.",
    )

    parser.add_argument("--llm-provider", default="together", help="LLM provider passed to check_reference_batch.")
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-7B-Instruct-Turbo",
        help="LLM model name passed to check_reference_batch.",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature.")

    parser.add_argument("--embedding-provider", default="local", help="Embedding provider for checker.")
    parser.add_argument(
        "--embedding-model",
        default="all-mpnet-base-v2",
        help="Embedding model name for checker.",
    )

    parser.add_argument("--max-concurrency", type=int, default=2, help="Batch checker max concurrency.")
    parser.add_argument("--save-chunks", action="store_true", help="Save retrieval chunks.")
    parser.add_argument("--chunks-output-dir", default="./retrieval_output", help="Chunk output directory.")
    parser.add_argument(
        "--cache-dir",
        default="./benchmark/arxiv_cache",
        help="Directory used to cache downloaded arXiv PDFs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else Path(
        f"benchmark/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(input_path)
    if args.max_arxiv_id_instances and args.max_arxiv_id_instances > 0:
        instances = select_instances_with_arxiv_id(instances, args.max_arxiv_id_instances)
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]

    if not instances:
        raise ValueError("No instances found in input after applying filters.")

    print(f"Loaded {len(instances)} instance(s) from {input_path}")
    if args.max_arxiv_id_instances and args.max_arxiv_id_instances > 0:
        print(f"Filtered to instances with arXiv IDs (limit {args.max_arxiv_id_instances})")

    batch, mapping = build_batch_for_checker(instances=instances, cache_dir=cache_dir)
    print(f"Resolved {len(batch)} instance(s) with arXiv reference text")

    batch_result: Dict[str, Any] = {"results": [], "llm_metrics": {}}
    if batch:
        batch_result = check_reference_batch(
            citation_reference_pairs=batch,
            llm_config={
                "provider": args.llm_provider,
                "model": args.llm_model,
                "temperature": args.temperature,
            },
            embedding_config={
                "provider": args.embedding_provider,
                "model_name": args.embedding_model,
            },
            save_chunks=args.save_chunks,
            output_dir=args.chunks_output_dir,
            max_concurrency=args.max_concurrency,
        )

    checker_results = batch_result.get("results", [])
    llm_metrics = batch_result.get("llm_metrics", {})
    result_by_pair_id = {item.get("metadata", {}).get("pair_id"): item for item in checker_results}

    evaluation_rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    pred_class_counter = Counter()

    for item in mapping:
        idx = item["index"]
        instance = instances[idx]
        truth_raw = (instance.get("true_outputs") or {}).get(args.target_label_field)
        truth = normalize_encoded_label(truth_raw)

        row: Dict[str, Any] = {
            "index": idx,
            "status": item["status"],
            "claim_text": instance.get("claim_text"),
            "target_label_field": args.target_label_field,
            "true_label": truth,
            "resolved_reference": item.get("resolved_reference"),
            "error": item.get("error"),
            "prediction": None,
        }

        if item["status"] == "ready":
            pair_id = item.get("pair_id")
            prediction = result_by_pair_id.get(pair_id)
            row["prediction"] = prediction

            predicted_class = (prediction or {}).get("classification")
            pred_class_counter[predicted_class or "UNKNOWN"] += 1

            pred_label = classification_to_label_index(predicted_class or "")

            if truth is not None and pred_label is not None:
                y_true.append(truth)
                y_pred.append(pred_label)

        evaluation_rows.append(row)

    metrics = compute_multiclass_metrics(y_true, y_pred)

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "input": str(input_path),
            "total_instances": len(instances),
            "resolved_for_evaluation": len(batch),
            "evaluated_instances": len(y_true),
            "target_label_field": args.target_label_field,
            "llm": {
                "provider": args.llm_provider,
                "model": args.llm_model,
                "temperature": args.temperature,
            },
            "embedding": {
                "provider": args.embedding_provider,
                "model_name": args.embedding_model,
            },
        },
        "classification_counts": dict(pred_class_counter),
        "metrics": metrics,
        "results": evaluation_rows,
        "llm_metrics": llm_metrics,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=json_default)

    print("\nBenchmark summary")
    print(f"  output: {output_path}")
    print(f"  evaluated: {metrics['count']}")
    print(f"  macro_precision: {metrics['macro_precision']:.4f}")
    print(f"  macro_recall: {metrics['macro_recall']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")

    if llm_metrics:
        print("\nLLM run metrics")
        print(f"  total_calls:            {llm_metrics.get('total_calls', 'N/A')}")
        print(f"  total_latency_seconds:  {llm_metrics.get('total_latency_seconds', 'N/A')}")
        print(f"  avg_latency_seconds:    {llm_metrics.get('avg_latency_seconds', 'N/A')}")
        print(f"  total_input_tokens:     {llm_metrics.get('total_input_tokens', 'N/A')}")
        print(f"  total_output_tokens:    {llm_metrics.get('total_output_tokens', 'N/A')}")
        print(f"  total_tokens:           {llm_metrics.get('total_tokens', 'N/A')}")
        print(f"  estimated_cost_usd:     {llm_metrics.get('estimated_cost_usd', 'N/A')}")
        print(f"  avg_time_per_instance:  {llm_metrics.get('avg_time_per_instance', 'N/A')}s")


if __name__ == "__main__":
    main()
