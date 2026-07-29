"""
Benchmark evaluator for citation alignment using Hybrid Evidence.

Takes an enriched dataset containing both extractive_chunks and abstractive_synthesis,
classifies each claim into the 4-level taxonomy (SUPPORTED / PARTIALLY_SUPPORTED / 
UNSUPPORTED / UNCERTAIN) via LLM, and computes rigorous classification metrics.

Complies with PEP-8 guidelines.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dotenv

from security_utils import sanitize_error_message
from llm_client import (
    BaseLLMClient,
    TogetherLLMClient,
    OpenRouterLLMClient,
    LLMResponse,
    summarize_llm_run_metrics,
    TOGETHER_MODEL_OPTIONS,
    OPENROUTER_MODEL_OPTIONS,
)

# ==============================================================================
# Environment & Logging
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIGNMENT_DIR = Path(__file__).resolve().parent

for path in (ALIGNMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

dotenv.load_dotenv(PROJECT_ROOT / ".env")

LOGS_DIR = str(PROJECT_ROOT / "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'benchmark_evaluator.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_formatter)

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# 4-Level Taxonomy for AdvCite Benchmark
LABEL_TO_INDEX = {
    "SUPPORTED": 0,
    "PARTIALLY_SUPPORTED": 1,
    "UNSUPPORTED": 2,
    "UNCERTAIN": 3,
}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

# ==============================================================================
# Helper Functions
# ==============================================================================

def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def compute_multiclass_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    labels = [0, 1, 2, 3]
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
        per_class[INDEX_TO_LABEL[label]] = {
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


def _parse_json_model_response(response: str) -> Dict[str, Any]:
    response_text = response.strip()
    if not response_text:
        return {"classification": "UNCERTAIN", "reasoning": "", "confidence_score": 0.0}

    def _try_parse(text: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    parsed_response = _try_parse(response_text)
    if parsed_response is not None:
        return _normalize_classification(parsed_response)

    # Attempt recovery for truncated JSON (missing closing brace)
    if response_text.startswith("{"):
        for suffix in ['"}', '"\n}', "}", "}\n", "\n}"]:
            recovered = _try_parse(response_text + suffix)
            if recovered is not None:
                logger.info("Recovered truncated JSON by appending %r", repr(suffix).strip("'"))
                parsed_response = recovered
                break

    if parsed_response is None:
        # fallback: use regex to find any JSON object
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if match:
            parsed_response = _try_parse(match.group(0))

    if parsed_response is None:
        # last resort: extract classification field via string search
        cls_match = re.search(r'"classification"\s*:\s*"([^"]+)"', response_text)
        label = cls_match.group(1).strip().upper() if cls_match else "UNCERTAIN"
        logger.warning("Could not parse JSON, extracted classification='%s' from truncated response", label)
        return {
            "classification": _resolve_label(label),
            "reasoning": "Truncated response — full reasoning unavailable.",
            "confidence_score": 0.0,
        }

    return _normalize_classification(parsed_response)


def _resolve_label(label: str) -> str:
    if label == "PARTIALLY":
        return "PARTIALLY_SUPPORTED"
    if label == "REFUTED":
        return "UNSUPPORTED"
    if label == "NEI":
        return "UNCERTAIN"
    if label not in LABEL_TO_INDEX:
        return "UNCERTAIN"
    return label


def _normalize_classification(parsed_response: Dict[str, Any]) -> Dict[str, Any]:
    classification = str(parsed_response.get("classification", "UNCERTAIN")).strip().upper()
    if classification == "PARTIALLY":
        classification = "PARTIALLY_SUPPORTED"
    if classification == "REFUTED":
        classification = "UNSUPPORTED"
    elif classification == "NEI":
        classification = "UNCERTAIN"
        
    if classification not in LABEL_TO_INDEX:
        classification = "UNCERTAIN"
        
    parsed_response["classification"] = classification
    parsed_response["reasoning"] = parsed_response.get("reasoning", "")
    parsed_response["confidence_score"] = float(parsed_response.get("confidence_score", 0.0))
    return parsed_response


def _build_evidence_classification_prompt(
    claim: str,
    context: str,
    chunks: List[Dict[str, Any]],
    abstractive_synthesis: Optional[str]
) -> str:
    evidence_texts = []
    for i, chunk in enumerate(chunks):
        score = chunk.get("relevance_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        evidence_texts.append(
            f"[Extractive Chunk {i + 1}] (score: {score_str})\n{chunk['extractive_text']}"
        )
    evidence_block = "\n\n".join(evidence_texts)
    
    synthesis_block = abstractive_synthesis if abstractive_synthesis else "No contextual synthesis available."

    return f"""You are an expert fact-checking assistant evaluating semantic alignment in academic texts. 
Analyze whether the provided evidence from a reference paper supports the given claim.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked. 
Focus strictly on the relationship between the claim's core assertion regarding this reference and the provided evidence.

Claim: "{claim}"

Surrounding Context: "{context}"

---
Abstractive Synthesis (Structural Denoised Context):
{synthesis_block}

Guidance: The Abstractive Synthesis above is a distilled summary of evidence.
If it tells us the evidence does not fully support the claim (or empty), the classification should lean toward
UNCERTAIN (no information) or UNSUPPORTED (chunks contradict), rather than SUPPORTED.
If it is empty then it is likely that the claim is not supported by the reference or UNCERTAIN.  

Raw Extractive Evidence Chunks:
{evidence_block}
---

Classify the alignment into EXACTLY ONE of the following 4 categories:
- SUPPORTED: The evidence directly backs the claim or a close paraphrase without logical gaps.
- PARTIALLY_SUPPORTED: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the evidence.
- UNSUPPORTED: The evidence explicitly contradicts the claim or has related findings but shifted in context.
- UNCERTAIN: There is no relevant information in the evidence to assess the claim.

Return your response as valid JSON with this exact structure:
{{
    "classification": "<SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed evidence-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}}

Return ONLY valid JSON, no other text."""

def _build_closed_book_classification_prompt(
    claim: str,
    context: str,
    citation_metadata: Dict[str, Any],
) -> str:
    meta = citation_metadata or {}
    title = (meta.get("title") or "").strip()
    authors = meta.get("authors") or []
    venue = (meta.get("venue") or "").strip()
    year = meta.get("year")

    author_str = "; ".join(authors) if authors else "Unknown"
    venue_str = f"{venue} ({year})" if year else venue

    ref_lines = ""
    if title:
        ref_lines += f"  Title: {title}\n"
    if author_str != "Unknown":
        ref_lines += f"  Authors: {author_str}\n"
    if venue_str:
        ref_lines += f"  Venue: {venue_str}\n"
    if not ref_lines:
        ref_lines = "  (no metadata available)\n"

    return f"""You are an expert fact-checking assistant evaluating semantic alignment in academic texts.
You are given a claim and the metadata of the reference paper it cites.
Based on the reference metadata and your own knowledge of the paper, classify whether
the claim is supported by the reference.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked.
Focus strictly on the relationship between the claim's core assertion regarding this reference.

Claim: "{claim}"

Surrounding Context: "{context}"

---
Reference Paper:
{ref_lines}---

Classify the alignment into EXACTLY ONE of the following 4 categories:
- SUPPORTED: The reference directly backs the claim or a close paraphrase without logical gaps.
- PARTIALLY_SUPPORTED: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the reference.
- UNSUPPORTED: The reference explicitly contradicts the claim or has related findings but shifted in context.
- UNCERTAIN: You do not have enough knowledge of the reference to assess the claim.

Return your response as valid JSON with this exact structure:
{{
    "classification": "<SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed knowledge-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}}

Return ONLY valid JSON, no other text."""


# ==============================================================================
# Core Evaluator Class
# ==============================================================================

class BenchmarkEvaluator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_concurrency: int = 5,
        closed_book: bool = False,
    ):
        self._llm_client = llm_client
        self.max_concurrency = max_concurrency
        self._closed_book = closed_book
        self._call_metrics: List[LLMResponse] = []

    async def _classify_evidence_async(
        self,
        claim: str,
        context: str,
        chunks: List[Dict[str, Any]],
        abstractive_synthesis: Optional[str]
    ) -> Dict[str, Any]:
        if not chunks and not abstractive_synthesis:
            return {
                "classification": "UNCERTAIN",
                "reasoning": "No evidence chunks or synthesis retrieved for this claim.",
                "confidence_score": 0.0,
            }
        prompt = _build_evidence_classification_prompt(claim, context, chunks, abstractive_synthesis)
        response = await self._llm_client.generate(prompt)
        # await asyncio.sleep(0.1)
        self._call_metrics.append(response)
        return _parse_json_model_response(response.content)

    async def _classify_closed_book_async(
        self,
        claim: str,
        context: str,
        citation_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = _build_closed_book_classification_prompt(claim, context, citation_metadata)
        response = await self._llm_client.generate(prompt)
        await asyncio.sleep(0.1)
        self._call_metrics.append(response)
        return _parse_json_model_response(response.content)

    def get_llm_run_metrics(self) -> Dict[str, Any]:
        summary = summarize_llm_run_metrics(self._call_metrics)
        if not summary.get("provider"):
            summary["provider"] = self._llm_client.get_provider()
        if summary["total_calls"] > 0:
            summary["avg_time_per_instance"] = round(
                summary["total_latency_seconds"] / summary["total_calls"], 4
            )
        else:
            summary["avg_time_per_instance"] = 0.0
        return summary

    async def evaluate_instances_async(
        self,
        instances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(instances)

        async def evaluate_one(idx: int, inst: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
            async with sem:
                claim = (inst.get("claim_text") or "").strip()
                context = (inst.get("surrounding_context") or "").strip()
                citation_metadata = inst.get("citation_metadata")
                retrieved = inst.get("retrieved_evidences") or {}
                chunks = retrieved.get("extractive_chunks") or []
                synthesis = retrieved.get("abstractive_synthesis")

                true_outputs = inst.get("true_outputs") or {}
                true_alignment = true_outputs.get("true_alignment")

                # Handling nested AdvCite structure if present
                if "ground_truth" in inst and "task2_alignment" in inst["ground_truth"]:
                    true_alignment = inst["ground_truth"]["task2_alignment"].get("label", true_alignment)

                try:
                    if self._closed_book:
                        result = await self._classify_closed_book_async(claim, context, citation_metadata)
                    else:
                        result = await self._classify_evidence_async(claim, context, chunks, synthesis)
                    classification = result["classification"]
                    reasoning = result["reasoning"]
                    confidence = result["confidence_score"]
                except Exception as e:
                    classification = "UNCERTAIN"
                    reasoning = f"Evaluation error: {sanitize_error_message(e)}"
                    confidence = 0.0

                pred_label = LABEL_TO_INDEX.get(classification)
                return idx, {
                    "instance_id": inst.get("instance_id", idx),
                    "claim_text": claim,
                    "surrounding_context": context,
                    "true_alignment": true_alignment,
                    "predicted_classification": classification,
                    "predicted_label": pred_label,
                    "reasoning": reasoning,
                    "confidence_score": confidence,
                    "num_evidence_chunks": len(chunks),
                    "extractive_chunks": chunks,
                    "abstractive_synthesis": synthesis,
                    "has_abstractive_synthesis": bool(synthesis),
                    "citation_metadata": citation_metadata,
                    "mode": "closed_book" if self._closed_book else "open_book",
                }

        tasks = [evaluate_one(idx, inst) for idx, inst in enumerate(instances)]
        results = [None] * total
        done = 0
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result
            done += 1
            logger.info("Processed %d/%d instances", done, total)

        return results


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate citation alignment benchmark using Hybrid Evidence."
    )
    parser.add_argument("--input", required=True, help="Path to enriched JSON with retrieved_evidences.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--max-instances", type=int, default=0, help="Limit number of instances (0 = all).")
    parser.add_argument(
        "--llm-provider",
        default="together",
        choices=["together", "openrouter"],
        help="LLM provider to use (default: together).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default: provider-specific default).",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature.")
    parser.add_argument("--api-key", default=None, help="TogetherAI API key (default: TOGETHER_API or TOGETHER_API_KEY2 env var).")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent evaluations.")
    parser.add_argument(
        "--closed-book",
        action="store_true",
        help="Run in closed-book mode: classify using reference metadata only (no evidence chunks).",
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"benchmark_final/benchmark_results_{args.llm_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instances = load_instances(input_path)
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]
    if not instances:
        raise ValueError("No instances found in input.")

    print(f"Loaded {len(instances)} instance(s) from {input_path}")

    model_name = args.llm_model

    if args.llm_provider == "openrouter":
        if not model_name:
            model_name = OPENROUTER_MODEL_OPTIONS[0]
        llm_client: BaseLLMClient = OpenRouterLLMClient(
            model=model_name,
            temperature=args.temperature,
            api_key=args.api_key,
        )
        logger.info("Using OpenRouter provider with model: %s", model_name)
    else:
        if not model_name:
            model_name = TOGETHER_MODEL_OPTIONS[0]
        llm_client = TogetherLLMClient(
            model=model_name,
            temperature=args.temperature,
            api_key=args.api_key,
        )
        logger.info("Using TogetherAI provider with model: %s", model_name)
    evaluator = BenchmarkEvaluator(
        llm_client=llm_client,
        max_concurrency=args.concurrency,
        closed_book=args.closed_book,
    )

    mode_str = "closed-book" if args.closed_book else "open-book"
    logger.info("Running in %s mode with %d instance(s)", mode_str, len(instances))

    results = await evaluator.evaluate_instances_async(instances)

    y_true: List[int] = []
    y_pred: List[int] = []
    pred_class_counter = Counter()

    for r in results:
        truth = r["true_alignment"]
        pred = r["predicted_label"]
        pred_class_counter[r["predicted_classification"]] += 1
        if truth is not None and pred is not None:
            y_true.append(truth)
            y_pred.append(pred)

    metrics = compute_multiclass_metrics(y_true, y_pred)
    llm_metrics = evaluator.get_llm_run_metrics()

    report = {
        "meta": {
            "created_at": datetime.now().isoformat(),
            "input": str(input_path),
            "total_instances": len(instances),
            "evaluated_instances": len(y_true),
            "mode": "closed_book" if args.closed_book else "open_book",
            "llm": {
                "model": args.llm_model,
                "temperature": args.temperature,
            },
        },
        "classification_counts": dict(pred_class_counter),
        "metrics": metrics,
        "llm_metrics": llm_metrics,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=json_default)

    print(f"\nResults written to {output_path}")
    print(f"  evaluated: {metrics['count']}")
    print(f"  macro_precision: {metrics['macro_precision']:.4f}")
    print(f"  macro_recall: {metrics['macro_recall']:.4f}")
    print(f"  macro_f1: {metrics['macro_f1']:.4f}")
    print(f"  accuracy: {metrics['accuracy']:.4f}")

def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()