"""
Benchmark evaluator for citation alignment using pre-retrieved evidence chunks.

Takes an enriched dataset (with ``retrieved_evidences.extractive_chunks``),
classifies each claim as SUPPORTED / UNSUPPORTED / UNCERTAIN via LLM,
and computes classification metrics against ground-truth labels.

Usage:
    python alignment/benchmark_evaluator.py --input enriched.json --output results.json
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
    LLMResponse,
    summarize_llm_run_metrics,
    TOGETHER_MODEL_OPTIONS,
)


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

LABEL_TO_INDEX = {
    "SUPPORTED": 0,
    "UNSUPPORTED": 1,
    "UNCERTAIN": 2,
}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


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
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            logger.warning(f"Could not parse JSON from model response: {response_text[:120]}")
            return {"classification": "UNCERTAIN", "reasoning": "", "confidence_score": 0.0}
        parsed_response = json.loads(match.group(0))
    classification = str(parsed_response.get("classification", "UNCERTAIN")).strip().upper()
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
) -> str:
    evidence_texts = []
    for i, chunk in enumerate(chunks):
        score = chunk.get("relevance_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        evidence_texts.append(
            f"[Chunk {i + 1}] (score: {score_str})\n{chunk['extractive_text']}"
        )
    evidence_block = "\n\n".join(evidence_texts)
    return f"""You are an expert fact-checking assistant. Analyze whether the evidence chunks from a reference paper support the given claim.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked. Focus on whether the evidence supports the portion of the claim attributed to that reference. The claim may contain additional context or comparisons not directly related to the reference — ignore those when judging support.

Claim: "{claim}"

Surrounding Context: "{context}"

Evidence Chunks:
{evidence_block}

Determine if the evidence supports the claim. Classify as one of:

SUPPORTED - The evidence directly backs the claim or a close paraphrase. Even if part of the claim contains unrelated context, if the core information tied to [CITATION] matches the evidence, label SUPPORTED.
UNSUPPORTED - The evidence explicitly contradicts the claim, or the claim makes a statement inconsistent with the evidence.
UNCERTAIN - There is not enough information in the evidence to determine support or contradiction. The evidence is partial, indirect, topic-adjacent, or ambiguous.

Guidelines:
- Focus on whether the evidence supports the part of the claim that refers to the [CITATION] marker. Extra surrounding context in the claim should not penalize the judgment.
- Return SUPPORTED if the evidence aligns with the claim's core assertion about the reference, even if some peripheral details in the claim are not verifiable from the evidence.
- Return UNSUPPORTED only when the evidence explicitly contradicts the claim's core assertion about the reference.
- Return UNCERTAIN only when the evidence is truly irrelevant, off-topic, or entirely insufficient to assess the reference-related portion of the claim.
- Do not be overly strict — if the evidence matches the gist of what the claim says about the reference, SUPPORTED is appropriate.

Return your response as valid JSON with this exact structure:
{{
    "classification": "one of: SUPPORTED, UNSUPPORTED, UNCERTAIN",
    "reasoning": "your detailed reasoning here",
    "confidence_score": <float between 0.0 and 1.0>
}}

Return ONLY valid JSON, no other text."""


class BenchmarkEvaluator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_concurrency: int = 5,
    ):
        self._llm_client = llm_client
        self.max_concurrency = max_concurrency
        self._call_metrics: List[LLMResponse] = []

    async def _classify_evidence_async(
        self,
        claim: str,
        context: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not chunks:
            return {
                "classification": "UNCERTAIN",
                "reasoning": "No evidence chunks retrieved for this claim.",
                "confidence_score": 0.0,
            }
        prompt = _build_evidence_classification_prompt(claim, context, chunks)
        response = await self._llm_client.generate(prompt)
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

        async def evaluate_one(idx: int, inst: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                claim = (inst.get("claim_text") or "").strip()
                context = (inst.get("surrounding_context") or "").strip()
                retrieved = inst.get("retrieved_evidences") or {}
                chunks = retrieved.get("extractive_chunks") or []
                true_outputs = inst.get("true_outputs") or {}
                true_alignment = true_outputs.get("true_alignment")

                try:
                    result = await self._classify_evidence_async(claim, context, chunks)
                    classification = result["classification"]
                    reasoning = result["reasoning"]
                    confidence = result["confidence_score"]
                except Exception as e:
                    classification = "UNCERTAIN"
                    reasoning = f"Evaluation error: {sanitize_error_message(e)}"
                    confidence = 0.0

                pred_label = LABEL_TO_INDEX.get(classification)
                return {
                    "instance_id": inst.get("instance_id", idx),
                    "claim_text": claim,
                    "surrounding_context": context,
                    "true_alignment": true_alignment,
                    "predicted_classification": classification,
                    "predicted_label": pred_label,
                    "reasoning": reasoning,
                    "confidence_score": confidence,
                    "num_evidence_chunks": len(chunks),
                    "citation_metadata": inst.get("citation_metadata"),
                }

        tasks = [evaluate_one(idx, inst) for idx, inst in enumerate(instances)]
        return await asyncio.gather(*tasks)


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
        description="Evaluate citation alignment benchmark using pre-retrieved evidence chunks."
    )
    parser.add_argument("--input", required=True, help="Path to enriched JSON with retrieved_evidences.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--max-instances", type=int, default=0, help="Limit number of instances (0 = all).")
    parser.add_argument(
        "--llm-model",
        default=TOGETHER_MODEL_OPTIONS[0],
        help="LLM model name (default: TogetherAI).",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM temperature.")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent evaluations.")
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(f"benchmark/benchmark_results_{args.llm_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instances = load_instances(input_path)
    if args.max_instances and args.max_instances > 0:
        instances = instances[: args.max_instances]
    if not instances:
        raise ValueError("No instances found in input.")

    print(f"Loaded {len(instances)} instance(s) from {input_path}")

    llm_client = TogetherLLMClient(
        model=args.llm_model,
        temperature=args.temperature,
    )
    evaluator = BenchmarkEvaluator(
        llm_client=llm_client,
        max_concurrency=args.concurrency,
    )

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

    if llm_metrics:
        print("\nLLM run metrics")
        print(f"  total_calls:            {llm_metrics['total_calls']}")
        print(f"  total_latency_seconds:  {llm_metrics['total_latency_seconds']}")
        print(f"  avg_latency_seconds:    {llm_metrics['avg_latency_seconds']}")
        print(f"  avg_time_per_instance:  {llm_metrics.get('avg_time_per_instance', 'N/A')}s")
        print(f"  total_input_tokens:     {llm_metrics['total_input_tokens']}")
        print(f"  total_output_tokens:    {llm_metrics['total_output_tokens']}")
        print(f"  total_tokens:           {llm_metrics['total_tokens']}")
        cost = llm_metrics.get('estimated_cost_usd')
        print(f"  estimated_cost_usd:     ${cost:.6f}" if cost is not None else "  estimated_cost_usd:     N/A")


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

# python .\alignment\benchmark_evaluator.py --input .\alignment\data\combined.json --output benchmark.json --max-instances 100