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
import time
from collections import Counter
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import dotenv

from security_utils import sanitize_error_message

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

TOGETHER_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.7-Plus",
]

TOGETHER_MODEL_PRICING = {
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {"input": 0.30, "output": 0.30},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {"input": 0.14, "output": 0.14},
    "Qwen/Qwen3.5-9B": {"input": 0.17, "output": 0.25},
    "google/gemma-4-31B-it": {"input": 0.39, "output": 0.97},
}

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


def _extract_usage_count(usage: Any, field_name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field_name, 0)
    else:
        value = getattr(usage, field_name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _estimate_together_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    llm_config: Dict[str, Any],
) -> Optional[float]:
    input_rate = llm_config.get("input_cost_per_million_tokens")
    output_rate = llm_config.get("output_cost_per_million_tokens")
    pricing = TOGETHER_MODEL_PRICING.get(model_name)
    if input_rate is None and pricing:
        input_rate = pricing.get("input")
    if output_rate is None and pricing:
        output_rate = pricing.get("output")
    if input_rate is None or output_rate is None:
        return None
    return round(
        (prompt_tokens * float(input_rate) + completion_tokens * float(output_rate)) / 1_000_000,
        6,
    )


def _summarize_llm_run_metrics(call_metrics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    metrics = list(call_metrics or [])
    total_calls = len(metrics)
    if total_calls == 0:
        return {
            "provider": "together",
            "model": None,
            "total_calls": 0,
            "avg_latency_seconds": 0.0,
            "total_latency_seconds": 0.0,
            "avg_input_tokens_per_sample": 0.0,
            "avg_output_tokens_per_sample": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "cost_estimation_available": False,
        }
    total_latency = sum(float(item.get("latency_seconds", 0.0)) for item in metrics)
    total_input_tokens = sum(int(item.get("input_tokens", 0)) for item in metrics)
    total_output_tokens = sum(int(item.get("output_tokens", 0)) for item in metrics)
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in metrics)
    estimated_costs = [
        item.get("estimated_cost_usd")
        for item in metrics
        if item.get("estimated_cost_usd") is not None
    ]
    estimated_cost_usd = round(sum(float(c) for c in estimated_costs), 6) if estimated_costs else None
    return {
        "provider": metrics[0].get("provider", "together"),
        "model": metrics[0].get("model"),
        "total_calls": total_calls,
        "avg_latency_seconds": round(total_latency / total_calls, 4),
        "total_latency_seconds": round(total_latency, 4),
        "avg_input_tokens_per_sample": round(total_input_tokens / total_calls, 2),
        "avg_output_tokens_per_sample": round(total_output_tokens / total_calls, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_estimation_available": estimated_cost_usd is not None,
    }


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
    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            logger.warning(f"Could not parse JSON from model response: {response_text[:120]}")
            return {"classification": "UNCERTAIN", "reasoning": "", "confidence_score": 0.0}
        parsed_response = json.loads(match.group(0))
        
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

Raw Extractive Evidence Chunks:
{evidence_block}
---

Classify the alignment into EXACTLY ONE of the following 4 categories:
- SUPPORTED: The evidence directly backs the claim or a close paraphrase without logical gaps.
- PARTIALLY_SUPPORTED: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the evidence.
- UNSUPPORTED: The evidence explicitly contradicts the claim, or the claim completely shifts the condition/context of the evidence.
- UNCERTAIN: There is no relevant information in the evidence to assess the claim (tangential hallucination).

Return your response as valid JSON with this exact structure:
{{
    "classification": "<SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed evidence-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}}

Return ONLY valid JSON, no other text."""

# ==============================================================================
# Core Evaluator Class
# ==============================================================================

class BenchmarkEvaluator:
    def __init__(
        self,
        model: str = TOGETHER_MODEL_OPTIONS[0],
        temperature: float = 0.7,
        max_concurrency: int = 5,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_concurrency = max_concurrency
        self._api_key = api_key or os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY")
        self._call_metrics: List[Dict[str, Any]] = []
        self._client: Optional[Any] = None

    def _lazy_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("Together API key not set in environment.")
        from together import AsyncTogether
        self._client = AsyncTogether(api_key=self._api_key)
        return self._client

    async def _llm_completion_async(self, prompt: str) -> str:
        client = self._lazy_client()
        model_name = self.model
        max_attempts = 3
        base_delay = 1.5

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                latency_seconds = time.perf_counter() - start_time

                usage = getattr(response, "usage", None)
                prompt_tokens = _extract_usage_count(usage, "prompt_tokens")
                completion_tokens = _extract_usage_count(usage, "completion_tokens")
                total_tokens = _extract_usage_count(usage, "total_tokens")
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens

                llm_config = {"input_cost_per_million_tokens": None, "output_cost_per_million_tokens": None}
                estimated_cost_usd = _estimate_together_cost(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    llm_config=llm_config,
                )

                self._call_metrics.append({
                    "provider": "together",
                    "model": model_name,
                    "latency_seconds": round(latency_seconds, 4),
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                })

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                        continue
                    raise ValueError("Empty model response after all retries")
                return content

            except Exception as e:
                safe_error = sanitize_error_message(e)
                error_text = str(e).lower()
                retryable = any(m in error_text for m in [
                    "rate limit", "too many requests", "429", "quota",
                    "resource exhausted", "server error", "502", "503", "504",
                    "timeout", "temporarily unavailable", "connection error",
                ])
                if retryable and attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue
                raise

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
        response = await self._llm_completion_async(prompt)
        return _parse_json_model_response(response)

    def get_llm_run_metrics(self) -> Dict[str, Any]:
        summary = _summarize_llm_run_metrics(self._call_metrics)
        summary["model"] = self.model
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
                synthesis = retrieved.get("abstractive_synthesis")
                
                true_outputs = inst.get("true_outputs") or {}
                true_alignment = true_outputs.get("true_alignment")
                
                # Handling nested AdvCite structure if present
                if "ground_truth" in inst and "task2_alignment" in inst["ground_truth"]:
                    true_alignment = inst["ground_truth"]["task2_alignment"].get("label", true_alignment)

                try:
                    result = await self._classify_evidence_async(claim, context, chunks, synthesis)
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
                    "has_abstractive_synthesis": bool(synthesis),
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
        description="Evaluate citation alignment benchmark using Hybrid Evidence."
    )
    parser.add_argument("--input", required=True, help="Path to enriched JSON with retrieved_evidences.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    parser.add_argument("--max-instances", type=int, default=0, help="Limit number of instances (0 = all).")
    parser.add_argument(
        "--llm-model",
        default=TOGETHER_MODEL_OPTIONS[0],
        help="TogetherAI model name.",
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

    evaluator = BenchmarkEvaluator(
        model=args.llm_model,
        temperature=args.temperature,
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

def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()