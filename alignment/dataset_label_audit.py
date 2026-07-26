"""
Audit script for negatives_added_over_claim_with_citation.json.

For each entry:
  1. Classify claim + evidence via LLM into one of: PARTIALLY, UNSUPPORTED, UNCERTAIN
  2. Compare with adversarial_metadata.target_alignment_label
  3. If mismatch, rewrite the claim_text so the label is clearly correct

Output: *_audited.json with appended audit fields.
"""

from __future__ import annotations

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
from typing import Any, Dict, List, Optional

import dotenv

from security_utils import sanitize_error_message

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
    os.path.join(LOGS_DIR, 'dataset_label_audit.log'),
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

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-Turbo"
MODEL_PRICING = {"input": 0.30, "output": 0.30}
MAX_CONCURRENCY = 5
TEMPERATURE = 0.7
MAX_RETRIES = 3
BASE_DELAY = 1.5

INPUT_PATH = ALIGNMENT_DIR / "data" / "negatives_added_over_claim_with_citation.json"
OUTPUT_PATH = ALIGNMENT_DIR / "data" / "negatives_added_over_claim_with_citation_audited.json"

VALID_LABELS = {"PARTIALLY", "UNSUPPORTED", "UNCERTAIN"}


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


def _estimate_together_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    pricing = MODEL_PRICING
    input_rate = pricing.get("input")
    output_rate = pricing.get("output")
    if input_rate is None or output_rate is None:
        return None
    return round(
        (prompt_tokens * float(input_rate) + completion_tokens * float(output_rate)) / 1_000_000,
        6,
    )


def _parse_json_response(response: str) -> Dict[str, Any]:
    response_text = response.strip()
    if not response_text:
        return {"classification": "UNCERTAIN", "reasoning": "", "confidence_score": 0.0}
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            logger.warning(f"Could not parse JSON from model response: {response_text[:120]}")
            return {"classification": "UNCERTAIN", "reasoning": "", "confidence_score": 0.0}
        parsed = json.loads(match.group(0))

    classification = str(parsed.get("classification", "UNCERTAIN")).strip().upper()
    if classification == "PARTIALLY_SUPPORTED":
        classification = "PARTIALLY"
    elif classification == "REFUTED":
        classification = "UNSUPPORTED"
    elif classification == "NEI":
        classification = "UNCERTAIN"
    elif classification == "SUPPORTED":
        classification = "SUPPORTED"

    if classification not in VALID_LABELS and classification != "SUPPORTED":
        classification = "UNCERTAIN"

    parsed["classification"] = classification
    parsed["reasoning"] = parsed.get("reasoning", "")
    parsed["confidence_score"] = float(parsed.get("confidence_score", 0.0))
    parsed["rewritten_claim"] = parsed.get("rewritten_claim", "")
    parsed["rewrite_rationale"] = parsed.get("rewrite_rationale", "")
    return parsed


def _build_classification_prompt(claim: str, context: str, chunks: List[Dict[str, Any]]) -> str:
    evidence_texts = []
    for i, chunk in enumerate(chunks):
        score = chunk.get("relevance_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        evidence_texts.append(
            f"[Extractive Chunk {i + 1}] (score: {score_str})\n{chunk['extractive_text']}"
        )
    evidence_block = "\n\n".join(evidence_texts)

    return f"""You are an expert fact-checking assistant evaluating semantic alignment in academic texts.
Analyze whether the provided evidence from a reference paper supports the given claim.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked.
Focus strictly on the relationship between the claim's core assertion regarding this reference and the provided evidence.

Claim: "{claim}"

Surrounding Context: "{context}"

Raw Extractive Evidence Chunks:
{evidence_block}

Classify the alignment into EXACTLY ONE of the following 3 categories:
- PARTIALLY: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the evidence.
- UNSUPPORTED: The evidence explicitly contradicts the claim, or the claim completely shifts the condition/context of the evidence.
- UNCERTAIN: There is no relevant information in the evidence to assess the claim (tangential hallucination).

Return your response as valid JSON with this exact structure:
{{
    "classification": "<PARTIALLY | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed evidence-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}}

Return ONLY valid JSON, no other text."""


def _build_rewrite_prompt(claim: str, context: str, chunks: List[Dict[str, Any]], target_label: str) -> str:
    evidence_texts = []
    for i, chunk in enumerate(chunks):
        score = chunk.get("relevance_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
        evidence_texts.append(
            f"[Extractive Chunk {i + 1}] (score: {score_str})\n{chunk['extractive_text']}"
        )
    evidence_block = "\n\n".join(evidence_texts)

    return f"""You are an expert in academic text generation.
Given a claim, its surrounding context, and evidence chunks, the dataset label for this claim is "{target_label}" but the evidence does not support that label.

Your task: Rewrite the claim_text so that the dataset label "{target_label}" is clearly and unambiguously correct given the provided evidence.

Guidelines:
- Keep the rewritten claim about the same topic and reference paper ([CITATION] marker).
- If label is PARTIALLY: The claim should exaggerate or over-generalize the evidence while still being grounded in it.
- If label is UNSUPPORTED: The claim should contradict or make claims absent from the evidence.
- If label is UNCERTAIN: The claim should be about a topic not covered by the evidence at all.
- The rewrite should be fluent, academic in tone, and plausible-sounding.

Original Claim: "{claim}"

Surrounding Context: "{context}"

Evidence Chunks:
{evidence_block}

Return your response as valid JSON with this exact structure:
{{
    "rewritten_claim": "<the rewritten claim text>",
    "rewrite_rationale": "<brief explanation of how the rewrite makes the label correct>"
}}

Return ONLY valid JSON, no other text."""


class DatasetAuditor:
    def __init__(self):
        self._api_key = os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY")
        if not self._api_key:
            raise RuntimeError("Together API key not set. Set TOGETHER_API or TOGETHER_API_KEY env var.")
        self._client = None
        self._call_metrics: List[Dict[str, Any]] = []

    def _lazy_client(self):
        if self._client is not None:
            return self._client
        from together import AsyncTogether
        self._client = AsyncTogether(api_key=self._api_key)
        return self._client

    async def _llm_completion(self, prompt: str) -> str:
        client = self._lazy_client()
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start = time.perf_counter()
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                )
                latency = time.perf_counter() - start

                usage = getattr(response, "usage", None)
                prompt_tokens = _extract_usage_count(usage, "prompt_tokens")
                completion_tokens = _extract_usage_count(usage, "completion_tokens")
                total_tokens = _extract_usage_count(usage, "total_tokens") or prompt_tokens + completion_tokens
                cost = _estimate_together_cost(MODEL_NAME, prompt_tokens, completion_tokens)

                self._call_metrics.append({
                    "latency_seconds": round(latency, 4),
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": cost,
                })

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < MAX_RETRIES:
                        delay = BASE_DELAY * (2 ** (attempt - 1))
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
                if retryable and attempt < MAX_RETRIES:
                    delay = BASE_DELAY * (2 ** (attempt - 1))
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"LLM error: {safe_error}")
                raise

    async def _classify(self, claim: str, context: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {"classification": "UNCERTAIN", "reasoning": "No evidence chunks.", "confidence_score": 0.0}
        prompt = _build_classification_prompt(claim, context, chunks)
        response = await self._llm_completion(prompt)
        return _parse_json_response(response)

    async def _rewrite(self, claim: str, context: str, chunks: List[Dict[str, Any]], target_label: str) -> Dict[str, Any]:
        if not chunks:
            return {"rewritten_claim": "", "rewrite_rationale": "No evidence chunks to base rewrite on."}
        prompt = _build_rewrite_prompt(claim, context, chunks, target_label)
        response = await self._llm_completion(prompt)
        return _parse_json_response(response)

    def get_stats(self) -> Dict[str, Any]:
        metrics = self._call_metrics or []
        total = len(metrics)
        if total == 0:
            return {"total_calls": 0, "total_cost_usd": None}
        total_cost = sum(m.get("estimated_cost_usd") or 0 for m in metrics)
        total_latency = sum(m.get("latency_seconds", 0) for m in metrics)
        total_tokens = sum(m.get("total_tokens", 0) for m in metrics)
        return {
            "total_calls": total,
            "total_latency_seconds": round(total_latency, 2),
            "avg_latency_seconds": round(total_latency / total, 4) if total else 0,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(total_cost, 6),
        }

    async def audit(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(MAX_CONCURRENCY)

        async def process_one(idx: int, inst: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                claim = (inst.get("claim_text") or "").strip()
                context = (inst.get("surrounding_context") or "").strip()
                retrieved = inst.get("retrieved_evidences") or {}
                chunks = retrieved.get("extractive_chunks") or []
                adv_meta = inst.get("adversarial_metadata") or {}
                target_label = adv_meta.get("target_alignment_label", "UNCERTAIN")

                result = dict(inst)

                try:
                    class_result = await self._classify(claim, context, chunks)
                    llm_label = class_result["classification"]
                    result["llm_judged_label"] = llm_label
                    result["llm_reasoning"] = class_result["reasoning"]
                    result["llm_confidence"] = class_result["confidence_score"]
                except Exception as e:
                    logger.error(f"Classification failed for instance {idx}: {sanitize_error_message(e)}")
                    result["llm_judged_label"] = "UNCERTAIN"
                    result["llm_reasoning"] = f"Error: {sanitize_error_message(e)}"
                    result["llm_confidence"] = 0.0
                    llm_label = "UNCERTAIN"

                if llm_label == "SUPPORTED":
                    label_mismatch = True
                else:
                    label_mismatch = llm_label != target_label

                result["label_needs_correction"] = label_mismatch

                if label_mismatch:
                    try:
                        rewrite_result = await self._rewrite(claim, context, chunks, target_label)
                        result["rewritten_claim"] = rewrite_result.get("rewritten_claim", "")
                        result["rewrite_rationale"] = rewrite_result.get("rewrite_rationale", "")
                    except Exception as e:
                        logger.error(f"Rewrite failed for instance {idx}: {sanitize_error_message(e)}")
                        result["rewritten_claim"] = ""
                        result["rewrite_rationale"] = f"Error: {sanitize_error_message(e)}"
                else:
                    result["rewritten_claim"] = ""
                    result["rewrite_rationale"] = ""

                if (idx + 1) % 50 == 0 or idx == 0 or idx == len(instances) - 1:
                    print(f"  [{idx + 1}/{len(instances)}] label={target_label}, llm={result['llm_judged_label']}, needs_correction={result['label_needs_correction']}")

                return result

        tasks = [process_one(i, inst) for i, inst in enumerate(instances)]
        return await asyncio.gather(*tasks)


def main():
    print(f"Loading {INPUT_PATH}...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        instances = json.load(f)
    print(f"Loaded {len(instances)} instances.")

    label_dist = Counter()
    for inst in instances:
        am = inst.get("adversarial_metadata") or {}
        label_dist[am.get("target_alignment_label", "N/A")] += 1
    print(f"Label distribution: {dict(label_dist)}")

    auditor = DatasetAuditor()
    print(f"\nStarting audit with {MODEL_NAME}...")
    results = asyncio.run(auditor.audit(instances))
    print(f"\nAudit complete.")

    correction_count = sum(1 for r in results if r.get("label_needs_correction"))
    print(f"Entries needing correction: {correction_count}/{len(results)}")

    corrected_by_label = Counter()
    for r in results:
        if r.get("label_needs_correction"):
            am = r.get("adversarial_metadata") or {}
            corrected_by_label[am.get("target_alignment_label", "N/A")] += 1
    print(f"Corrections by label: {dict(corrected_by_label)}")

    llm_pred_dist = Counter(r.get("llm_judged_label") for r in results)
    print(f"LLM judged distribution: {dict(llm_pred_dist)}")

    stats = auditor.get_stats()
    print(f"\nLLM Stats:")
    print(f"  Total calls: {stats['total_calls']}")
    print(f"  Total latency: {stats['total_latency_seconds']}s")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Estimated cost: ${stats['estimated_cost_usd']:.6f}" if stats['estimated_cost_usd'] else "  Estimated cost: N/A")

    print(f"\nWriting {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done.")


if __name__ == "__main__":
    main()
