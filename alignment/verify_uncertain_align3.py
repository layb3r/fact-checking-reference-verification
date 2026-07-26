"""
Verify and rewrite instances with true_alignment == 3 (UNCERTAIN).

For each instance where true_alignment == 3, the LLM checks if the claim
is genuinely UNCERTAIN (evidence chunks share no topical overlap with the
claim). If not, the claim is rewritten to be UNCERTAIN while preserving
the [CITATION] marker.
"""

import argparse
import asyncio
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv

from security_utils import sanitize_error_message

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(_PROJECT_ROOT / ".env")

LOGS_DIR = str(_PROJECT_ROOT / "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'verify_uncertain.log'),
    maxBytes=10 * 1024 * 1024, backupCount=5
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


class TogetherLLMClient:
    def __init__(
        self,
        model: str = TOGETHER_MODEL_OPTIONS[0],
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._api_key = api_key or os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY")
        self._client: Optional[Any] = None

    async def agenerate(self, prompt: str) -> str:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("Together API key not set.")
            from together import AsyncTogether
            self._client = AsyncTogether(api_key=self._api_key)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""


def _build_evidence_block(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(f"[Chunk {i + 1}]\n{c['extractive_text']}")
    return "\n\n".join(parts)


def _build_verify_prompt(claim: str, context: str, chunks: List[Dict[str, Any]]) -> str:
    evidence = _build_evidence_block(chunks)
    return f"""You are evaluating whether a claim is genuinely UNCERTAIN relative to given evidence.

A claim is UNCERTAIN if the evidence shares NO topical overlap, no related entities, methods, or findings with the claim — the evidence is entirely off-topic or unrelated.

Claim: "{claim}"

Surrounding Context: "{context}"

Evidence:
{evidence}

Question: Is the claim genuinely UNCERTAIN (evidence has zero topical overlap)?

Answer with exactly one word: YES or NO

YES = the evidence is entirely unrelated to the claim
NO = the evidence has ANY topical overlap, even partial"""


def _build_rewrite_prompt(claim: str, context: str, chunks: List[Dict[str, Any]]) -> str:
    evidence = _build_evidence_block(chunks)
    return f"""You are rewriting a claim to make it UNCERTAIN relative to the evidence.

The new claim MUST:
1. Be about a completely different topic from anything in the evidence
2. Still sound academically fluent and plausible
3. Preserve the [CITATION] marker at the same position

Original Claim: "{claim}"

Surrounding Context: "{context}"

Evidence:
{evidence}

Return ONLY the rewritten claim text, with [CITATION] preserved. No explanation."""


async def verify_one(
    llm: TogetherLLMClient,
    claim: str,
    context: str,
    chunks: List[Dict[str, Any]],
) -> bool:
    """Return True if the LLM judges the claim as genuinely UNCERTAIN."""
    prompt = _build_verify_prompt(claim, context, chunks)
    response = await llm.agenerate(prompt)
    answer = response.strip().upper()
    return answer.startswith("YES")


async def rewrite_one(
    llm: TogetherLLMClient,
    claim: str,
    context: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """Have the LLM rewrite the claim to be UNCERTAIN."""
    prompt = _build_rewrite_prompt(claim, context, chunks)
    response = await llm.agenerate(prompt)
    result = response.strip().strip("\"'")
    if "[CITATION]" not in result:
        result = result.rstrip(". ") + " [CITATION]."
    return result


async def async_main(args: argparse.Namespace) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not isinstance(instances, list):
        raise ValueError("Input must be a list of instances.")

    target_label = args.target_alignment

    # Collect (original_index, instance) for target instances
    target_indices = [(i, inst) for i, inst in enumerate(instances) if inst.get("true_outputs", {}).get("true_alignment") == target_label]

    logger.info(f"Total instances: {len(instances)} | target alignment=={target_label}: {len(target_indices)}")

    llm = TogetherLLMClient(
        model=args.llm_model,
        temperature=args.temperature,
    )

    sem = asyncio.Semaphore(args.concurrency)
    rewritten_count = 0

    async def process_one(orig_idx: int, seq: int, inst: Dict[str, Any]) -> None:
        nonlocal rewritten_count
        async with sem:
            claim = inst.get("claim_text", "")
            context = inst.get("surrounding_context", "")
            chunks = inst.get("retrieved_evidences", {}).get("extractive_chunks", [])

            if not chunks:
                logger.info(f"[{seq}/{len(target_indices)}] No chunks, keeping as UNCERTAIN")
                return

            is_uncertain = await verify_one(llm, claim, context, chunks)
            if is_uncertain:
                logger.info(f"[{seq}/{len(target_indices)}] Already UNCERTAIN, keeping")
                return

            logger.info(f"[{seq}/{len(target_indices)}] NOT UNCERTAIN, rewriting...")
            new_claim = await rewrite_one(llm, claim, context, chunks)
            instances[orig_idx]["claim_text"] = new_claim
            instances[orig_idx]["true_outputs"]["true_alignment"] = target_label
            rewritten_count += 1

    tasks = [process_one(orig_idx, seq + 1, inst) for seq, (orig_idx, inst) in enumerate(target_indices)]
    await asyncio.gather(*tasks)

    logger.info(f"Rewritten: {rewritten_count}/{len(target_indices)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(instances, f, ensure_ascii=False, indent=2)

    logger.info(f"Written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and rewrite instances where true_alignment==3 to ensure they are genuinely UNCERTAIN."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-alignment", type=int, default=3, help="Alignment value to target (default: 3)")
    parser.add_argument("--llm-model", default=TOGETHER_MODEL_OPTIONS[0])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
