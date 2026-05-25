"""
Smoke test for ReferenceChecker.check_citation_batch.

Default mode is mock so the batch orchestration can be validated without
calling external APIs. Use --mode live to run the real retrieval and LLM path.

Examples:
    python alignment/smoke_test_check_citation_batch.py
    python alignment/smoke_test_check_citation_batch.py --mode live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List

import dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALIGNMENT_DIR = Path(__file__).resolve().parent

for path in (ALIGNMENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

dotenv.load_dotenv(PROJECT_ROOT / ".env")

from claimcheck import ReferenceChecker


def build_batch() -> List[Dict[str, Any]]:
    """Create a small batch with clearly supported and unsupported claims."""
    return [
        {
            "pair_id": "paper-1",
            "citation": "The intervention improved accuracy by 25%.",
            "reference_text": (
                """Methods: The intervention was tested on 100 participants.  

                Results: Accuracy improved by 25.3% after the 6-week intervention period.\n"""
            ),
            "metadata": "Mock paper 1: intervention study",
        },
        {
            "pair_id": "paper-2",
            "citation": "The study found a 25% increase in performance after the intervention.",
            "reference_text": """
    Methods:
    The intervention was administered to 100 participants over 6 weeks.
    
    Results:
    Analysis showed that participants demonstrated a 25.3% improvement in 
    performance metrics (p < 0.001) following the 6-week intervention period.
    
    Discussion:
    The observed 25% increase in performance represents a significant improvement
    and aligns with previous studies in this domain.
    """,
            "metadata": "Mock paper 2: runtime trade-off",
        },
    ]


async def run_mock_smoke_test() -> List[Dict[str, Any]]:
    """Exercise the batch API without external network calls."""
    checker = ReferenceChecker(llm_provider="openai", embedding_provider="local")

    async def fake_check_citation_async(
        self,
        citation: str,
        reference_text: str,
        metadata: str | None = None,
        save_chunks: bool = True,
        output_dir: str = "./retrieval_output",
    ) -> Dict[str, Any]:
        _ = save_chunks, output_dir
        return {
            "citation_text": citation,
            "claim": citation,
            "classification": "SUPPORTED",
            "reasoning": {
                "summary": "Mock batch smoke test result",
                "details": [reference_text[:120]],
            },
            "evidence": [
                {
                    "text": reference_text[:120],
                    "location": {"chunk_id": 0, "source": "mock"},
                }
            ],
            "metadata": {
                "confidence_score": 0.99,
                "timestamp": "2026-05-23T00:00:00",
                "processing_time": 0.01,
                "reference_metadata": metadata,
            },
        }

    checker.check_citation_async = MethodType(fake_check_citation_async, checker)

    batch = build_batch()
    results = await checker.check_citation_batch_async(
        citation_reference_pairs=batch,
        save_chunks=False,
        max_concurrency=2,
    )

    assert len(results) == len(batch), "Batch output length mismatch"
    for index, result in enumerate(results):
        assert result["citation_text"] == batch[index]["citation"]
        assert result["metadata"]["batch_index"] == index
        assert result["metadata"]["pair_id"] == batch[index]["pair_id"]
        assert result["classification"] == "SUPPORTED"

    return results


async def run_live_smoke_test(max_concurrency: int, save_chunks: bool, output_dir: str) -> List[Dict[str, Any]]:
    """Run the real batch path with the configured provider settings."""
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for live mode.")

    checker = ReferenceChecker(llm_provider="gemini", embedding_provider="local")
    batch = build_batch()
    return await checker.check_citation_batch_async(
        citation_reference_pairs=batch,
        save_chunks=save_chunks,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test check_citation_batch.")
    parser.add_argument(
        "--mode",
        choices=("mock", "live"),
        default="mock",
        help="Mock mode validates batch orchestration without APIs; live mode calls Gemini.",
    )
    parser.add_argument("--max-concurrency", type=int, default=2, help="Concurrent tasks for live mode.")
    parser.add_argument("--save-chunks", action="store_true", help="Save retrieval chunks in live mode.")
    parser.add_argument(
        "--output-dir",
        default="./retrieval_output",
        help="Directory for retrieval chunk outputs in live mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "mock":
        results = asyncio.run(run_mock_smoke_test())
    else:
        results = asyncio.run(
            run_live_smoke_test(
                max_concurrency=args.max_concurrency,
                save_chunks=args.save_chunks,
                output_dir=args.output_dir,
            )
        )

    print(json.dumps(results, indent=2))
    print(f"\nBatch smoke test passed with {len(results)} result(s).")


if __name__ == "__main__":
    main()