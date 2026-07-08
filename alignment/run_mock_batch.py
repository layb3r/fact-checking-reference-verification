"""
Batch evidence retrieval on a dataset JSON.

Usage:
    python alignment/run_mock_batch.py --max-instances 50 --concurrency 2
    python alignment/run_mock_batch.py --max-instances 10 --example
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_retriever import (
    EvidenceRetriever,
    batch_retrieve_evidence_sync,
    RetrievedEvidence,
)


def load_instances(path: Path, max_instances: int = 0):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data.get("instances", [])
    if not instances:
        raise ValueError(f"No 'instances' key found in {path}")

    print(f"Loaded {len(instances)} instances from {path}")
    if max_instances and max_instances > 0:
        instances = instances[:max_instances]
        print(f"Limited to first {max_instances} instances")

    return instances, data


def build_index(retriever: EvidenceRetriever, instances: list[dict], pdf_dir: Path) -> dict[str, str]:
    """Index each unique PDF once. Returns {pdf_rel_path -> ref_id}."""
    unique_refs: OrderedDict[str, str] = OrderedDict()

    for inst in instances:
        pdf_rel = inst.get("existence_retrieval", {}).get("pdf_path")
        if not pdf_rel:
            continue
        if pdf_rel not in unique_refs:
            stem = Path(pdf_rel).stem
            unique_refs[pdf_rel] = stem

    print(f"Found {len(unique_refs)} unique PDFs to index")

    for pdf_rel, ref_id in unique_refs.items():
        pdf_abs = pdf_dir / pdf_rel
        if not pdf_abs.exists():
            print(f"  [SKIP] PDF not found: {pdf_abs}")
            continue
        print(f"  Indexing ref_id={ref_id}  <-  {pdf_rel}")
        retriever.index_reference(ref_id=ref_id, pdf_path=str(pdf_abs))

    return unique_refs


def build_batch_pairs(
    instances: list[dict], unique_refs: dict[str, str], use_hyde: bool
) -> list[dict]:
    pairs = []
    skipped = 0

    for inst in instances:
        pdf_rel = inst.get("existence_retrieval", {}).get("pdf_path")
        if not pdf_rel or pdf_rel not in unique_refs:
            skipped += 1
            continue
        pairs.append({
            "claim": inst["claim_text"],
            "ref_id": unique_refs[pdf_rel],
            "use_hyde": use_hyde,
        })

    if skipped:
        print(f"Skipped {skipped} instances with unresolvable PDFs")
    print(f"Built {len(pairs)} batch pairs")
    return pairs


def run_batch(
    pairs: list[dict],
    retriever: EvidenceRetriever,
    concurrency: int,
    use_hyde: bool,
) -> list[RetrievedEvidence]:
    t0 = time.time()
    results = batch_retrieve_evidence_sync(
        pairs=pairs,
        retriever=retriever,
        use_hyde=use_hyde,
        max_workers=concurrency,
    )
    elapsed = time.time() - t0
    print(f"Batch completed in {elapsed:.1f}s ({elapsed / max(len(pairs), 1):.2f}s per instance)")
    return results


def attach_chunks_to_instances(
    instances: list[dict],
    batch_pairs: list[dict],
    results: list[RetrievedEvidence],
) -> list[dict]:
    """Build an index from claim+ref_id → RetrievedEvidence, then attach chunks."""
    lookup: dict[tuple[str, str], RetrievedEvidence] = {}
    for pair, r in zip(batch_pairs, results):
        lookup[(pair["claim"], pair["ref_id"])] = r

    enriched = []
    for inst in instances:
        pdf_rel = inst.get("existence_retrieval", {}).get("pdf_path")
        claim = inst.get("claim_text", "")
        ref_id = Path(pdf_rel).stem if pdf_rel else ""

        r = lookup.get((claim, ref_id))
        inst["related_chunks"] = [
            {
                "text": c["text"],
                "location": c["location"],
                "rerank_score": float(c["rerank_score"]) if c.get("rerank_score") is not None else None,
            }
            for c in r.chunks
        ] if r else []

        enriched.append(inst)

    return enriched


def show_example(results: list[RetrievedEvidence], n: int = 5):
    print(f"\n--- Sample results (first {n}) ---")
    for i, r in enumerate(results[:n]):
        print(f"[{i}] ref_id={r.ref_id} | chunks={r.num_chunks_retrieved} | time={r.query_time:.2f}s")
        for j, c in enumerate(r.chunks):
            score = c.get("rerank_score", "N/A")
            preview = c["text"][:100].replace("\n", " ")
            print(f"    chunk[{j}] score={score}  {preview}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evidence retrieval on a dataset JSON")
    parser.add_argument("--input-json", default="data_generation/mock.json",
                        help="Path to input JSON with instances (default: data_generation/mock.json)")
    parser.add_argument("--pdf-dir", default="alignment/data/mock",
                        help="Directory containing PDF files (default: data/mock)")
    parser.add_argument("--output-json", default=None,
                        help="Output path for enriched JSON (default: derived from input name)")
    parser.add_argument("--max-instances", type=int, default=10, help="Limit number of instances (0 = all)")
    parser.add_argument("--concurrency", type=int, default=2, help="Max parallel workers")
    parser.add_argument("--use-hyde", action="store_true", default=False, help="Enable HyDE augmentation")
    parser.add_argument("--example", action="store_true", help="Only print sample results, skip writing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    pdf_dir = Path(args.pdf_dir)

    instances, raw_data = load_instances(input_path, max_instances=args.max_instances)

    retriever = EvidenceRetriever(
        llm_config={
            "provider": "together",
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "temperature": 0.7,
        } if args.use_hyde else None,
    )

    unique_refs = build_index(retriever, instances, pdf_dir=pdf_dir)
    batch_pairs = build_batch_pairs(instances, unique_refs, use_hyde=args.use_hyde)

    if not batch_pairs:
        print("No pairs to process")
        return

    results = run_batch(batch_pairs, retriever, concurrency=args.concurrency, use_hyde=args.use_hyde)

    if not args.example:
        enriched_instances = attach_chunks_to_instances(instances, batch_pairs, results)
        raw_data["instances"] = enriched_instances

        output = Path(args.output_json) if args.output_json else input_path.with_stem(input_path.stem + "_with_evidence")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        print(f"Enriched instances written to {output}")

    show_example(results, n=min(10, len(results)))


if __name__ == "__main__":
    main()
