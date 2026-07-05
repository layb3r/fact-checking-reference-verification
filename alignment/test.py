"""
Example usage of EvidenceRetriever.

Shows two workflows:
  1. Single-instance: one claim + one PDF.
  2. Batch: index unique papers by ref_id, then batch-retrieve evidence for all instances
     (multiple claims may share the same ref_id).
"""

from evidence_retriever import (
    EvidenceRetriever,
    batch_retrieve_evidence_sync,
    batch_retrieve_evidence_async,
    retrieve_evidence,
    RetrievedEvidence,
)


def example_single_instance() -> None:
    """Process one claim against a single PDF."""
    print("=" * 60)
    print("EXAMPLE 1: Single instance")
    print("=" * 60)

    claim = (
        "Solver Details. The solvers used in our experiments are the diffusion-dedicated "
        "DDIM [CITATION], the second-order multistep DPM-Solver++ <cit.>, and the learned "
        "solvers BNS-Solver <cit.> and DS-Solver <cit.>."
    )
    pdf_path = "./ddim.pdf"

    result = retrieve_evidence(
        claim=claim,
        pdf_path=pdf_path,
        ref_id="ddim_paper",
        use_hyde=True,
        llm_config={
            "provider": "together",
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "temperature": 0.7,
        },
        max_chunks=3,
    )

    print(f"Claim:    {result.claim}")
    print(f"Chunks:   {len(result.chunks)}")
    for i, c in enumerate(result.chunks):
        print(f"  [{i}] score={c.get('rerank_score', 'N/A'):.4f}  text={c['text'][:120]}...")
    print()


def example_batch_index_then_retrieve_sync() -> None:
    """
    Batch workflow (sync ThreadPoolExecutor):
      1. Collect instances (some sharing the same PDF).
      2. Index each unique PDF once with a stable ref_id.
      3. Batch-retrieve evidence for every instance.
    """
    print("=" * 60)
    print("EXAMPLE 2: Batch – index then retrieve (sync)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Define instances — dicts matching the benchmark format.
    #    Multiple instances can share the same pdf_path -> same ref_id.
    # ------------------------------------------------------------------
    instances = [
        {
            "claim_text": "Solver Details. The solvers used in our experiments are the "
                          "diffusion-dedicated DDIM [CITATION].",
            "surrounding_context": "We evaluate on CIFAR-10 and ImageNet.",
            "title": "Denoising Diffusion Implicit Models",
            "pdf_path": "./ddim.pdf",
        },
        {
            "claim_text": "DDIM [CITATION] enables faster sampling by using a non-Markovian "
                          "forward process.",
            "surrounding_context": "Previous diffusion models require thousands of steps.",
            "title": "Denoising Diffusion Implicit Models",
            "pdf_path": "./ddim.pdf",
        },
        {
            "claim_text": "The second-order multistep DPM-Solver++ [CITATION] improves "
                          "sample quality over the original DPM-Solver.",
            "surrounding_context": "We compare ODE solvers on several benchmarks.",
            "title": "DPM-Solver++: Fast Solver for Guided Sampling",
            "pdf_path": "./dpm_solver.pdf",
        },
    ]

    # 2. Build a mapping: unique pdf_path -> ref_id (safe filename stem).
    import os
    from collections import OrderedDict

    unique_refs: OrderedDict[str, str] = OrderedDict()
    for inst in instances:
        pdf = inst["pdf_path"]
        if pdf not in unique_refs:
            ref_id = os.path.splitext(os.path.basename(pdf))[0]
            unique_refs[pdf] = ref_id

    # 3. Create a single retriever and index every unique PDF once.
    retriever = EvidenceRetriever(
        llm_config={
            "provider": "together",
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "temperature": 0.7,
        },
    )

    print("Indexing unique papers ...")
    for pdf_path, ref_id in unique_refs.items():
        print(f"  ref_id={ref_id}  <-  {pdf_path}")
        retriever.index_reference(ref_id=ref_id, pdf_path=pdf_path)
    print()

    # 4. Prepare batch pairs — each pair needs at least 'claim' and 'ref_id'.
    batch_pairs = []
    for inst in instances:
        batch_pairs.append({
            "claim": inst["claim_text"],
            "ref_id": unique_refs[inst["pdf_path"]],
            "use_hyde": True,
        })

    # 5. Batch-retrieve.
    results: list[RetrievedEvidence] = batch_retrieve_evidence_sync(
        pairs=batch_pairs,
        retriever=retriever,
        use_hyde=True,
        max_chunks=3,
    )

    # 6. Print results.
    for i, r in enumerate(results):
        print(f"--- Instance {i} (ref_id={r.ref_id}) ---")
        print(f"  Claim: {r.claim[:80]}...")
        print(f"  Query time: {r.query_time:.2f}s")
        print(f"  Chunks retrieved: {r.num_chunks_retrieved}")
        for j, c in enumerate(r.chunks):
            score = c.get("rerank_score", "N/A")
            preview = c["text"][:100].replace("\n", " ")
            print(f"    [{j}] score={score}  {preview}...")
    print()


async def example_batch_index_then_retrieve_async() -> None:
    """
    Batch workflow (asyncio):
      Same index-once / query-many pattern, but uses the async batch path
      for non-blocking concurrent retrieval.
    """
    print("=" * 60)
    print("EXAMPLE 3: Batch – index then retrieve (async)")
    print("=" * 60)

    import asyncio
    import os

    instances = [
        {
            "claim_text": "DDIM [CITATION] enables faster sampling by using a non-Markovian "
                          "forward process.",
            "title": "Denoising Diffusion Implicit Models",
            "pdf_path": "./ddim.pdf",
        },
        {
            "claim_text": "The second-order multistep DPM-Solver++ [CITATION] improves "
                          "sample quality over the original DPM-Solver.",
            "title": "DPM-Solver++: Fast Solver for Guided Sampling",
            "pdf_path": "./dpm_solver.pdf",
        },
    ]

    unique_refs: dict[str, str] = {}
    for inst in instances:
        pdf = inst["pdf_path"]
        if pdf not in unique_refs:
            unique_refs[pdf] = os.path.splitext(os.path.basename(pdf))[0]

    retriever = EvidenceRetriever(
        llm_config={
            "provider": "together",
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "temperature": 0.7,
        },
    )

    print("Indexing unique papers ...")
    for pdf_path, ref_id in unique_refs.items():
        print(f"  ref_id={ref_id}  <-  {pdf_path}")
        retriever.index_reference(ref_id=ref_id, pdf_path=pdf_path)
    print()

    batch_pairs = [
        {"claim": inst["claim_text"], "ref_id": unique_refs[inst["pdf_path"]]}
        for inst in instances
    ]

    results = await batch_retrieve_evidence_async(
        pairs=batch_pairs,
        retriever=retriever,
        use_hyde=True,
        max_concurrency=2,
    )

    for i, r in enumerate(results):
        print(f"--- Instance {i} (ref_id={r.ref_id}) ---")
        print(f"  Claim:   {r.claim[:80]}...")
        print(f"  Time:    {r.query_time:.2f}s")
        print(f"  Chunks:  {r.num_chunks_retrieved}")
        for j, c in enumerate(r.chunks):
            score = c.get("rerank_score", "N/A")
            print(f"    [{j}] score={score}  text={c['text'][:100]}...")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evidence retrieval examples")
    parser.add_argument(
        "--example", type=int, default=1,
        help="Example to run: 1=single 2=batch-sync 3=batch-async  (default: 1)",
    )
    args = parser.parse_args()

    if args.example == 1:
        example_single_instance()
    elif args.example == 2:
        example_batch_index_then_retrieve_sync()
    elif args.example == 3:
        import asyncio
        asyncio.run(example_batch_index_then_retrieve_async())
    else:
        print(f"Unknown example {args.example}. Choose 1, 2, or 3.")
