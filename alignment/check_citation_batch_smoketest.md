# `check_citation_batch` Smoke Test

This repo now includes a small smoke test for the batch path in [alignment/claimcheck.py](alignment/claimcheck.py).

The batch API expects a list of items shaped like this:

```python
{
  "pair_id": "optional-id",
  "citation": "citation text to verify",
  "reference_text": "full reference document text",
  "metadata": "optional reference metadata"
}
```

## What The Smoke Test Covers

The script exercises `ReferenceChecker.check_citation_batch_async(...)`, which is what the synchronous `check_citation_batch(...)` wrapper uses internally.

Two modes are supported:

* `mock` mode validates batch orchestration, result shape, and metadata propagation without calling external APIs.
* `live` mode runs the real retrieval and LLM pipeline against OpenAI.

## Environment Setup

Use the same environment that the rest of the alignment code expects, then install the dependencies from [requirements.txt](requirements.txt).

Required for live mode:

* `OPENAI_API_KEY` for the async LLM path in [alignment/claimcheck.py](alignment/claimcheck.py)

Optional:

* `EMBEDDING_API_KEY` if you switch embedding provider to `endpoint`
* `ANTHROPIC_API_KEY` if you use `llm_provider="claude"`
* `GEMINI_API_KEY` if you use `llm_provider="gemini"`

For the default smoke test, embeddings stay local, so no embedding API key is needed.

## How To Run

From the repository root:

```bash
python alignment/smoke_test_check_citation_batch.py
```

That runs the mock batch test and should complete without any network access.

To run the live path:

```bash
python alignment/smoke_test_check_citation_batch.py --mode live
```

If you want to save retrieval chunks in live mode:

```bash
python alignment/smoke_test_check_citation_batch.py --mode live --save-chunks
```

## What To Verify

The smoke test should confirm that:

* the batch call returns one result per input item
* `batch_index` is attached to each result
* `pair_id` is preserved in output metadata
* each item has the expected `citation_text`, `classification`, `reasoning`, `evidence`, and `metadata` fields

## Notes

The live path is slower because it runs the full retrieval pipeline for each citation/reference pair, including chunking, dense retrieval, BM25 retrieval, reranking, and LLM-based support analysis.

If you only want to validate the batch orchestration code, keep using the default `mock` mode.