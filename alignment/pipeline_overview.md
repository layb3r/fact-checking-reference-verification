# Related Text Chunk Retrieval Pipeline

## Overview

The pipeline in `benchmark_builder-2.py` (`BenchmarkDataBuilder` class) retrieves evidence chunks from PDF reference documents to support/refute claims. It uses hybrid retrieval (dense + sparse) with fusion and neural reranking.

## Pipeline Stages

### 1. PDF Extraction (`pdf_to_markdown`)
Converts PDF to markdown text. Three methods:
- **PyMuPDF** (default) — fast, text-only extraction
- **Marker** — higher quality with layout preservation
- **MinerU** — external service for complex layouts

### 2. Document Indexing (`_prepare_document_index`)
| Step | Tool | Detail |
|------|------|--------|
| Structural split | `MarkdownHeaderTextSplitter` | Splits by `#`, `##`, `###` headers |
| Character split | `RecursiveCharacterTextSplitter` | Chunk size 750, overlap 150 |
| Dense index | ChromaDB | Embedding vectors for semantic search |
| Sparse index | BM25Retriever | Keyword-based lexical search |

### 3. Hybrid Retrieval & Reranking (`_hybrid_retrieve_and_rerank`)

```
Claim → [Optional: HyDE → LLM generates hypothetical passage]
     → Dense Retrieval (ChromaDB, top 15)
     → Sparse Retrieval (BM25, top 15)
     → RRF Fusion (combines both rank lists)
     → Neural Reranking (FlashRank)
     → Threshold Filter (score ≥ 0.85, top 3)
     → Extractive Chunks
```

- **HyDE**: If enabled, LLM generates a hypothetical document from the claim to improve dense retrieval recall.
- **Dense Retrieval**: Embeds the query (or HyDE passage) and queries ChromaDB.
- **Sparse Retrieval**: BM25 over the original claim text (complementary to dense).
- **RRF**: Reciprocal Rank Fusion combines both ranked lists into one.
- **FlashRank**: Neural cross-encoder re-scores fused candidates.

### 4. Abstractive Synthesis (optional, `_generate_abstractive_synthesis`)
LLM takes the top extractive chunks and produces a concise 2-4 sentence summary. Returns `None` if no relevant evidence found.

### Orchestration (`process_dataset`)

1. Loads input JSON containing claims
2. Groups claims by their reference PDF path
3. For each PDF: extract → index → process all its claims concurrently (bounded by `max_concurrency` semaphore)
4. Saves enriched dataset with `retrieved_evidences.extractive_chunks` and optionally `retrieved_evidences.abstractive_synthesis`
