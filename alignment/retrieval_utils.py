"""
Retrieval Utilities for SemanticCite

This module provides utility functions for saving and analyzing retrieval results
from BM25 and dense vector search methods.
"""

import os
import logging
from datetime import datetime
from typing import Dict, List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from security_utils import sanitize_error_message

logger = logging.getLogger(__name__)


def save_retrieval_chunks(claim: str,
                         output_dir: str = "./retrieval_output",
                         vector_store: Chroma = None,
                         documents: List[Document] = None,
                         num_initial_chunks: int = 15,
                         dense_docs: List[Document] = None,
                         bm25_docs: List[Document] = None,
                         rerank_info: List[Dict] = None) -> Dict[str, str]:
    """
    Save selected chunks from BM25, dense retrieval, and FlashRank reranking separately to text files.

    Args:
        claim: The processed citation claim to use for retrieval
        output_dir: Directory to save the output files
        vector_store: Chroma vector store for dense retrieval (optional if dense_docs provided)
        documents: List of documents for BM25 retrieval (optional if bm25_docs provided)
        num_initial_chunks: Number of chunks to retrieve from each method (ignored if docs provided)
        dense_docs: Pre-retrieved dense documents (if None, will retrieve using vector_store)
        bm25_docs: Pre-retrieved BM25 documents (if None, will retrieve using documents)
        rerank_info: FlashRank reranking information with scores and rankings

    Returns:
        Dictionary with paths to the saved files and chunk counts
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Dense retrieval - use provided docs or retrieve
    dense_chunks = []
    if dense_docs is not None:
        logger.info(f"Using pre-retrieved dense documents ({len(dense_docs)} documents)")
        for i, doc in enumerate(dense_docs):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            dense_chunks.append({
                "rank": i + 1,
                "text": doc.page_content,
                "chunk_id": chunk_id,
                "source": doc.metadata.get("source", "unknown")
            })
        if dense_docs:
            logger.info(f"Dense retrieval top chunk: chunk_id={dense_chunks[0]['chunk_id']}")
    elif vector_store is not None:
        logger.info(f"Performing dense retrieval for saving (k={num_initial_chunks})")
        try:
            dense_retriever = vector_store.as_retriever(
                search_kwargs={"k": num_initial_chunks}
            )
            retrieved_dense_docs = dense_retriever.invoke(claim)
            logger.info(f"Dense retrieval returned {len(retrieved_dense_docs)} documents")

            for i, doc in enumerate(retrieved_dense_docs):
                dense_chunks.append({
                    "rank": i + 1,
                    "text": doc.page_content,
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                })
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"Dense retrieval failed: {safe_error}")
    else:
        logger.warning("No dense documents or vector store provided, skipping dense retrieval")

    # 2. BM25 retrieval - use provided docs or retrieve
    bm25_chunks = []
    if bm25_docs is not None:
        logger.info(f"Using pre-retrieved BM25 documents ({len(bm25_docs)} documents)")
        for i, doc in enumerate(bm25_docs):
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            bm25_chunks.append({
                "rank": i + 1,
                "text": doc.page_content,
                "chunk_id": chunk_id,
                "source": doc.metadata.get("source", "unknown")
            })
        if bm25_docs:
            logger.info(f"BM25 retrieval top chunk: chunk_id={bm25_chunks[0]['chunk_id']}")
    elif documents is not None:
        logger.info(f"Performing BM25 retrieval for saving (k={num_initial_chunks})")
        try:
            bm25_retriever = BM25Retriever.from_documents(documents, k=num_initial_chunks)
            retrieved_bm25_docs = bm25_retriever.invoke(claim)
            logger.info(f"BM25 retrieval returned {len(retrieved_bm25_docs)} documents")

            for i, doc in enumerate(retrieved_bm25_docs):
                bm25_chunks.append({
                    "rank": i + 1,
                    "text": doc.page_content,
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                })
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"BM25 retrieval failed: {safe_error}")
    else:
        logger.warning("No BM25 documents or document list provided, skipping BM25 retrieval")

    # 3. Save dense chunks to file
    dense_file = os.path.join(output_dir, f"dense_chunks_{timestamp}.txt")
    with open(dense_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DENSE RETRIEVAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Claim: {claim}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total chunks retrieved: {len(dense_chunks)}\n")
        f.write("=" * 80 + "\n\n")

        for chunk in dense_chunks:
            f.write(f"Rank: {chunk['rank']}\n")
            f.write(f"Chunk ID: {chunk['chunk_id']}\n")
            f.write(f"Source: {chunk['source']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{chunk['text']}\n")
            f.write("=" * 80 + "\n\n")

    logger.info(f"Dense chunks saved to: {dense_file}")

    # 4. Save BM25 chunks to file
    bm25_file = os.path.join(output_dir, f"bm25_chunks_{timestamp}.txt")
    with open(bm25_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BM25 RETRIEVAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Claim: {claim}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total chunks retrieved: {len(bm25_chunks)}\n")
        f.write("=" * 80 + "\n\n")

        for chunk in bm25_chunks:
            f.write(f"Rank: {chunk['rank']}\n")
            f.write(f"Chunk ID: {chunk['chunk_id']}\n")
            f.write(f"Source: {chunk['source']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{chunk['text']}\n")
            f.write("=" * 80 + "\n\n")

    logger.info(f"BM25 chunks saved to: {bm25_file}")

    # 5. Save FlashRank reranked chunks to file (if provided)
    rerank_file = None
    rerank_count = 0
    logger.info(f"Checking rerank_info for saving: type={type(rerank_info)}, is_none={rerank_info is None}, length={len(rerank_info) if rerank_info else 0}")
    if rerank_info:
        rerank_file = os.path.join(output_dir, f"flashrank_reranked_{timestamp}.txt")
        rerank_count = len(rerank_info)

        with open(rerank_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FLASHRANK RERANKING RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Claim: {claim}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total chunks reranked: {len(rerank_info)}\n")
            f.write("=" * 80 + "\n\n")

            for item in rerank_info:
                doc = item['document']
                f.write(f"New Rank (after FlashRank): {item['new_rank']}\n")
                f.write(f"Original Position (before reranking): {item['original_position']}\n")
                f.write(f"FlashRank Score: {item['score']:.4f}\n")
                f.write(f"Passed Threshold: {'YES' if item['passed_threshold'] else 'NO'}\n")
                f.write(f"Included in Final Results: {'YES' if item['included_in_final'] else 'NO'}\n")
                f.write(f"Chunk ID: {doc.metadata.get('chunk_id', 'unknown')}\n")
                f.write(f"Source: {doc.metadata.get('source', 'unknown')}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{doc.page_content}\n")
                f.write("=" * 80 + "\n\n")

        logger.info(f"FlashRank reranked chunks saved to: {rerank_file}")

    # Return file paths
    result = {
        "dense_file": dense_file,
        "bm25_file": bm25_file,
        "dense_count": len(dense_chunks),
        "bm25_count": len(bm25_chunks)
    }

    if rerank_file:
        result["rerank_file"] = rerank_file
        result["rerank_count"] = rerank_count

    return result