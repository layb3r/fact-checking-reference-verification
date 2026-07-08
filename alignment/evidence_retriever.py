import os
import re
import json
import time
import uuid
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import dotenv
import chromadb

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

from sentence_transformers import SentenceTransformer

try:
    import chromadb.telemetry.product.posthog as chroma_posthog
    chroma_posthog.posthog.disabled = True
    chroma_posthog.posthog.capture = lambda *args, **kwargs: None
except Exception:
    pass

FLASHRANK_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), '.flashrank_cache'
)
os.makedirs(FLASHRANK_CACHE_DIR, exist_ok=True)
os.environ['FLASHRANK_CACHE_DIR'] = FLASHRANK_CACHE_DIR

from flashrank import Ranker, RerankRequest

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log_file_path = os.path.join(LOGS_DIR, 'evidence_retriever.log')
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

from langchain_openai import OpenAIEmbeddings
from security_utils import sanitize_error_message, get_user_friendly_error
from client import AsyncMinerUClient


TOGETHER_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-31B-it",
]

@dataclass
class RetrievedEvidence:
    chunks: List[Dict[str, Any]]
    claim: str
    query_time: float = 0.0
    num_chunks_retrieved: int = 0
    dense_docs: Optional[List[Document]] = None
    sparse_docs: Optional[List[Document]] = None
    rerank_info: Optional[List[Dict]] = None
    ref_id: Optional[str] = None


class SentenceTransformerWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()


class EndpointEmbeddings:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {'model': self.model, 'input': texts}
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        if response.status_code == 200:
            result = response.json()
            return [item['embedding'] for item in result['data']]
        raise Exception(f"Embedding API error: {response.status_code} - {response.text}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _get_flashrank_ranker(model_name: str = "ms-marco-MultiBERT-L-12") -> Ranker:
    return Ranker(model_name=model_name, cache_dir=str(FLASHRANK_CACHE_DIR))


def _reciprocal_rank_fusion(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    k: int = 60,
    max_docs: int = 15,
) -> List[Document]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        content = doc.page_content
        scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank + 1)
        doc_map[content] = doc

    for rank, doc in enumerate(sparse_docs):
        content = doc.page_content
        scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank + 1)
        doc_map[content] = doc

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [doc_map[content] for content, _score in ranked[:max_docs]]


class EvidenceRetriever:
    CHROMA_COLLECTION_NAME = "evidence_store"

    def __init__(
        self,
        embedding_provider: str = "local",
        embedding_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        llm_provider: str = "together",
        chroma_persist_dir: str = "./chroma_db",
    ):
        dotenv.load_dotenv("../.env")

        default_embedding_configs = {
            'local': {'model_name': 'all-mpnet-base-v2'},
            'openai': {
                'model': 'text-embedding-3-small',
                'api_key': os.getenv("OPENAI_API_KEY"),
            },
            'endpoint': {
                'model': 'custom-embedding-model',
                'base_url': 'http://localhost:8001/v1/',
                'api_key': os.getenv("EMBEDDING_API_KEY"),
            },
        }

        default_llm_configs = {
            'together': {
                'model': TOGETHER_MODEL_OPTIONS[0],
                'temperature': 0.7,
                'api_key': os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY"),
            }
        }

        if embedding_provider not in ('local', 'openai', 'endpoint'):
            raise ValueError("embedding_provider must be one of: 'local', 'openai', 'endpoint'")

        embedding_config = {**default_embedding_configs[embedding_provider], **(embedding_config or {})}
        llm_config = {**default_llm_configs[llm_provider], **(llm_config or {})}

        if embedding_provider == 'local':
            self.embeddings = SentenceTransformerWrapper(embedding_config['model_name'])
        elif embedding_provider == 'openai':
            self.embeddings = OpenAIEmbeddings(
                model=embedding_config['model'],
                openai_api_key=embedding_config['api_key'],
            )
        else:
            self.embeddings = EndpointEmbeddings(
                model=embedding_config['model'],
                base_url=embedding_config['base_url'],
                api_key=embedding_config['api_key'],
            )

        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header_1"),
                ("##", "header_2"),
                ("###", "header_3"),
            ]
        )

        os.makedirs(chroma_persist_dir, exist_ok=True)
        self._chroma_client = self._get_chroma_client(chroma_persist_dir)
        self._chroma_init_lock = threading.Lock()

        self._documents_cache: Dict[str, List[Document]] = {}

        self._async_together_client = None
        self._llm_config = llm_config
        if llm_config and llm_config.get('api_key'):
            from together import AsyncTogether
            self._async_together_client = AsyncTogether(api_key=llm_config['api_key'])

    @staticmethod
    def _get_chroma_client(chroma_persist_dir: str) -> "chromadb.ClientAPI":
        """Create a PersistentClient, self-healing if the on-disk DB was
        written by an incompatible (usually older) chromadb version.

        Symptom of an incompatible DB: chromadb tries to deserialize a
        collection's stored config JSON and raises `KeyError: '_type'`
        because older versions didn't write that discriminator key.
        Since this store is just a rebuildable cache (index_reference()
        re-embeds everything from source PDFs), the safe fix is to wipe
        the stale directory and start fresh rather than crash.
        """
        try:
            return chromadb.PersistentClient(
                path=chroma_persist_dir,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
        except KeyError as e:
            if "_type" not in str(e):
                raise
            logger.warning(
                f"Incompatible existing Chroma DB at '{chroma_persist_dir}' "
                f"(schema from an older chromadb version). Rebuilding it from scratch."
            )
            import shutil
            shutil.rmtree(chroma_persist_dir, ignore_errors=True)
            os.makedirs(chroma_persist_dir, exist_ok=True)
            return chromadb.PersistentClient(
                path=chroma_persist_dir,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )

    # ------------------------------------------------------------------
    # PDF -> Markdown (MinerU)
    # ------------------------------------------------------------------
    def pdf_to_markdown(self, pdf_path: str) -> str:
        import asyncio
        client = AsyncMinerUClient()
        markdown = asyncio.run(client.extract_markdown(pdf_path))
        logger.info(f"Converted PDF to markdown ({len(markdown)} chars) via MinerU CLI")
        return markdown

    # ------------------------------------------------------------------
    # Markdown -> Chunks
    # ------------------------------------------------------------------
    def split_markdown(self, markdown_text: str, ref_id: str) -> List[Document]:
        docs = self._md_splitter.split_text(markdown_text)
        for i, doc in enumerate(docs):
            doc.metadata["chunk_id"] = i
            doc.metadata["ref_id"] = ref_id
            doc.metadata["source"] = "reference_document"
        logger.info(
            f"Split markdown into {len(docs)} chunks for ref_id={ref_id}"
        )
        return docs

    # ------------------------------------------------------------------
    # Single-collection Chroma indexing
    # ------------------------------------------------------------------
    def _get_vector_store(self) -> Chroma:
        return Chroma(
            client=self._chroma_client,
            collection_name=self.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def index_reference(self, ref_id: str, pdf_path: Optional[str] = None, markdown_text: Optional[str] = None) -> List[Document]:
        if pdf_path:
            markdown_text = self.pdf_to_markdown(pdf_path)
        elif not markdown_text:
            raise ValueError("Either pdf_path or markdown_text must be provided")

        docs = self.split_markdown(markdown_text, ref_id)

        with self._chroma_init_lock:
            vector_store = self._get_vector_store()
            vector_store.add_documents(docs)

        self._documents_cache[ref_id] = docs
        logger.info(f"Indexed {len(docs)} chunks for ref_id={ref_id}")
        return docs

    def get_indexed_documents(self, ref_id: str) -> Optional[List[Document]]:
        return self._documents_cache.get(ref_id)

    # ------------------------------------------------------------------
    # Dense retrieval (single Chroma collection, metadata filter)
    # ------------------------------------------------------------------
    def _dense_retrieval(
        self,
        ref_id: str,
        claim: str,
        k: int = 15,
        use_hyde: bool = False,
    ) -> List[Document]:
        with self._chroma_init_lock:
            vector_store = self._get_vector_store()

        queries: List[Tuple[str, str]] = [("original claim", claim)]

        if use_hyde:
            try:
                hyde_doc = self.generate_hypothetical_document(claim)
                if hyde_doc and hyde_doc.strip() and hyde_doc.strip() != claim.strip():
                    queries.insert(0, ("hypothetical document", hyde_doc))
            except Exception as e:
                logger.warning(f"HyDE generation failed: {sanitize_error_message(e)}")

        seen_content: set = set()
        dense_docs: List[Document] = []

        for query_label, query_text in queries:
            logger.info(f"Dense retrieval for {query_label} (k={k}, ref_id={ref_id})")
            try:
                results = vector_store.similarity_search(
                    query_text,
                    k=k,
                    filter={"ref_id": ref_id},
                )
                for doc in results:
                    if doc.page_content not in seen_content:
                        dense_docs.append(doc)
                        seen_content.add(doc.page_content)
            except Exception as e:
                logger.error(f"Dense retrieval failed for {query_label}: {sanitize_error_message(e)}")

        logger.info(f"Dense retrieval returned {len(dense_docs)} unique docs")
        return dense_docs

    # ------------------------------------------------------------------
    # Sparse retrieval (BM25 — uses in-memory cache)
    # ------------------------------------------------------------------
    def _sparse_retrieval(self, ref_id: str, claim: str, k: int = 15) -> List[Document]:
        documents = self._documents_cache.get(ref_id)
        if not documents:
            logger.warning(f"No cached documents for ref_id={ref_id}; cannot run BM25")
            return []

        try:
            bm25 = BM25Retriever.from_documents(documents, k=k)
            results = bm25.invoke(claim)
            logger.info(f"BM25 retrieval returned {len(results)} docs for ref_id={ref_id}")
            return results
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {sanitize_error_message(e)}")
            return []

    # ------------------------------------------------------------------
    # HyDE (async via AsyncTogether)
    # ------------------------------------------------------------------
    def generate_hypothetical_document(self, claim: str) -> Optional[str]:
        if not self._async_together_client:
            return None

        prompt = (
            "Generate a short hypothetical scientific passage that could plausibly appear "
            "in a paper relevant to the claim below.\n\n"
            "Requirements:\n"
            "1. Preserve the key entities, quantities, methods, and outcomes from the claim.\n"
            "2. Write in a neutral academic style.\n"
            "3. Return only the passage text, with no bullets, labels, or explanation.\n\n"
            f'Claim:\n"{claim}"\n'
        )

        import asyncio

        try:
            model_name = self._llm_config['model']
            response = asyncio.run(
                self._async_together_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._llm_config.get('temperature', 0.7),
                )
            )
            content = response.choices[0].message.content
            if content and content.strip():
                logger.info("Generated hypothetical document for HyDE-style augmentation")
                return content.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {sanitize_error_message(e)}")

        return None

    async def _generate_hypothetical_document_async(self, claim: str) -> Optional[str]:
        if not self._async_together_client:
            return None

        prompt = (
            "Generate a short hypothetical scientific passage that could plausibly appear "
            "in a paper relevant to the claim below.\n\n"
            "Requirements:\n"
            "1. Preserve the key entities, quantities, methods, and outcomes from the claim.\n"
            "2. Write in a neutral academic style.\n"
            "3. Return only the passage text, with no bullets, labels, or explanation.\n\n"
            f'Claim:\n"{claim}"\n'
        )

        try:
            model_name = self._llm_config['model']
            response = await self._async_together_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._llm_config.get('temperature', 0.7),
            )
            content = response.choices[0].message.content
            if content and content.strip():
                logger.info("Generated hypothetical document for HyDE-style augmentation (async)")
                return content.strip()
        except Exception as e:
            logger.warning(f"Async HyDE generation failed: {sanitize_error_message(e)}")

        return None

    # ------------------------------------------------------------------
    # FlashRank reranking
    # ------------------------------------------------------------------
    def _rerank_with_flashrank(
        self,
        candidates: List[Document],
        claim: str,
        max_chunks: int = 3,
    ) -> Tuple[List[Dict], Optional[List[Dict]]]:
        rerank_info: Optional[List[Dict]] = None

        if len(candidates) <= 1:
            chunks = self._format_chunks(candidates)
            return chunks, rerank_info

        try:
            ranker = _get_flashrank_ranker()
            passages = [{"text": doc.page_content} for doc in candidates]
            flashrank_results = ranker.rerank(
                RerankRequest(query=claim, passages=passages)
            )

            sigmoid_threshold = 0.95
            reranked_docs: List[Document] = []
            doc_to_score: Dict[int, float] = {}
            rerank_info = []

            for i, result in enumerate(flashrank_results):
                original_rank = result.get('corpus_id', i)
                score = result.get('score', 0.0)
                passes = score >= sigmoid_threshold
                within_max = len(reranked_docs) < max_chunks

                if passes and within_max:
                    doc = candidates[original_rank]
                    reranked_docs.append(doc)
                    doc_to_score[id(doc)] = score

                rerank_info.append({
                    'new_rank': i + 1,
                    'original_position': original_rank,
                    'score': score,
                    'passed_threshold': passes,
                    'included_in_final': passes and within_max,
                    'document': candidates[original_rank],
                })

            logger.info(
                f"FlashRank reranking: {len(reranked_docs)} docs after threshold filter"
            )
            chunks = []
            for doc in reranked_docs:
                cd = {
                    "text": doc.page_content,
                    "location": {
                        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                        "ref_id": doc.metadata.get("ref_id", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                    },
                }
                if id(doc) in doc_to_score:
                    cd["rerank_score"] = doc_to_score[id(doc)]
                chunks.append(cd)
            return chunks, rerank_info

        except Exception as e:
            logger.error(f"FlashRank reranking failed: {sanitize_error_message(e)}")
            return self._format_chunks(candidates[:max_chunks]), None

    def _format_chunks(self, docs: List[Document]) -> List[Dict]:
        return [
            {
                "text": doc.page_content,
                "location": {
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "ref_id": doc.metadata.get("ref_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                },
            }
            for doc in docs
        ]

    # ------------------------------------------------------------------
    # Full retrieval pipeline
    # ------------------------------------------------------------------
    def get_relevant_chunks(
        self,
        ref_id: str,
        claim: str,
        use_hyde: bool = False,
        num_initial_chunks: int = 15,
        max_chunks: int = 3,
        return_separate_retrievals: bool = False,
    ) -> Tuple[List[Dict], Optional[List[Document]], Optional[List[Document]], Optional[List[Dict]]]:
        logger.info(
            f"Hybrid retrieval for ref_id={ref_id}, claim={claim[:80]}..."
        )

        dense_docs = self._dense_retrieval(
            ref_id=ref_id,
            claim=claim,
            k=num_initial_chunks,
            use_hyde=use_hyde,
        )

        sparse_docs = self._sparse_retrieval(
            ref_id=ref_id,
            claim=claim,
            k=num_initial_chunks,
        )

        unique_docs = _reciprocal_rank_fusion(
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            k=60,
            max_docs=min(15, num_initial_chunks),
        )
        logger.info(
            f"RRF selected {len(unique_docs)} unique documents"
        )

        candidates = unique_docs[:min(15, len(unique_docs))]
        chunks, rerank_info = self._rerank_with_flashrank(
            candidates=candidates,
            claim=claim,
            max_chunks=max_chunks,
        )

        logger.info(f"Returning {len(chunks)} evidence chunks")

        if return_separate_retrievals:
            return chunks, dense_docs, sparse_docs, rerank_info
        return chunks, None, None, None

    def retrieve_evidence(
        self,
        claim: str,
        ref_id: str,
        use_hyde: bool = False,
        num_initial_chunks: int = 15,
        max_chunks: int = 3,
        return_separate_retrievals: bool = False,
    ) -> RetrievedEvidence:
        start = time.time()

        chunks, dense_docs, sparse_docs, rerank_info = self.get_relevant_chunks(
            ref_id=ref_id,
            claim=claim,
            use_hyde=use_hyde,
            num_initial_chunks=num_initial_chunks,
            max_chunks=max_chunks,
            return_separate_retrievals=return_separate_retrievals,
        )

        return RetrievedEvidence(
            chunks=chunks,
            claim=claim,
            query_time=time.time() - start,
            num_chunks_retrieved=len(chunks),
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            rerank_info=rerank_info,
            ref_id=ref_id,
        )

    async def retrieve_evidence_async(
        self,
        claim: str,
        ref_id: str,
        use_hyde: bool = False,
        num_initial_chunks: int = 15,
        max_chunks: int = 3,
        return_separate_retrievals: bool = False,
    ) -> RetrievedEvidence:
        start = time.time()

        dense_docs = await asyncio.to_thread(
            self._dense_retrieval,
            ref_id=ref_id,
            claim=claim,
            k=num_initial_chunks,
            use_hyde=use_hyde,
        )

        sparse_docs = await asyncio.to_thread(
            self._sparse_retrieval,
            ref_id=ref_id,
            claim=claim,
            k=num_initial_chunks,
        )

        unique_docs = await asyncio.to_thread(
            _reciprocal_rank_fusion,
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            k=60,
            max_docs=min(15, num_initial_chunks),
        )

        candidates = unique_docs[:min(15, len(unique_docs))]
        chunks, rerank_info = await asyncio.to_thread(
            self._rerank_with_flashrank,
            candidates=candidates,
            claim=claim,
            max_chunks=max_chunks,
        )

        return RetrievedEvidence(
            chunks=chunks,
            claim=claim,
            query_time=time.time() - start,
            num_chunks_retrieved=len(chunks),
            dense_docs=dense_docs if return_separate_retrievals else None,
            sparse_docs=sparse_docs if return_separate_retrievals else None,
            rerank_info=rerank_info if return_separate_retrievals else None,
            ref_id=ref_id,
        )


# ------------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------------
async def batch_retrieve_evidence_async(
    pairs: List[Dict[str, Any]],
    retriever: Optional[EvidenceRetriever] = None,
    embedding_provider: str = "local",
    embedding_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    use_hyde: bool = False,
    num_initial_chunks: int = 15,
    max_chunks: int = 3,
    max_concurrency: Optional[int] = None,
) -> List[RetrievedEvidence]:
    import asyncio

    if not pairs:
        return []

    if retriever is None:
        retriever = EvidenceRetriever(
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            llm_config=llm_config,
        )

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None

    async def _run_one(index: int, pair: Dict[str, Any]) -> RetrievedEvidence:
        claim = pair.get("claim") or pair.get("citation", "")
        ref_id = pair.get("ref_id", str(index))

        try:
            if semaphore:
                async with semaphore:
                    return await retriever.retrieve_evidence_async(
                        claim=claim,
                        ref_id=ref_id,
                        use_hyde=pair.get("use_hyde", use_hyde),
                        num_initial_chunks=pair.get("num_initial_chunks", num_initial_chunks),
                        max_chunks=pair.get("max_chunks", max_chunks),
                    )
            else:
                return await retriever.retrieve_evidence_async(
                    claim=claim,
                    ref_id=ref_id,
                    use_hyde=pair.get("use_hyde", use_hyde),
                    num_initial_chunks=pair.get("num_initial_chunks", num_initial_chunks),
                    max_chunks=pair.get("max_chunks", max_chunks),
                )
        except Exception as e:
            logger.error(f"Batch item {index} (ref_id={ref_id}) failed: {sanitize_error_message(e)}")
            return RetrievedEvidence(
                chunks=[],
                claim=claim,
                query_time=0.0,
                num_chunks_retrieved=0,
                ref_id=ref_id,
            )

    tasks = [asyncio.create_task(_run_one(idx, pair)) for idx, pair in enumerate(pairs)]
    return await asyncio.gather(*tasks)


def batch_retrieve_evidence_sync(
    pairs: List[Dict[str, Any]],
    retriever: Optional[EvidenceRetriever] = None,
    embedding_provider: str = "local",
    embedding_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    use_hyde: bool = False,
    num_initial_chunks: int = 15,
    max_chunks: int = 3,
    max_workers: Optional[int] = None,
) -> List[RetrievedEvidence]:
    if not pairs:
        return []

    if retriever is None:
        retriever = EvidenceRetriever(
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            llm_config=llm_config,
        )

    results: List[Optional[RetrievedEvidence]] = [None] * len(pairs)

    def _process_one(index: int, pair: Dict[str, Any]) -> Tuple[int, RetrievedEvidence]:
        claim = pair.get("claim") or pair.get("citation", "")
        ref_id = pair.get("ref_id", str(index))

        evidence = retriever.retrieve_evidence(
            claim=claim,
            ref_id=ref_id,
            use_hyde=pair.get("use_hyde", use_hyde),
            num_initial_chunks=pair.get("num_initial_chunks", num_initial_chunks),
            max_chunks=pair.get("max_chunks", max_chunks),
        )
        return index, evidence

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, idx, pair): idx
            for idx, pair in enumerate(pairs)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                idx_result, evidence = future.result()
                results[idx_result] = evidence
            except Exception as e:
                logger.error(f"Batch item {idx} failed: {sanitize_error_message(e)}")
                claim = pairs[idx].get("claim") or pairs[idx].get("citation", "")
                results[idx] = RetrievedEvidence(
                    chunks=[],
                    claim=claim,
                    query_time=0.0,
                    num_chunks_retrieved=0,
                    ref_id=pairs[idx].get("ref_id", str(idx)),
                )

    return results


def retrieve_evidence(
    claim: str,
    pdf_path: Optional[str] = None,
    ref_id: str = "default",
    markdown_text: Optional[str] = None,
    embedding_provider: str = "local",
    embedding_config: Optional[Dict] = None,
    llm_config: Optional[Dict] = None,
    use_hyde: bool = False,
    num_initial_chunks: int = 15,
    max_chunks: int = 3,
) -> RetrievedEvidence:
    retriever = EvidenceRetriever(
        embedding_provider=embedding_provider,
        embedding_config=embedding_config,
        llm_config=llm_config,
    )

    if retriever.get_indexed_documents(ref_id) is None:
        retriever.index_reference(
            ref_id=ref_id,
            pdf_path=pdf_path,
            markdown_text=markdown_text,
        )

    return retriever.retrieve_evidence(
        claim=claim,
        ref_id=ref_id,
        use_hyde=use_hyde,
        num_initial_chunks=num_initial_chunks,
        max_chunks=max_chunks,
    )