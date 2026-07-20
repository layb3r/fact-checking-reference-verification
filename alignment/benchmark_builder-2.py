# -*- coding: utf-8 -*-
"""
Benchmark Data Builder for Citation Alignment Task.

Processes academic claims and reference documents to build
a high-quality benchmark dataset with hybrid retrieval
(Dense + Sparse BM25 → RRF → FlashRank), HyDE augmentation,
MinerU PDF extraction, and Abstractive Synthesis.

Complies with PEP-8 guidelines.
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import chromadb
import dotenv
from flashrank import Ranker, RerankRequest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Attempt to import custom modules, handle gracefully if missing during standalone run
try:
    from security_utils import sanitize_error_message
    from client import AsyncMinerUClient
except ImportError:
    def sanitize_error_message(e: Exception) -> str:
        return str(e)
    class AsyncMinerUClient:
        async def extract_markdown(self, pdf_path: str) -> str:
            return "Mock Markdown Content"

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(_PROJECT_ROOT / ".env")

FLASHRANK_CACHE_DIR = str(_PROJECT_ROOT / ".flashrank_cache")
os.makedirs(FLASHRANK_CACHE_DIR, exist_ok=True)
os.environ['FLASHRANK_CACHE_DIR'] = FLASHRANK_CACHE_DIR

LOGS_DIR = str(_PROJECT_ROOT / "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'benchmark_builder.log'),
    maxBytes=10 * 1024 * 1024,
    backupCount=5
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
    "Qwen/Qwen3.7-Plus"
]

# ==============================================================================
# Interfaces (Protocols) for dependency injection
# ==============================================================================

class AsyncLLMClient(Protocol):
    """Protocol for asynchronous LLM client."""
    async def agenerate(self, prompt: str) -> str:
        ...

class AsyncEmbeddingClient(Protocol):
    """Protocol for embedding client."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...
    def embed_query(self, text: str) -> List[float]:
        ...

# ==============================================================================
# Concrete embedding / LLM clients
# ==============================================================================

class SentenceTransformerWrapper:
    """Local SentenceTransformer embedding via the Protocol interface."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text])[0].tolist()


class OpenAIEmbeddingClient:
    """OpenAI embedding client via the Protocol interface."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from langchain_openai import OpenAIEmbeddings
        self._client = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._client.embed_query(text)


class EndpointEmbeddingClient:
    """Generic API endpoint embedding client."""

    def __init__(self, model: str, base_url: str, api_key: Optional[str] = None):
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or os.getenv("EMBEDDING_API_KEY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import requests
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json',
        }
        resp = requests.post(
            f"{self._base_url}/embeddings",
            headers=headers,
            json={'model': self._model, 'input': texts},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text}")
        return [item['embedding'] for item in resp.json()['data']]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class TogetherEmbeddingClient:
    """TogetherAI embedding client (e.g. intfloat/multilingual-e5-large-instruct)."""

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-large-instruct",
        api_key: Optional[str] = None,
    ):
        from together import Together
        self._model = model
        self._api_key = api_key or os.getenv("TOGETHER_API_KEY2")
        self._client: Optional[Any] = None

    def _lazy_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "Together embedding API key not set. Provide api_key or "
                "set the TOGETHER_API_KEY2 env var."
            )
        from together import Together
        self._client = Together(api_key=self._api_key)
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        client = self._lazy_client()
        response = client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class TogetherLLMClient:
    """TogetherAI async LLM client implementing AsyncLLMClient.

    Lazily creates the HTTP client on first use so that missing
    environment variables don't fail at instantiation time.
    """

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
                raise RuntimeError(
                    "Together API key not set. Provide api_key or set "
                    "the TOGETHER_API / TOGETHER_API_KEY env var."
                )
            from together import AsyncTogether
            self._client = AsyncTogether(api_key=self._api_key)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    @property
    def model(self) -> str:
        return self._model


def build_embedding_client(
    provider: str = "local",
    **kwargs: Any,
) -> AsyncEmbeddingClient:
    """Factory: returns an embedding client matching *provider*."""
    if provider == "local":
        return SentenceTransformerWrapper(kwargs.get("model_name", "all-mpnet-base-v2"))
    elif provider == "openai":
        return OpenAIEmbeddingClient(
            model=kwargs.get("model", "text-embedding-3-small"),
            api_key=kwargs.get("api_key"),
        )
    elif provider == "endpoint":
        return EndpointEmbeddingClient(
            model=kwargs.get("model", "custom-embedding-model"),
            base_url=kwargs.get("base_url", "http://localhost:8001/v1/"),
            api_key=kwargs.get("api_key"),
        )
    elif provider == "together":
        return TogetherEmbeddingClient(
            model=kwargs.get("model", "intfloat/multilingual-e5-large-instruct"),
            api_key=kwargs.get("api_key"),
        )
    raise ValueError(f"Unknown embedding provider: {provider}")

# ==============================================================================
# Core Builder Class
# ==============================================================================

class BenchmarkDataBuilder:
    """
    A robust builder for creating citation alignment benchmark datasets.

    Parameters
    ----------
    llm_client : AsyncLLMClient
        Async LLM client for abstractive synthesis (and optionally HyDE).
    embedding_client : AsyncEmbeddingClient
        Embedding client for dense retrieval.
    max_concurrency : int
        Max concurrent API calls when processing claims per document.
    flashrank_model : str
        FlashRank model name for neural reranking.
    use_hyde : bool
        Enable Hypothetical Document Embeddings (HyDE) augmentation before
        dense retrieval.
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        embedding_client: AsyncEmbeddingClient,
        max_concurrency: int = 5,
        flashrank_model: str = "ms-marco-MultiBERT-L-12",
        use_hyde: bool = False,
    ) -> None:
        self.llm = llm_client
        self.embeddings = embedding_client
        self.max_concurrency = max_concurrency
        self.use_hyde = use_hyde

        logger.info(f"Initializing FlashRank with model: {flashrank_model}")
        self.flashrank = Ranker(
            model_name=flashrank_model,
            cache_dir=str(FLASHRANK_CACHE_DIR),
        )

        # Configure Markdown-aware splitters (Optimized for MinerU output)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        # Increased chunk size to preserve academic context and complex formulations
        self.char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # PDF → Markdown (MinerU)
    # ------------------------------------------------------------------
    async def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert a PDF to markdown text via the MinerU CLI."""
        client = AsyncMinerUClient()
        markdown = await client.extract_markdown(pdf_path)
        logger.info(f"MinerU extracted {len(markdown)} chars from {Path(pdf_path).name}")
        return markdown

    # ------------------------------------------------------------------
    # HyDE
    # ------------------------------------------------------------------
    async def _generate_hypothetical_document(self, claim: str) -> Optional[str]:
        """Generate a hypothetical document to augment dense retrieval (HyDE)."""
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
            content = await self.llm.agenerate(prompt)
            if content and content.strip():
                logger.info("Generated hypothetical document for HyDE-style augmentation")
                return content.strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {sanitize_error_message(e)}")
        return None

    # ------------------------------------------------------------------
    # Document Indexing
    # ------------------------------------------------------------------
    def _prepare_document_index(
        self,
        doc_id: str,
        markdown_text: str,
    ) -> Tuple[chromadb.Collection, BM25Retriever, List[Document], chromadb.ClientAPI]:
        """
        Index a single document into ChromaDB and BM25.

        Returns
        -------
        Tuple of (Chroma Collection, BM25 Retriever, chunk list, Chroma Client).
        """
        # Step 1: Structural + character splitting
        md_docs = self.md_splitter.split_text(markdown_text)
        final_docs = self.char_splitter.split_documents(md_docs)

        for i, doc in enumerate(final_docs):
            doc.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"

        # Step 2: Ephemeral Vector DB
        chroma_client = chromadb.EphemeralClient()
        collection_name = f"temp_{uuid.uuid4().hex[:8]}"
        collection = chroma_client.create_collection(name=collection_name)

        texts = [doc.page_content for doc in final_docs]
        metadatas = [doc.metadata for doc in final_docs]
        ids = [doc.metadata["chunk_id"] for doc in final_docs]

        if texts:
            embedded_vectors = self.embeddings.embed_documents(texts)
            collection.add(
                documents=texts,
                embeddings=embedded_vectors,
                metadatas=metadatas,
                ids=ids,
            )

        bm25_retriever = BM25Retriever.from_documents(final_docs)

        return collection, bm25_retriever, final_docs, chroma_client

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Document],
        sparse_results: List[Document],
        k: int = 60
    ) -> List[Document]:
        """
        Fuses dense and sparse retrieval results using Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: Ranked list of documents from dense retrieval.
            sparse_results: Ranked list of documents from sparse retrieval.
            k: The RRF constant.

        Returns:
            Fused and re-ranked list of unique documents.
        """
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_results):
            chunk_id = doc.metadata["chunk_id"]
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        for rank, doc in enumerate(sparse_results):
            chunk_id = doc.metadata["chunk_id"]
            doc_map[chunk_id] = doc
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[cid] for cid in sorted_chunk_ids]

    async def _hybrid_retrieve_and_rerank(
        self,
        claim: str,
        collection: chromadb.Collection,
        bm25_retriever: BM25Retriever,
        top_k_initial: int = 15,
        top_k_final: int = 3,
        threshold: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Full retrieval pipeline: Dense + Sparse → RRF → FlashRank.
        Optionally uses HyDE augmentation before dense retrieval.
        """
        try:
            # 0. HyDE augmentation (optional)
            query_text = claim
            if self.use_hyde:
                hyde_doc = await self._generate_hypothetical_document(claim)
                if hyde_doc:
                    query_text = hyde_doc
                    logger.info("Using HyDE-augmented query for dense retrieval")

            # 1. Dense Retrieval
            query_emb = self.embeddings.embed_query(query_text)
            dense_query_res = collection.query(
                query_embeddings=[query_emb],
                n_results=top_k_initial,
            )

            dense_docs = []
            if dense_query_res.get("documents") and dense_query_res["documents"][0]:
                for text, metadata in zip(dense_query_res["documents"][0], dense_query_res["metadatas"][0]):
                    dense_docs.append(Document(page_content=text, metadata=metadata))

            # 2. Sparse Retrieval (always uses original claim)
            sparse_docs = bm25_retriever.invoke(claim)[:top_k_initial]

            # 3. RRF Fusion
            fused_candidates = self._reciprocal_rank_fusion(dense_docs, sparse_docs)

            # 4. Neural Reranking
            if not fused_candidates:
                return []
            
            # Pass metadata into FlashRank to preserve ID mapping after sorting
            passages = [
                {
                    "id": doc.metadata["chunk_id"], 
                    "text": doc.page_content, 
                    "meta": doc.metadata
                } 
                for doc in fused_candidates
            ]
            # passages = [{"text": doc.page_content} for doc in fused_candidates]
            rerank_request = RerankRequest(query=claim, passages=passages)
            reranked = await asyncio.to_thread(self.flashrank.rerank, rerank_request)

            # 5. Threshold Filtering & Formatting
            top_evidence = []
            for res in reranked[:top_k_final]:
                if res['score'] >= threshold:
                    top_evidence.append({
                        "chunk_id": res.get("chunk_id", "unknown"),
                        "extractive_text": res["text"],
                        "relevance_score": float(res['score']) if res.get("score") is not None else None,
                    })

            return top_evidence

        except Exception as e:
            logger.error(f"Retrieval error for claim '{claim[:60]}...': {sanitize_error_message(e)}")
            return []

    async def _generate_abstractive_synthesis(self, claim: str, evidences: List[Dict[str, Any]]) -> Optional[str]:
        """
        Synthesizes raw, noisy chunks into a coherent abstractive evidence block.

        Returns None when there is no evidence or the model finds nothing relevant.
        Never returns an empty string.
        """
        if not evidences:
            return None

        raw_context = "\n\n---\n\n".join(
            f"Chunk ID: {e['chunk_id']}\nText: {e['extractive_text']}"
            for e in evidences
        )

        prompt = f"""You are an expert scientific Research Assistant.
Your task is to review raw text chunks extracted from an academic paper and synthesize any contextual information relevant to the topics, entities, or methodologies mentioned in the Claim.

Claim: "{claim}"

Raw Evidence Chunks:
{raw_context}

Instructions:
1. Extract and summarize ANY information from the chunks that is topically related to the Claim.
2. Provide a neutral, objective summary (2-4 sentences) of what the chunks actually say about the topic. Do not evaluate whether the claim is true or false. Just report the facts found in the text.
3. Ignore formatting noise, markdown tags, or broken equations.
4. ONLY if the chunks discuss entirely different subjects and share ZERO entities or semantic overlap with the Claim, respond with EXACTLY: NO_EVIDENCE

Synthesis:"""

        try:
            synthesis = await self.llm.agenerate(prompt)
            cleaned = synthesis.strip()

            # if not cleaned or cleaned.upper() == "NO_EVIDENCE":
            if not cleaned or "NO_EVIDENCE" in cleaned.upper():
                return None

            # Strip leading/trailing quotes the model sometimes adds
            cleaned = cleaned.strip("\"'")

            return cleaned if cleaned else None

        except Exception as e:
            logger.error(f"Abstractive synthesis failed: {sanitize_error_message(e)}")
            return None

    async def process_dataset(
        self,
        raw_dataset_path: str,
        output_path: str,
        pdf_base_dir: str,
        use_mineru: bool = False,
        use_abstractive_synthesis: bool = False,
    ) -> None:
        """
        Main entry point to process the dataset, grouping by reference document
        to ensure O(1) indexing overhead per document.

        Parameters
        ----------
        raw_dataset_path : str
            Path to input JSON with an ``instances`` array.
        output_path : str
            Where to write the enriched benchmark JSON.
        pdf_base_dir : str
            Base directory containing reference documents (PDF or pre-extracted
            markdown).
        use_mineru : bool
            If True, run MinerU to convert PDF → markdown on the fly.
            If False, assume files are already markdown text.
        """
        logger.info(f"Loading raw dataset from {raw_dataset_path}")
        with open(raw_dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # instances = data.get("instances", [])
        instances = data if isinstance(data, list) else data.get("instances", [])

        groups = defaultdict(list)
        for inst in instances:
            if "instance_id" not in inst:
                inst["instance_id"] = str(uuid.uuid4())

            pdf_path = (
                inst.get("existence_retrieval", {}).get("pdf_path")
                or inst.get("reference_pdf_path")
                or inst.get("citation_metadata", {}).get("filepaths", [None])[0]
            )

            # pdf_path is in ["citation_metadata"]["filepaths"]
            # pdf_path = pdf_path or inst.get("citation_metadata", {}).get("filepaths", [None])[0]

            if not pdf_path:
                logger.warning(f"Instance {inst['instance_id']} has no reference PDF. Skipping.")
                continue
            groups[pdf_path].append(inst)

        processed_instances: List[Dict[str, Any]] = []

        for pdf_rel_path, group_instances in groups.items():
            pdf_abs_path = Path(pdf_base_dir) / pdf_rel_path

            if not pdf_abs_path.exists():
                logger.error(
                    f"Document not found: {pdf_abs_path}. "
                    f"Skipping {len(group_instances)} claims."
                )
                for inst in group_instances:
                    inst["retrieved_evidences"] = {
                        "extractive_chunks": [],
                        "abstractive_synthesis": None,
                    }
                    processed_instances.append(inst)
                continue

            logger.info(
                f"Processing document {pdf_abs_path.name} "
                f"for {len(group_instances)} claims."
            )

            try:
                # ---- get markdown text ----
                if use_mineru:
                    markdown_text = await self.pdf_to_markdown(str(pdf_abs_path))
                else:
                    with open(pdf_abs_path, 'r', encoding='utf-8') as f:
                        markdown_text = f.read()

                doc_id = pdf_abs_path.stem
                collection, bm25, final_docs, chroma_client = self._prepare_document_index(
                    doc_id, markdown_text
                )

                sem = asyncio.Semaphore(self.max_concurrency)

                async def process_single_claim(inst: Dict[str, Any]) -> Dict[str, Any]:
                    async with sem:
                        claim = (
                            inst.get("claim_text")
                            or inst.get("citing_context", {}).get("claim_text", "")
                        )
                        logger.info(f"Processing claim: {claim[:60]}...")
                        evidences = await self._hybrid_retrieve_and_rerank(
                            claim, collection, bm25
                        )

                        abstractive_text = None
                        if use_abstractive_synthesis:
                            abstractive_text = await self._generate_abstractive_synthesis(claim, evidences)

                        inst["retrieved_evidences"] = {
                            "extractive_chunks": evidences,
                            "abstractive_synthesis": abstractive_text,
                        }
                        return inst

                tasks = [process_single_claim(inst) for inst in group_instances]
                results = await asyncio.gather(*tasks)
                processed_instances.extend(results)

            except Exception as e:
                logger.error(
                    f"Failed to process document {pdf_abs_path.name}: "
                    f"{sanitize_error_message(e)}"
                )

            finally:
                if 'chroma_client' in locals() and 'collection' in locals():
                    try:
                        chroma_client.delete_collection(collection.name)
                        del chroma_client
                    except Exception as cleanup_err:
                        logger.warning(f"Chroma cleanup: {cleanup_err}")

        # ---- build output ----
        final_dataset = {
            "dataset_metadata": {
                "version": "2.0-hybrid-evidence",
                "creation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_instances": len(processed_instances),
                "description": (
                    "Benchmark dataset with Extractive Chunks "
                    "and Abstractive Synthesis."
                ),
            },
            "instances": processed_instances,
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully generated Benchmark Dataset at {output_path}")

# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a citation-alignment benchmark dataset with hybrid retrieval."
    )
    parser.add_argument("--input", required=True,
                        help="Path to input JSON with instances")
    parser.add_argument("--output", required=True,
                        help="Path for the output enriched benchmark JSON")
    parser.add_argument("--pdf-base-dir", default="data/papers",
                        help="Base directory containing reference documents")
    parser.add_argument("--mineru", action="store_true", default=False,
                        help="Use MinerU to convert PDF→markdown on the fly")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="Limit number of instances (0 = all)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent API calls per document")
    parser.add_argument("--embedding-provider", choices=["local", "openai", "endpoint", "together"],
                        default="local",
                        help="Embedding provider")
    parser.add_argument("--llm-model", default=TOGETHER_MODEL_OPTIONS[0],
                        help="TogetherAI model name for synthesis/HyDE")
    parser.add_argument("--hyde", action="store_true", default=False,
                        help="Enable Hypothetical Document Embeddings (HyDE)")
    parser.add_argument("--abstractive-synthesis", action="store_true", default=False,
                        help="Enable Abstractive Synthesis of retrieved evidence")
    parser.add_argument("--flashrank-model", default="ms-marco-MultiBERT-L-12",
                        help="FlashRank model for neural reranking")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    # Build clients
    llm_client = TogetherLLMClient(
        model=args.llm_model,
        temperature=0.7,
    )
    embedding_client = build_embedding_client(provider=args.embedding_provider)

    builder = BenchmarkDataBuilder(
        llm_client=llm_client,
        embedding_client=embedding_client,
        max_concurrency=args.concurrency,
        flashrank_model=args.flashrank_model,
        use_hyde=args.hyde,
    )

    # Optionally trim instances
    if args.max_instances > 0:
        import tempfile
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        # data["instances"] = data.get("instances", [])[:args.max_instances]
        data = data[:args.max_instances] if isinstance(data, list) else data
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.close()
        input_path = tmp.name
        logger.info(f"Limited to {args.max_instances} instances via temporary file")
    else:
        input_path = args.input

    await builder.process_dataset(
        raw_dataset_path=input_path,
        output_path=args.output,
        pdf_base_dir=args.pdf_base_dir,
        use_mineru=args.mineru,
        use_abstractive_synthesis=args.abstractive_synthesis
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

# python .\alignment\benchmark_builder-2.py --input data_generation\mock.json --output data_generation\mock_test.json --pdf-base-dir alignment\data\mock --mineru
