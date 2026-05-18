"""
SemanticCite Citation Verification Module

This module provides AI-powered semantic citation verification using hybrid retrieval
(BM25 + dense vector search) and neural reranking. It analyzes full-text documents to
classify citation claims as Supported, Partially Supported, Unsupported, or Uncertain,
with detailed reasoning and evidence snippets.

Main class:
    ReferenceChecker: Core citation verification engine supporting multiple LLM and
    embedding providers (OpenAI, Claude, Gemini, local models via Ollama/SentenceTransformers)

Usage:
    conda activate cite
    streamlit run src/app.py  # Web interface
    python src/citecheck.py   # CLI mode
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path

import numpy as np
import dotenv
import pymupdf
import aiohttp

# Set persistent cache directory for FlashRank before importing
FLASHRANK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.flashrank_cache')
os.makedirs(FLASHRANK_CACHE_DIR, exist_ok=True)
os.environ['FLASHRANK_CACHE_DIR'] = FLASHRANK_CACHE_DIR

# Security utilities
from security_utils import sanitize_error_message, get_user_friendly_error, SecureLogger

# Retrieval utilities
from retrieval_utils import save_retrieval_chunks

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Import flashrank Ranker for custom cache directory
from flashrank import Ranker, RerankRequest
# Fallback for structured output parsing
try:
    from langchain.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser

# Configure logging to both file and console
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create formatter
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler with rotation (10MB max, keep 5 backup files)
log_file_path = os.path.join(LOGS_DIR, 'citecheck.log')
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging initialized - writing to {log_file_path}")

# Vector store
from langchain_chroma import Chroma

# Model providers
import litellm
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

class EndpointEmbeddings:
    """Custom embedding wrapper for generic API endpoints."""
    
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'input': texts
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return [item['embedding'] for item in result['data']]
        else:
            raise Exception(f"Embedding API error: {response.status_code} - {response.text}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]

class SentenceTransformerWrapper:
    """Wrapper to make SentenceTransformer compatible with LangChain."""

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode([text])
        return embedding[0].tolist()

def _get_flashrank_ranker(model_name: str = "ms-marco-MultiBERT-L-12") -> Ranker:
    """
    Get a FlashRank Ranker instance with custom cache directory.

    Args:
        model_name: Name of the FlashRank model to use

    Returns:
        Ranker instance configured with persistent cache directory
    """
    return Ranker(model_name=model_name, cache_dir=str(FLASHRANK_CACHE_DIR))

def _initialize_flashrank():
    """
    Initialize FlashRank and ensure the model is downloaded.

    This function attempts to create a Ranker instance which will
    trigger model download if not already cached. It catches and logs any
    initialization errors.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info(f"Initializing FlashRank with cache directory: {FLASHRANK_CACHE_DIR}")
        # Create a test instance to trigger model download
        test_ranker = _get_flashrank_ranker()
        logger.info("FlashRank initialized successfully")
        return True
    except Exception as e:
        safe_error = sanitize_error_message(e)
        logger.error(f"Failed to initialize FlashRank: {safe_error}")
        logger.error("Please ensure you have internet connectivity for model download")
        return False

def _load_metadata(path_reference_metadata: Optional[str]) -> Optional[str]:
    """
    Load and process reference metadata from file.
    
    Args:
        path_reference_metadata: Path to metadata text file
        
    Returns:
        Processed metadata text (trimmed to 3000 chars) or None
    """
    if not path_reference_metadata:
        return None
        
    try:
        # Try UTF-8 first, fallback to latin-1
        try:
            with open(path_reference_metadata, 'r', encoding='utf-8') as f:
                metadata = f.read().strip()
        except UnicodeDecodeError:
            with open(path_reference_metadata, 'r', encoding='latin-1') as f:
                metadata = f.read().strip()
        
        if not metadata:
            logger.warning(f"Metadata file is empty: {path_reference_metadata}")
            return None

        # Trim to 3000 characters (truncate from end)
        if len(metadata) > 3000:
            metadata = metadata[:3000]
            logger.info(f"Metadata trimmed to 3000 characters")

        return metadata

    except FileNotFoundError:
        logger.warning(f"Metadata file not found: {path_reference_metadata}")
        return None
    except Exception as e:
        safe_error = sanitize_error_message(e)
        logger.warning(f"Error reading metadata file {path_reference_metadata}: {safe_error}")
        return None

async def download_pdf_from_url(url: str, timeout: int = 30) -> str:
    """
    Download a PDF from URL and save to temporary file.
    
    Args:
        url: URL to download PDF from
        timeout: Download timeout in seconds
        
    Returns:
        Path to downloaded temporary PDF file
        
    Raises:
        ValueError: If URL is invalid or not a PDF
        Exception: If download fails
    """
    # Validate URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}")
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"HTTP {response.status}: Failed to download from {url}")
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                    logger.warning(f"Content-Type is '{content_type}', but proceeding with download")

                # Write content to temporary file
                with open(temp_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

                logger.info(f"Successfully downloaded PDF from {url} to {temp_path}")
                return temp_path
                
    except Exception as e:
        # Clean up temporary file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        # Sanitize and raise user-friendly error
        user_error = get_user_friendly_error(e, "PDF download")
        safe_error = sanitize_error_message(e)
        logger.error(f"Failed to download PDF from {url}: {safe_error}")
        raise Exception(f"Failed to download PDF: {user_error}")

def setup_argparse():
    parser = argparse.ArgumentParser(
        description='Process citation and reference text.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--citation', '-c',
        type=str,
        help='Citation text to analyze'
    )
    
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Path to reference file'
    )
    
    return parser

class ReferenceChecker:
    """A system for checking citation accuracy against reference documents."""
    
    def __init__(self, 
                llm_provider: str = "openai",
                llm_config: Dict = None,
                embedding_provider: str = "openai",
                embedding_config: Dict = None):
        """
        Initialize the reference checker with flexible provider selection.
        
        Args:
            llm_provider: Provider for LLM ('openai' or 'nvidia', default: 'openai')
            llm_config: Configuration for LLM model
                For OpenAI: {
                    'model': str,              # e.g., 'gpt-4' (default)
                    'temperature': float,      # default: 0
                    'api_key': Optional[str]   # default: from env
                }
                For NVIDIA: {
                    'model': str,              # e.g., 'microsoft/phi-3-mini-4k-instruct'
                    'temperature': float,      # default: 0
                    'base_url': str,          # default: 'http://localhost:8000/v1/'
                    'api_key': Optional[str]   # default: from env
                }
            embedding_provider: Provider for embeddings ('openai' or 'nvidia', default: 'openai')
            embedding_config: Configuration for embedding model
                For OpenAI: {
                    'model': str,              # e.g., 'text-embedding-3-small'
                    'api_key': Optional[str]   # default: from env
                }
                For NVIDIA: {
                    'model': str,              # e.g., 'nvidia/nv-embedqa-e5-v5'
                    'base_url': str,          # default: 'http://localhost:8000/v1/'
                    'api_key': Optional[str]   # default: from env
                }
        """
        # Load environment variables
        dotenv.load_dotenv("../.env")
        
        # Set default configurations
        default_llm_configs = {
            'openai': {
                'model': 'gpt-4.1-mini',
                'temperature': 0.7,
                'api_key': os.getenv("OPENAI_API_KEY")
            },
            'claude': {
                'model': 'claude-sonnet-4-20250514',
                'temperature': 0.7,
                'api_key': os.getenv("ANTHROPIC_API_KEY")
            },
            'gemini': {
                'model': 'gemini-2.5-flash',
                'temperature': 0.7,
                'api_key': os.getenv("GEMINI_API_KEY")
            },
            'local': {
                'model': 'local/model',  # fallback for backward compatibility
                'preprocessing_model': 'ollama/SemanticCite-Refiner-Qwen3-1B',
                'classification_model': 'ollama/SemanticCite-Checker-Qwen3-4B',
                'temperature': 0.7,
                'base_url': 'http://localhost:11434',
                'api_key': 'fake-key'
            }
        }
        
        default_embedding_configs = {
            'local': {
                'model_name': 'all-mpnet-base-v2'
            },
            'openai': {
                'model': 'text-embedding-3-small',
                'api_key': os.getenv("OPENAI_API_KEY")
            },
            'endpoint': {
                'model': 'custom-embedding-model',
                'base_url': 'http://localhost:8001/v1/',
                'api_key': os.getenv("EMBEDDING_API_KEY")
            }
        }
        
        # Validate providers
        if llm_provider not in ['openai', 'claude', 'gemini', 'local']:
            raise ValueError("llm_provider must be one of: 'openai', 'claude', 'gemini', 'local'")
        if embedding_provider not in ['local', 'openai', 'endpoint']:
            raise ValueError("embedding_provider must be one of: 'local', 'openai', 'endpoint'")
        
        # Merge configurations with defaults
        llm_config = {**default_llm_configs[llm_provider], **(llm_config or {})}
        embedding_config = {**default_embedding_configs[embedding_provider], **(embedding_config or {})}
        
        # Set up embedding cache path
        if embedding_provider == 'local':
            emb_model_name = embedding_config['model_name']
        elif embedding_provider == 'openai':
            emb_model_name = os.path.basename(embedding_config['model'])
        else:  # endpoint
            emb_model_name = os.path.basename(embedding_config['model'])
        
        self.emb_persist_dir = os.path.join('./chroma_db', embedding_provider, emb_model_name)

        # Store LLM configuration for LiteLLM
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        
        # Initialize embeddings
        if embedding_provider == 'local':
            self.embeddings = SentenceTransformerWrapper(embedding_config['model_name'])
        elif embedding_provider == 'openai':
            self.embeddings = OpenAIEmbeddings(
                model=embedding_config['model'],
                openai_api_key=embedding_config['api_key']
            )
        else:  # endpoint
            # For generic endpoint, we'll use a custom embedding wrapper
            self.embeddings = EndpointEmbeddings(
                model=embedding_config['model'],
                base_url=embedding_config['base_url'],
                api_key=embedding_config['api_key']
            )
        
        # Configure text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
    
    def _llm_completion(self, prompt: str, model_type: str = "default") -> str:
        """
        Call LiteLLM completion with the configured provider and model.
        
        Args:
            prompt: The prompt to send to the LLM
            model_type: Type of model for local provider ("preprocessing", "classification", or "default")
            
        Returns:
            The LLM response as a string
        """
        model_name = self._get_llm_model_name(model_type)
        
        # Set API key for the provider
        if self.llm_provider == 'openai':
            os.environ["OPENAI_API_KEY"] = self.llm_config['api_key']
        elif self.llm_provider == 'claude':
            os.environ["ANTHROPIC_API_KEY"] = self.llm_config['api_key']
        elif self.llm_provider == 'gemini':
            os.environ["GEMINI_API_KEY"] = self.llm_config['api_key']
        elif self.llm_provider == 'local':
            # For local models, don't modify global settings
            pass
        
        try:
            # For local models, pass base_url directly to completion call
            if self.llm_provider == 'local':
                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config['temperature'],
                    base_url=self.llm_config['base_url'],
                    timeout=120  # 2 minute timeout for local models
                )
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config['temperature']
                )
            return response.choices[0].message.content
        except Exception as e:
            # Sanitize error message before raising
            safe_error = sanitize_error_message(e)
            user_friendly_msg = get_user_friendly_error(e, "LLM API call")
            # Log sanitized full error for debugging
            logger.error(f"LLM completion error: {safe_error}")
            # Raise user-friendly error
            raise Exception(f"LLM completion error: {user_friendly_msg}")

    def _get_llm_model_name(self, model_type: str = "default") -> str:
        """Resolve the model name for the current provider and model type."""
        if self.llm_provider == 'local' and model_type in ["preprocessing", "classification"]:
            model_key = f"{model_type}_model"
            return self.llm_config.get(model_key, self.llm_config['model'])
        return self.llm_config['model']

    async def _llm_completion_async(self, prompt: str, model_type: str = "default") -> str:
        """
        Async LLM completion using aiohttp for OpenAI and a thread fallback for other providers.
        """
        if self.llm_provider != 'openai':
            return await asyncio.to_thread(self._llm_completion, prompt, model_type)

        api_key = self.llm_config.get('api_key')
        if not api_key:
            raise Exception("OpenAI API key is not configured")

        model_name = self._get_llm_model_name(model_type)
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_config['temperature']
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        raise Exception(f"OpenAI API error: {response.status} - {response_text}")

                    data = json.loads(response_text)
                    return data["choices"][0]["message"]["content"]
        except Exception as e:
            safe_error = sanitize_error_message(e)
            user_friendly_msg = get_user_friendly_error(e, "LLM API call")
            logger.error(f"Async LLM completion error: {safe_error}")
            raise Exception(f"LLM completion error: {user_friendly_msg}")

    
    def _prepare_reference(self, text: str) -> Tuple[Chroma, List[Document]]:
        """
        Prepare reference text for searching by creating a vector store using ChromaDB.

        Args:
            text: Full text of the reference document

        Returns:
            Tuple of (Chroma vector store, list of documents) for hybrid retrieval
        """
        logger.info("Preparing reference text for retrieval")

        # Split text into chunks
        texts = self.text_splitter.split_text(text)
        logger.info(f"Split reference text into {len(texts)} chunks")

        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "reference_document"
                }
            ) for i, chunk in enumerate(texts)
        ]

        # Create vector store with Chroma (in-memory to avoid contamination)
        # Use unique collection name with timestamp to ensure isolation
        import uuid
        collection_name = f"reference_chunks_{uuid.uuid4().hex[:8]}"

        try:
            logger.info(f"Creating vector store with collection: {collection_name}")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name
                # No persist_directory - use in-memory to avoid cross-document contamination
            )
            logger.info("Vector store created successfully")
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"Failed to create vector store: {safe_error}")
            raise

        return vector_store, documents
    
    def _process_citation(self, citation: str) -> str:
        """
        Process the citation text to extract and standardize the core claim.
        
        Args:
            citation: Raw citation text
            
        Returns:
            Processed citation text as a clear, quantifiable claim without reference sources
        """
        prompt = f"""
Process this citation by:
1. Removing any reference markers, author names, publication identifiers or citations (e.g., [1], (Smith, 2020), et al.)
2. Converting author-centered statements to fact-centered statements (often using passive voice)
3. Maintaining all numerical values, specific metrics, and factual details
4. Ensuring the claim stands alone as a verifiable statement

Original citation:
"{citation}"

Return only the processed claim without any explanation or additional text.
"""
        
        response = self._llm_completion(prompt, "preprocessing")
        return response.strip()

    async def _process_citation_async(self, citation: str) -> str:
        """Async version of citation preprocessing."""
        prompt = f"""
Process this citation by:
1. Removing any reference markers, author names, publication identifiers or citations (e.g., [1], (Smith, 2020), et al.)
2. Converting author-centered statements to fact-centered statements (often using passive voice)
3. Maintaining all numerical values, specific metrics, and factual details
4. Ensuring the claim stands alone as a verifiable statement

Original citation:
"{citation}"

Return only the processed claim without any explanation or additional text.
"""

        response = await self._llm_completion_async(prompt, "preprocessing")
        return response.strip()
    

    def _get_relevant_chunks_hybrid(self, vector_store: Chroma, claim: str, documents: List[Document],
                                  num_initial_chunks: int = 15,
                                  relevance_threshold: float = 0.5,
                                  max_chunks: int = 5,
                                  return_separate_retrievals: bool = False) -> Tuple[List[Dict], Optional[List[Document]], Optional[List[Document]], Optional[List]]:
        """
        Retrieve and rerank chunks using hybrid BM25 + dense retrieval with FlashrankRerank filtering.

        Args:
            vector_store: Chroma vector store of the reference document
            claim: Citation text to check
            documents: List of documents for BM25 retrieval
            num_initial_chunks: Number of initial chunks to retrieve from each method
            relevance_threshold: Minimum relevance score (0.0-1.0) to return chunks
            max_chunks: Maximum number of chunks to return (actual count may be lower)
            return_separate_retrievals: If True, return separate BM25, dense, and reranked results

        Returns:
            Tuple of (chunks, dense_docs, bm25_docs, rerank_info) where:
            - chunks: List of dictionaries containing relevant text snippets (1-max_chunks based on threshold)
            - dense_docs: List of documents from dense retrieval (only if return_separate_retrievals=True)
            - bm25_docs: List of documents from BM25 retrieval (only if return_separate_retrievals=True)
            - rerank_info: List of dicts with reranking details (only if return_separate_retrievals=True)
        """
        logger.info("Starting hybrid retrieval (BM25 + dense vector search)")

        # 1. Dense retrieval using vector store
        try:
            logger.info(f"Performing dense retrieval (k={num_initial_chunks})")
            dense_retriever = vector_store.as_retriever(
                search_kwargs={"k": num_initial_chunks}
            )
            dense_docs = dense_retriever.invoke(claim)
            logger.info(f"Dense retrieval returned {len(dense_docs)} documents")
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"Dense retrieval failed: {safe_error}")
            dense_docs = []
            logger.warning("Continuing with BM25 only")

        # 2. BM25 sparse retrieval
        try:
            logger.info(f"Performing BM25 retrieval (k={num_initial_chunks})")
            bm25_retriever = BM25Retriever.from_documents(documents, k=num_initial_chunks)
            sparse_docs = bm25_retriever.invoke(claim)
            logger.info(f"BM25 retrieval returned {len(sparse_docs)} documents")
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"BM25 retrieval failed: {safe_error}")
            logger.warning("Falling back to dense retrieval only")
            sparse_docs = []

        # 3. Combine and deduplicate results
        all_docs = dense_docs + sparse_docs
        unique_docs = []
        seen_content = set()

        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        logger.info(f"Combined retrieval: {len(unique_docs)} unique documents from {len(all_docs)} total")

        # 4. Take top candidates for reranking (limit to avoid too many)
        candidates = unique_docs[:min(20, len(unique_docs))]
        logger.info(f"Selected {len(candidates)} candidates for reranking")

        # 5. Apply neural reranking using FlashRank with relevance filtering
        rerank_info = []
        try:
            if len(candidates) > 1:
                logger.info(f"Initializing FlashRank reranker (threshold={relevance_threshold}, top_n={max_chunks})")

                # Log first candidate (should be top from dense/BM25)
                if candidates:
                    first_chunk_preview = candidates[0].page_content[:100].replace('\n', ' ')
                    logger.info(f"First candidate before reranking (chunk_id={candidates[0].metadata.get('chunk_id')}): {first_chunk_preview}...")

                ranker = _get_flashrank_ranker()

                # Get scores from FlashRank (single source of truth)
                passages = [{"text": doc.page_content} for doc in candidates]
                rerank_request = RerankRequest(query=claim, passages=passages)
                flashrank_results = ranker.rerank(rerank_request)

                # Debug: Log raw result to understand score range
                if flashrank_results:
                    logger.info(f"FlashRank raw result sample: {flashrank_results[0]}")
                    scores = [result.get('score', 0.0) for result in flashrank_results]
                    logger.info(f"Score statistics: min={min(scores):.6f}, max={max(scores):.6f}, mean={sum(scores)/len(scores):.6f}")

                logger.info(f"FlashRank scores for all {len(flashrank_results)} candidates:")
                for i, result in enumerate(flashrank_results[:10]):  # Log top 10
                    original_rank = result.get('corpus_id', i)
                    score = result.get('score', 0.0)
                    chunk_id = candidates[original_rank].metadata.get('chunk_id', 'unknown')
                    logger.info(f"  Rank {i+1}: chunk_id={chunk_id}, score={score:.6f}, original_position={original_rank}")

                # Build reranked_docs directly from flashrank_results
                # FlashRank uses sigmoid transformation, so scores are probabilities (0-1)
                # Use adaptive threshold: 0.95 for sigmoid scores (vs 0.5 for raw logits)
                sigmoid_threshold = 0.95
                logger.info(f"Using sigmoid-adjusted threshold: {sigmoid_threshold} (FlashRank applies sigmoid to logits)")

                reranked_docs = []
                # Create a mapping from document to rerank score for later use
                doc_to_score = {}
                for i, result in enumerate(flashrank_results):
                    original_rank = result.get('corpus_id', i)
                    score = result.get('score', 0.0)

                    # Check if this document should be included in final results
                    # Use higher threshold appropriate for sigmoid probabilities
                    passes_threshold = score >= sigmoid_threshold
                    within_max = len(reranked_docs) < max_chunks
                    included_in_final = passes_threshold and within_max

                    # Add to final results if it passes threshold and we haven't hit max
                    if included_in_final:
                        doc = candidates[original_rank]
                        reranked_docs.append(doc)
                        # Store the score for this document
                        doc_to_score[id(doc)] = score

                    # Store reranking information for saving (all results, not just final)
                    if return_separate_retrievals:
                        rerank_info.append({
                            'new_rank': i + 1,
                            'original_position': original_rank,
                            'score': score,
                            'passed_threshold': passes_threshold,
                            'included_in_final': included_in_final,
                            'document': candidates[original_rank]
                        })

                logger.info(f"FlashRank reranking complete: {len(reranked_docs)} documents after filtering (threshold={relevance_threshold}, max={max_chunks})")

                # Log what made it through to final results
                if reranked_docs:
                    logger.info(f"Final chunks (passed threshold AND within top {max_chunks}):")
                    for i, doc in enumerate(reranked_docs):
                        chunk_id = doc.metadata.get('chunk_id', 'unknown')
                        logger.info(f"  Final Position {i+1}: chunk_id={chunk_id}")
            else:
                logger.warning(f"Only {len(candidates)} candidate(s), skipping reranking")
                reranked_docs = candidates
        except Exception as e:
            # Fallback to basic selection without reranking if FlashRank fails
            safe_error = sanitize_error_message(e)
            logger.error(f"FlashRank reranking failed: {safe_error}")
            logger.warning(f"Falling back to basic selection (top {max_chunks} candidates)")
            reranked_docs = candidates[:max_chunks]

        # 6. Format results - FlashRank already filtered by relevance threshold
        chunks = []
        for doc in reranked_docs:
            chunk_data = {
                "text": doc.page_content,
                "location": {
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown")
                }
            }
            # Add reranking score if available
            if 'doc_to_score' in locals() and id(doc) in doc_to_score:
                chunk_data["rerank_score"] = doc_to_score[id(doc)]
            chunks.append(chunk_data)

        logger.info(f"Returning {len(chunks)} relevant chunks")

        if return_separate_retrievals:
            return chunks, dense_docs, sparse_docs, rerank_info
        else:
            return chunks, None, None, None
        

    def _analyze_support(self, citation: str, chunks: List[Dict], metadata: Optional[str] = None) -> Tuple[str, Dict, float]:
        """
        Analyze the level of support for a citation based on relevant chunks.

        Args:
            citation: Citation text to analyze
            chunks: Relevant text chunks from the reference
            metadata: Optional reference metadata (title, abstract, etc.)

        Returns:
            Tuple of (classification, reasoning dictionary, confidence score)
        """

        # Process chunks to ensure JSON serialization
        processed_chunks = []
        for chunk in chunks:
            processed_chunk = {
                "text": chunk["text"],
                "location": chunk["location"]
            }
            processed_chunks.append(processed_chunk)

        # Create analysis prompt with optional metadata
        metadata_section = f"\nReference Document Information:\n{metadata}\n" if metadata else ""

        main_instruction = "Analyze how well the following citation is supported by the reference text snippets."
        if metadata:
            main_instruction = "Analyze how well the following citation is supported by the reference text snippets.\nUse the Reference Document Information to understand the study context."

        reasoning_instruction = ""
        if metadata:
            reasoning_instruction = "When explaining your reasoning, mention if the citation fits the overall study described in the document information.\n\n"

        # Use simpler JSON format for local models to avoid timeout issues
        if self.llm_provider == 'local':
            format_instructions = """{reasoning_instruction}Return your response as valid JSON with this exact structure:
{{
    "classification": "one of: SUPPORTED, REFUTED, NEI (means Not Enough Info)",
    "reasoning": "your detailed reasoning here",
    "confidence_score": 0.0
}}

Return ONLY valid JSON, no other text."""
        else:
            # Use StructuredOutputParser for cloud LLMs
            response_schemas = [
                ResponseSchema(name="classification",
                            description="Classification of the citation support level"),
                ResponseSchema(name="reasoning",
                            description="Dictionary containing summary and details of the analysis"),
                ResponseSchema(name="confidence_score",
                            description="Confidence score between 0.0 and 1.0")
            ]
            parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = parser.get_format_instructions()

        analysis_prompt = f"""{main_instruction}

Citation: "{citation}"{metadata_section}
Relevant Text Snippets:
{json.dumps(processed_chunks, indent=2)}

Classify the citation as one of:
SUPPORTED - Full alignment with source, complete representation
REFUTED - The source explicitly contradicts or denies the claim. The claim is deemed false based on the provided text.
NEI - The information within the source is insufficient to determine if the claim is either true or false. 

{format_instructions}"""

        # Call LiteLLM completion
        try:
            response = self._llm_completion(analysis_prompt, "classification")

            # Parse the response based on provider
            if self.llm_provider == 'local':
                # Simple JSON parsing for local models
                parsed_response = json.loads(response)
            else:
                # StructuredOutputParser for cloud LLMs
                parsed_response = parser.parse(response)

            return (
                parsed_response["classification"],
                parsed_response["reasoning"],
                float(parsed_response["confidence_score"])
            )

        except Exception as e:
            # Log sanitized error internally
            safe_error = sanitize_error_message(e)
            logger.error(f"Error in analysis: {safe_error}")

            # Generate user-friendly error message (no sensitive data)
            if "LLM completion error" in str(e):
                user_error = get_user_friendly_error(e, "LLM API call")
                error_msg = f"LLM API error: {user_error}"
            else:
                user_error = get_user_friendly_error(e, "response parsing")
                error_msg = f"Response parsing error: {user_error}"

            # Fallback response in case of error
            return (
                "UNCERTAIN",
                {
                    "summary": "Error occurred during analysis",
                    "details": [error_msg]
                },
                0.0
            )

    async def _analyze_support_async(self, citation: str, chunks: List[Dict], metadata: Optional[str] = None) -> Tuple[str, Dict, float]:
        """Async version of support analysis that preserves the same prompt and parsing logic."""

        processed_chunks = []
        for chunk in chunks:
            processed_chunk = {
                "text": chunk["text"],
                "location": chunk["location"]
            }
            processed_chunks.append(processed_chunk)

        metadata_section = f"\nReference Document Information:\n{metadata}\n" if metadata else ""

        main_instruction = "Analyze how well the following citation is supported by the reference text snippets."
        if metadata:
            main_instruction = "Analyze how well the following citation is supported by the reference text snippets.\nUse the Reference Document Information to understand the study context."

        reasoning_instruction = ""
        if metadata:
            reasoning_instruction = "When explaining your reasoning, mention if the citation fits the overall study described in the document information.\n\n"

        if self.llm_provider == 'local':
            format_instructions = f"""{reasoning_instruction}Return your response as valid JSON with this exact structure:
{{
    "classification": "one of: SUPPORTED, REFUTED, NEI (means Not Enough Info)",
    "reasoning": "your detailed reasoning here",
    "confidence_score": 0.0
}}

Return ONLY valid JSON, no other text."""
        else:
            response_schemas = [
                ResponseSchema(name="classification",
                            description="Classification of the citation support level"),
                ResponseSchema(name="reasoning",
                            description="Dictionary containing summary and details of the analysis"),
                ResponseSchema(name="confidence_score",
                            description="Confidence score between 0.0 and 1.0")
            ]
            parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = parser.get_format_instructions()

        analysis_prompt = f"""{main_instruction}

Citation: "{citation}"{metadata_section}
Relevant Text Snippets:
{json.dumps(processed_chunks, indent=2)}

Classify the citation as one of:
SUPPORTED - Full alignment with source, complete representation
REFUTED - The source explicitly contradicts or denies the claim. The claim is deemed false based on the provided text.
NEI - The information within the source is insufficient to determine if the claim is either true or false. 

{format_instructions}"""

        try:
            response = await self._llm_completion_async(analysis_prompt, "classification")

            if self.llm_provider == 'local':
                parsed_response = json.loads(response)
            else:
                parsed_response = parser.parse(response)

            return (
                parsed_response["classification"],
                parsed_response["reasoning"],
                float(parsed_response["confidence_score"])
            )

        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"Error in async analysis: {safe_error}")

            if "LLM completion error" in str(e):
                user_error = get_user_friendly_error(e, "LLM API call")
                error_msg = f"LLM API error: {user_error}"
            else:
                user_error = get_user_friendly_error(e, "response parsing")
                error_msg = f"Response parsing error: {user_error}"

            return (
                "UNCERTAIN",
                {
                    "summary": "Error occurred during analysis",
                    "details": [error_msg]
                },
                0.0
            )

    async def check_citation_async(
        self,
        citation: str,
        reference_text: str,
        metadata: Optional[str] = None,
        save_chunks: bool = True,
        output_dir: str = "./retrieval_output"
    ) -> Dict:
        """Async citation check that keeps one prompt per citation but allows concurrent batch execution."""
        logger.info("=" * 80)
        logger.info("Starting async citation check")
        logger.info(f"Citation: {citation[:100]}{'...' if len(citation) > 100 else ''}")
        logger.info(f"Reference text length: {len(reference_text)} characters")
        if metadata:
            logger.info(f"Metadata provided: {len(metadata)} characters")

        start_time = time.time()

        vector_store, documents = await asyncio.to_thread(self._prepare_reference, reference_text)
        logger.info("Processing citation to extract core claim")
        claim = await self._process_citation_async(citation)
        logger.info(f"Processed claim: {claim}")

        chunks, dense_docs, bm25_docs, rerank_info = await asyncio.to_thread(
            self._get_relevant_chunks_hybrid,
            vector_store,
            claim,
            documents,
            15,
            0.5,
            5,
            save_chunks
        )

        saved_files = None
        if save_chunks and (dense_docs is not None or bm25_docs is not None):
            logger.info("Saving BM25, dense, and FlashRank reranked chunks to separate files")
            logger.info(f"Rerank info available: {rerank_info is not None}, Rerank info length: {len(rerank_info) if rerank_info else 0}")
            try:
                saved_files = await asyncio.to_thread(
                    save_retrieval_chunks,
                    claim=claim,
                    output_dir=output_dir,
                    dense_docs=dense_docs,
                    bm25_docs=bm25_docs,
                    rerank_info=rerank_info
                )
                logger.info(f"Chunks saved: Dense={saved_files['dense_count']} to {saved_files['dense_file']}")
                logger.info(f"Chunks saved: BM25={saved_files['bm25_count']} to {saved_files['bm25_file']}")
                if 'rerank_file' in saved_files:
                    logger.info(f"Chunks saved: Reranked={saved_files['rerank_count']} to {saved_files['rerank_file']}")
                else:
                    logger.warning("No rerank file was created - rerank_info may be empty or None")
            except Exception as e:
                safe_error = sanitize_error_message(e)
                logger.error(f"Failed to save retrieval chunks: {safe_error}")

        logger.info("Analyzing citation support using async LLM call")
        classification, reasoning, confidence = await self._analyze_support_async(claim, chunks, metadata)
        logger.info(f"Classification: {classification} (confidence: {confidence:.2f})")

        processing_time = time.time() - start_time
        logger.info(f"Citation check completed in {processing_time:.2f} seconds")

        result = {
            "citation_text": citation,
            "claim": claim,
            "classification": classification,
            "reasoning": reasoning,
            "evidence": chunks,
            "metadata": {
                "confidence_score": confidence,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "reference_metadata": metadata,
                "saved_chunks": saved_files
            }
        }

        self.chunks = chunks

        try:
            await asyncio.to_thread(vector_store.delete_collection)
            logger.info("Vector store cleaned up successfully")
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.warning(f"Vector store cleanup failed (non-critical): {safe_error}")

        logger.info("=" * 80)
        return result

    async def check_citation_batch_async(
        self,
        citation_reference_pairs: List[Dict[str, Any]],
        save_chunks: bool = False,
        output_dir: str = "./retrieval_output",
        max_concurrency: Optional[int] = None
    ) -> List[Dict]:
        """Process citation/reference pairs concurrently with asyncio.gather."""

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None

        async def _run_one(index: int, pair: Dict[str, Any]) -> Dict:
            citation = pair.get("citation")
            reference_text = pair.get("reference_text")
            metadata = pair.get("metadata")

            if citation is None or reference_text is None:
                raise ValueError("Each batch item must include 'citation' and 'reference_text'.")

            try:
                if semaphore:
                    async with semaphore:
                        result = await self.check_citation_async(
                            citation=citation,
                            reference_text=reference_text,
                            metadata=metadata,
                            save_chunks=save_chunks,
                            output_dir=output_dir,
                        )
                else:
                    result = await self.check_citation_async(
                        citation=citation,
                        reference_text=reference_text,
                        metadata=metadata,
                        save_chunks=save_chunks,
                        output_dir=output_dir,
                    )

                result.setdefault("metadata", {})["batch_index"] = index
                if "pair_id" in pair:
                    result["metadata"]["pair_id"] = pair["pair_id"]
                return result
            except Exception as e:
                safe_error = sanitize_error_message(e)
                logger.error(f"Batch item {index} failed: {safe_error}")
                return {
                    "citation_text": citation,
                    "claim": None,
                    "classification": "ERROR",
                    "reasoning": {
                        "summary": "Batch item failed",
                        "details": [get_user_friendly_error(e, "batch citation processing")]
                    },
                    "evidence": [],
                    "metadata": {
                        "batch_index": index,
                        "pair_id": pair.get("pair_id"),
                        "processing_time": 0.0,
                        "error": True
                    }
                }

        tasks = [asyncio.create_task(_run_one(index, pair)) for index, pair in enumerate(citation_reference_pairs)]
        return await asyncio.gather(*tasks)

    def check_citation_batch(
        self,
        citation_reference_pairs: List[Dict[str, Any]],
        save_chunks: bool = False,
        output_dir: str = "./retrieval_output",
        max_concurrency: Optional[int] = None
    ) -> List[Dict]:
        """Synchronous wrapper for the async batch API."""
        return asyncio.run(self.check_citation_batch_async(
            citation_reference_pairs=citation_reference_pairs,
            save_chunks=save_chunks,
            output_dir=output_dir,
            max_concurrency=max_concurrency,
        ))

    def check_citation(self,
                      citation: str,
                      reference_text: str,
                      metadata: Optional[str] = None,
                      save_chunks: bool = True,
                      output_dir: str = "./retrieval_output") -> Dict:
        """
        Check a citation against a reference document.

        Args:
            citation: The citation text to check
            reference_text: The full text of the reference document
            metadata: Optional reference metadata (title, abstract, etc.)
            save_chunks: Whether to save BM25 and dense retrieval chunks to separate files
            output_dir: Directory to save chunk files (only used if save_chunks=True)

        Returns:
            Dictionary containing the analysis results
        """
        logger.info("=" * 80)
        logger.info("Starting citation check")
        logger.info(f"Citation: {citation[:100]}{'...' if len(citation) > 100 else ''}")
        logger.info(f"Reference text length: {len(reference_text)} characters")
        if metadata:
            logger.info(f"Metadata provided: {len(metadata)} characters")

        start_time = time.time()

        # Prepare reference text
        vector_store, documents = self._prepare_reference(reference_text)

        # Process citation into concise claim
        logger.info("Processing citation to extract core claim")
        claim = self._process_citation(citation)
        logger.info(f"Processed claim: {claim}")

        # Get relevant chunks using hybrid retrieval with relevance filtering
        # Request separate retrievals if we need to save them
        chunks, dense_docs, bm25_docs, rerank_info = self._get_relevant_chunks_hybrid(
            vector_store, claim, documents,
            relevance_threshold=0.5,
            max_chunks=5,
            return_separate_retrievals=save_chunks
        )

        # Save retrieval chunks if requested (using the actual retrieved chunks)
        saved_files = None
        if save_chunks and (dense_docs is not None or bm25_docs is not None):
            logger.info("Saving BM25, dense, and FlashRank reranked chunks to separate files")
            logger.info(f"Rerank info available: {rerank_info is not None}, Rerank info length: {len(rerank_info) if rerank_info else 0}")
            try:
                saved_files = save_retrieval_chunks(
                    claim=claim,
                    output_dir=output_dir,
                    dense_docs=dense_docs,
                    bm25_docs=bm25_docs,
                    rerank_info=rerank_info
                )
                logger.info(f"Chunks saved: Dense={saved_files['dense_count']} to {saved_files['dense_file']}")
                logger.info(f"Chunks saved: BM25={saved_files['bm25_count']} to {saved_files['bm25_file']}")
                if 'rerank_file' in saved_files:
                    logger.info(f"Chunks saved: Reranked={saved_files['rerank_count']} to {saved_files['rerank_file']}")
                else:
                    logger.warning("No rerank file was created - rerank_info may be empty or None")
            except Exception as e:
                safe_error = sanitize_error_message(e)
                logger.error(f"Failed to save retrieval chunks: {safe_error}")
                # Continue with citation checking even if saving fails

        # Analyze support
        logger.info("Analyzing citation support using LLM")
        classification, reasoning, confidence = self._analyze_support(claim, chunks, metadata)
        logger.info(f"Classification: {classification} (confidence: {confidence:.2f})")

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Citation check completed in {processing_time:.2f} seconds")

        # Construct result
        result = {
            "citation_text": citation,
            "claim": claim,
            "classification": classification,
            "reasoning": reasoning,
            "evidence": chunks,
            "metadata": {
                "confidence_score": confidence,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "reference_metadata": metadata,  # Include processed metadata
                "saved_chunks": saved_files  # Include saved file paths if chunks were saved
            }
        }

        self.chunks = chunks

        # Clean up vector store to prevent contamination across documents
        try:
            vector_store.delete_collection()
            logger.info("Vector store cleaned up successfully")
        except Exception as e:
            # Ignore cleanup errors - vector store will be garbage collected anyway
            safe_error = sanitize_error_message(e)
            logger.warning(f"Vector store cleanup failed (non-critical): {safe_error}")

        logger.info("=" * 80)
        return result
    


def check_reference(citation: str, path_reference_text: str, path_reference_metadata: Optional[str] = None) -> Dict:
    """
    Check a citation against a reference text file.
    
    This function is used directly or through the main() function.
    It reads a reference file, initializes a ReferenceChecker with default settings,
    and returns a classification result with metadata.
    
    Args:
        citation: Citation text to analyze
        path_reference_text: Path to the reference file
        path_reference_metadata: Optional path to metadata file
        
    Returns:
        Dict containing analysis results including classification, reasoning, evidence and metadata
    """
    
    # Load reference text
    with open(path_reference_text, 'r') as f:
        reference_text = f.read()
    
    # Load metadata if path provided
    metadata = _load_metadata(path_reference_metadata) if path_reference_metadata else None
    
    # Initialize checker
    checker = ReferenceChecker(llm_provider="openai", embedding_provider="local")
    
    # Check citation
    result = checker.check_citation(citation, reference_text, metadata)

    # add reference filename to metadata in result
    fname_ref = os.path.basename(path_reference_text)
    result['metadata']['reference_file'] = fname_ref
    
    return result


def check_reference_batch(
    citation_reference_pairs: List[Dict[str, Any]],
    llm_config: Optional[Dict] = None,
    embedding_config: Optional[Dict] = None,
    save_chunks: bool = False,
    output_dir: str = "./retrieval_output",
    max_concurrency: Optional[int] = None,
) -> List[Dict]:
    """Check a batch of citation/reference-text pairs concurrently."""

    if llm_config and embedding_config:
        checker = ReferenceChecker(
            llm_provider=llm_config['provider'],
            llm_config=llm_config,
            embedding_provider=embedding_config['provider'],
            embedding_config=embedding_config
        )
    else:
        checker = ReferenceChecker()

    return checker.check_citation_batch(
        citation_reference_pairs=citation_reference_pairs,
        save_chunks=save_chunks,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
    )


def test():
    """Example usage of the ReferenceChecker."""
    
    # Initialize checker
    checker = ReferenceChecker()
    
    # Example citation and reference
    citation = "The study found a 25% increase in performance after the intervention."
    reference_text = """
    Methods:
    The intervention was administered to 100 participants over 6 weeks.
    
    Results:
    Analysis showed that participants demonstrated a 25.3% improvement in 
    performance metrics (p < 0.001) following the 6-week intervention period.
    
    Discussion:
    The observed 25% increase in performance represents a significant improvement
    and aligns with previous studies in this domain.
    """
    
    result = checker.check_citation(citation, reference_text)
    print(json.dumps(result, indent=2))


def test_check_reference():
    """Test the check_reference function."""
    
    path_citation = "../data/selected_data/CLAIM_22.txt"
    path_reference_text = "../data/selected_data/TEXT_22.txt"

    with open(path_citation, 'r') as f:
        citation = f.read()
    print(f"Citation: {citation}")
    
    result = check_reference(citation, path_reference_text)
    print("Result:")
    print(result)
    # assert that the result classification is one of the supported types
    assert result['classification'] in ["SUPPORTED", 'REFUTED', 'NEI']


def main(citation, path_reference, llm_config=None, embedding_config=None, path_reference_metadata=None):
    # check if reference is a pdf
    if path_reference.endswith('.pdf'):
        try:
            logger.info(f"Reading PDF file: {path_reference}")
            pdf_document = pymupdf.open(path_reference)
        except Exception as e:
            safe_error = sanitize_error_message(e)
            logger.error(f"Failed to read pdf with exception: {safe_error}. Skipping document...")
            return None
        reference_text = ""
        for page in pdf_document:
            reference_text += page.get_text()
        pdf_document.close()
    elif path_reference.endswith('.txt') or path_reference.endswith('.md'):
        # read in reference text from text file
        with open(path_reference, 'r') as f:
            reference_text = f.read()
    else:
        raise ValueError("Reference file must be a PDF, TXT, or markdown file.")

    # Load metadata if path provided
    metadata = _load_metadata(path_reference_metadata) if path_reference_metadata else None
    
    # check the citation against the reference text
    # Initialize checker
    if llm_config and embedding_config:
        checker = ReferenceChecker(llm_provider=llm_config['provider'],
                                   llm_config=llm_config,
                                   embedding_provider=embedding_config['provider'],
                                   embedding_config=embedding_config)
    else:
        checker = ReferenceChecker()
    
    # Check citation
    result = checker.check_citation(citation, reference_text, metadata)
    # check if the result is valid
    if result['classification'] in ["SUPPORTED", 'REFUTED', 'NEI']:
        return result
    else:
        raise ValueError("Invalid classification in result.")
    

# call main function with cli arguments
if __name__ == "__main__":
    # check if the correct number of arguments are provided, if not ask for input

    parser = setup_argparse()
    args = parser.parse_args()
    # Get inputs either from arguments or user input
    citation = args.citation if args.citation else input("Enter the citation text: ")
    path_reference = args.reference if args.reference else input("Enter the path to the reference file: ")
    
    try:
        result = main(citation, path_reference)
    except Exception as e:
        safe_error = sanitize_error_message(e)
        user_error = get_user_friendly_error(e, "citation processing")
        print(f"Error processing citation: {user_error}", file=sys.stderr)
        logger.error(f"Citation processing error: {safe_error}")
        sys.exit(1)
    # print the result
    try:
        print(json.dumps(result, indent=2))
    except Exception as e:
        safe_error = sanitize_error_message(e)
        print(f"Error converting result to JSON: {safe_error}")
        print(result)

# test python -m citecheck.py 