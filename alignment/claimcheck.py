import os
import sys
import json
import time
import logging
import argparse
import tempfile
import asyncio
import re
import threading
from datetime import datetime
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from pathlib import Path

import numpy as np
import dotenv
import pymupdf
import aiohttp
import chromadb

from together import Together

try:
    import chromadb.telemetry.product.posthog as chroma_posthog

    chroma_posthog.posthog.disabled = True
    chroma_posthog.posthog.capture = lambda *args, **kwargs: None
except Exception:
    # Telemetry suppression is best-effort; failures here should not affect retrieval.
    pass

# Set persistent cache directory for FlashRank before importing
FLASHRANK_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.flashrank_cache')
os.makedirs(FLASHRANK_CACHE_DIR, exist_ok=True)
os.environ['FLASHRANK_CACHE_DIR'] = FLASHRANK_CACHE_DIR

from security_utils import sanitize_error_message, get_user_friendly_error, SecureLogger
from retrieval_utils import save_retrieval_chunks

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Import flashrank Ranker for custom cache directory
from flashrank import Ranker, RerankRequest

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
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

TOGETHER_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "openai/gpt-oss-20b",
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-31B-it",
]

TOGETHER_MODEL_PRICING = {
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {"input": 0.30, "output": 0.30},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
    "meta-llama/Meta-Llama-3-8B-Instruct-Lite": {"input": 0.14, "output": 0.14},
    "Qwen/Qwen3.5-9B": {"input": 0.17, "output": 0.25},
    "google/gemma-4-31B-it": {"input": 0.39, "output": 0.97},
}

_CURRENT_LLM_RUN_METRICS: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "_CURRENT_LLM_RUN_METRICS",
    default=None,
)

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


def _build_json_format_instructions(reasoning_instruction: str = "") -> str:
    """Build a provider-agnostic JSON output instruction block."""

    return f"""{reasoning_instruction}Return your response as valid JSON with this exact structure:
{{
    "classification": "one of: SUPPORTED (supported), REFUTED (contradicted), NEI (not enough info)",
    "reasoning": "your detailed reasoning here",
    "confidence_score": score of confidence
}}

Return ONLY valid JSON, no other text."""


def _parse_json_model_response(response: str) -> Dict[str, Any]:
    """Parse a JSON response, tolerating fenced or prefixed model output."""

    response_text = response.strip()
    if not response_text:
        raise ValueError("Empty model response while parsing JSON")

    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model response was not valid JSON: {response_text[:120]!r}")
        parsed_response = json.loads(match.group(0))

    classification = str(parsed_response.get("classification", "UNCERTAIN")).strip().upper()
    parsed_response["classification"] = classification
    parsed_response["reasoning"] = parsed_response.get("reasoning", "")
    parsed_response["confidence_score"] = float(parsed_response.get("confidence_score", 0.0))
    return parsed_response


def _build_citation_processing_prompt(citation: str) -> str:
    """Build the claim-extraction prompt with explicit marker semantics."""

#     return f"""
# You are extracting the exact claim that belongs to the cited reference.

# Important:
# - The token [CITATION] is a placeholder for the reference being checked.
# - Sometimes [CITATION] just refers to a term or concept introduced by the paper, not a specific claim.
# - Variants like <cit>, <cit.>, [citation], and [cite...] are not reference markers.
# - The marker can appear anywhere inside the claim, not only at the end.
# - Your job is to reconstruct the claim that the reference is asserting, not any unrelated claim from the surrounding paper text.

# Task:
# 1. Find the statement that directly corresponds to the marked reference.
# 2. If the marker splits one sentence or clause, remove the marker and rewrite the sentence so the claim is fluent and complete.
# 3. Remove nearby claims about other papers, examples, commentary, or bibliography language.
# 4. Preserve all factual content, numbers, quantities, names, methods, and outcomes that belong to the marked reference.
# 5. Return only the cleaned claim text.

# Citation:
# "{citation}"

# Return only the processed claim without any explanation or additional text.
# """
    return f"""
You are extracting the exact claim that belongs to the cited reference.

Important:
- The token [CITATION] is a placeholder for the reference being checked.
- Sometimes [CITATION] refers to a claim, result, method, dataset, concept, object, or topic.
- Variants like <cit>, <cit.>, [citation], and [cite...] are not reference markers.
- The marker can appear anywhere inside the claim.

Task:
1. Identify the content that the marked reference supports.
2. If the marker occurs inside a sentence, remove it and reconstruct a fluent statement.
3. Preserve all factual content, numbers, entities, methods, and outcomes supported by the reference.
4. Remove surrounding discussion, comparisons, commentary, or claims belonging to other references.
5. The output must be a standalone declarative statement.

Special rules:
- Do NOT introduce unsupported research-action phrases such as:
  "we study", "we examine", "the authors investigate",
  "this paper presents", "this work explores",
  unless those actions are explicitly stated in the cited text.
- If the cited span is only a noun phrase, method name, dataset name, concept, or topic,
  convert it into the shortest factual declarative statement that expresses what the reference is about.
- Prefer factual propositions over descriptions of the paper itself.
- Do not add information that cannot reasonably be inferred from the cited span.

Citation:
"{citation}"

Return only the processed claim without any explanation."""


def _normalize_claim_text(text: str) -> str:
    """Normalize whitespace and remove residual marker tokens."""

    cleaned = text or ""
    cleaned = re.sub(r"\s*\[(?:CITATION|citation|cite[^\]]*)\]\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*<\s*cit[^>]*>\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\(\s*cit(?:ation)?[^)]*\)\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" \t\r\n,.;:-\"'`")


def _extract_usage_count(usage: Any, field_name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get(field_name, 0)
    else:
        value = getattr(usage, field_name, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _estimate_together_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    llm_config: Dict[str, Any],
) -> Optional[float]:
    input_rate = llm_config.get("input_cost_per_million_tokens")
    output_rate = llm_config.get("output_cost_per_million_tokens")

    pricing = TOGETHER_MODEL_PRICING.get(model_name)
    if input_rate is None and pricing:
        input_rate = pricing.get("input")
    if output_rate is None and pricing:
        output_rate = pricing.get("output")

    if input_rate is None or output_rate is None:
        return None

    return round(
        (prompt_tokens * float(input_rate) + completion_tokens * float(output_rate)) / 1_000_000,
        6,
    )


def _summarize_llm_run_metrics(call_metrics: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    metrics = list(call_metrics or [])
    total_calls = len(metrics)

    if total_calls == 0:
        return {
            "provider": "together",
            "model": None,
            "total_calls": 0,
            "avg_latency_seconds": 0.0,
            "total_latency_seconds": 0.0,
            "avg_input_tokens_per_sample": 0.0,
            "avg_output_tokens_per_sample": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "cost_estimation_available": False,
            "calls": [],
        }

    total_latency = sum(float(item.get("latency_seconds", 0.0)) for item in metrics)
    total_input_tokens = sum(int(item.get("input_tokens", 0)) for item in metrics)
    total_output_tokens = sum(int(item.get("output_tokens", 0)) for item in metrics)
    total_tokens = sum(int(item.get("total_tokens", 0)) for item in metrics)

    estimated_costs = [item.get("estimated_cost_usd") for item in metrics if item.get("estimated_cost_usd") is not None]
    estimated_cost_usd = round(sum(float(cost) for cost in estimated_costs), 6) if estimated_costs else None

    return {
        "provider": metrics[0].get("provider", "together"),
        "model": metrics[0].get("model"),
        "total_calls": total_calls,
        "avg_latency_seconds": round(total_latency / total_calls, 4),
        "total_latency_seconds": round(total_latency, 4),
        "avg_input_tokens_per_sample": round(total_input_tokens / total_calls, 2),
        "avg_output_tokens_per_sample": round(total_output_tokens / total_calls, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_estimation_available": estimated_cost_usd is not None,
        "calls": metrics,
    }

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

# def setup_argparse():
#     parser = argparse.ArgumentParser(
#         description='Process citation and reference text.',
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )
    
#     parser.add_argument(
#         '--citation', '-c',
#         type=str,
#         help='Citation text to analyze'
#     )
    
#     parser.add_argument(
#         '--reference', '-r',
#         type=str,
#         help='Path to reference file'
#     )

#     parser.add_argument(
#         '--llm-model',
#         type=str,
#         choices=TOGETHER_MODEL_OPTIONS,
#         default=TOGETHER_MODEL_OPTIONS[0],
#         help='Together model to use for citation processing and classification'
#     )
    
#     return parser

class ReferenceChecker:
    """A system for checking citation accuracy against reference documents."""
    
    def __init__(self, 
            llm_provider: str = "together",
                llm_config: Dict = None,
                embedding_provider: str = "local",
                embedding_config: Dict = None):
        """
        Initialize the reference checker with flexible provider selection.
        
        Args:
            llm_provider: Provider for LLM ('together', default: 'together')
            llm_config: Configuration for LLM model
                For Together: {
                    'model': str,              # e.g., 'Qwen/Qwen2.5-7B-Instruct-Turbo' (default)
                    'temperature': float,      # default: 0
                    'api_key': Optional[str]   # default: from TOGETHER_API
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
            'together': {
                'model': TOGETHER_MODEL_OPTIONS[0],
                'temperature': 0.7,
                'api_key': os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY"),
                'max_retries': 3,
                'retry_backoff_seconds': 1.5,
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
        if llm_provider not in ['together']:
            raise ValueError("llm_provider must be one of: 'together'")
        if embedding_provider not in ['local', 'openai', 'endpoint']:
            raise ValueError("embedding_provider must be one of: 'local', 'openai', 'endpoint'")
        
        # Merge configurations with defaults
        llm_config = {**default_llm_configs[llm_provider], **(llm_config or {})}
        embedding_config = {**default_embedding_configs[embedding_provider], **(embedding_config or {})}

        if not llm_config.get('api_key'):
            raise ValueError("Together API key is not configured. Set TOGETHER_API or TOGETHER_API_KEY.")

        if llm_config.get('model') not in TOGETHER_MODEL_OPTIONS:
            raise ValueError(
                f"Unsupported Together model '{llm_config.get('model')}'. "
                f"Choose one of: {', '.join(TOGETHER_MODEL_OPTIONS)}"
            )
        
        # Set up embedding cache path
        if embedding_provider == 'local':
            emb_model_name = embedding_config['model_name']
        elif embedding_provider == 'openai':
            emb_model_name = os.path.basename(embedding_config['model'])
        else:  # endpoint
            emb_model_name = os.path.basename(embedding_config['model'])
        
        self.emb_persist_dir = os.path.join('./chroma_db', embedding_provider, emb_model_name)

        # Store Together configuration for completion calls
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        self._together_client = Together(api_key=self.llm_config['api_key'])
        
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

        # Chroma tenant/bootstrap can race under concurrent thread startup.
        # Serialize vector-store initialization to avoid intermittent tenant errors.
        self._chroma_init_lock = threading.Lock()
        self._retrieval_weights = {
            "dense": 0.65,
            "sparse": 0.35,
        }
        self._claim_keywords: List[str] = []

    def _append_llm_run_metric(self, metric: Dict[str, Any]) -> None:
        metrics = _CURRENT_LLM_RUN_METRICS.get()
        if metrics is not None:
            metrics.append(metric)

    def _get_current_llm_run_metrics(self) -> List[Dict[str, Any]]:
        metrics = _CURRENT_LLM_RUN_METRICS.get()
        return list(metrics or [])

    def _build_llm_run_metrics_summary(self) -> Dict[str, Any]:
        summary = _summarize_llm_run_metrics(self._get_current_llm_run_metrics())
        summary["model"] = self._get_llm_model_name()
        return summary
    
    def _llm_completion(self, prompt: str, model_type: str = "default") -> str:
        """
        Call Together completion with the configured model.
        
        Args:
            prompt: The prompt to send to the LLM
            model_type: Reserved for compatibility with older callers.
            
        Returns:
            The LLM response as a string
        """
        model_name = self._get_llm_model_name(model_type)
        
        max_attempts = int(self.llm_config.get('max_retries', 3))
        base_delay = float(self.llm_config.get('retry_backoff_seconds', 1.5))

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.perf_counter()
                request_kwargs: Dict[str, Any] = {
                    'model': model_name,
                    'messages': [{"role": "user", "content": prompt}],
                    'temperature': self.llm_config['temperature'],
                }

                if 'max_tokens' in self.llm_config and self.llm_config['max_tokens'] is not None:
                    request_kwargs['max_tokens'] = self.llm_config['max_tokens']

                response = self._together_client.chat.completions.create(**request_kwargs)
                latency_seconds = time.perf_counter() - start_time

                usage = getattr(response, 'usage', None)
                prompt_tokens = _extract_usage_count(usage, 'prompt_tokens')
                completion_tokens = _extract_usage_count(usage, 'completion_tokens')
                total_tokens = _extract_usage_count(usage, 'total_tokens')
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens

                estimated_cost_usd = _estimate_together_cost(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    llm_config=self.llm_config,
                )

                self._append_llm_run_metric({
                    "provider": self.llm_provider,
                    "model": model_name,
                    "latency_seconds": round(latency_seconds, 4),
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                })

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < max_attempts:
                        delay_seconds = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Empty model response on attempt {attempt}/{max_attempts}, "
                            f"retrying in {delay_seconds:.2f}s"
                        )
                        time.sleep(delay_seconds)
                        continue
                    raise ValueError("Empty model response after all retries")
                return content
            except Exception as e:
                safe_error = sanitize_error_message(e)
                user_friendly_msg = get_user_friendly_error(e, "Together API call")

                retryable_error = self._is_retryable_llm_error(e)
                if retryable_error and attempt < max_attempts:
                    delay_seconds = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"LLM completion retryable error on attempt {attempt}/{max_attempts}: {safe_error}. "
                        f"Retrying in {delay_seconds:.2f}s"
                    )
                    time.sleep(delay_seconds)
                    continue

                logger.error(f"LLM completion error: {safe_error}")
                raise Exception(f"Together completion error: {user_friendly_msg}")

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        """Return True for transient Together/API failures that benefit from retry."""
        error_text = str(error).lower()
        retry_markers = [
            'rate limit',
            'too many requests',
            '429',
            'quota',
            'resource exhausted',
            'server error',
            '502',
            '503',
            '504',
            'timeout',
            'temporarily unavailable',
            'connection error',
        ]
        return any(marker in error_text for marker in retry_markers)

    def _get_llm_model_name(self, model_type: str = "default") -> str:
        """Resolve the Together model name for the current run."""
        _ = model_type
        return self.llm_config['model']

    async def _llm_completion_async(self, prompt: str, model_type: str = "default") -> str:
        """
        Async Together completion using a thread fallback.
        """
        return await asyncio.to_thread(self._llm_completion, prompt, model_type)

    
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

        max_attempts = 3
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Creating vector store with collection: {collection_name} (attempt {attempt}/{max_attempts})")
                with self._chroma_init_lock:
                    chroma_client = chromadb.EphemeralClient(
                        settings=chromadb.config.Settings(anonymized_telemetry=False)
                    )
                    vector_store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=collection_name,
                        client=chroma_client
                        # Use an explicit ephemeral client to avoid cross-document contamination
                    )
                logger.info("Vector store created successfully")
                return vector_store, documents
            except Exception as e:
                last_error = e
                safe_error = sanitize_error_message(e)
                is_tenant_error = "default_tenant" in str(e).lower() or "tenant" in str(e).lower()
                if is_tenant_error and attempt < max_attempts:
                    backoff_seconds = 0.25 * attempt
                    logger.warning(
                        f"Vector store creation hit tenant initialization race (attempt {attempt}/{max_attempts}): {safe_error}. "
                        f"Retrying in {backoff_seconds:.2f}s"
                    )
                    time.sleep(backoff_seconds)
                    continue
                logger.error(f"Failed to create vector store: {safe_error}")
                raise

        raise last_error
    
    def _process_citation(self, citation: str) -> str:
        """
        Process the citation text to extract and standardize the core claim.
        
        Args:
            citation: Raw citation text
            
        Returns:
            Processed citation text as a clear, quantifiable claim without reference sources
        """
        prompt = _build_citation_processing_prompt(citation)
        
        response = self._llm_completion(prompt, "preprocessing")
        processed = response.strip()

        # Post-process to remove trailing reference-specific fragments and extract keywords
        try:
            final_claim, keywords, retrieval_weights = self._postprocess_claim(
                original_citation=citation,
                processed_claim=processed,
            )
            # store last keywords for later use
            self._last_claim_keywords = keywords
            self._claim_keywords = keywords
            self._retrieval_weights = retrieval_weights
            return final_claim
        except Exception:
            return processed

    def _generate_hypothetical_document(self, claim: str) -> str:
        """Generate a HyDE-style hypothetical document for dense retrieval."""
        prompt = f"""
Generate a short hypothetical scientific passage that could plausibly appear in a paper relevant to the claim below.

Requirements:
1. Preserve the key entities, quantities, methods, and outcomes from the claim.
2. Write in a neutral academic style.
3. Return only the passage text, with no bullets, labels, or explanation.

Claim:
"{claim}"
"""

        response = self._llm_completion(prompt, "preprocessing")
        hypothetical_document = response.strip()
        return hypothetical_document if hypothetical_document else claim

    def _postprocess_claim(self, original_citation: str, processed_claim: str) -> Tuple[str, List[str], Dict[str, float]]:
        """Normalize the model output and extract keywords plus retrieval weights."""
        stopwords = {
            'the','and','for','with','that','this','from','were','were','are','was','is','in','on','of','to','a','an',
            'by','as','it','be','which','or','these','those','their','its'
        }

        marker_pattern = re.compile(
            r"\[CITATION\]|\<\s*cit[^>]*\>|\[cite[^\]]*\]|\<\s*citation\s*\>|\[citation\]",
            flags=re.IGNORECASE,
        )

        candidate = _normalize_claim_text(processed_claim)

        if not candidate and original_citation:
            candidate = _normalize_claim_text(marker_pattern.sub(" ", original_citation))

        # If the model still echoed marker text, remove it before downstream scoring.
        candidate = _normalize_claim_text(marker_pattern.sub(" ", candidate))

        generic_lead_patterns = [
            r"^(?:common|main|typical|usual|several|many)\s+strateg(?:y|ies)\s+include(?:s)?\s+",
            r"^(?:common|main|typical|usual|several|many)\s+approach(?:es)?\s+include(?:s)?\s+",
            r"^(?:these|this|those)\s+include(?:s)?\s+",
            r"^(?:here|there)\s+(?:we|the study|the paper)\s+(?:show|showed|demonstrate|demonstrated|present|presents|describe|describes)\s+",
        ]
        for pattern in generic_lead_patterns:
            candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE)

        # Remove trailing conjunction phrases like 'and', 'or'
        candidate = re.sub(r"\b(and|or)\s*$", "", candidate, flags=re.IGNORECASE).strip()

        # Normalize spacing and punctuation
        candidate = candidate.rstrip(' ,.;:')

        # If the claim is still a long sentence after citation cleanup, retain the most contentful clause.
        clauses = [clause.strip() for clause in re.split(r"[\.;]\s+|\s+—\s+|\s+-\s+", candidate) if clause.strip()]
        if len(clauses) > 1:
            clause_scores: List[Tuple[int, str]] = []
            for clause in clauses:
                clause_tokens = re.findall(r"\b[a-zA-Z]{4,}\b", clause.lower())
                score = sum(1 for token in clause_tokens if token not in stopwords)
                clause_scores.append((score, clause))
            clause_scores.sort(key=lambda item: (-item[0], len(item[1])))
            candidate = clause_scores[0][1]

        # Keyword extraction: simple frequency-based filter
        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", candidate.lower())
        freqs: Dict[str, int] = {}
        for t in tokens:
            if t in stopwords:
                continue
            freqs[t] = freqs.get(t, 0) + 1

        sorted_terms = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
        keywords = [t for t, _ in sorted_terms][:6]

        # Adaptive retrieval weights: concise keyword-like claims benefit more from sparse retrieval,
        # while longer explanatory claims can lean more on dense retrieval.
        token_count = len(re.findall(r"\b[a-zA-Z0-9-]+\b", candidate))
        sparse_bias = 0.35
        dense_bias = 0.65

        if token_count <= 8:
            sparse_bias, dense_bias = 0.75, 0.25
        elif token_count <= 14:
            sparse_bias, dense_bias = 0.60, 0.40
        elif token_count <= 24:
            sparse_bias, dense_bias = 0.45, 0.55

        if any(prefix in (original_citation or "").lower() for prefix in ["[citation]", "<cit.", "<cit>"]):
            sparse_bias = min(0.80, sparse_bias + 0.05)
            dense_bias = 1.0 - sparse_bias

        if any(keyword in candidate.lower() for keyword in ["include", "common strategies", "approach", "methods"]):
            sparse_bias = min(0.85, sparse_bias + 0.10)
            dense_bias = 1.0 - sparse_bias

        return candidate, keywords, {"dense": round(dense_bias, 2), "sparse": round(sparse_bias, 2)}

    def _fuse_retrieval_documents(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        max_docs: int = 15,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Fuse dense and sparse retrieval results with weighted quotas.

        The goal is to keep the stronger retrieval source dominant while still giving the
        other source a bounded chance to contribute distinct evidence.
        """

        dense_docs = list(dense_docs or [])
        sparse_docs = list(sparse_docs or [])

        if max_docs <= 0:
            return [], {
                "max_docs": max_docs,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "selected_dense": 0,
                "selected_sparse": 0,
                "source_order": [],
            }

        if not dense_docs and not sparse_docs:
            return [], {
                "max_docs": max_docs,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "selected_dense": 0,
                "selected_sparse": 0,
                "source_order": [],
            }

        if not dense_docs:
            selected = sparse_docs[:max_docs]
            return selected, {
                "max_docs": max_docs,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "selected_dense": 0,
                "selected_sparse": len(selected),
                "source_order": ["sparse"],
            }

        if not sparse_docs:
            selected = dense_docs[:max_docs]
            return selected, {
                "max_docs": max_docs,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "selected_dense": len(selected),
                "selected_sparse": 0,
                "source_order": ["dense"],
            }

        total_weight = dense_weight + sparse_weight
        if total_weight <= 0:
            raise ValueError("dense_weight + sparse_weight must be greater than zero")

        dense_quota = int(round(max_docs * dense_weight / total_weight))
        sparse_quota = max_docs - dense_quota

        dense_quota = min(dense_quota, len(dense_docs))
        sparse_quota = min(sparse_quota, len(sparse_docs))

        # If one side could not use its full quota, let the other side absorb the slack.
        remaining = max_docs - (dense_quota + sparse_quota)
        while remaining > 0:
            if dense_quota < len(dense_docs) and dense_weight >= sparse_weight:
                dense_quota += 1
            elif sparse_quota < len(sparse_docs):
                sparse_quota += 1
            elif dense_quota < len(dense_docs):
                dense_quota += 1
            else:
                break
            remaining -= 1

        primary_first = dense_weight >= sparse_weight
        ordered_sources = (
            [("dense", dense_docs[:dense_quota]), ("sparse", sparse_docs[:sparse_quota])]
            if primary_first
            else [("sparse", sparse_docs[:sparse_quota]), ("dense", dense_docs[:dense_quota])]
        )

        selected: List[Document] = []
        seen_content = set()

        for _, docs in ordered_sources:
            for doc in docs:
                if doc.page_content not in seen_content:
                    selected.append(doc)
                    seen_content.add(doc.page_content)

        # Top up from the dominant source first, then the other source, if deduplication removed items.
        top_up_sources = (
            [("dense", dense_docs[dense_quota:]), ("sparse", sparse_docs[sparse_quota:])]
            if primary_first
            else [("sparse", sparse_docs[sparse_quota:]), ("dense", dense_docs[dense_quota:])]
        )

        for _, docs in top_up_sources:
            for doc in docs:
                if len(selected) >= max_docs:
                    break
                if doc.page_content not in seen_content:
                    selected.append(doc)
                    seen_content.add(doc.page_content)
            if len(selected) >= max_docs:
                break

        return selected[:max_docs], {
            "max_docs": max_docs,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "selected_dense": dense_quota,
            "selected_sparse": sparse_quota,
            "source_order": ["dense", "sparse"] if primary_first else ["sparse", "dense"],
        }

    def _build_support_analysis_prompt(
        self,
        citation: str,
        chunks: List[Dict],
        metadata: Optional[str] = None,
        keywords: Optional[List[str]] = None,
    ) -> str:
        """Build a conservative support-analysis prompt."""

        processed_chunks = []
        for chunk in chunks:
            processed_chunk = {
                "text": chunk["text"],
                "location": chunk["location"]
            }
            processed_chunks.append(processed_chunk)

        metadata_section = f"\nReference Document Information:\n{metadata}\n" if metadata else ""
        keywords_section = f"\nKey claim keywords: {', '.join(keywords)}\n" if keywords else ""

        main_instruction = (
            "Analyze how well the following citation is supported by the reference text snippets."
        )
        if metadata:
            main_instruction = (
                "Analyze how well the following citation is supported by the reference text snippets.\n"
                "Use the Reference Document Information to understand the study context."
            )

        reasoning_instruction = ""
        if metadata:
            reasoning_instruction = (
                "When explaining your reasoning, mention if the citation fits the overall study described in the document information.\n\n"
            )

        conservative_guidance = """
Be conservative and evidence-grounded.
- Return SUPPORTED only when the reference text directly supports the claim or a close paraphrase.
- Return REFUTED only when the reference text explicitly contradicts the claim.
- Return NEI when the evidence is partial, indirect, topic-adjacent, or requires inference.
- Do not treat general topical similarity, related methods, or overlapping keywords as support.
- If the evidence is mixed or ambiguous, prefer NEI.

Don't be so strict since sometimes unrelated claims may appear)
"""

        format_instructions = _build_json_format_instructions(reasoning_instruction)

        return f"""{main_instruction}

Citation: "{citation}"{metadata_section}{keywords_section}
Relevant Text Snippets:
{json.dumps(processed_chunks, indent=2)}

Classify the citation as one of:
SUPPORTED - Full alignment with source, complete representation.
REFUTED - The source explicitly contradicts or denies the claim.
NEI - The information within the source is insufficient to determine whether the claim is true or false.

{conservative_guidance}
{format_instructions}"""

    async def _process_citation_async(self, citation: str) -> str:
        """Async version of citation preprocessing."""
        prompt = _build_citation_processing_prompt(citation)

        response = await self._llm_completion_async(prompt, "preprocessing")
        processed = response.strip()

        try:
            final_claim, keywords, retrieval_weights = self._postprocess_claim(
                original_citation=citation,
                processed_claim=processed,
            )
            self._last_claim_keywords = keywords
            self._claim_keywords = keywords
            self._retrieval_weights = retrieval_weights
            return final_claim
        except Exception:
            return processed
    

    def _get_relevant_chunks_hybrid(self, vector_store: Chroma, claim: str, documents: List[Document],
                                  num_initial_chunks: int = 15,
                                  relevance_threshold: float = 0.5,
                                  max_chunks: int = 5,
                                  return_separate_retrievals: bool = False,
                                  dense_weight: float = 0.65,
                                  sparse_weight: float = 0.35) -> Tuple[List[Dict], Optional[List[Document]], Optional[List[Document]], Optional[List]]:
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
            dense_docs = []
            seen_dense_content = set()

            dense_queries = [("original claim", claim)]
            try:
                hypothetical_document = self._generate_hypothetical_document(claim)
                if hypothetical_document and hypothetical_document.strip() and hypothetical_document.strip() != claim.strip():
                    dense_queries.insert(0, ("hypothetical document", hypothetical_document))
                    logger.info(
                        "Generated hypothetical document for dense retrieval (HyDE-style augmentation)"
                    )
            except Exception as e:
                safe_error = sanitize_error_message(e)
                logger.warning(
                    f"Hypothetical document generation failed; continuing with original claim only: {safe_error}"
                )

            for query_label, query_text in dense_queries:
                logger.info(f"Performing dense retrieval for {query_label} (k={num_initial_chunks})")
                retrieved_docs = dense_retriever.invoke(query_text)
                logger.info(f"Dense retrieval for {query_label} returned {len(retrieved_docs)} documents")
                for doc in retrieved_docs:
                    if doc.page_content not in seen_dense_content:
                        dense_docs.append(doc)
                        seen_dense_content.add(doc.page_content)

            logger.info(f"Dense retrieval returned {len(dense_docs)} unique documents across {len(dense_queries)} query variant(s)")
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

        # # print first 3 sparse retrieved chunks for debugging
        # for i, doc in enumerate(sparse_docs[:3]):
        #     preview = doc.page_content[:100].replace('\n', ' ')
        #     logger.info(f"BM25 retrieved doc {i+1}: chunk_id={doc.metadata.get('chunk_id')} content_preview='{preview}...'")

        # 3. Fuse dense and sparse results with explicit weighting.
        unique_docs, fusion_details = self._fuse_retrieval_documents(
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
            max_docs=15,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        logger.info(
            "Weighted fusion selected %s unique documents (dense_weight=%s, sparse_weight=%s, source_order=%s)",
            len(unique_docs),
            dense_weight,
            sparse_weight,
            fusion_details.get("source_order"),
        )

        # 4. Take top candidates for reranking (limit to avoid too many)
        candidates = unique_docs[:min(15, len(unique_docs))]
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
        analysis_prompt = self._build_support_analysis_prompt(
            citation=citation,
            chunks=chunks,
            metadata=metadata,
            keywords=getattr(self, "_last_claim_keywords", None),
        )

        # Call Together completion
        try:
            response = self._llm_completion(analysis_prompt, "classification")

            parsed_response = _parse_json_model_response(response)

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

        analysis_prompt = self._build_support_analysis_prompt(
            citation=citation,
            chunks=chunks,
            metadata=metadata,
            keywords=getattr(self, "_last_claim_keywords", None),
        )

        try:
            response = await self._llm_completion_async(analysis_prompt, "classification")

            parsed_response = _parse_json_model_response(response)

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
            3,
            save_chunks,
            self._retrieval_weights.get("dense", 0.65),
            self._retrieval_weights.get("sparse", 0.35),
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
                "saved_chunks": saved_files,
            }
        }

        self.chunks = chunks

        try:
            del vector_store
            logger.info("Vector store released successfully")
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
    ) -> Dict[str, Any]:
        """Process citation/reference pairs concurrently with asyncio.gather.

        Returns:
            Dict with keys:
                "results": List[Dict] — per-instance results
                "llm_metrics": Dict — aggregated LLM run metrics across the batch
        """

        run_metrics_token = _CURRENT_LLM_RUN_METRICS.set([])
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
        results = await asyncio.gather(*tasks)

        llm_metrics = self._build_llm_run_metrics_summary()
        _CURRENT_LLM_RUN_METRICS.reset(run_metrics_token)

        processing_times = [
            r["metadata"]["processing_time"]
            for r in results
            if r.get("metadata") and not r["metadata"].get("error")
            and r["metadata"].get("processing_time") is not None
        ]
        llm_metrics["avg_time_per_instance"] = (
            round(sum(processing_times) / len(processing_times), 4) if processing_times else 0.0
        )

        logger.info("=" * 60)
        logger.info("Batch LLM run metrics summary")
        logger.info(f"  total_calls:            {llm_metrics['total_calls']}")
        logger.info(f"  total_latency_seconds:  {llm_metrics['total_latency_seconds']}")
        logger.info(f"  avg_latency_seconds:    {llm_metrics['avg_latency_seconds']}")
        logger.info(f"  total_input_tokens:     {llm_metrics['total_input_tokens']}")
        logger.info(f"  total_output_tokens:    {llm_metrics['total_output_tokens']}")
        logger.info(f"  total_tokens:           {llm_metrics['total_tokens']}")
        logger.info(f"  estimated_cost_usd:     {llm_metrics['estimated_cost_usd']}")
        logger.info(f"  avg_time_per_instance:  {llm_metrics['avg_time_per_instance']}s")
        logger.info("=" * 60)

        return {"results": results, "llm_metrics": llm_metrics}

    def check_citation_batch(
        self,
        citation_reference_pairs: List[Dict[str, Any]],
        save_chunks: bool = False,
        output_dir: str = "./retrieval_output",
        max_concurrency: Optional[int] = None
    ) -> Dict[str, Any]:
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
            return_separate_retrievals=save_chunks,
            dense_weight=self._retrieval_weights.get("dense", 0.65),
            sparse_weight=self._retrieval_weights.get("sparse", 0.35),
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
                "saved_chunks": saved_files,  # Include saved file paths if chunks were saved
            }
        }

        self.chunks = chunks

        # Clean up vector store to prevent contamination across documents
        try:
            del vector_store
            logger.info("Vector store released successfully")
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
    checker = ReferenceChecker(llm_provider="together", embedding_provider="local")
    
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
) -> Dict[str, Any]:
    """Check a batch of citation/reference-text pairs concurrently.

    Returns:
        Dict with keys:
            "results": List[Dict] — per-instance results
            "llm_metrics": Dict — aggregated LLM run metrics across the batch
    """

    if llm_config and embedding_config:
        checker = ReferenceChecker(
            llm_provider=llm_config['provider'],
            llm_config=llm_config,
            embedding_provider=embedding_config['provider'],
            embedding_config=embedding_config
        )
    else:
        checker = ReferenceChecker()

    result = checker.check_citation_batch(
        citation_reference_pairs=citation_reference_pairs,
        save_chunks=save_chunks,
        output_dir=output_dir,
        max_concurrency=max_concurrency,
    )

    return result