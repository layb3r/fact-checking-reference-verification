"""
LLM client abstraction layer — provides a pluggable interface for LLM providers.

Defines:
  - LLMResponse  : structured return type with content + metadata (tokens, latency, cost)
  - BaseLLMClient: abstract base class for any LLM provider
  - TogetherLLMClient: concrete implementation for Together AI
  - summarize_llm_run_metrics: aggregate a list of LLMResponse into a summary dict
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# ==============================================================================
# Together AI model catalogue (defaults / pricing)
# ==============================================================================

TOGETHER_MODEL_OPTIONS = [
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "Qwen/Qwen3.5-9B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "openai/gpt-oss-20b",
]

TOGETHER_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "Qwen/Qwen2.5-7B-Instruct-Turbo": {"input": 0.30, "output": 0.30},
    "Qwen/Qwen3.5-9B": {"input": 0.17, "output": 0.25},
    "meta-llama/Llama-3.1-8B-Instruct": {"input": 0.0, "output": 0.0},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.20},
}

OPENROUTER_MODEL_OPTIONS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen3.5-9b",
    "google/gemma-3-12b-it",
    "mistralai/mistral-nemo",
    "openai/gpt-oss-20b"
]

OPENROUTER_MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "meta-llama/llama-3.1-8b-instruct": {"input": 0.02, "output": 0.04},
    "qwen/qwen3.5-9b": {"input": 0.10, "output": 0.15},
    "google/gemma-3-12b-it": {"input": 0.05, "output": 0.15},
    "mistralai/mistral-nemo": {"input": 0.019, "output": 0.03},
    "openai/gpt-oss-20b": {"input": 0.03, "output": 0.13},
}

# ==============================================================================
# Structured response type
# ==============================================================================


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    model_name: str = ""
    provider: str = ""
    estimated_cost_usd: Optional[float] = None


# ==============================================================================
# Abstract base class for any LLM client
# ==============================================================================


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...

    @abstractmethod
    def get_provider(self) -> str:
        ...


# ==============================================================================
# Together AI concrete implementation
# ==============================================================================


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


class TogetherLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str = TOGETHER_MODEL_OPTIONS[0],
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        max_retries: int = 3,
        base_delay: float = 1.5,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("TOGETHER_API") or os.getenv("TOGETHER_API_KEY2")
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._client: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseLLMClient interface
    # ------------------------------------------------------------------

    def get_model_name(self) -> str:
        return self._model

    def get_provider(self) -> str:
        return "together"

    async def generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "Together API key not set. Provide api_key or set "
                    "the TOGETHER_API / TOGETHER_API_KEY env var."
                )
            from together import AsyncTogether
            self._client = AsyncTogether(api_key=self._api_key)

        temp = temperature if temperature is not None else self._temperature
        model_name = self._model

        for attempt in range(1, self._max_retries + 1):
            try:
                start_time = time.perf_counter()
                response = await self._client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=self._max_tokens,
                )
                latency_seconds = time.perf_counter() - start_time

                usage = getattr(response, "usage", None)
                prompt_tokens = _extract_usage_count(usage, "prompt_tokens")
                completion_tokens = _extract_usage_count(usage, "completion_tokens")

                llm_config: Dict[str, Any] = {
                    "input_cost_per_million_tokens": None,
                    "output_cost_per_million_tokens": None,
                }
                estimated_cost_usd = _estimate_together_cost(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    llm_config=llm_config,
                )

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < self._max_retries:
                        delay = self._base_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                        continue
                    raise ValueError("Empty model response after all retries")

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=round(latency_seconds, 4),
                    model_name=model_name,
                    provider="together",
                    estimated_cost_usd=estimated_cost_usd,
                )

            except Exception as e:
                error_text = str(e).lower()
                retryable = any(m in error_text for m in [
                    "rate limit", "too many requests", "429", "quota",
                    "resource exhausted", "server error", "502", "503", "504",
                    "timeout", "temporarily unavailable", "connection error",
                ])
                if retryable and attempt < self._max_retries:
                    delay = self._base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Retryable error attempt {attempt}/{self._max_retries}: {e}, "
                        f"retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise


class OpenRouterLLMClient(BaseLLMClient):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = OPENROUTER_MODEL_OPTIONS[0],
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 1.5,
        site_url: Optional[str] = None,
        app_title: Optional[str] = None,
    ):
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        if not self._api_key:
            raise RuntimeError(
                "OpenRouter API key not set. Provide api_key or set "
                "the OPENROUTER_API_KEY env var."
            )

        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_title:
            default_headers["X-OpenRouter-Title"] = app_title

        self._client: Optional[AsyncOpenAI] = None

    def get_model_name(self) -> str:
        return self._model

    def get_provider(self) -> str:
        return "openrouter"

    async def generate(self, prompt: str, temperature: Optional[float] = None) -> LLMResponse:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self.BASE_URL,
            )

        temp = temperature if temperature is not None else self._temperature
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(1, self._max_retries + 1):
            try:
                start = time.perf_counter()

                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=self._max_tokens,
                )

                latency = time.perf_counter() - start

                usage = getattr(response, "usage", None)
                prompt_tokens = _extract_usage_count(usage, "prompt_tokens")
                completion_tokens = _extract_usage_count(usage, "completion_tokens")

                content = response.choices[0].message.content
                if not content or not content.strip():
                    if attempt < self._max_retries:
                        delay = self._retry_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                        continue
                    raise ValueError("Empty model response after all retries")

                return LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_seconds=round(latency, 4),
                    model_name=self._model,
                    provider="openrouter",
                    estimated_cost_usd=None,
                )

            except Exception as e:
                error_text = str(e).lower()
                retryable = any(m in error_text for m in [
                    "rate limit", "too many requests", "429", "quota",
                    "resource exhausted", "server error", "502", "503", "504",
                    "timeout", "temporarily unavailable", "connection error",
                ])
                if retryable and attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"Retryable error attempt {attempt}/{self._max_retries}: {e}, "
                        f"retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

# ==============================================================================
# Utility: aggregate a list of LLMResponse into a summary metrics dict
# ==============================================================================


def summarize_llm_run_metrics(call_metrics: Optional[List[LLMResponse]]) -> Dict[str, Any]:
    metrics = list(call_metrics or [])
    total_calls = len(metrics)
    if total_calls == 0:
        return {
            "provider": "",
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
        }
    total_latency = sum(float(m.latency_seconds) for m in metrics)
    total_input_tokens = sum(int(m.prompt_tokens) for m in metrics)
    total_output_tokens = sum(int(m.completion_tokens) for m in metrics)
    total_tokens = total_input_tokens + total_output_tokens
    estimated_costs = [m.estimated_cost_usd for m in metrics if m.estimated_cost_usd is not None]
    estimated_cost_usd = round(sum(float(c) for c in estimated_costs), 6) if estimated_costs else None
    return {
        "provider": metrics[0].provider,
        "model": metrics[0].model_name,
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
    }
