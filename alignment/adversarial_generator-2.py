"""
Adversarial Negative Sample Generator for Citation Verification.

Generates high-quality "hard" negative claims by applying semantic drifts
(Over-Claim, Context-Shift, Reversal, Tangential) with an
Analyzer → Generator → Discriminator → Filter pipeline.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import time
import uuid
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import dotenv
import numpy as np

from security_utils import sanitize_error_message

# ---------------------------------------------------------------------------
# Environment & Logging
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(_PROJECT_ROOT / ".env")

LOGS_DIR = str(_PROJECT_ROOT / "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

_file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'adversarial_generator.log'),
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
]

# ==============================================================================
# Interfaces (Protocols)
# ==============================================================================

class AsyncLLMClient(Protocol):
    """Protocol for asynchronous LLM client."""
    async def agenerate(self, prompt: str) -> str:
        ...

class AsyncEmbeddingClient(Protocol):
    """Protocol for embedding client."""
    def embed_query(self, text: str) -> List[float]:
        ...

# ==============================================================================
# Concrete clients (reused from benchmark_builder-2.py)
# ==============================================================================

class TogetherLLMClient:
    """TogetherAI async LLM client implementing AsyncLLMClient."""

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


class TogetherEmbeddingClient:
    """TogetherAI embedding client implementing AsyncEmbeddingClient."""

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-large-instruct",
        api_key: Optional[str] = None,
    ):
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

    def embed_query(self, text: str) -> List[float]:
        client = self._lazy_client()
        response = client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding


def build_embedding_client(provider: str = "together", **kwargs: Any) -> AsyncEmbeddingClient:
    """Factory returning an embedding client."""
    if provider == "together":
        return TogetherEmbeddingClient(
            model=kwargs.get("model", "intfloat/multilingual-e5-large-instruct"),
            api_key=kwargs.get("api_key"),
        )
    raise ValueError(f"Unknown embedding provider: {provider}")

# ==============================================================================
# Enums
# ==============================================================================

class SemanticDriftType(str, Enum):
    """Taxonomy of Adversarial Alignment Errors."""
    OVER_CLAIM = "over_claim"               # Exaggeration (Label: Partially Supported)
    CONTEXT_SHIFT = "context_shift"         # Ignoring conditionals/domains (Label: Unsupported)
    REVERSAL = "reversal"                   # Flipping conclusions (Label: Unsupported)
    TANGENTIAL = "tangential"               # Related topic but unverified (Label: Uncertain)


# ==============================================================================
# Core Generator Class
# ==============================================================================

class AdversarialSampleGenerator:
    """
    Generates hard negative claims using an Analyzer-Generator-Discriminator-Filter pipeline.
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        embedding_client: AsyncEmbeddingClient,
        similarity_threshold: float = 0.80,
        max_retries: int = 3
    ) -> None:
        self.llm = llm_client
        self.embeddings = embedding_client
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two vectors."""
        v1, v2 = np.array(vec1), np.array(vec2)
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Safely parses JSON from LLM responses."""
        try:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(response)
        except Exception as e:
            logger.error(f"JSON Parsing Error: {e} | Raw response: {response[:100]}")
            return {}

    # --------------------------------------------------------------------------
    # 1. Applicability Analyzer Phase
    # --------------------------------------------------------------------------

    def _get_analyzer_prompt(self, true_claim: str, evidence: str) -> str:
        """Builds the prompt for the Applicability Analyzer Agent."""
        return f"""You are an expert linguistic analyst evaluating the feasibility of generating adversarial scientific claims.
Given a True Claim and its supporting Evidence, determine WHICH of the following 4 semantic drift techniques can logically and naturally be applied to create a deceptive negative claim.

1. OVER_CLAIM: Applicable ONLY IF the claim contains specific scopes, quantities, or modest findings that can be plausibly exaggerated into universal or absolute breakthroughs.
2. CONTEXT_SHIFT: Applicable ONLY IF the evidence relies on specific conditions, datasets, domains, or limitations that can be subtly swapped out for unsupported ones.
3. REVERSAL: Applicable ONLY IF the claim establishes a directional relationship, comparison, or boolean outcome (e.g., A improves B, X is faster than Y) that can be logically inverted.
4. TANGENTIAL: Generally applicable, but requires the topic to be broad enough to invent a plausible related methodology or application absolutely absent from the evidence.

True Claim: "{true_claim}"

Evidence:
{evidence}

Analyze the claim and evidence, then return a JSON object evaluating the applicability of EACH drift type. 
Return your response STRICTLY as a JSON object with this exact structure:
{{
    "evaluations": {{
        "over_claim": {{"is_applicable": true/false, "reason": "brief rationale"}},
        "context_shift": {{"is_applicable": true/false, "reason": "brief rationale"}},
        "reversal": {{"is_applicable": true/false, "reason": "brief rationale"}},
        "tangential": {{"is_applicable": true/false, "reason": "brief rationale"}}
    }}
}}
"""

    async def analyze_applicability(self, true_claim: str, evidence: str) -> List[SemanticDriftType]:
        """
        Asks the LLM to determine which drift types are applicable.
        Returns a list of 0 to 4 applicable SemanticDriftType enums.
        """
        prompt = self._get_analyzer_prompt(true_claim, evidence)
        response = await self.llm.agenerate(prompt)
        print(response)
        parsed_data = self._parse_json(response)
        
        evaluations = parsed_data.get("evaluations", {})
        applicable_drifts = []
        
        for drift_key, drift_enum in [
            ("over_claim", SemanticDriftType.OVER_CLAIM),
            ("context_shift", SemanticDriftType.CONTEXT_SHIFT),
            ("reversal", SemanticDriftType.REVERSAL),
            ("tangential", SemanticDriftType.TANGENTIAL)
        ]:
            drift_data = evaluations.get(drift_key, {})
            if drift_data.get("is_applicable") is True:
                applicable_drifts.append(drift_enum)
                
        return applicable_drifts

    # --------------------------------------------------------------------------
    # 2. Generator & Discriminator Phase
    # --------------------------------------------------------------------------

    def _get_generator_prompt(self, true_claim: str, evidence: str, drift_type: SemanticDriftType) -> str:
        """Builds the prompt for the Adversarial Agent."""
        instructions = {
            SemanticDriftType.OVER_CLAIM: (
                "Exaggerate the findings. Take a modest or specific claim and inflate it into "
                "a universal, absolute, or highly generalized breakthrough. Keep the same academic tone."
            ),
            SemanticDriftType.CONTEXT_SHIFT: (
                "Shift the context. The original evidence holds true under specific conditions "
                "(e.g., specific datasets, domains, or limitations). Rewrite the claim to apply "
                "these findings to a completely different, unsupported domain or condition."
            ),
            SemanticDriftType.REVERSAL: (
                "Reverse the conclusion. Flip the causal relationship, the comparison results "
                "(e.g., 'A is faster than B' to 'B is faster than A'), or negate the primary finding, "
                "WHILE using mostly the same vocabulary as the original claim."
            ),
            SemanticDriftType.TANGENTIAL: (
                "Introduce a tangential hallucination. Write a claim that shares the same overarching "
                "topic, but asserts a specific application, methodology, or result that is "
                "ABSOLUTELY NOT mentioned in the evidence."
            )
        }

        return f"""You are an adversarial AI researcher generating robustness tests.
Create a highly deceptive, "hard negative" academic claim based on the provided ground-truth evidence.

Original (True) Claim: "{true_claim}"

Ground-Truth Evidence:
{evidence}

Drift Strategy: {drift_type.value.upper()}
Instruction: {instructions[drift_type]}

Crucial Rules:
1. The generated claim MUST sound academically fluent and highly plausible.
2. DO NOT use simplistic lexical negations (e.g., do not just add the word "not"). 
3. The claim MUST NOT be fully supported by the Evidence.

Return your response strictly as a JSON object with this structure:
{{
    "adversarial_claim": "<your generated deceptive claim>",
    "rationale": "<brief explanation of how it fulfills the drift strategy>"
}}
"""

    def _get_judge_prompt(self, adversarial_claim: str, evidence: str) -> str:
        """Builds the prompt for the Zero-shot Discriminator."""
        return f"""You are a strict, impartial peer reviewer auditing scientific citations.
Evaluate whether the following Claim is supported by the provided Evidence.

Claim: "{adversarial_claim}"

Evidence:
{evidence}

Classify the alignment into ONE of the following categories:
- SUPPORTED: The claim is fully backed by the evidence.
- UNSUPPORTED: The claim contradicts, exaggerates, or shifts the context of the evidence.
- UNCERTAIN: The claim introduces information completely absent from the evidence.

Return your response strictly as a JSON object with this structure:
{{
    "label": "<SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<brief explanation>"
}}
"""

    async def _evaluate_candidate(
        self,
        adversarial_claim: str,
        evidence: str,
        true_claim: str
    ) -> Tuple[bool, str, float]:
        """Runs the Discriminator and Semantic Filter."""
        # 1. Discriminator (Judge)
        judge_prompt = self._get_judge_prompt(adversarial_claim, evidence)
        judge_response = await self.llm.agenerate(judge_prompt)
        judge_data = self._parse_json(judge_response)

        predicted_label = judge_data.get("label", "SUPPORTED").upper()

        if predicted_label == "SUPPORTED":
            return False, "SUPPORTED", 0.0

        # 2. Filter (Cosine Similarity)
        try:
            emb_true = self.embeddings.embed_query(true_claim)
            emb_fake = self.embeddings.embed_query(adversarial_claim)
            sim_score = self._cosine_similarity(emb_true, emb_fake)

            if sim_score < self.similarity_threshold:
                return False, f"SIMILARITY_FAIL:{sim_score:.2f}", sim_score

            return True, predicted_label, sim_score
        except Exception as e:
            logger.warning(f"Embedding failed: {sanitize_error_message(e)}")
            return False, "EMBEDDING_ERROR", 0.0

    async def generate_hard_negative(
        self, 
        true_claim: str, 
        evidence: str, 
        drift_type: SemanticDriftType
    ) -> Optional[Dict[str, Any]]:
        """End-to-end pipeline to generate, judge, and filter a hard negative claim."""
        prompt = self._get_generator_prompt(true_claim, evidence, drift_type)

        for attempt in range(self.max_retries):
            gen_response = await self.llm.agenerate(prompt)
            gen_data = self._parse_json(gen_response)
            
            adversarial_claim = gen_data.get("adversarial_claim")
            rationale = gen_data.get("rationale")
            
            if not adversarial_claim:
                continue
                
            is_valid, judge_status, sim_score = await self._evaluate_candidate(
                adversarial_claim, evidence, true_claim
            )
            
            if is_valid:
                return {
                    "original_claim": true_claim,
                    "adversarial_claim": adversarial_claim,
                    "drift_type": drift_type.value,
                    "target_alignment_label": judge_status,
                    "generator_rationale": rationale,
                    "lexical_similarity": round(sim_score, 4)
                }

        return None

    # --------------------------------------------------------------------------
    # 3. Batch Processing Endpoint
    # --------------------------------------------------------------------------

    async def process_batch_instances(
        self, 
        instances: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes applicability and generates ALL VALID negative samples 
        for each positive instance. Enhances the dataset significantly.
        """
        augmented_instances = []
        
        for inst in instances:
            true_claim = inst.get("claim_text") or inst.get("citing_context", {}).get("claim_text", "")
            
            # Combine evidence chunks
            evidences = inst.get("retrieved_evidences", {}).get("extractive_chunks", [])
            evidence_text = "\n".join([e["extractive_text"] for e in evidences])
            
            if not true_claim or not evidence_text:
                continue
                
            # Step 1: Analyze Applicability (0 to 4 types)
            logger.info(f"Analyzing applicability for claim: '{true_claim[:50]}...'")
            applicable_drifts = await self.analyze_applicability(true_claim, evidence_text)
            
            logger.info(f"Applicable drifts found ({len(applicable_drifts)}): {[d.value for d in applicable_drifts]}")
            
            if not applicable_drifts:
                logger.info("No valid adversarial taxonomies for this claim. Skipping.")
                continue
                
            # Step 2: Generate negatives for EVERY applicable drift
            # This allows 1 true claim to branch into multiple high-quality negative samples!
            for drift_type in applicable_drifts:
                negative_sample = await self.generate_hard_negative(
                    true_claim=true_claim, 
                    evidence=evidence_text, 
                    drift_type=drift_type
                )
                
                if negative_sample:
                    neg_inst = json.loads(json.dumps(inst)) # Deep copy
                    neg_inst["instance_id"] = str(uuid.uuid4())
                    neg_inst["is_adversarial"] = True
                    neg_inst["adversarial_metadata"] = negative_sample
                    
                    if "citing_context" in neg_inst:
                        neg_inst["citing_context"]["claim_text"] = negative_sample["adversarial_claim"]
                    else:
                        neg_inst["claim_text"] = negative_sample["adversarial_claim"]
                        
                    # Update ground truth label based on the judge's assessment
                    label_map = {"SUPPORTED": 0, "UNSUPPORTED": 1, "UNCERTAIN": 2}
                    if "ground_truth" in neg_inst and "task2_alignment" in neg_inst["ground_truth"]:
                        str_label = negative_sample["target_alignment_label"]
                        neg_inst["ground_truth"]["task2_alignment"]["label"] = label_map.get(str_label, 2)
                        
                    augmented_instances.append(neg_inst)

        return augmented_instances


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate adversarial negative samples for citation alignment."
    )
    parser.add_argument("--input", required=True,
                        help="Path to input JSON with instances (post-retrieval)")
    parser.add_argument("--output", required=True,
                        help="Path for the output augmented JSON")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="Limit number of input instances (0 = all)")
    parser.add_argument("--llm-model", default=TOGETHER_MODEL_OPTIONS[0],
                        help="TogetherAI model for generation/judging")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                        help="Temperature for LLM calls")
    parser.add_argument("--similarity-threshold", type=float, default=0.80,
                        help="Min cosine similarity between original and adversarial claim")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max generation attempts per drift type")
    parser.add_argument("--embedding-model",
                        default="intfloat/multilingual-e5-large-instruct",
                        help="TogetherAI embedding model for similarity filter")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    logger.info(f"Loading input from {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances = data if isinstance(data, list) else data.get("instances", [])
    if args.max_instances > 0:
        instances = instances[:args.max_instances]
        logger.info(f"Limited to {args.max_instances} instances")

    llm_client = TogetherLLMClient(
        model=args.llm_model,
        temperature=args.llm_temperature,
    )
    embedding_client = build_embedding_client(
        provider="together",
        model=args.embedding_model,
    )

    generator = AdversarialSampleGenerator(
        llm_client=llm_client,
        embedding_client=embedding_client,
        similarity_threshold=args.similarity_threshold,
        max_retries=args.max_retries,
    )

    augmented = await generator.process_batch_instances(instances)
    logger.info(f"Generated {len(augmented)} adversarial instances")

    output = {"adversarial_instances": augmented}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Written to {out_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

# python .\alignment\adversarial_generator-2.py --input .\alignment\data\results.json --output .\alignment\data\negatives.json