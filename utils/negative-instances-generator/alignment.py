"""
Alignment Hallucination Generator
===================================
Generates negative instances where the reference EXISTS (true_existence=1) but the
claim_text or surrounding_context does NOT align with what the cited paper says.

Alignment labels (FEVER-style):
  - true_alignment=0: SUPPORTED - claim is supported by the reference
  - true_alignment=1: REFUTED - claim contradicts or is refuted by the reference
  - true_alignment=2: NOT ENOUGH INFO - insufficient information to verify the claim

This generator produces negative instances for labels 1 and 2:

  REFUTED (true_alignment=1) - ~67% of generated instances
  ─────────────────────────────────────────────────────────
  Single strategy: Generate claims that contradict, misrepresent, or are unsupported
  by what the cited paper actually reports.

  NOT ENOUGH INFO (true_alignment=2) - ~33% of generated instances
  ──────────────────────────────────────────────────────────────────
  A. VAGUE_CLAIM          - claim too vague/broad to verify from the paper
  B. MISSING_DETAILS      - claim requires specific details the paper doesn't provide
  C. DIFFERENT_SCOPE      - claim about aspects/populations the paper doesn't study  
  D. UNVERIFIABLE_METRIC  - claim uses metrics/measures the paper doesn't report

Note: The [CITATION] marker in claim_text and surrounding_context must be preserved in all outputs.

Usage:
    python generate_alignment_hallucinations.py \
        --input  positives.json \
        --output negatives_alignment.json \
        --target 2000 \
        --api-key YOUR_GEMINI_API_KEY   # or set GEMINI_API_KEY env var

Dependencies:
    pip install google-generativeai
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Optional
import ast

import google.generativeai as genai
from ratelimiter import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemma-4-31b-it"

# How the instances split between the two alignment labels
ALIGNMENT_SPLIT = {
    1: 0.4,   # refuted
    2: 0.6,   # not enough info
}

# Label 1 (REFUTED) has no strategy subdivisions - just one approach

# How label 2 (NOT ENOUGH INFO) instances split across strategies
NOT_ENOUGH_INFO_STRATEGY_SPLIT = {
    "VAGUE_CLAIM":         0.55,
    "MISSING_DETAILS":     0.45,
}

# ---------------------------------------------------------------------------
# Prompt templates  (each returns a JSON object with new_claim, new_context, rationale)
# ---------------------------------------------------------------------------

# Shared JSON schema reminder appended to every prompt
_JSON_REMINDER = """
Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.

IMPORTANT: You MUST preserve the [CITATION] marker in the new_claim_text.

If the provided `CLAIM` is only a bibliographic mention (for example: "TRAPPIST [CITATION]" or
"TRAPPIST-South [CITATION]") and contains no propositional content (no verb/claimable fact),
return JSON with `new_claim_text` set to `null` and `rationale` explaining it is a citation-only
entry (e.g. "NO_CLAIM: citation-only mention").

Schema:
{{
    "new_claim_text": "<rewritten claim sentence(s) with [CITATION] marker preserved>",
    "rationale": "<one concise sentence explaining the misalignment type and how it was introduced>"
}}
"""

# ── true_alignment=1 (REFUTED) ─────────────────────────────────────────────

PROMPT_REFUTED = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Note: The [CITATION] marker indicates where the reference citation appears in the text.

Task — REFUTED:
Rewrite the claim text so that it DIRECTLY REFUTES/CONTRADICTS what the cited paper likely reports.
The rewritten claim must:
- Negate or invert a specific finding, result, or conclusion
- Still sound like a natural academic sentence (not obviously absurd)
- Keep the citation to the same paper (metadata unchanged)
- PRESERVE the [CITATION] marker in the exact same position
""" + _JSON_REMINDER

# ── true_alignment=2 (NOT ENOUGH INFO) ──────────────────────────────────────

PROMPT_VAGUE_CLAIM = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Note: The [CITATION] marker indicates where the reference citation appears in the text.

Task — VAGUE CLAIM:
Rewrite the claim text so that it is too VAGUE or BROAD to verify from the cited paper.
The claim should:
- Be worded so generally that it's impossible to confirm or refute from the paper
- Sound meaningful but be non-committal about specifics
- Avoid concrete details about methods, magnitudes, populations, or conditions
- Not be obviously wrong, just unverifiable

IMPORTANT: PRESERVE the [CITATION] marker in the exact same position.
""" + _JSON_REMINDER

PROMPT_MISSING_DETAILS = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Note: The [CITATION] marker indicates where the reference citation appears in the text.

Task — MISSING DETAILS:
Rewrite the claim text so that it requires SPECIFIC DETAILS that the cited paper doesn't provide.
The claim should:
- Make specific assertions about details, numbers, conditions, or outcomes
- Be reasonable but require information not present in the paper
- Not be contradicted by the paper, just unverifiable due to missing information

Examples:
- Claim specific percentages when paper gives qualitative results
- Claim specific subgroup results when paper only reports aggregates
- Claim specific parameter values when paper doesn't report them

IMPORTANT: PRESERVE the [CITATION] marker in the exact same position.
""" + _JSON_REMINDER

PROMPT_DIFFERENT_SCOPE = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Note: The [CITATION] marker indicates where the reference citation appears in the text.

Task — DIFFERENT SCOPE:
Rewrite the claim text so that it refers to ASPECTS, POPULATIONS, or CONTEXTS that the cited
paper doesn't actually study or discuss. The claim should:
- Be about something related but outside the paper's scope
- Not directly contradict the paper (that would be REFUTED)
- Simply be unverifiable because the paper doesn't cover that aspect

Examples:
- Paper studies method A, claim is about method B (not compared in paper)
- Paper studies population X, claim is about population Y (not studied)
- Paper focuses on problem P, claim is about problem Q (related but not covered)

IMPORTANT: PRESERVE the [CITATION] marker in the exact same position.
""" + _JSON_REMINDER

# PROMPT_UNVERIFIABLE_METRIC = """You are constructing a citation hallucination benchmark dataset.

# Given the following citation instance:

# CLAIM: {claim_text}
# PAPER TITLE: {title}
# VENUE: {venue}
# YEAR: {year}

# Note: The [CITATION] marker indicates where the reference citation appears in the text.

# Task — UNVERIFIABLE METRIC:
# Rewrite the claim text so that it uses METRICS, MEASURES, or FRAMEWORKS that the cited paper
# doesn't report or evaluate. The claim should:
# - Reference evaluation criteria or measurements not used in the paper
# - Sound plausible and relevant to the paper's topic
# - Be unverifiable because the paper uses different metrics

# Examples:
# - Paper reports accuracy, claim is about F1-score (not reported)
# - Paper measures runtime, claim is about memory usage (not measured)
# - Paper uses framework X, claim is about framework Y (not discussed)

# IMPORTANT: PRESERVE the [CITATION] marker in the exact same position.
# """ + _JSON_REMINDER

# Map strategy name → prompt template
STRATEGY_PROMPTS: dict[str, str] = {
    "REFUTED":             PROMPT_REFUTED,
    "VAGUE_CLAIM":         PROMPT_VAGUE_CLAIM,
    "MISSING_DETAILS":     PROMPT_MISSING_DETAILS,
    # "UNVERIFIABLE_METRIC": PROMPT_UNVERIFIABLE_METRIC,
}

# Which strategies map to which alignment label
STRATEGY_TO_ALIGNMENT: dict[str, int] = {
    "REFUTED":             1,
    "VAGUE_CLAIM":         2,
    "MISSING_DETAILS":     2,
    # "UNVERIFIABLE_METRIC": 2,
}

_CITATION_MARKERS = ("[CITATION]")
_VERB_LIKE_PATTERN = re.compile(
    r"\b(" 
    r"is|are|was|were|be|been|being|has|have|had|does|do|did|"
    r"shows?|showed|showing|demonstrates?|demonstrated|demonstrating|"
    r"indicates?|indicated|indicating|suggests?|suggested|suggesting|"
    r"includes?|included|including|reports?|reported|reporting|"
    r"proposes?|proposed|proposing|uses?|used|using|provides?|provided|providing|"
    r"finds?|found|finding|leads?|led|leading|achieves?|achieved|achieving|"
    r"improves?|improved|improving|requires?|required|requiring|"
    r"supports?|supported|supporting|compares?|compared|comparing|"
    r"evaluates?|evaluated|evaluating|focuses?|focused|focusing|"
    r"examines?|examined|examining|describes?|described|describing|"
    r"presents?|presented|presenting|predicts?|predicted|predicting|"
    r"increases?|increased|increasing|decreases?|decreased|decreasing|"
    r"measures?|measured|measuring|identifies?|identified|identifying|"
    r"introduces?|introduced|introducing|explores?|explored|exploring|"
    r"contains?|contained|containing|"
    r")\b",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_positives(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("instances", "data", "samples"):
            if key in data:
                return data[key]
    raise ValueError(f"Cannot parse JSON structure from {path}")


def deep_copy(inst: dict) -> dict:
    return copy.deepcopy(inst)


def _claim_prefix_before_citation(claim_text: str) -> str:
    text = claim_text or ""
    lowered = text.lower()
    marker_positions = [lowered.find(marker.lower()) for marker in _CITATION_MARKERS]
    marker_positions = [position for position in marker_positions if position != -1]
    if not marker_positions:
        return text.strip()
    return text[: min(marker_positions)].strip()


def is_substantive_claim(claim_text: str) -> bool:
    """Return False for rows that only name the cited work instead of stating a claim."""
    prefix = _claim_prefix_before_citation(claim_text)
    if not prefix:
        return False

    normalized = re.sub(r"[^A-Za-z0-9\s-]+", " ", prefix)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False

    word_count = len(normalized.split())
    if word_count <= 8 and not _VERB_LIKE_PATTERN.search(normalized):
        return False

    return True


def filter_substantive_positives(positives: list[dict]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    skipped = 0
    for inst in positives:
        if is_substantive_claim(inst.get("claim_text", "")):
            kept.append(inst)
        else:
            skipped += 1
    return kept, skipped

# ---------------------------------------------------------------------------
# Target count computation
# ---------------------------------------------------------------------------

def compute_targets(total: int) -> dict[str, int]:
    """
    Returns per-strategy target counts, e.g.:
      {"REFUTED": 1340, "VAGUE_CLAIM": 198, "MISSING_DETAILS": 198, ...}
    """
    n_refuted = round(total * ALIGNMENT_SPLIT[1])
    n_not_enough_info = total - n_refuted

    targets: dict[str, int] = {}
    
    # Label 1 (REFUTED) - single strategy, gets all instances
    targets["REFUTED"] = n_refuted

    # Label 2 (NOT ENOUGH INFO) - split across multiple strategies
    allocated = 0
    nei_items = list(NOT_ENOUGH_INFO_STRATEGY_SPLIT.items())
    for i, (strat, share) in enumerate(nei_items):
        if i == len(nei_items) - 1:
            targets[strat] = n_not_enough_info - allocated
        else:
            targets[strat] = round(n_not_enough_info * share)
            allocated += targets[strat]

    return targets

# ---------------------------------------------------------------------------
# Gemini call + JSON parsing
# ---------------------------------------------------------------------------

def _call_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    rate_limiter: RateLimiter,
    retries: int = 3,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            # Wait if approaching rate limits
            rate_limiter.wait_if_needed()
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            
            # Record usage (estimate input tokens, actual output from response)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response.text.split()) * 1.3 if response.text else 0
            rate_limiter.record_request(int(input_tokens), int(output_tokens))
            
            # Log every 50 requests
            if rate_limiter.total_requests % 50 == 0:
                log.info(f"📊 API Stats: {rate_limiter.total_requests} requests, "
                        f"{rate_limiter.total_tokens:,} tokens "
                        f"(avg {rate_limiter.total_tokens/rate_limiter.total_requests:.0f}/req)")
            
            return response.text.strip()
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle rate limit errors specifically
            if "429" in error_msg or "rate limit" in error_msg or "quota" in error_msg:
                wait_time = min(120, 30 * (2 ** attempt))  # Exponential backoff, max 2 min
                log.warning(f"Rate limit hit (attempt {attempt+1}/{retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait = 2 ** attempt
                log.warning(f"Gemini call failed (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
                time.sleep(wait)
    return None


def _parse_json(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None

    # Strip accidental markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()

    # Quick accept common explicit 'null' responses
    if clean.strip().lower() in ("null", "none"):
        return {"new_claim_text": None, "rationale": "NO_CLAIM: model returned null"}

    # Try to locate a JSON object/array substring if the model returned extra text
    cleaned_for_search = clean
    obj_match = re.search(r"\{.*\}", cleaned_for_search, flags=re.DOTALL)
    arr_match = re.search(r"\[.*\]", cleaned_for_search, flags=re.DOTALL)

    candidate = None
    if obj_match:
        candidate = obj_match.group(0)
    elif arr_match:
        candidate = arr_match.group(0)
    else:
        # No JSON-like structure found; likely model returned plain text
        log.warning(f"No JSON object or array found in response | raw[:300]: {raw[:300]}")
        return None

    try:
        parsed = json.loads(candidate)
        if parsed is None:
            return {"new_claim_text": None, "rationale": "NO_CLAIM: model returned null"}
        return parsed
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e} | raw[:300]: {raw[:300]}")
        return None

# ---------------------------------------------------------------------------
# Single instance generation
# ---------------------------------------------------------------------------

def generate_instance(
    inst: dict,
    strategy: str,
    model: genai.GenerativeModel,
    rate_limiter: RateLimiter,
) -> Optional[dict]:
    """
    Apply `strategy` to `inst`, returning a new instance with rewritten
    claim_text / surrounding_context and updated true_outputs.
    Returns None if the LLM call fails or produces invalid output.
    """
    if not is_substantive_claim(inst.get("claim_text", "")):
        return None

    meta  = inst.get("citation_metadata", {})
    title = meta.get("title") or "Unknown"
    venue = meta.get("venue") or "Unknown"
    year  = meta.get("year")  or "Unknown"

    prompt = STRATEGY_PROMPTS[strategy].format(
        claim_text          = inst.get("claim_text", ""),
        title               = title,
        venue               = venue,
        year                = year,
    )

    raw    = _call_gemini(model, prompt, rate_limiter)
    parsed = _parse_json(raw)
    if not parsed:
        return None

    # If the LLM explicitly signalled a citation-only/no-claim (JSON null), skip silently
    if parsed.get("new_claim_text") is None:
        log.info(f"LLM returned NO_CLAIM for strategy={strategy}: {parsed.get('rationale')}")
        return None

    new_claim   = parsed.get("new_claim_text")
    rationale   = parsed.get("rationale", f"Alignment strategy: {strategy}.")

    # Sanity check: claim should have actually changed
    old_claim = inst.get("claim_text", "").strip()
    if new_claim.strip() == old_claim:
        log.warning(f"LLM returned unchanged claim_text for strategy={strategy}, skipping.")
        return None
    
    # Validate [CITATION] marker is preserved
    if "[CITATION]" not in new_claim:
        log.warning(f"LLM did not preserve [CITATION] marker in claim for strategy={strategy}, skipping.")
        return None
    
    # Replace old claim with new claim in surrounding_context
    old_context = inst.get("surrounding_context", "")
    if old_claim in old_context:
        new_context = old_context.replace(old_claim, new_claim, 1)
    else:
        log.warning(f"Could not find claim_text in surrounding_context for strategy={strategy}, skipping.")
        return None

    result = deep_copy(inst)
    result["claim_text"]          = new_claim
    result["surrounding_context"] = new_context
    result["true_outputs"].update({
        "true_existence":            1,
        "true_hallucination_category": None,   # existence is fine; only alignment is wrong
        "true_alignment":            STRATEGY_TO_ALIGNMENT[strategy],
        "expert_rationale":          f"[{strategy}] {rationale}",
    })
    return result

# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(
    positives:    list[dict],
    target_total: int,
    api_key:      str,
    seed:         int = 42,
    requests_per_minute: int = 28,
    tokens_per_minute: int = 14000,
) -> list[dict]:

    random.seed(seed)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    positives, skipped_non_claim = filter_substantive_positives(positives)
    if skipped_non_claim:
        log.info(
            f"Filtered out {skipped_non_claim} non-claim positives that only name a cited work."
        )
    if not positives:
        raise ValueError("No substantive claim-bearing positives available after filtering.")
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute)
    log.info(f"🚦 Rate limiter initialized: {rate_limiter.requests_per_minute} req/min, "
             f"{rate_limiter.tokens_per_minute:,} tokens/min")

    targets = compute_targets(target_total)
    log.info("Per-strategy targets:")
    for strat, n in targets.items():
        log.info(f"  {strat:<20} alignment={STRATEGY_TO_ALIGNMENT[strat]}   n={n}")

    # Build a cycling pool so we never run out of seeds
    pool = positives[:]
    random.shuffle(pool)
    pool_cycle = pool * ((target_total // max(len(pool), 1)) + 2)
    pool_idx   = 0

    def next_seed() -> dict:
        nonlocal pool_idx
        inst = deep_copy(pool_cycle[pool_idx % len(pool_cycle)])
        pool_idx += 1
        return inst

    generated: dict[str, list[dict]] = defaultdict(list)
    total_skipped = 0

    for strategy, count in targets.items():
        log.info(f"\n── Generating {count} × {strategy} ──")
        attempts     = 0
        max_attempts = count * 4   # allow generous retries for LLM failures

        while len(generated[strategy]) < count and attempts < max_attempts:
            attempts += 1
            inst   = next_seed()
            result = generate_instance(inst, strategy, model, rate_limiter)

            if result is not None:
                generated[strategy].append(result)
                done = len(generated[strategy])
                if done % 50 == 0 or done == count:
                    log.info(f"  {strategy}: {done}/{count}")
            else:
                total_skipped += 1

        if len(generated[strategy]) < count:
            log.warning(
                f"  {strategy}: only generated {len(generated[strategy])}/{count} "
                f"after {attempts} attempts."
            )

    # Final statistics
    rate_limiter.log_stats()
    
    all_instances = [inst for insts in generated.values() for inst in insts]
    random.shuffle(all_instances)

    log.info(f"\nGeneration complete — total: {len(all_instances)}, skipped/failed: {total_skipped}")

    # Summary table
    alignment_counts = Counter(i["true_outputs"]["true_alignment"] for i in all_instances)
    strategy_counts  = Counter(
        i["true_outputs"]["expert_rationale"].split("]")[0].lstrip("[")
        for i in all_instances
    )
    log.info(f"Alignment label distribution: {dict(alignment_counts)}")
    log.info(f"Strategy distribution: {dict(strategy_counts)}")

    return all_instances

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate alignment-hallucination negative instances using Gemini."
    )
    parser.add_argument("--input",   required=True,
                        help="Path to positives JSON file")
    parser.add_argument("--output",  required=True,
                        help="Output path for generated negatives JSON")
    parser.add_argument("--target",  type=int, default=2000,
                        help="Total negative instances to generate (default: 2000)")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"),
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--requests-per-minute", type=int, default=15, 
                        help="Max API requests per minute (default: 15, conservative for 30 limit)")
    parser.add_argument("--tokens-per-minute", type=int, default=30000,
                        help="Max tokens per minute (default: 30000, conservative for 30K limit)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Gemini API key required via --api-key or GEMINI_API_KEY env var.")

    log.info(f"Loading positives from {args.input} ...")
    positives = load_positives(args.input)
    log.info(f"Loaded {len(positives)} positive instances.")

    negatives = generate(
        positives, 
        args.target, 
        args.api_key, 
        args.seed,
        args.requests_per_minute,
        args.tokens_per_minute
    )

    with open(args.output, "w", encoding='utf-8') as f:
        json.dump(negatives, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(negatives)} alignment-negative instances → {args.output}")

    # Final distribution printout
    print("\n=== Alignment label distribution ===")
    counts = Counter(i["true_outputs"]["true_alignment"] for i in negatives)
    for label, desc in [(1, "misaligned"), (2, "uncertain/ambiguous")]:
        print(f"  true_alignment={label}  ({desc:<20})  {counts.get(label, 0):>5}")

    print("\n=== Strategy distribution ===")
    strat_counts = Counter(
        i["true_outputs"]["expert_rationale"].split("]")[0].lstrip("[")
        for i in negatives
    )
    # Print REFUTED first, then NOT ENOUGH INFO strategies
    print(f"  [1] {'REFUTED':<25}  {strat_counts.get('REFUTED', 0):>5}")
    for strat in NOT_ENOUGH_INFO_STRATEGY_SPLIT.keys():
        print(f"  [2] {strat:<25}  {strat_counts.get(strat, 0):>5}")


if __name__ == "__main__":
    main()

# """
# python .\alignment.py --input ..\..\data\UCT_dataset\UCT_all_postprocessed_new_filtered_2.json --target 3 --output ./corrected_alignment.json --api-key AIzaSyCQnmIh_p-MLsw_kUjpBynSaZC7d8v0z-k
# """