"""
Alignment Hallucination Generator
===================================
Generates negative instances where the reference EXISTS (true_existence=1) but the
claim_text or surrounding_context does NOT align with what the cited paper says.

Alignment labels produced:
  - true_alignment=1  (misaligned / unsupported) : ~1333 instances  (~66.7%)
  - true_alignment=2  (uncertain / ambiguous)     :  ~667 instances  (~33.3%)

For each alignment type we define distinct misalignment strategies so the dataset
has diverse failure modes, not just one flavour of wrong claim:

  MISALIGNMENT STRATEGIES  (true_alignment=1)
  ─────────────────────────────────────────────
  A. CONTRADICT       – claim directly negates a finding of the paper
  B. OVERCLAIM        – claim exaggerates scope/magnitude beyond what the paper shows
  C. UNDERCLAIM       – claim downplays / omits a key finding
  D. ATTRIBUTE_SHIFT  – finding is real but attributed to wrong variable/group/condition
  E. SCOPE_SHIFT      – paper is about X; claim says it applies to Y (generalisation error)

  AMBIGUITY STRATEGIES  (true_alignment=2)
  ─────────────────────────────────────────
  F. PARTIAL_SUPPORT  – claim is partly right but leaves out critical qualifications
  G. VAGUE_CLAIM      – claim is worded so broadly it is neither clearly right nor wrong
  H. MISSING_CONTEXT  – claim could be true but the cited paper doesn't have enough info
                        to confirm or deny it

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

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-2.0-flash"

# How the 2000 instances split between the two alignment labels
ALIGNMENT_SPLIT = {
    1: 0.667,   # misaligned
    2: 0.333,   # uncertain / ambiguous
}

# How misaligned instances split across the 5 misalignment strategies
MISALIGNMENT_STRATEGY_SPLIT = {
    "CONTRADICT":      0.25,
    "OVERCLAIM":       0.22,
    "UNDERCLAIM":      0.18,
    "ATTRIBUTE_SHIFT": 0.18,
    "SCOPE_SHIFT":     0.17,
}

# How uncertain instances split across the 3 ambiguity strategies
AMBIGUITY_STRATEGY_SPLIT = {
    "PARTIAL_SUPPORT":  0.40,
    "VAGUE_CLAIM":      0.32,
    "MISSING_CONTEXT":  0.28,
}

# ---------------------------------------------------------------------------
# Prompt templates  (each returns a JSON object with new_claim, new_context, rationale)
# ---------------------------------------------------------------------------

# Shared JSON schema reminder appended to every prompt
_JSON_REMINDER = """
Return ONLY valid JSON — no markdown fences, no commentary outside the JSON.
Schema:
{
  "new_claim_text": "<rewritten claim sentence(s)>",
  "new_surrounding_context": "<rewritten surrounding context paragraph>",
  "rationale": "<one concise sentence explaining the misalignment type and how it was introduced>"
}
"""

# ── true_alignment=1 (misaligned) ───────────────────────────────────────────

PROMPT_CONTRADICT = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — CONTRADICT:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
DIRECTLY CONTRADICTS what the cited paper likely reports. The rewritten claim must:
- Negate or invert a specific finding, result, or conclusion
- Still sound like a natural academic sentence (not obviously absurd)
- Keep the citation to the same paper (metadata unchanged)
""" + _JSON_REMINDER

PROMPT_OVERCLAIM = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — OVERCLAIM:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
EXAGGERATES the paper's findings beyond what was actually reported. Examples of overclaiming:
- "showed improvement in one task" → "demonstrated universal improvement across all tasks"
- "correlated with" → "causally determined"
- "in a specific population" → "in all populations globally"
The rewritten claim must still sound like fluent academic writing.
""" + _JSON_REMINDER

PROMPT_UNDERCLAIM = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — UNDERCLAIM:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
DOWNPLAYS or OMITS a key finding of the paper. The rewritten claim should:
- Strip out a significant positive result, qualification, or nuance
- Make it seem the paper found less than it actually did
- Still read as a natural academic citation
""" + _JSON_REMINDER

PROMPT_ATTRIBUTE_SHIFT = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — ATTRIBUTE SHIFT:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that a real finding
is attributed to the WRONG variable, group, condition, or mechanism. Examples:
- "Model A outperformed Model B" → "Model B outperformed Model A"
- "effect observed in elderly patients" → "effect observed in pediatric patients"
- "driven by factor X" → "driven by factor Y"
The attribution swap should be subtle, not immediately obvious.
""" + _JSON_REMINDER

PROMPT_SCOPE_SHIFT = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — SCOPE SHIFT:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
applies the paper's findings to a DIFFERENT domain, population, or setting than the paper
actually studied (generalisation error). Examples:
- Paper studied English NLP → claim says "across all languages"
- Paper studied mice → claim says "in human clinical trials"
- Paper studied a narrow industrial process → claim says "in general manufacturing"
""" + _JSON_REMINDER

# ── true_alignment=2 (uncertain / ambiguous) ────────────────────────────────

PROMPT_PARTIAL_SUPPORT = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — PARTIAL SUPPORT:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
is PARTLY supported by the paper but omits critical qualifications or caveats that the
paper included. A careful reader with the paper in hand would be UNCERTAIN whether to
count this as supported or not. The claim must not be outright wrong — it just drops
important nuance (e.g., "under controlled conditions", "with p<0.05 but small effect size",
"only in a subset of participants").
""" + _JSON_REMINDER

PROMPT_VAGUE_CLAIM = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — VAGUE CLAIM:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
is so BROADLY or VAGUELY worded that it is impossible to determine from the paper alone
whether it is supported or not. The claim should sound meaningful but be non-committal
about specifics (method, magnitude, population, condition). A reader cannot confirm or
deny it without additional information.
""" + _JSON_REMINDER

PROMPT_MISSING_CONTEXT = """You are constructing a citation hallucination benchmark dataset.

Given the following citation instance:

CLAIM: {claim_text}
CONTEXT: {surrounding_context}
PAPER TITLE: {title}
VENUE: {venue}
YEAR: {year}

Task — MISSING CONTEXT:
Rewrite `claim_text` (and adjust `surrounding_context` consistently) so that the claim
could plausibly be true but the cited paper does NOT contain sufficient information to
verify it. This could be because:
- The claim refers to a follow-up experiment the paper doesn't describe
- The claim uses a metric or framework the paper doesn't measure
- The claim assumes background facts the paper takes for granted but never states
The claim must sound reasonable and on-topic, just unverifiable from the cited paper alone.
""" + _JSON_REMINDER

# Map strategy name → prompt template
STRATEGY_PROMPTS: dict[str, str] = {
    "CONTRADICT":      PROMPT_CONTRADICT,
    "OVERCLAIM":       PROMPT_OVERCLAIM,
    "UNDERCLAIM":      PROMPT_UNDERCLAIM,
    "ATTRIBUTE_SHIFT": PROMPT_ATTRIBUTE_SHIFT,
    "SCOPE_SHIFT":     PROMPT_SCOPE_SHIFT,
    "PARTIAL_SUPPORT": PROMPT_PARTIAL_SUPPORT,
    "VAGUE_CLAIM":     PROMPT_VAGUE_CLAIM,
    "MISSING_CONTEXT": PROMPT_MISSING_CONTEXT,
}

# Which strategies map to which alignment label
STRATEGY_TO_ALIGNMENT: dict[str, int] = {
    "CONTRADICT":      1,
    "OVERCLAIM":       1,
    "UNDERCLAIM":      1,
    "ATTRIBUTE_SHIFT": 1,
    "SCOPE_SHIFT":     1,
    "PARTIAL_SUPPORT": 2,
    "VAGUE_CLAIM":     2,
    "MISSING_CONTEXT": 2,
}

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

# ---------------------------------------------------------------------------
# Target count computation
# ---------------------------------------------------------------------------

def compute_targets(total: int) -> dict[str, int]:
    """
    Returns per-strategy target counts, e.g.:
      {"CONTRADICT": 333, "OVERCLAIM": 293, ..., "PARTIAL_SUPPORT": 267, ...}
    """
    n_misaligned = round(total * ALIGNMENT_SPLIT[1])
    n_uncertain  = total - n_misaligned

    targets: dict[str, int] = {}
    allocated_mis = 0
    mis_items = list(MISALIGNMENT_STRATEGY_SPLIT.items())
    for i, (strat, share) in enumerate(mis_items):
        if i == len(mis_items) - 1:
            targets[strat] = n_misaligned - allocated_mis
        else:
            targets[strat] = round(n_misaligned * share)
            allocated_mis += targets[strat]

    allocated_amb = 0
    amb_items = list(AMBIGUITY_STRATEGY_SPLIT.items())
    for i, (strat, share) in enumerate(amb_items):
        if i == len(amb_items) - 1:
            targets[strat] = n_uncertain - allocated_amb
        else:
            targets[strat] = round(n_uncertain * share)
            allocated_amb += targets[strat]

    return targets

# ---------------------------------------------------------------------------
# Gemini call + JSON parsing
# ---------------------------------------------------------------------------

def _call_gemini(
    model: genai.GenerativeModel,
    prompt: str,
    retries: int = 3,
) -> Optional[str]:
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"Gemini call failed (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    return None


def _parse_json(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    # Strip accidental markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    clean = re.sub(r"\s*```$",          "", clean, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
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
) -> Optional[dict]:
    """
    Apply `strategy` to `inst`, returning a new instance with rewritten
    claim_text / surrounding_context and updated true_outputs.
    Returns None if the LLM call fails or produces invalid output.
    """
    meta  = inst.get("citation_metadata", {})
    title = meta.get("title") or "Unknown"
    venue = meta.get("venue") or "Unknown"
    year  = meta.get("year")  or "Unknown"

    prompt = STRATEGY_PROMPTS[strategy].format(
        claim_text          = inst.get("claim_text", ""),
        surrounding_context = inst.get("surrounding_context", ""),
        title               = title,
        venue               = venue,
        year                = year,
    )

    raw    = _call_gemini(model, prompt)
    parsed = _parse_json(raw)

    if not parsed:
        return None

    new_claim   = parsed.get("new_claim_text")
    new_context = parsed.get("new_surrounding_context")
    rationale   = parsed.get("rationale", f"Alignment strategy: {strategy}.")

    if not new_claim or not new_context:
        log.warning(f"Missing fields in LLM response for strategy={strategy}")
        return None

    # Sanity check: claim should have actually changed
    if new_claim.strip() == inst.get("claim_text", "").strip():
        log.warning(f"LLM returned unchanged claim_text for strategy={strategy}, skipping.")
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
) -> list[dict]:

    random.seed(seed)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

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
            result = generate_instance(inst, strategy, model)

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
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Gemini API key required via --api-key or GEMINI_API_KEY env var.")

    log.info(f"Loading positives from {args.input} ...")
    positives = load_positives(args.input)
    log.info(f"Loaded {len(positives)} positive instances.")

    negatives = generate(positives, args.target, args.api_key, args.seed)

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
    for strat in list(MISALIGNMENT_STRATEGY_SPLIT) + list(AMBIGUITY_STRATEGY_SPLIT):
        label = STRATEGY_TO_ALIGNMENT[strat]
        print(f"  [{label}] {strat:<20}  {strat_counts.get(strat, 0):>5}")


if __name__ == "__main__":
    main()