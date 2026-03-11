"""
Existence Hallucination Generator
==================================
Generates negative instances (true_existence=0) from a pool of positive instances.

Taxonomy routing:
  RULE-BASED (no LLM needed):
    - AUTHOR_ERROR_ADD_DEL       -> add/remove authors from pool
    - AUTHOR_ERROR_PERTURBATION  -> char-level name mutations
    - META_ERROR_DOI             -> corrupt DOI checksum / format
    - META_ERROR_DATE            -> shift year by ±1–4
    - META_ERROR_VENUE           -> swap with same-field venue

  LLM-BASED (semantic understanding required):
    - TITLE_ERROR_SUBSTITUTE     -> replace with related but wrong title
    - TITLE_ERROR_PARAPHRASE     -> rephrase to subtly shift meaning
    - TITLE_ERROR_FULLY_FABRICATED   -> invent a plausible title from scratch
    - AUTHOR_ERROR_FULLY_FABRICATED  -> invent realistic author names
    - COMPOUND_ERROR             -> combine 2 single-error perturbations

Usage:
    python generate_existence_hallucinations.py \
        --input positives.json \
        --output negatives_existence.json \
        --target 1500 \
        --api-key YOUR_GOOGLE_API_KEY   # or set GOOGLE_API_KEY env var
"""

import json
import random
import re
import string
import copy
import argparse
import os
import time
import logging
from typing import Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Track and enforce rate limits for Gemma 27B API calls."""
    
    def __init__(self, requests_per_minute: int = 30, tokens_per_minute: int = 15000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Tracking windows (1 minute rolling)
        self.request_times = []
        self.token_usage = []  # (timestamp, token_count)
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
    def wait_if_needed(self):
        """Sleep if we're approaching rate limits."""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        self.token_usage = [(t, c) for t, c in self.token_usage if t > one_minute_ago]
        
        # Check request rate
        recent_requests = len(self.request_times)
        if recent_requests >= self.requests_per_minute - 2:  # Leave 2 request buffer
            sleep_time = 62  # Wait slightly over a minute
            log.warning(f"Approaching request limit ({recent_requests}/{self.requests_per_minute}). Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
            return
        
        # Check token rate
        recent_tokens = sum(c for _, c in self.token_usage)
        if recent_tokens >= self.tokens_per_minute - 1000:  # Leave 1K token buffer
            sleep_time = 62
            log.warning(f"Approaching token limit ({recent_tokens}/{self.tokens_per_minute}). Sleeping {sleep_time}s...")
            time.sleep(sleep_time)
            return
    
    def record_request(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record a successful API call."""
        now = datetime.now()
        self.request_times.append(now)
        
        total_tokens = input_tokens + output_tokens
        self.token_usage.append((now, total_tokens))
        
        self.total_requests += 1
        self.total_tokens += total_tokens
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
    
    def log_stats(self):
        """Log cumulative usage statistics."""
        log.info("=" * 60)
        log.info("API Usage Statistics:")
        log.info(f"  Total Requests:      {self.total_requests}")
        log.info(f"  Total Tokens:        {self.total_tokens:,}")
        log.info(f"    Input Tokens:      {self.total_input_tokens:,}")
        log.info(f"    Output Tokens:     {self.total_output_tokens:,}")
        log.info(f"  Avg Tokens/Request:  {self.total_tokens / max(1, self.total_requests):.1f}")
        log.info("=" * 60)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAXONOMY = [
    "TITLE_ERROR_SUBSTITUTE",
    "TITLE_ERROR_PARAPHRASE",
    "TITLE_ERROR_FULLY_FABRICATED",
    "AUTHOR_ERROR_ADD_DEL",
    "AUTHOR_ERROR_PERTURBATION",
    "AUTHOR_ERROR_FULLY_FABRICATED",
    "META_ERROR_DOI",
    "META_ERROR_DATE",
    "META_ERROR_VENUE",
    "COMPOUND_ERROR",
]

# Desired share of each taxonomy in the final output (must sum to 1.0)
TARGET_DISTRIBUTION = {
    "TITLE_ERROR_SUBSTITUTE":        0.12,
    "TITLE_ERROR_PARAPHRASE":        0.10,
    "TITLE_ERROR_FULLY_FABRICATED":  0.08,
    "AUTHOR_ERROR_ADD_DEL":          0.12,
    "AUTHOR_ERROR_PERTURBATION":     0.12,
    "AUTHOR_ERROR_FULLY_FABRICATED": 0.08,
    "META_ERROR_DOI":                0.12,
    "META_ERROR_DATE":               0.12,
    "META_ERROR_VENUE":              0.10,
    "COMPOUND_ERROR":                0.04,
}

# Venues per academic field (used for META_ERROR_VENUE swaps)
FIELD_VENUES: dict[str, list[str]] = {
    "Computer Science": [
        "NeurIPS", "ICML", "ICLR", "ACL", "CVPR", "AAAI", "ECCV",
        "EMNLP", "NAACL", "SIGIR", "WWW", "KDD", "VLDB", "OSDI",
        "IEEE Transactions on Neural Networks and Learning Systems",
        "Journal of Machine Learning Research",
        "ACM Computing Surveys",
    ],
    "Medicine": [
        "The Lancet", "NEJM", "JAMA", "BMJ", "Nature Medicine",
        "Annals of Internal Medicine", "PLOS Medicine",
        "Journal of Clinical Investigation", "Cell Host & Microbe",
        "American Journal of Respiratory and Critical Care Medicine",
    ],
    "Chemistry": [
        "Journal of the American Chemical Society", "Angewandte Chemie",
        "Nature Chemistry", "Chemical Science", "ACS Nano",
        "Organic Letters", "Inorganic Chemistry",
        "Journal of Physical Chemistry A", "Green Chemistry",
    ],
    "Biology": [
        "Cell", "Nature", "Science", "eLife", "PLOS Biology",
        "Molecular Cell", "Developmental Cell", "Current Biology",
        "Journal of Cell Biology", "Genetics",
    ],
    "Materials Science": [
        "Nature Materials", "Advanced Materials", "Acta Materialia",
        "ACS Applied Materials & Interfaces", "Journal of Materials Chemistry A",
        "npj Computational Materials", "Scripta Materialia",
        "Journal of the European Ceramic Society",
    ],
    "Physics": [
        "Physical Review Letters", "Physical Review B", "Nature Physics",
        "Journal of High Energy Physics", "Communications Physics",
        "New Journal of Physics", "EPL (Europhysics Letters)",
        "Reviews of Modern Physics",
    ],
    "Geology": [
        "Earth and Planetary Science Letters", "Journal of Geophysical Research",
        "Geology", "Geochimica et Cosmochimica Acta", "Tectonophysics",
        "Journal of Petrology", "Lithos", "Chemical Geology",
    ],
    "Psychology": [
        "Psychological Science", "Journal of Experimental Psychology",
        "Cognition", "Neuropsychologia", "Journal of Personality and Social Psychology",
        "Developmental Psychology", "Journal of Abnormal Psychology",
        "Behavior Research Methods",
    ],
}

# Flattened venue list used when we cannot infer the field from metadata
ALL_VENUES: list[str] = [v for vlist in FIELD_VENUES.values() for v in vlist]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_positives(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    # Accept either a bare list or {"instances": [...]}
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("instances", "data", "samples"):
            if key in data:
                return data[key]
    raise ValueError(f"Cannot parse JSON structure from {path}")


def deep_copy_instance(inst: dict) -> dict:
    return copy.deepcopy(inst)


def build_author_pool(positives: list[dict]) -> list[str]:
    """Collect all unique author names from the positive pool."""
    pool: set[str] = set()
    for inst in positives:
        for a in inst.get("citation_metadata", {}).get("authors", []):
            if a:
                pool.add(a)
    return list(pool)


def infer_venue_field(venue: Optional[str]) -> Optional[str]:
    """Try to map a venue string to one of the eight academic fields."""
    if not venue:
        return None
    venue_lower = venue.lower()
    for field, venues in FIELD_VENUES.items():
        for v in venues:
            if v.lower() in venue_lower or venue_lower in v.lower():
                return field
    return None


# ---------------------------------------------------------------------------
# Rule-based perturbation functions
# ---------------------------------------------------------------------------

def perturb_author_add_del(inst: dict, author_pool: list[str]) -> dict:
    """
    AUTHOR_ERROR_ADD_DEL
    Randomly add a foreign author or remove one existing author.
    """
    authors = inst["citation_metadata"]["authors"][:]
    action = random.choice(["add", "del"]) if len(authors) > 1 else "add"

    if action == "add":
        # Pick a name not already in the list
        foreign = [a for a in author_pool if a not in authors]
        if not foreign:
            foreign = ["J. Smith", "A. Kumar", "L. Chen"]
        new_author = random.choice(foreign)
        pos = random.randint(0, len(authors))
        authors.insert(pos, new_author)
        rationale = f"Author '{new_author}' was inserted at position {pos}; not an actual author of this paper."
    else:
        removed = random.choice(authors)
        authors.remove(removed)
        rationale = f"Author '{removed}' was removed from the author list."

    inst["citation_metadata"]["authors"] = authors
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "AUTHOR_ERROR_ADD_DEL",
        "true_alignment": None,
        "expert_rationale": rationale,
    })
    return inst


def _mutate_name(name: str) -> str:
    """Apply a small character-level mutation to a name string."""
    if not name:
        return name
    parts = name.split()
    if not parts:
        return name

    target = random.choice(parts)
    mutation = random.choice(["swap_chars", "replace_char", "initial_only", "double_char"])

    if mutation == "swap_chars" and len(target) >= 3:
        i = random.randint(0, len(target) - 2)
        mutated = target[:i] + target[i+1] + target[i] + target[i+2:]
    elif mutation == "replace_char" and len(target) >= 2:
        i = random.randint(0, len(target) - 1)
        replacement = random.choice(string.ascii_lowercase)
        mutated = target[:i] + replacement + target[i+1:]
    elif mutation == "initial_only":
        mutated = target[0] + "."
    elif mutation == "double_char" and len(target) >= 2:
        i = random.randint(0, len(target) - 1)
        mutated = target[:i] + target[i] + target[i] + target[i+1:]
    else:
        # Fallback: swap first two chars
        mutated = (target[1] + target[0] + target[2:]) if len(target) >= 2 else target

    return name.replace(target, mutated, 1)


def perturb_author_perturbation(inst: dict) -> dict:
    """
    AUTHOR_ERROR_PERTURBATION
    Apply a small character-level mutation to one author's name.
    """
    authors = inst["citation_metadata"]["authors"][:]
    if not authors:
        return None  # skip if no authors

    idx = random.randint(0, len(authors) - 1)
    original = authors[idx]
    mutated = _mutate_name(original)
    # Retry once if mutation didn't change anything
    if mutated == original:
        mutated = _mutate_name(original)
    authors[idx] = mutated

    inst["citation_metadata"]["authors"] = authors
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "AUTHOR_ERROR_PERTURBATION",
        "true_alignment": None,
        "expert_rationale": f"Author name perturbed: '{original}' -> '{mutated}'.",
    })
    return inst


def perturb_meta_doi(inst: dict) -> dict:
    """
    META_ERROR_DOI
    Corrupt the DOI — either flip digits in an existing one or fabricate a
    plausible-looking but invalid DOI.
    """
    doi = inst["citation_metadata"]["identifiers"].get("doi")

    if doi:
        # Flip one digit/char in the suffix
        parts = doi.split("/", 1)
        if len(parts) == 2:
            suffix = list(parts[1])
            digit_indices = [i for i, c in enumerate(suffix) if c.isdigit()]
            if digit_indices:
                idx = random.choice(digit_indices)
                original_digit = suffix[idx]
                new_digit = str((int(original_digit) + random.randint(1, 8)) % 10)
                suffix[idx] = new_digit
                corrupted_doi = parts[0] + "/" + "".join(suffix)
                rationale = f"DOI digit corrupted: '{doi}' -> '{corrupted_doi}'."
            else:
                corrupted_doi = doi + "x"  # append junk
                rationale = f"DOI suffix extended with invalid character: '{corrupted_doi}'."
        else:
            corrupted_doi = doi + ".invalid"
            rationale = f"DOI malformed: '{corrupted_doi}'."
    else:
        # Fabricate a DOI with a valid-looking prefix but random suffix
        prefix = random.choice(["10.1000", "10.1038", "10.1016", "10.1021", "10.1145"])
        suffix = "".join(random.choices(string.digits, k=4)) + \
                 "/" + \
                 "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        corrupted_doi = f"{prefix}/{suffix}"
        rationale = f"DOI fabricated with plausible format but non-existent reference: '{corrupted_doi}'."

    inst["citation_metadata"]["identifiers"]["doi"] = corrupted_doi
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "META_ERROR_DOI",
        "true_alignment": None,
        "expert_rationale": rationale,
    })
    return inst


def perturb_meta_date(inst: dict) -> dict:
    """
    META_ERROR_DATE
    Shift publication year by ±1–4 years (but keep it plausible: 1950–2024).
    """
    year = inst["citation_metadata"].get("year")
    if year is None:
        # Fabricate a plausible but wrong year from a random base
        year = random.randint(2000, 2022)

    delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    new_year = max(1960, min(2025, year + delta))
    # Ensure we actually changed it
    if new_year == year:
        new_year = year + (1 if year < 2025 else -1)

    inst["citation_metadata"]["year"] = new_year
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "META_ERROR_DATE",
        "true_alignment": None,
        "expert_rationale": f"Publication year shifted from {year} to {new_year}.",
    })
    return inst


def perturb_meta_venue(inst: dict) -> dict:
    """
    META_ERROR_VENUE
    Swap the venue with a different one from the same field (or any field if
    field cannot be inferred). Ensures the new venue is actually different.
    """
    original_venue = inst["citation_metadata"].get("venue")
    # field = infer_venue_field(original_venue)
    # candidates = FIELD_VENUES.get(field, ALL_VENUES) if field else ALL_VENUES
    candidates = ALL_VENUES

    # Filter out the current venue
    different = [v for v in candidates if v != original_venue]
    if not different:
        different = [v for v in ALL_VENUES if v != original_venue]

    new_venue = random.choice(different)
    inst["citation_metadata"]["venue"] = new_venue
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "META_ERROR_VENUE",
        "true_alignment": None,
        "expert_rationale": f"Venue replaced: '{original_venue}' -> '{new_venue}'.",
    })
    return inst


# ---------------------------------------------------------------------------
# LLM-based perturbation functions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a dataset construction assistant for an academic citation hallucination benchmark.
Your task is to perturb a real citation's metadata in a specified way to create a realistic-looking but incorrect citation.
Always respond with valid JSON only — no markdown fences, no explanation outside the JSON."""


def _call_llm(model, user_prompt: str, rate_limiter: RateLimiter, retries: int = 3) -> Optional[str]:
    """Call Gemini model and return the text content, or None on failure."""
    for attempt in range(retries):
        try:
            # Wait if approaching rate limits
            rate_limiter.wait_if_needed()
            
            full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            
            # Record usage (estimate input tokens, actual output from response)
            input_tokens = len(full_prompt.split()) * 1.3  # Rough estimate
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
                log.warning(f"LLM call failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(2 ** attempt)
    return None


def _parse_llm_json(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    # Strip accidental markdown fences
    clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        log.warning(f"JSON parse error: {e}\nRaw: {raw[:200]}")
        return None


def perturb_title_substitute(inst: dict, model, rate_limiter: RateLimiter) -> Optional[dict]:
    """
    TITLE_ERROR_SUBSTITUTE
    LLM replaces the title with a different real-sounding title in the same domain.
    """
    meta = inst["citation_metadata"]
    prompt = f"""Given this academic citation metadata:
Title: {meta.get('title')}
Venue: {meta.get('venue')}
Year: {meta.get('year')}
Authors: {', '.join(meta.get('authors', []))}

Task: Replace the title with a DIFFERENT but plausible academic title from the same research area.
The new title should sound like a real paper but refer to a different (possibly non-existent) work.
It must NOT be a paraphrase — it should feel like a completely different paper title.

Return JSON with exactly these keys:
{{
  "new_title": "<the substituted title>",
  "rationale": "<one sentence explaining how it differs>"
}}"""

    result = _parse_llm_json(_call_llm(model, prompt, rate_limiter))
    if not result or "new_title" not in result:
        return None

    inst["citation_metadata"]["title"] = result["new_title"]
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "TITLE_ERROR_SUBSTITUTE",
        "true_alignment": None,
        "expert_rationale": result.get("rationale", f"Title substituted with '{result['new_title']}'."),
    })
    return inst


def perturb_title_paraphrase(inst: dict, model, rate_limiter: RateLimiter) -> Optional[dict]:
    """
    TITLE_ERROR_PARAPHRASE
    LLM rewrites the title to subtly shift meaning while keeping surface similarity.
    """
    meta = inst["citation_metadata"]
    prompt = f"""Given this academic paper title:
"{meta.get('title')}"

Task: Rewrite it as a paraphrase that subtly changes the meaning — e.g., by altering scope,
changing a key term, or flipping a qualifier — so that it refers to a slightly different claim
than the original. Keep it plausible as an academic title. Do NOT just reorder words.

Return JSON with exactly these keys:
{{
  "new_title": "<the paraphrased title>",
  "rationale": "<one sentence describing what meaning was shifted and how>"
}}"""

    result = _parse_llm_json(_call_llm(model, prompt, rate_limiter))
    if not result or "new_title" not in result:
        return None

    inst["citation_metadata"]["title"] = result["new_title"]
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "TITLE_ERROR_PARAPHRASE",
        "true_alignment": None,
        "expert_rationale": result.get("rationale", f"Title paraphrased to '{result['new_title']}'."),
    })
    return inst


def perturb_title_fabricated(inst: dict, model, rate_limiter: RateLimiter) -> Optional[dict]:
    """
    TITLE_ERROR_FULLY_FABRICATED
    LLM invents a completely new title in the same general domain.
    """
    meta = inst["citation_metadata"]
    prompt = f"""Given this citation context:
Venue: {meta.get('venue')}
Year: {meta.get('year')}
Original title (for domain reference only, do NOT reuse): {meta.get('title')}

Task: Invent a completely fabricated academic paper title that:
1. Fits the venue/field
2. Sounds plausible and realistic
3. Does NOT resemble the original title at all

Return JSON with exactly these keys:
{{
  "new_title": "<the fabricated title>",
  "rationale": "<one sentence noting it is fully fabricated>"
}}"""

    result = _parse_llm_json(_call_llm(model, prompt, rate_limiter))
    if not result or "new_title" not in result:
        return None

    inst["citation_metadata"]["title"] = result["new_title"]
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "TITLE_ERROR_FULLY_FABRICATED",
        "true_alignment": None,
        "expert_rationale": result.get("rationale", f"Title fully fabricated: '{result['new_title']}'."),
    })
    return inst


def perturb_author_fabricated(inst: dict, model, rate_limiter: RateLimiter) -> Optional[dict]:
    """
    AUTHOR_ERROR_FULLY_FABRICATED
    LLM generates a realistic but entirely invented author list.
    """
    meta = inst["citation_metadata"]
    n_authors = max(1, len(meta.get("authors", [])))
    prompt = f"""Generate a list of {n_authors} realistic-sounding academic author names
for a paper in this venue: {meta.get('venue', 'an academic journal')}.
The names must be entirely fictional — do not reuse names from:
{', '.join(meta.get('authors', []))}

Return JSON with exactly this structure:
{{
  "new_authors": ["Author One", "Author Two", ...],
  "rationale": "Author list fully fabricated; none of these individuals authored this paper."
}}"""

    result = _parse_llm_json(_call_llm(model, prompt, rate_limiter))
    if not result or "new_authors" not in result:
        return None

    inst["citation_metadata"]["authors"] = result["new_authors"]
    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "AUTHOR_ERROR_FULLY_FABRICATED",
        "true_alignment": None,
        "expert_rationale": result.get("rationale", "Author list fully fabricated."),
    })
    return inst


# ---------------------------------------------------------------------------
# COMPOUND_ERROR: combine two single-error perturbations
# ---------------------------------------------------------------------------

# Taxonomy entries eligible as compound components (mix one rule-based + one LLM, or two rule-based)
COMPOUND_RULE_BASED = [
    "AUTHOR_ERROR_ADD_DEL",
    "AUTHOR_ERROR_PERTURBATION",
    "META_ERROR_DOI",
    "META_ERROR_DATE",
    "META_ERROR_VENUE",
]

COMPOUND_LLM_BASED = [
    "TITLE_ERROR_SUBSTITUTE",
    "TITLE_ERROR_PARAPHRASE",
    "TITLE_ERROR_FULLY_FABRICATED",
    "AUTHOR_ERROR_FULLY_FABRICATED",
]


def perturb_compound(
    inst: dict,
    author_pool: list[str],
    model,
    rate_limiter: RateLimiter,
) -> Optional[dict]:
    """
    COMPOUND_ERROR
    Apply two independent perturbations from different taxonomy categories.
    Prefer one rule-based + one LLM perturbation.
    """
    # Pick two distinct perturbation types
    first_type = random.choice(COMPOUND_RULE_BASED)
    # For the second, prefer LLM-based to maximise diversity
    second_type = random.choice(COMPOUND_LLM_BASED)

    applied = []

    def _apply(t: str, i: dict) -> Optional[dict]:
        if t == "AUTHOR_ERROR_ADD_DEL":
            return perturb_author_add_del(i, author_pool)
        elif t == "AUTHOR_ERROR_PERTURBATION":
            return perturb_author_perturbation(i)
        elif t == "META_ERROR_DOI":
            return perturb_meta_doi(i)
        elif t == "META_ERROR_DATE":
            return perturb_meta_date(i)
        elif t == "META_ERROR_VENUE":
            return perturb_meta_venue(i)
        elif t == "TITLE_ERROR_SUBSTITUTE":
            return perturb_title_substitute(i, model, rate_limiter)
        elif t == "TITLE_ERROR_PARAPHRASE":
            return perturb_title_paraphrase(i, model, rate_limiter)
        elif t == "TITLE_ERROR_FULLY_FABRICATED":
            return perturb_title_fabricated(i, model, rate_limiter)
        elif t == "AUTHOR_ERROR_FULLY_FABRICATED":
            return perturb_author_fabricated(i, model, rate_limiter)
        return None

    inst = _apply(first_type, inst)
    if inst is None:
        return None
    first_rationale = inst["true_outputs"]["expert_rationale"]
    applied.append(first_type)

    inst = _apply(second_type, inst)
    if inst is None:
        return None
    second_rationale = inst["true_outputs"]["expert_rationale"]
    applied.append(second_type)

    inst["true_outputs"].update({
        "true_existence": 0,
        "true_hallucination_category": "COMPOUND_ERROR",
        "true_alignment": None,
        "expert_rationale": (
            f"Two errors applied — [{applied[0]}]: {first_rationale} "
            f"[{applied[1]}]: {second_rationale}"
        ),
    })
    return inst


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def compute_targets(total: int) -> dict[str, int]:
    """Compute per-taxonomy target counts based on TARGET_DISTRIBUTION."""
    targets = {}
    allocated = 0
    items = list(TARGET_DISTRIBUTION.items())
    for i, (tax, share) in enumerate(items):
        if i == len(items) - 1:
            targets[tax] = total - allocated  # absorb rounding remainder
        else:
            targets[tax] = round(total * share)
            allocated += targets[tax]
    return targets


def generate(
    positives: list[dict],
    target_total: int,
    api_key: str,
    seed: int = 42,
    requests_per_minute: int = 28,
    tokens_per_minute: int = 14000,
) -> list[dict]:
    random.seed(seed)
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemma-3-27b-it')
    author_pool = build_author_pool(positives)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(requests_per_minute=requests_per_minute, tokens_per_minute=tokens_per_minute)
    log.info(f"🚦 Rate limiter initialized: {rate_limiter.requests_per_minute} req/min, "
             f"{rate_limiter.tokens_per_minute:,} tokens/min")

    targets = compute_targets(target_total)
    log.info(f"Target distribution: {targets}")

    generated: dict[str, list[dict]] = defaultdict(list)
    skipped = 0

    # Shuffle positives so we draw diverse seeds
    pool = positives[:]
    random.shuffle(pool)
    pool_cycle = pool * ((target_total // len(pool)) + 2)  # repeat if needed
    pool_idx = 0

    def next_instance() -> dict:
        nonlocal pool_idx
        inst = deep_copy_instance(pool_cycle[pool_idx % len(pool_cycle)])
        pool_idx += 1
        return inst

    for taxonomy, count in targets.items():
        log.info(f"Generating {count} instances for {taxonomy} ...")
        attempts = 0
        max_attempts = count * 3

        while len(generated[taxonomy]) < count and attempts < max_attempts:
            attempts += 1
            inst = next_instance()

            try:
                if taxonomy == "AUTHOR_ERROR_ADD_DEL":
                    result = perturb_author_add_del(inst, author_pool)
                elif taxonomy == "AUTHOR_ERROR_PERTURBATION":
                    result = perturb_author_perturbation(inst)
                elif taxonomy == "META_ERROR_DOI":
                    result = perturb_meta_doi(inst)
                elif taxonomy == "META_ERROR_DATE":
                    result = perturb_meta_date(inst)
                elif taxonomy == "META_ERROR_VENUE":
                    result = perturb_meta_venue(inst)
                elif taxonomy == "TITLE_ERROR_SUBSTITUTE":
                    result = perturb_title_substitute(inst, model, rate_limiter)
                elif taxonomy == "TITLE_ERROR_PARAPHRASE":
                    result = perturb_title_paraphrase(inst, model, rate_limiter)
                elif taxonomy == "TITLE_ERROR_FULLY_FABRICATED":
                    result = perturb_title_fabricated(inst, model, rate_limiter)
                elif taxonomy == "AUTHOR_ERROR_FULLY_FABRICATED":
                    result = perturb_author_fabricated(inst, model, rate_limiter)
                elif taxonomy == "COMPOUND_ERROR":
                    result = perturb_compound(inst, author_pool, model, rate_limiter)
                else:
                    result = None
            except Exception as e:
                log.warning(f"  Error during {taxonomy}: {e}")
                result = None

            if result is not None:
                generated[taxonomy].append(result)
                if len(generated[taxonomy]) % 50 == 0:
                    log.info(f"  {taxonomy}: {len(generated[taxonomy])}/{count}")
            else:
                skipped += 1

    # Final statistics
    rate_limiter.log_stats()
    
    all_instances = [inst for insts in generated.values() for inst in insts]
    random.shuffle(all_instances)

    actual_counts = {k: len(v) for k, v in generated.items()}
    log.info(f"Generation complete. Total: {len(all_instances)}, Skipped/failed: {skipped}")
    log.info(f"Actual counts per taxonomy: {actual_counts}")

    return all_instances


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate existence-hallucination negative instances.")
    parser.add_argument("--input",   required=True,  help="Path to positives JSON file")
    parser.add_argument("--output",  required=True,  help="Output path for generated negatives JSON")
    parser.add_argument("--target",  type=int, default=1500, help="Total negative instances to generate")
    parser.add_argument("--api-key", default=os.environ.get("GOOGLE_API_KEY"), help="Google API key for Gemini")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--requests-per-minute", type=int, default=28, 
                        help="Max API requests per minute (default: 28, conservative for 30 limit)")
    parser.add_argument("--tokens-per-minute", type=int, default=14000,
                        help="Max tokens per minute (default: 14000, conservative for 15K limit)")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Google API key required via --api-key or GOOGLE_API_KEY env var.")

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

    log.info(f"Saved {len(negatives)} negative instances to {args.output}")

    # Print final taxonomy distribution summary
    counts = Counter(i["true_outputs"]["true_hallucination_category"] for i in negatives)
    print("\n=== Final taxonomy distribution ===")
    for tax in TAXONOMY:
        print(f"  {tax:<40} {counts.get(tax, 0):>5}")


if __name__ == "__main__":
    main()