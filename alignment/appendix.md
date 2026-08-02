# Appendix: Prompts and Experimental Settings

## A.1 Prompts

### A.1.1 Adversarial Negative Sample Generation (adversarial_generator-2.py)

The adversarial module follows an Analyzer → Generator → Discriminator → Filter pipeline
to create hard negative claims via four semantic drift types:
`over_claim`, `context_shift`, `reversal`, and `tangential`.

#### A.1.1.1 Applicability Analyzer Prompt

```text
You are an expert linguistic analyst evaluating the feasibility of generating adversarial scientific claims.
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
{
    "evaluations": {
        "over_claim": {"is_applicable": true/false, "reason": "brief rationale"},
        "context_shift": {"is_applicable": true/false, "reason": "brief rationale"},
        "reversal": {"is_applicable": true/false, "reason": "brief rationale"},
        "tangential": {"is_applicable": true/false, "reason": "brief rationale"}
    }
}
```

#### A.1.1.2 Adversarial Generator Prompt

The drift-strategy-specific instruction is injected per strategy:

- **over_claim**: "Exaggerate the findings. Take a modest or specific claim and inflate it into a universal, absolute, or highly generalized breakthrough. Keep the same academic tone."
- **context_shift**: "Shift the context. The original evidence holds true under specific conditions (e.g., specific datasets, domains, or limitations). Rewrite the claim to apply these findings to a completely different, unsupported domain or condition."
- **reversal**: "Reverse the conclusion. Flip the causal relationship, the comparison results (e.g., 'A is faster than B' to 'B is faster than A'), or negate the primary finding, WHILE using mostly the same vocabulary as the original claim."
- **tangential**: "Introduce a tangential hallucination. Write a claim that shares the same overarching topic, but asserts a specific application, methodology, or result that is ABSOLUTELY NOT mentioned in the evidence."

```text
You are an adversarial AI researcher generating robustness tests.
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
{
    "adversarial_claim": "<your generated deceptive claim>",
    "rationale": "<brief explanation of how it fulfills the drift strategy>"
}
```

#### A.1.1.3 Zero-shot Discriminator (Judge) Prompt

```text
You are a strict, impartial peer reviewer auditing scientific citations.
Evaluate whether the following Claim is supported by the provided Evidence.

Claim: "{adversarial_claim}"

Evidence:
{evidence}

Classify the alignment into ONE of the following categories:
- SUPPORTED: The claim is fully backed by the evidence.
- PARTIALLY: The claim exaggerates or only partially aligns with the evidence.
- UNSUPPORTED: The claim contradicts or shifts the context of the evidence.
- UNCERTAIN: The claim introduces information completely absent from the evidence.

Return your response strictly as a JSON object with this structure:
{
    "label": "<SUPPORTED | PARTIALLY | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<brief explanation>"
}
```

### A.1.2 Retrieval Module (benchmark_builder-2.py)

#### A.1.2.1 HyDE Augmentation Prompt

```text
Generate a short hypothetical scientific passage that could plausibly appear in a paper relevant to the claim below.

Requirements:
1. Preserve the key entities, quantities, methods, and outcomes from the claim.
2. Write in a neutral academic style.
3. Return only the passage text, with no bullets, labels, or explanation.

Claim:
"{claim}"
```

#### A.1.2.2 Abstractive Synthesis Prompt

```text
You are an expert scientific Research Assistant.
Your task is to review raw text chunks extracted from an academic paper and synthesize any contextual information relevant to the topics, entities, or methodologies mentioned in the Claim.

Claim: "{claim}"

Raw Evidence Chunks:
{raw_context}

Instructions:
1. Extract and summarize ANY information from the chunks that is topically related to the Claim.
2. Provide a neutral, objective summary (2-4 sentences) of what the chunks actually say about the topic. Do not evaluate whether the claim is true or false. Just report the facts found in the text.
3. Ignore formatting noise, markdown tags, or broken equations.
4. ONLY if the chunks discuss entirely different subjects and share ZERO entities or semantic overlap with the Claim, respond with EXACTLY: NO_EVIDENCE

Synthesis:
```

### A.1.3 Adjudication Module (benchmark_evaluator-3.py)

#### A.1.3.1 Open-book Evidence Classification Prompt

```text
You are an expert fact-checking assistant evaluating semantic alignment in academic texts. 
Analyze whether the provided evidence from a reference paper supports the given claim.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked. 
Focus strictly on the relationship between the claim's core assertion regarding this reference and the provided evidence.

Claim: "{claim}"

Surrounding Context: "{context}"

---
Abstractive Synthesis (Structural Denoised Context):
{synthesis_block}

Guidance: The Abstractive Synthesis above is a distilled summary of evidence.
If it tells us the evidence does not fully support the claim (or empty), the classification should lean toward
UNCERTAIN (no information) or UNSUPPORTED (chunks contradict), rather than SUPPORTED.
If it is empty then it is likely that the claim is not supported by the reference or UNCERTAIN.  

Raw Extractive Evidence Chunks:
{evidence_block}
---

Classify the alignment into EXACTLY ONE of the following 4 categories:
- SUPPORTED: The evidence directly backs the claim or a close paraphrase without logical gaps.
- PARTIALLY_SUPPORTED: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the evidence.
- UNSUPPORTED: The evidence explicitly contradicts the claim or has related findings but shifted in context.
- UNCERTAIN: There is no relevant information in the evidence to assess the claim.

Return your response as valid JSON with this exact structure:
{
    "classification": "<SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed evidence-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}

Return ONLY valid JSON, no other text.
```

#### A.1.3.2 Closed-book Classification Prompt

```text
You are an expert fact-checking assistant evaluating semantic alignment in academic texts.
You are given a claim and the metadata of the reference paper it cites.
Based on the reference metadata and your own knowledge of the paper, classify whether
the claim is supported by the reference.

Important: The token [CITATION] in the claim is a placeholder marking the exact reference being checked.
Focus strictly on the relationship between the claim's core assertion regarding this reference.

Claim: "{claim}"

Surrounding Context: "{context}"

---
Reference Paper:
{ref_lines}---

Classify the alignment into EXACTLY ONE of the following 4 categories:
- SUPPORTED: The reference directly backs the claim or a close paraphrase without logical gaps.
- PARTIALLY_SUPPORTED: The claim exaggerates the findings, applies them to an unsupported domain (over-claiming), or only aligns with a fraction of the reference.
- UNSUPPORTED: The reference explicitly contradicts the claim or has related findings but shifted in context.
- UNCERTAIN: You do not have enough knowledge of the reference to assess the claim.

Return your response as valid JSON with this exact structure:
{
    "classification": "<SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCERTAIN>",
    "reasoning": "<detailed knowledge-grounded rationale>",
    "confidence_score": <float between 0.0 and 1.0>
}

Return ONLY valid JSON, no other text.
```

## A.2 Experimental Settings

### A.2.1 Hardware & Software Environment

| Category | Setting |
|---|---|
| Operating System | Microsoft Windows 11 Home (build 10.0.22631) |
| CPU | AMD Ryzen 7 8845H (8 cores / 16 threads) |
| Memory | 27.8 GB RAM |
| GPU | AMD Radeon 780M (integrated, 4 GB) |
| Python | 3.13.9 (Miniconda) |
| LLM Provider | Together AI (`Qwen/Qwen2.5-7B-Instruct-Turbo`) |
| Key packages | together 2.16.0, openai 2.38.0, sentence-transformers 5.5.1, torch 2.9.1, flashrank 0.2.10, chromadb 1.5.9, langchain 1.3.1, langchain-community 0.4.2, PyMuPDF 1.27.1, numpy 2.4.1, onnxruntime 1.26.0 |

### A.2.2 Common LLM Hyperparameters (all modules)

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct-Turbo (Together AI) |
| Temperature | 0.7 |
| max_tokens | 2048 |
| Max retries (exponential backoff, base 1.5 s) | 3 |

### A.2.3 Adversarial Module (adversarial_generator-2.py)

| Parameter | Value |
|---|---|
| Concurrency | 100–200 |
| Similarity threshold (cosine, original vs. adversarial claim) | 0.80 |
| Max generation attempts per drift type | 3 |
| Embedding model | intfloat/multilingual-e5-large-instruct |
| Semantic drift types | over_claim, context_shift, reversal, tangential |

### A.2.4 Retrieval Module (benchmark_builder-2.py)

| Parameter | Value |
|---|---|
| Concurrency | 100–200 |
| Retrieval mode | rrf (dense + BM25 sparse + Reciprocal Rank Fusion) |
| Dense embedding model | all-mpnet-base-v2 (local SentenceTransformer) |
| Reranker (FlashRank) | ms-marco-MultiBERT-L-12 |
| RRF constant k | 60 |
| Initial candidates (top_k_initial) | 15 |
| Final evidence chunks (top_k_final) | 3 |
| Relevance score threshold | 0.85 |
| Chunk size / overlap (character splitting) | 750 / 150 |
| HyDE augmentation | disabled |
| PDF-to-markdown method | PyMuPDF |

### A.2.5 Adjudication Module (benchmark_evaluator-3.py)

| Parameter | Value |
|---|---|
| Concurrency | 100–200 |
| Evaluation mode | open-book (hybrid evidence: extractive chunks + abstractive synthesis) |
| Taxonomy | SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / UNCERTAIN |
