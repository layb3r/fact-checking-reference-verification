# Unified Citation Trustworthiness Benchmark

## Task Definition

Given a claim and its cited reference, determine whether the citation exists and whether it supports the claim. Task formulation:
- Input: A claim and its cited reference.
- Output: Two labels:
    1. Existence label (binary classification): 
        - 1: Citation exists with correct metadata
        - 0: Citation does not exist or has incorrect metadata
        - Correctness criteria: A citation is considered to exist (label=1) if:
            - Title matches exactly or differs only by minor formatting (punctuation, capitalization)
            - Publication year matches exactly
            - At least one author surname matches
            - If DOI is provided, it must be correct
            - Minor variations in venue names are acceptable
    
    2. Claim-citation alignment label (multi-class classification, ordinal scale from most to least supportive):
        - 0: supported (fully aligned): Citation claim is fully supported by the reference
        - 1: partially supported (partially aligned): Citation claim has partial support with some discrepancies
        - 2: unsupported (misaligned): Citation claim contradicts or is not supported by the reference
        - 3: uncertain (ambiguous alignment): Insufficient information to determine support level

    3. (Optional) Explanation: A natural language explanation justifying the existence and alignment labels, highlighting key evidence from the cited reference that supports the model's decision.

- Output Workflow: 
    - When predicted_existence=1: Model must output an alignment label (0-3)
    - When predicted_existence=0: Model outputs N/A or null for alignment (no alignment check needed)

## Data Instance Definition
Each data instance consists of:
- Inputs:
    - Claim: A statement that is being evaluated for trustworthiness.
    - Cited Reference Metadata: Structured information about the cited reference, including:
        - Title
        - Authors
        - Publication Year
        - Journal/Conference Name
        - DOI (if available)
        - ...
    - Citation Context: Relevant text snippets from the cited reference that are pertinent to the claim, such as the abstract, conclusion, or specific sections that discuss the claim.
        - When existence=1: Provide actual content from the correctly cited reference
        - When existence=0: May provide empty context, content from a similar but different paper, or closest match to help the model detect the mismatch 
    
- Outputs:
    - Existence Label: Binary label (0 or 1) indicating whether the citation exists and has correct metadata
    - Claim-Citation Alignment Label: Multi-class label (0-3) or N/A
    - (Optional) Explanation: Natural language justification for the predicted labels, citing specific evidence from the reference

## Metrics

- Evaluate the existence label (binary classification):
    - Accuracy
    - Precision
    - Recall
    - F1-score

- Evaluate the claim-citation alignment label (conditional multi-class classification):
    - Evaluation Subset: Only evaluate instances where true_existence=1 AND predicted_existence=1
    - Rationale: Models only output alignment when they predict existence=1; cases where predicted_existence=0 are handled by hierarchical metrics
    - Accuracy
    - Precision (macro and micro)
    - Recall (macro and micro)
    - F1-score (macro and micro)
    - Per-class F1 scores (for each alignment level: supported, partially supported, unsupported, uncertain)
    - Confusion matrix to analyze misclassification patterns
    - Coverage: Report percentage of true positives (true_existence=1) where model also predicted existence=1

- Hierarchical F1 (Full-Path Hierarchical Evaluation):
    - Evaluates the complete prediction pipeline (existence -> alignment)
    - Uses ordinal relationship of alignment labels: supported (0) > partially supported (1) > unsupported (2) > uncertain (3)
    - True Positives (TP): 
        - true_existence=1 AND pred_existence=1 AND alignment_label_correct
    - False Positives (FP): 
        - Case 1: pred_existence=1 AND true_existence=0 (hallucinated citation)
        - Case 2: pred_existence=1 AND true_existence=1 AND predicted_alignment_is_better_than_true_alignment (over-optimistic)
        - "Better" means lower ordinal value (e.g., predicting supported when truth is partially supported)
    - False Negatives (FN): 
        - Case 1: pred_existence=0 AND true_existence=1 (missed citation - alignment not evaluated since model didn't predict it)
        - Case 2: pred_existence=1 AND true_existence=1 AND predicted_alignment_is_worse_than_true_alignment (under-pessimistic)
        - "Worse" means higher ordinal value (e.g., predicting unsupported when truth is partially supported)
    - True Negatives (TN): 
        - true_existence=0 AND pred_existence=0 (correctly identified non-existent citation)
    - Hierarchical Precision: TP / (TP + FP)
    - Hierarchical Recall: TP / (TP + FN)
    - Hierarchical F1: 2 × (Precision × Recall) / (Precision + Recall)
    - Note: This metric naturally handles the workflow where alignment is only predicted when pred_existence=1

- Hierarchical Accuracy: Percentage of instances where both existence AND alignment labels are correct (only for instances with true_existence=1)

- Calibration Metrics (if models output confidence scores):
    - Expected Calibration Error (ECE) for both existence and alignment predictions
    - Reliability diagrams

- Explanation Metrics:
    - Faithfulness: Measure whether the explanation is grounded in the provided citation context (using attribution scores or NLI-based verification)
    - Informativeness: Measure whether the explanation contains specific evidence (e.g., presence of direct quotes, specific facts)
    - Fluency: Measure the grammatical quality and coherence of explanations
    - Human Evaluation: Subset of explanations rated by human annotators for quality and helpfulness
    - Optional: ROUGE/BLEU scores against reference explanations (if available)