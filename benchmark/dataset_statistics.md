## Dataset Characteristics and Baselines

- **Dataset Size**: Minimum recommended size per split:
    - Training: 1,000+ instances
    - Validation: 200+ instances
    - Test: 500+ instances

- **Class Distribution**:
    - Existence: Target ~80-90% positive (exists), ~10-20% negative (not exists) to reflect real-world distribution
    - Alignment (for existing citations): Aim for representation across all levels with at least 15% per class
        - Supported: 40-50%
        - Partially supported: 25-35%
        - Unsupported: 15-25%
        - Uncertain: 5-15%

- **Difficulty Distribution**: Include varying difficulty levels:
    - Easy: Clear-cut cases with explicit evidence
    - Medium: Requires some inference or domain knowledge
    - Hard: Subtle distinctions, requires deep understanding

- **Inter-Annotator Agreement**: 
    - Target Cohen's Kappa ≥ 0.70 for existence labels
    - Target Cohen's Kappa ≥ 0.60 for alignment labels (allowing for adjacent agreement)
    - Document any systematic disagreements

- **Baseline Performance Expectations**:
    - Human expert performance should be documented as upper bound
    - Random baseline and majority class baseline for reference
    - Strong baseline: Fine-tuned language model (e.g., BERT, SciBERT for scientific citations)