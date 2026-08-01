# Failure-Mode Analysis of Misclassified Instances (FINAL_subsample_300)

Generated: 2026-08-01T11:32:09.704079
Judge model: `google/gemma-3-12b-it` (OpenRouter, temperature 0.2)

## Overview

| Model | Accuracy (sub-300) | Misclassified | Rate | Analyzed |
|---|---|---|---|---|
| Gemma-3-12B | 0.6600 | 102 | 34.0% | 102 |
| Llama-3.1-8B | 0.6800 | 96 | 32.0% | 96 |
| Mistral-Nemo | 0.5800 | 126 | 42.0% | 126 |
| GPT-OSS-20B | 0.6100 | 117 | 39.0% | 117 |
| Qwen2.5-7B | 0.7633 | 71 | 23.7% | 71 |

## Failure-Mode Definitions

- **semantic_ambiguity**: The claim is phrased in a way that is inherently ambiguous, making it unclear what specific factual assertion is being made or what evidence would be relevant.
- **evidence_retrieval_failure**: The retrieved evidence chunks lack the necessary information to properly evaluate the claim — the relevant passage was not retrieved or does not exist in the document.
- **annotation_error**: The ground-truth label appears incorrect or inconsistent with the available evidence, suggesting a mistake in the original dataset annotation.
- **complex_inferential_chain**: The claim requires multi-step reasoning across scattered pieces of evidence, and the model failed to connect all the necessary facts correctly.
- **inference_hallucination**: The model invented facts, made unsupported logical leaps, or cited evidence that does not actually support its conclusion.

## Error-Type Distribution per Model

| Model | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | parse_failure | unknown |
|---|---|---|---|---|---|---|---|
| Gemma-3-12B | 12 (11.8%) | 21 (20.6%) | 0 (0.0%) | 33 (32.4%) | 36 (35.3%) | 0 (0.0%) | 0 (0.0%) |
| Llama-3.1-8B | 6 (6.2%) | 35 (36.5%) | 0 (0.0%) | 25 (26.0%) | 29 (30.2%) | 0 (0.0%) | 0 (0.0%) |
| Mistral-Nemo | 6 (4.8%) | 54 (42.9%) | 0 (0.0%) | 24 (19.0%) | 42 (33.3%) | 0 (0.0%) | 0 (0.0%) |
| GPT-OSS-20B | 3 (2.6%) | 55 (47.0%) | 0 (0.0%) | 24 (20.5%) | 35 (29.9%) | 0 (0.0%) | 0 (0.0%) |
| Qwen2.5-7B | 2 (2.8%) | 37 (52.1%) | 0 (0.0%) | 14 (19.7%) | 18 (25.4%) | 0 (0.0%) | 0 (0.0%) |
| **All models** | **29** (5.7%) | **202** (39.5%) | **0** (0.0%) | **120** (23.4%) | **160** (31.2%) | **0** (0.0%) | **0** (0.0%) |

## Per-Label Error-Type Breakdown

### Gemma-3-12B

| True label | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | Total |
|---|---|---|---|---|---|---|
| SUPPORTED | 2 | 7 | 0 | 12 | 2 | 23 |
| UNSUPPORTED | 0 | 3 | 0 | 2 | 12 | 17 |
| UNCERTAIN | 10 | 11 | 0 | 19 | 22 | 62 |

### Llama-3.1-8B

| True label | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | Total |
|---|---|---|---|---|---|---|
| SUPPORTED | 2 | 10 | 0 | 14 | 9 | 35 |
| PARTIALLY_SUPPORTED | 0 | 0 | 0 | 0 | 2 | 2 |
| UNSUPPORTED | 0 | 10 | 0 | 1 | 7 | 19 |
| UNCERTAIN | 4 | 15 | 0 | 10 | 11 | 40 |

### Mistral-Nemo

| True label | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | Total |
|---|---|---|---|---|---|---|
| SUPPORTED | 1 | 12 | 0 | 17 | 11 | 41 |
| PARTIALLY_SUPPORTED | 0 | 1 | 0 | 0 | 0 | 1 |
| UNSUPPORTED | 0 | 19 | 0 | 1 | 10 | 30 |
| UNCERTAIN | 5 | 22 | 0 | 6 | 21 | 54 |

### GPT-OSS-20B

| True label | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | Total |
|---|---|---|---|---|---|---|
| SUPPORTED | 3 | 25 | 0 | 21 | 11 | 60 |
| PARTIALLY_SUPPORTED | 0 | 3 | 0 | 0 | 3 | 6 |
| UNSUPPORTED | 0 | 11 | 0 | 1 | 8 | 20 |
| UNCERTAIN | 0 | 16 | 0 | 2 | 13 | 31 |

### Qwen2.5-7B

| True label | semantic_ambiguity | evidence_retrieval_failure | annotation_error | complex_inferential_chain | inference_hallucination | Total |
|---|---|---|---|---|---|---|
| SUPPORTED | 0 | 4 | 0 | 10 | 4 | 18 |
| PARTIALLY_SUPPORTED | 0 | 2 | 0 | 0 | 3 | 5 |
| UNSUPPORTED | 0 | 2 | 0 | 0 | 5 | 7 |
| UNCERTAIN | 2 | 29 | 0 | 4 | 6 | 41 |

## Error Overlap Across Models

Instances misclassified by multiple models (top 15 by agreement count):

Instances missed by **1** model: 44, **2**: 49, **3**: 32, **4**: 36, **5**: 26

| Instance ID | Models that failed |
|---|---|
| 1001 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 1886 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 3407 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 4908 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 5317 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 308 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 540 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 928 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 1239 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 2169 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 2401 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 2479 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 2634 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 2789 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |
| 3719 | Gemma-3-12B, Llama-3.1-8B, Mistral-Nemo, GPT-OSS-20B, Qwen2.5-7B |

## Representative Examples

### Gemma-3-12B — semantic_ambiguity (12)

- **Instance 2204**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: These tools include code interpreters and specialized vision models [CITATION].
  - Judge rationale: The term "tools" is vague and open to interpretation; the claim doesn't specify what kind of tools are being referred to. The model's reasoning correctly points out that the evidence describes components within a system rather than external tools, but the ambiguity in the claim makes it difficult to definitively support or refute.
- **Instance 3030**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: This is reflected by the fact that exact ground state properties as well as the single-particle charged excitation spectrum can be obtained from the same single-particle Green's function. <cit.> Recently there has been significant interest in identifying and leveraging connections between coupled-cl
  - Judge rationale: The claim uses the term 'exact' which is open to interpretation in scientific contexts; it's unclear if it refers to a theoretical limit or a practical approximation. The model's focus on the lack of explicit 'exact' language, while valid, misses the broader point of semantic ambiguity in the claim itself.

### Gemma-3-12B — evidence_retrieval_failure (21)

- **Instance 833**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: Comparison with SoTA Text-to-Motion Methods. We evaluate EasyTune on text-to-motion generation using MLD <cit.> and MLD++ <cit.> as base models, comparing with state-of-the-art methods on the HumanML3D <cit.> and KIT-ML [CITATION] datasets, as shown in Tab. <ref> and <ref> (in App.
  - Judge rationale: The retrieved evidence focuses solely on the KIT Motion-Language Dataset itself, providing details about its creation and purpose, but it lacks any information regarding EasyTune or the comparative evaluation of text-to-motion methods mentioned in the claim.
- **Instance 907**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: While researchers have addressed this using non-negative feature maps like 1+ELU and ReLU [CITATION], these stability-ensuring modifications still underperform compared to standard softmax attention <cit.>.
  - Judge rationale: The retrieved evidence discusses alternative feature maps and computational efficiency but does not contain any direct comparison of the performance of 1+ELU and ReLU with standard softmax attention, which is the core of the claim.

### Gemma-3-12B — complex_inferential_chain (33)

- **Instance 223**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: In contrast, the PI and IPW estimators typically incur first-order bias terms driven by μ̂_a-μ_a and π̂-π, respectively, and thus generally require substantially stronger nuisance rates to achieve √(n)-consistent inference under flexible nonparametric learning [CITATION].
  - Judge rationale: The claim makes a specific assertion about PI and IPW estimators and their nuisance rate requirements, but the evidence primarily discusses general principles of bias terms and consistency. The model correctly identifies that the evidence touches on relevant concepts but fails to establish the direct link and magnitude of the effect on PI and IPW, requiring a more complex inferential chain than the evidence allows.
- **Instance 1789**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: While the lack of transparency in dataset creation is an open challenge in ML research [CITATION], persona datasets are a particularly sensitive domain.
  - Judge rationale: The model correctly identifies that the evidence discusses transparency issues and sensitivity, but fails to fully connect that persona datasets are *particularly* sensitive *due to* those transparency issues, as the evidence primarily focuses on the absence of PII and offensive content. A more nuanced understanding of the relationship between transparency and sensitivity in this context is required to fully support the claim.

### Gemma-3-12B — inference_hallucination (36)

- **Instance 2008**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: Following [CITATION], we choose a self-persona configuration where only the speaker’s persona is visible.
  - Judge rationale: The model's reasoning claims the evidence doesn't detail specific configurations used in Dinan et al. (2019), but the claim itself doesn't require detailed configuration information; it simply states a configuration was chosen. The evidence mentioning ConvAI2 and Dinan et al. is sufficient to suggest the claim is plausible, and the model's conclusion is not supported by the evidence.
- **Instance 5661**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Second, <cit.> argued that tidally driven crack opening exposes a subsurface ocean directly to space, later reinforced by [CITATION].
  - Judge rationale: The model's reasoning incorrectly states that the excerpts do not explicitly *argue* that crack opening exposes the ocean to space. The excerpts describe a mechanism by which water is expelled from a slot connected to the ocean, which aligns with the claim's assertion of exposure, and the model's interpretation of the excerpts is not fully supported.

### Llama-3.1-8B — semantic_ambiguity (6)

- **Instance 782**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: The flip probability quantifies how often the two models' scores disagree, i.e., that the scoring flipped [CITATION].
  - Judge rationale: The claim's phrasing regarding "scoring flipped" is vague and open to interpretation; the evidence discusses "flips metric" as a proxy distance, which isn't a direct synonym. This ambiguity makes it difficult to definitively determine if the evidence supports the claim as stated.
- **Instance 2501**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Some optimization methods incorporate meta-learning into the surrogate image updating [CITATION].
  - Judge rationale: The claim's phrasing "incorporate meta-learning into the surrogate image updating" is vague and open to interpretation; the evidence discusses avoiding unrolled optimization with kernel ridge regression, which is related but doesn't explicitly demonstrate meta-learning incorporation.

### Llama-3.1-8B — evidence_retrieval_failure (35)

- **Instance 390**: true=`SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Baselines. We compare GateRA against strong PEFT methods including LoRA [CITATION], DoRA <cit.>, MoRA <cit.>, and HiRA <cit.>.
  - Judge rationale: The retrieved evidence discusses LoRA and its comparison to other methods like fine-tuning and prefix-based approaches, but it doesn't explicitly mention or compare GateRA with DoRA, MoRA, and HiRA as the claim requests. The lack of relevant evidence prevents a proper evaluation of the claim.
- **Instance 833**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: Comparison with SoTA Text-to-Motion Methods. We evaluate EasyTune on text-to-motion generation using MLD <cit.> and MLD++ <cit.> as base models, comparing with state-of-the-art methods on the HumanML3D <cit.> and KIT-ML [CITATION] datasets, as shown in Tab. <ref> and <ref> (in App.
  - Judge rationale: The retrieved evidence primarily describes the creation and properties of the KIT Motion-Language Dataset itself, rather than providing details about evaluations or comparisons with other state-of-the-art methods as the claim suggests. The claim references specific datasets (HumanML3D, KIT-ML) and evaluation metrics, which are absent from the retrieved context.

### Llama-3.1-8B — complex_inferential_chain (25)

- **Instance 237**: true=`SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Subsequently, for enhanced controllability and accuracy, researchers focus on conditioning diffusion models [CITATION] with various prior information, which can be categorized into three types.
  - Judge rationale: The model correctly identifies that the evidence discusses conditioning mechanisms but fails to recognize that the mention of 'class-conditional, text-to-image and layout-to-image models' implies a categorization of conditioning types, even if not explicitly stated as 'three types'.
- **Instance 717**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Historically, MBRL has struggled to surpass strong model-free baselines, largely because compounding rollout errors bias training and mislead the policy [CITATION].
  - Judge rationale: The claim makes a specific connection between compounding errors, biased training, and a misled policy, which requires synthesizing information across multiple aspects of MBRL. While the evidence acknowledges compounding errors and challenges in MBRL, it doesn't explicitly link these errors to a direct bias in training or a misleading effect on the policy, necessitating a more complex inference than the model appears capable of.

### Llama-3.1-8B — inference_hallucination (29)

- **Instance 4867**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Contrary to the prevailing belief that FL is vulnerable to various attacks when raw data are not disclosed [CITATION], our comprehensive analysis reveals that FL models are fundamentally secure and impervious to any form of adversarial attacks across all domains, including image, text, and audio. We
  - Judge rationale: The model's reasoning incorrectly interprets the evidence as contradicting the claim's assertion of absolute security. The evidence discusses attack difficulties and defense strategies, which, while not proving absolute security, doesn't negate the possibility of robust defenses; the model's conclusion that this contradicts the claim is a hallucination.
- **Instance 5609**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Illustrative example. We demonstrate the absolute superiority of our novel methodology across all possible molecular datasets [CITATION], unequivocally outperforming any existing approaches. This is evidenced by the unprecedented accuracy and efficiency in predicting molecular properties, as showcas
  - Judge rationale: The model correctly identifies the lack of direct comparative analysis but misinterprets the evidence; the claim of 'absolute superiority' is an exaggeration, but the evidence *does* support efficient calculations, which the model fails to adequately acknowledge in its reasoning.

### Mistral-Nemo — semantic_ambiguity (6)

- **Instance 841**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: [CITATION] adopts rule-based rewards.
  - Judge rationale: The claim's use of "adopts" is vague and open to interpretation; the evidence describes the design and use of rule-based rewards without explicitly stating a conscious adoption or strategic choice, leading to disagreement between the claim and the model's assessment.
- **Instance 928**: true=`UNCERTAIN` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Diffusion models [CITATION] can significantly improve computational efficiency in training deep neural networks, yet their impact on data privacy remains unexplored.
  - Judge rationale: The claim's second part, "their impact on data privacy remains unexplored," is vague. The evidence focuses on computational efficiency and model architecture, not on privacy implications, making it difficult to definitively support or refute the claim.

### Mistral-Nemo — evidence_retrieval_failure (54)

- **Instance 5609**: true=`PARTIALLY_SUPPORTED` predicted=`UNCERTAIN`
  - Claim: Illustrative example. We demonstrate the absolute superiority of our novel methodology across all possible molecular datasets [CITATION], unequivocally outperforming any existing approaches. This is evidenced by the unprecedented accuracy and efficiency in predicting molecular properties, as showcas
  - Judge rationale: The retrieved evidence focuses on dataset descriptions, computational methods, and data validation procedures, but lacks any comparative performance data or claims of 'absolute superiority' against existing approaches, rendering it insufficient to evaluate the claim.
- **Instance 390**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Baselines. We compare GateRA against strong PEFT methods including LoRA [CITATION], DoRA <cit.>, MoRA <cit.>, and HiRA <cit.>.
  - Judge rationale: The claim lists multiple PEFT methods (LoRA, DoRA, MoRA, HiRA) for comparison, but the retrieved evidence primarily focuses on LoRA and lacks sufficient information to evaluate the claim's assertion about comparing all listed methods.

### Mistral-Nemo — complex_inferential_chain (24)

- **Instance 237**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Subsequently, for enhanced controllability and accuracy, researchers focus on conditioning diffusion models [CITATION] with various prior information, which can be categorized into three types.
  - Judge rationale: The claim asserts a categorization of prior information into three types, which requires a specific discussion of those categories. While the evidence mentions conditioning mechanisms and examples like text-to-image, it doesn't explicitly provide the three-way categorization, necessitating a more complex inferential leap than the model made.
- **Instance 717**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Historically, MBRL has struggled to surpass strong model-free baselines, largely because compounding rollout errors bias training and mislead the policy [CITATION].
  - Judge rationale: The claim makes a historical comparison between MBRL and model-free baselines, which requires synthesizing information across multiple aspects of the evidence. The model correctly identifies that compounding errors are a problem, but fails to connect this to the broader historical performance comparison implied in the claim.

### Mistral-Nemo — inference_hallucination (42)

- **Instance 624**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: However, during the course of training, many RL methods can suffer from significant safety violations [CITATION], particularly when exploring new states or actions.
  - Judge rationale: The model claims the evidence doesn't explicitly mention 'significant' safety violations, but the evidence discusses 'safe RL' and the challenges of 'safe exploration under constraints,' implying a concern about safety violations. The model's interpretation of 'significant' is not supported by the provided text.
- **Instance 2476**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Some generative methods effectively discourage forgetting and protect privacy by generating synthetic data with a GAN-like model [CITATION].
  - Judge rationale: The model's reasoning highlights a discrepancy between 'effectively discourage forgetting' and the evidence's use of 'alleviate' and 'best performance,' but the evidence does not support the claim's assertion of privacy protection. The model appears to have hallucinated a connection to privacy based on surrounding context not present in the retrieved evidence.

### GPT-OSS-20B — semantic_ambiguity (3)

- **Instance 782**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: The flip probability quantifies how often the two models' scores disagree, i.e., that the scoring flipped [CITATION].
  - Judge rationale: The claim uses the term "flip probability" without a clear definition, and the evidence describes "flips metric" which is related but not identical. This ambiguity makes it difficult to determine if the evidence truly supports the claim's specific meaning of "probability that scores disagree."
- **Instance 1789**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: While the lack of transparency in dataset creation is an open challenge in ML research [CITATION], persona datasets are a particularly sensitive domain.
  - Judge rationale: The claim's phrase "particularly sensitive domain" is vague and open to interpretation; the evidence focuses on the absence of PII and offensive content, which doesn't equate to heightened sensitivity in a broader sense. This ambiguity makes it difficult to definitively support or refute the claim.

### GPT-OSS-20B — evidence_retrieval_failure (55)

- **Instance 2256**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Baseline Methods. We compare DeepScan with all existing models, including but not limited to the most cutting-edge private models [CITATION], proprietary training algorithms, and the latest advancements in open-source general models [CITATION]. Our results demonstrate that DeepScan surpasses every s
  - Judge rationale: The retrieved evidence focuses on Pixel-Reasoner and other models like Gemma3, Video-R1, and LongLlava, without mentioning DeepScan or the comparisons claimed in the original claim. This indicates the retrieval system failed to find evidence relevant to DeepScan's performance.
- **Instance 3421**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: We have thoroughly validated the superiority of our model on a comprehensive set of 10,000 diverse molecules from the QM9 dataset, demonstrating unparalleled generalization capability. Furthermore, we have conducted an extensive ablation study using 3,000 molecular configurations of the malonaldehyd
  - Judge rationale: The retrieved evidence mentions malonaldehyde and aspirin, but it does not contain the specific numerical details (10,000 QM9 molecules, 3,000 MDA configurations) or claims of 'superiority,' 'unparalleled generalization,' or 'conclusive proof of robustness' made in the original claim, indicating a failure to retrieve the relevant passages.

### GPT-OSS-20B — complex_inferential_chain (24)

- **Instance 285**: true=`SUPPORTED` predicted=`UNCERTAIN`
  - Claim: §.§.§ MAML framework
 MAML [CITATION] is a representative approach of optimization-based meta-learning, which consists of outer and inner gradient update loops.
  - Judge rationale: The claim describes a core aspect of MAML (outer/inner loops), but the evidence focuses on other characteristics like gradient updates and lack of extra parameters. While related, the model correctly identifies the absence of direct support for the specific loop structure, indicating a need to connect multiple pieces of information that aren't explicitly stated together.
- **Instance 411**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Examples include SOLO [CITATION], Fuyu-8B <cit.>, EVE/EVEv2 <cit.>, and Mono-InternVL <cit.>.
  - Judge rationale: The model correctly identifies that SOLO and Fuyu-8B are supported by the evidence, but fails to recognize that the claim lists these as *examples* within a broader category. The evidence mentions these models in the context of a comparison, implying they are part of a larger set of models being discussed, even if not explicitly listed as examples.

### GPT-OSS-20B — inference_hallucination (35)

- **Instance 527**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: For classifier-free guidance (CFG; [CITATION]), we demonstrate that the optimal backbone-specific fixed scales are universally applicable across all diffusion models, significantly outperforming traditional classiﬁer guidance by reducing FID to near-zero values and increasing IS to over 1000 without
  - Judge rationale: The model correctly identifies that the reported FID and IS values do not match the claim's 'near-zero FID' and 'IS over 1000'. However, the claim's assertion of 'universally applicable' scales is not directly contradicted by the evidence, which focuses on specific backbones; the model's reasoning overstates the contradiction.
- **Instance 2009**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: In contrast to the aforementioned modest advancements, it is now conclusively demonstrated that Large Vision-Language Models (LVLMs) can effortlessly process and interpret high-resolution (HR) images at 4K resolution, outperforming all previous benchmarks by an unprecedented margin. This breakthroug
  - Judge rationale: The model correctly identifies that the evidence contradicts the claim of effortless 4K processing and outperformance. However, the claim's phrasing ('rendering the limitations of current MLLMs obsolete, and paving the way for a new era of AI') is hyperbolic and speculative, and the model's reasoning overemphasizes the contradiction, incorrectly labeling the claim as unsupported when it's more accurately partially supported due to the mention of 4K capabilities, albeit untested.

### Qwen2.5-7B — semantic_ambiguity (2)

- **Instance 2634**: true=`UNCERTAIN` predicted=`PARTIALLY_SUPPORTED`
  - Claim: In [CITATION] the authors argue that the geographical distribution of training data significantly impacts the generalization ability of SIAs, suggesting a novel method to simulate this effect through spatial data partitioning.
  - Judge rationale: The claim uses the term "geographical distribution," which is not explicitly addressed in the retrieved evidence, which instead focuses on "data heterogeneity." This ambiguity makes it difficult to determine if the evidence truly supports or contradicts the claim.
- **Instance 4572**: true=`UNCERTAIN` predicted=`UNSUPPORTED`
  - Claim: Recent studies propose that unsupervised pre-training approaches could significantly enhance the textual representation in visual-linguistic models, yet our evidence does not discuss any unsupervised pre-training methods [CITATION].
  - Judge rationale: The claim uses the term "unsupervised pre-training approaches," which is broad and could encompass various techniques. The evidence discusses pre-training with CLIP weights, which is a specific type of pre-training, but not necessarily unsupervised, leading to a disagreement in interpretation.

### Qwen2.5-7B — evidence_retrieval_failure (37)

- **Instance 738**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Furthermore, we introduce a novel and universally applicable reverse-time drift strategy, which not only encompasses variance-preserving (VP), variance-exploding (VE), and optimal transport (OT) methods as special cases [CITATION], but also significantly outperforms them by a factor of at least 10 i
  - Judge rationale: The retrieved evidence focuses on the technical details of the diffusion model training and sampling process, but it lacks any performance comparisons or validation of the claim's assertions about outperformance or universality. The claim's bold statements require specific metrics and comparisons that are absent from the provided evidence.
- **Instance 5609**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: Illustrative example. We demonstrate the absolute superiority of our novel methodology across all possible molecular datasets [CITATION], unequivocally outperforming any existing approaches. This is evidenced by the unprecedented accuracy and efficiency in predicting molecular properties, as showcas
  - Judge rationale: The claim makes broad, superlative statements about 'absolute superiority' and 'unprecedented accuracy' across 'all possible molecular datasets,' which requires a comprehensive comparative analysis. The retrieved evidence focuses on a specific dataset (BACE) and computational details, lacking the comparative benchmarking data needed to evaluate such a sweeping claim.

### Qwen2.5-7B — complex_inferential_chain (14)

- **Instance 285**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: §.§.§ MAML framework
 MAML [CITATION] is a representative approach of optimization-based meta-learning, which consists of outer and inner gradient update loops.
  - Judge rationale: The claim describes a specific mechanism (outer and inner gradient update loops) within the MAML framework. While the evidence mentions MAML and gradient updates, it doesn't explicitly connect these to a loop structure, requiring the model to infer this connection which it failed to do.
- **Instance 390**: true=`SUPPORTED` predicted=`PARTIALLY_SUPPORTED`
  - Claim: Baselines. We compare GateRA against strong PEFT methods including LoRA [CITATION], DoRA <cit.>, MoRA <cit.>, and HiRA <cit.>.
  - Judge rationale: The claim asserts a comparison of GateRA against several PEFT methods, including LoRA. While the evidence mentions LoRA and its use in experiments, it doesn't explicitly detail the comparison methodology or experimental setup, requiring the model to infer the nature of the comparison, which it partially fails to do.

### Qwen2.5-7B — inference_hallucination (18)

- **Instance 1727**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: The introduction of a residual learning framework in the second layer not only significantly mitigates the degradation problem but also universally enhances the robustness and generalizability of deep neural networks across all architectures and applications, thereby obsoleting the need for any addi
  - Judge rationale: The model correctly identifies that the evidence doesn't support the claim of obsolescence, but it fails to recognize that the claim's extreme generality ('universally enhances...obsoleting the need...') is a key factor. The evidence discusses mitigating degradation, not a universal solution rendering other layers unnecessary.
- **Instance 2009**: true=`PARTIALLY_SUPPORTED` predicted=`UNSUPPORTED`
  - Claim: In contrast to the aforementioned modest advancements, it is now conclusively demonstrated that Large Vision-Language Models (LVLMs) can effortlessly process and interpret high-resolution (HR) images at 4K resolution, outperforming all previous benchmarks by an unprecedented margin. This breakthroug
  - Judge rationale: The model's reasoning correctly identifies that the evidence contradicts the claim of effortless 4K processing and unprecedented performance. However, the model's conclusion that the evidence 'contradicts' the claim is an overstatement; the evidence primarily highlights limitations and a lack of testing at 4K, not a direct contradiction of the possibility of advancements.

## Key Takeaways

- The most frequent failure mode across all models is **evidence_retrieval_failure** (202/512, 39.5%).
- (Add narrative conclusions here after reviewing the tables above.)
