---
chapter: ch-31a
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/papers/pretraining-with-human-preferences.md on 2026-09-15)
source_url: https://arxiv.org/abs/2302.08582
source_version: arXiv v2 (2023-06-14); ICML 2023
created_at: "2026-09-15"
---

# Excerpt: Pretraining Language Models with Human Preferences (Korbak, Shi, Chen, Bhalerao, Buckley, Phang, et al.; Sussex, NYU, FAR AI, Northeastern, Anthropic)

Facts used by [[read]], read in the arXiv v2 PDF on 2026-09-15.

## Objectives compared (§2, Eq. 1-8)
- MLE; MLE with filtering (documents with average reward below threshold t get zero loss, Eq. 4); conditional training; unlikelihood; reward-weighted regression (RWR, Eq. 7); advantage-weighted regression (AWR, Eq. 8).
- Conditional training (Eq. 5): each segment x^i is prefixed with a control token c^i, <|good|> if R(x^i) ≥ t and <|bad|> otherwise; the loss is log π_θ(c^1, x^1, …, c^|x|, x^|x|). "At inference time, we sample from π_θ(· | c^1 = <|good|>)."
- Unlikelihood (Eq. 6): log-likelihood on segments with R(x^i) > t plus α·Σ log(1 − π_θ(x^i_j | ·)) on tokens of segments with R(x^i) ≤ t.

## Setting (§3)
- gpt2-small (124M) trained from scratch on 3.32B tokens (compute-optimal per Hoffmann et al.): 1.95M Pile documents for toxicity and PII; 1.5M Python files for PEP8.
- Rewards: Detoxify toxicity probability (sentence level), Scrubadub PII detections per character, pycodestyle PEP8 violations per character.
- Hyperparameters, toxicity pretraining (App. A Table 1): MLE and conditional LR 5·10⁻⁴, batch 64; conditional t = 5.6·10⁻⁴; unlikelihood t = 7.8·10⁻⁴, α = 1. PII (Table 2): conditional and UL t = 0.0, UL α = 1. PEP8 (Table 3): UL α = 0.01.

## Results
- Toxicity misalignment score (average over 4,096 unconditional samples): MLE 0.0141; conditional training 0.0011 (§4.1).
- Conditional training is strictly Pareto-optimal on toxicity and on the Pareto frontier for PII and PEP8, trading misalignment against KL from GPT-3 (§4.1, Fig. 2). Filtering pays the largest capability penalty on PII and PEP8. "The success of unlikelihood training is highly task-dependent; it reduces the misalignment score significantly for toxicity but only slightly for PII and PEP8" (§4.1).
- Red-teaming (§4.2, Fig. 3): conditional training and filtering are the most robust; unlikelihood is the most robust on toxicity and the least robust on PII.
- Capabilities (§4.3, Fig. 4): conditional training slightly exceeds MLE accuracy on LAMBADA for toxicity and PII and most closely matches MLE on GLUE; unlikelihood "obtains very low accuracy on toxicity"; on HumanEval (PEP8) "Unlikelihood consistently obtains the lowest scores."
- Diversity (§4.4, Fig. 5): conditional training and filtering show decreased diversity but a fraction of distinct unigrams close to MLE; unlikelihood, AWR, RWR match MLE diversity with slightly more degeneration; "none of the PHF objectives cause significant degeneration or entropy collapse."
- Pretraining vs fine-tuning with feedback (§5, Fig. 6): on PII, conditional pretraining converges to 0.0013 vs 0.0018 after MLE pretraining followed by 1.6B tokens of fine-tuning with feedback.
- New control tokens added during fine-tuning cause "a notable drop in alignment and capabilities" for the first 100M tokens (App. A).

## Limits stated by the source
- Models are 124M parameters; resulting LMs "are not completely aligned or safe in all deployment scenarios" (§4.2).
