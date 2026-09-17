<!-- scope: Shao et al. (2026): RLVR gains on Qwen2.5 models from random, format-only and deliberately incorrect rewards, and the failure of those same signals on OLMo2 and Llama3
     deps: [[grpo]]
     see-also: [[rlvr-beyond-base-model]], [[negative-sample-reinforcement]]
-->

# Spurious Rewards: Rethinking Training Signals in RLVR
- **Core Insight:** On Qwen2.5-Math-7B, GRPO with randomly assigned rewards raised MATH-500 by 21.4 points and with deliberately incorrect labels by 24.1 points, against 29.1 points for ground-truth rewards; the same spurious rewards leave OLMo2 flat and fail to help at least one non-Qwen model each (§2, §3, Figure 1, Figure 3).
- **Guideline:** When an RLVR result is used as evidence that a training signal works, repeat at least one cell on a second model family, because in this paper reward signals that consistently improved Qwen2.5 were flat or harmful on OLMo2-7B and Llama3 models, and the authors state the conclusion as a practical warning to test proposed reward signals on diverse models (§3).
- **Authors:** Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, et al. (University of Washington; Allen Institute for AI; UC Berkeley)
- **Year:** 2025 (arXiv v1 2025-06; v2 2026-02-25)
- **URL:** https://arxiv.org/abs/2506.10947
- **Source type:** paper
- **Relevant topics:** RLVR, reward quality, model-family dependence, clipping bias, pretrained priors

## Abstract
The paper shows that RLVR can improve mathematical reasoning even when the reward has little or no positive correlation with correctness. On Qwen2.5-Math-7B, GRPO with random rewards improves MATH-500 by 21.4 points, close to the 29.1-point gain from ground-truth rewards. The proposed explanation is a clipping bias in GRPO that amplifies behaviours already present in the pretrained model, including a "code reasoning" mode. The effect is specific to model families whose priors contain that behaviour.

## Key Contributions
- Reward ablation on Qwen2.5-Math-7B: ground truth, majority vote, format-only, incorrect labels, random labels (§2).
- Cross-family replication on Qwen2.5, Qwen2.5-Math, OLMo2-7B, OLMo2-7B-SFT, Llama3.1-8B(-Instruct), Llama3.2-3B(-Instruct) (§3, Figure 3).
- An account of the gains as amplification of high-prior pretrained behaviours through GRPO's clipping asymmetry (§4, §5).
- A stated methodological warning about single-family RLVR evidence (§3).

## Key Figures/Tables to Study
- **Figure 1:** MATH-500 gains by reward type on Qwen2.5-Math-7B.
- **Figure 3:** the same reward types across eight models from three families.

## Technical Details
- **MATH-500 gains on Qwen2.5-Math-7B (§2, Abstract).** Ground truth +29.1; incorrect labels +24.1; random rewards +21.4; majority vote in the 27–29 range; format reward lower.
- **AMC (§2).** Format +13.8, incorrect +24.1, random +21.4, against roughly 27–29 for majority-voted and ground-truth labels.
- **AIME (App. D).** On AIME 2024, format reward +10.3 against ground truth +15.3, with incorrect and random both +10.2. On AIME 2025, whose questions postdate the models' knowledge cutoffs, ground-truth labels have a clear advantage and other rewards give −0.4 to +4.5.
- **Family dependence (§3).** Within Qwen2.5, all non-random rewards improve MATH-500; OLMo2 stays flat under spurious rewards and gains mainly with ground-truth rewards. Smaller models benefit less from random rewards. Models that were already RL post-trained see minimal gains under nearly all rewards (App. J).
- **Interpretation (§4, §5).** The authors attribute the effect to clipping behaviour that gives non-negative gradients to certain high-prior tokens, amplifying a pre-existing behaviour rather than teaching a new one; models are grouped as "No-Code" (Llama, Qwen2.5-1.5B, OLMo2-7B) and "Bad-Code" (OLMo2-7B-SFT, Qwen2.5-7B) by whether they emit code in their reasoning.

## Findings relevant to generality
- **Attribution risk.** A measured in-domain gain after RLVR is not by itself evidence that the reward carried information; on one model family, pure noise produced most of the gain (§2, Result, single study).
- **Transfer limit.** The paper's own conclusion is about transfer of the method across model families, not across tasks: a reward design validated on Qwen2.5 should not be assumed to work elsewhere (§3).

## Connections
- [[rlvr-beyond-base-model]] — the coverage-based version of the same "sharpening, not new capability" reading.
- [[negative-sample-reinforcement]] — decomposes which half of the RLVR signal drives the sharpening.
- [[grpo]] — the objective whose clipping asymmetry is analysed.

## Verification
- Checked on 2026-09-15 against the cached primary text of https://arxiv.org/abs/2506.10947 (arXiv v2, 2026-02-25): Abstract, §2, §3, §4, §5, App. D, App. J.
- Corrections to the previous card version (wiki/raw-data/llm-training/papers/spurious-rewards-rlvr.md, which carries no Verification section): that card gives the 21.4 and 29.1 numbers without loci and does not record the cross-family failure, the AIME 2025 exception, or the author affiliations.
- Not reported by the source: held-out non-math evaluations after spurious-reward training; pass@k at large k for the spurious-reward runs.
