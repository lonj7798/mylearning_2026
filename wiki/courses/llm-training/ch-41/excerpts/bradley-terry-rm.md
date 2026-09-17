---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/bradley-terry-rm.md (formulas quoted from arXiv:2305.18290v3, 2203.02155v1, 2009.01325v3, 2307.09288v2 at their loci)
source_url: https://doi.org/10.2307/2334029
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; rewritten to match the verified card and primary source)"
---

# Excerpt: the Bradley–Terry model as a reward-model loss

Used by [[read]] §1 and the negatives section. The 1952 paper's full text was not accessible when the card was verified; every formula below is quoted from later open papers at the stated locus ([[bradley-terry-rm]] Verification).

The earlier version of this excerpt stated that InstructGPT's K-way loss is a sum of `log σ` terms along a Plackett–Luce chain, cited Stiennon Fig. 2 for RM scaling, and described per-prompt mean subtraction as standard practice. Those statements are corrected below.

## Model and loss
- `p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2))) = σ(r*(x,y1) − r*(x,y2))` (DPO §3 Eq. 1).
- `L_R(r_φ, D) = −E_(x,y_w,y_l)~D [log σ(r_φ(x,y_w) − r_φ(x,y_l))]` (DPO §3 Eq. 2). Equivalent to binary cross-entropy with label 1 and logit equal to the score difference (derived).
- Head: Stiennon et al. add a randomly initialized linear head with scalar output (§3.4); InstructGPT removes the final unembedding layer of the SFT model (§3.5).

## Identifiability
- Reward functions differing by any `f(x)` induce the same BT preference distribution (DPO §5.1 Definition 1, Lemma 1).
- Stiennon et al. normalize so reference summaries have mean score 0 (§3.4). InstructGPT adds a bias so labeler demonstrations have mean 0 before RL, "since the RM loss is invariant to shifts in reward" (§3.5).

## K-way rankings
- Plackett–Luce (DPO App. A.3 Eq. 18): `p*(τ | y1..yK, x) = ∏_{k=1}^{K} exp(r*(x, y_τ(k))) / Σ_{j=k}^{K} exp(r*(x, y_τ(j)))`. Its log-likelihood is a sum of log-softmax terms; K = 2 reduces to BT.
- InstructGPT does not use Plackett–Luce. Labelers rank K = 4 to 9 responses; the loss averages BT terms over all C(K,2) pairs, `loss(θ) = −(1/C(K,2)) E[log σ(r_θ(x,y_w) − r_θ(x,y_l))]` (§3.5 Eq. 1). Shuffling all comparisons into one dataset overfit in one pass; putting all comparisons of a prompt in one batch element fixed it (§3.5). A batch of 64 prompts holds up to 2,304 comparisons (App. C.2).

## Evidence
- Stiennon RM scaling: 7 RMs from 160M to 13B on 8k to 64k comparisons; doubling data +~1.1% accuracy, doubling size +~1.8% (§4.3, Fig. 6; not Fig. 2).
- Transfer: TL;DR RMs agree with labelers on CNN/DM 62.4% (1.3B), 66.5% (6.7B), against 66.9% inter-labeler agreement (§4.3).
- Length: the 6.7B RM prefers shortening edits 62.6% of the time vs 76.4% for humans (§4.3).
- Llama 2 helpfulness RM accuracy by preference strength, no margin: 79.1 (significantly better), 66.9 (better), 59.8 (slightly better), 54.5 (negligibly better / unsure) (App. A.3.3 Table 28). Average with margin: 62.5 (none), 63.0 (small), 62.9 (large).
- Deterministic labels: if `p*(y ≻ y′) = 1`, BT requires an infinite reward gap and the optimal regularized policy sets `π*(y′) = 0` for any KL strength (IPO §4.2).
- Non-transitive preferences cannot be represented by one score per response (NLHF §3.1).
