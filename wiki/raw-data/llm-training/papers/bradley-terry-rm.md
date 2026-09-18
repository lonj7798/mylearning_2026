<!-- scope: Bradley-Terry paired-comparison model (1952) and how later open papers use it as the pairwise reward-model loss and as the Chatbot Arena ranking model
     deps: []
     see-also: [[rlhf-instructgpt]], [[dpo]], [[ipo]], [[self-play-preference]], [[reward-model-overoptimization]], [[pairrm]], [[judge-llm-bias]]
-->

# Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons
- **Core Insight:** Under the Bradley-Terry (BT) model the probability that one response is preferred to another depends only on the difference of two scalar scores, so a reward model fit with the BT likelihood is identified only up to an additive function of the prompt (DPO arXiv:2305.18290 §3 Eq. 1, §5.1 Lemma 1).
- **Guideline:** When preference pairs include small quality differences or near-deterministic labels, report reward-model accuracy separately by preference strength, because Llama 2's helpfulness RM scores 79.1% on "significantly better" pairs but 54.5% on "negligibly better / unsure" pairs (arXiv:2307.09288 App. A.3.3 Table 28), and IPO shows that a preference probability of 1 drives the BT reward gap to infinity and removes the effect of the KL term (arXiv:2310.12036 §4.2); otherwise a single average accuracy hides the weakest pairs.
- **Authors:** Ralph Allan Bradley, Milton E. Terry
- **Year:** 1952 (Biometrika 39(3/4): 324–345)
- **URL:** https://doi.org/10.2307/2334029 (JSTOR stable 2334029; OUP DOI 10.1093/biomet/39.3-4.324)
- **Source type:** paper (statistics journal). Full text was not accessible for this verification; see Verification.
- **Relevant topics:** Bradley-Terry, paired comparisons, reward-model loss, Plackett-Luce, DPO derivation, Chatbot Arena ranking

## Abstract
Not verified against the original. JSTOR and Oxford Academic returned access challenges on 2026-09-14, and Semantic Scholar marks the abstract as elided by the publisher. Crossref confirms the title, authors, journal, volume, issue, and pages. The paper is the reference that open LLM papers cite for the Bradley-Terry model (DPO ref. [5]; Chatbot Arena arXiv:2403.04132 §4; IPO §1). All formulas below are quoted from those later papers, with their loci, not from the 1952 text.

## Key Contributions (as restated by later open papers)
- A pairwise preference model in which each item has a latent score and the preference probability is a logistic function of the score difference (DPO §3 Eq. 1).
- Maximum-likelihood fitting of the scores from observed comparisons, which LLM work uses as the reward-model loss (DPO §3 Eq. 2; Stiennon et al. arXiv:2009.01325 §3.4; InstructGPT arXiv:2203.02155 §3.5 Eq. 1).
- A special case of the Plackett-Luce ranking model: Plackett-Luce with K = 2 reduces to BT (DPO App. A.3 Eq. 18).
- A ranking method for models from human votes: Chatbot Arena estimates BT coefficients for each model (arXiv:2403.04132 §4 Eq. 2–3).

## Key Figures/Tables to Study
- DPO §3 Eq. 1–2, §4 Eq. 5–7, §5.1 Definition 1 and Lemma 1.
- InstructGPT §3.5 Eq. 1 and App. C.2 (K-way comparisons per batch element).
- Stiennon et al. §4.3 and Figure 6 (RM validation accuracy versus data size and model size).
- Llama 2 §3.2.2 Eq. 2 and App. A.3.3 Tables 27–28 (margin by preference strength).

## Technical Details
- **Model.** `p*(y1 ≻ y2 | x) = exp(r*(x,y1)) / (exp(r*(x,y1)) + exp(r*(x,y2))) = σ(r*(x,y1) − r*(x,y2))` (DPO §3 Eq. 1; σ form stated in §4 after Eq. 5).
  `x` is the prompt, `y1, y2` are two responses, `r*` is the latent reward, and `σ` is the logistic function.
- **Reward-model loss.** `L_R(r_φ, D) = −E_(x,y_w,y_l)~D [log σ(r_φ(x,y_w) − r_φ(x,y_l))]` (DPO §3 Eq. 2).
  `y_w` is the preferred response, `y_l` the dispreferred one, `r_φ` the parametric reward model, `D` the comparison dataset.
  DPO frames this as binary classification (§3). Derived: it equals binary cross-entropy with label 1 and logit `r_φ(x,y_w) − r_φ(x,y_l)`; Chatbot Arena writes BT fitting as a binary cross-entropy minimization (§4 Eq. 3).
- **Head.** Stiennon et al. add a randomly initialized linear head that outputs a scalar to the supervised baseline (§3.4). InstructGPT removes the final unembedding layer of the SFT model and outputs a scalar; it uses only 6B RMs because this "saves a lot of compute" and 175B RM training "could be unstable" (§3.5; App. C.2).
- **Identifiability.** Reward functions that differ by a function `f(x)` induce the same BT preference distribution (DPO §5.1 Definition 1, Lemma 1). Stiennon et al. normalize RM outputs so reference summaries have mean score 0 (§3.4). InstructGPT adds a bias so labeler demonstrations have mean score 0 before RL, "since the RM loss is invariant to shifts in reward" (§3.5). Chatbot Arena fixes `ξ_1 = 0` for the same reason (§4).
- **K-way labels.** InstructGPT labelers rank K = 4 to 9 responses, giving C(K,2) comparisons per prompt (§3.5). Shuffling all comparisons into one dataset made the RM overfit in a single pass; training all comparisons from one prompt as one batch element fixed this and needs one forward pass per completion (§3.5). Eq. 1 divides the loss by C(K,2). Ties were dropped; a batch of 64 prompts holds up to 2,304 comparisons (App. C.2).
- **Margin variant.** Llama 2 uses `−log σ(r_θ(x,y_c) − r_θ(x,y_r) − m(r))`, where `m(r)` is a discrete function of the four-level preference rating, larger for pairs with distinct responses (§3.2.2 Eq. 2). Margins: small {1, 2/3, 1/3, 0}, large {3, 2, 1, 0} (Table 27). Average helpfulness-RM accuracy: 62.5 (no margin), 63.0 (small), 62.9 (large) (Table 28).
- **Arena scores.** Chatbot Arena earlier reported Elo scores and changed to BT coefficients "because the BT coefficients are better for the purpose of statistical estimation"; it notes that the maximum-likelihood estimates stay asymptotically normal under misspecification when a sandwich covariance is used (§4).

## Findings relevant to generality, negative feedback
### Generality: what a BT score does and does not measure
- **RM scaling (Result, single study).** Stiennon et al. trained 7 RMs from 160M to 13B parameters on 8k to 64k comparisons. Doubling data raised validation accuracy by about 1.1%; doubling model size by about 1.8% (§4.3, Fig. 6).
- **Transfer.** TL;DR-trained RMs agree with labelers on CNN/DM 62.4% (1.3B) and 66.5% (6.7B) of the time, against 66.9% inter-labeler agreement (§4.3).
- **Length bias.** The 6.7B RM prefers improving edits that shorten the summary 62.6% of the time, against 76.4% for humans (§4.3).
- **Over-optimization.** Optimizing a policy further against the RM eventually makes the RM anti-correlated with human preference (Stiennon §4.3, Fig. 5); see [[reward-model-overoptimization]].
- **Single-score limit.** A reward model assigns one score per response and cannot represent non-transitive preferences; a learned pairwise preference model can (NLHF arXiv:2312.00886 §3.1).
- **Leaderboard assumptions.** "The Leaderboard Illusion" (arXiv:2504.20879v2) lists three BT assumptions behind Arena scores: unbiased sampling of comparisons, transitivity, and a connected comparison graph (§2.1). It reports 27 private Meta variants tested before Llama 4 (§3.1); a simulation in which testing 10 private variants raises the maximum discovered score by about 100 points (§3 findings, Fig. 7); Google and OpenAI receiving an estimated 19.2% and 20.4% of Arena data versus 29.7% for 83 open-weight models combined (Abstract); and 205 of 243 public models silently deprecated against 47 officially listed, which it argues violates BT assumptions (§1 finding 4).
### Negative feedback: the rejected response in RM training
- **Use.** The dispreferred response is used as a gradient signal on the reward model (negative as gradient): the loss lowers `r(y_l)` relative to `r(y_w)`, and only the difference is constrained (DPO Eq. 2).
- **Near-miss pairs.** Without a margin, Llama 2's helpfulness RM accuracy is 79.1% on "significantly better", 66.9% on "better", 59.8% on "slightly better", and 54.5% on "negligibly better / unsure" pairs; the large margin lowers the last value to 54.3% (App. A.3.3 Table 28).
- **Minimal edits.** Stiennon RMs prefer human minimal-edit improvements 79.4% (1.3B) and 82.8% (6.7B) of the time, against 84.1% for separate human evaluators; with role-reversed perturbed summaries they pick the original 92.9% and 97.2% of the time (§4.3).
- **Deterministic labels.** If `p*(y ≻ y′) = 1`, BT requires `r(y) − r(y′) → +∞`, so the optimal regularized policy sets `π*(y′) = 0` for any KL strength τ; a true 0.8 preference can be estimated as 1 from few samples (IPO §4.2). IPO notes that an explicitly trained RM is in practice underfit, which keeps the policy regularized (§4.2).

## Connections
- [[rlhf-instructgpt]] — K-way BT loss (Eq. 1) and the 6B RM used for all PPO models.
- [[dpo]] — rewrites the BT reward as `β log π/π_ref` so the partition function cancels (Eq. 5–7).
- [[ipo]] — sets Ψ to identity to bypass the BT pointwise-reward assumption (§5).
- [[self-play-preference]] — Nash learning from a preference model instead of a BT reward model (§3.1).
- [[reward-model-overoptimization]] — scaling of proxy-RM over-optimization.
- [[reward-ensembling]] — ensembles of reward models against over-optimization.
- [[pairrm]] — pairwise ranker that encodes two candidates jointly instead of scoring each alone.
- [[judge-llm-bias]] — MT-Bench and Chatbot Arena paper on LLM judges.
- [[generative-reward-models]], [[rlaif-scaling]] — library cards that build on this loss.

## Verification
- Checked on 2026-09-14 against: Crossref record for DOI 10.1093/biomet/39.3-4.324 (bibliographic data only; full text blocked at JSTOR and OUP). Formulas and LLM usage checked against arXiv:2305.18290v3, 2009.01325v3, 2203.02155v1, 2307.09288v2, 2310.12036v2, 2312.00886v4, 2403.04132v1, 2504.20879v2.
- Corrections to the previous card version:
  - Title "The Bradley-Terry Model for Pairwise Preferences" → exact 1952 paper title (Crossref; DPO ref. [5]).
  - Authors line listed Stiennon et al. and Ouyang et al. as authors → only Bradley and Terry; later papers are cited at their loci.
  - "Stiennon 2020 Fig. 2 (RM accuracy vs size)" → RM scaling is Figure 6 (§4.3); Figure 2 is the procedure diagram.
  - "InstructGPT §3.4 — the exact loss and head" → §3.5 Eq. 1 (§3.4 is human data collection).
  - "BT scores are Elo ratings up to a constant scaling ... exactly how Chatbot Arena ranks models" → Chatbot Arena replaced its earlier Elo scores with BT coefficients for statistical estimation (§4); no equivalence is stated.
  - "Margin m > 0 ... to suppress overconfidence" → Llama 2's margin grows with preference strength to make scores of distinct pairs more separated (§3.2.2).
  - "IPO and generalized preference objectives address [intransitivity]" → IPO addresses the pairwise-to-pointwise assumption and overfitting under deterministic preferences (§1, §4.2); non-transitivity is discussed in NLHF §3.1.
  - "per-prompt mean is subtracted" → Stiennon and InstructGPT apply one bias so reference summaries or demonstrations have mean 0 (§3.4; §3.5); DPO §3 describes normalization with E[r] = 0 for all x.
- Removed as unsupported by the checked sources: "every modern preference-based method trains a BT RM or exploits the BT form" (IPO and NLHF avoid it); "standard Tülu / DeepSeek recipe"; length-bias patches by regressing out length or length-matched pairs; "ensembling reduces hacking surface"; "Constitutional AI's preference model is also BT".
- Not reported: any statement from the 1952 text itself (notation, estimation procedure, examples), because the full text was not read.
