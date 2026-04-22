<!-- scope: Bradley-Terry pairwise preference model — the canonical RM training loss
     deps: []
     see-also: [[reward-model-overoptimization]], [[pairrm]], [[constitutional-ai]]
-->

# The Bradley-Terry Model for Pairwise Preferences
- **Core Insight:** Model the probability that item A beats item B as `σ(r(A) − r(B))` where `r(·)` is a scalar quality score; then fitting a reward model to pairwise human preferences is logistic regression on score differences.
- **Guideline:** Train a reward model with the Bradley-Terry negative-log-likelihood `−log σ(r(y_win) − r(y_lose))`, using the LM hidden state of the last token of the response to produce `r(·)`; this is the standard InstructGPT / Stiennon / Tülu / DeepSeek recipe and underlies DPO's closed-form rearrangement.
- **Authors:** Ralph Allan Bradley, Milton E. Terry (original statistics paper, 1952); canonical LLM usage: Stiennon et al. 2020, Ouyang et al. 2022
- **Year:** 1952 (model), 2020+ (LLM adoption)
- **URL:** https://www.jstor.org/stable/2334029 ; https://arxiv.org/abs/2009.01325 ; https://arxiv.org/abs/2203.02155
- **Relevant topics:** Bradley-Terry, preference model, pairwise loss, DPO derivation, Elo, PairRM

## Abstract
The Bradley-Terry (BT) model is a classical paired-comparison model: if items have latent scores `r_1, …, r_K`, the probability that item `i` is preferred to `j` in a pair is `P(i ≻ j) = σ(r_i − r_j) = exp(r_i) / (exp(r_i) + exp(r_j))`. In RLHF this is the assumed generative model behind a preference dataset of triples `(x, y_win, y_lose)`: a reward model `r_θ(x,y)` is trained with BT maximum-likelihood. Every modern preference-based post-training method (classical RLHF, DPO, IPO, Constitutional AI) either trains a BT-style RM or exploits the BT functional form analytically.

## Key Contributions
- **BT likelihood:**
  `P(y_1 ≻ y_2 | x) = σ(r(x, y_1) − r(x, y_2))`
- **RM training loss (Stiennon 2020 / InstructGPT):**
  `L_RM(θ) = − E_{(x, y_w, y_l) ~ D}[log σ(r_θ(x, y_w) − r_θ(x, y_l))]`
- **Parameterization:** `r_θ(x, y)` is usually the scalar head on top of the last-token hidden state of a pretrained LM.
- **Connection to cross-entropy:** BT loss on pairs is equivalent to binary cross-entropy where the label is always 1 and the logit is `r(y_w) − r(y_l)`.
- **Connection to Elo:** BT scores are Elo ratings up to a constant scaling; pairwise tournaments evaluated with BT are exactly how Chatbot Arena ranks models.
- **Connection to DPO (Rafailov 2023):** given the BT generative model + KL-regularized RL objective, the optimal policy satisfies `r*(x,y) = β log(π*(y|x)/π_ref(y|x)) + const`, turning the RM + RL pipeline into a single supervised objective on preference pairs.
- **Identifiability:** BT scores are identified only up to an additive constant per prompt; in practice the per-prompt mean is subtracted (or absorbed by the language model).

## Key Figures/Tables to Study
- **Stiennon 2020 Fig. 2** (RM accuracy vs size) — shows BT-trained RMs scale.
- **InstructGPT §3.4** — the exact loss and head architecture.
- **DPO paper Eq. 7** — the rearrangement that eliminates explicit RM training.

## Technical Details
- **Loss with margin or smoothing:** `− log σ(r_w − r_l − m)` with small `m > 0` is sometimes used to suppress overconfidence.
- **Length-bias patch:** since BT is scalar, simple tricks like regressing out response length from `r` or training on length-matched pairs are used to mitigate length bias.
- **K-way ranking extension (Plackett-Luce):** for K-way rankings, `P(y_1 ≻ y_2 ≻ … ≻ y_K) = ∏_i exp(r_i)/Σ_{j≥i} exp(r_j)` — used when humans rank more than 2 items.
- **Ensembling / calibration:** averaging scores from multiple independently-trained BT heads reduces hacking surface (**[[reward-ensembling]]**).
- **Failure modes:** BT assumes transitive, independent, additive preferences — real human preferences are intransitive and context-dependent; IPO and generalized preference objectives address this.

## Connections
- The loss underlying every modern RM; directly feeds **[[reward-model-overoptimization]]**.
- DPO / IPO are BT-plus-algebra; Constitutional AI's preference model is also BT over AI-generated labels (**[[constitutional-ai]]**).
- PairRM (**[[pairrm]]**) departs from BT's single-scalar structure and scores pairs directly.
- Chatbot Arena Elo ↔ BT equivalence underlies **[[judge-llm-bias]]** evaluations.
