<!-- scope: pairwise reward models that score two responses jointly
     deps: [[bradley-terry-rm]]
     see-also: [[reward-ensembling]], [[judge-llm-bias]]
-->

# PairRM: Pairwise Reward Models for Response Ranking
- **Core Insight:** Instead of scoring each response in isolation and subtracting, a pairwise RM takes `(x, y_A, y_B)` as joint input and emits a single preference logit — letting self-attention compare the two responses directly, which outperforms scalar BT RMs on reranking and Best-of-N at much smaller model size.
- **Guideline:** For reranking N candidates, use a PairRM-style joint encoder with swap-augmentation (both orders) and tournament aggregation; PairRM at 0.4B can match scalar RMs at 7B on LLM-Blender / MT-Bench preference tasks.
- **Authors:** Dongfu Jiang, Xiang Ren, Bill Yuchen Lin ("LLM-Blender", ACL 2023; PairRM / PairRanker released as part of LLM-Blender)
- **Year:** 2023
- **URL:** https://arxiv.org/abs/2306.02561 ; https://huggingface.co/llm-blender/PairRM
- **Relevant topics:** pairwise RM, joint-input scoring, reranking, Best-of-N, LLM-Blender, GenRanker

## Abstract
LLM-Blender introduces PairRanker (PairRM), an ensemble approach that jointly encodes two candidate responses to a prompt and emits a scalar preference logit. Unlike a Bradley-Terry RM (where each candidate is scored independently), PairRM's cross-attention between the two responses captures subtle differences — especially on tasks where absolute quality is hard but relative quality is easy (length-matched, paraphrase-similar, or near-tie cases). A 0.4B DeBERTa-based PairRM beats scalar RMs based on Llama-2-7B on MixInstruct and MT-Bench reranking. PairRM has become a popular reranker in open post-training stacks (e.g. Tülu SFT data selection, DPO pair filtering).

## Key Contributions
- **Joint encoding:** `f(x, y_A, y_B) → logit`; cross-attention sees both responses at once.
- **Swap-augmentation:** always evaluate `(y_A, y_B)` and `(y_B, y_A)`, average logits — cancels position bias at train and inference time.
- **Tournament Best-of-N:** for N candidates run O(N log N) pairwise comparisons, advance winners — avoids O(N²) full pairwise pass, retains near-optimal selection.
- **Size efficiency:** PairRM-0.4B matches or beats scalar RMs at 7B on LLM-Blender benchmark; small enough to run inline in DPO / RLHF pipelines.
- **Use cases beyond reranking:** preference-pair filtering for DPO (keep pairs where PairRM confidently prefers one), synthetic preference label generation, Best-of-N verification on instruction-following tasks.

## Key Figures/Tables to Study
- **LLM-Blender Fig. 3** (MixInstruct reranking results) — PairRM scales are flat across model sizes; scalar RMs lag at small sizes.
- **Ablation: no-swap vs swap-augment** — swap brings 2–3 pp.
- **Table comparing PairRM vs GPT-4 as judge** — PairRM-0.4B is within a few points of GPT-4 on tight pairs, at a fraction of cost.

## Technical Details
- **Input format:** `[CLS] x [SEP] y_A [SEP] y_B [SEP]`; score = `σ(w · [CLS])`.
- **Loss:** binary cross-entropy with label 1 if A is preferred, 0 otherwise; label from human annotations (MixInstruct) or from GPT-4 labels.
- **Tournament pseudocode:** round-robin brackets, advance winners, log per-round scores for calibration.
- **Failure modes:** inherits verbosity and position bias to some extent; swap-augmentation handles position but not verbosity — explicitly length-balance training pairs.
- **DPO pair filtering usage:** keep `(y_w, y_l)` pairs where `PairRM(y_w, y_l) > τ`; this is a simple quality gate that has been shown to lift DPO performance ~2 pp on held-out evals.

## Connections
- Structural alternative to scalar BT RMs (**[[bradley-terry-rm]]**); small enough to ensemble cheaply (**[[reward-ensembling]]**).
- Shares the biases catalogued in **[[judge-llm-bias]]**; swap-augmentation is the standard mitigation.
- Used in Tülu-line and OpenRLHF recipes for preference-pair selection before **[[rlvr-tulu3]]** and DPO stages.
- Compatible with generative RM line (**[[generative-reward-models]]**) — PairRM scores can be a prior that a GenRM critiques.
