<!-- scope: LLM-Blender's PairRanker (arXiv:2306.02561) and its released checkpoint PairRM (llm-blender/PairRM): a 0.4B cross-encoder that scores two candidate responses jointly
     deps: [[bradley-terry-rm]]
     see-also: [[reward-ensembling]], [[judge-llm-bias]], [[best-of-n]], [[generative-reward-models]]
-->

# LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion
- **Core Insight:** PairRanker encodes the instruction and two candidates in one DeBERTa-v3-large (400M) sequence; on MixInstruct its top-1 selections reach an average GPT-Rank of 3.20 versus 3.90 for the best single LLM, and its rankings correlate with ChatGPT pairwise judgments at Pearson 46.98 versus 41.13 for the pointwise SummaReranker (§5.2, Tables 2-3).
- **Guideline:** When ranking fluent candidates from several strong LLMs, use a joint pairwise encoder rather than a pointwise scorer, because on MixInstruct PairRanker has a better average GPT-Rank than SimCLS and SummaReranker (Table 2) and the highest correlation with GPT-Rank (Table 3); when the comparison budget is limited, use one bubble-sort pass (N − 1 comparisons), which App. C.4 reports performs well at that cost while MaxLogits overtakes it only with more comparisons (Fig. 6).
- **Authors:** Dongfu Jiang, Xiang Ren, Bill Yuchen Lin (Zhejiang University, USC, Allen Institute for AI)
- **Year:** 2023 (arXiv v1 2023-06; ACL 2023)
- **URL:** https://arxiv.org/abs/2306.02561 ; released checkpoint: https://huggingface.co/llm-blender/PairRM (model card revision 5b880cc, 2024-01-22)
- **Source type:** paper; official model card for the PairRM checkpoint (same authors' project)
- **Relevant topics:** pairwise reward model, cross-encoder ranking, reranking, best-of-n, LLM ensembling, position sensitivity

## Abstract
LLM-Blender is an ensembling framework for open-source LLMs with two modules. PairRanker compares candidate outputs pairwise: it
jointly encodes the input and a pair of candidates with a cross-attention encoder and decides which is better. GenFuser merges the
top-ranked candidates into a new output. The authors introduce MixInstruct, a benchmark of instruction data with oracle pairwise
comparisons made by ChatGPT. PairRanker shows the highest correlation with ChatGPT-based ranking, and LLM-Blender outperforms
individual LLMs and baseline rankers on the reported metrics (abstract). PairRM is the later released PairRanker checkpoint, trained
on six human-preference datasets (model card).

## Key Contributions
- Pairwise ranking with joint encoding of `[x; y_i; y_j]`, in contrast to pointwise rankers that score `f(x, y_i)` (§3.1-3.2, Fig. 3).
- Three aggregation rules over the comparison matrix: MaxLogits, MaxWins, and a single bubble-sort pass (§3.3, Fig. 4).
- GenFuser: a Flan-T5-XL (3B) seq2seq model that fuses the top K = 3 candidates (§2.3, §4).
- MixInstruct: 110K instruction examples split 100K/5K/5K, with outputs from 11 open LLMs per example; oracle rankings come from ChatGPT judging all 55 candidate pairs, and train/validation sets also carry reference-based metric scores (§1, §2.2, Table 1).
- PairRM checkpoint: 0.4B, total max length 2048, evaluated against RMs and judge LMs on Auto-J, HHH-Alignment, MT-Bench human judgments (model card).

## Key Figures/Tables to Study
- Fig. 1: Vicuna is the top-ranked LLM on only 21.22% of 5,000 instructions.
- Fig. 3-4: rankers compared; PairRanker aggregation.
- Table 2 (MixInstruct results) and Table 3 (correlation with GPT-Rank).
- App. C, Tables 4-6 and Fig. 6-7: transfer to other tasks and to GPT-3 candidates; comparison budget; consistency.
- Model card tables: Auto-J pairwise; HHH-Alignment and MT-Bench human judgments.

## Technical Details
**Input and scoring (§3.3, App. A).**
1. Sequence: `<s><source> x </s> <candidate1> y_i </s> <candidate2> y_j </s>`.
2. The embeddings of `<source>`, `<candidate1>`, `<candidate2>` represent x, y_i, y_j.
3. `[x; y_i]` and `[x; y_j]` each pass through a five-layer MLP with tanh; the output dimension equals the number of quality metrics Q. Averaging gives s^i_(i,j) and s^j_(i,j).
4. Pair score: `s_ij = s^i_(i,j) − s^j_(i,j)`, the confidence that y_i is better than y_j.

**Loss (§3.2).** For each metric Q: `L_Q = −z_i log σ(s^i_(i,j)) − z_j log σ(s^j_(i,j))`, with `(z_i, z_j) = (1,0)` if `Q(y_i, y) ≥ Q(y_j, y)` and `(0,1)` otherwise. σ is the sigmoid, y is the reference output, and the final loss averages L_Q over metrics. On MixInstruct the supervision metric is BARTScore (§5.1).

**Training-pair sampling (§3.3).** Pairs are sub-sampled rather than all N(N−1)/2; the reference y is added to the candidate pool; 5 pairs per input are reported as "sufficient for obtaining decent results". Candidate order within a pair is shuffled during training because `(x, y_i, y_j)` and `(x, y_j, y_i)` can disagree.

**Aggregation (§3.3).** With matrix `M_i^j = s_ij`: MaxLogits `s_i = Σ(M_i^* − M_*^i)`; MaxWins counts wins. Both need O(N²) comparisons. Bubble aggregation keeps a best index k and replaces it with i when `M_i^k − M_k^i > 0`, using N − 1 comparisons (O(N)). MaxLogits is the default because it performed best in the appendix experiments.

**MixInstruct results (Table 2, N = 11).** Average GPT-Rank (lower is better): best single LLM Open Assistant 3.90; SimCLS 3.50; SummaReranker 3.66; PairRanker 3.20; LLM-Blender (PairRanker K=3 + GenFuser) 3.01. PairRanker selections rank in the top 3 on 65.12% of examples. Table 3: PairRanker Pearson 46.98, Spearman 44.98, footrule 27.52; SummaReranker 41.13, 39.10, 29.69; BARTScore 38.49, 36.76, 30.93.

**PairRM checkpoint (model card @5b880cc).** Backbone microsoft/deberta-v3-large, 0.4B. Max lengths (units not stated): source 1224, candidate 412, total 2048 (earlier pair-ranker: 128, 128, 384). Results as pairwise agreement accuracy: Auto-J overall 59.05 (UltraRM-13B 59.85, AUTO-J 13B 54.8, GPT-4 61.9); HHH-Alignment total 84.62 (UltraRM-13B 83.71, GPT-4-0613 88.69); MT-Bench human judgments 59 (UltraRM-13B 56, GPT-4-0613 63.87). The card documents `blender.rank`, `blender.compare`, and `best_of_n_generate` usage.

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| PairRanker (LLM-Blender paper, MixInstruct) | 400M | reward-model | backbone | DeBERTa-v3-large | arXiv:2306.02561v3 §5.1, App. A | verified 2026-09-14 | no ablation reported |
| same | 400M | reward-model | epochs; batch size | 5; 64 | App. A | verified 2026-09-14 | no ablation reported |
| same | 400M | reward-model | optimizer; LR schedule | Adafactor; max LR 1e-5, 5% warm-up, linear | App. A | verified 2026-09-14 | no ablation reported |
| same | 400M | reward-model | compute | single RTX 8000 GPU, two days | App. A | verified 2026-09-14 | — |
| same | 400M | reward-model | loss; head | BCE; five-layer tanh MLP on `<source>`+candidate embeddings | §3.2, App. A | verified 2026-09-14 | App. A: MSE and ranking losses and other embedding combinations tried; no numbers given |
| same | 400M | reward-model | supervision metric Q | BARTScore | §5.1 | verified 2026-09-14 | Table 3: BARTScore has the highest correlation with GPT-Rank among the automatic metrics |
| same | 400M | reward-model | training pairs per input; data | 5; MixInstruct 100K train / 5K val / 5K test | §3.3; §2.2 Table 1 | verified 2026-09-14 | "sufficient for obtaining decent results" (§3.3), no table |
| GenFuser (LLM-Blender) | 3B | SFT | base model; candidates fused | Flan-T5-XL; top K = 3 | §2.3, §4, App. A | verified 2026-09-14 | App. A: 3B better than Flan-T5-large; XXL marginal gain; no numbers |
| PairRM (llm-blender/PairRM) | 0.4B | reward-model | training data | summarize_from_feedback, webgpt_comparisons, synthetic-instruct-gptj-pairwise, hh-rlhf, chatbot_arena_conversations, UltraFeedback | model card "Training Datasets" @5b880cc | verified 2026-09-14 | no ablation reported |
| PairRM | 0.4B | reward-model | LR, epochs, batch, pair construction | not reported | model card @5b880cc; paper describes the earlier pair-ranker | not reported | — |

## Findings relevant to generality
- No single open LLM is best across instructions: the most frequent winner (Vicuna) is top-ranked on 21.22% of 5,000 instructions (§1, Fig. 1).
- Rankers trained on CNN/DM, CommonGen, and WMT18 candidates are applied to GPT-3 (text-davinci-003) outputs without GPT-3 training candidates; gains are 6.64% ROUGE-1 on CNN/DM (MaxLogits), 26.53% CIDEr on CommonGen (MaxLogits), and 11.65% BLEU on WMT18 zh-en (MaxWins) (App. C.4, Tables 4-6).
- Candidate-quality mismatch: training a reranker on candidates from a base model that had seen those training inputs "will result in fairly bad performance", so candidates are generated by half-finetuned models on their unseen halves (App. C.3).
- Order sensitivity: with shuffled candidate order, the reranker agrees with itself more than 90% of the time; accuracy is higher for pairs with larger rank difference (App. C.5, Fig. 7).

## Connections
- [[bradley-terry-rm]] — pointwise reward models score each response alone; §3.1 places the InstructGPT RM in this category.
- [[best-of-n]] — the model card's second use case is best-of-n reranking with PairRM.
- [[judge-llm-bias]] — position sensitivity of pairwise judges; App. C.5 reports PairRanker's order consistency.
- [[ultrafeedback]] — one of PairRM's six training datasets.
- [[reward-ensembling]], [[generative-reward-models]] — other RM designs covered in the library.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2306.02561 (PDF v3, 2023-06-30) and https://huggingface.co/llm-blender/PairRM (revision 5b880cc73776ac75a835b3e0bd5169bcb5be013b).
- Corrections to the previous card version:
  - Title "PairRM: Pairwise Reward Models for Response Ranking" → no artifact has this title; the paper title is used and the model card is cited separately.
  - "Input format `[CLS] x [SEP] y_A [SEP] y_B [SEP]`; score = σ(w · [CLS])" → special tokens `<source>`, `<candidate1>`, `<candidate2>`; two candidate scores from an MLP over source+candidate embeddings (§3.3, App. A).
  - "Loss … label from human annotations (MixInstruct) or from GPT-4 labels" → per-metric BCE against reference-based metrics, BARTScore on MixInstruct; MixInstruct oracle comparisons are by ChatGPT (§2.2, §3.2, §5.1). PairRM uses six human-preference datasets (model card).
  - "Swap-augmentation: always evaluate both orders, average logits … at train and inference time" → the paper shuffles candidate order within training pairs; at inference the full matrix holds both orders of every pair and MaxLogits combines row and column (`M_i^* − M_*^i`), but no separate swap-and-average step or order ablation is described (§3.3).
  - "Tournament Best-of-N: O(N log N)" → one bubble-sort pass with N − 1 comparisons, O(N); full matrix is O(N²) (§3.3).
  - "LLM-Blender Fig. 3 (MixInstruct reranking results)" → Fig. 3 shows ranker architectures; results are in Table 2.
  - "A 0.4B PairRM beats scalar RMs based on Llama-2-7B on MixInstruct and MT-Bench" → the paper has no Llama-2 RMs; the model card compares PairRM (0.4B) with UltraRM-13B, Llama-2-Chat judge LMs, and GPT-4 on Auto-J, HHH-Alignment, and MT-Bench human judgments (numbers above).
- Removed as unsupported by the source: "Ablation: no-swap vs swap-augment — swap brings 2–3 pp"; "PairRM-0.4B matches or beats scalar RMs at 7B"; "within a few points of GPT-4 on tight pairs"; "PairRM scales are flat across model sizes"; "popular reranker in open post-training stacks (e.g. Tülu SFT data selection, DPO pair filtering)"; "DPO pair filtering … lift DPO ~2 pp"; "inherits verbosity bias … length-balance training pairs"; "Used in Tülu-line and OpenRLHF recipes … before [[rlvr-tulu3]]"; "PairRM scores can be a prior that a GenRM critiques".
- Not reported by the source: PairRM training hyperparameters; seeds; an ablation of candidate-order shuffling.
