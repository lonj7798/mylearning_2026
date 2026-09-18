<!-- scope: Retrieval Head (Wu, Wang, Xiao, Peng, Fu; arXiv Apr 2024) — detection of attention heads that copy needle tokens from long context, their sparsity, stability across long-context continued pretraining / chat tuning / upcycling, and causal masking effects on NIAH, extractive QA, and chain-of-thought
     deps: [[needle-in-haystack-data]], [[induction-heads]]
     see-also: [[long-context-data-engineering]], [[streamingllm-attention-sinks]], [[nextlong]], [[loongrl]], [[babilong]]
-->

# Retrieval Head Mechanistically Explains Long-Context Factuality
- **Core Insight:** In the 4 model families examined (Llama-2, Yi, Qwen1.5, Mistral/Mixtral), less than 5% of attention heads act as retrieval heads that copy needle tokens from the input, the same heads appear in the short-context base model and in its long-context, chat, or MoE derivative (Pearson correlation of retrieval scores above 0.8), and masking the top 50 of them (about 5% of heads) drops every model's Needle-in-a-Haystack score below 50 (Abstract; §3.1, §3.3, §4.1).
- **Guideline:** When a KV-cache compression, sparse-attention, or architecture change is evaluated for long-context use, test whether it preserves retrieval heads and full-context access for them, because masking these heads caused hallucination on NIAH and lowered extractive-QA F1 and chain-of-thought accuracy while masking random heads had smaller effects (§4.1–4.3, §5).
- **Authors:** Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu (Peking University; University of Washington; MIT; UIUC; University of Edinburgh)
- **Year:** 2024 (arXiv v1 2024-04; preprint)
- **URL:** https://arxiv.org/abs/2404.15574
- **Source type:** paper
- **Relevant topics:** mechanistic interpretability, long-context retrieval, attention heads, hallucination and factuality, context-length extension, chain-of-thought, KV-cache compression

## Abstract
The paper asks how transformer language models retrieve information from arbitrary positions in a long context. Across 4 model families, 6 model scales, and 3 types of finetuning, it identifies retrieval heads: attention heads that are mostly responsible for retrieving relevant information from long context. The reported properties are: (1) universal, every explored model with long-context capability has them; (2) sparse, less than 5% of heads; (3) intrinsic, they exist in models pretrained with short context and the same set performs retrieval after continual pretraining to 32–128K; (4) dynamically activated, for Llama-2 7B 12 retrieval heads attend to the required information in every context while the rest activate in some contexts; (5) causal, fully pruning them causes retrieval failure and hallucination, while pruning random non-retrieval heads does not. Retrieval heads also strongly affect chain-of-thought reasoning that refers back to the question and generated context, and affect less the tasks answered from intrinsic knowledge (Abstract). Code: github.com/nightdessert/Retrieval_Head.

## Key Contributions
- A retrieval score that measures how often a head's most-attended input token is the needle token being generated (§2, Eq. 1).
- A detection protocol based on Needle-in-a-Haystack samples with needles unrelated to the haystack and not answerable from model knowledge (§2).
- Measurements of sparsity, context sensitivity, and cross-variant stability of retrieval heads over 11 checkpoints (Table 1; §3; Figs. 3–6).
- Causal masking experiments on NIAH, extractive QA, and MMLU/MuSiQue/GSM8K with and without CoT (§4; Figs. 7–11).

## Key Figures/Tables to Study
- **Fig. 2** and **Eq. 1** — definition of copy-paste behavior and the retrieval score.
- **Fig. 3** — distribution of retrieval scores per model.
- **Fig. 5 / Fig. 6** — heatmaps and correlation of retrieval scores between base and derived models.
- **Fig. 7 / Fig. 8** — masking top-K retrieval heads vs K random heads on NIAH and extractive QA.
- **Fig. 10 / Fig. 11** — effect on tasks with and without CoT, with example failures.

## Technical Details
### Retrieval score
- During greedy decoding, head h copies a token when (1) the generated token w belongs to the needle k, and (2) the input token x_j with the highest attention weight of h is inside the needle span i_q and equals w (§2).
- Retrieval score of h = |g_h ∩ k| / |k|, where g_h is the set of tokens h copied and |k| the needle length in tokens (Eq. 1). Example from the paper: a score of 0.9 on a 10-token needle means 9 of its tokens were copied by that head (§2).
- Detection: three sets of (q, k, x) samples; for each, 20 lengths uniformly sampled from 1K–50K and 10 depths from start to end, about 600 retrieval tests per model; the head's score is averaged over tests; a head with score above 0.1 is counted as a retrieval head (§2).
- Activation frequency: fraction of tests in which a head copies at least one token, reported alongside the retrieval score to measure context sensitivity (§3.2, Fig. 4).

### Models (Table 1)
- Llama-2-7B → Llama-2-7B-80K (continued pretraining); Llama-2-13B-64K; Mistral-7B-v0.2 → Mistral-7B-Instruct-v0.2 (SFT and RLHF) and Mixtral-8x7B-v0.1 (sparse upcycling); Yi-6B → Yi-6B-200K; Yi-34B-200K; Qwen1.5-14B → Qwen1.5-14B-Chat (Table 1).

### Properties
- Sparsity: 45%–73% of heads have score 0; 25%–52% have scores between 0 and 0.1; about 3%–6% exceed 0.1 (§3.1). Fewer than 5% of heads exceed 0.5 (Fig. 3 caption).
- Context sensitivity: Llama-2-7B-80K has 12 and Yi-6B-200K has 36 strongest heads with activation frequency 1 in all tested contexts (§3.2, Fig. 4).
- Stability: Pearson correlation of retrieval-score maps exceeds 0.8 between a base model and its variants and is below 0.1 across families (§3.3); Fig. 6 shows 0.84 (Llama-2-7B vs 80K), 0.87 (Qwen1.5-14B vs Chat), 0.91 (Mistral-7B-v0.2 vs Mixtral-8x7B).

### Masking experiments
- NIAH (held-out needle set): masking retrieval heads lowers scores more than masking random heads; at K = 50 (about 5% of heads) all models fall below 50 (§4.1, Fig. 7). Llama-2-7B-80K: masking the top 20 retrieval heads gives 63.6 accuracy vs 94.7 for 20 random heads (Fig. 1).
- Error types: incomplete retrieval (appears when heads with score above 0.4 are masked), hallucination (retrieval heads attend mostly to the initial token, the "attention sink"), and wrong extraction (retrieval heads active but on the wrong span) (§4.1).
- Extractive QA on Mistral-7B-Instruct-v0.2 (32K): GPT-4-generated QA pairs from recent news paragraphs; masking retrieval heads reduces F1 by 9.2% and 23.1% (Fig. 8 plots 50 and 100 masked heads); random masking shows no significant change (§4.2).
- CoT: with answer-only prompting on MMLU, MuSiQue, and GSM8K, masking retrieval or random heads changes performance little; with CoT, masking retrieval heads lowers performance, and failure cases misread input facts (§4.3, Figs. 10–11).

## Findings relevant to generality and long context
- **Context extension.** The authors state that retrieval ability is "an intrinsic property" of the base model from large-scale pretraining, with "subsequent small-scale training exerting only minor alterations" to head activation patterns (§3.3, citing Fu et al. 2024). Evidence is score-map correlation on three base/variant pairs, not a training-data ablation. Interpretation.
- **Alignment.** Qwen1.5-14B-Chat (SFT and RLHF) keeps the retrieval-score pattern of Qwen1.5-14B (Fig. 5; Pearson 0.87 in Fig. 6). Mistral-7B-Instruct-v0.2 is listed in Table 1 but is not shown in Figs. 5–6. Result (single study).
- **Knowledge vs context tasks.** Masking retrieval heads affects tasks that read from context (NIAH, extractive QA, CoT that refers back) more than answer-only tasks that rely on parametric knowledge (§4.2–4.3). Result (single study; downstream tests use one model).
- **Architecture.** The authors know of no linear-attention or SSM model that passes NIAH, and report that Mistral v0.1 (sliding-window attention) fails NIAH while v0.2 (full attention) passes; they argue retrieval heads need the full KV cache (§5). Interpretation.
- **KV cache.** For Llama-2-7B, a 100K-token KV cache needs more than 50GB vs less than 1GB at 2K; the authors propose pruning KV entries of non-retrieval heads as future work (§5).

## Connections
- [[induction-heads]] — the head-level mechanism that motivated the copy-paste hypothesis (§1).
- [[needle-in-haystack-data]] — the test used for detection and for masking evaluations (§2, §4.1).
- [[long-context-data-engineering]] — Fu et al. 2024, source of the Llama-2-7B-80K and 13B long-context checkpoints and of the "intrinsic property" framing (§3, §3.3).
- [[streamingllm-attention-sinks]] — attention sink on initial tokens seen during hallucination (§4.1).
- [[babilong]] — the paper cites the earlier Kuratov et al. 2024 "needles in a 10M haystack" preprint (arXiv:2402.10790, the precursor of BABILong) for tasks that interleave retrieval and reasoning (§1).
- [[llama-2]], [[yi]], [[mixtral]] — model families analyzed (Table 1).
- [[nextlong]] — also probes attention mass on distant context as a long-dependency measure (NExtLong §5.1).
- [[loongrl]] — long-context RL whose outputs interleave retrieval and reasoning.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2404.15574 (arXiv v1, 2024-04-24; only version).
- Audit claims not found in the source: "This explains Fu 2024's finding that 0.5-5B tokens are enough: extension mainly unlocks positions for a retrieval ability already learned in pretraining" — the paper cites Fu et al. 2024 for the "intrinsic property" framing (§3.3) but gives no token counts and does not claim to explain a data-amount result.
- Internal inconsistencies in the source: the always-activated head count for Llama-2 7B is 12 (Abstract; §3.2, for Llama-2-7B-80K) and 13 (§1); the 13B long-context model is "Llama-2-13B-64K" in Table 1 and "Llama-2-13B-60K" in §3; "retrieval head" uses a 0.1 score threshold (§2) while the "less than 5%" statistic uses 0.5 (Fig. 3 caption).
- Not reported in text: per-model numeric values behind Figs. 7 and 10 (figure-only), the number of heads masked in the CoT experiments beyond the 50/100 axis ticks.
