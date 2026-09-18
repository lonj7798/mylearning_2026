<!-- scope: Llama 2 Long — continual long-context pretraining of Llama 2 (400B tokens, RoPE ABF base 500,000), self-instruct long QA instruction tuning, and ablations on PE, data mix, and length curriculum
     deps: [[llama-2]]
     see-also: [[position-interpolation]], [[long-context-llama3]], [[self-instruct]], [[long-context-data-engineering]], [[in2-film]]
-->

# Effective Long-Context Scaling of Foundation Models
- **Core Insight:** Continually pretraining Llama 2 on 400B additional tokens at 32,768-token (7B/13B) or 16,384-token (34B/70B) sequences, with the RoPE base frequency raised from 10,000 to 500,000, raised 70B QuALITY EM from 53.0 to 79.7 and 70B MMLU from 68.9 to 71.7, and the 70B chat model tuned without human-annotated long data averages 37.7 on ZeroSCROLLS vs 36.7 for gpt-3.5-turbo-16k (§1, §2.1, §4.1, Tables 1, 3, 4).
- **Guideline:** When extending a RoPE model's context, raise the RoPE base frequency and continue pretraining from the short-context checkpoint, because in 7B ablations RoPE ABF was the only variant that held FIRST-SENTENCE-RETRIEVAL performance to 32,768 tokens and a 4,096→32,768 switch at 80% of training used 2.270×10^22 instead of 3.783×10^22 FLOPs with three of four long-QA scores within 0.5 and Qasper F1 25.0 vs 28.6 (§4.1, Fig. 5b, Table 10). When choosing long-context pretraining data, check corpus quality before raising the long-document share, because removing most long texts kept most of the 7B long-task gain (§4.2, Table 7).
- **Authors:** Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, et al. (GenAI, Meta)
- **Year:** 2023 (arXiv v1 2023-09; v3 2023-11; no venue listed on arXiv)
- **URL:** https://arxiv.org/abs/2309.16039
- **Source type:** paper
- **Relevant topics:** long-context continual pretraining, RoPE base frequency (ABF), position interpolation, long-context data mix, length curriculum, synthetic long-context instruction data, short-task retention

## Abstract
The paper presents Llama 2 Long, a series of models with effective context windows up to 32,768 tokens, built by continual pretraining from Llama 2 with longer sequences on a dataset where long texts are upsampled. Evaluation covers language modeling, synthetic probing, and research benchmarks: the models improve most regular tasks and long-context tasks over Llama 2. With an instruction-tuning procedure that uses no human-annotated long data, the 70B chat variant exceeds gpt-3.5-turbo-16k's overall score on a suite of long-context tasks. Analyses cover Llama's position encoding and its limit on long dependencies, the pretraining data mix, and the sequence-length curriculum; the ablations indicate that abundant long texts are not the key to strong performance and that long-context continual pretraining is more efficient than, and similarly effective as, long-sequence pretraining from scratch (Abstract).

## Key Contributions
- RoPE ABF: increase the RoPE base frequency b from 10,000 to 500,000 to reduce the decay of attention scores for distant tokens (§4.1, Fig. 4), with a theoretical comparison to position interpolation (App. B).
- A continual-pretraining recipe for 7B-70B Llama 2 checkpoints: 400B tokens over 100,000 steps (§2.1).
- Instruction tuning from Llama 2 Chat RLHF data plus self-instruct long-document QA generated and self-critiqued by Llama 2 Chat, with LM loss on long inputs (§2.2, App. D).
- Ablations separating data quality from length distribution (§4.2) and continual vs from-scratch long training (§4.4).
- A power-law-plus-constant fit of validation loss vs context length per model size (Fig. 1).

## Key Figures/Tables to Study
- **Table 1 / Table 2** — short-task scores of Llama 2 vs Llama 2 Long at each size.
- **Table 3 / Figure 2** — long QA vs open long-context models; scores as prompt length grows 4,096→16,384.
- **Table 4** — ZeroSCROLLS for Llama 2 Long Chat 70B vs GPT-3.5/GPT-4/Claude.
- **Figure 4, Table 5, Table 6, Figure 5** — PE variants: attention decay, perplexity, short tasks, retrieval probe.
- **Tables 7-8** — data-mix ablation; **Table 9** — instruction-tuning mixes; **Tables 10-11** — curriculum.

## Technical Details
**Architecture and efficiency (§2.1).** The architecture is unchanged except the PE. Sparse attention is not used: for 70B (h = 8192), attention becomes the compute bottleneck only beyond 49,152 (6h) tokens. With FlashAttention, going from 4,096 to 16,384 tokens costs about 17% speed for 70B with negligible memory overhead.

**Position encoding (§4.1, 7B, 80B extra tokens at 32,768).**
- With unmodified RoPE, the model could not attend beyond 4,000-6,000 tokens on FIRST-SENTENCE-RETRIEVAL even after long continual pretraining.
- Validation perplexity (Books / CC / Wikipedia): RoPE 6.548/6.816/3.802; PI 6.341/6.786/3.775; RoPE ABF 6.323/6.780/3.771; xPos ABF 6.331/6.780/3.771 (Table 5).
- Short tasks (HumanEval / Math / MMLU / HellaSwag / TQA): RoPE 14.63/3.62/45.69/76.31/65.23; PI 15.24/3.08/45.84/76.65/65.96; ABF 17.07/3.52/46.24/76.73/66.04 (Table 6).
- RoPE ABF is the only variant that keeps FIRST-SENTENCE-RETRIEVAL performance to 32,768 tokens (Fig. 5b). All variants except RoPE reach perfect PASSKEY accuracy, which the authors consider too simple as a probe (§4.1 footnote 4).
- The authors argue that the relative distance between embedded vectors depends linearly on the PI parameter and logarithmically on the ABF base, consistent with the base being insensitive to its exact value (§4.1, App. B; Interpretation).

**Data mix (§4.2, 7B).** Long-task gains over 7B Llama 2 (4,096 window), as relative Δ for NarrativeQA / Qasper / QuALITY / QMSum: Llama 2 Long mix 23.70%/43.64%/75.5%/45.70%; Llama 2 mix 18.23%/38.12%/60.3%/44.87%; Llama 2 mix with long text removed 19.48%/39.14%/67.1%/36.60%; with long text upsampled 22.15%/36.82%/65.0%/42.83% (Table 7). MMLU: 48.62 (Llama 2 Long mix) vs 46.30 (Llama 2 mix) vs 46.25 (removed) vs 46.25 (upsampled) (Table 8). The authors conclude the gain of their mix comes mostly from data quality, not length distribution (§4.2; Interpretation).

**Curriculum (§4.4, 7B, 4M tokens per gradient update, equal total tokens).** FLOPs and NarrativeQA F1 / Qasper F1 / QuALITY EM / QMSum ROUGE-geo: 32K from scratch 3.783×10^22, 18.5/28.6/37.9/11.46; 4k→32k at 20% 3.405×10^22, 20.0/28.1/38.8/12.09; at 40% 3.026×10^22, 20.1/27.0/37.4/12.44; at 80% 2.270×10^22, 18.5/25.0/38.3/11.00 (Table 10). CC perplexity 7.67 (scratch) vs 7.59 (all switch points) (Table 11). The paper summarizes this as saving around 40% FLOPs with almost no loss (§4.4).

**Main results.**
- Short tasks, Llama 2 → Llama 2 Long 70B: Coding 37.4→39.9, Math 35.2→41.3, MMLU 68.9→71.7, Commonsense 71.9→72.7, OpenQA 63.6→64.0; 7B MMLU 45.3→47.8, 13B 54.8→60.1, 34B 62.6→65.0 (Table 1). The authors attribute these to additional FLOPs and knowledge from the new long data (§3.1).
- Long QA at 16,384 max prompt, Llama 2 70B → Llama 2 Long 70B: NarrativeQA 25.7→30.9, Qasper 27.5→35.7, QuALITY 53.0→79.7, QMSum 11.9→16.5 (Table 3). The paper states each long task improves monotonically as maximum input length rises from 4,096 to 16,384 (§3.1, Fig. 2); in Fig. 2 QuALITY EM is 80.3 at 8,192 and 79.7 at 16,384.
- Loss vs context length fits L(c) = (α/c)^β + γ, where L = validation loss, c = context length, and α, β, γ are fitted per model size. For 70B, β = 0.51, so doubling c multiplies the first term by 2^−β ≈ 0.7; larger models have larger β (Fig. 1, §3.1).
- Llama 2 Long Chat 70B on ZeroSCROLLS: average 37.7 vs gpt-3.5-turbo-16k 36.7, Claude (8k) 39.1, GPT-4 (8k) 41.7; higher than gpt-3.5-turbo-16k on 7 of 10 tasks (Table 4).
- App. C: the 70B model trained at 16,384 maintains validation loss in the 16,384-32,768 extrapolation region with ABF, with some degradation on the retrieval probe when extrapolating (Fig. 9).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Llama 2 Long | 7B-70B | long-context | initialization | Llama 2 checkpoints (4,096 context) | arXiv:2309.16039v3 §1, §2.1, Table 7 caption | verified 2026-09-14 | Tables 10-11 (7B): continual vs from-scratch curriculum |
| Llama 2 Long | 7B, 13B | long-context | sequence length | 32,768 tokens | §1 | verified 2026-09-14 | no ablation reported for the length choice |
| Llama 2 Long | 34B, 70B | long-context | sequence length | 16,384 tokens | §1 | verified 2026-09-14 | no ablation reported |
| Llama 2 Long | 7B-70B | long-context | tokens seen; steps | 400B tokens; 100,000 steps | §2.1 | verified 2026-09-14 | no ablation reported |
| Llama 2 Long | 7B-70B | long-context | tokens per batch | same number as Llama 2 (number not restated) | §2.1 | verified 2026-09-14 | no ablation reported |
| Llama 2 Long | 7B, 13B | long-context | peak LR; schedule; warmup | 2e-5; cosine; 2,000 steps | §2.1 | verified 2026-09-14 | no ablation reported |
| Llama 2 Long | 34B, 70B | long-context | peak LR | 1e-5 (schedule and warmup not separately stated) | §2.1 | verified 2026-09-14 | §2.1: smaller LR needed "to get monotonically decreasing validation losses"; no table |
| Llama 2 Long | 7B-70B | long-context | RoPE base frequency b | 10,000 → 500,000 (RoPE ABF) | §2.1, §4.1 | verified 2026-09-14 | 7B, 80B tokens: Table 5, Table 6, Fig. 5b vs RoPE, PI, xPos ABF |
| Llama 2 Long | 7B-70B | long-context | data mix | Llama 2 pretraining data + new long text data, long data up-weighted; proportions not reported | §2.1, §4.2 | verified 2026-09-14 | Tables 7-8 (7B) |
| PE ablation runs | 7B | long-context | extra tokens; sequence length | 80B; 32,768 | §4.1 | verified 2026-09-14 | not applicable |
| Curriculum ablation runs | 7B | long-context | tokens per update; switch point | 4M tokens; 4,096→32,768 at 20%, 40%, 80% of training | §4.4 | verified 2026-09-14 | Tables 10-11 |
| Llama 2 Long Chat | 70B | SFT | data | Llama 2 Chat "RLHF V5" short data + self-instruct long-document QA (normal and short answers) from Llama 2 Chat; counts not reported | §2.2, §4.3, App. D | verified 2026-09-14 | Table 9 (model size not stated) |
| Llama 2 Long Chat | 70B | SFT | packing | short data concatenated into 16,384-token sequences; long instances right-padded, one per sequence | §2.2 | verified 2026-09-14 | no ablation reported |
| Llama 2 Long Chat | 70B | SFT | loss masking | LM loss on output tokens and on long input prompts | §2.2, §4.3 | verified 2026-09-14 | Table 9: with vs without LM loss on inputs |
| Llama 2 Long Chat | 70B | SFT | LR, epochs, example counts, short/long ratio | not reported | checked §2.2, §3.2, §4.3, App. A-D | not reported | — |

## Findings relevant to generality, long context, and synthetic data
- **Short-task retention:** short benchmarks are on par or higher at every size (Table 1), but the comparison includes 400B extra training tokens and new data, which the authors name as the cause (§3.1). PE choice matters for short tasks at 7B: PI trails ABF on all five short tasks in Table 6.
- **Safety and factuality after long tuning:** TruthfulQA 64.14 (Llama 2 Chat 70B) vs 60.95 (Llama 2 Long Chat 70B); ToxiGen 0.01 vs 0.00; BOLD 0.41 vs 0.40 (Table 12).
- **Synthetic long instruction data:** stage = instruction tuning of the 70B chat model; generator = Llama 2 Chat (size not stated) on chunks of long pretraining documents; prompt types = normal-answer and short-answer QA, chosen with equal probability; quality control = XML-tag answer extraction and a self-critique step where Llama 2 Chat verifies the generated answer; training instance = full document truncated to the maximum context length + question + answer (§2.2, App. D, Fig. 10-11).
- **Instruction-mix ablation (model size not stated):** Qasper / NarrativeQA / QuALITY / SummScreenFD / QMSum: RLHF V5 only 22.3/13.2/71.4/14.8/16.9; + pretrain data 23.7/16.6/76.2/15.7/17.8; + self-instruct without LM loss on inputs 35.7/22.3/59.3/12.2/13.4; + self-instruct with LM loss 38.9/23.3/77.3/14.5/18.5 (Table 9). The authors report the chat model is strongest on QA, the main type of the self-instruct data (§3.2).
- **Limits stated by the authors:** not tuned for long-form output tasks such as creative writing; the 32k-vocabulary tokenizer produces about 10% more tokens than GPT-3.5's; hallucination observed (§6).

## Connections
- [[llama-2]] — the base checkpoints and the RLHF V5 short instruction data.
- [[position-interpolation]] — the PI baseline that ABF is compared against (§4.1).
- [[long-context-llama3]], [[llama-3]] — Llama 3 sets RoPE base 500,000 and cites Xiong et al. (2023) for its effectiveness up to 32,768 tokens (arXiv:2407.21783v3 §3.2).
- [[self-instruct]] — the method adapted for long-document QA generation.
- [[long-context-data-engineering]], [[prolong]] — later studies of long-context continual-pretraining data.
- [[in2-film]] — cites this paper for enlarging the RoPE base with context length and applies the idea to information density.
- [[rope-base-bounds-context-length]], [[yarn]], [[eleuther-extending-the-rope]] — other analyses of RoPE base and scaling.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2309.16039 (arXiv v3, 2023-11-14; v1 2023-09-27). Llama 3 citation checked in arXiv:2407.21783v3 §3.2.
- Audit claims not found in the source: "the full long document (truncated to 32,768 tokens)" (the paper says truncated to fit the model's maximum context length, §2.2); "Corrects ch-28/ch-32, which present 500,000 as a per-stage Llama 3 rescale" (a claim about course files, not this paper). Scope note: "ABF beats PI on short tasks" is tested only at 7B with 80B tokens (Table 6).
- Not reported by the source: pretraining data proportions and long-text share, tokens per batch as a number for the main runs, schedule and warmup for 34B/70B, instruction-tuning LR, epochs, example counts, and short/long mix ratio, the size of the Llama 2 Chat generator.
