<!-- scope: IN2 training — synthetic 4K-32K long-context QA with randomly placed evidence segments, applied to Mistral-7B-Instruct-v0.2 (FILM-7B); VAL Probing
     deps: [[lost-in-the-middle]]
     see-also: [[artificial-needles-real-haystacks]], [[longalign]], [[longbench]], [[needle-in-haystack-data]], [[llama-2-long]]
-->

# Make Your LLM Fully Utilize the Context
- **Core Insight:** Fine-tuning Mistral-7B-Instruct-v0.2 for one epoch on 1.1M + 300K synthesized long-context QA examples (evidence segments shuffled among random segments, 4K-32K tokens) plus 150K short QA and 200K general instruction examples lowered the VAL Probing min-max position gap (all three tasks) from 56.2 to 13.9 and raised the 9-task LongBench average from 30.6 to 39.9, while MMLU moved from 59.3 to 59.2 (Table 1, Table 2, Fig. 5).
- **Guideline:** When a long-context model uses information at the start and end of its window better than in the middle, train on QA whose evidence segments are placed at random positions in a context whose length is balanced across the target window, and keep short-context QA and general instruction data in the mix, because this reduced position gaps on three probing tasks at 7B while MMLU changed by 0.1 (§2.1, Table 1, Fig. 5). The result was tested only at 7B and a 32K window.
- **Authors:** Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng, Jian-Guang Lou (Xi'an Jiaotong University, Microsoft, Peking University)
- **Year:** 2024 (arXiv v1 2024-04; no venue listed on arXiv)
- **URL:** https://arxiv.org/abs/2404.16811 (code: github.com/microsoft/FILM)
- **Source type:** paper
- **Relevant topics:** lost-in-the-middle, long-context SFT, synthetic long-context QA, position bias, long-context probing, RoPE base, short-context retention

## Abstract
Many LLMs accept long inputs but do not use information in the middle of the context (lost-in-the-middle). The authors hypothesize that the cause is insufficient explicit supervision showing that any position can hold important information. INformation-INtensive (IN2) training is a data-only fix: a synthesized QA dataset whose answers need (1) fine-grained awareness of one ~128-token segment inside a 4K-32K-token context, or (2) integration and reasoning over two or more segments. Applied to Mistral-7B, it yields FILM-7B (FILl-in-the-Middle). Three probing tasks (VAL Probing) cover document, code, and structured-data contexts with forward, backward, and bi-directional retrieval. FILM-7B retrieves from different positions across its 32K window, improves real long-context tasks (NarrativeQA F1 23.5→26.9), and keeps short-context results comparable (MMLU 59.3→59.2) (Abstract).

## Key Contributions
- A position-bias hypothesis: next-token loss depends mostly on nearby tokens, and SFT system messages sit at the start, so standard training may teach that important information is at the beginning and end (§1; Interpretation).
- IN2 data construction: two GPT-4-Turbo QA types over 128-token segments of C4 realnewslike, with the needed segments shuffled among random segments (§2.1, Fig. 2, Eq. 1-2).
- FILM-7B, trained from Mistral-7B-Instruct-v0.2 with loss on answers only (§2.2).
- VAL Probing: three ~32K-token retrieval tasks that vary context style and retrieval direction, proposed because Needle-in-the-Haystack uses document text and forward retrieval (§3.1-3.2).
- Ablations on sliding-window attention and RoPE base during IN2 training (§4.3, Fig. 6, Table 3).

## Key Figures/Tables to Study
- **Figure 2** — the two data-construction paths (single segment vs multiple segments).
- **Table 1** — VAL Probing Avg and min-max Gap for 8 models.
- **Table 2** — 9 LongBench tasks; **Figure 5** — 8 short-context tasks before and after IN2 training.
- **Table 3** — RoPE base θ ablation at 20% of the data; **Figure 6** — 4K sliding-window ablation.
- **Appendix D** — the exact generation prompts and the training template.

## Technical Details
**Data construction (§2.1).**
1. Corpus C is the C4 realnewslike subset; raw texts average ~600 Mistral tokens (§2.1 footnote 3).
2. Fine-grained awareness: randomly extract one 128-token segment s_i; GPT-4-Turbo writes (q_i, a_i) with an instruction that the pair be "highly specific" to s_i; L_i = concatenation of Shuffle(s_i, [r_j]), where [r_j] are randomly sampled 128-token segments from C (Eq. 1).
3. Integration and reasoning: split C_i into 128-token segments [s_i]; GPT-4-Turbo writes a multi-hop pair needing at least two segments; [s_i] and [r_j] are jointly shuffled, so required segments can be far apart (Eq. 2).
4. Length balance: L_i lengths are evenly distributed from 4K to 32K tokens, implemented by rejection sampling on [r_j] (§2.1).
5. Retention: ~10% of QA pairs keep the original text C_i as a short context, and OpenOrca general instruction data is added (§2.1).
6. Final mix: 1.1M fine-grained long (~63%), 300K integration long (~17%), 150K short QA (~9%), 200K general instruction (~11%) (§2.1).
7. Decontamination: a raw text with a 10-gram overlap with any evaluation example (probing, real-world, short-context) is used neither for QA generation nor as a random segment (App. A).

**Evaluation setup (§3.2, §4.1).** VAL contexts are ~32K tokens, ~3K examples per task. Document: sentences from arXiv abstracts, retrieve the sentence containing a given piece (bi-directional), word-level recall. Code: StarCoder Python functions, retrieve the function name for a given line (backward), exact match. Database: Wikidata entities (ID, label, description), retrieve label and description for an ID (forward), relaxed exact match. All models use greedy decoding; LongBench inputs use middle truncation to 32K; ROUGE-L for summarization, F1 otherwise. Short tasks: MMLU 5-shot, GSM8K 8-shot, MATH 4-shot, others 0-shot.

**Results.**
- VAL Probing Avg/Gap, FILM-7B vs GPT-4-Turbo vs Mistral-7B-Instruct-v0.2: document 85.4/6.1 vs 81.3/31.7 vs 74.2/32.1; code 83.3/18.7 vs 66.1/46.5 vs 20.3/59.5; database 89.0/16.8 vs 89.6/18.0 vs 47.5/77.0; all 85.9/13.9 vs 79.0/32.1 vs 47.3/56.2 (Table 1).
- LongAlign-13B-64K and InternLM2-chat-20B score All Avg 68.5 and 68.2 with Gaps 27.1 and 43.3, although both are near-perfect on Needle-in-the-Haystack (Table 1, §4.2). FILM-7B is also near-perfect on it within 32K (App. C, Fig. 7).
- LongBench, Mistral-7B-Instruct-v0.2 → FILM-7B: NarrativeQA 23.5→26.9, Qasper 33.8→42.2, MultiFQA 45.9→56.0, HotpotQA 42.4→62.1, 2WikiMQA 24.3→47.0, MuSiQue 20.8→39.0, GovReport 33.3→33.8, QMSum 24.8→25.1, MultiNews 26.8→26.9; average 30.6→39.9; GPT-4-Turbo average 44.7 (Table 2).
- Training cost: ~300 GPU days on 16 nodes of 8×A100-80G (§2.2).

**Ablations (20% of the training examples, §4.3).**
- A 4K sliding window, applied in both pre-training and IN2 training (Mistral-7B-Instruct-v0.1 backbone) or only in IN2 training (v0.2 backbone), gives performance that drops when the question-to-information distance exceeds the window (Fig. 6).
- RoPE base θ, All Avg/Gap: 1.0×10^6 (backbone default) 80.3/23.6; 2.0×10^6 83.8/16.5; 1.0×10^7 84.9/14.3; 1.0×10^8 84.6/14.0 (Table 3). The authors recommend 10× the default base for IN2 training, since information intensity rises while context length stays the same (§4.3).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| FILM-7B | 7B | long-context | initialization | Mistral-7B-Instruct-v0.2 | arXiv:2404.16811v2 §2.2 | verified 2026-09-14 | no ablation for the main run; Fig. 6 (20% data): 4K sliding-window settings lose information beyond the window |
| FILM-7B | 7B | long-context | examples by type (count share) | 1.1M fine-grained long (~63%); 300K integration long (~17%); 150K short QA (~9%); 200K OpenOrca instruction (~11%) | §2.1 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | context length of long examples | evenly distributed 4K-32K tokens; segments of 128 tokens | §2.1 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | QA generator and prompts | GPT-4-Turbo; two prompts (single segment; at least two pieces) | §2.1, App. D Ex. 7-8 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | loss masking; template | loss on answer tokens only; Mistral [INST] template | §2.2, App. D Ex. 9 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | global batch; epochs; steps | 128; 1 epoch; ~14K steps | §2.2 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | peak LR; schedule; warmup | 1e-6; cosine decay; 3% of steps | §2.2 | verified 2026-09-14 | no ablation reported |
| FILM-7B | 7B | long-context | RoPE base θ | backbone default (1.0×10^6 per Table 3) | §4.3, Table 3 | verified 2026-09-14 | Table 3 (20% data): 1.0×10^7 gives All Avg 84.9 vs 80.3 at default; not used for released FILM-7B |
| FILM-7B | 7B | long-context | compute; parallelism | ~300 GPU days; 16×8 A100-80G; FSDP full sharding + CPU offload | §2.2 | verified 2026-09-14 | not applicable |
| FILM-7B | 7B | long-context | optimizer, betas, weight decay, final LR, packing, total tokens | not reported | checked §2, §4.3, App. A-D | not reported | — |

## Findings relevant to generality, long context, and distillation
- **Transfer from synthetic to real tasks:** all long contexts are synthesized from short news segments, and LongBench QA and multi-hop tasks improve (Table 2). Summarization tasks change by at most 0.5 (Table 2). The authors conclude the learned capability transfers to real tasks (§4.2; Interpretation).
- **Short-context retention:** Mistral-7B-Instruct-v0.2 → FILM-7B: MMLU 59.3→59.2, BoolQ 85.4→87.7, RACE-H 46.0→45.6, CSQA 70.0→75.3, ARC-C 55.9→52.5, HellaSwag 83.6→79.1, GSM8K 40.4→44.5, MATH 8.7→11.3 (Fig. 5). The authors describe these as "almost comparable with minor variances" (§4.2); ARC-C and HellaSwag decrease by 3.4 and 4.5.
- **Evaluation validity:** near-perfect Needle-in-the-Haystack scores coexist with large position gaps on VAL Probing, which the authors attribute to NIAH's document style and forward (copy-like) retrieval pattern (§3.1, §4.2; Interpretation).
- **Synthetic-data generation (distillation-style):** stage = long-context instruction tuning; teacher = GPT-4-Turbo; prompt types = single-segment specific QA and multi-piece multi-hop QA; quality control = prompt rules (answerable from the context, unambiguous, no "according to the context", answer in own words) and 10-gram decontamination (App. A, App. D). Sampling parameters and any answer verification step are not reported.

## Connections
- [[lost-in-the-middle]] — the position-bias problem IN2 training targets.
- [[needle-in-haystack-data]], [[ruler]] — probing tasks the paper contrasts with VAL Probing (§3.1, §5).
- [[longalign]], [[longbench]] — LongAlign models are baselines (Table 1-2); LongBench supplies the 9 real tasks.
- [[artificial-needles-real-haystacks]] — a separate study that fine-tunes on synthetic retrieval data to improve long-context retrieval.
- [[llama-2-long]], [[rope-base-bounds-context-length]] — RoPE base increases for long context, which §4.3 extends to higher information density.
- [[induction-heads]] — the mechanism the authors cite for why forward retrieval is easy (§3.1).
- [[pam-qa-never-lost-in-middle]], [[context-synthesis-short-to-long]] — other data-side responses to long-context utilization.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2404.16811 (arXiv v2, 2024-04-26; v1 2024-04-25). Figure 5 bar values read from the PDF text layer and paired by x-position; the MMLU pair matches the abstract.
- Audit claims not found in the source: "Caveat from Xiong et al. 2024: IN2 data cost 2.19 TriviaQA points on Mistral-7B" (a claim from a different paper); "rejection sampling puts the evidence at any position" (rejection sampling is used for length balance; random positions come from the shuffle, §2.1); "filler segments act as easy, non-retrieved distractors" (the paper describes [r_j] only as randomly sampled segments).
- Not reported by the source: optimizer and weight decay, sampling temperature for GPT-4-Turbo, total training tokens, selection of filler segments by similarity (hard distractors), results beyond 32K or at other model sizes.
