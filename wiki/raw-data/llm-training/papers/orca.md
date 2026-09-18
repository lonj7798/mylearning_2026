<!-- scope: Orca (Microsoft Research, Jun 2023) — explanation tuning of a 13B LLaMA student on FLAN-v2 zero-shot queries answered by ChatGPT (5M) then GPT-4 (1M) under 16 system messages
     deps: [[distilling-step-by-step]]
     see-also: [[orca-2]], [[dolphin]], [[self-instruct]], [[evol-instruct]], [[alpaca]]
-->

# Orca: Progressive Learning from Complex Explanation Traces of GPT-4
- **Core Insight:** A 13B LLaMA-based model trained on 5M FLAN-v2 zero-shot queries answered by ChatGPT, then on a 1M subset answered by GPT-4, with system messages that request explanations, scores 49.7 on Big-Bench Hard (ChatGPT 48.9, Vicuna-13B 23.3) and 41.7 on AGIEval (ChatGPT 47.2) in zero-shot evaluation without CoT (Tables 8, 11).
- **Guideline:** When evaluating a model distilled from a stronger teacher, use human-labeled reasoning benchmarks in addition to GPT-4-judged open-ended prompts, because the paper reports that GPT-4 judging places Vicuna-13B at 92% of ChatGPT while exam accuracy places it at 64% and BBH accuracy at 48% (§1, Figs. 1-3).
- **Authors:** Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, Ahmed Awadallah (Microsoft Research)
- **Year:** 2023 (arXiv v1 2023-06; the PDF is marked "Work in progress"; no later version)
- **URL:** https://arxiv.org/abs/2306.02707
- **Source type:** paper
- **Relevant topics:** explanation tuning, imitation of closed teachers, FLAN-v2 query sampling, system messages, teacher assistant, evaluation of imitation models

## Abstract
Earlier small models trained on outputs of large foundation models (LFMs) receive shallow imitation signals, use small and homogeneous data, and are evaluated in ways that overestimate them; the authors state that such models learn the teacher's style but not its reasoning process. Orca is a 13B model trained on GPT-4 responses that contain explanation traces and step-by-step thought processes, with ChatGPT used as a teacher assistant, on large and diverse imitation data selected by sampling. Orca exceeds Vicuna-13B by more than 100% on Big-Bench Hard (BBH) and by 42% on AGIEval, matches ChatGPT on BBH, and is within 4 points of ChatGPT on SAT, LSAT, GRE, and GMAT questions when the system message is optimized, in zero-shot settings without CoT; it trails GPT-4.

## Key Contributions
- Explanation tuning: each training example is a triple ⟨system message, user query, LFM response⟩, where 16 hand-written system messages request detailed, step-by-step, or explain-like-I'm-five answers, and also short answers (§3.1, §3.1.1, Table 2).
- Query source: zero-shot queries sampled from four FLAN-v2 (Flan 2022) sub-collections — CoT, NiV2, T0, Flan 2021; the Dialogue sub-collection is skipped (§3.1.2, Table 3).
- Two-stage teacher schedule ("progressive learning"): train on FLAN-5M with ChatGPT responses, then on FLAN-1M, a random 1M subset of FLAN-5M with GPT-4 responses (§3.1.3).
- Evaluation beyond GPT-4 judging: AGIEval, BBH, TruthfulQA-MC, and ToxiGen, all zero-shot (§4, §6, §7).

## Key Figures/Tables to Study
- **Table 2:** the 16 system messages. **Table 3:** FLAN-v2 sampling per sub-collection.
- **Table 8 / Table 11:** AGIEval and BBH per-task results against Text-Davinci-003, ChatGPT, GPT-4, Vicuna-13B.
- **Table 10:** teacher-assistant ablation (FLAN-5M then FLAN-1M vs FLAN-1M only).
- **Table 7 + footnote 18:** GPT-4-as-judge scores and the position bias observed in them.

## Technical Details
- Sampling (Table 3): CoT 150K of 150K (no sampling; 18 tasks); NIV2 440K of 5M (at most 300 queries per task over 1,560 tasks); Flan 2021 2.5M of >28.9M; T0 2M of 85.7M; Dialog 0 of 22.5M. Flan 2021 and T0 use Algorithm 1 (pick a task uniformly, then a query without replacement) (§3.1.2). The §3.1.2 prose describes the Flan 2021 pool as "85.7 million queries", which conflicts with Table 3's >28.9M.
- Only T0 tasks from the T0 training split are used, which excludes Big-Bench, because BBH is an evaluation benchmark (§3.1.2).
- System messages #8 and #10 are sampled only for multiple-choice questions (§3.1.1); the distribution per sub-collection is in Fig. 6.
- GPT-4 responses are on average 1.5× longer than ChatGPT responses (§3.1.3, Fig. 9).
- Reasons given for the ChatGPT stage: the capacity gap between a 13B student and GPT-4, and cost and rate limits (§3.1.3). Table 4: ChatGPT $0.002 per 1K tokens and 300 requests/min; GPT-4 (8K) $0.03 prompt / $0.06 completion per 1K tokens and 18 requests/min.
- Evaluation parsing: AGIEval prompt format, the first capital letter of the response is compared with the gold answer id (§4.2.2). Orca is evaluated with the empty system message and temperature 0.7 unless stated (footnote 14).
- AGIEval average (Table 8): Orca 41.7, Text-Davinci-003 41.9, ChatGPT 47.2, GPT-4 62, Vicuna-13B 29.3. With the best of three system messages per task, the gap to ChatGPT is 4.4 points (Table 9).
- BBH average (Table 11): Orca 49.7, ChatGPT 48.9, GPT-4 67.4, Vicuna-13B 23.3.
- GPT-4-judged open-ended prompts: Orca retains 95% of ChatGPT's score and 85% of GPT-4's score, aggregated over Vicuna, Awesome, and WizardLM prompts (Table 7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Orca-13B | 13B | distill-SFT (stage 1) | examples; teacher | FLAN-5M: 5M queries with ChatGPT (GPT-3.5-turbo) responses | arXiv:2306.02707v1 §3.1, §3.1.3 | verified 2026-09-14 | Table 10 (joint with stage 2, see below) |
| Orca-13B | 13B | distill-SFT (stage 2) | examples; teacher | FLAN-1M: 1M queries sampled from FLAN-5M, GPT-4 responses | §3.1.3 | verified 2026-09-14 | Table 10: stage 1 + stage 2 = 41.7 vs FLAN-1M only = 37.18 AGIEval avg |
| Orca-13B | 13B | distill-SFT (both) | epochs | 4 per stage | §3.2 "Compute" | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (both) | sequence length; packing | max_len 2,048 tokens; packing factor 2.7 examples per sequence | §3.2 "Packing" | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (both) | loss masking | loss only on teacher-generated tokens | §3.2 "Loss" | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (both) | tokenizer | LLaMA BPE plus one added padding token; vocabulary 32,001 | §3.2 "Tokenization" | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (both) | compute | 20 A100-80GB; 160 h (stage 1), 40 h (stage 2); data collection 2 weeks (ChatGPT), 3 weeks (GPT-4) | §3.2 "Compute" | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (data) | teacher sampling | GPT-4 temperature 0.7, top_p 0.95 (Azure OpenAI settings printed in footnote 9 to Fig. 4; settings for ChatGPT collection not stated) | footnote 9 | verified 2026-09-14 | no ablation reported |
| Orca-13B | 13B | distill-SFT (both) | optimizer, peak LR, schedule, batch size | not reported | checked §3.2, §4, all footnotes; the paper has no hyperparameter appendix | not reported | — |

## Findings relevant to generality, distillation, and long context
- **Response to the imitation critique (Interpretation by the authors).** §1.1 quotes Gudibande et al. [12] that "model imitation is a false promise" unless one collects enormous imitation datasets with more diversity and quality, and states that Orca shows both conditions are attainable. The evidence offered is zero-shot multiple-choice accuracy on AGIEval and BBH; the paper does not re-run the evaluations of [12].
- **Evaluation error.** GPT-4 judging overestimates small models relative to human-labeled benchmarks (§1, Figs. 1-3). GPT-4 as judge also favors the first response: Vicuna-13B scores 84.87 on WizardLM prompts when shown first, and 77.1 in the standard order (Table 7, footnote 18).
- **Where Orca does not match ChatGPT.** Knowledge-dependent BBH tasks: Sports Understanding 67.2 vs 79.6, Ruin Names 39.5 vs 56.0; Geometric Shapes 20.8 vs 25.6 (Table 11). Math-related AGIEval tasks show larger gaps (§6.1).
- **Stated scope limits (§8).** Training data simulate zero-shot prompts; multi-turn conversation, few-shot/in-context learning, and CoT prompting are untested. Performance "is likely to correlate strongly with the distribution of the tuning data".
- **Contamination (§6.2).** The authors note GPT-4's reported Big-Bench contamination and state they are not aware of such issues for LLaMA pre-training data, FLAN-v2, or ShareGPT.
- **Distillation ablation scope.** Table 10 changes two things at once, the ChatGPT stage and a 5× increase in explanation data (§6.1), so the capacity-gap effect is not separated from data size. No run compares explanation targets with answer-only targets on the same queries; the comparison with Vicuna-13B also differs in data source and size (70K vs 5M, Table 1).
- **Long context.** In 100 sampled AGIEval disagreements, 16% of ChatGPT-beats-Orca examples have long context vs 8% of Orca-beats-ChatGPT examples (§6.1).

## Connections
- [[distilling-step-by-step]] — cited as [17]: LFM rationales help close the gap for task-specific distillation (§1.1).
- [[orca-2]] — follow-up from the same group.
- [[dolphin]] — open dataset that its card describes as a reproduction of Orca's FLAN-augmented data.
- [[self-instruct]], [[alpaca]], [[wizardlm]] / [[evol-instruct]] — prior data methods that §1.1 and Table 1 describe as limited in diversity, complexity, or scale.
- External, no card in this library: Gudibande et al., "The False Promise of Imitating Proprietary LLMs" (arXiv:2305.15717, May 2023). Its abstract reports that imitation models "close little to none of the gap from the base LM to ChatGPT on tasks that are not heavily supported in the imitation data" and mimic "ChatGPT's style but not its factuality".
- [[open-thoughts]] — a later reasoning-trace SFT dataset in this library.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2306.02707 (v1, the only version).
- Corrections to the previous card version:
  - "Smaller models improve more from rich explanation traces … than from short answer-only imitation" → the paper has no answer-only ablation on the same queries; gains are measured against Vicuna-13B, which used different data (Table 1, §6).
  - "Used GPT-4 and ChatGPT together for richer teacher signals" → ChatGPT answers all 5M queries and is trained on first; GPT-4 answers a 1M subset used in a second stage (§3.1.3).
  - "The recipe is explicitly progressive: richer teachers, richer traces, more diverse tasks" → "progressive learning" in the paper means ChatGPT responses first, then GPT-4 responses (§3.1.3); task diversity is not staged.
  - "Training data mixes explanation traces, step-by-step reasoning, and more complex instructions" → FLAN-v2 zero-shot queries with 16 system messages, sampled per Table 3 (§3.1).
  - "Student is a 13B model" → 13B model built on LLaMA (abstract, §6.2, §8).
- Removed as unsupported by the source: "Popularized explanation-trace distillation as an SFT recipe"; "Main lesson is about supervision type, not only dataset size" (the conclusion §9 stresses data size, coverage, and base-model quality, and no ablation isolates supervision type); "major gains" (replaced with table values); "Closely related to … [[s1]]" (no relation stated in the source).
- Not reported by the source: optimizer, learning rate, batch size, warmup; released weights (front matter says a weight diff release was pending legal review).
