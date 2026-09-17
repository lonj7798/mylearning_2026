---
chapter: ch-30a
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/finetuning-new-knowledge-hallucination.md (planned card; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2405.05904
created_at: "2026-09-15"
---

# Excerpt: Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?

**Authors:** Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, Jonathan Herzig (Technion; Google Research)
**Version read:** arXiv:2405.05904v3 (1 Oct 2024); v1 May 2024.
**Status:** no library card existed for this slug on 2026-09-15; quotes checked against the v3 PDF text.

## Setup (§2-3, App. E)
- Model: PaLM 2-S base; closed-book QA from EntityQuestions (12 relations in-distribution, 7 reserved for an OOD test set); exact match.
- SliCK categories from P_Correct, estimated with 10 random 4-shot prompts, greedy (T = 0) and 16 samples at T = 0.5 per prompt (§3). Unknown: P_Correct(q, a; M, T ≥ 0) = 0. HighlyKnown: greedy always correct. MaybeKnown: greedy sometimes correct. WeaklyKnown: greedy never correct, sampling sometimes correct (Fig. 2).
- |D| = 6142 (App. E). "We fine-tune every model for 50 epochs ... We use learning rate of 1e-5, a batch size of 128, and a dropout rate of 0.05." EARLY_STOP "happens after 5-10 epochs" (§4).

## Linear model (§4.4, Eq. 1, Table 1)
Accuracy = β_0 + β_kn · N_kn/|D| + β_unk · N_unk/|D|, where N_kn and N_unk are the Known and Unknown training examples the model fits.
| Test set | β_0 | β_kn | β_unk | R² |
|---|---|---|---|---|
| In-distribution | 36.9 | 7.3 | −8.3 | 0.86 |
| Out-of-distribution | 36.2 | 3.2 | −3.0 | 0.95 |
Footnote 9: "this linear model is only valid in bounded region of N_kn ≤ |D|, N_unk ≤ |D|."

## Other results
- "M fits Unknown fine-tuning examples substantially slower than Known" (§4.3).
- "for EARLY_STOP the results for D are almost identical to D_Known ... Conversely, the CONVERGENCE results show that with longer training, Unknown examples are actually very harmful" (§4.2).
- OOD: "fine-tuning on Unknown examples such as 'Where is [E1] located?', can encourage hallucinations on seemingly unrelated questions, such as 'Who founded [E2]?'" (§4.5).
- Table 2: at CONVERGENCE, D_Unknown scores 25.8 on the full test set vs 37.5 at EARLY_STOP; D_MaybeKnown scores 43.6 (EARLY_STOP) and 43.2 (CONVERGENCE).
- Fig. 6: the same trends with LR 1e-4.

## How ch-30a uses it
§3.3 (hallucination as a form of narrowing), Negative-feedback section (Unknown examples as negative marginal value), Recipe row.
