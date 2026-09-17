---
chapter: ch-21
course: llm-training
phase: read
excerpt_of: "Phi-4 Technical Report (arXiv:2412.08905v1) and Phi-4-reasoning Technical Report (arXiv:2504.21318v1)"
source_url: https://arxiv.org/abs/2412.08905
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Phi-4 — synthetic pretraining data, mixture ablations, and fresh-test evaluation

This excerpt was rewritten on 2026-09-15 from the primary PDFs, because the library card `phi-4` had not been
verified and mixes the Phi-4 and Phi-4-reasoning reports. The April 2026 version stated a "~10% weighted" synthetic
share, an unspecified pretraining size, and "RL is polish"; these are corrected below.

- **Authors:** Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, et al.
  (Microsoft Research). arXiv v1 2024-12-12.

## Synthetic data (§2.2)
- "We created 50 broad types of synthetic datasets, each one relying on a different set of seeds and different
  multi-stage prompting procedure ... accumulating to a total of about 400B unweighted tokens."
- Seeds: excerpts from web pages, books, and code repositories, filtered in two stages (page-level educational
  potential, then passage-level factual and reasoning content); questions from websites, forums, and Q&A platforms,
  kept only when multiple independent answers neither all agree nor are entirely inconsistent; question-answer pairs
  extracted from deduction chains in books, papers, and code.
- Workflows: rewrite and augment; self-revision with rubrics; instruction reversal for code (kept only when
  regenerated code matches the original); execution loops and tests for code.
- §2.3: "organic questions are substantially more effective than synthetic questions" (ablation; no numbers printed).
  Web-dump filtering "tends to over-index on STEM-related keywords", so a separate pipeline amplifies non-STEM content.

## Pretraining (§3)
- 14B parameters; 4,096 context, extended to 16K in midtraining; "pretrained for approximately 10T tokens using linear
  warm-up and decay schedules with peak learning rate of 0.0003, constant weight decay of 0.1, and global batch size of 5760".
- §3.1 on phi-3-scale runs: more epochs over synthetic data beat fresh web tokens on reasoning benchmarks (Fig. 2,
  4 vs 12 epochs); "Models trained only with synthetic data underperformed on the knowledge-heavy benchmarks and
  demonstrated increased hallucinations."
- Table 3 (13B ablation models, no web data; differences vs phi-3-medium):

| Mixture | MMLU | MMLU-pro | GSM8k | HumanEval | ARC-C | MBPP | MATH | TQA |
|---|---|---|---|---|---|---|---|---|
| Synthetic | +0.8 | +4.0 | +2.2 | +12.1 | 0.0 | +5.0 | +4.9 | −14.8 |
| Synthetic + Web Rewrites | +0.3 | +4.1 | +1.8 | +13.3 | +3.0 | +7.6 | +8.1 | −7.7 |

- Table 4 (7B scale, 1T-token horizon; 75% of tokens reallocated among synthetic S, filtered web W, web rewrites WR;
  differences vs the final mixture):

| Allocation | MMLU | MATH | GSM8k | HumanEval | ARCC | MBPP | TQA | MMLU pro | Average |
|---|---|---|---|---|---|---|---|---|---|
| Uniform | −3.3 | −5.4 | −5.8 | −1.2 | +0.6 | −2.0 | +3.3 | −3.6 | −2.2 |
| S | +3.3 | +4.0 | +2.1 | −6.1 | +1.9 | +0.4 | −3.0 | +3.7 | +0.8 |
| S + WR | +0.6 | +1.2 | +1.5 | −1.2 | +1.6 | +1.6 | −3.7 | +1.2 | +0.4 |
| S + W | −0.6 | −0.7 | −0.7 | −4.3 | +0.3 | −2.0 | +6.9 | +0.9 | 0.0 |

  The authors chose a mixture with targeted and knowledge-heavy web data "to improve knowledge benchmarks", and
  report that the gap to synthetic-heavy runs "largely closes" after post-training (§3.2, no numbers).
- Table 5 (final pretraining mixture):

| Source | Fraction of training tokens | Unique token count | Number of epochs |
|---|---|---|---|
| Web | 15% | 1.3T | 1.2 |
| Web rewrites | 15% | 290B | 5.2 |
| Synthetic | 40% | 290B | 13.8 |
| Code data | 20% | 820B | 2.4 |
| Acquired sources | 10% | 580B | 1.7 |

  "Web rewrites is a sub-category of synthetic data" (footnote 5). The report does not reconcile the §2.2 figure of
  about 400B unweighted synthetic tokens with the Table 5 unique counts.
- Midtraining (§3.3): 250B tokens; 30% new long-context data and 70% recall tokens from pretraining; RoPE base 250K;
  maximum LR divided by 10.

## Post-training (§4)
- One round of SFT (LR 1e-6, about 8B tokens, 40 languages of multilingual data), one DPO round on Pivotal Token
  Search pairs, one judge-guided DPO round (about 850k pairs, GPT-4o judge).
- PTS targets questions with 0.2 ≤ p(success) ≤ 0.8; pairs are single tokens that raise or lower p(success) (§4.3).
- Refusal data (App. A.1): (question, correct answer) where the base model is usually correct, (question, refusal)
  where it is usually wrong, (bogus question, refusal); DPO pairs (correct > refusal) and (refusal > wrong) use the
  first 5 tokens of the response.
- SimpleQA over post-training (Fig. 6): correct / not attempted / incorrect = base 6.8 / 3.2 / 90.0%; SFT 3.7 / 57.5 /
  38.7%; DPO stage 1 2.9 / 79.8 / 17.4%; final 3.0 / 81.1 / 15.8%.
- Table 9 (SFT → DPO stage 1 → final): GPQA 47.3 → 53.6 → 56.1; MATH 77.1 → 80.5 → 80.4; ArenaHard 56.7 → 66.5 → 75.4;
  IFEval 66.2 → 63.0 → 63.0; DPO stage 2 only: GPQA 52.4, MATH 77.6, ArenaHard 69.8.

## Evaluation for overfitting (§1.1, §5, App. B, App. C)
- November 2024 AMC-10/12: 78 questions released on or after November 6, 2024, after all training data were
  collected; temperature 0.5; phi-4 averages 91.8 of 150 (Fig. 1 bar label), Qwen 2.5 14B-Instruct 77.4, GPT-4o 77.9,
  Gemini Pro 1.5 89.8. App. C states 10 generations per question; the Fig. 1 caption states 100 runs. Footnote 8: all
  three final candidates scored above 89, and the final model was chosen after seeing the other two scores.
- Decontamination: 13-gram and 7-gram hybrid algorithm against 20 benchmarks (App. B.1); §5 notes it is "not
  effective against all scenarios, including rephrasing".
- PhiBench: internal benchmark of original team-written questions, used to guide mixture and hyperparameter decisions (§5).
- Weaknesses (§6, §8): IFEval "reveals a real weakness"; factual hallucination, for example invented biographies.
- Table 1 (simple-evals): phi-4 SimpleQA 3.0, IFEval 63.0, MATH 80.4, GPQA 56.1.

## Phi-4-reasoning-plus RL (arXiv:2504.21318v1 §4.2)
"We select as our RL checkpoint the model with the best observed AIME 2024 score, which is the model trained for 90
steps, over only ∼ 6k examples (and 8 trajectories of responses per example)." "additional GRPO training for only 90
steps boosts AIME performance by more than 10% (Figure 7a)." Responses beyond 31k tokens are clipped during GRPO.

## Verification
- Read on 2026-09-15 in the cached text of arXiv:2412.08905v1 (§1.1, §2.2–2.4, §3–§6, §8, App. A.1, B.1, C) and
  arXiv:2504.21318v1 §4.2; Figures 1 and 6 read from the rendered PDF pages 3 and 16.
