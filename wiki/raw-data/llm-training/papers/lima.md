<!-- scope: 1,000-example SFT on LLaMa-65B, no RL — the "Superficial Alignment Hypothesis"
     deps: [[self-instruct]]
     see-also: [[alpaca]], [[instag]], [[ultrachat-pipeline]], [[limo]]
-->

# LIMA: Less Is More for Alignment
- **Core Insight:** Fine-tuning LLaMa-65B on 1,000 curated prompt-response pairs (~750,000 tokens) with the standard supervised loss and no RL produced a model whose responses human annotators rated equal or better than GPT-4 in 43% of head-to-head comparisons, 58% against Bard, and 65% against DaVinci003 (Abstract; §4.2, Figure 1).
- **Guideline:** When a strong pretrained base is available and the target is open-ended assistant response style, invest annotation effort in prompt diversity and response quality rather than dataset size, because in the paper's 7B ablations a 16-fold increase in Stack Exchange examples (2K→32K) did not raise the ChatGPT-graded quality score (§5, Figure 6). This was measured only by human and model preference judgments, not on objective benchmarks.
- **Authors:** Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao et al. (15 authors; Mike Lewis, Luke Zettlemoyer, Omer Levy last)
- **Year:** 2023 (arXiv v1 2023-05; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2305.11206
- **Source type:** paper
- **Relevant topics:** SFT, data quality, data diversity, minimal-data alignment, superficial alignment hypothesis, preference evaluation

## Abstract
The paper measures the relative importance of pretraining versus large-scale instruction tuning and RL. LIMA is a 65B-parameter LLaMa model fine-tuned with the standard supervised loss on 1,000 curated prompts and responses, with no reinforcement learning and no human preference modeling. The authors report that LIMA learns specific response formats from a small number of training examples and tends to generalize to unseen tasks. In a controlled human study, LIMA responses are equivalent or preferred to GPT-4 in 43% of cases, 58% versus Bard, and 65% versus DaVinci003, which was trained with human feedback. The authors conclude that almost all knowledge in large language models is acquired during pretraining and that limited instruction-tuning data suffices to make the model produce high-quality output.

## Key Contributions
- States the **Superficial Alignment Hypothesis**: alignment teaches the model the style or format of interaction with users, exposing knowledge and capabilities already acquired during pretraining (§1).
- Demonstrates the hypothesis with a 1,000-example SFT set, no RL, no reward model, on LLaMa-65B (§3).
- Publishes the exact source composition of the 1,000 examples, including input and output lengths (Table 1).
- Runs controlled 7B ablations that separate prompt diversity, response quality, and data quantity (§5, Figures 5 and 6).
- Shows that 30 added multi-turn dialogue chains raise the excellent-response rate from 45.2% to 76.1% over 10 test chats (§6, Figure 7).

## Key Figures/Tables to Study
- **Table 1** — source composition, example counts, and average input/output lengths for train, dev, and test.
- **Figure 1 / Figure 2** — human preference and GPT-4 preference results against five baselines.
- **Figure 3** — absolute assessment of 50 test prompts: 50% excellent, 38% pass, 12% fail.
- **Figure 5** — 7B ablation: wikiHow 3.49, unfiltered Stack Exchange 3.33, filtered Stack Exchange 3.83 on a 1-6 ChatGPT Likert scale.
- **Figure 6** — 7B quantity ablation over 2K, 4K, 8K, 16K, 32K filtered Stack Exchange examples; the curve is flat.
- **Figure 9 (App. B)** — held-out perplexity rises while generation quality rises, motivating manual checkpoint selection.

## Technical Details
**Training-set composition (1,000 examples, ~750,000 tokens total; Table 1):**
- Stack Exchange (STEM) 200; Stack Exchange (Other) 200; wikiHow 200; Pushshift r/WritingPrompts 150; Natural Instructions 50; paper authors (Group A) 200.
- 750 of the 1,000 come from community forums; 250 are authored by the paper's authors, of which 200 are used for training and 50 form the development set (§2.1, §2.2).
- The 50 Natural Instructions examples are one random example from each of 50 natural language generation tasks sampled from Super-NaturalInstructions, lightly edited to match the authors' style (§2.2).
- 13 training prompts carry some degree of toxicity or malevolence, with responses that partially or fully refuse and explain the refusal (§2.2).
- Stack Exchange has 179 exchanges; wikiHow has over 240,000 how-to articles (§2.1).
- Test set: 70 Pushshift r/AskReddit prompts plus 230 Group B author prompts (Table 1).

**Evaluation protocol (§4.1):**
- One response generated per test prompt; crowd workers label which of two responses they prefer; the experiment is repeated with GPT-4 as the annotator, with similar agreement levels.
- Baselines: Alpaca 65B (LLaMa-65B fine-tuned on Alpaca's 52,000 examples), DaVinci003, Bard, Claude, GPT-4 (§4.1).
- Ablation grading uses ChatGPT (GPT-3.5 Turbo) on a 6-point Likert scale, 5 samples per test prompt, reported with a p = 0.95 two-sided confidence interval (§5, App. D).

**Results (§4.2, §4.3):**
- GPT-4 prefers LIMA's output over its own in 19% of comparisons (§4.2).
- Bard produces the better response 42% of the time, so LIMA is at least as good 58% of the time (§4.2).
- Of 50 manually analyzed responses, 50% are excellent, 38% pass, 12% fail (Figure 3).
- On 20 out-of-distribution examples, 20% fail, 35% pass, 45% are excellent (§4.3).
- LIMA responds safely to 80% of 30 potentially sensitive test prompts, including 6 of 10 with malicious intent (§4.3).

**Ablations (§5, 7B LLaMa, same hyperparameters, at least 2,000 examples used for training stability):**
- *Diversity*: filtered Stack Exchange (heterogeneous prompts, high-quality responses) scores 3.83 versus wikiHow (homogeneous "how to" prompts, high-quality responses) at 3.49 (Figure 5).
- *Quality*: filtered Stack Exchange 3.83 versus unfiltered Stack Exchange 3.33, a 0.5-point difference (§5, Figure 5).
- *Quantity*: doubling the filtered Stack Exchange training set from 2K to 32K does not improve response quality (§5, Figure 6).

**Limitations stated by the authors (§7):** constructing such examples takes significant mental effort and is difficult to scale; LIMA is less robust than product-grade models, and an unlucky decoding sample or an adversarial prompt can produce a weak response.

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| LIMA | 65B | SFT | base model | LLaMa 65B | arXiv:2305.11206 §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | training examples; total tokens | 1,000 examples; roughly 750,000 tokens | §2, Table 1 | verified 2026-09-18 | §5 Fig. 6: 2K→32K on 7B does not improve ChatGPT-graded quality |
| LIMA | 65B | SFT | epochs | 15 | §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | optimizer | AdamW, β1 = 0.9, β2 = 0.95, weight decay 0.1 | §3 | verified 2026-09-18 | "standard fine-tuning hyperparameters"; no ablation reported |
| LIMA | 65B | SFT | warmup | none | §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | peak LR, final LR, decay shape | 1e-5 initial, linearly decaying to 1e-6 by end of training | §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | global batch (sequences) | 32 examples (64 for smaller models) | §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | sequence length | 2048; longer texts trimmed | §3 | verified 2026-09-18 | no ablation reported |
| LIMA | 65B | SFT | residual dropout | p_d = 0.0 at bottom layer rising linearly to 0.3 at last layer (0.2 for smaller models) | §3 | verified 2026-09-18 | follows Ouyang et al. 2022; no ablation reported |
| LIMA | 65B | SFT | turn separator | special end-of-turn (EOT) token, distinct from EOS | §3 | verified 2026-09-18 | avoids conflation with pretrained EOS meaning (§3) |
| LIMA | 65B | SFT | checkpoint selection rule | manual selection between the 5th and 10th epoch on a 50-example held-out dev set | §3 | verified 2026-09-18 | App. B Fig. 9: perplexity anticorrelates with generation quality |
| LIMA + dialogue | 65B | SFT | added multi-turn data | 30 dialogue chains (10 author-composed, 20 from Stack Exchange comment chains); 1,030 examples total | §6 | verified 2026-09-18 | §6 Fig. 7: excellent-response rate 45.2% → 76.1%, fails 15/42 turns → 1/46 |
| LIMA ablations | 7B | SFT | ablation training-set size | 2,000 examples per source (2K-32K for the quantity sweep) | §5 | verified 2026-09-18 | §5 footnote 5: at least 2,000 examples improved stability at 7B |
| LIMA | 65B | SFT | loss masking; packing; compute | not reported | body, App. A-E checked | not reported | — |

## Findings relevant to generality
- The paper reports that LIMA achieves similar absolute quality statistics on 20 out-of-distribution examples as in-distribution (20% fail, 35% pass, 45% excellent), and reads this as evidence of generalization (§4.3). The sample is described by the authors as small.
- All reported evaluation is preference-based: human crowd-worker preference, GPT-4 preference, and ChatGPT Likert grading (§4.1, §5). No objective capability benchmark (for example MMLU, GSM8K, or HumanEval) is reported.
- App. E reports that removing a task from the training set changes behavior, and that 6 examples can determine whether the model generates text with complex structure (§6 footnote 6, App. E).

## Connections
- [[alpaca]] — the 52,000-example set LIMA uses as a directly comparable 65B baseline.
- [[self-instruct]] — the quantity-first generation approach LIMA positions itself against.
- [[instag]] — later work that gives prompt diversity a measurable definition instead of LIMA's source proxy.
- [[limo]] — applies the same less-is-more argument to reasoning data.
- [[ultrachat-pipeline]], [[ultrafeedback]] — large synthetic sets that sit on the other side of the quantity question.
- [[physics-of-lm-3]] — tests where knowledge actually enters the model, bearing on the Superficial Alignment Hypothesis.
- [[rejection-sampling-finetuning]] — automates the response-quality filtering LIMA did by hand.
- [[tulu-3]] — a production SFT mix whose scale contrasts with LIMA's 1,000 examples.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2305.11206 (PDF, "Preprint. Under review." version).
- Corrections to the previous card version:
  - "Diversity > raw count: doubling examples within a single domain hurt performance" → the paper reports that doubling the training set does not improve response quality; it does not report a decrease (§5, Figure 6).
  - "Response quality > prompt quality: poor responses cap the model regardless of prompt richness" → the quality ablation is filtered versus unfiltered Stack Exchange, a 0.5-point difference on a 1-6 scale (§5, Figure 5). The diversity ablation separately shows diverse Stack Exchange prompts beating homogeneous wikiHow prompts (3.83 versus 3.49).
  - "Scaling data from 2K→32K StackExchange alone did not improve generation quality" → correct, but the setting was a 7B LLaMa graded by ChatGPT, not the 65B model (§5).
  - Composition "StackExchange, wikiHow, Reddit, hand-written" → the full breakdown is Stack Exchange STEM 200, Stack Exchange Other 200, wikiHow 200, r/WritingPrompts 150, Natural Instructions 50, author-written 200 (Table 1). The 50 Super-NaturalInstructions examples were omitted.
  - "learning rate 1e-5, batch size 32, AdamW" → complete setting is 1e-5 linearly decaying to 1e-6, no warmup, β1 = 0.9, β2 = 0.95, weight decay 0.1, batch 32 examples, max length 2048, residual dropout 0.0→0.3 (§3).
  - "A key trick: lowering LR as epochs progress was essential to avoid overfitting" → the paper states a linear decay from 1e-5 to 1e-6 but makes no claim that it was essential or that it prevented overfitting. What the paper does report is manual checkpoint selection between epochs 5 and 10, because perplexity anticorrelates with generation quality (§3, App. B).
  - "Standard supervised cross-entropy on response tokens only (prompt tokens masked)" → the paper says "standard supervised loss" and does not state a masking scheme; recorded as not reported.
  - "Multi-turn dialogue was addressed post-hoc with 30 extra dialogue examples; still weaker than GPT-4" → the 30 chains are real (§6), but the paper does not compare the dialogue-tuned model to GPT-4; it reports 45.2% → 76.1% excellent responses over 10 chats.
  - "1,000 SFT examples on LLaMA-65B rival DaVinci-003 (RLHF) and Bard" → LIMA is preferred or tied against DaVinci003 in 65% and against Bard in 58% of comparisons (Abstract; §4.2).
  - Author list truncated to first six plus "et al." per §9.
- Removed as unsupported by the source: "the curve is famously flat" (framing); "poor responses cap the model regardless of prompt richness"; "an adversarial prompt can knock LIMA off-script" (replaced with the authors' own wording from §7); the claim that LIMA "inspired the ablation design of [[ultrachat-pipeline]]/[[ultrafeedback]]".
- Not reported by the source: loss masking scheme, sequence packing, GPU count or training hours, random seeds or number of runs per ablation point, objective benchmark scores.
