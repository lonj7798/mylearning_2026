<!-- scope: phi-1.5 (Microsoft Research, 2023): 1.3B base model pretrained mostly on synthetic textbook-like data for common-sense reasoning, with filtered-web comparison models phi-1.5-web and phi-1.5-web-only
     deps: [[phi-textbooks]]
     see-also: [[hf-cosmopedia]], [[glan]], [[rephrasing-the-web]], [[physics-of-lm-3]], [[phi-3]], [[phi-4]]
-->

# Textbooks Are All You Need II: phi-1.5 technical report
- **Core Insight:** phi-1.5, a 1.3B model trained for 150B tokens on a 30B-token dataset whose only non-synthetic part is 6B tokens of filtered code, scores 0.734 on WinoGrande and 40.2% on GSM8K (via coding), against 0.691 and 14.6% for Llama2-7B; it scores lower than Llama2-7B on HellaSwag (0.476 vs 0.571) and MMLU (0.376 vs 0.453) (§2.2, Table 1, Tables 2–4).
- **Guideline:** When a small model must cover common-sense reasoning and basic math or code, use synthetic textbook-like data seeded from a curated topic list, because phi-1.5 (1.3B) scores above Llama2-7B on WinoGrande, ARC-Challenge, and GSM8K (Tables 2 and 4). Otherwise, when the target includes knowledge-heavy tasks, also train a filtered-web variant, because phi-1.5 trails Llama2-7B on HellaSwag and MMLU (Table 3), and adding filtered web data raised GSM8K from 40.2 to 44.6 and HumanEval from 34.1 to 41.4 (Table 4).
- **Authors:** Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, Yin Tat Lee (Microsoft Research)
- **Year:** 2023 (arXiv v1 2023-09)
- **URL:** https://arxiv.org/abs/2309.05463
- **Source type:** official technical report
- **Relevant topics:** synthetic pretraining data, common-sense reasoning, small models, web vs synthetic data, toxicity of base models

## Abstract
The report continues the TinyStories and phi-1 work on small Transformer language models. phi-1 used LLM-generated "textbook quality" data to train a 1.3B code model. phi-1.5 applies the same approach to common-sense reasoning in natural language. The new 1.3B model performs on natural-language tasks comparably to models 5x larger and surpasses most non-frontier LLMs on grade-school math and basic coding. phi-1.5 shows some traits of larger models, such as step-by-step reasoning and rudimentary in-context learning. It also hallucinates and can produce toxic or biased text, although the authors see improvement on that front, which they attribute to the absence of web data. The model is released as open source for research.

## Key Contributions
- A 1.3B base model trained mainly on synthetic "textbook-like" data (§2.2), with the same architecture as phi-1 (§2.1).
- Two comparison models, phi-1.5-web-only (filtered web, no synthetic data) and phi-1.5-web (web + phi-1 code + synthetic data) (§2.4).
- Benchmarks across common-sense reasoning, language understanding and knowledge, and multi-step reasoning, for all three variants against open baseline models from 1.3B to 65B parameters (Tables 2–4).
- A toxicity evaluation on 86 hand-written prompts and on ToxiGen (§4, Figure 2).
- Release of the raw base model without instruction finetuning or alignment (§1, §2.4).

## Key Figures/Tables to Study
- **Table 1** — training GPU hours, data size, and training tokens for Llama-7B, phi-1.5, and phi-1.5-web.
- **Table 2** — WinoGrande, ARC-Easy, ARC-Challenge, BoolQ, SIQA (zero-shot).
- **Table 3** — PIQA, HellaSwag, OpenbookQA (zero-shot), MMLU (2-shot), SQuAD (exact match).
- **Table 4** — GSM8K, HumanEval, MBPP (zero-shot pass@1).
- **Figure 2** — ToxiGen safety scores over 13 demographics.

## Technical Details
- **Architecture:** identical to phi-1: 24 layers, 32 heads of dimension 64, rotary dimension 32, context length 2048, FlashAttention, codegen-mono tokenizer (§2.1).
- **Data:** phi-1's training data (7B tokens) plus about 20B tokens of new synthetic "textbook-like" data for common-sense reasoning and general world knowledge (science, daily activities, theory of mind) (§2.2). The only non-synthetic part is the 6B tokens of filtered code from phi-1 (§2.2).
- **Seeding:** the authors "carefully selected 20K topics" to seed generation and use samples from web datasets in the generation prompts for diversity (§2.2). The generator model, prompts, and filtering steps for the new data are not described.
- **Filtered web data for the comparison models:** 95B tokens: 88B from Falcon RefinedWeb and 7B of code from The Stack and StackOverflow, filtered with the phi-1 technique (§2.4).
- **Common sense (Table 2):** phi-1.5 WinoGrande 0.734, ARC-Easy 0.756, ARC-Challenge 0.444, BoolQ 0.758, SIQA 0.526; Llama2-7B 0.691, 0.763, 0.434, 0.779, 0.480; phi-1.5-web-only 0.604, 0.666, 0.329, 0.632, 0.414.
- **Language and knowledge (Table 3):** phi-1.5 PIQA 0.766, HellaSwag 0.476, MMLU 0.376, OpenbookQA 0.372, SQuAD 0.72; Llama2-7B 0.781, 0.571, 0.453, 0.314, 0.67.
- **Multi-step reasoning (Table 4):** phi-1.5 GSM8K 40.2 (via coding), HumanEval 34.1, MBPP 37.7; phi-1.5-web 44.6, 41.4, 43.5; phi-1.5-web-only <3, 17.2, 27.3; Llama-65B 50.9, 23.7, 37.7.
- **Inference cost:** phi-1.5 uses under 3ms per token and 3.5G memory at 2048 context on one A100-80G in fp16, against 14ms and 18G for Llama-7B (Table 1).
- **Toxicity prompts:** of 86 prompts, phi-1.5 received 47 "pass", 34 "fail", and 4 "did not understand" labels (as printed, 85 in total); Llama2-7B and Falcon-7B failed 54 and 50, and passed fewer than 20 each (§4).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| phi-1.5 | 1.3B | pretrain-stable | initialization | random initialization | arXiv:2309.05463v1 §2.3 | verified 2026-09-14 | — |
| phi-1.5 | 1.3B | pretrain-stable | mixture (share of training tokens) | 80% new synthetic data, 20% phi-1 training data | §2.3 | verified 2026-09-14 | no ablation of the split reported |
| phi-1.5 | 1.3B | pretrain-stable | dataset size; tokens seen | 30B; 150B | Table 1; §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1.5 | 1.3B | pretrain-stable | epochs over dataset | 5 | derived: 150B / 30B (Table 1) | derived | — |
| phi-1.5 | 1.3B | pretrain-stable | LR; warmup; schedule | 2e-4; no warmup; constant | §2.3 and footnote 1 | verified 2026-09-14 | footnote 1: configuration "intentionally kept straightforward"; no ablation |
| phi-1.5 | 1.3B | pretrain-stable | optimizer; betas; epsilon; weight decay | Adam; momentum 0.9, 0.98; 1e-7; 0.1 | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1.5 | 1.3B | pretrain-stable | batch size (unit not stated) | 2048 | §2.3 | verified 2026-09-14 | no ablation reported |
| phi-1.5 | 1.3B | pretrain-stable | precision; parallelism; context | fp16; DeepSpeed ZeRO Stage 2; 2048 | §2.3; §2.1 | verified 2026-09-14 | — |
| phi-1.5 | 1.3B | pretrain-stable | compute | 1.5K GPU hours (A100-80G) | Table 1 | verified 2026-09-14 | — |
| phi-1.5-web | 1.3B | pretrain-stable | mixture | filtered web subset, phi-1 code data, synthetic NLP data "roughly 40%, 20%, 40%" | §2.4 | verified 2026-09-14 | Tables 2–4 compare with phi-1.5 |
| phi-1.5-web | 1.3B | pretrain-stable | dataset size; tokens seen; compute | 100B; 300B; 3K GPU hours | Table 1 | verified 2026-09-14 | — |
| phi-1.5-web-only | 1.3B | pretrain-stable | mixture | filtered web only, about 80% NLP and 20% code, no synthetic data | §2.4 | verified 2026-09-14 | Tables 2–4 |
| phi-1.5-web-only | 1.3B | pretrain-stable | tokens seen; compute | not reported | checked §2, Table 1 | not reported | — |
| phi-1.5-web, phi-1.5-web-only | 1.3B | pretrain-stable | LR, optimizer, batch | not reported separately (§2.3 describes phi-1.5) | checked §2.3–2.4 | not reported | — |
| phi-1.5, phi-1.5-web, phi-1.5-web-only | 1.3B | SFT, RL | instruction finetuning or RLHF | none performed | §2.4 Remark | verified 2026-09-14 | — |

## Findings relevant to generality
- **Web vs synthetic:** phi-1.5-web-only is trained on filtered web data equal to 15% of Falcon RefinedWeb. The authors state that it "already outperforms all existing models of similar size" (§3); on WinoGrande it scores 0.604 against 0.607 for Falcon-rw-1.3B (Table 2). Adding synthetic data (phi-1.5-web) gives results similar to models 5x larger (§3).
- **Where synthetic-only is weaker:** on multi-step reasoning, phi-1.5-web outperforms phi-1.5 "somewhat significantly" (§3, Table 4). On HellaSwag, which the authors say "arguably relies more on 'memorized knowledge'" (Figure 1 caption), phi-1.5 scores 0.476 vs 0.571 for Llama2-7B (Table 3).
- **Mixed-task retention:** phi-1.5 trains on natural language and code, and its coding ability is "quite close" to phi-1, which was trained only for code (§3). The authors interpret this as more efficient knowledge storage with textbook-like data (Interpretation, §3).
- **Instruction following without SFT:** the base model follows simple instructions and chat formats imperfectly, and does not stop properly. The authors tentatively attribute this to "exercises and answers" inside the synthetic textbooks (§5).
- **Measurement notes:** all benchmark numbers come from the authors' own evaluation pipeline and may differ from numbers reported elsewhere (Figure 1 caption). GSM8K is solved "via coding" (Table 4). The report contains no contamination or topic-overlap analysis.

## Connections
- [[phi-textbooks]] — phi-1, whose data (7B tokens) and filtering technique phi-1.5 reuses (§2.2, §2.4).
- [[hf-cosmopedia]] — an open synthetic corpus whose card compares its cosmo-1b model with phi-1.5.
- [[glan]] — later instruction-data work that expands a hierarchical taxonomy instead of a flat 20K-topic list.
- [[physics-of-lm-3]] — cites this report for the claim that much internet data lacks valuable knowledge for training.
- [[phi-3]], [[phi-4]] — later Phi model reports.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2309.05463 (arXiv v1, 11 Sep 2023)
- Corrections to the previous card version:
  - "~27B training tokens" → a 30B-token dataset trained for 150B tokens, 80% synthetic and 20% phi-1 data (Table 1, §2.3).
  - "GPT-3.5-authored" synthetic data, "Teacher model(s): GPT-3.5" → the report does not name the model that generated the new 20B tokens.
  - "matches or beats models up to 10× its size on ... HellaSwag ... MMLU-subset" → below Llama2-7B on HellaSwag (0.476 vs 0.571) and MMLU (0.376 vs 0.453, 2-shot MMLU; the report does not describe a subset) (Table 3). The abstract says "comparable to models 5x larger"; §1 says "ten times its size" for common sense.
  - "synthetic-only is almost as strong as synthetic+web, and better on reasoning" → phi-1.5-web is better on multi-step reasoning (Table 4).
  - "Table 1 — ... WinoGrande, ARC ..." → Table 1 is compute; benchmarks are in Tables 2–4.
  - "removing synthetic common-sense portion drops reasoning benchmarks sharply" → the comparison is with phi-1.5-web-only, a separate filtered-web mixture (§2.4, Tables 2–4).
  - "Matches or beats Llama-2-7B on WinoGrande, ARC, PIQA, HellaSwag" → higher on WinoGrande and ARC-Challenge, lower on ARC-Easy, PIQA, HellaSwag (Tables 2–3).
- Removed as unsupported by the source: "hand+LLM-curated" topic list covering "grade-school science, basic logic"; prompts specifying audience, style, length; "multiple passes per topic with varied sub-angles"; "templated expansions"; "deduplication; benchmark decontamination"; cost "~hundreds of thousands USD"; "first public demonstration"; "catalyzed independent reproductions"; "topic taxonomy leakage" critique; "reasoning parity weaker on MMLU-Pro, MATH"; "contamination concerns carry over from phi-1"; the model-collapse counter-evidence claim.
- Not reported by the source: generator model, generation prompts, filtering of synthetic data, data release, contamination analysis, sampling temperatures.
