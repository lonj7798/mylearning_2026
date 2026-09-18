<!-- scope: Instruction backtranslation (Humpback, Li et al., ICLR 2024): generate instructions for human-written web text with a backward model, self-curate the pairs, iterate
     deps: [[self-instruct]]
     see-also: [[lima]], [[openassistant]], [[alpagasus]], [[genie]], [[r2e-gym]]
-->

# Self-Alignment with Instruction Backtranslation
- **Core Insight:** LLaMA 65B finetuned on 3,200 human-written seed examples plus about 42k self-generated, self-curated (instruction, web text) pairs reaches an AlpacaEval win rate of 83.71% over text-davinci-003, the highest of the non-distilled 33B and 65B models in the paper's table (Table 3), while the same augmented data without curation does not raise win rate as its size grows (§3.3, Figure 2).
- **Guideline:** When building instruction data from unlabelled human-written text without a stronger external model, train a backward (output → instruction) model on a small seed set, score the generated pairs with the seed-trained model, keep only the highest-scored pairs, and train on them together with the seed data under a separate system-prompt tag, because uncurated pairs did not help and joint seed + curated training outperformed either set alone in the 7B ablations (§3.5, Figure 5, Table 5).
- **Authors:** Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, et al. (Meta)
- **Year:** 2023 (arXiv v1 2023-08; ICLR 2024)
- **URL:** https://arxiv.org/abs/2308.06259
- **Source type:** paper
- **Relevant topics:** instruction backtranslation, self-augmentation, self-curation, self-alignment, web-text-grounded synthetic SFT data, data quality vs quantity

## Abstract
The paper builds an instruction-following model by labelling human-written text with model-generated instructions. The method, instruction backtranslation, starts from a language model finetuned on a small seed set and a web corpus. The seed model generates instruction prompts for web documents (self-augmentation) and then selects high-quality examples among the candidates (self-curation). The selected data is used to finetune a stronger model. Two iterations on LLaMA give a model that outperforms all other LLaMA-based models on the Alpaca leaderboard that do not rely on distillation data.

## Key Contributions
- The two-step procedure: self-augmentation with a backward model p(x|y), and self-curation by prompting the model to rate each pair on a 5-point scale, repeated for two iterations (§2, Figure 1, Table 19).
- Tagging of seed and augmented data with different system prompts during finetuning (§2.3, Table 5).
- A data-scaling comparison across instruction datasets with a fitted coefficient α (§3.3, Table 2).
- AlpacaEval, human preference, commonsense, and MMLU evaluations at 33B, 65B, and LLaMA 2 70B (§3.4).
- Ablations on augmented-only, seed-only, and joint training, and on curation precision/recall (§3.5, App. B Table 9).

## Key Figures/Tables to Study
- Figure 1 (pipeline); Table 1 (data sizes and lengths); Table 19 (curation prompt).
- Figure 2 (no curation vs A4 vs A5 at 3,200–25,600 examples); Figure 5 (seed-only vs augmented-only vs joint).
- Table 2 (α per dataset); Table 3 (AlpacaEval win rates); Figure 4 (human preference).
- Table 4 and Table 7 (commonsense and MMLU); Table 9 (curation precision/recall); Table 18 (scaling-run steps).

## Technical Details
**Data.**
- Seed: 3,200 Open Assistant (instruction, output) pairs from the first turn of each conversation tree, English, human rank 0 (§3.1).
- Unlabelled: 502k segments sampled from the English portion of ClueWeb (§3.1); Table 1 lists 502,133. A segment is the text under an HTML header, including lower-level header subtrees (App. E).
- Segment filters: total length 600–3,000 characters; removal of segments with repetitive sentences by n-gram Jaccard similarity between sentence pairs; removal of empty or all-uppercase headers and navigation headers such as "advertisement", "forum", "quick link", "free newsletter" (App. E).

**Method.**
1. Finetune the base model on (output, instruction) seed pairs to get the backward model M_yx := p(x|y) (§2.2).
2. For each segment y_i, generate a candidate instruction x̂_i, giving candidates A = {(x̂_i, y_i)} (§2.2).
3. Finetune the base model on seed (instruction, output) pairs to get M0. M0 rates each candidate on a 5-point scale with the Table 19 prompt, which asks for brief reasoning and then "Score: <rating>"; pairs with score a_i ≥ k form A_k^(1) (§2.3, Table 19).
4. On iteration t, finetune M_t on seed data plus A_k^(t−1), rescore the candidates with M_t, and get A_k^(t). Two iterations give the final model M2 (§2.3).
5. Tag seed data with S_a = "Answer in the style of an AI Assistant." and augmented data with S_w = "Answer with knowledge from web search." (§2.3).
- Table 19 scale: 3 = helpful but written from another person's perspective (blog or web page style); 4 = AI-assistant perspective with minor room for improvement; 5 = a perfect AI-assistant answer (Table 19).
- Curated sets after iteration 2: A_5^(2) 41,821 examples (instruction 115 ± 175 chars, output 1,663 ± 616); A_4^(2) 195,043 (206 ± 298; 1,985 ± 649); seed 3,200 (148 ± 322; 1,072 ± 818); all candidates 502,133 (352 ± 134; 1,722 ± 653) (Table 1). §3.3 defines A4 as score ≥ 4 and A5 as score ≥ 4.5.

**Evaluation.** Human and dev evaluation use 1,130 unique prompts from Vicuna (80), Self-instruct (252), Open Assistant (188), Koala (156), HH_RLHF (129), LIMA (300), and author-crowdsourced prompts (64); 256 prompts outside the AlpacaEval test set form the dev set (§3.1). AlpacaEval uses 805 prompts and GPT-4 judging against text-davinci-003 (§3.4). Human raters are 29 MTurk workers selected from 1,000 applicants by a screening test with > 85% agreement (App. D.1).

**Results.**
- AlpacaEval win rate (annotated / total examples): Humpback 33B 79.84 (3k / 45k) vs OASST RLHF 33B 66.52, Guanaco 33B 65.96, OASST SFT 33B 54.97; Humpback 65B 83.71 vs Guanaco 65B 71.80, LIMA 65B 62.70; Humpback 70B (LLaMA 2) 87.94 vs LLaMa2 Chat 70B 92.66 (1.4m / 5.7m) in the same non-distilled group (Table 3, §3.4).
- Human preference, Humpback win / tie / lose %: vs LIMA 55.6 / 6.7 / 37.8; vs Claude 59.4 / 7.5 / 33.1; vs Guanaco 59.6 / 11.0 / 29.4; vs davinci-003 66.7 / 13.5 / 19.8; vs Falcon-Instruct 81.4 / 3.8 / 14.9 (Figure 4).
- Scaling coefficient α from w = α log N + C (w = win rate vs text-davinci-003 of LLaMA 7B finetuned on N examples): Humpback (k = 5, 2 iterations) 6.95, WizardLLM 5.69, Alpaca-GPT4 5.40, Vicuna 4.53, Open Assistant 4.43, LIMA 2.86, Alpaca 1.99, FLAN v2 0.22 (§3.3, Table 2).
- System prompts (7B, mean win rate ± s.e.): train S_a/S_w, infer {S_a, S_w} 66.47 ± 3.04; infer S_a 62.69 ± 3.06; infer none 62.70 ± 3.07; no tags in training or inference 59.96 ± 3.09 (Table 5).
- Curation quality on a 250-example dev set with 20% positives labelled by an author: precision / recall M0 0.44 / 0.09, M1 0.52 / 0.44, GPT-4 0.88 / 0.92; win rate of LLaMA 7B trained on 100 selected examples 35.71, 37.70, 41.04 (App. B, Table 9).
- Adding augmented data fixes seed-model failures on 41 of 251 test prompts (App. B, Table 8). A5 scaling improves both 7B and 65B, and neither is saturated at 40,000 instructions (App. B, Figure 7).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Humpback (LLaMA) | 7B, 33B, 65B | SFT | loss masking | loss on output tokens only | arXiv:2308.06259v3 §3.1 | verified 2026-09-14 | no ablation reported |
| Humpback, "most models" (which models is not stated) | 7B, 33B, 65B | SFT | LR schedule; weight decay; dropout | 1e-5 linearly decaying to 9e-6 at end of training; 0.1; 0.1 | arXiv:2308.06259v3 §3.1 | verified 2026-09-14 | copied from Zhou et al. 2023 and Touvron et al. 2023a; no ablation reported |
| Humpback (LLaMA) | 7B, 33B, 65B | SFT | batch size | 32 examples; 8 when finetuning on fewer than 3,000 examples | arXiv:2308.06259v3 §3.1; Table 18 | verified 2026-09-14 | no ablation reported |
| Humpback 33B / 65B / 70B | 33B, 65B, 70B | SFT | training examples | 3k annotated, 45k total | arXiv:2308.06259v3 Table 3 | verified 2026-09-14 | Figure 2, Figure 7 (A5 scaling) |
| Humpback M2 | all | SFT (data curation) | curation threshold; iterations | k = 5; 2 iterations (§3.3); A5 defined as score ≥ 4.5 (§3.3) | arXiv:2308.06259v3 §2.3; §3.3 | verified 2026-09-14 | Figure 2: A5 and A4 vs no curation, 7B |
| Backward model M_yx, seed model M0 | all | SFT | seed data | 3,200 Open Assistant first-turn, English, rank-0 pairs | arXiv:2308.06259v3 §3.1 | verified 2026-09-14 | no ablation of seed size reported |
| Candidate generation | all | SFT (data generation) | decoding | nucleus sampling, T = 0.7, p = 0.9 | arXiv:2308.06259v3 §3.1 | verified 2026-09-14 | no ablation reported |
| LLaMA 7B data-scaling runs | 7B | SFT | examples N → batch, steps | 100 → 8, 30; 800 → 8, 300; 1,600 → 8, 600; 3,200 → 32, 500; 6,400 → 32, 600; 12,800 → 32, 600; 25,600 → 32, 1,200; 51,200 → 32, 1,600 | arXiv:2308.06259v3 App. E; Table 18 | verified 2026-09-14 | fixed per N across datasets "for fair comparison" |
| All runs | all | SFT | epochs, sequence length, optimizer, warmup, packing, compute; separate settings for M_yx, M0, and LLaMA 2 70B | not reported | checked §2–§3, App. A–E, Tables 18–19 | not reported | — |

## Findings relevant to generality and negative samples
- Negative marginal value (sense 1): uncurated augmented data does not raise win rate as data grows, and augmented-only training without curation "does not improve, or even deteriorates" with more data (§3.3 Figure 2; §3.5 Figure 5). Low-scored candidates are discarded; no negative gradient is used.
- Quantity of curated data: the authors report continued gains from more high-quality data and present this as a contrast with the superficial alignment hypothesis of [[lima]] (§3.3) (Interpretation by the authors).
- Breadth vs narrowing: vs the base model, zero-shot Arc-C rises 54.8 → 68.5 (33B) and 56.0 → 73.0 (65B), and MMLU rises 49.5 → 55.4 and 54.8 → 59.0; PIQA falls at both sizes (82.2 → 74.5; 82.8 → 78.9) and OBQA falls at 33B (58.6 → 46.4) (Table 4). Humpback 65B 0-shot MMLU (59.0) is below LLaMA 65B 5-shot (63.4) (Table 7).
- Task diversity: augmented instructions add long-tail verb–noun tasks relative to the seed set, measured on the 8% of seed and 13% of augmented instructions that parse (§3.2, Figure 6). Failures remain on format-specific requests such as ASCII art (App. C, Table 17).
- Distillation: the pipeline uses no external model to generate or curate data (§4); a GPT-4 selector has higher precision and recall than M0 or M1 (Table 9).
- Safety and bias: no red-teaming data is used; on 30 sensitive prompts the model tends to respond cautiously, and S_a gives safer responses (App. A.2). CrowS-Pairs bias-detection accuracy 60.28 vs LLaMA 50.0, which the authors say does not show lower bias in generation (App. A.1, Table 6).

## Connections
- [[self-instruct]] — generates both instructions and outputs with a model; §4 contrasts such methods with using human-written web text as outputs.
- [[openassistant]] — source of the 3,200 seed pairs (§3.1).
- [[lima]] — 65B baseline in Table 3 and the superficial alignment hypothesis discussed in §3.3.
- [[alpagasus]] — cited in §4 as concurrent selection work that prompts ChatGPT to score distilled data.
- [[alpaca]], [[wizardlm]] — distilled datasets in the Table 2 scaling comparison.
- [[genie]] — separate 2024 work on data generation grounded in existing documents; not cited by this paper.
- [[r2e-gym]] — cites this paper as a backtranslation reference (per that card).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2308.06259 (v3, 2024-03-12, ICLR 2024 camera ready; v1 PDF also read for author order).
- Corrections to the previous card version:
  - Title "Humpback: Self-Alignment with Instruction Backtranslation" → "Self-Alignment with Instruction Backtranslation"; Humpback is the model name (§3.1).
  - Author order "Luke Zettlemoyer, Omer Levy" → "Omer Levy, Luke Zettlemoyer" (arXiv abs page and v3; v1 PDF lists the reverse).
  - "Start from a seed aligned model. Apply the model to web documents to infer likely instruction prompts" → two models are finetuned from base LLaMA on 3,200 seed pairs: a backward model p(x|y) that generates the instructions, and a forward model M0 that scores them (§2.2, §2.3).
  - "Curate the generated instruction-document pairs by quality" → the model itself rates each pair on a 5-point prompt; curation is repeated with M1 over two iterations (§2.3).
  - "Fine-tune a stronger student on the curated synthetic set" → the base model is finetuned on seed plus curated data with system-prompt tags (§2.3).
  - "Showed strong open alignment results without relying purely on proprietary teacher-chat distillation" → highest AlpacaEval win rate among non-distilled 33B and 65B models in Table 3; at 70B, LLaMa2 Chat 70B scores higher (92.66 vs 87.94).
  - "Year: 2023" → arXiv v1 2023-08; ICLR 2024.
- Removed as unsupported by the source: "A conceptual predecessor to richer grounded-generation methods like [[genie]]" (the paper does not cite Genie); see-also link to [[openhermes]] (no relation stated in the paper).
- Not reported by the source: epochs, sequence length, optimizer, warmup, and compute; separate settings for M_yx, M0, and the LLaMA 2 70B run; how a threshold of 4.5 is applied to a 5-point rating; released code or checkpoints; run-to-run variance for Table 3.
