<!-- scope: Tülu 1 (AI2 / UW, Jun 2023; NeurIPS 2023 D&B) — controlled comparison of 12 open instruction datasets and 4 base-model families under one SFT recipe, evaluated on knowledge, reasoning, multilinguality, coding, safety, GPT-4 preference, and human preference
     deps: [[self-instruct]], [[alpaca]]
     see-also: [[tulu-2]], [[tulu-3]], [[lima]], [[false-promise-imitating-proprietary-llms]], [[length-controlled-alpacaeval]]
-->

# How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources
- **Core Insight:** With one SFT recipe on LLaMA 13B, no single instruction dataset is best on all six capability evaluations (the Human+GPT mixture is best on average, 45.2, but best on only 2 of 6), and GPT-4-judged AlpacaEval win rate correlates with the average number of unique tokens in responses (Pearson r = 0.96), so preference-style evaluation can hide capability differences that benchmarks expose (§5.1, §5.4, Table 3, Fig. 2).
- **Guideline:** When choosing an SFT mixture for a general-purpose model, compare every candidate against the base model on knowledge, reasoning, multilingual, coding, safety, and open-ended evaluations together, because the dataset with the highest AlpacaEval win rate at 13B (ShareGPT, 70.5%) lowers TyDiQA F1 from 43.2 to 30.5, and 6 of 12 single datasets lower GSM (CoT) and 8 of 12 lower TyDiQA below vanilla LLaMA 13B (counts from Table 3; §5.1).
- **Authors:** Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, et al. (Allen Institute for AI; University of Washington)
- **Year:** 2023 (arXiv v1 2023-06; v2 2023-10 camera ready; NeurIPS 2023 Datasets and Benchmarks Track)
- **URL:** https://arxiv.org/abs/2306.04751
- **Source type:** paper
- **Relevant topics:** instruction tuning, SFT data mixtures, multi-axis evaluation, forgetting after SFT, LLM-as-judge bias, distilled instruction data, base-model quality

## Abstract
Claims that open instruction-tuned models match proprietary models often rest on limited evaluation. The paper trains models from 6.7B to 65B parameters on 12 instruction datasets, from manually curated (e.g., OpenAssistant) to synthetic and distilled (e.g., Alpaca), and evaluates factual knowledge, reasoning, multilinguality, coding, safety, and open-ended instruction following with automatic, model-based, and human metrics. It introduces Tülu, a suite fine-tuned on a combination of open resources. Different datasets uncover or enhance specific skills, no single dataset or combination is best on all evaluations, and model- and human-preference evaluations fail to reflect capability differences exposed by benchmarks. The best model in any given evaluation reaches on average 87% of ChatGPT and 73% of GPT-4 performance. Models, code, data, and the evaluation framework are released.

## Key Contributions
- Controlled dataset comparison: 12 datasets (Table 1) trained with the same format, loss, and hyperparameters on LLaMA 7B and 13B; 30B and 65B runs cover only ShareGPT, the Human mixture, and the Human+GPT mixture (Table 8).
- Base-model comparison on the Human+GPT mixture: LLaMA 7B, LLaMA-2 7B, OPT 6.7B, Pythia 6.9B (Table 4).
- Evaluation suite: MMLU, GSM, BBH, TyDiQA, Codex-Eval (HumanEval), AlpacaEval, ToxiGen, TruthfulQA, and a human study (§4, App. E).
- Tülu (LLaMA 7B-65B) and Tülu-1.1 (LLaMA-2 7B, 13B) trained on the Human+GPT mixture of 7 datasets (§3.3).
- Analysis of GPT-4 preference bias toward responses with more unique tokens (§5.4, App. H).

## Key Figures/Tables to Study
- **Table 1:** dataset sources, instance counts, turns, prompt and completion lengths.
- **Table 3:** per-dataset results at 13B, colored by gain or loss vs vanilla LLaMA. **Table 8:** all runs with direct/CoT and GP/CB splits.
- **Table 5:** Tülu at all sizes vs vanilla LLaMA, ChatGPT, GPT-4. **Table 6:** ToxiGen and TruthfulQA.
- **Figure 2 / Figure 6:** AlpacaEval win rate vs unique tokens; dummy unique-token evaluator vs GPT-4.
- **Figures 3-4:** human acceptance and pairwise preference.

## Technical Details
- Format (§3.1): `<|user|>` before user turns, `<|assistant|>` before targets, `</s>` after each assistant output. Loss L = −Σ_j log p_θ(t_j | t_<j) · 1[t_j ∈ Y], where t_j is the j-th token and Y the set of assistant tokens; all input tokens are masked (§3.2).
- Mixtures (§3.3): Human = FLAN V2, CoT, Dolly, Open Assistant 1 (SuperNI excluded because FLAN V2 contains most SuperNI tasks). Human+GPT = Human + GPT4-Alpaca, Code-Alpaca, ShareGPT. Datasets are concatenated with no sampling weights.
- Data handling: CoT and FLAN V2 subsampled to 100K (Table 1); CoT is the CoT mixture split out of FLAN v2 (App. C); ShareGPT is the HTML-cleaned reproduction, split into chunks of at most 2048 tokens with no further filtering (footnote 3, App. C).
- Evaluation settings (App. E): GSM uses 200 of 1319 test examples; BBH 3-shot; TyDiQA 1-shot, gold passage and closed book; Codex-Eval pass@1 at temperature 0.1 and pass@10 at 0.8; ToxiGen 500 hateful prompts per group; TruthfulQA 818 questions, 6-shot; AlpacaEval 805 prompts vs Davinci-003 with annotator `alpaca_eval_gpt4_0314`, up to 8192 generated tokens. Models are loaded in 8-bit for evaluation.
- 13B dataset averages (Table 3): Human+GPT mix 45.2; ShareGPT 42.0; Human mix 39.2; GPT4-Alpaca 37.6; SuperNI 20.9; Self-instruct 21.8. CoT raises GSM from 14.5 to 40.0; SuperNI lowers GSM to 4.0 and BBH from 39.3 to 4.5.
- Base models on the Human+GPT mix (Table 4, average): LLaMA-2 7B 45.7, LLaMA 7B 38.3, Pythia 6.9B 26.2, OPT 6.7B 22.2.
- Scale (Table 5): Tülu 65B average 56.7 vs ChatGPT 72.3 and GPT-4 86.9; Tülu-1.1 13B 52.9. Relative to vanilla LLaMA 65B, Tülu 65B changes MMLU by +0.5, BBH by −3.7, TyDiQA by −0.2 (§5.2).
- Open-ended (Table 7): ShareGPT win rate 62.4 / 70.5 / 69.1 / 73.6 at 7B / 13B / 30B / 65B; Tülu 48.6 / 56.5 / 62.3 / 61.8. Unique-token dummy evaluator vs GPT-4 win rate: R² = 0.91 (App. H).
- Human study (§4.3, App. G): 332 instructions (252 Self-Instruct + 80 Vicuna), 18 expert annotators, LLaMA-1 models. Acceptance: ChatGPT 90.1%, Tülu 65B 79.8%, Tülu 65B (Human mix) 72.3%, Tülu 7B 68.7% (Fig. 3). Tülu 65B vs ChatGPT: 27.7% Tülu better, 33.1% tie, 39.2% ChatGPT better (Fig. 4). Agreement: 0.84 acceptance, 0.72 tie-discounted pairwise (App. G.2).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Tülu 7B, 13B; Tülu-1.1 7B, 13B; all 7B/13B dataset runs | 7B, 13B | SFT | epochs; peak LR; schedule | 2 epochs; 2e-5; linear warmup for 3% of total steps, linear decay; no weight decay | arXiv:2306.04751v2 App. D | verified 2026-09-14 | no ablation reported |
| Tülu 30B, 65B; 30B/65B dataset runs | 30B, 65B | SFT | epochs; peak LR; schedule | 2 epochs; 1e-5; same warmup, decay, and weight decay | App. D | verified 2026-09-14 | no ablation reported |
| Tülu 7B, 13B; Tülu-1.1 | 7B, 13B | SFT | max sequence length | 2048 tokens, truncating longer samples | App. D | verified 2026-09-14 | no ablation reported |
| Tülu 30B, 65B | 30B, 65B | SFT | max sequence length | 1024 tokens, truncating longer samples | App. D | verified 2026-09-14 | no ablation reported |
| all runs | all | SFT | loss masking | loss only on assistant tokens (from the assistant role token to the next user role token); input tokens masked | §3.2 | verified 2026-09-14 | no ablation reported |
| Tülu (Human+GPT mix) | all | SFT | instances per dataset | FLAN V2 100,000; CoT 100,000; Dolly 15,011; Open Assistant 1 34,795; GPT4-Alpaca 52,002; Code-Alpaca 20,022; ShareGPT 168,864; concatenated | Table 1, §3.3 | verified 2026-09-14 | Table 3: mix average 45.2 vs best single dataset 42.0 (ShareGPT) at 13B |
| Tülu (Human+GPT mix) | all | SFT | total instances | 490,694 | sum of the Table 1 rows above | derived | — |
| all runs | all | SFT | optimizer; batch size; packing | not reported (DeepSpeed with ZeRO, full fine-tuning) | checked §3, App. D, App. E | not reported | — |
| all runs | all | SFT | hardware | CSC LUMI nodes with 4 AMD MI250x GPUs each; GPU count and hours not reported | App. D | verified 2026-09-14 | — |

## Findings relevant to generality and distillation
- **Skill-specific data (Result, single study).** Datasets targeted at a skill raise that skill, e.g. CoT on GSM and Code-Alpaca on Codex-Eval; the mixture wins on average but not per task (§5.1, Table 3).
- **Forgetting.** Most datasets lower GSM and TyDiQA relative to the base model (at 13B: 6 of 12 on GSM, 8 of 12 on TyDiQA, counted from Table 3); the authors attribute this to little CoT or multilingual data in those datasets (Interpretation) (§5.1). At 65B, instruction tuning does not improve MMLU, BBH, or TyDiQA over the base, and the authors state that care is needed to avoid forgetting base capabilities (§5.2).
- **Evaluation error.** AlpacaEval rewards long, diverse outputs: at 13B, CoT, FLAN V2, and SuperNI score 6.0, 3.2, and 4.2 despite benchmark gains (Table 7, §5.4). Tülu 7B beats Tülu 65B (Human mix) in AlpacaEval (48.6 vs 43.4) but has lower human acceptance (68.7% vs 72.3%) (§5.5). The reference model Davinci-003 may bias win rates (App. A).
- **Over-refusal.** Tülu 13B produces 0.1% toxic generations on ToxiGen vs 27.7% for ChatGPT; the authors hypothesize overfitting to refusal-style behavior (Interpretation) (§5.3, Table 6). TruthfulQA does not improve with size because larger models hedge and refuse more often (§5.3).
- **Distilled data.** GPT-distilled sets give the best open-ended and safety results, while Self-instruct data generated by base GPT-3 degrades most tasks (§5.1, §5.3). The authors state that imitation data can help if it covers diverse skills and domains, in contrast to [[false-promise-imitating-proprietary-llms]] (§6; Interpretation).
- **Scope limits.** No RL or preference training (§2.1); no multi-turn or summarization evaluation (App. A); contamination of ChatGPT/GPT-4 on the suite cannot be ruled out (§5.2).

## Connections
- [[self-instruct]], [[alpaca]], [[openassistant]], [[baize]] — four of the 12 compared datasets (Table 1).
- [[flan-collection]], [[super-natural-instructions]] — sources of the FLAN V2, CoT, and SuperNI sets.
- [[lima]] — cited ([56]) among claims supported mostly by model-based evaluation (§1).
- [[false-promise-imitating-proprietary-llms]] — concurrent work on style-only imitation discussed in §6.
- [[tulu-2]], [[tulu-3]], [[tulu-3-sft-mix]] — later releases in the Tülu line.
- [[open-instruct-allenai-recipes]] — the released training and evaluation code repository (github.com/allenai/open-instruct).
- [[length-controlled-alpacaeval]], [[rlhf-length-correlations]] — later work on length effects in automatic preference evaluation and RLHF.
- [[judge-llm-bias]], [[chatbot-arena]] — LLM-as-judge and human-vote evaluation; the May 2023 Chatbot Arena blog post is cited as [55].
- [[helm]] — cited ([28]) as a broad framework focused on base models (§6).

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2306.04751 (v2, 2023-10-30; PDF read in full including App. A-I; Figures 3-4 read from the rendered page 9).
- Audit claims not found in the source: none. Scope note: "trained on 12 open instruction datasets from 6.7B to 65B" is the abstract's wording; the per-dataset runs exist only at 7B and 13B (Table 8).
- Not reported by the source: optimizer name and betas, global batch size, packing, total GPU hours.
