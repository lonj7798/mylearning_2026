<!-- scope: Evol-Instruct (WizardLM paper): LLM rewrites of seed instructions into harder and rarer instructions for SFT data
     deps: [[self-instruct]], [[alpaca]]
     see-also: [[wizardlm]], [[wizardcoder]], [[code-evol-instruct]], [[wizardmath]], [[deita]], [[instag]], [[lima]]
-->

# WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
- **Core Insight:** LLaMA 13B fine-tuned on 70k instructions sampled from 250k Evol-Instruct instructions (evolved by ChatGPT from the 52k Alpaca set over 4 rounds) averages 58.96 over nine benchmarks, versus 54.60 for Vicuna-13b (70k ShareGPT conversations) and 43.44 for a re-trained Alpaca-13b (v3 §4.2, Table 1).
- **Guideline:** When an LLM generates SFT instructions from a seed set, evolve them over several rounds with small in-depth rewrites (each adds 10 to 20 words) plus in-breadth mutations, drop failed evolutions, and train on a mix of all rounds, because the average benchmark score rose with each round in this study (Fig. 5b); the seed set still matters, since an evolved ShareGPT seed scored 61.87 versus 58.96 for the Alpaca seed (Table 2).
- **Authors:** Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, et al. (Microsoft, Peking University)
- **Year:** 2023 (arXiv v1 2023-04; v3 2025-05; ICLR 2024)
- **URL:** https://arxiv.org/abs/2304.12244
- **Source type:** paper
- **Relevant topics:** synthetic SFT data, instruction complexity, instruction diversity, distillation from ChatGPT, data filtering

## Abstract
Writing open-domain instruction data by hand is slow and costly, and human-written sets contain few high-difficulty instructions. The paper proposes Evol-Instruct: an LLM rewrites an initial instruction set step by step into more complex instructions. All generated data is mixed and used to fine-tune LLaMA; the resulting model is WizardLM. In automatic and human evaluations, WizardLM outperforms Alpaca (trained on Self-Instruct data) and Vicuna (trained on human-shared ShareGPT conversations). Version note: the v1 abstract instead reported that WizardLM outputs were preferred over ChatGPT on the high-complexity part of the test set, and the v2 abstract added "more than 90% capacity of ChatGPT on 17 out of 29 skills" in GPT-4 evaluation.

## Key Contributions
- Evol-Instruct: five in-depth prompts, one in-breadth prompt, and an elimination filter, applied for M rounds (§3.2); full prompt texts in Examples 3.1-3.3 and App. A-G.
- WizardLM-13b on LLaMA 13B (§4.2); ablation models at 65B (LLaMA-1), 70B (Llama-2), and 7B (Mistral) (§4.5, Table 2). The v1 main model was LLaMA 7B (v1 §4.2).
- WizardEval: 218 real-world instructions covering 29 skills, built by the authors (§4.4, App. H).
- Ablations over seed data, data size, evolver model, and base model (Table 2); difficulty per evolution round (Fig. 5, App. I Table 3).

## Key Figures/Tables to Study
- Figure 1: evolution tree grown from the instruction "1+1=?"; Figure 2: pipeline (evolver, eliminator, instruction pool).
- Table 1: WizardLM-13b vs ChatGPT-3.5, Alpaca, Vicuna, Baize, CAMEL, Tulu on nine benchmarks.
- Table 2: seed, size, evolver, and base-model ablations. Table 3 (App. I): difficulty of ShareGPT, Alpaca, C1-C4 by three judges.
- Table 4 (App. L): scores at epochs 2.5, 2.75, 3.

## Technical Details
- Notation: D(0) = {(I_k, R_k)}, k = 1..N, where I_k is an instruction and R_k its response. Each evolution rewrites every I(t) into I(t+1) and generates R(t+1); M evolutions give D(1)..D(M) (§3.1).
- In-depth evolving has five operations: add constraints, deepening, concretizing, increased reasoning steps, complicating input (§3.2). Each rewrite must be "a bit harder" and may add only 10 to 20 words (Example 3.1). Complicating input uses in-context demonstrations with XML, SQL, Python, HTML, shell, and JSON inputs; the other four are zero-shot (§3.2, App. D).
- In-breadth evolving creates a new instruction in the same domain that is "even more rare", with similar length and difficulty (Example 3.3).
- Operation choice: for each instruction in each round, one of the six prompts is sampled with equal probability (§4.2).
- Responses come from the same LLM, gpt-3.5-turbo through Azure OpenAI, prompted with the instruction alone (§3.2, §4.2 fn. 5).
- Elimination (§3.2) marks four cases as failed: (1) no information gain over the original, judged by ChatGPT (the text points to App. G; the "Equal" prompt is printed in App. F); (2) the response contains "sorry" and is shorter than 80 words; (3) the response contains only punctuation and stop words; (4) the instruction copies words from the evolving prompt, such as "given prompt" or "#Rewritten Prompt#". Failed instructions return to the pool unchanged and are retried in the next round (§3.2).
- Scale: 52k Alpaca seeds, M = 4, 250k instructions in total, 52k × 4 × 3 = 624k API requests (§4.2).
- Training set: the seed set and all rounds are merged and shuffled; 70k examples are sampled with equal probability to match Vicuna's 70k (§3.3, §4.2). Prompt format: the Vicuna chat prompt (§3.3).
- Alpaca-13b baseline: Alpaca data expanded from 52k to 70k with Self-Instruct, text-davinci-003 responses replaced by ChatGPT responses, retrained from LLaMA 13B (§4.1).
- Evaluation: MMLU, ARC, HellaSwag, TruthfulQA (Open LLM Leaderboard code), HumanEval pass@1 (164 problems), GSM8k 4-shot pass@1 (1319 problems), AlpacaEval, MT-Bench, and WizardEval judged by GPT-4 (§4.3). Greedy decoding, max generation length 2048 (§4.2).
- Table 1 (WizardLM-13b vs Vicuna-13b): HumanEval 24.0 vs 12.5; GSM8k 37.15 vs 24.34; AlpacaEval 75.31 vs 70.43; MT-Bench 6.35 vs 6.21; WizardEval 89.1 vs 86.9; TruthfulQA 50.55 vs 52.68 (Vicuna higher). Tulu-13b has higher MMLU (53.19 vs 52.92).
- Human evaluation: 10 annotators rank four shuffled responses (Alpaca-13b, Vicuna-13b, WizardLM, ChatGPT) on 218 WizardEval instructions; all Kappa scores > 0.6 (§4.4). Win rates appear only in Fig. 4b.
- Table 2 ablations (average of nine benchmarks): ShareGPT seed 61.87 (GSM8k 31.46 vs 37.15); full 250k data 60.30; Llama-2-70B-Chat as evolver 56.27; 70k Super-NaturalInstructions 37.73; Mistral-7B base: WizardLM 65.81 vs Alpaca 52.87; WizardLM-65b 69.40; WizardLM-70b 71.33.
- Math share in 2000 sampled instructions: ShareGPT 4.3%, Alpaca 11.8% (§4.5).
- Difficulty (1-10 scale, App. E prompt) on 600 sampled instructions (Table 3): ChatGPT judge Alpaca 3.00, ShareGPT 4.63, C1 5.48, C4 7.08; human judges Alpaca 3.15, ShareGPT 4.55, C4 6.82. On 300 instruction pairs, human-human Kappa 0.68 and ChatGPT-human Kappa 0.66 (App. I).
- Rounds C0-C4 each hold about 52k instructions; average benchmark score per round is shown only in Fig. 5b (§4.5).
- Breadth: BERT 768-d embeddings, t-SNE to 2-D, k-means with 20 clusters; the comparison is qualitative (App. J).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WizardLM-13b | 13B | SFT (data gen) | evolver and response model | gpt-3.5-turbo (Azure OpenAI) | arXiv:2304.12244v3 §4.2 fn. 5 | verified 2026-09-14 | Table 2: Llama-2-70B-Chat evolver avg 56.27 vs 58.96 |
| WizardLM-13b | 13B | SFT (data gen) | seeds; rounds M; total instructions | Alpaca 52k; 4; 250k | v3 §4.2 | verified 2026-09-14 | Fig. 5b per-round scores (figure only) |
| WizardLM-13b | 13B | SFT (data gen) | response sampling | temperature 1, max tokens 2048, frequency penalty 0, top-p 0.9 | v3 §4.2 | verified 2026-09-14 | no ablation reported |
| WizardLM-13b | 13B | SFT | base model; training examples | pre-trained LLaMA 13B; 70k sampled from 250k | v3 §4.2 | verified 2026-09-14 | Table 2: 250k variant avg 60.30 vs 58.96 |
| WizardLM-13b | 13B | SFT | optimizer; initial LR; max tokens | Adam; 2 × 10^-5; 2048 | v3 §4.2 | verified 2026-09-14 | no ablation reported |
| WizardLM-13b | 13B | SFT | batch; epochs; compute | 4 per GPU; 3; 8 V100, DeepSpeed ZeRO-3, 140 hours | v3 §4.2 | verified 2026-09-14 | App. L Table 4: avg 57.92 / 58.24 / 58.96 at 2.5 / 2.75 / 3 epochs |
| WizardLM-13b | 13B | SFT | global batch, LR schedule, warmup, weight decay, loss masking, packing | not reported | v3 body and App. A-L | not reported | none |
| WizardLM (v1) | 7B | SFT | base; LR; batch; compute | LLaMA 7B; Adam 2 × 10^-5; 8 per GPU; 8 V100, ZeRO-3, 70 hours, 3 epochs | arXiv:2304.12244v1 §4.2 | verified 2026-09-14 | no ablation reported |
| WizardLM-65b, -70b, -7b (Mistral) | 65B, 70B, 7B | SFT | hyperparameters | not reported separately | v3 §4.5, App. L | not reported | none |

## Findings relevant to generality, negative feedback, distillation
- Generality (Interpretation by the authors): difficulty is raised in small steps because a set filled with extremely complex instructions "would harm the generalization performance" (§3.2). No ablation of step size is reported.
- Narrowing by seed choice (Result, single study): the ShareGPT seed raised the nine-benchmark average but lowered GSM8k from 37.15 to 31.46; the authors attribute this to fewer math instructions in ShareGPT (4.3% vs 11.8%) (§4.5, Table 2).
- Open-domain vs task-format data: 70k Super-NaturalInstructions examples gave avg 37.73 and MT-Bench 2.86, versus 58.96 and 6.35 for WizardLM data at the same size (Table 2).
- Distillation: all responses are ChatGPT outputs; replacing ChatGPT with Llama-2-70B-Chat as evolver lowered the average to 56.27, still above Vicuna-13b's 54.60 (Tables 1-2).
- Negative feedback: failed evolutions are discarded or retried (negative marginal value); no negative is used as content or gradient (§3.2).
- Measurement limits stated by the authors: GPT-4 and human evaluation have scalability and reliability limits, and WizardEval may not represent all use scenarios (§5). No contamination check is reported.

## Connections
- [[self-instruct]] — produced the Alpaca seed data from 175 human-created seed instructions (§1).
- [[alpaca]] — seed set and re-trained 13B baseline (§4.1).
- [[wizardlm]] — duplicate card for this paper; redirects here.
- [[wizardcoder]], [[code-evol-instruct]] — later code-specific Evol-Instruct paper (arXiv:2306.08568).
- [[wizardmath]] — later math application of evolved instructions (arXiv:2308.09583).
- [[deita]], [[instag]] — later work that scores instruction complexity with other methods (see those cards).
- [[lima]] — contrasting data strategy with a small curated set (see that card).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2304.12244 (v3, 2025-05-27, ICLR 2024 version); v1 (2023-04-24) and v2 (2023-06-10) PDFs read for version differences.
- Corrections to the previous card version:
  - Title "WizardLM: Empowering Large Language Models to Follow Complex Instructions (Evol-Instruct)" is the v1/v2 title → current arXiv and v3 title "…Large Pre-Trained Language Models…".
  - Abstract was the v2 text (17 of 29 skills, Vicuna testset) → v3 abstract, with v1/v2 claims marked as version notes.
  - "Released WizardLM (7B/13B/70B LLaMA derivatives) and its WizardCoder / WizardMath offshoots" → main model 13B (v3) or 7B (v1); 65B, 70B, 7B-Mistral appear as ablations (Table 2); WizardCoder and WizardMath are separate papers.
  - Elimination "(a) same-or-similar check, (b) sorry/refusal markers, (c) punctuation-only outputs, (d) copies the input verbatim" → no information gain; "sorry" with fewer than 80 words; only punctuation and stop words; copies words from the evolving prompt (§3.2).
  - "~250K evolved instructions (after filtering)" → 250k instructions in total; "after filtering" not stated (§4.2).
  - "SFT on LLaMA (7B/13B/70B) with the merged original + evolved set" → 70k sampled from the merged 250k, on LLaMA 13B (v3) or 7B (v1) (§3.3, §4.2).
  - "Complexity histogram: Alpaca flat vs WizardLM long tail" → the paper reports mean difficulty per dataset and round (Fig. 5a, Table 3); "uniform vs skewed" describes test sets (App. H).
  - "GPT-4 skill-wise evaluation, 29 skills" → v2 only; in v3 the 29 skills describe WizardEval (§4.4).
- Removed as unsupported by the source: "complexity distribution, not quantity or topic diversity, drives downstream ability" (the paper calls its analysis "preliminary", §1); "exactly reproducible"; "[[instag]] formalized complexity as tag count" as a claim about this paper; "[[deepseekmath]] often cites WizardMath/WizardCoder".
- Not reported by the source: global batch size, LR schedule, warmup, weight decay, loss masking, packing, per-size settings for 65B/70B/7B, contamination checks, human win-rate numbers in v3 text.
