<!-- scope: WizardCoder models — the checkpoints, training configuration and benchmark results of the Code Evol-Instruct paper (§3.2, §4, §5). The method itself (prompt template, five heuristics, stop rule, decontamination) is in [[code-evol-instruct]].
     deps: [[evol-instruct]]
     see-also: [[code-evol-instruct]], [[oss-instruct]], [[wizardmath]], [[opc-synthetic-code]]
-->

# WizardCoder: Empowering Code Large Language Models with Evol-Instruct
- **Core Insight:** Fine-tuning StarCoder 15B on approximately 78k instructions evolved from the roughly 20k Code Alpaca seed set gives HumanEval pass@1 of 57.3 and MBPP pass@1 of 51.8, and the same data on CodeLlama-34B-Python gives 71.5 and 61.2 (§4.2, §4.3 Table 1).
- **Guideline:** When instruction-tuning an existing code base model, evolve a small seed instruction set into a larger, more complex set and train for a short fixed schedule, because in this paper 200 steps at batch 512 over the evolved data moved StarCoder 15B from 33.6 to 57.3 HumanEval pass@1 (§4.2, Table 1). The result is reported only for StarCoder 15B and CodeLlama-34B-Python.
- **Authors:** Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, et al. (10 authors; Microsoft, Hong Kong Baptist University)
- **Year:** 2023 (arXiv v1 2023-06; v2 2025-05; ICLR 2024)
- **URL:** https://arxiv.org/abs/2306.08568
- **Source type:** paper
- **Relevant topics:** code SFT, instruction evolution, HumanEval, MBPP, MultiPL-E, DS-1000

## Abstract
The paper presents Code Evol-Instruct, an adaptation of the Evol-Instruct method of [[evol-instruct]] to
code instructions, and the WizardCoder models trained on the resulting data. The evolved data is produced
by prompting gpt-3.5-turbo with five heuristics that increase the difficulty of a programming question, run
iteratively over the Code Alpaca dataset. The authors fine-tune StarCoder 15B and CodeLlama-34B-Python on
the merged evolved data and evaluate on HumanEval, HumanEval+, MBPP, DS-1000 and MultiPL-E. WizardCoder 15B
exceeds Anthropic's Claude and Google's Bard on HumanEval and HumanEval+ under the EvalPlus leaderboard
setting, and WizardCoder 34B reaches a HumanEval score comparable to GPT-3.5 (ChatGPT) and exceeds it on
HumanEval+ (Abstract, §4.3).

## Key Contributions
- Code Evol-Instruct: five code-specific difficulty heuristics applied iteratively to a seed instruction set, with a dev-set stop rule (§3.1, §3.2). Details in [[code-evol-instruct]].
- WizardCoder 15B and 34B checkpoints, trained from StarCoder 15B and CodeLlama-34B-Python (§3.2, §4.2).
- Evaluation on five code benchmarks: HumanEval, HumanEval+, MBPP, DS-1000, MultiPL-E (§4.3–§4.5).
- Analyses separating the effect of instruction complexity from sample count, token count, and test-set similarity (§5, Tables 4–5, Fig. 4).

## Key Figures/Tables to Study
- **Table 1** (§4.3) — pass@1 on HumanEval and MBPP for WizardCoder 15B/34B against closed and open baselines.
- **Table 2** (§4.4) — MultiPL-E pass@1 across 8 languages.
- **Table 3** (§4.5) — DS-1000 pass@1 by library, completion and insertion modes, 15B only.
- **Table 5** (§5) — sample-matched and token-matched comparison of each evolution round against the seed data.

## Technical Details
- Seed data: Code Alpaca, approximately 20k samples; evolved data after merging rounds is approximately 78k samples (§3.2, §4.2).
- Evolution and response generation model: OpenAI gpt-3.5-turbo (§4.2). CodeLlama-Instruct-34B and GPT-4 are tested as alternative evolvers in the analysis (§5 Table 4, App. E).
- Base models: StarCoder 15B and CodeLlama-34B-Python (§3.2).
- HumanEval has 164 problems with an average of 9.6 test cases each; HumanEval+ raises this to an average of 774.8 test cases; MBPP has 500 problems with three test cases each (§4.3).
- Table 1 pass@1 is estimated from n=200 samples at temperature 0.2, top-p 0.95 (Table 1 caption). MultiPL-E uses temperature 0.2, top-p 0.95, max length 512, n=50 (Table 2 caption). DS-1000 uses temperature 0.2, top-p 0.5, max length 1024, 40 samples (Table 3 caption).
- Table 1 results: WizardCoder 15B HumanEval 57.3 / MBPP 51.8; WizardCoder 34B 71.5 / 61.2; StarCoder 15B 33.6 / 43.6 (reproduced); CodeLlama-Python 34B 53.7 / 56.2; GPT-3.5 48.1 / 52.2; GPT-4 67.0 (§4.3, Table 1).
- EvalPlus leaderboard setting, greedy decoding, reported in prose: WizardCoder 34B HumanEval+ 64.6 against GPT-3.5 63.4; WizardCoder 15B HumanEval 59.8 against Claude-Plus 53.0 and Bard 44.5 (§4.3).
- MultiPL-E, WizardCoder 34B: Java 44.9, JavaScript 55.3, C++ 47.2, PHP 47.2, R 39.8, Julia 41.5, Swift 44.3, Rust 46.2 (§4.4, Table 2).
- DS-1000, WizardCoder 15B insertion mode: 32.8 overall against StarCoder 25.4 (§4.5, Table 3).
- Evolver ablation: with StarCoder-15B as base, evolving with GPT-4 gives 62.2 pass@1, gpt-3.5 gives 59.8, CodeLlama gives 55.5; with CodeLlama-34B as base, 73.8 / 73.2 / 70.1 (§5, Table 4).
- Quantity control: at matched sample count, round 0 (seed, 20.0k) gives 45.7 and rounds 1–4 (18.8k–19.7k) give 56.1 / 53.0 / 54.3 / 51.2; at matched 2.3M tokens, round 0 gives 44.5 and rounds 1–4 give 51.8 / 52.4 / 50.0 / 49.4 (§5, Table 5).
- Decontamination: gte-large retrieves the top 5 training samples for each test sample, then GPT-4 makes a binary match decision and matching training samples are removed (App. C).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| WizardCoder 15B | 15B | SFT | base model | StarCoder 15B | arXiv:2306.08568v2 §3.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 34B | 34B | SFT | base model | CodeLlama-34B-Python | arXiv:2306.08568v2 §3.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | training examples | ~78k evolved samples (seed: Code Alpaca ~20k) | arXiv:2306.08568v2 §3.2, §4.2 | verified 2026-09-18 | §5 Table 5: matched-sample and matched-token comparisons against seed data |
| WizardCoder 15B / 34B | 15B, 34B | SFT | global batch (sequences) | 512 | arXiv:2306.08568v2 §4.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | sequence length | 2048 | arXiv:2306.08568v2 §4.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | steps / warmup | 200 steps, 30 warmup steps | arXiv:2306.08568v2 §4.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | peak LR, schedule | 2e-5, cosine | arXiv:2306.08568v2 §4.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | precision | fp16 mixed precision | arXiv:2306.08568v2 §4.2 | verified 2026-09-18 | no ablation reported |
| WizardCoder 15B / 34B | 15B, 34B | SFT | epochs, optimizer, weight decay, hardware, cost | not reported (checked §3.2, §4.2, appendices A–E) | arXiv:2306.08568v2 | not reported | — |

## Findings relevant to generality
- MultiPL-E covers 8 languages beyond Python and WizardCoder 34B is the best evaluated model on all 8 (§4.4, Table 2), so the gain from Python-centric evolved data transfers to other languages in this evaluation.
- The gain is not explained by data volume: at matched sample count and at matched token count, each evolved round beats the seed round (§5, Table 5).
- The gain is not explained by increased similarity to the test set: average GPT-4-scored similarity between HumanEval problems and the top-1 retrieved training sample does not increase across rounds and stays low (§5, Fig. 4).

## Connections
- [[code-evol-instruct]] — the method section of this same paper: prompt template, five heuristics, round and stop rules, decontamination.
- [[evol-instruct]] — the general-domain method this adapts.
- [[wizardmath]] — the math adaptation by an overlapping author group, which adds reward models and PPO.
- [[oss-instruct]] — an alternative code-data synthesis route grounded in real code snippets.

## Verification
- Checked on 2026-09-18 against: https://arxiv.org/abs/2306.08568 (arXiv v2, 27 May 2025; ICLR 2024 camera-ready)
- Corrections to the previous card version:
  - Title "WizardCoder: Empowering Code LLMs with Code Evol-Instruct" → "WizardCoder: Empowering Code Large Language Models with Evol-Instruct" (paper title page).
  - Affiliation "Microsoft" → Microsoft and Hong Kong Baptist University (title page).
  - Operator list: the previous card listed "deepen problem complexity" and "require specific language or library (port to Rust, use NumPy vectorization)". The paper's five heuristics are: add constraints and requirements (~10 additional words); replace a common requirement with a less common, more specific one; add reasoning steps; provide erroneous code as misdirection; propose higher time or space complexity requirements, used sparingly (§3.1).
  - "Per seed, one operator is randomly chosen" → the paper specifies iterative rounds whose outputs are merged with all previous rounds and the seed set; it does not describe random per-seed operator selection (§3.2).
  - "beats Claude, Bard" attached to the Table 1 numbers → those comparisons are from the EvalPlus leaderboard setting with greedy decoding (§4.3), which is a different setting from Table 1's n=200 estimate.
  - "WizardCoder-34B / -Python-34B: near GPT-3.5 on HumanEval" → Table 1 reports 71.5 for WizardCoder 34B and 48.1 for GPT-3.5; the "comparable to GPT-3.5" statement in the abstract refers to the EvalPlus leaderboard figure (64.6 vs 63.4 on HumanEval+).
- Removed as unsupported by the source:
  - "WizardCoder-Mistral-7B: best open 7B at release" — no Mistral-based WizardCoder in this paper.
  - "HumanEval+ 50.6" for the 15B model — no such number in the text or tables.
  - Filtering steps "drop evolutions where teacher refuses", "length-based sanity filter", "deduplicate via exact/near-exact match" — the paper describes a dev-set stop rule (§3.2) and an embedding+GPT-4 decontamination filter (App. C), not these.
  - "Cost estimate: O(one GPT-3.5 call per evolution)" — not stated.
  - "Mode collapse risk … operator randomization + filtering are load-bearing" — not stated.
  - "independent audits found stylistic overlap with HumanEval tasks" — not in this paper; the paper's own similarity analysis reports the opposite for its own data (§5, Fig. 4).
  - "License on WizardLM family has fluctuated (removed/restored)" — not stated in the paper.
  - "~4× expansion is enough" as a guideline — the paper reports best dev and HumanEval pass@1 after three rounds (§5, Fig. 3) but does not state an expansion-ratio rule.
- Not reported by the source: epochs, optimizer and weight decay, training hardware and cost, the licence of the evolved dataset, and whether the ~78k set was publicly released.
