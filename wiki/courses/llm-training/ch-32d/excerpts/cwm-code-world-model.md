---
chapter: ch-32d
course: llm-training
phase: read
excerpt_of: Meta FAIR CodeGen Team, "CWM: An Open-Weights LLM for Research on Code Generation with World Models", arXiv:2510.02387v1 (2025-09-30)
source_url: https://arxiv.org/abs/2510.02387
created_at: "2026-09-15"
source_type: official technical report
---

# Excerpt: CWM — execution-trace and ForagerAgent mid-training

No library card for this source exists at the time of writing (planned slug `cwm-code-world-model`). Only passages used by [[read]] §4, §9, §10, §11, the negatives section, and the Recipe table are extracted.

## Stages (Figure 1, §4.2)

| Stage | Tokens | Context |
|---|---|---|
| General pre-training | 8T | 8k |
| Code world modeling mid-training | 5T | 131k |
| SFT (instruction and reasoning) | 100B | 32k |
| Joint RL (agentic and reasoning) | 172B | 131k |

- Model: dense 32B decoder, 64 layers, alternating local (8,192-token) and global (131,072-token) sliding-window attention, Scaled RoPE θ = 10⁶ with scale factor 16 from mid-training onwards (Table 2, §4.1).
- Optimizer: AdamW, β1 = 0.9, β2 = 0.95, weight decay 0.1, gradient clipping 1.0; 2,000 warmup steps; cosine decay, peak LR 8 × 10⁻⁴, decaying by 100× over a 13T-token horizon, "with the last 5 T tokens of the scheduler used during mid-training" (§4.1).
- Pre-training: global batch 8.4M tokens, context 8,192, code "about 30 % of the mix". Mid-training: global batch 33M tokens, maximum context 131k (§4.2). Footnote 4: "We have observed lackluster performance when training on long-context data at smaller batch sizes."

## Mid-training data (§2.2, §2.3, §4.2)

- Function-level Python traces: "over 120 M traced Python functions"; CodeContests solution traces (262k generated solutions "filtered to ensure a balance of incorrect and correct submissions", 33k snippets, 70k traces); about 70k execution-traced repository commits; 75M natural-language trace descriptions from functions and 110k from CodeContests (§2.2).
- Trace format (Figure 3): source context, then frames separated by custom tokens `<|frame_sep|>`, `<|call_sep|>`, `<|line_sep|>`, `<|return_sep|>`, `<|action_sep|>`; each frame is a JSON dictionary of local variables followed by the executed line.
- ForagerAgent (§2.3, Table 1): SWE-Agent toolset (create file, edit file, run bash, view/navigate); agents Llama3-70B-Instruct or Qwen3-235B-A22B without thinking; SWE-bench repositories and forks filtered out; 3M trajectories from 10.2k images and 3.15k repositories; 55% issue-fix; mutate-fix split functions 7%, arguments 9%, variables 6%, statements 11%, operators 12%.
- Post-processing (§2.3): MinHash near-deduplication so that kept trajectories have pairwise Jaccard similarity below 0.5. "we do not filter trajectories based on whether they succeed at bug or issue resolution. Following the same motivation, we further train the model to predict both agent and environment turns, although we stochastically mask loss for 50 % of observations as they exhibit limited diversity."
- Mixture (§4.2): "CWM-specific data makes up 30 % of the overall mid-training datamix. We further increase the fraction of general code data to 40 % and keep 30 % for rehearsal of the initial pre-training datamix, as this proved essential in retaining performance on standard evaluations."
- Epochs (§4.2): scaling-law experiments that simulate epoching give "between 1 and 4 target epochs per dataset. The final proportion of a dataset in the mid-training mix is then calculated such that the desired number of epochs is reached at the end of mid-training."
- Long context (§4.2): "about 30 % of documents exceeding 65 k tokens"; no separate long-context phase; length buckets (0, 16385], (16385, 65537], (65537, ∞) with global attention capped at 32,768 in the middle bucket.

## Ablation (§7.1, Table 4): 8B models, 6T pre-training + 1T mid-training, SFT without RL

| PRs | Tracing | Forager | CruxEval-O | CruxEval-I | Oracle SBV NLL | Agentic SBV NLL (32k) | SBV pass@1 |
|---|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 45.4 | 44.1 | 0.64 | 0.39 | 14.6 |
| ✓ | ✗ | ✗ | 44.6 | 45.8 | 0.55 | 0.37 | 18.6 |
| ✓ | ✓ | ✗ | 73.9 | 51.5 | 0.54 | 0.38 | 18.4 |
| ✓ | ✓ | ✓ | 74.5 | 54.8 | 0.54 | 0.29 | 22.1 |

## General and long-context evaluation

- Table 13 (no reasoning, greedy): CWM / CWM-Mid (after mid-training) / Qwen3-32B: MMLU 77.7 / 73.6 / 83.6; MMLU-Pro 60.2 / 52.3 / 65.5; GPQA 40.6 / 31.7 / 49.5; GSM8k 93.3 / 84.7 / 93.4; HumanEval-Plus 75.0 / 68.3 / 72.1; MBPP 73.4 / 67.8 / 78.2; CRUX-O 83.4 / 78.9 / 72.5. No row for the checkpoint before mid-training.
- Table 14 (RULER): CWM 84.3 at 32k, 69.7 at 128k; Qwen3-32B 94.4 / 85.6; Gemma-3-27B 91.1 / 66.0.
- §5: "As we do not intend to develop a general-purpose chatbot we therefore deliberately omit an RLHF stage." §5.1: about 30% of the SFT mix is rehearsal from mid-training.

## Verification

- Checked on 2026-09-15 against https://arxiv.org/abs/2510.02387 (v1), §1-§2, §4.1-§4.2, §5.1, §7.1, §7.7, App. J.
- Not reported by the source: per-dataset token shares inside the 30% CWM-specific portion; ForagerAgent success rate; an ablation of success-filtered versus unfiltered trajectories; an ablation of the 30% rehearsal share; evaluation of the 8T pre-trained checkpoint on the Table 13 benchmarks.
