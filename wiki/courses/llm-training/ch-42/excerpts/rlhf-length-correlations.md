---
chapter: ch-42
course: llm-training
phase: read
excerpt_of: arXiv:2310.03716v2 (A Long Way to Go: Investigating Length Correlations in RLHF), COLM 2024; no library card exists for this slug yet
source_url: https://arxiv.org/abs/2310.03716
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library has no card for this source)"
---

# Excerpt: how much of the RLHF reward gain is length (Singhal, Goyal, Xu, Durrett)

Used by [[read]] §2.1 and the detection section. Checked against arXiv v2 (2024-07-10) on 2026-09-15.

## Setup (§2)
- Three settings: **WebGPT** (question answering, human labels), **Stack** (technical QA, upvote-derived labels), **RLCD** (multi-turn conversation, synthetic preferences; helpfulness subset, 40K preferences, mean 45 tokens per response).
- Llama-7B base for all experiments, LoRA rank 16, Huggingface TRL PPO with KL coefficient λ = 0.04 and batch size 64. SFT checkpoints: AlpacaFarm SFT (WebGPT, RLCD), TRL SFT (Stack).
- Evaluation: the task reward model (intrinsic) and AlpacaFarm simulated preferences (an LLM-judge win rate over 500 held-out prompts). The authors note this judge "may have length biases".

## Non-length reward gain (§3.1, Table 1)
Outputs are bucketed by length in 20-token bins. `ΔR` is the overall reward gain from PPO over SFT; **NRG** is the average within-bucket gain weighted by bucket counts; the ratio is the share of reward gain not explained by the shift to longer outputs.

| | WebGPT STD | WebGPT high λ | Stack STD | Stack high λ | RLCD STD | RLCD high λ |
|---|---|---|---|---|---|---|
| ΔR | 0.82 | 0.20 | 0.89 | 0.67 | 0.94 | 0.61 |
| NRG | 0.02 | 0.03 | 0.48 | 0.37 | 0.25 | 0.12 |
| ratio | 2.0% | 15.1% | 53.4% | 56.5% | 27.2% | 19.1% |

"For WebGPT and RLCD, 70%-90% of the improvement on WebGPT and RLCD can be explained by length shifts." Stack's higher NRG share is attributed by the authors to SFT outputs already being near the length limit (§3.1).

## Length-only reward (§3.2, Table 2)
`R*(y) = 1 − |len(y)/L − 1|`, with target `L` = 156 (WebGPT), 120 (RLCD), 250 (Stack). LPPO is PPO against this reward; SFT-LONG is longest-of-8 sampling from SFT.

| | W-GPT SFT | PPO | SFT-LONG | LPPO | LPPO λ=0 | STACK SFT | PPO | SFT-LONG | LPPO | RLCD SFT | PPO | SFT-LONG | LPPO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tokens | 100 | 230 | 141 | 118 | 167 | 203 | 257 | 249 | 252 | 59 | 94 | 117 | 98 |
| sim. pref. vs SFT | 50% | 58% | 48% | 56% | 53% | 50% | 58% | 57% | 59% | 50% | 63% | 52% | 64% |

"Purely optimizing for length actually reproduces most of the simulated preference improvements of PPO with the learned reward models" (§3.2). LPPO also beats SFT-LONG, whose outputs are longer, and beats LPPO with λ = 0; the authors attribute this to the KL term (Interpretation, §3.2).

## Attribution
The paper's third part tests length-countering interventions and identifies "the dominant source of these biases to be reward models, which ... are non-robust and easily influenced by length biases in preference data" (Abstract).

## Not in this source
Scales beyond 7B (stated as a limitation), and any RL run with a length-controlled judge as the evaluation.
