---
chapter: ch-15
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rlhf-length-correlations.md (planned card; not present on 2026-09-15; values below are taken from the primary source)
source_url: https://arxiv.org/abs/2310.03716
primary_version: arXiv:2310.03716v2 (2024-07-10); v1 2023-10; COLM 2024
created_at: "2026-09-15"
---

# Excerpt: A Long Way to Go: Investigating Length Correlations in RLHF

Authors: Prasann Singhal, Tanya Goyal, Jiacheng Xu, Greg Durrett (UT Austin, Princeton, Salesforce AI). Source type: paper. Read in the v2 PDF text on 2026-09-15 for ch-15 §5 and the negative-feedback section.

## Settings (§2.1)
| Setting | Labels | Size used | Mean tokens per response |
|---|---|---|---|
| WebGPT (long-form QA) | human preference labels | 19.6K | 169 |
| Stack (StackExchange) | derived from upvotes | 100K pairs | 236 |
| RLCD (HH prompts) | synthetic, from prompt heuristics | 40K | 45 |
Policy and RM base: Llama-7B; TRL PPO with LoRA rank 16; λ (KL) = 0.04; batch 64. Downstream metric: AlpacaFarm simulated preference with 12 API-based annotators on 500 held-out prompts; the authors note this metric "may have length biases" (§2.1).

## Length share of reward gain (Table 1)
NRG = within-length-bucket reward gain weighted by bucket size (20-token buckets); ratio = NRG / ΔR.
| | WGPT STD | WGPT HIGH λ | STACK STD | STACK HIGH λ | RLCD STD | RLCD HIGH λ |
|---|---|---|---|---|---|---|
| ΔR | 0.82 | 0.20 | 0.89 | 0.67 | 0.94 | 0.61 |
| NRG | 0.02 | 0.03 | 0.48 | 0.37 | 0.25 | 0.12 |
| ratio | 2.0% | 15.1% | 53.4% | 56.5% | 27.2% | 19.1% |
HIGH λ sets λ = 0.12 (§4.1).

## Length-only reward (§3.2, Table 2; simulated preference vs SFT)
LPPO uses R*(y) = 1 − |len(y)/L − 1| with target L = 156 (WebGPT), 120 (RLCD), 250 (Stack).
| | WGPT | STACK | RLCD |
|---|---|---|---|
| SFT length / win rate | 100 / 50% | 203 / 50% | 59 / 50% |
| PPO | 230 / 58% | 257 / 58% | 94 / 63% |
| SFT-LONG (longest of 8) | 141 / 48% | 249 / 57% | 117 / 52% |
| LPPO | 118 / 56% | 252 / 59% | 98 / 64% |
| LPPO λ = 0 | 167 / 53% | 248 / 58% | 163 / 51% |

## Preference data and RM interventions (§4.2, Tables 4-6)
- Table 5, accuracy of always preferring the longer response: WGPT 55.7%, STACK 59.6%, RLCD 63.1% ("Above random (50%) accuracy indicates length bias").
- Table 4, RM eval accuracy / within-batch Pearson correlation between length and reward over 8 generations per input: STND WGPT 61.5% / 0.72, STACK 70% / 0.55, RLCD 80% / 0.67. Length balancing (BAL): 52.6% / −0.13, 61.9% / −0.09, 73.1% / 0.62. Random pairing augmentation (R-DA): 62.5% / 0.35, 72.6% / 0.37, 80% / 0.43.
- Table 6: on STACK, BAL gave length 148 vs 203 for SFT and 257 for standard PPO, with simulated preference 57% vs standard PPO; results "are inconsistent" across settings.
- "no strategy works for all settings" (§1).

## Training dynamics (§5, Fig. 5)
Most training examples have near-zero RM confidence; "the length heuristic applies to most examples that are easy, and ... the overwhelming majority of strong negative predictions are cases where the model follows the length heuristic to confidently predict the wrong answer."
