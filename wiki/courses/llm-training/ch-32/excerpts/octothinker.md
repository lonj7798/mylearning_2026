---
chapter: ch-32
course: llm-training
phase: read
excerpt_of: primary source arXiv:2506.20512v1 (planned library card papers/octothinker.md; not present on 2026-09-15)
source_url: https://arxiv.org/abs/2506.20512
created_at: "2026-09-15"
---

# Excerpt: OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling

**Authors:** Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu (Shanghai Jiao Tong University, SII, GAIR Lab). arXiv v1 2025-06-25. Source type: paper.

## Observation that motivates the study (§2)

- Zero RL (R1-Zero-style RL from a base model) with GRPO in verl on MATH8K prompts: batch 128, 16 rollouts per query, PPO mini-batch 64, temperature 1.0, max output 4,096 tokens, LR 1e-6, KL loss coefficient 0 (§2.1).
- Llama-3.2-3B-Base produces responses that start with "\boxed:{}" and repeat until the 4,096-token limit, with marginal gains or regressions (GSM8K); Qwen2.5-3B-Base response length grows steadily and scores improve (§2.2, Figure 2).
- Paper's definition: "Mid-training is a mid-stage whose computational and data (token) requirements are intermediate between pre-training and post-training", aimed at objectives such as domain expansion, long-context extension, data-quality improvement, large-scale synthetic data, and "preparing for post-training" (§2 box).

## Controlled 20B-token mid-training on Llama-3.2-3B-Base (§3.1–§3.5)

- Default: 20B tokens, cosine LR without warmup, peak 3e-5, minimum 1/10 of peak, sequence 8,192, batch 4M tokens, Nanotron (§3.1).
- Math web corpus: MegaMath-Web-Pro and MegaMath-Web-Pro-Max give "significant gains" in RL; FineMath-4plus gives marginal gains and responses that grow to 4,096 tokens with repeated "Solution" statements (§3.2, Figure 5).
- QA data at a 9:1 web:QA ratio: short-CoT QA gives no RL gain over web-only; long-CoT QA gives RL gains but unstable training with sudden drops and length spikes (§3.3, Figure 6).
- Instruction data at a 1:89:10 instruction:web:QA ratio (about 0.8B tokens from TULU3 persona IF, WildChat, UltraChat) improves RL with short-CoT QA and stabilizes length; with long-CoT QA it does not prevent the late decline (§3.4, Figures 7–8).
- Stabilizers for long-CoT bases: an Open-Reasoner-Zero-style prompt template and a staged maximum response length (2,048 tokens for steps 0–200, 4,096 for 200–320, 8,192 for 320–400) (§3.4, Figures 9–10).
- Budget: checkpoints at 20B, 70B, 100B tokens of MegaMath-Web-Pro-Max; 70B and 100B have comparable base-model scores, but RL results improve from 20B to 70B and from 70B to 100B. Takeaway: "Increasing the mid-training budget can improve RL performance, even if such gains are not evident in base model evaluations" (§3.5, Figure 11).

## Stable-then-Decay recipe (§4, Tables 2–5)

| Stage | Tokens | Steps | LR (1B / 3B / 8B) | Context; batch | Data |
|---|---|---|---|---|---|
| Stable | 200B | 50,000 | constant 5e-5 / 2e-5 / 1e-5 | 8,192; 512 | MegaMath-Web-Pro-Max 0.725, DCLM-Baseline 0.10, MegaMath Text Code Block 0.10, MegaMath-QA 0.05, MegaMath-Code 0.0125, MegaMath Trans. Code 0.0125 |
| Decay (per branch) | 20B | 5,000 | cosine 5e-5→5e-6 / 2e-5→2e-6 / 1e-5→1e-6 | 8,192; 512 | all branches: DCLM 0.05, instruction following 0.10, MegaMath-Web-Pro 0.55; Long: OpenR1 0.15 + AM-DeepSeek-Distilled-40M 0.15; Short: MegaMath-QA 0.025 + OpenMathInstruct2 0.175 + NuminaMath1.5 0.10; Hybrid: OpenMathInstruct2 0.10 + NuminaMath1.5 0.10 + OpenR1 0.10 |

- Pilot (10B tokens on OctoThinker-3B-Base-Stable): QA share 10%, 20%, 30%, 40%; gains plateau beyond 30%; 30% adopted. QA sources derived from GSM8K/MATH-style datasets transfer better than web-sourced MegaMath-QA (§4.2.1, Figure 17).
- Weight decay 0.1, AdamW, warmup 0 in both stages (Tables 3–4).

## Results (§4.3, §5)

- Table 7 (3B, MATH500 4-shot): Llama-3.2-3B 7.40; Stable 22.40; Long 25.80; Hybrid 30.80; Short 31.40. Table 7 GSM8K 8-shot: 30.48; 55.95; 56.10; 64.37; 63.31.
- Figure 1 (MATH500): Llama-3.2-3B-Zero 10.0; OctoThinker-Long-3B-Zero 65.2; Qwen2.5-3B-Base 38.2; Qwen2.5-3B-Zero 66.4 (bar labels; base-model values match Table 7).
- All evaluations are mathematical benchmarks (14 math benchmarks; MMLU-STEM is the only MMLU subset). General-domain capability after mid-training or RL is not reported.

## Used in

ch-32 §5.1, §6 (distilled traces in mid-training), Recipe rows, Generalization lens.
