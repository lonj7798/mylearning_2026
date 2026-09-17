---
chapter: ch-03
course: llm-training
phase: read
excerpt_of: lr-schedules (composite) — Vaswani et al. 2017; Hu et al. 2024 (MiniCPM); Hägele et al. 2024; DeepSeek-AI 2024 (DeepSeek LLM); Llama Team 2024 (Llama 3); OLMo et al. 2025 (OLMo 2)
source_url: https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/2404.06395 ; https://arxiv.org/abs/2405.18392 ; https://arxiv.org/abs/2401.02954 ; https://arxiv.org/abs/2407.21783 ; https://arxiv.org/abs/2501.00656
created_at: "2026-04-23"
revised_at: "2026-09-15"
note: "Rewritten for the 2026-09 revision. The library card classics/lr-schedules.md was not yet re-verified on 2026-09-15 and contains unsupported numbers (a Chinchilla '0.3-1% perplexity' horizon penalty, '1-3% worse' constant LR, a 70B peak LR of 1.2e-4, a Llama 3 SFT LR of 2e-5, and WSD attributed to DeepSeek). This excerpt keeps only statements read in the primary sources at the loci given."
---

# Excerpt: learning-rate schedule formulas and where each is printed

## Inverse square root with warmup (Vaswani et al. 2017, arXiv:1706.03762 §5.3, Eq. 3)

`lrate = d_model^-0.5 · min(step_num^-0.5, step_num · warmup_steps^-1.5)`, with `warmup_steps = 4000`.
The paper states this "corresponds to increasing the learning rate linearly for the first warmup_steps
training steps, and decreasing it thereafter proportionally to the inverse square root of the step number."

Check by hand (derived): d_model = 512 and step 4000 give 512^-0.5 · 4000^-0.5 = 0.0442 · 0.0158 ≈ 6.99e-4;
step 16000 gives ≈ 3.49e-4, half of the peak.

## Warmup-Stable-Decay (Hu et al. 2024, arXiv:2404.06395v3 §4.2, Eq. 1)

`WSD(T; s) = (s/W)·η for s < W;  η for W < s < T;  f(s − T)·η for T < s < S`, where `0 < f ≤ 1` is decreasing and
`η` is the maximum LR. §4.3: loss "experiences a significant rapid decline" in the decay stage and reaches or
passes the cosine loss at T = S (Fig. 5, 0.036B model); among checkpoints at 40N, 60N, 80N tokens, a decay of 10%
of total tokens is "sufficient to achieve the best results, while a decay of 2.5% of total tokens falls short".
MiniCPM itself uses exponential decay `f(s − T) = 0.5^((s−S)/T)` "in which T is set to be 5000 steps (20B tokens)"
(§6.2; notation as printed).

## Constant LR plus cooldown (Hägele et al. 2024, arXiv:2405.18392v3 §3, Eq. 1; §3.2)

Same three-part form; the (1-sqrt) cooldown is `f = 1 − sqrt((n − (N − N_decay)) / N_decay)`. A 20% cooldown matches
a length-matched cosine; (1-sqrt) beats linear; the cooldown optimum peak LR is about half the cosine optimum
(Fig. 3 caption). Details and numbers: [[cooldown-scaling-beyond-fixed-durations]].

## Multi-step (DeepSeek LLM, arXiv:2401.02954 §2.3)

"the learning rate of the model reaches its maximum value after 2000 warmup steps, and then decreases to 31.6% of
the maximum value after processing 80% of the training tokens. It further reduces to 10% of the maximum value
after 90% of the tokens." Final performance is "essentially consistent" with cosine for a 1.6B model on 100B
tokens (Fig. 1a); the 80/10/10 split was chosen to balance reuse in continual training against performance (§2.3).

## Cosine with a truncated horizon (OLMo 2, arXiv:2501.00656 Table 3, §2.3, §4.1)

Warmup 0 → peak over 2000 steps, "followed by a cosine decay calibrated to reach 10% of the peak learning rate
after a specified max tokens" (§2.3). Table 3: horizon 5T tokens (7B, 13B), 6.5T (32B); truncation "after 4T"
(7B) and "after 6T" (32B). §4.1: "the last part of a cosine decay schedule can be cut off and replaced by a linear
decay to zero with little loss of performance" (from OLMo-0424 experience). Released config
`OLMo2-7B-stage1.yaml`: `scheduler: cosine_with_warmup, units: tokens, t_warmup: 8388608000, t_max: 5e12,
alpha_f: 0.1`; `OLMo2-7B-stage2-seed42.yaml`: `learning_rate: 0.000061499`, `linear_with_warmup, t_warmup: 0,
alpha_f: 0`, `max_duration: 50e9T`, loaded from `step928646`.

## Cosine to 1% plus a linear anneal (Llama 3, arXiv:2407.21783v3 §3.4.1, §3.4.3)

405B: "AdamW with a peak learning rate of 8 × 10−5, a linear warm up of 8,000 steps, and a cosine learning rate
schedule decaying to 8 × 10−7 over 1,200,000 steps." Annealing: "During pre-training on the final 40M tokens, we
linearly annealed the learning rate to 0, maintaining a context length of 128K tokens", with high-quality sources
upsampled and Polyak averaging of checkpoints. Recipe rows: [[llama-3-recipe]].

## Not supported by these sources

A fixed percentage penalty for a mismatched cosine horizon; a fixed percentage penalty for constant LR without
decay; "WSD (DeepSeek)". MiniCPM §4.1 (Fig. 4, 0.036B) reports only that Cosine(T) with T = S gave the lowest loss
at S = 20N, 40N, 60N, 80N, and that both T < S and T > S were worse.
