---
chapter: ch-40
course: llm-training
phase: read
excerpt_of: arXiv:2503.20783v2 (Understanding R1-Zero-Like Training: A Critical Perspective); library card [[dr-grpo]] (verified 2026-09-14)
source_url: https://arxiv.org/abs/2503.20783
created_at: "2026-04-23"
revised: "2026-09-15 (generality revision; aligned with the verified card and the v2 text)"
---

# Excerpt: Dr. GRPO — two terms removed, and what was measured

Used by [[read]] §5, §6, and the Recipe. Liu et al. (Sea AI Lab, NUS, SMU); arXiv v1 Mar 2025, v2 6 Oct 2025 (COLM 2025).

## The two biases (§3.1)
- **Response-length bias.** For `Â > 0`, the `1/|o_i|` factor gives shorter responses larger per-token updates. For `Â < 0`, a longer incorrect response is penalized less per token, so the policy comes to prefer long incorrect responses.
- **Question-difficulty bias.** Questions whose rewards are nearly all 1 or all 0 have a small `std(R)` and therefore receive larger weight after the division.

The paper also documents that length-normalized losses are present in trl, OpenRLHF, verl, SimpleRL-Zero, and Open-Reasoner-Zero (Table 2, Listing 1).

## Dr. GRPO (§3.2, Fig. 1, Listing 1)
Same clipped objective without `1/|o_i|`, and `Ã_{i,t} = R(q, o_i) − mean(R)`. In code, `masked_mean` becomes `(tensor * mask).sum(axis=-1) / MAX_TOKENS`, where MAX_TOKENS is the maximum number of generation tokens; "other constants also work with differences in gradient norm". The objective itself contains no length divisor — the constant lives in the implementation. App. A shows `(G/(G−1)) · Ã` equals the RLOO advantage.

## Setting and hyperparameters (§3, App. G Table 6)
Outcome reward 1 if the response contains the correct final answer, else 0, checked with Math-Verify. `β = 0` throughout: both KL coefficients are 0.0. 8 responses per question; temperature 1.0 (top-p 1.0, top-k −1); max response 3000 tokens; clip ε 0.2; one inner proximal epoch; AdamW(0.9, 0.95), weight decay 0.0, gradient clip 1.0, constant LR 1e-6; all runs on 8× A100. Questions per policy step and total policy steps are not reported.

## Results
- Qwen2.5-1.5B with the R1 template on MATH questions: under Dr. GRPO output length stops growing, and the benchmark length of incorrect responses is lower than under GRPO (Fig. 5, curves only).
- Ablation at 1.5B on a 3K mixed question set: removing either bias term raised reward and accuracy over vanilla GRPO; removing length normalization has the larger effect on length (App. C Figs. 8–9, Fig. 9 over 3 runs).
- Oat-Zero-7B (from Qwen2.5-Math-7B, Dr. GRPO, MATH level 3–5 prompts, Qwen-Math template, 27 h on 8× A100), 3k-token budget: AIME24 43.3, AMC 62.7, MATH500 80.0, Minerva 30.1, OlympiadBench 41.0, average 51.4 (Table 4). Oat-Zero-1.5B average 42.1; Llama-3.2-3B + Dr. GRPO average 6.8, and 20.7 after math continual pretraining (NuminaQA).

## Related findings the chapter uses
- Templates decide whether a base model answers or continues text; Qwen2.5-Math base models score highest with no template (§2.1–2.2, Table 1), so gains attributed to RL can be recovery of capability a template hid (§3.3).
- With the Qwen-Math template on Qwen2.5-Math-1.5B, RL on GSM-8K (simpler, out-of-distribution questions) gave the best final average (§3.3, Fig. 6).
- In DeepSeek-R1-Zero outputs, the average string length of incorrect responses (8206.1) exceeds that of correct ones (4965.4); the authors attribute this to harder questions (App. F, Table 5, Interpretation).

## Claims removed from the earlier version of this excerpt
"Dr. GRPO keeps the k3 KL term" (β = 0 throughout), "~30% shorter completions", "L_max e.g. 4096" (the runs use 3000), "dropping /std is strictly an improvement" (the paper reports comparable or better accuracy with better token efficiency, and [[gigpo-verl-agent]] §5.2 finds the choice task-dependent).
