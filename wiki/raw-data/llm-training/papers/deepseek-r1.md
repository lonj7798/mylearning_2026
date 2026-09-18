<!-- scope: redirect — DeepSeek-R1 (arXiv:2501.12948) is carded in full at model-reports/deepseek-r1.md
     deps: [[deepseek-v3]]
     see-also: [[deepseek-r1-recipe]], [[grpo]], [[rlvr-tulu3]], [[entropy-mechanism-llm-rl]]
-->

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **Core Insight:** GRPO training on DeepSeek-V3-Base with only rule-based accuracy and format rewards, and no SFT, raised AIME 2024 pass@1 from 15.6% to 77.9% (DeepSeek-R1-Zero, arXiv v2 §2.3).
- **Guideline:** See the full card; recommendations and their evidence are kept in one place.
- **Authors:** DeepSeek-AI (core contributors Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, et al.)
- **Year:** 2025 (arXiv v1 2025-01; v2 2026-01; Nature 645, 633–638, published 2025-09 under the title "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning")
- **URL:** https://arxiv.org/abs/2501.12948
- **Source type:** official technical report
- **Relevant topics:** GRPO, rule-based rewards, R1-Zero, cold-start SFT, distillation

> Duplicate of [[deepseek-r1]] (full card: `model-reports/deepseek-r1.md`; recipe ledger: [[deepseek-r1-recipe]]). This file is kept so existing links resolve.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2501.12948 (v2, 2026-01-04; v1 2025-01-22).
- Corrections to the previous card version (the full text now lives in the canonical card):
  - "G = 16–64 rollouts per prompt" → 16 outputs per question in training (§2.1, §3.2.1); 64 is the evaluation sample count for AIME and GPQA (Supp. D.1).
  - "KL-to-reference term ... applied per token on reward" → GRPO adds the KL estimator directly to the loss, not to the reward (Eq. 1–2, Supp. A.3); coefficient 0.001 (§2.1).
  - "Format reward: +1 iff output follows `<think>…</think><answer>…</answer>`" and "r_format ∈ {0,1}" → the format reward requires reasoning inside think tags and is added to the accuracy reward with equal weight (§2.2, Eq. 4); no numeric value is given for it.
  - "Accuracy reward: ... regex match for instruction following" → the paper describes answer matching for math and compiler test cases for code (§2.2).
  - "Language-mixing fix: add a language consistency reward in later RL stages" → it is introduced in the first RL stage of R1 and kept in the second (§3.2.1 Eq. 7, §3.2.2 Eq. 8).
  - "Fig. 2 ... ~15% to ~70%", "Fig. 3", "Fig. 4 aha moment" → v2 Fig. 1(a) (15.6% → 77.9%), Fig. 1(b), Table 2; v1 Fig. 2 reports 71.0%.
  - "'Aha moment' ... a phase transition" → the paper reports a sudden increase in use of "wait": nearly absent early, occasional at steps 4000–7000, spikes after step 8000 (Supp. C.2).
  - "Cold-start SFT (R1 only): ~thousands of hand-cleaned long-CoT examples" → "thousands" of examples built from R1-Zero outputs filtered by rules, refined by DeepSeek-V3, and verified by human annotators (Supp. B.3.2).
  - "Distillation into Qwen-7B / Llama-8B" → six students on Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct (Supp. B.4.3, Table 6).
  - "Year: 2025 (Nature 645, 633–638)" → the Nature version has a different title (above) and was published 2025-09.
- Removed as unsupported by the source: "long rollout budgets (≥8k tokens)"; "average response length grows from ~400 tokens ... to 10k+ tokens"; "the only regularizer preventing entropy collapse"; "no Goodhart drift ... only remaining exploit surfaces are bugs in the grader"; "reward hacking of the format tag via hollow `<think>` blocks"; "repetition loops" as an R1-Zero failure mode (the paper reports repetition for 7B dense and 16B MoE bases, Supp. G.1, and for greedy decoding at evaluation, Supp. D.1).
