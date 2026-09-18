<!-- scope: redirect — DeepSeekMath (arXiv:2402.03300) is carded in full at [[grpo]]
     deps: [[grpo]]
     see-also: [[grpo-recipe]]
-->

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, et al. (DeepSeek-AI, Tsinghua University, Peking University)
- **Year:** 2024 (arXiv v1 2024-02; v3 2024-04-27)
- **URL:** https://arxiv.org/abs/2402.03300
- **Source type:** paper

> Duplicate of [[grpo]]. This file is kept so existing links resolve.

The full card (corpus, continued pre-training, SFT, GRPO, generality findings) is [[grpo]]; the recipe ledger is
[[grpo-recipe]].

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2402.03300 (arXiv v3, 2024-04-27), body and App. A.1.
- Corrections to the previous version of this file (content moved to [[grpo]] in corrected form):
  - "MATH pass@1: 51.7% (base model)" → 51.7% is DeepSeekMath-RL 7B; DeepSeekMath-Base 7B scores 36.2% (Table 2, Table 5).
  - "60.9% with self-consistency (64 samples)" for the base model → the abstract attributes 60.9% to "DeepSeekMath 7B" without naming the variant.
  - "SFT 46.8 → RFT 49.0 → online RFT 49.4 → DPO 49.0 → PPO 51.0 → GRPO 51.7 on MATH" → no such table exists; 46.8% and 51.7% are DeepSeekMath-Instruct 7B and DeepSeekMath-RL 7B (Table 5); the method comparison is Figure 5 (1.3B; RFT, Online RFT, GRPO+OS, GRPO+PS; no numeric values in text).
  - "Unified comparison of SFT, RFT, DPO, online RFT, PPO, GRPO on the same base" → the six methods are compared analytically (Table 10, App. A.1); experiments cover RFT, Online RFT, and GRPO variants (Figure 5).
  - "Training batch size 1024 (16 prompts x 64 completions)" → "training batch size is 1024", unit not stated (§4.2).
  - "Max generation length: 1024 tokens" → "max length is set to 1024", unit not stated (§4.2).
  - "RL stage: prompts drawn from the GSM8K + MATH training set" → about 144K chain-of-thought-format questions related to GSM8K and MATH, taken from the SFT data (§4.2).
  - "SFT: 776K problems w/ CoT" → 776K examples in chain-of-thought, program-of-thought, and tool-integrated formats (§3.1).
  - "Reward model: a 7B RM trained on math preference data; also supports rule-based outcome reward" → reward model initialized from DeepSeekMath-Base 7B, LR 2e-5, training set built following Wang et al. (2023b) (§4.2); a rule-based reward for the released RL run is not stated.
  - "Equation 20 (or similar): the full GRPO objective" → the objective is Eq. 3 (§4.1.1); Eq. 19-21 are the simplified form and gradient (App. A.1.6).
  - Audit finding: the previous version omitted §5.1 (code training benefits math reasoning; arXiv papers seem ineffective) and Table 4 (MMLU 49.1% → 54.9%, BBH 55.2% → 59.5%); both are now in [[grpo]].
- Removed as unsupported: "Clip ratio eps: 0.2"; "halving the memory footprint"; "empirically matches or beats PPO"; "value network is a persistent source of bias"; "[[deepseek-v3]] uses GRPO in its own SFT+RL stage" (not stated in this paper).
