---
chapter: ch-49
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/direct-judgement-preference.md
source_url: https://arxiv.org/abs/2409.14664
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Direct Judgement Preference Optimization (SFR-Judge)

**Authors:** Peifeng Wang, Austin Xu, Yilun Zhou, Caiming Xiong, Shafiq Joty (Salesforce AI Research)
**Year:** 2024 (arXiv v1 2024-09; v3 2025-09)
**Source type:** paper
**Checked on:** 2026-09-15 against the library card, which was re-verified on 2026-09-14 against arXiv v3.

> Rewritten in the 2026-09 revision. The previous version of this excerpt described four different papers
> under this slug and attributed methods to the wrong ones. The corrections are listed below; the library
> card `papers/direct-judgement-preference.md` is the verified reference for the rest.

## Corrections to the previous version of this excerpt

1. "Con-J: DPO on contrastive judgment pairs with noisy-negative instruction perturbation" → the modified-instruction construction belongs to [[self-taught-evaluators]] (arXiv:2408.02666 §3.3). Con-J (Ye et al., arXiv:2410.03742 §3) pairs a judgement carrying the correct preference against one carrying the wrong or no preference, drawn from existing preference data, with DPO plus a small SFT weight. It does not remove the need for preference labels.
2. "Self-Taught Evaluators: judge_{k+1} trained with DPO on its own decisions; crosses GPT-4 after ~3 iterations" → SFT (NLL on judgment tokens) on rejection-sampled correct judgments, 5 iterations; GPT-4-0125 (84.3) is first exceeded at iteration 2 (86.0).
3. "J1: the verdict-token log-prob supplies the verifier signal; highest-accuracy open judge on RewardBench-hard" → J1 (Whitehouse et al., arXiv:2505.10320 §2.1, §2.3) is online GRPO training of judges on 22K synthetic pairs (17K WildChat, 5K MATH), with verifiable rewards against known verdicts plus consistency rewards. "RewardBench-hard" is not an existing benchmark.
4. "~40K synthetic preference pairs (20K SFT + 20K DPO) beat models trained with 2–40× more data" and the "judge-as-weapon" leakage quotation → not supported by the source; removed.

## What this paper actually is

SFR-Judge-8B / 12B / 70B are generative judges trained from Llama-3.1-8B-Instruct, NeMo-Instruct-12B, and Llama-3.1-70B-Instruct on 680K judgement preference pairs (§4.1).

**Three pair types (§3):**
- `D_CoT` (70%): a teacher (Llama-3.1-70B-Instruct, 20 samples per prompt, temperature 0.7) writes critique-plus-verdict samples; samples whose verdict matches the annotation are chosen, the rest rejected.
- `D_Std` (15%): the same pairs with the critique removed and the protocol asking for the verdict only. Motivation (§3.2, Fig. 3): in a CoT critique only a few tokens decide the verdict, so critique-length targets dilute the signal on those tokens.
- `D_Ded` (15%): given the protocol, the input, and a correct critique-plus-verdict, reconstruct the original responses; chosen = the originals, rejected = a reconstruction by the weaker Llama-3.1-8B-Instruct.

**Loss (§3.4):**

```
L = − log M_s(y_w | x) / (|y_w| + |x|)
    − log σ( β log[ M_s(y_w|x) / M_ref(y_w|x) ] − β log[ M_s(y_l|x) / M_ref(y_l|x) ] )
```

`M_s` is the judge being trained, `M_ref` a frozen copy of the same initialization, `x = (p, i, r)` the protocol, task input and responses, `y = {c, j}` a critique and verdict, `|·|` token length, `σ` the logistic function, `β` the DPO temperature (not reported for the main runs).

## Results

- Pairwise average over 7 benchmarks: 84.25 (70B), 81.49 (12B), 80.91 (8B); GPT-4o 76.78; Skywork-Critic-Llama-3.1-70B 80.03; Self-taught-evaluator-Llama-3.1-70B 82.26; Con-J-7B 75.51 (Table 1).
- RewardBench: 92.7 / 90.3 / 88.7; GPT-4o-2024-08-06 86.7 (Table 7).
- Single rating, average Pearson: 0.76 / 0.70 / 0.68; GPT-4o 0.75 (Table 2). Classification: 85.60 / 84.12 / 85.41; GPT-4o 85.47 (Table 3).
- Removing the critique at inference lowers 8B single-rating Pearson from 0.68 to 0.58 and the pairwise average from 80.97 to 80.05 (App. E.6, Table 9).
- Hard versus easy negatives at 8B (App. E.8, Table 11): negatives from the 70B teacher vs the 8B teacher give pairwise accuracy 78.83 vs 77.56 and pairwise consistency 85.94 vs 80.70.
- Specialization: continual fine-tuning the 8B judge on 12,500 RAGTruth pairs at β = 0.01 reaches 55.6% on ContextualJudgeBench (§5.5, Fig. 6).

## Measurement convention to note

For the six non-RewardBench pairwise benchmarks, each is run twice with the response order swapped and **the better of the two runs is reported** (§4.3). Order-averaged accuracy is not reported; consistency is given separately (App. E.1, Table 6).

## Connections

[[self-taught-evaluators]] (the method whose construction was previously misattributed here), [[generative-reward-models]] (concurrent generative-judge line), [[meta-rewarding-lm]] (source of the 5-point additive scoring prompt used in its §5.4 experiment), [[rewardbench]] (main evaluation).
