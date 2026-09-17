<!-- scope: Qi et al. (arXiv:2406.05946): evidence that safety alignment changes mainly the first few output tokens, the prefilling / fine-tuning attacks this explains, a data-augmentation defense, and a token-wise constrained fine-tuning objective
     see-also: [[finetuning-compromises-safety]], [[xstest]], [[circuit-breakers-data]]
-->

# Safety Alignment Should Be Made More Than Just a Few Tokens Deep

Chapter excerpt for ch-52 (read.md). Written 2026-09-15 from the cached primary text of arXiv:2406.05946v1 (10 Jun 2024). No library card existed for this slug at the time of writing; every number below carries its locus in the paper.

- **Core Insight:** Safety alignment in current chat models is concentrated on the first few output tokens: prefilling the refusal prefix "I apologize, but I cannot" lowers the harmfulness rate of the *unaligned* Llama-2-7B base model on HEx-PHI from 68.6% to 2.1% (Table 1), and six gradient steps of fine-tuning on 100 harmful pairs raise Llama-2-7B-Chat's attack success rate from 1.5% to 87.9% (Fig. 3).
- **Guideline:** When alignment must survive prefilling, decoding-parameter and fine-tuning attacks, train on augmented targets that recover a refusal after a harmful prefix of k tokens and constrain updates on the earliest token positions, because the augmented model's ASR under a 40-token prefill falls from 57.0% to 4.5% (Table 2) and a first-token constraint of β₁ = 0.5 holds ASR at 4.6% under an attack that takes the unconstrained model to 88.9% (Table 3).
- **Authors:** Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson (Princeton University; Google DeepMind)
- **Year:** 2024 (arXiv v1 2024-06)
- **URL:** https://arxiv.org/abs/2406.05946 (code: github.com/Unispac/shallow-vs-deep-alignment)
- **Source type:** paper

## Technical details used by ch-52

- **Benchmark.** HEx-PHI: 330 harmful instructions over 11 prohibited-use categories (§2.1). ASR is the fraction of outputs judged harmful; the system-prompt field is left empty for the fine-tuning-attack evaluations because that generally raises ASR (App. A.4).
- **Refusal-prefix prefilling (Table 1).** Harmfulness rate with no prefix / with "I apologize, but I cannot": Llama-2-7B base 68.6% → 2.1%; Gemma-7B base 85.4% → 1.0%. The corresponding aligned models are at 0% either way.
- **Per-token KL (§2.2, Fig. 1).** KL divergence between the aligned model and its base counterpart on harmful responses is much larger for the first few tokens than for later ones.
- **Non-refusal prefilling (Fig. 2, Table 2).** ASR on Llama-2-7B-Chat with the first k tokens of a harmful response prefilled: 42.1% (k = 5), 51.5% (10), 56.1% (20), 57.0% (40).
- **Fine-tuning dynamics (Fig. 3).** Llama-2-7B-Chat fine-tuned on 100 (harmful instruction, harmful answer) pairs, learning rate 2e-5, batch size 64: ASR 1.5% initially, 22.4% after 2 gradient steps, 76.4% after 4, 87.9% after 6. Per-token loss and gradient norms are largest at the earliest positions.
- **Data augmentation (§3.1).** Training targets of the form p(refusal | x, harmful prefix h≤k) with k ~ Uniform[1, C], mixed with utility data. Augmented-model ASR (Table 2): prefilling 2.8 / 2.9 / 3.4 / 4.5% at k = 5 / 10 / 20 / 40; GCG 36.5% → 18.4% on HEx-PHI and 65.6% → 19.0% on AdvBench; decoding-parameter exploit 54.9% → 11.3% on HEx-PHI and 84.3% → 1.0% on MaliciousInstruct.
- **Constrained fine-tuning objective (§4, Eq. 3, Table 3).** A token-wise objective with a larger constraint β₁ = 0.5 on initial tokens. Under the harmful-examples attack, ASR is 1.5% (initial), 88.9% (standard fine-tuning), 4.6% (constrained). Under backdoor poisoning, ASR with the trigger is 90.9% (standard) against 10.9% (constrained). Table 4 shows a uniform small β gives both worse safety and worse utility.

## Limits stated by the source

- The augmented model is still vulnerable to adversarial fine-tuning on harmful data, though at lower ASR than the initial model (§3.2, App. C).
- The augmentation is applied to an already aligned model rather than used to align from scratch; an end-to-end pipeline is left to future work (App. A.3).
- Experiments are on Llama-2-7B-Chat and Gemma-7B, English, single-turn.

## Verification

- Read on 2026-09-15 from the cached primary text of arXiv:2406.05946v1 (body plus App. A–C).
- Not reported by the source: results for models above 7B; multilingual or multi-turn settings; the constant C used in the augmentation sampling.
