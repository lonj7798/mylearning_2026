---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: arXiv:2402.06457v2 (V-STaR: Training Verifiers for Self-Taught Reasoners); library card [[v-star]]
source_url: https://arxiv.org/abs/2402.06457
created_at: "2026-09-15"
revised: "2026-09-15 (generality revision; written from the primary source because the library card has no Verification section)"
---

# Excerpt: V-STaR verifier training on the generator's own correct and incorrect solutions

Used by [[read]] Guideline, the negatives section, the Generalization lens, and Common mistakes. COLM 2024; checked against arXiv v2 (2024-08-14) on 2026-09-15.

Note on the library card: it lists the authors as Zelikman, Wu, Mu, and Goodman. The title page lists Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal (Mila, Microsoft Research, University of Edinburgh, Google DeepMind).

## Algorithm (§3)
1. Fine-tune a pretrained LLM on the original training data to get G_SFT.
2. Sample k completions per training problem from the current generator; label each correct or incorrect by ground-truth answers or test cases.
3. Add only correct solutions to the generator data D_GEN; add correct and incorrect solutions with labels to the verifier data D_VER.
4. Next iteration: fine-tune the pretrained model on the augmented D_GEN and sample again; repeat for T iterations.
5. The final verifier is trained from G_SFT on D_VER.
- Difference from Cobbe et al.'s ORM: verifier data are collected iteratively from better generators, while ORM data come from a fixed generator fine-tuned only on the original SFT data (§3).

## DPO verifier (§3.1, Eq. 2)
- Preference pairs are the Cartesian product of correct and incorrect solutions per problem.
- `L = −E log σ(r̂(x,y+) − r̂(x,y−))`, `r̂(x,y) = β log V(y|x)/G_SFT(y|x)`; at inference the verifier's likelihood `V(ŷ|x)` ranks candidates.
- "The DPO objective steers the verifier towards increasing the likelihood of correct solutions y+ and decreasing the likelihood of incorrect solutions y−."
- DPO verifiers were better than ORM-style verifiers with LoRA adapters (§3.1, §4.4).

## Results
- 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on code generation and math reasoning with LLaMA2 models (Abstract).
- Three iterations with 16 samples per query per iteration; Best-of-64 computed on 128 candidates per problem (Fig. 3 caption). A 4th MBPP iteration gave a marginal gain of 0.3% (§4).
- Transfer evaluated from GSM8K to a MATH subset and from MBPP to HumanEval (Fig. 4).
