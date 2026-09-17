---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/rest-em.md
source_url: https://arxiv.org/abs/2312.06585
created_at: "2026-04-23"
revised_at: "2026-09-15"
---

# Excerpt: Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (ReST-EM)

**Source library:** `wiki/raw-data/llm-training/papers/rest-em.md` (verified 2026-09-14 against arXiv:2312.06585v4)
**Revision note:** rewritten in the 2026-09 revision. The earlier version of this excerpt quoted numbers that are not in the paper (K = 32 at temperature 1.0 with top-p 0.95, a cap of 4, learning rate 1e-5, batch 128, "iteration 1 +8%, iteration 2 +6%", MATH 34.1% → 50.6%, APPS 16.4% → 31.2%). The values below come from the verified card.

- **Authors:** Avi Singh, John D. Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, et al. (Google DeepMind; R. Agarwal also Mila)
- **Year:** 2023 (arXiv v1 2023-12-12; TMLR 04/2024)
- **Used in:** ch-31 §1.1, §1.2, §1.5, §4, §5.4, Recipe, Generalization lens

## Objective (§3, Algorithm 1)

ReST-EM is expectation–maximization for RL with a binary optimality variable O, p(O = 1 | x, y) ∝ f(r(x, y)). The Generate (E) step samples outputs from the current policy and scores them. The Improve (M) step maximizes J(θ) = E_{(x,y)∼D_i}[ r(x, y) log p_θ(y | x) ] while reward improves on a validation set. With r ∈ {0, 1}, incorrect samples have zero weight (§3 Remark). Each Improve step fine-tunes the base pretrained model rather than the previous iterate, to limit task-specific overfitting (§3).

## Settings (§5 Implementation Details)

| Setting | Value |
|---|---|
| Models | PaLM 2-S (Bison), PaLM 2-S* (Codey), PaLM 2-L (Unicorn); parameter counts not reported |
| Training problems | MATH 7,500; APPS (Introductory) 2,342 |
| Samples per problem | MATH 32; APPS 64 |
| Sampling | top-K with K = 40; temperature 0.7 |
| Kept per problem | at most 10 correct solutions, to limit over-representation of easy problems |
| Loss | next-token loss on model solutions only; input = few-shot prompt + question |
| Iterations run | MATH 3; APPS 2 |
| Learning rate, batch, epochs, compute | not reported |
| Evaluation decoding | greedy (Figs. 2–3); pass@K at T = 1.0, top-p 0.95 (Fig. 5); majority voting over 64 samples (§5.2) |

## Results with loci

- **Versus human data.** Fine-tuning on self-generated, reward-filtered data beats fine-tuning on human solutions for PaLM 2-S, PaLM 2-S*, and PaLM 2-L on MATH and APPS (§5.1, Figs. 2–3).
- **Majority voting.** PaLM 2-L after ReST-EM reaches 48.82% on MATH test with 64-sample majority voting vs 44.02% for the base model (§5.2).
- **Iterations vs samples.** PaLM 2-L on MATH: one iteration with 3× samples per problem gives 40.3% pass@1, below 41.0% at iteration 2 and 41.9% at iteration 3 (§5.3).
- **Overfitting.** Training accuracy rises with iterations while test accuracy does not: small MATH test gains after iteration 1; APPS regresses at iteration 2, which the authors attribute to overfitting on a problem set about one third the size of MATH (§5.1, Fig. 4). More iterations also regress HumanEval transfer for the APPS model (§5.1).
- **Restart from base.** Compared with continuing from the previous iterate, restarting gives comparable APPS performance and better HumanEval transfer (PaLM 2-S*; §3, Fig. 7).
- **Transfer.** Held-out evaluations: GSM8K, HumanEval, the 2023 Hungarian high-school finals exam, and Big-Bench Hard. MATH- and APPS-trained PaLM 2-L show no major degradation on BBH (§5.4, Fig. 9).
- **pass@K.** ReST-EM exceeds the base model at all K in Fig. 5, with the largest gap at K = 1; the authors state it may not close the gap at large K (§5.2, §6).
- **Dataset size.** One iteration on 1,000 MATH questions gives gains that the authors call "significant" (§5.3, Fig. 8 left); 4,000 questions scored slightly below 2,000, attributed to single-run fine-tuning variance.
- **Difficulty.** Easy, medium, hard, and very hard bins (base success at T = 1.0 of 75–100%, 50–75%, 25–50%, < 25%) all improve; medium and hard gain most (§5.3, Fig. 8 right).
- **Distillation.** PaLM 2-S fine-tuned on PaLM 2-L solutions beats PaLM 2-S fine-tuned on its own ReST-EM data, attributed to more questions having solutions (§5.3, Fig. 6 right).
- **Rationalization.** In preliminary experiments, STaR-style rationalization increased false positives (correct final answer, incorrect reasoning), so ReST-EM does not use it (§4).

## Connections

- [[star]]: greedy decoding, one solution per problem, rationalization; ReST-EM uses temperature sampling and no rationalization (§4, Table 1).
- [[v-star]]: trains a DPO verifier on the incorrect samples that ReST-EM discards.
- [[raft-reinforce-rej-minimalist]]: RAFT is an instance of the same {0,1}-reward objective; the paper compares it with GRPO.
