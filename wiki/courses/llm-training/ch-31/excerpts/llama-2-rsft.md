---
chapter: ch-31
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-2.md
source_url: https://arxiv.org/abs/2307.09288
created_at: "2026-04-23"
---

# Excerpt: Llama-2 RSFT appendix — the iterative-SFT recipe that anchors ch-31

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-2.md`
**Artifact:** the V1..V5 iterative-RLHF schedule and the RSFT-only portion (V1..V3)

---

## Why this source anchors ch-31

Every modern iterative post-training pipeline in the open record traces back to the Llama-2 §3 appendix. The specific move — "for each prompt, sample K outputs, score with combined RMs, SFT on the best sample, no policy-gradient" — is the primitive ch-31 builds on. Llama-3's six-round recipe is the same loop, three rounds wider, with DPO replacing PPO. Tulu-3's SFT-DPO-RLVR is the same loop re-staged. DeepSeek's cold-start-plus-RL is this loop contracted to one round. Without understanding what Llama-2 put in V1..V3 and why it worked without PPO, the rest of the chapter is a vocabulary list.

---

## The attested recipe — quoted verbatim from the raw-data source

From the source (lines 51–57):

> **V1..V3:** Rejection-Sampling Fine-Tuning (RSFT). For each prompt, sample K outputs (K ~ 10+), score with combined RMs, SFT on the best sample. No policy-gradient.
>
> **V4, V5:** PPO added on top of RSFT checkpoint.
>   - **Learning rate:** 1e-6 (policy) for 70B.
>   - **KL coefficient beta:** 0.01.
>   - **Batch size:** 512.
>   - **Sequence length:** 4K.
>   - Standard PPO with clipped ratio, value function, GAE.

And from the source (lines 19–23):

> - Iterative RLHF: five successive checkpoints (V1..V5) with fresh weekly batches of human preferences.
> - Rejection-Sampling Fine-Tuning (RSFT): sample K completions per prompt, keep best by RM score, SFT on the top sample. Used for V1..V3.
> - PPO used only in V4 and V5 on top of the RSFT checkpoints.

Ch-31 quotes this block as the canonical RSFT recipe in §1.

---

## What ch-31 keeps, changes, drops

| Llama-2 default | Ch-31 choice | Reason |
|-----------------|--------------|--------|
| K ~ 10 | Default K=10; slider up to 64 in the HTML companion | 10 is the attested anchor; below K=4 selection is too noisy, above K=64 reward overoptimization from [[best-of-n]] Figure 4 dominates |
| top-1 of K | Default top-1; ch-31's `top_frac` slider allows top-K/4 per [[llama-3]] | top-1 maximizes per-prompt quality; top-K/4 uses more budget and tolerates a noisier RM |
| Dual RMs (helpfulness + safety) | Ch-31 assumes one verifier or one RM for clarity; discusses dual-RM composition in §7 | dual-RM composition is orthogonal to the iteration pattern itself |
| PPO in V4, V5 | Ch-31 treats DPO per [[llama-3]] as the standard finishing move | DPO matches or beats PPO at fixed data per [[iterative-sft-rl]]; Llama-3 confirms this |
| 5 rounds total | Ch-31 decision tree defaults to 2-3 RSFT rounds then switch | the [[rest-em]] 2-iter saturation curve is tighter evidence than Llama-2's 5-round plateau |
| Weekly fresh preference batches | Noted as a failure-mode mitigation (stale-RM drift) in §7 | the "retrain RM each round" practice generalizes to [[llama-3]] and Tulu-3 |

---

## The three hinges Llama-2 gets right

1. **No policy-gradient in V1..V3.** Most of the AlpacaEval win-rate gain comes from SFT-on-self-samples, not PPO. PPO is a finishing move, not the foundation. This is the Llama-2 finding that ch-31's decision tree node 6 leans on directly ("0-2 rounds -> do another RSFT").

2. **Combined RMs with piecewise scoring.** A single scalar RM collapsing both helpfulness and safety loses the piecewise ability to let safety dominate on safety prompts and helpfulness dominate elsewhere. Ch-31 does not inherit the dual-RM complexity but flags it as the real-world requirement whenever safety is a separate objective.

3. **Fresh preference data each week.** The RM drifts as the policy drifts. If you keep training the policy against an old RM, the policy exploits the RM's training-distribution blind spots. Llama-2's weekly refresh is what keeps the iteration honest. Ch-31 §7 (failure mode 5) names this explicitly.

---

## The one number to carry into ch-31

**K = 10, top-1.** That is the attested Llama-2 default. It is the number every downstream recipe either keeps ([[rest-em]] at K=32 because the verifier is exact-match and cheaper, so K can grow) or widens ([[llama-3]] at K=10-30 with top-K/4 because they have better RMs and want more samples per round). Ch-31's `rsft_round` default argument `k=10` is chosen to match this exactly.

---

## Connections

- **ch-30** — the SFT axes (chat template, loss masking, packing) apply per round of RSFT.
- **ch-33** — the case-study chapter will unpack the full Llama-2 -> Llama-3 progression.
- [[llama-3]] — six-round descendant; DPO + NLL replaces PPO.
- [[iterative-sft-rl]] — synthesis of Llama-2 and Tulu-3 pipelines.
- [[best-of-n]] — the theoretical ceiling (BoN-KL = log K - (K-1)/K) RSFT can approach but not exceed within a round.
