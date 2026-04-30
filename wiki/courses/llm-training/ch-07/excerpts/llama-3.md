---
chapter: ch-07
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/model-reports/llama-3.md
source_url: https://arxiv.org/abs/2407.21783
created_at: "2026-04-23"
---

# Excerpt: Llama 3 — scale, iteration, and the DPO NLL stabilizer

**Source library:** `wiki/raw-data/llm-training/model-reports/llama-3.md`
**Paper:** Grattafiori et al. 2024, *"The Llama 3 Herd of Models"* (Meta).

---

## Why this source anchors ch-07

Llama 3's 405B run is the largest publicly documented frontier training exercise in the ch-07 era: 15.6 T tokens, 3.8e25 FLOPs, 128K context window. At that scale every ch-07 failure mode fires multiple times per week by probability alone; the interesting engineering is the decisions that *keep the run alive* despite the near-certainty of individual-component failure. Ch-07 §5's NCCL hang branch uses Llama 3 as the canonical "infrastructure failures at scale" example — if a rank's MTBF is months, a 1024-rank cluster sees one failure every few days, and the training loop must be engineered around that reality.

The source states scale:

> *"Llama 3 is pre-trained on 15.6T tokens and post-trained via six rounds of SFT + Rejection Sampling + DPO. ... 405B compute: 3.8e25 FLOPs."*

Six iterative post-training rounds, each a separate checkpoint fork with its own evaluation gate — the ch-06 flywheel pattern applied at frontier scale.

---

## The DPO NLL stabilizer — ch-07 §6's chosen-logprob-collapse fix

The source's most ch-07-relevant technical contribution is the DPO auxiliary loss:

> *"DPO (the policy optimization step): Learning rate: 1e-5. Beta (KL coefficient): 0.1. Auxiliary NLL loss on chosen sequences: coefficient 0.2 — added to stabilize training by preventing chosen-logprob decay."*

Notice the word "decay." In standard DPO, the loss `L_DPO = -log σ(β (log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l))))` is minimized by increasing `log π(y_w)` *relative to* `log π(y_l)`. An optimizer can reduce the DPO loss by either raising `log π(y_w)` or lowering `log π(y_l)` — and empirically, DPO tends to do the latter more than the former, because decreasing an already-low-probability log-prob is "easier" than raising an already-high one. The result is chosen-logprob *collapse*: both `π(y_w)` and `π(y_l)` go down, but `π(y_l)` goes down faster. The model becomes less confident in its preferred outputs, which is a silent-failure mode — DPO loss looks healthy, chat quality degrades.

The NLL add-on (`0.2 · (- log π(y_w))`) pins the chosen log-prob in absolute terms, not just in ratio. Empirically, Llama 3's choice of `0.2` prevents the decay without dominating the preference signal — a single-hyperparameter fix for a silent-failure mode.

Ch-07 §6's RL section names this failure under "entropy-collapse triage," but the DPO instance is distinctive: DPO has no explicit entropy penalty, no explicit KL-to-reference in the loss (`β` controls the strength of the preference term, not the KL ball radius). The NLL stabilizer is DPO's implicit equivalent of entropy-regularization, preventing the chosen distribution from collapsing.

---

## Iterative post-training — six rounds, six fork points

From the source:

> *"Six rounds of: (a) Reward Model (RM) training, (b) Rejection Sampling to build the round's SFT pool, (c) SFT on curated pool, (d) DPO on preference data collected with the latest RM-ranked generations."*

Each round is a separate ch-06 checkpoint fork. Every ch-07 failure mode can fire at each of the six handoffs:

| Round | Fork artifact | Ch-07 failure risk |
|---|---|---|
| 1 → 2 | r1_dpo → r2_rm starting point | RM init bug drops to chance-level; entropy collapse on §6 |
| 2 → 3 | r2_dpo → r3_sft starting point | §4 masking bug in the new SFT pool; dead-pipeline §3 |
| 3 → 4 | r3_dpo → r4_rm | RM overfitting to the round-3 distribution; reward hacking |
| ... | ... | ... |

At six rounds, each with four phases (RM, RSFT, SFT, DPO), there are 24 distinct handoffs where a silent failure can ship downstream to the next round. Llama 3's operational discipline is that each handoff has a gate — the source mentions "topic classifier + quality classifier" filtering the rejection-sampled pool, and the NLL-stabilized DPO catches chosen-logprob collapse before round-(k+1) starts.

---

## Preference-data noise — documented silent failures

The source is unusually candid about failure modes:

> *"Full disclosure of failure modes (preference-data noise, multi-turn dialog drift) in the data section."*

**Preference-data noise** is the silent failure that feeds DPO. Annotators disagree on preference labels; the RM trains on noisy labels; DPO optimizes against an RM whose own confidence is mis-calibrated. Llama 3's fix: the margin labels ("significantly better / better / slightly better / negligibly better") are used for *up-weighting* high-confidence annotations in the preference training, not as direct optimization targets. A run without this up-weighting silently trains on low-confidence preferences with the same weight as high-confidence ones — a distribution shift that no ch-07 §2 diagnostic catches.

**Multi-turn dialog drift** is the ch-07 §4a multi-turn masking bug at scale. In a long dialog where the model's own earlier turns enter the prompt, a subtle bug in the assistant-turn masking on training data propagates into inference: the model learns patterns that conflate user and assistant turns. The source:

> *"Most-recent-batch preference data only (older batches cause format drift)."*

"Format drift" is the symptom — the chat template at round 5 differs slightly from round 3 in some subtle way (maybe a new thinking-tag convention, maybe a tokenizer update), and using round-3 preference data on a round-5 model produces gradients that shift the model's output-template distribution in a way that breaks inference.

---

## Scale implications for ch-07 §5 hangs

The source gives enough to estimate hang frequency:

> *"Pretraining: 15.6T tokens, 8K native context, 8-way sequence parallel for long-context extension."*

At 15.6T tokens and 4K sequence length, that's ~3.8 billion sequences processed. At ~1-second per distributed step and thousands of GPUs, the pretraining takes weeks of wall time. An H100 has MTBF measured in months; a cluster of 1024+ H100s sees a hardware failure every few hours. The NCCL timeout (default 30 minutes) applied to this rate would cost the entire cluster 30 minutes per failure — a huge fraction of available compute.

Llama 3's infrastructure discipline (not reported in this source but documented in Meta's engineering blog and reflected in the ch-07 §5 recommendation) includes tightened NCCL timeouts, aggressive per-rank heartbeat monitoring, and automated rank-replacement on failure detection. The ch-07 §5 check `rank_heartbeat_delta > 60s → alarm` is the practitioner form of this.

---

## SFT hyperparameters — the ch-07 baseline

From the source:

> *"SFT: Training: LR 1e-5 (405B), cosine decay, context 8K-32K (extended), loss on response tokens only."*

"Loss on response tokens only" — Llama 3 explicitly adopts the [[excerpts/loss-masking-prompt]] canonical practice. No off-by-one, no packed-block mask variant — the paper names the invariant directly. The learning rate scales down as size goes up: 1e-5 at 405B is an order below the small-model SFT default of 1e-4. This follows the general rule that SFT LR at frontier scale is conservative to avoid ch-07 §2 divergence in the face of noisy synthetic data.

The "Data mix (per round)" section:

> *"~50–80% synthetic rejection-sampled data. Remainder: human SFT demonstrations, preference data, capability-specific synthetic (code-exec-filtered code, math with verifier, multi-turn tool use traces, long-context QA)."*

Heavy synthetic SFT means the §3 dead-pipeline risk is high: if the rejection-sampling pipeline emits a run of low-diversity or truncated completions, the SFT pool silently degrades. Llama 3's "topic classifier + quality classifier (both distilled from Llama 3)" is a downstream filter that catches pipeline degradation; the ch-07 §3 `active_tokens_per_batch` assertion is its per-step equivalent.

---

## Rejection sampling — the ch-07 §1c RL connection

From the source:

> *"Rejection sampling: for each prompt, sample K=10–30 completions from the best round-(N-1) chat model at temperature T=0.6–1.0, then keep the top by RM score."*

K=10–30 per prompt is the Llama 3 choice. The §1c /0-in-advantage failure mode has a rejection-sampling cousin: if all K completions for a prompt earn the same RM score (which happens when the RM is confident on easy prompts), the filter's effective diversity is 1, the selected completion is identical across rollouts, and the SFT data becomes repetitive. The mitigation is to log RM-score-variance-per-prompt and drop low-variance prompts from the pool — the ch-07 §6 GRPO "drop all-same-reward groups" discipline applied to rejection sampling.

---

## What to take from Llama 3 for ch-07

1. **DPO can silently collapse the chosen log-prob.** Llama 3's 0.2·NLL auxiliary loss pins it; the fix is a single hyperparameter.
2. **Iterative post-training multiplies handoff surface.** Six rounds × four phases = 24 checkpoint forks; each is a ch-07 failure-mode opportunity.
3. **Preference-data noise is a documented silent-failure class.** Margin labels and most-recent-batch-only discipline are the operational fixes.
4. **Multi-turn dialog drift is the §4a bug at scale.** Chat-template invariance across rounds is a separate eng discipline.
5. **Frontier-scale hangs are probabilistic.** Tight NCCL timeout + per-rank heartbeat is the ch-07 §5 response.

---

## Connections

- [[excerpts/loss-masking-prompt]] — "loss on response tokens only" is the response-only rule this source names.
- [[excerpts/gradient-clipping]] — the three-layer spike-mitigation stack applies identically at Llama-3 scale.
- [[excerpts/fsdp-sft]] — 8-way sequence parallel for long-context extension is an FSDP-axis split beyond data-parallelism.
- [[excerpts/olmo-2]] — parallel production story; Llama 3 swaps RLVR for DPO, OLMo 2 keeps the Tulu-3 RLVR path.
- [[excerpts/openrlhf-entropy-debugging]] — DPO is a different RL loss than PPO but shares the collapse-surface; the NLL stabilizer is a DPO-specific triage tool.
- [[ch-07]] — §5 (NCCL hangs at scale), §6 (chosen-logprob collapse in DPO), §4 (multi-turn mask drift), §3 (synthetic-SFT dead-pipeline risk).
