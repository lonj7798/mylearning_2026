<!-- chapter: ch-31
     track: sft
     kind: content
     title: Iterative SFT-RL Bridges
     deps: [ch-30]
     sources: [[rejection-sampling-finetuning]], [[best-of-n]], [[star]], [[rest-em]], [[v-star]], [[iterative-sft-rl]], [[self-rewarding-lm]], [[meta-rewarding-lm]], [[spin]], [[self-play-preference]], [[llama-2]], [[llama-3]], [[interplay-pretraining-midtraining-rl]]
     figures: figures/iterative-loop.html
-->

# Chapter 31 — Iterative SFT-RL Bridges

> **Core insight.** The line between SFT and RL is not a wall; it is a *thin strip* called rejection-sampling fine-tuning, and the strongest post-training pipelines in the open record (Llama 2, Llama 3, Tulu 3, DeepSeek-R1's descendants) spend most of their time inside that strip. A single SFT-then-RL run buys you one turn of the flywheel; multiple rounds — each round's RL producing better on-policy samples for the next round's SFT, each round's SFT resetting a clean reference for the next round's RL — buy you the actual gains the reports attribute to "RLHF." If you have a verifier or a reward model, your first question is not "PPO or DPO?" It is "how many rounds of RSFT do I do before I ever write the PPO rollout loop?"
>
> **Guideline.** Budget post-training as `N` rounds of `(sample K per prompt -> score -> keep top-k -> SFT -> optional on-policy RL step -> reset reference)`. Llama-2 attests 5 rounds RSFT-then-PPO; Llama-3 attests 6 rounds SFT + rejection sampling + DPO; [[rest-em]] attests saturation at 2 iters without diversity controls and 3 with. Default K=10 per [[llama-2]], K=10-30 per [[llama-3]], K=32 per [[rest-em]]. Never skip the reference reset between rounds ([[iterative-sft-rl]] attests this is what keeps DPO/RLVR from being dragged backward to a stale SFT reference). The decision tree at the end of this chapter is the entire thing in one flowchart.

---

## Why this chapter exists

Chapter 30 treats SFT as a single pass. In production pipelines it is not. Every flagship post-2023 chat model the open record covers (Llama 2, Llama 3, Tulu 3, Qwen 2.5-Instruct, DeepSeek-V2/V3 Chat) post-trains in *rounds*, and the canonical unit of a round is not a PPO rollout — it is a rejection-sampling fine-tune. Chapters 30 -> 31 -> 32 form one logical unit: ch-30 names the design axes you fix *within* a round; ch-31 names the loop *between* rounds; ch-32 will stack those loops into a full mid-training-plus-post-training pipeline.

The mistake you are being inoculated against: treating RL as the "real" training and SFT as warmup. [[interplay-pretraining-midtraining-rl]] 2025 makes the causal case cleanly — RL creates new capability only when the base model still has headroom and the RL prompts sit at the edge of competence. Every other case, RL is reshaping probability mass on things the model could already occasionally produce. Iterative SFT on filtered self-samples is the cheapest, most stable way to do that reshaping.

---

## 1. RSFT — the Llama-2 appendix recipe, quoted

[[llama-2]] introduced Rejection-Sampling Fine-Tuning (RSFT) as the first three of five RLHF checkpoints. The recipe, quoted verbatim from the attested raw-data notes on [[llama-2]] (§Technical Details — Post-Training Pipeline, RLHF algorithms):

> **V1..V3:** Rejection-Sampling Fine-Tuning (RSFT). For each prompt, sample K outputs (K ~ 10+), score with combined RMs, SFT on the best sample. No policy-gradient.
>
> **V4, V5:** PPO added on top of RSFT checkpoint.
>   - **Learning rate:** 1e-6 (policy) for 70B.
>   - **KL coefficient beta:** 0.01.
>   - **Batch size:** 512.
>   - **Sequence length:** 4K.
>   - Standard PPO with clipped ratio, value function, GAE.

Three things in that quote do load-bearing work:

1. **K ~ 10+.** You need enough candidates that the top-1 by reward is meaningfully better than the median. Llama-2 uses K=10; [[llama-3]] widens to K=10-30; [[rest-em]] uses K=32 for MATH. Below K=4 the selection signal is too noisy; above K=64 the reward-model overoptimization problem from [[best-of-n]] Figure 4 starts dominating (the RM's ranking becomes a worse approximation of true quality the deeper you mine the tail).
2. **Combined RMs.** [[llama-2]] trains two — helpfulness and safety — and composes them with a piecewise rule (safety dominates on safety prompts, helpfulness elsewhere). A single scalar RM combining both objectives loses that piecewise ability and you eat either refusal-rate regressions or helpfulness regressions.
3. **No policy-gradient in V1..V3.** This is the core claim. For the first three iterations, Llama-2 gets most of its AlpacaEval win-rate gain with *SFT-loss only on self-samples* — no PPO, no value function, no KL controller. The PPO in V4/V5 is a finishing move, not the foundation.

### RSFT in 20 lines of real code

```python
# rsft_round.py — one round of RSFT
import torch
from typing import Callable, List

def rsft_round(policy, tokenizer, reward_fn: Callable[[str, str], float],
               prompts: List[str], k: int = 10, temperature: float = 0.8,
               top_frac: float = 1.0 / 10) -> List[tuple[str, str]]:
    """Llama-2 style RSFT: sample K per prompt, keep top-(top_frac*K) by reward."""
    kept: List[tuple[str, str]] = []
    keep_n = max(1, int(round(top_frac * k)))
    for prompt in prompts:
        cands = policy.generate(prompt, num_return_sequences=k,
                                temperature=temperature, do_sample=True,
                                max_new_tokens=1024)
        scored = sorted(
            ((reward_fn(prompt, c), c) for c in cands),
            key=lambda sc: -sc[0],
        )[:keep_n]
        kept.extend((prompt, c) for _, c in scored)
    return kept  # feed directly into next-round SFT loader

def sft_on_kept(policy, tokenizer, kept, lr: float = 2e-5,
                epochs: int = 1) -> None:
    """Standard SFT on response tokens only; prompt masked in loss."""
    # ... (use the same chat-template + loss-mask logic from ch-30)
    ...
```

Llama-2's `keep_n=1` with K=10 is the "top-1 of 10" recipe. [[llama-3]] widens this: `keep_n \approx K/4` with K=10-30. The lesson: top-1 maximizes per-prompt quality but throws 90% of the generation budget away; top-k with k=K/4 uses more of the budget and is more forgiving of a noisy RM.

---

## 2. Best-of-N SFT — the inference-time twin

[[best-of-n]] (Stiennon 2020, the summarization RLHF paper that predated InstructGPT) establishes two facts the whole iterative-SFT family rests on:

- **BoN is monotonic in N up to the overoptimization knee.** At `N \in {4, 16, 64}` with a trustworthy RM, BoN-64 is within 2 points of a well-tuned PPO on human pairwise preference, at roughly 1/10 the engineering cost and zero training instability.
- **BoN has a closed-form KL cost.** `KL(BoN_N || base) = log N - (N - 1) / N`. At N=10 this is ~2.3 - 0.9 = ~1.4 nats. You can directly match a PPO's KL budget to a BoN-N and compare them apples-to-apples.

RSFT is BoN at *training time*: instead of paying the BoN tax at every inference request, you pay it once to generate the SFT dataset and then the policy internalizes the BoN-filtered distribution. The BoN-KL formula is why the RSFT step cannot drift the policy arbitrarily far from the base — each round buys at most `log K - (K-1)/K` nats of KL worth of improvement, which bounds the per-round reward-hacking headroom.

Best-of-N SFT in the literal sense (distilling your own best-of-N back into SFT) is what [[llama-3]] does six times. The only difference from classical RSFT is Llama-3 replaces the PPO finishing move with DPO plus an auxiliary NLL loss on chosen sequences — the DPO loss-with-NLL is a DPO that cannot let the chosen log-prob collapse, which is the failure mode a lot of naive DPO implementations hit in round 3 and beyond.

---

## 3. STaR — the original self-improvement SFT bridge

[[star]] (Zelikman 2022) is the algorithm ch-31 is the direct descendant of. It predates RSFT and it is the cleanest statement of the pattern.

### The STaR loop

1. Prompt the base LM with a small number of rationale exemplars (few-shot).
2. Generate a rationale and final answer for an unlabeled training example.
3. If the answer is correct, keep `(question, rationale, answer)` as synthetic SFT supervision.
4. If the answer is wrong, prompt again while *providing* the correct answer and ask the model to produce a rationale that reaches it (*rationalization backoff*).
5. Keep only rationales that now yield the correct answer.
6. Fine-tune the model on all accepted rationale traces.
7. Repeat from step 2 with the fine-tuned model.

Two things STaR gets right that people still get wrong in 2026:

**The rationalization trick.** A wrong answer is not a discarded sample — it is a candidate for supervision conditioned on the gold answer. This is the first appearance of the "failures become training signal" idea that [[v-star]] later generalizes into a verifier.

**Verifier-first framing.** The bottleneck is not rationale-writing; it is *answer-checking*. Every descendant of STaR — ReST-EM, V-STaR, RLVR, GRPO — is the same loop with a different answer-checker. When you ask "can we do RSFT for creative writing?" the answer is "only as far as your verifier generalizes from verifiable tasks," which is the same as asking "how far does your reward model generalize?"

---

## 4. ReST-EM — STaR cast as expectation-maximization

[[rest-em]] (Singh 2023) reformulates STaR as EM over a latent rationale variable:

- **E-step:** sample K=32 solutions per problem at T=1.0, top-p=0.95; verifier-filter (exact-match or unit-test).
- **M-step:** SFT on survivors; 1 epoch; lr=1e-5; batch 128.
- **Diversity cap:** keep at most 4 distinct correct solutions per problem to prevent memorization of one path.

Two results from ReST-EM that are central to this chapter:

1. **2 iterations saturate.** On MATH with PaLM-2-L: iter-1 +8%, iter-2 +6%, iter-3 flat or regressing. This is the **iterative-SFT saturation curve**. It is the signal you watch for when deciding "should I run another RSFT round, or move to the on-policy RL stage?"
2. **Transfer effect.** Training on MATH improves Big-Bench-Hard tasks not in the training distribution. Iterative SFT on verifier-filtered reasoning is not a narrow distillation — it is pushing the base model's reasoning prior harder, and the prior generalizes.

---

## 5. Iterative-SFT-vs-RL crossover — when more SFT beats the first RL round

The crossover question: given a budget of compute `C`, do you spend it on one more round of RSFT or on one round of PPO/GRPO/DPO? The evidence is mixed by regime, but the pattern is consistent:

| Regime | Iterative SFT dominates | On-policy RL dominates |
|--------|-------------------------|------------------------|
| Verifier is exact-match on reasoning | Rounds 1-2; [[rest-em]] | Round 3+ if headroom remains per [[interplay-pretraining-midtraining-rl]] |
| Reward is a learned RM on chat | Rounds 1-3; [[llama-2]] V1..V3 | Final round for fine finish; [[llama-2]] V4, V5 |
| Judge is policy-as-judge | Rounds 1-3; [[self-rewarding-lm]] | After meta-judge stabilizes; [[meta-rewarding-lm]] |
| Preference data is synthetic or sparse | Rounds 1-3 (SPIN); [[spin]] | Only if preference signal is rich enough to avoid reward hacking |
| Task sits at the base model's edge of competence | First round | Subsequent rounds; [[interplay-pretraining-midtraining-rl]] |
| Task is well inside or well outside the base's competence | All rounds | Never (too easy wastes RL; too hard gives no signal) |

### Iterative vs single-round Llama-3 eval delta (attested)

From [[iterative-sft-rl]] and [[llama-3]]:

| Pipeline | Structure | Reported gain over single-pass |
|----------|-----------|-------------------------------|
| Llama-2 RSFT + PPO, 5 rounds | 3x RSFT then 2x PPO | ~3-5 points AlpacaEval/MT-Bench (attested in [[iterative-sft-rl]]) |
| Llama-3 SFT + RejSample + DPO, 6 rounds | 6x (SFT + rej-sample + DPO) | Multi-round ablation is the headline; single-pass baseline is not released, but [[iterative-sft-rl]] reports the 3-5 point gap on matched-data runs |
| Tulu-3 SFT -> DPO -> RLVR | 3 distinct stages with reference reset | Removing DPO or RLVR costs 3-5 pts averaged across eval ([[iterative-sft-rl]] Tulu-3 Table 2) |
| ReST-EM on PaLM-2-L MATH | 2 rounds, SFT-only | 34.1% -> 50.6% (+16.5 pts vs human-data SFT) |
| ReST-EM on PaLM-2-L APPS | 2 rounds, SFT-only | 16.4% -> 31.2% (+14.8 pts) |

Read this table as one unit. The two most dramatic deltas (+16 on MATH, +14 on APPS) come from *SFT-only* iteration when the verifier is hard (exact-match on correct answer). The Llama-2 / Llama-3 / Tulu-3 deltas are smaller (~3-5 pts) because the reward signal is softer (RM rating of open-ended chat), so each round buys less.

Interpretation: **the quality of your verifier sets the ceiling for iterative-SFT; the quality of your policy-gradient infrastructure sets the ceiling for on-policy RL.** If you have a strong verifier and weak RL infra, iterate SFT. If you have a strong RM/judge and strong RL infra, alternate. If both are weak, you are not ready for this chapter.

---

## 6. The decision tree

Use this as a flowchart at the top of every post-training plan.

```
START: you have an SFT'd base model and want to run one more round.

1. Do you have a verifier that labels >= 50% of your training
   prompts with a Bernoulli correct/incorrect signal?
   YES -> go to 2.  NO -> go to 4.

2. Is the current policy's pass@1 on your verified prompts
   between 0.1 and 0.8 (edge of competence per
   [[interplay-pretraining-midtraining-rl]])?
   YES -> go to 3.
   NO (< 0.1 or > 0.8) -> the prompts are too hard or too easy;
        go curate new prompts before touching the policy.

3. How many verifier-SFT rounds have you already run?
   0-1 rounds -> do another RSFT / ReST-EM round (K=10-32).
   2 rounds with monotone gain -> do one more RSFT round.
   2 rounds with flat or regressing gain -> switch to on-policy
        RL (GRPO / RLVR) with reference = latest SFT checkpoint.
   >= 3 rounds on same data -> your pool is exhausted; curate
        harder prompts or move to the next stage.

4. Do you have a trained reward model or a reliable LLM-as-judge?
   YES -> go to 5.  NO -> you cannot iterate; collect preferences.

5. Is the RM's held-out accuracy above 70% on fresh data?
   YES -> go to 6.  NO -> retrain the RM on more data first;
        iterating on a bad RM amplifies its biases.

6. How many RM-scored RSFT rounds have you run?
   0-2 rounds -> do another RSFT (Llama-2 recipe: K=10, top-1).
   3 rounds -> switch to DPO with reference-reset ([[llama-3]]
        recipe: DPO + NLL aux 0.2, beta 0.1).
   >= 4 rounds on same RM -> train a fresh RM (Llama-2 uses
        weekly fresh batches per [[iterative-sft-rl]]) OR move to
        meta-judge regime ([[meta-rewarding-lm]]) OR stop.

HARD RULE, every path: reset the reference model at each
    transition. Without a reset, DPO/RLVR is dragged backward
    toward a stale SFT reference and the whole chain leaks.
```

---

## 7. Failure modes iterative-SFT introduces

1. **Diversity collapse.** Round-over-round, the policy samples the same correct solution path. [[rest-em]] caps at 4 distinct accepted solutions per problem; [[v-star]] uses a verifier that incorporates diversity. Without a cap, round 3 starts memorizing.
2. **Reference drift.** If you do not reset the reference between DPO/RLVR stages, the KL regularizer drags you toward a stale distribution and the useful gradient disappears. [[iterative-sft-rl]] flags this as the failure mode that cost the Tulu-3 team the most debugging time.
3. **Judge calibration drift (self-rewarding family).** [[self-rewarding-lm]] saturates at 3 iters; [[meta-rewarding-lm]] breaks through with a meta-judge but at 2x compute per round. Without a meta-judge, stop at 3.
4. **Length gaming.** [[meta-rewarding-lm]] Figure 5 shows responses grow 2x over iterations without an explicit length-bias control term. Every iterative-DPO recipe must include length control.
5. **RM/verifier overfitting to the policy.** The RM was trained on outputs from the previous policy. After 3 rounds, the policy's distribution is far enough from the RM's training distribution that RM accuracy silently degrades. [[llama-2]] fixes this with weekly fresh preference batches; [[llama-3]] with RM retraining every round.

---

## 8. Connections

- **ch-30 (SFT design axes)** — every round of RSFT is a single SFT; the axes from ch-30 apply per round.
- **ch-32 (mid-training / cold-start / long-context)** — the iterative loop here is one stage inside the multi-stage pipeline ch-32 builds.
- **ch-33 (Tulu 3, Llama 3 case studies)** — the two canonical case studies of iterative pipelines; ch-33 reads this chapter's algorithms in production.
- **Track 4 (RL)** — GRPO, RLVR, PPO are the on-policy RL half of the alternation. The verifier primitives from [[rest-em]] / [[v-star]] are the reward functions RLVR uses.
- **[[interplay-pretraining-midtraining-rl]]** — the causal case for when alternation pays off vs when it does not.

## Further reading

- [[rejection-sampling-finetuning]] — the pattern definition.
- [[best-of-n]] — Stiennon 2020; BoN-KL formula; the canonical RL-vs-BoN comparison.
- [[star]] — Zelikman 2022; the original rationale-bootstrap loop.
- [[rest-em]] — Singh 2023; STaR as EM; 2-iter saturation.
- [[v-star]] — Zelikman 2024; verifier learns from failures.
- [[iterative-sft-rl]] — synthesis of Llama-2 and Tulu-3 multi-stage pipelines.
- [[self-rewarding-lm]] — actor as its own judge; 3-iter ceiling.
- [[meta-rewarding-lm]] — meta-judge breaks the 3-iter ceiling.
- [[spin]] — SFT-only iterative DPO via human-text-as-chosen.
- [[self-play-preference]] — Nash-MD game-theoretic framing.
- [[llama-2]] — 5-round RSFT + PPO.
- [[llama-3]] — 6-round SFT + rejection sampling + DPO.
- [[interplay-pretraining-midtraining-rl]] — when alternation actually adds capability.

## Companion visualization

**[figures/iterative-loop.html](figures/iterative-loop.html)** — self-contained animated walkthrough of the RSFT -> DPO -> RSFT loop. Sliders for N (rollouts per prompt), reward-threshold (top-fraction to keep), num-rounds; the canvas animates per-round policy-quality curves (illustrative, not fit to any specific run) so you can build intuition for which round the returns start flattening before you commit a cluster to the full recipe. Click a round-label to pin its accepted/rejected histogram.
