<!-- scope: the IFD (Instruction-Following Difficulty) metric as a standalone reference
     deps: [[cherry-llm]]
     see-also: [[superfiltering]], [[deita]], [[less]]
-->

# IFD — Instruction-Following Difficulty Score
- **Core Insight:** A single scalar — the ratio of `PPL(response|instruction)` to `PPL(response)` — captures how much conditioning signal a training instruction provides, and it's computable with one forward pass each under the target model.
- **Guideline:** Use IFD as the first-pass filter on any SFT pool; combine with a diversity constraint (embedding or gradient space) for best results.
- **Authors:** Ming Li et al. (see [[cherry-llm]])
- **Year:** 2023 (NAACL 2024)
- **URL:** https://arxiv.org/abs/2308.12032
- **Relevant topics:** SFT filter, perplexity metric, IFD

## Definition (exact)
For a sample `(q, a)` with instruction `q` and response `a`:
- `PPL_cond(a | q) = exp( -1/|a| · Σ_t log p_M(a_t | q, a_{<t}) )`
- `PPL_uncond(a)  = exp( -1/|a| · Σ_t log p_M(a_t | a_{<t}) )`
- **IFD(q, a) = PPL_cond(a | q) / PPL_uncond(a)**

where `M` is the target (warmed) model.

## Interpretation
- `IFD < 1`: instruction `q` reduces response uncertainty — the task is informative for the response.
- `IFD ≈ 1`: instruction irrelevant to response (noisy / decoupled pair).
- `IFD > 1`: conditioning on `q` *hurts* likelihood — likely a distribution-mismatched or pathological sample; drop.
- The **hardest-but-valid** samples cluster at IFD just below 1 — these are the cherry samples.

## Computation recipe
1. Warm the target LM on ~1K random samples for 1 epoch (avoids cold-start mis-calibration).
2. For each candidate `(q, a)`:
   - forward-pass `q ∥ a` and take per-token log-probs on `a` for `PPL_cond`,
   - forward-pass `a` alone and take per-token log-probs for `PPL_uncond`,
   - compute IFD.
3. Sort desc (within the `< 1` band) and keep top-K (5–15%).

## Practical guidance
- Use BF16/FP16; IFD is stable across precision.
- Batch-size 1 works; prefer right-padding for conditional PPL.
- Apply the same tokenizer & system prompt in both forward passes — otherwise the ratio is biased.
- For very long responses, consider length-normalizing separately per PPL.

## Extensions
- **Superfiltering** ([[superfiltering]]): compute IFD using a *smaller* proxy (Qwen-0.5B) — the ranking transfers to the target; orders-of-magnitude cheaper.
- **Reflective IFD** variants weight IFD by response length or token entropy.
- **Reverse-IFD** (some follow-ups) — `PPL(q|a) / PPL(q)` — for dialog data where prompts are long.

## Connections
- Central component of [[cherry-llm]].
- Cheap first stage when layered with [[deita]]'s quality+diversity or [[less]]'s gradient-similarity.
- Does not measure factual correctness; combine with an answer verifier for math/code.
- For synthetic data specifically, IFD is useful to prune "teacher hallucinated an instruction that doesn't match its own response" cases — a common failure mode in [[self-instruct]]-style pipelines.
