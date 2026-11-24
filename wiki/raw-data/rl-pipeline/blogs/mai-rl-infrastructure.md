<!-- scope: Microsoft AI "RL infrastructure at Microsoft AI" (Nov 2026): Rocket async RL framework — controller/problem/rollout workers, policy staleness control, MXFP8 sampler precision, bitwise sampler–learner parity, fault tolerance, elasticity, Table 1 savings
     deps: [[grpo]], [[async-rollout]]
     see-also: [[mai-thinking-1]], [[thinking-machines-nondeterminism]], [[flash-attention-stability]], [[nemo-rl-tp-accuracy]], [[deepseek-train-infer-consistency]], [[truncated-importance-sampling]], [[fp16-train-infer-mismatch]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[deepseek-open-infra]], [[rstar2-agent]]
-->

# RL infrastructure at Microsoft AI (Rocket)
- **Core Insight:** When RL climbs from a base model for weeks (no distilled head start), small sampler–learner numerical mismatches and system inefficiencies compound. The infrastructure must therefore bound **policy staleness**, achieve **bitwise sampler–learner parity**, and **recover at the smallest scope** instead of restarting the job (p. 1–2).
- **Guideline:** Keep the job inference-bound (overallocate learner GPUs), prioritize in-flight rollouts over new ones, drop failed rollouts instead of retrying, make the learner reproduce the sampler's quantization (fake-quantize what the sampler stores in MXFP8), use the same reduction order and attention tiling in sampler and learner, and separate supervision from the RL loop so each component can be repaired or resized alone.
- **Authors:** The Microsoft AI Team (no individual bylines; citation block p. 14).
- **Year:** 2026 (citation: "Microsoft AI, Nov 2026", p. 14).
- **URL:** https://microsoft.ai/news/rl-infrastructure-at-microsoft-ai/ (PDF: https://microsoft.ai/pdf/RL-Infrastructure-at-Microsoft-AI.pdf, 15 pages)
- **Source type:** lab engineering blog
- **Relevant topics:** asynchronous RL, policy staleness, GRPO group scheduling, weight transfer, low-precision rollout (MXFP8), train–inference numerical parity, fault tolerance, elasticity, agentic rollouts

## Summary
Rocket is MAI's in-house RL framework. The [[mai-thinking-1]] report introduced it as the framework that turned MAI Base 1 into MAI Thinking 1. It was then extended to post-train MAI Base 2, a "three-trillion-parameter base model", "climbing from the base model through RL without distillation from third-party models"; this work "is separate from the MAI Code 2 models" (p. 1). Runs last "many thousands of training steps spanning weeks to months on thousands of GPUs" (p. 1). The post names three challenges (p. 1): (1) staleness, because agentic rollouts take hours and GRPO waits for the whole group; (2) numerical consistency between a lower-precision sampler and the learner; (3) fault tolerance and elasticity. MAI owns the learner, Rocket, and an inference engine "built on an MAI-maintained fork of SGLang"; it reuses Kubernetes or Slurm for node allocation and runs components as Ray actors (p. 2). verl and slime are called "strong foundations", but long climbs from base "push these capabilities beyond the regime they are typically exercised in" (p. 2). The argument for the design: a distilled model starts RL near its ceiling, so a failed run can be restarted and mismatches "have little time to compound"; climbing from base removes both escapes (p. 1).

## Key Contributions
- Async controller / problem-worker / rollout-worker / learner loop; batches carry segment references and advantages, not token data (Fig. 1, p. 3).
- A staleness decomposition and four levers: frequent weight sync, rollout-age priority, drop-failed, inference-bound sizing (Fig. 4, p. 6–7).
- Selective MXFP8 for MoE weights and global-attention KV, with matching fake quantization in the learner (Fig. 5, p. 8–9).
- Bitwise sampler–learner parity "for identical checkpoints and inputs, with no noticeable performance degradation" (p. 10).
- Smallest-scope recovery, in-job recovery, CPU-only control plane, topology-aware elasticity, typed runtime config, with savings in Table 1 (p. 11–13).

## Key Figures/Tables to Study
- **Fig. 1 (p. 3)** architecture. CPU pools: Controller (select problems, admit work, filter + pack), Problem workers (request executor, group grading, advantages), Rollout workers (agent loop, tool dispatcher, per-rollout graders); Router (CPU) + SGLang (GPU) prefill/decode; Rollout store (tokens/logprobs, model-turn payloads); GPU-pool Learner; Environments (SEE, external tool services); Recovery checkpoints (controller + learner state); Runtime management.
- **Fig. 2 (p. 4)** median rollout duration / turns, 1.13M rollouts from one day of a production run: instruction following 2 min / 1; visual Q&A 2 min / 1; math & science 7 min / 1; tool use 12 min / 13; competitive coding 34 min / 1; web research 51 min / 89; web development 58 min / 104; software engineering 93 min / 249.
- **Fig. 3 (p. 5)** mean |log-prob difference| (y-axis 0–0.05) vs staleness (x-axis 0–40), one point per rollout, linear fit R² = 0.76; staleness 0 gives "Same weights: exact match".
- **Fig. 4 (p. 6)** one GRPO group: weights from learner steps 104–124, sampler refresh every 4 steps; each rollout waits 24 min before its first inference request (PDF p. 7 prose); slowest rollout starts on step-104 weights, 784 turns; fastest waits 7.6 h and median 5.6 h for the slowest; all three trained together in step 127 (3 min) after 12 s in the controller queue.
- **Fig. 5 (p. 8)** precision map (MXFP8 storage / BF16 storage-compute / FP32 accumulation). **Fig. 6 (p. 9)** in-house math benchmark avg pass@1 (≈20–60% axis) vs checkpoint iteration 20–2,000 (log), bitwise-matched above baseline. **Fig. 7 (p. 12)** two-layer scheduling.
- **Fig. 8 (p. 14)** y-axis labelled "Aggregate agentic benchmark (avg pass@1, %)" (the prose says "aggregate reasoning benchmark"), steps 40–2,000 (log), log-linear fit R² = 0.974; reference lines Claude Opus 4.8 = 66.0, GPT-5.6 Sol = 73.3.

**Table 1 (p. 11)**, "a portion of RL training, assuming 6,000 GPUs":

| Event type | Saved per event | Time saved (h) | % | GPU-hours saved |
|---|---|---|---|---|
| In-job recovery (n = 21) | 0.5 h | 10.5 | 0.42% | 63,000 |
| Learner recovery (n = 88) | 2 h | 176 | 7.07% | 1,056,000 |
| Runtime elasticity (n = 123) | 2 h | 246 | 9.88% | 1,476,000 |
| Overall training time | — | 2,490 | 100.00% | 14,940,000 |
| Total estimated savings | — | 432.5 | 17.37% | 2,595,000 |

## Technical Details
**Loop (Fig. 1 numbering, p. 2–4).** (1) Controller picks a problem from the curriculum → (2) problem worker dispatches the group's rollout requests → (3) rollout workers run agent loops: (4) sampler calls, (5) tool calls to environments (SEE, search, MCP tools); per-rollout graders score the trajectory; payload goes to the rollout store, scores + references go back to (2) → (2) runs group graders and computes GRPO advantages → (1) packs *training segments* (one example, one or more model turns) into batches of references + advantages → (6) learner fetches token IDs and sampling log-probs from the store. The controller periodically triggers weight transfer.

**Staleness (p. 4–7).** Definition: version gap between the policy that generated a token and the learner when it trains on it. Approximation: staleness ≈ sampler lag + (t_trained − t_generated) / (time between training steps). Levers:
- *Sampler lag*: two-stage weight transfer (selected sampler engines pull packed chunks from the learner, then distribute to matching ranks across replicas), so learner-side cost does not grow with fleet size; transfer takes 15 s "in our current configuration" (p. 7).
- *Priority*: each request inherits a priority from its rollout's start time; in-flight rollout count auto-scales to inference capacity. Waiting before the first token adds no staleness, but rollout timeouts must allow for it (p. 7).
- *Drop failed rollouts*: avoids retry gaps; cost is noise or bias in the GRPO baseline because failures "may share features that never make their way to the learner" (p. 7).
- *Inference-bound*: learner overallocated; cost is partly idle learner GPUs (p. 7).
- Counter-intuitive: adding inference servers at fixed batch size can *increase* staleness, because steps complete faster while a long rollout's wall-clock time is unchanged (p. 7).

**Precision (p. 7–9).** Thinking 1 kept BF16 weights and KV caches in the sampler. MAI Base 2 stores MoE experts (>95% of parameters) in MXFP8, which "nearly halves" sampler weight memory and allows a smaller EP degree; experts are dequantized to BF16 for the matmul because MXFP8 matmuls gave larger log-prob gaps. The learner quantizes MoE weights to MXFP8 and immediately dequantizes to BF16. Attention is local:global = 5:1; quantizing both KV types gave "substantially larger" gaps, so local KV and the final three global layers stay BF16, other global layers use MXFP8, and the learner applies KV fake-quantization. Result: "approximately 67% more global-context tokens" at roughly the same local capacity.

**Bitwise parity (p. 9–10).** A later RL checkpoint showed a larger gap than initialization on identical inputs, suggesting "optimization may amplify or exploit numerical differences". Fixes: (1) sampler keeps attention TP but replicates o_proj weights and all-gathers head outputs, so no cross-GPU reduction follows the projection; learner runs without TP (cites [[thinking-machines-nondeterminism]], [[nemo-rl-tp-accuracy]]); (2) MoE: EP without intra-expert TP, all-to-all back to the source device, routing-weighted sum locally in FP32; (3) one shared FA4 implementation for learner forward, sampler prefill and decode; sliding-window K-blocks aligned to learner tiles; KV-tokens-per-split tuned once and fixed for all three paths (cites [[flash-attention-stability]]). Evidence (Fig. 6): L42 (42 layers, 7B active / 180B total), two runs from the same checkpoint launched roughly one month apart; the bitwise-matched run scores higher at comparable steps.

**Fault tolerance (p. 11–12).** Health checks treat stalled processes as unhealthy. Learner recovery: pause admission, kill learner ranks without waiting for NCCL timeouts, rebuild, restore the latest checkpoint, reconnect to the live controller; "roughly five minutes"; inference keeps running and accepts the staleness cost. Inference recovery replaces one replica; below a live-fraction threshold the job restarts at cluster level. Escalation: in-job rebuild within the Ray job, then cluster requeue. Ray heads and CPU actors run on CPU-only nodes, "more stable than GPU nodes in our deployments".

**Elasticity (p. 12–13).** Learner, inference and controller scale independently; inference resizes immediately, the learner only after the next checkpoint save. Placement is topology-aware on rack-scale Blackwell; a spare node group sized to one learner slice is kept. Runtime settings are typed and validated (inference capacity and concurrency, admission, staleness limits, batch packing, mixture weights) and applied at step or sampling boundaries.

**Outcome (p. 13–14).** MAI's model "recently passed Claude Opus 4.8" on the aggregate benchmark and is "on track to overtake GPT-5.6 Sol in a few weeks"; the climb continues "without plateauing or destabilizing".

## Verification
First extracted from the HTML on 2026-11-24. Cross-checked against the 15-page PDF on 2026-11-24 (`pdftotext -layout`). All prose numbers in the earlier version matched the PDF; no numeric value was wrong. Added: Table 1 (all cells; per-row % and GPU-hours re-computed as hours × 6,000 and hours / 2,490, consistent), Fig. 2/3/4/8 values from the figure text layer, page loci. Fixed: pipeline steps now follow Fig. 1's (1)–(6) numbering; Fig. 6 wording now says "launched roughly one month apart" (PDF) instead of "baseline launched about 1 month earlier"; noted that Fig. 8's axis label ("agentic") differs from the prose ("reasoning"). Figure graphics themselves (curve shapes) were not read beyond their text layers.
