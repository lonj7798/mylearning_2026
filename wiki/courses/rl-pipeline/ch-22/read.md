<!-- chapter: ch-22
     track: rl-pipeline
     kind: content
     title: Capstone: Design an RL Pipeline for Your Own Setting
     deps: [ch-05, ch-07, ch-18, ch-21]
     sources: [[mai-rl-infrastructure]], [[mai-thinking-1]], [[slime]], [[verl-rollout]], [[areal-async-rl]], [[rl-design-guide]], [[algorithm-crosswalk]], [[recipe-crosswalk]], [[diversity-axes]], [[subjective-reward-guide]], [[olmo-3-rl-pipeline]], [[async-rollout]], [[areal-agent-pipelines]], [[open-instruct-rlvr-pipeline]], [[tito-token-in-token-out]], [[scalerl]], [[minimax-m1]]
     figures: figures/capstone-worksheet.html
-->

# Chapter 22 — Capstone: Design an RL Pipeline for Your Own Setting

> **Core insight.** Every design decision in this course is conditional on numbers that belong to a specific run: rollout-duration distribution, step time, judge latency, GPU count, failure rate, and which qualities can be verified. Rocket's choices (async loop, 4-step sync, MXFP8 sampler, bitwise parity, learner-only recovery, runtime settings) follow from a three-trillion-parameter model climbing from base for weeks on thousands of GPUs with rollouts from 2 minutes to 93 minutes at the median ([[mai-rl-infrastructure]] p. 1, Fig. 2). A smaller run with different rollouts should reach different answers by the same reasoning. This worksheet asks you to measure or estimate your own numbers first and then derive each decision from them, citing the chapter that gives the mechanism and the source that gives the evidence.
>
> **Guideline.** Fill the worksheet in the order given, because later items depend on earlier ones: staleness sizing (§3) needs the rollout-duration estimate from §1; the grader ladder (§7) needs the task design from §5; the algorithm (§11) needs the reward format from §8. For every answer, write the number it depends on and where the number comes from (your measurement, or a source card with location). Where you cannot measure yet, write the measurement you will run first. Mark each decision as **measured** (you or a source ran a comparison) or **practice** (a lab does it without a comparison), the same labels the library uses ([[rl-design-guide]], [[subjective-reward-guide]]).

---

## Why this chapter matters

The course taught mechanisms one at a time. A real run combines them, and the combinations create constraints that no single chapter states. Three examples from the sources:

- Adding inference capacity to reduce rollout latency can raise staleness at fixed batch size ([[mai-rl-infrastructure]] p. 7; ch-04, ch-21). Sizing and staleness are one decision.
- A judge that needs a whole GRPO group (pairwise or k-way) must run where the group is assembled, on the problem worker, not the rollout worker ([[mai-thinking-1]] Sec. 3.6; ch-02, ch-19). Grader format constrains loop architecture.
- Training style and constraint-following jointly produced a measured seesaw (creativity-only RL: Collie −6.0, IFEval −5.9; [[rubicon-rubric-anchors]] §4.4). Reward design constrains stage order.

**The applied target.** The learner's setting is a sales voice agent built on pipecat, with speech-to-text, the LLM, and text-to-speech each served from the learner's own OpenAI-compatible endpoints. The RL target in this worksheet is the LLM policy that produces the agent's turns and tool calls. A TTS policy is an optional second target (ch-23). The worksheet uses this setting; the items apply to any setting.

**How to use this page.** Each section has: the chapters that teach it; evidence from the sources to reason with (not answers); a decision table with blank cells; and WHY prompts. A blank cell is written `____`. Do not leave a cell filled with a value you cannot justify from a number.

> Interactive: [figures/capstone-worksheet.html](figures/capstone-worksheet.html) — this worksheet as a fillable page with per-section progress, answers saved in the browser, and Markdown export.

---

## §0 Describe the setting

| Field | Your value | Source of the value |
|---|---|---|
| Policy model (size, dense/MoE, context) | ____ | ____ |
| Policy serving engine (vLLM / SGLang / other) and whether it accepts weight updates | ____ | ____ |
| Judge / RM endpoints (model, latency per call) | ____ | ____ |
| User-simulator endpoint(s) | ____ | ____ |
| Tool endpoints (CRM, calendar, catalog) and their latency | ____ | ____ |
| TTS endpoint (if speech is rewarded) and latency | ____ | ____ |
| GPU budget for learner and for inference | ____ | ____ |
| Expected run length (steps, days) | ____ | ____ |

**WHY prompts.**
- Which of these endpoints change weights during RL, and which are frozen services? AReaL serves its search-agent LLM judge from a second SGLang engine whose staleness bound is set to `int(1e12)` because its weights never change ([[areal-agent-pipelines]] `train.py` L127–146). What does that tell you about how a frozen judge endpoint should be treated by the staleness logic?
- slime does not recover external engines ("External engines are not recovered by slime fault tolerance", `external-rollout-engines.md` L103). If your policy endpoint is managed outside the trainer, which component owns its recovery?

---

## §1 Rollout anatomy and duration (feeds §2–§4)

**Teaches:** ch-02, ch-08, ch-10.

**Evidence.** Rocket's median rollout durations (Fig. 2, 1.13M rollouts from one production day): instruction following 2 min / 1 turn; tool use 12 min / 13 turns; web research 51 min / 89 turns; SWE 93 min / 249 turns ([[mai-rl-infrastructure]]). In Rocket's GRPO example the fastest rollout in a group waited 7.6 h for the slowest (Fig. 4). Multi-turn rollouts must keep the engine's token IDs and log-probs and never re-tokenize decoded text ([[tito-token-in-token-out]]).

| Field | Your value |
|---|---|
| Turns per sales conversation (median, p95) | ____ |
| Tool calls per conversation (median, p95) | ____ |
| Per-turn latency: policy + simulator + tools + judge | ____ |
| Estimated rollout wall time (median, p95, max) | ____ |
| Ratio p95 / median | ____ |
| Where the agent loop runs (rollout worker process) and how it calls the policy (token-in/token-out?) | ____ |

**WHY prompts.**
- Why does the ratio of p95 to median rollout time matter more for the sync/async decision than the median itself?
- If your user simulator is an LLM endpoint, what fraction of rollout wall time is simulator latency, and does it scale with policy-fleet size?

---

## §2 Sync vs async

**Teaches:** ch-02, ch-03, ch-10.

**Evidence.**
- verl's mode benchmark, first 150 steps, same four-node budget, Qwen3.5-35B-A3B on DAPO-Math-17k with max response 32,768: `sync` 22.79 h, `colocate_async` 14.72 h (−35.4%), `separate_async` 14.10 h (−38.1%) (`verl/docs/advance/v1_async_trainer.md` L278–292 @8357f1e). One run per mode, math only.
- ScaleRL: asynchronous generation with bounded staleness (PipelineRL, k = 8) raised efficiency, not the asymptote ([[rl-design-guide]] §2 rule 7, citing [[scalerl]] Fig. 4).
- OpenRLHF's README lists synchronous as the default with "better stability" and async as "higher throughput, may affect convergence" ([[async-rollout]]).
- AReaL's default `max_head_offpolicyness` is 0 (synchronous) ([[areal-agent-pipelines]] `cli_args.py` L3304–3311).
- GLM-4.5 ran synchronous colocated slime for general and reasoning RL and asynchronous disaggregated slime for agentic SWE ([[recipe-crosswalk]]).

| Decision | Your choice | Number it depends on | Measured or practice |
|---|---|---|---|
| Sync, colocated async, or disaggregated async | ____ | ____ | ____ |
| If async: what bounds off-policyness (admission-time like AReaL, or training-time drop like MAI-Thinking-1) | ____ | ____ | ____ |

**WHY prompts.**
- Your sales conversations may all be short. Under what rollout-duration distribution is synchronous colocated RL the better choice, and why?
- The verl benchmark measures throughput, not final quality. What additional measurement would you need before claiming async does not hurt your policy?

---

## §3 Staleness budget and sizing

**Teaches:** ch-03, ch-04, ch-05, ch-21.

**Evidence.**
- Staleness ≈ sampler lag + (t_trained − t_generated) / (time between training steps) ([[mai-rl-infrastructure]] p. 4–7).
- Mean |log-prob difference| grows with staleness, linear fit R² = 0.76 (Fig. 3).
- Rocket refreshes the sampler every 4 learner steps; transfer takes 15 s (Fig. 4, p. 7).
- MAI-Thinking-1: 5 gradient steps per inference update; drop rollouts more than 8 updates stale (40 steps) ([[mai-thinking-1]] Sec. 3.1.5).
- AReaL default staleness η = 4 for coding and η = 8 for math ([[areal-async-rl]] §7.1); Olmo 3 max asynchrony 1 (7B Think) and 8 (others) ([[olmo-3-rl-pipeline]] Table 49).
- More inference at fixed batch can increase staleness (p. 7); Rocket stays inference-bound by overallocating the learner (p. 7).

| Field | Your value |
|---|---|
| Learner step time (estimate) | ____ |
| Weight-sync cadence (steps) and transfer time | ____ |
| Expected staleness distribution (median, p95) from §1 durations | ____ |
| Staleness limit and what happens above it (drop / mask / wait) | ____ |
| Inference : learner GPU ratio | ____ |
| Priority rule for in-flight vs new rollouts | ____ |
| Drop or retry failed rollouts | ____ |

**WHY prompts.**
- Compute your expected staleness with the formula using your §1 numbers. Which term dominates, sampler lag or rollout duration?
- Rocket drops failed rollouts and accepts baseline bias; MAI-Thinking-1's Rocket v1 retried them. For a sales simulator whose failures correlate with hard customers, which bias does dropping introduce, and is it acceptable?
- Why would you choose to be inference-bound rather than learner-bound at your scale?

---

## §4 Precision and parity plan

**Teaches:** ch-06, ch-07.

**Evidence.**
- MAI-Thinking-1 kept BF16 in both learner and inference, with MoE routing replay and top-p mask replay; without top-p mask replay, "catastrophic off-policy mismatch, causing divergence within a few training steps" ([[mai-thinking-1]] Sec. 3.1.3, 3.6).
- Rocket moved MoE experts to MXFP8 storage only at 3T scale, with learner fake-quantization (p. 7–9); and achieved bitwise parity by replicating o_proj, local FP32 MoE combine, and a shared FA4 with aligned tiles (p. 9–10). A later RL checkpoint showed a larger gap than initialization on identical inputs (p. 9).
- An FP32 LM head raised train/infer probability correlation from about 0.9x to 0.99x on MiniMax-M1 ([[minimax-m1]] §3.2) and raised ScaleRL's asymptote from 0.52 to 0.61 ([[rl-design-guide]] §2 rule 9).
- Open frameworks offer truncated importance sampling when parity is not achieved (ch-07; open-instruct ρ correction with upper clamp 2.0, [[olmo-3-rl-pipeline]]).

| Decision | Your choice | Evidence / measurement |
|---|---|---|
| Sampler precision (weights, KV) | ____ | ____ |
| Learner precision; LM head precision | ____ | ____ |
| Parity target (bitwise / bounded gap) and how you will measure the gap | ____ | ____ |
| Correction used if parity is not reached (TIS, masking) and its bounds | ____ | ____ |
| Does your serving endpoint return per-token log-probs and token IDs for every sampled token? | ____ | ____ |

**WHY prompts.**
- Why does the train–inference gap matter more for a long run than a short one? Use the Rocket observation about a later checkpoint.
- If your policy endpoint applies top-p, what must the learner replay, and what fails if it does not?

---

## §5 Task and reference-answer design

**Teaches:** ch-09, ch-13.

**Evidence.** Keep items with a uniquely verifiable answer; create references by agreement; keep an item only if some reference-policy rollout reproduces its reference; record per-prompt pass rate at build time ([[rl-design-guide]] §1 rules 1–6). For agentic tasks, define success as a final environment state matching a gold state ([[rl-design-guide]] §1 rule 9, [[tau2-bench]]).

| Task family | Verifiable part (reference) | Non-verifiable part | How the reference is created | Pass-rate window |
|---|---|---|---|---|
| Product and price questions | ____ | ____ | ____ | ____ |
| Objection handling | ____ | ____ | ____ | ____ |
| Booking a meeting (tool calls) | ____ | ____ | ____ | ____ |
| Discovery questions | ____ | ____ | ____ | ____ |
| Disclosure and compliance turns | ____ | ____ | ____ | ____ |

**WHY prompts.**
- Which of these families can be graded by final environment state, and why is that preferable to grading the transcript?
- MAI-Thinking-1 filters prompts twice by pass rate (early window [0.05, 0.8], full-group window [0.1, 0.8]) ([[mai-thinking-1]] Sec. 3.1.5). What happens to the gradient signal if your sales tasks are mostly at pass rate 0 or 1 under your starting policy?

---

## §6 Rollout settings

**Teaches:** ch-10.

**Evidence.** Sweep temperature per model and take the highest value before accuracy drops ([[rl-design-guide]] §2 rule 1); at fixed total batch, 8–32 generations per prompt gave near-identical curves (rule 6); soft overlong punishment (rule 10); train on engine token IDs and log-probs with tool and user turns masked (rule 13). MAI-Thinking-1 used G = 128 with an early exit after 16 ([[mai-thinking-1]] Sec. 3.1.5).

| Setting | Your value | Justification (number + source) |
|---|---|---|
| Temperature / top-p | ____ | ____ |
| Group size G (and early-exit size) | ____ | ____ |
| Max turns and max tokens per turn | ____ | ____ |
| Overlong / timeout handling | ____ | ____ |
| Loss mask for simulator, tool, and system tokens | ____ | ____ |
| User-simulator sampling (personas, temperature, number of simulator models) | ____ | ____ |

**WHY prompts.**
- Why must the simulator's tokens be masked from the policy loss, and what does the policy learn if they are not?
- How does group size interact with a pairwise judge whose cost grows with the number of comparisons per group?

---

## §7 Grader ladder per domain

**Teaches:** ch-11, ch-14, ch-16, ch-19.

**Evidence.** Rules first, a model verifier only on rule-rejected responses (+2.3 average; [[rl-design-guide]] §3 rule 2); probe every LLM judge with master-key responses before the run (rule 5; FPR up to 90.9%); per-prompt rubrics written with a reference (35.9 vs 32.0 vs 23.9 Likert; rule 8); order-swap pairwise judges (rule 11); a speech naturalness reward from a pairwise judge trained on your own human A/B labels, not UTMOS (53.7% agreement) ([[subjective-reward-guide]] §4).

| Quality | Lowest rung that can grade it (rule / execution / reference judge / rubric judge / pairwise GenRM) | Calibration test before the run | Latency per call |
|---|---|---|---|
| Constraint following (length, format, required disclosure) | ____ | ____ | ____ |
| Tool-call correctness and final CRM / calendar state | ____ | ____ | ____ |
| Product and price facts | ____ | ____ | ____ |
| Conversation quality (answers before pitching, turn length) | ____ | ____ | ____ |
| Speech naturalness (if TTS is a target) | ____ | ____ | ____ |
| Safety and compliance | ____ | ____ | ____ |

**WHY prompts.**
- For each row, why is the rung you chose the lowest that works? What would a lower rung miss?
- Which rows need group-level grading (and so must run on the problem worker, ch-02)?

---

## §8 Reward composition

**Teaches:** ch-12, ch-19.

**Evidence.**
- MAI-Thinking-1: `R = R_task + w_lang·R_lang − w_len·R_len` with `R_len = ρ_q·|y| / ℓ_max` so easy problems are penalized more for length; lexicographic combination with instruction-following primary; safety-gated aggregator; style judge applied only after verifiable and safety gates ([[mai-thinking-1]] Sec. 3.1.2, 3.4).
- IF-only rewards lowered AlpacaEval 2 from 33.5 to 21.3 ([[rl-design-guide]] §4 rule 5).
- Per-task advantage normalization across environments: 65.0 vs 59.4 without (rule 6).
- Veto rubrics for user praise and self-praise (practice; [[rubicon-rubric-anchors]] App. A.1).

| Term | Gate or weighted? | Weight or order | Why this form |
|---|---|---|---|
| Compliance / disclosure gate | ____ | ____ | ____ |
| Fact correctness | ____ | ____ | ____ |
| Task outcome (meeting booked, objection resolved) | ____ | ____ | ____ |
| Conversation quality | ____ | ____ | ____ |
| Length / turn-length term | ____ | ____ | ____ |
| Praise / self-praise / engagement-bait veto | ____ | ____ | ____ |

**WHY prompts.**
- Why is a gate better than a weighted sum for a compliance term? Construct a numerical example where a weighted sum rewards a non-compliant reply.
- A sales closer ("Would you like to schedule a call?") is the engagement-bait feature RLMT measured rising under a scalar RM (+0.232; [[rlmt-think-chat]] Table 9). How will your reward distinguish a legitimate closer from a hack?

---

## §9 Diversity axes

**Teaches:** ch-13.

**Evidence** ([[diversity-axes]]). User-simulator diversity is causal in one study: a single frozen GPT-5-mini user gave held-out τ²-Retail 46.1% at peak, returning toward 40.4%; persona-guided 49.2%; an ensemble of three 57.1%; population co-training 62.2% ([[simulator-collapse]] Table 1). τ²-bench pass^1 is near zero above 7 required actions ([[tau2-bench]] Fig. 5; observational). Multi-question prompts: observational evidence only ([[rest-multi-problem-stress]]).

| Axis | How you will measure it in the task set | Target range | Evidence that the axis matters |
|---|---|---|---|
| Customer personas and behavior (simulator diversity) | ____ | ____ | ____ |
| Turns per conversation | ____ | ____ | ____ |
| Tool calls per turn (single / parallel / dependent) | ____ | ____ | ____ |
| Intents per conversation (multi-intent, topic switches) | ____ | ____ | ____ |
| Products and price points | ____ | ____ | ____ |
| System-prompt variants (company, policy) | ____ | ____ | ____ |
| Language and register | ____ | ____ | ____ |

**WHY prompts.**
- Why does training against one frozen simulator lead to the peak-then-return pattern? What is the policy learning about the simulator rather than about customers?
- Which axis has only observational evidence, and what controlled run would establish causality?

---

## §10 Domain stage order

**Teaches:** ch-17.

**Evidence** ([[recipe-crosswalk]]). Staged recipes (Qwen3, GLM-4.5, MiniMax-M1, Nemotron) and joint recipes (Kimi K2, DeepSeek-V3.2) both exist; no lab prints a staged-vs-joint ablation with numbers. DeepSeek-V3.2's stated reason for joint RL: avoid "catastrophic forgetting issues commonly associated with multi-stage training". Nemotron trains all environments together because single-environment training "often results in un-recoverable degradation of other benchmarks". MAI-Thinking-1 trains specialists, consolidates by SFT, then runs a short final RL that keeps "a small proportion of STEM and coding data" ([[mai-thinking-1]] Sec. 3.5). Rubicon: constraint stage before style stage (seesaw evidence).

| Stage | Domains in the stage | Entry criterion | Exit criterion | Retention metric tracked |
|---|---|---|---|---|
| 1 | ____ | ____ | ____ | ____ |
| 2 | ____ | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ | ____ |

**WHY prompts.**
- Is your setting closer to the Rubicon seesaw case or to the joint-mixture case? What measurement would tell you?
- Which capability of the base model must be retained (for example, general instruction following), and which held-out benchmark will show its loss?

---

## §11 Algorithm choice per domain

**Teaches:** ch-18, ch-19; llm-training ch-40, ch-43, ch-45b.

**Evidence** ([[algorithm-crosswalk]]). Group-baseline methods dominate lab recipes; stated reasons attach to modifications tied to failures (clip-higher for entropy, token-level loss for length, sequence-level ratios for MoE, CISPO for clipped fork tokens). GRPO beat DPO on the same IF data (IFEval 89.65 vs 79.67). For subjective rewards, the evidence on online vs offline is split and domain-specific (ch-19 §8). Multi-turn agents: tool-output masking in every search source; GiGPO for step-level credit when states recur.

| Domain | Algorithm and modifications | The failure each modification addresses | Evidence (source, location) |
|---|---|---|---|
| Constraint following | ____ | ____ | ____ |
| Tool use / booking (multi-turn) | ____ | ____ | ____ |
| Conversation quality (judge reward) | ____ | ____ | ____ |
| TTS naturalness (optional) | ____ | ____ | ____ |

**WHY prompts.**
- For each row, name the failure you expect first, then the modification. Do not cite a lab's base-algorithm choice unless the card records its reason.
- If your conversation-quality judge is pairwise against a sibling rollout, why does that pair naturally with a group-baseline algorithm?

---

## §12 Recovery and elasticity

**Teaches:** ch-20, ch-21.

**Evidence.** Rocket's ladder: replica replacement → learner-only recovery (about 5 min; Table 1: n = 88, 7.07%) → in-job rebuild (n = 21, 0.42%) → cluster requeue; runtime elasticity n = 123, 9.88% ([[mai-rl-infrastructure]] p. 11–13). slime retains its serving cluster across trainer failures but resubmission is manual; AReaL relaunches the whole job up to 3 times; open-instruct raises on a 120 s weight-sync timeout (ch-20 §5).

| Decision | Your choice | Why, at your scale |
|---|---|---|
| Smallest recovery scope you will implement | ____ | ____ |
| Stall detection: what is probed, interval, timeout | ____ | ____ |
| Checkpoint interval and the replay or discard policy after a learner failure | ____ | ____ |
| Which settings are runtime-mutable, and at which boundary | ____ | ____ |
| How your external endpoints (judge, simulator, TTS) are monitored and replaced | ____ | ____ |

**WHY prompts.**
- At your run length and GPU count, is learner-only recovery worth building? Estimate expected failures per run and the cost of each whole-job restart.
- Which external endpoint failure would silently corrupt rewards rather than stop the job, and how will you detect it?

---

## §13 Which open framework to start from

**Teaches:** ch-01, ch-02, ch-08, ch-20.

**Evidence** (cards at pinned commits).

| Need | slime (79aa41a) | verl (8357f1e) | AReaL (bcef492) | open-instruct (f020e9a) |
|---|---|---|---|---|
| Serving engine | SGLang; external engines supported, not recovered | vLLM / SGLang; agent loop with tool parsers | SGLang / vLLM; OpenAI-compatible client for agents (`ArealOpenAI`) | vLLM |
| Async modes | colocated or disaggregated; fully-async rollout | `sync`, `colocate_async`, `separate_async` (+ switch) | admission-time staleness bound (`max_head_offpolicyness`) | `async_steps` bound, in-flight weight updates |
| Mismatch correction | TIS / RS helpers | rollout-correction block (off by default) | decoupled PPO; stale-token masking | ρ correction (clamp upper 2.0) |
| Recovery | retained serving; trainer resubmit with replay | `resume_mode: auto` | whole-job relaunch, 3 retries | checkpoint resume |
| Judge integration | reward hooks | reward loop with remote reward manager | second SGLang engine as judge | LM-judge verifier via API |
| Card | [[slime]], [[slime-agentic-and-mismatch-examples]] | [[verl-rollout]], [[verl-search-tool-pipeline]] | [[areal-async-rl]], [[areal-agent-pipelines]] | [[open-instruct-rlvr-pipeline]] |

| Decision | Your choice | The two properties that decided it |
|---|---|---|
| Framework | ____ | ____ |
| What you will have to add yourself | ____ | ____ |

**WHY prompts.**
- Your endpoints are OpenAI-compatible. Which framework's agent interface lets your existing pipecat-style agent loop call the policy without rewriting it, and what does it require to keep token IDs exact (ch-10, [[tito-token-in-token-out]])?
- MAI built Rocket because the open frameworks it cites did not serve "asynchronous RL across several thousand GPUs" ([[mai-thinking-1]] Sec. 3.6). At your scale, does that reason apply?

---

## Summary sheet

Copy this table when the sections above are complete. Every cell must point to a number.

| Item | Decision | Key number | Chapter | Measured / practice |
|---|---|---|---|---|
| Sync vs async | ____ | ____ | ch-02, ch-10 | ____ |
| Staleness budget and sizing | ____ | ____ | ch-03, ch-04, ch-05 | ____ |
| Precision / parity | ____ | ____ | ch-06, ch-07 | ____ |
| Task and reference design | ____ | ____ | ch-09 | ____ |
| Rollout settings | ____ | ____ | ch-10 | ____ |
| Grader ladder | ____ | ____ | ch-11, ch-19 | ____ |
| Reward composition | ____ | ____ | ch-12, ch-19 | ____ |
| Diversity axes | ____ | ____ | ch-13 | ____ |
| Stage order | ____ | ____ | ch-17 | ____ |
| Algorithm per domain | ____ | ____ | ch-18 | ____ |
| Recovery / elasticity | ____ | ____ | ch-20, ch-21 | ____ |
| Framework | ____ | ____ | ch-01, ch-02 | ____ |

---

## Recipe

1. Fill §0–§1 from measurements: run 50–100 conversations with your current policy, simulator and tools, and record turns, tool calls, per-component latency and total wall time.
2. From §1, decide §2 (sync/async) and compute §3 (expected staleness) with the formula.
3. Measure the train–inference log-prob gap of your serving stack against your learner on identical inputs before any RL step (§4).
4. Build the task set with references and pass rates (§5); reject families with no verifiable part unless a calibrated judge exists for them (§7).
5. Calibrate every judge on a gold set from your own conversations: order-swap consistency, length attack, master-key probes for reference parts, human agreement per slice (§7; ch-19 §6).
6. Write the reward as gates plus weighted terms (§8); plant a known hack and confirm your monitors detect it.
7. Check diversity coverage (§9) and write the stage order with entry, exit and retention metrics (§10).
8. Choose the algorithm per domain by expected failure (§11).
9. Decide recovery scope and runtime settings (§12), then the framework (§13).
10. Run a short pilot (a few hundred steps), and recompute every number on the summary sheet from the pilot's logs before the long run.

---

## Common mistakes and how to detect them

| Mistake | Symptom | Detection |
|---|---|---|
| Copying Rocket's settings without its numbers | async complexity with no throughput gain on short rollouts | compare your §1 p95/median ratio to Rocket's Fig. 2 spread |
| Sizing inference without re-checking staleness | staleness rises after adding GPUs | staleness distribution before and after each sizing change |
| Judges validated on human-written pairs only | reward rises, human ratings flat | human agreement on policy samples |
| Simulator tokens in the policy loss | policy imitates the customer | loss-mask audit on a sample rollout |
| Single frozen simulator | held-out score peaks, then returns toward baseline | held-out simulator family in evaluation |
| Weighted compliance term | non-compliant replies with high total reward | count of non-compliant replies with reward above the batch median |
| Choosing a framework by popularity | adding the missing pieces costs more than the framework saved | list the §13 needs before choosing |

---

## Check your understanding

1. Why must the rollout-duration distribution be estimated before deciding sync vs async, and why is the median insufficient?
2. Explain how adding inference GPUs to your run could make staleness worse, using the staleness formula and your own step-time estimate.
3. Why should a group-level judge run on the problem worker, and what happens to latency if it runs per rollout instead?
4. Why is a compliance requirement a gate rather than a weighted term? What evidence from the course supports gating style behind constraints?
5. Why does a single frozen user simulator produce a held-out score that peaks and then returns toward baseline?
6. Which item on your summary sheet has the weakest evidence (practice only), and what pilot measurement would upgrade it?

---

## Connections

- Course chapters: ch-01 (build vs adopt), ch-02 (loop anatomy), ch-03 and ch-04 (staleness), ch-05 (weight transfer), ch-06 and ch-07 (precision and parity), ch-08 (agentic rollouts), ch-09 to ch-13 (task, rollout, grader, reward, diversity), ch-14 to ch-17 (domain pipelines and stage order), ch-18 (algorithm crosswalk), ch-19 (subjective rewards), ch-20 and ch-21 (recovery and elasticity), ch-23 (human-like conversational and speech style for the same agent).
- llm-training: ch-16 (RL prompt distribution), ch-40 (group baselines), ch-43 (entropy and KL), ch-44 (verifiable rewards), ch-44a (length), ch-44b (multi-domain RL), ch-45b (agentic RL), ch-25 (multi-turn conversation synthesis for the simulator).
- Sources: [[rl-design-guide]], [[diversity-axes]], [[subjective-reward-guide]], [[algorithm-crosswalk]], [[recipe-crosswalk]] as the rule lists the worksheet draws on.

---

## Sources

- [[mai-rl-infrastructure]] — Rocket design decisions and the numbers behind them (Fig. 2, Fig. 4, Table 1).
- [[mai-thinking-1]] — Rocket v1 hyperparameters, precision choices, reward composition, consolidation.
- [[rl-design-guide]] — task, rollout, grader, reward rules (measured vs practice).
- [[diversity-axes]] — axis table with evidence labels.
- [[subjective-reward-guide]] — subjective reward rules and the sales-agent sketch.
- [[algorithm-crosswalk]], [[recipe-crosswalk]] — algorithm and stage-order evidence by lab.
- [[slime]], [[verl-rollout]], [[areal-async-rl]], [[areal-agent-pipelines]], [[open-instruct-rlvr-pipeline]] — framework capabilities at pinned commits.
- [[async-rollout]] — OpenRLHF sync vs async stability note.
- [[olmo-3-rl-pipeline]] — max-asynchrony settings, ρ correction, learner/inference sizing.
- [[tito-token-in-token-out]] — token-exact multi-turn rollouts.
- [[scalerl]], [[minimax-m1]] — async efficiency vs asymptote; FP32 LM head.
- [[simulator-collapse]], [[tau2-bench]], [[rest-multi-problem-stress]], [[rubicon-rubric-anchors]], [[rlmt-think-chat]] — evidence cited in individual worksheet sections.
