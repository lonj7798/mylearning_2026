---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/apigen-mt.md
source_url: https://arxiv.org/abs/2504.03601
created_at: "2026-04-23"
---

# Excerpt: APIGen-MT — blueprint-then-rollout for multi-turn function calling

**Source library:** `wiki/raw-data/llm-training/papers/apigen-mt.md`
**Paper:** Prabhakar, Liu, Zhu, Zhang et al. 2025, "APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay."

---

## Why this source anchors ch-26

APIGen-MT is the multi-turn analogue of APIGen's verifier-first thesis. The core move is **separating correctness from realism**: generate a verifiable task blueprint first, then roll it out as a dialog. This breaks the compound-error problem that makes naive multi-turn synthesis (generate dialog, check at end) uneconomic at ~12 turns per trajectory.

Ch-26 §4 walks through the two-phase pipeline and the blueprint JSON shape. This excerpt expands the validation stages and the "reverse task recombination" scaling trick that makes 5K trajectories sufficient for xLAM-2-8B to beat cited GPT-4o baselines on BFCL-V3 multi-turn.

---

## Phase 1 — the blueprint is validated before dialog exists

From source lines 24–30:

> ### Phase 1: blueprint generation
> - Model the agent task as a POMDP and represent each τ-bench domain as a directed graph of APIs and dependencies.
> - Use multiple samplers to diversify the task context: API, policy, domain data, persona, and example samplers.
> - Generate a task configuration with a user instruction, a sequence of ground-truth actions, and the expected final output.
> - Validate each configuration in three stages: action validation checks format, execution, and policy compliance; alignment validation checks that the action sequence satisfies the user intent; final semantic review uses committee aggregation and refinement.
> - Apply **reverse task recombination** to compose more complex tasks from validated building blocks.

Two things about Phase 1 that are architecturally important.

**The blueprint is not a dialog.** It is a structured task configuration with user instruction, ground-truth action sequence, and expected final state. No conversational realism is attempted yet. All verification is done against this structural representation. This is the paper's key observation: verifying a 12-turn dialog is subjective ("did the agent succeed?"); verifying that an action sequence matches a ground-truth plan is mechanical.

**Three validators, stacked APIGen-style.**
- *Action validation* — each call is format/execution/policy-checked exactly as APIGen's single-turn verifier would do.
- *Alignment validation* — an LLM committee judges whether the action sequence satisfies the user intent. This is the multi-turn analogue of APIGen's semantic check.
- *Final semantic review* — committee aggregation and refinement. The paper uses GPT-4o + DeepSeek V3 as the committee.

**Reverse task recombination** is the scaling knob. Once you have a pool of validated blueprint primitives (e.g. `search_flight`, `book_flight`, `search_hotel`), you can compose them into more complex blueprints ("flight + hotel for the same trip") without re-validating each primitive from scratch. This is how the paper scales beyond hand-written seed tasks without paying linear committee cost per new task.

---

## Phase 2 — rollout as realisation, not synthesis

From source lines 31–35:

> ### Phase 2: simulated human-agent interplay
> - Convert the validated blueprint into a multi-turn conversation by simulating both the user and the agent with LLMs.
> - Use rejection sampling so only trajectories that actually reach the task goal are kept.
> - In the τ-bench case study, the authors source **15 read** and **13 write** APIs and use **GPT-4o** plus **DeepSeek V3** during generation, validation, and interplay.
> - The paper reports that the collected tasks average around **12 turns** and that agentic feedback improves task-collection success rate to about **70%**.

The framing shift worth internalising: **Phase 2 *realises* a validated blueprint as a dialog; it does not invent new task content.** The user-simulator has the blueprint's persona and intended behaviour; the agent executes real calls against the API substrate; rejection sampling discards any trajectory that fails to reach `expected_final_state`. Because correctness is pre-validated, the rejection rate in Phase 2 is about conversational realism (did the user push back plausibly? did the agent discover the right action sequence?), not about task correctness.

The 70% task-collection success rate reported here is post-rejection-sampling. Without the Phase-1 blueprint anchor, naive multi-turn synthesis sees compound errors accumulate: a 12-turn trajectory at 95% per-turn correctness lands at 54% end-to-end, meaning you discard nearly half of a ~$1/sample rollout. Blueprint-first collapses that to "did the rollout realise the pre-validated plan?" — a much cheaper rejection criterion.

---

## Results and the consistency argument

From source lines 37–40:

> ## Quality / diversity evaluation
> - On **BFCL v3**, the xLAM-2-8b-fc-r model is reported to reach **69.25%** on multi-turn tasks, above the cited GPT-4o baseline in the paper materials.
> - On **τ-bench**, xLAM-2-70b-fc-r is reported at **56.2% overall**, above Llama 3.1 70B Instruct and GPT-4o in the cited comparison.
> - The main empirical point is not just average score but **consistency across trials**, where the APIGen-MT-trained models hold up better in multi-turn settings.

The **consistency** framing connects to BFCL-V4's `pass^k` metric ([[bfcl]]). A model with 70% single-trial success and low consistency can land 30% on `pass^4`; a model with 60% single-trial success but high consistency can land 45%. The blueprint-then-rollout design directly addresses this: because trajectories are realisations of pre-validated plans, the training distribution has a structural regularity that single-trial SFT does not.

This is the argument for why 5K trajectories is enough: the data is *densely correct along a known plan*, not sparsely correct across a diverse distribution. Scaling up the corpus provides diminishing returns; the 5K number reflects the point where marginal addition doesn't raise the consistency floor.

---

## Limits named in the paper

From source lines 42–45:

> ## Risks + gotchas
> - **Schema dependence:** the data quality is tied to the API graph and policy structure of the source benchmark.
> - **Simulation gap:** an LLM-simulated user is useful for scale, but it can still miss some real human interaction quirks.
> - **Validation overhead:** the blueprint stage adds substantial filtering and review cost, which is the price of getting verifiability.

The simulation-gap point is the open problem. τ-bench's 15-read / 13-write API domains are narrow; real production user behaviour includes class of push-backs (ambiguous restatements, mood-driven reversal, off-topic intermissions) that LLM-simulated users rarely produce. This caps how far blueprint-then-rollout can substitute for real user data. It remains, however, the strongest open-source multi-turn function-calling data recipe as of 2025.

---

## Connections

- Direct predecessor: [[apigen]] — single-turn verifier-first pipeline; APIGen-MT applies the same philosophy to multi-turn.
- Substrate: τ-bench + the ToolBench API graph ([[toolllm]]).
- Downstream consumer: [[xlam]] — xLAM-2 family's Stage-2 multi-turn SFT corpus.
- Broader-coverage alternative: [[toolace]] — reaches multi-turn via complexity-controlled multi-agent dialog (MAI) rather than blueprint anchoring.
- Evaluation target: [[bfcl]]-V3 (multi-turn) and τ-bench; pass^k consistency metric from V4 rewards APIGen-MT-style structural regularity.
