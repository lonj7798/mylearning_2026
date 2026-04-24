<!-- chapter: ch-26
     track: synthetic
     title: Modality — Tool and Function-Calling Data
     sources: [[toolformer]], [[toolllm]], [[apigen]], [[apigen-mt]], [[toolace]], [[xlam]], [[bfcl]], [[hammer]], [[nexusraven]], [[gorilla]], [[api-bank]], [[glaive-function-calling]], [[granite-function-calling]]
     figures: figures/tool-pipeline.html
-->

# Chapter 26 — Modality: Tool and Function-Calling Data

> **Core insight.** Tool-use data is the modality where **the environment is the label**. Unlike math (symbolic equivalence grades free-form CoT) or code (unit tests grade a program), a function call is graded by whether it *parses against the schema, executes against a real implementation, and fulfils the user's intent*. Every pipeline in this chapter — Toolformer, ToolLLM, APIGen, ToolACE, APIGen-MT — is a different answer to one question: *how do you get a verification signal cheap enough to filter millions of candidates, strong enough to beat a GPT-4 teacher's own errors?*
>
> **Guideline.** Build the pipeline around the verifier, not the generator. Minimum viable stack: (1) schema-enforced format check (free), (2) sandboxed execution against a reference implementation (expensive but load-bearing — removing it costs ~11 BFCL points), (3) semantic LLM-as-judge on `(query, call, result)` (catches wrong-unit / wrong-target errors). For multi-turn, add a blueprint phase that locks correctness *before* conversational realism is rolled out. Never train on a single source — the xLAM / ToolACE / Hammer split shows each source covers one BFCL axis and leaves gaps elsewhere.

---

## Why this chapter exists

Function calling is the first LLM modality where the loss the model trains on and the metric it ships against diverged completely. In math SFT you train on a gold CoT and eval on the same gold format. In dialogue you train on a preferred response and eval against an approximately equivalent judge. In tool use, you train on `{"name": "get_weather", "arguments": {"location": "Paris"}}` and you eval against `get_weather(location="Paris")` Python syntax scored by an AST matcher that normalises literal representations and accepts any kwarg order ([[bfcl]]). Training distribution, call template, evaluation parser, and BFCL leaderboard are four different specifications of the same act, and every data pipeline in the literature is an attempt to bridge them.

The chapter traces that bridge through five generations.

1. **[[toolformer]] (2023)** — self-supervised annotation on raw text via a perplexity-delta filter. No teacher stronger than the model itself.
2. **[[toolllm]] (2023)** — synthetic trajectories grounded in 16K real REST APIs, with DFS-DT search replacing brittle ReACT. *Real substrate, synthetic supervision.*
3. **[[apigen]] (2024)** — the first pipeline that refuses a sample unless it passes all three of format / execution / semantic. ~40% rejection is the feature, not a bug.
4. **[[apigen-mt]] + [[toolace]] (2024–25)** — extend verification to multi-turn via blueprint-then-rollout, and to broader coverage via complexity-controlled multi-agent dialog.
5. **Benchmark-shaped specialists (Hammer / NexusRaven / Granite)** — each targets one [[bfcl]] axis with one data trick.

The through-line — **the verifier is the data pipeline** — is why this is a standalone chapter rather than folded into [[ch-18]]'s general recipe. The generator is largely interchangeable (DeepSeek-Coder-V2 vs GPT-4 differ by ~2% in ablations); the verifier is not.

---

## 1. Toolformer: the annotation bootstrap

[[toolformer]] is the conceptual preamble. In 2023, before any 16K-API corpus existed, the open question was: *can you generate tool-use supervision from plain text without a human or a stronger teacher?* Toolformer's answer: yes, if the model can verify itself. The signal is **loss reduction on future tokens when the API result is visible**.

**Stage 1 — propose.** Prompt GPT-J 6.7B with a few demonstrations of inline API calls. At each token position `i`, compute the probability the model assigns to starting an API call. Keep positions where that probability exceeds **5%** (appendix threshold). Of ~10M positions scanned, ~2% survive.

**Stage 2 — sample + execute.** Sample candidate calls at surviving positions, then actually execute them against one of five tools (QA, Wikipedia search, calculator, translation, calendar). Store the real returned result `r`.

**Stage 3 — filter by perplexity delta.** For continuation tokens `y` that followed position `i`:

```
L_no_call    = −Σ_t log P(y_t | prompt)
L_call_empty = −Σ_t log P(y_t | prompt + "[API(args) → ]")
L_call_full  = −Σ_t log P(y_t | prompt + "[API(args) → result]")

Δ = min(L_no_call, L_call_empty) − L_call_full
```

Accept iff `Δ > τ` with default `τ = 1.0` nat. Two subtleties.

- **Why compare against `L_call_empty`?** Without it, the model gets credit for merely seeing the string `Calculator(...)`; the `min` forces the result itself to add predictive value. This is the earliest "reward outcome, not format" rule — APIGen's semantic check and BFCL's relevance-detection are downstream variants.
- **Why is τ = 1.0 nat aggressive?** A 1-nat lift on a ~5-token window is a factor-of-`e` reduction in perplexity. Table 2 of the paper keeps only a few thousand annotations per tool out of hundreds of thousands of candidates; the budget assumes 99%+ rejection.

Toolformer does not plan, does not synthesise multi-turn, and does not retrieve — the five tools are always in the prompt. But it establishes three rules every later system inherits: execute real calls during generation, test usefulness not just validity, and high rejection rates are a feature.

---

## 2. ToolLLM and DFS-DT: synthetic trajectories on real APIs

[[toolllm]] jumps from *annotating one call* to *generating a whole solution path*. The paper contributes **ToolBench** (16,464 real RapidAPI endpoints), a scenario taxonomy, and **DFS-DT** — a depth-first decision-tree trajectory generator that replaces brittle ReACT rollouts.

**What is real vs synthetic.** The durable contribution is the split: API catalog + executions are real; user instruction + reasoning trace + final answer are synthetic (generated by `gpt-3.5-turbo-16k`). Every later pipeline inherits this split. APIGen narrows the API set (3,673 executable) but keeps it. ToolACE mutates the API set (26,507 evolved) while keeping "execute where possible." *The environment under the trace is always real; the trace is always synthetic.*

**Scenario grid.** Data loops over a 3×2 grid: `G1/I1` single-tool, `G2/I2` intra-category multi-tool, `G3/I3` intra-collection multi-tool. Public release: **126,486 instances, 469,585 real API calls, ~4 reasoning steps per instance**.

**DFS-DT.** Generating a *training* trajectory differs from running one at inference: you want the highest probability of *any* correct path, because a failed rollout is pure API waste. Plain ReACT has no recovery — first wrong action poisons every subsequent observation. DFS-DT searches with retraction:

```python
def dfs_dt(instruction, apis, model, max_depth, beam):
    root = Node(history=[], depth=0)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_terminal() or node.depth >= max_depth:
            if node.is_successful():
                return node.trajectory     # first accepted trajectory wins
            continue
        candidates = model.sample_actions(node.history, apis, k=beam)
        for (thought, action) in sort_by_score(candidates):   # preorder
            obs = execute(action)
            if obs.is_error:
                continue                    # retract: do NOT commit this child
            child = Node(history=node.history + [(thought, action, obs)],
                         depth=node.depth + 1)
            stack.append(child)
    return None     # all branches exhausted; reject this instance
```

The appendix notes that *without the `if obs.is_error: continue` retraction, DFS-DT degrades to ReACT*. The whole lift comes from the backtrack. Table 3 at matched API budget:

| Setting | ReACT | ReACT@N | DFS-DT |
|---|---|---|---|
| I1 (single-tool) | 42.2 | 47.7 | **57.3** |
| I2 (intra-cat)  | 30.0 | 34.3 | **48.2** |
| I3 (intra-col)  | 21.7 | 26.0 | **43.2** |

A 15–20 point lift on the hardest setting is what makes multi-tool annotation tractable. ToolEval (LLM-judge) reports **87.1% pass-rate agreement with humans** — high enough to be useful, not high enough to trust blindly, which is the gap APIGen closes.

---

## 3. APIGen's three-layer verifier

[[apigen]] reframes the field: synthetic function-calling data is trustable *iff* every sample clears three independent checks, in sequence.

1. **Format check** — JSON parses, required fields present, types match schema. Rejects malformed JSON, wrong arg name, string-for-int.
2. **Execution check** — Python sandbox, 5-second timeout, against reference implementation. Rejects any exception or timeout.
3. **Semantic check** — GPT-4 judge sees `(query, call, execution_result)` and must answer "Yes" to "does the call correctly fulfil the query?". Rejects any non-"Yes".

**Concrete numbers.** 3,673 executable APIs (curated subset of ToolBench's 16K where Salesforce either ran the endpoint or wrote a Python mock). 60,000 accepted samples across four data types — simple (~40%), multiple (~25%), parallel (~20%), parallel-multiple (~15%). Each API appears ~16× with different arg combinations. Dedup: MinHash on `(query, call)`. Teacher: DeepSeek-Coder-V2-Instruct primary, GPT-4 ablation.

**The ablation that proves each layer earns its keep:**

| Verifier config | BFCL-V1 overall | Δ |
|---|---|---|
| Full 3-layer (format + execution + semantic) | **88.24** | — |
| Remove semantic check | 82.2 | −6.0 |
| Remove execution check | 77.3 | −10.9 |
| Remove format check | 70.1 | −18.1 |

- **Format-only lands at ~70%** — the Glaive-style 2023 ceiling ([[glaive-function-calling]]).
- **Execution is the biggest single-layer lift (−11 points removed).** This is why APIGen bottlenecked on 3,673 APIs — Salesforce needed reference implementations.
- **Semantic catches the residual 6%** — calls that run without error but answer the wrong question (wrong unit, wrong target, right function with wrong arg semantics). Only an LLM judge (or a human) catches these.

Rejection rate: **~40%** across the three stages combined. To produce 60K gold you generate ~100K raw. The 60K corpus is not "60K of teacher output" but "60K that survived a strict gate." See [figures/tool-pipeline.html](figures/tool-pipeline.html) for the per-stage reject examples.

**Downstream:** The corpus trains [[xlam]]. **xLAM-7B (Mistral base) reaches 88.24% BFCL-V1**, #1 among <13B at Sept-2024 release. The xLAM-2 staged recipe — **APIGen-60k SFT → APIGen-MT-5k SFT → optional DPO (β=0.1) on (correct, hallucinated-name) pairs** — is the clearest open multi-turn FC specialist recipe as of 2025.

---

## 4. Multi-turn: APIGen-MT's blueprint-then-rollout

Single-turn is nearly solved by APIGen. Multi-turn — ~12 messages, state mutates between calls, correctness is a *sequence* of (call, observation, reasoning) triples — is what [[apigen-mt]] addresses. The trick is **separate correctness from realism**.

**Phase 1 — blueprint.** Phase 1 generates no dialog. It generates a structured task config:

```jsonc
{
  "domain": "airline",
  "user_persona": "budget traveller, prefers refundable fares",
  "instruction": "Book JFK→LAX under $250 and a hotel at LAX under $150 for Friday.",
  "ground_truth_actions": [
    {"api": "search_flights", "args": {"from": "JFK", "to": "LAX", "max_price": 250}},
    {"api": "book_flight",    "args": {"flight_id": "$F.id"}},
    {"api": "search_hotels",  "args": {"near": "LAX", "max_price": 150}},
    {"api": "book_hotel",     "args": {"hotel_id": "$H.id"}}
  ],
  "expected_final_state": {"flight_booked": true, "hotel_booked": true}
}
```

The blueprint passes three validators before dialog exists: **action validation** (format/execution/policy-check per call, APIGen-style), **alignment validation** (LLM committee: does the sequence satisfy the intent?), **semantic review** (committee aggregation + refinement; the paper uses GPT-4o + DeepSeek V3). The architectural move is **reverse task recombination**: compose complex blueprints from validated primitives.

**Phase 2 — rollout.** Two LLM actors (user-simulator, agent) role-play the conversation. The agent executes real calls; the user-simulator carries the persona and pushes back, clarifies, or adds constraints. **Rejection sampling** keeps only trajectories that reach `expected_final_state` with matching ground-truth actions. Reported ~**70% task-collection success rate**, ~**12-turn average**, τ-bench substrate of 15 read + 13 write APIs.

**Why blueprint-first beats rollout-then-verify.** A 12-turn dialog at 95% per-turn correctness has 54% end-to-end correctness — you throw away half of a ~$1/sample rollout. Blueprint-first fixes the ground-truth *once*; the rollout only *realises* it. Secondly, end-to-end dialog verification is subjective ("did it succeed?"); blueprint-then-rollout decomposes it into a structural check against a known plan.

Result on **BFCL-V3 multi-turn: xLAM-2-8B reaches 69.25%**, above cited GPT-4o baselines, on only 5K trajectories.

---

## 5. ToolACE: breadth through self-evolution

[[toolace]] pushes a different axis: **API diversity**. If APIGen chokes at 3,673 APIs because it needs executable implementations, how do you reach 26,507 while keeping verifiability?

**Tool Self-Evolution Synthesis (TSS).** From a 3K seed of real APIs, an LLM mutates via three operators: parameter extension, domain transfer (weather→stock), functionality refinement. Filter on schema parseability, name uniqueness, LLM-judge novelty/utility. Result: **26,507 APIs across 390 domains** — largest public pool as of 2024. Tradeoff: most have no real implementation; responses are LLM-simulated.

**Multi-Agent Interactive Dialog (MAI).** Three LLM roles (user / assistant / tool-simulator). A **complexity evaluator** classifies each dialog into 5 difficulty levels and *conditions generation* to hit a target mix: simple single-call (~30%), multiple (~25%), parallel (~20%), nested/multi-turn (~15%), info-incomplete (~10%). This distribution is hand-tuned to match BFCL category distribution — an explicit instance of **benchmark-shaped data** (§7).

**Dual-layer verification.** Rule-based (schema + param + type + execution where mock exists) plus model-based (GPT-4 judge, 3-way verdict: query-clarity / call-correctness / response-consistency). ~40% rejection, final 11,300 dialogs.

**ToolACE-8B (Llama-3.1-8B base): 91.41% BFCL-V1**, beating xLAM-7B's 88.24% at matched scale. Ablation: −4.3% removing TSS, −3.1% removing complexity controller, −5.2% removing model-judge, −2.8% removing rule-checks. Honest tradeoff vs APIGen: 10× more APIs but fewer real executions; the tool-LLM can hallucinate realistic-looking-but-wrong outputs. This is why the §8 recipe uses *both* sources.

---

## 6. BFCL: how the benchmark became the data spec

[[bfcl]]'s **ability-axis decomposition** is the data-design spec every pipeline now follows.

**The seven categories:** Simple (1 call, 1 function) · Multiple (pick right from ≥2 candidates) · Parallel (≥2 calls, same function) · Parallel-Multiple (≥2 calls, multiple functions) · **Relevance-Detection** (irrelevant query → correct answer is *no call*) · Multi-Turn (V3+, stateful) · Multi-Step (sequential calls for one task). Compare to APIGen's four types, ToolACE's five levels, Granite's seven capabilities ([[granite-function-calling]]) — the same taxonomy rediscovered. **Whoever sets the eval taxonomy sets the data-generation taxonomy.**

**The AST matcher.** Scoring parses predicted and gold calls into `(name, kwargs)`, then: name exact-match; kwargs sorted by key, whitespace stripped, literals canonicalised (`1.0 ≡ 1`, `"red" ≡ 'red'`); required args present. Tolerates `get_weather(city="Paris")` vs `get_weather(city='Paris',)`; rejects `get_weather(loc="Paris")`. *Lenient on representation, strict on semantics.*

**Versions and induced data targets:**

| Version | Key addition | Induced data requirement |
|---|---|---|
| V1 (Feb 2024) | 7 categories, single-turn | APIGen / NexusRaven ~100K |
| V2 Live (Aug 2024) | +1,500 real user queries | ToolACE complexity sampler; Hammer irrelevance aug |
| V3 Multi-turn (Sep 2024) | Stateful multi-turn | APIGen-MT 5K trajectories |
| V4 Agentic (2025) | Long-horizon + web/memory | SWE-Gym agent data ([[ch-27]]) |

**pass^k.** V3+ requires success on *all* k independent trials. xLAM-2-70B τ-bench: **pass^1 = 56.2%, pass^4 = 39.4%** — exposing a consistency gap that rewards pipelines producing *structurally consistent* trajectories (APIGen-MT's blueprint anchor), not just average-correct ones.

Current 2025 snapshot: top open <13B — ToolACE-8B, xLAM-2-8B, Hammer 2.1; top open overall — xLAM-2-70B-fc-r. Even frontier models still call a tool on ~10% of irrelevant queries.

---

## 7. Benchmark-shaped specialists

Once BFCL decomposed tool use into axes, each axis could be targeted with a specific data trick.

**Hammer — relevance via function-name masking.** Small on-device models fail relevance-detection because of **naming bias**: they fire `send_email` whenever "email" appears in the query. [[hammer]]'s fix is augmentation, not architecture: (a) replace tool names with random placeholders `func_[a-z0-9]{6}` consistently across a sample, in **30%** of training data; (b) **~30%** irrelevance samples where gold = refuse / clarify. Masking-ratio ablation: 30% is the optimum; 50% destroys recall; 10% is too weak. Combined lift: **+13 points** on BFCL relevance; Hammer-7B ~90%, matching GPT-4.

**NexusRaven — nested calls via curriculum.** [[nexusraven]]'s training mix is simple (60%) / parallel (20%) / **nested (20%)**. Nested examples look like `save_file(name="r.txt", content=summarize(translate("texto", to="en")))`. Ablation: **removing nested → −15 points on a nested-eval track**, zero drop on simple/parallel. The data tells the model "arguments can themselves be function calls"; no amount of general SFT teaches this if it's absent from the distribution.

**Gorilla — retriever-aware fine-tuning.** [[gorilla]] includes top-k retrieved API docs in the training prompt; the model learns `condition on doc → emit call` rather than `recall API name from memory`. 80% noisy retrieval + 20% oracle, on 16K instruction-API pairs. **Hallucination rate: 11% retriever-aware vs 40% GPT-4 without** — 4× reduction. Every modern pipeline at serving time uses retrieval when the API pool exceeds a few hundred tools; Gorilla showed it must be in *training* too.

**Granite — multi-source mix.** [[granite-function-calling]]'s seven-capability taxonomy (Nested / Parallel / Multiple / Multi-turn / Relevance / Sequencing / Slot-Filling) is covered by a blend: APIGen 25% / ToolLLM 20% / Glaive 15% / Nexus 10% / IBM in-house 20% / general chat 10%. Per-capability resampling ensures each capability ≥10% of the mix. Naive equal-mix loses 5–10 points vs the tuned mix. Removing ToolLLM → multi-turn drops 12 points; removing Nexus → nested drops 18; removing relevance slice → relevance drops 10. **Each source is load-bearing for one capability and null for the others.**

---

## 8. Drop-in recipe

Combining the chapter into a recipe for a 7B-class FC specialist on open data:

```
Mix (~200K samples)
├── 25% APIGen-FC-60k             — single-turn, 3-layer verified
├── 20% ToolBench (DFS-DT filter) — real-API multi-tool trajectories
├── 15% APIGen-MT-5k (upsampled)  — multi-turn blueprint-verified
├── 10% ToolACE nested + info-incomplete subset
├── 10% Hammer masking + irrelevance aug (applied to above)
├── 10% NexusRaven nested-call curriculum
└── 10% OpenHermes subset         — preserves chat quality

Verifier chain for any new-source additions:
  1. Format check          — JSON parse + schema + type check
  2. Execution check       — Python sandbox, 5-s timeout
  3. Semantic check        — GPT-4o judge on (query, call, result)
  4. MinHash dedup         — on (query, call), threshold 0.9
  Expected acceptance ~55–60%.

Call format (match the BFCL AST matcher):
  - Tool schemas in system prompt (JSON-schema style).
  - Assistant emits OpenAI-compatible `tool_calls` JSON.
  - Tool responses in `tool` role messages.

Training (xLAM-2 staged):
  Stage 1: SFT on single-turn mix   (LR 2e-5, 3 epochs, 8k seq)
  Stage 2: SFT on multi-turn mix    (LR 1e-5, 2 epochs, 16k seq)
  Stage 3 (optional): DPO on (correct, hallucinated-name), β=0.1
```

Two operational rules.

- **Match the call template to the AST matcher.** Training on Glaive's `<functioncall>{...}</functioncall>` XML instead of OpenAI JSON costs 5–10 points through inference-time format translation.
- **Gate on relevance-detection before shipping.** A model at 92% overall BFCL but 65% relevance hallucinates tools on irrelevant queries 35% of the time in the wild — the #1 production failure mode.

---

## Connections and what's next

- **[[ch-18]]** — Synthetic-data design pattern. APIGen's 3-layer verifier is the pattern's verify step with teeth.
- **[[ch-19]]** — Generation methods. Self-Instruct / Evol-Instruct / Magpie are the *generator* side; every pipeline here uses one of them (ToolLLM = Self-Instruct-style; ToolACE's TSS = Evol-Instruct-over-APIs).
- **[[ch-25]]** — Multi-turn conversation synthesis. APIGen-MT's Phase-2 rollout is that machinery *constrained by a blueprint*.
- **[[ch-27]]** — Agentic trajectories. Extends tool use to long-horizon environment-grounded tasks (web, terminal, repo). BFCL-V4 is the eval bridge.
- **[[ch-28]]** — Long-context synthesis. Multi-turn tool use is one of the stronger long-context signals available in post-training.

## Further reading

- [[toolformer]] — Schick et al. 2023. Self-supervised annotation origin; read §4 for the filter derivation.
- [[toolllm]] — Qin et al. 2023. ToolBench + DFS-DT; the real-substrate / synthetic-supervision split.
- [[apigen]] — Liu et al. 2024. The 3-layer verifier and Table 4 ablation.
- [[apigen-mt]] — Prabhakar et al. 2025. Blueprint-then-rollout for multi-turn.
- [[toolace]] — Liu et al. 2024/ICLR 2025. Self-evolving API pool + complexity-controlled dialog.
- [[xlam]] — Zhang et al. 2024/2025. The open model family; staged recipe in §3.
- [[bfcl]] — Patil et al. 2024. The benchmark whose taxonomy became the data taxonomy.
- [[hammer]] / [[nexusraven]] / [[gorilla]] / [[granite-function-calling]] / [[glaive-function-calling]] / [[api-bank]] — the specialists and historical baselines.

## Companion visualization

**[figures/tool-pipeline.html](figures/tool-pipeline.html)** — interactive walkthrough of APIGen's three-layer verifier. Click each stage (format → execution → semantic) to see a concrete reject example, the layer's pass rate, and the cumulative BFCL impact of removing it. Use it to internalise *why the rejection rate is a feature* and to calibrate how a ~40% total reject rate composes across three independent layers.
