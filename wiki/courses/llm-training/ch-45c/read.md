<!-- chapter: ch-45c
     track: rl
     kind: content
     title: Context Management for Long-Horizon Agents
     deps: [ch-45b, ch-32c]
     sources: [[chroma-context-rot]], [[anthropic-context-engineering]], [[anthropic-context-management]],
              [[minimax-m2-interleaved-thinking]], [[deepseek-v3.1]], [[glm-5]], [[kimi-k2-5]],
              [[supo-summarization-rl]], [[resum]], [[context-folding]], [[memagent]], [[mem1]],
              [[manus-context-engineering]], [[openai-codex-max-compaction]], [[mirothinker]],
              [[lost-in-the-middle]], [[context-length-alone-hurts]], [[browsecomp-plus]], [[grpo]]
     figures: figures/context-budget.html
     revised: 2026-09 (generality revision)
-->

# Chapter 45c — Context Management for Long-Horizon Agents

> **Core insight.** A long-horizon agent produces more tokens of history than its context window holds, and the reported score of a fixed checkpoint depends on the rule that decides what stays. On BrowseComp the same checkpoints move 51.4 → 67.6 (DeepSeek-V3.2, Discard-all at 80% of the window; [[deepseek-v3.1]] §4.1, §4.4), 60.6 → 74.9 (Kimi K2.5, Discard-all; [[kimi-k2-5]] §5), and 62.0 → 75.9 (GLM-5; the report gives two different descriptions of the policy behind 75.9, see §4.3; [[glm-5]] §4.2.4, Table 7, App. B.2). The gap is larger than the gap between most model releases in the same table, so an agentic score without a stated context policy is not comparable to another one. The second result is that the policy can be trained rather than hand-written: with an identical base model and identical budget, summarization trained inside the RL objective reaches 53.0% on a held-out BrowseComp-Plus split against 39.0% for multi-turn GRPO at the same 64K working context ([[supo-summarization-rl]] Table 1), and a folding agent with a 32K working context and up to 10 branches reaches 0.620 pass@1 against 0.540 for a 327K-context ReAct agent trained the same way ([[context-folding]] Table 1).
>
> **Guideline.** When you report an agentic benchmark, report the score with and without context management and state the trigger, the policy, and the step budget, because the three reports that disclose both numbers for their own checkpoint differ by 13.9 to 16.2 points on it ([[deepseek-v3.1]] §4.4, [[kimi-k2-5]] §5, [[glm-5]] Table 7). When the harness sends tool results as tool messages and the model has an interleaved reasoning mode, keep prior reasoning in the history, because MiniMax reports SWE-Bench Verified 69.4 vs 67.2, τ² 87 vs 64, BrowseComp 44.0 vs 31.4, GAIA 75.7 vs 67.9 and xBench 72.0 vs 66.0 for retained versus discarded prior-round thinking ([[minimax-m2-interleaved-thinking]]); when the harness wraps tool results as user messages, DeepSeek recommends the non-thinking mode instead ([[deepseek-v3.1]] §3.2.1). When you cannot train and need one change, fold tool observations older than the last k rounds while keeping reasoning and actions, because that alone moved GLM-5 from 55.3% to 62.0% at k = 5 ([[glm-5]] §4.2.4). When you can train and the task needs more history than one window, train the compression step inside the RL objective rather than bolting it on at inference, because plain GRPO applied to a summarizing rollout did not transfer the gain in either report that tested it ([[supo-summarization-rl]] Table 1; [[resum]] Table 2). When the agent runs against a priced inference API, keep the context append-only and the prefix stable, because a single changed token invalidates the KV-cache from that position onward and Manus reports a 10× price difference between cached and uncached input tokens at the time of writing ([[manus-context-engineering]]).

## Why this chapter matters for a general-purpose model

The previous chapter trained an agent over multiple turns and treated the trajectory as one object: mask the observations, assign credit to the model's own tokens, keep the update stable (ch-45b). That treatment assumed the trajectory fits in the context window. For the task families that current agentic reports target, it does not. DeepSeek measured that "20%+ of the test cases exceed" a 128K limit on BrowseComp ([[deepseek-v3.1]] §4.1). The folding paper reports main trajectories of about 8K tokens while over 100K tokens are processed, and response lengths on its hard subset rising from about 100K to over 160K tokens during training ([[context-folding]] §4.2–4.3).

Once the trajectory exceeds the window, something must be removed, and the removal rule becomes part of the policy. This has three consequences for a generally capable model.

1. **It changes what the model is trained on.** If the rollout that generated a reward was produced under a summarization policy, then the policy gradient is defined over summarized states, not raw ones. Ignoring that mismatch is the same class of error as a chat-template mismatch in SFT (ch-04): no exception is raised, the loss curve looks ordinary, and the deployed agent behaves differently from the trained one.
2. **It changes what the score means.** A benchmark number produced with Discard-all and 800 steps is a different measurement from the same benchmark with no context management and a stop at the window limit. Both appear in published tables under the same benchmark name.
3. **It is a capability, not only plumbing.** Deciding what to carry forward is a skill that transfers: agents trained to summarize keep working past the horizon they were trained on ([[supo-summarization-rl]] App. C.2 Table 3; [[memagent]] Table 1; [[mem1]] Table 1). Whether that transfer holds outside the trained task family is the open part.

The chapter sits inside the RL phase, after multi-turn agentic RL (ch-45b) and after the long-context evaluation chapter that separates claimed from effective context length (ch-32c). It is the bridge between them: ch-32c measured that a window is not usable to its stated size; this chapter is what agent builders do about it.

## §1 The measurable problem

**Definition.** The *working context* is the token sequence the model sees at one decoding step. The *effective context* of a run is the total history the run processes, which can exceed the working context if a policy resets or compresses it. [[supo-summarization-rl]] §5.1 writes this as `L_effect := L_RL × (S + 1)`, where `L_RL` is the working context length used during RL and `S` is the maximum number of summarization events allowed in a rollout.

**The problem, stated measurably.** Two effects compound.

*Effect one: the window fills.* With a system prompt of `s` tokens and per-turn cost `r + a + o` (reasoning, action, observation), a keep-everything agent reaches the window `W` after `n = ⌊(W − s) / (r + a + o)⌋` turns. The counts are not small: [[manus-context-engineering]] reports "around 50 tool calls" for a typical Manus task and an average input-to-output token ratio near 100:1; [[mirothinker]] runs up to 600 tool calls in a 256K window (§6.1).

*Effect two: accuracy falls with length before the window fills.* [[chroma-context-rot]] holds the task fixed and varies only input length across 18 models: on LongMemEval, every model family scores higher on the ~300-token focused prompts than on the ~113k-token full prompts built from the same questions. [[context-length-alone-hurts]] reaches the same conclusion while controlling for retrieval success. [[lost-in-the-middle]] gives the position-dependent form of the effect. The practitioner reports name the same thing: "model accuracy degrades substantially under extremely long contexts (e.g., beyond 100k tokens)" ([[glm-5]] §4.2.4), and "Model performance tends to degrade beyond a certain context length, even if the window technically supports it" ([[manus-context-engineering]]).

The mechanism is not established. [[anthropic-context-engineering]] offers an interpretation — every token can attend to every other, giving n² pairwise relationships for n tokens, and training distributions contain more short sequences than long ones, so the model has fewer specialized parameters for context-wide dependencies — and labels the consequence an "attention budget". Treat this as **Interpretation**; [[chroma-context-rot]] explicitly declines to explain the mechanism.

**Implication.** Enlarging the window does not remove the problem, it moves it. [[resum]] Table 3 is the direct test: with Tongyi-DeepResearch-30B-A3B at a 128k context limit and a 120-call budget, summarization still beats keep-everything on BrowseComp (44.5 vs 42.2 Pass@1) — a smaller gap than at 32k (34.5 vs 27.7), but not zero.

To see the token bookkeeping for a configuration of your choice, open [figures/context-budget.html](figures/context-budget.html); it plots context length against turn number for five policies and reports how many turns each one completes before the window is exhausted.

## §2 The operations a context policy can perform

A context is a sequence of blocks: system prompt and task `s`; and per round `i`, a reasoning block `rᵢ`, an action `aᵢ`, and an observation `oᵢ`. [[glm-5]] §4.2.4 writes the trajectory exactly this way: `(q, r₁, a₁, o₁, r₂, a₂, o₂, …, r_n, a_n, o_n)`. Every policy in the literature is a choice of which of these blocks survives.

| Operation | What it removes | What it keeps | Where it is used |
|---|---|---|---|
| Keep everything (ReAct) | nothing | all | baselines in [[supo-summarization-rl]], [[resum]], [[context-folding]] |
| Hide-Tool-Result | observations older than the most recent round | all reasoning, all actions | HLE-with-tools in [[kimi-k2-5]] |
| keep-recent-k | observations older than the last k rounds, replaced by a placeholder string | reasoning, actions, last k observations | [[glm-5]] (k = 5); [[mirothinker]] (K = 5) |
| Recent-history truncation | everything but the last fixed token budget | the tail | the "Recent History" baseline in [[resum]] (latest 22k tokens) |
| Summary / compaction | the overflowing trajectory | `s` plus a generated summary | [[deepseek-v3.1]] §4.4; [[anthropic-context-engineering]]; [[openai-codex-max-compaction]]; [[supo-summarization-rl]]; [[resum]] |
| Discard-all | all prior tool-call history | `s` only | [[deepseek-v3.1]] §4.4; [[kimi-k2-5]]; [[glm-5]] |
| Branch and fold | the interior of a completed sub-trajectory | a templated return message in the main thread | [[context-folding]] §2.2 |
| External memory | content moved out of the window | file paths and identifiers inside the window | [[anthropic-context-management]]; [[manus-context-engineering]]; [[memagent]]; [[mem1]] |
| Sub-agent isolation | the sub-agent's trace | the routed result | [[kimi-k2-5]] Agent Swarm; [[anthropic-context-engineering]] |

Two independent axes are being decided here, and reports often conflate them:

- **Which block type is dropped.** Observations are the largest and the most redundant, which is why every heuristic policy starts there. Reasoning is smaller and, as §3 shows, the more expensive to drop.
- **Whether the removal is recoverable.** [[manus-context-engineering]] states the design rule: keep compression restorable — drop a page's content while keeping its URL, drop a document while keeping its path — with the stated reason that "you can't reliably predict which observation might become critical ten steps later", so "any irreversible compression carries risk". Discard-all is at the other end: it is irreversible within the run.

## §3 Reasoning retention across tool calls is a template decision

**Definition.** *Interleaved thinking* is the pattern in which the model emits a reasoning block, calls a tool, receives the result, and emits the next reasoning block, with earlier reasoning blocks still present in the history ([[minimax-m2-interleaved-thinking]]).

**The problem.** The chat-completion convention inherited from single-turn reasoning models discards reasoning content at the start of the next turn. [[deepseek-v3.1]] §3.2.1 states what that costs in a tool loop: "replicating DeepSeek-R1's strategy — discarding reasoning content upon the arrival of the second round of messages — results in significant token inefficiency. This approach forces the model to redundantly re-reason through the entire problem for each subsequent tool call."

**The rule DeepSeek adopted** (§3.2.1, two sentences quoted in full):

- "Historical reasoning content is discarded only when a new user message is introduced to the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is retained throughout the interaction."
- "When reasoning traces are removed, the history of tool calls and their results remains preserved in the context."

A consequence the report draws out: frameworks that simulate tool interactions as user messages (Roo Code, Terminus are named) trigger the discard on every tool result, so DeepSeek recommends non-thinking mode with those harnesses. This is a harness-level bug with no error message, in the same family as the template mismatches of ch-04.

**Evidence for the size of the effect.** [[minimax-m2-interleaved-thinking]] reports the same model under the two conditions:

| Benchmark | Prior thinking retained | Prior thinking discarded | Difference |
|---|---|---|---|
| SWE-Bench Verified | 69.4 | 67.2 | +2.2 |
| τ² | 87 | 64 | +23 |
| BrowseComp | 44.0 | 31.4 | +12.6 |
| GAIA | 75.7 | 67.9 | +7.8 |
| xBench | 72.0 | 66.0 | +6.0 |

**Result (single study)**, and a weak one in its reporting: the post gives no harness, no run count, no tool budget and no table caption. It is **official** for MiniMax-M2 and nothing more. The ordering across benchmarks is informative even so: the smallest gap is on SWE-Bench Verified, where the file system carries state between turns, and the largest are on τ² and BrowseComp, where the state lives only in the conversation.

**Conditions and limits.** Retention interacts with the window. Keeping every reasoning block is what makes the trajectory long in the first place. [[kimi-k2-5]] resolves the tension by choosing which block type to sacrifice: under Hide-Tool-Result, "only the most recent round of tool messages (observations and return values) is retained, while the reasoning chain and thinking processes from all previous steps are preserved in full". [[mirothinker]] §3.3 makes the same choice with a k-round window: tool responses outside the last K rounds are replaced with nothing, while thoughts and actions are never masked; K = 5 at evaluation, and the authors state this "does not lead to degradation in performance" without giving an ablation.

**Implication for a general-purpose model.** Reasoning blocks are the agent's own summary of what it has learned; observations are raw evidence. When the budget forces a choice, the four independent reports above all keep the derived text and drop the raw text. That is **Replicated** as a design choice and **not** as a controlled measurement — none of them ablates the reverse.

## §4 Heuristic policies with measured effects

### 4.1 DeepSeek-V3.2: three strategies at one trigger

[[deepseek-v3.1]] §4.4 defines the trigger and the strategies: context management fires "when the token usage exceeds 80% of the context window length", and the three strategies are (1) **Summary**, which summarizes the overflowed trajectory and re-initiates the rollout; (2) **Discard-75%**, which discards the first 75% of the tool-call history; (3) **Discard-all**, which resets the context by discarding all previous tool-call history — the report notes this is "similar to the new context tool (Anthropic, 2025a)", the feature described in [[anthropic-context-management]]. A parallel baseline, Parallel-fewest-step, samples N independent trajectories and keeps the one with fewest steps.

Reported results on BrowseComp: 51.4 without context management; Summary "extends the average steps to 364, achieving a performance improvement of up to 60.2"; Discard-all reaches 67.6, "comparable to parallel scaling while using significantly fewer steps" (§4.1, §4.4, Figure 6).

The report draws the methodological conclusion itself: "test-time compute can be scaled either serially through context management or in parallel ... Thus, it is crucial to account for actual compute costs when benchmarking model performance."

### 4.2 GLM-5: fold old observations, then reset

[[glm-5]] §4.2.4 keeps reasoning and actions and folds only stale observations:

```
o_i  ←  "Tool result is omitted to save tokens."      for i = 1, …, n − k
```

with k = 5. Reported effect on BrowseComp: 55.3% without keep-recent-k, 62.0% with it. The report adds that other values of k, and triggering on a token threshold instead of a round count, "leads to the same results" — no sweep is printed, so this is an assertion, not a measurement.

The hybrid, *Hierarchical Context Management*, runs keep-recent-k continuously and additionally performs a Discard-all when the total context exceeds `T = 32k`, "selected via parameter search". Against Discard-all alone it gives "consistent gains across all budgets, reaching a final score of 75.9".

### 4.3 The cross-model table

[[glm-5]] Table 7 reports the same benchmark twice for six models, which is the closest thing in the literature to a controlled comparison of "with" against "without":

| Model | BrowseComp | BrowseComp with context management | Difference |
|---|---|---|---|
| GLM-5 | 62.0 | 75.9 | +13.9 |
| GLM-4.7 | 52.0 | 67.5 | +15.5 |
| DeepSeek-V3.2 | 51.4 | 67.6 | +16.2 |
| Kimi K2.5 | 60.6 | 74.9 | +14.3 |
| Claude Opus 4.5 | 37.0 | 57.8 | +20.8 |
| Gemini 3 Pro | 37.8 | 59.2 | +21.4 |

The DeepSeek and Kimi pairs match the numbers those two reports give for themselves ([[deepseek-v3.1]] §4.4; [[kimi-k2-5]] §5), so the effect is **Replicated** across three independent reports. The Claude and Gemini rows were produced by the GLM-5 team, not by the model owners, and carry the usual third-party-evaluation caveats. Three further limits apply.

First, the "without" conditions are not identical: GLM-5 states that without context management it retains details from the most recent 5 turns ([[glm-5]] App. B.2), while [[kimi-k2-5]] counts a context overflow as an outright failure, and [[deepseek-v3.1]] §4.1 reports 51.4 as the score at a 128K limit where "20%+ of the test cases exceed this limit".

Second, the report is not internally consistent about what produced its own 75.9. §4.2.4 attributes that score to Hierarchical Context Management, which is keep-recent-k plus a Discard-all at `T = 32k`; App. B.2 describes the with-context-management column as "the same discard-all strategy as DeepSeek-V3.2 and Kimi K2.5". The two descriptions cannot both be complete, and the report does not print a Discard-all-only number for GLM-5 to separate them. Treat the 13.9-point GLM-5 gap as measured and its decomposition into keep-recent-k and Discard-all as **not reported**.

Third, the step budgets differ between conditions by construction — context management exists to allow more steps — so part of every gain in this table is additional serial test-time compute, not better use of the same compute.

### 4.4 Sub-agents as a context policy

[[kimi-k2-5]] treats multi-agent orchestration as a context-management architecture rather than a separate topic: sub-agents "maintain independent working memories and perform local reasoning without directly mutating or contaminating the global context of the central orchestrator. Only task-relevant outputs — rather than full interaction traces — are selectively routed back." Reported: BrowseComp 78.4 for the Agent Swarm against 60.6 for the single agent, and a 3×–4.5× reduction in execution time to reach a target Item-F1 on WideSearch. The report does not separate the effect of parallel compute from the effect of context isolation, so the 17.8-point gain cannot be attributed to context management alone. [[anthropic-context-engineering]] describes the same pattern with a size for the return channel: a sub-agent may spend tens of thousands of tokens and return "often 1,000-2,000 tokens".

## §5 Learned context management

The heuristics above are applied at inference to a model that was never trained under them. The four methods here train the compression step itself. All four face the same structural problem — one reward at the end of a trajectory that has been cut into segments — and they differ in how they assign credit across the cut.

### 5.1 SUPO: summarization inside the MDP

**Mechanism** ([[supo-summarization-rl]] §3–§4, Algorithm 2), step by step:

1. Run the ordinary tool loop while `L_t = |(s_t, a_t, o_t)| < L`, where `L` is the summarization threshold.
2. When the length crosses `L` and fewer than `S` summaries have been issued, set `s_{t+1} := (s_t, v_sum)` — append the summarization instruction and **discard the last action-observation pair**.
3. The model emits a summary `a_t`; the next state is `s_{t+1} := (s₁, a_t)` — the original task prompt plus the summary. Increment the summary count.
4. Stop at `H` steps or `S` summaries.

Step 2's discard is a length-control device, not an accident. Without it the reachable context is `L + 2L_A + L_O + |v_sum|` (`L_A` an action's length, `L_O` an observation's); with it the bound is `L + |v_sum| + L_A`, which "ensures that the summary at the end of the trajectory is not clipped by the RL training context length due to a long observation before summarization" (§4.2).

**Gradient.** Theorem 3.2 decomposes the policy gradient of the long rollout into a sum over the `I^j + 1` summarized sub-trajectories, so each segment can be handed to an unmodified trainer as an ordinary trajectory. Segments are padded to `⌈N / B_mini⌉ × B_mini` with zero-mask dummy trajectories that do not affect the update.

**Advantage.** All segments of rollout `j` share one advantage computed inside the **rollout** group (Eq. 3). The ablated alternative (Eq. 4) normalizes inside the **trajectory** group, repeating the rollout's reward once per segment. The difference matters because rollouts that summarize more often contribute more segments, so the trajectory-group version weights long rollouts more heavily in the group statistics.

**Results** (held-out sets of 128 and 100 instances; [[supo-summarization-rl]] Table 1):

| Environment | Algorithm | Working context | Effective context | Accuracy after | Tool calls |
|---|---|---|---|---|---|
| CodeGym (Qwen2.5-32B-Instruct) | GRPO | 32K | 32K | 44.5% | 52.1 |
| CodeGym | SUPO | 4K | 32K (4K×8) | 47.7% | 54.7 |
| BrowseComp-Plus (Seed-OSS-36B-Instruct) | GRPO | 64K | 64K | 39.0% | 6.7 |
| BrowseComp-Plus | SUPO | 64K | 192K (64K×3) | 53.0% | 19.2 |

The CodeGym row is the sharper one: an agent whose working context is **4K** matches and exceeds a GRPO agent with 32K, at the same effective budget. **Result (single study)**, one seed, two environments.

**Extrapolation.** The 64K/192K-trained checkpoint keeps improving when evaluated with larger summary budgets than it was trained with: 35.0% at 64K×1, 53.0% at 64K×3 (the training setting), 56.0% at 64K×6, 59.0% at 64K×12, 60.0% at 64K×24 (App. C.2 Table 3). The base model over the same sweep goes 28.0 → 37.0, and the GRPO model 39.0 → 50.0, so the trained summarizer both starts higher and keeps a slope.

### 5.2 ReSum: advantage broadcasting over segments

[[resum]] keeps the agent unchanged and calls an external summary model when the history approaches the limit, which makes it applicable to released checkpoints without retraining them. Its RL variant, ReSum-GRPO, resolves credit assignment by broadcasting:

```
Â_g   = (R_g − mean({R_1, …, R_G})) / std({R_1, …, R_G}),   R_g ∈ {0, 1}
Â_g^(i) = Â_g        for every segment i = 1, …, n_g of rollout g
```

`R_g` is the binary correctness of the final answer extracted from the last segment; `G` is the group size; `n_g` is the number of segments rollout `g` was cut into. The stated purpose is "ensuring early exploration steps receive appropriate bonus when they contribute to the final success", and short trajectories that never summarize are handled identically to standard GRPO.

Results with 1K training samples and 4 epochs (Pass@1, [[resum]] Tables 1–2): WebSailor-3B goes 8.2% → 20.5% on BrowseComp-zh under ReSum-GRPO, while GRPO on the ReAct paradigm reaches only 11.8%; WebSailor-30B reaches 33.3% on BrowseComp-zh. The paper's own summary of the two settings is "a 4.5% improvement over ReAct in training-free settings, with ReSum-GRPO yielding a further 8.2% gain". Applying GRPO to ReAct and then evaluating under ReSum does not transfer: "GRPO fails to enable agents to master summary-conditioned reasoning". A trained 30B summary model (ReSumTool-30B) matches or beats Qwen3-235B and DeepSeek-R1-671B used as summarizers on the same backbone (13.7% vs 11.1% and 13.0% on BrowseComp-zh with WebSailor-3B), which says the summarization role is a narrow skill rather than a test of general capability.

### 5.3 Context folding: a bounded main thread

[[context-folding]] replaces a flat summary with two tools: `branch(description, prompt)` opens a sub-context for a sub-task, and `return(message)` folds it — the interior action-observation pairs are removed and only a templated message stays in the main thread. At inference the KV-cache is rolled back to the branch position, where the prefix equals the pre-branch context, so the fold costs no re-prefill.

FoldGRPO adds token-level process rewards to the outcome reward inside the advantage, aimed at two failure modes the authors observed with outcome reward alone (token-heavy work left unfolded in the main thread, and branches never returned): `Q = −1` on main-thread tokens once the main thread exceeds 50% of the working-context limit, with the tokens of branch-creating turns exempt; `Q = −0.2` on all tokens of a branch that GPT-5-nano judges to have acted outside its sub-task; `Q = −1` on the tokens of a failed tool-call turn; `Q = 0` otherwise (§2.3.2).

Results at a 32K working context with up to 10 branches, pass@1 on BrowseComp-Plus / SWE-Bench Verified ([[context-folding]] Table 1): folding agent untrained 0.420 / 0.492; + GRPO 0.567 / 0.564; + FoldGRPO 0.620 / 0.580. The comparison that matters for this chapter is against context size: a 327K-context ReAct agent trained with GRPO on the same data reaches 0.540 / 0.574, and a 32K × 10 summary agent trained with GRPO reaches 0.527 / 0.550. Behaviour statistics (Table 2) show why: FoldGRPO raises the finish rate to 0.935 / 0.962 from 0.738 / 0.612 under plain GRPO, and cuts the main-thread length to 7,752 / 8,885 tokens from 22,285 / 48,908.

### 5.4 Memory as a trained skill

Two papers remove the trajectory entirely and keep a fixed-size state.

[[memagent]] streams a document through the model in chunks and **overwrites** a fixed-length memory after each chunk, so "the total compute per chunk stays O(1) and end-to-end complexity is strictly linear to the number of chunks". Training uses an 8K window split as 1024 query / 5000 chunk / 1024 memory / 1024 output, on 60K-token documents, with a multi-conversation extension of DAPO. On RULER-HQA the 7B model scores 81.25 at 7K and 71.88 at 3.5M tokens, while Qwen2.5-Instruct-7B-1M goes 61.72 → 0.00 by 896K (Table 1).

[[mem1]] applies the same idea to tool use: each turn consolidates the previous internal state, query and observation into a new internal state, and "all tags from the previous turn are pruned", leaving at most two `<IS>`, two `<query>` and one `<info>` in context. Trained only on 2-objective tasks, it reaches EM 1.97 on 16-objective tasks against 0.567 for Qwen2.5-14B-Instruct, using 10.4 × 10² peak tokens against 38.4 × 10² — "27.1% of the peak tokens and 29.3% of the total inference time" (§4.2, Table 1). The paper also reports that "SFT significantly underperforms RL" for acquiring this behaviour.

The cost side is reported by a third party: [[resum]] App. F.3 finds MEM1-GRPO "consumes nearly 3× more tokens than ReSum for a mere 1.2% improvement" on BrowseComp, because consolidating at every turn is more generation than summarizing at a threshold.

### 5.5 What the four share

| Method | What is compressed | Trigger | Credit assignment across the cut | Control on failures |
|---|---|---|---|---|
| SUPO | whole history → summary | context ≥ 95% of working length | one rollout-group advantage shared by all segments | overlong mask on rollouts that never answer |
| ReSum | whole history → summary from an external tool | history approaches the limit | trajectory advantage broadcast to all segments | none reported; short rollouts unchanged |
| FoldGRPO | sub-trajectory → return message | agent-issued `branch` / `return` | token-level process rewards added inside the advantage | `clip(R_i + Q, 0, 1)` leaves failed rollouts unpenalized |
| MEM1 / MemAgent | everything → fixed-size state | every turn / every chunk | ordinary group advantage per conversation | not reported |

The common pattern: one binary reward, many segments, and a decision about whether a segment's tokens inherit the whole rollout's fate. That decision is where the next section applies.

## §6 Worked example: how many turns fit, and what each reset costs

Take `W = 128,000`, `s = 1,500`, `r = 400`, `a = 80`, `o = 1,500`, so one full turn costs `r + a + o = 1,980` tokens. All numbers below are checkable by hand.

**Keep everything.** `n = ⌊(128,000 − 1,500) / 1,980⌋ = ⌊63.88⌋ = 63` turns, then the run stops. Total history processed: `63 × 1,980 = 124,740` tokens.

**keep-recent-k with k = 5.** After `n` turns the context is `1,500 + n(400 + 80) + 5 × 1,500 + (n − 5) × 12`, taking 12 tokens for the placeholder string. Setting this equal to 128,000: `1,500 + 480n + 7,500 + 12n − 60 = 128,000`, so `492n = 119,060` and `n = 241`. The agent completes 241 turns instead of 63 — a 3.8× longer horizon at the same window — and it has read `241 × 1,980 = 477,180` tokens of observation, of which `236 × 1,500 = 354,000` are no longer visible. This arithmetic is the reason the k-round rule is the cheapest intervention available: it costs one line of harness code and no model change.

**Discard-all at 80%.** The trigger is at `0.8 × 128,000 = 102,400` tokens. Starting from `s = 1,500`, the context crosses the trigger after `⌈(102,400 − 1,500) / 1,980⌉ = ⌈50.96⌉ = 51` turns, then resets to 1,500. Each subsequent cycle is 51 turns. Reaching 364 steps — the average [[deepseek-v3.1]] §4.4 reports for its Summary strategy — takes about `364 / 51 ≈ 7` resets. Nothing from cycle 1 is visible in cycle 7 except what the model rewrote into its answer or into an external file.

**Summary at 80% with an 800-token summary.** Same cycle length, but each cycle restarts at `1,500 + 800 = 2,300` tokens, so cycles after the first hold `⌈(102,400 − 2,300) / 1,980⌉ = 51` turns as well, and each reset costs an extra generation of 800 tokens plus the prefill of the summarization prompt. Over 7 resets that is 5,600 extra generated tokens — small next to the history, which is why [[deepseek-v3.1]] describes Summary as effective but of "relatively low" efficiency: the cost is not the summary tokens, it is that the summarization call itself consumes a step and re-prefills a new prefix.

**Reading the arithmetic.** Every policy in the table of §2 buys turns with information. keep-recent-k gives up old observations and keeps the reasoning that referenced them; Discard-all gives up everything including the reasoning; Summary gives up detail but keeps a model-chosen trace. The figure at [figures/context-budget.html](figures/context-budget.html) lets you vary `W`, `o`, `k` and the trigger and watch the three curves cross.

## §7 Cost, KV-cache, and compaction as a trained behaviour

Agentic inference is prefill-dominated. [[manus-context-engineering]] reports an average input-to-output token ratio near 100:1 in production, so the per-step cost is set by how much of the prefix can be served from cache. Three operational rules follow, each with a stated mechanism:

1. **Stable prefix.** One changed token invalidates the cache from that position onward; a timestamp at the top of the system prompt is named as the frequent cause.
2. **Append-only context with deterministic serialization.** Unstable JSON key ordering silently breaks the cache.
3. **Mask, do not remove, tools.** Removing a tool definition invalidates the cache for everything after it, and leaves earlier actions in the history referring to tools that no longer exist, which "often leads to schema violations or hallucinated actions". Manus instead constrains decoding by prefilling the response up to a shared tool-name prefix (`browser_`, `shell_`).

Rule 2 is in direct tension with every policy in §2, all of which rewrite the context. The tension is resolved by *where* the rewrite happens: [[context-folding]] §2.2 rolls the KV-cache back to the branch position, where the prefix is byte-identical to the pre-branch context, so a fold costs nothing to re-prefill; a Discard-all, by contrast, discards the cache along with the history. This is a reason to prefer resets at a small number of well-defined positions over continuous rewriting.

Compaction is also now a trained behaviour rather than a harness feature. [[openai-codex-max-compaction]] describes GPT-5.1-Codex-Max as "our first model natively trained to operate across multiple context windows through a process called *compaction*", which "automatically compacts its session when it approaches its context window limit, giving it a fresh context window. It repeats this process until the task is completed", with internal observations of tasks running "for more than 24 hours". The post reports 30% fewer thinking tokens than GPT-5.1-Codex at medium reasoning effort on SWE-bench Verified, and an appendix table comparing GPT-5.1-Codex (high) with GPT-5.1-Codex-Max (xhigh): 73.7% → 77.9% on SWE-bench Verified, 66.3% → 79.9% on SWE-Lancer IC SWE, 52.8% → 58.1% on Terminal-Bench 2.0. Those two columns differ in both model and reasoning effort, so they do not isolate compaction; treat the compaction claim as **official but unablated**.

The platform-side numbers have the same shape. [[anthropic-context-management]] reports, on an internal agentic-search evaluation, "39%" improvement from the memory tool combined with context editing and "29%" from context editing alone, and in a 100-turn web-search evaluation an 84% reduction in token consumption while completing workflows that would otherwise fail. The evaluation set, baseline, metric and run count are not disclosed, and the post does not state whether 39% and 29% are relative or absolute. It is usable as evidence that the direction is positive and not as a magnitude.

## Negative samples and negative feedback

Context management produces negatives in three distinct senses of §6.1 of the authoring standard, and confusing them is the source of most of the instability reported in these papers.

**Where the negatives come from.** A long-horizon rollout can fail in three ways: a wrong final answer (verified by a judge or unit tests), a run that exhausts `H` steps or `S` summaries without producing an answer, and a turn whose tool call is malformed. The first is labeled by the environment; the second is labeled by the harness; the third is labeled syntactically. False-negative rates are not reported by any source in this chapter.

**What current practice does with each.**

1. **Negative as gradient (sense 4).** A wrong final answer gives `R_g = 0`, which becomes a below-mean group advantage in GRPO ([[grpo]]) and — in [[resum]] and [[supo-summarization-rl]] — is applied to *every segment of the rollout*, including the summarization tokens. This is the specific hazard of segmented training: a rollout can fail for reasons unrelated to its summaries, and the broadcast lowers the likelihood of those summary tokens anyway. The softmax logit gradient `∂ log p_y / ∂ z_j = 1[j = y] − p_j` says where the removed mass goes: it is redistributed in proportion to the current probabilities, so pushing down a competent summary concentrates mass on whatever the model would otherwise have written, which on a long compressed history is frequently a shorter and less informative summary.
2. **Negative as gradient, localized.** [[context-folding]] §2.3.2 attaches token-level penalties instead of trajectory-level ones: `Q = −1` on unfolded main-thread tokens above the 50% threshold except in branch-creating turns, `−0.2` on out-of-scope branch tokens, `−1` on failed tool-call turns. The advantage is `Â_{i,t} = (clip(R_i + Q_{i,t}, 0, 1) − mean({R_i})) / std({R_i})`. Reading the clip: when `R_i = 0` a negative `Q` cannot push the sum below 0, so penalties change advantages **only inside successful trajectories**. The design deliberately declines to penalize a failed rollout twice.
3. **Negative as discarded sample (sense 1).** [[supo-summarization-rl]] §4.2 masks rollouts that reach `H` or `S` without answering: "Without masking, the objective could be biased towards suppressing long rollout that exhibits good summarization strategies despite its failure to provide answers within step or trajectory limits. This could further lead to collapse of summarization patterns in essentially long-horizon tasks." The ablation is the evidence: removing the mask drops BrowseComp-Plus from 53.0% to 44.0% and CodeGym from 47.7% to 45.3% (Table 1), and the authors note the unmasked BrowseComp-Plus run was stopped at 3 epochs instead of 5 "for its degenerated performance". [[mirothinker]] applies the same idea at the data level, dropping incorrect trajectories that fail on trivial format errors or show action loops rather than penalizing them.
4. **Negative as content (sense 2).** [[manus-context-engineering]] argues the opposite move at inference: "leave the wrong turns in the context. When the model sees a failed action — and the resulting observation or stack trace — it implicitly updates its internal beliefs." No measurement is attached, so this is **anecdotal** and cannot support a quantitative claim; it is, however, in direct conflict with aggressive observation dropping, since the stack trace is an observation. A policy that folds all old observations removes exactly the evidence this recommendation depends on.

**Controls that make the negatives safe here.** Mask non-terminating rollouts rather than scoring them zero ([[supo-summarization-rl]]); clip the process penalty so it cannot compound with an outcome failure ([[context-folding]]); keep the advantage normalization inside the rollout group rather than the segment group, so that rollouts producing more segments do not dominate the group statistics (Eq. 3 vs Eq. 4 in [[supo-summarization-rl]], worth 47.7% vs 42.1% on CodeGym and 53.0% vs 49.0% on BrowseComp-Plus); and mask tool-observation tokens from the loss, which is the ch-45b rule and is restated in [[context-folding]] §2.3.1.

**Diagnostics.** Log the summarization rate `p_summary = #rollouts with a summary / #rollouts` and the conditional success rate `p_success|summary = #successful rollouts with a summary / #rollouts with a summary` separately; [[supo-summarization-rl]] §5.2.2 tracks both and reports that the second rises through training, which is the signal that the summarization skill itself is improving rather than the agent avoiding summarization. Add the finish rate and the mean main-thread length from [[context-folding]] Table 2, and per-advantage-sign token counts from ch-45b.

**Effect on generality.** The measured risk is the collapse described in [[supo-summarization-rl]] §4.2: a badly signed penalty removes the behaviour rather than improving it, and the observable is the summarization rate going to zero while the working context saturates. None of the sources in this chapter reports pass@k at large k or a general-capability panel for context-managed agents, so the coverage question of ch-43 (entropy, output diversity, and pass@k under RL) is **Open** for this stage.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3.2 | 671B total / 37B activated (V3.1 model card) | eval-gate | context-management trigger | token usage > 80% of context window length | arXiv:2512.02556 §4.4 | verified 2026-09-15 | Fig. 6 compares Summary, Discard-75%, Discard-all, Parallel-fewest-step on BrowseComp |
| DeepSeek-V3.2 | same | eval-gate | BrowseComp without / with Discard-all | 51.4 / 67.6 | §4.1, §4.4 | verified 2026-09-15 | Fig. 6 (accuracy vs real steps) |
| DeepSeek-V3.2 | same | eval-gate | Summary strategy: average steps, best score | 364 steps; up to 60.2 | §4.4 | verified 2026-09-15 | Fig. 6 |
| DeepSeek-V3.2 | same | SFT/RL template | reasoning retention in tool use | reasoning discarded only on a new user message; tool-call history always kept | §3.2.1, Fig. 4 | verified 2026-09-15 | no ablation reported |
| GLM-5 | not reported in the cited sections | eval-gate | keep-recent-k | k = 5; observations older than the last k rounds replaced by a placeholder string | arXiv:2602.15763v2 §4.2.4 | verified 2026-09-15 | BrowseComp 55.3% → 62.0% |
| GLM-5 | same | eval-gate | Hierarchical Context Management threshold T | T = 32k, selected by parameter search | §4.2.4 | verified 2026-09-15 | final BrowseComp 75.9 vs Discard-all alone (Fig. 8) |
| GLM-5 | same | eval-gate | BrowseComp judge | official OpenAI evaluation prompt, o3-mini as judge | §4.2.4 | verified 2026-09-15 | case studies showing sensitivity to judge prompt and model |
| Kimi K2.5 | not reported in the cited sections | eval-gate | HLE-with-tools policy | Hide-Tool-Result: keep only the most recent round of tool messages; keep all reasoning | App. "Context Management Strategies" | verified 2026-09-15 | no ablation reported |
| Kimi K2.5 | same | eval-gate | BrowseComp without / with Discard-all | 60.6 / 74.9 | §5 prose (the 60.6 also appears in Table 6; 74.9 is not in a table) | verified 2026-09-15 | Table 6 against single-agent and proprietary baselines |
| MiroThinker-v1.0 | 8B / 30B / 72B | eval-gate | recency retention K | K = 5 most recent tool responses; thoughts and actions always kept | arXiv:2511.11793 v3 §3.3, §6.1 | verified 2026-09-14 (card) | authors state no degradation; no ablation table |
| SUPO on Qwen2.5-32B-Instruct (CodeGym) | 32B | RL | working context / max summaries / effective | 4K / S = 7 / 32K | arXiv:2510.06727v1 §5.1 | verified 2026-09-15 | Table 1: 47.7% vs GRPO 44.5% at equal effective context |
| SUPO on Seed-OSS-36B-Instruct (BrowseComp-Plus) | 36B | RL | working context / max summaries / effective | 64K / S = 2 / 192K | §5.1 | verified 2026-09-15 | Table 1: 53.0% vs GRPO 39.0% |
| SUPO (both) | 32B / 36B | RL | summarization threshold L; max steps H | L = 95% of working context length; H = 100 | §5.1 | verified 2026-09-15 | no ablation of L or H reported |
| SUPO (both) | 32B / 36B | RL | group size; clip; LR; KL and entropy | G = 8; ε_low 0.20, ε_high 0.28; 1 × 10⁻⁶ constant; no KL term, no entropy term | §5.1 | verified 2026-09-15 | no ablation reported |
| ReSum-GRPO on WebSailor-3B/-7B/-30B | 3B / 7B / 30B | RL | training samples; epochs; context; tool budget | 1K samples; 4 epochs; 32k window; 60 tool calls | arXiv:2509.13313v3 §4.1, §4.3 | verified 2026-09-15 | Table 2 against GRPO and MEM1-GRPO |
| ReSum (inference) | any WebSailor backbone | eval-gate | summary tool and trigger | ReSumTool-30B, invoked as history approaches the context limit (4k prompt + 28k responses) | §4.1, App. D | verified 2026-09-15 | Table 1: matches or beats Qwen3-235B and DeepSeek-R1-671B as summarizers |
| Folding agent on Seed-OSS-36B-Instruct + FoldGRPO | 36B | RL | working context; branches | 32,768 tokens; up to 10 branches (327,680 theoretical) | arXiv:2510.11967v1 §3.2 | verified 2026-09-14 (card) | Fig. 5: plateau beyond 320K |
| same | 36B | RL | process penalties | −1 unfolded main-thread tokens above 50% of the limit; −0.2 out-of-scope branch; −1 failed tool call | §2.3.2 | verified 2026-09-14 (card) | Tables 1–2: 0.620/0.580 vs GRPO 0.567/0.564 with finish rate 0.935/0.962 vs 0.738/0.612 |
| RL-MemAgent-7B / -14B | 7B / 14B | RL | training window allocation | 8K total: 1024 query, 5000 chunk, 1024 memory, 1024 output; 60K-token documents | arXiv:2507.02259v2 §4 | verified 2026-09-15 | Table 1: 71.88% at 3.5M tokens for the 7B model |
| MEM1-7B | 7B | RL | retained context per turn | at most two `<IS>`, two `<query>`, one `<info>` | arXiv:2506.15841v2 §3.1 | verified 2026-09-15 | Table 1: EM 1.97 at 16 objectives with 10.4 × 10² peak tokens |
| Claude Sonnet 4.5 context editing + memory tool | not reported | eval-gate | trigger threshold; cleared block types; eval set | not reported (checked the announcement post only) | anthropic.com/news/context-management | not reported | "39%" and "29%" improvements and 84% token reduction, with no baseline or metric definition given |

**Starting point for a small general-purpose run.** With a released instruction model and no training budget, the configuration supported by verified rows above is: keep interleaved reasoning in the history and send tool results as tool messages rather than user messages ([[deepseek-v3.1]] §3.2.1, 671B MoE, evaluated on agentic benchmarks); fold tool observations older than the last 5 rounds ([[glm-5]] §4.2.4, measured on BrowseComp with a 128K-class model); and add a full reset when the total context passes 32k, which is the threshold GLM-5 selected by parameter search for its own model and harness. Every one of these numbers was tuned on a frontier-scale model with web-search tools and a judge-scored benchmark; the k = 5 and T = 32k values have no reported transfer to other model sizes or task families, so re-select them on your own evaluation before treating them as defaults.

## Generalization lens

**(a) What increases breadth.** Training the compression step rather than applying it at inference generalizes past the trained budget in every report that tested it: SUPO's 64K/192K checkpoint keeps improving out to 64K×24 (60.0% vs 53.0% at the training setting; [[supo-summarization-rl]] App. C.2), MemAgent trained at 8K holds 71.88% at 3.5M tokens ([[memagent]] Table 1), MEM1 trained on 2-objective tasks improves as objectives rise to 16 while baselines collapse ([[mem1]] Table 1), and the folding agent trained with at most 10 branches uses an average of 32.6 branches on 50-question compound tasks ([[context-folding]] §4.4.2). A second breadth effect is architectural: [[resum]]'s summary tool is external, so it upgrades released agents of three different sizes without retraining them (Table 1).

**(b) What causes narrowing or forgetting.** Three measured mechanisms. First, irreversible compression discards evidence whose value appears later; [[manus-context-engineering]] states the argument and builds restorable compression instead, and [[anthropic-context-engineering]] names the same risk for compaction prompts tuned for precision before recall. Second, penalizing non-terminating rollouts removes the summarization behaviour itself — the mask ablation costs 9 points on BrowseComp-Plus and the unmasked run degraded enough to be stopped at 3 epochs instead of 5 ([[supo-summarization-rl]] §4.2, §5.1, Table 1). Third, uniform context induces uniform behaviour: [[manus-context-engineering]] reports agents falling into repeated action patterns when the context is filled with similar action-observation pairs, and injects structured variation as the countermeasure — which is a narrowing of the output distribution caused purely by context construction, with no gradient involved. A fourth is reported but not measured: [[kimi-k2-5]] states that reactive truncation "often sacrifice[s] structural information or intermediate reasoning" without quantifying the loss.

**(c) How to measure it for this stage.** Four requirements, each with a source that violated or satisfied it.
- **Report both conditions.** With and without context management, on the same checkpoint and the same judge ([[glm-5]] Table 7 does this for six models).
- **Hold the step budget fixed, or plot against it.** [[deepseek-v3.1]] Figure 6 and [[glm-5]] Figure 8 both plot accuracy against real steps, which is the only presentation that separates "better context policy" from "more serial compute".
- **Use a held-out split and say what was held out.** [[supo-summarization-rl]] holds out 100 of 830 BrowseComp-Plus questions and 128 CodeGym problems from different seed problems; [[resum]] holds out by benchmark. Neither reports a non-agentic capability panel, so retention of general ability after context-management RL is unmeasured.
- **Score abstention separately from wrong answers.** [[chroma-context-rot]] shows the two failure modes move differently with length and differ by model family; an agentic score that merges them cannot tell "the policy discarded the evidence" from "the policy discarded the evidence and the model asserted an answer anyway".

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing agentic scores across reports without checking the context policy | Two models 15 points apart on BrowseComp with no other difference | Read the evaluation appendix for the trigger and strategy; compare only like with like ([[glm-5]] Table 7 has both columns) |
| Harness sends tool results as user messages while the model is in thinking mode | Token counts per turn rise steadily; the model re-derives its plan in every reasoning block | Inspect the serialized request: are tool results `role: tool` or `role: user`? ([[deepseek-v3.1]] §3.2.1) |
| Prior reasoning silently dropped by an OpenAI-compatible client | Long-horizon benchmarks fall while single-turn benchmarks are unchanged | Confirm the client returns and resends the reasoning field; compare one retained vs discarded run ([[minimax-m2-interleaved-thinking]]) |
| Training with a context policy the agent will not have at inference, or the reverse | Training reward rises, evaluation does not move | Evaluate the training-time policy and the deployment policy separately; [[resum]] Table 2 shows GRPO-on-ReAct failing to transfer to ReSum inference |
| Non-terminating rollouts scored as failures | Summarization rate falls toward zero during training; working context saturates | Log `p_summary` and mask step/summary-limit rollouts ([[supo-summarization-rl]] §4.2) |
| Advantage normalized over segments rather than rollouts | Rollouts with more summaries dominate the group statistics; short rollouts get small advantages | Compare Eq. 3 and Eq. 4 forms on a held-out split (47.7% vs 42.1% on CodeGym) |
| Dynamic tool sets added and removed mid-run | Cache hit rate drops; schema violations and calls to undefined tools appear | Keep the tool list fixed and mask logits instead ([[manus-context-engineering]]) |
| Timestamp or nondeterministic serialization at the top of the prompt | Cache hit rate near zero with no functional change | Diff consecutive request prefixes byte for byte ([[manus-context-engineering]]) |
| Attributing a benchmark gain to compaction when the comparison changes two things | A table whose two columns differ in model and in reasoning effort | Require a with/without comparison on one checkpoint ([[openai-codex-max-compaction]] does not provide one) |
| Folding observations while relying on error traces for self-correction | Repeated identical failed actions after the fold point | Keep failed tool-call observations outside the fold window, or keep a restorable pointer ([[manus-context-engineering]] vs [[glm-5]]) |

## Check your understanding

1. GLM-5's keep-recent-k folds observations and keeps reasoning blocks. MiniMax reports that keeping reasoning blocks is worth 12.6 points on BrowseComp. Explain why these two findings are consistent rather than contradictory, in terms of which block type carries state forward.
2. In the worked example of §6, keep-recent-k extended the horizon from 63 to 241 turns. Derive what changes if observations are 6,000 tokens instead of 1,500, and explain why the ratio between the two policies moves in the direction it does.
3. [[supo-summarization-rl]] computes the advantage inside the rollout group rather than the trajectory group, and the trajectory-group variant scores 5.6 points lower on CodeGym. Give the causal account: what property of the trajectory-group statistics changes when some rollouts produce eight segments and others produce one?
4. [[context-folding]]'s advantage uses `clip(R_i + Q_{i,t}, 0, 1)`. Derive what happens to a process penalty inside a failed rollout, and explain why the authors would want that behaviour rather than letting penalties stack on failures.
5. A colleague reports that their agent scores 74 on BrowseComp and a competitor's scores 61, concluding their model is better. List the three specific configuration facts you would need before accepting the comparison, and say which published table you would point to as precedent for each.
6. [[manus-context-engineering]] recommends keeping failed actions in the context; [[glm-5]] folds all observations older than five rounds. Construct a task on which these two recommendations give different behaviour, and say which observable would tell you which one is right for that task.
7. [[resum]] finds that plain GRPO on ReAct rollouts does not transfer to ReSum inference. Explain the mismatch in terms of the state distribution the policy gradient was computed over.
8. MemAgent trains at an 8K window and holds 71.88% accuracy at 3.5M tokens, while a 1M-context model trained for long context reaches 0.00% at 896K. State what this comparison does and does not establish about long-context architecture work.

## Connections

- **Previous:** ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability. Supplies the loss-masking rule and the turn-level credit assignment that every method here modifies for segmented trajectories.
- **Dependency:** ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation. Supplies the measurement that motivates this chapter: a stated window is not usable to its stated size.
- **Next:** ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL. Places the context policies of this chapter inside the full recipes of the models that reported them.

## Sources

- [[chroma-context-rot]] — 18-model measurement of accuracy against input length with the task held fixed; the LongMemEval focused-vs-full comparison and the abstention-versus-hallucination split used in §1 and the Generalization lens.
- [[context-length-alone-hurts]] — length-driven degradation with retrieval success controlled; corroborates §1.
- [[lost-in-the-middle]] — position dependence of retrieval inside a long context; background for §1 and for the recitation rule in §7.
- [[anthropic-context-engineering]] — definitions of context engineering and the attention budget; compaction, structured note-taking and sub-agent architectures as the three long-horizon techniques; the 1,000–2,000-token sub-agent return size.
- [[anthropic-context-management]] — context editing and the memory tool as platform features, with the 39% / 29% / 84% figures and their missing baseline; the "new context tool" that DeepSeek's Discard-all resembles.
- [[minimax-m2-interleaved-thinking]] — retained-versus-discarded prior-round thinking on five agentic benchmarks (§3).
- [[deepseek-v3.1]] — the V3.2 report: the thinking-retention rule for tool calls (§3.2.1), the 80%-of-window trigger, the Summary / Discard-75% / Discard-all comparison, and the 51.4 → 67.6 BrowseComp pair (§4.1, §4.4).
- [[glm-5]] — keep-recent-k with k = 5 and its 55.3 → 62.0 effect, Hierarchical Context Management with T = 32k, the judge standardization, and the six-model with/without table (§4.2.4, §6.1.3, Table 7).
- [[kimi-k2-5]] — Hide-Tool-Result for HLE, Discard-all for BrowseComp with the 60.6 → 74.9 pair, the overflow-counts-as-failure default, and Agent Swarm framed as proactive context management (§5, appendix).
- [[supo-summarization-rl]] — the summarization-augmented MDP, the gradient decomposition, the rollout-group advantage, the overlong mask and its ablation, and the test-time summary-budget sweep (§3–§5, Tables 1 and 3).
- [[resum]] — external summary tool, advantage broadcasting, ReSumTool-30B, the 1K-sample RL results, the MEM1 token-cost comparison, and the 32k–128k context sweep (§3–§4, Tables 1–3).
- [[context-folding]] — branch and return tools, KV-cache rollback at the fold, FoldGRPO's token-level process rewards with the clip, and the 32K × 10 versus 327K ReAct comparison (§2–§4, Tables 1–2).
- [[memagent]] — fixed-size overwritten memory trained with multi-conversation DAPO; the 8K training window allocation and the RULER-HQA table out to 3.5M tokens.
- [[mem1]] — constant-memory consolidation with `<IS>` tags; the 2-objective to 16-objective generalization and the peak-token comparison.
- [[manus-context-engineering]] — KV-cache hit rate as the operational metric, the 100:1 input-to-output ratio, mask-do-not-remove for tools, restorable compression, recitation, and keeping failures in context (practitioner evidence, no controlled experiment).
- [[openai-codex-max-compaction]] — compaction as a natively trained multi-window capability, the 30%-fewer-thinking-tokens claim, and the unablated appendix table.
- [[mirothinker]] — recency-based tool-output retention with K = 5 while thoughts and actions are kept, at 600 tool calls in a 256K window.
- [[browsecomp-plus]] — the retrieval-grounded BrowseComp corpus used as the training and evaluation environment by [[supo-summarization-rl]] and [[context-folding]].
- [[grpo]] — the group-relative advantage that every segmented method in §5 modifies.
