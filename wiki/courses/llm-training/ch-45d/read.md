<!-- chapter: ch-45d
     track: rl
     kind: content
     title: Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL
     deps: [ch-45c, ch-35]
     sources: [[kimi-k2]], [[kimi-k2-recipe]], [[kimi-k2-agentic-data]], [[kimi-k2-5]], [[kimi-k3]], [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[deepseek-v3.1]], [[deepseek-v3.1-recipe]], [[qwen3-coder]], [[qwen3-coder-next]], [[minimax-m2-aligning-to-what]], [[minimax-forge-agent-rl]], [[nemotron-3-super]], [[tongyi-deepresearch]], [[deepswe]], [[deepswe-recipe]], [[toucan-mcp]], [[agent-data-protocol]], [[swe-smith]], [[r2e-gym]], [[swe-gym]], [[grpo]], [[rlvr-beyond-base-model]]
     figures: figures/agentic-stage-map.html
     revised: 2026-09 (generality revision)
-->

# Chapter 45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL

> **Core insight.** Open agentic recipes disagree about almost every hyperparameter but agree on
> three placements. First, agentic trajectories enter the base model at **mid-training**, not at
> pre-training: [[glm-4-5]] adds 100B tokens of agent trajectories at 128K after 22T tokens of
> pre-training (§2.3, Figure 3), [[glm-5]] extends that to a 32K/128K/200K mid-training block of
> 1.55T tokens inside a 28.5T budget (§2.3), [[qwen3-coder-next]] puts about 600B repository-level
> tokens there (§3.1.1), and [[nemotron-3-super]] lists no agentic corpus in its 25T-token
> pre-training data at all (§2.3). Second, the **RL environment count**, not the RL algorithm, is
> what the reports scale: [[deepseek-v3.1]] builds 1,827 synthesized general-agent environments
> (§3.2.3) and 85,267 agent prompts (the four task counts in Table 1 sum to that figure),
> [[glm-5]] over 10k verifiable software-engineering environments
> across nine languages (§4.2.1), [[qwen3-coder-next]] about 800K verifiable task instances (§2.1),
> and [[tongyi-deepresearch]] states outright that agentic RL "depends more on the quality of the
> data and the stability of the training environment than on the specific algorithm being used"
> (§3.4.3). Third, **a model trained on one agent harness does not transfer to another**:
> [[qwen3-coder-next]] measures limited cross-scaffold transfer (§3.1.2, Figure 3),
> [[minimax-m2-aligning-to-what]] reports benchmark scores that "plummet" after a scaffold swap, and
> [[kimi-k3]] answers it by randomizing the harness during RL (§4.2.1).
>
> **Guideline.** When agentic capability must coexist with general capability, train one mixed RL
> stage over all environments rather than a sequence of single-environment stages, because
> [[nemotron-3-super]] reports that "training on all environments simultaneously yields stable gains,
> whereas single-environment training leads to severe regressions on other benchmarks" (§3.2.1) and
> [[agent-data-protocol]] measures the SFT analogue: a mixed 13-dataset corpus reaches 10.4% on
> SWE-Bench Verified against 1.0% for SWE-smith-only tuning at 7B, and 9.1% on GAIA against 0.6% for
> AgentInstruct-only tuning (§6.2, Table 6). When a stage has to be separated for throughput reasons
> — long software-engineering rollouts co-trained with short-horizon environments starve the batch —
> separate it and then recover the other stages, either by an explicit distillation step
> ([[glm-5]] §3.5, [[kimi-k3]] §4.1.3) or by following the separated stage with a further stage whose
> reward covers the general objective again ([[nemotron-3-super]] separates SWE-RL for throughput in
> §3.2.2 and then runs an RLHF stage in §3.2.3 scored by the same generative reward model used during
> multi-environment RL). When only one harness is available for training, expect the
> resulting model to be harness-specific and say so, because no source reports a recipe that
> generalizes across scaffolds without training on several.

## Why this chapter matters for a general-purpose model

The preceding chapters isolate one stage each: agentic mid-training data in ch-32d, multi-skill SFT
mixtures in ch-30b, environment and task synthesis in ch-29c, multi-turn agentic RL mechanics in
ch-45b, and context management in ch-45c. This chapter reads eleven open recipes across all of those
stages at once and asks the question the single-stage chapters cannot answer: given a fixed budget,
where should agentic capability be built, and what does each placement cost in general capability?

An agentic recipe touches every stage of the pipeline: pre-training → mid-training → SFT →
preference optimization → RL → evaluation. That is the source of the generality risk. A model whose
agentic ability is added in one late RL stage on one harness has a narrow competence added to a broad
one, and only 3 of the 11 reports read here run a non-agentic evaluation before and after the agentic
stage (§7: [[kimi-k2-5]] Table 2, [[qwen3-coder-next]] §5.3, [[nemotron-3-super]] §3.2.1-§3.3).

The interactive cross-lab map at [figures/agentic-stage-map.html](figures/agentic-stage-map.html)
holds every cell of §1 with its locus; use it while reading §1-§4 to see which cells any given report
leaves empty.

## §1 The six places a recipe can put agentic capability

**Definition.** An *agentic stage placement* is the training stage at which a recipe first exposes
the model to full action-observation trajectories: a sequence alternating model actions (tool calls,
code, messages) with environment observations. The six candidate stages are pre-training,
mid-training, SFT, specialist or expert RL, the main RL stage, and a final distillation or merge.

**The measurable problem.** Each placement buys capability at a different price in tokens, in
verification effort, and in forgetting risk. Pre-training tokens are cheap per token but cannot be
verified; RL rollouts are verified but cost orders of magnitude more per sample. A recipe that puts
everything in RL pays the highest price per unit of capability; one that puts everything in
mid-training gets no verification signal at all.

**The cross-lab map.** Reading the eleven reports at their loci gives the following. A blank cell
means the report does not describe that stage for agentic data. Absence of a statement is not
evidence that the stage was skipped. One naming note: the library card [[deepseek-v3.1]] holds the
**DeepSeek-V3.2** technical report (arXiv:2512.02556); the slug was kept so earlier chapter links
resolve. Every §-locus given for that link below is a locus in the V3.2 report, and the Recipe table
names the checkpoint as DeepSeek-V3.2.

| Report | Pre-train | Mid-train | SFT | Specialist RL | Main RL | Final distill |
|---|---|---|---|---|---|---|
| [[kimi-k2]] | not reported | anneal + long-context only (§2.5) | MCP + synthetic tool trajectories (§3.1.1) | not reported | one joint RL stage (§3.2) | not reported |
| [[kimi-k2-5]] | GUI/agent trajectories in the vision corpus (App. B.3) | long-context activation (§4.3) | zero-vision, text-only (§2.2) | ability-based domains inside RL (§2.3) | joint text-vision RL + PARL (§4.4.2, §3) | not reported |
| [[kimi-k3]] | not reported | cooldown 256K→1M (§3.4) | cold start from prior-Kimi experts (§4.1.1) | 9 experts = 3 domains × 3 efforts (§4.1.2) | partial rollout, budget reward −1 (§4.1.2) | MOPD (§4.1.3) |
| [[glm-4-5]] | not reported | 100B agent+long at 128K (§2.3) | four-step agent pipeline (§3.1) | Reasoning / Agent / General-chat (§3) | search + SWE agentic RL (§3.3) | unified SFT over 3 experts (§3.1) |
| [[glm-5]] | not reported | 32K 1T / 128K 500B / 200K 50B (§2.3) | expanded agent share, 202,752 ctx (§3.1) | slide-generation expert only (§4.2.5) | Reasoning → Agentic → General RL (§3.2-§3.4) | on-policy cross-stage distillation (§3.5) |
| [[deepseek-v3.1]] | not reported | sparse-attention continued training (§2.1.1) | prompt-based cold start (§3.2.2) | 6 agentic/reasoning + 2 other domains (§3) | one mixed RL stage (§3) | specialist data distilled pre-RL (§3) |
| [[qwen3-coder]] | 7.5T tokens, 70% code (blog) | not reported | not reported | not reported | code RL + agent RL, 20,000 envs (blog) | not reported |
| [[qwen3-coder-next]] | not reported | ~600B repo tokens at 262,144 (§3.1.1) | execution-verified trajectories (§4.1) | WebDev / UX / QA / SWE (§4.2) | ~800K verifiable SWE tasks (§2.1) | expert distillation into SFT model (§4.2.5) |
| [[nemotron-3-super]] | 25T tokens, no agentic corpus listed (§2.3) | not reported | 7M samples / 80B tokens (§3.1.2) | not reported | RLVR(21 env) → SWE-RL → RLHF (§3.2) | not reported |
| [[tongyi-deepresearch]] | not reported | Agentic CPT 32K then 128K (§3.3.1) | ReAct + context-management cold start (§3.4.2) | not reported | on-policy GRPO variant (§3.4.3) | model merging (§3.4.4) |
| [[minimax-forge-agent-rl]] | not reported | not reported | not reported | not reported | unified mixed-domain CISPO (§4.1) | not reported |

**Reading the map.** Three columns are dense and three are sparse. Mid-training, SFT and main RL are
described by 8, 9 and 11 of the 11 reports respectively. Only two reports place any agentic content
in pre-training, and in both cases it is incidental rather than targeted: [[qwen3-coder]] raises the code ratio to 70% of 7.5T
tokens without claiming trajectories, and [[kimi-k2-5]] lists GUI action trajectories as one of seven
vision-data categories (caption, interleaving, OCR, knowledge, perception, video, agent data;
App. B.3) without a token share. Specialist RL and final distillation are filled together by five
reports — [[glm-4-5]], [[glm-5]], [[deepseek-v3.1]], [[kimi-k3]] and [[qwen3-coder-next]] — and left
empty by four — [[kimi-k2]], [[qwen3-coder]], [[nemotron-3-super]] and [[minimax-forge-agent-rl]].
The remaining two fill one column each: [[tongyi-deepresearch]] merges models at the end without
training specialists, and [[kimi-k2-5]] organizes RL by ability domain inside one run and states that
there is no separate specialist-distillation stage (§2.3). The split is not chronological: the 2025
[[glm-4-5]] report already has both columns.

**Implication for a general-purpose model.** The placements that dominate are the two with the
highest data volume per unit of engineering, mid-training and SFT, plus the one stage with a
verifier. That ordering is consistent with the generality argument: broad competence comes from
volume, correctness comes from verification, and neither substitutes for the other.

## §2 Mid-training: token counts and the synthetic-data ceiling

**Definition.** *Agentic mid-training* is continued pre-training with a next-token-prediction loss on
agent trajectories and on repository-scale or long-document data, run after the main pre-training run
and before SFT. [[tongyi-deepresearch]] names its version Agentic Continual Pre-training and states
its purpose as providing "a base model endowed with a strong inductive bias for agentic behavior,
while simultaneously preserving broad linguistic competence" (§3.3.1).

**Mechanism, in the order the reports run it.**
1. Extend the context window in stages, because agent trajectories are long. [[glm-4-5]] runs
   4K → 32K → 128K (§2.3); [[glm-5]] runs 32K → 128K → 200K (§2.3); [[tongyi-deepresearch]] runs
   32K → 128K (§3.3.1); [[qwen3-coder-next]] moves from 32,768 to 262,144 (§3.1.1).
2. Place natural long-form data first and synthetic trajectories last. [[glm-5]] up-samples "Long
   documents and synthetic agent trajectories" at the later stages (§2.3).
3. Interleave general pre-training data throughout, to hold the base distribution.
   [[tongyi-deepresearch]]: "a small proportion of general pre-training data is interleaved" (§3.3.1).
4. Keep the packing policy aligned with the trajectory boundary. [[glm-4-5]] applies best-fit packing
   only in mid-training (§2.3); [[qwen3-coder-next]] uses best-fit packing plus masking of "highly
   repetitive segments" (§3.2).

**Worked numeric example: what fraction of the budget is agentic?** Take the two reports that print
both the mid-training and the total token counts.

| Report | Total base tokens | Mid-training block | Stage containing agent trajectories | Share of total |
|---|---|---|---|---|
| [[glm-4-5]] | 23T (Abstract) | 500B + 500B + 100B = 1.1T (Figure 3) | 100B at 128K | 100B / 23T = 0.43% |
| [[glm-5]] | 28.5T (§1, §2) | 1T + 500B + 50B = 1.55T (§2.3) | 550B at 128K and 200K | 550B / 28.5T = 1.9% |

Checking the arithmetic by hand: 100 / 23,000 = 0.00435, and (500 + 50) / 28,500 = 0.0193. GLM-5
raises the agentic-adjacent share by a factor of about 4.4 between the two releases, while the total
budget grows by a factor of 1.24. The growth in agentic capability between GLM-4.5 and GLM-5 is
therefore not explained by the total token budget.

**Evidence: how much synthetic data is too much.** [[qwen3-coder-next]] is the only report that
states a ceiling and a reason (§3.1):

```
heavy reliance on synthetic data can significantly improve performance on targeted tasks,
but may lead to over-specialization, reduced response diversity, and weaker adaptation to
other tasks during fine-tuning. Therefore, our goal is to introduce the minimum amount of
synthetic data required for the model to reliably perform common user tasks, while
preserving response diversity and maintaining strong general-purpose capabilities.
```

The corresponding positive result is on rewriting rather than on synthesis: passing web documents
through Qwen3-Coder-480B-A35B-Instruct to produce clean Markdown raises Evalplus from 54.38 to 63.09,
MultiplE from 36.02 to 48.35, and CRUX-Eval from 57.13 to 58.94 (§3.1.1, Table 1). **Result (single
study)**; the base model, token count and seed count for the ablation are not reported.

**Conditions and limits.** No report ablates the mid-training agentic share against a control run at
the same total budget. Every number in the table above is a recipe value, not a measured optimum.
This is an **Open question**: the share that maximizes agentic capability per lost point of general
capability has not been measured by any open report.

## §3 The SFT share: trajectory sourcing and open datasets

**Definition.** The *SFT agentic share* is the fraction of supervised-fine-tuning samples that are
full action-observation trajectories rather than single-turn responses.

**Where trajectories come from.** Four sourcing patterns appear, in increasing order of verification
strength.

1. **Simulated tools and simulated users.** [[kimi-k2]] fetches 3000+ real MCP tools from GitHub and
   evolves over 20,000 synthetic tools, builds thousands of agents from synthesized system prompts,
   and generates trajectories with an LLM-simulated user and a stateful tool simulator, keeping only
   trajectories that an LLM judge marks as meeting the task rubric (§3.1.1; see [[kimi-k2-agentic-data]]).
   [[glm-4-5]] runs the same four steps and requires agreement from multiple judge agents (§3.1).
2. **Synthetic multi-role simulation at scale.** [[nemotron-3-super]] runs a six-stage pipeline
   (domain → policy and tools → scenario → 16 simulated interactions per policy-scenario pair →
   LM-as-a-Judge verification at outcome and process level → difficulty filtering that drops
   all-success and all-failure scenarios), producing "279,116 conversations across 838 domains",
   against 15,588 conversations across 5 domains in the preceding Nano release (§3.1.1).
3. **Teacher trajectories in real harnesses.** [[nemotron-3-super]] records CLI interactions from
   Qwen3-Coder-480B and MiniMax M2.5 across Codex, OpenCode, Qwen Code CLI and Stirrup, then
   normalizes them to OpenAI message format (§3.1.1). [[qwen3-coder-next]] generates trajectories
   with Qwen3-Coder-480B-A35B-Instruct across six scaffolds and rule-filters for missing termination
   signals, task failures and malformed tool calls (§3.1.2).
4. **Execution-verified trajectories.** [[qwen3-coder-next]] deploys a Mini-SWE-agent instance as a
   user simulator that executes the proposed code and judges from compiler output, runtime errors and
   environment state whether the response advances the task (§4.1).

**Worked numeric example: the agentic share of one disclosed SFT blend.** [[nemotron-3-super]] prints
"over 7M total samples" for stage-1 SFT (§3.1.2) and the sizes of most agentic components. Summing
the disclosed agentic components:

```
general-purpose tool calling      1,500,000
conversational tool use             279,116
terminal use                         84,864
agentic CLI synthesis                15,000
SWE tasks                             3,000
web development                      10,000
                                 ----------
subtotal                          1,891,980   →  1.89M / 7M ≈ 27% of samples
```

The report does not print the exact blend percentages (they appear only in Figure 16), so 27% is a
lower bound derived from the disclosed component sizes: it omits search and long-context agentic
data, whose sizes are not printed. The point of the arithmetic is the order of magnitude. A recipe
that targets agentic capability at SFT gives roughly a quarter to a third of its samples to
trajectories, not a few per cent.

**Open datasets, for readers without a synthesis pipeline.** Three released sets cover the same
ground at reproducible cost.

| Dataset | Environments | Size | Result at the locus |
|---|---|---|---|
| [[toucan-mcp]] | 495 curated MCP servers (from ~2,800 crawled), 2,000+ tools | 1.5M trajectories; 119.3K SFT subset | Qwen2.5-32B-Instruct BFCL V3 61.73% → 70.45% (+8.72); 14B 57.69% → 65.09%; 7B 55.10% → 58.26% (Table 2) |
| [[swe-smith]] | 128 Python repositories, one Docker image each | 50,137 task instances; 5,016 curated trajectories | SWE-agent-LM-32B 40.2% on SWE-bench Verified, 30.7% on Lite, pass@1 (Table 3) |
| [[agent-data-protocol]] | 13 converted datasets across coding, SWE, tool use, browsing | 1.3M trajectories | Qwen2.5-Coder-32B 40.3% on SWE-bench Verified with SWE-Agent; ~20% average gain over base models (§6.1) |

**Worked numeric example: the cost of an executable environment.** [[swe-smith]] reports $1360 total
and about 20 hours of one author's labour for 50,137 instances over 128 repositories, at 295 GB of
environment storage (§2.2, Table 2). Per instance that is $1360 / 50,137 ≈ 2.7 cents and
295 GB / 50,137 ≈ 5.9 MB. (Table 1 separately prints an average model cost of 2.32 cents per
instance for bug generation alone; the 2.7-cent figure divides the full $1360, which also covers
repository installation and issue writing.) The comparison row in the same table is the R2E-Gym subset ([[r2e-gym]]) at
4 TB for 4.6k tasks, or about 870 MB per task, and [[swe-gym]] at 6 TB for 2.4k tasks, or 2.5 GB per
task. The difference is the design choice of one image per repository rather than one per instance:
5.9 MB per task against 870 MB, a factor of about 147 in storage per task. The three task counts
differ (50,137 against 4.6k and 2.4k), so the comparison holds per task, not at a fixed total.

**Negative result to carry with the SFT numbers.** [[deepswe]] §6 reports that RL on SWE-Smith and
SWE-Gym data "gave limited improvement with a high solve-none rate" in its own setting. A dataset
that is effective as SFT material is not automatically effective as an RL prompt pool, because RL
needs prompts whose pass rate under the current policy is strictly between 0 and 1.

## §4 Specialist distillation against one mixed RL stage

**Definition.** *Specialist distillation* trains several domain-specific policies from a common
checkpoint and then merges them into one model by training the student on the specialists' outputs.
The alternative, *one mixed RL stage*, routes prompts from every domain into a single RL run with
per-domain reward functions.

**The measurable problem.** Sequential single-domain optimization degrades earlier domains.
[[glm-5]] states it directly: "sequentially optimizing for distinct objectives can lead to the
cumulative degradation of previously acquired capabilities" (§3.5). [[deepseek-v3.1]] gives the
opposite framing for the same problem, merging its domains into one RL stage to avoid "the
catastrophic forgetting issues commonly associated with multi-stage training paradigms" (§3).

**Six designs in the current reports.**

| Design | Report | Structure |
|---|---|---|
| Specialists → distill → one mixed RL | [[deepseek-v3.1]] | 6 agentic/reasoning domains plus writing and general QA, each RL-trained from the same base; their data distills into one model; then one mixed RL stage (§3) |
| Experts → unified SFT | [[glm-4-5]] | Reasoning, Agent and General-chat experts, each cold-start SFT then RL, distilled by an SFT set of millions of samples (§3, §3.1) |
| Sequential RL → on-policy distillation | [[glm-5]] | Reasoning RL → Agentic RL → General RL, then distillation from the earlier stages' final checkpoints (§3.2-§3.5) |
| Experts × efforts → MOPD | [[kimi-k3]] | 3 domains × 3 reasoning efforts = 9 experts, consolidated by Multi-Teacher On-Policy Distillation (§4.1.2-§4.1.3) |
| No specialists, staged RL | [[nemotron-3-super]] | One model through RLVR → SWE-RL → RLHF → MTP healing (§3.2) |
| No specialists, one stage | [[minimax-forge-agent-rl]] | Reasoning, General QA and Agent mixed simultaneously under CISPO (§4.1) |

**Mechanism of on-policy distillation, as both labs write it.** The student samples its own
trajectories; the teacher scores the student's tokens; the ratio replaces the advantage.
[[glm-5]] (§3.5, Eq. 2):

```
Â_{i,t} = sg[ log( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ]
```

[[kimi-k3]] (§4.1.3, Eq. 15) adds a clip:

```
r_opd^d(y_t | e, x, y_<t) = clip( sg[ log( π_teacher^(d,e)(y_t | x, y_<t) / π_θ(y_t | e, x, y_<t) ) ],
                                  −R_max, +R_max )
```

`sg` is the stop-gradient operator; `π_teacher` is the frozen specialist for domain d at reasoning
effort e; `π_θ` is the student; `R_max` bounds the per-token reward.

**Worked numeric example.** Suppose at some token the teacher assigns probability 0.40 and the
student 0.05. The unclipped reward is log(0.40 / 0.05) = log 8 = 2.08. With R_max = 1.0 the reward is
clipped to 1.0. At a token where teacher and student agree (0.30 against 0.30) the reward is
log 1 = 0. At a token the student over-produces (teacher 0.02, student 0.50) the reward is
log(0.04) = −3.22, clipped to −1.0. The signal is therefore dense — every token carries one — and
bounded, which is what lets [[kimi-k3]] reuse its partial-rollout RL infrastructure for distillation
without a separate trainer (§4.1.3).

**Why the group size changes.** [[glm-5]] sets "the group size in the GRPO algorithm ... to 1 to
increase data throughput, and the batch size ... to 1024" during distillation, because "the advantage
is computed directly from the gap with the teacher models instead" (§3.5). Under [[grpo]] the group
exists only to estimate a baseline; when the teacher supplies the baseline, the group collapses to a
single sample and throughput rises by the former group factor. Going from group 32 (the Reasoning RL
setting in §3.2) to group 1 at the same total sample count means 32× more distinct prompts per step.

**Conditions and limits.** No report prints a before/after benchmark pair that isolates the
distillation step. [[deepseek-v3.1]] states that the distilled model scores "slightly below the
specialists" and that subsequent RL removes the gap, with no numbers (§3). [[kimi-k3]] reports one
negative result at the locus: finer-grained top-k distillation objectives showed "no clear advantage
in either convergence speed or final performance" (§4.1.3). **Interpretation:** the choice between
specialist distillation and one mixed stage currently rests on throughput and engineering constraints
rather than on measured capability, because no open report runs both at equal compute.

## §5 The agentic RL ledger: environments, budgets, reward, staleness, context

This section reads the settings that ch-45b treats mechanically as they are actually configured in
released recipes.

### 5.1 Environments and prompts

| Report | Environments | Prompts / tasks | Locus |
|---|---|---|---|
| [[deepseek-v3.1]] | 1,827 synthesized general-agent environments (kept when pass@100 under V3.2 is non-zero) | 24,667 code-agent, 50,275 search-agent, 4,417 general-agent, 5,908 code-interpreter | §3.2.3, Table 1 |
| [[glm-5]] | over 10k verifiable SWE environments across 9 languages; thousands of terminal environments at >90% Docker build accuracy | not reported | §4.2.1-§4.2.2 |
| [[qwen3-coder]] | 20,000 independent environments run in parallel | not reported | blog, Post-Training |
| [[qwen3-coder-next]] | Docker environments built by an environment-building agent | ~800K verifiable SWE task instances over 9+ languages | §2.1 |
| [[nemotron-3-super]] | 21 environments, 37 RL datasets; up to 4,000 environment instances per batch | 256 prompts per step × 16 responses | §3.2.1, §3.2.4, Figure 12 |
| [[kimi-k2]] | Kubernetes sandbox with over 10,000 concurrent instances | not reported | §3.2.1 |
| [[kimi-k2-5]] | Rollout Manager with up to 100,000 concurrent agent tasks | not reported | App. D |
| [[minimax-forge-agent-rl]] | over 100,000 distinct real-world scaffolds and environments | millions of samples per day | Forge post, opening summary (before §1) |
| [[deepswe]] | 512 Docker containers per RL iteration | 4.5K R2E-Gym problems | blog §2.1-§2.2 |

**Worked numeric example: the mixture the prompt counts imply.** [[deepseek-v3.1]] Table 1 sums to
24,667 + 50,275 + 4,417 + 5,908 = 85,267 prompts, which matches the §1 statement of "over 1,800
environments and 85,000 prompts". The shares are 50,275 / 85,267 = 59.0% search agent,
24,667 / 85,267 = 28.9% code agent, 5,908 / 85,267 = 6.9% code interpreter, and
4,417 / 85,267 = 5.2% general agent. The smallest slice is the one built entirely from synthesis,
and it is also the slice the report uses to demonstrate transfer (§4.3, Figure 5): RL on
V3.2-SFT with only synthetic general-agent tasks improves Tau2Bench, MCP-Mark and MCP-Universe, while
V3.2-Exp, trained with RL only in search and code environments, does not improve on them. Five per
cent of the prompt budget carries the out-of-domain generalization result.

**Worked numeric example: batch arithmetic.** [[nemotron-3-super]] §3.2.4 states 256 prompts per step,
16 responses per prompt, and a training batch size of 4096. Since 256 × 16 = 4096, every rollout
produces exactly one gradient update, so the run is on-policy up to the asynchronous lag the same
section bounds at "at most one step behind the latest model version". A reader who changes the group
size to 8 without changing the batch must either halve the prompts per step or accept two updates per
rollout, which doubles the policy lag.

### 5.2 Algorithm and off-policy control

Every 2026 recipe treats the gap between the inference engine's distribution and the training
engine's distribution as the primary stability problem, and every one of them handles it by masking
tokens or sequences rather than by a KL penalty.

| Report | Baseline | Off-policy control | KL |
|---|---|---|---|
| [[deepseek-v3.1]] | Â = R_i − mean(R), no std division (§3.1) | Off-Policy Sequence Masking: M = 0 when Â < 0 and mean-token log(π_old/π_θ) > δ (Eq. 9); Keep Routing; Keep Sampling Mask | K3 estimate multiplied by π_θ/π_old for an unbiased gradient; strength differs by domain; weak or zero for mathematics (§3.1) |
| [[glm-5]] | GRPO group-normalized advantage, group 32, batch 32 (§3.2) | IcePop `pop(ρ, 1/β, β)` with β = 2, ε_low = 0.2, ε_high = 0.28; in agentic RL, double-sided token clipping against rollout log-probabilities plus staleness filter | KL term removed (§3.2) |
| [[kimi-k2-5]] | K1.5-style squared objective (Eq. 1) | token-level log-ratio clip [α, β], applied regardless of advantage sign | τ log(π_θ/π_old) inside the objective (§4.4.2) |
| [[kimi-k3]] | follows K2.5 (§4.1.2) | per-token regularization stated to tolerate the extreme off-policy regime created by partial rollouts | as K2.5 |
| [[nemotron-3-super]] | asynchronous GRPO (§3.2.4) | importance-sampling ratio from training and inference log-probabilities is masked; inference workers at most one step stale | not reported |
| [[tongyi-deepresearch]] | GRPO variant with leave-one-out advantage (Eq. 4-5) | strictly on-policy, ratio fixed at 1.0 | not reported |
| [[deepswe]] | GRPO++ (blog §2.3) | compact filtering masks trajectories ending by context, step or timeout limit | no KL loss |

**Mechanism: the staleness filter.** [[glm-5]] §4.1.2 logs, for each response, the sequence of
rollout model versions (w_0, …, w_k) that produced it, and discards the sample when
`w′ − w_0 > τ` for the current version w′. A long-horizon trajectory generated across several weight
pushes therefore leaves training entirely rather than entering it with a large importance ratio.

**Worked numeric example: what the filter and the group padding do to a batch.** Take group size
G = 32 with τ = 4. Suppose in one group 6 trajectories are stale by more than 4 versions and 5 more
fail because the sandbox crashed. Valid samples: 32 − 6 − 5 = 21. GLM-5's rule pads the group by
repeating valid samples when the valid count exceeds half the group size; 21 > 16, so the group is
padded back to 32 by duplication and kept. Had 17 samples been dropped instead of 11, leaving 15, the
whole group would be discarded. The rule exists because a group-mean baseline computed over a
selection-biased subset is biased: crashed sandboxes correlate with harder tasks, so dropping them
without padding shifts the baseline upward and turns correct-but-hard trajectories into negative
advantages.

### 5.3 Token budgets

Three reports control output length inside the reward rather than by truncation alone.

- [[kimi-k2]] sets a per-sample maximum token budget by task type; over-budget responses are
  truncated and penalized (§3.2.3). Budgets and penalty values are not printed.
- [[kimi-k2-5]] names the failure mode that a fixed budget creates — "length-overfitting", where
  "models trained under rigid budget constraints often fail to generalize to higher compute scales" —
  and alternates phases every m iterations (§4.4.2, Eq. 2). Phase 0 applies the budget only when the
  problem's mean accuracy is above λ; Phase 1 lifts it. The budget itself is the ρ-th percentile of
  correct-response lengths, estimated once and then fixed. Reported effect on K2 Thinking:
  "Toggle decreases output tokens by 25∼30% with a negligible impact on performance" (Figure 5), and
  the reduction transfers to GPQA and MMLU-Pro when training used only mathematics and programming.
  **Result (single study)**; no per-benchmark table is printed.
- [[kimi-k3]] overrides the task reward with −1 when the trajectory exceeds τ · b_0(x), where b_0(x)
  is a per-problem budget estimated from the cold-start model, then anneals τ downward across a
  curriculum to produce the low-, high- and max-effort experts (§4.1.2). For agentic tasks the count
  includes tool-call arguments, not thinking tokens alone.

**Worked numeric example.** Let b_0(x) = 8,000 tokens and τ = 1.5, so the threshold is 12,000. In a
group of 8 rollouts, 3 are correct within budget (reward 1), 3 are wrong within budget (reward 0),
and 2 are correct at 13,500 tokens (reward −1). The group mean is (3·1 + 3·0 + 2·(−1)) / 8 = 0.125.
The over-budget-but-correct rollouts get advantage −1 − 0.125 = −1.125, a larger push-down than the
wrong-but-short rollouts at 0 − 0.125 = −0.125. The budget reward therefore penalizes verbosity more
strongly than it penalizes being wrong, which is the behaviour the annealing curriculum is designed
to introduce gradually rather than from step 0.

### 5.4 Context policy

ch-45c covers the mechanics; here is what the recipes select and what they measure.

| Report | Policy | Measured effect |
|---|---|---|
| [[deepseek-v3.1]] | Discard-all, triggered above 80% of the window | BrowseComp 51.4 without → 67.6 with (§4.4, Figure 6) |
| [[glm-5]] | Keep-recent-k with k = 5, then hierarchical with Discard-all above T = 32k | 55.3% → 62.0% from keep-recent alone; 75.9 final with the hybrid (§4.2.4) |
| [[kimi-k2-5]] | Agent Swarm: sub-agents hold bounded local contexts and return only task-relevant outputs | BrowseComp 78.4 (swarm) against 60.6 (single agent); WideSearch Item-F1 79.0 against 72.7 (Table 6) |
| [[minimax-forge-agent-rl]] | context management modelled as an agent action inside the RL loop | no numbers; the stated reason is that inference-only context management "introduces a severe distribution shift from the RL training data" |
| [[tongyi-deepresearch]] | context-management mode trained in SFT alongside ReAct mode | no isolating ablation |

The pattern across the rows: a context policy applied only at inference is worth between 6.7 and 16.2
points on BrowseComp, and two independent labs ([[minimax-forge-agent-rl]], [[tongyi-deepresearch]])
argue that training under the same policy removes a distribution shift. No source has yet measured
trained-versus-inference-only context management at equal compute. **Open question.**

## §6 Parallel orchestration as a training target

[[kimi-k2-5]] §3 is the one open report that trains the orchestration decision itself rather than a
single sequential agent. The design is worth stating precisely because its choices are all responses
to credit-assignment problems covered in ch-45b.

1. **Decoupling.** A trainable orchestrator plus "frozen subagents instantiated from fixed
   intermediate policy checkpoints". Subagent trajectories are excluded from the objective, so
   subagent outputs are environment observations rather than differentiable decisions. Stated
   motivation: "credit assignment ambiguity and training instability", since "a correct final answer
   does not guarantee flawless subagent execution, just as a failure does not imply universal
   subagent error".
2. **Reward.** `r_PARL(x, y) = λ1 · r_parallel + λ2 · r_finish + r_perf(x, y)`. `r_perf` is the task
   outcome. `r_parallel` rewards sub-agent instantiation, countering "serial collapse", the local
   optimum of always running one agent. `r_finish` rewards completed subtasks, countering "spurious
   parallelism", defined as spawning many subagents "without meaningful task decomposition".
   λ1 and λ2 "are annealed to zero over the course of training", so the final policy optimizes the
   task outcome alone.
3. **Cost metric.** `CriticalSteps = Σ_t (S_main^(t) + max_i S_sub,i^(t))`, the analogue of the
   critical path in a computation graph. Training and evaluation budgets are expressed in critical
   steps, so adding subagents that do not shorten the longest branch buys nothing.

**Worked numeric example.** A three-stage episode where the main agent takes 1 step per stage and
spawns, in stage 2, four subagents taking 6, 9, 4 and 7 steps. Total steps executed:
1 + (1 + 26) + 1 = 29. Critical steps: 1 + (1 + 9) + 1 = 12. Splitting the same 26 subagent steps
evenly across four subagents (7, 7, 6, 6) would give critical steps 1 + (1 + 7) + 1 = 10 for the same
total work — the metric rewards balance, not concurrency.

**Result.** Table 6: BrowseComp 78.4% with the swarm against 60.6% single-agent, WideSearch Item-F1
79.0% against 72.7%, in-house Swarm Bench 58.3% against 41.6%. Execution time to reach a target
WideSearch Item-F1 falls by "3× ∼ 4.5×". **Result (single study)**, one lab, one model, no ablation
separating the two auxiliary reward terms.

## §7 Which reports measure general ability after agentic training

The course goal makes this the decisive column, and it is the sparsest one.

| Report | General (non-agentic) evaluation after agentic training | Evidence |
|---|---|---|
| [[kimi-k2-5]] | yes, before/after on text benchmarks | MMLU-Pro 84.7 → 86.4, GPQA-Diamond 84.3 → 86.4, LongBench v2 56.7 → 58.9 after visual RL (Table 2) |
| [[qwen3-coder-next]] | yes, against the base model | "competitive with Qwen-Next, slightly improving on MMLU-Redux and GPQA, while remaining close on MMLU, MMLU-Pro, and SuperGPQA"; large math gains on HMMT25 Feb and AIME25 (§5.3, Tables 8-9) |
| [[nemotron-3-super]] | yes, as a stated training finding | "training on all environments simultaneously yields stable gains, whereas single-environment training leads to severe regressions on other benchmarks" (§3.2.1); full benchmark suite in §3.3 |
| [[deepseek-v3.1]] | partially: unseen environments only | MCP-Universe, MCP-Mark and Tool-Decathlon were not in RL and are read as out-of-domain generalization (§4.1); synthetic-only general-agent RL transfers to all three (§4.3, Figure 5) |
| [[glm-4-5]] | qualitative only | RL on web search and SWE "leads to generalized performance improvements across other tasks and benchmarks, such as general tool usage and coding tasks like Terminal-Bench" (§3.3.2); no numbers |
| [[glm-5]] | indirect: distillation is the mechanism, not a measurement | cross-stage distillation is motivated by "cumulative degradation" (§3.5); no before/after pair |
| [[minimax-forge-agent-rl]] | stated, not measured | unified mixed-domain training "significantly enhances the model's generalizability across diverse tasks" (§4.1) |
| [[kimi-k2]] | no post-agentic comparison | Table 3 reports absolute scores; §5 notes performance "may decline on some tasks when tool use is enabled unnecessarily" |
| [[deepswe]] | no | The post evaluates only on SWE-Bench-Verified and its SWE-Bench-Hard validation curve (blog §4, Figure 2) |
| [[agent-data-protocol]] | yes, as the central experiment | mixed corpus avoids the negative transfer of single-domain tuning (§6.2, Table 6) |

**Replicated finding.** Two independent sources measure that a mixture over many task sources beats
single-source training on the source's own task: [[agent-data-protocol]] at SFT (Qwen3-8B on
SWE-Bench: 16.6% mixed against 11.0% SWE-smith-only and 0.2% CodeActInstruct+Code-Feedback-only) and
[[nemotron-3-super]] at RL (stated, not tabulated). The agreement across two stages and two labs is
the strongest generality result in this chapter.

**Replicated finding on harness specificity.** [[qwen3-coder-next]] measures it —
"Models trained on trajectories from one scaffold do not transfer strongly to others", with OpenHands
transferring poorly to SWE-Agent and the reverse "moderately successful" (§3.1.2, Figure 3) — and
[[minimax-m2-aligning-to-what]] reports it as a design post-mortem: benchmark scores climbed under
tool scaling, then "if we changed the environment even slightly—like swapping to a different
scaffolding framework—its performance would plummet". Three recipes act on it: [[kimi-k3]] randomizes
harness configurations during RL (§4.2.1), [[nemotron-3-super]] implements OpenCode and Codex tool
formats inside OpenHands so that "multi-harness training improves the model's generalization and
performance across all target harnesses at inference time" (§3.2.2), and
[[minimax-forge-agent-rl]] trains against black-box scaffolds through a gateway (§2.3).

## Negative samples and negative feedback

Most agentic rollouts fail: the one report that prints a rate, [[swe-smith]] §4, records a 36%
resolve rate over 17,906 expert-trajectory attempts, so roughly two thirds of the attempts produce
failures. What a recipe does with those failures is therefore a recipe decision with measurable
consequences. Using the four senses of "negative" from ch-43a:

**1. Negative marginal value (discard).** The default. [[kimi-k2]] keeps only trajectories that pass
the task rubric under an LLM judge (§3.1.1); [[glm-4-5]] keeps only trajectories that multiple judge
agents mark complete (§3.1); [[qwen3-coder-next]] rule-filters missing termination signals, task
failures and malformed tool calls (§3.1.2); [[swe-smith]] caps any instance at 3 trajectories because
repeatedly-solved instances "degrade model performance" (§4). Rejection rates are mostly not reported;
[[swe-smith]] is the exception, at a 36% resolve rate over 17,906 attempts (§4).

**2. Negative as content (train with cross-entropy, masked).** [[glm-5]] §3.1: "Erroneous segments
within trajectories are retained but masked out in the loss function, allowing the model to learn
error correction behaviors without reinforcing incorrect actions." The failure stays in the context
so the model sees the recovery; the failing tokens carry no gradient. This is the only appearance of
sense 2 in these reports.

**3. Negative as conditioning.** No recipe in this chapter trains failures under a control token.

**4. Negative as gradient.** Four distinct controls appear, all of them bounding the push-down rather
than removing it:

- **Sequence masking by divergence.** [[deepseek-v3.1]] Eq. 9 zeroes a sequence when Â < 0 **and**
  the mean token log(π_old/π_θ) exceeds δ. Only negative-advantage sequences are eligible; the
  authors state that "highly off-policy negative samples can be detrimental" (§3.1).
- **Trajectory masking by termination cause.** [[deepswe]] compact filtering masks trajectories that
  end by context limit, step limit or a 20-minute generation timeout, so a trajectory that was cut
  off is neither rewarded nor punished (blog §2.3).
- **Selective exclusion of over-length negatives.** [[tongyi-deepresearch]] §3.4.3 reports the
  failure this prevents: "directly optimizing on an unfiltered set of negative rollouts significantly
  degrade training stability and can lead to policy collapse after extended training".
- **Environment-failure exclusion.** [[glm-5]] §4.1.2 records the failure reason per sample and
  removes sandbox crashes, because "Such failures introduce noisy training signals because they
  reflect environment instability rather than the model's capability".

**Mechanism.** The softmax logit gradient is `∂ log p_y / ∂ z_j = 1[j = y] − p_j`. A negative
advantage on token y removes mass from y and redistributes it in proportion to p_j over the
alternatives. When the failure was caused by the environment rather than the policy, the
redistribution is toward whatever the model would otherwise have said at that position, which is
noise. When the failure was a timeout on a hard problem, the push-down lands on the correct-but-slow
behaviour. This is why all four controls above mask rather than reweight: they set the gradient to
zero on samples whose sign cannot be trusted.

**Effect on generality.** None of these reports logs pass@k at large k after agentic RL, which is the
diagnostic ch-43a recommends for coverage loss. [[deepswe]] is the only one that reports pass@k at
all: 42.2% Pass@1 and 71.0% Pass@16 on SWE-Bench-Verified after RL, with no base-model pass@16 for
comparison (blog §4). **Open question:** whether agentic RL narrows the solution distribution the way
[[rlvr-beyond-base-model]] measures for single-turn RLVR has not been tested in any open agentic
report.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLM-4.5 | 355B / 32B act. | mid-train | Agent + long-context stage | 100B tokens at 128K (long documents up-sampled, synthetic agent trajectories added) | arXiv:2508.06471v1 §2.3, Figure 3 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | mid-train | Context stages and tokens | 32K (1T), 128K (500B), 200K (50B) | arXiv:2602.15763v2 §2.3 | verified 2026-09-15 | §2.3: the 200K stage improved results "even within the 128K context window"; no numbers |
| GLM-5 | 744B / 40B act. | mid-train | Issue–PR corpus | ~10M issue–PR pairs; ~160B unique tokens after filtering | §2.3 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | SFT | Max context | 202,752 tokens | §3.1 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | RL (reasoning) | Algorithm and settings | GRPO + IcePop, no KL term; β = 2, ε_low = 0.2, ε_high = 0.28; on-policy; group 32; batch 32 | §3.2 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | RL (agentic) | Off-policy control | double-sided token clip on rollout log-probabilities; staleness filter w′ − w_0 > τ; environment-failure exclusion with group padding above G/2 | §4.1.2 | verified 2026-09-15 (τ, ε_ℓ, ε_h not printed) | stated to improve stability; no numbers |
| GLM-5 | 744B / 40B act. | RL (agentic) | Environments | over 10k verifiable SWE environments, 9 languages; thousands of terminal environments, >90% Docker build accuracy | §4.2.1-§4.2.2 | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | distill | On-policy cross-stage distillation | advantage replaced by sg[log(π_teacher/π_θ)]; group size 1; batch 1024 | §3.5, Eq. 2 | verified 2026-09-15 | no before/after number reported |
| GLM-5 | 744B / 40B act. | eval-gate | Context management for search | keep-recent k = 5; hybrid with Discard-all above T = 32k | §4.2.4 | verified 2026-09-15 | BrowseComp 55.3% → 62.0% with keep-recent; 75.9 final with the hybrid |
| Kimi K2.5 | built on 1.04T / 32B act. K2 | pretrain (joint) | Vision-text joint tokens | ~15T additional vision-text tokens at 4K | arXiv:2602.02276v2 §4.3 | verified 2026-09-15 | Table 1: early fusion at 10%:90% beats mid and late fusion at fixed budget |
| Kimi K2.5 | as above | RL | Token-level clip | gradients zeroed for tokens with log-ratio outside [α, β], regardless of advantage sign | §4.4.2, Eq. 1 | verified 2026-09-15 (α, β not printed) | stated "essential for maintaining training stability" in long-horizon tool use |
| Kimi K2.5 | as above | RL | Toggle budget schedule | alternate Phase0/Phase1 every m iterations; budget = ρ-th percentile of correct-response lengths, fixed after estimation | §4.4.2, Eq. 2 | verified 2026-09-15 (λ, m, ρ not printed) | Figure 5: 25-30% fewer output tokens on K2 Thinking with negligible performance impact |
| Kimi K2.5 | as above | RL infra | Concurrency | up to 100,000 concurrent agent tasks; token-in-token-out with recorded log-probabilities | App. D | verified 2026-09-15 | no ablation reported |
| Kimi K3 | 2.8T / 104B act. | RL | Partial rollout | generation pauses when a fraction λ of the N × K trajectories completes; paused rollouts resume first next iteration | arXiv:2607.24653v2 §4.1.2 | verified 2026-09-15 (λ, N, K not printed) | no ablation reported |
| Kimi K3 | 2.8T / 104B act. | RL | Budget reward | task reward overridden to −1 when tokens exceed τ · b_0(x); τ annealed per domain | §4.1.2 | verified 2026-09-15 (τ, b_0 estimator not printed) | produces the low/high/max effort experts; no ablation |
| Kimi K3 | 2.8T / 104B act. | distill | MOPD | per-token reward clip(sg[log(π_teacher/π_θ)], −R_max, R_max) over 9 experts | §4.1.3, Eq. 15 | verified 2026-09-15 (R_max not printed) | top-k objectives showed "no clear advantage" |
| DeepSeek-V3.2 | not printed | RL | Agent tasks | 24,667 code-agent; 50,275 search-agent; 4,417 general-agent; 5,908 code-interpreter | arXiv:2512.02556v1 Table 1 | verified 2026-09-14 (card) | §4.3, Figure 5: synthetic-only general-agent RL improves Tau2Bench, MCP-Mark, MCP-Universe |
| DeepSeek-V3.2 | not printed | RL | General-agent environments | 1,827 (kept when pass@100 under V3.2 is non-zero) | §3.2.3 | verified 2026-09-14 (card) | Table 5: V3.2-Exp pass@1 12% on 50 sampled tasks |
| DeepSeek-V3.2 | not printed | RL | Off-Policy Sequence Masking | mask when Â < 0 and mean-token log(π_old/π_θ) > δ | §3.1, Eq. 9 | verified 2026-09-14 (card); δ not printed | stated to improve stability in otherwise unstable runs |
| Nemotron 3 Super | 120B / 12B act. | SFT | Blend | over 7M samples, 80B tokens; agentic share "a much larger proportion" than Nano | report 2026-04-03 §3.1.2, Figure 12 | verified 2026-09-15 (percentages in figure only) | no ablation reported |
| Nemotron 3 Super | 120B / 12B act. | SFT | Conversational tool use | 279,116 conversations across 838 domains (Nano: 15,588 across 5) | §3.1.1 | verified 2026-09-15 | no ablation reported |
| Nemotron 3 Super | 120B / 12B act. | RL | RLVR batch | 256 prompts per step × 16 responses; batch 4096; one gradient update per rollout; max generation 49K then 64K | §3.2.4 | verified 2026-09-15 | no ablation reported |
| Nemotron 3 Super | 120B / 12B act. | RL | Environments | 21 environments, 37 datasets; up to 4,000 environment instances per batch | §3.2.1, Figure 12 | verified 2026-09-15 | §3.2.1: all-environment training gives stable gains; single-environment training gives severe regressions elsewhere (stated, no table) |
| Nemotron 3 Super | 120B / 12B act. | RL | Policy lag | inference workers at most one step behind; KV cache not recomputed after weight updates | §3.2.4 | verified 2026-09-15 | no ablation reported |
| Qwen3-Coder-Next | 80B / 3B act. | mid-train | Repository-level tokens and context | ~600B tokens; context 32,768 → 262,144; best-fit packing | report 2026-03-03 §3.1.1, §3.2 | verified 2026-09-15 | Figure 3: within-scaffold performance rises with mid-training tokens; cross-scaffold transfer limited |
| Qwen3-Coder-Next | 80B / 3B act. | mid-train | Web-document reformatting | rewrite with Qwen3-Coder-480B-A35B-Instruct | §3.1.1, Table 1 | verified 2026-09-15 | Evalplus 54.38 → 63.09; MultiplE 36.02 → 48.35; CRUX-Eval 57.13 → 58.94 |
| Qwen3-Coder-Next | 80B / 3B act. | RL | Task pool | ~800K verifiable SWE task instances, 9+ languages; SFT and RL prompts fully disjoint; pass-rate filtering | §2.1, §4.2.4 | verified 2026-09-15 | no ablation reported |
| Qwen3-Coder-Next | 80B / 3B act. | RL | Reward shaping | trajectory-level outcome reward + unfinished-trajectory penalty + turn-level token penalty on invalid tool calls | §4.2.4 | verified 2026-09-15 (values not printed) | no ablation reported |
| Qwen3-Coder | 480B / 35B act. | pretrain | Tokens and code ratio | 7.5T tokens at 70% code; 256K native context, 1M with YaRN | blog 2025-07-22 | verified 2026-09-15 | no ablation reported |
| Qwen3-Coder | 480B / 35B act. | RL | Environment parallelism | 20,000 independent environments in parallel | blog 2025-07-22 | verified 2026-09-15 | no ablation reported |
| Tongyi DeepResearch | 30.5B / 3.3B act. | mid-train | Agentic CPT | 32K then 128K, with 64K-128K agentic behaviour data in stage 2; general pre-training data interleaved | arXiv:2510.24701 §3.3.1 | verified 2026-09-15 (token counts not printed) | no ablation reported |
| Tongyi DeepResearch | 30.5B / 3.3B act. | SFT | Context schedule | stage 1 at 40K (all context-management samples); stage 2 at 128K plus a small 40K portion for stability | §3.4.2 | verified 2026-09-15 | no ablation reported |
| Tongyi DeepResearch | 30.5B / 3.3B act. | RL | Algorithm and reward | strictly on-policy GRPO variant, token-level loss, clip-higher, leave-one-out advantage; binary answer reward, no format reward | §3.4.3, Eq. 4-5 | verified 2026-09-15 (ε values, G not printed) | unfiltered negatives caused policy collapse; over-length negatives excluded |
| MiniMax-M2.5 | not printed | RL | Scale and algorithm | CISPO; over 100,000 scaffolds/environments; contexts to 200K; millions of samples per day | Forge post 2026-02-13 opening summary and §4.1 | verified 2026-09-15 | no ablation reported; the post prints no benchmark table (prefix-tree merging is separately reported at a 40x training speedup, §3.2) |
| DeepSWE-Preview | 32B | RL | Data and compute | 4.5K R2E-Gym problems; 512 containers per iteration; 64 H100 for six days; ~200 steps | blog 2025-07-02 intro, §2.1-§2.2 | verified 2026-09-14 (card); conflicts with the released script | Figure 2: Pass@1 23% → 42% |
| SWE-smith | dataset | SFT | Cost and storage | $1360 total; ~20h human labour; 50,137 instances over 128 repositories; 295 GB | arXiv:2504.21798 §2.2, Table 2 | verified 2026-09-15 | Table 3: SWE-agent-LM-32B 40.2% on SWE-bench Verified from 5,016 trajectories |
| Toucan-1.5M | dataset | SFT | Curation | 495 MCP servers kept from ~2,800 crawled; 2,000+ tools; 119.3K SFT subset from 1.5M | arXiv:2510.01179 §3, §4.1 | verified 2026-09-15 | Table 2: Qwen2.5-32B-Instruct BFCL V3 61.73% → 70.45% |
| ADP Dataset V1 | dataset | SFT | Composition | 13 datasets, over 1.3M trajectories, subsampled so no source dominates | arXiv:2510.24702v2 §3; mixture weights App. C | verified 2026-09-15 | Table 6: mixed corpus beats single-domain tuning on the target task and avoids negative transfer |

**Starting point for a small general-purpose run.** Every value below comes from a `verified` row
above, with the conditions under which its source used it. For a model in the 7B-32B range with a
handful of executable environments: build environments at one Docker image per repository rather than
one per instance ([[swe-smith]], 128 Python repositories, $1360, 295 GB); assemble the SFT mixture
from several task families rather than one ([[agent-data-protocol]], 13 datasets, 1.3M trajectories,
Qwen2.5-Coder 7B-32B); run one RL stage over all environments with one gradient update per rollout
([[nemotron-3-super]], 256 prompts × 16 responses = batch 4096, at 120B/12B); mask rather than
penalize trajectories that end by context, step or time limit ([[deepswe]], Qwen3-32B, 4.5K problems,
64 H100 × 6 days); and exclude over-length negatives from the loss ([[tongyi-deepresearch]], 30.5B /
3.3B). None of these values was tuned at small scale by its source, so treat each as a starting point
with a logged diagnostic, not as an optimum.

## Generalization lens

**(a) What increases breadth.**
- **One mixed RL stage over many environments.** [[nemotron-3-super]] §3.2.1: all-environment
  training gives stable gains, single-environment training gives severe regressions elsewhere.
  [[minimax-forge-agent-rl]] §4.1 reports the same design choice with the same stated reason.
- **A mixed SFT corpus over several task families.** [[agent-data-protocol]] §6.2, Table 6: on
  SWE-Bench, 10.4% mixed against 1.0% SWE-smith-only at Qwen2.5-7B-Instruct; on GAIA, 9.1% against
  0.6% AgentInstruct-only.
- **Harness randomization during training.** [[kimi-k3]] §4.2.1 composes harness modules per task
  group specifically because a fixed harness causes overfitting to one tool schema;
  [[nemotron-3-super]] §3.2.2 reports that multi-harness SWE-RL "improves the model's generalization
  and performance across all target harnesses at inference time".
- **Interleaving general data inside agentic mid-training.** [[tongyi-deepresearch]] §3.3.1 keeps a
  small proportion of general pre-training data throughout, stated as preserving foundational
  generalization.
- **Cross-domain transfer that is measured rather than assumed.** [[kimi-k2-5]] Table 2 (visual RL
  raising MMLU-Pro, GPQA-Diamond and LongBench v2) and [[qwen3-coder-next]] §5.3 (code reasoning
  transferring to competition math) are the two measured positive transfers in this chapter.

**(b) What causes narrowing or forgetting.**
- **Sequential single-objective stages.** [[glm-5]] §3.5 names "cumulative degradation of previously
  acquired capabilities" as the reason for adding a distillation stage; [[deepseek-v3.1]] §3 merges
  RL domains for the same reason.
- **Heavy synthetic mid-training data.** [[qwen3-coder-next]] §3.1: "over-specialization, reduced
  response diversity, and weaker adaptation to other tasks during fine-tuning".
- **Tool scaling alone.** [[minimax-m2-aligning-to-what]]: benchmark scores rose, then collapsed on a
  scaffold swap; the post concludes that generalization requires perturbing the tool info, system
  prompt, user prompt, environment and tool responses, not the toolset alone.
- **Rigid token budgets.** [[kimi-k2-5]] §4.4.2 names "length-overfitting": models trained under a
  fixed budget "fail to generalize to higher compute scales" and default to truncated reasoning.
- **Unfiltered negative rollouts.** [[tongyi-deepresearch]] §3.4.3: policy collapse after extended
  training.
- **Unnecessary tool use.** [[kimi-k2]] §5 lists declining performance on some tasks when tool use is
  enabled where it is not needed — a narrowing that shows up only in evaluations that allow the model
  to decline the tool.

**(c) How to measure it for this stage.**
1. **Held-out environments, not held-out prompts.** [[deepseek-v3.1]] §4.1 treats MCP-Universe,
   MCP-Mark and Tool-Decathlon as out-of-domain because they were not in RL. Reserve whole
   environments, since prompts inside a trained environment share its tool schema.
2. **A second harness at evaluation time.** [[qwen3-coder-next]] §5.1 evaluates every model on
   SWE-Agent, MiniSWE-Agent and OpenHands separately, with 300 maximum turns and hacking-free
   protections, and replicates all baselines per scaffold. The spread across scaffolds for one model
   is the harness-specificity measurement.
3. **Non-agentic benchmarks before and after.** [[kimi-k2-5]] Table 2 is the template: run the same
   text benchmarks on the checkpoint before and after the agentic stage.
4. **Reward-hacking probes.** [[qwen3-coder-next]] §4.2.4 shows that removing git remotes is not
   enough; agents reconnected through `git remote add`, `git clone` and `curl`. Log blocked tool
   calls as a rate, not as a one-time audit.
5. **Coverage.** Log pass@1 and pass@k at large k. No agentic report in this chapter does this beyond
   [[deepswe]]'s pass@16, so the diagnostic is available and unused.

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Training and evaluating on one agent harness | Benchmark score high on the training scaffold, much lower on another with the same model | Evaluate on at least two scaffolds with identical turn limits ([[qwen3-coder-next]] §5.1); report the spread |
| Re-tokenizing rollout text on the trainer side | Loss and reward look normal; gradients are attached to the wrong tokens near tool-call boundaries | Adopt token-in-token-out and compare stored rollout token ids against retokenized text on a sample ([[glm-5]] §4.1.2, [[kimi-k2-5]] App. D) |
| Scoring sandbox crashes as model failures | Reward variance spikes on specific repositories; reward correlates with environment rather than difficulty | Record a failure reason per sample and exclude environment collapse ([[glm-5]] §4.1.2) |
| Dropping invalid samples without padding the group | Group baselines drift upward; correct-but-hard trajectories acquire negative advantages | Pad the group when valid samples exceed G/2, otherwise drop the whole group ([[glm-5]] §4.1.2) |
| Greedy asynchronous scheduling | Early training dominated by short easy tasks, later by clustered hard ones; reward curve non-monotonic for distributional reasons | Use a bounded visibility window over the generation queue ([[minimax-forge-agent-rl]] §3.1) or a staleness threshold ([[glm-5]] §4.1.2) |
| Applying context management only at inference | Agent behaves inconsistently right after a context fold; training reward does not predict deployed accuracy | Train under the same context policy, or model context management as an action ([[minimax-forge-agent-rl]] §2.2) |
| A fixed token budget from step 0 | Output length falls and stays low; the model cannot use a larger inference budget | Alternate budget-limited and free phases ([[kimi-k2-5]] Eq. 2); measure accuracy at two inference budgets |
| Reusing SFT prompts as RL prompts | RL reward rises immediately with no benchmark movement | Keep SFT and RL prompt sets disjoint ([[qwen3-coder-next]] §4.2.1) and filter by pass rate |
| Leaving future commits reachable in SWE environments | Anomalously high resolve rate on instances whose repository has network access | Block tool calls combining a repository URL with git/curl/wget and log the block rate ([[qwen3-coder-next]] §4.2.4) |
| Treating an SFT-effective dataset as an RL prompt pool | High solve-none rate; most groups have zero advantage | Measure the pass-rate distribution under the current policy before the run ([[deepswe]] §6) |

## Check your understanding

1. GLM-5 grows its total token budget by a factor of 1.24 over GLM-4.5, its agent-containing
   mid-training tokens by a factor of 5.5 (100B → 550B), and the share of the budget those tokens
   occupy by a factor of about 4.4 (0.43% → 1.9%). Explain which of the three numbers is the one that
   could predict agentic capability, and name the measurement that would be needed to confirm it.
2. DeepSeek-V3.2 spends 5.2% of its agent prompts on synthesized general-agent tasks and uses exactly
   that slice to demonstrate out-of-domain generalization. Why is a synthetic slice the natural place
   for a transfer result, and what would falsify the interpretation?
3. Both GLM-5 and Kimi K3 replace the group-relative advantage with a teacher log-ratio during
   distillation. Derive why the group size can then drop to 1, and state what is lost when it does.
4. Nemotron 3 Super separates SWE-RL from multi-environment RLVR for throughput reasons, and
   Nemotron 3 Super also reports that single-environment training causes regressions elsewhere. Are
   these two statements in tension? Explain what makes the separation acceptable in their pipeline.
5. Four reports mask negative samples rather than down-weighting them (DeepSeek-V3.2 sequence
   masking, DeepSWE compact filtering, Tongyi over-length exclusion, GLM-5 environment-failure
   exclusion). Using the softmax logit gradient, explain why masking is the right operation when the
   sign of the signal is untrustworthy, and why down-weighting is not equivalent.
6. Qwen3-Coder-Next states a ceiling on synthetic mid-training data and Kimi K2.5 names
   length-overfitting under fixed budgets. Both are over-specialization failures at different stages.
   Construct the common mechanism in one paragraph.
7. Kimi K2.5's PARL anneals λ1 and λ2 to zero over training. Explain what would go wrong if they were
   held constant, and what would go wrong if they were zero from step 0.
8. Only three reports in §7 run a non-agentic evaluation before and after the agentic stage. Design
   the minimum evaluation protocol that would let a lab claim its agentic training did not narrow the
   model, and say which of this chapter's recipes would pass it on the evidence they publish.

## Connections

- **Previous chapter:** ch-45c — Context Management for Long-Horizon Agents. §5.4 here reads the
  context policies that chapter derives as recipe selections, with the BrowseComp deltas each lab
  reports.
- **Dependency:** ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data. §4 here is the
  agentic special case of that chapter's question, with the on-policy variants added.
- **Next chapter:** ch-46 — Lab: DPO or RLVR Experiment with Negative-Signal Ablation and Held-Out
  Capability Retention. The held-out-retention design in that lab is the small-scale version of the
  §7 column that most of these reports leave empty.
- Earlier in the track: ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and
  Stability (the mechanics configured in §5); ch-45a — Preference-Optimization and RL Stage Recipes
  Side by Side (the same cross-lab reading for non-agentic stages); ch-45 — Self-Improvement Loops and
  Multi-Stage Reasoning Pipelines (GLM-4.5's iterative self-distillation).
- Earlier stages: ch-32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data
  Before Post-Training (§2 here); ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and
  Agentic and Long-Context Shares (§3 here); ch-29c — Agentic Environment and Task Synthesis at Scale
  and ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories (§3 and §5.1 here).

## Sources

- [[kimi-k2]], [[kimi-k2-recipe]] — one joint RL stage, MCP plus synthetic tool synthesis, budget
  control, PTX loss, 10,000-instance sandbox; the 2025 baseline for the stage map.
- [[kimi-k2-agentic-data]] — the K2 tool-use synthesis section read on its own.
- [[kimi-k2-5]] — zero-vision SFT, token-level log-ratio clipping, Toggle budget schedule, PARL and
  Agent Swarm, and the one before/after general-benchmark table in this chapter.
- [[kimi-k3]] — nine experts across three domains and three reasoning efforts, MOPD, partial rollout
  across iterations, budget reward −1, unified white-box harness environment.
- [[glm-4-5]], [[glm-4-5-recipe]] — three experts distilled by unified SFT, 100B-token agent
  mid-training stage, agentic RL on search and SWE with format-violation reward 0.
- [[glm-5]] — three-stage mid-training to 200K, asynchronous agentic RL with staleness filtering and
  token-in-token-out, on-policy cross-stage distillation, keep-recent-k context management.
- [[deepseek-v3.1]], [[deepseek-v3.1-recipe]] — specialist distillation followed by one mixed RL
  stage, the Table 1 agent-task counts, Off-Policy Sequence Masking, unseen-environment evidence.
- [[qwen3-coder]] — 7.5T tokens at 70% code, 20,000 parallel environments; the earliest scale point
  for agentic RL infrastructure in this set.
- [[qwen3-coder-next]] — the minimum-synthetic-data principle, the cross-scaffold transfer
  measurement, four experts distilled into one model, reward-hacking blocker.
- [[nemotron-3-super]] — the most disclosed open post-training blend: 7M SFT samples, 21 RL
  environments, the all-environments-versus-single-environment finding, PivotRL.
- [[tongyi-deepresearch]] — Agentic CPT as a named mid-training stage, three environment types,
  strictly on-policy RL with selective negative exclusion.
- [[minimax-m2-aligning-to-what]] — the tool-scaling negative result and the perturbation view of
  agent generalization.
- [[minimax-forge-agent-rl]] — unified mixed-domain CISPO, context management as an action, windowed
  FIFO scheduling, prefix-tree merging.
- [[deepswe]], [[deepswe-recipe]] — RL-only training with GRPO++ and compact filtering; the only
  pass@k number after agentic RL in this chapter.
- [[toucan-mcp]] — 1.5M MCP trajectories and the BFCL V3 gains from its 119.3K SFT subset.
- [[agent-data-protocol]] — the interlingua schema and the cross-task transfer table that measures
  negative transfer from single-domain tuning.
- [[swe-smith]] — 50,137 instances from 128 repositories at $1360 and 295 GB; the trajectory-capping
  rule against repeatedly-solved instances.
- [[r2e-gym]], [[swe-gym]] — the executable-environment baselines that the [[swe-smith]] storage and
  cost comparison is drawn against.
- [[grpo]] — the group-relative baseline whose role the distillation objectives of §4 replace.
- [[rlvr-beyond-base-model]] — the pass@k coverage question that no agentic report in this chapter
  answers.
