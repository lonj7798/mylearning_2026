<!-- chapter: ch-32d
     track: midtraining
     kind: content
     title: Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training
     deps: [ch-32b]
     sources: [[glm-4-5]], [[glm-4-5-recipe]], [[glm-5]], [[agentfounder]], [[cwm-code-world-model]], [[kimi-dev]], [[midtool]], [[state2state]], [[skill-pretraining]], [[kimi-k2]], [[kimi-k2-recipe]], [[kimi-k2-agentic-data]], [[kimi-k2-5]], [[agent-early-experience]]
     figures: figures/agentic-stage-map.html
     revised: 2026-09 (generality revision)
-->

# Chapter 32d — Agentic Mid-Training: Repository, Execution-Trace, and Trajectory Data Before Post-Training

> **Core insight.** In the reports read for this chapter, model-generated agent trajectories enter training in mid-training or continued pre-training stages (GLM-4.5 at 128K, GLM-5 at 128K and 200K, CWM at 131K, AgentFounder at 32K and 128K, Kimi-Dev at 32K, MidTool at 8K), and none of these reports places them in the main general pre-training phase. Kimi K2 puts its agentic data synthesis in post-training SFT, not in pre-training ([[kimi-k2]] §2.2, §3.1.1). Controlled comparisons show agentic gains after the same post-training: +3.7 SWE-bench Verified points from adding ForagerAgent trajectories to an 8B mid-training mix ([[cwm-code-world-model]] Table 4), and +10.5 BFCL overall points at 4B from a 20.3B-token tool-use mix ([[midtool]] Table 6). Only one study measures general ability under matched token budgets: skill-document mid-training changed a six-benchmark general average by −0.85 to +0.51, while trajectory mid-training lowered MMLU by 4.12 points against a general-data control in one 3B setting ([[skill-pretraining]] Table 1).
>
> **Guideline.** When adding agentic data to a mid-training stage, keep general data in the mix and select its share against a matched-token general-data control, because CWM keeps 30% rehearsal of the pre-training mix "as this proved essential in retaining performance on standard evaluations" ([[cwm-code-world-model]] §4.2) and SPT's 30% skill / 70% general mixture beat pure skill data on agentic score (38.60 vs 22.06 at 1.6B, [[skill-pretraining]] Table 12). When a report claims an effect of agentic mid-training on general ability, check that it evaluates the same checkpoint before and after the stage on non-agentic benchmarks; otherwise treat the claim as not measured. When agent trajectories are longer than the training window, place them in a long-context stage instead of truncating them, because AgentFounder's two-stage run with a 128K second stage scored higher on all six Table 4 metrics than a single 32K stage that truncated some data at the same 50B-token budget ([[agentfounder]] Table 4).

## Why this chapter matters for a general-purpose model

Mid-training is continued training of a pre-trained checkpoint on curated or domain-weighted data before post-training ([[skill-pretraining]], Related Work). Agentic mid-training adds data about acting in an environment: repository code with issues, pull requests (PRs), and commits; execution traces that pair code with program states; agent trajectories (sequences of reasoning, tool call, and tool or environment response); tool documentation; and, in one paper, reinforcement learning (RL) on environment-derived goals.

In the pipeline pre-training → mid-training → SFT → preference optimization → RL → evaluation, this chapter sits after context extension (ch-32b) because long agent trajectories are long-context data. Three measurable problems follow.

1. **Placement errors.** A claim about where agentic data enters training can be wrong by a whole stage. The earlier version of ch-27 and the unverified card [[kimi-k2-agentic-data]] state that Kimi K2 mixed about 1T tokens of synthetic agent trajectories into pre-training at 3-5%. The Kimi K2 report describes a 15.5T-token pre-training corpus of web text, code, mathematics, and knowledge, and places its agentic data synthesis in SFT ([[kimi-k2]] §2.2, §3.1.1).
2. **Narrowing.** Mid-training on a narrow agentic distribution can lower scores outside that distribution. The size of this effect is measured in one study (§8, §9).
3. **Attribution.** The GLM-5 versus GLM-4.5 comparison changes model size, data, token count, and attention type at once, and CWM prints no row for its checkpoint before mid-training (§2, §4), so a base-model score after agentic mid-training cannot be attributed to the agentic data without a matched control.

The [stage map figure](figures/agentic-stage-map.html) lets the reader filter the reports in this chapter by where agent trajectories appear, see what each report measured about general ability, and check unit conversions for upsampled sources and long trajectories.

## §1 Stage map: where agentic data enters, report by report

**Definition.** A stage map lists, for one model, which data types are used in pre-training, mid-training, SFT, and RL, with the token budget and context length of each stage.

**Problem.** Without a locus for each entry, stage placement is copied between documents and errors spread. The measurable check is whether each entry can be found in the primary report at a stated section.

| Report | Main pre-training | Mid-training or continued pre-training | First stage with agent trajectories | Locus |
|---|---|---|---|---|
| Kimi K2 (1.04T total / 32B activated) | 15.5T at 4K: web, code, math, knowledge | anneal 400B at 4K, then 60B at 32K; YaRN to 128K; agent data not listed | SFT (synthetic tool-use trajectories) | [[kimi-k2]] §2.2, §2.5, §3.1.1 |
| Kimi K2.5 | 15T joint vision-text at 4K; code upweighted with repository code, issues, code reviews, commits; GUI screenshots and action trajectories "including human-annotated demonstrations" | 32K → 262K, 500B → 200B: high-quality text and multimodal, long text, long video, reasoning, long-CoT | pre-training (GUI action trajectories; share and model-generated fraction not reported) | [[kimi-k2-5]] Table 3, App. B.2-B.3 |
| GLM-4.5 (355B / 32B activated) | 15T + 7T at 4K | 500B repo code at 32K; 500B synthetic reasoning at 32K; 100B long documents + synthetic agent trajectories at 128K | mid-training, 128K stage | [[glm-4-5]] §2.3, Figure 3 |
| GLM-5 (744B / 40B activated) | 27T corpus | 1T at 32K; 500B at 128K; 50B at 200K; agent trajectories up-sampled in later stages | mid-training, later stages | [[glm-5]] §1, §2.3 |
| CWM (32B dense) | 8T at 8K | 5T at 131K: 30% CWM-specific (traces, ForagerAgent), 40% code, 30% rehearsal | mid-training | [[cwm-code-world-model]] Figure 1, §4.2 |
| AgentFounder-30B (from Qwen3-30B-A3B-Base) | not part of the paper | ≈200B at 32K; 100B at 128K | continued pre-training | [[agentfounder]] §2.1 |
| Kimi-Dev (from Qwen2.5-72B-Base) | not part of the paper | ≈150B seen at 32K | mid-training | [[kimi-dev]] §3.2, App. A |
| MidTool (from Qwen3-4B/8B-Base) | not part of the paper | 20.3B at 8,192 | mid-training | [[midtool]] Table 2, Table 10 |

**Result.** In every report above that uses model-generated agent trajectories before post-training, they enter after the main pre-training phase. Kimi K2.5 is the one report with action trajectories in its pre-training corpus; the report says the collection includes human-annotated demonstrations and does not give a token share ([[kimi-k2-5]] App. B.3). The statement "no frontier report places synthetic agent trajectories in pre-training" is limited to the reports read here (Result, per report; the absence claim is an Open question beyond this set).

**Correction to ch-27 (the version the learner studied).** "Kimi-K2's pretraining-mix injection" and "top-scoring trajectories enter the pretraining mix at ~3–5%" → Kimi K2 pre-training uses web text, code, mathematics, and knowledge with rephrasing; its tool-use synthesis (3000+ real MCP tools, over 20,000 synthetic tools, simulated users and tools, LLM judge) produces SFT data ([[kimi-k2]] §2.2, §3.1.1; card Verification lists "any injection of agentic trajectories into pre-training data" as not reported). Attested agentic mid-training comes from GLM-4.5, GLM-5, CWM, AgentFounder, and Kimi-Dev.

**Implication.** A recipe that copies "agent trajectories in pre-training" has no support in these reports. The supported placements are a long-context mid-training stage (GLM, CWM, AgentFounder) or a domain mid-training stage from a released base model (Kimi-Dev, MidTool).

## §2 GLM-4.5 and GLM-5: repository data at 32K, agent trajectories in long-context stages

**Definition.** Repository-level code training concatenates files from one repository, and related issues, PRs, and commits, into one training sequence ([[glm-4-5]] §2.3).

**Mechanism (GLM-4.5, [[glm-4-5]] §2.3, Figure 3).**
1. Pre-train on 15T general tokens, then 7T tokens with code, math, and science up-sampled, at 4K.
2. Repo-level code stage at 32K: files of one repository; model-filtered issues, PRs, and commits concatenated with commits in a diff-like format (500B tokens).
3. Synthetic reasoning stage at 32K: reasoning processes generated by a reasoning model for collected questions (500B tokens).
4. Long-context and agent stage at 128K: long documents up-sampled and "Large-scale synthetic agent trajectories are also incorporated at this stage" (100B tokens).
5. Best-fit packing only in mid-training, to avoid truncating reasoning or repository code; RoPE base 10,000 → 1,000,000 at 32K; cosine LR from 2.5e-4 decays to 2.5e-5 at the end of mid-training ([[glm-4-5-recipe]]).

**GLM-5 changes ([[glm-5]] §2.3, App. A).** Stages are 32K (1T tokens), 128K (500B tokens), and 200K (50B tokens); "Long documents and synthetic agent trajectories are up-sampled at the later stages accordingly." Relaxed repository filtering yields about 10 million issue–PR pairs and about 160B unique issue–PR tokens. The mid-training LR decreases linearly from 4e-5 to 1e-5.

**Worked example (derived shares).**
- GLM-4.5: stage sum 15T + 7T + 0.5T + 0.5T + 0.1T = 23.1T, consistent with the reported 23T. Mid-training share = 1.1 / 23.1 = 4.8%. The agent-bearing 128K stage is 0.1 / 23.1 = 0.43% of all tokens.
- GLM-5: 27T + 1T + 0.5T + 0.05T = 28.55T, consistent with the reported 28.5T. Mid-training share = 1.55 / 28.55 = 5.4%; the 200K stage is 0.05 / 28.55 = 0.18%.
- The agent-trajectory share inside each stage is not reported, so the agent-trajectory share of all tokens cannot be computed.

**Evidence on general ability.** GLM-4.5 prints a base-model table (MMLU 86.1, GSM8K 79.4, MATH 61.0, EvalPlus 78.1) without a before/after-mid-training comparison ([[glm-4-5]] §4.1 Table 2). GLM-5-Base against GLM-4.5-Base: MMLU 86.1 → 88.3, EvalPlus 78.1 → 87.0, GSM8K 79.4 → 68.8, MATH 61.0 → 56.4 ([[glm-5]] App. B.1 Table 11). Model size, attention, data, and tokens all differ between the two, so the GSM8K and MATH decreases cannot be attributed to mid-training (Result, uncontrolled). GLM-5 states that the 200K stage "further bolstered the model's performance even within the 128K context window" without numbers (Result, qualitative).

**Conditions and limits.** Neither report ablates the agent trajectories, gives their share, or states how the mid-training trajectories were generated.

**Implication.** The GLM reports show a placement (agent trajectories with long documents, at the longest context stages) and budgets, not an effect size.

## §3 Agentic continued pre-training: AgentFounder

**Definition.** Agentic continued pre-training (Agentic CPT) is next-token prediction on synthesized agent data, inserted between a released base model and post-training ([[agentfounder]] §2.1).

**Problem as stated by the authors.** The authors state that the absence of agentic foundation models "forces models during post-training to simultaneously learn diverse agentic behaviors while aligning them to expert demonstrations, thereby creating fundamental optimization tensions" ([[agentfounder]] Abstract). This is the authors' explanation (Interpretation). The measurable version is whether the same SFT data gives higher agent scores from a CPT base than from the original base.

**Mechanism ([[agentfounder]] §2.1-§2.3).**
1. Build an entity-anchored knowledge memory from web data, tool-call results, Wikipedia, and discarded post-training trajectories.
2. First-order Action Synthesis (FAS): generate multi-style questions; generate problem analyses with first-step tool calls, and two-step reasoning answers, without calling tools; filter with an LLM judge.
3. High-order Action Synthesis (HAS): for each step of an existing trajectory, generate N alternative "thought and invocation" candidates, shuffle them with the original step, write "I will choose option n_k", append the real response R_k, and end with "My decision is {Correct/Incorrect}".
4. Stage 1: ≈200B tokens at 32K (FAS, short HAS, knowledge reasoning). Stage 2: 100B tokens at 128K of high-quality HAS data.

**Formula.** A trajectory with K steps and N alternatives per step contains (N + 1) × K candidate reasoning-actions, of which K were executed. K: number of steps; N: generated alternatives per step. Worked example: K = 4, N = 3 gives 16 candidates and 4 real environment responses, so 12 of 16 candidates carry no observed consequence.

**Evidence.**
- Same SFT data, Qwen3-30B-A3B-Base vs AgentFounder-30B-Base, Pass@1 (Table 3): with SFT-B, BrowseComp-en 28.6 → 39.9, BrowseComp-zh 35.6 → 43.3, GAIA 71.8 → 72.8, HLE 27.0 → 31.5. Average gains for SFT-A/B/C: 5.75%, 6.13%, 6.45% (Result, single study; seeds not reported).
- Two stages vs one stage at 50B tokens (Table 4): BrowseComp-en Pass@1 31.4 → 35.5, BrowseComp-zh 34.3 → 37.2, GAIA 69.9 → 72.8.
- Data type at ≈50B tokens (Table 5): FAS+HAS vs FAS gives BrowseComp-zh Pass@1 37.0 → 40.1 but GAIA 72.8 → 69.9; the authors treat the GAIA change as evaluation fluctuation (Interpretation).
- Data scaling (§3.5.2): average Pass@3 from 54.2% to 62.2% over 315B tokens, with 3.8 points "within the initial 15B tokens".
- SFT loss with identical SFT data after 1,340 steps: 0.8656 (baseline) vs 0.7953 (315B CPT) (§3.6.1). Lower SFT loss is an optimization measurement, not a capability measurement.

**Worked example (diminishing returns).** First 15B tokens: 3.8 / 15 = 0.25 points per billion tokens. Remaining 300B tokens: (8.0 − 3.8) / 300 = 0.014 points per billion tokens. The marginal gain per token is 18 times lower after the first 15B tokens (derived from §3.5.2).

**Conditions and limits.** The only non-deep-research evaluation is ACEBench, 67.2 for released Qwen3-30B-A3B vs 70.0 for AgentFounder-30B (Table 6); these two models did not receive the same post-training. No knowledge, math, or chat benchmark is reported, and CPT learning rate, batch size, and replay are not reported.

**Implication.** AgentFounder is evidence that CPT on synthesized agent data raises deep-research scores after fixed SFT. It is not evidence about retention of general ability.

## §4 Execution traces and Docker trajectories: CWM

**Definitions.** An execution trace records, for each executed line, the local variables before and after execution ([[cwm-code-world-model]] §2.2). A ForagerAgent trajectory records an LLM agent's shell commands and file edits and the environment's responses in a Docker image of a real repository (§2.3).

**Problem.** Code is usually trained as static text; the authors state that "teaching LLMs such code world modeling capabilities is typically not considered before post-training" (§1). The measurable question is whether trace and trajectory data in mid-training raise SWE-bench Verified after the same SFT.

**Mechanism ([[cwm-code-world-model]] §2.2-§2.3, §4.2).**
1. Build executable repository images; trace 120M+ Python functions, CodeContests solutions (balanced between incorrect and correct submissions), and about 70k repository commits.
2. Format each trace as source context, then frames separated by custom tokens (`<|frame_sep|>`, `<|line_sep|>`, `<|action_sep|>`), each frame a JSON dictionary of local variables plus the executed line.
3. Run ForagerAgent on mutate-fix tasks (a synthetic bug introduced into a tested function) and issue-fix tasks (real issues); remove SWE-bench repositories and forks; keep 3M trajectories after MinHash deduplication at Jaccard < 0.5.
4. Mid-train 5T tokens at 131K with a global batch of 33M tokens: 30% CWM-specific data, 40% general code, 30% rehearsal of the pre-training mix.
5. Set each dataset's share so that it reaches a target of 1 to 4 epochs at the end of mid-training, with targets chosen by scaling-law experiments on epoching.

**Formula (share from target epochs, §4.2 described in words).** p_d = e_d × U_d / T. p_d: sampling share of dataset d; e_d: target epochs; U_d: unique tokens of d; T: mid-training token budget. Worked example with a hypothetical dataset: U_d = 120B, e_d = 4, T = 5,000B gives p_d = 480 / 5,000 = 9.6%. With e_d = 1 the share is 2.4%.

**Evidence (8B, 6T pre-training + 1T mid-training ablation, SFT without RL; Table 4).**

| PRs | Tracing | Forager | CruxEval-O | CruxEval-I | Agentic SBV NLL | SBV pass@1 |
|---|---|---|---|---|---|---|
| ✗ | ✗ | ✗ | 45.4 | 44.1 | 0.39 | 14.6 |
| ✓ | ✗ | ✗ | 44.6 | 45.8 | 0.37 | 18.6 |
| ✓ | ✓ | ✗ | 73.9 | 51.5 | 0.38 | 18.4 |
| ✓ | ✓ | ✓ | 74.5 | 54.8 | 0.29 | 22.1 |

Tracing data raises CruxEval-O by 29.3 points and leaves SWE-bench Verified (SBV) unchanged; ForagerAgent data lowers agentic SBV negative log-likelihood from 0.38 to 0.29 and raises SBV pass@1 by 3.7 points (Result, single study; seeds not reported).

**General ability.** Table 13 reports CWM after mid-training (CWM-Mid): MMLU 73.6, MMLU-Pro 52.3, GPQA 31.7, GSM8k 84.7, against Qwen3-32B at 83.6, 65.5, 49.5, 93.4. No row exists for the 8T checkpoint before mid-training, so the effect of the 5T mid-training stage on these benchmarks is not measured. The 30% rehearsal share is described as essential for standard evaluations without a printed ablation (§4.2). The authors state that CWM is not intended as a general-purpose chatbot and omit an RLHF stage (§5).

**Implication.** CWM is the only report in this chapter with a controlled ablation of agent trajectories inside a mid-training mix. The ablation measures code and SWE metrics only.

## §5 Agentless training as a skill prior: Kimi-Dev

**Definition.** An Agentless format splits issue resolution into fixed single-turn steps (file localization, code edit, test writing) instead of free multi-turn agent interaction; each step "could be optimized separately as a single-turn problem with verifiable rewards" ([[kimi-dev]] §2.1). A skill prior is a checkpoint used as the starting point for later agent fine-tuning.

**Mechanism ([[kimi-dev]] §3.2, App. A).**
1. Start from Qwen2.5-72B-Base.
2. Build ≈50B tokens of natural diff patches in Agentless prompt templates, with loss masked on the prompt; ≈20B tokens of PR commit packs (commit message followed by code change).
3. Build ≈10B tokens of localization rollouts, keeping "only the rollouts that achieve exactly correct file localizations".
4. Build ≈10B tokens of simulated agent interactions with non-executing file-view and search tools; the loss mask covers only the system prompt, so the model learns actions and observations; the agent opens wrongly localized files and the data inserts "I realize that I do not need to modify this file".
5. Upsample the synthetic part 4×; train at 32K, global batch 256, LR 2e-5 cosine to 2e-6, about 3B warm-up tokens, until about 150B tokens.

**Worked example (unique vs seen tokens).** Unique tokens: 50 + 20 + 20 = 90B. Seen tokens: 50 + 20 + 20 × 4 = 150B, consistent with the reported ≈150B (derived). The synthetic share is 20 / 90 = 22.2% of unique tokens and 80 / 150 = 53.3% of seen tokens. Quoting the unique-token share (22.2%) instead of the seen-token share (53.3%) understates the synthetic share by a factor of 53.3 / 22.2 = 2.4.

**Evidence.**
- Mid-training budget: 50B, 100B, and ≈150B subsets, each followed by the same light SFT on 2,000 pairs, give increasing BugFixer pass@1 (Figure 2; values printed only in the figure).
- Prior comparison (§4.2, Figure 5): the RL prior reaches the Base prior's best pass@1 with 2²³ SWE-Agent SFT tokens instead of 1.5 × 2²⁸, a 48-fold reduction (derived: 1.5 × 2⁵ = 48). The mid-trained (MT) prior lags in zero-shot and one-step settings and "quickly becomes on par" with SFT and RL priors after 200 trajectories (2²⁴ tokens).
- End-to-end RL after one SFT gradient step (§4.2): 260 of 6,202 problems have pass@8 > 0 for the MT prior against 2,062 for the SFT and RL priors; MT pass@1 fell below 2% after 10 RL steps. Figure 7 plots only the SFT-prior and RL-prior runs.

**Conditions and limits.** Evaluation covers SWE-bench Verified, SWE-bench-Live, and SWE-bench Multilingual only. The MT prior alone was the weakest starting point for end-to-end RL, so mid-training was one part of a mid-training → SFT → RL prior.

**Implication.** Agentic mid-training raised later-stage results in this study only when it was followed by reasoning SFT or RL before end-to-end agent RL.

## §6 Tool-use mid-training data synthesis: MidTool

**Definition.** MidTool-Mix is a 20.3B-token mid-training corpus of filtered web, PDF, and code documents, QA and trajectories grounded in those documents, and native tool-use trajectories built from real APIs and MCP skills ([[midtool]] §2, Table 2).

**Mechanism ([[midtool]] §2.3-§2.4, §3.1).**
1. Collect FineWeb dumps (2020-2025), English FinePDFs, GitHub code, and tool artifacts; filter with fastText classifiers and deduplicate.
2. Context-grounded augmentation: turn documents into QA and trajectories about tool boundaries, parameters, and workflows.
3. Native trajectories: score tool sources, synthesize personas and plans, generate with GPT-5-family models, validate turn order, schema grounding, required arguments, and response consistency; add AWM rollouts and filtered Nemotron Agentic traces.
4. Normalize all trajectories to a plain chat template "without special control tokens".
5. Mid-train Qwen3-4B/8B-Base for 1 epoch at 8,192 tokens, LR 3 × 10⁻⁵ (WSD), 4M-token batches; then SFT on 100K TOUCAN samples; optional GRPO for 64 steps.

**Worked example (composition).** Source tokens 4.4 + 2.6 + 3.8 = 10.8B; augmentation 4.1 + 2.1 + 1.5 = 7.7B; native trajectories 1.8B; total 20.3B. Native trajectories are 1.8 / 20.3 = 8.9% of tokens, printed as 9%.

**Evidence (Qwen3-4B-Base + SFT, only the mid-training corpus changes; Table 6).**

| Mid-training data | BFCL overall | τ²-Bench Pass@1 | MCP-Universe score |
|---|---|---|---|
| none | 39.73% | 8.54% | 13.20 |
| Dolmino-20BT (matched budget, general) | 43.10% | 7.37% | 5.41 |
| processed sources, no trajectories | 42.30% | 7.30% | 12.20 |
| processed sources + native agentic trajectories | 47.59% | 4.23% | 6.80 |
| processed sources + context-grounded trajectories | 44.66% | 8.99% | 8.46 |
| MidTool-Mix (all) | 50.25% | 12.23% | 18.66 |

The native-trajectory and context-grounded rows each add one synthesis branch to the processed sources; they are alternatives, not successive steps (Table 6 caption). Processed sources plus native trajectories raise BFCL by 7.9 points and lower τ²-Bench Pass@1 by 4.3 points relative to no mid-training. Only the full mixture is above the no-mid-training row on all eight Table 6 metrics (derived from Table 6; Result, single study; seeds not reported). The variant rows are not matched in token budget; only Dolmino-20BT is budget-matched to MidTool-Mix, and the authors list budget-matched variants as future work (App. D). At 8B, BFCL overall is 47.62% → 51.12% after SFT (Table 3).

**Conditions and limits.** In the SFT rows of Table 3, the BFCL hallucination column (higher is better) falls from 60.46% to 56.95% (4B) and from 65.03% to 59.82% (8B) with MidTool-Mix; after RL it is higher with MidTool-Mix (4B 55.19% → 60.10%; 8B 52.79% → 64.22%). τ²-Bench telecom Pass@1 after SFT falls from 3.73% to 1.54% at 4B and from 4.39% to 1.54% at 8B (Table 4). MCP-Universe web search stays at 0.00 in every row (Table 5). No non-tool-use benchmark is reported.

**Implication.** In this ablation, adding native trajectories to the processed sources without the context-grounded branch lowered τ²-Bench and MCP-Universe scores below the no-mid-training row; the full mixture was the only variant above that row on all three benchmarks.

## §7 Environment-derived mid-training: State2State

**Definition.** State2State is an RL stage in which the goal is to reach an environment observation that a random explorer reached earlier; it runs before RL on human-specified tasks ([[state2state]] §3). It differs from the other methods in this chapter because it uses a policy-gradient objective, not next-token prediction on a corpus.

**Mechanism ([[state2state]] §3.2-§3.4).**
1. Explore with a random policy over valid actions.
2. Remove invalid observations; sample diverse targets; replay each exploration trajectory 3 times and keep only reproducible targets.
3. Reward r_t = 1 if match(o(s_t), o⋆), else 0. o(s_t): observation at state s_t; o⋆: target observation; match: normalized exact match.
4. Optimize with GRPO and dynamic sampling (discard groups whose rewards are all equal) for 80 steps, and continue downstream training from the checkpoint with the best state-reaching validation score (§4.3); LR 1e-6, KL loss coefficient 0.01, batch 16, group size 8 (App. E.1).

**Worked example (group advantages).** In a group of 8 rollouts with 2 successes, the mean reward is 0.25 and the standard deviation is √(0.25 × 0.75) = 0.433. A success gets advantage (1 − 0.25) / 0.433 = +1.73; a failure gets (0 − 0.25) / 0.433 = −0.58. A group with 0 or 8 successes has standard deviation 0 and is discarded by dynamic sampling.

**Evidence ([[state2state]] Table 1, average of ID and OOD success).** Qwen3-4B: task RL 86.13 (ALFWorld) and 49.63 (ScienceWorld); State2State + RL 92.00 and 55.50. Qwen3-8B: 92.36 and 52.13 vs 97.45 and 56.00. Standalone State2State lowers Qwen3-4B on ScienceWorld from 23.63 to 21.00. Order matters (Table 2, Qwen3-4B, ScienceWorld average): SFT + State2State + RL gives 65.88 vs 60.50 for State2State + SFT + RL. Cross-environment (Table 5): ALFWorld RL after ScienceWorld State2State gives 89.44, vs 86.87 after ScienceWorld task RL and 86.13 with no mid-stage.

**Conditions and limits.** Two text environments plus a GUI subset; no general benchmark; seeds not reported. The authors attribute the order effect to SFT overwriting environment priors (Interpretation).

## §8 Skills as pre-training data: SPT

**Definition.** Skill Pre-Training (SPT) applies causal language modeling to public skill packages (a SKILL.md instruction file plus references, scripts, and configs), with no trajectories and no environment ([[skill-pretraining]] Skill Pre-Training).

**Formula (Eq. 1).** p_α(x) = α · p_skill(x) + (1 − α) · p_gen(x), with total tokens fixed at B. α: skill fraction; p_skill, p_gen: empirical distributions over skill and general blocks; B: mid-training token budget. Worked example with a hypothetical B = 100M: α = 0.3 gives 30M skill tokens and 70M general tokens.

**Evidence ([[skill-pretraining]] Table 1, equal 347,770,880-token budgets, 5 seeds).** At 7B with xLAM-FC SFT, agentic average: none 28.50, Dolmino 33.33, AgentBank trajectories 44.05, SkillCorpus 53.46; general average 75.31 / 75.16 / 74.38 / 75.07. Across 3 backbones and 2 SFT recipes, SkillCorpus changes the general average by −0.85 to +0.51 relative to direct SFT. In the Instella-3B + xLAM-FC block, AgentBank trajectory mid-training gives MMLU 50.10 and general average 70.34, against 54.22 and 72.05 for Dolmino (−4.12 MMLU, −1.71 general average); SkillCorpus in the same block gives MMLU 51.62 and general average 71.43 (−2.60 MMLU, −0.62 general average against Dolmino). In the other five backbone and SFT blocks, AgentBank MMLU is within 3.0 points of Dolmino (largest gap 60.57 vs 57.60 at 7B + xLAM-FC). Mixture sweep at 1.6B (Table 12): 30% skill gives agentic 38.60 with Dolmino as the general source, against 22.06 for pure skill data, with general 58.64 vs 59.29 for general-only.

**Conditions and limits.** Backbones of 1.6B-7B; one skill repository; no interactive long-horizon evaluation (Limitations).

## §9 Effect on general ability: what each report measures

**Definition.** For this chapter, "measured" means the same model is evaluated on non-agentic benchmarks with and without the agentic mid-training stage, under the same later stages.

| Report | Non-agentic evaluation | Controlled for the agentic stage | Reported effect |
|---|---|---|---|
| GLM-4.5 | base-model table | no | not attributable |
| GLM-5 | base table vs GLM-4.5-Base | no (size, data, attention differ) | GSM8K 79.4 → 68.8, MMLU 86.1 → 88.3; not attributable |
| CWM | CWM-Mid on MMLU, GPQA, GSM8k | no pre-mid row; rehearsal called essential | not measured |
| AgentFounder | none (ACEBench is tool use) | — | not measured |
| Kimi-Dev | none | — | not measured |
| MidTool | none | — | not measured |
| State2State | none | — | not measured |
| SPT | six OLMES tasks, 5 seeds | yes, equal tokens | skills −0.85 to +0.51 general average; trajectories MMLU −4.12 vs general control in one block |
| Kimi K2.5 | MMLU-Pro, GPQA after visual RL (not agentic mid-training) | — | not applicable to agent data |

**Status.** The effect of agentic mid-training on general ability at frontier scale is an Open question. The only controlled numbers come from 1.6B-7B models ([[skill-pretraining]]). Related reward-free evidence at the post-training stage ([[agent-early-experience]]) reports agent benchmarks and no general-capability evaluation (card Verification).

## §10 Long agent trajectories as long-context training data

**Problem.** Agent trajectories accumulate tool outputs, so one trajectory can exceed the pre-training window. A trajectory that is truncated loses its later steps, often including the final answer.

**Mechanism, by report.**
- GLM-4.5 adds agent trajectories at 128K, GLM-5 up-samples them in its 128K and 200K stages ([[glm-4-5]] §2.3; [[glm-5]] §2.3).
- CWM runs all 5T mid-training tokens at 131K because about 30% of mid-training documents exceed 65K tokens; it groups documents into length buckets so that data-parallel ranks do not wait on long documents, and it reports "lackluster performance when training on long-context data at smaller batch sizes" ([[cwm-code-world-model]] §4.2, footnote 4). RULER for the final post-trained CWM, not the mid-training checkpoint: 84.3 at 32k and 69.7 at 128k (App. J Table 14).
- AgentFounder trains long HAS data in a 128K second stage; the single-stage 32K run truncates some HAS data (Table 4, §3.4.1).
- Kimi K2.5 extends from 32,768 to 262,144 tokens in a mid-training stage with long text, long video, reasoning, and long-CoT data ([[kimi-k2-5]] Table 3). Kimi K2 anneals at 4K (400B) and then 32K (60B) before YaRN to 128K ([[kimi-k2]] §2.5).

**Formula.** For a trajectory of length L and a training window W, tokens beyond the window = max(0, L − W). The number of full-length sequences in a stage of D tokens is at most ⌊D / W⌋.

**Worked example.** A 200,000-token trajectory in a 128,000-token window has 72,000 tokens (36%) beyond the window. AgentFounder Stage 2 has D = 100B and W = 128,000, so it holds at most 781,250 full-length sequences (upper bound, derived; the report does not give a sequence count).

**Evidence.** At 50B tokens, Stage 1 & 2 vs Stage 1 only: BrowseComp-en Pass@1 35.5 vs 31.4, Pass@3 52.0 vs 49.9 ([[agentfounder]] Table 4). The authors note that single-stage training at 128K was not evaluated because of cost, so the comparison mixes context length with stage order.

**Conditions and limits.** No report measures short-context regression caused by the agentic long-context stage. Short-context regression from context extension in general is covered in ch-32b.

## §11 What the model is trained to predict in agentic mid-training data

**Definition.** The loss mask m_t selects which tokens contribute to the next-token loss.

**Formula.** L = −(Σ_t m_t log p_θ(x_t | x_<t)) / (Σ_t m_t). x_t: token t; m_t ∈ {0, 1}; p_θ: model distribution.

| Report | Actions (agent tokens) | Observations (tool/environment output) | Masked |
|---|---|---|---|
| CWM ForagerAgent | trained | trained, loss masked for 50% of observations | half of observations, stochastically |
| Kimi-Dev simulated interactions | trained | trained | system prompt only |
| Kimi-Dev diff patches | trained | — | prompt part |
| AgentFounder, MidTool, GLM | not stated beyond next-token prediction | not stated | not reported |

**Worked example (hypothetical trajectory).** A trajectory has 2,000 agent tokens and four observations of 1,500 tokens (6,000 observation tokens). Without masking, observations are 6,000 / 8,000 = 75% of trained tokens. With CWM's 50% observation masking, the expected trained tokens are 2,000 + 3,000 = 5,000, and observations are 3,000 / 5,000 = 60%.

**Interpretation.** Training on observations teaches prediction of environment responses, which Kimi-Dev calls integrating "policy and world modeling" ([[kimi-dev]] App. A). Neither report ablates observation loss against action-only loss.

## Negative samples and negative feedback

Terms follow the four meanings used in ch-31a (Negative Samples in Supervised Training) and ch-43a (Negative Samples and Negative Gradients): negative marginal value, negative as content, negative as conditioning, negative as gradient.

**1. Where negatives come from in this stage.**
- LLM-judge rejection in AgentFounder FAS; App. B.1 reports 50% correct / 50% incorrect before filtering, 43.5% removed, and 82% accuracy after filtering ([[agentfounder]]).
- Exact-localization filtering of Kimi-Dev rollouts; validation failures in MidTool native trajectories, retried with feedback before discard ([[kimi-dev]] App. A; [[midtool]] §2.3).
- Unsuccessful ForagerAgent trajectories and incorrect CodeContests submissions, labeled by tests ([[cwm-code-world-model]] §2.2-§2.3).
- Failed state-reaching rollouts, labeled by exact observation match ([[state2state]] Eq. 1).

**Worked example (derived filter error rates, AgentFounder App. B.1).** Take 100 samples: 50 correct, 50 incorrect. Removing 43.5 leaves 56.5, of which 82% = 46.3 are correct and 10.2 incorrect. The filter removed 3.7 correct samples (7.3% of correct ones) and 39.8 incorrect samples (79.7% of incorrect ones). This assumes the 82% and 50% figures use the same correctness labels.

**2. What current practice does with them.**
- Discard (negative marginal value assumed): AgentFounder FAS, Kimi-Dev localization rollouts, MidTool invalid generations.
- Keep as ordinary targets: CWM does "not filter trajectories based on whether they succeed", for world-model coverage; failed agent actions are trained with cross-entropy like successful ones. None of the four categories fits exactly: the failure is not corrected or labeled.
- Negative as content: AgentFounder HAS appends "My decision is Incorrect" after failed trajectories; Kimi-Dev inserts wrong file openings followed by "I realize that I do not need to modify this file". Both are trained with cross-entropy and remove no probability mass from the failure.
- Negative as gradient: State2State failures receive negative GRPO advantages.
- For contrast, GLM-5 SFT keeps erroneous trajectory segments but masks their loss ([[glm-5]] §3.1); this is an SFT choice, not a mid-training one.

**3. Mechanism for negatives as gradient.** For a softmax over logits z, ∂ log p_y / ∂ z_j = 1[j = y] − p_j. With three actions p = (0.7, 0.2, 0.1), a failed sample y = 3, and advantage A = −0.58, gradient ascent on A · log p_y moves logits in direction A · (1[j = y] − p_j) = −0.58 × (−0.7, −0.2, 0.9) = (+0.41, +0.12, −0.52). To first order, the probability change is Δp_j ∝ p_j (Δz_j − Σ_k p_k Δz_k), with Σ_k p_k Δz_k = 0.26, which gives Δp ∝ (+0.106, −0.028, −0.078): action 1 (p = 0.7) receives all of the removed mass, and action 2 also loses mass. This is how pushing down an unlikely failure concentrates probability on the current mode.

**4. Evidence with numbers.** No mid-training paper in this chapter ablates keeping versus discarding failures. CWM's ForagerAgent data, which includes failures, adds 3.7 SBV points as a whole (Table 4). AgentFounder's FAS+HAS vs FAS comparison (Table 5) mixes the Correct/Incorrect labels with the rest of HAS. State2State improves downstream RL (Table 1) but does not separate positive and negative advantages. No measured share of improvement is attributed to negatives in any of these sources.

**5. Controls.** Localize failures (Kimi-Dev injects a single wrong file opening followed by a correction); label instead of imitate (HAS verdict text); mask uncertain segments (GLM-5 SFT); discard groups without contrast (State2State dynamic sampling); keep general data in the mix (CWM 30% rehearsal).

**6. Diagnostics.** When failures are kept as targets, evaluate the rate of repeated failed actions at inference; log agentic negative log-likelihood on held-out successful trajectories (CWM Table 4 column); for RL-based mid-training, log success rate split by target difficulty and entropy per step.

**7. Effect on generality.** Keeping unfiltered failures as targets could raise the probability of failed actions, and no source measures this (Open question). Hallucination-related scores move in both directions in MidTool (§6), without a negatives ablation.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| GLM-4.5 | 355B total / 32B act. | mid-train | repo-level code tokens; sequence length | 500B; 32K | arXiv:2508.06471v1 Figure 3, §2.3 ([[glm-4-5-recipe]]) | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | long-context | long documents + synthetic agent trajectories | 100B at 128K | Figure 3, §2.3 | verified 2026-09-14 | no ablation reported |
| GLM-4.5 | 355B / 32B act. | mid-train | RoPE base; LR at end of mid-training | 10,000 → 1,000,000 at 32K; cosine to 2.5e-5 | §2.4 | verified 2026-09-14 | no ablation reported |
| GLM-5 | 744B / 40B act. | mid-train / long-context | stage tokens | 32K: 1T; 128K: 500B; 200K: 50B | arXiv:2602.15763v2 §2.3 ([[glm-5]]) | verified 2026-09-15 | §2.3: 200K stage improved performance within 128K (no numbers) |
| GLM-5 | 744B / 40B act. | mid-train | LR schedule | linear 4e-5 → 1e-5 | App. A | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B act. | mid-train | issue–PR data | ≈10M pairs; ≈160B unique tokens; stage and epochs not stated | §2.3 | verified 2026-09-15 | no ablation reported |
| GLM-4.5 / GLM-5 | as above | mid-train | agent-trajectory share per stage; generator model | not reported | GLM-4.5 §2.3 and Figure 3; GLM-5 §2.3 and App. A | not reported | — |
| CWM | 32B | pretrain | tokens; context; global batch | 8T; 8,192; 8.4M tokens | arXiv:2510.02387v1 §4.2 ([[cwm-code-world-model]]) | verified 2026-09-15 | hyper-parameters from scaling laws (§4.3) |
| CWM | 32B | mid-train | tokens; context; global batch | 5T; 131k; 33M tokens | §4.2 | verified 2026-09-15 | footnote 4: smaller batches gave "lackluster" long-context performance (no numbers) |
| CWM | 32B | mid-train | mixture (token share) | CWM-specific 30%; general code 40%; pre-training rehearsal 30% | §4.2 | verified 2026-09-15 | §4.2: rehearsal "proved essential" for standard evaluations (no numbers) |
| CWM | 32B | mid-train | epochs per dataset | 1-4 target epochs; share set to reach target at end of mid-training | §4.2 | verified 2026-09-15 | scaling-law epoching experiments (no table printed) |
| CWM | 32B | pretrain → mid-train | LR schedule; optimizer | 2,000 warmup steps; cosine, peak 8 × 10⁻⁴, 100× decay over 13T horizon, last 5T in mid-training; AdamW (0.9, 0.95), weight decay 0.1, clip 1.0 | §4.1 | verified 2026-09-15 | scaling laws (§4.3) |
| CWM | 32B | mid-train | ForagerAgent loss | agent and environment turns trained; loss masked for 50% of observations; no success filter | §2.3 | verified 2026-09-15 | no ablation reported |
| CWM 8B ablation | 8B | mid-train | budget | 6T pre-training + 1T mid-training | §7.1 Table 4 | verified 2026-09-15 | Table 4: + ForagerAgent SBV 18.4 → 22.1 after SFT |
| CWM | 32B | SFT | tokens; batch; context; LR; rehearsal | 100B; 2M tokens; 32k; 1k warmup then constant 1 × 10⁻⁵; ≈30% from mid-training mix | §5.1 | verified 2026-09-15 | §5.1: constant LR "similar evaluation metrics" to cosine (no numbers) |
| AgentFounder-30B | 30B total (A3B) | mid-train | CPT stage 1 | ≈200B tokens at 32K | arXiv:2509.13310v1 §2.1 ([[agentfounder]]) | verified 2026-09-15 | Table 4 (at 50B): Stage 1 & 2 > Stage 1 only |
| AgentFounder-30B | 30B total (A3B) | long-context | CPT stage 2 | 100B tokens at 128K | §2.1 | verified 2026-09-15 | §3.5.2: +1.8 (65B vs 50B), +1.0 (315B vs 210B) average Pass@3 |
| AgentFounder-30B | 30B total (A3B) | mid-train | LR, batch, general replay | not reported | §2-§3, App. B | not reported | — |
| Kimi-Dev | 72B | mid-train | data (unique tokens) | ≈50B diff patches; ≈20B PR commit packs; ≈20B synthetic, upsampled 4× | arXiv:2509.23045v3 §3.2, App. A ([[kimi-dev]]) | verified 2026-09-15 | Figure 2: 50B < 100B < 150B BugFixer pass@1 (values in figure) |
| Kimi-Dev | 72B | mid-train | seen tokens; context; batch; LR; warmup | ≈150B; 32K; 256 sequences; 2e-5 cosine to 2e-6; ≈3B tokens | App. A | verified 2026-09-15 | no ablation reported |
| MidTool | Qwen3-4B-Base, Qwen3-8B-Base | mid-train | corpus; epochs; sequence length | 20.3B tokens (native trajectories 9%); 1; 8,192 | arXiv:2608.20314v1 Table 2, Table 10 ([[midtool]]) | verified 2026-09-15 | Table 6 corpus ablation (4B + SFT) |
| MidTool | 4B, 8B | mid-train | LR; schedule; warmup; batch; optimizer | 3 × 10⁻⁵; WSD; 50 steps; 4M tokens packed; AdamW (0.9, 0.999), wd 0.01 | Table 10 | verified 2026-09-15 | no ablation reported |
| MidTool | 4B, 8B | SFT | data; length; LR | 100K TOUCAN samples; 32,768; 2 × 10⁻⁵ cosine | §3.1, Table 11 | verified 2026-09-15 | no ablation reported |
| SPT (MeCo-1.6B-DCLM-160B) | 1.6B | mid-train | skill share α; LR; length; epochs | 30% best in sweep; 2 × 10⁻⁵ cosine; 4,096; 1 | arXiv:2608.26563v1 Tables 9, 12 ([[skill-pretraining]]) | verified 2026-09-14 | Table 12: agentic 38.60 (30%) vs 22.06 (100%), 5 seeds |
| State2State | Qwen3-4B, Qwen3-8B | mid-train (RL) | LR; KL loss; batch; group; steps | 1e-6; 0.01; 16; 8; 80 (best validation checkpoint kept) | arXiv:2608.04934v1 App. E.1, §4.3 ([[state2state]]) | verified 2026-09-15 | Table 1: State2State + RL > RL at both sizes |
| Kimi K2 (Kimi-K2-Base) | 1.04T / 32B act. | pretrain-decay/anneal, long-context | annealing and activation | LR 2e-5 → 7e-6; 400B at 4k, then 60B at 32k; YaRN to 128k | arXiv:2507.20534v2 §2.5 ([[kimi-k2-recipe]]) | verified 2026-09-14 | no ablation reported |
| Kimi K2.5 | 1.04T / 32B act. | long-context | joint long-context mid-training | 32,768 → 262,144 tokens; 500B → 200B tokens | arXiv:2602.02276v2 Table 3 ([[kimi-k2-5]]) | verified 2026-09-15 | no ablation reported |

**Starting point for a small general-purpose run.** For a 4B-8B base model, the verified MidTool rows give one epoch of mid-training on a 20.3B-token tool-use corpus at 8,192 tokens, LR 3 × 10⁻⁵ with WSD, 50 warmup steps, and 4M-token batches, followed by SFT at 32,768 tokens and LR 2 × 10⁻⁵; this was run on 32 H200 GPUs ([[midtool]] §3.1). MidTool-Mix contains no separate general replay beyond its filtered web, PDF, and code sources, and no general benchmark was measured. For a 1.6B backbone, the verified SPT rows add a 30% skill-document / 70% general-data split under a fixed token budget, selected at 1.6B with Tulu 3 SFT over 5 seeds ([[skill-pretraining]] Table 12). CWM's 30% rehearsal share is verified at 32B but has no printed ablation.

## Generalization lens

**(a) What increases breadth.**
- Mixing agentic data with general data: SPT mixtures at 20-50% skill share beat both pure skill data and general-only data on agentic score, with general average within 0.65 points of general-only at 30% (Table 12: 58.64 vs 59.29) ([[skill-pretraining]]).
- Rehearsal of the pre-training mix during mid-training and SFT: CWM 30% in mid-training, ≈30% in SFT ([[cwm-code-world-model]] §4.2, §5.1); reported as essential, not ablated (Result, qualitative).
- Heterogeneous agentic sources: in MidTool only the combination of document-grounded data and trajectories improved all eight Table 6 metrics ([[midtool]]).
- Environment-derived objectives transfer across environments in one test: 89.44 vs 86.87 on ALFWorld ([[state2state]] Table 5). Agentless-trained priors generalize to SWE-bench-Live and Multilingual with few trajectories ([[kimi-dev]] App. G.4).

**(b) What causes narrowing or forgetting.**
- Trajectory-only mid-training lowered MMLU by 4.12 points against a general-data control at 3B ([[skill-pretraining]] Table 1).
- Processed sources plus native trajectories, without the context-grounded branch, lowered τ²-Bench Pass@1 by 4.3 points and MCP-Universe score by 6.4 relative to no mid-training ([[midtool]] Table 6).
- After mid-training and SFT, MidTool models score lower on the BFCL hallucination column (higher is better) at both sizes and on τ²-Bench telecom Pass@1 at both sizes ([[midtool]] Tables 3-4).
- A mid-training-only prior can be a poor RL starting point: 260 of 6,202 problems usable vs 2,062 for SFT and RL priors ([[kimi-dev]] Figure 7).
- Later imitation can remove environment-derived priors: State2State before SFT gains less than after SFT (Table 2; authors' Interpretation).

**(c) How to measure it for this stage.**
1. Evaluate the checkpoint before and after the agentic stage on a fixed non-agentic suite (knowledge, math, code, chat), and add a matched-token general-data control (the SPT and MidTool Dolmino design).
2. Evaluate after a fixed SFT recipe as well as at the base checkpoint, because CWM, Kimi-Dev, MidTool, and AgentFounder report their mid-training effects after SFT.
3. Include held-out tool ecosystems and out-of-distribution splits (MCP-Universe; ALFWorld and ScienceWorld OOD).
4. Decontaminate against evaluation repositories and tasks: CWM and Kimi-Dev remove SWE-bench repositories; SPT uses a 13-gram diagnostic and removes packages; MidTool runs DeCon.
5. Report seeds. SPT reports 5; the other papers do not report a seed count.
6. Do not use in-environment RL reward as the generality signal: MidTool's RL reward curves converge while the post-RL benchmark gap persists ([[midtool]] App. C.2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Placing a report's agentic data in the wrong stage | A recipe cites "agentic pre-training" for Kimi K2 | Find the section: [[kimi-k2]] §3.1.1 is under Post-Training |
| Quoting a share in unique tokens when the source upsampled | Synthetic share quoted as 22% for Kimi-Dev | Multiply each source by its upsampling factor; seen share is 53.3% |
| Attributing a base-model score change to agentic mid-training across model versions | "GLM-5's agentic mid-training lowered GSM8K" | Require the same model before and after the stage |
| Using SFT loss as a capability measure | Lower SFT loss reported as better agent ability | Compare benchmark scores; MidTool App. C.1 states loss is not capability |
| Truncating long trajectories at the pre-training window | Final answers or last tool calls absent from training sequences | Histogram trajectory lengths against W; count tokens beyond W |
| Mid-training on trajectories without document or general data | General or transfer benchmark drops (MMLU −4.12 vs general control in SPT at 3B; τ² −4.3 vs no mid-training in MidTool's native-trajectory variant) | Run a matched-token general-data control |
| Keeping failed trajectories as ordinary targets without measurement | Model repeats failed actions | Measure repeated-failure rate at inference; ablate success filtering |
| Evaluating only in the training environment | RL reward converges; benchmark gap unknown | Evaluate on held-out environments after RL |

## Check your understanding

1. The Kimi K2 report and an old course page disagree about where agentic data enters training. Describe the procedure you would use to settle the disagreement, and explain why the procedure prevents the same error in other reports.
2. In CWM's Table 4, tracing data raises CruxEval-O by 29.3 points but does not change SWE-bench Verified, while ForagerAgent data changes both agentic NLL and SWE-bench Verified. Explain what this pattern suggests about which data teaches which skill, and what additional experiment would test the explanation.
3. Kimi-Dev's mid-trained prior is close to the SFT and RL priors after 200 SFT trajectories but supports only 260 usable RL problems after one SFT step. Explain why a prior can be adequate for imitation and inadequate for RL.
4. MidTool's variant with processed sources plus native trajectories (no context-grounded branch) improves BFCL by 7.9 points and lowers τ²-Bench by 4.3 points. Give a causal explanation based on what native trajectories and document-grounded data contain, and state how you would test it.
5. In SPT's Instella-3B + xLAM-FC block at equal tokens, trajectory mid-training lowers MMLU by 4.12 points against the general-data control, and skill documents lower it by 2.60 points. Explain two mechanisms that could make trajectory data lower MMLU more than skill documents, and one measurement that would separate them.
6. AgentFounder's gain per billion tokens falls by a factor of 18 after 15B tokens. Explain how this changes the decision between a larger agentic CPT budget and a larger general-data share, and what evidence is missing to make that decision for a general-purpose model.
7. CWM trains on failed ForagerAgent trajectories as ordinary targets, while Kimi-Dev keeps only exactly correct localizations. Explain the expected effect of each choice on a policy and on a world model, and why neither report settles which is better.
8. State2State gains more when placed after SFT than before SFT. Explain why imitation learning could remove priors learned by RL, and how you would measure that removal directly.

## Connections

- Dependency: ch-32b — Context-Length Extension: Methods, Data Mixtures, and Short-Context Regression (long agent trajectories need an extended window; short-context regression checks).
- Previous chapter: ch-32c — Claimed versus Effective Context Length and Long-Context Evaluation.
- Next chapter: ch-32e — Mid-Training, Annealing, and Context-Extension Recipes Side by Side.
- Related: ch-32 — Mid-Training: Annealing Data, Stage Gates, and Effects on Later SFT and RL; ch-32a — Continual Pretraining Without Forgetting: Replay, Learning-Rate Re-Warming, and Synthetic Continued Pretraining (rehearsal share); ch-27 — Agentic Trajectory Data (trajectory formats; the Kimi K2 placement corrected here); ch-29c — Agentic Environment and Task Synthesis at Scale; ch-30b — Multi-Skill SFT Mixtures: Interference, Transfer, and Agentic and Long-Context Shares (the SFT portion of agentic data placement); ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages; ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability; ch-45d — Open Agentic Recipes Side by Side: Stage Placement, Data Mixture, and Agentic RL (cross-lab stage map); ch-46a — Lab: Small Agentic SFT-then-RL Run with a Generality Gate; ch-51a — Evaluating Agent Generality and Reliability.

## Sources

- [[glm-4-5]] — GLM-4.5 mid-training stages (repo code, synthetic reasoning, 128K agent trajectories) and base-model table.
- [[glm-4-5-recipe]] — verified GLM-4.5 mid-training token, context, RoPE, and LR rows.
- [[glm-5]] — GLM-5 32K/128K/200K stages, issue–PR data, mid-training LR, base-model comparison (chapter excerpt from arXiv:2602.15763v2).
- [[agentfounder]] — Agentic CPT rationale, FAS and HAS synthesis, two-stage 32K/128K training, Tables 3-6, scaling, FAS filter statistics (chapter excerpt from arXiv:2509.13310v1).
- [[cwm-code-world-model]] — execution traces, ForagerAgent, 30/40/30 mixture, epoch targets, Table 4 ablation, Table 13 general scores, long-context settings (chapter excerpt from arXiv:2510.02387v1).
- [[kimi-dev]] — Agentless mid-training recipe, upsampling, loss masks, prior comparison (chapter excerpt from arXiv:2509.23045v3).
- [[midtool]] — tool-use mid-training corpus, hyperparameters, Tables 3-6 results and narrowing signals (chapter excerpt from arXiv:2608.20314v1).
- [[state2state]] — environment-derived state-reaching RL as a mid-stage; Tables 1-5 (chapter excerpt from arXiv:2608.04934v1).
- [[skill-pretraining]] — the one controlled general-average measurement, trajectory vs skill vs general mid-training, mixture equation and sweep.
- [[kimi-k2]] — pre-training data, annealing, and the post-training location of agentic data synthesis.
- [[kimi-k2-recipe]] — verified Kimi K2 annealing and long-context activation row.
- [[kimi-k2-agentic-data]] — cited only as the unverified source of the incorrect "agentic pre-training" claim; not used as evidence.
- [[kimi-k2-5]] — K2.5 training-stage table, code and GUI action-trajectory data in pre-training (chapter excerpt from arXiv:2602.02276v2).
- [[agent-early-experience]] — related reward-free agent training without general-capability evaluation.
