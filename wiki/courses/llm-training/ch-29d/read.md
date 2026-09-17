<!-- chapter: ch-29d
     track: synthetic
     kind: content
     title: User Simulators, Trajectory Verification, and Failed Trajectories
     deps: [ch-29c, ch-22]
     sources: [[tau-bench]], [[tau2-bench]], [[apigen-mt]], [[kimi-k2]], [[userrl]], [[simulator-collapse]], [[simulator-collapse-recipe]], [[lost-in-multi-turn]], [[explorer]], [[agent-early-experience]], [[agent-early-experience-recipe]], [[os-genesis]], [[learning-from-failure-nat]], [[eto-trial-and-error]], [[ipr-step-level-refinement]], [[glm-5]], [[deepswe]], [[deepswe-recipe]], [[openthoughts-agent-glm46-teacher]], [[together-coderforge-agent-trajectories]], [[likelihood-displacement]]
     figures: figures/failed-rollout-signal.html
     revised: 2026-09 (generality revision)
-->

# Chapter 29d — User Simulators, Trajectory Verification, and Failed Trajectories

> **Core insight.** A user simulator is part of the training data distribution, and its errors are measurable: in τ²-bench, manual annotation found user-simulator errors in 47% of airline, 40% of retail, and 16% of telecom conversations, with 13%, 12%, and 6% of conversations containing errors that prevent task completion ([[tau2-bench]] Table 2). When an agent is trained with RL against one frozen LLM simulator, held-out success peaks early and returns toward the untrained level while policy entropy falls; Verbalized Sampling, a three-family simulator ensemble, and co-training raised τ²-bench Retail held-out success for Qwen3-4B-Instruct from 46.1% (best single-simulator checkpoint) to 55.5-62.2% ([[simulator-collapse]] Table 1). Verification decides which trajectories become targets, and verifiers make measurable mistakes: 14% of 100 Explorer trajectories were human-labeled failures that the GPT-4o verifier accepted ([[explorer]] Table 10). Failed trajectories are most often discarded, but four alternatives have published evidence: masking erroneous segments ([[glm-5]] §3.1), training failures under an "incorrectly" prefix ([[learning-from-failure-nat]] Table 2), trajectory-level DPO against expert successes ([[eto-trial-and-error]] Table 2), and step-level contrastive pairs from Monte Carlo rollouts ([[ipr-step-level-refinement]] Table 2).
>
> **Guideline.** When a user simulator generates training trajectories or RL rewards, annotate a sample of its conversations for critical and benign errors, evaluate on simulator families and real users not used in training, and track policy entropy and the share of zero-variance groups, because single-simulator training reward rose while held-out reward declined in all three single-simulator runs of [[simulator-collapse]] (Fig. 3). When failures are labeled by a reliable outcome check and expert or successful trajectories exist for the same task, pair them (ETO, IPR) or condition on them (NAT) instead of discarding them, and keep a positive likelihood term, because removing the SFT term from IPR lowered InterCodeSQL reward from 61.3 to 31.7 ([[ipr-step-level-refinement]] Table 4). When a rollout ends by environment crash, remove it from the group instead of scoring it as a failure, because GLM-5 states that such failures reflect environment instability rather than model capability ([[glm-5]] §4.1.2). When a rollout ends by timeout, context limit, or step limit, record the finish reason and mask it as DeepSWE does, because rewarding or penalizing such runs preceded reward collapse in DeepSWE's Qwen3-14B ablation, shown only as a curve ([[deepswe]] §2.3, Figure 6); if loops or redundant exploration cause most of these endings, masking removes the pressure against that behavior (Open question, §6). When failures are labeled only by an LLM judge, keep them out of any negative-gradient term, because 0.05 / 0.47 = 10.6% of the trajectories that Explorer's verifier rejected were human-labeled successes (derived from [[explorer]] Table 10) and no source in this chapter tests negative-gradient training on judge-labeled failures.

## Why this chapter matters for a general-purpose model

Agentic post-training data consists of trajectories: sequences of model messages, tool calls, tool observations, and, in conversational tasks, user messages. The pipeline for such data has three parts that this chapter separates. A **user simulator** is a language model that plays the user from a hidden task instruction. A **trajectory generator** is the policy or teacher that acts in the environment. A **verifier** is the rule, test, or judge that labels a finished trajectory as success or failure. ch-29c built the environments and tasks; this chapter covers the three components that turn an environment into SFT data and RL signal.

Each component can narrow the model. A simulator with one dominant behavior teaches the policy to handle that behavior only. A verifier that accepts failures turns errors into positive targets. A pipeline that discards every failure removes the information about which actions fail, and a pipeline that penalizes every timeout pushes down trajectories that may have been correct. The measurable questions are: (1) how often each component is wrong, (2) how its errors change held-out performance, and (3) which uses of failed trajectories improve held-out performance and under which labels. The stage in the course pipeline is SFT data construction and the SFT-to-RL hand-off (pre-training → mid-training → **SFT** → preference optimization → **RL** → evaluation). The quality and diversity selection methods of ch-22 apply to trajectories as well; this chapter adds the agent-specific verification and failure handling.

Negative in this chapter always carries one of the four meanings defined in the Negative samples section: negative marginal value (discarded), negative as content, negative as conditioning, or negative as gradient.

## §1 User simulators as data components

**Definition.** A user simulator is an LLM that receives a task instruction hidden from the agent and produces the user side of a multi-turn interaction. The instruction usually fixes the user's identity, goal, and preferences so that a single outcome is correct ([[tau-bench]] §3).

**Problem it addresses.** A task that requires the agent to collect information across turns cannot be run at scale with human users. τ-bench estimates $0.23 of simulator cost and $0.38 of agent cost per τ-retail task with gpt-4 as the simulator and gpt-4o as the agent ([[tau-bench]] §5.1). UserRL estimates about 4M simulator requests for one RL run (2k tasks × 15 epochs × 8 rollouts × 16 turns) ([[userrl]] App. C). The measurable problem is whether conversations with a simulator lead to the same agent behavior and the same success estimates as conversations with users.

**Five designs in the sources.**

1. **Hidden-instruction LLM user (τ-bench, 2024-06).** gpt-4-0613 plays the user from a system-prompt instruction and sees the conversation but not the agent's tool calls; the episode ends when it emits `###STOP###` ([[tau-bench]] §3). Reward is 1 only if the final database equals the annotated goal and the agent's messages contain the required outputs.
2. **Tool-constrained user under dual control (τ²-bench, 2025-06).** In the telecom domain the user also has tools on a shared state (for example toggling airplane mode). User tools return human-readable outputs, and the prompt restricts the user to reactive tool calls: "Only call a tool if the agent has requested it" and "Disclose information progressively" ([[tau2-bench]] §1, App. C.2).
3. **Persona-guided user with Best-of-N (APIGen-MT, 2025-04).** A human LM reveals a validated task blueprint over turns; to reduce randomness the simulated user samples N = 4 candidate replies and self-critiques before replying. On τ-bench retail this raised gpt-4o's success from 62.8 to 67.0 and lowered its variance over 5 trials from 11.1 to 2.6 ([[apigen-mt]] §4.2, Table 3).
4. **LLM-generated personas with a tool simulator (Kimi K2, 2025-07).** "LLM-generated user personas with distinct communication styles and preferences engage in multi-turn dialogues", while a stateful tool simulator "introduces controlled stochasticity to produce varied outcomes including successes, partial failures, and edge cases" ([[kimi-k2]] §3.1.1). Real sandboxes replace simulation for coding and software engineering.
5. **Gym users with rule-based completion (UserRL, 2025-09).** Eight gyms keep task completion rule-based and let an LLM produce user replies; training uses Qwen3-32B as the simulator and evaluation uses GPT-4o ([[userrl]] §3, App. C).

**Evidence that simulator choice changes results.** In τ²-bench telecom, agents score higher on tasks with an Easy persona than with a Hard persona ([[tau2-bench]] Figure 7). In UserRL, training against GPT-4o instead of Qwen3-32B "generally yields higher performance", and the authors attribute part of this to using the same simulator in training and evaluation (Interpretation) ([[userrl]] §5.3). In UserRL's small human study, five PhD students replaced GPT-4o on TurtleGym and TelepathyGym, and Qwen3-8B scores rose from 0.1854 to 0.3127 and from 0.5610 to 0.7805, because humans "sometimes offered subtle hints" while GPT-4o replied "Yes", "No", or "Maybe" ([[userrl]] Table 5, App. C). The direction of the simulator-human gap therefore depends on the task: this study found the simulator less informative than humans, while [[lost-in-multi-turn]] §9 argues that its simulator is more orderly than humans.

**Callback to the ch-25 customer-simulator question.** ch-25 qa.md Q12 recorded that an LLM playing a telemarketing customer refused almost every time (no count was recorded). The sources in §2 give that observation a measurable form: a simulator that emits one reply with high probability is "mode-collapsed" in the sense of [[simulator-collapse]] Def. 3.1, and a co-trained simulator rewarded adversarially reached 98% refusal and an evaluation reward of 0.07 ([[simulator-collapse]] App. F.8). Relating the ch-25 observation to this definition is the course's interpretation; neither paper studies sales calls.

**Implication for a general-purpose model.** The simulator defines which user behaviors appear in training. A model trained against one persona style is expected to learn to serve that style (Interpretation, supported for RL against one frozen simulator by [[simulator-collapse]] Fig. 3). When a simulator generates training data, treat the simulator population as a mixture to be designed and measured, like the task mixture.

## §2 Simulator failure modes and how to validate a simulator

**Four failure modes with evidence.**

| Failure mode | Definition | Evidence |
|---|---|---|
| Instruction deviation | The simulated user omits or contradicts a constraint in its instruction | τ²-bench retail: 4 of 20 annotated errors are missing constraints, 3 are premature termination; most critical errors come from these two types ([[tau2-bench]] App. E.1) |
| Over-compliance and limited reasoning | The user accepts agent proposals without the checks a careful user would make | τ-bench: "the user authorizes the agent-recommended lamp without double checking its features" ([[tau-bench]] §6) |
| Orderly underspecification | The simulator reveals information in a fixed, cooperative order that real users do not follow | Lost in Multi-Turn reveals exactly one shard per turn and guarantees the final turn completes the task; the authors expect real-world degradation to be larger (Interpretation) ([[lost-in-multi-turn]] §9) |
| Simulator collapse | The simulator's reply distribution concentrates on one mode, and the policy learns to exploit that mode | Held-out reward peaks early and declines in single-simulator RL; entropy falls toward zero ([[simulator-collapse]] §3.3, Fig. 3) |

**Measured error rates.** τ²-bench annotated conversations in which gpt-4.1 played both user and agent. Two annotators reviewed each conversation against four criteria: adherence to simulator guidelines, adherence to the task instruction, correct user-tool use, and natural continuation ([[tau2-bench]] §4.3).

| Domain | Conversations | Critical | Benign | Total |
|---|---|---|---|---|
| airline | 100 | 13% | 34% | 47% |
| retail | 50 | 12% | 28% | 40% |
| telecom | 50 | 6% | 10% | 16% |

The Table 2 caption says telecom had no critical errors while the table reports 3 of 50; the table value is used here. The authors attribute the lower telecom error rate to constraining the user through tools and observable state instead of prose instructions (Interpretation; not tested on retail or airline, §5). Lost in Multi-Turn inspected its GPT-4o-mini simulator and reports 97.8% of inspected conversations as valid, with shards fully revealed in 96.0% of user turns ([[lost-in-multi-turn]] App. D, Table 5); the number of inspected conversations is printed as "several hundred", 200, and 100 in different places.

**Worked example: what a critical-error rate does to a success estimate (derived).** Suppose an agent would succeed on 60% of retail tasks with an error-free user, and critical simulator errors occur in 12% of conversations independently of task difficulty. A critical error precludes completion by definition, so the measured success is at most 0.60 × (1 − 0.12) = 0.528. The same errors in SFT data generation remove 12% of otherwise successful trajectories, and in RL they assign reward 0 to rollouts whose agent actions may have been correct. The independence assumption is the course's; τ²-bench does not report how errors correlate with difficulty.

**Simulator collapse: mechanism.** [[simulator-collapse]] (arXiv 2026-08) defines, for a simulator φ and its most likely reply a⋆ at state s after agent message a^π:

```
ε_φ(s, a^π) = 1 − φ(a⋆ | s, a^π)
‖∇_θ J_φ(θ) − ∇_θ J_mode(θ)‖ ≤ 2 · B · R_max · ε̄_H(θ)
```

- ε_φ is the probability mass the simulator places outside its mode; a simulator is ε⋆-collapsed if the expected ε_φ over visited turns is at most ε⋆ (Def. 3.1).
- ε̄_H(θ) is the expected sum of ε_φ over the H turns of a rollout; J_φ is the RL objective with the real simulator and J_mode the objective when every simulator turn emits a⋆.
- B bounds the norm of the trajectory score Σ_t ∇_θ log π_θ; rewards lie in [0, R_max] (Thm. 3.2).

When ε̄_H is small, the policy gradient is close to the gradient for a user who always gives the modal reply, so training optimizes against one user. The authors call the bound "an analytic guide to the bias direction rather than a tight bound" (§3.2).

**Worked example: why collapse also removes RL signal (derived).** With group-normalized advantages, a group of G rollouts gives zero advantage to every member when all rewards are equal. If each rollout succeeds independently with probability p, the chance that a group of G = 8 is all-equal is p⁸ + (1 − p)⁸. At p = 0.5 this is 2/256 = 0.8%; at p = 0.1 or p = 0.9 it is 43.0%; at p = 0.95 it is 66.3%. A simulator that nearly always refuses pushes p toward 0 on most tasks. The measured diagnostics match this direction: under single-simulator RL on τ²-bench Retail, zero-variance batches rose from 60% to over 85%, all-failure batches rose to 70%, and entropy fell from 1.9 to 0.4 nats ([[simulator-collapse]] App. F.3, Fig. 19).

**Evidence for fixes.** Qwen3-4B-Instruct on τ²-bench (Retail / Airline held-out success): Base 40.4 / 24.0; RL against one frozen GPT-5-mini simulator 46.1 / 29.8 (best checkpoint, a transient peak); Persona-Guided 49.2 / 31.6; ensemble of three simulator families 57.1 / 40.1; Verbalized Sampling 55.5 / 36.9; Co-Training 60.5 / 44.4; Population Co-Training 62.2 / 45.7 ([[simulator-collapse]] Table 1). In the pre-registered human study (N = 40 per cell), task outcome was 0.41 for Base, 0.43 for single-simulator RL, 0.63 for Verbalized Sampling, and 0.70 for Co-Training ([[simulator-collapse]] Table 3). **Result (single study)**; text-only, English, two-agent tasks.

**Conditions and limits.** Table 1 reports best checkpoints selected on a panel that includes the three training simulator families, which the authors say "leaks partial training-simulator signal into selection" (§4.1). A co-trained simulator needs a reward that keeps policy reward variance near a target: an adversarial simulator reward gave 98% refusal and evaluation reward 0.07, a cooperative one cut pushback to 2% and gave 0.17, and the variance-targeting reward gave 0.40 ([[simulator-collapse]] App. F.8).

**Validation protocol drawn from these sources.**

1. Annotate at least a sample of simulated conversations for critical and benign simulator errors, by domain ([[tau2-bench]] §4.3).
2. Run a CONCAT-style control: give the full instruction in one turn and check that the multi-turn loss is not caused by information lost in rewriting ([[lost-in-multi-turn]] §3.3; CONCAT averaged 95.1% of FULL, §6.1).
3. Evaluate trained agents against simulator families not used in training and against real users ([[simulator-collapse]] §4.1, Table 3; [[userrl]] Table 5).
4. Report pass^k and unreliability, not only mean success (§4 below).

**Implication for a general-purpose model.** A simulator validated only by its fluency can still teach one interaction pattern. The measured quantities are simulator error rate, held-out-simulator gap, human-user gap, and policy entropy.

## §3 Trajectory generation strategies

**Definition.** A trajectory generation strategy specifies which model acts, in which environment, from which task, and whether the task text is written before or after acting.

| Strategy | Mechanism | Evidence (setting, metric, number) |
|---|---|---|
| Strong-teacher rollouts in real environments | A teacher model acts in executable environments; successful runs are kept | CoderForge: Qwen3-Coder-480B, 258,134 trajectories over 51K tasks, 155,144 pass all tests; Qwen3-32B SFT reaches 59.4% pass@1 on SWE-Bench Verified ([[together-coderforge-agent-trajectories]]) |
| Blueprint first, conversation second | Validate ground-truth actions by execution, policy unit tests, and an LLM committee, then simulate the dialogue and keep state-matching runs | APIGen-MT: task validation success 70% with agentic feedback vs 28% without; xLAM-2-70b-fc-r 56.2% τ-bench pass@1 vs 38.2% for Llama 3.1 70B Instruct ([[apigen-mt]] Fig. 4, Table 2) |
| Explore then refine the task | The proposer acts on a live site; the refiner rewrites the task after each action; the summarizer writes the final task from executed actions | Explorer: 94K of 175K trajectories kept at $0.28 per success; Qwen2-VL-7B Mind2Web-Live full-task success 14.5 → 19.3 on 83 tasks ([[explorer]] Table 3, Table 5, App. B) |
| Reverse task synthesis | Explore GUI elements by rule, then write low-level and high-level instructions from observed ⟨pre-state, action, post-state⟩ triplets | OS-Genesis: 1K trajectories give Qwen2-VL-7B 17.41% AndroidWorld success vs 6.25% for task-driven data ([[os-genesis]] Table 1) |
| Reward-free early experience | At expert states, execute the agent's own alternative actions and train on the resulting next states or on rationales grounded in them | WebShop, Llama-3.2-3B: imitation 41.8, implicit world modeling 60.2, self-reflection 52.7 ([[agent-early-experience]] Table 1) |
| Monte Carlo continuation from intermediate steps | Branch from step t of an expert trajectory and estimate step value by N rollouts | IPR: N = 5 continuations per step; step-reward order agrees with a WebShop page-scoring rule in up to 82% of pairs ([[ipr-step-level-refinement]] §4.1, §5.3) |

**Search with backtracking.** None of the verified sources in this chapter trains on trajectories produced by tree search with explicit backtracking; IPR's branching from expert prefixes (§5) is the closest mechanism covered here. **Open question** for this chapter's evidence base.

**Conditions and limits.** Each strategy has a stated narrowing risk. CoderForge generates all data with one scaffold and tool set and warns that models "might perform worse when used with different scaffolds, tools, and prompt templates" ([[together-coderforge-agent-trajectories]] Limitations). APIGen-MT evaluates on the same two τ-bench domains, APIs, and policies used for generation and reports no overlap check ([[apigen-mt]] §4.3, §5.1). Explorer's synthetic-only model scores 43.0 average step success on Multimodal-Mind2Web, below 49.5 for training on Mind2Web human data and 54.3 for both ([[explorer]] Table 6). OS-Genesis gains saturate with trajectory count, which the authors attribute to model capacity and GPT-4o's task completion ability ([[os-genesis]] §5.3, Fig. 6).

**Long trajectories.** CoderForge trajectories have median lengths of 36K-42K tokens and a P99 near 100K; the 32B model is trained at 128K context with sequence parallelism and packing with boundary-aware masking ([[together-coderforge-agent-trajectories]]). APIGen-MT trajectories reach 29 turns ([[apigen-mt]] Fig. 4). Context management for such trajectories at inference and in RL is covered in ch-45c.

**Implication for a general-purpose model.** Strategies differ in where task diversity comes from: fixed task pools (CoderForge, APIGen-MT), the environment's own states (Explorer, OS-Genesis), or the agent's own alternatives (early experience). Mixing sources with different diversity mechanisms is expected to reduce dependence on one generator's task distribution (Interpretation); no source in this chapter measures that mixture directly.

## §4 Trajectory verification and filtering

**Definition.** Trajectory verification assigns a label (success, failure, or a graded score) to a finished trajectory or to individual steps. Filtering uses the label to decide what enters training.

**Four verification layers.**

1. **Outcome checks.** Compare final environment state or test results with a reference. τ-bench compares the final database with the unique goal database and checks required output substrings ([[tau-bench]] §3). CoderForge keeps trajectories whose final patches pass all repository tests ([[together-coderforge-agent-trajectories]]). Limit: τ-bench states that r = 1 "might be a necessary but not sufficient condition", for example when the agent acts without the confirmation the policy requires ([[tau-bench]] §3).
2. **Step validity and policy checks.** APIGen-MT runs format checks, executes candidate actions and records the state diff, and translates domain policies into Python unit tests before any dialogue is simulated ([[apigen-mt]] §4.1.2). GLM-5 filters search questions in three stages, including a verification agent that rejects "non-unique answers, inconsistent evidence, or incorrect labels" ([[glm-5]] §4.2.3).
3. **Tool-call protocol checks.** Count calls with invalid identifiers, malformed arguments, or repeated failing edits. On τ-retail, gpt-4o function calling makes 0.46 calls with non-existent IDs per task, gpt-3.5-turbo function calling 2.08, and gpt-3.5-turbo Act 6.34 ([[tau-bench]] §5.2). CoderForge reports "a lot of instances" of `str_replace_editor` failures in early student training, caused by the model repeating the string it wanted to replace, and tested a repetition penalty in response ([[together-coderforge-agent-trajectories]]).
4. **Judge-based filtering.** An LLM scores the trajectory against the task or a rubric. Kimi K2 keeps only trajectories that an LLM judge rates as meeting the rubric; the rejection rate is not reported ([[kimi-k2]] §3.1.1). Explorer's GPT-4o verifier labels success from the task, actions, screenshots, and the final page ([[explorer]] §3.2).

**Worked example: precision of a judge filter (Explorer, derived).** On 100 random trajectories, the share matrix is: human success and judge success 0.39, human success and judge failure 0.05, human failure and judge success 0.14, human failure and judge failure 0.42 ([[explorer]] Table 10). The judge accepts 0.39 + 0.14 = 0.53 of trajectories. Precision is 0.39 / 0.53 = 73.6%, so 26.4% of accepted trajectories are human-labeled failures that become positive SFT targets. Recall is 0.39 / (0.39 + 0.05) = 88.6%. Raw agreement is 0.39 + 0.42 = 81%, the number the paper reports; agreement alone hides that the errors are concentrated on the accept side. On the reject side, the judge rejects 0.47 of trajectories, and 0.05 / 0.47 = 10.6% of them are human-labeled successes; these are the false negatives that would receive a negative gradient if rejected trajectories were used as DPO losers.

**Judge bias.** Three biases appear in the sources. (a) **Same-judge evaluation**: Explorer's in-domain test uses the same verifier that filtered the training data ([[explorer]] §5.1), so errors shared by filter and evaluator are not detected. (b) **Curation bias**: τ-bench tuned user instructions by running a gpt-4-turbo agent, which the authors name as "implicit bias" ([[tau-bench]] §6). (c) **Graded judges instead of binary ones**: OS-Genesis samples trajectories with probability P(g_i) = R_i / Σ_k R_k from a 1-5 GPT-4o score; with scores 5, 3, 1 the probabilities are 5/9, 3/9, 1/9, so a trajectory scored 1 is still drawn 11% of the time ([[os-genesis]] Algorithm 1). The GPT-4o score has Spearman 0.813 (mobile) and 0.798 (web) with human annotators on 100 trajectories ([[os-genesis]] Table 6). Judge calibration in general is the subject of ch-49.

**Consistency as a verification target: pass^k.** For a task with n trials and c successes ([[tau-bench]] §3):

```
pass^k = E_task[ C(c, k) / C(n, k) ]          (all k trials succeed)
pass@k = 1 − E_task[ C(n − c, k) / C(n, k) ]  (at least one of k succeeds)
```

- C(a, b) is the binomial coefficient; the expectation is over tasks; trials are i.i.d. reruns with the same instruction and database.

Worked example: n = 8, c = 6. pass^2 = C(6,2)/C(8,2) = 15/28 = 0.536; pass@2 = 1 − C(2,2)/C(8,2) = 27/28 = 0.964; pass^4 = 15/70 = 0.214. For gpt-4o function calling on τ-retail, pass^8 is below 25% ([[tau-bench]] §5.1). Lost in Multi-Turn measures the same property per instruction as unreliability U = 90th − 10th percentile of scores over 10 runs, which rose on average by 112% from single-turn to sharded multi-turn conversations while aptitude fell 16% ([[lost-in-multi-turn]] §6.2). A filter that keeps one success per task cannot distinguish a task solved 1 time in 8 from a task solved 8 times in 8; recording c/n per task preserves that information.

**Implication for a general-purpose model.** The verifier determines what the model learns is a success. Outcome checks have no measured false-accept rate in these sources but can pass policy-violating runs ([[tau-bench]] §3); judges extend verification to open tasks with a measured false-accept share (26.4% of accepted trajectories in [[explorer]]). For RL use, a false accept becomes a positive advantage, so verifier precision matters more than raw agreement.

## §5 Failed trajectories: five uses

A failed trajectory is one that the verifier labels as not completing the task. The sources use failures in five ways, listed from no use to the most direct use as gradient. The interactive figure [figures/failed-rollout-signal.html](figures/failed-rollout-signal.html) lets the reader set counts of successes, failures, timeouts, and environment crashes in a group and see which rollouts receive gradient and with what advantage under each handling rule; a second panel computes pass^k and pass@k for one task from n and c.

### 5.1 Discard (negative marginal value)
APIGen-MT keeps only trajectories with reward 1 and lists "additional contrastive signal" from failures as future work ([[apigen-mt]] §4.2, §6). CoderForge trains only on the 155,144 passing trajectories and releases the 103K failures; failed trajectories use 18-28% more steps than successful ones, and the authors discard them to "push the model toward efficient task resolution" (Interpretation) ([[together-coderforge-agent-trajectories]]). Kimi K2 discards trajectories that fail the rubric judge ([[kimi-k2]] §3.1.1). OS-Genesis instead keeps incomplete trajectories as positive targets with lower sampling probability (§4).

### 5.2 Loss-mask erroneous segments (negative as content)
**Definition.** The trajectory keeps its erroneous steps in the context, but those tokens receive zero loss; the later correction is trained normally. GLM-5: "Erroneous segments within trajectories are retained but masked out in the loss function, allowing the model to learn error correction behaviors without reinforcing incorrect actions" ([[glm-5]] §3.1).

```
L(θ) = − Σ_t m_t · log π_θ(y_t | y_<t, x),   m_t ∈ {0, 1}
```

- y_t is token t of the trajectory; x is the prompt; m_t = 0 for environment observations, user turns, and tokens inside an erroneous segment, and m_t = 1 for the remaining assistant tokens.

Worked example: a trajectory has 10 assistant turns of 400 tokens each. Turn 4 issues a wrong tool call, turn 5 reads the error message, and turn 6 issues the corrected call. Masking turn 4 removes 400 of 4,000 assistant tokens (10%) from the loss; turn 6 is trained with turn 4 and its error observation in context, so the model learns the correction conditional on having seen the error. Discarding the trajectory would remove all 4,000 tokens. GLM-5 applies the same idea to defective slide pages inside otherwise good trajectories, stating that discarding would "reduce effective data utilization and increase generation cost" ([[glm-5]] §4.2.5).

**Evidence and limits.** GLM-5 reports no ablation of masking versus discarding, no share of masked tokens, and no description of how erroneous segments are detected ([[glm-5]]). **Result (single study), not isolated.**

### 5.3 Failure-conditioned training (negative as conditioning)
**Definition.** Failed trajectories are trained with ordinary cross-entropy under an instruction that marks them as incorrect; at inference only the "correct" instruction is used. NAT appends "Please generate a solution that **incorrectly** answers the question." to negatives and the "correctly" variant to positives ([[learning-from-failure-nat]] §3.3).

**Evidence.** LLaMA-2-7B-Chat, GPT-3.5 ReAct trajectories, math tasks with a calculator, 2k positives and 10k negatives: average accuracy 55.90 (positives only), 63.39 (negatives added without a marker, NUT), 64.64 (NAT) ([[learning-from-failure-nat]] Table 2). With 5k positives: 64.17, 64.95, 67.42. At 13B with 5k positives, unmarked negatives lowered accuracy (70.76 → 69.10) while NAT gave 71.28. Replacing the marker text with "A"/"B" or random strings gave 63.15-64.04, within one point of the Correct/Incorrect prefix (63.55), so the authors conclude that separating the two groups matters, not the words (Table 7). Negatives from a weaker fine-tuned LLaMA-2-7B lowered accuracy by 3.16 and 6.20 points instead of raising it (Table 5). **Result (single study)**; tasks with ground-truth answers only.

**Mechanism (Interpretation).** Unmarked negatives raise the likelihood of wrong actions: NUT's GSM8K action error was 10.47% against 3.58% for positives only, and NAT's 7.90% (Table 6). The marker moves most of that likelihood into a condition that is not used at inference.

### 5.4 Success-versus-failure preference pairs (negative as gradient)
**Definition.** ETO (arXiv 2024-03) pairs the SFT agent's lower-reward trajectory with the expert trajectory for the same task and trains with DPO, then repeats exploration with the updated agent ([[eto-trial-and-error]] §3):

```
L_DPO = − E log σ( β log πθ(e_w | u)/πref(e_w | u) − β log πθ(e_l | u)/πref(e_l | u) )
```

- u is the task instruction; e_w and e_l are the higher- and lower-reward trajectories; πref is the policy at the start of the iteration; β is the constraint weight; σ is the sigmoid.

**Evidence.** Llama-2-7B-Chat, average reward (WebShop / ScienceWorld seen / unseen / ALFWorld seen / unseen): SFT 63.1 / 67.4 / 53.0 / 60.0 / 67.2; RFT (successes added to SFT) 63.6 / 71.6 / 54.3 / 62.9 / 66.4; ETO 67.4 / 73.8 / 65.0 / 68.6 / 72.4 ([[eto-trial-and-error]] Table 2). The ETO-vs-RFT comparison changes both the loss and the chosen samples, so the share of the gain due to the rejected term is not measured.

**Failure modes.** Step-level pairs built from final rewards dropped WebShop reward to 8.3 at LR 1e-6 and β 0.1 (Table 4). ETO without a behavioral-cloning stage scored 12.5, below the untuned model's 17.9 (Table 5). Performance declined after the third iteration on WebShop and ScienceWorld and after the first on ALFWorld (§4.3). In a different study, DPO with expert actions as chosen and the agent's own alternative actions as rejected "collapses within tens of optimization steps" on WebShop and ALFWorld, and its best pre-collapse checkpoints (53.1, 82.8) stayed below implicit world modeling (58.6, 85.9), which uses the same alternatives as content ([[agent-early-experience]] §6.3, Table 3).

### 5.5 Step-level refinement (negative as gradient, localized)
**Definition.** IPR (arXiv 2024-06) localizes the failure to a step. The agent continues from the first t − 1 steps of an expert trajectory; Monte Carlo rollouts by the frozen SFT scorer estimate the value of the expert action and of the agent action at step t ([[ipr-step-level-refinement]] §3.2-3.3):

```
r_s(s_t, a_t) ≈ (1/N) Σ_{i=1..N} r_o(u, e^(i))
L = L_o-DPO + L_s-DPO + L_SFT
```

- r_o(u, e^(i)) is the outcome reward of rollout i from step t; N is the number of rollouts.
- L_o-DPO is DPO on whole trajectories; L_s-DPO is DPO on the two continuations given the shared prefix e_{t−1}; L_SFT is the negative log-likelihood of the winning trajectory.

Worked example: N = 5. Rollouts after the expert action end with rewards 1, 0, 1, 1, 0, so r_s = 0.6. Rollouts after the agent action end with 0, 0, 1, 0, 0, so r_s = 0.2. The gap 0.4 exceeds the thresholds used for WebShop (τ = 0.01) and InterCodeSQL (τ = 0.1) but not ALFWorld (τ = 0.5). If the agent's own continuation also has a lower outcome reward, a step pair is created with the expert continuation as chosen.

**Evidence.** Llama-2-7B average reward (WebShop / InterCodeSQL / ALFWorld seen / unseen): ETO 67.4 / 57.2 / 68.6 / 72.4; IPR 71.3 / 61.3 / 70.3 / 74.7 (best iteration) ([[ipr-step-level-refinement]] Table 2). Ablations on WebShop / InterCodeSQL / ALFWorld unseen: without outcome DPO 70.2 / 59.3 / 72.4; without step DPO 66.4 / 58.0 / 70.2; without the SFT term 61.8 / 31.7 / 64.9 (Table 4). Removing the positive likelihood term causes the largest loss, which supports keeping an anchor next to any rejected-sample gradient. Training time was "less than three times" that of ETO (App. C). **Result (single study)** with one base model family in the main table.

**Implication for a general-purpose model.** The five uses differ in what happens to the likelihood of the failed action: unchanged (discard), unchanged but trained around (mask), moved into an unused condition (NAT), or reduced (ETO, IPR). Only the last two can displace probability mass onto other outputs. The sources that use them add an anchor term (IPR L_SFT), cap iterations (ETO, IPR), and label failures with environment rewards rather than judges.

## §6 Mask versus penalize for truncated, timed-out, and environment-error rollouts

**Definition.** A rollout that ends because of a context limit, step limit, wall-clock timeout, or environment crash has no verified outcome. **Penalize** assigns it the failure reward. **Mask** keeps it out of the loss (and, for group methods, out of the group statistics).

**Problem.** A penalized truncated rollout receives a negative advantage although its actions may have been correct. A masked rollout provides no signal, and if masking is frequent, groups shrink.

**Mechanism with numbers (derived).** Group of 8 rollouts: 2 successes, 4 genuine failures, 2 timeouts. Group z-score advantage A_i = (r_i − mean)/std.

1. Penalize timeouts as 0: mean = 0.25, std = 0.433. Successes get +1.732; all six other rollouts, including the timeouts, get −0.577.
2. Mask timeouts: 6 rollouts remain, mean = 0.333, std = 0.471. Successes get +1.414; failures get −0.707; timeouts get no gradient.
3. Leave-one-out baseline without std normalization (the DeepSWE form, A_i = r_i − mean of the other rewards): penalize gives +0.857 and −0.286; mask gives +0.800 and −0.400.

The figure reproduces these values and lets the counts vary.

**What the sources do.** DeepSWE's compact filtering masks trajectories that reach maximum context, maximum environment steps, or a 20-minute generation timeout. The authors give two reasons: an agent can pass all tests by chance, for example by producing a correct fix in the first 10 steps and editing unrelated files afterwards, and rewarding such trajectories led to collapse in a Qwen3-14B ablation shown only as a curve; and response length per step decreases while environment steps increase ([[deepswe]] §2.3, Figures 6-7). The released training script sets `agent.overlong_filter=True` and `algorithm.mask_truncated_samples=False`, and its timeout differs from the blog ([[deepswe-recipe]]). GLM-5 records the failure reason of each agentic RL sample and excludes samples that fail because of environment collapse; if more than half of a group remains valid it pads the group by repeating valid samples, otherwise it drops the group ([[glm-5]] §4.1.2). Which valid samples are repeated is not reported, and the choice changes the group mean: repeating the two successes in the example above gives mean 0.5 and advantages ±1.0 (derived). In multi-turn RL against simulated users, all-failure groups contribute zero advantage under Eq. 2 of [[simulator-collapse]], so an environment or simulator that fails most rollouts reduces the effective batch without any explicit mask.

**Conditions and limits.** No source in this chapter reports a controlled number for masking versus penalizing timeouts; DeepSWE's evidence is a curve and GLM-5's is a stated design choice. Masking every overlong rollout removes any pressure against overlong behavior. Kimi K2 takes the other option for response length: it truncates responses that exceed a maximum token budget set by task type and applies a penalty ([[kimi-k2]] §3.2.3). **Open question**: which timeout causes are policy-caused (loops, redundant exploration) and should be penalized, and which are environment-caused and should be masked.

**Implication for a general-purpose model.** Failure labels in RL must separate policy failures from infrastructure failures. Recording a finish reason for every rollout (CoderForge's released records include `finish_reason` and `reward` fields ([[together-coderforge-agent-trajectories]])) is the precondition for either choice. The multi-turn RL treatment continues in ch-45b.

## §7 Open agent SFT sets and teacher ablations as distillation evidence

**Definition.** Agent distillation trains a student on trajectories produced by a stronger teacher acting in environments. The teacher choice and the task source are the two main variables.

**Evidence: teacher benchmark strength did not predict student gain.** OpenThoughts-Agent trained Qwen3-8B students on traces from one task source (Freelancer) with different teachers and scored them on a 70-task dev set: GLM-4.6 17.2, GPT5-Nano 8.2, GPT5 8.0, GPT5-Mini 7.6 ([[openthoughts-agent-glm46-teacher]] teacher chart). The authors describe GPT5 as "the best model on TerminalBench itself". **Result (single study)**: one source, one student size, run count not stated.

**Evidence: some task sources lower the student below its starting point.** In the GPT-5-Nano source ablation, the starting Qwen3-8B scored 5.7 on the dev set, and students trained on Mind2Web (2), Synatra (2), and CodeActInstruct scored 5.3, 4.9, and 3.5 ([[openthoughts-agent-glm46-teacher]] source chart). These are trajectories with negative marginal value at the level of a whole source.

**Evidence: scale with test verification.** CoderForge distills Qwen3-Coder-480B into Qwen3-32B with 155,144 test-passing trajectories and reports 59.4% pass@1 on SWE-Bench Verified; the dataset card prints the change as "23.0% → 59.4%", while the blog prose says "23.0% above the base model" ([[together-coderforge-agent-trajectories]]). Tasks sharing a repository and base commit or problem statement with SWE-Bench Verified are excluded. The same data trains Qwen3-4B to 43.0% at epoch 5. APIGen-MT distills gpt-4o and DeepSeek V3 trajectories into xLAM-2 models; xLAM-2-8b-fc-r (46.7) exceeds Llama 3.1 70B Instruct (38.2) on τ-bench ([[apigen-mt]] §5.2, Table 2). UserRL distills a 1k GPT-4o trajectory set as an SFT cold start before RL, without which RL plateaus early ([[userrl]] §5.3).

**Conditions and limits.** OpenThoughts-Agent does not state whether its released SFT traces were filtered by verifier success ([[openthoughts-agent-glm46-teacher]]). CoderForge evaluates mainly on SWE-Bench Verified and names scaffold, bug-fix scope, and missing user interaction as limits. The distillation stage placement across labs is covered in ch-35.

**Implication for a general-purpose model.** When choosing a teacher for agent SFT, compare teachers by the student's score on held-out tasks with the task source fixed, because the teacher's own benchmark rank did not predict student score in [[openthoughts-agent-glm46-teacher]]. When adding a task source, check whether it lowers the student below its starting point, because three sources did so in the same study.

## Negative samples and negative feedback

This section uses four meanings of "negative". (1) **Negative marginal value**: a sample that lowers performance when used as a positive target, and is therefore discarded. (2) **Negative as content**: a failure placed in the input or in a corrected target and trained with ordinary cross-entropy. (3) **Negative as conditioning**: a failure trained under a marker such as "incorrectly" that is not used at inference. (4) **Negative as gradient**: an explicit decrease of the sample's likelihood, such as the rejected term of DPO or a negative advantage. Only (4) removes probability mass from the sample.

**1. Where negatives come from and how they are labeled.** Outcome checks: database state match ([[tau-bench]]), repository tests ([[together-coderforge-agent-trajectories]]), final reward in [0, 1] ([[eto-trial-and-error]]). Step values: Monte Carlo continuations ([[ipr-step-level-refinement]]). Judges: rubric judge ([[kimi-k2]]), trajectory verifier with a measured 26.4% false-accept share among accepted trajectories ([[explorer]] Table 10, derived in §4), graded 1-5 reward model ([[os-genesis]]). Simulated users add label noise: 6-13% of τ²-bench conversations contain a critical simulator error ([[tau2-bench]] Table 2). Infrastructure produces unlabeled endings: timeouts, context and step limits, sandbox crashes ([[deepswe]]; [[glm-5]]).

**2. What current practice does with them.**

| Source | Use | Type (as defined above) |
|---|---|---|
| APIGen-MT, CoderForge, Kimi K2 | Discard failed trajectories | 1. negative marginal value |
| OpenThoughts-Agent | Three task sources lower the student below base | 1. negative marginal value (source level) |
| OS-Genesis | Keep incomplete trajectories as targets with lower sampling probability | not a negative signal; down-weighted positive |
| GLM-5 SFT | Keep erroneous segments in context, mask their loss | 2. negative as content |
| Early experience (IWM, SR) | Train on outcomes of non-expert actions as targets or rationale inputs | 2. negative as content |
| NAT | Train failures under an "incorrectly" marker | 3. negative as conditioning |
| ETO, IPR, early-experience DPO baseline | Rejected side of DPO | 4. negative as gradient |
| GRPO-style agent RL ([[simulator-collapse]], [[deepswe]]) | Negative advantage for below-mean rollouts; masking for truncated or crashed rollouts | 4. negative as gradient; masked rollouts carry none |

**3. Mechanism.** For a softmax over logits z with p = softmax(z), ∂ log p_y / ∂ z_j = 1[j = y] − p_j. A step that decreases log p_c for a rejected token c moves each logit by −η(1[j = c] − p_j): the rejected logit falls by η(1 − p_c) and every other logit rises by η p_j, so the removed mass goes to alternatives in proportion to their current probability. Worked example: p = (A 0.6, B 0.3, C 0.1), reject C with step η = 1. Logit changes are (+0.6, +0.3, −0.9). New probabilities are (0.710, 0.263, 0.026): A gains 0.110 while B loses 0.037. Pushing down an already unlikely sample concentrates mass on the most likely alternative, which may itself be wrong. For multi-token responses, [[likelihood-displacement]] shows that the preferred response can also lose probability when chosen and rejected responses have similar hidden representations (Theorem 6, CHES score). Agent trajectories for the same task often share long prefixes and tool-call formats, so chosen and rejected trajectories are similar in this sense (Interpretation); ETO's step-level pairs scoring 8.3 on WebShop at LR 1e-6 and the early-experience DPO collapse are consistent with this concern, but neither source measures chosen-trajectory likelihood.

**4. Evidence with numbers.** Benefit: ETO over SFT on ScienceWorld unseen 53.0 → 65.0 ([[eto-trial-and-error]] Table 2); IPR over ETO on InterCodeSQL 57.2 → 61.3 ([[ipr-step-level-refinement]] Table 2); NAT over positives-only 55.90 → 64.64 ([[learning-from-failure-nat]] Table 2). Failure modes: step-level DPO pairs 8.3 on WebShop ([[eto-trial-and-error]] Table 4); DPO over non-expert actions collapsed within tens of steps ([[agent-early-experience]] §6.3); weak-model negatives −3.16 and −6.20 points ([[learning-from-failure-nat]] Table 5); iteration beyond 1-4 rounds lowered reward in ETO (after 1 on ALFWorld, after 3 on WebShop and ScienceWorld) and IPR (after 4 on WebShop).

**Size of effect.** No source in this chapter measures what share of the gain comes from the rejected term alone. The IPR ablation shows that removing the positive SFT term costs more than removing either DPO term (Table 4). The claim that negatives are the main driver of these gains is not supported for any setting here.

**5. Controls.** Localize the failure (IPR step pairs with threshold τ). Keep a positive NLL term (IPR L_SFT). Limit iterations (ETO 1-2, IPR up to 4). Take negatives from the same generator as the positives (NAT: GPT-3.5 negatives raised accuracy, while negatives from a fine-tuned LLaMA-2-7B lowered it by 3.16 and 6.20 points); this chapter's sources do not test negatives sampled from the current policy for SFT. Mark negatives instead of training them unmarked (NAT vs NUT). Mask segments and rollouts whose failure is not the policy's (GLM-5, DeepSWE). Exclude judge-labeled failures from gradient terms when verifier precision is unmeasured.

**6. Diagnostics.** Log chosen and rejected log-probabilities separately for DPO on trajectories (not reported by ETO). Log the share of zero-variance and all-failure groups, entropy, and advantage statistics by sign in agent RL ([[simulator-collapse]] App. F.3). Record finish reasons per rollout. Report pass@k and pass^k per task, action error rate ([[learning-from-failure-nat]] Table 6), and invalid tool-call counts ([[tau-bench]] §5.2).

**7. Effect on generality.** Negatives-as-gradient gains in ETO held on unseen ScienceWorld variations and ALFWorld unseen splits ([[eto-trial-and-error]] Table 2), but each ETO agent is trained for one task and multi-task transfer was not tested (Limitations). Discarding long failed trajectories biases data toward shorter solutions, which CoderForge intends. No source here measures calibration, over-refusal, or pass@k at large k after training on failures.

## Recipe

All rows were checked against the linked card or chapter excerpt on 2026-09-14 or 2026-09-15.

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| τ-bench evaluation (gpt-4o agent) | — | eval-gate | user simulator; temperatures; action cap; trials | gpt-4-0613; agent 0.0, user 1.0; 30 agent actions; ≥3 trials per task | arXiv:2406.12045v1 §3, §5 ([[tau-bench]]) | verified 2026-09-15 | no ablation reported |
| τ²-bench evaluation | — | eval-gate | user simulator; runs; temperature | gpt-4.1-2025-04-14; 4 runs per task; 0 | arXiv:2506.07982v1 §4.1 ([[tau2-bench]]) | verified 2026-09-15 | no ablation reported |
| xLAM-2-fc-r series | 1B-70B | SFT | trajectory filter; attempts; user; epochs; masking | keep r = 1; up to 3 attempts per task; BoN user N = 4; at most 3 epochs; loss on assistant tokens | arXiv:2504.03601v4 §4.2, §5.1 ([[apigen-mt]]) | verified 2026-09-14 | Table 3: BoN user lowers variance (gpt-4o 11.1 → 2.6) |
| Qwen3-8B / Qwen3-4B (UserRL Equalized/R2G) | 4B, 8B | distill-SFT | data; LR; epochs; schedule; cutoff | 1k GPT-4o agent-and-user trajectories (top-K per gym); 1.0e-5; 3; cosine, warmup 0.1; 16,384 | arXiv:2509.19736v1 App. B Table 6 ([[userrl]]) | verified 2026-09-15 | Figure 2 left: SFT cold start vs direct RL |
| Qwen3-8B / Qwen3-4B (UserRL) | 4B, 8B | RL | simulator; LR; batch; rollouts; turns; KL; γ; epochs | Qwen3-32B (train), GPT-4o (eval); 1e-6; 128; 8; 16; off; 0.8; 15 | arXiv:2509.19736v1 §5.1, Table 6 | verified 2026-09-15 | Table 3: Equalized/R2G best average; Figure 2 right: GPT-4o training simulator |
| Qwen3-4B-Instruct-2507, Qwen3-8B (τ²-bench) | 4B, 8B | RL | steps; prompts × G; LR; KL coef; clip; temperature | 250; 16 × 8; 1×10⁻⁶ constant; 0.005; 0.2 / 0.28; 0.7 | arXiv:2608.12253v2 App. C.4 Table 6 ([[simulator-collapse-recipe]]) | verified 2026-09-14 | LR tuned on 50 steps of Ensemble training |
| Population Co-Training | 4B, 8B | RL | simulator pool; simulator reward | K = 5 checkpoints, one per 4 steps; exp(−(σ_π² − 0.25)²/0.02) | Table 5; App. F.1, F.8 ([[simulator-collapse-recipe]]) | verified 2026-09-14 | Fig. 17: K=5 > K=10 > K=3 > K=1; App. F.8: eval 0.40 vs 0.07 and 0.17 |
| Llama-3.2-3B / Qwen-2.5-7B / Llama-3.1-8B, Tau-Bench retail | 3B-8B | SFT | epochs and LR per method; batch | IL 6, 1e-5; IWM 1, 5e-6; SR 6, 1e-5; 16 | arXiv:2510.08558v3 App. C.4 ([[agent-early-experience-recipe]]) | verified 2026-09-14 | no ablation reported |
| ETO on Llama-2-7B-Chat | 7B | SFT | batch; LR; warmup; schedule; epochs | 64; 1e-5; 3%; cosine; 3 | arXiv:2403.02502v2 §4.1 ([[eto-trial-and-error]]) | verified 2026-09-14 | no ablation reported |
| ETO on Llama-2-7B-Chat | 7B | preference | DPO LR; batch; epochs; β; iterations | 1e-6; 32; 3; 0.1 (WebShop, ScienceWorld), 0.5 (ALFWorld); 2, 2, 1 | arXiv:2403.02502v2 §4.1 | verified 2026-09-14 | §4.3 Fig. 4: decline after iteration 3 (WebShop, ScienceWorld), after 1 (ALFWorld) |
| IPR on Llama-2-7B | 7B | preference | epochs; batch; MC samples; τ; iteration cap | 3; 48; N = 5 at temperature 1; 0.01 (WebShop), 0.1 (InterCodeSQL), 0.5 (ALFWorld); 4 | arXiv:2406.11176v2 §4.1 ([[ipr-step-level-refinement]]) | verified 2026-09-15 | Table 4: iteration 4 best on WebShop |
| IPR on Llama-2-7B | 7B | preference | LR; β | searched 1e-5 to 5e-5; searched 0.1 to 0.5 (selected values not reported) | arXiv:2406.11176v2 §4.1 | not reported (selected values; checked §4.1, App. A-E) | — |
| NAT on LLaMA-2-Chat | 7B, 13B | SFT | epochs; batch; max LR; schedule; negatives | 2; 64; 5×10⁻⁵; cosine, 3% warm-up; 10k (math main runs) | arXiv:2402.11651v2 §4.1, App. B ([[learning-from-failure-nat]]) | verified 2026-09-15 | Fig. 2: gains plateau near 11k negatives |
| GLM-5 | 744B total / 40B active | SFT | erroneous segments in agent trajectories | retained in context, masked from loss; share not reported | arXiv:2602.15763v2 §3.1 ([[glm-5]]) | verified 2026-09-15 | no ablation reported |
| GLM-5 | 744B / 40B | RL | environment-collapse samples; incomplete groups | excluded; pad by repeating valid samples if > half valid, else drop group | arXiv:2602.15763v2 §4.1.2 | verified 2026-09-15 | no ablation reported |
| DeepSWE-Preview | 32B | RL | masked trajectories (blog) | max context, max environment steps, 20-minute generation timeout | together.ai/blog/deepswe §2.3 ([[deepswe]]) | conflict (blog describes the final run) | Blog Figure 6 (Qwen3-14B): curve only |
| DeepSWE-Preview | 32B | RL | masked trajectories (released script) | agent.overlong_filter=True; algorithm.mask_truncated_samples=False; agent.trajectory_timeout=5400 | rllm@709dec43b740 scripts/agent/swe/deepswe_32b.sh L55, L70, L71 ([[deepswe-recipe]]) | conflict (script; repository does not state it reproduces the final run) | no ablation reported |
| OpenThinker-Agent-v1-SFT | 8B | distill-SFT | teacher; traces; max turns; LR; epochs | GLM-4.6 (AWQ); ~15,200; 32; 4e-05; 7.0 | HF dataset and model cards ([[openthoughts-agent-glm46-teacher]]) | verified 2026-09-14 | teacher chart: GLM-4.6 17.2 vs GPT5 8.0 |
| CoderForge-Preview-32B | 32B | distill-SFT | teacher; sampling; attempts; filter; data; context; LR | Qwen3-Coder-480B; T 0.7, top_p 0.8; 8 / 8 / 4 per task; all tests pass; 155,144 of 258,134; 128K; 1e-5 cosine | Together AI blog 2026-02-25 ([[together-coderforge-agent-trajectories]]) | verified 2026-09-15 | solve rate by attempts (R2E-Gym 62.9% → 80.3% at 8) motivates multiple attempts; no LR ablation |

**Starting point for a small general-purpose run.** For a 4B-8B instruct model trained as a multi-turn user-facing agent, the verified rows give the following configuration. SFT cold start on 1k teacher trajectories at LR 1.0e-5 for 3 epochs with cosine schedule and warmup ratio 0.1 (UserRL, Qwen3-4B/8B, 5 training gyms, 4 H200 GPUs). RL with 16 prompts × 8 rollouts at constant LR 1×10⁻⁶, clip 0.2 / 0.28, and a pool of 5 recent simulator checkpoints or an ensemble of simulator families instead of one frozen simulator ([[simulator-collapse-recipe]]; τ²-bench, P4G, CooperBench on 8×H100). If failed trajectories with outcome labels and matching expert trajectories are available, the verified reference for a preference stage is ETO: one or two DPO iterations at LR 1e-6 and β 0.1 after SFT (Llama-2-7B-Chat, single-task agents, 8×A100); ETO did not follow this stage with RL, so its interaction with RL is untested. Exclude environment-crash rollouts from groups as GLM-5 does, and record finish reasons before choosing whether timeouts are masked.

## Generalization lens

**(a) What increases breadth.**
- Simulator diversity: ensembles, verbalized sampling, and co-trained simulator populations raised held-out-simulator and human-user success over one frozen simulator ([[simulator-collapse]] Tables 1, 3).
- Agent's own alternative actions as content: implicit world modeling raised ALFWorld out-of-domain success for Llama-3.1-8B from 63.3 to 78.1 ([[agent-early-experience]] Table 2).
- Failures paired with successes: ETO raised ScienceWorld unseen reward from 53.0 to 65.0 and ALFWorld unseen from 67.2 to 72.4 while RFT and PPO fell below SFT on ALFWorld unseen ([[eto-trial-and-error]] Table 2).
- Task descriptions derived from executed behavior: OS-Genesis instructions and trajectories have the highest diversity among compared synthetic sets by Sentence-BERT cosine distance ([[os-genesis]] §5.1).

**(b) What causes narrowing or forgetting.**
- One frozen simulator: held-out reward declines after an early peak and entropy falls ([[simulator-collapse]] Fig. 3).
- One scaffold, one task type: CoderForge names scaffold dependence and bug-fix scope ([[together-coderforge-agent-trajectories]] Limitations).
- Same domains for generation and evaluation without overlap checks: APIGen-MT on τ-bench ([[apigen-mt]] §4.3).
- Iterating preference training on a fixed task set: ETO and IPR decline after 1-4 iterations ([[eto-trial-and-error]] §4.3; [[ipr-step-level-refinement]] Table 4).
- Weak or mismatched sources: three task sources lowered the OpenThoughts-Agent student below base; weak-model negatives lowered NAT accuracy ([[openthoughts-agent-glm46-teacher]]; [[learning-from-failure-nat]] Table 5).
- No source in this chapter reports a general-capability benchmark (knowledge, math, instruction following) before and after agent trajectory training; forgetting is not measured here.

**(c) How to measure it for this stage.**
- pass^k and per-instruction unreliability, not only pass@1 ([[tau-bench]] §3; [[lost-in-multi-turn]] §4.2).
- Held-out simulator families and real users, with checkpoint selection on simulators disjoint from training ([[simulator-collapse]] §4.1 states the leakage when this is not done).
- Held-out environments: UserRL reserves IntentionGym, TelepathyGym, and SearchGym ([[userrl]] §5.1); ETO and IPR report unseen splits.
- Simulator error annotation per domain ([[tau2-bench]] Table 2) and verifier confusion matrices ([[explorer]] Table 10).
- Decontamination by repository and commit ([[together-coderforge-agent-trajectories]]) or by repository ([[deepswe]] §2.1).
- Known measurement errors: best-of-three runs on Mind2Web-Live ([[explorer]] App. A.1); same verifier for filtering and in-domain evaluation ([[explorer]] §5.1); tables of best checkpoints hiding end-of-training collapse ([[simulator-collapse]] §4.2); benchmark reward that passes policy-violating runs ([[tau-bench]] §3).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Training and evaluating with the same simulator | Training reward and in-distribution eval rise; human or other-simulator success does not | Evaluate on simulator families not used in training and on a human sample |
| Selecting checkpoints on a panel that includes training simulators | Reported best checkpoint outperforms the end-of-training checkpoint on held-out users | Select on disjoint simulators; report the final checkpoint too |
| Treating judge agreement as precision | High agreement, but accepted trajectories contain failures | Compute precision on accepted trajectories from a labeled sample |
| Keeping one success per task | pass@1 on train tasks looks high; pass^k is low | Store c/n per task; report pass^k |
| Unmarked failed trajectories in SFT | Action or tool-call error rate rises after training | Compare action error rate with and without negatives (NAT Table 6) |
| DPO on trajectory pairs without an anchor or with many iterations | Rejected and chosen log-probs both fall; reward drops after a few iterations | Log both log-probs; add an NLL term; cap iterations |
| Penalizing timeouts and crashes as failures | Negative advantages on rollouts with no verified outcome; reward collapse | Record finish reason; compare masked vs penalized runs |
| Padding groups without recording the rule | Group mean changes with padding order | Log which samples were repeated and the pre-padding statistics |
| Simulator that refuses or complies almost always | Zero-variance and all-failure group share above half | Log group variance share, entropy, and simulator reply distribution per task |
| Assuming the strongest benchmark model is the best teacher | Student score on held-out tasks does not track teacher score | Ablate teachers on a held-out dev set with fixed task source |

## Check your understanding

1. τ²-bench reports a lower user-simulator error rate in telecom than in retail. Explain why constraining the user with tools could lower errors, and what additional experiment is needed before applying the same conclusion to retail.
2. Using the bound in [[simulator-collapse]] Thm. 3.2, explain why a policy trained against a low-entropy simulator can raise training reward while held-out reward falls. Then explain why the same collapse also reduces the number of groups with non-zero advantage.
3. Explorer reports 81% judge agreement. Compute precision on accepted trajectories from Table 10 and explain why this number, not agreement, controls how many failures become positive SFT targets.
4. NAT's Random-string prefixes performed as well as Correct/Incorrect. What does this imply about the mechanism by which marked negatives help, and why would unmarked negatives raise action error?
5. In IPR, removing the SFT term costs more than removing either DPO term. Using the softmax gradient and likelihood displacement, explain why a DPO-only objective on similar trajectories can reduce the probability of correct actions.
6. For a GRPO group with 2 successes, 4 failures, and 2 timeouts, compute the success advantage when timeouts are penalized and when they are masked. Describe a case where masking is the wrong choice.
7. OpenThoughts-Agent found GLM-4.6 traces better than GPT5 traces for an 8B student. Give two hypotheses that could produce this result and a measurement that separates them.
8. UserRL's models scored higher with human users than with GPT-4o users, while Lost in Multi-Turn argues that real users would lower scores. Explain how both can hold, and what this means for validating a simulator for a new domain.

## Connections

- Previous: ch-29c — Agentic Environment and Task Synthesis at Scale (environments and tasks that the simulators, verifiers, and failure handling of this chapter operate on).
- Next: ch-29e — Instruction Tuning and Generalization to Unseen Tasks (held-out task protocols that apply to agent SFT sets).
- Dependency: ch-22 — Quality, Diversity, and Gradient-Based Data Selection (selection methods that trajectory filtering extends).
- Earlier: ch-25 — Multi-Turn Conversation Synthesis (qa.md Q12 customer-simulator observation, §1 callback); ch-27 — Agentic Trajectory Data (trajectory formats and sources).
- Later: ch-31a — Negative Samples in Supervised Training: Corrections, Failure Conditioning, Critiques, and Unlikelihood (full treatment of types 2 and 3); ch-35 — Distillation in Practice A: Where Labs Insert Teacher Data; ch-43a — Negative Samples and Negative Gradients: Likelihood Displacement, Squeezing, and Negative Advantages (derivations for type 4); ch-45b — Multi-Turn Agentic RL: Observation Masking, Credit Assignment, and Stability (continues §6); ch-45c — Context Management for Long-Horizon Agents; ch-49 — Judge Models: Bias, Calibration, and Judge-Specific Overfitting; ch-51a — Evaluating Agent Generality and Reliability.

## Sources

- [[tau-bench]] — LLM user simulator design, state-based reward and its insufficiency, pass^k definition, simulator limitations, invalid tool-call counts (chapter excerpt).
- [[tau2-bench]] — dual control, tool-constrained user prompt, personas, simulator error rates and error types (chapter excerpt).
- [[apigen-mt]] — blueprint-then-dialogue generation, Best-of-N user, discarding failures, distillation into xLAM-2.
- [[kimi-k2]] — LLM-generated personas, stateful tool simulator, rubric judge filtering, length-budget penalty.
- [[userrl]] — gym simulators, training vs evaluation simulator, SFT cold start, real-user comparison, RL and SFT configurations (chapter excerpt).
- [[simulator-collapse]] — collapse definition and bound, diagnostics, fixes, human study, measurement caveats.
- [[simulator-collapse-recipe]] — RL configuration and co-training settings for the Recipe table.
- [[lost-in-multi-turn]] — sharded simulation, aptitude and unreliability metrics, simulator inspection and limitations (chapter excerpt).
- [[explorer]] — explore-then-refine generation, verifier confusion matrix, synthetic-only limits.
- [[agent-early-experience]] — reward-free use of non-expert actions, DPO collapse comparison, out-of-domain results.
- [[agent-early-experience-recipe]] — Tau-Bench retail SFT settings.
- [[os-genesis]] — reverse task synthesis, graded trajectory reward model and reward-proportional sampling.
- [[learning-from-failure-nat]] — failure-conditioned training, negative quality and prefix ablations (chapter excerpt).
- [[eto-trial-and-error]] — trajectory-level DPO on success-failure pairs, iteration and granularity failure modes.
- [[ipr-step-level-refinement]] — Monte Carlo step rewards, step-level DPO, SFT-term ablation (chapter excerpt).
- [[glm-5]] — masked erroneous segments in SFT, exclusion of environment-collapse samples and group padding in agentic RL (chapter excerpt).
- [[deepswe]] — compact filtering of truncated and timed-out trajectories.
- [[deepswe-recipe]] — blog-versus-script conflict for masking flags.
- [[openthoughts-agent-glm46-teacher]] — teacher and task-source ablations for agent SFT.
- [[together-coderforge-agent-trajectories]] — large test-verified trajectory set, failed-trajectory statistics, decontamination, limitations (chapter excerpt).
- [[likelihood-displacement]] — mechanism by which a rejected-sample gradient can lower the preferred response's likelihood.
