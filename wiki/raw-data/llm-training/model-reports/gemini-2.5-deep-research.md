<!-- scope: Gemini 2.5 technical report (arXiv:2507.06261): Gemini 2.X family, sparse MoE, k-sparse distillation for Flash-size models, post-training and RL-trained thinking at the level of detail Google discloses, Deep Research and Deep Think descriptions, long-context evals, the Gemini Plays Pokémon agent study, RL*F safety training and over-refusal. The slug keeps "deep-research" so existing links resolve.
     see-also: [[gemma-2]], [[gemini-long-context-tricks]], [[generative-reward-models]], [[constitutional-ai]], [[qwen-3]], [[grok-4-1]]
-->

# Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities
- **Core Insight:** The report attributes Gemini 2.5 post-training gains to data-quality work across SFT, reward modeling, and RL, more RL compute, verifiable and model-based generative rewards, and RL stability changes, and reports LMArena Elo gains of 122 (2.5 Pro) and 111 (2.5 Flash) over the Gemini 1.5 counterparts; it names no RL algorithm and gives no RL hyperparameters (§2.4).
- **Guideline:** When long context is meant for an agent, evaluate multi-step generation at the context lengths the agent will reach and not only retrieval, because 2.5 Pro scores 69.8% on LOFT hard retrieval but 16.4% on MRCR-V2 8-needle at 1M tokens (Table 3), and its Pokémon agent tended to repeat past actions once the context grew well beyond 100k tokens (§4.1, labeled anecdotal by the authors).
- **Authors:** Gemini Team, Google (arXiv listing: Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, et al.)
- **Year:** 2025 (arXiv v1 2025-07; v6 2025-12-19)
- **URL:** https://arxiv.org/abs/2507.06261 (same report: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
- **Source type:** official technical report
- **Relevant topics:** sparse MoE, distillation with k-sparse teacher distributions, RL with verifiable and generative rewards, thinking budget, long-context evaluation, agent harness, Deep Research, Deep Think, RL*F safety training, over-refusal

## Abstract
The report introduces Gemini 2.5 Pro and Gemini 2.5 Flash together with the earlier Gemini 2.0 Flash and Flash-Lite. Gemini 2.5 Pro is described as a thinking model with state-of-the-art results on frontier coding and reasoning benchmarks, multimodal understanding, and the ability to process up to 3 hours of video. Gemini 2.5 Flash offers reasoning at lower compute and latency, and the 2.0 models target low latency and cost. The authors state that the family spans the Pareto frontier of capability versus cost and that the combination of long context, multimodality, and reasoning enables new agentic workflows.

## Key Contributions
- A model family of sparse mixture-of-experts transformers with native text, vision, and audio input, more than 1M input tokens, and native tool use; 2.5 Flash is a hybrid reasoning model with a controllable thinking budget (§1, §2.1, Table 1).
- Distillation of Flash-size and smaller models against a k-sparse approximation of the teacher's next-token distribution (§2.1).
- A qualitative description of post-training changes (§2.4) and of thinking trained with RL (§2.5).
- Capability-specific sections, including the Deep Research agent (§2.6) and Deep Think (§2.7).
- A case study of an agent that completed Pokémon Blue, with its harness and failure modes (§4.1, App. 8.2).
- Safety training with RL from Human and Critic Feedback (RL*F), and a reported reduction of over-refusal relative to 2.0 (§5.3-5.4).

## Key Figures/Tables to Study
- Table 1: family comparison (input/output length, thinking, tool use, knowledge cutoff).
- Table 3: core benchmarks, including LOFT and MRCR-V2 at ≤128K and at 1M. Table 4: 2.5 Pro against other labs' models.
- Figures 3-4: accuracy with and without thinking, and accuracy against thinking budget.
- Figure 6 and App. 8.2 Figure 14: Pokémon progress timeline and agent harness.
- Table 7: safety and helpfulness deltas of 2.0 and 2.5 relative to 1.5.

## Technical Details
**Architecture and distillation.** Gemini 2.5 models are sparse MoE transformers; the report states that training-stability and optimization work gave "a considerable boost in performance straight out of pre-training" (§2.1). Models of Flash size and below use distillation, as in Gemini 1.5. The teacher's next-token distribution is approximated by a k-sparse distribution over the vocabulary, which still multiplies training data throughput and storage by k (§2.1). The value of k and the teacher model are not stated.

**Data.** Pre-training data cutoff is June 2024 for 2.0 and January 2025 for 2.5, with new filtering and deduplication methods (§2.2). Post-training data consists of multimodal instruction-response pairs plus human preference and tool-use data (§2.2).

**Infrastructure.** First family trained on TPUv5p, with synchronous data parallelism over multiple 8960-chip pods across datacenters (§2.3). During slice recovery, training continues at about 97% throughput (§2.3). About 0.25% of steps were replayed for suspected silent data corruption, and 6% of those replays were genuine hardware corruption (§2.3). 93.4% of run time was TPU computation, and about 4.5% of computed steps were replays or rollbacks (§2.3).

**Post-training.** Quality control uses the model itself across SFT, RM, and RL data (§2.4). RL compute was increased, rewards focus on "verifiable rewards and model-based generative rewards", and "algorithmic changes to the RL process" improved stability in longer training (§2.4). RL environments include ones "requiring multi-step actions and tool use" (§2.4).

**Thinking.** Thinking models are trained with RL to spend "tens of thousands of forward passes" before answering (§2.5). The recipe evolved from Gemini 2.0 Flash Thinking (December 2024) to thinking integrated across domains (§2.5). A thinking budget limits thinking tokens; Figure 4 shows accuracy rising with budget on AIME 2025, LiveCodeBench, and GPQA diamond (§2.5; values only in the figure).

**Video.** Models were trained to perform competitively with 66 instead of 258 visual tokens per frame, allowing about 3 hours of video instead of 1 hour in a 1M-token context (§2.6).

**Deep Research and Deep Think.** Deep Research is an agent built on 2.5 Pro that browses the web (§2.6). Its Humanity's Last Exam score rose from 7.95% (December 2024) to 26.9%, and to 32.4% with higher compute (June 2025) (§2.6). Deep Think "blends in parallel thinking techniques during response generation" to produce and critique multiple hypotheses; it was announced at Google I/O and released experimentally to trusted testers in June 2025 (§2.7). The report gives no training details for either.

**Safety training.** Methods are data filtering, conditional pre-training, SFT, and RL*F (§5.3). SFT fixes policy violations and unnecessary refusals with data-generation recipes "loosely inspired by Constitutional AI" plus human revision (§5.3). The RL*F reward combines a Data Reward Model trained on human preference comparisons and a Critic, a prompted model that grades against rubrics (§5.3).

## Recipe ledger

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Gemini 2.5 Pro, 2.5 Flash | not reported | pretrain | architecture | sparse MoE transformer; parameter counts not stated | arXiv:2507.06261v6 §2.1 | verified 2026-09-14 | no ablation reported |
| Gemini 2.5 Pro, 2.5 Flash | not reported | pretrain | data cutoff | January 2025 | §2.2; Table 1 | verified 2026-09-14 | n/a |
| Gemini 2.5 (run described in §2.3; model not named) | not reported | pretrain | compute | TPUv5p; synchronous data parallelism over multiple 8960-chip pods; 93.4% of time in TPU computation | §2.3 | verified 2026-09-14 | n/a |
| Gemini 2.5 Pro, 2.5 Flash | not reported | pretrain | tokens, batch, LR, optimizer, schedule, mixture shares | not reported | checked §2.1-§2.3, App. 8 | not reported | n/a |
| Gemini 2.5 Flash and smaller | not reported | pretrain | distillation target | k-sparse approximation of the teacher's next-token distribution; data throughput and storage × k; k and teacher not stated | §2.1 | verified 2026-09-14 | §2.1 cites "significant quality improvement" without numbers; Figure 2 shows serving throughput only |
| Gemini 2.5 Pro, 2.5 Flash | not reported | long-context | input / output length | 1M / 64K tokens | Table 1 | verified 2026-09-14 | no ablation reported; Table 3 evaluates at ≤128K and 1M |
| Gemini 2.5 Pro, 2.5 Flash | not reported | SFT | data | multimodal instruction-response pairs, human preference and tool-use data; sizes not stated | §2.2 | verified 2026-09-14 | no ablation reported |
| Gemini 2.5 Pro, 2.5 Flash | not reported | RL | reward types | verifiable rewards and model-based generative rewards | §2.4 | verified 2026-09-14 | LMArena Elo +122 Pro, +111 Flash vs 1.5 (§2.4, Figure 1); not isolated to rewards |
| Gemini 2.5 Pro, 2.5 Flash | not reported | RL | algorithm, KL, clip ε, LR, batch, samples per prompt, steps, max length | not reported | checked §2.4, §2.5, §5.3, App. 8 | not reported | n/a |
| Gemini 2.5 Pro, 2.5 Flash | not reported | RL | safety reward (RL*F) | Data Reward Model + prompted rubric Critic; combination weights not stated | §5.3 | verified 2026-09-14 | Table 7 deltas vs 1.5 |

The report does not separate pre-training into stable and decay phases, so pre-training rows use the stage label `pretrain`. The report discloses no optimizer, batch, token, or RL hyperparameter values, so no starting configuration can be derived from it.

## Findings relevant to generality, negative feedback, long context, agentic training, distillation
- **Long context.** 2.5 Pro scores 87.0% (≤128K) and 69.8% (1M) on LOFT hard retrieval, and 58.0% (≤128K) and 16.4% (1M) on MRCR-V2 8-needle (Table 3). 2.5 Flash scores 21.0% on MRCR-V2 at 1M, above 2.5 Pro (Table 3). The ≤128K variant covers contexts up to 128K; the 1M variant uses exactly 1M (§3.1).
- **Long-context agents.** The pathfinder tool, a prompted 2.5 Pro instance, often reasoned over 100K+ token contexts and found paths up to 50 actions long, and up to 150 actions in the extreme case (§4.1). "As the context grew significantly beyond 100k tokens, the agent showed a tendency toward favoring repeating actions from its vast history rather than synthesizing novel plans"; the authors call this anecdotal and separate long context for retrieval from long context for multi-step generative reasoning (§4.1). The harness keeps the action sequence in context with a "summary clear" every 100 turns and compresses the stored summaries again every 1000 turns (App. 8.2). App. 8.2 also reports "context poisoning", where wrong goals or summaries persist in context and the agent fixates on impossible goals.
- **Agentic use (training details not disclosed).** The Pokémon agent finished the game in 813 hours with Gemini 2.5 Pro Exp 03-25 while the harness was being changed, and in 406.5 hours with Gemini 2.5 Pro Preview 05-06 in a fully autonomous run with a fixed harness (§4.1). The harness supplies screen information as text derived from game RAM; in one ablation with all vision removed, the model functioned "roughly as well" (§4.1). No RL environment counts or Deep Research training details are disclosed.
- **Distillation.** 2.5 Flash, a distilled Flash-size model (§2.1), is reported to overtake Gemini 1.5 Pro (§3.2).
- **Negative feedback and over-refusal.** Gemini 2.0 models "over-refused on a wide variety of benign user requests", and 2.5 training targeted reduced refusals (§5.4). Helpfulness/instruction following changes by +13.6% (2.5 Flash vs 1.5 Flash 002) and +14.8% (2.5 Pro vs 1.5 Pro 002), against −13.2% for 2.0 Flash vs 1.5 Flash 002 (Table 7). Failing responses are revised into SFT targets (negative as content, §5.3); how RL*F penalizes responses is not described.
- **Generality and measurement.** The authors state that most recipe changes since 1.5 improved all capabilities (§2.6). SWE-bench Verified numbers from providers use different scaffolds and are "not directly comparable" (§3.1). LiveCodeBench for 1.5 Pro is 30.5% in §2.6 and 29.7% in Table 3; Table 3 states its date window (1/1/2025-5/1/2025, §3.1) and §2.6 does not. The authors report benchmark saturation: Aider Polyglot rose 5× and SWE-bench Verified 2× in one year (§6).

## Connections
- [[gemma-2]]: Google DeepMind open models whose 2B and 9B are trained with distillation; that report also notes distillation in Gemini 1.5.
- [[gemini-long-context-tricks]]: Gemini API guidance on using long context; covers usage, not training.
- [[generative-reward-models]]: background for the "model-based generative rewards" (§2.4) and the rubric Critic (§5.3).
- [[constitutional-ai]]: named in §5.3 as the loose inspiration for safety SFT data recipes.
- [[qwen-3]]: another 2025 report with a thinking budget in one model.
- [[grok-4-1]]: xAI release whose card describes reasoning models used as reward models; compare the prompted rubric Critic (§5.3).

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2507.06261 (v6, 2025-12-19) and https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf (PDF dated 2025-10-16); all quoted passages appear in both.
- Corrections to the previous card version:
  - Card used a descriptive title "Gemini 2.5 (Deep Think / Deep Research)" and mixed a Google blog post (gemini-model-thinking-updates-march-2025) with the report → card now describes the technical report under its exact title.
  - "no arXiv paper for 2.5 Pro's post-training" → the report is arXiv:2507.06261.
  - "post-training was the primary driver of the generational jump" → the report credits both pre-training stability work (§2.1) and post-training (§2.4) and gives no split.
  - "Deep Research (May 2025)" → Deep Research first launched December 2024; HLE 26.9%/32.4% as of June 2025 (§2.6).
  - "Deep Think rolled out Aug 2025" → announced at Google I/O, experimental release to trusted testers in June 2025 (§2.7).
  - "verifiable components implied by tool-use environments" → §2.4 states verifiable rewards and model-based generative rewards directly.
  - "RM architecture not disclosed" → §5.3 describes the safety reward as a Data Reward Model plus a prompted rubric Critic.
  - Omissions added: k-sparse distillation (§2.1), LMArena +122/+111 (§2.4), over-refusal (§5.4, Table 7), LOFT/MRCR-V2 (Table 3), Pokémon long-context finding (§4.1).
- Removed as unsupported by the source: RL "co-designed with sample-then-refine-then-combine inference"; guideline on reward shaping for multi-sample aggregation; "Deep Think combines RL with multi-sampling"; "Deep Research also RL-trained"; "Deep Think pairwise-preference plots"; "Deep Research agentic-task breakdown"; "consistent with industry-wide entropy-collapse mitigations"; "Google has historically not named its RL algorithm"; similarity to Kimi K1.5 prioritized sampling; "implies iterative exploration + reward during training".
- Not reported by the source: parameter counts, pre-training tokens, optimizer settings, k, teacher identity, SFT and preference data sizes, RL algorithm and hyperparameters, RL environment counts, Deep Think sample counts, Deep Research training method.
