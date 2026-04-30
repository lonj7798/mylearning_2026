<!-- scope: agentic data synthesis — blueprint-first multi-turn trajectories for tool-using agents
     deps: [[self-instruct]], [[magpie]]
     see-also: [[evol-instruct]], [[oss-instruct]], [[rlvr-tulu3]]
-->

# APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay
- **Core Insight:** Multi-turn agent data is much more reliable if you separate it into a verifiable blueprint phase and a dialogue-realization phase; correctness is locked in before conversational realism is added.
- **Guideline:** For tool-use / agent SFT, build a ground-truth task plan first, validate it with reviewers and execution checks, then simulate the human-agent exchange and reject incomplete trajectories.
- **Authors:** Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Weiran Yao, Huan Wang, Silvio Savarese, Caiming Xiong
- **Year:** 2025
- **URL:** https://arxiv.org/abs/2504.03601
- **Relevant topics:** agentic data synthesis, multi-turn SFT, function calling, tool-use, simulated human-agent interaction

## Abstract
APIGen-MT is a two-phase framework for generating verifiable multi-turn agent data. Phase 1 builds detailed task blueprints with ground-truth actions using a committee of LLM reviewers and iterative feedback. Phase 2 turns those blueprints into full interaction trajectories through simulated human-agent interplay, then keeps only successful rollouts. The resulting xLAM-2-fc-r family, spanning 1B to 70B parameters, is reported to outperform GPT-4o and Claude 3.5 on τ-bench and BFCL, especially in multi-turn settings, while also improving consistency across repeated trials.

## Key Contributions
- Introduces a **blueprint-to-dialogue** pipeline for multi-turn agent data rather than one-shot conversation synthesis.
- Uses **committee review, iterative feedback, and rejection sampling** to make the generated data verifiable.
- Releases **5K synthetic trajectories** plus the trained **xLAM-2-fc-r** model family.
- Shows that smaller models trained on this data can outperform larger frontier baselines in multi-turn tool-use settings.

## Technical Details
### Phase 1: blueprint generation
- Model the agent task as a POMDP and represent each τ-bench domain as a directed graph of APIs and dependencies.
- Use multiple samplers to diversify the task context: API, policy, domain data, persona, and example samplers.
- Generate a task configuration with a user instruction, a sequence of ground-truth actions, and the expected final output.
- Validate each configuration in three stages: action validation checks format, execution, and policy compliance; alignment validation checks that the action sequence satisfies the user intent; final semantic review uses committee aggregation and refinement.
- Apply **reverse task recombination** to compose more complex tasks from validated building blocks.

### Phase 2: simulated human-agent interplay
- Convert the validated blueprint into a multi-turn conversation by simulating both the user and the agent with LLMs.
- Use rejection sampling so only trajectories that actually reach the task goal are kept.
- In the τ-bench case study, the authors source **15 read** and **13 write** APIs and use **GPT-4o** plus **DeepSeek V3** during generation, validation, and interplay.
- The paper reports that the collected tasks average around **12 turns** and that agentic feedback improves task-collection success rate to about **70%**.

## Quality / diversity evaluation
- On **BFCL v3**, the xLAM-2-8b-fc-r model is reported to reach **69.25%** on multi-turn tasks, above the cited GPT-4o baseline in the paper materials.
- On **τ-bench**, xLAM-2-70b-fc-r is reported at **56.2% overall**, above Llama 3.1 70B Instruct and GPT-4o in the cited comparison.
- The main empirical point is not just average score but **consistency across trials**, where the APIGen-MT-trained models hold up better in multi-turn settings.

## Risks + gotchas
- **Schema dependence:** the data quality is tied to the API graph and policy structure of the source benchmark.
- **Simulation gap:** an LLM-simulated user is useful for scale, but it can still miss some real human interaction quirks.
- **Validation overhead:** the blueprint stage adds substantial filtering and review cost, which is the price of getting verifiability.

## Connections
- Follows the synthetic-data lineage of [[self-instruct]] and [[magpie]], but targets tool-using multi-turn agents rather than single-turn SFT.
- Shares the iterative refinement flavor of [[evol-instruct]], though APIGen-MT is more explicitly grounded in executable task blueprints.
- Complements [[oss-instruct]] and [[rlvr-tulu3]] as part of the agentic / post-training stack for open models.
