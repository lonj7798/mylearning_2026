<!-- scope: Agent-FLAN (arXiv 2403.12881): agent SFT data redesign — ReAct-to-chat alignment, capability-weighted mixing, negative samples for tool-call hallucination, Agent-H benchmark
     deps: [[agenttuning]], [[toolllm]]
     see-also: [[fireact]], [[lumos]], [[agentinstruct]]
-->

# Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for Large Language Models
- **Core Insight:** Rewriting ReAct-format agent data as multi-turn chat, re-weighting it by capability, and adding two types of negative samples raises the Llama2-7B overall agent score to 41.7, versus 38.2 for a same-data re-implementation of AgentTuning (Table 1).
- **Guideline:** When agent SFT data uses a fixed ReAct/JSON format, convert it to multi-turn conversation and add samples whose correct target is a plain-text reply (no tools given, or only irrelevant tools given), because in the Llama2-7B ablations these steps raised T-Eval from 61.8 to 64.9 (Table 2) and Agent-H HScore from 84.5 to 89.1 (Table 3).
- **Authors:** Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, et al. (USTC; Shanghai AI Laboratory)
- **Year:** 2024 (arXiv v1 2024-03; arXiv comment "Technical Report")
- **URL:** https://arxiv.org/abs/2403.12881
- **Source type:** paper
- **Relevant topics:** agent SFT, data format alignment, capability-weighted data mixing, tool-call hallucination, negative samples, general vs agent ability

## Abstract
The paper reports three observations about agent tuning of open LLMs: (1) agent training data mixes format following with reasoning and differs from the natural-conversation data of pre-training; (2) when the data is split by capability, the loss for each capability converges at a different speed; (3) existing agent-tuning methods introduce hallucinations. Agent-FLAN redesigns the training corpus in response: it converts agent data to chat format, decomposes it by capability and rebalances it, and adds negative samples. With Llama2-7B it outperforms prior work by 3.5% across agent evaluation sets (Table 1: overall 41.7 vs 38.2). The authors also build the Agent-H hallucination benchmark and report that Agent-FLAN improves agent scores as model size grows while slightly improving general capability.

## Key Contributions
- Three pilot observations on format overfitting, per-capability learning speed, and hallucination (§3, Figs. 2-4).
- Alignment of agent data to chat format: ReAct templates replaced by multi-turn dialogue (§4.1, Fig. 5).
- Capability decomposition (reasoning, retrieval, understanding, instruction following) and loss-curve-guided data balancing (§4.2, Table 2).
- Agent-H benchmark and two negative-sample types for tool-call hallucination (§4.3, Table 3).
- Data-scaling, model-scaling, and general-ability analyses (§5, Figs. 7-8, Table 4).

## Key Figures/Tables to Study
- Table 1: held-in, held-out, T-Eval, Agent-H, and overall scores for Llama2-7B variants and GPT-3.5/GPT-4.
- Table 2: token counts and T-Eval/HotpotQA for each capability sub-mixture.
- Fig. 6: the four query x system-prompt circumstances; Table 3: Agent-H with and without negative samples.
- Table 4: MMLU, GSM8K, HumanEval at 7B, 13B, 70B. Tables 5-6: hyper-parameters and data counts.

## Technical Details
- **Setup.** Llama2 series; 7B used for ablations (§4 Experimental Setup). Held-in data follows AgentTuning: ALFWorld, WebShop, Mind2Web, Knowledge Graph, Operating System, Database, plus ToolBench. Held-out: HotpotQA, WebArena, SciWorld, T-Eval (§4).
- **Data counts.** OS 195, Database 538, Knowledge Graph 300, WebShop 351, ALFWorld 336, Mind2Web 116, ToolBench 22,867; total 24,703 (App. B, Table 6). Few-shot examples and non-"Thought-Action-ActionInput" samples were removed; ToolBench kept only samples with a final answer and removed DFSDT restart samples (App. B). The ToolBench portion is "roughly 20,000 valid samples", 10% of ToolBench (§7).
- **Observation 1.** On ToolBench ReAct data, the loss on formatted tokens falls faster than the content loss ("0.54 vs 0.04" as printed) (§3, Fig. 2).
- **Capability definitions.** Instruction following = format generation; reasoning = thought quality per step; retrieval = choosing the function name; understanding = the function's parameter inputs (§3 Obs. 2). Retrieval and understanding losses fall faster than reasoning; instruction following falls fastest (Fig. 3).
- **Chat alignment.** "Thought-Action-Action Input" turns become multi-turn dialogue; JSON arguments are split by inserted elicit statements; loss is applied only to assistant turns; instruction-following pairs requesting ReAct and JSON output are added (§4.1). Effect: T-Eval 61.8 → 64.9, HotpotQA 25.4 → 27.9 (Table 2).
- **Capability balancing.** Halving reasoning data lowers T-Eval to 63.8 (−1.1) and halving understanding to 64.6 (−0.3); halving retrieval (65.3) or instruction following (65.9) does not lower it. The weighted mixture uses 18.1M tokens versus 37.3M for the full aligned set and scores T-Eval 66.3, HotpotQA 28.5 (§4.2, Table 2).
- **Hallucination types.** Format hallucination and action hallucination (§4.3, Fig. 4). Agent-H has a format level (requests for various response formats) and an action level (four circumstances split by user query and system prompt) (§4.3, Fig. 6). It uses 1,845 samples from glaive-function-calling-v2 (§4.3).
- **Agent-H metric.** H_ReAct and H_General = responses containing ReAct keywords ("Thought:", "Action:") or general keywords ("I will use", "I need to call") divided by the number of raw-response ground truths. HScore = 0.5·((1 − H_ReAct) + (1 − H_General)). The printed Eq. 1 repeats H_ReAct; the Table 3 values match H_General in the second term (0.5 × (90.1 + 88.1) = 89.1) (§4.3).
- **Negative samples.** Type 1: no tools provided, user query asks for a tool. Type 2: tools provided, user query needs a normal conversation (§4.3). Construction: 761 ToolBench queries answered by gpt-3.5-turbo without tool information; irrelevant tools appended to half of them (App. C, Figs. 10-11).
- **Main result (7B).** Agent-FLAN: held-in 2.01, HotpotQA 28.5, SciWorld 20.0, WebArena 4.68, T-Eval 66.0, Agent-H 89.1, overall 41.7. AgentTuning*: 1.89, 25.4, 16.8, 2.71, 61.8, 84.5, 38.2. AgentLM-7B overall 31.7; Llama2-7B 27.2; GPT-3.5 47.8; GPT-4 55.1 (Table 1).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Agent-FLAN (Llama2 series) | 7B, 13B, 70B | SFT | Epochs | 1 | arXiv:2403.12881v1 App. A | verified 2026-09-14 | no ablation reported |
| Agent-FLAN (Llama2 series) | 7B, 13B, 70B | SFT | Peak LR; warmup LR; schedule | 2e-5; 4e-6; cosine | App. A Table 5 | verified 2026-09-14 | no ablation reported |
| Agent-FLAN Llama2-7B | 7B | SFT | Batch size (unit not stated); GPUs | 32; 16 | App. A Table 5 | verified 2026-09-14 | no ablation reported |
| Agent-FLAN Llama2-13B | 13B | SFT | Batch size (unit not stated); GPUs | 32; 32 | App. A Table 5 | verified 2026-09-14 | no ablation reported |
| Agent-FLAN Llama2-70B | 70B | SFT | Batch size (unit not stated); GPUs | 128; 128 | App. A Table 5 | verified 2026-09-14 | no ablation reported |
| Agent-FLAN (Llama2 series) | all | SFT | ShareGPT : agent corpus mix ratio (unit not stated) | 1:1 | App. A | verified 2026-09-14 | App. A says this follows AgentTuning practice, but [[agenttuning]] reports an agent sampling ratio η = 0.2 (its §2.2.2, App. A); no ablation reported |
| Agent-FLAN (Llama2 series) | all | SFT | Format mix | 10% ReAct format, 90% conversation format | App. A; App. B | verified 2026-09-14 | "empirically find" ReAct data helps (App. A); no table |
| Agent-FLAN (Llama2 series) | all | SFT | Capability weights reasoning : retrieval : understanding | 1 : 0.25 : 0.75 | App. A | verified 2026-09-14 | Table 2 half-data ablation (7B, T-Eval/HotpotQA; number of seeds not stated) and Fig. 3 loss curves |
| Agent-FLAN (Llama2 series) | all | SFT | Extra instruction-following samples | 2,000 | App. A | verified 2026-09-14 | §4.1 states a small portion is enough; no table |
| Agent-FLAN (Llama2 series) | all | SFT | Agent samples (before weighting) | 24,703 | App. B Table 6 | verified 2026-09-14 | no ablation reported |
| Agent-FLAN (Llama2 series) | all | SFT | Negative-sample source queries | 761 ToolBench queries; gpt-3.5-turbo responses | App. C | verified 2026-09-14 | Table 3: HScore 89.1 with vs 84.5 without (7B) |
| Agent-FLAN Llama2-7B | 7B | SFT | Training tokens, weighted mixture | 18.1M | §4.2 Table 2 | verified 2026-09-14 | Table 2: T-Eval 66.3 vs 64.9 for 37.3M unweighted |

Not reported (checked body, App. A-C): sequence length, optimizer and betas, weight decay, warmup length, number of ShareGPT conversations, chat vs base Llama2 checkpoint.

## Findings relevant to generality, negative feedback, agentic training
- **Generality.** General data vs Agent-FLAN rows: Llama2-7B MMLU 50.0 → 49.7, GSM8K 21.9 → 22.1, HumanEval 15.1 → 15.5; 13B 54.7 → 55.8, 34.8 → 35.2, 15.2 → 15.8; 70B 68.0 → 68.5, 64.5 → 64.6, 32.1 → 32.9 (§5.2, Table 4). The authors attribute the gain to reasoning and instruction following in agent data (Interpretation, §5.2).
- **Data scaling.** On HotpotQA, the first 25% of Agent-FLAN data gives the largest gain; 50% and 75% add less (§5.1.1, Fig. 7). The authors conclude that diversity or quality, not volume, is the next lever (Interpretation).
- **Model scaling.** Scores rise from 7B to 13B to 70B without saturation, and the margin over ReAct tuning holds across sizes (§5.1.2, Fig. 8).
- **Negative samples.** These are negative as content: a tool-inducing prompt paired with a plain-text target, trained with ordinary cross-entropy on assistant turns. No likelihood is pushed down. At 7B they lower H_ReAct 15.6 → 9.9 and H_General 13.5 → 11.9, and change T-Eval 66.3 → 66.0 (Table 3). No false-negative rate is reported.
- **Agentic training.** The authors attribute the fast drop of format loss to the model overfitting to fixed ReAct/JSON structure (Interpretation, §3 Obs. 1). The training and evaluation sets cover only part of agent tasks (§7).

## Connections
- [[agenttuning]] — source of the held-in data and the ShareGPT mixing practice; re-implemented as the AgentTuning* baseline.
- [[fireact]] — FireAct-7B is a baseline in Table 1.
- [[toolllm]] — ToolBench supplies 22,867 training samples and the 761 negative-sample queries.
- [[agentinstruct]] — different artifact: the Microsoft AgentInstruct paper, unrelated to AgentTuning's AgentInstruct dataset used here.
- [[lumos]] — another open agent-tuning approach; not compared in this paper.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2403.12881 (arXiv v1, 19 Mar 2024; full PDF incl. App. A-C).
- Corrections: "three capabilities: instruction following, reasoning/decision, generalization" → four capabilities: reasoning, retrieval, understanding, instruction following (§3, §4.2). "Four hallucination modes: format, action, parameter, relevance" → two categories, format and action; Agent-H action level uses four query x system-prompt circumstances (§4.3, Fig. 6). "Negatives generated by prompting GPT-4 with failure patterns" → 761 ToolBench queries answered by gpt-3.5-turbo, half with irrelevant tools (App. C). "Instruction data ShareGPT/Alpaca/Evol-Instruct ~50K" → ShareGPT mixed 1:1 with agent data plus 2,000 instruction-following samples (App. A). "~85K total examples" → 24,703 agent samples (Table 6). "Environments ALFWorld, WebShop, HotpotQA, Mind2Web" → HotpotQA is held-out; ToolBench is held-in (§4). "Align to Llama-2 pretraining tokens/special tokens" → convert ReAct turns to multi-turn chat (§4.1). "Authors: all Shanghai AI Lab" → USTC and Shanghai AI Laboratory.
- Removed as unsupported: AgentBench 3.92 and AgentTuning-13B 2.8 comparison; "5x fewer hallucinated calls"; "~$15K GPT-4 cost"; "6 turns avg", "6-15 turns, 1K-4K tokens"; "generalization data ~10K on held-out APIs"; "MT-Bench within 0.5 of Llama2-Chat"; "removing a capability costs 0.3-0.5 AgentBench points"; "removing negatives triples hallucination rate"; "downstream InternLM agent line"; risk bullets not stated by the paper.
