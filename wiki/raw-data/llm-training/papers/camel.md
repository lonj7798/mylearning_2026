<!-- scope: multi-turn conversation synthesis — CAMEL role-playing between an AI user agent and an AI assistant agent with Inception Prompting; AI Society, Code, Math, Science datasets; LLaMA-7B fine-tuning
     deps: [[self-instruct]]
     see-also: [[baize-construction]], [[ultrachat-pipeline]], [[apigen-mt]], [[agentinstruct]]
-->

# CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
- **Core Insight:** Two gpt-3.5-turbo agents, one instructing (AI user) and one solving (AI assistant), are started from a task-specifier prompt and two system prompts and then prompt each other until a termination condition; for 100 AI Society tasks, GPT-4-summarized role-play solutions beat single-shot gpt-3.5-turbo solutions in 76.3% of human votes and 73.0% of GPT-4 judgments (§3.2, §5.1 Table 1).
- **Guideline:** When generating multi-turn instruction data with two agents, write explicit role-lock, output-format, and end-of-task rules into both system prompts and add hard stops (no instruction for 3 rounds, role reversal, token limit, 40 messages), because the paper observed role flipping, flake replies, and endless "thank you" loops and designed these rules to stop them (§3.2, §4.1); the end-of-task token only records that the user agent judged the task done, so add a separate check when task completion matters (not tested in the paper).
- **Authors:** Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, Bernard Ghanem (KAUST)
- **Year:** 2023 (arXiv v1 2023-03; NeurIPS 2023)
- **URL:** https://arxiv.org/abs/2303.17760
- **Source type:** paper
- **Relevant topics:** role-playing, multi-agent data generation, Inception Prompting, multi-turn synthesis, termination conditions, distillation to LLaMA-7B

## Abstract
Chat LLMs depend on human input to steer a conversation toward completing a complex task. The paper proposes a communicative-agent framework called role-playing, in which inception prompting guides two chat agents toward task completion while keeping them consistent with human intentions. It uses role-playing to generate conversational data for studying the behavior and capabilities of a society of agents, studies instruction-following cooperation between agents, and releases an open-source library (Abstract).

## Key Contributions
- Role-playing framework: a human gives an idea and two roles; a task-specifier agent makes the task specific; the AI user and AI assistant then cooperate through multi-turn instruction–solution exchanges (§3.1, Fig. 1).
- Inception Prompting: task-specifier prompt P_T, assistant system prompt P_A, user system prompt P_U, with role, format, and termination rules (§3.2, Fig. 2).
- Two conversational datasets (AI Society, Code), two single-turn problem–solution datasets (Math, Science), and a Misalignment dataset (§1). Role and domain lists are in App. E, Math/Science generation in App. F, and a "bad mind" role-play example in App. B.
- Agent evaluation against single-shot gpt-3.5-turbo (§5.1) and progressive fine-tuning of LLaMA-7B on the generated data (§5.2–5.3).
- Critic-in-the-loop option in which an AI or human critic selects among proposals (§3.1; App. O).

## Key Figures/Tables to Study
- Fig. 2 (AI Society Inception Prompt), §4.1 termination conditions, Table 1 (agent evaluation), Table 2 (per-domain wins after each added dataset), Table 3 (HumanEval), App. J Figs. 8–10 (termination reasons, prompt ablations, flake messages).

## Technical Details
- **Loop.** `M_t = {(I_0,S_0), …, (I_t,S_t)}`; the user produces `I_{t+1} = U(M_t)` and the assistant produces `S_{t+1} = A(M_t, I_{t+1})`; then `M_{t+1} ← M_t ∪ (I_{t+1}, S_{t+1})` (§3.1 Eq. 1–4). `I_t` is the user instruction and `S_t` the assistant solution at step t; both agents receive the full message history.
- **Prompt rules.** Assistant prompt: "Never forget you are a <ASSISTANT_ROLE> and I am a <USER_ROLE>. Never flip roles! Never instruct me!"; decline instructions it cannot perform for "physical, moral, legal reasons or your capability"; start with "Solution:" and end with "Next request." (§3.2, Fig. 2). User prompt: instruct only as "Instruction: … Input: …" or "Input: None", one instruction at a time; reply with the single word `<CAMEL_TASK_DONE>` when the task is completed (§3.2, Fig. 2).
- **Task specifier.** "Please make it more specific. Be creative and imaginative. Please reply with the specified task in <WORD_LIMIT> words or less." (Fig. 2).
- **Termination.** User gives no instruction for 3 rounds; assistant gives an instruction (role reversal); end-of-task token; either agent reaches the gpt-3.5-turbo token limit; maximum of 40 messages, set because cost grows quadratically with conversation length (§4.1).
- **Observed failure modes.** Role flipping, assistant repeating the instruction, flake replies ("I will…"), and infinite message loops (§4.1).
- **AI Society data.** LLM-generated 50 assistant roles and 50 user roles, 10 tasks per role pair, 25,000 conversations (§4.1). Example assistant roles: Accountant, Actor, Administrator, Analyst; example user roles: Accountant, Actor, Artist, Athlete (App. E, Fig. 6). Agents: two gpt-3.5-turbo instances (§4).
- **Code data.** 20 LLM-generated programming languages (Java, Python, JavaScript, …) and 50 domains (App. E, Fig. 6); tasks per pair and conversation count are not reported.
- **Math and Science data.** GPT-4 generates 25 topics, 25 subtopics per topic, and 80 problems per (topic, subtopic) for Math, giving 50K pairs; Physics, Biology, and Chemistry use 32 problems per pair, giving 20K pairs each (App. F, Table 4). These are single-turn (§1).
- **Termination statistics.** AI Society conversations end mostly by "Assistant Instruct"; Code conversations end mostly by "Token Limit"; termination by the 40-message cap is low in both (App. J, Fig. 8). Numeric shares are given only in the figure.
- **Prompt ablation.** Removing the Solution/Next-request format (Prompt V2) or adding a task planner raises end-of-task-token terminations and lowers assistant-instruct terminations, but increases flake messages relative to the original prompt (App. J, Figs. 9–10).
- **Release.** Datasets at https://huggingface.co/camel-ai (App. J); library at https://github.com/camel-ai/camel (Abstract).

## Recipe ledger
| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| CAMEL-7B (LLaMA-7B fine-tuned on AI Society, Code, Math, Science) | 7B | distill-SFT | epochs | 3 | arXiv:2303.17760v2 App. K.2 Table 5 | verified 2026-09-14 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | peak LR; schedule; warmup | 2e-5; cosine; warmup ratio 0.04 | App. K.2 Table 5 | verified 2026-09-14 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | weight decay | 0 | App. K.2 Table 5 | verified 2026-09-14 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | batch | 4 per GPU, gradient accumulation 8, 4×A100-80GB | App. K.2 Table 5; App. K.3 | verified 2026-09-14 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | global batch (sequences) | 128 | derived: 4 per GPU × 8 accumulation × 4 GPUs, assuming data parallelism across the 4 GPUs of App. K.3 | derived | — |
| CAMEL-7B | 7B | distill-SFT | precision; memory | BF16, TF32, gradient checkpointing enabled | App. K.2 Table 5 | verified 2026-09-14 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | data order | AI Society, then + Code, + Math, + Science (progressively growing datasets) | §5.2 | verified 2026-09-14 | Table 2 caption: "Model 2 almost always performs better than Model 1, especially on the added dataset" |
| CAMEL-7B | 7B | distill-SFT | sequence length; loss masking; packing; example counts per stage | not reported | checked §5, App. K | not reported | — |
| CAMEL 13B; CAMEL∗ 33B (CAMEL + ShareGPT + Alpaca) | 13B; 33B | distill-SFT | all training settings | not reported (Table 5 covers LLaMA-7B only) | checked App. K.2, App. R | not reported | — |

## Findings relevant to generality, long context, agentic training, distillation
- **Agent evaluation (Result, single study).** 100 AI Society and 100 Code tasks; GPT-4 summarizes the CAMEL conversation into one solution, compared with single-shot gpt-3.5-turbo. AI Society, human evaluation (453 responses): CAMEL 76.3%, gpt-3.5-turbo 10.4%, draw 13.3%. AI Society, GPT-4 evaluation: 73.0% / 23.0% / 4.0%. Code, GPT-4 evaluation: 76.0% / 24.0% / 0.0% (§5.1, Table 1). Human evaluation was not run for Code (§5.1).
- **Cross-domain transfer (Table 2, GPT-4 judge, 20/20/20/60 questions).** LLaMA-7B vs AI-Society model: Code 0/0/20 (draw / LLaMA / AI Society) and Math 9/5/6. Adding Code: Science 1/19/40, Math 5/8/7. Adding Math: Math 1/3/16. Adding Science: Science 9/2/49, Math 10/5/5. LLaMA-7B vs the all-four model: the all-four model wins all 120 questions (§5.2, Table 2). The authors attribute Code → Science gains to scientific domains inside Code tasks (§5.2).
- **Code benchmark.** HumanEval pass@1 / pass@100: gpt-3.5-turbo 69.4 / 94.0, LLaMA-7B 10.5 / 36.5, Vicuna-7B 11.0 / 42.9, CAMEL-7B 14.0 / 57.9. HumanEval+: gpt-3.5-turbo 61.7 / 89.8, Vicuna-7B 9.9 / 34.7, CAMEL-7B 12.2 / 50.0; LLaMA-7B not reported (§5.3, Table 3).
- **Other baselines.** GPT-4 judge, CAMEL agent solution vs gpt-3.5-turbo with zero-shot CoT: 68.0% / 28.0% / 4.0% draw (App. S, Table 8). Original vs ablated Inception Prompt (communication-protocol and alignment lines removed): original wins 75.0%, ablated 25.0%, draw 0.0% (App. T, Table 9).
- **Broad benchmarks after SFT (App. R, Table 7, lm-evaluation-harness).** LLaMA 13B → CAMEL 13B: ARC-C 56.2 → 55.6, HellaSwag 80.9 → 79.3, MMLU 47.7 → 49.7, TruthfulQA 39.5 → 47.4, average 56.1 → 58.0. CAMEL∗ 33B: 63.0 / 83.8 / 59.0 / 50.2, average 64, against LLaMA 33B average 61.7. At 13B, fine-tuning on the role-play data lowered two of four scores while raising the average (derived from Table 7); the authors describe the CAMEL variants as "substantial improvements" (App. R).
- **Evaluation limits stated by the authors.** Human and LLM evaluators "may be biased or unreliable"; human evaluators may prefer longer answers; evaluating task completion at scale would need domain experts (App. K.1). Table 2 questions were generated by gpt-3.5-turbo with few-shot questions partly from the Vicuna evaluation (App. I).
- **Multi-turn dependence.** Each turn is conditioned on the full history (Eq. 2–3), but the paper does not measure whether later instructions depend on earlier turns, whether constraints from earlier turns are kept, or whether the task is solved when `<CAMEL_TASK_DONE>` is emitted. No per-turn filtering of generated conversations is reported.
- **Conversation length.** Turn-count distributions and token counts per conversation are not reported; the paper reports only the 40-message cap and termination reasons (§4.1; App. J).

## Connections
- [[self-instruct]] — cited in §2 as a semi-automated instruction-generation process; CAMEL replaces single-model generation with two interacting agents.
- [[baize-construction]] — Baize collects data by ChatGPT self-chat from seed questions, one model playing both sides.
- [[ultrachat-pipeline]] — UltraChat §2 describes CAMEL as producing "115k instruction-response pairs"; the CAMEL paper itself reports 25,000 AI Society conversations and does not give a pair count.
- [[apigen-mt]] — 2025 multi-turn agent data pipeline that first builds task blueprints with ground-truth actions checked by a committee of LLM reviewers, then simulates human–agent interplay (APIGen-MT Abstract); it cites CAMEL as prior agent work (§1, ref. [21]).
- [[agentinstruct]] — cites CAMEL as an example of multi-agent workflows for synthetic data (§1, ref. [13]).
- [[openassistant]] — human-written multi-turn conversations as a contrast to agent-generated ones.
- [[soda]], [[system-prompt-diversity]] — other conversation-synthesis cards that link here.

## Verification
- Checked on 2026-09-14 against: https://arxiv.org/abs/2303.17760 (v2, 2023-11-02, NeurIPS 2023 version) including App. E, F, I, J, K, O, R, S, T.
- Corrections to the previous card version:
  - Title "…Exploration of Large Language Models" → "…Exploration of Large Language Model Society".
  - "50 assistant roles × 50 user roles × 20 topic domains yields 25K conversations" → 50 × 50 role pairs × 10 tasks = 25,000 AI Society conversations (§4.1); 20 is the number of programming languages in Code (App. E).
  - "1M-instance AI Society dataset" / "~1M role-play instruction pairs" → 25,000 conversations (§4.1); no 1M figure appears in the paper.
  - "turn cap (~20)" → maximum of 40 messages, plus four other termination conditions (§4.1).
  - "Code, Math ~50K–100K each" → Math 50K and Science 20K per subject, both single-turn and GPT-4-generated (App. F); Code size not reported.
  - "Teacher model: GPT-3.5-Turbo" (for all data) → gpt-3.5-turbo for role-play agents (§4); GPT-4 for Math and Science (App. F).
  - Role examples "Architect, Astronaut, Biologist" and "Graduate Student" → not in the generated role lists; actual lists in App. E Fig. 6.
  - "CAMEL-Math exceeds baseline on GSM8K" → no GSM8K evaluation; math ability is judged by GPT-4 on 20 questions (§5.2, Table 2).
- Removed as unsupported by the source: filtering by "min 4 turns" and "coherence check"; "avg 6–10 turns", "median 6–8, tail to 20"; cost "~$5K–10K"; "diversity higher than Baize, comparable to UltraChat"; "adopted in Airoboros, OpenHermes-2.5"; "Inception-prompting widely reused in APIGen-MT, AgentInstruct" (both only cite CAMEL); "agents introduce themselves in every turn"; "`<CAMEL_TASK_DONE>` leaks into student models"; "Astronaut × Bartender low-quality pairs"; "safety post-filter inherits GPT-3.5's safety layer"; "the role-specification is what unlocks diversity".
- Not reported by the source: turn-count and token-length distributions, API cost of the datasets, number of Code conversations, SFT sequence length and loss masking.
