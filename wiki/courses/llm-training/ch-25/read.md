<!-- chapter: ch-25
     track: synthetic
     kind: content
     title: Multi-Turn Conversation Synthesis
     deps: [ch-24]
     sources: [[baize-construction]], [[baize-construction-recipe]], [[ultrachat-pipeline]], [[ultrachat-two-model-protocol]], [[camel]], [[prosocial-dialog]], [[soda]], [[soda-commonsense-grounding]], [[openassistant]], [[persona-hub]], [[capybara]], [[smol-talk]], [[modern-mix-composition]], [[tulu-3]], [[wildchat]], [[wildchat-statistics]], [[lmsys-chat-1m]], [[lost-in-multi-turn]], [[mt-eval]], [[false-promise-imitating-proprietary-llms]], [[xstest]], [[agentinstruct]]
     figures: figures/turn-distribution.html, figures/aptitude-unreliability.html
     revised: 2026-09 (generality revision)
-->

# Chapter 25 — Multi-Turn Conversation Synthesis

> **Core insight.** Published multi-turn synthesis methods produce the second speaker in three ways: one API call writes both sides (Baize), two separate calls alternate user and assistant under a topic taxonomy (UltraChat), or two role-locked agents instruct and solve a task (CAMEL). Grounding the scene in external content (commonsense triples in SODA, situations from existing corpora in ProsocialDialog) was preferred by human judges over 100 ungrounded GPT-3.5 dialogues in SODA's controlled comparison ([[soda-commonsense-grounding]] §3.3). Conversations in LMSYS-Chat-1M and WildChat average 2.02 and 2.54 turns ([[wildchat-statistics]] Table 1; [[lmsys-chat-1m]] Table 1), and the models tested by [[lost-in-multi-turn]] lose 39% of their single-turn score on average when a task is revealed over several turns. A dialogue corpus that looks realistic does not guarantee this robustness: imitation data in [[false-promise-imitating-proprietary-llms]] raised crowd ratings without raising benchmark accuracy.
>
> **Guideline.** When synthesizing multi-turn SFT data for a general-purpose model, generate user and assistant turns with separate calls and condition each dialogue on an external content source, because both choices are the documented response to observed failures (role exchange in [[ultrachat-pipeline]] §4.4; dull ungrounded dialogues in [[soda-commonsense-grounding]] §3.3). When the goal is robustness to underspecified requests, include dialogues whose requirements arrive over several turns, because fully specified single-turn instructions do not measure that behaviour ([[lost-in-multi-turn]] §6). Measure the corpus against a real-user reference (turn counts, token lengths, embedding coverage) and evaluate the trained model on held-out multi-turn suites with a single-turn control ([[mt-eval]] Table 3). Otherwise, if only short assistant-style chat is needed, a three-turn synthetic mix generated with Llama-3.1-405B-Instruct is a measured baseline at 1.7B ([[modern-mix-composition]], SmolLM2 Table 10).

## Corrections to the version you studied

1. "Ch-24 handed you the generic synthetic-pipeline loop" → the generic generate-filter-deduplicate-verify-select-mix loop is ch-18 (The Synthetic-Data Design Pattern). ch-24 covers reasoning traces.
2. "Seed pools: Quora ~54K, StackOverflow ~57K, Alpaca ~52K, MedQuAD ~47K → 111.5K dialogues" → the 111.5k self-chat dialogues come from about 55k Quora and 55k Stack Overflow questions; Alpaca's 51,942 examples are single-turn data added to training; MedQuAD is a separate 46,867-dialogue healthcare set ([[baize-construction]], §3, Table 2).
3. Baize template with `[|Human|]` / `[|AI|]` and "~8-turn cap" → the self-chat template uses `[Human]` / `[AI]` and runs until a natural stopping point; `[|Human|]` belongs to the inference prompt ([[baize-construction]], App. A, App. B).
4. "Median 4 turns, IQR 3–6, ~100 tokens per turn; total cost ~$1,000" → average 3.6–3.9 turns and average response length about 36 (unit not stated); cost approximately $100 ([[baize-construction]], §3, Table 2).
5. "Self-Distill with Feedback ... use critiques as DPO pairs" → SDF samples four responses per Quora question, ChatGPT scores each from 1 to 100, and the best one trains new LoRA modules; there is no DPO. Baize (arXiv 2023-04) predates DPO (arXiv 2023-05) ([[baize-construction]], §4, App. C).
6. "Baize-13B beat Alpaca-13B on 58% of human-eval comparisons" → not in the paper. The reported evaluation is a GPT-4 score relative to ChatGPT: Alpaca-13B 76%, Baize-v2-13B 92% ([[baize-construction]], Fig. 3).
7. UltraChat "30 meta-topics → 1,100+ subtopics; ~100K C4 passages; Q-world ~600K / Writing ~400K / Assistance ~500K" → the paper states 30 topics with 30–50 subtopics each, about 500,000 opening questions for Sector I, 10,000 C4 pieces with five instructions each, and a total of 1,468,352 dialogues; per-sector dialogue counts are not reported. The 1,100+ and ~100k figures come from the repository README ([[ultrachat-two-model-protocol]]; [[ultrachat-pipeline]] arXiv v1 §4.1–4.3, Table 5).
8. "The filter stage is opaque" → the paper names one filter: excessively polite user statements ("Thank you", "Thanks", "You're welcome") are excluded to make user turns more realistic ([[ultrachat-pipeline]] §4.4).
9. CAMEL "50 × 50 × 20 grid", "AI Society ~1M dialogues", "cost ~$5–10K" → 50 assistant roles × 50 user roles × 10 tasks = 25,000 conversations; no cost is reported ([[camel]], §4.1).
10. CAMEL "median 6–8 turns, tail to 20, ~50 tokens per turn" and "semantic termination" as the stop rule → turn and token distributions are not reported; conversations stop on any of five conditions (no instruction for 3 rounds, role reversal, end-of-task token, token limit, 40 messages), and AI Society conversations end mostly by assistant role reversal ([[camel]], §4.1, App. J Fig. 8).
11. Role examples "Architect, Astronaut, Biologist, Graduate Student", the "Astronaut × Bartender" pair, and the "As an accountant" leak → not in the paper's role lists or findings; the reported failure modes are role flipping, instruction repetition, flake replies, and infinite loops ([[camel]], App. E, §4.1).
12. "Inception prompting propagates into AgentInstruct, APIGen-MT, ToolACE" → AgentInstruct and APIGen-MT cite CAMEL; neither is described as reusing the Inception Prompt. AgentInstruct's suggester agents make tasks more complex or unsolvable; they do not pre-filter implausible role pairs ([[camel]] Connections; [[agentinstruct]] §2).
13. "The Yejin Choi lineage self-instruct → SODA → Prosocial-Dialog is a single arc" → ProsocialDialog (arXiv 2022-05) precedes SODA and Self-Instruct (both arXiv 2022-12); Self-Instruct is not by this group. SODA uses ProsocialDialog's Canary as a safety filter and ProsocialDialog as extra COSMO training data ([[prosocial-dialog]] Verification; [[soda-commonsense-grounding]], §3.1, §4).
14. "Atomic 10X, a commonsense-triple database of ~10M records", "1.5M triples balanced across {xWant, xNeed, xEffect, xReact}" → the SODA paper does not state Atomic10x's size; it uses 1.5 million triples from six x-relations with shares xAttr 18%, xEffect 17%, xIntent 23%, xNeed 7%, xReact 25%, xWant 11% ([[soda-commonsense-grounding]], §3.2 Table 3, App. A.1).
15. SODA "4 to 10 turns total", "median 7–8, max ~12, avg 20 tokens per turn", "cost ~$10K" → filtering removed conversations with fewer than 4 or more than 20 turns; SODA averages 7.6 turns and 16.1 utterance length (unit not stated); one triple-to-dialogue costs about $0.02 ([[soda-commonsense-grounding]], §3.1, Table 2).
16. "COSMO-3B ... beats BlenderBot-3B ... at a third the parameters" → both are 3B; on DailyDialog, human judges preferred COSMO-3B overall in 72% of comparisons against BlenderBot-3B ([[soda-commonsense-grounding]], §5.1 Table 5).
17. ProsocialDialog "crowd workers author ~10K problematic prompts across 10 harm categories", "300+ unique RoTs", "median 3 turns" → GPT-3 writes the problematic turns from Social Chemistry, ETHICS, and SBIC situations; 160,295 unique RoTs; 5.7 average turns, at most six ([[prosocial-dialog]], §3.2–3.4, App. A.1).
18. "CANARY-400M engages constructively with 89% of problematic prompts vs. BlenderBot-3B's 32%" and "precursor to Constitutional AI" → unsupported. Canary is a T5-large safety-label and RoT generator; the dialogue agent is Prost (2.7B) ([[prosocial-dialog]], §4, Verification).
19. OASST1 "10K fully-labeled trees, CC-BY 4.0", "too small (~250K messages)", "OASST2 adds ~90K" → 10,968 complete trees of 66,497; 161,443 messages; the paper states a fully permissive licence and the Hugging Face card lists Apache-2.0; OASST2 is not covered by the source ([[openassistant]], §4, Verification).
20. OASST "~3% typo rate", "power-law tail to 20+ turns", "near-zero sibling variance in synthetic corpora" → no source measures these. The OASST guidelines ask prompters to avoid typos, and the current collection settings cap tree depth at 5 ([[openassistant]], App. A, App. G).
21. "Persona-conditioning yields up to 2× unique vocabulary, 3× persona-consistent follow-ups, +5–10 IFEval; integrated into Tülu-3 and Qwen 2.5" → no primary source gives these numbers. Tülu 3 reports that removing its persona datasets lowers IFEval from 72.8 to 53.6 and changes the unseen IFEval-OOD from 17.6 to 18.0 ([[modern-mix-composition]]; [[tulu-3]] arXiv v5 Tables 10, 32).
22. Persona Hub "1B personas mined from web bios / about-me / LinkedIn" → personas are inferred from web text in RedPajama v2 (Text-to-Persona) and expanded by relationships (Persona-to-Persona), giving 1,015,863,523 after deduplication ([[persona-hub]], §2).
23. Capybara "~20K dialogues, 60%+ multi-turn; iterate 2–6 rounds; competitive MT-Bench on a single consumer GPU" → the released file has 16,006 rows, 66.7% with two or more exchanges; ~20,000 is the 7B V1.9 training set; the round rule and the GPU and MT-Bench claims are not in the cards ([[capybara]]).
24. SmolTalk "explicit multi-turn dialogue is ~30K of 1M — 3%" → MagPie-Ultra, the largest component (431k of about 1.1M), was generated as three-turn conversations, and Everyday-Conversations (2.38k) is multi-turn, so at least 39.4% of samples are multi-turn (derived). The component list also includes APIGen-Function-Calling 87.5k, LongAlign 3.73k, and Explore-Instruct-Rewriting 32k ([[modern-mix-composition]]; SmolLM2 §5.1.1, App. F Table 9).
25. "5–15% OASST/WildChat anchor slice" and "RoT-grounded engagement outperforms refusal on helpfulness and safety" → no source gives either. Tülu 3 8B SFT contains 100,000 WildChat prompts out of 939,344 (10.6%, derived), and ProsocialDialog warns that training on it alone can produce a "negativity-prone chatbot" ([[tulu-3]] Table 7; [[prosocial-dialog]] §9).

## Why this chapter matters for a general-purpose model

A general-purpose assistant is used in conversations. In LMSYS-Chat-1M the average conversation has 2.0 turns ([[lmsys-chat-1m]] Table 1), and in WildChat about 41% of conversations have more than one turn and 3.7% have more than 10 ([[wildchat]] §3). The multi-turn skills that matter are measurable: recalling an instruction from turn 1, applying a revision without losing earlier constraints, and recovering when the request is underspecified at first ([[mt-eval]] §1; [[lost-in-multi-turn]] §1). This chapter sits in the synthetic-data stage, before SFT (ch-29e, ch-30). It covers how multi-turn SFT data is produced, how its distribution is compared with real users, and which evaluations detect narrowing. Reasoning-trace data (ch-24) supplies single-response depth; this chapter supplies the conversation structure that tool-calling data (ch-26) and long-conversation data (ch-29b) extend.

## §1 Units: turns, exchanges, and what the statistics mean

**Definition.** A *message* (or utterance) is one speaker's text. An *exchange* (also called a round) is one user message followed by one assistant message. Papers use "turn" for either unit, and some do not state which.

**Problem.** Turn counts from different papers are not comparable unless the unit is known, so a synthetic corpus can appear longer or shorter than real traffic by a factor of 2.

**Worked example.** The SODA sample in Table 1 of [[soda-commonsense-grounding]] has 8 utterances (Madeleine and Coach, four each), which is 4 exchanges. SODA's own Table 2 reports 7.6 average turns, which counts utterances. UltraChat's Table 5 re-measures SODA at 3.6 turns, which is close to 7.6 / 2 = 3.8 and counts exchanges ([[ultrachat-pipeline]] Table 5). The two numbers describe the same data.

**Reference statistics.** The table collects numbers as printed; tokenizers and units are given where the source states them.

| Corpus | How the user side is produced | Conversations | Avg. turns (unit) | User / assistant length | Source |
|---|---|---|---|---|---|
| Baize v1 Quora / StackOverflow | same call as assistant | 54,456 / 57,046 | 3.9 / 3.6 | response 35.9 / 36.0 (unit not stated) | [[baize-construction]] Table 2 |
| UltraChat | separate user-model call | 1,468,352 | 3.8 (exchanges) | utterance 309.3 tokens | [[ultrachat-pipeline]] Table 5 |
| CAMEL AI Society | role-locked agent | 25,000 | not reported (cap 40 messages) | not reported | [[camel]] §4.1 |
| SODA | narrative-conditioned LLM | 1,486,896 | 7.6 (utterances) | utterance 16.1 (unit not stated) | [[soda-commonsense-grounding]] Table 2 |
| ProsocialDialog | GPT-3 problematic speaker, human feedback | 58,137 | 5.7 (utterances) | utterance 20.0 (unit not stated) | [[prosocial-dialog]] Table 1 |
| Capybara | not documented (Amplify-Instruct) | 16,006 | mean 2.58, median 3, max 17 (exchanges) | not reported | [[capybara]] |
| OpenAssistant | human volunteers | 46,283 (as measured by WildChat) | 2.34 | 33.41 / 211.76 Llama-2 tokens | [[wildchat-statistics]] Table 1 |
| ShareGPT | real users | 94,145 | 3.51 | 94.46 / 348.45 Llama-2 tokens | [[wildchat-statistics]] Table 1 |
| LMSYS-Chat-1M | real users, 25 models | 1,000,000 | 2.02 | 69.83 / 215.71 Llama-2 tokens | [[wildchat-statistics]] Table 1 |
| WildChat | real users, ChatGPT and GPT-4 | 1,039,785 | 2.54 | 295.58 ± 1609.18 / 441.34 Llama-2 tokens | [[wildchat-statistics]] Table 1 |

[figures/turn-distribution.html](figures/turn-distribution.html) plots these reported averages side by side and lets the reader switch between the turn-count view and the same-tokenizer user/assistant length view.

**Implication.** The real-user references are short on average and have a long tail of long prompts (WildChat user tokens have a mean of 295.58 and a standard deviation of 1609.18). The synthetic rows report 2.58–7.6 average turns in mixed units and give no user/assistant length split, so their match to real traffic in mean and spread cannot be judged until the unit and the tokenizer are matched.

## §2 Family 1: single-call self-chat (Baize)

**Definition.** Self-chat is a method in which one model call writes an entire transcript with both speaker roles around a seed topic ([[baize-construction]], arXiv 2023-04).

**Problem addressed.** In early 2023 open models lacked multi-turn training data; ShareGPT logs carried privacy and copyright concerns that the authors wanted to avoid (§3).

**Mechanism.**
1. Sample a seed question from Quora or Stack Overflow (about 55k each).
2. Insert it into the self-chat template and call `gpt-3.5-turbo` once.
3. The model writes `[Human]` and `[AI]` turns until the human "has no more question".
4. Train LLaMA with LoRA on the parsed dialogues, plus 51,942 single-turn Alpaca examples.

The template, verbatim (App. A):

```
Forget the instruction you have previously received. The following is a conversation between a human and an AI assistant. The human and the AI assistant take turns chatting about the topic: '${SEED}'. Human statements start with [Human] and AI assistant statements start with [AI]. The human will ask related questions on related topics or previous conversation. The human will stop the conversation when they have no more question. The AI assistant tries not to ask questions. Complete the transcript in exactly that format.
[Human] Hello!
[AI] Hi! How can I help you?
```

**Evidence.** 111.5k dialogues cost about $100 (§3). Single-call generation gave shorter AI turns (average response length 35.9 for Quora) than regenerating each AI turn with a separate call (149.6 for Quora v2), so the v1.5 data replaced the AI turns one call at a time (§3, Table 2). SDF scored four sampled responses per question with ChatGPT and trained on the best one; the GPT-4 score relative to ChatGPT rose from 87% to 90% at 7B and from 89% to 92% at 13B (§4, Fig. 3). Result (single study; one judge; 80 Vicuna prompts). On the Open LLM Leaderboard, Baize-v2-13B moved MMLU from 37.7 to 39.4 and TruthfulQA from 39.9 to 48.3 relative to LLaMA-13B, while HellaSwag fell from 78.9 to 77.1 (Table 7).

**Conditions and limits.** The user side is written by the same call that plans the answers, and the authors list "diversify simulated user queries" as future work (§7). No filtering, deduplication, or cross-turn dependency check is reported.

**Implication for a general-purpose model.** Self-chat gives multi-turn format at low cost, but its user turns come from the same distribution as the assistant turns, so it should not be the only source of user behaviour in an SFT mix (Interpretation).

## §3 Family 2: two-call dialogue under a topic taxonomy (UltraChat)

**Definition.** A two-call pipeline uses one model call to write the next user message and a separate call to write the assistant reply, each conditioned on the full history ([[ultrachat-pipeline]], arXiv 2023-05).

**Problem addressed.** The authors state that the diversity of a dialogue corpus depends on the diversity of opening lines and of user response style (§4), and that a user model given only the history "tends to assume the role of an AI assistant" (§4.4).

**Mechanism.** With history `h_t` after `t` exchanges, user prompt `p_U`, and assistant prompt `p_A`:

```
u_{t+1} = UserLLM(p_U, h_t)
a_{t+1} = AssistantLLM(p_A, h_t ∪ {u_{t+1}})
h_{t+1} = h_t ∪ {u_{t+1}, a_{t+1}}
```

`u` is a user message, `a` an assistant message; both calls were ChatGPT Turbo. Opening lines come from three sectors (arXiv v1 §4.1–4.3):
1. Questions about the World: ChatGPT lists 30 topics, 30–50 subtopics each, 10 questions per subtopic plus 10 more per question; a second branch uses the 10,000 most frequent Wikidata entities with 5 meta-questions, 10 specific and 20 extended questions each; about 500,000 questions are sampled as openers; Figure 1 shows 3–7 rounds.
2. Creation and Writing: instructions for 20 material types (Table 3), about 80% of which are expanded into more detailed instructions.
3. Assistance on Existing Materials: 10,000 C4 pieces, five instructions each, joined with seven templates such as `{text} Based on the passage above, {instruction}` (Table 4); the paper states that the concatenated set is 500,000 openers; 10,000 × 5 gives 50,000 text-instruction pairs, and the paper does not state how the template step yields 500,000 (unresolved inside the source).

The user prompt instructs the model "to adopt various user personalities", and a post-generation filter removes excessively polite user statements (§4.4). The repository README gives 1,100+ subtopics, ~100k C4 materials, and 2–4 rounds for Sectors 2–3 ([[ultrachat-two-model-protocol]]).

**Evidence.** UltraChat has 1,468,352 dialogues, 3.8 average exchanges, 1,467.4 tokens per dialogue, MTLD lexical diversity 74.3, and ChatGPT-rated coherence 9.06 (Table 5). UltraLLaMA (LLaMA-13B) was trained with loss on responses only, dialogues split to at most 2,048 tokens, and total batch 512 (§4.6). In a later controlled comparison at 1.7B (one epoch each), SFT on UltraChat gave IFEval 27.26 and MT-Bench 4.66, against 35.49 and 5.22 for the three-turn MagPie-Ultra set ([[modern-mix-composition]], SmolLM2 Table 10). Result (single study).

**Conditions and limits.** No ablation separates the effect of the two-call design from the taxonomy. The user model is still an assistant-tuned LLM with a persona instruction.

## §4 Family 3: role-playing agents with Inception Prompting (CAMEL)

**Definition.** Role-playing is a two-agent protocol in which an *AI user* gives one instruction at a time and an *AI assistant* returns a solution, under system prompts that fix roles, output format, and termination ([[camel]], arXiv 2023-03; this predates Baize and UltraChat).

**Problem addressed.** Two chat agents left to converse showed role flipping, repeated instructions, "flake" replies, and infinite loops (§4.1).

**Mechanism.** A task-specifier prompt makes a human-given task specific. Then, with message history `M_t = {(I_0,S_0), …, (I_t,S_t)}`, the user produces `I_{t+1} = U(M_t)`, the assistant produces `S_{t+1} = A(M_t, I_{t+1})`, and the pair is appended (§3.1 Eq. 1–4). `I` is an instruction and `S` a solution. The user system prompt, verbatim (Fig. 2):

```
Never forget you are a <USER_ROLE> and I am a <ASSISTANT_ROLE>.
Never flip roles! You will always instruct me.
We share a common interest in collaborating to successfully complete a task.
I must help you to complete the task.
Here is the task: <TASK>. Never forget our task!
You must instruct me based on my expertise and your needs to complete the task ONLY in the following two ways:
1. Instruct with a necessary input:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>
2. Instruct without any input:
Instruction: <YOUR_INSTRUCTION>
Input: None
...
You should instruct me not ask me questions.
...
When the task is completed, you must only reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my responses have solved your task.
```

The assistant prompt requires answers to start with `Solution:` and end with `Next request.` Conversations stop on any of five conditions: no instruction for 3 rounds, assistant gives an instruction, end-of-task token, token limit, or 40 messages (§4.1).

**Evidence.** AI Society has 50 assistant roles × 50 user roles × 10 tasks = 25,000 conversations (§4.1). Conversations end mostly by "Assistant Instruct" (role reversal) and rarely at the 40-message cap (App. J, Fig. 8). For 100 AI Society tasks, humans preferred the GPT-4-summarized role-play solution over single-shot gpt-3.5-turbo in 76.3% of votes (10.4% against) (Table 1). LLaMA-7B fine-tuned progressively on AI Society, Code, Math, and Science won all 120 GPT-4-judged questions against LLaMA-7B (Table 2), and CAMEL-7B reached HumanEval pass@1 14.0 against 10.5 for LLaMA-7B (Table 3). At 13B, fine-tuning lowered ARC-C (56.2 → 55.6) and HellaSwag (80.9 → 79.3) while raising MMLU and TruthfulQA (App. R, Table 7).

**Conditions and limits.** The end-of-task token records the user agent's judgment only; the paper does not check task completion, cross-turn dependency, or turn-length distributions. The "instruct, not ask" rule excludes user questions and clarifications by design, so the user side covers one speech act (Interpretation).

**Implication.** CAMEL data trains instruction-solution exchanges across domains; it does not model underspecified users or users who ask questions; §9 reports that requests revealed over several turns lower average scores by 39%.

## §5 Grounded dialogue: ProsocialDialog (2022-05) and SODA (2022-12)

**Definition.** Grounded dialogue synthesis inserts externally sourced content (a social situation, a commonsense triple, a document) into the generation prompt so that the dialogue is about that content. This course calls such content an *information anchor*, as opposed to a persona or format instruction that changes only style (Interpretation; developed in qa.md Q9).

**ProsocialDialog.** [[prosocial-dialog]] addresses dialogue agents that ignore or agree with unsafe user utterances (Abstract). Mechanism (§3.2, App. A): (1) take situations from Social Chemistry (36k), ETHICS (9.7k), and SBIC (12k); (2) few-shot GPT-3 turns a situation into a problematic first-person utterance; (3) a crowdworker selects or writes rules-of-thumb (RoTs) and writes feedback grounded in them; (4) GPT-3 continues the problematic side and workers respond again, for at most six turns; (5) three workers label each GPT-3 utterance for safety. The result has 58,137 dialogues, 331,362 utterances, 160,295 unique RoTs, and 497,043 safety annotations (§3.4). Canary (T5-large) generates a safety label and RoTs; Prost (2.7B) generates an RoT and a response. Against GPT-3 on 400 test examples, humans preferred Prost (RoT & Response) on prosociality 63.4% vs 9.3% (Table 4). The authors warn that training on ProsocialDialog alone "can result in a negativity-prone chatbot" and mix it 9:3:3:3:3:3:1 with six casual-dialogue datasets for Prost (§9, App. B.2).

**SODA (CO3 framework).** [[soda-commonsense-grounding]] addresses the finding that sampling dialogues "without contexts results in dull conversations" (§1). Mechanism:
1. Retrieve social x-relation triples from Atomic10x, e.g. `Head: PersonX moves a step closer to the goal; Relation: xNeed; Tail: to take the first step` (§2.2).
2. Convert to a sentence with a relation template, e.g. xNeed → `PersonX [Tail in past tense]. [Head].`, and replace PersonX with a common US name (§2.3, Table 8).
3. Prompt text-davinci-002: `[sentence-form commonsense] Rewrite this story with more specific details in two or three sentences:` (App. A.1).
4. Infer the second speaker when the triple has one person, then generate the conversation with the first speaker as a prefix (§2.4).
5. Filter: erroneous patterns (6.3%), fewer than 4 or more than 20 turns (5.7%), more than two speakers (11.3%), non-human speakers (5.6%), Canary "needs intervention" (4.3%), Rewire API toxicity above 0.5 (about 1%), and a GPT-3.5 check that the head event is implied (§3.1).

**Worked example (filter yield).** SODA states that 68.9% of about 2.2 million generated conversations remain, giving 1,486,896. With exactly 2,200,000 the yield would be 1,486,896 / 2,200,000 = 67.6%; the stated 68.9% implies about 1,486,896 / 0.689 ≈ 2,158,000 initial conversations. The "2.2 million" is rounded (derived).

**Evidence.** In head-to-head human evaluation, SODA dialogues were preferred over 100 dialogues sampled from GPT-3.5 with the prompt "The following is a long in-depth conversation between two people." on specificity, interestingness, and other criteria (all significant except natural flow; §3.3, Fig. 3). COSMO-3B, trained on SODA plus ProsocialDialog with the narrative dropped 30% and the role instruction 50% of the time, was preferred over BlenderBot-3B overall in 72% of comparisons on the unseen DailyDialog set and over the human-written DailyDialog responses in 55% (§4, §5.1 Table 5). Result (single study; 100 test examples; three judges).

**Conditions and limits.** Both datasets model social chit-chat or safety feedback, not information-seeking assistant use. SODA's grounding source covers social situations only; the paper leaves physical and event-centered relations to future work (§2.2 fn. 3).

## §6 Human-written reference: OpenAssistant conversation trees

**Definition.** A conversation tree stores a root prompt, several assistant replies per prompt, several user follow-ups per reply, and ranks among siblings; a thread is any root-to-node path ([[openassistant]] §2, arXiv 2023-04).

**Evidence.** OASST1 has 161,443 messages (91,829 prompter, 69,614 assistant) in 66,497 trees, 10,968 of them complete, from over 13,500 volunteers in 35 languages, with 461,292 quality ratings (§4). 8,576 messages are synthetic (§4). Current collection settings cap tree depth at 5 and children at 2, with a goal tree size of 9 messages (App. G, Table 5). WildChat's re-measurement gives 2.34 turns, 33.41 user tokens and 211.76 assistant tokens per conversation (Llama-2 tokenizer) ([[wildchat-statistics]] Table 1). SFT on OASST1 raised the lm-eval-harness average for Falcon-40B from 72.29 to 74.04 but not for Pythia-12B (60.33 → 60.28) (Table 1).

**Conditions and limits.** OASST is written by volunteers following guidelines that ask prompters to avoid typos and unannounced topic changes (App. A), and 89.1% of annotators identify as male (§7). It is a reference for human-written assistant dialogue, not for unconstrained user behaviour.

**Implication.** The sibling structure provides ranked alternatives at the same context, which is preference data rather than SFT data (Interpretation; the paper trains reward models on these rankings, App. H). The number of preference pairs it yields is not reported.

## §7 Conditioning axes: persona, scenario, system prompt

**Definition.** A *conditioning axis* is a variable sampled per dialogue and inserted into the generation prompt: a persona, a scenario or intent, a system prompt, a topic, or a document.

**Mechanism and evidence.**
- **Persona.** [[persona-hub]] builds 1,015,863,523 personas by asking an LLM who is likely to read or write a RedPajama v2 web text, then expanding by relationships and deduplicating at cosine similarity 0.9 (§2). Generated problems are less similar to each other than their personas are, and adding a topic constraint raises their similarity (§4.1.2, Fig. 10). Tülu 3's persona datasets include Persona IF with 29,980 prompts (Table 7). Removing all persona datasets from Tülu 3 8B SFT lowered IFEval from 72.8 to 53.6, while the unseen IFEval-OOD went from 17.6 to 18.0 ([[modern-mix-composition]]; [[tulu-3]] Tables 10, 32). Result (single study): the gain is on the development benchmark, and the held-out constraint benchmark did not move.
- **System prompt.** MagPie-Ultra adds system prompts to Magpie-style generation with Llama-3.1-405B-Instruct-FP8 and produces three-turn conversations, filtered with Llama-3.1-8B-Instruct, Llama-Guard-3-8B, ArmoRM scores, and gte-large embedding deduplication ([[modern-mix-composition]], SmolLM2 §5.1.1). SmolTalk adds SystemChats2.0 samples "to make the model support a variety of system prompt formats" (30k in the SmolTalk dataset card; 35.9k in SmolLM2 Table 9). No ablation isolates the system-prompt variable.
- **Scenario and intent.** No source in this library measures scenario-first generation for user simulators. The two-stage design from qa.md Q13–Q14 (sample and filter scenarios, then generate) is a course proposal (Interpretation); its evaluation belongs to ch-29d.

**Measured formulaic user turns.** In Capybara's rows with at least two exchanges, the second user turn starts with "Considering" in 1,787 of 10,681 rows (16.7%) and with "Given" in 1,090 (10.2%) ([[capybara]], derived). A count of the first words of user turns requires no model calls and is a check for a user generator that repeats one template.

**Implication.** A persona or system prompt changes what is generated, but the Tülu 3 result shows that a 19.2-point gain on the benchmark whose constraints resemble the training data can coexist with a 0.4-point change on unseen constraints. Report both.

## §8 Generality target: coverage of the real-user distribution

**Definition.** For conversation data, *coverage* means that the prompts, intents, languages, and turn structures of real users are represented in training. It is measured against a real-traffic reference such as [[wildchat]] or [[lmsys-chat-1m]].

**Reference corpora.** LMSYS-Chat-1M has 1,000,000 conversations from 210,479 users in 154 languages with 25 models, collected April–August 2023; the authors report that the majority of questions in 100K sampled English conversations relate to coding and software (clusters 1, 2, 6, 16, and 18 of 20) (§2, §3.1–3.2, Table 1). WildChat has 1,039,785 conversations from 204,736 IP addresses in 68 languages; English is 53% of turns; 10.46% of user turns are flagged as toxic by Detoxify or the OpenAI Moderation API (§2–4) ([[wildchat-statistics]]).

**Measurement 1: cross-dataset NLL.** WildChat fine-tunes Llama-2 7B on the first-turn prompts of dataset A and reports the average negative log-likelihood (NLL) on dataset B, `NLL(B | θ_A) = −(1/N) Σ log p_θA(x_i)`, where `θ_A` are the weights trained on A, `x_i` are tokens of B's prompts, and N is the token count (§3, Fig. 3; per-token averaging assumed from "average NLLs"). Worked example: the Alpaca-trained model gives 10.87 on WildChat prompts, the WildChat-trained model gives 5.18 on the same prompts and 3.28 on OpenAssistant prompts, against 3.09 for the OpenAssistant-trained model (Fig. 3). The WildChat-trained model is within 1.04 NLL of each other dataset's own model on that dataset (Alpaca 3.28 vs 2.24, Dolly 3.29 vs 2.74, OpenAssistant 3.28 vs 3.09, ShareGPT 5.91 vs 5.23), while the Alpaca-trained model scores 8.42, 11.11, and 10.87 on OpenAssistant, ShareGPT, and WildChat prompts.

**Measurement 2: embedding overlap.** WildChat embeds 10,000 first-turn prompts per dataset with text-embedding-ada-002 and shows by t-SNE that WildChat overlaps the other datasets and covers additional areas (§3, Fig. 4). For a synthetic corpus, the same procedure applies with synthetic user turns in place of dataset A (course proposal).

**Measurement 3: held-out suites drawn from real traffic.** Arena-Hard-200 is 200 Chatbot Arena prompts that GPT-3.5-Turbo, Claude-2, and GPT-4 all scored 9 or higher for benchmarking potential; on 50 prompts scored above 8 by GPT-3.5-Turbo, GPT-4 beat GPT-3.5-Turbo in 52% of user votes, and on 50 prompts scored below 2 in 22% (§4.4, Fig. 5, which the prose calls Table 5) ([[lmsys-chat-1m]]). WildChat reserves 14,743 conversations for WildBench (§2); WildBench's construction is not covered by a card in this library.

**Evidence that realistic style is not capability.** Crowdworkers rated about 70% of imitation-model outputs as equal or better than ChatGPT, yet broad imitation data (ShareGPT-Mix) lowered Natural Questions accuracy from 20 to 15 at 13B while targeted NQ-synthetic data raised it to 27 ([[false-promise-imitating-proprietary-llms]] Fig. 1, Table 1). As imitation data grew from 20M to 150M tokens, the share of outputs that use a list when ChatGPT does rose from 50% to 81% (Table 2). Result (single study; LLaMA 7B/13B).

**Evidence that real-user prompts help broad chat.** Removing 100,000 WildChat prompts from Tülu 3 8B SFT lowered the average from 60.1 to 58.9 and AlpacaEval 2 from 12.4 to 7.5, raised safety from 93.1 to 95.2, and raised IFEval-OOD from 17.6 to 20.8 ([[tulu-3]] Tables 10, 32). Result (single study): including real-user prompts raised the chat metric by 4.9 points and the average by 1.2, and lowered safety by 2.1 and IFEval-OOD by 3.2.

## §9 Multi-turn degradation: what current models fail at

**Lost in Conversation.** [[lost-in-multi-turn]] (arXiv 2025-05) takes fully specified instructions from six tasks (code, database, actions, math, data-to-text, summary), splits each into *shards* (the first states the intent, each later one adds one requirement), and has a GPT-4o-mini user simulator reveal at most one shard per turn (§3.1–3.2). Settings: FULL (original single turn), CONCAT (all shards in one turn), SHARDED (one shard per turn). For N = 10 simulations with scores `S = {S_i}` in 0–100:

```
P = (1/N) Σ S_i          A = percentile_90(S)          U = percentile_90(S) − percentile_10(S)
```

`P` is average performance, `A` aptitude (best-case score), and `U` unreliability (spread between good and bad runs) (§4.2).

**Worked example.** With nearest-rank percentiles over 10 sorted scores (rank ⌈0.9·10⌉ = 9 for the 90th and ⌈0.1·10⌉ = 1 for the 10th; the paper does not state its interpolation), FULL runs `[70,80,80,80,90,90,90,90,100,100]` give P = 87, A = 100, U = 100 − 70 = 30. SHARDED runs `[0,10,30,40,60,80,90,90,100,100]` give P = 60, A = 100, U = 100 − 0 = 100. Average performance drops 31% while aptitude is unchanged; the loss is in reliability. [figures/aptitude-unreliability.html](figures/aptitude-unreliability.html) lets the reader edit the ten scores and see P, A, and U recomputed.

**Evidence.** Over 15 LLMs and more than 200,000 simulated conversations, SHARDED averaged 39% below FULL; CONCAT averaged 95.1% of FULL, so the drop is not caused by rephrasing (§5–6). From FULL to SHARDED, aptitude fell 16% on average and unreliability rose 112% (§6.2). Adding a final recap turn or repeating all shards each turn (SNOWBALL) recovered part of the loss; SNOWBALL mitigated the FULL-to-SHARDED deterioration by 15–20% (§7.1). For GPT-4o-mini and GPT-4o, setting user and assistant temperature to 0 still left SHARDED unreliability of about 30 points (§7.2, Table 3). The authors attribute the failures to early assumptions and premature answer attempts (Abstract). Result (single study; simulated users).

**MT-Eval.** [[mt-eval]] (arXiv 2024-01) defines four patterns from LMSYS-Chat-1M: recollection, expansion, refinement, follow-up; it has 168 dialogues, 1,170 turns, 6.96 turns per dialogue, and a single-turn version of each query except follow-up (§1, §3, Table 1). Worked example from Table 3: Llama-2-chat-13B scores 7.55 single-turn and 5.47 multi-turn (−2.08, a 27.5% relative drop), while GPT-4 scores 9.17 and 8.84 (−0.33, 3.6%). Llama-2-chat models beat Vicuna single-turn but trail it multi-turn (§4.4). In a manual analysis of 200 responses, 99 (49.5%) did not comply with earlier instructions and 96 (48%) were misdirected by errors in earlier turns; conditioning on gold history improved recollection and refinement (§4.5–4.6, Table 7).

**Controlled data evidence.** At 1.7B, SFT on the multi-turn MagPie-Pro-MT gave MT-Bench 5.40 and IFEval 31.66 against 4.31 and 30.45 for single-turn MagPie-Pro, with MMLU-Pro 11.97 vs 12.19 ([[modern-mix-composition]], SmolLM2 Table 10). Result (single study; one epoch; MT-Bench has two turns).

**Implication for data.** The pipelines in §2–§5 open each dialogue from a seed question, a task, or a narrative and add related follow-ups (Interpretation of the prompts quoted above). None of those pipelines produces underspecified, sharded requests, and none is evaluated on the SHARDED setting. Whether sharded-instruction SFT data reduces unreliability is an Open question for this chapter; ch-29b covers training data built from sharded and multi-turn instruction-following sources.

## Negative samples and negative feedback

Four senses of "negative" apply.
- **Negative marginal value (discarded).** SODA discards 31.1% of generated conversations by pattern, length, speaker, safety, and grounding filters (§3.1). UltraChat removes polite user statements (§4.4). OASST deletes spam-flagged and moderator-removed messages together with their children (§3.4). Baize SDF scores four candidates and keeps only the best; the other three are unused (§4).
- **Negative as content.** ProsocialDialog places GPT-3's problematic utterances in the context and trains Prost with cross-entropy only on the prosocial responses; the authors call training on the problematic side misuse (§4.2, §9). Canary's safety labels (for example Needs Intervention) are also trained as ordinary target text.
- **Negative as conditioning.** Not used by the sources in this chapter.
- **Negative as gradient.** Not used by the SFT pipelines here. OASST sibling rankings feed a pairwise reward-model loss `−E[log σ(r(x,y_w) − r(x,y_l))]`, where `r` is the reward model, `x` the context, `y_w` and `y_l` the higher- and lower-ranked replies (App. H); reward modeling is covered in ch-41 (Reward Modeling: Bradley–Terry, Over-Optimization, and Reward-Model Generalization).

**Diagnostics and effect on generality.** For safety-grounded dialogue data, measure over-refusal alongside harm: in [[xstest]], a guardrail system prompt raised Mistral-7B's full refusal of unsafe prompts from 23.5% to 87.5% and of safe prompts from 0.8% to 9.6% (Table 2). Tülu 3 reports that contrastive prompts such as CoCoNot helped prevent over-refusal of safe prompts ([[tulu-3]] §4.2). For discard filters, log the rejection share per filter as SODA does, so that a filter removing a whole user style (for example short or rude messages) is visible.

## Recipe

| Model (exact release) | Size | Stage | Setting | Value | Source location | Status | Evidence for this value |
|---|---|---|---|---|---|---|---|
| Baize v1 | 7B, 13B, 30B | distill-SFT | data; max input length | 111.5k self-chat (Quora 54,456; StackOverflow 57,046) + Alpaca 51,942; 512 tokens | arXiv:2304.01196v4 §3, §5, Table 2 ([[baize-construction-recipe]]) | verified 2026-09-14 | no ablation reported |
| Baize v1.5 | 7B, 13B | distill-SFT | AI-turn generation; loss masking | AI turns regenerated one call at a time; loss on AI responses only | arXiv:2304.01196v4 §3, §4 | verified 2026-09-14 | no ablation against v1 data reported |
| Baize v2 | 7B, 13B | SFT (SDF) | candidates; scorer; target | 4 per Quora question; ChatGPT score 1–100; best-ranked | arXiv:2304.01196v4 §4, App. C | verified 2026-09-14 | Fig. 3: 87% → 90% (7B), 89% → 92% (13B) |
| UltraLLaMA | 13B | distill-SFT | data; sequence length; loss; batch | UltraChat 1,468,352 dialogues; split to ≤2,048 tokens; responses only; total batch 512 on 128 A100 | arXiv:2305.14233v1 §4.6, Table 5 ([[ultrachat-pipeline]]) | verified 2026-09-15 | no ablation reported |
| CAMEL-7B | 7B | distill-SFT | epochs; LR; schedule | 3; 2e-5; cosine, warmup ratio 0.04 | arXiv:2303.17760v2 App. K.2 Table 5 ([[camel]]) | verified 2026-09-14 | no ablation reported |
| COSMO-3B / 11B | 3B, 11B | SFT | data; input dropout | SODA + ProsocialDialog; narrative dropped 30%, role instruction 50% | arXiv:2212.10465v3 §4 ([[soda-commonsense-grounding]]) | verified 2026-09-15 | no ablation reported; LR and steps not reported (checked body and appendix) |
| Prost | 2.7B | SFT | mixture weights | ProsocialDialog : DailyDialog : TopicalChat : PersonaChat : WoW : EmpatheticDialogues : BST = 9:3:3:3:3:3:1 | arXiv:2205.12688v2 App. B.2 ([[prosocial-dialog]]) | verified 2026-09-14 | §9: training on ProsocialDialog alone "can result in a negativity-prone chatbot"; no mixture ablation |
| WildLlama | 7B | SFT | data; batch; LR; length; epochs | WildChat to 2023-07-16; 128 conversations; 2e-5; 2,048 tokens; 3 | arXiv:2405.01470v1 §5 ([[wildchat-statistics]]) | verified 2026-09-15 | Table 9: MT-Bench average 6.35 vs Vicuna 6.13, Llama-2 Chat 6.26 |
| Tülu 3 8B SFT | 8B | SFT | WildChat share; Persona IF | 100,000 of 939,344 prompts (10.6%, derived); 29,980 | arXiv:2411.15124v5 Table 7 ([[tulu-3]]) | verified 2026-09-15 | Table 10: w/o WildChat AE2 12.4 → 7.5; w/o Persona IFEval 72.8 → 53.6 |
| Tülu 3 8B SFT | 8B | SFT | LR; batch; length; epochs | 5e-6 linear, warmup 0.03; 128; 4,096; 2 | arXiv:2411.15124v5 §4.3, Table 11 | verified 2026-09-15 | "found after a hyperparameter search" (§4.3); table not shown here |
| SmolLM2-1.7B SFT | 1.7B | SFT | data; epochs | SmolTalk ~1.1M samples (MagPie-Ultra 431k, three-turn); 2 | arXiv:2502.02737 §5.1.1, App. F Tables 9–10 ([[modern-mix-composition]]) | verified 2026-09-15 | Table 10: SmolLM2-SFT IFEval 57.09, MT-Bench 6.11 |
| SmolTalk components | — | SFT | MagPie-Ultra size | 400K (dataset card) vs 431k (paper Table 9) | HF HuggingFaceTB/smoltalk card; arXiv:2502.02737 Table 9 ([[smol-talk]]) | conflict | paper table describes the trained mix; prefer 431k |
| Nous-Capybara-34B V1.9 | 34B | SFT | base; epochs | Yi-34B-200K; 3 on Capybara | HF NousResearch/Nous-Capybara-34B@6beb706 README ([[capybara]]) | verified 2026-09-14 | no ablation reported |

**Starting point for a small general-purpose run.** For a 1.7B model, the SmolLM2 recipe trains 2 epochs on about 1.1M samples in which the largest component is three-turn system-prompted conversations from a 405B generator (SmolLM2 App. F); this was run on SmolLM2's own base model. For an 8B model, Tülu 3 trains 2 epochs at learning rate 5e-6, effective batch 128, and 4,096 tokens on 939,344 prompts that include 100,000 real WildChat prompts (Tülu 3 §4.3, Table 7). Neither source ablates the share of multi-turn data.

## Long-context scope: from short dialogues to long conversations

Every corpus in this chapter is short: synthetic averages range from 2.58 to 7.6 turns (units in §1), SODA filters out conversations above 20 turns, CAMEL caps at 40 messages, OASST caps tree depth at 5, and UltraChat splits dialogues at 2,048 tokens for training. Real traffic has a tail: 3.7% of WildChat conversations exceed 10 turns (§3). MT-Eval identifies distance to relevant content and error propagation as the factors behind multi-turn degradation (§9); in its recollection task all models except GPT-4 have more difficulty following the initial global instruction as the conversation grows, and in its refinement task 9 of 11 models score higher on the first six turns than on the last six (§4.4, Table 4).

The pipelines here do not produce: sessions separated in time with summaries or memory; injected facts that later turns must retrieve, update, or abstain on; same-distribution distractors; long system prompts with many guardrails; or accumulating tool output. Those are the subject of ch-29b (Long-Conversation and Accumulating-Context Synthesis), which builds on Conversation Chronicles and LongMemEval. The anchor and scenario-first frameworks from qa.md Q9, Q13, and Q14 carry over there as design inputs. The ranking in qa.md Q18 of length coverage as a lower priority holds only for short chat.

## Generalization lens

**(a) What increases breadth.**
- Real-user prompts: removing WildChat from Tülu 3 SFT lowered the average and AlpacaEval 2 ([[tulu-3]] Table 10); a WildChat-trained model stayed within 1.04 NLL of each in-domain model on four other corpora, while an Alpaca-trained model reached 8.42–11.11 ([[wildchat-statistics]] Fig. 3).
- External grounding: SODA's contextualized dialogues beat context-free sampling on specificity and interestingness in human judgments ([[soda-commonsense-grounding]] §3.3), and COSMO generalized to unseen DailyDialog (Table 5).
- Multi-turn format itself: MagPie-Pro-MT improved MT-Bench over MagPie-Pro at 1.7B ([[modern-mix-composition]], SmolLM2 Table 10).
- Targeted data for a skill: NQ-synthetic imitation improved NQ where broad imitation did not ([[false-promise-imitating-proprietary-llms]] Table 1).

**(b) What causes narrowing or forgetting.**
- Style without capability: broad ChatGPT imitation raised list use and authoritative tone and lowered NQ ([[false-promise-imitating-proprietary-llms]] Tables 1–2).
- Single-style safety data: training on ProsocialDialog alone is described as producing a negativity-prone chatbot ([[prosocial-dialog]] §9).
- Dialogue SFT on broad benchmarks: CAMEL 13B lost ARC-C and HellaSwag points ([[camel]] App. R); Baize-v2-13B lost HellaSwag points ([[baize-construction]] Table 7).
- Development-benchmark fit: persona IF data moved IFEval by 19.2 points and IFEval-OOD by 0.4 ([[tulu-3]] Table 32).

**(c) How to measure it at this stage.**
- Distribution: units-matched turn counts and same-tokenizer lengths against WildChat or LMSYS-Chat-1M (§1); cross-dataset NLL and embedding overlap (§8); first-word counts of user turns (§7).
- Held-out multi-turn behaviour: MT-Eval single-turn vs multi-turn gap; FULL vs CONCAT vs SHARDED with P, A, U (§9); Arena-Hard-style prompts selected from real traffic (§8).
- Retention: a broad benchmark set (MMLU, ARC, HellaSwag, TruthfulQA) before and after dialogue SFT, and an unseen-constraint suite next to IFEval.
- Known measurement errors: judges prefer longer responses and first positions ([[baize-construction]] Limitations); crowd raters are misled by confident style ([[false-promise-imitating-proprietary-llms]] §4.4); MT-Bench has two turns; public chat logs such as ShareGPT, WildChat, and LMSYS Chat are likely to overlap with benchmark test sets ([[tulu-3]] App. B.2).

## Common mistakes and how to detect them

| Mistake | Observable symptom | Check |
|---|---|---|
| Comparing turn counts with different units | Synthetic corpus appears 2× longer than real traffic | Recount one sample by utterances and by exchanges (SODA 7.6 vs 3.6, §1) |
| User turns written by the assistant call | User turns ask for exactly what the next answer covers; user length close to assistant length | Same-tokenizer user/assistant length ratio against WildChat (295.58 / 441.34) |
| Role exchange in the user model | User turns contain answers or "How can I help?" | Classifier or regex on user turns; UltraChat §4.4 reports this failure |
| Template-repeated follow-ups | One opening word dominates second user turns | First-word histogram (Capybara: "Considering" 16.7%) |
| Taking the end-of-task token as success | Dialogues marked done with unsolved tasks | Separate completion check; CAMEL does not verify (§4.1) |
| Judging data by crowd or judge preference only | High preference, flat or lower NQ/MMLU | Held-out factual and broad benchmarks before/after SFT |
| Only fully specified first messages | Model answers immediately on underspecified requests | FULL vs SHARDED evaluation with U (§9) |
| Safety dialogue data without over-refusal eval | Refusals on safe prompts that share vocabulary with unsafe ones | XSTest safe-prompt refusal rate |
| Reporting only the benchmark the data targets | IFEval up, unseen constraint suite flat | Pair IFEval with IFEval-OOD or IFBench |
| Citing composition from a dataset card | Numbers differ from the trained mix | Compare card with the paper table (SmolTalk 400K vs 431k) |

## Check your understanding

1. SODA reports 7.6 turns per dialogue and UltraChat reports 3.6 for SODA. Explain why both can be correct and what error follows if a synthetic corpus is compared with WildChat without resolving it.
2. Baize v1.5 regenerated AI turns one call at a time. What change in Table 2 motivated this, and why does a single call that writes both sides tend to produce that pattern?
3. CAMEL AI Society conversations end mostly by role reversal rather than by `<CAMEL_TASK_DONE>`. What does this imply about using the end token as a success label for SFT data selection?
4. In the Tülu 3 ablation, removing persona data lowered IFEval by 19.2 points and did not lower IFEval-OOD. Give two causal explanations and the experiment that would separate them.
5. Lost in Conversation finds that aptitude falls 16% but unreliability rises 112%. Why would more SFT data with fully specified first messages be unlikely to fix this, and what data property might?
6. A WildChat-trained model has NLL 5.18 on WildChat prompts while an Alpaca-trained model has 10.87. Explain what this measures and one reason it could overstate the usefulness of WildChat for capability.
7. The False Promise paper finds that imitation data changes style faster than accuracy. Which measurement in this chapter's lens would detect the same failure in a synthetic multi-turn corpus, and why would a GPT-4 judge alone not detect it?
8. ProsocialDialog trains only on the prosocial speaker's turns. Why is the problematic speaker's text used as context rather than as a target, and what failure would the other choice cause?

## Connections

- Previous: ch-24 — Reasoning-Trace Synthesis: Chain-of-Thought, Long Chain-of-Thought, and Step-Level Data.
- Next: ch-26 — Tool and Function-Calling Data (tool calls add typed arguments and executable results to the multi-turn structure here).
- ch-18 — The Synthetic-Data Design Pattern: Generate, Filter, Deduplicate, Verify, Select, Mix (the generic loop that §2–§5 instantiate).
- ch-19 — Generation Methods: Bootstrap, Evolution, Extraction, Persona, and Rephrasing (persona conditioning in §7).
- ch-22 — Quality, Diversity, and Gradient-Based Data Selection; ch-23 — Model Collapse and Verification of Synthetic Data (filtering and real-data anchoring in §5, §8).
- ch-29b — Long-Conversation and Accumulating-Context Synthesis (long and multi-session conversations beyond this chapter's scope).
- ch-29d — User Simulators, Trajectory Verification, and Failed Trajectories (user-simulator realism, including the customer-simulator question from qa.md Q12).
- ch-29e — Instruction Tuning and Generalization to Unseen Tasks; ch-30 — SFT Design Choices and Their Effect on Generalization: Masking, Packing, Templates, Epochs, and Learning Rate (how these dialogues are trained).
- ch-52 — Safety Evaluation, Over-Refusal, and Red-Teaming (over-refusal measurement for §5 data).

## Sources

- [[baize-construction]] — self-chat template, corpus statistics, SDF, GPT-4 scores, leaderboard changes. [[baize-construction-recipe]] — Baize training settings.
- [[ultrachat-pipeline]] — two-call design, sector construction, role-exchange observation, Table 5 statistics, UltraLLaMA training. [[ultrachat-two-model-protocol]] — excerpt reconciling paper and README numbers.
- [[camel]] — Inception Prompt, termination conditions, AI Society size, agent and transfer evaluations, CAMEL-7B recipe.
- [[prosocial-dialog]] — collection pipeline, statistics, Canary and Prost, human evaluation, mixture warning.
- [[soda]] — card for SODA (unverified card; numbers in this chapter come from the excerpt). [[soda-commonsense-grounding]] — CO3 steps, filters and rates, contextualization ablation, COSMO evaluation.
- [[openassistant]] — tree format, composition, collection settings, guidelines, SFT results.
- [[persona-hub]] — persona construction and similarity findings.
- [[capybara]] — released-file turn statistics and follow-up phrasing counts.
- [[smol-talk]] — SmolTalk card (composition conflicts noted). [[modern-mix-composition]] — excerpt of SmolLM2 Tables 9–10 and Tülu 3 Tables 10, 32.
- [[tulu-3]] — WildChat and persona ablations, SFT hyperparameters, CoCoNot note.
- [[wildchat]] — card for WildChat. [[wildchat-statistics]] — excerpt of Table 1, turn distribution, coverage NLL, embedding analysis, WildLlama.
- [[lmsys-chat-1m]] — excerpt of statistics, topic clusters, Arena-Hard-200.
- [[lost-in-multi-turn]] — excerpt of sharded simulation, P/A/U metrics, results, recap and temperature experiments.
- [[mt-eval]] — excerpt of the four interaction patterns, single- vs multi-turn scores, error analysis.
- [[false-promise-imitating-proprietary-llms]] — excerpt of crowd ratings, NQ results, style convergence.
- [[xstest]] — over-refusal measurement for safety dialogue data.
- [[agentinstruct]] — suggester-editor refinement (correction 12).
