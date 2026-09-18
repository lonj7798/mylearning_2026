<!-- scope: MultiChallenge (Scale AI, 2025): 273-conversation multi-turn benchmark over instruction retention, inference memory, versioned editing, and self-coherence; multi-agent generation with human editing; LLM judge with per-item binary rubrics
     deps: [[judge-llm-bias]], [[persona-hub]]
     see-also: [[mt-eval]], [[multi-if]], [[lost-in-multi-turn]], [[parrot-multi-turn]], [[prefeval]], [[sycophancy-in-lms]], [[rubrics-as-rewards]]
-->

# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs
- **Core Insight:** Under human grading, all six frontier models tested score below 50% average accuracy on 273 multi-turn conversations, with Claude 3.5 Sonnet (June 2024) highest at 41.42% and GPT-4o (August 2024) at 12.52% (§5.1, Table 2); an LLM judge given a per-item yes/no rubric question agrees with human raters on 93.95% of judgments versus 37.33% for a judge given the full conversation and category criteria (§5.2, Table 4).
- **Guideline:** When multi-turn responses have no single ground-truth answer and must be graded by an LLM judge, attach to each item a binary rubric question that can be answered from the final response alone, because this raised judge-human alignment above 90% in every category (Table 4); leave out items whose rubric question is beyond current judge capability, as the authors did (Limitations).
- **Authors:** Ved Sirdeshmukh, Kaustubh Deshpande, Johannes Mols, Lifeng Jin, Ed-Yeremai Cardona, Dean Lee, et al. (Scale AI)
- **Year:** 2025 (arXiv v1 2025-01; v2 2025-03)
- **URL:** https://arxiv.org/abs/2501.17399
- **Source type:** paper (benchmark)
- **Relevant topics:** multi-turn evaluation, instruction retention, conversational memory, versioned editing, sycophancy and self-coherence, LLM-as-judge with instance-level rubrics, multi-agent synthetic data with human editing

## Abstract
The paper introduces MultiChallenge, a benchmark for how LLMs conduct multi-turn conversations with human users. It defines four challenge categories that are common in real human-LLM interaction and difficult for current frontier models. Each category requires instruction following, allocation of attention over the conversation context, and in-context reasoning at the same time. The authors also build an automatic evaluation that uses an LLM judge with instance-level rubrics and reports its agreement with experienced human raters. Frontier models that score near the maximum on earlier multi-turn benchmarks such as MT-Bench all score below 50% here, and the best model, Claude 3.5 Sonnet (June 2024), reaches 41.4% average accuracy (Abstract).

## Key Contributions
- Four categories: instruction retention, inference memory of user information, reliable versioned editing, and self-coherence (§3.1).
- MMSE, a multi-agent generator (planner, user, responder agents) that produces candidate conversations in which a responder fails, followed by human editing and two review layers (§4.1-4.2).
- Instance-level binary rubric questions that make LLM-judge evaluation track human grading (§3.2, §5.2).
- Human and automatic evaluation of six frontier models, automatic evaluation of seven open-weight models, and analyses of turn count, model size, and human editing cost (§5).

## Key Figures/Tables to Study
- **Figs. 1-2:** one example per category, with the constraint or earlier information highlighted.
- **Fig. 3 and App. A.3 (Figs. 6-7):** MMSE workflow and the planner and user agent system prompts.
- **Table 1:** benchmark statistics. **Tables 2-3:** human-graded and auto-graded accuracy by category.
- **Table 4:** judge-human alignment, baseline judge vs rubric judge. **Table 5:** open-weight models.
- **Fig. 4:** accuracy versus number of turns. **App. A.8:** per-model failure case studies.

## Technical Details
- **Item format:** a conversation history of at most 10 turns that ends with a final user turn; the model under test writes the next response (§1).
- **Size (Table 1):** 273 conversations: Inference Memory 113, Instruction Retention 69, Reliable Version Editing 41, Self-Coherence 50; average 5 turns and 1,231.7 words per conversation.
- **Category rules:** instruction-retention items state in turn 1 that the instruction applies to the whole conversation, and later user turns never contradict it (§3.1.1). Inference-memory final turns do not ask for the earlier information directly; the information is only implicitly required (§3.1.2). Versioned-editing items require resolving references such as "the plan we had before we adjusted the workshop's ending time" and copying that version without hallucination (§3.1.3). Self-coherence items test whether the model keeps its earlier answers when the user repeats or questions them, instead of agreeing sycophantically (§3.1.4).
- **MMSE inputs:** hierarchical topic seeds (App. A.1), a persona sampled from PersonaHub, and a category configuration with five parts: category name, definition, pass criteria, failure criteria, and K-shot failure examples (§4.1, App. A.2).
- **MMSE agents:** the planner writes and updates a conversation blueprint and checks whether a failure condition has been met; the user agent turns the blueprint into user turns; the responder is sampled per item from six frontier models (o1-preview, GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, Mistral Large, Llama 3.1 405B Instruct) so generation does not fit one model's weaknesses (§4.1). A conversation with a detected failure is saved for human editing; one that reaches the maximum turn limit without a failure is discarded (§4.1). The planner prompt forbids gaslighting and says failures on technicalities do not count (App. A.3, Fig. 6).
- **Human review:** annotators check category fit, naturalness, and whether the six frontier models fail reasonably; an item is accepted only if at least 3 of the 6 models fail; accepted edits pass 2 further review layers drawn from a reviewer pool separate from the first annotators (§4.2).
- **Rubric judge:** human raters write, as the last step of each item, a yes/no question that needs only the final response, for example "does any of the dessert recipes suggested in this response contain any nuts?" (§3.2). GPT-4o is the judge model; using Claude as judge gave the same conclusions (§5.1).
- **Editing cost:** 73.6 minutes per item when editing MMSE output versus 154.4 minutes when writing from scratch (§5.2). Levenshtein-based string similarity between the synthetic and final versions averages 74.5%, which the authors report as 25.5% difference from human editing (§5.2).
- **Inference temperatures (App. A.5, Table 6):** GPT-4o 0.0; Llama 3.1 405B, Mistral Large, Claude 3.5 Sonnet 0.2; Gemini 1.5 Pro and o1-preview 1.0; Llama-3.2-3B and Llama-3.3-70B 0.6; Qwen2/Qwen2.5 and Mixtral models 0.7.
- **Release:** data and code at github.com/ekwinox117/multi-challenge (§1); harder items whose rubric questions exceed current LLM judges are kept private (Limitations).

**Human-graded accuracy, % (Table 2)**

| Model | Instr. retention | Inference memory | Versioned editing | Self-coherence | Average |
|---|---|---|---|---|---|
| Claude 3.5 Sonnet (June 2024) | 58.57 | 37.29 | 24.39 | 45.45 | 41.42 |
| o1-preview | 34.29 | 41.53 | 39.02 | 34.09 | 37.23 |
| Gemini 1.5 Pro (Aug 27 2024) | 31.43 | 15.25 | 19.51 | 13.64 | 19.96 |
| Llama 3.1 405B Instruct | 12.86 | 16.95 | 4.88 | 25.0 | 14.92 |
| Mistral Large | 21.43 | 9.32 | 7.32 | 20.45 | 14.63 |
| GPT-4o (August 2024) | 14.29 | 5.08 | 17.07 | 13.64 | 12.52 |

- **Auto-graded (Table 3):** averages 42.75 (Claude 3.5 Sonnet), 35.73 (o1-preview), 21.3 (Gemini 1.5 Pro), 16.5 (Llama 3.1 405B), 13.32 (Mistral Large), 12.9 (GPT-4o); the ranking matches Table 2.
- **Judge alignment (Table 4):** rubric judge 92.26 / 94.62 / 94.85 / 94.12 by category, 93.95 overall; baseline judge 44.44 / 37.53 / 31.82 / 31.05, 37.33 overall.
- **Open-weight models (Table 5, auto-graded averages):** Llama-3.3-70B-Instruct 23.19, Qwen2-72B-Instruct 20.58, Qwen2.5-14B-Instruct 17.07, Qwen2.5-72B-Instruct 16.91, Llama-3.2-3B-Instruct 16.85, Mixtral-8x22B 14.18, Mixtral-8x7B 11.08.

## Findings relevant to generality, negative feedback, long context
- **Separate capabilities:** model rankings differ by category; Claude 3.5 Sonnet leads on retention and self-coherence but trails o1-preview on inference memory and versioned editing (§5.1, Table 2). The authors use this to argue the four categories measure distinct capabilities (Interpretation).
- **Length is not the difficulty source:** accuracy shows no visible trend with the number of turns for any model (§5.2, Fig. 4). The authors attribute difficulty to reasoning over context rather than length, noting the conversations are short relative to model context windows (§5.2; Interpretation).
- **Sycophancy under multi-turn pressure:** in the self-coherence example, most frontier models switch to agree with a user claim that contradicts their earlier step-by-step instructions (§3.1.4). The paper evaluates this behaviour; it does not train against it.
- **Scale and release date:** 70-72B open-weight models tend to outperform smaller ones (§5.2, Fig. 5); newer models tend to score higher, but Claude 3.5 Sonnet (June 2024) outperforms several later releases (§5.2, App. Fig. 8).
- **Measurement limits:** items were selected by failures of the same six frontier models, so the benchmark is biased against them relative to other models (Limitations). The item-selection rule also means a low score partly reflects adversarial filtering rather than an unfiltered sample of conversations.

## Connections
- [[judge-llm-bias]]: MT-Bench, which the paper describes as saturated by frontier models (§1).
- [[mt-eval]], [[multi-if]], [[ifeval]]: earlier multi-turn and instruction-following benchmarks contrasted in §2.
- [[persona-hub]]: source of persona seeds for MMSE (§4.1).
- [[agentinstruct]]: cited multi-agent synthetic data framework (§2).
- [[sycophancy-in-lms]]: cited definition of sycophancy used for the self-coherence category (§3.1.4).
- [[tau-bench]]: cited agent-user interaction benchmark (§2).
- [[parrot-multi-turn]]: a training-side method for context-dependent follow-up queries; MultiChallenge is evaluation only.
- [[prefeval]], [[lost-in-multi-turn]]: related evaluations of preference retention and multi-turn degradation.
- [[rubrics-as-rewards]]: uses instance-level rubrics as rewards; MultiChallenge uses them only for grading.

## Verification
- Created on 2026-09-14 from https://arxiv.org/abs/2501.17399 (v2, 2025-03-06; v1 2025-01-29 confirmed from the v1 PDF stamp).
- Audit claims not found in the source: "36.01% for a plain LLM judge" appears only in §5.2 prose; Table 4 prints 37.33% overall, and §3.2 rounds to 36%. The rubric judge figure is 93.95% in Table 4 and 93.91% in §5.2 prose.
- Internal inconsistencies in the source: §5.1 prose gives average accuracy 38.3 (Claude 3.5 Sonnet), 34.6 (o1-preview), and 19.1 (Gemini 1.5 Pro), while Table 2 and the abstract give 41.42 / 41.4, 37.23, and 19.96; this card uses Table 2. The main text names Claude 3.5 Sonnet (June 2024) and Mistral Large, while App. A.4-A.5 list Claude 3.5 Sonnet (October 2024) and Mistral Large (November 2024).
- Derived check: Table 2 percentages are integer fractions of 70, 118, 41, and 44 items per category (for example 58.57% = 41/70, 41.53% = 49/118, 45.45% = 20/44), not of the Table 1 counts 69, 113, 41, and 50; Table 5 percentages match the Table 1 counts (for example 15.94% = 11/69). The paper does not explain the difference.
- Not reported by the source: judge sampling settings, the maximum turn limit value used by MMSE, inter-annotator agreement among human raters, and any training experiment.
