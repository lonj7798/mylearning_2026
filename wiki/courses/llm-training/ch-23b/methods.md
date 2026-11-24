# 23b — Four Ways to Construct or Test Multi-Turn Data

<!-- scope: primary-source setups, measured outcomes, and transfer limits for state updates, policy scope, counterfactual skill families, and causal supervision
     deps: [[read]]
     see-also: [[read]], [[practice]], [[long-tail-papers]]
     sources: [[user-oriented-dialogue-generation]], [[consistentchat]], [[josh-dialogue-self-training]], [[multichallenge]]
-->

## 1. Task generation versus user simulation

**Result (single study).** [[user-oriented-dialogue-generation]] separates task construction from a dedicated user simulator, producing incremental requests and turn feedback rather than assuming every requirement arrives at once. §5.1 seeds generation with Nemotron tool material and τ2 database tasks. The paper reports tested tool-task improvements, but those seeds limit a claim of wholly unseen environment generalization. Limitations include generation cost and brittle access to partially visible databases.

**Interpretation.** A sales simulator can have a hidden scenario plan while revealing requirements naturally. The student must receive only revealed information; task success should depend on valid actions, not agreement with a hidden template. Simulation is useful when the task environment can verify actions. It is not evidence that the resulting wording follows actual customer frequencies.

## 2. Skeletons and a future-query mismatch

**Result (single study).** [[consistentchat]] builds intent/information-flow skeletons, generates user queries, and generates responses in one pass. Its v2 evaluation uses roughly 15,000 training conversations. For Qwen2.5-7B, average consistency is 7.32 versus 7.10 for ShareGPT; MT-Eval multi-turn is 8.38 versus 7.86 (Tables 1–2). These are judged scores, not deployment success percentages.

The generator can see future user queries when constructing responses. Its judge and generator also overlap. A skeleton can help consistency, but future access can encourage answers that anticipate undisclosed facts. For your dataset, validate each response against its prefix even when the whole dialogue was planned together. Shared model-family judges and students are another reason to use independent tool and human checks.

## 3. Search for successful dialogue trajectories

**Result (single study).** [[josh-dialogue-self-training]] uses turn-level branching with sparse task-success feedback in simulated tool-use dialogues. Successful paths and preference supervision are extracted for self-training. Sparse success makes search valuable: a full conversation can fail even when many individual turns sound plausible. First released in 2024 and published in Findings of ACL2025, this is a task-reward method rather than a general lexical-diversity recipe.

**Interpretation.** If booking success is executable and policy-constrained, search can find valid trajectories through corrections and missing fields. But a reward that checks only whether a booking exists can approve an unauthorized booking. Encode preconditions or independently audit them. Simulator errors and narrow task coverage can make apparently successful paths unsuitable for real customers. Detailed optimizer results are not extracted in this card, so this lesson does not invent a best search width or training ratio.

## 4. A stress benchmark is not a deployment census

**Result (single study).** [[multichallenge]] combines generated candidate conversations with human editing and failure-based acceptance. It evaluates 273 conversations covering instruction retention, inference memory, version editing, and self-coherence. Instance-level binary rubrics improve judge-human agreement; existing source-card figures are specific to that grading setup. The benchmark intentionally includes challenging failures, so its category shares are not real user frequencies.

Use its rubric idea to write a pass condition for a sales case, such as whether the final call uses the corrected date. Do not train on the held-out benchmark and then treat its score as independent. New synthetic examples inspired by a capability should have distinct families and no copied benchmark answer patterns. The lesson is to expose a behavior and check it precisely, not to claim all conversation tails have been covered.

## Comparison

| Method | Main control | Main supervision | Coverage limitation |
|---|---|---|---|
| User-oriented generation | Hidden task and simulated user | Incremental requests/feedback | Simulator/environment fidelity |
| ConsistentChat | Intent/flow skeleton | Generated conversation targets | Future-query and judge mismatch |
| JOSH | Branch search | Task success and preferences | Reward validity and search coverage |
| MultiChallenge | Failure construction and human review | Evaluation rubrics | Adversarial selection, not frequency |

[Main lesson](read.md) · [Practice](practice.md) · [한국어](methods_kor.md).

