---
chapter: ch-29d
course: llm-training
phase: read
excerpt_of: primary source (no library card at wiki/raw-data/llm-training/**/tau-bench.md on 2026-09-15)
source_url: https://arxiv.org/abs/2406.12045
source_version: arXiv:2406.12045v1 (2024-06-17)
created_at: "2026-09-15"
---

# Excerpt: τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (Yao, Shinn, Razavi, Narasimhan; Sierra)

Facts used by [[read]] §1, §2, and the Recipe table. Read in the arXiv v1 PDF on 2026-09-15.

## Environment and user simulator (§3)
- Each task is a POMDP (S, A, O, T, R, U). The state combines a hidden database state and a user state; the agent acts on the database through API tools and on the user through messages (§3).
- "We use a language model (gpt-4-0613) to simulate a human user interacting with the agent." The user state is the task instruction as a system prompt plus the conversation so far. "The user cannot see the interaction history between the agent and API tools." The episode ends when the user emits "###STOP###" (§3, User simulation).
- Task instance: a user instruction hidden from the agent, plus the ground-truth database write actions and optional required outputs. The instruction "sets up user identity, intent, and preferences in a way that guarantees only one possible outcome under the domain policy" (§3, Task instances).

## Reward and pass^k (§3)
- r = r_action × r_output ∈ {0, 1}: r_action checks that the final database equals the unique ground-truth database; r_output checks that agent messages contain the required information as substrings.
- The authors note: "r = 1 might be a necessary but not sufficient condition for a successful episode e.g., the agent might issue the return without explicit user confirmation, which violates the policy."
- With n trials of a task and c successes: pass^k = E_task[C(c,k)/C(n,k)] ("the chance that all k i.i.d. task trials are successful"), pass@k = 1 − E_task[C(n−c,k)/C(n,k)]. The default metric is pass^1 = E[c/n].

## Construction (§4)
- Stage III: write a user instruction, run a gpt-4-turbo function-calling agent, and edit the instruction until no ambiguity remains; each τ-retail task was run with more than 40 gpt-4-turbo trials and low-success tasks were checked (§4, App. A Figure 7).
- Domains (Table 1): τ-retail 500 users, 50 products, 1,000 orders, 7 write and 8 non-write tools, 115 tasks; τ-airline 500 users, 300 flights, 2,000 reservations, 6 write and 7 non-write tools, 50 tasks.

## Experimental settings and results (§5)
- At most 30 agent actions per task; at least 3 trials per task for Table 2; temperature 0.0 for the agent and 1.0 for the user (§5, Methods).
- pass^1 with function calling (Table 2): gpt-4o 61.2 retail / 35.2 airline; claude-3-opus 44.2 / 34.7; gpt-3.5-turbo 20.0 / 10.8; meta-llama-3-70B (ReAct) 14.8 / 14.4.
- For gpt-4o function calling on τ-retail, pass^8 falls below 25% (§5.1, Figure 4).
- Cost: gpt-4o agent $0.38 and gpt-4 user simulator $0.23 per τ-retail task (§5.1).
- Failure breakdown: of 40 failed gpt-4o trajectories on τ-retail (1 trial per task), 4 were caused by user instruction typos or ambiguity and were fixed; the other 36 are agent failures (§5.2, Figure 5).

## Simulator limitations stated by the authors (§6)
- "(1) the user instruction might contain typos or ambiguities [...]; (2) the user instruction may not contain all domain knowledge [...]; or (3) the user simulation LM might have limited capacity at reasoning, calculation, long-context memorization, or alignment with the instruction prompt, e.g., in § C.2.2 the user authorizes the agent-recommended lamp without double checking its features."
- "There is also some element of implicit bias during the task curation process since we use the gpt-4-turbo FC agent to tune the user's system prompt."

## Not reported
A quantitative user-simulator error rate (the later [[tau2-bench]] reports one for retail and airline).
