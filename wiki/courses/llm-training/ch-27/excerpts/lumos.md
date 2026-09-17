---
chapter: ch-27
course: llm-training
phase: read
excerpt_of: "Agent Lumos: Unified and Modular Training for Open-Source Language Agents (arXiv:2311.05657v3)"
source_url: https://arxiv.org/abs/2311.05657
created_at: "2026-04-23"
revised: "2026-09-15 (rewritten from the primary text; the library card papers/lumos.md predates verification and gives data sizes and generalization numbers not in the paper)"
---

# Excerpt: Lumos — planning and grounding modules trained on converted rationales

**Authors:** Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, Bill Yuchen Lin (UCLA; Allen Institute for AI; University of Washington). arXiv v1 2023-11, v3 2024-07-10; ACL 2024.

## Architecture (§2, Fig. 2)
- Planning module: decomposes a task into high-level subgoals.
- Grounding module: translates subgoals into executable low-level actions for a predefined action interface.
- Execution module: tools and APIs (for example a QA model, a calculator, a browser) that run the actions.
- Lumos-OnePass (Lumos-O): all subgoals and actions in one inference call. Lumos-Iterative (Lumos-I): one subgoal at a time, conditioned on previous execution results.

## Training data (§3, §4.1)
- GPT-4 (8/13/2023 and 9/13/2023 versions) and GPT-4V convert ground-truth reasoning steps from existing benchmarks (math, QA, web, multimodal) into subgoal and action annotations.
- After removing annotations with mismatched parentheses, invalid execution outputs, or excessive length: 55,382 planning and 55,499 grounding annotations ("56K").
- Base models: Llama-2-7B and Llama-2-13B for both modules.

## Results
- Mind2Web step success (Table 1a): Lumos-I-Web 27.6 (7B), 31.3 (13B); GPT-4 22.6; AgentLM-70B 13.5.
- Formulation comparison on the same data (Table 3): HotpotQA LLM accuracy CoT training 22.1, integrated single-module training 39.6, Lumos-O 39.2, Lumos-I 45.9; Mind2Web integrated 25.3 vs Lumos-I 27.6.
- Unseen tasks with 2–3-shot examples of the new action interface (§4.4, Table 1e): WebShop average reward / InterCodeSQL success — Lumos-I-All-13B 50.3 / 7.3; single-type variants Web 46.2 / 4.2, Math 45.7 / 5.8, QA 47.3 / 3.5, Multimodal 43.8 / 4.0; Vicuna-v1.3-33B 23.9 / 6.7; Claude-instant 49.7 / –.
- Annotation quality (Table 2): on the same Llama-7B base and 2,000 examples, Lumos data vs ReWOO-open data: StrategyQA 58.3 vs ≈57; HotpotQA 38.1 vs ≈37.
- General instruction following (§4.5): mixing Alpaca data with Lumos annotations gives SuperNI ROUGE-L 39.3 vs 39.8 for Llama-2-7B trained on Alpaca alone.

## Not in the paper
"~40K tasks → ~200K training turns", "Lumos-O (onetime)", "~$15K GPT-4 cost", "~8-point drop vs ~20 for monolithic ReAct", "Lumos-13B HotpotQA 39.3 EM", unified grammar `Search/Retrieve/Calculate/Click/Type/Back/Finish` as a fixed list.
