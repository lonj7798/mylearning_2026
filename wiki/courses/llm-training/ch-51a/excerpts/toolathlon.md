---
chapter: ch-51a
course: llm-training
phase: read
excerpt_of: "The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution (arXiv:2510.25726v2, 2026-02-26)"
source_url: https://arxiv.org/abs/2510.25726
created_at: "2026-09-15"
note: "No library card exists at wiki/raw-data/llm-training/**/toolathlon.md on 2026-09-15. Values read from the v2 PDF on 2026-09-15."
---

# Excerpt: Toolathlon — cross-application tasks with pass@1, pass@3, and pass^3

**Authors:** Junlong Li, Wenshuo Zhao, Jian Zhao, Weihao Zeng, Haoze Wu and colleagues (HKUST, All Hands AI,
Carnegie Mellon University, Duke University; corresponding author Junxian He). Used by [[read]] §2, §4, and the
Recipe table.

## Composition (Abstract, §3.1, §3.2; Table 2)

- 32 software applications and 604 tools, "ranging from everyday platforms such as Google Calendar and Notion to
  professional applications like WooCommerce, Kubernetes, and BigQuery"; most tools come from Model Context
  Protocol (MCP) servers the authors revised or implemented.
- 108 tasks, plus 7 local toolkits with 16 tools. Tools visible per task: mean 69.9, min 28, max 128.
  72 of 108 tasks (67%) begin from an initialized environment state (Table 2).
- Task prompts are written to be short and underspecified, "to mirror authentic user queries", so the agent must
  infer intent and plan (§3.1). Each task has a deterministic evaluation script comparing the final state against
  static or dynamically regenerated ground truth (§2.4).
- Construction cost: about 4–6 hours of graduate-student work per task, plus review rounds of about 5 hours per
  task per round by 5–6 experienced authors (§3.2).

## Harness (§4.1)

- Maximum 100 turns for every model. Only the MCP servers and common tools useful for the task are attached, but
  "the models will still see many unnecessary tools during evaluation" because each server exposes many tools.
- "We evaluate each model three times and report the average pass@1 success rate as well as the standard
  deviation. We also include the pass@3 ... and pass^3 (Yao et al., 2025) — the fraction of tasks where all three
  trajectories are correct, to measure the model's potential capability coverage and its ability to complete tasks
  reliably."

## Results (Table 3)

| Model | P@1 | P@3 | P^3 | Avg turns |
|---|---|---|---|---|
| Claude-4.5-Sonnet | 38.6 ± 2.7 | 51.9 | 20.4 | 20.2 |
| GPT-5 | 30.6 ± 1.5 | 43.5 | 16.7 | 18.7 |
| Claude-4-Sonnet | 29.9 ± 1.6 | 41.7 | 17.6 | 27.3 |
| GPT-5-high | 29.0 ± 3.1 | 42.6 | 16.7 | 19.0 |
| Grok-4 | 27.5 ± 1.7 | 38.9 | 16.7 | 20.3 |
| DeepSeek-v3.2-Exp (best open-weight) | 20.1 ± 1.2 | 27.8 | 12.0 | 26.0 |
| GLM-4.6 | 18.8 ± 2.2 | 29.6 | 9.3 | 27.9 |
| Gemini-2.5-Pro | 10.5 ± 1.9 | 21.3 | 2.8 | 26.5 |
| Gemini-2.5-Flash | 3.7 ± 1.5 | 8.3 | 0.0 | 8.3 |

- §4.2: "increased reasoning effort for thinking-oriented models (e.g., GPT-5 vs. GPT-5-high) shows no benefit,
  suggesting that exploring new observations matters more than extended internal reasoning in agentic tasks."
- §4.2: "We also observe significant differences between Pass@3 and Pass^3 success rates. This indicates that
  while many models have certain capability coverage, they lack consistency in producing reliable results."
- Per-category scores (Research / Campus / Finance / Tech / Business / Daily / E-com) differ by model:
  Claude-4.5-Sonnet leads in most categories, GPT-5 is highest on Daily (39.2), Grok-4 on Tech (43.9).

## Limit stated or observable

Three runs per model give a standard deviation but no confidence interval; with 108 tasks a one-run difference of
a few points is within the printed spread. The benchmark has no private held-out split.
