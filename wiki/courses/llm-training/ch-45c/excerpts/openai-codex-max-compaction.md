---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: OpenAI product post, "Building more with GPT-5.1-Codex-Max" (no library card as of 2026-09-15)
source_url: https://openai.com/index/gpt-5-1-codex-max/
created_at: "2026-09-15"
---

# Excerpt: Building more with GPT-5.1-Codex-Max — compaction

**Publisher:** OpenAI. Published 2025-11-17. Source type: official product announcement. No system-card-level detail is included here; the linked system card was not read for this excerpt.
**Used in:** [[read]] §3, §7, Recipe.

## Compaction as a trained capability
- "It's our first model natively trained to operate across multiple context windows through a process called *compaction*, coherently working over millions of tokens in a single task."
- Operational description: "Compaction enables GPT‑5.1‑Codex‑Max to complete tasks that would have previously failed due to context-window limits, such as complex refactors and long-running agent loops by pruning its history while preserving the most important context over long horizons. In Codex applications, GPT‑5.1‑Codex‑Max automatically compacts its session when it approaches its context window limit, giving it a fresh context window. It repeats this process until the task is completed."
- Duration claim: "GPT‑5.1‑Codex‑Max can work independently for hours at a time. In our internal evaluations, we've observed GPT‑5.1‑Codex‑Max work on tasks for more than 24 hours."
- Attribution of benchmark gains to compaction: "Because it can coherently work across multiple context windows using compaction, the model delivers improved results on challenges in areas like long-horizon coding and cybersecurity." No ablation separating compaction from the rest of the model update is given.

## Token efficiency
- "On SWE-bench Verified, GPT‑5.1‑Codex‑Max with 'medium' reasoning effort achieves better performance than GPT‑5.1‑Codex with the same reasoning effort, while using 30% fewer thinking tokens." The two scores themselves are not printed for the medium setting.
- A new "Extra High ('xhigh') reasoning effort" is introduced; "We still recommend medium as the daily driver for most tasks."

## Appendix table (as printed)

| Benchmark | GPT‑5.1‑Codex (high) | GPT‑5.1‑Codex‑Max (xhigh) |
|---|---|---|
| SWE-bench Verified (n=500) | 73.7% | 77.9% |
| SWE-Lancer IC SWE | 66.3% | 79.9% |
| Terminal-Bench 2.0 | 52.8% | 58.1% |

These two columns differ in both model and reasoning effort, so the table does not isolate compaction.

## Not reported by the source
The context-window size, the compaction trigger threshold, what the compaction step keeps or discards, whether compaction is exposed through the API, how compaction was trained, and any with/without-compaction comparison at a fixed checkpoint.

## Verification
- Read on 2026-09-15 from the page text of https://openai.com/index/gpt-5-1-codex-max/ (published 2025-11-17), retrieved through a text-extraction proxy after a direct fetch returned HTTP 403.
