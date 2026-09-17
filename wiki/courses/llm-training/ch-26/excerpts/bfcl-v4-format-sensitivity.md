---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: primary source (official blog; no library card existed on 2026-09-15)
source_url: https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
created_at: "2026-09-15"
---

# Excerpt: BFCL V4 Part 3 — format sensitivity of tool calls

**Source:** Huanzhi Mao, Mingxuan Li, Ruoxi Wu, Shishir G. Patil, Joseph E. Gonzalez, "BFCL V4 • Agentic, Part 3: Evaluating Format Sensitivity for Tool Calls", Gorilla blog, released 2025-07-17, updated 2025-07-20. Source type: official blog (the benchmark maintainers). Read 2026-09-15.

## Scope

BFCL V4 adds web search (Part 1), memory (Part 2), and format sensitivity (Part 3). Premise: "for any given query, a model should produce the correct output regardless of variations in input format."

## Variations (§2)

| Component | Values |
|---|---|
| return_format | python, json, verbose_xml, concise_xml |
| function_doc_format | python, xml, json |
| has_tool_call_tag | True (wrap calls in `<TOOLCALL>…</TOOLCALL>`), False |
| prompt_format | plaintext, markdown |
| prompt_style | classic, experimental (rephrased) |

All combinations of return format, doc format, and tag (24) plus a markdown and an experimental variant of the original prompt give 26 variations.

## Test entries (§3)

200 single-turn entries from simple, multiple, parallel, and parallel-multiple categories of the curated and live BFCL V2 data (2,351 entries in total): simple 30, parallel 15, multiple 15, parallel_multiple 15, live_simple 24, live_parallel 86, live_multiple 6, live_parallel_multiple 9. For 20 models, accuracy on the 200 entries correlates linearly with accuracy on all 2,351 (Fig. 1). 39 models are evaluated.

## Findings (§4–§5)

- Return format: accuracy is generally higher for Python or JSON than for either XML format, particularly for smaller models; watt-tool-70B and claude-3.7-sonnet show large drops with XML (§4.1).
- Function documentation format: the text says JSON > XML > Python, while the Figure 4 caption says JSON > Python > XML (§4.2). The two statements disagree; both place JSON first.
- Tool-call tags: a slight average drop, with large drops for some small models such as Llama-3.1-8B-Instruct and BitAgent-8B (§4.3).
- Prompt format and style: no consistent trend (§4.4).
- Tool-use-specialized models: watt-tool-70B cannot output JSON-format calls and returns Python-style calls; CoALM-70B shows near-zero accuracy when tool-call tags are required (§4.5).
- Common failure: with XML return format and Python function docs, models write type "int" where the instruction requires "integer" (§5.1).

Numeric per-model accuracies are shown only in heatmaps and bar charts.
