<!-- scope: tool-calling synthesis — Glaive V2 protocol for synthetic function-calling dialogs
     deps: [[apigen]]
     see-also: [[nexusraven]], [[granite-function-calling]]
-->

# Glaive Function Calling V2
- **Core Insight:** An early (late-2023) open synthetic function-calling dataset that established the de-facto "system prompt contains tools in JSON, assistant emits `<functioncall>` XML wrappers" template widely adopted in open fine-tunes before the OpenAI-compatible `tool_calls` format became standard.
- **Guideline:** When building cheap function-calling SFT data at moderate quality, Glaive V2's template (JSON tool schemas in system prompt, `<functioncall>{…}</functioncall>` assistant tokens, `FUNCTION RESPONSE:` user-role replies) remains a reasonable starting point, though 2024+ corpora (APIGen, ToolACE) dominate on quality.
- **Authors:** Glaive (glaiveai.github.io)
- **Year:** 2023 (V1), V2 late 2023
- **URL:** https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2
- **Relevant topics:** function calling, early open FC data, chat template, synthetic dialogs

## Abstract
Glaive released glaive-function-calling-v2 in late 2023 as a ~112K-example open synthetic dataset of function-calling conversations. The data was generated with a proprietary Glaive pipeline (internal LLM teacher + schema-conformance filter), covering general-purpose APIs (weather, calendar, search, math). It became widely adopted for open fine-tunes (Dolphin, Hermes-function-calling, early Mistral variants) due to permissive licensing and availability before APIGen-style verified data existed.

## Key Contributions
- **First widely-used open FC dataset at scale** (Oct 2023).
- **Chat-template convention** — system-prompt tool schemas + `<functioncall>…</functioncall>` assistant tokens + `FUNCTION RESPONSE: {…}` tool-role replies — adopted broadly in 2023/early-2024 open releases.
- **Apache-2.0 style permissive license** enabled commercial use of derivative fine-tunes.
- Used as data source by Granite-FC, OpenHermes-2.5, Dolphin-FC.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)
- **Seed input:** ~50 hand-authored function schemas (weather, calendar, email, search, math, calculator, ticket booking, etc.).
- **Dialog generation:** proprietary Glaive platform generates user-assistant dialogs that naturally lead to function calls. Each dialog has a user intent, 1–4 turns, and 1 to several function calls.
- **Schema-conformance filter:** generated calls must have valid JSON matching the function schema; type-check enforced.
- **Format standardization:** every assistant turn with a call uses `<functioncall>{"name": "...", "arguments": {…}}</functioncall>`. Tool responses inserted as `FUNCTION RESPONSE: {…}`.
- **Output shape:** ~112K conversations (glaive-function-calling-v2); mostly 2–4 turns; 1–3 calls per conversation.
- **Teacher model(s):** not publicly disclosed (Glaive's internal platform, likely GPT-3.5 / GPT-4 via API at 2023 rates).
- **Cost:** not disclosed.

## Modality-specific technical details (REQUIRED — tool-calling)
- **API registry size:** ~50 base schemas; variations in parameter values produce effective diversity.
- **Exact verification rules:** schema JSON validity only. No execution check. No LLM-judge semantic check. This is Glaive's weak spot vs APIGen / ToolACE.
- **Hallucination-rate measurement:** not directly measured in the dataset card; downstream models trained on it show ~15% hallucinated tool-call rate on BFCL relevance (vs ~5% for APIGen-trained models).
- **Call format:** `<functioncall>…</functioncall>` XML wrapper — pre-dates OpenAI tool-calls JSON standard.
- **Why still relevant:** its template became a semi-standard for "pre-OpenAI-JSON" tooling; some open models (early Hermes-FC, LLaMA-2 tool fine-tunes) still follow it.

## Quality / diversity evaluation
- Not formally benchmarked in a paper; quality judged downstream via derivative fine-tunes.
- Hermes-2-Pro-Mistral-7B (trained on OpenHermes + Glaive-FC-V2): BFCL-V1 ~70% — respectable for 2023-era data but ~20 points below APIGen-trained models of similar size.
- Best used today as **supplementary** rather than primary training source.

## Risks + gotchas
- **No execution verification:** some function calls produce arguments that would fail in real execution.
- **Proprietary generation pipeline:** non-reproducible; users must trust Glaive's internal filters.
- **Template drift:** `<functioncall>` XML is not the OpenAI-standard; training on it locks models into a pre-2024 format unless reformatted.
- **Small schema pool** means limited coverage of real enterprise APIs.

## Connections
- Consumed by: [[granite-function-calling]], OpenHermes-2.5, Dolphin-FC.
- Superseded in quality by: [[apigen]], [[toolace]], [[xlam]].
- Template ancestor of modern `tool_calls` conventions.
