---
chapter: ch-45c
course: llm-training
phase: read
excerpt_of: Manus blog, "Context Engineering for AI Agents: Lessons from Building Manus" (no library card as of 2026-09-15)
source_url: https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
created_at: "2026-09-15"
---

# Excerpt: Context Engineering for AI Agents — Lessons from Building Manus

**Author:** Yichao 'Peak' Ji (Manus). Published 2025-07-18. Source type: practitioner evidence — a production report with operational numbers but no controlled experiment, no benchmark table and no ablation. The author states the scope himself: "None of what we've shared here is universal truth—but these are the patterns that worked for us."
**Used in:** [[read]] §7, Common mistakes.

## KV-cache economics (section "Design Around the KV-Cache")
- "If I had to choose just one metric, I'd argue that the KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost."
- Shape of an agent's token traffic: "the context grows with every step, while the output—usually a structured function call—remains relatively short ... In Manus, for example, the average input-to-output token ratio is around 100:1."
- Price gap quoted at the time of writing: "with Claude Sonnet, for instance, cached input tokens cost 0.30 USD/MTok, while uncached ones cost 3 USD/MTok—a 10x difference."
- Three practices: keep the prompt prefix stable ("even a single-token difference can invalidate the cache from that token onward"; a timestamp in the system prompt is named as the common cause); make the context append-only, with deterministic serialization (JSON key ordering is named); mark cache breakpoints explicitly when the provider does not do incremental prefix caching.

## Masking instead of removing tools (section "Mask, Don't Remove")
- "avoid dynamically adding or removing tools mid-iteration", for two stated reasons: tool definitions sit near the front of the context, so a change invalidates the KV-cache for everything after it; and "When previous actions and observations still refer to tools that are no longer defined in the current context, the model gets confused ... this often leads to schema violations or hallucinated actions."
- The alternative used: a state machine masks token logits during decoding. Three function-calling modes are implemented by prefilling, using the Hermes format as the example: Auto (`<|im_start|>assistant`), Required (`<|im_start|>assistant<tool_call>`), Specified (`<|im_start|>assistant<tool_call>{"name": "browser_`). Tool names share prefixes (`browser_`, `shell_`) so a group can be selected by prefill alone.

## File system as externalized context (section "Use the File System as Context")
- Three stated pain points with 128K-plus windows: observations can exceed the limit; "Model performance tends to degrade beyond a certain context length, even if the window technically supports it"; long inputs cost money even with prefix caching.
- Compression rule: "Our compression strategies are always designed to be restorable. For instance, the content of a web page can be dropped from the context as long as the URL is preserved, and a document's contents can be omitted if its path remains available in the sandbox."
- Stated reason for preferring restorable compression: "an agent, by nature, must predict the next action based on all prior state—and you can't reliably predict which observation might become critical ten steps later. From a logical standpoint, any irreversible compression carries risk."

## Recitation (section "Manipulate Attention Through Recitation")
- "A typical task in Manus requires around 50 tool calls on average."
- Manus writes and rewrites a `todo.md` file: "By constantly rewriting the todo list, Manus is reciting its objectives into the end of the context. This pushes the global plan into the model's recent attention span, avoiding 'lost-in-the-middle' issues and reducing goal misalignment." No measurement is attached.

## Keeping failures in context (section "Keep the Wrong Stuff In")
- "leave the wrong turns in the context. When the model sees a failed action—and the resulting observation or stack trace—it implicitly updates its internal beliefs. This shifts its prior away from similar actions, reducing the chance of repeating the same mistake." No measurement is attached.
- "error recovery is one of the clearest indicators of true agentic behavior. Yet it's still underrepresented in most academic work and public benchmarks, which often focus on task success under ideal conditions."

## Uniform context as a failure mode (section "Don't Get Few-Shotted")
- "If your context is full of similar past action-observation pairs, the model will tend to follow that pattern, even when it's no longer optimal." Example given: reviewing a batch of 20 resumes. The countermeasure used is "small amounts of structured variation in actions and observations—different serialization templates, alternate phrasing, minor noise in order or formatting."

## Not reported by the source
Model identity for the 100:1 ratio and the 50-tool-call average, task mix, measurement window, any A/B result for masking versus removal, recitation, or keeping failures in context. Prices quoted are as of July 2025.

## Verification
- Read on 2026-09-15 from the page text at https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus (dated 2025/7/18).
