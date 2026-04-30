---
chapter: ch-25
course: llm-training
phase: read
excerpt_of: System-prompt diversity (Persona-Hub / SystemChat-2.0); Capybara (LDJ / Nous); SmolTalk (HuggingFace SmolLM2)
source_url: https://arxiv.org/abs/2406.20094 ; https://huggingface.co/datasets/cognitivecomputations/SystemChat-2.0 ; https://huggingface.co/datasets/LDJnr/Capybara ; https://huggingface.co/datasets/HuggingFaceTB/smoltalk
created_at: "2026-04-23"
---

# Excerpt: Modern Multi-Turn Mix Composition — System-Prompt Diversity, Capybara, SmolTalk

**Sources:**
- `wiki/raw-data/llm-training/papers/system-prompt-diversity.md`
- `wiki/raw-data/llm-training/papers/capybara.md`
- `wiki/raw-data/llm-training/papers/smol-talk.md`

---

## Why this excerpt anchors ch-25 §7–§8

Baize, UltraChat, CAMEL, SODA, Prosocial, OASST each answer "how do I generate one kind of multi-turn data." Ch-25's final move is showing how 2024–2025 production mixes *compose* those recipes. Three reference recipes at three different scale points:

- **System-prompt diversity (Persona-Hub + SystemChat-2.0)** — the 2024 extension that adds the behavioral axis.
- **Capybara (~20K)** — small, curated, composed from heterogeneous seeds.
- **SmolTalk (~1M)** — canonical modern open recipe, Magpie-anchored with targeted sub-mixes.

Together they show the shift: multi-turn is no longer the SFT bottleneck it was in 2023; it's now a minority slice for realism and steerability.

---

## Part 1 — System-prompt diversity as the behavioral axis

From [[system-prompt-diversity]]:

> *"Conditioning synthetic conversations on a diverse pool of system prompts (persona-defining, task-framing, constraint-setting) is the highest-leverage source of diversity in modern chat SFT corpora — more so than topic/seed-question diversity."*

> *"Corpora like SystemChat and Persona-Hub-derived dialogs demonstrate that swapping system prompts across thousands of personas produces substantially more varied behavior than swapping seed topics."*

### The two canonical sources

From [[system-prompt-diversity]]:

> *"Persona-Hub: 1B personas mined from web bios, about-me pages, LinkedIn-style profiles."*

> *"SystemChat: hand-curated + GPT-4-synthesized system prompts covering roles, constraints ('always answer in bullet points'), personas, styles, safety modes."*

Persona-Hub is *mined*, not generated — the 1B count comes from actually crawling public profiles. SystemChat-2.0 is *generated* (hand-curated + GPT-4-expanded) — only 7K dialogues, each with a distinct system prompt. Two different strategies for producing diverse system-prompt pools.

### The generation loop

From [[system-prompt-diversity]]:

> *"Dialog generation:*
> *- Step 1: sample a system prompt from the pool.*
> *- Step 2: teacher model (GPT-4 / Claude) generates a multi-turn dialog under that system prompt.*
> *- Step 3: optional — constrain the user side via a user-persona system prompt (compounded diversity)."*

**Notice** Step 3 — the user side can be *separately* persona-conditioned. This is where the behavioral axis compounds with topic + role axes: one sample carries `(assistant_persona, user_persona, topic, role)` as four independent conditioning tuples.

### Empirical findings

From [[system-prompt-diversity]]:

> *"The core finding across several 2024/2025 open recipes: system-prompt conditioning yields up to 2× more unique vocabulary and 3× more persona-consistent follow-ups than topic-only conditioning."*

> *"Open evals (IFEval, AgentBench): +5–10 points over topic-only-diverse SFT baselines when persona-SFT is added."*

+5–10 on IFEval is the headline number. IFEval measures instruction-following adherence when the model is given a novel system prompt at inference; persona-SFT'd models have seen thousands of different system prompts during training, so novel system prompts at test time no longer break them.

### Where this fits in 2025 production

From [[system-prompt-diversity]]:

> *"Key datasets:*
> *- Persona-Hub-Instruct (Tencent, 2024): 1M+ persona-conditioned instructions.*
> *- SystemChat-2.0 (Eric Hartford, 2024): 7K dialogs with diverse system prompts.*
> *- Tülu-3 SFT mix includes persona-conditioned sub-mix."*

---

## Part 2 — Capybara: compact seed-composition

From [[capybara]]:

> *"Multi-turn quality beats raw scale — ~20K carefully curated conversations (60%+ multi-turn) generated via the 'Amplify-Instruct' pipeline that composes seeds from multiple existing synthetic methods (Airoboros + Evol-Instruct + Orca + Vicuna + CamelAI + LessWrong posts) can train competitive chat models."*

### The Amplify-Instruct pipeline

From [[capybara]]:

> *"1. Sample a seed instruction from the composed pool.*
> *2. Generate initial assistant response.*
> *3. Amplify: synthesize a plausible follow-up user turn (varied via distributional samplers to promote diversity).*
> *4. Generate assistant response to follow-up.*
> *5. Iterate for N turns (typically 2–6).*
> *6. Filter by length, dedup, and quality (manual + LLM-check)."*

Two interesting design choices:

1. **Seeds come from other synthetic datasets, not from a single taxonomy.** Airoboros + Evol-Instruct + Orca + Vicuna + LessWrong + CamelAI. Each source contributes its own stylistic DNA — Airoboros's red-team-style prompts, LessWrong's analytic-rationality register, CamelAI's domain-expert personas. Composition is the diversity mechanism; a single homogeneous taxonomy (UltraChat-style) would miss the stylistic variance.
2. **Amplification is one-direction.** The user-turn synthesizer only generates *follow-ups* — continuations of the conversation — never restart prompts. This is what makes 20K seeds turn into 20K *multi-turn* dialogues rather than 60K single-turn ones.

### Why small can win

From [[capybara]]:

> *"Nous-Capybara-7B-V1.9 (Mistral-7B) achieves strong MT-Bench / AGIEval numbers at release given its small SFT set."*

> *"Practitioner takeaways: Seed diversity > raw scale — Capybara's compact size is deliberate."*

The argument: if seed composition captures the stylistic axes well, marginal returns to additional dialogues diminish quickly. Capybara deliberately stops at ~20K because (a) it fits LoRA fine-tuning on a single consumer GPU, and (b) further dialogues at the same seed diversity would be near-duplicates.

**Notice** this is the opposite of UltraChat's bet (1.5M dialogues, single teacher, taxonomy-enumerated). Both can work; the tradeoff is `quality × diversity` vs. `scale × coverage`.

---

## Part 3 — SmolTalk: canonical 1M-scale modern recipe

From [[smol-talk]]:

> *"A 1M-sample SFT mix built mostly from Magpie-Ultra (400K) generated with Llama-3.1-405B-Instruct plus targeted public datasets — this recipe gives best-in-class small-model post-training results (SmolLM2-1.7B-Instruct) and validates Magpie-scale synthetic SFT in the open."*

### The composition table — verbatim

From [[smol-talk]]:

| Component | Size | Source / Purpose |
|---|---|---|
| **Smol-Magpie-Ultra** | **400K** | Magpie pipeline w/ Llama-3.1-405B-Instruct — core instruction coverage |
| Smol-constraints | 36K | Constraint-following tasks (precise formatting, JSON, word limits) |
| Smol-rewrite | 50K | Rewrite / paraphrase / tone-shift tasks |
| Smol-summarize | 100K | Summarization tasks |
| OpenHermes 2.5 (subset) | 100K | Teknium's open catalogue — general instruction + reasoning |
| MetaMathQA (subset) | 50K | Math word problems |
| NuminaMath-CoT | (subset) | Competition-grade math with CoT |
| Self-Oss-Starcoder2-Instruct | (subset) | Open-source code instructions |
| **SystemChats 2.0** | **30K** | **System-prompt-conditioned dialogues** |
| LongAlign | (subset) | Long-context alignment |
| Everyday-conversations | (subset) | Casual dialogue |
| Explore-Instruct-Rewriting | (subset) | Instruction rewriting |

### What the proportions reveal

The multi-turn dialogue slice (SystemChats-2.0) is **30K of 1M — 3%**. Not UltraChat, not SODA, not CAMEL. Just 30K system-prompt-conditioned dialogues as realism-insurance.

This is the structural shift from the 2023 era (when multi-turn was the SFT bottleneck and UltraChat's 1.5M dominated open mixes) to 2025 (when multi-turn capability is learned mostly from in-context follow-through patterns pervading single-turn SFT, and explicit dialogue data is a small targeted slice).

From [[smol-talk]]:

> *"Why each piece:*
> *- Magpie-Ultra (40%): captures the bulk of general instruction diversity via a strong 405B teacher; cheap (no API) because Llama-3.1-405B is open.*
> *- Smol-constraints / Smol-rewrite / Smol-summarize: targeted synthesis for capabilities that Magpie underweights.*
> *- Public math/code datasets: verifiable domains where synthetic alone isn't enough; hand-curated mixes.*
> *- SystemChats 2.0: system-prompt steerability."*

Magpie gets the bulk because a 405B open teacher is cheaper than GPT-4o and (empirically) produces comparable quality. Targeted sub-mixes fill specific capability gaps that Magpie underweights — constraint-following, rewriting, summarization. SystemChats is the *only* explicit multi-turn slice.

### The shift is real and intentional

From [[smol-talk]]:

> *"Open Magpie-style synthesis at ~400K scale is the new 'default open SFT anchor.'"*

> *"Composition table is tunable — for a coding-focused variant, inflate the code share."*

The 2025 modal recipe: one large Magpie anchor + many small targeted slices + one small multi-turn realism slice. No single massive multi-turn corpus. The pipelines in ch-25 §1–§5 all still exist as *sources* (SystemChats is a successor of UltraChat's system-prompt generation; Magpie-Ultra borrows UltraChat's taxonomy instincts), but their role has moved from "provide multi-turn data" to "provide *seeds* for in-context generation or small realism slices."

---

## Connections

- Chapter synthesis: [[ch-25]] §7–§8.
- The 2023-era "multi-turn dominates SFT" anchor: [[excerpts/ultrachat-two-model-protocol]].
- The role-pair axis that carries forward into persona-conditioning: [[excerpts/camel-inception-prompting]].
- The grounding axis that complements persona-conditioning: [[excerpts/soda-commonsense-grounding]].
- The real-human anchor slice in modern mixes: [[excerpts/openassistant-tree]] (5–15% of safety / realism sub-mix).
