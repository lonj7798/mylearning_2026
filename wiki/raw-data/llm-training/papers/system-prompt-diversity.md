<!-- scope: multi-turn conversation synthesis — system-prompt-conditioned dialogue diversity
     deps: [[persona-hub]]
     see-also: [[ultrachat-pipeline]], [[camel]]
-->

# System-Prompt Diversity in Conversation Synthesis
- **Core Insight:** Conditioning synthetic conversations on a diverse pool of **system prompts** (persona-defining, task-framing, constraint-setting) is the highest-leverage source of diversity in modern chat SFT corpora — more so than topic/seed-question diversity. Corpora like SystemChat and Persona-Hub-derived dialogs demonstrate that swapping system prompts across thousands of personas produces substantially more varied behavior than swapping seed topics.
- **Guideline:** For diverse chat SFT, generate each dialog under a different system prompt drawn from a large persona / task / constraint pool (Persona-Hub's 1B personas is the canonical source); this outperforms topic-diversity strategies like UltraChat's sector taxonomy.
- **Authors:** Multiple contributors; canonical 2024+ references include Persona-Hub (Ge 2024), SystemChat by Eric Hartford, and the persona-conditioned SFT subsections in Tülu 3, Qwen 2.5 reports.
- **Year:** 2024–2025
- **URL:** https://arxiv.org/abs/2406.20094 (Persona-Hub) ; https://huggingface.co/datasets/cognitivecomputations/SystemChat-2.0 (SystemChat)
- **Relevant topics:** system-prompt conditioning, persona diversity, conversation SFT, Persona-Hub

## Abstract
This file covers the class of dialog synthesis techniques where a **diverse system-prompt pool** is the primary diversity knob. Persona-Hub (Ge 2024) provides 1B personas extracted from the web; derivative datasets (Persona-Hub-Instruct, SystemChat-2.0) use these personas as system prompts to generate dialogs, producing behavior variations that topic-diversity approaches miss. The core finding across several 2024/2025 open recipes: **system-prompt conditioning yields up to 2× more unique vocabulary and 3× more persona-consistent follow-ups than topic-only conditioning**.

## Key Contributions
- Establishment of **system-prompt diversity** as a first-class concern in chat-data synthesis.
- **Persona-Hub** and derivative persona-conditioned datasets (see [[persona-hub]]).
- **SystemChat-2.0** — 7K dialogs each conditioned on a distinct non-default system prompt.
- Persona-conditioned SFT sub-mix integrated into Tülu 3, Qwen 2.5, many 2024+ open models.

## Synthesis pipeline (REQUIRED — concrete, modality-specific)

### Persona / system-prompt pool
- Persona-Hub: 1B personas mined from web bios, about-me pages, LinkedIn-style profiles.
- SystemChat: hand-curated + GPT-4-synthesized system prompts covering roles, constraints ("always answer in bullet points"), personas, styles, safety modes.
- Task-framing prompts: "You are an expert X"; behavior-framing: "Respond only in rhyming couplets".

### Dialog generation
- **Step 1:** sample a system prompt from the pool.
- **Step 2:** teacher model (GPT-4 / Claude) generates a multi-turn dialog under that system prompt.
- **Step 3:** optional — constrain the user side via a user-persona system prompt (compounded diversity).

### Filtering
- Consistency: responses should remain in the persona/constraint across turns.
- Coherence: dialog must make sense.
- Non-degenerate: avoid empty or repetitive turns.

### Key datasets
- **Persona-Hub-Instruct** (Tencent, 2024): 1M+ persona-conditioned instructions.
- **SystemChat-2.0** (Eric Hartford, 2024): 7K dialogs with diverse system prompts.
- **Anthropic persona dataset** (internal, via Claude's "persona" generation pipelines).
- **Tülu-3 SFT mix** includes persona-conditioned sub-mix.

- **Output shape:** highly variable — the point is that no two dialogs share a system prompt.
- **Teacher model:** GPT-4, Claude, or strong open models.
- **Cost:** per-dialog cost similar to UltraChat; scale determines total.

## Modality-specific technical details (REQUIRED — conversation)
- **Turn-count distribution:** typically 3–8 per dialog.
- **Speaker-role protocol:** assistant role conditioned on system prompt; user role free or also persona-conditioned.
- **Persona conditioning:** explicit and compositional (role + style + constraint).
- **Safety post-filter:** system prompts can include explicit safety-mode instructions — a lever for safety-data generation.
- **Diversity metric:** unique n-grams, persona-consistent-follow-up-rate, embedding-cluster count.

## Quality / diversity evaluation
- Models trained with persona-conditioned SFT exhibit measurably better **instruction-following adherence** when given unseen system prompts at inference.
- Open evals (IFEval, AgentBench): +5–10 points over topic-only-diverse SFT baselines when persona-SFT is added.
- Behavior: models become reliably controllable via system prompt, reducing the need for prompt-engineering at inference.

## Risks + gotchas
- **Persona extraction bias:** web-mined personas overrepresent online professionals and English speakers.
- **Compositional overloading:** stacking too many constraints (role + style + mode + safety) confuses teachers and students.
- **System-prompt over-specialization:** model may become overly literal about system prompts, refusing to deviate when appropriate.
- **No canonical single source** — this technique is represented across many recipes rather than one paper.

## Connections
- Primary source of personas: [[persona-hub]].
- Contrast: [[ultrachat-pipeline]] (topic-taxonomy diversity), [[camel]] (role-pair diversity).
- Complementary: [[openassistant]] (real-human diversity as the ceiling).
- Downstream use: Tülu 3 SFT mix, Qwen 2.5 post-training, many 2024+ open models.
