---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/xlam.md
source_url: https://arxiv.org/abs/2409.03215
created_at: "2026-04-23"
---

# Excerpt: xLAM — the model family that consumes the APIGen pipeline

**Source library:** `wiki/raw-data/llm-training/papers/xlam.md`
**Paper:** Zhang, Lan, Zhu, Liu et al. 2024 (xLAM-v1); 2025 (xLAM-2 via APIGen-MT). Salesforce AI Research.

---

## Why this source anchors ch-26

xLAM is the canonical open function-calling model family, and the clearest operationalisation of "the data pipeline is the lever, not the parameter count." xLAM-v1 is trained on APIGen-60K single-turn; xLAM-2 adds APIGen-MT multi-turn plus optional DPO. Scaling runs from 1B to 70B, but the key empirical finding is that the data-mix transition (single-turn → + multi-turn → + DPO) drives more gain than scale alone.

Ch-26 §3 and §4 name xLAM as the downstream consumer of APIGen and APIGen-MT. This excerpt expands the staged training recipe, the data-mix ratio, and the DPO details that make xLAM-2 reproducible.

---

## Data mix — the 40/60 function-calling / general-chat split

From source lines 25–34:

> ### Data mix for xLAM-v1
> - **APIGen-60k** (single-turn function-calling, 3-layer verified). See [[apigen]].
> - General-purpose instruction data (preserve chat quality): OpenOrca, WildChat subsets.
> - Ratio roughly 40% function-calling / 60% general chat.
>
> ### Data mix for xLAM-2
> - APIGen-60k single-turn.
> - **APIGen-MT-5k** public + larger internal multi-turn split. See [[apigen-mt]].
> - Optional DPO step on (preferred, rejected) tool-call pairs synthesized by sampling failure modes.

The **40/60 FC-vs-chat ratio** is the load-bearing design decision. Train with higher FC share (e.g. 80%) and the model becomes a narrow tool-call specialist that degrades on open-ended conversation. Train with lower FC share (e.g. 20%) and the model doesn't consolidate the function-calling capability against general chat interference. The 40% number is the empirical sweet spot.

This matters because it frames what "function-calling specialist" means in practice: not "trained only on function calls," but "trained on a balanced mix where general chat prevents over-specialisation without diluting the FC signal."

---

## Staged training recipe

From source lines 35–41:

> ### Training recipe
> - **SFT:** LR 2e-5 → 5e-6 cosine, 3 epochs, seq len 8K. Prompt masked (loss only on assistant tokens including tool calls).
> - **Optional DPO:** β = 0.1; preference pairs = (correct tool call, common failure mode like hallucinated function name).
> - **Bases:** Mistral-7B, Mixtral-8x7B, Llama-3.1-70B, DeepSeek-Coder-V2-8x22B.
> - **Output shape:** SFT corpus ~100K (single-turn) + tens-of-thousands (multi-turn).
> - **Teacher(s):** data comes from APIGen pipeline (DeepSeek-Coder-V2, GPT-4, Claude-3.5).
> - **Cost / compute:** training not separately disclosed; dominated by the 70B / 8×22B SFT runs.

Two operational details worth lifting.

**Prompt-masked loss.** The loss fires only on assistant tokens, *including* the tool-call JSON. This means the model is not trained to reproduce the system prompt's tool schemas (which change per deployment); it's trained to produce calls conditioned on them. The implication for inference: at serve time you can swap the tool schema without re-training.

**DPO with synthetic rejects.** The β=0.1 setting is standard. The interesting piece is how the rejected samples are constructed: by *sampling failure modes* — hallucinated function names, wrong argument types, missing required parameters. This is programmatic rejection generation, not collected real model outputs. It's cheaper but narrower; it addresses known failure modes but cannot surface unknown ones. For xLAM-2, this was sufficient to lift BFCL-V3 multi-turn by a few points on top of SFT.

---

## Scaling behaviour

From source lines 51–54:

> - **xLAM-7B-fc-r:** BFCL-V1 88.24% — #1 among <13B at release (Sept 2024).
> - **xLAM-8x22B-fc-r:** BFCL-V1 ~89% — near GPT-4 overall.
> - **xLAM-2-70B-fc-r:** τ-bench pass^1 56.2% / pass^4 39.4% — leading open model on multi-turn; BFCL-V3 multi-turn ~72%.
> - Smaller variants surprisingly competitive: xLAM-2-8B beats GPT-4o on τ-bench retail.

The small gap between xLAM-7B (88.24) and xLAM-8x22B (~89) on BFCL-V1 is the key scaling observation: **scale alone adds <1 BFCL-V1 point at this data recipe.** The actual lifts from Sept 2024 → 2025:

- APIGen-60k only → APIGen-60k + APIGen-MT-5k: +4 points on BFCL-V3 multi-turn at 8B scale.
- APIGen-60k + APIGen-MT-5k → + DPO: +1–2 points on relevance-detection.

**Data-recipe transitions, not model scale, produce the visible improvements.** This is the argument for why the chapter is taught at all — a 7B model with the right data recipe (xLAM-7B at 88.24) beats a 70B model with a weaker recipe.

---

## Risks and gotchas

From source lines 57–60:

> - **License:** CC-BY-NC-4.0 (non-commercial). Not drop-in for product use.
> - **Narrow skill:** function-calling specialist — general chat quality below general-purpose models of same size.
> - **Chat-template sensitivity:** xLAM expects its exact tool-call format; prompt-engineering to other schemas degrades performance.
> - **BFCL overfit risk:** community questions whether BFCL-V1 scores (91% ceiling) reflect real production reliability — pushed V2/V3 benchmarks.

The chat-template-sensitivity point is where ch-26 §8's rule "match the call template to the AST matcher" comes from. xLAM's template is OpenAI `tool_calls` JSON; the AST matcher canonicalises to that form; schemas that deviate (Glaive XML, NexusRaven Python-call syntax) lose points at inference translation. Training and evaluation template alignment is a silent design axis.

The narrow-skill point is the cost of the 40/60 FC mix: xLAM preserves chat quality but does not match a general-purpose chat model of the same size. This is why Granite ([[granite-function-calling]]) adds 10% OpenHermes/Dolphin to its mix — the minimum chat-preservation slice that still allows the FC capability to consolidate.

---

## Connections

- Data pipelines: [[apigen]] + [[apigen-mt]].
- Competing 2024/25 family: [[toolace]] (same 7B-ish scale, different data pipeline — broader coverage via TSS).
- Small-model siblings: [[hammer]] (relevance via masking), [[nexusraven]] (nested-call curriculum).
- Enterprise mix: [[granite-function-calling]] (blends APIGen + ToolLLM + Glaive + Nexus + in-house).
- Evaluation target: [[bfcl]] — xLAM-7B's 88.24 BFCL-V1 is the benchmark's single most-cited open-model number.
