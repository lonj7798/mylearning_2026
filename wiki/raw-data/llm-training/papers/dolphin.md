<!-- scope: Eric Hartford Dolphin datasets and models — uncensored Orca reproduction + later multi-source mixes
     deps: [[orca]]
     see-also: [[openhermes]], [[capybara]], [[tulu-3-sft-mix]]
-->

# Dolphin Datasets (Eric Hartford / Cognitive Computations)
- **Core Insight:** An open, commercially-permissive reproduction of Orca's FLAN-augmented synthetic SFT data, filtered to remove alignment/refusal/bias so that downstream fine-tuners control alignment at their own layer — later Dolphin versions expand to multi-source mixes and DPO variants.
- **Guideline:** When you want a base SFT dataset without baked-in refusal patterns, use Dolphin-v1 (1M FLAN+GPT-4 + 3.5M FLAN+GPT-3.5) as the un-aligned backbone; layer your own alignment SFT + preference stage on top for controlled behavior.
- **Author(s):** Eric Hartford (Cognitive Computations) + community
- **Year:** 2023 (v1) → 2024 (Dolphin 2.x, Dolphin-Mixtral, Dolphin-Llama3, Dolphin-Qwen) → 2025 (Dolphin-R1 data)
- **URL:** https://huggingface.co/datasets/cognitivecomputations/dolphin ; https://erichartford.com/dolphin
- **Relevant topics:** Orca reproduction, uncensored SFT, commercial-license datasets, DPO variants

## Dataset line

### Dolphin v1 (flanv2-augmented)
- **~1M samples** of FLAN-v2 augmented with **GPT-4 completions**.
- **~3.5M samples** of FLAN-v2 augmented with **GPT-3.5 completions**.
- Follows Orca's submix and system-prompt distribution.
- Filters removed examples of alignment, refusal, avoidance, and bias ("uncensoring").
- License: Apache-2.0 (unusual for this scale).

### Dolphin 2.x → Dolphin-Mixtral / Dolphin-Llama3 / Dolphin-Qwen (2024)
- Incorporates community datasets: Airoboros, SlimOrca, SynthIA, Magicoder OSS-Instruct, CodeFeedback, Samantha, and more.
- Targets Mixtral / Llama-3 / Qwen backbones with the Dolphin "no-refusal" philosophy.
- Released models and underlying mixes on HF.

### Dolphin-R1 (2025)
- Synthetic reasoning-trace dataset modeled after DeepSeek-R1 distill patterns; ~800K CoT traces from R1-class teachers.
- Incorporates filtering for correctness + format + language.

## Method — "Uncensoring" as filtering
Hartford's recipe:
1. Identify refusal patterns in base synthetic data (regex + classifier).
2. Identify alignment-steering responses ("As an AI language model…", moral-preamble templates).
3. Identify bias markers (formulaic disclaimers).
4. Drop those samples.
5. Train on the remainder with the same recipe as the original Orca.

The claim is that refusal/alignment should live in a separate layer (your RLHF / system prompt) rather than being baked into SFT — giving downstream users control.

## Composition notes (Dolphin v2.x approximate)
| Ingredient | Role |
|---|---|
| SlimOrca | Orca reproduction core |
| Airoboros | Multi-skill synthetic (see [[oss-instruct]] adjacent) |
| Magicoder OSS-Instruct | Code |
| SynthIA | Math/reasoning |
| CodeFeedback | Code with execution feedback |
| Samantha | Character / persona dialogues |
| Dolphin-original FLAN splits | Broad task diversity |

## Practitioner takeaways
- **Commercial license** (Apache-2.0) for the core dataset is rare and valuable.
- **"Uncensoring" ≠ unsafe by default** — it moves alignment to a separate training stage; you still choose to align.
- **Evolving composition** — each Dolphin version tracks the community's best open synthetic mixes.
- Popular base for community / indie model fine-tunes.

## Risks + gotchas
- **Downstream safety responsibility:** users who skip alignment get a raw capability model.
- **Teacher-output license:** GPT-4 / GPT-3.5 outputs were used — Apache-2.0 on the *dataset* doesn't override API TOS in all jurisdictions.
- **Quality variance across versions** — later mixes inherit whatever the included components' flaws are.
- **Eric Hartford's explicit stance on "uncensored"** is controversial; the community is split on whether this is a data-layer vs alignment-layer issue.

## Connections
- Direct heir to [[orca]] / [[orca-2]] philosophy.
- Parallel catalog to [[openhermes]] (Teknium).
- Later Dolphin-R1 dataset aligns with [[deepseek-r1-distill-synth]].
- Frequently combined with [[capybara]] in community mixes.
