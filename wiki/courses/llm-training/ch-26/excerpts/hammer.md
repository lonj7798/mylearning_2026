---
chapter: ch-26
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/hammer.md
source_url: https://arxiv.org/abs/2410.04587
created_at: "2026-04-23"
---

# Excerpt: Hammer — relevance-detection via function-name masking

**Source library:** `wiki/raw-data/llm-training/papers/hammer.md`
**Paper:** Lin, Wen, Peng et al. 2024, "Hammer: Robust Function-Calling for On-Device Language Models via Function Masking" (MadeAgents / Peking University / BAAI).

---

## Why this source anchors ch-26

Hammer is the clearest instance of the chapter's "benchmark-shaped specialist" pattern. BFCL added a Relevance-Detection category in V1 to test whether models refuse to call tools when the user's query is unrelated to the offered tools. Small on-device models (0.5B–7B) failed this category badly because of **naming bias** — they shortcut via lexical overlap between query words and tool names. Hammer's entire contribution is a training-data augmentation that disentangles query semantics from tool-name lexical cues.

Ch-26 §7 names Hammer as the relevance-axis specialist. This excerpt expands the function-name masking protocol, the irrelevance augmentation ratio, and the ablation showing each trick's standalone contribution.

---

## The naming-bias failure mode

From source lines 14–15:

> Hammer is a family of on-device function-calling models (0.5B, 1.5B, 7B) trained with two data-augmentation tricks addressing the *naming-bias* failure mode seen in small tool-use models.

The failure mode in concrete terms: a small model sees a tool named `send_email` and a user query containing "email" (even in a factual question like "what's the origin of the word email?"), and fires the call. The model has learned *name mentioned → call fires*, which is a lexical shortcut — a form of Clever Hans on the tool side.

Large models don't exhibit this failure as strongly because their semantics are richer: they actually read the tool description and the query, not just the tool name. Small models, with less capacity to represent semantic content, lean harder on the lexical cue. Naming bias is therefore a small-model failure, which is why Hammer targets 0.5B–7B specifically.

---

## Trick 1 — function-name masking

From source lines 26–27:

> **Step 1 — Function-name masking:** for ~30% of training samples, replace every tool name in both the system prompt (tool list) and the assistant's tool call with a random placeholder (sampled from `func_[a-z0-9]{6}`). Consistency enforced: same mapping used throughout a single example.

The mechanics:
- Pick 30% of training samples uniformly at random.
- Within a sample, build a consistent name → placeholder map (`send_email → func_a1b2c3`).
- Apply the map to the tool schema in the system prompt *and* to the call in the assistant message.

The consistency requirement is crucial. If the map is inconsistent (different placeholders in prompt and call), the model cannot learn the task at all. If the map is consistent, the model is forced to connect the query to the tool description and call structure, *not* the tool name.

**Why 30%?** From source line 40:

> **Masking ratio analysis:** 30% masking is the authors' found optimum; 50% degrades general tool recall; 10% gives weak debiasing.

The intuition: at 10%, the model still learns to shortcut via name for 90% of samples; the debiasing is too weak. At 50%, too much of training is on random names, and the model loses the ability to recognise real tool names when it sees them (e.g. can't generalise "this tool called `get_weather` is probably the weather tool"). 30% is the point where debiasing is strong enough to fix relevance-detection but real-name recall is preserved.

---

## Trick 2 — irrelevance augmentation

From source lines 28:

> **Step 2 — Irrelevance augmentation:** ~30% of samples are irrelevance examples — user query is about topic X, offered tools are about unrelated topic Y. Gold label = text response refusing / asking clarification, no tool call.

This adds explicit negative examples. In APIGen-style data, every training sample has a tool call as the answer; the model never sees "no tool call" as the correct output. Hammer adds ~30% of samples where the correct answer is a refusal or a clarification request.

The distribution matter (from source line 51):

> **Irrelevance distribution matters:** if irrelevant tools are too different from query topics, the model learns cheap lexical refusal only.

"Ask about weather, offered tools about cryptocurrency" is too easy — the model can refuse on lexical non-overlap. "Ask about the history of email, offered tools for sending email" is the hard case: lexical overlap is high, semantic relevance is low, and only a model that actually reads the tool description and query correctly refuses. The training distribution has to mix both kinds of irrelevance to stress the hard case.

---

## The combined lift

From source lines 43–47:

> **Quality / diversity evaluation**
> - **Hammer-7B:** BFCL-V1 overall 87.9%, relevance 88.6% — matches GPT-4 on relevance.
> - **Hammer-0.5B:** BFCL-V1 ~78%, remarkable for on-device.
> - **Hammer 2.1 (2025):** BFCL-V3 multi-turn strong performance; competitive with xLAM-2 at same size.
> - Ablation: function-name masking alone gives +7 relevance; irrelevance augmentation alone +10; combined +13.

The ablation decomposition is the key line. **+7 (masking) + +10 (irrelevance) = +13 combined**, not +17. The tricks are partially redundant — both push the model away from the name-lexical shortcut — but they remain complementary enough that doing both beats doing either alone.

Hammer-7B at ~88% BFCL-V1 matches xLAM-7B's 88.24% overall, but on the **relevance sub-category** Hammer-7B hits ~88.6% vs xLAM-7B's ~80%. Hammer loses on other BFCL categories (nested, parallel-multiple) where xLAM's APIGen-derived training is stronger; the trade is a specialisation on relevance.

---

## Carryover for the chapter's production recipe

Hammer is one of the sources in ch-26 §8's blended recipe, contributing the masking + irrelevance augmentation applied to ~10% of the total mix. The intuition: if you train xLAM-style on APIGen-60k + APIGen-MT-5k, you inherit APIGen's relevance blind spot. Adding 10% Hammer-style augmentation lifts the relevance score without materially disturbing the other categories.

The operational rule from §8: **gate on relevance-detection ≥80% before release.** A model at 92% BFCL overall but 65% relevance hallucinates tool calls on irrelevant queries ~35% of the time in production — the #1 production failure mode, and the one that erodes user trust fastest.

---

## Hammer 2.0 and 2.1 — the natural extensions

From source lines 29–30:

> **Step 3 — Parameter perturbation (Hammer 2.0):** introduce realistic parameter errors and their corrections in training data — teaches the model to repair malformed calls.
> **Step 4 — Multi-turn augmentation (Hammer 2.1):** adopts APIGen-MT-style multi-turn trajectories.

These aren't core to ch-26's framing but are the natural follow-ups: parameter perturbation addresses a second class of small-model bugs (wrong arg type / wrong arg name), and multi-turn augmentation catches Hammer up with xLAM-2 on BFCL-V3. Hammer 2.1 in 2025 is competitive with xLAM-2-8B, which is the validation that the specialisation recipe ports across BFCL versions without fundamental redesign.

---

## Connections

- Base data lineage: [[apigen]], [[xlam]] — Hammer's seed data is xLAM-FC-60k, with masking/irrelevance applied on top.
- Multi-turn extension: [[apigen-mt]] (Hammer 2.1).
- Evaluation target: [[bfcl]], especially the Relevance-Detection sub-category where Hammer is the benchmark-shaping specialist.
- Competing small-FC families: [[toolace]] (broader coverage via TSS), [[nexusraven]] (nested-call curriculum), [[granite-function-calling]] (multi-source blend).
