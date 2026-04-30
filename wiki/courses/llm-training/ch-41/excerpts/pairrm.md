---
chapter: ch-41
course: llm-training
phase: read
excerpt_of: wiki/raw-data/llm-training/papers/pairrm.md
source_url: https://arxiv.org/abs/2306.02561
created_at: "2026-04-23"
---

# Excerpt: PairRM — §4 of ch-41 uses as scalar-RM alternative

**Source library:** `wiki/raw-data/llm-training/papers/pairrm.md`
**Artifact:** joint-encoder preference model + swap augmentation + tournament Best-of-N

---

## Why this source anchors ch-41

§4 opens with the observation that BT's `r(x, y_A) − r(x, y_B)` is a *structural* choice — the two responses are scored independently and compared via subtraction. PairRM is the alternative: score them *jointly* so cross-attention can compare them directly. At matched performance, PairRM is ~20× smaller than scalar RMs, which is why §6's decision framework routes reranking tasks to PairRM by default.

---

## The architectural shift §4 explains

From the source (line 18):

> **Joint encoding:** `f(x, y_A, y_B) → logit`; cross-attention sees both responses at once.

This is the line that justifies a separate RM category in §6. Scalar BT RMs process A and B on separate forward passes — the subtraction is algebraic, not attentional. PairRM concatenates `[CLS] x [SEP] y_A [SEP] y_B [SEP]` and the transformer's self-attention compares A-tokens to B-tokens inside the same forward pass. For near-tie cases (length-matched, paraphrase-similar), that comparison is the bit of information scalar RMs throw away.

---

## The two tricks that make it work

From the source (line 19):

> **Swap-augmentation:** always evaluate `(y_A, y_B)` and `(y_B, y_A)`, average logits — cancels position bias at train and inference time.

From the source (line 20):

> **Tournament Best-of-N:** for N candidates run O(N log N) pairwise comparisons, advance winners.

Ch-41 §4 keeps both tricks as non-optional. Swap-augmentation adds 2–3 pp and removes position bias — skipping it is the #1 PairRM reproduction failure. Tournament BoN keeps the reranking tractable at N = 64 where full pairwise `O(N²)` would be 2016 forward passes instead of 6 rounds.

---

## The performance claim §4 cites

From the source (line 21):

> PairRM-0.4B matches or beats scalar RMs at 7B on LLM-Blender benchmark.

Ch-41 §4 treats this as the key empirical fact. The interpretation: most of the information in "which response is better" is *relative*, and cross-attention captures relative differences at parameter counts where the scalar-subtraction approach cannot.

---

## The production use case §4 names

From the source (line 22):

> **Use cases beyond reranking:** preference-pair filtering for DPO (keep pairs where PairRM confidently prefers one), synthetic preference label generation, Best-of-N verification on instruction-following tasks.

And (line 34):

> keep `(y_w, y_l)` pairs where `PairRM(y_w, y_l) > τ`; this is a simple quality gate that has been shown to lift DPO performance ~2 pp on held-out evals.

Ch-41 §4 reports this as PairRM's current production footprint: not PPO reward-shaping, but DPO pair filtering. The mental model is "PairRM is a gate, not a gradient." It screens noisy preference pairs before they become DPO training data.

---

## The failure surface §4 still owes

From the source (line 33):

> inherits verbosity and position bias to some extent; swap-augmentation handles position but not verbosity — explicitly length-balance training pairs.

Ch-41 §4 flags this but routes the fix to §5: if verbosity is the bias you cannot kill, use HelpSteer2's separate Verbosity head and reweight at RL time ([[nemotron-4-synthetic]]). PairRM is not the right structural answer to verbosity — it is the right structural answer to *relative comparison*.

---

## The decision §6 derives from this excerpt

§6's row: *"Reranking N candidates inline (DPO pair filtering, inference-time BoN, prompt-wise selection) → PairRM."* The reason is the joint-encoder insight in line 18 plus the size-efficiency claim in line 21. PairRM is wrong for "give me a scalar reward to shape PPO with" — that is §1's BT RM or §4's GenRM job.

---

## Connections to the rest of ch-41

- **§1** — structural alternative to [[bradley-terry-rm]]; does not use BT loss directly, uses binary cross-entropy on the joint logit.
- **§3** — small enough to ensemble cheaply; `std_k PairRM_k` is a cheap OOD flag.
- **§4** — sits alongside GenRM as the two non-scalar RM categories.
- **§6** — default for reranking; not default for PPO reward-shaping.
- **ch-44+ (DPO)** — PairRM's dominant role today is DPO pair filtering, not standalone RM.
